# Task 035: FLOWVPM GPU U/J-solve tuning sweep on H200.
#
# Drives the FLOWVPM (gpu-full) device-resident radix FMM coupling over a
# pre-registered configuration grid on the exact 033 case constructions
# (benchmark_033_common.jl seeds/sigma/FMM settings), recording per-config:
#   - warm U/J-solve wall times (median/min over FM035_REPS, full `UJ_fmm`)
#   - component times: reset / refresh / eval / finalize (+ FLOWVPM overhead)
#   - per-stage CUDA-event medians (b2m/m2m/m2l/l2l/l2b) and hierarchical
#     refresh telemetry (grid/occupancy/direct_gen/route_gen/groups) when
#     profile=1
#   - sampled U (gate) and J (diagnostic) errors vs the checksummed 033
#     sampled-direct references
#   - 023 counter-contract flatness across the timed reps, steady-state
#     CUDA.@allocated, structure telemetry (cells/nodes/routes/direct pairs),
#     construction time, device memory
#   - optional full RK3 step time (rk3=1; relax + U_prev bookkeeping off)
#
# Config file (FM035_CASES): one config per line, whitespace-separated
# key=value tokens. Keys:
#   label   unique row id (required; used for resume-skip)
#   case    cube | wake            n  body count (033 grid point)
#   kernel  regularized | partitioned | twopass    (default regularized)
#   tf      Float64 | Float32      (lifecycle precision; default Float64)
#   expansion_order  FastMultipole code order (default 3); literature P is
#                    expansion_order + 1, so literature P=4/5/6 maps to 3/4/5
#   ell     tree depth (omit for the 034 auto derivation)
#   q       leaf near_radius2 (default 16)
#   sched   comma list level_radii2 (levels 2:ell, coarse->fine; optional)
#   strategy concat | dense | precomputed_y        (default concat)
#   K       window_classes (default 256, the 034 coupling default)
#   rho_t   kernel cutoff override (optional)
#   rho_c   TwoPassVortex primary/direct cutoff override (optional)
#   twopass_aabb 1 = target-point/source-cell pass-2 AABB prune (default 0)
#   profile 1 = per-stage profile (default 0)
#   rk3     1 = also time one full RK3 step (default 0)
# Lines starting with # and blank lines are ignored.
#
# Usage (GPU node):
#   julia --project=$ENV MATRIX_OPERATOR_REFACTOR/scripts/benchmark_035_gpu.jl
# with env: FM035_FMDIR (FastMultipole tree), FM035_CASES, FM035_OUT,
# FM035_REPS (default 15), FM035_WARMUP (default 2).

# FM035_DRYRUN=1: parse + validate the config grid (incl. settings/kernel/
# strategy resolution through FLOWVPM) without CUDA, then exit.
const FM035_DRYRUN = get(ENV, "FM035_DRYRUN", "0") == "1"

if !FM035_DRYRUN
    import CUDA
    CUDA.functional() || error("CUDA is not functional on this node")
end

using Statistics
using Printf

const FM035_FMDIR = abspath(get(ENV, "FM035_FMDIR", get(ENV, "HOME", "") * "/FastMultipole-034"))
const FM035_CASEFILE = abspath(ENV["FM035_CASES"])
const FM035_OUT = abspath(ENV["FM035_OUT"])
const FM035_REPS = parse(Int, get(ENV, "FM035_REPS", "15"))
const FM035_WARMUP = parse(Int, get(ENV, "FM035_WARMUP", "2"))
const FM035_U_GATE = 1e-3

# 033 harness: case builders, reference reader (schema + sha256 validation).
# Defines `vpm = FLOWVPM`, fm033_build, fm033_reference_path, fm033_read_reference.
include(joinpath(FM035_FMDIR, "MATRIX_OPERATOR_REFACTOR", "scripts",
    "benchmark_033_common.jl"))

const FM = vpm.fmm   # FastMultipole
const FM035_REFDIR = joinpath(FM035_FMDIR, "MATRIX_OPERATOR_REFACTOR",
    "data", "flowvpm_baseline", "references")

vpm._FMM_HAS_RADIX || error("installed FastMultipole lacks the radix device interface")

# ---------------------------------------------------------------- parsing ----

function parse_config(line::AbstractString)
    kv = Dict{Symbol,String}()
    for tok in split(strip(line))
        m = match(r"^([A-Za-z_0-9]+)=(.*)$", tok)
        m === nothing && error("bad config token: $tok")
        kv[Symbol(m.captures[1])] = m.captures[2]
    end
    haskey(kv, :label) || error("config line missing label=: $line")
    tfs = get(kv, :tf, "Float64")
    return (
        label = kv[:label],
        case = get(kv, :case, "cube"),
        n = parse(Int, get(kv, :n, "100000")),
        kernel = Symbol(get(kv, :kernel, "regularized")),
        tf = tfs == "Float64" ? Float64 : tfs == "Float32" ? Float32 :
            error("tf must be Float64 or Float32"),
        expansion_order = parse(Int, get(kv, :expansion_order, "3")),
        ell = haskey(kv, :ell) ? parse(Int, kv[:ell]) : nothing,
        q = parse(Int, get(kv, :q, "16")),
        sched = haskey(kv, :sched) ?
            Tuple(parse.(Int, split(kv[:sched], ','))) : nothing,
        strategy = Symbol(get(kv, :strategy, "concat")),
        K = parse(Int, get(kv, :K, "256")),
        rho_t = haskey(kv, :rho_t) ? parse(Float64, kv[:rho_t]) : nothing,
        rho_c = haskey(kv, :rho_c) ? parse(Float64, kv[:rho_c]) : nothing,
        rectangular = get(kv, :rectangular, "0") == "1",   # task 037 stage 5
        twopass_aabb = get(kv, :twopass_aabb, "0") == "1",
        profile = get(kv, :profile, "0") == "1",
        rk3 = get(kv, :rk3, "0") == "1",
    )
end

configs = [parse_config(l) for l in readlines(FM035_CASEFILE)
           if !isempty(strip(l)) && !startswith(strip(l), '#')]
labels = [c.label for c in configs]
length(unique(labels)) == length(labels) || error("duplicate config labels")

if FM035_DRYRUN
    for cfg in configs
        cfg.expansion_order >= 0 || error(
            "expansion_order must be nonnegative; got $(cfg.expansion_order)")
        s = vpm.RadixFMMSettings(; expansion_order=cfg.expansion_order,
            ell=cfg.ell, near_radius2=cfg.q,
            level_radii2=cfg.sched, window_classes=cfg.K, precision=cfg.tf,
            direct_kernel=cfg.kernel, rho_t=cfg.rho_t, rho_c=cfg.rho_c,
            m2l_strategy=cfg.strategy, rectangular=cfg.rectangular)
        k = vpm._radix_direct_kernel(s)
        strat, op = vpm._radix_m2l_strategy(s)
        rho_c_text = k isa FM.TwoPassVortex ? string(k.rho_c) : "n/a"
        println("[dryrun ok] $(cfg.label): $(cfg.case) n=$(cfg.n) " *
            "$(typeof(k)) rho_t=$(k.rho_t) " *
            "rho_c=$rho_c_text " *
            "$(typeof(strat)) tf=$(cfg.tf) " *
            "literature_P=$(cfg.expansion_order + 1) " *
            "expansion_order=$(cfg.expansion_order) " *
            "ell=$(cfg.ell) q=$(cfg.q) sched=$(cfg.sched) K=$(cfg.K) " *
            "rectangular=$(cfg.rectangular) " *
            "twopass_aabb=$(cfg.twopass_aabb) " *
            "profile=$(cfg.profile) rk3=$(cfg.rk3)")
    end
    println("dryrun: $(length(configs)) configs valid")
    exit(0)
end

# ----------------------------------------------------------------- helpers ----

# CuArray-backed copy of a CPU-built 033 field (cuda_034_refcheck.jl pattern);
# maxparticles = n (capacity == live count; the sweep never grows the field).
function fm035_to_gpu(cpu_pfield)
    n = vpm.get_np(cpu_pfield)
    gpu = vpm.ParticleField(n, Float64;
        formulation=vpm.rVPM, kernel=vpm.gaussianerf, viscous=vpm.Inviscid(),
        SFS=vpm.noSFS, transposed=true, integration=vpm.rungekutta3,
        UJ=vpm.UJ_fmm, fmm=fm033_settings(), arraytype=CUDA.CuArray)
    gpu.np = n
    gpu.particles .= CUDA.CuArray{Float64}(Array(cpu_pfield.particles)[:, 1:n])
    return gpu
end

function fm035_sampled_errors(pfield, reference)
    A = Array(pfield.particles)
    u_err2 = u_ref2 = j_err2 = j_ref2 = 0.0
    u_max = 0.0
    for (si, bi) in enumerate(reference.indices)
        uref = reference.U[si]
        e2 = 0.0
        for k in 1:3
            d = Float64(A[vpm.U_INDEX[k], bi]) - uref[k]
            e2 += d * d
            u_ref2 += uref[k]^2
        end
        u_err2 += e2
        u_max = max(u_max, sqrt(e2))
        jref = reference.J[si]
        for k in 1:9
            d = Float64(A[vpm.J_INDEX[k], bi]) - jref[k]
            j_err2 += d * d
            j_ref2 += jref[k]^2
        end
    end
    return (u_rel_rms=sqrt(u_err2 / max(u_ref2, eps())),
            u_max_err=u_max,
            j_rel_rms=sqrt(j_err2 / max(j_ref2, eps())))
end

function _gpu_samples_ms(f!, state, reps)
    f!(state); CUDA.synchronize()
    return [Float64(CUDA.@elapsed f!(state)) * 1e3 for _ in 1:reps]
end
_median_gpu_ms(f!, state, reps) = median(_gpu_samples_ms(f!, state, reps))

_wall_ms(f) = (CUDA.synchronize(); t = @elapsed (f(); CUDA.synchronize()); t * 1e3)

function _direct_body_pair_total(state)
    n_cells = state.counts.n_cells
    n_direct = state.counts.n_direct
    ranges = Array(state.grid.cell_ranges)[:, 1:n_cells]
    sizes = Int64.(ranges[2, :])
    targets = Array(state.direct_targets)[1:n_direct]
    sources = Array(state.direct_sources)[1:n_direct]
    all(1 .<= targets .<= n_cells) && all(1 .<= sources .<= n_cells) ||
        error("direct route cell index outside 1:$n_cells")
    return sum(sizes[t] * sizes[s] for (t, s) in zip(targets, sources))
end

const CSV_COLUMNS = [
    "label", "job", "host", "case", "n", "kernel", "tf", "ell", "q", "sched",
    "strategy", "K", "rho_t", "rho_c", "status", "message",
    "leaf_q", "rectangular", "twopass_aabb", "ell_axes",
    "uj_ms_median", "uj_ms_min", "reset_ms", "refresh_ms", "eval_ms",
    "finalize_ms", "overhead_ms", "rk3_step_ms",
    "b2m_ms", "m2m_ms", "m2l_ms", "l2l_ms", "l2b_ms",
    "grid_ms", "occupancy_ms", "direct_gen_ms", "route_gen_ms", "groups_ms",
    "u_rel_rms", "u_max_err", "j_rel_rms", "gate_pass",
    "n_cells", "n_nodes", "n_routes", "n_direct", "direct_body_pairs",
    "twopass_candidate_pairs", "twopass_shell_pairs", "total_cells",
    "nodes_per_level", "routes_per_level",
    "construct_s", "alloc_bytes_step", "device_mem_gb", "counters_flat",
    "literature_P", "expansion_order",
]

function write_row!(io, row::Dict{String,Any})
    println(io, join((string(get(row, c, "")) for c in CSV_COLUMNS), ','))
    flush(io)
end

completed_labels() = !isfile(FM035_OUT) ? Set{String}() :
    Set(first.(split.(readlines(FM035_OUT)[2:end], ',')))

if isfile(FM035_OUT)
    existing_header = first(readlines(FM035_OUT))
    expected_header = join(CSV_COLUMNS, ',')
    existing_header == expected_header || error(
        "FM035_OUT schema mismatch: refusing to append rows with the current " *
        "schema to $(FM035_OUT)")
end

# ------------------------------------------------------------------- sweep ----

done = completed_labels()
newfile = !isfile(FM035_OUT)
io = open(FM035_OUT, "a")
newfile && (println(io, join(CSV_COLUMNS, ',')); flush(io))

println("=== 035 GPU tuning sweep: $(length(configs)) configs " *
    "($(length(done)) already done), reps=$(FM035_REPS), out=$(FM035_OUT)")
println("node=$(gethostname()) gpu=$(CUDA.name(CUDA.device())) " *
    "julia=$(VERSION) cuda=$(CUDA.runtime_version())")

# CPU field cache keyed by (case, n): construction is deterministic, so one
# build serves every config at that point.
cpu_fields = Dict{Tuple{String,Int},Any}()
references = Dict{Tuple{String,Int},Any}()

for cfg in configs
    cfg.label in done && (println("[skip] $(cfg.label)"); continue)
    row = Dict{String,Any}(
        "label" => cfg.label, "job" => get(ENV, "SLURM_JOB_ID", ""),
        "host" => gethostname(), "case" => cfg.case, "n" => cfg.n,
        "kernel" => cfg.kernel, "tf" => cfg.tf,
        "literature_P" => cfg.expansion_order + 1,
        "expansion_order" => cfg.expansion_order,
        "ell" => cfg.ell === nothing ? "auto" : cfg.ell, "q" => cfg.q,
        "sched" => cfg.sched === nothing ? "uniform" : join(cfg.sched, ' '),
        "strategy" => cfg.strategy, "K" => cfg.K,
        "rho_t" => cfg.rho_t === nothing ? "default" : cfg.rho_t,
        "rho_c" => cfg.rho_c === nothing ? "default" : cfg.rho_c,
        "rectangular" => cfg.rectangular,
        "twopass_aabb" => cfg.twopass_aabb,
        "status" => "failed", "message" => "")
    gpu = nothing
    try
        FM.CUDA_TWOPASS_TARGET_AABB_PRUNE[] = cfg.twopass_aabb
        key = (cfg.case, cfg.n)
        if !haskey(cpu_fields, key)
            t = @elapsed cpu_fields[key] = fm033_build(cfg.case, cfg.n)
            na = vpm.get_np(cpu_fields[key])
            na == cfg.n || error("n_actual=$na != n_target=$(cfg.n)")
            refpath = fm033_reference_path(FM035_REFDIR, cfg.case, cfg.n)
            references[key] = fm033_read_reference(refpath, cfg.case, cfg.n, na)
            println("[build] $(cfg.case) n=$(cfg.n) ($(round(t, digits=1))s), " *
                "reference $(basename(refpath)) ok")
        end
        cpu = cpu_fields[key]
        reference = references[key]

        gpu = fm035_to_gpu(cpu)
        vpm.radix_fmm_settings!(gpu;
            expansion_order=cfg.expansion_order,
            ell=cfg.ell, near_radius2=cfg.q, level_radii2=cfg.sched,
            window_classes=cfg.K, precision=cfg.tf,
            direct_kernel=cfg.kernel, rho_t=cfg.rho_t, rho_c=cfg.rho_c,
            m2l_strategy=cfg.strategy, rectangular=cfg.rectangular)

        construct_s = @elapsed (vpm.UJ_fmm(gpu); CUDA.synchronize())
        row["construct_s"] = round(construct_s, digits=2)
        st = vpm._radix_fmm_couplings[gpu]
        cache = st.cache
        state = cache.state
        row["ell"] = cache.ell
        row["ell_axes"] = join(Tuple(cache.ell_axes), ' ')
        row["total_cells"] = 8^cache.ell
        pol = cache.policy
        row["leaf_q"] = pol isa FM.HierarchicalRigidStencil ?
            (isempty(pol.level_radii2) ? pol.near_radius2 : last(pol.level_radii2)) : -1

        for _ in 1:FM035_WARMUP
            vpm.UJ_fmm(gpu)
        end
        CUDA.synchronize()

        # counter-contract snapshot before the timed reps
        c0 = deepcopy(state.counters)

        uj_ms = [_wall_ms(() -> vpm.UJ_fmm(gpu)) for _ in 1:FM035_REPS]
        row["uj_ms_median"] = round(median(uj_ms), digits=3)
        row["uj_ms_min"] = round(minimum(uj_ms), digits=3)

        c1 = state.counters
        flat = c1.body_uploads == c0.body_uploads == 0 &&
               c1.expansion_host_copies == c0.expansion_host_copies == 0 &&
               c1.route_uploads == c0.route_uploads &&
               c1.operator_uploads == c0.operator_uploads
        row["counters_flat"] = flat
        flat || (row["message"] *= "counter drift; ")

        alloc = CUDA.@allocated vpm.UJ_fmm(gpu)
        row["alloc_bytes_step"] = alloc

        # component times (median of 5): reset / refresh / eval / finalize
        targets = (gpu,)
        switches = FM.DerivativesSwitch(
            FM.to_vector(false, 1), FM.to_vector(true, 1),
            FM.to_vector(true, 1), targets)
        tbuf = FM._radix_cache_target_buffers!(cache, switches)
        reset_ms = median([_wall_ms(() -> vpm._reset_particles(gpu)) for _ in 1:5])
        refresh_ms = median([_wall_ms(() ->
            FM.update_cuda_radix_state!(cache, targets)) for _ in 1:5])
        eval_ms = median([_wall_ms(() ->
            FM.run_cuda_radix_lifecycle!(state)) for _ in 1:5])
        finalize_ms = median([_wall_ms(() ->
            FM.finalize_cuda_radix_output!(state, targets;
                derivatives_switches=switches,
                host_output_staging=cache.device_ctx.host_output,
                target_buffers=tbuf,
                device_target_buffers=cache.device_ctx.device_target_buffers))
            for _ in 1:5])
        row["reset_ms"] = round(reset_ms, digits=3)
        row["refresh_ms"] = round(refresh_ms, digits=3)
        row["eval_ms"] = round(eval_ms, digits=3)
        row["finalize_ms"] = round(finalize_ms, digits=3)
        row["overhead_ms"] = round(median(uj_ms) -
            (reset_ms + refresh_ms + eval_ms + finalize_ms), digits=3)

        # structure telemetry
        counts = state.counts
        row["n_cells"] = counts.n_cells
        row["n_nodes"] = counts.n_nodes
        row["n_routes"] = counts.n_routes
        row["n_direct"] = counts.n_direct
        row["direct_body_pairs"] = _direct_body_pair_total(state)
        if state.options.direct_kernel isa FM.TwoPassVortex
            shell = FM.cuda_twopass_shell_homogeneity(state)
            row["twopass_candidate_pairs"] = shell.candidate_pairs
            row["twopass_shell_pairs"] = shell.shell_pairs
        end
        row["device_mem_gb"] = round(
            (CUDA.total_memory() - CUDA.free_memory()) / 2^30, digits=2)

        if cfg.profile
            # refresh leaves state consistent; stage launchers replay in order
            FM.update_cuda_radix_state!(cache, targets)
            row["b2m_ms"] = round(_median_gpu_ms(FM._launch_cuda_b2m!, state, 9), digits=3)
            row["m2m_ms"] = round(_median_gpu_ms(FM._launch_cuda_resident_m2m!, state, 9), digits=3)
            row["m2l_ms"] = round(_median_gpu_ms(FM._launch_cuda_resident_m2l!, state, 9), digits=3)
            row["l2l_ms"] = round(_median_gpu_ms(FM._launch_cuda_resident_l2l!, state, 9), digits=3)
            row["l2b_ms"] = round(_median_gpu_ms(FM._launch_cuda_resident_l2b!, state, 9), digits=3)
            hctx = state.interaction_list
            if hctx isa FM.DeviceHierarchicalM2LContext
                hctx.profile_stages = true
                vpm.UJ_fmm(gpu); CUDA.synchronize()
                hctx.profile_stages = false
                row["grid_ms"] = round(hctx.update_stage_ns[1] / 1e6, digits=3)
                row["occupancy_ms"] = round(hctx.update_stage_ns[2] / 1e6, digits=3)
                row["direct_gen_ms"] = round(hctx.update_stage_ns[3] / 1e6, digits=3)
                row["route_gen_ms"] = round(hctx.update_stage_ns[4] / 1e6, digits=3)
                row["groups_ms"] = round(hctx.update_stage_ns[5] / 1e6, digits=3)
                ell = cache.ell
                row["nodes_per_level"] = join((hctx.nodes_per_level[L + 1] for L in 0:ell), ' ')
                row["routes_per_level"] = join((hctx.routes_per_level[L + 1] for L in 2:ell), ' ')
            end
        end

        # final accuracy measurement (state after a clean solve)
        vpm.UJ_fmm(gpu); CUDA.synchronize()
        err = fm035_sampled_errors(gpu, reference)
        row["u_rel_rms"] = @sprintf("%.4e", err.u_rel_rms)
        row["u_max_err"] = @sprintf("%.4e", err.u_max_err)
        row["j_rel_rms"] = @sprintf("%.4e", err.j_rel_rms)
        row["gate_pass"] = err.u_rel_rms <= FM035_U_GATE

        if cfg.rk3
            dt = 1e-6
            vpm.nextstep(gpu, dt; relax=false, update_U_prev=false)  # warm
            CUDA.synchronize()
            rk3_ms = median([_wall_ms(() -> vpm.nextstep(gpu, dt;
                relax=false, update_U_prev=false)) for _ in 1:3])
            row["rk3_step_ms"] = round(rk3_ms, digits=3)
        end

        row["status"] = "ok"
        println("[ok] $(cfg.label): uj=$(row["uj_ms_median"]) ms " *
            "(refresh=$(row["refresh_ms"]) eval=$(row["eval_ms"]) " *
            "finalize=$(row["finalize_ms"])) u=$(row["u_rel_rms"]) " *
            "gate=$(row["gate_pass"]) J=$(row["j_rel_rms"])")
    catch err
        msg = sprint(showerror, err)
        row["message"] *= replace(first(msg, 300), r"[,\n]" => ";")
        println("[FAIL] $(cfg.label): $(first(msg, 300))")
    finally
        gpu !== nothing && vpm.clear_radix_fmm_cache!(gpu)
        gpu = nothing
        GC.gc(); CUDA.reclaim()
    end
    write_row!(io, row)
end

close(io)
println("035 sweep complete")
