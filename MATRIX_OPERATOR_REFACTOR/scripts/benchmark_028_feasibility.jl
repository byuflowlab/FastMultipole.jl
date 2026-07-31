# Task 028 Phase A: 1e6-bodies-in-10-ms feasibility measurement.
#
# One CSV row per (policy, strategy, precision, expansion_order, LH, n, ell, K).
# Each row reports the three task-028 timing boundaries:
#   (a) eval_ms          — steady-state resident evaluation only
#                          (run_cuda_radix_lifecycle!, CUDA-event median)
#   (b) verdict_step_ms  — refresh + lifecycle + finalize + device Euler
#                          convection, device-resident bodies, counters flat
#   (c) host_step_ms     — the same fmm! step with a host-resident system
#                          (per-step body H2D + influence D2H included),
#                          measured on a second cache
# plus per-stage CUDA-event medians, hierarchical `profile_stages` telemetry,
# per-level M2L and class-occupancy histograms (companion `.classes.csv`),
# host/device allocation per step, transfer counters (asserted flat on the
# verdict boundary), the five 024b sampled-direct error metrics at step 0, and
# the post-convection accuracy re-check against a fresh on-device Float64
# direct reference.
#
# Env knobs (comma lists where plural):
#   FM028_N        body counts                        (default "1000000")
#   FM028_P        expansion orders (P_literature-1)  (default "3")
#   FM028_ELL      radix depths                       (default "4,5,6")
#   FM028_K        window_classes, passed explicitly  (default "256,1740")
#   FM028_POLICY   flat,hier12,hier3                  (default "hier12,hier3")
#   FM028_STRAT    dense,precomputed_y,concat,factored (default "dense,precomputed_y")
#   FM028_TF       Float64,Float32                    (default "Float64,Float32")
#   FM028_LH       0,1                                (default "0")
#   FM028_REPS     timing repetitions (median)        (default "5")
#   FM028_STEPS    convection steps for the accuracy re-check (default "5")
#   FM028_STALE    extra refresh-skipped steps for the stale-tree accuracy
#                  probe; 0 disables                  (default "0")
#   FM028_DT       Euler dt                           (default "1e-5")
#   FM028_BOUND    boundaries to measure, subset of "abc" (default "abc")
#   FM028_OUT      output CSV path
#   FM028_REFDIR   024b direct-reference directory
#
# Flat policy rows reproduce the 024b configuration exactly (seed 24025, box
# (-0.01, 1.02), stencil epsilon 0.19542385331034917 * 2^(ell-4)) so the
# n=1e6 flat dense ell=4 row is a direct cross-check against the recorded
# 0.4246 s (F64) / 0.3213 s (F32) baselines — those baselines used a
# host-resident system, i.e. compare against this script's host_step_ms.

using FastMultipole
using FastMultipole.StaticArrays
using Statistics
using LinearAlgebra
using Dates
using Printf
using SHA

const REPO = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(REPO, "test", "gravitational.jl"))
include(joinpath(@__DIR__, "benchmark_024b_common.jl"))

FastMultipole.load_cuda_radix_lifecycle!() ||
    error("CUDA radix lifecycle failed to load: $(FastMultipole.cuda_radix_status())")
using CUDA

include(joinpath(@__DIR__, "fm028_device_system.jl"))

const FM = FastMultipole

const NS = parse.(Int, split(get(ENV, "FM028_N", "1000000"), ','))
const PS = parse.(Int, split(get(ENV, "FM028_P", "3"), ','))
const ELLS = parse.(Int, split(get(ENV, "FM028_ELL", "4,5,6"), ','))
const KS = parse.(Int, split(get(ENV, "FM028_K", "256,1740"), ','))
const POLICIES = split(get(ENV, "FM028_POLICY", "hier12,hier3"), ',')
const STRATS = split(get(ENV, "FM028_STRAT", "dense,precomputed_y"), ',')
const TFS = [t == "Float32" ? Float32 : Float64
             for t in split(get(ENV, "FM028_TF", "Float64,Float32"), ',')]
const LHS = [v == "1" for v in split(get(ENV, "FM028_LH", "0"), ',')]
const REPS = parse(Int, get(ENV, "FM028_REPS", "5"))
const STEPS = parse(Int, get(ENV, "FM028_STEPS", "5"))
const STALE = parse(Int, get(ENV, "FM028_STALE", "0"))
const DT = parse(Float64, get(ENV, "FM028_DT", "1e-5"))
const BOUND = get(ENV, "FM028_BOUND", "abc")
const STAMP = Dates.format(now(), "yyyymmdd-HHMMSS")
const OUT = get(ENV, "FM028_OUT", joinpath(@__DIR__, "..", "data",
    "feasibility_1m_10ms", "cuda_$(gethostname())_$(STAMP).csv"))
const CLASS_OUT = OUT * ".classes.csv"
const REFDIR = get(ENV, "FM028_REFDIR", joinpath(@__DIR__, "..", "data",
    "cpu_gpu_scaling", "references"))

# 024b body/box/error conventions (seed 24025, sampler seed 24026)
const SEED = 24025
const SAMPLER_SEED = 24026
const BOX_MIN = SVector(-0.01, -0.01, -0.01)
const BOX_SIZE = 1.02
const CLAMP_LO = 0.0    # bodies start in [0,1]^3; clamp keeps them there
const CLAMP_HI = 1.0
_flat_epsilon(ell) = 0.19542385331034917 * 2.0^(ell - 4)

# ---- provenance -------------------------------------------------------------

function _source_manifest()
    srcdir = joinpath(REPO, "src")
    files = sort(filter(f -> endswith(f, ".jl"), readdir(srcdir)))
    ctx = SHA.SHA256_CTX()
    for f in files
        SHA.update!(ctx, codeunits(f))
        SHA.update!(ctx, read(joinpath(srcdir, f)))
    end
    return bytes2hex(SHA.digest!(ctx))[1:16]
end

const MANIFEST = _source_manifest()
const JOBID = get(ENV, "SLURM_JOB_ID", "")
const JULIA_V = string(VERSION)
const CUDA_V = string(CUDA.runtime_version())
const GPU = CUDA.name(CUDA.device())
const HOST = gethostname()
const BLAS_THREADS = BLAS.get_num_threads()

_used_device_bytes() = CUDA.total_memory() - CUDA.free_memory()

function _median_gpu_ms(f!, state, reps)
    f!(state); CUDA.synchronize()
    ts = Float64[]
    for _ in 1:reps
        push!(ts, Float64(CUDA.@elapsed f!(state)) * 1e3)
    end
    return median(ts)
end

function _median_wall_ms(f!, reps)
    f!(); CUDA.synchronize()
    ts = Float64[]
    for _ in 1:reps
        t = @elapsed (f!(); CUDA.synchronize())
        push!(ts, t * 1e3)
    end
    return median(ts)
end

# ---- configuration ----------------------------------------------------------

const STRATEGY_SPECS = Dict(
    "concat" => (operator=MaterializedYRotationM2L(),
        strategy=FM.ConcatenatedFixedZM2L()),
    "factored" => (operator=FactoredRotationM2L(),
        strategy=FM.ConcatenatedFixedZM2L()),
    "precomputed_y" => (operator=FactoredRotationM2L(),
        strategy=FM.PrecomputedFactoredYM2L()),
    "dense" => (operator=MaterializedYRotationM2L(),
        strategy=DenseTranslationM2L()),
)

# policy-specific cache kwargs; the K-default trap means window_classes is
# ALWAYS passed explicitly for hierarchical rows
function _cache_kwargs(policy, ell, K)
    policy == "flat" && return (; stencil_epsilon=_flat_epsilon(ell))
    q = policy == "hier3" ? 3 : 12
    return (; near_radius2=q, window_classes=K)
end

# ---- per-class occupancy over the hierarchical windows (from 027) -----------

function _class_occupancy(state)
    hctx = state.interaction_list
    hctx isa FM.DeviceHierarchicalM2LContext || return Dict{Int,Vector{Int}}()
    noffsets = hctx.noffsets
    K = hctx.window_classes
    per_level = Dict{Int,Vector{Int}}()
    for level in 2:hctx.ell
        counts = Int[]
        for first in 1:K:noffsets
            last = min(first + K - 1, noffsets)
            FM.cuda_hierarchical_route_window!(state, level, first, last)
            kn = last - first + 1
            prev = 0
            for i in 1:kn
                c = Int(hctx.host_window_cum[i]) - prev
                prev = Int(hctx.host_window_cum[i])
                c > 0 && push!(counts, c)
            end
        end
        per_level[level] = counts
    end
    return per_level
end

_q(v, p) = isempty(v) ? 0.0 : quantile(sort(Float64.(v)), p)

# ---- row schema -------------------------------------------------------------

# Every row is `merge(ROW_DEFAULTS, overrides)` so success and failure rows can
# never diverge in schema or column order.
const ROW_DEFAULTS = (
    manifest=MANIFEST, job=JOBID, host=HOST, gpu=GPU, julia=JULIA_V,
    cuda=CUDA_V, blas_threads=BLAS_THREADS, seed=SEED,
    policy="", strategy="", precision="", expansion_order=0, p_literature=0,
    lh=false, n=0, ell=0, window_classes=0, fit=false, note="",
    n_cells=0, n_nodes=0, nodes_per_level="", routes=0, routes_per_level="",
    n_direct=0, nonempty_classes=0, max_class=0, mean_class=0.0,
    class_p50=0.0, class_p90=0.0,
    construction_ms=NaN, persistent_device_bytes=0, peak_device_bytes=0,
    grid_ms=NaN, occupancy_ms=NaN, direct_gen_ms=NaN, route_gen_ms=NaN,
    groups_ms=NaN,
    b2m_ms=NaN, m2m_ms=NaN, m2l_ms=NaN, m2l_per_level="", l2l_ms=NaN,
    l2b_ms=NaN,
    eval_ms=NaN, refresh_ms=NaN, finalize_ms=NaN, euler_ms=NaN,
    verdict_step_ms=NaN, stale_step_ms=NaN,
    host_construction_ms=NaN, host_step_ms=NaN, h2d_ms=NaN, d2h_ms=NaN,
    m2l_device_alloc_bytes=0, step_device_alloc_bytes=0,
    verdict_step_host_alloc_bytes=0,
    route_uploads=0, operator_uploads=0, body_uploads=0,
    metadata_downloads=0, expansion_host_copies=0,
    reference_source="", reference_samples=0, reference_checksum="",
    ref_cross_check_grad_rel=NaN,
    err_potential_abs_rms=NaN, err_potential_rel_rms=NaN,
    err_gradient_abs_rms=NaN, err_gradient_rel_rms=NaN, err_gradient_max=NaN,
    conv_steps=0, conv_dt=DT,
    conv_err_potential_rel_rms=NaN, conv_err_gradient_rel_rms=NaN,
    conv_err_gradient_max=NaN,
    stale_steps=STALE, stale_err_gradient_rel_rms=NaN,
    host_err_gradient_rel_rms=NaN,
)

_note(err) = first(replace(sprint(showerror, err), ',' => ';', '\n' => ' '), 500)

# ---- one measurement --------------------------------------------------------

function measure(policy_name, strat_name, ::Type{TF}, P, LH, N, ell, K) where TF
    spec = STRATEGY_SPECS[strat_name]
    bodies = fm028_body_matrix(SEED, N)
    indices = fm024b_expected_reference_indices(N, SAMPLER_SEED)

    # step-0 reference: the checksummed 024b CSV when present, else an
    # on-device Float64 direct sum at the original (Float64) positions
    ref_path = fm024b_reference_path(REFDIR, N)
    ref_potential = Float64[]
    ref_gradient = zeros(3, 0)
    reference_source = ""
    reference_checksum = ""
    ref_cross = NaN
    d_pos64 = CUDA.CuArray(bodies[1:3, :])
    d_str64 = CUDA.CuArray(bodies[5, :])
    dev_ref = fm028_direct_sample_reference(d_pos64, d_str64, indices)
    if isfile(ref_path)
        r = fm024b_read_reference(ref_path, N;
            expected_seed=SEED, expected_sampler_seed=SAMPLER_SEED)
        ref_potential = r.potential
        ref_gradient = reshape(reduce(vcat, [collect(g) for g in r.gradient]), 3, :)
        reference_source = "024b_csv"
        reference_checksum = r.checksum
        # methodology guard: the on-device kernel must agree with the CSV
        m = fm028_accuracy_metrics(dev_ref[1, :], dev_ref[2:4, :],
            ref_potential, ref_gradient)
        ref_cross = m.gradient_rel_rms
    else
        ref_potential = dev_ref[1, :]
        ref_gradient = dev_ref[2:4, :]
        reference_source = "device_direct"
    end
    CUDA.unsafe_free!(d_pos64); CUDA.unsafe_free!(d_str64)

    sys = FM028DeviceSystem{TF}(bodies)
    opts = CUDARadixLifecycleOptions(; precision=TF, operator=spec.operator,
        m2l_strategy=spec.strategy)
    GC.gc(); CUDA.reclaim()
    used0 = _used_device_bytes()
    t0 = time_ns()
    cache = RadixFMMCache(sys; expansion_order=P, ell, max_n_bodies=N,
        bounds=(BOX_MIN, BOX_SIZE), lamb_helmholtz=LH, device=true,
        options=opts, _cache_kwargs(policy_name, ell, K)...)
    CUDA.synchronize()
    construction_ms = (time_ns() - t0) / 1e6
    persistent_device_bytes = _used_device_bytes() - used0

    state = cache.state
    hctx = state.interaction_list
    counters = state.counters
    step!() = (fmm!(sys, cache; scalar_potential=!LH, gradient=true);
        fm028_euler!(sys, DT, CLAMP_LO, CLAMP_HI))

    fmm!(sys, cache; scalar_potential=!LH, gradient=true)   # warm + step-0 values

    # step-0 sampled accuracy (positions have not moved yet)
    pot0, grad0 = fm028_sampled_output(sys, indices)
    m0 = fm028_accuracy_metrics(pot0, grad0, ref_potential, ref_gradient)

    base_route = counters.route_uploads
    base_operator = counters.operator_uploads

    # per-stage CUDA-event medians
    b2m_ms = _median_gpu_ms(FM._launch_cuda_b2m!, state, REPS)
    m2m_ms = _median_gpu_ms(FM._launch_cuda_resident_m2m!, state, REPS)
    m2l_ms = _median_gpu_ms(FM._launch_cuda_resident_m2l!, state, REPS)
    l2l_ms = _median_gpu_ms(FM._launch_cuda_resident_l2l!, state, REPS)
    l2b_ms = _median_gpu_ms(FM._launch_cuda_resident_l2b!, state, REPS)
    m2l_alloc = CUDA.@allocated FM._launch_cuda_resident_m2l!(state)

    switches = (FM.DerivativesSwitch(!LH, true, false, sys),)

    # boundary (a): steady-state evaluation only
    eval_ms = occursin('a', BOUND) ?
        _median_gpu_ms(FM.run_cuda_radix_lifecycle!, state, REPS) : NaN

    # verdict-step components
    refresh_ms = _median_wall_ms(() -> FM.update_cuda_radix_state!(cache, (sys,)), REPS)
    FM.run_cuda_radix_lifecycle!(state)
    finalize_ms = _median_wall_ms(() -> FM.finalize_cuda_radix_output!(state, (sys,);
        derivatives_switches=switches), REPS)
    euler_ms = _median_wall_ms(() -> fm028_euler!(sys, 0.0, CLAMP_LO, CLAMP_HI), REPS)

    # boundary (b): full verdict step (refresh + lifecycle + finalize + Euler)
    verdict_step_ms = NaN
    step_alloc = 0
    host_alloc = 0
    if occursin('b', BOUND)
        step_alloc = CUDA.@allocated step!()
        host_alloc = @allocated step!()
        verdict_step_ms = _median_wall_ms(step!, REPS)
    end

    # stale-tree policy: the only cheaper refresh policy that exists is
    # skipping the refresh entirely (lifecycle + finalize + Euler on the
    # stale sorted bodies/tree)
    stale_step!() = (FM.run_cuda_radix_lifecycle!(state);
        FM.finalize_cuda_radix_output!(state, (sys,); derivatives_switches=switches);
        fm028_euler!(sys, DT, CLAMP_LO, CLAMP_HI))
    stale_step_ms = _median_wall_ms(stale_step!, REPS)
    FM.update_cuda_radix_state!(cache, (sys,))   # resync after stale probes

    # counter contract on the verdict boundary: everything flat per step
    counters.route_uploads == base_route ||
        error("route uploads grew during recurring steps")
    counters.operator_uploads == base_operator ||
        error("operator uploads grew during recurring steps")
    counters.expansion_host_copies == 0 || error("expansion host copies observed")
    base_body = counters.body_uploads
    base_infl = counters.influence_downloads
    base_meta = counters.metadata_downloads
    step!()
    counters.body_uploads == base_body ||
        error("per-step body upload observed on the device-resident boundary")
    counters.influence_downloads == base_infl ||
        error("per-step influence download observed on the device-resident boundary")
    counters.metadata_downloads == base_meta ||
        error("per-step metadata download observed on the device-resident boundary")

    # hierarchical host telemetry + per-level M2L + class occupancy
    grid_ms = occupancy_ms = direct_gen_ms = route_gen_ms = groups_ms = NaN
    m2l_per_level = ""
    routes_per_level = ""
    nodes_per_level = ""
    class_counts = Int[]
    class_rows = NamedTuple[]
    if hctx isa FM.DeviceHierarchicalM2LContext
        hctx.profile_stages = true
        fmm!(sys, cache; scalar_potential=!LH, gradient=true)
        hctx.profile_stages = false
        grid_ms = hctx.update_stage_ns[1] / 1e6
        occupancy_ms = hctx.update_stage_ns[2] / 1e6
        direct_gen_ms = hctx.update_stage_ns[3] / 1e6
        route_gen_ms = hctx.update_stage_ns[4] / 1e6
        groups_ms = hctx.update_stage_ns[5] / 1e6
        m2l_per_level = join((hctx.m2l_level_ns[L + 1] / 1e6 for L in 2:ell), ' ')
        routes_per_level = join((hctx.routes_per_level[L + 1] for L in 2:ell), ' ')
        nodes_per_level = join((hctx.nodes_per_level[L + 1] for L in 0:ell), ' ')
        per_level = _class_occupancy(state)
        for L in 2:ell
            counts = get(per_level, L, Int[])
            append!(class_counts, counts)
            push!(class_rows, (manifest=MANIFEST, job=JOBID, policy=policy_name,
                strategy=strat_name, precision=string(TF), expansion_order=P,
                lh=LH, n=N, ell=ell, window_classes=K, level=L,
                nonempty_classes=length(counts),
                total_routes=sum(counts; init=0),
                max_class=maximum(counts; init=0),
                mean_class=isempty(counts) ? 0.0 : mean(counts),
                p10=_q(counts, 0.1), p50=_q(counts, 0.5), p90=_q(counts, 0.9)))
        end
        fmm!(sys, cache; scalar_potential=!LH, gradient=true)  # restore state
    else
        plan = state.scratch.m2l_concat
        if plan isa FM.ResidentM2LPrecomputedYPlan
            class_counts = [c for c in plan.offset_counts if c > 0]
        elseif plan isa Union{FM.ResidentM2LDenseCUDAPlan,FM.ResidentM2LFactoredPlan}
            class_counts = [Int(c) for c in plan.host_class_counts if c > 0]
        end
        nodes_per_level = join((cache.level_offsets[L + 2] - cache.level_offsets[L + 1]
                                for L in 0:ell), ' ')
        routes_per_level = string(state.counts.n_routes)
    end

    # convection sanity: STEPS verdict steps, then re-evaluate and compare
    # against a fresh on-device Float64 direct reference at the moved positions
    conv_m = (potential_rel_rms=NaN, gradient_rel_rms=NaN, gradient_max=NaN)
    if STEPS > 0
        for _ in 1:STEPS
            step!()
        end
        fmm!(sys, cache; scalar_potential=!LH, gradient=true)
        refc = fm028_direct_sample_reference(sys.positions, sys.strengths, indices)
        potc, gradc = fm028_sampled_output(sys, indices)
        conv_m = fm028_accuracy_metrics(potc, gradc, refc[1, :], refc[2:4, :])
    end

    # stale-tree accuracy probe: STALE refresh-skipped steps, then refresh and
    # compare against a fresh reference at the final positions
    stale_err = NaN
    if STALE > 0
        for _ in 1:STALE
            stale_step!()
        end
        fmm!(sys, cache; scalar_potential=!LH, gradient=true)
        refs = fm028_direct_sample_reference(sys.positions, sys.strengths, indices)
        pots, grads = fm028_sampled_output(sys, indices)
        ms = fm028_accuracy_metrics(pots, grads, refs[1, :], refs[2:4, :])
        stale_err = ms.gradient_rel_rms
    end

    peak_device_bytes = _used_device_bytes() - used0

    # boundary (c): host-resident variant on its own cache (the construction
    # branches on residency, so the device cache cannot serve host systems)
    host_construction_ms = NaN
    host_step_ms = NaN
    h2d_ms = NaN
    d2h_ms = NaN
    host_err = NaN
    if occursin('c', BOUND)
        gsys = generate_gravitational(SEED, N)
        t0h = time_ns()
        host_cache = RadixFMMCache(gsys; expansion_order=P, ell, max_n_bodies=N,
            bounds=(BOX_MIN, BOX_SIZE), lamb_helmholtz=LH, device=true,
            options=opts, _cache_kwargs(policy_name, ell, K)...)
        CUDA.synchronize()
        host_construction_ms = (time_ns() - t0h) / 1e6
        fmm!(gsys, host_cache; scalar_potential=!LH, gradient=true)  # warm; values valid
        if !LH
            hm = fm024b_accuracy_metrics(gsys,
                (; indices, potential=ref_potential,
                    gradient=[Tuple(ref_gradient[:, k]) for k in 1:length(indices)],
                    samples=length(indices)))
            host_err = hm.gradient_rel_rms
        end
        host_step_ms = _median_wall_ms(
            () -> fmm!(gsys, host_cache; scalar_potential=!LH, gradient=true), REPS)
        # raw transfer costs at this n: positions/strengths up, influence down
        h_up = Matrix{TF}(undef, 5, N)
        d_up = CUDA.CuArray(h_up)
        h2d_ms = median([Float64(CUDA.@elapsed copyto!(d_up, h_up)) * 1e3
                         for _ in 1:REPS])
        h_down = Matrix{TF}(undef, 4, N)
        d_down = CUDA.CuArray(h_down)
        d2h_ms = median([Float64(CUDA.@elapsed copyto!(h_down, d_down)) * 1e3
                         for _ in 1:REPS])
        host_cache = nothing
        gsys = nothing
        GC.gc(); CUDA.reclaim()
    end

    row = merge(ROW_DEFAULTS, (;
        policy=policy_name, strategy=strat_name, precision=string(TF),
        expansion_order=P, p_literature=P + 1, lh=LH, n=N, ell=ell,
        window_classes=K, fit=true,
        n_cells=state.counts.n_cells, n_nodes=state.counts.n_nodes,
        nodes_per_level, routes=state.counts.n_routes, routes_per_level,
        n_direct=state.counts.n_direct,
        nonempty_classes=length(class_counts),
        max_class=maximum(class_counts; init=0),
        mean_class=isempty(class_counts) ? 0.0 : mean(class_counts),
        class_p50=_q(class_counts, 0.5), class_p90=_q(class_counts, 0.9),
        construction_ms, persistent_device_bytes, peak_device_bytes,
        grid_ms, occupancy_ms, direct_gen_ms, route_gen_ms, groups_ms,
        b2m_ms, m2m_ms, m2l_ms, m2l_per_level, l2l_ms, l2b_ms,
        eval_ms, refresh_ms, finalize_ms, euler_ms, verdict_step_ms,
        stale_step_ms,
        host_construction_ms, host_step_ms, h2d_ms, d2h_ms,
        m2l_device_alloc_bytes=m2l_alloc, step_device_alloc_bytes=step_alloc,
        verdict_step_host_alloc_bytes=host_alloc,
        route_uploads=counters.route_uploads,
        operator_uploads=counters.operator_uploads,
        body_uploads=counters.body_uploads,
        metadata_downloads=counters.metadata_downloads,
        expansion_host_copies=counters.expansion_host_copies,
        reference_source, reference_samples=length(indices),
        reference_checksum, ref_cross_check_grad_rel=ref_cross,
        err_potential_abs_rms=LH ? NaN : m0.potential_abs_rms,
        err_potential_rel_rms=LH ? NaN : m0.potential_rel_rms,
        err_gradient_abs_rms=m0.gradient_abs_rms,
        err_gradient_rel_rms=m0.gradient_rel_rms,
        err_gradient_max=m0.gradient_max,
        conv_steps=STEPS,
        conv_err_potential_rel_rms=LH ? NaN : conv_m.potential_rel_rms,
        conv_err_gradient_rel_rms=conv_m.gradient_rel_rms,
        conv_err_gradient_max=conv_m.gradient_max,
        stale_err_gradient_rel_rms=stale_err,
        host_err_gradient_rel_rms=host_err,
    ))
    return row, class_rows
end

# ---- sweep ------------------------------------------------------------------

rows = NamedTuple[]
class_rows = NamedTuple[]
for N in NS, ell in ELLS, P in PS, TF in TFS, LH in LHS, policy in POLICIES,
        strat in STRATS
    ks = policy == "flat" ? KS[1:1] : KS
    for K in ks
        local row, crows
        try
            row, crows = measure(policy, strat, TF, P, LH, N, ell, K)
        catch err
            row = merge(ROW_DEFAULTS, (; policy, strategy=strat,
                precision=string(TF), expansion_order=P, p_literature=P + 1,
                lh=LH, n=N, ell, window_classes=K, note=_note(err)))
            crows = NamedTuple[]
        end
        push!(rows, row)
        append!(class_rows, crows)
        @printf("%-7s %-14s %-8s eo=%-2d LH=%-5s n=%-8d ell=%d K=%-5d fit=%-5s eval %9.3f  verdict %9.3f  host %9.3f ms  grad_err %.3e\n",
            row.policy, row.strategy, row.precision, row.expansion_order,
            string(row.lh), row.n, row.ell, row.window_classes, string(row.fit),
            row.eval_ms, row.verdict_step_ms, row.host_step_ms,
            row.err_gradient_rel_rms)
        row.fit || println("    note: ", row.note)
        flush(stdout)
        GC.gc(); CUDA.reclaim()
    end
end

mkpath(dirname(OUT))
open(OUT, "w") do io
    println(io, join(string.(keys(rows[1])), ','))
    for row in rows
        println(io, join(string.(values(row)), ','))
    end
end
println("wrote ", OUT)
if !isempty(class_rows)
    open(CLASS_OUT, "w") do io
        println(io, join(string.(keys(class_rows[1])), ','))
        for row in class_rows
            println(io, join(string.(values(row)), ','))
        end
    end
    println("wrote ", CLASS_OUT)
end
