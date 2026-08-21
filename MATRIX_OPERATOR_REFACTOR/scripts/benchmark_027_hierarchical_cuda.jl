# Task 027: CUDA device-resident hierarchical rigid M2L benchmark.
#
# Compares the flat `ConstantPAnalyticStencil` device path against the genuine
# multilevel `HierarchicalRigidStencil` device path at both near radii
# (`near_radius2 = 12`, the 024b theta=0.5 stencil, and `near_radius2 = 3`, the
# classic FMM stencil), across all four resident M2L strategy selections.
#
# One CSV row per (policy, strategy, precision, P, LH, n, ell, K). Each row carries
# the source manifest/host/GPU/toolchain identification, the per-level occupancy and
# route distribution, per-stage and per-level GPU timings, construction cost,
# persistent/peak device bytes, transfer counters, warmed device allocation, and the
# sampled accuracy vs `direct!` — the inputs task 028 needs to evaluate a
# heterogeneous per-level strategy mix, which this row does NOT select.
#
# Companion per-class occupancy rows (one per (row, level)) go to `<OUT>.classes.csv`.
#
# Env knobs:
#   FM027_N        comma list of body counts          (default "2000,20000,200000")
#   FM027_P        comma list of expansion orders     (default "4")
#   FM027_ELL      comma list of radix depths         (default "3,4,5")
#   FM027_K        comma list of window_classes       (default "4")
#   FM027_POLICY   comma list of flat,hier12,hier3    (default all three)
#   FM027_STRAT    comma list of strategy labels      (default all four)
#   FM027_TF       comma list of Float64,Float32      (default "Float64")
#   FM027_LH       comma list of 0,1                  (default "0")
#   FM027_REPS     timing repetitions (median)        (default "5")
#   FM027_DIRECT_MAX  max n for the direct! accuracy reference (default 20000)
#   FM027_OUT      output CSV path

using FastMultipole
using FastMultipole.StaticArrays
using Statistics
using LinearAlgebra
using Dates
using Printf
using SHA

include(joinpath(@__DIR__, "..", "..", "test", "gravitational.jl"))

FastMultipole.load_cuda_radix_lifecycle!() ||
    error("CUDA radix lifecycle failed to load: $(FastMultipole.cuda_radix_status())")
using CUDA

const FM = FastMultipole

const NS = parse.(Int, split(get(ENV, "FM027_N", "2000,20000,200000"), ','))
const PS = parse.(Int, split(get(ENV, "FM027_P", "4"), ','))
const ELLS = parse.(Int, split(get(ENV, "FM027_ELL", "3,4,5"), ','))
const KS = parse.(Int, split(get(ENV, "FM027_K", "4"), ','))
const POLICIES = split(get(ENV, "FM027_POLICY", "flat,hier12,hier3"), ',')
const STRATS = split(get(ENV, "FM027_STRAT", "concat,factored,precomputed_y,dense"), ',')
const TFS = [t == "Float32" ? Float32 : Float64
             for t in split(get(ENV, "FM027_TF", "Float64"), ',')]
const LHS = [v == "1" for v in split(get(ENV, "FM027_LH", "0"), ',')]
const REPS = parse(Int, get(ENV, "FM027_REPS", "5"))
const DIRECT_MAX = parse(Int, get(ENV, "FM027_DIRECT_MAX", "20000"))
const STAMP = Dates.format(now(), "yyyymmdd-HHMMSS")
const OUT = get(ENV, "FM027_OUT", joinpath(@__DIR__, "..", "data",
    "hierarchical_m2l_cuda", "cuda_$(gethostname())_$(STAMP).csv"))
const CLASS_OUT = OUT * ".classes.csv"

# ---- provenance -------------------------------------------------------------

function _source_manifest()
    srcdir = joinpath(@__DIR__, "..", "..", "src")
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

# The hierarchical accuracy gate requires an epsilon whose analytic classifier
# rejects exactly the requested rigid near set at this (h0, ell).
function _hier_epsilon(P, q, h0, ell, ::Type{TF}, LH) where TF
    q == 3 && return TF(1e12)
    probe = ConstantPStencilConfig(P, one(TF); lamb_helmholtz=LH)
    upper = constant_p_stencil_bound(TF(h0), ell, probe, SVector(2, 2, 2))
    lower = constant_p_stencil_bound(TF(h0), ell, probe, SVector(3, 2, 0))
    return (upper + lower) / 2
end

const BOX_MIN = SVector(-0.1, -0.1, -0.1)
const BOX_SIZE = 1.2
const BOX_H0 = BOX_SIZE / 2

function _policy(name, P, ell, ::Type{TF}, LH, K) where TF
    name == "flat" && return ConstantPAnalyticStencil(
        ConstantPStencilConfig(P, TF(1e-4); lamb_helmholtz=LH))
    q = name == "hier3" ? 3 : 12
    return HierarchicalRigidStencil(
        ConstantPStencilConfig(P, _hier_epsilon(P, q, BOX_H0, ell, TF, LH);
            lamb_helmholtz=LH);
        near_radius2=q, window_classes=K)
end

# ---- per-class occupancy over the hierarchical windows -----------------------

# Regenerate every window once with route generation only, accumulating the
# per-(level, class) route counts the 028 strategy-mix study needs. This mutates
# only the reusable route buffers and the step counts, both of which the next
# lifecycle call refreshes.
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

# ---- one measurement --------------------------------------------------------

function _empty_row(policy, strat, TF, P, LH, N, ell, K, note)
    return (manifest=MANIFEST, job=JOBID, host=HOST, gpu=GPU, julia=JULIA_V,
        cuda=CUDA_V, blas_threads=BLAS_THREADS, seed=27000,
        policy=policy, strategy=strat, precision=string(TF), P=P, lh=LH, n=N,
        ell=ell, window_classes=K, fit=false,
        note=replace(note, ',' => ';', '\n' => ' '),
        n_cells=0, n_nodes=0, nodes_per_level="", routes=0, routes_per_level="",
        n_direct=0, nonempty_classes=0, max_class=0, mean_class=0.0,
        class_p50=0.0, class_p90=0.0,
        construction_ms=NaN, persistent_device_bytes=0, peak_device_bytes=0,
        grid_ms=NaN, occupancy_ms=NaN, direct_gen_ms=NaN, route_gen_ms=NaN,
        groups_ms=NaN,
        b2m_ms=NaN, m2m_ms=NaN, m2l_ms=NaN, m2l_per_level="", l2l_ms=NaN,
        l2b_ms=NaN, full_step_ms=NaN, m2l_device_alloc_bytes=0,
        step_device_alloc_bytes=0, full_step_host_alloc_bytes=0,
        route_uploads=0, operator_uploads=0, body_uploads=0,
        metadata_downloads=0, expansion_host_copies=0,
        err_potential=NaN, err_gradient=NaN)
end

function measure(policy_name, strat_name, ::Type{TF}, P, LH, N, ell, K) where TF
    spec = STRATEGY_SPECS[strat_name]
    sys = generate_gravitational(27000, N)
    opts = CUDARadixLifecycleOptions(; precision=TF, operator=spec.operator,
        m2l_strategy=spec.strategy)
    GC.gc(); CUDA.reclaim()
    used0 = _used_device_bytes()
    t0 = time_ns()
    cache = RadixFMMCache(sys; expansion_order=P, ell, max_n_bodies=N,
        bounds=(BOX_MIN, BOX_SIZE), lamb_helmholtz=LH, device=true, options=opts,
        policy=_policy(policy_name, P, ell, TF, LH, K))
    CUDA.synchronize()
    construction_ms = (time_ns() - t0) / 1e6
    persistent_device_bytes = _used_device_bytes() - used0

    state = cache.state
    hctx = state.interaction_list
    counters = state.counters
    base_route = counters.route_uploads
    base_operator = counters.operator_uploads

    fmm!(sys, cache; scalar_potential=!LH, gradient=true)   # warm
    b2m_ms = _median_gpu_ms(FM._launch_cuda_b2m!, state, REPS)
    m2m_ms = _median_gpu_ms(FM._launch_cuda_resident_m2m!, state, REPS)
    m2l_ms = _median_gpu_ms(FM._launch_cuda_resident_m2l!, state, REPS)
    l2l_ms = _median_gpu_ms(FM._launch_cuda_resident_l2l!, state, REPS)
    l2b_ms = _median_gpu_ms(FM._launch_cuda_resident_l2b!, state, REPS)
    m2l_alloc = CUDA.@allocated FM._launch_cuda_resident_m2l!(state)
    step_alloc = CUDA.@allocated fmm!(sys, cache; scalar_potential=!LH, gradient=true)
    host_alloc = @allocated fmm!(sys, cache; scalar_potential=!LH, gradient=true)
    full_ms = Float64[]
    for _ in 1:REPS
        push!(full_ms, (@elapsed fmm!(sys, cache; scalar_potential=!LH,
            gradient=true)) * 1e3)
    end
    peak_device_bytes = _used_device_bytes() - used0

    counters.route_uploads == base_route ||
        error("route uploads grew during recurring steps")
    counters.operator_uploads == base_operator ||
        error("operator uploads grew during recurring steps")
    counters.expansion_host_copies == 0 || error("expansion host copies observed")

    # per-stage host-visible timings from the hierarchical telemetry
    grid_ms = occupancy_ms = direct_gen_ms = route_gen_ms = groups_ms = NaN
    m2l_per_level = ""
    routes_per_level = ""
    nodes_per_level = ""
    class_counts = Int[]
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
            append!(class_counts, get(per_level, L, Int[]))
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

    err_potential = NaN
    err_gradient = NaN
    if N <= DIRECT_MAX
        ref = generate_gravitational(27000, N)
        FM.direct!(ref; scalar_potential=true, gradient=true)
        LH || (err_potential = maximum(abs.(Float64.(sys.potential[1, :]) .-
            ref.potential[1, :])))
        err_gradient = maximum(abs.(Float64.(sys.potential[5:7, :]) .-
            ref.potential[5:7, :]))
    end

    row = (manifest=MANIFEST, job=JOBID, host=HOST, gpu=GPU, julia=JULIA_V,
        cuda=CUDA_V, blas_threads=BLAS_THREADS, seed=27000,
        policy=policy_name, strategy=strat_name, precision=string(TF), P=P, lh=LH,
        n=N, ell=ell, window_classes=K, fit=true, note="",
        n_cells=state.counts.n_cells, n_nodes=state.counts.n_nodes,
        nodes_per_level=nodes_per_level, routes=state.counts.n_routes,
        routes_per_level=routes_per_level, n_direct=state.counts.n_direct,
        nonempty_classes=length(class_counts),
        max_class=maximum(class_counts; init=0),
        mean_class=isempty(class_counts) ? 0.0 : mean(class_counts),
        class_p50=_q(class_counts, 0.5), class_p90=_q(class_counts, 0.9),
        construction_ms=construction_ms,
        persistent_device_bytes=persistent_device_bytes,
        peak_device_bytes=peak_device_bytes,
        grid_ms=grid_ms, occupancy_ms=occupancy_ms, direct_gen_ms=direct_gen_ms,
        route_gen_ms=route_gen_ms, groups_ms=groups_ms,
        b2m_ms=b2m_ms, m2m_ms=m2m_ms, m2l_ms=m2l_ms, m2l_per_level=m2l_per_level,
        l2l_ms=l2l_ms, l2b_ms=l2b_ms, full_step_ms=median(full_ms),
        m2l_device_alloc_bytes=m2l_alloc, step_device_alloc_bytes=step_alloc,
        full_step_host_alloc_bytes=host_alloc,
        route_uploads=counters.route_uploads,
        operator_uploads=counters.operator_uploads,
        body_uploads=counters.body_uploads,
        metadata_downloads=counters.metadata_downloads,
        expansion_host_copies=counters.expansion_host_copies,
        err_potential=err_potential, err_gradient=err_gradient)
    class_rows = NamedTuple[]
    if hctx isa FM.DeviceHierarchicalM2LContext
        for (L, counts) in sort(collect(_class_occupancy(state)); by=first)
            push!(class_rows, (manifest=MANIFEST, job=JOBID, policy=policy_name,
                strategy=strat_name, precision=string(TF), P=P, lh=LH, n=N, ell=ell,
                window_classes=K, level=L, nonempty_classes=length(counts),
                total_routes=sum(counts; init=0),
                max_class=maximum(counts; init=0),
                mean_class=isempty(counts) ? 0.0 : mean(counts),
                p10=_q(counts, 0.1), p50=_q(counts, 0.5), p90=_q(counts, 0.9)))
        end
        fmm!(sys, cache; scalar_potential=!LH, gradient=true)
    end
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
            row = _empty_row(policy, strat, TF, P, LH, N, ell, K,
                sprint(showerror, err))
            crows = NamedTuple[]
        end
        push!(rows, row)
        append!(class_rows, crows)
        @printf("%-7s %-14s %-8s P=%-2d LH=%-5s n=%-7d ell=%d K=%-4d fit=%-5s m2l %9.3f ms  step %9.3f ms  mem %8.1f MB  routes %d\n",
            row.policy, row.strategy, row.precision, row.P, string(row.lh), row.n,
            row.ell, row.window_classes, string(row.fit), row.m2l_ms,
            row.full_step_ms, row.persistent_device_bytes / 1e6, row.routes)
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
