# Task 023b: CUDA device-resident factored M2L benchmark.
#
# Compares the production concat device path (MaterializedYRotationM2L +
# ConcatenatedFixedZM2L) against the factored device path (FactoredRotationM2L)
# on the recurring RadixFMMCache(device=true) lifecycle: construction cost,
# persistent device memory, per-stage GPU timings, per-step wall time, device
# allocation on the M2L stage, and accuracy vs direct!. One CSV row per
# (variant, P, LH, N).
#
# Env knobs:
#   FM023B_N      comma list of body counts        (default "150,2000,20000")
#   FM023B_P      comma list of expansion orders   (default "4,8,12")
#   FM023B_ELL    radix depth                      (default "3")
#   FM023B_REPS   timing repetitions (median)      (default "5")
#   FM023B_OUT    output CSV path
#   FM023B_DIRECT_MAX  max N for the direct! accuracy reference (default 20000)

using FastMultipole
using FastMultipole.StaticArrays
using Statistics
using Dates
using Printf

include(joinpath(@__DIR__, "..", "..", "test", "gravitational.jl"))

FastMultipole.load_cuda_radix_lifecycle!() ||
    error("CUDA radix lifecycle failed to load: $(FastMultipole.cuda_radix_status())")
using CUDA

const NS = parse.(Int, split(get(ENV, "FM023B_N", "150,2000,20000"), ','))
const PS = parse.(Int, split(get(ENV, "FM023B_P", "4,8,12"), ','))
const ELL = parse(Int, get(ENV, "FM023B_ELL", "3"))
const REPS = parse(Int, get(ENV, "FM023B_REPS", "5"))
const DIRECT_MAX = parse(Int, get(ENV, "FM023B_DIRECT_MAX", "20000"))
const OUT = get(ENV, "FM023B_OUT", joinpath(@__DIR__, "..", "data",
    "factored_resident_m2l_cuda",
    "cuda_$(gethostname())_$(Dates.format(now(), "yyyymmdd-HHMMSS")).csv"))

_used_device_bytes() = CUDA.total_memory() - CUDA.free_memory()

function _median_gpu_ms(f!, state, reps)
    f!(state); CUDA.synchronize()                      # warm/compile
    ts = Float64[]
    for _ in 1:reps
        push!(ts, Float64(CUDA.@elapsed f!(state)) * 1e3)
    end
    return median(ts)
end

haskey(ENV, "FM023B_CHUNK") &&
    (FastMultipole.FACTORED_CUDA_CHUNK[] = parse(Int, ENV["FM023B_CHUNK"]))

function measure_variant(P, LH, N, operator, label; whole_pass::Bool=true)
    saved_wp = FastMultipole.FACTORED_CUDA_WHOLE_PASS[]
    FastMultipole.FACTORED_CUDA_WHOLE_PASS[] = whole_pass
    try
        return _measure_variant(P, LH, N, operator, label)
    finally
        FastMultipole.FACTORED_CUDA_WHOLE_PASS[] = saved_wp
    end
end

function _measure_variant(P, LH, N, operator, label)
    sys = generate_gravitational(23000 + P + LH, N)
    opts = CUDARadixLifecycleOptions(; operator,
        m2l_strategy=FastMultipole.ConcatenatedFixedZM2L())
    GC.gc(); CUDA.reclaim()
    used0 = _used_device_bytes()
    t0 = time_ns()
    cache = RadixFMMCache(sys; expansion_order=P, ell=ELL, max_n_bodies=N,
        lamb_helmholtz=LH, device=true, options=opts)
    CUDA.synchronize()
    construct_ms = (time_ns() - t0) / 1e6
    persistent_device_bytes = _used_device_bytes() - used0
    counters = cache.state.counters
    base_route_uploads = counters.route_uploads
    base_operator_uploads = counters.operator_uploads

    fmm!(sys, cache; scalar_potential=!LH, gradient=true)   # warm full step
    state = cache.state
    b2m_ms = _median_gpu_ms(FastMultipole._launch_cuda_b2m!, state, REPS)
    m2m_ms = _median_gpu_ms(FastMultipole._launch_cuda_resident_m2m!, state, REPS)
    m2l_ms = _median_gpu_ms(FastMultipole._launch_cuda_resident_m2l!, state, REPS)
    l2l_ms = _median_gpu_ms(FastMultipole._launch_cuda_resident_l2l!, state, REPS)
    l2b_ms = _median_gpu_ms(FastMultipole._launch_cuda_resident_l2b!, state, REPS)
    m2l_device_alloc = CUDA.@allocated FastMultipole._launch_cuda_resident_m2l!(state)
    full_ms = Float64[]
    full_host_alloc = @allocated fmm!(sys, cache; scalar_potential=!LH, gradient=true)
    for _ in 1:REPS
        push!(full_ms, (@elapsed fmm!(sys, cache; scalar_potential=!LH, gradient=true)) * 1e3)
    end
    peak_device_bytes = _used_device_bytes() - used0

    counters.route_uploads == base_route_uploads ||
        error("route uploads grew during recurring steps")
    counters.operator_uploads == base_operator_uploads ||
        error("operator uploads grew during recurring steps")
    counters.expansion_host_copies == 0 || error("expansion host copies observed")

    err_potential = NaN
    err_gradient = NaN
    if N <= DIRECT_MAX
        ref = generate_gravitational(23000 + P + LH, N)
        FastMultipole.direct!(ref; scalar_potential=true, gradient=true)
        LH || (err_potential = maximum(abs.(sys.potential[1, :] .- ref.potential[1, :])))
        err_gradient = maximum(abs.(sys.potential[5:7, :] .- ref.potential[5:7, :]))
    end

    plan = state.scratch.m2l_concat
    class_counts = plan isa FastMultipole.ResidentM2LFactoredPlan ?
        [Int(c) for c in plan.host_class_counts if c > 0] : Int[]
    resident_groups = plan isa FastMultipole.ResidentM2LFactoredPlan ?
        length(plan.groups) : 0
    return (host=gethostname(), gpu=CUDA.name(CUDA.device()),
        variant=label, P=P, lh=LH, n=N, ell=ELL,
        routes=state.counts.n_routes, n_direct=state.counts.n_direct,
        resident_groups=resident_groups,
        nonempty_classes=length(class_counts),
        max_class=maximum(class_counts; init=0),
        mean_class=isempty(class_counts) ? 0.0 : mean(class_counts),
        construction_ms=construct_ms,
        persistent_device_bytes=persistent_device_bytes,
        peak_device_bytes=peak_device_bytes,
        b2m_ms=b2m_ms, m2m_ms=m2m_ms, m2l_ms=m2l_ms, l2l_ms=l2l_ms, l2b_ms=l2b_ms,
        m2l_device_alloc_bytes=m2l_device_alloc,
        full_step_ms=median(full_ms), full_step_host_alloc_bytes=full_host_alloc,
        err_potential=err_potential, err_gradient=err_gradient)
end

rows = NamedTuple[]
for N in NS, P in PS, LH in (false, true)
    for (operator, label, wp) in ((MaterializedYRotationM2L(), "concat", true),
                                  (FactoredRotationM2L(), "factored", true),
                                  (FactoredRotationM2L(), "factored_perclass", false))
        row = measure_variant(P, LH, N, operator, label; whole_pass=wp)
        push!(rows, row)
        @printf("%-9s P=%-2d LH=%-5s N=%-6d  m2l %8.3f ms  step %8.3f ms  mem %7.1f MB\n",
            row.variant, row.P, string(row.lh), row.n, row.m2l_ms, row.full_step_ms,
            row.persistent_device_bytes / 1e6)
        flush(stdout)
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
