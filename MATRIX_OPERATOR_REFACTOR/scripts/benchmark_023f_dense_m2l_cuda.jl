# Task 023f: CUDA device-resident dense-translation M2L benchmark.
#
# Compares the resident device M2L strategies on the recurring RadixFMMCache
# (device=true) lifecycle: the production concat path, the 023b factored
# whole-pass path, the 023d precomputed-y whole-pass path, the 023f dense
# whole-pass path, and the 023f dense per-class reference baseline. Records
# construction cost, persistent/peak device memory, dense operator/persistent
# bytes, the full dense CUDA lifecycle estimate (`dense_estimated_peak_bytes`),
# per-stage GPU timings, per-step wall time, device allocation on the M2L stage,
# class occupancy, the selected apply/chunk policy, and accuracy vs direct!. One
# CSV row per (variant, precision, P, LH, N). Dense configurations that exceed
# the memory gate or materialize non-finite operators are recorded as `fit=false`
# rows (the failure is explicit and informative for task 024). Pattern:
# benchmark_023d_precomputed_y_cuda.jl.
#
# Env knobs:
#   FM023F_N      comma list of body counts        (default "150,2000,20000")
#   FM023F_P      comma list of expansion orders   (default "4,8,12")
#   FM023F_ELL    radix depth                      (default "3")
#   FM023F_REPS   timing repetitions (median)      (default "5")
#   FM023F_OUT    output CSV path
#   FM023F_DIRECT_MAX  max N for the direct! accuracy reference (default 20000)
#   FM023F_CHUNK  override the dense whole-pass chunk width (DENSE_CUDA_CHUNK)
#   FM023F_F32    also run a Float32 dense row per config ("1" to enable, default 1)

using FastMultipole
using FastMultipole.StaticArrays
using Statistics
using Dates
using Printf

include(joinpath(@__DIR__, "..", "..", "test", "gravitational.jl"))

FastMultipole.load_cuda_radix_lifecycle!() ||
    error("CUDA radix lifecycle failed to load: $(FastMultipole.cuda_radix_status())")
using CUDA

const NS = parse.(Int, split(get(ENV, "FM023F_N", "150,2000,20000"), ','))
const PS = parse.(Int, split(get(ENV, "FM023F_P", "4,8,12"), ','))
const ELL = parse(Int, get(ENV, "FM023F_ELL", "3"))
const REPS = parse(Int, get(ENV, "FM023F_REPS", "5"))
const DIRECT_MAX = parse(Int, get(ENV, "FM023F_DIRECT_MAX", "20000"))
const RUN_F32 = get(ENV, "FM023F_F32", "1") == "1"
const OUT = get(ENV, "FM023F_OUT", joinpath(@__DIR__, "..", "data",
    "dense_translation_m2l_cuda",
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

if haskey(ENV, "FM023F_CHUNK")
    FastMultipole.DENSE_CUDA_CHUNK[] = parse(Int, ENV["FM023F_CHUNK"])
end

# (label, operator, strategy, whole-pass ref, whole-pass value) per variant
const VARIANTS = (
    (label="concat", operator=MaterializedYRotationM2L(),
        strategy=FastMultipole.ConcatenatedFixedZM2L(),
        wp_ref=FastMultipole.FACTORED_CUDA_WHOLE_PASS, wp=true),
    (label="factored", operator=FactoredRotationM2L(),
        strategy=FastMultipole.ConcatenatedFixedZM2L(),
        wp_ref=FastMultipole.FACTORED_CUDA_WHOLE_PASS, wp=true),
    (label="precomputed_y", operator=FactoredRotationM2L(),
        strategy=FastMultipole.PrecomputedFactoredYM2L(),
        wp_ref=FastMultipole.PRECOMPUTED_CUDA_WHOLE_PASS, wp=true),
    (label="dense", operator=MaterializedYRotationM2L(),
        strategy=DenseTranslationM2L(),
        wp_ref=FastMultipole.DENSE_CUDA_WHOLE_PASS, wp=true),
    (label="dense_perclass", operator=MaterializedYRotationM2L(),
        strategy=DenseTranslationM2L(),
        wp_ref=FastMultipole.DENSE_CUDA_WHOLE_PASS, wp=false),
    # fused per-route kernel driver (optimize phase, production default since
    # the 2026-07-22 A/B): the toggled ref here is DENSE_CUDA_FUSED (the
    # whole_pass CSV column reads true for this row by construction of the
    # shared toggle plumbing; the label disambiguates)
    (label="dense_fused", operator=MaterializedYRotationM2L(),
        strategy=DenseTranslationM2L(),
        wp_ref=FastMultipole.DENSE_CUDA_FUSED, wp=true),
)

function _empty_row(variant, TF, P, LH, N, fit, note)
    return (host=gethostname(), gpu=CUDA.name(CUDA.device()), variant=variant.label,
        precision=string(TF), P=P, lh=LH, n=N, ell=ELL, fit=fit, note=note,
        routes=0, n_direct=0, nonempty_classes=0, max_class=0, mean_class=0.0,
        construction_ms=NaN, persistent_device_bytes=0, peak_device_bytes=0,
        dense_operator_bytes=0, dense_persistent_bytes=0, dense_estimated_peak_bytes=0,
        chunk=FastMultipole.DENSE_CUDA_CHUNK[], whole_pass=variant.wp,
        b2m_ms=NaN, m2m_ms=NaN, m2l_ms=NaN, l2l_ms=NaN, l2b_ms=NaN,
        m2l_device_alloc_bytes=0, full_step_ms=NaN, full_step_host_alloc_bytes=0,
        err_potential=NaN, err_gradient=NaN)
end

function measure_variant(P, LH, N, variant; precision=Float64)
    saved_wp = variant.wp_ref[]
    saved_fused = FastMultipole.DENSE_CUDA_FUSED[]
    # fused is the production default; only the dense_fused row measures it —
    # the GEMM rows force it off so their drivers actually run
    FastMultipole.DENSE_CUDA_FUSED[] = false
    variant.wp_ref[] = variant.wp
    try
        return _measure_variant(P, LH, N, variant, precision)
    catch err
        err isa ArgumentError && occursin("DenseTranslationM2L", sprint(showerror, err)) &&
            return _empty_row(variant, precision, P, LH, N, false,
                replace(sprint(showerror, err), ',' => ';', '\n' => ' '))
        rethrow()
    finally
        variant.wp_ref[] = saved_wp
        FastMultipole.DENSE_CUDA_FUSED[] = saved_fused
    end
end

function _measure_variant(P, LH, N, variant, ::Type{TF}) where TF
    sys = generate_gravitational(23000 + P + LH, N)
    opts = CUDARadixLifecycleOptions(; precision=TF, operator=variant.operator,
        m2l_strategy=variant.strategy)
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
        LH || (err_potential = maximum(abs.(Float64.(sys.potential[1, :]) .- ref.potential[1, :])))
        err_gradient = maximum(abs.(Float64.(sys.potential[5:7, :]) .- ref.potential[5:7, :]))
    end

    plan = state.scratch.m2l_concat
    class_counts = Int[]
    dense_operator_bytes = 0
    dense_persistent_bytes = 0
    dense_estimated_peak_bytes = 0
    if plan isa FastMultipole.ResidentM2LDenseCUDAPlan
        class_counts = [Int(c) for c in plan.host_class_counts if c > 0]
        dense_operator_bytes = plan.operator_bytes
        dense_persistent_bytes = plan.persistent_bytes
        dense_estimated_peak_bytes = plan.estimated_peak_bytes
    elseif plan isa FastMultipole.ResidentM2LFactoredPlan
        class_counts = [Int(c) for c in plan.host_class_counts if c > 0]
    elseif plan isa FastMultipole.ResidentM2LPrecomputedYPlan
        class_counts = [c for c in plan.offset_counts if c > 0]
    end
    return (host=gethostname(), gpu=CUDA.name(CUDA.device()), variant=variant.label,
        precision=string(TF), P=P, lh=LH, n=N, ell=ELL, fit=true, note="",
        routes=state.counts.n_routes, n_direct=state.counts.n_direct,
        nonempty_classes=length(class_counts), max_class=maximum(class_counts; init=0),
        mean_class=isempty(class_counts) ? 0.0 : mean(class_counts),
        construction_ms=construct_ms,
        persistent_device_bytes=persistent_device_bytes,
        peak_device_bytes=peak_device_bytes,
        dense_operator_bytes=dense_operator_bytes,
        dense_persistent_bytes=dense_persistent_bytes,
        dense_estimated_peak_bytes=dense_estimated_peak_bytes,
        chunk=FastMultipole.DENSE_CUDA_CHUNK[], whole_pass=variant.wp,
        b2m_ms=b2m_ms, m2m_ms=m2m_ms, m2l_ms=m2l_ms, l2l_ms=l2l_ms, l2b_ms=l2b_ms,
        m2l_device_alloc_bytes=m2l_device_alloc,
        full_step_ms=median(full_ms), full_step_host_alloc_bytes=full_host_alloc,
        err_potential=err_potential, err_gradient=err_gradient)
end

rows = NamedTuple[]
for N in NS, P in PS, LH in (false, true)
    for variant in VARIANTS
        row = measure_variant(P, LH, N, variant)
        push!(rows, row)
        @printf("%-16s %-8s P=%-2d LH=%-5s N=%-6d fit=%-5s m2l %8.3f ms  step %8.3f ms  mem %7.1f MB\n",
            row.variant, row.precision, row.P, string(row.lh), row.n, string(row.fit),
            row.m2l_ms, row.full_step_ms, row.persistent_device_bytes / 1e6)
        flush(stdout)
    end
    if RUN_F32
        for variant in (VARIANTS[4], VARIANTS[5], VARIANTS[6])   # dense variants, Float32
            row = measure_variant(P, LH, N, variant; precision=Float32)
            push!(rows, row)
            @printf("%-16s %-8s P=%-2d LH=%-5s N=%-6d fit=%-5s m2l %8.3f ms  step %8.3f ms  mem %7.1f MB\n",
                row.variant, row.precision, row.P, string(row.lh), row.n, string(row.fit),
                row.m2l_ms, row.full_step_ms, row.persistent_device_bytes / 1e6)
            flush(stdout)
        end
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
