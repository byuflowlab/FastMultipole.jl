using FastMultipole
using FastMultipole.StaticArrays
using Statistics
using Dates
using LinearAlgebra

include(joinpath(@__DIR__, "..", "..", "test", "gravitational.jl"))

# FM023A_N is a comma list; small N gives sparse offset classes, large N fills the
# leaf grid so classes approach full width and exercise the GEMM branch.
const NS = parse.(Int, split(get(ENV, "FM023A_N", "150,2000,20000"), ','))
const REPS = parse(Int, get(ENV, "FM023A_REPS", "3"))
const BLAS_THREADS = parse(Int, get(ENV, "FM023A_BLAS_THREADS", string(BLAS.get_num_threads())))
# optional extra crossover candidates, e.g. FM023A_SWEEP_COLS=8,16,64,128
const SWEEP_COLS = [parse(Int, s) for s in split(get(ENV, "FM023A_SWEEP_COLS", ""), ',') if !isempty(s)]
const OUT = get(ENV, "FM023A_OUT", joinpath(@__DIR__, "..", "data",
    "factored_resident_m2l_host", "host_$(gethostname())_$(Dates.format(now(), "yyyymmdd-HHMMSS")).csv"))
BLAS.set_num_threads(BLAS_THREADS)

# variant => (operator, gemm thresholds (min_cols, min_dim; nothing = defaults),
#             stage launcher). `factored_functional` runs the same factored-plan cache
# through the retained allocating shared-rotation loop (_launch_resident_m2l_shared!)
# as the pre-optimization functional baseline; its stage is not the production
# dispatch, so full-step columns are reported as NaN for that variant.
_stage_default(state) = FastMultipole._launch_resident_m2l!(state)
_stage_shared(state) =
    FastMultipole._launch_resident_m2l_shared!(state, state.scratch.m2l_concat.groups)

function measure_variant(P, LH, N, operator, label; thresholds=nothing, stage=_stage_default,
        full_step=true)
    saved_cols = FastMultipole.FACTORED_Y_GEMM_MIN_COLS[]
    saved_dim = FastMultipole.FACTORED_Y_GEMM_MIN_DIM[]
    thresholds !== nothing &&
        ((FastMultipole.FACTORED_Y_GEMM_MIN_COLS[], FastMultipole.FACTORED_Y_GEMM_MIN_DIM[]) = thresholds)
    try
        sys = generate_gravitational(23000 + P + LH, N)
        opts = CUDARadixLifecycleOptions(; operator,
            m2l_strategy=FastMultipole.ConcatenatedFixedZM2L())
        t0 = time_ns()
        cache = RadixFMMCache(sys; expansion_order=P, ell=3,
            max_n_bodies=N, lamb_helmholtz=LH, options=opts)
        construct_ms = (time_ns() - t0) / 1e6
        fmm!(sys, cache; scalar_potential=!LH, gradient=true)
        stage(cache.state)
        stage_alloc = @allocated stage(cache.state)
        stage_ms = [(@elapsed stage(cache.state)) * 1e3 for _ in 1:REPS]
        if full_step
            full_alloc = @allocated fmm!(sys, cache; scalar_potential=!LH, gradient=true)
            full_ms = [(@elapsed fmm!(sys, cache; scalar_potential=!LH, gradient=true)) * 1e3 for _ in 1:REPS]
        else
            full_alloc = -1
            full_ms = [NaN]
        end
        plan = cache.state.scratch.m2l_concat
        counts = plan isa FastMultipole.ResidentM2LFactoredPlan ?
            [g.count[] for g in plan.groups if g.count[] > 0] : Int[]
        return (P=P, lh=LH, variant=label, blas_threads=BLAS.get_num_threads(),
            n=N, routes=cache.state.counts.n_routes,
            nonempty_classes=length(counts), max_class=maximum(counts; init=0),
            mean_class=isempty(counts) ? 0.0 : mean(counts),
            gemm_min_cols=FastMultipole.FACTORED_Y_GEMM_MIN_COLS[],
            gemm_min_dim=FastMultipole.FACTORED_Y_GEMM_MIN_DIM[],
            construction_ms=construct_ms,
            persistent_bytes=Base.summarysize(cache.state.scratch),
            stage_ms=median(stage_ms), stage_alloc_bytes=stage_alloc,
            full_step_ms=median(full_ms), full_step_alloc_bytes=full_alloc)
    finally
        FastMultipole.FACTORED_Y_GEMM_MIN_COLS[] = saved_cols
        FastMultipole.FACTORED_Y_GEMM_MIN_DIM[] = saved_dim
    end
end

rows = NamedTuple[]
for N in NS, P in (4, 8, 12), LH in (false, true)
    push!(rows, measure_variant(P, LH, N, MaterializedYRotationM2L(), "concat"))
    push!(rows, measure_variant(P, LH, N, FactoredRotationM2L(), "factored_functional";
        stage=_stage_shared, full_step=false))
    push!(rows, measure_variant(P, LH, N, FactoredRotationM2L(), "factored_scalar";
        thresholds=(typemax(Int), typemax(Int))))
    push!(rows, measure_variant(P, LH, N, FactoredRotationM2L(), "factored_gemm";
        thresholds=(1, 1)))
    push!(rows, measure_variant(P, LH, N, FactoredRotationM2L(), "factored_optimized"))
    for cols in SWEEP_COLS
        push!(rows, measure_variant(P, LH, N, FactoredRotationM2L(), "factored_cols$(cols)";
            thresholds=(cols, FastMultipole.FACTORED_Y_GEMM_MIN_DIM[])))
    end
    GC.gc()
end
mkpath(dirname(OUT))
open(OUT, "w") do io
    names = propertynames(first(rows))
    println(io, join(names, ','))
    for row in rows
        println(io, join((getproperty(row, name) for name in names), ','))
    end
end
println(OUT)
