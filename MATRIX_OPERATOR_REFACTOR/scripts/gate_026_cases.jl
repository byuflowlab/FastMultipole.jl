# Task 027 Checkpoint D: one side of the mandatory 026 old-versus-new gate.
#
# Runs a fixed device case matrix against WHATEVER SOURCE IS CURRENTLY IN `src/`
# and writes a `cases.csv` in the schema `compare_026_regression.jl` consumes. The
# driver (`cuda_027_gate.sh`) swaps `src/` between the pre-026 snapshot and the
# current tree and calls this script alternately, so both sides run on the same
# node, the same GPU, the same Julia/CUDA/BLAS, the same seeds, and the same case
# order.
#
# Deliberately restricted to the FLAT `ConstantPAnalyticStencil` path, selected by
# passing `stencil_epsilon` explicitly: the pre-026 source has no hierarchical
# policy at all (task 026 introduced it), so flat is the only common surface. The
# 026 refactor's target paths -- flat precomputed-y and flat dense, whose residual
# allocations it removed -- are both covered.
#
# Env knobs:
#   FM026G_OUT    output directory (a `cases.csv` is written inside)
#   FM026G_LABEL  free-form label recorded in every row (e.g. "old_r1")
#   FM026G_REPS   timing repetitions (median)   (default "7")

using FastMultipole
using FastMultipole.StaticArrays
using Statistics
using LinearAlgebra
using Dates

include(joinpath(@__DIR__, "..", "..", "test", "gravitational.jl"))

FastMultipole.load_cuda_radix_lifecycle!() ||
    error("CUDA radix lifecycle failed to load: $(FastMultipole.cuda_radix_status())")
using CUDA

const FM = FastMultipole
const OUTDIR = get(ENV, "FM026G_OUT", @__DIR__)
const LABEL = get(ENV, "FM026G_LABEL", "unlabeled")
const REPS = parse(Int, get(ENV, "FM026G_REPS", "7"))
const ELL = 3
const SEED = 20260730

const STRATEGIES = (
    ("concat", MaterializedYRotationM2L(), FM.ConcatenatedFixedZM2L()),
    ("factored", FactoredRotationM2L(), FM.ConcatenatedFixedZM2L()),
    ("precomputed_y", FactoredRotationM2L(), FM.PrecomputedFactoredYM2L()),
    ("dense", MaterializedYRotationM2L(), DenseTranslationM2L()),
)

_median_ms(f) = begin
    f()
    ts = Float64[]
    for _ in 1:REPS
        push!(ts, (@elapsed f()) * 1e3)
    end
    (median(ts), minimum(ts), ts)
end

function measure(label, strat, TF, P, n)
    name, operator, m2l_strategy = strat
    sys = generate_gravitational(SEED, n)
    opts = CUDARadixLifecycleOptions(; precision=TF, operator, m2l_strategy)
    GC.gc(); CUDA.reclaim()
    t0 = time_ns()
    # `stencil_epsilon` pins the flat classifier on both sources
    cache = RadixFMMCache(sys; expansion_order=P, ell=ELL, max_n_bodies=n,
        device=true, options=opts, stencil_epsilon=1e-4)
    CUDA.synchronize()
    construction_ms = (time_ns() - t0) / 1e6
    state = cache.state
    counters = state.counters

    fmm!(sys, cache; scalar_potential=true, gradient=true)   # warm
    m2l_med, m2l_min, m2l_raw = _median_ms(() -> begin
        FM._launch_cuda_resident_m2l!(state); CUDA.synchronize()
    end)
    step_med, step_min, step_raw = _median_ms(
        () -> fmm!(sys, cache; scalar_potential=true, gradient=true))
    update_med, _, _ = _median_ms(
        () -> FM.update_cuda_radix_state!(cache, (sys,)))

    m2l_alloc = CUDA.@allocated FM._launch_cuda_resident_m2l!(state)
    update_alloc = CUDA.@allocated FM.update_cuda_radix_state!(cache, (sys,))
    step_alloc = CUDA.@allocated fmm!(sys, cache; scalar_potential=true, gradient=true)

    return (; label, policy="flat", strategy=name, blas_threads=BLAS.get_num_threads(),
        ell=ELL, n, window_classes=0, distribution="uniform", precision=string(TF),
        lh=false, P,
        m2l_ms_median=m2l_med, m2l_ms_min=m2l_min,
        step_ms_median=step_med, step_ms_min=step_min,
        update_ms_median=update_med, construction_ms,
        m2l_alloc_bytes=m2l_alloc, update_alloc_bytes=update_alloc,
        step_alloc_bytes=step_alloc,
        expansion_host_copies=counters.expansion_host_copies,
        route_uploads=counters.route_uploads,
        operator_uploads=counters.operator_uploads,
        n_routes=state.counts.n_routes,
        m2l_raw=join(round.(m2l_raw; digits=5), ' '),
        step_raw=join(round.(step_raw; digits=5), ' '))
end

# Optional single-cell restriction, used by the Checkpoint D tiebreaker to run one
# configuration many times with alternating sources.
const ONLY_STRAT = get(ENV, "FM026G_ONLY_STRATEGY", "")
const ONLY_P = get(ENV, "FM026G_ONLY_P", "")
const ONLY_N = get(ENV, "FM026G_ONLY_N", "")
_selected(name, P, n) =
    (isempty(ONLY_STRAT) || name == ONLY_STRAT) &&
    (isempty(ONLY_P) || P == parse(Int, ONLY_P)) &&
    (isempty(ONLY_N) || n == parse(Int, ONLY_N))

rows = NamedTuple[]
for P in (4, 8), n in (2000, 20000), strat in STRATEGIES
    _selected(strat[1], P, n) || continue
    row = try
        measure(LABEL, strat, Float64, P, n)
    catch err
        @warn "case failed" strategy=strat[1] P n exception=err
        nothing
    end
    row === nothing && continue
    push!(rows, row)
    println(rpad(LABEL, 10), rpad(row.strategy, 14), " P=", P, " n=", n,
        "  m2l=", round(row.m2l_ms_median; digits=4), " ms  step=",
        round(row.step_ms_median; digits=4), " ms  m2l_alloc=", row.m2l_alloc_bytes)
    flush(stdout)
    GC.gc(); CUDA.reclaim()
end

mkpath(OUTDIR)
out = joinpath(OUTDIR, "cases.csv")
open(out, "w") do io
    println(io, join(string.(keys(rows[1])), ','))
    for row in rows
        println(io, join(string.(values(row)), ','))
    end
end
println("wrote ", out, "  (", length(rows), " cases)")
