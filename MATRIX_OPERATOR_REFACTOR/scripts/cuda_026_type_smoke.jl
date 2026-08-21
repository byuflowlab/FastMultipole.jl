using FastMultipole
using FastMultipole.StaticArrays
using CUDA

include(joinpath(@__DIR__, "..", "..", "test", "gravitational.jl"))

FastMultipole.load_cuda_radix_lifecycle!() ||
    error("failed to load CUDA radix lifecycle: $(FastMultipole.cuda_radix_status())")

const SMOKE_STRATEGIES = (
    ("concat", MaterializedYRotationM2L(), ConcatenatedFixedZM2L()),
    ("factored", FactoredRotationM2L(), ConcatenatedFixedZM2L()),
    ("precomputed_y", FactoredRotationM2L(), PrecomputedFactoredYM2L()),
    ("dense", MaterializedYRotationM2L(), DenseTranslationM2L()),
)

function persistent_arrays(state)
    return (
        state.source_bodies, state.body_perm, state.cell_centers, state.cell_ranges,
        state.multipoles.phi, state.multipoles.chi, state.locals.phi, state.locals.chi,
        state.route_levels, state.route_offsets, state.route_targets, state.route_sources,
        state.direct_targets, state.direct_sources, state.output,
    )
end

for (i, (label, operator, strategy)) in enumerate(SMOKE_STRATEGIES)
    sys = generate_gravitational(260260 + i, 256)
    options = CUDARadixLifecycleOptions(; precision=Float64, operator,
        m2l_strategy=strategy)
    @assert isconcretetype(typeof(options))
    cache = RadixFMMCache(sys; expansion_order=4, ell=3, max_n_bodies=256,
        bounds=(SVector(-0.01, -0.01, -0.01), 1.02), device=true, options)
    state = cache.state
    @assert fieldtype(typeof(state), :options) === typeof(options)
    arrays = persistent_arrays(state)
    route_uploads = state.counters.route_uploads
    operator_uploads = state.counters.operator_uploads

    for _ in 1:2
        fmm!(sys, cache; scalar_potential=true, gradient=true)
        CUDA.synchronize()
    end

    @assert all(a === b for (a, b) in zip(arrays, persistent_arrays(state)))
    @assert state.counters.expansion_host_copies == 0
    @assert state.counters.route_uploads == route_uploads
    @assert state.counters.operator_uploads == operator_uploads
    @assert all(isfinite, sys.potential)
    println(label, ": pass; state=", typeof(state),
        "; routes=", state.counts.n_routes, "; direct=", state.counts.n_direct)
end

println("CUDA 026 concrete-container smoke: PASS")
