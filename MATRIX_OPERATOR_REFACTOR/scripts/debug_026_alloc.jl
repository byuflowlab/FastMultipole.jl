using FastMultipole
using FastMultipole.StaticArrays
include(joinpath(@__DIR__, "..", "..", "test", "gravitational.jl"))

const N = parse(Int, get(ENV, "FM026_N", "2000"))
const WINDOW = parse(Int, get(ENV, "FM026_WINDOW_CLASSES", "8"))

# ---------------------------------------------------------------------------
# Matrix mode (FM026_MATRIX=true): post-warmup `@allocated _launch_resident_m2l!`
# for every strategy x policy x window cell. This is verification step (a) of the
# concretely-typed resident struct refactor: the M2L allocation contract is 64 KiB
# per pass, and `Any`-typed struct fields leak ~20-25 B of boxing per field read,
# so a whole-matrix probe is the only thing that catches a boxing regression that
# the inline typeasserts happen not to cover.
# ---------------------------------------------------------------------------

const MATRIX_STRATEGIES = (
    (label="concat", operator=MaterializedYRotationM2L(),
        strategy=ConcatenatedFixedZM2L()),
    (label="factored", operator=FactoredRotationM2L(),
        strategy=ConcatenatedFixedZM2L()),
    (label="precomputed_y", operator=FactoredRotationM2L(),
        strategy=PrecomputedFactoredYM2L()),
    (label="dense", operator=MaterializedYRotationM2L(),
        strategy=DenseTranslationM2L()),
)

# Mid-gap epsilon: accept |offset|^2 = 13, reject 12 (see
# test/hierarchical_m2l_host_test.jl:11-18).
function matrix_epsilon(::Type{TF}, P, h0, ell) where TF
    probe = ConstantPStencilConfig(P, one(TF))
    upper = constant_p_stencil_bound(TF(h0), ell, probe, SVector(2, 2, 2))
    lower = constant_p_stencil_bound(TF(h0), ell, probe, SVector(3, 2, 0))
    return (upper + lower) / 2
end

function matrix_probe()
    n = parse(Int, get(ENV, "FM026_MATRIX_N", "20000"))
    ell = parse(Int, get(ENV, "FM026_MATRIX_ELL", "4"))
    P = parse(Int, get(ENV, "FM026_MATRIX_P", "4"))
    windows = parse.(Int, split(get(ENV, "FM026_MATRIX_WINDOWS", "1,4,8,128"), ','))
    reps = parse(Int, get(ENV, "FM026_MATRIX_REPS", "3"))
    h0 = 0.51
    eps = matrix_epsilon(Float64, P, h0, ell)
    config = ConstantPStencilConfig(P, eps)
    policies = Any[("flat", ConstantPAnalyticStencil(config))]
    for w in windows
        push!(policies, ("hierarchical_12_w$w",
            HierarchicalRigidStencil(config; near_radius2=12, window_classes=w)))
    end
    base = generate_gravitational(26026, n)
    println("strategy,policy,m2l_alloc_bytes")
    for st in MATRIX_STRATEGIES, (plabel, policy) in policies
        sys = deepcopy(base)
        options = CUDARadixLifecycleOptions(; precision=Float64,
            operator=st.operator, m2l_strategy=st.strategy)
        cache = RadixFMMCache(sys; expansion_order=P, ell, max_n_bodies=n,
            bounds=(SVector(-0.01, -0.01, -0.01), 1.02), policy, options)
        state = cache.state
        fmm!(sys, cache; scalar_potential=true, gradient=true)
        FastMultipole._launch_resident_m2l!(state)
        best = typemax(Int)
        for _ in 1:reps
            best = min(best, @allocated FastMultipole._launch_resident_m2l!(state))
        end
        println(st.label, ',', plabel, ',', best)
        flush(stdout)
    end
end

if lowercase(get(ENV, "FM026_MATRIX", "false")) == "true"
    matrix_probe()
    exit(0)
end

sys = generate_gravitational(26026, N)
lo, hi = FastMultipole._radix_bounds((sys,), Float64)
h0 = maximum((hi - lo) * 0.5) * 1.05
probe = ConstantPStencilConfig(4, 1.0)
upper = constant_p_stencil_bound(h0, 4, probe, SVector(2, 2, 2))
lower = constant_p_stencil_bound(h0, 4, probe, SVector(3, 2, 0))
policy = HierarchicalRigidStencil(4, (upper + lower) / 2;
    window_classes=WINDOW)
cache = RadixFMMCache(sys; expansion_order=4, ell=4, policy)
state = cache.state
ctx = state.interaction_list
plan = ctx.apply_plan
ctx.profile_stages = lowercase(get(ENV, "FM026_PROFILE", "false")) == "true"
println("route_capacity=", length(state.route_sources),
    " route_class_capacity=", length(plan.route_class),
    " noffsets=", length(ctx.tables.push_offsets))
FastMultipole._launch_host_b2m!(state)
FastMultipole._launch_host_m2m!(state)
FastMultipole._launch_host_m2l!(state)
println("full=", @allocated(FastMultipole._launch_host_m2l!(state)))

function diagnose_windows(state, ctx, plan)
    build_alloc = 0
    apply_alloc = 0
    calls = 0
    noffsets = length(ctx.tables.push_offsets)
    for level in 2:state.grid.ell
        for first_offset in 1:ctx.window_classes:noffsets
            last_offset = min(first_offset + ctx.window_classes - 1, noffsets)
            n = 0
            build_alloc += @allocated begin
                n = FastMultipole.build_hierarchical_routes_window!(
                    state.route_levels, state.route_offsets, state.route_targets,
                    state.route_sources, plan.route_class, ctx, state.grid, level,
                    first_offset, last_offset)
            end
            apply_alloc += @allocated FastMultipole._launch_hierarchical_concat_window!(
                state, state.scratch, plan, n)
            calls += 1
        end
    end
    println("calls=$calls build=$build_alloc apply=$apply_alloc")
end
diagnose_windows(state, ctx, plan)
