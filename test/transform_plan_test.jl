#=##############################################################################
transform_plan!: rigid-motion plan reuse, including NearfieldInfluenceCache
persistence (FLOWPanel BRAINSTORM 021).

For the scalar-potential operator the cached near-field blocks are EXACTLY
invariant under rigid motion (scalar kernel of relative distances), so a
cache built before the motion must keep reproducing the freshly-computed
near field after transform_plan!. Direction-carrying outputs (gradient/
hessian/extra) with a stored cache refuse loudly in v1.

Requires gravitational.jl.
=###############################################################################

@testset "transform_plan!: rigid motion with a nearfield cache" begin

    n_bodies = 2000
    R = _rodrigues(SVector(-0.5, 0.8, 0.33), 24.0 * pi / 180)
    t = SVector(0.15, 0.45, -0.3)

    kwargs = (; expansion_order=8, multipole_acceptance=0.4,
              leaf_size_source=30, scalar_potential=true, gradient=false,
              hessian=false)

    sys = generate_gravitational(321, n_bodies)
    plan = FastMultipole.FmmPlan((sys,), (sys,); kwargs...)
    cache = FastMultipole.build_nearfield_cache!(plan, (sys,), (sys,))

    # premise guards
    @test length(plan.m2l_list) > 0
    @test length(plan.direct_list) > 0
    @test length(cache.entries) > 0
    @test plan.nearfield_cache[] === cache

    sys.potential .= 0
    FastMultipole.fmm!((sys,), (sys,), plan)
    phi0 = copy(sys.potential[1, :])
    @test any(!iszero, phi0)

    # rigid motion; the plan (trees + lists + CACHE) is reused, not rebuilt
    rotate!(sys, i) = (b = sys.bodies[i];
        sys.bodies[i] = typeof(b)(R * b.position + t, b.radius, b.strength))
    for i in 1:n_bodies
        rotate!(sys, i)
    end
    FastMultipole.transform_plan!(plan, (sys,), R, t)
    @test plan.nearfield_cache[] === cache      # cache survived the transform

    sys.potential .= 0
    FastMultipole.fmm!((sys,), (sys,), plan)
    phi1 = copy(sys.potential[1, :])

    # scalar potential is invariant under rigid motion; the cached blocks must
    # reproduce it exactly (same tree, same arithmetic path as pre-motion)
    @test isapprox(phi1, phi0; rtol=1e-12)

    # cached near field == freshly computed near field on the MOVED geometry
    plan.nearfield_cache[] = nothing
    sys.potential .= 0
    FastMultipole.fmm!((sys,), (sys,), plan)
    phi_nocache = copy(sys.potential[1, :])
    @test isapprox(phi1, phi_nocache; rtol=1e-12)
    plan.nearfield_cache[] = cache

    # second accumulated motion: composability across steps
    R2 = _rodrigues(SVector(0.9, -0.1, 0.6), 41.0 * pi / 180)
    t2 = SVector(-0.2, 0.1, 0.55)
    rotate2!(sys, i) = (b = sys.bodies[i];
        sys.bodies[i] = typeof(b)(R2 * b.position + t2, b.radius, b.strength))
    for i in 1:n_bodies
        rotate2!(sys, i)
    end
    FastMultipole.transform_plan!(plan, (sys,), R2, t2)
    sys.potential .= 0
    FastMultipole.fmm!((sys,), (sys,), plan)
    @test isapprox(sys.potential[1, :], phi0; rtol=1e-12)

end

@testset "transform_plan!: refusals" begin

    n_bodies = 800
    R = _rodrigues(SVector(0.0, 0.0, 1.0), 15.0 * pi / 180)
    t = SVector(0.0, 0.0, 0.0)

    # gradient outputs + stored cache: loud v1 refusal
    sys = generate_gravitational(11, n_bodies)
    plan = FastMultipole.FmmPlan((sys,), (sys,); expansion_order=6,
        multipole_acceptance=0.4, leaf_size_source=30,
        scalar_potential=true, gradient=true, hessian=false)
    cache = FastMultipole.build_nearfield_cache!(plan, (sys,), (sys,))
    @test length(cache.entries) > 0                       # non-vacuous
    @test_throws ArgumentError FastMultipole.transform_plan!(plan, (sys,), R, t)

    # same switches WITHOUT a cache: transform proceeds (gradient recomputed
    # fresh from buffers each call) and equivariance holds
    plan2 = FastMultipole.FmmPlan((sys,), (sys,); expansion_order=6,
        multipole_acceptance=0.4, leaf_size_source=30,
        scalar_potential=true, gradient=true, hessian=false)
    sys.potential .= 0
    FastMultipole.fmm!((sys,), (sys,), plan2)
    g0 = [SVector{3}(sys.potential[i_gradient, i]) for i in 1:n_bodies]
    @test any(g -> norm(g) > 0, g0)
    for i in 1:n_bodies
        b = sys.bodies[i]
        sys.bodies[i] = typeof(b)(R * b.position + t, b.radius, b.strength)
    end
    FastMultipole.transform_plan!(plan2, (sys,), R, t)
    sys.potential .= 0
    FastMultipole.fmm!((sys,), (sys,), plan2)
    g1 = [SVector{3}(sys.potential[i_gradient, i]) for i in 1:n_bodies]
    g0r = reduce(vcat, [R * g for g in g0])
    @test isapprox(norm(reduce(vcat, g1) - g0r), 0.0; atol=1e-12 * norm(g0r))

    # body-count mismatch refuses
    sys_small = generate_gravitational(12, n_bodies ÷ 2)
    @test_throws ArgumentError FastMultipole.transform_plan!(plan2, (sys_small,), R, t)

end
