using LinearAlgebra

@testset "precomputed-y resident M2L (task 023c)" begin
    key = FastMultipole._precomputed_y_angle_key
    @test key(SVector(1, 0, 1)) == key(SVector(2, 0, 2))
    @test key(SVector(1, 1, 0)) == key(SVector(2, 2, 0)) == (0, 0, 1)
    @test key(SVector(0, 0, 1)) == (1, 1, 0)
    @test key(SVector(0, 0, -3)) == (-1, 1, 0)
    @test key(SVector(1, 0, 1)) != key(SVector(1, 0, -1))

    offsets = [SVector(1, 0, 1), SVector(0, 0, -1), SVector(2, 0, 2),
               SVector(1, 1, 0), SVector(0, 0, 3)]
    keys, offset_to_angle, starts, packed =
        FastMultipole._precomputed_y_angle_metadata(offsets)
    @test offset_to_angle == [1, 2, 1, 3, 4]
    @test keys == [(1, 1, 1), (-1, 1, 0), (0, 0, 1), (1, 1, 0)]
    @test starts == [1, 3, 4, 5, 6]
    @test packed == [1, 3, 2, 4, 5]

    @test_throws ArgumentError CUDARadixLifecycleOptions(
        m2l_strategy=PrecomputedFactoredYM2L())

    one_shot_sys = generate_gravitational(20260718, 80)
    one_shot_grid = RadixGrid(one_shot_sys, 3)
    one_shot_list = build_radix_interaction_list(
        LazyMaterializedBatches(1), ParentNeighborM2L(), one_shot_grid)
    one_shot = host_radix_state(one_shot_sys, one_shot_grid, one_shot_list, 4;
        options=CUDARadixLifecycleOptions(; operator=FactoredRotationM2L(),
            m2l_strategy=PrecomputedFactoredYM2L()))
    @test one_shot.scratch.m2l_concat isa FastMultipole.ResidentM2LPrecomputedYPlan
    @test run_host_radix_lifecycle!(one_shot) === one_shot

    # Task 027 made HierarchicalRigidStencil the default policy. Every cache below
    # verifies the FLAT precomputed-y resident plan (task 023c) -- angle-major
    # packing, angle_counts/angle_capacities, and the empty-angle-class property --
    # which the hierarchical windowed driver deliberately does not populate. Passing
    # stencil_epsilon explicitly pins these caches to the flat classifier.
    seed = 20260718
    for (TF, LH) in ((Float64, false), (Float64, true),
                     (Float32, false), (Float32, true))
        a = generate_gravitational(seed, 120)
        b = generate_gravitational(seed, 120)
        pre = RadixFMMCache(a; expansion_order=4, ell=3, lamb_helmholtz=LH,
            stencil_epsilon=1e-4,
            options=CUDARadixLifecycleOptions(; precision=TF,
                operator=FactoredRotationM2L(), m2l_strategy=PrecomputedFactoredYM2L()))
        ref = RadixFMMCache(b; expansion_order=4, ell=3, lamb_helmholtz=LH,
            stencil_epsilon=1e-4,
            options=CUDARadixLifecycleOptions(; precision=TF,
                operator=FactoredRotationM2L(), m2l_strategy=ConcatenatedFixedZM2L()))
        plan = pre.state.scratch.m2l_concat
        @test plan isa FastMultipole.ResidentM2LPrecomputedYPlan
        @test sum(plan.angle_counts) == pre.state.counts.n_routes
        @test all(plan.angle_counts .<= plan.angle_capacities)
        matrix_ids = objectid.(Iterators.flatten(plan.y_mult))
        route_id = objectid(plan.route_class)
        packed_id = objectid(plan.packed_sources)

        fmm!(a, pre; scalar_potential=!LH, gradient=true)
        fmm!(b, ref; scalar_potential=!LH, gradient=true)
        tol = TF === Float64 ? 2e-10 : 3f-5
        @test maximum(abs.(a.potential .- b.potential)) < tol

        # A repeated refresh changes only counts/prefix contents, never capacities
        # or construction-time matrices.
        FastMultipole.update_radix_state!(pre, (a,))
        @test objectid(plan.route_class) == route_id
        @test objectid(plan.packed_sources) == packed_id
        @test objectid.(Iterators.flatten(plan.y_mult)) == matrix_ids
        FastMultipole._launch_resident_m2l!(pre.state)
        @test (@allocated FastMultipole._launch_resident_m2l!(pre.state)) <= 64 * 1024

        # Empty angle classes/routes are valid and leave locals exactly zero.
        saved_counts = copy(plan.angle_counts)
        fill!(plan.angle_counts, 0)
        FastMultipole._launch_resident_m2l!(pre.state)
        @test all(iszero, pre.state.locals.phi)
        LH && @test all(iszero, pre.state.locals.chi)
        copyto!(plan.angle_counts, saved_counts)
    end

    # Construction-time M_n(theta) agrees with the retained 023a U/D/V kernel.
    sys = generate_gravitational(seed + 1, 80)
    cache = RadixFMMCache(sys; expansion_order=4, ell=3,
        stencil_epsilon=1e-4,
        options=CUDARadixLifecycleOptions(; operator=FactoredRotationM2L(),
            m2l_strategy=PrecomputedFactoredYM2L()))
    ws = cache.state.scratch
    plan = ws.m2l_concat
    rng = MersenneTwister(seed)
    for angle in (1, length(plan.angle_keys)), n in 0:4
        d = 2n + 1
        X = randn(rng, d, 5)
        Y = zeros(d, 5)
        Gre = similar(Y); Gim = similar(Y)
        Ure, Uim = ws.y_mult_U[n + 1]
        Vre, Vim = ws.y_mult_V[n + 1]
        FastMultipole._factored_y_degree_block_noalloc!(Y, X, Ure, Uim, Vre, Vim,
            fill(plan.angle_thetas[angle], 5), Gre, Gim, 1, d, 5)
        @test plan.y_mult[angle][n + 1] * X ≈ Y rtol=2e-13 atol=2e-13
    end


    # Direct and materialized-concat parity at the stencil design order.
    pre_sys = generate_gravitational(seed + 2, 180)
    concat_sys = generate_gravitational(seed + 2, 180)
    direct_sys = generate_gravitational(seed + 2, 180)
    direct!(direct_sys; scalar_potential=true, gradient=true)
    pre_cache = RadixFMMCache(pre_sys; expansion_order=8, ell=3,
        stencil_epsilon=1e-4,
        options=CUDARadixLifecycleOptions(; operator=FactoredRotationM2L(),
            m2l_strategy=PrecomputedFactoredYM2L()))
    concat_cache = RadixFMMCache(concat_sys; expansion_order=8, ell=3,
        stencil_epsilon=1e-4,
        options=CUDARadixLifecycleOptions(; operator=MaterializedYRotationM2L(),
            m2l_strategy=ConcatenatedFixedZM2L()))
    fmm!(pre_sys, pre_cache; scalar_potential=true, gradient=true)
    fmm!(concat_sys, concat_cache; scalar_potential=true, gradient=true)
    @test maximum(abs.(pre_sys.potential .- concat_sys.potential)) < 2e-10
    @test maximum(abs.(pre_sys.potential[1, :] .- direct_sys.potential[1, :])) < 1e-6
    @test maximum(abs.(pre_sys.potential[5:7, :] .- direct_sys.potential[5:7, :])) < 1e-4

    # Varying body-count refreshes reuse packed metadata and matrix storage.
    full = generate_gravitational(seed + 3, 160)
    moving_cache = RadixFMMCache(full; expansion_order=4, ell=3, max_n_bodies=160,
        stencil_epsilon=1e-4,
        bounds=(SVector(-0.1, -0.1, -0.1), 1.2),
        options=CUDARadixLifecycleOptions(; operator=FactoredRotationM2L(),
            m2l_strategy=PrecomputedFactoredYM2L()))
    moving_plan = moving_cache.state.scratch.m2l_concat
    fixed_ids = (objectid(moving_plan.route_class), objectid(moving_plan.packed_sources),
        objectid(moving_plan.angle_counts), objectid(moving_plan.offset_counts))
    for nbody in (73, 160, 91)
        bodies = copy(full.bodies[1:nbody])
        step_sys = Gravitational(bodies, zeros(16, nbody))
        fmm!(step_sys, moving_cache; scalar_potential=true, gradient=true)
        @test (objectid(moving_plan.route_class), objectid(moving_plan.packed_sources),
            objectid(moving_plan.angle_counts), objectid(moving_plan.offset_counts)) == fixed_ids
        @test sum(moving_plan.angle_counts) == moving_cache.state.counts.n_routes
        @test sum(moving_plan.offset_counts) == moving_cache.state.counts.n_routes
    end
    fmm!(full, moving_cache; scalar_potential=true, gradient=true)
    @test (@allocated fmm!(full, moving_cache; scalar_potential=true, gradient=true)) < 512_000
end
