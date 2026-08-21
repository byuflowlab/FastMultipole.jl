using FastMultipole
using FastMultipole.StaticArrays
using Random
using Test

if !isdefined(@__MODULE__, :Gravitational)
    include("gravitational.jl")
end

const HIER_FM = FastMultipole

function _hier_epsilon(P, q, h0, ell)
    return rigid_stencil_epsilon(P, h0, ell, q)
end

function _hier_system(coords, ell)
    G = 1 << ell
    bodies = zeros(Float64, 8, length(coords))
    for (i, c) in enumerate(coords)
        bodies[1:3, i] .= (c .+ 0.5) ./ G
        bodies[5, i] = 1 / length(coords)
    end
    return Gravitational(bodies)
end

function _hier_policy(P, q, ell; window_classes=8)
    eps = _hier_epsilon(P, q, 0.5, ell)
    return HierarchicalRigidStencil(P, eps; near_radius2=q,
        window_classes)
end

function _hier_policy(P, q, h0, ell, ::Type{TF}, LH;
        window_classes=8) where TF
    eps = rigid_stencil_epsilon(P, h0, ell, q;
        lamb_helmholtz=LH, TF)
    return HierarchicalRigidStencil(ConstantPStencilConfig(P, eps;
        lamb_helmholtz=LH); near_radius2=q, window_classes)
end

function _hier_coverage(cache)
    state = cache.state
    ctx = state.interaction_list
    grid = state.grid
    C = state.counts.n_cells
    hits = zeros(Int, C, C)
    for i in 1:state.counts.n_direct
        hits[state.direct_targets[i], state.direct_sources[i]] += 1
    end
    noffsets = length(ctx.tables.push_offsets)
    leaf_coords = [SVector{3,Int}(grid.node_coords[:, grid.leaf_to_node[c]])
                   for c in 1:C]
    for level in 2:grid.ell, first in 1:ctx.window_classes:noffsets
        last = min(first + ctx.window_classes - 1, noffsets)
        n = HIER_FM.build_hierarchical_routes_window!(
            state.route_levels, state.route_offsets, state.route_targets,
            state.route_sources, state.scratch.m2l_concat.route_class,
            ctx, grid, level, first, last)
        shift = grid.ell - level
        for r in 1:n
            sn = state.route_sources[r]
            tn = state.route_targets[r]
            sc = SVector{3,Int}(grid.node_coords[:, sn])
            tc = SVector{3,Int}(grid.node_coords[:, tn])
            source_leaves = findall(c -> (c .>> shift) == sc, leaf_coords)
            target_leaves = findall(c -> (c .>> shift) == tc, leaf_coords)
            for t in target_leaves, s in source_leaves
                hits[t, s] += 1
            end
        end
    end
    return hits
end

function _hier_random_physical!(buf, rng)
    buf.phi .= randn(rng, eltype(buf.phi), size(buf.phi))
    for n in 0:buf.basis_info.orders.P_phi
        buf.phi[HIER_FM.flat_basis_index(n, 0, 2), :] .= 0
    end
    buf.chi .= randn(rng, eltype(buf.chi), size(buf.chi))
    for n in 0:buf.basis_info.orders.P_active
        buf.chi[HIER_FM.flat_basis_index(n, 0, 2), :] .= 0
    end
    return buf
end

_hier_m2l_allocated(state) =
    @allocated HIER_FM._launch_resident_m2l!(state)
_hier_step_allocated(sys, cache) =
    @allocated fmm!(sys, cache; scalar_potential=true, gradient=true)

@testset "hierarchical rigid host M2L (task 026)" begin
    supported_q = (3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20)
    for q in supported_q
        tables = RigidHierarchicalTables(q)
        @test all(sum(abs2, o) <= q for o in tables.near_offsets)
        @test length(Set(tables.near_offsets)) == length(tables.near_offsets)
        @test length(Set(tables.push_offsets)) == length(tables.push_offsets)
        @test all(tables.phase_starts[p + 1] > tables.phase_starts[p]
                  for p in 1:8)
        @test tables.phase_starts[end] == length(tables.phase_index) + 1
        @test maximum(maximum(abs, o) for o in tables.push_offsets) ==
            2isqrt(q) + 1
        @test minimum(sum(abs2, o) for o in tables.push_offsets) > q
        @test count(!iszero, tables.class_of) == length(tables.phase_index)
        @test rigid_stencil_epsilon(4, 0.5, 3, q) > 0
    end

    @test classic_fmm_stencil(4, 1e12).near_radius2 == 3
    @test HierarchicalRigidStencil(4, 1.0).window_classes == 4
    @test HierarchicalRigidStencil(4, 1.0; window_classes=7).window_classes == 7

    ell = 3
    dense_coords = [SVector(x, y, z) for z in 0:7 for y in 0:7 for x in 0:7]
    boundary_coords = [SVector(0, 0, 0), SVector(1, 0, 0),
        SVector(7, 7, 7), SVector(6, 7, 7), SVector(0, 7, 3)]
    sparse_coords = [SVector(0, 0, 0), SVector(3, 1, 0),
        SVector(7, 7, 7), SVector(4, 6, 2), SVector(1, 7, 5)]
    for q in supported_q, coords in (dense_coords, boundary_coords, sparse_coords)
        sys = _hier_system(coords, ell)
        cache = RadixFMMCache(sys; expansion_order=4, ell,
            bounds=(SVector(0.0, 0.0, 0.0), 1.0),
            policy=_hier_policy(4, q, ell))
        @test all(_hier_coverage(cache) .== 1)
        ctx = cache.state.interaction_list
        @test length(ctx.class_level) ==
            (ell - 1) * length(ctx.tables.push_offsets)
        @test any(==(2), ctx.class_level)
        @test any(==(ell), ctx.class_level)
        @test sum(ctx.routes_per_level) == ctx.total_routes
        @test ctx.last_window_routes <= cache.route_capacity
    end


    # Task 028 Stage 7 internal per-level schedule. Every level uses a complete
    # rigid/cubic orbit table; the leaf radius alone controls direct pairs.
    for schedule in ((5, 5), (6, 5), (6, 6)),
            coords in (dense_coords, boundary_coords, sparse_coords)
        qleaf = last(schedule)
        base = _hier_policy(4, qleaf, ell)
        policy = HIER_FM._hierarchical_stencil_with_schedule(base, schedule)
        sys = _hier_system(coords, ell)
        cache = RadixFMMCache(sys; expansion_order=4, ell,
            bounds=(SVector(0.0, 0.0, 0.0), 1.0), policy)
        @test all(_hier_coverage(cache) .== 1)
        ctx = cache.state.interaction_list
        @test size(ctx.level_class_of) ==
            (8, length(ctx.tables.push_offsets), ell + 1)
        @test ctx.tables.near_offsets == RigidHierarchicalTables(qleaf).near_offsets
    end

    # Random occupancy at ell = 4 must remain exactly-once for both radii.
    rand_cells = shuffle(MersenneTwister(0x026), 0:(16^3 - 1))[1:48]
    rand_coords = [SVector(c & 15, (c >> 4) & 15, (c >> 8) & 15)
                   for c in rand_cells]
    for q in supported_q
        rsys = _hier_system(rand_coords, 4)
        rcache = RadixFMMCache(rsys; expansion_order=4, ell=4,
            bounds=(SVector(0.0, 0.0, 0.0), 1.0),
            policy=_hier_policy(4, q, 4))
        @test all(_hier_coverage(rcache) .== 1)
    end

    # Multi-level structure and route-count reduction at n = 20000: nonleaf
    # levels must carry routes, and the total must stay well under C^2.
    big_bodies = zeros(Float64, 8, 20000)
    let lcg = UInt64(2026)
        for i in 1:20000, d in 1:3
            lcg = lcg * 6364136223846793005 + 1442695040888963407
            big_bodies[d, i] = (lcg >> 11) / 2.0^53
        end
    end
    big_bodies[5, :] .= 1 / 20000
    big_sys = Gravitational(big_bodies)
    big_cache = RadixFMMCache(big_sys; expansion_order=4, ell=5,
        bounds=(SVector(0.0, 0.0, 0.0), 1.0),
        policy=_hier_policy(4, 12, 5))
    big_ctx = big_cache.state.interaction_list
    bigC = big_cache.state.counts.n_cells
    @test big_ctx.total_routes > 0
    @test big_ctx.total_routes < 0.1 * bigC^2
    @test count(>(0), big_ctx.routes_per_level) > 1
    @test sum(big_ctx.routes_per_level[3:5]) > 0   # levels 2-4 are nonleaf

    # Direct body-to-node oracle for every propagated nonleaf multipole.
    coords = sparse_coords
    sys = _hier_system(coords, ell)
    cache = RadixFMMCache(sys; expansion_order=4, ell,
        bounds=(SVector(0.0, 0.0, 0.0), 1.0),
        policy=_hier_policy(4, 12, ell))
    state = cache.state
    HIER_FM._launch_host_b2m!(state)
    HIER_FM._launch_resident_m2m!(state)
    P = cache.expansion_order
    for node in 1:cache.level_offsets[ell + 1]
        level = state.grid.node_levels[node]
        nc = SVector{3,Int}(state.grid.node_coords[:, node])
        center = SVector{3,Float64}(state.grid.node_centers[:, node])
        for n in 0:P, m in 0:n
            er = 0.0
            ei = 0.0
            sgn = isodd(n + m) ? -1.0 : 1.0
            for k in 1:state.counts.n_bodies
                leaf = HIER_FM.radix_cell_coord(cache.x_min, cache.h0, ell,
                    SVector{3}(state.source_bodies[1:3, k]))
                (leaf .>> (ell - level)) == nc || continue
                dx, dy, dz = SVector{3}(state.source_bodies[1:3, k]) - center
                rr, ri = HIER_FM._resident_regular_harmonic_coeff(dx, dy, dz, n, m)
                q = state.source_bodies[5, k]
                er += sgn * q * rr
                ei -= sgn * q * ri
            end
            row = HIER_FM.flat_basis_index(n, m, 1)
            @test state.multipoles.phi[row, node] ≈ er atol=2e-12 rtol=2e-12
            @test state.multipoles.phi[row + 1, node] ≈ ei atol=2e-12 rtol=2e-12
        end
    end

    # Full-level and eight-class windows must be numerically invariant.
    a = generate_gravitational(26026, 180)
    b = generate_gravitational(26026, 180)
    direct_ref = generate_gravitational(26026, 180)
    lo, hi = HIER_FM._radix_bounds((a,), Float64)
    h0 = maximum((hi - lo) * 0.5) * 1.05
    eps = _hier_epsilon(4, 12, h0, ell)
    p8 = HierarchicalRigidStencil(4, eps; near_radius2=12, window_classes=8)
    pfull = HierarchicalRigidStencil(4, eps; near_radius2=12, window_classes=1740)
    c8 = RadixFMMCache(a; expansion_order=4, ell, policy=p8)
    cfull = RadixFMMCache(b; expansion_order=4, ell, policy=pfull)
    fmm!(a, c8; scalar_potential=true, gradient=true)
    fmm!(b, cfull; scalar_potential=true, gradient=true)
    direct!(direct_ref; scalar_potential=true, gradient=true)
    @test a.potential[1, :] ≈ b.potential[1, :] atol=2e-12 rtol=2e-12
    @test a.potential[5:7, :] ≈ b.potential[5:7, :] atol=2e-11 rtol=2e-12
    @test maximum(abs.(a.potential[1, :] - direct_ref.potential[1, :])) < 2e-6
    @test maximum(abs.(a.potential[5:7, :] - direct_ref.potential[5:7, :])) < 2e-4

    # Resident allocation and identity contracts under eight-class windowing.
    fmm!(a, c8; scalar_potential=true, gradient=true)
    @test _hier_m2l_allocated(c8.state) <= 64 * 1024
    @test _hier_step_allocated(a, c8) < 512 * 1024
    ctx8 = c8.state.interaction_list
    identities = (objectid(c8.state.route_sources),
        objectid(c8.state.route_targets),
        objectid(c8.state.scratch.m2l_concat.route_class),
        objectid(ctx8.occupancy.node_at))
    update_radix_state!(c8, (a,))
    @test identities == (objectid(c8.state.route_sources),
        objectid(c8.state.route_targets),
        objectid(c8.state.scratch.m2l_concat.route_class),
        objectid(ctx8.occupancy.node_at))
    timing_ids = (objectid(ctx8.update_stage_ns), objectid(ctx8.m2l_level_ns))
    ctx8.profile_stages = true
    update_radix_state!(c8, (a,))
    HIER_FM._launch_host_b2m!(c8.state)
    HIER_FM._launch_host_m2m!(c8.state)
    HIER_FM._launch_host_m2l!(c8.state)
    @test all(>(0), ctx8.update_stage_ns)
    @test all(>(0), ctx8.m2l_level_ns[3:(ell + 1)])
    @test timing_ids ==
        (objectid(ctx8.update_stage_ns), objectid(ctx8.m2l_level_ns))
    ctx8.profile_stages = false
    @test c8.state.counters.expansion_host_copies == 0
    @test c8.state.counters.route_uploads == 0
    @test c8.state.counters.operator_uploads == 0

    # A hierarchical class window is applied whole. Its stage slabs must cover
    # the full route-window capacity rather than the flat strategy's chunk;
    # otherwise windows above the chunk write past the scratch matrices. This is a
    # concat-engine sizing test, so it pins that strategy rather than taking the
    # measured default (which is dense at this order).
    wide_sys = generate_gravitational(26027, 520)
    wlo, whi = HIER_FM._radix_bounds((wide_sys,), Float64)
    wh0 = maximum((whi - wlo) * 0.5) * 1.05
    wide_cache = RadixFMMCache(wide_sys; expansion_order=4, ell=4,
        options=CUDARadixLifecycleOptions(; precision=Float64,
            m2l_strategy=ConcatenatedFixedZM2L()),
        policy=HierarchicalRigidStencil(4,
            _hier_epsilon(4, 12, wh0, 4); near_radius2=12, window_classes=64))
    wide_plan = wide_cache.state.interaction_list.apply_plan
    @test length(wide_cache.state.route_sources) > 32_768
    @test size(wide_plan.aphi, 2) ==
        length(wide_cache.state.route_sources)
    @test size(wide_plan.ops_phi.Cy, 2) ==
        length(wide_cache.state.route_sources)

    # Dense hierarchical storage is offset-only; level dependence is applied by
    # the exact scalar/LH diagonal factors in the active window.
    dense_sys = generate_gravitational(26028, 60)
    dlo, dhi = HIER_FM._radix_bounds((dense_sys,), Float64)
    dh0 = maximum((dhi - dlo) * 0.5) * 1.05
    dense_policy = HierarchicalRigidStencil(4,
        _hier_epsilon(4, 12, dh0, ell); near_radius2=12, window_classes=8)
    dense_cache = RadixFMMCache(dense_sys; expansion_order=4, ell,
        policy=dense_policy,
        options=CUDARadixLifecycleOptions(;
            m2l_strategy=DenseTranslationM2L(apply_chunk=8, build_chunk=8)))
    dense_plan = dense_cache.state.scratch.m2l_concat
    @test length(dense_plan.operators) ==
        length(dense_cache.state.interaction_list.tables.push_offsets)
    @test size(dense_plan.source_scale, 2) ==
        length(dense_cache.state.interaction_list.class_level)
    @test size(dense_plan.target_scale) == size(dense_plan.source_scale)

    # All four resident strategy selections execute the same hierarchical
    # route partition and agree at matched geometry.
    strategy_specs = (
        CUDARadixLifecycleOptions(; m2l_strategy=ConcatenatedFixedZM2L()),
        CUDARadixLifecycleOptions(; operator=FactoredRotationM2L(),
            m2l_strategy=ConcatenatedFixedZM2L()),
        CUDARadixLifecycleOptions(; operator=FactoredRotationM2L(),
            m2l_strategy=PrecomputedFactoredYM2L()),
        CUDARadixLifecycleOptions(; m2l_strategy=DenseTranslationM2L(
            apply_chunk=8, build_chunk=8)),
    )
    strategy_outputs = Matrix{Float64}[]
    for opts in strategy_specs
        ss = generate_gravitational(26030, 50)
        slo, shi = HIER_FM._radix_bounds((ss,), Float64)
        sh0 = maximum((shi - slo) * 0.5) * 1.05
        sc = RadixFMMCache(ss; expansion_order=4, ell,
            policy=_hier_policy(4, 12, sh0, ell, Float64, false),
            options=opts)
        fmm!(ss, sc; scalar_potential=true, gradient=true)
        push!(strategy_outputs, copy(ss.potential))
        # The specialized plans must be the live hierarchical apply plans; the
        # concat engine may not silently stand in for them.
        if opts.m2l_strategy isa PrecomputedFactoredYM2L
            @test sc.state.interaction_list.apply_plan ===
                sc.state.scratch.m2l_concat
            @test sc.state.interaction_list.apply_plan isa
                HIER_FM.ResidentM2LPrecomputedYPlan
        elseif opts.m2l_strategy isa DenseTranslationM2L
            @test sc.state.interaction_list.apply_plan ===
                sc.state.scratch.m2l_concat
            @test sc.state.interaction_list.apply_plan isa
                HIER_FM.ResidentM2LDensePlan
        end
    end
    for output in strategy_outputs[2:end]
        @test output[1, :] ≈ strategy_outputs[1][1, :] atol=3e-12 rtol=3e-12
        @test output[5:7, :] ≈ strategy_outputs[1][5:7, :] atol=3e-11 rtol=3e-12
    end

    # All four strategies must agree on a seeded nonzero chi channel. This
    # exercises the dense asymmetric Lamb-Helmholtz source/target diagonals and
    # the precomputed-y hierarchical window path with nontrivial dual-channel
    # input (gravitational sources alone leave chi identically zero).
    lh_rng = MersenneTwister(0x26b3)
    lh_seed = nothing
    lh_outputs = Tuple{Matrix{Float64},Matrix{Float64}}[]
    for opts in strategy_specs
        ss = generate_gravitational(26033, 50)
        slo, shi = HIER_FM._radix_bounds((ss,), Float64)
        sh0 = maximum((shi - slo) * 0.5) * 1.05
        sc = RadixFMMCache(ss; expansion_order=4, ell, lamb_helmholtz=true,
            policy=_hier_policy(4, 12, sh0, ell, Float64, true),
            options=opts)
        mult = sc.state.multipoles
        if lh_seed === nothing
            _hier_random_physical!(mult, lh_rng)
            lh_seed = (copy(mult.phi), copy(mult.chi))
        else
            mult.phi .= lh_seed[1]
            mult.chi .= lh_seed[2]
        end
        HIER_FM._launch_resident_m2l!(sc.state)
        push!(lh_outputs, (copy(sc.state.locals.phi),
            copy(sc.state.locals.chi)))
    end
    for out in lh_outputs[2:end]
        @test out[1] ≈ lh_outputs[1][1] rtol=1e-8 atol=1e-9
        @test out[2] ≈ lh_outputs[1][2] rtol=1e-8 atol=1e-9
    end

    # Precision/LH surface at the standing P=4 literature order.
    for TF in (Float32, Float64), LH in (false, true)
        sysv = generate_gravitational(26029 + LH, 70)
        refv = generate_gravitational(26029 + LH, 70)
        vlo, vhi = HIER_FM._radix_bounds((sysv,), TF)
        vh0 = maximum((vhi - vlo) * TF(0.5)) * TF(1.05)
        pv = _hier_policy(4, 12, vh0, ell, TF, LH)
        cv = RadixFMMCache(sysv; expansion_order=4, ell,
            lamb_helmholtz=LH, policy=pv,
            options=CUDARadixLifecycleOptions(; precision=TF,
                m2l_strategy=ConcatenatedFixedZM2L()))
        fmm!(sysv, cv; scalar_potential=!LH, gradient=true)
        direct!(refv; scalar_potential=!LH, gradient=true)
        tol = TF === Float32 ? 2e-3 : 2e-4
        @test maximum(abs.(sysv.potential[5:7, :] -
            refv.potential[5:7, :])) < tol
    end

    bad = generate_gravitational(1, 20)
    @test_throws ArgumentError RadixFMMCache(bad; expansion_order=4, ell=3,
        policy=HierarchicalRigidStencil(4, 1e-4))

    # Flat caches at ell <= 1 must still construct and evaluate (the nonleaf
    # level maximum is hierarchical-only and must not run on an empty range).
    tiny = generate_gravitational(26034, 12)
    tiny_ref = generate_gravitational(26034, 12)
    tiny_cache = RadixFMMCache(tiny; expansion_order=4, ell=1)
    fmm!(tiny, tiny_cache; scalar_potential=true, gradient=true)
    direct!(tiny_ref; scalar_potential=true, gradient=true)
    @test tiny.potential[1, :] ≈ tiny_ref.potential[1, :] atol=1e-4 rtol=1e-4

    # Stencil constructor negatives and cache keyword conflicts must throw.
    conflict_sys = generate_gravitational(26035, 20)
    @test_throws ArgumentError HierarchicalRigidStencil(4, 1.0; near_radius2=7)
    @test_throws ArgumentError RigidHierarchicalTables(21)
    @test_throws ArgumentError rigid_stencil_epsilon(4, 0.5, 3, 7)
    @test_throws ArgumentError HierarchicalRigidStencil(4, 1.0; window_classes=0)
    @test_throws ArgumentError HIER_FM._hierarchical_stencil_with_schedule(
        _hier_policy(4, 5, 3), (5, 6))
    @test_throws ArgumentError HIER_FM._hierarchical_stencil_with_schedule(
        _hier_policy(4, 5, 3), (6, 6))
    # The schedule is public on the policy constructor and on the cache; the
    # same three invariants are enforced through either entry point.
    @test HierarchicalRigidStencil(4, 1.0; near_radius2=5,
        level_radii2=(6, 5, 5)).level_radii2 == (6, 5, 5)
    @test_throws ArgumentError HierarchicalRigidStencil(4, 1.0; near_radius2=5,
        level_radii2=(5, 6))            # increasing with depth
    @test_throws ArgumentError HierarchicalRigidStencil(4, 1.0; near_radius2=5,
        level_radii2=(6, 6))            # leaf entry != near_radius2
    @test_throws ArgumentError HierarchicalRigidStencil(4, 1.0; near_radius2=5,
        level_radii2=(6, 7, 5))         # unsupported shell
    @test_throws ArgumentError RadixFMMCache(conflict_sys; expansion_order=4,
        ell=3, policy=_hier_policy(4, 5, 3), level_radii2=(6, 5))
    @test_throws ArgumentError RadixFMMCache(conflict_sys; expansion_order=4,
        ell=3, stencil_epsilon=1e-4, level_radii2=(6, 5))

    # Shipped default geometry (task 028 Stage 7): q = 5 at the leaf with the
    # coarsest M2L level at q = 6, sized to `ell`. An explicit `near_radius2`
    # means the caller asked for a uniform geometry and must get one.
    for ell_default in (2, 3, 5)
        dsys = generate_gravitational(26036, 200)
        dcache = RadixFMMCache(dsys; expansion_order=4, ell=ell_default)
        @test dcache.policy isa HierarchicalRigidStencil
        @test dcache.policy.near_radius2 == HIER_FM.RADIX_DEFAULT_NEAR_RADIUS2
        @test dcache.policy.level_radii2 == (ell_default == 2 ? () :
            (6, ntuple(_ -> 5, ell_default - 2)...))
        @test all(_hier_coverage(dcache) .== 1)
    end
    @test RadixFMMCache(generate_gravitational(26037, 200); expansion_order=4,
        ell=4, near_radius2=12).policy.level_radii2 == ()
    @test_throws ArgumentError RadixFMMCache(conflict_sys; expansion_order=4,
        ell=3, policy=_hier_policy(4, 12, 3), near_radius2=12)
    @test_throws ArgumentError RadixFMMCache(conflict_sys; expansion_order=4,
        ell=3, policy=_hier_policy(4, 12, 3), window_classes=8)
    @test_throws ArgumentError RadixFMMCache(conflict_sys; expansion_order=4,
        ell=3, stencil_epsilon=1e-4, near_radius2=12)
    # An explicit hierarchical policy below the first M2L level must throw.
    @test_throws ArgumentError RadixFMMCache(conflict_sys; expansion_order=4,
        ell=1, policy=_hier_policy(4, 12, 3))

    # The window generator must reject malformed windows and fire its capacity
    # assertion on undersized route storage.
    neg_sys = _hier_system(dense_coords, 3)
    neg_cache = RadixFMMCache(neg_sys; expansion_order=4, ell=3,
        bounds=(SVector(0.0, 0.0, 0.0), 1.0), policy=_hier_policy(4, 12, 3))
    neg_state = neg_cache.state
    neg_ctx = neg_state.interaction_list
    @test_throws ArgumentError HIER_FM.build_hierarchical_routes_window!(
        neg_state.route_levels, neg_state.route_offsets, neg_state.route_targets,
        neg_state.route_sources, nothing, neg_ctx, neg_state.grid, 3, 0, 4)
    @test_throws ArgumentError HIER_FM.build_hierarchical_routes_window!(
        neg_state.route_levels, neg_state.route_offsets, neg_state.route_targets,
        neg_state.route_sources, nothing, neg_ctx, neg_state.grid, 3, 9, 8)
    @test_throws AssertionError HIER_FM.build_hierarchical_routes_window!(
        view(neg_state.route_levels, 1:1), view(neg_state.route_offsets, :, 1:1),
        view(neg_state.route_targets, 1:1), view(neg_state.route_sources, 1:1),
        nothing, neg_ctx, neg_state.grid, 3, 1, 8)
    # Flat launchers must refuse a hierarchical-policy state outright: its
    # route arrays hold only the last generated window.
    @test_throws ArgumentError HIER_FM._launch_resident_m2l_concat!(neg_state)
    @test_throws ArgumentError HIER_FM._launch_resident_m2l_shared!(neg_state)

    # Independent flat-engine oracle for the hierarchical dual-channel path:
    # at ell = 2 with near_radius2 = 12 every parent offset is near, so the
    # hierarchical route partition is exactly the flat leaf partition at
    # matched epsilon and the two engines must agree to roundoff on a seeded
    # nonzero-chi state (the cross-strategy block above only proves the
    # strategies agree with each other).
    oa = generate_gravitational(26036, 60)
    ob = generate_gravitational(26036, 60)
    olo, ohi = HIER_FM._radix_bounds((oa,), Float64)
    oh0 = maximum((ohi - olo) * 0.5) * 1.05
    oprobe = ConstantPStencilConfig(4, 1.0; lamb_helmholtz=true)
    oeps = (constant_p_stencil_bound(oh0, 2, oprobe, SVector(2, 2, 2)) +
        constant_p_stencil_bound(oh0, 2, oprobe, SVector(3, 2, 0))) / 2
    ohier = RadixFMMCache(oa; expansion_order=4, ell=2, lamb_helmholtz=true,
        policy=HierarchicalRigidStencil(ConstantPStencilConfig(4, oeps;
            lamb_helmholtz=true); near_radius2=12, window_classes=8))
    oflat = RadixFMMCache(ob; expansion_order=4, ell=2, lamb_helmholtz=true,
        policy=ConstantPAnalyticStencil(ConstantPStencilConfig(4, oeps;
            lamb_helmholtz=true)))
    _hier_random_physical!(ohier.state.multipoles, MersenneTwister(0x26c))
    # Task 037 stage 3: the hierarchical cache trims level 0, so its node
    # indexing is shifted against the untrimmed flat cache — align by the leaf
    # blocks (the only columns either engine touches at ell = 2, q = 12).
    onc = ohier.state.counts.n_cells
    oh_leaf = ohier.level_offsets[ohier.ell + 1]
    of_leaf = oflat.level_offsets[oflat.ell + 1]
    oflat.state.multipoles.phi[:, of_leaf .+ (1:onc)] .=
        ohier.state.multipoles.phi[:, oh_leaf .+ (1:onc)]
    oflat.state.multipoles.chi[:, of_leaf .+ (1:onc)] .=
        ohier.state.multipoles.chi[:, oh_leaf .+ (1:onc)]
    HIER_FM._launch_resident_m2l!(ohier.state)
    HIER_FM._launch_resident_m2l!(oflat.state)
    @test ohier.state.locals.phi[:, oh_leaf .+ (1:onc)] ≈
        oflat.state.locals.phi[:, of_leaf .+ (1:onc)] rtol=1e-11
    @test ohier.state.locals.chi[:, oh_leaf .+ (1:onc)] ≈
        oflat.state.locals.chi[:, of_leaf .+ (1:onc)] rtol=1e-11

    # Hierarchical-vs-flat end-to-end parity at matched epsilon: the same-P
    # engines differ only through the coarse-level expansion centers, so the
    # gap is bounded by the truncation error on both sides — a far tighter
    # oracle than the direct! gates, and it exercises P = 8 and ell = 4.
    parity_cases = Tuple{DataType,Bool,Int,Int}[]
    for TF in (Float32, Float64), LH in (false, true), P in (4, 8)
        push!(parity_cases, (TF, LH, P, 3))
    end
    push!(parity_cases, (Float64, false, 4, 4))
    for (TF, LH, P, pell) in parity_cases
        pa = generate_gravitational(26040, 80)
        pb = generate_gravitational(26040, 80)
        plo, phi_ = HIER_FM._radix_bounds((pa,), TF)
        ph0 = maximum((phi_ - plo) * TF(0.5)) * TF(1.05)
        pprobe = ConstantPStencilConfig(P, one(TF); lamb_helmholtz=LH)
        peps = (constant_p_stencil_bound(TF(ph0), pell, pprobe, SVector(2, 2, 2)) +
            constant_p_stencil_bound(TF(ph0), pell, pprobe, SVector(3, 2, 0))) / 2
        popts = CUDARadixLifecycleOptions(; precision=TF,
            m2l_strategy=ConcatenatedFixedZM2L())
        ph = RadixFMMCache(pa; expansion_order=P, ell=pell, lamb_helmholtz=LH,
            policy=HierarchicalRigidStencil(ConstantPStencilConfig(P, peps;
                lamb_helmholtz=LH); near_radius2=12, window_classes=8),
            options=popts)
        pf = RadixFMMCache(pb; expansion_order=P, ell=pell, lamb_helmholtz=LH,
            policy=ConstantPAnalyticStencil(ConstantPStencilConfig(P, peps;
                lamb_helmholtz=LH)), options=popts)
        fmm!(pa, ph; scalar_potential=!LH, gradient=true)
        fmm!(pb, pf; scalar_potential=!LH, gradient=true)
        ggap = maximum(abs.(pa.potential[5:7, :] - pb.potential[5:7, :]))
        pgap = LH ? 0.0 :
            maximum(abs.(pa.potential[1, :] - pb.potential[1, :]))
        # Measured 2026-07-31 (Julia 1.12.5 host): P=4 pot <= 5.5e-7 /
        # grad <= 3.2e-5; Float64 P=8 pot 1.5e-9 / grad 4.6e-8; Float32 P=8
        # sits at the roundoff floor. Tolerances carry >= 5x margin.
        gtol = TF === Float32 ? 1e-4 : (P == 8 ? 5e-7 : 2e-4)
        ptol = TF === Float32 ? 5e-6 : (P == 8 ? 1e-8 : 3e-6)
        @test ggap < gtol
        LH || @test pgap < ptol
    end

    # Float32 hierarchical coverage for the specialized strategies (dense and
    # precomputed-y were Float64-only on this path before this block).
    f32_specs = (
        CUDARadixLifecycleOptions(; precision=Float32,
            m2l_strategy=ConcatenatedFixedZM2L()),
        CUDARadixLifecycleOptions(; precision=Float32,
            operator=FactoredRotationM2L(),
            m2l_strategy=PrecomputedFactoredYM2L()),
        CUDARadixLifecycleOptions(; precision=Float32,
            m2l_strategy=DenseTranslationM2L(apply_chunk=8, build_chunk=8)),
    )
    f32_outputs = Matrix{Float64}[]
    for opts in f32_specs
        ss = generate_gravitational(26041, 50)
        slo, shi = HIER_FM._radix_bounds((ss,), Float32)
        sh0 = maximum((shi - slo) * 0.5f0) * 1.05f0
        sc = RadixFMMCache(ss; expansion_order=4, ell,
            policy=_hier_policy(4, 12, sh0, ell, Float32, false),
            options=opts)
        fmm!(ss, sc; scalar_potential=true, gradient=true)
        push!(f32_outputs, copy(ss.potential))
    end
    for output in f32_outputs[2:end]
        # Measured cross-strategy gap 7.5e-9 (2026-07-31); 1e-6 keeps >100x
        # margin while still catching any single-precision scaling error.
        @test output[1, :] ≈ f32_outputs[1][1, :] atol=1e-6 rtol=1e-6
        @test output[5:7, :] ≈ f32_outputs[1][5:7, :] atol=1e-6 rtol=1e-6
    end
end
