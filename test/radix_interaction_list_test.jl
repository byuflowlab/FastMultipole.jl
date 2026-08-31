using FastMultipole
using FastMultipole.LinearAlgebra
using FastMultipole.StaticArrays
using Test

function _radix_test_grid(coords::Vector{SVector{3,Int}}, ell::Int; counts=fill(1, length(coords)))
    order = sortperm(coords; by=c -> FastMultipole.morton_key(c, ell))
    sorted_coords = coords[order]
    sorted_counts = counts[order]
    cell_keys = [FastMultipole.morton_key(c, ell) for c in sorted_coords]
    n_bodies = sum(sorted_counts)
    perm = Int[]
    cell_ranges = Matrix{Int}(undef, 2, length(sorted_coords))
    i_body = 1
    for i_cell in eachindex(sorted_coords)
        first = length(perm) + 1
        for _ in 1:sorted_counts[i_cell]
            push!(perm, i_body)
            i_body += 1
        end
        cell_ranges[1, i_cell] = first
        cell_ranges[2, i_cell] = sorted_counts[i_cell]
    end
    invperm = Vector{Int}(undef, n_bodies)
    for i_sorted in eachindex(perm)
        invperm[perm[i_sorted]] = i_sorted
    end
    return RadixGrid{Float64}(
        SVector(0.0, 0.0, 0.0), 0.5, ell, perm, invperm, cell_keys, cell_ranges,
        fill(1, n_bodies), collect(1:n_bodies),
    )
end

_cheb(d) = max(abs(Int(d[1])), abs(Int(d[2])), abs(Int(d[3])))
_level_coord(coord, ell, level) = SVector(coord[1] >> (ell - level), coord[2] >> (ell - level), coord[3] >> (ell - level))

function _m2l_route_set(strategy, policy, grid; kwargs...)
    routes = Set{Tuple{Int,SVector{3,Int},Int,Int}}()
    foreach_radix_m2l_route(strategy, policy, grid; kwargs...) do level, offset, target_cell, source_cell
        push!(routes, (level, offset, target_cell, source_cell))
    end
    return routes
end

function _direct_pair_set(strategy, policy, grid; kwargs...)
    pairs = Set{Tuple{SVector{3,Int},Int,Int}}()
    foreach_radix_direct_pair(strategy, policy, grid; kwargs...) do offset, target_cell, source_cell
        @test FastMultipole.radix_offset(grid, target_cell, source_cell) == offset
        push!(pairs, (offset, target_cell, source_cell))
    end
    return pairs
end

function _expected_parent_neighbor_routes(grid)
    m2l = Set{Tuple{Int,SVector{3,Int},Int,Int}}()
    direct = Set{Tuple{SVector{3,Int},Int,Int}}()
    for target_cell in eachindex(grid.cell_keys), source_cell in eachindex(grid.cell_keys)
        target_leaf = FastMultipole.radix_cell_coord(grid, target_cell)
        source_leaf = FastMultipole.radix_cell_coord(grid, source_cell)
        leaf_offset = target_leaf - source_leaf
        if _cheb(leaf_offset) <= 1
            push!(direct, (leaf_offset, target_cell, source_cell))
            continue
        end
        routed = false
        for level in 1:grid.ell
            target_coord = _level_coord(target_leaf, grid.ell, level)
            source_coord = _level_coord(source_leaf, grid.ell, level)
            offset = target_coord - source_coord
            target_parent = _level_coord(target_leaf, grid.ell, level - 1)
            source_parent = _level_coord(source_leaf, grid.ell, level - 1)
            parent_offset = target_parent - source_parent
            if _cheb(parent_offset) <= 1 && _cheb(offset) > 1
                push!(m2l, (level, offset, target_cell, source_cell))
                routed = true
                break
            end
        end
        @test routed
    end
    return m2l, direct
end

function _expected_constant_p_routes(grid, policy)
    m2l = Set{Tuple{Int,SVector{3,Int},Int,Int}}()
    direct = Set{Tuple{SVector{3,Int},Int,Int}}()
    for target_cell in eachindex(grid.cell_keys), source_cell in eachindex(grid.cell_keys)
        offset = FastMultipole.radix_offset(grid, target_cell, source_cell)
        if constant_p_stencil_accepts(grid, policy.config, offset)
            push!(m2l, (grid.ell, offset, target_cell, source_cell))
        else
            push!(direct, (offset, target_cell, source_cell))
        end
    end
    return m2l, direct
end

function _body_pair_counts(grid, m2l_routes, direct_pairs)
    counts = Dict{Tuple{Int,Int},Int}()
    for (_, _, target_cell, source_cell) in m2l_routes
        for target_body in FastMultipole.radix_body_indices(grid, target_cell)
            for source_body in FastMultipole.radix_body_indices(grid, source_cell)
                counts[(target_body, source_body)] = get(counts, (target_body, source_body), 0) + 1
            end
        end
    end
    for (_, target_cell, source_cell) in direct_pairs
        for target_body in FastMultipole.radix_body_indices(grid, target_cell)
            for source_body in FastMultipole.radix_body_indices(grid, source_cell)
                counts[(target_body, source_body)] = get(counts, (target_body, source_body), 0) + 1
            end
        end
    end
    return counts
end

function _assert_complete_body_coverage(grid, m2l_routes, direct_pairs)
    counts = _body_pair_counts(grid, m2l_routes, direct_pairs)
    n = length(grid.perm)
    @test length(counts) == n * n
    @test all(==(1), values(counts))
end

@testset "radix interaction traversal" begin
    policy = ParentNeighborM2L()

    bound_grid = _radix_test_grid([SVector(0, 0, 0), SVector(7, 7, 7)], 3)
    analytic = ConstantPStencilConfig(3, 1.0, 2.0; normalization=:analytic)
    production = ConstantPStencilConfig(3, 1.0, 2.0; normalization=:production)
    far_offset = SVector(6, 0, 0)
    near_offset = SVector(1, 1, 1)
    raw_bound = constant_p_stencil_bound(
        analytic.P_phi, far_offset, analytic.source_strength,
        FastMultipole.radix_cell_half_width(bound_grid),
    )
    @test constant_p_stencil_bound(bound_grid, analytic, far_offset) ≈ raw_bound
    @test constant_p_stencil_bound(bound_grid, production, far_offset) ≈ raw_bound / (4π)
    @test !constant_p_stencil_accepts(bound_grid, analytic, near_offset)

    lh = ConstantPStencilConfig(3, 1.0, 2.0; chi_strength=3.0, lamb_helmholtz=true)
    B_phi = constant_p_stencil_bound(
        3, far_offset, lh.source_strength, FastMultipole.radix_cell_half_width(bound_grid),
    )
    B_chi = constant_p_stencil_bound(
        4, far_offset, lh.chi_strength, FastMultipole.radix_cell_half_width(bound_grid),
    )
    R = norm(FastMultipole.radix_displacement(bound_grid, far_offset))
    @test constant_p_stencil_bound(bound_grid, lh, far_offset) ≈ B_phi + (1 + 2R) * B_chi

    direct_offsets = FastMultipole._radix_direct_offsets(policy)
    @test length(direct_offsets) == 27
    @test Set(direct_offsets) == Set(SVector(i, j, k) for k in -1:1 for j in -1:1 for i in -1:1)
    @test all(FastMultipole._radix_is_leaf_direct(policy, offset) for offset in direct_offsets)

    for phase in (SVector(i, j, k) for k in 0:1 for j in 0:1 for i in 0:1)
        candidates = FastMultipole._radix_m2l_candidates(policy, phase)
        @test length(candidates) == 189
        @test length(Set(candidates)) == 189
        @test isempty(intersect(Set(candidates), Set(direct_offsets)))
        for child_offset in candidates
            target_child = phase
            source_child = target_child - child_offset
            source_parent = SVector(fld(source_child[1], 2), fld(source_child[2], 2), fld(source_child[3], 2))
            parent_offset = -source_parent
            @test FastMultipole._radix_is_m2l(policy, child_offset, parent_offset)
        end
    end

    strategies = (
        RigidImplicitStencil(),
        SparseOffsetIntersection(),
        BlockedOccupancyBitsets(),
        LazyMaterializedBatches(1),
    )

    target_coord = SVector(8, 8, 8)
    dense_coords = unique!([
        target_coord - offset
        for offset in union(collect(direct_offsets), collect(FastMultipole._radix_m2l_candidates(policy, SVector(0, 0, 0))))
    ])
    dense_grid = _radix_test_grid(dense_coords, 4)
    target_cell = FastMultipole.radix_cell_index(dense_grid, target_coord)

    expected_m2l, expected_direct = _expected_parent_neighbor_routes(dense_grid)
    for strategy in strategies
        m2l = _m2l_route_set(strategy, policy, dense_grid; farfield=true)
        direct = _direct_pair_set(strategy, policy, dense_grid; nearfield=true, self_induced=true)
        @test m2l == expected_m2l
        @test direct == expected_direct
        @test length(filter(p -> p[3] == target_cell && p[1] == dense_grid.ell, m2l)) == 189
        @test length(filter(p -> p[2] == target_cell, direct)) == 27
        _assert_complete_body_coverage(dense_grid, m2l, direct)
    end

    list = build_radix_interaction_list(LazyMaterializedBatches(1), policy, dense_grid)
    listed_m2l = Set{Tuple{Int,SVector{3,Int},Int,Int}}()
    for batch in list.m2l_batches
        @test length(batch.targets) == length(batch.sources)
        for i in eachindex(batch.targets)
            push!(listed_m2l, (batch.level, batch.offset, batch.targets[i], batch.sources[i]))
        end
    end
    @test listed_m2l == expected_m2l
    @test Set((FastMultipole.radix_offset(dense_grid, p[1], p[2]), p[1], p[2]) for p in list.direct_pairs) == expected_direct

    boundary_grid = _radix_test_grid([SVector(0, 0, 0), SVector(1, 0, 0), SVector(15, 15, 15)], 4)
    expected_m2l, expected_direct = _expected_parent_neighbor_routes(boundary_grid)
    for strategy in strategies
        m2l = _m2l_route_set(strategy, policy, boundary_grid; farfield=true)
        direct = _direct_pair_set(strategy, policy, boundary_grid; nearfield=true, self_induced=true)
        @test m2l == expected_m2l
        @test direct == expected_direct
        _assert_complete_body_coverage(boundary_grid, m2l, direct)
        @test any(p[1] == 2 && abs.(p[2]) == SVector(3, 3, 3) for p in m2l)
    end

    sparse_grid = _radix_test_grid([target_coord, target_coord - SVector(2, 0, 0), target_coord - SVector(1, 0, 0)], 4)
    sparse_target = FastMultipole.radix_cell_index(sparse_grid, target_coord)
    sparse_m2l = _m2l_route_set(RigidImplicitStencil(), policy, sparse_grid; farfield=true)
    sparse_direct = _direct_pair_set(RigidImplicitStencil(), policy, sparse_grid; nearfield=true, self_induced=true)
    @test Set(p[2] for p in sparse_m2l if p[3] == sparse_target) == Set([SVector(2, 0, 0)])
    @test Set(p[1] for p in sparse_direct if p[2] == sparse_target) == Set([SVector(0, 0, 0), SVector(1, 0, 0)])
    _assert_complete_body_coverage(sparse_grid, sparse_m2l, sparse_direct)

    multibody_grid = _radix_test_grid(
        [target_coord, target_coord - SVector(2, 0, 0), target_coord - SVector(1, 0, 0)],
        4; counts=[2, 3, 1],
    )
    multibody_m2l = _m2l_route_set(RigidImplicitStencil(), policy, multibody_grid; farfield=true)
    multibody_direct = _direct_pair_set(RigidImplicitStencil(), policy, multibody_grid; nearfield=true, self_induced=true)
    _assert_complete_body_coverage(multibody_grid, multibody_m2l, multibody_direct)

    @test isempty(_m2l_route_set(RigidImplicitStencil(), policy, dense_grid; farfield=false))
    no_near = _direct_pair_set(RigidImplicitStencil(), policy, sparse_grid; nearfield=false, self_induced=true)
    @test Set(p[1] for p in no_near if p[2] == sparse_target) == Set([SVector(0, 0, 0)])
    no_self = _direct_pair_set(RigidImplicitStencil(), policy, sparse_grid; nearfield=true, self_induced=false)
    @test Set(p[1] for p in no_self if p[2] == sparse_target) == Set([SVector(1, 0, 0)])
    @test isempty(_direct_pair_set(
        RigidImplicitStencil(), policy, sparse_grid;
        nearfield=false, self_induced=false,
    ))

    constant_policy = ConstantPAnalyticStencil(ConstantPStencilConfig(3, 1.0e6, 2.0))
    constant_grid = _radix_test_grid([SVector(0, 0, 0), SVector(1, 0, 0), SVector(6, 0, 0), SVector(7, 7, 7)], 3)
    expected_m2l, expected_direct = _expected_constant_p_routes(constant_grid, constant_policy)
    @test !isempty(expected_m2l)
    @test all(p[1] == constant_grid.ell for p in expected_m2l)
    @test all(FastMultipole.radix_offset(constant_grid, p[3], p[4]) == p[2] for p in expected_m2l)
    for strategy in strategies
        m2l = _m2l_route_set(strategy, constant_policy, constant_grid; farfield=true)
        direct = _direct_pair_set(strategy, constant_policy, constant_grid; nearfield=true, self_induced=true)
        @test m2l == expected_m2l
        @test direct == expected_direct
        _assert_complete_body_coverage(constant_grid, m2l, direct)
    end

    # The specialized constant-P builder assembles batches from the implicit stencil
    # (no per-pair enumeration); its output must match the oracle pair sets and keep
    # the canonical (level, z, y, x) batch order with per-batch targets increasing.
    constant_list = build_radix_interaction_list(LazyMaterializedBatches(1), constant_policy, constant_grid)
    tagged_list = build_radix_interaction_list(LazyMaterializedBatches(1),
        constant_policy, constant_grid, RadixRouteSelection())
    @test [(b.level, b.offset, b.targets, b.sources) for b in tagged_list.m2l_batches] ==
        [(b.level, b.offset, b.targets, b.sources) for b in constant_list.m2l_batches]
    @test tagged_list.direct_pairs == constant_list.direct_pairs
    listed_m2l = Set{Tuple{Int,SVector{3,Int},Int,Int}}()
    for batch in constant_list.m2l_batches
        @test length(batch.targets) == length(batch.sources)
        @test issorted(batch.targets)
        for i in eachindex(batch.targets)
            push!(listed_m2l, (batch.level, batch.offset, batch.targets[i], batch.sources[i]))
        end
    end
    @test listed_m2l == expected_m2l
    @test issorted([(b.level, b.offset[3], b.offset[2], b.offset[1]) for b in constant_list.m2l_batches])
    @test Set(
        (FastMultipole.radix_offset(constant_grid, p[1], p[2]), p[1], p[2])
        for p in constant_list.direct_pairs
    ) == expected_direct

    # Implicit-stencil invariants: accepted/rejected partition the bounded box, and the
    # occupancy map resolves exactly the occupied coordinates.
    stencil = FastMultipole.radix_implicit_stencil(constant_grid, constant_policy.config)
    G = FastMultipole.radix_resolution(constant_grid)
    @test length(stencil.accepted_offsets) + length(stencil.rejected_offsets) == (2G - 1)^3
    @test isempty(intersect(Set(stencil.accepted_offsets), Set(stencil.rejected_offsets)))
    @test SVector(0, 0, 0) in stencil.rejected_offsets
    for cell in eachindex(constant_grid.cell_keys)
        coord = FastMultipole.radix_cell_coord(constant_grid, cell)
        @test stencil.cell_at[coord[1] + 1, coord[2] + 1, coord[3] + 1] == cell
    end
    @test count(!=(0), stencil.cell_at) == length(constant_grid.cell_keys)
end
