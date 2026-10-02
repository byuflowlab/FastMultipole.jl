# Test-only reference code for the radix path. Nothing here runs in production:
# a `RadixFMMCache` refreshes a capacity-sized `DeviceRadixGrid` in place and
# builds its routes with `build_radix_routes!` / the hierarchical window
# generator. These are the independent host oracles the tests check the
# production path against:
#
#   * standalone `RadixGrid(system(s), ell)` construction and its accessors;
#   * the classic multi-level ParentNeighborM2L interaction list
#     (`build_radix_interaction_list`);
#   * `host_radix_state`, a complete host `DeviceResidentRadixState` built from
#     that grid and list (the reference the KA lifecycle suite mirrors);
#   * the hard-coded singular direct-pair kernels the functor kernels are
#     checked against.
#
# Include it once per module, guarded:
#     @isdefined(host_radix_state) || include("radix_reference.jl")

using FastMultipole
using FastMultipole.StaticArrays

#------- standalone RadixGrid -------#

function FastMultipole.RadixGrid(system, ell::Integer;
        TF=FastMultipole.numtype(system), h0_fallback=one(TF))
    TF = promote_type(TF, FastMultipole.numtype(system))
    return _radix_grid((system,), ell, TF, h0_fallback)
end

function FastMultipole.RadixGrid(systems::Tuple, ell::Integer;
        TF=FastMultipole.get_type(systems), h0_fallback=one(TF))
    for system in systems
        TF = promote_type(TF, FastMultipole.numtype(system))
    end
    return _radix_grid(systems, ell, TF, h0_fallback)
end

function _radix_grid(systems::Tuple, ell::Integer, ::Type{TF}, h0_fallback) where TF
    ell < 0 && throw(ArgumentError("RadixGrid depth ell must be nonnegative"))
    ell > FastMultipole.RADIX_GRID_MAX_ELL && throw(ArgumentError(
        "RadixGrid depth ell must be <= $(FastMultipole.RADIX_GRID_MAX_ELL) for UInt64 Morton keys"))

    n_bodies = FastMultipole.get_n_bodies(systems)
    fallback = TF(h0_fallback)
    fallback > zero(TF) || throw(ArgumentError("h0_fallback must be positive"))

    body_system = Vector{Int}(undef, n_bodies)
    body_index = Vector{Int}(undef, n_bodies)

    if n_bodies == 0
        return FastMultipole.RadixGrid{TF}(zero(SVector{3,TF}), fallback, Int(ell), Int[], Int[],
            UInt64[], Matrix{Int}(undef, 2, 0), body_system, body_index)
    end

    x_min_data, x_max_data = FastMultipole._radix_bounds(systems, TF)
    center = (x_min_data + x_max_data) * TF(0.5)
    box = (x_max_data - x_min_data) * TF(0.5)
    h0 = max(box[1], box[2], box[3])
    h0 = ifelse(h0 > zero(TF), h0, fallback)
    x_min = center - SVector{3,TF}(h0, h0, h0)

    body_keys = Vector{UInt64}(undef, n_bodies)
    FastMultipole._radix_fill_body_data!(body_keys, body_system, body_index, systems,
        x_min, h0, Int(ell))

    perm = _host_radix_sort_permutation(body_keys)
    invperm = Vector{Int}(undef, n_bodies)
    @inbounds for i_sorted in eachindex(perm)
        invperm[perm[i_sorted]] = i_sorted
    end

    cell_keys, cell_ranges = _compress_radix_cells(body_keys, perm)

    return FastMultipole.RadixGrid{TF}(x_min, h0, Int(ell), perm, invperm, cell_keys,
        cell_ranges, body_system, body_index)
end

# LSD radix sort of 1:n by 64-bit key, 8 byte passes (the allocating host
# oracle for the in-place/device sorts).
function _host_radix_sort_permutation(body_keys::AbstractVector{UInt64})
    n = length(body_keys)
    perm = collect(1:n)
    n <= 1 && return perm

    scratch = similar(perm)
    counts = zeros(Int, 256)
    offsets = Vector{Int}(undef, 256)

    @inbounds for shift in 0:8:56
        fill!(counts, 0)
        for i in perm
            counts[Int((body_keys[i] >> shift) & UInt64(0xff)) + 1] += 1
        end

        next = 1
        for ibucket in 1:256
            offsets[ibucket] = next
            next += counts[ibucket]
        end

        for i in perm
            ibucket = Int((body_keys[i] >> shift) & UInt64(0xff)) + 1
            scratch[offsets[ibucket]] = i
            offsets[ibucket] += 1
        end

        perm, scratch = scratch, perm
    end

    return perm
end

# Run-length compression of the sorted body keys into (cell_keys, first/count).
function _compress_radix_cells(body_keys::AbstractVector{UInt64}, perm::AbstractVector{Int})
    n_bodies = length(perm)
    n_bodies == 0 && return UInt64[], Matrix{Int}(undef, 2, 0)

    cell_keys = UInt64[]
    firsts = Int[]
    counts = Int[]
    current_key = body_keys[perm[1]]
    current_first = 1
    current_count = 1

    for i_sorted in 2:n_bodies
        key = body_keys[perm[i_sorted]]
        if key == current_key
            current_count += 1
        else
            push!(cell_keys, current_key)
            push!(firsts, current_first)
            push!(counts, current_count)
            current_key = key
            current_first = i_sorted
            current_count = 1
        end
    end

    push!(cell_keys, current_key)
    push!(firsts, current_first)
    push!(counts, current_count)

    cell_ranges = Matrix{Int}(undef, 2, length(cell_keys))
    @inbounds for i_cell in eachindex(cell_keys)
        cell_ranges[1, i_cell] = firsts[i_cell]
        cell_ranges[2, i_cell] = counts[i_cell]
    end
    return cell_keys, cell_ranges
end

@inline radix_resolution(grid::FastMultipole.RadixGrid) = 1 << grid.ell
@inline radix_cell_width(grid::FastMultipole.RadixGrid) = (2 * grid.h0) / radix_resolution(grid)
@inline Base.eltype(::FastMultipole.RadixGrid{TF}) where TF = TF
@inline Base.length(grid::FastMultipole.RadixGrid) = length(grid.cell_keys)

@inline FastMultipole.radix_cell_coord(grid::FastMultipole.RadixGrid, x) =
    FastMultipole.radix_cell_coord(grid.x_min, grid.h0, grid.ell, x)
@inline FastMultipole.radix_cell_coord(grid::FastMultipole.RadixGrid, i_cell::Integer) =
    FastMultipole.morton_decode(grid.cell_keys[i_cell], grid.ell)

function radix_cell_center(grid::FastMultipole.RadixGrid{TF}, coord::SVector{3,<:Integer}) where TF
    Δ = radix_cell_width(grid)
    return grid.x_min + Δ * (SVector{3,TF}(coord[1], coord[2], coord[3]) +
        SVector{3,TF}(TF(0.5), TF(0.5), TF(0.5)))
end

@inline radix_cell_center(grid::FastMultipole.RadixGrid, i_cell::Integer) =
    radix_cell_center(grid, FastMultipole.radix_cell_coord(grid, i_cell))

function radix_cell_index(grid::FastMultipole.RadixGrid, key::UInt64)
    i = searchsortedfirst(grid.cell_keys, key)
    return (i <= length(grid.cell_keys) && grid.cell_keys[i] == key) ? i : 0
end

function radix_cell_index(grid::FastMultipole.RadixGrid, coord::SVector{3,<:Integer})
    G = 1 << grid.ell
    (0 <= coord[1] < G && 0 <= coord[2] < G && 0 <= coord[3] < G) || return 0
    return radix_cell_index(grid, FastMultipole.morton_key(coord, grid.ell))
end

#------- ParentNeighborM2L interaction list -------#
#
# The classic multi-level parent-neighbor list over a standalone `RadixGrid`.
# `host_radix_state` builds its reference state from it, and the tests compare
# the KA lifecycle against that state operator by operator (M2M/L2L groups
# included, which a leaf-only constant-P list would leave idle).

@inline _chebyshev_norm(d::SVector{3,<:Integer}) =
    max(abs(Int(d[1])), abs(Int(d[2])), abs(Int(d[3])))

@inline _radix_is_m2l(child_offset::SVector{3,<:Integer}, parent_offset::SVector{3,<:Integer}) =
    _chebyshev_norm(parent_offset) <= 1 && _chebyshev_norm(child_offset) > 1

const _RADIX_CHILD_PHASES = ntuple(i -> SVector{3,Int}(
    (i - 1) & 0x1,
    ((i - 1) >> 1) & 0x1,
    ((i - 1) >> 2) & 0x1,
), 8)

const _RADIX_DIRECT_OFFSETS = Tuple(SVector{3,Int}(i, j, k)
    for k in -1:1 for j in -1:1 for i in -1:1)

function _radix_parent_neighbor_m2l_candidates(target_phase::SVector{3,<:Integer})
    target_child_coord = SVector{3,Int}(target_phase[1], target_phase[2], target_phase[3])
    candidates = SVector{3,Int}[]
    for k_parent in -1:1, j_parent in -1:1, i_parent in -1:1
        parent_offset = SVector{3,Int}(i_parent, j_parent, k_parent)
        for source_phase in _RADIX_CHILD_PHASES
            source_child_coord = 2 * (-parent_offset) + source_phase
            child_offset = target_child_coord - source_child_coord
            if _radix_is_m2l(child_offset, parent_offset)
                push!(candidates, child_offset)
            end
        end
    end
    sort!(candidates; by=o -> (o[3], o[2], o[1]))
    return Tuple(candidates)
end

const _RADIX_PARENT_NEIGHBOR_M2L_CANDIDATES = ntuple(
    i -> _radix_parent_neighbor_m2l_candidates(_RADIX_CHILD_PHASES[i]), 8,
)

@inline _radix_phase_index(phase::SVector{3,<:Integer}) =
    Int(phase[1] + 2 * phase[2] + 4 * phase[3] + 1)

@inline _radix_leaf_phase(coord::SVector{3,<:Integer}) =
    SVector{3,Int}(coord[1] & 0x1, coord[2] & 0x1, coord[3] & 0x1)

@inline _radix_level_coord(leaf_coord::SVector{3,<:Integer}, leaf_level::Integer, level::Integer) =
    SVector{3,Int}(
        Int(leaf_coord[1]) >> (Int(leaf_level) - Int(level)),
        Int(leaf_coord[2]) >> (Int(leaf_level) - Int(level)),
        Int(leaf_coord[3]) >> (Int(leaf_level) - Int(level)),
    )

function _radix_ancestor_leaf_map(grid::FastMultipole.RadixGrid)
    map = Dict{Tuple{Int,SVector{3,Int}},Vector{Int}}()
    for cell in eachindex(grid.cell_keys)
        leaf_coord = FastMultipole.radix_cell_coord(grid, cell)
        for level in 0:grid.ell
            coord = _radix_level_coord(leaf_coord, grid.ell, level)
            push!(get!(() -> Int[], map, (level, coord)), cell)
        end
    end
    return map
end

# Call `f(level, offset, target_cell, source_cell)` for each M2L route.
function _foreach_radix_m2l_route(f, grid::FastMultipole.RadixGrid)
    ancestor_leaf_cells = _radix_ancestor_leaf_map(grid)
    for target_cell in eachindex(grid.cell_keys)
        target_leaf_coord = FastMultipole.radix_cell_coord(grid, target_cell)
        for level in 1:grid.ell
            target_coord = _radix_level_coord(target_leaf_coord, grid.ell, level)
            target_phase = _radix_leaf_phase(target_coord)
            for offset in _RADIX_PARENT_NEIGHBOR_M2L_CANDIDATES[_radix_phase_index(target_phase)]
                source_coord = target_coord - offset
                FastMultipole._radix_coord_inbounds_at_level(source_coord, level) || continue
                source_cells = get(ancestor_leaf_cells, (level, source_coord), nothing)
                source_cells === nothing && continue
                for source_cell in source_cells
                    f(level, offset, target_cell, source_cell)
                end
            end
        end
    end
    return nothing
end

# Call `f(target_cell, source_cell)` for each direct (near or self) leaf pair.
function _foreach_radix_direct_pair(f, grid::FastMultipole.RadixGrid)
    for target_cell in eachindex(grid.cell_keys)
        target_coord = FastMultipole.radix_cell_coord(grid, target_cell)
        for offset in _RADIX_DIRECT_OFFSETS
            source_cell = radix_cell_index(grid, target_coord - offset)
            source_cell == 0 && continue
            f(target_cell, source_cell)
        end
    end
    return nothing
end

"""
    build_radix_interaction_list(policy::ParentNeighborM2L, grid::RadixGrid)

Materialize the multi-level parent-neighbor M2L batches and direct leaf pairs of a
standalone `RadixGrid`, for `host_radix_state`. Batches are sorted by
`(level, z, y, x)` offset.
"""
function build_radix_interaction_list(::FastMultipole.ParentNeighborM2L,
        grid::FastMultipole.RadixGrid)
    batches_by_route = Dict{Tuple{Int,SVector{3,Int}},FastMultipole.RadixM2LBatch{Int}}()
    _foreach_radix_m2l_route(grid) do level, offset, target_cell, source_cell
        batch = get!(() -> FastMultipole.RadixM2LBatch(level, offset, Int[], Int[]),
            batches_by_route, (level, offset))
        push!(batch.targets, target_cell)
        push!(batch.sources, source_cell)
    end
    direct_pairs = SVector{2,Int}[]
    _foreach_radix_direct_pair(grid) do target_cell, source_cell
        push!(direct_pairs, SVector{2,Int}(target_cell, source_cell))
    end
    batches = collect(values(batches_by_route))
    sort!(batches; by=batch -> (batch.level, batch.offset[3], batch.offset[2], batch.offset[1]))
    return FastMultipole.RadixInteractionList{Int}(batches, direct_pairs)
end

#------- host reference state -------#

"""
    host_resident_radix_grid(grid::RadixGrid)

Expand a `RadixGrid` into level-major node metadata in a new host
`DeviceRadixGrid`; the input grid is not modified.
"""
function host_resident_radix_grid(grid::FastMultipole.RadixGrid{TF}) where TF
    level_keys = [sort(unique(key >> (3 * (grid.ell - level)) for key in grid.cell_keys))
                  for level in 0:grid.ell]
    level_offsets = zeros(Int, grid.ell + 2)
    for level in 0:grid.ell
        level_offsets[level + 2] = level_offsets[level + 1] + length(level_keys[level + 1])
    end

    n_nodes = level_offsets[end]
    node_levels = Vector{Int}(undef, n_nodes)
    node_keys = Vector{UInt64}(undef, n_nodes)
    node_coords = Matrix{Int}(undef, 3, n_nodes)
    node_centers = Matrix{TF}(undef, 3, n_nodes)
    index_by_node = Dict{Tuple{Int,UInt64},Int}()

    for level in 0:grid.ell
        width = (2 * grid.h0) / (1 << level)
        for (local_i, key) in pairs(level_keys[level + 1])
            node = level_offsets[level + 1] + local_i
            coord = FastMultipole.morton_decode(key, level)
            node_levels[node] = level
            node_keys[node] = key
            node_coords[:, node] .= coord
            node_centers[:, node] .= grid.x_min + width * (SVector{3,TF}(coord) .+ SVector{3,TF}(0.5, 0.5, 0.5))
            index_by_node[(level, key)] = node
        end
    end

    parent_index = zeros(Int, n_nodes)
    child_ranges = zeros(Int, 2, n_nodes)
    for node in 1:n_nodes
        level = node_levels[node]
        if level > 0
            parent_index[node] = index_by_node[(level - 1, node_keys[node] >> 3)]
        end
    end
    for node in 1:n_nodes
        children = findall(==(node), parent_index)
        if !isempty(children)
            child_ranges[1, node] = first(children)
            child_ranges[2, node] = length(children)
        end
    end

    leaf_to_node = level_offsets[grid.ell + 1] .+ collect(1:length(grid.cell_keys))
    cell_centers = Matrix{TF}(undef, 3, length(grid.cell_keys))
    for i_cell in eachindex(grid.cell_keys)
        cell_centers[:, i_cell] .= radix_cell_center(grid, i_cell)
    end

    return FastMultipole.DeviceRadixGrid(
        grid.x_min, grid.h0, grid.ell, length(grid.perm), length(grid.cell_keys),
        copy(grid.perm), copy(grid.invperm), copy(grid.cell_keys), copy(grid.cell_ranges),
        copy(grid.body_system), copy(grid.body_index), cell_centers,
        node_levels, node_keys, node_coords, node_centers, parent_index, child_ranges,
        leaf_to_node,
    )
end

function _host_radix_tree_routes(grid::FastMultipole.DeviceRadixGrid)
    n_edges = max(length(grid.parent_index) - 1, 0)
    m2m_parent = Vector{Int}(undef, n_edges)
    m2m_child = Vector{Int}(undef, n_edges)
    l2l_parent = Vector{Int}(undef, n_edges)
    l2l_child = Vector{Int}(undef, n_edges)
    @inbounds for edge in 1:n_edges
        node = edge + 1
        parent = grid.parent_index[node]
        m2m_parent[edge] = parent
        m2m_child[edge] = node
        l2l_parent[edge] = parent
        l2l_child[edge] = node
    end
    return m2m_parent, m2m_child, l2l_parent, l2l_child
end

function _host_radix_source_buffers(systems::Tuple, ::Type{TF}) where TF
    return map(systems) do system
        buffer = FastMultipole.allocate_source_buffer(TF, system)
        FastMultipole.source_to_buffer!(buffer, system, 1:FastMultipole.get_n_bodies(system))
        buffer
    end
end

# canonical all-rows packed layout: every source-buffer row is carried,
# including radius row 4; systems narrower than the widest are zero-padded
function _host_radix_body_matrix(grid::FastMultipole.DeviceRadixGrid{TF},
        source_buffers::Tuple) where TF
    nrows = maximum(size(buffer, 1) for buffer in source_buffers)
    body = Matrix{TF}(undef, nrows, length(grid.perm))
    @inbounds for sorted_i in eachindex(grid.perm)
        global_i = grid.perm[sorted_i]
        isys = grid.body_system[global_i]
        ibody = grid.body_index[global_i]
        source = source_buffers[isys]
        nsys = size(source, 1)
        for row in 1:nsys
            body[row, sorted_i] = source[row, ibody]
        end
        for row in (nsys + 1):nrows
            body[row, sorted_i] = zero(TF)
        end
    end
    return body
end

function _host_radix_body_matrix(grid::FastMultipole.DeviceRadixGrid{TF},
        bodies::AbstractMatrix) where TF
    size(bodies, 1) >= 5 ||
        throw(ArgumentError("resident radix bodies must have at least 5 rows: x/y/z/radius/strength"))
    size(bodies, 2) == grid.n_bodies ||
        throw(ArgumentError("resident radix bodies must have one column per grid body"))
    return Matrix{TF}(bodies[:, grid.perm])
end

function _flatten_radix_routes_host(list::FastMultipole.RadixInteractionList,
        grid::FastMultipole.DeviceRadixGrid)
    nroute = sum((length(batch.targets) for batch in list.m2l_batches); init=0)
    levels = Vector{Int}(undef, nroute)
    offsets = Matrix{Int}(undef, 3, nroute)
    targets = Vector{Int}(undef, nroute)
    sources = Vector{Int}(undef, nroute)
    i = 0
    for batch in list.m2l_batches
        for j in eachindex(batch.targets)
            i += 1
            levels[i] = batch.level
            offsets[:, i] .= batch.offset
            targets[i] = grid.leaf_to_node[batch.targets[j]]
            sources[i] = grid.leaf_to_node[batch.sources[j]]
        end
    end
    return levels, offsets, targets, sources
end

function _flatten_radix_direct_pairs_host(list::FastMultipole.RadixInteractionList)
    direct_targets = Vector{Int}(undef, length(list.direct_pairs))
    direct_sources = Vector{Int}(undef, length(list.direct_pairs))
    for (i, pair) in pairs(list.direct_pairs)
        direct_targets[i] = pair[1]
        direct_sources[i] = pair[2]
    end
    return direct_targets, direct_sources
end

"""
    host_radix_state(systems, grid, list, P, lamb_helmholtz=Val(false); options)

Build a host `DeviceResidentRadixState` from a standalone grid and a
ParentNeighborM2L list. `systems` may be a system, tuple of systems, or an
already packed body matrix. `options` must select the `ConcatenatedFixedZM2L`
materialized-y plan (the default). Run it with `run_host_radix_lifecycle!`.
"""
function host_radix_state(systems, grid::FastMultipole.RadixGrid,
        list::FastMultipole.RadixInteractionList, P::Integer,
        lamb_helmholtz::Val{LH}=Val(false);
        options::FastMultipole.RadixLifecycleOptions=FastMultipole.RadixLifecycleOptions()) where LH
    return host_radix_state(systems, host_resident_radix_grid(grid), list, P, lamb_helmholtz;
        options)
end

function host_radix_state(systems, grid::FastMultipole.DeviceRadixGrid,
        list::FastMultipole.RadixInteractionList, P::Integer,
        lamb_helmholtz::Val{LH}=Val(false);
        options::FastMultipole.RadixLifecycleOptions=FastMultipole.RadixLifecycleOptions()) where LH
    TF = options.precision
    basis_info = FastMultipole.OperatorBasisInfo(FastMultipole.CompressedComplexBasis(), P,
        lamb_helmholtz)

    source_bodies = Matrix{TF}(systems isa AbstractMatrix ?
        _host_radix_body_matrix(grid, systems) :
        _host_radix_body_matrix(grid,
            _host_radix_source_buffers(FastMultipole.to_tuple(systems), TF)))

    m2m_parent_routes, m2m_child_routes, l2l_parent_routes, l2l_child_routes =
        _host_radix_tree_routes(grid)
    multipoles = FastMultipole.FlatCoefficientBuffer(TF, basis_info, length(grid.node_keys))
    locals = FastMultipole.FlatCoefficientBuffer(TF, basis_info, length(grid.node_keys))
    output = zeros(TF, 4, grid.n_bodies)

    levels, offsets, targets, sources = _flatten_radix_routes_host(list, grid)
    direct_targets, direct_sources = _flatten_radix_direct_pairs_host(list)
    cache = FastMultipole.OperatorInvariantCache(TF, basis_info)
    scratch = list_resident_workspace(
        TF, basis_info, multipoles, grid, list,
        m2m_parent_routes, m2m_child_routes, l2l_parent_routes, l2l_child_routes,
        grid.node_levels, grid.node_centers, targets, sources;
        m2l_strategy=options.m2l_strategy, operator=options.operator,
    )
    counts = FastMultipole.RadixStepCounts(size(source_bodies, 2), size(grid.cell_ranges, 2),
        size(multipoles.phi, 2), length(targets), length(direct_targets))
    return FastMultipole.DeviceResidentRadixState{TF,FastMultipole.CompressedComplexBasis,LH}(
        grid, list, source_bodies,
        grid.perm, grid.body_system, grid.body_index,
        grid.perm, grid.body_system, grid.body_index, grid.node_levels,
        grid.cell_centers, grid.cell_ranges,
        m2m_parent_routes, m2m_child_routes, l2l_parent_routes, l2l_child_routes,
        multipoles, locals, levels, offsets, targets, sources,
        direct_targets, direct_sources, output,
        cache, scratch, FastMultipole.RadixTransferCounters(), options, counts,
    )
end

#------- list-based resident workspace -------#
#
# Builds the M2M/L2L groups from a concrete grid's node table and one M2L
# geometry class per leaf batch (per route below the leaf), so the workspace
# addresses exactly the nodes `host_radix_state` holds. Backend generic: every
# array type follows `exemplar`, so a device exemplar puts the workspace on the
# device. Production caches build theirs with `_radix_cache_workspace`.

function _degree_major_buffer_like(::Type{TF}, basis_info::FastMultipole.OperatorBasisInfo{B,LH},
        exemplar::FastMultipole.FlatCoefficientBuffer, batch::Integer) where {TF,B,LH}
    phi = similar(exemplar.phi, TF, FastMultipole.degree_major_dof(basis_info.orders.P_phi), batch)
    chi = LH ? similar(exemplar.chi, TF, FastMultipole.degree_major_dof(basis_info.orders.P_active), batch) :
        similar(exemplar.phi, TF, 0, 0)
    fill!(phi, zero(TF))
    fill!(chi, zero(TF))
    return FastMultipole.DegreeMajorRealBuffer{TF,typeof(phi),B,LH}(phi, chi, basis_info)
end

function _collect_level_edges(parent_routes, child_routes, node_levels, node_centers,
        level::Integer, direction::Symbol, ::Type{TF}) where TF
    parents = Int[]
    children = Int[]
    phis = TF[]
    thetas = TF[]
    rs = TF[]
    @inbounds for edge in eachindex(parent_routes)
        parent = parent_routes[edge]
        child = child_routes[edge]
        parent == 0 && continue
        if direction === :parent_to_child
            node_levels[child] == level || continue
            dx = node_centers[1, child] - node_centers[1, parent]
            dy = node_centers[2, child] - node_centers[2, parent]
            dz = node_centers[3, child] - node_centers[3, parent]
        elseif direction === :child_to_parent
            node_levels[parent] == level || continue
            dx = node_centers[1, parent] - node_centers[1, child]
            dy = node_centers[2, parent] - node_centers[2, child]
            dz = node_centers[3, parent] - node_centers[3, child]
        else
            throw(ArgumentError("unknown resident edge direction $direction"))
        end
        r, theta, phi = FastMultipole.cartesian_to_spherical(SVector{3,TF}(dx, dy, dz))
        push!(parents, parent)
        push!(children, child)
        push!(rs, TF(r))
        push!(thetas, TF(theta))
        push!(phis, TF(phi))
    end
    return parents, children, phis, thetas, rs
end

function _resident_m2m_groups(exemplar, ::Type{TF}, basis_info, grid, parent_routes,
        child_routes, node_levels, node_centers) where TF
    groups = FastMultipole.ResidentOperatorGroup[]
    for level in (grid.ell - 1):-1:0
        parents, children, phis, thetas, rs = _collect_level_edges(
            parent_routes, child_routes, node_levels, node_centers, level, :child_to_parent, TF,
        )
        isempty(children) && continue
        for r_group in unique(rs)
            group = findall(==(r_group), rs)
            # target_idx is per-column (one parent per child); the launcher
            # accumulates atomically, so repeated parents need no scatter matrix.
            push!(groups, FastMultipole._resident_group(
                exemplar, TF, basis_info, :m2m, level,
                children[group], parents[group], phis[group], thetas[group], rs[group],
            ))
        end
    end
    return FastMultipole._homogeneous_groups(groups)
end

function _resident_l2l_groups(exemplar, ::Type{TF}, basis_info, grid, parent_routes,
        child_routes, node_levels, node_centers) where TF
    groups = FastMultipole.ResidentOperatorGroup[]
    for level in 1:grid.ell
        parents, children, phis, thetas, rs = _collect_level_edges(
            parent_routes, child_routes, node_levels, node_centers, level, :parent_to_child, TF,
        )
        isempty(children) && continue
        for r_group in unique(rs)
            group = findall(==(r_group), rs)
            push!(groups, FastMultipole._resident_group(
                exemplar, TF, basis_info, :l2l, level,
                parents[group], children[group], phis[group], thetas[group], rs[group],
            ))
        end
    end
    return FastMultipole._homogeneous_groups(groups)
end

# Concatenated M2L plan over a flattened ParentNeighborM2L list. Flattened
# route order is batch-major, and every route of a leaf-level batch shares one
# center displacement, so one (r, θ, φ) class per such batch suffices; batches
# below the leaf pair leaf descendants with varying displacements and get one
# class per route.
function list_m2l_concat_plan(::Type{TF}, basis_info::FastMultipole.OperatorBasisInfo{B,LH},
        exemplar, strategy::FastMultipole.ConcatenatedFixedZM2L,
        invariant::FastMultipole.OperatorInvariantCache,
        list::FastMultipole.RadixInteractionList, leaf_level::Integer,
        host_route_targets, host_route_sources, host_node_centers) where {TF,B,LH}
    FM_ = FastMultipole
    P_phi = basis_info.orders.P_phi
    P_active = basis_info.orders.P_active
    nroutes = length(host_route_targets)
    route_class = Vector{Int32}(undef, nroutes)
    phis = TF[]
    thetas = TF[]
    rs = TF[]
    route_geometry = i -> begin
        t = host_route_targets[i]
        s = host_route_sources[i]
        dx = host_node_centers[1, t] - host_node_centers[1, s]
        dy = host_node_centers[2, t] - host_node_centers[2, s]
        dz = host_node_centers[3, t] - host_node_centers[3, s]
        FM_.cartesian_to_spherical(SVector{3,TF}(dx, dy, dz))
    end
    push_class! = (r, theta, phi) -> begin
        push!(rs, TF(r)); push!(thetas, TF(theta)); push!(phis, TF(phi))
        return Int32(length(rs))
    end
    i = 0
    for batch in list.m2l_batches
        nb = length(batch.targets)
        if batch.level == leaf_level
            cls = push_class!(route_geometry(i + 1)...)
            fill!(view(route_class, (i + 1):(i + nb)), cls)
        else
            for j in 1:nb
                route_class[i + j] = push_class!(route_geometry(i + j)...)
            end
        end
        i += nb
    end
    i == nroutes || throw(ArgumentError(
        "interaction-list batches ($i routes) do not match flattened routes ($nroutes)"))
    chunk = max(min(strategy.chunk, max(nroutes, 1)), 1)
    FM_._check_m2l_range(TF, rs, LH ? P_active : P_phi, "ConcatenatedFixedZM2L")
    ndof_phi = FM_.degree_major_dof(P_phi)
    ndof_chi = LH ? FM_.degree_major_dof(P_active) : 0
    lh_arow_unit, lh_brow_unit = LH ?
        FM_._resident_lh_rows_like(exemplar, TF, P_phi, P_active, one(TF), :local) :
        (nothing, nothing)
    mkphi() = similar(exemplar, TF, ndof_phi, chunk)
    mkchi() = similar(exemplar, TF, ndof_chi, LH ? chunk : 0)
    return FM_.ResidentM2LConcatPlan(
        nroutes, chunk,
        FM_._array_like_vector(exemplar, TF, phis),
        FM_._array_like_vector(exemplar, TF, thetas),
        FM_._array_like_vector(exemplar, TF, rs),
        FM_._array_like_vector(exemplar, TF, inv.(rs)),
        FM_._array_like_vector(exemplar, Int32, route_class),
        similar(exemplar, TF, chunk),
        similar(exemplar, TF, chunk),
        similar(exemplar, TF, chunk),
        similar(exemplar, TF, chunk),
        FM_._degree_row_exponents(exemplar, TF, P_phi),
        LH ? FM_._degree_row_exponents(exemplar, TF, P_active) : nothing,
        FM_.ConcatChannelOps(exemplar, TF, invariant, P_phi, chunk),
        LH ? FM_.ConcatChannelOps(exemplar, TF, invariant, P_active, chunk) : nothing,
        lh_arow_unit, lh_brow_unit,
        mkphi(), mkphi(), mkphi(), mkphi(),
        mkchi(), mkchi(), mkchi(), mkchi(),
        LH ? mkphi() : similar(exemplar, TF, 0, 0), mkchi(),
        LH ? mkphi() : nothing, LH ? mkchi() : nothing,
    )
end

function list_resident_workspace(::Type{TF}, basis_info::FastMultipole.OperatorBasisInfo{B,LH},
        exemplar::FastMultipole.FlatCoefficientBuffer, grid, list, host_m2m_parent_routes,
        host_m2m_child_routes, host_l2l_parent_routes, host_l2l_child_routes,
        host_node_levels, host_node_centers, host_route_targets, host_route_sources;
        m2l_strategy::FastMultipole.AbstractResidentM2LStrategy=FastMultipole.ConcatenatedFixedZM2L(),
        operator::FastMultipole.AbstractM2LOperator=FastMultipole.MaterializedYRotationM2L()) where {TF,B,LH}
    FM_ = FastMultipole
    (m2l_strategy isa FM_.ConcatenatedFixedZM2L && operator isa FM_.MaterializedYRotationM2L) ||
        throw(ArgumentError("list_resident_workspace supports only " *
            "ConcatenatedFixedZM2L with MaterializedYRotationM2L; got " *
            "$(typeof(m2l_strategy)) with $(typeof(operator))"))
    P_phi = basis_info.orders.P_phi
    P_active = basis_info.orders.P_active
    phi_flat_idx = FM_._array_like_vector(exemplar.phi, Int, FM_._degree_major_to_flat_indices(P_phi))
    chi_flat_idx = LH ?
        FM_._array_like_vector(exemplar.chi, Int, FM_._degree_major_to_flat_indices(P_active)) :
        FM_._array_like_vector(exemplar.phi, Int, Int[])
    maps_phi = FM_.DegreeMajorMaps(TF, P_phi, exemplar.phi)
    maps_chi = LH ? FM_.DegreeMajorMaps(TF, P_active, exemplar.chi) : maps_phi
    invariant = FM_.OperatorInvariantCache(TF, basis_info)
    yblocks(modes) = FM_._ymode_real_blocks(
        FM_._array_like_vector(exemplar.phi, Complex{TF}, modes), P_active, TF)
    y_mult_U = yblocks(invariant.y_mult_U)
    y_mult_V = yblocks(invariant.y_mult_V)
    y_loc_U = yblocks(invariant.y_loc_U)
    y_loc_V = yblocks(invariant.y_loc_V)

    m2m_groups = _resident_m2m_groups(
        exemplar.phi, TF, basis_info, grid, host_m2m_parent_routes, host_m2m_child_routes,
        host_node_levels, host_node_centers,
    )
    m2l_concat = list_m2l_concat_plan(
        TF, basis_info, exemplar.phi, m2l_strategy, invariant, list, grid.ell,
        host_route_targets, host_route_sources, host_node_centers,
    )
    l2l_groups = _resident_l2l_groups(
        exemplar.phi, TF, basis_info, grid, host_l2l_parent_routes, host_l2l_child_routes,
        host_node_levels, host_node_centers,
    )
    max_batch = maximum((
        maximum((length(g.source_idx) for g in m2m_groups); init=0),
        maximum((length(g.source_idx) for g in l2l_groups); init=0),
        1,
    ))

    m2l_sources = _degree_major_buffer_like(TF, basis_info, exemplar, max_batch)
    aphi = similar(m2l_sources.phi); yphi = similar(m2l_sources.phi)
    zphi = similar(m2l_sources.phi); rphi = similar(m2l_sources.phi)
    achi = similar(m2l_sources.chi); ychi = similar(m2l_sources.chi)
    zchi = similar(m2l_sources.chi); rchi = similar(m2l_sources.chi)
    cphi = similar(m2l_sources.phi); cchi = similar(m2l_sources.chi)
    ystk_phi = FM_.StackedYChannel(exemplar.phi, TF, invariant, P_phi, max_batch)
    ystk_chi = LH ? FM_.StackedYChannel(exemplar.phi, TF, invariant, P_active, max_batch) :
        nothing
    return FM_.ResidentOperatorWorkspace{TF,B,LH}(
        basis_info, phi_flat_idx, chi_flat_idx, maps_phi, maps_chi,
        y_mult_U, y_mult_V, y_loc_U, y_loc_V,
        aphi, yphi, zphi, rphi, achi, ychi, zchi, rchi, cphi, cchi,
        m2m_groups, l2l_groups, m2l_concat,
        ystk_phi, ystk_chi,
    )
end

#------- hard-coded singular direct-pair kernels -------#
#
# Reference kernels the functor direct-pair path is checked against; production
# dispatch goes through `_host_direct_pairs_functor_kernel!`.

# Singular Biot-Savart direct kernel for Point{Vortex} sources:
# U = -Δx×Γ/(4πr³), J with g→1. No scalar potential is produced.
function _host_direct_pairs_vortex_kernel!(output::AbstractMatrix{TF}, source_bodies,
        cell_ranges, direct_targets, direct_sources, n_direct::Int,
        ::Val{HS}) where {TF,HS}
    c = inv(TF(4) * TF(pi))
    @inbounds for pair_i in 1:n_direct
        target_cell = direct_targets[pair_i]
        source_cell = direct_sources[pair_i]
        tfirst = cell_ranges[1, target_cell]
        tcount = cell_ranges[2, target_cell]
        sfirst = cell_ranges[1, source_cell]
        scount = cell_ranges[2, source_cell]
        for i in tfirst:(tfirst + tcount - 1)
            xi = source_bodies[1, i]
            yi = source_bodies[2, i]
            zi = source_bodies[3, i]
            for j in sfirst:(sfirst + scount - 1)
                i == j && continue
                dx = xi - source_bodies[1, j]
                dy = yi - source_bodies[2, j]
                dz = zi - source_bodies[3, j]
                r2 = dx * dx + dy * dy + dz * dz
                r2 == zero(TF) && continue
                gx = source_bodies[5, j]
                gy = source_bodies[6, j]
                gz = source_bodies[7, j]
                invr = inv(sqrt(r2))
                invr2 = invr * invr
                denom = c * invr * invr2
                output[2, i] += (dz * gy - dy * gz) * denom
                output[3, i] += (dx * gz - dz * gx) * denom
                output[4, i] += (dy * gx - dx * gy) * denom
                if HS
                    denom *= invr2
                    output[5, i] += -3 * dx * (gy * dz - gz * dy) * denom
                    output[6, i] += (-3 * dx * (gz * dx - gx * dz) + gz * r2) * denom
                    output[7, i] += (-3 * dx * (gx * dy - gy * dx) - gy * r2) * denom
                    output[8, i] += (-3 * dy * (gy * dz - gz * dy) - gz * r2) * denom
                    output[9, i] += -3 * dy * (gz * dx - gx * dz) * denom
                    output[10, i] += (-3 * dy * (gx * dy - gy * dx) + gx * r2) * denom
                    output[11, i] += (-3 * dz * (gy * dz - gz * dy) + gy * r2) * denom
                    output[12, i] += (-3 * dz * (gz * dx - gx * dz) - gx * r2) * denom
                    output[13, i] += -3 * dz * (gx * dy - gy * dx) * denom
                end
            end
        end
    end
    return output
end

function _host_direct_pairs_kernel!(output::AbstractMatrix{TF}, source_bodies,
        cell_ranges, direct_targets, direct_sources, n_direct::Int) where TF
    c = inv(TF(4) * TF(pi))
    @inbounds for pair_i in 1:n_direct
        target_cell = direct_targets[pair_i]
        source_cell = direct_sources[pair_i]
        tfirst = cell_ranges[1, target_cell]
        tcount = cell_ranges[2, target_cell]
        sfirst = cell_ranges[1, source_cell]
        scount = cell_ranges[2, source_cell]
        for i in tfirst:(tfirst + tcount - 1)
            xi = source_bodies[1, i]
            yi = source_bodies[2, i]
            zi = source_bodies[3, i]
            for j in sfirst:(sfirst + scount - 1)
                i == j && continue
                dx = xi - source_bodies[1, j]
                dy = yi - source_bodies[2, j]
                dz = zi - source_bodies[3, j]
                r2 = dx * dx + dy * dy + dz * dz
                r2 == zero(TF) && continue
                invr = inv(sqrt(r2))
                q = source_bodies[5, j] * c
                output[1, i] += q * invr
                invr3 = invr * invr * invr
                output[2, i] -= q * dx * invr3
                output[3, i] -= q * dy * invr3
                output[4, i] -= q * dz * invr3
            end
        end
    end
    return output
end

# Singular scalar direct kernel with the 9-component hessian:
# u = qc/r, g = -qc·Δx/r³, H = qc·(3ΔxΔxᵀ/r⁵ - I/r³) — symmetric, so the
# column-major linear order equals the row-major one.
function _host_direct_pairs_hessian_kernel!(output::AbstractMatrix{TF}, source_bodies,
        cell_ranges, direct_targets, direct_sources, n_direct::Int) where TF
    c = inv(TF(4) * TF(pi))
    @inbounds for pair_i in 1:n_direct
        target_cell = direct_targets[pair_i]
        source_cell = direct_sources[pair_i]
        tfirst = cell_ranges[1, target_cell]
        tcount = cell_ranges[2, target_cell]
        sfirst = cell_ranges[1, source_cell]
        scount = cell_ranges[2, source_cell]
        for i in tfirst:(tfirst + tcount - 1)
            xi = source_bodies[1, i]
            yi = source_bodies[2, i]
            zi = source_bodies[3, i]
            for j in sfirst:(sfirst + scount - 1)
                i == j && continue
                dx = xi - source_bodies[1, j]
                dy = yi - source_bodies[2, j]
                dz = zi - source_bodies[3, j]
                r2 = dx * dx + dy * dy + dz * dz
                r2 == zero(TF) && continue
                invr = inv(sqrt(r2))
                q = source_bodies[5, j] * c
                output[1, i] += q * invr
                invr2 = invr * invr
                invr3 = invr * invr2
                output[2, i] -= q * dx * invr3
                output[3, i] -= q * dy * invr3
                output[4, i] -= q * dz * invr3
                q3invr5 = 3 * q * invr3 * invr2
                qinvr3 = q * invr3
                output[5, i] += q3invr5 * dx * dx - qinvr3
                output[6, i] += q3invr5 * dx * dy
                output[7, i] += q3invr5 * dx * dz
                output[8, i] += q3invr5 * dy * dx
                output[9, i] += q3invr5 * dy * dy - qinvr3
                output[10, i] += q3invr5 * dy * dz
                output[11, i] += q3invr5 * dz * dx
                output[12, i] += q3invr5 * dz * dy
                output[13, i] += q3invr5 * dz * dz - qinvr3
            end
        end
    end
    return output
end

#------- policy helper -------#

# `policy` with its per-level radius schedule replaced (same as constructing it
# with the `level_radii2` keyword).
function _hierarchical_stencil_with_schedule(policy::FastMultipole.HierarchicalRigidStencil,
        level_radii2)
    isempty(level_radii2) &&
        throw(ArgumentError("hierarchical level schedule must be nonempty"))
    return FastMultipole.HierarchicalRigidStencil(policy.config;
        policy.near_radius2, level_radii2, policy.window_classes,
        policy.dense_occupancy_max_bytes, policy.dense_occupancy_max_ell)
end
