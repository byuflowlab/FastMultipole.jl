const RADIX_GRID_MAX_ELL = 21

function RadixGrid(system, ell::Integer; TF=numtype(system), h0_fallback=one(TF),
        sort::RadixSortBackend=HostRadixSort())
    TF = promote_type(TF, numtype(system))
    return _radix_grid((system,), ell, TF, h0_fallback, sort)
end

function RadixGrid(systems::Tuple, ell::Integer; TF=get_type(systems),
        h0_fallback=one(TF), sort::RadixSortBackend=HostRadixSort())
    for system in systems
        TF = promote_type(TF, numtype(system))
    end
    return _radix_grid(systems, ell, TF, h0_fallback, sort)
end

radix_grid(args...; kwargs...) = RadixGrid(args...; kwargs...)

function _radix_grid(systems::Tuple, ell::Integer, ::Type{TF}, h0_fallback,
        sort::RadixSortBackend) where TF
    ell < 0 && throw(ArgumentError("RadixGrid depth ell must be nonnegative"))
    ell > RADIX_GRID_MAX_ELL && throw(ArgumentError("RadixGrid depth ell must be <= $RADIX_GRID_MAX_ELL for UInt64 Morton keys"))

    n_bodies = get_n_bodies(systems)
    fallback = TF(h0_fallback)
    fallback > zero(TF) || throw(ArgumentError("h0_fallback must be positive"))

    body_system = Vector{Int}(undef, n_bodies)
    body_index = Vector{Int}(undef, n_bodies)

    if n_bodies == 0
        return RadixGrid{TF}(zero(SVector{3,TF}), fallback, Int(ell), Int[], Int[], UInt64[], Matrix{Int}(undef, 2, 0), body_system, body_index)
    end

    x_min_data, x_max_data = _radix_bounds(systems, TF)
    center = (x_min_data + x_max_data) * TF(0.5)
    box = (x_max_data - x_min_data) * TF(0.5)
    h0 = max(box[1], box[2], box[3])
    h0 = ifelse(h0 > zero(TF), h0, fallback)
    x_min = center - SVector{3,TF}(h0, h0, h0)

    body_keys = Vector{UInt64}(undef, n_bodies)
    _radix_fill_body_data!(body_keys, body_system, body_index, systems, x_min, h0, Int(ell))

    perm = _radix_sort_permutation(body_keys, sort)
    invperm = Vector{Int}(undef, n_bodies)
    @inbounds for i_sorted in eachindex(perm)
        invperm[perm[i_sorted]] = i_sorted
    end

    cell_keys, cell_ranges = _compress_radix_cells(body_keys, perm)

    return RadixGrid{TF}(x_min, h0, Int(ell), perm, invperm, cell_keys, cell_ranges, body_system, body_index)
end

_radix_sort_permutation(body_keys::AbstractVector{UInt64}, ::HostRadixSort) =
    _host_radix_sort_permutation(body_keys)

_radix_sort_permutation(body_keys::AbstractVector{UInt64}, ::AutoRadixSort) =
    _host_radix_sort_permutation(body_keys)

function _radix_sort_permutation(body_keys::AbstractVector{UInt64}, ::DeviceRadixSort)
    throw(ArgumentError(
        "DeviceRadixSort is available only after load_cuda_radix_lifecycle!(); " *
        "use cuda_radix_grid for CUDA-resident construction",
    ))
end

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

function _radix_fill_body_data!(body_keys, body_system, body_index, systems::Tuple, x_min, h0, ell::Int)
    i_global = 0
    @inbounds for (i_system, system) in enumerate(systems)
        for i_body in 1:get_n_bodies(system)
            i_global += 1
            position = get_position(system, i_body)
            coord = radix_cell_coord(x_min, h0, ell, position)
            body_keys[i_global] = morton_key(coord, ell)
            body_system[i_global] = i_system
            body_index[i_global] = i_body
        end
    end
    return nothing
end

function _radix_bounds(systems::Tuple, ::Type{TF}) where TF
    x_min = zero(SVector{3,TF})
    x_max = zero(SVector{3,TF})
    initialized = false
    @inbounds for system in systems
        for i_body in 1:get_n_bodies(system)
            x = TF.(get_position(system, i_body))
            if initialized
                x_min = min.(x_min, x)
                x_max = max.(x_max, x)
            else
                x_min = SVector{3,TF}(x)
                x_max = SVector{3,TF}(x)
                initialized = true
            end
        end
    end
    return x_min, x_max
end

function _radix_bounds(system, ::Type{TF}) where TF
    x = TF.(get_position(system, 1))
    x_min = SVector{3,TF}(x)
    x_max = SVector{3,TF}(x)
    for i_body in 2:get_n_bodies(system)
        x = TF.(get_position(system, i_body))
        x_min = min.(x_min, x)
        x_max = max.(x_max, x)
    end
    return x_min, x_max
end

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

@inline radix_resolution(grid::RadixGrid) = 1 << grid.ell
@inline radix_resolution(grid::DeviceRadixGrid) = 1 << grid.ell
@inline radix_cell_width(grid::RadixGrid) = (2 * grid.h0) / radix_resolution(grid)
@inline radix_cell_width(grid::DeviceRadixGrid) = (2 * grid.h0) / radix_resolution(grid)
@inline radix_cell_half_width(grid::RadixGrid) = grid.h0 / radix_resolution(grid)
@inline radix_cell_half_width(grid::DeviceRadixGrid) = grid.h0 / radix_resolution(grid)
@inline radix_cell_radius(grid::RadixGrid) = radix_cell_half_width(grid) * sqrt(eltype(grid)(3))
@inline radix_cell_radius(grid::DeviceRadixGrid) = radix_cell_half_width(grid) * sqrt(eltype(grid)(3))
@inline Base.eltype(::RadixGrid{TF}) where TF = TF
@inline Base.eltype(::DeviceRadixGrid{TF}) where TF = TF
@inline Base.length(grid::RadixGrid) = length(grid.cell_keys)
@inline Base.length(grid::DeviceRadixGrid) = grid.n_cells
@inline radix_body_system(grid::RadixGrid, global_i::Integer) = grid.body_system[global_i]
@inline radix_body_index(grid::RadixGrid, global_i::Integer) = grid.body_index[global_i]
@inline radix_body_ref(grid::RadixGrid, global_i::Integer) =
    SVector{2,Int}(radix_body_system(grid, global_i), radix_body_index(grid, global_i))

function radix_cell_coord(x_min::SVector{3,TF}, h0::TF, ell::Integer, x) where TF
    G = 1 << Int(ell)
    Δ = (2 * h0) / G
    xi = TF.(x)
    return SVector{3,Int}(
        clamp(floor(Int, (xi[1] - x_min[1]) / Δ), 0, G - 1),
        clamp(floor(Int, (xi[2] - x_min[2]) / Δ), 0, G - 1),
        clamp(floor(Int, (xi[3] - x_min[3]) / Δ), 0, G - 1),
    )
end

@inline radix_cell_coord(grid::RadixGrid, x) = radix_cell_coord(grid.x_min, grid.h0, grid.ell, x)

function morton_key(coord::SVector{3,<:Integer}, ell::Integer)
    ell < 0 && throw(ArgumentError("Morton depth ell must be nonnegative"))
    ell > RADIX_GRID_MAX_ELL && throw(ArgumentError("Morton depth ell must be <= $RADIX_GRID_MAX_ELL for UInt64 keys"))
    key = UInt64(0)
    @inbounds for bit in 0:(Int(ell)-1)
        key |= (UInt64((coord[1] >> bit) & 0x1) << (3 * bit))
        key |= (UInt64((coord[2] >> bit) & 0x1) << (3 * bit + 1))
        key |= (UInt64((coord[3] >> bit) & 0x1) << (3 * bit + 2))
    end
    return key
end

morton_key(i::Integer, j::Integer, k::Integer, ell::Integer) = morton_key(SVector{3,Int}(i, j, k), ell)

function _radix_coord_inbounds(coord::SVector{3,<:Integer}, ell::Integer)
    G = 1 << Int(ell)
    return 0 <= coord[1] < G && 0 <= coord[2] < G && 0 <= coord[3] < G
end

function morton_decode(key::UInt64, ell::Integer)
    ell < 0 && throw(ArgumentError("Morton depth ell must be nonnegative"))
    ell > RADIX_GRID_MAX_ELL && throw(ArgumentError("Morton depth ell must be <= $RADIX_GRID_MAX_ELL for UInt64 keys"))
    i = 0
    j = 0
    k = 0
    for bit in 0:(Int(ell)-1)
        i |= Int((key >> (3 * bit)) & UInt64(0x1)) << bit
        j |= Int((key >> (3 * bit + 1)) & UInt64(0x1)) << bit
        k |= Int((key >> (3 * bit + 2)) & UInt64(0x1)) << bit
    end
    return SVector{3,Int}(i, j, k)
end

@inline radix_cell_coord(grid::RadixGrid, i_cell::Integer) = morton_decode(grid.cell_keys[i_cell], grid.ell)

function radix_cell_center(grid::RadixGrid{TF}, coord::SVector{3,<:Integer}) where TF
    Δ = radix_cell_width(grid)
    return grid.x_min + Δ * (SVector{3,TF}(coord[1], coord[2], coord[3]) + SVector{3,TF}(TF(0.5), TF(0.5), TF(0.5)))
end

@inline radix_cell_center(grid::RadixGrid, i_cell::Integer) = radix_cell_center(grid, radix_cell_coord(grid, i_cell))

function radix_body_range(grid::RadixGrid, i_cell::Integer)
    first = grid.cell_ranges[1, i_cell]
    count = grid.cell_ranges[2, i_cell]
    return first:(first + count - 1)
end

@inline radix_body_indices(grid::RadixGrid, i_cell::Integer) = view(grid.perm, radix_body_range(grid, i_cell))

function radix_cell_index(grid::RadixGrid, key::UInt64)
    i = searchsortedfirst(grid.cell_keys, key)
    return (i <= length(grid.cell_keys) && grid.cell_keys[i] == key) ? i : 0
end

@inline radix_cell_index(grid::RadixGrid, coord::SVector{3,<:Integer}) =
    _radix_coord_inbounds(coord, grid.ell) ? radix_cell_index(grid, morton_key(coord, grid.ell)) : 0

@inline radix_offset(target_coord::SVector{3,<:Integer}, source_coord::SVector{3,<:Integer}) =
    SVector{3,Int}(target_coord[1] - source_coord[1], target_coord[2] - source_coord[2], target_coord[3] - source_coord[3])

@inline radix_offset(grid::RadixGrid, target_cell::Integer, source_cell::Integer) =
    radix_offset(radix_cell_coord(grid, target_cell), radix_cell_coord(grid, source_cell))

@inline radix_displacement(grid::RadixGrid, offset::SVector{3,<:Integer}) =
    radix_cell_width(grid) * SVector{3,eltype(grid)}(offset[1], offset[2], offset[3])

@inline radix_displacement(grid::RadixGrid, target_cell::Integer, source_cell::Integer) =
    radix_displacement(grid, radix_offset(grid, target_cell, source_cell))

#------- in-place radix grid refresh (Matrix Operator Refactor, task 023) -------#
#
# Recurring time steps rebuild the entire grid — body keys, sort permutation, cell
# compression, and level-major node metadata — inside a capacity-sized host
# DeviceRadixGrid without allocating: Vectors are resize!d within their original
# capacity (identity-preserving) and Matrices are prefix-valid up to the current
# counts. The Morton-key domain (x_min, h0, ell) is fixed by the owning
# RadixFMMCache, which is what keeps every per-offset-class geometry table
# step-invariant.

# In-place LSD radix sort of 1:n by body_keys. `perm`/`scratch` must have length
# >= n (resize!d by the caller); `counts`/`offsets` are 256-long scratch. The 8
# byte passes swap src/dst an even number of times, so the result lands in `perm`.
function _host_radix_sort_permutation!(perm::Vector{Int}, scratch::Vector{Int},
        counts::Vector{Int}, offsets::Vector{Int}, body_keys::AbstractVector{UInt64},
        n::Integer)
    @inbounds for i in 1:n
        perm[i] = i
    end
    n <= 1 && return perm
    src = perm
    dst = scratch
    @inbounds for shift in 0:8:56
        fill!(counts, 0)
        for ii in 1:n
            counts[Int((body_keys[src[ii]] >> shift) & UInt64(0xff)) + 1] += 1
        end
        next = 1
        for ibucket in 1:256
            offsets[ibucket] = next
            next += counts[ibucket]
        end
        for ii in 1:n
            i = src[ii]
            ibucket = Int((body_keys[i] >> shift) & UInt64(0xff)) + 1
            dst[offsets[ibucket]] = i
            offsets[ibucket] += 1
        end
        src, dst = dst, src
    end
    return perm
end

# In-place run-length compression of the sorted body keys into occupied cells.
# Returns n_cells; cell_keys is resize!d to it and cell_ranges holds first/count
# in its column prefix.
function _refresh_radix_cells!(cell_keys::Vector{UInt64}, cell_ranges::AbstractMatrix{Int},
        body_keys::AbstractVector{UInt64}, perm::AbstractVector{Int}, n::Integer)
    if n == 0
        resize!(cell_keys, 0)
        return 0
    end
    resize!(cell_keys, size(cell_ranges, 2))
    n_cells = 0
    current_key = body_keys[perm[1]]
    current_first = 1
    current_count = 1
    @inbounds for i_sorted in 2:n
        key = body_keys[perm[i_sorted]]
        if key == current_key
            current_count += 1
        else
            n_cells += 1
            cell_keys[n_cells] = current_key
            cell_ranges[1, n_cells] = current_first
            cell_ranges[2, n_cells] = current_count
            current_key = key
            current_first = i_sorted
            current_count = 1
        end
    end
    n_cells += 1
    cell_keys[n_cells] = current_key
    cell_ranges[1, n_cells] = current_first
    cell_ranges[2, n_cells] = current_count
    resize!(cell_keys, n_cells)
    return n_cells
end

# In-place level-major node metadata rebuild, matching host_resident_radix_grid
# exactly: node keys per level are the sorted distinct shifted leaf keys, parents
# resolve by a sorted merge against the previous level, and each parent's children
# are contiguous in the next level. Returns n_nodes; fills level_offsets
# (level_offsets[L + 2] - level_offsets[L + 1] nodes at level L).
function _refresh_radix_nodes!(grid::DeviceRadixGrid{TF}, level_offsets::Vector{Int},
        n_cells::Integer) where TF
    ell = grid.ell
    length(level_offsets) == ell + 2 ||
        throw(ArgumentError("level_offsets must have length ell + 2"))
    # count distinct shifted keys per level
    level_offsets[1] = 0
    @inbounds for level in 0:ell
        shift = 3 * (ell - level)
        count = 0
        prev = ~UInt64(0)
        first = true
        for cell in 1:n_cells
            key = grid.cell_keys[cell] >> shift
            if first || key != prev
                count += 1
                prev = key
                first = false
            end
        end
        level_offsets[level + 2] = level_offsets[level + 1] + count
    end
    n_nodes = level_offsets[end]
    resize!(grid.node_levels, n_nodes)
    resize!(grid.node_keys, n_nodes)
    resize!(grid.parent_index, n_nodes)
    resize!(grid.leaf_to_node, n_cells)
    @inbounds for node in 1:n_nodes
        grid.child_ranges[1, node] = 0
        grid.child_ranges[2, node] = 0
    end
    # fill node entries level by level; resolve parents by sorted merge
    @inbounds for level in 0:ell
        shift = 3 * (ell - level)
        width = (2 * grid.h0) / (1 << level)
        node = level_offsets[level + 1]
        # sorted-merge walker into the (level - 1) node block (unused at level 0)
        parent_node = level == 0 ? 1 : level_offsets[level] + 1
        prev = ~UInt64(0)
        first = true
        for cell in 1:n_cells
            key = grid.cell_keys[cell] >> shift
            (first || key != prev) || continue
            prev = key
            first = false
            node += 1
            grid.node_levels[node] = level
            grid.node_keys[node] = key
            coord = morton_decode(key, level)
            grid.node_coords[1, node] = coord[1]
            grid.node_coords[2, node] = coord[2]
            grid.node_coords[3, node] = coord[3]
            grid.node_centers[1, node] = grid.x_min[1] + width * (TF(coord[1]) + TF(0.5))
            grid.node_centers[2, node] = grid.x_min[2] + width * (TF(coord[2]) + TF(0.5))
            grid.node_centers[3, node] = grid.x_min[3] + width * (TF(coord[3]) + TF(0.5))
            if level == 0
                grid.parent_index[node] = 0
            else
                parent_key = key >> 3
                while grid.node_keys[parent_node] != parent_key
                    parent_node += 1
                end
                grid.parent_index[node] = parent_node
                if grid.child_ranges[2, parent_node] == 0
                    grid.child_ranges[1, parent_node] = node
                    grid.child_ranges[2, parent_node] = 1
                else
                    grid.child_ranges[2, parent_node] += 1
                end
            end
        end
    end
    @inbounds for cell in 1:n_cells
        grid.leaf_to_node[cell] = level_offsets[ell + 1] + cell
    end
    return n_nodes
end

"""
    update_radix_grid!(grid, systems, body_keys, sort_scratch, sort_counts,
        sort_offsets, level_offsets)

Rebuild a capacity-sized host `DeviceRadixGrid` in place from the systems' current
positions using the grid's **fixed** `x_min`/`h0`/`ell` (task 023). Every array
field keeps its identity; `n_bodies`/`n_cells` are refreshed. Returns the grid.
"""
function update_radix_grid!(grid::DeviceRadixGrid{TF}, systems::Tuple,
        body_keys::Vector{UInt64}, sort_scratch::Vector{Int}, sort_counts::Vector{Int},
        sort_offsets::Vector{Int}, level_offsets::Vector{Int}) where TF
    n = get_n_bodies(systems)
    n > 0 || throw(ArgumentError("update_radix_grid! requires at least one body"))
    resize!(body_keys, n)
    resize!(grid.body_system, n)
    resize!(grid.body_index, n)
    _radix_fill_body_data!(body_keys, grid.body_system, grid.body_index, systems,
        grid.x_min, grid.h0, grid.ell)
    resize!(grid.perm, n)
    resize!(sort_scratch, n)
    _host_radix_sort_permutation!(grid.perm, sort_scratch, sort_counts, sort_offsets,
        body_keys, n)
    resize!(grid.invperm, n)
    @inbounds for i_sorted in 1:n
        grid.invperm[grid.perm[i_sorted]] = i_sorted
    end
    n_cells = _refresh_radix_cells!(grid.cell_keys, grid.cell_ranges, body_keys, grid.perm, n)
    Δ = (2 * grid.h0) / (1 << grid.ell)
    @inbounds for cell in 1:n_cells
        coord = morton_decode(grid.cell_keys[cell], grid.ell)
        grid.cell_centers[1, cell] = grid.x_min[1] + Δ * (TF(coord[1]) + TF(0.5))
        grid.cell_centers[2, cell] = grid.x_min[2] + Δ * (TF(coord[2]) + TF(0.5))
        grid.cell_centers[3, cell] = grid.x_min[3] + Δ * (TF(coord[3]) + TF(0.5))
    end
    _refresh_radix_nodes!(grid, level_offsets, n_cells)
    grid.n_bodies = n
    grid.n_cells = n_cells
    return grid
end

# Capacity-sized host DeviceRadixGrid with fixed Morton domain; contents are
# populated by update_radix_grid!.
function _allocate_host_radix_grid(::Type{TF}, x_min::SVector{3,TF}, h0::TF, ell::Int,
        max_n_bodies::Int, max_cells::Int, max_nodes::Int) where TF
    return DeviceRadixGrid(
        x_min, h0, ell, 0, 0,
        Vector{Int}(undef, max_n_bodies), Vector{Int}(undef, max_n_bodies),
        Vector{UInt64}(undef, max_cells), Matrix{Int}(undef, 2, max_cells),
        Vector{Int}(undef, max_n_bodies), Vector{Int}(undef, max_n_bodies),
        Matrix{TF}(undef, 3, max_cells),
        Vector{Int}(undef, max_nodes), Vector{UInt64}(undef, max_nodes),
        Matrix{Int}(undef, 3, max_nodes), Matrix{TF}(undef, 3, max_nodes),
        Vector{Int}(undef, max_nodes), Matrix{Int}(undef, 2, max_nodes),
        Vector{Int}(undef, max_cells),
    )
end

function RadixLevelOccupancy(ell::Integer; max_bytes::Integer=256 << 20,
        max_dense_ell::Integer=8)
    depth = Int(ell)
    depth >= 0 || throw(ArgumentError("occupancy depth must be nonnegative"))
    # The Morton fallback never reads `level_base`; return it all-zero so no
    # partially built prefix survives an overflow or over-budget bailout.
    sparse_occupancy() = RadixLevelOccupancy(depth, zeros(Int, depth + 2), Int32[])
    depth <= Int(max_dense_ell) || return sparse_occupancy()
    level_base = Vector{Int}(undef, depth + 2)
    level_base[1] = 0
    total = 0
    cells = 1
    for level in 0:depth
        total = try
            Base.checked_add(total, cells)
        catch err
            err isa OverflowError || rethrow()
            return sparse_occupancy()
        end
        level_base[level + 2] = total
        if level < depth
            cells = try
                Base.checked_mul(cells, 8)
            catch err
                err isa OverflowError || rethrow()
                return sparse_occupancy()
            end
        end
    end
    (total <= typemax(Int) ÷ sizeof(Int32) &&
        total * sizeof(Int32) <= Int(max_bytes)) || return sparse_occupancy()
    return RadixLevelOccupancy(depth, level_base, zeros(Int32, total))
end

"""
Refresh the dense per-level node lookup with one zero-fill and one occupied-node
scatter.  The Morton fallback has no refresh work because `grid.node_keys`
already contains sorted level blocks.
"""
function refresh_radix_level_occupancy!(occupancy::RadixLevelOccupancy,
        grid::DeviceRadixGrid, level_offsets::Vector{Int})
    isempty(occupancy.node_at) && return occupancy
    fill!(occupancy.node_at, Int32(0))
    n_nodes = level_offsets[end]
    n_nodes <= typemax(Int32) || throw(ArgumentError(
        "dense hierarchical occupancy requires flat node indices to fit Int32"))
    @inbounds for node in 1:n_nodes
        level = grid.node_levels[node]
        G = 1 << level
        x = grid.node_coords[1, node]
        y = grid.node_coords[2, node]
        z = grid.node_coords[3, node]
        linear = x + G * (y + G * z)
        occupancy.node_at[occupancy.level_base[level + 1] + linear + 1] = Int32(node)
    end
    return occupancy
end
