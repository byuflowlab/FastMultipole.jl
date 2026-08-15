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
        n_cells::Integer, first_level::Integer=0) where TF
    ell = grid.ell
    length(level_offsets) == ell + 2 ||
        throw(ArgumentError("level_offsets must have length ell + 2"))
    0 <= first_level <= ell ||
        throw(ArgumentError("node-build first_level must lie in 0:ell"))
    # active-level trimming (task 037 stage 3): levels below first_level are
    # never built — their level_offsets prefix stays 0 and nodes at first_level
    # are roots (parent_index 0)
    first_lvl = Int(first_level)
    # count distinct shifted keys per level
    @inbounds for i in 1:(first_lvl + 1)
        level_offsets[i] = 0
    end
    @inbounds for level in first_lvl:ell
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
    @inbounds for level in first_lvl:ell
        shift = 3 * (ell - level)
        width = (2 * grid.h0) / (1 << level)
        node = level_offsets[level + 1]
        # sorted-merge walker into the (level - 1) node block (unused at roots)
        parent_node = level == first_lvl ? 1 : level_offsets[level] + 1
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
            if level == first_lvl
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
        sort_offsets::Vector{Int}, level_offsets::Vector{Int},
        first_level::Integer=0) where TF
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
    _refresh_radix_nodes!(grid, level_offsets, n_cells, first_level)
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

#------- adaptive radix octree construction (Matrix Operator Refactor, task 039) -------#
#
# Host reference implementation of theory/adaptive-radix-octree.md §1 (tree
# construction as sort/scan/compact), §1.4 (Sundar-style 2:1 balance sweep), and
# the §5.2/§5.4 per-cell sigma machinery the list builder consumes
# (per-node subtree sigma_max, population-split veto). The dual-tree U/V/W/X
# list generation itself lives in interaction_list_batched.jl.
#
# Contract (task 023 invariant style): the root cube, depth cap, and every
# capacity are fixed at construction; update_adaptive_tree! rebuilds the whole
# tree in place — keys, sort, top-down split, balance, level-major finalize,
# sigma sweep — with zero allocation, and any capacity violation is a loud
# error, never a silent realloc.

# Theory §6.4 node capacity: population splits are disjoint per level with
# > K_max bodies each (<= ell_max * cld(n, K_max + 1) splits), every node is the
# root or one of <= 8 children of a split, and balance splits are covered by the
# beta_balance allowance. Hard cap: every node is occupied, so each level holds
# at most n nodes.
function _adaptive_node_capacity(policy::AdaptiveTreePolicy, max_n_bodies::Int)
    policy.node_capacity > 0 && return policy.node_capacity
    nsplit = policy.ell_max * cld(max_n_bodies, policy.K_max + 1)
    theory = ceil(Int, policy.beta_balance * (1 + 8 * nsplit))
    hard = (policy.ell_max + 1) * max_n_bodies + 1
    return max(9, min(theory, hard))
end

function _allocate_adaptive_radix_tree(::Type{TF}, x_min::SVector{3,TF}, h0::TF,
        policy::AdaptiveTreePolicy, max_n_bodies::Int) where TF
    h0 > zero(TF) || throw(ArgumentError("adaptive octree requires h0 > 0"))
    node_cap = _adaptive_node_capacity(policy, max_n_bodies)
    node_cap <= typemax(Int32) || throw(ArgumentError(
        "adaptive node capacity $node_cap exceeds the Int32 node-index range"))
    # DFS split stack: pop one node, push <= 8 children; depth <= ell_max
    stack_cap = 8 * (policy.ell_max + 2) + 8
    gate_gmin = Float64(_ball_stencil_min_gap(policy.near_radius2))
    return AdaptiveRadixTree{TF}(
        policy, x_min, h0, max_n_bodies, node_cap, stack_cap, gate_gmin,
        # per-body
        Vector{UInt64}(undef, max_n_bodies),
        Vector{Int}(undef, max_n_bodies), Vector{Int}(undef, max_n_bodies),
        Vector{Int}(undef, max_n_bodies), Vector{Int}(undef, max_n_bodies),
        Vector{Int}(undef, max_n_bodies), zeros(Int, 256), zeros(Int, 256),
        zeros(TF, max_n_bodies),
        # pool
        Vector{Int32}(undef, node_cap), Vector{UInt64}(undef, node_cap),
        Vector{Int}(undef, node_cap), Vector{Int}(undef, node_cap),
        Vector{Int32}(undef, node_cap), Vector{Int32}(undef, node_cap),
        Vector{Int32}(undef, node_cap), Vector{Bool}(undef, node_cap),
        Vector{Int32}(undef, stack_cap),
        # balance + finalize scratch
        Vector{UInt64}(undef, node_cap), Vector{Int32}(undef, node_cap),
        Vector{Int}(undef, node_cap), Vector{Int}(undef, node_cap),
        Vector{UInt64}(undef, node_cap), Vector{Int32}(undef, node_cap),
        fill(false, node_cap), Vector{Int32}(undef, node_cap),
        Vector{Int32}(undef, node_cap), Vector{Int32}(undef, node_cap),
        zeros(Int, policy.ell_max + 2),
        # final level-major table
        Vector{Int32}(undef, node_cap), Vector{UInt64}(undef, node_cap),
        Matrix{Int32}(undef, 3, node_cap), Matrix{TF}(undef, 3, node_cap),
        Vector{Int}(undef, node_cap), Vector{Int}(undef, node_cap),
        Vector{Int32}(undef, node_cap), Matrix{Int32}(undef, 2, node_cap),
        Vector{Int32}(undef, node_cap), zeros(TF, node_cap),
        zeros(Int, policy.ell_max + 2),
        # step state
        0, 0, 0, 0, 0, false, false, 0,
    )
end

"""
    AdaptiveRadixTree(systems; policy=AdaptiveTreePolicy(), max_n_bodies=nothing,
        root=nothing, bounds_margin=0.05, TF=Float64, sigma=nothing)

Standalone host constructor for the task-038 adaptive octree (task 039).
`systems` is a user system or tuple of systems implementing the standard
`get_position`/`get_n_bodies` interface. `root=(x_min, h0)` fixes the root cube
explicitly (lower corner + half-width); otherwise a cube is derived from the
body bounds with `bounds_margin`. `sigma`, when given, is a per-body smoothing
radius vector in **global body-ordinal order** (systems enumerated in order);
combined with `policy.rho_t > 0` it arms the per-cell geometry gate of theory
§5. Capacities derive from `max_n_bodies` (default: the current body count);
[`update_adaptive_tree!`](@ref) then refreshes in place with zero allocation.
"""
function AdaptiveRadixTree(systems; policy::AdaptiveTreePolicy=AdaptiveTreePolicy(),
        max_n_bodies::Union{Nothing,Integer}=nothing, root=nothing,
        bounds_margin::Real=0.05, TF::Type=Float64,
        sigma::Union{Nothing,AbstractVector}=nothing)
    systems_tuple = to_tuple(systems)
    n0 = get_n_bodies(systems_tuple)
    n0 > 0 || throw(ArgumentError("AdaptiveRadixTree requires at least one body"))
    maxn = max_n_bodies === nothing ? n0 : Int(max_n_bodies)
    maxn >= n0 || throw(ArgumentError(
        "max_n_bodies=$maxn is smaller than the current body count $n0"))
    if root === nothing
        x_min_data, x_max_data = _radix_bounds(systems_tuple, TF)
        center = (x_min_data + x_max_data) * TF(0.5)
        box = (x_max_data - x_min_data) * TF(0.5)
        h0 = max(box[1], box[2], box[3]) * (1 + TF(bounds_margin))
        h0 > zero(TF) || (h0 = one(TF))     # single body / degenerate cloud
        x_min = center - SVector{3,TF}(h0, h0, h0)
    else
        x_min = SVector{3,TF}(root[1])
        h0 = TF(root[2])
    end
    tree = _allocate_adaptive_radix_tree(TF, x_min, h0, policy, maxn)
    _update_adaptive_tree!(tree, systems_tuple, sigma)
    tree.built = true
    return tree
end

"""
    update_adaptive_tree!(tree, systems; sigma=nothing)

Rebuild the adaptive octree in place from the systems' current positions using
the tree's **fixed** root cube, depth cap, and capacities: Morton keys at
`ell_max`, LSD radix sort, top-down `K_max` split (with the §5.4 population
split veto when the σ gate is armed), the §1.4 2:1 balance sweep, the
level-major node finalize, and the per-node subtree `sigma_max` sweep. Zero
allocation; capacity violations throw. Returns the tree.
"""
update_adaptive_tree!(tree::AdaptiveRadixTree, systems;
        sigma::Union{Nothing,AbstractVector}=nothing) =
    _update_adaptive_tree!(tree, to_tuple(systems), sigma)

function _update_adaptive_tree!(tree::AdaptiveRadixTree{TF}, systems::Tuple,
        sigma::Union{Nothing,AbstractVector}) where TF
    p = tree.policy
    n = get_n_bodies(systems)
    n > 0 || throw(ArgumentError("update_adaptive_tree! requires at least one body"))
    n <= tree.max_n_bodies || throw(ArgumentError(
        "n=$n exceeds the adaptive tree capacity max_n_bodies=$(tree.max_n_bodies)"))
    _radix_fill_body_data!(tree.body_keys, tree.body_system, tree.body_index,
        systems, tree.x_min, tree.h0, p.ell_max)
    _host_radix_sort_permutation!(tree.perm, tree.sort_scratch, tree.sort_counts,
        tree.sort_offsets, tree.body_keys, n)
    @inbounds for i in 1:n
        tree.invperm[tree.perm[i]] = i
    end
    armed = sigma !== nothing && p.rho_t > 0
    if sigma !== nothing
        length(sigma) >= n || throw(ArgumentError(
            "sigma must supply one value per body (got $(length(sigma)) for n=$n)"))
        if sigma !== tree.body_sigma
            @inbounds for i in 1:n
                tree.body_sigma[i] = TF(sigma[i])
            end
        end
    end
    tree.sigma_armed = armed
    tree.n_bodies = n
    _adaptive_build_pool!(tree, n)
    tree.n_balance_splits = p.balance ? _adaptive_balance!(tree) : 0
    _adaptive_finalize!(tree)
    armed && _adaptive_sigma_sweep!(tree)
    tree.step += 1
    return tree
end

@inline function _adaptive_range_sigma_max(tree::AdaptiveRadixTree{TF}, lo::Int,
        hi::Int) where TF
    m = zero(TF)
    @inbounds for r in lo:hi
        s = tree.body_sigma[tree.perm[r]]
        s > m && (m = s)
    end
    return m
end

# Split pool node `idx` into its occupied children: pigeonhole on the next 3 key
# bits — the sorted-key prefix property makes each child a contiguous range
# (theory §1.2 step 4). Children are appended contiguously in ascending key
# order. Returns the child count.
function _adaptive_split_pool!(tree::AdaptiveRadixTree, idx::Int)
    p = tree.policy
    lev_child = Int(tree.pool_level[idx]) + 1
    lev_child <= p.ell_max || throw(AssertionError(
        "adaptive split requested below the depth cap ell_max=$(p.ell_max)"))
    shift = 3 * (p.ell_max - lev_child)
    lo = tree.pool_lo[idx]
    hi = tree.pool_hi[idx]
    first_child = tree.n_pool + 1
    r = lo
    @inbounds while r <= hi
        c = Int((tree.body_keys[tree.perm[r]] >> shift) & UInt64(0x7))
        r2 = r
        while r2 < hi &&
                Int((tree.body_keys[tree.perm[r2 + 1]] >> shift) & UInt64(0x7)) == c
            r2 += 1
        end
        np = tree.n_pool + 1
        np <= tree.node_capacity || throw(AssertionError(
            "adaptive octree node capacity $(tree.node_capacity) exceeded; " *
            "raise AdaptiveTreePolicy node_capacity (or beta_balance)"))
        tree.n_pool = np
        tree.pool_level[np] = Int32(lev_child)
        tree.pool_key[np] = (tree.pool_key[idx] << 3) | UInt64(c)
        tree.pool_lo[np] = r
        tree.pool_hi[np] = r2
        tree.pool_parent[np] = Int32(idx)
        tree.pool_child_first[np] = Int32(0)
        tree.pool_child_count[np] = Int32(0)
        tree.pool_leaf[np] = true
        r = r2 + 1
    end
    tree.pool_child_first[idx] = Int32(first_child)
    tree.pool_child_count[idx] = Int32(tree.n_pool - first_child + 1)
    tree.pool_leaf[idx] = false
    return tree.n_pool - first_child + 1
end

# Theory §1.2: top-down frontier split while population > K_max below the depth
# cap, realized as an explicit DFS over the pool. The §5.4 veto skips a
# *population* split whose children could not clear the regularization cutoff
# for the cell's own sources (g_min * Delta_{l+1} < rho_t * sigma_max(cell));
# balance splits are exempt and the §5 demotion gate remains the correctness
# backstop.
function _adaptive_build_pool!(tree::AdaptiveRadixTree{TF}, n::Int) where TF
    p = tree.policy
    tree.pool_level[1] = Int32(0)
    tree.pool_key[1] = UInt64(0)
    tree.pool_lo[1] = 1
    tree.pool_hi[1] = n
    tree.pool_parent[1] = Int32(0)
    tree.pool_child_first[1] = Int32(0)
    tree.pool_child_count[1] = Int32(0)
    tree.pool_leaf[1] = true
    tree.n_pool = 1
    stack = tree.split_stack
    sp = 1
    stack[1] = Int32(1)
    veto_active = p.split_veto && tree.sigma_armed
    @inbounds while sp > 0
        idx = Int(stack[sp])
        sp -= 1
        lev = Int(tree.pool_level[idx])
        pop = tree.pool_hi[idx] - tree.pool_lo[idx] + 1
        (pop > p.K_max && lev < p.ell_max) || continue
        if veto_active
            smax = _adaptive_range_sigma_max(tree, tree.pool_lo[idx], tree.pool_hi[idx])
            delta_child = 2 * Float64(tree.h0) / (1 << (lev + 1))
            tree.gate_gmin * delta_child < p.rho_t * Float64(smax) && continue
        end
        first_child = tree.n_pool + 1
        nchild = _adaptive_split_pool!(tree, idx)
        for c in 0:(nchild - 1)
            sp += 1
            sp <= tree.split_stack_capacity ||
                throw(AssertionError("adaptive split stack overflow"))
            stack[sp] = Int32(first_child + c)
        end
    end
    return tree
end

# Last index j in 1:nl with starts[j] <= key (0 when none): the leaf whose
# full-depth key interval could contain `key` — occupied leaves have disjoint
# sorted key intervals, and a (coarser) ancestor of a cell contains the cell's
# whole interval including its start key.
@inline function _adaptive_start_search(starts::Vector{UInt64}, nl::Int, key::UInt64)
    lo = 1
    hi = nl
    ans = 0
    @inbounds while lo <= hi
        mid = (lo + hi) >>> 1
        if starts[mid] <= key
            ans = mid
            lo = mid + 1
        else
            hi = mid - 1
        end
    end
    return ans
end

# Theory §1.4 Sundar-style 2:1 balance sweep. Each round: (1) build the sorted
# table of current leaf full-depth interval starts (the coarse-leaf key
# intervals); (2) deepest-first, every leaf B at level l emits its <= 8 touching
# parent-level (l - 1) cells and matches them by binary search — a matched leaf
# A (level <= l - 2, interval containing the emitted cell) violates 2:1 and is
# flagged; (3) flagged leaves split one level. Rounds repeat to the fixed point
# (splits only deepen leaves, so at most ~ell_max rounds; guarded). A cell
# touches exactly two parent-level cells per axis ([fld(c-1,2), fld(c-1,2)+1]),
# hence 8 emissions per leaf rather than the 26-neighbor form — same set, fewer
# lookups. Balance splits are never vetoed (theory §5.4). Returns the number of
# balance-induced splits.
function _adaptive_balance!(tree::AdaptiveRadixTree)
    p = tree.policy
    ell_max = p.ell_max
    total = 0
    rounds = 0
    changed = true
    @inbounds while changed
        changed = false
        rounds += 1
        rounds <= ell_max + 2 || throw(AssertionError(
            "adaptive 2:1 balance sweep failed to reach a fixed point"))
        for i in 1:tree.n_pool
            tree.balance_mark[i] = false
        end
        nl = 0
        for i in 1:tree.n_pool
            tree.pool_leaf[i] || continue
            nl += 1
            tree.scratch_keys[nl] =
                tree.pool_key[i] << (3 * (ell_max - Int(tree.pool_level[i])))
            tree.scratch_ids[nl] = Int32(i)
        end
        _host_radix_sort_permutation!(tree.scratch_perm, tree.scratch_sort,
            tree.sort_counts, tree.sort_offsets, tree.scratch_keys, nl)
        for j in 1:nl
            tree.leaf_sorted_start[j] = tree.scratch_keys[tree.scratch_perm[j]]
            tree.leaf_sorted_id[j] = tree.scratch_ids[tree.scratch_perm[j]]
        end
        for lev in ell_max:-1:2
            npool_lev = tree.n_pool
            for i in 1:npool_lev
                (tree.pool_leaf[i] && Int(tree.pool_level[i]) == lev) || continue
                coord = morton_decode(tree.pool_key[i], lev)
                Gc = 1 << (lev - 1)
                qx0 = fld(coord[1] - 1, 2)
                qy0 = fld(coord[2] - 1, 2)
                qz0 = fld(coord[3] - 1, 2)
                for dz in 0:1, dy in 0:1, dx in 0:1
                    qx = qx0 + dx
                    qy = qy0 + dy
                    qz = qz0 + dz
                    (0 <= qx < Gc && 0 <= qy < Gc && 0 <= qz < Gc) || continue
                    qstart = morton_key(SVector{3,Int}(qx, qy, qz), lev - 1) <<
                        (3 * (ell_max - (lev - 1)))
                    j = _adaptive_start_search(tree.leaf_sorted_start, nl, qstart)
                    j == 0 && continue
                    a = Int(tree.leaf_sorted_id[j])
                    tree.pool_leaf[a] || continue          # split earlier this round
                    la = Int(tree.pool_level[a])
                    la <= lev - 2 || continue
                    astart = tree.leaf_sorted_start[j]
                    alen = UInt64(1) << (3 * (ell_max - la))
                    qstart < astart + alen || continue     # not an ancestor of Q
                    tree.balance_mark[a] = true
                end
            end
            for i in 1:npool_lev
                (tree.balance_mark[i] && tree.pool_leaf[i]) || continue
                tree.balance_mark[i] = false
                _adaptive_split_pool!(tree, i)
                total += 1
                changed = true
            end
        end
    end
    return total
end

# Counting-sort the pool into the final level-major, Morton-sorted-within-level
# node table (the uniform path's level_offsets convention), then resolve final
# parent/child indices and the compact leaf list. Children of a node are
# contiguous in the next level block (shared key prefix), asserted below.
function _adaptive_finalize!(tree::AdaptiveRadixTree{TF}) where TF
    p = tree.policy
    ell_max = p.ell_max
    off = tree.level_offsets
    cur = tree.level_cursor
    np = tree.n_pool
    @inbounds begin
        fill!(off, 0)
        for i in 1:np
            off[Int(tree.pool_level[i]) + 2] += 1
        end
        for L in 1:(ell_max + 1)
            off[L + 1] += off[L]
        end
        for L in 1:(ell_max + 2)
            cur[L] = off[L]
        end
        for i in 1:np
            L = Int(tree.pool_level[i])
            cur[L + 1] += 1
            tree.pool_by_level[cur[L + 1]] = Int32(i)
        end
        tree.n_nodes = np
        for L in 0:ell_max
            lo = off[L + 1] + 1
            hi = off[L + 2]
            m = hi - lo + 1
            m <= 0 && continue
            for j in 1:m
                tree.scratch_keys[j] = tree.pool_key[Int(tree.pool_by_level[lo + j - 1])]
            end
            _host_radix_sort_permutation!(tree.scratch_perm, tree.scratch_sort,
                tree.sort_counts, tree.sort_offsets, tree.scratch_keys, m)
            width = (2 * tree.h0) / (1 << L)
            for j in 1:m
                f = lo + j - 1
                pid = Int(tree.pool_by_level[lo + tree.scratch_perm[j] - 1])
                tree.final_to_pool[f] = Int32(pid)
                tree.node_of_pool[pid] = Int32(f)
                key = tree.pool_key[pid]
                tree.node_levels[f] = Int32(L)
                tree.node_keys[f] = key
                coord = morton_decode(key, L)
                tree.node_coords[1, f] = Int32(coord[1])
                tree.node_coords[2, f] = Int32(coord[2])
                tree.node_coords[3, f] = Int32(coord[3])
                tree.node_centers[1, f] = tree.x_min[1] + width * (TF(coord[1]) + TF(0.5))
                tree.node_centers[2, f] = tree.x_min[2] + width * (TF(coord[2]) + TF(0.5))
                tree.node_centers[3, f] = tree.x_min[3] + width * (TF(coord[3]) + TF(0.5))
                tree.node_lo[f] = tree.pool_lo[pid]
                tree.node_hi[f] = tree.pool_hi[pid]
            end
        end
        n_nodes = np
        for f in 1:n_nodes
            tree.child_ranges[1, f] = Int32(0)
            tree.child_ranges[2, f] = Int32(0)
            tree.parent_index[f] = Int32(0)
        end
        nleaves = 0
        for f in 1:n_nodes
            pid = Int(tree.final_to_pool[f])
            pp = Int(tree.pool_parent[pid])
            if pp != 0
                par = Int(tree.node_of_pool[pp])
                tree.parent_index[f] = Int32(par)
                if tree.child_ranges[2, par] == 0
                    tree.child_ranges[1, par] = Int32(f)
                    tree.child_ranges[2, par] = Int32(1)
                else
                    Int(tree.child_ranges[1, par]) + Int(tree.child_ranges[2, par]) == f ||
                        throw(AssertionError("adaptive finalize: non-contiguous children"))
                    tree.child_ranges[2, par] += Int32(1)
                end
            end
            if tree.pool_leaf[pid]
                nleaves += 1
                tree.leaf_index[nleaves] = Int32(f)
            end
        end
        tree.n_leaves = nleaves
    end
    return tree
end

# Theory §5.2: per-node subtree sigma_max by one upward sweep. The level-major
# layout puts every child at a larger index than its parent, so a single
# reverse pass suffices.
function _adaptive_sigma_sweep!(tree::AdaptiveRadixTree{TF}) where TF
    @inbounds for f in tree.n_nodes:-1:1
        if tree.child_ranges[2, f] == 0
            tree.node_sigma_max[f] =
                _adaptive_range_sigma_max(tree, tree.node_lo[f], tree.node_hi[f])
        else
            m = zero(TF)
            c0 = Int(tree.child_ranges[1, f])
            for c in c0:(c0 + Int(tree.child_ranges[2, f]) - 1)
                s = tree.node_sigma_max[c]
                s > m && (m = s)
            end
            tree.node_sigma_max[f] = m
        end
    end
    return tree
end

@inline adaptive_is_leaf(tree::AdaptiveRadixTree, node::Integer) =
    tree.child_ranges[2, node] == 0
@inline adaptive_node_range(tree::AdaptiveRadixTree, node::Integer) =
    tree.node_lo[node]:tree.node_hi[node]

# RadixFMMCache integration (opt-in): refresh the adaptive tree and its lists
# after the uniform structures. Function barrier: the cache's adaptive fields
# are `Any`, so the typed inner methods do all the work.
function _adaptive_fill_sigma_from_buffers!(body_sigma::Vector{TF},
        buffers::NTuple{N,Matrix{TF}}, systems::Tuple, sigma_row::Int) where {TF,N}
    i_global = 0
    @inbounds for (isys, system) in enumerate(systems)
        buf = buffers[isys]
        for j in 1:get_n_bodies(system)
            i_global += 1
            body_sigma[i_global] = buf[sigma_row, j]
        end
    end
    return body_sigma
end

function _refresh_adaptive_radix!(cache, systems::Tuple)
    _refresh_adaptive_radix_typed!(cache.adaptive_tree::AdaptiveRadixTree,
        cache.adaptive_lists::AdaptiveInteractionLists, cache.source_buffers,
        systems)
    return cache
end

function _refresh_adaptive_radix_typed!(tree::AdaptiveRadixTree{TF},
        lists::AdaptiveInteractionLists, buffers, systems::Tuple) where TF
    pol = tree.policy
    sig = nothing
    if pol.sigma_row > 0 && pol.rho_t > 0
        _adaptive_fill_sigma_from_buffers!(tree.body_sigma, buffers, systems,
            pol.sigma_row)
        sig = tree.body_sigma
    end
    _update_adaptive_tree!(tree, systems, sig)
    build_adaptive_interaction_lists!(lists, tree)
    return nothing
end
