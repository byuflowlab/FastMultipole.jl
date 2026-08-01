#------- CUDA DEVICE-RESIDENT RADIX LIFECYCLE (Matrix Operator Refactor, task 022) -------#
#
# This file is opt-in via load_cuda_radix_lifecycle!(). It keeps CUDA symbols out of
# the default CPU include path while providing a task-023-ready state object and
# transfer accounting for the resident radix operator path.

const CUDA = Base.require(Base.PkgId(Base.UUID("052768ef-5323-5732-b1bb-66c8b64840ba"), "CUDA"))

# Device intrinsics used bare inside the kernels below; without these bindings
# they are undefined globals in FastMultipole and every kernel infers to Any.
const blockIdx = CUDA.blockIdx
const blockDim = CUDA.blockDim
const threadIdx = CUDA.threadIdx
const gridDim = CUDA.gridDim

function cuda_radix_available()
    try
        return CUDA.functional()
    catch
        return false
    end
end

function cuda_radix_status()
    try
        CUDA.functional() && return "CUDA functional: $(CUDA.name(CUDA.device()))"
        return "CUDA is installed but no functional CUDA device is available"
    catch err
        return "CUDA availability check failed: $(err)"
    end
end

function _require_cuda_radix_available()
    cuda_radix_available() || throw(CUDARadixUnavailable(cuda_radix_status()))
    return nothing
end

function _cuda_flat_buffer(::Type{TF}, basis_info::OperatorBasisInfo{B,LH}, batch::Integer) where {TF,B,LH}
    phi = CUDA.zeros(TF, basis_info.basis_dof_phi, batch)
    chi = LH ? CUDA.zeros(TF, basis_info.basis_dof_chi, batch) : CUDA.zeros(TF, 0, 0)
    return FlatCoefficientBuffer{TF,typeof(phi),B,LH}(phi, chi, basis_info)
end

struct CUDARadixMaterializedScratch
    m2m_targets::Any
    m2m_sources::Any
    m2l_targets::Any
    m2l_sources::Any
    l2l_targets::Any
    l2l_sources::Any
    local_expansion::Any
    phis::Any
    thetas::Any
    rs::Any
end

function CUDARadixMaterializedScratch(::Type{TF}, basis_info::OperatorBasisInfo{B,LH},
        tree_batch::Integer, m2l_batch::Integer) where {TF,B,LH}
    tree_batch = max(Int(tree_batch), 1)
    m2l_batch = max(Int(m2l_batch), 1)
    param_batch = max(tree_batch, m2l_batch)
    return CUDARadixMaterializedScratch(
        _cuda_flat_buffer(TF, basis_info, tree_batch),
        _cuda_flat_buffer(TF, basis_info, tree_batch),
        _cuda_flat_buffer(TF, basis_info, m2l_batch),
        _cuda_flat_buffer(TF, basis_info, m2l_batch),
        _cuda_flat_buffer(TF, basis_info, tree_batch),
        _cuda_flat_buffer(TF, basis_info, tree_batch),
        _cuda_flat_buffer(TF, basis_info, 1),
        CUDA.CuArray{TF}(undef, param_batch),
        CUDA.CuArray{TF}(undef, param_batch),
        CUDA.CuArray{TF}(undef, param_batch),
    )
end

@inline function _cuda_morton_key(ix, iy, iz, ell)
    key = UInt64(0)
    for bit in 0:(ell - 1)
        key |= (UInt64((ix >> bit) & 0x1) << (3 * bit))
        key |= (UInt64((iy >> bit) & 0x1) << (3 * bit + 1))
        key |= (UInt64((iz >> bit) & 0x1) << (3 * bit + 2))
    end
    return key
end

function _cuda_extract_source_positions_kernel!(positions, body_system, body_index,
        source_buffer, offset, isys)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > size(source_buffer, 2) && return nothing
    global_i = offset + i
    @inbounds begin
        positions[1, global_i] = source_buffer[1, i]
        positions[2, global_i] = source_buffer[2, i]
        positions[3, global_i] = source_buffer[3, i]
        body_system[global_i] = isys
        body_index[global_i] = i
    end
    return nothing
end

function _cuda_extract_matrix_positions_kernel!(positions, body_system, body_index, bodies)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > size(bodies, 2) && return nothing
    @inbounds begin
        positions[1, i] = bodies[1, i]
        positions[2, i] = bodies[2, i]
        positions[3, i] = bodies[3, i]
        body_system[i] = 1
        body_index[i] = i
    end
    return nothing
end

function _cuda_radix_keys_kernel!(keys, positions, x_min, h0, ell)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > size(positions, 2) && return nothing
    G = 1 << ell
    delta = (2 * h0) / G
    @inbounds begin
        ix = clamp(floor(Int, (positions[1, i] - x_min[1]) / delta), 0, G - 1)
        iy = clamp(floor(Int, (positions[2, i] - x_min[2]) / delta), 0, G - 1)
        iz = clamp(floor(Int, (positions[3, i] - x_min[3]) / delta), 0, G - 1)
        keys[i] = _cuda_morton_key(ix, iy, iz, ell)
    end
    return nothing
end

# Fixed-box variant for the RadixFMMCache recurring step (task 023): the domain
# is part of the cache's invariant contract, so a body outside it must raise the
# out-of-box flag (checked host-side once per step) instead of being silently
# clamped into an edge cell.
function _cuda_radix_keys_checked_kernel!(keys, oob_flag, positions, x_min, h0, ell)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > length(keys) && return nothing
    G = 1 << ell
    delta = (2 * h0) / G
    two_h0 = 2 * h0
    @inbounds begin
        px = positions[1, i]
        py = positions[2, i]
        pz = positions[3, i]
        if !(x_min[1] <= px <= x_min[1] + two_h0 &&
             x_min[2] <= py <= x_min[2] + two_h0 &&
             x_min[3] <= pz <= x_min[3] + two_h0)
            oob_flag[1] = Int32(1)
        end
        ix = clamp(floor(Int, (px - x_min[1]) / delta), 0, G - 1)
        iy = clamp(floor(Int, (py - x_min[2]) / delta), 0, G - 1)
        iz = clamp(floor(Int, (pz - x_min[3]) / delta), 0, G - 1)
        keys[i] = _cuda_morton_key(ix, iy, iz, ell)
    end
    return nothing
end

function _cuda_gather_sorted_keys_kernel!(sorted_keys, keys, perm)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > length(perm) && return nothing
    @inbounds sorted_keys[i] = keys[perm[i]]
    return nothing
end

# Stage every level's unique-node count into one device vector so a single
# download replaces the per-level blocking scalar reads (task 023).
function _cuda_gather_level_counts_kernel!(level_counts, level_prefix, n_cells)
    l = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    l > length(level_counts) && return nothing
    @inbounds level_counts[l] = n_cells == 0 ? 0 : level_prefix[n_cells, l]
    return nothing
end

function _cuda_fill_invperm_kernel!(invperm, perm)
    sorted_i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    sorted_i > length(perm) && return nothing
    @inbounds invperm[perm[sorted_i]] = sorted_i
    return nothing
end

function _cuda_key_change_flags_kernel!(flags, sorted_keys)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > length(sorted_keys) && return nothing
    @inbounds flags[i] = (i == 1 || sorted_keys[i] != sorted_keys[i - 1]) ? 1 : 0
    return nothing
end

function _cuda_fill_cell_firsts_kernel!(cell_keys, cell_ranges, sorted_keys, flags, prefix)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > length(sorted_keys) && return nothing
    @inbounds if flags[i] == 1
        icell = prefix[i]
        cell_keys[icell] = sorted_keys[i]
        cell_ranges[1, icell] = i
    end
    return nothing
end

function _cuda_fill_cell_counts_kernel!(cell_ranges, flags, prefix)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    n = length(flags)
    i > n && return nothing
    # Cell first indices must be written by a prior kernel launch; the launch
    # boundary provides the synchronization needed before this read.
    @inbounds begin
        if i == n || flags[i + 1] == 1
            icell = prefix[i]
            first = cell_ranges[1, icell]
            cell_ranges[2, icell] = i - first + 1
        end
    end
    return nothing
end

function _cuda_decode_morton_key(key, ell)
    ix = 0
    iy = 0
    iz = 0
    for bit in 0:(ell - 1)
        ix |= Int((key >> (3 * bit)) & UInt64(0x1)) << bit
        iy |= Int((key >> (3 * bit + 1)) & UInt64(0x1)) << bit
        iz |= Int((key >> (3 * bit + 2)) & UInt64(0x1)) << bit
    end
    return ix, iy, iz
end

function _cuda_cell_centers_kernel!(centers, coords, cell_keys, x_min, h0, ell)
    icell = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    icell > length(cell_keys) && return nothing
    G = 1 << ell
    delta = (2 * h0) / G
    @inbounds begin
        ix, iy, iz = _cuda_decode_morton_key(cell_keys[icell], ell)
        coords[1, icell] = ix
        coords[2, icell] = iy
        coords[3, icell] = iz
        centers[1, icell] = x_min[1] + delta * (ix + 0.5)
        centers[2, icell] = x_min[2] + delta * (iy + 0.5)
        centers[3, icell] = x_min[3] + delta * (iz + 0.5)
    end
    return nothing
end

function _cuda_fill_constant_kernel!(dest, value)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > length(dest) && return nothing
    @inbounds dest[i] = value
    return nothing
end

function _cuda_identity_kernel!(dest)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > length(dest) && return nothing
    @inbounds dest[i] = i
    return nothing
end

function _cuda_leaf_ancestor_keys_kernel!(ancestor_keys, cell_keys, leaf_level, level)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > length(cell_keys) && return nothing
    shift = 3 * (leaf_level - level)
    @inbounds ancestor_keys[i] = cell_keys[i] >> shift
    return nothing
end

function _cuda_fill_unique_keys_kernel!(dest, sorted_keys, flags, prefix, offset)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > length(sorted_keys) && return nothing
    @inbounds if flags[i] == 1
        dest[offset + prefix[i]] = sorted_keys[i]
    end
    return nothing
end

function _cuda_fill_node_geometry_kernel!(node_levels, node_coords, node_centers,
        node_keys, level_offsets, x_min, h0, max_level)
    level = (blockIdx().y - 1)
    local_i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    level > max_level && return nothing
    first = level_offsets[level + 1] + 1
    stop = level_offsets[level + 2]
    node = first + local_i - 1
    node > stop && return nothing
    G = 1 << level
    delta = (2 * h0) / G
    @inbounds begin
        key = node_keys[node]
        ix, iy, iz = _cuda_decode_morton_key(key, level)
        node_levels[node] = level
        node_coords[1, node] = ix
        node_coords[2, node] = iy
        node_coords[3, node] = iz
        node_centers[1, node] = x_min[1] + delta * (ix + 0.5)
        node_centers[2, node] = x_min[2] + delta * (iy + 0.5)
        node_centers[3, node] = x_min[3] + delta * (iz + 0.5)
    end
    return nothing
end

@inline function _cuda_lower_bound(keys, first, stop, key)
    lo = first
    hi = stop + 1
    while lo < hi
        mid = (lo + hi) >>> 1
        if keys[mid] < key
            lo = mid + 1
        else
            hi = mid
        end
    end
    return lo
end

@inline function _cuda_upper_bound(keys, first, stop, key)
    lo = first
    hi = stop + 1
    while lo < hi
        mid = (lo + hi) >>> 1
        if keys[mid] <= key
            lo = mid + 1
        else
            hi = mid
        end
    end
    return lo
end

function _cuda_parent_index_kernel!(parent_index, node_keys, level_offsets, max_level)
    level = (blockIdx().y - 1)
    local_i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    level > max_level && return nothing
    first = level_offsets[level + 1] + 1
    stop = level_offsets[level + 2]
    node = first + local_i - 1
    node > stop && return nothing
    @inbounds begin
        if level == 0
            parent_index[node] = 0
        else
            parent_key = node_keys[node] >> 3
            parent_first = level_offsets[level] + 1
            parent_stop = level_offsets[level + 1]
            parent = _cuda_lower_bound(node_keys, parent_first, parent_stop, parent_key)
            parent_index[node] = (parent <= parent_stop && node_keys[parent] == parent_key) ? parent : 0
        end
    end
    return nothing
end

function _cuda_child_ranges_kernel!(child_ranges, node_keys, level_offsets, max_level)
    level = (blockIdx().y - 1)
    local_i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    level > max_level && return nothing
    first = level_offsets[level + 1] + 1
    stop = level_offsets[level + 2]
    node = first + local_i - 1
    node > stop && return nothing
    @inbounds begin
        if level == max_level
            child_ranges[1, node] = 0
            child_ranges[2, node] = 0
        else
            child_first = level_offsets[level + 2] + 1
            child_stop = level_offsets[level + 3]
            lo_key = node_keys[node] << 3
            hi_key = lo_key + UInt64(7)
            lo = _cuda_lower_bound(node_keys, child_first, child_stop, lo_key)
            hi = _cuda_upper_bound(node_keys, child_first, child_stop, hi_key)
            count = hi - lo
            child_ranges[1, node] = count > 0 ? lo : 0
            child_ranges[2, node] = count
        end
    end
    return nothing
end

function _cuda_fill_leaf_to_node_kernel!(leaf_to_node, leaf_offset)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > length(leaf_to_node) && return nothing
    @inbounds leaf_to_node[i] = leaf_offset + i
    return nothing
end

function _cuda_tree_routes_kernel!(m2m_parent, m2m_child, l2l_parent, l2l_child,
        parent_index)
    edge = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    node = edge + 1
    node > length(parent_index) && return nothing
    @inbounds begin
        parent = parent_index[node]
        m2m_parent[edge] = parent
        m2m_child[edge] = node
        l2l_parent[edge] = parent
        l2l_child[edge] = node
    end
    return nothing
end

function _cuda_collect_source_positions(source_buffers::Tuple, ::Type{TF},
        counters::CUDARadixTransferCounters) where TF
    n = sum(size(buffer, 2) for buffer in source_buffers)
    positions = CUDA.CuArray{TF}(undef, 3, n)
    body_system = CUDA.CuArray{Int}(undef, n)
    body_index = CUDA.CuArray{Int}(undef, n)
    threads = 128
    offset = 0
    for isys in eachindex(source_buffers)
        nb = size(source_buffers[isys], 2)
        blocks = cld(nb, threads)
        if blocks > 0
            CUDA.@cuda threads=threads blocks=blocks _cuda_extract_source_positions_kernel!(
                positions, body_system, body_index, source_buffers[isys], offset, isys,
            )
        end
        offset += nb
    end
    return positions, body_system, body_index
end

function _cuda_collect_matrix_positions(bodies, ::Type{TF}) where TF
    n = size(bodies, 2)
    positions = CUDA.CuArray{TF}(undef, 3, n)
    body_system = CUDA.CuArray{Int}(undef, n)
    body_index = CUDA.CuArray{Int}(undef, n)
    threads = 128
    blocks = cld(n, threads)
    if blocks > 0
        CUDA.@cuda threads=threads blocks=blocks _cuda_extract_matrix_positions_kernel!(
            positions, body_system, body_index, bodies,
        )
    end
    return positions, body_system, body_index
end

function _cuda_root_domain(positions, ::Type{TF}, h0_fallback, bounds) where TF
    if bounds !== nothing
        x_min_data, x_max_data = bounds
        x_min_data = SVector{3,TF}(TF.(x_min_data))
        x_max_data = SVector{3,TF}(TF.(x_max_data))
    else
        mins = vec(Array(minimum(positions; dims=2)))
        maxs = vec(Array(maximum(positions; dims=2)))
        x_min_data = SVector{3,TF}(mins[1], mins[2], mins[3])
        x_max_data = SVector{3,TF}(maxs[1], maxs[2], maxs[3])
    end
    center = (x_min_data + x_max_data) * TF(0.5)
    box = (x_max_data - x_min_data) * TF(0.5)
    h0 = max(box[1], box[2], box[3])
    fallback = TF(h0_fallback)
    fallback > zero(TF) || throw(ArgumentError("h0_fallback must be positive"))
    h0 = ifelse(h0 > zero(TF), h0, fallback)
    x_min = center - SVector{3,TF}(h0, h0, h0)
    return x_min, h0
end

function _cuda_unique_level_keys(cell_keys, ell::Int, level::Int, threads::Int)
    n_cells = length(cell_keys)
    ancestor_keys = CUDA.CuArray{UInt64}(undef, n_cells)
    blocks = cld(n_cells, threads)
    if blocks > 0
        CUDA.@cuda threads=threads blocks=blocks _cuda_leaf_ancestor_keys_kernel!(
            ancestor_keys, cell_keys, ell, level,
        )
    end
    sorted_keys = CUDA.sort(ancestor_keys)
    flags = CUDA.CuArray{Int}(undef, n_cells)
    if blocks > 0
        CUDA.@cuda threads=threads blocks=blocks _cuda_key_change_flags_kernel!(flags, sorted_keys)
    end
    prefix = accumulate(+, flags)
    n_unique = n_cells == 0 ? 0 : Int(Array(prefix[n_cells:n_cells])[1])
    unique_keys = CUDA.CuArray{UInt64}(undef, n_unique)
    if blocks > 0
        CUDA.@cuda threads=threads blocks=blocks _cuda_fill_unique_keys_kernel!(
            unique_keys, sorted_keys, flags, prefix, 0,
        )
    end
    CUDA.unsafe_free!(ancestor_keys)
    CUDA.unsafe_free!(sorted_keys)
    CUDA.unsafe_free!(flags)
    CUDA.unsafe_free!(prefix)
    return unique_keys, n_unique
end

function _cuda_radix_node_metadata(cell_keys, x_min::SVector{3,TF}, h0::TF,
        ell::Int, threads::Int) where TF
    n_cells = length(cell_keys)
    level_keys = Vector{Any}(undef, ell + 1)
    counts = Vector{Int}(undef, ell + 1)
    for level in 0:ell
        unique_keys, n_unique = _cuda_unique_level_keys(cell_keys, ell, level, threads)
        level_keys[level + 1] = unique_keys
        counts[level + 1] = n_unique
    end

    level_offsets_host = zeros(Int, ell + 2)
    for level in 0:ell
        level_offsets_host[level + 2] = level_offsets_host[level + 1] + counts[level + 1]
    end
    n_nodes = level_offsets_host[end]
    level_offsets = CUDA.CuArray(level_offsets_host)

    node_keys = CUDA.CuArray{UInt64}(undef, n_nodes)
    for level in 0:ell
        count = counts[level + 1]
        if count > 0
            first = level_offsets_host[level + 1]
            copyto!(node_keys, first + 1, level_keys[level + 1], 1, count)
        end
        CUDA.unsafe_free!(level_keys[level + 1])
    end

    node_levels = CUDA.CuArray{Int}(undef, n_nodes)
    node_coords = CUDA.CuArray{Int}(undef, 3, n_nodes)
    node_centers = CUDA.CuArray{TF}(undef, 3, n_nodes)
    parent_index = CUDA.CuArray{Int}(undef, n_nodes)
    child_ranges = CUDA.CuArray{Int}(undef, 2, n_nodes)
    max_count = maximum(counts; init=0)
    if max_count > 0
        blocks_x = cld(max_count, threads)
        CUDA.@cuda threads=threads blocks=(blocks_x, ell + 1) _cuda_fill_node_geometry_kernel!(
            node_levels, node_coords, node_centers, node_keys, level_offsets, x_min, h0, ell,
        )
        CUDA.@cuda threads=threads blocks=(blocks_x, ell + 1) _cuda_parent_index_kernel!(
            parent_index, node_keys, level_offsets, ell,
        )
        CUDA.@cuda threads=threads blocks=(blocks_x, ell + 1) _cuda_child_ranges_kernel!(
            child_ranges, node_keys, level_offsets, ell,
        )
    end

    leaf_to_node = CUDA.CuArray{Int}(undef, n_cells)
    blocks_cells = cld(n_cells, threads)
    if blocks_cells > 0
        CUDA.@cuda threads=threads blocks=blocks_cells _cuda_fill_leaf_to_node_kernel!(
            leaf_to_node, level_offsets_host[ell + 1],
        )
    end
    return node_levels, node_keys, node_coords, node_centers, parent_index, child_ranges,
        leaf_to_node, level_offsets_host
end

function _cuda_radix_grid_from_positions(positions, body_system, body_index,
        ell::Integer, ::Type{TF}; h0_fallback=one(TF), bounds=nothing,
        domain=nothing) where TF
    ell < 0 && throw(ArgumentError("RadixGrid depth ell must be nonnegative"))
    ell > RADIX_GRID_MAX_ELL && throw(ArgumentError("RadixGrid depth ell must be <= $RADIX_GRID_MAX_ELL for UInt64 Morton keys"))
    n = size(positions, 2)

    if n == 0
        empty_int = CUDA.CuArray{Int}(undef, 0)
        empty_key = CUDA.CuArray{UInt64}(undef, 0)
        empty_ranges = CUDA.CuArray{Int}(undef, 2, 0)
        empty_centers = CUDA.CuArray{TF}(undef, 3, 0)
        return DeviceRadixGrid(
            zero(SVector{3,TF}), TF(h0_fallback), Int(ell), 0, 0,
            empty_int, empty_int, empty_key, empty_ranges, empty_int, empty_int,
            empty_centers, empty_int, empty_key, empty_ranges, empty_centers, empty_int,
            empty_ranges, empty_int,
        ), zeros(Int, Int(ell) + 2)
    end

    # `domain=(x_min, h0)` bypasses data-derived bounds so a fixed-box cache (task
    # 023) reproduces the exact Morton domain of its invariant contract.
    if domain !== nothing
        x_min = SVector{3,TF}(TF.(domain[1]))
        h0 = TF(domain[2])
    else
        x_min, h0 = _cuda_root_domain(positions, TF, h0_fallback, bounds)
    end
    keys = CUDA.CuArray{UInt64}(undef, n)
    threads = 128
    blocks = cld(n, threads)
    CUDA.@cuda threads=threads blocks=blocks _cuda_radix_keys_kernel!(
        keys, positions, x_min, h0, Int(ell),
    )

    # Rely on the default CUDA sortperm path to preserve deterministic ordering
    # for equal Morton keys, keeping same-cell bodies in global body order.
    perm = CUDA.sortperm(keys)
    sorted_keys = keys[perm]
    CUDA.unsafe_free!(keys)
    invperm = CUDA.CuArray{Int}(undef, n)
    CUDA.@cuda threads=threads blocks=blocks _cuda_fill_invperm_kernel!(invperm, perm)

    flags = CUDA.CuArray{Int}(undef, n)
    CUDA.@cuda threads=threads blocks=blocks _cuda_key_change_flags_kernel!(flags, sorted_keys)
    prefix = accumulate(+, flags)
    n_cells = Int(Array(prefix[n:n])[1])
    cell_keys = CUDA.CuArray{UInt64}(undef, n_cells)
    cell_ranges = CUDA.CuArray{Int}(undef, 2, n_cells)
    cell_blocks = cld(n, threads)
    CUDA.@cuda threads=threads blocks=cell_blocks _cuda_fill_cell_firsts_kernel!(
        cell_keys, cell_ranges, sorted_keys, flags, prefix,
    )
    CUDA.@cuda threads=threads blocks=cell_blocks _cuda_fill_cell_counts_kernel!(
        cell_ranges, flags, prefix,
    )
    CUDA.unsafe_free!(sorted_keys)
    CUDA.unsafe_free!(flags)
    CUDA.unsafe_free!(prefix)

    centers = CUDA.CuArray{TF}(undef, 3, n_cells)
    coords = CUDA.CuArray{Int}(undef, 3, n_cells)
    blocks_cells = cld(n_cells, threads)
    if blocks_cells > 0
        CUDA.@cuda threads=threads blocks=blocks_cells _cuda_cell_centers_kernel!(
            centers, coords, cell_keys, x_min, h0, Int(ell),
        )
    end
    CUDA.unsafe_free!(coords)

    node_levels, node_keys, node_coords, node_centers, parent_index, child_ranges,
        leaf_to_node, level_offsets_host =
        _cuda_radix_node_metadata(cell_keys, x_min, h0, Int(ell), threads)

    return DeviceRadixGrid(
        x_min, h0, Int(ell), n, n_cells,
        perm, invperm, cell_keys, cell_ranges, body_system, body_index, centers,
        node_levels, node_keys, node_coords, node_centers, parent_index, child_ranges, leaf_to_node,
    ), level_offsets_host
end

function cuda_radix_grid(systems, ell::Integer; sort::RadixSortBackend=DeviceRadixSort(),
        options::CUDARadixLifecycleOptions=CUDARadixLifecycleOptions(),
        h0_fallback=one(options.precision), bounds=nothing)
    _require_cuda_radix_available()
    sort isa HostRadixSort && return RadixGrid(systems, ell; h0_fallback=h0_fallback, sort=sort)
    if sort isa AutoRadixSort && get_n_bodies(systems) < sort.min_device_bodies
        return RadixGrid(systems, ell; h0_fallback=h0_fallback, sort=HostRadixSort())
    end
    sort isa Union{DeviceRadixSort,AutoRadixSort} ||
        throw(ArgumentError("unsupported CUDA radix sort backend $(typeof(sort))"))
    TF = options.precision
    counters = CUDARadixTransferCounters()
    source_buffers = _canonical_cuda_source_buffers(to_tuple(systems), TF, counters)
    positions, body_system, body_index = _cuda_collect_source_positions(source_buffers, TF, counters)
    grid, _ = _cuda_radix_grid_from_positions(
        positions, body_system, body_index, ell, TF;
        h0_fallback=h0_fallback, bounds=bounds,
    )
    return grid
end

function cuda_radix_grid(bodies::CUDA.AnyCuArray, ell::Integer;
        sort::RadixSortBackend=DeviceRadixSort(),
        options::CUDARadixLifecycleOptions=CUDARadixLifecycleOptions(),
        h0_fallback=one(options.precision), bounds=nothing)
    _require_cuda_radix_available()
    sort isa HostRadixSort && return RadixGrid(Array(bodies), ell; h0_fallback=h0_fallback, sort=sort)
    if sort isa AutoRadixSort && size(bodies, 2) < sort.min_device_bodies
        return RadixGrid(Array(bodies), ell; h0_fallback=h0_fallback, sort=HostRadixSort())
    end
    TF = options.precision
    device_bodies = eltype(bodies) === TF ? bodies : CUDA.CuArray{TF}(bodies)
    positions, body_system, body_index = _cuda_collect_matrix_positions(device_bodies, TF)
    grid, _ = _cuda_radix_grid_from_positions(
        positions, body_system, body_index, ell, TF;
        h0_fallback=h0_fallback, bounds=bounds,
    )
    return grid
end

_to_cuda_array(x, ::Type{TF}, counters::CUDARadixTransferCounters, field::Symbol) where TF =
    _to_cuda_array(x, TF, counters, Val(field))

function _to_cuda_array(x::CUDA.AnyCuArray, ::Type{TF}, counters::CUDARadixTransferCounters, ::Val{field}) where {TF,field}
    return eltype(x) === TF ? x : CUDA.CuArray{TF}(x)
end

function _to_cuda_array(x::CUDA.AnyCuArray, ::Type{TF}, counters::CUDARadixTransferCounters, ::Val{:body}) where TF
    return eltype(x) === TF ? x : CUDA.CuArray{TF}(x)
end

function _to_cuda_array(x::CUDA.AnyCuArray, ::Type{TF}, counters::CUDARadixTransferCounters, ::Val{:route}) where TF
    return x
end

function _to_cuda_array(x, ::Type{TF}, counters::CUDARadixTransferCounters, ::Val{:body}) where TF
    counters.body_uploads += 1
    return CUDA.CuArray{TF}(x)
end

function _to_cuda_array(x, ::Type{TF}, counters::CUDARadixTransferCounters, ::Val{:route}) where TF
    counters.route_uploads += 1
    return CUDA.CuArray(x)
end

function _flatten_radix_routes(list::RadixInteractionList)
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
            targets[i] = batch.targets[j]
            sources[i] = batch.sources[j]
        end
    end
    return levels, offsets, targets, sources
end

function _max_radix_m2l_batch_width(list::RadixInteractionList)
    return maximum((length(batch.targets) for batch in list.m2l_batches); init=0)
end

function _flatten_radix_node_routes(list::RadixInteractionList, grid::DeviceRadixGrid, ::Type{TF},
        counters::CUDARadixTransferCounters) where TF
    levels, offsets, targets, sources = _flatten_radix_routes(list)
    route_levels = _to_cuda_array(levels, TF, counters, :route)
    route_offsets = _to_cuda_array(offsets, TF, counters, :route)
    target_cells = _to_cuda_array(targets, TF, counters, :route)
    source_cells = _to_cuda_array(sources, TF, counters, :route)
    route_targets = grid.leaf_to_node[target_cells]
    route_sources = grid.leaf_to_node[source_cells]
    return route_levels, route_offsets, route_targets, route_sources
end

function _flatten_radix_direct_pairs(list::RadixInteractionList, ::Type{TF},
        counters::CUDARadixTransferCounters) where TF
    direct_targets = Vector{Int}(undef, length(list.direct_pairs))
    direct_sources = Vector{Int}(undef, length(list.direct_pairs))
    for (i, pair) in pairs(list.direct_pairs)
        direct_targets[i] = pair[1]
        direct_sources[i] = pair[2]
    end
    return (
        _to_cuda_array(direct_targets, TF, counters, :route),
        _to_cuda_array(direct_sources, TF, counters, :route),
    )
end

function _radix_cell_geometry(grid::RadixGrid{TF}) where TF
    centers = Matrix{TF}(undef, 3, length(grid.cell_keys))
    ranges = Matrix{Int}(undef, 2, length(grid.cell_keys))
    for i_cell in eachindex(grid.cell_keys)
        centers[:, i_cell] .= radix_cell_center(grid, i_cell)
        ranges[:, i_cell] .= grid.cell_ranges[:, i_cell]
    end
    return centers, ranges
end

function _empty_radix_tree_routes()
    route = Matrix{Int}(undef, 2, 0)
    return route, route, route, route
end

function _cuda_radix_tree_routes(grid::DeviceRadixGrid)
    n_edges = max(length(grid.parent_index) - 1, 0)
    m2m_parent = CUDA.CuArray{Int}(undef, n_edges)
    m2m_child = CUDA.CuArray{Int}(undef, n_edges)
    l2l_parent = CUDA.CuArray{Int}(undef, n_edges)
    l2l_child = CUDA.CuArray{Int}(undef, n_edges)
    threads = 128
    blocks = cld(n_edges, threads)
    if blocks > 0
        CUDA.@cuda threads=threads blocks=blocks _cuda_tree_routes_kernel!(
            m2m_parent, m2m_child, l2l_parent, l2l_child, grid.parent_index,
        )
    end
    return m2m_parent, m2m_child, l2l_parent, l2l_child
end

function _cuda_upload_resident_grid(grid::RadixGrid{TF},
        counters::CUDARadixTransferCounters) where TF
    return _cuda_upload_resident_grid(host_resident_radix_grid(grid), counters)
end

function _cuda_upload_resident_grid(host_grid::DeviceRadixGrid{TF},
        counters::CUDARadixTransferCounters) where TF
    return DeviceRadixGrid(
        host_grid.x_min, host_grid.h0, host_grid.ell, host_grid.n_bodies, host_grid.n_cells,
        _to_cuda_array(host_grid.perm, TF, counters, :route),
        _to_cuda_array(host_grid.invperm, TF, counters, :route),
        _to_cuda_array(host_grid.cell_keys, TF, counters, :route),
        _to_cuda_array(host_grid.cell_ranges, TF, counters, :route),
        _to_cuda_array(host_grid.body_system, TF, counters, :route),
        _to_cuda_array(host_grid.body_index, TF, counters, :route),
        _to_cuda_array(host_grid.cell_centers, TF, counters, :route),
        _to_cuda_array(host_grid.node_levels, TF, counters, :route),
        _to_cuda_array(host_grid.node_keys, TF, counters, :route),
        _to_cuda_array(host_grid.node_coords, TF, counters, :route),
        _to_cuda_array(host_grid.node_centers, TF, counters, :route),
        _to_cuda_array(host_grid.parent_index, TF, counters, :route),
        _to_cuda_array(host_grid.child_ranges, TF, counters, :route),
        _to_cuda_array(host_grid.leaf_to_node, TF, counters, :route),
    )
end

function _assert_matching_host_radix_grid(device_grid::DeviceRadixGrid, host_grid::DeviceRadixGrid)
    host_grid.ell == device_grid.ell ||
        throw(ArgumentError("host_grid ell does not match DeviceRadixGrid"))
    host_grid.n_bodies == device_grid.n_bodies ||
        throw(ArgumentError("host_grid n_bodies does not match DeviceRadixGrid"))
    host_grid.n_cells == device_grid.n_cells ||
        throw(ArgumentError("host_grid n_cells does not match DeviceRadixGrid"))
    length(host_grid.node_keys) == length(device_grid.node_keys) ||
        throw(ArgumentError("host_grid node count does not match DeviceRadixGrid"))
    return nothing
end

function _has_device_source_to_buffer_method(device_buffer, system, sort_index)
    sig = Tuple{typeof(device_buffer),typeof(system),typeof(sort_index)}
    return hasmethod(source_to_buffer!, sig)
end

function _has_deprecated_cuda_source_buffer_method(device_buffer, system, sort_index)
    sig = Tuple{typeof(device_buffer),typeof(system),typeof(sort_index)}
    return hasmethod(source_system_to_device_buffer!, sig)
end

function _canonical_cuda_source_buffer(system, ::Type{TF},
        counters::CUDARadixTransferCounters, ::HostResident) where TF
    sort_index = collect(1:get_n_bodies(system))
    host_buffer = allocate_source_buffer(TF, system)
    source_to_buffer!(host_buffer, system, sort_index)
    counters.body_uploads += 1
    return CUDA.CuArray(host_buffer)
end

function _canonical_cuda_source_buffer(system, ::Type{TF},
        counters::CUDARadixTransferCounters, ::DeviceResident) where TF
    # identity permutation: a range, matching the documented `sort_index`
    # default in compatibility.jl. `collect` here allocated an 8 MB Vector{Int}
    # every step at n=1e6 (14% of per-step host allocation, task 028).
    sort_index = Base.OneTo(get_n_bodies(system))
    device_buffer = CUDA.CuArray{TF}(undef, data_per_body(system), length(sort_index))
    if _has_device_source_to_buffer_method(device_buffer, system, sort_index)
        source_to_buffer!(device_buffer, system, sort_index)
        return device_buffer
    elseif _has_deprecated_cuda_source_buffer_method(device_buffer, system, sort_index)
        Base.depwarn(
            "source_system_to_device_buffer! is deprecated; define residency(system) = DeviceResident() and overload source_to_buffer!(device_buffer, system, sort_index)",
            :source_system_to_device_buffer!,
        )
        source_system_to_device_buffer!(device_buffer, system, sort_index)
        return device_buffer
    end
    throw(ArgumentError(
        "DeviceResident CUDA source systems must overload FastMultipole.source_to_buffer!(device_buffer, system, sort_index)",
    ))
end

function _canonical_cuda_source_buffers(systems::Tuple, ::Type{TF},
        counters::CUDARadixTransferCounters) where TF
    return map(system -> _canonical_cuda_source_buffer(system, TF, counters, residency(system)), systems)
end

function _cuda_pack_radix_body_kernel!(body, source_buffer, perm, body_system, body_index, isys)
    sorted_i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    sorted_i > length(perm) && return nothing
    global_i = perm[sorted_i]
    body_system[global_i] == isys || return nothing
    ibody = body_index[global_i]
    @inbounds begin
        body[1, sorted_i] = source_buffer[1, ibody]
        body[2, sorted_i] = source_buffer[2, ibody]
        body[3, sorted_i] = source_buffer[3, ibody]
        body[4, sorted_i] = zero(eltype(body))
        body[5, sorted_i] = source_buffer[5, ibody]
    end
    return nothing
end

function _pack_radix_body_matrix!(body, source_buffers::Tuple, body_perm,
        body_system_ids, body_indices)
    threads = 128
    blocks = cld(length(body_perm), threads)
    blocks == 0 && return body
    for isys in eachindex(source_buffers)
        CUDA.@cuda threads=threads blocks=blocks _cuda_pack_radix_body_kernel!(
            body, source_buffers[isys], body_perm, body_system_ids, body_indices, isys,
        )
    end
    return body
end

function _radix_body_matrix_from_source_buffers(grid::Union{RadixGrid{TF},DeviceRadixGrid{TF}},
        source_buffers::Tuple,
        body_perm, body_system_ids, body_indices) where TF
    n = length(grid.perm)
    body = CUDA.CuArray{TF}(undef, 5, n)
    return _pack_radix_body_matrix!(body, source_buffers, body_perm, body_system_ids, body_indices)
end

function _radix_body_matrix(grid::RadixGrid{TF}, bodies::AbstractMatrix) where TF
    size(bodies, 1) >= 5 ||
        throw(ArgumentError("device-origin CUDA radix bodies must have at least 5 rows: x/y/z/output-placeholder/strength"))
    size(bodies, 2) == length(grid.perm) ||
        throw(ArgumentError("device-origin CUDA radix bodies must already be radix-sorted with one column per grid body"))
    return bodies
end

function _radix_body_matrix(grid::DeviceRadixGrid{TF}, bodies::CUDA.AnyCuArray) where TF
    size(bodies, 1) >= 5 ||
        throw(ArgumentError("device-origin CUDA radix bodies must have at least 5 rows: x/y/z/output-placeholder/strength"))
    size(bodies, 2) == grid.n_bodies ||
        throw(ArgumentError("device-origin CUDA radix bodies must have one column per grid body"))
    device_bodies = eltype(bodies) === TF ? bodies : CUDA.CuArray{TF}(bodies)
    return device_bodies[:, grid.perm]
end

function _radix_body_matrix(grid::RadixGrid{TF}, bodies::CUDA.AnyCuArray) where TF
    size(bodies, 1) >= 5 ||
        throw(ArgumentError("device-origin CUDA radix bodies must have at least 5 rows: x/y/z/output-placeholder/strength"))
    size(bodies, 2) == length(grid.perm) ||
        throw(ArgumentError("device-origin CUDA radix bodies must already be radix-sorted with one column per grid body"))
    return bodies
end

@inline function _cuda_harmonic_index(n, m)
    return (n * (n + 1)) ÷ 2 + m + 1
end

@inline function _cuda_flat_basis_index(n, m, reim)
    return 2 * (_cuda_harmonic_index(n, m) - 1) + reim
end

@inline function _cuda_regular_harmonic_coeff(dx, dy, dz, nt, mt)
    ρ = sqrt(dx * dx + dy * dy + dz * dz)
    if ρ == zero(ρ)
        return nt == 0 && mt == 0 ? (one(ρ), zero(ρ)) : (zero(ρ), zero(ρ))
    end
    θ = acos(dz / ρ)
    ϕ = atan(dy, dx)
    y, x = sincos(θ)
    fact = one(ρ)
    pn = one(ρ)
    ρm = one(ρ)
    i_ei_imag, i_ei_real = sincos(ϕ + convert(typeof(ρ), π / 2))
    i_eim_real = one(ρ)
    i_eim_imag = zero(ρ)
    for m in 0:nt
        p = pn
        ρm_p = ρm * p
        if m == mt && nt == m
            return ρm_p * i_eim_real, ρm_p * i_eim_imag
        end
        p1 = p
        p = x * (2m + 1) * p1
        ρm *= ρ
        ρn = ρm
        for n in (m + 1):nt
            ρn /= -(n + m)
            ρn_p = ρn * p
            if m == mt && n == nt
                return ρn_p * i_eim_real, ρn_p * i_eim_imag
            end
            p2 = p1
            p1 = p
            p = (x * (2n + 1) * p1 - (n + m) * p2) / (n - m + 1)
            ρn *= ρ
        end
        ρm /= -(2m + 2) * (2m + 1)
        pn = -pn * fact * y
        fact += 2
        i_eim_real_tmp = i_eim_real
        i_eim_imag_tmp = i_eim_imag
        i_eim_real = i_eim_real_tmp * i_ei_real - i_eim_imag_tmp * i_ei_imag
        i_eim_imag = i_eim_real_tmp * i_ei_imag + i_eim_imag_tmp * i_ei_real
    end
    return zero(ρ), zero(ρ)
end

function _cuda_b2m_kernel!(phi, source_bodies, cell_centers, cell_ranges, P, ncell)
    i_cell = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i_cell > ncell && return nothing
    first = cell_ranges[1, i_cell]
    count = cell_ranges[2, i_cell]
    cx = cell_centers[1, i_cell]
    cy = cell_centers[2, i_cell]
    cz = cell_centers[3, i_cell]
    @inbounds for n in 0:P
        for m in 0:n
            acc_re = zero(eltype(phi))
            acc_im = zero(eltype(phi))
            sgn = isodd(n + m) ? -one(eltype(phi)) : one(eltype(phi))
            for k in first:(first + count - 1)
                dx = source_bodies[1, k] - cx
                dy = source_bodies[2, k] - cy
                dz = source_bodies[3, k] - cz
                q = source_bodies[5, k]
                rre, rim = _cuda_regular_harmonic_coeff(dx, dy, dz, n, m)
                scale = sgn * q
                acc_re += rre * scale
                acc_im -= rim * scale
            end
            row = _cuda_flat_basis_index(n, m, 1)
            phi[row, i_cell] = acc_re
            phi[row + 1, i_cell] = acc_im
        end
    end
    return nothing
end

# Task 028 note: a warp-per-cell variant (lanes striding the (n, m) list) was
# measured SLOWER on H200 (1.02 -> 2.03 ms F64 at n=1e6/ell=5, job 13015315:
# 113 registers and only 15 of 32 active lanes at P=4), so the original
# thread-per-cell form is retained.
function _cuda_b2m_leaf_nodes_kernel!(phi, source_bodies, cell_centers, cell_ranges,
        leaf_to_node, P, ncell)
    i_cell = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i_cell > ncell && return nothing
    first = cell_ranges[1, i_cell]
    count = cell_ranges[2, i_cell]
    cx = cell_centers[1, i_cell]
    cy = cell_centers[2, i_cell]
    cz = cell_centers[3, i_cell]
    node = leaf_to_node[i_cell]
    @inbounds for n in 0:P
        for m in 0:n
            acc_re = zero(eltype(phi))
            acc_im = zero(eltype(phi))
            sgn = isodd(n + m) ? -one(eltype(phi)) : one(eltype(phi))
            for k in first:(first + count - 1)
                dx = source_bodies[1, k] - cx
                dy = source_bodies[2, k] - cy
                dz = source_bodies[3, k] - cz
                q = source_bodies[5, k]
                rre, rim = _cuda_regular_harmonic_coeff(dx, dy, dz, n, m)
                scale = sgn * q
                acc_re += rre * scale
                acc_im -= rim * scale
            end
            row = _cuda_flat_basis_index(n, m, 1)
            phi[row, node] = acc_re
            phi[row + 1, node] = acc_im
        end
    end
    return nothing
end

function _cuda_direct_source_output_kernel!(output, target_bodies, source_bodies, ntarget)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > ntarget && return nothing
    xi = target_bodies[1, i]
    yi = target_bodies[2, i]
    zi = target_bodies[3, i]
    u = zero(eltype(output))
    gx = zero(eltype(output))
    gy = zero(eltype(output))
    gz = zero(eltype(output))
    c = inv(eltype(output)(4) * eltype(output)(π))
    @inbounds for j in 1:ntarget
        dx = xi - source_bodies[1, j]
        dy = yi - source_bodies[2, j]
        dz = zi - source_bodies[3, j]
        r2 = dx * dx + dy * dy + dz * dz
        if r2 > zero(r2)
            invr = inv(sqrt(r2))
            q = source_bodies[5, j] * c
            u += q * invr
            invr3 = invr * invr * invr
            gx -= q * dx * invr3
            gy -= q * dy * invr3
            gz -= q * dz * invr3
        end
    end
    output[1, i] = u
    output[2, i] = gx
    output[3, i] = gy
    output[4, i] = gz
    return nothing
end

function _cuda_scatter_output_to_target_buffer_kernel!(target_buffer, output, perm,
        body_system, body_index, isys, scalar_row, gradient_start, gradient_stop, n_bodies)
    sorted_i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    sorted_i > n_bodies && return nothing
    global_i = perm[sorted_i]
    body_system[global_i] == isys || return nothing
    ibody = body_index[global_i]
    @inbounds begin
        if scalar_row > 0
            target_buffer[scalar_row, ibody] = output[1, sorted_i]
        end
        if gradient_start <= gradient_stop
            target_buffer[gradient_start, ibody] = output[2, sorted_i]
            target_buffer[gradient_start + 1, ibody] = output[3, sorted_i]
            target_buffer[gradient_start + 2, ibody] = output[4, sorted_i]
        end
    end
    return nothing
end

function _copy_radix_output_to_device_target_buffer!(target_buffer, output,
        body_perm, body_system_ids, body_indices, isys::Integer, derivatives_switch,
        n_bodies::Integer=size(output, 2))
    fill!(target_buffer, zero(eltype(target_buffer)))
    isempty(hessian_range(derivatives_switch)) ||
        throw(ArgumentError("CUDA radix output finalization does not provide hessian rows"))
    grange = gradient_range(derivatives_switch)
    gradient_start = isempty(grange) ? 1 : first(grange)
    gradient_stop = isempty(grange) ? 0 : last(grange)
    threads = 128
    blocks = cld(n_bodies, threads)
    blocks == 0 && return target_buffer
    CUDA.@cuda threads=threads blocks=blocks _cuda_scatter_output_to_target_buffer_kernel!(
        target_buffer, output, body_perm, body_system_ids, body_indices, isys,
        scalar_potential_index(derivatives_switch), gradient_start, gradient_stop, n_bodies,
    )
    return target_buffer
end

function _launch_cuda_b2m!(state::DeviceResidentRadixState{TF}) where TF
    fill!(state.multipoles.phi, zero(TF))
    fill!(state.multipoles.chi, zero(TF))
    threads = 128
    ncell = state.counts.n_cells
    blocks = cld(ncell, threads)
    blocks == 0 && return state
    if state.grid isa DeviceRadixGrid
        CUDA.@cuda threads=threads blocks=blocks _cuda_b2m_leaf_nodes_kernel!(
            state.multipoles.phi, state.source_bodies, state.cell_centers,
            state.cell_ranges, state.grid.leaf_to_node,
            state.invariant_cache.basis_info.orders.P_phi, ncell,
        )
    else
        CUDA.@cuda threads=threads blocks=blocks _cuda_b2m_kernel!(
            state.multipoles.phi, state.source_bodies, state.cell_centers,
            state.cell_ranges, state.invariant_cache.basis_info.orders.P_phi, ncell,
        )
    end
    return state
end

function _launch_cuda_direct_output!(state::DeviceResidentRadixState)
    threads = 128
    ntarget = state.counts.n_bodies
    blocks = cld(ntarget, threads)
    blocks == 0 && return state
    CUDA.@cuda threads=threads blocks=blocks _cuda_direct_source_output_kernel!(
        state.output, state.target_bodies, state.source_bodies, ntarget,
    )
    return state
end

# One-shot cuda_radix_state builders: the factored plan there would need
# per-geometry groups with no in-place refresh; the factored device path is
# supported through RadixFMMCache(device=true) only (task 023b).
function _assert_cuda_materialized_operator!(options::CUDARadixLifecycleOptions)
    options.m2l_strategy isa DenseTranslationM2L && throw(ArgumentError(
        "DenseTranslationM2L is not supported by the one-shot cuda_radix_state " *
        "builders; use RadixFMMCache(...; device=true, options=...) for the " *
        "device-resident dense M2L lifecycle (task 023f)"))
    options.operator isa MaterializedYRotationM2L && return nothing
    if options.operator isa FactoredRotationM2L
        throw(ArgumentError(
            "FactoredRotationM2L is not supported by the one-shot cuda_radix_state " *
            "builders; use RadixFMMCache(...; device=true, options=...) for the " *
            "factored CUDA resident lifecycle (task 023b)",
        ))
    end
    throw(ArgumentError("CUDA resident lifecycle supports MaterializedYRotationM2L; got $(typeof(options.operator))"))
end

# Recurring RadixFMMCache device lifecycle: materialized/concat, factored (task
# 023b), and precomputed-y (task 023d) are all implemented.
function _assert_cuda_supported_operator!(options::CUDARadixLifecycleOptions)
    # DenseTranslationM2L (task 023f) pairs with MaterializedYRotationM2L (enforced
    # by CUDARadixLifecycleOptions) and is supported on the recurring cache.
    options.operator isa Union{MaterializedYRotationM2L,FactoredRotationM2L} && return nothing
    throw(ArgumentError(
        "CUDA resident lifecycle supports MaterializedYRotationM2L and " *
        "FactoredRotationM2L; got $(typeof(options.operator))"))
end

function _assert_cuda_resident_stage!(state::DeviceResidentRadixState, stage::Symbol)
    state.counters.expansion_host_copies == 0 ||
        throw(AssertionError("CUDA resident lifecycle observed expansion host copies after $stage"))
    fields = (
        :source_bodies, :target_bodies, :body_perm, :body_system_ids, :body_indices,
        :cell_centers, :cell_ranges, :m2m_parent_routes, :m2m_child_routes,
        :l2l_parent_routes, :l2l_child_routes, :route_levels, :route_offsets,
        :route_targets, :route_sources, :direct_targets, :direct_sources, :output,
    )
    for field in fields
        value = getfield(state, field)
        value isa CUDA.AnyCuArray ||
            throw(AssertionError("CUDA resident lifecycle $stage has non-device $field"))
    end
    state.multipoles.phi isa CUDA.AnyCuArray ||
        throw(AssertionError("CUDA resident lifecycle $stage has non-device multipoles.phi"))
    state.locals.phi isa CUDA.AnyCuArray ||
        throw(AssertionError("CUDA resident lifecycle $stage has non-device locals.phi"))
    size(state.multipoles.chi, 1) == 0 || state.multipoles.chi isa CUDA.AnyCuArray ||
        throw(AssertionError("CUDA resident lifecycle $stage has non-device multipoles.chi"))
    size(state.locals.chi, 1) == 0 || state.locals.chi isa CUDA.AnyCuArray ||
        throw(AssertionError("CUDA resident lifecycle $stage has non-device locals.chi"))
    _assert_cuda_scratch_resident!(state.scratch, stage)
    return nothing
end

function _assert_cuda_scratch_resident!(scratch, stage::Symbol)
    scratch === nothing && return nothing
    # Fast path (task 028): `_assert_cuda_scratch_value!` builds a `$path[$i]`
    # String for every element it walks, purely to have a message ready if the
    # invariant fails. At n=1e6/ell=5 that was 1,346,598 allocations and
    # 33.8 MB per step -- 59% of all per-step host allocation -- on a path that
    # essentially never throws. Prove the invariant allocation-free first and
    # fall through to the walker only when the cheap proof fails, so failures
    # still report the exact same path and message.
    _cuda_scratch_value_ok(scratch) && return nothing
    _assert_cuda_scratch_value!(scratch, stage, "scratch")
    return nothing
end

# Unrolled, type-stable struct walk. A plain `for f in fieldnames(typeof(v))`
# loop with `getfield(v, f)` is type-unstable and boxes on every field, which
# would reintroduce the allocation this fast path exists to remove.
@generated function _cuda_scratch_struct_ok(value)
    checks = [:(_cuda_scratch_value_ok(getfield(value, $(QuoteNode(f)))) || return false)
              for f in fieldnames(value)]
    return Expr(:block, checks..., :(return true))
end

"""
Allocation-free residency predicate mirroring the accept conditions of
`_assert_cuda_scratch_value!`.

**This must stay in sync with `_assert_cuda_scratch_value!` below.** It is
deliberately conservative: it may return `false` for a value that is in fact
valid (the walker then confirms it), but it must never return `true` for a
value the walker would reject, since that would silently weaken the
device-residency invariant contract.
"""
function _cuda_scratch_value_ok(value)
    if value isa FlatCoefficientBuffer || value isa DegreeMajorRealBuffer
        value.phi isa CUDA.AnyCuArray || return false
        return size(value.chi, 1) == 0 || value.chi isa CUDA.AnyCuArray
    elseif value isa CUDA.AnyCuArray || value === nothing ||
            value isa Number || value isa Symbol || value isa Type ||
            value isa OperatorBasisInfo || value isa Base.RefValue
        return true
    elseif value isa AbstractVector
        # a concretely device-typed vector satisfies the invariant by type, so
        # the large route/class vectors need no element-wise walk at all
        eltype(value) <: CUDA.AnyCuArray && isconcretetype(eltype(value)) &&
            return true
        for item in value
            _cuda_scratch_value_ok(item) || return false
        end
        return true
    elseif value isa Union{Tuple,NamedTuple}
        for item in value
            _cuda_scratch_value_ok(item) || return false
        end
        return true
    elseif value isa ResidentM2LFactoredPlan
        _cuda_scratch_value_ok(value.route_class) || return false
        groups = value.groups
        isempty(groups) || _cuda_scratch_value_ok(groups[1]) || return false
        _cuda_scratch_value_ok(value.class_counts) || return false
        _cuda_scratch_value_ok(value.ym_flat) || return false
        _cuda_scratch_value_ok(value.z_flat) || return false
        return _cuda_scratch_value_ok(value.whole_pass[])
    elseif value isa ResidentM2LPrecomputedYPlan
        _cuda_scratch_value_ok(value.route_class) || return false
        _cuda_scratch_value_ok(value.class_counts) || return false
        _cuda_scratch_value_ok(value.y_flat_mult) || return false
        _cuda_scratch_value_ok(value.y_flat_loc) || return false
        _cuda_scratch_value_ok(value.z_flat) || return false
        return _cuda_scratch_value_ok(value.whole_pass[])
    elseif value isa ResidentM2LDenseCUDAPlan
        _cuda_scratch_value_ok(value.route_class) || return false
        _cuda_scratch_value_ok(value.operators) || return false
        _cuda_scratch_value_ok(value.class_counts) || return false
        _cuda_scratch_value_ok(value.src_slab) || return false
        _cuda_scratch_value_ok(value.dst_slab) || return false
        return _cuda_scratch_value_ok(value.whole_pass[])
    elseif value isa ResidentOperatorGroup || value isa ResidentOperatorWorkspace ||
            value isa DegreeMajorMaps || value isa ResidentM2LConcatPlan ||
            value isa ConcatChannelOps || value isa StackedYChannel
        return _cuda_scratch_struct_ok(value)
    end
    return false   # unsupported/unknown -> let the walker produce the error
end

# NOTE: keep the accept conditions here in sync with `_cuda_scratch_value_ok`
# above, which is the allocation-free fast path guarding this walker.
function _assert_cuda_scratch_value!(value, stage::Symbol, path::AbstractString)
    if value isa FlatCoefficientBuffer || value isa DegreeMajorRealBuffer
        value.phi isa CUDA.AnyCuArray ||
            throw(AssertionError("CUDA resident lifecycle $stage has non-device $path.phi"))
        size(value.chi, 1) == 0 || value.chi isa CUDA.AnyCuArray ||
            throw(AssertionError("CUDA resident lifecycle $stage has non-device $path.chi"))
    elseif value isa CUDA.AnyCuArray || value === nothing ||
            value isa Number || value isa Symbol || value isa Type ||
            value isa OperatorBasisInfo || value isa Base.RefValue
        return nothing
    elseif value isa AbstractVector
        for (i, item) in pairs(value)
            _assert_cuda_scratch_value!(item, stage, "$path[$i]")
        end
    elseif value isa Union{Tuple,NamedTuple}
        for (i, item) in pairs(value)
            _assert_cuda_scratch_value!(item, stage, "$path[$i]")
        end
    elseif value isa ResidentM2LFactoredPlan
        # host_class_counts/class_starts/class_theta/class_phi are host-side
        # per-step refresh metadata by design (mirrors the pinned host scalar
        # staging); only the operator/route arrays must be device-resident.
        # Compact CUDA factored plans intentionally have no per-offset groups.
        # Keep the representative check for older/non-compact plans used by
        # focused construction tests.
        _assert_cuda_scratch_value!(value.route_class, stage, "$path.route_class")
        groups = value.groups
        isempty(groups) ||
            _assert_cuda_scratch_value!(groups[1], stage, "$path.groups[1]")
        _assert_cuda_scratch_value!(value.class_counts, stage, "$path.class_counts")
        _assert_cuda_scratch_value!(value.ym_flat, stage, "$path.ym_flat")
        _assert_cuda_scratch_value!(value.z_flat, stage, "$path.z_flat")
        _assert_cuda_scratch_value!(value.whole_pass[], stage, "$path.whole_pass")
    elseif value isa ResidentM2LPrecomputedYPlan
        # Host angle/offset metadata and the histogram host mirror are per-step
        # refresh staging by design (task 023d); the operative device fields are
        # the route classes, offset histogram, flat operator tables, and the
        # whole-pass bundle. The nested host operator storage is intentionally
        # empty on compact CUDA plans and never walked.
        _assert_cuda_scratch_value!(value.route_class, stage, "$path.route_class")
        _assert_cuda_scratch_value!(value.class_counts, stage, "$path.class_counts")
        _assert_cuda_scratch_value!(value.y_flat_mult, stage, "$path.y_flat_mult")
        _assert_cuda_scratch_value!(value.y_flat_loc, stage, "$path.y_flat_loc")
        _assert_cuda_scratch_value!(value.z_flat, stage, "$path.z_flat")
        _assert_cuda_scratch_value!(value.whole_pass[], stage, "$path.whole_pass")
    elseif value isa ResidentM2LDenseCUDAPlan
        # host_class_counts/class_starts/class_capacities are per-step count
        # staging by design (task 023f); the operative device fields are the route
        # classes, packed operators, the offset histogram, and the gather/GEMM/
        # scatter slabs.
        _assert_cuda_scratch_value!(value.route_class, stage, "$path.route_class")
        _assert_cuda_scratch_value!(value.operators, stage, "$path.operators")
        _assert_cuda_scratch_value!(value.class_counts, stage, "$path.class_counts")
        _assert_cuda_scratch_value!(value.src_slab, stage, "$path.src_slab")
        _assert_cuda_scratch_value!(value.dst_slab, stage, "$path.dst_slab")
        _assert_cuda_scratch_value!(value.whole_pass[], stage, "$path.whole_pass")
    elseif value isa ResidentOperatorGroup || value isa ResidentOperatorWorkspace ||
            value isa DegreeMajorMaps || value isa ResidentM2LConcatPlan ||
            value isa ConcatChannelOps || value isa StackedYChannel
        for field in fieldnames(typeof(value))
            _assert_cuda_scratch_value!(getfield(value, field), stage, "$path.$field")
        end
    else
        throw(AssertionError("CUDA resident lifecycle $stage has unsupported $path::$(typeof(value))"))
    end
    return nothing
end

function _host_operator_cache_for_cuda(state::DeviceResidentRadixState{TF,B,LH}) where {TF,B,LH}
    return OperatorInvariantCache(TF, state.invariant_cache.basis_info)
end

function _copy_cuda_column_to_host!(dest::FlatCoefficientBuffer, j::Integer,
        source::FlatCoefficientBuffer, col::Integer, counters::CUDARadixTransferCounters,
        ::Val{LH}) where LH
    counters.expansion_host_copies += 1
    @inbounds for row in axes(dest.phi, 1)
        dest.phi[row, j] = source.phi[row, col]
    end
    if LH
        @inbounds for row in axes(dest.chi, 1)
            dest.chi[row, j] = source.chi[row, col]
        end
    end
    return dest
end

function _scatter_host_column_to_cuda!(target::FlatCoefficientBuffer, col::Integer,
        source::FlatCoefficientBuffer, j::Integer, counters::CUDARadixTransferCounters,
        ::Val{LH}) where LH
    counters.expansion_host_copies += 1
    @inbounds for row in axes(source.phi, 1)
        target.phi[row, col] = target.phi[row, col] + source.phi[row, j]
    end
    if LH
        @inbounds for row in axes(source.chi, 1)
            target.chi[row, col] = target.chi[row, col] + source.chi[row, j]
        end
    end
    return target
end

function _zero_cuda_nonleaf_multipoles!(state::DeviceResidentRadixState{TF,B,LH}) where {TF,B,LH}
    @inbounds for node in 1:length(state.grid.node_keys)
        state.grid.node_levels[node] < state.grid.ell || continue
        for row in axes(state.multipoles.phi, 1)
            state.multipoles.phi[row, node] = zero(TF)
        end
        if LH
            for row in axes(state.multipoles.chi, 1)
                state.multipoles.chi[row, node] = zero(TF)
            end
        end
    end
    return state
end

# CUDA.rsqrt lowers to rsqrt.approx (~1e-7 relative error in Float64, i.e.
# single-precision quality), so the Float64 method refines it with two Newton
# steps back to ~1-2 ulp — still far cheaper than the sqrt + divide it replaces.
# Callers must guard x > 0: the Newton step turns rsqrt(0) = Inf into NaN.
@inline _cuda_fast_rsqrt(x::Float32) = CUDA.rsqrt(x)
@inline function _cuda_fast_rsqrt(x::Float64)
    y = CUDA.rsqrt(x)
    hx = 0.5 * x
    y *= 1.5 - hx * y * y
    y *= 1.5 - hx * y * y
    return y
end

# Task 028 lever 1: warp-per-pair, grid-stride. The original kernel gave each
# thread a whole cell-pair — a serial ~(30x30)-interaction dependent chain with
# warp divergence on ragged cell counts, measured at ~4% of the FP32 peak rate.
# Now each warp owns one pair: lanes stride the target bodies, so the inner
# source loop is lane-uniform (the four loads of body `j` broadcast from cache)
# and the per-thread chain shrinks by ~32x. Warps advance by the total warp
# count, so the block count is capped by DIRECT_CUDA_MAX_BLOCKS instead of
# scaling with npairs. `_cuda_fast_rsqrt` replaces inv(sqrt): hardware-rate in
# Float32, Newton-refined approx in Float64 (the stage was FP64-rsqrt-limited).
# Accumulation stays 4 atomics per (pair, target body) — a target cell appears
# in many pairs, so plain stores would race across warps.
function _cuda_direct_pairs_output_kernel!(output, source_bodies, cell_ranges,
        direct_targets, direct_sources, npairs)
    T = eltype(output)
    lane = (threadIdx().x - Int32(1)) % Int32(32)
    warps_per_block = blockDim().x ÷ Int32(32)
    warp_in_block = (threadIdx().x - Int32(1)) ÷ Int32(32)
    pair_i = (blockIdx().x - 1) * warps_per_block + warp_in_block + 1
    warp_stride = gridDim().x * warps_per_block
    c = inv(T(4) * T(π))
    @inbounds while pair_i <= npairs
        target_cell = direct_targets[pair_i]
        source_cell = direct_sources[pair_i]
        tfirst = cell_ranges[1, target_cell]
        tlast = tfirst + cell_ranges[2, target_cell] - 1
        sfirst = cell_ranges[1, source_cell]
        slast = sfirst + cell_ranges[2, source_cell] - 1
        i = tfirst + lane
        while i <= tlast
            xi = source_bodies[1, i]
            yi = source_bodies[2, i]
            zi = source_bodies[3, i]
            u = zero(T)
            gx = zero(T)
            gy = zero(T)
            gz = zero(T)
            for j in sfirst:slast
                i == j && continue
                dx = xi - source_bodies[1, j]
                dy = yi - source_bodies[2, j]
                dz = zi - source_bodies[3, j]
                r2 = dx * dx + dy * dy + dz * dz
                if r2 > zero(r2)
                    invr = _cuda_fast_rsqrt(r2)
                    q = source_bodies[5, j] * c
                    u += q * invr
                    invr3 = invr * invr * invr
                    gx -= q * dx * invr3
                    gy -= q * dy * invr3
                    gz -= q * dz * invr3
                end
            end
            CUDA.@atomic output[1, i] += u
            CUDA.@atomic output[2, i] += gx
            CUDA.@atomic output[3, i] += gy
            CUDA.@atomic output[4, i] += gz
            i += 32
        end
        pair_i += warp_stride
    end
    return nothing
end

# Task 028 rider: warp-per-cell (was one thread per cell looping its ~30 bodies
# serially, i.e. only ncell threads of parallelism). Lanes stride the cell's
# bodies; each body belongs to exactly one cell and one lane, so the `+=` into
# `output` stays non-atomic.
function _cuda_l2b_output_kernel!(output, source_bodies, cell_centers, cell_ranges,
        leaf_to_node, local_phi, local_chi, P_phi, P_active, lhv, ncell)
    lane = (threadIdx().x - Int32(1)) % Int32(32)
    warps_per_block = blockDim().x ÷ Int32(32)
    warp_in_block = (threadIdx().x - Int32(1)) ÷ Int32(32)
    cell = (blockIdx().x - 1) * warps_per_block + warp_in_block + 1
    cell > ncell && return nothing
    node = leaf_to_node[cell]
    first = cell_ranges[1, cell]
    last = first + cell_ranges[2, cell] - 1
    cx = cell_centers[1, cell]
    cy = cell_centers[2, cell]
    cz = cell_centers[3, cell]
    i = first + lane
    @inbounds while i <= last
        scalar_potential, gx, gy, gz = _resident_local_eval_flat(
            local_phi, local_chi, node,
            source_bodies[1, i] - cx,
            source_bodies[2, i] - cy,
            source_bodies[3, i] - cz,
            P_phi, P_active, lhv,
        )
        output[1, i] += scalar_potential
        output[2, i] += gx
        output[3, i] += gy
        output[4, i] += gz
        i += 32
    end
    return nothing
end

function _cuda_scatter_accumulate_columns_kernel!(dest, row_idx, col_targets, slab)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    nrow = size(slab, 1)
    i > nrow * size(slab, 2) && return nothing
    row = (i - 1) % nrow + 1
    col = (i - 1) ÷ nrow + 1
    @inbounds CUDA.@atomic dest[row_idx[row], col_targets[col]] += slab[row, col]
    return nothing
end

# Duplicate targets across concatenated M2L routes require atomic accumulation.
function _scatter_accumulate_columns!(dest::CUDA.AnyCuArray, row_idx, col_targets, slab)
    length(slab) == 0 && return dest
    threads = 256
    blocks = cld(length(slab), threads)
    CUDA.@cuda threads=threads blocks=blocks _cuda_scatter_accumulate_columns_kernel!(
        dest, row_idx, col_targets, slab,
    )
    return dest
end

function _cuda_gather_rotate_z_kernel!(dst, src, flat_idx, cols, row_m, row_ssign,
        row_pair, phis, sgn)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    nrow = size(dst, 1)
    i > nrow * size(dst, 2) && return nothing
    row = (i - 1) % nrow + 1
    col = (i - 1) ÷ nrow + 1
    @inbounds begin
        s, c = sincos(row_m[row] * phis[col])
        csrc = cols[col]
        a = src[flat_idx[row], csrc]
        b = src[flat_idx[row_pair[row]], csrc]
        dst[row, col] = c * a + sgn * row_ssign[row] * s * b
    end
    return nothing
end

# Fused flat-column gather + z rotation (see the generic method in
# translate_batched.jl): one kernel replaces the allocating fancy-index gather plus
# the broadcast z-rotation stage.
function _gather_rotate_z!(dst::CUDA.AnyCuArray, src::CUDA.AnyCuArray, flat_idx, cols,
        row_m, row_ssign, row_pair, phis, inverse::Bool)
    length(dst) == 0 && return dst
    sgn = inverse ? -one(eltype(dst)) : one(eltype(dst))
    threads = 256
    blocks = cld(length(dst), threads)
    CUDA.@cuda threads=threads blocks=blocks _cuda_gather_rotate_z_kernel!(
        dst, src, flat_idx, cols, row_m, row_ssign, row_pair, phis, sgn,
    )
    return dst
end

function _cuda_rotate_z_scatter_accumulate_kernel!(dest, slab, flat_idx, col_targets,
        row_m, row_ssign, row_pair, phis)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    nrow = size(slab, 1)
    i > nrow * size(slab, 2) && return nothing
    row = (i - 1) % nrow + 1
    col = (i - 1) ÷ nrow + 1
    @inbounds begin
        s, c = sincos(row_m[row] * phis[col])
        v = c * slab[row, col] - row_ssign[row] * s * slab[row_pair[row], col]
        CUDA.@atomic dest[flat_idx[row], col_targets[col]] += v
    end
    return nothing
end

# Fused inverse z rotation + atomic accumulating scatter (see the generic method in
# translate_batched.jl).
function _rotate_z_scatter_accumulate!(dest::CUDA.AnyCuArray, slab, flat_idx,
        col_targets, row_m, row_ssign, row_pair, phis)
    length(slab) == 0 && return dest
    threads = 256
    blocks = cld(length(slab), threads)
    CUDA.@cuda threads=threads blocks=blocks _cuda_rotate_z_scatter_accumulate_kernel!(
        dest, slab, flat_idx, col_targets, row_m, row_ssign, row_pair, phis,
    )
    return dest
end

function _cuda_gather_rows_kernel!(dst, src, rows)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    nrow = size(dst, 1)
    i > nrow * size(dst, 2) && return nothing
    row = (i - 1) % nrow + 1
    col = (i - 1) ÷ nrow + 1
    @inbounds dst[row, col] = src[rows[row], col]
    return nothing
end

# Allocation-free row gather (see the generic method in translate_batched.jl).
function _gather_rows!(dst::CUDA.AnyCuArray, src::CUDA.AnyCuArray, rows)
    length(dst) == 0 && return dst
    threads = 256
    blocks = cld(length(dst), threads)
    CUDA.@cuda threads=threads blocks=blocks _cuda_gather_rows_kernel!(dst, src, rows)
    return dst
end

function _cuda_gather_values_kernel!(dst, src, ids)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > length(dst) && return nothing
    @inbounds dst[i] = src[ids[i]]
    return nothing
end

# Allocation-free value gather (see the generic method in translate_batched.jl).
function _gather_values!(dst::CUDA.AnyCuArray, src::CUDA.AnyCuArray, ids)
    length(dst) == 0 && return dst
    threads = 256
    blocks = cld(length(dst), threads)
    CUDA.@cuda threads=threads blocks=blocks _cuda_gather_values_kernel!(dst, src, ids)
    return dst
end

#------- device-resident factored M2L (Matrix Operator Refactor, task 023b) -------#
#
# CUDA mirror of the 023a host factored stage: per accepted offset class (shared
# phi/theta/r), the Plain-H chain gather+Z_phi -> per-degree U_n/V_n forward Y ->
# fixed-m z translation -> [LH rows] -> return Y -> Z_phi^{-1}+scatter, at
# O(P^3)/column. Device route emission is class-major and contiguous (see
# _cuda_generate_radix_routes!), so class k's columns are one contiguous range of
# route_sources/route_targets; the per-step refresh is a histogram plus one pinned
# counts download, with no per-group index repacking. All operator data (flat
# Plain-H mode blocks, per-class flat z tables, LH rows) uploads at construction
# inside the plan/groups; the per-class stages run on the capacity-sized workspace
# slabs, so the recurring step performs no device allocation.

# DEBUG-gated physical-subspace guard for device expansion buffers: downloads the
# buffer (diagnostic only; production keeps DEBUG[] off) and reuses the host scan.
@inline function _assert_factored_input_physical(
        source::FlatCoefficientBuffer{TF,<:CUDA.AnyCuArray,B,LH},
    ) where {TF,B<:AbstractOperatorBasis,LH}
    DEBUG[] || return nothing
    host = FlatCoefficientBuffer(TF, source.basis_info, size(source.phi, 2))
    copyto!(host.phi, source.phi)
    LH && copyto!(host.chi, source.chi)
    return _assert_factored_input_physical(host)
end

function _cuda_class_histogram_kernel!(counts, route_class, n_routes)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > n_routes && return nothing
    @inbounds CUDA.@atomic counts[route_class[i]] += Int32(1)
    return nothing
end

# Device mirror of _refresh_factored_m2l_routes!: per-class counts by histogram,
# one pinned host download (uncounted, like the existing host_scalar staging),
# host prefix-sum into class_starts, and group count updates. Class contiguity is
# guaranteed by the class-major device route emission.
function _cuda_refresh_factored_m2l_routes!(plan::ResidentM2LFactoredPlan,
        route_class, n_routes::Int)
    counts = plan.class_counts
    fill!(counts, Int32(0))
    if n_routes > 0
        threads = 256
        blocks = cld(n_routes, threads)
        CUDA.@cuda threads=threads blocks=blocks _cuda_class_histogram_kernel!(
            counts, route_class, n_routes,
        )
    end
    host_counts = plan.host_class_counts::Vector{Int32}
    copyto!(host_counts, counts)
    starts = plan.class_starts
    starts[1] = 0
    @inbounds for k in eachindex(host_counts)
        c = Int(host_counts[k])
        starts[k + 1] = starts[k] + c
        isempty(plan.groups) || (plan.groups[k].count[] = c)
    end
    starts[end] == n_routes ||
        throw(AssertionError("factored route classes do not partition the device routes"))
    return plan
end

# Degree n of a 1-based degree-major row (n^2 < row <= (n+1)^2), with integer
# fixups against float rounding at the square boundaries.
@inline function _cuda_degree_of_row(row::Int)
    n = unsafe_trunc(Int, sqrt(Float64(row - 1)))
    while n > 0 && n * n >= row
        n -= 1
    end
    while (n + 1) * (n + 1) < row
        n += 1
    end
    return n
end

# Forward half of one factored y application: per mode row (degree n, mode k),
# G = V_n * x over the degree's contiguous storage rows, then the shared-angle
# e^{i nu theta} phase (theta is a class constant scalar). Flat mode indexing
# matches _ymode_real_blocks: entry [k, q] of degree n lives at
# ymode_offset(n) + (q - 1) * (2n + 1) + k.
function _cuda_factored_y_phase_kernel!(gre, gim, in_slab, V_re, V_im, theta)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    ndof = size(gre, 1)
    idx > ndof * size(gre, 2) && return nothing
    row = (idx - 1) % ndof + 1
    j = (idx - 1) ÷ ndof + 1
    n = _cuda_degree_of_row(row)
    row0 = n * n
    k = row - row0
    d = 2 * n + 1
    off = ymode_offset(n)
    TF = eltype(gre)
    gr = zero(TF)
    gi = zero(TF)
    @inbounds begin
        for q in 1:d
            x = in_slab[row0 + q, j]
            v = off + (q - 1) * d + k
            gr += V_re[v] * x
            gi += V_im[v] * x
        end
        s, c = sincos(TF(k - n - 1) * theta)
        gre[row, j] = c * gr - s * gi
        gim[row, j] = s * gr + c * gi
    end
    return nothing
end

# Return half: y = U_re * G' - U_im * G'' over the degree's mode rows.
function _cuda_factored_y_out_kernel!(out_slab, gre, gim, U_re, U_im)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    ndof = size(out_slab, 1)
    idx > ndof * size(out_slab, 2) && return nothing
    row = (idx - 1) % ndof + 1
    j = (idx - 1) ÷ ndof + 1
    n = _cuda_degree_of_row(row)
    row0 = n * n
    k = row - row0
    d = 2 * n + 1
    off = ymode_offset(n)
    TF = eltype(out_slab)
    y = zero(TF)
    @inbounds begin
        for q in 1:d
            u = off + (q - 1) * d + k
            y += U_re[u] * gre[row0 + q, j]
            y -= U_im[u] * gim[row0 + q, j]
        end
        out_slab[row, j] = y
    end
    return nothing
end

# One factored y application on a degree-major slab: two fused block-diagonal
# kernels around the class-constant phase (O(P^3)/column, 2 launches).
function _cuda_factored_y!(out_slab, in_slab, U_re, U_im, V_re, V_im, theta,
        scratch_re, scratch_im)
    n_el = length(out_slab)
    n_el == 0 && return out_slab
    threads = 256
    blocks = cld(n_el, threads)
    CUDA.@cuda threads=threads blocks=blocks _cuda_factored_y_phase_kernel!(
        scratch_re, scratch_im, in_slab, V_re, V_im, theta,
    )
    CUDA.@cuda threads=threads blocks=blocks _cuda_factored_y_out_kernel!(
        out_slab, scratch_re, scratch_im, U_re, U_im,
    )
    return out_slab
end

# Fused fixed-m z translation from the class's flat block table (m2l_z_blocks!
# layout): out(n, m, ri) = sum(np = m:P_loop) K_m[n, np] * in(np, m, ri). Each
# per-m block acts identically on the re and im rows, so the row's k position
# within its degree is preserved across source degrees.
function _cuda_ztranslate_fixed_m_kernel!(out_slab, in_slab, zcol, P_loop, P_block)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    ndof = size(out_slab, 1)
    idx > ndof * size(out_slab, 2) && return nothing
    row = (idx - 1) % ndof + 1
    j = (idx - 1) ÷ ndof + 1
    n = _cuda_degree_of_row(row)
    k = row - n * n
    m = k == 1 ? 0 : (k >> 1)
    w = P_block - m + 1
    base = m2l_z_block_offset(m, P_block) + (n - m) + 1
    TF = eltype(out_slab)
    acc = zero(TF)
    @inbounds begin
        for np in m:P_loop
            acc += zcol[base + (np - m) * w] * in_slab[np * np + k, j]
        end
        out_slab[row, j] = acc
    end
    return nothing
end

function _cuda_ztranslate_fixed_m!(out_slab, in_slab, zcol, P_loop::Int, P_block::Int)
    n_el = length(out_slab)
    n_el == 0 && return out_slab
    threads = 256
    blocks = cld(n_el, threads)
    CUDA.@cuda threads=threads blocks=blocks _cuda_ztranslate_fixed_m_kernel!(
        out_slab, in_slab, zcol, P_loop, P_block,
    )
    return out_slab
end

# One offset class of the device factored M2L: same slab rotation as the host
# _resident_factored_m2l_group_apply! (every stage's scratch is disjoint from its
# in/out at that point in the chain).
function _cuda_factored_m2l_class_apply!(state::DeviceResidentRadixState{TF,B,LH},
        ws::ResidentOperatorWorkspace{TF,B,LH}, plan::ResidentM2LFactoredPlan,
        wp::NamedTuple, kclass::Int, cols::UnitRange{Int}) where {TF,B,LH}
    n = length(cols)
    P_phi = ws.basis_info.orders.P_phi
    P_active = ws.basis_info.orders.P_active
    theta = (plan.class_theta::Vector{TF})[kclass]
    phi = (plan.class_phi::Vector{TF})[kclass]
    r = (plan.class_r::Vector{TF})[kclass]
    ym = plan.ym_flat
    zcol = view(plan.z_flat, :, kclass)
    src_cols = view(state.route_sources, cols)
    tgt_cols = view(state.route_targets, cols)
    group_phis = _vector_prefix_view(ws.phis, n)
    fill!(group_phis, phi)
    aphi = _matrix_col_view(ws.aphi, n); yphi = _matrix_col_view(ws.yphi, n)
    zphi = _matrix_col_view(ws.zphi, n); rphi = _matrix_col_view(ws.rphi, n)
    cphi = _matrix_col_view(ws.cphi, n)
    _gather_rotate_z!(aphi, state.multipoles.phi, ws.phi_flat_idx, src_cols,
        ws.maps_phi.row_m, ws.maps_phi.row_ssign, ws.maps_phi.row_pair, group_phis, false)
    _cuda_factored_y!(yphi, aphi, ym.mult_U_re, ym.mult_U_im, ym.mult_V_re, ym.mult_V_im,
        theta, rphi, cphi)
    _cuda_ztranslate_fixed_m!(zphi, yphi, zcol, P_phi, P_active)
    ret_phi = zphi
    if LH
        achi = _matrix_col_view(ws.achi, n); ychi = _matrix_col_view(ws.ychi, n)
        zchi = _matrix_col_view(ws.zchi, n); rchi = _matrix_col_view(ws.rchi, n)
        cchi = _matrix_col_view(ws.cchi, n)
        _gather_rotate_z!(achi, state.multipoles.chi, ws.chi_flat_idx, src_cols,
            ws.maps_chi.row_m, ws.maps_chi.row_ssign, ws.maps_chi.row_pair, group_phis, false)
        _cuda_factored_y!(ychi, achi, ym.mult_U_re, ym.mult_U_im, ym.mult_V_re,
            ym.mult_V_im, theta, rchi, cchi)
        _cuda_ztranslate_fixed_m!(zchi, ychi, zcol, P_active, P_active)
        # LH local row mix: operand gathers land in the (now free) yphi/ychi slabs.
        # Local LH rows are linear in r, so compact plans retain one unit-radius
        # row pair and scale it by the class radius.
        _gather_rows!(yphi, zchi, ws.maps_phi.row_pair)
        _gather_rows!(ychi, zchi, ws.maps_chi.row_up)
        cphi .= zphi .+ (wp.lh_arow_unit .* r) .* yphi
        cchi .= zchi .+ (wp.lh_brow_unit .* r) .* ychi
        _cuda_factored_y!(rchi, cchi, ym.loc_U_re, ym.loc_U_im, ym.loc_V_re,
            ym.loc_V_im, theta, achi, ychi)
        _rotate_z_scatter_accumulate!(state.locals.chi, rchi, ws.chi_flat_idx, tgt_cols,
            ws.maps_chi.row_m, ws.maps_chi.row_ssign, ws.maps_chi.row_pair, group_phis)
        ret_phi = cphi
    end
    _cuda_factored_y!(rphi, ret_phi, ym.loc_U_re, ym.loc_U_im, ym.loc_V_re, ym.loc_V_im,
        theta, aphi, yphi)
    _rotate_z_scatter_accumulate!(state.locals.phi, rphi, ws.phi_flat_idx, tgt_cols,
        ws.maps_phi.row_m, ws.maps_phi.row_ssign, ws.maps_phi.row_pair, group_phis)
    return state
end

# Whole-pass chunked execution (task 023b optimize phase). The per-class path
# above is launch-bound at production stencils (~3000 nonempty classes × ~7
# launches per channel measured 110-360 ms M2L on H200); every stage is (or can
# be) per-column parameterized, so processing routes in wide chunks with
# per-column class indirection collapses launches to ~7 per channel per chunk.
# The per-class path is retained as the separately testable reference
# (FACTORED_CUDA_WHOLE_PASS[] = false).
const FACTORED_CUDA_WHOLE_PASS = Ref(true)
const FACTORED_CUDA_CHUNK = Ref(1 << 14)

# Per-column variant of the forward y half: theta gathered per column.
function _cuda_factored_y_phase_cols_kernel!(gre, gim, in_slab, V_re, V_im, thetas)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    ndof = size(gre, 1)
    idx > ndof * size(gre, 2) && return nothing
    row = (idx - 1) % ndof + 1
    j = (idx - 1) ÷ ndof + 1
    n = _cuda_degree_of_row(row)
    row0 = n * n
    k = row - row0
    d = 2 * n + 1
    off = ymode_offset(n)
    TF = eltype(gre)
    gr = zero(TF)
    gi = zero(TF)
    @inbounds begin
        for q in 1:d
            x = in_slab[row0 + q, j]
            v = off + (q - 1) * d + k
            gr += V_re[v] * x
            gi += V_im[v] * x
        end
        s, c = sincos(TF(k - n - 1) * thetas[j])
        gre[row, j] = c * gr - s * gi
        gim[row, j] = s * gr + c * gi
    end
    return nothing
end

function _cuda_factored_y_cols!(out_slab, in_slab, U_re, U_im, V_re, V_im, thetas,
        scratch_re, scratch_im)
    n_el = length(out_slab)
    n_el == 0 && return out_slab
    threads = 256
    blocks = cld(n_el, threads)
    CUDA.@cuda threads=threads blocks=blocks _cuda_factored_y_phase_cols_kernel!(
        scratch_re, scratch_im, in_slab, V_re, V_im, thetas,
    )
    CUDA.@cuda threads=threads blocks=blocks _cuda_factored_y_out_kernel!(
        out_slab, scratch_re, scratch_im, U_re, U_im,
    )
    return out_slab
end

# Per-column variant of the fixed-m z translation: each column reads its own
# class's flat block table.
function _cuda_ztranslate_fixed_m_cols_kernel!(out_slab, in_slab, z_flat, cls,
        P_loop, P_block)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    ndof = size(out_slab, 1)
    idx > ndof * size(out_slab, 2) && return nothing
    row = (idx - 1) % ndof + 1
    j = (idx - 1) ÷ ndof + 1
    n = _cuda_degree_of_row(row)
    k = row - n * n
    m = k == 1 ? 0 : (k >> 1)
    w = P_block - m + 1
    base = m2l_z_block_offset(m, P_block) + (n - m) + 1
    TF = eltype(out_slab)
    acc = zero(TF)
    @inbounds begin
        kc = Int(cls[j])
        for np in m:P_loop
            acc += z_flat[base + (np - m) * w, kc] * in_slab[np * np + k, j]
        end
        out_slab[row, j] = acc
    end
    return nothing
end

function _cuda_ztranslate_fixed_m_cols!(out_slab, in_slab, z_flat, cls,
        P_loop::Int, P_block::Int)
    n_el = length(out_slab)
    n_el == 0 && return out_slab
    threads = 256
    blocks = cld(n_el, threads)
    CUDA.@cuda threads=threads blocks=blocks _cuda_ztranslate_fixed_m_cols_kernel!(
        out_slab, in_slab, z_flat, cls, P_loop, P_block,
    )
    return out_slab
end

# Build the whole-pass bundle at cache construction: device per-class geometry
# tables, per-chunk column-parameter gather targets, unit local LH rows (linear
# in r, scaled per column like the concat plan), and chunk-width stage slabs.
function _cuda_factored_whole_pass_setup!(plan::ResidentM2LFactoredPlan, ::Type{TF},
        basis_info::OperatorBasisInfo{B,LH}) where {TF,B,LH}
    W = max(min(FACTORED_CUDA_CHUNK[], length(plan.route_class)), 1)
    P_phi = basis_info.orders.P_phi
    P_active = basis_info.orders.P_active
    ndof_phi = degree_major_dof(P_phi)
    ndof_chi = LH ? degree_major_dof(P_active) : 0
    exemplar = CUDA.zeros(TF, 0)
    lh_arow_unit, lh_brow_unit = LH ?
        _resident_lh_rows_like(exemplar, TF, P_phi, P_active, one(TF), :local) :
        (nothing, nothing)
    mkphi() = CUDA.zeros(TF, ndof_phi, W)
    mkchi() = CUDA.zeros(TF, ndof_chi, LH ? W : 0)
    plan.whole_pass[] = (;
        chunk=W,
        d_class_phi=CUDA.CuArray{TF}(plan.class_phi::Vector{TF}),
        d_class_theta=CUDA.CuArray{TF}(plan.class_theta::Vector{TF}),
        d_class_r=CUDA.CuArray{TF}(plan.class_r::Vector{TF}),
        col_phi=CUDA.zeros(TF, W), col_theta=CUDA.zeros(TF, W), col_r=CUDA.zeros(TF, W),
        lh_arow_unit, lh_brow_unit,
        aphi=mkphi(), yphi=mkphi(), zphi=mkphi(), rphi=mkphi(), cphi=mkphi(),
        achi=mkchi(), ychi=mkchi(), zchi=mkchi(), rchi=mkchi(), cchi=mkchi(),
    )
    return plan
end

function _launch_resident_m2l_factored_whole!(state::DeviceResidentRadixState{TF,B,LH},
        ws::ResidentOperatorWorkspace{TF,B,LH}, plan::ResidentM2LFactoredPlan,
        wp::NamedTuple) where {TF,B,LH}
    fill!(state.locals.phi, zero(TF))
    LH && fill!(state.locals.chi, zero(TF))
    n_routes = state.counts.n_routes
    n_routes == 0 && return state
    P_phi = ws.basis_info.orders.P_phi
    P_active = ws.basis_info.orders.P_active
    ym = plan.ym_flat
    W = wp.chunk
    @inbounds for c0 in 1:W:n_routes
        cols = c0:min(c0 + W - 1, n_routes)
        n = length(cols)
        cls = view(plan.route_class, cols)
        phis = view(wp.col_phi, 1:n)
        thetas = view(wp.col_theta, 1:n)
        _gather_values!(phis, wp.d_class_phi, cls)
        _gather_values!(thetas, wp.d_class_theta, cls)
        src_cols = view(state.route_sources, cols)
        tgt_cols = view(state.route_targets, cols)
        aphi = _matrix_col_view(wp.aphi, n); yphi = _matrix_col_view(wp.yphi, n)
        zphi = _matrix_col_view(wp.zphi, n); rphi = _matrix_col_view(wp.rphi, n)
        cphi = _matrix_col_view(wp.cphi, n)
        _gather_rotate_z!(aphi, state.multipoles.phi, ws.phi_flat_idx, src_cols,
            ws.maps_phi.row_m, ws.maps_phi.row_ssign, ws.maps_phi.row_pair, phis, false)
        _cuda_factored_y_cols!(yphi, aphi, ym.mult_U_re, ym.mult_U_im, ym.mult_V_re,
            ym.mult_V_im, thetas, rphi, cphi)
        _cuda_ztranslate_fixed_m_cols!(zphi, yphi, plan.z_flat, cls, P_phi, P_active)
        ret_phi = zphi
        if LH
            rs = view(wp.col_r, 1:n)
            _gather_values!(rs, wp.d_class_r, cls)
            achi = _matrix_col_view(wp.achi, n); ychi = _matrix_col_view(wp.ychi, n)
            zchi = _matrix_col_view(wp.zchi, n); rchi = _matrix_col_view(wp.rchi, n)
            cchi = _matrix_col_view(wp.cchi, n)
            _gather_rotate_z!(achi, state.multipoles.chi, ws.chi_flat_idx, src_cols,
                ws.maps_chi.row_m, ws.maps_chi.row_ssign, ws.maps_chi.row_pair, phis, false)
            _cuda_factored_y_cols!(ychi, achi, ym.mult_U_re, ym.mult_U_im, ym.mult_V_re,
                ym.mult_V_im, thetas, rchi, cchi)
            _cuda_ztranslate_fixed_m_cols!(zchi, ychi, plan.z_flat, cls, P_active, P_active)
            # LH local rows are linear in r (lamb_helmholtz_local_coeffs!), so the
            # unit-radius rows scale per column (same trick as the concat plan).
            _gather_rows!(yphi, zchi, ws.maps_phi.row_pair)
            _gather_rows!(ychi, zchi, ws.maps_chi.row_up)
            rs_row = transpose(rs)
            cphi .= zphi .+ (wp.lh_arow_unit .* rs_row) .* yphi
            cchi .= zchi .+ (wp.lh_brow_unit .* rs_row) .* ychi
            _cuda_factored_y_cols!(rchi, cchi, ym.loc_U_re, ym.loc_U_im, ym.loc_V_re,
                ym.loc_V_im, thetas, achi, ychi)
            _rotate_z_scatter_accumulate!(state.locals.chi, rchi, ws.chi_flat_idx,
                tgt_cols, ws.maps_chi.row_m, ws.maps_chi.row_ssign,
                ws.maps_chi.row_pair, phis)
            ret_phi = cphi
        end
        _cuda_factored_y_cols!(rphi, ret_phi, ym.loc_U_re, ym.loc_U_im, ym.loc_V_re,
            ym.loc_V_im, thetas, aphi, yphi)
        _rotate_z_scatter_accumulate!(state.locals.phi, rphi, ws.phi_flat_idx, tgt_cols,
            ws.maps_phi.row_m, ws.maps_phi.row_ssign, ws.maps_phi.row_pair, phis)
    end
    return state
end

# Device dispatch for the factored plan launch (the host method in
# translate_batched.jl keeps the generic signature): whole-pass chunked execution
# by default, or the per-class reference path over contiguous class ranges.
function _launch_resident_m2l_factored_plan!(state::DeviceResidentRadixState{TF,B,LH},
        ws::ResidentOperatorWorkspace{TF,B,LH},
        plan::ResidentM2LFactoredPlan{<:CUDA.AnyCuArray}) where {TF,B,LH}
    wp = plan.whole_pass[]
    if wp !== nothing && FACTORED_CUDA_WHOLE_PASS[]
        return _launch_resident_m2l_factored_whole!(state, ws, plan, wp::NamedTuple)
    end
    fill!(state.locals.phi, zero(TF))
    LH && fill!(state.locals.chi, zero(TF))
    wp isa NamedTuple || throw(ArgumentError(
        "CUDA factored per-class reference requires the compact whole-pass scratch bundle"))
    starts = plan.class_starts
    counts = plan.host_class_counts::Vector{Int32}
    @inbounds for k in eachindex(counts)
        n = Int(counts[k])
        n == 0 && continue
        cols = (starts[k] + 1):(starts[k] + n)
        _cuda_factored_m2l_class_apply!(state, ws, plan, wp, k, cols)
    end
    return state
end

#------- device-resident precomputed-y M2L (Matrix Operator Refactor, task 023d) -------#
#
# CUDA mirror of the 023c host precomputed-y stage. The chain and route layout are
# identical to the factored device path above (device route emission is
# offset-class-major and contiguous; classes ARE the accepted offsets), except the
# two-kernel V -> e^{i nu theta} -> U factored y application is replaced by ONE
# kernel reading the per-angle precomputed real block M_n(theta) = U D(theta) V:
# each column looks up its angle class through the per-offset angle table and
# contracts the degree's (2n+1)x(2n+1) block from the flat y table. Exact integer
# polar-angle classes deduplicate the offsets' thetas, so the flat y storage is
# ymode-block layout x nangles instead of per-offset. The fixed-m z translation
# reads the per-offset z_flat column through the existing per-column kernel, and
# the LH local rows use the unit-radius rows scaled per column (the concat trick).
# All operator tables upload once at cache construction; the recurring step
# performs no device allocation on this stage.

const PRECOMPUTED_CUDA_WHOLE_PASS = Ref(true)
const PRECOMPUTED_CUDA_CHUNK = Ref(1 << 14)

# One precomputed-y application on a degree-major slab: per (row, column), the
# column's angle selects the flat block column; entry [k, q] of degree n lives at
# ymode_offset(n) + (q - 1) * (2n + 1) + k (same layout as _ymode_real_blocks).
function _cuda_precomputed_y_cols_kernel!(out_slab, in_slab, y_flat, angles)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    ndof = size(out_slab, 1)
    idx > ndof * size(out_slab, 2) && return nothing
    row = (idx - 1) % ndof + 1
    j = (idx - 1) ÷ ndof + 1
    n = _cuda_degree_of_row(row)
    row0 = n * n
    k = row - row0
    d = 2 * n + 1
    off = ymode_offset(n)
    TF = eltype(out_slab)
    acc = zero(TF)
    @inbounds begin
        a = Int(angles[j])
        for q in 1:d
            acc += y_flat[off + (q - 1) * d + k, a] * in_slab[row0 + q, j]
        end
        out_slab[row, j] = acc
    end
    return nothing
end

function _cuda_precomputed_y_cols!(out_slab, in_slab, y_flat, angles)
    n_el = length(out_slab)
    n_el == 0 && return out_slab
    threads = 256
    blocks = cld(n_el, threads)
    CUDA.@cuda threads=threads blocks=blocks _cuda_precomputed_y_cols_kernel!(
        out_slab, in_slab, y_flat, angles,
    )
    return out_slab
end

# Device mirror of _refresh_precomputed_y_m2l_routes!: per-offset counts by
# histogram, one pinned host download (uncounted, like the existing host_scalar
# staging), and a host prefix-sum. Device routes keep the emission's
# offset-class-major contiguous order, so no angle-major repack exists here;
# offset_starts holds 1-based route-order starts (+1 sentinel) and angle_counts
# carries the derived per-angle totals for diagnostics. angle_starts is
# meaningless on the device layout and stays untouched.
function _cuda_refresh_precomputed_y_m2l_routes!(plan::ResidentM2LPrecomputedYPlan,
        route_class, n_routes::Int)
    counts = plan.class_counts
    fill!(counts, Int32(0))
    if n_routes > 0
        threads = 256
        blocks = cld(n_routes, threads)
        CUDA.@cuda threads=threads blocks=blocks _cuda_class_histogram_kernel!(
            counts, route_class, n_routes,
        )
    end
    host_counts = plan.host_class_counts::Vector{Int32}
    copyto!(host_counts, counts)
    starts = plan.offset_starts
    starts[1] = 1
    @inbounds for k in eachindex(host_counts)
        c = Int(host_counts[k])
        plan.offset_counts[k] = c
        starts[k + 1] = starts[k] + c
    end
    starts[end] - 1 == n_routes ||
        throw(AssertionError("precomputed-y offset classes do not partition the device routes"))
    fill!(plan.angle_counts, 0)
    @inbounds for k in eachindex(host_counts)
        plan.angle_counts[plan.offset_to_angle[k]] += Int(host_counts[k])
    end
    return plan
end

# Build the whole-pass bundle at cache construction: device per-offset geometry
# and angle tables, per-chunk column-parameter gather targets, unit local LH rows
# (linear in r, scaled per column), and chunk-width stage slabs. The precomputed-y
# stage needs no phase scratch, so the slab set matches the host stage names.
function _cuda_precomputed_y_whole_pass_setup!(plan::ResidentM2LPrecomputedYPlan,
        ::Type{TF}, basis_info::OperatorBasisInfo{B,LH}) where {TF,B,LH}
    W = max(min(PRECOMPUTED_CUDA_CHUNK[], length(plan.route_class)), 1)
    P_phi = basis_info.orders.P_phi
    P_active = basis_info.orders.P_active
    ndof_phi = degree_major_dof(P_phi)
    ndof_chi = LH ? degree_major_dof(P_active) : 0
    exemplar = CUDA.zeros(TF, 0)
    lh_arow_unit, lh_brow_unit = LH ?
        _resident_lh_rows_like(exemplar, TF, P_phi, P_active, one(TF), :local) :
        (nothing, nothing)
    mkphi() = CUDA.zeros(TF, ndof_phi, W)
    mkchi() = CUDA.zeros(TF, ndof_chi, LH ? W : 0)
    plan.whole_pass[] = (;
        chunk=W,
        d_offset_phi=CUDA.CuArray{TF}(plan.offset_phis),
        d_offset_r=CUDA.CuArray{TF}(plan.offset_rs::Vector{TF}),
        d_offset_angle=CUDA.CuArray{Int32}(Int32.(plan.offset_to_angle)),
        col_phi=CUDA.zeros(TF, W), col_r=CUDA.zeros(TF, W),
        col_angle=CUDA.zeros(Int32, W), col_cls=CUDA.zeros(Int32, W),
        lh_arow_unit, lh_brow_unit,
        aphi=mkphi(), yphi=mkphi(), zphi=mkphi(), rphi=mkphi(), cphi=mkphi(),
        achi=mkchi(), ychi=mkchi(), zchi=mkchi(), rchi=mkchi(), cchi=mkchi(),
    )
    return plan
end

# Shared per-column stage chain over one contiguous route range: gather + Z_phi,
# one precomputed-y kernel, per-column fixed-m z translation, LH row mix, one
# local precomputed-y kernel, inverse-Z_phi scatter/accumulate. `phis`, `angles`,
# `cls`, and (LH) `rs` are per-column parameter views the caller fills — gathered
# by the whole-pass driver, constant-filled by the per-class reference driver —
# so both drivers exercise the identical mathematical stage boundaries.
function _cuda_precomputed_y_apply_cols!(state::DeviceResidentRadixState{TF,B,LH},
        ws::ResidentOperatorWorkspace{TF,B,LH}, plan::ResidentM2LPrecomputedYPlan,
        wp::NamedTuple, cols::UnitRange{Int}, phis, angles, cls, rs) where {TF,B,LH}
    n = length(cols)
    P_phi = ws.basis_info.orders.P_phi
    P_active = ws.basis_info.orders.P_active
    src_cols = view(state.route_sources, cols)
    tgt_cols = view(state.route_targets, cols)
    aphi = _matrix_col_view(wp.aphi, n); yphi = _matrix_col_view(wp.yphi, n)
    zphi = _matrix_col_view(wp.zphi, n); rphi = _matrix_col_view(wp.rphi, n)
    cphi = _matrix_col_view(wp.cphi, n)
    _gather_rotate_z!(aphi, state.multipoles.phi, ws.phi_flat_idx, src_cols,
        ws.maps_phi.row_m, ws.maps_phi.row_ssign, ws.maps_phi.row_pair, phis, false)
    _cuda_precomputed_y_cols!(yphi, aphi, plan.y_flat_mult, angles)
    _cuda_ztranslate_fixed_m_cols!(zphi, yphi, plan.z_flat, cls, P_phi, P_active)
    ret_phi = zphi
    if LH
        achi = _matrix_col_view(wp.achi, n); ychi = _matrix_col_view(wp.ychi, n)
        zchi = _matrix_col_view(wp.zchi, n); rchi = _matrix_col_view(wp.rchi, n)
        cchi = _matrix_col_view(wp.cchi, n)
        _gather_rotate_z!(achi, state.multipoles.chi, ws.chi_flat_idx, src_cols,
            ws.maps_chi.row_m, ws.maps_chi.row_ssign, ws.maps_chi.row_pair, phis, false)
        _cuda_precomputed_y_cols!(ychi, achi, plan.y_flat_mult, angles)
        _cuda_ztranslate_fixed_m_cols!(zchi, ychi, plan.z_flat, cls, P_active, P_active)
        # LH local rows are linear in r (lamb_helmholtz_local_coeffs!), so the
        # unit-radius rows scale per column; operand gathers land in the now-free
        # yphi/ychi slabs.
        _gather_rows!(yphi, zchi, ws.maps_phi.row_pair)
        _gather_rows!(ychi, zchi, ws.maps_chi.row_up)
        rs_row = transpose(rs)
        cphi .= zphi .+ (wp.lh_arow_unit .* rs_row) .* yphi
        cchi .= zchi .+ (wp.lh_brow_unit .* rs_row) .* ychi
        _cuda_precomputed_y_cols!(rchi, cchi, plan.y_flat_loc, angles)
        _rotate_z_scatter_accumulate!(state.locals.chi, rchi, ws.chi_flat_idx,
            tgt_cols, ws.maps_chi.row_m, ws.maps_chi.row_ssign,
            ws.maps_chi.row_pair, phis)
        ret_phi = cphi
    end
    _cuda_precomputed_y_cols!(rphi, ret_phi, plan.y_flat_loc, angles)
    _rotate_z_scatter_accumulate!(state.locals.phi, rphi, ws.phi_flat_idx, tgt_cols,
        ws.maps_phi.row_m, ws.maps_phi.row_ssign, ws.maps_phi.row_pair, phis)
    return state
end

# Whole-pass chunked execution: per-column phi/angle/r parameters are gathered
# from the per-offset device tables through the route classes, so launches scale
# with chunks, not offset or angle classes (~6 per channel per chunk).
function _launch_resident_m2l_precomputed_y_whole!(
        state::DeviceResidentRadixState{TF,B,LH},
        ws::ResidentOperatorWorkspace{TF,B,LH}, plan::ResidentM2LPrecomputedYPlan,
        wp::NamedTuple; clear_locals::Bool=true) where {TF,B,LH}
    clear_locals && fill!(state.locals.phi, zero(TF))
    clear_locals && LH && fill!(state.locals.chi, zero(TF))
    n_routes = state.counts.n_routes
    n_routes == 0 && return state
    W = wp.chunk
    @inbounds for c0 in 1:W:n_routes
        cols = c0:min(c0 + W - 1, n_routes)
        n = length(cols)
        cls = view(plan.route_class, cols)
        phis = view(wp.col_phi, 1:n)
        angles = view(wp.col_angle, 1:n)
        _gather_values!(phis, wp.d_offset_phi, cls)
        _gather_values!(angles, wp.d_offset_angle, cls)
        rs = view(wp.col_r, 1:n)
        LH && _gather_values!(rs, wp.d_offset_r, cls)
        _cuda_precomputed_y_apply_cols!(state, ws, plan, wp, cols, phis, angles,
            cls, rs)
    end
    return state
end

# Device dispatch for the precomputed-y plan launch (the host method in
# translate_batched.jl keeps the generic signature): whole-pass chunked execution
# by default, or the per-class reference path over contiguous offset-class
# ranges with constant-filled column parameters (sub-chunked at the bundle
# width, so any class occupancy is supported).
function _launch_resident_m2l_precomputed_y_plan!(
        state::DeviceResidentRadixState{TF,B,LH},
        ws::ResidentOperatorWorkspace{TF,B,LH},
        plan::ResidentM2LPrecomputedYPlan{TF,<:Any,<:CUDA.AnyCuArray};
        clear_locals::Bool=true) where {TF,B,LH}
    wp = plan.whole_pass[]
    wp isa NamedTuple || throw(ArgumentError(
        "CUDA precomputed-y M2L requires the whole-pass scratch bundle"))
    if PRECOMPUTED_CUDA_WHOLE_PASS[]
        return _launch_resident_m2l_precomputed_y_whole!(state, ws, plan, wp;
            clear_locals)
    end
    clear_locals && fill!(state.locals.phi, zero(TF))
    clear_locals && LH && fill!(state.locals.chi, zero(TF))
    starts = plan.offset_starts
    counts = plan.host_class_counts::Vector{Int32}
    offset_rs = plan.offset_rs::Vector{TF}
    W = wp.chunk
    @inbounds for k in eachindex(counts)
        remaining = Int(counts[k])
        remaining == 0 && continue
        phi = plan.offset_phis[k]
        angle = Int32(plan.offset_to_angle[k])
        r = offset_rs[k]
        c0 = starts[k]
        while remaining > 0
            n = min(remaining, W)
            cols = c0:(c0 + n - 1)
            phis = view(wp.col_phi, 1:n)
            angles = view(wp.col_angle, 1:n)
            cls = view(wp.col_cls, 1:n)
            rs = view(wp.col_r, 1:n)
            fill!(phis, phi)
            fill!(angles, angle)
            fill!(cls, Int32(k))
            LH && fill!(rs, r)
            _cuda_precomputed_y_apply_cols!(state, ws, plan, wp, cols, phis,
                angles, cls, rs)
            c0 += n
            remaining -= n
        end
    end
    return state
end

#------- device-resident dense-translation M2L (Matrix Operator Refactor, task 023f) -------#
#
# CUDA mirror of the 023e host DenseTranslationM2L strategy: one complete real
# degree-major stacked-[phi;chi] `D x D` coefficient operator per accepted
# displacement class, applied as gather -> per-class GEMM -> scatter-add. The
# oracle-built operators (identical construction to the host plan) upload once at
# cache construction into a packed `D x D x nclasses` device array; device route
# emission is offset-class-major and contiguous (see _cuda_generate_radix_routes!),
# so class k's columns are one contiguous range of route_sources/route_targets and
# the per-step refresh is a histogram plus one pinned counts download (no repack).
#
# The scatter is atomic because multiple source routes may target the same local
# column. Three independently selectable drivers: the fused per-route kernel
# (DENSE_CUDA_FUSED[], gather/matvec/scatter in one launch, no slabs or cuBLAS)
# and two GEMM drivers sharing the gather/scatter kernels and the chunk-width
# slabs — the whole-pass driver (default) gathers/scatters once per route chunk
# with per-class GEMMs over the chunk, and the per-class reference driver
# (DENSE_CUDA_WHOLE_PASS[] = false) gathers/GEMMs/scatters per class, matching
# the host _launch_resident_m2l_dense_plan! stage boundaries. Dense classes each
# carry a full dense operator, so unlike factored/precomputed-y there is no
# flatten-and-per-column trick; the GEMM stays genuinely O(D^2 * width) per class.
const DENSE_CUDA_WHOLE_PASS = Ref(true)
const DENSE_CUDA_CHUNK = Ref(1 << 14)
# Fused per-route kernel driver (optimize phase): takes precedence over the
# GEMM drivers when enabled. Default true per the H200 A/B measurement
# (2026-07-22, job 12871790): fastest dense driver in every supported finite
# measured config. Float32/P=12 dense operator materialization is rejected as
# non-finite. Example: 0.042 / 1.0 / 4.5 ms M2L at P=4/8/12 (F64, LH off,
# n=2e4) vs 12.4 / 26.8 / 28.5 ms for the whole-pass per-class GEMM driver,
# which is launch-bound at 1e3+ gemms per step.
const DENSE_CUDA_FUSED = Ref(true)

# Persistent-CTA cap for the fused dense kernel (task 028 lever 2). The kernel
# originally launched one 32-thread block per route: at n=1e6/ell=5 that is
# 31,307,680 blocks for 20.28 ms of leaf M2L, i.e. ~1.5 G blocks/s, which is
# block-dispatch rate rather than compute or bandwidth (a Float32/Float64 A/B
# moved the stage 22.41 -> 23.55 ms, ruling both out). Each block now
# grid-strides over many routes, so the block count is bounded by this cap
# instead of the route count -- ~1900x less dispatch at the leaf level. At
# P=4 the block is 32 threads, so 16384 blocks is 524288 threads against an
# H200's 132 SMs x 2048 = 270336 thread capacity: still oversubscribed.
const DENSE_CUDA_FUSED_MAX_BLOCKS = Ref(16384)

# Operator-tiled hierarchical dense M2L (task 028 cycle 2). The fused kernel
# re-reads its route's full D x D operator from L2 on every route; the derisk
# atomic/store/no-store A/B (job 13015316) proved the leaf M2L is bound by
# exactly those loads plus the matvec, not by atomics. Window routes are
# class-sorted, so a block can stage one class's operator in shared memory —
# with both level diagonals folded into the tile — and stream that class's
# routes through it. The tiled path is taken when it fits in default dynamic
# shared memory and the window is large enough to amortize the tile loads;
# small windows (coarse levels) keep the plain fused kernel.
const DENSE_CUDA_TILED = Ref(true)
const DENSE_CUDA_TILED_MIN_ROUTES = Ref(65536)

# Grid-stride cap for the warp-per-pair nearfield kernel (task 028 lever 1),
# same role as DENSE_CUDA_FUSED_MAX_BLOCKS above: blocks = min(cld(npairs,
# warps_per_block), this cap), and each warp walks pairs `total_warps` apart.
# At n=1e6/ell=5 there are 5,189,728 direct cell-pairs of ~30x30 bodies.
const DIRECT_CUDA_MAX_BLOCKS = Ref(16384)

# Gather the stacked degree-major [phi; chi] slab column j from the source column
# `src_cols[j]`, mapping degree-major row i to flat storage through phi/chi_flat_idx.
function _cuda_dense_gather_kernel!(slab, phi, chi, phi_flat_idx, chi_flat_idx,
        src_cols, ndof_phi, ::Val{LH}) where LH
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    ndof = size(slab, 1)
    idx > ndof * size(slab, 2) && return nothing
    row = (idx - 1) % ndof + 1
    j = (idx - 1) ÷ ndof + 1
    @inbounds begin
        col = src_cols[j]
        if row <= ndof_phi
            slab[row, j] = phi[phi_flat_idx[row], col]
        elseif LH
            slab[row, j] = chi[chi_flat_idx[row - ndof_phi], col]
        end
    end
    return nothing
end

function _cuda_dense_gather!(slab, source::FlatCoefficientBuffer, phi_flat_idx,
        chi_flat_idx, src_cols, ndof_phi::Int, lh::Val)
    n_el = length(slab)
    n_el == 0 && return slab
    threads = 256
    blocks = cld(n_el, threads)
    CUDA.@cuda threads=threads blocks=blocks _cuda_dense_gather_kernel!(
        slab, source.phi, source.chi, phi_flat_idx, chi_flat_idx, src_cols,
        ndof_phi, lh,
    )
    return slab
end

# Atomic scatter-add of the stacked degree-major [phi; chi] slab into the target
# local columns `tgt_cols`. Atomic because multiple routes may share a target.
function _cuda_dense_scatter_kernel!(phi, chi, slab, phi_flat_idx, chi_flat_idx,
        tgt_cols, ndof_phi, ::Val{LH}) where LH
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    ndof = size(slab, 1)
    idx > ndof * size(slab, 2) && return nothing
    row = (idx - 1) % ndof + 1
    j = (idx - 1) ÷ ndof + 1
    @inbounds begin
        col = tgt_cols[j]
        if row <= ndof_phi
            CUDA.@atomic phi[phi_flat_idx[row], col] += slab[row, j]
        elseif LH
            CUDA.@atomic chi[chi_flat_idx[row - ndof_phi], col] += slab[row, j]
        end
    end
    return nothing
end

function _cuda_dense_scatter_add!(target::FlatCoefficientBuffer, slab, phi_flat_idx,
        chi_flat_idx, tgt_cols, ndof_phi::Int, lh::Val)
    n_el = length(slab)
    n_el == 0 && return target
    threads = 256
    blocks = cld(n_el, threads)
    CUDA.@cuda threads=threads blocks=blocks _cuda_dense_scatter_kernel!(
        target.phi, target.chi, slab, phi_flat_idx, chi_flat_idx, tgt_cols,
        ndof_phi, lh,
    )
    return target
end

# Device mirror of _refresh_dense_m2l_routes!: per-class counts by histogram, one
# pinned host download (uncounted), host prefix-sum into the 1-based class_starts
# (+1 sentinel), class-capacity validation, and the partition assertion. Class
# contiguity is guaranteed by the class-major device route emission, so no
# source/target repack exists here.
function _cuda_refresh_dense_m2l_routes!(plan::ResidentM2LDenseCUDAPlan, route_class,
        n_routes::Int)
    0 <= n_routes <= length(route_class) || throw(ArgumentError(
        "dense M2L route count $n_routes exceeds plan capacity $(length(route_class))"))
    counts = plan.class_counts
    fill!(counts, Int32(0))
    if n_routes > 0
        threads = 256
        blocks = cld(n_routes, threads)
        CUDA.@cuda threads=threads blocks=blocks _cuda_class_histogram_kernel!(
            counts, route_class, n_routes,
        )
    end
    host_counts = plan.host_class_counts::Vector{Int32}
    copyto!(host_counts, counts)
    starts = plan.class_starts
    starts[1] = 1
    @inbounds for k in eachindex(host_counts)
        c = Int(host_counts[k])
        c <= plan.class_capacities[k] || throw(AssertionError(
            "dense M2L class $k count $c exceeds capacity $(plan.class_capacities[k])"))
        starts[k + 1] = starts[k] + c
    end
    starts[end] - 1 == n_routes ||
        throw(AssertionError("dense M2L classes do not partition the device routes"))
    return plan
end

# One dense GEMM over a slab-local column range: dst_base[:, lo:hi] =
# operators[:, :, k] * src_base[:, lo:hi]. The class slice of the packed operator
# array is a contiguous (D, D) strided view and `dst_base`/`src_base` are the
# D x W base slabs, so the operands stay on the classic zero-workspace cuBLAS
# gemm. The alpha/beta scalars are the plan's construction-time device staging
# (length-1 CuVectors): CUDA.jl runs cuBLAS in CUBLAS_POINTER_MODE_DEVICE and
# `mul!` re-uploads both scalars through a fresh device `CuRef` on every call
# (2 * sizeof(TF) bytes), which broke the zero-steady-state-allocation contract
# at 1e3+ class GEMMs per step; passing preallocated device arrays converts by
# pointer without allocating.
@inline function _cuda_dense_class_gemm!(dst_base, operators, k::Int, src_base,
        lo::Int, hi::Int, alpha, beta)
    CUDA.CUBLAS.gemm!('N', 'N', alpha, (@view operators[:, :, k]),
        (@view src_base[:, lo:hi]), beta, (@view dst_base[:, lo:hi]))
    return nothing
end

# Fused whole-M2L kernel (task 023f optimize phase): one block per route performs
# gather -> D x D matvec -> atomic scatter-add without slabs or cuBLAS launches.
# The per-class GEMM drivers issue one gemm per nonempty class per chunk (1e3+
# launches per step), which the H200 baseline measured as launch-bound (dense M2L
# nearly flat in P); this kernel reduces the whole pass to a single launch. The
# source column is staged in dynamic shared memory (D elements); operator reads
# `ops[r, i, k]` are coalesced across the row-threads `r`, and per-class operator
# reuse across routes is served by L2.
function _cuda_dense_fused_m2l_kernel!(loc_phi, loc_chi, ops, route_class,
        route_sources, route_targets, mp_phi, mp_chi, phi_flat_idx, chi_flat_idx,
        ndof_phi, ::Val{LH}) where LH
    T = eltype(ops)
    D = size(ops, 1)
    j = blockIdx().x
    tid = threadIdx().x
    nthreads = blockDim().x
    shm = CUDA.CuDynamicSharedArray(T, D)
    @inbounds begin
        src_col = route_sources[j]
        k = Int(route_class[j])
        tgt_col = route_targets[j]
        i = tid
        while i <= D
            if i <= ndof_phi
                shm[i] = mp_phi[phi_flat_idx[i], src_col]
            elseif LH
                shm[i] = mp_chi[chi_flat_idx[i - ndof_phi], src_col]
            else
                shm[i] = zero(T)
            end
            i += nthreads
        end
        CUDA.sync_threads()
        r = tid
        while r <= D
            acc = zero(T)
            for i in 1:D
                acc += ops[r, i, k] * shm[i]
            end
            if r <= ndof_phi
                CUDA.@atomic loc_phi[phi_flat_idx[r], tgt_col] += acc
            elseif LH
                CUDA.@atomic loc_chi[chi_flat_idx[r - ndof_phi], tgt_col] += acc
            end
            r += nthreads
        end
    end
    return nothing
end

function _launch_resident_m2l_dense_fused!(state::DeviceResidentRadixState{TF,B,LH},
        ws::ResidentOperatorWorkspace{TF,B,LH},
        plan::ResidentM2LDenseCUDAPlan) where {TF,B,LH}
    fill!(state.locals.phi, zero(TF))
    LH && fill!(state.locals.chi, zero(TF))
    n_routes = state.counts.n_routes
    n_routes == 0 && return state
    threads = min(256, cld(plan.ndof, 32) * 32)
    shmem = plan.ndof * sizeof(TF)
    CUDA.@cuda threads=threads blocks=n_routes shmem=shmem _cuda_dense_fused_m2l_kernel!(
        state.locals.phi, state.locals.chi, plan.operators, plan.route_class,
        state.route_sources, state.route_targets, state.multipoles.phi,
        state.multipoles.chi, ws.phi_flat_idx, ws.chi_flat_idx, plan.ndof_phi,
        Val(LH),
    )
    return state
end

# Whole-pass driver (production default): iterate route chunks; gather once, GEMM
# per class overlapping the chunk (classes are contiguous, so a running class
# cursor advances monotonically), scatter once. Launches scale with chunks plus
# nonempty classes, not with 3 * nclasses.
function _launch_resident_m2l_dense_whole!(state::DeviceResidentRadixState{TF,B,LH},
        ws::ResidentOperatorWorkspace{TF,B,LH}, plan::ResidentM2LDenseCUDAPlan,
        wp::NamedTuple) where {TF,B,LH}
    fill!(state.locals.phi, zero(TF))
    LH && fill!(state.locals.chi, zero(TF))
    n_routes = state.counts.n_routes
    n_routes == 0 && return state
    starts = plan.class_starts        # 1-based, length nclasses+1, starts[end]=n_routes+1
    W = wp.chunk
    ndof_phi = plan.ndof_phi
    kcur = 1
    @inbounds for c0 in 1:W:n_routes
        n = min(W, n_routes - c0 + 1)
        chi_hi = c0 + n - 1
        _cuda_dense_gather!(_matrix_col_view(plan.src_slab, n), state.multipoles,
            ws.phi_flat_idx, ws.chi_flat_idx, view(state.route_sources, c0:chi_hi),
            ndof_phi, Val(LH))
        while starts[kcur + 1] <= c0
            kcur += 1
        end
        k = kcur
        while k <= plan.nclasses && starts[k] <= chi_hi
            lo = max(starts[k], c0)
            hi = min(starts[k + 1] - 1, chi_hi)
            if hi >= lo
                _cuda_dense_class_gemm!(plan.dst_slab, plan.operators, k, plan.src_slab,
                    lo - c0 + 1, hi - c0 + 1, wp.alpha, wp.beta)
            end
            k += 1
        end
        _cuda_dense_scatter_add!(state.locals, _matrix_col_view(plan.dst_slab, n),
            ws.phi_flat_idx, ws.chi_flat_idx, view(state.route_targets, c0:chi_hi),
            ndof_phi, Val(LH))
    end
    return state
end

# Per-class reference driver (DENSE_CUDA_WHOLE_PASS[] = false): gather/GEMM/scatter
# per class, sub-chunked at the slab width so any class occupancy is supported.
# Mirrors the host _launch_resident_m2l_dense_plan! stage boundaries.
function _launch_resident_m2l_dense_perclass!(state::DeviceResidentRadixState{TF,B,LH},
        ws::ResidentOperatorWorkspace{TF,B,LH}, plan::ResidentM2LDenseCUDAPlan,
        wp::NamedTuple) where {TF,B,LH}
    fill!(state.locals.phi, zero(TF))
    LH && fill!(state.locals.chi, zero(TF))
    counts = plan.host_class_counts::Vector{Int32}
    starts = plan.class_starts
    W = wp.chunk
    ndof_phi = plan.ndof_phi
    @inbounds for k in eachindex(counts)
        cnt = Int(counts[k])
        cnt == 0 && continue
        c0 = starts[k]
        done = 0
        while done < cnt
            n = min(W, cnt - done)
            base = c0 + done
            _cuda_dense_gather!(_matrix_col_view(plan.src_slab, n), state.multipoles,
                ws.phi_flat_idx, ws.chi_flat_idx,
                view(state.route_sources, base:(base + n - 1)), ndof_phi, Val(LH))
            _cuda_dense_class_gemm!(plan.dst_slab, plan.operators, k, plan.src_slab, 1, n,
                wp.alpha, wp.beta)
            _cuda_dense_scatter_add!(state.locals, _matrix_col_view(plan.dst_slab, n),
                ws.phi_flat_idx, ws.chi_flat_idx,
                view(state.route_targets, base:(base + n - 1)), ndof_phi, Val(LH))
            done += n
        end
    end
    return state
end

# Device dispatch for the dense plan launch (the host method in translate_batched.jl
# keeps the ResidentM2LDensePlan signature): whole-pass chunked execution by
# default, or the per-class reference path.
function _launch_resident_m2l_dense_plan!(state::DeviceResidentRadixState{TF,B,LH},
        ws::ResidentOperatorWorkspace{TF,B,LH},
        plan::ResidentM2LDenseCUDAPlan) where {TF,B,LH}
    DENSE_CUDA_FUSED[] && return _launch_resident_m2l_dense_fused!(state, ws, plan)
    wp = plan.whole_pass[]
    wp isa NamedTuple || throw(ArgumentError(
        "CUDA dense M2L requires the whole-pass scratch bundle"))
    if DENSE_CUDA_WHOLE_PASS[]
        return _launch_resident_m2l_dense_whole!(state, ws, plan, wp)
    end
    return _launch_resident_m2l_dense_perclass!(state, ws, plan, wp)
end

# Overflow-safe estimate of the complete dense device lifecycle footprint, in the
# category breakdown reported on rejection. `_dense_m2l_footprint` supplies the
# operator/slab/route-metadata payloads (apply_width = the CUDA chunk width); the
# remaining device categories (expansion buffers and the grid/route/body/output
# scratch allocated by _radix_cache_device_build) are summed here so the free-memory
# gate sees the whole picture before any device allocation occurs.
function _dense_cuda_lifecycle_footprint(::Type{TF}, basis_info::OperatorBasisInfo{B,LH},
        nclasses::Int, route_capacity::Int, direct_capacity::Int, maxn::Int,
        max_cells::Int, max_nodes::Int, ell::Int, chunk::Int,
        body_cols::Int; hierarchical_occupancy_words::Int=0,
        hierarchical_window_words::Int=0,
        hierarchical_levels::Int=0) where {TF,B,LH}
    eltbytes = sizeof(TF)
    intbytes = sizeof(Int)
    D = _dense_m2m_dof(basis_info, Val(LH))
    # host-shared operator/slab/route-metadata payload accounting (device slabs are
    # 2 x D x chunk, matching the host apply-slab formula with apply_width = chunk)
    base = _dense_m2l_footprint(TF, basis_info, nclasses, route_capacity, chunk, D)
    operator_bytes = base.operator_bytes
    slab_bytes = base.scratch_bytes
    # device route metadata: Int32 route_class (capacity) + Int32 class histogram
    route_class_bytes = _dense_checked_mul(route_capacity, sizeof(Int32), "route class bytes")
    class_hist_bytes = _dense_checked_mul(nclasses, sizeof(Int32), "class histogram bytes")
    route_metadata_bytes = _dense_sum_checked(
        (route_class_bytes, class_hist_bytes), "dense CUDA route metadata")
    # expansion buffers: multipoles + locals flat storage at max_nodes width, plus
    # the resident degree-major workspace stage slabs (a conservative multiple of a
    # degree-major node buffer; the exact workspace has ~12 phi/chi stage buffers)
    flat_rows = _dense_checked_add(basis_info.basis_dof_phi,
        LH ? basis_info.basis_dof_chi : 0, "expansion flat rows")
    dm_rows = _dense_checked_add(degree_major_dof(basis_info.orders.P_phi),
        LH ? degree_major_dof(basis_info.orders.P_active) : 0, "expansion degree-major rows")
    expansion_elts = _dense_sum_checked((
        _dense_checked_mul(_dense_checked_mul(2, flat_rows, "flat buffers"), max_nodes,
            "flat buffer elements"),
        _dense_checked_mul(_dense_checked_mul(16, dm_rows, "workspace stage slabs"),
            max_cells, "workspace stage elements")),
        "expansion buffer elements")
    expansion_bytes = _dense_checked_mul(expansion_elts, eltbytes, "expansion buffer bytes")
    # other device scratch: grid arrays, routes/direct, source bodies, output, and
    # the occupancy map are the sizeable categories. The flat path allocates a
    # dense G^3 leaf `cell_at`; the hierarchical path replaces it with the
    # 1.1428 * 8^ell per-level `node_at` plus the single-window flag/prefix pair
    # (pass ell = 0 with the hierarchical word counts instead of G^3).
    G = 1 << ell
    g3 = _dense_checked_mul(_dense_checked_mul(G, G, "grid plane"), G, "grid volume")
    other_words = _dense_sum_checked((
        _dense_checked_mul(10, maxn, "per-body grid/scratch arrays"),
        _dense_checked_mul(10, max_nodes, "per-node grid arrays"),
        _dense_checked_mul(8, max_cells, "per-cell grid arrays"),
        _dense_checked_mul(6, route_capacity, "route index arrays"),
        _dense_checked_mul(4, direct_capacity, "direct index arrays"),
        g3, hierarchical_occupancy_words, hierarchical_window_words),
        "dense CUDA other-scratch words")
    other_bytes = _dense_sum_checked((
        _dense_checked_mul(other_words, intbytes, "other scratch int bytes"),
        _dense_checked_mul(_dense_checked_mul(9, maxn, "body/output float arrays"),
            eltbytes, "other scratch float bytes"),
        _dense_checked_mul(_dense_checked_mul(body_cols, maxn, "source staging"),
            eltbytes, "source staging bytes"),
        # hierarchical dense source/target level-scale columns (D x (ell - 1) each)
        _dense_checked_mul(_dense_checked_mul(_dense_checked_mul(2, D,
            "level scale rows"), hierarchical_levels, "level scale columns"),
            eltbytes, "level scale bytes")),
        "dense CUDA other scratch")
    persistent_bytes = _dense_sum_checked(
        (operator_bytes, slab_bytes, route_metadata_bytes), "dense CUDA persistent")
    estimated_peak_bytes = _dense_sum_checked(
        (persistent_bytes, expansion_bytes, other_bytes), "dense CUDA estimated peak")
    return (; ndof=D, operator_bytes, slab_bytes, route_metadata_bytes,
        expansion_bytes, other_bytes, persistent_bytes, estimated_peak_bytes)
end

function _dense_cuda_limit_error(strategy::DenseTranslationM2L, footprint, nclasses,
        chunk, free_bytes, gate::Symbol)
    mib(x) = x / 2.0^20
    common =
        "dense operators=$(footprint.operator_bytes) bytes ($(mib(footprint.operator_bytes)) MiB), " *
        "route metadata=$(footprint.route_metadata_bytes) bytes, " *
        "expansion buffers=$(footprint.expansion_bytes) bytes ($(mib(footprint.expansion_bytes)) MiB), " *
        "packing/application slabs=$(footprint.slab_bytes) bytes ($(mib(footprint.slab_bytes)) MiB), " *
        "other scratch=$(footprint.other_bytes) bytes ($(mib(footprint.other_bytes)) MiB), " *
        "persistent total=$(footprint.persistent_bytes) bytes ($(mib(footprint.persistent_bytes)) MiB), " *
        "estimated device peak=$(footprint.estimated_peak_bytes) bytes ($(mib(footprint.estimated_peak_bytes)) MiB); " *
        "D=$(footprint.ndof), classes=$nclasses, chunk=$chunk. " *
        "Lower P, disable Lamb-Helmholtz, reduce the applicable scratch chunk " *
        "(DENSE_CUDA_CHUNK), or select PrecomputedFactoredYM2L / FactoredRotationM2L " *
        "/ ConcatenatedFixedZM2L (5-30x smaller footprint); chunking does not reduce " *
        "dense operator storage."
    if gate === :persistent
        throw(ArgumentError(
            "DenseTranslationM2L device persistent payload exceeds max_persistent_bytes " *
            "(=$(strategy.max_persistent_bytes) bytes ($(mib(strategy.max_persistent_bytes)) MiB)): " *
            common))
    else
        throw(ArgumentError(
            "DenseTranslationM2L estimated device footprint exceeds the free-memory " *
            "budget: free memory=$free_bytes bytes ($(mib(free_bytes)) MiB), reserved " *
            "headroom=$(strategy.cuda_headroom_bytes) bytes ($(mib(strategy.cuda_headroom_bytes)) MiB), " *
            "usable=$(free_bytes - strategy.cuda_headroom_bytes) bytes. " * common))
    end
end

# Redefinition of the host stub (translate_batched_resident.jl): build the device
# dense M2L plan. The complete free-memory gate is applied earlier in
# _radix_cache_device_build (a pure preflight before any large device allocation);
# here the persistent-payload gate is applied from the host-shared footprint, then
# every D x D operator is oracle-built on the host (identical to the host
# ResidentM2LDensePlan), packed column-major, and uploaded once. Route
# histogram/counts, the gather/GEMM/scatter slabs, and the whole-pass bundle are all
# allocated here at construction capacity, so the recurring step never reallocates.
function _build_cuda_dense_m2l_plan(::Type{TF}, basis_info::OperatorBasisInfo{B,LH},
        accepted_offsets::AbstractVector{<:SVector{3,<:Integer}}, cell_width::Real,
        route_capacity::Integer, max_cells::Integer, grid_resolution::Integer,
        strategy::DenseTranslationM2L,
        invariant::OperatorInvariantCache{TF,B,LH},
        estimated_peak_bytes::Integer=0) where {TF,B,LH}
    nroutes = _dense_to_int(route_capacity, "route capacity")
    ncells = _dense_to_int(max_cells, "cell capacity")
    G = _dense_to_int(grid_resolution, "grid resolution")
    nroutes >= 0 && ncells >= 0 && G >= 0 || throw(ArgumentError(
        "dense M2L capacities must be nonnegative"))
    nclasses = length(accepted_offsets)
    D = _dense_m2m_dof(basis_info, Val(LH))
    class_capacities = Vector{Int}(undef, nclasses)
    @inbounds for i in eachindex(accepted_offsets)
        class_capacities[i] = _dense_m2l_capacity(accepted_offsets[i], nroutes, ncells, G)
    end
    W = max(min(DENSE_CUDA_CHUNK[], nroutes), 1)

    payload = _dense_m2l_footprint(TF, basis_info, nclasses, nroutes, W, D)
    route_class_bytes = _dense_checked_mul(nroutes, sizeof(Int32), "route class bytes")
    class_hist_bytes = _dense_checked_mul(nclasses, sizeof(Int32), "class histogram bytes")
    route_metadata_bytes = _dense_sum_checked((route_class_bytes, class_hist_bytes),
        "dense CUDA route metadata")
    persistent_bytes = _dense_sum_checked(
        (payload.operator_bytes, payload.scratch_bytes, route_metadata_bytes),
        "dense CUDA persistent")
    plan_estimated_peak_bytes = max(_dense_to_int(estimated_peak_bytes,
        "dense CUDA estimated peak bytes"), persistent_bytes)
    persistent_bytes <= strategy.max_persistent_bytes || _dense_cuda_limit_error(
        strategy,
        (; ndof=D, operator_bytes=payload.operator_bytes, slab_bytes=payload.scratch_bytes,
            route_metadata_bytes, expansion_bytes=0, other_bytes=0, persistent_bytes,
            estimated_peak_bytes=plan_estimated_peak_bytes),
        nclasses, W, 0, :persistent)

    # Oracle-build every operator on the host (identical to the host plan) and
    # upload each once into its column-major slice of the packed D x D x nclasses
    # device array. The packed layout serves both execution paths: the fused
    # per-route kernel indexes it directly by route class, and the per-class GEMM
    # drivers slice a contiguous (D, D) strided view that the direct
    # `CUBLAS.gemm!` call (with the plan's device alpha/beta staging) accepts
    # without any workspace allocation.
    build_width = strategy.build_chunk > 0 ? min(D, strategy.build_chunk) : D
    workspace = DenseM2LBuilderWorkspace(TF, basis_info, invariant, build_width)
    Kbuf = Matrix{TF}(undef, D, D)
    d_operators = CUDA.CuArray{TF,3}(undef, D, D, nclasses)
    @inbounds for (i, offset) in enumerate(accepted_offsets)
        delta = TF(cell_width) * SVector{3,TF}(offset)
        r, theta, phi = cartesian_to_spherical(delta)
        build_dense_m2l_operator!(Kbuf, r, theta, phi, invariant, workspace, Val(LH))
        _check_dense_m2l_operator_finite!(Kbuf, basis_info, offset)
        copyto!(view(d_operators, :, :, i), Kbuf)
    end

    route_class = CUDA.zeros(Int32, nroutes)
    class_counts = CUDA.zeros(Int32, nclasses)
    host_class_counts = _pin_host_array(zeros(Int32, nclasses))
    class_starts = zeros(Int, nclasses + 1)
    src_slab = CUDA.zeros(TF, D, W)
    dst_slab = CUDA.zeros(TF, D, W)
    ndof_phi = degree_major_dof(basis_info.orders.P_phi)
    # Construction-time device staging of the gemm alpha=1 / beta=0 scalars
    # (CUBLAS_POINTER_MODE_DEVICE; see _cuda_dense_class_gemm!). 2 * sizeof(TF)
    # bytes, excluded from the byte accounting as negligible.
    gemm_alpha = CUDA.CuArray(TF[one(TF)])
    gemm_beta = CUDA.zeros(TF, 1)
    plan = ResidentM2LDenseCUDAPlan{TF,typeof(d_operators),typeof(route_class),
        typeof(class_counts),typeof(src_slab)}(
        route_class, d_operators, class_counts, host_class_counts, class_starts,
        class_capacities, src_slab, dst_slab, nclasses, D, ndof_phi, W,
        payload.operator_bytes, payload.scratch_bytes, route_metadata_bytes,
        persistent_bytes, plan_estimated_peak_bytes,
        Ref{Any}((; chunk=W, alpha=gemm_alpha, beta=gemm_beta)))
    return plan
end

function _launch_cuda_resident_m2m!(state::DeviceResidentRadixState{TF,B,LH}) where {TF,B,LH}
    return _launch_resident_m2m!(state, state.options.m2m_strategy)
end

function _launch_cuda_resident_m2l!(state::DeviceResidentRadixState{TF,B,LH}) where {TF,B,LH}
    # Concrete-type dispatch on the resident interaction context (no residency
    # boolean): the hierarchical policy generates and applies its route windows
    # here, after B2M/M2M have produced the source expansions.
    hctx = state.interaction_list
    hctx isa DeviceHierarchicalM2LContext &&
        return _launch_cuda_hierarchical_m2l!(state, hctx)
    return _launch_resident_m2l!(state, state.options.m2l_strategy)
end

function _launch_cuda_resident_l2l!(state::DeviceResidentRadixState{TF,B,LH}) where {TF,B,LH}
    return _launch_resident_l2l!(state)
end

function _launch_cuda_resident_l2b!(state::DeviceResidentRadixState{TF,B,LH}) where {TF,B,LH}
    fill!(state.output, zero(TF))
    threads = 128
    npairs = state.counts.n_direct
    # warp-per-pair (task 028 lever 1): 4 warps per 128-thread block
    direct_blocks = min(cld(npairs, threads ÷ 32), DIRECT_CUDA_MAX_BLOCKS[])
    if direct_blocks > 0
        CUDA.@cuda threads=threads blocks=direct_blocks _cuda_direct_pairs_output_kernel!(
            state.output, state.source_bodies, state.cell_ranges,
            state.direct_targets, state.direct_sources, npairs,
        )
    end
    P_phi = state.invariant_cache.basis_info.orders.P_phi
    P_active = state.invariant_cache.basis_info.orders.P_active
    ncell = state.counts.n_cells
    # warp-per-cell (task 028 rider): 4 warps per 128-thread block
    l2b_blocks = cld(ncell, threads ÷ 32)
    if l2b_blocks > 0
        CUDA.@cuda threads=threads blocks=l2b_blocks _cuda_l2b_output_kernel!(
            state.output, state.source_bodies, state.cell_centers, state.cell_ranges,
            state.grid.leaf_to_node, state.locals.phi, state.locals.chi,
            P_phi, P_active, Val(LH), ncell,
        )
    end
    return state
end

function _launch_cuda_resident_operator_pipeline!(state::DeviceResidentRadixState)
    _assert_cuda_supported_operator!(state.options)
    _assert_cuda_resident_stage!(state, :b2m)
    _launch_cuda_resident_m2m!(state)
    _assert_cuda_resident_stage!(state, :m2m)
    # 016b watch item 2 (mirrors _launch_host_resident_operator_pipeline!): the
    # FactoredRotation* operators are exact only on the physical subspace; guard
    # the upward-pass output once per lifecycle run, DEBUG[]-gated.
    if DEBUG[] && _radix_uses_factored_rotation(state.options)
        _assert_factored_input_physical(state.multipoles)
    end
    _launch_cuda_resident_m2l!(state)
    _assert_cuda_resident_stage!(state, :m2l)
    _launch_cuda_resident_l2l!(state)
    _assert_cuda_resident_stage!(state, :l2l)
    _launch_cuda_resident_l2b!(state)
    _assert_cuda_resident_stage!(state, :l2b)
    return state
end

function cuda_radix_state(systems, grid::RadixGrid, list::RadixInteractionList,
        P::Integer, lamb_helmholtz::Val{LH}=Val(false);
    options::CUDARadixLifecycleOptions=CUDARadixLifecycleOptions()) where LH
    _require_cuda_radix_available()
    _assert_cuda_materialized_operator!(options)
    TF = options.precision
    basis_info = OperatorBasisInfo(CompressedComplexBasis(), P, lamb_helmholtz)
    counters = CUDARadixTransferCounters()
    host_grid = host_resident_radix_grid(grid)
    resident_grid = _cuda_upload_resident_grid(host_grid, counters)
    body_perm = resident_grid.perm
    body_system_ids = resident_grid.body_system
    body_indices = resident_grid.body_index
    host_body_perm = grid.perm
    host_body_system_ids = grid.body_system
    host_body_indices = grid.body_index
    host_m2m_parent_routes, host_m2m_child_routes, host_l2l_parent_routes, host_l2l_child_routes =
        _host_radix_tree_routes(host_grid)

    if systems isa AbstractMatrix || systems isa CUDA.AnyCuArray
        radix_bodies = _radix_body_matrix(grid, systems)
        source_bodies = _to_cuda_array(radix_bodies, TF, counters, :body)
    else
        source_buffers = _canonical_cuda_source_buffers(to_tuple(systems), TF, counters)
        source_bodies = _radix_body_matrix_from_source_buffers(
            grid, source_buffers, body_perm, body_system_ids, body_indices,
        )
    end
    target_bodies = source_bodies
    cell_centers = resident_grid.cell_centers
    cell_ranges = resident_grid.cell_ranges
    m2m_parent_routes, m2m_child_routes, l2l_parent_routes, l2l_child_routes =
        _cuda_radix_tree_routes(resident_grid)

    multipoles = _cuda_flat_buffer(TF, basis_info, length(resident_grid.node_keys))
    locals = _cuda_flat_buffer(TF, basis_info, length(resident_grid.node_keys))
    fill!(multipoles.phi, zero(TF)); fill!(multipoles.chi, zero(TF))
    fill!(locals.phi, zero(TF)); fill!(locals.chi, zero(TF))

    route_levels, route_offsets, route_targets, route_sources =
        _flatten_radix_node_routes(list, resident_grid, TF, counters)
    _host_route_levels, _host_route_offsets, host_route_targets, host_route_sources =
        _flatten_radix_routes_host(list, host_grid)
    direct_targets, direct_sources = _flatten_radix_direct_pairs(list, TF, counters)

    # Task 019 storage review: the resident lifecycle reads only basis_info metadata
    # from the invariant cache (all device operator data lives in the workspace /
    # concat plan), so the O(P^4) S_pos/S_neg blocks and the other cache arrays are
    # kept host-side instead of being mirrored onto the device.
    device_cache = OperatorInvariantCache(TF, basis_info)
    scratch = ResidentOperatorWorkspace(
        TF, basis_info, multipoles, host_grid, list,
        host_m2m_parent_routes, host_m2m_child_routes,
        host_l2l_parent_routes, host_l2l_child_routes,
        host_grid.node_levels, host_grid.node_centers, host_route_targets, host_route_sources;
        m2l_strategy=options.m2l_strategy,
    )
    output = CUDA.zeros(TF, 4, length(grid.perm))

    return DeviceResidentRadixState{TF,CompressedComplexBasis,LH}(
        resident_grid, list, source_bodies, target_bodies,
        body_perm, body_system_ids, body_indices,
        host_body_perm, host_body_system_ids, host_body_indices,
        host_grid.cell_centers, host_m2m_parent_routes, host_m2m_child_routes,
        host_l2l_parent_routes, host_l2l_child_routes,
        host_grid.node_levels, host_grid.node_centers, host_route_targets, host_route_sources,
        cell_centers, cell_ranges, m2m_parent_routes, m2m_child_routes,
        l2l_parent_routes, l2l_child_routes,
        multipoles, locals,
        route_levels, route_offsets, route_targets, route_sources,
        direct_targets, direct_sources, output,
        device_cache, scratch, counters, options,
        RadixStepCounts(source_bodies, cell_ranges, multipoles, route_targets, direct_targets),
    )
end

function cuda_radix_state(systems, grid::DeviceRadixGrid, list::RadixInteractionList,
        P::Integer, lamb_helmholtz::Val{LH}=Val(false);
    options::CUDARadixLifecycleOptions=CUDARadixLifecycleOptions(),
    host_grid::Union{Nothing,DeviceRadixGrid}=nothing) where LH
    _require_cuda_radix_available()
    _assert_cuda_materialized_operator!(options)
    host_grid === nothing && throw(ArgumentError(
        "cuda_radix_state with a DeviceRadixGrid requires host_grid metadata; " *
        "pass host_resident_radix_grid(cpu_grid) from the matching CPU RadixGrid",
    ))
    _assert_matching_host_radix_grid(grid, host_grid)
    TF = options.precision
    basis_info = OperatorBasisInfo(CompressedComplexBasis(), P, lamb_helmholtz)
    counters = CUDARadixTransferCounters()
    body_perm = grid.perm
    body_system_ids = grid.body_system
    body_indices = grid.body_index
    host_body_perm = host_grid.perm
    host_body_system_ids = host_grid.body_system
    host_body_indices = host_grid.body_index
    host_m2m_parent_routes, host_m2m_child_routes, host_l2l_parent_routes, host_l2l_child_routes =
        _host_radix_tree_routes(host_grid)

    if systems isa CUDA.AnyCuArray
        source_bodies = _radix_body_matrix(grid, systems)
    else
        source_buffers = _canonical_cuda_source_buffers(to_tuple(systems), TF, counters)
        source_bodies = _radix_body_matrix_from_source_buffers(
            grid, source_buffers, body_perm, body_system_ids, body_indices,
        )
    end
    target_bodies = source_bodies
    cell_centers = grid.cell_centers
    cell_ranges = grid.cell_ranges
    m2m_parent_routes, m2m_child_routes, l2l_parent_routes, l2l_child_routes =
        _cuda_radix_tree_routes(grid)

    n_nodes = length(grid.node_keys)
    multipoles = _cuda_flat_buffer(TF, basis_info, n_nodes)
    locals = _cuda_flat_buffer(TF, basis_info, n_nodes)
    fill!(multipoles.phi, zero(TF)); fill!(multipoles.chi, zero(TF))
    fill!(locals.phi, zero(TF)); fill!(locals.chi, zero(TF))

    route_levels, route_offsets, route_targets, route_sources =
        _flatten_radix_node_routes(list, grid, TF, counters)
    _host_route_levels, _host_route_offsets, host_route_targets, host_route_sources =
        _flatten_radix_routes_host(list, host_grid)
    direct_targets, direct_sources = _flatten_radix_direct_pairs(list, TF, counters)

    # Task 019 storage review: the resident lifecycle reads only basis_info metadata
    # from the invariant cache (all device operator data lives in the workspace /
    # concat plan), so the O(P^4) S_pos/S_neg blocks and the other cache arrays are
    # kept host-side instead of being mirrored onto the device.
    device_cache = OperatorInvariantCache(TF, basis_info)
    scratch = ResidentOperatorWorkspace(
        TF, basis_info, multipoles, host_grid, list,
        host_m2m_parent_routes, host_m2m_child_routes,
        host_l2l_parent_routes, host_l2l_child_routes,
        host_grid.node_levels, host_grid.node_centers, host_route_targets, host_route_sources;
        m2l_strategy=options.m2l_strategy,
    )
    output = CUDA.zeros(TF, 4, grid.n_bodies)

    return DeviceResidentRadixState{TF,CompressedComplexBasis,LH}(
        grid, list, source_bodies, target_bodies,
        body_perm, body_system_ids, body_indices,
        host_body_perm, host_body_system_ids, host_body_indices,
        host_grid.cell_centers, host_m2m_parent_routes, host_m2m_child_routes,
        host_l2l_parent_routes, host_l2l_child_routes,
        host_grid.node_levels, host_grid.node_centers, host_route_targets, host_route_sources,
        cell_centers, cell_ranges, m2m_parent_routes, m2m_child_routes,
        l2l_parent_routes, l2l_child_routes,
        multipoles, locals,
        route_levels, route_offsets, route_targets, route_sources,
        direct_targets, direct_sources, output,
        device_cache, scratch, counters, options,
        RadixStepCounts(source_bodies, cell_ranges, multipoles, route_targets, direct_targets),
    )
end

function run_cuda_radix_lifecycle!(state::DeviceResidentRadixState)
    _require_cuda_radix_available()
    _assert_cuda_supported_operator!(state.options)
    state.counters.expansion_host_copies == 0 ||
        throw(AssertionError("resident CUDA radix lifecycle observed expansion host copies before execution"))
    _launch_cuda_b2m!(state)
    _assert_cuda_resident_stage!(state, :b2m)
    _launch_cuda_resident_operator_pipeline!(state)
    return state
end

function copy_cuda_radix_output!(dest::CUDA.AnyCuArray, state::DeviceResidentRadixState)
    _require_cuda_radix_available()
    dest === state.output && return dest
    copyto!(dest, state.output)
    return dest
end

function copy_cuda_radix_output!(dest::AbstractArray, state::DeviceResidentRadixState)
    _require_cuda_radix_available()
    dest === state.output && return dest
    state.counters.influence_downloads += 1
    copyto!(dest, state.output)
    return dest
end

function _has_deprecated_cuda_target_method(target_system, device_output_buffer, sort_index, derivatives_switch)
    sig = Tuple{typeof(target_system),typeof(device_output_buffer),typeof(sort_index),typeof(derivatives_switch)}
    return hasmethod(target_system_from_device_buffer!, sig)
end

function buffer_to_target!(target_system, device_output_buffer::CUDA.AnyCuArray,
        derivatives_switch, sort_index=1:get_n_bodies(target_system))
    if _has_deprecated_cuda_target_method(target_system, device_output_buffer, sort_index, derivatives_switch)
        Base.depwarn(
            "target_system_from_device_buffer! is deprecated; define residency(system) = DeviceResident() and overload buffer_to_target!(target_system, device_output_buffer, derivatives_switch, sort_index)",
            :target_system_from_device_buffer!,
        )
        return target_system_from_device_buffer!(target_system, device_output_buffer, sort_index, derivatives_switch)
    end
    throw(ArgumentError(
        "DeviceResident CUDA target systems must overload FastMultipole.buffer_to_target!(target_system, device_output_buffer, derivatives_switch, sort_index)",
    ))
end

# Per-system cached device scatter buffer for the recurring finalize (task 028
# rider): CUDA.zeros here was a fresh pool allocation plus memset every step,
# and the scatter copy zero-fills the buffer again anyway. With a cache dict the
# buffer is allocated (undef) once per (rows, n_bodies) layout and reused.
function _cuda_cached_target_buffer(cache, isys::Integer, ::Type{TF},
        rows::Integer, nb::Integer) where TF
    cache === nothing && return CUDA.CuArray{TF}(undef, rows, nb)
    buf = get(cache, isys, nothing)
    if !(buf isa CUDA.CuArray{TF,2}) || size(buf) != (rows, nb)
        buf = CUDA.CuArray{TF}(undef, rows, nb)
        cache[isys] = buf
    end
    return buf
end

function finalize_cuda_radix_output!(state::DeviceResidentRadixState{TF}, target_systems;
        derivatives_switches=DerivativesSwitch(true, true, false, to_tuple(target_systems)),
        host_output_staging=nothing, target_buffers=nothing,
        device_target_buffers=nothing) where TF
    _require_cuda_radix_available()
    systems = to_tuple(target_systems)
    switches = to_tuple(derivatives_switches)
    length(systems) == length(switches) ||
        throw(ArgumentError("target systems and derivatives switches must have the same length"))

    host_output = nothing
    for (isys, target_system, switch) in zip(eachindex(systems), systems, switches)
        if residency(target_system) isa DeviceResident
            target_buffer = _cuda_cached_target_buffer(device_target_buffers, isys,
                TF, target_buffer_rows(switch), get_n_bodies(target_system))
            _copy_radix_output_to_device_target_buffer!(
                target_buffer, state.output, state.body_perm, state.body_system_ids,
                state.body_indices, isys, switch, state.counts.n_bodies,
            )
            buffer_to_target!(target_system, target_buffer, switch, 1:get_n_bodies(target_system))
        else
            if host_output === nothing
                if host_output_staging === nothing
                    host_output = Array(state.output)
                else
                    # recurring path: download only the valid column prefix into
                    # the preallocated (pinned) staging
                    nb = state.counts.n_bodies
                    copyto!(host_output_staging, 1, state.output, 1,
                        size(state.output, 1) * nb)
                    host_output = host_output_staging
                end
                state.counters.influence_downloads += 1
            end
            target_buffer = target_buffers === nothing ?
                allocate_target_buffer(TF, target_system, switch) : target_buffers[isys]
            _copy_radix_output_to_host_target_buffer!(
                target_buffer, host_output, state.host_body_perm,
                state.host_body_system_ids, state.host_body_indices, isys, switch,
                state.counts.n_bodies,
            )
            buffer_to_target!(target_system, target_buffer, switch, 1:get_n_bodies(target_system))
        end
    end
    return target_systems
end

function take_cuda_radix_output!(state::DeviceResidentRadixState)
    dest = Array{eltype(state.output)}(undef, size(state.output))
    return copy_cuda_radix_output!(dest, state)
end

#------- device-resident RadixFMMCache path (Matrix Operator Refactor, task 023) -------#
#
# Recurring GPU steps: bodies upload once per step into persistent device
# buffers; the grid is rebuilt **in place** on device (020a sort machinery over
# capacity-sized persistent arrays) inside the cache's fixed Morton domain, and
# any body outside the box throws ArgumentError (invariant contract); M2L routes
# and direct pairs are generated on device (flag -> scan -> compact, matching
# the host build_radix_routes! element order exactly); the per-level operator
# groups refresh their edge columns with a kernel. Everything — dense operators,
# the concat M2L plan, expansion buffers, grid/route/output storage, pinned host
# mirrors — persists from construction, so no array is reallocated across steps
# (CUDA's internal sort scratch comes from the memory pool),
# route_uploads/operator_uploads stay constant, body_uploads grows by one per
# step per host-resident system, and metadata_downloads counts the per-step
# perm/system/index mirrors used only for host-target finalization.

function _cuda_cell_at_scatter_kernel!(cell_at, cell_keys, n_cells, ell)
    c = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    c > n_cells && return nothing
    @inbounds begin
        ix, iy, iz = _cuda_decode_morton_key(cell_keys[c], ell)
        cell_at[ix + 1, iy + 1, iz + 1] = Int32(c)
    end
    return nothing
end

# M2L flags over (offset class, occupied cell), class-major with cells ascending —
# the same emission order as the host builder. `kbase` selects the class chunk.
function _cuda_route_flags_kernel!(flags, cell_at, cell_keys, offsets, kbase,
        n_cells, kn, ell)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    idx > kn * n_cells && return nothing
    kloc = (idx - 1) ÷ n_cells + 1
    c = (idx - 1) % n_cells + 1
    k = kbase + kloc
    G = 1 << ell
    @inbounds begin
        ix, iy, iz = _cuda_decode_morton_key(cell_keys[c], ell)
        sx = ix - offsets[1, k]
        sy = iy - offsets[2, k]
        sz = iz - offsets[3, k]
        src = Int32(0)
        if 0 <= sx < G && 0 <= sy < G && 0 <= sz < G
            src = cell_at[sx + 1, sy + 1, sz + 1]
        end
        flags[idx] = src == Int32(0) ? Int32(0) : Int32(1)
    end
    return nothing
end

function _cuda_route_compact_kernel!(route_levels, route_offsets, route_targets,
        route_sources, route_class, flags, prefix, cell_at, cell_keys, offsets,
        kbase, n_cells, kn, ell, leaf_offset, base)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    idx > kn * n_cells && return nothing
    @inbounds begin
        flags[idx] == Int32(1) || return nothing
        kloc = (idx - 1) ÷ n_cells + 1
        c = (idx - 1) % n_cells + 1
        k = kbase + kloc
        ix, iy, iz = _cuda_decode_morton_key(cell_keys[c], ell)
        sx = ix - offsets[1, k]
        sy = iy - offsets[2, k]
        sz = iz - offsets[3, k]
        src = cell_at[sx + 1, sy + 1, sz + 1]
        p = base + Int(prefix[idx])
        route_levels[p] = ell
        route_offsets[1, p] = Int(offsets[1, k])
        route_offsets[2, p] = Int(offsets[2, k])
        route_offsets[3, p] = Int(offsets[3, k])
        route_targets[p] = leaf_offset + c
        route_sources[p] = leaf_offset + Int(src)
        route_class[p] = Int32(k)
    end
    return nothing
end

# Direct pairs over (occupied cell, rejected offset), target-major with offsets in
# rejected order — the host _foreach_radix_direct_pair_implicit emission order. The
# dense (cell, offset) flag grid can exceed the flag buffer when the stencil rejects
# more offsets than max_cells (small P), so the grid is processed in flat ascending
# chunks (`fbase` global offset, `len` chunk length) — chunking in flat order keeps
# the emission order elementwise identical to the host builder (task 023b fix).
function _cuda_direct_flags_kernel!(flags, cell_at, cell_keys, offsets, fbase, len,
        kn, ell)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    idx > len && return nothing
    g = fbase + idx
    c = (g - 1) ÷ kn + 1
    k = (g - 1) % kn + 1
    G = 1 << ell
    @inbounds begin
        ix, iy, iz = _cuda_decode_morton_key(cell_keys[c], ell)
        sx = ix - offsets[1, k]
        sy = iy - offsets[2, k]
        sz = iz - offsets[3, k]
        src = Int32(0)
        if 0 <= sx < G && 0 <= sy < G && 0 <= sz < G
            src = cell_at[sx + 1, sy + 1, sz + 1]
        end
        flags[idx] = src == Int32(0) ? Int32(0) : Int32(1)
    end
    return nothing
end

function _cuda_direct_compact_kernel!(direct_targets, direct_sources, flags, prefix,
        cell_at, cell_keys, offsets, fbase, len, kn, ell, base)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    idx > len && return nothing
    @inbounds begin
        flags[idx] == Int32(1) || return nothing
        g = fbase + idx
        c = (g - 1) ÷ kn + 1
        k = (g - 1) % kn + 1
        ix, iy, iz = _cuda_decode_morton_key(cell_keys[c], ell)
        sx = ix - offsets[1, k]
        sy = iy - offsets[2, k]
        sz = iz - offsets[3, k]
        src = cell_at[sx + 1, sy + 1, sz + 1]
        p = base + Int(prefix[idx])
        direct_targets[p] = c
        direct_sources[p] = Int(src)
    end
    return nothing
end

# Per-level operator-group edge refresh: spherical angles replicate
# cartesian_to_spherical (rotate.jl) including its EPSILON degeneracy handling.
function _cuda_refresh_group_edges_kernel!(source_idx, target_idx, phis, thetas,
        parent_index, node_centers, first_child, n_edges, child_to_parent)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > n_edges && return nothing
    TF = eltype(phis)
    @inbounds begin
        child = first_child + i - 1
        parent = parent_index[child]
        if child_to_parent
            dx = node_centers[1, parent] - node_centers[1, child]
            dy = node_centers[2, parent] - node_centers[2, child]
            dz = node_centers[3, parent] - node_centers[3, child]
            source_idx[i] = child
            target_idx[i] = parent
        else
            dx = node_centers[1, child] - node_centers[1, parent]
            dy = node_centers[2, child] - node_centers[2, parent]
            dz = node_centers[3, child] - node_centers[3, parent]
            source_idx[i] = parent
            target_idx[i] = child
        end
        x2y2 = dx * dx + dy * dy
        r2 = x2y2 + dz * dz
        eps2 = TF(1e-10) * TF(1e-10)
        r = sqrt(r2)
        theta = zero(TF)
        if r2 > eps2
            if x2y2 > eps2
                theta = acos(clamp(dz / r, -one(TF), one(TF)))
            else
                theta = TF(pi) * (dz < 0)
            end
        end
        phis[i] = iszero(x2y2) ? zero(TF) : atan(dy, dx)
        thetas[i] = theta
    end
    return nothing
end

function _cuda_refresh_resident_stage_groups!(ws::ResidentOperatorWorkspace,
        grid::DeviceRadixGrid, level_offsets::Vector{Int}, ell::Int)
    threads = 128
    for (gi, parent_level) in enumerate((ell - 1):-1:0)
        child_level = parent_level + 1
        first_child = level_offsets[child_level + 1] + 1
        n_edges = level_offsets[child_level + 2] - level_offsets[child_level + 1]
        group = ws.m2m_groups[gi]
        n_edges <= length(group.source_idx) ||
            throw(AssertionError("resident m2m group at child level $child_level exceeded its capacity"))
        group.count[] = n_edges
        blocks = cld(n_edges, threads)
        blocks > 0 && CUDA.@cuda threads=threads blocks=blocks _cuda_refresh_group_edges_kernel!(
            group.source_idx, group.target_idx, group.phis, group.thetas,
            grid.parent_index, grid.node_centers, first_child, n_edges, true,
        )
    end
    for (gi, child_level) in enumerate(1:ell)
        first_child = level_offsets[child_level + 1] + 1
        n_edges = level_offsets[child_level + 2] - level_offsets[child_level + 1]
        group = ws.l2l_groups[gi]
        n_edges <= length(group.source_idx) ||
            throw(AssertionError("resident l2l group at child level $child_level exceeded its capacity"))
        group.count[] = n_edges
        blocks = cld(n_edges, threads)
        blocks > 0 && CUDA.@cuda threads=threads blocks=blocks _cuda_refresh_group_edges_kernel!(
            group.source_idx, group.target_idx, group.phis, group.thetas,
            grid.parent_index, grid.node_centers, first_child, n_edges, false,
        )
    end
    return ws
end

# Device flag -> scan -> compact route generation. M2L classes are processed in
# chunks of `class_chunk` offsets so the flags/prefix buffers stay bounded by
# route_capacity; emission order is class-major (chunks are consecutive class
# ranges), elementwise identical to the host builder. Returns (n_routes, n_direct).
function _cuda_generate_radix_routes!(ctx, grid::DeviceRadixGrid, n_cells::Int,
        leaf_offset::Int, ell::Int, route_class)
    threads = 256
    fill!(ctx.cell_at, Int32(0))
    blocks_cells = cld(n_cells, threads)
    blocks_cells > 0 && CUDA.@cuda threads=threads blocks=blocks_cells _cuda_cell_at_scatter_kernel!(
        ctx.cell_at, grid.cell_keys, n_cells, ell,
    )

    naccept = size(ctx.d_accepted, 2)
    n_routes = 0
    k0 = 1
    while k0 <= naccept && n_cells > 0
        kn = min(ctx.class_chunk, naccept - k0 + 1)
        used = kn * n_cells
        used <= length(ctx.route_flags) ||
            throw(AssertionError("device route flag buffer exceeded its capacity"))
        blocks = cld(used, threads)
        CUDA.@cuda threads=threads blocks=blocks _cuda_route_flags_kernel!(
            ctx.route_flags, ctx.cell_at, grid.cell_keys, ctx.d_accepted, k0 - 1,
            n_cells, kn, ell,
        )
        fv = view(ctx.route_flags, 1:used)
        pv = view(ctx.route_prefix, 1:used)
        accumulate!(+, pv, fv)
        copyto!(ctx.host_scalar32, 1, ctx.route_prefix, used, 1)
        chunk_total = Int(ctx.host_scalar32[1])
        if chunk_total > 0
            n_routes + chunk_total <= length(ctx.route_targets) ||
                throw(AssertionError("device route buffer exceeded its capacity"))
            CUDA.@cuda threads=threads blocks=blocks _cuda_route_compact_kernel!(
                ctx.route_levels, ctx.route_offsets, ctx.route_targets,
                ctx.route_sources, route_class, ctx.route_flags, ctx.route_prefix,
                ctx.cell_at, grid.cell_keys, ctx.d_accepted, k0 - 1, n_cells, kn,
                ell, leaf_offset, n_routes,
            )
        end
        n_routes += chunk_total
        k0 += kn
    end

    nreject = size(ctx.d_rejected, 2)
    n_direct = 0
    if n_cells > 0 && nreject > 0
        total = nreject * n_cells
        flag_capacity_direct = length(ctx.direct_flags)
        flag_capacity_direct > 0 ||
            throw(AssertionError("device direct flag buffer has zero capacity"))
        f0 = 0
        while f0 < total
            len = min(flag_capacity_direct, total - f0)
            blocks = cld(len, threads)
            CUDA.@cuda threads=threads blocks=blocks _cuda_direct_flags_kernel!(
                ctx.direct_flags, ctx.cell_at, grid.cell_keys, ctx.d_rejected,
                f0, len, nreject, ell,
            )
            fv = view(ctx.direct_flags, 1:len)
            pv = view(ctx.direct_prefix, 1:len)
            accumulate!(+, pv, fv)
            copyto!(ctx.host_scalar32, 1, ctx.direct_prefix, len, 1)
            chunk_total = Int(ctx.host_scalar32[1])
            if chunk_total > 0
                n_direct + chunk_total <= length(ctx.direct_targets) ||
                    throw(AssertionError("device direct pair buffer exceeded its capacity"))
                CUDA.@cuda threads=threads blocks=blocks _cuda_direct_compact_kernel!(
                    ctx.direct_targets, ctx.direct_sources, ctx.direct_flags,
                    ctx.direct_prefix, ctx.cell_at, grid.cell_keys, ctx.d_rejected,
                    f0, len, nreject, ell, n_direct,
                )
            end
            n_direct += chunk_total
            f0 += len
        end
    end
    return n_routes, n_direct
end

function _radix_offsets_matrix(offsets::Vector{SVector{3,Int}})
    out = Matrix{Int32}(undef, 3, length(offsets))
    for (k, offset) in enumerate(offsets)
        out[1, k] = Int32(offset[1])
        out[2, k] = Int32(offset[2])
        out[3, k] = Int32(offset[3])
    end
    return out
end

# Pin host staging memory when the runtime supports it so the recurring
# host<->device copies take the fast pinned path; registration failure only
# costs the pinned speedup.
function _pin_host_array(a::Array)
    try
        CUDA.pin(a)
    catch
    end
    return a
end

# sortperm into a preallocated index vector: CUDA.jl dispatches AnyCuArray
# arguments (contiguous views included) to its device sortperm!, the same
# backend as the one-shot builder's CUDA.sortperm, so equal-key ordering matches.
function _cuda_sortperm_into!(ix, keys)
    sortperm!(ix, keys)
    return ix
end

_radix_any_host_resident(systems::Tuple) =
    any(residency(system) isa HostResident for system in systems)

function _radix_cache_device_build(sources::Tuple, P::Int, ell::Int,
        x_min::SVector{3,TF}, h0::TF, maxn::Int, options::CUDARadixLifecycleOptions,
        stencil_policy, accepted::Vector{SVector{3,Int}},
        rejected::Vector{SVector{3,Int}}, max_cells::Int, max_nodes::Int,
        route_capacity::Int, direct_capacity::Int,
        basis_info::OperatorBasisInfo{B,LH}, ::Val{LH};
        hierarchical_tables::Union{Nothing,RigidHierarchicalTables}=nothing,
        class_level::Vector{Int32}=Int32[],
        class_offset::Matrix{Int32}=Matrix{Int32}(undef, 3, 0),
        max_level_nodes::Int=0) where {TF,B,LH}
    _require_cuda_radix_available()
    _assert_cuda_supported_operator!(options)
    hierarchical = stencil_policy isa HierarchicalRigidStencil
    hierarchical == (hierarchical_tables !== nothing) || throw(ArgumentError(
        "device radix cache construction requires the rigid hierarchical tables " *
        "exactly when the policy is a HierarchicalRigidStencil"))
    # The concatenated and grouped-factored selections share the bounded concat
    # engine in hierarchical mode (the same host semantic choice); precomputed-y
    # and dense keep their specialized plans.  The dense operator table is stored
    # per union offset and level-scaled per window, so it is built from the
    # unscaled push offsets rather than the `(level, offset)` effective classes.
    hierarchical_specialized = hierarchical &&
        options.m2l_strategy isa Union{PrecomputedFactoredYM2L,DenseTranslationM2L}
    plan_offsets = hierarchical && options.m2l_strategy isa DenseTranslationM2L ?
        hierarchical_tables.push_offsets : accepted
    workspace_strategy = hierarchical ?
        (hierarchical_specialized ? options.m2l_strategy : ConcatenatedFixedZM2L()) :
        options.m2l_strategy
    workspace_operator = hierarchical ?
        (hierarchical_specialized ? options.operator : MaterializedYRotationM2L()) :
        options.operator
    # Dense device free-memory gate: a pure preflight before any large device
    # allocation (task 023f). The persistent-payload gate is applied later inside
    # _build_cuda_dense_m2l_plan; here the complete estimated lifecycle footprint
    # must fit within CUDA.free_memory() minus the reserved headroom.
    if options.m2l_strategy isa DenseTranslationM2L
        chunk = max(min(DENSE_CUDA_CHUNK[], route_capacity), 1)
        body_cols = sum(data_per_body, sources)
        dense_cuda_footprint = _dense_cuda_lifecycle_footprint(TF, basis_info,
            length(plan_offsets), route_capacity, direct_capacity, maxn, max_cells,
            max_nodes, hierarchical ? 0 : ell, chunk, body_cols;
            hierarchical_occupancy_words=hierarchical ?
                _cuda_hier_occupancy_words(stencil_policy, ell) : 0,
            hierarchical_window_words=hierarchical ?
                2 * max(min(stencil_policy.window_classes,
                    length(hierarchical_tables.push_offsets)) * max_level_nodes, 1) : 0,
            hierarchical_levels=hierarchical ? max(ell - 1, 0) : 0)
        free_bytes = Int(CUDA.free_memory())
        dense_cuda_footprint.estimated_peak_bytes <=
            free_bytes - options.m2l_strategy.cuda_headroom_bytes ||
            _dense_cuda_limit_error(options.m2l_strategy, dense_cuda_footprint,
                length(plan_offsets), chunk, free_bytes, :freemem)
    else
        dense_cuda_footprint = nothing
    end
    counters = CUDARadixTransferCounters()
    multipoles = _cuda_flat_buffer(TF, basis_info, max_nodes)
    locals = _cuda_flat_buffer(TF, basis_info, max_nodes)
    fill!(multipoles.phi, zero(TF)); fill!(multipoles.chi, zero(TF))
    fill!(locals.phi, zero(TF)); fill!(locals.chi, zero(TF))
    invariant = OperatorInvariantCache(TF, basis_info)
    workspace = _radix_cache_workspace(TF, basis_info, multipoles, ell, h0,
        max_cells, max_nodes, route_capacity, plan_offsets, invariant,
        workspace_strategy, workspace_operator; compact_cuda_factored=true,
        dense_cuda_estimated_peak_bytes=dense_cuda_footprint === nothing ? 0 :
            dense_cuda_footprint.estimated_peak_bytes)
    if workspace.m2l_concat isa ResidentM2LFactoredPlan
        _pin_host_array(workspace.m2l_concat.host_class_counts)
        _cuda_factored_whole_pass_setup!(workspace.m2l_concat, TF, basis_info)
    elseif workspace.m2l_concat isa ResidentM2LPrecomputedYPlan
        _pin_host_array(workspace.m2l_concat.host_class_counts)
        _cuda_precomputed_y_whole_pass_setup!(workspace.m2l_concat, TF, basis_info)
    end
    # accepted/rejected offsets and every dense operator upload once, here. The
    # hierarchical path uses the task-025 stencil tables on the device context
    # instead of a flat accepted/rejected pair, so these stay empty.
    d_accepted = _to_cuda_array(
        _radix_offsets_matrix(hierarchical ? SVector{3,Int}[] : accepted),
        TF, counters, :route)
    d_rejected = _to_cuda_array(
        _radix_offsets_matrix(hierarchical ? SVector{3,Int}[] : rejected),
        TF, counters, :route)
    counters.operator_uploads += 1

    # capacity-sized persistent grid: counts (n_bodies/n_cells and the caller's
    # level offsets) bound the valid prefixes, so recurring steps refresh the
    # same device arrays in place and never reallocate
    grid = DeviceRadixGrid(
        x_min, h0, ell, 0, 0,
        CUDA.zeros(Int, maxn), CUDA.zeros(Int, maxn),
        CUDA.zeros(UInt64, max_cells), CUDA.zeros(Int, 2, max_cells),
        CUDA.zeros(Int, maxn), CUDA.zeros(Int, maxn),
        CUDA.zeros(TF, 3, max_cells),
        CUDA.zeros(Int, max_nodes), CUDA.zeros(UInt64, max_nodes),
        CUDA.zeros(Int, 3, max_nodes), CUDA.zeros(TF, 3, max_nodes),
        CUDA.zeros(Int, max_nodes), CUDA.zeros(Int, 2, max_nodes),
        CUDA.zeros(Int, max_cells),
    )
    n_edges_capacity = max(max_nodes - 1, 0)

    # per-system upload staging: host-resident systems get a pinned host buffer
    # plus a persistent device buffer; device-resident systems materialize their
    # own canonical buffer each step (see _radix_cache_refresh_source_buffers!)
    host_stagings = Tuple(
        residency(system) isa HostResident ?
            _pin_host_array(Matrix{TF}(undef, data_per_body(system), maxn)) : nothing
        for system in sources)
    device_sources = Tuple(
        residency(system) isa HostResident ?
            CUDA.zeros(TF, data_per_body(system), maxn) : nothing
        for system in sources)

    # Flat occupancy/flag storage is leaf-only and unused by the hierarchical
    # path, which keeps its per-level `node_at` and its single-window flag/prefix
    # pair on the device hierarchical context instead.
    G = hierarchical ? 0 : 1 << ell
    class_chunk = hierarchical ? 1 : max(min(length(accepted), max_cells), 1)
    flag_capacity = hierarchical ? 0 : class_chunk * max_cells
    hierarchical_ctx = hierarchical ?
        _build_cuda_hierarchical_context(TF, basis_info, stencil_policy,
            hierarchical_tables, class_level, class_offset, accepted,
            workspace.m2l_concat, ell, max_level_nodes, counters) :
        nothing
    ctx = (;
        multipoles, locals, workspace, invariant, counters, grid,
        counts=RadixStepCounts(0, 0, 0, 0, 0),
        source_bodies=CUDA.zeros(TF, 5, maxn),
        output=CUDA.zeros(TF, 4, maxn),
        cell_at=CUDA.zeros(Int32, G, G, G),
        hierarchical_ctx,
        d_accepted, d_rejected, class_chunk,
        route_levels=CUDA.zeros(Int, route_capacity),
        route_offsets=CUDA.zeros(Int, 3, route_capacity),
        route_targets=CUDA.zeros(Int, route_capacity),
        route_sources=CUDA.zeros(Int, route_capacity),
        direct_targets=CUDA.zeros(Int, direct_capacity),
        direct_sources=CUDA.zeros(Int, direct_capacity),
        route_flags=CUDA.zeros(Int32, flag_capacity),
        route_prefix=CUDA.zeros(Int32, flag_capacity),
        direct_flags=CUDA.zeros(Int32, direct_capacity),
        direct_prefix=CUDA.zeros(Int32, direct_capacity),
        # grid-update scratch (task 023 cost fix): persistent so the recurring
        # step performs no device allocation beyond CUDA's pool-served sort scratch
        positions=CUDA.zeros(TF, 3, maxn),
        keys=CUDA.zeros(UInt64, maxn),
        sorted_keys=CUDA.zeros(UInt64, maxn),
        body_flags=CUDA.zeros(Int, maxn),
        body_prefix=CUDA.zeros(Int, maxn),
        cell_coords=CUDA.zeros(Int, 3, max_cells),
        level_keys=CUDA.zeros(UInt64, max_cells, ell + 1),
        level_flags=CUDA.zeros(Int, max_cells, ell + 1),
        level_prefix=CUDA.zeros(Int, max_cells, ell + 1),
        level_counts=CUDA.zeros(Int, ell + 1),
        d_level_offsets=CUDA.zeros(Int, ell + 2),
        oob_flag=CUDA.zeros(Int32, 1),
        m2m_parent_routes=CUDA.zeros(Int, n_edges_capacity),
        m2m_child_routes=CUDA.zeros(Int, n_edges_capacity),
        l2l_parent_routes=CUDA.zeros(Int, n_edges_capacity),
        l2l_child_routes=CUDA.zeros(Int, n_edges_capacity),
        host_stagings, device_sources,
        # pinned host mirrors/staging for the step downloads
        host_oob=_pin_host_array(zeros(Int32, 1)),
        host_scalar=_pin_host_array(zeros(Int, 1)),
        host_scalar32=_pin_host_array(zeros(Int32, 1)),
        host_level_counts=_pin_host_array(zeros(Int, ell + 1)),
        host_perm=_pin_host_array(zeros(Int, maxn)),
        host_body_system=_pin_host_array(zeros(Int, maxn)),
        host_body_index=_pin_host_array(zeros(Int, maxn)),
        host_output=_pin_host_array(zeros(TF, 4, maxn)),
        # per-system device scatter buffers for the recurring finalize (028 rider)
        device_target_buffers=Dict{Int,Any}(),
    )
    cache = RadixFMMCache{TF,LH}(
        P, ell, x_min, h0, maxn, true, options, stencil_policy,
        accepted, rejected, max_cells, max_nodes, route_capacity, direct_capacity,
        nothing, zeros(Int32, 0, 0, 0), SVector{3,Int}[], zeros(Int, ell + 2),
        UInt64[], Int[], Int[], Int[], nothing, nothing, ctx,
        length(sources), false, 0,
    )
    update_cuda_radix_state!(cache, sources)
    return cache
end

# Refresh the persistent per-system device source buffers. Host-resident systems
# repack into their pinned staging and upload the valid column prefix (one upload
# per system per step); device-resident systems materialize a canonical buffer
# through their own overloads.
function _radix_cache_refresh_source_buffers!(ctx, systems::Tuple, ::Type{TF}) where TF
    return ntuple(length(systems)) do isys
        system = systems[isys]
        n_sys = get_n_bodies(system)
        if residency(system) isa HostResident
            staging = ctx.host_stagings[isys]
            device_buffer = ctx.device_sources[isys]
            source_to_buffer!(staging, system, 1:n_sys)
            # linear-prefix copy: the first n_sys columns are contiguous
            copyto!(device_buffer, 1, staging, 1, size(staging, 1) * n_sys)
            ctx.counters.body_uploads += 1
            view(device_buffer, :, 1:n_sys)
        else
            _canonical_cuda_source_buffer(system, TF, ctx.counters, DeviceResident())
        end
    end
end

function _radix_cache_collect_positions!(ctx, source_buffers::Tuple)
    grid = ctx.grid
    threads = 128
    offset = 0
    for isys in eachindex(source_buffers)
        nb = size(source_buffers[isys], 2)
        blocks = cld(nb, threads)
        if blocks > 0
            CUDA.@cuda threads=threads blocks=blocks _cuda_extract_source_positions_kernel!(
                ctx.positions, grid.body_system, grid.body_index,
                source_buffers[isys], offset, isys,
            )
        end
        offset += nb
    end
    return offset
end

# In-place device grid rebuild inside the cache's fixed Morton domain: same
# stage order as _cuda_radix_grid_from_positions, but every output lands in the
# capacity-sized persistent grid/scratch, the per-level ancestor sorts are
# dropped (ancestor keys inherit the leaf-key order under a right shift), and
# the per-level count reads collapse into a single download.
function _cuda_update_radix_grid_in_place!(ctx, cache::RadixFMMCache{TF}, n::Int) where TF
    grid = ctx.grid
    ell = cache.ell
    x_min = cache.x_min
    h0 = cache.h0
    threads = 128
    blocks = cld(n, threads)

    # Morton keys + fixed-box enforcement; ctx.keys is scratch, so throwing here
    # leaves the persistent grid at its previous consistent step
    fill!(ctx.oob_flag, Int32(0))
    CUDA.@cuda threads=threads blocks=blocks _cuda_radix_keys_checked_kernel!(
        view(ctx.keys, 1:n), ctx.oob_flag, ctx.positions, x_min, h0, ell,
    )
    copyto!(ctx.host_oob, ctx.oob_flag)
    if ctx.host_oob[1] != 0
        x_max = x_min .+ 2 * h0
        throw(ArgumentError(
            "at least one body lies outside the fixed RadixFMMCache box " *
            "[$(Tuple(x_min)), $(Tuple(x_max))]; the box is part of the cache's " *
            "invariant contract — construct a new cache (or pass explicit " *
            "bounds=(x_min, box_size) covering the trajectory)"))
    end

    # sort bodies by key; same backend as the one-shot builder, so equal-key
    # (same-cell) bodies keep their deterministic global order
    pv = view(grid.perm, 1:n)
    kv = view(ctx.keys, 1:n)
    _cuda_sortperm_into!(pv, kv)
    sk = view(ctx.sorted_keys, 1:n)
    CUDA.@cuda threads=threads blocks=blocks _cuda_gather_sorted_keys_kernel!(sk, ctx.keys, pv)
    CUDA.@cuda threads=threads blocks=blocks _cuda_fill_invperm_kernel!(grid.invperm, pv)

    # occupied leaf cells
    fv = view(ctx.body_flags, 1:n)
    CUDA.@cuda threads=threads blocks=blocks _cuda_key_change_flags_kernel!(fv, sk)
    pfx = view(ctx.body_prefix, 1:n)
    accumulate!(+, pfx, fv)
    copyto!(ctx.host_scalar, 1, ctx.body_prefix, n, 1)
    n_cells = ctx.host_scalar[1]
    n_cells <= cache.max_cells ||
        throw(AssertionError("device radix grid exceeded the cache cell capacity"))
    CUDA.@cuda threads=threads blocks=blocks _cuda_fill_cell_firsts_kernel!(
        grid.cell_keys, grid.cell_ranges, sk, fv, pfx,
    )
    CUDA.@cuda threads=threads blocks=blocks _cuda_fill_cell_counts_kernel!(
        grid.cell_ranges, fv, pfx,
    )
    blocks_cells = cld(n_cells, threads)
    ckv = view(grid.cell_keys, 1:n_cells)
    CUDA.@cuda threads=threads blocks=blocks_cells _cuda_cell_centers_kernel!(
        grid.cell_centers, ctx.cell_coords, ckv, x_min, h0, ell,
    )

    # per-level unique ancestors: cell_keys is ascending and a right shift is
    # monotone, so each level's ancestor keys are already sorted
    for level in 0:ell
        col = level + 1
        lk = view(ctx.level_keys, 1:n_cells, col)
        lf = view(ctx.level_flags, 1:n_cells, col)
        lp = view(ctx.level_prefix, 1:n_cells, col)
        CUDA.@cuda threads=threads blocks=blocks_cells _cuda_leaf_ancestor_keys_kernel!(
            lk, ckv, ell, level,
        )
        CUDA.@cuda threads=threads blocks=blocks_cells _cuda_key_change_flags_kernel!(lf, lk)
        accumulate!(+, lp, lf)
    end
    CUDA.@cuda threads=32 blocks=1 _cuda_gather_level_counts_kernel!(
        ctx.level_counts, ctx.level_prefix, n_cells,
    )
    copyto!(ctx.host_level_counts, ctx.level_counts)
    level_offsets = cache.level_offsets
    level_offsets[1] = 0
    for level in 0:ell
        level_offsets[level + 2] = level_offsets[level + 1] + ctx.host_level_counts[level + 1]
    end
    n_nodes = level_offsets[end]
    n_nodes <= cache.max_nodes ||
        throw(AssertionError("device radix grid exceeded the cache node capacity"))
    for level in 0:ell
        col = level + 1
        CUDA.@cuda threads=threads blocks=blocks_cells _cuda_fill_unique_keys_kernel!(
            grid.node_keys, view(ctx.level_keys, 1:n_cells, col),
            view(ctx.level_flags, 1:n_cells, col), view(ctx.level_prefix, 1:n_cells, col),
            level_offsets[col],
        )
    end
    copyto!(ctx.d_level_offsets, level_offsets)
    max_count = maximum(ctx.host_level_counts; init=0)
    if max_count > 0
        blocks_x = cld(max_count, threads)
        CUDA.@cuda threads=threads blocks=(blocks_x, ell + 1) _cuda_fill_node_geometry_kernel!(
            grid.node_levels, grid.node_coords, grid.node_centers, grid.node_keys,
            ctx.d_level_offsets, x_min, h0, ell,
        )
        CUDA.@cuda threads=threads blocks=(blocks_x, ell + 1) _cuda_parent_index_kernel!(
            grid.parent_index, grid.node_keys, ctx.d_level_offsets, ell,
        )
        CUDA.@cuda threads=threads blocks=(blocks_x, ell + 1) _cuda_child_ranges_kernel!(
            grid.child_ranges, grid.node_keys, ctx.d_level_offsets, ell,
        )
    end
    CUDA.@cuda threads=threads blocks=blocks_cells _cuda_fill_leaf_to_node_kernel!(
        view(grid.leaf_to_node, 1:n_cells), level_offsets[ell + 1],
    )
    grid.n_bodies = n
    grid.n_cells = n_cells
    return n_cells
end

"""
    update_cuda_radix_state!(cache, systems)

Device-resident step refresh for a `RadixFMMCache(device=true)`: uploads the
bodies (one upload per host-resident system), rebuilds the grid **in place** on
device inside the cache's fixed Morton domain (throwing `ArgumentError` if any
body left the box), regenerates M2L routes and direct pairs on device, and
refreshes the per-level operator-group edge columns. All device and host-mirror
storage is persistent and capacity-sized, so no array is reallocated across
steps; `route_uploads`/`operator_uploads` stay constant after construction,
`body_uploads` grows by one per host-resident system, and `metadata_downloads`
grows by three (perm/system/index mirrors) per step with host-resident targets.
"""
function update_cuda_radix_state!(cache::RadixFMMCache{TF,LH}, systems::Tuple) where {TF,LH}
    _require_cuda_radix_available()
    ctx = cache.device_ctx
    ctx === nothing &&
        throw(ArgumentError("update_cuda_radix_state! requires a cache built with device=true"))
    length(systems) == cache.n_systems ||
        throw(ArgumentError("cache was built for $(cache.n_systems) source systems, got $(length(systems))"))
    n = get_n_bodies(systems)
    n > 0 || throw(ArgumentError("update_cuda_radix_state! requires at least one body"))
    n <= cache.max_n_bodies ||
        throw(ArgumentError("n=$n exceeds the cache capacity max_n_bodies=$(cache.max_n_bodies)"))
    counters = ctx.counters
    grid = ctx.grid
    hctx = ctx.hierarchical_ctx
    profiling = hctx !== nothing && hctx.profile_stages
    profiling && fill!(hctx.update_stage_ns, 0)

    t_stage = profiling ? (CUDA.synchronize(); time_ns()) : UInt64(0)
    source_buffers = _radix_cache_refresh_source_buffers!(ctx, systems, TF)
    _radix_cache_collect_positions!(ctx, source_buffers)
    n_cells = _cuda_update_radix_grid_in_place!(ctx, cache, n)
    n_nodes = cache.level_offsets[end]
    if profiling
        CUDA.synchronize()
        hctx.update_stage_ns[1] = time_ns() - t_stage
    end

    _pack_radix_body_matrix!(ctx.source_bodies, source_buffers, view(grid.perm, 1:n),
        grid.body_system, grid.body_index)

    # host mirrors serve host-resident target finalization only
    if _radix_any_host_resident(systems)
        copyto!(ctx.host_perm, 1, grid.perm, 1, n)
        copyto!(ctx.host_body_system, 1, grid.body_system, 1, n)
        copyto!(ctx.host_body_index, 1, grid.body_index, 1, n)
        counters.metadata_downloads += 3
    end

    n_edges = max(n_nodes - 1, 0)
    if n_edges > 0
        blocks = cld(n_edges, 128)
        CUDA.@cuda threads=128 blocks=blocks _cuda_tree_routes_kernel!(
            ctx.m2m_parent_routes, ctx.m2m_child_routes,
            ctx.l2l_parent_routes, ctx.l2l_child_routes,
            view(grid.parent_index, 1:n_nodes),
        )
    end

    plan = ctx.workspace.m2l_concat
    route_class = plan.route_class
    if hctx === nothing
        n_routes, n_direct = _cuda_generate_radix_routes!(
            ctx, grid, n_cells, cache.level_offsets[cache.ell + 1], cache.ell, route_class,
        )
        plan isa ResidentM2LFactoredPlan &&
            _cuda_refresh_factored_m2l_routes!(plan, route_class, n_routes)
        plan isa ResidentM2LPrecomputedYPlan &&
            _cuda_refresh_precomputed_y_m2l_routes!(plan, route_class, n_routes)
        plan isa ResidentM2LDenseCUDAPlan &&
            _cuda_refresh_dense_m2l_routes!(plan, route_class, n_routes)
    else
        # Hierarchical: the update refreshes occupancy, direct pairs, and tree
        # metadata only. Far-field route windows are generated and applied inside
        # the M2L stage, after B2M/M2M have produced the source expansions —
        # generating them here would either violate lifecycle ordering or require
        # the forbidden whole-tree pair list.
        t_stage = profiling ? (CUDA.synchronize(); time_ns()) : UInt64(0)
        _cuda_hier_refresh_occupancy!(hctx, grid, cache.level_offsets)
        if profiling
            CUDA.synchronize()
            hctx.update_stage_ns[2] = time_ns() - t_stage
            t_stage = time_ns()
        end
        n_direct = _cuda_hier_generate_direct_pairs!(ctx, hctx, grid, n_cells,
            cache.level_offsets[cache.ell + 1], cache.ell)
        if profiling
            CUDA.synchronize()
            hctx.update_stage_ns[3] = time_ns() - t_stage
        end
        n_routes = 0
    end
    t_stage = profiling ? (CUDA.synchronize(); time_ns()) : UInt64(0)
    _cuda_refresh_resident_stage_groups!(ctx.workspace, grid, cache.level_offsets, cache.ell)
    if profiling
        CUDA.synchronize()
        hctx.update_stage_ns[5] = time_ns() - t_stage
    end

    counts = ctx.counts
    counts.n_bodies = n
    counts.n_cells = n_cells
    counts.n_nodes = n_nodes
    counts.n_routes = n_routes
    counts.n_direct = n_direct

    if cache.state === nothing
        # every array below is persistent; the state wrapper is built once and
        # refreshed in place on later steps
        cache.state = DeviceResidentRadixState{TF,CompressedComplexBasis,LH}(
            grid, hctx, ctx.source_bodies, ctx.source_bodies,
            grid.perm, grid.body_system, grid.body_index,
            ctx.host_perm, ctx.host_body_system, ctx.host_body_index,
            nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing,
            grid.cell_centers, grid.cell_ranges,
            ctx.m2m_parent_routes, ctx.m2m_child_routes,
            ctx.l2l_parent_routes, ctx.l2l_child_routes,
            ctx.multipoles, ctx.locals,
            ctx.route_levels, ctx.route_offsets, ctx.route_targets, ctx.route_sources,
            ctx.direct_targets, ctx.direct_sources, ctx.output,
            ctx.invariant, ctx.workspace, counters, cache.options, counts,
        )
    end
    cache.step += 1
    return cache
end

update_cuda_radix_state!(cache::RadixFMMCache, systems) =
    update_cuda_radix_state!(cache, to_tuple(systems))

function _radix_cache_device_step!(cache::RadixFMMCache, targets::Tuple, switches::Tuple)
    update_cuda_radix_state!(cache, targets)
    run_cuda_radix_lifecycle!(cache.state)
    finalize_cuda_radix_output!(cache.state, targets; derivatives_switches=switches,
        host_output_staging=cache.device_ctx.host_output,
        target_buffers=_radix_cache_target_buffers!(cache, switches),
        device_target_buffers=cache.device_ctx.device_target_buffers)
    return cache
end

#------- device-resident hierarchical rigid M2L (Matrix Operator Refactor, task 027) -------#
#
# CUDA mirror of the 026 host hierarchical lifecycle. The flat device path walks a
# single leaf-level V-list; this one walks the genuine multilevel task-025 rigid
# stencil: levels `2:ell`, source-major inside each `(level, union-offset)` class,
# with the phase mask `class_of[phase, k]` deciding which push offsets a source
# coordinate may emit. The complete pair list is never compiled — one
# `(level, consecutive-offset-class window)` at a time is flagged, scanned, and
# compacted into the reusable route buffers and applied immediately, so the peak
# route storage is the window capacity `min(K * max_level_nodes, max_level_nodes^2)`
# rather than the many-GiB whole-tree list.
#
# Emission order is exactly the host `build_hierarchical_routes_window!` loop nest
# (offset/class major, sources ascending flat node index), so host/device parity
# holds elementwise, not merely as a multiset.
#
# Transfer contract (task 023): the stencil tables upload once at construction and
# are counted as operator uploads; the recurring step performs no route or operator
# upload, no expansion host copy, and no new metadata download. Per window the only
# host traffic is the `kn`-entry Int32 window prefix (bounded by `window_classes`),
# which replaces the flat path's `nclasses` histogram download — hierarchical
# `nclasses` reaches `(ell - 1) * 1740`, so a whole-plan histogram per window would
# be orders of magnitude more traffic than the flat contract allows.

# Per-level occupied-node scatter: `node_at[level_base[L + 1] + linear + 1]` is the
# flat node index of the occupied node at level `L` with coordinate `linear`, or 0.
# The flat path's `_cuda_cell_at_scatter_kernel!` decodes a Morton key per cell;
# here `grid.node_coords`/`grid.node_levels` are already resident from the 020a
# device grid refresh, so no decode is needed.
function _cuda_hier_node_at_scatter_kernel!(node_at, node_levels, node_coords,
        level_base, n_nodes)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > n_nodes && return nothing
    @inbounds begin
        L = node_levels[i]
        G = 1 << L
        linear = node_coords[1, i] + G * (node_coords[2, i] + G * node_coords[3, i])
        node_at[level_base[L + 1] + linear + 1] = Int32(i)
    end
    return nothing
end

# Source-major window flags over `(offset class k, occupied node at level L)`,
# k-major with sources ascending — the host loop nest. A candidate is flagged iff
# the source phase admits offset k and the pushed target coordinate is occupied.
function _cuda_hier_route_flags_kernel!(flags, node_at, node_coords, push_offsets,
        class_of, level_base_L, first_source, n_sources, first_offset, kn, L)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    idx > kn * n_sources && return nothing
    kloc = (idx - 1) ÷ n_sources + 1
    s = (idx - 1) % n_sources + 1
    k = first_offset + kloc - 1
    source = first_source + s - 1
    G = 1 << L
    @inbounds begin
        cx = node_coords[1, source]
        cy = node_coords[2, source]
        cz = node_coords[3, source]
        # same x/y/z bit convention as _rigid_phase_index
        phase = 1 + (cx & 1) + 2 * (cy & 1) + 4 * (cz & 1)
        hit = Int32(0)
        if class_of[phase, k] != Int32(0)
            tx = cx + push_offsets[1, k]
            ty = cy + push_offsets[2, k]
            tz = cz + push_offsets[3, k]
            if 0 <= tx < G && 0 <= ty < G && 0 <= tz < G
                linear = tx + G * (ty + G * tz)
                node_at[level_base_L + linear + 1] == Int32(0) || (hit = Int32(1))
            end
        end
        flags[idx] = hit
    end
    return nothing
end

# Per-class cumulative window counts read straight off the inclusive scan: class
# `kloc` ends at flat index `kloc * n_sources`. This makes the whole per-window
# host download `kn` Int32 entries and removes the class histogram entirely.
function _cuda_hier_window_cum_kernel!(cum, prefix, n_sources, kn)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > kn && return nothing
    @inbounds cum[i] = prefix[i * n_sources]
    return nothing
end

# Compact one window into the start of the reusable route buffers. Offsets are the
# unscaled integer push offsets; endpoints are flat node indices (not leaf indices).
# `class_base` is `(L - 2) * noffsets` for the level-true concatenated / factored /
# precomputed-y plans and 0 for the hierarchical dense plan, whose operator table is
# stored per union offset and scaled per level.
function _cuda_hier_route_compact_kernel!(route_levels, route_offsets, route_targets,
        route_sources, route_class, flags, prefix, node_at, node_coords, push_offsets,
        level_base_L, first_source, n_sources, first_offset, kn, L, class_base)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    idx > kn * n_sources && return nothing
    @inbounds begin
        flags[idx] == Int32(1) || return nothing
        kloc = (idx - 1) ÷ n_sources + 1
        s = (idx - 1) % n_sources + 1
        k = first_offset + kloc - 1
        source = first_source + s - 1
        G = 1 << L
        ox = push_offsets[1, k]
        oy = push_offsets[2, k]
        oz = push_offsets[3, k]
        tx = node_coords[1, source] + ox
        ty = node_coords[2, source] + oy
        tz = node_coords[3, source] + oz
        linear = tx + G * (ty + G * tz)
        target = Int(node_at[level_base_L + linear + 1])
        p = Int(prefix[idx])
        route_levels[p] = L
        route_offsets[1, p] = Int(ox)
        route_offsets[2, p] = Int(oy)
        route_offsets[3, p] = Int(oz)
        route_targets[p] = target
        route_sources[p] = source
        route_class[p] = Int32(class_base + k)
    end
    return nothing
end

# Hierarchical direct pairs: leaf level only, target cells ascending with near
# offsets in table order, matching build_hierarchical_direct_pairs! elementwise.
# The near set is a separate table from the push-offset phase mask and must not be
# phase-masked. Endpoints are the leaf cell index and the leaf-relative source
# index the direct stage expects.
function _cuda_hier_direct_flags_kernel!(flags, node_at, node_coords, near_offsets,
        fbase, len, kn, leaf_base, level_base_L, ell)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    idx > len && return nothing
    g = fbase + idx
    c = (g - 1) ÷ kn + 1
    k = (g - 1) % kn + 1
    G = 1 << ell
    @inbounds begin
        target_node = leaf_base + c
        sx = node_coords[1, target_node] - near_offsets[1, k]
        sy = node_coords[2, target_node] - near_offsets[2, k]
        sz = node_coords[3, target_node] - near_offsets[3, k]
        src = Int32(0)
        if 0 <= sx < G && 0 <= sy < G && 0 <= sz < G
            linear = sx + G * (sy + G * sz)
            src = node_at[level_base_L + linear + 1]
        end
        flags[idx] = src == Int32(0) ? Int32(0) : Int32(1)
    end
    return nothing
end

function _cuda_hier_direct_compact_kernel!(direct_targets, direct_sources, flags,
        prefix, node_at, node_coords, near_offsets, fbase, len, kn, leaf_base,
        level_base_L, ell, base)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    idx > len && return nothing
    @inbounds begin
        flags[idx] == Int32(1) || return nothing
        g = fbase + idx
        c = (g - 1) ÷ kn + 1
        k = (g - 1) % kn + 1
        G = 1 << ell
        target_node = leaf_base + c
        sx = node_coords[1, target_node] - near_offsets[1, k]
        sy = node_coords[2, target_node] - near_offsets[2, k]
        sz = node_coords[3, target_node] - near_offsets[3, k]
        linear = sx + G * (sy + G * sz)
        src = Int(node_at[level_base_L + linear + 1])
        p = base + Int(prefix[idx])
        direct_targets[p] = c
        direct_sources[p] = src - leaf_base
    end
    return nothing
end

# Refresh the persistent per-level occupancy lookup: one zero fill plus one scatter
# over the valid occupied-node prefix. `node_at` keeps its object identity for the
# cache's lifetime.
function _cuda_hier_refresh_occupancy!(hctx::DeviceHierarchicalM2LContext,
        grid::DeviceRadixGrid, level_offsets::Vector{Int})
    copyto!(hctx.level_offsets, level_offsets)
    n_nodes = level_offsets[end]
    n_nodes <= typemax(Int32) || throw(ArgumentError(
        "device hierarchical occupancy requires flat node indices to fit Int32; " *
        "got $n_nodes occupied nodes"))
    fill!(hctx.node_at, Int32(0))
    @inbounds for level in 0:hctx.ell
        hctx.nodes_per_level[level + 1] =
            level_offsets[level + 2] - level_offsets[level + 1]
    end
    n_nodes == 0 && return hctx
    threads = 256
    blocks = cld(n_nodes, threads)
    CUDA.@cuda threads=threads blocks=blocks _cuda_hier_node_at_scatter_kernel!(
        hctx.node_at, grid.node_levels, grid.node_coords, hctx.d_level_base, n_nodes,
    )
    return hctx
end

function _cuda_hier_generate_direct_pairs!(ctx, hctx::DeviceHierarchicalM2LContext,
        grid::DeviceRadixGrid, n_cells::Int, leaf_base::Int, ell::Int)
    kn = size(hctx.d_near_offsets, 2)
    (n_cells > 0 && kn > 0) || return 0
    threads = 256
    level_base_L = hctx.level_base[ell + 1]
    total = kn * n_cells
    capacity = length(ctx.direct_flags)
    capacity > 0 ||
        throw(AssertionError("device direct flag buffer has zero capacity"))
    n_direct = 0
    f0 = 0
    while f0 < total
        len = min(capacity, total - f0)
        blocks = cld(len, threads)
        CUDA.@cuda threads=threads blocks=blocks _cuda_hier_direct_flags_kernel!(
            ctx.direct_flags, hctx.node_at, grid.node_coords, hctx.d_near_offsets,
            f0, len, kn, leaf_base, level_base_L, ell,
        )
        fv = view(ctx.direct_flags, 1:len)
        pv = view(ctx.direct_prefix, 1:len)
        accumulate!(+, pv, fv)
        copyto!(ctx.host_scalar32, 1, ctx.direct_prefix, len, 1)
        chunk_total = Int(ctx.host_scalar32[1])
        if chunk_total > 0
            n_direct + chunk_total <= length(ctx.direct_targets) ||
                throw(AssertionError("device hierarchical direct pair buffer exceeded its capacity"))
            CUDA.@cuda threads=threads blocks=blocks _cuda_hier_direct_compact_kernel!(
                ctx.direct_targets, ctx.direct_sources, ctx.direct_flags,
                ctx.direct_prefix, hctx.node_at, grid.node_coords,
                hctx.d_near_offsets, f0, len, kn, leaf_base, level_base_L, ell,
                n_direct,
            )
        end
        n_direct += chunk_total
        f0 += len
    end
    return n_direct
end

# Generate exactly one `(level, offset window)` into the start of the route buffers
# and return its route count. One scan scalar set (the `kn`-entry window prefix) is
# downloaded per window; nothing is appended to a growing list.
function _cuda_hier_generate_window!(state::DeviceResidentRadixState,
        hctx::DeviceHierarchicalM2LContext, route_class, L::Int, first_offset::Int,
        last_offset::Int, class_base::Int)
    first_source = hctx.level_offsets[L + 1] + 1
    n_sources = hctx.level_offsets[L + 2] - hctx.level_offsets[L + 1]
    kn = last_offset - first_offset + 1
    (n_sources > 0 && kn > 0) || return 0
    used = kn * n_sources
    used <= length(hctx.route_flags) || throw(AssertionError(
        "device hierarchical window flag buffer exceeded its capacity " *
        "($(length(hctx.route_flags)) < $used); reduce window_classes"))
    threads = 256
    blocks = cld(used, threads)
    level_base_L = hctx.level_base[L + 1]
    CUDA.@cuda threads=threads blocks=blocks _cuda_hier_route_flags_kernel!(
        hctx.route_flags, hctx.node_at, state.grid.node_coords, hctx.d_push_offsets,
        hctx.d_class_of, level_base_L, first_source, n_sources, first_offset, kn, L,
    )
    fv = view(hctx.route_flags, 1:used)
    pv = view(hctx.route_prefix, 1:used)
    accumulate!(+, pv, fv)
    cum_blocks = cld(kn, threads)
    CUDA.@cuda threads=threads blocks=cum_blocks _cuda_hier_window_cum_kernel!(
        hctx.window_cum, hctx.route_prefix, n_sources, kn,
    )
    copyto!(hctx.host_window_cum, 1, hctx.window_cum, 1, kn)
    n_routes = Int(hctx.host_window_cum[kn])
    n_routes == 0 && return 0
    n_routes <= length(state.route_targets) || throw(AssertionError(
        "device hierarchical route window exceeded capacity " *
        "$(length(state.route_targets)); increase window storage or reduce window_classes"))
    CUDA.@cuda threads=threads blocks=blocks _cuda_hier_route_compact_kernel!(
        state.route_levels, state.route_offsets, state.route_targets,
        state.route_sources, route_class, hctx.route_flags, hctx.route_prefix,
        hctx.node_at, state.grid.node_coords, hctx.d_push_offsets, level_base_L,
        first_source, n_sources, first_offset, kn, L, class_base,
    )
    return n_routes
end

# Window-local class partition for the plans that consume per-class counts/starts.
# The route buffers are overwritten every window, so a whole-plan histogram would be
# both wrong (stale counts from earlier windows) and far more host traffic than the
# task-023 contract allows. Counts come from the window prefix already downloaded by
# the generator; entries outside the active window are cleared, and `normalize`
# additionally makes the starts globally monotone for the GEMM / per-class reference
# drivers that scan every class.
function _cuda_hier_refresh_precomputed_y_window!(plan::ResidentM2LPrecomputedYPlan,
        hctx::DeviceHierarchicalM2LContext, lo::Int, hi::Int, n_routes::Int)
    host_counts = plan.host_class_counts::Vector{Int32}
    counts = plan.offset_counts
    starts = plan.offset_starts
    @inbounds for k in hctx.window_lo:hctx.window_hi
        host_counts[k] = Int32(0)
        counts[k] = 0
    end
    cursor = 1
    total = 0
    @inbounds for k in lo:hi
        prev = k == lo ? 0 : Int(hctx.host_window_cum[k - lo])
        c = Int(hctx.host_window_cum[k - lo + 1]) - prev
        host_counts[k] = Int32(c)
        counts[k] = c
        starts[k] = cursor
        cursor += c
        total += c
    end
    total == n_routes || throw(AssertionError(
        "hierarchical precomputed-y M2L window classes $lo:$hi do not partition " *
        "the window's $n_routes routes (summed $total)"))
    starts[hi + 1] = cursor
    if !PRECOMPUTED_CUDA_WHOLE_PASS[]
        # the per-class reference driver scans every class, so the starts outside
        # the active window must stay monotone around it
        @inbounds for k in 1:(lo - 1)
            starts[k] = 1
        end
        @inbounds for k in (hi + 2):length(starts)
            starts[k] = cursor
        end
    end
    hctx.window_lo = lo
    hctx.window_hi = hi
    return plan
end

function _cuda_hier_refresh_dense_window!(plan::ResidentM2LDenseCUDAPlan,
        hctx::DeviceHierarchicalM2LContext, lo::Int, hi::Int, n_routes::Int)
    host_counts = plan.host_class_counts::Vector{Int32}
    @inbounds for k in hctx.window_lo:hctx.window_hi
        host_counts[k] = Int32(0)
    end
    starts = plan.class_starts
    cursor = 1
    total = 0
    @inbounds for k in lo:hi
        prev = k == lo ? 0 : Int(hctx.host_window_cum[k - lo])
        c = Int(hctx.host_window_cum[k - lo + 1]) - prev
        c <= plan.class_capacities[k] || throw(AssertionError(
            "hierarchical dense M2L class $k count $c exceeds capacity $(plan.class_capacities[k])"))
        host_counts[k] = Int32(c)
        starts[k] = cursor
        cursor += c
        total += c
    end
    total == n_routes || throw(AssertionError(
        "hierarchical dense M2L window classes $lo:$hi do not partition the " *
        "window's $n_routes routes (summed $total)"))
    starts[hi + 1] = cursor
    if !DENSE_CUDA_FUSED[]
        @inbounds for k in 1:(lo - 1)
            starts[k] = 1
        end
        @inbounds for k in (hi + 2):length(starts)
            starts[k] = cursor
        end
    end
    hctx.window_lo = lo
    hctx.window_hi = hi
    return plan
end

# Fused hierarchical dense M2L: identical to the flat fused kernel except that the
# task-025 level scaling `K(s r0) = s^-1 Lambda(s) K(r0) Lambda(s)` is applied as
# row factors — the source diagonal before the operator, the target diagonal after —
# so one operator per union offset serves every level. The two diagonals differ (and
# are asymmetric across the Lamb-Helmholtz phi/chi row blocks), so they are separate
# columns of `src_scale`/`tgt_scale` indexed by the window's level.
# Task 028 lever 2: grid-stride over routes instead of one block per route.
# Previously `blocks = n_routes`, which at n=1e6/ell=5 is 31,307,680 blocks for
# 20.28 ms of leaf M2L (~1.5 G blocks/s) -- block-dispatch rate, not compute or
# bandwidth (a Float32/Float64 A/B moved the stage 22.41 -> 23.55 ms, ruling
# both out). The block count is now capped by DENSE_CUDA_FUSED_MAX_BLOCKS and
# each block walks many routes.
#
# `j` advances by `gridDim()`, so it is uniform across the block and every
# thread reaches both barriers the same number of times.
function _cuda_hier_dense_fused_kernel!(loc_phi, loc_chi, ops, route_class,
        route_sources, route_targets, mp_phi, mp_chi, phi_flat_idx, chi_flat_idx,
        ndof_phi, src_scale, tgt_scale, lcol, n_routes, ::Val{LH}) where LH
    T = eltype(ops)
    D = size(ops, 1)
    tid = threadIdx().x
    nthreads = blockDim().x
    shm = CUDA.CuDynamicSharedArray(T, D)
    j = blockIdx().x
    @inbounds while j <= n_routes
        src_col = route_sources[j]
        k = Int(route_class[j])
        tgt_col = route_targets[j]
        i = tid
        while i <= D
            if i <= ndof_phi
                shm[i] = mp_phi[phi_flat_idx[i], src_col] * src_scale[i, lcol]
            elseif LH
                shm[i] = mp_chi[chi_flat_idx[i - ndof_phi], src_col] * src_scale[i, lcol]
            else
                shm[i] = zero(T)
            end
            i += nthreads
        end
        CUDA.sync_threads()
        r = tid
        while r <= D
            acc = zero(T)
            for i in 1:D
                acc += ops[r, i, k] * shm[i]
            end
            acc *= tgt_scale[r, lcol]
            if r <= ndof_phi
                CUDA.@atomic loc_phi[phi_flat_idx[r], tgt_col] += acc
            elseif LH
                CUDA.@atomic loc_chi[chi_flat_idx[r - ndof_phi], tgt_col] += acc
            end
            r += nthreads
        end
        # WAR barrier: the next iteration's gather must not overwrite `shm`
        # while a slower thread is still reading it in the matvec above. The
        # single-route-per-block original never looped, so it needed only the
        # RAW barrier.
        CUDA.sync_threads()
        j += gridDim().x
    end
    return nothing
end

# Task 028 cycle 2: operator-tiled variant of the fused kernel above. Each block
# owns a contiguous route chunk; within it, same-class segments (route_class is
# non-decreasing inside a window) are processed with the class operator staged
# once in shared memory, both level diagonals pre-folded:
# tile[r + (i-1)D] = tgt_scale[r] * ops[r,i,k] * src_scale[i]. Warps then stream
# routes through the tile — lanes stride rows, the multipole column sits in a
# per-warp shared slice — so the per-route global traffic drops from D^2 + D
# loads to D loads (plus the unchanged D atomics). Block-level barriers only
# bracket tile (re)loads; j0/je/hi are computed identically by every thread from
# the same route_class reads, so the segment loop is block-uniform.
function _cuda_hier_dense_tiled_kernel!(loc_phi, loc_chi, ops, route_class,
        route_sources, route_targets, mp_phi, mp_chi, phi_flat_idx, chi_flat_idx,
        ndof_phi, src_scale, tgt_scale, lcol, n_routes, ::Val{LH}) where LH
    T = eltype(ops)
    D = size(ops, 1)
    tid = threadIdx().x
    nthreads = blockDim().x
    lane = Int((tid - Int32(1)) % Int32(32))
    w = Int((tid - Int32(1)) ÷ Int32(32))
    nwarps = Int(nthreads ÷ Int32(32))
    tile = CUDA.CuDynamicSharedArray(T, D * D)
    mp_buf = CUDA.CuDynamicSharedArray(T, (D, nwarps), D * D * sizeof(T))
    chunk = cld(n_routes, gridDim().x)
    j0 = (blockIdx().x - 1) * chunk + 1
    hi = min(j0 + chunk - 1, n_routes)
    @inbounds while j0 <= hi
        k = Int(route_class[j0])
        # binary search for the segment end (last route of class k in [j0, hi])
        slo = j0
        shi = hi
        while slo < shi
            mid = (slo + shi + 1) >> 1
            if Int(route_class[mid]) == k
                slo = mid
            else
                shi = mid - 1
            end
        end
        je = slo
        idx = Int(tid)
        while idx <= D * D
            r = (idx - 1) % D + 1
            i = (idx - 1) ÷ D + 1
            tile[idx] = tgt_scale[r, lcol] * ops[r, i, k] * src_scale[i, lcol]
            idx += nthreads
        end
        CUDA.sync_threads()
        j = j0 + w
        while j <= je
            src_col = route_sources[j]
            tgt_col = route_targets[j]
            i = lane + 1
            while i <= D
                if i <= ndof_phi
                    mp_buf[i, w + 1] = mp_phi[phi_flat_idx[i], src_col]
                elseif LH
                    mp_buf[i, w + 1] = mp_chi[chi_flat_idx[i - ndof_phi], src_col]
                else
                    mp_buf[i, w + 1] = zero(T)
                end
                i += 32
            end
            CUDA.sync_warp()
            r = lane + 1
            while r <= D
                acc = zero(T)
                for i in 1:D
                    acc += tile[r + (i - 1) * D] * mp_buf[i, w + 1]
                end
                if r <= ndof_phi
                    CUDA.@atomic loc_phi[phi_flat_idx[r], tgt_col] += acc
                elseif LH
                    CUDA.@atomic loc_chi[chi_flat_idx[r - ndof_phi], tgt_col] += acc
                end
                r += 32
            end
            # WAR: this warp's next route rewrites its mp_buf slice
            CUDA.sync_warp()
            j += nwarps
        end
        # WAR: the next segment's tile load must wait for every warp's matvec
        CUDA.sync_threads()
        j0 = je + 1
    end
    return nothing
end

# Scaled gather/scatter for the unfused GEMM reference drivers: the same level
# diagonals applied at the slab boundaries instead of inside the fused kernel, so
# the two routes are independently testable.
function _cuda_hier_dense_gather_kernel!(slab, phi, chi, phi_flat_idx, chi_flat_idx,
        src_cols, ndof_phi, src_scale, lcol, ::Val{LH}) where LH
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    ndof = size(slab, 1)
    idx > ndof * size(slab, 2) && return nothing
    row = (idx - 1) % ndof + 1
    j = (idx - 1) ÷ ndof + 1
    @inbounds begin
        col = src_cols[j]
        if row <= ndof_phi
            slab[row, j] = phi[phi_flat_idx[row], col] * src_scale[row, lcol]
        elseif LH
            slab[row, j] = chi[chi_flat_idx[row - ndof_phi], col] * src_scale[row, lcol]
        end
    end
    return nothing
end

function _cuda_hier_dense_scatter_kernel!(phi, chi, slab, phi_flat_idx, chi_flat_idx,
        tgt_cols, ndof_phi, tgt_scale, lcol, ::Val{LH}) where LH
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    ndof = size(slab, 1)
    idx > ndof * size(slab, 2) && return nothing
    row = (idx - 1) % ndof + 1
    j = (idx - 1) ÷ ndof + 1
    @inbounds begin
        col = tgt_cols[j]
        if row <= ndof_phi
            CUDA.@atomic phi[phi_flat_idx[row], col] += slab[row, j] * tgt_scale[row, lcol]
        elseif LH
            CUDA.@atomic chi[chi_flat_idx[row - ndof_phi], col] +=
                slab[row, j] * tgt_scale[row, lcol]
        end
    end
    return nothing
end

function _cuda_hier_dense_gather!(slab, source::FlatCoefficientBuffer, phi_flat_idx,
        chi_flat_idx, src_cols, ndof_phi::Int, src_scale, lcol::Int, lh::Val)
    n_el = length(slab)
    n_el == 0 && return slab
    threads = 256
    blocks = cld(n_el, threads)
    CUDA.@cuda threads=threads blocks=blocks _cuda_hier_dense_gather_kernel!(
        slab, source.phi, source.chi, phi_flat_idx, chi_flat_idx, src_cols,
        ndof_phi, src_scale, lcol, lh,
    )
    return slab
end

function _cuda_hier_dense_scatter_add!(target::FlatCoefficientBuffer, slab,
        phi_flat_idx, chi_flat_idx, tgt_cols, ndof_phi::Int, tgt_scale, lcol::Int,
        lh::Val)
    n_el = length(slab)
    n_el == 0 && return target
    threads = 256
    blocks = cld(n_el, threads)
    CUDA.@cuda threads=threads blocks=blocks _cuda_hier_dense_scatter_kernel!(
        target.phi, target.chi, slab, phi_flat_idx, chi_flat_idx, tgt_cols,
        ndof_phi, tgt_scale, lcol, lh,
    )
    return target
end

# Apply one generated window through the hierarchical dense plan at level `L`.
function _cuda_hier_dense_apply_window!(state::DeviceResidentRadixState{TF,B,LH},
        ws::ResidentOperatorWorkspace{TF,B,LH}, plan::ResidentM2LDenseCUDAPlan,
        hctx::DeviceHierarchicalM2LContext, L::Int) where {TF,B,LH}
    n_routes = state.counts.n_routes
    n_routes == 0 && return state
    lcol = L - 1
    if DENSE_CUDA_FUSED[]
        D = plan.ndof
        # tiled path (task 028 cycle 2): 128 threads = 4 warps; shm holds the
        # folded D x D class tile plus one multipole column per warp
        tiled_shmem = (D * D + 4 * D) * sizeof(TF)
        if DENSE_CUDA_TILED[] && n_routes >= DENSE_CUDA_TILED_MIN_ROUTES[] &&
                tiled_shmem <= 48 * 1024
            blocks = min(cld(n_routes, 4), DENSE_CUDA_FUSED_MAX_BLOCKS[])
            CUDA.@cuda threads=128 blocks=blocks shmem=tiled_shmem _cuda_hier_dense_tiled_kernel!(
                state.locals.phi, state.locals.chi, plan.operators, plan.route_class,
                state.route_sources, state.route_targets, state.multipoles.phi,
                state.multipoles.chi, ws.phi_flat_idx, ws.chi_flat_idx, plan.ndof_phi,
                hctx.source_scale, hctx.target_scale, lcol, n_routes, Val(LH),
            )
            return state
        end
        threads = min(256, cld(D, 32) * 32)
        shmem = D * sizeof(TF)
        blocks = min(n_routes, DENSE_CUDA_FUSED_MAX_BLOCKS[])
        CUDA.@cuda threads=threads blocks=blocks shmem=shmem _cuda_hier_dense_fused_kernel!(
            state.locals.phi, state.locals.chi, plan.operators, plan.route_class,
            state.route_sources, state.route_targets, state.multipoles.phi,
            state.multipoles.chi, ws.phi_flat_idx, ws.chi_flat_idx, plan.ndof_phi,
            hctx.source_scale, hctx.target_scale, lcol, n_routes, Val(LH),
        )
        return state
    end
    wp = plan.whole_pass[]
    wp isa NamedTuple || throw(ArgumentError(
        "CUDA hierarchical dense M2L requires the whole-pass scratch bundle"))
    starts = plan.class_starts
    W = wp.chunk
    ndof_phi = plan.ndof_phi
    kcur = hctx.window_lo
    @inbounds for c0 in 1:W:n_routes
        n = min(W, n_routes - c0 + 1)
        chi_hi = c0 + n - 1
        _cuda_hier_dense_gather!(_matrix_col_view(plan.src_slab, n), state.multipoles,
            ws.phi_flat_idx, ws.chi_flat_idx, view(state.route_sources, c0:chi_hi),
            ndof_phi, hctx.source_scale, lcol, Val(LH))
        while kcur < hctx.window_hi && starts[kcur + 1] <= c0
            kcur += 1
        end
        k = kcur
        while k <= hctx.window_hi && starts[k] <= chi_hi
            lo = max(starts[k], c0)
            hi = min(starts[k + 1] - 1, chi_hi)
            if hi >= lo
                _cuda_dense_class_gemm!(plan.dst_slab, plan.operators, k, plan.src_slab,
                    lo - c0 + 1, hi - c0 + 1, wp.alpha, wp.beta)
            end
            k += 1
        end
        _cuda_hier_dense_scatter_add!(state.locals, _matrix_col_view(plan.dst_slab, n),
            ws.phi_flat_idx, ws.chi_flat_idx, view(state.route_targets, c0:chi_hi),
            ndof_phi, hctx.target_scale, lcol, Val(LH))
    end
    return state
end

"""
    _launch_cuda_hierarchical_m2l!(state, hctx)

Genuine multilevel device M2L. Locals are zeroed once, then every
`(level, offset window)` is generated and applied immediately with
`clear_locals=false`, so the reusable route buffers hold one window at a time.
`state.counts.n_routes` tracks the active window during the pass and is restored to
the step's total telemetry count at the end.
"""
function _launch_cuda_hierarchical_m2l!(state::DeviceResidentRadixState{TF,B,LH},
        hctx::DeviceHierarchicalM2LContext) where {TF,B,LH}
    ws = state.scratch
    ws isa ResidentOperatorWorkspace || throw(ArgumentError(
        "hierarchical device M2L requires ResidentOperatorWorkspace scratch"))
    plan = hctx.apply_plan
    plan isa Union{ResidentM2LConcatPlan,ResidentM2LPrecomputedYPlan,
        ResidentM2LDenseCUDAPlan} || throw(ArgumentError(
        "hierarchical device M2L has no compatible construction-time plan; got $(typeof(plan))"))
    fill!(state.locals.phi, zero(TF))
    LH && fill!(state.locals.chi, zero(TF))
    route_class = plan.route_class
    noffsets = hctx.noffsets
    K = hctx.window_classes
    ell = hctx.ell
    dense = plan isa ResidentM2LDenseCUDAPlan
    total = 0
    fill!(hctx.routes_per_level, 0)
    if hctx.profile_stages
        fill!(hctx.m2l_level_ns, 0)
        # stage 4 accumulates the flag/scan/compact cost across every window, so
        # it can be separated from the per-level apply cost in m2l_level_ns
        hctx.update_stage_ns[4] = 0
    end
    for L in 2:ell
        if hctx.profile_stages
            CUDA.synchronize()
            t_level = time_ns()
        else
            t_level = UInt64(0)
        end
        level_total = 0
        class_base = dense ? 0 : (L - 2) * noffsets
        for first_offset in 1:K:noffsets
            last_offset = min(first_offset + K - 1, noffsets)
            t_gen = hctx.profile_stages ? time_ns() : UInt64(0)
            n = _cuda_hier_generate_window!(state, hctx, route_class, L, first_offset,
                last_offset, class_base)
            hctx.profile_stages &&
                (hctx.update_stage_ns[4] += time_ns() - t_gen)
            hctx.last_window_routes = n
            state.counts.n_routes = n
            if n > 0
                if dense
                    _cuda_hier_refresh_dense_window!(plan, hctx, first_offset,
                        last_offset, n)
                    _cuda_hier_dense_apply_window!(state, ws, plan, hctx, L)
                elseif plan isa ResidentM2LPrecomputedYPlan
                    _cuda_hier_refresh_precomputed_y_window!(plan, hctx,
                        class_base + first_offset, class_base + last_offset, n)
                    _launch_resident_m2l_precomputed_y_plan!(state, ws, plan;
                        clear_locals=false)
                else
                    _launch_resident_m2l_concat!(state; clear_locals=false)
                end
            end
            level_total += n
        end
        hctx.routes_per_level[L + 1] = level_total
        if hctx.profile_stages
            CUDA.synchronize()
            hctx.m2l_level_ns[L + 1] = time_ns() - t_level
        end
        total += level_total
    end
    hctx.total_routes = total
    state.counts.n_routes = total
    return state
end

# Per-level Lambda columns for the hierarchical dense operator table. Level L uses
# s = 2^(ell - L) (the leaf level is the reference at which the operators were
# built); the phi rows scale as s^-n / s^-(n+1) and the Lamb-Helmholtz chi rows as
# s^-(n-1) / s^-(n+2) — the asymmetric pair the host plan encodes per class.
function _cuda_hier_dense_scales(::Type{TF}, basis_info::OperatorBasisInfo{B,LH},
        ell::Int, D::Int) where {TF,B,LH}
    nlevels = max(ell - 1, 0)
    source_scale = ones(TF, D, nlevels)
    target_scale = ones(TF, D, nlevels)
    Dphi = degree_major_dof(basis_info.orders.P_phi)
    @inbounds for L in 2:ell
        col = L - 1
        s = TF(1 << (ell - L))
        for n in 0:basis_info.orders.P_phi, row in degree_row_range(n)
            source_scale[row, col] = s^(-n)
            target_scale[row, col] = s^(-(n + 1))
        end
        if LH
            for n in 0:basis_info.orders.P_active, row in degree_row_range(n)
                rr = Dphi + row
                source_scale[rr, col] = s^(-(n - 1))
                target_scale[rr, col] = s^(-(n + 2))
            end
        end
    end
    return source_scale, target_scale
end

# Construct the device hierarchical context. Every table here is step-invariant and
# uploads exactly once (counted with the construction operator upload); the window
# flag/prefix buffers and the per-level occupancy lookup are persistent and refreshed
# in place, so recurring steps allocate nothing and preserve array identity.
function _build_cuda_hierarchical_context(::Type{TF}, basis_info::OperatorBasisInfo{B,LH},
        policy::HierarchicalRigidStencil, tables::RigidHierarchicalTables,
        class_level::Vector{Int32}, class_offset::Matrix{Int32},
        effective_offsets::Vector{SVector{3,Int}}, plan, ell::Int,
        max_level_nodes::Int, counters::CUDARadixTransferCounters) where {TF,B,LH}
    occupancy = RadixLevelOccupancy(ell; max_bytes=policy.dense_occupancy_max_bytes,
        max_dense_ell=policy.dense_occupancy_max_ell)
    isempty(occupancy.node_at) && throw(ArgumentError(
        "device HierarchicalRigidStencil requires the dense per-level occupancy " *
        "lookup, but ell=$ell exceeds the configured budget " *
        "(dense_occupancy_max_bytes=$(policy.dense_occupancy_max_bytes), " *
        "dense_occupancy_max_ell=$(policy.dense_occupancy_max_ell)); the host " *
        "Morton binary-search fallback has no device implementation. Raise the " *
        "budget, lower ell, or run this policy host-resident."))
    noffsets = length(tables.push_offsets)
    K = min(policy.window_classes, noffsets)
    flag_capacity = max(K * max_level_nodes, 1)
    node_at = CUDA.zeros(Int32, length(occupancy.node_at))
    d_level_base = CUDA.CuArray{Int}(occupancy.level_base)
    d_push_offsets = CUDA.CuArray{Int32}(_radix_offsets_matrix(tables.push_offsets))
    d_class_of = CUDA.CuArray{Int32}(tables.class_of)
    d_near_offsets = CUDA.CuArray{Int32}(_radix_offsets_matrix(tables.near_offsets))
    counters.operator_uploads += 1
    dense = plan isa ResidentM2LDenseCUDAPlan
    host_source_scale, host_target_scale = dense ?
        _cuda_hier_dense_scales(TF, basis_info, ell, plan.ndof) :
        (Matrix{TF}(undef, 0, 0), Matrix{TF}(undef, 0, 0))
    source_scale = CUDA.CuArray{TF}(host_source_scale)
    target_scale = CUDA.CuArray{TF}(host_target_scale)
    return DeviceHierarchicalM2LContext(
        tables, class_level, class_offset, effective_offsets, plan,
        K, ell, noffsets,
        copy(occupancy.level_base), zeros(Int, ell + 2),
        node_at, d_level_base, d_push_offsets, d_class_of, d_near_offsets,
        CUDA.zeros(Int32, flag_capacity), CUDA.zeros(Int32, flag_capacity),
        CUDA.zeros(Int32, max(K, 1)), _pin_host_array(zeros(Int32, max(K, 1))),
        source_scale, target_scale,
        0, zeros(Int, ell + 1), zeros(Int, ell + 1), 0, 1, 0,
        false, zeros(UInt64, 5), zeros(UInt64, ell + 1),
    )
end

# Int32 entry count of the dense per-level `node_at` lookup, or 0 when the
# configured budget disables it (construction then throws with a precise message).
function _cuda_hier_occupancy_words(policy::HierarchicalRigidStencil, ell::Int)
    occupancy = RadixLevelOccupancy(ell; max_bytes=policy.dense_occupancy_max_bytes,
        max_dense_ell=policy.dense_occupancy_max_ell)
    return length(occupancy.node_at)
end

"""
    cuda_hierarchical_route_window!(state, level, first_offset, last_offset)

Generate one `(level, offset window)` into the state's route buffers and return its
route count, using the plan's own class convention. Exposed for the host/device
route-parity tests; the lifecycle drives windows through
[`_launch_cuda_hierarchical_m2l!`](@ref).
"""
function cuda_hierarchical_route_window!(state::DeviceResidentRadixState,
        level::Integer, first_offset::Integer, last_offset::Integer)
    hctx = state.interaction_list
    hctx isa DeviceHierarchicalM2LContext || throw(ArgumentError(
        "cuda_hierarchical_route_window! requires a hierarchical device state"))
    plan = hctx.apply_plan
    class_base = plan isa ResidentM2LDenseCUDAPlan ? 0 :
        (Int(level) - 2) * hctx.noffsets
    n = _cuda_hier_generate_window!(state, hctx, plan.route_class, Int(level),
        Int(first_offset), Int(last_offset), class_base)
    hctx.last_window_routes = n
    state.counts.n_routes = n
    return n
end
