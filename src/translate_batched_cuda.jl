#------- CUDA DEVICE-RESIDENT RADIX LIFECYCLE (Matrix Operator Refactor, task 022) -------#
#
# This file is opt-in via load_cuda_radix_lifecycle!(). It keeps CUDA symbols out of
# the default CPU include path while providing a task-023-ready state object and
# transfer accounting for the resident radix operator path.

const CUDA = Base.require(Base.PkgId(Base.UUID("052768ef-5323-5732-b1bb-66c8b64840ba"), "CUDA"))
# BFloat16 lives in CUDACore on CUDA.jl >= 6.2 and comes from BFloat16s.jl on
# the 5.8 line (which task 052 uses to coexist with PrettyTables-2 geo stacks).
const CUDABFloat16 = isdefined(CUDA, :CUDACore) ? CUDA.CUDACore.BFloat16 :
    Base.require(Base.PkgId(Base.UUID("ab4f0b2a-ad5b-11e8-123f-65d77653426b"), "BFloat16s")).BFloat16

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
# clamped into an edge cell. The in-box check is per-axis (task 037 stage 2):
# rectangular caches pass box_extent < 2h0 on their short axes; cubic caches
# pass (2h0, 2h0, 2h0). Key quantization stays on the virtual cube (h0, ell).
function _cuda_radix_keys_checked_kernel!(keys, oob_flag, positions, x_min,
        box_extent, h0, ell)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > length(keys) && return nothing
    G = 1 << ell
    delta = (2 * h0) / G
    @inbounds begin
        px = positions[1, i]
        py = positions[2, i]
        pz = positions[3, i]
        if !(x_min[1] <= px <= x_min[1] + box_extent[1] &&
             x_min[2] <= py <= x_min[2] + box_extent[2] &&
             x_min[3] <= pz <= x_min[3] + box_extent[3])
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

# Task 028 Stage 6 bounded-key counting-sort prototype.  The cache's fixed
# Morton depth bounds keys to `0:2^(3ell)-1`; histogram, scan and scatter storage
# is persistent and the whole path is included in recurring refresh timing.
const RADIX_CUDA_COUNTING_SORT = Ref(true)
const RADIX_CUDA_COUNTING_SORT_MAX_ELL = Ref(6)
@inline _cuda_counting_sort_enabled(ell::Int) =
    radix_setting(:RADIX_CUDA_COUNTING_SORT) && ell <= radix_setting(:RADIX_CUDA_COUNTING_SORT_MAX_ELL)

# The histogram is sized at construction from `_cuda_counting_sort_enabled`, so a
# step must also confirm the buffer it is about to scatter through actually spans
# the key domain: flipping the knob on after construction would otherwise drive
# `@inbounds` atomics through a length-1 array. Falling back is always safe.
@inline _cuda_counting_sort_ready(ctx, ell::Int) =
    _cuda_counting_sort_enabled(ell) && length(ctx.counting_histogram) == 1 << (3ell)

function _cuda_counting_histogram_kernel!(histogram, keys)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > length(keys) && return nothing
    @inbounds CUDA.@atomic histogram[Int(keys[i]) + 1] += Int32(1)
    return nothing
end

function _cuda_counting_cursor_kernel!(cursor, prefix)
    k = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    k > length(cursor) && return nothing
    @inbounds cursor[k] = k == 1 ? Int32(0) : prefix[k - 1]
    return nothing
end


function _cuda_counting_scatter_kernel!(perm, sorted_keys, cursor, keys)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > length(keys) && return nothing
    @inbounds begin
        key = keys[i]
        slot = Int(CUDA.atomic_add!(pointer(cursor, Int(key) + 1), Int32(1))) + 1
        perm[slot] = i
        sorted_keys[slot] = key
    end
    return nothing
end

# Construct BF16 tensor-operator storage on device.  Broadcasting
# `CUDABFloat16.(Kbuf)` on the host makes Julia's x86 LLVM backend select a
# vector BF16 rounding instruction that is unavailable on the cluster login and
# compute-node CPUs, even when the optional tensor path is disabled at runtime.
# H200 supports the scalar conversion natively, and this cache is built only
# once with the rest of the resident operator plan.
function _cuda_convert_bf16_kernel!(dest, source)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > length(dest) && return nothing
    @inbounds dest[i] = CUDABFloat16(source[i])
    return nothing
end

function _cuda_counting_sort_into!(perm, sorted_keys, keys, histogram, prefix,
        cursor, threads::Int)
    fill!(histogram, Int32(0))
    blocks = cld(length(keys), threads)
    CUDA.@cuda threads=threads blocks=blocks _cuda_counting_histogram_kernel!(
        histogram, keys)
    accumulate!(+, prefix, histogram)
    domain_blocks = cld(length(histogram), threads)
    CUDA.@cuda threads=threads blocks=domain_blocks _cuda_counting_cursor_kernel!(
        cursor, prefix)
    CUDA.@cuda threads=threads blocks=blocks _cuda_counting_scatter_kernel!(
        perm, sorted_keys, cursor, keys)
    return perm
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
        node_keys, level_offsets, x_min, h0, max_level, min_level)
    level = min_level + (blockIdx().y - 1)
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

function _cuda_parent_index_kernel!(parent_index, node_keys, level_offsets, max_level,
        min_level)
    level = min_level + (blockIdx().y - 1)
    local_i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    level > max_level && return nothing
    first = level_offsets[level + 1] + 1
    stop = level_offsets[level + 2]
    node = first + local_i - 1
    node > stop && return nothing
    @inbounds begin
        if level == min_level
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

function _cuda_child_ranges_kernel!(child_ranges, node_keys, level_offsets, max_level,
        min_level)
    level = min_level + (blockIdx().y - 1)
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
        parent_index, n_root_nodes)
    # edges are the children of the retained levels (task 037 stage 3): the
    # first n_root_nodes nodes are roots with parent_index 0 (legacy: 1)
    edge = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    node = edge + n_root_nodes
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
            node_levels, node_coords, node_centers, node_keys, level_offsets, x_min, h0, ell, 0,
        )
        CUDA.@cuda threads=threads blocks=(blocks_x, ell + 1) _cuda_parent_index_kernel!(
            parent_index, node_keys, level_offsets, ell, 0,
        )
        CUDA.@cuda threads=threads blocks=(blocks_x, ell + 1) _cuda_child_ranges_kernel!(
            child_ranges, node_keys, level_offsets, ell, 0,
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
            m2m_parent, m2m_child, l2l_parent, l2l_child, grid.parent_index, 1,
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
    # one-shot path only; the recurring cache path fills its persistent
    # per-system buffer in _radix_cache_refresh_source_buffers! (task 032)
    device_buffer = CUDA.CuArray{TF}(undef, data_per_body(system), get_n_bodies(system))
    return _fill_device_source_buffer!(device_buffer, system)
end

# identity permutation: a range, matching the documented `sort_index` default in
# compatibility.jl. `collect` here allocated an 8 MB Vector{Int} every step at
# n=1e6 (14% of per-step host allocation, task 028).
function _fill_device_source_buffer!(device_buffer, system)
    sort_index = Base.OneTo(get_n_bodies(system))
    _has_device_source_to_buffer_method(device_buffer, system, sort_index) ||
        throw(ArgumentError(
            "DeviceResident CUDA source systems must overload FastMultipole.source_to_buffer!(device_buffer, system, sort_index)",
        ))
    source_to_buffer!(device_buffer, system, sort_index)
    return device_buffer
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
    # canonical all-rows packed layout (task 032): every source-buffer row is
    # carried, including radius row 4; systems narrower than the packed matrix
    # are zero-padded
    nrows = size(body, 1)
    nsys = min(size(source_buffer, 1), nrows)
    @inbounds begin
        for row in 1:nsys
            body[row, sorted_i] = source_buffer[row, ibody]
        end
        for row in (nsys + 1):nrows
            body[row, sorted_i] = zero(eltype(body))
        end
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
    nrows = maximum(size(buffer, 1) for buffer in source_buffers)
    body = CUDA.CuArray{TF}(undef, nrows, n)
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

# Task 028 note: a warp-per-cell variant with lanes striding the (n, m) list was
# measured SLOWER on H200 (1.02 -> 2.03 ms F64 at n=1e6/ell=5, job 13015315:
# 113 registers and only 15 of 32 active lanes at P=4), so thread-per-cell was
# retained through task 030. Task 035 cycle 2 (user-approved 2026-08-12)
# replaces it with a block-per-cell, BODY-parallel form: the FLOWVPM vortex
# workload's sigma-adequacy gate forces shallow trees (~100-600 bodies per
# occupied leaf, a few hundred cells), where one thread per cell left the GPU
# nearly idle (measured 52-80% of the whole evaluation). Threads stride the
# cell's bodies inside the (n, m) loop — per-thread state stays tiny, unlike
# the failed (n, m)-striding variant — and a shared-memory tree reduction
# collapses the block partials. Launch config is epoch-constant host data
# (n_cells), so graph-capture eligibility is unchanged.
const CUDA_B2M_BLOCK = 128

# Tree-reduce (acc_re, acc_im) across the block; the returned pair is valid on
# thread 1 only. blockDim must be CUDA_B2M_BLOCK (a power of two). All threads
# of the block must call this (uniform control flow around it).
@inline function _cuda_b2m_block_reduce(shre, shim, tid, acc_re, acc_im)
    @inbounds shre[tid] = acc_re
    @inbounds shim[tid] = acc_im
    CUDA.sync_threads()
    s = CUDA_B2M_BLOCK >> 1
    while s >= 1
        if tid <= s
            @inbounds shre[tid] += shre[tid + s]
            @inbounds shim[tid] += shim[tid + s]
        end
        CUDA.sync_threads()
        s >>= 1
    end
    return (@inbounds shre[1]), (@inbounds shim[1])
end

function _cuda_b2m_leaf_nodes_kernel!(phi, source_bodies, cell_centers, cell_ranges,
        leaf_to_node, P, ncell)
    i_cell = blockIdx().x
    i_cell > ncell && return nothing
    tid = threadIdx().x
    nt = blockDim().x
    TF = eltype(phi)
    shre = CUDA.CuStaticSharedArray(TF, CUDA_B2M_BLOCK)
    shim = CUDA.CuStaticSharedArray(TF, CUDA_B2M_BLOCK)
    first = cell_ranges[1, i_cell]
    count = cell_ranges[2, i_cell]
    cx = cell_centers[1, i_cell]
    cy = cell_centers[2, i_cell]
    cz = cell_centers[3, i_cell]
    node = leaf_to_node[i_cell]
    @inbounds for n in 0:P
        for m in 0:n
            acc_re = zero(TF)
            acc_im = zero(TF)
            sgn = isodd(n + m) ? -one(TF) : one(TF)
            k = first + tid - 1
            while k <= first + count - 1
                dx = source_bodies[1, k] - cx
                dy = source_bodies[2, k] - cy
                dz = source_bodies[3, k] - cz
                q = source_bodies[5, k]
                rre, rim = _cuda_regular_harmonic_coeff(dx, dy, dz, n, m)
                scale = sgn * q
                acc_re += rre * scale
                acc_im -= rim * scale
                k += nt
            end
            re, im = _cuda_b2m_block_reduce(shre, shim, tid, acc_re, acc_im)
            if tid == 1
                row = _cuda_flat_basis_index(n, m, 1)
                phi[row, node] = re
                phi[row + 1, node] = im
            end
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
        body_system, body_index, isys, scalar_row, gradient_start, gradient_stop,
        hessian_start, hessian_stop, n_bodies)
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
        if hessian_start <= hessian_stop
            for k in 0:8
                target_buffer[hessian_start + k, ibody] = output[5 + k, sorted_i]
            end
        end
    end
    return nothing
end

function _copy_radix_output_to_device_target_buffer!(target_buffer, output,
        body_perm, body_system_ids, body_indices, isys::Integer, derivatives_switch,
        n_bodies::Integer=size(output, 2))
    fill!(target_buffer, zero(eltype(target_buffer)))
    hrange = hessian_range(derivatives_switch)
    isempty(hrange) || size(output, 1) >= 13 ||
        throw(ArgumentError("hessian output requested but the CUDA radix output " *
            "carries potential + gradient only; construct RadixFMMCache(...; hessian=true)"))
    grange = gradient_range(derivatives_switch)
    gradient_start = isempty(grange) ? 1 : first(grange)
    gradient_stop = isempty(grange) ? 0 : last(grange)
    hessian_start = isempty(hrange) ? 1 : first(hrange)
    hessian_stop = isempty(hrange) ? 0 : last(hrange)
    threads = 128
    blocks = cld(n_bodies, threads)
    blocks == 0 && return target_buffer
    CUDA.@cuda threads=threads blocks=blocks _cuda_scatter_output_to_target_buffer_kernel!(
        target_buffer, output, body_perm, body_system_ids, body_indices, isys,
        scalar_potential_index(derivatives_switch), gradient_start, gradient_stop,
        hessian_start, hessian_stop, n_bodies,
    )
    return target_buffer
end

# Vortex B2M (task 032): device mirror of `_host_b2m_vortex_kernel!`, sharing
# the `_resident_vortex_{phi,chi}_contrib` per-(n, m) math. Block per leaf
# cell with body-parallel threads and a shared-memory reduction, matching the
# task 035 cycle-2 scalar form (see the note above `CUDA_B2M_BLOCK`).
function _cuda_b2m_vortex_leaf_nodes_kernel!(phi, chi, source_bodies, cell_centers,
        cell_ranges, leaf_to_node, P_phi, P_chi, ncell)
    i_cell = blockIdx().x
    i_cell > ncell && return nothing
    tid = threadIdx().x
    nt = blockDim().x
    TF = eltype(phi)
    shre = CUDA.CuStaticSharedArray(TF, CUDA_B2M_BLOCK)
    shim = CUDA.CuStaticSharedArray(TF, CUDA_B2M_BLOCK)
    first = cell_ranges[1, i_cell]
    count = cell_ranges[2, i_cell]
    cx = cell_centers[1, i_cell]
    cy = cell_centers[2, i_cell]
    cz = cell_centers[3, i_cell]
    node = leaf_to_node[i_cell]
    @inbounds for n in 0:P_phi
        for m in 0:n
            acc_re = zero(TF)
            acc_im = zero(TF)
            k = first + tid - 1
            while k <= first + count - 1
                mdx = cx - source_bodies[1, k]
                mdy = cy - source_bodies[2, k]
                mdz = cz - source_bodies[3, k]
                vx = source_bodies[5, k]
                vy = source_bodies[6, k]
                vz = source_bodies[7, k]
                re, im = _resident_vortex_phi_contrib(mdx, mdy, mdz, vx, vy, vz, n, m)
                acc_re += re
                acc_im += im
                k += nt
            end
            re, im = _cuda_b2m_block_reduce(shre, shim, tid, acc_re, acc_im)
            if tid == 1
                row = _cuda_flat_basis_index(n, m, 1)
                phi[row, node] = re
                phi[row + 1, node] = im
            end
        end
    end
    @inbounds for n in 1:P_chi
        for m in 0:n
            acc_re = zero(TF)
            acc_im = zero(TF)
            k = first + tid - 1
            while k <= first + count - 1
                mdx = cx - source_bodies[1, k]
                mdy = cy - source_bodies[2, k]
                mdz = cz - source_bodies[3, k]
                vx = source_bodies[5, k]
                vy = source_bodies[6, k]
                vz = source_bodies[7, k]
                re, im = _resident_vortex_chi_contrib(mdx, mdy, mdz, vx, vy, vz, n, m)
                acc_re += re
                acc_im += im
                k += nt
            end
            re, im = _cuda_b2m_block_reduce(shre, shim, tid, acc_re, acc_im)
            if tid == 1
                row = _cuda_flat_basis_index(n, m, 1)
                chi[row, node] = re
                chi[row + 1, node] = im
            end
        end
    end
    return nothing
end

_launch_cuda_b2m!(state::DeviceResidentRadixState) =
    _launch_cuda_b2m!(state, state.options.body_type)

function _launch_cuda_b2m!(state::DeviceResidentRadixState{TF},
        ::Type{<:Point{Source}}) where TF
    fill!(state.multipoles.phi, zero(TF))
    fill!(state.multipoles.chi, zero(TF))
    ncell = state.counts.n_cells
    ncell == 0 && return state
    if state.grid isa DeviceRadixGrid
        # task 035 cycle 2: block per cell, body-parallel reduction
        CUDA.@cuda threads=CUDA_B2M_BLOCK blocks=ncell _cuda_b2m_leaf_nodes_kernel!(
            state.multipoles.phi, state.source_bodies, state.cell_centers,
            state.cell_ranges, state.grid.leaf_to_node,
            state.invariant_cache.basis_info.orders.P_phi, ncell,
        )
    else
        threads = 128
        CUDA.@cuda threads=threads blocks=cld(ncell, threads) _cuda_b2m_kernel!(
            state.multipoles.phi, state.source_bodies, state.cell_centers,
            state.cell_ranges, state.invariant_cache.basis_info.orders.P_phi, ncell,
        )
    end
    return state
end

function _launch_cuda_b2m!(state::DeviceResidentRadixState{TF,B,LH},
        ::Type{<:Point{Vortex}}) where {TF,B,LH}
    LH || throw(ArgumentError(
        "Point{Vortex} sources require the Lamb-Helmholtz channel; construct the " *
        "cache with lamb_helmholtz=true"))
    state.grid isa DeviceRadixGrid || throw(ArgumentError(
        "the CUDA vortex B2M is supported on the recurring RadixFMMCache " *
        "(DeviceRadixGrid) lifecycle only"))
    fill!(state.multipoles.phi, zero(TF))
    fill!(state.multipoles.chi, zero(TF))
    ncell = state.counts.n_cells
    ncell == 0 && return state
    orders = state.invariant_cache.basis_info.orders
    # task 035 cycle 2: block per cell, body-parallel reduction
    CUDA.@cuda threads=CUDA_B2M_BLOCK blocks=ncell _cuda_b2m_vortex_leaf_nodes_kernel!(
        state.multipoles.phi, state.multipoles.chi, state.source_bodies,
        state.cell_centers, state.cell_ranges, state.grid.leaf_to_node,
        orders.P_phi, orders.P_active, ncell,
    )
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
        direct_targets, direct_sources, npairs, ::Val{HS}=Val(false)) where HS
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
            hxx = zero(T); hxy = zero(T); hxz = zero(T)
            hyy = zero(T); hyz = zero(T); hzz = zero(T)
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
                    invr2 = invr * invr
                    invr3 = invr * invr2
                    gx -= q * dx * invr3
                    gy -= q * dy * invr3
                    gz -= q * dz * invr3
                    if HS
                        # H = qc·(3ΔxΔxᵀ/r⁵ - I/r³), symmetric
                        q3invr5 = 3 * q * invr3 * invr2
                        qinvr3 = q * invr3
                        hxx += q3invr5 * dx * dx - qinvr3
                        hxy += q3invr5 * dx * dy
                        hxz += q3invr5 * dx * dz
                        hyy += q3invr5 * dy * dy - qinvr3
                        hyz += q3invr5 * dy * dz
                        hzz += q3invr5 * dz * dz - qinvr3
                    end
                end
            end
            CUDA.@atomic output[1, i] += u
            CUDA.@atomic output[2, i] += gx
            CUDA.@atomic output[3, i] += gy
            CUDA.@atomic output[4, i] += gz
            if HS
                CUDA.@atomic output[5, i] += hxx
                CUDA.@atomic output[6, i] += hxy
                CUDA.@atomic output[7, i] += hxz
                CUDA.@atomic output[8, i] += hxy
                CUDA.@atomic output[9, i] += hyy
                CUDA.@atomic output[10, i] += hyz
                CUDA.@atomic output[11, i] += hxz
                CUDA.@atomic output[12, i] += hyz
                CUDA.@atomic output[13, i] += hzz
            end
            i += 32
        end
        pair_i += warp_stride
    end
    return nothing
end

# Singular Biot-Savart pairs kernel for Point{Vortex} sources (task 032 stage 1):
# device mirror of `_host_direct_pairs_vortex_kernel!`. No symmetric variant —
# the symmetric kernel's shared-work trick assumes the scalar kernel.
function _cuda_direct_pairs_vortex_kernel!(output, source_bodies, cell_ranges,
        direct_targets, direct_sources, npairs, ::Val{HS}) where HS
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
            ux = zero(T); uy = zero(T); uz = zero(T)
            j11 = zero(T); j12 = zero(T); j13 = zero(T)
            j21 = zero(T); j22 = zero(T); j23 = zero(T)
            j31 = zero(T); j32 = zero(T); j33 = zero(T)
            for j in sfirst:slast
                i == j && continue
                dx = xi - source_bodies[1, j]
                dy = yi - source_bodies[2, j]
                dz = zi - source_bodies[3, j]
                r2 = dx * dx + dy * dy + dz * dz
                if r2 > zero(r2)
                    gx = source_bodies[5, j]
                    gy = source_bodies[6, j]
                    gz = source_bodies[7, j]
                    invr = _cuda_fast_rsqrt(r2)
                    invr2 = invr * invr
                    denom = c * invr * invr2
                    ux += (dz * gy - dy * gz) * denom
                    uy += (dx * gz - dz * gx) * denom
                    uz += (dy * gx - dx * gy) * denom
                    if HS
                        denom *= invr2
                        j11 += -3 * dx * (gy * dz - gz * dy) * denom
                        j12 += (-3 * dx * (gz * dx - gx * dz) + gz * r2) * denom
                        j13 += (-3 * dx * (gx * dy - gy * dx) - gy * r2) * denom
                        j21 += (-3 * dy * (gy * dz - gz * dy) - gz * r2) * denom
                        j22 += -3 * dy * (gz * dx - gx * dz) * denom
                        j23 += (-3 * dy * (gx * dy - gy * dx) + gx * r2) * denom
                        j31 += (-3 * dz * (gy * dz - gz * dy) + gy * r2) * denom
                        j32 += (-3 * dz * (gz * dx - gx * dz) - gx * r2) * denom
                        j33 += -3 * dz * (gx * dy - gy * dx) * denom
                    end
                end
            end
            CUDA.@atomic output[2, i] += ux
            CUDA.@atomic output[3, i] += uy
            CUDA.@atomic output[4, i] += uz
            if HS
                CUDA.@atomic output[5, i] += j11
                CUDA.@atomic output[6, i] += j12
                CUDA.@atomic output[7, i] += j13
                CUDA.@atomic output[8, i] += j21
                CUDA.@atomic output[9, i] += j22
                CUDA.@atomic output[10, i] += j23
                CUDA.@atomic output[11, i] += j31
                CUDA.@atomic output[12, i] += j32
                CUDA.@atomic output[13, i] += j33
            end
            i += 32
        end
        pair_i += warp_stride
    end
    return nothing
end

# Task 037f :lut mode device machinery: cooperative block-start load of the
# construction-built (2, _NF_GH_LUT_N) Float32 G/H table into shared memory
# (8 KB/block), then per-pair linear interpolation in x = rho^2 via the shared
# `_gh_from_lut` math (translate_batched_resident.jl).  The load is uniform
# across the block (sync_threads before any pair work).
@inline function _nf_load_gh_lut!(gh_lut)
    shlut = CUDA.CuStaticSharedArray(Float32, (2, _NF_GH_LUT_N))
    ii = threadIdx().x
    while ii <= Int32(_NF_GH_LUT_N)
        @inbounds shlut[1, ii] = gh_lut[1, ii]
        @inbounds shlut[2, ii] = gh_lut[2, ii]
        ii += blockDim().x
    end
    CUDA.sync_threads()
    return shlut
end

# LUT-mode pair math for the regularized family: identical branch structure to
# `_direct_pair_ug(h)` (sigma <= 0 -> singular; split kernels switch at the
# pass-1 cutoff; x >= rho_t^2 -> singular, the table's own domain end).
@inline _lut_pair_cutoff(kernel::AbstractRegularizedVortex) = kernel.rho_t
@inline _lut_pair_cutoff(kernel::TwoPassVortex) = kernel.rho_c

@inline function _lut_pair_gh(kernel::AbstractRegularizedVortex, shlut,
        r2::T, invr::T, sigma::T) where T
    g = one(T)
    h = -T(3)
    if sigma > zero(T)
        rho = r2 * invr / sigma
        if rho <= T(_lut_pair_cutoff(kernel))
            g, h = _gh_from_lut(shlut, rho, T(kernel.rho_t)^2)
        end
    end
    return g, h
end

@inline function _lut_pair_ug(kernel::AbstractRegularizedVortex, shlut,
        dx, dy, dz, r2, invr, source_bodies, j)
    @inbounds gsx = source_bodies[5, j]
    @inbounds gsy = source_bodies[6, j]
    @inbounds gsz = source_bodies[7, j]
    @inbounds sigma = source_bodies[kernel.sigma_row, j]
    g, _ = _lut_pair_gh(kernel, shlut, r2, invr, sigma)
    return _vortex_pair_ug(dx, dy, dz, invr, gsx, gsy, gsz, g)
end

@inline function _lut_pair_ugh(kernel::AbstractRegularizedVortex, shlut,
        dx, dy, dz, r2, invr, source_bodies, j)
    @inbounds gsx = source_bodies[5, j]
    @inbounds gsy = source_bodies[6, j]
    @inbounds gsz = source_bodies[7, j]
    @inbounds sigma = source_bodies[kernel.sigma_row, j]
    g, h = _lut_pair_gh(kernel, shlut, r2, invr, sigma)
    return _vortex_pair_ugh(dx, dy, dz, r2, invr, gsx, gsy, gsz, g, h)
end

# compile-time predicate: the LUT path engages only when the mode is :lut, a
# table was passed, and the functor is a regularized-family kernel
@inline _nf_lut_active(::Val{GH}, gh_lut, kernel) where GH =
    GH === :lut && gh_lut !== nothing && kernel isa AbstractRegularizedVortex

# Generic functor pair kernel (task 032 stage 2): identical warp-per-pair
# structure to `_cuda_direct_pairs_output_kernel!`, but the per-pair math comes
# from the `direct_kernel` functor stamped into the options at construction —
# compile-time specialization, one kernel instantiation per functor type, no
# runtime branch in the pair loop. The hard-coded kernels above remain as the
# functor-abstraction benchmark reference (031 sign-off (b)).
# Task 037f: `ghv` threads the cheapened g/h mode into the regularized-family
# pair math (:shipped routes to the bitwise-identical 8-arg methods); `gh_lut`
# carries the :lut device table (or `nothing`).
function _cuda_direct_pairs_functor_kernel!(kernel, output, source_bodies,
        cell_ranges, direct_targets, direct_sources, npairs, ::Val{HS},
        ghv::Val=Val(:shipped), gh_lut=nothing) where HS
    T = eltype(output)
    ep = _emits_potential(kernel)
    shlut = _nf_lut_active(ghv, gh_lut, kernel) ? _nf_load_gh_lut!(gh_lut) : nothing
    lane = (threadIdx().x - Int32(1)) % Int32(32)
    warps_per_block = blockDim().x ÷ Int32(32)
    warp_in_block = (threadIdx().x - Int32(1)) ÷ Int32(32)
    pair_i = (blockIdx().x - 1) * warps_per_block + warp_in_block + 1
    warp_stride = gridDim().x * warps_per_block
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
            gx = zero(T); gy = zero(T); gz = zero(T)
            h1 = zero(T); h2 = zero(T); h3 = zero(T)
            h4 = zero(T); h5 = zero(T); h6 = zero(T)
            h7 = zero(T); h8 = zero(T); h9 = zero(T)
            for j in sfirst:slast
                i == j && continue
                dx = xi - source_bodies[1, j]
                dy = yi - source_bodies[2, j]
                dz = zi - source_bodies[3, j]
                r2 = dx * dx + dy * dy + dz * dz
                if r2 > zero(r2)
                    invr = _cuda_fast_rsqrt(r2)
                    if HS
                        du, dgx, dgy, dgz, dh1, dh2, dh3, dh4, dh5, dh6, dh7, dh8, dh9 =
                            shlut === nothing ?
                            _direct_pair_ugh(kernel, dx, dy, dz, r2, invr,
                                source_bodies, j, ghv) :
                            _lut_pair_ugh(kernel, shlut, dx, dy, dz, r2, invr,
                                source_bodies, j)
                        u += du
                        gx += dgx; gy += dgy; gz += dgz
                        h1 += dh1; h2 += dh2; h3 += dh3
                        h4 += dh4; h5 += dh5; h6 += dh6
                        h7 += dh7; h8 += dh8; h9 += dh9
                    else
                        du, dgx, dgy, dgz = shlut === nothing ?
                            _direct_pair_ug(kernel, dx, dy, dz,
                                r2, invr, source_bodies, j, ghv) :
                            _lut_pair_ug(kernel, shlut, dx, dy, dz, r2, invr,
                                source_bodies, j)
                        u += du
                        gx += dgx; gy += dgy; gz += dgz
                    end
                end
            end
            ep && (CUDA.@atomic output[1, i] += u)
            CUDA.@atomic output[2, i] += gx
            CUDA.@atomic output[3, i] += gy
            CUDA.@atomic output[4, i] += gz
            if HS
                CUDA.@atomic output[5, i] += h1
                CUDA.@atomic output[6, i] += h2
                CUDA.@atomic output[7, i] += h3
                CUDA.@atomic output[8, i] += h4
                CUDA.@atomic output[9, i] += h5
                CUDA.@atomic output[10, i] += h6
                CUDA.@atomic output[11, i] += h7
                CUDA.@atomic output[12, i] += h8
                CUDA.@atomic output[13, i] += h9
            end
            i += 32
        end
        pair_i += warp_stride
    end
    return nothing
end

#------- 041e: target-owned fused nearfield kernels -------#
#
# Both shapes consume the target-major U CSR built on occupancy epochs by
# `_cuda_adaptive_build_u_csr!` (tree_batched_cuda.jl): `u_csr_offsets[l]` is
# the first CSR edge of target leaf slot `l`, `u_csr_sources[e]` the source
# leaf slot of edge `e`.  Grid-strides over target leaf slots; every leaf has
# exactly one CTA owner, and each target body is retired once (atomic adds —
# see the CUDA_NEARFIELD_SHAPE comment for why plain stores are not safe).
# Pair math, predicate, and g/h modes are the shipped `_direct_pair_ug/_ugh`
# functor path unchanged.

function _cuda_direct_pairs_fused_cta_kernel!(kernel, output, source_bodies,
        cell_ranges, u_csr_offsets, u_csr_sources, n_leaves, ::Val{HS},
        ghv::Val=Val(:shipped)) where HS
    T = eltype(output)
    ep = _emits_potential(kernel)
    leaf = Int(blockIdx().x)
    @inbounds while leaf <= n_leaves
        tfirst = cell_ranges[1, leaf]
        tlast = tfirst + cell_ranges[2, leaf] - 1
        estart = Int(u_csr_offsets[leaf])
        eend = Int(u_csr_offsets[leaf + 1]) - 1
        i = tfirst + Int(threadIdx().x) - 1
        while i <= tlast && eend >= estart
            xi = source_bodies[1, i]
            yi = source_bodies[2, i]
            zi = source_bodies[3, i]
            u = zero(T)
            gx = zero(T); gy = zero(T); gz = zero(T)
            h1 = zero(T); h2 = zero(T); h3 = zero(T)
            h4 = zero(T); h5 = zero(T); h6 = zero(T)
            h7 = zero(T); h8 = zero(T); h9 = zero(T)
            for e in estart:eend
                sc = Int(u_csr_sources[e])
                sfirst = cell_ranges[1, sc]
                slast = sfirst + cell_ranges[2, sc] - 1
                for j in sfirst:slast
                    i == j && continue
                    dx = xi - source_bodies[1, j]
                    dy = yi - source_bodies[2, j]
                    dz = zi - source_bodies[3, j]
                    r2 = dx * dx + dy * dy + dz * dz
                    if r2 > zero(r2)
                        invr = _cuda_fast_rsqrt(r2)
                        if HS
                            du, dgx, dgy, dgz, dh1, dh2, dh3, dh4, dh5, dh6, dh7, dh8, dh9 =
                                _direct_pair_ugh(kernel, dx, dy, dz, r2, invr,
                                    source_bodies, j, ghv)
                            u += du
                            gx += dgx; gy += dgy; gz += dgz
                            h1 += dh1; h2 += dh2; h3 += dh3
                            h4 += dh4; h5 += dh5; h6 += dh6
                            h7 += dh7; h8 += dh8; h9 += dh9
                        else
                            du, dgx, dgy, dgz = _direct_pair_ug(kernel, dx, dy,
                                dz, r2, invr, source_bodies, j, ghv)
                            u += du
                            gx += dgx; gy += dgy; gz += dgz
                        end
                    end
                end
            end
            ep && (CUDA.@atomic output[1, i] += u)
            CUDA.@atomic output[2, i] += gx
            CUDA.@atomic output[3, i] += gy
            CUDA.@atomic output[4, i] += gz
            if HS
                CUDA.@atomic output[5, i] += h1
                CUDA.@atomic output[6, i] += h2
                CUDA.@atomic output[7, i] += h3
                CUDA.@atomic output[8, i] += h4
                CUDA.@atomic output[9, i] += h5
                CUDA.@atomic output[10, i] += h6
                CUDA.@atomic output[11, i] += h7
                CUDA.@atomic output[12, i] += h8
                CUDA.@atomic output[13, i] += h9
            end
            i += Int(blockDim().x)
        end
        leaf += Int(gridDim().x)
    end
    return nothing
end

function _cuda_direct_pairs_fused_packed_kernel!(kernel, output, source_bodies,
        cell_ranges, u_csr_offsets, u_csr_sources, body_leaf, n_bodies, ::Val{HS},
        ghv::Val=Val(:shipped)) where HS
    T = eltype(output)
    ep = _emits_potential(kernel)
    i = Int((blockIdx().x - 1) * blockDim().x + threadIdx().x)
    stride = Int(gridDim().x) * Int(blockDim().x)
    @inbounds while i <= n_bodies
        leaf = Int(body_leaf[i])
        if leaf > 0
            estart = Int(u_csr_offsets[leaf])
            eend = Int(u_csr_offsets[leaf + 1]) - 1
            if eend >= estart
                xi = source_bodies[1, i]
                yi = source_bodies[2, i]
                zi = source_bodies[3, i]
                u = zero(T)
                gx = zero(T); gy = zero(T); gz = zero(T)
                h1 = zero(T); h2 = zero(T); h3 = zero(T)
                h4 = zero(T); h5 = zero(T); h6 = zero(T)
                h7 = zero(T); h8 = zero(T); h9 = zero(T)
                for e in estart:eend
                    sc = Int(u_csr_sources[e])
                    sfirst = cell_ranges[1, sc]
                    slast = sfirst + cell_ranges[2, sc] - 1
                    for j in sfirst:slast
                        i == j && continue
                        dx = xi - source_bodies[1, j]
                        dy = yi - source_bodies[2, j]
                        dz = zi - source_bodies[3, j]
                        r2 = dx * dx + dy * dy + dz * dz
                        if r2 > zero(r2)
                            invr = _cuda_fast_rsqrt(r2)
                            if HS
                                du, dgx, dgy, dgz, dh1, dh2, dh3, dh4, dh5, dh6, dh7, dh8, dh9 =
                                    _direct_pair_ugh(kernel, dx, dy, dz, r2,
                                        invr, source_bodies, j, ghv)
                                u += du
                                gx += dgx; gy += dgy; gz += dgz
                                h1 += dh1; h2 += dh2; h3 += dh3
                                h4 += dh4; h5 += dh5; h6 += dh6
                                h7 += dh7; h8 += dh8; h9 += dh9
                            else
                                du, dgx, dgy, dgz = _direct_pair_ug(kernel, dx,
                                    dy, dz, r2, invr, source_bodies, j, ghv)
                                u += du
                                gx += dgx; gy += dgy; gz += dgz
                            end
                        end
                    end
                end
                ep && (CUDA.@atomic output[1, i] += u)
                CUDA.@atomic output[2, i] += gx
                CUDA.@atomic output[3, i] += gy
                CUDA.@atomic output[4, i] += gz
                if HS
                    CUDA.@atomic output[5, i] += h1
                    CUDA.@atomic output[6, i] += h2
                    CUDA.@atomic output[7, i] += h3
                    CUDA.@atomic output[8, i] += h4
                    CUDA.@atomic output[9, i] += h5
                    CUDA.@atomic output[10, i] += h6
                    CUDA.@atomic output[11, i] += h7
                    CUDA.@atomic output[12, i] += h8
                    CUDA.@atomic output[13, i] += h9
                end
            end
        end
        i += stride
    end
    return nothing
end

@inline function _nf_warp_reduce(v)
    v += CUDA.shfl_down_sync(CUDA.FULL_MASK, v, 16)
    v += CUDA.shfl_down_sync(CUDA.FULL_MASK, v, 8)
    v += CUDA.shfl_down_sync(CUDA.FULL_MASK, v, 4)
    v += CUDA.shfl_down_sync(CUDA.FULL_MASK, v, 2)
    v += CUDA.shfl_down_sync(CUDA.FULL_MASK, v, 1)
    return v
end

function _cuda_direct_pairs_fused_srclanes_kernel!(kernel, output, source_bodies,
        cell_ranges, u_csr_offsets, u_csr_sources, n_leaves, ::Val{HS},
        ghv::Val=Val(:shipped)) where HS
    T = eltype(output)
    ep = _emits_potential(kernel)
    lane = (threadIdx().x - Int32(1)) % Int32(32)
    warps_per_block = blockDim().x ÷ Int32(32)
    warp_in_block = (threadIdx().x - Int32(1)) ÷ Int32(32)
    leaf = Int(blockIdx().x)
    @inbounds while leaf <= n_leaves
        tfirst = cell_ranges[1, leaf]
        tlast = tfirst + cell_ranges[2, leaf] - 1
        estart = Int(u_csr_offsets[leaf])
        eend = Int(u_csr_offsets[leaf + 1]) - 1
        i = tfirst + Int(warp_in_block)
        while i <= tlast && eend >= estart
            xi = source_bodies[1, i]
            yi = source_bodies[2, i]
            zi = source_bodies[3, i]
            u = zero(T)
            gx = zero(T); gy = zero(T); gz = zero(T)
            h1 = zero(T); h2 = zero(T); h3 = zero(T)
            h4 = zero(T); h5 = zero(T); h6 = zero(T)
            h7 = zero(T); h8 = zero(T); h9 = zero(T)
            for e in estart:eend
                sc = Int(u_csr_sources[e])
                sfirst = cell_ranges[1, sc]
                slast = sfirst + cell_ranges[2, sc] - 1
                j = sfirst + Int(lane)
                while j <= slast
                    if i != j
                        dx = xi - source_bodies[1, j]
                        dy = yi - source_bodies[2, j]
                        dz = zi - source_bodies[3, j]
                        r2 = dx * dx + dy * dy + dz * dz
                        if r2 > zero(r2)
                            invr = _cuda_fast_rsqrt(r2)
                            if HS
                                du, dgx, dgy, dgz, dh1, dh2, dh3, dh4, dh5, dh6, dh7, dh8, dh9 =
                                    _direct_pair_ugh(kernel, dx, dy, dz, r2,
                                        invr, source_bodies, j, ghv)
                                u += du
                                gx += dgx; gy += dgy; gz += dgz
                                h1 += dh1; h2 += dh2; h3 += dh3
                                h4 += dh4; h5 += dh5; h6 += dh6
                                h7 += dh7; h8 += dh8; h9 += dh9
                            else
                                du, dgx, dgy, dgz = _direct_pair_ug(kernel, dx,
                                    dy, dz, r2, invr, source_bodies, j, ghv)
                                u += du
                                gx += dgx; gy += dgy; gz += dgz
                            end
                        end
                    end
                    j += 32
                end
            end
            u = _nf_warp_reduce(u)
            gx = _nf_warp_reduce(gx); gy = _nf_warp_reduce(gy); gz = _nf_warp_reduce(gz)
            if HS
                h1 = _nf_warp_reduce(h1); h2 = _nf_warp_reduce(h2); h3 = _nf_warp_reduce(h3)
                h4 = _nf_warp_reduce(h4); h5 = _nf_warp_reduce(h5); h6 = _nf_warp_reduce(h6)
                h7 = _nf_warp_reduce(h7); h8 = _nf_warp_reduce(h8); h9 = _nf_warp_reduce(h9)
            end
            if lane == Int32(0)
                ep && (CUDA.@atomic output[1, i] += u)
                CUDA.@atomic output[2, i] += gx
                CUDA.@atomic output[3, i] += gy
                CUDA.@atomic output[4, i] += gz
                if HS
                    CUDA.@atomic output[5, i] += h1
                    CUDA.@atomic output[6, i] += h2
                    CUDA.@atomic output[7, i] += h3
                    CUDA.@atomic output[8, i] += h4
                    CUDA.@atomic output[9, i] += h5
                    CUDA.@atomic output[10, i] += h6
                    CUDA.@atomic output[11, i] += h7
                    CUDA.@atomic output[12, i] += h8
                    CUDA.@atomic output[13, i] += h9
                end
            end
            i += Int(warps_per_block)
        end
        leaf += Int(gridDim().x)
    end
    return nothing
end

#------- distance-binned nearfield pair stream (task 032a stage C, 031a §6.3) -------#
#
# An unbinned split kernel pays both branch paths on essentially every warp at
# the shipped operating points (`031a` §6.3: modeled 1.56x SLOWER than the
# regularized-everywhere baseline at ell >= 4), so the split vortex kernels
# (`PartitionedVortex`, `TwoPassVortex` pass 1) get a measured menu of stream
# mechanisms selected by `radix_setting(:CUDA_NEARFIELD_BINNING)`:
#
#   :unbinned          — the plain predicated functor kernel (the §6.3 negative
#                        control; also the fallback on flat-policy caches, which
#                        carry no `CUDANearfieldBinContext`);
#   :classsplit        — mechanism (c): a per-step three-way compaction of the
#                        direct pair list into pure-singular / pure-regularized
#                        / mixed buckets by the shared `_nearfield_pair_bucket`
#                        cell-AABB rule; pure buckets run branch-free kernels,
#                        the mixed bucket keeps the predicated functor kernel;
#   :ballot            — mechanism (b): every pair through the warp-ballot
#                        queue kernel — per-(warp, source) predicate votes
#                        evaluate branch-homogeneous instants inline and defer
#                        mixed instants into per-lane shared-memory index
#                        queues drained side-at-a-time (compacted-index
#                        streaming with zero global scratch: a global body-pair
#                        bitmask cannot be construction-bounded under the
#                        RadixFMMCache capacity contract, because a single fat
#                        cell overflows any mask sized short of n²);
#   :classsplit_ballot — (c) for the pure buckets plus (b) for the mixed one.
#
# Mechanism (a) — within-cell sub-Morton body ordering — is orthogonal and
# toggled by `radix_setting(:CUDA_NEARFIELD_SUBSORT)`: the update composes a per-cell
# sub-key sort into `grid.perm` before packing, so warp lanes (consecutive
# target bodies) span a compact spatial sub-block and the ρ predicate becomes
# lane-coherent when cells are much larger than a warp.
#
# All recurring work is device kernels on the launch stream (capture-safe, zero
# per-step allocation, no transfers), preserving the 023 counter contract. The
# scratch lives in the construction-built `CUDANearfieldBinContext` on the
# hierarchical device context.
# NOTE: CUDA_NEARFIELD_BINNING, CUDA_TWOPASS_PASS2_QUEUED, and the task-037f
# CUDA_NEARFIELD_GH_MODE (translate_batched_resident.jl) are read inside
# the lifecycle body, so — like every runtime flag there — the selection is
# baked into a captured CUDA graph at record time: flip them only before cache
# construction (or force a new occupancy epoch/cache), or the replayed graph
# keeps the old mechanism silently. CUDA_NEARFIELD_SUBSORT is read in the
# (uncaptured) host refresh and may be flipped per step.
# Defaults set by the Stage C H200 measurement (job 13064834, three adequate
# overlap-2 cube points, both precisions): :classsplit was fastest everywhere
# (nearfield stage 1.14-1.34x over the regularized baseline, vs 1.06-1.22x
# unbinned), sub-Morton ordering added a further 2-4% and raised warp
# homogeneity (e.g. 0.826 -> 0.887 at n=1e5/ell=3/q=16); the ballot queue was
# a measured loss at every point (votes + drains cost more than per-instant
# hardware predication at the achievable regularized fractions) and is
# retained as a selectable mechanism, not a default. These Refs affect only
# the split vortex kernels — the shipped nearfield default is unchanged.
const CUDA_NEARFIELD_BINNING = Ref{Symbol}(:classsplit)

# Task 041e: target-owned fused nearfield kernel shape (adaptive path only).
#   :pairs           — the shipped warp-per-U-edge organization (default);
#   :fused_cta       — shape 1: CTA per target leaf, thread-per-target lanes,
#                      one fused traversal of the leaf's complete U-source
#                      adjacency, single accumulator retirement per target;
#   :fused_srclanes  — shape 4: CTA per target leaf, warp per target body,
#                      lanes stride the concatenated CSR sources, 13-component
#                      warp-shuffle reduction, single retirement per target;
#   :fused_packed    — shape 2 (repacking): thread per target body over the
#                      dense leaf-major body order (zero lane underfill by
#                      construction; divergence only where a warp spans a
#                      leaf boundary), fused CSR traversal of the body's own
#                      leaf adjacency, single retirement per target.
# Selection is read at construction (CSR buffers are sized only when a fused
# shape is armed) and inside the lifecycle body (graph-baked at record time,
# exactly like CUDA_NEARFIELD_BINNING above): flip only before cache
# construction. Unsupported configurations (uniform hierarchical/flat caches,
# unarmed CSR buffers) fall back to the shipped shapes automatically — they
# never throw during a resident step. Final writes stay atomic adds because
# the adaptive M2T kernel accumulates into `output` concurrently on the far
# stream; the fused win retained is one retirement per target instead of one
# per (edge, target). `:lut` g/h mode is not supported by the fused shapes.
const CUDA_NEARFIELD_SHAPE = Ref{Symbol}(:pairs)
const NEARFIELD_SHAPES = (:pairs, :fused_cta, :fused_srclanes, :fused_packed)
# Regime selector (041e Stage C promotion gate 6): even when a fused shape is
# selected, engage it only at or above this body count — the measured H200
# win envelope is n=1e6-scale (rotor/cube/wake) with regressions at n=1e5,
# and the crossover measurement sets this default.  Read at launch (graph-
# baked per epoch like the shape Ref); below the threshold the shipped
# organization runs (automatic fallback, never a throw).
const CUDA_NEARFIELD_FUSED_MIN_BODIES = Ref{Int}(400_000)
const CUDA_NEARFIELD_SUBSORT = Ref(true)
# TwoPassVortex pass-2 deficit sweep kernel mode: shell-queue (ballot-compacted)
# versus plain predicated evaluation of the (rho_c, rho_t] shell.
const CUDA_TWOPASS_PASS2_QUEUED = Ref(false)
# Exact target-point/source-cell AABB pruning for the pass-2 correction. The
# final pair predicate remains authoritative; this only avoids body scans for
# source cells that cannot intersect a lane's physical correction shell.
const CUDA_TWOPASS_TARGET_AABB_PRUNE = Ref(false)
# Task 037e: exact target-point/source-cell AABB fast path in the MIXED-bucket
# direct traversal (bucket kernel and ballot-queue kernel; the pure buckets are
# untouched). Per (warp, 32-target lane block) the lanes vote on
# `_nearfield_point_aabb_reach` against rho_cut·σ_max(source cell); when no
# lane can reach the regularized zone the inner source loop runs the exact
# singular pair math directly — FP-identical in kind and order to the outcome
# the per-pair split branch would have produced — skipping the σ loads, ρ
# predicate, and expensive-branch candidacy entirely. No pair is ever dropped:
# the fast path only changes HOW the (provably all-singular) lane block is
# evaluated, so output is bitwise the flag-off result. Like every Ref read in
# the lifecycle body, the selection is BAKED into a captured CUDA graph at
# record time: flip it only before cache construction (or force a new
# occupancy epoch/cache). Off by default — the shipped classsplit stream is
# the control until the 037e H200 measurement.
const CUDA_NEARFIELD_PAIR_AABB = Ref(false)
# per-lane shared-memory queue depth of the ballot kernels
const _NF_QUEUE_CAP = 8

@inline _nearfield_bin_ctx(state::DeviceResidentRadixState) =
    state.interaction_list isa DeviceHierarchicalM2LContext ?
        state.interaction_list.nearfield : nothing

# Morton encode/decode without host-side argument checks (device-safe).
@inline function _cuda_morton_key(x::Int, y::Int, z::Int, ell::Int)
    key = UInt64(0)
    bit = 0
    while bit < ell
        key |= (UInt64((x >> bit) & 0x1) << (3 * bit))
        key |= (UInt64((y >> bit) & 0x1) << (3 * bit + 1))
        key |= (UInt64((z >> bit) & 0x1) << (3 * bit + 2))
        bit += 1
    end
    return key
end

# Occupied-cell lookup by Morton key over the sorted prefix cell_keys[1:n_cells];
# 0 when unoccupied (device mirror of the host `_twopass_cell_lookup`).
@inline function _cuda_cell_key_search(cell_keys, n_cells::Int, key::UInt64)
    lo = 1
    hi = n_cells
    @inbounds while lo <= hi
        mid = (lo + hi) >>> 1
        k = cell_keys[mid]
        if k < key
            lo = mid + 1
        elseif k > key
            hi = mid - 1
        else
            return mid
        end
    end
    return 0
end

# Per-cell σ extrema over the packed source rows (thread per cell; cells are a
# few tens to hundreds of bodies, so a serial scan per cell is cheap).
function _cuda_cell_sigma_kernel!(cell_sigma_max, cell_sigma_min, source_bodies,
        cell_ranges, sigma_row, n_cells)
    T = eltype(cell_sigma_max)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > n_cells && return nothing
    @inbounds begin
        first = cell_ranges[1, i]
        cnt = cell_ranges[2, i]
        smax = zero(T)
        smin = typemax(T)
        j = first
        while j < first + cnt
            s = source_bodies[sigma_row, j]
            smax = max(smax, s)
            smin = min(smin, s)
            j += 1
        end
        cnt == 0 && (smin = zero(T))
        cell_sigma_max[i] = smax
        cell_sigma_min[i] = smin
    end
    return nothing
end

# Single-block reduction of the per-cell maxima into the Float64 scalars
# [σ_max, (rho_t·σ_max)²] the pass-2 sweep prunes against. Device-resident so
# the value may change every step inside a captured graph.
function _cuda_nf_scalars_kernel!(nf_scalars, cell_sigma_max, n_cells, rho_t)
    sh = CUDA.CuStaticSharedArray(Float64, 256)
    t = threadIdx().x
    m = 0.0
    i = Int(t)
    @inbounds while i <= n_cells
        m = max(m, Float64(cell_sigma_max[i]))
        i += Int(blockDim().x)
    end
    @inbounds sh[t] = m
    CUDA.sync_threads()
    s = Int32(128)
    while s >= Int32(1)
        if t <= s
            @inbounds sh[t] = max(sh[t], sh[t + s])
        end
        CUDA.sync_threads()
        s >>= Int32(1)
    end
    if t == Int32(1)
        @inbounds begin
            sm = sh[1]
            nf_scalars[1] = sm
            nf_scalars[2] = (rho_t * sm)^2
        end
    end
    return nothing
end

# Three-way bucket compaction of the direct pair list (mechanism c). Bucket
# order within each third is atomic-claimed (nondeterministic), which is
# admissible because the consuming kernels accumulate with atomics already.
function _cuda_nearfield_bin_kernel!(bin_targets, bin_sources, bin_counts,
        direct_targets, direct_sources, npairs, cell_coords, cell_sigma_max,
        cell_sigma_min, h_leaf, rho_cut, capacity)
    T = typeof(h_leaf)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > npairs && return nothing
    @inbounds begin
        t = direct_targets[i]
        s = direct_sources[i]
        ox = cell_coords[1, t] - cell_coords[1, s]
        oy = cell_coords[2, t] - cell_coords[2, s]
        oz = cell_coords[3, t] - cell_coords[3, s]
        b = _nearfield_pair_bucket(ox, oy, oz, h_leaf, rho_cut,
            T(cell_sigma_max[s]), T(cell_sigma_min[s]))
        old = CUDA.@atomic bin_counts[b] += Int32(1)
        pos = (Int(b) - 1) * capacity + Int(old) + 1
        bin_targets[pos] = Int32(t)
        bin_sources[pos] = Int32(s)
    end
    return nothing
end

# Warp-per-pair functor kernel over one compacted bucket: identical structure to
# `_cuda_direct_pairs_functor_kernel!`, but the pair count is read from the
# device bucket counters (it varies per step inside a captured graph, so it can
# never be a baked host launch argument).
function _cuda_direct_pairs_bucket_kernel!(kernel, output, source_bodies,
        cell_ranges, bin_targets, bin_sources, bin_counts, bucket::Int32,
        base::Int, ::Val{HS}, ghv::Val=Val(:shipped), gh_lut=nothing) where HS
    T = eltype(output)
    ep = _emits_potential(kernel)
    shlut = _nf_lut_active(ghv, gh_lut, kernel) ? _nf_load_gh_lut!(gh_lut) : nothing
    npairs = Int(@inbounds bin_counts[bucket])
    lane = (threadIdx().x - Int32(1)) % Int32(32)
    warps_per_block = blockDim().x ÷ Int32(32)
    warp_in_block = (threadIdx().x - Int32(1)) ÷ Int32(32)
    pair_i = (blockIdx().x - 1) * warps_per_block + warp_in_block + 1
    warp_stride = gridDim().x * warps_per_block
    @inbounds while pair_i <= npairs
        target_cell = Int(bin_targets[base + pair_i])
        source_cell = Int(bin_sources[base + pair_i])
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
            gx = zero(T); gy = zero(T); gz = zero(T)
            h1 = zero(T); h2 = zero(T); h3 = zero(T)
            h4 = zero(T); h5 = zero(T); h6 = zero(T)
            h7 = zero(T); h8 = zero(T); h9 = zero(T)
            for j in sfirst:slast
                i == j && continue
                dx = xi - source_bodies[1, j]
                dy = yi - source_bodies[2, j]
                dz = zi - source_bodies[3, j]
                r2 = dx * dx + dy * dy + dz * dz
                if r2 > zero(r2)
                    invr = _cuda_fast_rsqrt(r2)
                    if HS
                        du, dgx, dgy, dgz, dh1, dh2, dh3, dh4, dh5, dh6, dh7, dh8, dh9 =
                            shlut === nothing ?
                            _direct_pair_ugh(kernel, dx, dy, dz, r2, invr,
                                source_bodies, j, ghv) :
                            _lut_pair_ugh(kernel, shlut, dx, dy, dz, r2, invr,
                                source_bodies, j)
                        u += du
                        gx += dgx; gy += dgy; gz += dgz
                        h1 += dh1; h2 += dh2; h3 += dh3
                        h4 += dh4; h5 += dh5; h6 += dh6
                        h7 += dh7; h8 += dh8; h9 += dh9
                    else
                        du, dgx, dgy, dgz = shlut === nothing ?
                            _direct_pair_ug(kernel, dx, dy, dz,
                                r2, invr, source_bodies, j, ghv) :
                            _lut_pair_ug(kernel, shlut, dx, dy, dz, r2, invr,
                                source_bodies, j)
                        u += du
                        gx += dgx; gy += dgy; gz += dgz
                    end
                end
            end
            ep && (CUDA.@atomic output[1, i] += u)
            CUDA.@atomic output[2, i] += gx
            CUDA.@atomic output[3, i] += gy
            CUDA.@atomic output[4, i] += gz
            if HS
                CUDA.@atomic output[5, i] += h1
                CUDA.@atomic output[6, i] += h2
                CUDA.@atomic output[7, i] += h3
                CUDA.@atomic output[8, i] += h4
                CUDA.@atomic output[9, i] += h5
                CUDA.@atomic output[10, i] += h6
                CUDA.@atomic output[11, i] += h7
                CUDA.@atomic output[12, i] += h8
                CUDA.@atomic output[13, i] += h9
            end
            i += 32
        end
        pair_i += warp_stride
    end
    return nothing
end

# Task 037e mixed-bucket pair-AABB kernel (`CUDA_NEARFIELD_PAIR_AABB`): the
# bucket-3 replacement for `_cuda_direct_pairs_bucket_kernel!` when the flag is
# on. Identical warp-per-pair work assignment; the target loop is restructured
# into uniform 32-lane blocks (queue-kernel style `has_i` masking, so the warp
# votes are convergent) and each block first votes on the exact
# point-vs-source-cell-AABB reachability predicate
# (`_nearfield_point_aabb_reach`, per-source-cell σ_max — the AABB is
# reconstructed from cell coords · h_leaf + x_min exactly as the twopass
# deficit kernel does). Blocks where no lane can reach rho_cut·σ_max(src) run
# the exact singular pair math (`SingularVortex` path — FP-identical in kind
# and order to the split branch's own singular outcome, so the per-body sums
# are bitwise the flag-off values); reachable blocks run the ordinary split
# functor. APPLY=false skips the output atomics (telemetry replay); `diag`
# (Nothing in the production APPLY launch) accumulates slot 11 = mixed pairs
# tested and slot 12 = pairs whose every lane block was skippable.
function _cuda_direct_pairs_mixed_aabb_kernel!(kernel, output, source_bodies,
        cell_ranges, bin_targets, bin_sources, bin_counts, bucket::Int32,
        base::Int, cell_coords, cell_sigma_max, h_leaf, x_min,
        ::Val{HS}, ::Val{APPLY}, diag) where {HS,APPLY}
    T = eltype(output)
    npairs = Int(@inbounds bin_counts[bucket])
    rho_cut = T(_pass1_regularized_cutoff(kernel))
    hl = T(h_leaf)
    lane = (threadIdx().x - Int32(1)) % Int32(32)
    warps_per_block = blockDim().x ÷ Int32(32)
    warp_in_block = (threadIdx().x - Int32(1)) ÷ Int32(32)
    pair_i = (blockIdx().x - 1) * warps_per_block + warp_in_block + 1
    warp_stride = gridDim().x * warps_per_block
    d_tested = UInt64(0); d_skipped = UInt64(0)
    @inbounds while pair_i <= npairs
        target_cell = Int(bin_targets[base + pair_i])
        source_cell = Int(bin_sources[base + pair_i])
        smax = T(cell_sigma_max[source_cell])
        slo_x = T(x_min[1]) + T(cell_coords[1, source_cell]) * hl
        slo_y = T(x_min[2]) + T(cell_coords[2, source_cell]) * hl
        slo_z = T(x_min[3]) + T(cell_coords[3, source_cell]) * hl
        tfirst = cell_ranges[1, target_cell]
        tcount = cell_ranges[2, target_cell]
        sfirst = cell_ranges[1, source_cell]
        slast = sfirst + cell_ranges[2, source_cell] - 1
        nblk = (tcount + 31) ÷ 32
        blk = 0
        all_skip = true
        while blk < nblk
            i = tfirst + blk * 32 + Int(lane)
            has_i = i < tfirst + tcount
            xi = zero(T); yi = zero(T); zi = zero(T)
            if has_i
                xi = source_bodies[1, i]
                yi = source_bodies[2, i]
                zi = source_bodies[3, i]
            end
            can_reach = has_i && _nearfield_point_aabb_reach(xi, yi, zi,
                slo_x, slo_y, slo_z, hl, rho_cut, smax)
            any_reach = CUDA.vote_any_sync(0xffffffff, can_reach)
            any_reach && (all_skip = false)
            u = zero(T)
            gx = zero(T); gy = zero(T); gz = zero(T)
            h1 = zero(T); h2 = zero(T); h3 = zero(T)
            h4 = zero(T); h5 = zero(T); h6 = zero(T)
            h7 = zero(T); h8 = zero(T); h9 = zero(T)
            j = sfirst
            while j <= slast
                if has_i && i != j
                    dx = xi - source_bodies[1, j]
                    dy = yi - source_bodies[2, j]
                    dz = zi - source_bodies[3, j]
                    r2 = dx * dx + dy * dy + dz * dz
                    if r2 > zero(r2)
                        invr = _cuda_fast_rsqrt(r2)
                        if HS
                            du, dgx, dgy, dgz, dh1, dh2, dh3, dh4, dh5, dh6,
                                dh7, dh8, dh9 = any_reach ?
                                _direct_pair_ugh(kernel, dx, dy, dz, r2, invr,
                                    source_bodies, j) :
                                _direct_pair_ugh(SingularVortex(), dx, dy, dz,
                                    r2, invr, source_bodies, j)
                            u += du
                            gx += dgx; gy += dgy; gz += dgz
                            h1 += dh1; h2 += dh2; h3 += dh3
                            h4 += dh4; h5 += dh5; h6 += dh6
                            h7 += dh7; h8 += dh8; h9 += dh9
                        else
                            du, dgx, dgy, dgz = any_reach ?
                                _direct_pair_ug(kernel, dx, dy, dz, r2, invr,
                                    source_bodies, j) :
                                _direct_pair_ug(SingularVortex(), dx, dy, dz,
                                    r2, invr, source_bodies, j)
                            u += du
                            gx += dgx; gy += dgy; gz += dgz
                        end
                    end
                end
                j += 1
            end
            if APPLY && has_i
                CUDA.@atomic output[2, i] += gx
                CUDA.@atomic output[3, i] += gy
                CUDA.@atomic output[4, i] += gz
                if HS
                    CUDA.@atomic output[5, i] += h1
                    CUDA.@atomic output[6, i] += h2
                    CUDA.@atomic output[7, i] += h3
                    CUDA.@atomic output[8, i] += h4
                    CUDA.@atomic output[9, i] += h5
                    CUDA.@atomic output[10, i] += h6
                    CUDA.@atomic output[11, i] += h7
                    CUDA.@atomic output[12, i] += h8
                    CUDA.@atomic output[13, i] += h9
                end
            end
            blk += 1
        end
        if diag !== nothing && lane == Int32(0)
            d_tested += UInt64(1)
            all_skip && (d_skipped += UInt64(1))
        end
        pair_i += warp_stride
    end
    if diag !== nothing && lane == Int32(0) && d_tested != UInt64(0)
        CUDA.@atomic diag[11] += d_tested
        CUDA.@atomic diag[12] += d_skipped
    end
    return nothing
end

# 13-slot accumulator update shared by the ballot kernels (u slot stays zero —
# the vortex kernels emit no scalar potential).
@inline function _nf_acc(::Val{HS}, acc::NTuple{13,T}, dx, dy, dz, r2, invr,
        gsx, gsy, gsz, g, h) where {HS,T}
    if HS
        v = _vortex_pair_ugh(dx, dy, dz, r2, invr, gsx, gsy, gsz, g, h)
        return (acc[1], acc[2] + v[2], acc[3] + v[3], acc[4] + v[4],
            acc[5] + v[5], acc[6] + v[6], acc[7] + v[7], acc[8] + v[8],
            acc[9] + v[9], acc[10] + v[10], acc[11] + v[11], acc[12] + v[12],
            acc[13] + v[13])
    else
        v = _vortex_pair_ug(dx, dy, dz, invr, gsx, gsy, gsz, g)
        return (acc[1], acc[2] + v[2], acc[3] + v[3], acc[4] + v[4],
            acc[5], acc[6], acc[7], acc[8], acc[9], acc[10], acc[11], acc[12],
            acc[13])
    end
end

# Drain one side of the per-lane queues: every lane streams its own compacted
# source indices, so the math path is warp-uniform (REG selects it); lanes past
# their own count idle, which is the drain-efficiency cost the diagnostics
# report. Warp-converged by construction (the vote bounds the loop).
@inline function _nf_queue_drain(::Val{HS}, ::Val{REG}, qbuf, tid, side::Int32,
        cnt::Int32, xi, yi, zi, source_bodies, sigma_row::Int,
        acc::NTuple{13,T}, ghv::Val=Val(:shipped), shlut=nothing,
        x_max::T=zero(T)) where {HS,REG,T}
    k = Int32(1)
    while CUDA.vote_any_sync(0xffffffff, k <= cnt)
        if k <= cnt
            j = Int(@inbounds qbuf[k, side, tid])
            @inbounds begin
                dx = xi - source_bodies[1, j]
                dy = yi - source_bodies[2, j]
                dz = zi - source_bodies[3, j]
                r2 = dx * dx + dy * dy + dz * dz
                invr = _cuda_fast_rsqrt(r2)
                gsx = source_bodies[5, j]
                gsy = source_bodies[6, j]
                gsz = source_bodies[7, j]
                g = one(T)
                h = -T(3)
                if REG
                    # task 037f: mode-dispatched g/h (the ballot mechanism
                    # cheapens only the transcendental; assembly stays T)
                    rho = r2 * invr / source_bodies[sigma_row, j]
                    g, h = shlut === nothing ? _gaussianerf_g_h(rho, ghv) :
                        _gh_from_lut(shlut, rho, x_max)
                end
                acc = _nf_acc(Val(HS), acc, dx, dy, dz, r2, invr, gsx, gsy, gsz, g, h)
            end
        end
        k += Int32(1)
    end
    return acc
end

# Mechanism (b): warp-ballot queue kernel. Per (warp, source-body) instant the
# lanes vote on the ρ ≤ cutoff predicate; branch-homogeneous instants evaluate
# inline (no overhead beyond the votes), mixed instants defer their source index
# into per-lane per-side shared-memory queues drained side-at-a-time, so the
# expensive-vs-singular math is never predicated against itself. `bucket == 0`
# streams the full direct list with the host count; a positive bucket streams a
# compacted bucket with its device count. `diag` (Nothing in production)
# accumulates the §6.3 homogeneity telemetry.
function _cuda_direct_pairs_queue_kernel!(kernel, output, source_bodies,
        cell_ranges, tgts, srcs, bin_counts, bucket::Int32, base::Int,
        npairs_static::Int, cell_coords, cell_sigma_max, h_leaf, x_min,
        ::Val{HS}, ::Val{PAABB}, diag, ghv::Val=Val(:shipped),
        gh_lut=nothing) where {HS,PAABB}
    T = eltype(output)
    shlut = _nf_lut_active(ghv, gh_lut, kernel) ? _nf_load_gh_lut!(gh_lut) : nothing
    x_max = T(kernel.rho_t)^2
    npairs = bucket == Int32(0) ? npairs_static : Int(@inbounds bin_counts[bucket])
    sigma_row = kernel.sigma_row
    cutoff = T(_pass1_regularized_cutoff(kernel))
    hl = T(h_leaf)
    lane = (threadIdx().x - Int32(1)) % Int32(32)
    warps_per_block = blockDim().x ÷ Int32(32)
    warp_in_block = (threadIdx().x - Int32(1)) ÷ Int32(32)
    pair_i = (blockIdx().x - 1) * warps_per_block + warp_in_block + 1
    warp_stride = gridDim().x * warps_per_block
    tid = threadIdx().x
    qbuf = CUDA.CuStaticSharedArray(Int32, (_NF_QUEUE_CAP, 2, 128))
    d_inst = UInt64(0); d_uni = UInt64(0); d_mixed = UInt64(0)
    d_push = UInt64(0); d_drain = UInt64(0)
    @inbounds while pair_i <= npairs
        target_cell = Int(tgts[base + pair_i])
        source_cell = Int(srcs[base + pair_i])
        # 037e pair-AABB fast path (mixed bucket only): source-cell AABB and
        # per-source-cell σ_max for the per-block reachability vote
        smax_s = zero(T)
        slo_x = zero(T); slo_y = zero(T); slo_z = zero(T)
        if PAABB
            smax_s = T(cell_sigma_max[source_cell])
            slo_x = T(x_min[1]) + T(cell_coords[1, source_cell]) * hl
            slo_y = T(x_min[2]) + T(cell_coords[2, source_cell]) * hl
            slo_z = T(x_min[3]) + T(cell_coords[3, source_cell]) * hl
        end
        tfirst = cell_ranges[1, target_cell]
        tcount = cell_ranges[2, target_cell]
        sfirst = cell_ranges[1, source_cell]
        slast = sfirst + cell_ranges[2, source_cell] - 1
        nblk = (tcount + 31) ÷ 32
        blk = 0
        while blk < nblk
            i = tfirst + blk * 32 + Int(lane)
            has_i = i < tfirst + tcount
            xi = zero(T); yi = zero(T); zi = zero(T)
            if has_i
                xi = source_bodies[1, i]
                yi = source_bodies[2, i]
                zi = source_bodies[3, i]
            end
            acc = (zero(T), zero(T), zero(T), zero(T), zero(T), zero(T), zero(T),
                zero(T), zero(T), zero(T), zero(T), zero(T), zero(T))
            if PAABB
                can_reach = has_i && _nearfield_point_aabb_reach(xi, yi, zi,
                    slo_x, slo_y, slo_z, hl, cutoff, smax_s)
                if !CUDA.vote_any_sync(0xffffffff, can_reach)
                    # no lane can reach the regularized zone: every instant of
                    # this block is a uniform singular instant — run it inline,
                    # FP-identical in kind and order to the shipped uniform-
                    # singular path below (no votes, no σ loads, no queues)
                    j = sfirst
                    while j <= slast
                        dxs = xi - source_bodies[1, j]
                        dys = yi - source_bodies[2, j]
                        dzs = zi - source_bodies[3, j]
                        r2s = dxs * dxs + dys * dys + dzs * dzs
                        if has_i && i != j && r2s > zero(T)
                            invrs = _cuda_fast_rsqrt(r2s)
                            gsx = source_bodies[5, j]
                            gsy = source_bodies[6, j]
                            gsz = source_bodies[7, j]
                            acc = _nf_acc(Val(HS), acc, dxs, dys, dzs, r2s,
                                invrs, gsx, gsy, gsz, one(T), -T(3))
                        end
                        j += 1
                    end
                    if has_i
                        CUDA.@atomic output[2, i] += acc[2]
                        CUDA.@atomic output[3, i] += acc[3]
                        CUDA.@atomic output[4, i] += acc[4]
                        if HS
                            CUDA.@atomic output[5, i] += acc[5]
                            CUDA.@atomic output[6, i] += acc[6]
                            CUDA.@atomic output[7, i] += acc[7]
                            CUDA.@atomic output[8, i] += acc[8]
                            CUDA.@atomic output[9, i] += acc[9]
                            CUDA.@atomic output[10, i] += acc[10]
                            CUDA.@atomic output[11, i] += acc[11]
                            CUDA.@atomic output[12, i] += acc[12]
                            CUDA.@atomic output[13, i] += acc[13]
                        end
                    end
                    blk += 1
                    continue
                end
            end
            cntR = Int32(0)
            cntS = Int32(0)
            j = sfirst
            while j <= slast
                xj = source_bodies[1, j]
                yj = source_bodies[2, j]
                zj = source_bodies[3, j]
                dx = xi - xj
                dy = yi - yj
                dz = zi - zj
                r2 = dx * dx + dy * dy + dz * dz
                valid = has_i && i != j && r2 > zero(T)
                invr = zero(T)
                rho = zero(T)
                pred_in = false
                if valid
                    invr = _cuda_fast_rsqrt(r2)
                    sigma = source_bodies[sigma_row, j]
                    if sigma > zero(T)
                        rho = r2 * invr / sigma
                        pred_in = rho <= cutoff
                    end
                end
                mask_in = CUDA.vote_ballot_sync(0xffffffff, valid && pred_in)
                mask_out = CUDA.vote_ballot_sync(0xffffffff, valid && !pred_in)
                if diag !== nothing && lane == Int32(0) && (mask_in | mask_out) != 0x00000000
                    d_inst += UInt64(1)
                    if mask_in == 0x00000000 || mask_out == 0x00000000
                        d_uni += UInt64(1)
                    else
                        d_mixed += UInt64(1)
                    end
                end
                if mask_in == 0x00000000
                    # uniform singular instant
                    if valid
                        gsx = source_bodies[5, j]
                        gsy = source_bodies[6, j]
                        gsz = source_bodies[7, j]
                        acc = _nf_acc(Val(HS), acc, dx, dy, dz, r2, invr,
                            gsx, gsy, gsz, one(T), -T(3))
                    end
                elseif mask_out == 0x00000000
                    # uniform regularized instant
                    if valid
                        gsx = source_bodies[5, j]
                        gsy = source_bodies[6, j]
                        gsz = source_bodies[7, j]
                        g, h = shlut === nothing ? _gaussianerf_g_h(rho, ghv) :
                            _gh_from_lut(shlut, rho, x_max)
                        acc = _nf_acc(Val(HS), acc, dx, dy, dz, r2, invr,
                            gsx, gsy, gsz, g, h)
                    end
                else
                    # mixed instant: defer into the per-lane queues
                    if valid
                        if pred_in
                            cntR += Int32(1)
                            qbuf[cntR, 1, tid] = Int32(j)
                        else
                            cntS += Int32(1)
                            qbuf[cntS, 2, tid] = Int32(j)
                        end
                        diag === nothing || (d_push += UInt64(1))
                    end
                    if CUDA.vote_any_sync(0xffffffff, cntR == Int32(_NF_QUEUE_CAP))
                        diag === nothing || (d_drain += UInt64(cntR))
                        acc = _nf_queue_drain(Val(HS), Val(true), qbuf, tid,
                            Int32(1), cntR, xi, yi, zi, source_bodies, sigma_row,
                            acc, ghv, shlut, x_max)
                        cntR = Int32(0)
                    end
                    if CUDA.vote_any_sync(0xffffffff, cntS == Int32(_NF_QUEUE_CAP))
                        diag === nothing || (d_drain += UInt64(cntS))
                        acc = _nf_queue_drain(Val(HS), Val(false), qbuf, tid,
                            Int32(2), cntS, xi, yi, zi, source_bodies, sigma_row, acc)
                        cntS = Int32(0)
                    end
                end
                j += 1
            end
            if CUDA.vote_any_sync(0xffffffff, cntR > Int32(0))
                diag === nothing || (d_drain += UInt64(cntR))
                acc = _nf_queue_drain(Val(HS), Val(true), qbuf, tid, Int32(1),
                    cntR, xi, yi, zi, source_bodies, sigma_row, acc, ghv,
                    shlut, x_max)
            end
            if CUDA.vote_any_sync(0xffffffff, cntS > Int32(0))
                diag === nothing || (d_drain += UInt64(cntS))
                acc = _nf_queue_drain(Val(HS), Val(false), qbuf, tid, Int32(2),
                    cntS, xi, yi, zi, source_bodies, sigma_row, acc)
            end
            if has_i
                CUDA.@atomic output[2, i] += acc[2]
                CUDA.@atomic output[3, i] += acc[3]
                CUDA.@atomic output[4, i] += acc[4]
                if HS
                    CUDA.@atomic output[5, i] += acc[5]
                    CUDA.@atomic output[6, i] += acc[6]
                    CUDA.@atomic output[7, i] += acc[7]
                    CUDA.@atomic output[8, i] += acc[8]
                    CUDA.@atomic output[9, i] += acc[9]
                    CUDA.@atomic output[10, i] += acc[10]
                    CUDA.@atomic output[11, i] += acc[11]
                    CUDA.@atomic output[12, i] += acc[12]
                    CUDA.@atomic output[13, i] += acc[13]
                end
            end
            blk += 1
        end
        pair_i += warp_stride
    end
    if diag !== nothing
        if lane == Int32(0) && (d_inst | d_uni | d_mixed) != UInt64(0)
            CUDA.@atomic diag[1] += d_inst
            CUDA.@atomic diag[2] += d_uni
            CUDA.@atomic diag[4] += d_mixed
        end
        if (d_push | d_drain) != UInt64(0)
            CUDA.@atomic diag[7] += d_push
            CUDA.@atomic diag[8] += d_drain
        end
    end
    return nothing
end

# §6.3 homogeneity telemetry for the predicated kernels: replays the warp-per-
# pair traversal computing only the ρ predicate ballots. diag (UInt64[8]):
#   1 instants (warp × source × lane-block with ≥ 1 valid lane)
#   2 branch-homogeneous instants   3 unused   4 mixed instants
#   5 valid lane-pairs              6 regularized lane-pairs
function _cuda_nearfield_divergence_kernel!(diag, source_bodies, cell_ranges,
        tgts, srcs, base::Int, npairs::Int, sigma_row::Int, cutoff)
    T = typeof(cutoff)
    lane = (threadIdx().x - Int32(1)) % Int32(32)
    warps_per_block = blockDim().x ÷ Int32(32)
    warp_in_block = (threadIdx().x - Int32(1)) ÷ Int32(32)
    pair_i = (blockIdx().x - 1) * warps_per_block + warp_in_block + 1
    warp_stride = gridDim().x * warps_per_block
    d_inst = UInt64(0); d_uni = UInt64(0); d_mixed = UInt64(0)
    d_valid = UInt64(0); d_reg = UInt64(0)
    @inbounds while pair_i <= npairs
        target_cell = Int(tgts[base + pair_i])
        source_cell = Int(srcs[base + pair_i])
        tfirst = cell_ranges[1, target_cell]
        tcount = cell_ranges[2, target_cell]
        sfirst = cell_ranges[1, source_cell]
        slast = sfirst + cell_ranges[2, source_cell] - 1
        nblk = (tcount + 31) ÷ 32
        blk = 0
        while blk < nblk
            i = tfirst + blk * 32 + Int(lane)
            has_i = i < tfirst + tcount
            xi = zero(T); yi = zero(T); zi = zero(T)
            if has_i
                xi = T(source_bodies[1, i])
                yi = T(source_bodies[2, i])
                zi = T(source_bodies[3, i])
            end
            j = sfirst
            while j <= slast
                dx = xi - T(source_bodies[1, j])
                dy = yi - T(source_bodies[2, j])
                dz = zi - T(source_bodies[3, j])
                r2 = dx * dx + dy * dy + dz * dz
                valid = has_i && i != j && r2 > zero(T)
                pred_in = false
                if valid
                    sigma = T(source_bodies[sigma_row, j])
                    if sigma > zero(T)
                        rho = r2 * _cuda_fast_rsqrt(r2) / sigma
                        pred_in = rho <= cutoff
                    end
                end
                mask_in = CUDA.vote_ballot_sync(0xffffffff, valid && pred_in)
                mask_out = CUDA.vote_ballot_sync(0xffffffff, valid && !pred_in)
                if lane == Int32(0) && (mask_in | mask_out) != 0x00000000
                    d_inst += UInt64(1)
                    if mask_in == 0x00000000 || mask_out == 0x00000000
                        d_uni += UInt64(1)
                    else
                        d_mixed += UInt64(1)
                    end
                    d_valid += UInt64(count_ones(mask_in | mask_out))
                    d_reg += UInt64(count_ones(mask_in))
                end
                j += 1
            end
            blk += 1
        end
        pair_i += warp_stride
    end
    if lane == Int32(0) && d_inst != UInt64(0)
        CUDA.@atomic diag[1] += d_inst
        CUDA.@atomic diag[2] += d_uni
        CUDA.@atomic diag[4] += d_mixed
        CUDA.@atomic diag[5] += d_valid
        CUDA.@atomic diag[6] += d_reg
    end
    return nothing
end

# TwoPassVortex pass-2 deficit sweep (031a §6.1), device mirror of
# `_host_twopass_deficit_kernel!`. Work item = (occupied leaf cell, offset-ball
# entry); warp per item, grid-stride. The construction-built gap-ascending ball
# is pruned per entry against the live device (rho_t σ_max)² scalar, so reach
# coverage holds every step with no per-step list rebuild; entries whose
# far-corner lies inside rho_c·σ_min(source cell) are skipped whole (all their
# pairs are pass-1-complete). QUEUED drains shell pairs through the per-lane
# ballot queue; otherwise the shell membership is predicated. APPLY=false skips
# the output atomics (homogeneity telemetry only, diag slots 1/2/4).
function _cuda_twopass_deficit_kernel!(kernel::TwoPassVortex, output,
        source_bodies, cell_ranges, cell_coords, cell_sigma_max, cell_sigma_min,
        cell_keys,
        n_cells::Int, tp_offsets, tp_gap2, K::Int, nf_scalars, ell::Int,
        h_leaf, x_min, ::Val{HS}, ::Val{QUEUED}, ::Val{AABB}, ::Val{APPLY},
        diag) where {HS,QUEUED,AABB,APPLY}
    T = eltype(output)
    # the ball prune runs in Float64 so Float32 rounding can never exclude a
    # boundary offset class the host sweep would visit
    reach2 = @inbounds nf_scalars[2]
    hl2d = Float64(h_leaf) * Float64(h_leaf)
    hl2 = h_leaf * h_leaf
    rho_c = T(kernel.rho_c)
    rho_t = T(kernel.rho_t)
    sigma_row = kernel.sigma_row
    lane = (threadIdx().x - Int32(1)) % Int32(32)
    warps_per_block = blockDim().x ÷ Int32(32)
    warp_in_block = (threadIdx().x - Int32(1)) ÷ Int32(32)
    w = (blockIdx().x - 1) * warps_per_block + warp_in_block + 1
    warp_stride = gridDim().x * warps_per_block
    tid = threadIdx().x
    qbuf = CUDA.CuStaticSharedArray(Int32, (_NF_QUEUE_CAP, 2, 128))
    d_inst = UInt64(0); d_uni = UInt64(0); d_mixed = UInt64(0)
    d_candidate_pairs = UInt64(0); d_shell_pairs = UInt64(0)
    total = n_cells * K
    G = 1 << ell
    @inbounds while w <= total
        cell = (w - 1) ÷ K + 1
        k = (w - 1) % K + 1
        gap2h = Float64(tp_gap2[k]) * hl2d
        if gap2h <= reach2
            sx = cell_coords[1, cell] + Int(tp_offsets[1, k])
            sy = cell_coords[2, cell] + Int(tp_offsets[2, k])
            sz = cell_coords[3, cell] + Int(tp_offsets[3, k])
            if 0 <= sx < G && 0 <= sy < G && 0 <= sz < G
                src = _cuda_cell_key_search(cell_keys, n_cells,
                    _cuda_morton_key(sx, sy, sz, ell))
                if src != 0
                    # inner-core prune: farthest corner inside rho_c·σ_min ⇒
                    # every pair is below the shell and pass-1-complete
                    smin = T(cell_sigma_min[src])
                    ox = Int(tp_offsets[1, k]); oy = Int(tp_offsets[2, k])
                    oz = Int(tp_offsets[3, k])
                    mx = T(abs(ox) + 1); my = T(abs(oy) + 1); mz = T(abs(oz) + 1)
                    dmax2 = (mx * mx + my * my + mz * mz) * hl2
                    core = smin > zero(T) && dmax2 <= (rho_c * smin)^2
                    if !core
                        smax = T(cell_sigma_max[src])
                        slo_x = T(x_min[1]) + T(sx) * h_leaf
                        slo_y = T(x_min[2]) + T(sy) * h_leaf
                        slo_z = T(x_min[3]) + T(sz) * h_leaf
                        shi_x = slo_x + h_leaf
                        shi_y = slo_y + h_leaf
                        shi_z = slo_z + h_leaf
                        tfirst = cell_ranges[1, cell]
                        tcount = cell_ranges[2, cell]
                        sfirst = cell_ranges[1, src]
                        slast = sfirst + cell_ranges[2, src] - 1
                        nblk = (tcount + 31) ÷ 32
                        blk = 0
                        while blk < nblk
                            i = tfirst + blk * 32 + Int(lane)
                            has_i = i < tfirst + tcount
                            xi = zero(T); yi = zero(T); zi = zero(T)
                            if has_i
                                xi = source_bodies[1, i]
                                yi = source_bodies[2, i]
                                zi = source_bodies[3, i]
                            end
                            could_shell = has_i
                            if AABB && has_i
                                qx = xi < slo_x ? slo_x - xi :
                                    (xi > shi_x ? xi - shi_x : zero(T))
                                qy = yi < slo_y ? slo_y - yi :
                                    (yi > shi_y ? yi - shi_y : zero(T))
                                qz = zi < slo_z ? slo_z - zi :
                                    (zi > shi_z ? zi - shi_z : zero(T))
                                near2 = qx*qx + qy*qy + qz*qz
                                fx = max(abs(xi - slo_x), abs(xi - shi_x))
                                fy = max(abs(yi - slo_y), abs(yi - shi_y))
                                fz = max(abs(zi - slo_z), abs(zi - shi_z))
                                far2 = fx*fx + fy*fy + fz*fz
                                could_shell = near2 <= (rho_t * smax)^2 &&
                                    !(smin > zero(T) &&
                                      far2 <= (rho_c * smin)^2)
                            end
                            acc = (zero(T), zero(T), zero(T), zero(T), zero(T),
                                zero(T), zero(T), zero(T), zero(T), zero(T),
                                zero(T), zero(T), zero(T))
                            cntQ = Int32(0)
                            any_shell = !AABB || CUDA.vote_any_sync(0xffffffff,
                                could_shell)
                            j = sfirst
                            while any_shell && j <= slast
                                dx = xi - source_bodies[1, j]
                                dy = yi - source_bodies[2, j]
                                dz = zi - source_bodies[3, j]
                                r2 = dx * dx + dy * dy + dz * dz
                                valid = could_shell && i != j && r2 > zero(T)
                                invr = zero(T)
                                rho = zero(T)
                                in_shell = false
                                if valid
                                    invr = _cuda_fast_rsqrt(r2)
                                    sigma = source_bodies[sigma_row, j]
                                    if sigma > zero(T)
                                        rho = r2 * invr / sigma
                                        in_shell = rho_c < rho <= rho_t
                                    end
                                end
                                if QUEUED || diag !== nothing
                                    mask_sh = CUDA.vote_ballot_sync(0xffffffff,
                                        valid && in_shell)
                                    mask_no = CUDA.vote_ballot_sync(0xffffffff,
                                        valid && !in_shell)
                                    if diag !== nothing && lane == Int32(0) &&
                                            (mask_sh | mask_no) != 0x00000000
                                        d_inst += UInt64(1)
                                        d_candidate_pairs += UInt64(count_ones(mask_sh | mask_no))
                                        d_shell_pairs += UInt64(count_ones(mask_sh))
                                        if mask_sh == 0x00000000 || mask_no == 0x00000000
                                            d_uni += UInt64(1)
                                        else
                                            d_mixed += UInt64(1)
                                        end
                                    end
                                    if QUEUED
                                        if mask_sh != 0x00000000
                                            if mask_no == 0x00000000
                                                # uniform shell instant: inline
                                                if valid && in_shell
                                                    acc = _nf_twopass_acc(Val(HS),
                                                        acc, dx, dy, dz, r2, invr,
                                                        rho, source_bodies, j)
                                                end
                                            else
                                                if valid && in_shell
                                                    cntQ += Int32(1)
                                                    qbuf[cntQ, 1, tid] = Int32(j)
                                                end
                                                if CUDA.vote_any_sync(0xffffffff,
                                                        cntQ == Int32(_NF_QUEUE_CAP))
                                                    acc = _nf_twopass_drain(Val(HS),
                                                        qbuf, tid, cntQ, xi, yi, zi,
                                                        source_bodies, sigma_row,
                                                        rho_c, rho_t, acc)
                                                    cntQ = Int32(0)
                                                end
                                            end
                                        end
                                    end
                                end
                                if !QUEUED
                                    if valid && in_shell
                                        acc = _nf_twopass_acc(Val(HS), acc, dx, dy,
                                            dz, r2, invr, rho, source_bodies, j)
                                    end
                                end
                                j += 1
                            end
                            if QUEUED && CUDA.vote_any_sync(0xffffffff, cntQ > Int32(0))
                                acc = _nf_twopass_drain(Val(HS), qbuf, tid, cntQ,
                                    xi, yi, zi, source_bodies, sigma_row, rho_c,
                                    rho_t, acc)
                            end
                            if APPLY && has_i
                                CUDA.@atomic output[2, i] += acc[2]
                                CUDA.@atomic output[3, i] += acc[3]
                                CUDA.@atomic output[4, i] += acc[4]
                                if HS
                                    CUDA.@atomic output[5, i] += acc[5]
                                    CUDA.@atomic output[6, i] += acc[6]
                                    CUDA.@atomic output[7, i] += acc[7]
                                    CUDA.@atomic output[8, i] += acc[8]
                                    CUDA.@atomic output[9, i] += acc[9]
                                    CUDA.@atomic output[10, i] += acc[10]
                                    CUDA.@atomic output[11, i] += acc[11]
                                    CUDA.@atomic output[12, i] += acc[12]
                                    CUDA.@atomic output[13, i] += acc[13]
                                end
                            end
                            blk += 1
                        end
                    end
                end
            end
        end
        w += warp_stride
    end
    if diag !== nothing && lane == Int32(0) && d_inst != UInt64(0)
        CUDA.@atomic diag[1] += d_inst
        CUDA.@atomic diag[2] += d_uni
        CUDA.@atomic diag[4] += d_mixed
        CUDA.@atomic diag[9] += d_candidate_pairs
        CUDA.@atomic diag[10] += d_shell_pairs
    end
    return nothing
end

# §6.1 deficit accumulation for one shell pair: effective (g, h) = (−ḡ, ρg′+3ḡ)
# through the shared vortex assembly (see `_twopass_deficit_gh`).
@inline function _nf_twopass_acc(::Val{HS}, acc::NTuple{13,T}, dx, dy, dz, r2,
        invr, rho, source_bodies, j) where {HS,T}
    gbar, rhogp = _gaussianerf_gbar_rhogp(rho)
    @inbounds begin
        gsx = source_bodies[5, j]
        gsy = source_bodies[6, j]
        gsz = source_bodies[7, j]
    end
    return _nf_acc(Val(HS), acc, dx, dy, dz, r2, invr, gsx, gsy, gsz,
        -gbar, muladd(T(3), gbar, rhogp))
end

@inline function _nf_twopass_drain(::Val{HS}, qbuf, tid, cnt::Int32, xi, yi, zi,
        source_bodies, sigma_row::Int, rho_c::T, rho_t::T,
        acc::NTuple{13,T}) where {HS,T}
    k = Int32(1)
    while CUDA.vote_any_sync(0xffffffff, k <= cnt)
        if k <= cnt
            j = Int(@inbounds qbuf[k, 1, tid])
            @inbounds begin
                dx = xi - source_bodies[1, j]
                dy = yi - source_bodies[2, j]
                dz = zi - source_bodies[3, j]
                r2 = dx * dx + dy * dy + dz * dz
                invr = _cuda_fast_rsqrt(r2)
                rho = r2 * invr / source_bodies[sigma_row, j]
            end
            acc = _nf_twopass_acc(Val(HS), acc, dx, dy, dz, r2, invr, rho,
                source_bodies, j)
        end
        k += Int32(1)
    end
    return acc
end

# One warp per compact entry. Positive target ids denote an unordered cell pair;
# negative ids retain an oversized directed fallback entry. Cross-cell and
# triangular same-cell branches evaluate each ordinary body pair once and update
# both endpoints. This is valid only for the same-source/target scalar kernel.
function _cuda_symmetric_pairs_output_kernel!(output, source_bodies, cell_ranges,
        pair_targets, pair_sources, npairs, ::Val{HS}=Val(false)) where HS
    T = eltype(output)
    lane = (threadIdx().x - Int32(1)) % Int32(32)
    warps_per_block = blockDim().x ÷ Int32(32)
    warp_in_block = (threadIdx().x - Int32(1)) ÷ Int32(32)
    pair_i = (blockIdx().x - 1) * warps_per_block + warp_in_block + 1
    warp_stride = gridDim().x * warps_per_block
    c = inv(T(4) * T(π))
    @inbounds while pair_i <= npairs
        encoded_target = pair_targets[pair_i]
        source_cell = pair_sources[pair_i]
        fallback = encoded_target < 0
        target_cell = abs(encoded_target)
        tfirst = cell_ranges[1, target_cell]
        tlast = tfirst + cell_ranges[2, target_cell] - 1
        sfirst = cell_ranges[1, source_cell]
        slast = sfirst + cell_ranges[2, source_cell] - 1
        i = tfirst + lane
        while i <= tlast
            xi = source_bodies[1, i]
            yi = source_bodies[2, i]
            zi = source_bodies[3, i]
            qi = source_bodies[5, i] * c
            ui = zero(T)
            gxi = zero(T)
            gyi = zero(T)
            gzi = zero(T)
            hxxi = zero(T); hxyi = zero(T); hxzi = zero(T)
            hyyi = zero(T); hyzi = zero(T); hzzi = zero(T)
            jfirst = fallback ? sfirst :
                (target_cell == source_cell ? max(i + 1, sfirst) : sfirst)
            for j in jfirst:slast
                fallback && i == j && continue
                dx = xi - source_bodies[1, j]
                dy = yi - source_bodies[2, j]
                dz = zi - source_bodies[3, j]
                r2 = dx * dx + dy * dy + dz * dz
                if r2 > zero(r2)
                    invr = _cuda_fast_rsqrt(r2)
                    qj = source_bodies[5, j] * c
                    invr2 = invr * invr
                    invr3 = invr * invr2
                    ui += qj * invr
                    gxi -= qj * dx * invr3
                    gyi -= qj * dy * invr3
                    gzi -= qj * dz * invr3
                    if HS
                        # geometric hessian factor is even in Δx, so both
                        # endpoints receive the same pattern scaled by the
                        # other body's strength
                        f3invr5 = 3 * invr3 * invr2
                        hxx = f3invr5 * dx * dx - invr3
                        hxy = f3invr5 * dx * dy
                        hxz = f3invr5 * dx * dz
                        hyy = f3invr5 * dy * dy - invr3
                        hyz = f3invr5 * dy * dz
                        hzz = f3invr5 * dz * dz - invr3
                        hxxi += qj * hxx; hxyi += qj * hxy; hxzi += qj * hxz
                        hyyi += qj * hyy; hyzi += qj * hyz; hzzi += qj * hzz
                        if !fallback
                            CUDA.@atomic output[5, j] += qi * hxx
                            CUDA.@atomic output[6, j] += qi * hxy
                            CUDA.@atomic output[7, j] += qi * hxz
                            CUDA.@atomic output[8, j] += qi * hxy
                            CUDA.@atomic output[9, j] += qi * hyy
                            CUDA.@atomic output[10, j] += qi * hyz
                            CUDA.@atomic output[11, j] += qi * hxz
                            CUDA.@atomic output[12, j] += qi * hyz
                            CUDA.@atomic output[13, j] += qi * hzz
                        end
                    end
                    if !fallback
                        CUDA.@atomic output[1, j] += qi * invr
                        CUDA.@atomic output[2, j] += qi * dx * invr3
                        CUDA.@atomic output[3, j] += qi * dy * invr3
                        CUDA.@atomic output[4, j] += qi * dz * invr3
                    end
                end
            end
            CUDA.@atomic output[1, i] += ui
            CUDA.@atomic output[2, i] += gxi
            CUDA.@atomic output[3, i] += gyi
            CUDA.@atomic output[4, i] += gzi
            if HS
                CUDA.@atomic output[5, i] += hxxi
                CUDA.@atomic output[6, i] += hxyi
                CUDA.@atomic output[7, i] += hxzi
                CUDA.@atomic output[8, i] += hxyi
                CUDA.@atomic output[9, i] += hyyi
                CUDA.@atomic output[10, i] += hyzi
                CUDA.@atomic output[11, i] += hxzi
                CUDA.@atomic output[12, i] += hyzi
                CUDA.@atomic output[13, i] += hzzi
            end
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

# 13-row variant (task 032): identical warp-per-cell shape; the hessian rows
# come from `_resident_local_eval_flat_hessian` (shared host/device math).
function _cuda_l2b_output_hessian_kernel!(output, source_bodies, cell_centers,
        cell_ranges, leaf_to_node, local_phi, local_chi, P_phi, P_active, lhv, ncell)
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
        vals = _resident_local_eval_flat_hessian(
            local_phi, local_chi, node,
            source_bodies[1, i] - cx,
            source_bodies[2, i] - cy,
            source_bodies[3, i] - cz,
            P_phi, P_active, lhv,
        )
        for row in 1:13
            output[row, i] += vals[row]
        end
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
# (radix_setting(:FACTORED_CUDA_WHOLE_PASS) = false).
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
    W = max(min(radix_setting(:FACTORED_CUDA_CHUNK), length(plan.route_class)), 1)
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
    if wp !== nothing && radix_setting(:FACTORED_CUDA_WHOLE_PASS)
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
    W = max(min(radix_setting(:PRECOMPUTED_CUDA_CHUNK), length(plan.route_class)), 1)
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
    if radix_setting(:PRECOMPUTED_CUDA_WHOLE_PASS)
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
# (radix_setting(:DENSE_CUDA_FUSED), gather/matvec/scatter in one launch, no slabs or cuBLAS)
# and two GEMM drivers sharing the gather/scatter kernels and the chunk-width
# slabs — the whole-pass driver (default) gathers/scatters once per route chunk
# with per-class GEMMs over the chunk, and the per-class reference driver
# (radix_setting(:DENSE_CUDA_WHOLE_PASS) = false) gathers/GEMMs/scatters per class, matching
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
const DENSE_CUDA_TILED_THREADS = Ref(64)
const DENSE_CUDA_TILED_MAX_BLOCKS = Ref(65536)
"""
Input format of the hierarchical dense leaf M2L: `:off` (FP32 tiled kernel),
`:fp16`, or `:bf16` (WMMA 16x16 tensor kernel with FP32 accumulation).

`:fp16` is the default: task 028 Stage 8 measured it at 2.54 ms against 5.47 ms
for the FP32 tiled kernel on the shipped geometry, and it is the difference
between 12.5 ms and 9.591 ms per complete resident step at `n = 1e6`, `P = 4`
(job 13029878), at 1.0593e-3 sampled gradient relative RMS versus 1.0498e-3 for
FP32 — inside the same accuracy gate.

The tensor kernel is only reachable for `Float32`, no Lamb-Helmholtz, and
`D == 16` (`expansion_order = 3`); every other configuration silently uses the
FP32 tiled kernel, and switching this after cache construction falls back rather
than using a cache that was never built.

!!! warning "FP16 dynamic range is not scale-invariant"
    Each operator column is rescaled to the top of the FP16 range and the
    reciprocal is applied to the multipole side, so the *inputs* carry a factor
    `max|K[:,i]| / 6e4`. The scale is derived from the operator alone, not from
    the source strengths, so a problem whose strengths (and hence multipole
    coefficients) are many orders of magnitude away from the validated
    benchmark's can underflow the FP16 input to zero and silently lose a
    contribution. Set this to `:off` for such problems, or validate against
    `:off` on a sample.
"""
const DENSE_CUDA_TENSOR_FORMAT = Ref(:fp16) # :off, :fp16, or :bf16 (task 028 Stage 8)

# Grid-stride cap for the warp-per-pair nearfield kernel (task 028 lever 1),
# same role as DENSE_CUDA_FUSED_MAX_BLOCKS above: blocks = min(cld(npairs,
# warps_per_block), this cap), and each warp walks pairs `total_warps` apart.
# At n=1e6/ell=5 there are 5,189,728 direct cell-pairs of ~30x30 bodies.
const DIRECT_CUDA_MAX_BLOCKS = Ref(16384)
# Same-system scalar-only specialization remains internal until the Stage 8
# bake-off banks it. Construction always provisions/refreshes its compact list
# so A/B timing includes the required recurring compaction cost.
const CUDA_SYMMETRIC_NEARFIELD = Ref(false)
const SYMMETRIC_CUDA_MAX_CELL_BODIES = Ref(128)

#------- task 029 cycle 1: sync-free far-field chain -------#
#
# Two cooperating mechanisms behind runtime flags (current path preserved as
# the fallback, mirroring CUDA_OVERLAP_NEARFIELD):
#
# 1. `CUDA_CACHED_WINDOWS`: the hierarchical M2L route windows, direct pairs,
#    node metadata, and operator-group edges are pure functions of the occupied
#    cell set (the cache's Morton box is fixed, so cell/node centers and
#    parent/child topology depend only on which cells are occupied). The
#    refresh detects occupancy change by comparing the sorted unique leaf keys
#    against the previous step's snapshot and regenerates all of the above only
#    on change ("occupancy epoch"). The M2L windows are additionally cached as
#    a per-level concatenation in generation order, so the steady-state M2L
#    stage launches only the per-level apply kernels — no flags/scan/compact
#    work and no blocking route-count D2H (job 13059955: ~1.2 ms device-busy
#    plus a 1.25 ms sync per step at the robust n=1e6 baseline).
# 2. `CUDA_GRAPH_LIFECYCLE`: with the windows cached and the GEMM scalars
#    device-staged, the complete far-field chain (side-stream fill+nearfield,
#    B2M, M2M, per-level M2L applies, L2L, L2B) is sync-free and
#    capacity-static within an epoch, so it is captured once per occupancy
#    epoch into a CUDA graph and replayed with a single launch per step
#    (job 13059955: 420 launches / 27 syncs / ~3 ms n-independent floor).
#
# Graph re-capture rule (counted at its real recurrence in any verdict): the
# captured graph is valid exactly while the occupancy epoch is unchanged.
# `update_cuda_radix_state!` increments `hctx.epoch_id` when the occupied cell
# set changes (which covers node sets, window contents, direct-pair lists, and
# every launch shape baked into the graph); the first lifecycle of a new epoch
# runs uncaptured (JIT/handle warm-up), the second records and instantiates,
# and subsequent steps replay. Geometry-changing operations (bodies crossing
# cell boundaries, body-count changes) therefore cost one regeneration pass
# plus one capture; a workload whose occupancy churns every step degrades to
# the pre-029 launch pattern plus capture overhead, which the epoch check makes
# visible rather than silent.
const CUDA_CACHED_WINDOWS = Ref(true)
const CUDA_GRAPH_LIFECYCLE = Ref(true)

# Construction-staged device alpha=1/beta=0 for the resident-chain GEMMs, keyed
# by (context handle, eltype) so device resets or multi-context test runs never
# reuse stale device memory. In CUBLAS_POINTER_MODE_DEVICE a scalar-alpha
# `mul!` stages a fresh `CuRef` per call: one pool allocation plus one pageable
# H2D memcpy, measured as 101 pageable H2Ds per step across the M2M/L2L groups
# (job 13059955) and incompatible with graph capture.
const _CUDA_GEMM_SCALARS = Dict{Tuple{UInt,DataType},Any}()

function _cuda_gemm_scalars(::Type{TF}) where TF
    key = (objectid(CUDA.context()), TF)
    return get!(_CUDA_GEMM_SCALARS, key) do
        (CUDA.CuArray(TF[one(TF)]), CUDA.zeros(TF, 1))
    end::Tuple{CUDA.CuVector{TF},CUDA.CuVector{TF}}
end

function _resident_mul!(C::CUDA.StridedCuMatrix{TF}, A::CUDA.StridedCuMatrix{TF},
        B::CUDA.StridedCuMatrix{TF}) where {TF<:Union{Float32,Float64}}
    alpha, beta = _cuda_gemm_scalars(TF)
    CUDA.CUBLAS.gemm!('N', 'N', alpha, A, B, beta, C)
    return C
end

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

# Per-class reference driver (radix_setting(:DENSE_CUDA_WHOLE_PASS) = false): gather/GEMM/scatter
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
    radix_setting(:DENSE_CUDA_FUSED) && return _launch_resident_m2l_dense_fused!(state, ws, plan)
    wp = plan.whole_pass[]
    wp isa NamedTuple || throw(ArgumentError(
        "CUDA dense M2L requires the whole-pass scratch bundle"))
    if radix_setting(:DENSE_CUDA_WHOLE_PASS)
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
    tensor_cached = radix_setting(:DENSE_CUDA_TENSOR_FORMAT) in (:fp16, :bf16)
    tensor_operator_bytes = TF === Float32 && !LH && D == 16 && tensor_cached ?
        _dense_sum_checked((
            _dense_checked_mul(_dense_checked_mul(nclasses, D * D,
                "tensor operator elements"), sizeof(Float16),
                "tensor operator bytes"),
            _dense_checked_mul(_dense_checked_mul(nclasses, D,
                "tensor scale elements"), sizeof(Float32),
                "tensor scale bytes")), "tensor cache bytes") : 0
    operator_bytes = _dense_checked_add(base.operator_bytes, tensor_operator_bytes,
        "dense plus tensor operator bytes")
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
        _dense_checked_mul(radix_setting(:CUDA_SYMMETRIC_NEARFIELD) && !LH ? 6 : 4,
            direct_capacity, "direct/symmetric index arrays"),
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
    W = max(min(radix_setting(:DENSE_CUDA_CHUNK), nroutes), 1)

    payload = _dense_m2l_footprint(TF, basis_info, nclasses, nroutes, W, D)
    tensor_format = radix_setting(:DENSE_CUDA_TENSOR_FORMAT)
    tensor_supported = TF === Float32 && !LH && D == 16 &&
        tensor_format in (:fp16, :bf16)
    tensor_operator_bytes = tensor_supported ?
        _dense_sum_checked((
            _dense_checked_mul(_dense_checked_mul(nclasses, D * D,
                "tensor operator elements"), sizeof(Float16),
                "tensor operator bytes"),
            _dense_checked_mul(_dense_checked_mul(nclasses, D,
                "tensor scale elements"), sizeof(Float32),
                "tensor scale bytes")), "tensor cache bytes") : 0
    operator_bytes = _dense_checked_add(payload.operator_bytes,
        tensor_operator_bytes, "dense plus tensor operator bytes")
    route_class_bytes = _dense_checked_mul(nroutes, sizeof(Int32), "route class bytes")
    class_hist_bytes = _dense_checked_mul(nclasses, sizeof(Int32), "class histogram bytes")
    route_metadata_bytes = _dense_sum_checked((route_class_bytes, class_hist_bytes),
        "dense CUDA route metadata")
    persistent_bytes = _dense_sum_checked(
        (operator_bytes, payload.scratch_bytes, route_metadata_bytes),
        "dense CUDA persistent")
    plan_estimated_peak_bytes = max(_dense_to_int(estimated_peak_bytes,
        "dense CUDA estimated peak bytes"), persistent_bytes)
    persistent_bytes <= strategy.max_persistent_bytes || _dense_cuda_limit_error(
        strategy,
        (; ndof=D, operator_bytes, slab_bytes=payload.scratch_bytes,
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
    d_tensor_fp16 = tensor_supported && tensor_format === :fp16 ?
        CUDA.CuArray{Float16,3}(undef, D, D, nclasses) :
        CUDA.CuArray{Float16,3}(undef, 0, 0, 0)
    d_tensor_bf16 = tensor_supported && tensor_format === :bf16 ?
        CUDA.CuArray{CUDABFloat16,3}(undef, D, D, nclasses) :
        CUDA.CuArray{CUDABFloat16,3}(undef, 0, 0, 0)
    d_tensor_scale = tensor_supported ? CUDA.CuArray{Float32,2}(undef, D, nclasses) :
        CUDA.CuArray{Float32,2}(undef, 0, 0)
    tensor_scale = ones(Float32, D)
    @inbounds for (i, offset) in enumerate(accepted_offsets)
        delta = TF(cell_width) * SVector{3,TF}(offset)
        r, theta, phi = cartesian_to_spherical(delta)
        build_dense_m2l_operator!(Kbuf, r, theta, phi, invariant, workspace, Val(LH))
        _check_dense_m2l_operator_finite!(Kbuf, basis_info, offset)
        copyto!(view(d_operators, :, :, i), Kbuf)
        if tensor_format === :fp16 && tensor_supported
            for col in 1:D
                # Column balancing keeps the physical inverse-distance powers
                # inside FP16; the reciprocal factor is applied to B's row.
                tensor_scale[col] = max(maximum(abs, @view Kbuf[:, col]) / 60000f0,
                    floatmin(Float32))
            end
            copyto!(view(d_tensor_fp16, :, :, i),
                Float16.(Kbuf ./ reshape(TF.(tensor_scale), 1, :)))
        end
        tensor_supported && copyto!(view(d_tensor_scale, :, i), tensor_scale)
    end
    if tensor_supported && tensor_format === :bf16
        threads = 256
        blocks = cld(length(d_tensor_bf16), threads)
        CUDA.@cuda threads=threads blocks=blocks _cuda_convert_bf16_kernel!(
            d_tensor_bf16, d_operators)
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
    plan = ResidentM2LDenseCUDAPlan{TF,typeof(d_operators),typeof(d_tensor_fp16),
        typeof(d_tensor_bf16),typeof(d_tensor_scale),typeof(route_class),
        typeof(class_counts),typeof(src_slab)}(
        route_class, d_operators, d_tensor_fp16, d_tensor_bf16, d_tensor_scale,
        class_counts, host_class_counts, class_starts,
        class_capacities, src_slab, dst_slab, nclasses, D, ndof_phi, W,
        operator_bytes, payload.scratch_bytes, route_metadata_bytes,
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

# Task 028 cycle 3: nearfield/far-field stream overlap. The nearfield kernel
# reads only refresh-final data (source_bodies, cell_ranges, direct pair lists)
# and accumulates into `output` with atomics, while B2M/M2M/M2L/L2L touch only
# the multipole/local buffers — the two are independent, so the fill+nearfield
# runs on a non-blocking side stream concurrently with the whole far-field
# chain. Ordering is device-side events only, no host syncs:
#   begin event (default stream)  -> side stream waits: the previous step's
#     finalize scatter reads `output` on the default stream, so the side
#     stream's fill must not overtake it;
#   done event (side stream)      -> default stream waits before L2B: L2B's
#     `+=` on `output` is non-atomic and must not race the nearfield atomics.
const CUDA_OVERLAP_NEARFIELD = Ref(true)
const _NEARFIELD_STREAM = Ref{Any}(nothing)
const _NEARFIELD_BEGIN = Ref{Any}(nothing)
const _NEARFIELD_DONE = Ref{Any}(nothing)

function _nearfield_overlap_handles()
    s = _NEARFIELD_STREAM[]
    if s === nothing
        s = CUDA.CuStream(; flags=CUDA.STREAM_NON_BLOCKING)
        _NEARFIELD_STREAM[] = s
        _NEARFIELD_BEGIN[] = CUDA.CuEvent(CUDA.EVENT_DISABLE_TIMING)
        _NEARFIELD_DONE[] = CUDA.CuEvent(CUDA.EVENT_DISABLE_TIMING)
    end
    return s::CUDA.CuStream, _NEARFIELD_BEGIN[]::CUDA.CuEvent,
        _NEARFIELD_DONE[]::CUDA.CuEvent
end

function _launch_cuda_nearfield_kernel!(state::DeviceResidentRadixState{TF,B,LH}) where {TF,B,LH}
    fill!(state.output, zero(TF))
    threads = 128
    hctx = state.interaction_list
    hsv = size(state.output, 1) >= 13 ? Val(true) : Val(false)
    dk = state.options.direct_kernel
    # the symmetric Newton-pair trick assumes the same-source/target scalar
    # singular kernel; every other functor takes the generic pair kernel
    symmetric = radix_setting(:CUDA_SYMMETRIC_NEARFIELD) && !LH && dk isa SingularSource &&
        hctx isa DeviceHierarchicalM2LContext
    symmetric && isempty(hctx.symmetric_targets) && throw(ArgumentError(
        "symmetric nearfield must be selected before cache construction"))
    # task 041e: target-owned fused nearfield shapes (adaptive path only; read
    # here => graph-baked at record time; automatic shipped fallback when the
    # configuration is unsupported — never throws during a resident step)
    shape = radix_setting(:CUDA_NEARFIELD_SHAPE)
    shape in NEARFIELD_SHAPES || throw(ArgumentError(
        "CUDA_NEARFIELD_SHAPE must be one of $(NEARFIELD_SHAPES); got $shape"))
    if shape !== :pairs
        actx = state.interaction_list
        if actx isa DeviceAdaptiveCUDAContext &&
                _radix_count_len(actx.u_csr_sources) > 0 &&
                actx.u_csr_built_epoch == actx.epoch_id &&
                state.counts.n_bodies >= radix_setting(:CUDA_NEARFIELD_FUSED_MIN_BODIES)
            _launch_cuda_fused_nearfield!(state, actx, shape, hsv, threads)
            return state
        end
    end
    # task 032a stage C: split vortex kernels route through the binned pair
    # stream when the cache carries the nearfield bin context (hierarchical
    # policy); flat-policy caches fall back to the unbinned functor kernel
    nfctx = _nearfield_bin_ctx(state)
    if dk isa Union{PartitionedVortex,TwoPassVortex} && nfctx !== nothing
        _launch_cuda_split_nearfield!(state, nfctx::CUDANearfieldBinContext, dk,
            hsv, threads)
        return state
    end
    # task 037f: cheapened g/h mode for the unbinned regularized-family path
    # (read here, i.e. baked into a captured graph at record time — flip only
    # before cache construction); :lut needs the construction-built table on
    # the hierarchical bin context, which this path does not carry
    ghm = radix_setting(:CUDA_NEARFIELD_GH_MODE)
    ghm in NEARFIELD_GH_MODES || throw(ArgumentError(
        "CUDA_NEARFIELD_GH_MODE must be one of $(NEARFIELD_GH_MODES); got $ghm"))
    ghm === :lut && dk isa AbstractRegularizedVortex && throw(ArgumentError(
        "CUDA_NEARFIELD_GH_MODE = :lut requires a split vortex kernel on a " *
        "hierarchical device cache (the construction-built g/h table lives " *
        "on the nearfield bin context); got $(typeof(dk)) without one"))
    ghv = Val(ghm === :lut ? :shipped : ghm)
    npairs = symmetric ? hctx.n_symmetric_pairs : state.counts.n_direct
    # warp-per-pair (task 028 lever 1): 4 warps per 128-thread block
    direct_blocks = min(cld(npairs, threads ÷ 32), radix_setting(:DIRECT_CUDA_MAX_BLOCKS))
    if direct_blocks > 0
        if symmetric
            CUDA.@cuda threads=threads blocks=direct_blocks _cuda_symmetric_pairs_output_kernel!(
                state.output, state.source_bodies, state.cell_ranges,
                hctx.symmetric_targets, hctx.symmetric_sources, npairs, hsv)
        else
            CUDA.@cuda threads=threads blocks=direct_blocks _cuda_direct_pairs_functor_kernel!(
                dk, state.output, state.source_bodies, state.cell_ranges,
                state.direct_targets, state.direct_sources, npairs, hsv, ghv,
                nothing)
        end
    end
    return state
end

# 041e: function barrier over the Any-typed adaptive U-CSR arrays; recurring
# launches are device kernels on the current stream (capture-safe, zero
# allocation, no transfers).  n_leaves is epoch-stable, so baking it into a
# captured graph is safe (the graph is recaptured on occupancy epochs).
function _launch_cuda_fused_nearfield!(state::DeviceResidentRadixState,
        actx::DeviceAdaptiveCUDAContext, shape::Symbol, hsv, threads::Int)
    dk = state.options.direct_kernel
    ghm = radix_setting(:CUDA_NEARFIELD_GH_MODE)
    ghm in NEARFIELD_GH_MODES || throw(ArgumentError(
        "CUDA_NEARFIELD_GH_MODE must be one of $(NEARFIELD_GH_MODES); got $ghm"))
    ghm === :lut && throw(ArgumentError(
        "the 041e fused nearfield shapes do not support " *
        "CUDA_NEARFIELD_GH_MODE = :lut (no shared-memory g/h table); select " *
        ":pairs or a non-lut g/h mode"))
    ghv = Val(ghm)
    nl = actx.n_leaves
    nl == 0 && return state
    offs = actx.u_csr_offsets::CUDA.CuVector{Int32}
    srcs = actx.u_csr_sources::CUDA.CuVector{Int32}
    blocks = min(nl, radix_setting(:DIRECT_CUDA_MAX_BLOCKS))
    if shape === :fused_packed
        nb = state.counts.n_bodies
        if nb > 0
            pblocks = min(cld(nb, threads), radix_setting(:DIRECT_CUDA_MAX_BLOCKS))
            CUDA.@cuda threads=threads blocks=pblocks _cuda_direct_pairs_fused_packed_kernel!(
                dk, state.output, state.source_bodies, state.cell_ranges,
                offs, srcs, actx.u_csr_body_leaf::CUDA.CuVector{Int32}, nb,
                hsv, ghv)
        end
    elseif shape === :fused_srclanes
        CUDA.@cuda threads=threads blocks=blocks _cuda_direct_pairs_fused_srclanes_kernel!(
            dk, state.output, state.source_bodies, state.cell_ranges,
            offs, srcs, nl, hsv, ghv)
    else    # :fused_cta (the shape Ref was validated by the caller)
        CUDA.@cuda threads=threads blocks=blocks _cuda_direct_pairs_fused_cta_kernel!(
            dk, state.output, state.source_bodies, state.cell_ranges,
            offs, srcs, nl, hsv, ghv)
    end
    return state
end

# Function barrier over the Any-typed bin-context arrays: every recurring
# launch below is a device kernel on the current stream (capture-safe, zero
# allocation, no transfers).
function _launch_cuda_split_nearfield!(state::DeviceResidentRadixState,
        nfctx::CUDANearfieldBinContext, dk::Union{PartitionedVortex,TwoPassVortex},
        hsv, threads::Int)
    return _launch_cuda_split_nearfield_typed!(state, dk, hsv, threads, nfctx,
        nfctx.cell_sigma_max, nfctx.cell_sigma_min, nfctx.nf_scalars,
        nfctx.bin_targets, nfctx.bin_sources, nfctx.bin_counts, nfctx.cell_coords,
        nfctx.twopass_offsets, nfctx.twopass_gap2, nfctx.gh_lut)
end

function _launch_cuda_split_nearfield_typed!(state::DeviceResidentRadixState{TF,B,LH},
        dk::Union{PartitionedVortex,TwoPassVortex}, hsv::Val, threads::Int,
        nfctx::CUDANearfieldBinContext, cell_sigma_max, cell_sigma_min, nf_scalars,
        bin_targets, bin_sources, bin_counts, cell_coords, tp_offsets,
        tp_gap2, gh_lut) where {TF,B,LH}
    npairs = state.counts.n_direct
    n_cells = state.counts.n_cells
    mode = radix_setting(:CUDA_NEARFIELD_BINNING)
    mode in (:unbinned, :classsplit, :ballot, :classsplit_ballot) ||
        throw(ArgumentError(
            "CUDA_NEARFIELD_BINNING must be :unbinned, :classsplit, :ballot, " *
            "or :classsplit_ballot; got $mode"))
    # task 037f: cheapened g/h mode (read inside the lifecycle body -> baked
    # into a captured graph at record time, like the binning Refs above)
    ghm = radix_setting(:CUDA_NEARFIELD_GH_MODE)
    ghm in NEARFIELD_GH_MODES || throw(ArgumentError(
        "CUDA_NEARFIELD_GH_MODE must be one of $(NEARFIELD_GH_MODES); got $ghm"))
    ghv = Val(ghm)
    twopass = dk isa TwoPassVortex
    classsplit = mode === :classsplit || mode === :classsplit_ballot
    h_leaf = TF(nfctx.h_leaf)
    cutoff = TF(_pass1_regularized_cutoff(dk))
    if (classsplit || twopass) && n_cells > 0
        CUDA.@cuda threads=256 blocks=cld(n_cells, 256) _cuda_cell_sigma_kernel!(
            cell_sigma_max, cell_sigma_min, state.source_bodies, state.cell_ranges,
            dk.sigma_row, n_cells)
        CUDA.@cuda threads=256 blocks=1 _cuda_nf_scalars_kernel!(
            nf_scalars, cell_sigma_max, n_cells, dk.rho_t)
    end
    direct_blocks = min(cld(npairs, threads ÷ 32), radix_setting(:DIRECT_CUDA_MAX_BLOCKS))
    if direct_blocks > 0
        if classsplit
            fill!(bin_counts, Int32(0))
            CUDA.@cuda threads=256 blocks=cld(npairs, 256) _cuda_nearfield_bin_kernel!(
                bin_targets, bin_sources, bin_counts, state.direct_targets,
                state.direct_sources, npairs, cell_coords, cell_sigma_max,
                cell_sigma_min, h_leaf, cutoff, nfctx.capacity)
            cap = nfctx.capacity
            CUDA.@cuda threads=threads blocks=direct_blocks _cuda_direct_pairs_bucket_kernel!(
                SingularVortex(), state.output, state.source_bodies,
                state.cell_ranges, bin_targets, bin_sources, bin_counts,
                Int32(1), 0, hsv, Val(:shipped), nothing)
            CUDA.@cuda threads=threads blocks=direct_blocks _cuda_direct_pairs_bucket_kernel!(
                RegularizedVortex(; sigma_row=dk.sigma_row, rho_t=dk.rho_t),
                state.output, state.source_bodies, state.cell_ranges,
                bin_targets, bin_sources, bin_counts, Int32(2), cap, hsv, ghv,
                gh_lut)
            # 037e: only the mixed bucket carries the pair-AABB fast path; the
            # Ref is read here in the lifecycle body, so — like every mechanism
            # Ref — the selection is baked into a captured graph at record time.
            # NOTE (037e+037f composition): the dedicated mixed-AABB kernel
            # always evaluates the shipped g/h — with both levers on, the mixed
            # bucket runs shipped math while the pure-regularized bucket runs
            # the selected gh_mode (both budget-passing; conservative).
            if mode === :classsplit_ballot
                CUDA.@cuda threads=threads blocks=direct_blocks _cuda_direct_pairs_queue_kernel!(
                    dk, state.output, state.source_bodies, state.cell_ranges,
                    bin_targets, bin_sources, bin_counts, Int32(3), 2 * cap, 0,
                    cell_coords, cell_sigma_max, h_leaf, nfctx.x_min,
                    hsv, Val(radix_setting(:CUDA_NEARFIELD_PAIR_AABB)), nothing, ghv, gh_lut)
            elseif radix_setting(:CUDA_NEARFIELD_PAIR_AABB)
                CUDA.@cuda threads=threads blocks=direct_blocks _cuda_direct_pairs_mixed_aabb_kernel!(
                    dk, state.output, state.source_bodies, state.cell_ranges,
                    bin_targets, bin_sources, bin_counts, Int32(3), 2 * cap,
                    cell_coords, cell_sigma_max, h_leaf, nfctx.x_min, hsv,
                    Val(true), nothing)
            else
                CUDA.@cuda threads=threads blocks=direct_blocks _cuda_direct_pairs_bucket_kernel!(
                    dk, state.output, state.source_bodies, state.cell_ranges,
                    bin_targets, bin_sources, bin_counts, Int32(3), 2 * cap, hsv,
                    ghv, gh_lut)
            end
        elseif mode === :ballot
            # whole-list stream (pure buckets included): the 037e fast path is
            # mixed-bucket-only by spec, so PAABB stays off here
            CUDA.@cuda threads=threads blocks=direct_blocks _cuda_direct_pairs_queue_kernel!(
                dk, state.output, state.source_bodies, state.cell_ranges,
                state.direct_targets, state.direct_sources, bin_counts, Int32(0),
                0, npairs, cell_coords, cell_sigma_max, h_leaf, nfctx.x_min,
                hsv, Val(false), nothing, ghv, gh_lut)
        else # :unbinned — the §6.3 negative control
            CUDA.@cuda threads=threads blocks=direct_blocks _cuda_direct_pairs_functor_kernel!(
                dk, state.output, state.source_bodies, state.cell_ranges,
                state.direct_targets, state.direct_sources, npairs, hsv, ghv,
                gh_lut)
        end
    end
    if twopass && n_cells > 0 && nfctx.twopass_K > 0
        total = n_cells * nfctx.twopass_K
        blocks2 = min(cld(total, threads ÷ 32), radix_setting(:DIRECT_CUDA_MAX_BLOCKS))
        CUDA.@cuda threads=threads blocks=blocks2 _cuda_twopass_deficit_kernel!(
            dk, state.output, state.source_bodies, state.cell_ranges, cell_coords,
            cell_sigma_max, cell_sigma_min, state.grid.cell_keys, n_cells,
            tp_offsets, tp_gap2, nfctx.twopass_K, nf_scalars, state.grid.ell,
            h_leaf, nfctx.x_min, hsv, Val(radix_setting(:CUDA_TWOPASS_PASS2_QUEUED)),
            Val(radix_setting(:CUDA_TWOPASS_TARGET_AABB_PRUNE)), Val(true), nothing)
    end
    return state
end

"""
    cuda_nearfield_homogeneity(state; stream=:all)

Diagnostic (031a §6.3 deliverable): measure the achieved warp branch
homogeneity of the split-kernel pair stream as currently ordered, by replaying
the warp-per-pair traversal with predicate ballots only. `stream` selects
`:all` (the full direct list against the pass-1 cutoff) or `:mixed` (the mixed
bucket after a fresh classification — requires a classsplit-capable cache).
Returns a NamedTuple of instant counts and fractions. Synchronizes; never part
of the recurring step.
"""
function cuda_nearfield_homogeneity(state::DeviceResidentRadixState{TF};
        stream::Symbol=:all) where TF
    dk = state.options.direct_kernel
    dk isa Union{PartitionedVortex,TwoPassVortex} || throw(ArgumentError(
        "cuda_nearfield_homogeneity requires a split direct kernel; got $(typeof(dk))"))
    nfctx = _nearfield_bin_ctx(state)
    nfctx === nothing && throw(ArgumentError(
        "cuda_nearfield_homogeneity requires the hierarchical nearfield bin context"))
    diag = nfctx.diag
    fill!(diag, UInt64(0))
    npairs = state.counts.n_direct
    n_cells = state.counts.n_cells
    threads = 128
    cutoff = TF(_pass1_regularized_cutoff(dk))
    if stream === :all
        tgts, srcs, base, n = state.direct_targets, state.direct_sources, 0, npairs
    elseif stream === :mixed
        n_cells > 0 || return _nf_homogeneity_result(diag)
        CUDA.@cuda threads=256 blocks=cld(n_cells, 256) _cuda_cell_sigma_kernel!(
            nfctx.cell_sigma_max, nfctx.cell_sigma_min, state.source_bodies,
            state.cell_ranges, dk.sigma_row, n_cells)
        fill!(nfctx.bin_counts, Int32(0))
        npairs > 0 && CUDA.@cuda threads=256 blocks=cld(npairs, 256) _cuda_nearfield_bin_kernel!(
            nfctx.bin_targets, nfctx.bin_sources, nfctx.bin_counts,
            state.direct_targets, state.direct_sources, npairs, nfctx.cell_coords,
            nfctx.cell_sigma_max, nfctx.cell_sigma_min, TF(nfctx.h_leaf), cutoff,
            nfctx.capacity)
        counts = Array(nfctx.bin_counts)
        tgts, srcs, base, n = nfctx.bin_targets, nfctx.bin_sources,
            2 * nfctx.capacity, Int(counts[3])
    else
        throw(ArgumentError("stream must be :all or :mixed; got $stream"))
    end
    if n > 0
        blocks = min(cld(n, threads ÷ 32), radix_setting(:DIRECT_CUDA_MAX_BLOCKS))
        CUDA.@cuda threads=threads blocks=blocks _cuda_nearfield_divergence_kernel!(
            diag, state.source_bodies, state.cell_ranges, tgts, srcs, base, n,
            dk.sigma_row, cutoff)
    end
    CUDA.synchronize()
    return _nf_homogeneity_result(diag)
end

function _nf_homogeneity_result(diag)
    d = Array(diag)
    instants = Int(d[1]); uniform = Int(d[2]); mixed = Int(d[4])
    lanes_valid = Int(d[5]); lanes_reg = Int(d[6])
    return (; instants, uniform, mixed,
        homogeneous_fraction=instants == 0 ? 1.0 : uniform / instants,
        lanes_valid, lanes_reg,
        regularized_fraction=lanes_valid == 0 ? 0.0 : lanes_reg / lanes_valid,
        queue_pushes=Int(d[7]), queue_drained=Int(d[8]))
end

"""
    cuda_twopass_shell_homogeneity(state)

Diagnostic: warp homogeneity of the `TwoPassVortex` pass-2 shell predicate over
the deficit sweep traversal (telemetry-only launch; output untouched).
"""
function cuda_twopass_shell_homogeneity(state::DeviceResidentRadixState{TF}) where TF
    dk = state.options.direct_kernel
    dk isa TwoPassVortex || throw(ArgumentError(
        "cuda_twopass_shell_homogeneity requires TwoPassVortex; got $(typeof(dk))"))
    nfctx = _nearfield_bin_ctx(state)
    nfctx === nothing && throw(ArgumentError(
        "cuda_twopass_shell_homogeneity requires the hierarchical nearfield bin context"))
    n_cells = state.counts.n_cells
    diag = nfctx.diag
    fill!(diag, UInt64(0))
    threads = 128
    hsv = size(state.output, 1) >= 13 ? Val(true) : Val(false)
    if n_cells > 0 && nfctx.twopass_K > 0
        CUDA.@cuda threads=256 blocks=cld(n_cells, 256) _cuda_cell_sigma_kernel!(
            nfctx.cell_sigma_max, nfctx.cell_sigma_min, state.source_bodies,
            state.cell_ranges, dk.sigma_row, n_cells)
        CUDA.@cuda threads=256 blocks=1 _cuda_nf_scalars_kernel!(
            nfctx.nf_scalars, nfctx.cell_sigma_max, n_cells, dk.rho_t)
        total = n_cells * nfctx.twopass_K
        blocks2 = min(cld(total, threads ÷ 32), radix_setting(:DIRECT_CUDA_MAX_BLOCKS))
        CUDA.@cuda threads=threads blocks=blocks2 _cuda_twopass_deficit_kernel!(
            dk, state.output, state.source_bodies, state.cell_ranges,
            nfctx.cell_coords, nfctx.cell_sigma_max, nfctx.cell_sigma_min,
            state.grid.cell_keys, n_cells, nfctx.twopass_offsets,
            nfctx.twopass_gap2, nfctx.twopass_K, nfctx.nf_scalars,
            state.grid.ell, TF(nfctx.h_leaf), nfctx.x_min, hsv, Val(false),
            Val(radix_setting(:CUDA_TWOPASS_TARGET_AABB_PRUNE)), Val(false), diag)
    end
    CUDA.synchronize()
    result = _nf_homogeneity_result(diag)
    d = Array(diag)
    return merge(result,
        (; candidate_pairs=Int(d[9]), shell_pairs=Int(d[10])))
end

"""
    cuda_nearfield_pair_aabb_stats(state)

Diagnostic (task 037e deliverable): replay a fresh mixed-bucket classification
and the pair-AABB traversal in telemetry mode (output untouched), returning
`(; mixed_pairs, tested, skipped, skipped_fraction)` where `tested` counts the
mixed cell pairs visited and `skipped` those whose every 32-lane target block
failed the exact point-vs-source-cell-AABB reachability vote (the fully
warp-skippable pairs the `CUDA_NEARFIELD_PAIR_AABB` fast path converts to pure
singular loops). Works regardless of the current flag value — the replay always
runs the AABB kernel with counters on. Synchronizes; never part of the
recurring step.
"""
function cuda_nearfield_pair_aabb_stats(state::DeviceResidentRadixState{TF}) where TF
    dk = state.options.direct_kernel
    dk isa Union{PartitionedVortex,TwoPassVortex} || throw(ArgumentError(
        "cuda_nearfield_pair_aabb_stats requires a split direct kernel; got $(typeof(dk))"))
    nfctx = _nearfield_bin_ctx(state)
    nfctx === nothing && throw(ArgumentError(
        "cuda_nearfield_pair_aabb_stats requires the hierarchical nearfield bin context"))
    diag = nfctx.diag
    fill!(diag, UInt64(0))
    npairs = state.counts.n_direct
    n_cells = state.counts.n_cells
    threads = 128
    hsv = size(state.output, 1) >= 13 ? Val(true) : Val(false)
    cutoff = TF(_pass1_regularized_cutoff(dk))
    nmixed = 0
    if n_cells > 0
        CUDA.@cuda threads=256 blocks=cld(n_cells, 256) _cuda_cell_sigma_kernel!(
            nfctx.cell_sigma_max, nfctx.cell_sigma_min, state.source_bodies,
            state.cell_ranges, dk.sigma_row, n_cells)
        fill!(nfctx.bin_counts, Int32(0))
        npairs > 0 && CUDA.@cuda threads=256 blocks=cld(npairs, 256) _cuda_nearfield_bin_kernel!(
            nfctx.bin_targets, nfctx.bin_sources, nfctx.bin_counts,
            state.direct_targets, state.direct_sources, npairs, nfctx.cell_coords,
            nfctx.cell_sigma_max, nfctx.cell_sigma_min, TF(nfctx.h_leaf), cutoff,
            nfctx.capacity)
        counts = Array(nfctx.bin_counts)
        nmixed = Int(counts[3])
        if nmixed > 0
            blocks = min(cld(nmixed, threads ÷ 32), radix_setting(:DIRECT_CUDA_MAX_BLOCKS))
            CUDA.@cuda threads=threads blocks=blocks _cuda_direct_pairs_mixed_aabb_kernel!(
                dk, state.output, state.source_bodies, state.cell_ranges,
                nfctx.bin_targets, nfctx.bin_sources, nfctx.bin_counts,
                Int32(3), 2 * nfctx.capacity, nfctx.cell_coords,
                nfctx.cell_sigma_max, TF(nfctx.h_leaf), nfctx.x_min, hsv,
                Val(false), diag)
        end
    end
    CUDA.synchronize()
    d = Array(diag)
    tested = Int(d[11]); skipped = Int(d[12])
    return (; mixed_pairs=nmixed, tested, skipped,
        skipped_fraction=tested == 0 ? 0.0 : skipped / tested)
end

# fill + nearfield on the side stream; returns the event L2B must wait on
function _launch_cuda_nearfield_async!(state::DeviceResidentRadixState)
    s, begin_ev, done_ev = _nearfield_overlap_handles()
    CUDA.record(begin_ev)          # current (default) stream
    CUDA.wait(begin_ev, s)
    CUDA.stream!(s) do
        _launch_cuda_nearfield_kernel!(state)
    end
    CUDA.record(done_ev, s)
    return done_ev
end

function _launch_cuda_resident_l2b_only!(state::DeviceResidentRadixState{TF,B,LH},
        nearfield_done) where {TF,B,LH}
    nearfield_done === nothing || CUDA.wait(nearfield_done)
    threads = 128
    P_phi = state.invariant_cache.basis_info.orders.P_phi
    P_active = state.invariant_cache.basis_info.orders.P_active
    ncell = state.counts.n_cells
    # warp-per-cell (task 028 rider): 4 warps per 128-thread block
    l2b_blocks = cld(ncell, threads ÷ 32)
    if l2b_blocks > 0
        if size(state.output, 1) >= 13
            CUDA.@cuda threads=threads blocks=l2b_blocks _cuda_l2b_output_hessian_kernel!(
                state.output, state.source_bodies, state.cell_centers, state.cell_ranges,
                state.grid.leaf_to_node, state.locals.phi, state.locals.chi,
                P_phi, P_active, Val(LH), ncell,
            )
        else
            CUDA.@cuda threads=threads blocks=l2b_blocks _cuda_l2b_output_kernel!(
                state.output, state.source_bodies, state.cell_centers, state.cell_ranges,
                state.grid.leaf_to_node, state.locals.phi, state.locals.chi,
                P_phi, P_active, Val(LH), ncell,
            )
        end
    end
    return state
end

# standalone (non-overlapped) fused stage: benchmarks time this directly, and
# the pipeline uses it whenever CUDA_OVERLAP_NEARFIELD is off
function _launch_cuda_resident_l2b!(state::DeviceResidentRadixState)
    _launch_cuda_nearfield_kernel!(state)
    return _launch_cuda_resident_l2b_only!(state, nothing)
end

function _launch_cuda_resident_operator_pipeline!(state::DeviceResidentRadixState;
        nearfield_done=nothing)
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
    if nearfield_done === nothing
        _launch_cuda_resident_l2b!(state)
    else
        # cycle 3 overlap: the nearfield already ran on the side stream
        _launch_cuda_resident_l2b_only!(state, nearfield_done)
    end
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
    # task 029 cycle 1: replay the captured far-field graph when it is valid
    # for the current occupancy epoch; otherwise run (and possibly record) the
    # launch-sequence body
    _cuda_graph_eligible(state) && return _run_cuda_radix_lifecycle_graph!(state)
    return _cuda_lifecycle_body!(state)
end

# The complete lifecycle launch sequence (unchanged pre-029 semantics). All
# host work in here is launch bookkeeping and residency assertions — no device
# synchronization, no D2H, and (with the staged GEMM scalars and the cached
# M2L windows) no device allocation — so the same body serves direct execution
# and stream capture.
function _cuda_lifecycle_body!(state::DeviceResidentRadixState)
    # cycle 3: launch fill+nearfield on the side stream before B2M so it runs
    # concurrently with the whole far-field chain; L2B joins on the event
    nearfield_done = radix_setting(:CUDA_OVERLAP_NEARFIELD) ?
        _launch_cuda_nearfield_async!(state) : nothing
    _launch_cuda_b2m!(state)
    _assert_cuda_resident_stage!(state, :b2m)
    _launch_cuda_resident_operator_pipeline!(state; nearfield_done)
    return state
end

# Graph capture is restricted to configurations whose lifecycle body is
# sync-free and capacity-static within the occupancy epoch: hierarchical dense
# fused M2L with a valid window cache (per-level applies only), no per-step
# symmetric-pair compaction (its pair count varies with per-cell body counts),
# and no stage profiling (which synchronizes between levels).
function _cuda_graph_eligible(state::DeviceResidentRadixState)
    radix_setting(:CUDA_GRAPH_LIFECYCLE) && radix_setting(:CUDA_CACHED_WINDOWS) || return false
    hctx = state.interaction_list
    hctx isa DeviceHierarchicalM2LContext || return false
    # typemin sentinel: a previous capture attempt hit a capture-illegal
    # operation (CUDA error 900); this context stays on the launch-sequence
    # path permanently rather than throwing once per step
    hctx.graph_warm_epoch == typemin(Int) && return false
    hctx.win_valid && _cuda_windows_cacheable(hctx) || return false
    hctx.profile_stages && return false
    isempty(hctx.symmetric_targets) || return false
    DEBUG[] && return false
    return true
end

function _run_cuda_radix_lifecycle_graph!(state::DeviceResidentRadixState)
    hctx = state.interaction_list::DeviceHierarchicalM2LContext
    exec = hctx.graph_exec
    if exec !== nothing && hctx.graph_epoch == hctx.epoch_id
        CUDA.launch(exec::CUDA.CuGraphExec)
        return state
    end
    if hctx.graph_warm_epoch != hctx.epoch_id
        # first lifecycle of a new epoch: run uncaptured so kernel JIT, CUBLAS
        # handle/workspace setup, and the staged-scalar cache are warm before
        # recording (capture tolerates none of them)
        _cuda_lifecycle_body!(state)
        hctx.graph_warm_epoch = hctx.epoch_id
        return state
    end
    graph = try
        CUDA.capture(; throw_error=false) do
            _cuda_lifecycle_body!(state)
        end
    catch err
        # `throw_error=false` only tolerates capture invalidation; a
        # capture-illegal API inside the body throws eagerly (e.g. CUDA error
        # 900). Disable graphing for this context and fall back for good — the
        # launch-sequence path is always correct.
        err isa CUDA.CuError || rethrow()
        hctx.graph_warm_epoch = typemin(Int)
        nothing
    end
    if graph === nothing
        # capture failed (e.g. a residual allocation); execute normally and
        # leave the warm marker so the next step retries the recording
        _cuda_lifecycle_body!(state)
        return state
    end
    hctx.graph_exec = CUDA.instantiate(graph)
    hctx.graph_epoch = hctx.epoch_id
    # stream capture records without executing, so this step still runs
    CUDA.launch(hctx.graph_exec::CUDA.CuGraphExec)
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

function buffer_to_target!(target_system, device_output_buffer::CUDA.AnyCuArray,
        derivatives_switch, sort_index=1:get_n_bodies(target_system))
    throw(ArgumentError(
        "DeviceResident CUDA target systems must overload FastMultipole.buffer_to_target!(target_system, device_output_buffer, derivatives_switch, sort_index)",
    ))
end

# Per-system cached device scatter buffer for the recurring finalize (task 028
# rider): CUDA.zeros here was a fresh pool allocation plus memset every step,
# and the scatter copy zero-fills the buffer again anyway. Capacity contract
# (052 long-run leak, job 13508681): a shedding run changes `nb` every step,
# and an exact-size cache then reallocates every step — the replaced device
# buffer survives a full step before dying, gets promoted, and no major GC
# ever runs because device bytes are invisible to the host GC heuristics, so
# ~rows*nb*8 bytes of dead pool blocks accumulate per step. The cache instead
# holds a grow-only capacity buffer (geometric headroom) and serves the live
# `nb` as a contiguous column-prefix view.
function _cuda_cached_target_buffer(cache, isys::Integer, ::Type{TF},
        rows::Integer, nb::Integer) where TF
    cache === nothing && return CUDA.CuArray{TF}(undef, rows, nb)
    buf = get(cache, isys, nothing)
    if !(buf isa CUDA.CuArray{TF,2}) || size(buf, 1) != rows || size(buf, 2) < nb
        cap = buf isa CUDA.CuArray{TF,2} && size(buf, 1) == rows ?
            max(nb, size(buf, 2) + cld(size(buf, 2), 4)) : nb
        buf = CUDA.CuArray{TF}(undef, rows, cap)
        cache[isys] = buf
    end
    buf = buf::CUDA.CuArray{TF,2}
    return size(buf, 2) == nb ? buf : view(buf, :, 1:nb)
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

#------- SFS (subfilter-scale vortex stretching) device pass (task 048) -------#
#
# CUDA mirror of the host SFS pass in translate_batched_resident.jl (see the
# math/comment block there): (a) thread-per-body TG precompute T = op(J)Γ +
# accumulator zeroing, (b) warp-per-pair ζ sweep over the FULL direct pair
# list (clone of `_cuda_direct_pairs_functor_kernel!`'s loop skeleton; self
# pair i == j skipped, matching the host mirror). The pair pass is launched
# only for an evaluation that requests `sfs=true`, after the U/J lifecycle
# (and any U/J graph replay) has completed. It uses persistent buffers and has
# no allocation, synchronization, or D2H. E-formation + scatter happen in
# `finalize_cuda_radix_sfs_output!`, mirroring `finalize_cuda_radix_output!`.
# The transposed-scheme flag is baked at construction (Val at launch).

function _cuda_sfs_tg_kernel!(tg, om, q, output, source_bodies,
        ::Val{TRANSPOSED}, n_bodies) where TRANSPOSED
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > n_bodies && return nothing
    T = eltype(tg)
    @inbounds begin
        g1 = source_bodies[5, i]
        g2 = source_bodies[6, i]
        g3 = source_bodies[7, i]
        if TRANSPOSED
            tg[1, i] = output[5, i] * g1 + output[6, i] * g2 + output[7, i] * g3
            tg[2, i] = output[8, i] * g1 + output[9, i] * g2 + output[10, i] * g3
            tg[3, i] = output[11, i] * g1 + output[12, i] * g2 + output[13, i] * g3
        else
            tg[1, i] = output[5, i] * g1 + output[8, i] * g2 + output[11, i] * g3
            tg[2, i] = output[6, i] * g1 + output[9, i] * g2 + output[12, i] * g3
            tg[3, i] = output[7, i] * g1 + output[10, i] * g2 + output[13, i] * g3
        end
        om[1, i] = zero(T); om[2, i] = zero(T); om[3, i] = zero(T)
        q[1, i] = zero(T); q[2, i] = zero(T); q[3, i] = zero(T)
    end
    return nothing
end

function _cuda_sfs_zeta_pairs_kernel!(om, q, tg, source_bodies, cell_ranges,
        direct_targets, direct_sources, npairs, rc2, K1, active_row)
    T = eltype(om)
    half = T(0.5)
    lane = (threadIdx().x - Int32(1)) % Int32(32)
    warps_per_block = blockDim().x ÷ Int32(32)
    warp_in_block = (threadIdx().x - Int32(1)) ÷ Int32(32)
    pair_i = (blockIdx().x - 1) * warps_per_block + warp_in_block + 1
    warp_stride = gridDim().x * warps_per_block
    @inbounds while pair_i <= npairs
        target_cell = direct_targets[pair_i]
        source_cell = direct_sources[pair_i]
        tfirst = cell_ranges[1, target_cell]
        tlast = tfirst + cell_ranges[2, target_cell] - 1
        sfirst = cell_ranges[1, source_cell]
        slast = sfirst + cell_ranges[2, source_cell] - 1
        i = tfirst + lane
        while i <= tlast
            if active_row != 0 && iszero(source_bodies[active_row, i])
                i += 32
                continue
            end
            xi = source_bodies[1, i]
            yi = source_bodies[2, i]
            zi = source_bodies[3, i]
            o1 = zero(T); o2 = zero(T); o3 = zero(T)
            q1 = zero(T); q2 = zero(T); q3 = zero(T)
            for j in sfirst:slast
                i == j && continue
                active_row != 0 && iszero(source_bodies[active_row, j]) && continue
                dx = xi - source_bodies[1, j]
                dy = yi - source_bodies[2, j]
                dz = zi - source_bodies[3, j]
                r2 = dx * dx + dy * dy + dz * dz
                sigma = source_bodies[8, j]
                rho2 = r2 / (sigma * sigma)
                if rho2 <= rc2
                    z = K1 * exp(-half * rho2) / (sigma * sigma * sigma)
                    o1 += z * source_bodies[5, j]
                    o2 += z * source_bodies[6, j]
                    o3 += z * source_bodies[7, j]
                    q1 += z * tg[1, j]
                    q2 += z * tg[2, j]
                    q3 += z * tg[3, j]
                end
            end
            CUDA.@atomic om[1, i] += o1
            CUDA.@atomic om[2, i] += o2
            CUDA.@atomic om[3, i] += o3
            CUDA.@atomic q[1, i] += q1
            CUDA.@atomic q[2, i] += q2
            CUDA.@atomic q[3, i] += q3
            i += 32
        end
        pair_i += warp_stride
    end
    return nothing
end

# Per-evaluation launcher: the caller invokes this only for `sfs=true`.
function _launch_cuda_sfs!(state::DeviceResidentRadixState{TF}) where TF
    sfs = state.sfs
    sfs === nothing && return state
    size(state.output, 1) >= 13 || throw(AssertionError(
        "the SFS pass requires the 13-row (hessian) output"))
    _launch_cuda_sfs_typed!(state, sfs.tg, sfs.om, sfs.q,
        sfs.transposed ? Val(true) : Val(false), sfs.active_row)
    return state
end

# function barrier over the Any-typed sfs NamedTuple
function _launch_cuda_sfs_typed!(state::DeviceResidentRadixState{TF},
        tg::CUDA.CuMatrix{TF}, om::CUDA.CuMatrix{TF}, q::CUDA.CuMatrix{TF},
        tv::Val, active_row::Int) where TF
    threads = 128
    n = state.counts.n_bodies
    tg_blocks = cld(n, threads)
    tg_blocks > 0 || return state
    CUDA.@cuda threads=threads blocks=tg_blocks _cuda_sfs_tg_kernel!(
        tg, om, q, state.output, state.source_bodies, tv, n)
    npairs = state.counts.n_direct
    # warp-per-pair, grid-stride (the `_cuda_direct_pairs_functor_kernel!`
    # launch shape)
    pair_blocks = min(cld(npairs, threads ÷ 32), radix_setting(:DIRECT_CUDA_MAX_BLOCKS))
    if pair_blocks > 0
        CUDA.@cuda threads=threads blocks=pair_blocks _cuda_sfs_zeta_pairs_kernel!(
            om, q, tg, state.source_bodies, state.cell_ranges,
            state.direct_targets, state.direct_sources, npairs,
            _sfs_saturation_rc2(TF), TF(_SFS_ZETA_K1), active_row)
    end
    return state
end

# E-formation into `tg` (dead after the pair sweep), sorted body order
function _cuda_sfs_form_e_kernel!(tg, om, q, output, ::Val{TRANSPOSED},
        n_bodies) where TRANSPOSED
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > n_bodies && return nothing
    @inbounds begin
        o1 = om[1, i]; o2 = om[2, i]; o3 = om[3, i]
        if TRANSPOSED
            e1 = output[5, i] * o1 + output[6, i] * o2 + output[7, i] * o3
            e2 = output[8, i] * o1 + output[9, i] * o2 + output[10, i] * o3
            e3 = output[11, i] * o1 + output[12, i] * o2 + output[13, i] * o3
        else
            e1 = output[5, i] * o1 + output[8, i] * o2 + output[11, i] * o3
            e2 = output[6, i] * o1 + output[9, i] * o2 + output[12, i] * o3
            e3 = output[7, i] * o1 + output[10, i] * o2 + output[13, i] * o3
        end
        tg[1, i] = e1 - q[1, i]
        tg[2, i] = e2 - q[2, i]
        tg[3, i] = e3 - q[3, i]
    end
    return nothing
end

# sorted -> global permute of the 3-row E slab into a per-system device buffer
# (clone of `_cuda_scatter_output_to_target_buffer_kernel!`)
function _cuda_sfs_scatter_kernel!(target_buffer, e, perm, body_system,
        body_index, isys, n_bodies)
    sorted_i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    sorted_i > n_bodies && return nothing
    global_i = perm[sorted_i]
    body_system[global_i] == isys || return nothing
    ibody = body_index[global_i]
    @inbounds begin
        target_buffer[1, ibody] = e[1, sorted_i]
        target_buffer[2, ibody] = e[2, sorted_i]
        target_buffer[3, ibody] = e[3, sorted_i]
    end
    return nothing
end

"""
    finalize_cuda_radix_sfs_output!(state, target_systems;
        host_sfs_staging=nothing, sfs_target_buffers=nothing,
        device_sfs_buffers=nothing)

Form E = op(J)Ω − Q from the device SFS accumulators, de-permute
sorted -> global, and deliver a per-system `3 x n_bodies` global-order buffer
(device buffer for `DeviceResident` targets, host buffer otherwise) through
[`sfs_to_target!`](@ref). Called OUTSIDE the captured lifecycle graph, next to
[`finalize_cuda_radix_output!`](@ref). Pass the preallocated stagings/caches
(the recurring `_radix_cache_device_step!` path does) to keep steps
allocation-free.
"""
function finalize_cuda_radix_sfs_output!(state::DeviceResidentRadixState{TF},
        target_systems; host_sfs_staging=nothing, sfs_target_buffers=nothing,
        device_sfs_buffers=nothing) where TF
    _require_cuda_radix_available()
    sfs = state.sfs
    sfs === nothing && throw(ArgumentError(
        "sfs=true evaluation requires a RadixFMMCache built with sfs=true"))
    systems = to_tuple(target_systems)
    n = state.counts.n_bodies
    threads = 128
    blocks = cld(n, threads)
    blocks > 0 || return target_systems
    tv = sfs.transposed ? Val(true) : Val(false)
    CUDA.@cuda threads=threads blocks=blocks _cuda_sfs_form_e_kernel!(
        sfs.tg, sfs.om, sfs.q, state.output, tv, n)
    host_e = nothing
    for (isys, target_system) in enumerate(systems)
        nb = get_n_bodies(target_system)
        if residency(target_system) isa DeviceResident
            buf = _cuda_cached_target_buffer(device_sfs_buffers, isys, TF, 3, nb)
            fill!(buf, zero(TF))
            CUDA.@cuda threads=threads blocks=blocks _cuda_sfs_scatter_kernel!(
                buf, sfs.tg, state.body_perm, state.body_system_ids,
                state.body_indices, isys, n)
            sfs_to_target!(target_system, buf, 1:nb)
        else
            if host_e === nothing
                if host_sfs_staging === nothing
                    host_e = Array(sfs.tg)
                else
                    copyto!(host_sfs_staging, 1, sfs.tg, 1, 3 * n)
                    host_e = host_sfs_staging
                end
                state.counters.influence_downloads += 1
            end
            buf_full = sfs_target_buffers === nothing ?
                Matrix{TF}(undef, 3, nb) : sfs_target_buffers[isys]
            buf = size(buf_full, 2) == nb ? buf_full : view(buf_full, :, 1:nb)
            _scatter_sfs_host!(buf, host_e, state.host_body_perm,
                state.host_body_system_ids, state.host_body_indices, isys, n)
            sfs_to_target!(target_system, buf, 1:nb)
        end
    end
    return target_systems
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
        grid::DeviceRadixGrid, level_offsets::Vector{Int}, ell::Int,
        first_level::Int=0)
    threads = 128
    length(ws.m2m_groups) == ell - first_level ||
        throw(ArgumentError("resident cache workspace does not match the trimmed level range"))
    for (gi, parent_level) in enumerate((ell - 1):-1:first_level)
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
    for (gi, child_level) in enumerate((first_level + 1):ell)
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

# Occupancy-epoch change detection (task 029 cycle 1): benign-race flag store —
# any lane observing a difference sets the flag, order irrelevant.
function _cuda_keys_differ_kernel!(flag, keys, snapshot, n)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > n && return nothing
    @inbounds keys[i] != snapshot[i] && (flag[1] = Int32(1))
    return nothing
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
        hierarchical_level_class_of::Array{Int32,3}=Array{Int32}(undef, 0, 0, 0),
        hierarchical_level_radii2::Vector{Int}=Int[],
        max_level_nodes::Int=0, hessian::Bool=false,
        # task 048: SFS device pass — persistent 3 x capacity accumulators plus
        # the FLOWVPM transposed-scheme flag and optional packed active-mask row
        sfs::Bool=false, sfs_transposed::Bool=true, sfs_active_row::Int=0,
        # rectangular geometry contract (task 037 stage 2): per-axis leaf depths
        # and physical extents; cubic callers keep the virtual-cube defaults
        ell_axes::SVector{3,Int}=SVector(ell, ell, ell),
        box_extent::SVector{3,TF}=SVector{3,TF}(2 * h0, 2 * h0, 2 * h0),
        # active-level trimming (task 037 stage 3): node levels root_level:ell,
        # M2L levels first_m2l_level:ell; flat-policy callers keep 0/2
        root_level::Int=0, first_m2l_level::Int=2,
        # task 041: opt-in adaptive octree policy (device-resident mirror)
        adaptive_policy=nothing, dpb_adaptive::Int=0) where {TF,B,LH}
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
        chunk = max(min(radix_setting(:DENSE_CUDA_CHUNK), route_capacity), 1)
        body_cols = sum(data_per_body, sources)
        dense_cuda_footprint = _dense_cuda_lifecycle_footprint(TF, basis_info,
            length(plan_offsets), route_capacity, direct_capacity, maxn, max_cells,
            max_nodes, hierarchical ? 0 : ell, chunk, body_cols;
            hierarchical_occupancy_words=hierarchical ?
                _cuda_hier_occupancy_words(stencil_policy, ell) : 0,
            hierarchical_window_words=hierarchical ?
                2 * max(min(stencil_policy.window_classes,
                    length(hierarchical_tables.push_offsets)) * max_level_nodes, 1) : 0,
            hierarchical_levels=hierarchical ? max(ell - first_m2l_level + 1, 0) : 0)
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
            dense_cuda_footprint.estimated_peak_bytes,
        ell_axes, first_level=root_level)
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

    # per-system source staging: host-resident systems get a pinned host buffer
    # plus a persistent device buffer (one upload per step); device-resident
    # systems get a persistent device buffer their source_to_buffer! overload
    # fills in place each step (task 032 gap-5 fix — no per-step allocation,
    # no transfer)
    host_stagings = Tuple(
        residency(system) isa HostResident ?
            _pin_host_array(Matrix{TF}(undef, data_per_body(system), maxn)) : nothing
        for system in sources)
    device_sources = Tuple(
        CUDA.zeros(TF, data_per_body(system), maxn) for system in sources)

    # Flat occupancy/flag storage is leaf-only and unused by the hierarchical
    # path, which keeps its per-level `node_at` and its single-window flag/prefix
    # pair on the device hierarchical context instead.
    G = hierarchical ? 0 : 1 << ell
    class_chunk = hierarchical ? 1 : max(min(length(accepted), max_cells), 1)
    flag_capacity = hierarchical ? 0 : class_chunk * max_cells
    hierarchical_ctx = hierarchical ?
        _build_cuda_hierarchical_context(TF, basis_info, stencil_policy,
            hierarchical_tables, class_level, class_offset, accepted,
            hierarchical_level_class_of, hierarchical_level_radii2,
            workspace.m2l_concat, ell, first_m2l_level, max_level_nodes,
            direct_capacity, counters) :
        nothing
    # canonical all-rows packed layout + construction-chosen output rows (032)
    dpb = maximum(data_per_body(system) for system in sources)
    n_output_rows = hessian ? 13 : 4
    # task 048: SFS device accumulators (persistent, default-stream only, so no
    # cross-stream synchronization concerns; `nothing` disables the pass and
    # its launches entirely)
    sfs_device_ctx = sfs ?
        (; tg=CUDA.zeros(TF, 3, maxn), om=CUDA.zeros(TF, 3, maxn),
           q=CUDA.zeros(TF, 3, maxn), transposed=sfs_transposed,
           active_row=sfs_active_row) : nothing
    ctx = (;
        multipoles, locals, workspace, invariant, counters, grid,
        counts=RadixStepCounts(0, 0, 0, 0, 0),
        source_bodies=CUDA.zeros(TF, dpb, maxn),
        output=CUDA.zeros(TF, n_output_rows, maxn),
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
        counting_histogram=CUDA.zeros(Int32,
            _cuda_counting_sort_enabled(ell) ? 1 << (3ell) : 1),
        counting_prefix=CUDA.zeros(Int32,
            _cuda_counting_sort_enabled(ell) ? 1 << (3ell) : 1),
        counting_cursor=CUDA.zeros(Int32,
            _cuda_counting_sort_enabled(ell) ? 1 << (3ell) : 1),
        body_flags=CUDA.zeros(Int, maxn),
        body_prefix=CUDA.zeros(Int, maxn),
        # 032a stage C mechanism (a): per-sorted-position sub-Morton keys
        subsort_keys=CUDA.zeros(UInt32, maxn),
        cell_coords=CUDA.zeros(Int, 3, max_cells),
        level_keys=CUDA.zeros(UInt64, max_cells, ell + 1),
        level_flags=CUDA.zeros(Int, max_cells, ell + 1),
        level_prefix=CUDA.zeros(Int, max_cells, ell + 1),
        level_counts=CUDA.zeros(Int, ell + 1),
        d_level_offsets=CUDA.zeros(Int, ell + 2),
        oob_flag=CUDA.zeros(Int32, 1),
        # occupancy-epoch snapshot (task 029 cycle 1): hierarchical caches
        # compare the sorted unique leaf keys against the previous step to skip
        # node-metadata/window/direct-pair regeneration; flat caches keep the
        # per-step rebuild (zero-length snapshot disables the check)
        epoch_cell_keys=CUDA.zeros(UInt64, hierarchical ? max_cells : 0),
        epoch_flag=CUDA.zeros(Int32, 1),
        host_epoch_flag=_pin_host_array(zeros(Int32, 1)),
        epoch_prev_n=Ref(0),
        epoch_prev_n_cells=Ref(0),
        epoch_have=Ref(false),
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
        # must track the output row count, or the prefix copyto! in
        # finalize_cuda_radix_output! silently mis-strides
        host_output=_pin_host_array(zeros(TF, n_output_rows, maxn)),
        # per-system device scatter buffers for the recurring finalize (028 rider)
        device_target_buffers=Dict{Int,Any}(),
        # task 048: SFS pass context + finalize staging (separate scatter-buffer
        # dict — SFS buffers are 3-row, UJ buffers 4/13-row, and the cache dict
        # keys on isys only)
        sfs_ctx=sfs_device_ctx,
        host_sfs_staging=sfs ? _pin_host_array(zeros(TF, 3, maxn)) : nothing,
        device_sfs_buffers=Dict{Int,Any}(),
    )
    # 029 cycle 1: the nearfield side stream's ordering against the main stream
    # is enforced by the cycle-3 begin/done events. CUDACore's per-array managed
    # memory would ADDITIONALLY host-synchronize the previous owner stream on
    # every cross-stream access — redundant given the events, a hidden per-step
    # blocking sync, and illegal inside stream capture (CUDA error 900, observed
    # job 13060540) — so implicit synchronization is disabled for exactly the
    # arrays the side-stream fill+nearfield touches. Every ordering these
    # arrays need is event- or stream-ordered by construction (see the
    # CUDA_OVERLAP_NEARFIELD comment block).
    for arr in (ctx.output, ctx.source_bodies, grid.cell_ranges,
            ctx.direct_targets, ctx.direct_sources)
        CUDA.enable_synchronization!(arr, false)
    end
    if hierarchical_ctx !== nothing && !isempty(hierarchical_ctx.symmetric_targets)
        CUDA.enable_synchronization!(hierarchical_ctx.symmetric_targets, false)
        CUDA.enable_synchronization!(hierarchical_ctx.symmetric_sources, false)
    end
    # 032a stage C: split vortex kernels get the binned-pair-stream scratch on
    # the hierarchical context (flat-policy caches fall back to unbinned; a
    # flat-policy TwoPassVortex cache was already refused at construction)
    if hierarchical_ctx !== nothing &&
            options.direct_kernel isa Union{PartitionedVortex,TwoPassVortex}
        hierarchical_ctx.nearfield = _build_cuda_nearfield_bin_context(TF,
            options.direct_kernel, stencil_policy, accepted, ell, h0, x_min,
            max_cells, direct_capacity, ctx, counters)
    end
    # task 041: opt-in device-resident adaptive octree. cache.adaptive_tree
    # holds the DeviceAdaptiveCUDAContext (device tree + lists), cache.
    # adaptive_state the adaptive DeviceResidentRadixState; adaptive_lists is
    # unused on the device path (lists live in the context).
    if adaptive_policy === nothing
        adaptive_actx = nothing
        adaptive_state = nothing
    else
        adaptive_actx, adaptive_state = _cuda_allocate_adaptive_lifecycle(TF,
            basis_info, options, adaptive_policy::AdaptiveTreePolicy, x_min, h0,
            maxn, dpb_adaptive > 0 ? dpb_adaptive : dpb, hessian,
            ctx.invariant, counters, ctx; sfs, sfs_transposed, sfs_active_row)
    end
    cache = RadixFMMCache{TF,LH}(
        P, ell, x_min, h0, ell_axes, box_extent, root_level, maxn, true, hessian,
        options, stencil_policy,
        accepted, rejected, max_cells, max_nodes, route_capacity, direct_capacity,
        nothing, zeros(Int32, 0, 0, 0), SVector{3,Int}[], zeros(Int, ell + 2),
        UInt64[], Int[], Int[], Int[], nothing, nothing, ctx,
        length(sources), false, 0,
        adaptive_policy, adaptive_actx, nothing, adaptive_state,
        snapshot_locked_radix_settings(),
        sfs, sfs_transposed, nothing,
    )
    update_cuda_radix_state!(cache, sources)
    return cache
end

# Refresh the persistent per-system device source buffers. Host-resident systems
# repack into their pinned staging and upload the valid column prefix (one upload
# per system per step); device-resident systems fill the valid prefix of their
# persistent buffer in place through their source_to_buffer! overload (no
# transfer, no allocation — task 032 gap-5 fix).
function _radix_cache_refresh_source_buffers!(ctx, systems::Tuple, ::Type{TF}) where TF
    return ntuple(length(systems)) do isys
        system = systems[isys]
        n_sys = get_n_bodies(system)
        device_buffer = ctx.device_sources[isys]
        if residency(system) isa HostResident
            staging = ctx.host_stagings[isys]
            source_to_buffer!(staging, system, 1:n_sys)
            # linear-prefix copy: the first n_sys columns are contiguous
            copyto!(device_buffer, 1, staging, 1, size(staging, 1) * n_sys)
            ctx.counters.body_uploads += 1
        else
            _fill_device_source_buffer!(view(device_buffer, :, 1:n_sys), system)
        end
        view(device_buffer, :, 1:n_sys)
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
        view(ctx.keys, 1:n), ctx.oob_flag, ctx.positions, x_min, cache.box_extent,
        h0, ell,
    )
    copyto!(ctx.host_oob, ctx.oob_flag)
    if ctx.host_oob[1] != 0
        x_max = x_min .+ cache.box_extent
        throw(ArgumentError(
            "at least one body lies outside the fixed RadixFMMCache box " *
            "[$(Tuple(x_min)), $(Tuple(x_max))]; the box is part of the cache's " *
            "invariant contract — construct a new cache (or pass explicit " *
            "bounds=(x_min, box_size) covering the trajectory)"))
    end

    # Sort bodies by key. The comparison backend is the same as the one-shot
    # builder's, so equal-key (same-cell) bodies keep their deterministic global
    # order; the bounded counting sort claims its slots with an atomic cursor and
    # is therefore *unstable* — same-cell ordering, and hence the summation order
    # of same-cell atomics, varies between otherwise identical runs.
    pv = view(grid.perm, 1:n)
    kv = view(ctx.keys, 1:n)
    sk = view(ctx.sorted_keys, 1:n)
    if _cuda_counting_sort_ready(ctx, ell)
        _cuda_counting_sort_into!(pv, sk, kv, ctx.counting_histogram,
            ctx.counting_prefix, ctx.counting_cursor, threads)
    else
        _cuda_sortperm_into!(pv, kv)
        CUDA.@cuda threads=threads blocks=blocks _cuda_gather_sorted_keys_kernel!(
            sk, ctx.keys, pv)
    end
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

    # Occupancy-epoch check (task 029 cycle 1): every array below this point —
    # cell centers, per-level unique node keys, node geometry/parent/child
    # topology, leaf_to_node — is a pure function of the occupied leaf-cell SET
    # inside the cache's fixed Morton box (cell_ranges/perm above are NOT and
    # always refresh). When the sorted unique keys match the previous step's
    # snapshot exactly, the whole node-metadata rebuild is skipped and the
    # persistent arrays remain valid. The compare costs one kernel plus one
    # pinned 4-byte D2H, replacing ~40 launches, several device scans, and two
    # blocking downloads on the steady occupancy-static step.
    track_epoch = length(ctx.epoch_cell_keys) > 0 && radix_setting(:CUDA_CACHED_WINDOWS)
    occ_changed = true
    if track_epoch && ctx.epoch_have[] && ctx.epoch_prev_n[] == n &&
            ctx.epoch_prev_n_cells[] == n_cells
        fill!(ctx.epoch_flag, Int32(0))
        CUDA.@cuda threads=threads blocks=blocks_cells _cuda_keys_differ_kernel!(
            ctx.epoch_flag, ckv, ctx.epoch_cell_keys, n_cells,
        )
        copyto!(ctx.host_epoch_flag, ctx.epoch_flag)
        occ_changed = ctx.host_epoch_flag[1] != Int32(0)
    end
    if !occ_changed
        grid.n_bodies = n
        grid.n_cells = n_cells
        return n_cells, false
    end
    if track_epoch
        copyto!(ctx.epoch_cell_keys, 1, grid.cell_keys, 1, n_cells)
        ctx.epoch_prev_n[] = n
        ctx.epoch_prev_n_cells[] = n_cells
        ctx.epoch_have[] = true
    end
    CUDA.@cuda threads=threads blocks=blocks_cells _cuda_cell_centers_kernel!(
        grid.cell_centers, ctx.cell_coords, ckv, x_min, h0, ell,
    )

    # per-level unique ancestors: cell_keys is ascending and a right shift is
    # monotone, so each level's ancestor keys are already sorted. Levels below
    # the cache root are trimmed (task 037 stage 3): never keyed, never built.
    first_level = cache.root_level
    for level in first_level:ell
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
    for level in 0:(first_level - 1)
        # trimmed levels: the gathered counts read unfilled prefix columns
        ctx.host_level_counts[level + 1] = 0
        level_offsets[level + 2] = 0
    end
    for level in first_level:ell
        level_offsets[level + 2] = level_offsets[level + 1] + ctx.host_level_counts[level + 1]
    end
    n_nodes = level_offsets[end]
    n_nodes <= cache.max_nodes ||
        throw(AssertionError("device radix grid exceeded the cache node capacity"))
    for level in first_level:ell
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
        n_levels = ell - first_level + 1
        CUDA.@cuda threads=threads blocks=(blocks_x, n_levels) _cuda_fill_node_geometry_kernel!(
            grid.node_levels, grid.node_coords, grid.node_centers, grid.node_keys,
            ctx.d_level_offsets, x_min, h0, ell, first_level,
        )
        CUDA.@cuda threads=threads blocks=(blocks_x, n_levels) _cuda_parent_index_kernel!(
            grid.parent_index, grid.node_keys, ctx.d_level_offsets, ell, first_level,
        )
        CUDA.@cuda threads=threads blocks=(blocks_x, n_levels) _cuda_child_ranges_kernel!(
            grid.child_ranges, grid.node_keys, ctx.d_level_offsets, ell, first_level,
        )
    end
    CUDA.@cuda threads=threads blocks=blocks_cells _cuda_fill_leaf_to_node_kernel!(
        view(grid.leaf_to_node, 1:n_cells), level_offsets[ell + 1],
    )
    grid.n_bodies = n
    grid.n_cells = n_cells
    return n_cells, true
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

Task 029 cycle 1: on hierarchical caches with `radix_setting(:CUDA_CACHED_WINDOWS)` (the
default), the node metadata, occupancy lookup, direct pairs, operator-group
edges, and the cached M2L route windows are regenerated only when the occupied
leaf-cell set changed since the previous step (they are pure functions of that
set inside the fixed box); the occupancy-static step replaces all of that work
with one key-compare kernel and a pinned 4-byte flag download. Window-cache
growth (and CUDA-graph re-recording downstream) therefore recurs exactly with
occupancy change.
"""
function update_cuda_radix_state!(cache::RadixFMMCache{TF,LH}, systems::Tuple) where {TF,LH}
    _require_cuda_radix_available()
    ctx = cache.device_ctx
    ctx === nothing &&
        throw(ArgumentError("update_cuda_radix_state! requires a cache built with device=true"))
    # task 041: with the adaptive policy armed, the adaptive refresh REPLACES
    # the uniform grid/route refresh (no double refresh on device)
    cache.adaptive === nothing ||
        return _cuda_update_adaptive_radix_state!(cache, systems)
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
    n_cells, occ_changed = _cuda_update_radix_grid_in_place!(ctx, cache, n)
    n_nodes = cache.level_offsets[end]
    if profiling
        CUDA.synchronize()
        hctx.update_stage_ns[1] = time_ns() - t_stage
    end

    # 032a stage C mechanism (a): optional within-cell sub-Morton ordering,
    # composed into the perm before packing (device kernels only; the sorted
    # cell keys, cell ranges, and node metadata are unaffected)
    if radix_setting(:CUDA_NEARFIELD_SUBSORT) &&
            cache.options.direct_kernel isa Union{PartitionedVortex,TwoPassVortex}
        _cuda_nearfield_subsort!(ctx, cache, n, n_cells)
    end

    _pack_radix_body_matrix!(ctx.source_bodies, source_buffers, view(grid.perm, 1:n),
        grid.body_system, grid.body_index)
    # near-set adequacy for regularized kernels (032 stage 2): a device
    # max-reduction over the packed σ row (pool-served scratch, six-byte-scale
    # download), no-op for singular kernels
    _direct_kernel_geometry_gate!(cache, cache.options.direct_kernel,
        ctx.source_bodies, n)

    # host mirrors serve host-resident target finalization only
    if _radix_any_host_resident(systems)
        copyto!(ctx.host_perm, 1, grid.perm, 1, n)
        copyto!(ctx.host_body_system, 1, grid.body_system, 1, n)
        copyto!(ctx.host_body_index, 1, grid.body_index, 1, n)
        counters.metadata_downloads += 3
    end

    # Occupancy-epoch fold (task 029 cycle 1): tree routes, hierarchical
    # occupancy, direct pairs, cached M2L windows, and the operator-group edges
    # are all pure functions of the occupied cell set, so they are regenerated
    # only when `occ_changed` (always, when window caching is disabled — the
    # grid update then reports every step as changed).
    # multi-root tree edges (task 037 stage 3): every node at root_level is a
    # root, so the edge count is n_nodes - n_root_nodes (legacy: n_nodes - 1)
    n_root_nodes = cache.level_offsets[cache.root_level + 2]
    n_edges = max(n_nodes - n_root_nodes, 0)
    if occ_changed && n_edges > 0
        blocks = cld(n_edges, 128)
        CUDA.@cuda threads=128 blocks=blocks _cuda_tree_routes_kernel!(
            ctx.m2m_parent_routes, ctx.m2m_child_routes,
            ctx.l2l_parent_routes, ctx.l2l_child_routes,
            view(grid.parent_index, 1:n_nodes), n_root_nodes,
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
        # metadata (on occupancy change), and — when the window cache is
        # eligible — regenerates the per-level M2L route windows here from
        # refresh-final node metadata (windows never read expansions, so this
        # respects lifecycle ordering). With caching off or an incompatible
        # plan, the windows are generated and applied inside the M2L stage as
        # before.
        if occ_changed
            hctx.epoch_id += 1
            hctx.win_valid = false
            t_stage = profiling ? (CUDA.synchronize(); time_ns()) : UInt64(0)
            _cuda_hier_refresh_occupancy!(hctx, grid, cache.level_offsets)
            if profiling
                CUDA.synchronize()
                hctx.update_stage_ns[2] = time_ns() - t_stage
                t_stage = time_ns()
            end
            n_direct = _cuda_hier_generate_direct_pairs!(ctx, hctx, grid, n_cells,
                cache.level_offsets[cache.ell + 1], cache.ell)
            hctx.epoch_n_direct = n_direct
            if profiling
                CUDA.synchronize()
                hctx.update_stage_ns[3] = time_ns() - t_stage
            end
        else
            n_direct = hctx.epoch_n_direct
            profiling && (hctx.update_stage_ns[2] = 0; hctx.update_stage_ns[3] = 0)
        end
        if isempty(hctx.symmetric_targets)
            hctx.n_symmetric_pairs = 0
        else
            # oversized-cell fallback selection reads per-cell body counts, so
            # the symmetric compaction refreshes every step
            _cuda_compact_symmetric_pairs!(ctx, hctx, grid.cell_ranges, n_direct,
                radix_setting(:SYMMETRIC_CUDA_MAX_CELL_BODIES))
        end
        if radix_setting(:CUDA_CACHED_WINDOWS) && !hctx.win_valid && _cuda_windows_cacheable(hctx)
            t_stage = profiling ? (CUDA.synchronize(); time_ns()) : UInt64(0)
            _cuda_hier_cache_windows!(ctx, hctx, grid)
            profiling && (CUDA.synchronize();
                hctx.update_stage_ns[4] = time_ns() - t_stage)
        end
        # with a valid window cache the step total is already known here; a
        # replayed (graph-captured) lifecycle performs no host bookkeeping, so
        # the refresh is the place that keeps `counts.n_routes` truthful
        n_routes = hctx.win_valid ? hctx.total_routes : 0
    end
    t_stage = profiling ? (CUDA.synchronize(); time_ns()) : UInt64(0)
    if occ_changed
        _cuda_refresh_resident_stage_groups!(ctx.workspace, grid, cache.level_offsets,
            cache.ell, cache.root_level)
    end
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
            ctx.invariant, ctx.workspace, counters, cache.options, counts;
            sfs=ctx.sfs_ctx,
        )
    end
    cache.step += 1
    return cache
end

update_cuda_radix_state!(cache::RadixFMMCache, systems) =
    update_cuda_radix_state!(cache, to_tuple(systems))

function _radix_cache_device_step!(cache::RadixFMMCache, targets::Tuple, switches::Tuple;
        sfs::Bool=false)
    # task 047: construction-locked settings must not have drifted — a late
    # flip is baked-in-silently otherwise (buffers/captured graph).
    verify_locked_radix_settings(cache.locked_settings)
    update_cuda_radix_state!(cache, targets)
    if cache.adaptive === nothing
        run_cuda_radix_lifecycle!(cache.state)
        state = cache.state
    else
        # task 041: adaptive device lifecycle (B2M -> M2M -> V M2L -> X S2L ->
        # L2L -> U direct + L2B -> W M2T) over the device adaptive tree
        state = cache.adaptive_state::DeviceResidentRadixState
        run_cuda_adaptive_radix_lifecycle!(state,
            cache.adaptive_tree::DeviceAdaptiveCUDAContext)
    end
    # SFS is a per-evaluation option, not merely a cache capability. Keep it
    # outside the U/J graph so an sfs-armed cache executes no TG/ζ kernels on
    # the (default) sfs=false path. Stream order guarantees completed J here.
    sfs && _launch_cuda_sfs!(state)
    finalize_cuda_radix_output!(state, targets; derivatives_switches=switches,
        host_output_staging=cache.device_ctx.host_output,
        target_buffers=_radix_cache_target_buffers!(cache, switches),
        device_target_buffers=cache.device_ctx.device_target_buffers)
    # task 048: the per-call flag gates both the SFS launch above and delivery
    sfs && finalize_cuda_radix_sfs_output!(state, targets;
        host_sfs_staging=cache.device_ctx.host_sfs_staging,
        sfs_target_buffers=_radix_cache_sfs_buffers!(cache, targets),
        device_sfs_buffers=cache.device_ctx.device_sfs_buffers)
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
#
# Task 029 cycle 1 tightens this further: with the occupancy-epoch window cache
# (CUDA_CACHED_WINDOWS, default on, fused dense plans) the per-window prefix
# download recurs only when the occupied cell set changes; the steady-state step's
# entire host traffic for M2L is the 4-byte epoch flag read in the refresh. The
# cached concatenation stores the compacted routes of the current epoch
# (~3 x total_routes words) — bounded storage the 026/027 design deferred, now
# accepted deliberately in exchange for removing the per-step scan/compact work
# and its blocking synchronization (job 13059955 attribution).

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
        if class_of[phase, k, L + 1] != Int32(0)
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

# Task 028 Stage 8: compact the directed leaf-cell list to one unordered entry
# for ordinary pairs. Oversized cells retain both directed entries and encode
# fallback by a negative target id, so the measured refresh performs all
# selection on device without a host decision or a second pair list.
function _cuda_symmetric_pair_flags_kernel!(flags, direct_targets, direct_sources,
        cell_ranges, n_direct, max_bodies)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > n_direct && return nothing
    @inbounds begin
        t = direct_targets[i]
        s = direct_sources[i]
        oversized = cell_ranges[2, t] > max_bodies ||
            cell_ranges[2, s] > max_bodies
        flags[i] = (oversized || t <= s) ? Int32(1) : Int32(0)
    end
    return nothing
end

function _cuda_symmetric_pair_compact_kernel!(targets, sources, flags, prefix,
        direct_targets, direct_sources, cell_ranges, n_direct, max_bodies)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > n_direct && return nothing
    @inbounds begin
        flags[i] == Int32(1) || return nothing
        t = direct_targets[i]
        s = direct_sources[i]
        oversized = cell_ranges[2, t] > max_bodies ||
            cell_ranges[2, s] > max_bodies
        p = Int(prefix[i])
        targets[p] = oversized ? -t : t
        sources[p] = s
    end
    return nothing
end

function _cuda_compact_symmetric_pairs!(ctx, hctx::DeviceHierarchicalM2LContext,
        cell_ranges, n_direct::Int, max_bodies::Int)
    n_direct == 0 && (hctx.n_symmetric_pairs = 0; return 0)
    n_direct <= length(ctx.direct_flags) || throw(AssertionError(
        "symmetric compaction exceeds direct scratch capacity"))
    threads = 256
    blocks = cld(n_direct, threads)
    CUDA.@cuda threads=threads blocks=blocks _cuda_symmetric_pair_flags_kernel!(
        ctx.direct_flags, ctx.direct_targets, ctx.direct_sources,
        cell_ranges, n_direct, max_bodies)
    fv = view(ctx.direct_flags, 1:n_direct)
    pv = view(ctx.direct_prefix, 1:n_direct)
    accumulate!(+, pv, fv)
    copyto!(ctx.host_scalar32, 1, ctx.direct_prefix, n_direct, 1)
    n = Int(ctx.host_scalar32[1])
    n <= length(hctx.symmetric_targets) || throw(AssertionError(
        "symmetric pair buffer exceeded capacity"))
    n > 0 && CUDA.@cuda threads=threads blocks=blocks _cuda_symmetric_pair_compact_kernel!(
        hctx.symmetric_targets, hctx.symmetric_sources, ctx.direct_flags,
        ctx.direct_prefix, ctx.direct_targets, ctx.direct_sources,
        cell_ranges, n_direct, max_bodies)
    hctx.n_symmetric_pairs = n
    return n
end

# Generate exactly one `(level, offset window)` into the start of the route buffers
# and return its route count. One scan scalar set (the `kn`-entry window prefix) is
# downloaded per window; nothing is appended to a growing list.
function _cuda_hier_generate_window!(state::DeviceResidentRadixState,
        hctx::DeviceHierarchicalM2LContext, route_class, L::Int, first_offset::Int,
        last_offset::Int, class_base::Int)
    return _cuda_hier_generate_window_core!(state.route_levels, state.route_offsets,
        state.route_targets, state.route_sources, state.grid, hctx, route_class,
        L, first_offset, last_offset, class_base)
end

# State-free core (task 029 cycle 1): the occupancy-epoch window cache generates
# windows during the refresh, before `cache.state` exists on the construction
# step, so the generator takes the persistent route buffers and grid explicitly.
function _cuda_hier_generate_window_core!(route_levels, route_offsets,
        route_targets, route_sources, grid::DeviceRadixGrid,
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
        hctx.route_flags, hctx.node_at, grid.node_coords, hctx.d_push_offsets,
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
    n_routes <= length(route_targets) || throw(AssertionError(
        "device hierarchical route window exceeded capacity " *
        "$(length(route_targets)); increase window storage or reduce window_classes"))
    CUDA.@cuda threads=threads blocks=blocks _cuda_hier_route_compact_kernel!(
        route_levels, route_offsets, route_targets,
        route_sources, route_class, hctx.route_flags, hctx.route_prefix,
        hctx.node_at, grid.node_coords, hctx.d_push_offsets, level_base_L,
        first_source, n_sources, first_offset, kn, L, class_base,
    )
    return n_routes
end

# ---- task 029 cycle 1: occupancy-epoch M2L window cache ----------------------

# The cache is eligible exactly when the per-level apply consumes only
# (route_class, route_sources, route_targets, n_routes): the fused kernel family
# of the dense CUDA plan (fused / tiled / tensor16). The GEMM reference drivers
# additionally consume per-window class starts/counts and stay on the
# generate-and-apply-per-window path.
_cuda_windows_cacheable(hctx::DeviceHierarchicalM2LContext) =
    hctx.apply_plan isa ResidentM2LDenseCUDAPlan && radix_setting(:DENSE_CUDA_FUSED)

# Grow the cached-window arrays to `needed`, preserving the first `cursor`
# entries. Growth happens only inside epoch regeneration (never on the
# steady-state step), so this allocation recurs exactly with occupancy change.
function _cuda_hier_win_ensure!(hctx::DeviceHierarchicalM2LContext, cursor::Int,
        needed::Int)
    old_class = hctx.win_class
    cap = old_class === nothing ? 0 : length(old_class::CUDA.CuVector{Int32})
    needed <= cap && return nothing
    newcap = max(needed, cap + cld(cap, 2), 1024)
    new_class = CUDA.CuVector{Int32}(undef, newcap)
    new_sources = CUDA.CuVector{Int}(undef, newcap)
    new_targets = CUDA.CuVector{Int}(undef, newcap)
    if cursor > 0
        copyto!(new_class, 1, old_class::CUDA.CuVector{Int32}, 1, cursor)
        copyto!(new_sources, 1, hctx.win_sources::CUDA.CuVector{Int}, 1, cursor)
        copyto!(new_targets, 1, hctx.win_targets::CUDA.CuVector{Int}, 1, cursor)
    end
    hctx.win_class = new_class
    hctx.win_sources = new_sources
    hctx.win_targets = new_targets
    return nothing
end

# Regenerate the complete per-level window concatenation for the current
# occupancy epoch. Runs inside the refresh (node metadata is final; windows
# never read expansions) and reuses the single-window generator verbatim, so the
# cached route stream is elementwise identical — same class-major order, same
# per-window class contiguity — to the per-step generate-and-apply loop; with
# `window_classes = noffsets` (the production K=full configs) each level is one
# window and the concatenation is bit-identical to the uncached stream. The
# per-window `kn`-entry prefix D2H still happens here, but once per epoch
# instead of once per step.
function _cuda_hier_cache_windows!(ctx, hctx::DeviceHierarchicalM2LContext,
        grid::DeviceRadixGrid)
    plan = hctx.apply_plan::ResidentM2LDenseCUDAPlan
    route_class = plan.route_class
    noffsets = hctx.noffsets
    K = hctx.window_classes
    ell = hctx.ell
    cursor = 0
    fill!(hctx.win_level_starts, 0)
    fill!(hctx.win_level_counts, 0)
    fill!(hctx.routes_per_level, 0)
    for L in hctx.first_m2l_level:ell
        hctx.win_level_starts[L + 1] = cursor
        level_total = 0
        for first_offset in 1:K:noffsets
            last_offset = min(first_offset + K - 1, noffsets)
            n = _cuda_hier_generate_window_core!(ctx.route_levels,
                ctx.route_offsets, ctx.route_targets, ctx.route_sources, grid,
                hctx, route_class, L, first_offset, last_offset, 0)
            n == 0 && continue
            _cuda_hier_win_ensure!(hctx, cursor, cursor + n)
            copyto!(hctx.win_class::CUDA.CuVector{Int32}, cursor + 1, route_class, 1, n)
            copyto!(hctx.win_sources::CUDA.CuVector{Int}, cursor + 1, ctx.route_sources, 1, n)
            copyto!(hctx.win_targets::CUDA.CuVector{Int}, cursor + 1, ctx.route_targets, 1, n)
            cursor += n
            level_total += n
        end
        hctx.win_level_counts[L + 1] = level_total
        hctx.routes_per_level[L + 1] = level_total
    end
    hctx.total_routes = cursor
    hctx.last_window_routes = 0
    hctx.win_valid = true
    return hctx
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
    if !radix_setting(:PRECOMPUTED_CUDA_WHOLE_PASS)
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
    if !radix_setting(:DENSE_CUDA_FUSED)
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

# Code-order-3 scalar tensor prototype: one warp applies a 16x16 operator to
# sixteen same-class routes with FP32 accumulation. Global 16-route batches that
# cross a class boundary (or the sparse final tail) execute the tiled FP32
# matvec in the same kernel, preserving exact route coverage without host work.
function _cuda_hier_dense_tensor16_kernel!(loc_phi, ops, ops_low, input_scale, route_class,
        route_sources, route_targets, mp_phi, phi_flat_idx, src_scale, tgt_scale,
        lcol, n_routes)
    TI = eltype(ops_low)
    lane = Int(threadIdx().x - Int32(1))
    conf = CUDA.WMMA.Config{16,16,16,Float32}
    bbuf = CUDA.CuDynamicSharedArray(TI, (16, 16))
    dbuf = CUDA.CuDynamicSharedArray(Float32, (16, 16), 256 * sizeof(TI))
    batch = Int(blockIdx().x)
    batch_stride = Int(gridDim().x)
    @inbounds while true
        j0 = (batch - 1) * 16 + 1
        j0 > n_routes && break
        je = min(j0 + 15, n_routes)
        k = Int(route_class[j0])
        tensor_batch = je == j0 + 15 && Int(route_class[je]) == k
        if tensor_batch
            idx = lane + 1
            while idx <= 256
                r = (idx - 1) % 16 + 1
                col = (idx - 1) ÷ 16 + 1
                src = route_sources[j0 + col - 1]
                bbuf[r, col] = TI(mp_phi[phi_flat_idx[r], src] *
                    src_scale[r, lcol] * input_scale[r, k])
                idx += 32
            end
            CUDA.sync_warp()
            aoff = (k - 1) * 256 + 1
            afrag = CUDA.WMMA.load_a(pointer(ops_low, aoff), 16,
                CUDA.WMMA.ColMajor, conf)
            bfrag = CUDA.WMMA.load_b(pointer(bbuf), 16,
                CUDA.WMMA.ColMajor, conf)
            cfrag = CUDA.WMMA.fill_c(0.0f0, conf)
            dfrag = CUDA.WMMA.mma(afrag, bfrag, cfrag, conf)
            CUDA.WMMA.store_d(pointer(dbuf), dfrag, 16,
                CUDA.WMMA.ColMajor, conf)
            CUDA.sync_warp()
            idx = lane + 1
            while idx <= 256
                r = (idx - 1) % 16 + 1
                col = (idx - 1) ÷ 16 + 1
                tgt = route_targets[j0 + col - 1]
                v = dbuf[r, col] * tgt_scale[r, lcol]
                CUDA.@atomic loc_phi[phi_flat_idx[r], tgt] += v
                idx += 32
            end
        else
            # Sparse/class-boundary tail: route-wise FP32 reference arithmetic.
            for j in j0:je
                kk = Int(route_class[j])
                src = route_sources[j]
                tgt = route_targets[j]
                r = lane + 1
                if r <= 16
                    acc = 0.0f0
                    for i in 1:16
                        acc += tgt_scale[r, lcol] * ops[r, i, kk] *
                            src_scale[i, lcol] * mp_phi[phi_flat_idx[i], src]
                    end
                    CUDA.@atomic loc_phi[phi_flat_idx[r], tgt] += acc
                end
            end
        end
        batch += batch_stride
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
# Fused-family per-level apply on an explicit route slice (task 029 cycle 1):
# consumes only (route_class, route_sources, route_targets, n_routes), so the
# same body serves both the per-window path (state route buffers) and the
# occupancy-epoch window cache (per-level views of the cached concatenation).
function _cuda_hier_dense_apply_routes!(state::DeviceResidentRadixState{TF,B,LH},
        ws::ResidentOperatorWorkspace{TF,B,LH}, plan::ResidentM2LDenseCUDAPlan,
        # `hctx` is duck-typed (task 041): only first_m2l_level and the
        # per-level source/target scales are read, so the adaptive device
        # context can drive the same per-level applies over its CSR stream
        hctx, L::Int, route_class, route_sources,
        route_targets, n_routes::Int) where {TF,B,LH}
    n_routes == 0 && return state
    lcol = L - hctx.first_m2l_level + 1
    D = plan.ndof
    tensor_format = radix_setting(:DENSE_CUDA_TENSOR_FORMAT)
    if tensor_format !== :off && TF === Float32 && !LH && D == 16
        tensor_format in (:fp16, :bf16) || throw(ArgumentError(
            "DENSE_CUDA_TENSOR_FORMAT must be :off, :fp16, or :bf16"))
        ops_low = tensor_format === :fp16 ? plan.tensor_fp16_operators :
            plan.tensor_bf16_operators
        isempty(ops_low) && throw(ArgumentError(
            "tensor M2L operator cache is unavailable for this configuration"))
        blocks = min(cld(n_routes, 16), radix_setting(:DENSE_CUDA_TILED_MAX_BLOCKS))
        shmem = 256 * sizeof(eltype(ops_low)) + 256 * sizeof(Float32)
        CUDA.@cuda threads=32 blocks=blocks shmem=shmem _cuda_hier_dense_tensor16_kernel!(
            state.locals.phi, plan.operators, ops_low, plan.tensor_input_scale,
            route_class,
            route_sources, route_targets, state.multipoles.phi,
            ws.phi_flat_idx, hctx.source_scale, hctx.target_scale, lcol, n_routes)
        return state
    end
    # tiled path (task 028 cycle 2/Stage 6): shared memory holds the folded
    # D x D class tile plus one multipole column per warp.  Launch controls
    # are internal Refs so complete-verdict A/Bs can be run without adding
    # public cache/API surface.
    tiled_threads = radix_setting(:DENSE_CUDA_TILED_THREADS)
    32 <= tiled_threads <= 1024 && tiled_threads % 32 == 0 ||
        throw(ArgumentError("DENSE_CUDA_TILED_THREADS must be a warp multiple in 32:1024"))
    tiled_cap = radix_setting(:DENSE_CUDA_TILED_MAX_BLOCKS)
    tiled_cap > 0 || throw(ArgumentError("DENSE_CUDA_TILED_MAX_BLOCKS must be positive"))
    tiled_warps = tiled_threads ÷ 32
    tiled_shmem = (D * D + tiled_warps * D) * sizeof(TF)
    if radix_setting(:DENSE_CUDA_TILED) && n_routes >= radix_setting(:DENSE_CUDA_TILED_MIN_ROUTES) &&
            tiled_shmem <= 48 * 1024
        blocks = min(cld(n_routes, tiled_warps), tiled_cap)
        CUDA.@cuda threads=tiled_threads blocks=blocks shmem=tiled_shmem _cuda_hier_dense_tiled_kernel!(
            state.locals.phi, state.locals.chi, plan.operators, route_class,
            route_sources, route_targets, state.multipoles.phi,
            state.multipoles.chi, ws.phi_flat_idx, ws.chi_flat_idx, plan.ndof_phi,
            hctx.source_scale, hctx.target_scale, lcol, n_routes, Val(LH),
        )
        return state
    end
    threads = min(256, cld(D, 32) * 32)
    shmem = D * sizeof(TF)
    blocks = min(n_routes, radix_setting(:DENSE_CUDA_FUSED_MAX_BLOCKS))
    CUDA.@cuda threads=threads blocks=blocks shmem=shmem _cuda_hier_dense_fused_kernel!(
        state.locals.phi, state.locals.chi, plan.operators, route_class,
        route_sources, route_targets, state.multipoles.phi,
        state.multipoles.chi, ws.phi_flat_idx, ws.chi_flat_idx, plan.ndof_phi,
        hctx.source_scale, hctx.target_scale, lcol, n_routes, Val(LH),
    )
    return state
end

function _cuda_hier_dense_apply_window!(state::DeviceResidentRadixState{TF,B,LH},
        ws::ResidentOperatorWorkspace{TF,B,LH}, plan::ResidentM2LDenseCUDAPlan,
        hctx::DeviceHierarchicalM2LContext, L::Int) where {TF,B,LH}
    n_routes = state.counts.n_routes
    n_routes == 0 && return state
    lcol = L - hctx.first_m2l_level + 1
    if radix_setting(:DENSE_CUDA_FUSED)
        return _cuda_hier_dense_apply_routes!(state, ws, plan, hctx, L,
            plan.route_class, state.route_sources, state.route_targets, n_routes)
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
        hctx::DeviceHierarchicalM2LContext; replay_levels=nothing,
        replay_orbit=nothing, clear_locals::Bool=true) where {TF,B,LH}
    ws = state.scratch
    ws isa ResidentOperatorWorkspace || throw(ArgumentError(
        "hierarchical device M2L requires ResidentOperatorWorkspace scratch"))
    plan = hctx.apply_plan
    plan isa Union{ResidentM2LConcatPlan,ResidentM2LPrecomputedYPlan,
        ResidentM2LDenseCUDAPlan} || throw(ArgumentError(
        "hierarchical device M2L has no compatible construction-time plan; got $(typeof(plan))"))
    # Task 029 cycle 1: production calls consume the occupancy-epoch window
    # cache when it is valid — per-level applies only, no per-step generation.
    # The benchmark-only replay controls always take the windowed path.
    if replay_levels === nothing && replay_orbit === nothing &&
            radix_setting(:CUDA_CACHED_WINDOWS) && hctx.win_valid && _cuda_windows_cacheable(hctx)
        return _launch_cuda_hierarchical_m2l_cached!(state, hctx; clear_locals)
    end
    clear_locals && fill!(state.locals.phi, zero(TF))
    clear_locals && LH && fill!(state.locals.chi, zero(TF))
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
    for L in hctx.first_m2l_level:ell
        replay_levels === nothing || L in replay_levels || continue
        if hctx.profile_stages
            CUDA.synchronize()
            t_level = time_ns()
        else
            t_level = UInt64(0)
        end
        level_total = 0
        class_base = dense ? 0 : (L - hctx.first_m2l_level) * noffsets
        replay_K = replay_orbit === nothing ? K : 1
        for first_offset in 1:replay_K:noffsets
            last_offset = min(first_offset + replay_K - 1, noffsets)
            replay_orbit === nothing ||
                _rigid_orbit_key(hctx.tables.push_offsets[first_offset]) == replay_orbit ||
                continue
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


# Task 029 cycle 1: apply the cached per-level window concatenation. This is
# the entire steady-state M2L stage — locals clear plus one fused-family launch
# per nonempty level — with no route generation, no scan, and no host
# synchronization, so it is graph-capturable. Telemetry (`routes_per_level`,
# `total_routes`) was fixed at generation time and stays valid for the epoch.
function _launch_cuda_hierarchical_m2l_cached!(
        state::DeviceResidentRadixState{TF,B,LH},
        hctx::DeviceHierarchicalM2LContext; clear_locals::Bool=true) where {TF,B,LH}
    ws = state.scratch
    ws isa ResidentOperatorWorkspace || throw(ArgumentError(
        "hierarchical device M2L requires ResidentOperatorWorkspace scratch"))
    plan = hctx.apply_plan::ResidentM2LDenseCUDAPlan
    clear_locals && fill!(state.locals.phi, zero(TF))
    clear_locals && LH && fill!(state.locals.chi, zero(TF))
    wc = hctx.win_class::CUDA.CuVector{Int32}
    wsrc = hctx.win_sources::CUDA.CuVector{Int}
    wtgt = hctx.win_targets::CUDA.CuVector{Int}
    profile = hctx.profile_stages
    if profile
        fill!(hctx.m2l_level_ns, 0)
        # generation happens at epoch boundaries inside the refresh; the
        # steady-state M2L stage has no flag/scan/compact cost to report
        hctx.update_stage_ns[4] = 0
    end
    for L in hctx.first_m2l_level:hctx.ell
        n = hctx.win_level_counts[L + 1]
        s = hctx.win_level_starts[L + 1]
        t_level = profile ? (CUDA.synchronize(); time_ns()) : UInt64(0)
        n > 0 && _cuda_hier_dense_apply_routes!(state, ws, plan, hctx, L,
            view(wc, (s + 1):(s + n)), view(wsrc, (s + 1):(s + n)),
            view(wtgt, (s + 1):(s + n)), n)
        if profile
            CUDA.synchronize()
            hctx.m2l_level_ns[L + 1] = time_ns() - t_level
        end
    end
    state.counts.n_routes = hctx.total_routes
    return state
end

"""
Benchmark-only linear replay of complete hierarchical M2L groups. `levels`
selects whole levels and `orbit=(a,b,c)` selects a complete signed/permuted
cubic orbit via sorted absolute coordinates. Production calls never pass these
controls; they exist for task-028 sampled-field reconstruction and attribution.
"""
function _launch_cuda_hierarchical_m2l_replay!(state::DeviceResidentRadixState;
        levels=nothing, orbit=nothing, clear_locals::Bool=true)
    hctx = state.interaction_list
    hctx isa DeviceHierarchicalM2LContext || throw(ArgumentError(
        "hierarchical M2L replay requires a device hierarchical state"))
    orbit === nothing || (orbit isa NTuple{3,Int} &&
        orbit[1] >= orbit[2] >= orbit[3] >= 0) || throw(ArgumentError(
        "replay orbit must be a descending nonnegative integer triple"))
    return _launch_cuda_hierarchical_m2l!(state, hctx;
        replay_levels=levels, replay_orbit=orbit, clear_locals)
end

# Per-level Lambda columns for the hierarchical dense operator table. Level L uses
# s = 2^(ell - L) (the leaf level is the reference at which the operators were
# built); the phi rows scale as s^-n / s^-(n+1) and the Lamb-Helmholtz chi rows as
# s^-(n-1) / s^-(n+2) — the asymmetric pair the host plan encodes per class.
function _cuda_hier_dense_scales(::Type{TF}, basis_info::OperatorBasisInfo{B,LH},
        ell::Int, D::Int, first_m2l_level::Int=2) where {TF,B,LH}
    # columns are sized by the active level count (task 037 stage 3); column
    # L - first_m2l_level + 1 carries level L
    nlevels = max(ell - first_m2l_level + 1, 0)
    source_scale = ones(TF, D, nlevels)
    target_scale = ones(TF, D, nlevels)
    Dphi = degree_major_dof(basis_info.orders.P_phi)
    @inbounds for L in first_m2l_level:ell
        col = L - first_m2l_level + 1
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
        effective_offsets::Vector{SVector{3,Int}}, level_class_of::Array{Int32,3},
        level_radii2::Vector{Int}, plan, ell::Int, first_m2l_level::Int,
        max_level_nodes::Int, direct_capacity::Int,
        counters::CUDARadixTransferCounters) where {TF,B,LH}
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
    # Clamp to >= 1 so `1:K:noffsets` window strides stay well-formed on the
    # zero-M2L degenerate cache (noffsets == 0, task 052c) — every such loop
    # is empty anyway, but a zero step would be an ArgumentError.
    K = max(min(policy.window_classes, noffsets), 1)
    flag_capacity = max(K * max_level_nodes, 1)
    node_at = CUDA.zeros(Int32, length(occupancy.node_at))
    d_level_base = CUDA.CuArray{Int}(occupancy.level_base)
    d_push_offsets = CUDA.CuArray{Int32}(_radix_offsets_matrix(tables.push_offsets))
    size(level_class_of) == (8, noffsets, ell + 1) || throw(ArgumentError(
        "invalid hierarchical per-level class table dimensions $(size(level_class_of))"))
    d_class_of = CUDA.CuArray{Int32}(level_class_of)
    d_near_offsets = CUDA.CuArray{Int32}(_radix_offsets_matrix(tables.near_offsets))
    counters.operator_uploads += 1
    dense = plan isa ResidentM2LDenseCUDAPlan
    host_source_scale, host_target_scale = dense ?
        _cuda_hier_dense_scales(TF, basis_info, ell, plan.ndof, first_m2l_level) :
        (Matrix{TF}(undef, 0, 0), Matrix{TF}(undef, 0, 0))
    source_scale = CUDA.CuArray{TF}(host_source_scale)
    target_scale = CUDA.CuArray{TF}(host_target_scale)
    symmetric_capacity = radix_setting(:CUDA_SYMMETRIC_NEARFIELD) && !LH ? direct_capacity : 0
    return DeviceHierarchicalM2LContext(
        tables, level_radii2, class_level, class_offset, effective_offsets, plan,
        K, ell, first_m2l_level, noffsets,
        copy(occupancy.level_base), zeros(Int, ell + 2),
        node_at, d_level_base, d_push_offsets, d_class_of, d_near_offsets,
        CUDA.zeros(Int, symmetric_capacity), CUDA.zeros(Int, symmetric_capacity),
        CUDA.zeros(Int32, flag_capacity), CUDA.zeros(Int32, flag_capacity),
        CUDA.zeros(Int32, max(K, 1)), _pin_host_array(zeros(Int32, max(K, 1))),
        source_scale, target_scale,
        0, zeros(Int, ell + 1), zeros(Int, ell + 1), 0, 0, 1, 0,
        false, zeros(UInt64, 5), zeros(UInt64, ell + 1),
        # 029 cycle 1: epoch/window-cache/graph state (windows and graph are
        # generated lazily on the first refresh/lifecycle of each epoch)
        0, 0, false, zeros(Int, ell + 2), zeros(Int, ell + 2),
        nothing, nothing, nothing, nothing, -1, -1,
        # 032a stage C nearfield bin context: attached after the update context
        # exists (it shares the update context's cell_coords/subsort scratch)
        nothing,
    )
end

# Build the task-032a Stage C nearfield bin context for a split direct kernel on
# a hierarchical device cache. All storage is construction-sized: three bucket
# thirds at the direct-pair capacity (Int32 cell ids), max_cells σ extrema, and
# the TwoPassVortex pass-2 offset ball at its gate-derived reach capacity
# (rho_t/rho_c)·g_min cells (see `_twopass_offset_ball`). The one construction
# upload (the offset ball) is counted as a route upload; the recurring step
# performs no transfer. Implicit per-array cross-stream synchronization is
# disabled for everything the side-stream nearfield touches (see the
# CUDA_OVERLAP_NEARFIELD comment block — orderings are event-based, and
# implicit sync is illegal inside stream capture).
function _build_cuda_nearfield_bin_context(::Type{TF},
        dk::Union{PartitionedVortex,TwoPassVortex}, policy, accepted,
        ell::Int, h0, x_min, max_cells::Int, direct_capacity::Int, ctx,
        counters::CUDARadixTransferCounters) where TF
    h_leaf = 2 * Float64(h0) / (1 << ell)
    cap = direct_capacity
    if dk isa TwoPassVortex
        g_min = _leaf_stencil_min_gap(policy, accepted)
        reach_cap = (dk.rho_t / dk.rho_c) * g_min
        host_offsets, host_gap2 = _twopass_offset_ball(reach_cap)
        tp_offsets = CUDA.CuArray{Int32}(host_offsets)
        tp_gap2 = CUDA.CuArray{Int32}(host_gap2)
        counters.route_uploads += 1
        K = length(host_gap2)
    else
        tp_offsets = CUDA.zeros(Int32, 3, 0)
        tp_gap2 = CUDA.zeros(Int32, 0)
        K = 0
        reach_cap = 0.0
    end
    # task 037f: the :lut g/h table is built unconditionally (8 KB device
    # memory) so the mode Ref can be flipped between constructions without a
    # separate cache shape; one construction upload, counted as an operator
    # upload like the hierarchical tables
    gh_lut = CUDA.CuArray{Float32}(_build_gh_lut(dk.rho_t))
    counters.operator_uploads += 1
    nf = CUDANearfieldBinContext(
        CUDA.zeros(TF, max_cells), CUDA.zeros(TF, max_cells),
        CUDA.zeros(Float64, 2),
        CUDA.zeros(Int32, 3 * cap), CUDA.zeros(Int32, 3 * cap),
        CUDA.zeros(Int32, 3), cap,
        ctx.cell_coords, h_leaf, SVector{3,Float64}(x_min),
        tp_offsets, tp_gap2, K, reach_cap,
        ctx.subsort_keys, CUDA.zeros(UInt64, 12), gh_lut)
    for arr in (nf.cell_sigma_max, nf.cell_sigma_min, nf.nf_scalars,
            nf.bin_targets, nf.bin_sources, nf.bin_counts, nf.diag, nf.gh_lut,
            ctx.cell_coords, ctx.grid.cell_keys)
        CUDA.enable_synchronization!(arr, false)
    end
    if K > 0
        CUDA.enable_synchronization!(nf.twopass_offsets, false)
        CUDA.enable_synchronization!(nf.twopass_gap2, false)
    end
    return nf
end

# Mechanism (a): compose a within-cell sub-Morton ordering into `grid.perm`
# after the sort and before body packing, so consecutive sorted bodies (= warp
# lanes) span a compact spatial sub-block of their cell. Cells larger than the
# shared-memory sort capacity (1024) keep their unspecified order — the
# mechanism is an approximate-coherence lever, not a correctness requirement.
function _cuda_nearfield_subsort!(ctx, cache::RadixFMMCache, n::Int, n_cells::Int)
    sub = min(3, RADIX_GRID_MAX_ELL - cache.ell)
    (sub > 0 && n > 0 && n_cells > 0) || return nothing
    grid = ctx.grid
    threads = 128
    blocks = cld(n, threads)
    CUDA.@cuda threads=threads blocks=blocks _cuda_subsort_keys_kernel!(
        ctx.subsort_keys, ctx.positions, grid.perm, cache.x_min, cache.h0,
        cache.ell, sub, n)
    CUDA.@cuda threads=256 blocks=min(n_cells, 8192) _cuda_subsort_cell_sort_kernel!(
        grid.perm, ctx.subsort_keys, grid.cell_ranges, n_cells)
    pv = view(grid.perm, 1:n)
    CUDA.@cuda threads=threads blocks=blocks _cuda_fill_invperm_kernel!(
        grid.invperm, pv)
    return nothing
end

# Per-sorted-position sub-Morton key: the body's Morton bits below the leaf
# level, at `sub` extra levels of resolution (positions are inside the fixed
# box by the oob gate that already ran this step).
function _cuda_subsort_keys_kernel!(subsort_keys, positions, perm, x_min, h0,
        ell, sub, n)
    p = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    p > n && return nothing
    @inbounds begin
        b = perm[p]
        Gs = 1 << (ell + sub)
        m = (1 << sub) - 1
        T = eltype(positions)
        delta = (2 * h0) / T(Gs)
        cx = min(max(unsafe_trunc(Int32, (positions[1, b] - x_min[1]) / delta),
            Int32(0)), Int32(Gs - 1)) & Int32(m)
        cy = min(max(unsafe_trunc(Int32, (positions[2, b] - x_min[2]) / delta),
            Int32(0)), Int32(Gs - 1)) & Int32(m)
        cz = min(max(unsafe_trunc(Int32, (positions[3, b] - x_min[3]) / delta),
            Int32(0)), Int32(Gs - 1)) & Int32(m)
        key = UInt32(0)
        bit = 0
        while bit < sub
            key |= (UInt32((cx >> bit) & Int32(1)) << (3 * bit))
            key |= (UInt32((cy >> bit) & Int32(1)) << (3 * bit + 1))
            key |= (UInt32((cz >> bit) & Int32(1)) << (3 * bit + 2))
            bit += 1
        end
        subsort_keys[p] = key
    end
    return nothing
end

# Block-per-cell odd-even transposition sort of the perm segment by sub-key in
# shared memory (capacity 1024 bodies; larger cells are skipped). The condition
# is uniform per block, so the barriers are safe.
function _cuda_subsort_cell_sort_kernel!(perm, subsort_keys, cell_ranges, n_cells)
    keys_sh = CUDA.CuStaticSharedArray(UInt32, 1024)
    perm_sh = CUDA.CuStaticSharedArray(Int64, 1024)
    cell = Int(blockIdx().x)
    stride = Int(gridDim().x)
    nt = Int(blockDim().x)
    t = Int(threadIdx().x)
    @inbounds while cell <= n_cells
        first = cell_ranges[1, cell]
        cnt = cell_ranges[2, cell]
        if 1 < cnt <= 1024
            idx = t
            while idx <= cnt
                keys_sh[idx] = subsort_keys[first + idx - 1]
                perm_sh[idx] = perm[first + idx - 1]
                idx += nt
            end
            CUDA.sync_threads()
            phase = 0
            while phase < cnt
                base = 1 + (phase & 1)
                idx = base + 2 * (t - 1)
                while idx <= cnt - 1
                    ka = keys_sh[idx]
                    kb = keys_sh[idx + 1]
                    if kb < ka
                        keys_sh[idx] = kb
                        keys_sh[idx + 1] = ka
                        pa = perm_sh[idx]
                        perm_sh[idx] = perm_sh[idx + 1]
                        perm_sh[idx + 1] = pa
                    end
                    idx += 2 * nt
                end
                CUDA.sync_threads()
                phase += 1
            end
            idx = t
            while idx <= cnt
                subsort_keys[first + idx - 1] = keys_sh[idx]
                perm[first + idx - 1] = perm_sh[idx]
                idx += nt
            end
            CUDA.sync_threads()
        end
        cell += stride
    end
    return nothing
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
        (Int(level) - hctx.first_m2l_level) * hctx.noffsets
    n = _cuda_hier_generate_window!(state, hctx, plan.route_class, Int(level),
        Int(first_offset), Int(last_offset), class_base)
    hctx.last_window_routes = n
    state.counts.n_routes = n
    return n
end

#------- adaptive octree device-resident lifecycle (task 041) -------#
#
# CUDA mirror of the task-040 host adaptive lifecycle. Construction/refresh and
# the DTR lists live in tree_batched_cuda.jl (included below); this section adds
# the device M2T/S2L kernels, the adaptive V-CSR M2L driver over the UNCHANGED
# resident window plans, the lifecycle body with the uniform path's nearfield
# overlap + graph-capture semantics, and the cache build/step plumbing.
#
# Contract mirror of the uniform device path: construction-only operator/route
# uploads (route_uploads/operator_uploads constant after construction),
# expansion_host_copies == 0 around the pipeline, zero recurring allocation
# outside CUDA's pool-served sort scratch, and the adaptive path is entirely
# opt-in (nothing here runs unless the cache carries an AdaptiveTreePolicy).

include(joinpath(@__DIR__, "tree_batched_cuda.jl"))

#------- device M2T (W list) -------#

# Fixed harmonic-kernel grid: blocks x 128 threads share the preallocated
# per-thread irregular-harmonic scratch slab (grid-stride loops cover any list
# length). 512 x 128 slots x 66 complex terms (P=8) is 69 MB in Float64.
const _ADT_CUDA_HARMONIC_BLOCKS = 512

# Block-per-W-pair, thread-per-target-body: each thread evaluates the source
# node's multipole at its body via thread-local irregular harmonics (the
# validated host irregular_harmonics! + _resident_multipole_eval_flat run as
# device functions — sign conventions are shared with the host by
# construction). Different W pairs may share a target leaf, so output
# accumulation is atomic.
function _cuda_adaptive_m2t_kernel!(output, source_bodies, cell_ranges,
        leaf_slot_of, node_centers, w_targets, w_sources, n_w, ph, ch, Hall,
        ::Val{P_phi}, ::Val{P_active}, lhv::Val{LH},
        ::Val{HS}) where {P_phi,P_active,LH,HS}
    TF = eltype(output)
    slot = (Int(blockIdx().x) - 1) * Int(blockDim().x) + Int(threadIdx().x)
    H = view(Hall, :, slot:slot, :)
    k = Int(blockIdx().x)
    stride = Int(gridDim().x)
    @inbounds while k <= n_w
        ia = Int(w_targets[k])
        ib = Int(w_sources[k])
        slot = Int(leaf_slot_of[ia])
        first = cell_ranges[1, slot]
        count = cell_ranges[2, slot]
        cx = node_centers[1, ib]
        cy = node_centers[2, ib]
        cz = node_centers[3, ib]
        i = first + Int(threadIdx().x) - 1
        while i <= first + count - 1
            dx = source_bodies[1, i] - cx
            dy = source_bodies[2, i] - cy
            dz = source_bodies[3, i] - cz
            r, theta, phi = cartesian_to_spherical(dx, dy, dz)
            irregular_harmonics!(H, r, theta, phi, P_phi + 2)
            if HS
                vals = _resident_multipole_eval_flat_hessian(ph, ch, ib, H,
                    P_phi, P_active, lhv)
                for row in 1:13
                    CUDA.@atomic output[row, i] += vals[row]
                end
            else
                u, gx, gy, gz = _resident_multipole_eval_flat(ph, ch, ib, H,
                    P_phi, P_active, lhv)
                CUDA.@atomic output[1, i] += u
                CUDA.@atomic output[2, i] += gx
                CUDA.@atomic output[3, i] += gy
                CUDA.@atomic output[4, i] += gz
            end
            i += Int(blockDim().x)
        end
        k += stride
    end
    return nothing
end

function _launch_cuda_adaptive_m2t!(state::DeviceResidentRadixState{TF,B,LH},
        actx::DeviceAdaptiveCUDAContext) where {TF,B,LH}
    n_w = actx.n_w
    n_w == 0 && return state
    orders = state.invariant_cache.basis_info.orders
    P_phi = orders.P_phi
    hs = size(state.output, 1) >= 13
    grid = state.grid::DeviceRadixGrid
    threads = 128
    blocks = min(n_w, _ADT_CUDA_HARMONIC_BLOCKS)
    CUDA.@cuda threads=threads blocks=blocks _cuda_adaptive_m2t_kernel!(
        state.output, state.source_bodies, state.cell_ranges,
        actx.leaf_slot_of::CUDA.CuVector{Int32}, grid.node_centers,
        actx.w_targets::CUDA.CuVector{Int32},
        actx.w_sources::CUDA.CuVector{Int32}, n_w,
        phi_slab(state.multipoles), chi_slab(state.multipoles),
        actx.harmonics_scratch::CUDA.CuArray{TF,3},
        Val(P_phi), Val(orders.P_active), Val(LH),
        hs ? Val(true) : Val(false))
    return state
end

#------- device S2L (X list) -------#

# Block-per-X-pair, thread-per-source-body, atomic accumulation into the finer
# target node's local expansion. Scalar rule (resident convention, task 040):
# L_n^m += (-1)^(n+m) q conj(S_n^m(x_s - c_A)) — no legacy strength negation.
function _cuda_adaptive_s2l_kernel!(lp, source_bodies, cell_ranges,
        leaf_slot_of, node_centers, x_targets, x_sources, n_x, Hall,
        ::Val{P_phi}) where {P_phi}
    TF = eltype(lp)
    slot = (Int(blockIdx().x) - 1) * Int(blockDim().x) + Int(threadIdx().x)
    H = view(Hall, :, slot:slot, :)
    k = Int(blockIdx().x)
    stride = Int(gridDim().x)
    @inbounds while k <= n_x
        ia = Int(x_targets[k])
        ib = Int(x_sources[k])
        slot = Int(leaf_slot_of[ib])
        first = cell_ranges[1, slot]
        count = cell_ranges[2, slot]
        cx = node_centers[1, ia]
        cy = node_centers[2, ia]
        cz = node_centers[3, ia]
        s = first + Int(threadIdx().x) - 1
        while s <= first + count - 1
            dx = source_bodies[1, s] - cx
            dy = source_bodies[2, s] - cy
            dz = source_bodies[3, s] - cz
            q = source_bodies[5, s]
            r, theta, phi = cartesian_to_spherical(dx, dy, dz)
            irregular_harmonics!(H, r, theta, phi, P_phi)
            for n in 0:P_phi, m in 0:n
                i = harmonic_index(n, m)
                sq = isodd(n + m) ? -q : q
                row = flat_basis_index(n, m, 1)
                CUDA.@atomic lp[row, ia] += sq * _adt_S_re(H, i)
                CUDA.@atomic lp[row + 1, ia] -= sq * _adt_S_im(H, i)
            end
            s += Int(blockDim().x)
        end
        k += stride
    end
    return nothing
end

# Vortex S2L: verbatim device port of the task-040 host kernel (legacy
# strength-to-channel map, chi rows through P_active = P_phi + 1 per 008h).
function _cuda_adaptive_s2l_vortex_kernel!(lp, lc, source_bodies, cell_ranges,
        leaf_slot_of, node_centers, x_targets, x_sources, n_x, Hall,
        ::Val{P_phi}, ::Val{P_active}) where {P_phi,P_active}
    TF = eltype(lp)
    slot = (Int(blockIdx().x) - 1) * Int(blockDim().x) + Int(threadIdx().x)
    H = view(Hall, :, slot:slot, :)
    k = Int(blockIdx().x)
    stride = Int(gridDim().x)
    @inbounds while k <= n_x
        ia = Int(x_targets[k])
        ib = Int(x_sources[k])
        slot = Int(leaf_slot_of[ib])
        first = cell_ranges[1, slot]
        count = cell_ranges[2, slot]
        cx = node_centers[1, ia]
        cy = node_centers[2, ia]
        cz = node_centers[3, ia]
        s = first + Int(threadIdx().x) - 1
        while s <= first + count - 1
            dx = source_bodies[1, s] - cx
            dy = source_bodies[2, s] - cy
            dz = source_bodies[3, s] - cz
            wx = source_bodies[5, s]
            wy = source_bodies[6, s]
            wz = source_bodies[7, s]
            r, theta, phi = cartesian_to_spherical(dx, dy, dz)
            irregular_harmonics!(H, r, theta, phi, P_phi + 2)
            # phi channel (phi_00 = 0)
            for n in 1:P_phi
                _1_n = isodd(n) ? -one(TF) : one(TF)
                n_inv = inv(TF(n))
                for m in 0:n
                    _1_m = isodd(m) ? -one(TF) : one(TF)
                    i = harmonic_index(n, m)
                    local Spre::TF, Spim::TF, Smre::TF, Smim::TF
                    if m < n
                        Spre = -_1_m * _adt_S_re(H, i + 1)
                        Spim = _1_m * _adt_S_im(H, i + 1)
                    else
                        Spre = zero(TF); Spim = zero(TF)
                    end
                    Sre = _1_m * _adt_S_re(H, i)
                    Sim = -_1_m * _adt_S_im(H, i)
                    if m == 0
                        Smre = -_1_m * Spre; Smim = _1_m * Spim
                    else
                        Smre = -_1_m * _adt_S_re(H, i - 1)
                        Smim = _1_m * _adt_S_im(H, i - 1)
                    end
                    row = flat_basis_index(n, m, 1)
                    CUDA.@atomic lp[row, ia] += -_1_n * n_inv * (
                        (n - m) * TF(0.5) * (wx * Spre - wy * Spim) -
                        (n + m) * TF(0.5) * (wx * Smre + wy * Smim) +
                        wz * m * Sim)
                    CUDA.@atomic lp[row + 1, ia] += -_1_n * n_inv * (
                        (n - m) * TF(0.5) * (wx * Spim + wy * Spre) -
                        (n + m) * TF(0.5) * (wx * Smim - wy * Smre) -
                        wz * m * Sre)
                end
            end
            # chi channel through P_active (008h neighbor row included)
            for n in 0:P_active
                _1_np1 = isodd(n + 1) ? -one(TF) : one(TF)
                np1_inv = inv(TF(n + 1))
                for m in 0:n
                    _1_m = isodd(m) ? -one(TF) : one(TF)
                    i_np1 = harmonic_index(n + 1, m)
                    Sp1pre = -_1_m * _adt_S_re(H, i_np1 + 1)
                    Sp1pim = _1_m * _adt_S_im(H, i_np1 + 1)
                    Sp1re = _1_m * _adt_S_re(H, i_np1)
                    Sp1im = -_1_m * _adt_S_im(H, i_np1)
                    local Sp1mre::TF, Sp1mim::TF
                    if m == 0
                        Sp1mre = -_1_m * Sp1pre; Sp1mim = _1_m * Sp1pim
                    else
                        Sp1mre = -_1_m * _adt_S_re(H, i_np1 - 1)
                        Sp1mim = _1_m * _adt_S_im(H, i_np1 - 1)
                    end
                    row = flat_basis_index(n, m, 1)
                    CUDA.@atomic lc[row, ia] += _1_np1 * np1_inv * (
                        TF(0.5) * (wy * Sp1mre - wx * Sp1mim) -
                        TF(0.5) * (wy * Sp1pre + wx * Sp1pim) - wz * Sp1re)
                    CUDA.@atomic lc[row + 1, ia] += _1_np1 * np1_inv * (
                        TF(0.5) * (wy * Sp1mim + wx * Sp1mre) -
                        TF(0.5) * (wy * Sp1pim - wx * Sp1pre) - wz * Sp1im)
                end
            end
            s += Int(blockDim().x)
        end
        k += stride
    end
    return nothing
end

function _launch_cuda_adaptive_s2l!(state::DeviceResidentRadixState{TF,B,LH},
        actx::DeviceAdaptiveCUDAContext) where {TF,B,LH}
    n_x = actx.n_x
    n_x == 0 && return state
    orders = state.invariant_cache.basis_info.orders
    P_phi = orders.P_phi
    grid = state.grid::DeviceRadixGrid
    threads = 128
    blocks = min(n_x, _ADT_CUDA_HARMONIC_BLOCKS)
    if state.options.body_type <: Point{Vortex}
        CUDA.@cuda threads=threads blocks=blocks _cuda_adaptive_s2l_vortex_kernel!(
            phi_slab(state.locals), chi_slab(state.locals), state.source_bodies,
            state.cell_ranges, actx.leaf_slot_of::CUDA.CuVector{Int32},
            grid.node_centers, actx.x_targets::CUDA.CuVector{Int32},
            actx.x_sources::CUDA.CuVector{Int32}, n_x,
            actx.harmonics_scratch::CUDA.CuArray{TF,3},
            Val(P_phi), Val(orders.P_active))
    elseif state.options.body_type <: Point{Source}
        CUDA.@cuda threads=threads blocks=blocks _cuda_adaptive_s2l_kernel!(
            phi_slab(state.locals), state.source_bodies, state.cell_ranges,
            actx.leaf_slot_of::CUDA.CuVector{Int32}, grid.node_centers,
            actx.x_targets::CUDA.CuVector{Int32},
            actx.x_sources::CUDA.CuVector{Int32}, n_x,
            actx.harmonics_scratch::CUDA.CuArray{TF,3}, Val(P_phi))
    else
        throw(ArgumentError("adaptive S2L supports Point{Source} and " *
            "Point{Vortex}; got $(state.options.body_type)"))
    end
    return state
end

#------- adaptive V-list M2L over the unchanged resident plans -------#

# Device mirror of the host _launch_adaptive_resident_m2l!: the class-
# partitioned CSR stream feeds the UNCHANGED window plans. The dense CUDA
# family applies whole level segments in place (per-offset class ids +
# per-level expansion scales — no window copies at all); the precomputed-y and
# concat families walk device-to-device windows exactly like the host driver.
# No new operator tables; no transfers.
function _launch_cuda_adaptive_m2l!(state::DeviceResidentRadixState{TF,B,LH},
        actx::DeviceAdaptiveCUDAContext) where {TF,B,LH}
    ws = state.scratch
    ws isa ResidentOperatorWorkspace ||
        throw(ArgumentError("adaptive M2L requires ResidentOperatorWorkspace scratch"))
    plan = ws.m2l_concat
    plan isa Union{ResidentM2LConcatPlan,ResidentM2LPrecomputedYPlan,
        ResidentM2LDenseCUDAPlan} || throw(ArgumentError(
        "adaptive device M2L requires a concat, precomputed-y, or dense CUDA " *
        "window plan; got $(typeof(plan))"))
    fill!(state.locals.phi, zero(TF))
    LH && fill!(state.locals.chi, zero(TF))
    nroutes = actx.n_routes
    nroutes == 0 && (state.counts.n_routes = 0; return state)
    ell_max = (actx.policy::AdaptiveTreePolicy).ell_max
    if plan isa ResidentM2LDenseCUDAPlan
        ls = actx.level_starts
        rco = actx.route_class_offset::CUDA.CuVector{Int32}
        rsrc = actx.route_sources::CUDA.CuVector{Int}
        rtgt = actx.route_targets::CUDA.CuVector{Int}
        for L in actx.first_m2l_level:ell_max
            s = ls[L + 1]
            e = ls[L + 2]
            n = e - s
            n > 0 && _cuda_hier_dense_apply_routes!(state, ws, plan, actx, L,
                view(rco, s:(e - 1)), view(rsrc, s:(e - 1)),
                view(rtgt, s:(e - 1)), n)
        end
        state.counts.n_routes = nroutes
        return state
    end
    window = actx.window_capacity
    i = 1
    while i <= nroutes
        count = min(window, nroutes - i + 1)
        copyto!(state.route_targets, 1, actx.route_targets::CUDA.CuVector{Int},
            i, count)
        copyto!(state.route_sources, 1, actx.route_sources::CUDA.CuVector{Int},
            i, count)
        copyto!(plan.route_class, 1, actx.route_class::CUDA.CuVector{Int32},
            i, count)
        state.counts.n_routes = count
        if plan isa ResidentM2LPrecomputedYPlan
            _cuda_refresh_precomputed_y_m2l_routes!(plan, plan.route_class, count)
            _launch_resident_m2l_precomputed_y_plan!(state, ws, plan;
                clear_locals=false)
        else
            _launch_resident_m2l_concat!(state; clear_locals=false)
        end
        i += count
    end
    state.counts.n_routes = nroutes
    return state
end

#------- adaptive lifecycle body + graph capture -------#

function _cuda_adaptive_lifecycle_body!(state::DeviceResidentRadixState,
        actx::DeviceAdaptiveCUDAContext)
    nearfield_done = radix_setting(:CUDA_OVERLAP_NEARFIELD) ?
        _launch_cuda_nearfield_async!(state) : nothing
    _launch_cuda_b2m!(state)
    _assert_cuda_resident_stage!(state, :b2m)
    # adaptive M2M: per-level edge groups WITHOUT the uniform nonleaf prefix
    # zeroing (which would zero coarse adaptive leaves); B2M refilled the
    # whole multipole buffer above (task 040 semantics)
    _launch_adaptive_resident_m2m!(state)
    _assert_cuda_resident_stage!(state, :m2m)
    _launch_cuda_adaptive_m2l!(state, actx)
    _launch_cuda_adaptive_s2l!(state, actx)
    _assert_cuda_resident_stage!(state, :m2l)
    _launch_resident_l2l!(state)
    _assert_cuda_resident_stage!(state, :l2l)
    if nearfield_done === nothing
        _launch_cuda_resident_l2b!(state)
    else
        _launch_cuda_resident_l2b_only!(state, nearfield_done)
    end
    _launch_cuda_adaptive_m2t!(state, actx)
    _assert_cuda_resident_stage!(state, :l2b)
    return state
end

# Graph eligibility mirrors the uniform rule: the dense fused family's body is
# sync-free and capacity-static within an occupancy epoch (the adaptive CSR
# stream is fully materialized on device — it IS the window cache). The
# precomputed-y/concat windows perform per-window host work and stay uncaptured,
# exactly as on the uniform path.
function _cuda_adaptive_graph_eligible(state::DeviceResidentRadixState,
        actx::DeviceAdaptiveCUDAContext)
    radix_setting(:CUDA_GRAPH_LIFECYCLE) || return false
    actx.graph_warm_epoch == typemin(Int) && return false
    ws = state.scratch
    ws isa ResidentOperatorWorkspace || return false
    ws.m2l_concat isa ResidentM2LDenseCUDAPlan && radix_setting(:DENSE_CUDA_FUSED) || return false
    actx.profile_stages && return false
    DEBUG[] && return false
    return true
end

function run_cuda_adaptive_radix_lifecycle!(state::DeviceResidentRadixState,
        actx::DeviceAdaptiveCUDAContext)
    _require_cuda_radix_available()
    _assert_cuda_supported_operator!(state.options)
    state.counters.expansion_host_copies == 0 ||
        throw(AssertionError("adaptive CUDA radix lifecycle observed expansion host copies before execution"))
    _cuda_adaptive_graph_eligible(state, actx) &&
        return _run_cuda_adaptive_lifecycle_graph!(state, actx)
    return _cuda_adaptive_lifecycle_body!(state, actx)
end

function _run_cuda_adaptive_lifecycle_graph!(state::DeviceResidentRadixState,
        actx::DeviceAdaptiveCUDAContext)
    exec = actx.graph_exec
    if exec !== nothing && actx.graph_epoch == actx.epoch_id
        CUDA.launch(exec::CUDA.CuGraphExec)
        return state
    end
    if actx.graph_warm_epoch != actx.epoch_id
        _cuda_adaptive_lifecycle_body!(state, actx)
        actx.graph_warm_epoch = actx.epoch_id
        return state
    end
    graph = try
        CUDA.capture(; throw_error=false) do
            _cuda_adaptive_lifecycle_body!(state, actx)
        end
    catch err
        err isa CUDA.CuError || rethrow()
        actx.graph_warm_epoch = typemin(Int)
        nothing
    end
    if graph === nothing
        _cuda_adaptive_lifecycle_body!(state, actx)
        return state
    end
    actx.graph_exec = CUDA.instantiate(graph)
    actx.graph_epoch = actx.epoch_id
    CUDA.launch(actx.graph_exec::CUDA.CuGraphExec)
    return state
end

#------- adaptive cache build + per-step refresh -------#

# Device analog of the host _allocate_adaptive_resident_lifecycle: allocate the
# adaptive context + device state over the UNCHANGED workspace/plan machinery.
function _cuda_allocate_adaptive_lifecycle(::Type{TF},
        basis_info::OperatorBasisInfo{B,LH}, options::CUDARadixLifecycleOptions,
        policy::AdaptiveTreePolicy, x_min::SVector{3,TF}, h0::TF, maxn::Int,
        dpb::Int, hessian::Bool, invariant::OperatorInvariantCache,
        counters::CUDARadixTransferCounters, ctx;
        sfs::Bool=false, sfs_transposed::Bool=true,
        sfs_active_row::Int=0) where {TF,B,LH}
    actx = _cuda_allocate_adaptive_context(TF, policy, maxn, x_min, h0, counters)
    # alias the ordinal-indexed body maps filled by the shared position collector
    grid = actx.grid::DeviceRadixGrid
    grid.body_system = ctx.grid.body_system
    grid.body_index = ctx.grid.body_index
    ell = policy.ell_max
    node_cap = actx.node_capacity
    leaf_cap = actx.leaf_capacity
    window_cap = actx.window_capacity
    tables = actx.tables::RigidHierarchicalTables
    _, _, effective_offsets = _hierarchical_class_metadata(tables, ell,
        actx.first_m2l_level)
    multipoles = _cuda_flat_buffer(TF, basis_info, node_cap)
    locals_buf = _cuda_flat_buffer(TF, basis_info, node_cap)
    fill!(multipoles.phi, zero(TF)); fill!(multipoles.chi, zero(TF))
    fill!(locals_buf.phi, zero(TF)); fill!(locals_buf.chi, zero(TF))
    specialized = options.m2l_strategy isa Union{PrecomputedFactoredYM2L,
        DenseTranslationM2L}
    ws_strategy = specialized ? options.m2l_strategy : ConcatenatedFixedZM2L()
    ws_operator = specialized ? options.operator : MaterializedYRotationM2L()
    # The dense CUDA table is stored per UNSCALED union offset and level-scaled
    # per apply (the uniform device convention; per-offset class ids +
    # source/target scales); every other plan is built over the (level, offset)
    # effective classes exactly like the host adaptive workspace.
    plan_offsets = options.m2l_strategy isa DenseTranslationM2L ?
        tables.push_offsets : effective_offsets
    workspace = _radix_cache_workspace(TF, basis_info, multipoles, ell, h0,
        leaf_cap, node_cap, window_cap, plan_offsets, invariant,
        ws_strategy, ws_operator; compact_cuda_factored=true,
        hierarchical_noffsets=actx.noffsets,
        ell_axes=SVector(ell, ell, ell), first_level=0)
    if workspace.m2l_concat isa ResidentM2LFactoredPlan
        _pin_host_array(workspace.m2l_concat.host_class_counts)
        _cuda_factored_whole_pass_setup!(workspace.m2l_concat, TF, basis_info)
    elseif workspace.m2l_concat isa ResidentM2LPrecomputedYPlan
        _pin_host_array(workspace.m2l_concat.host_class_counts)
        _cuda_precomputed_y_whole_pass_setup!(workspace.m2l_concat, TF, basis_info)
    elseif workspace.m2l_concat isa ResidentM2LDenseCUDAPlan
        # dense family: per-level expansion scales (025 level-scaling law),
        # built at the leaf reference depth ell_max
        hs_scale, ht_scale = _cuda_hier_dense_scales(TF, basis_info, ell,
            workspace.m2l_concat.ndof, actx.first_m2l_level)
        actx.source_scale = CUDA.CuArray{TF}(hs_scale)
        actx.target_scale = CUDA.CuArray{TF}(ht_scale)
    end
    counters.operator_uploads += 1     # construction-only operator upload
    # M2T/S2L per-thread irregular-harmonic scratch (2 x slots x NH at order
    # P_phi + 2); slot count = the fixed harmonic grid (blocks x 128 threads)
    P_phi = basis_info.orders.P_phi
    nH2 = harmonic_index(P_phi + 2, P_phi + 2)
    actx.harmonics_scratch = CUDA.zeros(TF, 2, _ADT_CUDA_HARMONIC_BLOCKS * 128, nH2)
    source_bodies = CUDA.zeros(TF, dpb, maxn)
    n_output_rows = hessian ? 13 : 4
    output = CUDA.zeros(TF, n_output_rows, maxn)
    route_levels = CUDA.zeros(Int, window_cap)
    route_offsets = CUDA.zeros(Int, 3, window_cap)
    route_targets = CUDA.zeros(Int, window_cap)
    route_sources = CUDA.zeros(Int, window_cap)
    direct_targets = CUDA.zeros(Int, actx.u_capacity)
    direct_sources = CUDA.zeros(Int, actx.u_capacity)
    edge_placeholder = CUDA.zeros(Int, 0)
    # task 048: the adaptive lifecycle owns its own SFS accumulators (its body
    # sort differs from the uniform grid's, so slabs are not shared)
    sfs_device_ctx = sfs ?
        (; tg=CUDA.zeros(TF, 3, maxn), om=CUDA.zeros(TF, 3, maxn),
           q=CUDA.zeros(TF, 3, maxn), transposed=sfs_transposed,
           active_row=sfs_active_row) : nothing
    state = DeviceResidentRadixState{TF,CompressedComplexBasis,LH}(
        grid, actx, source_bodies, source_bodies,
        grid.perm, grid.body_system, grid.body_index,
        ctx.host_perm, ctx.host_body_system, ctx.host_body_index,
        nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing,
        nothing,
        grid.cell_centers, grid.cell_ranges,
        edge_placeholder, edge_placeholder, edge_placeholder, edge_placeholder,
        multipoles, locals_buf,
        route_levels, route_offsets, route_targets, route_sources,
        direct_targets, direct_sources, output,
        invariant, workspace, counters, options,
        RadixStepCounts(0, 0, 0, 0, 0);
        sfs=sfs_device_ctx,
    )
    # side-stream nearfield ordering is event-based; disable implicit per-array
    # cross-stream synchronization exactly as on the uniform path (illegal
    # inside stream capture)
    for arr in (output, source_bodies, grid.cell_ranges, direct_targets,
            direct_sources)
        CUDA.enable_synchronization!(arr, false)
    end
    return actx, state
end

# Per-step refresh of the adaptive device cache: single full-depth sort +
# device tree rebuild + (on occupancy epochs) DTR lists, CSR partition, U slot
# mapping, and stage-group refresh. The uniform grid/route machinery does NOT
# run on this path (the 040 double-refresh lever lands on device by
# construction). The global geometry gate is replaced by the per-cell sticky
# demotion (armed at construction for regularized kernels).
function _cuda_update_adaptive_radix_state!(cache::RadixFMMCache{TF,LH},
        systems::Tuple) where {TF,LH}
    ctx = cache.device_ctx
    actx = cache.adaptive_tree::DeviceAdaptiveCUDAContext
    state = cache.adaptive_state::DeviceResidentRadixState
    length(systems) == cache.n_systems ||
        throw(ArgumentError("cache was built for $(cache.n_systems) source systems, got $(length(systems))"))
    n = get_n_bodies(systems)
    n > 0 || throw(ArgumentError("update requires at least one body"))
    n <= cache.max_n_bodies ||
        throw(ArgumentError("n=$n exceeds the cache capacity max_n_bodies=$(cache.max_n_bodies)"))
    source_buffers = _radix_cache_refresh_source_buffers!(ctx, systems, TF)
    _radix_cache_collect_positions!(ctx, source_buffers)
    grid = actx.grid::DeviceRadixGrid
    occ_changed = _cuda_refresh_adaptive_tree!(ctx, actx, cache, n)
    _pack_radix_body_matrix!(state.source_bodies, source_buffers,
        view(grid.perm, 1:n), grid.body_system, grid.body_index)
    policy = actx.policy::AdaptiveTreePolicy
    actx.sigma_armed = policy.sigma_row > 0 && policy.rho_t > 0
    actx.sigma_armed && _cuda_adaptive_sigma_sweep!(actx, state.source_bodies)
    if occ_changed
        _cuda_refresh_adaptive_lists!(actx, state.direct_targets,
            state.direct_sources)
        profile = actx.profile_stages
        t0 = profile ? (CUDA.synchronize(); time_ns()) : UInt64(0)
        _cuda_refresh_resident_stage_groups!(
            state.scratch::ResidentOperatorWorkspace, grid, actx.level_offsets,
            policy.ell_max, 0)
        profile && (CUDA.synchronize(); actx.stage_ns[8] = time_ns() - t0)
    end
    if _radix_any_host_resident(systems)
        copyto!(ctx.host_perm, 1, grid.perm, 1, n)
        copyto!(ctx.host_body_system, 1, grid.body_system, 1, n)
        copyto!(ctx.host_body_index, 1, grid.body_index, 1, n)
        ctx.counters.metadata_downloads += 3
    end
    counts = state.counts
    counts.n_bodies = n
    counts.n_cells = actx.n_leaves
    counts.n_nodes = actx.n_nodes
    counts.n_routes = actx.n_routes
    counts.n_direct = actx.n_u
    cache.step += 1
    return cache
end

#------- RECTANGULAR DIRECT-EVALUATION KERNELS (task 051 stage 1) -------#
#
# Device counterparts of src/direct_rectangular.jl: tiled
# source-in-shared-memory rectangular sweep (041k tiled pattern, see
# MATRIX_OPERATOR_REFACTOR/scripts/fm041k_direct_bruteforce_gpu.jl), one
# thread per target, block-level grid stride over target chunks, no atomics
# (each target column owned by exactly one thread). Pair math is the SAME
# inlined host functions (_rect_point_pair / _rect_panel_pair), so host and
# device agree to summation-order roundoff. Precision-generic: F64 primary;
# pass Float32 CuMatrices for the F32 point variant.
#
# WRITTEN BLIND (no CUDA hardware on the dev machine): parse-checked only,
# following the existing kernel idioms in this file. Verify on the cluster via
# FLOWVPM.jl/scripts/fm051_rect_bench.jl before trusting numbers.

const _RECT_TILE_POINTS = 256
const _RECT_TILE_PANELS = 128
const _RECT_MAX_BLOCKS = 65535

function _cuda_rect_points_kernel!(out, targets, sources, n_targets, n_sources,
        ::Val{GRAD}) where GRAD
    T = eltype(out)
    tid = threadIdx().x
    sh = CUDA.CuStaticSharedArray(T, (7, _RECT_TILE_POINTS))
    nchunks = cld(n_targets, _RECT_TILE_POINTS)
    ntiles = cld(n_sources, _RECT_TILE_POINTS)
    chunk = blockIdx().x
    while chunk <= nchunks
        i = (chunk - 1) * _RECT_TILE_POINTS + tid
        active = i <= n_targets
        tx = ty = tz = zero(T)
        if active
            @inbounds begin
                tx = targets[1, i]; ty = targets[2, i]; tz = targets[3, i]
            end
        end
        u1 = u2 = u3 = zero(T)
        j1 = j2 = j3 = j4 = j5 = j6 = j7 = j8 = j9 = zero(T)
        for t in 1:ntiles
            q0 = (t - 1) * _RECT_TILE_POINTS
            ql = q0 + tid
            if ql <= n_sources
                @inbounds for r in 1:7
                    sh[r, tid] = sources[r, ql]
                end
            end
            CUDA.sync_threads()
            if active
                @inbounds for k in 1:min(_RECT_TILE_POINTS, n_sources - q0)
                    Ux, Uy, Uz, a1, a2, a3, a4, a5, a6, a7, a8, a9 =
                        _rect_point_pair(RectangularGaussianErfVortex(), tx, ty, tz,
                            sh[1, k], sh[2, k], sh[3, k],
                            sh[4, k], sh[5, k], sh[6, k], sh[7, k], Val(GRAD))
                    u1 += Ux; u2 += Uy; u3 += Uz
                    if GRAD
                        j1 += a1; j2 += a2; j3 += a3; j4 += a4; j5 += a5
                        j6 += a6; j7 += a7; j8 += a8; j9 += a9
                    end
                end
            end
            CUDA.sync_threads()
        end
        if active
            @inbounds begin
                out[1, i] += u1; out[2, i] += u2; out[3, i] += u3
                if GRAD
                    out[4, i] += j1; out[5, i] += j2; out[6, i] += j3
                    out[7, i] += j4; out[8, i] += j5; out[9, i] += j6
                    out[10, i] += j7; out[11, i] += j8; out[12, i] += j9
                end
            end
        end
        chunk += gridDim().x
    end
    return nothing
end

function _cuda_rect_panels_kernel!(out, targets, sources, n_targets, n_sources,
        ::Val{GRAD}, ::Val{POT}, ::Val{REG}=Val(1)) where {GRAD,POT,REG}
    T = eltype(out)
    tid = threadIdx().x
    sh = CUDA.CuStaticSharedArray(T, (17, _RECT_TILE_PANELS))
    nchunks = cld(n_targets, _RECT_TILE_PANELS)
    ntiles = cld(n_sources, _RECT_TILE_PANELS)
    chunk = blockIdx().x
    while chunk <= nchunks
        i = (chunk - 1) * _RECT_TILE_PANELS + tid
        active = i <= n_targets
        target = zero(SVector{3,T})
        if active
            @inbounds target = SVector{3,T}(targets[1, i], targets[2, i], targets[3, i])
        end
        u = zero(SVector{3,T})
        g = zero(SMatrix{3,3,T,9})
        p = zero(T)
        for t in 1:ntiles
            q0 = (t - 1) * _RECT_TILE_PANELS
            ql = q0 + tid
            if ql <= n_sources
                @inbounds for r in 1:17
                    sh[r, tid] = sources[r, ql]
                end
            end
            CUDA.sync_threads()
            if active
                @inbounds for k in 1:min(_RECT_TILE_PANELS, n_sources - q0)
                    # rows 1:2 are validated integral/in-range host-side by
                    # _rect_check_args; unchecked truncation avoids the
                    # InexactError trap path inside the pair loop
                    tag = unsafe_trunc(Int, sh[1, k])
                    nv = unsafe_trunc(Int, sh[2, k])
                    v1 = SVector{3,T}(sh[3, k], sh[4, k], sh[5, k])
                    v2 = SVector{3,T}(sh[6, k], sh[7, k], sh[8, k])
                    v3 = SVector{3,T}(sh[9, k], sh[10, k], sh[11, k])
                    v4 = SVector{3,T}(sh[12, k], sh[13, k], sh[14, k])
                    s1 = sh[15, k]
                    s2 = sh[16, k]
                    koff = sh[17, k]
                    uq, gq = _rect_panel_pair(RectangularPanelInfluence(), target,
                        tag, nv, v1, v2, v3, v4, s1, s2, koff, Val(GRAD), Val(REG))
                    u += uq
                    if GRAD
                        g += gq
                    end
                    if POT
                        p += _rect_panel_potential(target, tag, nv,
                            v1, v2, v3, s1, s2)
                    end
                end
            end
            CUDA.sync_threads()
        end
        if active
            @inbounds begin
                out[1, i] += u[1]; out[2, i] += u[2]; out[3, i] += u[3]
                if GRAD
                    for j in 1:3, kk in 1:3
                        out[3 + (j-1)*3 + kk, i] += g[kk, j]
                    end
                end
                POT && (out[rect_potential_row(GRAD), i] += p)
            end
        end
        chunk += gridDim().x
    end
    return nothing
end

function direct_rectangular!(out::CUDA.CuMatrix{T}, targets::CUDA.CuMatrix{T},
        kernel::RectangularGaussianErfVortex, sources::CUDA.CuMatrix{T};
        gradient::Bool=false, scalar_potential::Bool=false) where T
    _rect_check_args(out, targets, kernel, sources, gradient, scalar_potential)
    n_targets = size(targets, 2)
    n_sources = size(sources, 2)
    n_targets == 0 && return out
    blocks = min(cld(n_targets, _RECT_TILE_POINTS), _RECT_MAX_BLOCKS)
    if gradient
        CUDA.@cuda threads=_RECT_TILE_POINTS blocks=blocks _cuda_rect_points_kernel!(
            out, targets, sources, n_targets, n_sources, Val(true))
    else
        CUDA.@cuda threads=_RECT_TILE_POINTS blocks=blocks _cuda_rect_points_kernel!(
            out, targets, sources, n_targets, n_sources, Val(false))
    end
    return out
end

function direct_rectangular!(out::CUDA.CuMatrix{T}, targets::CUDA.CuMatrix{T},
        kernel::RectangularPanelInfluence, sources::CUDA.CuMatrix{T};
        gradient::Bool=false, scalar_potential::Bool=false) where T
    _rect_check_args(out, targets, kernel, sources, gradient, scalar_potential)
    n_targets = size(targets, 2)
    n_sources = size(sources, 2)
    n_targets == 0 && return out
    blocks = min(cld(n_targets, _RECT_TILE_PANELS), _RECT_MAX_BLOCKS)
    regv = _rect_reg_val(kernel.filament_reg)
    if gradient
        CUDA.@cuda threads=_RECT_TILE_PANELS blocks=blocks _cuda_rect_panels_kernel!(
            out, targets, sources, n_targets, n_sources, Val(true),
            Val(scalar_potential), regv)
    else
        CUDA.@cuda threads=_RECT_TILE_PANELS blocks=blocks _cuda_rect_panels_kernel!(
            out, targets, sources, n_targets, n_sources, Val(false),
            Val(scalar_potential), regv)
    end
    return out
end
