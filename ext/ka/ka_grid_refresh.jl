#------- in-place grid rebuild, stage 1: keys + sort + leaf-cell compression -------#
#
# First two stages of the in-place grid rebuild on the uniform path of
# `ka_update_radix_state!`. Gated stage by stage
# rather than big-bang: this covers everything through the occupied-leaf-cell
# compression, which is the natural seam -- `perm`/`invperm`/`cell_ranges` are
# functions of the body positions and always refresh, while everything after the
# compression is a pure function of the occupied cell SET and sits behind the
# occupancy-epoch check.
#
# The key kernel here carries an out-of-bounds flag, which a from-scratch build
# could drop. The live
# refresh loop cannot: the fixed Morton box is part of the cache's invariant
# contract, and a body leaving it must throw rather than clamp. Hence a second,
# checked kernel here, which also takes the per-axis `box_extent` (the port
# rectangular geometry) instead of assuming the cubic `2h0`.
#
# Host oracle for the gate: `_radix_fill_body_data!`,
# `_host_radix_sort_permutation` and `_compress_radix_cells`
# (src/tree_batched.jl), which are exactly these three steps on the CPU.

# leaf index of a scaled coordinate, clamped to 0:G-1 without throwing (NaN
# lands on an arbitrary in-range cell; the caller flags it)
@inline _ka_cell_index(s, G) =
    clamp(unsafe_trunc(Int, floor(clamp(s, zero(s), oftype(s, G)))), 0, G - 1)

@kernel function ka_radix_keys_checked_kernel!(keys, oob_flag, @Const(positions),
        x_min, box_extent, h0, ell, n)
    i = @index(Global)
    @inbounds if i <= n
        G = 1 << ell
        delta = (2 * h0) / G
        px = positions[1, i]
        py = positions[2, i]
        pz = positions[3, i]
        # benign-race flag store: any lane observing an escape sets it.
        # Tolerate ulp-scale overshoot: a tight box built as center - h0 can sit
        # a rounding error below the true data max (the clamp handles the key).
        hx = x_min[1] + box_extent[1]
        hy = x_min[2] + box_extent[2]
        hz = x_min[3] + box_extent[3]
        tx = 4 * eps(max(abs(x_min[1]), abs(hx)))
        ty = 4 * eps(max(abs(x_min[2]), abs(hy)))
        tz = 4 * eps(max(abs(x_min[3]), abs(hz)))
        if !(x_min[1] - tx <= px <= hx + tx &&
             x_min[2] - ty <= py <= hy + ty &&
             x_min[3] - tz <= pz <= hz + tz)
            oob_flag[1] = Int32(1)
        end
        # clamp before the integer conversion: a NaN or huge position must reach
        # the flag check on the host, not throw InexactError in the kernel
        ix = _ka_cell_index((px - x_min[1]) / delta, G)
        iy = _ka_cell_index((py - x_min[2]) / delta, G)
        iz = _ka_cell_index((pz - x_min[3]) / delta, G)
        keys[i] = ka_morton_key(ix, iy, iz, ell)
    end
end

@kernel function ka_gather_sorted_keys_kernel!(sorted_keys, @Const(keys),
        @Const(perm), n)
    i = @index(Global)
    @inbounds if i <= n
        sorted_keys[i] = keys[perm[i]]
    end
end

@kernel function ka_key_change_flags_kernel!(flags, @Const(sorted_keys), n)
    i = @index(Global)
    @inbounds if i <= n
        flags[i] = (i == 1 || sorted_keys[i] != sorted_keys[i - 1]) ? 1 : 0
    end
end

@kernel function ka_fill_cell_firsts_kernel!(cell_keys, cell_ranges,
        @Const(sorted_keys), @Const(flags), @Const(prefix), n)
    i = @index(Global)
    @inbounds if i <= n && flags[i] == 1
        icell = prefix[i]
        cell_keys[icell] = sorted_keys[i]
        cell_ranges[1, icell] = i
    end
end

@kernel function ka_fill_cell_counts_kernel!(cell_ranges, @Const(flags),
        @Const(prefix), n)
    # `cell_ranges[1, :]` must be written by a prior launch; the launch boundary
    # is the synchronization this read needs (same contract as the CUDA kernel).
    i = @index(Global)
    @inbounds if i <= n && (i == n || flags[i + 1] == 1)
        icell = prefix[i]
        first = cell_ranges[1, icell]
        cell_ranges[2, icell] = i - first + 1
    end
end

"""
    ka_radix_keys_checked!(keys, oob_flag, host_oob, positions, x_min, box_extent,
                           h0, ell; workgroup=KA_AUTO_WORKGROUP)

Backend-agnostic port of `_cuda_radix_keys_checked_kernel!` plus its host-side
out-of-bounds check. Writes the full-depth (`ell`-level) Morton key of each of
the `length(keys)` bodies and throws `ArgumentError` if any body lies outside the
fixed box `[x_min, x_min + box_extent]`. `keys` is scratch on the caller's side,
so throwing here leaves the persistent grid at its previous consistent step.
"""
function ka_radix_keys_checked!(keys, oob_flag, host_oob, positions, x_min,
        box_extent, h0, ell::Int; workgroup=KA_AUTO_WORKGROUP)
    n = length(keys)
    n == 0 && return keys
    backend = KA.get_backend(keys)
    fill!(oob_flag, Int32(0))
    kernel = _cached_kernel(ka_radix_keys_checked_kernel!, backend, workgroup)
    kernel(keys, oob_flag, positions, x_min, box_extent, h0, ell, n; ndrange=n)
    KA.synchronize(backend)
    copyto!(host_oob, oob_flag)
    if host_oob[1] != 0
        x_max = x_min .+ box_extent
        throw(ArgumentError(
            "at least one body lies outside the fixed RadixFMMCache box " *
            "[$(Tuple(x_min)), $(Tuple(x_max))]; the box is part of the cache's " *
            "invariant contract — construct a new cache (or pass explicit " *
            "bounds=(x_min, box_size) covering the trajectory)"))
    end
    return keys
end

#------- bounded-key counting sort: the stage 6 fast path -------#
#
# The cache's fixed Morton depth bounds
# keys to `0:2^(3ell)-1`, so a histogram over the whole key domain plus one scan
# and an atomic-cursor scatter replaces the comparison sort.
#
# This path is deliberately UNSTABLE, exactly as CUDA's is: the scatter claims
# its slot with an atomic cursor, so bodies sharing a cell land in an order that
# varies between otherwise identical runs. That is the CUDA behavior being
# matched, and it is why this sits behind the same runtime gate rather than
# simply replacing `sortperm!`. Two consequences, both inherited from CUDA:
#
#   * within-cell body order is not reproducible run to run, so neither is the
#     summation order of the same-cell nearfield atomics -- identical inputs
#     move in the last bits between runs;
#   * `perm` can no longer be compared elementwise against the stable host sort.
#     Everything downstream still is exact: `cell_keys`, `cell_ranges`, cell
#     centers and the whole node table are pure functions of the occupied-cell
#     SET, not of within-cell ordering.
#
# The gate mirrors CUDA's two conditions (`_cuda_counting_sort_ready`): `ell`
# must be within `KA_COUNTING_SORT_MAX_ELL`, AND the histogram actually handed in
# must span the key domain -- otherwise `@inbounds` atomics would run through a
# length-1 array. Falling back is always safe.

@kernel function ka_counting_histogram_kernel!(histogram, @Const(keys), n)
    i = @index(Global)
    @inbounds if i <= n
        KA.@atomic histogram[Int(keys[i]) + 1] += Int32(1)
    end
end

@kernel function ka_counting_cursor_kernel!(cursor, @Const(prefix), n)
    k = @index(Global)
    @inbounds if k <= n
        cursor[k] = k == 1 ? Int32(0) : prefix[k - 1]
    end
end

@kernel function ka_counting_scatter_kernel!(perm, sorted_keys, cursor, @Const(keys), n)
    i = @index(Global)
    @inbounds if i <= n
        key = keys[i]
        # CUDA calls `atomic_add!`, which returns the OLD value, and takes
        # `old + 1` as the 1-based slot. KA's `@atomic x += v` returns the NEW
        # value, which is that same slot -- verified on Metal, not assumed.
        slot = KA.@atomic cursor[Int(key) + 1] += Int32(1)
        perm[Int(slot)] = i
        sorted_keys[Int(slot)] = key
    end
end

# Deepest uniform level whose full 8^ell key-domain histogram is allocated for
# the counting sort; deeper grids take the stable `sortperm!` path.
const KA_COUNTING_SORT_MAX_ELL = 6

@inline ka_counting_sort_enabled(ell::Int) = ell <= KA_COUNTING_SORT_MAX_ELL

@inline ka_counting_sort_ready(histogram, ell::Int) =
    histogram !== nothing && ell >= 0 && ka_counting_sort_enabled(ell) &&
        length(histogram) == 1 << (3 * ell)

"""
    ka_counting_sort_into!(perm, sorted_keys, keys, histogram, prefix, cursor;
                           workgroup=KA_AUTO_WORKGROUP)

Backend-agnostic port of `_cuda_counting_sort_into!`: histogram the bounded
Morton keys, inclusive-scan the histogram, shift it into an exclusive cursor,
then scatter each body into the slot its atomic cursor claims. `histogram`,
`prefix` and `cursor` must each span the full `2^(3ell)` key domain. Unstable by
construction -- see the note above.
"""
function ka_counting_sort_into!(perm, sorted_keys, keys, histogram, prefix, cursor;
        workgroup=KA_AUTO_WORKGROUP)
    n = length(keys)
    n == 0 && return perm
    backend = KA.get_backend(keys)
    fill!(histogram, Int32(0))
    hk = _cached_kernel(ka_counting_histogram_kernel!, backend, workgroup)
    hk(histogram, keys, n; ndrange=n)
    accumulate!(+, prefix, histogram)
    nd = length(histogram)
    ck = _cached_kernel(ka_counting_cursor_kernel!, backend, workgroup)
    ck(cursor, prefix, nd; ndrange=nd)
    sc = _cached_kernel(ka_counting_scatter_kernel!, backend, workgroup)
    sc(perm, sorted_keys, cursor, keys, n; ndrange=n)
    return perm
end

"""
    ka_radix_sort_bodies!(perm, sorted_keys, invperm, keys; workgroup=KA_AUTO_WORKGROUP,
                          ell=-1, histogram=nothing, prefix=nothing, cursor=nothing)

Sort the bodies by Morton key: `perm` receives the sorting permutation,
`sorted_keys` the gathered keys, `invperm` the scatter inverse. Port of the
`_cuda_sortperm_into!` / `_cuda_gather_sorted_keys_kernel!` /
`_cuda_fill_invperm_kernel!` triple, and -- when `ell` and the counting buffers
are supplied and the gate passes -- of CUDA's bounded counting-sort fast path
too, branching exactly where `_cuda_update_radix_grid_in_place!` branches.

Callers that omit the counting buffers (the isolated correctness suites) keep
the stable `sortperm!` path, which is what makes an elementwise `perm`
comparison against the host sort meaningful for them.
"""
function ka_radix_sort_bodies!(perm, sorted_keys, invperm, keys;
        workgroup=KA_AUTO_WORKGROUP, ell::Int=-1,
        histogram=nothing, prefix=nothing, cursor=nothing)
    n = length(keys)
    n == 0 && return perm
    backend = KA.get_backend(keys)
    if ka_counting_sort_ready(histogram, ell)
        ka_counting_sort_into!(perm, sorted_keys, keys, histogram, prefix, cursor;
            workgroup)
    else
        sortperm!(perm, keys)
        gather = _cached_kernel(ka_gather_sorted_keys_kernel!, backend, workgroup)
        gather(sorted_keys, keys, perm, n; ndrange=n)
    end
    ka_fill_invperm!(invperm, perm; workgroup)
    KA.synchronize(backend)
    return perm
end

"""
    ka_radix_compress_cells!(cell_keys, cell_ranges, sorted_keys, flags, prefix,
                             host_scalar; workgroup=KA_AUTO_WORKGROUP)

Compress the sorted body keys into occupied leaf cells, writing `cell_keys` and
`cell_ranges` (row 1 = first sorted body index, row 2 = body count) and
returning `n_cells`. Port of the `_cuda_key_change_flags_kernel!` /
`accumulate!` / `_cuda_fill_cell_firsts_kernel!` /
`_cuda_fill_cell_counts_kernel!` block. `flags`/`prefix` are caller-owned
`1:n` scratch views; `host_scalar` is a 1-element host vector for the count
download, the single unavoidable sync point (the caller needs `n_cells` to
bounds-check against the cache's cell capacity).
"""
function ka_radix_compress_cells!(cell_keys, cell_ranges, sorted_keys, flags,
        prefix, host_scalar; workgroup=KA_AUTO_WORKGROUP)
    n = length(sorted_keys)
    n == 0 && return 0
    backend = KA.get_backend(sorted_keys)
    flagk = _cached_kernel(ka_key_change_flags_kernel!, backend, workgroup)
    flagk(flags, sorted_keys, n; ndrange=n)
    accumulate!(+, prefix, flags)
    KA.synchronize(backend)
    copyto!(host_scalar, 1, prefix, n, 1)
    n_cells = Int(host_scalar[1])
    firstsk = _cached_kernel(ka_fill_cell_firsts_kernel!, backend, workgroup)
    firstsk(cell_keys, cell_ranges, sorted_keys, flags, prefix, n; ndrange=n)
    countsk = _cached_kernel(ka_fill_cell_counts_kernel!, backend, workgroup)
    countsk(cell_ranges, flags, prefix, n; ndrange=n)
    KA.synchronize(backend)
    return n_cells
end


#------- in-place grid rebuild, stage 2: occupancy-epoch check + cell centers -------#
#
# Third stage of the in-place grid rebuild, directly after the leaf-cell compression
# ported in stage 1. This is the seam the epoch check defines: everything from
# here on -- cell centers, per-level unique node keys, node geometry/parent/child
# topology, leaf_to_node -- is a pure function of the occupied leaf-cell SET
# inside the cache's fixed Morton box, so when the sorted unique keys match the
# previous step's snapshot exactly the whole node-metadata rebuild is skipped
# and the persistent arrays stay valid. The compare costs one kernel plus one
# 4-byte D2H, replacing ~40 launches, several device scans and two blocking
# downloads on the steady occupancy-static step.
#
# Host oracle for the gate: the cell-center loop of `_refresh_radix_grid!`
# (src/tree_batched.jl), plus `morton_decode` for the integer coords, which
# the host grid does not store (`ctx.cell_coords` is device-side only).

@kernel function ka_keys_differ_kernel!(flag, @Const(keys), @Const(snapshot), n)
    i = @index(Global)
    @inbounds if i <= n && keys[i] != snapshot[i]
        # benign-race flag store: any lane observing a difference sets it
        flag[1] = Int32(1)
    end
end

@kernel function ka_cell_centers_kernel!(centers, coords, @Const(cell_keys),
        x_min, h0, ell, n_cells)
    icell = @index(Global)
    @inbounds if icell <= n_cells
        TF = eltype(centers)
        delta = (2 * h0) / (1 << ell)
        ix, iy, iz = ka_decode_morton_key(cell_keys[icell], ell)
        coords[1, icell] = ix
        coords[2, icell] = iy
        coords[3, icell] = iz
        centers[1, icell] = x_min[1] + delta * (TF(ix) + TF(0.5))
        centers[2, icell] = x_min[2] + delta * (TF(iy) + TF(0.5))
        centers[3, icell] = x_min[3] + delta * (TF(iz) + TF(0.5))
    end
end

"""
    ka_radix_occupancy_changed!(flag, host_flag, cell_keys, snapshot, n_cells;
                                workgroup=KA_AUTO_WORKGROUP)

Backend-agnostic port of the `_cuda_keys_differ_kernel!` compare: returns `true`
when the first `n_cells` occupied leaf-cell keys differ anywhere from
`snapshot`. Both key arrays are ascending and of equal length by the caller's
own `n`/`n_cells` precheck, so an elementwise compare is a set compare. `flag`
is a 1-element device buffer, `host_flag` its 1-element host mirror.
"""
function ka_radix_occupancy_changed!(flag, host_flag, cell_keys, snapshot,
        n_cells::Int; workgroup=KA_AUTO_WORKGROUP)
    n_cells == 0 && return false
    backend = KA.get_backend(cell_keys)
    fill!(flag, Int32(0))
    kernel = _cached_kernel(ka_keys_differ_kernel!, backend, workgroup)
    kernel(flag, cell_keys, snapshot, n_cells; ndrange=n_cells)
    KA.synchronize(backend)
    copyto!(host_flag, flag)
    return host_flag[1] != Int32(0)
end

"""
    ka_radix_cell_centers!(centers, coords, cell_keys, x_min, h0, ell, n_cells;
                           workgroup=KA_AUTO_WORKGROUP)

Port of `_cuda_cell_centers_kernel!`: decode each occupied leaf cell's Morton
key into its integer grid coordinate (`coords`) and the cell's physical center
(`centers`). `x_min` is a plain host scalar triple (an `SVector{3,TF}`), `h0`
the box half-width; the `0.5` is `TF`-typed, since a bare literal would promote
the whole expression to `Float64` and fail to compile on Metal.
"""
function ka_radix_cell_centers!(centers, coords, cell_keys, x_min, h0, ell::Int,
        n_cells::Int; workgroup=KA_AUTO_WORKGROUP)
    n_cells == 0 && return centers
    backend = KA.get_backend(cell_keys)
    kernel = _cached_kernel(ka_cell_centers_kernel!, backend, workgroup)
    kernel(centers, coords, cell_keys, x_min, h0, ell, n_cells; ndrange=n_cells)
    KA.synchronize(backend)
    return centers
end



#------- in-place grid rebuild, stage 3: per-level unique node keys -------#
#
# Fourth stage of the in-place grid rebuild, the first block behind the stage-2
# occupancy-epoch check. `cell_keys` is ascending and a right shift is monotone,
# so each level's ancestor keys are *already sorted* -- no per-level sort is
# needed, and the same flag/scan/compact triple that compressed bodies into
# leaf cells in stage 1 compresses each level's ancestor keys into that level's
# unique nodes. Only the counts round-trip to the host, because the level
# offsets are a running host-side prefix that the node-capacity check and the
# stage-4 launch geometry (`max_count`) both read.
#
# Levels below the cache root are trimmed (): never keyed, never
# built. `ka_gather_level_counts_kernel!` still reads every column, including
# the unfilled prefix columns of the trimmed levels, exactly as the CUDA kernel
# does -- the host loop zeroes those counts immediately after the download, so
# the garbage never reaches an offset.
#
# Host oracle for the gate: `_refresh_radix_nodes!` (src/tree_batched.jl), whose
# count and fill loops are what `_radix_grid` runs on the CPU.

@kernel function ka_leaf_ancestor_keys_kernel!(ancestor_keys, @Const(cell_keys),
        leaf_level, level, n_cells)
    i = @index(Global)
    @inbounds if i <= n_cells
        ancestor_keys[i] = cell_keys[i] >> (3 * (leaf_level - level))
    end
end

@kernel function ka_gather_level_counts_kernel!(level_counts, @Const(level_prefix),
        n_cells, n_levels)
    l = @index(Global)
    @inbounds if l <= n_levels
        level_counts[l] = n_cells == 0 ? 0 : level_prefix[n_cells, l]
    end
end

@kernel function ka_fill_unique_keys_kernel!(dest, @Const(sorted_keys), @Const(flags),
        @Const(prefix), offset, n)
    i = @index(Global)
    @inbounds if i <= n && flags[i] == 1
        dest[offset + prefix[i]] = sorted_keys[i]
    end
end

"""
    ka_radix_level_nodes!(node_keys, level_offsets, level_keys, level_flags,
                          level_prefix, level_counts, host_level_counts,
                          d_level_offsets, cell_keys, n_cells, ell, first_level,
                          max_nodes; workgroup=KA_AUTO_WORKGROUP)

Backend-agnostic port of the per-level unique-node block of
`_cuda_update_radix_grid_in_place!`: for each active level `first_level:ell`,
shift the occupied leaf-cell keys to that level's ancestor keys and compress the
runs into `node_keys`, level-major. Fills the host `level_offsets` (length
`ell + 2`, trimmed levels left at 0), mirrors it to `d_level_offsets`, and
returns `(n_nodes, max_count)` -- the node total for the caller's capacity
bookkeeping and the largest per-level node count, which is the x-extent of the
stage-4 2D launches.

`level_keys`/`level_flags`/`level_prefix` are `max_cells x (ell + 1)` device
scratch matrices, one column per level; `level_counts` is an `ell + 1` device
vector and `host_level_counts` its host mirror. Throws `AssertionError` if the
node total exceeds `max_nodes`.
"""
function ka_radix_level_nodes!(node_keys, level_offsets, level_keys, level_flags,
        level_prefix, level_counts, host_level_counts, d_level_offsets,
        cell_keys, n_cells::Int, ell::Int, first_level::Int, max_nodes::Int;
        workgroup=KA_AUTO_WORKGROUP)
    backend = KA.get_backend(cell_keys)
    n_levels = length(level_counts)
    if n_cells > 0
        ancestor = _cached_kernel(ka_leaf_ancestor_keys_kernel!, backend, workgroup)
        flagk = _cached_kernel(ka_key_change_flags_kernel!, backend, workgroup)
        for level in first_level:ell
            col = level + 1
            lk = view(level_keys, 1:n_cells, col)
            lf = view(level_flags, 1:n_cells, col)
            ancestor(lk, cell_keys, ell, level, n_cells; ndrange=n_cells)
            flagk(lf, lk, n_cells; ndrange=n_cells)
            accumulate!(+, view(level_prefix, 1:n_cells, col), lf)
        end
    end
    gather = _cached_kernel(ka_gather_level_counts_kernel!, backend, workgroup)
    gather(level_counts, level_prefix, n_cells, n_levels; ndrange=n_levels)
    KA.synchronize(backend)
    copyto!(host_level_counts, level_counts)

    level_offsets[1] = 0
    for level in 0:(first_level - 1)
        # trimmed levels: the gathered counts read unfilled prefix columns
        host_level_counts[level + 1] = 0
        level_offsets[level + 2] = 0
    end
    for level in first_level:ell
        level_offsets[level + 2] = level_offsets[level + 1] + host_level_counts[level + 1]
    end
    n_nodes = level_offsets[end]
    n_nodes <= max_nodes ||
        throw(AssertionError("device radix grid exceeded the cache node capacity"))

    if n_cells > 0
        uniquek = _cached_kernel(ka_fill_unique_keys_kernel!, backend, workgroup)
        for level in first_level:ell
            col = level + 1
            uniquek(node_keys, view(level_keys, 1:n_cells, col),
                view(level_flags, 1:n_cells, col),
                view(level_prefix, 1:n_cells, col), level_offsets[col], n_cells;
                ndrange=n_cells)
        end
    end
    copyto!(d_level_offsets, level_offsets)
    KA.synchronize(backend)
    return n_nodes, maximum(host_level_counts; init=0)
end



#------- in-place grid rebuild, stage 4: node geometry, parents, children -------#
#
# Final stage of the in-place grid rebuild: with the level-major `node_keys` and the
# `level_offsets` prefix in hand from stage 3, fill each node's level/coord/
# center, its parent index, and its contiguous child range, then map each leaf
# cell to its deepest-level node.
#
# The three CUDA kernels launch 2D, `blockIdx().y` carrying the level, so every
# active level runs concurrently and the per-level launch loop collapses to one
# launch. The KA ports keep that -- one launch over all levels -- but **flatten**
# the grid to 1D and decode the level from the flat index, the same idiom the
# hierarchical route generator uses. A 2D `ndrange` would have to
# carry a matching 2D workgroup size, which the per-backend scalar workgroup
# policy above does not produce; flattening keeps one tunable launch geometry
# for both backends. The x-extent is `max_count` from stage 3, so the flat range
# is `max_count * n_levels` with the ragged tail masked per level, exactly as
# the CUDA `node > stop` guard does.
#
# Parents and children resolve by binary search into the adjacent level's node
# block rather than by the host builder's sorted-merge walk: both blocks are
# ascending in key, and a search is what makes the levels independent enough to
# launch together. Roots (`level == min_level`) get `parent_index = 0` and the
# deepest level gets an empty child range.
#
# Host oracle for the gate: `_refresh_radix_nodes!` (src/tree_batched.jl) again
# -- stage 3 compared the part of its output stage 3 owns, this stage compares
# the rest.

# One pass over (level, node) filling every per-node array: the three kernels
# this replaces decoded the same index, read the same node_keys and
# level_offsets, wrote disjoint outputs and ran back to back with no
# synchronization between them, so they were three launches and three index
# decodes for one pass of work.
@kernel function ka_node_arrays_levels_kernel!(node_levels, node_coords, node_centers,
        parent_index, child_ranges, @Const(node_keys), @Const(level_offsets),
        x_min, h0, max_level, min_level, max_count)
    idx = @index(Global)
    @inbounds begin
        level = min_level + (idx - 1) ÷ max_count
        node = level_offsets[level + 1] + (idx - 1) % max_count + 1
        if node <= level_offsets[level + 2]
            TF = eltype(node_centers)
            key = node_keys[node]

            # geometry
            delta = (2 * h0) / (1 << level)
            ix, iy, iz = ka_decode_morton_key(key, level)
            node_levels[node] = level
            node_coords[1, node] = ix
            node_coords[2, node] = iy
            node_coords[3, node] = iz
            node_centers[1, node] = x_min[1] + delta * (TF(ix) + TF(0.5))
            node_centers[2, node] = x_min[2] + delta * (TF(iy) + TF(0.5))
            node_centers[3, node] = x_min[3] + delta * (TF(iz) + TF(0.5))

            # parent
            if level == min_level
                parent_index[node] = 0
            else
                parent_key = key >> 3
                parent_first = level_offsets[level] + 1
                parent_stop = level_offsets[level + 1]
                parent = ka_lower_bound(node_keys, parent_first, parent_stop, parent_key)
                parent_index[node] =
                    (parent <= parent_stop && node_keys[parent] == parent_key) ? parent : 0
            end

            # children
            if level == max_level
                child_ranges[1, node] = 0
                child_ranges[2, node] = 0
            else
                child_first = level_offsets[level + 2] + 1
                child_stop = level_offsets[level + 3]
                lo_key = key << 3
                hi_key = lo_key + UInt64(7)
                lo = ka_lower_bound(node_keys, child_first, child_stop, lo_key)
                hi = ka_upper_bound(node_keys, child_first, child_stop, hi_key)
                count = hi - lo
                child_ranges[1, node] = count > 0 ? lo : 0
                child_ranges[2, node] = count
            end
        end
    end
end

@kernel function ka_fill_leaf_to_node_kernel!(leaf_to_node, leaf_offset, n_cells)
    i = @index(Global)
    @inbounds if i <= n_cells
        leaf_to_node[i] = leaf_offset + i
    end
end

"""
    ka_radix_node_topology!(node_levels, node_coords, node_centers, parent_index,
                            child_ranges, leaf_to_node, node_keys, d_level_offsets,
                            level_offsets, x_min, h0, n_cells, ell, first_level,
                            max_count; workgroup=KA_AUTO_WORKGROUP)

Backend-agnostic port of the node geometry / parent / child-range trio plus
`_cuda_fill_leaf_to_node_kernel!` -- the last block of
`_cuda_update_radix_grid_in_place!`. Consumes stage 3's level-major `node_keys`,
its device-side `d_level_offsets` mirror and the host `level_offsets` vector,
and the `max_count` it returned (the per-level x-extent of the flattened
launches). Levels below `first_level` are trimmed and never touched; nodes at
`first_level` are roots with `parent_index = 0`.
"""
function ka_radix_node_topology!(node_levels, node_coords, node_centers,
        parent_index, child_ranges, leaf_to_node, node_keys, d_level_offsets,
        level_offsets, x_min, h0, n_cells::Int, ell::Int, first_level::Int,
        max_count::Int; workgroup=KA_AUTO_WORKGROUP)
    backend = KA.get_backend(node_keys)
    if max_count > 0
        n_levels = ell - first_level + 1
        flat = max_count * n_levels
        nodes = _cached_kernel(ka_node_arrays_levels_kernel!, backend, workgroup)
        nodes(node_levels, node_coords, node_centers, parent_index, child_ranges,
            node_keys, d_level_offsets, x_min, h0, ell, first_level, max_count;
            ndrange=flat)
    end
    if n_cells > 0
        l2n = _cached_kernel(ka_fill_leaf_to_node_kernel!, backend, workgroup)
        l2n(leaf_to_node, level_offsets[ell + 1], n_cells; ndrange=n_cells)
    end
    KA.synchronize(backend)
    return leaf_to_node
end



