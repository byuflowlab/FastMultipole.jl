#------- Step 5/6: adaptive octree construction, Phase A (K_max frontier split) -------#
#
# Backend-agnostic port of src/tree_batched_cuda.jl's Phase A
# (`_adt_cuda_seed_root_kernel!`/`_adt_cuda_split_flags_kernel!`/
# `_adt_cuda_split_compact_kernel!`/`_cuda_adaptive_build_leaves!`): the
# level-synchronous K_max frontier split that builds the adaptive octree's leaf
# set from a device array of full-depth-sorted Morton keys, before 2:1 balance
# (tree_batched_cuda.jl Phase B, not yet ported) and finalize (Phase C, not yet
# ported). This is the first phase of the tree/grid construction subsystem that
# `RadixFMMCache(device=true)` currently requires CUDA for (see the ka-migration
# plan/memory note on why `fmm!()` can't reach the KA M2M/M2L/L2L path on Metal
# without this). `_cuda_lower_bound` is CUDA-lifecycle-gated (only defined once
# `load_cuda_radix_lifecycle!()` runs), so `ka_lower_bound` below is a
# self-contained duplicate, not a shared call -- consistent with how the rest of
# this ext never calls into the lazy CUDA-only kernel file.

@inline function ka_lower_bound(keys, first, stop, key)
    lo = first
    hi = stop + 1
    while lo < hi
        mid = (lo + hi) >>> 1
        if @inbounds(keys[mid]) < key
            lo = mid + 1
        else
            hi = mid
        end
    end
    return lo
end

# Child tuple of frontier cell `i`, child bits `c` (0-based): occupied range by
# binary search over the full-depth-sorted body keys. Matches
# `_adt_cuda_child_range` (tree_batched_cuda.jl) exactly.
@inline function ka_child_range(sorted_keys, alev, akey, alo, ahi, i, c, ell_max)
    @inbounds begin
        lc = Int(alev[i]) + 1
        shift = 3 * (ell_max - lc)
        ckey = (akey[i] << 3) | UInt64(c)
        startk = ckey << shift
        endk = startk + (UInt64(1) << shift)
        lo_c = ka_lower_bound(sorted_keys, Int(alo[i]), Int(ahi[i]), startk)
        hi_c = ka_lower_bound(sorted_keys, Int(alo[i]), Int(ahi[i]), endk) - 1
    end
    return lc, ckey, lo_c, hi_c
end

@kernel function ka_seed_root_kernel!(lev, key, lo, hi, n)
    i = @index(Global)
    @inbounds if i == 1
        lev[1] = Int32(0)
        key[1] = UInt64(0)
        lo[1] = Int32(1)
        hi[1] = Int32(n)
    end
end

# flags over the 8 x nact virtual child slots; want_leaf=1 flags children that
# become leaves, want_leaf=0 flags children that stay active (split again).
@kernel function ka_split_flags_kernel!(flags, @Const(sorted_keys), @Const(alev),
        @Const(akey), @Const(alo), @Const(ahi), nact, K_max, ell_max, want_leaf)
    j = @index(Global)
    @inbounds if j <= 8 * nact
        i = (j - 1) >> 3 + 1
        c = (j - 1) & 7
        lc, _, lo_c, hi_c = ka_child_range(sorted_keys, alev, akey, alo, ahi, i, c, ell_max)
        f = Int32(0)
        if lo_c <= hi_c
            isleaf = (hi_c - lo_c + 1 <= K_max) || (lc == ell_max)
            f = ((want_leaf == Int32(1)) == isleaf) ? Int32(1) : Int32(0)
        end
        flags[j] = f
    end
end

@kernel function ka_split_compact_kernel!(dlev, dkey, dlo, dhi, base, @Const(flags),
        @Const(prefix), @Const(sorted_keys), @Const(alev), @Const(akey), @Const(alo),
        @Const(ahi), nact, ell_max)
    j = @index(Global)
    @inbounds if j <= 8 * nact && flags[j] == Int32(1)
        i = (j - 1) >> 3 + 1
        c = (j - 1) & 7
        lc, ckey, lo_c, hi_c = ka_child_range(sorted_keys, alev, akey, alo, ahi, i, c, ell_max)
        idx = base + Int(prefix[j])
        dlev[idx] = Int32(lc)
        dkey[idx] = ckey
        dlo[idx] = Int32(lo_c)
        dhi[idx] = Int32(hi_c)
    end
end

# Inclusive scan of flags[1:m] into prefix[1:m], returning the total. Mirrors
# `_adt_cuda_scan_total!`; `accumulate!` is already backend-generic (KA/GPUArrays),
# so only the final host-scalar readback needs to be written out explicitly.
function _ka_scan_total!(flags, prefix, m::Int)
    m == 0 && return 0
    fv = view(flags, 1:m)
    pv = view(prefix, 1:m)
    accumulate!(+, pv, fv)
    return Int(Array(view(prefix, m:m))[1])
end

#------- Preallocated tree-build context (zero recurring allocation) -------#
#
# Mirrors CUDA-native's `DeviceAdaptiveCUDAContext`/`actx` convention
# (src/tree_batched_cuda.jl:60-160): every working buffer used by
# `ka_build_adaptive_tree!`'s phases is allocated once, here, at capacity, and
# reused across calls instead of being rebuilt via `KA.zeros(...)` on every
# invocation -- the source of the ~1.95GB/trial n=1e6 allocation pressure
# bisected in job 13506038 (see project_fastmultipole_ka_migration memory).
# Scoped to tree construction (Phases A/B/C/D); CUDA's DTR/interaction-list
# buffers (`u_capacity`/`v_capacity`/`wx_capacity`) have no KA counterpart yet.

struct KAAdaptiveTreeContext{B,G,NT<:NamedTuple}
    backend::B
    maxn::Int
    leaf_capacity::Int
    frontier_capacity::Int
    node_capacity::Int
    # The tree's public output, in the form the resident FMM path consumes
    # (`DeviceResidentRadixState.grid`). Allocated here at capacity and mutated
    # in place by the phases -- `bufs` aliases its arrays, so the phase code is
    # unchanged and there is no per-build tuple-to-grid conversion (which would
    # allocate and break the task-023 contract). Mirrors CUDA, where
    # `DeviceAdaptiveCUDAContext` owns the grid and every phase writes `grid.*`.
    grid::G
    bufs::NT
end

"""
    ka_allocate_adaptive_context(backend, TF, maxn; leaf_capacity, frontier_capacity, node_capacity)

Allocate a `KAAdaptiveTreeContext`: every scratch/output buffer
`ka_build_adaptive_tree!` and its phase functions need, sized once at
`maxn`/`leaf_capacity`/`frontier_capacity`/`node_capacity` and reused across
calls. Construct once per (backend, capacity) combination, outside any
trial/timestep loop.

The node- and cell-indexed outputs are allocated as the fields of a capacity-sized
`DeviceRadixGrid` (`actx.grid`), which `bufs` aliases; the geometry fields
(`x_min`/`h0`/`ell`) and the prefix lengths (`n_bodies`/`n_cells`) are placeholders
until `ka_build_adaptive_tree!` sets them from its build arguments, exactly as
`_cuda_refresh_adaptive_tree!` does. `grid.body_system`/`grid.body_index` are filled
for a single source system (see `ka_fill_single_system_attribution!`); multi-system
attribution belongs to the repack path rather than the octree build.
"""
function ka_allocate_adaptive_context(backend, ::Type{TF}, maxn::Int;
        leaf_capacity::Int, frontier_capacity::Int, node_capacity::Int) where TF
    LC, FC, NC = leaf_capacity, frontier_capacity, node_capacity
    grid = FastMultipole.DeviceRadixGrid(
        zero(SVector{3,TF}), one(TF), 0, 0, 0,
        KA.zeros(backend, Int, maxn), KA.zeros(backend, Int, maxn),
        KA.zeros(backend, UInt64, LC), KA.zeros(backend, Int, 2, LC),
        KA.zeros(backend, Int, maxn), KA.zeros(backend, Int, maxn),
        KA.zeros(backend, TF, 3, LC),
        KA.zeros(backend, Int, NC), KA.zeros(backend, UInt64, NC),
        KA.zeros(backend, Int, 3, NC), KA.zeros(backend, TF, 3, NC),
        KA.zeros(backend, Int, NC), KA.zeros(backend, Int, 2, NC),
        KA.zeros(backend, Int, LC),
    )
    bufs = (
        keys=KA.zeros(backend, UInt64, maxn), perm=grid.perm,
        invperm=grid.invperm,
        sorted_keys=KA.zeros(backend, UInt64, maxn),

        llev=KA.zeros(backend, Int32, LC), lkey=KA.zeros(backend, UInt64, LC),
        llo=KA.zeros(backend, Int32, LC), lhi=KA.zeros(backend, Int32, LC),
        a_lev=KA.zeros(backend, Int32, FC), a_key=KA.zeros(backend, UInt64, FC),
        a_lo=KA.zeros(backend, Int32, FC), a_hi=KA.zeros(backend, Int32, FC),
        b_lev=KA.zeros(backend, Int32, FC), b_key=KA.zeros(backend, UInt64, FC),
        b_lo=KA.zeros(backend, Int32, FC), b_hi=KA.zeros(backend, Int32, FC),
        bl_flags=KA.zeros(backend, Int32, FC), bl_prefix=KA.zeros(backend, Int32, FC),

        bal_shifted=KA.zeros(backend, UInt64, LC), bal_order=KA.zeros(backend, Int, LC),
        bal_scratch_starts=KA.zeros(backend, UInt64, LC), bal_marks=KA.zeros(backend, Int32, LC),
        bal_flags=KA.zeros(backend, Int32, LC), bal_prefix=KA.zeros(backend, Int32, LC),
        dlev=KA.zeros(backend, Int32, LC), dkey=KA.zeros(backend, UInt64, LC),
        dlo=KA.zeros(backend, Int32, LC), dhi=KA.zeros(backend, Int32, LC),

        fin_shifted=KA.zeros(backend, UInt64, LC), fin_order=KA.zeros(backend, Int, LC),
        fin_skey=KA.zeros(backend, UInt64, LC), fin_slev=KA.zeros(backend, Int32, LC),
        fin_cand=KA.zeros(backend, UInt64, LC),
        fin_flags=KA.zeros(backend, Int32, NC), fin_prefix=KA.zeros(backend, Int32, NC),
        node_keys=grid.node_keys, node_levels=grid.node_levels,
        node_coords=grid.node_coords, node_centers=grid.node_centers,
        node_lo=KA.zeros(backend, Int32, NC), node_hi=KA.zeros(backend, Int32, NC),
        parent_index=grid.parent_index, child_ranges=grid.child_ranges,
        leaf_index=KA.zeros(backend, Int32, NC), leaf_slot_of=KA.zeros(backend, Int32, NC),
        cell_ranges=grid.cell_ranges, cell_centers=grid.cell_centers,
        cell_keys=grid.cell_keys, leaf_to_node=grid.leaf_to_node,

        node_sigma=KA.zeros(backend, TF, NC),
    )
    return KAAdaptiveTreeContext(backend, maxn, leaf_capacity, frontier_capacity,
        node_capacity, grid, bufs)
end

"""
    ka_adaptive_build_leaves!(actx, sorted_keys, ell_max, K_max, n; workgroup=64)

Backend-agnostic port of `_cuda_adaptive_build_leaves!` (tree_batched_cuda.jl):
builds the K_max leaf set (adaptive octree theory §1.2) from `sorted_keys`, a
KA-backend array of the `n` bodies' full-depth Morton keys in ascending sorted
order. Returns `(nl, leaf_levels, leaf_keys, leaf_lo, leaf_hi)`, each a
backend array truncated logically to `1:nl` (allocated at `actx.leaf_capacity`).
`leaf_lo`/`leaf_hi` are 1-based inclusive ranges into `sorted_keys`. Scratch/
output buffers come from `actx` (see `KAAdaptiveTreeContext`) -- no allocation.
"""
function ka_adaptive_build_leaves!(actx::KAAdaptiveTreeContext, sorted_keys::AbstractVector{UInt64},
        ell_max::Int, K_max::Int, n::Int; workgroup::Int=KA_AUTO_WORKGROUP)
    backend = actx.backend
    b = actx.bufs
    leaf_capacity = actx.leaf_capacity
    frontier_capacity = actx.frontier_capacity
    llev, lkey, llo, lhi = b.llev, b.lkey, b.llo, b.lhi

    seedk = _cached_kernel(ka_seed_root_kernel!, backend, 1)

    if n <= K_max || ell_max == 0
        seedk(llev, lkey, llo, lhi, n; ndrange=1)
        KA.synchronize(backend)
        return 1, llev, lkey, llo, lhi
    end

    a_lev, a_key, a_lo, a_hi = b.a_lev, b.a_key, b.a_lo, b.a_hi
    b_lev, b_key, b_lo, b_hi = b.b_lev, b.b_key, b.b_lo, b.b_hi
    flags, prefix = b.bl_flags, b.bl_prefix

    seedk(a_lev, a_key, a_lo, a_hi, n; ndrange=1)
    KA.synchronize(backend)

    flagsk = _cached_kernel(ka_split_flags_kernel!, backend, workgroup)
    compactk = _cached_kernel(ka_split_compact_kernel!, backend, workgroup)

    nact = 1
    nl = 0
    round = 0
    while nact > 0
        round += 1
        round <= ell_max + 1 || error("adaptive KA K_max split failed to terminate")
        m = 8 * nact
        m <= frontier_capacity || error(
            "adaptive KA split frontier capacity $frontier_capacity exceeded")

        flagsk(flags, sorted_keys, a_lev, a_key, a_lo, a_hi, nact, K_max, ell_max,
            Int32(1); ndrange=m)
        KA.synchronize(backend)
        nleaf = _ka_scan_total!(flags, prefix, m)
        nl + nleaf <= leaf_capacity || error(
            "adaptive KA leaf capacity $leaf_capacity exceeded")
        if nleaf > 0
            compactk(llev, lkey, llo, lhi, nl, flags, prefix, sorted_keys, a_lev, a_key,
                a_lo, a_hi, nact, ell_max; ndrange=m)
            KA.synchronize(backend)
        end
        nl += nleaf

        flagsk(flags, sorted_keys, a_lev, a_key, a_lo, a_hi, nact, K_max, ell_max,
            Int32(0); ndrange=m)
        KA.synchronize(backend)
        nact2 = _ka_scan_total!(flags, prefix, m)
        nact2 <= frontier_capacity || error(
            "adaptive KA active frontier count $nact2 exceeded capacity $frontier_capacity")
        if nact2 > 0
            compactk(b_lev, b_key, b_lo, b_hi, 0, flags, prefix, sorted_keys, a_lev, a_key,
                a_lo, a_hi, nact, ell_max; ndrange=m)
            KA.synchronize(backend)
        end
        a_lev, b_lev = b_lev, a_lev
        a_key, b_key = b_key, a_key
        a_lo, b_lo = b_lo, a_lo
        a_hi, b_hi = b_hi, a_hi
        nact = nact2
    end
    return nl, llev, lkey, llo, lhi
end

#------- Phase B: 2:1 balance (Jacobi rounds over the leaf key set) -------#

@inline function ka_upper_bound(keys, first, stop, key)
    lo = first
    hi = stop + 1
    while lo < hi
        mid = (lo + hi) >>> 1
        if @inbounds(keys[mid]) <= key
            lo = mid + 1
        else
            hi = mid
        end
    end
    return lo
end

@inline function ka_morton_key(ix, iy, iz, ell)
    key = UInt64(0)
    for bit in 0:(ell - 1)
        key |= (UInt64((ix >> bit) & 0x1) << (3 * bit))
        key |= (UInt64((iy >> bit) & 0x1) << (3 * bit + 1))
        key |= (UInt64((iz >> bit) & 0x1) << (3 * bit + 2))
    end
    return key
end

@inline function ka_decode_morton_key(key, ell)
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

@kernel function ka_leaf_shifted_kernel!(shifted, @Const(lev), @Const(key), ell_max)
    i = @index(Global)
    @inbounds shifted[i] = key[i] << (3 * (ell_max - Int(lev[i])))
end

# Mark every leaf that violates 2:1 against the current leaf set (mirrors
# `_adt_cuda_balance_mark_kernel!`): leaf B at level lev emits its <=8 touching
# parent-level cells; a leaf A at level <= lev-2 whose interval contains the
# emitted cell start is marked. `marks` is pre-zeroed by the driver (a
# same-kernel clear would race with concurrent mark writes from other threads).
@kernel function ka_balance_mark_kernel!(marks, @Const(lev), @Const(key), nl,
        @Const(sorted_starts), @Const(order), @Const(slev), ell_max)
    i = @index(Global)
    @inbounds if i <= nl
        l = Int(lev[i])
        if l >= 2
            cx, cy, cz = ka_decode_morton_key(key[i], l)
            Gc = 1 << (l - 1)
            qx0 = (cx - 1) >> 1
            qy0 = (cy - 1) >> 1
            qz0 = (cz - 1) >> 1
            for dz in 0:1, dy in 0:1, dx in 0:1
                qx = qx0 + dx
                qy = qy0 + dy
                qz = qz0 + dz
                if 0 <= qx < Gc && 0 <= qy < Gc && 0 <= qz < Gc
                    qstart = ka_morton_key(qx, qy, qz, l - 1) << (3 * (ell_max - (l - 1)))
                    j = ka_upper_bound(sorted_starts, 1, nl, qstart) - 1
                    if j != 0
                        aid = Int(order[j])
                        la = Int(slev[aid])
                        if la <= l - 2
                            astart = sorted_starts[j]
                            alen = UInt64(1) << (3 * (ell_max - la))
                            if qstart < astart + alen
                                marks[aid] = Int32(1)
                            end
                        end
                    end
                end
            end
        end
    end
end

# Per-leaf emission count: unmarked leaves keep one slot; marked leaves emit
# their occupied children (theory §1.4 -- the split leaf is occupied, so >= 1).
@kernel function ka_balance_count_kernel!(cnt, @Const(marks), @Const(lev), @Const(key),
        @Const(lo), @Const(hi), nl, @Const(sorted_keys), ell_max)
    i = @index(Global)
    @inbounds if i <= nl
        if marks[i] == Int32(0)
            cnt[i] = Int32(1)
        else
            m = 0
            for c in 0:7
                _, _, lo_c, hi_c = ka_child_range(sorted_keys, lev, key, lo, hi, i, c, ell_max)
                if lo_c <= hi_c
                    m += 1
                end
            end
            cnt[i] = Int32(m)
        end
    end
end

@kernel function ka_balance_emit_kernel!(dlev, dkey, dlo, dhi, @Const(marks), @Const(prefix),
        @Const(lev), @Const(key), @Const(lo), @Const(hi), nl, @Const(sorted_keys), ell_max)
    i = @index(Global)
    @inbounds if i <= nl
        base = i == 1 ? 0 : Int(prefix[i - 1])
        if marks[i] == Int32(0)
            dlev[base + 1] = lev[i]
            dkey[base + 1] = key[i]
            dlo[base + 1] = lo[i]
            dhi[base + 1] = hi[i]
        else
            w = 0
            for c in 0:7
                lc, ckey, lo_c, hi_c = ka_child_range(sorted_keys, lev, key, lo, hi, i, c, ell_max)
                if lo_c <= hi_c
                    w += 1
                    dlev[base + w] = Int32(lc)
                    dkey[base + w] = ckey
                    dlo[base + w] = Int32(lo_c)
                    dhi[base + w] = Int32(hi_c)
                end
            end
        end
    end
end

"""
    ka_adaptive_balance!(actx, nl, leaf_levels, leaf_keys, leaf_lo, leaf_hi, sorted_keys, ell_max;
                          workgroup=64)

Backend-agnostic port of `_cuda_adaptive_balance!` (tree_batched_cuda.jl): Jacobi
2:1-balance sweep (theory §1.4, Sundar-style) to the fixed point over the K_max
leaf set produced by `ka_adaptive_build_leaves!`. `leaf_levels`/`leaf_keys`/
`leaf_lo`/`leaf_hi` are backend arrays (allocated at `actx.leaf_capacity`, logically
truncated to `1:nl`); `sorted_keys` is the same full-depth-sorted body key
array `ka_adaptive_build_leaves!` was given. Returns
`(nl, n_balance_splits, leaf_levels, leaf_keys, leaf_lo, leaf_hi)` -- the final
leaf arrays may be either input array (ping-pong), not necessarily the ones
passed in. Scratch/output buffers come from `actx` -- no allocation.
"""
function ka_adaptive_balance!(actx::KAAdaptiveTreeContext, nl::Int, leaf_levels, leaf_keys,
        leaf_lo, leaf_hi, sorted_keys::AbstractVector{UInt64}, ell_max::Int; workgroup::Int=KA_AUTO_WORKGROUP)
    backend = actx.backend
    b = actx.bufs
    leaf_capacity = actx.leaf_capacity

    shifted, order, scratch_starts = b.bal_shifted, b.bal_order, b.bal_scratch_starts
    marks, flags, prefix = b.bal_marks, b.bal_flags, b.bal_prefix
    dlev, dkey, dlo, dhi = b.dlev, b.dkey, b.dlo, b.dhi

    shiftedk = _cached_kernel(ka_leaf_shifted_kernel!, backend, workgroup)
    markk = _cached_kernel(ka_balance_mark_kernel!, backend, workgroup)
    countk = _cached_kernel(ka_balance_count_kernel!, backend, workgroup)
    emitk = _cached_kernel(ka_balance_emit_kernel!, backend, workgroup)

    src = (leaf_levels, leaf_keys, leaf_lo, leaf_hi)
    dst = (dlev, dkey, dlo, dhi)
    total = 0
    round = 0
    while true
        round += 1
        round <= 2 * ell_max + 4 || error(
            "adaptive KA 2:1 balance failed to reach a fixed point")

        shiftedk(shifted, src[1], src[2], ell_max; ndrange=nl)
        KA.synchronize(backend)
        ov = view(order, 1:nl)
        sortperm!(ov, view(shifted, 1:nl))
        ka_gather_values!(view(scratch_starts, 1:nl), view(shifted, 1:nl), ov; workgroup=workgroup)

        fill!(view(marks, 1:nl), Int32(0))
        markk(marks, src[1], src[2], nl, scratch_starts, order, src[1], ell_max; ndrange=nl)
        KA.synchronize(backend)

        copyto!(view(flags, 1:nl), view(marks, 1:nl))
        nmark = _ka_scan_total!(flags, prefix, nl)
        nmark == 0 && break
        total += nmark

        countk(flags, marks, src[1], src[2], src[3], src[4], nl, sorted_keys, ell_max; ndrange=nl)
        KA.synchronize(backend)
        nl2 = _ka_scan_total!(flags, prefix, nl)
        nl2 <= leaf_capacity || error(
            "adaptive KA leaf capacity $leaf_capacity exceeded during the balance sweep")

        emitk(dst[1], dst[2], dst[3], dst[4], marks, prefix, src[1], src[2], src[3], src[4],
            nl, sorted_keys, ell_max; ndrange=nl)
        KA.synchronize(backend)

        src, dst = dst, src
        nl = nl2
    end
    return nl, total, src[1], src[2], src[3], src[4]
end

#------- Phase C: level-major node table finalize -------#

@kernel function ka_ancestor_flags_kernel!(flags, @Const(slev), nl, L)
    i = @index(Global)
    @inbounds if i <= nl
        flags[i] = Int(slev[i]) >= L ? Int32(1) : Int32(0)
    end
end

@kernel function ka_ancestor_compact_kernel!(cand, @Const(flags), @Const(prefix),
        @Const(slev), @Const(skey), nl, L)
    i = @index(Global)
    @inbounds if i <= nl && flags[i] == Int32(1)
        cand[Int(prefix[i])] = skey[i] >> (3 * (Int(slev[i]) - L))
    end
end

@kernel function ka_unique_flags_kernel!(flags, @Const(cand), m)
    i = @index(Global)
    @inbounds if i <= m
        flags[i] = (i == 1 || cand[i] != cand[i - 1]) ? Int32(1) : Int32(0)
    end
end

@kernel function ka_unique_compact_kernel!(node_keys, node_levels, base, @Const(flags),
        @Const(prefix), @Const(cand), m, L)
    i = @index(Global)
    @inbounds if i <= m && flags[i] == Int32(1)
        idx = base + Int(prefix[i])
        node_keys[idx] = cand[i]
        node_levels[idx] = L
    end
end

@kernel function ka_node_ranges_kernel!(node_lo, node_hi, @Const(node_keys),
        @Const(node_levels), n_nodes, @Const(sorted_keys), n, ell_max)
    i = @index(Global)
    @inbounds if i <= n_nodes
        shift = 3 * (ell_max - Int(node_levels[i]))
        startk = node_keys[i] << shift
        endk = startk + (UInt64(1) << shift)
        node_lo[i] = Int32(ka_lower_bound(sorted_keys, 1, n, startk))
        node_hi[i] = Int32(ka_lower_bound(sorted_keys, 1, n, endk) - 1)
    end
end

@kernel function ka_node_geometry_kernel!(node_coords, node_centers, @Const(node_keys),
        @Const(node_levels), n_nodes, x_min, h0)
    i = @index(Global)
    @inbounds if i <= n_nodes
        L = Int(node_levels[i])
        cx, cy, cz = ka_decode_morton_key(node_keys[i], L)
        node_coords[1, i] = cx
        node_coords[2, i] = cy
        node_coords[3, i] = cz
        TF = eltype(node_centers)
        width = (2 * h0) / (1 << L)
        node_centers[1, i] = x_min[1] + width * (TF(cx) + TF(0.5))
        node_centers[2, i] = x_min[2] + width * (TF(cy) + TF(0.5))
        node_centers[3, i] = x_min[3] + width * (TF(cz) + TF(0.5))
    end
end

@kernel function ka_parent_kernel!(parent_index, @Const(node_keys), base, count,
        base_prev, count_prev)
    i = @index(Global)
    @inbounds if i <= count
        node = base + i
        pk = node_keys[node] >> 3
        parent_index[node] = ka_lower_bound(node_keys, base_prev + 1,
            base_prev + count_prev, pk)
    end
end

@kernel function ka_children_kernel!(child_ranges, @Const(node_keys), base, count,
        base_next, count_next)
    i = @index(Global)
    @inbounds if i <= count
        node = base + i
        k = node_keys[node] << 3
        firstc = ka_lower_bound(node_keys, base_next + 1, base_next + count_next, k)
        endc = ka_lower_bound(node_keys, base_next + 1, base_next + count_next, k + UInt64(8))
        child_ranges[1, node] = endc > firstc ? firstc : 0
        child_ranges[2, node] = endc - firstc
    end
end

@kernel function ka_leaf_flags_kernel!(flags, @Const(child_ranges), n_nodes)
    i = @index(Global)
    @inbounds if i <= n_nodes
        flags[i] = child_ranges[2, i] == 0 ? Int32(1) : Int32(0)
    end
end

@kernel function ka_leaf_compact_kernel!(leaf_index, leaf_slot_of, @Const(flags),
        @Const(prefix), n_nodes)
    i = @index(Global)
    @inbounds if i <= n_nodes
        leaf_slot_of[i] = Int32(0)
        if flags[i] == Int32(1)
            slot = Int(prefix[i])
            leaf_index[slot] = Int32(i)
            leaf_slot_of[i] = Int32(slot)
        end
    end
end

@kernel function ka_cell_arrays_kernel!(cell_ranges, cell_centers, cell_keys, leaf_to_node,
        @Const(leaf_index), @Const(node_lo), @Const(node_hi), @Const(node_centers),
        @Const(node_keys), n_leaves)
    c = @index(Global)
    @inbounds if c <= n_leaves
        f = Int(leaf_index[c])
        leaf_to_node[c] = f
        cell_ranges[1, c] = Int(node_lo[f])
        cell_ranges[2, c] = Int(node_hi[f]) - Int(node_lo[f]) + 1
        cell_centers[1, c] = node_centers[1, f]
        cell_centers[2, c] = node_centers[2, f]
        cell_centers[3, c] = node_centers[3, f]
        cell_keys[c] = node_keys[f]
    end
end

"""
    ka_adaptive_finalize!(actx, nl, leaf_levels, leaf_keys, leaf_lo, leaf_hi, sorted_keys,
                           ell_max, n, x_min, h0::TF; workgroup=64)

Backend-agnostic port of `_cuda_adaptive_finalize!` (tree_batched_cuda.jl): builds the
level-major node table (ancestor + leaf nodes, Morton-sorted within each level) from
the final (post 2:1-balance) leaf set produced by `ka_adaptive_balance!`, then resolves
node body-ranges, geometry, parent/child links, leaf compaction, and the leaf-indexed
cell presentation arrays. `leaf_levels`/`leaf_keys`/`leaf_lo`/`leaf_hi` are backend
arrays logically truncated to `1:nl`; `sorted_keys` is the same full-depth-sorted body
key array used throughout the adaptive-tree pipeline. `x_min`/`h0` are the tree's
bounding-box origin/half-width (plain scalars, e.g. an `SVector{3,TF}`/`TF`, not device
arrays -- mirrors CUDA's `grid.x_min`/`grid.h0` convention).

Returns a `NamedTuple` `(n_nodes, n_leaves, node_keys, node_levels, node_coords,
node_centers, node_lo, node_hi, parent_index, child_ranges, leaf_index, leaf_slot_of,
cell_ranges, cell_centers, cell_keys, leaf_to_node)`. The node-indexed arrays are
allocated at `actx.node_capacity` and logically truncated to `1:n_nodes`; the
leaf-indexed cell arrays are allocated at `actx.leaf_capacity` and logically truncated
to `1:n_leaves` (== `nl`, asserted). Scratch/output buffers come from `actx` -- no
allocation.
"""
function ka_adaptive_finalize!(actx::KAAdaptiveTreeContext, nl::Int, leaf_levels, leaf_keys,
        leaf_lo, leaf_hi, sorted_keys::AbstractVector{UInt64}, ell_max::Int, n::Int, x_min,
        h0::TF; workgroup::Int=KA_AUTO_WORKGROUP) where TF
    backend = actx.backend
    b = actx.bufs

    shifted, order = b.fin_shifted, b.fin_order
    skey, slev, cand = b.fin_skey, b.fin_slev, b.fin_cand
    flags, prefix = b.fin_flags, b.fin_prefix

    node_keys, node_levels = b.node_keys, b.node_levels
    node_coords, node_centers = b.node_coords, b.node_centers
    node_lo, node_hi = b.node_lo, b.node_hi
    parent_index, child_ranges = b.parent_index, b.child_ranges
    leaf_index, leaf_slot_of = b.leaf_index, b.leaf_slot_of

    shiftedk = _cached_kernel(ka_leaf_shifted_kernel!, backend, workgroup)
    ancflagsk = _cached_kernel(ka_ancestor_flags_kernel!, backend, workgroup)
    anccompactk = _cached_kernel(ka_ancestor_compact_kernel!, backend, workgroup)
    uflagsk = _cached_kernel(ka_unique_flags_kernel!, backend, workgroup)
    ucompactk = _cached_kernel(ka_unique_compact_kernel!, backend, workgroup)
    rangesk = _cached_kernel(ka_node_ranges_kernel!, backend, workgroup)
    geomk = _cached_kernel(ka_node_geometry_kernel!, backend, workgroup)
    parentk = _cached_kernel(ka_parent_kernel!, backend, workgroup)
    childrenk = _cached_kernel(ka_children_kernel!, backend, workgroup)
    leafflagsk = _cached_kernel(ka_leaf_flags_kernel!, backend, workgroup)
    leafcompactk = _cached_kernel(ka_leaf_compact_kernel!, backend, workgroup)
    cellk = _cached_kernel(ka_cell_arrays_kernel!, backend, workgroup)

    shiftedk(shifted, leaf_levels, leaf_keys, ell_max; ndrange=nl)
    KA.synchronize(backend)
    ov = view(order, 1:nl)
    sortperm!(ov, view(shifted, 1:nl))
    ka_gather_values!(view(skey, 1:nl), leaf_keys, ov; workgroup=workgroup)
    ka_gather_values!(view(slev, 1:nl), leaf_levels, ov; workgroup=workgroup)

    node_capacity = actx.node_capacity
    off = zeros(Int, ell_max + 2)
    n_nodes = 0
    for L in 0:ell_max
        off[L + 1] = n_nodes
        ancflagsk(flags, slev, nl, L; ndrange=nl)
        KA.synchronize(backend)
        me = _ka_scan_total!(flags, prefix, nl)
        me == 0 && continue
        anccompactk(cand, flags, prefix, slev, skey, nl, L; ndrange=nl)
        KA.synchronize(backend)
        uflagsk(flags, cand, me; ndrange=me)
        KA.synchronize(backend)
        mL = _ka_scan_total!(flags, prefix, me)
        n_nodes + mL <= node_capacity || error(
            "adaptive KA node capacity $node_capacity exceeded; raise node_capacity")
        ucompactk(node_keys, node_levels, n_nodes, flags, prefix, cand, me, L; ndrange=me)
        KA.synchronize(backend)
        n_nodes += mL
    end
    off[ell_max + 2] = n_nodes

    rangesk(node_lo, node_hi, node_keys, node_levels, n_nodes, sorted_keys, n, ell_max;
        ndrange=n_nodes)
    geomk(node_coords, node_centers, node_keys, node_levels, n_nodes, x_min, h0;
        ndrange=n_nodes)
    KA.synchronize(backend)

    fill!(view(parent_index, 1:n_nodes), 0)
    fill!(view(child_ranges, :, 1:n_nodes), 0)
    for L in 1:ell_max
        base = off[L + 1]
        count = off[L + 2] - off[L + 1]
        count > 0 || continue
        base_prev = off[L]
        count_prev = off[L + 1] - off[L]
        parentk(parent_index, node_keys, base, count, base_prev, count_prev; ndrange=count)
    end
    for L in 0:(ell_max - 1)
        base = off[L + 1]
        count = off[L + 2] - off[L + 1]
        count > 0 || continue
        base_next = off[L + 2]
        count_next = off[L + 3] - off[L + 2]
        childrenk(child_ranges, node_keys, base, count, base_next, count_next; ndrange=count)
    end
    KA.synchronize(backend)

    leafflagsk(flags, child_ranges, n_nodes; ndrange=n_nodes)
    KA.synchronize(backend)
    n_leaves = _ka_scan_total!(flags, prefix, n_nodes)
    n_leaves == nl || throw(AssertionError(
        "adaptive KA finalize: leaf count mismatch ($n_leaves vs $nl)"))
    leafcompactk(leaf_index, leaf_slot_of, flags, prefix, n_nodes; ndrange=n_nodes)
    KA.synchronize(backend)

    cell_ranges = view(b.cell_ranges, :, 1:n_leaves)
    cell_centers = view(b.cell_centers, :, 1:n_leaves)
    cell_keys = view(b.cell_keys, 1:n_leaves)
    leaf_to_node = view(b.leaf_to_node, 1:n_leaves)
    cellk(cell_ranges, cell_centers, cell_keys, leaf_to_node, leaf_index, node_lo, node_hi,
        node_centers, node_keys, n_leaves; ndrange=n_leaves)
    KA.synchronize(backend)

    return (n_nodes=n_nodes, n_leaves=n_leaves, node_keys=node_keys, node_levels=node_levels,
        node_coords=node_coords, node_centers=node_centers, node_lo=node_lo, node_hi=node_hi,
        parent_index=parent_index, child_ranges=child_ranges, leaf_index=leaf_index,
        leaf_slot_of=leaf_slot_of, cell_ranges=cell_ranges, cell_centers=cell_centers,
        cell_keys=cell_keys, leaf_to_node=leaf_to_node, level_offsets=off)
end

#------- Phase D: per-node subtree sigma_max upward pass -------#

@kernel function ka_leaf_sigma_kernel!(node_sigma, @Const(node_lo), @Const(node_hi),
        @Const(child_ranges), @Const(source_bodies), sigma_row, n_nodes)
    i = @index(Global)
    @inbounds if i <= n_nodes && child_ranges[2, i] == 0
        TF = eltype(node_sigma)
        m = zero(TF)
        for r in Int(node_lo[i]):Int(node_hi[i])
            s = source_bodies[sigma_row, r]
            s > m && (m = s)
        end
        node_sigma[i] = m
    end
end

@kernel function ka_sigma_up_kernel!(node_sigma, @Const(child_ranges), base, count)
    i = @index(Global)
    @inbounds if i <= count
        node = base + i
        nc = Int(child_ranges[2, node])
        if nc != 0
            c0 = Int(child_ranges[1, node])
            TF = eltype(node_sigma)
            m = zero(TF)
            for c in c0:(c0 + nc - 1)
                s = node_sigma[c]
                s > m && (m = s)
            end
            node_sigma[node] = m
        end
    end
end

"""
    ka_adaptive_sigma_sweep!(actx, node_lo, node_hi, child_ranges, n_nodes, level_offsets,
                              ell_max, source_bodies, sigma_row; workgroup=64)

Backend-agnostic port of `_cuda_adaptive_sigma_sweep!` (tree_batched_cuda.jl): computes,
for every node of the finalized level-major node table (see `ka_adaptive_finalize!`),
the max of `source_bodies[sigma_row, :]` over its subtree -- leaves take the max over
their own body range (`node_lo`/`node_hi`), interior nodes take the max over their
children's already-computed `node_sigma_max`, processed level-by-level from `ell_max - 1`
up to `0` using `level_offsets` (so children are always finalized before their parent is
visited). `source_bodies` must be indexed in the same sorted-body ordering as
`node_lo`/`node_hi` (i.e. the same `sorted_keys` passed to `ka_adaptive_finalize!`).

Returns `node_sigma_max`, an array sized `actx.node_capacity`, logically truncated to
`1:n_nodes`. Buffer comes from `actx` -- no allocation.
"""
function ka_adaptive_sigma_sweep!(actx::KAAdaptiveTreeContext, node_lo, node_hi, child_ranges,
        n_nodes::Int, level_offsets::Vector{Int}, ell_max::Int,
        source_bodies::AbstractMatrix{TF}, sigma_row::Int; workgroup::Int=KA_AUTO_WORKGROUP) where TF
    backend = actx.backend
    node_sigma = actx.bufs.node_sigma

    leafsigmak = _cached_kernel(ka_leaf_sigma_kernel!, backend, workgroup)
    upk = _cached_kernel(ka_sigma_up_kernel!, backend, workgroup)

    leafsigmak(node_sigma, node_lo, node_hi, child_ranges, source_bodies, sigma_row,
        n_nodes; ndrange=n_nodes)
    KA.synchronize(backend)

    for L in (ell_max - 1):-1:0
        base = level_offsets[L + 1]
        count = level_offsets[L + 2] - level_offsets[L + 1]
        count > 0 || continue
        upk(node_sigma, child_ranges, base, count; ndrange=count)
        KA.synchronize(backend)
    end

    return node_sigma
end

#------- Phase E: DTR interaction-list build (U/V/W/X) -------#
#
# Backend-agnostic port of `_cuda_adaptive_build_lists!` and its six kernels
# (src/tree_batched_cuda.jl:910-1139), the frontier dual-tree-recursion sweep of
# adaptive octree theory §2.7 with sticky sigma demotion (§5.2).
#
# Shape note: the CPU reference (`build_adaptive_interaction_lists!`,
# src/interaction_list_batched.jl:1182) is a DFS over an explicit pair stack,
# while the device version is a level-synchronous BFS over a pair frontier with
# the same flags/scan/compact decomposition as Phases A-C. Both enumerate the
# same pair SET; they emit it in different orders. Any comparison between them
# must canonicalize (sort) first -- element-wise equality is meaningless here.
#
# Precision note: CUDA's `_adt_cuda_classify` hardcodes Float64 for the sigma
# gate. Metal has no Float64 at all, so the KA port makes the gate arithmetic
# generic in a float type `TG` instead (the CUDA path is left exactly as it is).
# `ka_gate_float_type` picks a default that is always valid for the backend the
# tree lives on -- the sigma array's own float type, which is whatever that
# backend supports -- and any float type can be forced via the `gate_type`
# keyword: Float64 on CUDA to mirror the native path bit-for-bit, Float32 on
# Metal, Float16 or a custom AbstractFloat if some future backend wants it.
# Only the gate comparison `delta_min2*g2 < cut^2` is precision-sensitive, and
# only for pairs sitting within a rounding step of the threshold; everything
# else in the classification is exact integer lattice arithmetic, so the choice
# cannot change which pairs are geometrically near.
#
# `g2` is bounded by 3*(2^ell_max)^2, so it is exactly representable in Float32
# for ell_max <= 12 and in Float16 for ell_max <= 4 -- past those the gate
# arithmetic, not the tree, is what limits precision.

"""
    ka_gate_float_type(node_sigma) -> Type{<:AbstractFloat}

Default float type for the DTR sigma gate: the float type the tree's own sigma
array already uses, which is by construction one the backend supports. Override
with the `gate_type` keyword on `ka_adaptive_build_lists!` when a specific
precision is wanted (e.g. Float64 on CUDA, to match `_adt_cuda_classify`).
"""
ka_gate_float_type(node_sigma) = float(eltype(node_sigma))

const _KA_KIND_U = Int32(1)
const _KA_KIND_V = Int32(2)
const _KA_KIND_W = Int32(3)
const _KA_KIND_X = Int32(4)
const _KA_KIND_EXPAND = Int32(5)

@inline function ka_axis_clamp(ca::Int, la::Int, cb::Int, lb::Int)
    k = lb - la
    a0 = ca << k
    a1 = ((ca + 1) << k) - 1
    return cb < a0 ? a0 - cb : (cb > a1 ? cb - a1 : 0)
end

# Mirror of `_adt_cuda_classify`: returns (kind, dem_out, leaf_a, leaf_b).
@inline function ka_dtr_classify(node_levels, node_coords, child_ranges, node_sigma,
        ia::Int, ib::Int, dem::Bool, q::Int, gate::Bool, rho_t::TG,
        delta_min2::TG, ell_max::Int) where TG
    @inbounds begin
        la = Int(node_levels[ia]); lb = Int(node_levels[ib])
        ax = Int(node_coords[1, ia]); ay = Int(node_coords[2, ia]); az = Int(node_coords[3, ia])
        bx = Int(node_coords[1, ib]); by = Int(node_coords[2, ib]); bz = Int(node_coords[3, ib])
        local dx::Int, dy::Int, dz::Int
        if la == lb
            dx = bx - ax; dy = by - ay; dz = bz - az
        elseif la < lb
            dx = ka_axis_clamp(ax, la, bx, lb)
            dy = ka_axis_clamp(ay, la, by, lb)
            dz = ka_axis_clamp(az, la, bz, lb)
        else
            dx = ka_axis_clamp(bx, lb, ax, la)
            dy = ka_axis_clamp(by, lb, ay, la)
            dz = ka_axis_clamp(bz, lb, az, la)
        end
        near = dem || (dx * dx + dy * dy + dz * dz <= q)
        dem_out = dem
        if !near && gate
            # integer-exact squared AABB gap on the finest lattice (host mirror)
            sa = 1 << (ell_max - la)
            sb = 1 << (ell_max - lb)
            g2 = 0
            g = max(ax * sa - (bx * sb + sb), bx * sb - (ax * sa + sa), 0); g2 += g * g
            g = max(ay * sa - (by * sb + sb), by * sb - (ay * sa + sa), 0); g2 += g * g
            g = max(az * sa - (bz * sb + sb), bz * sb - (az * sa + sa), 0); g2 += g * g
            cut = rho_t * TG(node_sigma[ib])
            if delta_min2 * TG(g2) < cut * cut
                near = true
                dem_out = true
            end
        end
        leaf_a = child_ranges[2, ia] == 0
        leaf_b = child_ranges[2, ib] == 0
        local kind::Int32
        if !near
            kind = la == lb ? _KA_KIND_V : (la < lb ? _KA_KIND_W : _KA_KIND_X)
        elseif leaf_a && leaf_b
            kind = _KA_KIND_U
        else
            kind = _KA_KIND_EXPAND
        end
    end
    return kind, dem_out, leaf_a, leaf_b
end

@inline function ka_expand_count(node_levels, child_ranges, ia::Int, ib::Int,
        leaf_a::Bool, leaf_b::Bool)
    @inbounds begin
        la = Int(node_levels[ia]); lb = Int(node_levels[ib])
        na = Int(child_ranges[2, ia]); nb = Int(child_ranges[2, ib])
        if la == lb
            leaf_a && return nb
            leaf_b && return na
            return na * nb
        end
        return la < lb ? nb : na
    end
end

# j-th (1-based) child pair of an EXPAND pair, in the host's deterministic
# (ja-major, jb-minor) order.
@inline function ka_expand_get(node_levels, child_ranges, ia::Int, ib::Int,
        leaf_a::Bool, leaf_b::Bool, j::Int)
    @inbounds begin
        la = Int(node_levels[ia]); lb = Int(node_levels[ib])
        if la == lb
            if leaf_a
                return ia, Int(child_ranges[1, ib]) + j - 1
            elseif leaf_b
                return Int(child_ranges[1, ia]) + j - 1, ib
            else
                nb = Int(child_ranges[2, ib])
                return Int(child_ranges[1, ia]) + (j - 1) ÷ nb,
                    Int(child_ranges[1, ib]) + (j - 1) % nb
            end
        elseif la < lb
            return ia, Int(child_ranges[1, ib]) + j - 1
        else
            return Int(child_ranges[1, ia]) + j - 1, ib
        end
    end
end

@kernel function ka_dtr_seed_kernel!(fa, fb, fdem)
    i = @index(Global)
    @inbounds if i == 1
        fa[1] = Int32(1)
        fb[1] = Int32(1)
        fdem[1] = Int32(0)
    end
end

# want: 1..5 kind flags; 6 = demotion-trigger diagnostic count
@kernel function ka_dtr_flags_kernel!(flags, @Const(fa), @Const(fb), @Const(fdem), np,
        @Const(node_levels), @Const(node_coords), @Const(child_ranges), @Const(node_sigma),
        q, gate, rho_t, delta_min2, ell_max, want)
    p = @index(Global)
    @inbounds if p <= np
        ia = Int(fa[p]); ib = Int(fb[p]); dem = fdem[p] != Int32(0)
        kind, dem_out, _, _ = ka_dtr_classify(node_levels, node_coords, child_ranges,
            node_sigma, ia, ib, dem, q, gate, rho_t, delta_min2, ell_max)
        if want == Int32(6)
            flags[p] = (dem_out && !dem) ? Int32(1) : Int32(0)
        else
            flags[p] = kind == want ? Int32(1) : Int32(0)
        end
    end
end

# Emit U/W/X node-id pairs at base offsets (deterministic scan order).
@kernel function ka_dtr_emit_pairs_kernel!(dst_t, dst_s, base, @Const(flags),
        @Const(prefix), @Const(fa), @Const(fb), np)
    p = @index(Global)
    @inbounds if p <= np && flags[p] == Int32(1)
        idx = base + Int(prefix[p])
        dst_t[idx] = fa[p]
        dst_s[idx] = fb[p]
    end
end

# Emit V pairs with the global class id; validity per the 025 phase-table
# membership (sticky-demotion invariant) via the violation flag.
@kernel function ka_dtr_emit_v_kernel!(vt, vs, vc, base, @Const(flags), @Const(prefix),
        @Const(fa), @Const(fb), np, @Const(node_levels), @Const(node_coords),
        @Const(offset_lut), @Const(level_class_of), reach, noffsets, first_m2l_level,
        violation_flags)
    p = @index(Global)
    @inbounds if p <= np && flags[p] == Int32(1)
        ia = Int(fa[p]); ib = Int(fb[p])
        la = Int(node_levels[ia])
        ox = Int(node_coords[1, ia]) - Int(node_coords[1, ib])
        oy = Int(node_coords[2, ia]) - Int(node_coords[2, ib])
        oz = Int(node_coords[3, ia]) - Int(node_coords[3, ib])
        k = 0
        if abs(ox) <= reach && abs(oy) <= reach && abs(oz) <= reach
            k = Int(offset_lut[ox + reach + 1, oy + reach + 1, oz + reach + 1])
        end
        phase = 1 + (Int(node_coords[1, ib]) & 1) + 2 * (Int(node_coords[2, ib]) & 1) +
            4 * (Int(node_coords[3, ib]) & 1)
        ok = k != 0 && la >= first_m2l_level &&
            level_class_of[phase, k, la + 1] != Int32(0)
        ok || (violation_flags[1] = Int32(1))
        idx = base + Int(prefix[p])
        vt[idx] = fa[p]
        vs[idx] = fb[p]
        vc[idx] = Int32((la - first_m2l_level) * noffsets + k)
    end
end

@kernel function ka_dtr_expand_count_kernel!(flags, @Const(fa), @Const(fb), @Const(fdem),
        np, @Const(node_levels), @Const(node_coords), @Const(child_ranges),
        @Const(node_sigma), q, gate, rho_t, delta_min2, ell_max)
    p = @index(Global)
    @inbounds if p <= np
        ia = Int(fa[p]); ib = Int(fb[p]); dem = fdem[p] != Int32(0)
        kind, _, leaf_a, leaf_b = ka_dtr_classify(node_levels, node_coords, child_ranges,
            node_sigma, ia, ib, dem, q, gate, rho_t, delta_min2, ell_max)
        flags[p] = kind == _KA_KIND_EXPAND ?
            Int32(ka_expand_count(node_levels, child_ranges, ia, ib, leaf_a, leaf_b)) :
            Int32(0)
    end
end

@kernel function ka_dtr_expand_emit_kernel!(ga, gb, gdem, @Const(fa), @Const(fb),
        @Const(fdem), np, @Const(prefix), @Const(node_levels), @Const(node_coords),
        @Const(child_ranges), @Const(node_sigma), q, gate, rho_t, delta_min2, ell_max)
    p = @index(Global)
    @inbounds if p <= np
        ia = Int(fa[p]); ib = Int(fb[p]); dem = fdem[p] != Int32(0)
        kind, dem_out, leaf_a, leaf_b = ka_dtr_classify(node_levels, node_coords,
            child_ranges, node_sigma, ia, ib, dem, q, gate, rho_t, delta_min2, ell_max)
        if kind == _KA_KIND_EXPAND
            # exclusive prefix from the inclusive scan, as CUDA's emit kernel does
            base = p == 1 ? 0 : Int(prefix[p - 1])
            cnt = ka_expand_count(node_levels, child_ranges, ia, ib, leaf_a, leaf_b)
            d = dem_out ? Int32(1) : Int32(0)
            for j in 1:cnt
                ja, jb = ka_expand_get(node_levels, child_ranges, ia, ib, leaf_a, leaf_b, j)
                ga[base + j] = Int32(ja)
                gb[base + j] = Int32(jb)
                gdem[base + j] = d
            end
        end
    end
end

"""
    KAAdaptiveListsContext

Preallocated buffers for the DTR/list phases (E/F/G), mirroring the list portion
of CUDA's `DeviceAdaptiveCUDAContext`.

Constructed against the `KAAdaptiveTreeContext` whose tree it will build lists
for, and **shares that context's frontier-sized scratch** rather than allocating
a second copy: the DTR pair frontier aliases the tree-build frontier ping-pong
(`a_lev`/`a_lo`/`a_hi` and `b_lev`/`b_lo`/`b_hi`, all `Int32` at
`frontier_capacity`), and the DTR scan aliases `bl_flags`/`bl_prefix`. That is
safe because a tree build always completes before its list build, and CUDA's own
`actx` shares buffers the same way (the U CSR reusing the V sort scratch is the
same trick, kept below). At n=1e6 with `frontier_capacity` ~2e7 this is the
difference between ~640MB of duplicate frontier scratch and none.

**Ordering requirement**: do not interleave `ka_build_adaptive_tree!` and
`ka_adaptive_build_lists!` on the same pair of contexts -- finish the tree, then
build its lists. Rebuilding the tree invalidates any list built from it anyway.

`offset_lut`/`level_class_of` are the geometry LUTs from an
`AdaptiveInteractionLists` (moved to the backend by the caller).
"""
struct KAAdaptiveListsContext{B,L3,C3,NT<:NamedTuple}
    backend::B
    frontier_capacity::Int
    u_capacity::Int
    v_capacity::Int
    wx_capacity::Int
    offset_lut::L3
    level_class_of::C3
    lut_reach::Int
    noffsets::Int
    first_m2l_level::Int
    ell_max::Int
    nclasses::Int
    class_starts::Vector{Int}
    level_starts::Vector{Int}
    host_class_counts::Vector{Int32}
    bufs::NT
end

function ka_allocate_lists_context(actx::KAAdaptiveTreeContext, offset_lut,
        level_class_of; u_capacity::Int, v_capacity::Int, wx_capacity::Int,
        lut_reach::Int, noffsets::Int, first_m2l_level::Int, ell_max::Int=0,
        nclasses::Int=max(1, noffsets * (ell_max + 1 - first_m2l_level)),
        leaf_capacity::Int=actx.leaf_capacity, maxn::Int=actx.maxn)
    backend = actx.backend
    FC = actx.frontier_capacity
    UC, VC, WC = u_capacity, v_capacity, wx_capacity
    LC, MN = max(1, leaf_capacity), max(1, maxn)
    t = actx.bufs
    bufs = (
        # DTR pair frontier + scan: aliases of the tree-build frontier scratch
        # (see the docstring). Same element type and length; the tree build is
        # finished by the time any of these are read.
        fa=t.a_lev, fb=t.a_lo, fdem=t.a_hi,
        fa2=t.b_lev, fb2=t.b_lo, fdem2=t.b_hi,
        flags=t.bl_flags, prefix=t.bl_prefix,
        u_targets=KA.zeros(backend, Int32, UC), u_sources=KA.zeros(backend, Int32, UC),
        vstage_targets=KA.zeros(backend, Int32, VC),
        vstage_sources=KA.zeros(backend, Int32, VC),
        vstage_class=KA.zeros(backend, Int32, VC),
        w_targets=KA.zeros(backend, Int32, WC), w_sources=KA.zeros(backend, Int32, WC),
        x_targets=KA.zeros(backend, Int32, WC), x_sources=KA.zeros(backend, Int32, WC),
        # slot 1: V phase-table violation; slot 2: U endpoint not a leaf slot
        violation_flags=KA.zeros(backend, Int32, 2),

        # Phase F: V class partition into the CSR route stream
        vsort_keys=KA.zeros(backend, UInt64, VC), vsort_ix=KA.zeros(backend, Int, VC),
        route_targets=KA.zeros(backend, Int, VC), route_sources=KA.zeros(backend, Int, VC),
        route_class=KA.zeros(backend, Int32, VC),
        route_class_offset=KA.zeros(backend, Int32, VC),
        class_counts_dev=KA.zeros(backend, Int32, nclasses),

        # Phase G: U endpoints -> leaf slots, then target-major U CSR
        direct_targets=KA.zeros(backend, Int, UC),
        direct_sources=KA.zeros(backend, Int, UC),
        u_csr_offsets=KA.zeros(backend, Int32, LC + 1),
        u_csr_sources=KA.zeros(backend, Int32, UC),
        u_csr_body_leaf=KA.zeros(backend, Int32, MN),
    )
    # Guard the aliasing assumption rather than trusting it silently.
    all(x -> length(x) >= FC, (bufs.fa, bufs.fb, bufs.fdem, bufs.fa2, bufs.fb2,
        bufs.fdem2, bufs.flags, bufs.prefix)) || throw(ArgumentError(
        "tree context's frontier scratch is smaller than frontier_capacity=$FC"))
    return KAAdaptiveListsContext(backend, FC, UC, VC, WC, offset_lut, level_class_of,
        lut_reach, noffsets, first_m2l_level, ell_max, nclasses,
        zeros(Int, nclasses + 1), zeros(Int, ell_max + 2), zeros(Int32, nclasses), bufs)
end

"""
    ka_adaptive_build_lists!(lctx, node_levels, node_coords, child_ranges, node_sigma;
                             ell_max, near_radius2, gate, rho_t, delta_min2, workgroup=64)

Backend-agnostic port of `_cuda_adaptive_build_lists!`: the frontier DTR sweep
(theory §2.7) producing the U/V/W/X interaction lists from a finalized adaptive
octree. Returns `(n_u, n_v, n_w, n_x, n_dem)`; the lists themselves live in
`lctx.bufs`, valid over `1:n_*`.

`rho_t`/`delta_min2` may be any `Real`; they are converted to `gate_type`,
which defaults to `ka_gate_float_type(node_sigma)` and controls the sigma-gate
precision (see the precision note above). Pass `gate_type=Float64` on CUDA to
mirror `_adt_cuda_classify` bit-for-bit.
"""
function ka_adaptive_build_lists!(lctx::KAAdaptiveListsContext, node_levels, node_coords,
        child_ranges, node_sigma; ell_max::Int, near_radius2::Int, gate::Bool,
        rho_t::Real, delta_min2::Real,
        gate_type::Type{TG}=ka_gate_float_type(node_sigma),
        workgroup::Int=KA_AUTO_WORKGROUP) where {TG<:AbstractFloat}
    backend = lctx.backend
    b = lctx.bufs
    # Convert once, on the host: the kernels take these as scalar arguments, so
    # every pair sees the identical value and the gate stays in exactly TG.
    rho_t_g = TG(rho_t)
    delta_min2_g = TG(delta_min2)
    fill!(b.violation_flags, Int32(0))

    seedk = _cached_kernel(ka_dtr_seed_kernel!, backend, 1)
    flagk = _cached_kernel(ka_dtr_flags_kernel!, backend, workgroup)
    emitk = _cached_kernel(ka_dtr_emit_pairs_kernel!, backend, workgroup)
    emitvk = _cached_kernel(ka_dtr_emit_v_kernel!, backend, workgroup)
    ecountk = _cached_kernel(ka_dtr_expand_count_kernel!, backend, workgroup)
    eemitk = _cached_kernel(ka_dtr_expand_emit_kernel!, backend, workgroup)

    a = (b.fa, b.fb, b.fdem)
    bb = (b.fa2, b.fb2, b.fdem2)
    seedk(a[1], a[2], a[3]; ndrange=1)

    node_args = (node_levels, node_coords, child_ranges, node_sigma)
    gate_args = (near_radius2, gate, rho_t_g, delta_min2_g, ell_max)

    np = 1
    n_u = 0; n_v = 0; n_w = 0; n_x = 0; n_dem = 0
    rounds = 0
    while np > 0
        rounds += 1
        rounds <= 2 * ell_max + 3 || throw(AssertionError(
            "adaptive KA DTR failed to terminate"))

        # U / W / X share the plain pair-emit path; only the `want` code, the
        # running count and the destination buffers differ. Written out rather
        # than looped, mirroring `_cuda_adaptive_build_lists!`.
        emit_kind! = function (want, base, cap, dst_t, dst_s, label)
            flagk(b.flags, a[1], a[2], a[3], np, node_args..., gate_args..., want;
                ndrange=np)
            m = _ka_scan_total!(b.flags, b.prefix, np)
            base + m <= cap || throw(AssertionError(
                "adaptive KA $label list capacity $cap exceeded"))
            m > 0 && emitk(dst_t, dst_s, base, b.flags, b.prefix, a[1], a[2], np; ndrange=np)
            return m
        end
        n_u += emit_kind!(_KA_KIND_U, n_u, lctx.u_capacity, b.u_targets, b.u_sources, "U")
        n_w += emit_kind!(_KA_KIND_W, n_w, lctx.wx_capacity, b.w_targets, b.w_sources, "W")
        n_x += emit_kind!(_KA_KIND_X, n_x, lctx.wx_capacity, b.x_targets, b.x_sources, "X")

        # V carries the geometry class id, so it needs its own emit kernel.
        flagk(b.flags, a[1], a[2], a[3], np, node_args..., gate_args..., _KA_KIND_V;
            ndrange=np)
        m = _ka_scan_total!(b.flags, b.prefix, np)
        n_v + m <= lctx.v_capacity || throw(AssertionError(
            "adaptive KA V route capacity $(lctx.v_capacity) exceeded"))
        if m > 0
            emitvk(b.vstage_targets, b.vstage_sources, b.vstage_class, n_v, b.flags,
                b.prefix, a[1], a[2], np, node_levels, node_coords, lctx.offset_lut,
                lctx.level_class_of, lctx.lut_reach, lctx.noffsets, lctx.first_m2l_level,
                b.violation_flags; ndrange=np)
        end
        n_v += m

        # demotion diagnostic
        if gate
            flagk(b.flags, a[1], a[2], a[3], np, node_args..., gate_args..., Int32(6);
                ndrange=np)
            n_dem += _ka_scan_total!(b.flags, b.prefix, np)
        end

        # expand: count child pairs per frontier entry, scan, emit next frontier
        ecountk(b.flags, a[1], a[2], a[3], np, node_args..., gate_args...; ndrange=np)
        np2 = _ka_scan_total!(b.flags, b.prefix, np)
        np2 <= lctx.frontier_capacity || throw(AssertionError(
            "adaptive KA DTR frontier capacity $(lctx.frontier_capacity) exceeded"))
        np2 > 0 && eemitk(bb[1], bb[2], bb[3], a[1], a[2], a[3], np, b.prefix,
            node_args..., gate_args...; ndrange=np)
        a, bb = bb, a
        np = np2
    end

    # One sync at the end, not one per launch. KA kernels on a backend are
    # ordered on that backend's own queue, so consecutive launches need no
    # explicit barrier, and each `_ka_scan_total!` already forces a sync via its
    # host readback of the scan total. Per-kernel `KA.synchronize` is what cost
    # the M2M port 53.6ms vs 20ms before it was removed there (commit ff14b7e).
    KA.synchronize(backend)
    Int(Array(view(b.violation_flags, 1:1))[1]) == 0 || throw(AssertionError(
        "adaptive KA V pair lies outside the task-025 phase-table class set — " *
        "the sticky demotion invariant (theory §2.4/§5.2) is violated"))

    return n_u, n_v, n_w, n_x, n_dem
end

#------- Phase F: V-list class partition into the CSR route stream -------#
#
# Port of `_cuda_adaptive_partition_v!` and its three kernels
# (src/tree_batched_cuda.jl:1143-1218). The V stage stream carries a class id
# per pair; the routes have to be grouped by class so each M2L class becomes one
# contiguous slab. Sorting on `(class << 32) | emission_index` makes the
# partition stable by construction -- ties inside a class keep DTR emission
# order -- so no separate stable-sort primitive is needed.

@kernel function ka_vsort_keys_kernel!(keys, @Const(vclass), n_v)
    i = @index(Global)
    @inbounds if i <= n_v
        keys[i] = (UInt64(vclass[i]) << 32) | UInt64(i)
    end
end

@kernel function ka_csr_gather_kernel!(route_targets, route_sources, route_class,
        route_class_offset, @Const(vsort_ix), @Const(vt), @Const(vs), @Const(vc),
        n_v, noffsets)
    p = @index(Global)
    @inbounds if p <= n_v
        i = Int(vsort_ix[p])
        route_targets[p] = Int(vt[i])
        route_sources[p] = Int(vs[i])
        c = Int(vc[i])
        route_class[p] = Int32(c)
        # per-offset id for the dense family (class_base = 0 convention)
        route_class_offset[p] = Int32(c - ((c - 1) ÷ noffsets) * noffsets)
    end
end

@kernel function ka_class_histogram_kernel!(counts, @Const(vclass), n_v)
    i = @index(Global)
    @inbounds if i <= n_v
        KA.@atomic counts[Int(vclass[i])] += Int32(1)
    end
end

"""
    ka_adaptive_partition_v!(lctx, n_v; workgroup=64)

Backend-agnostic port of `_cuda_adaptive_partition_v!`: deterministically
partitions the `n_v` staged V pairs by class into the CSR route stream
(`lctx.bufs.route_*`, valid over `1:n_v`) and fills `lctx.class_starts` /
`lctx.level_starts` on the host. Returns `n_v` (the route count).
"""
function ka_adaptive_partition_v!(lctx::KAAdaptiveListsContext, n_v::Int;
        workgroup::Int=KA_AUTO_WORKGROUP)
    backend = lctx.backend
    b = lctx.bufs
    cc = lctx.host_class_counts

    if n_v == 0
        fill!(cc, Int32(0))
    else
        keysk = _cached_kernel(ka_vsort_keys_kernel!, backend, workgroup)
        gatherk = _cached_kernel(ka_csr_gather_kernel!, backend, workgroup)
        histk = _cached_kernel(ka_class_histogram_kernel!, backend, workgroup)

        keysk(b.vsort_keys, b.vstage_class, n_v; ndrange=n_v)
        # `sortperm!` dispatches to the backend's own sort (confirmed working on
        # Metal in Phase B), matching CUDA's `_cuda_sortperm_into!` convention.
        sortperm!(view(b.vsort_ix, 1:n_v), view(b.vsort_keys, 1:n_v))
        gatherk(b.route_targets, b.route_sources, b.route_class, b.route_class_offset,
            b.vsort_ix, b.vstage_targets, b.vstage_sources, b.vstage_class, n_v,
            lctx.noffsets; ndrange=n_v)
        fill!(b.class_counts_dev, Int32(0))
        histk(b.class_counts_dev, b.vstage_class, n_v; ndrange=n_v)
        copyto!(cc, Array(b.class_counts_dev))   # forces the sync
    end

    cs = lctx.class_starts
    cs[1] = 1
    @inbounds for c in 1:lctx.nclasses
        cs[c + 1] = cs[c] + Int(cc[c])
    end
    ls = lctx.level_starts
    first = lctx.first_m2l_level
    @inbounds for L in 0:lctx.ell_max
        ls[L + 1] = L < first ? 1 : cs[(L - first) * lctx.noffsets + 1]
    end
    ls[lctx.ell_max + 2] = cs[end]
    return n_v
end


#------- Phase G: U endpoints -> leaf slots, then the target-major U CSR -------#
#
# Port of `_adt_cuda_u_slots_kernel!` and `_cuda_adaptive_build_u_csr!` with its
# three kernels (src/tree_batched_cuda.jl:1222-1324). The DTR emits U pairs in
# frontier-major order, which is NOT target-major, so building the fused
# nearfield's target-owned CSR needs the same stable (target-slot, index) key
# sort the V partition uses.
#
# The offsets kernel writes the half-open slot range (tprev, t] for each CSR
# position, which fills in leaves that own no U pairs at all; slots past the
# last occupied target keep the `n_u + 1` prefill. Targets are sorted, so those
# ranges are disjoint and the concurrent writes never overlap.

@kernel function ka_u_slots_kernel!(direct_targets, direct_sources, @Const(u_targets),
        @Const(u_sources), @Const(leaf_slot_of), n_u, violation_flags)
    k = @index(Global)
    @inbounds if k <= n_u
        ts = leaf_slot_of[Int(u_targets[k])]
        ss = leaf_slot_of[Int(u_sources[k])]
        (ts == Int32(0) || ss == Int32(0)) && (violation_flags[2] = Int32(1))
        direct_targets[k] = Int(ts)
        direct_sources[k] = Int(ss)
    end
end

@kernel function ka_body_leaf_kernel!(body_leaf, @Const(cell_ranges), n_leaves)
    l = @index(Global)
    @inbounds if l <= n_leaves
        first = cell_ranges[1, l]
        last = first + cell_ranges[2, l] - 1
        i = first
        while i <= last
            body_leaf[i] = Int32(l)
            i += 1
        end
    end
end

@kernel function ka_usort_keys_kernel!(keys, @Const(direct_targets), n_u)
    i = @index(Global)
    @inbounds if i <= n_u
        keys[i] = (UInt64(direct_targets[i]) << 32) | UInt64(i)
    end
end

@kernel function ka_ucsr_gather_kernel!(u_csr_sources, @Const(usort_ix),
        @Const(direct_sources), n_u)
    p = @index(Global)
    @inbounds if p <= n_u
        u_csr_sources[p] = Int32(direct_sources[Int(usort_ix[p])])
    end
end

@kernel function ka_ucsr_offsets_kernel!(u_csr_offsets, @Const(usort_ix),
        @Const(direct_targets), n_u)
    p = @index(Global)
    @inbounds if p <= n_u
        t = Int(direct_targets[Int(usort_ix[p])])
        tprev = p == 1 ? 0 : Int(direct_targets[Int(usort_ix[p - 1])])
        s = tprev + 1
        while s <= t
            u_csr_offsets[s] = Int32(p)
            s += 1
        end
    end
end

"""
    ka_adaptive_u_slots!(lctx, leaf_slot_of, n_u; workgroup=64)

Map the `n_u` U pairs' node ids to leaf-cell slots, into
`lctx.bufs.direct_targets`/`direct_sources`. Throws if either endpoint of any
pair is not a leaf (which would mean the DTR produced a non-leaf U pair).
"""
function ka_adaptive_u_slots!(lctx::KAAdaptiveListsContext, leaf_slot_of, n_u::Int;
        workgroup::Int=KA_AUTO_WORKGROUP)
    n_u == 0 && return nothing
    backend = lctx.backend
    b = lctx.bufs
    k = _cached_kernel(ka_u_slots_kernel!, backend, workgroup)
    k(b.direct_targets, b.direct_sources, b.u_targets, b.u_sources, leaf_slot_of,
        n_u, b.violation_flags; ndrange=n_u)
    # the readback below is itself the sync point
    Int(Array(view(b.violation_flags, 2:2))[1]) == 0 || throw(AssertionError(
        "adaptive KA U pair endpoint is not a leaf cell slot"))
    return nothing
end

"""
    ka_adaptive_build_u_csr!(lctx, cell_ranges, n_leaves, n_u; workgroup=64)

Backend-agnostic port of `_cuda_adaptive_build_u_csr!`: builds the target-major
CSR (`u_csr_offsets` over `1:n_leaves+1`, `u_csr_sources` over `1:n_u`) from the
slot-mapped U list produced by `ka_adaptive_u_slots!`, plus the body -> leaf-slot
map for the dense body-packed nearfield shape. Reuses the Phase F sort scratch,
as CUDA does.
"""
function ka_adaptive_build_u_csr!(lctx::KAAdaptiveListsContext, cell_ranges,
        n_leaves::Int, n_u::Int; workgroup::Int=KA_AUTO_WORKGROUP)
    backend = lctx.backend
    b = lctx.bufs
    n_leaves + 1 <= length(b.u_csr_offsets) || throw(AssertionError(
        "KA U-CSR offsets capacity $(length(b.u_csr_offsets)) exceeded " *
        "(n_leaves=$n_leaves)"))
    fill!(view(b.u_csr_offsets, 1:(n_leaves + 1)), Int32(n_u + 1))

    if n_u > 0
        n_u <= length(b.vsort_keys) || throw(AssertionError(
            "KA U-CSR reuses the V sort scratch; n_u=$n_u exceeds v_capacity " *
            "$(length(b.vsort_keys))"))
        keysk = _cached_kernel(ka_usort_keys_kernel!, backend, workgroup)
        gatherk = _cached_kernel(ka_ucsr_gather_kernel!, backend, workgroup)
        offsk = _cached_kernel(ka_ucsr_offsets_kernel!, backend, workgroup)

        keysk(b.vsort_keys, b.direct_targets, n_u; ndrange=n_u)
        sortperm!(view(b.vsort_ix, 1:n_u), view(b.vsort_keys, 1:n_u))
        gatherk(b.u_csr_sources, b.vsort_ix, b.direct_sources, n_u; ndrange=n_u)
        offsk(b.u_csr_offsets, b.vsort_ix, b.direct_targets, n_u; ndrange=n_u)
    end

    if n_leaves > 0
        bodyk = _cached_kernel(ka_body_leaf_kernel!, backend, workgroup)
        bodyk(b.u_csr_body_leaf, cell_ranges, n_leaves; ndrange=n_leaves)
    end
    KA.synchronize(backend)     # single sync: results are caller-visible after this
    return nothing
end

#------- Harness front end: position -> full-depth key, and a full-build driver -------#
#
# Not a CUDA-parity phase (no `_cuda_*` counterpart is ported one-to-one here) --
# this stitches Phases A-D together into a single from-scratch tree build, the way
# `_cuda_refresh_adaptive_tree!` (tree_batched_cuda.jl:1387) stitches the hand-written
# CUDA stages, so the 4-way tree-build benchmark can drive local-Metal and HPC-KA off
# one shared, backend-agnostic entry point.

@kernel function ka_radix_keys_kernel!(keys, @Const(positions), x_min, h0, ell, n)
    i = @index(Global)
    @inbounds if i <= n
        G = 1 << ell
        delta = (2 * h0) / G
        px = positions[1, i]
        py = positions[2, i]
        pz = positions[3, i]
        ix = clamp(floor(Int, (px - x_min[1]) / delta), 0, G - 1)
        iy = clamp(floor(Int, (py - x_min[2]) / delta), 0, G - 1)
        iz = clamp(floor(Int, (pz - x_min[3]) / delta), 0, G - 1)
        keys[i] = ka_morton_key(ix, iy, iz, ell)
    end
end

"""
    ka_radix_keys!(keys, positions, x_min, h0, ell; workgroup=64)

Backend-agnostic port of `_cuda_radix_keys_checked_kernel!` (translate_batched_cuda.jl),
minus its out-of-bounds flag (bodies are assumed to already lie in the fixed root cube
`[x_min, x_min + 2*h0]^3` -- true by construction for a from-scratch benchmark build;
callers needing the OOB guard for a live refresh loop should add it at the call site).
Writes each body's full-depth (`ell`-level) Morton key from its position into `keys`.
`positions` is a `3 x n` backend matrix; `x_min` is a plain 3-tuple/SVector, not a
device array (mirrors CUDA's `grid.x_min` convention).
"""
function ka_radix_keys!(keys::AbstractVector{UInt64}, positions::AbstractMatrix,
        x_min, h0, ell::Int; workgroup::Int=KA_AUTO_WORKGROUP)
    n = length(keys)
    backend = KA.get_backend(keys)
    keysk = _cached_kernel(ka_radix_keys_kernel!, backend, workgroup)
    keysk(keys, positions, x_min, h0, ell, n; ndrange=n)
    KA.synchronize(backend)
    return keys
end

"""
    ka_build_adaptive_tree!(actx, positions, ell_max, K_max, balance, x_min, h0::TF;
                             sigma_row=0, source_bodies=nothing, workgroup=64)

From-scratch adaptive-octree build from raw body positions: full-depth Morton keys
(`ka_radix_keys!`) + sort (`sortperm!`/`ka_gather_values!`, same as Phases B/C) feeding
`ka_adaptive_build_leaves!` (Phase A) -> `ka_adaptive_balance!` (Phase B, if `balance`)
-> `ka_adaptive_finalize!` (Phase C) -> `ka_adaptive_sigma_sweep!` (Phase D, if
`sigma_row > 0` and `source_bodies` given). `positions` is a `3 x n` backend matrix
(`n <= actx.maxn`); `x_min`/`h0` are the fixed root cube (plain scalars, not device
arrays). Mirrors the stage order of `_cuda_refresh_adaptive_tree!` for a first
(non-incremental) build. All scratch/output buffers come from `actx` (see
`KAAdaptiveTreeContext`/`ka_allocate_adaptive_context`) -- zero recurring allocation,
matching CUDA-native's `actx` convention.

Returns Phase C's `NamedTuple` merged with `grid`, `perm`, `invperm`, `sorted_keys`,
`n_balance_splits`, and `node_sigma_max` (`nothing` if the sigma sweep was not armed).
"""
function ka_build_adaptive_tree!(actx::KAAdaptiveTreeContext, positions::AbstractMatrix,
        ell_max::Int, K_max::Int, balance::Bool, x_min, h0::TF; sigma_row::Int=0,
        source_bodies=nothing, workgroup::Int=KA_AUTO_WORKGROUP,
        stage_ns::Union{Nothing,Vector{UInt64}}=nothing) where TF
    backend = actx.backend
    n = size(positions, 2)
    n <= actx.maxn || error("adaptive KA context maxn=$(actx.maxn) exceeded by n=$n")
    b = actx.bufs

    # Optional per-stage timing (mirrors CUDA-native's actx.profile_stages/
    # stage_ns convention, tree_batched_cuda.jl:1387-1431) to root-cause the
    # n>=1e5 timing anomaly (project_fastmultipole_ka_migration memory).
    t0 = stage_ns !== nothing ? (KernelAbstractions.synchronize(backend); time_ns()) : UInt64(0)

    keys = view(b.keys, 1:n)
    ka_radix_keys!(keys, positions, x_min, h0, ell_max; workgroup=workgroup)

    perm = view(b.perm, 1:n)
    sortperm!(perm, keys)
    sorted_keys = view(b.sorted_keys, 1:n)
    ka_gather_values!(sorted_keys, keys, perm; workgroup=workgroup)
    invperm = view(b.invperm, 1:n)
    ka_fill_invperm!(invperm, perm; workgroup=workgroup)

    if stage_ns !== nothing
        KernelAbstractions.synchronize(backend)
        stage_ns[1] = time_ns() - t0; t0 = time_ns()
    end

    nl, llev, lkey, llo, lhi = ka_adaptive_build_leaves!(actx, sorted_keys, ell_max, K_max, n;
        workgroup=workgroup)

    if stage_ns !== nothing
        KernelAbstractions.synchronize(backend)
        stage_ns[2] = time_ns() - t0; t0 = time_ns()
    end

    n_balance_splits = 0
    if balance
        nl, n_balance_splits, llev, lkey, llo, lhi = ka_adaptive_balance!(actx, nl, llev, lkey,
            llo, lhi, sorted_keys, ell_max; workgroup=workgroup)
    end

    if stage_ns !== nothing
        KernelAbstractions.synchronize(backend)
        stage_ns[3] = time_ns() - t0; t0 = time_ns()
    end

    fin = ka_adaptive_finalize!(actx, nl, llev, lkey, llo, lhi, sorted_keys, ell_max, n, x_min,
        h0; workgroup=workgroup)

    # Publish the build into the context's DeviceRadixGrid. The arrays were
    # written in place through the `bufs` aliases; only the scalars need setting,
    # the same five `_cuda_refresh_adaptive_tree!` assigns.
    grid = actx.grid
    grid.x_min = SVector{3,TF}(x_min[1], x_min[2], x_min[3])
    grid.h0 = h0
    grid.ell = ell_max
    grid.n_bodies = n
    grid.n_cells = fin.n_leaves
    ka_fill_single_system_attribution!(grid.body_system, grid.body_index, n;
        workgroup=workgroup)

    if stage_ns !== nothing
        KernelAbstractions.synchronize(backend)
        stage_ns[4] = time_ns() - t0
    end

    node_sigma_max = if sigma_row > 0 && source_bodies !== nothing
        ka_adaptive_sigma_sweep!(actx, fin.node_lo, fin.node_hi, fin.child_ranges, fin.n_nodes,
            fin.level_offsets, ell_max, source_bodies, sigma_row; workgroup=workgroup)
    else
        nothing
    end

    return merge(fin, (grid=grid, perm=perm, invperm=invperm, sorted_keys=sorted_keys,
        n_balance_splits=n_balance_splits, node_sigma_max=node_sigma_max))
end

#------- Step (v): handing the built tree to a DeviceResidentRadixState -------#
#
# The analogue is `_cuda_allocate_adaptive_lifecycle` (translate_batched_cuda.jl),
# the ADAPTIVE path's state constructor -- not `cuda_radix_state`, which serves the
# uniform path and assumes exact-length grid arrays (`length(grid.node_keys)` ==
# n_nodes). That distinction is what makes this cheap: the adaptive path already
# hands its own capacity-sized `actx.grid` straight into the state and carries the
# logical extents in `state.counts::RadixStepCounts`, so `actx.grid` goes in as-is,
# by reference, with no truncating views and no host-mirror cross-check.
#
# Scope: this constructs the state. It does not build the interaction list or the
# operator workspace (`ResidentOperatorWorkspace`, whose plan construction is
# CUDA-specific), and it does not run the lifecycle -- B2M and L2B have no KA port.
# `interaction_list` and `scratch` are therefore `nothing`, and the route arrays are
# allocated empty. Those are the next steps, not omissions this one papers over.

function _ka_flat_buffer(backend, ::Type{TF}, basis_info::FastMultipole.OperatorBasisInfo{B,LH},
        batch::Integer) where {TF,B,LH}
    phi = KA.zeros(backend, TF, basis_info.basis_dof_phi, batch)
    chi = LH ? KA.zeros(backend, TF, basis_info.basis_dof_chi, batch) :
        KA.zeros(backend, TF, 0, 0)
    return FastMultipole.FlatCoefficientBuffer{TF,typeof(phi),B,LH}(phi, chi, basis_info)
end

"""
    ka_refresh_adaptive_lists!(lctx, actx, build; near_radius2, ell_max,
                               rho_t=0, sigma_armed=false)

Backend-agnostic port of `tree_batched_cuda.jl`'s `_cuda_refresh_adaptive_lists!`:
run the DTR sweep, partition the V stream into CSR routes, map the U endpoints to
leaf slots and rebuild the target-major U CSR, in that order, against the tree
`ka_build_adaptive_tree!` most recently wrote into `actx.grid`.

Returns the NamedTuple `ka_radix_state`'s `lists` keyword expects. `n_direct` is
the U-pair count and `n_routes` the CSR route count -- the two logical extents that
land in `state.counts`; the remaining counts are diagnostics (`n_dem` is the
sticky-demotion trigger count, nonzero only when the sigma gate is armed).

The lists live in `lctx.bufs` and are valid over `1:n_*`; nothing is copied.
"""
function ka_refresh_adaptive_lists!(lctx::KAAdaptiveListsContext,
        actx::KAAdaptiveTreeContext, build; near_radius2::Int, ell_max::Int,
        rho_t::Real=0, sigma_armed::Bool=false, workgroup::Int=KA_AUTO_WORKGROUP)
    grid = actx.grid
    b = actx.bufs
    n_nodes = build.n_nodes
    n_leaves = build.n_leaves
    gate = sigma_armed && rho_t > 0
    # the finest-lattice cell width, squared; the gate compares an integer-exact
    # squared AABB gap against (rho_t * sigma)^2 in these units
    delta_min = 2 * Float64(grid.h0) / (1 << ell_max)

    n_u, n_v, n_w, n_x, n_dem = ka_adaptive_build_lists!(lctx,
        grid.node_levels, grid.node_coords, grid.child_ranges, b.node_sigma;
        ell_max=ell_max, near_radius2=near_radius2, gate=gate, rho_t=rho_t,
        delta_min2=delta_min * delta_min, workgroup=workgroup)

    n_routes = ka_adaptive_partition_v!(lctx, n_v; workgroup=workgroup)

    n_u <= length(lctx.bufs.direct_targets) || throw(AssertionError(
        "adaptive KA direct capacity $(length(lctx.bufs.direct_targets)) exceeded " *
        "by n_u=$n_u"))
    ka_adaptive_u_slots!(lctx, b.leaf_slot_of, n_u; workgroup=workgroup)
    ka_adaptive_build_u_csr!(lctx, grid.cell_ranges, n_leaves, n_u; workgroup=workgroup)

    return (lctx=lctx, n_direct=n_u, n_routes=n_routes,
        n_u=n_u, n_v=n_v, n_w=n_w, n_x=n_x, n_dem=n_dem, n_nodes=n_nodes)
end

"""
    ka_radix_state(actx, build, source_buffer, P, lamb_helmholtz=Val(false);
                   options, n_root_nodes=1, workgroup=64)

Build a `DeviceResidentRadixState` around the tree `ka_build_adaptive_tree!` just
wrote into `actx.grid`. `build` is that call's return value (its `n_nodes`/`n_leaves`
supply the logical extents, which the grid itself does not carry); `source_buffer` is
the `dpb x n` source buffer in *global* (unsorted) body order, which is gathered into
the state's sorted-order body matrix.

`actx.grid` is stored by reference, not copied: `state.grid === actx.grid`, so a later
rebuild through the same context is visible to the state without reconstructing it.
The grid's arrays stay capacity-sized; every logical extent lives in `state.counts`
(`n_bodies`, `n_cells`, `n_nodes`), exactly as on the CUDA adaptive path.

Pass `lists` -- a `ka_refresh_adaptive_lists!` return -- to wire the interaction
list: `interaction_list` becomes the lists context, the route/direct arrays alias
its device buffers, and `counts.n_routes`/`counts.n_direct` carry their extents.
Omit it and those stay `nothing`/empty with both counts zero.

Still not wired, and `nothing`/empty rather than silently wrong: `scratch` (the
operator workspace builds CUDA-specific M2L plans), the `route_levels`/`route_offsets`
pair (the adaptive path routes by CSR class instead), and the host node/route mirrors. The host *body* mirrors
(`host_body_perm`/`host_body_system_ids`/`host_body_indices`) are downloaded once here,
so they are correct for this build and go stale on the next one -- the CUDA path
re-downloads them per step in `_cuda_update_adaptive_radix_state!`.
"""
function ka_radix_state(actx::KAAdaptiveTreeContext, build, source_buffer,
        P::Integer, lamb_helmholtz::Val{LH}=Val(false);
        options::FastMultipole.RadixLifecycleOptions, lists=nothing,
        scratch=nothing, output_rows::Int=4, n_root_nodes::Int=1,
        workgroup::Int=KA_AUTO_WORKGROUP) where LH
    backend = actx.backend
    grid = actx.grid
    TF = typeof(grid.h0)
    options.precision === TF || throw(ArgumentError(
        "options.precision=$(options.precision) does not match the KA context's " *
        "element type $TF; allocate the context and the options at the same precision"))
    n = grid.n_bodies
    n_cells = grid.n_cells
    n_nodes = build.n_nodes
    n_nodes <= actx.node_capacity || throw(ArgumentError(
        "build n_nodes=$n_nodes exceeds the context node_capacity=$(actx.node_capacity)"))
    n_cells == build.n_leaves || throw(ArgumentError(
        "grid.n_cells=$n_cells disagrees with build.n_leaves=$(build.n_leaves); " *
        "`build` must be the result of the most recent ka_build_adaptive_tree! on `actx`"))

    basis_info = FastMultipole.OperatorBasisInfo(FastMultipole.CompressedComplexBasis(),
        P, lamb_helmholtz)
    counters = FastMultipole.RadixTransferCounters()

    # sorted-order body matrix. Single-system attribution (step iv), so one isys=1
    # pack call covers every column; the kernel keeps CUDA's system indirection so
    # it stays correct when multi-system attribution lands.
    # One row past `data_per_body` carries 1/sigma for the regularized nearfield
    # (see `ka_pack_body_matrix_kernel!`). Rows 1:dpb keep their meaning exactly,
    # so `ka_node_sigma_max!` -- which maxes the TRUE sigma over each subtree to
    # drive the sigma-adequacy ell gate -- reads the same values as before.
    dpb = size(source_buffer, 1)
    sigma_row = _ka_kernel_sigma_row(options.direct_kernel)
    # CONVENTION (shared with `ka_radix_cache_device_build` and
    # `_ka_nf_inv_sigma_row`): for a regularized kernel (sigma_row > 0) the
    # LAST row of `source_bodies` is one past `dpb` and carries 1/sigma. It was
    # disabled only to keep the shape identical with the former native CUDA
    # allocator, which no longer exists.
    inv_sigma_row = sigma_row > 0 ? dpb + 1 : 0
    source_bodies = KA.zeros(backend, TF, dpb + (sigma_row > 0), actx.maxn)
    ka_pack_body_matrix!(source_bodies, source_buffer, grid.perm, grid.body_system,
        grid.body_index, n; isys=1, sigma_row=sigma_row, inv_sigma_row=inv_sigma_row,
        workgroup=workgroup)

    m2m_parent = KA.zeros(backend, Int, actx.node_capacity)
    m2m_child = KA.zeros(backend, Int, actx.node_capacity)
    l2l_parent = KA.zeros(backend, Int, actx.node_capacity)
    l2l_child = KA.zeros(backend, Int, actx.node_capacity)
    ka_tree_routes!(m2m_parent, m2m_child, l2l_parent, l2l_child, grid.parent_index,
        n_nodes; n_root_nodes=n_root_nodes, workgroup=workgroup)

    multipoles = _ka_flat_buffer(backend, TF, basis_info, actx.node_capacity)
    locals = _ka_flat_buffer(backend, TF, basis_info, actx.node_capacity)
    output = KA.zeros(backend, TF, output_rows, actx.maxn)

    # Route/direct wiring. With `lists` given (a `ka_refresh_adaptive_lists!`
    # return), the state ALIASES the lists context's device buffers rather than
    # copying them -- the task-023 zero-recurring-allocation contract, and the same
    # shape as the CUDA adaptive path, where the U-slot kernel writes straight into
    # `state.direct_targets`. `route_levels`/`route_offsets` stay empty on both
    # paths: the adaptive lifecycle routes by CSR class, not by (level, offset), so
    # a capacity-sized zero array there would only look meaningful.
    empty_iv = KA.zeros(backend, Int, 0)
    empty_im = KA.zeros(backend, Int, 3, 0)
    if lists === nothing
        route_targets = route_sources = direct_targets = direct_sources = empty_iv
        n_routes = n_direct = 0
        interaction_list = nothing
    else
        lb = lists.lctx.bufs
        route_targets, route_sources = lb.route_targets, lb.route_sources
        direct_targets, direct_sources = lb.direct_targets, lb.direct_sources
        n_routes, n_direct = lists.n_routes, lists.n_direct
        lists.n_nodes == n_nodes || throw(ArgumentError(
            "lists were built for n_nodes=$(lists.n_nodes) but this build has " *
            "n_nodes=$n_nodes; refresh the lists after the tree build"))
        interaction_list = lists.lctx
    end

    # one-shot download of the host body mirrors (see docstring on staleness)
    host_body_perm = Array{Int}(undef, n)
    host_body_system_ids = Array{Int}(undef, n)
    host_body_indices = Array{Int}(undef, n)
    copyto!(host_body_perm, 1, grid.perm, 1, n)
    copyto!(host_body_system_ids, 1, grid.body_system, 1, n)
    copyto!(host_body_indices, 1, grid.body_index, 1, n)
    counters.metadata_downloads += 3

    return FastMultipole.DeviceResidentRadixState{TF,FastMultipole.CompressedComplexBasis,LH}(
        grid, interaction_list, source_bodies, source_bodies,
        grid.perm, grid.body_system, grid.body_index,
        host_body_perm, host_body_system_ids, host_body_indices,
        nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing,
        grid.cell_centers, grid.cell_ranges,
        m2m_parent, m2m_child, l2l_parent, l2l_child,
        multipoles, locals,
        empty_iv, empty_im, route_targets, route_sources,
        direct_targets, direct_sources, output,
        FastMultipole.OperatorInvariantCache(TF, basis_info), scratch, counters, options,
        FastMultipole.RadixStepCounts(n, n_cells, n_nodes, n_routes, n_direct),
    )
end

