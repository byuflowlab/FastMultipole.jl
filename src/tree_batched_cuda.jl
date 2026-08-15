#------- adaptive octree: CUDA device-resident construction + DTR lists (task 041) -------#
#
# Device mirror of the task-039 host adaptive octree (theory §1.2/§1.4/§2.7),
# included at runtime from translate_batched_cuda.jl (placement rule: radix
# driver code in tree_batched*.jl, GPU code in *_cuda.jl files).
#
# Construction is the theory §1.2 sweep realized as host-orchestrated rounds of
# flag/scan/compact kernels over device data:
#   Phase A: full-depth Morton keys + device sort, then a level-synchronous
#            K_max frontier split (child occupancy by binary search over the
#            sorted body keys — the sorted-Morton occupancy lookup of this row;
#            the dense Σ8^L `node_at` table is never built, so there is no
#            ell <= 8 cap on this path).
#   Phase B: 2:1 balance as Jacobi rounds over the leaf key set (mark by
#            parent-cell binary search, split marked leaves into occupied
#            children). Host processes levels deepest-first within a round;
#            both procedures monotonically split only leaves that must split
#            in the (unique) balance closure, so they terminate at the SAME
#            final leaf set — structural parity with the host tree is exact
#            and asserted by tests.
#   Phase C: finalize the level-major node table from the leaf set: per-level
#            ancestor compaction (sorted-unique), subtree body ranges, node
#            geometry, parent/child links by per-level-block binary search,
#            leaf compaction, and the leaf-as-cell presentation.
#   DTR:     theory §2.7 frontier of (target, source, demoted) pairs, with the
#            §2.1 finer-lattice near clamp, the §5.2 source-side sticky sigma
#            demotion, and deterministic scan-ordered emission of U/V/W/X.
#            V pairs are checked against the 025 phase-table class set on
#            device (violation flag, host-asserted) and class-partitioned
#            into the CSR stream by a deterministic (class, index) key sort.
#
# Zero recurring allocation: every array is capacity-sized at construction and
# every growth point is a loud AssertionError. Host<->device traffic during a
# refresh is limited to pinned 4-byte scalars (scan totals, flags) plus one
# per-epoch class-histogram download — no route or operator uploads after
# construction (023 counter contract; asserted by tests).

#------- capacities -------#

# Device list capacities: the host §6.4 formulas hard-capped to device-sane
# bounds (the measured 039 n=1e6 maxima are V 4.8e7, U 7.7e6, W/X 2.1e6 —
# each default below carries >= 30% headroom over the worst measured case).
# AdaptiveTreePolicy u/v/wx overrides take precedence, as on the host.
function _cuda_adaptive_capacities(policy::AdaptiveTreePolicy, maxn::Int)
    node_cap = _adaptive_node_capacity(policy, maxn)
    leaf_cap = min(node_cap, maxn)
    tables = RigidHierarchicalTables(policy.near_radius2)
    push_max = maximum(tables.phase_starts[ph + 1] - tables.phase_starts[ph]
        for ph in 1:8)
    u_host = 8 * length(tables.near_offsets) * leaf_cap
    v_host = push_max * min(node_cap, 4 * maxn)
    u_cap = policy.u_capacity > 0 ? policy.u_capacity :
        min(u_host, max(16 * maxn, 1 << 20))
    v_cap = policy.v_capacity > 0 ? policy.v_capacity :
        min(v_host, max(64 * maxn, 1 << 22))
    wx_cap = policy.wx_capacity > 0 ? policy.wx_capacity :
        min(u_host, max(4 * maxn, 1 << 18))
    # DTR frontier peak width scales with the V emission volume (measured:
    # wake n=1e6 K=64 with V ~ 2.2e7 overflowed a fixed 1.6e7 frontier,
    # job 13180706), so the frontier tracks the V capacity
    frontier_cap = max(16 * maxn, v_cap, 1 << 22)
    return node_cap, leaf_cap, u_cap, v_cap, wx_cap, frontier_cap, tables
end

#------- allocation -------#

function _cuda_allocate_adaptive_context(::Type{TF}, policy::AdaptiveTreePolicy,
        maxn::Int, x_min::SVector{3,TF}, h0::TF,
        counters::CUDARadixTransferCounters) where TF
    policy.split_veto && throw(ArgumentError(
        "the adaptive device path does not implement the §5.4 split veto " *
        "(default OFF, pending user ratification); construct with " *
        "split_veto=false or run host-resident"))
    node_cap, leaf_cap, u_cap, v_cap, wx_cap, frontier_cap, tables =
        _cuda_adaptive_capacities(policy, maxn)
    node_cap <= typemax(Int32) - 8 || throw(ArgumentError(
        "adaptive device node capacity $node_cap exceeds the Int32 index range"))
    ell_max = policy.ell_max
    q = policy.near_radius2
    first_m2l_level = 2
    noffsets = length(tables.push_offsets)
    level_class_of = zeros(Int32, 8, noffsets, ell_max + 1)
    for L in first_m2l_level:ell_max
        @views level_class_of[:, :, L + 1] .= tables.class_of
    end
    nclasses = max(ell_max - first_m2l_level + 1, 0) * noffsets
    reach = 2 * isqrt(q) + 1
    lut = zeros(Int32, 2 * reach + 1, 2 * reach + 1, 2 * reach + 1)
    for (k, o) in enumerate(tables.push_offsets)
        lut[o[1] + reach + 1, o[2] + reach + 1, o[3] + reach + 1] = Int32(k)
    end
    d_offset_lut = CUDA.CuArray{Int32}(lut)
    d_level_class_of = CUDA.CuArray{Int32}(level_class_of)
    counters.operator_uploads += 1        # construction-only class-table upload
    grid = DeviceRadixGrid(
        x_min, h0, ell_max, 0, 0,
        CUDA.zeros(Int, maxn), CUDA.zeros(Int, maxn),
        CUDA.zeros(UInt64, leaf_cap), CUDA.zeros(Int, 2, leaf_cap),
        CUDA.zeros(Int, maxn), CUDA.zeros(Int, maxn),
        CUDA.zeros(TF, 3, leaf_cap),
        CUDA.zeros(Int, node_cap), CUDA.zeros(UInt64, node_cap),
        CUDA.zeros(Int, 3, node_cap), CUDA.zeros(TF, 3, node_cap),
        CUDA.zeros(Int, node_cap), CUDA.zeros(Int, 2, node_cap),
        CUDA.zeros(Int, leaf_cap),
    )
    return DeviceAdaptiveCUDAContext(
        policy, tables, first_m2l_level, noffsets, nclasses, reach,
        node_cap, leaf_cap, u_cap, v_cap, wx_cap, frontier_cap,
        max(1, min(v_cap, 1 << 15)),
        grid,
        CUDA.zeros(Int32, node_cap), CUDA.zeros(Int32, node_cap),
        CUDA.zeros(TF, node_cap),
        CUDA.zeros(Int32, node_cap), CUDA.zeros(Int32, leaf_cap),
        zeros(Int, ell_max + 2),
        CUDA.zeros(UInt64, maxn), CUDA.zeros(UInt64, maxn),
        CUDA.zeros(Int32, leaf_cap), CUDA.zeros(UInt64, leaf_cap),
        CUDA.zeros(Int32, leaf_cap), CUDA.zeros(Int32, leaf_cap),
        CUDA.zeros(Int32, leaf_cap), CUDA.zeros(UInt64, leaf_cap),
        CUDA.zeros(Int32, leaf_cap), CUDA.zeros(Int32, leaf_cap),
        CUDA.zeros(UInt64, leaf_cap), CUDA.zeros(Int, leaf_cap),
        CUDA.zeros(Int32, leaf_cap), CUDA.zeros(UInt64, leaf_cap),
        CUDA.zeros(Int32, frontier_cap), CUDA.zeros(Int32, frontier_cap),
        CUDA.zeros(Int32, frontier_cap),
        CUDA.zeros(Int32, frontier_cap), CUDA.zeros(Int32, frontier_cap),
        CUDA.zeros(Int32, frontier_cap),
        CUDA.zeros(Int32, frontier_cap), CUDA.zeros(Int32, frontier_cap),
        CUDA.zeros(Int32, u_cap), CUDA.zeros(Int32, u_cap),
        CUDA.zeros(Int32, wx_cap), CUDA.zeros(Int32, wx_cap),
        CUDA.zeros(Int32, wx_cap), CUDA.zeros(Int32, wx_cap),
        CUDA.zeros(Int32, v_cap), CUDA.zeros(Int32, v_cap),
        CUDA.zeros(Int32, v_cap),
        CUDA.zeros(UInt64, v_cap), CUDA.zeros(Int, v_cap),
        CUDA.zeros(Int, v_cap), CUDA.zeros(Int, v_cap),
        CUDA.zeros(Int32, v_cap), CUDA.zeros(Int32, v_cap),
        CUDA.zeros(Int32, nclasses),
        _pin_host_array(zeros(Int32, nclasses)),
        zeros(Int, nclasses + 1), zeros(Int, ell_max + 2),
        d_offset_lut, d_level_class_of,
        CUDA.zeros(TF, 0, 0), CUDA.zeros(TF, 0, 0),
        CUDA.zeros(TF, 0, 0, 0),      # harmonics_scratch: sized by the lifecycle allocator
        policy.sigma_row, policy.rho_t,
        _ball_stencil_min_gap(q), false,
        CUDA.zeros(Int32, 2), _pin_host_array(zeros(Int32, 2)),
        _pin_host_array(zeros(Int32, 1)), _pin_host_array(zeros(Int, ell_max + 1)),
        CUDA.zeros(UInt64, leaf_cap), CUDA.zeros(Int32, leaf_cap),
        CUDA.zeros(Int32, 1),
        0, false, 0,
        nothing, -1, -1,
        0, 0, 0, 0, 0, 0, 0, 0,
        false, zeros(UInt64, 8), 0,
    )
end

#------- generic scan/compact helpers -------#

# Inclusive scan of `flags[1:n]` into `prefix[1:n]`, returning the total via the
# pinned 4-byte staging scalar (the established device-scan pattern).
function _adt_cuda_scan_total!(actx, n::Int)
    n == 0 && return 0
    fv = view(actx.flags::CUDA.CuVector{Int32}, 1:n)
    pv = view(actx.prefix::CUDA.CuVector{Int32}, 1:n)
    accumulate!(+, pv, fv)
    hs = actx.host_scalar32::Vector{Int32}
    copyto!(hs, 1, actx.prefix::CUDA.CuVector{Int32}, n, 1)
    return Int(hs[1])
end

#------- Phase A: K_max frontier split -------#

function _adt_cuda_seed_root_kernel!(lev, key, lo, hi, n)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > 1 && return nothing
    @inbounds begin
        lev[1] = Int32(0)
        key[1] = UInt64(0)
        lo[1] = Int32(1)
        hi[1] = Int32(n)
    end
    return nothing
end

# Child tuple of frontier cell `i`, child bits `c` (0-based): occupied range by
# binary search over the sorted full-depth body keys.
@inline function _adt_cuda_child_range(sorted_keys, alev, akey, alo, ahi, i, c,
        ell_max)
    lc = Int(alev[i]) + 1
    shift = 3 * (ell_max - lc)
    ckey = (akey[i] << 3) | UInt64(c)
    startk = ckey << shift
    endk = startk + (UInt64(1) << shift)
    lo_c = _cuda_lower_bound(sorted_keys, Int(alo[i]), Int(ahi[i]), startk)
    hi_c = _cuda_lower_bound(sorted_keys, Int(alo[i]), Int(ahi[i]), endk) - 1
    return lc, ckey, lo_c, hi_c
end

# flags over the 8 x n_active virtual child slots; want_leaf=1 flags children
# that become leaves, want_leaf=0 flags children that stay active (split again).
function _adt_cuda_split_flags_kernel!(flags, sorted_keys, alev, akey, alo, ahi,
        nact, K_max, ell_max, want_leaf)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j > 8 * nact && return nothing
    i = (j - 1) >> 3 + 1
    c = (j - 1) & 7
    @inbounds begin
        lc, _, lo_c, hi_c = _adt_cuda_child_range(sorted_keys, alev, akey, alo,
            ahi, i, c, ell_max)
        f = Int32(0)
        if lo_c <= hi_c
            isleaf = (hi_c - lo_c + 1 <= K_max) || (lc == ell_max)
            f = ((want_leaf == 1) == isleaf) ? Int32(1) : Int32(0)
        end
        flags[j] = f
    end
    return nothing
end

function _adt_cuda_split_compact_kernel!(dlev, dkey, dlo, dhi, base, flags,
        prefix, sorted_keys, alev, akey, alo, ahi, nact, ell_max)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j > 8 * nact && return nothing
    @inbounds begin
        flags[j] == Int32(1) || return nothing
        i = (j - 1) >> 3 + 1
        c = (j - 1) & 7
        lc, ckey, lo_c, hi_c = _adt_cuda_child_range(sorted_keys, alev, akey,
            alo, ahi, i, c, ell_max)
        idx = base + Int(prefix[j])
        dlev[idx] = Int32(lc)
        dkey[idx] = ckey
        dlo[idx] = Int32(lo_c)
        dhi[idx] = Int32(hi_c)
    end
    return nothing
end

# Build the K_max leaf set (theory §1.2). Ping buffers hold the active
# frontier, pong buffers collect the next frontier; leaves append to the leaf
# arrays. Returns the leaf count.
function _cuda_adaptive_build_leaves!(actx, n::Int)
    policy = actx.policy::AdaptiveTreePolicy
    ell_max = policy.ell_max
    K_max = policy.K_max
    threads = 256
    sorted_keys = actx.sorted_keys::CUDA.CuVector{UInt64}
    llev = actx.leaf_levels::CUDA.CuVector{Int32}
    lkey = actx.leaf_keys::CUDA.CuVector{UInt64}
    llo = actx.leaf_lo::CUDA.CuVector{Int32}
    lhi = actx.leaf_hi::CUDA.CuVector{Int32}
    # active frontier lives in the pong buffers (ping-ponged per round)
    a = (actx.leaf_levels2::CUDA.CuVector{Int32},
        actx.leaf_keys2::CUDA.CuVector{UInt64},
        actx.leaf_lo2::CUDA.CuVector{Int32}, actx.leaf_hi2::CUDA.CuVector{Int32})
    b = (actx.fa::CUDA.CuVector{Int32}, actx.vsort_keys::CUDA.CuVector{UInt64},
        actx.fb::CUDA.CuVector{Int32}, actx.fdem::CUDA.CuVector{Int32})
    nl = 0
    if n <= K_max || ell_max == 0
        CUDA.@cuda threads=1 blocks=1 _adt_cuda_seed_root_kernel!(llev, lkey,
            llo, lhi, n)
        return 1
    end
    CUDA.@cuda threads=1 blocks=1 _adt_cuda_seed_root_kernel!(a[1], a[2], a[3],
        a[4], n)
    nact = 1
    round = 0
    while nact > 0
        round += 1
        round <= ell_max + 1 || throw(AssertionError(
            "adaptive device K_max split failed to terminate"))
        m = 8 * nact
        m <= actx.frontier_capacity || throw(AssertionError(
            "adaptive device split frontier capacity exceeded; raise the " *
            "AdaptiveTreePolicy node capacity inputs"))
        blocks = cld(m, threads)
        # leaves
        CUDA.@cuda threads=threads blocks=blocks _adt_cuda_split_flags_kernel!(
            actx.flags::CUDA.CuVector{Int32}, sorted_keys, a[1], a[2], a[3],
            a[4], nact, K_max, ell_max, 1)
        nleaf = _adt_cuda_scan_total!(actx, m)
        nl + nleaf <= actx.leaf_capacity || throw(AssertionError(
            "adaptive device leaf capacity $(actx.leaf_capacity) exceeded"))
        nleaf > 0 && CUDA.@cuda threads=threads blocks=blocks _adt_cuda_split_compact_kernel!(
            llev, lkey, llo, lhi, nl, actx.flags::CUDA.CuVector{Int32},
            actx.prefix::CUDA.CuVector{Int32}, sorted_keys, a[1], a[2], a[3],
            a[4], nact, ell_max)
        nl += nleaf
        # next actives
        CUDA.@cuda threads=threads blocks=blocks _adt_cuda_split_flags_kernel!(
            actx.flags::CUDA.CuVector{Int32}, sorted_keys, a[1], a[2], a[3],
            a[4], nact, K_max, ell_max, 0)
        nact2 = _adt_cuda_scan_total!(actx, m)
        nact2 <= actx.leaf_capacity || throw(AssertionError(
            "adaptive device active frontier exceeded the leaf capacity"))
        nact2 > 0 && CUDA.@cuda threads=threads blocks=blocks _adt_cuda_split_compact_kernel!(
            b[1], b[2], b[3], b[4], 0, actx.flags::CUDA.CuVector{Int32},
            actx.prefix::CUDA.CuVector{Int32}, sorted_keys, a[1], a[2], a[3],
            a[4], nact, ell_max)
        a, b = b, a
        nact = nact2
    end
    return nl
end

#------- Phase B: 2:1 balance (Jacobi rounds over the leaf key set) -------#

function _adt_cuda_leaf_shifted_kernel!(shifted, lev, key, nl, ell_max)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > nl && return nothing
    @inbounds shifted[i] = key[i] << (3 * (ell_max - Int(lev[i])))
    return nothing
end

function _adt_cuda_gather_u64_kernel!(dst, src, order, nl)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > nl && return nothing
    @inbounds dst[i] = src[order[i]]
    return nothing
end

# Mark every leaf that violates 2:1 against the current leaf set: leaf B at
# level lev emits its <= 8 touching parent-level cells; a leaf A at level
# <= lev - 2 whose interval contains the emitted cell start is marked.
function _adt_cuda_balance_mark_kernel!(marks, lev, key, nl, sorted_starts,
        order, slev, ell_max)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > nl && return nothing
    # marks are pre-zeroed by the driver (a same-kernel clear would race with
    # concurrent mark writes from other threads)
    @inbounds begin
        l = Int(lev[i])
        l >= 2 || return nothing
    end
    @inbounds begin
        cx, cy, cz = _cuda_decode_morton_key(key[i], Int(lev[i]))
        l = Int(lev[i])
        Gc = 1 << (l - 1)
        qx0 = (cx - 1) >> 1
        qy0 = (cy - 1) >> 1
        qz0 = (cz - 1) >> 1
        for dz in 0:1, dy in 0:1, dx in 0:1
            qx = qx0 + dx
            qy = qy0 + dy
            qz = qz0 + dz
            (0 <= qx < Gc && 0 <= qy < Gc && 0 <= qz < Gc) || continue
            qstart = _cuda_morton_key(qx, qy, qz, l - 1) <<
                (3 * (ell_max - (l - 1)))
            # last sorted start <= qstart
            j = _cuda_upper_bound(sorted_starts, 1, nl, qstart) - 1
            j == 0 && continue
            aid = Int(order[j])
            la = Int(slev[aid])
            la <= l - 2 || continue
            astart = sorted_starts[j]
            alen = UInt64(1) << (3 * (ell_max - la))
            qstart < astart + alen || continue
            marks[aid] = Int32(1)
        end
    end
    return nothing
end

# Per-leaf emission count: unmarked leaves keep one slot; marked leaves emit
# their occupied children (theory §1.4 — the split leaf is occupied, so >= 1).
function _adt_cuda_balance_count_kernel!(cnt, marks, lev, key, lo, hi, nl,
        sorted_keys, ell_max)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > nl && return nothing
    @inbounds begin
        if marks[i] == Int32(0)
            cnt[i] = Int32(1)
        else
            m = 0
            for c in 0:7
                _, _, lo_c, hi_c = _adt_cuda_child_range(sorted_keys, lev, key,
                    lo, hi, i, c, ell_max)
                lo_c <= hi_c && (m += 1)
            end
            cnt[i] = Int32(m)
        end
    end
    return nothing
end

function _adt_cuda_balance_emit_kernel!(dlev, dkey, dlo, dhi, marks, prefix,
        lev, key, lo, hi, nl, sorted_keys, ell_max)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > nl && return nothing
    @inbounds begin
        base = i == 1 ? 0 : Int(prefix[i - 1])
        if marks[i] == Int32(0)
            dlev[base + 1] = lev[i]
            dkey[base + 1] = key[i]
            dlo[base + 1] = lo[i]
            dhi[base + 1] = hi[i]
        else
            w = 0
            for c in 0:7
                lc, ckey, lo_c, hi_c = _adt_cuda_child_range(sorted_keys, lev,
                    key, lo, hi, i, c, ell_max)
                lo_c <= hi_c || continue
                w += 1
                dlev[base + w] = Int32(lc)
                dkey[base + w] = ckey
                dlo[base + w] = Int32(lo_c)
                dhi[base + w] = Int32(hi_c)
            end
        end
    end
    return nothing
end

# Balance sweep to the fixed point. The leaf set lives in the leaf_* arrays;
# rounds ping-pong through leaf_*2. Returns (n_leaves, n_balance_splits).
function _cuda_adaptive_balance!(actx, nl::Int)
    policy = actx.policy::AdaptiveTreePolicy
    ell_max = policy.ell_max
    threads = 256
    sorted_keys = actx.sorted_keys::CUDA.CuVector{UInt64}
    total = 0
    round = 0
    src = (actx.leaf_levels::CUDA.CuVector{Int32},
        actx.leaf_keys::CUDA.CuVector{UInt64},
        actx.leaf_lo::CUDA.CuVector{Int32}, actx.leaf_hi::CUDA.CuVector{Int32})
    dst = (actx.leaf_levels2::CUDA.CuVector{Int32},
        actx.leaf_keys2::CUDA.CuVector{UInt64},
        actx.leaf_lo2::CUDA.CuVector{Int32}, actx.leaf_hi2::CUDA.CuVector{Int32})
    while true
        round += 1
        round <= 2 * ell_max + 4 || throw(AssertionError(
            "adaptive device 2:1 balance failed to reach a fixed point"))
        blocks = cld(nl, threads)
        CUDA.@cuda threads=threads blocks=blocks _adt_cuda_leaf_shifted_kernel!(
            actx.leaf_shifted::CUDA.CuVector{UInt64}, src[1], src[2], nl, ell_max)
        ov = view(actx.leaf_order::CUDA.CuVector{Int}, 1:nl)
        _cuda_sortperm_into!(ov, view(actx.leaf_shifted::CUDA.CuVector{UInt64}, 1:nl))
        CUDA.@cuda threads=threads blocks=blocks _adt_cuda_gather_u64_kernel!(
            actx.scratch_keys::CUDA.CuVector{UInt64},
            actx.leaf_shifted::CUDA.CuVector{UInt64},
            actx.leaf_order::CUDA.CuVector{Int}, nl)
        fill!(view(actx.leaf_marks::CUDA.CuVector{Int32}, 1:nl), Int32(0))
        CUDA.@cuda threads=threads blocks=blocks _adt_cuda_balance_mark_kernel!(
            actx.leaf_marks::CUDA.CuVector{Int32}, src[1], src[2], nl,
            actx.scratch_keys::CUDA.CuVector{UInt64},
            actx.leaf_order::CUDA.CuVector{Int}, src[1], ell_max)
        # marked total
        copyto!(view(actx.flags::CUDA.CuVector{Int32}, 1:nl),
            view(actx.leaf_marks::CUDA.CuVector{Int32}, 1:nl))
        nmark = _adt_cuda_scan_total!(actx, nl)
        nmark == 0 && break
        total += nmark
        CUDA.@cuda threads=threads blocks=blocks _adt_cuda_balance_count_kernel!(
            actx.flags::CUDA.CuVector{Int32},
            actx.leaf_marks::CUDA.CuVector{Int32}, src[1], src[2], src[3],
            src[4], nl, sorted_keys, ell_max)
        nl2 = _adt_cuda_scan_total!(actx, nl)
        nl2 <= actx.leaf_capacity || throw(AssertionError(
            "adaptive device leaf capacity $(actx.leaf_capacity) exceeded " *
            "during the balance sweep"))
        CUDA.@cuda threads=threads blocks=blocks _adt_cuda_balance_emit_kernel!(
            dst[1], dst[2], dst[3], dst[4],
            actx.leaf_marks::CUDA.CuVector{Int32},
            actx.prefix::CUDA.CuVector{Int32}, src[1], src[2], src[3], src[4],
            nl, sorted_keys, ell_max)
        src, dst = dst, src
        nl = nl2
    end
    if src[1] !== actx.leaf_levels
        # odd number of swaps: copy the final leaf set back to the ping arrays
        for (d, s) in ((actx.leaf_levels, src[1]), (actx.leaf_keys, src[2]),
                (actx.leaf_lo, src[3]), (actx.leaf_hi, src[4]))
            copyto!(view(d, 1:nl), view(s, 1:nl))
        end
    end
    return nl, total
end

#------- Phase C: level-major node table finalize -------#

function _adt_cuda_ancestor_flags_kernel!(flags, slev, nl, L)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > nl && return nothing
    @inbounds flags[i] = Int(slev[i]) >= L ? Int32(1) : Int32(0)
    return nothing
end

function _adt_cuda_ancestor_compact_kernel!(cand, flags, prefix, slev, skey, nl, L)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > nl && return nothing
    @inbounds begin
        flags[i] == Int32(1) || return nothing
        cand[Int(prefix[i])] = skey[i] >> (3 * (Int(slev[i]) - L))
    end
    return nothing
end

function _adt_cuda_unique_flags_kernel!(flags, cand, m)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > m && return nothing
    @inbounds flags[i] = (i == 1 || cand[i] != cand[i - 1]) ? Int32(1) : Int32(0)
    return nothing
end

function _adt_cuda_unique_compact_kernel!(node_keys, node_levels, base, flags,
        prefix, cand, m, L)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > m && return nothing
    @inbounds begin
        flags[i] == Int32(1) || return nothing
        idx = base + Int(prefix[i])
        node_keys[idx] = cand[i]
        node_levels[idx] = L
    end
    return nothing
end

function _adt_cuda_node_ranges_kernel!(node_lo, node_hi, node_keys, node_levels,
        n_nodes, sorted_keys, n, ell_max)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > n_nodes && return nothing
    @inbounds begin
        shift = 3 * (ell_max - Int(node_levels[i]))
        startk = node_keys[i] << shift
        endk = startk + (UInt64(1) << shift)
        node_lo[i] = Int32(_cuda_lower_bound(sorted_keys, 1, n, startk))
        node_hi[i] = Int32(_cuda_lower_bound(sorted_keys, 1, n, endk) - 1)
    end
    return nothing
end

function _adt_cuda_node_geometry_kernel!(node_coords, node_centers, node_keys,
        node_levels, n_nodes, x_min, h0)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > n_nodes && return nothing
    @inbounds begin
        L = Int(node_levels[i])
        cx, cy, cz = _cuda_decode_morton_key(node_keys[i], L)
        node_coords[1, i] = cx
        node_coords[2, i] = cy
        node_coords[3, i] = cz
        TF = eltype(node_centers)
        width = (2 * h0) / (1 << L)
        node_centers[1, i] = x_min[1] + width * (TF(cx) + TF(0.5))
        node_centers[2, i] = x_min[2] + width * (TF(cy) + TF(0.5))
        node_centers[3, i] = x_min[3] + width * (TF(cz) + TF(0.5))
    end
    return nothing
end

function _adt_cuda_parent_kernel!(parent_index, node_keys, base, count,
        base_prev, count_prev)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > count && return nothing
    @inbounds begin
        node = base + i
        pk = node_keys[node] >> 3
        parent_index[node] = _cuda_lower_bound(node_keys, base_prev + 1,
            base_prev + count_prev, pk)
    end
    return nothing
end

function _adt_cuda_children_kernel!(child_ranges, node_keys, base, count,
        base_next, count_next)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > count && return nothing
    @inbounds begin
        node = base + i
        k = node_keys[node] << 3
        firstc = _cuda_lower_bound(node_keys, base_next + 1,
            base_next + count_next, k)
        endc = _cuda_lower_bound(node_keys, base_next + 1,
            base_next + count_next, k + UInt64(8))
        child_ranges[1, node] = endc > firstc ? firstc : 0
        child_ranges[2, node] = endc - firstc
    end
    return nothing
end

function _adt_cuda_leaf_flags_kernel!(flags, child_ranges, n_nodes)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > n_nodes && return nothing
    @inbounds flags[i] = child_ranges[2, i] == 0 ? Int32(1) : Int32(0)
    return nothing
end

function _adt_cuda_leaf_compact_kernel!(leaf_index, leaf_slot_of, flags, prefix,
        n_nodes)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > n_nodes && return nothing
    @inbounds begin
        leaf_slot_of[i] = Int32(0)
        flags[i] == Int32(1) || return nothing
        slot = Int(prefix[i])
        leaf_index[slot] = Int32(i)
        leaf_slot_of[i] = Int32(slot)
    end
    return nothing
end

function _adt_cuda_cell_arrays_kernel!(cell_ranges, cell_centers, cell_keys,
        leaf_to_node, leaf_index, node_lo, node_hi, node_centers, node_keys,
        n_leaves)
    c = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    c > n_leaves && return nothing
    @inbounds begin
        f = Int(leaf_index[c])
        leaf_to_node[c] = f
        cell_ranges[1, c] = Int(node_lo[f])
        cell_ranges[2, c] = Int(node_hi[f]) - Int(node_lo[f]) + 1
        cell_centers[1, c] = node_centers[1, f]
        cell_centers[2, c] = node_centers[2, f]
        cell_centers[3, c] = node_centers[3, f]
        cell_keys[c] = node_keys[f]
    end
    return nothing
end

# Finalize the level-major node table from the final leaf set. Fills the grid
# node block, node_lo/hi, leaf compaction, cell presentation, and the host
# level_offsets. Returns (n_nodes, n_leaves).
function _cuda_adaptive_finalize!(actx, nl::Int, n::Int)
    policy = actx.policy::AdaptiveTreePolicy
    ell_max = policy.ell_max
    threads = 256
    grid = actx.grid::DeviceRadixGrid
    blocks_l = cld(nl, threads)
    # order leaves by full-depth-shifted key (ancestor keys are then sorted per level)
    CUDA.@cuda threads=threads blocks=blocks_l _adt_cuda_leaf_shifted_kernel!(
        actx.leaf_shifted::CUDA.CuVector{UInt64},
        actx.leaf_levels::CUDA.CuVector{Int32},
        actx.leaf_keys::CUDA.CuVector{UInt64}, nl, ell_max)
    ov = view(actx.leaf_order::CUDA.CuVector{Int}, 1:nl)
    _cuda_sortperm_into!(ov, view(actx.leaf_shifted::CUDA.CuVector{UInt64}, 1:nl))
    # ordered (level, key) into the pong leaf arrays
    CUDA.@cuda threads=threads blocks=blocks_l _adt_cuda_gather_u64_kernel!(
        actx.leaf_keys2::CUDA.CuVector{UInt64},
        actx.leaf_keys::CUDA.CuVector{UInt64},
        actx.leaf_order::CUDA.CuVector{Int}, nl)
    CUDA.@cuda threads=threads blocks=blocks_l _adt_cuda_gather_i32_kernel!(
        actx.leaf_levels2::CUDA.CuVector{Int32},
        actx.leaf_levels::CUDA.CuVector{Int32},
        actx.leaf_order::CUDA.CuVector{Int}, nl)
    slev = actx.leaf_levels2::CUDA.CuVector{Int32}
    skey = actx.leaf_keys2::CUDA.CuVector{UInt64}
    off = actx.level_offsets
    fill!(off, 0)
    n_nodes = 0
    for L in 0:ell_max
        off[L + 1] = n_nodes
        CUDA.@cuda threads=threads blocks=blocks_l _adt_cuda_ancestor_flags_kernel!(
            actx.flags::CUDA.CuVector{Int32}, slev, nl, L)
        me = _adt_cuda_scan_total!(actx, nl)
        me == 0 && continue
        CUDA.@cuda threads=threads blocks=blocks_l _adt_cuda_ancestor_compact_kernel!(
            actx.scratch_keys::CUDA.CuVector{UInt64},
            actx.flags::CUDA.CuVector{Int32},
            actx.prefix::CUDA.CuVector{Int32}, slev, skey, nl, L)
        blocks_m = cld(me, threads)
        CUDA.@cuda threads=threads blocks=blocks_m _adt_cuda_unique_flags_kernel!(
            actx.flags::CUDA.CuVector{Int32},
            actx.scratch_keys::CUDA.CuVector{UInt64}, me)
        mL = _adt_cuda_scan_total!(actx, me)
        n_nodes + mL <= actx.node_capacity || throw(AssertionError(
            "adaptive device node capacity $(actx.node_capacity) exceeded; " *
            "raise AdaptiveTreePolicy node_capacity (or beta_balance)"))
        CUDA.@cuda threads=threads blocks=blocks_m _adt_cuda_unique_compact_kernel!(
            grid.node_keys, grid.node_levels, n_nodes,
            actx.flags::CUDA.CuVector{Int32},
            actx.prefix::CUDA.CuVector{Int32},
            actx.scratch_keys::CUDA.CuVector{UInt64}, me, L)
        n_nodes += mL
    end
    off[ell_max + 2] = n_nodes
    blocks_n = cld(n_nodes, threads)
    CUDA.@cuda threads=threads blocks=blocks_n _adt_cuda_node_ranges_kernel!(
        actx.node_lo::CUDA.CuVector{Int32}, actx.node_hi::CUDA.CuVector{Int32},
        grid.node_keys, grid.node_levels, n_nodes,
        actx.sorted_keys::CUDA.CuVector{UInt64}, n, ell_max)
    CUDA.@cuda threads=threads blocks=blocks_n _adt_cuda_node_geometry_kernel!(
        grid.node_coords, grid.node_centers, grid.node_keys, grid.node_levels,
        n_nodes, grid.x_min, grid.h0)
    fill!(view(grid.parent_index, 1:n_nodes), 0)
    fill!(view(grid.child_ranges, :, 1:n_nodes), 0)
    for L in 1:ell_max
        base = off[L + 1]
        count = off[L + 2] - off[L + 1]
        count > 0 || continue
        base_prev = off[L]
        count_prev = off[L + 1] - off[L]
        CUDA.@cuda threads=threads blocks=cld(count, threads) _adt_cuda_parent_kernel!(
            grid.parent_index, grid.node_keys, base, count, base_prev, count_prev)
    end
    for L in 0:(ell_max - 1)
        base = off[L + 1]
        count = off[L + 2] - off[L + 1]
        count > 0 || continue
        base_next = off[L + 2]
        count_next = off[L + 3] - off[L + 2]
        CUDA.@cuda threads=threads blocks=cld(count, threads) _adt_cuda_children_kernel!(
            grid.child_ranges, grid.node_keys, base, count, base_next, count_next)
    end
    CUDA.@cuda threads=threads blocks=blocks_n _adt_cuda_leaf_flags_kernel!(
        actx.flags::CUDA.CuVector{Int32}, grid.child_ranges, n_nodes)
    n_leaves = _adt_cuda_scan_total!(actx, n_nodes)
    n_leaves == nl || throw(AssertionError(
        "adaptive device finalize: leaf count mismatch ($n_leaves vs $nl)"))
    CUDA.@cuda threads=threads blocks=blocks_n _adt_cuda_leaf_compact_kernel!(
        actx.d_leaf_index::CUDA.CuVector{Int32},
        actx.leaf_slot_of::CUDA.CuVector{Int32},
        actx.flags::CUDA.CuVector{Int32}, actx.prefix::CUDA.CuVector{Int32},
        n_nodes)
    CUDA.@cuda threads=threads blocks=cld(n_leaves, threads) _adt_cuda_cell_arrays_kernel!(
        grid.cell_ranges, grid.cell_centers, grid.cell_keys, grid.leaf_to_node,
        actx.d_leaf_index::CUDA.CuVector{Int32},
        actx.node_lo::CUDA.CuVector{Int32}, actx.node_hi::CUDA.CuVector{Int32},
        grid.node_centers, grid.node_keys, n_leaves)
    return n_nodes, n_leaves
end

function _adt_cuda_gather_i32_kernel!(dst, src, order, nl)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > nl && return nothing
    @inbounds dst[i] = src[order[i]]
    return nothing
end

# In-place refresh of node/cell body ranges for an unchanged leaf set (the
# epoch fast path): bodies moved within cells, topology frozen.
function _cuda_adaptive_refresh_ranges!(actx, n::Int)
    threads = 256
    grid = actx.grid::DeviceRadixGrid
    n_nodes = actx.n_nodes
    ell_max = (actx.policy::AdaptiveTreePolicy).ell_max
    CUDA.@cuda threads=threads blocks=cld(n_nodes, threads) _adt_cuda_node_ranges_kernel!(
        actx.node_lo::CUDA.CuVector{Int32}, actx.node_hi::CUDA.CuVector{Int32},
        grid.node_keys, grid.node_levels, n_nodes,
        actx.sorted_keys::CUDA.CuVector{UInt64}, n, ell_max)
    CUDA.@cuda threads=threads blocks=cld(actx.n_leaves, threads) _adt_cuda_cell_arrays_kernel!(
        grid.cell_ranges, grid.cell_centers, grid.cell_keys, grid.leaf_to_node,
        actx.d_leaf_index::CUDA.CuVector{Int32},
        actx.node_lo::CUDA.CuVector{Int32}, actx.node_hi::CUDA.CuVector{Int32},
        grid.node_centers, grid.node_keys, actx.n_leaves)
    return nothing
end

#------- per-node sigma_max sweep (theory §5.2) -------#

function _adt_cuda_leaf_sigma_kernel!(node_sigma, node_lo, node_hi,
        child_ranges, source_bodies, sigma_row, n_nodes)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > n_nodes && return nothing
    @inbounds begin
        child_ranges[2, i] == 0 || return nothing
        TF = eltype(node_sigma)
        m = zero(TF)
        for r in Int(node_lo[i]):Int(node_hi[i])
            s = source_bodies[sigma_row, r]
            s > m && (m = s)
        end
        node_sigma[i] = m
    end
    return nothing
end

function _adt_cuda_sigma_up_kernel!(node_sigma, child_ranges, base, count)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > count && return nothing
    @inbounds begin
        node = base + i
        nc = Int(child_ranges[2, node])
        nc == 0 && return nothing
        c0 = Int(child_ranges[1, node])
        TF = eltype(node_sigma)
        m = zero(TF)
        for c in c0:(c0 + nc - 1)
            s = node_sigma[c]
            s > m && (m = s)
        end
        node_sigma[node] = m
    end
    return nothing
end

function _cuda_adaptive_sigma_sweep!(actx, source_bodies)
    threads = 256
    grid = actx.grid::DeviceRadixGrid
    n_nodes = actx.n_nodes
    off = actx.level_offsets
    ell_max = (actx.policy::AdaptiveTreePolicy).ell_max
    CUDA.@cuda threads=threads blocks=cld(n_nodes, threads) _adt_cuda_leaf_sigma_kernel!(
        actx.node_sigma_max::CUDA.CuVector, actx.node_lo::CUDA.CuVector{Int32},
        actx.node_hi::CUDA.CuVector{Int32}, grid.child_ranges, source_bodies,
        actx.sigma_row, n_nodes)
    for L in (ell_max - 1):-1:0
        base = off[L + 1]
        count = off[L + 2] - off[L + 1]
        count > 0 || continue
        CUDA.@cuda threads=threads blocks=cld(count, threads) _adt_cuda_sigma_up_kernel!(
            actx.node_sigma_max::CUDA.CuVector, grid.child_ranges, base, count)
    end
    return nothing
end

#------- DTR list generation (theory §2.2/§2.7, sticky demotion §5.2) -------#

# Pair classification codes
const _ADT_KIND_U = Int32(1)
const _ADT_KIND_V = Int32(2)
const _ADT_KIND_W = Int32(3)
const _ADT_KIND_X = Int32(4)
const _ADT_KIND_EXPAND = Int32(5)

@inline function _adt_cuda_axis_clamp(ca::Int, la::Int, cb::Int, lb::Int)
    k = lb - la
    a0 = ca << k
    a1 = ((ca + 1) << k) - 1
    return cb < a0 ? a0 - cb : (cb > a1 ? cb - a1 : 0)
end

# Mirror of the host classification in build_adaptive_interaction_lists!:
# returns (kind, dem_out, la, lb). `gate` folds sigma demotion (sticky).
@inline function _adt_cuda_classify(node_levels, node_coords, child_ranges,
        node_sigma, ia::Int, ib::Int, dem::Bool, q::Int, gate::Bool,
        rho_t::Float64, delta_min2::Float64, ell_max::Int)
    la = Int(node_levels[ia])
    lb = Int(node_levels[ib])
    ax = Int(node_coords[1, ia]); ay = Int(node_coords[2, ia]); az = Int(node_coords[3, ia])
    bx = Int(node_coords[1, ib]); by = Int(node_coords[2, ib]); bz = Int(node_coords[3, ib])
    local dx::Int, dy::Int, dz::Int
    if la == lb
        dx = bx - ax; dy = by - ay; dz = bz - az
    elseif la < lb
        dx = _adt_cuda_axis_clamp(ax, la, bx, lb)
        dy = _adt_cuda_axis_clamp(ay, la, by, lb)
        dz = _adt_cuda_axis_clamp(az, la, bz, lb)
    else
        dx = _adt_cuda_axis_clamp(bx, lb, ax, la)
        dy = _adt_cuda_axis_clamp(by, lb, ay, la)
        dz = _adt_cuda_axis_clamp(bz, lb, az, la)
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
        cut = rho_t * Float64(node_sigma[ib])
        if delta_min2 * Float64(g2) < cut * cut
            near = true
            dem_out = true
        end
    end
    leaf_a = child_ranges[2, ia] == 0
    leaf_b = child_ranges[2, ib] == 0
    local kind::Int32
    if !near
        kind = la == lb ? _ADT_KIND_V : (la < lb ? _ADT_KIND_W : _ADT_KIND_X)
    elseif leaf_a && leaf_b
        kind = _ADT_KIND_U
    else
        kind = _ADT_KIND_EXPAND
    end
    return kind, dem_out, leaf_a, leaf_b
end

# Number of child pairs an EXPAND pair produces (host descent rules).
@inline function _adt_cuda_expand_count(node_levels, child_ranges, ia::Int,
        ib::Int, leaf_a::Bool, leaf_b::Bool)
    la = Int(node_levels[ia]); lb = Int(node_levels[ib])
    na = Int(child_ranges[2, ia]); nb = Int(child_ranges[2, ib])
    if la == lb
        leaf_a && return nb
        leaf_b && return na
        return na * nb
    end
    return la < lb ? nb : na
end

# j-th (1-based) child pair of an EXPAND pair, in the host's deterministic
# (ja-major, jb-minor) order.
@inline function _adt_cuda_expand_get(node_levels, child_ranges, ia::Int,
        ib::Int, leaf_a::Bool, leaf_b::Bool, j::Int)
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

function _adt_cuda_dtr_seed_kernel!(fa, fb, fdem)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > 1 && return nothing
    @inbounds begin
        fa[1] = Int32(1)
        fb[1] = Int32(1)
        fdem[1] = Int32(0)
    end
    return nothing
end

# want: 1..5 kind flags; 6 = demotion-trigger diagnostic count
function _adt_cuda_dtr_flags_kernel!(flags, fa, fb, fdem, np, node_levels,
        node_coords, child_ranges, node_sigma, q, gate, rho_t, delta_min2,
        ell_max, want)
    p = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    p > np && return nothing
    @inbounds begin
        ia = Int(fa[p]); ib = Int(fb[p]); dem = fdem[p] != Int32(0)
        kind, dem_out, _, _ = _adt_cuda_classify(node_levels, node_coords,
            child_ranges, node_sigma, ia, ib, dem, q, gate, rho_t, delta_min2,
            ell_max)
        if want == Int32(6)
            flags[p] = (dem_out && !dem) ? Int32(1) : Int32(0)
        else
            flags[p] = kind == want ? Int32(1) : Int32(0)
        end
    end
    return nothing
end

# Emit U/W/X node-id pairs at base offsets (deterministic scan order).
function _adt_cuda_dtr_emit_pairs_kernel!(dst_t, dst_s, base, flags, prefix,
        fa, fb, np)
    p = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    p > np && return nothing
    @inbounds begin
        flags[p] == Int32(1) || return nothing
        idx = base + Int(prefix[p])
        dst_t[idx] = fa[p]
        dst_s[idx] = fb[p]
    end
    return nothing
end

# Emit V pairs with the global class id; validity per the 025 phase-table
# membership (sticky-demotion invariant) via the violation flag.
function _adt_cuda_dtr_emit_v_kernel!(vt, vs, vc, base, flags, prefix, fa, fb,
        np, node_levels, node_coords, offset_lut, level_class_of, reach,
        noffsets, first_m2l_level, violation_flags)
    p = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    p > np && return nothing
    @inbounds begin
        flags[p] == Int32(1) || return nothing
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
    return nothing
end

function _adt_cuda_dtr_expand_count_kernel!(flags, fa, fb, fdem, np,
        node_levels, node_coords, child_ranges, node_sigma, q, gate, rho_t,
        delta_min2, ell_max)
    p = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    p > np && return nothing
    @inbounds begin
        ia = Int(fa[p]); ib = Int(fb[p]); dem = fdem[p] != Int32(0)
        kind, _, leaf_a, leaf_b = _adt_cuda_classify(node_levels, node_coords,
            child_ranges, node_sigma, ia, ib, dem, q, gate, rho_t, delta_min2,
            ell_max)
        flags[p] = kind == _ADT_KIND_EXPAND ?
            Int32(_adt_cuda_expand_count(node_levels, child_ranges, ia, ib,
                leaf_a, leaf_b)) : Int32(0)
    end
    return nothing
end

function _adt_cuda_dtr_expand_emit_kernel!(ga, gb, gdem, fa, fb, fdem, np,
        prefix, node_levels, node_coords, child_ranges, node_sigma, q, gate,
        rho_t, delta_min2, ell_max)
    p = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    p > np && return nothing
    @inbounds begin
        ia = Int(fa[p]); ib = Int(fb[p]); dem = fdem[p] != Int32(0)
        kind, dem_out, leaf_a, leaf_b = _adt_cuda_classify(node_levels,
            node_coords, child_ranges, node_sigma, ia, ib, dem, q, gate, rho_t,
            delta_min2, ell_max)
        kind == _ADT_KIND_EXPAND || return nothing
        base = p == 1 ? 0 : Int(prefix[p - 1])
        cnt = _adt_cuda_expand_count(node_levels, child_ranges, ia, ib, leaf_a,
            leaf_b)
        d = dem_out ? Int32(1) : Int32(0)
        for j in 1:cnt
            ja, jb = _adt_cuda_expand_get(node_levels, child_ranges, ia, ib,
                leaf_a, leaf_b, j)
            ga[base + j] = Int32(ja)
            gb[base + j] = Int32(jb)
            gdem[base + j] = d
        end
    end
    return nothing
end

# Frontier DTR sweep (theory §2.7). Returns (n_u, n_v, n_w, n_x, n_demoted).
function _cuda_adaptive_build_lists!(actx)
    policy = actx.policy::AdaptiveTreePolicy
    ell_max = policy.ell_max
    q = policy.near_radius2
    gate = actx.sigma_armed && actx.rho_t > 0
    grid = actx.grid::DeviceRadixGrid
    delta_min = 2 * Float64(grid.h0) / (1 << ell_max)
    delta_min2 = delta_min * delta_min
    threads = 256
    fill!(actx.violation_flags::CUDA.CuVector{Int32}, Int32(0))
    a = (actx.fa::CUDA.CuVector{Int32}, actx.fb::CUDA.CuVector{Int32},
        actx.fdem::CUDA.CuVector{Int32})
    b = (actx.fa2::CUDA.CuVector{Int32}, actx.fb2::CUDA.CuVector{Int32},
        actx.fdem2::CUDA.CuVector{Int32})
    CUDA.@cuda threads=1 blocks=1 _adt_cuda_dtr_seed_kernel!(a[1], a[2], a[3])
    np = 1
    n_u = 0; n_v = 0; n_w = 0; n_x = 0; n_dem = 0
    rounds = 0
    node_args = (grid.node_levels, grid.node_coords, grid.child_ranges,
        actx.node_sigma_max::CUDA.CuVector)
    while np > 0
        rounds += 1
        rounds <= 2 * ell_max + 3 || throw(AssertionError(
            "adaptive device DTR failed to terminate"))
        blocks = cld(np, threads)
        # U
        CUDA.@cuda threads=threads blocks=blocks _adt_cuda_dtr_flags_kernel!(
            actx.flags::CUDA.CuVector{Int32}, a[1], a[2], a[3], np,
            node_args..., q, gate, actx.rho_t, delta_min2, ell_max, _ADT_KIND_U)
        m = _adt_cuda_scan_total!(actx, np)
        n_u + m <= actx.u_capacity || throw(AssertionError(
            "adaptive device U list capacity $(actx.u_capacity) exceeded; " *
            "raise AdaptiveTreePolicy u_capacity"))
        m > 0 && CUDA.@cuda threads=threads blocks=blocks _adt_cuda_dtr_emit_pairs_kernel!(
            actx.u_targets::CUDA.CuVector{Int32},
            actx.u_sources::CUDA.CuVector{Int32}, n_u,
            actx.flags::CUDA.CuVector{Int32}, actx.prefix::CUDA.CuVector{Int32},
            a[1], a[2], np)
        n_u += m
        # V
        CUDA.@cuda threads=threads blocks=blocks _adt_cuda_dtr_flags_kernel!(
            actx.flags::CUDA.CuVector{Int32}, a[1], a[2], a[3], np,
            node_args..., q, gate, actx.rho_t, delta_min2, ell_max, _ADT_KIND_V)
        m = _adt_cuda_scan_total!(actx, np)
        n_v + m <= actx.v_capacity || throw(AssertionError(
            "adaptive device V route capacity $(actx.v_capacity) exceeded; " *
            "raise AdaptiveTreePolicy v_capacity"))
        m > 0 && CUDA.@cuda threads=threads blocks=blocks _adt_cuda_dtr_emit_v_kernel!(
            actx.vstage_targets::CUDA.CuVector{Int32},
            actx.vstage_sources::CUDA.CuVector{Int32},
            actx.vstage_class::CUDA.CuVector{Int32}, n_v,
            actx.flags::CUDA.CuVector{Int32}, actx.prefix::CUDA.CuVector{Int32},
            a[1], a[2], np, grid.node_levels, grid.node_coords,
            actx.d_offset_lut::CUDA.CuArray{Int32,3},
            actx.d_level_class_of::CUDA.CuArray{Int32,3}, actx.lut_reach,
            actx.noffsets, actx.first_m2l_level,
            actx.violation_flags::CUDA.CuVector{Int32})
        n_v += m
        # W
        CUDA.@cuda threads=threads blocks=blocks _adt_cuda_dtr_flags_kernel!(
            actx.flags::CUDA.CuVector{Int32}, a[1], a[2], a[3], np,
            node_args..., q, gate, actx.rho_t, delta_min2, ell_max, _ADT_KIND_W)
        m = _adt_cuda_scan_total!(actx, np)
        n_w + m <= actx.wx_capacity || throw(AssertionError(
            "adaptive device W list capacity $(actx.wx_capacity) exceeded; " *
            "raise AdaptiveTreePolicy wx_capacity"))
        m > 0 && CUDA.@cuda threads=threads blocks=blocks _adt_cuda_dtr_emit_pairs_kernel!(
            actx.w_targets::CUDA.CuVector{Int32},
            actx.w_sources::CUDA.CuVector{Int32}, n_w,
            actx.flags::CUDA.CuVector{Int32}, actx.prefix::CUDA.CuVector{Int32},
            a[1], a[2], np)
        n_w += m
        # X
        CUDA.@cuda threads=threads blocks=blocks _adt_cuda_dtr_flags_kernel!(
            actx.flags::CUDA.CuVector{Int32}, a[1], a[2], a[3], np,
            node_args..., q, gate, actx.rho_t, delta_min2, ell_max, _ADT_KIND_X)
        m = _adt_cuda_scan_total!(actx, np)
        n_x + m <= actx.wx_capacity || throw(AssertionError(
            "adaptive device X list capacity $(actx.wx_capacity) exceeded; " *
            "raise AdaptiveTreePolicy wx_capacity"))
        m > 0 && CUDA.@cuda threads=threads blocks=blocks _adt_cuda_dtr_emit_pairs_kernel!(
            actx.x_targets::CUDA.CuVector{Int32},
            actx.x_sources::CUDA.CuVector{Int32}, n_x,
            actx.flags::CUDA.CuVector{Int32}, actx.prefix::CUDA.CuVector{Int32},
            a[1], a[2], np)
        n_x += m
        # demotion diagnostic
        if gate
            CUDA.@cuda threads=threads blocks=blocks _adt_cuda_dtr_flags_kernel!(
                actx.flags::CUDA.CuVector{Int32}, a[1], a[2], a[3], np,
                node_args..., q, gate, actx.rho_t, delta_min2, ell_max, Int32(6))
            n_dem += _adt_cuda_scan_total!(actx, np)
        end
        # expand
        CUDA.@cuda threads=threads blocks=blocks _adt_cuda_dtr_expand_count_kernel!(
            actx.flags::CUDA.CuVector{Int32}, a[1], a[2], a[3], np,
            node_args..., q, gate, actx.rho_t, delta_min2, ell_max)
        np2 = _adt_cuda_scan_total!(actx, np)
        np2 <= actx.frontier_capacity || throw(AssertionError(
            "adaptive device DTR frontier capacity $(actx.frontier_capacity) " *
            "exceeded"))
        np2 > 0 && CUDA.@cuda threads=threads blocks=blocks _adt_cuda_dtr_expand_emit_kernel!(
            b[1], b[2], b[3], a[1], a[2], a[3], np,
            actx.prefix::CUDA.CuVector{Int32}, node_args..., q, gate,
            actx.rho_t, delta_min2, ell_max)
        a, b = b, a
        np = np2
    end
    return n_u, n_v, n_w, n_x, n_dem
end

#------- V CSR class partition -------#

function _adt_cuda_vsort_keys_kernel!(keys, vclass, n_v)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > n_v && return nothing
    @inbounds keys[i] = (UInt64(vclass[i]) << 32) | UInt64(i)
    return nothing
end

function _adt_cuda_csr_gather_kernel!(route_targets, route_sources, route_class,
        route_class_offset, vsort_ix, vt, vs, vc, n_v, noffsets, first_m2l_level)
    p = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    p > n_v && return nothing
    @inbounds begin
        i = Int(vsort_ix[p])
        route_targets[p] = Int(vt[i])
        route_sources[p] = Int(vs[i])
        c = Int(vc[i])
        route_class[p] = Int32(c)
        # per-offset id for the dense family (class_base = 0 convention)
        route_class_offset[p] = Int32(c - ((c - 1) ÷ noffsets) * noffsets)
    end
    return nothing
end

function _adt_cuda_class_histogram_kernel!(counts, vclass, n_v)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > n_v && return nothing
    @inbounds CUDA.@atomic counts[Int(vclass[i])] += Int32(1)
    return nothing
end

# Deterministic class partition of the V stage stream into the CSR route
# stream (stable by construction: the sort key appends the emission index).
# Fills the host class_starts/level_starts (one per-epoch pinned download).
function _cuda_adaptive_partition_v!(actx, n_v::Int)
    threads = 256
    actx.n_routes = n_v
    cc = actx.host_class_counts::Vector{Int32}
    if n_v == 0
        fill!(cc, Int32(0))
    else
        blocks = cld(n_v, threads)
        CUDA.@cuda threads=threads blocks=blocks _adt_cuda_vsort_keys_kernel!(
            actx.vsort_keys::CUDA.CuVector{UInt64},
            actx.vstage_class::CUDA.CuVector{Int32}, n_v)
        ixv = view(actx.vsort_ix::CUDA.CuVector{Int}, 1:n_v)
        _cuda_sortperm_into!(ixv, view(actx.vsort_keys::CUDA.CuVector{UInt64}, 1:n_v))
        CUDA.@cuda threads=threads blocks=blocks _adt_cuda_csr_gather_kernel!(
            actx.route_targets::CUDA.CuVector{Int},
            actx.route_sources::CUDA.CuVector{Int},
            actx.route_class::CUDA.CuVector{Int32},
            actx.route_class_offset::CUDA.CuVector{Int32},
            actx.vsort_ix::CUDA.CuVector{Int},
            actx.vstage_targets::CUDA.CuVector{Int32},
            actx.vstage_sources::CUDA.CuVector{Int32},
            actx.vstage_class::CUDA.CuVector{Int32}, n_v, actx.noffsets,
            actx.first_m2l_level)
        fill!(actx.class_counts_dev::CUDA.CuVector{Int32}, Int32(0))
        CUDA.@cuda threads=threads blocks=blocks _adt_cuda_class_histogram_kernel!(
            actx.class_counts_dev::CUDA.CuVector{Int32},
            actx.vstage_class::CUDA.CuVector{Int32}, n_v)
        copyto!(cc, actx.class_counts_dev::CUDA.CuVector{Int32})
    end
    cs = actx.class_starts
    cs[1] = 1
    @inbounds for c in 1:actx.nclasses
        cs[c + 1] = cs[c] + Int(cc[c])
    end
    ls = actx.level_starts
    ell_max = (actx.policy::AdaptiveTreePolicy).ell_max
    first = actx.first_m2l_level
    @inbounds for L in 0:(ell_max)
        ls[L + 1] = L < first ? 1 : cs[(L - first) * actx.noffsets + 1]
    end
    ls[ell_max + 2] = cs[end]
    return nothing
end

#------- U endpoints -> leaf cell slots -------#

function _adt_cuda_u_slots_kernel!(direct_targets, direct_sources, u_targets,
        u_sources, leaf_slot_of, n_u, violation_flags)
    k = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    k > n_u && return nothing
    @inbounds begin
        ts = leaf_slot_of[Int(u_targets[k])]
        ss = leaf_slot_of[Int(u_sources[k])]
        (ts == Int32(0) || ss == Int32(0)) && (violation_flags[2] = Int32(1))
        direct_targets[k] = Int(ts)
        direct_sources[k] = Int(ss)
    end
    return nothing
end

#------- occupancy epoch over the adaptive leaf set -------#

function _adt_cuda_epoch_snapshot_kernel!(snap_keys, snap_levels, shifted, lev, nl)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > nl && return nothing
    @inbounds begin
        snap_keys[i] = shifted[i]
        snap_levels[i] = lev[i]
    end
    return nothing
end

function _adt_cuda_epoch_compare_kernel!(flag, snap_keys, snap_levels, shifted,
        lev, nl)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > nl && return nothing
    @inbounds (snap_keys[i] != shifted[i] || snap_levels[i] != lev[i]) &&
        (flag[1] = Int32(1))
    return nothing
end

# Compare the (level, shifted key) leaf set against the epoch snapshot; update
# the snapshot when changed. Uses the deterministic post-A/B leaf order.
# Returns true when the occupancy epoch changed.
function _cuda_adaptive_epoch_changed!(actx, nl::Int)
    threads = 256
    blocks = cld(nl, threads)
    llev = actx.leaf_levels::CUDA.CuVector{Int32}
    lkey = actx.leaf_keys::CUDA.CuVector{UInt64}
    ell_max = (actx.policy::AdaptiveTreePolicy).ell_max
    CUDA.@cuda threads=threads blocks=blocks _adt_cuda_leaf_shifted_kernel!(
        actx.leaf_shifted::CUDA.CuVector{UInt64}, llev, lkey, nl, ell_max)
    changed = true
    if actx.epoch_have && actx.epoch_prev_leaves == nl
        fill!(actx.epoch_flag::CUDA.CuVector{Int32}, Int32(0))
        CUDA.@cuda threads=threads blocks=blocks _adt_cuda_epoch_compare_kernel!(
            actx.epoch_flag::CUDA.CuVector{Int32},
            actx.epoch_leaf_keys::CUDA.CuVector{UInt64},
            actx.epoch_leaf_levels::CUDA.CuVector{Int32},
            actx.leaf_shifted::CUDA.CuVector{UInt64}, llev, nl)
        hf = actx.host_scalar32::Vector{Int32}
        copyto!(hf, 1, actx.epoch_flag::CUDA.CuVector{Int32}, 1, 1)
        changed = hf[1] != Int32(0)
    end
    if changed
        CUDA.@cuda threads=threads blocks=blocks _adt_cuda_epoch_snapshot_kernel!(
            actx.epoch_leaf_keys::CUDA.CuVector{UInt64},
            actx.epoch_leaf_levels::CUDA.CuVector{Int32},
            actx.leaf_shifted::CUDA.CuVector{UInt64}, llev, nl)
        actx.epoch_prev_leaves = nl
        actx.epoch_have = true
        actx.epoch_id += 1
    end
    return changed
end

#------- refresh drivers -------#

# Rebuild the device adaptive tree from the current positions (full rebuild —
# the host 039 semantics; theory §7 tier-1 frozen-set refresh is a recorded
# future lever). Requires ctx.positions to be current. Returns `occ_changed`.
function _cuda_refresh_adaptive_tree!(ctx, actx::DeviceAdaptiveCUDAContext,
        cache, n::Int)
    policy = actx.policy::AdaptiveTreePolicy
    ell_max = policy.ell_max
    threads = 256
    grid = actx.grid::DeviceRadixGrid
    profile = actx.profile_stages
    t0 = profile ? (CUDA.synchronize(); time_ns()) : UInt64(0)
    # full-depth keys + fixed-box check
    fill!(ctx.oob_flag, Int32(0))
    kv = view(actx.keys::CUDA.CuVector{UInt64}, 1:n)
    CUDA.@cuda threads=threads blocks=cld(n, threads) _cuda_radix_keys_checked_kernel!(
        kv, ctx.oob_flag, ctx.positions, grid.x_min, cache.box_extent, grid.h0,
        ell_max)
    copyto!(ctx.host_oob, 1, ctx.oob_flag, 1, 1)
    ctx.host_oob[1] == Int32(0) || throw(ArgumentError(
        "bodies moved outside the fixed adaptive root cube; call recenter! " *
        "before update"))
    pv = view(grid.perm, 1:n)
    _cuda_sortperm_into!(pv, kv)
    CUDA.@cuda threads=threads blocks=cld(n, threads) _cuda_gather_sorted_keys_kernel!(
        view(actx.sorted_keys::CUDA.CuVector{UInt64}, 1:n), kv, pv)
    CUDA.@cuda threads=threads blocks=cld(n, threads) _cuda_fill_invperm_kernel!(
        grid.invperm, pv)
    profile && (CUDA.synchronize(); actx.stage_ns[1] = time_ns() - t0; t0 = time_ns())
    nl = _cuda_adaptive_build_leaves!(actx, n)
    profile && (CUDA.synchronize(); actx.stage_ns[2] = time_ns() - t0; t0 = time_ns())
    if policy.balance
        nl, nsplit = _cuda_adaptive_balance!(actx, nl)
        actx.n_balance_splits = nsplit
    else
        actx.n_balance_splits = 0
    end
    profile && (CUDA.synchronize(); actx.stage_ns[3] = time_ns() - t0; t0 = time_ns())
    occ_changed = _cuda_adaptive_epoch_changed!(actx, nl)
    if occ_changed
        n_nodes, n_leaves = _cuda_adaptive_finalize!(actx, nl, n)
        actx.n_nodes = n_nodes
        actx.n_leaves = n_leaves
    else
        _cuda_adaptive_refresh_ranges!(actx, n)
    end
    grid.n_bodies = n
    grid.n_cells = actx.n_leaves
    profile && (CUDA.synchronize(); actx.stage_ns[4] = time_ns() - t0)
    actx.step += 1
    return occ_changed
end

# Rebuild the DTR lists + CSR + U slot mapping (epoch path only; the sigma
# sweep must have run when the gate is armed). `direct_targets/direct_sources`
# are the state's device arrays.
function _cuda_refresh_adaptive_lists!(actx::DeviceAdaptiveCUDAContext,
        direct_targets, direct_sources)
    threads = 256
    profile = actx.profile_stages
    t0 = profile ? (CUDA.synchronize(); time_ns()) : UInt64(0)
    n_u, n_v, n_w, n_x, n_dem = _cuda_adaptive_build_lists!(actx)
    actx.n_u = n_u
    actx.n_w = n_w
    actx.n_x = n_x
    actx.n_demoted = n_dem
    profile && (CUDA.synchronize(); actx.stage_ns[5] = time_ns() - t0; t0 = time_ns())
    _cuda_adaptive_partition_v!(actx, n_v)
    profile && (CUDA.synchronize(); actx.stage_ns[6] = time_ns() - t0; t0 = time_ns())
    n_u <= length(direct_targets) || throw(AssertionError(
        "adaptive device direct capacity exceeded"))
    n_u > 0 && CUDA.@cuda threads=threads blocks=cld(n_u, threads) _adt_cuda_u_slots_kernel!(
        direct_targets, direct_sources, actx.u_targets::CUDA.CuVector{Int32},
        actx.u_sources::CUDA.CuVector{Int32},
        actx.leaf_slot_of::CUDA.CuVector{Int32}, n_u,
        actx.violation_flags::CUDA.CuVector{Int32})
    # invariant flags: V-class membership + U-leaf endpoints (loud, host-side)
    hf = actx.host_flag::Vector{Int32}
    copyto!(hf, actx.violation_flags::CUDA.CuVector{Int32})
    hf[1] == Int32(0) || throw(AssertionError(
        "adaptive device V pair outside the task-025 phase-table class set — " *
        "the sticky demotion invariant (theory §2.4/§5.2) is violated"))
    hf[2] == Int32(0) || throw(AssertionError(
        "adaptive device U-list endpoints must be leaves"))
    profile && (CUDA.synchronize(); actx.stage_ns[7] = time_ns() - t0)
    return nothing
end
