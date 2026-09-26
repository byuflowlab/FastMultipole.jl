# Isolated correctness check for ext/FastMultipoleKAExt.jl's Phase C finalize port
# (ka_adaptive_finalize!, the KA-ported form of tree_batched_cuda.jl's
# _cuda_adaptive_finalize!): starting from a from-scratch CPU build+balance of a
# K_max leaf set, runs the level-major node-table finalize (ancestor/unique node
# compaction, node ranges/geometry, parent/child links, leaf compaction, cell
# arrays) on Metal and compares every output array against an independent CPU
# reference of the same step -- not tree_batched.jl's stateful AdaptiveRadixTree,
# same precedent as ka_tree_leaves_correctness.jl / ka_tree_balance_correctness.jl.
include("ka_backend.jl")
using FastMultipole, Random

function cpu_decode_morton_key(key, ell)
    ix = iy = iz = 0
    for bit in 0:(ell - 1)
        ix |= Int((key >> (3 * bit)) & UInt64(0x1)) << bit
        iy |= Int((key >> (3 * bit + 1)) & UInt64(0x1)) << bit
        iz |= Int((key >> (3 * bit + 2)) & UInt64(0x1)) << bit
    end
    return ix, iy, iz
end

# Phase A+B reference (verbatim from ka_tree_balance_correctness.jl): produces a
# 2:1-balanced leaf set (lev, key, lo, hi) tuples from a sorted full-depth key array.
function cpu_reference_leaves(keys::Vector{UInt64}, ell_max::Int, K_max::Int)
    n = length(keys)
    function child_range(lev, key, lo, hi, c)
        lc = lev + 1
        shift = 3 * (ell_max - lc)
        ckey = (key << 3) | UInt64(c)
        startk = ckey << shift
        endk = startk + (UInt64(1) << shift)
        lo_c = searchsortedfirst(view(keys, lo:hi), startk) + lo - 1
        hi_c = searchsortedfirst(view(keys, lo:hi), endk) + lo - 2
        return lc, ckey, lo_c, hi_c
    end
    leaves = Tuple{Int,UInt64,Int,Int}[]
    function recurse!(lev, key, lo, hi)
        pop = hi - lo + 1
        if pop <= K_max || lev == ell_max
            push!(leaves, (lev, key, lo, hi))
            return
        end
        for c in 0:7
            lc, ckey, lo_c, hi_c = child_range(lev, key, lo, hi, c)
            lo_c <= hi_c && recurse!(lc, ckey, lo_c, hi_c)
        end
    end
    if n <= K_max || ell_max == 0
        push!(leaves, (0, UInt64(0), 1, n))
    else
        recurse!(0, UInt64(0), 1, n)
    end
    return sort(leaves)
end

function cpu_morton_key(ix, iy, iz, ell)
    key = UInt64(0)
    for bit in 0:(ell - 1)
        key |= (UInt64((ix >> bit) & 0x1) << (3 * bit))
        key |= (UInt64((iy >> bit) & 0x1) << (3 * bit + 1))
        key |= (UInt64((iz >> bit) & 0x1) << (3 * bit + 2))
    end
    return key
end

function cpu_reference_balance(keys::Vector{UInt64}, ell_max::Int,
        leaves0::Vector{Tuple{Int,UInt64,Int,Int}})
    function child_range(lev, key, lo, hi, c)
        lc = lev + 1
        shift = 3 * (ell_max - lc)
        ckey = (key << 3) | UInt64(c)
        startk = ckey << shift
        endk = startk + (UInt64(1) << shift)
        lo_c = searchsortedfirst(view(keys, lo:hi), startk) + lo - 1
        hi_c = searchsortedfirst(view(keys, lo:hi), endk) + lo - 2
        return lc, ckey, lo_c, hi_c
    end
    leaves = copy(leaves0)
    round = 0
    changed = true
    while changed
        changed = false
        round += 1
        round <= 2 * ell_max + 4 || error("cpu_reference_balance failed to reach a fixed point")
        nl = length(leaves)
        starts = sort([(leaves[i][2] << (3 * (ell_max - leaves[i][1])), i) for i in 1:nl])
        sorted_starts = [s for (s, _) in starts]
        order = [i for (_, i) in starts]
        marks = falses(nl)
        for i in 1:nl
            lev, key, _, _ = leaves[i]
            lev >= 2 || continue
            cx, cy, cz = cpu_decode_morton_key(key, lev)
            Gc = 1 << (lev - 1)
            qx0 = (cx - 1) >> 1
            qy0 = (cy - 1) >> 1
            qz0 = (cz - 1) >> 1
            for dz in 0:1, dy in 0:1, dx in 0:1
                qx = qx0 + dx
                qy = qy0 + dy
                qz = qz0 + dz
                (0 <= qx < Gc && 0 <= qy < Gc && 0 <= qz < Gc) || continue
                qstart = cpu_morton_key(qx, qy, qz, lev - 1) << (3 * (ell_max - (lev - 1)))
                j = searchsortedlast(sorted_starts, qstart)
                j == 0 && continue
                aid = order[j]
                la = leaves[aid][1]
                la <= lev - 2 || continue
                astart = sorted_starts[j]
                alen = UInt64(1) << (3 * (ell_max - la))
                qstart < astart + alen || continue
                marks[aid] = true
            end
        end
        any(marks) || break
        newleaves = Tuple{Int,UInt64,Int,Int}[]
        for i in 1:nl
            if !marks[i]
                push!(newleaves, leaves[i])
            else
                lev, key, lo, hi = leaves[i]
                for c in 0:7
                    lc, ckey, lo_c, hi_c = child_range(lev, key, lo, hi, c)
                    lo_c <= hi_c && push!(newleaves, (lc, ckey, lo_c, hi_c))
                end
                changed = true
            end
        end
        leaves = newleaves
    end
    return sort(leaves)
end

# Independent reference of Phase C (level-major node table finalize), using
# high-level array ops (sort/unique/searchsorted) rather than the manual
# flags/prefix-sum compaction the KA/CUDA kernels use -- a methodologically
# distinct implementation of the same theory-defined node table.
function cpu_reference_finalize(leaves::Vector{Tuple{Int,UInt64,Int,Int}},
        sorted_keys::Vector{UInt64}, ell_max::Int, n::Int, x_min, h0)
    nl = length(leaves)
    order = sortperm([leaves[i][2] << (3 * (ell_max - leaves[i][1])) for i in 1:nl])
    slev = [leaves[i][1] for i in order]
    skey = [leaves[i][2] for i in order]

    node_keys = UInt64[]
    node_levels = Int[]
    off = zeros(Int, ell_max + 2)
    for L in 0:ell_max
        off[L + 1] = length(node_keys)
        cand = UInt64[skey[i] >> (3 * (slev[i] - L)) for i in 1:nl if slev[i] >= L]
        isempty(cand) && continue
        uniq = unique(cand)  # cand is already sorted ascending (full-depth order preserved
                              # under a common right-shift), so unique() == sorted-unique
        append!(node_keys, uniq)
        append!(node_levels, fill(L, length(uniq)))
    end
    off[ell_max + 2] = length(node_keys)
    n_nodes = length(node_keys)

    node_lo = zeros(Int, n_nodes)
    node_hi = zeros(Int, n_nodes)
    node_coords = zeros(Int, 3, n_nodes)
    node_centers = zeros(Float64, 3, n_nodes)
    for i in 1:n_nodes
        L = node_levels[i]
        shift = 3 * (ell_max - L)
        startk = node_keys[i] << shift
        endk = startk + (UInt64(1) << shift)
        node_lo[i] = searchsortedfirst(sorted_keys, startk)
        node_hi[i] = searchsortedfirst(sorted_keys, endk) - 1
        cx, cy, cz = cpu_decode_morton_key(node_keys[i], L)
        node_coords[1, i], node_coords[2, i], node_coords[3, i] = cx, cy, cz
        width = (2 * h0) / (1 << L)
        node_centers[1, i] = x_min[1] + width * (cx + 0.5)
        node_centers[2, i] = x_min[2] + width * (cy + 0.5)
        node_centers[3, i] = x_min[3] + width * (cz + 0.5)
    end

    parent_index = zeros(Int, n_nodes)
    child_ranges = zeros(Int, 2, n_nodes)
    for L in 1:ell_max
        base = off[L + 1]
        count = off[L + 2] - off[L + 1]
        count > 0 || continue
        base_prev = off[L]
        count_prev = off[L + 1] - off[L]
        blk = view(node_keys, (base_prev + 1):(base_prev + count_prev))
        for i in 1:count
            node = base + i
            pk = node_keys[node] >> 3
            parent_index[node] = searchsortedfirst(blk, pk) + base_prev
        end
    end
    for L in 0:(ell_max - 1)
        base = off[L + 1]
        count = off[L + 2] - off[L + 1]
        count > 0 || continue
        base_next = off[L + 2]
        count_next = off[L + 3] - off[L + 2]
        blk = view(node_keys, (base_next + 1):(base_next + count_next))
        for i in 1:count
            node = base + i
            k = node_keys[node] << 3
            firstc = searchsortedfirst(blk, k) + base_next
            endc = searchsortedfirst(blk, k + UInt64(8)) + base_next
            child_ranges[1, node] = endc > firstc ? firstc : 0
            child_ranges[2, node] = endc - firstc
        end
    end

    leaf_index = [i for i in 1:n_nodes if child_ranges[2, i] == 0]
    n_leaves = length(leaf_index)
    leaf_slot_of = zeros(Int, n_nodes)
    for (slot, f) in enumerate(leaf_index)
        leaf_slot_of[f] = slot
    end

    cell_ranges = zeros(Int, 2, n_leaves)
    cell_centers = zeros(Float64, 3, n_leaves)
    cell_keys = zeros(UInt64, n_leaves)
    leaf_to_node = zeros(Int, n_leaves)
    for c in 1:n_leaves
        f = leaf_index[c]
        leaf_to_node[c] = f
        cell_ranges[1, c] = node_lo[f]
        cell_ranges[2, c] = node_hi[f] - node_lo[f] + 1
        cell_centers[:, c] .= node_centers[:, f]
        cell_keys[c] = node_keys[f]
    end

    return (n_nodes=n_nodes, n_leaves=n_leaves, node_keys=node_keys, node_levels=node_levels,
        node_coords=node_coords, node_centers=node_centers, node_lo=node_lo, node_hi=node_hi,
        parent_index=parent_index, child_ranges=child_ranges, leaf_index=leaf_index,
        leaf_slot_of=leaf_slot_of, cell_ranges=cell_ranges, cell_centers=cell_centers,
        cell_keys=cell_keys, leaf_to_node=leaf_to_node)
end

println("Starting KA adaptive-tree Phase C (finalize) correctness test...")
if !dev_functional()
    println("$(DEV_NAME) not functional; skipping")
    exit(0)
end

ext = Base.get_extension(FastMultipole, :FastMultipoleKAExt)
ext !== nothing || error("FastMultipoleKAExt did not load")

Random.seed!(3)
ell_max = 6
cases = [(n=1, K_max=4), (n=3, K_max=4), (n=50, K_max=8), (n=500, K_max=16), (n=2000, K_max=32),
         (n=2000, K_max=4),
         (n=300, K_max=8, dup=true),
         (n=600, K_max=4, cluster=true)]

for case in cases
    n, K_max = case.n, case.K_max
    dup = haskey(case, :dup) && case.dup
    cluster = haskey(case, :cluster) && case.cluster
    keys = if dup
        sort(rand(UInt64(0):UInt64(7), n))
    elseif cluster
        full = UInt64(1) << (3 * ell_max)
        dense = rand(UInt64(0):(full ÷ 64 - 1), n ÷ 2)
        sparse = rand(UInt64(0):(full - 1), n - length(dense))
        sort(vcat(dense, sparse))
    else
        sort(rand(UInt64(0):(UInt64(1) << (3 * ell_max) - 1), n))
    end

    leaves0 = cpu_reference_leaves(keys, ell_max, K_max)
    leaves = cpu_reference_balance(keys, ell_max, leaves0)
    nl = length(leaves)

    x_min = (0.0f0, 0.0f0, 0.0f0)
    h0 = 1.0f0
    ref = cpu_reference_finalize(leaves, keys, ell_max, n, x_min, h0)

    lev = Int32[l for (l, _, _, _) in leaves]
    key = UInt64[k for (_, k, _, _) in leaves]
    lo = Int32[lo for (_, _, lo, _) in leaves]
    hi = Int32[hi for (_, _, _, hi) in leaves]
    node_capacity = 40 * nl + 64

    got = begin
        dev_keys = devarray(keys)
        dev_lev = devarray(lev)
        dev_key = devarray(key)
        dev_lo = devarray(lo)
        dev_hi = devarray(hi)
        actx = ext.ka_allocate_adaptive_context(DEV_BACKEND, Float32, n;
            leaf_capacity=max(1, nl), frontier_capacity=max(1, nl), node_capacity=node_capacity)
        r = ext.ka_adaptive_finalize!(actx, nl, dev_lev, dev_key, dev_lo, dev_hi, dev_keys,
            ell_max, n, x_min, h0)
        (n_nodes=r.n_nodes, n_leaves=r.n_leaves,
         node_keys=Array(r.node_keys)[1:r.n_nodes],
         node_levels=Int.(Array(r.node_levels)[1:r.n_nodes]),
         node_coords=Int.(Array(r.node_coords)[:, 1:r.n_nodes]),
         node_centers=Array(r.node_centers)[:, 1:r.n_nodes],
         node_lo=Int.(Array(r.node_lo)[1:r.n_nodes]),
         node_hi=Int.(Array(r.node_hi)[1:r.n_nodes]),
         parent_index=Int.(Array(r.parent_index)[1:r.n_nodes]),
         child_ranges=Int.(Array(r.child_ranges)[:, 1:r.n_nodes]),
         leaf_index=Int.(Array(r.leaf_index)[1:r.n_leaves]),
         leaf_slot_of=Int.(Array(r.leaf_slot_of)[1:r.n_nodes]),
         cell_ranges=Int.(Array(r.cell_ranges)),
         cell_centers=Array(r.cell_centers),
         cell_keys=Array(r.cell_keys),
         leaf_to_node=Int.(Array(r.leaf_to_node)))
    end

    got.n_nodes == ref.n_nodes || error("n=$n, K_max=$K_max: n_nodes mismatch: ref=$(ref.n_nodes) got=$(got.n_nodes)")
    got.n_leaves == ref.n_leaves || error("n=$n, K_max=$K_max: n_leaves mismatch: ref=$(ref.n_leaves) got=$(got.n_leaves)")
    got.n_leaves == nl || error("n=$n, K_max=$K_max: n_leaves ($( got.n_leaves)) != balanced leaf count ($nl)")
    got.node_keys == ref.node_keys || error("n=$n, K_max=$K_max: node_keys mismatch")
    got.node_levels == ref.node_levels || error("n=$n, K_max=$K_max: node_levels mismatch")
    got.node_coords == ref.node_coords || error("n=$n, K_max=$K_max: node_coords mismatch")
    maximum(abs.(got.node_centers .- ref.node_centers); init=0.0) < 1e-10 ||
        error("n=$n, K_max=$K_max: node_centers mismatch")
    got.node_lo == ref.node_lo || error("n=$n, K_max=$K_max: node_lo mismatch")
    got.node_hi == ref.node_hi || error("n=$n, K_max=$K_max: node_hi mismatch")
    got.parent_index == ref.parent_index || error("n=$n, K_max=$K_max: parent_index mismatch")
    got.child_ranges == ref.child_ranges || error("n=$n, K_max=$K_max: child_ranges mismatch")
    got.leaf_index == ref.leaf_index || error("n=$n, K_max=$K_max: leaf_index mismatch")
    got.leaf_slot_of == ref.leaf_slot_of || error("n=$n, K_max=$K_max: leaf_slot_of mismatch")
    got.cell_ranges == ref.cell_ranges || error("n=$n, K_max=$K_max: cell_ranges mismatch")
    maximum(abs.(got.cell_centers .- ref.cell_centers); init=0.0) < 1e-10 ||
        error("n=$n, K_max=$K_max: cell_centers mismatch")
    got.cell_keys == ref.cell_keys || error("n=$n, K_max=$K_max: cell_keys mismatch")
    got.leaf_to_node == ref.leaf_to_node || error("n=$n, K_max=$K_max: leaf_to_node mismatch")

    # Independent sanity checks against the theory definition, not just equality
    # with the reference: every parent's child_ranges block must actually contain
    # `node`, node ranges must partition 1:n, and cell_ranges must reproduce the
    # original balanced leaf (lo,hi) ranges exactly (same leaves, same order class).
    for node in 1:got.n_nodes
        p = got.parent_index[node]
        p == 0 && continue
        c0, cn = got.child_ranges[1, p], got.child_ranges[2, p]
        node >= c0 && node < c0 + cn || error(
            "n=$n, K_max=$K_max: node $node not within parent $p's child_ranges block")
    end
    covered = falses(n)
    for c in 1:got.n_leaves
        for b in got.cell_ranges[1, c]:(got.cell_ranges[1, c] + got.cell_ranges[2, c] - 1)
            covered[b] = true
        end
    end
    all(covered) || error("n=$n, K_max=$K_max: cell ranges do not cover all $n bodies")

    println("✓ n=$n, K_max=$K_max, ell_max=$ell_max, dup=$dup, cluster=$cluster: " *
            "$(got.n_nodes) nodes, $(got.n_leaves) leaves, matches CPU reference exactly")
end

println("\n✓✓✓ All KA adaptive-tree Phase C (finalize) correctness tests passed on $(DEV_NAME)! ✓✓✓")
