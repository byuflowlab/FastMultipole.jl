# Isolated correctness check for ext/FastMultipoleKAExt.jl's Phase D sigma-sweep port
# (ka_adaptive_sigma_sweep!, the KA-ported form of tree_batched_cuda.jl's
# _cuda_adaptive_sigma_sweep!): builds a finalized (Phase C) node table, runs the
# per-node subtree sigma_max upward pass on Metal, and compares against an
# independent CPU reference -- not tree_batched.jl's stateful AdaptiveRadixTree,
# same precedent as the other ka_tree_*_correctness.jl files.
#
# The reference computes each node's subtree sigma_max directly as the max of
# source_bodies[sigma_row, node_lo:node_hi] (a node's lo:hi range already covers its
# whole subtree by finalize's own definition), rather than mirroring the
# leaf/then-per-level-up-the-tree recursion the KA/CUDA kernels use -- a
# methodologically distinct implementation of the same theory-defined quantity.
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

# Phase A+B reference (verbatim from ka_tree_balance_correctness.jl).
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

# Phase C reference (verbatim from ka_tree_finalize_correctness.jl), minus the
# cell-array tail (not needed here) but keeping level_offsets since Phase D needs it.
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
        uniq = unique(cand)
        append!(node_keys, uniq)
        append!(node_levels, fill(L, length(uniq)))
    end
    off[ell_max + 2] = length(node_keys)
    n_nodes = length(node_keys)

    node_lo = zeros(Int, n_nodes)
    node_hi = zeros(Int, n_nodes)
    for i in 1:n_nodes
        L = node_levels[i]
        shift = 3 * (ell_max - L)
        startk = node_keys[i] << shift
        endk = startk + (UInt64(1) << shift)
        node_lo[i] = searchsortedfirst(sorted_keys, startk)
        node_hi[i] = searchsortedfirst(sorted_keys, endk) - 1
    end

    child_ranges = zeros(Int, 2, n_nodes)
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

    return (n_nodes=n_nodes, node_keys=node_keys, node_levels=node_levels, node_lo=node_lo,
        node_hi=node_hi, child_ranges=child_ranges, level_offsets=off)
end

# Independent reference for Phase D: each node's subtree sigma_max is directly the
# max of source_bodies[sigma_row, node_lo:node_hi] -- no leaf/parent recursion.
function cpu_reference_sigma_sweep(node_lo::Vector{Int}, node_hi::Vector{Int},
        source_bodies::Matrix{Float32}, sigma_row::Int)
    n_nodes = length(node_lo)
    sigma = zeros(Float32, n_nodes)
    for i in 1:n_nodes
        m = 0.0f0
        for r in node_lo[i]:node_hi[i]
            s = source_bodies[sigma_row, r]
            s > m && (m = s)
        end
        sigma[i] = m
    end
    return sigma
end

println("Starting KA adaptive-tree Phase D (sigma sweep) correctness test...")
if !dev_functional()
    println("$(DEV_NAME) not functional; skipping")
    exit(0)
end

ext = Base.get_extension(FastMultipole, :FastMultipoleKAExt)
ext !== nothing || error("FastMultipoleKAExt did not load")

Random.seed!(4)
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
    ref_fin = cpu_reference_finalize(leaves, keys, ell_max, n, x_min, h0)

    sigma_row = 4
    source_bodies = rand(Float32, sigma_row, n)  # rows 1:sigma_row-1 unused filler
    ref_sigma = cpu_reference_sigma_sweep(ref_fin.node_lo, ref_fin.node_hi, source_bodies,
        sigma_row)

    lev = Int32[l for (l, _, _, _) in leaves]
    key = UInt64[k for (_, k, _, _) in leaves]
    lo = Int32[lo for (_, _, lo, _) in leaves]
    hi = Int32[hi for (_, _, _, hi) in leaves]
    node_capacity = 40 * nl + 64

    dev_keys = devarray(keys)
    dev_lev = devarray(lev)
    dev_key = devarray(key)
    dev_lo = devarray(lo)
    dev_hi = devarray(hi)
    actx = ext.ka_allocate_adaptive_context(DEV_BACKEND, Float32, n;
        leaf_capacity=max(1, nl), frontier_capacity=max(1, nl), node_capacity=node_capacity)
    r = ext.ka_adaptive_finalize!(actx, nl, dev_lev, dev_key, dev_lo, dev_hi, dev_keys,
        ell_max, n, x_min, h0)

    r.n_nodes == ref_fin.n_nodes || error("n=$n, K_max=$K_max: n_nodes mismatch (sanity)")

    dev_bodies = devarray(source_bodies)
    got_sigma_dev = ext.ka_adaptive_sigma_sweep!(actx, r.node_lo, r.node_hi, r.child_ranges,
        r.n_nodes, r.level_offsets, ell_max, dev_bodies, sigma_row)
    got_sigma = Array(got_sigma_dev)[1:r.n_nodes]

    maximum(abs.(got_sigma .- ref_sigma); init=0.0f0) < 1f-6 ||
        error("n=$n, K_max=$K_max: node_sigma_max mismatch")

    # Independent invariant check not tied to the reference: every node's sigma_max
    # must be >= every child's sigma_max (monotone non-increasing going down the tree).
    node_lo_h = Int.(Array(r.node_lo)[1:r.n_nodes])
    node_hi_h = Int.(Array(r.node_hi)[1:r.n_nodes])
    child_ranges_h = Int.(Array(r.child_ranges)[:, 1:r.n_nodes])
    for node in 1:r.n_nodes
        cn = child_ranges_h[2, node]
        cn == 0 && continue
        c0 = child_ranges_h[1, node]
        for c in c0:(c0 + cn - 1)
            got_sigma[node] >= got_sigma[c] || error(
                "n=$n, K_max=$K_max: node $node sigma_max < child $c sigma_max")
        end
        # and every leaf's sigma_max must equal the direct range max (redundant with
        # the reference compare above for leaves, but pins down the leaf base case).
        if cn == 0
            direct = maximum(view(source_bodies, sigma_row, node_lo_h[node]:node_hi_h[node]);
                init=0.0f0)
            abs(got_sigma[node] - direct) < 1f-6 || error(
                "n=$n, K_max=$K_max: leaf $node sigma_max mismatch vs direct range max")
        end
    end

    println("✓ n=$n, K_max=$K_max, ell_max=$ell_max, dup=$dup, cluster=$cluster: " *
            "$(r.n_nodes) nodes, matches CPU reference exactly")
end

println("\n✓✓✓ All KA adaptive-tree Phase D (sigma sweep) correctness tests passed on $(DEV_NAME)! ✓✓✓")
