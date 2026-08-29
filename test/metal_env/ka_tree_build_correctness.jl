# Isolated correctness check for ext/FastMultipoleKAExt.jl's harness front end
# (ka_radix_keys!/ka_build_adaptive_tree!): not a CUDA-parity phase (no single
# `_cuda_*` counterpart), but the from-scratch driver that stitches Phases A-D
# together starting from raw body positions instead of a pre-sorted key array --
# needed so the planned 4-way tree-build benchmark can drive local-Metal/HPC-KA off
# one shared entry point. Compares against an independent CPU reference built the
# same way as ka_tree_finalize_correctness.jl / ka_tree_sigma_sweep_correctness.jl,
# but now also re-deriving the sorted key array from positions on the CPU side
# rather than taking it as a given input.
using Metal, KernelAbstractions, FastMultipole, Random

function cpu_decode_morton_key(key, ell)
    ix = iy = iz = 0
    for bit in 0:(ell - 1)
        ix |= Int((key >> (3 * bit)) & UInt64(0x1)) << bit
        iy |= Int((key >> (3 * bit + 1)) & UInt64(0x1)) << bit
        iz |= Int((key >> (3 * bit + 2)) & UInt64(0x1)) << bit
    end
    return ix, iy, iz
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

# Independent reference for the harness front end: encode each body's full-depth key
# directly from its position (same formula as `_cuda_radix_keys_checked_kernel!`/
# `ka_radix_keys_kernel!`), then sort. `positions` is `3 x n`.
function cpu_reference_keys(positions::Matrix{Float32}, x_min, h0::Float32, ell::Int)
    n = size(positions, 2)
    G = 1 << ell
    delta = (2 * h0) / G
    keys = zeros(UInt64, n)
    for i in 1:n
        ix = clamp(floor(Int, (positions[1, i] - x_min[1]) / delta), 0, G - 1)
        iy = clamp(floor(Int, (positions[2, i] - x_min[2]) / delta), 0, G - 1)
        iz = clamp(floor(Int, (positions[3, i] - x_min[3]) / delta), 0, G - 1)
        keys[i] = cpu_morton_key(ix, iy, iz, ell)
    end
    return sort(keys)
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

# Phase C reference (trimmed to the fields this test checks; verbatim logic from
# ka_tree_finalize_correctness.jl).
function cpu_reference_finalize(leaves::Vector{Tuple{Int,UInt64,Int,Int}},
        sorted_keys::Vector{UInt64}, ell_max::Int)
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

    return (n_nodes=n_nodes, node_keys=node_keys, node_levels=node_levels, node_lo=node_lo,
        node_hi=node_hi)
end

println("Starting KA adaptive-tree harness-front-end (position -> full build) correctness test...")
if !Metal.functional()
    println("Metal not functional; skipping")
    exit(0)
end

ext = Base.get_extension(FastMultipole, :FastMultipoleKAExt)
ext !== nothing || error("FastMultipoleKAExt did not load")

Random.seed!(5)
ell_max = 6
cases = [(n=1, K_max=4), (n=3, K_max=4), (n=50, K_max=8), (n=500, K_max=16),
         (n=2000, K_max=32), (n=2000, K_max=4)]

for case in cases
    n, K_max = case.n, case.K_max
    x_min = (0.0f0, 0.0f0, 0.0f0)
    h0 = 1.0f0
    positions = Float32.(2 * h0) .* rand(Float32, 3, n)  # uniform in [0, 2h0]^3

    ref_keys = cpu_reference_keys(positions, x_min, h0, ell_max)
    leaves0 = cpu_reference_leaves(ref_keys, ell_max, K_max)
    leaves = cpu_reference_balance(ref_keys, ell_max, leaves0)
    ref_fin = cpu_reference_finalize(leaves, ref_keys, ell_max)

    nl = length(leaves)
    node_capacity = 40 * nl + 64
    leaf_capacity = 4 * nl + 64
    frontier_capacity = 8 * leaf_capacity

    dev_positions = Metal.MtlArray(positions)
    actx = ext.ka_allocate_adaptive_context(Metal.MetalBackend(), Float32, n;
        leaf_capacity=leaf_capacity, frontier_capacity=frontier_capacity,
        node_capacity=node_capacity)
    got = ext.ka_build_adaptive_tree!(actx, dev_positions, ell_max, K_max, true, x_min, h0)

    got_sorted_keys = Array(got.sorted_keys)
    got_sorted_keys == ref_keys || error("n=$n, K_max=$K_max: sorted_keys mismatch")

    got.n_nodes == ref_fin.n_nodes ||
        error("n=$n, K_max=$K_max: n_nodes mismatch: ref=$(ref_fin.n_nodes) got=$(got.n_nodes)")
    got.n_leaves == nl ||
        error("n=$n, K_max=$K_max: n_leaves ($(got.n_leaves)) != balanced leaf count ($nl)")
    Array(got.node_keys)[1:got.n_nodes] == ref_fin.node_keys ||
        error("n=$n, K_max=$K_max: node_keys mismatch")
    Int.(Array(got.node_levels)[1:got.n_nodes]) == ref_fin.node_levels ||
        error("n=$n, K_max=$K_max: node_levels mismatch")
    Int.(Array(got.node_lo)[1:got.n_nodes]) == ref_fin.node_lo ||
        error("n=$n, K_max=$K_max: node_lo mismatch")
    Int.(Array(got.node_hi)[1:got.n_nodes]) == ref_fin.node_hi ||
        error("n=$n, K_max=$K_max: node_hi mismatch")

    # Independent invariant not tied to the reference: cell_ranges (leaf-indexed body
    # ranges) must partition all n bodies with no gaps or overlaps.
    cell_ranges = Int.(Array(got.cell_ranges))
    covered = falses(n)
    for c in 1:got.n_leaves
        for b in cell_ranges[1, c]:(cell_ranges[1, c] + cell_ranges[2, c] - 1)
            covered[b] = true
        end
    end
    all(covered) || error("n=$n, K_max=$K_max: cell ranges do not cover all $n bodies")

    # invperm must invert perm exactly: invperm[perm[i]] == i for every sorted slot,
    # which also proves perm is a genuine permutation of 1:n.
    hperm = Int.(Array(got.perm))
    hinv = Int.(Array(got.invperm))
    length(hinv) == n || error("n=$n, K_max=$K_max: invperm length $(length(hinv)) != $n")
    all(hinv[hperm[i]] == i for i in 1:n) ||
        error("n=$n, K_max=$K_max: invperm is not the inverse of perm")

    println("✓ n=$n, K_max=$K_max, ell_max=$ell_max: $(got.n_nodes) nodes, " *
            "$(got.n_leaves) leaves, matches CPU reference exactly")
end

println("\n✓✓✓ All KA adaptive-tree harness-front-end correctness tests passed on Metal! ✓✓✓")
