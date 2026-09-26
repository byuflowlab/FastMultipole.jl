# Isolated correctness check for ext/FastMultipoleKAExt.jl's Phase A adaptive
# octree port (ka_adaptive_build_leaves!, the KA-ported form of
# tree_batched_cuda.jl's _cuda_adaptive_build_leaves!): builds the K_max leaf
# set of a full-depth Morton-sorted body-key array on Metal and compares the
# resulting leaf set against a from-scratch recursive CPU reference of the same
# K_max split criterion (pigeonhole on 3 key bits per level; a range becomes a
# leaf once its population is <= K_max or it hits ell_max), independent of
# tree_batched.jl's stateful AdaptiveRadixTree so this checks the algorithm
# itself, not any pre-existing host implementation (same precedent as the M2M
# ka_m2m_correctness.jl building-block check).
include("ka_backend.jl")
using FastMultipole, Random

# Reference: recursive K_max split directly on a sorted key array, returning
# the leaf set as (level, key, lo, hi) tuples (lo/hi 1-based inclusive into
# `keys`). Mirrors _adt_cuda_child_range's semantics exactly.
function cpu_reference_leaves(keys::Vector{UInt64}, ell_max::Int, K_max::Int)
    leaves = Tuple{Int,UInt64,Int,Int}[]
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

println("Starting KA adaptive-tree Phase A (leaf split) correctness test...")
if !dev_functional()
    println("$(DEV_NAME) not functional; skipping")
    exit(0)
end

ext = Base.get_extension(FastMultipole, :FastMultipoleKAExt)
ext !== nothing || error("FastMultipoleKAExt did not load")

Random.seed!(1)
ell_max = 6
cases = [(n=1, K_max=4), (n=3, K_max=4), (n=50, K_max=8), (n=500, K_max=16), (n=2000, K_max=32),
         (n=2000, K_max=4),   # forces many leaves down to ell_max (depth-cap branch)
         (n=300, K_max=8, dup=true)]  # coincident bodies: leaf population can exceed K_max at ell_max
for case in cases
    n, K_max = case.n, case.K_max
    dup = haskey(case, :dup) && case.dup
    # full-depth Morton keys: random 3*ell_max-bit values, sorted (as the real
    # radix-sort pipeline would feed ka_adaptive_build_leaves!); `dup` collapses
    # the key range to force many bodies onto identical full-depth keys.
    keyrange = dup ? (UInt64(0):UInt64(7)) : (UInt64(0):(UInt64(1) << (3 * ell_max) - 1))
    keys = sort(rand(keyrange, n))

    ref = cpu_reference_leaves(keys, ell_max, K_max)
    got = begin
        dev_keys = devarray(keys)
        actx = ext.ka_allocate_adaptive_context(DEV_BACKEND, Float32, n;
            leaf_capacity=8 * n + 8, frontier_capacity=8 * n + 64, node_capacity=8 * n + 8)
        nl, llev, lkey, llo, lhi = ext.ka_adaptive_build_leaves!(actx, dev_keys, ell_max, K_max, n)
        lev_h = Array(llev)[1:nl]; key_h = Array(lkey)[1:nl]
        lo_h = Array(llo)[1:nl]; hi_h = Array(lhi)[1:nl]
        sort([(Int(lev_h[i]), key_h[i], Int(lo_h[i]), Int(hi_h[i])) for i in 1:nl])
    end

    if got != ref
        error("n=$n, K_max=$K_max: leaf set mismatch.\nref ($(length(ref)) leaves) = $ref\ngot ($(length(got)) leaves) = $got")
    end
    # sanity: leaves partition [1, n] exactly
    total = sum(hi - lo + 1 for (_, _, lo, hi) in got)
    total == n || error("n=$n, K_max=$K_max: leaf ranges do not cover all $n bodies (covered $total)")

    println("✓ n=$n, K_max=$K_max, ell_max=$ell_max, dup=$dup: $(length(got)) leaves, matches CPU reference exactly")
end

println("\n✓✓✓ All KA adaptive-tree Phase A (leaf split) correctness tests passed on $(DEV_NAME)! ✓✓✓")
