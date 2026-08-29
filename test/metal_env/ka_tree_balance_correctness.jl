# Isolated correctness check for ext/FastMultipoleKAExt.jl's Phase B 2:1-balance
# port (ka_adaptive_balance!, the KA-ported form of tree_batched_cuda.jl's
# _cuda_adaptive_balance!): starting from a Phase-A K_max leaf set, runs the
# Jacobi balance sweep (theory §1.4, Sundar-style) on Metal and compares the
# resulting leaf set (and split count) against a from-scratch CPU reference of
# the same sweep, independent of tree_batched.jl's stateful AdaptiveRadixTree
# (same precedent as ka_tree_leaves_correctness.jl).
include("ka_backend.jl")
using FastMultipole, Random

# Phase A reference (verbatim from ka_tree_leaves_correctness.jl): recursive
# K_max split directly on a sorted key array.
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

# Independent reference of the sort/scan/compact 2:1 balance sweep (theory
# §1.4): deepest-first per round, each leaf emits its <=8 touching parent-level
# neighbor cells, matched by binary search against the sorted current-leaf
# interval starts; matched coarser leaves split into their occupied children.
# Returns (final leaf set, n_balance_splits).
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
    total = 0
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
                total += 1
                changed = true
            end
        end
        leaves = newleaves
    end
    return sort(leaves), total
end

# Direct check of the 2:1-balance property itself (theory §1.4 definition):
# every pair of touching occupied leaves differs by at most one level. Cheap
# O(n^2) box-touch test, fine at the small n used here.
function leaf_box(lev, key, ell_max)
    ix, iy, iz = cpu_decode_morton_key(key, lev)
    w = 1 << (ell_max - lev)  # width in ell_max-resolution units
    return ix * w, iy * w, iz * w, w
end

function is_2to1_balanced(leaves::Vector{Tuple{Int,UInt64,Int,Int}}, ell_max::Int)
    boxes = [leaf_box(lev, key, ell_max) for (lev, key, _, _) in leaves]
    n = length(leaves)
    for i in 1:n, j in (i + 1):n
        xi, yi, zi, wi = boxes[i]
        xj, yj, zj, wj = boxes[j]
        touch = (xi <= xj + wj && xj <= xi + wi) &&
                (yi <= yj + wj && yj <= yi + wi) &&
                (zi <= zj + wj && zj <= zi + wi)
        touch || continue
        abs(leaves[i][1] - leaves[j][1]) <= 1 || return false, (leaves[i], leaves[j])
    end
    return true, nothing
end

println("Starting KA adaptive-tree Phase B (2:1 balance) correctness test...")
if !dev_functional()
    println("$(DEV_NAME) not functional; skipping")
    exit(0)
end

ext = Base.get_extension(FastMultipole, :FastMultipoleKAExt)
ext !== nothing || error("FastMultipoleKAExt did not load")

Random.seed!(2)
ell_max = 6
cases = [(n=1, K_max=4), (n=3, K_max=4), (n=50, K_max=8), (n=500, K_max=16), (n=2000, K_max=32),
         (n=2000, K_max=4),
         (n=300, K_max=8, dup=true),
         (n=600, K_max=4, cluster=true)]  # dense cluster + sparse spread: forces adjacent
                                           # leaves with a >=2-level gap, exercising real splits
for case in cases
    n, K_max = case.n, case.K_max
    dup = haskey(case, :dup) && case.dup
    cluster = haskey(case, :cluster) && case.cluster
    keys = if dup
        sort(rand(UInt64(0):UInt64(7), n))
    elseif cluster
        full = UInt64(1) << (3 * ell_max)
        dense = rand(UInt64(0):(full ÷ 64 - 1), n ÷ 2)         # one fine corner, packed
        sparse = rand(UInt64(0):(full - 1), n - length(dense)) # rest spread over the whole domain
        sort(vcat(dense, sparse))
    else
        sort(rand(UInt64(0):(UInt64(1) << (3 * ell_max) - 1), n))
    end

    leaves0 = cpu_reference_leaves(keys, ell_max, K_max)
    ref, ref_splits = cpu_reference_balance(keys, ell_max, leaves0)
    ok, violation = is_2to1_balanced(ref, ell_max)
    ok || error("n=$n, K_max=$K_max: CPU reference itself is not 2:1 balanced: $violation")

    lev0 = Int32[l for (l, _, _, _) in leaves0]
    key0 = UInt64[k for (_, k, _, _) in leaves0]
    lo0 = Int32[lo for (_, _, lo, _) in leaves0]
    hi0 = Int32[hi for (_, _, _, hi) in leaves0]
    nl0 = length(leaves0)
    leaf_capacity = 8 * n + 64

    got, got_splits = begin
        dev_keys = devarray(keys)
        pad(a) = vcat(a, zeros(eltype(a), leaf_capacity - length(a)))
        dev_lev = devarray(pad(lev0))
        dev_key = devarray(pad(key0))
        dev_lo = devarray(pad(lo0))
        dev_hi = devarray(pad(hi0))
        actx = ext.ka_allocate_adaptive_context(DEV_BACKEND, Float32, n;
            leaf_capacity=leaf_capacity, frontier_capacity=8 * n + 64, node_capacity=leaf_capacity)
        nl, splits, flev, fkey, flo, fhi = ext.ka_adaptive_balance!(actx, nl0, dev_lev, dev_key,
            dev_lo, dev_hi, dev_keys, ell_max)
        lev_h = Array(flev)[1:nl]; key_h = Array(fkey)[1:nl]
        lo_h = Array(flo)[1:nl]; hi_h = Array(fhi)[1:nl]
        sort([(Int(lev_h[i]), key_h[i], Int(lo_h[i]), Int(hi_h[i])) for i in 1:nl]), splits
    end

    if got != ref
        error("n=$n, K_max=$K_max: balanced leaf set mismatch.\nref ($(length(ref)) leaves, " *
              "$ref_splits splits) = $ref\ngot ($(length(got)) leaves, $got_splits splits) = $got")
    end
    got_splits == ref_splits || error(
        "n=$n, K_max=$K_max: split count mismatch: ref=$ref_splits got=$got_splits")
    total = sum(hi - lo + 1 for (_, _, lo, hi) in got)
    total == n || error("n=$n, K_max=$K_max: balanced leaf ranges do not cover all $n bodies (covered $total)")

    println("✓ n=$n, K_max=$K_max, ell_max=$ell_max, dup=$dup, cluster=$cluster: $(length(leaves0)) -> " *
            "$(length(got)) leaves ($got_splits splits), matches CPU reference exactly, 2:1 balanced")
end

println("\n✓✓✓ All KA adaptive-tree Phase B (2:1 balance) correctness tests passed on $(DEV_NAME)! ✓✓✓")
