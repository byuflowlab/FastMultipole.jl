# Isolated correctness check for ext/FastMultipoleKAExt.jl's interaction-list
# ports -- Phase E (ka_adaptive_build_lists!, the DTR sweep), Phase F
# (ka_adaptive_partition_v!, the V class partition into CSR routes) and Phase G
# (ka_adaptive_u_slots! + ka_adaptive_build_u_csr!, the target-major U CSR) --
# the KA forms of tree_batched_cuda.jl's _cuda_adaptive_build_lists! /
# _cuda_adaptive_partition_v! / _cuda_adaptive_build_u_csr!.
#
# Builds a finalized node table with the CPU references below, uploads it to
# Metal, runs all three phases in sequence, and checks each against an
# independent CPU reference plus reference-independent structural invariants.
#
# Two things make the reference independent rather than a transcription:
#   1. It is a plain RECURSION over pairs, not the device's level-synchronous
#      frontier BFS and not the production host version's explicit pair stack
#      (src/interaction_list_batched.jl:1182). Same pair set, different
#      traversal.
#   2. Phase E is fed the CPU reference's OWN node table, not Phase A-D's
#      output, so a tree-construction bug cannot mask or manufacture a
#      list-build bug -- the same isolation precedent as the earlier phases.
#
# The two traversals emit the same set in different orders, so every comparison
# canonicalizes (sorts) first. Element-wise equality would be meaningless.
#
# The geometry LUTs are synthesized here (every offset outside the near radius
# is V-admissible, all levels enabled) rather than taken from a real
# AdaptiveInteractionLists: that keeps the test standalone, and makes the
# violation flag a genuine signal -- with first_m2l_level=0 and a reach wide
# enough to cover any offset the sweep can produce, it must never fire.
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


# Phase D reference (verbatim from ka_tree_sigma_sweep_correctness.jl): a node's
# subtree sigma_max is directly the max over its own lo:hi body range.
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

# Leaf slots and cell ranges, matching Phase C's convention (verbatim from
# ka_tree_finalize_correctness.jl): leaves are the childless nodes in node
# order, and each leaf's cell range is its node's body range.
function cpu_reference_leaf_slots(child_ranges::Matrix{Int}, node_lo::Vector{Int},
        node_hi::Vector{Int})
    n_nodes = size(child_ranges, 2)
    leaf_index = [i for i in 1:n_nodes if child_ranges[2, i] == 0]
    n_leaves = length(leaf_index)
    leaf_slot_of = zeros(Int32, n_nodes)
    for (slot, f) in enumerate(leaf_index)
        leaf_slot_of[f] = Int32(slot)
    end
    cell_ranges = zeros(Int32, 2, n_leaves)
    for c in 1:n_leaves
        f = leaf_index[c]
        cell_ranges[1, c] = Int32(node_lo[f])
        cell_ranges[2, c] = Int32(node_hi[f] - node_lo[f] + 1)
    end
    return leaf_index, leaf_slot_of, cell_ranges
end

# node_coords: lattice coordinate of each node, decoded from its Morton key at
# its own level (Phase C computes this on device; the reference re-derives it).
function cpu_reference_node_coords(node_keys::Vector{UInt64}, node_levels::Vector{Int})
    n_nodes = length(node_keys)
    coords = zeros(Int32, 3, n_nodes)
    for i in 1:n_nodes
        ix, iy, iz = cpu_decode_morton_key(node_keys[i], node_levels[i])
        coords[1, i] = Int32(ix); coords[2, i] = Int32(iy); coords[3, i] = Int32(iz)
    end
    return coords
end

# ---- Independent Phase E reference: recursive DTR over node pairs ----

axis_clamp(ca, la, cb, lb) = begin
    k = lb - la
    a0 = ca << k
    a1 = ((ca + 1) << k) - 1
    cb < a0 ? a0 - cb : (cb > a1 ? cb - a1 : 0)
end

function cpu_reference_lists(node_levels::Vector{Int}, node_coords::Matrix{Int32},
        child_ranges::Matrix{Int}, node_sigma::Vector{Float32}, ell_max::Int,
        q::Int, gate::Bool, rho_t::Float32, delta_min2::Float32,
        offset_lut::Array{Int32,3}, level_class_of::Array{Int32,3},
        reach::Int, noffsets::Int, first_m2l_level::Int)
    U = Tuple{Int,Int}[]; W = Tuple{Int,Int}[]; X = Tuple{Int,Int}[]
    V = Tuple{Int,Int,Int}[]
    n_dem = Ref(0)
    violated = Ref(false)

    function visit(ia::Int, ib::Int, dem::Bool)
        la = node_levels[ia]; lb = node_levels[ib]
        ax, ay, az = Int(node_coords[1, ia]), Int(node_coords[2, ia]), Int(node_coords[3, ia])
        bx, by, bz = Int(node_coords[1, ib]), Int(node_coords[2, ib]), Int(node_coords[3, ib])
        if la == lb
            dx, dy, dz = bx - ax, by - ay, bz - az
        elseif la < lb
            dx = axis_clamp(ax, la, bx, lb); dy = axis_clamp(ay, la, by, lb); dz = axis_clamp(az, la, bz, lb)
        else
            dx = axis_clamp(bx, lb, ax, la); dy = axis_clamp(by, lb, ay, la); dz = axis_clamp(bz, lb, az, la)
        end
        near = dem || (dx * dx + dy * dy + dz * dz <= q)
        if !near && gate
            sa = 1 << (ell_max - la); sb = 1 << (ell_max - lb)
            g2 = 0
            g = max(ax * sa - (bx * sb + sb), bx * sb - (ax * sa + sa), 0); g2 += g * g
            g = max(ay * sa - (by * sb + sb), by * sb - (ay * sa + sa), 0); g2 += g * g
            g = max(az * sa - (bz * sb + sb), bz * sb - (az * sa + sa), 0); g2 += g * g
            cut = rho_t * node_sigma[ib]
            if delta_min2 * Float32(g2) < cut * cut
                near = true; dem = true; n_dem[] += 1
            end
        end
        leaf_a = child_ranges[2, ia] == 0
        leaf_b = child_ranges[2, ib] == 0
        if !near
            if la == lb
                ox = ax - bx; oy = ay - by; oz = az - bz
                k = (abs(ox) <= reach && abs(oy) <= reach && abs(oz) <= reach) ?
                    Int(offset_lut[ox + reach + 1, oy + reach + 1, oz + reach + 1]) : 0
                phase = 1 + (bx & 1) + 2 * (by & 1) + 4 * (bz & 1)
                (k != 0 && la >= first_m2l_level &&
                    level_class_of[phase, k, la + 1] != 0) || (violated[] = true)
                push!(V, (ia, ib, (la - first_m2l_level) * noffsets + k))
            elseif la < lb
                push!(W, (ia, ib))
            else
                push!(X, (ia, ib))
            end
            return
        end
        if leaf_a && leaf_b
            push!(U, (ia, ib))
            return
        end
        # descend, mirroring the host's deterministic child ordering
        if la == lb
            if leaf_a
                c0 = child_ranges[1, ib]
                for jb in c0:(c0 + child_ranges[2, ib] - 1); visit(ia, jb, dem); end
            elseif leaf_b
                c0 = child_ranges[1, ia]
                for ja in c0:(c0 + child_ranges[2, ia] - 1); visit(ja, ib, dem); end
            else
                a0 = child_ranges[1, ia]; na = child_ranges[2, ia]
                b0 = child_ranges[1, ib]; nb = child_ranges[2, ib]
                for ja in a0:(a0 + na - 1), jb in b0:(b0 + nb - 1); visit(ja, jb, dem); end
            end
        elseif la < lb
            c0 = child_ranges[1, ib]
            for jb in c0:(c0 + child_ranges[2, ib] - 1); visit(ia, jb, dem); end
        else
            c0 = child_ranges[1, ia]
            for ja in c0:(c0 + child_ranges[2, ia] - 1); visit(ja, ib, dem); end
        end
        return
    end

    visit(1, 1, false)
    return (U=sort(U), V=sort(V), W=sort(W), X=sort(X), n_dem=n_dem[], violated=violated[])
end

# Synthetic geometry LUTs: every equal-level offset strictly outside the near
# radius gets a distinct nonzero class id; all phases/levels enabled.
function build_luts(reach::Int, q::Int, ell_max::Int)
    side = 2 * reach + 1
    offset_lut = zeros(Int32, side, side, side)
    k = 0
    for oz in -reach:reach, oy in -reach:reach, ox in -reach:reach
        if ox * ox + oy * oy + oz * oz > q
            k += 1
            offset_lut[ox + reach + 1, oy + reach + 1, oz + reach + 1] = Int32(k)
        end
    end
    noffsets = k
    level_class_of = ones(Int32, 8, max(noffsets, 1), ell_max + 1)
    return offset_lut, level_class_of, noffsets
end

println("Starting KA adaptive-tree Phase E/F/G (interaction lists + CSR) correctness test...")
if !Metal.functional()
    println("Metal not functional; skipping")
    exit(0)
end

ext = Base.get_extension(FastMultipole, :FastMultipoleKAExt)
ext !== nothing || error("FastMultipoleKAExt did not load")

ell_max = 4
# q = near_radius2. gate=false exercises the pure geometric DTR; gate=true adds
# the sticky sigma demotion path (theory 5.2), which is where U grows and V/W/X
# shrink, so both must be covered.
cases = [
    (n=40,   K_max=4,  q=3, gate=false),
    (n=200,  K_max=8,  q=3, gate=false),
    (n=200,  K_max=8,  q=1, gate=false),
    (n=500,  K_max=16, q=3, gate=false),
    (n=300,  K_max=8,  q=3, gate=true,  rho_t=0.5f0),
    (n=300,  K_max=8,  q=3, gate=true,  rho_t=2.0f0),
    (n=400,  K_max=4,  q=3, gate=false, cluster=true),
    (n=400,  K_max=4,  q=3, gate=true,  rho_t=1.0f0, cluster=true),
    (n=150,  K_max=4,  q=3, gate=false, dup=true),
]

# Seeded PER CASE, not once up front. Metal.jl draws one `Random.rand(UInt32)`
# from the task-local default RNG on every kernel launch, to seed that kernel's
# on-device RNG state (Metal/src/compiler/execution.jl:431). So the number of
# kernels this test launches shifts the host RNG stream, and every case after
# the first would otherwise depend on how much GPU work preceded it. It is not
# sortperm! specifically -- Metal's sort goes through a path that draws nothing;
# a broadcast draws 1, `accumulate!` draws 5. CUDA.jl does the same per-launch
# seeding but deliberately draws from a SEPARATE task-local RNG
# (CUDACore/src/compiler/execution.jl:478, `launch_rng`) precisely so kernel
# launches do not perturb the user-visible stream, so this is Metal-only.
# Per-case seeding makes each case's data fixed regardless.
npass = 0
for (ci, case) in enumerate(cases)
    Random.seed!(1000 + ci)
    n, K_max, q = case.n, case.K_max, case.q
    gate = case.gate
    rho_t = haskey(case, :rho_t) ? case.rho_t : 0.0f0
    dup = haskey(case, :dup) && case.dup
    cluster = haskey(case, :cluster) && case.cluster
    tag = "n=$n K_max=$K_max q=$q gate=$gate rho_t=$rho_t" *
        (cluster ? " cluster" : "") * (dup ? " dup" : "")

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
    x_min = (0.0f0, 0.0f0, 0.0f0); h0 = 1.0f0
    fin = cpu_reference_finalize(leaves, keys, ell_max, n, x_min, h0)
    coords = cpu_reference_node_coords(fin.node_keys, fin.node_levels)

    sigma_row = 4
    source_bodies = rand(Float32, sigma_row, n)
    node_sigma = cpu_reference_sigma_sweep(fin.node_lo, fin.node_hi, source_bodies, sigma_row)

    delta_min = 2.0f0 * h0 / Float32(1 << ell_max)
    delta_min2 = delta_min * delta_min
    reach = 1 << ell_max          # wide enough that no in-range offset is missed
    offset_lut, level_class_of, noffsets = build_luts(reach, q, ell_max)
    first_m2l_level = 0

    ref = cpu_reference_lists(fin.node_levels, coords, fin.child_ranges, node_sigma,
        ell_max, q, gate, rho_t, delta_min2, offset_lut, level_class_of, reach,
        noffsets, first_m2l_level)
    ref.violated && error("$tag: CPU reference itself reports a phase-table violation " *
                          "— the synthetic LUT is too narrow, fix the test not the port")

    backend = Metal.MetalBackend()
    # The lists context shares the tree context's frontier scratch, so the tree
    # context has to exist first and carry the frontier capacity the DTR needs.
    # Sized off n_nodes^2, not a multiple of n_nodes: under heavy sigma demotion
    # (rho_t=2) nearly every pair terminates in U, so the list approaches the
    # leaf-pair bound. A linear-in-n_nodes cap overflows there, which is a
    # property of the test data, not of the port.
    cap = max(4096, 4 * fin.n_nodes^2)
    actx = ext.ka_allocate_adaptive_context(backend, Float32, n;
        leaf_capacity=fin.n_nodes, frontier_capacity=cap, node_capacity=fin.n_nodes)
    lctx = ext.ka_allocate_lists_context(actx,
        Metal.MtlArray(offset_lut), Metal.MtlArray(level_class_of);
        u_capacity=cap, v_capacity=cap, wx_capacity=cap,
        lut_reach=reach, noffsets=noffsets, first_m2l_level=first_m2l_level,
        ell_max=ell_max, leaf_capacity=fin.n_nodes, maxn=n)

    dev_levels = Metal.MtlArray(Int32.(fin.node_levels))
    dev_coords = Metal.MtlArray(coords)
    dev_child = Metal.MtlArray(Int32.(fin.child_ranges))
    dev_sigma = Metal.MtlArray(node_sigma)

    n_u, n_v, n_w, n_x, n_dem = ext.ka_adaptive_build_lists!(lctx, dev_levels, dev_coords,
        dev_child, dev_sigma; ell_max, near_radius2=q, gate, rho_t, delta_min2)

    got_U = sort([(Int(a), Int(b)) for (a, b) in zip(
        Array(lctx.bufs.u_targets)[1:n_u], Array(lctx.bufs.u_sources)[1:n_u])])
    got_W = sort([(Int(a), Int(b)) for (a, b) in zip(
        Array(lctx.bufs.w_targets)[1:n_w], Array(lctx.bufs.w_sources)[1:n_w])])
    got_X = sort([(Int(a), Int(b)) for (a, b) in zip(
        Array(lctx.bufs.x_targets)[1:n_x], Array(lctx.bufs.x_sources)[1:n_x])])
    got_V = sort([(Int(a), Int(b), Int(c)) for (a, b, c) in zip(
        Array(lctx.bufs.vstage_targets)[1:n_v], Array(lctx.bufs.vstage_sources)[1:n_v],
        Array(lctx.bufs.vstage_class)[1:n_v])])

    got_U == ref.U || error("$tag: U list mismatch ($(length(got_U)) vs $(length(ref.U)))")
    got_V == ref.V || error("$tag: V list mismatch ($(length(got_V)) vs $(length(ref.V)))")
    got_W == ref.W || error("$tag: W list mismatch ($(length(got_W)) vs $(length(ref.W)))")
    got_X == ref.X || error("$tag: X list mismatch ($(length(got_X)) vs $(length(ref.X)))")
    n_dem == ref.n_dem || error("$tag: demotion count $n_dem != reference $(ref.n_dem)")

    # Invariants independent of the reference (theory 2.2/2.7):
    # every U pair is leaf-leaf; W's coarser member is the target and is a leaf;
    # X's coarser member is the source and is a leaf; V pairs are equal-level.
    cr = fin.child_ranges; lv = fin.node_levels
    for (ia, ib) in got_U
        (cr[2, ia] == 0 && cr[2, ib] == 0) || error("$tag: U pair ($ia,$ib) is not leaf-leaf")
    end
    for (ia, ib) in got_W
        (lv[ia] < lv[ib] && cr[2, ia] == 0) ||
            error("$tag: W pair ($ia,$ib) violates the coarser-member-is-a-leaf-target rule")
    end
    for (ia, ib) in got_X
        (lv[ia] > lv[ib] && cr[2, ib] == 0) ||
            error("$tag: X pair ($ia,$ib) violates the coarser-member-is-a-leaf-source rule")
    end
    for (ia, ib, _) in got_V
        lv[ia] == lv[ib] || error("$tag: V pair ($ia,$ib) is not equal-level")
    end

    # ---- gate_type genericity: the KA port's gate arithmetic is generic in
    # any AbstractFloat (CUDA's hardcodes Float64, which Metal cannot run at
    # all). Passing the default explicitly must reproduce the default run
    # exactly; a different float type must still compile, run, and produce a
    # structurally valid classification.
    if gate
        n2 = ext.ka_adaptive_build_lists!(lctx, dev_levels, dev_coords, dev_child,
            dev_sigma; ell_max, near_radius2=q, gate, rho_t, delta_min2,
            gate_type=Float32)
        n2 == (n_u, n_v, n_w, n_x, n_dem) ||
            error("$tag: explicit gate_type=Float32 differs from the default")
        n16 = ext.ka_adaptive_build_lists!(lctx, dev_levels, dev_coords, dev_child,
            dev_sigma; ell_max, near_radius2=q, gate, rho_t, delta_min2,
            gate_type=Float16)
        sum(n16[1:4]) > 0 || error("$tag: gate_type=Float16 produced no pairs at all")
        # Re-run at the default so the buffers the checks below read are the
        # ones the default gate produced.
        ext.ka_adaptive_build_lists!(lctx, dev_levels, dev_coords, dev_child,
            dev_sigma; ell_max, near_radius2=q, gate, rho_t, delta_min2)
    end

    # ---- Phase F: class partition of the V stream into CSR routes ----
    n_routes = ext.ka_adaptive_partition_v!(lctx, n_v)
    n_routes == n_v || error("$tag: partition_v returned $n_routes routes, expected $n_v")

    rt = Int.(Array(lctx.bufs.route_targets)[1:n_v])
    rs = Int.(Array(lctx.bufs.route_sources)[1:n_v])
    rc = Int.(Array(lctx.bufs.route_class)[1:n_v])
    rco = Int.(Array(lctx.bufs.route_class_offset)[1:n_v])
    cs = lctx.class_starts

    # The reference partition: a plain stable sort of the emitted V stream by
    # class, which is what the device's (class<<32 | index) sort key encodes.
    # got_V is already canonicalized, so re-derive the device's own emission
    # order from the staged buffers instead.
    stage_t = Int.(Array(lctx.bufs.vstage_targets)[1:n_v])
    stage_s = Int.(Array(lctx.bufs.vstage_sources)[1:n_v])
    stage_c = Int.(Array(lctx.bufs.vstage_class)[1:n_v])
    perm = sortperm(1:n_v; by = i -> (stage_c[i], i))   # stable by construction
    rt == stage_t[perm] || error("$tag: route_targets != class-stable-sorted stage stream")
    rs == stage_s[perm] || error("$tag: route_sources != class-stable-sorted stage stream")
    rc == stage_c[perm] || error("$tag: route_class != class-stable-sorted stage stream")

    # Reference-independent CSR invariants: classes are contiguous and
    # non-decreasing; class_starts brackets exactly the routes of that class;
    # the per-offset id is the class id folded into [1, noffsets].
    issorted(rc) || error("$tag: route_class is not non-decreasing (CSR broken)")
    for c in 1:lctx.nclasses
        lo, hi = cs[c], cs[c + 1] - 1
        for p in lo:hi
            rc[p] == c || error("$tag: route $p in class-$c block has class $(rc[p])")
        end
    end
    cs[lctx.nclasses + 1] == n_v + 1 ||
        error("$tag: class_starts ends at $(cs[end]), expected $(n_v + 1)")
    for p in 1:n_v
        expect = rc[p] - ((rc[p] - 1) ÷ noffsets) * noffsets
        rco[p] == expect || error("$tag: route_class_offset[$p]=$(rco[p]) != $expect")
    end
    # level_starts must bracket each level's classes within the CSR stream.
    ls = lctx.level_starts
    issorted(ls) || error("$tag: level_starts is not non-decreasing")
    ls[ell_max + 2] == n_v + 1 ||
        error("$tag: level_starts tail $(ls[end]) != $(n_v + 1)")
    for p in 1:n_v
        L = fin.node_levels[rt[p]]
        (ls[L + 1] <= p < ls[L + 2]) ||
            error("$tag: route $p at level $L outside level_starts bracket " *
                  "[$(ls[L + 1]), $(ls[L + 2]))")
    end

    # ---- Phase G: U endpoints -> leaf slots, then the target-major U CSR ----
    leaf_index, leaf_slot_of, cell_ranges = cpu_reference_leaf_slots(
        fin.child_ranges, fin.node_lo, fin.node_hi)
    n_leaves = length(leaf_index)
    dev_slot_of = Metal.MtlArray(leaf_slot_of)
    dev_cell_ranges = Metal.MtlArray(cell_ranges)

    ext.ka_adaptive_u_slots!(lctx, dev_slot_of, n_u)
    ext.ka_adaptive_build_u_csr!(lctx, dev_cell_ranges, n_leaves, n_u)

    # Reference CSR, from the device's own U emission order (frontier-major)
    # stably regrouped by target slot -- which is what the (target<<32 | index)
    # sort key encodes.
    emit_t = Int.(Array(lctx.bufs.u_targets)[1:n_u])
    emit_s = Int.(Array(lctx.bufs.u_sources)[1:n_u])
    tslot = [Int(leaf_slot_of[t]) for t in emit_t]
    sslot = [Int(leaf_slot_of[s]) for s in emit_s]
    uperm = sortperm(1:n_u; by = i -> (tslot[i], i))
    tsorted = tslot[uperm]

    got_ucsr_src = Int.(Array(lctx.bufs.u_csr_sources)[1:n_u])
    got_ucsr_off = Int.(Array(lctx.bufs.u_csr_offsets)[1:(n_leaves + 1)])

    got_ucsr_src == sslot[uperm] ||
        error("$tag: u_csr_sources != target-stable-regrouped U stream")
    exp_off = [min(searchsortedfirst(tsorted, s), n_u + 1) for s in 1:(n_leaves + 1)]
    got_ucsr_off == exp_off || error("$tag: u_csr_offsets mismatch")

    # Reference-independent CSR invariants: offsets non-decreasing, bracketing
    # the whole stream, and each leaf's bracket holding exactly the U sources
    # whose target is that leaf.
    issorted(got_ucsr_off) || error("$tag: u_csr_offsets is not non-decreasing")
    got_ucsr_off[n_leaves + 1] == n_u + 1 ||
        error("$tag: u_csr_offsets tail $(got_ucsr_off[end]) != $(n_u + 1)")
    n_u == 0 || got_ucsr_off[1] == 1 ||
        error("$tag: u_csr_offsets[1] = $(got_ucsr_off[1]), expected 1")
    for l in 1:n_leaves
        lo, hi = got_ucsr_off[l], got_ucsr_off[l + 1] - 1
        expect = sort([sslot[i] for i in 1:n_u if tslot[i] == l])
        sort(got_ucsr_src[lo:hi]) == expect ||
            error("$tag: leaf $l CSR block does not match its U sources")
    end

    # body -> leaf-slot map: every body in a leaf's cell range maps to that leaf.
    got_body_leaf = Int.(Array(lctx.bufs.u_csr_body_leaf)[1:n])
    for l in 1:n_leaves
        first = Int(cell_ranges[1, l]); cnt = Int(cell_ranges[2, l])
        for i in first:(first + cnt - 1)
            got_body_leaf[i] == l ||
                error("$tag: body $i maps to leaf $(got_body_leaf[i]), expected $l")
        end
    end

    println("  PASS  $tag  ->  |U|=$n_u |V|=$n_v |W|=$n_w |X|=$n_x dem=$n_dem " *
            "routes=$n_routes leaves=$n_leaves (nodes=$(fin.n_nodes))")
    global npass += 1
end

println("Phase E/F/G (interaction lists + V/U CSR): $npass/$(length(cases)) cases passed")
