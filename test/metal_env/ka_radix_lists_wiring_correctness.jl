# Correctness check for step (vi)a of the dispatch-wiring plan: the interaction
# list and route arrays reaching a `DeviceResidentRadixState` --
# `ka_refresh_adaptive_lists!` (the KA form of `_cuda_refresh_adaptive_lists!`)
# and `ka_radix_state(...; lists=...)`.
#
# Scope note: the CONTENTS of the U/V/W/X lists and the CSR partition are already
# validated exhaustively, against an independent CPU recursion, by
# ka_tree_lists_correctness.jl (Phases E/F/G). This file does not re-litigate
# that. What is new here, and untested until now, is the WIRING:
#   * the full pipeline composes -- a real ka_build_adaptive_tree! tree feeds the
#     refresh, which feeds the state -- rather than the phases being driven from a
#     synthetic node table as the Phase E/F/G suite does;
#   * the state ALIASES the lists context buffers instead of copying them
#     (the task-023 zero-recurring-allocation contract);
#   * counts.n_routes / counts.n_direct carry the logical extents;
#   * omitting `lists` leaves step (v)'s nothing/empty state untouched.
#
# One independent check is kept rather than trusting the composition blindly: a
# plain CPU recursion over the DOWNLOADED grid reproduces the U and V pair sets.
# It is the same recursion shape as the Phase E reference, but it runs on the
# real tree the KA build produced, so it closes the one gap the Phase E suite
# leaves open by construction (that suite feeds Phase E its own node table, to
# isolate list bugs from tree bugs -- here the two are deliberately connected).
#
# The two traversals emit the same set in different orders, so comparisons
# canonicalize (sort) first.
#
# The geometry LUTs are synthesized as in the Phase E/F/G suite: every equal-level
# offset outside the near radius is V-admissible, all phases/levels enabled. With
# first_m2l_level=0 and reach=2^ell_max the phase-table violation flag must never
# fire, which makes it a genuine signal rather than a formality.
include("ka_backend.jl")
using FastMultipole, Random, Test

const FM = FastMultipole

axis_clamp(ca, la, cb, lb) = begin
    k = lb - la
    a0 = ca << k
    a1 = ((ca + 1) << k) - 1
    cb < a0 ? a0 - cb : (cb > a1 ? cb - a1 : 0)
end

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
    return offset_lut, ones(Int32, 8, max(k, 1), ell_max + 1), k
end

# Independent CPU dual-tree recursion over the downloaded grid. U and V only --
# those are the two streams that reach the state.
function cpu_uv_lists(node_levels, node_coords, child_ranges, node_sigma, ell_max,
        q, gate, rho_t, delta_min2, offset_lut, level_class_of, reach, noffsets)
    U = Tuple{Int,Int}[]; V = Tuple{Int,Int,Int}[]
    violated = Ref(false)
    function visit(ia::Int, ib::Int, dem::Bool)
        la = node_levels[ia]; lb = node_levels[ib]
        ax, ay, az = node_coords[1, ia], node_coords[2, ia], node_coords[3, ia]
        bx, by, bz = node_coords[1, ib], node_coords[2, ib], node_coords[3, ib]
        if la == lb
            dx, dy, dz = bx - ax, by - ay, bz - az
        elseif la < lb
            dx = axis_clamp(ax, la, bx, lb); dy = axis_clamp(ay, la, by, lb)
            dz = axis_clamp(az, la, bz, lb)
        else
            dx = axis_clamp(bx, lb, ax, la); dy = axis_clamp(by, lb, ay, la)
            dz = axis_clamp(bz, lb, az, la)
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
                near = true; dem = true
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
                (k != 0 && level_class_of[phase, k, la + 1] != 0) || (violated[] = true)
                push!(V, (ia, ib, la * noffsets + k))
            end
            return
        end
        if leaf_a && leaf_b
            push!(U, (ia, ib))
            return
        end
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
    return sort(U), sort(V), violated[]
end

println("Starting KA step-(vi)a lists-into-resident-state wiring test...")
if !dev_functional()
    println("$(DEV_NAME) not functional; skipping")
    exit(0)
end

ext = Base.get_extension(FastMultipole, :FastMultipoleKAExt)
ext !== nothing || error("FastMultipoleKAExt did not load")

# (n, K_max, ell_max, q, P, lh, dpb, balance, gate, rho_t)
const CASES = [
    (   60,  4, 3, 3, 2, false, 5,  true, false, 0.0f0),
    (  400,  8, 4, 3, 4, false, 8,  true, false, 0.0f0),
    (  400,  8, 4, 3, 4,  true, 8,  true, false, 0.0f0),
    (  400,  8, 4, 1, 4, false, 8,  true, false, 0.0f0),
    (  800, 16, 4, 3, 4, false, 8, false, false, 0.0f0),
    (  300,  8, 4, 3, 2, false, 6,  true,  true, 0.5f0),
    (  300,  8, 4, 3, 2, false, 6,  true,  true, 2.0f0),
]

npass = 0
for (case_i, (n, K_max, ell_max, q, P, lh, dpb, balance, gate, rho_t)) in pairs(CASES)
    Random.seed!(5100 + case_i)
    positions = rand(Float32, 3, n)
    source_buffer = rand(Float32, dpb, n)
    # sigma row: last row. Scaled against the finest-lattice width (2*h0/2^ell_max
    # = 0.125 here) so that rho_t=0.5 demotes only the closest pairs and rho_t=2.0
    # demotes a broad band -- a sigma far below that width arms the gate but never
    # fires it, which would make the gate cases silently vacuous.
    sigma_row = gate ? dpb : 0
    gate && (source_buffer[dpb, :] .= 0.3f0 .* rand(Float32, n))
    x_min = (0.0f0, 0.0f0, 0.0f0)
    h0 = 1.0f0

    nl_estimate = max(2 * n ÷ K_max, 16)
    node_capacity = 100 * nl_estimate + 256
    leaf_capacity = 10 * nl_estimate + 256
    frontier_capacity = 16 * leaf_capacity

    actx = ext.ka_allocate_adaptive_context(DEV_BACKEND, Float32, n;
        leaf_capacity, frontier_capacity, node_capacity)
    dev_bodies = devarray(source_buffer)
    build = ext.ka_build_adaptive_tree!(actx, devarray(positions), ell_max, K_max,
        balance, x_min, h0; sigma_row=sigma_row,
        source_bodies=(sigma_row > 0 ? dev_bodies : nothing))

    reach = 1 << ell_max
    offset_lut, level_class_of, noffsets = build_luts(reach, q, ell_max)
    # sized off n_nodes^2: under heavy demotion nearly every pair terminates in U,
    # approaching the node-pair bound (a property of the data, not of the port)
    cap = max(4096, 4 * build.n_nodes^2)
    lctx = ext.ka_allocate_lists_context(actx, devarray(offset_lut),
        devarray(level_class_of); u_capacity=cap, v_capacity=cap, wx_capacity=cap,
        lut_reach=reach, noffsets=noffsets, first_m2l_level=0, ell_max=ell_max,
        leaf_capacity=leaf_capacity, maxn=n)

    lists = ext.ka_refresh_adaptive_lists!(lctx, actx, build;
        near_radius2=q, ell_max=ell_max, rho_t=rho_t, sigma_armed=gate)

    options = FM.CUDARadixLifecycleOptions(; precision=Float32)
    state = ext.ka_radix_state(actx, build, dev_bodies, P, Val(lh);
        options=options, lists=lists)

    n_nodes = build.n_nodes
    n_leaves = build.n_leaves

    # --- counts carry the logical extents ---
    @test state.counts.n_routes == lists.n_routes
    @test state.counts.n_direct == lists.n_direct
    @test state.counts.n_nodes == n_nodes
    @test state.counts.n_cells == n_leaves
    @test lists.n_routes == lists.n_v

    # --- aliases, not copies: the whole point of the wiring ---
    @test state.route_targets === lctx.bufs.route_targets
    @test state.route_sources === lctx.bufs.route_sources
    @test state.direct_targets === lctx.bufs.direct_targets
    @test state.direct_sources === lctx.bufs.direct_sources
    @test state.interaction_list === lctx

    # --- (level, offset) route arrays stay empty on the adaptive path ---
    @test length(state.route_levels) == 0
    @test size(state.route_offsets) == (3, 0)

    # --- independent CPU recursion over the downloaded tree ---
    h_levels = Int.(Array(actx.grid.node_levels)[1:n_nodes])
    h_coords = Int.(Array(actx.grid.node_coords)[:, 1:n_nodes])
    h_child = Int.(Array(actx.grid.child_ranges)[:, 1:n_nodes])
    h_sigma = Float32.(Array(actx.bufs.node_sigma)[1:n_nodes])
    delta_min = 2.0f0 * h0 / (1 << ell_max)
    refU, refV, violated = cpu_uv_lists(h_levels, h_coords, h_child, h_sigma,
        ell_max, q, gate, rho_t, delta_min * delta_min, offset_lut, level_class_of,
        reach, noffsets)
    violated && error("case $case_i: the CPU reference itself reports a phase-table " *
                      "violation -- the synthetic LUT is too narrow, fix the test")

    # routes: same (target, source) node pairs as the reference, order-independent
    gotV = sort([(Int(t), Int(s)) for (t, s) in zip(
        Array(state.route_targets)[1:lists.n_routes],
        Array(state.route_sources)[1:lists.n_routes])])
    @test gotV == sort([(t, s) for (t, s, _) in refV])

    # the CSR contract: routes come out grouped by class, non-decreasing
    rclass = Int.(Array(lctx.bufs.route_class)[1:lists.n_routes])
    @test issorted(rclass)
    @test sort(rclass) == sort([c for (_, _, c) in refV])

    # direct pairs: the U list mapped to leaf slots. Compare in NODE ids by
    # inverting leaf_to_node, so this checks the slot mapping too.
    leaf_to_node = Int.(Array(actx.grid.leaf_to_node)[1:n_leaves])
    dt = Int.(Array(state.direct_targets)[1:lists.n_direct])
    ds = Int.(Array(state.direct_sources)[1:lists.n_direct])
    @test all(s -> 1 <= s <= n_leaves, dt)
    @test all(s -> 1 <= s <= n_leaves, ds)
    @test sort([(leaf_to_node[t], leaf_to_node[s]) for (t, s) in zip(dt, ds)]) == refU

    # every U endpoint must be a leaf node (the device asserts this too)
    @test all(i -> h_child[2, i] == 0, unique(vcat(first.(refU), last.(refU))))

    # --- gate actually did something where it was armed ---
    gate && @test lists.n_dem > 0
    !gate && @test lists.n_dem == 0

    # --- step (v) behavior is untouched when `lists` is omitted ---
    bare = ext.ka_radix_state(actx, build, dev_bodies, P, Val(lh); options=options)
    @test bare.interaction_list === nothing
    @test bare.counts.n_routes == 0
    @test bare.counts.n_direct == 0
    @test length(bare.route_targets) == 0
    @test length(bare.direct_targets) == 0

    global npass += 1
    println("✓ n=$n K_max=$K_max ell_max=$ell_max q=$q P=$P lh=$lh balance=$balance " *
            "gate=$gate -> nodes=$n_nodes leaves=$n_leaves routes=$(lists.n_routes) " *
            "direct=$(lists.n_direct) dem=$(lists.n_dem)")
end

println("\nStep (vi)a lists wiring: $npass/$(length(CASES)) cases passed")
println("\n✓✓✓ All KA lists-into-resident-state wiring tests passed on $(DEV_NAME)! ✓✓✓")
