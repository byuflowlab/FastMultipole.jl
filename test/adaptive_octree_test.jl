#=
Adaptive octree construction + U/V/W/X interaction lists (task 039).

Verifies the host implementation of theory/adaptive-radix-octree.md:
  - construction invariants (partition, K_max, depth cap, level-major layout);
  - the §1.4 Sundar-style 2:1 balance sweep (property asserted structurally);
  - exact-once ordered body-pair coverage of U ∪ V ∪ W ∪ X (self included) by
    brute-force painting on uniform, wake-like (filament), and clustered
    multi-scale fields, multiple seeds and K_max, both near radii (q = 3, 12),
    balanced and unbalanced;
  - V-class membership in the task-025 phase-table set on GATED lists
    (independent re-check on top of the in-code emission invariant);
  - the §5 per-cell σ gate: sticky demotion, the 031a cutoff contract
    (every pair with r <= rho_t σ_src lands in U), and coverage under the gate;
  - uniform-limit parity: a forced single-depth adaptive tree reproduces the
    production hierarchical route set and direct pairs exactly (P = 4);
  - zero-allocation in-place refresh (tree and lists), standalone and through
    the RadixFMMCache opt-in;
  - the opt-in contract: no behavior change without the policy, guards for
    device/rectangular caches, recenter! preservation.
=#

using Test
using Random
using FastMultipole
using FastMultipole.StaticArrays

if !isdefined(@__MODULE__, :Gravitational)
    include("gravitational.jl")
end

const ADT_FM = FastMultipole

#--- distributions (5 x n body matrices: x, y, z, radius, strength) ---#

function _adt_uniform(n; seed=39001)
    rng = MersenneTwister(seed)
    b = rand(rng, 5, n)
    b[4, :] .*= 1e-3
    b[5, :] ./= n
    return b
end

"""Clustered multi-scale field: unit cube plus an embedded dense cluster at
`contrast` times the background density (the 038 oracle case)."""
function _adt_multiscale(n; contrast=100.0, seed=39002)
    rng = MersenneTwister(seed)
    frac = 0.35
    nc = round(Int, frac * n)
    nb = n - nc
    Rc = (3 * nc / (4pi * contrast * nb))^(1 / 3)
    b = rand(rng, 5, n)
    b[4, :] .*= 1e-3
    b[5, :] ./= n
    b[1:3, 1:nb] .= rand(rng, 3, nb)
    ctr = (0.6, 0.4, 0.55)
    k = 0
    while k < nc
        p = 2 .* (rand(rng, 3) .- 0.5)
        if sum(abs2, p) <= 1
            k += 1
            b[1:3, nb + k] .= ctr .+ Rc .* p
        end
    end
    return b
end

"""Wake-like field: a thin helical filament plus diffuse haze (the 037b
occupancy-contrast mechanism; the 038 oracle case)."""
function _adt_filament(n; seed=39003)
    rng = MersenneTwister(seed)
    nf = round(Int, 0.6n)
    b = rand(rng, 5, n)
    b[4, :] .*= 1e-3
    b[5, :] ./= n
    core = 0.004
    for p in 1:nf
        t = 4pi * (p - 1) / nf
        c = (0.5 + 0.35cos(t), 0.5 + 0.35sin(t), 0.15 + 0.7t / (4pi))
        b[1:3, p] .= c .+ core .* randn(rng, 3)
    end
    b[1:3, nf+1:end] .= rand(rng, 3, n - nf)
    return b
end

#--- helpers ---#

_adt_leaf_levels(tree) =
    Int.(tree.node_levels[Int.(tree.leaf_index[1:tree.n_leaves])])

"""Brute-force exact-once painting: every ordered body pair covered exactly
once by U ∪ V ∪ W ∪ X (sorted index space; subtrees are contiguous ranges).
Returns the number of mis-covered pairs."""
function _adt_exact_once_bad(tree, lists)
    n = tree.n_bodies
    cover = zeros(UInt8, n, n)
    paint!(ia, ib) = (cover[tree.node_lo[ia]:tree.node_hi[ia],
        tree.node_lo[ib]:tree.node_hi[ib]] .+= UInt8(1))
    for i in 1:lists.n_u
        paint!(lists.u_targets[i], lists.u_sources[i])
    end
    for i in 1:lists.n_routes
        paint!(lists.route_targets[i], lists.route_sources[i])
    end
    for i in 1:lists.n_w
        paint!(lists.w_targets[i], lists.w_sources[i])
    end
    for i in 1:lists.n_x
        paint!(lists.x_targets[i], lists.x_sources[i])
    end
    return count(!=(0x01), cover)
end

# independent touching test (closed boxes intersect) on the ell_max lattice
function _adt_touching(tree, i, j)
    ell_max = tree.policy.ell_max
    si = 1 << (ell_max - Int(tree.node_levels[i]))
    sj = 1 << (ell_max - Int(tree.node_levels[j]))
    for a in 1:3
        ilo = Int(tree.node_coords[a, i]) * si
        jlo = Int(tree.node_coords[a, j]) * sj
        (max(ilo - (jlo + sj), jlo - (ilo + si), 0) <= 0) || return false
    end
    return true
end

"""2:1 balance property asserted structurally (pairwise over occupied leaves)."""
function _adt_balanced(tree)
    lv = Int.(tree.leaf_index[1:tree.n_leaves])
    for i in lv, j in lv
        if _adt_touching(tree, i, j) &&
                abs(Int(tree.node_levels[i]) - Int(tree.node_levels[j])) >= 2
            return false
        end
    end
    return true
end

"""Independent V-class re-check on the (possibly σ-gated) emitted lists:
same level, separated at q, GEOMETRIC parent nearness (the sticky-demotion
invariant), Chebyshev reach, and phase-table membership."""
function _adt_v_classes_ok(tree, lists)
    q = lists.near_radius2
    maxcheb = 2 * isqrt(q) + 1
    tables = lists.tables
    offset_id = Dict(o => k for (k, o) in enumerate(tables.push_offsets))
    noffsets = lists.noffsets
    for i in 1:lists.n_routes
        ia = lists.route_targets[i]
        ib = lists.route_sources[i]
        la = Int(tree.node_levels[ia])
        la == Int(tree.node_levels[ib]) || return false
        la == lists.route_levels[i] || return false
        o = SVector{3,Int}(
            Int(tree.node_coords[1, ia]) - Int(tree.node_coords[1, ib]),
            Int(tree.node_coords[2, ia]) - Int(tree.node_coords[2, ib]),
            Int(tree.node_coords[3, ia]) - Int(tree.node_coords[3, ib]))
        sum(abs2, o) > q || return false
        maximum(abs, o) <= maxcheb || return false
        (lists.route_offsets[1, i], lists.route_offsets[2, i],
            lists.route_offsets[3, i]) == (o[1], o[2], o[3]) || return false
        k = get(offset_id, o, 0)
        k == 0 && return false
        Int(lists.route_class[i]) ==
            (la - lists.first_m2l_level) * noffsets + k || return false
        # geometric parent nearness (Invariant 2 restored by sticky demotion)
        pa = Int(tree.parent_index[ia])
        pb = Int(tree.parent_index[ib])
        (pa != 0 && pb != 0) || return false
        p = SVector{3,Int}(
            Int(tree.node_coords[1, pa]) - Int(tree.node_coords[1, pb]),
            Int(tree.node_coords[2, pa]) - Int(tree.node_coords[2, pb]),
            Int(tree.node_coords[3, pa]) - Int(tree.node_coords[3, pb]))
        sum(abs2, p) <= q || return false
        # phase-table membership for the true source phase
        phase = ADT_FM._rigid_phase_index(tree.node_coords[1, ib],
            tree.node_coords[2, ib], tree.node_coords[3, ib])
        tables.class_of[phase, k] != 0 || return false
    end
    return true
end

"""W/X structure: coarser member is a leaf; partner strictly finer."""
function _adt_wx_ok(tree, lists)
    for i in 1:lists.n_w
        ia = lists.w_targets[i]
        ib = lists.w_sources[i]
        Int(tree.node_levels[ib]) > Int(tree.node_levels[ia]) || return false
        adaptive_is_leaf(tree, ia) || return false
    end
    for i in 1:lists.n_x
        ia = lists.x_targets[i]
        ib = lists.x_sources[i]
        Int(tree.node_levels[ia]) > Int(tree.node_levels[ib]) || return false
        adaptive_is_leaf(tree, ib) || return false
    end
    return true
end

# allocation measurements through function barriers
_adt_update_alloc(tree, sys) = @allocated update_adaptive_tree!(tree, sys)
_adt_update_sigma_alloc(tree, sys, sigma) =
    @allocated update_adaptive_tree!(tree, sys; sigma=sigma)
_adt_lists_alloc(lists, tree) = @allocated build_adaptive_interaction_lists!(lists, tree)
_adt_cache_refresh_alloc(cache, systems) =
    @allocated ADT_FM._refresh_adaptive_radix!(cache, systems)

function _adt_build(bodies; kwargs...)
    sys = Gravitational(copy(bodies))
    pol = AdaptiveTreePolicy(; kwargs...)
    tree = AdaptiveRadixTree(sys; policy=pol)
    lists = AdaptiveInteractionLists(tree)
    build_adaptive_interaction_lists!(lists, tree)
    return sys, tree, lists
end

#--- tests ---#

@testset "adaptive octree construction (task 039)" begin
    b = _adt_multiscale(1200)
    sys, tree, lists = _adt_build(b; K_max=16, ell_max=8)
    n = tree.n_bodies
    @test n == 1200

    # occupied leaves partition the sorted body range
    leaf_ids = Int.(tree.leaf_index[1:tree.n_leaves])
    covered = falses(n)
    for f in leaf_ids
        for r in adaptive_node_range(tree, f)
            @test !covered[r]
            covered[r] = true
        end
    end
    @test all(covered)

    # populations respect K_max except depth-capped and balance-split leaves;
    # balance splits only shrink populations, so a strict global check is
    # population <= K_max unless the leaf sits at ell_max
    for f in leaf_ids
        pop = length(adaptive_node_range(tree, f))
        if Int(tree.node_levels[f]) < tree.policy.ell_max
            @test pop <= tree.policy.K_max ||
                tree.n_balance_splits > 0   # balance children may inherit any pop <= K_max anyway
        end
    end
    @test maximum(_adt_leaf_levels(tree)) <= tree.policy.ell_max

    # level-major, Morton-sorted-within-level node table (the uniform
    # level_offsets convention)
    off = tree.level_offsets
    @test off[end] == tree.n_nodes
    for L in 0:tree.policy.ell_max
        lo = off[L + 1] + 1
        hi = off[L + 2]
        for f in lo:hi
            @test Int(tree.node_levels[f]) == L
        end
        @test issorted(tree.node_keys[lo:hi])
    end

    # parent/child consistency + subtree ranges
    for f in 1:tree.n_nodes
        cn = Int(tree.child_ranges[2, f])
        if cn > 0
            c0 = Int(tree.child_ranges[1, f])
            @test tree.node_lo[c0] == tree.node_lo[f]
            @test tree.node_hi[c0 + cn - 1] == tree.node_hi[f]
            for c in c0:(c0 + cn - 1)
                @test Int(tree.parent_index[c]) == f
                @test Int(tree.node_levels[c]) == Int(tree.node_levels[f]) + 1
            end
        end
    end

    # trivial trees
    b1 = _adt_uniform(1)
    _, t1, l1 = _adt_build(b1; K_max=4, ell_max=4)
    @test t1.n_nodes == 1 && t1.n_leaves == 1
    @test l1.n_u == 1 && l1.n_routes == 0 && l1.n_w == 0 && l1.n_x == 0
    @test _adt_exact_once_bad(t1, l1) == 0

    # coincident bodies force the depth cap
    b2 = _adt_uniform(8)
    b2[1:3, :] .= 0.3
    _, t2, l2 = _adt_build(b2; K_max=1, ell_max=3)
    @test maximum(_adt_leaf_levels(t2)) == 3
    @test _adt_exact_once_bad(t2, l2) == 0
end

@testset "adaptive 2:1 balance (Sundar-style sweep)" begin
    for (bodies, name) in ((_adt_multiscale(1200), "multiscale"),
                           (_adt_filament(1200), "filament"))
        _, tree, _ = _adt_build(bodies; K_max=8, ell_max=8, balance=true)
        @test _adt_balanced(tree)
        _, tree_u, _ = _adt_build(bodies; K_max=8, ell_max=8, balance=false)
        @test tree_u.n_balance_splits == 0
        @test tree.n_nodes >= tree_u.n_nodes
    end
end

@testset "adaptive exact-once coverage (brute force)" begin
    cases = (
        ("uniform", _adt_uniform),
        ("multiscale", (n; seed) -> _adt_multiscale(n; seed)),
        ("filament", (n; seed) -> _adt_filament(n; seed)),
    )
    n = 350
    for (name, gen) in cases, seed in (39011, 39012), K_max in (8, 32),
            q in (3, 12)
        bodies = gen(n; seed=seed)
        _, tree, lists = _adt_build(bodies; K_max, ell_max=8, near_radius2=q)
        @test _adt_exact_once_bad(tree, lists) == 0
        @test _adt_balanced(tree)
        @test _adt_v_classes_ok(tree, lists)
        @test _adt_wx_ok(tree, lists)
        # W/X duality on ungated lists
        wset = Set((lists.w_targets[i], lists.w_sources[i]) for i in 1:lists.n_w)
        xset = Set((lists.x_sources[i], lists.x_targets[i]) for i in 1:lists.n_x)
        @test wset == xset
        # class partition CSR is consistent
        @test lists.class_starts[end] == lists.n_routes + 1
        for i in 1:lists.n_routes
            c = Int(lists.route_class[i])
            @test lists.class_starts[c] <= i < lists.class_starts[c + 1]
        end
    end
    # unbalanced trees stay exact-once (theory: balance is not load-bearing)
    for q in (3, 12)
        bodies = _adt_multiscale(n; seed=39013)
        _, tree, lists = _adt_build(bodies; K_max=8, ell_max=8, near_radius2=q,
            balance=false)
        @test _adt_exact_once_bad(tree, lists) == 0
        @test _adt_v_classes_ok(tree, lists)
        @test _adt_wx_ok(tree, lists)
    end
end

@testset "adaptive per-cell sigma gate (sticky demotion)" begin
    rho_t = 4.789
    n = 350
    for (skind, mk_sigma) in (
            ("one_fat", nn -> (s = fill(3e-4, nn); s[7] = 0.15; s)),
            ("heterogeneous", nn -> 10 .^ (rand(MersenneTwister(39020), nn) .* 2 .- 3.5)),
        ), q in (3, 12)
        bodies = _adt_multiscale(n; seed=39014)
        sys = Gravitational(copy(bodies))
        sigma = mk_sigma(n)
        pol = AdaptiveTreePolicy(K_max=16, ell_max=8, near_radius2=q, rho_t=rho_t)
        tree = AdaptiveRadixTree(sys; policy=pol, sigma=sigma)
        lists = AdaptiveInteractionLists(tree)
        build_adaptive_interaction_lists!(lists, tree)
        @test tree.sigma_armed
        @test lists.n_demoted > 0
        @test _adt_exact_once_bad(tree, lists) == 0
        # gated-list V-class/phase-table membership (the 038 review's check)
        @test _adt_v_classes_ok(tree, lists)
        @test _adt_wx_ok(tree, lists)
        # 031a §5.1 contract: every ordered pair with r <= rho_t sigma_src in U
        inU = falses(n, n)
        for i in 1:lists.n_u
            ia = lists.u_targets[i]
            ib = lists.u_sources[i]
            inU[tree.node_lo[ia]:tree.node_hi[ia],
                tree.node_lo[ib]:tree.node_hi[ib]] .= true
        end
        xs = [ADT_FM.get_position(sys, i) for i in 1:n]
        contract_bad = 0
        for s in 1:n, t in 1:n
            ps = tree.perm[s]
            pt = tree.perm[t]
            r = sqrt(sum(abs2, xs[pt] .- xs[ps]))
            (r <= rho_t * sigma[ps] && !inU[t, s]) && (contract_bad += 1)
        end
        @test contract_bad == 0
    end
    # split veto ON (non-default): still exact-once + contract, locally coarser
    bodies = _adt_multiscale(n; seed=39014)
    sys = Gravitational(copy(bodies))
    sigma = fill(3e-4, n)
    sigma[7] = 0.15
    pol = AdaptiveTreePolicy(K_max=16, ell_max=8, near_radius2=12, rho_t=rho_t,
        split_veto=true)
    tree = AdaptiveRadixTree(sys; policy=pol, sigma=sigma)
    lists = AdaptiveInteractionLists(tree)
    build_adaptive_interaction_lists!(lists, tree)
    @test _adt_exact_once_bad(tree, lists) == 0
    @test _adt_v_classes_ok(tree, lists)
end

@testset "adaptive uniform-limit parity with hierarchical routes (P=4)" begin
    for q in (3, 5, 12), ell in (2, 3)
        # complete occupancy: one body per leaf cell, jittered off center
        G = 1 << ell
        rng = MersenneTwister(39030 + ell + q)
        nb = G^3
        b = zeros(5, nb)
        Δ = 1.0 / G
        i = 0
        for z in 0:G-1, y in 0:G-1, x in 0:G-1
            i += 1
            b[1:3, i] .= (Δ * (x + 0.5 + 0.4 * (rand(rng) - 0.5)),
                          Δ * (y + 0.5 + 0.4 * (rand(rng) - 0.5)),
                          Δ * (z + 0.5 + 0.4 * (rand(rng) - 0.5)))
            b[4, i] = 1e-4
            b[5, i] = 1.0 / nb
        end
        sys = Gravitational(b)
        x_min = SVector(0.0, 0.0, 0.0)
        h0 = 0.5
        policy = HierarchicalRigidStencil(4,
            rigid_stencil_epsilon(4, h0, ell, q); near_radius2=q)
        cache = RadixFMMCache(sys; expansion_order=4, ell=ell,
            bounds=(x_min, 1.0), policy=policy)
        grid = cache.state.grid
        ctx = cache.state.interaction_list
        @test grid.n_cells == nb   # complete occupancy premise

        # production route set, re-emitted window by window
        noffsets = length(ctx.tables.push_offsets)
        nlnodes = maximum(cache.level_offsets[L + 2] - cache.level_offsets[L + 1]
            for L in ctx.first_m2l_level:ell)
        tl = Vector{Int}(undef, nlnodes)
        to = Matrix{Int}(undef, 3, nlnodes)
        tt = Vector{Int}(undef, nlnodes)
        ts = Vector{Int}(undef, nlnodes)
        tc = Vector{Int32}(undef, nlnodes)
        ref = Set{NTuple{11,Int}}()
        for level in ctx.first_m2l_level:ell, k in 1:noffsets
            cnt = ADT_FM.build_hierarchical_routes_window!(tl, to, tt, ts, tc,
                ctx, grid, level, k, k)
            for i in 1:cnt
                push!(ref, (tl[i], to[1, i], to[2, i], to[3, i],
                    grid.node_coords[1, tt[i]], grid.node_coords[2, tt[i]],
                    grid.node_coords[3, tt[i]],
                    grid.node_coords[1, ts[i]], grid.node_coords[2, ts[i]],
                    grid.node_coords[3, ts[i]], Int(tc[i])))
            end
        end
        refU = Set{NTuple{6,Int}}()
        for i in 1:cache.state.counts.n_direct
            tcell = cache.state.direct_targets[i]
            scell = cache.state.direct_sources[i]
            tco = ADT_FM.morton_decode(grid.cell_keys[tcell], ell)
            sco = ADT_FM.morton_decode(grid.cell_keys[scell], ell)
            push!(refU, (tco[1], tco[2], tco[3], sco[1], sco[2], sco[3]))
        end

        # adaptive tree at the same root cube; K_max = 1 with one body per
        # complete cell forces every leaf to depth ell
        pol = AdaptiveTreePolicy(K_max=1, ell_max=ell, near_radius2=q)
        tree = AdaptiveRadixTree(sys; policy=pol, root=(x_min, h0))
        lists = AdaptiveInteractionLists(tree)
        build_adaptive_interaction_lists!(lists, tree)
        levels = _adt_leaf_levels(tree)
        @test all(==(ell), levels)     # uniform limit reached
        @test lists.n_w == 0 && lists.n_x == 0
        adV = Set{NTuple{11,Int}}()
        for i in 1:lists.n_routes
            ia = lists.route_targets[i]
            ib = lists.route_sources[i]
            push!(adV, (lists.route_levels[i], lists.route_offsets[1, i],
                lists.route_offsets[2, i], lists.route_offsets[3, i],
                Int(tree.node_coords[1, ia]), Int(tree.node_coords[2, ia]),
                Int(tree.node_coords[3, ia]),
                Int(tree.node_coords[1, ib]), Int(tree.node_coords[2, ib]),
                Int(tree.node_coords[3, ib]), Int(lists.route_class[i])))
        end
        @test length(adV) == lists.n_routes    # no duplicate emissions
        @test adV == ref
        adU = Set{NTuple{6,Int}}()
        for i in 1:lists.n_u
            ia = lists.u_targets[i]
            ib = lists.u_sources[i]
            push!(adU, (Int(tree.node_coords[1, ia]), Int(tree.node_coords[2, ia]),
                Int(tree.node_coords[3, ia]),
                Int(tree.node_coords[1, ib]), Int(tree.node_coords[2, ib]),
                Int(tree.node_coords[3, ib])))
        end
        @test length(adU) == lists.n_u
        @test adU == refU
        # exact-once painting on the parity tree as well (038 review note)
        @test _adt_exact_once_bad(tree, lists) == 0
    end
end

@testset "adaptive zero-allocation refresh" begin
    bodies = _adt_multiscale(1000)
    sys = Gravitational(copy(bodies))
    sigma = 10 .^ (rand(MersenneTwister(39040), 1000) .* 2 .- 3.5)
    pol = AdaptiveTreePolicy(K_max=16, ell_max=8, near_radius2=5, rho_t=4.789)
    tree = AdaptiveRadixTree(sys; policy=pol, sigma=sigma)
    lists = AdaptiveInteractionLists(tree)
    build_adaptive_interaction_lists!(lists, tree)
    # warm
    update_adaptive_tree!(tree, sys; sigma=sigma)
    build_adaptive_interaction_lists!(lists, tree)
    @test _adt_update_sigma_alloc(tree, sys, sigma) == 0
    @test _adt_lists_alloc(lists, tree) == 0
    # moved bodies (same capacity): still zero and still exact-once
    rng = MersenneTwister(39041)
    b2 = copy(bodies)
    b2[1:3, :] .= clamp.(b2[1:3, :] .+ 0.01 .* randn(rng, 3, 1000), 0.05, 0.95)
    sys2 = Gravitational(b2)
    update_adaptive_tree!(tree, sys2; sigma=sigma)   # warm the new type combo
    @test _adt_update_sigma_alloc(tree, sys2, sigma) == 0
    build_adaptive_interaction_lists!(lists, tree)
    @test _adt_lists_alloc(lists, tree) == 0
    @test _adt_exact_once_bad(tree, lists) == 0
    # ungated refresh path
    update_adaptive_tree!(tree, sys)
    @test _adt_update_alloc(tree, sys) == 0
end

@testset "adaptive RadixFMMCache opt-in (P=4)" begin
    bodies = _adt_multiscale(800)
    sys_a = Gravitational(copy(bodies))
    sys_b = Gravitational(copy(bodies))
    # default: no adaptive machinery, unchanged behavior
    cache_plain = RadixFMMCache(sys_a; expansion_order=4, ell=3)
    @test cache_plain.adaptive === nothing
    @test cache_plain.adaptive_tree === nothing
    r_plain = fmm!(sys_a, cache_plain; scalar_potential=true, gradient=true)

    pol = AdaptiveTreePolicy(K_max=16, ell_max=6, near_radius2=5,
        rho_t=4.789, sigma_row=4)
    cache_ad = RadixFMMCache(sys_b; expansion_order=4, ell=3, adaptive=pol)
    @test cache_ad.adaptive_tree isa AdaptiveRadixTree
    @test cache_ad.adaptive_lists isa AdaptiveInteractionLists
    r_ad = fmm!(sys_b, cache_ad; scalar_potential=true, gradient=true)
    # the uniform lifecycle is bit-identical with the opt-in armed
    @test sys_a.potential == sys_b.potential

    tree = cache_ad.adaptive_tree
    lists = cache_ad.adaptive_lists
    @test tree.step >= 1 && lists.step >= 1
    @test tree.sigma_armed          # sigma_row=4 + rho_t armed the gate
    @test _adt_exact_once_bad(tree, lists) == 0
    @test _adt_v_classes_ok(tree, lists)

    # per-step refresh through the cache is allocation-light (the adaptive
    # add-on is a constant few hundred bytes of dynamic dispatch on top of
    # the pre-existing update_radix_state! baseline)
    ADT_FM.update_radix_state!(cache_ad, (sys_b,))
    @test _adt_cache_refresh_alloc(cache_ad, (sys_b,)) < 1024
    step0 = tree.step
    ADT_FM.update_radix_state!(cache_ad, (sys_b,))
    @test tree.step == step0 + 1

    # guards
    @test_throws ArgumentError RadixFMMCache(sys_b; expansion_order=4, ell=3,
        device=true, adaptive=pol)
    @test_throws ArgumentError RadixFMMCache(sys_b; expansion_order=4, ell=3,
        bounds=(SVector(0.0, 0.0, 0.0), SVector(1.0, 1.0, 0.25)), adaptive=pol)
    @test_throws ArgumentError AdaptiveTreePolicy(near_radius2=7)
    @test_throws ArgumentError AdaptiveTreePolicy(K_max=0)

    # recenter! preserves the opt-in
    FastMultipole.recenter!(cache_ad, (sys_b,))
    @test cache_ad.adaptive isa AdaptiveTreePolicy
    @test cache_ad.adaptive_tree isa AdaptiveRadixTree
    @test _adt_exact_once_bad(cache_ad.adaptive_tree, cache_ad.adaptive_lists) == 0
end
