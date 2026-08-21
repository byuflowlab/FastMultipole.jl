# smooth_nearfield_prekill_census.jl — task 041d part 1: registered pre-kill
# census of the smooth-basis (regularized) P2M/M2P nearfield substitution.
#
# Question (user-directed 2026-08-17): can expansions in a sigma-regularized
# basis — P2M + M2P only, no M2L — substitute direct interactions beyond the
# nearest-neighbor list, including *inside* the singular sigma floor where
# solid harmonics are inadmissible?
#
# This script imports task 038's standalone tree/list/case machinery (same
# import used by the 041c census), prices candidate M2P routes with the same
# measured production rates recorded in
# data/multilevel_nearfield_shells/cost_calibration.csv, and writes compact
# checksummed CSVs.  It never evaluates a particle field, is deterministic,
# and is limited to at most 4 local threads.
#
# Registered admissibility model (two policies, bracketing):
#   conservative:  E = q^(p+1)/(1-q),          q = r_S / max(d_min, sigma_min(S))
#   hermite_opt :  E = q^(p+1)/sqrt((p+1)!),   same q
# The sigma-regularized denominator max(d, sigma) encodes the "regularized
# solid harmonics" intuition: the gaussianerf kernel is entire, with Taylor
# coefficients at distance d decaying on the scale max(d, sigma), so a source
# cluster of radius r_S is expandable even inside the singular floor provided
# r_S < max(d, sigma).  The Hermite policy adds the FGT 1/sqrt((p+1)!) gain
# and is deliberately optimistic; a kill under it is airtight on the model.
# Half-budget threshold 5e-4 matches the 041c census.
#
# Registered cost model (anchored to the same 041a-derived rates as 041c):
#   c_direct        = per-body-pair direct rate (case-specific)
#   c_eval(p)       = (m2t_rate/64) * n_c(p)/25   per target per route
#                     (production M2T entries average ~64 targets at P=4 with
#                      25 harmonic coefficients; Cartesian total-degree
#                      n_c(p) = (p+1)(p+2)(p+3)/6)
#   formation       = |S| * n_c(p) * c_flop, c_flop = (m2t_rate/64)/25,
#                     charged once per promoted cluster (reuse counted)
# Per-route promotion rule: c_eval(p) * |T| < |S| * |T| * c_direct, i.e. the
# cluster must contain more than n_break(p) = (m2t_rate/64)/c_direct * n_c(p)/25
# sources.  Aggregates additionally price formation and report the selector
# outcome with the 041c uncertainty margins (25%, 35% sigma-heterogeneous).
#
# Scope caveat (review 2026-08-17): c_eval is a *proxy* — the measured harmonic
# M2T kernel linearly rescaled by coefficient count — not a lower bound for a
# purpose-built Hermite/interpolation/proxy-point/separable M2P kernel.  The
# census verdict is therefore scoped to this registered model, and the report
# says so explicitly.
#
# Two census families are run:
#   1. Subdivision census: refine terminal U source leaves to virtual depths
#      0–3 (clusters <= K_max = 32).
#   2. Coarsened/merged census (added after external review): merge each
#      target's U source leaves by common ancestor at 1–3 coarsening levels,
#      plus the extreme all-sources union, so clusters *larger* than K_max are
#      candidates too.  Outputs coarsened_census.csv.

using SHA
using Statistics
using Printf

include(joinpath(@__DIR__, "adaptive_octree_verify.jl"))

const OUT = joinpath(@__DIR__, "..", "data", "smooth_nearfield_prekill")
const SEED = 41003        # identical to the 041c census for comparability
const RHO_J = 4.789
const TOL_HALF = 5e-4
const PS = 2:2:8
const UNC = Dict("cube"=>0.25, "wake"=>0.25, "multiscale"=>0.25,
    "sigma_multiscale"=>0.35)
# Same measured rates as data/multilevel_nearfield_shells/cost_calibration.csv
# (direct ns/body pair, m2t ns/route); m2l/s2l rates are irrelevant here
# because the tested proposal is P2M/M2P only.
const CAL = Dict(
    "cube"             => (0.006720, 79.65),
    "wake"             => (0.009813, 44.62),
    "multiscale"       => (0.008641, 42.88),
    "sigma_multiscale" => (0.008641, 42.88))

ncoeff(p) = (p + 1) * (p + 2) * (p + 3) ÷ 6
pop(t, i) = t.nodes[i].hi - t.nodes[i].lo + 1

function make_case(name, n)
    if name == "cube"
        xs = make_uniform(n; seed=SEED); sigma = fill(2.0n^(-1/3), n)
    elseif name == "wake"
        xs = make_filament(n; seed=SEED + 1); sigma = fill(3.15n^(-1/3), n)
    elseif name == "multiscale"
        xs = make_multiscale(n; contrast=100.0, seed=SEED + 2)
        sigma = fill(0.30n^(-1/3), n)
    else
        xs = make_multiscale(n; contrast=100.0, seed=SEED + 3)
        r = [sqrt(sum((xs[k,i] - (0.6,0.4,0.55)[k])^2 for k in 1:3)) for i in 1:n]
        sigma = 0.08n^(-1/3) .* exp.(log(18.0) .* (r .<= median(r)))
    end
    return xs, sigma
end

function materialize_virtual!(t, roots, depth)
    frontier = unique(roots)
    for _ in 1:depth
        next = Int[]
        for i in frontier
            nd = t.nodes[i]
            if nd.leaf && nd.level < t.ell_max && nd.lo < nd.hi
                split_node!(t.nodes, t.keys, i, t.ell_max)
            end
            append!(next, t.nodes[i].children)
        end
        frontier = next
    end
    return nothing
end

function sigma_min_per_node(t, sigma)
    smin = fill(Inf, length(t.nodes))
    for (i, nd) in enumerate(t.nodes)
        smin[i] = minimum(@view sigma[t.perm[nd.lo:nd.hi]])
    end
    return smin
end

node_radius(t, i) = sqrt(3.0) * cellwidth(t, t.nodes[i].level) / 2

"""Smallest admissible order for source subnode `is` against target leaf `it`
under the given policy; 0 if none in PS."""
function admissible_p(t, it, is, smin, policy)
    r_s = node_radius(t, is)
    w_s = cellwidth(t, t.nodes[is].level)
    d_min = max(aabb_gap(t, t.nodes[it], t.nodes[is]) + w_s / 2, eps())
    q = r_s / max(d_min, smin[is])
    q < 1 || return 0
    for p in PS
        E = policy == :conservative ? q^(p + 1) / (1 - q) :
            q^(p + 1) / sqrt(factorial(big(p + 1)))
        E <= TOL_HALF && return p
    end
    return 0
end

mutable struct Tally
    original::Int; direct::Int; promoted::Int
    routes::Int; eval_ns::Float64
    clusters::Dict{Int,Int}   # promoted cluster => max order actually used
    porder_hist::Dict{Int,Int}
    inside_floor_pairs::Int   # promoted pairs the singular gap test rejects
end
Tally() = Tally(0, 0, 0, 0, 0.0, Dict{Int,Int}(), Dict{Int,Int}(), 0)

"""Recursively refine one terminal U pair; promote source subnodes to M2P
against the (unsplit) target leaf when smooth-admissible AND per-route
economical.  Target side stays at actual bodies (M2P has no target
truncation), so only the source is split — matching the P2M/M2P-only rule."""
function route_pair!(T, t, it, is, depth, maxdepth, smin, smaxall, cd, ct1, policy)
    np = pop(t, it) * pop(t, is)
    p = it == is ? 0 : admissible_p(t, it, is, smin, policy)
    if p > 0
        nbreak = ct1 / cd * ncoeff(p) / 25
        if pop(t, is) > nbreak
            T.promoted += np; T.routes += 1
            T.eval_ns += pop(t, it) * ct1 * ncoeff(p) / 25
            T.clusters[is] = max(get(T.clusters, is, 0), p)
            T.porder_hist[p] = get(T.porder_hist, p, 0) + 1
            gap = aabb_gap(t, t.nodes[it], t.nodes[is])
            gap < RHO_J * smaxall[is] && (T.inside_floor_pairs += np)
            return nothing
        end
    end
    ch = depth < maxdepth ? t.nodes[is].children : Int[]
    if isempty(ch)
        T.direct += np
    else
        for c in ch
            route_pair!(T, t, it, c, depth + 1, maxdepth, smin, smaxall, cd, ct1, policy)
        end
    end
    return nothing
end

function sigma_max_per_node(t, sigma)
    smax = zeros(Float64, length(t.nodes))
    for (i, nd) in enumerate(t.nodes)
        smax[i] = maximum(@view sigma[t.perm[nd.lo:nd.hi]])
    end
    return smax
end

function census_row(name, n, depth, policy)
    xs, sigma = make_case(name, n)
    t = build_tree(xs, 32, 12)
    balance!(t)
    snode0 = sigma_upward(t, sigma)
    base = build_lists(t, 5; sigma_node=snode0, rho_t=RHO_J)
    @assert check_exact_once(t, base) == 0
    roots = unique(vcat(first.(base.U), last.(base.U)))
    materialize_virtual!(t, roots, depth)
    smin = sigma_min_per_node(t, sigma)
    smax = sigma_max_per_node(t, sigma)
    cd, m2t = CAL[name]
    ct1 = m2t / 64
    T = Tally()
    for pair in base.U
        T.original += pop(t, pair[1]) * pop(t, pair[2])
        route_pair!(T, t, pair[1], pair[2], 0, depth, smin, smax, cd, ct1, policy)
    end
    @assert T.original == T.direct + T.promoted
    # Formation charged once per promoted cluster at the largest order that
    # cluster is actually used at (per-cluster max, not a global pmax — the
    # global-pmax accounting biased mixed-order censuses against promotion).
    form_ns = sum(pop(t, i) * ncoeff(pm) * (ct1 / 25) for (i, pm) in T.clusters; init=0.0)
    baseline = T.original * cd
    proposed = T.direct * cd + T.eval_ns + form_ns
    u = UNC[name]
    promote = proposed * (1 + u) < baseline
    return T, baseline, proposed, promote, form_ns
end

# ---------------------------------------------------------------------------
# Coarsened/merged-cluster census (added 2026-08-17 after external review).
# The base census only subdivides terminal U-list source leaves, so its leaf
# cap cannot by itself bound clusters *larger* than one leaf.  This census
# closes that gap empirically: for each U-list target leaf it merges that
# target's U source leaves into candidate super-clusters — grouped by common
# ancestor at 1..3 levels of coarsening, plus the extreme all-sources union —
# and tests each merged cluster under the same admissibility policies and
# break-even economics, with |S| now free to exceed K_max = 32.
#
# Monotonicity note (recorded in the theory doc): merging near sources never
# improves admissibility — for a union S ⊇ leaf L, r_S >= r_L,
# d_min(S) <= d_min(L), sigma_min(S) <= sigma_min(L), so q(S) >= q(L) for the
# largest member — while economics improves with |S|.  Which effect wins is
# quantitative, hence this census.

anc(t, i, c) = (j = i; for _ in 1:c; t.nodes[j].parent == 0 && break; j = t.nodes[j].parent; end; j)

function node_box(t, nd)
    w = cellwidth(t, nd.level)
    lo = (t.x0[1] + nd.cx * w, t.x0[2] + nd.cy * w, t.x0[3] + nd.cz * w)
    return lo, lo .+ w
end

function union_stats(t, xs, sigma, members)
    lo = [Inf, Inf, Inf]; hi = [-Inf, -Inf, -Inf]
    S = 0; smn = Inf; smx = 0.0
    for l in members
        nd = t.nodes[l]
        for s in nd.lo:nd.hi
            p = t.perm[s]
            for a in 1:3
                lo[a] = min(lo[a], xs[a, p]); hi[a] = max(hi[a], xs[a, p])
            end
            smn = min(smn, sigma[p]); smx = max(smx, sigma[p])
        end
        S += nd.hi - nd.lo + 1
    end
    ctr = ntuple(a -> (lo[a] + hi[a]) / 2, 3)
    r = sqrt(sum(((hi[a] - lo[a]) / 2)^2 for a in 1:3))
    return S, ctr, r, smn, smx, (lo[1], lo[2], lo[3]), (hi[1], hi[2], hi[3])
end

point_box_dist(pt, blo, bhi) =
    sqrt(sum(max(blo[a] - pt[a], pt[a] - bhi[a], 0.0)^2 for a in 1:3))
box_box_gap(alo, ahi, blo, bhi) =
    sqrt(sum(max(alo[a] - bhi[a], blo[a] - ahi[a], 0.0)^2 for a in 1:3))

"""Smallest admissible order in PS for a merged cluster of radius r about ctr
against target box (tlo,thi); 0 if none."""
function admissible_p_union(r, d_min, smn, policy)
    q = r / max(d_min, smn)
    q < 1 || return 0
    for p in PS
        E = policy == :conservative ? q^(p + 1) / (1 - q) :
            q^(p + 1) / sqrt(factorial(big(p + 1)))
        E <= TOL_HALF && return p
    end
    return 0
end

function coarsened_row(name, n, coarsen, policy)
    xs, sigma = make_case(name, n)
    t = build_tree(xs, 32, 12)
    balance!(t)
    snode0 = sigma_upward(t, sigma)
    base = build_lists(t, 5; sigma_node=snode0, rho_t=RHO_J)
    @assert check_exact_once(t, base) == 0
    cd, m2t = CAL[name]
    ct1 = m2t / 64
    # target => its U source leaves (same target orientation as the base census)
    tgts = Dict{Int,Vector{Int}}()
    original = 0
    for pr in base.U
        original += pop(t, pr[1]) * pop(t, pr[2])
        pr[1] == pr[2] && continue
        push!(get!(tgts, pr[1], Int[]), pr[2])
    end
    ncand = 0; nadm = 0; maxS = 0
    minq = Inf; maxratio = 0.0
    promoted = 0; routes = 0; eval_ns = 0.0
    inside_floor = 0
    clusters = Dict{Any,Tuple{Int,Int}}()   # key => (|S|, max order used)
    for (it, srcs) in tgts
        tlo, thi = node_box(t, t.nodes[it])
        groups = coarsen == :all ? Dict(:all => srcs) :
            begin
                g = Dict{Int,Vector{Int}}()
                for s in srcs
                    push!(get!(g, anc(t, s, coarsen), Int[]), s)
                end
                g
            end
        for (key, members) in groups
            length(members) >= 2 || continue      # singletons = base census
            S, ctr, r, smn, smx, ulo, uhi = union_stats(t, xs, sigma, members)
            d_min = max(point_box_dist(ctr, tlo, thi), eps())
            ncand += 1
            maxS = max(maxS, S)
            minq = min(minq, r / max(d_min, smn))
            p = admissible_p_union(r, d_min, smn, policy)
            p > 0 || continue
            nadm += 1
            nbreak = ct1 / cd * ncoeff(p) / 25
            maxratio = max(maxratio, S / nbreak)
            S > nbreak || continue
            np = pop(t, it) * S
            promoted += np; routes += 1
            eval_ns += pop(t, it) * ct1 * ncoeff(p) / 25
            ckey = coarsen == :all ? (it, :all) : key
            old = get(clusters, ckey, (S, 0))
            clusters[ckey] = (S, max(old[2], p))
            box_box_gap(tlo, thi, ulo, uhi) < RHO_J * smx && (inside_floor += np)
        end
    end
    form_ns = sum(S * ncoeff(pm) * (ct1 / 25) for (S, pm) in values(clusters); init=0.0)
    baseline = original * cd
    proposed = (original - promoted) * cd + eval_ns + form_ns
    u = UNC[name]
    promote = proposed * (1 + u) < baseline
    return (; original, ncand, nadm, maxS, minq, maxratio, promoted, routes,
        inside_floor, nclusters=length(clusters), eval_ns, form_ns,
        baseline, proposed, promote)
end

function write_outputs()
    mkpath(OUT)
    rows = String["case,n,seed,policy,K_max,virtual_depth,p_orders_swept,original_body_pairs,residual_direct_body_pairs,promoted_body_pairs,promoted_fraction,inside_floor_promoted_pairs,m2p_routes,promoted_clusters,order_histogram,baseline_ns,proposed_ns,formation_ns,selector_decision,rate_source"]
    for name in ("cube", "wake", "multiscale", "sigma_multiscale"),
            policy in (:conservative, :hermite_opt), depth in 0:3
        n = 2048
        T, baseline, proposed, promote, form_ns = census_row(name, n, depth, policy)
        hist = join(("$(k):$(v)" for (k, v) in sort(collect(T.porder_hist))), ';')
        push!(rows, join((name, n, SEED, policy, 32, depth, "2:2:8",
            T.original, T.direct, T.promoted,
            @sprintf("%.9f", T.promoted / T.original), T.inside_floor_pairs,
            T.routes, length(T.clusters), isempty(hist) ? "none" : hist,
            @sprintf("%.3f", baseline), @sprintf("%.3f", proposed),
            @sprintf("%.3f", form_ns),
            promote ? "promote" : "direct_fallback",
            "multilevel_nearfield_shells/cost_calibration.csv"), ','))
    end
    write(joinpath(OUT, "m2p_census.csv"), join(rows, '\n') * "\n")
    # Coarsened/merged-cluster census: ancestor-grouped and all-union merges of
    # each target's U source leaves, |S| free to exceed K_max.
    crows = String["case,n,seed,policy,coarsen,merged_candidates,admissible_candidates,max_cluster_sources,min_q,max_econ_ratio,promoted_clusters,promoted_routes,promoted_body_pairs,inside_floor_promoted_pairs,original_body_pairs,baseline_ns,proposed_ns,formation_ns,selector_decision"]
    coarsened_promoted_any = false
    coarsened_maxratio = 0.0
    for name in ("cube", "wake", "multiscale", "sigma_multiscale"),
            policy in (:conservative, :hermite_opt), coarsen in (1, 2, 3, :all)
        n = 2048
        R = coarsened_row(name, n, coarsen, policy)
        coarsened_promoted_any |= R.promote
        coarsened_maxratio = max(coarsened_maxratio, R.maxratio)
        push!(crows, join((name, n, SEED, policy, coarsen, R.ncand, R.nadm,
            R.maxS, @sprintf("%.4f", R.minq), @sprintf("%.4f", R.maxratio),
            R.nclusters, R.routes, R.promoted, R.inside_floor, R.original,
            @sprintf("%.3f", R.baseline), @sprintf("%.3f", R.proposed),
            @sprintf("%.3f", R.form_ns),
            R.promote ? "promote" : "direct_fallback"), ','))
    end
    write(joinpath(OUT, "coarsened_census.csv"), join(crows, '\n') * "\n")
    # Analytic break-even table: sources per cluster required for one M2P
    # evaluation to beat the direct rate, per case and order.
    brk = String["case,p,n_coeff,c_eval_per_target_ns,c_direct_ns,break_even_cluster_sources,K_max_cap,possible_at_K32"]
    for (name, (cd, m2t)) in sort(collect(CAL)), p in PS
        ct = m2t / 64 * ncoeff(p) / 25
        nb = ct / cd
        push!(brk, join((name, p, ncoeff(p), @sprintf("%.4f", ct),
            @sprintf("%.6f", cd), @sprintf("%.1f", nb), 32, nb < 32), ','))
    end
    write(joinpath(OUT, "break_even.csv"), join(brk, '\n') * "\n")
    manifest = "seed,case_source,tree_source,rate_source,n,K_max,threads\n" *
        "$(SEED),adaptive_octree_verify.jl seeded constructors (041c-identical),adaptive_octree_verify.jl," *
        "data/multilevel_nearfield_shells/cost_calibration.csv,2048,32,$(Threads.nthreads())\n"
    write(joinpath(OUT, "manifest.csv"), manifest)
    promoted_any = any(endswith(r, "promote,multilevel_nearfield_shells/cost_calibration.csv") for r in rows[2:end])
    nb_min = minimum((m2t / 64 * ncoeff(2) / 25) / cd for (name, (cd, m2t)) in CAL)
    any_promo = promoted_any || coarsened_promoted_any
    report = "041d part 1: smooth-basis P2M/M2P pre-kill census\n" *
        "break-even cluster size (best case, p=2): $(@sprintf("%.0f", nb_min)) sources vs K_max=32 cap\n" *
        "selector promotions across all case/policy/depth rows: $(promoted_any ? "PRESENT" : "NONE")\n" *
        "coarsened/merged-cluster selector promotions: $(coarsened_promoted_any ? "PRESENT" : "NONE")\n" *
        "coarsened max economic ratio |S|/n_break over admissible candidates: $(@sprintf("%.3f", coarsened_maxratio))\n" *
        "verdict=$(any_promo ? "REVIEW" : "NO-GO (tested census; universal linear-basis closure NOT claimed)"): see theory/sigma-adaptive-smooth-nearfield.md\n" *
        "threads=$(Threads.nthreads())\n"
    write(joinpath(OUT, "report.txt"), report)
    files = sort(filter(f -> !endswith(f, "checksums.sha256"), readdir(OUT; join=true)))
    open(joinpath(OUT, "checksums.sha256"), "w") do io
        for f in files
            println(io, bytes2hex(sha256(read(f))), "  ", basename(f))
        end
    end
    println(report)
end

Threads.nthreads() <= 4 || error("041d local census is limited to at most 4 threads")
write_outputs()
