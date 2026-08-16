# fm041a_host_widen.jl — task 041a pre-registered measurement (CPU node), part 2/3.
#
# Purpose (041a carry-over from the 040 approval): the fm040 host baseline
# swept only uniform ell in {5,6} and adaptive K in {64,128}, leaving the
# best-uniform (and best-adaptive) configuration at a sweep ENDPOINT in every
# case. Before any publishable "adaptive vs best-uniform" host claim, this
# job widens both sweeps until the per-case best is interior wherever
# feasible, and probes the honest-negative wake n=1e5 case at smaller K.
#
# Pre-registered protocol (committed before job submission; identical
# generators/seeds, system, accuracy instrument, and timing conventions to
# fm040_lifecycle_cost.jl — single Julia thread, one CPU node, warm medians
# of 5 after 2 warm-ups, 2000-target sampled-direct velocity rel RMS,
# incremental CSV, fail rows recorded):
#   - Same-job anchors: for every (case, n) the fm040 best-of-{5,6} uniform
#     depth AND adaptive K=64 are RE-RUN in this job, so widened comparisons
#     never cross jobs (fm040 rows remain the record for ell in {5,6} and
#     K in {64,128}; ratios quoted in figures use this job's anchors).
#   - Widened uniform depths (chosen from the fm040 endpoint analysis):
#       unitcube:      ell=4 at n in {1e5, 1e6}; ell=3 at n=1e5 only
#                      (ell=3 at 1e6 is a ~1.2e11-body-pair direct — priced
#                      out; recorded as infeasible-by-cost, not run).
#       wake:          ell=7 at n in {1e5, 1e6}.
#       multiscale100: ell=4 at n=1e5; ell=7 at n in {1e5, 1e6}.
#   - Widened adaptive K: K=32 at both n for all cases; K=16 at n=1e5 for
#     all cases (wake n=1e5 negative probe; K=16 at n=1e6 priced out).
#   - Gravitational point-mass, P=4, Float64, q=5; adaptive ell_max=10,
#     balance on, veto off, no sigma gate, 039 capacity overrides
#     node_cap = 8*cld(n,K)+1024, u = wx = 60*node_cap, v = 250*node_cap.
#
# Output: MATRIX_OPERATOR_REFACTOR/data/fm041a_host_widen.csv
# Usage:  julia --project=. -t 1 fm041a_host_widen.jl [nlist]

using FastMultipole
using FastMultipole.StaticArrays
using Random
using Printf
using Statistics

const FM = FastMultipole
const OUTFILE = joinpath(@__DIR__, "..", "data", "fm041a_host_widen.csv")
const REPO = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(REPO, "test", "gravitational.jl"))

#--- cases (identical to fm039/fm040/fm041) ---#

function make_unitcube(n; seed=39101)
    rng = MersenneTwister(seed)
    return rand(rng, 3, n)
end

function make_wake(n; seed=39102)
    rng = MersenneTwister(seed)
    x = Matrix{Float64}(undef, 3, n)
    for i in 1:n
        r = 0.5 * sqrt(rand(rng))
        th = 2pi * rand(rng)
        x[1, i] = r * cos(th)
        x[2, i] = r * sin(th)
        x[3, i] = 5.0 * rand(rng)
    end
    return x
end

function make_multiscale(n; contrast=100.0, seed=39103)
    rng = MersenneTwister(seed)
    frac = 0.35
    nc = round(Int, frac * n)
    nb = n - nc
    Rc = (3 * nc / (4pi * contrast * nb))^(1 / 3)
    x = rand(rng, 3, n)
    ctr = (0.6, 0.4, 0.55)
    k = 0
    while k < nc
        p = 2 .* (rand(rng, 3) .- 0.5)
        if sum(abs2, p) <= 1
            k += 1
            x[:, nb + k] .= ctr .+ Rc .* p
        end
    end
    return x
end

function make_bodies(x)
    n = size(x, 2)
    b = zeros(5, n)
    b[1:3, :] .= x
    b[4, :] .= 1e-4
    b[5, :] .= 1.0 / n
    return b
end

function sampled_direct(b, targets::Vector{Int})
    n = size(b, 2)
    g = zeros(3, length(targets))
    c = 1 / (4pi)
    @inbounds for (k, t) in enumerate(targets)
        tx = b[1, t]; ty = b[2, t]; tz = b[3, t]
        gx = 0.0; gy = 0.0; gz = 0.0
        for j in 1:n
            j == t && continue
            dx = tx - b[1, j]; dy = ty - b[2, j]; dz = tz - b[3, j]
            r2 = dx * dx + dy * dy + dz * dz
            r2 == 0 && continue
            q = b[5, j] * c / (r2 * sqrt(r2))
            gx -= q * dx; gy -= q * dy; gz -= q * dz
        end
        g[1, k] = gx; g[2, k] = gy; g[3, k] = gz
    end
    return g
end

function sampled_rel_rms(sys, targets, gref)
    err = 0.0; nrm = 0.0
    for (k, t) in enumerate(targets)
        for a in 1:3
            d = sys.potential[4 + a, t] - gref[a, k]
            err += d * d
            nrm += gref[a, k]^2
        end
    end
    return sqrt(err / nrm)
end

median5(f) = median([(t0 = time_ns(); f(); (time_ns() - t0) / 1e6) for _ in 1:5])

function write_rows(rows)
    mkpath(dirname(OUTFILE))
    open(OUTFILE, "w") do io
        for r in rows
            println(io, r)
        end
    end
    flush(stdout)
    return nothing
end

function adaptive_counts(tree, lists)
    leaf_ids = Int.(tree.leaf_index[1:tree.n_leaves])
    popmax = maximum(length(adaptive_node_range(tree, f)) for f in leaf_ids)
    u_pairs = 0
    for i in 1:lists.n_u
        ia = lists.u_targets[i]; ib = lists.u_sources[i]
        u_pairs += (tree.node_hi[ia] - tree.node_lo[ia] + 1) *
                   (tree.node_hi[ib] - tree.node_lo[ib] + 1)
    end
    return popmax, u_pairs
end

function run_row!(rows, name, n, structure, param, b, targets, gref, P)
    local sys, cache
    t0 = time_ns()
    try
        sys = Gravitational(copy(b))
        if structure == "adaptive"
            node_cap = 8 * cld(n, param) + 1024
            pol = AdaptiveTreePolicy(K_max=param, ell_max=10, near_radius2=5,
                node_capacity=node_cap, u_capacity=60 * node_cap,
                v_capacity=250 * node_cap, wx_capacity=60 * node_cap)
            cache = FM.RadixFMMCache(sys; expansion_order=P, ell=5, adaptive=pol)
        else
            cache = FM.RadixFMMCache(sys; expansion_order=P, ell=param)
        end
    catch err
        emsg = replace(first(sprint(showerror, err), 90), "," => ";", "\n" => " ")
        push!(rows, "$name,$n,$structure,$param,fail:$(typeof(err)) $emsg," *
            join(fill("", 11), ","))
        @printf("%-13s n=%-8d %s %-3d FAILED %s %s\n", name, n, structure, param,
            typeof(err), emsg)
        write_rows(rows)
        return nothing
    end
    t_cold = (time_ns() - t0) / 1e9
    fmm!(sys, cache; scalar_potential=false, gradient=true)
    fmm!(sys, cache; scalar_potential=false, gradient=true)
    t_step = median5(() -> fmm!(sys, cache; scalar_potential=false, gradient=true))
    t_update = median5(() -> FM.update_radix_state!(cache, (sys,)))
    if structure == "adaptive"
        t_life = median5(() -> FM.run_adaptive_host_radix_lifecycle!(cache))
    else
        t_life = median5(() -> FM.run_host_radix_lifecycle!(cache.state))
    end
    rel = sampled_rel_rms(sys, targets, gref)
    if structure == "adaptive"
        tree = cache.adaptive_tree; lists = cache.adaptive_lists
        popmax, u_pairs = adaptive_counts(tree, lists)
        n_leaves = tree.n_leaves
        v_routes = lists.n_routes; n_w = lists.n_w; n_x = lists.n_x
    else
        grid = cache.state.grid
        counts = cache.state.counts
        popmax = maximum(grid.cell_ranges[2, c] for c in 1:grid.n_cells)
        u_pairs = 0
        for i in 1:counts.n_direct
            u_pairs += grid.cell_ranges[2, cache.state.direct_targets[i]] *
                       grid.cell_ranges[2, cache.state.direct_sources[i]]
        end
        n_leaves = counts.n_cells
        v_routes = cache.state.interaction_list.total_routes; n_w = 0; n_x = 0
    end
    push!(rows, join(Any[name, n, structure, param, "ok",
        round(t_cold; digits=3), round(t_update; digits=2),
        round(t_life; digits=2), round(t_step; digits=2),
        @sprintf("%.3e", rel), n_leaves, popmax, u_pairs,
        v_routes, n_w, n_x], ","))
    @printf("%-13s n=%-8d %s %-3d cold=%.2fs update=%.1fms life=%.1fms step=%.1fms rel=%.3e popmax=%d\n",
        name, n, structure, param, t_cold, t_update, t_life, t_step, rel, popmax)
    write_rows(rows)
    sys = nothing; cache = nothing; GC.gc()
    return nothing
end

# per-(case, n) config plan: (structure, param) in run order.
# anchors first (fm040 best-of-{5,6} + adaptive K=64), then widened configs.
function config_plan(name, n)
    plan = Tuple{String,Int}[]
    # anchors
    best56 = name == "unitcube" ? 5 :
             name == "wake" ? 6 :
             (n == 100_000 ? 5 : 6)          # multiscale100
    push!(plan, ("uniform", best56))
    push!(plan, ("adaptive", 64))
    # widened uniform depths
    if name == "unitcube"
        push!(plan, ("uniform", 4))
        n == 100_000 && push!(plan, ("uniform", 3))
    elseif name == "wake"
        push!(plan, ("uniform", 7))
    else # multiscale100
        n == 100_000 && push!(plan, ("uniform", 4))
        push!(plan, ("uniform", 7))
    end
    # widened adaptive K
    push!(plan, ("adaptive", 32))
    n == 100_000 && push!(plan, ("adaptive", 16))
    return plan
end

function main()
    nlist = length(ARGS) >= 1 ? parse.(Int, split(ARGS[1], ",")) : [100_000, 1_000_000]
    P = 4
    rows = String[]
    push!(rows, "case,n,structure,param,status,t_cold_s,t_update_ms,t_lifecycle_ms," *
        "t_step_ms,vel_rel_rms,n_leaves,popmax,u_pairs,v_routes,w_entries,x_entries")
    for n in nlist
        nsample = 2000
        cases = [
            ("unitcube", make_unitcube(n)),
            ("wake", make_wake(n)),
            ("multiscale100", make_multiscale(n)),
        ]
        for (name, x) in cases
            b = make_bodies(x)
            rngs = MersenneTwister(40201)
            targets = sort!(Random.shuffle(rngs, collect(1:n))[1:min(nsample, n)])
            gref = sampled_direct(b, targets)
            for (structure, param) in config_plan(name, n)
                run_row!(rows, name, n, structure, param, b, targets, gref, P)
            end
        end
    end
    write_rows(rows)
    println("wrote ", OUTFILE)
end

main()
