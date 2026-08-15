# fm040_lifecycle_cost.jl — task 040 acceptance measurement (pre-registered).
#
# Measures the full HOST resident lifecycle (fmm!: update + B2M/M2M/V-M2L/S2L/
# L2L/U-direct/L2B/M2T + finalize) on the adaptive octree against the uniform
# production host path, on the two Integration-Phase cases plus the multi-scale
# case (identical generators/seeds to fm039_construction_cost.jl):
#
#   unitcube      — random field, uniform in the unit cube;
#   wake          — helical wake cylinder positions (033): solid cylinder,
#                   D = 1, length 5D, particles uniform in its volume;
#   multiscale100 — unit cube + embedded cluster at 100x density contrast
#                   (038 oracle case).
#
# Pre-registered protocol (committed before job submission):
#   - Gravitational point-mass system (test/gravitational.jl), P = 4, Float64,
#     q = 5; n in {1e5, 1e6}.
#   - adaptive: K_max in {64, 128} at ell_max = 10, balance on, veto off,
#     no sigma gate; explicit capacities node_cap = 8*cld(n,K)+1024,
#     u = wx = 60*node_cap, v = 250*node_cap (the 039 measured-ratio-backed
#     overrides). Cache built with adaptive=..., ell = 5 (the uniform
#     structures still refresh per step — the known 040 double-refresh cost is
#     part of the measured adaptive step and is reported separately).
#   - uniform baseline: RadixFMMCache default hierarchical policy at
#     ell in {5, 6}.
#   - Same-job anchors: all configs of one n timed in one process, one CPU
#     node, single Julia thread. Warm step = median of 5 fmm! calls after 2
#     warm-ups; t_update / t_lifecycle also medianed separately (adaptive:
#     update_radix_state! and run_adaptive_host_radix_lifecycle!; uniform:
#     update_radix_state! and run_host_radix_lifecycle!).
#   - Accuracy: velocity rel RMS at 2000 deterministically sampled targets
#     against an exact direct sum over ALL sources (one reference per
#     (case, n), reused by every config) — the 1e-3 gate is REPORTED per row,
#     asserted only in the analysis (a failure is data, not a crash).
#   - Counts: leaves, popmax, U body pairs, V routes, W/X entries (adaptive);
#     cells, popmax, direct pairs, routes (uniform) — the 038 cost-mechanism
#     record (bounded leaf population at per-region depth).
#   - A per-config failure is recorded as a row with status != ok and the
#     sweep continues.
#
# Amendment (2026-08-14 22:37 MDT, logged in the campaign decision log,
# committed before resubmission): the first submission (job 13179268) was
# cancelled at ~1h because the CSV was only written at sweep end (a wall-limit
# timeout would have lost every completed row) and stdout was block-buffered
# (no progress visibility). The CSV is now (re)written after EVERY row and
# stdout is flushed per row; the wall limit is raised to 12h. The measurement
# protocol itself (cases, params, medians, anchors, accuracy sampling) is
# UNCHANGED.
#
# Output: MATRIX_OPERATOR_REFACTOR/data/fm040_lifecycle_cost.csv
# Usage:  julia --project=. -t 1 MATRIX_OPERATOR_REFACTOR/scripts/fm040_lifecycle_cost.jl [nlist]

using FastMultipole
using FastMultipole.StaticArrays
using Random
using Printf
using Statistics

const FM = FastMultipole
const OUTFILE = joinpath(@__DIR__, "..", "data", "fm040_lifecycle_cost.csv")
const REPO = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(REPO, "test", "gravitational.jl"))

#--- cases (identical to fm039_construction_cost.jl) ---#

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

#--- sampled direct reference (velocity at sampled targets, all sources) ---#

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
            # adaptive sweep
            for K in (64, 128)
                node_cap = 8 * cld(n, K) + 1024
                pol = AdaptiveTreePolicy(K_max=K, ell_max=10, near_radius2=5,
                    node_capacity=node_cap, u_capacity=60 * node_cap,
                    v_capacity=250 * node_cap, wx_capacity=60 * node_cap)
                local sys, cache
                t0 = time_ns()
                try
                    sys = Gravitational(copy(b))
                    cache = FM.RadixFMMCache(sys; expansion_order=P, ell=5,
                        adaptive=pol)
                catch err
                    push!(rows, "$name,$n,adaptive,$K,fail:$(typeof(err))," *
                        join(fill("", 11), ","))
                    @printf("%-13s n=%-8d adaptive K=%-3d FAILED %s\n", name, n, K, typeof(err))
                    write_rows(rows)
                    continue
                end
                t_cold = (time_ns() - t0) / 1e9
                fmm!(sys, cache; scalar_potential=false, gradient=true)
                fmm!(sys, cache; scalar_potential=false, gradient=true)
                t_step = median5(() -> fmm!(sys, cache; scalar_potential=false, gradient=true))
                t_update = median5(() -> FM.update_radix_state!(cache, (sys,)))
                t_life = median5(() -> FM.run_adaptive_host_radix_lifecycle!(cache))
                rel = sampled_rel_rms(sys, targets, gref)
                tree = cache.adaptive_tree; lists = cache.adaptive_lists
                popmax, u_pairs = adaptive_counts(tree, lists)
                push!(rows, join(Any[name, n, "adaptive", K, "ok",
                    round(t_cold; digits=3), round(t_update; digits=2),
                    round(t_life; digits=2), round(t_step; digits=2),
                    @sprintf("%.3e", rel), tree.n_leaves, popmax, u_pairs,
                    lists.n_routes, lists.n_w, lists.n_x], ","))
                @printf("%-13s n=%-8d adaptive K=%-3d cold=%.2fs update=%.1fms life=%.1fms step=%.1fms rel=%.3e popmax=%d\n",
                    name, n, K, t_cold, t_update, t_life, t_step, rel, popmax)
                write_rows(rows)
                sys = nothing; cache = nothing; GC.gc()
            end
            # uniform baseline
            for ell in (5, 6)
                local sys, cache
                t0 = time_ns()
                try
                    sys = Gravitational(copy(b))
                    cache = FM.RadixFMMCache(sys; expansion_order=P, ell=ell)
                catch err
                    push!(rows, "$name,$n,uniform,$ell,fail:$(typeof(err))," *
                        join(fill("", 11), ","))
                    @printf("%-13s n=%-8d uniform ell=%d FAILED %s\n", name, n, ell, typeof(err))
                    write_rows(rows)
                    continue
                end
                t_cold = (time_ns() - t0) / 1e9
                fmm!(sys, cache; scalar_potential=false, gradient=true)
                fmm!(sys, cache; scalar_potential=false, gradient=true)
                t_step = median5(() -> fmm!(sys, cache; scalar_potential=false, gradient=true))
                t_update = median5(() -> FM.update_radix_state!(cache, (sys,)))
                t_life = median5(() -> FM.run_host_radix_lifecycle!(cache.state))
                rel = sampled_rel_rms(sys, targets, gref)
                grid = cache.state.grid
                counts = cache.state.counts
                popmax = maximum(grid.cell_ranges[2, c] for c in 1:grid.n_cells)
                u_pairs = 0
                for i in 1:counts.n_direct
                    u_pairs += grid.cell_ranges[2, cache.state.direct_targets[i]] *
                               grid.cell_ranges[2, cache.state.direct_sources[i]]
                end
                ctx = cache.state.interaction_list
                push!(rows, join(Any[name, n, "uniform", ell, "ok",
                    round(t_cold; digits=3), round(t_update; digits=2),
                    round(t_life; digits=2), round(t_step; digits=2),
                    @sprintf("%.3e", rel), counts.n_cells, popmax, u_pairs,
                    ctx.total_routes, 0, 0], ","))
                @printf("%-13s n=%-8d uniform  ell=%-3d cold=%.2fs update=%.1fms life=%.1fms step=%.1fms rel=%.3e popmax=%d\n",
                    name, n, ell, t_cold, t_update, t_life, t_step, rel, popmax)
                write_rows(rows)
                sys = nothing; cache = nothing; GC.gc()
            end
        end
    end
    write_rows(rows)
    println("wrote ", OUTFILE)
end

main()
