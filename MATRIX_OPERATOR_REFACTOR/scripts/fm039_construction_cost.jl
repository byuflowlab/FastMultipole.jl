# fm039_construction_cost.jl — task 039 acceptance measurement (pre-registered).
#
# Measures host construction and in-place refresh cost of the adaptive octree
# (tree + U/V/W/X lists) against the uniform-depth radix grid
# (RadixFMMCache update_radix_state! at the production hierarchical default,
# q = 5), on the two Integration-Phase cases plus the multi-scale case:
#
#   unitcube     — random field, uniform in the unit cube;
#   wake         — helical wake cylinder positions (033): solid cylinder,
#                  D = 1, length 5D, particles uniform in its volume;
#   multiscale100 — unit cube + embedded cluster at 100x density contrast
#                  (038 oracle case).
#
# Pre-registered protocol (committed before job submission):
#   - n in {1e5, 1e6}; adaptive K_max in {32, 64, 128} at ell_max = 10, q = 5,
#     balance on, veto off (039 default), no sigma gate; uniform ell in {5, 6}
#     at the default HierarchicalRigidStencil (q = 5 leaf).
#   - Same-job anchors: adaptive and uniform timed in one process, one node,
#     single thread (construction code is single-threaded host code).
#   - Cold = allocate + first build. Warm refresh = median of 5 in-place
#     rebuilds (adaptive: update_adaptive_tree! + build_adaptive_interaction_
#     lists!, reported separately and summed; uniform: update_radix_state!).
#   - Counts recorded per config (nodes, leaves, popmax, list sizes, work
#     terms) plus measured-to-capacity ratios for 040 capacity tightening.
#   - A per-config capacity/argument failure is recorded as a row with
#     status != ok and the sweep continues.
#
# Output: MATRIX_OPERATOR_REFACTOR/data/fm039_construction_cost.csv
# Usage:  julia --project=. -t 1 MATRIX_OPERATOR_REFACTOR/scripts/fm039_construction_cost.jl [nlist]

using FastMultipole
using FastMultipole.StaticArrays
using Random
using Printf
using Statistics

const OUTFILE = joinpath(@__DIR__, "..", "data", "fm039_construction_cost.csv")

#--- minimal position-only system ---#

struct CostBodies
    x::Matrix{Float64}
end
FastMultipole.get_n_bodies(s::CostBodies) = size(s.x, 2)
FastMultipole.get_position(s::CostBodies, i) = SVector{3,Float64}(s.x[1, i], s.x[2, i], s.x[3, i])
FastMultipole.data_per_body(::CostBodies) = 5
FastMultipole.strength_dims(::CostBodies) = 1
FastMultipole.has_vector_potential(::CostBodies) = false
Base.eltype(::CostBodies) = Float64
function FastMultipole.source_system_to_buffer!(buffer, i_buffer, s::CostBodies, i_body)
    buffer[1, i_buffer] = s.x[1, i_body]
    buffer[2, i_buffer] = s.x[2, i_body]
    buffer[3, i_buffer] = s.x[3, i_body]
    buffer[4, i_buffer] = 1e-4
    buffer[5, i_buffer] = 1.0
end
FastMultipole.body_to_multipole!(system::CostBodies, args...) =
    FastMultipole.body_to_multipole!(Point{Source}, system, args...)

#--- cases ---#

function make_unitcube(n; seed=39101)
    rng = MersenneTwister(seed)
    return CostBodies(rand(rng, 3, n))
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
    return CostBodies(x)
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
    return CostBodies(x)
end

#--- measurement helpers ---#

median5(f) = median([(t0 = time_ns(); f(); (time_ns() - t0) / 1e6) for _ in 1:5])

function adaptive_counts(tree, lists)
    leaf_ids = Int.(tree.leaf_index[1:tree.n_leaves])
    popmax = maximum(length(adaptive_node_range(tree, f)) for f in leaf_ids)
    u_pairs = 0
    for i in 1:lists.n_u
        ia = lists.u_targets[i]
        ib = lists.u_sources[i]
        u_pairs += (tree.node_hi[ia] - tree.node_lo[ia] + 1) *
                   (tree.node_hi[ib] - tree.node_lo[ib] + 1)
    end
    w_evals = sum(Int[tree.node_hi[lists.w_targets[i]] - tree.node_lo[lists.w_targets[i]] + 1
        for i in 1:lists.n_w]; init=0)
    x_evals = sum(Int[tree.node_hi[lists.x_sources[i]] - tree.node_lo[lists.x_sources[i]] + 1
        for i in 1:lists.n_x]; init=0)
    return popmax, u_pairs, w_evals, x_evals
end

function main()
    nlist = length(ARGS) >= 1 ? parse.(Int, split(ARGS[1], ",")) : [100_000, 1_000_000]
    rows = String[]
    push!(rows, "case,n,structure,param,status,t_cold_s,t_tree_ms,t_lists_ms,t_refresh_ms," *
        "n_nodes,n_leaves,popmax,u_entries,u_pairs,v_routes,w_entries,x_entries," *
        "w_evals,x_evals,node_cap_ratio,u_cap_ratio,v_cap_ratio,balance_splits")
    for n in nlist
        cases = [
            ("unitcube", make_unitcube(n)),
            ("wake", make_wake(n)),
            ("multiscale100", make_multiscale(n)),
        ]
        for (name, sys) in cases
            # adaptive sweep
            for K in (32, 64, 128)
                node_cap = 8 * cld(n, K) + 1024
                pol = AdaptiveTreePolicy(K_max=K, ell_max=10, near_radius2=5,
                    node_capacity=node_cap, u_capacity=60 * node_cap,
                    v_capacity=250 * node_cap, wx_capacity=60 * node_cap)
                local tree, lists
                t0 = time_ns()
                try
                    tree = AdaptiveRadixTree(sys; policy=pol)
                    lists = AdaptiveInteractionLists(tree)
                    build_adaptive_interaction_lists!(lists, tree)
                catch err
                    push!(rows, "$name,$n,adaptive,$K,fail:$(typeof(err))," *
                        join(fill("", 19), ","))
                    @printf("%-13s n=%-8d adaptive K=%-3d FAILED %s\n", name, n, K, typeof(err))
                    continue
                end
                t_cold = (time_ns() - t0) / 1e9
                t_tree = median5(() -> update_adaptive_tree!(tree, sys))
                t_lists = median5(() -> build_adaptive_interaction_lists!(lists, tree))
                popmax, u_pairs, w_evals, x_evals = adaptive_counts(tree, lists)
                push!(rows, join(Any[name, n, "adaptive", K, "ok",
                    round(t_cold; digits=3), round(t_tree; digits=2),
                    round(t_lists; digits=2), round(t_tree + t_lists; digits=2),
                    tree.n_nodes, tree.n_leaves, popmax, lists.n_u, u_pairs,
                    lists.n_routes, lists.n_w, lists.n_x, w_evals, x_evals,
                    round(tree.n_nodes / tree.node_capacity; digits=4),
                    round(lists.n_u / lists.u_capacity; digits=4),
                    round(lists.n_routes / lists.v_capacity; digits=4),
                    tree.n_balance_splits], ","))
                @printf("%-13s n=%-8d adaptive K=%-3d  cold=%.2fs refresh=%.1f+%.1fms nodes=%d leaves=%d popmax=%d U=%d V=%d W=%d X=%d\n",
                    name, n, K, t_cold, t_tree, t_lists, tree.n_nodes,
                    tree.n_leaves, popmax, lists.n_u, lists.n_routes,
                    lists.n_w, lists.n_x)
                tree = nothing; lists = nothing; GC.gc()
            end
            # uniform baseline (production hierarchical default policy, q=5)
            for ell in (5, 6)
                local cache
                t0 = time_ns()
                try
                    cache = RadixFMMCache(sys; expansion_order=4, ell=ell)
                catch err
                    push!(rows, "$name,$n,uniform,$ell,fail:$(typeof(err))," *
                        join(fill("", 19), ","))
                    @printf("%-13s n=%-8d uniform ell=%d FAILED %s\n", name, n, ell, typeof(err))
                    continue
                end
                t_cold = (time_ns() - t0) / 1e9
                t_refresh = median5(() -> FastMultipole.update_radix_state!(cache, (sys,)))
                grid = cache.state.grid
                counts = cache.state.counts
                popmax = maximum(grid.cell_ranges[2, c] for c in 1:grid.n_cells)
                u_pairs = 0
                leaf_base = cache.level_offsets[ell + 1]
                for i in 1:counts.n_direct
                    tcell = cache.state.direct_targets[i]
                    scell = cache.state.direct_sources[i]
                    u_pairs += grid.cell_ranges[2, tcell] * grid.cell_ranges[2, scell]
                end
                ctx = cache.state.interaction_list
                push!(rows, join(Any[name, n, "uniform", ell, "ok",
                    round(t_cold; digits=3), "", "", round(t_refresh; digits=2),
                    counts.n_nodes, counts.n_cells, popmax, counts.n_direct,
                    u_pairs, ctx.total_routes, 0, 0, 0, 0, "", "", "", 0], ","))
                @printf("%-13s n=%-8d uniform  ell=%-3d cold=%.2fs refresh=%.1fms nodes=%d cells=%d popmax=%d direct=%d V=%d\n",
                    name, n, ell, t_cold, t_refresh, counts.n_nodes,
                    counts.n_cells, popmax, counts.n_direct, ctx.total_routes)
                cache = nothing; GC.gc()
            end
        end
    end
    mkpath(dirname(OUTFILE))
    open(OUTFILE, "w") do io
        for r in rows
            println(io, r)
        end
    end
    println("wrote ", OUTFILE)
end

main()
