# fm041_cuda_cost.jl — task 041 acceptance measurement (pre-registered).
#
# Measures the DEVICE-resident adaptive octree lifecycle against the uniform
# device path on a single H200, on the three fm039/fm040 cases (identical
# generators/seeds):
#
#   unitcube      — random field, uniform in the unit cube;
#   wake          — helical wake cylinder positions (033): solid cylinder,
#                   D = 1, length 5D, particles uniform in its volume;
#   multiscale100 — unit cube + embedded cluster at 100x density contrast.
#
# Pre-registered protocol (committed before job submission):
#   - Gravitational point-mass system, P = 4 (expansion_order=3 convention is
#     NOT used here: expansion_order = P = 4 matches fm039/fm040 rows), q = 5;
#     n in {1e5, 1e6}; precision TF in {Float64, Float32}.
#   - adaptive: K_max in {64, 128} at ell_max = 10, balance on, veto off, no
#     sigma gate; the fm040 measured-ratio-backed capacity overrides
#     (node_cap = 8*cld(n,K)+1024, u = wx = 60*node_cap, v = 250*node_cap).
#     Cache ell = 5 (the uniform device structures are BUILT at construction
#     but do not refresh per step — the device path branches; their
#     construction cost and memory are part of the recorded cold/memory
#     columns).
#   - uniform baseline: device cache at ell in {5, 6}, default hierarchical
#     policy.
#   - both structures measured under TWO M2L strategy options: the shipped
#     default options (concat engine) and DenseTranslationM2L (the 028
#     fused-dense record family, graph-capturable) — strategy is a row key.
#   - Same-job anchors: all configs of one (case, n) in one process on one
#     H200. Warm step = median of 5 fmm! calls (CUDA.synchronize() fenced)
#     after 2 warm-ups; t_update / t_lifecycle also medianed separately.
#     One profiled adaptive update per row records the refresh sub-stages
#     (sort / K_max build / balance / finalize+epoch / DTR / CSR / U-map /
#     stage-groups) via actx.profile_stages.
#   - Accuracy: velocity rel RMS at 2000 deterministically sampled targets
#     against an exact Float64 direct sum over ALL sources (one reference per
#     (case, n), reused by every config) — the 1e-3 gate is REPORTED per row,
#     asserted only in the analysis.
#   - Memory: device bytes in use (total - free) before/after construction.
#   - A per-config failure is recorded as a row with status != ok and the
#     sweep continues. The CSV is (re)written after EVERY row; stdout is
#     flushed per row (fm040 amendment convention).
#
# Output: MATRIX_OPERATOR_REFACTOR/data/fm041_cuda_cost.csv
# Usage:  julia --project=<env> -t 1 MATRIX_OPERATOR_REFACTOR/scripts/fm041_cuda_cost.jl [nlist]

using FastMultipole
using FastMultipole.StaticArrays
using CUDA
using Random
using Printf
using Statistics

const FM = FastMultipole
const OUTFILE = joinpath(@__DIR__, "..", "data", "fm041_cuda_cost.csv")
const REPO = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(REPO, "test", "gravitational.jl"))

FastMultipole.load_cuda_radix_lifecycle!() ||
    error("CUDA radix lifecycle failed to load: " * FastMultipole.cuda_radix_status())

#--- cases (identical to fm039/fm040) ---#

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

#--- sampled direct reference (Float64, all sources) ---#

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
            d = Float64(sys.potential[4 + a, t]) - gref[a, k]
            err += d * d
            nrm += gref[a, k]^2
        end
    end
    return sqrt(err / nrm)
end

median5(f) = median([(CUDA.synchronize(); t0 = time_ns(); f(); CUDA.synchronize();
    (time_ns() - t0) / 1e6) for _ in 1:5])

_used_gb() = (CUDA.total_memory() - CUDA.free_memory()) / 2^30

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

const STRATS = (
    # "concat": the shipped concat window engine, selected EXPLICITLY (the
    # bare-default SharedRotationM2L options are rejected by the device build
    # — job 13180706; the intended engine is what the 041 tests exercise)
    ("concat", TF -> FM.CUDARadixLifecycleOptions(precision=TF,
        m2l_strategy=FM.ConcatenatedFixedZM2L())),
    ("dense", TF -> FM.CUDARadixLifecycleOptions(precision=TF,
        m2l_strategy=FM.DenseTranslationM2L(apply_chunk=64, build_chunk=8))),
)

function main()
    nlist = length(ARGS) >= 1 ? parse.(Int, split(ARGS[1], ",")) : [100_000, 1_000_000]
    P = 4
    rows = String[]
    push!(rows, "case,n,tf,strategy,structure,param,status,t_cold_s,mem_gb," *
        "t_update_ms,t_lifecycle_ms,t_step_ms,vel_rel_rms," *
        "n_nodes,n_leaves,popmax,n_u,u_pairs,v_routes,w_entries,x_entries," *
        "t_sort_ms,t_build_ms,t_balance_ms,t_finalize_ms,t_dtr_ms,t_csr_ms," *
        "t_umap_ms,t_groups_ms")
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
            for TF in (Float64, Float32), (sname, mkopts) in STRATS
                bTF = TF.(b)
                # adaptive sweep
                for K in (64, 128)
                    node_cap = 8 * cld(n, K) + 1024
                    pol = AdaptiveTreePolicy(K_max=K, ell_max=10, near_radius2=5,
                        node_capacity=node_cap, u_capacity=60 * node_cap,
                        v_capacity=250 * node_cap, wx_capacity=60 * node_cap)
                    key = "$name,$n,$TF,$sname,adaptive,$K"
                    local sys, cache
                    mem0 = _used_gb()
                    t0 = time_ns()
                    try
                        sys = Gravitational(copy(bTF))
                        cache = FM.RadixFMMCache(sys; expansion_order=P, ell=5,
                            adaptive=pol, options=mkopts(TF), device=true)
                        CUDA.synchronize()
                    catch err
                        emsg = replace(first(sprint(showerror, err), 90), "," => ";", "\n" => " ")
                        push!(rows, "$key,fail:$(typeof(err)) $emsg," * join(fill("", 21), ","))
                        @printf("%s FAILED %s %s\n", key, typeof(err), emsg)
                        write_rows(rows)
                        CUDA.reclaim(); GC.gc()
                        continue
                    end
                    t_cold = (time_ns() - t0) / 1e9
                    mem1 = _used_gb()
                    fmm!(sys, cache; scalar_potential=false, gradient=true)
                    fmm!(sys, cache; scalar_potential=false, gradient=true)
                    t_step = median5(() -> fmm!(sys, cache; scalar_potential=false, gradient=true))
                    t_update = median5(() -> FM.update_cuda_radix_state!(cache, (sys,)))
                    t_life = median5(() -> FM.run_cuda_adaptive_radix_lifecycle!(
                        cache.adaptive_state, cache.adaptive_tree))
                    actx = cache.adaptive_tree
                    # one profiled update for the refresh sub-stage record.
                    # Bodies are unmoved here, so the occupancy-epoch fast path
                    # would skip DTR/CSR/U-map/groups and zero those stages;
                    # dropping the snapshot forces the FULL-REBUILD breakdown
                    # (t_update medians above remain the honest warm per-step
                    # cost, which includes the epoch fast path).
                    actx.epoch_have = false
                    actx.profile_stages = true
                    FM.update_cuda_radix_state!(cache, (sys,))
                    actx.profile_stages = false
                    stages = actx.stage_ns ./ 1e6
                    fmm!(sys, cache; scalar_potential=false, gradient=true)
                    rel = sampled_rel_rms(sys, targets, gref)
                    cr = Array(actx.grid.cell_ranges)
                    nl = actx.n_leaves
                    popmax = maximum(cr[2, c] for c in 1:nl)
                    dt = Array(cache.adaptive_state.direct_targets)[1:actx.n_u]
                    ds = Array(cache.adaptive_state.direct_sources)[1:actx.n_u]
                    u_pairs = sum(Int(cr[2, dt[i]]) * Int(cr[2, ds[i]]) for i in 1:actx.n_u)
                    push!(rows, join(Any[name, n, TF, sname, "adaptive", K, "ok",
                        round(t_cold; digits=3), round(mem1 - mem0; digits=3),
                        round(t_update; digits=3), round(t_life; digits=3),
                        round(t_step; digits=3), @sprintf("%.3e", rel),
                        actx.n_nodes, nl, popmax, actx.n_u, u_pairs,
                        actx.n_routes, actx.n_w, actx.n_x,
                        [round(stages[j]; digits=3) for j in 1:8]...], ","))
                    @printf("%s cold=%.2fs mem=%.2fGB update=%.2fms life=%.2fms step=%.2fms rel=%.3e popmax=%d\n",
                        key, t_cold, mem1 - mem0, t_update, t_life, t_step, rel, popmax)
                    write_rows(rows)
                    sys = nothing; cache = nothing; GC.gc(); CUDA.reclaim()
                end
                # uniform device baseline
                for ell in (5, 6)
                    key = "$name,$n,$TF,$sname,uniform,$ell"
                    local sys, cache
                    mem0 = _used_gb()
                    t0 = time_ns()
                    try
                        sys = Gravitational(copy(bTF))
                        cache = FM.RadixFMMCache(sys; expansion_order=P, ell=ell,
                            options=mkopts(TF), device=true)
                        CUDA.synchronize()
                    catch err
                        emsg = replace(first(sprint(showerror, err), 90), "," => ";", "\n" => " ")
                        push!(rows, "$key,fail:$(typeof(err)) $emsg," * join(fill("", 21), ","))
                        @printf("%s FAILED %s %s\n", key, typeof(err), emsg)
                        write_rows(rows)
                        CUDA.reclaim(); GC.gc()
                        continue
                    end
                    t_cold = (time_ns() - t0) / 1e9
                    mem1 = _used_gb()
                    fmm!(sys, cache; scalar_potential=false, gradient=true)
                    fmm!(sys, cache; scalar_potential=false, gradient=true)
                    t_step = median5(() -> fmm!(sys, cache; scalar_potential=false, gradient=true))
                    t_update = median5(() -> FM.update_cuda_radix_state!(cache, (sys,)))
                    t_life = median5(() -> FM.run_cuda_radix_lifecycle!(cache.state))
                    fmm!(sys, cache; scalar_potential=false, gradient=true)
                    rel = sampled_rel_rms(sys, targets, gref)
                    counts = cache.state.counts
                    cr = Array(cache.state.grid.cell_ranges)
                    popmax = maximum(cr[2, c] for c in 1:counts.n_cells)
                    dt = Array(cache.state.direct_targets)[1:counts.n_direct]
                    ds = Array(cache.state.direct_sources)[1:counts.n_direct]
                    u_pairs = sum(Int(cr[2, dt[i]]) * Int(cr[2, ds[i]]) for i in 1:counts.n_direct)
                    push!(rows, join(Any[name, n, TF, sname, "uniform", ell, "ok",
                        round(t_cold; digits=3), round(mem1 - mem0; digits=3),
                        round(t_update; digits=3), round(t_life; digits=3),
                        round(t_step; digits=3), @sprintf("%.3e", rel),
                        counts.n_nodes, counts.n_cells, popmax, counts.n_direct,
                        u_pairs, counts.n_routes, 0, 0,
                        fill("", 8)...], ","))
                    @printf("%s cold=%.2fs mem=%.2fGB update=%.2fms life=%.2fms step=%.2fms rel=%.3e popmax=%d\n",
                        key, t_cold, mem1 - mem0, t_update, t_life, t_step, rel, popmax)
                    write_rows(rows)
                    sys = nothing; cache = nothing; GC.gc(); CUDA.reclaim()
                end
            end
        end
    end
    write_rows(rows)
    println("wrote ", OUTFILE)
end

main()
