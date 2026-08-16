# fm041a_gpu_widen.jl — task 041a pre-registered measurement (H200), part 1/3.
#
# Purpose (041a figure obligations):
#   (A) WIDEN the uniform depth sweep and the adaptive K sweep so the
#       best-uniform and best-adaptive curves are INTERIOR, not endpoints
#       (040 approval carry-over: the fm040/fm041 baselines swept only
#       ell in {5,6} / K in {64,128}); densify n for the time-vs-n figure.
#   (B) Float32 extension of the widened configs at the fm041 n points.
#   (C) P sweep at fixed geometry for the accuracy-cost frontier figure.
#   (E) leaf-population histograms (structure readout, no timing) for the
#       fat-cell mechanism figure.
#
# Pre-registered protocol (committed before job submission; identical
# generators/seeds, system, accuracy instrument, and timing conventions to
# fm041_cuda_cost.jl — same-job anchors: every (case, n) block re-runs the
# fm041 anchor configs K=64/128, ell=5/6 in the SAME process so widened
# comparisons never cross jobs):
#   - Gravitational point-mass, P = 4 (except section C), q = 5, dense
#     strategy only (DenseTranslationM2L(apply_chunk=64, build_chunk=8) —
#     the measured-best family in fm041 for every case at both precisions).
#   - Section A (Float64): cases {unitcube, wake, multiscale100} x
#     n in {10_000, 31_623, 100_000, 316_228, 1_000_000} x
#     configs {uniform ell in 3..7} + {adaptive K in 32,64,128,256}.
#   - Section B (Float32): same cases, n in {1e5, 1e6}, only the NEW configs
#     {ell 3,4,7; K 32,256} (fm041 already holds F32 ell 5/6, K 64/128).
#   - Section C (Float64, dense): cases {wake, multiscale100}, n = 1e6,
#     P in {2, 3, 6, 8} x configs {adaptive K=64, uniform ell=6, uniform
#     ell=7} (P=4 rows come from section A same-job).
#   - Adaptive policy: ell_max=10, near_radius2=5 (q=5), balance on, veto
#     off, no sigma gate; fm040/fm041 capacity-override formula
#     node_cap = 8*cld(n,K)+1024, u = wx = 60*node_cap, v = 250*node_cap.
#   - Uniform baseline: device cache at the given ell, default hierarchical
#     policy. Expected-infeasible rows (e.g. ell=7 capacity/OOM) are
#     recorded as fail rows with the error text — that is DATA for the
#     capacity-memory figure, not a protocol breach; the sweep continues.
#   - Warm step/update/lifecycle: median of 5 after 2 warm-ups, CUDA.
#     synchronize() fenced; accuracy: velocity rel RMS at 2000 sampled
#     targets vs an exact Float64 all-source direct sum (one reference per
#     (case, n)); 1e-3 gate reported per row, asserted in analysis.
#   - Memory: device bytes in use (total - free) before/after construction.
#   - CSV rewritten after EVERY row; stdout flushed per row.
#   - Section E: for (case, n=1e6) x {adaptive K=64, uniform ell 5, 6},
#     per-leaf population histogram (pop -> leaf count) appended to
#     fm041a_leafpop.csv. Structure readout from already-built caches.
#
# Outputs: MATRIX_OPERATOR_REFACTOR/data/fm041a_gpu_widen.csv
#          MATRIX_OPERATOR_REFACTOR/data/fm041a_pweep.csv
#          MATRIX_OPERATOR_REFACTOR/data/fm041a_leafpop.csv
# Usage:  julia --project=<env> -t 1 fm041a_gpu_widen.jl [nlist]

using FastMultipole
using FastMultipole.StaticArrays
using CUDA
using Random
using Printf
using Statistics

const FM = FastMultipole
const OUTFILE = joinpath(@__DIR__, "..", "data", "fm041a_gpu_widen.csv")
const PFILE = joinpath(@__DIR__, "..", "data", "fm041a_pweep.csv")
const LEAFFILE = joinpath(@__DIR__, "..", "data", "fm041a_leafpop.csv")
const REPO = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(REPO, "test", "gravitational.jl"))

FastMultipole.load_cuda_radix_lifecycle!() ||
    error("CUDA radix lifecycle failed to load: " * FastMultipole.cuda_radix_status())

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

function write_rows(rows, file)
    mkpath(dirname(file))
    open(file, "w") do io
        for r in rows
            println(io, r)
        end
    end
    flush(stdout)
    return nothing
end

mkopts(TF) = FM.CUDARadixLifecycleOptions(precision=TF,
    m2l_strategy=FM.DenseTranslationM2L(apply_chunk=64, build_chunk=8))

adaptive_policy(n, K) = begin
    node_cap = 8 * cld(n, K) + 1024
    AdaptiveTreePolicy(K_max=K, ell_max=10, near_radius2=5,
        node_capacity=node_cap, u_capacity=60 * node_cap,
        v_capacity=250 * node_cap, wx_capacity=60 * node_cap)
end

const HEADER = "case,n,tf,P,structure,param,status,t_cold_s,mem_gb," *
    "t_update_ms,t_lifecycle_ms,t_step_ms,vel_rel_rms," *
    "n_nodes,n_leaves,popmax,n_u,u_pairs,v_routes,w_entries,x_entries"

# Build + measure one config; returns (rowstring, popvec or nothing).
function run_config(name, n, TF, P, structure, param, bTF, targets, gref;
        want_pops::Bool=false)
    key = "$name,$n,$TF,$P,$structure,$param"
    local sys, cache
    mem0 = _used_gb()
    t0 = time_ns()
    try
        sys = Gravitational(copy(bTF))
        if structure == "adaptive"
            cache = FM.RadixFMMCache(sys; expansion_order=P, ell=5,
                adaptive=adaptive_policy(n, param), options=mkopts(TF), device=true)
        else
            cache = FM.RadixFMMCache(sys; expansion_order=P, ell=param,
                options=mkopts(TF), device=true)
        end
        CUDA.synchronize()
    catch err
        emsg = replace(first(sprint(showerror, err), 90), "," => ";", "\n" => " ")
        @printf("%s FAILED %s %s\n", key, typeof(err), emsg)
        CUDA.reclaim(); GC.gc()
        return ("$key,fail:$(typeof(err)) $emsg," * join(fill("", 14), ","), nothing)
    end
    t_cold = (time_ns() - t0) / 1e9
    mem1 = _used_gb()
    fmm!(sys, cache; scalar_potential=false, gradient=true)
    fmm!(sys, cache; scalar_potential=false, gradient=true)
    t_step = median5(() -> fmm!(sys, cache; scalar_potential=false, gradient=true))
    t_update = median5(() -> FM.update_cuda_radix_state!(cache, (sys,)))
    if structure == "adaptive"
        t_life = median5(() -> FM.run_cuda_adaptive_radix_lifecycle!(
            cache.adaptive_state, cache.adaptive_tree))
    else
        t_life = median5(() -> FM.run_cuda_radix_lifecycle!(cache.state))
    end
    fmm!(sys, cache; scalar_potential=false, gradient=true)
    rel = sampled_rel_rms(sys, targets, gref)
    if structure == "adaptive"
        actx = cache.adaptive_tree
        cr = Array(actx.grid.cell_ranges)
        nl = actx.n_leaves
        dt = Array(cache.adaptive_state.direct_targets)[1:actx.n_u]
        ds = Array(cache.adaptive_state.direct_sources)[1:actx.n_u]
        n_u = actx.n_u
        n_nodes = actx.n_nodes; n_routes = actx.n_routes
        n_w = actx.n_w; n_x = actx.n_x
    else
        counts = cache.state.counts
        cr = Array(cache.state.grid.cell_ranges)
        nl = counts.n_cells
        dt = Array(cache.state.direct_targets)[1:counts.n_direct]
        ds = Array(cache.state.direct_sources)[1:counts.n_direct]
        n_u = counts.n_direct
        n_nodes = counts.n_nodes; n_routes = counts.n_routes
        n_w = 0; n_x = 0
    end
    pops = Int[cr[2, c] for c in 1:nl]
    popmax = maximum(pops)
    u_pairs = sum(Int(cr[2, dt[i]]) * Int(cr[2, ds[i]]) for i in 1:n_u)
    row = join(Any[name, n, TF, P, structure, param, "ok",
        round(t_cold; digits=3), round(mem1 - mem0; digits=3),
        round(t_update; digits=3), round(t_life; digits=3),
        round(t_step; digits=3), @sprintf("%.3e", rel),
        n_nodes, nl, popmax, n_u, u_pairs, n_routes, n_w, n_x], ",")
    @printf("%s cold=%.2fs mem=%.2fGB update=%.2fms life=%.2fms step=%.2fms rel=%.3e popmax=%d\n",
        key, t_cold, mem1 - mem0, t_update, t_life, t_step, rel, popmax)
    sys = nothing; cache = nothing; GC.gc(); CUDA.reclaim()
    return (row, want_pops ? pops : nothing)
end

function main()
    nlistA = length(ARGS) >= 1 ? parse.(Int, split(ARGS[1], ",")) :
        [10_000, 31_623, 100_000, 316_228, 1_000_000]
    rows = String[HEADER]
    prows = String[HEADER]
    lrows = String["case,n,structure,param,pop,count"]
    for n in nlistA
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

            # Section A: Float64 widened grid (anchors K64/128, ell5/6 included)
            b64 = Float64.(b)
            for K in (32, 64, 128, 256)
                want = (n == 1_000_000 && K == 64)
                row, pops = run_config(name, n, Float64, 4, "adaptive", K, b64,
                    targets, gref; want_pops=want)
                push!(rows, row); write_rows(rows, OUTFILE)
                if pops !== nothing
                    for (p, c) in sort!(collect(pairs(Dict{Int,Int}(
                            p => count(==(p), pops) for p in unique(pops)))))
                        push!(lrows, "$name,$n,adaptive,$K,$p,$c")
                    end
                    write_rows(lrows, LEAFFILE)
                end
            end
            for ell in (3, 4, 5, 6, 7)
                want = (n == 1_000_000 && (ell == 5 || ell == 6))
                row, pops = run_config(name, n, Float64, 4, "uniform", ell, b64,
                    targets, gref; want_pops=want)
                push!(rows, row); write_rows(rows, OUTFILE)
                if pops !== nothing
                    for (p, c) in sort!(collect(pairs(Dict{Int,Int}(
                            p => count(==(p), pops) for p in unique(pops)))))
                        push!(lrows, "$name,$n,uniform,$ell,$p,$c")
                    end
                    write_rows(lrows, LEAFFILE)
                end
            end

            # Section B: Float32 extension (new configs only) at fm041 n points
            if n in (100_000, 1_000_000)
                b32 = Float32.(b)
                for K in (32, 256)
                    row, _ = run_config(name, n, Float32, 4, "adaptive", K, b32,
                        targets, gref)
                    push!(rows, row); write_rows(rows, OUTFILE)
                end
                for ell in (3, 4, 7)
                    row, _ = run_config(name, n, Float32, 4, "uniform", ell, b32,
                        targets, gref)
                    push!(rows, row); write_rows(rows, OUTFILE)
                end
            end

            # Section C: P sweep for the accuracy-cost frontier
            if n == 1_000_000 && name in ("wake", "multiscale100")
                for P in (2, 3, 6, 8)
                    row, _ = run_config(name, n, Float64, P, "adaptive", 64, b64,
                        targets, gref)
                    push!(prows, row); write_rows(prows, PFILE)
                    for ell in (6, 7)
                        row, _ = run_config(name, n, Float64, P, "uniform", ell,
                            b64, targets, gref)
                        push!(prows, row); write_rows(prows, PFILE)
                    end
                end
            end
        end
    end
    write_rows(rows, OUTFILE)
    write_rows(prows, PFILE)
    write_rows(lrows, LEAFFILE)
    println("wrote ", OUTFILE)
end

main()
