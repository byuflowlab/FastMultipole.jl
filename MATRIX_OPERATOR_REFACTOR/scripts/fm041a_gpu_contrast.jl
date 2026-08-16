# fm041a_gpu_contrast.jl — task 041a pre-registered measurement (H200), part 3/3.
#
# Three sections, three CSVs (each incremental, fail rows recorded):
#
# S1 CONTRAST SWEEP — the task-file "time vs cluster contrast" figure: the
#    multiscale field at cluster density contrast c in {1,3,10,30,100,300,
#    1000} (same generator as fm039-fm041 multiscale100, same seed, c as the
#    contrast parameter), n = 1e6, Float64, dense strategy, P = 4, q = 5:
#    adaptive K=64 vs uniform ell in {4,5,6,7}. Every row records time,
#    device memory, popmax, counts, and sampled-direct accuracy; per-leaf
#    population histograms for every row go to fm041a_contrast_leafpop.csv.
#    -> data/fm041a_gpu_contrast.csv, data/fm041a_contrast_leafpop.csv
#
# S2 LIFECYCLE STAGE BREAKDOWN — the task-file per-stage figure (no lifecycle
#    stage profiler exists; per the 028/029 precedent each stage launcher is
#    timed directly with CUDA.@elapsed medians of 7 on a warmed state, with
#    CUDA_OVERLAP_NEARFIELD[] = false and CUDA_GRAPH_LIFECYCLE[] = false so
#    stages are serialized and not graph-replayed):
#      adaptive: b2m, m2m, m2l(V), s2l(X), l2l, near(U), l2b, m2t(W)
#      uniform:  b2m, m2m, m2l,        l2l, near,    l2b
#    plus, per config: whole-lifecycle time at (graph on, overlap on) /
#    (off, on) / (off, off) — the graph-engagement measurement priced by 041 —
#    and warm t_update (epoch fast path = the frozen-leaf-set refresh cost)
#    vs full-rebuild refresh (epoch dropped).
#    Configs: fm041 cases x n=1e6, Float64, dense, {adaptive K=64, uniform
#    ell in {5,6,7}}. -> data/fm041a_gpu_stages.csv
#
# S3 SIGMA-HETEROGENEOUS VARIANT — the task-file "global-sigma_max-gate
#    weakness" case (038 shipped the per-cell gate, so the conditional row is
#    active): SmoothedVortex (test/interface_test_systems.jl; RegularizedVortex
#    sigma_row=8, Lamb-Helmholtz), n = 1e5, Float64, concat strategy (the
#    041-test-covered LH engine), unit-cube positions (generate_vortex seed
#    40301), per-body sigma log-uniform in [1e-4, spread*1e-4] for
#    spread in {1, 10, 100, 300}:
#      - adaptive: K=64, ell_max=10, per-cell gate armed
#        (rho_t = RegularizedVortex().rho_t, sigma_row=8), u_capacity doubled
#        (sticky demotion inflates U; overflow would be a loud fail row);
#      - uniform: ell in {2,3,4,5,6}; rows above the global-gate admissible
#        depth THROW ArgumentError — recorded as fail rows: that throw IS the
#        measured failure mechanism of the global gate, and the shallowest
#        passing depth's cost is the forced-shallow cost.
#    Accuracy: velocity rel RMS at 2000 sampled targets vs an exact Float64
#    regularized (gaussianerf) direct sum over all sources (the
#    _interface_regularized_direct formula, sampled); velocity read from
#    base.gradient_stretching[1:3, :]. -> data/fm041a_gpu_sigma.csv
#
# Same-job anchors: every comparison in each section is within-section,
# one process, one H200. Warm medians of 5 (sections S1/S3) after 2 warm-ups;
# stage medians of 7 (S2). 1e-3 velocity gate reported per row, asserted in
# analysis only.
#
# Usage: julia --project=<env> -t 1 fm041a_gpu_contrast.jl

using FastMultipole
using FastMultipole.StaticArrays
using CUDA
using Random
using Printf
using Statistics

const FM = FastMultipole
const CONTRASTFILE = joinpath(@__DIR__, "..", "data", "fm041a_gpu_contrast.csv")
const CLEAFFILE = joinpath(@__DIR__, "..", "data", "fm041a_contrast_leafpop.csv")
const STAGEFILE = joinpath(@__DIR__, "..", "data", "fm041a_gpu_stages.csv")
const SIGMAFILE = joinpath(@__DIR__, "..", "data", "fm041a_gpu_sigma.csv")
const REPO = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(REPO, "test", "gravitational.jl"))
include(joinpath(REPO, "test", "vortex.jl"))
include(joinpath(REPO, "test", "interface_test_systems.jl"))

FastMultipole.load_cuda_radix_lifecycle!() ||
    error("CUDA radix lifecycle failed to load: " * FastMultipole.cuda_radix_status())

#--- field generators (identical to fm039-fm041) ---#

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

mkdense(TF) = FM.CUDARadixLifecycleOptions(precision=TF,
    m2l_strategy=FM.DenseTranslationM2L(apply_chunk=64, build_chunk=8))

adaptive_policy(n, K; ufac=60, kw...) = begin
    node_cap = 8 * cld(n, K) + 1024
    AdaptiveTreePolicy(K_max=K, ell_max=10, near_radius2=5,
        node_capacity=node_cap, u_capacity=ufac * node_cap,
        v_capacity=250 * node_cap, wx_capacity=60 * node_cap; kw...)
end

pop_histogram(pops) = sort!(collect(pairs(Dict{Int,Int}(
    p => count(==(p), pops) for p in unique(pops)))))

#======================= S1: contrast sweep =======================#

function section_contrast()
    n = 1_000_000
    P = 4
    rows = String["contrast,structure,param,status,t_cold_s,mem_gb,t_update_ms," *
        "t_lifecycle_ms,t_step_ms,vel_rel_rms,n_nodes,n_leaves,popmax,n_u," *
        "u_pairs,v_routes,w_entries,x_entries"]
    lrows = String["contrast,structure,param,pop,count"]
    for contrast in (1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)
        x = make_multiscale(n; contrast=contrast)
        b = make_bodies(x)
        rngs = MersenneTwister(40201)
        targets = sort!(Random.shuffle(rngs, collect(1:n))[1:2000])
        gref = sampled_direct(b, targets)
        configs = [("adaptive", 64), ("uniform", 4), ("uniform", 5),
                   ("uniform", 6), ("uniform", 7)]
        for (structure, param) in configs
            key = "$contrast,$structure,$param"
            local sys, cache
            mem0 = _used_gb()
            t0 = time_ns()
            try
                sys = Gravitational(copy(b))
                if structure == "adaptive"
                    cache = FM.RadixFMMCache(sys; expansion_order=P, ell=5,
                        adaptive=adaptive_policy(n, param), options=mkdense(Float64),
                        device=true)
                else
                    cache = FM.RadixFMMCache(sys; expansion_order=P, ell=param,
                        options=mkdense(Float64), device=true)
                end
                CUDA.synchronize()
            catch err
                emsg = replace(first(sprint(showerror, err), 120), "," => ";", "\n" => " ")
                push!(rows, "$key,fail:$(typeof(err)) $emsg," * join(fill("", 13), ","))
                @printf("%s FAILED %s %s\n", key, typeof(err), emsg)
                write_rows(rows, CONTRASTFILE)
                CUDA.reclaim(); GC.gc()
                continue
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
                actx = cache.adaptive_tree
                cr = Array(actx.grid.cell_ranges)
                nl = actx.n_leaves
                dt = Array(cache.adaptive_state.direct_targets)[1:actx.n_u]
                ds = Array(cache.adaptive_state.direct_sources)[1:actx.n_u]
                n_u = actx.n_u; n_nodes = actx.n_nodes
                n_routes = actx.n_routes; n_w = actx.n_w; n_x = actx.n_x
            else
                t_life = median5(() -> FM.run_cuda_radix_lifecycle!(cache.state))
                counts = cache.state.counts
                cr = Array(cache.state.grid.cell_ranges)
                nl = counts.n_cells
                dt = Array(cache.state.direct_targets)[1:counts.n_direct]
                ds = Array(cache.state.direct_sources)[1:counts.n_direct]
                n_u = counts.n_direct; n_nodes = counts.n_nodes
                n_routes = counts.n_routes; n_w = 0; n_x = 0
            end
            fmm!(sys, cache; scalar_potential=false, gradient=true)
            rel = sampled_rel_rms(sys, targets, gref)
            pops = Int[cr[2, c] for c in 1:nl]
            popmax = maximum(pops)
            u_pairs = sum(Int(cr[2, dt[i]]) * Int(cr[2, ds[i]]) for i in 1:n_u)
            push!(rows, join(Any[contrast, structure, param, "ok",
                round(t_cold; digits=3), round(mem1 - mem0; digits=3),
                round(t_update; digits=3), round(t_life; digits=3),
                round(t_step; digits=3), @sprintf("%.3e", rel),
                n_nodes, nl, popmax, n_u, u_pairs, n_routes, n_w, n_x], ","))
            for (p, c) in pop_histogram(pops)
                push!(lrows, "$contrast,$structure,$param,$p,$c")
            end
            @printf("%s cold=%.2fs mem=%.2fGB step=%.2fms rel=%.3e popmax=%d\n",
                key, t_cold, mem1 - mem0, t_step, rel, popmax)
            write_rows(rows, CONTRASTFILE)
            write_rows(lrows, CLEAFFILE)
            sys = nothing; cache = nothing; GC.gc(); CUDA.reclaim()
        end
    end
end

#======================= S2: stage breakdown =======================#

median7(f) = median([(CUDA.synchronize(); Float64(CUDA.@elapsed f()) * 1e3) for _ in 1:7])

function section_stages()
    n = 1_000_000
    P = 4
    rows = String["case,structure,param,status,t_b2m_ms,t_m2m_ms,t_m2l_ms," *
        "t_s2l_ms,t_l2l_ms,t_near_ms,t_l2b_ms,t_m2t_ms,t_stagesum_ms," *
        "t_life_graph_overlap_ms,t_life_nograph_overlap_ms,t_life_nograph_serial_ms," *
        "t_update_warm_ms,t_update_rebuild_ms"]
    cases = [
        ("unitcube", make_unitcube(n)),
        ("wake", make_wake(n)),
        ("multiscale100", make_multiscale(n)),
    ]
    for (name, x) in cases
        b = make_bodies(x)
        configs = [("adaptive", 64), ("uniform", 5), ("uniform", 6), ("uniform", 7)]
        for (structure, param) in configs
            key = "$name,$structure,$param"
            local sys, cache
            try
                sys = Gravitational(copy(b))
                if structure == "adaptive"
                    cache = FM.RadixFMMCache(sys; expansion_order=P, ell=5,
                        adaptive=adaptive_policy(n, param), options=mkdense(Float64),
                        device=true)
                else
                    cache = FM.RadixFMMCache(sys; expansion_order=P, ell=param,
                        options=mkdense(Float64), device=true)
                end
                CUDA.synchronize()
                fmm!(sys, cache; scalar_potential=false, gradient=true)
                fmm!(sys, cache; scalar_potential=false, gradient=true)
            catch err
                emsg = replace(first(sprint(showerror, err), 120), "," => ";", "\n" => " ")
                push!(rows, "$key,fail:$(typeof(err)) $emsg," * join(fill("", 13), ","))
                @printf("%s FAILED %s %s\n", key, typeof(err), emsg)
                write_rows(rows, STAGEFILE)
                CUDA.reclaim(); GC.gc()
                continue
            end
            # whole-lifecycle timings: shipped (graph + overlap), graph off,
            # graph and overlap off (serialized)
            if structure == "adaptive"
                lifef = () -> FM.run_cuda_adaptive_radix_lifecycle!(
                    cache.adaptive_state, cache.adaptive_tree)
            else
                lifef = () -> FM.run_cuda_radix_lifecycle!(cache.state)
            end
            t_go = median5(lifef)
            FM.CUDA_GRAPH_LIFECYCLE[] = false
            lifef(); t_no = median5(lifef)
            FM.CUDA_OVERLAP_NEARFIELD[] = false
            lifef(); t_ns = median5(lifef)
            # per-stage timings (serialized, no graph)
            state = structure == "adaptive" ? cache.adaptive_state : cache.state
            if structure == "adaptive"
                actx = cache.adaptive_tree
                stages = [
                    ("b2m", () -> FM._launch_cuda_b2m!(state)),
                    ("m2m", () -> FM._launch_adaptive_resident_m2m!(state)),
                    ("m2l", () -> FM._launch_cuda_adaptive_m2l!(state, actx)),
                    ("s2l", () -> FM._launch_cuda_adaptive_s2l!(state, actx)),
                    ("l2l", () -> FM._launch_resident_l2l!(state)),
                    ("near", () -> FM._launch_cuda_nearfield_kernel!(state)),
                    ("l2b", () -> FM._launch_cuda_resident_l2b_only!(state, nothing)),
                    ("m2t", () -> FM._launch_cuda_adaptive_m2t!(state, actx)),
                ]
            else
                stages = [
                    ("b2m", () -> FM._launch_cuda_b2m!(state)),
                    ("m2m", () -> FM._launch_cuda_resident_m2m!(state)),
                    ("m2l", () -> FM._launch_cuda_resident_m2l!(state)),
                    ("s2l", () -> nothing),
                    ("l2l", () -> FM._launch_resident_l2l!(state)),
                    ("near", () -> FM._launch_cuda_nearfield_kernel!(state)),
                    ("l2b", () -> FM._launch_cuda_resident_l2b_only!(state, nothing)),
                    ("m2t", () -> nothing),
                ]
            end
            ts = Float64[]
            for (sname, f) in stages
                f(); CUDA.synchronize()   # stage warm-up
                push!(ts, median7(f))
            end
            # refresh: warm (epoch fast path) vs forced full rebuild
            t_upd_warm = median5(() -> FM.update_cuda_radix_state!(cache, (sys,)))
            t_upd_rebuild = t_upd_warm
            if structure == "adaptive"
                actx = cache.adaptive_tree
                t_upd_rebuild = median5(() -> begin
                    actx.epoch_have = false
                    FM.update_cuda_radix_state!(cache, (sys,))
                end)
            end
            FM.CUDA_OVERLAP_NEARFIELD[] = true
            FM.CUDA_GRAPH_LIFECYCLE[] = true
            push!(rows, join(Any[name, structure, param, "ok",
                [round(t; digits=3) for t in ts]...,
                round(sum(ts); digits=3),
                round(t_go; digits=3), round(t_no; digits=3),
                round(t_ns; digits=3), round(t_upd_warm; digits=3),
                round(t_upd_rebuild; digits=3)], ","))
            @printf("%s stagesum=%.2fms life(g+o)=%.2f life(serial)=%.2f\n",
                key, sum(ts), t_go, t_ns)
            write_rows(rows, STAGEFILE)
            sys = nothing; cache = nothing; GC.gc(); CUDA.reclaim()
        end
    end
end

#======================= S3: sigma-heterogeneous =======================#

# sampled regularized (gaussianerf) direct velocity reference — the
# _interface_regularized_direct U formula restricted to sampled targets.
function sampled_regularized_direct(ssys::SmoothedVortex, targets::Vector{Int})
    n = FastMultipole.get_n_bodies(ssys)
    U = zeros(3, length(targets))
    A = sqrt(2 / pi)
    for (k, t) in enumerate(targets)
        xt = FastMultipole.get_position(ssys, t)
        for j in 1:n
            j == t && continue
            d = xt - FastMultipole.get_position(ssys, j)
            G = ssys.inner.bodies[j].strength
            sigma = Float64(ssys.sigma[j])
            r2 = d[1]^2 + d[2]^2 + d[3]^2
            r2 == 0 && continue
            r = sqrt(r2)
            rho = r / sigma
            g = _ref_erf(rho / sqrt2) - A * rho * exp(-rho^2 / 2)
            cr3 = 1 / (4pi * r2 * r)
            U[1, k] += g * (d[3] * G[2] - d[2] * G[3]) * cr3
            U[2, k] += g * (d[1] * G[3] - d[3] * G[1]) * cr3
            U[3, k] += g * (d[2] * G[1] - d[1] * G[2]) * cr3
        end
    end
    return U
end

function vortex_rel_rms(base, targets, Uref)
    err = 0.0; nrm = 0.0
    for (k, t) in enumerate(targets)
        for a in 1:3
            d = Float64(base.gradient_stretching[a, t]) - Uref[a, k]
            err += d * d
            nrm += Uref[a, k]^2
        end
    end
    return sqrt(err / nrm)
end

function section_sigma()
    n = 100_000
    P = 4
    rows = String["spread,structure,param,status,t_cold_s,mem_gb,t_update_ms," *
        "t_lifecycle_ms,t_step_ms,vel_rel_rms,n_leaves,popmax,u_pairs,v_routes," *
        "w_entries,x_entries,sigma_max"]
    dk = FM.RegularizedVortex(; sigma_row=8)
    opts = FM.CUDARadixLifecycleOptions(precision=Float64,
        m2l_strategy=FM.ConcatenatedFixedZM2L(),
        body_type=FM.Point{FM.Vortex}, direct_kernel=dk)
    for spread in (1.0, 10.0, 100.0, 300.0)
        rng = MersenneTwister(40302)
        sigma_min = 1e-4
        sigma = sigma_min .* exp.(log(spread) .* rand(rng, n))
        sigma_max = maximum(sigma)
        base0 = generate_vortex(40301, n)
        ssys0 = SmoothedVortex(base0, copy(sigma))
        rngs = MersenneTwister(40201)
        targets = sort!(Random.shuffle(rngs, collect(1:n))[1:2000])
        Uref = sampled_regularized_direct(ssys0, targets)
        configs = [("adaptive", 64), ("uniform", 2), ("uniform", 3),
                   ("uniform", 4), ("uniform", 5), ("uniform", 6)]
        for (structure, param) in configs
            key = "$spread,$structure,$param"
            local base, ssys, cache
            mem0 = _used_gb()
            t0 = time_ns()
            try
                base = generate_vortex(40301, n)
                ssys = SmoothedVortex(base, copy(sigma))
                if structure == "adaptive"
                    pol = adaptive_policy(n, param; ufac=120, rho_t=dk.rho_t,
                        sigma_row=8)
                    cache = FM.RadixFMMCache(ssys; expansion_order=P, ell=5,
                        adaptive=pol, options=opts, lamb_helmholtz=true,
                        device=true)
                else
                    cache = FM.RadixFMMCache(ssys; expansion_order=P, ell=param,
                        options=opts, lamb_helmholtz=true, device=true)
                end
                CUDA.synchronize()
            catch err
                emsg = replace(first(sprint(showerror, err), 140), "," => ";", "\n" => " ")
                push!(rows, "$key,fail:$(typeof(err)) $emsg," *
                    join(fill("", 11), ",") * "," * @sprintf("%.3e", sigma_max))
                @printf("%s FAILED %s %s\n", key, typeof(err), emsg)
                write_rows(rows, SIGMAFILE)
                CUDA.reclaim(); GC.gc()
                continue
            end
            t_cold = (time_ns() - t0) / 1e9
            mem1 = _used_gb()
            fmm!(ssys, cache; scalar_potential=false, gradient=true)
            fmm!(ssys, cache; scalar_potential=false, gradient=true)
            t_step = median5(() -> fmm!(ssys, cache; scalar_potential=false, gradient=true))
            t_update = median5(() -> FM.update_cuda_radix_state!(cache, (ssys,)))
            if structure == "adaptive"
                t_life = median5(() -> FM.run_cuda_adaptive_radix_lifecycle!(
                    cache.adaptive_state, cache.adaptive_tree))
                actx = cache.adaptive_tree
                cr = Array(actx.grid.cell_ranges)
                nl = actx.n_leaves
                dt = Array(cache.adaptive_state.direct_targets)[1:actx.n_u]
                ds = Array(cache.adaptive_state.direct_sources)[1:actx.n_u]
                n_u = actx.n_u
                n_routes = actx.n_routes; n_w = actx.n_w; n_x = actx.n_x
            else
                t_life = median5(() -> FM.run_cuda_radix_lifecycle!(cache.state))
                counts = cache.state.counts
                cr = Array(cache.state.grid.cell_ranges)
                nl = counts.n_cells
                dt = Array(cache.state.direct_targets)[1:counts.n_direct]
                ds = Array(cache.state.direct_sources)[1:counts.n_direct]
                n_u = counts.n_direct
                n_routes = counts.n_routes; n_w = 0; n_x = 0
            end
            fmm!(ssys, cache; scalar_potential=false, gradient=true)
            rel = vortex_rel_rms(base, targets, Uref)
            pops = Int[cr[2, c] for c in 1:nl]
            popmax = maximum(pops)
            u_pairs = sum(Int(cr[2, dt[i]]) * Int(cr[2, ds[i]]) for i in 1:n_u)
            push!(rows, join(Any[spread, structure, param, "ok",
                round(t_cold; digits=3), round(mem1 - mem0; digits=3),
                round(t_update; digits=3), round(t_life; digits=3),
                round(t_step; digits=3), @sprintf("%.3e", rel),
                nl, popmax, u_pairs, n_routes, n_w, n_x,
                @sprintf("%.3e", sigma_max)], ","))
            @printf("%s cold=%.2fs step=%.2fms rel=%.3e popmax=%d\n",
                key, t_cold, t_step, rel, popmax)
            write_rows(rows, SIGMAFILE)
            base = nothing; ssys = nothing; cache = nothing
            GC.gc(); CUDA.reclaim()
        end
    end
end

function main()
    println("=== S1 contrast sweep ===")
    section_contrast()
    println("=== S2 stage breakdown ===")
    section_stages()
    println("=== S3 sigma-heterogeneous ===")
    section_sigma()
    println("done")
end

main()
