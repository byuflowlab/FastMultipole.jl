# fm041e_target_owned_bench.jl — task 041e Stage B pre-registered H200
# microbenchmark: target-owned fused CUDA nearfield shapes vs the shipped
# :pairs kernel. Cluster driver (H200), not a local script.
#
# REGISTRATION (predeclared; the Stage B analysis checksums the outputs):
#   cases: cube / wake (041a deterministic Gravitational constructors, seeds
#     39101/39102), sigma_multiscale (041a multiscale positions seed 39103 as
#     a Lamb-Helmholtz SmoothedVortex with log-uniform per-body sigma in
#     [1e-4, 1e-2], strength seed 41503, sigma seed 41504, RegularizedVortex
#     sigma_row=8 with the armed per-cell gate), and the DJI-9443 rotor
#     (benchmark_033_common.jl fm033_rotor_foreach, jittered positions,
#     rng = MersenneTwister(FM033_SEED + FM033_ROTOR_SEED_OFFSET + n) — the
#     fm033_build_rotor convention), skipped when FLOWVPM is unavailable.
#   n in (1e5, 1e6); adaptive K_max winners (041a): cube 64, wake 256,
#     sigma_multiscale 64, rotor 64; near_radius2 = 5, ell_max = 10 (the 041a
#     adaptive_policy); precision Float32/Float64; P = 4 everywhere plus the
#     P = 8 contract rows (cube, n = 1e5, Float64, all shapes).
#   shapes: :pairs (shipped control), :fused_cta, :fused_srclanes, :fused_packed. The shape
#     Ref is read at cache construction AND inside the lifecycle body, so it
#     is set BEFORE construction and restored in a finally block per row.
#   strategies: dense fused M2L (041a S2 convention) for the Gravitational
#     cases; concatenated fixed-z (the 041-test-covered LH engine, 041a S3
#     convention) for the vortex cases.
#   timing: warm 3 steps, then (041a conventions) median-of-5 synchronized
#     wall times for the graph-overlapped lifecycle, the serialized lifecycle
#     (graph + overlap off), and the refresh; median-of-7 CUDA.@elapsed for
#     the ISOLATED nearfield stage launcher with CUDA_GRAPH_LIFECYCLE[] and
#     CUDA_OVERLAP_NEARFIELD[] forced off (041a S2 stage convention).
#   contracts: route_uploads / operator_uploads constant across a warmed step
#     and expansion_host_copies == 0, recorded per row.
#   accuracy: velocity rel RMS at 200 seeded sampled targets (seed 41201)
#     against an exact Float64 direct sum over all bodies (singular 1/r^2 for
#     the Gravitational cases; gaussianerf regularized for the vortex cases —
#     the fm041a sampled-direct utilities).
#
# Output (append-safe, one row per config, job column from SLURM_JOB_ID):
#   MATRIX_OPERATOR_REFACTOR/data/target_owned_nearfield/stageB_bench.csv
#   MATRIX_OPERATOR_REFACTOR/data/target_owned_nearfield/stageB_manifest.csv
#
# Usage: julia --project=<env> -t 1 fm041e_target_owned_bench.jl

using Random
using Printf
using Statistics

# ---- CUDA guard (clear exit, no stacktrace spam on CPU nodes) --------------
try
    @eval using CUDA
    CUDA.functional() || error("CUDA loaded but not functional")
catch err
    println("fm041e Stage B bench requires a functional CUDA GPU: ", sprint(showerror, err))
    exit(1)
end

using FastMultipole
using FastMultipole.StaticArrays

const FM = FastMultipole
const REPO = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(REPO, "test", "gravitational.jl"))
include(joinpath(REPO, "test", "vortex.jl"))
include(joinpath(REPO, "test", "interface_test_systems.jl"))

FastMultipole.load_cuda_radix_lifecycle!() ||
    (println("CUDA radix lifecycle failed to load: " * FastMultipole.cuda_radix_status());
     exit(1))

# rotor case needs the FLOWVPM project on the load path (Stage A convention)
const HAVE_ROTOR = Ref(false)
try
    include(joinpath(@__DIR__, "benchmark_033_common.jl"))
    HAVE_ROTOR[] = true
catch err
    @warn "FLOWVPM project unavailable; rotor case skipped" err
end

const OUTDIR = joinpath(@__DIR__, "..", "data", "target_owned_nearfield")
const CSVFILE = joinpath(OUTDIR, "stageB_bench.csv")
const MANIFEST = joinpath(OUTDIR, "stageB_manifest.csv")
mkpath(OUTDIR)

const JOB = get(ENV, "SLURM_JOB_ID", "local")
const N_TARGETS = 200
const SEED_TARGETS = 41201
const SIGMA_SEED = 41504
const STR_SEED = 41503
const SHAPES = (:pairs, :fused_cta, :fused_srclanes, :fused_packed)
const KWIN = Dict("cube" => 64, "wake" => 256, "sigma_multiscale" => 64,
    "rotor" => 64)

const HEADER = "job,case,n,K,precision,P,shape,status,fused_engaged," *
    "t_cold_s,mem_gb,t_near_ms,t_life_ms,t_life_serial_ms,t_update_ms," *
    "vel_rel_rms,n_leaves,n_u,route_uploads,operator_uploads," *
    "expansion_host_copies,route_flat,operator_flat"

function append_row(row::String)
    fresh = !isfile(CSVFILE)
    open(CSVFILE, "a") do io
        fresh && println(io, HEADER)
        println(io, row)
    end
    flush(stdout)
    return nothing
end

#--- field generators (identical to fm041a_gpu_contrast.jl / fm039-fm041) ---#

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

#--- sampled-direct references (fm041a conventions) ---#

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

function sampled_rel_rms(sys::Gravitational, targets, gref)
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

# sampled regularized (gaussianerf) direct velocity — the fm041a S3 utility
# (the _interface_regularized_direct U formula restricted to sampled targets)
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

function vortex_rel_rms(base::VortexParticles, targets, Uref)
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

#--- timing helpers (fm041a conventions) ---#

median5(f) = median([(CUDA.synchronize(); t0 = time_ns(); f(); CUDA.synchronize();
    (time_ns() - t0) / 1e6) for _ in 1:5])

median7(f) = median([(CUDA.synchronize(); Float64(CUDA.@elapsed f()) * 1e3) for _ in 1:7])

_used_gb() = (CUDA.total_memory() - CUDA.free_memory()) / 2^30

#--- cache constructors (041a conventions) ---#

mkdense(TF) = FM.CUDARadixLifecycleOptions(precision=TF,
    m2l_strategy=FM.DenseTranslationM2L(apply_chunk=64, build_chunk=8))

adaptive_policy(n, K; ufac=60, vfac=250, kw...) = begin
    node_cap = 8 * cld(n, K) + 1024
    AdaptiveTreePolicy(K_max=K, ell_max=10, near_radius2=5,
        node_capacity=node_cap, u_capacity=ufac * node_cap,
        v_capacity=vfac * node_cap, wx_capacity=60 * node_cap; kw...)
end

#--- case data (built once per (case, n), reused across rows) ---#

struct GravCase
    b::Matrix{Float64}
end

struct VortexCase
    pos::Matrix{Float64}
    str::Matrix{Float64}
    sigma::Vector{Float64}
end

function build_case_data(cname::String, n::Int)
    if cname == "cube"
        return GravCase(make_bodies(make_unitcube(n)))
    elseif cname == "wake"
        return GravCase(make_bodies(make_wake(n)))
    elseif cname == "sigma_multiscale"
        pos = make_multiscale(n)
        str = randn(MersenneTwister(STR_SEED), 3, n) ./ n
        rng = MersenneTwister(SIGMA_SEED)
        sigma = 1e-4 .* exp.(log(100.0) .* rand(rng, n))   # log-uniform [1e-4, 1e-2]
        return VortexCase(pos, str, sigma)
    elseif cname == "rotor"
        # prefer the pre-extracted deterministic snapshot (no FLOWVPM needed
        # on the GPU node); fall back to the fm033 generator when present
        snap = joinpath(@__DIR__, "..", "data", "rotor_wake",
            "rotor_snapshot_n$(n).bin")
        if isfile(snap)
            X = Matrix{Float64}(undef, 3, n)
            G = Matrix{Float64}(undef, 3, n)
            s = Vector{Float64}(undef, n)
            open(snap, "r") do io
                nn = read(io, Int64)
                nn == n || error("rotor snapshot $snap holds n=$nn, wanted $n")
                read!(io, X); read!(io, G); read!(io, s)
            end
            return VortexCase(X, G, s)
        end
        HAVE_ROTOR[] || error("rotor requires the FLOWVPM project or a " *
            "pre-extracted data/rotor_wake/rotor_snapshot_n$(n).bin")
        X = Matrix{Float64}(undef, 3, n)
        G = Matrix{Float64}(undef, 3, n)
        s = Vector{Float64}(undef, n)
        rng = MersenneTwister(FM033_SEED + FM033_ROTOR_SEED_OFFSET + n)
        i = Ref(0)
        emitted = fm033_rotor_foreach(n; rng) do x, gamma, sig
            i[] += 1
            X[1, i[]] = x[1]; X[2, i[]] = x[2]; X[3, i[]] = x[3]
            G[1, i[]] = gamma[1]; G[2, i[]] = gamma[2]; G[3, i[]] = gamma[3]
            s[i[]] = sig
        end
        (emitted == n && i[] == n) || error("rotor emitted $(i[]) != $n particles")
        return VortexCase(X, G, s)
    end
    error("unknown case $cname")
end

make_system(cd::GravCase) = Gravitational(copy(cd.b))
make_system(cd::VortexCase) = SmoothedVortex(
    VortexParticles(copy(cd.pos), copy(cd.str)), copy(cd.sigma))

function make_cache(cd::GravCase, sys, n, K, TF, P)
    return FM.RadixFMMCache(sys; expansion_order=P, ell=5,
        adaptive=adaptive_policy(n, K), options=mkdense(TF), device=true)
end

function make_cache(cd::VortexCase, sys, n, K, TF, P)
    dk = FM.RegularizedVortex(; sigma_row=8)
    opts = FM.CUDARadixLifecycleOptions(precision=TF,
        m2l_strategy=FM.ConcatenatedFixedZM2L(),
        body_type=FM.Point{FM.Vortex}, direct_kernel=dk)
    # vfac=500: the sigma_multiscale log-uniform proxy overflows the DTR
    # frontier (which tracks v_capacity) at the 250x default (job 13193465)
    pol = adaptive_policy(n, K; ufac=120, vfac=500, rho_t=dk.rho_t, sigma_row=8)
    return FM.RadixFMMCache(sys; expansion_order=P, ell=5, adaptive=pol,
        options=opts, lamb_helmholtz=true, device=true)
end

function sampled_reference(cd::GravCase, targets)
    return sampled_direct(cd.b, targets)
end

function sampled_reference(cd::VortexCase, targets)
    return sampled_regularized_direct(make_system(cd), targets)
end

accuracy(cd::GravCase, sys, targets, ref) = sampled_rel_rms(sys, targets, ref)
accuracy(cd::VortexCase, sys, targets, ref) =
    vortex_rel_rms(sys.inner, targets, ref)

#--- one measured row ---#

function run_row!(cd, cname, n, K, TF, P, shape, targets, ref)
    key = "$JOB,$cname,$n,$K,$TF,$P,$shape"
    old_shape = FM.CUDA_NEARFIELD_SHAPE[]
    old_min = FM.CUDA_NEARFIELD_FUSED_MIN_BODIES[]
    FM.CUDA_NEARFIELD_FUSED_MIN_BODIES[] = 0   # measure the envelope, don't bake it
    old_graph = FM.CUDA_GRAPH_LIFECYCLE[]
    old_overlap = FM.CUDA_OVERLAP_NEARFIELD[]
    FM.CUDA_NEARFIELD_SHAPE[] = shape          # BEFORE construction (graph-baked)
    local sys, cache
    try
        mem0 = _used_gb()
        t0 = time_ns()
        sys = make_system(cd)
        cache = make_cache(cd, sys, n, K, TF, P)
        CUDA.synchronize()
        t_cold = (time_ns() - t0) / 1e9
        mem1 = _used_gb()

        stepf = () -> fmm!(sys, cache; scalar_potential=false, gradient=true)
        # warm 3 steps (step 1 warms, 2 records, 3 replays the captured graph)
        for _ in 1:3
            stepf()
        end
        actx = cache.adaptive_tree
        state = cache.adaptive_state
        fused_engaged = shape !== :pairs && length(actx.u_csr_sources) > 0 &&
            actx.u_csr_built_epoch == actx.epoch_id

        # 023 transfer-counter contract across one warmed step
        c = state.counters
        route0 = c.route_uploads
        op0 = c.operator_uploads
        stepf()
        route_flat = c.route_uploads == route0
        op_flat = c.operator_uploads == op0
        ehc = c.expansion_host_copies

        # graph-overlapped lifecycle + refresh (shipped defaults)
        lifef = () -> FM.run_cuda_adaptive_radix_lifecycle!(state, actx)
        t_life = median5(lifef)
        t_update = median5(() -> FM.update_cuda_radix_state!(cache, (sys,)))

        # isolated nearfield stage + serialized lifecycle (041a S2 convention)
        FM.CUDA_GRAPH_LIFECYCLE[] = false
        lifef()
        FM.CUDA_OVERLAP_NEARFIELD[] = false
        lifef()
        t_life_serial = median5(lifef)
        nearf = () -> FM._launch_cuda_nearfield_kernel!(state)
        nearf(); CUDA.synchronize()
        t_near = median7(nearf)
        FM.CUDA_GRAPH_LIFECYCLE[] = old_graph
        FM.CUDA_OVERLAP_NEARFIELD[] = old_overlap

        # sampled accuracy from a full warmed step
        stepf()
        rel = accuracy(cd, sys, targets, ref)

        append_row(join(Any[JOB, cname, n, K, TF, P, shape, "ok",
            fused_engaged, round(t_cold; digits=3),
            round(mem1 - mem0; digits=3), round(t_near; digits=4),
            round(t_life; digits=3), round(t_life_serial; digits=3),
            round(t_update; digits=3), @sprintf("%.3e", rel),
            actx.n_leaves, actx.n_u, c.route_uploads, c.operator_uploads,
            ehc, route_flat, op_flat], ","))
        @printf("%s near=%.3fms life=%.2fms serial=%.2fms upd=%.2fms rel=%.3e fused=%s\n",
            key, t_near, t_life, t_life_serial, t_update, rel, fused_engaged)
    catch err
        emsg = replace(first(sprint(showerror, err), 140), "," => ";", "\n" => " ")
        append_row("$key,fail:$(typeof(err)) $emsg," * join(fill("", 15), ","))
        @printf("%s FAILED %s %s\n", key, typeof(err), emsg)
    finally
        FM.CUDA_NEARFIELD_SHAPE[] = old_shape
        FM.CUDA_NEARFIELD_FUSED_MIN_BODIES[] = old_min
        FM.CUDA_GRAPH_LIFECYCLE[] = old_graph
        FM.CUDA_OVERLAP_NEARFIELD[] = old_overlap
        sys = nothing; cache = nothing
        GC.gc(); CUDA.reclaim()
    end
    return nothing
end

#--- manifest (registration constants) ---#

function write_manifest()
    isfile(MANIFEST) && return nothing   # append-safe: registered once
    open(MANIFEST, "w") do io
        println(io, "key,value")
        for (k, v) in (
                ("registration", "see script header (predeclared 2026-08-18)"),
                ("cases", "cube|wake|sigma_multiscale|rotor"),
                ("k_winners", "cube:64|wake:256|sigma_multiscale:64|rotor:64"),
                ("n_grid", "100000|1000000"),
                ("precisions", "Float32|Float64"),
                ("P", "4 (+ P=8 contract rows: cube n=1e5 Float64)"),
                ("shapes", join(String.(Symbol.(SHAPES)), "|")),
                ("near_radius2", "5"), ("ell_max", "10"),
                ("case_seeds", "cube:39101|wake:39102|multiscale:39103"),
                ("sigma_multiscale_str_seed", string(STR_SEED)),
                ("sigma_multiscale_sigma_seed",
                 "$SIGMA_SEED (log-uniform 1e-4..1e-2)"),
                ("rotor", HAVE_ROTOR[] ?
                 "fm033_rotor_foreach jittered (FM033 seeds)" : "unavailable"),
                ("targets", "$N_TARGETS seeded (seed $SEED_TARGETS)"),
                ("timing", "warm3; median5 sync wall (life/serial/update); " *
                 "median7 CUDA.@elapsed isolated nearfield (graph+overlap off)"),
                ("strategies", "grav:dense(64;8)|vortex:concat LH " *
                 "RegularizedVortex sigma_row=8"),
                ("job", JOB), ("julia", string(VERSION)),
                ("date", "2026-08-18"))
            println(io, "$k,$v")
        end
    end
    return nothing
end

#--- main ---#

function main()
    write_manifest()
    configs = Tuple{String,Int,Int,DataType,Int,Symbol}[]
    for (cname, K) in (("cube", 64), ("wake", 256),
            ("sigma_multiscale", 64), ("rotor", 64))
        cname == "rotor" && !HAVE_ROTOR[] &&
            !isfile(joinpath(@__DIR__, "..", "data", "rotor_wake",
                "rotor_snapshot_n100000.bin")) && continue
        for n in (100_000, 1_000_000), TF in (Float32, Float64), shape in SHAPES
            push!(configs, (cname, n, K, TF, 4, shape))
        end
    end
    for shape in SHAPES   # the P = 8 contract rows
        push!(configs, ("cube", 100_000, 64, Float64, 8, shape))
    end
    # Stage C crossover rows: locate the fused_packed win-envelope boundary
    # between the measured n=1e5 losses and n=1e6 wins
    for (cname, K) in (("cube", 64), ("wake", 256), ("rotor", 64)),
            TF in (Float32, Float64), shape in (:pairs, :fused_packed)
        cname == "rotor" && !HAVE_ROTOR[] &&
            !isfile(joinpath(@__DIR__, "..", "data", "rotor_wake",
                "rotor_snapshot_n316228.bin")) && continue
        push!(configs, (cname, 316_228, K, TF, 4, shape))
    end

    # case data + sampled references cached per (case, n)
    cases = Dict{Tuple{String,Int},Any}()
    refs = Dict{Tuple{String,Int},Tuple{Vector{Int},Matrix{Float64}}}()
    for (cname, n, K, TF, P, shape) in configs
        ck = (cname, n)
        if !haskey(cases, ck)
            cases[ck] = build_case_data(cname, n)
            rng = MersenneTwister(SEED_TARGETS)
            targets = sort!(Random.shuffle(rng, collect(1:n))[1:N_TARGETS])
            t0 = time()
            refs[ck] = (targets, sampled_reference(cases[ck], targets))
            @printf("[%s n=%d] case + sampled reference built in %.1fs\n",
                cname, n, time() - t0)
        end
        targets, ref = refs[ck]
        run_row!(cases[ck], cname, n, K, TF, P, shape, targets, ref)
    end
    println("done -> $CSVFILE")
    return nothing
end

main()
