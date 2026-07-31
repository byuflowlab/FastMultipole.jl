# Task 023 production-integration benchmark: recurring-step cost of the radix
# fmm! path vs the legacy octree fmm! and direct!.
#
# CPU (default):
#   julia --project=. MATRIX_OPERATOR_REFACTOR/scripts/benchmark_023_integration.jl
# GPU (H200 host, after the CUDA lifecycle loads):
#   julia --project=. MATRIX_OPERATOR_REFACTOR/scripts/benchmark_023_integration.jl --gpu
#
# Success criterion (plan): on H200 at n=1e5 / ell=4 / P=4 the per-step total is
# within ~1.2x of the 0.116 s task-019 evaluation time, i.e. the one-shot
# interaction-list build (~0.35 s) and state construction (~0.62 s) are gone from
# the loop.
#
# Results land in MATRIX_OPERATOR_REFACTOR/data/production_integration/
# (CSV + markdown summary), suffixed by hostname.

using FastMultipole
using FastMultipole.StaticArrays
using Random
using Printf
using Dates

const REPO = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(REPO, "test", "gravitational.jl"))

const RUN_GPU = "--gpu" in ARGS
const DATA_DIR = joinpath(REPO, "MATRIX_OPERATOR_REFACTOR", "data", "production_integration")
mkpath(DATA_DIR)
const HOST = replace(gethostname(), r"[^A-Za-z0-9.-]" => "-")

rows = NamedTuple[]

function record!(; kwargs...)
    push!(rows, (; kwargs...))
    return nothing
end

_time_best(f, reps) = minimum((f(); @elapsed f()) for _ in 1:reps)

function jitter!(sys::Gravitational, rng, scale, lo, hi)
    for i in eachindex(sys.bodies)
        b = sys.bodies[i]
        pos = clamp.(b.position .+ scale .* (rand(rng, SVector{3,Float64}) .- 0.5), lo, hi)
        sys.bodies[i] = Body(pos, b.radius, b.strength)
    end
end

function bench_cpu(n, ell, P; steps=5, direct_ok=n <= 20_000)
    println("\n== CPU n=$n ell=$ell P=$P ==")
    rng = MersenneTwister(2023)
    sys = generate_gravitational(n, n)
    bounds = (SVector(-0.1, -0.1, -0.1), 1.2)

    # radix path: first call (construction) vs recurring step
    t_construct = @elapsed cache = RadixFMMCache(sys; expansion_order=P, ell=ell,
        bounds=bounds)
    fmm!(sys, cache; scalar_potential=true, gradient=true)   # compile
    t_steps = Float64[]
    for _ in 1:steps
        jitter!(sys, rng, 0.02, -0.05, 1.05)
        push!(t_steps, @elapsed fmm!(sys, cache; scalar_potential=true, gradient=true))
    end
    t_radix_step = minimum(t_steps)
    @printf("  radix   construct %.4f s, per-step %.4f s (mean %.4f)\n",
        t_construct, t_radix_step, sum(t_steps) / length(t_steps))
    record!(; backend="cpu", path="radix", n, ell, P, phase="construct", seconds=t_construct)
    record!(; backend="cpu", path="radix", n, ell, P, phase="step", seconds=t_radix_step)

    # legacy octree fmm! (fresh trees each call, as in production time stepping)
    legacy = generate_gravitational(n, n)
    fmm!(legacy; expansion_order=P, scalar_potential=true, gradient=true)  # compile
    t_legacy = _time_best(() -> fmm!(legacy; expansion_order=P,
        scalar_potential=true, gradient=true), 3)
    @printf("  legacy  per-call %.4f s\n", t_legacy)
    record!(; backend="cpu", path="legacy", n, ell, P, phase="step", seconds=t_legacy)

    if direct_ok
        dsys = generate_gravitational(n, n)
        FastMultipole.direct!(dsys; scalar_potential=true, gradient=true)  # compile
        t_direct = @elapsed FastMultipole.direct!(dsys; scalar_potential=true, gradient=true)
        @printf("  direct  per-call %.4f s\n", t_direct)
        record!(; backend="cpu", path="direct", n, ell, P, phase="step", seconds=t_direct)
    end
    return nothing
end

function bench_gpu(n, ell, P; steps=5)
    println("\n== GPU n=$n ell=$ell P=$P ==")
    @eval using CUDA
    rng = MersenneTwister(2023)
    sys = generate_gravitational(n, n)
    bounds = (SVector(-0.1, -0.1, -0.1), 1.2)
    t_construct = @elapsed cache = RadixFMMCache(sys; expansion_order=P, ell=ell,
        bounds=bounds, device=true)
    fmm!(sys, cache; scalar_potential=true, gradient=true)   # compile + warm
    record!(; backend="gpu", path="radix", n, ell, P, phase="construct", seconds=t_construct)

    # per-step decomposition
    t_update = Float64[]; t_life = Float64[]; t_final = Float64[]; t_total = Float64[]
    switches = DerivativesSwitch((true,), (true,), (false,), (sys,))
    for _ in 1:steps
        jitter!(sys, rng, 0.02, -0.05, 1.05)
        t0 = time()
        FastMultipole.update_cuda_radix_state!(cache, (sys,))
        Base.invokelatest(CUDA.synchronize)
        t1 = time()
        run_cuda_radix_lifecycle!(cache.state)
        Base.invokelatest(CUDA.synchronize)
        t2 = time()
        finalize_cuda_radix_output!(cache.state, (sys,); derivatives_switches=switches)
        Base.invokelatest(CUDA.synchronize)
        t3 = time()
        push!(t_update, t1 - t0)
        push!(t_life, t2 - t1)
        push!(t_final, t3 - t2)
        push!(t_total, t3 - t0)
    end
    @printf("  construct %.4f s\n", t_construct)
    @printf("  per-step: update %.4f s  lifecycle %.4f s  finalize %.4f s  total %.4f s\n",
        minimum(t_update), minimum(t_life), minimum(t_final), minimum(t_total))
    record!(; backend="gpu", path="radix", n, ell, P, phase="update", seconds=minimum(t_update))
    record!(; backend="gpu", path="radix", n, ell, P, phase="lifecycle", seconds=minimum(t_life))
    record!(; backend="gpu", path="radix", n, ell, P, phase="finalize", seconds=minimum(t_final))
    record!(; backend="gpu", path="radix", n, ell, P, phase="step", seconds=minimum(t_total))
    c = cache.state.counters
    println("  counters: body_uploads=$(c.body_uploads) route_uploads=$(c.route_uploads) " *
        "operator_uploads=$(c.operator_uploads) influence_downloads=$(c.influence_downloads)")
    return nothing
end

# --- CPU runs (019 tuned points) ---
bench_cpu(10_000, 4, 4)
bench_cpu(100_000, 4, 4; direct_ok=false)

# --- GPU runs ---
if RUN_GPU
    if FastMultipole.load_cuda_radix_lifecycle!()
        Base.invokelatest(bench_gpu, 10_000, 4, 4)
        Base.invokelatest(bench_gpu, 100_000, 4, 4)
    else
        @warn "CUDA radix lifecycle failed to load" FastMultipole.cuda_radix_status()
    end
end

# --- write records ---
stamp = Dates.format(now(), "yyyymmdd-HHMMSS")
csv_path = joinpath(DATA_DIR, "benchmark_023_$(HOST)_$(stamp).csv")
open(csv_path, "w") do io
    println(io, "backend,path,n,ell,P,phase,seconds")
    for r in rows
        println(io, "$(r.backend),$(r.path),$(r.n),$(r.ell),$(r.P),$(r.phase),$(r.seconds)")
    end
end
md_path = joinpath(DATA_DIR, "benchmark_023_$(HOST)_$(stamp).md")
open(md_path, "w") do io
    println(io, "# Task 023 integration benchmark — $(HOST), $(now())")
    println(io, "\nJulia $(VERSION), threads=$(Threads.nthreads()), gpu=$(RUN_GPU)\n")
    println(io, "| backend | path | n | ell | P | phase | seconds |")
    println(io, "|---|---|---|---|---|---|---|")
    for r in rows
        @printf(io, "| %s | %s | %d | %d | %d | %s | %.5f |\n",
            r.backend, r.path, r.n, r.ell, r.P, r.phase, r.seconds)
    end
end
println("\nwrote $csv_path")
println("wrote $md_path")
