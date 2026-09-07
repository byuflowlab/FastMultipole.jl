#=##############################################################################
Disabled-path performance benchmark for the third-derivative feature
(HIGHER_DERIVATIVES/REVIEW_PLAN.md, Test and Acceptance Plan).

Paired ALTERNATING baseline/feature runs: two persistent worker processes are
launched (one per checkout), warmed up, then sampled alternately so machine
drift affects both sides equally. Acceptance: with at least 30 warmed samples
of at least 0.2 s of work each, median(feature)/median(baseline) <= 1.03.
Rerun one failure; a repeated failure blocks the I7 milestone and requires
profiling.

Usage (driver):
    julia benchmark_ts_disabled_path.jl <baseline_dir> <feature_dir> [n_samples] [n_bodies]

  <baseline_dir>  checkout (git worktree) of the pre-third-derivative commit
  <feature_dir>   checkout of the feature branch (e.g. the repo root)
  n_samples       samples per side, default 30 (minimum 30)
  n_bodies        gravitational bodies per fmm! sample, default 30_000 —
                  raise it if a sample completes in under 0.2 s

Each directory must contain Project.toml for FastMultipole and
test/gravitational.jl. Workers run single-threaded for stable timing.
The workload is a full fmm! call with third_derivative NOT requested
(scalar_potential=false, gradient=true, hessian=false), which is the path
that must not regress.
=###############################################################################

if length(ARGS) >= 1 && ARGS[1] == "--worker"
    # ---- worker mode: --worker <project_dir> <n_bodies> ----
    project_dir = ARGS[2]
    n_bodies = parse(Int, ARGS[3])
    using FastMultipole
    using FastMultipole.StaticArrays
    using FastMultipole.LinearAlgebra
    using Random
    include(joinpath(project_dir, "test", "gravitational.jl"))

    system = generate_gravitational(2026, n_bodies)
    sample_kwargs = (; scalar_potential=false, gradient=true, hessian=false,
        leaf_size=50, expansion_order=5, multipole_acceptance=0.4)
    run_sample() = (system.potential .= 0; @elapsed fmm!(system; sample_kwargs...))

    # warm-up (JIT + first-touch); not recorded
    run_sample(); run_sample()
    t_check = run_sample()
    println("READY $t_check")
    flush(stdout)
    while true
        line = readline(stdin)
        (line == "q" || isempty(line)) && break
        println(run_sample())
        flush(stdout)
    end
    exit(0)
end

# ---- driver mode ----
using Statistics

length(ARGS) >= 2 || error("usage: julia benchmark_ts_disabled_path.jl <baseline_dir> <feature_dir> [n_samples] [n_bodies]")
baseline_dir = abspath(ARGS[1])
feature_dir = abspath(ARGS[2])
n_samples = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 30
n_bodies = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 30_000
n_samples >= 30 || error("acceptance requires at least 30 samples per side")

this_script = abspath(@__FILE__)
open_worker(dir) = open(
    `julia --project=$dir --threads=1 $this_script --worker $dir $n_bodies`, "r+")

println("launching workers (baseline=$baseline_dir, feature=$feature_dir, " *
    "n_samples=$n_samples, n_bodies=$n_bodies)...")
workers = (baseline=open_worker(baseline_dir), feature=open_worker(feature_dir))
for (name, w) in pairs(workers)
    ready = readline(w)
    startswith(ready, "READY") || error("$name worker failed: $ready")
    t = parse(Float64, split(ready)[2])
    println("  $name warmed, check sample $(round(t; digits=3)) s")
    t >= 0.2 || error("$name sample took $(round(t; digits=3)) s < 0.2 s — " *
        "raise n_bodies")
end

sample!(w) = (println(w, "s"); flush(w); parse(Float64, readline(w)))
times = (baseline=Float64[], feature=Float64[])
for i in 1:n_samples
    push!(times.baseline, sample!(workers.baseline))
    push!(times.feature, sample!(workers.feature))
    i % 5 == 0 && println("  $i/$n_samples sample pairs done")
end
for w in workers
    println(w, "q"); flush(w); close(w)
end

med_b, med_f = median(times.baseline), median(times.feature)
ratio = med_f / med_b
println("\nbaseline median: $(round(med_b; digits=4)) s over $n_samples samples")
println("feature  median: $(round(med_f; digits=4)) s over $n_samples samples")
println("feature/baseline ratio: $(round(ratio; digits=4)) (acceptance: <= 1.03)")
println(ratio <= 1.03 ? "PASS" : "FAIL — rerun once; a repeated failure blocks I7 and requires profiling")
exit(ratio <= 1.03 ? 0 : 1)
