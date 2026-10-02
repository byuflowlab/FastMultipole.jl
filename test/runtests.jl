using Test, Distributed

#--- CPU test files ---#
# Each job is a list of files that run in one process, in order; in parallel the
# jobs share no state but what helpers/setup.jl defines. fgs_coloring_test.jl
# uses the FGS overloads that solve_test.jl defines, so they form one job.
const JOBS = [
    ["auxilliary_test.jl"],
    ["metadata_extra_test.jl"],
    ["third_derivative_test.jl"],
    ["operator_cache_types_test.jl"],
    ["coefficient_buffer_layout_test.jl"],
    ["direct_conditioning_test.jl"],
    ["direct_test.jl"],
    ["direct_rectangular_test.jl"],
    ["core_regression_test.jl"],
    ["extra_systems_regression_test.jl"],
    ["extra_systems_test.jl"],
    ["extra_tree_test.jl"],
    ["ka_cpu_regression_test.jl"],
    ["harmonics_test.jl"],
    ["rotate_test.jl"],
    ["rotate_batched_test.jl"],
    ["bodytomultipole_test.jl"],
    ["multipole_power_test.jl"],
    ["translate_multipole_test.jl"],
    ["translate_multipole_to_local_test.jl"],
    ["m2l_operator_test.jl"],
    ["precomputed_y_resident_m2l_test.jl"],
    ["dense_translation_m2l_test.jl"],
    ["radix_settings_test.jl"],
    ["translate_local_test.jl"],
    ["evaluate_expansions_test.jl"],
    ["lamb_helmholtz_test.jl"],
    ["tree_test.jl"],
    ["radix_grid_clustering_test.jl"],
    ["hierarchical_m2l_host_test.jl"],
    ["radix_fmm_integration_test.jl"],
    ["radix_nearfield_test.jl"],
    ["radix_trimming_test.jl"],
    ["radix_fmm_timestepping_test.jl"],
    ["device_system_interface_test.jl"],
    ["ka_target_buffer_cache_test.jl"],
    ["point_body_types_test.jl"],
    ["filament_body_types_test.jl"],
    ["panel_body_types_test.jl"],
    ["dynamic_expansion_order_test.jl"],
    ["fmm_test.jl"],
    ["fmm_plan_test.jl"],
    ["transform_tree_test.jl"],
    ["transform_plan_test.jl"],
    ["nearfield_cache_test.jl"],
    ["solve_test.jl", "transform_solver_test.jl", "fgs_coloring_test.jl"],
]
# the slowest jobs, started first so the parallel queue drains evenly
const SLOW_FIRST = ["core_regression_test.jl", "device_system_interface_test.jl",
    "extra_systems_regression_test.jl", "radix_fmm_integration_test.jl",
    "hierarchical_m2l_host_test.jl", "third_derivative_test.jl",
    "panel_body_types_test.jl", "extra_tree_test.jl"]

# FASTMULTIPOLE_TEST_WORKERS=1 runs everything in this process, in JOBS order.
# Otherwise each worker loads helpers/setup.jl once and takes jobs from a shared
# queue; a job's output goes to test/logs/<file>.log and is printed only if it
# fails. Compilation dominates the suite and is single-threaded, so processes,
# not threads, are what run it in parallel.
const NWORKERS = parse(Int, get(ENV, "FASTMULTIPOLE_TEST_WORKERS",
    string(clamp(Sys.CPU_THREADS ÷ 2, 1, 4))))

if NWORKERS <= 1
    include("helpers/setup.jl")
    foreach(job -> foreach(include, job), JOBS)
else
    # workers run with this process's project, coverage and bounds checking
    flags = ["--project=$(Base.active_project())", "--threads=2", "--startup-file=no"]
    Base.JLOptions().code_coverage != 0 && push!(flags, "--code-coverage=user")
    Base.JLOptions().check_bounds == 1 && push!(flags, "--check-bounds=yes")
    addprocs(NWORKERS; exeflags=flags)
    logdir = mkpath(joinpath(@__DIR__, "logs"))
    @everywhere include($(joinpath(@__DIR__, "helpers", "setup.jl")))
    @everywhere function run_job(job, logdir)
        log = joinpath(logdir, first(job) * ".log")
        t0 = time(); ok = true
        open(log, "w") do io
            redirect_stdout(io) do; redirect_stderr(io) do
                try
                    foreach(f -> Base.include(Main, joinpath($(@__DIR__), f)), job)
                catch err
                    ok = false
                    showerror(io, err, catch_backtrace()); println(io)
                end
            end; end
        end
        return (job=job, ok=ok, time=time() - t0, log=log)
    end
    slow(job) = any(in(SLOW_FIRST), job)
    queue = vcat(filter(slow, JOBS), filter(!slow, JOBS))
    results = pmap(job -> run_job(job, logdir), queue)
    rmprocs(workers())
    for r in sort(results; by=r -> -r.time)
        println(rpad(join(r.job, " + "), 60), r.ok ? "PASS" : "FAIL", "  ",
            round(r.time; digits=1), " s")
    end
    for r in filter(r -> !r.ok, results)
        println("\n--- ", r.log, " ---"); print(read(r.log, String))
    end
    @testset "CPU test files" begin
        @test all(r -> r.ok, results)
    end
end

#--- GPU correctness suites ---#
# The ka_*_correctness.jl suites live in their own env (test/gpu) so the
# main test env stays free of CUDA/Metal. They run only when a device is
# plausibly present; each suite still checks `dev_functional()` itself.
# run_suites.sh exits with the number of failing suites; per-suite logs go to
# test/gpu/logs/. FASTMULTIPOLE_GPU_TESTS=0|1 overrides detection and
# FASTMULTIPOLE_GPU_TEST_PROJECT points the suites at another env.
gpu_present = Sys.isapple() || Sys.which("nvidia-smi") !== nothing
# test/gpu/Project.toml declares Metal only; an NVIDIA machine must point
# FASTMULTIPOLE_GPU_TEST_PROJECT at an environment with CUDA (see docs/src/gpu.md)
gpu_env_ok = Sys.isapple() || haskey(ENV, "FASTMULTIPOLE_GPU_TEST_PROJECT")
if get(ENV, "FASTMULTIPOLE_GPU_TESTS", gpu_present ? "1" : "0") == "1" && !gpu_env_ok
    @warn "GPU detected but FASTMULTIPOLE_GPU_TEST_PROJECT is unset: the device suites need an environment with CUDA; skipping them (set FASTMULTIPOLE_GPU_TESTS=0 to silence)"
end
if get(ENV, "FASTMULTIPOLE_GPU_TESTS", gpu_present ? "1" : "0") == "1" && gpu_env_ok
    @testset "GPU correctness suites" begin
        @test success(`bash $(joinpath(@__DIR__, "gpu", "run_suites.sh"))`)
    end
end
