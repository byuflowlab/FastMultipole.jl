# Isolate a single n=1e6, uniform tree-build case in its own fresh Julia
# process (vs. tree_build_benchmark_4way.jl's 8-cases-in-one-process loop).
# Last untried lever for the n>=1e5 staircase anomaly after clock throttling,
# node contention, cross-case accumulation, per-stage isolation, HBM
# throttling, and pool-trim were all refuted (see
# project_fastmultipole_ka_migration memory). Distinguishes "something about
# this specific case" from "something accumulated by running 8 cases
# sequentially in one process."
#
# Job 13508342 answered that (it reproduces in isolation) and, via per-trial
# NVML, killed the clock/pstate hypothesis AND the "staircase" itself: the
# earlier ramp was an artifact of printing sorted times. The real shape is
# bimodal with slow trials scattered from trial 1, no drift.
#
# This run localizes where the stalled time goes. Per trial it records host CPU
# time (process and thread) alongside wall time — a slow trial that burns
# proportional CPU is host-side, one that burns floor-level CPU was blocked in
# the driver — plus NVML compute-process count (co-tenancy on the device, which
# --gpus=h200:1 should prevent but has never been checked) and the per-stage
# breakdown in trial order.
#
# NOTE: --cpus-per-task stays at 1 here deliberately. This run characterizes the
# current configuration; changing the CPU allocation at the same time would
# confound the measurement. Enabling profile=true also adds inter-stage syncs to
# the KA arm, so its absolute times are not directly comparable to 13508342's.
include("tree_build_benchmark_4way.jl")

function main_isolated()
    println("\n=== Isolated single-case tree-build benchmark: n=1e6, uniform ===\n")

    ell_max = 6
    K_max = 16
    n = Int(1e6)
    dist = :uniform
    positions = generate_uniform_random(n, 42)

    cpu_times, cpu_nodes, cpu_leaves = benchmark_cpu(positions, ell_max, K_max;
        nwarmup=1, ntrials=1)
    verify_tree_structure("CPU", cpu_nodes, cpu_leaves, n)
    println("  CPU: $cpu_nodes nodes, $cpu_leaves leaves")

    ntrials = max(100, min(200, div(10_000_000, n)))
    nwarmup = max(1, div(ntrials, 5))

    if !isapple() && CUDA.functional()
        println("  Running CUDA-KA (HPC, isolated process)...")
        cuda_ka_times, cuda_ka_nodes, cuda_ka_leaves, cuda_ka_allocs,
        cuda_ka_stages, _cuda_ka_used, _cuda_ka_cached, cuda_ka_tele,
        cuda_ka_pcpu, cuda_ka_tcpu =
            benchmark_cuda_ka(positions, ell_max, K_max; nwarmup, ntrials,
                telemetry=true, profile=true)
        verify_tree_structure("CUDA-KA", cuda_ka_nodes, cuda_ka_leaves, n)
        verify_matches_reference("CUDA-KA", cuda_ka_nodes, cuda_ka_leaves, cpu_nodes, cpu_leaves)
        median_ns = median(cuda_ka_times)
        iqr_ns = quantile(cuda_ka_times, 0.75) - quantile(cuda_ka_times, 0.25)
        res = BenchmarkResult("CUDA-KA", n, dist, cuda_ka_nodes, cuda_ka_leaves,
            median_ns, iqr_ns, ntrials)
        print_result(res)
        print_raw_trials("CUDA-KA", n, dist, cuda_ka_times)
        print_raw_allocs("CUDA-KA", n, dist, cuda_ka_allocs)
        print_telemetry("CUDA-KA", n, dist, cuda_ka_times, cuda_ka_tele)
        print_cpu_split("CUDA-KA", n, dist, cuda_ka_times, cuda_ka_pcpu, cuda_ka_tcpu)
        print_stage_breakdown("CUDA-KA", n, dist, cuda_ka_stages)
    end

    if !isapple() && CUDA.functional()
        println("  Running CUDA-native (HPC, isolated process)...")
        cuda_native_times, cuda_native_nodes, cuda_native_leaves, cuda_native_allocs,
        cuda_native_stages, _cuda_nat_used, _cuda_nat_cached, cuda_native_tele,
        cuda_native_pcpu, cuda_native_tcpu =
            benchmark_cuda_native(positions, ell_max, K_max; nwarmup, ntrials, telemetry=true)
        verify_tree_structure("CUDA-native", cuda_native_nodes, cuda_native_leaves, n)
        verify_matches_reference("CUDA-native", cuda_native_nodes, cuda_native_leaves, cpu_nodes, cpu_leaves)
        median_ns = median(cuda_native_times)
        iqr_ns = quantile(cuda_native_times, 0.75) - quantile(cuda_native_times, 0.25)
        res = BenchmarkResult("CUDA-native", n, dist, cuda_native_nodes, cuda_native_leaves,
            median_ns, iqr_ns, ntrials)
        print_result(res)
        print_raw_trials("CUDA-native", n, dist, cuda_native_times)
        print_raw_allocs("CUDA-native", n, dist, cuda_native_allocs)
        print_telemetry("CUDA-native", n, dist, cuda_native_times, cuda_native_tele)
        print_cpu_split("CUDA-native", n, dist, cuda_native_times, cuda_native_pcpu, cuda_native_tcpu)
        print_stage_breakdown("CUDA-native", n, dist, cuda_native_stages)
    end
end

main_isolated()
