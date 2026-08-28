# 4-way tree-build benchmark: CPU, local Metal-KA, HPC KA, HPC CUDA
# Dispatch on Metal.functional()/CUDA.functional() to enable the appropriate arms
# For cluster use: this same script runs on orc with both KA and CUDA arms active;
# combine both into a single sbatch job for efficiency.

using FastMultipole
using KernelAbstractions
using Random
using Statistics
import Base.Sys: isapple

# Metal/CUDA are mutually exclusive here: the HPC (CUDA) env has no Metal
# package installed at all (correctly — no Apple GPU there), so `using Metal`
# must not even be attempted off-Apple, or package resolution fails before
# any code runs.
const HAS_METAL = isapple()
if HAS_METAL
    using Metal
else
    using CUDA
    FastMultipole.load_cuda_radix_lifecycle!()
end

include("../gravitational.jl")

# ===== Benchmark infrastructure =====

struct BenchmarkResult
    name::String
    n_bodies::Int
    distribution::Symbol
    n_nodes::Int
    n_leaves::Int
    median_ns::Float64
    iqr_ns::Float64
    ntrials::Int
end

function print_result(res::BenchmarkResult)
    time_μs = res.median_ns / 1000
    iqr_μs = res.iqr_ns / 1000
    println("$(res.name) (n=$(res.n_bodies), $(res.distribution)): " *
            "$(res.n_nodes) nodes, $(res.n_leaves) leaves, " *
            "$(time_μs)μs [IQR ±$(iqr_μs)μs] ($(res.ntrials) trials)")
end

# Sanity check: tree has reasonable structure (not empty, leaves <= nodes)
function verify_tree_structure(name::String, n_nodes::Int, n_leaves::Int, n_bodies::Int)
    n_nodes > 0 || error("$name: tree has no nodes")
    n_leaves > 0 || error("$name: tree has no leaves")
    n_leaves <= n_nodes || error("$name: leaves > nodes (sanity check failed)")
    return true
end

# Correctness gate: node/leaf counts must match the CPU reference exactly
# (all 4 arms build the same fixed-root-cube, fixed-policy octree from the
# same positions, so they must agree bit-for-bit on counts).
function verify_matches_reference(name::String, n_nodes::Int, n_leaves::Int,
        ref_nodes::Int, ref_leaves::Int)
    n_nodes == ref_nodes || error(
        "$name: n_nodes=$n_nodes != CPU reference n_nodes=$ref_nodes")
    n_leaves == ref_leaves || error(
        "$name: n_leaves=$n_leaves != CPU reference n_leaves=$ref_leaves")
    return true
end

# ===== CPU arm =====

function benchmark_cpu(positions::Matrix{Float32}, ell_max::Int, K_max::Int;
        nwarmup::Int=1, ntrials::Int=5)
    n = size(positions, 2)
    x_min = (0.0f0, 0.0f0, 0.0f0)
    h0 = 1.0f0
    # Pack into bodies matrix: [x, y, z, radius, strength] per column
    bodies = zeros(Float32, 5, n)
    bodies[1:3, :] = positions
    bodies[4, :] .= 0.01f0  # dummy radius
    bodies[5, :] .= 1.0f0 / n  # normalized strength

    sys = Gravitational(bodies)
    policy = AdaptiveTreePolicy(; ell_max, K_max, balance=true)
    # Must fix the same root cube (x_min, h0) the KA/CUDA arms use, or CPU's
    # auto-computed data-bounds root (the default when root=nothing) encodes
    # different Morton keys and produces a structurally different tree.
    root = (x_min, h0)

    # Warmup
    for _ in 1:nwarmup
        tree = AdaptiveRadixTree(sys; policy, root)
    end

    # Measure
    times = Float64[]
    ref_nodes, ref_leaves = nothing, nothing
    for trial in 1:ntrials
        t0 = time_ns()
        tree = AdaptiveRadixTree(sys; policy, root)
        t1 = time_ns()
        push!(times, Float64(t1 - t0))

        if trial == 1
            ref_nodes = tree.n_nodes
            ref_leaves = tree.n_leaves
        end
    end

    return times, ref_nodes, ref_leaves
end

# ===== Metal-KA arm =====

function benchmark_metal_ka(positions::Matrix{Float32}, ell_max::Int, K_max::Int;
        nwarmup::Int=1, ntrials::Int=5)
    if !HAS_METAL || !Metal.functional()
        return nothing, nothing, nothing
    end

    n = size(positions, 2)
    x_min = (0.0f0, 0.0f0, 0.0f0)
    h0 = 1.0f0

    ext = Base.get_extension(FastMultipole, :FastMultipoleKAExt)
    ext !== nothing || error("FastMultipoleKAExt did not load")

    dev_positions = Metal.MtlArray(positions)

    # Capacity sizing: overestimate to be safe (conservative multipliers)
    # For a balanced tree, leaves ≈ n/K_max, but CPU AdaptiveRadixTree shows
    # it can be much larger. Use generous overestimation.
    nl_estimate = max(2 * n ÷ K_max, 16)
    node_capacity = 100 * nl_estimate + 256
    leaf_capacity = 10 * nl_estimate + 256
    frontier_capacity = 16 * leaf_capacity

    # Warmup
    for _ in 1:nwarmup
        _ = ext.ka_build_adaptive_tree!(dev_positions, ell_max, K_max, true,
            x_min, h0; leaf_capacity, frontier_capacity, node_capacity)
    end

    # Measure
    times = Float64[]
    ref_nodes, ref_leaves = nothing, nothing
    for trial in 1:ntrials
        t0 = time_ns()
        result = ext.ka_build_adaptive_tree!(dev_positions, ell_max, K_max, true,
            x_min, h0; leaf_capacity, frontier_capacity, node_capacity)
        KernelAbstractions.synchronize(Metal.MetalBackend())
        t1 = time_ns()
        push!(times, Float64(t1 - t0))

        if trial == 1
            ref_nodes = result.n_nodes
            ref_leaves = result.n_leaves
        end
    end

    return times, ref_nodes, ref_leaves
end

# ===== HPC-KA arm (CUDA dispatch, same code as Metal arm) =====

function benchmark_cuda_ka(positions::Matrix{Float32}, ell_max::Int, K_max::Int;
        nwarmup::Int=1, ntrials::Int=5)
    if !CUDA.functional()
        return nothing, nothing, nothing
    end

    n = size(positions, 2)
    x_min = (0.0f0, 0.0f0, 0.0f0)
    h0 = 1.0f0

    ext = Base.get_extension(FastMultipole, :FastMultipoleKAExt)
    ext !== nothing || error("FastMultipoleKAExt did not load")

    dev_positions = CUDA.CuArray(positions)

    nl_estimate = max(2 * n ÷ K_max, 16)
    node_capacity = 100 * nl_estimate + 256
    leaf_capacity = 10 * nl_estimate + 256
    frontier_capacity = 16 * leaf_capacity

    # Warmup
    for _ in 1:nwarmup
        _ = ext.ka_build_adaptive_tree!(dev_positions, ell_max, K_max, true,
            x_min, h0; leaf_capacity, frontier_capacity, node_capacity)
    end

    # Measure
    times = Float64[]
    ref_nodes, ref_leaves = nothing, nothing
    for trial in 1:ntrials
        CUDA.synchronize()
        t0 = time_ns()
        result = ext.ka_build_adaptive_tree!(dev_positions, ell_max, K_max, true,
            x_min, h0; leaf_capacity, frontier_capacity, node_capacity)
        CUDA.synchronize()
        t1 = time_ns()
        push!(times, Float64(t1 - t0))

        if trial == 1
            ref_nodes = result.n_nodes
            ref_leaves = result.n_leaves
        end
    end

    return times, ref_nodes, ref_leaves
end

# ===== HPC-CUDA arm (native CUDA driver) =====

function benchmark_cuda_native(positions::Matrix{Float32}, ell_max::Int, K_max::Int;
        nwarmup::Int=1, ntrials::Int=5)
    if !CUDA.functional()
        return nothing, nothing, nothing
    end

    n = size(positions, 2)
    x_min = (0.0f0, 0.0f0, 0.0f0)
    h0 = 1.0f0
    # Pack bodies the same way as CPU
    bodies = zeros(Float32, 5, n)
    bodies[1:3, :] = positions
    bodies[4, :] .= 0.01f0
    bodies[5, :] .= 1.0f0 / n

    sys = Gravitational(bodies)
    policy = AdaptiveTreePolicy(; ell_max, K_max, balance=true)

    # Fix the same root cube (x_min, h0) as the CPU/KA arms via `bounds`
    # (RadixFMMCache's box_size = 2*h0; leaving bounds=nothing would let it
    # auto-fit to the data instead, encoding different Morton keys — the same
    # gotcha fixed in the CPU arm above).
    cache = RadixFMMCache(sys; expansion_order=4, ell=ell_max,
        device=true, adaptive=policy, bounds=(x_min, 2 * h0))

    ctx = cache.device_ctx
    actx = cache.adaptive_tree::FastMultipole.DeviceAdaptiveCUDAContext

    # Warmup
    for _ in 1:nwarmup
        source_bufs = FastMultipole._radix_cache_refresh_source_buffers!(ctx, (sys,), Float32)
        FastMultipole._radix_cache_collect_positions!(ctx, source_bufs)
        _ = FastMultipole._cuda_refresh_adaptive_tree!(ctx, actx, cache, n)
    end

    # Measure (use actx.stage_ns for per-stage timing if profiling=true)
    actx.profile_stages = true
    times = Float64[]
    ref_nodes, ref_leaves = nothing, nothing
    for trial in 1:ntrials
        CUDA.synchronize()
        t0 = time_ns()
        source_bufs = FastMultipole._radix_cache_refresh_source_buffers!(ctx, (sys,), Float32)
        FastMultipole._radix_cache_collect_positions!(ctx, source_bufs)
        _ = FastMultipole._cuda_refresh_adaptive_tree!(ctx, actx, cache, n)
        CUDA.synchronize()
        t1 = time_ns()
        push!(times, Float64(t1 - t0))

        if trial == 1
            ref_nodes = actx.n_nodes
            ref_leaves = actx.n_leaves
        end
    end

    return times, ref_nodes, ref_leaves
end

# ===== Distribution generators =====

function generate_uniform_random(n::Int, seed::Int)
    Random.seed!(seed)
    positions = rand(Float32, 3, n)  # uniform in [0,1]^3
    positions .*= 2.0f0  # scale to [0,2]^3 (h0=1 domain is [0,2]^3)
    return positions
end

function generate_dense_cluster(n::Int, seed::Int; n_cluster::Int=100, cluster_radius::Float32=0.1f0)
    Random.seed!(seed)
    positions = rand(Float32, 3, n)
    if n >= n_cluster
        # Add a tight cluster in the middle of the pre-scale [0,1)^3 domain
        # (0.5, not 1.0 — 1.0 is the domain's edge, and cluster_radius=0.1
        # pushed points past it, outside the [0,2]^3 fixed root cube post-scale)
        cluster_idx = randperm(n)[1:n_cluster]
        cluster_center = 0.5f0 * ones(Float32, 3)
        positions[:, cluster_idx] = cluster_center .+ cluster_radius .* (rand(Float32, 3, n_cluster) .- 0.5f0)
    end
    positions .*= 2.0f0
    return positions
end

# ===== Main benchmark =====

function main()
    println("\n=== 4-way tree-build benchmark (CPU, Metal-KA, CUDA-KA, CUDA-native) ===\n")

    # Parameters
    ell_max = 6
    K_max = 16
    ns = [Int(1e3), Int(1e4), Int(1e5), Int(1e6)]
    distributions = [:uniform, :clustered]

    results = BenchmarkResult[]

    for n in ns
        for dist in distributions
            println("Building test case: n=$n, distribution=$dist")

            if dist == :uniform
                positions = generate_uniform_random(n, 42)
            else
                positions = generate_dense_cluster(n, 42)
            end

            # CPU reference (for sanity check only — CPU and KA trees differ structurally)
            cpu_times, cpu_nodes, cpu_leaves = benchmark_cpu(positions, ell_max, K_max;
                nwarmup=1, ntrials=1)
            verify_tree_structure("CPU", cpu_nodes, cpu_leaves, n)
            println("  CPU: $cpu_nodes nodes, $cpu_leaves leaves")

            # Determine trial counts based on n (smaller n → more trials)
            ntrials = max(10, min(200, div(10_000_000, n)))
            nwarmup = max(1, div(ntrials, 5))

            # Metal-KA arm
            if HAS_METAL && Metal.functional()
                println("  Running Metal-KA (local)...")
                metal_times, metal_nodes, metal_leaves = benchmark_metal_ka(positions, ell_max, K_max;
                    nwarmup, ntrials)
                verify_tree_structure("Metal-KA", metal_nodes, metal_leaves, n)
                verify_matches_reference("Metal-KA", metal_nodes, metal_leaves, cpu_nodes, cpu_leaves)
                median_ns = median(metal_times)
                iqr_ns = quantile(metal_times, 0.75) - quantile(metal_times, 0.25)
                res = BenchmarkResult("Metal-KA", n, dist, metal_nodes, metal_leaves,
                    median_ns, iqr_ns, ntrials)
                print_result(res)
                push!(results, res)
            else
                println("  Metal not functional; skipping Metal-KA")
            end

            # CUDA-KA arm
            if !isapple() && CUDA.functional()
                println("  Running CUDA-KA (HPC)...")
                cuda_ka_times, cuda_ka_nodes, cuda_ka_leaves =
                    benchmark_cuda_ka(positions, ell_max, K_max; nwarmup, ntrials)
                if cuda_ka_times !== nothing
                    verify_tree_structure("CUDA-KA", cuda_ka_nodes, cuda_ka_leaves, n)
                    verify_matches_reference("CUDA-KA", cuda_ka_nodes, cuda_ka_leaves, cpu_nodes, cpu_leaves)
                    median_ns = median(cuda_ka_times)
                    iqr_ns = quantile(cuda_ka_times, 0.75) - quantile(cuda_ka_times, 0.25)
                    res = BenchmarkResult("CUDA-KA", n, dist, cuda_ka_nodes, cuda_ka_leaves,
                        median_ns, iqr_ns, ntrials)
                    print_result(res)
                    push!(results, res)
                end
            end

            # CUDA-native arm
            if !isapple() && CUDA.functional()
                println("  Running CUDA-native (HPC)...")
                cuda_native_times, cuda_native_nodes, cuda_native_leaves =
                    benchmark_cuda_native(positions, ell_max, K_max; nwarmup, ntrials)
                if cuda_native_times !== nothing
                    verify_tree_structure("CUDA-native", cuda_native_nodes, cuda_native_leaves, n)
                    verify_matches_reference("CUDA-native", cuda_native_nodes, cuda_native_leaves, cpu_nodes, cpu_leaves)
                    median_ns = median(cuda_native_times)
                    iqr_ns = quantile(cuda_native_times, 0.75) - quantile(cuda_native_times, 0.25)
                    res = BenchmarkResult("CUDA-native", n, dist, cuda_native_nodes, cuda_native_leaves,
                        median_ns, iqr_ns, ntrials)
                    print_result(res)
                    push!(results, res)
                end
            end
        end
    end

    println("\n=== Summary ===")
    for res in results
        print_result(res)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
