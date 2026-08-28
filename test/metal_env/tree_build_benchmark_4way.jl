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

# Diagnostic: at n=1e6 the IQR/median ratio is 5-8x and unexplained (see
# project_fastmultipole_ka_migration memory). Dump the raw per-trial times so
# the distribution shape (bimodal? monotonic drift? isolated outliers?) can be
# inspected instead of guessing from median/IQR alone.
function print_raw_trials(name::String, n::Int, dist::Symbol, times::Vector{Float64})
    times_μs = round.(times ./ 1000; digits=1)
    println("  [raw] $(name) (n=$n, $dist) sorted trial times (μs): $(sort(times_μs))")
end

# Confirm/deny the CUDA-allocator-pressure hypothesis for n=1e6 (see
# project_fastmultipole_ka_migration memory): sortperm!/accumulate! calls
# inside the timed tree-refresh path may allocate their own scratch per call
# instead of using preallocated actx.* buffers. Diff device bytes allocated
# per trial directly instead of guessing from memory.used telemetry.
function print_raw_allocs(name::String, n::Int, dist::Symbol, allocs::Vector{Int64})
    allocs_kb = round.(allocs ./ 1024; digits=1)
    println("  [raw-allocs] $(name) (n=$n, $dist) sorted per-trial GPU allocs (KB): $(sort(allocs_kb))")
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

    actx = ext.ka_allocate_adaptive_context(Metal.MetalBackend(), Float32, n;
        leaf_capacity, frontier_capacity, node_capacity)

    # Warmup
    for _ in 1:nwarmup
        _ = ext.ka_build_adaptive_tree!(actx, dev_positions, ell_max, K_max, true, x_min, h0)
    end

    # Measure
    times = Float64[]
    ref_nodes, ref_leaves = nothing, nothing
    for trial in 1:ntrials
        t0 = time_ns()
        result = ext.ka_build_adaptive_tree!(actx, dev_positions, ell_max, K_max, true, x_min, h0)
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
#
# `benchmark_cuda_ka`/`benchmark_cuda_native` bodies use `CUDA.@allocated`,
# which (unlike a plain `CUDA.foo()` call) is macro-expanded when this file is
# *parsed*, not when the function is called — so it needs `CUDA` to already be
# a loaded module at that point. A plain runtime `if` still macroexpands both
# branches while lowering the whole top-level form, so `@static if` (elides
# the untaken branch before macroexpansion) is required here, not `if`.
@static if !HAS_METAL

function benchmark_cuda_ka(positions::Matrix{Float32}, ell_max::Int, K_max::Int;
        nwarmup::Int=1, ntrials::Int=5)
    if !CUDA.functional()
        return nothing, nothing, nothing, nothing
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

    actx = ext.ka_allocate_adaptive_context(CUDABackend(), Float32, n;
        leaf_capacity, frontier_capacity, node_capacity)

    # Warmup
    for _ in 1:nwarmup
        _ = ext.ka_build_adaptive_tree!(actx, dev_positions, ell_max, K_max, true, x_min, h0)
    end

    # Measure
    times = Float64[]
    allocs = Int64[]
    ref_nodes, ref_leaves = nothing, nothing
    for trial in 1:ntrials
        CUDA.synchronize()
        t0 = time_ns()
        local result
        alloc_bytes = CUDA.@allocated (result = ext.ka_build_adaptive_tree!(actx, dev_positions, ell_max, K_max, true, x_min, h0))
        CUDA.synchronize()
        t1 = time_ns()
        push!(times, Float64(t1 - t0))
        push!(allocs, alloc_bytes)

        if trial == 1
            ref_nodes = result.n_nodes
            ref_leaves = result.n_leaves
        end
    end

    return times, ref_nodes, ref_leaves, allocs
end

# ===== HPC-CUDA arm (native CUDA driver) =====

function benchmark_cuda_native(positions::Matrix{Float32}, ell_max::Int, K_max::Int;
        nwarmup::Int=1, ntrials::Int=5)
    if !CUDA.functional()
        return nothing, nothing, nothing, nothing
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
    # RadixFMMCache's constructor builds interaction lists (V/U/WX + DTR
    # frontier), not just the tree, even though only tree-build timing is
    # wanted here — unavoidable without bypassing the public constructor
    # entirely. Its default auto-sized capacities (_cuda_adaptive_capacities,
    # tree_batched_cuda.jl:44-58) are calibrated against real wake-simulation
    # V-list volumes; this benchmark's synthetic uniform/dense-cluster test
    # data is denser/more uniform than that and overflowed the default at
    # n=100000 (job 13505541: "DTR frontier capacity 6400000 exceeded").
    # Oversize generously (4x the observed overflow point at the same n) —
    # UNVERIFIED past n=100000 until this runs again.
    policy = AdaptiveTreePolicy(; ell_max, K_max, balance=true,
        v_capacity=256 * n, u_capacity=64 * n, wx_capacity=16 * n)

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
    allocs = Int64[]
    ref_nodes, ref_leaves = nothing, nothing
    for trial in 1:ntrials
        CUDA.synchronize()
        t0 = time_ns()
        alloc_bytes = CUDA.@allocated begin
            source_bufs = FastMultipole._radix_cache_refresh_source_buffers!(ctx, (sys,), Float32)
            FastMultipole._radix_cache_collect_positions!(ctx, source_bufs)
            _ = FastMultipole._cuda_refresh_adaptive_tree!(ctx, actx, cache, n)
        end
        CUDA.synchronize()
        t1 = time_ns()
        push!(times, Float64(t1 - t0))
        push!(allocs, alloc_bytes)

        if trial == 1
            ref_nodes = actx.n_nodes
            ref_leaves = actx.n_leaves
        end
    end

    return times, ref_nodes, ref_leaves, allocs
end

end # @static if !HAS_METAL

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
            ntrials = max(100, min(200, div(10_000_000, n)))
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
                cuda_ka_times, cuda_ka_nodes, cuda_ka_leaves, cuda_ka_allocs =
                    benchmark_cuda_ka(positions, ell_max, K_max; nwarmup, ntrials)
                if cuda_ka_times !== nothing
                    verify_tree_structure("CUDA-KA", cuda_ka_nodes, cuda_ka_leaves, n)
                    verify_matches_reference("CUDA-KA", cuda_ka_nodes, cuda_ka_leaves, cpu_nodes, cpu_leaves)
                    median_ns = median(cuda_ka_times)
                    iqr_ns = quantile(cuda_ka_times, 0.75) - quantile(cuda_ka_times, 0.25)
                    res = BenchmarkResult("CUDA-KA", n, dist, cuda_ka_nodes, cuda_ka_leaves,
                        median_ns, iqr_ns, ntrials)
                    print_result(res)
                    if n == Int(1e6)
                        print_raw_trials("CUDA-KA", n, dist, cuda_ka_times)
                        print_raw_allocs("CUDA-KA", n, dist, cuda_ka_allocs)
                    end
                    push!(results, res)
                end
            end

            # CUDA-native arm
            if !isapple() && CUDA.functional()
                println("  Running CUDA-native (HPC)...")
                cuda_native_times, cuda_native_nodes, cuda_native_leaves, cuda_native_allocs =
                    benchmark_cuda_native(positions, ell_max, K_max; nwarmup, ntrials)
                if cuda_native_times !== nothing
                    verify_tree_structure("CUDA-native", cuda_native_nodes, cuda_native_leaves, n)
                    verify_matches_reference("CUDA-native", cuda_native_nodes, cuda_native_leaves, cpu_nodes, cpu_leaves)
                    median_ns = median(cuda_native_times)
                    iqr_ns = quantile(cuda_native_times, 0.75) - quantile(cuda_native_times, 0.25)
                    res = BenchmarkResult("CUDA-native", n, dist, cuda_native_nodes, cuda_native_leaves,
                        median_ns, iqr_ns, ntrials)
                    print_result(res)
                    if n == Int(1e6)
                        print_raw_trials("CUDA-native", n, dist, cuda_native_times)
                        print_raw_allocs("CUDA-native", n, dist, cuda_native_allocs)
                    end
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
