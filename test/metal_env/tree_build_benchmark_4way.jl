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
#
# TRIAL ORDER FIRST, sorted second. This printer used to emit ONLY the sorted
# array, which is what produced the "staircase"/"ramp starting at trial 50-55"
# readings in jobs 13506267/13506486 — a sorted bimodal sample is a step
# function by construction, and its apparent "onset" is just the fraction of
# fast trials, which is exactly the "onset tracks trial-loop fraction, not
# trial number" property that was treated as a clue. Job 13508342 showed the
# slow trials are in fact scattered from trial 1 onward with no drift at all.
# Never diagnose ordering/drift from the sorted dump.
function print_raw_trials(name::String, n::Int, dist::Symbol, times::Vector{Float64})
    times_μs = round.(times ./ 1000; digits=1)
    println("  [raw] $(name) (n=$n, $dist) trial-order times (μs): $times_μs")
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

# Per-stage breakdown (key+sort, build_leaves, balance, finalize) to localize
# which internal stage drives the n>=1e5 timing anomaly (see
# project_fastmultipole_ka_migration memory) instead of guessing from the
# whole-call time alone.
const STAGE_NAMES = ("sort", "build_leaves", "balance", "finalize")
function print_stage_breakdown(name::String, n::Int, dist::Symbol, stage_hist)
    for (s, sname) in enumerate(STAGE_NAMES)
        vals_us = round.(stage_hist[s] ./ 1000; digits=1)
        med_us = round(median(stage_hist[s]) / 1000; digits=1)
        iqr_us = round((quantile(stage_hist[s], 0.75) - quantile(stage_hist[s], 0.25)) / 1000; digits=1)
        # Trial order, not sorted — see print_raw_trials. The "all 4 stages
        # staircase at the same relative point" reading (job 13506267) was an
        # artifact of sorting each stage's samples independently.
        println("  [stage] $(name) (n=$n, $dist) $sname: median=$(med_us)μs IQR=±$(iqr_us)μs " *
                "trial order: $vals_us")
    end
end

# Test whether the n>=1e5 staircase coincides with a stream-ordered-allocator
# pool-trim/growth event, invisible to CUDA.@allocated (which only reports net
# bytes attributed to a call, not pool housekeeping). CUDA.used_memory()/
# cached_memory() read the pool's MEMPOOL_ATTR_USED_MEM_CURRENT/
# MEMPOOL_ATTR_RESERVED_MEM_CURRENT attributes directly and are cheap
# (no kernel launch), so sample every trial rather than subsampling — printed
# in trial order (not sorted, unlike the other raw dumps) since a pool-trim
# event is a one-time step, not a distribution to characterize.
function print_pool_samples(name::String, n::Int, dist::Symbol,
        used_hist::Vector{Int64}, cached_hist::Vector{Int64})
    used_mb = round.(used_hist ./ 1024^2; digits=2)
    cached_mb = round.(cached_hist ./ 1024^2; digits=2)
    println("  [pool] $(name) (n=$n, $dist) used (MiB, trial order): $used_mb")
    println("  [pool] $(name) (n=$n, $dist) cached/reserved (MiB, trial order): $cached_mb")
end

# Per-trial GPU telemetry. Every prior clock/power/throttle check (jobs
# 13506034, 13506267, 13506362) sampled `nvidia-smi` once per second from a
# background shell loop — far too coarse, and unaligned to trial index, to see
# whether a clock/pstate transition lands at the specific trial where the
# n>=1e5 ramp starts (~trial 50-55 of 100, job 13506486). NVML is queried
# in-process here, once per trial, *outside* the timed region, so each sample
# carries a trial index. Queries are host-side driver reads (no kernel launch,
# no sync), ~tens of μs against a ~6ms trial.
@static if !HAS_METAL

struct TrialTelemetry
    sm_mhz::Int
    mem_mhz::Int
    power_w::Float64
    temp_c::Int
    pstate::Int
    util_compute::Float64
    events::String
    nprocs::Int
end

# Host CPU time, to separate "the host thread was busy/spinning" from "the host
# thread was blocked waiting on the device". Job 13508342 showed n=1e6 trials are
# bimodal (~5.3ms floor vs. scattered 10-90ms) with no drift, and that the stall
# is not GPU compute — but NVML's utilization is sampled over a driver-chosen
# window not aligned to a trial, so it cannot say where the stalled time goes.
# Wall-vs-CPU per trial can:
#   wall 90ms / cpu ~90ms  -> host-side: spinning or descheduled-but-runnable
#                             (CUDA.jl's non-blocking sync spins, so it counts)
#   wall 90ms / cpu  ~5ms  -> host blocked in the driver: device/driver-side
# CLOCK_PROCESS_CPUTIME_ID(2) covers CUDA's internal threads too;
# CLOCK_THREAD_CPUTIME_ID(3) isolates the thread running the trial. Both are
# vDSO reads (~20-30ns), negligible against a 5ms trial.
const CLOCK_PROCESS_CPUTIME_ID = Cint(2)
const CLOCK_THREAD_CPUTIME_ID = Cint(3)

function clock_ns(clockid::Cint)
    ts = Ref{NTuple{2,Int64}}((Int64(0), Int64(0)))
    rc = ccall(:clock_gettime, Cint, (Cint, Ptr{Cvoid}), clockid, ts)
    rc == 0 || return UInt64(0)
    s, ns = ts[]
    return UInt64(s) * 1_000_000_000 + UInt64(ns)
end

process_cpu_ns() = clock_ns(CLOCK_PROCESS_CPUTIME_ID)
thread_cpu_ns() = clock_ns(CLOCK_THREAD_CPUTIME_ID)

function nvml_device()
    try
        return NVML.Device(CUDA.uuid(CUDA.device()))
    catch
        return NVML.Device(0)
    end
end

function sample_nvml(dev)
    clocks = NVML.clock_info(dev)
    sm = Int(get(clocks, :sm, get(clocks, :graphics, 0)))
    mem = Int(get(clocks, :memory, 0))
    pstate = try
        ref = Ref{NVML.nvmlPstates_t}()
        NVML.nvmlDeviceGetPerformanceState(dev, ref)
        Int(ref[])
    catch
        -1
    end
    reasons = try
        NVML.clock_event_reasons(dev)
    catch
        NamedTuple()
    end
    active = join(String.([k for (k, v) in pairs(reasons) if v]), "|")
    # Co-tenancy: --gpus=h200:1 should give us the device exclusively, but that
    # has never actually been verified, and another process holding a CUDA
    # context on the same device would serialize against ours via context
    # switching while leaving device-wide utilization low — which is exactly
    # the signature seen so far.
    nprocs = try
        length(NVML.compute_processes(dev))
    catch
        -1
    end
    return TrialTelemetry(sm, mem, NVML.power_usage(dev), NVML.temperature(dev),
        pstate, NVML.utilization_rates(dev).compute, isempty(active) ? "-" : active,
        nprocs)
end

# Printed in trial order (not sorted) and next to the trial's own time: the
# whole point is to see whether a telemetry change coincides with the ramp
# onset, which sorting would destroy.
function print_telemetry(name::String, n::Int, dist::Symbol,
        times::Vector{Float64}, tele::Vector{TrialTelemetry})
    println("  [tele] $(name) (n=$n, $dist) per-trial telemetry:")
    println("  [tele]  trial       t_us   sm_MHz  mem_MHz   power_W  temp_C  pstate  util  nproc  events")
    for (i, s) in enumerate(tele)
        t_us = round(times[i] / 1000; digits=1)
        println("  [tele]  ", lpad(i, 5), lpad(t_us, 11), lpad(s.sm_mhz, 9),
            lpad(s.mem_mhz, 9), lpad(round(s.power_w; digits=1), 10),
            lpad(s.temp_c, 8), lpad(s.pstate, 8),
            lpad(round(s.util_compute; digits=2), 6), lpad(s.nprocs, 7), "  ", s.events)
    end
end

# The discriminator. `cpu_frac` = host CPU time consumed during the trial /
# wall time. Near 1.0 means the host thread was running the whole trial
# (CPU-bound, or spinning in a non-blocking sync); near the fast-trial floor's
# absolute CPU cost means the host was blocked in the driver and the time went
# to the device/driver side. Compare the fast and slow populations directly:
# if slow trials burn proportionally more CPU, it's host-side.
function print_cpu_split(name::String, n::Int, dist::Symbol,
        times::Vector{Float64}, pcpu::Vector{Float64}, tcpu::Vector{Float64})
    println("  [cpu] $(name) (n=$n, $dist) per-trial wall vs host CPU:")
    println("  [cpu]  trial     wall_us   proc_cpu_us   thr_cpu_us   proc_frac   thr_frac")
    for i in eachindex(times)
        w = times[i]
        println("  [cpu]  ", lpad(i, 5), lpad(round(w / 1000; digits=1), 12),
            lpad(round(pcpu[i] / 1000; digits=1), 14),
            lpad(round(tcpu[i] / 1000; digits=1), 13),
            lpad(round(pcpu[i] / max(w, 1.0); digits=3), 12),
            lpad(round(tcpu[i] / max(w, 1.0); digits=3), 11))
    end
    # Split fast vs slow at 1.5x the observed floor and summarize, so the
    # answer doesn't depend on reading 100 rows by eye.
    floor_ns = minimum(times)
    fast = [i for i in eachindex(times) if times[i] < 1.5 * floor_ns]
    slow = [i for i in eachindex(times) if times[i] >= 1.5 * floor_ns]
    for (label, idx) in (("fast", fast), ("slow", slow))
        isempty(idx) && continue
        mw = median(times[idx]) / 1000
        mp = median(pcpu[idx]) / 1000
        mt = median(tcpu[idx]) / 1000
        println("  [cpu] $(name) (n=$n, $dist) $label (n=$(length(idx))): " *
                "median wall=$(round(mw; digits=1))μs proc_cpu=$(round(mp; digits=1))μs " *
                "thr_cpu=$(round(mt; digits=1))μs proc_frac=$(round(mp / mw; digits=3))")
    end
end

end # @static if !HAS_METAL

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
        nwarmup::Int=1, ntrials::Int=5, profile::Bool=false, telemetry::Bool=false)
    if !CUDA.functional()
        return nothing, nothing, nothing, nothing, nothing
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
    # stage_hist[stage] = per-trial ns for that stage (sort, build_leaves,
    # balance, finalize), only populated when profile=true.
    stage_hist = [Float64[] for _ in 1:4]
    used_hist = Int64[]
    cached_hist = Int64[]
    tele_hist = TrialTelemetry[]
    pcpu_hist = Float64[]
    tcpu_hist = Float64[]
    nvdev = telemetry ? nvml_device() : nothing
    ref_nodes, ref_leaves = nothing, nothing
    for trial in 1:ntrials
        stage_ns = profile ? zeros(UInt64, 4) : nothing
        CUDA.synchronize()
        p0 = process_cpu_ns()
        c0 = thread_cpu_ns()
        t0 = time_ns()
        local result
        alloc_bytes = CUDA.@allocated (result = ext.ka_build_adaptive_tree!(actx, dev_positions, ell_max, K_max, true, x_min, h0; stage_ns))
        CUDA.synchronize()
        t1 = time_ns()
        push!(pcpu_hist, Float64(process_cpu_ns() - p0))
        push!(tcpu_hist, Float64(thread_cpu_ns() - c0))
        push!(times, Float64(t1 - t0))
        push!(allocs, alloc_bytes)
        if profile
            for s in 1:4
                push!(stage_hist[s], Float64(stage_ns[s]))
            end
            push!(used_hist, CUDA.used_memory())
            push!(cached_hist, CUDA.cached_memory())
        end
        telemetry && push!(tele_hist, sample_nvml(nvdev))

        if trial == 1
            ref_nodes = result.n_nodes
            ref_leaves = result.n_leaves
        end
    end

    return times, ref_nodes, ref_leaves, allocs, stage_hist, used_hist, cached_hist,
        tele_hist, pcpu_hist, tcpu_hist
end

# ===== HPC-CUDA arm (native CUDA driver) =====

function benchmark_cuda_native(positions::Matrix{Float32}, ell_max::Int, K_max::Int;
        nwarmup::Int=1, ntrials::Int=5, telemetry::Bool=false)
    if !CUDA.functional()
        return nothing, nothing, nothing, nothing, nothing
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
    # stage_hist[stage] = per-trial ns (stage_ns[1]=key+sort, [2]=build_leaves,
    # [3]=balance, [4]=finalize — see _cuda_refresh_adaptive_tree!).
    stage_hist = [Float64[] for _ in 1:4]
    used_hist = Int64[]
    cached_hist = Int64[]
    tele_hist = TrialTelemetry[]
    pcpu_hist = Float64[]
    tcpu_hist = Float64[]
    nvdev = telemetry ? nvml_device() : nothing
    ref_nodes, ref_leaves = nothing, nothing
    for trial in 1:ntrials
        CUDA.synchronize()
        p0 = process_cpu_ns()
        c0 = thread_cpu_ns()
        t0 = time_ns()
        alloc_bytes = CUDA.@allocated begin
            source_bufs = FastMultipole._radix_cache_refresh_source_buffers!(ctx, (sys,), Float32)
            FastMultipole._radix_cache_collect_positions!(ctx, source_bufs)
            _ = FastMultipole._cuda_refresh_adaptive_tree!(ctx, actx, cache, n)
        end
        CUDA.synchronize()
        t1 = time_ns()
        push!(pcpu_hist, Float64(process_cpu_ns() - p0))
        push!(tcpu_hist, Float64(thread_cpu_ns() - c0))
        push!(times, Float64(t1 - t0))
        push!(allocs, alloc_bytes)
        for s in 1:4
            push!(stage_hist[s], Float64(actx.stage_ns[s]))
        end
        push!(used_hist, CUDA.used_memory())
        push!(cached_hist, CUDA.cached_memory())
        telemetry && push!(tele_hist, sample_nvml(nvdev))

        if trial == 1
            ref_nodes = actx.n_nodes
            ref_leaves = actx.n_leaves
        end
    end

    return times, ref_nodes, ref_leaves, allocs, stage_hist, used_hist, cached_hist,
        tele_hist, pcpu_hist, tcpu_hist
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

            # Profile per-stage timing at n>=1e5, where the timing anomaly lives.
            profile_this_case = n >= Int(1e5)

            # CUDA-KA arm
            if !isapple() && CUDA.functional()
                println("  Running CUDA-KA (HPC)...")
                cuda_ka_times, cuda_ka_nodes, cuda_ka_leaves, cuda_ka_allocs, cuda_ka_stages,
                    cuda_ka_used, cuda_ka_cached =
                    benchmark_cuda_ka(positions, ell_max, K_max; nwarmup, ntrials, profile=profile_this_case)
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
                    if profile_this_case
                        print_stage_breakdown("CUDA-KA", n, dist, cuda_ka_stages)
                        print_pool_samples("CUDA-KA", n, dist, cuda_ka_used, cuda_ka_cached)
                    end
                    push!(results, res)
                end
            end

            # CUDA-native arm
            if !isapple() && CUDA.functional()
                println("  Running CUDA-native (HPC)...")
                cuda_native_times, cuda_native_nodes, cuda_native_leaves, cuda_native_allocs, cuda_native_stages,
                    cuda_native_used, cuda_native_cached =
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
                    if profile_this_case
                        print_stage_breakdown("CUDA-native", n, dist, cuda_native_stages)
                        print_pool_samples("CUDA-native", n, dist, cuda_native_used, cuda_native_cached)
                    end
                    push!(results, res)
                end
            end

            # Reclaim device memory before the next test case. actx/cache device
            # buffers created above go out of scope when their benchmark_* function
            # returns, but are otherwise only freed whenever Julia's own GC pressure
            # happens to trigger — without this, they sit live for the rest of the
            # process, growing the shared CUDA caching allocator's live set across
            # cases and degrading later (larger-n) cases' allocator performance
            # (confirmed via job 13506207's gpu.csv: memory.used grew monotonically
            # 0->41GB across all 8 cases with no drops).
            GC.gc()
            if !isapple() && CUDA.functional()
                CUDA.reclaim()
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
