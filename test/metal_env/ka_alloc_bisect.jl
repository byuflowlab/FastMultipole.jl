# Bisect the ~1.95GB/trial CUDA-KA device allocation (job 13506036,
# tree_build_benchmark_4way.jl n=1e6) between the two library scan/sort
# primitives implicated by the leading hypothesis: `sortperm!` (full-depth
# Morton key sort, called once per trial in `ka_build_adaptive_tree!` plus
# once each inside `ka_adaptive_balance!`/`ka_adaptive_finalize!`) and
# `accumulate!` (inclusive scan inside `_ka_scan_total!`, called ~14x/trial
# per project_fastmultipole_ka_migration memory).
#
# Reports two kinds of numbers side by side:
#   1. isolated: `sortperm!`/`accumulate!` timed alone on preallocated
#      dest/src buffers at the real problem's array sizes -- isolates the
#      primitive's own scratch allocation from any surrounding KA.zeros(...)
#      fresh-buffer churn.
#   2. phase totals: each of `ka_adaptive_build_leaves!` / `ka_adaptive_balance!`
#      / `ka_adaptive_finalize!` timed as a whole (as `ka_build_adaptive_tree!`
#      calls them) -- includes both their internal sortperm!/accumulate! calls
#      AND their own fresh KA.zeros(...) buffers, for cross-checking that the
#      isolated numbers add up to close to the full pipeline's ~1.95GB.

using FastMultipole
using KernelAbstractions
using Random
using CUDA

const KA = KernelAbstractions

FastMultipole.load_cuda_radix_lifecycle!()
include("../gravitational.jl")

const ext = Base.get_extension(FastMultipole, :FastMultipoleKAExt)
ext !== nothing || error("FastMultipoleKAExt did not load")

function generate_uniform_random(n::Int, seed::Int)
    Random.seed!(seed)
    positions = rand(Float32, 3, n)
    positions .*= 2.0f0
    return positions
end

function report(label::String, bytes::Int64)
    println("  $(label): $(round(bytes / 1024^2; digits=3)) MB")
end

function main()
    CUDA.functional() || error("CUDA not functional")

    n = Int(1e6)
    ell_max = 6
    K_max = 16
    x_min = (0.0f0, 0.0f0, 0.0f0)
    h0 = 1.0f0
    ntrials = 20

    positions = generate_uniform_random(n, 42)
    dev_positions = CUDA.CuArray(positions)

    nl_estimate = max(2 * n ÷ K_max, 16)
    node_capacity = 100 * nl_estimate + 256
    leaf_capacity = 10 * nl_estimate + 256
    frontier_capacity = 16 * leaf_capacity

    println("=== KA alloc bisect: n=$n, ell_max=$ell_max, K_max=$K_max, ",
        "leaf_capacity=$leaf_capacity, frontier_capacity=$frontier_capacity, ",
        "node_capacity=$node_capacity, ntrials=$ntrials ===\n")

    # ----- Part 1: isolated primitive allocation, preallocated buffers -----
    println("--- Part 1: isolated sortperm!/accumulate! (preallocated dest/src) ---")

    keys = KA.zeros(CUDABackend(), UInt64, n)
    ext.ka_radix_keys!(keys, dev_positions, x_min, h0, ell_max)
    perm_full = KA.zeros(CUDABackend(), Int, n)

    sortperm_full_allocs = Int64[]
    for _ in 1:ntrials
        CUDA.synchronize()
        push!(sortperm_full_allocs, CUDA.@allocated sortperm!(perm_full, keys))
    end
    CUDA.synchronize()
    report("sortperm!(perm, keys) at n=$n (full-depth key sort)", round(Int64, sum(sortperm_full_allocs) / ntrials))

    # accumulate! at leaf-scale (nl_estimate) and frontier-scale sizes, the two
    # regimes _ka_scan_total! actually runs at inside build_leaves/balance/finalize.
    for m in (nl_estimate, frontier_capacity)
        fv = KA.zeros(CUDABackend(), Int32, m)
        pv = KA.zeros(CUDABackend(), Int32, m)
        fill!(fv, Int32(1))
        accs = Int64[]
        for _ in 1:ntrials
            CUDA.synchronize()
            push!(accs, CUDA.@allocated accumulate!(+, pv, fv))
        end
        CUDA.synchronize()
        report("accumulate!(+, pv, fv) at m=$m", round(Int64, sum(accs) / ntrials))
    end

    println()

    # ----- Part 2: phase totals (as called by ka_build_adaptive_tree!) -----
    println("--- Part 2: whole-phase totals (internal sortperm!/accumulate! + KA.zeros) ---")

    radix_allocs = Int64[]
    sortperm_step_allocs = Int64[]
    gather_allocs = Int64[]
    build_leaves_allocs = Int64[]
    balance_allocs = Int64[]
    finalize_allocs = Int64[]

    actx = ext.ka_allocate_adaptive_context(CUDABackend(), Float32, n;
        leaf_capacity, frontier_capacity, node_capacity)

    for trial in 1:ntrials
        keys_t = KA.zeros(CUDABackend(), UInt64, n)
        CUDA.synchronize()
        push!(radix_allocs, CUDA.@allocated ext.ka_radix_keys!(keys_t, dev_positions, x_min, h0, ell_max))

        perm_t = KA.zeros(CUDABackend(), Int, n)
        CUDA.synchronize()
        push!(sortperm_step_allocs, CUDA.@allocated sortperm!(perm_t, keys_t))

        sorted_keys_t = KA.zeros(CUDABackend(), UInt64, n)
        CUDA.synchronize()
        push!(gather_allocs, CUDA.@allocated ext.ka_gather_values!(sorted_keys_t, keys_t, perm_t))

        CUDA.synchronize()
        local nl, llev, lkey, llo, lhi
        push!(build_leaves_allocs, CUDA.@allocated ((nl, llev, lkey, llo, lhi) =
            ext.ka_adaptive_build_leaves!(actx, sorted_keys_t, ell_max, K_max, n)))

        CUDA.synchronize()
        local nl2, n_splits, llev2, lkey2, llo2, lhi2
        push!(balance_allocs, CUDA.@allocated ((nl2, n_splits, llev2, lkey2, llo2, lhi2) =
            ext.ka_adaptive_balance!(actx, nl, llev, lkey, llo, lhi, sorted_keys_t, ell_max)))

        CUDA.synchronize()
        push!(finalize_allocs, CUDA.@allocated ext.ka_adaptive_finalize!(actx, nl2, llev2, lkey2, llo2, lhi2,
            sorted_keys_t, ell_max, n, x_min, h0))
        CUDA.synchronize()
    end

    report("ka_radix_keys! (avg/trial)", round(Int64, sum(radix_allocs) / ntrials))
    report("sortperm! step, n=$n keys (avg/trial)", round(Int64, sum(sortperm_step_allocs) / ntrials))
    report("ka_gather_values! sorted_keys (avg/trial)", round(Int64, sum(gather_allocs) / ntrials))
    report("ka_adaptive_build_leaves! total (avg/trial)", round(Int64, sum(build_leaves_allocs) / ntrials))
    report("ka_adaptive_balance! total (avg/trial)", round(Int64, sum(balance_allocs) / ntrials))
    report("ka_adaptive_finalize! total (avg/trial)", round(Int64, sum(finalize_allocs) / ntrials))

    total = sum(radix_allocs) + sum(sortperm_step_allocs) + sum(gather_allocs) +
            sum(build_leaves_allocs) + sum(balance_allocs) + sum(finalize_allocs)
    report("SUM of all phases (avg/trial)", round(Int64, total / ntrials))
    println("\n(compare against job 13506036's measured ~1.95GB/trial for the full ka_build_adaptive_tree! call)")

    # ----- Part 3: residual ~18.1KB/trial after the KAAdaptiveTreeContext fix -----
    # Production code (ka_build_adaptive_tree! / ka_adaptive_balance! / ka_adaptive_finalize!)
    # calls `sortperm!(view(dest,1:nl), view(src,1:nl))` on views into actx buffers, not
    # plain CuArrays -- Part 1 above only tested sortperm! on plain CuArrays (0.0 MB).
    # Test whether CUDA.jl's sortperm! falls back to an allocating generic path for
    # SubArray (view) arguments, which plain-array dispatch avoids.
    println("\n--- Part 3: sortperm! on views vs. plain CuArrays (job 13506207 residual) ---")

    for m in (Int(1e6), nl_estimate)
        keys_plain = KA.zeros(CUDABackend(), UInt64, m)
        rand!(keys_plain)
        perm_plain = KA.zeros(CUDABackend(), Int, m)
        plain_allocs = Int64[]
        for _ in 1:ntrials
            CUDA.synchronize()
            push!(plain_allocs, CUDA.@allocated sortperm!(perm_plain, keys_plain))
        end
        CUDA.synchronize()
        report("sortperm!(plain CuArray, plain CuArray) at m=$m", round(Int64, sum(plain_allocs) / ntrials))

        keys_big = KA.zeros(CUDABackend(), UInt64, m + 256)
        rand!(view(keys_big, 1:m))
        perm_big = KA.zeros(CUDABackend(), Int, m + 256)
        view_allocs = Int64[]
        for _ in 1:ntrials
            CUDA.synchronize()
            push!(view_allocs, CUDA.@allocated sortperm!(view(perm_big, 1:m), view(keys_big, 1:m)))
        end
        CUDA.synchronize()
        report("sortperm!(view, view) at m=$m (production pattern)", round(Int64, sum(view_allocs) / ntrials))
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
