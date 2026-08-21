# 028 Phase B step 2 — de-risking measurements.
#
# Phase A left three load-bearing claims as inferences. Each one changes what
# the optimization work should target, and all three resolve without touching
# `src/`. This script answers them:
#
#   A. Is the fused L2B+nearfield stage really ~85% nearfield?
#      Phase A inferred the split from an ell-sweep fit (2 params, 3 points,
#      residuals up to 11%). `_launch_cuda_resident_l2b!` gates the nearfield
#      kernel on `state.counts.n_direct`, so setting it to 0 launches L2B alone
#      and gives an EXACT split by difference.
#
#   B. Where exactly are the 57.7 MB/step of host allocation?
#      Phase A localized it to the per-cell refresh path by scaling analysis
#      (88% cell-proportional) but could not name the line. `Profile.Allocs`
#      over one `update_cuda_radix_state!` names it.
#
#   C. Why is leaf M2L overhead-bound?
#      A precision A/B proved it is neither bandwidth- nor compute-bound
#      (22.41 ms F64 -> 23.55 ms F32, unchanged). `CUDA.@profile trace=true`
#      gives the per-kernel table (launch counts, durations, occupancy) that
#      says what it actually is.
#
# No `src/` changes. Usage mirrors the feasibility benchmark:
#   FM028_N=1000000 julia --project=$ENVDIR .../benchmark_028_derisk.jl

using FastMultipole
using FastMultipole.StaticArrays
using Statistics
using LinearAlgebra
using Dates
using Printf
using Profile

const REPO = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(REPO, "test", "gravitational.jl"))

FastMultipole.load_cuda_radix_lifecycle!() ||
    error("CUDA radix lifecycle failed to load: $(FastMultipole.cuda_radix_status())")
using CUDA

include(joinpath(@__DIR__, "fm028_device_system.jl"))
const FM = FastMultipole

const N     = parse(Int, get(ENV, "FM028_N", "1000000"))
const P     = parse(Int, get(ENV, "FM028_P", "3"))
const ELLS  = parse.(Int, split(get(ENV, "FM028_ELL", "4,5,6"), ','))
const K     = parse(Int, get(ENV, "FM028_K", "1740"))
const REPS  = parse(Int, get(ENV, "FM028_REPS", "5"))
const SEED  = 24025
const BOX_MIN  = SVector(-0.01, -0.01, -0.01)
const BOX_SIZE = 1.02
const STAMP = Dates.format(now(), "yyyymmdd-HHMMSS")
const OUT = get(ENV, "FM028_OUT", joinpath(@__DIR__, "..", "data",
    "feasibility_1m_10ms", "derisk_$(gethostname())_$(STAMP).csv"))
const JOBID = get(ENV, "SLURM_JOB_ID", "")

# dense = the verdict config's strategy; operator/strategy are objects, not
# symbols (same spec table as benchmark_028_feasibility.jl)
_opts(TF) = CUDARadixLifecycleOptions(; precision=TF,
    operator=MaterializedYRotationM2L(), m2l_strategy=DenseTranslationM2L())
_kw(ell) = (; near_radius2=12, window_classes=K)

# identical to benchmark_028_feasibility.jl so Part A timings are directly
# comparable to the Phase A l2b_ms figures
function _median_gpu_ms(f!, state, reps)
    f!(state); CUDA.synchronize()
    ts = Float64[]
    for _ in 1:reps
        push!(ts, Float64(CUDA.@elapsed f!(state)) * 1e3)
    end
    return median(ts)
end

function build(::Type{TF}, ell) where TF
    bodies = fm028_body_matrix(SEED, N)
    sys = FM028DeviceSystem{TF}(bodies)
    cache = RadixFMMCache(sys; expansion_order=P, ell, max_n_bodies=N,
        bounds=(BOX_MIN, BOX_SIZE), lamb_helmholtz=false, device=true,
        options=_opts(TF), _kw(ell)...)
    fmm!(sys, cache; scalar_potential=true, gradient=true)
    return sys, cache
end

# ---- A. exact L2B / nearfield split ----------------------------------------

println("\n", "="^78)
println("A. EXACT L2B / nearfield split  (n=$N, P=$P, K=$K, hier12/dense)")
println("="^78)
println(@sprintf("%-8s %-4s %10s %10s %10s %8s %8s %14s",
    "prec", "ell", "fused_ms", "l2b_ms", "near_ms", "near_%", "cells", "interactions"))
rows = NamedTuple[]
for TF in (Float64, Float32), ell in ELLS
    TF === Float32 && ell != 5 && continue     # F32 only at the verdict ell
    sys, cache = build(TF, ell)
    state = cache.state
    fused = _median_gpu_ms(FM._launch_cuda_resident_l2b!, state, REPS)
    # gate the nearfield kernel off: direct_blocks = cld(n_direct, threads) = 0
    saved = state.counts.n_direct
    state.counts.n_direct = 0
    l2b = _median_gpu_ms(FM._launch_cuda_resident_l2b!, state, REPS)
    state.counts.n_direct = saved
    near = fused - l2b
    cells = cache.state.counts.n_cells
    bpc = N / cells
    inter = saved * bpc * bpc
    println(@sprintf("%-8s %-4d %10.3f %10.3f %10.3f %7.1f%% %8d %14.3e",
        TF, ell, fused, l2b, near, 100 * near / fused, cells, inter))
    push!(rows, (; job=JOBID, precision=string(TF), n=N, ell, window_classes=K,
        n_cells=cells, n_direct=saved, interactions=inter,
        fused_ms=fused, l2b_ms=l2b, near_ms=near,
        near_frac=near / fused,
        g_interactions_per_s=inter / near / 1e6))
    GC.gc(); CUDA.reclaim()
end
mkpath(dirname(OUT))
open(OUT, "w") do io
    println(io, join(string.(keys(rows[1])), ','))
    for r in rows; println(io, join(string.(values(r)), ',')); end
end
println("wrote ", OUT)

# ---- B. host allocation profile of the per-step refresh ---------------------

println("\n", "="^78)
println("B. HOST ALLOCATION PROFILE — update_cuda_radix_state! (ell=5)")
println("="^78)
function alloc_report(label, thunk)
    thunk()                                             # warm
    bytes = @allocated thunk()
    println("\n---- ", label)
    @printf("measured host allocation: %.2f MB\n", bytes / 1e6)
    Profile.Allocs.clear()
    try
    Profile.Allocs.@profile sample_rate=1.0 thunk()
    res = Profile.Allocs.fetch()
    @printf("sampled %d allocations, %.2f MB total\n\n",
        length(res.allocs), sum(a.size for a in res.allocs) / 1e6)

    agg = Dict{String,NTuple{2,Int}}()
    for a in res.allocs
        # attribute to the shallowest frame inside FastMultipole/ the scripts
        loc = "unknown"
        for fr in a.stacktrace
            f = string(fr.file)
            if occursin("FastMultipole", f) || occursin("translate_batched", f)
                loc = "$(basename(f)):$(fr.line) $(fr.func)"
                break
            end
        end
        c, s = get(agg, loc, (0, 0))
        agg[loc] = (c + 1, s + a.size)
    end
    println("top allocation sites by bytes:")
    @printf("  %-52s %8s %12s\n", "site", "count", "MB")
    for (loc, (c, s)) in first(sort(collect(agg); by = kv -> -kv[2][2]), 20)
        @printf("  %-52s %8d %12.3f\n", first(loc, 52), c, s / 1e6)
    end

    println("\ntop allocation types by bytes:")
    tagg = Dict{String,NTuple{2,Int}}()
    for a in res.allocs
        k = string(a.type)
        c, s = get(tagg, k, (0, 0)); tagg[k] = (c + 1, s + a.size)
    end
    for (t, (c, s)) in first(sort(collect(tagg); by = kv -> -kv[2][2]), 12)
        @printf("  %-52s %8d %12.3f\n", first(t, 52), c, s / 1e6)
    end
    catch err
        println("alloc profile failed: ", first(sprint(showerror, err), 300))
    end
    return nothing
end

let
    sys, cache = build(Float64, 5)
    state = cache.state
    switches = (FM.DerivativesSwitch(true, true, false, sys),)
    # Phase A measured 57.7 MB for the WHOLE verdict step; the refresh alone is
    # only ~8 MB, so the cell-proportional bulk must live in the lifecycle.
    # Profile each component of the step separately to place it.
    alloc_report("refresh: update_cuda_radix_state!",
        () -> FM.update_cuda_radix_state!(cache, (sys,)))
    alloc_report("lifecycle: run_cuda_radix_lifecycle!",
        () -> FM.run_cuda_radix_lifecycle!(state))
    alloc_report("M2L stage alone: _launch_cuda_resident_m2l!",
        () -> FM._launch_cuda_resident_m2l!(state))
    alloc_report("finalize: finalize_cuda_radix_output!",
        () -> FM.finalize_cuda_radix_output!(state, (sys,);
            derivatives_switches=switches))
    alloc_report("full verdict step",
        () -> (FM.update_cuda_radix_state!(cache, (sys,));
               FM.run_cuda_radix_lifecycle!(state);
               FM.finalize_cuda_radix_output!(state, (sys,);
                   derivatives_switches=switches);
               fm028_euler!(sys, 1e-5, 0.0, 1.0)))
    GC.gc(); CUDA.reclaim()
end

# ---- C. per-kernel trace of the lifecycle ----------------------------------

println("\n", "="^78)
println("C. PER-KERNEL TRACE — one full lifecycle (ell=5, F64)")
println("   look for: leaf-M2L kernel launch count, duration, occupancy, regs")
println("="^78)
let
    sys, cache = build(Float64, 5)
    state = cache.state
    FM.run_cuda_radix_lifecycle!(state); CUDA.synchronize()
    # CUPTI tracing can be restricted on shared cluster nodes; parts A and B
    # are the load-bearing results, so never let C take the job down
    # CUDA.@profile returns a results object that only renders when displayed;
    # in a script the return value is otherwise discarded (this printed nothing
    # in job 12998146).
    show_trace(label, res) = (println("\n--- ", label);
        show(stdout, MIME"text/plain"(), res); println())
    try
        show_trace("full lifecycle",
            CUDA.@profile trace=true FM.run_cuda_radix_lifecycle!(state))
        show_trace("M2L stage alone (isolates the leaf apply)",
            CUDA.@profile trace=true FM._launch_cuda_resident_m2l!(state))
        show_trace("fused L2B+nearfield stage alone",
            CUDA.@profile trace=true FM._launch_cuda_resident_l2b!(state))
    catch err
        println("CUPTI trace unavailable: ", first(sprint(showerror, err), 300))
    end
    GC.gc(); CUDA.reclaim()
end

# ---- D. leaf M2L: atomic-throughput vs bandwidth A/B (028 lever-2 follow-up) --
#
# After the lever-2 grid-stride fix the leaf M2L became precision-SENSITIVE
# (23.27 ms F64 vs 19.60 ms F32), i.e. limited by bandwidth or by atomic
# throughput (~500M atomic accumulations/step); the two were indistinguishable.
# Discriminator: relaunch the leaf window's fused kernel on throwaway locals
# with (a) the production atomics, (b) plain stores (same loads/flops/traffic,
# no atomics; results are garbage — timing only), (c) no stores at all
# (loads+flops only). near(b) ≈ (a) → not atomic-bound; (b) ≪ (a) → atomic-bound.

function _derisk_fused_kernel_store!(loc_phi, loc_chi, ops, route_class,
        route_sources, route_targets, mp_phi, mp_chi, phi_flat_idx, chi_flat_idx,
        ndof_phi, src_scale, tgt_scale, lcol, n_routes, ::Val{LH}) where LH
    T = eltype(ops)
    D = size(ops, 1)
    tid = threadIdx().x
    nthreads = blockDim().x
    shm = CUDA.CuDynamicSharedArray(T, D)
    j = blockIdx().x
    @inbounds while j <= n_routes
        src_col = route_sources[j]
        k = Int(route_class[j])
        tgt_col = route_targets[j]
        i = tid
        while i <= D
            if i <= ndof_phi
                shm[i] = mp_phi[phi_flat_idx[i], src_col] * src_scale[i, lcol]
            elseif LH
                shm[i] = mp_chi[chi_flat_idx[i - ndof_phi], src_col] * src_scale[i, lcol]
            else
                shm[i] = zero(T)
            end
            i += nthreads
        end
        CUDA.sync_threads()
        r = tid
        while r <= D
            acc = zero(T)
            for i in 1:D
                acc += ops[r, i, k] * shm[i]
            end
            acc *= tgt_scale[r, lcol]
            # plain racing stores: identical loads and address stream, no atomics
            if r <= ndof_phi
                loc_phi[phi_flat_idx[r], tgt_col] = acc
            elseif LH
                loc_chi[chi_flat_idx[r - ndof_phi], tgt_col] = acc
            end
            r += nthreads
        end
        CUDA.sync_threads()
        j += gridDim().x
    end
    return nothing
end

function _derisk_fused_kernel_nostore!(loc_phi, loc_chi, ops, route_class,
        route_sources, route_targets, mp_phi, mp_chi, phi_flat_idx, chi_flat_idx,
        ndof_phi, src_scale, tgt_scale, lcol, n_routes, ::Val{LH}) where LH
    T = eltype(ops)
    D = size(ops, 1)
    tid = threadIdx().x
    nthreads = blockDim().x
    shm = CUDA.CuDynamicSharedArray(T, D)
    j = blockIdx().x
    @inbounds while j <= n_routes
        src_col = route_sources[j]
        k = Int(route_class[j])
        i = tid
        while i <= D
            if i <= ndof_phi
                shm[i] = mp_phi[phi_flat_idx[i], src_col] * src_scale[i, lcol]
            elseif LH
                shm[i] = mp_chi[chi_flat_idx[i - ndof_phi], src_col] * src_scale[i, lcol]
            else
                shm[i] = zero(T)
            end
            i += nthreads
        end
        CUDA.sync_threads()
        r = tid
        while r <= D
            acc = zero(T)
            for i in 1:D
                acc += ops[r, i, k] * shm[i]
            end
            acc *= tgt_scale[r, lcol]
            # defeat DCE without a memory write on any realizable value
            acc == T(Inf) && (loc_phi[1, route_targets[j]] = acc)
            r += nthreads
        end
        CUDA.sync_threads()
        j += gridDim().x
    end
    return nothing
end

println("\n", "="^78)
println("D. LEAF M2L ATOMIC vs BANDWIDTH A/B  (ell=5, K=$K, hier12/dense)")
println("="^78)
println(@sprintf("%-8s %10s %12s %12s %12s", "prec", "routes", "atomic_ms",
    "store_ms", "nostore_ms"))
for TF in (Float64, Float32)
    sys, cache = build(TF, 5)
    state = cache.state
    hctx = cache.device_ctx.hierarchical_ctx
    ws = state.scratch
    plan = hctx.apply_plan
    L = hctx.ell
    noffsets = hctx.noffsets
    if noffsets > hctx.window_classes
        println("SKIP $(TF): leaf level spans multiple windows (noffsets=$noffsets > K)")
        continue
    end
    # regenerate the (single) leaf-level window into the reusable route buffers
    n = FM._cuda_hier_generate_window!(state, hctx, plan.route_class, L, 1,
        noffsets, 0)
    FM._cuda_hier_refresh_dense_window!(plan, hctx, 1, noffsets, n)
    state.counts.n_routes = n
    scratch_phi = CUDA.zeros(TF, size(state.locals.phi))
    scratch_chi = CUDA.zeros(TF, size(state.locals.chi))
    lcol = L - 1
    threads = min(256, cld(plan.ndof, 32) * 32)
    shmem = plan.ndof * sizeof(TF)
    blocks = min(n, FM.DENSE_CUDA_FUSED_MAX_BLOCKS[])
    args = (plan.operators, plan.route_class, state.route_sources,
        state.route_targets, state.multipoles.phi, state.multipoles.chi,
        ws.phi_flat_idx, ws.chi_flat_idx, plan.ndof_phi,
        hctx.source_scale, hctx.target_scale, lcol, n, Val(false))
    launch_atomic!(_) = CUDA.@cuda threads=threads blocks=blocks shmem=shmem FM._cuda_hier_dense_fused_kernel!(
        scratch_phi, scratch_chi, args...)
    launch_store!(_) = CUDA.@cuda threads=threads blocks=blocks shmem=shmem _derisk_fused_kernel_store!(
        scratch_phi, scratch_chi, args...)
    launch_nostore!(_) = CUDA.@cuda threads=threads blocks=blocks shmem=shmem _derisk_fused_kernel_nostore!(
        scratch_phi, scratch_chi, args...)
    t_atomic = _median_gpu_ms(launch_atomic!, nothing, REPS)
    t_store = _median_gpu_ms(launch_store!, nothing, REPS)
    t_nostore = _median_gpu_ms(launch_nostore!, nothing, REPS)
    println(@sprintf("%-8s %10d %12.3f %12.3f %12.3f", TF, n, t_atomic, t_store,
        t_nostore))
    GC.gc(); CUDA.reclaim()
end

println("\ndone.")
