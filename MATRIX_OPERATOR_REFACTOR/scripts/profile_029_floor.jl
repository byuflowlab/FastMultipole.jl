# Task 029 step 2 (evidence-only): mechanism attribution of the launch/sync
# floor in the robust baseline config (sched6-5-4-4, ell=5, Float32, FP16-WMMA
# dense M2L, K=full, counting sort, 64/65536 tiled launch).
#
# Motivation: the 030 cost-vs-n table shows M2M+L2L is ~1.9 ms at ell=5 for
# EVERY n from 1e3 to 1e6 (an n-independent per-level floor), and the fresh 029
# baselines put M2M+L2L+coarse-M2L+refresh+finalize at ~3 ms of the 7.44 ms
# robust step. This script decides, per stage, how much of that wall time is
#   (a) device-busy kernel time      -> needs kernel/work changes,
#   (b) device-idle host/launch gap  -> CUDA graphs / launch fusion territory,
#   (c) synchronization API time     -> blocking-sync/stream fixes,
# using CUDA.@profile trace=true (CUPTI), per stage and for the whole verdict
# step, at n=1e6 and at an n=1e3 pure-floor control with identical geometry.
#
# NO production src/ changes; benchmark/evidence side only.
#
# Env knobs:
#   FM029P_N       comma list of body counts     (default "1000000,1000")
#   FM029P_POLICY  scheduled policy              (default "sched6-5-4-4")
#   FM029P_REPS    iterations inside each trace  (default "5")
#   FM029P_OUT     output CSV path prefix
#
# Outputs:
#   <OUT>.csv          per (n, stage): wall/device-busy/idle ms, launches,
#                      sync API count+ms, memcpy/memset counts, control wall
#   <OUT>.kernels.csv  per (n, stage, kernel): count + total device ms
#   stdout             the standard CUDA.@profile summary tables per stage

using FastMultipole
using FastMultipole.StaticArrays
using Statistics
using LinearAlgebra
using Dates
using Printf
using SHA

const REPO = normpath(joinpath(@__DIR__, "..", ".."))

FastMultipole.load_cuda_radix_lifecycle!() ||
    error("CUDA radix lifecycle failed to load: $(FastMultipole.cuda_radix_status())")
using CUDA

include(joinpath(@__DIR__, "fm028_device_system.jl"))

const FM = FastMultipole

const NS = parse.(Int, split(get(ENV, "FM029P_N", "1000000,1000"), ','))
const POLICY = get(ENV, "FM029P_POLICY", "sched6-5-4-4")
const REPS = parse(Int, get(ENV, "FM029P_REPS", "5"))
const ELL = 5
const P = 3
const STAMP = Dates.format(now(), "yyyymmdd-HHMMSS")
const OUT = get(ENV, "FM029P_OUT", joinpath(@__DIR__, "..", "data",
    "performance_high_score_1m_1ms",
    "cuda029_profile_$(gethostname())_$(STAMP)"))
const SEED = 24025
const BOX_MIN = SVector(-0.01, -0.01, -0.01)
const BOX_SIZE = 1.02
const DT = 1e-5

# production-default knobs of the 029 baseline record
FM.DENSE_CUDA_TILED_THREADS[] = 64
FM.DENSE_CUDA_TILED_MAX_BLOCKS[] = 65536
FM.RADIX_CUDA_COUNTING_SORT[] = true
FM.CUDA_SYMMETRIC_NEARFIELD[] = false
FM.DENSE_CUDA_TENSOR_FORMAT[] = :fp16

function _source_manifest()
    srcdir = joinpath(REPO, "src")
    files = sort(filter(f -> endswith(f, ".jl"), readdir(srcdir)))
    ctx = SHA.SHA256_CTX()
    for f in files
        SHA.update!(ctx, codeunits(f))
        SHA.update!(ctx, read(joinpath(srcdir, f)))
    end
    return bytes2hex(SHA.digest!(ctx))[1:16]
end

function _schedule_kwargs(policy, ell, ::Type{TF}) where TF
    qs = parse.(Int, split(policy[6:end], '-'))
    length(qs) == ell - 1 || error("policy $policy needs $(ell - 1) entries")
    q = last(qs)
    h0 = TF(BOX_SIZE / 2)
    eps = rigid_stencil_epsilon(P, h0, ell, q; lamb_helmholtz=false, TF)
    K = length(union((Set((j == 1 ? RigidHierarchicalTables(x) :
        FM._rigid_transition_tables(qs[j - 1], x)).push_offsets)
        for (j, x) in enumerate(qs))...))
    base = HierarchicalRigidStencil(ConstantPStencilConfig(P, eps;
        lamb_helmholtz=false); near_radius2=q, window_classes=K)
    return (; policy=FM._hierarchical_stencil_with_schedule(base, qs))
end

# ---- trace reduction --------------------------------------------------------

# merged union length (ms) of [start, stop] intervals given in seconds
function _busy_ms(starts, stops)
    isempty(starts) && return 0.0
    p = sortperm(collect(starts))
    s = collect(starts)[p]; e = collect(stops)[p]
    total = 0.0
    cs, ce = s[1], e[1]
    for i in 2:length(s)
        if s[i] > ce
            total += ce - cs
            cs, ce = s[i], e[i]
        else
            ce = max(ce, e[i])
        end
    end
    total += ce - cs
    return total * 1e3
end

_issync(name) = occursin("Synchronize", name) || occursin("cuCtxSynchronize", name)
_ismem(name) = occursin("Memcpy", name) || occursin("Memset", name)
# 029 cycle 1: host-side launch APIs — the graph-captured chain replaces many
# cuLaunchKernel calls with one cuGraphLaunch, which device-record counts
# (`launches` above) cannot see because the same kernels still execute
_islaunch(name) = occursin("Launch", name)
_isgraphlaunch(name) = occursin("GraphLaunch", name)

function _reduce_trace(prof, reps)
    dev = prof.device      # NamedTuple of vectors (CUDATools) or DataFrame
    host = prof.host
    n_launch = length(dev.name)
    busy = _busy_ms(dev.start, dev.stop)
    span = isempty(dev.start) ? 0.0 :
        (maximum(dev.stop) - minimum(dev.start)) * 1e3
    hostnames = String.(host.name)
    syncmask = _issync.(hostnames)
    memmask = _ismem.(hostnames)
    launchmask = _islaunch.(hostnames)
    graphmask = _isgraphlaunch.(hostnames)
    sync_ms = sum((host.stop .- host.start)[syncmask]; init=0.0) * 1e3
    host_span = isempty(host.start) ? 0.0 :
        (maximum(host.stop) - minimum(host.start)) * 1e3
    # per-kernel totals
    per = Dict{String,Tuple{Int,Float64}}()
    for i in 1:n_launch
        k = String(dev.name[i])
        c, t = get(per, k, (0, 0.0))
        per[k] = (c + 1, t + (dev.stop[i] - dev.start[i]) * 1e3)
    end
    return (;
        wall_ms=host_span / reps,
        device_busy_ms=busy / reps,
        device_span_ms=span / reps,
        idle_ms=max(host_span - busy, 0.0) / reps,
        launches=n_launch / reps,
        sync_calls=count(syncmask) / reps,
        sync_ms=sync_ms / reps,
        mem_ops=count(memmask) / reps,
        host_launches=count(launchmask) / reps,
        graph_launches=count(graphmask) / reps,
        per_kernel=per,
    )
end

# ---- per-config profiling ---------------------------------------------------

function profile_config(n, rows, krows)
    println("\n================ n=$n policy=$POLICY ================")
    bodies = fm028_body_matrix(SEED, n)
    sys = FM028DeviceSystem{Float32}(bodies)
    opts = CUDARadixLifecycleOptions(; precision=Float32,
        operator=MaterializedYRotationM2L(), m2l_strategy=DenseTranslationM2L())
    cache = RadixFMMCache(sys; expansion_order=P, ell=ELL, max_n_bodies=n,
        bounds=(BOX_MIN, BOX_SIZE), lamb_helmholtz=false, device=true,
        options=opts, _schedule_kwargs(POLICY, ELL, Float32)...)
    CUDA.synchronize()
    state = cache.state
    switches = (FM.DerivativesSwitch(true, true, false, sys),)
    step!() = (fmm!(sys, cache; scalar_potential=true, gradient=true);
        fm028_euler!(sys, DT, 0.0, 1.0))

    stages = [
        ("refresh", () -> FM.update_cuda_radix_state!(cache, (sys,))),
        ("b2m", () -> FM._launch_cuda_b2m!(state)),
        ("m2m", () -> FM._launch_cuda_resident_m2m!(state)),
        ("m2l", () -> FM._launch_cuda_resident_m2l!(state)),
        ("l2l", () -> FM._launch_cuda_resident_l2l!(state)),
        ("l2b_near", () -> FM._launch_cuda_resident_l2b!(state)),
        ("eval", () -> FM.run_cuda_radix_lifecycle!(state)),
        ("finalize", () -> FM.finalize_cuda_radix_output!(state, (sys,);
            derivatives_switches=switches)),
        ("euler", () -> fm028_euler!(sys, DT, 0.0, 1.0)),
        ("full_step", step!),
    ]

    # warm everything (JIT + caches) before any trace
    step!(); CUDA.synchronize()
    for (_, f!) in stages
        f!(); CUDA.synchronize()
    end

    for (name, f!) in stages
        # control wall time without tracing
        f!(); CUDA.synchronize()
        ctrl = [(@elapsed (f!(); CUDA.synchronize())) * 1e3 for _ in 1:REPS]
        prof = CUDA.@profile trace=true begin
            for _ in 1:REPS
                f!()
                CUDA.synchronize()
            end
        end
        r = _reduce_trace(prof, REPS)
        @printf("%-10s wall %8.3f ms  busy %8.3f  idle %8.3f  launches %7.1f (host %7.1f, graph %5.1f)  syncs %5.1f (%.3f ms)  ctrl %8.3f ms\n",
            name, r.wall_ms, r.device_busy_ms, r.idle_ms, r.launches,
            r.host_launches, r.graph_launches,
            r.sync_calls, r.sync_ms, median(ctrl))
        push!(rows, (n=n, stage=name, wall_ms=r.wall_ms,
            device_busy_ms=r.device_busy_ms, device_span_ms=r.device_span_ms,
            idle_ms=r.idle_ms, launches=r.launches, sync_calls=r.sync_calls,
            sync_ms=r.sync_ms, mem_ops=r.mem_ops, control_wall_ms=median(ctrl),
            host_launches=r.host_launches, graph_launches=r.graph_launches))
        for (k, (c, t)) in sort(collect(r.per_kernel); by=x -> -x[2][2])
            push!(krows, (n=n, stage=name, kernel=k, count=c / REPS,
                total_ms=t / REPS))
        end
        # full summary table for the two most informative stages
        if name in ("m2m", "full_step")
            println("---- CUDA.@profile summary for $name (REPS=$REPS) ----")
            show(stdout, prof)
            println()
        end
    end

    # free device memory before the next config
    cache = nothing; sys = nothing; state = nothing
    GC.gc(); CUDA.reclaim()
end

# ---- main -------------------------------------------------------------------

println("SOURCE_MANIFEST=", _source_manifest())
println("julia=", VERSION, " CUDA_runtime=", CUDA.runtime_version(),
    " gpu=", CUDA.name(CUDA.device()), " host=", gethostname(),
    " job=", get(ENV, "SLURM_JOB_ID", ""))

rows = NamedTuple[]
krows = NamedTuple[]
for n in NS
    profile_config(n, rows, krows)
end

open(OUT * ".csv", "w") do io
    println(io, "n,stage,wall_ms,device_busy_ms,device_span_ms,idle_ms," *
        "launches,sync_calls,sync_ms,mem_ops,control_wall_ms," *
        "host_launches,graph_launches")
    for r in rows
        println(io, join([r.n, r.stage, r.wall_ms, r.device_busy_ms,
            r.device_span_ms, r.idle_ms, r.launches, r.sync_calls, r.sync_ms,
            r.mem_ops, r.control_wall_ms, r.host_launches, r.graph_launches], ','))
    end
end
open(OUT * ".kernels.csv", "w") do io
    println(io, "n,stage,kernel,count,total_ms")
    for r in krows
        println(io, join([r.n, r.stage, "\"" * replace(r.kernel, '"' => "'") *
            "\"", r.count, r.total_ms], ','))
    end
end
println("WROTE ", OUT, ".csv")
println("PROFILE_EXIT=0")
