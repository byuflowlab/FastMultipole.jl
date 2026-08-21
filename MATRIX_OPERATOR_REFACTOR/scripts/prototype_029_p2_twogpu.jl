# Task 029 prototype P2: 2-GPU octant-decomposition feasibility slice.
#
# Frozen workload (n=1e6, seeds 24025/24026, bounds (-0.01, 1.02), P=4
# literature = expansion_order 3, sched6-5-4-4 robust geometry, ell=5, FP16
# tensor format, K=full). Scheme, gates, and record requirements: see
# fm029_p2_common.jl and the 029 task file P2 section.
#
# Phases:
#   0. environment/provenance + P2P copy microbenchmark (both directions)
#   1. single-GPU in-job reference: stock cycle-1 defaults, verdict boundary,
#      REPS samples (efficiency denominator on the SAME node; the 4.657 ms
#      record is also reported against)
#   2. dual-GPU setup: mirrored caches, partition-exactness gate, warm+record
#   3. step-0 sampled accuracy (union output) vs the checksummed 024b
#      reference, on BOTH GPUs, plus mirrored-output lockstep check
#   4. verdict loop: REPS complete steps (refresh + graphs + exchange +
#      finalize + Euler + all sync), full segment telemetry
#   5. 5-step convection re-check vs a fresh on-device Float64 direct
#      reference at the moved positions + bitwise lockstep re-check
#   6. counter contract, memory, CSV
#
# Env knobs: FM029P2_N (1000000), FM029P2_REPS (25), FM029P2_STEPS (5),
# FM029P2_DT (1e-5), FM029P2_ELL (5), FM029P2_P (3), FM029P2_POLICY
# (sched6-5-4-4), FM029P2_TF (Float32), FM029P2_TENSOR (fp16), FM029P2_OUT,
# FM029P2_REFDIR, FM029P2_GRAPH (1).

using FastMultipole
using FastMultipole.StaticArrays
using Statistics
using LinearAlgebra
using Dates
using Printf
using SHA

const REPO = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(REPO, "test", "gravitational.jl"))
include(joinpath(@__DIR__, "benchmark_024b_common.jl"))

FastMultipole.load_cuda_radix_lifecycle!() ||
    error("CUDA radix lifecycle failed to load: $(FastMultipole.cuda_radix_status())")
using CUDA

include(joinpath(@__DIR__, "fm028_device_system.jl"))
include(joinpath(@__DIR__, "fm029_p2_common.jl"))

const FM = FastMultipole

Threads.nthreads() >= 3 || error("run with julia -t >= 3 (got $(Threads.nthreads()))")
length(CUDA.devices()) >= 2 || error("P2 needs 2 GPUs; visible: $(length(CUDA.devices()))")

const N = parse(Int, get(ENV, "FM029P2_N", "1000000"))
const REPS = parse(Int, get(ENV, "FM029P2_REPS", "25"))
const STEPS = parse(Int, get(ENV, "FM029P2_STEPS", "5"))
const DT = parse(Float64, get(ENV, "FM029P2_DT", "1e-5"))
const ELL = parse(Int, get(ENV, "FM029P2_ELL", "5"))
const P = parse(Int, get(ENV, "FM029P2_P", "3"))
const POLICY = get(ENV, "FM029P2_POLICY", "sched6-5-4-4")
const TF = get(ENV, "FM029P2_TF", "Float32") == "Float32" ? Float32 : Float64
const TENSOR = Symbol(get(ENV, "FM029P2_TENSOR", "fp16"))
const USE_GRAPH = get(ENV, "FM029P2_GRAPH", "1") == "1"
const STAMP = Dates.format(now(), "yyyymmdd-HHMMSS")
const OUT = get(ENV, "FM029P2_OUT", joinpath(@__DIR__, "..", "data",
    "performance_high_score_1m_1ms", "cuda029p2_$(gethostname())_$(STAMP).csv"))
const REFDIR = get(ENV, "FM029P2_REFDIR", joinpath(@__DIR__, "..", "data",
    "cpu_gpu_scaling", "references"))
const RECORD_1GPU_MS = 4.657   # leaderboard row 2, robust sched6-5-4-4, job 13060804

# production kernel knobs, identical to the cycle-1 leaderboard rows
FM.DENSE_CUDA_TILED_THREADS[] = 64
FM.DENSE_CUDA_TILED_MAX_BLOCKS[] = 65536
FM.RADIX_CUDA_COUNTING_SORT[] = true
FM.CUDA_SYMMETRIC_NEARFIELD[] = false
FM.DENSE_CUDA_TENSOR_FORMAT[] = TENSOR

const SEED = 24025
const SAMPLER_SEED = 24026
const BOX_MIN = SVector(-0.01, -0.01, -0.01)
const BOX_SIZE = 1.02

# ---- provenance --------------------------------------------------------------

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
const MANIFEST = _source_manifest()
const JOBID = get(ENV, "SLURM_JOB_ID", "")
println("P2 manifest=$MANIFEST job=$JOBID node=$(gethostname()) julia=$VERSION ",
    "cuda=$(CUDA.runtime_version()) devices=", length(CUDA.devices()))
for (i, d) in enumerate(CUDA.devices())
    println("  device $(i-1): ", CUDA.name(d))
end

# ---- sched policy kwargs (mirrors benchmark_028_feasibility.jl exactly) ------

function sched_cache_kwargs(policy, ell, Pexp, ::Type{T}, LH) where T
    startswith(policy, "sched") || error("P2 expects a sched policy, got $policy")
    schedule = parse.(Int, split(policy[6:end], '-'))
    length(schedule) == ell - 1 || error("policy $policy needs $(ell-1) entries at ell=$ell")
    foreach(q -> RigidHierarchicalTables(q), schedule)
    K = length(union((Set((j == 1 ? RigidHierarchicalTables(x) :
            FM._rigid_transition_tables(schedule[j - 1], x)).push_offsets)
        for (j, x) in enumerate(schedule))...))
    q = last(schedule)
    h0 = T(BOX_SIZE / 2)
    eps = rigid_stencil_epsilon(Pexp, h0, ell, q; lamb_helmholtz=LH, TF=T)
    base = HierarchicalRigidStencil(ConstantPStencilConfig(Pexp, eps;
        lamb_helmholtz=LH); near_radius2=q, window_classes=K)
    return (; policy=FM._hierarchical_stencil_with_schedule(base, schedule)), K
end

const CACHE_KWARGS, KFULL = sched_cache_kwargs(POLICY, ELL, P, TF, false)
const OPTS = CUDARadixLifecycleOptions(; precision=TF,
    operator=MaterializedYRotationM2L(), m2l_strategy=DenseTranslationM2L())

_used_bytes() = CUDA.total_memory() - CUDA.free_memory()

# ---- phase 0: reference data + P2P microbenchmark ---------------------------

bodies = fm028_body_matrix(SEED, N)
indices = fm024b_expected_reference_indices(N, SAMPLER_SEED)
ref_path = fm024b_reference_path(REFDIR, N)
isfile(ref_path) || error("missing 024b reference $ref_path (checksum gate must run first)")
r024 = fm024b_read_reference(ref_path, N;
    expected_seed=SEED, expected_sampler_seed=SAMPLER_SEED)
ref_potential = r024.potential
ref_gradient = reshape(reduce(vcat, [collect(g) for g in r024.gradient]), 3, :)
println("reference: 024b_csv checksum=", r024.checksum, " samples=", length(indices))

# map each device's memory pool for the peer BEFORE measuring: without this the
# driver stages pool-backed cross-device copies at ~32 GB/s despite NV18
pool_p2p = p2_enable_pool_peer_access!(0, 1)
println("pool peer access granted: ", pool_p2p)

# raw cross-device copy microbenchmark at the exchange size (rows x n/2)
p2p_gbps = zeros(2)
let nb = 4 * (N ÷ 2) * sizeof(TF)
    CUDA.device!(0); a0 = CUDA.rand(TF, 4, N ÷ 2)
    CUDA.device!(1); a1 = CUDA.zeros(TF, 4, N ÷ 2)
    for (i, (src, dst, dev)) in enumerate(((a0, a1, 0), (a1, a0, 1)))
        CUDA.device!(dev)
        copyto!(dst, src); CUDA.synchronize()
        ts = [(CUDA.@elapsed copyto!(dst, src)) for _ in 1:10]
        p2p_gbps[i] = nb / (median(ts) * 1e9)
        @printf("cross-device copy dir%d: %.1f MB in %.3f ms => %.1f GB/s\n",
            i, nb / 1e6, median(ts) * 1e3, p2p_gbps[i])
    end
    CUDA.device!(0); CUDA.unsafe_free!(a0)
    CUDA.device!(1); CUDA.unsafe_free!(a1)
end

# ---- phase 1: single-GPU in-job reference (stock cycle-1 defaults) ----------

println("\n=== phase 1: single-GPU reference (device 0, stock defaults)")
FM.CUDA_CACHED_WINDOWS[] = true
FM.CUDA_GRAPH_LIFECYCLE[] = true
FM.CUDA_OVERLAP_NEARFIELD[] = true
CUDA.device!(0)
single = let
    sys = FM028DeviceSystem{TF}(bodies)
    GC.gc(); CUDA.reclaim()
    cache = RadixFMMCache(sys; expansion_order=P, ell=ELL, max_n_bodies=N,
        bounds=(BOX_MIN, BOX_SIZE), lamb_helmholtz=false, device=true,
        options=OPTS, CACHE_KWARGS...)
    fmm!(sys, cache; scalar_potential=true, gradient=true)
    pot0, grad0 = fm028_sampled_output(sys, indices)
    m0 = fm028_accuracy_metrics(pot0, grad0, ref_potential, ref_gradient)
    step!() = (fmm!(sys, cache; scalar_potential=true, gradient=true);
        fm028_euler!(sys, DT, 0.0, 1.0))
    step!(); step!(); step!(); CUDA.synchronize()   # warm + graph record + replay
    samples = Float64[]
    for _ in 1:REPS
        push!(samples, @elapsed(begin step!(); CUDA.synchronize() end) * 1e3)
    end
    med = median(samples)
    @printf("single-GPU verdict: %.3f ms [%.3f, %.3f]  err %.5e\n",
        med, minimum(samples), maximum(samples), m0.gradient_rel_rms)
    sys = nothing; cache = nothing
    GC.gc(); CUDA.reclaim()
    (; med, lo=minimum(samples), hi=maximum(samples), err=m0.gradient_rel_rms)
end

# ---- phase 2: dual-GPU setup -------------------------------------------------

println("\n=== phase 2: dual-GPU setup (mirrored sources, partitioned targets)")
# the script-owned per-device graphs replace the production ones; the global
# nearfield stream/event singletons are single-device, so both flags go off
FM.CUDA_GRAPH_LIFECYCLE[] = false
FM.CUDA_OVERLAP_NEARFIELD[] = false
FM.CUDA_CACHED_WINDOWS[] = true

G = Vector{P2Gpu}(undef, 2)
setup_info = Vector{Any}(undef, 2)
for g in 1:2
    G[g], setup_info[g] = p2_setup_gpu!(g - 1, g, bodies, TF, P, ELL,
        CACHE_KWARGS, OPTS; max_n_bodies=N, bounds=(BOX_MIN, BOX_SIZE),
        use_graph=USE_GRAPH)
    st = G[g].cache.state
    @printf("GPU%d: construction %.0f ms, persistent %.3f GB, owned cells %d:%d, bodies %d:%d, routes %d, direct %d\n",
        g - 1, setup_info[g].construction_ms, setup_info[g].persistent_bytes / 1e9,
        G[g].part.c0, G[g].part.c1, G[g].part.b0, G[g].part.b1,
        st.interaction_list.total_routes, st.counts.n_direct)
end
G[1].part.b0 == 1 && G[2].part.b1 == N && G[1].part.b1 + 1 == G[2].part.b0 ||
    error("owned body ranges do not partition 1:$N")

ok, pdetail = p2_slice_coverage(G)
println("slice coverage gate (per-level window + pair slices tile the work lists): ",
    ok ? "PASS" : "FAIL", "  list sizes=", pdetail)
ok || error("slice coverage gate failed")

bar = P2Barrier(2)
seg = zeros(7, 2)
# graph recording is SOLO and serialized (GLOBAL capture mode outlaws
# concurrent CUDA API use — see the P2GraphSlot contract); then two concurrent
# replay steps verify the steady state and produce the union outputs at t=0
p2_require_peer_access!(G)
p2_record_graphs!(G)
for i in 1:2
    p2_step_pair!(G, bar; dt=0.0, do_euler=false, seg=seg)
end
graph_captured = [G[g].slot.exec_a !== nothing && G[g].slot.exec_b !== nothing
    for g in 1:2]
println("graph captured per GPU: ", graph_captured,
    USE_GRAPH ? "" : " (graph disabled by FM029P2_GRAPH=0)")

# ---- phase 3: step-0 accuracy + lockstep ------------------------------------

println("\n=== phase 3: step-0 sampled accuracy (union output)")
errs = map(1:2) do g
    CUDA.device!(G[g].dev)
    pot, grad = fm028_sampled_output(G[g].sys, indices)
    fm028_accuracy_metrics(pot, grad, ref_potential, ref_gradient)
end
for g in 1:2
    @printf("GPU%d: err_grad_rel_rms %.5e  err_pot_rel_rms %.5e  err_grad_max %.5e\n",
        g - 1, errs[g].gradient_rel_rms, errs[g].potential_rel_rms, errs[g].gradient_max)
end
lockstep0 = let
    CUDA.device!(G[1].dev); g1 = Array(G[1].sys.gradient)
    CUDA.device!(G[2].dev); g2 = Array(G[2].sys.gradient)
    g1 == g2
end
println("mirrored gradient lockstep (bitwise): ", lockstep0 ? "PASS" : "FAIL")

# ---- phase 4: verdict loop ---------------------------------------------------

println("\n=== phase 4: verdict loop (REPS=$REPS complete steps)")
host_alloc = @allocated p2_step_pair!(G, bar; dt=DT, do_euler=true, seg=seg)
walls = Float64[]
segs = zeros(7, 2, REPS)
for rep in 1:REPS
    wall, _ = p2_step_pair!(G, bar; dt=DT, do_euler=true, seg=seg)
    push!(walls, wall)
    segs[:, :, rep] .= seg
end
med(v) = median(v)
wall_med = med(walls)
segmed = [med(segs[i, g, :]) for i in 1:7, g in 1:2]
compute = [segs[1, g, r] + segs[2, g, r] + segs[4, g, r] for g in 1:2, r in 1:REPS]
commorch = [walls[r] - maximum(compute[:, r]) for r in 1:REPS]
imbalance = [abs(compute[1, r] - compute[2, r]) for r in 1:REPS]
orch_only = [walls[r] - max(segs[5, 1, r], segs[5, 2, r]) for r in 1:REPS]
exch = [max(segs[3, 1, r], segs[3, 2, r]) for r in 1:REPS]
exchange_bytes = 2 * sizeof(TF) * (length(G[1].cache.state.locals.phi) + 4 * N)
    # both directions: partial locals.phi + partial 4-row output
eff_injob = single.med / (2 * wall_med)
eff_record = RECORD_1GPU_MS / (2 * wall_med)

@printf("\n2-GPU verdict: %.3f ms [%.3f, %.3f]\n", wall_med, minimum(walls), maximum(walls))
for g in 1:2
    @printf("GPU%d: refresh %.3f  graph %.3f  exchange-copy %.3f  finalize+euler(dev) %.3f  gpu-wall %.3f  barrier %.3f  fin-host %.3f ms\n",
        g - 1, segmed[1, g], segmed[2, g], segmed[3, g], segmed[4, g],
        segmed[5, g], segmed[6, g], segmed[7, g])
end
@printf("comm+orchestration (wall - max compute): %.3f ms   [gate <= 0.4 ms]\n", med(commorch))
@printf("  of which exchange copies %.3f ms, task spawn/join %.3f ms, imbalance %.3f ms\n",
    med(exch), med(orch_only), med(imbalance))
@printf("parallel efficiency: %.1f%% vs in-job single (%.3f), %.1f%% vs 4.657 record   [gate >= 75%%]\n",
    100 * eff_injob, single.med, 100 * eff_record)
@printf("host alloc per pair step: %d bytes; refilters: %d/%d\n",
    host_alloc, G[1].slot.refilters, G[2].slot.refilters)

# ---- phase 5: convection re-check + lockstep --------------------------------

println("\n=== phase 5: $STEPS-step convection re-check")
for _ in 1:STEPS
    p2_step_pair!(G, bar; dt=DT, do_euler=true, seg=seg)
end
p2_step_pair!(G, bar; dt=0.0, do_euler=false, seg=seg)   # re-evaluate, no move
conv = let
    CUDA.device!(G[1].dev)
    refc = fm028_direct_sample_reference(G[1].sys.positions, G[1].sys.strengths, indices)
    potc, gradc = fm028_sampled_output(G[1].sys, indices)
    fm028_accuracy_metrics(potc, gradc, refc[1, :], refc[2:4, :])
end
lockstep5 = let
    CUDA.device!(G[1].dev); p1 = Array(G[1].sys.positions); gr1 = Array(G[1].sys.gradient)
    CUDA.device!(G[2].dev); p2_ = Array(G[2].sys.positions); gr2 = Array(G[2].sys.gradient)
    p1 == p2_ && gr1 == gr2
end
@printf("post-convection: err_grad_rel_rms %.5e  lockstep %s\n",
    conv.gradient_rel_rms, lockstep5 ? "PASS" : "FAIL")

# ---- phase 6: counter contract + memory -------------------------------------

base_ctr = [let c = G[g].cache.state.counters
    (c.route_uploads, c.operator_uploads, c.body_uploads,
        c.influence_downloads, c.metadata_downloads, c.expansion_host_copies)
end for g in 1:2]
p2_step_pair!(G, bar; dt=DT, do_euler=true, seg=seg)
counters_flat = all(1:2) do g
    c = G[g].cache.state.counters
    (c.route_uploads, c.operator_uploads, c.body_uploads,
        c.influence_downloads, c.metadata_downloads, c.expansion_host_copies) == base_ctr[g]
end
println("recurring transfer counters flat: ", counters_flat ? "PASS" : "FAIL")

mem = map(1:2) do g
    CUDA.device!(G[g].dev)
    _used_bytes()
end

gate_comm = med(commorch) <= 0.4
gate_eff = eff_record >= 0.75
println("\n=== P2 GATES: comm+orch $(round(med(commorch); digits=3)) ms ",
    gate_comm ? "PASS" : "FAIL", " (<=0.4); efficiency vs record ",
    round(100 * eff_record; digits=1), "% ", gate_eff ? "PASS" : "FAIL", " (>=75%)")
println("accuracy gate: ", maximum(e.gradient_rel_rms for e in errs) <= 1.19e-3 ?
    "PASS" : "FAIL", " (<=1.19e-3)")

# ---- CSV ---------------------------------------------------------------------

row = (;
    manifest=MANIFEST, job=JOBID, host=gethostname(),
    gpu0=CUDA.name(collect(CUDA.devices())[1]), gpu1=CUDA.name(collect(CUDA.devices())[2]),
    julia=string(VERSION), cuda=string(CUDA.runtime_version()),
    seed=SEED, sampler_seed=SAMPLER_SEED, n=N, ell=ELL, expansion_order=P,
    p_literature=P + 1, policy=POLICY, precision=string(TF),
    tensor_format=string(TENSOR), window_classes=KFULL, reps=REPS,
    use_graph=USE_GRAPH, graph_captured_g1=graph_captured[1],
    graph_captured_g2=graph_captured[2],
    single_ref_ms=single.med, single_ref_min_ms=single.lo, single_ref_max_ms=single.hi,
    single_ref_err=single.err, record_ref_ms=RECORD_1GPU_MS,
    dual_wall_ms=wall_med, dual_wall_min_ms=minimum(walls), dual_wall_max_ms=maximum(walls),
    refresh_g1_ms=segmed[1, 1], refresh_g2_ms=segmed[1, 2],
    graph_g1_ms=segmed[2, 1], graph_g2_ms=segmed[2, 2],
    exchange_g1_ms=segmed[3, 1], exchange_g2_ms=segmed[3, 2],
    finalize_euler_g1_ms=segmed[4, 1], finalize_euler_g2_ms=segmed[4, 2],
    gpu_wall_g1_ms=segmed[5, 1], gpu_wall_g2_ms=segmed[5, 2],
    barrier_g1_ms=segmed[6, 1], barrier_g2_ms=segmed[6, 2],
    fin_host_g1_ms=segmed[7, 1], fin_host_g2_ms=segmed[7, 2],
    comm_orch_ms=med(commorch), exchange_max_ms=med(exch),
    orch_spawn_join_ms=med(orch_only), imbalance_ms=med(imbalance),
    exchange_bytes_total=exchange_bytes,
    p2p_gbps_dir1=p2p_gbps[1], p2p_gbps_dir2=p2p_gbps[2],
    eff_vs_injob=eff_injob, eff_vs_record=eff_record,
    gate_comm_orch=gate_comm, gate_efficiency=gate_eff,
    err_gradient_rel_rms_g1=errs[1].gradient_rel_rms,
    err_gradient_rel_rms_g2=errs[2].gradient_rel_rms,
    err_potential_rel_rms_g1=errs[1].potential_rel_rms,
    err_gradient_max_g1=errs[1].gradient_max,
    conv_steps=STEPS, conv_dt=DT, conv_err_gradient_rel_rms=conv.gradient_rel_rms,
    lockstep_step0=lockstep0, lockstep_conv=lockstep5,
    slice_cover=ok, counters_flat=counters_flat,
    routes_g1=G[1].cache.state.interaction_list.total_routes,
    routes_g2=G[2].cache.state.interaction_list.total_routes,
    n_direct_g1=G[1].cache.state.counts.n_direct,
    n_direct_g2=G[2].cache.state.counts.n_direct,
    construction_g1_ms=setup_info[1].construction_ms,
    construction_g2_ms=setup_info[2].construction_ms,
    persistent_bytes_g1=setup_info[1].persistent_bytes,
    persistent_bytes_g2=setup_info[2].persistent_bytes,
    used_bytes_g1=mem[1], used_bytes_g2=mem[2],
    host_alloc_per_step=host_alloc,
    refilters_g1=G[1].slot.refilters, refilters_g2=G[2].slot.refilters,
    reference_checksum=r024.checksum, reference_samples=length(indices),
)
mkpath(dirname(OUT))
open(OUT, "w") do io
    println(io, join(string.(keys(row)), ','))
    println(io, join(string.(values(row)), ','))
end
println("wrote ", OUT)

# raw wall samples for the record (full measured range, not just the median)
open(OUT * ".walls.csv", "w") do io
    println(io, "rep,wall_ms,commorch_ms,imbalance_ms,exchange_ms")
    for rr in 1:REPS
        println(io, "$rr,$(walls[rr]),$(commorch[rr]),$(imbalance[rr]),$(exch[rr])")
    end
end
println("wrote ", OUT, ".walls.csv")
println("P2_EXIT_OK")
