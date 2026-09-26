# Full UJ profile: every major step of one U/J evaluation, timed AND with its
# allocation count and allocated bytes, on the threaded CPU and on the GPU.
#
# Both arms run FLOWVPM's OWN entry (`UJ_fmm`) once for the ground-truth wall
# time, then re-run the same evaluation step by step so the total is attributed.
# The step decomposition is a transcription of the production call chain, not a
# re-implementation:
#
#   CPU  (Array-backed field)  UJ_fmm -> fmm.fmm!(pfield; ...)
#        src/fmm.jl:1030 (trees) -> :1072 (lists) -> :1318 (passes)
#   GPU  (device-backed field)  UJ_fmm -> UJ_fmm_gpu! -> _radix_fmm_evaluate!
#        -> fmm!(pfield, cache) -> ka_radix_cache_device_step!  (ext:5886)
#        whose body is ka_lifecycle_body! (ext:3274)
#
# Field: the SMALLEST real for_ryan dump (step 36, 15984 particles), NOT a
# subsample. FastMultipole forces n_threads=1 below MIN_BODIES=10000
# (src/FastMultipole.jl:35), so a subsampled field would silently profile the
# single-threaded CPU while the report claimed "threaded".

include("ka_backend.jl")
include("pipeline_field.jl")

using FastMultipole, Printf
using FastMultipole.StaticArrays
const FM = FastMultipole
const V = FLOWVPM
import KernelAbstractions as KA

dev_functional() || (println("$(DEV_NAME) not functional; skipping"); exit(0))
ext = Base.get_extension(FastMultipole, :FastMultipoleKAExt)
ext === nothing && error("the KA extension is not loaded")

const DEV_TF = getfield(Base, Symbol(get(ENV, "DEV_TF", "Float32")))  # device field type; Int width follows it
const STEPS = [parse(Int, x) for x in split(get(ENV, "UJ_STEPS", "36"), ",")]
const P     = 5           # cache expansion order; field is built with p = P+1
const CALLS = 12          # GPU: Metal needs ~5 calls of kernel-cache ramp before it plateaus
const CPU_CALLS = 5       # CPU is flat after call 1; 12 reps of a multi-second
                          # call at the top of the ladder buys nothing
const STRATEGY = :concat  # the one m2l_strategy KA implements (see pipeline_uj_fmm.jl)

################################################################################
# measurement
################################################################################

"Run `f` `calls` times; return rows[i] = (seconds, allocs, bytes)."
function measured(f; calls::Int=CALLS, gc::Bool=true)
    rows = Tuple{Float64,Int,Int}[]
    for _ in 1:calls
        gc && GC.gc()
        st = Base.gc_num(); t0 = time_ns()
        f()
        t1 = time_ns(); d = Base.GC_Diff(Base.gc_num(), st)
        push!(rows, ((t1 - t0)/1e9, Base.gc_alloc_count(d), d.allocd))
    end
    return rows
end

# Per-step accumulator: label => rows, one row per repeat.
const Sink = Dict{String,Vector{Tuple{Float64,Int,Int}}}
mk_sink() = (Sink(), String[])

function step!(sink, order, label, f)
    haskey(sink, label) || (sink[label] = Tuple{Float64,Int,Int}[]; push!(order, label))
    st = Base.gc_num(); t0 = time_ns()
    v = f()
    t1 = time_ns(); d = Base.GC_Diff(Base.gc_num(), st)
    push!(sink[label], ((t1 - t0)/1e9, Base.gc_alloc_count(d), d.allocd))
    return v
end

"Median over the LAST HALF of the repeats -- the plateau, not the ramp."
function summarize(rows)
    n = length(rows)
    idx = n > 1 ? ((cld(n, 2) + 1):n) : (1:1)
    med(v) = sort(collect(v))[cld(length(v), 2)]
    (t = med(rows[i][1] for i in idx),
     a = med(rows[i][2] for i in idx),
     b = med(rows[i][3] for i in idx),
     first = rows[1][1])
end

function table(title, sink, order, total)
    println("\n$title")
    @printf("  %-30s %10s %7s %12s %12s   %s\n",
            "step", "sec", "% tot", "allocs", "MiB", "call-1 s")
    println("  ", "-"^88)
    acc = 0.0
    for lbl in order
        s = summarize(sink[lbl]); acc += s.t
        @printf("  %-30s %10.5f %6.1f%% %12d %12.3f   %9.4f\n",
                lbl, s.t, 100*s.t/total, s.a, s.b/2^20, s.first)
    end
    println("  ", "-"^88)
    @printf("  %-30s %10.5f %6.1f%%\n", "sum of steps", acc, 100*acc/total)
    @printf("  %-30s %10.5f %6.1f%%   (unattributed: %.5f s)\n",
            "UJ_fmm wall (ground truth)", total, 100.0, total - acc)
    flush(stdout)
end

################################################################################
# one size
################################################################################

println("=== full UJ step profile: $(WAKE_CASE), steps $(STEPS) ===")
println("julia threads = ", Threads.nthreads(),
        "   (FastMultipole MIN_BODIES = ", FM.MIN_BODIES, ")")
println("device = ", DEV_NAME, "   m2l_strategy = ", STRATEGY,
        "   P = ", P, "   gpu calls = ", CALLS, "   cpu calls = ", CPU_CALLS)
Threads.nthreads() > 1 || @warn "julia started with 1 thread; the CPU arm is not threaded"
flush(stdout)

const SUMMARY = NamedTuple[]

function device_copy_of(h)
    d = V.ParticleField(h.maxparticles, DEV_TF; arraytype=devmatrix, np=h.np,
        fmm=V.FMM(; p=P+1, autotune_p=false, autotune_ncrit=false,
                    autotune_reg_error=false, default_rho_over_sigma=1.0))
    d.particles .= devarray(DEV_TF.(Array(h.particles)))
    return d
end

function run_size(STEP::Int)
    TF = Float64
    host = load_wake(STEP; TF=TF, P=P)
    NP = V.get_np(host)
    wk = wake_stats(host)
    println("\n", "="^92)
    @printf("step %d:  np = %d   box L = %.4g   sigma in [%.4g, %.4g]\n",
            STEP, NP, wk.L, wk.sigma_min, wk.sigma_max)
    NP >= FM.MIN_BODIES || @warn "np < MIN_BODIES: fmm! will force n_threads=1"
    flush(stdout)

    OPT = host.fmm
    EXPANSION_ORDER = OPT.p - 1
    LEAF_SIZE = max(OPT.ncrit, OPT.min_ncrit)
    ERRTOL = FM.PowerRelativeGradient{OPT.relative_tolerance, OPT.absolute_tolerance, true}()
    ILM = FM.SelfTuningTargetStop()

    #--- CPU ground truth ---------------------------------------------------
    cpu_rows = measured(() -> V.UJ_fmm(host; autotune=false); calls=CPU_CALLS)
    U_ref = copy(Array(host.particles)[V.U_INDEX, 1:NP])
    cpu_total = summarize(cpu_rows).t
    @printf("\nCPU UJ_fmm wall: %.5f s  (ramp %s)\n", cpu_total,
            join((@sprintf("%.4f", r[1]) for r in cpu_rows), "/")); flush(stdout)

    #--- CPU decomposition --------------------------------------------------
    cpu_sink, cpu_order = mk_sink()
    targets = (host,); sources = (host,)
    switches = FM.DerivativesSwitch(FM.to_vector(false, 1), FM.to_vector(true, 1),
                                    FM.to_vector(true, 1), targets)
    lsrc = FM.to_vector(LEAF_SIZE, 1)
    ltgt = FM.to_vector(minimum(lsrc), 1)
    lh = Val(FM.has_vector_potential(sources))
    nthr = Threads.nthreads()
    nm2l = Ref(0); ndir = Ref(0)

    for rep in 1:CPU_CALLS
        GC.gc()
        step!(cpu_sink, cpu_order, "reset particles", () -> V._reset_particles(host))
        cache = step!(cpu_sink, cpu_order, "Cache alloc",
            () -> FM.Cache(targets, sources, switches))
        ttree = step!(cpu_sink, cpu_order, "tree build (target)",
            () -> FM.Tree(targets, FM.TargetTree(), switches, TF;
                  buffers=cache.target_buffers, small_buffers=cache.target_small_buffers,
                  expansion_order=EXPANSION_ORDER, leaf_size=ltgt,
                  shrink=OPT.shrink_recenter, recenter=OPT.shrink_recenter,
                  interaction_list_method=ILM))
        stree = step!(cpu_sink, cpu_order, "tree build (source)",
            () -> FM.Tree(sources, FM.SourceTree(), switches, TF;
                  buffers=cache.source_buffers, small_buffers=cache.source_small_buffers,
                  expansion_order=EXPANSION_ORDER, leaf_size=lsrc,
                  shrink=OPT.shrink_recenter, recenter=OPT.shrink_recenter,
                  interaction_list_method=ILM))
        lists = step!(cpu_sink, cpu_order, "interaction lists", () -> begin
            m2l, dir = FM.build_interaction_lists(ttree.branches, stree.branches, lsrc,
                OPT.theta, true, true, true, ILM)
            (FM.sort_by_target(m2l, ttree.branches), FM.sort_by_target(dir, ttree.branches))
        end)
        m2l_list, direct_list = lists
        nm2l[] = length(m2l_list); ndir[] = length(direct_list)
        step!(cpu_sink, cpu_order, "reset expansions", () -> begin
            FM.reset_expansions!(ttree); FM.reset_expansions!(stree)
        end)
        step!(cpu_sink, cpu_order, "nearfield (direct)",
            () -> FM.nearfield_multithread!(ttree.buffers, ttree.branches, sources,
                  stree.buffers, stree.branches, switches, direct_list, ILM, nthr, ()))
        step!(cpu_sink, cpu_order, "upward pass (B2M+M2M)",
            () -> FM.upward_pass_multithread!(stree, sources, EXPANSION_ORDER, lh, nthr))
        step!(cpu_sink, cpu_order, "horizontal pass (M2L)",
            () -> FM.horizontal_pass_multithread!(ttree, stree, m2l_list, lh,
                  EXPANSION_ORDER, ERRTOL, ILM, nthr))
        step!(cpu_sink, cpu_order, "downward pass (L2L+L2B)",
            () -> FM.downward_pass_multithread!(ttree, ttree.buffers, switches,
                  EXPANSION_ORDER, lh, nthr))
        step!(cpu_sink, cpu_order, "buffer_to_target!",
            () -> FM.buffer_to_target!(targets, ttree, switches))
    end
    @printf("  [cpu] n_threads=%d  leaf=%d  P=%d  |m2l|=%d  |direct|=%d\n",
            nthr, lsrc[1], EXPANSION_ORDER, nm2l[], ndir[]); flush(stdout)
    table("--- CPU (threaded, np=$NP) ---", cpu_sink, cpu_order, cpu_total)
    U_dec = Array(host.particles)[V.U_INDEX, 1:NP]
    @printf("  decomposition vs UJ_fmm: max relerr(U) = %.3e\n",
            maximum(abs.(U_dec .- U_ref)) / maximum(abs.(U_ref))); flush(stdout)

    #--- GPU ground truth ---------------------------------------------------
    fresh = load_wake(STEP; TF=TF, P=P)
    dev = device_copy_of(fresh)
    V.radix_fmm_settings!(dev; m2l_strategy=STRATEGY)

    gpu_rows = measured(() -> V.UJ_fmm(dev); calls=CALLS)
    gpu_total = summarize(gpu_rows).t
    U_dev = Array(dev.particles)[V.U_INDEX, 1:NP]
    relerr = maximum(abs.(U_dev .- U_ref)) / maximum(abs.(U_ref))
    @printf("\nGPU UJ_fmm wall: %.5f s  (ramp %s)   relerr(U) vs CPU = %.3e\n",
            gpu_total, join((@sprintf("%.4f", r[1]) for r in gpu_rows), "/"), relerr)
    flush(stdout)

    dcache = V._radix_fmm_couplings[dev].cache
    dstate = dcache.state
    dws = dstate.scratch
    dtargets = (dev,)
    dswitches = FM.DerivativesSwitch(FM.to_vector(false, 1), FM.to_vector(true, 1),
                                     FM.to_vector(true, 1), dtargets)
    backend = KA.get_backend(dstate.output)
    sync() = KA.synchronize(backend)
    @printf("  [gpu] ell=%d  P=%d  max_cells=%d  n_cells=%d  n_direct=%d  n_routes=%d\n",
            dcache.ell, dcache.expansion_order, dcache.max_cells,
            dstate.counts.n_cells, dstate.counts.n_direct, dstate.counts.n_routes)
    # the direct-list size in INTERACTIONS, which is what the nearfield actually costs
    rr = Array(dstate.cell_ranges)
    dt = Array(dstate.direct_targets); ds = Array(dstate.direct_sources)
    ninter = sum(Int(rr[2, dt[k]]) * Int(rr[2, ds[k]]) for k in 1:dstate.counts.n_direct)
    @printf("  [gpu] direct interactions = %.4g   (all-pairs would be %.4g; %.1f%%)\n",
            ninter, float(NP)^2, 100*ninter/float(NP)^2)
    flush(stdout)

    gpu_sink, gpu_order = mk_sink()
    for rep in 1:CALLS
        GC.gc()
        step!(gpu_sink, gpu_order, "reset particles",
            () -> (V._reset_particles(dev); sync()))
        step!(gpu_sink, gpu_order, "update state (sort/tree)",
            () -> (ext.ka_update_radix_state!(dcache, dtargets); sync()))
        step!(gpu_sink, gpu_order, "nearfield (direct)",
            () -> (ext.ka_launch_nearfield!(dstate; workgroup=64, clear=true); sync()))
        step!(gpu_sink, gpu_order, "B2M",
            () -> (ext.ka_launch_b2m!(dstate; workgroup=128); sync()))
        step!(gpu_sink, gpu_order, "M2M", () -> begin
            FM._zero_resident_nonleaf_multipoles!(dstate)
            for g in dws.m2m_groups
                ext.ka_resident_stage_group_apply!(dstate.multipoles, dstate.multipoles,
                                                   g, dws, :m2m)
            end
            sync()
        end)
        step!(gpu_sink, gpu_order, "M2L",
            () -> (ext.ka_launch_m2l!(dstate, dws); sync()))
        step!(gpu_sink, gpu_order, "L2L", () -> begin
            for g in dws.l2l_groups
                ext.ka_resident_stage_group_apply!(dstate.locals, dstate.locals,
                                                   g, dws, :l2l)
            end
            sync()
        end)
        step!(gpu_sink, gpu_order, "L2B",
            () -> (ext.ka_launch_l2b!(dstate; workgroup=64); sync()))
        step!(gpu_sink, gpu_order, "finalize output (D2H)", () -> begin
            ext.ka_finalize_radix_output!(dstate, dtargets;
                derivatives_switches=dswitches,
                host_output_staging=dcache.device_ctx.host_output,
                target_buffers=FM._radix_cache_target_buffers!(dcache, dswitches),
                device_target_buffers=dcache.device_ctx.device_target_buffers)
            sync()
        end)
    end
    table("--- $(DEV_NAME) GPU (np=$NP) ---", gpu_sink, gpu_order, gpu_total)
    U_dec_d = Array(dev.particles)[V.U_INDEX, 1:NP]
    @printf("  decomposition vs UJ_fmm: max relerr(U) = %.3e\n",
            maximum(abs.(U_dec_d .- U_dev)) / maximum(abs.(U_dev)))
    @printf("\n>>> step %d  np=%d:  CPU %.5f s   GPU %.5f s   speedup %.2fx   relerr %.3e\n",
            STEP, NP, cpu_total, gpu_total, cpu_total/gpu_total, relerr)
    flush(stdout)

    nf = summarize(gpu_sink["nearfield (direct)"]).t
    push!(SUMMARY, (; step=STEP, np=NP, ell=dcache.ell, cpu=cpu_total, gpu=gpu_total,
                      speedup=cpu_total/gpu_total, relerr=relerr,
                      nf_frac=nf/gpu_total, ninter=ninter,
                      inter_frac=ninter/float(NP)^2))

    # drop the coupling so the next size does not benchmark against a device
    # allocator still holding this one's buffers
    V.clear_radix_fmm_cache!(dev)
    delete!(V._radix_fmm_couplings, dev)
    GC.gc(); GC.gc()
    return nothing
end

for s in STEPS
    run_size(s)
end

println("\n", "="^92)
println("LADDER SUMMARY")
@printf("  %8s %8s %5s %10s %10s %9s %10s %10s %10s\n",
        "step", "np", "ell", "CPU s", "GPU s", "speedup", "GPU nf%", "direct/n^2", "relerr")
for r in SUMMARY
    @printf("  %8d %8d %5d %10.5f %10.5f %8.2fx %9.1f%% %9.1f%% %10.2e\n",
            r.step, r.np, r.ell, r.cpu, r.gpu, r.speedup,
            100*r.nf_frac, 100*r.inter_frac, r.relerr)
end
flush(stdout)
