# Warm per-step timings for the KA device RadixFMMCache, against two CPU arms.
#
# Everything in ka_device_cache_correctness.jl's timing was first-call
# compilation (82 s for the P-specialized hierarchical M2L, 84% of it compile).
# This measures the steady state: every arm is warmed once, then timed over
# repeated identical steps on the same cache, so no arm pays for allocation,
# plan construction or codegen.
#
# Three arms, and the distinction between them matters:
#
#   metal   -- ka_radix_cache_device_step! on the KA device cache.
#   cpu-rdx -- fmm! on a host RadixFMMCache. The SAME algorithm, the same
#              uniform hierarchical stencil, the same expansion order: the
#              honest apples-to-apples comparison. It is SINGLE-THREADED --
#              src/translate_batched_resident.jl contains no Threads.@threads,
#              @spawn or nthreads() anywhere, so -t changes nothing for it.
#   cpu-mt  -- fmm! on a classic `Cache`. This IS the multithreaded CPU path and
#              is what a user actually runs on CPU today, but it is a DIFFERENT
#              algorithm (adaptive tree, self-tuning interaction lists), so a
#              ratio against it measures algorithm+hardware together, not the
#              KA port. Reported because "how fast is the GPU vs my CPU" is the
#              real question; not to be read as a KA-vs-native number.
#
# Not a KA-vs-CUDA measurement. That is one GPU, two backends, and belongs on
# H200; Metal-vs-CPU here is different silicon and says nothing about it.
include("ka_backend.jl")
using FastMultipole, Random, Test, Printf
using LinearAlgebra
using FastMultipole.StaticArrays

const FM = FastMultipole
include(joinpath(@__DIR__, "..", "vortex.jl"))

if !dev_functional()
    println("$(DEV_NAME) not functional; skipping")
    exit(0)
end
ext = Base.get_extension(FastMultipole, :FastMultipoleKAExt)
ext !== nothing || error("FastMultipoleKAExt did not load")

make_system(seed, n, TF) = (Random.seed!(seed);
    VortexParticles(rand(TF, 3, n), (randn(TF, 3, n) ./ TF(n)), zeros(TF, n);
        potential=zeros(TF, 13, n), gradient_stretching=zeros(TF, 6, n)))

function device_build_args(hcache)
    sp = hcache.policy
    ell = hcache.ell
    tables, level_class_of, level_radii2, root_level, first_m2l_level =
        FM._hierarchical_scheduled_tables(sp, ell, hcache.ell_axes)
    class_level, class_offset, _ =
        FM._hierarchical_class_metadata(tables, ell, first_m2l_level)
    max_level_nodes = ell >= 2 ? maximum(
        (FM._radix_level_node_capacity(L, hcache.ell_axes, ell, hcache.max_cells)
         for L in first_m2l_level:ell); init=0) : 0
    return (; tables, level_class_of, level_radii2, root_level, first_m2l_level,
        class_level, class_offset, max_level_nodes)
end

# min and median over `trials`, after `warmup` untimed calls. min is the number
# to compare (least contaminated by GC and by whatever else the machine is
# doing); the median is printed so a large min/median gap flags a noisy sample
# instead of hiding inside a single reported figure.
function timeit(f, trials, warmup)
    for _ in 1:warmup; f(); end
    ts = Float64[]
    for _ in 1:trials
        t0 = time_ns(); f(); push!(ts, (time_ns() - t0) / 1e6)
    end
    sort!(ts)
    return (min=ts[1], med=ts[(length(ts)+1) ÷ 2])
end

# (P, ell, n, window_classes, trials)
const CASES = [
    (4, 3,    256, 8, 3),

]

println("threads = $(Threads.nthreads()), device = $DEV_NAME\n")
@printf("%-8s %4s %4s | %9s %9s | %9s %9s | %9s %9s | %7s %7s\n",
    "n", "P", "ell", "metal_min", "metal_med",
    "rdx_min", "rdx_med", "mt_min", "mt_med", "x_rdx", "x_mt")


P, ell, n = 4, 3, 256
TF = Float32; ci = 1
sys_h = make_system(6100+ci, n, TF)
opts = FM.CUDARadixLifecycleOptions(; precision=TF,
    m2l_strategy=FM.ConcatenatedFixedZM2L(), body_type=FM.Point{FM.Vortex})

function build_for(wc)
    sys_d = make_system(6100+ci, n, TF)
    hc = RadixFMMCache(sys_h; expansion_order=P, ell=ell, window_classes=wc, options=opts)
    a = device_build_args(hc); LH = typeof(hc).parameters[2]
    dc = ext.ka_radix_cache_device_build(DEV_BACKEND, (sys_d,),
        hc.expansion_order, ell, hc.x_min, hc.h0, hc.max_n_bodies,
        hc.options, hc.policy, hc.accepted_offsets, hc.rejected_offsets,
        hc.max_cells, hc.max_nodes, hc.route_capacity, hc.direct_capacity,
        hc.state.multipoles.basis_info, Val(LH);
        hierarchical_tables=a.tables, class_level=a.class_level,
        class_offset=a.class_offset, hierarchical_level_class_of=a.level_class_of,
        hierarchical_level_radii2=a.level_radii2,
        max_level_nodes=a.max_level_nodes, hessian=hc.hessian,
        ell_axes=hc.ell_axes, box_extent=hc.box_extent,
        root_level=a.root_level, first_m2l_level=a.first_m2l_level)
    sw = FM.DerivativesSwitch(FM.to_vector(false,1), FM.to_vector(true,1),
        FM.to_vector(false,1), (sys_d,))
    return hc, dc, sys_d, sw
end

hc_ref, _, _, _ = build_for(8)
fmm!(sys_h, hc_ref)          # host reference output for the accuracy check
ref = copy(sys_h.gradient_stretching[1:3, :])
refscale = maximum(abs.(ref))

function tmin(f, k=10)
    for _ in 1:3; f(); KernelAbstractions.synchronize(DEV_BACKEND); end
    ts = Float64[]
    for _ in 1:k
        t0 = time_ns(); f(); KernelAbstractions.synchronize(DEV_BACKEND)
        push!(ts, (time_ns()-t0)/1e6)
    end
    minimum(ts)
end

println("\nnoffsets = ", 874, "   levels = ", ell - 2 + 1)
@printf("%6s %8s %10s %10s %10s %10s\n", "wc", "windows", "step_ms", "m2l_ms",
        "ms/window", "relerr")
for wc in (8, 16, 32, 64, 128, 256, 512, 874)
    local hc, dc, sys_d, sw
    try
        hc, dc, sys_d, sw = build_for(wc)
    catch e
        @printf("%6d  BUILD FAILED: %s\n", wc, sprint(showerror, e)[1:min(end,90)])
        continue
    end
    st = dc.state; hctx = st.interaction_list; ws = st.scratch
    nwin = 0
    for L in hctx.first_m2l_level:hctx.ell, fo in 1:hctx.window_classes:hctx.noffsets
        nwin += 1
    end
    local step_ms, m2l_ms
    try
        step_ms = tmin(() -> ext.ka_radix_cache_device_step!(dc, (sys_d,), sw))
        m2l_ms  = tmin(() -> ext.ka_launch_m2l!(st, ws))
    catch e
        @printf("%6d %8d  RUN FAILED: %s\n", wc, nwin, sprint(showerror, e)[1:min(end,90)])
        continue
    end
    ext.ka_radix_cache_device_step!(dc, (sys_d,), sw)
    KernelAbstractions.synchronize(DEV_BACKEND)
    d = maximum(abs.(sys_d.gradient_stretching[1:3, :] .- ref))
    err = refscale == 0 ? d : d / refscale
    @printf("%6d %8d %10.2f %10.2f %10.3f %10.2e\n", wc, nwin, step_ms, m2l_ms,
            m2l_ms/nwin, err)
    flush(stdout)
end
