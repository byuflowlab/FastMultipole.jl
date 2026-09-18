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
    (4, 4,  4_096, 256, 10),
    (4, 5, 32_768, 256, 10),
]

println("threads = $(Threads.nthreads()), device = $DEV_NAME\n")
@printf("%-8s %4s %4s | %9s %9s | %9s %9s | %9s %9s | %7s %7s\n",
    "n", "P", "ell", "metal_min", "metal_med",
    "rdx_min", "rdx_med", "mt_min", "mt_med", "x_rdx", "x_mt")

for (ci, (P, ell, n, wc, trials)) in pairs(CASES)
    TF = Float32
    sys_h = make_system(6100 + ci, n, TF)
    sys_d = make_system(6100 + ci, n, TF)
    sys_c = make_system(6100 + ci, n, TF)

    opts = FM.CUDARadixLifecycleOptions(; precision=TF,
        m2l_strategy=FM.ConcatenatedFixedZM2L(), body_type=FM.Point{FM.Vortex})
    hcache = RadixFMMCache(sys_h; expansion_order=P, ell=ell, window_classes=wc,
        options=opts)
    fmm!(sys_h, hcache)

    a = device_build_args(hcache)
    LH = typeof(hcache).parameters[2]
    build(backend, sysx) = ext.ka_radix_cache_device_build(backend, (sysx,),
        hcache.expansion_order, ell, hcache.x_min, hcache.h0, hcache.max_n_bodies,
        hcache.options, hcache.policy, hcache.accepted_offsets,
        hcache.rejected_offsets, hcache.max_cells, hcache.max_nodes,
        hcache.route_capacity, hcache.direct_capacity,
        hcache.state.multipoles.basis_info, Val(LH);
        hierarchical_tables=a.tables, class_level=a.class_level,
        class_offset=a.class_offset, hierarchical_level_class_of=a.level_class_of,
        hierarchical_level_radii2=a.level_radii2,
        max_level_nodes=a.max_level_nodes, hessian=hcache.hessian,
        ell_axes=hcache.ell_axes, box_extent=hcache.box_extent,
        root_level=a.root_level, first_m2l_level=a.first_m2l_level)
    dcache = build(DEV_BACKEND, sys_d)
    switches = FM.DerivativesSwitch(FM.to_vector(false, 1), FM.to_vector(true, 1),
        FM.to_vector(false, 1), (sys_d,))

    ccache = FM.Cache((sys_c,), (sys_c,), switches)
    fmm!(sys_c, ccache; expansion_order=P, gradient=true)

    # the device step ends in a blocking host copy, but synchronize explicitly
    # rather than rely on that: a timing that stops before the GPU does is the
    # classic way to publish a fake speedup
    metal = timeit(trials, 3) do
        ext.ka_radix_cache_device_step!(dcache, (sys_d,), switches)
        KernelAbstractions.synchronize(DEV_BACKEND)
    end
    e_d = (d = maximum(abs.(sys_d.gradient_stretching[1:3, :] .-
                            sys_h.gradient_stretching[1:3, :]));
           s = maximum(abs.(sys_h.gradient_stretching[1:3, :])); s == 0 ? d : d / s)
    e_d < 3e-4 || error("$DEV_NAME disagrees with the host lifecycle: $e_d")

    rdx = timeit(() -> fmm!(sys_h, hcache), trials, 3)
    mt  = timeit(() -> fmm!(sys_c, ccache; expansion_order=P, gradient=true),
                 trials, 3)

    @printf("%-8d %4d %4d | %9.2f %9.2f | %9.2f %9.2f | %9.2f %9.2f | %6.2fx %6.2fx\n",
        n, P, ell, metal.min, metal.med, rdx.min, rdx.med,
        mt.min, mt.med, rdx.min / metal.min, mt.min / metal.min)
    flush(stdout)
end
