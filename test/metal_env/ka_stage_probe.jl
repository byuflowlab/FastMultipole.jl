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


P, ell, n, wc = 4, 3, 256, 8
TF = Float32; ci = 1
sys_h = make_system(6100+ci, n, TF); sys_d = make_system(6100+ci, n, TF)
opts = FM.CUDARadixLifecycleOptions(; precision=TF,
    m2l_strategy=FM.ConcatenatedFixedZM2L(), body_type=FM.Point{FM.Vortex})
hcache = RadixFMMCache(sys_h; expansion_order=P, ell=ell, window_classes=wc, options=opts)
fmm!(sys_h, hcache)
a = device_build_args(hcache); LH = typeof(hcache).parameters[2]
dcache = ext.ka_radix_cache_device_build(DEV_BACKEND, (sys_d,),
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
switches = FM.DerivativesSwitch(FM.to_vector(false,1), FM.to_vector(true,1),
    FM.to_vector(false,1), (sys_d,))

st = dcache.state; hctx = st.interaction_list; ws = st.scratch
println("hctx type: ", typeof(hctx).name.name)
println("noffsets = ", hctx.noffsets, "  window_classes = ", hctx.window_classes,
        "  first_m2l_level = ", hctx.first_m2l_level, "  ell = ", hctx.ell)
nwin = 0
for L in hctx.first_m2l_level:hctx.ell, fo in 1:hctx.window_classes:hctx.noffsets
    global nwin += 1
end
println("windows per step = ", nwin, "  -> device syncs in M2L alone = ", nwin)

# warm everything
for _ in 1:3; ext.ka_radix_cache_device_step!(dcache, (sys_d,), switches);
    KernelAbstractions.synchronize(DEV_BACKEND); end

function t(f, k=10)
    for _ in 1:2; f(); KernelAbstractions.synchronize(DEV_BACKEND); end
    ts = Float64[]
    for _ in 1:k
        t0 = time_ns(); f(); KernelAbstractions.synchronize(DEV_BACKEND)
        push!(ts, (time_ns()-t0)/1e6)
    end
    minimum(ts)
end
full  = t(() -> ext.ka_radix_cache_device_step!(dcache, (sys_d,), switches))
upd   = t(() -> ext.ka_update_radix_state!(dcache, (sys_d,)))
bodyt = t(() -> ext.ka_lifecycle_body!(st))
m2l   = t(() -> ext.ka_launch_m2l!(st, ws))
nf    = t(() -> ext.ka_launch_nearfield!(st; clear=true))
b2m   = t(() -> ext.ka_launch_b2m!(st))
l2b   = t(() -> ext.ka_launch_l2b!(st))
@printf("full step      %8.2f ms\n", full)
@printf("  update state %8.2f ms\n", upd)
@printf("  lifecycle    %8.2f ms\n", bodyt)
@printf("    nearfield  %8.2f ms\n", nf)
@printf("    b2m        %8.2f ms\n", b2m)
@printf("    m2l        %8.2f ms   (%d windows -> %.2f ms/window)\n", m2l, nwin, m2l/nwin)
@printf("    l2b        %8.2f ms\n", l2b)
@printf("  finalize+etc %8.2f ms\n", full - upd - bodyt)
