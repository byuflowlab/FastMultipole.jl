# Does a SECOND case in the same process cost minutes? Isolates whether the
# ka_tree_vs_cpu_correctness.jl slowdown is per-case work or a respecialization
# triggered by the cache type changing with ell (Val(LH) in the device build).
include(joinpath(@__DIR__, "ka_backend.jl"))
using FastMultipole, Random, Printf
using FastMultipole.StaticArrays
using KernelAbstractions
const FM = FastMultipole
include(joinpath(@__DIR__, "..", "vortex.jl"))

t0 = time()
mark(s) = (@printf("[%7.1fs] %s\n", time() - t0, s); flush(stdout))
ext = Base.get_extension(FastMultipole, :FastMultipoleKAExt)

function mkfield(n, TF)
    Random.seed!(4242); pos = rand(TF, 3, n)
    VortexParticles(pos, (randn(TF, 3, n) ./ TF(n)), zeros(TF, n);
        potential=zeros(TF, 13, n), gradient_stretching=zeros(TF, 6, n))
end

function one_case(n, ell, TF=Float32, P=4)
    sys_h = mkfield(n, TF); sys_d = mkfield(n, TF)
    opts = FM.CUDARadixLifecycleOptions(; precision=TF,
        m2l_strategy=FM.ConcatenatedFixedZM2L(), body_type=FM.Point{FM.Vortex})
    hcache = RadixFMMCache(sys_h; expansion_order=P, ell=ell, window_classes=256, options=opts)
    mark("  n=$n ell=$ell cache built")
    fmm!(sys_h, hcache)
    mark("  n=$n ell=$ell host fmm! done")
    sp = hcache.policy
    tables, lco, lr2, root_level, first_m2l_level =
        FM._hierarchical_scheduled_tables(sp, ell, hcache.ell_axes)
    cl, co, _ = FM._hierarchical_class_metadata(tables, ell, first_m2l_level)
    mln = ell >= 2 ? maximum((FM._radix_level_node_capacity(L, hcache.ell_axes, ell, hcache.max_cells)
        for L in first_m2l_level:ell); init=0) : 0
    LH = typeof(hcache).parameters[2]
    mark("  n=$n ell=$ell LH=$LH  cachetype=$(typeof(hcache))")
    dcache = ext.ka_radix_cache_device_build(DEV_BACKEND, (sys_d,),
        hcache.expansion_order, ell, hcache.x_min, hcache.h0, hcache.max_n_bodies,
        hcache.options, hcache.policy, hcache.accepted_offsets,
        hcache.rejected_offsets, hcache.max_cells, hcache.max_nodes,
        hcache.route_capacity, hcache.direct_capacity,
        hcache.state.multipoles.basis_info, Val(LH);
        hierarchical_tables=tables, class_level=cl, class_offset=co,
        hierarchical_level_class_of=lco, hierarchical_level_radii2=lr2,
        max_level_nodes=mln, hessian=hcache.hessian, ell_axes=hcache.ell_axes,
        box_extent=hcache.box_extent, root_level=root_level,
        first_m2l_level=first_m2l_level)
    mark("  n=$n ell=$ell device cache built")
    ext.ka_update_radix_state!(dcache, (sys_d,))
    KernelAbstractions.synchronize(DEV_BACKEND)
    mark("  n=$n ell=$ell state updated")
    return hcache, dcache
end

mark("CASE 1: n=512 ell=2")
one_case(512, 2)
mark("CASE 2: n=1024 ell=3  <-- the suspect")
one_case(1024, 3)
mark("CASE 3: n=1024 ell=3 again (same types, should be fast)")
one_case(1024, 3)
mark("DONE")
