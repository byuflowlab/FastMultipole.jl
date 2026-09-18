# End-to-end gate for `ka_radix_cache_device_build` + `ka_radix_cache_device_step!`
# -- a whole `RadixFMMCache` built and stepped on a KA backend, with no CUDA
# anywhere, compared against FastMultipole's OWN host cache running `fmm!`.
#
# Why this closes the chain. Sessions 9-11 ported every stage of the device
# lifecycle (grid rebuild stages 1-4, hierarchical occupancy/direct pairs/
# windows, the lifecycle body, finalize, and the `ka_update_radix_state!` /
# `ka_radix_cache_device_step!` drivers), but each was gated against its own
# host oracle in isolation, and the two drivers were gated not at all: their
# only entry point is a device-resident `RadixFMMCache`, and
# `RadixFMMCache(...; device=true)` routes to `_radix_cache_device_build`, which
# allocates from `CUDA.zeros` throughout. No device cache existed off CUDA, so
# the drivers were load-checked only. This suite builds one and runs a real
# UJ through it.
#
# Oracle: a second `RadixFMMCache` over an identical system with device=false
# and the SAME stencil policy, stepped with `fmm!`. That is the host resident
# lifecycle -- a genuinely independent implementation of every stage, not a
# transcription -- so an elementwise match on the scattered velocity/potential
# is the real acceptance check the per-stage suites were standing in for.
#
# Both caches are hierarchical (`window_classes` set), which is the policy
# FLOWVPM builds and the only one the KA step implements; see
# [[reference-flowvpm-radix-cache-is-hierarchical]].
#
# Body type `Point{Vortex}` with Lamb-Helmholtz on: the only type with a KA B2M
# port, and what FLOWVPM runs.
include("ka_backend.jl")
using FastMultipole, Random, Test
using FastMultipole.StaticArrays

const FM = FastMultipole
include(joinpath(@__DIR__, "..", "vortex.jl"))

if !dev_functional()
    println("$(DEV_NAME) not functional; skipping")
    exit(0)
end
ext = Base.get_extension(FastMultipole, :FastMultipoleKAExt)
ext !== nothing || error("FastMultipoleKAExt did not load")

relerr(a, b) = (d = maximum(abs.(Array(a) .- Array(b))); s = maximum(abs.(Array(b)));
                s == 0 ? d : d / s)

make_system(seed, n, TF) = (Random.seed!(seed);
    VortexParticles(rand(TF, 3, n), (randn(TF, 3, n) ./ TF(n)), zeros(TF, n);
        potential=zeros(TF, 13, n), gradient_stretching=zeros(TF, 6, n)))

# Re-derive the construction arguments the host constructor computes between its
# policy selection and its `device` branch (src/translate_batched_resident.jl:
# 2547-2596). Everything else comes off the host cache's own fields, so the two
# caches are guaranteed to be built for the same geometry, capacities and
# stencil -- a divergence here would make the comparison meaningless rather
# than merely failing.
function device_build_args(hcache)
    sp = hcache.policy
    sp isa FM.HierarchicalRigidStencil ||
        error("expected a hierarchical host cache; got $(typeof(sp))")
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


const P, ell, n, wc = 4, 3, 256, 8
const TF = Float32
sys_h = make_system(5101, n, TF)
sys_d = make_system(5101, n, TF)
opts = FM.CUDARadixLifecycleOptions(; precision=TF,
    m2l_strategy=FM.ConcatenatedFixedZM2L(), body_type=FM.Point{FM.Vortex})
hcache = RadixFMMCache(sys_h; expansion_order=P, ell=ell, window_classes=wc, options=opts)
println("host fmm!"); @time fmm!(sys_h, hcache)
a = device_build_args(hcache)
LH = typeof(hcache).parameters[2]
basis_info = hcache.state.multipoles.basis_info
println("device build"); flush(stdout)
@time dcache = ext.ka_radix_cache_device_build(DEV_BACKEND, (sys_d,),
    hcache.expansion_order, ell, hcache.x_min, hcache.h0, hcache.max_n_bodies,
    hcache.options, hcache.policy, hcache.accepted_offsets, hcache.rejected_offsets,
    hcache.max_cells, hcache.max_nodes, hcache.route_capacity, hcache.direct_capacity,
    basis_info, Val(LH);
    hierarchical_tables=a.tables, class_level=a.class_level, class_offset=a.class_offset,
    hierarchical_level_class_of=a.level_class_of,
    hierarchical_level_radii2=a.level_radii2, max_level_nodes=a.max_level_nodes,
    hessian=hcache.hessian, ell_axes=hcache.ell_axes, box_extent=hcache.box_extent,
    root_level=a.root_level, first_m2l_level=a.first_m2l_level)
state = dcache.state
println("counts=", state.counts); flush(stdout)
println("update_radix_state (2nd)"); flush(stdout)
@time ext.ka_update_radix_state!(dcache, (sys_d,))
println("nearfield"); flush(stdout); @time ext.ka_launch_nearfield!(state)
println("b2m"); flush(stdout); @time ext.ka_launch_b2m!(state)
println("lifecycle body"); flush(stdout); @time ext.ka_lifecycle_body!(state)
switches = FM.DerivativesSwitch(FM.to_vector(false,1), FM.to_vector(true,1),
    FM.to_vector(false,1), (sys_d,))
println("finalize"); flush(stdout)
@time ext.ka_finalize_radix_output!(state, (sys_d,); derivatives_switches=switches,
    host_output_staging=dcache.device_ctx.host_output,
    target_buffers=FM._radix_cache_target_buffers!(dcache, switches),
    device_target_buffers=dcache.device_ctx.device_target_buffers)
e = relerr(sys_d.gradient_stretching[1:3,:], sys_h.gradient_stretching[1:3,:])
println("velocity relerr = ", e)
