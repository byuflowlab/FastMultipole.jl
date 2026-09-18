include("ka_backend.jl"); include("pipeline_field.jl")
using FastMultipole, Printf
using FastMultipole.StaticArrays
const FM = FastMultipole; const V = FLOWVPM
ext = Base.get_extension(FastMultipole, :FastMultipoleKAExt)
include(joinpath(@__DIR__, "pipeline_device_args.jl"))

function measured(f)
    f(); GC.gc(); st = Base.gc_num(); t0 = time_ns()
    v = f(); t1 = time_ns(); d = Base.GC_Diff(Base.gc_num(), st)
    (v, (t1-t0)/1e9, Base.gc_alloc_count(d), d.allocd)
end
rep(l,t,a,b) = @printf("  %-30s %8.4f s %12d allocs %10.2f MiB\n", l,t,a,b/2^20)

np = 512
pf_h = load_wake(36; np=np); pf_d = load_wake(36; np=np)
st = V.RadixFMMSettings(; precision=Float32, window_classes=256, m2l_strategy=:concat)
hcache = V._build_radix_fmm_cache(pf_h, st)
fmm!(pf_h, hcache)
dcache = build_ka_cache(ext, hcache, pf_d, hcache.ell)
state = dcache.state
sw = FM.DerivativesSwitch(FM.to_vector(false,1), FM.to_vector(true,1),
                          FM.to_vector(true,1), (pf_d,))

println("np=$np  phase split of ka_radix_cache_device_step!")
(_,t,a,b) = measured(() -> ext.ka_update_radix_state!(dcache, (pf_d,))); rep("ka_update_radix_state!",t,a,b)
(_,t,a,b) = measured(() -> ext.ka_lifecycle_body!(state));               rep("ka_lifecycle_body!",t,a,b)
(_,t,a,b) = measured(() -> ext.ka_finalize_radix_output!(state, (pf_d,);
        derivatives_switches=sw, host_output_staging=dcache.device_ctx.host_output,
        target_buffers=FM._radix_cache_target_buffers!(dcache, sw),
        device_target_buffers=dcache.device_ctx.device_target_buffers));  rep("ka_finalize_radix_output!",t,a,b)

println("\nlifecycle_body sub-stages:")
ws = state.scratch
(_,t,a,b) = measured(() -> ext.ka_launch_nearfield!(state; clear=true)); rep("  nearfield",t,a,b)
(_,t,a,b) = measured(() -> ext.ka_launch_b2m!(state));                   rep("  b2m",t,a,b)
(_,t,a,b) = measured(() -> ext.ka_launch_m2l!(state, ws));               rep("  m2l",t,a,b)
(_,t,a,b) = measured(() -> ext.ka_launch_l2b!(state));                   rep("  l2b",t,a,b)
