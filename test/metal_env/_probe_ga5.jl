include("ka_backend.jl"); include("pipeline_field.jl")
using FastMultipole, Printf
using FastMultipole.StaticArrays
const FM = FastMultipole; const V = FLOWVPM
ext = Base.get_extension(FastMultipole, :FastMultipoleKAExt)
include(joinpath(@__DIR__, "pipeline_device_args.jl"))
function measured(f)
    GC.gc(); st = Base.gc_num(); t0 = time_ns()
    v = f(); t1 = time_ns(); d = Base.GC_Diff(Base.gc_num(), st)
    (v, (t1-t0)/1e9, Base.gc_alloc_count(d), d.allocd)
end
rep(l,t,a,b) = @printf("  %-30s %8.4f s %12d allocs %10.2f MiB\n", l,t,a,b/2^20)
np = 512
pf_h = load_wake(36; np=np); pf_d = load_wake(36; np=np)
stg = V.RadixFMMSettings(; precision=Float32, window_classes=256, m2l_strategy=:concat)
hcache = V._build_radix_fmm_cache(pf_h, stg)
dcache = build_ka_cache(ext, hcache, pf_d, hcache.ell)
state = dcache.state
println("=== ka_lifecycle_body! called 8x, NO warmup discarded ===")
for i in 1:8
    (_,t,a,b) = measured(() -> ext.ka_lifecycle_body!(state)); rep("call $i",t,a,b); flush(stdout)
end
println("=== ka_fmm! called 5x ===")
for i in 1:5
    (_,t,a,b) = measured(() -> ext.ka_fmm!(pf_d, dcache; hessian=true)); rep("ka_fmm! $i",t,a,b); flush(stdout)
end
