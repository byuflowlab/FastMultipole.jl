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
rep(l,t,a,b) = @printf("  %-34s %8.4f s %12d allocs %10.2f MiB\n", l,t,a,b/2^20)
np = 512
pf_h = load_wake(36; np=np); pf_d = load_wake(36; np=np)
stg = V.RadixFMMSettings(; precision=Float32, window_classes=256, m2l_strategy=:concat)
hcache = V._build_radix_fmm_cache(pf_h, stg)
fmm!(pf_h, hcache)
dcache = build_ka_cache(ext, hcache, pf_d, hcache.ell)
state = dcache.state; ws = state.scratch
G = ws.m2m_groups
ap(g) = ext.ka_resident_stage_group_apply!(state.multipoles, state.multipoles, g, ws, :m2m)
bk = KernelAbstractions.get_backend(state.output)
loop() = (for g in G; ap(g); end)
println("--- same closure, 6 consecutive measured() calls")
for i in 1:6
    (_,t,a,b) = measured(loop); rep("call $i",t,a,b)
end
println("--- alloc profile of the loop")
using Profile
Profile.Allocs.clear()
Profile.Allocs.@profile sample_rate=0.001 (for _ in 1:3; loop(); end)
res = Profile.Allocs.fetch()
println("nsamples=", length(res.allocs))
using Base: StackTraces
counts = Dict{String,Int}()
for a in res.allocs
    for fr in a.stacktrace
        k = string(fr.func, " @ ", fr.file, ":", fr.line)
        counts[k] = get(counts,k,0)+1
    end
end
for (k,v) in sort(collect(counts), by=x->-x[2])[1:min(35,end)]
    @printf("  %7d  %s\n", v, k)
end
flush(stdout)
