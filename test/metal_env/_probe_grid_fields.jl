include(joinpath(@__DIR__,"ka_backend.jl"))
using FastMultipole, Random, Printf
using FastMultipole.StaticArrays
using KernelAbstractions
const FM = FastMultipole
include("/Users/bvarela/Library/CloudStorage/Box-Box/Research/FLOWUnsteady-update/FastMultipole/test/vortex.jl")
dev_functional() || (println("no dev"); exit(0))
ext = Base.get_extension(FastMultipole, :FastMultipoleKAExt)
mk(seed,n,TF)=(Random.seed!(seed); VortexParticles(rand(TF,3,n),(randn(TF,3,n)./TF(n)),zeros(TF,n);
    potential=zeros(TF,13,n), gradient_stretching=zeros(TF,6,n)))
n, ell, TF = 512, 3, Float32
sh = mk(11,n,TF); sd = mk(11,n,TF)
opts=FM.CUDARadixLifecycleOptions(;precision=TF,m2l_strategy=FM.ConcatenatedFixedZM2L(),body_type=FM.Point{FM.Vortex})
hc = RadixFMMCache(sh; expansion_order=4, ell=ell, window_classes=256, options=opts)
fmm!(sh, hc)
hg = hc.state.grid
println("host grid type: ", typeof(hg).name.name)
for f in fieldnames(typeof(hg))
    v = getfield(hg, f)
    println(rpad(string(f),16), " ", rpad(string(typeof(v).name.name),20),
        v isa AbstractArray ? string(size(v)) : string(v))
end
println("\nlevel_offsets = ", hc.level_offsets)
