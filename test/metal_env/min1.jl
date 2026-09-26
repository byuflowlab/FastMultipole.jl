include("ka_backend.jl")
using FastMultipole, Random, Test, Printf, LinearAlgebra
using FastMultipole.StaticArrays
const FM = FastMultipole
include(joinpath(@__DIR__, "..", "vortex.jl"))
make_system(seed, n, TF) = (Random.seed!(seed);
    VortexParticles(rand(TF, 3, n), (randn(TF, 3, n) ./ TF(n)), zeros(TF, n);
        potential=zeros(TF, 13, n), gradient_stretching=zeros(TF, 6, n)))
TF=Float32; P=4; ell=3; n=256; wc=8
println("A"); flush(stdout)
sys_h = make_system(6101, n, TF)
opts = FM.CUDARadixLifecycleOptions(; precision=TF, m2l_strategy=FM.ConcatenatedFixedZM2L(), body_type=FM.Point{FM.Vortex})
println("B"); flush(stdout)
hcache = RadixFMMCache(sys_h; expansion_order=P, ell=ell, window_classes=wc, options=opts)
println("C"); flush(stdout)
fmm!(sys_h, hcache)
println("D"); flush(stdout)
