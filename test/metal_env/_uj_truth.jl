include("ka_backend.jl"); include("pipeline_field.jl")
using FastMultipole, Printf
const V = FLOWVPM
const TF = Float32; const STEP = 36; const P = 5
mkf() = V.FMM(; p=P+1, autotune_p=false, autotune_ncrit=false,
                autotune_reg_error=false, default_rho_over_sigma=1.0)
devrun(np, strat) = begin
    f = load_wake(STEP; np, TF, P)
    d = V.ParticleField(f.maxparticles, TF; arraytype=devmatrix, np=f.np, fmm=mkf())
    d.particles .= devarray(f.particles)
    V.radix_fmm_settings!(d; m2l_strategy=strat)
    t = @elapsed V.UJ_fmm(d)
    (Array(d.particles)[V.U_INDEX, 1:d.np], t)
end
for np in (512, 2048, 8192)
    truth = load_wake(STEP; np, TF, P); V.UJ_direct(truth)
    Ut = Array(truth.particles)[V.U_INDEX, 1:truth.np]; scale = maximum(abs.(Ut))
    Uc, tc = devrun(np, :concat)
    Ud, td = devrun(np, :dense)
    @printf("np=%-6d concat-vs-direct %.3e (%.2fs)  dense-vs-direct %.3e (%.2fs)  dense-vs-concat %.3e\n",
        np, maximum(abs.(Uc.-Ut))/scale, tc, maximum(abs.(Ud.-Ut))/scale, td,
        maximum(abs.(Ud.-Uc))/scale); flush(stdout)
end
