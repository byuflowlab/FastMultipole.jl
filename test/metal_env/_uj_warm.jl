include("ka_backend.jl"); include("pipeline_field.jl")
using FastMultipole, Printf
const V = FLOWVPM
const TF = Float32; const STEP = 36; const P = 5
mkf() = V.FMM(; p=P+1, autotune_p=false, autotune_ncrit=false,
                autotune_reg_error=false, default_rho_over_sigma=1.0)
for np in (2048, 8192)
    for strat in (:concat, :dense)
        f = load_wake(STEP; np, TF, P)
        d = V.ParticleField(f.maxparticles, TF; arraytype=devmatrix, np=f.np, fmm=mkf())
        d.particles .= devarray(f.particles)
        V.radix_fmm_settings!(d; m2l_strategy=strat)
        ts = [ (@elapsed V.UJ_fmm(d)) for _ in 1:6 ]
        @printf("np=%-6d %-7s  ramp %s   plateau %.4f s\n", np, strat,
            join((@sprintf("%.3f", t) for t in ts), "/"), minimum(ts[4:6]))
        flush(stdout)
    end
end
