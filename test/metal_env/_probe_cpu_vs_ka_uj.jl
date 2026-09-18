# End-to-end: does the KA device route still agree with the CPU route after the
# session-41 harmonic-sweep rewrite of B2M and L2B?
#
# Three arms on the SAME real wake, over a size ladder:
#   direct  -- V.UJ_direct, Float64 all-pairs. Ground truth for both.
#   cpu     -- V.UJ_fmm on a host Float64 field: FastMultipole's CPU route,
#              which the rewrite does NOT touch (it lives in the KA extension).
#   ka      -- V.UJ_fmm on a device Float32 field: the rewritten path.
# Scored on U and on the full 9-component J (the hessian branch of L2B, which
# is what production takes and what the rewrite changed most).
#
# The ka-vs-cpu column is the one that matters: it must not have moved from
# where it was before the rewrite (Float32-vs-Float64 discretisation, ~1e-3).

include("ka_backend.jl")
include("pipeline_field.jl")
using FastMultipole, Printf, Statistics
const FM = FastMultipole
const V = FLOWVPM
dev_functional() || (println("skipping"); exit(0))

const STEP = 36
const P = 5
const SIZES = (512, 2048, 8192, nothing)

mkf(TF) = V.FMM(; p=P+1, autotune_p=false, autotune_ncrit=false,
                  autotune_reg_error=false, default_rho_over_sigma=1.0)

relerr(a, b) = (s = maximum(abs.(b)); s == 0 ? maximum(abs.(a .- b)) : maximum(abs.(a .- b))/s)
rmserr(a, b) = (s = sqrt(mean(abs2, b)); s == 0 ? sqrt(mean(abs2, a .- b)) : sqrt(mean(abs2, a .- b))/s)

const JROWS = collect(V.J_INDEX)

@printf("%7s | %10s %10s | %10s %10s | %10s %10s\n",
        "np", "U ka/dir", "U cpu/dir", "J ka/dir", "J cpu/dir", "U ka/cpu", "J ka/cpu")
for np in SIZES
    truth = load_wake(STEP; np=np, TF=Float64, P=P)
    NP = V.get_np(truth)
    V.UJ_direct(truth)
    Ut = Array(truth.particles)[V.U_INDEX, 1:NP]
    Jt = Array(truth.particles)[JROWS, 1:NP]

    hostf = load_wake(STEP; np=np, TF=Float64, P=P)
    V.UJ_fmm(hostf; autotune=false)
    Uc = Array(hostf.particles)[V.U_INDEX, 1:NP]
    Jc = Array(hostf.particles)[JROWS, 1:NP]

    src = load_wake(STEP; np=np, TF=Float64, P=P)
    d = V.ParticleField(src.maxparticles, Float32; arraytype=devmatrix, np=src.np, fmm=mkf(Float32))
    d.particles .= devarray(Float32.(Array(src.particles)))
    V.radix_fmm_settings!(d; m2l_strategy=:concat)
    V.UJ_fmm(d)
    Uk = Float64.(Array(d.particles)[V.U_INDEX, 1:NP])
    Jk = Float64.(Array(d.particles)[JROWS, 1:NP])

    @printf("%7d | %10.3e %10.3e | %10.3e %10.3e | %10.3e %10.3e\n",
            NP, relerr(Uk,Ut), relerr(Uc,Ut), relerr(Jk,Jt), relerr(Jc,Jt),
            relerr(Uk,Uc), relerr(Jk,Jc))
    flush(stdout)
    haskey(V._radix_fmm_couplings, d) && (V.clear_radix_fmm_cache!(d); delete!(V._radix_fmm_couplings, d))
    GC.gc(); GC.gc()
end
