# Host-side exhaustive check that the KA three-in-one harmonic walk returns
# BIT-IDENTICAL values to the shared `_resident_vortex_q` it replaces, over
# every (n,m) either kernel can ask for, at both precisions, including the
# coincident-point and on-axis degeneracies. No GPU needed: the functions are
# plain Julia.
include("ka_backend.jl")
using FastMultipole, Random, Printf
const FM = FastMultipole
ext = Base.get_extension(FastMultipole, :FastMultipoleKAExt)
ext !== nothing || error("FastMultipoleKAExt did not load")

const CNT = Ref(0); const BAD = Ref(0)
for TF in (Float32, Float64)
    Random.seed!(99)
    offsets = Any[]
    for _ in 1:200
        push!(offsets, (TF(randn()), TF(randn()), TF(randn())))
    end
    # degeneracies: coincident, on +z/-z axis (theta = 0, pi), in-plane (dz = 0),
    # and tiny/huge magnitudes
    append!(offsets, [(zero(TF),zero(TF),zero(TF)), (zero(TF),zero(TF),one(TF)),
                      (zero(TF),zero(TF),-one(TF)), (one(TF),zero(TF),zero(TF)),
                      (zero(TF),one(TF),zero(TF)), (TF(1e-6),TF(1e-6),TF(1e-6)),
                      (TF(1e4),TF(-1e4),TF(1e4))])
    for (dx,dy,dz) in offsets
        setup = FM._resident_harmonic_setup(dx,dy,dz)
        for nt in 0:10, mt in 0:nt
            a_re,a_im,b_re,b_im,c_re,c_im = ext.ka_vortex_q3(setup, nt, mt)
            for (m, gr, gi) in ((mt-1,a_re,a_im), (mt,b_re,b_im), (mt+1,c_re,c_im))
                er, ei = FM._resident_vortex_q(setup, nt, m)
                CNT[] += 1
                if !(isequal(gr,er) && isequal(gi,ei))
                    BAD[] += 1
                    BAD[] <= 5 && @printf("  MISMATCH %s (dx,dy,dz)=(%g,%g,%g) nt=%d m=%d: got (%.9g,%.9g) want (%.9g,%.9g)\n",
                        TF, dx,dy,dz, nt, m, gr, gi, er, ei)
                end
            end
        end
    end
end
@printf("ka_vortex_q3 vs _resident_vortex_q: %d comparisons, %d mismatches\n", CNT[], BAD[])
BAD[] == 0 || error("q3 walk is not bit-identical")
println("PASS: bit-identical")
