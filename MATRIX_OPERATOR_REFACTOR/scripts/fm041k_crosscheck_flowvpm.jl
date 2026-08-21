# fm041k_crosscheck_flowvpm.jl — validate the 041k transcribed pair math
# against production FLOWVPM (CPU path): UJ_direct(sfs=true) with the
# gaussianerf kernel and transposed scheme, on the pre-registered 041k body
# set at n = 1e3. Run LOCALLY (≤4 threads) from the FLOWVPM.jl project env:
#   cd ../FLOWVPM.jl && julia --project=. -t 4 \
#       ../FastMultipole/MATRIX_OPERATOR_REFACTOR/scripts/fm041k_crosscheck_flowvpm.jl
# Compares U, J, and SFS (= E) blocks; passes if max rel err ≤ 1e-11.

import FLOWVPM
const vpm = FLOWVPM
using Printf

include(joinpath(@__DIR__, "fm041k_direct_bruteforce.jl"))  # library mode

const NCHK = 1_000

ref, _ = build_reference(NCHK)

pfield = vpm.ParticleField(NCHK; kernel=vpm.gaussianerf, transposed=true)
for i in 1:NCHK
    vpm.add_particle(pfield, ref.P[:, i], ref.G[:, i], 1.0 / ref.si)
end
vpm.UJ_direct(pfield; sfs=true, reset=true, reset_sfs=true)

U = zeros(3, NCHK); J = zeros(9, NCHK); S = zeros(3, NCHK)
for i in 1:NCHK
    U[:, i] .= pfield.particles[vpm.U_INDEX, i]
    J[:, i] .= pfield.particles[vpm.J_INDEX, i]
    S[:, i] .= pfield.particles[vpm.SFS_INDEX, i]
end

fail = false
for (blk, mine, prod) in (("U", ref.U, U), ("J", ref.J, J), ("E/SFS", ref.E, S))
    m, r = relerr(mine, prod)
    @printf("crosscheck %-5s vs FLOWVPM: max rel %.3e  rms rel %.3e\n", blk, m, r)
    global fail |= m > 1e-11
end
fail ? error("041k cross-check vs FLOWVPM FAILED") :
       println("041k cross-check vs FLOWVPM PASSED")
