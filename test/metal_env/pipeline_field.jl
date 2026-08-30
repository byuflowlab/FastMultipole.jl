# Chunk A of the top-to-bottom pipeline bench: a REAL particle field.
#
# Every existing ka_* suite builds its system as `VortexParticles(rand(3,n),
# randn(3,n)./n, zeros(n))` -- uniform noise in the unit cube with sigma = 0.
# That is not what FLOWVPM hands FastMultipole: a wake is spatially structured
# (a thin, curved, locally dense tube in a mostly empty box) and every particle
# carries a smoothing radius, which is what drives `_radix_auto_geometry`'s
# choice of `ell` and `near_radius2` (FLOWVPM_fmm_radix.jl:533). With sigma = 0
# the adequacy term cannot bind, so the suites never exercise the depth rule
# production actually uses.
#
# This file builds a vortex-ring wake: `nrings` rings of `nper` particles each,
# advected along +x with a small jitter, sigma set from the particle spacing so
# the overlap ratio is the ~1.5-2.5 a real FLOWVPM run carries.

using FLOWVPM, Random
const vpm = FLOWVPM

"""
    make_wake(np; seed, R, overlap, nper, TF) -> ParticleField

`np` particles as a train of vortex rings. Returns a CPU (Array-backed) field,
so `_build_radix_fmm_cache` takes the `device=false` host branch.
"""
function make_wake(np::Int; seed::Int=20260830, Rring::Real=1.0, overlap::Real=2.0,
                   nper::Int=64, TF=Float64)
    rng = MersenneTwister(seed)
    nrings = max(1, cld(np, nper))
    pfield = vpm.ParticleField(np, TF)

    dtheta = 2pi / nper
    h = Rring * dtheta               # in-ring particle spacing
    dx = h                        # ring-to-ring spacing, same order
    sigma = TF(overlap * h)       # overlap = sigma / spacing
    # circulation per particle for a ring of total strength Gamma_ring
    Gamma_ring = TF(1.0)

    added = 0
    for ir in 1:nrings, ip in 1:nper
        added >= np && break
        theta = (ip - 1) * dtheta + 0.37 * (ir - 1)   # stagger successive rings
        x = TF((ir - 1) * dx + 0.02 * h * randn(rng))
        y = TF(Rring * cos(theta) + 0.02 * h * randn(rng))
        z = TF(Rring * sin(theta) + 0.02 * h * randn(rng))
        # vorticity tangent to the ring, magnitude Gamma_ring * h
        gx = TF(0.0)
        gy = TF(-Gamma_ring * h * sin(theta))
        gz = TF( Gamma_ring * h * cos(theta))
        vpm.add_particle(pfield, (x, y, z), (gx, gy, gz), sigma; vol=TF(h^3))
        added += 1
    end
    return pfield
end

"Report the geometry the auto rule will see, without building a cache."
function wake_stats(pfield)
    np = vpm.get_np(pfield)
    X = zeros(3, np)
    sig = zeros(np)
    for i in 1:np
        p = vpm.get_particle(pfield, i)
        X[:, i] .= vpm.get_X(p)
        sig[i] = vpm.get_sigma(p)[]
    end
    lo = minimum(X; dims=2); hi = maximum(X; dims=2)
    (np=np, box_lo=vec(lo), box_hi=vec(hi), L=maximum(hi .- lo),
     sigma_min=minimum(sig), sigma_max=maximum(sig))
end

################################################################################
# Real FLOWUnsteady wake, loaded from an HDF5 pfield dump
################################################################################
#
# `make_wake` above is a synthetic stand-in. These load an ACTUAL rotor wake
# written by FLOWUnsteady (`for_ryan/NREL_50_36_2_1.125`), which is what the
# production path really sees: a rotor wake is a set of interleaved helical
# sheets, far denser near the blades than in the far wake, with a spread of
# sigma rather than one value. Subsampling keeps `np` small while preserving
# that structure -- a uniformly-strided subsample of a helix is still a helix.

using HDF5

const WAKE_DIR = "/Users/bvarela/Downloads/for_ryan/NREL_50_36_2_1.125"
const WAKE_CASE = "NREL_50_36_2_1.125"

wake_path(step::Int) = joinpath(WAKE_DIR, "$(WAKE_CASE)_pfield.$(step).h5")

"""
    load_wake(step; np=nothing, TF=Float64) -> ParticleField

Read timestep `step` of the FLOWUnsteady dump. If `np` is given, take a
uniformly-strided subsample of that many particles (stride, not the first `np`,
so the whole wake is represented rather than one blade's worth).
"""
function load_wake(step::Int; np::Union{Nothing,Int}=nothing, TF=Float64, P::Int=4)
    X, Gamma, sigma, vol, circ, static = h5open(wake_path(step)) do h
        (read(h["X"]), read(h["Gamma"]), read(h["sigma"]),
         read(h["vol"]), read(h["circulation"]), read(h["static"]))
    end
    n_all = length(sigma)
    idx = np === nothing || np >= n_all ? (1:n_all) :
        round.(Int, range(1, n_all; length=np))
    n = length(idx)
    # Autotuning must be off: the radix path fixes p/ncrit/rho at cache
    # construction and `_validate_radix_fmm_settings` (FLOWVPM_fmm_radix.jl:381)
    # rejects a field whose FMM still has the autotune flags set.
    pfield = vpm.ParticleField(n, TF;
        fmm=vpm.FMM(; p=P + 1, autotune_p=false, autotune_ncrit=false,
                      autotune_reg_error=false, default_rho_over_sigma=1.0))
    for i in idx
        vpm.add_particle(pfield,
            (TF(X[1,i]), TF(X[2,i]), TF(X[3,i])),
            (TF(Gamma[1,i]), TF(Gamma[2,i]), TF(Gamma[3,i])),
            TF(sigma[i]); vol=TF(vol[i]), circulation=TF(circ[i]),
            static=static[i] != 0)
    end
    return pfield
end
