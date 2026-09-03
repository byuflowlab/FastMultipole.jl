# Leapfrogging vortex rings through FLOWVPM's full time integration (RK3,
# relaxation, UJ_fmm each stage) on the device -- the whole step, not one UJ.
#
# Same script for every arm; only the env decides the lifecycle:
#   MODE=ka      device field, current checkout (KA extension)
#   MODE=native  device field, pre-deletion checkout (native CUDA lifecycle)
#   MODE=cpu     host field, Float64 -- the reference
# Env: MODE  DEV_TF  NPHI (cross sections per ring)  NC (core layers)  NSTEPS  OUT
#
# Geometry is the package example (examples/vortexrings/run_leapfrog.jl): two
# coaxial rings, R = 0.7906, spacing R, core 0.1 R, circulation 1, at the
# example's dt = 0.01 R / U_ring.

const MODE = Symbol(get(ENV, "MODE", "ka"))
if MODE !== :cpu
    include("ka_backend.jl")
    dev_functional() || (println("$(DEV_NAME) not functional; skipping"); exit(0))
end
using FLOWVPM; const vpm = FLOWVPM
using LinearAlgebra, Printf, Statistics, Serialization, EllipticFunctions
import Roots, HCubature
import KernelAbstractions as KA
include(joinpath(dirname(pathof(vpm)), "..", "examples", "vortexrings", "vortexrings_functions.jl"))

const DEV_TF = MODE === :cpu ? Float64 : getfield(Base, Symbol(get(ENV, "DEV_TF", "Float32")))   # the CPU arm is always Float64
const NPHI   = parse(Int, get(ENV, "NPHI", "1000"))
const NC     = parse(Int, get(ENV, "NC", "6"))
const NSTEPS = parse(Int, get(ENV, "NSTEPS", "12"))
const OUT    = get(ENV, "OUT", joinpath(@__DIR__, "logs", "leapfrog"))
mkpath(OUT)

const R = 0.7906; const dZ = 0.7906; const Rcross = 0.1R; const sigma = Rcross
const nrings = 2
fmm_settings() = vpm.FMM(; p=5, autotune_p=false, autotune_ncrit=false,
                           autotune_reg_error=false, default_rho_over_sigma=1.0)
function make_field(maxp, TF; arraytype=Matrix, np=0)
    kw = arraytype === Matrix ? (;) : (; arraytype, np)
    return vpm.ParticleField(maxp, TF; formulation=vpm.rVPM, kernel=vpm.gaussianerf,
        viscous=vpm.Inviscid(), SFS=vpm.noSFS, transposed=true,
        integration=vpm.rungekutta3, UJ=vpm.UJ_fmm, fmm=fmm_settings(), kw...)
end

# host field with the two rings (always built in Float64)
np_ring = number_particles(NPHI, NC)
maxp = nrings * np_ring
host = make_field(maxp, Float64)
for ri in 1:nrings
    addvortexring(host, 1.0, R, 1.0, Rcross, NPHI, NC, sigma;
                  O=[0.0, 0.0, dZ * (ri - 1)], Oaxis=I, verbose=false)
end
n = vpm.get_np(host)
Uref = Uring(1.0, R, Rcross, 0.5)
dt = 0.01 * R / Uref
println("=== leapfrog: MODE=$(MODE) TF=$(DEV_TF) np=$n (Nphi=$NPHI nc=$NC) steps=$NSTEPS dt=$(round(dt, sigdigits=4)) ===")

# NATIVE_NF_UNBINNED=1 (native arm only): plain functor kernel over every pair
if MODE === :native && get(ENV, "NATIVE_NF_UNBINNED", "0") == "1"
    vpm.fmm.set_radix_setting!(:CUDA_NEARFIELD_BINNING, :unbinned)
end

function device_field(h, TF)
    d = make_field(h.maxparticles, TF; arraytype=devmatrix, np=h.np)
    d.particles .= devarray(TF.(Array(h.particles)))
    vpm.radix_fmm_settings!(d; m2l_strategy=:concat)
    return d
end
sync!(pf) = pf.particles isa Array ? nothing : KA.synchronize(KA.get_backend(pf.particles))

if MODE === :cpu
    pf = host
else
    # JIT warm-up on a throwaway copy so the timed run starts from t = 0
    warm = device_field(host, DEV_TF); vpm.nextstep(warm, dt; update_U_prev=false); sync!(warm)
    vpm.clear_radix_fmm_cache!(warm); delete!(vpm._radix_fmm_couplings, warm); GC.gc()
    pf = device_field(host, DEV_TF)
end

step_ms = Float64[]
t_all = time()
for s in 1:NSTEPS
    t0 = time(); vpm.nextstep(pf, dt; update_U_prev=false); sync!(pf)
    push!(step_ms, (time() - t0) * 1e3)
end
wall = time() - t_all
@printf("steps: median %.2f ms  min %.2f  max %.2f  total %.3f s\n",
        median(step_ms), minimum(step_ms), maximum(step_ms), wall)
println("per-step ms: ", join((@sprintf("%.1f", x) for x in step_ms), " "))

# UJ alone at the end state (what the earlier tables measured)
uj = Float64[]
for _ in 1:8
    vpm._reset_particles(pf); t0 = time(); vpm.UJ_fmm(pf); sync!(pf); push!(uj, (time() - t0) * 1e3)
end
@printf("UJ_fmm at end state: median %.2f ms (min %.2f)\n", median(uj), minimum(uj))

P = Array(pf.particles)
serialize(joinpath(OUT, "final_$(MODE)_$(DEV_TF).jls"),
          (; X=Float64.(P[vpm.X_INDEX, 1:n]), G=Float64.(P[vpm.GAMMA_INDEX, 1:n]),
             U=Float64.(P[vpm.U_INDEX, 1:n]), n, dt, NSTEPS, np_ring))
println("saved final state")
