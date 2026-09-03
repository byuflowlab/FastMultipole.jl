# GPU all-pairs UJ_direct: the tiled CUDA kernels (pre-consolidation FLOWVPM,
# MODE=native) vs the brute-force KA kernel (MODE=ka), same script.
# Env: MODE=ka|native  DEV_TF  NPS  CALLS
include("ka_backend.jl")
dev_functional() || (println("$(DEV_NAME) not functional; skipping"); exit(0))
using FLOWVPM; const vpm = FLOWVPM
using Printf, Statistics, Random
import KernelAbstractions as KA
const MODE  = Symbol(get(ENV, "MODE", "ka"))
const TF    = getfield(Base, Symbol(get(ENV, "DEV_TF", "Float32")))
const NPS   = [parse(Int, s) for s in split(get(ENV, "NPS", "8192,16384,32768,65536"), ",")]
const CALLS = parse(Int, get(ENV, "CALLS", "10"))
fmmset() = vpm.FMM(; p=5, autotune_p=false, autotune_ncrit=false, autotune_reg_error=false, default_rho_over_sigma=1.0)
function field(n, R; kw...)
    pf = vpm.ParticleField(n, R; formulation=vpm.rVPM, kernel=vpm.gaussianerf, viscous=vpm.Inviscid(),
        SFS=vpm.noSFS, transposed=true, integration=vpm.rungekutta3, UJ=vpm.UJ_direct, fmm=fmmset(), kw...)
    rng = MersenneTwister(7 + n); sigma = 2.0 * (1.0 / n)^(1 / 3)
    for _ in 1:n
        vpm.add_particle(pf, rand(rng, 3), (2 .* rand(rng, 3) .- 1) ./ n, sigma)
    end
    return pf
end
relmax(a, b) = maximum(abs.(a .- b)) / maximum(abs.(b))
println("=== UJ_direct all-pairs: MODE=$(MODE) $(DEV_NAME) $(TF) calls=$(CALLS) ===")
@printf("%8s | %10s %10s | %10s %10s\n", "np", "med ms", "min ms", "U relerr", "J relerr")
for n in NPS
    host = field(n, Float64)
    vpm.UJ_direct(host)
    Uref = copy(host.particles[vpm.U_INDEX, 1:n]); Jref = copy(host.particles[vpm.J_INDEX, 1:n])
    dev = field(n, TF; arraytype=devmatrix, np=0)
    dev.particles .= devarray(TF.(Array(host.particles)))
    vpm._reset_particles(dev); vpm.UJ_direct(dev); KA.synchronize(KA.get_backend(dev.particles))
    ts = Float64[]
    for _ in 1:CALLS
        vpm._reset_particles(dev); t0 = time(); vpm.UJ_direct(dev); KA.synchronize(KA.get_backend(dev.particles))
        push!(ts, (time() - t0) * 1e3)
    end
    P = Array(dev.particles)
    @printf("%8d | %10.2f %10.2f | %10.3e %10.3e\n", n, median(ts), minimum(ts),
            relmax(Float64.(P[vpm.U_INDEX, 1:n]), Uref), relmax(Float64.(P[vpm.J_INDEX, 1:n]), Jref))
    flush(stdout)
end
