# Extra TARGETS (probes) on the all-pairs direct arm (:RADIX_DIRECT_ARM).
# That arm runs no lifecycle, so the local expansions and near lists left in
# the state are stale; the probes must be summed all-pairs against the packed
# bodies and match the exact host sum to roundoff.
using FastMultipole, Random, Printf, LinearAlgebra, Test
using FastMultipole.StaticArrays
const FM = FastMultipole
include(joinpath(@__DIR__, "..", "helpers", "vortex.jl"))
include(joinpath(@__DIR__, "ka_backend.jl"))

npass = Ref(0); nfail = Ref(0)
check(ok, msg) = (ok ? (npass[] += 1) : (nfail[] += 1); println(ok ? "  PASS  $msg" : "  FAIL  $msg"))
relerr(a, b) = maximum(abs.(a .- b)) / maximum(abs, b)

if !dev_functional()
    println("$(DEV_NAME) not functional; skipping")
    exit(0)
end

# targets: most among the particles, a few outside the unit box
function targets(nt, TFt)
    Random.seed!(7)
    xt = TFt.(rand(3, nt))
    xt[:, end-3:end] .= TFt.(1.5 .* rand(3, 4) .+ 1.0)
    return xt
end

let n = 3000, nt = 200, P = 5, ell = 3, DTF = (DEV_NAME == "Metal" ? Float32 : Float64)
    Random.seed!(1)
    pos = DTF.(rand(3, n)); str = DTF.(randn(3, n) ./ n)
    mk() = VortexParticles(copy(pos), copy(str), fill(DTF(0.01), n);
        potential = zeros(DTF, 13, n), gradient_stretching = zeros(DTF, 6, n))
    opts = FM.RadixLifecycleOptions(; precision = DTF,
        m2l_strategy = FM.ConcatenatedFixedZM2L(), body_type = FM.Point{FM.Vortex})
    FM.device_backend(::VortexParticles) = DEV_BACKEND
    xt = targets(nt, DTF)
    mkp() = (p = FM.ProbeSystem(nt, DTF); for i in 1:nt; p.position[i] = SVector{3,DTF}(xt[:, i]); end; p)

    sysa = mk(); probes_a = mkp()
    ca = RadixFMMCache(sysa; expansion_order = P, ell = ell, window_classes = 64,
                       options = opts, hessian = true, device = true)
    FM.set_radix_setting!(:RADIX_DIRECT_ARM, true)
    try
        FM.fmm!((sysa, probes_a), (sysa,), ca; scalar_potential = false, gradient = true, hessian = (true, false))
    finally
        FM.set_radix_setting!(:RADIX_DIRECT_ARM, false)
    end
    ga = reduce(hcat, probes_a.gradient)
    sysr = mk(); probes_r = mkp()
    FM.direct!((probes_r,), (sysr,); gradient = true)      # generic exact sum, the system's own kernel
    ea = relerr(ga, reduce(hcat, probes_r.gradient))
    tol_a = 5e-4                                            # the partitioned kernel's regularization tail
    check(ea <= tol_a, @sprintf("direct-arm probes are the all-pairs sum (%.2e, tol %.0e)", ea, tol_a))
end

@printf("\n%d passed, %d failed\n", npass[], nfail[])
nfail[] == 0 || error("$(nfail[]) check(s) failed")
println("gate passed: direct-arm probes summed all-pairs")
