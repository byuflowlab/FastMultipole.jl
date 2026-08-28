# k03c — intermediate dump for the exact-axis parity offenders (debug aid).
# Run: julia --project=../FLOWPanel.jl k03c_axis_dump.jl

include(joinpath(@__DIR__, "linegauss.jl"))
using .LineGauss
import FLOWPanel as pnl
using StaticArrays, LinearAlgebra, Random, Printf

const σ = 1.0
rng = MersenneTwister(3)
function config(rng, L, ξ, h)
    t̂ = normalize(randn(rng, SVector{3,Float64}))
    tmp = normalize(cross(t̂, normalize(randn(rng, SVector{3,Float64}))))
    P1 = randn(rng, SVector{3,Float64})
    P2 = P1 + L * t̂
    x = P1 + ξ * L * t̂ + h * tmp
    return P1 - x, P2 - x
end

# regenerate ALL configs in k03 order so rng state matches, keep the wanted ones
want = Set([(1.79, -0.5, 0.0), (0.5, 0.99, 0.0), (3.0, 0.5, 0.0)])
picked = Dict{Tuple{Float64,Float64,Float64},Tuple{SVector{3,Float64},SVector{3,Float64}}}()
for L in (2e-3, 0.05, 0.124, 0.5, 1.79, 3.0, 16.0, 1000.0),
    ξ in (-0.5, -0.01, 0.1, 0.5, 0.99, 1.5),
    h in (0.0, 1e-10, 1e-6, 1e-3, 0.05, 0.124, 0.3, 1.0, 3.0, 5.9, 8.0, 12.0)
    r1, r2 = config(rng, L, ξ, h)
    (L, ξ, h) in want && (picked[(L, ξ, h)] = (r1, r2))
end

fam = Val(pnl.LineGaussRegularization)
for (key, (r1, r2)) in sort(collect(picked); by=first)
    L, ξ, h = key
    println("--- config L=$L ξ=$ξ h=$h ---")
    s = r1 - r2; B = dot(s, s); Lg = sqrt(B)
    that = -s / Lg
    z1 = -dot(that, r1)
    ẑ1a = z1 / σ                       # prototype _geom rounding
    ẑ2a = (z1 - Lg) / σ
    c = cross(r1, r2)
    ĥ2 = dot(c, c) / (B * σ * σ)
    R̂1 = norm(r1) / σ; R̂2 = norm(r2) / σ
    @printf("ĥ2=%.6e ẑ1=%.17g ẑ2=%.17g R̂1=%.6g R̂2=%.6g\n", ĥ2, ẑ1a, ẑ2a, R̂1, R̂2)
    @printf("guards: small=%d endpoint=%d axis=%d\n",
            max(R̂1, R̂2) < 0.125,
            LineGauss.endpoint_split_guard(ĥ2, ẑ1a, ẑ2a, R̂1, R̂2),
            LineGauss.axis_guard(ĥ2, ẑ1a, ẑ2a))
    hvec = -r1 - (σ * ẑ1a) * that
    @printf("norm(hvec)=%.6e\n", norm(hvec))
    Mp, _ = LineGauss.lg_M(ẑ1a, ẑ2a, ĥ2, R̂1, R̂2)
    Mf = pnl._lg_M(ẑ1a, ẑ2a, ĥ2, R̂1, R̂2)
    @printf("M proto=%.17g  M port=%.17g  reldiff=%.3e\n", Mp, Mf, abs(Mp - Mf) / max(abs(Mp), 1e-300))
    u_p = lg_velocity(r1, r2, σ)
    u_f = pnl._bound_vortex_velocity(r1, r2, true, σ, fam)
    @printf("|u_p|=%.3e |u_f|=%.3e dv=%.3e\n", norm(u_p), norm(u_f),
            norm(u_f - u_p) / max(norm(u_p), 1e-300))
    G_p = lg_gradient(r1, r2, σ)
    G_f = pnl._bound_vortex_gradient(r1, r2, true, σ, fam)
    println("G_p = ", round.(G_p; sigdigits=6))
    println("G_f = ", round.(G_f; sigdigits=6))
    # radial factors on the endpoint-split path
    if LineGauss.endpoint_split_guard(ĥ2, ẑ1a, ẑ2a, R̂1, R̂2)
        Mp2, Dp2 = LineGauss.endpoint_split_MD(ẑ1a, ẑ2a, ĥ2)
        Mf2, Df2 = pnl._lg_endpoint_split_MD(ẑ1a, ẑ2a, ĥ2)
        @printf("split proto M=%.17g D=%.17g | port M=%.17g D=%.17g\n", Mp2, Dp2, Mf2, Df2)
    end
end
