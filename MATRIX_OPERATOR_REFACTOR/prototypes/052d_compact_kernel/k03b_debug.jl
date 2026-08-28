# k03b — isolate P3 parity offenders (debug aid, not a gate).
# Run: JULIA_NUM_THREADS=4 julia --project=../FLOWPanel.jl k03b_debug.jl

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
fam = Val(pnl.LineGaussRegularization)
rows = Tuple{Float64,NTuple{3,Float64},Float64,Float64,SVector{3,Float64},SVector{3,Float64}}[]
nanrows = 0
for L in (2e-3, 0.05, 0.124, 0.5, 1.79, 3.0, 16.0, 1000.0),
    ξ in (-0.5, -0.01, 0.1, 0.5, 0.99, 1.5),
    h in (0.0, 1e-10, 1e-6, 1e-3, 0.05, 0.124, 0.3, 1.0, 3.0, 5.9, 8.0, 12.0)
    r1, r2 = config(rng, L, ξ, h)
    (norm(r1) < 5eps() || norm(r2) < 5eps()) && continue
    u_p = lg_velocity(r1, r2, σ)
    u_f = pnl._bound_vortex_velocity(r1, r2, true, σ, fam)
    G_p = lg_gradient(r1, r2, σ)
    G_f = pnl._bound_vortex_gradient(r1, r2, true, σ, fam)
    dv = norm(u_f - u_p) / max(norm(u_p), 1e-300)
    dg = norm(G_f - G_p) / max(norm(G_p), 1e-300)
    if isnan(dg) || isnan(dv)
        global nanrows += 1
        if nanrows <= 6
            @printf("NaN: L=%.3g ξ=%.3g h=%.3g  nan(u_p)=%d nan(u_f)=%d nan(G_p)=%d nan(G_f)=%d\n",
                    L, ξ, h, any(isnan, u_p), any(isnan, u_f), any(isnan, G_p), any(isnan, G_f))
            # branch diagnostics
            s = r1 - r2; B = dot(s, s); Lg = sqrt(B)
            that = -s / Lg
            ẑ1 = -dot(that, r1) / σ; ẑ2 = ẑ1 - Lg / σ
            c = cross(r1, r2); ĥ2 = dot(c, c) / (B * σ^2)
            @printf("     ẑ1=%.6g ẑ2=%.6g ĥ2=%.6g R̂1=%.6g R̂2=%.6g\n",
                    ẑ1, ẑ2, ĥ2, norm(r1), norm(r2))
        end
    else
        push!(rows, (max(dv, dg), (L, ξ, h), dv, dg, u_p, u_f))
    end
end
sort!(rows; by=first, rev=true)
println("nan configs: ", nanrows)
println("worst finite parity offenders:")
for (m, (L, ξ, h), dv, dg, u_p, u_f) in rows[1:min(8, length(rows))]
    r1, r2 = nothing, nothing
    @printf("  L=%.3g ξ=%.3g h=%.3g  dv=%.3e dg=%.3e  |u|=%.3e\n", L, ξ, h, dv, dg, norm(u_p))
end
