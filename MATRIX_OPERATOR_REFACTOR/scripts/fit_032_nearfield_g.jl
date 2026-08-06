# Task 032 stage 2: fix the numerical constants of the erf-free gaussianerf
# g/h evaluation shipped in `RegularizedVortex` (src/translate_batched_resident.jl).
#
# Decides, by measurement against a 256-bit reference (stdlib only, local OK):
#   1. the Horner term counts for the theory-§3 series
#        g(ρ) = Aρ³ Σ_k (−1)^k ρ^{2k} / ((2k+3) 2^k k!)
#        h(ρ) = ρg′−3g = Aρ⁵ Σ_k (−1)^{k+1} ρ^{2k} / ((2k+5) 2^k k!)
#      needed to hold RELATIVE accuracy over the whole series branch ρ ∈ (0, ρ_c]
#      with ρ_c = 2 (the theory validated the 6/10-term counts only to ρ = 0.5;
#      the erf-free design needs the series to reach ρ_c, so the counts are
#      re-measured here — this is the "measured switch point" decision recorded
#      in the 032 work record);
#   2. the degree-3 least-squares coefficients of s(u), u = 1/ρ²,
#        ḡ(ρ) = e^{−ρ²/2} (Aρ + s(ρ)),  s(ρ) = erfc(ρ/√2) e^{ρ²/2}
#      fitted on [ρ_c, ρ_t] (031a §6.2), giving g = 1 − ḡ to ABSOLUTE accuracy
#      within the (ε/2)g(ρ_c) = 3.69e-4 budget at ε = 1e-3, with the error
#      decaying like e^{−ρ²/2} beyond ρ_t;
#   3. verification of the assembled two-branch evaluator in Float32 and
#      Float64 over ρ ∈ (0, 40] (max relative error on the series branch, max
#      absolute error against the budget on the outer branch).
#
# Output: ../data/kernel_splitting/nearfield_g_eval.csv plus Julia literals to
# paste into src (printed to stdout).

using Printf
using LinearAlgebra

setprecision(BigFloat, 256)

const A_BIG = sqrt(big(2) / big(pi))

# --- 256-bit reference (patterns from validate_031a_kernel_split.jl) ---------

function erf_series_big(x::BigFloat)
    s = zero(BigFloat)
    term = x
    n = 0
    while true
        add = term / (2n + 1)
        s += add
        n += 1
        term *= -x * x / n
        abs(add) < eps(BigFloat) * max(abs(s), one(BigFloat)) && break
    end
    return 2 / sqrt(big(pi)) * s
end

function erfc_big(x::BigFloat)
    x < 2 && return one(BigFloat) - erf_series_big(x)
    # bottom-up continued fraction: erfc(x) = e^{-x²}/(x√π) · 1/(1 + u/(1 + 2u/(1 + 3u/…)))
    u = 1 / (2 * x * x)
    cf = one(BigFloat)
    for k in 2000:-1:1
        cf = 1 + k * u / cf
    end
    return exp(-x * x) / (x * sqrt(big(pi))) / cf
end

# erf via the series only below x = 2 (its alternating terms reach ~e^{x²} and
# cancel catastrophically for large x); the continued fraction covers the rest.
erf_ref_big(x::BigFloat) = x < 2 ? erf_series_big(x) : 1 - erfc_big(x)
g_big(rho::BigFloat) = erf_ref_big(rho / sqrt(big(2))) - A_BIG * rho * exp(-rho^2 / 2)
gp_big(rho::BigFloat) = A_BIG * rho^2 * exp(-rho^2 / 2)
h_big(rho::BigFloat) = rho * gp_big(rho) - 3 * g_big(rho)
gbar_big(rho::BigFloat) = erfc_big(rho / sqrt(big(2))) + A_BIG * rho * exp(-rho^2 / 2)
s_big(rho::BigFloat) = erfc_big(rho / sqrt(big(2))) * exp(rho^2 / 2)

# sanity: erfc CF vs series at the switch
@assert abs(erfc_big(big(2.0)) / (1 - erf_series_big(big(2.0))) - 1) < big(1e-40)

# --- working-precision series (the exact form to ship) ------------------------

function g_series(rho::T, nterms::Int) where T<:AbstractFloat
    z = rho * rho
    p = zero(T)
    for k in (nterms - 1):-1:0
        coeff = T((isodd(k) ? -1 : 1) / ((2k + 3) * big(2)^k * factorial(big(k))))
        p = muladd(p, z, coeff)
    end
    return T(A_BIG) * rho * z * p
end

function h_series(rho::T, nterms::Int) where T<:AbstractFloat
    z = rho * rho
    p = zero(T)
    for k in (nterms - 1):-1:0
        coeff = T((iseven(k) ? -1 : 1) / ((2k + 5) * big(2)^k * factorial(big(k))))
        p = muladd(p, z, coeff)
    end
    return T(A_BIG) * rho * z * z * p
end

# --- outer-branch fit ---------------------------------------------------------

const RHO_C = 2.0
const RHO_T = 4.789          # rho_t at beta-independent eps = 1e-3 (031a §4)
const EPS_PHASE = 1e-3

function fit_s_poly(deg; m=400)
    rs = collect(range(RHO_C, RHO_T, length=m))
    V = [(1 / r^2)^j for r in rs, j in 0:deg]
    y = [Float64(s_big(big(r))) for r in rs]
    c = (V' * V) \ (V' * y)
    return c
end

# ships as: gbar = e * muladd-chain(c, u), g = 1 - gbar, h = rho^3*A*e - 3g
function g_h_outer(rho::T, c::Vector{Float64}) where T<:AbstractFloat
    z = rho * rho
    e = exp(-z / 2)
    u = inv(z)
    s = T(c[end])
    for j in (length(c) - 1):-1:1
        s = muladd(s, u, T(c[j]))
    end
    gbar = e * muladd(T(A_BIG), rho, s)
    g = one(T) - gbar
    rgp = T(A_BIG) * rho * z * e     # rho * g'(rho)
    return g, rgp - 3 * g
end

# --- sweeps -------------------------------------------------------------------

outdir = joinpath(@__DIR__, "..", "data", "kernel_splitting")
rows = String[]

# needed *relative* accuracy on the series branch: the phase gate is 1e-3 on U;
# aim two orders below with margin, capped by working precision.
grid_series(T) = T === Float32 ? range(1f-3, 2f0, length=4001) :
    range(1e-3, 2.0, length=4001)

best = Dict{DataType,Int}()
for T in (Float32, Float64)
    target = T === Float32 ? 3e-6 : 1e-11   # ~20x above each type's rounding floor
    chosen = 0
    for nterms in 4:24
        eg = 0.0
        eh = 0.0
        for rho in grid_series(T)
            rb = big(Float64(rho))
            eg = max(eg, abs(Float64(g_series(T(rho), nterms)) / Float64(g_big(rb)) - 1))
            eh = max(eh, abs(Float64(h_series(T(rho), nterms)) / Float64(h_big(rb)) - 1))
        end
        push!(rows, @sprintf("series,%s,%d,%.3e,%.3e,,", T, nterms, eg, eh))
        if chosen == 0 && max(eg, eh) <= target
            chosen = nterms
        end
        # stop once well past the rounding plateau
        nterms > 12 && chosen > 0 && break
    end
    chosen == 0 && error("no term count meets the $T series target")
    best[T] = chosen
end

# outer branch: fit once (degree 3 per 031a §6.2), then measure both precisions
c3 = fit_s_poly(3)
budget = 0.5 * EPS_PHASE * Float64(g_big(big(RHO_C)))
for T in (Float32, Float64)
    eabs = 0.0     # absolute error in g (== error in gbar)
    ehabs = 0.0    # absolute error in h
    for rho in range(RHO_C, 40.0, length=8001)
        g, h = g_h_outer(T(rho), c3)
        rb = big(rho)
        eabs = max(eabs, abs(Float64(g) - Float64(g_big(rb))))
        ehabs = max(ehabs, abs(Float64(h) - Float64(h_big(rb))))
    end
    push!(rows, @sprintf("outer,%s,3,%.3e,%.3e,%.3e,%d", T, eabs, ehabs, budget,
        eabs <= budget))
end

# assembled evaluator: series below RHO_C with the chosen counts, outer above
function g_h_assembled(rho::T) where T
    rho <= T(RHO_C) ? (g_series(rho, best[T]), h_series(rho, best[T])) :
        g_h_outer(rho, c3)
end
for T in (Float32, Float64)
    erel = 0.0
    for rho in grid_series(T)
        g, h = g_h_assembled(T(rho))
        rb = big(Float64(rho))
        erel = max(erel, abs(Float64(g) / Float64(g_big(rb)) - 1),
            abs(Float64(h) / Float64(h_big(rb)) - 1))
    end
    eabs = 0.0
    for rho in range(RHO_C + 1e-9, 40.0, length=8001)
        g, h = g_h_assembled(T(rho))
        rb = big(rho)
        eabs = max(eabs, abs(Float64(g) - Float64(g_big(rb))))
    end
    push!(rows, @sprintf("assembled,%s,%d,%.3e,%.3e,%.3e,%d", T, best[T], erel,
        eabs, budget, eabs <= budget))
end

mkpath(outdir)
open(joinpath(outdir, "nearfield_g_eval.csv"), "w") do io
    println(io, "branch,precision,nterms_or_degree,err1_relg_or_absg,err2_relh_or_absh,budget_abs,meets_budget")
    foreach(r -> println(io, r), rows)
end

println("chosen series terms: Float32 => $(best[Float32]), Float64 => $(best[Float64])")
println("degree-3 s(u) coefficients on [ρ_c=$(RHO_C), ρ_t=$(RHO_T)] (paste into src):")
for (j, cj) in enumerate(c3)
    @printf("    c%d = %.17g\n", j - 1, cj)
end
println("budget |δg| (absolute, outer branch) = ", @sprintf("%.3e", budget))
println("rows written: ", joinpath(outdir, "nearfield_g_eval.csv"))
