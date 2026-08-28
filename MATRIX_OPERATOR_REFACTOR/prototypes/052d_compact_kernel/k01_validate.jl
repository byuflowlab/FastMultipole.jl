# k01 — validation of the LineGauss closed-form kernel (DERIVATION.md).
# T1 erf refs; T2 closed form vs quadrature; T3 singular limit + convention
# check vs FLOWPanel; T4 infinite-line limit = shipped Gaussian; T5 gradient
# vs finite differences; T6 axis-guard seam; T7 error decay vs segment
# distance + Δr(tol) for {LineGauss, shipped Gaussian, shipped Compact};
# T8 peak velocity/gradient (phase_00 table extension).
#
# Run: JULIA_NUM_THREADS=4 julia --project=../FLOWPanel.jl k01_validate.jl

include(joinpath(@__DIR__, "linegauss.jl"))
include(joinpath(@__DIR__, "guard_utils.jl"))
using .LineGauss
import FLOWPanel as pnl
using StaticArrays, LinearAlgebra, Random, Printf

const σ = 1.0   # work in core-size units throughout

fails = Ref(0)
function check(name, val, tol)
    ok = val <= tol
    ok || (fails[] += 1)
    @printf("%-58s %10.3e  (tol %8.1e)  %s\n", name, val, tol, ok ? "PASS" : "FAIL")
end

# --- geometry builder: segment length L along random direction, target at
# axial position ξ·L (ξ<0 or >1 → beyond the ends) and line-distance h
function config(rng, L, ξ, h)
    t̂ = normalize(randn(rng, SVector{3,Float64}))
    tmp = normalize(cross(t̂, normalize(randn(rng, SVector{3,Float64}))))
    P1 = randn(rng, SVector{3,Float64})
    P2 = P1 + L * t̂
    x = P1 + ξ * L * t̂ + h * tmp
    return P1 - x, P2 - x   # FLOWPanel convention r1, r2
end

# Independent high-precision reference.  This evaluates the defining power
# series and convolution directly; it does not call gfun or the closed form.
function hp_g(t::BigFloat)
    t2 = t * t
    term = t * t2 / 3
    acc = term
    m = 0
    while abs(term) > eps(BigFloat) * max(abs(acc), one(BigFloat))
        m += 1
        term *= -t2 * (2m + 1) / (2m * (2m + 3))
        acc += term
        m > 500 && error("hp_g did not converge")
    end
    return sqrt(big(2) / big(pi)) * acc
end

function hp_M(z1, z2, h; n=4000)
    iseven(n) || error("Simpson panel count must be even")
    a, b, hb = BigFloat(z2), BigFloat(z1), BigFloat(h)
    f(z) = begin
        R = sqrt(z * z + hb * hb)
        R == 0 ? sqrt(big(2) / big(pi)) / 3 : hp_g(R) / R^3
    end
    step = (b - a) / n
    acc = f(a) + f(b)
    for i in 1:(n - 1)
        acc += (isodd(i) ? 4 : 2) * f(a + i * step)
    end
    return acc * step / 3
end

println("=== T0: small-argument and short-segment references ===")
setprecision(BigFloat, 256) do
    eg = maximum(abs(BigFloat(gfun(t)) - hp_g(BigFloat(t))) /
                 max(abs(hp_g(BigFloat(t))), eps(BigFloat))
                 for t in (1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 0.1))
    check("T0a g(t) vs independent 256-bit series", Float64(eg), 2e-15)

    worst_hp = 0.0
    for L in (1e-12, 1e-9, 1e-6, 1e-3, 0.05), ξ in (-1.0, 0.0, 0.5, 1.0, 2.0),
            h in (1e-12, 1e-9, 1e-6, 1e-3, 0.03)
        r1 = SVector(-h, 0.0, -ξ * L)
        r2 = SVector(-h, 0.0, (1 - ξ) * L)
        u = lg_velocity(r1, r2, σ)[2]
        uhp = Float64(BigFloat(h) * hp_M(ξ * L, (ξ - 1) * L, h) / (4big(pi)))
        worst_hp = max(worst_hp, abs(u - uhp) / max(abs(uhp), 1e-300))
    end
    check("T0b short/near-endpoint velocity vs 256-bit quadrature", worst_hp, 2e-11)

    worst_mixed = 0.0
    worst_mixed_grad = 0.0
    for z in (1e-12, 1e-10, 1e-8, 1e-6), fac in (0.3, 1.0, 3.0)
        h = fac * z
        r1 = SVector(-h, 0.0, -z)
        r2 = SVector(-h, 0.0, 2.0 - z)
        uhp = Float64(BigFloat(h) * hp_M(z, z - 2, h) / (4big(pi)))
        worst_mixed = max(worst_mixed, abs(lg_velocity(r1, r2, σ)[2] - uhp) / abs(uhp))
        δ = z / 100
        Gfd = hcat(ntuple(j -> begin
            e = SVector{3,Float64}(j == 1, j == 2, j == 3) * δ
            (lg_velocity(r1 - e, r2 - e, σ) - lg_velocity(r1 + e, r2 + e, σ)) / (2δ)
        end, 3)...)
        G = lg_gradient(r1, r2, σ)
        worst_mixed_grad = max(worst_mixed_grad, norm(G - Gfd) / norm(G))
    end
    check("T0c mixed-scale near-endpoint velocity", worst_mixed, 2e-11)
    check("T0d mixed-scale near-endpoint gradient vs FD", worst_mixed_grad, 2e-10)

    # Exact endpoint: u=0 but the transverse gradient is finite.
    for L in (1e-12, 1e-6, 0.1, 2.0), z in (0.0, L)
        r1 = SVector(0.0, 0.0, -z)
        r2 = SVector(0.0, 0.0, L - z)
        G = lg_gradient(r1, r2, σ)
        Mref = Float64(hp_M(z, z - L, 0.0))
        Gref = (Mref / (4π)) * LineGauss.skewmat(SVector(0.0, 0.0, 1.0))
        check("T0e endpoint gradient L=$(L), z=$(z)", norm(G - Gref) /
              max(norm(Gref), 1e-300), 2e-11)
        check("T0f endpoint velocity L=$(L), z=$(z)", norm(lg_velocity(r1, r2, σ)), 0.0)
    end
end

# The leading core asymptote is especially sensitive to cancellation in g.
worst_core = 0.0
for L in (1e-12, 1e-9, 1e-6), h in (1e-12, 1e-9, 1e-6)
    r1 = SVector(-h, 0.0, -L / 2)
    r2 = SVector(-h, 0.0, L / 2)
    ref = sqrt(2 / π) * h * L / (12π * σ^3)
    global worst_core = max(worst_core, abs(norm(lg_velocity(r1, r2, σ)) - ref) / ref)
end
check("T0g analytic small-core |u| asymptote", worst_core, 2e-12)

println("=== T1: local erf vs reference values ===")
refs = [0.5 0.5204998778130465; 1.0 0.8427007929497149; 1.5 0.9661051464753107;
        2.0 0.9953222650189527; 3.0 0.9999779095030014; 4.0 0.9999999845827421;
        5.0 0.9999999999984626; 0.01 0.011283415555849618]
e1 = maximum(abs(erf_local(refs[i, 1]) - refs[i, 2]) for i in 1:size(refs, 1))
check("T1 max abs erf error", e1, 1e-15)

println("\n=== T2: closed form vs quadrature ===")
rng = MersenneTwister(52)
worst = 0.0
for L in (0.5, 2.0, 10.0), ξ in (-1.5, -0.01, 0.25, 0.5, 1.2, 2.5),
        h in (0.05, 0.3, 1.0, 3.0, 8.0)
    r1, r2 = config(rng, L, ξ, h)
    u = lg_velocity(r1, r2, σ)
    uq = lg_velocity_quad(r1, r2, σ; n=8001)
    global worst = max(worst, norm(u - uq) / max(norm(uq), 1e-300))
end
check("T2 max rel err (90 configs)", worst, 1e-8)

println("\n=== T3: singular limit + convention check vs FLOWPanel ===")
worst_conv = 0.0
for L in (0.5, 2.0), ξ in (-1.0, 0.5, 2.0), h in (0.5, 3.0, 10.0)
    r1, r2 = config(rng, L, ξ, h)
    us = lg_velocity_sing(r1, r2, σ)
    up = pnl._bound_vortex_velocity(r1, r2, false, σ, Val(pnl.GaussianRegularization))
    global worst_conv = max(worst_conv, norm(us - up) / max(norm(up), 1e-300))
end
check("T3a singular assembly vs FLOWPanel singular", worst_conv, 1e-12)
# T3c: analytic singular gradient (used as the well-conditioned reference in
# T7; pnl's direct q/A forms lose precision by cancellation near the axis)
worst_g = 0.0
for L in (0.5, 2.0), ξ in (-1.0, 0.5, 2.0), h in (0.5, 3.0, 10.0)
    r1, r2 = config(rng, L, ξ, h)
    Gs = LineGauss.lg_gradient_sing(r1, r2)
    Gp = pnl._bound_vortex_gradient(r1, r2, false, σ, Val(pnl.GaussianRegularization))
    global worst_g = max(worst_g, norm(Gs - Gp) / norm(Gp))   # strict: also pins convention
end
check("T3c singular gradient vs FLOWPanel singular gradient", worst_g, 1e-9)
println("T3b deviation from singular vs exp(-d²/2σ²) (L=2, side approach):")
for d in (4.0, 6.0, 8.0, 10.0)
    r1, r2 = config(rng, 2.0, 0.5, d)
    rel = norm(lg_velocity(r1, r2, σ) - lg_velocity_sing(r1, r2, σ)) /
          norm(lg_velocity_sing(r1, r2, σ))
    @printf("  d=%5.1fσ : relerr = %.3e   exp(-d²/2) = %.3e   ratio = %.2f\n",
            d, rel, exp(-d^2 / 2), rel / exp(-d^2 / 2))
end

println("\n=== T4: infinite-line limit → shipped Gaussian profile ===")
worst4 = 0.0
for h in (0.5, 1.0, 2.0, 5.0)
    r1, r2 = config(rng, 4000.0, 0.5, h)
    W = LineGauss.lg_W(r1, r2, σ)
    global worst4 = max(worst4, abs(W - (1 - exp(-h^2 / 2))) / (1 - exp(-h^2 / 2)))
end
check("T4 max rel err W vs 1-exp(-h²/2)", worst4, 1e-6)

println("\n=== T5: analytic gradient vs central finite differences ===")
function fd_grad(u, r1, r2, δ)
    cols = ntuple(j -> begin
        e = SVector{3,Float64}(j == 1, j == 2, j == 3) * δ
        (u(r1 - e, r2 - e) - u(r1 + e, r2 + e)) / (2δ)
    end, 3)   # target shift x → x+e means r_i → r_i - e
    return hcat(cols...)
end
worst5 = 0.0
for L in (0.5, 2.0, 10.0), ξ in (-1.2, 0.3, 0.7, 1.6), h in (1e-3, 0.1, 1.0, 4.0)
    r1, r2 = config(rng, L, ξ, h)
    Ga = lg_gradient(r1, r2, σ)
    Gfd = fd_grad((a, c) -> lg_velocity(a, c, σ), r1, r2, 1e-6)
    err = norm(Ga - Gfd) / max(norm(Gfd), 1e-300)
    if err > 5e-6
        # diagnose: FD noise (shrinks with Richardson δ-sweep / quadrature FD)
        # or a genuine analytic-gradient defect (persists)?
        e5 = norm(Ga - fd_grad((a, c) -> lg_velocity(a, c, σ), r1, r2, 1e-5)) / norm(Gfd)
        eq = norm(Ga - fd_grad((a, c) -> lg_velocity_quad(a, c, σ; n=4001), r1, r2, 1e-5)) /
             norm(Gfd)
        @printf("  T5 diag: L=%5.2f ξ=%5.2f h=%7.1e  err(δ1e-6)=%.2e err(δ1e-5)=%.2e err(quadFD δ1e-5)=%.2e\n",
                L, ξ, h, err, e5, eq)
        err = min(err, e5)   # score by the better-conditioned FD
    end
    global worst5 = max(worst5, err)
end
check("T5 max rel err gradient vs FD (48 configs)", worst5, 2e-5)

println("\n=== T6: axis-guard seam (closed form vs quadrature) ===")
worst6 = 0.0
for L in (0.5, 2.0), ξ in (-0.8, 1.5, 2.5), fac in (0.3, 1.0, 3.0)
    # h chosen to straddle the fixed guard threshold ĥ² = 1e-7 (2026-08-28
    # fix: threshold no longer scales with min ẑ²)
    h = sqrt(fac * 1e-7)
    r1, r2 = config(rng, L, ξ, h)
    u = lg_velocity(r1, r2, σ)
    uq = lg_velocity_quad(r1, r2, σ; n=8001)
    global worst6 = max(worst6, norm(u - uq) / max(norm(uq), 1e-300))
end
check("T6 max rel err at guard seam", worst6, 1e-6)

worst6u_cont = 0.0
worst6g_cont = 0.0
for L in (0.5, 2.0, 10.0), ξ in (-1.5, -0.2, 1.2, 2.5)
    z1 = ξ * L
    z2 = z1 - L
    h0 = sqrt(1e-7)   # fixed axis-guard seam (2026-08-28 fix)
    make(h) = (SVector(-h, 0.0, -z1), SVector(-h, 0.0, L - z1))
    r1m, r2m = make(h0 * (1 - 1e-6))
    r1p, r2p = make(h0 * (1 + 1e-6))
    um, up = lg_velocity(r1m, r2m, σ), lg_velocity(r1p, r2p, σ)
    Gm, Gp = lg_gradient(r1m, r2m, σ), lg_gradient(r1p, r2p, σ)
    global worst6u_cont = max(worst6u_cont, norm(up - um) / max(norm(up), norm(um)))
    global worst6g_cont = max(worst6g_cont, norm(Gp - Gm) / max(norm(Gp), norm(Gm)))
end
check("T6b velocity continuity across corrected axis seam", worst6u_cont, 5e-6)
check("T6c gradient continuity across corrected axis seam", worst6g_cont, 5e-6)

# Pin the guard-selection semantics with an isolated crossing followed by a
# regression.  The safe answer is index 4, not the first crossing at index 2.
check("T6d suffix-safe nonmonotonic guard fixture",
      first_suffix_safe([2e-3, 5e-5, 2e-4, 4e-5, 3e-5], 1e-4) == 4 ? 0.0 : 1.0, 0.0)

println("\n=== T7: error decay vs SEGMENT distance; Δr(tol) ===")
# capsule sampling at distance d from the segment: side points + endpoint arcs
# (θ from the outward axis direction; θ=~0 is the along-line channel)
function capsule_targets(P1, P2, d)
    t̂ = normalize(P2 - P1)
    n̂ = normalize(cross(t̂, abs(t̂[1]) < 0.9 ? SVector(1.0, 0, 0) : SVector(0, 1.0, 0)))
    pts = SVector{3,Float64}[]
    for ξ in (0.1, 0.5, 0.9)
        push!(pts, P1 + ξ * (P2 - P1) + d * n̂)
    end
    for θdeg in (0.001, 5, 15, 30, 60, 89), (P, dir) in ((P1, -t̂), (P2, t̂))
        θ = θdeg * π / 180
        push!(pts, P + d * (cos(θ) * dir + sin(θ) * n̂))
    end
    return pts
end

function decay_scan(L, dgrid)
    P1 = SVector(0.0, 0.0, 0.0)
    P2 = SVector(0.0, 0.0, L)
    fams = (:linegauss, :gauss, :compact)
    ev = Dict(f => zeros(length(dgrid)) for f in fams)
    eg = Dict(f => zeros(length(dgrid)) for f in fams)
    for (i, d) in enumerate(dgrid)
        for x in capsule_targets(P1, P2, d)
            r1 = P1 - x
            r2 = P2 - x
            # guarded singular references for the candidate (pnl's direct singular
            # forms cancel catastrophically near the axis; validated in T3a/T3c);
            # shipped families keep pnl's own singular reference (identical
            # convention; their near-axis errors are O(1), immune to the roundoff)
            us = lg_velocity_sing(r1, r2, σ)
            Gs = LineGauss.lg_gradient_sing(r1, r2)
            us_p = pnl._bound_vortex_velocity(r1, r2, false, σ, Val(pnl.GaussianRegularization))
            Gs_p = pnl._bound_vortex_gradient(r1, r2, false, σ, Val(pnl.GaussianRegularization))
            for f in fams
                if f === :linegauss
                    u = lg_velocity(r1, r2, σ)
                    G = lg_gradient(r1, r2, σ)
                    ev[f][i] = max(ev[f][i], norm(u - us) / max(norm(us), 1e-300))
                    eg[f][i] = max(eg[f][i], norm(G - Gs) / max(norm(Gs), 1e-300))
                else
                    fam = f === :gauss ? Val(pnl.GaussianRegularization) :
                                         Val(pnl.CompactRegularization)
                    u = pnl._bound_vortex_velocity(r1, r2, true, σ, fam)
                    G = pnl._bound_vortex_gradient(r1, r2, true, σ, fam)
                    ev[f][i] = max(ev[f][i], norm(u - us_p) / max(norm(us_p), 1e-300))
                    eg[f][i] = max(eg[f][i], norm(G - Gs_p) / max(norm(Gs_p), 1e-300))
                end
            end
        end
    end
    return ev, eg
end

for L in (0.5, 1.79, 3.0)
    dgrid = collect(2.0:0.25:11.0)
    ev, eg = decay_scan(L, dgrid)
    @printf("L = %.2fσ:\n  %5s | %23s | %23s | %23s\n", L, "d/σ",
            "linegauss (vel, grad)", "gauss (vel, grad)", "compact (vel, grad)")
    for (i, d) in enumerate(dgrid)
        d == round(d) || continue
        @printf("  %5.2f | %10.2e %10.2e | %10.2e %10.2e | %10.2e %10.2e\n", d,
                ev[:linegauss][i], eg[:linegauss][i], ev[:gauss][i], eg[:gauss][i],
                ev[:compact][i], eg[:compact][i])
    end
    # Δr(tol): smallest d with suffix-max error ≤ tol (velocity and gradient)
    for tol in (1e-4, 1e-5, 1e-6)
        sfx(v) = [maximum(v[i:end]) for i in eachindex(v)]
        for f in (:linegauss, :gauss, :compact)
            iv = findfirst(<=(tol), sfx(max.(ev[f], eg[f])))
            @printf("  Δr(%g, %s) = %s\n", tol, f,
                    iv === nothing ? "> $(dgrid[end])σ (channel open)" : "$(dgrid[iv])σ")
        end
    end
end

println("\n=== T7b: dense finite-segment radius check at tol=1e-6 ===")
function dense_capsule_error(L, d; nangles=181, nz=101)
    P1 = SVector(0.0, 0.0, 0.0)
    P2 = SVector(0.0, 0.0, L)
    maxv = 0.0
    maxg = 0.0
    # Cylinder plus both hemispherical caps; axial symmetry removes azimuth.
    targets = [SVector(d, 0.0, z) for z in range(0.0, L; length=nz)]
    for θ in range(1e-6, π / 2; length=nangles), (P, sgn) in ((P1, -1.0), (P2, 1.0))
        push!(targets, P + SVector(d * sin(θ), 0.0, sgn * d * cos(θ)))
    end
    for x in targets
        r1, r2 = P1 - x, P2 - x
        us = lg_velocity_sing(r1, r2, σ)
        Gs = LineGauss.lg_gradient_sing(r1, r2)
        maxv = max(maxv, norm(lg_velocity(r1, r2, σ) - us) / max(norm(us), 1e-300))
        maxg = max(maxg, norm(lg_gradient(r1, r2, σ) - Gs) / max(norm(Gs), 1e-300))
    end
    return maxv, maxg
end

length_grid = vcat(10.0 .^ range(-3, 1; length=17), [0.5, 1.79, 3.0, 16.0, 64.0])
worst_rv = maximum(first(dense_capsule_error(L, 5.90)) for L in length_grid)
worst_rg = maximum(last(dense_capsule_error(L, 6.25)) for L in length_grid)
check("T7b velocity radius 5.90σ (dense L/capsule scan)", worst_rv, 1e-6)
check("T7b gradient radius 6.25σ (dense L/capsule scan)", worst_rg, 1e-6)

println("\n=== T8: peaks at matched core size (phase_00 extension) ===")
for L in (1.79, 1000.0)
    P1 = SVector(0.0, 0.0, 0.0)
    P2 = SVector(0.0, 0.0, L)
    umax = 0.0
    gmax = 0.0
    zlo, zhi = L > 100 ? (L / 2 - 3, L / 2 + 3) : (-3.0, L + 3)
    for h in 0.02:0.02:6.0, z in zlo:0.05:zhi
        x = SVector(h, 0.0, z)
        umax = max(umax, norm(lg_velocity(P1 - x, P2 - x, σ)))
        gmax = max(gmax, opnorm(Matrix(lg_gradient(P1 - x, P2 - x, σ))))
    end
    @printf("L = %-7.4gσ : u_max·rc = %.4f  ‖∇u‖_max·rc² = %.4f  [Γ/2π units]\n",
            L, umax * 2π, gmax * 2π)
end
println("(reference: shipped Gaussian infinite-line 0.4514 / 0.50)")

println(fails[] == 0 ? "\nALL CHECKS PASSED" : "\n$(fails[]) CHECK(S) FAILED")
