# k03 — validation of the FLOWPanel LineGaussRegularization port (2026-08-28).
# P1 enum/setter plumbing; P2 radius_inflation rule values vs measured radii;
# P3 velocity/gradient parity FLOWPanel port vs prototype across all guard
# regimes; P4 dense segment-distance verification of the inflation rule
# (T7b protocol, tol grid 1e-4..1e-7) using the PORTED kernel; P5 device-path
# guard (FastMultipole rejects the unported family loudly).
#
# Run: JULIA_NUM_THREADS=4 julia --project=../FLOWPanel.jl k03_flowpanel_port.jl

include(joinpath(@__DIR__, "linegauss.jl"))
using .LineGauss
import FLOWPanel as pnl
import FastMultipole
using StaticArrays, LinearAlgebra, Random, Printf

const σ = 1.0

fails = Ref(0)
function check(name, val, tol)
    ok = val <= tol
    ok || (fails[] += 1)
    @printf("%-58s %10.3e  (tol %8.1e)  %s\n", name, val, tol, ok ? "PASS" : "FAIL")
end

println("=== P1: enum + setter plumbing ===")
saved = pnl.FILAMENT_REGULARIZATION[]
pnl.set_filament_regularization!(:linegauss)
check("P1 setter selects LineGaussRegularization",
      pnl.FILAMENT_REGULARIZATION[] == pnl.LineGaussRegularization ? 0.0 : 1.0, 0.0)
check("P1 enum code is 4th member (Int = 3, gpu code 4)",
      Int(pnl.LineGaussRegularization) == 3 ? 0.0 : 1.0, 0.0)
bogus_ok = try pnl.set_filament_regularization!(:bogus); 1.0 catch; 0.0 end
check("P1 :bogus still throws", bogus_ok, 0.0)
pnl.FILAMENT_REGULARIZATION[] = saved

println("\n=== P2: radius_inflation rule ===")
# expected: Gaussian fixed point + 0.35σ; must dominate the measured
# segment-distance matching radii 5.25/5.75/6.25σ (k01 T7). Those are
# quantized UP to T7's 0.25σ grid, so allow one grid step here — the dense
# continuous-radius verification is P4 below, which is the real gate.
measured = Dict(1e-4 => 5.25, 1e-5 => 5.75, 1e-6 => 6.25)
pnl.set_filament_regularization!(:linegauss)
for tol in (1e-4, 1e-5, 1e-6)
    dr = pnl.radius_inflation(pnl.VortexRing, σ, tol)
    @printf("  tol %g: Δr = %.4fσ (measured %.2fσ, 0.25σ grid)\n", tol, dr, measured[tol])
    check("P2 rule ≥ measured radius − grid step @tol $tol", measured[tol] - 0.25 - dr, 0.0)
end
check("P2 tol=Inf disables", pnl.radius_inflation(pnl.VortexRing, σ, Inf), 0.0)
pnl.FILAMENT_REGULARIZATION[] = saved

println("\n=== P3: FLOWPanel port vs prototype parity ===")
# regimes: main branch, axis guard, small-radius, endpoint-split, far field
rng = MersenneTwister(3)
function config(rng, L, ξ, h)
    t̂ = normalize(randn(rng, SVector{3,Float64}))
    tmp = normalize(cross(t̂, normalize(randn(rng, SVector{3,Float64}))))
    P1 = randn(rng, SVector{3,Float64})
    P2 = P1 + L * t̂
    x = P1 + ξ * L * t̂ + h * tmp
    return P1 - x, P2 - x
end
worst_v = 0.0
worst_g = 0.0
worst_vax = 0.0
worst_gax = 0.0
nonfinite = 0
fam = Val(pnl.LineGaussRegularization)
for L in (2e-3, 0.05, 0.124, 0.5, 1.79, 3.0, 16.0, 1000.0),
    ξ in (-0.5, -0.01, 0.1, 0.5, 0.99, 1.5),
    h in (0.0, 1e-10, 1e-6, 1e-3, 0.05, 0.124, 0.3, 1.0, 3.0, 5.9, 8.0, 12.0)
    r1, r2 = config(rng, L, ξ, h)
    (norm(r1) < 5eps() || norm(r2) < 5eps()) && continue   # endpoint contract differs by design
    u_p = lg_velocity(r1, r2, σ)
    G_p = lg_gradient(r1, r2, σ)
    u_f = pnl._bound_vortex_velocity(r1, r2, true, σ, fam)
    G_f = pnl._bound_vortex_gradient(r1, r2, true, σ, fam)
    (all(isfinite, u_f) && all(isfinite, G_f)) || (global nonfinite += 1; continue)
    dv = norm(u_f - u_p) / max(norm(u_p), 1e-300)
    dg = norm(G_f - G_p) / max(norm(G_p), 1e-300)
    if h >= 1e-3
        if dv > worst_v
            # conditioning of the numerator difference q̃ = ẑ1/R̂1 − ẑ2/R̂2
            # (shared with the shipped singular kernel): erf-backend ulps
            # amplify by κ_cond on distant axial-ish configs
            s = r1 - r2; Lg = sqrt(dot(s, s))
            ẑ1 = dot(s, r1) / Lg; ẑ2 = ẑ1 - Lg
            q̃ = ẑ1 / norm(r1) - ẑ2 / norm(r2)
            κc = (abs(ẑ1) / norm(r1) + abs(ẑ2) / norm(r2)) / max(abs(q̃), 1e-300)
            @printf("  new worst vel parity %.3e at L=%g ξ=%g h=%g (κ_cond=%.2e, κ·1e-16=%.2e)\n",
                    dv, L, ξ, h, κc, κc * 1e-16)
        end
        global worst_v = max(worst_v, dv)
        global worst_g = max(worst_g, dg)
    else
        # near/on-axis band: the transverse frame direction is conditioned as
        # eps·R²/(hL), so bit-level input differences rotate the (magnitude-
        # bounded) gradient; only loose parity is meaningful here
        global worst_vax = max(worst_vax, norm(u_f - u_p))
        global worst_gax = max(worst_gax, min(dg, norm(G_f - G_p)))
    end
end
check("P3 non-finite port outputs (count)", Float64(nonfinite), 0.0)
# The erf backends differ (SpecialFunctions vs the prototype's series/clamp
# erf), and their ~1e-16 disagreements are amplified by the intrinsic
# conditioning κ_cond of the q̃-cancellation on distant axial-ish configs
# (identical conditioning to the shipped singular kernel) — observed up to
# ~5e-8. 1e-6 still catches any real transcription slip; the CORRECTNESS
# gates are P4 (dense radius rule vs singular refs) and k01 (256-bit refs).
check("P3 velocity parity (h ≥ 1e-3σ)", worst_v, 1e-6)
check("P3 gradient parity (h ≥ 1e-3σ)", worst_g, 1e-6)
check("P3 velocity abs parity (axis band)", worst_vax, 1e-12)
check("P3 gradient parity (axis band, rel-or-abs)", worst_gax, 1e-4)

# endpoint contract: FLOWPanel zeroes velocity AND gradient at a vertex
let L = 1.79
    r1 = SVector(0.0, 0.0, 0.0)
    r2 = SVector(0.0, 0.0, -L)
    check("P3 endpoint velocity zero (FLOWPanel contract)",
          norm(pnl._bound_vortex_velocity(r1, r2, true, σ, fam)), 0.0)
    check("P3 endpoint gradient zero (FLOWPanel contract)",
          norm(pnl._bound_vortex_gradient(r1, r2, true, σ, fam)), 0.0)
end

println("\n=== P4: dense radius-rule verification (ported kernel) ===")
function dense_capsule_error(L, d; nangles=181, nz=101)
    P1 = SVector(0.0, 0.0, 0.0)
    P2 = SVector(0.0, 0.0, L)
    maxv = 0.0
    maxg = 0.0
    targets = [SVector(d, 0.0, z) for z in range(0.0, L; length=nz)]
    for θ in range(1e-6, π / 2; length=nangles), (P, sgn) in ((P1, -1.0), (P2, 1.0))
        push!(targets, P + SVector(d * sin(θ), 0.0, sgn * d * cos(θ)))
    end
    for x in targets
        r1, r2 = P1 - x, P2 - x
        us = lg_velocity_sing(r1, r2, σ)
        Gs = LineGauss.lg_gradient_sing(r1, r2)
        u = pnl._bound_vortex_velocity(r1, r2, true, σ, fam)
        G = pnl._bound_vortex_gradient(r1, r2, true, σ, fam)
        maxv = max(maxv, norm(u - us) / max(norm(us), 1e-300))
        maxg = max(maxg, norm(G - Gs) / max(norm(Gs), 1e-300))
    end
    return maxv, maxg
end
length_grid = vcat(10.0 .^ range(-3, 1; length=17), [0.5, 1.79, 3.0, 16.0, 64.0])
pnl.set_filament_regularization!(:linegauss)
for tol in (1e-4, 1e-5, 1e-6, 1e-7)
    dr = pnl.radius_inflation(pnl.VortexRing, σ, tol)
    wv = maximum(first(dense_capsule_error(L, dr)) for L in length_grid)
    wg = maximum(last(dense_capsule_error(L, dr)) for L in length_grid)
    @printf("  tol %g: Δr = %.3fσ  max vel %.3e  max grad %.3e\n", tol, dr, wv, wg)
    check("P4 velocity ≤ tol at rule radius @tol $tol", wv, tol)
    check("P4 gradient ≤ tol at rule radius @tol $tol", wg, tol)
end
pnl.FILAMENT_REGULARIZATION[] = saved

println("\n=== P5: device-path arm (Step 3, 2026-08-28: LineGauss ported) ===")
# pre-Step-3 this section asserted the THROW guard; the rectangular kernel
# now carries the Val{4} LineGauss arm (k04_rect_linegauss.jl is its parity
# harness), so assert the plumbing routes code 4 and still rejects unknowns
p5a = try
    FastMultipole.RectangularPanelInfluence(:linegauss).filament_reg == Int32(4) ? 0.0 : 1.0
catch; 1.0 end
check("P5 RectangularPanelInfluence(:linegauss) -> code 4", p5a, 0.0)
p5b = try FastMultipole._rect_reg_val(Int32(4)) === Val(4) ? 0.0 : 1.0 catch; 1.0 end
check("P5 _rect_reg_val(4) -> Val(4)", p5b, 0.0)
p5c = try FastMultipole._rect_reg_val(Int32(5)); 1.0 catch; 0.0 end
check("P5 _rect_reg_val(5) throws (no silent Vatistas)", p5c, 0.0)

println(fails[] == 0 ? "\nALL PASS" : "\n$(fails[]) FAILURES")
exit(fails[] == 0 ? 0 : 1)
