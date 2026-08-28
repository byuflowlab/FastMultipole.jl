# k04 — Step-3 validation (2026-08-28): LineGauss Val{4} arms in FastMultipole
# src/direct_rectangular.jl (host fallback + CUDA kernel shared pair math) vs
# the certified FLOWPanel host port (commit 8b07f96). CPU only.
#
# Q1: segment-level parity _rect_bound_vortex_velocity/_gradient(Val(4)) vs
#     FLOWPanel _bound_vortex_velocity/_gradient(Val(LineGauss)) over the k03
#     P3 config grid (all guard regimes). Both sides share branch structure;
#     residual = erf backend only (fdlibm _rect_erf vs SpecialFunctions.erf),
#     ulps amplified by q̃-conditioning — same tolerances as k03 P3.
# Q2: segment-level parity vs the 256-bit-validated prototype (linegauss.jl).
# Q3: endpoint contract (zero U and G at a vertex).
# Q4: functor-level direct_rectangular! (tag-3 nv=2 filaments + tag-3 tri
#     rings, gradient armed) vs the FLOWPanel per-segment sums.
# Q5: plumbing — :linegauss -> code 4 -> Val(4); code 5 throws.
#
# Run from FastMultipole repo root:
#   JULIA_DEPOT_PATH=/private/tmp/flowpanel-052b-depot:/Users/ryan/.julia \
#   JULIA_NUM_THREADS=4 julia --project=../FLOWPanel.jl \
#     MATRIX_OPERATOR_REFACTOR/prototypes/052d_compact_kernel/k04_rect_linegauss.jl

include(joinpath(@__DIR__, "linegauss.jl"))
using .LineGauss
import FLOWPanel as pnl
import FastMultipole
const FMR = FastMultipole
using StaticArrays, LinearAlgebra, Random, Printf

const σ = 1.0

fails = Ref(0)
function check(name, val, tol)
    ok = val <= tol
    ok || (fails[] += 1)
    @printf("%-58s %10.3e  (tol %8.1e)  %s\n", name, val, tol, ok ? "PASS" : "FAIL")
end

fam = Val(pnl.LineGaussRegularization)
v4 = Val(4)

println("=== Q1/Q2: segment-level parity (k03 P3 grid) ===")
rng = MersenneTwister(3)
function config(rng, L, ξ, h)
    t̂ = normalize(randn(rng, SVector{3,Float64}))
    tmp = normalize(cross(t̂, normalize(randn(rng, SVector{3,Float64}))))
    P1 = randn(rng, SVector{3,Float64})
    P2 = P1 + L * t̂
    x = P1 + ξ * L * t̂ + h * tmp
    return P1 - x, P2 - x
end
worst_v = 0.0; worst_g = 0.0          # vs FLOWPanel port, h ≥ 1e-3
worst_vp = 0.0; worst_gp = 0.0        # vs prototype, h ≥ 1e-3
worst_vax = 0.0; worst_gax = 0.0      # axis band vs FLOWPanel
nonfinite = 0
for L in (2e-3, 0.05, 0.124, 0.5, 1.79, 3.0, 16.0, 1000.0),
    ξ in (-0.5, -0.01, 0.1, 0.5, 0.99, 1.5),
    h in (0.0, 1e-10, 1e-6, 1e-3, 0.05, 0.124, 0.3, 1.0, 3.0, 5.9, 8.0, 12.0)
    r1, r2 = config(rng, L, ξ, h)
    (norm(r1) < 5eps() || norm(r2) < 5eps()) && continue
    u_r = FMR._rect_bound_vortex_velocity(r1, r2, σ, v4)
    G_r = FMR._rect_bound_vortex_gradient(r1, r2, σ, v4)
    (all(isfinite, u_r) && all(isfinite, G_r)) || (global nonfinite += 1; continue)
    u_f = pnl._bound_vortex_velocity(r1, r2, true, σ, fam)
    G_f = pnl._bound_vortex_gradient(r1, r2, true, σ, fam)
    u_p = lg_velocity(r1, r2, σ)
    G_p = lg_gradient(r1, r2, σ)
    if h >= 1e-3
        global worst_v = max(worst_v, norm(u_r - u_f) / max(norm(u_f), 1e-300))
        global worst_g = max(worst_g, norm(G_r - G_f) / max(norm(G_f), 1e-300))
        global worst_vp = max(worst_vp, norm(u_r - u_p) / max(norm(u_p), 1e-300))
        global worst_gp = max(worst_gp, norm(G_r - G_p) / max(norm(G_p), 1e-300))
    else
        # near/on-axis: transverse frame conditioned as eps·R²/(hL) — loose
        # parity only (k03 P3 rationale)
        global worst_vax = max(worst_vax, norm(u_r - u_f))
        global worst_gax = max(worst_gax,
            min(norm(G_r - G_f) / max(norm(G_f), 1e-300), norm(G_r - G_f)))
    end
end
check("Q1 non-finite rect outputs (count)", Float64(nonfinite), 0.0)
check("Q1 velocity parity vs FLOWPanel (h ≥ 1e-3σ)", worst_v, 1e-6)
check("Q1 gradient parity vs FLOWPanel (h ≥ 1e-3σ)", worst_g, 1e-6)
check("Q1 velocity abs parity (axis band)", worst_vax, 1e-12)
check("Q1 gradient parity (axis band, rel-or-abs)", worst_gax, 1e-4)
check("Q2 velocity parity vs prototype (h ≥ 1e-3σ)", worst_vp, 1e-6)
check("Q2 gradient parity vs prototype (h ≥ 1e-3σ)", worst_gp, 1e-6)

println("\n=== Q3: endpoint contract ===")
let L = 1.79
    r1 = SVector(0.0, 0.0, 0.0)
    r2 = SVector(0.0, 0.0, -L)
    check("Q3 endpoint velocity zero", norm(FMR._rect_bound_vortex_velocity(r1, r2, σ, v4)), 0.0)
    check("Q3 endpoint gradient zero", norm(FMR._rect_bound_vortex_gradient(r1, r2, σ, v4)), 0.0)
end

println("\n=== Q4: functor-level direct_rectangular! vs FLOWPanel sums ===")
rng4 = MersenneTwister(7)
cs = 2e-2
nfil = 12
srcp = zeros(17, nfil + 6)
fverts = Vector{NTuple{2,SVector{3,Float64}}}(undef, nfil)
fgam = randn(rng4, nfil)
function pack!(srcp, q, tag, verts, s1, cs)
    srcp[1, q] = tag
    srcp[2, q] = length(verts)
    for (iv, v) in enumerate(verts), d in 1:3
        srcp[2 + 3*(iv-1) + d, q] = v[d]
    end
    srcp[15, q] = s1
    srcp[17, q] = cs
end
for q in 1:nfil
    p1 = SVector{3}(randn(rng4, 3))
    p2 = p1 + SVector{3}(0.3 .* randn(rng4, 3))
    fverts[q] = (p1, p2)
    pack!(srcp, q, 3, (p1, p2), fgam[q], cs)
end
tverts = Vector{NTuple{3,SVector{3,Float64}}}(undef, 6)
tgam = randn(rng4, 6)
for q in 1:6
    a = SVector{3}(randn(rng4, 3))
    b = a + SVector{3}(0.4 .* randn(rng4, 3))
    c = a + SVector{3}(0.4 .* randn(rng4, 3))
    tverts[q] = (a, b, c)
    pack!(srcp, nfil + q, 3, (a, b, c), tgam[q], cs)
end
n_tgt = 40
tgt = 1.5 .* randn(rng4, 3, n_tgt)
out = zeros(12, n_tgt)
FMR.direct_rectangular!(out, tgt, FMR.RectangularPanelInfluence(:linegauss),
    srcp; gradient=true)
ref = zeros(12, n_tgt)
for i in 1:n_tgt
    x = SVector{3}(tgt[:, i])
    segsum = Tuple{SVector{3,Float64},SVector{3,Float64},Float64}[]
    for q in 1:nfil
        push!(segsum, (fverts[q][1], fverts[q][2], fgam[q]))
    end
    for q in 1:6
        a, b, c = tverts[q]
        push!(segsum, (a, b, tgam[q]))
        push!(segsum, (b, c, tgam[q]))
        push!(segsum, (c, a, tgam[q]))
    end
    for (p1, p2, gam) in segsum
        U = pnl._bound_vortex_velocity(p1 - x, p2 - x, true, cs, fam) * gam
        G = pnl._bound_vortex_gradient(p1 - x, p2 - x, true, cs, fam) * gam
        ref[1:3, i] .+= U
        for j in 1:3, k in 1:3
            ref[3 + (j-1)*3 + k, i] += G[k, j]
        end
    end
end
relerr(a, b) = norm(a .- b) / max(norm(b), 1e-300)
check("Q4 functor velocity vs FLOWPanel sums", relerr(out[1:3, :], ref[1:3, :]), 1e-7)
check("Q4 functor gradient vs FLOWPanel sums", relerr(out[4:12, :], ref[4:12, :]), 1e-7)

println("\n=== Q5: plumbing ===")
q5a = try FMR.RectangularPanelInfluence(:linegauss).filament_reg == Int32(4) ? 0.0 : 1.0 catch; 1.0 end
check("Q5 :linegauss -> code 4", q5a, 0.0)
q5b = try FMR._rect_reg_val(Int32(4)) === Val(4) ? 0.0 : 1.0 catch; 1.0 end
check("Q5 _rect_reg_val(4) -> Val(4)", q5b, 0.0)
q5c = try FMR._rect_reg_val(Int32(5)); 1.0 catch; 0.0 end
check("Q5 _rect_reg_val(5) throws", q5c, 0.0)

println(fails[] == 0 ? "\nALL PASS" : "\n$(fails[]) FAILURES")
exit(fails[] == 0 ? 0 : 1)
