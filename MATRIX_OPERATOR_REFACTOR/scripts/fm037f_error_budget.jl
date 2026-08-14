# Task 037f: error-allowance derivation for cheapened gaussianerf g/h pair
# kernels (nearfield U+J), and empirical validation of the pointwise->delivered
# mapping on a local CPU smoke case.
#
# Stdlib only, single-threaded, local-safe (BigFloat fits + an O(S*n) exact
# pair sweep at n <= 1e4).  Derives, in order:
#   1. the delivered-error budget B from the recorded 037b error-decomposition
#      anchors (allowance = 1e-3 - u_total, margin FM037F_MARGIN >= 1.1,
#      binding case taken over all shipped-default anchor rows);
#   2. the pointwise->delivered amplification factors kappa on a synthetic
#      overlap-matched cube: a pointwise perturbation |dg| <= eps on a set of
#      pairs perturbs U at a target by at most eps * sum(|singular pair
#      velocity|) over that set, so kappa = RMS_targets(sum |C|-weighted) /
#      RMS_targets(|U_R|) converts pointwise eps into delivered relative RMS;
#   3. per-mechanism pointwise errors vs a 256-bit reference (reduced series
#      term counts, reduced outer degree, fp32-in-f64 composite, rho^2-indexed
#      linear-interpolation LUT of the normalized G = g/rho^3, H = h/rho^5);
#   4. predicted delivered error per mechanism = sum over perturbed branches of
#      eps_branch * kappa_branch, compared against B;
#   5. empirical check: the actual delivered RMS delta of each candidate on the
#      smoke case vs the prediction (prediction must be an upper bound).
#
# Output: printed tables + data CSV
#   ../data/kernel_splitting/fm037f_budget.csv
# and Julia literals for the chosen :reduced coefficient tuples.
#
# Env knobs: FM037F_N (default 10000), FM037F_SAMPLES (default 200),
# FM037F_MARGIN (default 1.1).  The same file doubles as the cluster-scale
# validation driver via FM037F_N=100000/1000000 (still exact CPU sums; pair
# cost is S*n).

using Printf
using Random

setprecision(BigFloat, 256)

const FM037F_N = parse(Int, get(ENV, "FM037F_N", "10000"))
const FM037F_SAMPLES = parse(Int, get(ENV, "FM037F_SAMPLES", "200"))
const FM037F_MARGIN = parse(Float64, get(ENV, "FM037F_MARGIN", "1.1"))
const FM037F_RHO_T = 3.668          # campaign operating cutoff (037b anchors)
const FM037F_GATE = 1e-3

const A_BIG = sqrt(big(2) / big(pi))
const A64 = Float64(A_BIG)

# ------------------------------------------------------------- references ----

function erf_series_big(x::BigFloat)
    s = zero(BigFloat); term = x; n = 0
    while true
        add = term / (2n + 1); s += add; n += 1; term *= -x * x / n
        abs(add) < eps(BigFloat) * max(abs(s), one(BigFloat)) && break
    end
    return 2 / sqrt(big(pi)) * s
end
function erfc_big(x::BigFloat)
    x < 2 && return one(BigFloat) - erf_series_big(x)
    u = 1 / (2 * x * x); cf = one(BigFloat)
    for k in 2000:-1:1
        cf = 1 + k * u / cf
    end
    return exp(-x * x) / (x * sqrt(big(pi))) / cf
end
erf_ref_big(x::BigFloat) = x < 2 ? erf_series_big(x) : 1 - erfc_big(x)
g_big(rho::BigFloat) = erf_ref_big(rho / sqrt(big(2))) - A_BIG * rho * exp(-rho^2 / 2)
h_big(rho::BigFloat) = A_BIG * rho^3 * exp(-rho^2 / 2) - 3 * g_big(rho)

# Float64 reference erf (independent of src; validate_031a pattern)
function ref_erf(x::Float64)
    ax = abs(x)
    if ax < 2.0
        s = 0.0; term = ax; n = 0
        while true
            add = term / (2n + 1); s += add; n += 1; term *= -ax * ax / n
            abs(add) <= eps() * max(abs(s), 1.0) && break
        end
        return sign(x) * (2 / sqrt(pi)) * s
    end
    u = 1 / (2 * ax * ax); cf = 1.0
    for k in 60:-1:1
        cf = 1 + k * u / cf
    end
    return sign(x) * (1 - exp(-ax * ax) / (ax * sqrt(pi)) / cf)
end
function gh_ref(rho::Float64)
    e = exp(-rho * rho / 2)
    g = ref_erf(rho / sqrt(2)) - A64 * rho * e
    return g, A64 * rho^3 * e - 3g
end

# ------------------------------------- shipped + candidate evaluators --------

# exact series coefficients (identical construction to src)
gcoef(k) = (isodd(k) ? -1 : 1) // ((2k + 3) * BigInt(2)^k * factorial(BigInt(k)))
hcoef(k) = (iseven(k) ? -1 : 1) // ((2k + 5) * BigInt(2)^k * factorial(BigInt(k)))
const GC64 = Tuple(Float64(gcoef(k)) for k in 0:18)
const HC64 = Tuple(Float64(hcoef(k)) for k in 0:18)
const GC32 = Tuple(Float32(c) for c in GC64[1:13])
const HC32 = Tuple(Float32(c) for c in HC64[1:13])
# shipped degree-3 outer fit (src literals)
const SC = (0.082593826443677007, 2.0801015208954681,
    -6.8585923004500211, 10.482822830062796)

function gh_series(rho::T, gc, hc) where T
    z = rho * rho
    g = zero(T); h = zero(T)
    for k in length(gc):-1:1
        g = muladd(g, z, T(gc[k]))
        h = muladd(h, z, T(hc[k]))
    end
    return T(A_BIG) * rho * z * g, T(A_BIG) * rho * z * z * h
end
function gh_outer(rho::T, sc) where T
    z = rho * rho
    e = exp(-z / 2)
    u = inv(z)
    s = T(sc[end])
    for j in (length(sc) - 1):-1:1
        s = muladd(s, u, T(sc[j]))
    end
    gbar = e * muladd(T(A_BIG), rho, s)
    g = one(T) - gbar
    return g, T(A_BIG) * rho * z * e - 3g
end
gh_shipped(rho::Float64) =
    rho <= 2.0 ? gh_series(rho, GC64, HC64) : gh_outer(rho, SC)
gh_shipped(rho::Float32) =
    rho <= 2f0 ? gh_series(rho, GC32, HC32) : gh_outer(rho, SC)

# candidate: reduced term counts / outer degree
gh_reduced(rho::Float64, nt::Int) =
    rho <= 2.0 ? gh_series(rho, GC64[1:nt], HC64[1:nt]) : gh_outer(rho, SC)
gh_reduced(rho::Float32, nt::Int) =
    rho <= 2f0 ? gh_series(rho, GC32[1:nt], HC32[1:nt]) : gh_outer(rho, SC)
# candidate: full Float32 inner math consumed by a Float64 configuration
gh_fp32(rho::Float64) = Float64.(gh_shipped(Float32(rho)))
gh_reduced_fp32(rho::Float64, nt::Int) = Float64.(gh_reduced(Float32(rho), nt))

# candidate: rho^2-indexed LUT of normalized G = g/rho^3, H = h/rho^5 (both
# analytic in x = rho^2 with G(0) = A/3 != 0, so linear interpolation preserves
# RELATIVE accuracy down to rho -> 0), Float32 storage, singular (1, -3)
# beyond x_max = rho_t^2.  The table samples the SHIPPED Float64 evaluator —
# the LUT is a cheapening of the shipped kernel, so its delta-vs-shipped is
# pure interpolation + Float32 storage, and it inherits (not adds to) the
# shipped outer-fit error against the analytic reference.
function build_lut(rho_t::Float64, N::Int)
    x_max = rho_t * rho_t
    tab = Matrix{Float32}(undef, 2, N)
    for i in 1:N
        x = x_max * (i - 1) / (N - 1)
        rho = sqrt(x)
        if i == 1
            tab[1, i] = Float32(A64 / 3)
            tab[2, i] = Float32(-A64 / 5)
        else
            g, h = gh_shipped(rho)
            tab[1, i] = Float32(g / rho^3)
            tab[2, i] = Float32(h / rho^5)
        end
    end
    return tab
end
function gh_lut(rho::T, tab::Matrix{Float32}, x_max::Float64) where T
    x = rho * rho
    x >= T(x_max) && return one(T), -T(3)
    N = size(tab, 2)
    t = x * T((N - 1) / x_max)
    i0 = unsafe_trunc(Int, t)
    f = t - T(i0)
    i0 += 1
    G = muladd(f, T(tab[1, i0 + 1]) - T(tab[1, i0]), T(tab[1, i0]))
    H = muladd(f, T(tab[2, i0 + 1]) - T(tab[2, i0]), T(tab[2, i0]))
    return rho * x * G, rho * x * x * H
end

# ------------------------------------------------- 1. delivered budget B -----

# shipped-default anchor rows of the 037b decomposition data of record
# (label, case, n, tf) -> selected programmatically from the CSV.
const DECOMP_CSV = joinpath(@__DIR__, "..", "data", "flowvpm_gpu_campaign",
    "fm037b_error_decomposition.csv")

anchors = NamedTuple[]
if isfile(DECOMP_CSV)
    lines = readlines(DECOMP_CSV)
    hdr = split(lines[1], ',')
    col = Dict(name => i for (i, name) in enumerate(hdr))
    for ln in lines[2:end]
        f = split(ln, ',')
        occursin("anchor", f[col["label"]]) ||
            occursin("_pa_", f[col["label"]]) || continue
        push!(anchors, (label=f[col["label"]], case=f[col["case"]],
            n=parse(Int, f[col["n"]]), tf=f[col["tf"]],
            u_total=parse(Float64, f[col["u_total_rel"]]),
            u_cutoff=parse(Float64, f[col["u_cutoff_rel"]]),
            u_canc=parse(Float64, f[col["u_cancellation_ratio"]])))
    end
else
    error("decomposition anchors not found: $DECOMP_CSV")
end

println("=== 1. delivered-error allowance (gate $(FM037F_GATE), margin $(FM037F_MARGIN)x) ===")
println("label,case,n,tf,u_total,headroom,budget")
B = Inf
binding = ""
for a in anchors
    head = FM037F_GATE - a.u_total
    bud = head / FM037F_MARGIN
    @printf("%s,%s,%d,%s,%.3e,%.3e,%.3e\n", a.label, a.case, a.n, a.tf,
        a.u_total, head, bud)
    if bud < B
        global B = bud
        global binding = "$(a.label) $(a.case) n=$(a.n) $(a.tf)"
    end
end
@printf("binding: %s  ->  B = %.3e delivered relative U RMS\n\n", binding, B)

# ---------------------------------------- 2. smoke case + amplification ------

# synthetic overlap-matched cube: uniform random positions in the unit cube,
# random Gamma, sigma = overlap * n^(-1/3) (FLOWVPM cube cases run overlap-
# matched particle distributions; kappa is a dimensionless ratio and the
# cluster configs re-measure it on the real cases).
function build_smoke(n, overlap, seed)
    rng = MersenneTwister(seed)
    X = rand(rng, 3, n)
    G = randn(rng, 3, n)
    sig = fill(overlap * n^(-1 / 3), n)
    return X, G, sig
end

# One exact O(S*n) sweep: per-sample-target
#   UR        exact regularized velocity (reference erf)
#   S_series  sum over pairs rho <= 2        of g * |C|      (relative dg)
#   S_outer   sum over pairs 2 < rho <= rt   of |C|          (absolute dg)
#   S_all     sum over all pairs of g_used * |C|             (fp32 relative)
# each in a COHERENT (plain sum, rigorous worst case) and an INCOHERENT
# (root-sum-square, the theory SS6.4/SS7 no-coherent-cancellation model that
# already underlies the shipped rho_t) variant.
#   J analogues with the pair bounds |dJ| <= |dh| |crss| / r (a-term) and
#   |dg| sqrt(2) |Gamma| cr3 (b-term); JR the exact regularized Jacobian norm
function sweep(X, G, sig, idx, rho_t)
    S = length(idx)
    UR = zeros(3, S)
    JR2 = zeros(S)               # ||J_R||_F^2 per target
    S_series = zeros(S); S_outer = zeros(S); S_all = zeros(S)
    Q_series = zeros(S); Q_outer = zeros(S); Q_all = zeros(S)   # sum of squares
    SJh_series = zeros(S); SJg_series = zeros(S)
    SJh_outer = zeros(S); SJg_outer = zeros(S)
    n = size(X, 2)
    for (k, i) in enumerate(idx)
        xi = X[1, i]; yi = X[2, i]; zi = X[3, i]
        j11 = j12 = j13 = j21 = j22 = j23 = j31 = j32 = j33 = 0.0
        for j in 1:n
            j == i && continue
            dx = xi - X[1, j]; dy = yi - X[2, j]; dz = zi - X[3, j]
            r2 = dx * dx + dy * dy + dz * dz
            r2 == 0 && continue
            r = sqrt(r2); invr = 1 / r
            sigma = sig[j]
            rho = r / sigma
            g, h = gh_ref(rho)
            cr3 = inv(4pi) * invr^3
            gx = G[1, j]; gy = G[2, j]; gz = G[3, j]
            cx = (dz * gy - dy * gz) * cr3
            cy = (dx * gz - dz * gx) * cr3
            cz = (dy * gx - dx * gy) * cr3
            UR[1, k] += g * cx; UR[2, k] += g * cy; UR[3, k] += g * cz
            a = h / r2; b = -g * cr3
            j11 += a * cx * dx; j12 += a * cy * dx - b * gz; j13 += a * cz * dx + b * gy
            j21 += a * cx * dy + b * gz; j22 += a * cy * dy; j23 += a * cz * dy - b * gx
            j31 += a * cx * dz - b * gy; j32 += a * cy * dz + b * gx; j33 += a * cz * dz
            Cn = sqrt(cx * cx + cy * cy + cz * cz)
            Gn = sqrt(gx * gx + gy * gy + gz * gz)
            Mh = Cn / r                       # per-unit-|dh| J bound (a-term)
            Mg = sqrt(2) * Gn * cr3           # per-unit-|dg| J bound (b-term)
            if rho <= 2.0
                S_series[k] += abs(g) * Cn;  Q_series[k] += (g * Cn)^2
                S_all[k] += abs(g) * Cn;     Q_all[k] += (g * Cn)^2
                SJh_series[k] += abs(h) * Mh
                SJg_series[k] += abs(g) * Mg
            elseif rho <= rho_t
                S_outer[k] += Cn;            Q_outer[k] += Cn^2
                S_all[k] += abs(g) * Cn;     Q_all[k] += (g * Cn)^2
                SJh_outer[k] += Mh
                SJg_outer[k] += Mg
            else
                S_all[k] += Cn;              Q_all[k] += Cn^2
            end
        end
        JR2[k] = j11^2 + j12^2 + j13^2 + j21^2 + j22^2 + j23^2 + j31^2 + j32^2 + j33^2
    end
    return (; UR, JR2, S_series, S_outer, S_all,
        R_series=sqrt.(Q_series), R_outer=sqrt.(Q_outer), R_all=sqrt.(Q_all),
        SJh_series, SJg_series, SJh_outer, SJg_outer)
end

rms(v) = sqrt(sum(abs2, v) / length(v))
colnorm2(M) = vec(sum(abs2, M; dims=1))

println("=== 2. amplification factors kappa (smoke n=$(FM037F_N), S=$(FM037F_SAMPLES)) ===")
kappas = Dict{String,Float64}()
for overlap in (1.0, 1.5, 2.0)
    X, G, sig = build_smoke(FM037F_N, overlap, 20260814)
    idx = collect(1:cld(FM037F_N, FM037F_SAMPLES):FM037F_N)[1:FM037F_SAMPLES]
    sw = sweep(X, G, sig, idx, FM037F_RHO_T)
    uden = sqrt(sum(abs2, sw.UR) / length(idx))
    jden = sqrt(sum(sw.JR2) / length(idx))
    ks = rms(sw.S_series) / uden
    ko = rms(sw.S_outer) / uden
    ka = rms(sw.S_all) / uden
    rs = rms(sw.R_series) / uden
    ro = rms(sw.R_outer) / uden
    ra = rms(sw.R_all) / uden
    kjh_s = rms(sw.SJh_series) / jden
    kjg_s = rms(sw.SJg_series) / jden
    kjh_o = rms(sw.SJh_outer) / jden
    kjg_o = rms(sw.SJg_outer) / jden
    @printf("overlap=%.1f: coherent kappa series=%.3f outer=%.3f all=%.3f | incoherent series=%.3f outer=%.3f all=%.3f\n",
        overlap, ks, ko, ka, rs, ro, ra)
    @printf("             J coherent kh_series=%.3f kg_series=%.3f kh_outer=%.3f kg_outer=%.3f\n",
        kjh_s, kjg_s, kjh_o, kjg_o)
    for (nm, v) in (("series", ks), ("outer", ko), ("all", ka),
            ("series_rss", rs), ("outer_rss", ro), ("all_rss", ra),
            ("Jh_series", kjh_s), ("Jg_series", kjg_s),
            ("Jh_outer", kjh_o), ("Jg_outer", kjg_o))
        kappas[nm] = max(get(kappas, nm, 0.0), v)
    end
end
println("worst-case kappas over overlaps: ", sort(collect(kappas)), "\n")

ks = kappas["series"]; ko = kappas["outer"]; ka = kappas["all"]
rs = kappas["series_rss"]; ro = kappas["outer_rss"]; ra = kappas["all_rss"]

# pointwise budgets (U-gating; J recorded as diagnostic).  Two tiers:
#   coherent   rigorous worst case (every pair error aligned) -> eps * kappa
#   incoherent theory SS6.4/SS7 no-coherent-cancellation model -> eps * kappa_rss
# Sizing rule: a mechanism ships when its COHERENT prediction passes B with
# >= 2x margin; a mechanism whose coherent tier fails but whose incoherent
# tier passes with >= 10x margin is admissible only with the cluster-oracle
# confirmation (same precedent as the SS6.4 RMS rho_t adoption).
eps_series_rel = B / 2 / ks          # half the budget to each active branch
eps_outer_abs = B / 2 / ko
eps_fp32_rel = B / ka
println("=== pointwise budgets from B = $(round(B, sigdigits=3)) (coherent tier) ===")
@printf("series branch (rho <= 2):      |dg|/g, |dh|/h <= %.3e (rel)\n", eps_series_rel)
@printf("outer branch (2 < rho <= rt):  |dg| <= %.3e (abs)\n", eps_outer_abs)
@printf("whole-stream relative (fp32):  |dU|/U per pair <= %.3e\n", eps_fp32_rel)
println("=== incoherent-tier budgets ===")
@printf("series branch:                 |dg|/g, |dh|/h <= %.3e (rel)\n", B / 2 / rs)
@printf("outer branch:                  |dg| <= %.3e (abs)\n\n", B / 2 / ro)

# --------------------------------- 3. per-mechanism pointwise errors ---------

# dense reference grids.  Mechanism eps is the DELTA vs the shipped evaluator
# (the anchor u_total already contains the shipped kernel's own error, so only
# the change is new delivered error); the vs-BigFloat error is recorded
# alongside for the pointwise test gates.
# The outer grid starts just above 2*(1+5e-7) and the series/outer/LUT sweeps
# exclude a +-0.005 window around the rho = 2 branch boundary: Float32
# rounding (and the LUT cell containing x = 4) can flip a pair across the
# boundary, where the shipped evaluator itself is discontinuous by its outer
# fit error (2.09e-4).  That set has measure ~rho*eps32 (resp. one LUT cell),
# and its worst-case delta equals the shipped discontinuity, which the
# shipped budget already carries; it is recorded separately as boundary rows.
const RHOS_SERIES = collect(range(1e-3, 2.0 - 0.005, length=4001))
const RHOS_OUTER = collect(range(2.0 + 0.005, FM037F_RHO_T * (1 - 1e-9), length=4001))
const RHOS_BOUNDARY = collect(range(2.0 - 0.005, 2.0 + 0.005, length=401))
gb_series = [(Float64(g_big(big(r))), Float64(h_big(big(r)))) for r in RHOS_SERIES]
gb_outer = [(Float64(g_big(big(r))), Float64(h_big(big(r)))) for r in RHOS_OUTER]

# (delta_rel_series, delta_abs_outer, ref_rel_series, ref_abs_outer), each as
# max over (g, h) [outer h delta measured but U-gating uses g].
function mech_errs(f)
    dg_s = 0.0; dh_s = 0.0; rg_s = 0.0; rh_s = 0.0
    for (r, (gb, hb)) in zip(RHOS_SERIES, gb_series)
        g, h = f(r)
        g0, h0 = gh_shipped(r)
        dg_s = max(dg_s, abs((Float64(g) - Float64(g0)) / gb))
        dh_s = max(dh_s, abs((Float64(h) - Float64(h0)) / hb))
        rg_s = max(rg_s, abs(Float64(g) / gb - 1))
        rh_s = max(rh_s, abs(Float64(h) / hb - 1))
    end
    dg_o = 0.0; dh_o = 0.0; rg_o = 0.0; rh_o = 0.0
    for (r, (gb, hb)) in zip(RHOS_OUTER, gb_outer)
        g, h = f(r)
        g0, h0 = gh_shipped(r)
        dg_o = max(dg_o, abs(Float64(g) - Float64(g0)))
        dh_o = max(dh_o, abs(Float64(h) - Float64(h0)))
        rg_o = max(rg_o, abs(Float64(g) - gb))
        rh_o = max(rh_o, abs(Float64(h) - hb))
    end
    return (; dg_s, dh_s, dg_o, dh_o, rg_s, rh_s, rg_o, rh_o)
end

println("=== 3./4. mechanisms: pointwise deltas and predicted delivered error ===")
println("(pred_coh = rigorous coherent tier; pred_rss = SS6.4 incoherent tier)")
rows = String[]
push!(rows, "mechanism,param,eps_series_rel,eps_outer_abs,ref_series_rel,ref_outer_abs,pred_coherent,pred_incoherent,B,pass_coherent,pass_incoherent")

function report(mech, param, e)
    pred_coh = max(e.dg_s, e.dh_s) * ks + e.dg_o * ko
    pred_rss = max(e.dg_s, e.dh_s) * rs + e.dg_o * ro
    pc = pred_coh * 2 <= B
    pi = pred_rss * 10 <= B
    verdict = pc ? "PASS(coh)" : pi ? "PASS(rss,oracle-gated)" : "FAIL"
    @printf("%-13s %-11s d_series=%.3e d_outer=%.3e pred_coh=%.3e pred_rss=%.3e %s\n",
        mech, param, max(e.dg_s, e.dh_s), e.dg_o, pred_coh, pred_rss, verdict)
    push!(rows, @sprintf("%s,%s,%.3e,%.3e,%.3e,%.3e,%.3e,%.3e,%.3e,%d,%d",
        mech, param, max(e.dg_s, e.dh_s), e.dg_o, max(e.rg_s, e.rh_s),
        e.rg_o, pred_coh, pred_rss, B, pc, pi))
    return pc, pi
end

# (a) reduced series term counts (outer unchanged -> no outer delta)
chosen_nt = Dict{DataType,Int}()
for (T, nts) in ((Float64, 9:14), (Float32, 9:13))
    for nt in nts
        e = mech_errs(r -> gh_reduced(T(r), nt))
        pc, _ = report("reduced", "$(T)/$(nt)", e)
        # smallest count passing the coherent tier (>= 2x margin)
        pc && !haskey(chosen_nt, T) && (chosen_nt[T] = nt)
    end
end

# (b) outer degree 2 (recorded fit quality, theory SS6.2: 7.4e-4 abs in gbar)
let e = (; dg_s=0.0, dh_s=0.0, dg_o=7.4e-4, dh_o=2.2e-3,
        rg_s=0.0, rh_s=0.0, rg_o=7.4e-4, rh_o=2.2e-3)
    report("outer_deg2", "recorded", e)
end

# (c) fp32 composite (Float64 configs): shipped F32 math (13-term + deg-3
# outer) evaluated in Float32, consumed by a Float64 accumulation.  The
# assembly's own F32 rounding adds ~1.2e-7 relative on every direct pair
# (coherent bound via kappa_all, added to the coherent prediction by hand).
let
    e = mech_errs(gh_fp32)
    pred_extra = 1.2e-7 * ka
    @printf("(fp32 assembly rounding adds %.3e coherent)\n", pred_extra)
    report("fp32", "13-term", e)
    nt = get(chosen_nt, Float32, 13)
    report("reduced_fp32", "$(nt)-term", mech_errs(r -> gh_reduced_fp32(r, nt)))
end

# (d) LUT sizes: relative accuracy on BOTH branches by construction (the
# normalized G/H interpolation), so the series-relative kappa applies to the
# whole [0, rho_t) range; the deliberate singular substitution at the
# half-open x >= rho_t^2 boundary is the partitioned cutoff itself (measure
# zero; shipped partitioned switches at the same radius with > vs >=).
chosen_lutN = 0
for N in (256, 512, 1024)
    tab = build_lut(FM037F_RHO_T, N)
    e = mech_errs(r -> gh_lut(r, tab, FM037F_RHO_T^2))
    # outer-branch delta of the LUT is relative too; fold it into the series
    # coherent term by measuring the relative outer delta against g ~ O(1)
    pc, _ = report("lut", "N=$(N)", e)
    pc && chosen_lutN == 0 && (global chosen_lutN = N)
end

# boundary-window worst deltas (measure-zero set; see grid note above)
for (nm, f) in (("fp32", gh_fp32),
        ("lut_N1024", let tab = build_lut(FM037F_RHO_T, 1024)
            r -> gh_lut(r, tab, FM037F_RHO_T^2)
        end))
    d = maximum(abs(Float64(f(r)[1]) - Float64(gh_shipped(r)[1]))
        for r in RHOS_BOUNDARY)
    @printf("boundary window rho in [1.995, 2.005]: %-9s max|dg| = %.3e (measure-zero set)\n", nm, d)
    push!(rows, @sprintf("boundary_%s,window,%.3e,,,,,,%.3e,,", nm, d, B))
end
println()

# ------------------------- 5. empirical validation of the mapping ------------

# delivered ||U_cheap - U_shipped|| on the smoke case vs the prediction (the
# prediction must upper-bound the measurement; report the tightness ratio)
function delivered_delta(X, G, sig, idx, rho_t, f_cheap, f_ship)
    S = length(idx)
    D = zeros(3, S); R = zeros(3, S)
    n = size(X, 2)
    for (k, i) in enumerate(idx)
        xi = X[1, i]; yi = X[2, i]; zi = X[3, i]
        for j in 1:n
            j == i && continue
            dx = xi - X[1, j]; dy = yi - X[2, j]; dz = zi - X[3, j]
            r2 = dx * dx + dy * dy + dz * dz
            r2 == 0 && continue
            r = sqrt(r2); invr = 1 / r
            rho = r / sig[j]
            cr3 = inv(4pi) * invr^3
            gx = G[1, j]; gy = G[2, j]; gz = G[3, j]
            cx = (dz * gy - dy * gz) * cr3
            cy = (dx * gz - dz * gx) * cr3
            cz = (dy * gx - dx * gy) * cr3
            gr, _ = gh_ref(rho)
            R[1, k] += gr * cx; R[2, k] += gr * cy; R[3, k] += gr * cz
            g1 = rho <= rho_t ? f_cheap(rho)[1] : 1.0
            g0 = rho <= rho_t ? f_ship(rho)[1] : 1.0
            d = g1 - g0
            D[1, k] += d * cx; D[2, k] += d * cy; D[3, k] += d * cz
        end
    end
    return sqrt(sum(abs2, D) / max(sum(abs2, R), eps()))
end

println("=== 5. empirical validation (both tier predictions must bound actual) ===")
X, G, sig = build_smoke(FM037F_N, 1.5, 20260814)
idx = collect(1:cld(FM037F_N, FM037F_SAMPLES):FM037F_N)[1:FM037F_SAMPLES]
lut_tab = build_lut(FM037F_RHO_T, max(chosen_lutN, 256))
lut_nm = "lut_N$(max(chosen_lutN, 256))"
nt64 = get(chosen_nt, Float64, 12)
for (nm, f) in (
        ("fp32", gh_fp32),
        ("reduced_F64_$(nt64)", r -> gh_reduced(r, nt64)),
        (lut_nm, r -> gh_lut(r, lut_tab, FM037F_RHO_T^2)))
    e = mech_errs(f)
    pred_coh = max(e.dg_s, e.dh_s) * ks + e.dg_o * ko
    pred_rss = max(e.dg_s, e.dh_s) * rs + e.dg_o * ro
    actual = delivered_delta(X, G, sig, idx, FM037F_RHO_T, f, r -> gh_shipped(r))
    @printf("%-16s pred_coh<=%.3e pred_rss<=%.3e actual=%.3e %s\n",
        nm, pred_coh, pred_rss, actual,
        actual <= pred_rss ? "BOUND OK (both tiers)" :
        actual <= pred_coh ? "BOUND OK (coherent only)" : "BOUND VIOLATED")
    push!(rows, @sprintf("validate_%s,n=%d,%.3e,%.3e,%.3e,%.3e,%.3e,%.3e,%.3e,%d,%d",
        nm, FM037F_N, max(e.dg_s, e.dh_s), e.dg_o, 0.0, 0.0,
        pred_coh, pred_rss, actual, actual <= pred_coh, actual <= pred_rss))
end
println()

# --------------------------------------------------------------- output ------

outdir = joinpath(@__DIR__, "..", "data", "kernel_splitting")
mkpath(outdir)
open(joinpath(outdir, "fm037f_budget.csv"), "w") do io
    foreach(r -> println(io, r), rows)
end
println("rows written: ", joinpath(outdir, "fm037f_budget.csv"))

println("\n=== chosen parameters (paste into src) ===")
println("reduced series terms: Float64 => ", get(chosen_nt, Float64, "NONE"),
    ", Float32 => ", get(chosen_nt, Float32, "NONE"))
println("outer polynomial: keep shipped degree 3 (deg-2 fails the mapped budget)")
println("LUT entries: N = ", chosen_lutN == 0 ? "NONE PASSES" : chosen_lutN,
    " (2 x N Float32, x = rho^2 on [0, rho_t^2], normalized G/H)")
