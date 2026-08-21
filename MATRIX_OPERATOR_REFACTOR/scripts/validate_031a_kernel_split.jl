# 031a validation: partitioned regularized/singular Biot-Savart nearfield.
# Stdlib-only; formulas transcribed from FLOWVPM gpu-full:
#   g, g'          src/FLOWVPM_kernel.jl:51-57
#   U, J pair math src/FLOWVPM_fmm.jl:102-168
#
# Outputs:
#   data/kernel_splitting/rt_table.csv
#   data/kernel_splitting/partitioned_replacement.csv
#   data/kernel_splitting/geometry_coverage.csv

using Random
using Printf

outdir = joinpath(@__DIR__, "..", "data", "kernel_splitting")
mkpath(outdir)
setprecision(BigFloat, 256)

const SQRT2OPI = sqrt(2 / pi)

# --- independent 256-bit reference ------------------------------------------

function erf_series(x::BigFloat)
    s = zero(BigFloat)
    term = x
    n = 0
    while true
        s += term / (2n + 1)
        n += 1
        term *= -x * x / n
        abs(term / (2n + 1)) < eps(BigFloat) * max(abs(s), one(BigFloat)) && break
        n > 800 && error("erf series did not converge")
    end
    return 2 / sqrt(big(pi)) * s
end

function erfc_cf(x::BigFloat)
    # The continued fraction converges slowly for small x and its termination
    # test can fire early there (~1e-9 at x=0.3); the series is exact and cheap
    # in that range. All published cutoff roots sit at x ~ 3-6, where the CF is
    # accurate to full BigFloat precision, so this only removes a latent trap.
    x < 2 && return one(BigFloat) - erf_series(x)
    tiny = big(1e-80)
    f = tiny
    C = f
    D = zero(BigFloat)
    for k in 0:600
        a = k == 0 ? one(BigFloat) : big(k) / 2
        D = x + a * D
        iszero(D) && (D = tiny)
        C = x + a / C
        iszero(C) && (C = tiny)
        D = inv(D)
        delta = C * D
        f *= delta
        abs(delta - 1) < eps(BigFloat) && break
    end
    return exp(-x * x) / sqrt(big(pi)) * f
end

function g_gp_big(rho::BigFloat)
    A = sqrt(big(2) / big(pi))
    g = erf_series(rho / sqrt(big(2))) - A * rho * exp(-rho * rho / 2)
    gp = A * rho * rho * exp(-rho * rho / 2)
    return g, gp
end

function gbar_big(rho::Real)
    rb = BigFloat(rho)
    x = rb / sqrt(big(2))
    A = sqrt(big(2) / big(pi))
    return erfc_cf(x) + A * rb * exp(-rb * rb / 2)
end

# --- working-precision erf and stable small-rho series -----------------------

function erf_work(x::T) where T<:AbstractFloat
    if abs(x) < one(T)
        s = zero(T)
        term = x
        n = 0
        while true
            add = term / T(2n + 1)
            s += add
            n += 1
            term *= -x * x / T(n)
            abs(add) <= eps(T) * max(abs(s), one(T)) && break
        end
        return T(2 / sqrt(pi)) * s
    end
    # Offline validation stand-in for FLOWVPM's FDLIBM custom_erf outer branch.
    return sign(x) * (one(T) - T(erfc_cf(BigFloat(abs(x)))))
end

function g_small_series(rho::T) where T<:AbstractFloat
    z = rho * rho
    p = zero(T)
    nterms = T === Float32 ? 6 : 10
    for k in (nterms - 1):-1:0
        coeff = (isodd(k) ? -one(T) : one(T)) /
            (T(2k + 3) * T(2)^k * T(factorial(k)))
        p = muladd(p, z, coeff)
    end
    return T(SQRT2OPI) * rho * z * p
end

function h_small_series(rho::T) where T<:AbstractFloat
    z = rho * rho
    p = zero(T)
    nterms = T === Float32 ? 6 : 10
    for k in (nterms - 1):-1:0
        coeff = (iseven(k) ? -one(T) : one(T)) /
            (T(2k + 5) * T(2)^k * T(factorial(k)))
        p = muladd(p, z, coeff)
    end
    return T(SQRT2OPI) * rho * z * z * p
end

# --- exact-once source-directed cell geometry --------------------------------

function aabb_min_distance(cell_t, cell_s, h)
    d2 = 0.0
    for q in 1:3
        lt = cell_t[q] * h
        ut = (cell_t[q] + 1) * h
        ls = cell_s[q] * h
        us = (cell_s[q] + 1) * h
        gap = max(0.0, ls - ut, lt - us)
        d2 += gap * gap
    end
    return sqrt(d2)
end

function validate_geometry_coverage(; nc=4, bodies_per_cell=2, rho_t=4.789,
        sigma_lo=0.12, sigma_hi=0.20, seed=31026)
    rng = MersenneTwister(seed)
    h = 1 / nc
    positions = NTuple{3,Float64}[]
    sigmas = Float64[]
    body_cells = NTuple{3,Int}[]
    cells = [(i, j, k) for k in 0:nc-1 for j in 0:nc-1 for i in 0:nc-1]
    for cell in cells, _ in 1:bodies_per_cell
        pos = ntuple(q -> (cell[q] + 0.1 + 0.8rand(rng)) * h, 3)
        push!(positions, pos)
        # Vary source radii enough that cutoff pairs cross cell boundaries;
        # this exercises the directional sigma_max predicate nontrivially.
        # sigma_lo/sigma_hi are in units of the cell width h: the default pair
        # is the sub-cell-cutoff case, and the target-regime case below uses
        # overlap-2 leaf values where rho_t*sigma spans several cells.
        push!(sigmas, h * (sigma_lo + (sigma_hi - sigma_lo) * rand(rng)))
        push!(body_cells, cell)
    end
    sigma_max = Dict(cell => maximum(sigmas[i] for i in eachindex(sigmas)
        if body_cells[i] == cell) for cell in cells)
    direct = Set{Tuple{Int,Int}}()
    m2l = Set{Tuple{Int,Int}}()
    cutoff_pairs = 0
    cutoff_in_m2l = 0
    n = length(positions)
    for it in 1:n, is in 1:n
        it == is && continue
        ct = body_cells[it]
        cs = body_cells[is]
        is_direct = aabb_min_distance(ct, cs, h) <= rho_t * sigma_max[cs]
        pair = (it, is)
        push!(is_direct ? direct : m2l, pair)
        r = sqrt(sum(q -> (positions[it][q] - positions[is][q])^2, 1:3))
        if r / sigmas[is] <= rho_t
            cutoff_pairs += 1
            !is_direct && (cutoff_in_m2l += 1)
        end
    end
    all_pairs = Set((it, is) for it in 1:n for is in 1:n if it != is)
    missing = length(setdiff(all_pairs, union(direct, m2l)))
    duplicates = length(intersect(direct, m2l))
    missing == 0 || error("geometry coverage has $missing missing pairs")
    duplicates == 0 || error("geometry coverage has $duplicates duplicates")
    cutoff_in_m2l == 0 || error("geometry routed $cutoff_in_m2l cutoff pairs to M2L")
    return (n_bodies=n, n_pairs=length(all_pairs), n_direct=length(direct),
        n_m2l=length(m2l), n_cutoff=cutoff_pairs, missing=missing,
        duplicates=duplicates, cutoff_in_m2l=cutoff_in_m2l,
        cutoff_over_h=rho_t * maximum(values(sigma_max)) / h)
end

# --- near-set adequacy of a fixed translation-invariant stencil ---------------
#
# The exact-once predicate above is a per-cell-pair test. A shipped radix
# stencil instead fixes one offset set at every level, so its adequacy is
# governed by the SMALLEST AABB gap it leaves to M2L -- not by its outer
# centre-to-centre radius. For offset o (in cells) the gap is
#   gap(o) = h * sqrt( sum_q max(0, |o_q| - 1)^2 ),
# so the stencil is adequate iff min over M2L offsets of gap(o) > rho_t*sigma.
# With sigma = beta*d, d = L/n^(1/3) and h = L/2^ell this reduces to the
# level-invariant bodies-per-leaf floor  n/8^ell > (rho_t*beta/g_min)^3.

offset_gap(o) = sqrt(sum(q -> max(0, abs(q) - 1)^2, o))

const STENCILS = (
    ("theta0.5_|o|^2<=12", o -> sum(abs2, o) <= 12),
    ("classic_|o|inf<=1", o -> maximum(abs, o) <= 1),
)

function stencil_min_gap(is_direct; reach=8)
    offsets = [(i, j, k) for i in -reach:reach for j in -reach:reach
        for k in -reach:reach]
    n_direct = count(is_direct, offsets)
    gaps = (offset_gap(o) for o in offsets if !is_direct(o))
    return n_direct, minimum(gaps)
end

"""
Minimal direct offset count that satisfies `gap(o) > required_gap_h` for every
M2L offset; this is the "enlarge the stencil" remedy at fixed `ell`.

Exact for any `required_gap_h`: since `gap(o) >= |o|_inf - 1`, no offset with
`|o|_inf > required_gap_h + 1` can qualify, so enumerating per-coordinate
magnitudes up to that bound (with multiplicity 1 for 0 and 2 for +/-a) counts
every qualifying offset and no others. A fixed enumeration half-width would
silently clip the count wherever the required gap outgrows it.
"""
function required_direct_count(required_gap_h)
    g2 = required_gap_h^2
    reach = floor(Int, required_gap_h) + 1
    contrib(a) = a <= 1 ? 0.0 : float(a - 1)^2
    mult(a) = a == 0 ? 1 : 2
    total = 0
    for a in 0:reach
        ca = contrib(a)
        ca > g2 && break
        for b in 0:reach
            cab = ca + contrib(b)
            cab > g2 && break
            for c in 0:reach
                cab + contrib(c) > g2 && break
                total += mult(a) * mult(b) * mult(c)
            end
        end
    end
    return total
end

# agreement with the direct enumeration used elsewhere, on the shipped radii
let brute = required_gap_h -> count(o -> offset_gap(o) <= required_gap_h,
        ((i, j, k) for i in -9:9, j in -9:9, k in -9:9))
    for gh in (0.0, 0.766, 1.0, 1.533, 3.065, 6.130, 8.0)
        required_direct_count(gh) == brute(gh) ||
            error("required_direct_count disagrees with enumeration at $gh")
    end
end

function near_set_adequacy_rows(; betas=(1.5, 2.0, 2.5),
        eps_targets=(1e-3, 1e-4, 1e-6), ns=(1e3, 1e4, 1e5, 1e6), ells=0:6)
    rows = NamedTuple[]
    for (name, is_direct) in STENCILS
        n_classes, g_min = stencil_min_gap(is_direct)
        for eps in eps_targets
            rho_t = solve_rt(trunc_j, TRUNC_BOUND_FRACTION * eps)
            for beta in betas, n in ns, ell in ells
                required_gap_h = rho_t * beta * 2.0^ell / cbrt(n)
                bodies_per_leaf = n / 8.0^ell
                required_bodies = (rho_t * beta / g_min)^3
                compliant = g_min > required_gap_h
                n_required = required_direct_count(required_gap_h)
                # Cost model (§6): regularized pairs per target are the
                # physics floor and are independent of ell; direct pairs per
                # target scale with the compliant class count.
                reg_pairs = 4pi / 3 * (rho_t * beta)^3
                direct_pairs = n_required * bodies_per_leaf
                f = min(1.0, reg_pairs / direct_pairs)
                # Production geometry: the near set can never be SMALLER than
                # the stencil's own class count, which is fixed by far-field
                # multipole accuracy (theta=0.5 -> 179, classic -> 27), not by
                # regularization. Where the adequacy floor is looser than the
                # stencil, the stencil binds and the direct volume is larger.
                n_production = max(n_classes, n_required)
                direct_pairs_prod = n_production * bodies_per_leaf
                f_prod = min(1.0, reg_pairs / direct_pairs_prod)
                push!(rows, (stencil=name, n_classes=n_classes, g_min=g_min,
                    n=n, ell=ell, beta=beta, eps=eps, rho_t=rho_t,
                    required_gap_h=required_gap_h, bodies_per_leaf=bodies_per_leaf,
                    required_bodies_per_leaf=required_bodies, compliant=compliant,
                    n_classes_required=n_required, reg_pairs_per_target=reg_pairs,
                    direct_pairs_per_target=direct_pairs, f_regularized=f,
                    speedup_q15=1 / (f + (1 - f) / 1.5),
                    speedup_q25=1 / (f + (1 - f) / 2.5),
                    n_classes_production=n_production,
                    direct_pairs_production=direct_pairs_prod,
                    f_production=f_prod,
                    speedup_production_q15=1 / (f_prod + (1 - f_prod) / 1.5),
                    speedup_production_q25=1 / (f_prod + (1 - f_prod) / 2.5)))
            end
        end
    end
    # Self-consistency: the closed-form bodies-per-leaf floor must agree with
    # the direct gap comparison in every row.
    for r in rows
        agree = (r.bodies_per_leaf > r.required_bodies_per_leaf) == r.compliant
        # allow disagreement only within rounding of the compliance boundary
        borderline = abs(r.g_min / r.required_gap_h - 1) < 1e-9
        agree || borderline ||
            error("bodies-per-leaf rule disagrees with gap test at $(r)")
    end
    return rows
end

function g_h_stable(rho::T) where T<:AbstractFloat
    if abs(rho) <= T(0.5)
        return g_small_series(rho), h_small_series(rho)
    end
    A = T(SQRT2OPI)
    g = erf_work(rho / sqrt(T(2))) - A * rho * exp(-rho * rho / 2)
    gp = A * rho * rho * exp(-rho * rho / 2)
    return g, rho * gp - 3g
end

# --- exact component layout --------------------------------------------------

function pair_common(dx, Gamma)
    T = eltype(dx)
    r2 = sum(abs2, dx)
    r = sqrt(r2)
    r3inv = inv(r2 * r)
    c4 = -T(1 / (4pi))
    crss = (c4 * r3inv * (dx[2] * Gamma[3] - dx[3] * Gamma[2]),
             c4 * r3inv * (dx[3] * Gamma[1] - dx[1] * Gamma[3]),
             c4 * r3inv * (dx[1] * Gamma[2] - dx[2] * Gamma[1]))
    return r, r2, r3inv, c4, crss
end

function j_components(crss, dx, Gamma, a, b)
    return (
        a * crss[1] * dx[1],
        a * crss[2] * dx[1] - b * Gamma[3],
        a * crss[3] * dx[1] + b * Gamma[2],
        a * crss[1] * dx[2] + b * Gamma[3],
        a * crss[2] * dx[2],
        a * crss[3] * dx[2] - b * Gamma[1],
        a * crss[1] * dx[3] - b * Gamma[2],
        a * crss[2] * dx[3] + b * Gamma[1],
        a * crss[3] * dx[3],
    )
end

function uj_regularized_stable(dx, Gamma, sigma)
    r, r2, r3inv, c4, crss = pair_common(dx, Gamma)
    g, h = g_h_stable(r / sigma)
    U = ntuple(i -> g * crss[i], 3)
    return U, j_components(crss, dx, Gamma, h / r2, c4 * g * r3inv)
end

function uj_singular(dx, Gamma)
    r, r2, r3inv, c4, crss = pair_common(dx, Gamma)
    return crss, j_components(crss, dx, Gamma, -3 / r2, c4 * r3inv)
end

function uj_regularized_big(dx, Gamma, sigma)
    xb = map(BigFloat, dx)
    gb = map(BigFloat, Gamma)
    sb = BigFloat(sigma)
    r2 = sum(abs2, xb)
    r = sqrt(r2)
    rho = r / sb
    g, gp = g_gp_big(rho)
    c4 = -inv(4big(pi))
    r3inv = inv(r2 * r)
    crss = (c4 * r3inv * (xb[2] * gb[3] - xb[3] * gb[2]),
             c4 * r3inv * (xb[3] * gb[1] - xb[1] * gb[3]),
             c4 * r3inv * (xb[1] * gb[2] - xb[2] * gb[1]))
    a = gp / (sb * r) - 3g / r2
    U = ntuple(i -> g * crss[i], 3)
    return U, j_components(crss, xb, gb, a, c4 * g * r3inv)
end

norm2(v) = sqrt(sum(abs2, v))
relerr(v, ref) = Float64(norm2(map(BigFloat, v) .- ref) / norm2(ref))

# --- cutoff table ------------------------------------------------------------

trunc_u(rho) = gbar_big(rho)
function trunc_j(rho)
    rb = BigFloat(rho)
    gp = sqrt(big(2) / big(pi)) * rb * rb * exp(-rb * rb / 2)
    return gbar_big(rb) + rb * gp / 2
end

function solve_rt(f, eps_target)
    lo, hi = big(0.5), big(15)
    target = BigFloat(eps_target)
    for _ in 1:220
        mid = (lo + hi) / 2
        f(mid) > target ? (lo = mid) : (hi = mid)
    end
    return Float64((lo + hi) / 2)
end

const TRUNC_BOUND_FRACTION = 0.5
rt_rows = Tuple{Float64,Float64,Float64}[]
open(joinpath(outdir, "rt_table.csv"), "w") do io
    println(io, "eps,rt_U,rt_J")
    for e in [1e-3, 1e-4, 1e-6, 1e-7, 1e-12, 1e-15]
        ru = solve_rt(trunc_u, TRUNC_BOUND_FRACTION * e)
        rj = solve_rt(trunc_j, TRUNC_BOUND_FRACTION * e)
        push!(rt_rows, (e, ru, rj))
        @printf(io, "%.0e,%.3f,%.3f\n", e, ru, rj)
    end
end

# --- partitioned replacement validation -------------------------------------

open(joinpath(outdir, "partitioned_replacement.csv"), "w") do io
    println(io, "precision,rho_min,rho_max,max_rel_U_close,max_rel_J_close,max_rel_U_ordinary,max_rel_J_ordinary,max_rel_U_tail,max_rel_J_tail")
    rt = solve_rt(trunc_j, TRUNC_BOUND_FRACTION * 1e-6)
    for T in (Float64, Float32)
        close_u = close_j = ordinary_u = ordinary_j = tail_u = tail_j = 0.0
        Gamma = (T(0.73), T(-1.17), T(0.41))
        for rho64 in 10.0 .^ range(-6, log10(0.5), length=1200)
            rho = T(rho64)
            dx0 = (rho * T(0.37), rho * T(-0.51), rho * T(0.775))
            scale = rho / sqrt(sum(abs2, dx0))
            dx = ntuple(i -> dx0[i] * scale, 3)
            Uw, Jw = uj_regularized_stable(dx, Gamma, one(T))
            Ur, Jr = uj_regularized_big(dx, Gamma, one(T))
            close_u = max(close_u, relerr(Uw, Ur))
            close_j = max(close_j, relerr(Jw, Jr))
        end
        for rho64 in range(0.5, rt, length=1200)
            rho = T(rho64)
            dx = (rho * T(0.31), rho * T(-0.72), rho * T(0.62))
            scale = rho / sqrt(sum(abs2, dx))
            dx = ntuple(i -> dx[i] * scale, 3)
            Uw, Jw = uj_regularized_stable(dx, Gamma, one(T))
            Ur, Jr = uj_regularized_big(dx, Gamma, one(T))
            ordinary_u = max(ordinary_u, relerr(Uw, Ur))
            ordinary_j = max(ordinary_j, relerr(Jw, Jr))
        end
        for rho64 in range(nextfloat(rt), 10.0, length=1200)
            rho = T(rho64)
            dx = (rho * T(0.31), rho * T(-0.72), rho * T(0.62))
            scale = rho / sqrt(sum(abs2, dx))
            dx = ntuple(i -> dx[i] * scale, 3)
            Uw, Jw = uj_singular(dx, Gamma)
            Ur, Jr = uj_regularized_big(dx, Gamma, one(T))
            tail_u = max(tail_u, relerr(Uw, Ur))
            tail_j = max(tail_j, relerr(Jw, Jr))
        end
        precision_limit = T === Float64 ? 2e-14 : 2e-6
        max(close_u, close_j, ordinary_u, ordinary_j) <= precision_limit ||
            error("$T regularized branch exceeded $precision_limit")
        max(tail_u, tail_j) <= 1e-6 || error("$T tail exceeded 1e-6")
        @printf(io, "%s,1e-6,10,%.3e,%.3e,%.3e,%.3e,%.3e,%.3e\n",
            T, close_u, close_j, ordinary_u, ordinary_j, tail_u, tail_j)
        @printf("%s: close U/J %.3e %.3e; ordinary %.3e %.3e; tail %.3e %.3e\n",
            T, close_u, close_j, ordinary_u, ordinary_j, tail_u, tail_j)
    end
end

# --- two-pass additive correction (third 032a candidate) ---------------------
#
# Pass 1 is the UNMODIFIED singular FMM (singular far field + singular direct).
# Pass 2 adds the regularization deficit over the cutoff shell only:
#   dU_i  = -gbar(rho) * C_i
#   dJ_ij = da * C_i * dx_j + db * eps_ijk * Gamma_k
#   da = (rho*g' + 3*gbar)/r^2,   db = +gbar/(4*pi*r^3)
# so that (sing + correction) == regularized exactly in exact arithmetic.
# Neither term needs erfc: gbar = 1 - g loses digits only where gbar is already
# a negligible fraction of the retained singular value.

function uj_correction(dx, Gamma, sigma)
    r, r2, r3inv, c4, crss = pair_common(dx, Gamma)
    rho = r / sigma
    g, _ = g_h_stable(rho)          # h is unused; g via the same stable branch
    gbar = one(g) - g
    gp = eltype(dx)(SQRT2OPI) * rho * rho * exp(-rho * rho / 2)
    dU = ntuple(i -> -gbar * crss[i], 3)
    return dU, j_components(crss, dx, Gamma, (rho * gp + 3gbar) / r2,
        -c4 * gbar * r3inv)
end

"Pass 1 (singular) + pass 2 (deficit), accumulated in working precision T."
function uj_two_pass(dx, Gamma, sigma)
    Us, Js = uj_singular(dx, Gamma)
    dU, dJ = uj_correction(dx, Gamma, sigma)
    return ntuple(i -> Us[i] + dU[i], 3), ntuple(i -> Js[i] + dJ[i], 9)
end

"Hybrid: stable regularized inside rho_c, singular+correction outside."
function uj_two_pass_hybrid(dx, Gamma, sigma, rho_c)
    sqrt(sum(abs2, dx)) / sigma <= rho_c && return uj_regularized_stable(dx, Gamma, sigma)
    return uj_two_pass(dx, Gamma, sigma)
end

# predicted amplification of a unit rounding error, per pair
amp_u_pred(rho) = 1 / first(g_h_stable(rho))
function amp_j_pred(rho)
    g, _ = g_h_stable(rho)
    gp = SQRT2OPI * rho * rho * exp(-rho * rho / 2)
    return 2 / abs(rho * gp - 2g)
end

const RHO_C = 2.0

# The transverse regularized J coefficient rho*g' - 2g changes sign, so a rho_c
# placed at the crossing would compare against a vanishing component. Locate it
# and require rho_c to sit safely beyond.
let f = rho -> (g = first(g_h_stable(rho)); rho * SQRT2OPI * rho * rho * exp(-rho * rho / 2) - 2g)
    lo, hi = 1.0, 1.5
    f(lo) * f(hi) < 0 || error("expected a sign change of rho*g'-2g in (1, 1.5)")
    for _ in 1:80
        mid = (lo + hi) / 2
        f(lo) * f(mid) <= 0 ? (hi = mid) : (lo = mid)
    end
    crossing = (lo + hi) / 2
    RHO_C > crossing + 0.5 ||
        error("rho_c=$RHO_C is too close to the rho*g'-2g crossing at $crossing")
    @printf("rho*g'-2g sign crossing at rho=%.4f; rho_c=%.1f\n", crossing, RHO_C)
end

rt_j_gate = solve_rt(trunc_j, TRUNC_BOUND_FRACTION * 1e-3)
open(joinpath(outdir, "two_pass_conditioning.csv"), "w") do io
    println(io, "precision,rho,amp_U_pred,amp_J_pred,relerr_U_two_pass," *
        "relerr_J_two_pass,relerr_J_single_pass,relerr_J_hybrid")
    for T in (Float64, Float32)
        Gamma = (T(0.73), T(-1.17), T(0.41))
        dirn = (0.31, -0.72, 0.62)
        nrm = sqrt(sum(abs2, dirn))
        measured = Tuple{Float64,Float64}[]
        for rho64 in (0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, RHO_C, 3.0, rt_j_gate)
            rho = T(rho64)
            dx = ntuple(i -> T(dirn[i] / nrm) * rho, 3)
            Ur, Jr = uj_regularized_big(dx, Gamma, one(T))
            U2, J2 = uj_two_pass(dx, Gamma, one(T))
            U1, J1 = uj_regularized_stable(dx, Gamma, one(T))
            _, Jh = uj_two_pass_hybrid(dx, Gamma, one(T), T(RHO_C))
            e2u, e2j = relerr(U2, Ur), relerr(J2, Jr)
            e1j, ehj = relerr(J1, Jr), relerr(Jh, Jr)
            push!(measured, (rho64, e2j))
            @printf(io, "%s,%.4g,%.3e,%.3e,%.3e,%.3e,%.3e,%.3e\n", T, rho64,
                amp_u_pred(rho64), amp_j_pred(rho64), e2u, e2j, e1j, ehj)
        end
        # the hybrid and the single-pass form must both stay at working precision
        # everywhere, while two-pass degrades like amp*eps below rho_c
        for (rho64, _) in measured
            rho = T(rho64)
            dx = ntuple(i -> T(dirn[i] / nrm) * rho, 3)
            Ur, Jr = uj_regularized_big(dx, Gamma, one(T))
            _, Jh = uj_two_pass_hybrid(dx, Gamma, one(T), T(RHO_C))
            limit = T === Float64 ? 2e-14 : 2e-6
            relerr(Jh, Jr) <= limit ||
                error("$T rho_c hybrid exceeded $limit at rho=$rho64")
        end
        # measured two-pass degradation must track the predicted amplification
        for (rho64, ej) in measured
            rho64 >= 0.5 && continue
            pred = amp_j_pred(rho64) * eps(T)
            (ej <= 30 * pred + 10 * eps(T)) ||
                error("$T two-pass error $ej exceeds 30x predicted $pred at rho=$rho64")
        end
        # per-pair breakdown radius: where two-pass relative error reaches 1e-3.
        # When the scan finds no breakdown, its floor is an upper bound on the
        # breakdown radius, so report the pair count below the floor instead of
        # a meaningless zero.
        scan_floor = 1e-4
        brk = 0.0
        for rho64 in 10.0 .^ range(log10(scan_floor), 0, length=2000)
            rho = T(rho64)
            dx = ntuple(i -> T(dirn[i] / nrm) * rho, 3)
            Ur, Jr = uj_regularized_big(dx, Gamma, one(T))
            J2 = last(uj_two_pass(dx, Gamma, one(T)))
            relerr(J2, Jr) >= 1e-3 && (brk = rho64)
        end
        # unordered pairs within rho of a target in a uniform n=1e6, beta=2 field
        pairs_within(rho) = 0.5e6 * (4pi / 3) * (rho * 2)^3
        if brk == 0.0
            @printf("two-pass %s: per-pair J error never reaches 1e-3 above rho=%.0e (pairs below that radius at n=1e6, beta=2: %.3g)\n",
                T, scan_floor, pairs_within(scan_floor))
        else
            @printf("two-pass %s: per-pair J error reaches 1e-3 below rho=%.3g (expected such pairs at n=1e6, beta=2: %.3g)\n",
                T, brk, pairs_within(brk))
        end
    end
end

# --- three-strategy cost model on adequate geometry --------------------------
#
# Pass 1's near set cannot be shrunk to rho_c: its size is fixed by far-field
# multipole accuracy (the theta=0.5 stencil), not by regularization. Two-pass
# therefore pays a second traversal over the rho_t neighbourhood.
#   c(reg-everywhere) = P_t*(lambda + q)
#   c(partitioned)    = P_t*lambda + N_t*q + (P_t - N_t)
#   c(two-pass)       = (P_1 + P_t)*lambda + N_t*q + (P_1 - N_c)
# with lambda the per-visit load/index cost and q the regularized-math cost,
# both in units of one singular U+J evaluation.
open(joinpath(outdir, "two_pass_cost_model.csv"), "w") do io
    println(io, "n,ell,beta,eps,rho_c,rho_t,shipped_classes,pass1_pairs," *
        "reg_pairs_inside_rho_c,pass2_classes,pass2_pairs,partition_classes," *
        "partition_pairs,cutoff_pairs,extra_visits,saved_singular_evals," *
        "lambda_crossover_vs_partitioned")
    n, beta, epsg = 1e6, 2.0, 1e-3
    rho_t = rt_j_gate
    for ell in 3:6
        bodies = n / 8.0^ell
        gap_t = rho_t * beta * 2.0^ell / cbrt(n)
        gap_c = RHO_C * beta * 2.0^ell / cbrt(n)
        shipped = 179
        # pass 1 keeps the shipped near set where that is adequate for rho_c,
        # else it must already be enlarged to cover rho_c
        pass1_classes = max(shipped, required_direct_count(gap_c))
        # pass 2 builds its own tight list reaching rho_t; where that is a
        # subset of pass 1's near set it is still realizable as a subset
        pass2_classes = required_direct_count(gap_t)
        # the single-pass baselines run on the production near set, which is
        # the larger of the stencil and the adequacy requirement
        partition_classes = max(shipped, pass2_classes)
        P1 = pass1_classes * bodies
        Pt = pass2_classes * bodies
        Pd = partition_classes * bodies
        Nt = 4pi / 3 * (rho_t * beta)^3
        Nc = 4pi / 3 * (RHO_C * beta)^3
        # two_pass - partitioned
        #   = (P1 + Pt - Pd)*lambda + (P1 - Nc) - (Pd - Nt)
        dvisit = P1 + Pt - Pd
        dmath = (Pd - Nt) - (P1 - Nc)
        lambda_star = dvisit > 0 ? dmath / dvisit : (dmath >= 0 ? Inf : -Inf)
        @printf(io, "%.0f,%d,%.1f,%.0e,%.1f,%.3f,%d,%.1f,%.1f,%d,%.1f,%d,%.1f,%.1f,%.1f,%.1f,%.3f\n",
            n, ell, beta, epsg, RHO_C, rho_t, pass1_classes, P1, Nc,
            pass2_classes, Pt, partition_classes, Pd, Nt, dvisit, dmath,
            lambda_star)
        @printf("two-pass cost ell=%d: pass1 %d cls/%.0f pairs, pass2 %d cls/%.0f pairs, partitioned %d cls/%.0f pairs, extra visits %.0f, saved singular %.0f, lambda* = %.3f\n",
            ell, pass1_classes, P1, pass2_classes, Pt, partition_classes, Pd,
            dvisit, dmath, lambda_star)
    end
end

# --- working-precision erfc for the shell analyses below ---------------------

function erfc_f64(x::Float64)
    if x < 2
        s = 0.0; t = x; n = 0
        while true
            a = t / (2n + 1); s += a; n += 1; t *= -x * x / n
            abs(a) <= 1e-18 * max(abs(s), 1.0) && break
        end
        return 1 - 2 / sqrt(pi) * s
    end
    tiny = 1e-300; f = tiny; C = f; D = 0.0
    for k in 0:200
        a = k == 0 ? 1.0 : k / 2
        D = x + a * D; iszero(D) && (D = tiny)
        C = x + a / C; iszero(C) && (C = tiny)
        D = inv(D); d = C * D; f *= d
        abs(d - 1) <= 1e-17 && break
    end
    return exp(-x * x) / sqrt(pi) * f
end
for x in (0.3, 1.0, 1.9, 2.1, 3.4, 6.0)
    abs(erfc_f64(x) / Float64(erfc_cf(BigFloat(x))) - 1) <= 1e-13 ||
        error("erfc_f64 disagrees with the 256-bit reference at x=$x")
end

gbar_f64(rho) = erfc_f64(rho / sqrt(2)) + SQRT2OPI * rho * exp(-rho * rho / 2)
g_f64(rho) = 1 - gbar_f64(rho)
gp_f64(rho) = SQRT2OPI * rho * rho * exp(-rho * rho / 2)

# --- 6.2 cheap outer-shell gbar: one exp, no erf -----------------------------
#
# Above rho_c the retained result is O(1), so gbar is needed only to ABSOLUTE
# tolerance (eps/2)*g, not relative. Writing
#   gbar(rho) = exp(-rho^2/2) * (A*rho + s(rho)),  s(rho) = erfc(rho/sqrt2)*exp(rho^2/2)
# the whole erf disappears into s, which is smooth and monotone on the shell and
# is captured by a short polynomial in u = 1/rho^2. rho*g' = A*rho^3*exp(-rho^2/2)
# reuses the same exponential, so the outer branch costs one exp plus FMAs.

s_scaled(rho) = erfc_f64(rho / sqrt(2)) * exp(rho * rho / 2)

function fit_s_poly(deg, rho_c, rho_t; m=400)
    rs = collect(range(rho_c, rho_t, length=m))
    V = [(1 / r^2)^j for r in rs, j in 0:deg]
    c = (V' * V) \ (V' * [s_scaled(r) for r in rs])
    ds = maximum(abs(sum(c[j + 1] * (1 / r^2)^j for j in 0:deg) - s_scaled(r))
                 for r in rs)
    dgbar = maximum(exp(-r * r / 2) *
                    abs(sum(c[j + 1] * (1 / r^2)^j for j in 0:deg) - s_scaled(r))
                    for r in rs)
    return c, ds, dgbar
end

open(joinpath(outdir, "cheap_gbar_fit.csv"), "w") do io
    println(io, "rho_c,rho_t,degree,max_abs_err_s,max_abs_err_gbar,budget_abs_gbar,meets_budget")
    for (rho_c, rho_t) in ((RHO_C, rt_j_gate),)
        # the binding point is rho_c: |dgbar| decays like exp(-rho^2/2) while the
        # budget (eps/2)*g(rho) grows toward eps/2
        budget = 0.5e-3 * g_f64(rho_c)
        best = 0
        for deg in 1:4
            _, ds, dgb = fit_s_poly(deg, rho_c, rho_t)
            ok = dgb <= budget
            ok && best == 0 && (best = deg)
            @printf(io, "%.1f,%.3f,%d,%.3e,%.3e,%.3e,%d\n",
                rho_c, rho_t, deg, ds, dgb, budget, ok)
        end
        best == 0 && error("no degree<=4 polynomial meets the gbar budget")
        @printf("cheap gbar: degree %d in 1/rho^2 meets the %.2e absolute budget on [%.1f, %.3f] (one exp, no erf)\n",
            best, budget, rho_c, rho_t)
    end
end

# --- 6.3 warp-divergence cost model ------------------------------------------
#
# The lambda* model assumes a branch-free pair stream. On a GPU a warp pays a
# branch taken by ANY of its lanes, so with a regularized fraction f the
# unbinned partitioned kernel pays BOTH paths on essentially every warp.

const WARP = 32
p_homogeneous(f) = f^WARP + (1 - f)^WARP

open(joinpath(outdir, "divergence_model.csv"), "w") do io
    println(io, "n,ell,beta,eps,f_regularized,p_homogeneous_warp,classes_direct," *
        "classes_entirely_inside,classes_mixed,cost_binned_q15,cost_unbinned_q15," *
        "cost_unbinned_warpweighted_q15,cost_reg_everywhere_q15," *
        "unbinned_over_reg_everywhere_weighted,unbinned_beats_reg_everywhere")
    n, beta = 1e6, 2.0
    rho_t = rt_j_gate
    lambda = 0.3   # representative per-visit load cost, in singular-eval units
    for ell in 3:6
        bodies = n / 8.0^ell
        gap_t = rho_t * beta * 2.0^ell / cbrt(n)
        nD = max(179, required_direct_count(gap_t))
        # classes whose FARTHEST corner is still inside the cutoff: every pair in
        # them is regularized, so they never diverge
        reach = floor(Int, gap_t) + 1
        inside = count(o -> sqrt(sum(q -> (abs(q) + 1)^2, o)) <= gap_t,
            ((i, j, k) for i in -reach:reach, j in -reach:reach, k in -reach:reach))
        Pd = nD * bodies
        Nt = 4pi / 3 * (rho_t * beta)^3
        f = min(1.0, Nt / Pd)
        q = 1.5
        c_binned = Pd * (lambda + f * q + (1 - f))
        c_unbinned = Pd * (lambda + q + 1)          # every warp pays both paths
        c_regall = Pd * (lambda + q)
        # A warp is homogeneous with probability f^32 (all regularized) or
        # (1-f)^32 (all singular) and pays only that path; only a mixed warp
        # pays q+1. The fully-diverged bound above is exact at the shipped
        # ell=5 operating point (p = 7e-6) but overstates the penalty wherever
        # f is near 0 or 1 -- at ell=3 the nearly all-singular stream keeps 71%
        # of warps homogeneous.
        p_reg, p_sing = f^WARP, (1 - f)^WARP
        p_mix = 1 - p_reg - p_sing
        c_unbinned_w = Pd * (lambda + p_reg * q + p_sing * 1 + p_mix * (q + 1))
        @printf(io, "%.0f,%d,%.1f,%.0e,%.4f,%.3e,%d,%d,%d,%.1f,%.1f,%.1f,%.1f,%.4f,%d\n",
            n, ell, beta, 1e-3, f, p_homogeneous(f), nD, inside, nD - inside,
            c_binned, c_unbinned, c_unbinned_w, c_regall,
            c_unbinned_w / c_regall, c_unbinned_w <= c_regall)
        ell == 5 && @printf("divergence ell=5: f=%.3f, P(homogeneous warp)=%.2e, %d of %d classes entirely inside; unbinned partitioned costs %.2fx regularized-everywhere\n",
            f, p_homogeneous(f), inside, nD, c_unbinned_w / c_regall)
        ell == 3 && @printf("divergence ell=3: f=%.3f, P(homogeneous warp)=%.2e; warp-weighted unbinned costs %.2fx regularized-everywhere (fully-diverged bound %.2fx)\n",
            f, p_homogeneous(f), c_unbinned_w / c_regall, c_unbinned / c_regall)
    end
end

# --- 6.4 accumulated-tail cutoff radius --------------------------------------
#
# The section-4 radii bound the PER-PAIR relative error. The phase gate is a
# sampled RMS over targets, and for a uniform field with random Gamma
# orientations the omitted tail adds incoherently. Modelling shell pair counts
# as 4*pi*r^2*(n/V)dr and per-pair magnitudes as |C| ~ 1/r^2 (U) and 1/r^3 (J),
# the relative RMS error is sqrt(I_tail(rho_t)/I_total) with
#   U: weight gbar^2/rho^2   against  g^2/rho^2
#   J: weight (rho*g'+2gbar)^2/rho^4  against  (rho*g'-2g)^2/rho^4.

function simpson(f, a, b, n)
    n = 2 * div(n, 2)
    h = (b - a) / n
    s = f(a) + f(b)
    for i in 1:n-1
        s += (isodd(i) ? 4 : 2) * f(a + i * h)
    end
    return s * h / 3
end

wU_tail(r) = gbar_f64(r)^2 / r^2
wU_tot(r) = g_f64(r)^2 / r^2
wJ_tail(r) = (r * gp_f64(r) + 2gbar_f64(r))^2 / r^4
wJ_tot(r) = (r * gp_f64(r) - 2g_f64(r))^2 / r^4

function solve_rt_rms(w, den, target)
    lo, hi = 0.5, 12.0
    for _ in 1:60
        mid = (lo + hi) / 2
        sqrt(simpson(w, mid, 18.0, 4000) / den) > target ? (lo = mid) : (hi = mid)
    end
    return (lo + hi) / 2
end

open(joinpath(outdir, "rt_accumulated.csv"), "w") do io
    denU = simpson(wU_tot, 1e-6, 60.0, 40000)
    denJ = simpson(wJ_tot, 1e-6, 60.0, 40000)
    # denU carries a weak 1/R domain tail; denJ is domain-independent
    denU20 = simpson(wU_tot, 1e-6, 20.0, 20000)
    denU100 = simpson(wU_tot, 1e-6, 100.0, 60000)
    println(io, "eps,rt_U_perpair,rt_U_rms,rt_J_perpair,rt_J_rms,N_reg_ratio_J," *
        "classes_perpair_ell5,classes_rms_ell5,denU_R20,denU_R60,denU_R100,denJ_R60")
    for (e, ru_pp, rj_pp) in rt_rows
        e in (1e-3, 1e-4, 1e-6) || continue
        ru = solve_rt_rms(wU_tail, denU, e / 2)
        rj = solve_rt_rms(wJ_tail, denJ, e / 2)
        gap_pp = rj_pp * 2.0 * 32 / 100
        gap_rms = rj * 2.0 * 32 / 100
        @printf(io, "%.0e,%.3f,%.3f,%.3f,%.3f,%.3f,%d,%d,%.4f,%.4f,%.4f,%.4f\n",
            e, ru_pp, ru, rj_pp, rj, (rj / rj_pp)^3,
            required_direct_count(gap_pp), required_direct_count(gap_rms),
            denU20, denU, denU100, denJ)
        e == 1e-3 && @printf("accumulated-tail cutoff: rho_t(J) %.3f -> %.3f, expensive pairs x%.3f, ell=5 classes %d -> %d\n",
            rj_pp, rj, (rj / rj_pp)^3, required_direct_count(gap_pp),
            required_direct_count(gap_rms))
    end
end

geometry_cases = (
    # sub-cell cutoff: rho_t*sigma_max < h
    ("subcell", validate_geometry_coverage()),
    # target regime: overlap-2 leaf values, rho_t*sigma_max ~ 3h, so cutoff
    # pairs cross several cells and the predicate is exercised where a fixed
    # |o|^2<=12 stencil would be inadequate.
    ("overlap2_leaf", validate_geometry_coverage(; nc=6, bodies_per_cell=2,
        sigma_lo=0.55, sigma_hi=0.64, seed=31027)),
)
open(joinpath(outdir, "geometry_coverage.csv"), "w") do io
    println(io, "case,cutoff_over_h,n_bodies,n_pairs,n_direct,n_m2l,n_cutoff,missing,duplicates,cutoff_in_m2l")
    for (name, g) in geometry_cases
        @printf(io, "%s,%.3f,%d,%d,%d,%d,%d,%d,%d,%d\n", name, g.cutoff_over_h,
            g.n_bodies, g.n_pairs, g.n_direct, g.n_m2l, g.n_cutoff, g.missing,
            g.duplicates, g.cutoff_in_m2l)
        @printf("geometry[%s]: rho_t*sigma_max=%.3f h, %d pairs, missing=%d duplicate=%d cutoff_in_m2l=%d\n",
            name, g.cutoff_over_h, g.n_pairs, g.missing, g.duplicates, g.cutoff_in_m2l)
    end
end

adequacy = near_set_adequacy_rows()
open(joinpath(outdir, "near_set_adequacy.csv"), "w") do io
    println(io, "stencil,n_classes,g_min_h,n,ell,beta,eps,rho_t,required_gap_h," *
        "bodies_per_leaf,required_bodies_per_leaf,compliant,n_classes_required," *
        "reg_pairs_per_target,direct_pairs_per_target,f_regularized,speedup_q15,speedup_q25," *
        "n_classes_production,direct_pairs_production,f_production," *
        "speedup_production_q15,speedup_production_q25")
    for r in adequacy
        @printf(io, "%s,%d,%.4f,%.0f,%d,%.1f,%.0e,%.3f,%.4f,%.1f,%.1f,%d,%d,%.1f,%.1f,%.4f,%.3f,%.3f,%d,%.1f,%.4f,%.3f,%.3f\n",
            r.stencil, r.n_classes, r.g_min, r.n, r.ell, r.beta, r.eps, r.rho_t,
            r.required_gap_h, r.bodies_per_leaf, r.required_bodies_per_leaf,
            r.compliant, r.n_classes_required, r.reg_pairs_per_target,
            r.direct_pairs_per_target, r.f_regularized, r.speedup_q15, r.speedup_q25,
            r.n_classes_production, r.direct_pairs_production, r.f_production,
            r.speedup_production_q15, r.speedup_production_q25)
    end
end
println("near-set adequacy (beta=2, eps=1e-3, n=1e6):")
for r in adequacy
    (r.beta == 2.0 && r.eps == 1e-3 && r.n == 1e6 && 3 <= r.ell <= 5) || continue
    @printf("  %-20s ell=%d  need gap>%.3f h (have %.3f)  bodies/leaf %.0f (need %.0f)  %s  classes %d->%d (production %d)  f=%.3f  speedup %.2f-%.2fx\n",
        r.stencil, r.ell, r.required_gap_h, r.g_min, r.bodies_per_leaf,
        r.required_bodies_per_leaf, r.compliant ? "COMPLIANT" : "NON-COMPLIANT",
        r.n_classes, r.n_classes_required, r.n_classes_production,
        r.f_production, r.speedup_production_q15, r.speedup_production_q25)
end

# --- 5.2 the two 033 cases through the experimenter's overlap parameter ------
#
# The 5.1 floor counts bodies per OCCUPIED cell; n/8^ell equals that only for a
# field that fills its box uniformly and isotropically. The depth ceiling needs
# no occupancy estimate at all -- rho_t*sigma_max < g_min*h with h = L/2^ell is
# just
#       2^ell < g_min * L_box / (rho_t * sigma_max).
#
# Case constants are transcribed from scripts/benchmark_033_common.jl. Both
# cases use the SAME overlap convention -- beta = 2 against the local mean
# spacing (V_occupied/n)^(1/3) -- so the two forms differ purely through
# occupancy:
#   cube: unit box, sigma = 2*(1/n)^(1/3); box-filling, so the uniform rule is
#         exact.
#   wake: solid cylinder of diameter D=1 and length 5D, so its bounding CUBE has
#         L = 5 while the cylinder occupies only pi/4*D^2*5D = 3.927 of the 125
#         available -- a 3.14% fill, so occupied cells hold ~32x the all-cell
#         average n/8^ell.
const FM033_N_GRID_CUBE = (1000, 3162, 10000, 31623, 100000, 316228, 1000000)
const WAKE_R, WAKE_LEN = 0.5, 5.0
const WAKE_L = WAKE_LEN                # bounding cube side = the long axis
wake_volume() = pi * WAKE_R^2 * WAKE_LEN

"Largest integer ell with 2^ell < bound (bound > 1)."
function max_level_below(bound)
    ell = floor(Int, log2(bound))
    while 2.0^ell >= bound
        ell -= 1
    end
    return ell
end

"Largest integer ell with n/8^ell > required bodies per occupied cell."
function max_level_uniform(n, required_bodies)
    ell = 0
    while n / 8.0^(ell + 1) > required_bodies
        ell += 1
    end
    return n / 8.0^ell > required_bodies ? ell : -1
end

case_rows = let rho_t = rt_j_gate, g_min = sqrt(5.0)   # theta=0.5 stencil
    rows = NamedTuple[]
    for n in FM033_N_GRID_CUBE
        sigma = 2.0 * (1.0 / n)^(1 / 3)
        push!(rows, (case="cube", n_target=float(n), n_actual=float(n),
            L=1.0, sigma=sigma, beta=2.0, spacing=(1.0 / n)^(1 / 3),
            fill=1.0, ell_geom=max_level_below(g_min * 1.0 / (rho_t * sigma)),
            ell_uniform=max_level_uniform(float(n), (rho_t * 2.0 / g_min)^3)))
    end
    Vw = wake_volume()
    for n in FM033_N_GRID_CUBE
        spacing = (Vw / n)^(1 / 3)
        sigma = 2.0 * spacing
        push!(rows, (case="wake", n_target=float(n), n_actual=float(n),
            L=WAKE_L, sigma=sigma, beta=2.0, spacing=spacing,
            fill=Vw / WAKE_L^3,
            ell_geom=max_level_below(g_min * WAKE_L / (rho_t * sigma)),
            ell_uniform=max_level_uniform(float(n), (rho_t * 2.0 / g_min)^3)))
    end
    rows
end
for r in case_rows
    # the cube fills its box, so there the uniform rule is exact
    r.case == "cube" && (r.ell_geom == r.ell_uniform ||
        error("uniform rule disagrees with the gap test on the box-filling cube: $r"))
    # a clustered field concentrates bodies into occupied cells, so the uniform
    # rule (an all-cell average) can only under-report the admissible depth
    r.case == "wake" && (r.ell_geom >= r.ell_uniform ||
        error("uniform rule exceeded the geometric ceiling on the wake: $r"))
end
open(joinpath(outdir, "case_adequacy.csv"), "w") do io
    println(io, "case,n,L_box,sigma,beta,spacing,fill_fraction," *
        "rho_t,g_min,ell_max_geometric,ell_max_uniform_rule," *
        "occupied_cells_at_ell_geom,bodies_per_occupied_leaf")
    for r in case_rows
        h = r.L / 2.0^r.ell_geom
        occupied = max(1.0, r.fill * r.L^3 / h^3)
        @printf(io, "%s,%.0f,%.4f,%.6f,%.1f,%.6f,%.4f,%.3f,%.4f,%d,%d,%.0f,%.1f\n",
            r.case, r.n_actual, r.L, r.sigma, r.beta, r.spacing, r.fill,
            rt_j_gate, sqrt(5.0), r.ell_geom, r.ell_uniform,
            occupied, r.n_actual / occupied)
    end
end
@printf("033 case adequacy (theta=0.5 stencil, eps=1e-3, rho_t=%.3f): wake fills %.2f%% of its bounding cube\n",
    rt_j_gate, 100 * case_rows[end].fill)
for r in case_rows
    (r.case == "wake" && r.n_actual in (1e3, 1e4, 1e5, 1e6)) || continue
    h = r.L / 2.0^r.ell_geom
    occupied = max(1.0, r.fill * r.L^3 / h^3)
    @printf("  wake n=%-8.0f sigma=%.5f  ell <= %d (geometric) vs %d (n/8^ell rule)  %.0f occupied cells, %.1f bodies each\n",
        r.n_actual, r.sigma, r.ell_geom, r.ell_uniform, occupied,
        r.n_actual / occupied)
end

println("rt table:")
for (e, ru, rj) in rt_rows
    @printf("  eps=%.0e  rt_U=%.3f  rt_J=%.3f\n", e, ru, rj)
end
println("wrote CSVs to $outdir")
