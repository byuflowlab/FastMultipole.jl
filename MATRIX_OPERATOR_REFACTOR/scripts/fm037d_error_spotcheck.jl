# fm037d_error_spotcheck.jl -- 037d error-model validation (paper study support)
#
# Purpose: empirically determine the mesh spacing h (relative to the smoothing
# radius sigma) and B-spline spreading/interpolation order p required for a
# full particle-mesh (VIC-style) evaluation of the gaussianerf-regularized
# velocity field to meet the campaign gate (sampled velocity relative RMS
# <= 1e-3 against the exact regularized direct sum). The Jacobian J,
# evaluated by analytic differentiation of the B-spline interpolant of the
# U meshes (the 6-transform scheme priced in the cost model), is logged as a
# diagnostic, matching the campaign's J policy.
#
# Method (mirrors the modeled production pipeline exactly):
#   1. N_p random vector-strength particles in a unit box, plus injected
#      close pairs at r ~ sigma (the near-pair regime where mesh error peaks).
#   2. Spread point circulations Gamma_i onto an M^3 interior mesh with a
#      centered cardinal B-spline of even order p.
#   3. Hockney free-space convolution: zero-pad to (2M)^3, FFT, multiply by
#      the FFT of the tabulated mollified Biot-Savart kernel
#      K_sigma(r) = -g(|r|/sigma) r / (4 pi |r|^3),  g the gaussianerf factor,
#      with sinc^(2p) deconvolution of spread+interp, cross product in k.
#   4. Inverse FFT, interpolate U (order-p B-spline) and J (derivative
#      interpolant) at sampled particle positions; compare against the exact
#      regularized direct sum (erf via Abramowitz-Stegun 7.1.26, |err|<1.5e-7,
#      far below the 1e-4..1e-3 scale under test).
#
# Free-space handling, deconvolution, aliasing, and the near-pair regime are
# therefore all IN the measured number -- nothing is asserted analytically.
#
# stdlib only (Random, Printf); serial; runtime ~1-3 minutes; no hardware runs.
# Output: MATRIX_OPERATOR_REFACTOR/data/fm037d_error_spotcheck.csv
#
# Known conservatism: deconvolution uses the continuous sinc^p B-spline
# transform, not the exact discrete PME (Euler-spline) factor. The measured
# error flattens near ~5e-4 below h/sigma ~ 0.55, consistent with that
# systematic; production PME/NUFFT implementations (exact b(m) factors,
# Kaiser-Bessel windows) sit below this floor in the literature. The numbers
# here are therefore an ACHIEVABLE upper bound on the error, which is the
# safe direction for the 037d go/no-go cost model. Verifying the exact-factor
# variant is a follow-up for any implementation row (kept off the local
# machine per the 2026-08-14 compute directive).

using Random, Printf

# ---------------------------------------------------------------- constants
const M_INT = 32                 # interior mesh points per axis
const N_PAD = 2 * M_INT          # Hockney-doubled FFT size (radix-2)
const H = 1.0 / M_INT            # mesh spacing (unit box)
const NP_BULK = 1500             # bulk random particles
const NP_PAIRS = 150             # injected near pairs (partner at r ~ sigma)
const NT = 250                   # sampled targets (particle positions)
const SEED = 20260814
const P_ORDERS = (2, 4, 6)       # even B-spline orders (grid-centered)
# 0.40/0.475 were run for p in {4,6} only (appended rows; identical code and
# seed, probing the small-h floor). Full-list reruns regenerate everything.
const H_OVER_SIGMA = (0.55, 0.7, 0.85, 1.0, 1.2, 1.4, 0.40, 0.475)

# ---------------------------------------------------------------- erf and g
# Abramowitz & Stegun 7.1.26, |error| <= 1.5e-7 (absolute).
function erf_as(x::Float64)
    s = x < 0 ? -1.0 : 1.0
    x = abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t -
                0.284496736) * t + 0.254829592) * t * exp(-x * x)
    return s * y
end

# gaussianerf regularization factor g(rho) = erf(rho/sqrt2) - sqrt(2/pi) rho e^{-rho^2/2}.
# Direct evaluation cancels catastrophically for small rho (g ~ rho^3 while
# erf_as carries 1.5e-7 absolute error), so G(rho) = g(rho)/rho^3 is computed
# by series for rho < 0.5:
#   G = sqrt(2/pi) (1/3 - rho^2/10 + rho^4/56 - rho^6/432 + rho^8/3840 ...)
# and G'(rho) = sqrt(2/pi) (-rho/5 + rho^3/14 - rho^5/72 + rho^7/480 ...).
const SQ2PI = sqrt(2.0 / pi)
function G_gerf(rho::Float64)
    if rho < 0.5
        r2 = rho * rho
        return SQ2PI * (1 / 3 - r2 / 10 + r2^2 / 56 - r2^3 / 432 + r2^4 / 3840)
    end
    g = erf_as(rho / sqrt(2.0)) - SQ2PI * rho * exp(-rho^2 / 2)
    return g / rho^3
end
function Gp_gerf(rho::Float64)
    if rho < 0.5
        r2 = rho * rho
        return SQ2PI * (-rho / 5 + rho * r2 / 14 - rho * r2^2 / 72 + rho * r2^3 / 480)
    end
    g = erf_as(rho / sqrt(2.0)) - SQ2PI * rho * exp(-rho^2 / 2)
    gp = SQ2PI * rho^2 * exp(-rho^2 / 2)
    return (gp * rho - 3 * g) / rho^4
end
g_gerf(rho::Float64) = G_gerf(rho) * rho^3

# Mollified Biot-Savart kernel vector K_sigma(r) = -g(|r|/sigma) r/(4 pi |r|^3)
#                                              = -G(rho) r / (4 pi sigma^3).
function kvec(rx, ry, rz, sigma)
    r2 = rx * rx + ry * ry + rz * rz
    r2 < 1e-30 && return (0.0, 0.0, 0.0)
    r = sqrt(r2)
    c = -G_gerf(r / sigma) / (4pi * sigma^3)
    return (c * rx, c * ry, c * rz)
end

# ------------------------------------------------------- centered B-splines
# Cardinal centered B-spline of even order p, support (-p/2, p/2).
function bspline(p::Int, t::Float64)
    abs(t) >= p / 2 && return 0.0
    p == 1 && return abs(t) < 0.5 ? 1.0 : 0.0
    return ((t + p / 2) * bspline(p - 1, t + 0.5) +
            (p / 2 - t) * bspline(p - 1, t - 0.5)) / (p - 1)
end
dbspline(p::Int, t::Float64) = bspline(p - 1, t + 0.5) - bspline(p - 1, t - 0.5)

# ------------------------------------------------------------------- 3D FFT
# Iterative radix-2 Cooley-Tukey, in place, on a length-n=2^k vector.
function fft1!(a::Vector{ComplexF64}, invflag::Bool)
    n = length(a)
    j = 0
    for i in 0:n-2
        if i < j
            a[i+1], a[j+1] = a[j+1], a[i+1]
        end
        m = n >> 1
        while m >= 1 && j >= m
            j -= m
            m >>= 1
        end
        j += m
    end
    len = 2
    while len <= n
        ang = (invflag ? 2.0 : -2.0) * pi / len
        wl = cis(ang)
        for i in 0:len:n-1
            w = 1.0 + 0.0im
            half = len >> 1
            @inbounds for k in 0:half-1
                u = a[i+k+1]
                v = a[i+k+half+1] * w
                a[i+k+1] = u + v
                a[i+k+half+1] = u - v
                w *= wl
            end
        end
        len <<= 1
    end
    if invflag
        inv_n = 1.0 / n
        @inbounds for i in eachindex(a)
            a[i] *= inv_n
        end
    end
    return a
end

# 3D FFT by pencils along each dimension.
function fft3!(A::Array{ComplexF64,3}, invflag::Bool)
    n = size(A, 1)
    buf = Vector{ComplexF64}(undef, n)
    @inbounds for k in 1:n, j in 1:n
        for i in 1:n; buf[i] = A[i, j, k]; end
        fft1!(buf, invflag)
        for i in 1:n; A[i, j, k] = buf[i]; end
    end
    @inbounds for k in 1:n, i in 1:n
        for j in 1:n; buf[j] = A[i, j, k]; end
        fft1!(buf, invflag)
        for j in 1:n; A[i, j, k] = buf[j]; end
    end
    @inbounds for j in 1:n, i in 1:n
        for k in 1:n; buf[k] = A[i, j, k]; end
        fft1!(buf, invflag)
        for k in 1:n; A[i, j, k] = buf[k]; end
    end
    return A
end

# ------------------------------------------------------------ problem setup
rng = MersenneTwister(SEED)

# Bulk particles in [0.18, 0.82]^3 (keeps order-6 spread support inside the
# interior mesh), Gamma ~ N(0,1)^3 / N.
npart = NP_BULK + NP_PAIRS
xs = Vector{NTuple{3,Float64}}(undef, npart)
gs = Vector{NTuple{3,Float64}}(undef, npart)
for i in 1:NP_BULK
    xs[i] = (0.18 + 0.64 * rand(rng), 0.18 + 0.64 * rand(rng),
             0.18 + 0.64 * rand(rng))
    gs[i] = (randn(rng), randn(rng), randn(rng)) ./ npart
end
# Near-pair partners are attached per-sigma below (distance depends on sigma);
# here just draw hosts and directions.
pair_host = [rand(rng, 1:NP_BULK) for _ in 1:NP_PAIRS]
pair_dir = Vector{NTuple{3,Float64}}(undef, NP_PAIRS)
pair_rfac = Vector{Float64}(undef, NP_PAIRS)     # r = rfac * sigma
for i in 1:NP_PAIRS
    d = (randn(rng), randn(rng), randn(rng))
    nd = sqrt(sum(abs2, d))
    pair_dir[i] = d ./ nd
    pair_rfac[i] = 0.5 + rand(rng)               # r in [0.5, 1.5] sigma
    gs[NP_BULK+i] = (randn(rng), randn(rng), randn(rng)) ./ npart
end
targets = [rand(rng, 1:npart) for _ in 1:NT]     # sample velocities AT particles

# Exact regularized direct sums (U and analytic J at targets).
# u_a = eps_abc K_b Gamma_c with K(r) = -q(r) r, q(r) = G(rho)/(4 pi sigma^3);
# dK_b/dx_d = -(q'(r) r_b r_d / r + q delta_bd), q'(r) = G'(rho)/(4 pi sigma^4).
# The self term (i == target) contributes 0 to U and -q(0) eps_abd Gamma_... via
# the delta_bd part -- kept, since the mesh field also carries the self blob.
levi(a, b, c) = (a, b, c) in ((1, 2, 3), (2, 3, 1), (3, 1, 2)) ? 1.0 :
                (a, b, c) in ((3, 2, 1), (2, 1, 3), (1, 3, 2)) ? -1.0 : 0.0
const EPS3 = [levi(a, b, c) for a in 1:3, b in 1:3, c in 1:3]
function exact_uj(xs, gs, sigma, tlist)
    nT = length(tlist)
    U = zeros(3, nT)
    J = zeros(3, 3, nT)
    c3 = 1.0 / (4pi * sigma^3)
    c4 = 1.0 / (4pi * sigma^4)
    for (ti, t) in enumerate(tlist)
        xt = xs[t]
        for i in 1:length(xs)
            rx = xt[1] - xs[i][1]; ry = xt[2] - xs[i][2]; rz = xt[3] - xs[i][3]
            rr = (rx, ry, rz)
            r = sqrt(rx * rx + ry * ry + rz * rz)
            rho = r / sigma
            q = G_gerf(rho) * c3
            qp = Gp_gerf(rho) * c4
            G = gs[i]
            # U
            U[1, ti] += -q * (ry * G[3] - rz * G[2])
            U[2, ti] += -q * (rz * G[1] - rx * G[3])
            U[3, ti] += -q * (rx * G[2] - ry * G[1])
            # J: dU_a/dx_d = eps_abc dK_b/dx_d Gamma_c
            rinv = r > 1e-300 ? 1.0 / r : 0.0
            for a in 1:3, d in 1:3
                s = 0.0
                for b in 1:3, c in 1:3
                    e = EPS3[a, b, c]
                    e == 0.0 && continue
                    dKbd = -(qp * rr[b] * rr[d] * rinv + (b == d ? q : 0.0))
                    s += e * dKbd * G[c]
                end
                J[a, d, ti] += s
            end
        end
    end
    return U, J
end

# Tabulated free-space kernel spectra on the doubled grid (3 components).
function kernel_hat(sigma)
    Ks = [zeros(ComplexF64, N_PAD, N_PAD, N_PAD) for _ in 1:3]
    for kk in 0:N_PAD-1, jj in 0:N_PAD-1, ii in 0:N_PAD-1
        dx = (ii <= M_INT ? ii : ii - N_PAD) * H
        dy = (jj <= M_INT ? jj : jj - N_PAD) * H
        dz = (kk <= M_INT ? kk : kk - N_PAD) * H
        K = kvec(dx, dy, dz, sigma)
        Ks[1][ii+1, jj+1, kk+1] = K[1]
        Ks[2][ii+1, jj+1, kk+1] = K[2]
        Ks[3][ii+1, jj+1, kk+1] = K[3]
    end
    for c in 1:3; fft3!(Ks[c], false); end
    return Ks
end

# One full mesh evaluation at (p, sigma); returns (u_rel_rms, j_rel_rms).
function run_case(p, sigma, xs, gs, tlist, Uex, Jex, Khat)
    W = [zeros(ComplexF64, N_PAD, N_PAD, N_PAD) for _ in 1:3]
    half = p / 2
    # ---- spread Gamma with order-p B-spline (point spreading, PME-style)
    for i in 1:length(xs)
        x = xs[i]; G = gs[i]
        u1, u2, u3 = x[1] / H, x[2] / H, x[3] / H
        j1lo = ceil(Int, u1 - half); j1hi = floor(Int, u1 + half)
        j2lo = ceil(Int, u2 - half); j2hi = floor(Int, u2 + half)
        j3lo = ceil(Int, u3 - half); j3hi = floor(Int, u3 + half)
        for j3 in j3lo:j3hi
            w3 = bspline(p, u3 - j3); w3 == 0.0 && continue
            for j2 in j2lo:j2hi
                w2 = bspline(p, u2 - j2); w2 == 0.0 && continue
                w23 = w2 * w3
                for j1 in j1lo:j1hi
                    w = bspline(p, u1 - j1) * w23; w == 0.0 && continue
                    W[1][j1+1, j2+1, j3+1] += w * G[1]
                    W[2][j1+1, j2+1, j3+1] += w * G[2]
                    W[3][j1+1, j2+1, j3+1] += w * G[3]
                end
            end
        end
    end
    for c in 1:3; fft3!(W[c], false); end
    # ---- deconvolve (spread + interp: sinc^(2p)) and cross-multiply in k
    U = [zeros(ComplexF64, N_PAD, N_PAD, N_PAD) for _ in 1:3]
    sincp = Vector{Float64}(undef, N_PAD)
    for m in 0:N_PAD-1
        mt = m <= N_PAD ÷ 2 ? m : m - N_PAD
        arg = pi * mt / N_PAD
        sincp[m+1] = mt == 0 ? 1.0 : (sin(arg) / arg)^p
    end
    @inbounds for kk in 1:N_PAD, jj in 1:N_PAD, ii in 1:N_PAD
        d = (sincp[ii] * sincp[jj] * sincp[kk])^2
        w1 = W[1][ii, jj, kk] / d
        w2 = W[2][ii, jj, kk] / d
        w3 = W[3][ii, jj, kk] / d
        k1 = Khat[1][ii, jj, kk]; k2 = Khat[2][ii, jj, kk]; k3 = Khat[3][ii, jj, kk]
        U[1][ii, jj, kk] = k2 * w3 - k3 * w2
        U[2][ii, jj, kk] = k3 * w1 - k1 * w3
        U[3][ii, jj, kk] = k1 * w2 - k2 * w1
    end
    for c in 1:3; fft3!(U[c], true); end
    # ---- interpolate U and J (derivative interpolant) at targets
    err2u = 0.0; ref2u = 0.0; err2j = 0.0; ref2j = 0.0
    for (ti, t) in enumerate(tlist)
        x = xs[t]
        u1, u2, u3 = x[1] / H, x[2] / H, x[3] / H
        uu = zeros(3); jj_ = zeros(3, 3)
        j1lo = ceil(Int, u1 - half); j1hi = floor(Int, u1 + half)
        j2lo = ceil(Int, u2 - half); j2hi = floor(Int, u2 + half)
        j3lo = ceil(Int, u3 - half); j3hi = floor(Int, u3 + half)
        for j3 in j3lo:j3hi
            w3 = bspline(p, u3 - j3); dw3 = dbspline(p, u3 - j3) / H
            for j2 in j2lo:j2hi
                w2 = bspline(p, u2 - j2); dw2 = dbspline(p, u2 - j2) / H
                for j1 in j1lo:j1hi
                    w1 = bspline(p, u1 - j1); dw1 = dbspline(p, u1 - j1) / H
                    for c in 1:3
                        v = real(U[c][j1+1, j2+1, j3+1])
                        uu[c] += w1 * w2 * w3 * v
                        jj_[c, 1] += dw1 * w2 * w3 * v
                        jj_[c, 2] += w1 * dw2 * w3 * v
                        jj_[c, 3] += w1 * w2 * dw3 * v
                    end
                end
            end
        end
        for c in 1:3
            err2u += (uu[c] - Uex[c, ti])^2
            ref2u += Uex[c, ti]^2
            for b in 1:3
                err2j += (jj_[c, b] - Jex[c, b, ti])^2
                ref2j += Jex[c, b, ti]^2
            end
        end
    end
    return sqrt(err2u / ref2u), sqrt(err2j / ref2j)
end

# ------------------------------------------------------------------- driver
outpath = joinpath(@__DIR__, "..", "data", "fm037d_error_spotcheck.csv")
open(outpath, "w") do io
    println(io, "p,h_over_sigma,sigma,mesh_interior,mesh_padded,n_particles," *
                "n_targets,u_rel_rms,j_rel_rms")
    for ratio in H_OVER_SIGMA
        sigma = H / ratio
        # attach near-pair partners at r = rfac * sigma
        for i in 1:NP_PAIRS
            hx = xs[pair_host[i]]
            r = pair_rfac[i] * sigma
            xp = (clamp(hx[1] + r * pair_dir[i][1], 0.18, 0.82),
                  clamp(hx[2] + r * pair_dir[i][2], 0.18, 0.82),
                  clamp(hx[3] + r * pair_dir[i][3], 0.18, 0.82))
            xs[NP_BULK+i] = xp
        end
        t0 = time()
        Uex, Jex = exact_uj(xs, gs, sigma, targets)
        Khat = kernel_hat(sigma)
        @printf(stderr, "sigma=%.5f (h/sigma=%.2f): refs+kernel %.1fs\n",
                sigma, ratio, time() - t0)
        for p in P_ORDERS
            t1 = time()
            eu, ej = run_case(p, sigma, xs, gs, targets, Uex, Jex, Khat)
            @printf(stderr, "  p=%d: u_rel_rms=%.3e j_rel_rms=%.3e (%.1fs)\n",
                    p, eu, ej, time() - t1)
            @printf(io, "%d,%.4f,%.6f,%d,%d,%d,%d,%.6e,%.6e\n",
                    p, ratio, sigma, M_INT, N_PAD, npart, NT, eu, ej)
        end
    end
end
println("wrote $outpath")
