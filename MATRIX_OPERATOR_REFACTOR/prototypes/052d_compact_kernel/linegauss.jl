# LineGauss — closed-form velocity/gradient of a straight vortex segment with
# a Gaussian (FLOWVPM erf-blob) core. Derivation in DERIVATION.md (this dir).
# Standalone prototype: no FLOWPanel edits. FLOWPanel edge convention:
# r1 = P1 - x, r2 = P2 - x; returned velocity is per unit circulation.
module LineGauss

using StaticArrays, LinearAlgebra
export lg_velocity, lg_velocity_sing, lg_gradient, lg_W, lg_velocity_quad,
       erf_local, gfun

const SQ2OPI = sqrt(2 / pi)
const TWOOSQPI = 2 / sqrt(pi)

# double-precision erf via the cancellation-free series
# erf(x) = (2/√π) x e^{-x²} Σ_{n≥0} (2x²)^n / (2n+1)!!  (SpecialFunctions is
# not resolvable in the FLOWPanel env, so the prototype carries its own)
function erf_local(x::Float64)
    ax = abs(x)
    ax >= 6.0 && return copysign(1.0, x)   # 1 - erf(6) < 2e-17
    t = 2 * ax * ax
    term = 1.0
    s = 1.0
    n = 0
    while term > 1e-17 * s && n < 300
        n += 1
        term *= t / (2n + 1)
        s += term
    end
    return copysign(TWOOSQPI * ax * exp(-ax * ax) * s, x)
end

"Blob velocity function g(t) = erf(t/√2) - √(2/π) t e^{-t²/2}, odd in t."
function gfun(t::Float64)
    at = abs(t)
    at >= 9.3 && return copysign(1.0, t)   # deviation < 2e-18
    if at < 0.125
        # Direct subtraction loses all useful digits as t → 0.  This is
        # g(t) = √(2/π) Σ (-1)^m t^(2m+3)/(2^m m! (2m+3)).
        t2 = t * t
        term = t * t2 / 3
        s = term
        for m in 1:12
            term *= -t2 * (2m + 1) / (2m * (2m + 3))
            s += term
        end
        return SQ2OPI * s
    end
    return erf_local(t / sqrt(2)) - SQ2OPI * t * exp(-t * t / 2)
end

# axis-series helpers (DERIVATION.md §7); odd in ẑ with ψ(0) = 0
@inline function psifun(z)
    z == 0 && return 0.0
    if abs(z) < 0.125
        # ψ(z) = √(2/π)(z/3 - z³/30 + z⁵/280 - z⁷/3024 + ⋯).
        z2 = z * z
        return SQ2OPI * z * (1 / 3 - z2 / 30 + z2^2 / 280 -
                              z2^3 / 3024 + z2^4 / 38016 - z2^5 / 549120)
    end
    return SQ2OPI * (z / 2) * exp(-z * z / 2) -
           gfun(z) / (2 * z * z) + gfun(z) / 2
end
@inline chifun(z) = -1 / (2 * z * abs(z))

# Fixed threshold at the error crossover (axis-limit truncation O(ĥ²) vs
# general-branch cancellation ~eps/ĥ²; both ≤ ~2.5e-8 at ĥ² = 1e-7). The
# pre-2026-08-28 min-ẑ²-scaled form fired at physically large ĥ for long
# segments (ẑ ~ 4e3 ⇒ ĥ² < 0.16), dropping the O(ĥ²) correction (~1% @ ĥ=0.2).
@inline axis_guard(ĥ2, ẑ1, ẑ2) =
    ĥ2 < 1e-7 && ẑ1 != 0 && ẑ2 != 0

# g(R)/R³, including its finite R=0 limit.
@inline function kfun(R)
    if R < 0.125
        r2 = R * R
        return SQ2OPI * (1 / 3 - r2 / 10 + r2^2 / 56 -
                         r2^3 / 432 + r2^4 / 4224 - r2^5 / 49920)
    end
    return gfun(R) / R^3
end

# For a wholly small configuration, evaluate
# M = ∫[z2,z1] g(sqrt(z²+h²))/sqrt(z²+h²)³ dz term-by-term.
# Also return D = M + 2h² ∂M/∂(h²), the radial-gradient factor.
function small_radius_MD(z1, z2, h2)
    M = 0.0
    dM = 0.0
    coeff = 1 / 3
    for m in 0:12
        Im = 0.0
        dIm = 0.0
        for k in 0:m
            p = m - k
            dzpow = (z1^(2k + 1) - z2^(2k + 1)) / (2k + 1)
            bc = binomial(m, k)
            Im += bc * h2^p * dzpow
            p > 0 && (dIm += bc * p * h2^(p - 1) * dzpow)
        end
        M += coeff * Im
        dM += coeff * dIm
        coeff *= -(2m + 3) / (2 * (m + 1) * (2m + 5))
    end
    M *= SQ2OPI
    return M, M + 2h2 * SQ2OPI * dM
end

# When only one endpoint is in the small-radius region, split the defining
# integral at |z| = SMALL_R.  The core piece uses the convergent power series;
# the other piece uses the fixed-z axis limit, now safely away from z=0.
const SMALL_R = 0.125
@inline endpoint_split_guard(h2, z1, z2, R1, R2) =
    h2 < 1e-8 * SMALL_R^2 && min(abs(z1), abs(z2)) < SMALL_R &&
    max(R1, R2) >= SMALL_R

function endpoint_split_MD(z1, z2, h2)
    if abs(z1) < SMALL_R
        split = -SMALL_R
        Mc, Dc = small_radius_MD(z1, split, h2)
        Mf = psifun(split) - psifun(z2)
    else
        split = SMALL_R
        Mc, Dc = small_radius_MD(split, z2, h2)
        Mf = psifun(z1) - psifun(split)
    end
    return Mf + Mc, Mf + Dc
end

"""
    lg_M(ẑ1, ẑ2, ĥ2, R̂1, R̂2) -> (M, Ms)

M = N/ĥ² (LineGauss) and Ms = q̃/ĥ² (singular), σ-scaled inputs, guarded
near the axis. u = c·M/(4πσ²L); u_sing = c·Ms/(4πσ²L).
"""
function lg_M(ẑ1, ẑ2, ĥ2, R̂1, R̂2)
    if ĥ2 == 0
        M = psifun(ẑ1) - psifun(ẑ2)
        return M, Inf
    elseif max(R̂1, R̂2) < SMALL_R
        M, _ = small_radius_MD(ẑ1, ẑ2, ĥ2)
        q̃ = ẑ1 / R̂1 - ẑ2 / R̂2
        return M, q̃ / ĥ2
    elseif endpoint_split_guard(ĥ2, ẑ1, ẑ2, R̂1, R̂2)
        M, _ = endpoint_split_MD(ẑ1, ẑ2, ĥ2)
        q̃ = ẑ1 / R̂1 - ẑ2 / R̂2
        return M, q̃ / ĥ2
    elseif axis_guard(ĥ2, ẑ1, ẑ2)
        M = psifun(ẑ1) - psifun(ẑ2)
        # sign(ẑ1) - sign(ẑ2) ≠ 0 only beside the segment, where Ms ~ 2/ĥ²
        s12 = (sign(ẑ1) - sign(ẑ2)) / ĥ2   # Inf on the axis is handled by c = 0 upstream
        Ms = s12 + chifun(ẑ1) - chifun(ẑ2)
        return M, Ms
    end
    # The Gaussian terms inside the four g functions cancel exactly.  This
    # form costs four erf evaluations and one exponential and is much less
    # cancellation-prone near the core.
    G = exp(-ĥ2 / 2)
    N = ẑ1 * erf_local(R̂1 / sqrt(2)) / R̂1 -
        ẑ2 * erf_local(R̂2 / sqrt(2)) / R̂2 -
        G * (erf_local(ẑ1 / sqrt(2)) - erf_local(ẑ2 / sqrt(2)))
    q̃ = ẑ1 / R̂1 - ẑ2 / R̂2
    return N / ĥ2, q̃ / ĥ2
end

@inline function _geom(r1::SVector{3,Float64}, r2::SVector{3,Float64}, σ)
    s = r1 - r2                    # = P1 - P2 = -L t̂
    B = dot(s, s)
    L = sqrt(B)
    that = -s / L
    z1 = -dot(that, r1)            # t̂·(x - P1)
    z2 = z1 - L
    c = cross(r1, r2)              # = h L b̂
    A = dot(c, c)
    ĥ2 = A / (B * σ * σ)
    return L, that, z1 / σ, z2 / σ, c, ĥ2
end

"LineGauss velocity per unit Γ (closed form, DERIVATION.md §2–3)."
function lg_velocity(r1::SVector{3,Float64}, r2::SVector{3,Float64}, σ)
    nr1 = norm(r1)
    nr2 = norm(r2)
    (nr1 < 5eps() || nr2 < 5eps()) && return zero(r1)
    L, that, ẑ1, ẑ2, c, ĥ2 = _geom(r1, r2, σ)
    L < 5eps() && return zero(r1)
    ĥ2 == 0 && return zero(r1)     # exactly on the line: u = 0 (c = 0)
    M, _ = lg_M(ẑ1, ẑ2, ĥ2, nr1 / σ, nr2 / σ)
    return c * (M / (4π * σ * σ * L))
end

"Singular Biot–Savart segment velocity per unit Γ via the same assembly (guarded)."
function lg_velocity_sing(r1::SVector{3,Float64}, r2::SVector{3,Float64}, σ)
    nr1 = norm(r1)
    nr2 = norm(r2)
    (nr1 < 5eps() || nr2 < 5eps()) && return zero(r1)
    L, that, ẑ1, ẑ2, c, ĥ2 = _geom(r1, r2, σ)
    (L < 5eps() || ĥ2 == 0) && return zero(r1)
    _, Ms = lg_M(ẑ1, ẑ2, ĥ2, nr1 / σ, nr2 / σ)
    return c * (Ms / (4π * σ * σ * L))
end

"Modulation W = u_lg/u_sing (diagnostic)."
function lg_W(r1::SVector{3,Float64}, r2::SVector{3,Float64}, σ)
    nr1 = norm(r1)
    nr2 = norm(r2)
    L, that, ẑ1, ẑ2, c, ĥ2 = _geom(r1, r2, σ)
    M, Ms = lg_M(ẑ1, ẑ2, ĥ2, nr1 / σ, nr2 / σ)
    return M / Ms
end

@inline skewmat(t::SVector{3,Float64}) = SMatrix{3,3,Float64,9}(
    0.0, t[3], -t[2],
    -t[3], 0.0, t[1],
    t[2], -t[1], 0.0)

"LineGauss velocity gradient ∂u_i/∂x_j per unit Γ (DERIVATION.md §5)."
function lg_gradient(r1::SVector{3,Float64}, r2::SVector{3,Float64}, σ)
    nr1 = norm(r1)
    nr2 = norm(r2)
    Z = zero(SMatrix{3,3,Float64,9})
    norm(r1 - r2) < 5eps() && return Z
    L, that, ẑ1, ẑ2, c, ĥ2 = _geom(r1, r2, σ)
    L < 5eps() && return Z
    R̂1 = nr1 / σ
    R̂2 = nr2 / σ
    M, _ = lg_M(ẑ1, ẑ2, ĥ2, R̂1, R̂2)
    C = 1 / (4π * σ * σ)
    if ĥ2 == 0
        # Includes either endpoint: the regularized transverse derivative is
        # finite even though the endpoint velocity itself is zero.
        return C * M * skewmat(that)
    end
    ĥ = sqrt(ĥ2)
    # frame: h n̂ = (x - P1) - z1 t̂ = -r1 - (σ ẑ1) t̂
    hvec = -r1 - (σ * ẑ1) * that
    nh = norm(hvec)
    if nh <= 1e-10 * max(nr1, nr2)
        # transverse direction lost to projection roundoff (can even be
        # exactly zero → NaN): the assembly collapses to its axis limit
        # duθdh → uθ/h, which is the deterministic skew form
        return C * M * skewmat(that)
    end
    n̂ = hvec / nh
    b̂ = cross(that, n̂)
    k1 = kfun(R̂1)
    k2 = kfun(R̂2)
    duθdz = C * ĥ * (k1 - k2)
    if max(R̂1, R̂2) < SMALL_R
        _, radial = small_radius_MD(ẑ1, ẑ2, ĥ2)
        duθdh = C * radial
        uθ_h = C * M
    elseif endpoint_split_guard(ĥ2, ẑ1, ẑ2, R̂1, R̂2)
        _, radial = endpoint_split_MD(ẑ1, ẑ2, ĥ2)
        duθdh = C * radial
        uθ_h = C * M
    elseif axis_guard(ĥ2, ẑ1, ẑ2)
        duθdh = C * M          # bracket → 2M on the axis, so brk - M → M
        uθ_h = C * M
    else
        G = exp(-ĥ2 / 2)
        # Cancellation-reduced (1/ĥ)∂N/∂ĥ: all endpoint
        # exponentials cancel, as in the velocity numerator.
        brk = -ẑ1 * k1 + ẑ2 * k2 +
              G * (erf_local(ẑ1 / sqrt(2)) - erf_local(ẑ2 / sqrt(2)))
        duθdh = C * (brk - M)
        uθ_h = C * M
    end
    return duθdh * (b̂ * n̂') + duθdz * (b̂ * that') - uθ_h * (n̂ * b̂')
end

"Singular Biot–Savart segment gradient ∂u_i/∂x_j per unit Γ (guarded; same
cylindrical assembly as lg_gradient with g → 1, exp → 0, physical units)."
function lg_gradient_sing(r1::SVector{3,Float64}, r2::SVector{3,Float64})
    nr1 = norm(r1)
    nr2 = norm(r2)
    Z = zero(SMatrix{3,3,Float64,9})
    (nr1 < 5eps() || nr2 < 5eps()) && return Z
    s = r1 - r2
    B = dot(s, s)
    L = sqrt(B)
    L < 5eps() && return Z
    that = -s / L
    z1 = -dot(that, r1)
    z2 = z1 - L
    c = cross(r1, r2)
    h2 = dot(c, c) / B
    C = 1 / (4π)
    # Q/h² with the sign parts separated analytically (no ½-constant cancellation)
    mq = if h2 < 1e-8 * min(z1 * z1, z2 * z2) && z1 != 0 && z2 != 0
        (sign(z1) - sign(z2)) / h2 + chifun(z1) - chifun(z2)
    else
        (z1 / nr1 - z2 / nr2) / h2
    end
    if h2 < 1e-30
        isfinite(mq) || return Z    # on the segment axis itself: leave zero
        return C * mq * skewmat(that)
    end
    ĥ = sqrt(h2)
    hvec = -r1 - z1 * that
    n̂ = hvec / norm(hvec)
    b̂ = cross(that, n̂)
    duθdh = C * (-(z1 / nr1^3 - z2 / nr2^3) - mq)
    duθdz = C * ĥ * (1 / nr1^3 - 1 / nr2^3)
    return duθdh * (b̂ * n̂') + duθdz * (b̂ * that') - C * mq * (n̂ * b̂')
end

"Composite-Simpson quadrature of the blob-line convolution (validation only)."
function lg_velocity_quad(r1::SVector{3,Float64}, r2::SVector{3,Float64}, σ;
        n::Int=4001)
    s = r1 - r2
    L = norm(s)
    that = -s / L
    f(l) = begin
        d = -r1 - l * that         # x - s(l)
        nd = norm(d)
        nd < 1e-300 ? zero(r1) : gfun(nd / σ) * cross(that, d) / nd^3
    end
    h = L / (n - 1)
    acc = f(0.0) + f(L)
    for i in 1:(n - 2)
        acc += (isodd(i) ? 4.0 : 2.0) * f(i * h)
    end
    return acc * (h / 3) / (4π)
end

end # module
