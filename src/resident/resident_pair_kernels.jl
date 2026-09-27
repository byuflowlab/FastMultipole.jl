#------- gaussianerf g/h evaluation and direct-kernel pair functors (032 stage 2) -------#
#
# Erf-free evaluation of the gaussianerf regularization factor
#   g(ρ) = erf(ρ/√2) − Aρe^{−ρ²/2},  A = √(2/π),
# and the combined Jacobian numerator h(ρ) = ρg′(ρ) − 3g(ρ), used by the
# `RegularizedVortex` nearfield (`theory/kernel-splitting-nearfield.md` §3, §6.2).
#
# Below ρ = 2: the cancellation-safe alternating Horner series
#   g = Aρ³ Σ_k (−1)^k ρ^{2k}/((2k+3) 2^k k!),  h = Aρ⁵ Σ_k (−1)^{k+1} ρ^{2k}/((2k+5) 2^k k!),
# with term counts measured against a 256-bit reference over the whole branch
# (scripts/fit_032_nearfield_g.jl → data/kernel_splitting/nearfield_g_eval.csv):
# 13 terms hold ≤ 6.8e-7 relative in Float32 and 19 terms ≤ 1.7e-12 in Float64
# (the theory-§3 6/10-term counts are valid only to its ρ = 0.5 partitioning
# switch; the erf-free design runs the series to ρ = 2, hence the re-measured
# counts). Above ρ = 2: ḡ = e^{−ρ²/2}(Aρ + s), with s ≈ degree-3 polynomial in
# u = 1/ρ² least-squares fitted on [2, 4.789]; g = 1 − ḡ holds ≤ 2.1e-4 absolute
# against the 3.69e-4 budget (031a §6.2: absolute tolerance suffices where the
# retained result is O(1)), decaying like e^{−ρ²/2} beyond ρ_t, with the correct
# singular limit (g→1, h→−3). One hardware exp, no erf, both branches GPU-safe.
const _GAUSSERF_A = 0.7978845608028654           # √(2/π)
const _GAUSSERF_G_COEFFS = Tuple(Float64((isodd(k) ? -1 : 1) //
    ((2k + 3) * BigInt(2)^k * factorial(BigInt(k)))) for k in 0:18)
const _GAUSSERF_H_COEFFS = Tuple(Float64((iseven(k) ? -1 : 1) //
    ((2k + 5) * BigInt(2)^k * factorial(BigInt(k)))) for k in 0:18)
const _GAUSSERF_G_COEFFS32 = Tuple(Float32(c) for c in _GAUSSERF_G_COEFFS[1:13])
const _GAUSSERF_H_COEFFS32 = Tuple(Float32(c) for c in _GAUSSERF_H_COEFFS[1:13])
# degree-3 fit of s(u) on [ρ_c = 2, ρ_t = 4.789] (fit_032_nearfield_g.jl)
const _GAUSSERF_S_COEFFS = (0.082593826443677007, 2.0801015208954681,
    -6.8585923004500211, 10.482822830062796)

@inline _gausserf_series_g(z::Float64) = evalpoly(z, _GAUSSERF_G_COEFFS)
@inline _gausserf_series_h(z::Float64) = evalpoly(z, _GAUSSERF_H_COEFFS)
@inline _gausserf_series_g(z::Float32) = evalpoly(z, _GAUSSERF_G_COEFFS32)
@inline _gausserf_series_h(z::Float32) = evalpoly(z, _GAUSSERF_H_COEFFS32)

# Outer-branch (ρ > 2) complement pair: ḡ = e^{−ρ²/2}(Aρ + s(1/ρ²)) with the
# 031a §6.2 degree-3 fit of s, and ρg′ = Aρ³e^{−ρ²/2} reusing the same
# exponential. Factored out of `_gaussianerf_g_h` so the two-pass deficit
# () consumes ḡ directly to absolute tolerance instead of
# reconstructing it as 1 − g.
@inline function _gaussianerf_gbar_rhogp(rho::T) where T<:AbstractFloat
    z = rho * rho
    e = exp(-z / 2)
    u = inv(z)
    s = muladd(muladd(muladd(T(_GAUSSERF_S_COEFFS[4]), u, T(_GAUSSERF_S_COEFFS[3])),
        u, T(_GAUSSERF_S_COEFFS[2])), u, T(_GAUSSERF_S_COEFFS[1]))
    gbar = e * muladd(T(_GAUSSERF_A), rho, s)
    rhogp = T(_GAUSSERF_A) * rho * z * e
    return gbar, rhogp
end

@inline function _gaussianerf_g_h(rho::T) where T<:AbstractFloat
    z = rho * rho
    if rho <= T(2)
        g = T(_GAUSSERF_A) * rho * z * _gausserf_series_g(z)
        h = T(_GAUSSERF_A) * rho * z * z * _gausserf_series_h(z)
        return g, h
    end
    gbar, rhogp = _gaussianerf_gbar_rhogp(rho)
    g = one(T) - gbar
    h = rhogp - 3 * g
    return g, h
end

# Per-pair math behind the `direct_kernel` functor trait. Shared verbatim by the
# host loops and the CUDA pair kernels (arithmetic + exp only); the caller
# computes `invr` with its preferred reciprocal sqrt and skips r2 == 0 pairs.
# Conventions (theory §1 / legacy direct!): dx,dy,dz = target − source;
# crss_i = −(Δx×Γ)_i/(4πr³); U = g·crss; J[i,j] = ∂u_i/∂x_j column-major with
# a = h/r², b = −g/(4πr³).

@inline function _direct_pair_ug(::SingularSource, dx, dy, dz, r2, invr,
        source_bodies, j)
    T = typeof(r2)
    @inbounds q = source_bodies[5, j] * inv(T(4) * T(π))
    u = q * invr
    invr3 = invr * invr * invr
    return u, -q * dx * invr3, -q * dy * invr3, -q * dz * invr3
end

@inline function _direct_pair_ugh(::SingularSource, dx, dy, dz, r2, invr,
        source_bodies, j)
    T = typeof(r2)
    @inbounds q = source_bodies[5, j] * inv(T(4) * T(π))
    u = q * invr
    invr2 = invr * invr
    invr3 = invr * invr2
    q3invr5 = 3 * q * invr3 * invr2
    qinvr3 = q * invr3
    return (u, -q * dx * invr3, -q * dy * invr3, -q * dz * invr3,
        q3invr5 * dx * dx - qinvr3, q3invr5 * dx * dy, q3invr5 * dx * dz,
        q3invr5 * dy * dx, q3invr5 * dy * dy - qinvr3, q3invr5 * dy * dz,
        q3invr5 * dz * dx, q3invr5 * dz * dy, q3invr5 * dz * dz - qinvr3)
end

# Point dipole: u = p·d / (4π r³) with d = target − source, the source-position
# derivative of the SingularSource potential; g = ∇u, h = ∇∇u.
@inline function _direct_pair_ug(::SingularDipole, dx, dy, dz, r2, invr,
        source_bodies, j)
    T = typeof(r2)
    c = inv(T(4) * T(π))
    @inbounds px = source_bodies[5, j] * c
    @inbounds py = source_bodies[6, j] * c
    @inbounds pz = source_bodies[7, j] * c
    invr2 = invr * invr
    invr3 = invr * invr2
    pd = px * dx + py * dy + pz * dz
    u = pd * invr3
    t = 3 * pd * invr3 * invr2
    return u, px * invr3 - t * dx, py * invr3 - t * dy, pz * invr3 - t * dz
end

@inline function _direct_pair_ugh(::SingularDipole, dx, dy, dz, r2, invr,
        source_bodies, j)
    T = typeof(r2)
    c = inv(T(4) * T(π))
    @inbounds px = source_bodies[5, j] * c
    @inbounds py = source_bodies[6, j] * c
    @inbounds pz = source_bodies[7, j] * c
    invr2 = invr * invr
    invr3 = invr * invr2
    invr5 = invr3 * invr2
    pd = px * dx + py * dy + pz * dz
    u = pd * invr3
    t = 3 * pd * invr5
    gx = px * invr3 - t * dx
    gy = py * invr3 - t * dy
    gz = pz * invr3 - t * dz
    # ∂_j g_i = −3 (p_i d_j + p_j d_i) / r⁵ − 3 (p·d) δ_ij / r⁵ + 15 (p·d) d_i d_j / r⁷
    a = 3 * invr5
    b = 15 * pd * invr5 * invr2
    return (u, gx, gy, gz,
        -a * (2 * px * dx + pd) + b * dx * dx, -a * (px * dy + py * dx) + b * dx * dy, -a * (px * dz + pz * dx) + b * dx * dz,
        -a * (py * dx + px * dy) + b * dy * dx, -a * (2 * py * dy + pd) + b * dy * dy, -a * (py * dz + pz * dy) + b * dy * dz,
        -a * (pz * dx + px * dz) + b * dz * dx, -a * (pz * dy + py * dz) + b * dz * dy, -a * (2 * pz * dz + pd) + b * dz * dz)
end

#------- straight filaments -------#
#
# Line-source derivatives. With r1, r2 the distances from the target to the
# endpoints, S = r1 + r2, L the length, A = S + L, B = S - L, the unit-strength
# potential is u = c ln(A/B), c = 1/(4π). Its derivatives follow the chain
# through S: f1 = 1/A - 1/B, f2 = -1/A^2 + 1/B^2, f3 = 2/A^3 - 2/B^3, with
# grad S = e1 + e2 (unit vectors to the endpoints), hess S = sum (I - e e')/r and
# the third derivative sum (3 e_i e_j e_k - d_ij e_k - d_ik e_j - d_jk e_i)/r^2.
# The dipole filament is -p . grad of this, so its hessian needs the third order.
@inline function _line_source_setup(tx, ty, tz, source_bodies, j, v1row)
    @inbounds begin
        ax = tx - source_bodies[v1row, j]; ay = ty - source_bodies[v1row + 1, j]; az = tz - source_bodies[v1row + 2, j]
        bx = tx - source_bodies[v1row + 3, j]; by = ty - source_bodies[v1row + 4, j]; bz = tz - source_bodies[v1row + 5, j]
        Lx = source_bodies[v1row + 3, j] - source_bodies[v1row, j]
        Ly = source_bodies[v1row + 4, j] - source_bodies[v1row + 1, j]
        Lz = source_bodies[v1row + 5, j] - source_bodies[v1row + 2, j]
    end
    r1 = sqrt(ax * ax + ay * ay + az * az)
    r2 = sqrt(bx * bx + by * by + bz * bz)
    L = sqrt(Lx * Lx + Ly * Ly + Lz * Lz)
    return ax, ay, az, bx, by, bz, r1, r2, L
end

# B = S - L without cancellation: (S-L)(S+L) = 2(r1 r2 + a.b), and near the
# segment (a.b < 0) r1 r2 + a.b = |a x b|^2 / (r1 r2 - a.b)
@inline function _line_source_B(ax, ay, az, bx, by, bz, r1, r2, A)
    p = r1 * r2
    d = ax * bx + ay * by + az * bz
    if d < zero(d)
        cx = ay * bz - az * by; cy = az * bx - ax * bz; cz = ax * by - ay * bx
        s = (cx * cx + cy * cy + cz * cz) / (p - d)
    else
        s = p + d
    end
    return 2 * s / A
end

# potential, gradient and hessian of the unit line source (13 values)
@inline function _line_source_ugh(ax, ay, az, bx, by, bz, r1, r2, L)
    T = typeof(ax)
    c = inv(T(4) * T(π))
    S = r1 + r2
    A = S + L
    A <= zero(T) && return ntuple(_ -> zero(T), Val(13))     # collapsed segment at the target
    B = _line_source_B(ax, ay, az, bx, by, bz, r1, r2, A)
    B <= zero(T) && return ntuple(_ -> zero(T), Val(13))     # exactly on the segment: no finite value
    u = c * log(A / B)
    f1 = inv(A) - inv(B)
    f2 = -inv(A * A) + inv(B * B)
    i1 = inv(r1); i2 = inv(r2)
    e1x = ax * i1; e1y = ay * i1; e1z = az * i1
    e2x = bx * i2; e2y = by * i2; e2z = bz * i2
    Sx = e1x + e2x; Sy = e1y + e2y; Sz = e1z + e2z
    gx = c * f1 * Sx; gy = c * f1 * Sy; gz = c * f1 * Sz
    Sxx = (1 - e1x * e1x) * i1 + (1 - e2x * e2x) * i2
    Syy = (1 - e1y * e1y) * i1 + (1 - e2y * e2y) * i2
    Szz = (1 - e1z * e1z) * i1 + (1 - e2z * e2z) * i2
    Sxy = -(e1x * e1y) * i1 - (e2x * e2y) * i2
    Sxz = -(e1x * e1z) * i1 - (e2x * e2z) * i2
    Syz = -(e1y * e1z) * i1 - (e2y * e2z) * i2
    hxx = c * (f2 * Sx * Sx + f1 * Sxx); hyy = c * (f2 * Sy * Sy + f1 * Syy); hzz = c * (f2 * Sz * Sz + f1 * Szz)
    hxy = c * (f2 * Sx * Sy + f1 * Sxy); hxz = c * (f2 * Sx * Sz + f1 * Sxz); hyz = c * (f2 * Sy * Sz + f1 * Syz)
    return (u, gx, gy, gz, hxx, hxy, hxz, hxy, hyy, hyz, hxz, hyz, hzz)
end

@inline function _direct_pair_ug(::SourceFilamentKernel, dx, dy, dz, r2_, invr,
        source_bodies, j)
    @inbounds tx = source_bodies[1, j] + dx; ty = source_bodies[2, j] + dy; tz = source_bodies[3, j] + dz
    @inbounds q = source_bodies[5, j]
    v = _line_source_ugh(_line_source_setup(tx, ty, tz, source_bodies, j, 6)...)
    return q * v[1], q * v[2], q * v[3], q * v[4]
end

@inline function _direct_pair_ugh(::SourceFilamentKernel, dx, dy, dz, r2_, invr,
        source_bodies, j)
    @inbounds tx = source_bodies[1, j] + dx; ty = source_bodies[2, j] + dy; tz = source_bodies[3, j] + dz
    @inbounds q = source_bodies[5, j]
    v = _line_source_ugh(_line_source_setup(tx, ty, tz, source_bodies, j, 6)...)
    return ntuple(i -> q * v[i], Val(13))
end

# the dipole filament: u = -p . grad(u_s), g_j = -p_i H_ij, h_jk = -p_i T_ijk
# (no closures: a kernel must not capture a reassigned accumulator)
@inline _ls_delta(i, k, ::Type{T}) where T = i == k ? one(T) : zero(T)
@inline _ls_e(e1, e2, i, w1, w2) = e1[i] * w1 + e2[i] * w2
# S_ij and S_ijk summed over the two endpoints
@inline function _ls_Sij(e1, e2, i1, i2, i, k)
    T = typeof(i1)
    d = _ls_delta(i, k, T)
    return (d - e1[i] * e1[k]) * i1 + (d - e2[i] * e2[k]) * i2
end
@inline function _ls_Sijk(e1, e2, i1, i2, i, k, l)
    T = typeof(i1)
    dik = _ls_delta(i, k, T); dil = _ls_delta(i, l, T); dkl = _ls_delta(k, l, T)
    return (3 * e1[i] * e1[k] * e1[l] - dik * e1[l] - dil * e1[k] - dkl * e1[i]) * i1 * i1 +
           (3 * e2[i] * e2[k] * e2[l] - dik * e2[l] - dil * e2[k] - dkl * e2[i]) * i2 * i2
end
@inline function _line_source_dipole(ax, ay, az, bx, by, bz, r1, r2, L, px, py, pz)
    T = typeof(ax)
    c = inv(T(4) * T(π))
    S = r1 + r2
    A = S + L
    A <= zero(T) && return ntuple(_ -> zero(T), Val(13))       # collapsed segment at the target
    B = _line_source_B(ax, ay, az, bx, by, bz, r1, r2, A)
    B <= zero(T) && return ntuple(_ -> zero(T), Val(13))       # exactly on the segment
    f1 = inv(A) - inv(B)
    f2 = -inv(A * A) + inv(B * B)
    f3 = 2 * inv(A * A * A) - 2 * inv(B * B * B)
    i1 = inv(r1); i2 = inv(r2)
    e1 = SVector{3,T}(ax * i1, ay * i1, az * i1)
    e2 = SVector{3,T}(bx * i2, by * i2, bz * i2)
    Sg = e1 + e2
    pv = SVector{3,T}(px, py, pz)
    pS = pv[1] * Sg[1] + pv[2] * Sg[2] + pv[3] * Sg[3]
    u = -c * f1 * pS
    # g_j = -c sum_i p_i (f2 S_i S_j + f1 S_ij)
    g1 = zero(T); g2 = zero(T); g3 = zero(T)
    @inbounds for i in 1:3
        g1 += pv[i] * (f2 * Sg[i] * Sg[1] + f1 * _ls_Sij(e1, e2, i1, i2, i, 1))
        g2 += pv[i] * (f2 * Sg[i] * Sg[2] + f1 * _ls_Sij(e1, e2, i1, i2, i, 2))
        g3 += pv[i] * (f2 * Sg[i] * Sg[3] + f1 * _ls_Sij(e1, e2, i1, i2, i, 3))
    end
    g1 *= -c; g2 *= -c; g3 *= -c
    # h_jk = -c sum_i p_i (f3 S_i S_j S_k + f2 (S_ik S_j + S_i S_jk + S_ij S_k) + f1 S_ijk), column-major (j fast)
    h11 = zero(T); h21 = zero(T); h31 = zero(T)
    h12 = zero(T); h22 = zero(T); h32 = zero(T)
    h13 = zero(T); h23 = zero(T); h33 = zero(T)
    @inbounds for i in 1:3
        p = pv[i]; Si = Sg[i]
        Si1 = _ls_Sij(e1, e2, i1, i2, i, 1); Si2 = _ls_Sij(e1, e2, i1, i2, i, 2); Si3 = _ls_Sij(e1, e2, i1, i2, i, 3)
        S11 = _ls_Sij(e1, e2, i1, i2, 1, 1); S12 = _ls_Sij(e1, e2, i1, i2, 1, 2); S13 = _ls_Sij(e1, e2, i1, i2, 1, 3)
        S22 = _ls_Sij(e1, e2, i1, i2, 2, 2); S23 = _ls_Sij(e1, e2, i1, i2, 2, 3); S33 = _ls_Sij(e1, e2, i1, i2, 3, 3)
        h11 += p * (f3 * Si * Sg[1] * Sg[1] + f2 * (Si1 * Sg[1] + Si * S11 + Si1 * Sg[1]) + f1 * _ls_Sijk(e1, e2, i1, i2, i, 1, 1))
        h21 += p * (f3 * Si * Sg[2] * Sg[1] + f2 * (Si1 * Sg[2] + Si * S12 + Si2 * Sg[1]) + f1 * _ls_Sijk(e1, e2, i1, i2, i, 2, 1))
        h31 += p * (f3 * Si * Sg[3] * Sg[1] + f2 * (Si1 * Sg[3] + Si * S13 + Si3 * Sg[1]) + f1 * _ls_Sijk(e1, e2, i1, i2, i, 3, 1))
        h12 += p * (f3 * Si * Sg[1] * Sg[2] + f2 * (Si2 * Sg[1] + Si * S12 + Si1 * Sg[2]) + f1 * _ls_Sijk(e1, e2, i1, i2, i, 1, 2))
        h22 += p * (f3 * Si * Sg[2] * Sg[2] + f2 * (Si2 * Sg[2] + Si * S22 + Si2 * Sg[2]) + f1 * _ls_Sijk(e1, e2, i1, i2, i, 2, 2))
        h32 += p * (f3 * Si * Sg[3] * Sg[2] + f2 * (Si2 * Sg[3] + Si * S23 + Si3 * Sg[2]) + f1 * _ls_Sijk(e1, e2, i1, i2, i, 3, 2))
        h13 += p * (f3 * Si * Sg[1] * Sg[3] + f2 * (Si3 * Sg[1] + Si * S13 + Si1 * Sg[3]) + f1 * _ls_Sijk(e1, e2, i1, i2, i, 1, 3))
        h23 += p * (f3 * Si * Sg[2] * Sg[3] + f2 * (Si3 * Sg[2] + Si * S23 + Si2 * Sg[3]) + f1 * _ls_Sijk(e1, e2, i1, i2, i, 2, 3))
        h33 += p * (f3 * Si * Sg[3] * Sg[3] + f2 * (Si3 * Sg[3] + Si * S33 + Si3 * Sg[3]) + f1 * _ls_Sijk(e1, e2, i1, i2, i, 3, 3))
    end
    return (u, g1, g2, g3, -c * h11, -c * h21, -c * h31, -c * h12, -c * h22, -c * h32, -c * h13, -c * h23, -c * h33)
end

@inline function _direct_pair_ug(::DipoleFilamentKernel, dx, dy, dz, r2_, invr,
        source_bodies, j)
    @inbounds tx = source_bodies[1, j] + dx; ty = source_bodies[2, j] + dy; tz = source_bodies[3, j] + dz
    @inbounds px = source_bodies[5, j]; py = source_bodies[6, j]; pz = source_bodies[7, j]
    v = _line_source_dipole(_line_source_setup(tx, ty, tz, source_bodies, j, 8)..., px, py, pz)
    return v[1], v[2], v[3], v[4]
end

@inline function _direct_pair_ugh(::DipoleFilamentKernel, dx, dy, dz, r2_, invr,
        source_bodies, j)
    @inbounds tx = source_bodies[1, j] + dx; ty = source_bodies[2, j] + dy; tz = source_bodies[3, j] + dz
    @inbounds px = source_bodies[5, j]; py = source_bodies[6, j]; pz = source_bodies[7, j]
    return _line_source_dipole(_line_source_setup(tx, ty, tz, source_bodies, j, 8)..., px, py, pz)
end

# vortex filament: the bound-vortex functions of direct_rectangular.jl, with
# r1 = x1 - target, r2 = x2 - target and the circulation Γ . d/|d| (Γ points
# along the segment; a reversed Γ flips the sign)
@inline function _vortex_filament_pair(kernel::VortexFilamentKernel, dx, dy, dz, source_bodies, j, ::Val{GRAD}) where GRAD
    T = typeof(dx)
    @inbounds begin
        tx = source_bodies[1, j] + dx; ty = source_bodies[2, j] + dy; tz = source_bodies[3, j] + dz
        gx = source_bodies[5, j]; gy = source_bodies[6, j]; gz = source_bodies[7, j]
        x1 = SVector{3,T}(source_bodies[8, j], source_bodies[9, j], source_bodies[10, j])
        x2 = SVector{3,T}(source_bodies[11, j], source_bodies[12, j], source_bodies[13, j])
        core = kernel.core_row == 0 ? zero(T) : T(source_bodies[kernel.core_row, j])
    end
    d = x2 - x1
    Ld = sqrt(d[1] * d[1] + d[2] * d[2] + d[3] * d[3])
    Ld == zero(T) && return zero(SVector{3,T}), zero(SMatrix{3,3,T,9})   # collapsed segment (0 * NaN otherwise)
    gamma = (gx * d[1] + gy * d[2] + gz * d[3]) / Ld
    target = SVector{3,T}(tx, ty, tz)
    r1 = x1 - target; r2 = x2 - target
    # singular core: a target on the segment's line (its own midpoint, a shared
    # vertex, a colinear neighbour) has no finite value; return nothing there.
    # A regularized core is finite on the line (the gradient is not zero), so
    # the guard applies only for core == 0.
    cr = (r1[2] * r2[3] - r1[3] * r2[2])^2 + (r1[3] * r2[1] - r1[1] * r2[3])^2 + (r1[1] * r2[2] - r1[2] * r2[1])^2
    n1 = r1[1]^2 + r1[2]^2 + r1[3]^2; n2 = r2[1]^2 + r2[2]^2 + r2[3]^2
    core == zero(T) && cr <= T(1e-24) * n1 * n2 && return zero(SVector{3,T}), zero(SMatrix{3,3,T,9})
    # the Gaussian family divides by core^2; with no core it is the singular kernel
    fam = core == zero(T) ? 1 : kernel.family
    if fam == 2
        u = _rect_bound_vortex_velocity(r1, r2, core, Val(2))
        g = GRAD ? _rect_bound_vortex_gradient(r1, r2, core, Val(2)) : zero(SMatrix{3,3,T,9})
    elseif fam == 3
        u = _rect_bound_vortex_velocity(r1, r2, core, Val(3))
        g = GRAD ? _rect_bound_vortex_gradient(r1, r2, core, Val(3)) : zero(SMatrix{3,3,T,9})
    else
        u = _rect_bound_vortex_velocity(r1, r2, core, Val(1))
        g = GRAD ? _rect_bound_vortex_gradient(r1, r2, core, Val(1)) : zero(SMatrix{3,3,T,9})
    end
    return gamma * u, gamma * g
end

@inline function _direct_pair_ug(kernel::VortexFilamentKernel, dx, dy, dz, r2_, invr,
        source_bodies, j)
    u, _ = _vortex_filament_pair(kernel, dx, dy, dz, source_bodies, j, Val(false))
    return zero(typeof(dx)), u[1], u[2], u[3]
end

@inline function _direct_pair_ugh(kernel::VortexFilamentKernel, dx, dy, dz, r2_, invr,
        source_bodies, j)
    u, g = _vortex_filament_pair(kernel, dx, dy, dz, source_bodies, j, Val(true))
    T = typeof(dx)
    # column-major J[i,j] = du_i/dx_j, as the vortex point kernels return it
    return (zero(T), u[1], u[2], u[3],
        g[1, 1], g[2, 1], g[3, 1], g[1, 2], g[2, 2], g[3, 2], g[1, 3], g[2, 3], g[3, 3])
end

#------- planar triangular panels -------#
#
# Source and dipole panels use the closed forms of direct_rectangular.jl
# (`_rect_panel_pair` with `Val(true)` for the potential: velocity, gradient and
# potential from one pass over the edges, with the self-pair limits). Those follow FLOWPanel's sign
# convention, in which a source panel's potential is −σ/(4π) ∫ dA/r; the
# resident lifecycle's sources are +q/(4π r), so the panel results are negated
# to be the area integrals of the point kernels.
@inline function _panel_vertices(source_bodies, j, v1row, ::Type{T}) where T
    @inbounds begin
        v1 = SVector{3,T}(source_bodies[v1row, j], source_bodies[v1row + 1, j], source_bodies[v1row + 2, j])
        v2 = SVector{3,T}(source_bodies[v1row + 3, j], source_bodies[v1row + 4, j], source_bodies[v1row + 5, j])
        v3 = SVector{3,T}(source_bodies[v1row + 6, j], source_bodies[v1row + 7, j], source_bodies[v1row + 8, j])
    end
    return v1, v2, v3
end

@inline function _panel_pair(tag::Int, s1, s2, dx, dy, dz, source_bodies, j, v1row, ::Val{GRAD}) where GRAD
    T = typeof(dx)
    @inbounds target = SVector{3,T}(source_bodies[1, j] + dx, source_bodies[2, j] + dy, source_bodies[3, j] + dz)
    v1, v2, v3 = _panel_vertices(source_bodies, j, v1row, T)
    u, g, p = _rect_panel_pair(RectangularPanelInfluence(), target, tag, 3, v1, v2, v3, v3,
        T(s1), T(s2), zero(T), Val(GRAD), Val(1), Val(true))
    return -p, -u, -g
end

for (K, tag, v1row, srow, drow) in ((:SourcePanelKernel, 1, 6, 5, 5), (:DipolePanelKernel, 2, 6, 5, 5),
                                     (:SourceDipolePanelKernel, 5, 7, 5, 6))
    @eval begin
        @inline function _direct_pair_ug(::$K, dx, dy, dz, r2_, invr, source_bodies, j)
            @inbounds s1 = source_bodies[$srow, j]; s2 = source_bodies[$drow, j]
            p, u, _ = _panel_pair($tag, s1, s2, dx, dy, dz, source_bodies, j, $v1row, Val(false))
            return p, u[1], u[2], u[3]
        end
        @inline function _direct_pair_ugh(::$K, dx, dy, dz, r2_, invr, source_bodies, j)
            @inbounds s1 = source_bodies[$srow, j]; s2 = source_bodies[$drow, j]
            p, u, g = _panel_pair($tag, s1, s2, dx, dy, dz, source_bodies, j, $v1row, Val(true))
            return (p, u[1], u[2], u[3], g[1, 1], g[2, 1], g[3, 1], g[1, 2], g[2, 2], g[3, 2], g[1, 3], g[2, 3], g[3, 3])
        end
    end
end

# Dunavant rules on the reference triangle (barycentric points, weights summing to 1)
# (typed on T so no Float64 constant reaches a Float32 device kernel)
@inline function _dunavant(::Val{1}, ::Type{T}) where T
    return ((SVector{3,T}(1/3, 1/3, 1/3), one(T)),)
end
@inline function _dunavant(::Val{2}, ::Type{T}) where T
    a = T(0.797426985353087); b = T(0.101286507323456); c = T(0.470142064105115); d = T(0.059715871789770)
    wa = T(0.125939180544827); wc = T(0.132394152788506)
    return ((SVector{3,T}(1/3, 1/3, 1/3), T(0.225)),
            (SVector{3,T}(a, b, b), wa), (SVector{3,T}(b, a, b), wa), (SVector{3,T}(b, b, a), wa),
            (SVector{3,T}(c, c, d), wc), (SVector{3,T}(c, d, c), wc), (SVector{3,T}(d, c, c), wc))
end
# degree 7, 13 points (Dunavant 1985 table 7; one negative weight)
@inline function _dunavant(::Val{3}, ::Type{T}) where T
    a1 = T(0.479308067841923); b1 = T(0.260345966079038)
    a2 = T(0.869739794195568); b2 = T(0.065130102902216)
    p = T(0.638444188569809); q = T(0.312865496004875); r = T(0.048690315425316)
    w0 = T(-0.149570044467670); w1 = T(0.175615257433204); w2 = T(0.053347235608839); w3 = T(0.077113760890257)
    return ((SVector{3,T}(1/3, 1/3, 1/3), w0),
            (SVector{3,T}(a1, b1, b1), w1), (SVector{3,T}(b1, a1, b1), w1), (SVector{3,T}(b1, b1, a1), w1),
            (SVector{3,T}(a2, b2, b2), w2), (SVector{3,T}(b2, a2, b2), w2), (SVector{3,T}(b2, b2, a2), w2),
            (SVector{3,T}(p, q, r), w3), (SVector{3,T}(p, r, q), w3), (SVector{3,T}(q, p, r), w3),
            (SVector{3,T}(q, r, p), w3), (SVector{3,T}(r, p, q), w3), (SVector{3,T}(r, q, p), w3))
end

# uniform vortex sheet: Biot-Savart of γ over the triangle by quadrature
@inline function _vortex_sheet_pair(kernel::VortexSheetPanelKernel, dx, dy, dz, source_bodies, j, ::Val{GRAD}) where GRAD
    T = typeof(dx)
    @inbounds begin
        target = SVector{3,T}(source_bodies[1, j] + dx, source_bodies[2, j] + dy, source_bodies[3, j] + dz)
        gx = source_bodies[5, j]; gy = source_bodies[6, j]; gz = source_bodies[7, j]
    end
    v1, v2, v3 = _panel_vertices(source_bodies, j, 8, T)
    e1 = v2 - v1; e2 = v3 - v1
    nx = e1[2] * e2[3] - e1[3] * e2[2]; ny = e1[3] * e2[1] - e1[1] * e2[3]; nz = e1[1] * e2[2] - e1[2] * e2[1]
    area = T(0.5) * sqrt(nx * nx + ny * ny + nz * nz)
    # one rule type per branch: a runtime-selected tuple would be a union on the device
    if kernel.order >= 3
        return _vortex_sheet_quadrature(_dunavant(Val(3), T), target, v1, v2, v3, area, gx, gy, gz, Val(GRAD))
    elseif kernel.order >= 2
        return _vortex_sheet_quadrature(_dunavant(Val(2), T), target, v1, v2, v3, area, gx, gy, gz, Val(GRAD))
    else
        return _vortex_sheet_quadrature(_dunavant(Val(1), T), target, v1, v2, v3, area, gx, gy, gz, Val(GRAD))
    end
end

@inline function _vortex_sheet_quadrature(rule, target, v1, v2, v3, area::T, gx, gy, gz, ::Val{GRAD}) where {T,GRAD}
    ux = zero(T); uy = zero(T); uz = zero(T)
    j11 = zero(T); j21 = zero(T); j31 = zero(T); j12 = zero(T); j22 = zero(T); j32 = zero(T); j13 = zero(T); j23 = zero(T); j33 = zero(T)
    for (lam, w) in rule
        y = lam[1] * v1 + lam[2] * v2 + lam[3] * v3
        d = target - y
        r2 = d[1] * d[1] + d[2] * d[2] + d[3] * d[3]
        r2 == zero(T) && continue
        invr = inv(sqrt(r2))
        s = w * area
        if GRAD
            v = _vortex_pair_ugh(d[1], d[2], d[3], r2, invr, gx * s, gy * s, gz * s, one(T), -T(3))
            ux += v[2]; uy += v[3]; uz += v[4]
            j11 += v[5]; j21 += v[6]; j31 += v[7]; j12 += v[8]; j22 += v[9]; j32 += v[10]; j13 += v[11]; j23 += v[12]; j33 += v[13]
        else
            v = _vortex_pair_ug(d[1], d[2], d[3], invr, gx * s, gy * s, gz * s, one(T))
            ux += v[2]; uy += v[3]; uz += v[4]
        end
    end
    return (zero(T), ux, uy, uz, j11, j21, j31, j12, j22, j32, j13, j23, j33)
end

@inline function _direct_pair_ug(kernel::VortexSheetPanelKernel, dx, dy, dz, r2_, invr, source_bodies, j)
    v = _vortex_sheet_pair(kernel, dx, dy, dz, source_bodies, j, Val(false))
    return v[1], v[2], v[3], v[4]
end
@inline _direct_pair_ugh(kernel::VortexSheetPanelKernel, dx, dy, dz, r2_, invr, source_bodies, j) =
    _vortex_sheet_pair(kernel, dx, dy, dz, source_bodies, j, Val(true))

# Shared vortex U/J assembly for a given regularization pair (g, h); g = 1,
# h = −3 reproduces the singular Biot-Savart kernel exactly.
@inline function _vortex_pair_ug(dx, dy, dz, invr, gsx, gsy, gsz, g)
    T = typeof(dx)
    cr3 = inv(T(4) * T(π)) * invr * invr * invr
    ux = (dz * gsy - dy * gsz) * cr3
    uy = (dx * gsz - dz * gsx) * cr3
    uz = (dy * gsx - dx * gsy) * cr3
    return zero(T), g * ux, g * uy, g * uz
end

@inline function _vortex_pair_ugh(dx, dy, dz, r2, invr, gsx, gsy, gsz, g, h)
    T = typeof(dx)
    cr3 = inv(T(4) * T(π)) * invr * invr * invr
    crss1 = (dz * gsy - dy * gsz) * cr3
    crss2 = (dx * gsz - dz * gsx) * cr3
    crss3 = (dy * gsx - dx * gsy) * cr3
    a = h * invr * invr
    b = -g * cr3
    return (zero(T), g * crss1, g * crss2, g * crss3,
        a * crss1 * dx, a * crss2 * dx - b * gsz, a * crss3 * dx + b * gsy,
        a * crss1 * dy + b * gsz, a * crss2 * dy, a * crss3 * dy - b * gsx,
        a * crss1 * dz - b * gsy, a * crss2 * dz + b * gsx, a * crss3 * dz)
end

# Source (row 5) plus vortex (rows 6:8) on one body: the sum of the two singular pairs.
@inline function _direct_pair_ug(::SingularSourceVortex, dx, dy, dz, r2, invr,
        source_bodies, j)
    T = typeof(r2)
    @inbounds q = source_bodies[5, j] * inv(T(4) * T(π))
    @inbounds gsx = source_bodies[6, j]
    @inbounds gsy = source_bodies[7, j]
    @inbounds gsz = source_bodies[8, j]
    invr3 = invr * invr * invr
    _, vx, vy, vz = _vortex_pair_ug(dx, dy, dz, invr, gsx, gsy, gsz, one(T))
    return q * invr, vx - q * dx * invr3, vy - q * dy * invr3, vz - q * dz * invr3
end

@inline function _direct_pair_ugh(::SingularSourceVortex, dx, dy, dz, r2, invr,
        source_bodies, j)
    T = typeof(r2)
    s = _direct_pair_ugh(SingularSource(), dx, dy, dz, r2, invr, source_bodies, j)
    @inbounds gsx = source_bodies[6, j]
    @inbounds gsy = source_bodies[7, j]
    @inbounds gsz = source_bodies[8, j]
    v = _vortex_pair_ugh(dx, dy, dz, r2, invr, gsx, gsy, gsz, one(T), -T(3))
    return ntuple(i -> s[i] + v[i], Val(13))
end

@inline function _direct_pair_ug(::SingularVortex, dx, dy, dz, r2, invr,
        source_bodies, j)
    T = typeof(r2)
    @inbounds gsx = source_bodies[5, j]
    @inbounds gsy = source_bodies[6, j]
    @inbounds gsz = source_bodies[7, j]
    return _vortex_pair_ug(dx, dy, dz, invr, gsx, gsy, gsz, one(T))
end

@inline function _direct_pair_ugh(::SingularVortex, dx, dy, dz, r2, invr,
        source_bodies, j)
    T = typeof(r2)
    @inbounds gsx = source_bodies[5, j]
    @inbounds gsy = source_bodies[6, j]
    @inbounds gsz = source_bodies[7, j]
    return _vortex_pair_ugh(dx, dy, dz, r2, invr, gsx, gsy, gsz, one(T), -T(3))
end

@inline function _direct_pair_ug(kernel::RegularizedVortex, dx, dy, dz, r2, invr,
        source_bodies, j)
    T = typeof(r2)
    @inbounds gsx = source_bodies[5, j]
    @inbounds gsy = source_bodies[6, j]
    @inbounds gsz = source_bodies[7, j]
    @inbounds sigma = source_bodies[kernel.sigma_row, j]
    g = one(T)
    if sigma > zero(T)
        g, _ = _gaussianerf_g_h(r2 * invr / sigma)
    end
    return _vortex_pair_ug(dx, dy, dz, invr, gsx, gsy, gsz, g)
end

@inline function _direct_pair_ugh(kernel::RegularizedVortex, dx, dy, dz, r2, invr,
        source_bodies, j)
    T = typeof(r2)
    @inbounds gsx = source_bodies[5, j]
    @inbounds gsy = source_bodies[6, j]
    @inbounds gsz = source_bodies[7, j]
    @inbounds sigma = source_bodies[kernel.sigma_row, j]
    g = one(T)
    h = -T(3)
    if sigma > zero(T)
        g, h = _gaussianerf_g_h(r2 * invr / sigma)
    end
    return _vortex_pair_ugh(dx, dy, dz, r2, invr, gsx, gsy, gsz, g, h)
end

# Partitioned replacement (the port candidate 2, theory §2): stable regularized
# g/h inside the smoothing cutoff, exact singular limits beyond it — the branch
# selects HOW a direct pair is evaluated, never WHICH pairs are direct (the
# adequacy gate guarantees every cutoff pair is in the direct set).
#
# The two-pass hybrid's pass 1 (, candidate 3) is the same
# math with the branch at rho_c instead of rho_t: stable regularized U/J for
# ρ ≤ rho_c, exact singular beyond, with the pass-2 deficit sweep
# (_add_host_twopass_deficit!) supplying the correction on (rho_c, rho_t].
@inline _pass1_regularized_cutoff(kernel::PartitionedVortex) = kernel.rho_t
@inline _pass1_regularized_cutoff(kernel::TwoPassVortex) = kernel.rho_c

@inline function _direct_pair_ug(kernel::Union{PartitionedVortex,TwoPassVortex},
        dx, dy, dz, r2, invr, source_bodies, j)
    T = typeof(r2)
    @inbounds gsx = source_bodies[5, j]
    @inbounds gsy = source_bodies[6, j]
    @inbounds gsz = source_bodies[7, j]
    @inbounds sigma = source_bodies[kernel.sigma_row, j]
    g = one(T)
    if sigma > zero(T)
        rho = r2 * invr / sigma
        if rho <= T(_pass1_regularized_cutoff(kernel))
            g, _ = _gaussianerf_g_h(rho)
        end
    end
    return _vortex_pair_ug(dx, dy, dz, invr, gsx, gsy, gsz, g)
end

@inline function _direct_pair_ugh(kernel::Union{PartitionedVortex,TwoPassVortex},
        dx, dy, dz, r2, invr, source_bodies, j)
    T = typeof(r2)
    @inbounds gsx = source_bodies[5, j]
    @inbounds gsy = source_bodies[6, j]
    @inbounds gsz = source_bodies[7, j]
    @inbounds sigma = source_bodies[kernel.sigma_row, j]
    g = one(T)
    h = -T(3)
    if sigma > zero(T)
        rho = r2 * invr / sigma
        if rho <= T(_pass1_regularized_cutoff(kernel))
            g, h = _gaussianerf_g_h(rho)
        end
    end
    return _vortex_pair_ugh(dx, dy, dz, r2, invr, gsx, gsy, gsz, g, h)
end

# Two-pass deficit coefficients (031a §6.1) as an effective (g, h) pair for the
# shared vortex U/J assembly: with g_e = −ḡ and h_e = ρg′ + 3ḡ,
# `_vortex_pair_ug(h)` yields exactly ΔU = −ḡC, Δa = h_e/r² = (ρg′+3ḡ)/r², and
# Δb = −g_e/(4πr³) = ḡ/(4πr³), and (singular) + (deficit) = (regularized)
# identically: (1 − ḡ, ρg′ + 3ḡ − 3) = (g, ρg′ − 3g). Pairs outside the shell
# (rho_c, rho_t] contribute nothing (ρ ≤ rho_c is fully handled by pass 1's
# regularized branch; beyond rho_t the tail is inside the §4 error budget).
@inline function _twopass_deficit_gh(kernel::TwoPassVortex, rho::T) where T<:AbstractFloat
    (T(kernel.rho_c) < rho <= T(kernel.rho_t)) || return zero(T), zero(T)
    gbar, rhogp = _gaussianerf_gbar_rhogp(rho)
    return -gbar, muladd(T(3), gbar, rhogp)
end

#------- cheapened g/h evaluation modes -------#
#
# `CUDA_NEARFIELD_GH_MODE` selects how the regularized-family pair functors
# evaluate the gaussianerf (g, h) — budgets, sizing, and the pointwise ->
# delivered-error mapping in `theory/nearfield-kernel-cheapening-budget.md`
# (derivation script `scripts/fm037f_error_budget.jl`):
#
#   :shipped       the unmodified evaluation above (default; every other call
#                  path is bitwise-identical to pre-037f code);
#   :reduced       12-term series in both precisions (from 19/13), outer
#                  branch unchanged (deg-2 s(u) fails the mapped budget);
#   :fp32          Float64 configurations only: the shipped Float32 math
#                  (13-term series + deg-3 outer), pair U/J assembled in
#                  Float32, accumulated in Float64.  On Float32
#                  configurations this is the shipped path (documented no-op);
#   :reduced_fp32  :fp32 with the 12-term reduced series;
#   :lut           device-only shared-memory lookup table (see
#                  translate_batched_cuda.jl).  The HOST reference path falls
#                  back to :shipped under :lut — host/device parity for :lut
#                  is gated at its budgeted pointwise error, not bitwise.
#
# Like the stage-C mechanism Refs, the CUDA side reads this inside the
# lifecycle body, so the selection is baked into a captured CUDA graph at
# record time: flip it only BEFORE cache construction (or force a new epoch),
# or a replayed graph keeps the old mode silently.
#
# DEFAULT = :fp32: on Float64
# configurations the g/h transcendental (and functor-path assembly) runs in
# Float32 with Float64 accumulation — measured +6.8-10.2% end-to-end U/J on
# cube and +7.4-7.8% on the wake at delivered-error deltas of ~1e-8 relative
# RMS (fm037f_screen.csv / fm037f_decomposition.csv, H200 job 13170769). On
# Float32 configurations :fp32 is bitwise the shipped path (documented
# no-op), so this default changes nothing there. :shipped remains the
# control/opt-out.
const NEARFIELD_GH_MODES = (:shipped, :reduced, :fp32, :reduced_fp32, :lut)
const CUDA_NEARFIELD_GH_MODE = Ref{Symbol}(:fp32)

# 12-term truncations of the exact series (the port sizing: delta vs shipped
# <= 4.9e-6 relative on the series branch, >= 7x under the coherent-tier
# delivered budget B = 2.66e-4; fm037f_budget.csv)
const _GAUSSERF_G_COEFFS_R = _GAUSSERF_G_COEFFS[1:12]
const _GAUSSERF_H_COEFFS_R = _GAUSSERF_H_COEFFS[1:12]
const _GAUSSERF_G_COEFFS32_R = _GAUSSERF_G_COEFFS32[1:12]
const _GAUSSERF_H_COEFFS32_R = _GAUSSERF_H_COEFFS32[1:12]

@inline _gausserf_series_g_r(z::Float64) = evalpoly(z, _GAUSSERF_G_COEFFS_R)
@inline _gausserf_series_h_r(z::Float64) = evalpoly(z, _GAUSSERF_H_COEFFS_R)
@inline _gausserf_series_g_r(z::Float32) = evalpoly(z, _GAUSSERF_G_COEFFS32_R)
@inline _gausserf_series_h_r(z::Float32) = evalpoly(z, _GAUSSERF_H_COEFFS32_R)

@inline function _gaussianerf_g_h_reduced(rho::T) where T<:AbstractFloat
    z = rho * rho
    if rho <= T(2)
        g = T(_GAUSSERF_A) * rho * z * _gausserf_series_g_r(z)
        h = T(_GAUSSERF_A) * rho * z * z * _gausserf_series_h_r(z)
        return g, h
    end
    gbar, rhogp = _gaussianerf_gbar_rhogp(rho)
    g = one(T) - gbar
    return g, rhogp - 3 * g
end

# scalar mode dispatch (:lut resolves at the kernel level on the device and
# falls back to :shipped here; the fp32 modes narrow rho when the caller has
# not already narrowed the whole pair computation)
@inline _gaussianerf_g_h(rho::T, ::Val{:shipped}) where T<:AbstractFloat =
    _gaussianerf_g_h(rho)
@inline _gaussianerf_g_h(rho::T, ::Val{:reduced}) where T<:AbstractFloat =
    _gaussianerf_g_h_reduced(rho)
@inline _gaussianerf_g_h(rho::T, ::Val{:lut}) where T<:AbstractFloat =
    _gaussianerf_g_h(rho)
@inline _gaussianerf_g_h(rho::Float32, ::Val{:fp32}) = _gaussianerf_g_h(rho)
@inline _gaussianerf_g_h(rho::Float32, ::Val{:reduced_fp32}) =
    _gaussianerf_g_h_reduced(rho)
@inline function _gaussianerf_g_h(rho::Float64, ::Val{:fp32})
    g, h = _gaussianerf_g_h(Float32(rho))
    return Float64(g), Float64(h)
end
@inline function _gaussianerf_g_h(rho::Float64, ::Val{:reduced_fp32})
    g, h = _gaussianerf_g_h_reduced(Float32(rho))
    return Float64(g), Float64(h)
end

# (g, h) under a mode with the kernel's branch structure: RegularizedVortex is
# regularized everywhere; the split kernels switch to singular beyond the
# pass-1 cutoff.  sigma <= 0 padding falls back to singular as shipped.
@inline function _pair_gh_mode(::RegularizedVortex, r2::T, invr::T, sigma::T,
        mv::Val) where T
    sigma > zero(T) || return one(T), -T(3)
    return _gaussianerf_g_h(r2 * invr / sigma, mv)
end
@inline function _pair_gh_mode(kernel::Union{PartitionedVortex,TwoPassVortex},
        r2::T, invr::T, sigma::T, mv::Val) where T
    g = one(T)
    h = -T(3)
    if sigma > zero(T)
        rho = r2 * invr / sigma
        if rho <= T(_pass1_regularized_cutoff(kernel))
            g, h = _gaussianerf_g_h(rho, mv)
        end
    end
    return g, h
end

@inline _widen_pair(::Type{T}, v::NTuple{N,Float32}) where {T,N} =
    ntuple(i -> T(v[i]), Val(N))

# Mode-threaded functor entry points.  The generic fallback ignores the mode,
# so consumer functors keep the documented 8-argument contract and the
# singular kernels are mode-independent; :shipped routes straight to the
# 8-argument methods (bitwise-identical code paths).
@inline _direct_pair_ug(kernel::AbstractDirectKernel, dx, dy, dz, r2, invr,
        source_bodies, j, ::Val) =
    _direct_pair_ug(kernel, dx, dy, dz, r2, invr, source_bodies, j)
@inline _direct_pair_ugh(kernel::AbstractDirectKernel, dx, dy, dz, r2, invr,
        source_bodies, j, ::Val) =
    _direct_pair_ugh(kernel, dx, dy, dz, r2, invr, source_bodies, j)

@inline function _direct_pair_ug(kernel::AbstractRegularizedVortex, dx, dy, dz,
        r2, invr, source_bodies, j, ::Val{GH}) where GH
    T = typeof(r2)
    if GH === :shipped || GH === :lut || (GH === :fp32 && T === Float32)
        return _direct_pair_ug(kernel, dx, dy, dz, r2, invr, source_bodies, j)
    elseif (GH === :fp32 || GH === :reduced_fp32) && T === Float64
        @inbounds gsx = Float32(source_bodies[5, j])
        @inbounds gsy = Float32(source_bodies[6, j])
        @inbounds gsz = Float32(source_bodies[7, j])
        @inbounds sigma = Float32(source_bodies[kernel.sigma_row, j])
        r232 = Float32(r2)
        invr32 = Float32(invr)
        g, _ = _pair_gh_mode(kernel, r232, invr32, sigma,
            GH === :fp32 ? Val(:shipped) : Val(:reduced))
        v = _vortex_pair_ug(Float32(dx), Float32(dy), Float32(dz), invr32,
            gsx, gsy, gsz, g)
        return _widen_pair(T, v)
    else # :reduced (also :reduced_fp32 on a Float32 configuration)
        @inbounds gsx = source_bodies[5, j]
        @inbounds gsy = source_bodies[6, j]
        @inbounds gsz = source_bodies[7, j]
        @inbounds sigma = source_bodies[kernel.sigma_row, j]
        g, _ = _pair_gh_mode(kernel, r2, invr, sigma, Val(:reduced))
        return _vortex_pair_ug(dx, dy, dz, invr, gsx, gsy, gsz, g)
    end
end

@inline function _direct_pair_ugh(kernel::AbstractRegularizedVortex, dx, dy, dz,
        r2, invr, source_bodies, j, ::Val{GH}) where GH
    T = typeof(r2)
    if GH === :shipped || GH === :lut || (GH === :fp32 && T === Float32)
        return _direct_pair_ugh(kernel, dx, dy, dz, r2, invr, source_bodies, j)
    elseif (GH === :fp32 || GH === :reduced_fp32) && T === Float64
        @inbounds gsx = Float32(source_bodies[5, j])
        @inbounds gsy = Float32(source_bodies[6, j])
        @inbounds gsz = Float32(source_bodies[7, j])
        @inbounds sigma = Float32(source_bodies[kernel.sigma_row, j])
        r232 = Float32(r2)
        invr32 = Float32(invr)
        g, h = _pair_gh_mode(kernel, r232, invr32, sigma,
            GH === :fp32 ? Val(:shipped) : Val(:reduced))
        v = _vortex_pair_ugh(Float32(dx), Float32(dy), Float32(dz), r232,
            invr32, gsx, gsy, gsz, g, h)
        return _widen_pair(T, v)
    else # :reduced (also :reduced_fp32 on a Float32 configuration)
        @inbounds gsx = source_bodies[5, j]
        @inbounds gsy = source_bodies[6, j]
        @inbounds gsz = source_bodies[7, j]
        @inbounds sigma = source_bodies[kernel.sigma_row, j]
        g, h = _pair_gh_mode(kernel, r2, invr, sigma, Val(:reduced))
        return _vortex_pair_ugh(dx, dy, dz, r2, invr, gsx, gsy, gsz, g, h)
    end
end

# Host-side :lut table builder (the port; also the construction source of the
# device table).  Linear interpolation in x = rho^2 over [0, rho_t^2] of the
# NORMALIZED functions G(x) = g/rho^3 and H(x) = h/rho^5 — analytic in x with
# G(0) = A/3 != 0, so the table preserves relative accuracy down to rho -> 0.
# Values sample the shipped Float64 evaluator, so the LUT inherits (never adds
# to) the shipped outer-fit error; Float32 storage; N = 1024 sized in
# fm037f_budget.csv (interp delta <= 3.0e-6 relative, 6x under budget).
const _NF_GH_LUT_N = 1024

function _build_gh_lut(rho_t::Float64)
    x_max = rho_t * rho_t
    tab = Matrix{Float32}(undef, 2, _NF_GH_LUT_N)
    tab[1, 1] = Float32(_GAUSSERF_A / 3)
    tab[2, 1] = Float32(-_GAUSSERF_A / 5)
    for i in 2:_NF_GH_LUT_N
        x = x_max * (i - 1) / (_NF_GH_LUT_N - 1)
        rho = sqrt(x)
        g, h = _gaussianerf_g_h(rho)
        tab[1, i] = Float32(g / rho^3)
        tab[2, i] = Float32(h / rho^5)
    end
    return tab
end

# Shared LUT lookup (host mirror of the device math; `tab` is any 2 x N
# indexable).  Returns the singular (1, -3) for x >= x_max — the partitioned
# cutoff itself (half-open boundary, measure zero vs the shipped `<=`).
@inline function _gh_from_lut(tab, rho::T, x_max::T) where T
    x = rho * rho
    x >= x_max && return one(T), -T(3)
    t = x * (T(_NF_GH_LUT_N - 1) / x_max)
    i0 = unsafe_trunc(Int32, t)
    f = t - T(i0)
    i1 = i0 + Int32(1)
    @inbounds G0 = T(tab[1, i1])
    @inbounds G1 = T(tab[1, i1 + Int32(1)])
    @inbounds H0 = T(tab[2, i1])
    @inbounds H1 = T(tab[2, i1 + Int32(1)])
    G = muladd(f, G1 - G0, G0)
    H = muladd(f, H1 - H0, H0)
    return rho * x * G, rho * x * x * H
end

# LUT-mode pair math for the regularized family: identical branch structure to
# `_direct_pair_ug(h)` (sigma <= 0 -> singular; split kernels switch at the
# pass-1 cutoff; x >= rho_t^2 -> singular, the table's own domain end).
@inline _lut_pair_cutoff(kernel::AbstractRegularizedVortex) = kernel.rho_t
@inline _lut_pair_cutoff(kernel::TwoPassVortex) = kernel.rho_c

@inline function _lut_pair_gh(kernel::AbstractRegularizedVortex, shlut,
        r2::T, invr::T, sigma::T) where T
    g = one(T)
    h = -T(3)
    if sigma > zero(T)
        rho = r2 * invr / sigma
        if rho <= T(_lut_pair_cutoff(kernel))
            g, h = _gh_from_lut(shlut, rho, T(kernel.rho_t)^2)
        end
    end
    return g, h
end

@inline function _lut_pair_ug(kernel::AbstractRegularizedVortex, shlut,
        dx, dy, dz, r2, invr, source_bodies, j)
    @inbounds gsx = source_bodies[5, j]
    @inbounds gsy = source_bodies[6, j]
    @inbounds gsz = source_bodies[7, j]
    @inbounds sigma = source_bodies[kernel.sigma_row, j]
    g, _ = _lut_pair_gh(kernel, shlut, r2, invr, sigma)
    return _vortex_pair_ug(dx, dy, dz, invr, gsx, gsy, gsz, g)
end

@inline function _lut_pair_ugh(kernel::AbstractRegularizedVortex, shlut,
        dx, dy, dz, r2, invr, source_bodies, j)
    @inbounds gsx = source_bodies[5, j]
    @inbounds gsy = source_bodies[6, j]
    @inbounds gsz = source_bodies[7, j]
    @inbounds sigma = source_bodies[kernel.sigma_row, j]
    g, h = _lut_pair_gh(kernel, shlut, r2, invr, sigma)
    return _vortex_pair_ugh(dx, dy, dz, r2, invr, gsx, gsy, gsz, g, h)
end

# Validated host-side mode (the host reference path maps :lut -> :shipped)
function _validated_host_gh_mode()
    m = radix_setting(:CUDA_NEARFIELD_GH_MODE)
    m in NEARFIELD_GH_MODES || throw(ArgumentError(
        "CUDA_NEARFIELD_GH_MODE must be one of $(NEARFIELD_GH_MODES); got $m"))
    return m === :lut ? :shipped : m
end

# Singular Biot-Savart direct kernel for Point{Vortex} sources ():
# U = -Δx×Γ/(4πr³), J per theory §1 with g→1 (transcribed from the legacy
# test-reference vortex direct!). No scalar potential is produced. Retained
# verbatim (with the scalar kernels below) as the hard-coded reference for the
# stage-2 functor-abstraction benchmark; production dispatch now routes through
# `_host_direct_pairs_functor_kernel!`.
function _host_direct_pairs_vortex_kernel!(output::AbstractMatrix{TF}, source_bodies,
        cell_ranges, direct_targets, direct_sources, n_direct::Int,
        ::Val{HS}) where {TF,HS}
    c = inv(TF(4) * TF(pi))
    @inbounds for pair_i in 1:n_direct
        target_cell = direct_targets[pair_i]
        source_cell = direct_sources[pair_i]
        tfirst = cell_ranges[1, target_cell]
        tcount = cell_ranges[2, target_cell]
        sfirst = cell_ranges[1, source_cell]
        scount = cell_ranges[2, source_cell]
        for i in tfirst:(tfirst + tcount - 1)
            xi = source_bodies[1, i]
            yi = source_bodies[2, i]
            zi = source_bodies[3, i]
            for j in sfirst:(sfirst + scount - 1)
                i == j && continue
                dx = xi - source_bodies[1, j]
                dy = yi - source_bodies[2, j]
                dz = zi - source_bodies[3, j]
                r2 = dx * dx + dy * dy + dz * dz
                r2 == zero(TF) && continue
                gx = source_bodies[5, j]
                gy = source_bodies[6, j]
                gz = source_bodies[7, j]
                invr = inv(sqrt(r2))
                invr2 = invr * invr
                denom = c * invr * invr2
                output[2, i] += (dz * gy - dy * gz) * denom
                output[3, i] += (dx * gz - dz * gx) * denom
                output[4, i] += (dy * gx - dx * gy) * denom
                if HS
                    denom *= invr2
                    output[5, i] += -3 * dx * (gy * dz - gz * dy) * denom
                    output[6, i] += (-3 * dx * (gz * dx - gx * dz) + gz * r2) * denom
                    output[7, i] += (-3 * dx * (gx * dy - gy * dx) - gy * r2) * denom
                    output[8, i] += (-3 * dy * (gy * dz - gz * dy) - gz * r2) * denom
                    output[9, i] += -3 * dy * (gz * dx - gx * dz) * denom
                    output[10, i] += (-3 * dy * (gx * dy - gy * dx) + gx * r2) * denom
                    output[11, i] += (-3 * dz * (gy * dz - gz * dy) + gy * r2) * denom
                    output[12, i] += (-3 * dz * (gz * dx - gx * dz) - gx * r2) * denom
                    output[13, i] += -3 * dz * (gx * dy - gy * dx) * denom
                end
            end
        end
    end
    return output
end

function _host_direct_pairs_kernel!(output::AbstractMatrix{TF}, source_bodies,
        cell_ranges, direct_targets, direct_sources, n_direct::Int) where TF
    c = inv(TF(4) * TF(pi))
    @inbounds for pair_i in 1:n_direct
        target_cell = direct_targets[pair_i]
        source_cell = direct_sources[pair_i]
        tfirst = cell_ranges[1, target_cell]
        tcount = cell_ranges[2, target_cell]
        sfirst = cell_ranges[1, source_cell]
        scount = cell_ranges[2, source_cell]
        for i in tfirst:(tfirst + tcount - 1)
            xi = source_bodies[1, i]
            yi = source_bodies[2, i]
            zi = source_bodies[3, i]
            for j in sfirst:(sfirst + scount - 1)
                i == j && continue
                dx = xi - source_bodies[1, j]
                dy = yi - source_bodies[2, j]
                dz = zi - source_bodies[3, j]
                r2 = dx * dx + dy * dy + dz * dz
                r2 == zero(TF) && continue
                invr = inv(sqrt(r2))
                q = source_bodies[5, j] * c
                output[1, i] += q * invr
                invr3 = invr * invr * invr
                output[2, i] -= q * dx * invr3
                output[3, i] -= q * dy * invr3
                output[4, i] -= q * dz * invr3
            end
        end
    end
    return output
end

# Singular scalar direct kernel with the 9-component hessian:
# u = qc/r, g = -qc·Δx/r³, H = qc·(3ΔxΔxᵀ/r⁵ - I/r³) — symmetric, so the
# column-major linear order equals the row-major one.
function _host_direct_pairs_hessian_kernel!(output::AbstractMatrix{TF}, source_bodies,
        cell_ranges, direct_targets, direct_sources, n_direct::Int) where TF
    c = inv(TF(4) * TF(pi))
    @inbounds for pair_i in 1:n_direct
        target_cell = direct_targets[pair_i]
        source_cell = direct_sources[pair_i]
        tfirst = cell_ranges[1, target_cell]
        tcount = cell_ranges[2, target_cell]
        sfirst = cell_ranges[1, source_cell]
        scount = cell_ranges[2, source_cell]
        for i in tfirst:(tfirst + tcount - 1)
            xi = source_bodies[1, i]
            yi = source_bodies[2, i]
            zi = source_bodies[3, i]
            for j in sfirst:(sfirst + scount - 1)
                i == j && continue
                dx = xi - source_bodies[1, j]
                dy = yi - source_bodies[2, j]
                dz = zi - source_bodies[3, j]
                r2 = dx * dx + dy * dy + dz * dz
                r2 == zero(TF) && continue
                invr = inv(sqrt(r2))
                q = source_bodies[5, j] * c
                output[1, i] += q * invr
                invr2 = invr * invr
                invr3 = invr * invr2
                output[2, i] -= q * dx * invr3
                output[3, i] -= q * dy * invr3
                output[4, i] -= q * dz * invr3
                q3invr5 = 3 * q * invr3 * invr2
                qinvr3 = q * invr3
                output[5, i] += q3invr5 * dx * dx - qinvr3
                output[6, i] += q3invr5 * dx * dy
                output[7, i] += q3invr5 * dx * dz
                output[8, i] += q3invr5 * dy * dx
                output[9, i] += q3invr5 * dy * dy - qinvr3
                output[10, i] += q3invr5 * dy * dz
                output[11, i] += q3invr5 * dz * dx
                output[12, i] += q3invr5 * dz * dy
                output[13, i] += q3invr5 * dz * dz - qinvr3
            end
        end
    end
    return output
end

@inline _resident_flat_phi_re(ph, j, P, n, m) =
    (n < 0 || n > P || m < 0 || m > n) ? zero(eltype(ph)) : ph[flat_basis_index(n, m, 1), j]

@inline _resident_flat_phi_im(ph, j, P, n, m) =
    (n < 0 || n > P || m < 0 || m > n) ? zero(eltype(ph)) : ph[flat_basis_index(n, m, 2), j]

@inline _resident_flat_chi_re(ch, j, P, n, m) =
    (n < 0 || n > P || m < 0 || m > n) ? zero(eltype(ch)) : ch[flat_basis_index(n, m, 1), j]

@inline _resident_flat_chi_im(ch, j, P, n, m) =
    (n < 0 || n > P || m < 0 || m > n) ? zero(eltype(ch)) : ch[flat_basis_index(n, m, 2), j]

@inline function _resident_local_eval_flat(ph, ch, node, dx, dy, dz, P_phi, P_active, ::Val{LH}) where LH
    TF = eltype(ph)
    c = inv(TF(4) * TF(pi))
    u = zero(TF)
    vx = zero(TF); vy = zero(TF); vz = zero(TF)
    # every coefficient below is taken at the SAME offset, so the transcendental
    # prologue is evaluated once here instead of once per (n,m)
    setup = _resident_harmonic_setup(dx, dy, dz)
    @inbounds for n in 0:P_active
        rre, rim = _resident_regular_harmonic_coeff(setup, n, 0)
        if n <= P_phi && (!LH || n == 0)
            u += rre * _resident_flat_phi_re(ph, node, P_phi, n, 0) -
                 rim * _resident_flat_phi_im(ph, node, P_phi, n, 0)
        end

        phi1c = _resident_flat_phi_re(ph, node, P_phi, n + 1, 1)
        phi1s = _resident_flat_phi_im(ph, node, P_phi, n + 1, 1)
        phi0c = _resident_flat_phi_re(ph, node, P_phi, n + 1, 0)
        phi0s = _resident_flat_phi_im(ph, node, P_phi, n + 1, 0)
        vxr = -phi1s; vxi = zero(TF)
        vyr = -phi1c; vyi = zero(TF)
        vzr = -phi0c; vzi = -phi0s
        if LH
            vxr += n * _resident_flat_chi_re(ch, node, P_active, n, 1)
            vyr -= n * _resident_flat_chi_im(ch, node, P_active, n, 1)
        end
        vx += vxr * rre - vxi * rim
        vy += vyr * rre - vyi * rim
        vz += vzr * rre - vzi * rim

        for m in 1:n
            rre, rim = _resident_regular_harmonic_coeff(setup, n, m)
            if n <= P_phi && !LH
                u += 2 * (rre * _resident_flat_phi_re(ph, node, P_phi, n, m) -
                          rim * _resident_flat_phi_im(ph, node, P_phi, n, m))
            end

            amm1 = _resident_flat_phi_re(ph, node, P_phi, n + 1, m - 1)
            bmm1 = _resident_flat_phi_im(ph, node, P_phi, n + 1, m - 1)
            amp1 = _resident_flat_phi_re(ph, node, P_phi, n + 1, m + 1)
            bmp1 = _resident_flat_phi_im(ph, node, P_phi, n + 1, m + 1)
            am = _resident_flat_phi_re(ph, node, P_phi, n + 1, m)
            bm = _resident_flat_phi_im(ph, node, P_phi, n + 1, m)

            vxr = -(bmm1 + bmp1) * TF(0.5)
            vxi = (amm1 + amp1) * TF(0.5)
            vyr = (amm1 - amp1) * TF(0.5)
            vyi = (bmm1 - bmp1) * TF(0.5)
            vzr = -am
            vzi = -bm

            if LH
                cmm1 = _resident_flat_chi_re(ch, node, P_active, n, m - 1)
                dmm1 = _resident_flat_chi_im(ch, node, P_active, n, m - 1)
                cmp1 = _resident_flat_chi_re(ch, node, P_active, n, m + 1)
                dmp1 = _resident_flat_chi_im(ch, node, P_active, n, m + 1)
                cm = _resident_flat_chi_re(ch, node, P_active, n, m)
                dm = _resident_flat_chi_im(ch, node, P_active, n, m)
                vxr += ((n - m) * cmp1 - (n + m) * cmm1) * TF(0.5)
                vxi += ((n - m) * dmp1 - (n + m) * dmm1) * TF(0.5)
                vyr -= ((n - m) * dmp1 + (n + m) * dmm1) * TF(0.5)
                vyi += ((n - m) * cmp1 + (n + m) * cmm1) * TF(0.5)
                vzr += m * dm
                vzi -= m * cm
            end

            vx += 2 * (vxr * rre - vxi * rim)
            vy += 2 * (vyr * rre - vyi * rim)
            vz += 2 * (vzr * rre - vzi * rim)
        end
    end
    return u * c, vx * c, vy * c, vz * c
end

# Per-(n, m) local-expansion gradient coefficient: the values the
# legacy `evaluate_local` stores in its `gradient_n_m` scratch, recomputed on
# the fly from the flat zero-padded accessors so the hessian pass needs no
# per-thread coefficient array. Must stay in lockstep with the coefficient
# blocks inside `_resident_local_eval_flat` above.
@inline function _resident_gradient_coeff(ph, ch, node, P_phi, P_active, n, m,
        ::Val{LH}) where LH
    TF = eltype(ph)
    if m == 0
        phi1c = _resident_flat_phi_re(ph, node, P_phi, n + 1, 1)
        phi1s = _resident_flat_phi_im(ph, node, P_phi, n + 1, 1)
        phi0c = _resident_flat_phi_re(ph, node, P_phi, n + 1, 0)
        phi0s = _resident_flat_phi_im(ph, node, P_phi, n + 1, 0)
        vxr = -phi1s
        vyr = -phi1c
        vzr = -phi0c
        vzi = -phi0s
        if LH
            vxr += n * _resident_flat_chi_re(ch, node, P_active, n, 1)
            vyr -= n * _resident_flat_chi_im(ch, node, P_active, n, 1)
        end
        return vxr, zero(TF), vyr, zero(TF), vzr, vzi
    end
    amm1 = _resident_flat_phi_re(ph, node, P_phi, n + 1, m - 1)
    bmm1 = _resident_flat_phi_im(ph, node, P_phi, n + 1, m - 1)
    amp1 = _resident_flat_phi_re(ph, node, P_phi, n + 1, m + 1)
    bmp1 = _resident_flat_phi_im(ph, node, P_phi, n + 1, m + 1)
    am = _resident_flat_phi_re(ph, node, P_phi, n + 1, m)
    bm = _resident_flat_phi_im(ph, node, P_phi, n + 1, m)

    vxr = -(bmm1 + bmp1) * TF(0.5)
    vxi = (amm1 + amp1) * TF(0.5)
    vyr = (amm1 - amp1) * TF(0.5)
    vyi = (bmm1 - bmp1) * TF(0.5)
    vzr = -am
    vzi = -bm

    if LH
        cmm1 = _resident_flat_chi_re(ch, node, P_active, n, m - 1)
        dmm1 = _resident_flat_chi_im(ch, node, P_active, n, m - 1)
        cmp1 = _resident_flat_chi_re(ch, node, P_active, n, m + 1)
        dmp1 = _resident_flat_chi_im(ch, node, P_active, n, m + 1)
        cm = _resident_flat_chi_re(ch, node, P_active, n, m)
        dm = _resident_flat_chi_im(ch, node, P_active, n, m)
        vxr += ((n - m) * cmp1 - (n + m) * cmm1) * TF(0.5)
        vxi += ((n - m) * dmp1 - (n + m) * dmm1) * TF(0.5)
        vyr -= ((n - m) * dmp1 + (n + m) * dmm1) * TF(0.5)
        vyi += ((n - m) * cmp1 + (n + m) * cmm1) * TF(0.5)
        vzr += m * dm
        vzi -= m * cm
    end
    return vxr, vxi, vyr, vyi, vzr, vzi
end

# Hessian-emitting variant of `_resident_local_eval_flat`: identical
# potential/gradient math (via `_resident_gradient_coeff`), plus a second pass
# porting the `evaluate_expansions.jl` hessian recurrences — each velocity
# component's coefficient field is differentiated with the same operator pattern
# as the first pass. Returns
# `(u, vx, vy, vz, hxx, hxy, hxz, hyx, hyy, hyz, hzx, hzy, hzz)` in the legacy
# `SMatrix{3,3}` column-major linear order (`set_hessian!` order).
@inline function _resident_local_eval_flat_hessian(ph, ch, node, dx, dy, dz,
        P_phi, P_active, lhv::Val{LH}) where LH
    TF = eltype(ph)
    c = inv(TF(4) * TF(pi))
    u = zero(TF)
    vx = zero(TF); vy = zero(TF); vz = zero(TF)
    # one transcendental prologue per body, shared by both passes below
    setup = _resident_harmonic_setup(dx, dy, dz)
    @inbounds for n in 0:P_active
        rre, rim = _resident_regular_harmonic_coeff(setup, n, 0)
        if n <= P_phi && (!LH || n == 0)
            u += rre * _resident_flat_phi_re(ph, node, P_phi, n, 0) -
                 rim * _resident_flat_phi_im(ph, node, P_phi, n, 0)
        end
        vxr, vxi, vyr, vyi, vzr, vzi =
            _resident_gradient_coeff(ph, ch, node, P_phi, P_active, n, 0, lhv)
        vx += vxr * rre - vxi * rim
        vy += vyr * rre - vyi * rim
        vz += vzr * rre - vzi * rim
        for m in 1:n
            rre, rim = _resident_regular_harmonic_coeff(setup, n, m)
            if n <= P_phi && !LH
                u += 2 * (rre * _resident_flat_phi_re(ph, node, P_phi, n, m) -
                          rim * _resident_flat_phi_im(ph, node, P_phi, n, m))
            end
            vxr, vxi, vyr, vyi, vzr, vzi =
                _resident_gradient_coeff(ph, ch, node, P_phi, P_active, n, m, lhv)
            vx += 2 * (vxr * rre - vxi * rim)
            vy += 2 * (vyr * rre - vyi * rim)
            vz += 2 * (vzr * rre - vzi * rim)
        end
    end

    hxx = zero(TF); hxy = zero(TF); hxz = zero(TF)
    hyx = zero(TF); hyy = zero(TF); hyz = zero(TF)
    hzx = zero(TF); hzy = zero(TF); hzz = zero(TF)
    @inbounds for n in 0:(P_active - 1)
        rre, rim = _resident_regular_harmonic_coeff(setup, n, 0)
        # gradient coefficients at (n+1, 0) and (n+1, 1)
        g0x_r, g0x_i, g0y_r, g0y_i, g0z_r, g0z_i =
            _resident_gradient_coeff(ph, ch, node, P_phi, P_active, n + 1, 0, lhv)
        g1x_r, g1x_i, g1y_r, g1y_i, g1z_r, g1z_i =
            _resident_gradient_coeff(ph, ch, node, P_phi, P_active, n + 1, 1, lhv)
        hxx += -g1x_i * rre
        hyx += -g1x_r * rre
        hzx += -g0x_r * rre + g0x_i * rim
        hxy += -g1y_i * rre
        hyy += -g1y_r * rre
        hzy += -g0y_r * rre + g0y_i * rim
        hxz += -g1z_i * rre
        hyz += -g1z_r * rre
        hzz += -g0z_r * rre + g0z_i * rim
        for m in 1:n
            rre, rim = _resident_regular_harmonic_coeff(setup, n, m)
            amx_r, amx_i, amy_r, amy_i, amz_r, amz_i =
                _resident_gradient_coeff(ph, ch, node, P_phi, P_active, n + 1, m - 1, lhv)
            bmx_r, bmx_i, bmy_r, bmy_i, bmz_r, bmz_i =
                _resident_gradient_coeff(ph, ch, node, P_phi, P_active, n + 1, m, lhv)
            cmx_r, cmx_i, cmy_r, cmy_i, cmz_r, cmz_i =
                _resident_gradient_coeff(ph, ch, node, P_phi, P_active, n + 1, m + 1, lhv)
            # x column: ∂x, ∂y, ∂z of vx
            tr = -(amx_i + cmx_i) * TF(0.5); ti = (amx_r + cmx_r) * TF(0.5)
            hxx += 2 * (tr * rre - ti * rim)
            tr = (amx_r - cmx_r) * TF(0.5); ti = (amx_i - cmx_i) * TF(0.5)
            hyx += 2 * (tr * rre - ti * rim)
            hzx += 2 * (-bmx_r * rre + bmx_i * rim)
            # y column
            tr = -(amy_i + cmy_i) * TF(0.5); ti = (amy_r + cmy_r) * TF(0.5)
            hxy += 2 * (tr * rre - ti * rim)
            tr = (amy_r - cmy_r) * TF(0.5); ti = (amy_i - cmy_i) * TF(0.5)
            hyy += 2 * (tr * rre - ti * rim)
            hzy += 2 * (-bmy_r * rre + bmy_i * rim)
            # z column
            tr = -(amz_i + cmz_i) * TF(0.5); ti = (amz_r + cmz_r) * TF(0.5)
            hxz += 2 * (tr * rre - ti * rim)
            tr = (amz_r - cmz_r) * TF(0.5); ti = (amz_i - cmz_i) * TF(0.5)
            hyz += 2 * (tr * rre - ti * rim)
            hzz += 2 * (-bmz_r * rre + bmz_i * rim)
        end
    end
    return u * c, vx * c, vy * c, vz * c,
        hxx * c, hxy * c, hxz * c, hyx * c, hyy * c, hyz * c, hzx * c, hzy * c, hzz * c
end

function _launch_host_l2b!(state::DeviceResidentRadixState{TF,B,LH}) where {TF,B,LH}
    fill!(state.output, zero(TF))
    _add_host_direct_pairs!(state)
    _add_host_twopass_deficit!(state)
    P_phi = state.invariant_cache.basis_info.orders.P_phi
    P_active = state.invariant_cache.basis_info.orders.P_active
    if size(state.output, 1) >= 13
        _host_l2b_hessian_kernel!(state.output, state.source_bodies, state.cell_ranges,
            state.cell_centers, state.grid.leaf_to_node, phi_slab(state.locals),
            chi_slab(state.locals), P_phi, P_active, Val(LH), state.counts.n_cells)
    else
        _host_l2b_kernel!(state.output, state.source_bodies, state.cell_ranges,
            state.cell_centers, state.grid.leaf_to_node, phi_slab(state.locals),
            chi_slab(state.locals), P_phi, P_active, Val(LH), state.counts.n_cells)
    end
    return state
end

function _host_l2b_hessian_kernel!(output::AbstractMatrix, source_bodies, cell_ranges,
        cell_centers, leaf_to_node, ph, ch, P_phi::Int, P_active::Int,
        lhv::Val{LH}, n_cells::Int) where LH
    @inbounds for cell in 1:n_cells
        node = leaf_to_node[cell]
        first_body = cell_ranges[1, cell]
        count = cell_ranges[2, cell]
        cx = cell_centers[1, cell]
        cy = cell_centers[2, cell]
        cz = cell_centers[3, cell]
        for i in first_body:(first_body + count - 1)
            vals = _resident_local_eval_flat_hessian(
                ph, ch, node,
                source_bodies[1, i] - cx,
                source_bodies[2, i] - cy,
                source_bodies[3, i] - cz,
                P_phi, P_active, lhv,
            )
            for row in 1:13
                output[row, i] += vals[row]
            end
        end
    end
    return output
end

function _host_l2b_kernel!(output::AbstractMatrix, source_bodies, cell_ranges,
        cell_centers, leaf_to_node, ph, ch, P_phi::Int, P_active::Int,
        lhv::Val{LH}, n_cells::Int) where LH
    @inbounds for cell in 1:n_cells
        node = leaf_to_node[cell]
        first_body = cell_ranges[1, cell]
        count = cell_ranges[2, cell]
        cx = cell_centers[1, cell]
        cy = cell_centers[2, cell]
        cz = cell_centers[3, cell]
        for i in first_body:(first_body + count - 1)
            scalar_potential, gx, gy, gz = _resident_local_eval_flat(
                ph, ch, node,
                source_bodies[1, i] - cx,
                source_bodies[2, i] - cy,
                source_bodies[3, i] - cz,
                P_phi, P_active, lhv,
            )
            output[1, i] += scalar_potential
            output[2, i] += gx
            output[3, i] += gy
            output[4, i] += gz
        end
    end
    return output
end

_radix_uses_factored_rotation(options::RadixLifecycleOptions) =
    options.operator isa FactoredRotationM2L

function _launch_host_resident_operator_pipeline!(state::DeviceResidentRadixState)
    _launch_host_m2m!(state)
    # 016b watch item 2: the FactoredRotation* operators are exact only on the
    # physical subspace (m = 0 imaginary rows == 0); guard the upward-pass output
    # once per lifecycle run, DEBUG[]-gated (off in production).
    if DEBUG[] && _radix_uses_factored_rotation(state.options)
        _assert_factored_input_physical(state.multipoles)
    end
    _launch_host_m2l!(state)
    _launch_host_l2l!(state)
    _launch_host_l2b!(state)
    state.counters.expansion_host_copies == 0 ||
        throw(AssertionError("resident host radix lifecycle observed expansion host copies"))
    return state
end

"""
    host_radix_state(systems, grid, list, P, lamb_helmholtz=Val(false); options)

Build the host-resident state used to mirror the radix lifecycle. `systems` may
be a system, tuple of systems, or an already packed body matrix. The state owns
its packed-body copy, flattened routes, expansions, scratch space, output, and
transfer counters, and retains the supplied grid metadata. The `RadixGrid`
overload first creates a fresh host-resident grid with
[`host_resident_radix_grid`](@ref). This is a host-only construction even though
its container type is shared with device backends.
"""
function host_radix_state(systems, grid::RadixGrid, list::RadixInteractionList,
        P::Integer, lamb_helmholtz::Val{LH}=Val(false);
        options::RadixLifecycleOptions=RadixLifecycleOptions()) where LH
    return host_radix_state(systems, host_resident_radix_grid(grid), list, P, lamb_helmholtz; options)
end

function host_radix_state(systems, grid::DeviceRadixGrid, list::RadixInteractionList,
        P::Integer, lamb_helmholtz::Val{LH}=Val(false);
        options::RadixLifecycleOptions=RadixLifecycleOptions()) where LH
    TF = options.precision
    basis_info = OperatorBasisInfo(CompressedComplexBasis(), P, lamb_helmholtz)
    counters = RadixTransferCounters()
    body_perm = grid.perm
    body_system_ids = grid.body_system
    body_indices = grid.body_index
    host_body_perm = grid.perm
    host_body_system_ids = grid.body_system
    host_body_indices = grid.body_index

    source_bodies_raw = systems isa AbstractMatrix ?
        _host_radix_body_matrix(grid, systems) :
        _host_radix_body_matrix(grid, _host_radix_source_buffers(to_tuple(systems), TF))
    source_bodies = Matrix{TF}(source_bodies_raw)

    m2m_parent_routes, m2m_child_routes, l2l_parent_routes, l2l_child_routes =
        _host_radix_tree_routes(grid)
    multipoles = _host_flat_buffer(TF, basis_info, length(grid.node_keys))
    locals = _host_flat_buffer(TF, basis_info, length(grid.node_keys))
    output = zeros(TF, 4, grid.n_bodies)

    levels, offsets, targets, sources = _flatten_radix_routes_host(list, grid)
    direct_targets, direct_sources = _flatten_radix_direct_pairs_host(list)
    cache = OperatorInvariantCache(TF, basis_info)
    scratch = ResidentOperatorWorkspace(
        TF, basis_info, multipoles, grid, list,
        m2m_parent_routes, m2m_child_routes, l2l_parent_routes, l2l_child_routes,
        grid.node_levels, grid.node_centers, targets, sources;
        m2l_strategy=options.m2l_strategy, operator=options.operator,
    )
    return DeviceResidentRadixState{TF,CompressedComplexBasis,LH}(
        grid, list, source_bodies, source_bodies,
        body_perm, body_system_ids, body_indices,
        host_body_perm, host_body_system_ids, host_body_indices,
        grid.cell_centers, m2m_parent_routes, m2m_child_routes,
        l2l_parent_routes, l2l_child_routes, grid.node_levels, grid.node_centers,
        targets, sources,
        grid.cell_centers, grid.cell_ranges,
        m2m_parent_routes, m2m_child_routes, l2l_parent_routes, l2l_child_routes,
        multipoles, locals, levels, offsets, targets, sources,
        direct_targets, direct_sources, output,
        cache, scratch, counters, options,
        RadixStepCounts(source_bodies, grid.cell_ranges, multipoles, targets, direct_targets),
    )
end

function _flatten_radix_routes_host(list::RadixInteractionList, grid::DeviceRadixGrid)
    nroute = sum((length(batch.targets) for batch in list.m2l_batches); init=0)
    levels = Vector{Int}(undef, nroute)
    offsets = Matrix{Int}(undef, 3, nroute)
    targets = Vector{Int}(undef, nroute)
    sources = Vector{Int}(undef, nroute)
    i = 0
    for batch in list.m2l_batches
        for j in eachindex(batch.targets)
            i += 1
            levels[i] = batch.level
            offsets[:, i] .= batch.offset
            targets[i] = grid.leaf_to_node[batch.targets[j]]
            sources[i] = grid.leaf_to_node[batch.sources[j]]
        end
    end
    return levels, offsets, targets, sources
end

function _flatten_radix_direct_pairs_host(list::RadixInteractionList)
    direct_targets = Vector{Int}(undef, length(list.direct_pairs))
    direct_sources = Vector{Int}(undef, length(list.direct_pairs))
    for (i, pair) in pairs(list.direct_pairs)
        direct_targets[i] = pair[1]
        direct_sources[i] = pair[2]
    end
    return direct_targets, direct_sources
end

