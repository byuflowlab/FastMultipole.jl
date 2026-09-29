#------- closed-form panel and filament element influence -------#
#
# Planar triangular source/doublet panels of constant strength and straight
# bound-vortex segments, evaluated in closed form: potential, velocity and
# velocity gradient. Used by the nearfield pair kernels of `Panel{3,...}` and
# `Filament{Vortex}` bodies (resident/resident_pair_kernels.jl). The panel sums
# run over the edges in the panel frame, in the sign convention where a source
# panel's potential is −σ/(4π) ∫ dA/r; callers negate where the point-kernel
# convention differs.

# panel frame: returns the panel-frame axes (columns of R) as SVectors
@inline function _elem_rotate_to_panel(v1::SVector{3,T}, v2::SVector{3,T}, v3::SVector{3,T}) where T
    e1 = v2 - v1
    e2 = v3 - v1
    nz = SVector{3,T}(e1[2]*e2[3] - e1[3]*e2[2],
                      e1[3]*e2[1] - e1[1]*e2[3],
                      e1[1]*e2[2] - e1[2]*e2[1])
    nznorm = sqrt(nz[1]*nz[1] + nz[2]*nz[2] + nz[3]*nz[3])
    nz = nz / (nznorm + eps(nznorm))
    nx = e2
    nxnorm = sqrt(nx[1]*nx[1] + nx[2]*nx[2] + nx[3]*nx[3])
    nx = nx / (nxnorm + eps(nxnorm))
    ny = SVector{3,T}(nz[2]*nx[3] - nz[3]*nx[2],
                      nz[3]*nx[1] - nz[1]*nx[3],
                      nz[1]*nx[2] - nz[2]*nx[1])
    return nx, ny, nz   # columns of R
end

# point-to-segment distance
@inline function _elem_minimum_distance(A::SVector{3,T}, B::SVector{3,T}, p::SVector{3,T}) where T
    AB = B - A
    Ap = p - A
    denom = AB[1]*AB[1] + AB[2]*AB[2] + AB[3]*AB[3]
    proj = (Ap[1]*AB[1] + Ap[2]*AB[2] + Ap[3]*AB[3]) / denom
    proj = clamp(proj, zero(T), one(T))
    d = p - (A + proj * AB)
    return sqrt(d[1]*d[1] + d[2]*d[2] + d[3]*d[3])
end

# compact-support doublet regularizer
@inline function _elem_regularize(distance::T, core_size::T) where T
    return distance < core_size ? (distance - core_size)*(distance - core_size) : zero(T)
end

# per-edge geometric preliminaries shared by the source and doublet edge terms
@inline function _elem_edge_prelims(tRx::T, tRy, tRz, vx_i, vy_i, vx_ip1, vy_ip1) where T
    dxp = tRx - vx_ip1
    dyp = tRy - vy_ip1
    eip1 = dxp*dxp + tRz*tRz
    hip1 = dxp*dyp
    rip1 = sqrt(eip1 + dyp*dyp)
    dxi = tRx - vx_i
    dyi = tRy - vy_i
    ei = dxi*dxi + tRz*tRz
    hi = dxi*dyi
    ri = sqrt(ei + dyi*dyi)
    dx = vx_ip1 - vx_i
    dy = vy_ip1 - vy_i
    ds = sqrt(dx*dx + dy*dy)
    R_dot_s = dx*dxi + dy*dyi
    return eip1, hip1, rip1, ei, hi, ri, ds, dx, dy, R_dot_s
end

# shared solid-angle (tan) term with the extension-singularity guard.
# The on-plane clause fires on tRz == 0 alone: tRz is snapped to an exact zero
# by _elem_tri_source_doublet whenever it is at roundoff scale, so all edges of
# a panel take the same branch and the PV cannot flip with the sign of
# roundoff (with fused multiply-add, den < 0 on 2 of 3 edges at a centroid
# makes atan(num, den) jump by ±π with the sign of num).
# relative tolerance of the singularity guards (extension line, on-plane snap,
# self pair): 1e-12 in Float64, 1e-5 in Float32 (roundoff there is ~1e-7)
@inline _elem_guard_tol(::Type{Float64}) = 1e-12
@inline _elem_guard_tol(::Type{T}) where T = T(1e-5)

@inline function _elem_solid_angle_tan(tRx::T, tRy, tRz, ei, hi, ri, eip1, hip1, rip1,
        ds, dx, dy, R_dot_s) where T
    if tRz == zero(T) ||
       abs(abs(R_dot_s) - ri*ds) <= _elem_guard_tol(T) * ri * ds
        return zero(T)
    else
        arg1 = (dy*ei - hi*dx) / ri
        arg2 = (dy*eip1 - hip1*dx) / rip1
        num = dx * tRz * (arg1 - arg2)
        den = tRz*tRz*dx*dx + arg1*arg2
        return atan(num, den)
    end
end

# ConstantSource per-edge velocity/gradient in the panel frame
# (velocity, gradient and this edge's potential term)
@inline function _elem_edge_source(tRx::T, tRy, tRz, vx_i, vy_i, vx_ip1, vy_ip1,
        ::Val{GRAD}) where {T,GRAD}
    eip1, hip1, rip1, ei, hi, ri, ds, dx, dy, R_dot_s =
        _elem_edge_prelims(tRx, tRy, tRz, vx_i, vy_i, vx_ip1, vy_ip1)
    num = max(eps(T), ri + rip1 - ds)
    log_term = log(num / (ri + rip1 + ds))
    tan_term = _elem_solid_angle_tan(tRx, tRy, tRz, ei, hi, ri, eip1, hip1, rip1,
        ds, dx, dy, R_dot_s)
    u = SVector{3,T}(dy/ds*log_term, -dx/ds*log_term, tan_term)
    # potential term of this edge (the source-panel potential edge sum), from the same prelims
    pe = ((tRx - vx_i)*dy - (tRy - vy_i)*dx) / ds * log_term + tRz * tan_term
    if !GRAD
        return u, zero(SMatrix{3,3,T,9}), pe
    end
    d2 = ds*ds
    r_plus_rp1 = ri + rip1
    r_plus_rp1_2 = r_plus_rp1 * r_plus_rp1
    r_times_rp1 = ri * rip1
    rho = r_times_rp1 + (tRx - vx_i)*(tRx - vx_ip1) + (tRy - vy_i)*(tRy - vy_ip1) + tRz*tRz
    lambda = (tRx - vx_i)*(tRy - vy_ip1) - (tRx - vx_ip1)*(tRy - vy_i)
    ri_inv = 1/ri
    rip1_inv = 1/rip1
    val1 = r_plus_rp1_2 - d2
    val2 = (tRx - vx_i)*ri_inv + (tRx - vx_ip1)*rip1_inv
    val3 = (tRy - vy_i)*ri_inv + (tRy - vy_ip1)*rip1_inv
    val4 = r_plus_rp1 / (r_times_rp1 * rho)
    phi_xx = 2*dy/val1*val2
    phi_xy = -2*dx/val1*val2
    phi_xz = tRz*dy*val4
    phi_yy = -2*dx/val1*val3
    phi_yz = -tRz*dx*val4
    phi_zz = lambda*val4
    g = SMatrix{3,3,T,9}(phi_xx, phi_xy, phi_xz,
                         phi_xy, phi_yy, phi_yz,
                         phi_xz, phi_yz, phi_zz)
    return u, g, pe
end

# ConstantDoublet per-edge velocity/gradient in the panel frame
# (velocity, gradient and this edge's potential term). `reg_term` regularizes
# the velocity denominator only.
@inline function _elem_edge_doublet(tRx::T, tRy, tRz, vx_i, vy_i, vx_ip1, vy_ip1,
        reg_term::T, ::Val{GRAD}) where {T,GRAD}
    eip1, hip1, rip1, ei, hi, ri, ds, dx, dy, R_dot_s =
        _elem_edge_prelims(tRx, tRy, tRz, vx_i, vy_i, vx_ip1, vy_ip1)
    tan_term = _elem_solid_angle_tan(tRx, tRy, tRz, ei, hi, ri, eip1, hip1, rip1,
        ds, dx, dy, R_dot_s)
    r_plus_rp1 = ri + rip1
    r_times_rp1 = ri * rip1
    rho = r_times_rp1 + (tRx - vx_i)*(tRx - vx_ip1) + (tRy - vy_i)*(tRy - vy_ip1) + tRz*tRz
    lambda = (tRx - vx_i)*(tRy - vy_ip1) - (tRx - vx_ip1)*(tRy - vy_i)
    val4v = r_plus_rp1 / (r_times_rp1 * rho + reg_term)
    u = -SVector{3,T}(tRz*dy*val4v, -tRz*dx*val4v, lambda*val4v)
    pe = -tan_term                                         # potential term (unregularized)
    if !GRAD
        return u, zero(SMatrix{3,3,T,9}), pe
    end
    r_plus_rp1_2 = r_plus_rp1 * r_plus_rp1
    val1 = r_times_rp1 * r_plus_rp1_2 + rho * rip1 * rip1
    val1 /= rho * ri * r_plus_rp1
    val2 = r_times_rp1 * r_plus_rp1_2 + rho * ri * ri
    val2 /= rho * rip1 * r_plus_rp1
    val3 = r_plus_rp1 / (rho * r_times_rp1 * r_times_rp1)
    psi_xx = tRz*dy*val3*((tRx - vx_i)*val1 + (tRx - vx_ip1)*val2)
    psi_xy = tRz*dy*val3*((tRy - vy_i)*val1 + (tRy - vy_ip1)*val2)
    psi_yy = -tRz*dx*val3*((tRy - vy_i)*val1 + (tRy - vy_ip1)*val2)
    val4 = r_plus_rp1_2 / rho
    val5 = (ri*ri - r_times_rp1 + rip1*rip1) / r_times_rp1
    val6 = tRz*(val4 + val5)
    psi_zz = lambda*val3*val6
    val7 = r_times_rp1 - tRz*val6
    val8 = val3*val7
    psi_xz = -dy*val8
    psi_yz = dx*val8
    g = SMatrix{3,3,T,9}(psi_xx, psi_xy, psi_xz,
                         psi_xy, psi_yy, psi_yz,
                         psi_xz, psi_yz, psi_zz)
    return u, g, pe
end

# planar triangle source/doublet influence: rotated-frame edge loop, then the
# rotation back u <- -1/4pi R u, g <- -1/4pi R g R^T. Per-strength-unit result
# (the caller multiplies).
@inline function _elem_tri_source_doublet(target::SVector{3,T}, v1::SVector{3,T},
        v2::SVector{3,T}, v3::SVector{3,T}, core_size::T,
        ::Val{DOUBLET}, ::Val{GRAD}) where {T,DOUBLET,GRAD}
    nx, ny, nz = _elem_rotate_to_panel(v1, v2, v3)
    centroid = (v1 + v2 + v3) * T(0.3333333333333333)
    tc = target - centroid
    tRx = nx[1]*tc[1] + nx[2]*tc[2] + nx[3]*tc[3]          # transpose(R)*(target-centroid)
    tRy = ny[1]*tc[1] + ny[2]*tc[2] + ny[3]*tc[3]
    tRz = nz[1]*tc[1] + nz[2]*tc[2] + nz[3]*tc[3]
    u = zero(SVector{3,T})
    g = zero(SMatrix{3,3,T,9})
    p = zero(T)
    # panel-frame vertex coordinates
    w1 = v1 - centroid
    w2 = v2 - centroid
    w3 = v3 - centroid
    vx1 = nx[1]*w1[1] + nx[2]*w1[2] + nx[3]*w1[3]; vy1 = ny[1]*w1[1] + ny[2]*w1[2] + ny[3]*w1[3]
    vx2 = nx[1]*w2[1] + nx[2]*w2[2] + nx[3]*w2[3]; vy2 = ny[1]*w2[1] + ny[2]*w2[2] + ny[3]*w2[3]
    vx3 = nx[1]*w3[1] + nx[2]*w3[2] + nx[3]*w3[3]; vy3 = ny[1]*w3[1] + ny[2]*w3[2] + ny[3]*w3[3]
    # on-plane snap: roundoff-scale tRz means the target IS on the panel plane;
    # an exact zero routes every edge through _elem_solid_angle_tan's PV branch
    # so the ±2π solid-angle side cannot follow the sign of FMA junk on device
    # (roundoff junk ≤ ~1e-17·L vs genuine ≥ ~1e-8·L)
    L2 = w1[1]*w1[1] + w1[2]*w1[2] + w1[3]*w1[3] +
         w2[1]*w2[1] + w2[2]*w2[2] + w2[3]*w2[3] +
         w3[1]*w3[1] + w3[2]*w3[2] + w3[3]*w3[3]
    tRz = ifelse(tRz*tRz <= _elem_guard_tol(T)^2 * L2, zero(T), tRz)
    for i in 1:3
        vxa, vya, wa = i == 1 ? (vx1, vy1, w1) : (i == 2 ? (vx2, vy2, w2) : (vx3, vy3, w3))
        vxb, vyb, wb = i == 1 ? (vx2, vy2, w2) : (i == 2 ? (vx3, vy3, w3) : (vx1, vy1, w1))
        if DOUBLET
            # reg_term from the side's minimum distance
            m_dist = _elem_minimum_distance(wa, wb, tc)
            reg_term = _elem_regularize(m_dist, core_size)
            ue, ge, pe = _elem_edge_doublet(tRx, tRy, tRz, vxa, vya, vxb, vyb, reg_term, Val(GRAD))
        else
            ue, ge, pe = _elem_edge_source(tRx, tRy, tRz, vxa, vya, vxb, vyb, Val(GRAD))
        end
        u += ue
        GRAD && (g += ge)
        p += pe
    end
    # rotate back with the -1/(4pi) factor
    c = -T(ONE_OVER_4π)
    R = SMatrix{3,3,T,9}(nx[1], nx[2], nx[3], ny[1], ny[2], ny[3], nz[1], nz[2], nz[3])
    u_out = c * (R * u)
    g_out = GRAD ? c * (R * g * transpose(R)) : zero(SMatrix{3,3,T,9})
    return u_out, g_out, c * p                             # potential shares the edge prelims
end


# Bound-vortex (straight vortex segment) velocity per unit circulation, with the
# regularization family chosen at compile time: 1 Vatistas n=2, 2 compact
# support, 3 Gaussian / Lamb-Oseen. r1, r2 are the segment endpoints relative
# to the target.
@inline function _elem_bound_vortex_velocity(r1::SVector{3,T}, r2::SVector{3,T},
        core_size::T, ::Val{REG}) where {T,REG}
    nr1 = sqrt(r1[1]*r1[1] + r1[2]*r1[2] + r1[3]*r1[3])
    nr2 = sqrt(r2[1]*r2[1] + r2[2]*r2[2] + r2[3]*r2[3])
    if nr1 < 5*eps(T) || nr2 < 5*eps(T)
        return zero(SVector{3,T})
    end
    num = SVector{3,T}(r1[2]*r2[3] - r1[3]*r2[2],
                       r1[3]*r2[1] - r1[1]*r2[3],
                       r1[1]*r2[2] - r1[2]*r2[1])
    r0 = r1 - r2
    dotrixrj = num[1]*num[1] + num[2]*num[2] + num[3]*num[3]   # A = |r1×r2|²
    r0sqr = r0[1]*r0[1] + r0[2]*r0[2] + r0[3]*r0[3]            # B = |r0|²
    rh = r1/nr1 - r2/nr2
    rijdothat = r0[1]*rh[1] + r0[2]*rh[2] + r0[3]*rh[3]
    if REG == 2
        # compact support:
        # 1/h² → 1/(h² + δ(h)), δ = (h-rc)² inside support, 0 beyond;
        # D = A + δB
        h = sqrt(dotrixrj / r0sqr)
        D = h < core_size ?
            dotrixrj + (h - core_size)*(h - core_size) * r0sqr : dotrixrj
        return num * rijdothat / D / (4*T(pi))
    elseif REG == 3
        # Gaussian / Lamb-Oseen:
        # u = c*q*g(h)/(4π A), g = 1 - exp(-h²/2rc²); evaluated as
        # g/A = (g/x²)/(B rc²), x² = (h/rc)², exact h → 0 limit
        x2 = dotrixrj / (r0sqr * core_size * core_size)
        gscaled = x2 < T(1e-12) ? T(0.5) : T(-expm1(-x2/2) / x2)
        return num * rijdothat * gscaled / (r0sqr * core_size * core_size) / (4*T(pi))
    else
        # Vatistas n=2
        rc4 = core_size*core_size*core_size*core_size
        return num * rijdothat / sqrt(dotrixrj*dotrixrj + rc4*r0sqr*r0sqr) / (4*T(pi))
    end
end

# Bound-vortex velocity gradient per unit circulation for the same families,
# written as u = c q/(4π D) with c = r1×r2, q = s·(r̂1 − r̂2), s = r1 − r2 and a
# family-specific denominator D(A, B), A = |c|², B = |s|²; ∇D = κ ∇A, ∇A = 2 s×c.
@inline function _elem_bound_vortex_gradient(r1::SVector{3,T}, r2::SVector{3,T},
        core_size::T, ::Val{REG}) where {T,REG}
    nr1 = sqrt(r1[1]*r1[1] + r1[2]*r1[2] + r1[3]*r1[3])
    nr2 = sqrt(r2[1]*r2[1] + r2[2]*r2[2] + r2[3]*r2[3])
    if nr1 < 5*eps(T) || nr2 < 5*eps(T)
        return zero(SMatrix{3,3,T,9})
    end
    c = SVector{3,T}(r1[2]*r2[3] - r1[3]*r2[2],
                     r1[3]*r2[1] - r1[1]*r2[3],
                     r1[1]*r2[2] - r1[2]*r2[1])
    s = r1 - r2
    A = c[1]*c[1] + c[2]*c[2] + c[3]*c[3]
    B = s[1]*s[1] + s[2]*s[2] + s[3]*s[3]
    rh = r1/nr1 - r2/nr2
    q = s[1]*rh[1] + s[2]*rh[2] + s[3]*rh[3]
    if REG == 2
        # compact support
        B == zero(B) && return zero(SMatrix{3,3,T,9})
        h = sqrt(A / B)
        if h < core_size
            D = A + (h - core_size)*(h - core_size) * B
            # κ = 2 - rc/h; h → 0 clamp keeps κ finite where ∇A → 0 anyway
            kappa = 2 - core_size / max(h, eps(T)*core_size)
        else
            D = A
            kappa = one(T)
        end
        D == zero(D) && return zero(SMatrix{3,3,T,9})
    elseif REG == 3
        # Gaussian
        B == zero(B) && return zero(SMatrix{3,3,T,9})
        x2 = A / (B * core_size * core_size)               # (h/rc)²
        if x2 < T(1e-12)
            # series limits: D → 2 B rc², κ → 1/2
            D = 2 * B * core_size * core_size
            kappa = T(0.5)
        else
            gg = T(-expm1(-x2/2))
            D = A / gg
            kappa = (1 - x2 * exp(-x2/2) / (2*gg)) / gg
        end
    else
        # Vatistas n=2
        rc4 = core_size*core_size*core_size*core_size
        D = sqrt(A*A + rc4*B*B)
        D == zero(D) && return zero(SMatrix{3,3,T,9})
        kappa = A / D
    end
    sxc = SVector{3,T}(s[2]*c[3] - s[3]*c[2],
                       s[3]*c[1] - s[1]*c[3],
                       s[1]*c[2] - s[2]*c[1])
    dD_coeff = kappa * (2 * sxc)
    r1hat = r1 / nr1
    r2hat = r2 / nr2
    # dq_coeff = -((I - r1hat r1hat^T) s)/nr1 + ((I - r2hat r2hat^T) s)/nr2

    r1hs = r1hat[1]*s[1] + r1hat[2]*s[2] + r1hat[3]*s[3]
    r2hs = r2hat[1]*s[1] + r2hat[2]*s[2] + r2hat[3]*s[3]
    dq_coeff = -(s - r1hat*r1hs)/nr1 + (s - r2hat*r2hs)/nr2
    dc_dx = SMatrix{3,3,T,9}(zero(T), -s[3], s[2],
                             s[3], zero(T), -s[1],
                             -s[2], s[1], zero(T))
    df_coeff = dq_coeff/D - q*dD_coeff/(D*D)
    return T(ONE_OVER_4π) * (dc_dx * (q/D) + c * transpose(df_coeff))
end


# self-pair detection: relative tolerance vs sqrt(A)
@inline function _elem_is_self_pair(target::SVector{3,T}, control_point::SVector{3,T},
        v1::SVector{3,T}, v2::SVector{3,T}, v3::SVector{3,T}) where T
    e1 = v2 - v1
    e2 = v3 - v1
    nxc = e1[2]*e2[3] - e1[3]*e2[2]
    nyc = e1[3]*e2[1] - e1[1]*e2[3]
    nzc = e1[1]*e2[2] - e1[2]*e2[1]
    area = T(0.5) * sqrt(nxc*nxc + nyc*nyc + nzc*nzc)
    d = target - control_point
    return (d[1]*d[1] + d[2]*d[2] + d[3]*d[3]) < _elem_guard_tol(T)^2 * area
end

@inline function _elem_panel_normal(v1::SVector{3,T}, v2::SVector{3,T}, v3::SVector{3,T}) where T
    e1 = v2 - v1
    e2 = v3 - v1
    nxc = e1[2]*e2[3] - e1[3]*e2[2]
    nyc = e1[3]*e2[1] - e1[1]*e2[3]
    nzc = e1[1]*e2[2] - e1[2]*e2[1]
    inv_n = one(T) / sqrt(nxc*nxc + nyc*nyc + nzc*nzc)
    return SVector{3,T}(nxc*inv_n, nyc*inv_n, nzc*inv_n)
end

# one planar triangle carrying a constant source (tag 1), a constant doublet
# (tag 2) or both (tag 5, source s1 and doublet s2): velocity, gradient and
# potential, with the on-panel limits at the centroid. There the source
# velocity takes its exterior surface limit (tag 1: u + (σ/2 − u·n) n; tag 5:
# u + σ/2 n), the doublet velocity is already the principal value, the
# gradient is zeroed and the doublet potential is μ/2.
@inline function _elem_tri_panel_pair(target::SVector{3,T}, tag::Int,
        v1::SVector{3,T}, v2::SVector{3,T}, v3::SVector{3,T}, s1::T, s2::T,
        ::Val{GRAD}) where {T,GRAD}
    u = zero(SVector{3,T})
    g = zero(SMatrix{3,3,T,9})
    p = zero(T)
    if tag == 1 || tag == 5
        us, gs, ps = _elem_tri_source_doublet(target, v1, v2, v3, zero(T), Val(false), Val(GRAD))
        u += s1 * us
        GRAD && (g += s1 * gs)
        p += s1 * ps
    end
    if tag == 2 || tag == 5
        mu = tag == 2 ? s1 : s2
        ud, gd, pd = _elem_tri_source_doublet(target, v1, v2, v3, zero(T), Val(true), Val(GRAD))
        u += mu * ud
        GRAD && (g += mu * gd)
        p += mu * pd
    end
    control_point = (v1 + v2 + v3) * T(0.3333333333333333)
    if _elem_is_self_pair(target, control_point, v1, v2, v3)
        n_gt = _elem_panel_normal(v1, v2, v3)
        if tag == 1
            un = u[1]*n_gt[1] + u[2]*n_gt[2] + u[3]*n_gt[3]
            u = u + (s1*T(0.5) - un) * n_gt
        elseif tag == 5
            u = u + s1*T(0.5) * n_gt
        end
        g = zero(SMatrix{3,3,T,9})
        if tag == 2
            p = s1 * T(0.5)
        elseif tag == 5
            p += s2 * T(0.5)
        end
    end
    return u, g, p
end
