"""
Designed for source and dipole panels.
"""
function induced(target, vertices, normal, strength, centroid, kernel::Union{Type{Panel{Source}}, Type{Panel{Dipole}}, Type{Panel{SourceDipole}}}, derivatives_switch=DerivativesSwitch(true,true,true))

    Rprime, Rxprime, Ryprime, Rzprime = rotate_to_panel(vertices, normal)

    potential, velocity, hessian = _induced(target, vertices, normal, strength, centroid, kernel, Rprime, Rxprime, Ryprime, Rzprime, derivatives_switch)

    return potential, velocity, hessian
end

"""
Designed for use with a vortex ring panel.
"""
function induced(target, vertices, normal, kernel, derivatives_switch=DerivativesSwitch(true,false,true,true))

    potential, velocity, hessian = _induced(target, vertices, normal, strength, centroid, kernel, derivatives_switch)

    return potential, velocity, hessian
end

@inline function get_vertices_xyR(Rxyprime::SVector{3,TF}, vertices::SVector{NS,<:SVector}, centroid) where {NS,TF}
    vec1 = SVector{NS,TF}(
        Rxyprime' * (vertices[i] - centroid) for i in 1:NS
    )
    return SVector{NS+1,TF}(vec1..., vec1[1])
end

@inline function get_dxys(vertices_dxyR::SVector{NSP1,TF}) where {NSP1,TF}
    return SVector{NSP1-1,TF}(
        vertices_dxyR[i+1] - vertices_dxyR[i] for i in 1:NSP1-1
    )
end

@inline function get_ds(dxs::SVector{NS,TF}, dys::SVector{NS,TF}) where {TF,NS}
    return SVector{NS,TF}(
        sqrt(dxs[i]^2 + dys[i]^2) for i in 1:NS
    )
end

@inline function get_ms(dxs::SVector{NS,TF}, dys::SVector{NS,TF}) where {NS,TF}
    return SVector{NS,TF}(
        dys[i] / dxs[i] for i in 1:NS
    )
end

@inline function get_rxys(target_Rxy::TFR, vertices_xyR::SVector{NSP1,TF}) where {TFR,NSP1,TF}
    return SVector{NSP1,promote_type(TFR,TF)}(
        target_Rxy - vertices_xyR[i] for i in 1:NSP1
    )
end

@inline function get_es_hs(rxs::SVector{NSP1,TF}, rys::SVector{NSP1,TF}, dz2) where {NSP1,TF}
    es = SVector{NSP1,TF}(
        rxs[i]^2 + dz2 for i in 1:NSP1
    )

    hs = SVector{NSP1,TF}(
        rxs[i] * rys[i] for i in 1:NSP1
    )

    return es, hs
end

@inline function get_rs(es::SVector{NSP1,TFT}, target_Ry::TFT, vertices_yR::SVector{NSP1,TF}) where {NSP1,TF,TFT}
    return SVector{NSP1,promote_type(TF,TFT)}(
        sqrt(es[i] + (vertices_yR[i] - target_Ry)^2) for i in 1:NSP1
    )
end

@inline function get_rxy_over_rs(rxys::SVector{NSP1,TF}, rs::SVector{NSP1,TF}) where {NSP1,TF}
    return SVector{NSP1,TF}(
        rxys[i] / rs[i] for i in 1:NSP1
    )
end

@inline function rotate_to_panel(vertices, normal)
    # rotate into panel frame
    new_z = normal
    new_x = vertices[3] - vertices[1]
    new_x /= norm(new_x)
    new_y = cross(new_z, new_x)
    R = hcat(new_x, new_y, new_z)
    Rprime = R'

    TFP = eltype(vertices[1])
    Rxprime = SVector{3,TFP}(Rprime[1,1],Rprime[1,2],Rprime[1,3])
    Ryprime = SVector{3,TFP}(Rprime[2,1],Rprime[2,2],Rprime[2,3])
    Rzprime = SVector{3,TFP}(Rprime[2,1],Rprime[2,2],Rprime[2,3])

    return R, Rxprime, Ryprime, Rzprime
end

@inline function source_dipole_preliminaries(target, vertices, normal, strength, centroid, TFT, TFP, R, Rxprime, Ryprime)

    # promote types
    TF = promote_type(TFT,TFP)

    # rotate target
    target_R = R' * (target-centroid)

    # induced potential, velocity, gradient
    potential = zero(TF)
    velocity = @SVector zeros(TF,3)
    hessian = @SMatrix zeros(TF,3,3)

    # intermediate quantities
    vertices_xR = get_vertices_xyR(Rxprime, vertices, centroid)
    vertices_yR = get_vertices_xyR(Ryprime, vertices, centroid)

    dxs = get_dxys(vertices_xR)
    dys = get_dxys(vertices_yR)

    ds = get_ds(dxs, dys)
    ms = get_ms(dxs, dys)

    dz = target_R[3]# - Rzprime' * centroid
    iszero(dz) && (dz = eps())

    dz2 = dz^2

    rxs = get_rxys(target_R[1], vertices_xR)
    rys = get_rxys(target_R[2], vertices_yR)

    es, hs = get_es_hs(rxs, rys, dz2)

    rs = get_rs(es, target_R[2], vertices_yR)
    rx_over_rs = get_rxy_over_rs(rxs, rs)
    ry_over_rs = get_rxy_over_rs(rys, rs)

    return strength, TF, potential, velocity, hessian, dxs, dys, ds, ms, dz, dz2, rxs, rys, es, hs, rs, rx_over_rs, ry_over_rs
end

#####
##### constant source
#####

@inline function _induced(target::AbstractVector{TFT}, vertices::SVector{NS,<:SVector}, normal, strength, centroid, kernel::Type{Panel{Source}}, R, Rxprime, Ryprime, Rzprime, derivatives_switch::DerivativesSwitch{PS,GS,HS}) where {TFT,NS,PS,GS,HS}

    # prelimilary computations
    TFP = eltype(vertices[1])
    strength, TF, potential, velocity, hessian, dxs, dys, ds, ms, dz, dz2, rxs, rys, es, hs, rs, rx_over_rs, ry_over_rs = source_dipole_preliminaries(target, vertices, normal, strength, centroid, TFT, TFP, R, Rxprime, Ryprime)

    # loop over side contributions
    for i in 1:NS
        # intermediate quantities
        # singularity if probing on a side [ SOLVED ]; (easy way out is to perturb the evaluation point slightly)
        num = rs[i] + rs[i+1] - ds[i]
        iszero(num) && (num += eps())
        log_term = log(num / (rs[i] + rs[i+1] + ds[i]))
        # singularity if d_z=0 [ SOLVED ] or probing at a vertex [ SOLVED ]; (easy way out is to perturb the evaluation point slightly)
        ri = rs[i]
        iszero(ri) && (ri += eps())
        rip1 = rs[i+1]
        iszero(rip1) && (rip1 += eps())
        tan_term = atan((ms[i] * es[i] - hs[i]) / dz / ri) - atan((ms[i] * es[i+1] - hs[i+1]) / dz / rip1)
        dx = dxs[i]
        dy = dys[i]

        if PS# && !isinf(ms[i])
            potential += (rxs[i] * dy - rys[i] * dx) / ds[i] * log_term
            potential += dz * tan_term
        end

        if GS
            velocity += SVector{3}(
                dy / ds[i] * log_term,
                -dx / ds[i] * log_term,
                tan_term
            )
        end

        if HS
            # intermediate values
            d2 = ds[i]^2
            r_plus_rp1 = rs[i] + rs[i+1]
            r_plus_rp1_2 = r_plus_rp1^2
            r_times_rp1 = rs[i] * rs[i+1]

            rho = r_times_rp1 + rxs[i] * rxs[i+1] + rys[i] * rys[i+1] + dz2
            lambda = rxs[i] * rys[i+1] - rxs[i+1] * rys[i]

            val1 = r_plus_rp1_2 - d2
            val2 = rx_over_rs[i] + rx_over_rs[i+1]
            val3 = ry_over_rs[i] + ry_over_rs[i+1]
            val4 = r_plus_rp1 / (r_times_rp1 * rho)

            # construct hessian
            phi_xx = 2 * dy / val1 * val2
            phi_xy = -2 * dx / val1 * val2
            phi_xz = dz * dy * val4
            phi_yy = -2 * dx / val1 * val3
            phi_yz = -dz * dx * val4
            phi_zz = lambda * val4
            hessian += SMatrix{3,3,eltype(hessian),9}(
                phi_xx, phi_xy, phi_xz,
                phi_xy, phi_yy, phi_yz,
                phi_xz, phi_yz, phi_zz
            )
        end

    end

    if PS
        potential *= strength[1] * FastMultipole.ONE_OVER_4π
    end
    if GS
        velocity = -strength[1] * FastMultipole.ONE_OVER_4π * R * velocity
    end
    if HS
        hessian = -strength[1] * FastMultipole.ONE_OVER_4π * R * hessian * R'
    end

    return potential, velocity, hessian
end

#@inline kernel_multiplicity(::ConstantSource) = 1

#####
##### constant normal doublet
#####

function _induced(target::AbstractVector{TFT}, vertices::SVector{NS,<:SVector}, normal, strength, centroid, kernel::Type{Panel{Dipole}}, R, Rxprime, Ryprime, Rzprime, derivatives_switch::DerivativesSwitch{PS,GS,HS}) where {TFT,NS,PS,GS,HS}

    # prelimilary computations
    TFP = eltype(vertices[1])
    strength, TF, potential, velocity, hessian, dxs, dys, ds, ms, dz, dz2, rxs, rys, es, hs, rs, rx_over_rs, ry_over_rs = source_dipole_preliminaries(target, vertices, normal, strength, centroid, TFT, TFP, R, Rxprime, Ryprime)

    # loop over side contributions
    for i in 1:NS
        # intermediate quantities
        # singularity if probing on a side [ SOLVED ]; (easy way out is to perturb the evaluation point slightly)
        num = rs[i] + rs[i+1] - ds[i]
        iszero(num) && (num += eps())
        log_term = log(num / (rs[i] + rs[i+1] + ds[i]))
        # singularity if d_z=0 [ SOLVED ] or probing at a vertex [ SOLVED ]; (easy way out is to perturb the evaluation point slightly)
        ri = rs[i]
        iszero(ri) && (ri += eps())
        rip1 = rs[i+1]
        iszero(rip1) && (rip1 += eps())
        tan_term = atan((ms[i] * es[i] - hs[i]) / dz / ri) - atan((ms[i] * es[i+1] - hs[i+1]) / dz / rip1)

        if PS# && !isinf(ms[i])
            potential -= tan_term
        end

        if GS
            r_plus_rp1 = rs[i] + rs[i+1]
            r_times_rp1 = rs[i] * rs[i+1]
            rho = r_times_rp1 + rxs[i] * rxs[i+1] + rys[i] * rys[i+1] + dz2
            lambda = rxs[i] * rys[i+1] - rxs[i+1] * rys[i]
            val4 = r_plus_rp1 / (r_times_rp1 * rho)
            velocity += SVector{3}(
                dz * dys[i] * val4,
                -dz * dxs[i] * val4,
                lambda * val4
            )
        end

        if HS
            # intermediate values
            d2 = ds[i]^2
            r_plus_rp1 = rs[i] + rs[i+1]
            r_plus_rp1_2 = r_plus_rp1^2
            r_times_rp1 = rs[i] * rs[i+1]

            rho = r_times_rp1 + rxs[i] * rxs[i+1] + rys[i] * rys[i+1] + dz2
            lambda = rxs[i] * rys[i+1] - rxs[i+1] * rys[i]

            val1 = r_times_rp1 * r_plus_rp1_2 + rho * rs[i+1]^2
            val1 /= rho * rs[i] * r_plus_rp1
            val2 = r_times_rp1 * r_plus_rp1_2 + rho * rs[i]^2
            val2 /= rho * rs[i+1] * r_plus_rp1
            val3 = r_plus_rp1 / (rho * r_times_rp1^2)

            # construct hessian
            psi_xx = dz * dys[i] * val3 * (rxs[i] * val1 + rxs[i+1] * val2)
            psi_xy = dz * dys[i] * val3 * (rys[i] * val1 + rys[i+1] * val2)
            psi_yy = -dz * dxs[i] * val3 * (rys[i] * val1 + rys[i+1] * val2)
            val4 = r_plus_rp1_2 / rho
            val5 = (rs[i]^2 - r_times_rp1 + rs[i+1]^2) / r_times_rp1
            val6 = dz * (val4 + val5)
            psi_zz = lambda * val3 * val6
            val7 = r_times_rp1 - dz * val6
            val8 = val3 * val7
            psi_xz = -dys[i] * val8
            psi_yz = dxs[i] * val8
            hessian += SMatrix{3,3,eltype(hessian),9}(
                psi_xx, psi_xy, psi_xz,
                psi_xy, psi_yy, psi_yz,
                psi_xz, psi_yz, psi_zz
            )
        end

    end

    if PS
        potential *= strength[1] * FastMultipole.ONE_OVER_4π
    end
    if GS
        velocity = strength[1] * FastMultipole.ONE_OVER_4π * R * velocity
    end
    if HS
        hessian = -strength[1] * FastMultipole.ONE_OVER_4π * R * hessian * R'
    end

    # dx = target - panel.control_point
    # if isapprox(dx' * dx, 0.0; atol=2*eps())
    #     velocity /= 2
    #     hessian /= 2
    # end

    return potential, velocity, hessian
end

#@inline kernel_multiplicity(::ConstantNormalDoublet) = 1

#####
##### constant source plus constant normal doublet
#####

function _induced(target::AbstractVector{TFT}, vertices::SVector{NS,<:SVector}, normal, strength, centroid, kernel::Type{Panel{SourceDipole}}, R, Rxprime, Ryprime, Rzprime, derivatives_switch::DerivativesSwitch{PS,GS,HS}) where {TFT,NS,PS,GS,HS}
    # prelimilary computations;
    # note that strength[1] is the source strength and strength[2] is the dipole strength
    TFP = eltype(vertices[1])
    strength, TF, potential, velocity, hessian, dxs, dys, ds, ms, dz, dz2, rxs, rys, es, hs, rs, rx_over_rs, ry_over_rs = source_dipole_preliminaries(target, vertices, normal, strength, centroid, TFT, TFP, R, Rxprime, Ryprime)

    # loop over side contributions
    for i in 1:NS
        # intermediate quantities
        # singularity if probing on a side [ SOLVED ]; (easy way out is to perturb the evaluation point slightly)
        num = rs[i] + rs[i+1] - ds[i]
        iszero(num) && (num += eps())
        log_term = log(num / (rs[i] + rs[i+1] + ds[i]))
        # singularity if d_z=0 [ SOLVED ] or probing at a vertex [ SOLVED ]; (easy way out is to perturb the evaluation point slightly)
        ri = rs[i]
        iszero(ri) && (ri += eps())
        rip1 = rs[i+1]
        iszero(rip1) && (rip1 += eps())
        tan_term = atan((ms[i] * es[i] - hs[i]) / dz / ri) - atan((ms[i] * es[i+1] - hs[i+1]) / dz / rip1)

        if PS# && !isinf(ms[i])
            potential += strength[1] * ((rxs[i] * dys[i] - rys[i] * dxs[i]) / ds[i] * log_term + dz * tan_term)
            potential -= strength[2] * tan_term
        end

        if GS
            r_plus_rp1 = rs[i] + rs[i+1]
            r_times_rp1 = rs[i] * rs[i+1]
            rho = r_times_rp1 + rxs[i] * rxs[i+1] + rys[i] * rys[i+1] + dz2
            lambda = rxs[i] * rys[i+1] - rxs[i+1] * rys[i]
            val4 = r_plus_rp1 / (r_times_rp1 * rho)
            velocity -= strength[1] * SVector{3}(
                dys[i] / ds[i] * log_term,
                -dxs[i] / ds[i] * log_term,
                tan_term
            )
            velocity += strength[2] * SVector{3}(
                dz * dys[i] * val4,
                -dz * dxs[i] * val4,
                lambda * val4
            )
        end

        if HS
            # intermediate values
            d2 = ds[i]^2
            r_plus_rp1 = rs[i] + rs[i+1]
            r_plus_rp1_2 = r_plus_rp1^2
            r_times_rp1 = rs[i] * rs[i+1]

            rho = r_times_rp1 + rxs[i] * rxs[i+1] + rys[i] * rys[i+1] + dz2
            lambda = rxs[i] * rys[i+1] - rxs[i+1] * rys[i]

            val1 = r_plus_rp1_2 - d2
            val2 = rx_over_rs[i] + rx_over_rs[i+1]
            val2 *= strength[1]
            val3 = ry_over_rs[i] + ry_over_rs[i+1]
            val3 *= strength[1]
            val4 = r_plus_rp1 / (r_times_rp1 * rho)
            val4 *= strength[1]

            # construct hessian
            phi_xx = 2 * dys[i] / val1 * val2
            phi_xy = -2 * dxs[i] / val1 * val2
            phi_xz = dz * dys[i] * val4
            phi_yy = -2 * dxs[i] / val1 * val3
            phi_yz = -dz * dxs[i] * val4
            phi_zz = lambda * val4
            hessian += SMatrix{3,3,eltype(hessian),9}(
                phi_xx, phi_xy, phi_xz,
                phi_xy, phi_yy, phi_yz,
                phi_xz, phi_yz, phi_zz
            )

            val1 = r_times_rp1 * r_plus_rp1_2 + rho * rs[i+1]^2
            val1 /= rho * rs[i] * r_plus_rp1
            val2 = r_times_rp1 * r_plus_rp1_2 + rho * rs[i]^2
            val2 /= rho * rs[i+1] * r_plus_rp1
            val3 = r_plus_rp1 / (rho * r_times_rp1^2)

            # construct hessian
            psi_xx = dz * dys[i] * val3 * (rxs[i] * val1 + rxs[i+1] * val2)
            psi_xy = dz * dys[i] * val3 * (rys[i] * val1 + rys[i+1] * val2)
            psi_yy = -dz * dxs[i] * val3 * (rys[i] * val1 + rys[i+1] * val2)
            val4 = r_plus_rp1_2 / rho
            val5 = (rs[i]^2 - r_times_rp1 + rs[i+1]^2) / r_times_rp1
            val6 = dz * (val4 + val5)
            psi_zz = lambda * val3 * val6
            val7 = r_times_rp1 - dz * val6
            val8 = val3 * val7
            psi_xz = -dys[i] * val8
            psi_yz = dxs[i] * val8
            hessian += strength[2] * SMatrix{3,3,eltype(hessian),9}(
                psi_xx, psi_xy, psi_xz,
                psi_xy, psi_yy, psi_yz,
                psi_xz, psi_yz, psi_zz
            )
        end

    end

    PS && (potential *= FastMultipole.ONE_OVER_4π)
    GS && (velocity = FastMultipole.ONE_OVER_4π * R * velocity)
    HS && (hessian = -FastMultipole.ONE_OVER_4π * R * hessian * R')

    return potential, velocity, hessian
end

#@inline kernel_multiplicity(::ConstantSourceNormalDoublet) = 2

#####
##### vortex ring panel
#####

#=
struct VortexRing <: AbstractUnrotatedKernel end

function _induced(target::AbstractVector{TFT}, vertices, normal, strength, centroid, kernel::VortexRing, derivatives_switch::DerivativesSwitch{PS,<:Any,GS,HS}) where {TFT,TFP,NS,PS,GS,HS}
    TF = promote_type(TFT,TFP)
    corner_vectors = SVector{NS,SVector{3,TF}}(corner - target for corner in vertices)
    velocity = zero(SVector{3,TF})
    gradient = zero(SMatrix{3,3,TF,9})

    # finite core settinHS
    finite_core = false
    core_size = 1e-3

    # evaluate velocity/gradient
    for i in 1:NS-1
        r1 = vertices[i] - target
        r2 = vertices[i+1] - target

        # parameters
        r1norm = sqrt(r1'*r1)
        r2norm = sqrt(r2'*r2)
        r1normr2norm = r1norm*r2norm
        rcross = cross(r1, r2)
        rdot = dot(r1, r2)
        FastMultipole.ONE_OVER_4π = 1/4/pi

        if GS
            # velocity
            v = _bound_vortex_velocity(r1norm, r2norm, r1normr2norm, rcross, rdot, finite_core, core_size; epsilon=10*eps())
            velocity += v
        end
        if HS
            # velocity gradient
            g = _bound_vortex_gradient(r1, r2, r1norm, r2norm, r1normr2norm, rcross, rdot; epsilon=10*eps())
            gradient += g
        end
    end

    # wrap vertex
    r1 = vertices[NS] - target
    r2 = vertices[1] - target

    # parameters
    r1norm = sqrt(r1'*r1)
    r2norm = sqrt(r2'*r2)
    r1normr2norm = r1norm*r2norm
    rcross = cross(r1, r2)
    rdot = dot(r1, r2)
    FastMultipole.ONE_OVER_4π = 1/4/pi

    if GS
        # velocity
        v = _bound_vortex_velocity(r1norm, r2norm, r1normr2norm, rcross, rdot, finite_core, core_size; epsilon=10*eps())
        velocity += v
    end
    if HS
        # velocity gradient
        g = _bound_vortex_gradient(r1, r2, r1norm, r2norm, r1normr2norm, rcross, rdot; epsilon=10*eps())
        gradient += g
    end

    return zero(TF), velocity, gradient
end

function _bound_vortex_velocity(r1norm::TF, r2norm, r1normr2norm, rcross, rdot, finite_core::Bool, core_size::Number; epsilon=10*eps()) where TF
    # check if evaluation point is colinear with the bound vortex
    if norm(rcross) < epsilon # colinear if true
        if isapprox(rdot, -r1normr2norm; atol=epsilon) # at the midpoint, so return zero
            return zero(SVector{3,TF})
        elseif rdot <= 0.0 && finite_core # coincident with the filament so use the finite core model
            r1s, r2s, εs = r1norm^2, r2norm^2, core_size^2
            f1 = rcross/(r1s*r2s - rdot^2 + εs*(r1s + r2s - 2*r1normr2norm))
            f2 = (r1s - rdot)/sqrt(r1s + εs) + (r2s - rdot)/sqrt(r2s + εs)
            velocity = (f1*f2)/(4*pi)
            return velocity
        end
    end

    # otherwise, use singular kernel
    f1 = rcross/(r1normr2norm + rdot)
    f2 = (1/r1norm + 1/r2norm)

    velocity = (f1*f2) * FastMultipole.ONE_OVER_4π

    return velocity
end

function _bound_vortex_gradient(r1, r2, r1norm::TF, r2norm, r1normr2norm, rcross, rdot; epsilon=10*eps()) where TF
    # zeta
    t1 = 1/(r1norm*r2norm + rdot)
    t2 = 1/r1norm + 1/r2norm
    z = t1*t2*FastMultipole.ONE_OVER_4π

    # zeta gradient
    r1norm3 = r1norm^3
    r2norm3 = r2norm^3
    t4 = SVector{3,TF}(r1[i]/r1norm^3 + r2[i]/r2norm^3 for i in 1:3)
    t5 = SVector{3,TF}(r1norm/r2norm*r2[i] + r2norm/r1norm*r1[i] + r1[i] + r2[i] for i in 1:3)
    zgrad = FastMultipole.ONE_OVER_4π*(-t1*t4 - t2*t5*t1^2)

    # Omega
    o = cross(r1,r2)

    # Omega gradient
    ograd = SMatrix{3,3,TF,9}(
        0.0,# 1,1
        r1[3]-r2[3], # 2,1
        r2[2]-r1[2], # 3,1
        r2[3]-r1[3], # 1,2
        0.0, # 2,2
        r1[1]-r2[1], # 3,2
        r1[2]-r2[2], # 1,3
        r2[1]-r1[1], # 2,3
        0.0 # 3,3
    )
    gradient = transpose(zgrad * transpose(o)) + z * ograd

    return gradient
end

@inline kernel_multiplicity(::VortexRing) = 1
=#

function U_constant_vortexsheet(nodes::Arr1, panel,
                                gammat::Number, gammao::Number,
                                targets::Arr2, out::Arr3;
                                dot_with=nothing,
                                cutoff=1e-14, offset=1e-8
                              ) where{T1, Arr1<:AbstractArray{T1,2},
                                      T2, Arr2<:AbstractArray{T2,2},
                                      T3, Arr3<:AbstractArray{T3}}

    nt = size(targets, 2)                   # Number of targets
    no = dot_with!=nothing ? length(out) : size(out, 2) # Number of outputs
    nn = length(panel)                      # Number of nodes

    if no!=nt
        error("Invalid `out` argument. Expected size $(nt), got $(no).")
    end

    # @warn("Sheet thickness has been hardcoded!")

    #=
        TODO
        * [ ] Implement efficient H00, H10, and H01 formulas
    =#

    # Tangent, oblique, and normal vectors
    #t1, t2, t3 = gt._calc_t1(nodes, panel), gt._calc_t2(nodes, panel), gt._calc_t3(nodes, panel)
    #o1, o2, o3 = gt._calc_o1(nodes, panel), gt._calc_o2(nodes, panel), gt._calc_o3(nodes, panel)
    #n1, n2, n3 = gt._calc_n1(nodes, panel), gt._calc_n2(nodes, panel), gt._calc_n3(nodes, panel)
    t1, t2, t3 = 1.0, 0.0, 0.0
    o1, o2, o3 = 0.0, 1.0, 0.0
    n1, n2, n3 = 0.0, 0.0, 1.0

    @inbounds p1i = panel[1]                    # Index of first vertex

    # V1, V2, V3 = zero(T3), zero(T3), zero(T3)

    # Iterate over targets
    for ti in 1:nt

        # V1 *= 0
        # V2 *= 0
        # V3 *= 0

        V1, V2, V3 = zero(T3), zero(T3), zero(T3)

        # Projection of target onto plane of the panel
        @inbounds begin
            z = n1*(targets[1,ti]-nodes[1,p1i]) + n2*(targets[2,ti]-nodes[2,p1i]) + n3*(targets[3,ti]-nodes[3,p1i])
            px1 = targets[1,ti] - z*n1
            px2 = targets[2,ti] - z*n2
            px3 = targets[3,ti] - z*n3
        end

        # Iterate over triangles of integration
        # @simd
        for Si in 1:nn


            @inbounds begin
                # Indices of first and second vertices of this triangle
                pi = panel[Si]
                pip1 = Si == nn ? panel[1] : panel[Si+1]

                # Vertices in the coordinate system of integration
                q11 = nodes[1, pi] - px1
                q12 = nodes[2, pi] - px2
                q13 = nodes[3, pi] - px3
                q21 = nodes[1, pip1] - px1
                q22 = nodes[2, pip1] - px2
                q23 = nodes[3, pip1] - px3
            end

            sqrtq2mq1 = sqrt((q21 - q11)^2 + (q22 - q12)^2 + (q23 - q13)^2)

            # Axes of coordinate system for integration
            c21 = (q21 - q11) / sqrtq2mq1
            c22 = (q22 - q12) / sqrtq2mq1
            c23 = (q23 - q13) / sqrtq2mq1
            c11 = c22*n3 - c23*n2
            c12 = c23*n1 - c21*n3
            c13 = c21*n2 - c22*n1

            # Decompose strength with coordinate system of integration
            a00 = gammat*(t1*c11 + t2*c12 + t3*c13) + gammao*(o1*c11 + o2*c12 + o3*c13)
            b00 = gammat*(t1*c21 + t2*c22 + t3*c23) + gammao*(o1*c21 + o2*c22 + o3*c23)

            # Limits of integration
            a = q11*c11 + q12*c12 + q13*c13
            l1 = q11*c21 + q12*c22 + q13*c23
            l2 = q21*c21 + q22*c22 + q23*c23

            # Regularized height
            h = sqrt(z^2 + offset^2)
            # h = sqrt(z^2 + (1.0*1e-2)^2)
            # h = sqrt(z^2 + (2.5*1e-3)^2)
            # h = sqrt(z^2 + (1.0*1e-8)^2)

            sqrtl1a = sqrt(l1^2 + a^2)
            sqrtl2a = sqrt(l2^2 + a^2)
            sqrtl1ah = sqrt(l1^2 + a^2 + h^2)
            sqrtl2ah = sqrt(l2^2 + a^2 + h^2)
            logh = log(h)

            # Integral terms
            # NOTE: What is the codomain of atan? Should this be atan or atan2?
            H00 =  1/h * atan(a*l2, a^2 + h^2 + h*sqrtl2ah)
            H00 -= 1/h * atan(a*l1, a^2 + h^2 + h*sqrtl1ah)
            H10 =  l2/sqrtl2a * log(sqrtl2ah + sqrtl2a) - log(l2 + sqrtl2ah) - l2*logh/sqrtl2a
            H10 -= l1/sqrtl1a * log(sqrtl1ah + sqrtl1a) - log(l1 + sqrtl1ah) - l1*logh/sqrtl1a
            H01 =  a/sqrtl2a * (logh - log(sqrtl2ah + sqrtl2a))
            H01 -= a/sqrtl1a * (logh - log(sqrtl1ah + sqrtl1a))

            # Avoid zero divided by zero when the projection lays on a vertex
            if abs(a)<=cutoff && (abs(l2)<=cutoff || abs(l1)<=cutoff)
                nothing
            else
                V1 += z*b00*H00*c11 - z*a00*H00*c21 + (b00*H10 - a00*H01)*n1
                V2 += z*b00*H00*c12 - z*a00*H00*c22 + (b00*H10 - a00*H01)*n2
                V3 += z*b00*H00*c13 - z*a00*H00*c23 + (b00*H10 - a00*H01)*n3
            end
        end

        V1 /= 4*pi
        V2 /= 4*pi
        V3 /= 4*pi

        if dot_with!=nothing
            @inbounds out[ti] += V1*dot_with[1,ti] + V2*dot_with[2,ti] + V3*dot_with[3,ti]
        else
            @inbounds out[1, ti] += V1
            @inbounds out[2, ti] += V2
            @inbounds out[3, ti] += V3
        end

    end

end
