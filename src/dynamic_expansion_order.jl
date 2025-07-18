#--- choose expansion order based on error tolerance ---#

function get_P(Δx, Δy, Δz, target_branch, source_branch, i_source_branch, expansions, expansion_order::Int, error_method::ErrorMethod, check_dipole::Val)
    return expansion_order
end

function get_P(Δx, Δy, Δz, target_branch, source_branch, i_source_branch, expansions, ::Dynamic{PMAX,ε}, error_method::Union{EqualSpheres, UnequalSpheres, UnequalBoxes, UniformUnequalSpheres, UniformUnequalBoxes}, check_dipole::Val{CD}) where {PMAX,ε,CD}
    ΔC2 = Δx*Δx+Δy*Δy+Δz*Δz
    r_min, r_max, ρ_min, ρ_max = get_r_ρ(target_branch, source_branch, ΔC2, error_method)

    # monopole term
    multipole_expansion = view(expansions, :, :, :, i_source_branch)
    q_monopole = multipole_expansion[1,1,1]

    # dipole term
    if CD
        q_dipole_x, q_dipole_y, q_dipole_z = dipole_from_multipole(multipole_expansion)
        rx, ry, rz = minimum_distance(source_branch.center, target_branch.center, target_branch.box)
        q_dipole_multipole = abs(q_dipole_x * rx + q_dipole_y * ry + q_dipole_z * rz) / r_min
        rx, ry, rz = closest_corner(source_branch.center, target_branch.center, target_branch.box)
        q_dipole_local = abs(q_dipole_x * rx + q_dipole_y * ry + q_dipole_z * rz) / r_max
    else
        q_dipole_multipole = q_dipole_local = zero(ΔC2)
    end


    return get_P(r_min, r_max, ρ_min, ρ_max, ΔC2, PMAX, abs(q_monopole), q_dipole_multipole, q_dipole_local, ε, error_method)
end

# function get_P(Δx, Δy, Δz, target_branch, source_branch, ::Dynamic{PMAX,ε}, error_method::UniformCubes, ::Val{check_dipole}) where {PMAX,ε,check_dipole}

#     #--- extract fields ---#

#     source_box = source_branch.source_box
#     target_box = target_branch.box
#     source_center = source_branch.source_center
#     target_center = target_branch.center

#     # monopole term
#     q_monopole = source_branch.multipole_expansion[1,1,1]

#     # dipole term
#     if check_dipole
#         q_dipole_x, q_dipole_y, q_dipole_z = dipole_from_multipole(source_branch.multipole_expansion)
#     end

#     #--- multipole preliminary values ---#

#     # choose the closest point in the local branch
#     r_vec = minimum_distance(source_center, target_center, target_box)

#     # length scale
#     iszero(source_box) && (return zero(r_vec[1]))
#     s = (source_box[1] + source_box[2] + source_box[3]) * 0.6666666666666666

#     # determine whether monopole/dipole dominates
#     rx, ry, rz = r_vec
#     r2 = rx*rx + ry*ry + rz*rz
#     r_inv = 1/sqrt(r2)
#     if check_dipole
#         q_dot_r̂ = abs(q_dipole_x * rx + q_dipole_y * ry + q_dipole_z * rz) * r_inv
#         q_dot_r̂_r_inv_2 = q_dot_r̂ * r_inv * r_inv
#     else
#         q_dot_r̂_r_inv_2 = zero(r_inv)
#     end

#     # get polar/azimuth angles for accessing precomputed integrals
#     _, θr, ϕr = cartesian_to_spherical(r_vec)
#     iθ = get_iθ(θr)
#     iϕ = get_iϕ(ϕr)

#     # get s^(P+4)/r^(P+1)
#     s_rinv = s * r_inv
#     # scalar_multipole = s_rinv / s
#     scalar_multipole = r_inv

#     #--- local preliminary values ---#

#     # choose the closest point in the local branch
#     rx, ry, rz = closest_corner(source_center, target_center, target_box)
#     r = sqrt(rx*rx + ry*ry + rz*rz)
#     r_inv = 1/r

#     # determine whether monopole/dipole dominates
#     if check_dipole
#         q_dot_r̂ = abs(q_dipole_x * rx + q_dipole_y * ry + q_dipole_z * rz) * r_inv
#         q_dot_r̂_r_inv = q_dot_r̂ * r_inv
#     else
#         q_dot_r̂_r_inv = zero(rx)
#     end

#     # get polar/azimuth angles for accessing error database
#     s = sum(target_box) * 0.666666666666666
#     Rinv = 1/sqrt(Δx*Δx + Δy*Δy + Δz*Δz)
#     ω = asin(s * 0.5 * Rinv)
#     iω = get_iω(ω)
#     cθ = (Δx*rx + Δy*ry + Δz*rz) * Rinv * r_inv # cos(θ)
#     γ = acos(cθ)
#     iγ = get_iγ(γ)

#     r_over_R = r * Rinv
#     scalar_local = Rinv * r_over_R

#     # prepare recursive computatation of the first term of the summation
#     Q_local = max(abs(q_monopole), q_dot_r̂_r_inv)
#     ε_local_last = LOCAL_INTEGRALS[1,iω,iγ] * scalar_local * Q_local

#     # recurse
#     scalar_local *= r_over_R

#     #--- find minimum P

#     ε_abs = 4.0π * ε

#     for P in 0:PMAX

#         # get effective charge
#         Q_multipole = max(abs(q_monopole), (P+2) * q_dot_r̂_r_inv_2)
#         Q_local = max(abs(q_monopole), (P+2) * q_dot_r̂_r_inv)

#         # if Q_multipole != abs(q_monopole) || Q_local != abs(q_monopole)
#         #     println("<<<< ITS A DIPOLE! >>>>")
#         # end

#         # calaculate error
#         ε_multipole = MULTIPOLE_INTEGRALS[P+1,iθ,iϕ] * scalar_multipole * Q_multipole
#         ε_local = LOCAL_INTEGRALS[P+2,iω,iγ] * scalar_local * Q_local

#         # check if tolerance is reached
#         ε_multipole + ε_local + ε_local_last <= ε_abs && (return P)
#         # if ε_multipole + ε_local + ε_local_last <= ε_abs
#         #     print("found P: ")
#         #     if ε_multipole > ε_local
#         #         print("multipole ")
#         #     else
#         #         print("local ")
#         #     end
#         #     print("error dominates: ")
#         #     @show ε_multipole, ε_local, ε_local_last
#         #     return P
#         # end

#         # recurse
#         scalar_multipole *= s_rinv
#         scalar_local *= r_over_R
#         ε_local_last = ε_local

#     end

#     # unable to satisfy error bounds
#     warn_Pmax()

#     return PMAX
# end

# function get_P(Δx, Δy, Δz, target_branch, source_branch, ::Dynamic{PMAX,ε}, error_method::UniformCubesGradient, ::Val{check_dipole}) where {PMAX,ε,check_dipole}

#     #--- extract fields ---#

#     source_box = source_branch.source_box
#     target_box = target_branch.box
#     source_center = source_branch.source_center
#     target_center = target_branch.center

#     # monopole term
#     q_monopole = source_branch.multipole_expansion[1,1,1]

#     # dipole term
#     if check_dipole
#         q_dipole_x, q_dipole_y, q_dipole_z = dipole_from_multipole(source_branch.multipole_expansion)
#         q_dipole_2 = q_dipole_x * q_dipole_x + q_dipole_y * q_dipole_y + q_dipole_z * q_dipole_z
#     end

#     #--- multipole preliminary values ---#

#     # choose the closest point in the local branch
#     r_vec = minimum_distance(source_center, target_center, target_box)

#     # length scale
#     iszero(source_box) && (return zero(r_vec[1]))
#     s = (source_box[1] + source_box[2] + source_box[3]) * 0.6666666666666666

#     # determine whether monopole/dipole dominates
#     rx, ry, rz = r_vec
#     r2 = rx*rx + ry*ry + rz*rz
#     r_inv_2 = 1/r2
#     r_inv = sqrt(r_inv_2)
#     if check_dipole
#         q_dot_r̂ = abs(q_dipole_x * rx + q_dipole_y * ry + q_dipole_z * rz) * r_inv
#         q_dot_r̂_2 = q_dot_r̂ * q_dot_r̂
#         q_dot_r̂_2_r_inv_2 = q_dot_r̂_2 * r_inv_2
#         r_inv_4_q = r_inv_2 * r_inv_2 * (q_dipole_2 - q_dot_r̂_2)
#     else
#         q_dot_r̂_2_r_inv_2 = zero(r_inv)
#         r_inv_4_q = zero(r_inv_2)
#     end

#     # get polar/azimuth angles for accessing precomputed integrals
#     _, θr, ϕr = cartesian_to_spherical(r_vec)
#     iθ = get_iθ(θr)
#     iϕ = get_iϕ(ϕr)

#     # get s^(P+4)/r^(P+1)
#     s_rinv = s * r_inv
#     # scalar_multipole = s_rinv / s
#     scalar_multipole = r_inv * r_inv

#     #--- local preliminary values ---#

#     # choose the closest point in the local branch
#     rx, ry, rz = closest_corner(source_center, target_center, target_box)
#     r = sqrt(rx*rx + ry*ry + rz*rz)
#     r_inv = 1/r

#     # determine whether monopole/dipole dominates
#     if check_dipole
#         q_dot_r̂ = abs(q_dipole_x * rx + q_dipole_y * ry + q_dipole_z * rz) * r_inv
#         q_dot_r̂_r_inv = q_dot_r̂ * r_inv
#         q_dot_r̂_2_r_inv_2 = q_dot_r̂_r_inv * q_dot_r̂_r_inv
#         q_dipole_2_r_inv_2 = q_dipole_2 * r_inv * r_inv
#     else
#         q_dot_r̂_2_r_inv_2 = zero(rx)
#         q_dipole_2_r_inv_2 = zero(rx)
#     end

#     # get polar/azimuth angles for accessing error database
#     s = sum(target_box) * 0.666666666666666
#     Rinv = 1/sqrt(Δx*Δx + Δy*Δy + Δz*Δz)
#     ω = asin(s * 0.5 * Rinv)
#     iω = get_iω(ω)
#     cθ = (Δx*rx + Δy*ry + Δz*rz) * Rinv * r_inv # cos(θ)
#     γ = acos(cθ)
#     iγ = get_iγ(γ)

#     r_over_R = r * Rinv
#     # scalar_local = Rinv * r_over_R * r_inv
#     scalar_local = Rinv * Rinv

#     # prepare recursive computatation of the first term of the summation
#     ε_local_last = LOCAL_INTEGRALS[1,iω,iγ] * scalar_local

#     # recurse
#     scalar_local *= r_over_R

#     #--- find minimum P

#     ε_abs = 4.0π * ε

#     for P in 0:PMAX

#         # get effective charge
#         Q_multipole = max(abs(q_monopole), sqrt(q_dot_r̂_2_r_inv_2 * (P+3)*(P+3) + r_inv_4_q))
#         Q_local = max(abs(q_monopole), sqrt((P*P-1) * q_dot_r̂_2_r_inv_2 + q_dipole_2_r_inv_2))

#         # calaculate error
#         ε_multipole = MULTIPOLE_INTEGRALS[P+1,iθ,iϕ] * scalar_multipole * Q_multipole * (P+2)
#         ε_local = LOCAL_INTEGRALS[P+2,iω,iγ] * scalar_local * (P+2)

#         # check if tolerance is reached
#         ε_multipole + Q_local * (ε_local + ε_local_last) <= ε_abs && (return P)
#         # if ε_multipole + ε_local + ε_local_last <= ε_abs
#         #     print("found P: ")
#         #     if ε_multipole > ε_local
#         #         print("multipole ")
#         #     else
#         #         print("local ")
#         #     end
#         #     print("error dominates: ")
#         #     @show ε_multipole, ε_local, ε_local_last
#         #     return P
#         # end

#         # recurse
#         scalar_multipole *= s_rinv
#         scalar_local *= r_over_R
#         ε_local_last = ε_local

#     end

#     # unable to satisfy error bounds
#     warn_Pmax()

#     return PMAX
# end

# function get_P(Δx, Δy, Δz, target_branch, source_branch, ::Dynamic{PMAX,ε}, error_method::LambHelmholtzΧGradient, ::Val{check_dipole}) where {PMAX,ε,check_dipole}

#     #--- extract fields ---#

#     # extract fields
#     target_box = local_branch.box
#     source_center = multipole_branch.source_center
#     target_center = local_branch.center
#     source_box = multipole_branch.source_box
#     multipole_expansion = multipole_branch.multipole_expansion

#     # choose the closest point in the local branch
#     r⃗ = minimum_distance(source_center, target_center, target_box)
#     r = norm(r⃗)
#     rinv = 1/r
#     r̂ = r⃗ * rinv

#     # get polar/azimuth angles for accessing error database
#     _, θr, ϕr = cartesian_to_spherical(r̂)
#     iθ = get_iθ_χ(θr)
#     iϕ = get_iϕ_χ(ϕr)

#     # rotate coordinate system
#     R = rotate(r̂)
#     r⃗ = R * r⃗

#     # length scale
#     iszero(source_box) && (return zero(r⃗[1]))
#     s = (source_box[1] + source_box[2] + source_box[3]) * 0.6666666666666666

#     # calculate vector strength of multipole branch
#     ωx, ωy, ωz = vortex_from_multipole(multipole_expansion)
#     ωx, ωy, ωz = R * SVector{3,eltype(source_box)}(ωx, ωy, ωz)

#     # estimate induced vector by approximating a point vortex
#     # at the expansion center
#     vx = ωy * r⃗[3] - ωz * r⃗[2]
#     vy = ωz * r⃗[1] - ωx * r⃗[3]
#     vinv = 1/sqrt(vx*vx + vy*vy)
#     v̂x = vx*vinv
#     v̂y = vy*vinv

#     # initialize values
#     εχ_real = zero(eltype(source_box))
#     εχ_imag = zero(eltype(source_box))
#     s_over_r = s * rinv
#     snm1_over_rnp1 = rinv * rinv # P=0 -> n=1

#     n = 1
#     Iχx_real = MULTIPOLE_INTEGRALS_Χ[1,1,n,iθ,iϕ]
#     Iχx_imag = MULTIPOLE_INTEGRALS_Χ[2,1,n,iθ,iϕ]
#     Iχy_real = MULTIPOLE_INTEGRALS_Χ[1,2,n,iθ,iϕ]
#     Iχy_imag = MULTIPOLE_INTEGRALS_Χ[2,2,n,iθ,iϕ]

#     εχ_real = (Iχx_real * ωx + Iχy_real * ωy) * snm1_over_rnp1
#     εχ_imag = (Iχx_imag * ωx + Iχy_imag * ωy) * snm1_over_rnp1

#     #--- find minimum P ---#

#     ε_abs = 4.0π * ε
#     ε_abs_2 = ε_abs * ε_abs

#     for P in 0:PMAX

#         # uniform vorticity integrals for n=P+2
#         n = P+2
#         Iχx_real = MULTIPOLE_INTEGRALS_Χ[1,1,n,iθ,iϕ]
#         Iχx_imag = MULTIPOLE_INTEGRALS_Χ[2,1,n,iθ,iϕ]
#         Iχy_real = MULTIPOLE_INTEGRALS_Χ[1,2,n,iθ,iϕ]
#         Iχy_imag = MULTIPOLE_INTEGRALS_Χ[2,2,n,iθ,iϕ]
#         snm1_over_rnp1 *= s_over_r

#         # estimated χ
#         χ_real_next = (Iχx_real * ωx + Iχy_real * ωy) * snm1_over_rnp1
#         χ_imag_next = (Iχx_imag * ωx + Iχy_imag * ωy) * snm1_over_rnp1

#         # check if error is below tolerance
#         εχ_real = (χ_real + χ_real_next) * -v̂x + (χ_imag + χ_imag_next) * v̂y
#         εχ_imag = (χ_imag + χ_imag_next) * -v̂x - (χ_real + χ_real_next) * v̂y
#         εχ_real * εχ_real + εχ_imag * εχ_imag <= ε_abs_2 && (return P)

#         # recurse
#         χ_real = χ_real_next
#         χ_imag = χ_imag_next
#     end

#     # unable to satisfy error bounds
#     warn_Pmax()

#     return PMAX
# end

@inline function warn_Pmax()
    if WARNING_FLAG_PMAX[]
        @warn "error tolerance not met with dynamic expansion order; try increasing `Pmax`"
        WARNING_FLAG_PMAX[] = false
    end
end


"""
    get_P(r_min, r_max, ρ_min, ρ_max, ΔC2, Pmax, ε_rel, error_method)

Returns the smallest expansion order not greater than `Pmax` and satisfying the specified relative error tolerance.

# Inputs

* `r_min::Flaot64`: distance from the multipole expansion to the closest target
* `r_max::Float64`: distance from the local expansion to the farthest target
* `ρ_min::Float64`: distance from the local expansion to the closest source
* `ρ_max::Float64`: distance from the multipole expansion to the farthest source
* `ΔC2::Float64`: distance squared between multipole and local centers
* `Pmax::Int64`: maximum allowable expansion order
* `ε_rel::Float64`: relative error tolerance
* `error_method::ErrorMethod`: type used to dispatch on the desired error method

# Ouputs

* `P::Int`: the smallest expansion order to satisfy the error tolerance

"""
function get_P(r_min, r_max, ρ_min, ρ_max, ΔC2, Pmax, q_monopole, q_dipole_multipole, q_dipole_local, ε, ::Union{EqualSpheres, UnequalSpheres, UnequalBoxes})
    ρ_max_over_r_min = ρ_max / r_min
    r_max_over_ρ_min = r_max / ρ_min
    t1 = ρ_max_over_r_min / (r_min - ρ_max) # absolute error
    # t1 = ρ_max / (r_min - ρ_max) # relative error
    t2 = r_max_over_ρ_min / (ρ_min - r_max) # absolute error
    # t2 = ρ_min / (ρ_min - r_max) # relative error
    for P in 0:Pmax-1
        t1 * max(q_monopole, q_dipole_multipole * (P+2)) + t2 * max(q_monopole, q_dipole_local * (P+1)) < ε && (return P)
        t1 *= ρ_max_over_r_min
        t2 *= r_max_over_ρ_min
    end

    warn_Pmax()

    return Pmax
end

function get_P(r_min, r_max, ρ_min, ρ_max, ΔC2, Pmax, q_monopole, q_dipole_multipole, q_dipole_local, ε, ::Union{UniformUnequalSpheres, UniformUnequalBoxes})

    # multipole error
    ρ_max_over_r_min = ρ_max / r_min
    # ε_multipole = 1.5 * ρ_max / (r_min - ρ_max) # relative error
    ε_multipole = 1.5 / (r_min - ρ_max) * ρ_max_over_r_min # absolute error

    # local error
    ρ_max2 = ρ_max * ρ_max
    # Γ = 3 * r_max * r_min / (2 * ρ_max2 * ρ_max) # relative error
    Γ = 3 * r_max / (2 * ρ_max2 * ρ_max) # absolute error

    # distance from multipole center to point closest to the local expansion
    ΔC = sqrt(ΔC2)
    ρ_max_line = ΔC - ρ_min

    # distance from local center to farthest multipole point
    ΔC_plus = ρ_min + ρ_max_line + ρ_max_line

    # distance from local center to nearest multipole point
    ΔC_minus = ρ_min
    t_plus = r_max / ΔC_plus
    one_over_ΔC_minus = 1/ΔC_minus
    t_minus = r_max * one_over_ΔC_minus
    L = log((1-t_plus)/(1-t_minus))
    ΔC2_inv = 1/(2*ΔC)
    ρ2_ΔC2_2ΔC = (ρ_max2 - ΔC2) * ΔC2_inv
    r2_ΔC2 = r_max * r_max * ΔC2_inv

    # recursively tracked quantities
    Lζ_sum_pm1 = L
    Lζ_sum_pm2 = L
    Lζ_sum_pm3 = L
    t_plus_n = t_plus
    t_minus_n = t_minus
    η_p = log(ΔC_plus * one_over_ΔC_minus)
    r_inv = 1/r_max
    η_pm1 = η_p + (ΔC_plus - ΔC_minus) * r_inv
    η_pm2 = η_pm1 + (ΔC_plus*ΔC_plus - ΔC_minus*ΔC_minus) * r_inv * r_inv * 0.5

    #--- test p=0 ---#

    # test error
    ε_local = Γ * (r_max * (η_pm1 + Lζ_sum_pm2) + ρ2_ΔC2_2ΔC * (η_p + Lζ_sum_pm1) - r2_ΔC2 * (η_pm2 + Lζ_sum_pm3))
    ε_multipole * 0.25 + ε_local < ε_rel && (return 0)

    # recurse multipole
    ε_multipole *= ρ_max_over_r_min

    # recurse local
    η_pm2 = η_pm1
    η_pm1 = η_p
    η_p = zero(r_min)

    #--- test p>0 ---#

    for P in 1:Pmax-1
        # get local error
        ε_local = Γ * (r_max * (η_pm1 + Lζ_sum_pm2) + ρ2_ΔC2_2ΔC * (η_p + Lζ_sum_pm1) - r2_ΔC2 * (η_pm2 + Lζ_sum_pm3))

        # check error
        ε_multipole / (P+4) * max(q_monopole, q_dipole_multipole * (P+2)) + ε_local * max(q_monopole, q_dipole_local * (P+1)) < ε_rel && (return P)

        # recurse multipole
        ε_multipole *= ρ_max_over_r_min

        # recurse local
        Lζ_sum_pm3 = Lζ_sum_pm2
        Lζ_sum_pm2 = Lζ_sum_pm1
        Lζ_sum_pm1 += (t_plus_n - t_minus_n) / P
        t_plus_n *= t_plus
        t_minus_n *= t_minus
        η_pm2 = η_pm1
        η_pm1 = η_p
        η_p = zero(r_min)
    end

    warn_Pmax()

    return Pmax
end

#--- dispatching dynamic expansion order ---#

@inline function get_Pmax(expansion_order::Int)
    return expansion_order
end

@inline function get_Pmax(expansion_order::Dynamic{PMAX,<:Any}) where PMAX
    return PMAX
end
