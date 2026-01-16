include("../test/evaluate_multipole.jl")

#--- determine distances for error formulae ---#

@inline minabs(x, y) = x * (abs(x) <= abs(y)) + y * (abs(y) < abs(x))

function minimum_distance(dx, rx_minus, rx_plus)
    right = dx + rx_plus
    left = dx + rx_minus
    same_sign = right * left >= 0
    return same_sign * minabs(right, left)
end

function minimum_distance(center1, center2, box2::SVector{3,<:Any})
    x1, y1, z1 = center1
    x2, y2, z2 = center2
    bx, by, bz = box2
    Δx = minimum_distance(x2-x1, -bx, bx)
    Δy = minimum_distance(y2-y1, -by, by)
    Δz = minimum_distance(z2-z1, -bz, bz)
    return SVector{3}(Δx, Δy, Δz)
end

"""
Vector from center 2 to the corner of box2 closest to center.
"""
function closest_corner(center1, center2, box2::SVector{3,<:Any})
    Δxc, Δyc, Δzc = center1 - center2
    bx, by, bz = box2
    Δx = minabs(Δxc-bx,Δxc+bx)
    Δy = minabs(Δyc-by,Δyc+by)
    Δz = minabs(Δzc-bz,Δzc+bz)
    return SVector{3}(Δxc-Δx, Δyc-Δy, Δzc-Δz)
end

@inline function get_r_ρ(local_branch, multipole_branch, separation_distance_squared, ::UnequalBoxes)
    r_max = local_branch.radius
    r_min = norm(minimum_distance(multipole_branch.center, local_branch.center, local_branch.box))
    ρ_max = multipole_branch.radius
    ρ_min = norm(minimum_distance(local_branch.center, multipole_branch.center, multipole_branch.box))

    return r_min, r_max, ρ_min, ρ_max
end

@inline function get_r_ρ(local_branch, multipole_branch, separation_distance_squared, ::UniformUnequalBoxes)
    r_max = local_branch.radius
    r_min = norm(minimum_distance(multipole_branch.center, local_branch.center, local_branch.box))
    ρ_max = multipole_branch.radius
    ρ_min = sqrt(separation_distance_squared) - ρ_max

    return r_min, r_max, ρ_min, ρ_max
end

@inline function get_r_ρ(local_branch, multipole_branch, separation_distance_squared, ::Union{UnequalSpheres, UniformUnequalSpheres})
    separation_distance = sqrt(separation_distance_squared)
    r_max = local_branch.radius
    r_min = separation_distance - r_max
    ρ_max = multipole_branch.radius
    ρ_min = separation_distance - ρ_max

    return r_min, r_max, ρ_min, ρ_max
end

# function dipole_from_multipole(expansion)
#     # dipole term
#     qx = -2 * expansion[2,1,3]
#     qy = -2 * expansion[1,1,3]
#     qz = expansion[1,1,2]
#     return qx, qy, qz
# end
#
# function vortex_from_multipole(expansion)
#     ωx = -2 * expansion[2,2,3]
#     ωy = -2 * expansion[1,2,3]
#     ωz = expansion[1,2,2]
#     return ωx, ωy, ωz
# end
#
# function get_A_local(multipole_branch, r_vec, expansion_order)
#     # extract expansion
#     expansion = multipole_branch.multipole_expansion
#
#     # monopole term
#     q_monopole = expansion[1,1,1]
#
#     # dipole term
#     q_dipole_x, q_dipole_y, q_dipole_z = dipole_from_multipole(expansion)
#
#     # scaling factor
#     rx, ry, rz = r_vec
#     r2 = rx*rx + ry*ry + rz*rz
#     Q = max(abs(q_monopole), (expansion_order+1) * abs(q_dipole_x*r_vec[1] + q_dipole_y*r_vec[2] + q_dipole_z*r_vec[3]) / r2)
#
#     Q = q_monopole
#     return Q
# end

#--- actual error prediction functions ---#

function multipole_error(local_branch, multipole_branch, i_source_branch, expansions, P, error_method::Union{UnequalSpheres, UnequalBoxes}, ::Val{LH}) where LH

    @assert LH == false "$(error_method) not implmemented with `lamb_helmholtz=true`"

    # get distances
    Δx, Δy, Δz = local_branch.center - multipole_branch.center
    separation_distance_squared = Δx*Δx + Δy*Δy + Δz*Δz
    r_min, _, _, ρ_max = get_r_ρ(local_branch, multipole_branch, separation_distance_squared, error_method)

    # calculate error upper bound
    multipole_expansion = view(expansions, :, :, :, i_source_branch)
    A = multipole_expansion[1,1,1]
    ρ_max_over_r_min = ρ_max / r_min
    ε = ρ_max_over_r_min * A / (r_min - ρ_max)
    for p in 1:P
        ε *= ρ_max_over_r_min
    end

    return abs(ε) * ONE_OVER_4π
end

function multipole_error(local_branch, multipole_branch, i_source_branch, expansions, P, error_method::Union{UniformUnequalSpheres, UniformUnequalBoxes}, ::Val{LH}) where LH
    @assert LH == false "$(error_method) not implmemented with `lamb_helmholtz=true`"
    return 3 / (2*P+8) * multipole_error(local_branch, multipole_branch, i_source_branch, expansions, P, UnequalSpheres())
    #return 3 / (2*P+8) * sqrt(2/(2*P+3)) * multipole_error(local_branch, multipole_branch, P, UnequalSpheres())
end

"""
    Multipole coefficients up to expansion order `P+1` must be added to `multipole_branch` before calling this function.
"""
function multipole_error(local_branch, multipole_branch, i_source_branch, expansions, P, error_method::RotatedCoefficients, lamb_helmholtz::Val)

    # vector from multipole center to local center
    t⃗ = local_branch.center - multipole_branch.center

    # multipole max error location
    Δx, Δy, Δz = minimum_distance(multipole_branch.center, local_branch.center, local_branch.box)
    r_mp = sqrt(Δx * Δx + Δy * Δy + Δz * Δz)

    # calculate error
    multipole_expansion = view(expansions, :, :, :, i_source_branch)
    return multipole_error(t⃗, r_mp, multipole_expansion, P, error_method, lamb_helmholtz)
end

function multipole_error(t⃗, r_mp, multipole_expansion, P, error_method, lamb_helmholtz::Val{LH}) where LH
    # ensure enough multipole coefficients exist
    Nmax = (-3 + Int(sqrt(9 - 4*(2-2*size(multipole_expansion,3))))) >> 1
    @assert P < Nmax "Error method `RotatedCoefficients` can only predict error up to one lower expansion order than multipole coefficients have been computed; expansion order P=$P was requested, but branches only contain coefficients up to $Nmax"

    # container for rotated multipole coefficients
    nmax = P + 1
    weights_tmp_1 = Array{eltype(multipole_expansion)}(undef, 2, 2, (nmax*(nmax+1))>>1 + nmax + 1)
    weights_tmp_2 = Array{eltype(multipole_expansion)}(undef, 2, 2, (nmax*(nmax+1))>>1 + nmax + 1)
    eimϕs = zeros(eltype(multipole_expansion), 2, nmax+1)
    Ts = zeros(eltype(multipole_expansion), length_Ts(nmax))

    # rotate coefficients
    _, θ, ϕ = cartesian_to_spherical(t⃗)

    # rotate about z axis
    rotate_z!(weights_tmp_1, multipole_expansion, eimϕs, ϕ, nmax, lamb_helmholtz)

    # rotate about y axis
    # NOTE: the method used here results in an additional rotatation of π about the new z axis
    rotate_multipole_y!(weights_tmp_2, weights_tmp_1, Ts, Hs_π2, ζs_mag, θ, nmax, lamb_helmholtz)

    # multipole error
    in0 = (nmax*(nmax+1))>>1 + 1
    ϕn0 = weights_tmp_2[1,1,in0]
    ϕn1_real = weights_tmp_2[1,1,in0+1]
    ϕn1_imag = weights_tmp_2[2,1,in0+1]
    if LH
        χn1_real = weights_tmp_2[1,2,in0+1]
        χn1_imag = weights_tmp_2[2,2,in0+1]
    end

    # calculate multipole error
    np1!_over_r_mp_np2 = Float64(factorial(big(nmax+1))) / r_mp^(nmax+2)
    ε_mp = (sqrt(ϕn1_real * ϕn1_real + ϕn1_imag * ϕn1_imag) + abs(ϕn0)) * np1!_over_r_mp_np2
    if LH
        ε_mp += sqrt(χn1_real * χn1_real + χn1_imag * χn1_imag) * np1!_over_r_mp_np2 * r_mp
    end

    return ε_mp * ONE_OVER_4π
end

function local_error(local_branch, multipole_branch, i_source_branch, expansions, P, error_method::Union{UnequalSpheres, UnequalBoxes}, lamb_helmholtz)

    @assert LH == false "$(error_method) not implmemented with `lamb_helmholtz=true`"

    # get distances
    Δx, Δy, Δz = local_branch.center - multipole_branch.center
    separation_distance_squared = Δx*Δx + Δy*Δy + Δz*Δz
    r_min, r_max, ρ_min, ρ_max = get_r_ρ(local_branch, multipole_branch, separation_distance_squared, error_method)

    # calculate error upper bound
    r_max_over_ρ_min = r_max / ρ_min
    ε = r_max_over_ρ_min

    multipole_expansion = view(expansions, :, :, :, i_source_branch)
    A = multipole_expansion[1,1,1]
    ε *= A / (ρ_min - r_max)

    for p in 1:P
        ε *= r_max_over_ρ_min
    end

    return abs(ε) * ONE_OVER_4π
end

function local_error(local_branch, multipole_branch, i_source_branch, expansions, P, error_method::Union{UniformUnequalSpheres, UniformUnequalBoxes}, lamb_helmholtz)

    @assert LH == false "$(error_method) not implmemented with `lamb_helmholtz=true`"

    # get distances
    Δx, Δy, Δz = local_branch.center - multipole_branch.center
    separation_distance_squared = Δx*Δx + Δy*Δy + Δz*Δz
    r_min, r_max, ρ_min, ρ_max = get_r_ρ(local_branch, multipole_branch, separation_distance_squared, error_method)

    # local error
    ρ_max2 = ρ_max * ρ_max
    r_bar = r_min - ρ_max
    multipole_expansion = view(expansions, :, :, :, i_source_branch)
    A = multipole_expansion[1,1,1]
    Γ = 3 * r_max * A / (2 * ρ_max2 * ρ_max)

    # distance from multipole center to point closest to the local expansion
    ΔC = sqrt(separation_distance_squared)
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
    ρ2_ΔC2_2ΔC = (ρ_max2 - separation_distance_squared) * ΔC2_inv
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

    if P==0
        return ε_local
    end

    # recurse local
    η_pm2 = η_pm1
    η_pm1 = η_p
    η_p = zero(r_min)

    #--- test p>0 ---#
    for p in 1:P
        # get local error
        ε_local = Γ * (r_max * (η_pm1 + Lζ_sum_pm2) + ρ2_ΔC2_2ΔC * (η_p + Lζ_sum_pm1) - r2_ΔC2 * (η_pm2 + Lζ_sum_pm3))

        # recurse local
        Lζ_sum_pm3 = Lζ_sum_pm2
        Lζ_sum_pm2 = Lζ_sum_pm1
        Lζ_sum_pm1 += (t_plus_n - t_minus_n) / p
        t_plus_n *= t_plus
        t_minus_n *= t_minus
        η_pm2 = η_pm1
        η_pm1 = η_p
        η_p = zero(r_min)
    end

    ε_local = Γ * (r_max * (η_pm1 + Lζ_sum_pm2) + ρ2_ΔC2_2ΔC * (η_p + Lζ_sum_pm1) - r2_ΔC2 * (η_pm2 + Lζ_sum_pm3))

    return abs(ε_local) * ONE_OVER_4π
end

"""
    Multipole coefficients up to expansion order `P+1` must be added to `multipole_branch` before calling this function.
"""
function local_error(local_branch, multipole_branch, i_source_branch, expansions, P, error_method::RotatedCoefficients, lamb_helmholtz::Val)

    # vector from multipole center to local center
    t⃗ = local_branch.center - multipole_branch.center

    # local max error location
    r_l = sum(local_branch.box) * 0.33333333333333333333 * sqrt(3)

    # calculate error
    multipole_expansion = view(expansions, :, :, :, i_source_branch)
    return local_error(t⃗, r_l, multipole_expansion, P, error_method, lamb_helmholtz)
end

function local_error(t⃗, r_l, multipole_expansion, P, error_method, lamb_helmholtz::Val{LH}) where LH
    # ensure enough multipole coefficients exist
    Nmax = (-3 + Int(sqrt(9 - 4*(2-2*size(multipole_expansion,3))))) >> 1
    @assert P < Nmax "Error method `RotatedCoefficients` can only predict error up to one lower expansion order than multipole coefficients have been computed; expansion order P=$P was requested, but branches only contain coefficients up to $Nmax"

    # container for rotated multipole coefficients
    nmax = P + 1
    weights_tmp_1 = Array{eltype(multipole_expansion)}(undef, 2, 2, (nmax*(nmax+1))>>1 + nmax + 1)
    weights_tmp_2 = Array{eltype(multipole_expansion)}(undef, 2, 2, (nmax*(nmax+1))>>1 + nmax + 1)
    eimϕs = zeros(eltype(multipole_expansion), 2, nmax+1)
    Ts = zeros(eltype(multipole_expansion), length_Ts(nmax))

    # rotate coefficients
    r, θ, ϕ = cartesian_to_spherical(t⃗)

    # rotate about z axis
    rotate_z!(weights_tmp_1, multipole_expansion, eimϕs, ϕ, nmax, lamb_helmholtz)

    # rotate about y axis
    # NOTE: the method used here results in an additional rotatation of π about the new z axis
    rotate_multipole_y!(weights_tmp_2, weights_tmp_1, Ts, Hs_π2, ζs_mag, θ, nmax, lamb_helmholtz)

    # get local coefficients
    # extract degree n, order 0-1 coefficients for error prediction
    # this function also performs the lamb-helmholtz transformation
    if nmax < 21
        n!_t_np1 = factorial(nmax) / r^(nmax+1)
    else
        n!_t_np1 = typeof(r)(factorial(big(nmax)) / r^(nmax+1))
    end
    ϕn0_real, ϕn1_real, ϕn1_imag, χn1_real, χn1_imag = translate_multipole_to_local_z_m01_n(weights_tmp_2, r, 1.0/r, lamb_helmholtz, n!_t_np1, nmax)

    # other values
    r_l_nm1_over_nm1! = r_l^(nmax - 1) / Float64(factorial(big(nmax-1)))

    # calculate local error
    ε_l = (abs(ϕn1_real) + abs(ϕn1_imag) + abs(ϕn0_real)) * r_l_nm1_over_nm1!
    if LH
        ε_l += sqrt(χn1_real * χn1_real + χn1_imag * χn1_imag) * r_l_nm1_over_nm1! * r_l
    end

    # calculate local error

    return ε_l * ONE_OVER_4π
end

function total_error(local_branch, multipole_branch, i_source_branch, expansions, expansion_order, error_method, lamb_helmholtz)
    ε_multipole = multipole_error(local_branch, multipole_branch, i_source_branch, expansions, expansion_order, error_method, lamb_helmholtz)
    ε_local = local_error(local_branch, multipole_branch, i_source_branch, expansions, expansion_order, error_method, lamb_helmholtz)
    return ε_multipole + ε_local
end

#------- try more efficient error predictors -------#

function predict_error(target_branch, source_weights, source_branch, weights_tmp_1, weights_tmp_2, weights_tmp_3, Ts, eimϕs, ζs_mag, Hs_π2, M̃, L̃, expansion_order, lamb_helmholtz::Val{LH}, ::HeuristicRelativePotential) where LH

    # --- distance information ---#

    # multipole error location
    Δx, Δy, Δz = minimum_distance(source_branch.center, target_branch.center, target_branch.box)
    r_mp = sqrt(Δx * Δx + Δy * Δy + Δz * Δz)

    # max source radius
    ρ_max = source_branch.radius

    @assert r_mp > ρ_max "Error predictor requires target branch to be outside the source branch's bounding sphere"

    ρ_max_over_r_mp = ρ_max / r_mp
    p = expansion_order

    # @assert (p+1) * ρ_max - (p+2) * r_mp >= 0.0 "Error predictor requires target branch to be outside the source branch's bounding sphere"

    ε_mp = ρ_max_over_r_mp^p / r_mp

    return ε_mp, ε_mp
end

function predict_error(target_branch, source_weights, source_branch, weights_tmp_1, weights_tmp_2, weights_tmp_3, Ts, eimϕs, ζs_mag, Hs_π2, M̃, L̃, expansion_order, lamb_helmholtz::Val{LH}, ::HeuristicAbsolutePotential) where LH

    # --- distance information ---#

    # multipole error location
    Δx, Δy, Δz = minimum_distance(source_branch.center, target_branch.center, target_branch.box)
    r_mp = sqrt(Δx * Δx + Δy * Δy + Δz * Δz)

    # max source radius
    ρ_max = source_branch.radius

    @assert r_mp > ρ_max "Error predictor requires target branch to be outside the source branch's bounding sphere"

    ρ_max_over_r_mp = ρ_max / r_mp
    p = expansion_order + 1

    # @assert (p+1) * ρ_max - (p+2) * r_mp >= 0.0 "Error predictor requires target branch to be outside the source branch's bounding sphere"

    ε_mp = ρ_max_over_r_mp^p / r_mp

    # evaluate the multipole expansion to get scaling factor
    derivatives_switch = DerivativesSwitch(true, true, true)
    ϕ, _ = evaluate_multipole(source_branch.center + SVector{3}(Δx, Δy, Δz), source_branch.center, source_weights, derivatives_switch, expansion_order, lamb_helmholtz)

    return ε_mp * ϕ, ε_mp * ϕ
end

function predict_error(target_branch, source_weights, source_branch, weights_tmp_1, weights_tmp_2, weights_tmp_3, Ts, eimϕs, ζs_mag, Hs_π2, M̃, L̃, expansion_order, lamb_helmholtz::Val{LH}, ::UnequalSpheresMultipoleGradient) where LH

    # --- distance information ---#

    # multipole error location
    Δx, Δy, Δz = minimum_distance(source_branch.center, target_branch.center, target_branch.box)
    r_mp = sqrt(Δx * Δx + Δy * Δy + Δz * Δz)

    # max source radius
    ρ_max = source_branch.radius

    @assert r_mp > ρ_max "Error predictor requires target branch to be outside the source branch's bounding sphere"

    ρ_max_over_r_mp = ρ_max / r_mp
    p = expansion_order

    # @assert (p+1) * ρ_max - (p+2) * r_mp >= 0.0 "Error predictor requires target branch to be outside the source branch's bounding sphere"

    A = abs(source_weights[1,1,1])
    ε_mp = -A * ρ_max_over_r_mp^(p+1) * ((p+1) * ρ_max - (p+2) * r_mp) / (r_mp * (r_mp - ρ_max) * (r_mp - ρ_max))'
    ε_mp *= ONE_OVER_4π

    return ε_mp, ε_mp
end

function predict_error(target_branch, source_weights, source_branch, weights_tmp_1, weights_tmp_2, weights_tmp_3, Ts, eimϕs, ζs_mag, Hs_π2, M̃, L̃, expansion_order, lamb_helmholtz::Val{LH}, ::UnequalSpheresGradient) where LH

    # --- multipole error ---#
    
    # # multipole error location
    # Δx, Δy, Δz = minimum_distance(source_branch.center, target_branch.center, target_branch.box)
    # r_mp = sqrt(Δx * Δx + Δy * Δy + Δz * Δz)
    
    # # max source radius
    # ρ_max = source_branch.radius
    
    # @assert r_mp > ρ_max "Error predictor requires target branch to be outside the source branch's bounding sphere"
    
    # ρ_max_over_r_mp = ρ_max / r_mp
    # p = expansion_order
    
    # # @assert (p+1) * ρ_max - (p+2) * r_mp >= 0.0 "Error predictor requires target branch to be outside the source branch's bounding sphere"
    
    # A = abs(source_weights[1,1,1])
    # ε_mp = -A * ρ_max_over_r_mp^(p+1) * ((p+1) * ρ_max - (p+2) * r_mp) / (r_mp * (r_mp - ρ_max) * (r_mp - ρ_max))'
    # ε_mp *= ONE_OVER_4π
    
    # --- local error ---#

    # --- combined error (chatgpt) ---#
    
    A = abs(source_weights[1,1,1])
    p = expansion_order + 1
    η = (source_branch.radius + target_branch.radius) / norm(source_branch.center - target_branch.center)
    ε_Σ = A * (p+1) * η^p / ((1 - η) * (1 - η)) * 0.5 * ONE_OVER_4π

    return ε_Σ, ε_Σ
end

function predict_error(target_branch, source_weights, source_branch, weights_tmp_1, weights_tmp_2, weights_tmp_3, Ts, eimϕs, ζs_mag, Hs_π2, M̃, L̃, expansion_order, lamb_helmholtz::Val{LH}, ::UnequalSpheres) where LH

    # --- combined error (chatgpt) ---#
    
    A = abs(source_weights[1,1,1])
    p = expansion_order + 1
    η = (source_branch.radius + target_branch.radius) / norm(source_branch.center - target_branch.center)
    ε_Σ = A * η^(p+1) / (1 - η) * 0.5 * ONE_OVER_4π

    return ε_Σ, ε_Σ
end

function predict_error(target_branch, source_weights, source_branch, weights_tmp_1, weights_tmp_2, weights_tmp_3, Ts, eimϕs, ζs_mag, Hs_π2, M̃, L̃, expansion_order, lamb_helmholtz::Val{LH}, ::PringleAbsolutePotential) where LH

    A = abs(source_weights[1,1,1])
    p = expansion_order + 1

    # multipole error
    ρ = source_branch.radius
    r = norm(minimum_distance(source_branch.center, target_branch.center, target_branch.box))
    c = r / ρ
    ε_mp = A * (1 / c)^(p+1) / (c - 1.0) / ρ * ONE_OVER_4π

    # local error
    a = target_branch.radius
    ρ = norm(minimum_distance(target_branch.center, source_branch.center, source_branch.box))
    c = ρ / a
    ε_l = A * (1 / c)^(p+1) / (c - 1.0) / a * ONE_OVER_4π

    return ε_mp, ε_l
end

function predict_error(target_branch, source_weights, source_branch, weights_tmp_1, weights_tmp_2, weights_tmp_3, Ts, eimϕs, ζs_mag, Hs_π2, M̃, L̃, expansion_order, lamb_helmholtz::Val{LH}, ::DehnenAbsoluteGradient) where LH

    # multipole power
    E_AB = zero(eltype(source_weights))
    A = abs(source_weights[1,1,1])
    p = expansion_order
    # r = norm(minimum_distance(source_branch.center, target_branch.center, target_branch.box))
    r = norm(target_branch.center - source_branch.center)
    rp = r^p
    for k in 0:p
        P2 = zero(eltype(source_weights))
        m = 0
        in_index = (k*(k+1))>>1 + k + m + 1
        coeff_real = source_weights[1,1,in_index]
        coeff_imag = source_weights[2,1,in_index]
        P2 += abs(coeff_real + im * coeff_imag)^2 * Float64(factorial(big(k - m))) * Float64(factorial(big(k + m)))
        for m in 1:k
            in_index = (k*(k+1))>>1 + k + m + 1
            coeff_real = source_weights[1,1,in_index]
            coeff_imag = source_weights[2,1,in_index]
            P2 += 2 * abs(coeff_real + im * coeff_imag)^2 * Float64(factorial(big(k - m))) * Float64(factorial(big(k + m)))
        end
        # compute binomial coefficient (p choose k) as Float64
        binom_pk = Float64(binomial(p, k))

        # P2 = (A * source_branch.radius^k)^2

        # check P computation
        P = sqrt(P2)
        @assert P <= A * source_branch.radius^k "Computed multipole power exceeds maximum possible value"

        # use it to accumulate E_AB
        E_AB += P * binom_pk * target_branch.radius^(p-k) / rp
    end
    # E_AB /= A

    # gradient
    Ẽ_AB = 8 * max(source_branch.radius, target_branch.radius) * E_AB / (source_branch.radius + target_branch.radius)

    # error
    ε_mp = Ẽ_AB / (r*r) * ONE_OVER_4π

    return ε_mp, zero(ε_mp)
end

function predict_error(target_branch, source_weights, source_branch, weights_tmp_1, weights_tmp_2, weights_tmp_3, Ts, eimϕs, ζs_mag, Hs_π2, M̃, L̃, expansion_order, lamb_helmholtz::Val{LH}, ::PringleRelativePotential) where LH

    # number of terms in expansion
    p = expansion_order + 1

    # multipole error
    ρ = source_branch.radius
    r = norm(minimum_distance(source_branch.center, target_branch.center, target_branch.box))
    c = r / ρ
    ε_mp = (1/c)^p / (c-1) * ONE_OVER_4π

    # local error
    a = target_branch.radius
    ρ = norm(minimum_distance(target_branch.center, source_branch.center, source_branch.box))
    c = ρ / a
    ε_l = (1/c)^p / (c-1) * ONE_OVER_4π

    return ε_mp, ε_l
end

function predict_error(target_branch, source_weights, source_branch, weights_tmp_1, weights_tmp_2, weights_tmp_3, Ts, eimϕs, ζs_mag, Hs_π2, M̃, L̃, expansion_order, lamb_helmholtz::Val{LH}, ::RotatedCoefficientsAbsoluteGradient) where LH
    
    # translation vector
    Δx = target_branch.center - source_branch.center
    r, θ, ϕ = cartesian_to_spherical(Δx)

    #--- distance information ---#

    # multipole error location
    Δx, Δy, Δz = minimum_distance(source_branch.center, target_branch.center, target_branch.box)
    r_mp = sqrt(Δx * Δx + Δy * Δy + Δz * Δz)

    # local error location
    r_l = target_branch.radius

    #--- initialize recursive values ---#

    rinv = one(r) / r
    r_mp_inv = one(r_mp) / r_mp
    r_mp_inv_np2 = r_mp_inv * r_mp_inv * r_mp_inv
    r_l_nm1 = one(r_l)

    #--- n = 0 coefficients ---#

    # rotate about z axis
    eiϕ_real, eiϕ_imag, eimϕ_real, eimϕ_imag = rotate_z_0!(weights_tmp_1, source_weights, eimϕs, ϕ, expansion_order, lamb_helmholtz)

    # get y-axis Wigner rotation matrix
    sβ, cβ, _1_n = update_Ts_0!(Ts, Hs_π2, θ, expansion_order)

    # perform rotation
    i_ζ, i_T = _rotate_multipole_y_n!(weights_tmp_2, weights_tmp_1, Ts, ζs_mag, expansion_order, lamb_helmholtz, 0, 0, 0)

    # (can't predict error yet, as we need degree n+1 coefficients)
    # preallocate recursive values
    n!_t_np1 = rinv * rinv
    nm1!_inv = 1.0
    np1! = 2.0

    #--- n > 0 and error predictions ---#

    # preallocate values to be available for recursion and debugging
    ε_mp, ε_l = zero(r), zero(r)
    # n = 1
    error_success = false

    # n>0
    for n in 1:expansion_order+1

        # rotate about z axis
        eimϕ_real, eimϕ_imag = rotate_z_n!(weights_tmp_1, source_weights, eimϕs, eiϕ_real, eiϕ_imag, eimϕ_real, eimϕ_imag, lamb_helmholtz, n)

        # get y-axis Wigner rotation matrix
        _1_n = update_Ts_n!(Ts, Hs_π2, sβ, cβ, _1_n, n)

        # perform rotation about the y axis
        # NOTE: the method used here results in an additional rotatation of π about the new z axis
        i_ζ, i_T = _rotate_multipole_y_n!(weights_tmp_2, weights_tmp_1, Ts, ζs_mag, expansion_order, lamb_helmholtz, i_ζ, i_T, n)

        # check multipole error
        in0 = (n*(n+1))>>1 + 1
        ϕn0 = weights_tmp_2[1,1,in0]
        ϕn1_real = weights_tmp_2[1,1,in0+1]
        ϕn1_imag = weights_tmp_2[2,1,in0+1]
        if LH
            χn1_real = weights_tmp_2[1,2,in0+1]
            χn1_imag = weights_tmp_2[2,2,in0+1]
        end

        # calculate multipole error
        ε_mp = (sqrt(ϕn1_real * ϕn1_real + ϕn1_imag * ϕn1_imag) + abs(ϕn0)) * np1! * r_mp_inv_np2
        if LH
            ε_mp += sqrt(χn1_real * χn1_real + χn1_imag * χn1_imag) * np1! * r_mp_inv_np2 * r_mp
        end

        # check local error if multipole error passes
        if n == expansion_order + 1

            # extract degree n, order 0-1 coefficients for error prediction
            # this function also performs the lamb-helmholtz transformation
            ϕn0_real, ϕn1_real, ϕn1_imag, χn1_real, χn1_imag = translate_multipole_to_local_z_m01_n(weights_tmp_2, r, rinv, lamb_helmholtz, n!_t_np1, n)

            # calculate local error
            ε_l = (abs(ϕn1_real) + abs(ϕn1_imag) + abs(ϕn0_real)) * r_l_nm1 * nm1!_inv
            if LH
                ε_l += sqrt(χn1_real * χn1_real + χn1_imag * χn1_imag) * r_l_nm1 * r_l * nm1!_inv
            end

        end

        # recurse
        if n < expansion_order # if statement ensures that n == desired P
                                   # when tolerance is reached (breaks the loop early)
                                   # or that n == Pmax when tolerance is not reached
            n!_t_np1 *= (n+1) * rinv
            r_l_nm1 *= r_l
            r_mp_inv_np2 *= r_mp_inv
            nm1!_inv /= n

            # increment n
            # n += 1
            np1! *= n+1
        end
    end

    return ε_mp * ONE_OVER_4π, ε_l * ONE_OVER_4π
end

"""
performs a partial M2L for expansion_order+1, and predicts the error along the way;

expects that the upward pass has been performed up to expansion_order+1
"""
function predict_error(target_branch, source_weights, source_branch, weights_tmp_1, weights_tmp_2, weights_tmp_3, Ts, eimϕs, ζs_mag, Hs_π2, M̃, L̃, expansion_order, lamb_helmholtz::Val{LH}, ::PowerAbsolutePotential) where LH
    
    # translation vector
    Δx = target_branch.center - source_branch.center
    r, θ, ϕ = cartesian_to_spherical(Δx)

    #--- multipole error prediction ---#
    
    # distance to closest point
    Δx, Δy, Δz = minimum_distance(source_branch.center, target_branch.center, target_branch.box)
    r_mp = sqrt(Δx * Δx + Δy * Δy + Δz * Δz)
    
    # check multipole magnitude
    mp_power = 0.0
    n=expansion_order+1
    
    # multipole power
    i = (n*(n+1)) >> 1 + 1
    for m in 0:n
        # get Ñ 
        Ñ = sqrt(4*pi*Float64(factorial(big(n+abs(m)))) * Float64(factorial(big(n-abs(m)))) / (2*n+1) )

        # conservative multipole power
        ϕnm_real = source_weights[1,1,i+m]
        ϕnm_imag = source_weights[2,1,i+m]
        mp_power += (ϕnm_real * ϕnm_real + ϕnm_imag * ϕnm_imag) * Ñ * Ñ
    end
    mp_power = sqrt(abs(mp_power))

    # multipole power error prediction
    nfact = Float64(factorial(big(n)))
    Ñ0 = sqrt(4*pi*nfact * nfact / (2*n+1))
    ε_mp_power = mp_power / Ñ0 * nfact / r_mp^(n+1)

    #--- rotate coordinate system ---#

    # rotate about z axis
    rotate_z!(weights_tmp_1, source_weights, eimϕs, ϕ, expansion_order+1, lamb_helmholtz)

    # rotate about y axis
    # NOTE: the method used here results in an additional rotatation of π about the new z axis
    rotate_multipole_y!(weights_tmp_2, weights_tmp_1, Ts, Hs_π2, ζs_mag, θ, expansion_order+1, lamb_helmholtz)

    #--- translate along new z axis ---#

    translate_multipole_to_local_z!(weights_tmp_1, weights_tmp_2, r, expansion_order+1, lamb_helmholtz)

    #--- transform Lamb-Helmholtz decomposition for the new center ---#

    LH && transform_lamb_helmholtz_local!(weights_tmp_1, r, expansion_order+1)

    #--- local error prediction ---#

    l_power = 0.0
    n=expansion_order+1
    
    # multipole power
    i = (n*(n+1)) >> 1 + 1
    for m in 0:n
        # get Ñ 
        L̃ = sqrt(4*pi / (Float64(factorial(big(n-abs(m)))) * Float64(factorial(big(n+abs(m)))) * (2*n+1)) )

        # conservative multipole power
        ϕnm_real = weights_tmp_1[1,1,i+m]
        ϕnm_imag = weights_tmp_1[2,1,i+m]
        l_power += (ϕnm_real * ϕnm_real + ϕnm_imag * ϕnm_imag) * L̃ * L̃ * (2-(m==0))
    end
    l_power = sqrt(abs(l_power))

    # local power error prediction
    nfact = Float64(factorial(big(n)))
    L̃0 = sqrt(4*pi/((2*n+1) * nfact * nfact))
    r_l = target_branch.radius
    ε_l_power = l_power / L̃0 / nfact * r_l^n

    return ε_mp_power * ONE_OVER_4π, ε_l_power * ONE_OVER_4π
end

function predict_error(target_branch, source_weights, source_branch, weights_tmp_1, weights_tmp_2, weights_tmp_3, Ts, eimϕs, ζs_mag, Hs_π2, M̃, L̃, expansion_order, lamb_helmholtz::Val{LH}, ::PowerAbsolutePotentialMultipole) where LH
    
    # translation vector
    Δx = target_branch.center - source_branch.center
    r, θ, ϕ = cartesian_to_spherical(Δx)

    #--- multipole error prediction ---#
    
    # distance to closest point
    Δx, Δy, Δz = minimum_distance(source_branch.center, target_branch.center, target_branch.box)
    r_mp = sqrt(Δx * Δx + Δy * Δy + Δz * Δz)
    
    # check multipole magnitude
    mp_power = 0.0
    n = expansion_order + 1
    
    # multipole power
    i = (n*(n+1)) >> 1 + 1 # index of degree n=p, order m=0
    for m in 0:n
        # get Ñ 
        Ñ = sqrt(4*pi*Float64(factorial(big(n+abs(m)))) * Float64(factorial(big(n-abs(m)))) / (2*n+1) )

        # conservative multipole power
        ϕnm_real = source_weights[1,1,i+m]
        ϕnm_imag = source_weights[2,1,i+m]
        mp_power += (ϕnm_real * ϕnm_real + ϕnm_imag * ϕnm_imag) * Ñ * Ñ
    end
    mp_power = sqrt(mp_power)

    # multipole power error prediction
    nfact = Float64(factorial(big(n)))
    Ñ0 = sqrt(4*pi*nfact * nfact / (2*n+1))
    ε_mp_power = mp_power / Ñ0 * nfact / r_mp^(n+1)

    return ε_mp_power * ONE_OVER_4π, ε_mp_power * ONE_OVER_4π
end

"""
performs a partial M2L for expansion_order+1, and predicts the error along the way;

expects that the upward pass has been performed up to expansion_order+1
"""
function predict_error(target_branch, source_weights, source_branch, weights_tmp_1, weights_tmp_2, weights_tmp_3, Ts, eimϕs, ζs_mag, Hs_π2, M̃, L̃, expansion_order, lamb_helmholtz::Val{LH}, ::PowerRelativePotential) where LH
    
    # translation vector
    Δx = target_branch.center - source_branch.center
    r, θ, ϕ = cartesian_to_spherical(Δx)

    #--- multipole error prediction ---#
    
    # distance to closest point
    Δx, Δy, Δz = minimum_distance(source_branch.center, target_branch.center, target_branch.box)
    r_mp = sqrt(Δx * Δx + Δy * Δy + Δz * Δz)
    
    # check multipole magnitude
    mp_power = 0.0
    n=expansion_order+1
    
    # multipole power
    i = (n*(n+1)) >> 1 + 1
    for m in 0:n
        # get Ñ 
        Ñ = sqrt(4*pi*Float64(factorial(big(n+abs(m)))) * Float64(factorial(big(n-abs(m)))) / (2*n+1) )

        # conservative multipole power
        ϕnm_real = source_weights[1,1,i+m]
        ϕnm_imag = source_weights[2,1,i+m]
        mp_power += (ϕnm_real * ϕnm_real + ϕnm_imag * ϕnm_imag) * Ñ * Ñ
    end
    mp_power = sqrt(abs(mp_power))

    # multipole power error prediction
    nfact = Float64(factorial(big(n)))
    Ñ0 = sqrt(4*pi*nfact * nfact / (2*n+1))
    ε_mp_power = mp_power / Ñ0 * nfact / r_mp^(n+1)

    #--- rotate coordinate system ---#

    # rotate about z axis
    rotate_z!(weights_tmp_1, source_weights, eimϕs, ϕ, expansion_order+1, lamb_helmholtz)

    # rotate about y axis
    # NOTE: the method used here results in an additional rotatation of π about the new z axis
    rotate_multipole_y!(weights_tmp_2, weights_tmp_1, Ts, Hs_π2, ζs_mag, θ, expansion_order+1, lamb_helmholtz)

    #--- translate along new z axis ---#

    translate_multipole_to_local_z!(weights_tmp_1, weights_tmp_2, r, expansion_order+1, lamb_helmholtz)

    #--- transform Lamb-Helmholtz decomposition for the new center ---#

    LH && transform_lamb_helmholtz_local!(weights_tmp_1, r, expansion_order+1)

    #--- local error prediction ---#

    l_power = 0.0
    n=expansion_order+1
    
    # multipole power
    i = (n*(n+1)) >> 1 + 1
    for m in 0:n
        # get Ñ 
        L̃ = sqrt(4*pi / (Float64(factorial(big(n-abs(m)))) * Float64(factorial(big(n+abs(m)))) * (2*n+1)) )

        # conservative multipole power
        ϕnm_real = weights_tmp_1[1,1,i+m]
        ϕnm_imag = weights_tmp_1[2,1,i+m]
        l_power += (ϕnm_real * ϕnm_real + ϕnm_imag * ϕnm_imag) * L̃ * L̃ * (2-(m==0))
    end
    l_power = sqrt(abs(l_power))

    # local power error prediction
    nfact = Float64(factorial(big(n)))
    L̃0 = sqrt(4*pi/((2*n+1) * nfact * nfact))
    r_l = target_branch.radius
    ε_l_power = l_power / L̃0 / nfact * r_l^n

    return ε_mp_power * ONE_OVER_4π, ε_l_power * ONE_OVER_4π
end

function predict_error(target_branch, source_weights, source_branch, weights_tmp_1, weights_tmp_2, weights_tmp_3, Ts, eimϕs, ζs_mag, Hs_π2, M̃, L̃, expansion_order, lamb_helmholtz::Val{LH}, ::PowerAbsoluteGradient) where LH
    
    # translation vector
    Δx = target_branch.center - source_branch.center
    r, θ, ϕ = cartesian_to_spherical(Δx)

    #--- multipole error prediction ---#
    
    # distance to closest point
    Δx, Δy, Δz = minimum_distance(source_branch.center, target_branch.center, target_branch.box)
    r_mp = sqrt(Δx * Δx + Δy * Δy + Δz * Δz)
    
    # check multipole magnitude
    mp_power = 0.0
    n=expansion_order
    
    # multipole power
    i = (n*(n+1)) >> 1 + 1 # index of ϕn0
    for m in 0:n
        # get Ñ 
        Ñ = sqrt(4*pi*Float64(factorial(big(n+abs(m)))) * Float64(factorial(big(n-abs(m)))) / (2*n+1) )

        # conservative multipole power
        ϕnm_real = source_weights[1,1,i+m]
        ϕnm_imag = source_weights[2,1,i+m]
        mp_power += (ϕnm_real * ϕnm_real + ϕnm_imag * ϕnm_imag) * Ñ * Ñ
    end
    mp_power = sqrt(abs(mp_power))

    # multipole power error prediction
    nfact = Float64(factorial(big(n)))
    Ñ0 = sqrt(4*pi*nfact * nfact / (2*n+1))
    ε_mp_power = mp_power / Ñ0 * nfact * (n+1) / r_mp^(n+2) * sqrt(3)
    
    # stuff = zero(r_mp)
    # for np in n+1:n+100
    #     stuff += Float64(factorial(big(np))) / r_mp^(np+1)
    # end
    # ε_mp_power = mp_power / Ñ0 * stuff

    if LH
        i = ((n+1)*(n+2)) >> 1 + 1
        mp_power = 0.0
        for m in 0:n
            # get Ñ 
            Ñ = sqrt(4*pi*Float64(factorial(big(n+abs(m)))) * Float64(factorial(big(n-abs(m)))) / (2*n+1) )
    
            # conservative multipole power
            χnm_real = source_weights[1,2,i+m]
            χnm_imag = source_weights[2,2,i+m]
            mp_power += (χnm_real * χnm_real + χnm_imag * χnm_imag) * Ñ * Ñ
        end
        mp_power = sqrt(abs(mp_power))
    
        # multipole power error prediction
        Ñ0 = sqrt(4*pi*nfact * nfact * (n+1) * (n+1) / (2*(n+1)+1))
        ε_mp_power += n * mp_power / Ñ0 * nfact * (n+1) / r_mp^(n+2) * sqrt(3)

        # stuff = zero(r_mp)
        # for np in n+1:n+100
        #     stuff += Float64(factorial(big(np))) / r_mp^(np+1)
        # end
        # ε_mp_power += mp_power / Ñ0 * stuff
    end

    #--- rotate coordinate system ---#

    # rotate about z axis
    nplus = 1
    rotate_z!(weights_tmp_1, source_weights, eimϕs, ϕ, expansion_order+nplus, lamb_helmholtz)

    # rotate about y axis
    # NOTE: the method used here results in an additional rotatation of π about the new z axis
    rotate_multipole_y!(weights_tmp_2, weights_tmp_1, Ts, Hs_π2, ζs_mag, θ, expansion_order+nplus, lamb_helmholtz)

    #--- translate along new z axis ---#

    translate_multipole_to_local_z!(weights_tmp_1, weights_tmp_2, r, expansion_order+nplus, lamb_helmholtz)

    #--- transform Lamb-Helmholtz decomposition for the new center ---#

    LH && transform_lamb_helmholtz_local!(weights_tmp_1, r, expansion_order+nplus)
    
    #--- local error prediction ---#
    
    ε_l_power = 0.0
    # n=expansion_order # try reducing n to get a more conservative prediction
    for n=expansion_order+1:expansion_order+nplus
        
    # local power
    # n -= 1 # temporarily decrement n to get a more conservative local power
    l_power = 0.0
    i = (n*(n+1)) >> 1 + 1
    for m in 0:n
        # de-normalization
        L̃ = sqrt(4*pi / (Float64(factorial(big(n-abs(m)))) * Float64(factorial(big(n+abs(m)))) * (2*n+1)) )

        # conservative local power
        ϕnm_real = weights_tmp_1[1,1,i+m]
        ϕnm_imag = weights_tmp_1[2,1,i+m]
        l_power += (ϕnm_real * ϕnm_real + ϕnm_imag * ϕnm_imag) * L̃ * L̃ * (2-(m==0))
    end
    l_power = sqrt(abs(l_power))
    l_power_phi_unrotated = l_power
    # n += 1 # return n back to its original value

    # local power error prediction
    nfact = Float64(factorial(big(n)))
    L̃0 = sqrt(4*pi/((2*n+1) * nfact * nfact))
    r_l = target_branch.radius
    ε_l_power += l_power / L̃0 * r_l^(n-1) / nfact * n * sqrt(3)

    # stuff = zero(r_l)
    # for np in 0:n-1
    #     stuff += r_l^(np) / Float64(factorial(big(np)))
    # end
    # ε_l_power = l_power / L̃0 * sqrt(3) * (exp(r) - stuff)
    
    # stuff = zero(r_l)
    # for np in n-1:n+100
    #     stuff += r_l^(np) / Float64(factorial(big(np)))
    # end
    # ε_l_power = l_power / L̃0 * sqrt(3) * stuff

    if LH
        
        # local power
        l_power = 0.0
        # n -= 1 # temporarily decrement n to get a more conservative local power
        i = (n*(n+1)) >> 1 + 1
        for m in 0:n
            # de-normalization
            L̃ = sqrt(4*pi / (Float64(factorial(big(n-abs(m)))) * Float64(factorial(big(n+abs(m)))) * (2*n+1)) )

            # conservative local power
            χnm_real = weights_tmp_1[1,2,i+m]
            χnm_imag = weights_tmp_1[2,2,i+m]
            l_power += (χnm_real * χnm_real + χnm_imag * χnm_imag) * L̃ * L̃ * (1+(m>0))
        end
        l_power = sqrt(abs(l_power))
        l_power_chi_unrotated = l_power
        
        # local power error prediction
        L̃0 = sqrt(4*pi/((2*n+1) * nfact * nfact))
        # println("checking:")
        # @show l_power / L̃0
        # n += 1 # return n back to its original value
        # nfact = Float64(factorial(big(n)))
        r_l = target_branch.radius
        ε_l_power += n * l_power / L̃0 * r_l^(n) / nfact * sqrt(3)

        
        # l_power = 0.0
        # i = (n*(n+1)) >> 1 + 1
        # for m in 0:n
        #     # de-normalization
        #     L̃ = sqrt(4*pi / (Float64(factorial(big(n-abs(m)))) * Float64(factorial(big(n+abs(m)))) * (2*n+1)) )

        #     # conservative local power
        #     χnm_real = weights_tmp_1[1,2,i+m]
        #     χnm_imag = weights_tmp_1[2,2,i+m]
        #     l_power += (χnm_real * χnm_real + χnm_imag * χnm_imag) * L̃ * L̃ * (1+(m>0))
        # end
        # l_power = sqrt(abs(l_power))
        # l_power_chi_unrotated = l_power
        
        # # local power error prediction
        # L̃0 = sqrt(4*pi/((2*n+1) * nfact * nfact))
        # @show l_power / L̃0

        
        # check that l_power / L0 is greater than all coefficients
        stuff = l_power / L̃0
        for m in 0:n
            χnm_real, χnm_imag = weights_tmp_1[1,2,i+m], weights_tmp_1[2,2,i+m]
            χnm = sqrt(χnm_real * χnm_real + χnm_imag * χnm_imag)
            @assert χnm < stuff "l_power / L0 is not greater than all coefficients"
            # @show χnm / stuff
        end
            
        # stuff = zero(r_l)
        # for np in 0:n
        #     stuff += r_l^(np) / Float64(factorial(big(np)))
        # end
        # ε_l_power += l_power / L̃0 * sqrt(3) * (exp(r) - stuff)
        # @show exp(r), stuff, exp(r) - stuff, r_l, nfact, r_l^(n) / nfact
        
        # stuff = zero(r_l)
        # for np in n:n+100
        #     stuff += r_l^(np) / Float64(factorial(big(np)))
        # end
        # ε_l_power += l_power / L̃0 * sqrt(3) * stuff
        # @show l_power / L̃0, stuff
        
    end
    

end
    #--- brute force check ---#




    # # ------- rotate back to original coordinate system -------#
    
    # # back rotate about y axis
    # back_rotate_local_y!(weights_tmp_2, weights_tmp_1, Ts, Hs_π2, ηs_mag, expansion_order+2, lamb_helmholtz)

    # l_power = 0.0
    # for m in 0:n
    #     # de-normalization
    #     L̃ = sqrt(4*pi / (Float64(factorial(big(n-abs(m)))) * Float64(factorial(big(n+abs(m)))) * (2*n+1)) )

    #     # conservative local power
    #     ϕnm_real = weights_tmp_2[1,1,i+m]
    #     ϕnm_imag = weights_tmp_2[2,1,i+m]
    #     l_power += (ϕnm_real * ϕnm_real + ϕnm_imag * ϕnm_imag) * L̃ * L̃ * (2-(m==0))
    # end
    # l_power = sqrt(abs(l_power))
    # l_power_phi_rotated_y = l_power

    # l_power = 0.0
    # for m in 0:n
    #     # de-normalization
    #     L̃ = sqrt(4*pi / (Float64(factorial(big(n-abs(m)))) * Float64(factorial(big(n+abs(m)))) * (2*n+1)) )

    #     # conservative local power
    #     χnm_real = weights_tmp_2[1,2,i+m]
    #     χnm_imag = weights_tmp_2[2,2,i+m]
    #     l_power += (χnm_real * χnm_real + χnm_imag * χnm_imag) * L̃ * L̃ * (1+(m>0))
    # end
    # l_power = sqrt(abs(l_power))
    # l_power_chi_rotated_y = l_power
    
    # # back rotate about z axis and accumulate on target branch
    # back_rotate_z!(weights_tmp_3, weights_tmp_2, eimϕs, expansion_order+2, lamb_helmholtz)

    # l_power = 0.0
    # for m in 0:n
    #     # de-normalization
    #     L̃ = sqrt(4*pi / (Float64(factorial(big(n-abs(m)))) * Float64(factorial(big(n+abs(m)))) * (2*n+1)) )

    #     # conservative local power
    #     ϕnm_real = weights_tmp_3[1,1,i+m]
    #     ϕnm_imag = weights_tmp_3[2,1,i+m]
    #     l_power += (ϕnm_real * ϕnm_real + ϕnm_imag * ϕnm_imag) * L̃ * L̃ * (2-(m==0))
    # end
    # l_power = sqrt(abs(l_power))
    # l_power_phi_rotated_z = l_power

    # l_power = 0.0
    # for m in 0:n
    #     # de-normalization
    #     L̃ = sqrt(4*pi / (Float64(factorial(big(n-abs(m)))) * Float64(factorial(big(n+abs(m)))) * (2*n+1)) )

    #     # conservative local power
    #     χnm_real = weights_tmp_3[1,2,i+m]
    #     χnm_imag = weights_tmp_3[2,2,i+m]
    #     l_power += (χnm_real * χnm_real + χnm_imag * χnm_imag) * L̃ * L̃ * (1+(m>0))
    # end
    # l_power = sqrt(abs(l_power))
    # l_power_chi_rotated_z = l_power

    # @show l_power_phi_unrotated, l_power_phi_rotated_y, l_power_phi_rotated_z
    # @show l_power_chi_unrotated, l_power_chi_rotated_y, l_power_chi_rotated_z
    
    # #--- local error prediction: another term in the series ---#
    
    # l_power = 0.0
    # n=expansion_order + 2
    
    # # local power
    # i = (n*(n+1)) >> 1 + 1
    # for m in 0:n
    #     # de-normalization
    #     L̃ = sqrt(4*pi / (Float64(factorial(big(n-abs(m)))) * Float64(factorial(big(n+abs(m)))) * (2*n+1)) )

    #     # conservative local power
    #     ϕnm_real = weights_tmp_1[1,1,i+m]
    #     ϕnm_imag = weights_tmp_1[2,1,i+m]
    #     l_power += (ϕnm_real * ϕnm_real + ϕnm_imag * ϕnm_imag) * L̃ * L̃ * (2-(m==0))
    # end
    # l_power = sqrt(abs(l_power))

    # # local power error prediction
    # nfact = Float64(factorial(big(n)))
    # L̃0 = sqrt(4*pi/((2*n+1) * nfact * nfact))
    # r_l = target_branch.radius
    # println("\nSherlock!")
    # @show ε_l_power
    # ε_l_power += l_power / L̃0 * r_l^(n-1) / nfact * n * sqrt(3)
    # @show ε_l_power

    # if LH
        
    #     # local power
    #     l_power = 0.0
    #     i = (n*(n+1)) >> 1 + 1
    #     for m in 0:n
    #         # de-normalization
    #         L̃ = sqrt(4*pi / (Float64(factorial(big(n-abs(m)))) * Float64(factorial(big(n+abs(m)))) * (2*n+1)) )

    #         # conservative local power
    #         χnm_real = weights_tmp_1[1,2,i+m]
    #         χnm_imag = weights_tmp_1[2,2,i+m]
    #         l_power += (χnm_real * χnm_real + χnm_imag * χnm_imag) * L̃ * L̃ * (1+(m>0))
    #     end
    #     l_power = sqrt(abs(l_power))

    #     # local power error prediction
    #     nfact = Float64(factorial(big(n)))
    #     L̃0 = sqrt(4*pi/((2*n+1) * nfact * nfact))
    #     r_l = target_branch.radius
    #     ε_l_power += l_power / L̃0 * r_l^(n) / nfact * sqrt(3)
    #     @show ε_l_power
    # end

    return ε_mp_power * ONE_OVER_4π, ε_l_power * ONE_OVER_4π
end
