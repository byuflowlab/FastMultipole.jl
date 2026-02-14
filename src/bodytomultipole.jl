#------- multipole generation function -------#

#=

Convenience functions are accessed by overloading as follows:

body_to_multipole!(system::<mysystemtype>, args...) = body_to_multipole!(<geometry>{<kernel>}, system, args...)

For example, if my system type were called VortexSheetSurface and consisted of vortex panels, I would overload

body_to_multipole!(system::VortexSheetSurface, args...) = body_to_multipole!(Panel{Vortex}, system, args...)

=#

# @inline function body_to_multipole!(branch::SingleBranch, system, harmonics, expansion_order)
#     body_to_multipole!(system, branch, branch.bodies_index, harmonics, expansion_order)
# end

function body_to_multipole!(tree::Tree, harmonics, system, i_system, expansion_order)
    for i_branch in tree.leaf_index
        branch = tree.branches[i_branch]
        buffer = tree.buffers[i_system]
        multipole_coefficients = view(tree.expansions, :, :, :, i_branch)
        body_to_multipole!(system, multipole_coefficients, buffer, branch.center, branch.bodies_index[i_system], harmonics, expansion_order)
    end
end

# function body_to_multipole!(multipole_coefficients, branch::Branch, systems::Tuple, harmonics, expansion_order)
#     # iterate over systems
#     for (system, bodies_index) in zip(systems, branch.bodies_index)
#         body_to_multipole!(system, branch, bodies_index, harmonics, expansion_order)
#     end
# end

#------- multipole generation convenience functions -------#

#--- helper functions for integrating over simplices as in Gumerov 2023 ---#

function xyz_to_ξηz(x)
    ξ_real = x[1] * 0.5
    ξ_imag = x[2] * 0.5
    η_real = ξ_real
    η_imag = -ξ_imag
    z = x[3]
    return ξ_real, ξ_imag, η_real, η_imag, z
end

function get_n(harmonics, index, n, m, i, _1_m)
    if m > 0
        x_n_mm1_real = harmonics[1,index,i-1]
        x_n_mm1_imag = harmonics[2,index,i-1]
    elseif m == 0
        if n == 0
            x_n_mm1_real = zero(eltype(harmonics))
            x_n_mm1_imag = zero(eltype(harmonics))
        else
            x_n_mm1_real = -_1_m * harmonics[1,index,i+1]
            x_n_mm1_imag = _1_m * harmonics[2,index,i+1]
        end
    end

    x_n_m_real = harmonics[1,index,i]
    x_n_m_imag = harmonics[2,index,i]

    if m < n
        x_n_mp1_real = harmonics[1,index,i+1]
        x_n_mp1_imag = harmonics[2,index,i+1]
    else
        x_n_mp1_real = zero(eltype(harmonics))
        x_n_mp1_imag = zero(eltype(harmonics))
    end

    return x_n_mm1_real, x_n_mm1_imag, x_n_m_real, x_n_m_imag, x_n_mp1_real, x_n_mp1_imag
end


function get_nm1(harmonics, index, n, m, i_nm1_m, _1_m)
    if m > 0
        @assert 0 <= m-1 <= n-1
        x_nm1_mm1_real = harmonics[1,index,i_nm1_m-1]
        x_nm1_mm1_imag = harmonics[2,index,i_nm1_m-1]
    elseif n > 1
        @assert m-1==-1
        x_nm1_mm1_real = -_1_m * harmonics[1,index,i_nm1_m+1]
        x_nm1_mm1_imag = _1_m * harmonics[2,index,i_nm1_m+1]
    else
        x_nm1_mm1_real = zero(eltype(harmonics))
        x_nm1_mm1_imag = zero(eltype(harmonics))
    end

    if m < n
        @assert 0 <= m <= n-1
        x_nm1_m_real = harmonics[1,index,i_nm1_m]
        x_nm1_m_imag = harmonics[2,index,i_nm1_m]
    else
        x_nm1_m_real = zero(eltype(harmonics))
        x_nm1_m_imag = zero(eltype(harmonics))
    end

    if m+1 < n
        @assert 0 <= m+1 <= n-1
        x_nm1_mp1_real = harmonics[1,index,i_nm1_m+1]
        x_nm1_mp1_imag = harmonics[2,index,i_nm1_m+1]
    else
        x_nm1_mp1_real = zero(eltype(harmonics))
        x_nm1_mp1_imag = zero(eltype(harmonics))
    end

    return x_nm1_mm1_real, x_nm1_mm1_imag, x_nm1_m_real, x_nm1_m_imag, x_nm1_mp1_real, x_nm1_mp1_imag
end

function calculate_q!(harmonics, ξ_real, ξ_imag, η_real, η_imag, z, expansion_order)

    #--- n = 0, m = 0 ---#

    i = 1
    harmonics[1,1,i] = one(eltype(harmonics))
    harmonics[2,1,i] = zero(eltype(harmonics))
    i += 1

    #--- n > 0 ---#
    for n in 1:expansion_order
        i_nm1_m = i - n
        _1_m = 1.0
        for m in 0:n
            # get n-1 values
            q_nm1_mm1_real, q_nm1_mm1_imag, q_nm1_m_real, q_nm1_m_imag, q_nm1_mp1_real, q_nm1_mp1_imag = get_nm1(harmonics, 1, n, m, i_nm1_m, _1_m)

            # set coefficient
            harmonics[1,1,i] = (-(ξ_real * q_nm1_mm1_imag + ξ_imag * q_nm1_mm1_real) - (η_real * q_nm1_mp1_imag + η_imag * q_nm1_mp1_real) - z * q_nm1_m_real) / n
            harmonics[2,1,i] = ((ξ_real * q_nm1_mm1_real - ξ_imag * q_nm1_mm1_imag) + (η_real * q_nm1_mp1_real - η_imag * q_nm1_mp1_imag) - z * q_nm1_m_imag) / n

            # recurse
            i += 1
            i_nm1_m += 1
            _1_m = -_1_m
        end
    end
end

"""
assumes q has already been calculated

If X0_real is replaced with X0_real + Xv_real, etc., the result becomes jnm instead of pnm, as used for panels.
"""
function calculate_pj!(harmonics, ξ0_real, ξ0_imag, η0_real, η0_imag, z0, expansion_order)
    #--- n = 0, m = 0 ---#

    i = 1
    harmonics[1,2,1] = one(eltype(harmonics))
    harmonics[2,2,1] = zero(eltype(harmonics))
    i += 1

    #--- n > 0 ---#
    for n in 1:expansion_order
        i_nm1_m = i - n
        _1_m = 1
        for m in 0:n
            # get n-1 values
            p_nm1_mm1_real, p_nm1_mm1_imag, p_nm1_m_real, p_nm1_m_imag, p_nm1_mp1_real, p_nm1_mp1_imag = get_nm1(harmonics, 2, n, m, i_nm1_m, _1_m)

            # get qnm
            qnm_real = harmonics[1,1,i]
            qnm_imag = harmonics[2,1,i]

            # set coefficient
            harmonics[1,2,i] = (-(ξ0_real * p_nm1_mm1_imag + ξ0_imag * p_nm1_mm1_real) - (η0_real * p_nm1_mp1_imag + η0_imag * p_nm1_mp1_real) - z0 * p_nm1_m_real + qnm_real) / (n+1)
            harmonics[2,2,i] = ((ξ0_real * p_nm1_mm1_real - ξ0_imag * p_nm1_mm1_imag) + (η0_real * p_nm1_mp1_real - η0_imag * p_nm1_mp1_imag) - z0 * p_nm1_m_imag + qnm_imag) / (n+1)

            # recurse
            i += 1
            i_nm1_m += 1
            _1_m = -_1_m
        end
    end
end

"""
assumes j has already been calculated

Note: if X0_real is replaced with X0_real + Xw_real, etc., this becomes bnm (as used for volumes) instead of inm (as used for panels)
"""
function calculate_ib!(harmonics, ξ0_real, ξ0_imag, η0_real, η0_imag, z0, expansion_order)
    #--- n = 0, m = 0 ---#

    i = 1
    harmonics[1,1,1] = eltype(harmonics)(0.5)
    harmonics[2,1,1] = zero(eltype(harmonics))
    i += 1

    #--- n > 0 ---#
    for n in 1:expansion_order
        i_nm1_m = i - n
        _1_m = 1
        for m in 0:n
            # get n-1 values
            j_nm1_mm1_real, j_nm1_mm1_imag, j_nm1_m_real, j_nm1_m_imag, j_nm1_mp1_real, j_nm1_mp1_imag = get_nm1(harmonics, 1, n, m, i_nm1_m, _1_m)

            # get jnm
            jnm_real = harmonics[1,2,i]
            jnm_imag = harmonics[2,2,i]

            # set coefficient
            harmonics[1,1,i] = (-(ξ0_real * j_nm1_mm1_imag + ξ0_imag * j_nm1_mm1_real) - (η0_real * j_nm1_mp1_imag + η0_imag * j_nm1_mp1_real) - z0 * j_nm1_m_real + jnm_real) / (n+2)
            harmonics[2,1,i] = ((ξ0_real * j_nm1_mm1_real - ξ0_imag * j_nm1_mm1_imag) + (η0_real * j_nm1_mp1_real - η0_imag * j_nm1_mp1_imag) - z0 * j_nm1_m_imag + jnm_imag) / (n+2)

            # recurse
            i += 1
            i_nm1_m += 1
            _1_m = -_1_m
        end
    end
end

function calculate_a!(harmonics, ξ0_real, ξ0_imag, η0_real, η0_imag, z0, expansion_order)
    #--- n = 0, m = 0 ---#

    i = 1
    harmonics[1,2,1] = eltype(harmonics)(1/6)
    harmonics[2,2,1] = zero(eltype(harmonics))
    i += 1

    #--- n > 0 ---#
    for n in 1:expansion_order
        i_nm1_m = i - n
        _1_m = 1
        for m in 0:n
            # get n-1 values
            a_nm1_mm1_real, a_nm1_mm1_imag, a_nm1_m_real, a_nm1_m_imag, a_nm1_mp1_real, a_nm1_mp1_imag = get_nm1(harmonics, 2, n, m, i_nm1_m, _1_m)

            # get bnm
            bnm_real = harmonics[1,1,i]
            bnm_imag = harmonics[2,1,i]

            # set coefficient
            harmonics[1,2,i] = (-(ξ0_real * a_nm1_mm1_imag + ξ0_imag * a_nm1_mm1_real) - (η0_real * a_nm1_mp1_imag + η0_imag * a_nm1_mp1_real) - z0 * a_nm1_m_real + bnm_real) / (n+1)
            harmonics[2,2,i] = ((ξ0_real * a_nm1_mm1_real - ξ0_imag * a_nm1_mm1_imag) + (η0_real * a_nm1_mp1_real - η0_imag * a_nm1_mp1_imag) - z0 * a_nm1_m_imag + bnm_imag) / (n+1)

            # recurse
            i += 1
            i_nm1_m += 1
            _1_m = -_1_m
        end
    end
end


"""
assumes source expansion coefficients have already been calculated
"""
function source_to_dipole!(harmonics, i_dipole, i_source, qx, qy, qz, expansion_order)

    #--- n = 0 ---#

    harmonics[1,i_dipole,1] = 0.0
    harmonics[2,i_dipole,1] = 0.0

    #--- n > 0 ---#

    i = 2
    for n in 1:expansion_order
        i_nm1_m = i - n
        _1_m = 1
        for m in 0:n
            # get n-1 values
            p_nm1_mm1_real, p_nm1_mm1_imag, p_nm1_m_real, p_nm1_m_imag, p_nm1_mp1_real, p_nm1_mp1_imag = get_nm1(harmonics, i_source, n, m, i_nm1_m, _1_m)

            # set coefficient
            harmonics[1,i_dipole,i] = -qx * 0.5 * (p_nm1_mp1_imag + p_nm1_mm1_imag) + qy * 0.5 * (p_nm1_mp1_real - p_nm1_mm1_real) - qz * p_nm1_m_real
            harmonics[2,i_dipole,i] = qx * 0.5 * (p_nm1_mp1_real + p_nm1_mm1_real) + qy * 0.5 * (p_nm1_mp1_imag - p_nm1_mm1_imag) - qz * p_nm1_m_imag

            # recurse
            i += 1
            i_nm1_m += 1
            _1_m = -_1_m
        end
    end
end

"""
my derivation
"""
function mirrored_source_to_vortex!(multipole_coefficients, harmonics, strength, i_source, multiplier, expansion_order)

    #--- first potential ϕ ---#

    vx, vy, vz = strength

    i = 1
    for n in 0:expansion_order
        _1_m = 1.0
        _1_np1 = 1.0/(n+1)
        for m in 0:n
            # get source coefficient values
            q_n_mm1_real, q_n_mm1_imag, q_n_m_real, q_n_m_imag, q_n_mp1_real, q_n_mp1_imag = get_n(harmonics, i_source, n, m, i, _1_m)

            # set coefficient
            nmmp1_2 = Float64(n-m+1)*0.5
            npmp1_2 = Float64(n+m+1)*0.5
            multipole_coefficients[1,1,i] += multiplier * _1_m * ((-vx * q_n_mm1_real + vy * q_n_mm1_imag) * nmmp1_2 + (vx * q_n_mp1_real + vy * q_n_mp1_imag) * npmp1_2 - vz * m * q_n_m_imag) * _1_np1
            multipole_coefficients[2,1,i] += multiplier * _1_m * ((vx * q_n_mm1_imag + vy * q_n_mm1_real) * nmmp1_2 + (-vx * q_n_mp1_imag + vy * q_n_mp1_real) * npmp1_2 - vz * m * q_n_m_real) * _1_np1

            # if n == 0 && m == 0 && (abs(multiplier * _1_m * ((-vx * q_n_mm1_real + vy * q_n_mm1_imag) * nmmp1_2 + (vx * q_n_mp1_real + vy * q_n_mp1_imag) * npmp1_2 - vz * m * q_n_m_imag) * _1_np1) > 0.0 || (multiplier * _1_m * ((vx * q_n_mm1_imag + vy * q_n_mm1_real) * nmmp1_2 + (-vx * q_n_mp1_imag + vy * q_n_mp1_real) * npmp1_2 - vz * m * q_n_m_real) * _1_np1) > 0.0)
            #     throw("vortex element not zero for n=m=0")
            # end

            # recurse
            i += 1
            _1_m = -_1_m
        end
    end

    #--- second potential χ ---#

    i = 2
    for n in 1:expansion_order
        i_nm1_m = i - n
        _1_m = 1.0
        _1_over_n = 1.0/n
        for m in 0:n
            # get n-1 values
            q_nm1_mm1_real, q_nm1_mm1_imag, q_nm1_m_real, q_nm1_m_imag, q_nm1_mp1_real, q_nm1_mp1_imag = get_nm1(harmonics, i_source, n, m, i_nm1_m, _1_m)

            # set coefficient
            multipole_coefficients[1,2,i] -= multiplier * _1_m * _1_over_n * (0.5 * (-vy * q_nm1_mm1_real - vx * q_nm1_mm1_imag + vy * q_nm1_mp1_real - vx * q_nm1_mp1_imag) - vz * q_nm1_m_real)
            multipole_coefficients[2,2,i] -= multiplier * _1_m * _1_over_n * (0.5 * (vy * q_nm1_mm1_imag - vx * q_nm1_mm1_real - vy * q_nm1_mp1_imag - vx * q_nm1_mp1_real) + vz * q_nm1_m_imag)

            # recurse
            i += 1
            i_nm1_m += 1
            _1_m = -_1_m
        end
    end
end

"""
might be slightly faster than mirrored_source_to_vortex
"""
function source_to_vortex_point!(multipole_coefficients, harmonics, strength, Δx, i_source, i_dipole, expansion_order)

    #--- first scalar potential ---#

    # analogous dipole
    qx, qy, qz = cross(Δx, strength)
    source_to_dipole!(harmonics, i_dipole, i_source, qx, qy, qz, expansion_order)

    # update coefficients
    _1_n = -1.0
    i = 2
    for n in 1:expansion_order
        _1_n_m = _1_n
        _1_np1 = -1.0/(n+1)
        for m in 0:n
            multipole_coefficients[1,1,i] += harmonics[1,i_dipole,i] * _1_n_m * _1_np1
            multipole_coefficients[2,1,i] -= harmonics[2,i_dipole,i] * _1_n_m * _1_np1 # Rnm*
            _1_n_m = -_1_n_m
            i += 1
        end
        _1_n = -_1_n
    end

    #--- second scalar potential ---#

    # analogous dipole
    qx, qy, qz = -strength
    source_to_dipole!(harmonics, i_dipole, i_source, qx, qy, qz, expansion_order)

    # update coefficients
    _1_n = -1.0
    i = 2
    for n in 1:expansion_order
        _1_n_m = _1_n
        _1_over_n = -1.0/n
        for m in 0:n
            multipole_coefficients[1,2,i] += harmonics[1,i_dipole,i] * _1_n_m * _1_over_n
            multipole_coefficients[2,2,i] -= harmonics[2,i_dipole,i] * _1_n_m * _1_over_n # Rnm*
            _1_n_m = -_1_n_m
            i += 1
        end
        _1_n = -_1_n
    end

end

#--- point elements ---#

function body_to_multipole!(element::Type{<:Point}, system, multipole_coefficients, buffer::Matrix, center, bodies_index, harmonics, expansion_order)
    # loop over bodies
    for i_body in bodies_index
        # relative body position
        x = get_position(buffer, i_body)
        Δx = x - center

        # get strength
        strength = get_strength(buffer, system, i_body)

        # update values
        body_to_multipole_point!(element, multipole_coefficients, harmonics, Δx, strength, expansion_order)
    end

end

@inline function body_to_multipole_point!(::Type{Point{Source}}, multipole_coefficients, harmonics, Δx, strength, expansion_order)

    # transform to ξ, η, z
    ξ_real, ξ_imag, η_real, η_imag, z = xyz_to_ξηz(Δx)

    # invert the sign of the strength so v=∇ϕ instead of v=-∇ϕ
    # this will allow induced velocities to be added to those induced by vortex elements
    # we'll invert the potential again at the end
    strength = -strength[1]

    # calculate q (regular harmonics)
    calculate_q!(harmonics, ξ_real, ξ_imag, η_real, η_imag, z, expansion_order)

    # update coefficients
    _1_n = 1.0
    i = 1
    for n in 0:expansion_order
        _1_n_m = _1_n
        for m in 0:n
            multipole_coefficients[1,1,i] += harmonics[1,1,i] * _1_n_m * strength
            multipole_coefficients[2,1,i] -= harmonics[2,1,i] * _1_n_m * strength # Rnm*
            _1_n_m = -_1_n_m
            i += 1
        end
        _1_n = -_1_n
    end
end

@inline function body_to_multipole_point!(::Type{Point{Dipole}}, multipole_coefficients, harmonics, Δx, strength, expansion_order)

    # transform to ξ, η, z
    ξ_real, ξ_imag, η_real, η_imag, z = xyz_to_ξηz(Δx)

    # calculate q
    calculate_q!(harmonics, ξ_real, ξ_imag, η_real, η_imag, z, expansion_order)

    # invert the sign of the strength so v=∇ϕ instead of v=-∇ϕ
    # this will allow induced velocities to be added to those induced by vortex elements
    # we'll invert the potential again at the end
    strength = -strength

    # convert to dipole
    i_source, i_dipole = 1, 2
    qx, qy, qz = strength
    source_to_dipole!(harmonics, i_dipole, i_source, qx, qy, qz, expansion_order)

    # update coefficients
    _1_n = 1.0
    i = 1
    for n in 0:expansion_order
        _1_n_m = _1_n
        for m in 0:n
            multipole_coefficients[1,1,i] += harmonics[1,i_dipole,i] * _1_n_m
            multipole_coefficients[2,1,i] -= harmonics[2,i_dipole,i] * _1_n_m # Rnm*
            _1_n_m = -_1_n_m
            i += 1
        end
        _1_n = -_1_n
    end

end

@inline function body_to_multipole_point!(::Type{Point{Vortex}}, multipole_coefficients, harmonics, Δx, strength, expansion_order)

    ## transform to ξ, η, z
    #ξ_real, ξ_imag, η_real, η_imag, z = xyz_to_ξηz(Δx)

    ## calculate q (regular harmonics for source point)
    #calculate_q!(harmonics, ξ_real, ξ_imag, η_real, η_imag, z, expansion_order)

    ## convert to vortex
    #i_source, i_dipole = 1, 2
    #source_to_vortex_point!(multipole_coefficients, harmonics, strength, Δx, i_source, i_dipole, expansion_order)

    # not sure if above or below is more efficient, but below generalizes to filaments/panels/volumes

    # transform mirrored geometry to ξ, η, z
    ξ_real, ξ_imag, η_real, η_imag, z = xyz_to_ξηz(-Δx)

    # calculate q (regular harmonics for source point)
    calculate_q!(harmonics, ξ_real, ξ_imag, η_real, η_imag, z, expansion_order)

    # convert to vortex
    i_source = 1
    multiplier = 1.0
    mirrored_source_to_vortex!(multipole_coefficients, harmonics, strength, i_source, multiplier, expansion_order)
end

@inline function body_to_multipole_point!(::Type{Point{SourceVortex}}, multipole_coefficients, harmonics, Δx, strength, expansion_order)
    scalar_strength = strength[1]
    vector_strength = SVector{3}(strength[2], strength[3], strength[4])
    body_to_multipole_point!(Point{Source}, multipole_coefficients, harmonics, Δx, scalar_strength, expansion_order)
    body_to_multipole_point!(Point{Vortex}, multipole_coefficients, harmonics, Δx, vector_strength, expansion_order)
end

#--- filament elements ---#

function body_to_multipole!(element::Type{<:Filament}, system, multipole_coefficients, buffer, center, bodies_index, harmonics, expansion_order)

    # loop over bodies
    for i_body in bodies_index

        # relative body position
        i1 = 5 + strength_dims(system)
        x1 = SVector{3}(buffer[i1,i_body], buffer[i1+1,i_body], buffer[i1+2,i_body])
        x2 = SVector{3}(buffer[i1+3,i_body], buffer[i1+4,i_body], buffer[i1+5,i_body])

        # strength
        strength = get_strength(buffer, system, i_body)

        # delta
        xu = x2 - x1
        x0 = x1 - center

        # update values
        body_to_multipole_filament!(element, multipole_coefficients, harmonics, x0, xu, strength, expansion_order)
    end
end

function body_to_multipole_filament!(::Type{Filament{Source}}, multipole_coefficients, harmonics, x0, xu, strength, expansion_order)
    # transform to ξ, η, z
    ξ0_real, ξ0_imag, η0_real, η0_imag, z0 = xyz_to_ξηz(x0)
    ξu_real, ξu_imag, ηu_real, ηu_imag, zu = xyz_to_ξηz(xu)

    ξ0ξu_real = ξ0_real + ξu_real
    ξ0ξu_imag = ξ0_imag + ξu_imag
    η0ηu_real = η0_real + ηu_real
    η0ηu_imag = η0_imag + ηu_imag
    z0zu = z0 + zu

    # calculate q
    calculate_q!(harmonics, ξ0ξu_real, ξ0ξu_imag, η0ηu_real, η0ηu_imag, z0zu, expansion_order)

    # calculate p
    calculate_pj!(harmonics, ξ0_real, ξ0_imag, η0_real, η0_imag, z0, expansion_order)

    # invert the sign of the strength so v=∇ϕ instead of v=-∇ϕ
    # this will allow induced velocities to be added to those induced by vortex elements
    # we'll invert the potential again at the end
    strength = -strength[1]

    # jacobian of the transformation to simplex
    Δx = norm(xu)
    Jq = Δx * strength

    # update expansion coefficients
    i = 1
    _1_n = 1.0
    for n in 0:expansion_order
        _1_n_m = _1_n
        for m in 0:n
            # set coefficients
            multipole_coefficients[1,1,i] += Jq * _1_n_m * harmonics[1,2,i]
            multipole_coefficients[2,1,i] -= Jq * _1_n_m * harmonics[2,2,i]

            # recurse
            i += 1
            _1_n_m = -_1_n_m
        end
        _1_n = -_1_n
    end
end

function body_to_multipole_filament!(::Type{Filament{Dipole}}, multipole_coefficients, harmonics, x0, xu, strength, expansion_order)
    # transform to ξ, η, z
    ξ0_real, ξ0_imag, η0_real, η0_imag, z0 = xyz_to_ξηz(x0)
    ξu_real, ξu_imag, ηu_real, ηu_imag, zu = xyz_to_ξηz(xu)

    ξ0ξu_real = ξ0_real + ξu_real
    ξ0ξu_imag = ξ0_imag + ξu_imag
    η0ηu_real = η0_real + ηu_real
    η0ηu_imag = η0_imag + ηu_imag
    z0zu = z0 + zu

    # calculate q
    calculate_q!(harmonics, ξ0ξu_real, ξ0ξu_imag, η0ηu_real, η0ηu_imag, z0zu, expansion_order)

    # calculate p
    calculate_pj!(harmonics, ξ0_real, ξ0_imag, η0_real, η0_imag, z0, expansion_order)

    # invert the sign of the strength so v=∇ϕ instead of v=-∇ϕ
    # this will allow induced velocities to be added to those induced by vortex elements
    # we'll invert the potential again at the end
    strength = -strength

    # calculate s
    qx, qy, qz = strength
    i_source, i_dipole = 2, 1
    source_to_dipole!(harmonics, i_dipole, i_source, qx, qy, qz, expansion_order)

    # jacobian of the transformation to simplex
    J = norm(xu)

    # update expansion coefficients
    i = 1
    _1_n = 1.0
    for n in 0:expansion_order
        _1_n_m = _1_n
        for m in 0:n
            # set coefficients
            multipole_coefficients[1,1,i] += J * _1_n_m * harmonics[1,i_dipole,i]
            multipole_coefficients[2,1,i] -= J * _1_n_m * harmonics[2,i_dipole,i]

            # recurse
            i += 1
            _1_n_m = -_1_n_m
        end
        _1_n = -_1_n
    end
end

function body_to_multipole_filament!(::Type{Filament{Vortex}}, multipole_coefficients, harmonics, x0, xu, strength, expansion_order)

    # transform to ξ, η, z
    ξ0_real, ξ0_imag, η0_real, η0_imag, z0 = xyz_to_ξηz(-x0)
    ξu_real, ξu_imag, ηu_real, ηu_imag, zu = xyz_to_ξηz(-xu)

    ξ0ξu_real = ξ0_real + ξu_real
    ξ0ξu_imag = ξ0_imag + ξu_imag
    η0ηu_real = η0_real + ηu_real
    η0ηu_imag = η0_imag + ηu_imag
    z0zu = z0 + zu

    # calculate q
    calculate_q!(harmonics, ξ0ξu_real, ξ0ξu_imag, η0ηu_real, η0ηu_imag, z0zu, expansion_order)

    # calculate p
    calculate_pj!(harmonics, ξ0_real, ξ0_imag, η0_real, η0_imag, z0, expansion_order)

    # jacobian of the transformation to simplex
    J = norm(xu)

    # convert to vortex
    i_source = 2
    mirrored_source_to_vortex!(multipole_coefficients, harmonics, strength, i_source, J, expansion_order)

end

#--- panel elements ---#

function body_to_multipole!(element::Type{<:Panel}, system, multipole_coefficients, buffer, center, bodies_index, harmonics, expansion_order)
    # loop over bodies
    for i_body in bodies_index
        # relative body position
        # i1 = 5 + strength_dims(system)
        # x1 = SVector{3}(buffer[i1,i_body], buffer[i1+1,i_body], buffer[i1+2,i_body])
        # x2 = SVector{3}(buffer[i1+3,i_body], buffer[i1+4,i_body], buffer[i1+5,i_body])
        # x3 = SVector{3}(buffer[i1+6,i_body], buffer[i1+7,i_body], buffer[i1+8,i_body])
        x1 = get_vertex(buffer, system, i_body, 1)
        x2 = get_vertex(buffer, system, i_body, 2)
        x3 = get_vertex(buffer, system, i_body, 3)
        x0 = x1 - center
        xu = x2 - x1
        xv = x3 - x1

        # get normal
        normal = get_normal(buffer, system, i_body)

        # get strength
        strength = get_strength(buffer, system, i_body)

        # update values
        body_to_multipole_panel!(element, multipole_coefficients, harmonics, x0, xu, xv, normal, strength, expansion_order)
    end
end

function body_to_multipole_quad!(element::Type{<:Panel}, system, multipole_coefficients, buffer, center, bodies_index, harmonics, expansion_order)
    # loop over bodies
    for i_body in bodies_index
        
        #--- first triangle ---#

        # relative body position
        # i1 = 5 + strength_dims(system)
        # x1 = SVector{3}(buffer[i1,i_body], buffer[i1+1,i_body], buffer[i1+2,i_body])
        # x2 = SVector{3}(buffer[i1+3,i_body], buffer[i1+4,i_body], buffer[i1+5,i_body])
        # x3 = SVector{3}(buffer[i1+6,i_body], buffer[i1+7,i_body], buffer[i1+8,i_body])
        x1 = get_vertex(buffer, system, i_body, 1)
        x2 = get_vertex(buffer, system, i_body, 2)
        x3 = get_vertex(buffer, system, i_body, 3)
        x0 = x1 - center
        xu = x2 - x1
        xv = x3 - x1

        # get normal
        normal = cross(x2-x1, x3-x1)
        normal /= norm(normal)

        # get strength
        strength = get_strength(buffer, system, i_body)

        # update values
        body_to_multipole_panel!(element, multipole_coefficients, harmonics, x0, xu, xv, normal, strength, expansion_order)

        #--- second triangle ---#

        x1 = get_vertex(buffer, system, i_body, 1)
        x2 = get_vertex(buffer, system, i_body, 3)
        x3 = get_vertex(buffer, system, i_body, 4)
        x0 = x1 - center
        xu = x2 - x1
        xv = x3 - x1

        # get normal
        normal = cross(x2-x1, x3-x1) 
        normal /= norm(normal)

        # update values
        body_to_multipole_panel!(element, multipole_coefficients, harmonics, x0, xu, xv, normal, strength, expansion_order)
    end
end

function body_to_multipole_panel!(::Type{Panel{SourceDipole}}, multipole_coefficients, harmonics, x0, xu, xv, normal, strength, expansion_order)
    body_to_multipole_panel!(Panel{Source}, multipole_coefficients, harmonics, x0, xu, xv, normal, SVector{1}(strength[1]), expansion_order)
    body_to_multipole_panel!(Panel{Dipole}, multipole_coefficients, harmonics, x0, xu, xv, normal, SVector{1}(strength[2]), expansion_order)
end

function body_to_multipole_panel!(::Type{Panel{Source}}, multipole_coefficients, harmonics, x0, xu, xv, normal, strength, expansion_order)
    # transform to ξ, η, z
    ξ0_real, ξ0_imag, η0_real, η0_imag, z0 = xyz_to_ξηz(x0)
    ξu_real, ξu_imag, ηu_real, ηu_imag, zu = xyz_to_ξηz(xu)
    ξv_real, ξv_imag, ηv_real, ηv_imag, zv = xyz_to_ξηz(xv)

    # calculate q
    ξ0ξu_real = ξ0_real + ξu_real
    ξ0ξu_imag = ξ0_imag + ξu_imag
    η0ηu_real = η0_real + ηu_real
    η0ηu_imag = η0_imag + ηu_imag
    z0zu = z0 + zu

    calculate_q!(harmonics, ξ0ξu_real, ξ0ξu_imag, η0ηu_real, η0ηu_imag, z0zu, expansion_order)

    # calculate j
    ξ0ξv_real = ξ0_real + ξv_real
    ξ0ξv_imag = ξ0_imag + ξv_imag
    η0ηv_real = η0_real + ηv_real
    η0ηv_imag = η0_imag + ηv_imag
    z0zv = z0 + zv

    calculate_pj!(harmonics, ξ0ξv_real, ξ0ξv_imag, η0ηv_real, η0ηv_imag, z0zv, expansion_order)

    # calculate i
    calculate_ib!(harmonics, ξ0_real, ξ0_imag, η0_real, η0_imag, z0, expansion_order)

    # invert the sign of the strength so v=∇ϕ instead of v=-∇ϕ
    # this will allow induced velocities to be added to those induced by vortex elements
    # we'll invert the potential again at the end
    strength = -strength[1]

    # jacobian of the transformation to simplex
    Δx = norm(cross(xu,xv))
    Jq = Δx * strength

    # update expansion coefficients
    i = 1
    _1_n = 1.0
    for n in 0:expansion_order
        _1_n_m = _1_n
        for m in 0:n
            # set coefficients
            multipole_coefficients[1,1,i] += Jq * _1_n_m * harmonics[1,1,i]
            multipole_coefficients[2,1,i] -= Jq * _1_n_m * harmonics[2,1,i]

            # recurse
            i += 1
            _1_n_m = -_1_n_m
        end
        _1_n = -_1_n
    end
end

function body_to_multipole_panel!(::Type{Panel{Dipole}}, multipole_coefficients, harmonics, x0, xu, xv, normal, strength, expansion_order)
    # transform to ξ, η, z
    ξ0_real, ξ0_imag, η0_real, η0_imag, z0 = xyz_to_ξηz(x0)
    ξu_real, ξu_imag, ηu_real, ηu_imag, zu = xyz_to_ξηz(xu)
    ξv_real, ξv_imag, ηv_real, ηv_imag, zv = xyz_to_ξηz(xv)

    # calculate q
    ξ0ξu_real = ξ0_real + ξu_real
    ξ0ξu_imag = ξ0_imag + ξu_imag
    η0ηu_real = η0_real + ηu_real
    η0ηu_imag = η0_imag + ηu_imag
    z0zu = z0 + zu

    calculate_q!(harmonics, ξ0ξu_real, ξ0ξu_imag, η0ηu_real, η0ηu_imag, z0zu, expansion_order)

    # calculate j
    ξ0ξv_real = ξ0_real + ξv_real
    ξ0ξv_imag = ξ0_imag + ξv_imag
    η0ηv_real = η0_real + ηv_real
    η0ηv_imag = η0_imag + ηv_imag
    z0zv = z0 + zv

    calculate_pj!(harmonics, ξ0ξv_real, ξ0ξv_imag, η0ηv_real, η0ηv_imag, z0zv, expansion_order)

    # calculate i
    calculate_ib!(harmonics, ξ0_real, ξ0_imag, η0_real, η0_imag, z0, expansion_order)

    # invert the sign of the strength so v=∇ϕ instead of v=-∇ϕ
    # this will allow induced velocities to be added to those induced by vortex elements
    # we'll invert the potential again at the end
    qx, qy, qz = normal * strength[1]

    # convert to dipole coefficients
    i_source, i_dipole = 1, 2
    source_to_dipole!(harmonics, i_dipole, i_source, qx, qy, qz, expansion_order)

    # jacobian of the transformation to simplex
    J = norm(cross(xu,xv))

    # update expansion coefficients
    i = 1
    _1_n = 1.0
    for n in 0:expansion_order
        _1_n_m = _1_n
        for m in 0:n
            # set coefficients
            multipole_coefficients[1,1,i] += J * _1_n_m * harmonics[1,i_dipole,i]
            multipole_coefficients[2,1,i] -= J * _1_n_m * harmonics[2,i_dipole,i]

            # recurse
            i += 1
            _1_n_m = -_1_n_m
        end
        _1_n = -_1_n
    end
end

function body_to_multipole_panel!(::Type{Panel{Vortex}}, multipole_coefficients, harmonics, x0, xu, xv, normal, strength, expansion_order)
    # transform to ξ, η, z
    ξ0_real, ξ0_imag, η0_real, η0_imag, z0 = xyz_to_ξηz(-x0)
    ξu_real, ξu_imag, ηu_real, ηu_imag, zu = xyz_to_ξηz(-xu)

    # calculate q
    ξ0ξu_real = ξ0_real + ξu_real
    ξ0ξu_imag = ξ0_imag + ξu_imag
    η0ηu_real = η0_real + ηu_real
    η0ηu_imag = η0_imag + ηu_imag
    z0zu = z0 + zu

    calculate_q!(harmonics, ξ0ξu_real, ξ0ξu_imag, η0ηu_real, η0ηu_imag, z0zu, expansion_order)

    # calculate j
    ξv_real, ξv_imag, ηv_real, ηv_imag, zv = xyz_to_ξηz(-xv)
    ξ0ξv_real = ξ0_real + ξv_real
    ξ0ξv_imag = ξ0_imag + ξv_imag
    η0ηv_real = η0_real + ηv_real
    η0ηv_imag = η0_imag + ηv_imag
    z0zv = z0 + zv

    calculate_pj!(harmonics, ξ0ξv_real, ξ0ξv_imag, η0ηv_real, η0ηv_imag, z0zv, expansion_order)

    # calculate i
    calculate_ib!(harmonics, ξ0_real, ξ0_imag, η0_real, η0_imag, z0, expansion_order)

    # TODO
    # jacobian of the transformation to simplex
    J = norm(cross(xu,xv))

    # convert to vortex
    i_source = 1
    mirrored_source_to_vortex!(multipole_coefficients, harmonics, strength, i_source, J, expansion_order)

end
