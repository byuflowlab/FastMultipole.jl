#------- MULTIPOLE TO MULTIPOLE -------#

"""
Overwrites translated_weights
"""
function translate_multipole_z!(translated_weights, source_weights, t, expansion_order, ::Val{LH}) where LH
#function translate_multipole_z!(translated_weights, source_weights, t, expansion_order::Val{P}, ::Val{LH}) where {P,LH}
    _t = -t
    i = 1
    for n in 0:expansion_order
    	for m in 0:n
    		# preallocate recursive variables
            val1_real = zero(eltype(translated_weights))
            val1_imag = zero(eltype(translated_weights))
            if LH
                val2_real = zero(eltype(translated_weights))
                val2_imag = zero(eltype(translated_weights))
            end
    		n_np! = 1.0
    		n_np = 1
    		_t_n_np = one(t)

            for np in n:-1:m
                tmp = _t_n_np / n_np!
                i_weight = harmonic_index(np,m)
                val1_real += tmp * source_weights[1,1,i_weight]
                val1_imag += tmp * source_weights[2,1,i_weight]
                if LH
                    val2_real += tmp * source_weights[1,2,i_weight]
                    val2_imag += tmp * source_weights[2,2,i_weight]
                end
    			_t_n_np *= _t
    			n_np! *= n_np
    			n_np += 1
    		end

            # set translated coefficient
            translated_weights[1,1,i] = val1_real
            translated_weights[2,1,i] = val1_imag

            if LH
                translated_weights[1,2,i] = val2_real
                translated_weights[2,2,i] = val2_imag
            end

            # increment index
            i += 1
    	end
    end
end

function transform_lamb_helmholtz_multipole!(multipole_expansion, r, expansion_order)
    i_P_m = harmonic_index(expansion_order, 0)
    for m in 0:expansion_order
        # declare recursive variable for inner loop over n
        χ̂_nm1_real = multipole_expansion[1,2,i_P_m]
        χ̂_nm1_imag = multipole_expansion[2,2,i_P_m]

        # declare index
        i_n_m = i_P_m

        for n in expansion_order:-1:max(m,1)
            # transform ϕ potential
            ϕ̂_real = multipole_expansion[1,1,i_n_m]
            ϕ̂_imag = multipole_expansion[2,1,i_n_m]

            # recursively access χ̂ to reduce cost
            χ̂_real = χ̂_nm1_real
            χ̂_imag = χ̂_nm1_imag

            # ϕ̃ = ϕ̂ - im * r * m/(n+1) * χ̂
            r_m_np1 = r * m / (n+1)
            multipole_expansion[1,1,i_n_m] = ϕ̂_real + r_m_np1 * χ̂_imag
            multipole_expansion[2,1,i_n_m] = ϕ̂_imag - r_m_np1 * χ̂_real

            # recurse χ̂_nm1
            if m < n
                χ̂_nm1_real = multipole_expansion[1,2,i_n_m-n]
                χ̂_nm1_imag = multipole_expansion[2,2,i_n_m-n]
            else # χ_{n-1}^m doesn't exist
                χ̂_nm1_real = zero(eltype(multipole_expansion))
                χ̂_nm1_imag = zero(eltype(multipole_expansion))
            end

            # χ̃ = χ̂ + r/n * χ̂_nm1
            r_n = r / n
            multipole_expansion[1,2,i_n_m] = χ̂_real + r_n * χ̂_nm1_real
            multipole_expansion[2,2,i_n_m] = χ̂_imag + r_n * χ̂_nm1_imag

            # update index
            i_n_m -= n

        end
        i_P_m += 1
    end
end

function transform_lamb_helmholtz_local!(local_expansion, r, expansion_order)
    i_m_m = 1
    for m in 0:expansion_order
        # declare recursive variable for inner loop over n
        χ̂_np1_real = local_expansion[1,2,i_m_m]
        χ̂_np1_imag = local_expansion[2,2,i_m_m]

        # declare index for fast access
        i_n_m = i_m_m

        for n in m:expansion_order # skip n=0
            # recursively access χ̂ to reduce cost
            χ̂_real = χ̂_np1_real
            χ̂_imag = χ̂_np1_imag

            if n > 0
                # transform ϕ potential
                ϕ̂_real = local_expansion[1,1,i_n_m]
                ϕ̂_imag = local_expansion[2,1,i_n_m]

                # ϕ̃ = ϕ̂ + im * r * m/n * χ̂
                r_m_n = r * m / n
                local_expansion[1,1,i_n_m] = ϕ̂_real - r_m_n * χ̂_imag
                local_expansion[2,1,i_n_m] = ϕ̂_imag + r_m_n * χ̂_real
            end

            if n < expansion_order # χ_{n+1}^m = 0 when n+1>P
                # recurse χ̂_np1
                χ̂_np1_real = local_expansion[1,2,i_n_m+n+1]
                χ̂_np1_imag = local_expansion[2,2,i_n_m+n+1]

                # χ̃ = χ̂ - r/(n+1) * χ̂_nm1
                r_np1 = r / (n+1)
                local_expansion[1,2,i_n_m] = χ̂_real - r_np1 * χ̂_np1_real
                local_expansion[2,2,i_n_m] = χ̂_imag - r_np1 * χ̂_np1_imag
            end

            # update index
            i_n_m += n + 1

        end

        i_m_m += m + 2
    end
end

function multipole_to_multipole!(target_weights, target_branch, source_weights, source_branch, weights_tmp_1, weights_tmp_2, Ts, eimϕs, ζs_mag, Hs_π2, expansion_order, lamb_helmholtz::Val{LH}) where LH
    # translation vector
    Δx = target_branch.center - source_branch.center
    r, θ, ϕ = cartesian_to_spherical(Δx)

    #--- rotate coordinate system ---#

    # rotate about z axis
    rotate_z!(weights_tmp_1, source_weights, eimϕs, ϕ, expansion_order, lamb_helmholtz)

    # rotate about y axis
    # NOTE: the method used here results in an additional rotatation of π about the new z axis
    rotate_multipole_y!(weights_tmp_2, weights_tmp_1, Ts, Hs_π2, ζs_mag, θ, expansion_order, lamb_helmholtz)

    #--- translate along new z axis ---#

    translate_multipole_z!(weights_tmp_1, weights_tmp_2, r, expansion_order, lamb_helmholtz)

    #--- transform Lamb-Helmholtz decomposition for the new center ---#

    LH && transform_lamb_helmholtz_multipole!(weights_tmp_1, r, expansion_order)

    #--- back rotate coordinate system ---#

    # back rotate about y axis
    back_rotate_multipole_y!(weights_tmp_2, weights_tmp_1, Ts, ζs_mag, expansion_order, lamb_helmholtz)

    # back rotate about z axis and accumulate on target branch
    back_rotate_z!(target_weights, weights_tmp_2, eimϕs, expansion_order, lamb_helmholtz)

end

#------- MULTIPOLE TO LOCAL -------#

"""
Overwrites translated_weights
"""
function translate_multipole_to_local_z!(translated_weights, source_weights, t, expansion_order, ::Val{LH}) where LH
    one_over_t = 1/t
    n!_t_np1 = one_over_t
    i = 1
    for n in 0:expansion_order
        n_m!_t_nmp1 = n!_t_np1
        n_m = n
        
    	for m in 0:n
            # inner summation
            val1_real = zero(eltype(translated_weights))
            val1_imag = zero(eltype(translated_weights))

            if LH
                val2_real = zero(eltype(translated_weights))
                val2_imag = zero(eltype(translated_weights))
            end
    		n_np! = n_m!_t_nmp1
            n_np = n_m
    		for np in m:expansion_order
                i_weight = harmonic_index(np,m)
                val1_real += n_np! * source_weights[1,1,i_weight]
                val1_imag += n_np! * source_weights[2,1,i_weight]

                if LH
                    val2_real += n_np! * source_weights[1,2,i_weight]
                    val2_imag += n_np! * source_weights[2,2,i_weight]
                end

                n_np += 1
                n_np! *= n_np * one_over_t
    		end

            # set translated coefficient
            translated_weights[1,1,i] = val1_real
            translated_weights[2,1,i] = val1_imag

            if LH
                translated_weights[1,2,i] = val2_real
                translated_weights[2,2,i] = val2_imag
            end

            # increment index
            i += 1

            # recurse values
            n_m += 1
            n_m!_t_nmp1 *= n_m * one_over_t
    	end
        n!_t_np1 *= (n+1)*one_over_t
    end
end


"""
Overwrites translated_weights
"""
function translate_multipole_to_local_z_n!(translated_weights, source_weights, one_over_t, n!_t_np1, n, ::Val{LH}) where LH

    # first index
    i = (n * (n+1)) >> 1 + 1

    # recursive values
    n_m!_t_nmp1 = n!_t_np1
    n_m = n

    # all coefficients with the same m are required for translation
    for m in 0:n
        # inner summation
        val1_real = zero(eltype(translated_weights))
        val1_imag = zero(eltype(translated_weights))

        if LH
            val2_real = zero(eltype(translated_weights))
            val2_imag = zero(eltype(translated_weights))
        end
        n_np! = n_m!_t_nmp1
        n_np = n_m

        for np in m:n
            i_weight = harmonic_index(np,m)
            val1_real += n_np! * source_weights[1,1,i_weight]
            val1_imag += n_np! * source_weights[2,1,i_weight]

            if LH
                val2_real += n_np! * source_weights[1,2,i_weight]
                val2_imag += n_np! * source_weights[2,2,i_weight]
            end

            n_np += 1
            n_np! *= n_np * one_over_t
        end

        # set translated coefficient
        translated_weights[1,1,i] = val1_real
        translated_weights[2,1,i] = val1_imag

        if LH
            translated_weights[1,2,i] = val2_real
            translated_weights[2,2,i] = val2_imag
        end

        # increment index
        i += 1

        # recurse values
        n_m += 1
        n_m!_t_nmp1 *= n_m * one_over_t
    end

    # return recursive values for next time
    n!_t_np1 *= (n+1)*one_over_t
 
    return n!_t_np1
end

"""
This function should receive n!_t_np1 = 1/t when n=0

Calculates ϕn0, ϕn1, χn1 for P=n
"""
function translate_multipole_to_local_z_m01_n(source_weights, t, one_over_t, ::Val{LH}, n!_t_np1, n) where LH
    # initialize values
    n_m!_t_nmp1 = n!_t_np1
    n_m = n

    #--- m = 0 ---#

    # inner summation
    ϕn0_real = zero(eltype(source_weights))

    n_np! = n_m!_t_nmp1
    n_np = n_m
    for np in 0:n
        i_weight = harmonic_index(np,0)
        ϕn0_real += n_np! * source_weights[1,1,i_weight]

        n_np += 1
        n_np! *= n_np * one_over_t
    end

    # recurse values
    n_m += 1
    n_m!_t_nmp1 *= n_m * one_over_t

    #--- m = 1 ---#

    # inner summation
    ϕn1_real = zero(eltype(source_weights))
    ϕn1_imag = zero(eltype(source_weights))
    χn1_real = zero(eltype(source_weights))
    χn1_imag = zero(eltype(source_weights))

    n_np! = n_m!_t_nmp1
    n_np = n_m
    for np in 1:n
        i_weight = harmonic_index(np,1)
        ϕn1_real += n_np! * source_weights[1,1,i_weight]
        ϕn1_imag += n_np! * source_weights[2,1,i_weight]

        if LH
            χn1_real += n_np! * source_weights[1,2,i_weight]
            χn1_imag += n_np! * source_weights[2,2,i_weight]
        end

        n_np += 1
        n_np! *= n_np * one_over_t
    end

    if LH
        t_over_n = t / n
        ϕn1_real -= t_over_n * χn1_imag
        ϕn1_imag += t_over_n * χn1_real

        # ordinarily, χn1 should also be modified here
        # (see Gumerov 2013, Eq. 48)
        # however, we're assuming, for now, that P=n,
        # so χ_{n+1}^m=0 and the correction reduces to 0
    end

    return ϕn0_real, ϕn1_real, ϕn1_imag, χn1_real, χn1_imag
end

#--- multipole power normalization ---#

function length_M̃(expansion_order)
    return ((expansion_order+1) * (expansion_order+2)) >> 1
end

function update_M̃!(M̃, expansion_order)
    l = length(M̃)
    l_desired = length_M̃(expansion_order)

    if l_desired > l # we need to compute more matrices
        # determine how many we already have
        n_already, l_already = 0, 0
        for n in 0:expansion_order
            l_already += (n+1) * (n+1)
            l_already == l && (n_already = n)
        end

        # calculate the rest
        resize!(M̃, l_desired)
        update_M̃!(M̃, n_already+1, expansion_order)

    end # otherwise, we already have enough, so do nothing
end

function update_M̃!(M̃, n_start, n_end)
    # starting index
    i = (n_start * (n_start+1)) >> 1 + 1

    # loop over degree n
    for n in n_start:n_end

        # loop over order m
        for m in 0:n
            # calculate the multipole power normalization
            M̃[i] = sqrt(Float64(factorial(big(n+m))) * Float64(factorial(big(n-m))) / (2*n + 1))

            # recurse index
            i += 1
        end
    end
end

#--- local power normalization ---#

function length_L̃(expansion_order)
    return ((expansion_order+1) * (expansion_order+2)) >> 1
end

function update_L̃!(L̃, expansion_order)
    l = length(L̃)
    l_desired = length_L̃(expansion_order)

    if l_desired > l # we need to compute more matrices
        # determine how many we already have
        n_already, l_already = 0, 0
        for n in 0:expansion_order
            l_already += (n+1) * (n+1)
            l_already == l && (n_already = n)
        end

        # calculate the rest
        resize!(L̃, l_desired)
        update_L̃!(L̃, n_already+1, expansion_order)

    end # otherwise, we already have enough, so do nothing
end

function update_L̃!(L̃, n_start, n_end)
    # starting index
    i = (n_start * (n_start+1)) >> 1 + 1

    # loop over degree n
    for n in n_start:n_end

        # loop over order m
        for m in 0:n
            # calculate the multipole power normalization
            L̃[i] = sqrt(1 / (Float64(factorial(big(n+m))) * Float64(factorial(big(n-m))) * (2*n + 1)))

            # recurse index
            i += 1
        end
    end
end

#--- local powers ---#

"""
performs the lamb-helmholtz transformation assuming that the coordinate system is still aligned with the z axis
"""
function local_power(weights, r, n, L̃, ::Val{LH}) where LH
    l_power_ϕ = zero(eltype(weights))
    l_power_χ = zero(eltype(weights))
    Ñ = zero(eltype(L̃))

    # initialize χ coefficients for recursive access
    χ_real_np1, χ_imag_np1 = zero(eltype(weights)), zero(eltype(weights))

    # index
    i = (n * (n+1)) >> 1 + 1
    
    for m in n:-1:0
        # local normalization
        Ñ = L̃[i+m]

        # local power
        ϕ_real, ϕ_imag = weights[1,1,i+m], weights[2,1,i+m]
        
        if LH
            # extract χ coefficients
            χ_real_old, χ_imag_old = weights[1,2,i+m], weights[2,2,i+m]

            # transform coefficient under LH
            ϕ_real = ϕ_real - χ_imag_old * r * m / n
            ϕ_imag = ϕ_imag + χ_real_old * r * m / n
            χ_real = χ_real_old - χ_real_np1 * r / (n+1)
            χ_imag = χ_imag_old - χ_imag_np1 * r / (n+1)

            # recurse χ_np1
            χ_real_np1, χ_imag_np1 = χ_real_old, χ_imag_old

            # calculate local power
            l_power_χ += (χ_real * χ_real + χ_imag * χ_imag) * Ñ * Ñ * (1 + m>0)
        end
        
        # calculate local power
        l_power_ϕ += (ϕ_real * ϕ_real + ϕ_imag * ϕ_imag) * Ñ * Ñ * (1 + m>0)

    end
    
    # sqrt
    l_power_ϕ = sqrt(l_power_ϕ)
    if LH
        l_power_χ = sqrt(l_power_χ)
    end
    
    return l_power_ϕ, l_power_χ, Ñ
end

#--- multipole-to-local translation ---#

function multipole_to_local!(target_weights, target_branch::Branch{TF}, source_weights, source_branch, expansion_order, lamb_helmholtz, ε) where TF
    weights_tmp_1 = initialize_expansion(expansion_order, TF)
    weights_tmp_2 = initialize_expansion(expansion_order, TF)
    weights_tmp_3 = initialize_expansion(expansion_order, TF)
    Ts = zeros(TF, length_Ts(expansion_order))
    eimϕs = zeros(TF, 2, expansion_order+1)

    return multipole_to_local!(target_weights, target_branch, source_weights, source_branch, weights_tmp_1, weights_tmp_2, weights_tmp_3, Ts, eimϕs, ζs_mag, ηs_mag, Hs_π2, M̃, L̃, expansion_order, lamb_helmholtz, ε)
end

"""
Expects ζs_mag, ηs_mag, and Hs_π2 to be computed a priori.
"""
function multipole_to_local!(target_weights, target_branch, source_weights, source_branch, weights_tmp_1, weights_tmp_2, weights_tmp_3, Ts, eimϕs, ζs_mag, ηs_mag, Hs_π2, M̃, L̃, expansion_order, lamb_helmholtz::Val{LH}, error_method::Nothing) where LH
    # translation vector
    Δx = target_branch.center - source_branch.center
    r, θ, ϕ = cartesian_to_spherical(Δx)

    #--- rotate coordinate system ---#

    # rotate about z axis
    rotate_z!(weights_tmp_1, source_weights, eimϕs, ϕ, expansion_order, lamb_helmholtz)

    # rotate about y axis
    # NOTE: the method used here results in an additional rotatation of π about the new z axis
    rotate_multipole_y!(weights_tmp_2, weights_tmp_1, Ts, Hs_π2, ζs_mag, θ, expansion_order, lamb_helmholtz)

    # finish multipole-to-local translation
    multipole_to_local_II!(target_weights, weights_tmp_1, weights_tmp_2, r, expansion_order, Ts, ηs_mag, Hs_π2, eimϕs, lamb_helmholtz, true)

    return expansion_order, true
end

function multipole_to_local!(target_weights, target_branch, source_weights, source_branch, weights_tmp_1, weights_tmp_2, weights_tmp_3, Ts, eimϕs, ζs_mag, ηs_mag, Hs_π2, M̃, L̃, expansion_order, lamb_helmholtz::Val, error_method::RotatedCoefficientsAbsoluteGradient{ε,BE}) where {ε,BE}
    
    # determine expansion order to satisfy error tolerance
    r, error_success, expansion_order = multipole_to_local_RCAG!(weights_tmp_1, weights_tmp_2, source_weights, target_branch, source_branch, eimϕs, Ts, ζs_mag, Hs_π2, expansion_order, lamb_helmholtz, ε)

    # finish multipole-to-local translation
    multipole_to_local_II!(target_weights, weights_tmp_1, weights_tmp_2, r, expansion_order, Ts, ηs_mag, Hs_π2, eimϕs, lamb_helmholtz, BE)

    return expansion_order, error_success
end

function multipole_to_local!(target_weights, target_branch, source_weights, source_branch, weights_tmp_1, weights_tmp_2, weights_tmp_3, Ts, eimϕs, ζs_mag, ηs_mag, Hs_π2, M̃, L̃, expansion_order, lamb_helmholtz::Val, error_method::RotatedCoefficientsRelativeGradient{RET,AET,BE}) where {RET,AET,BE}

    # determine ε from the error method
    ε = max(target_branch.min_gradient * RET, AET)

    # determine expansion order to satisfy error tolerance
    r, error_success, expansion_order = multipole_to_local_RCAG!(weights_tmp_1, weights_tmp_2, source_weights, target_branch, source_branch, eimϕs, Ts, ζs_mag, Hs_π2, expansion_order, lamb_helmholtz, ε)

    # finish multipole-to-local translation
    multipole_to_local_II!(target_weights, weights_tmp_1, weights_tmp_2, r, expansion_order, Ts, ηs_mag, Hs_π2, eimϕs, lamb_helmholtz, BE)

    return expansion_order, error_success
end

function multipole_to_local!(target_weights, target_branch, source_weights, source_branch, weights_tmp_1, weights_tmp_2, weights_tmp_3, Ts, eimϕs, ζs_mag, ηs_mag, Hs_π2, M̃, L̃, expansion_order, lamb_helmholtz::Val, error_method::PowerAbsoluteGradient{ε,BE}) where {ε,BE}
    
    # determine expansion order to satisfy error tolerance
    r, error_success, expansion_order = multipole_to_local_PAG!(weights_tmp_1, weights_tmp_2, weights_tmp_3, source_weights, target_branch, source_branch, eimϕs, Ts, ζs_mag, Hs_π2, M̃, L̃, expansion_order, lamb_helmholtz, ε)

    # finish multipole-to-local translation
    multipole_to_local_II!(target_weights, weights_tmp_1, weights_tmp_2, r, expansion_order, Ts, ηs_mag, Hs_π2, eimϕs, lamb_helmholtz, BE)

    return expansion_order, error_success
end

function multipole_to_local!(target_weights, target_branch, source_weights, source_branch, weights_tmp_1, weights_tmp_2, weights_tmp_3, Ts, eimϕs, ζs_mag, ηs_mag, Hs_π2, M̃, L̃, expansion_order, lamb_helmholtz::Val, error_method::PowerRelativeGradient{RET,AET,BE}) where {RET,AET,BE}

    # determine ε from the error method
    ε = max(target_branch.min_gradient * RET, AET)

    # determine expansion order to satisfy error tolerance
    r, error_success, expansion_order = multipole_to_local_PAG!(weights_tmp_1, weights_tmp_2, weights_tmp_3, source_weights, target_branch, source_branch, eimϕs, Ts, ζs_mag, Hs_π2, M̃, L̃, expansion_order, lamb_helmholtz, ε)

    # finish multipole-to-local translation
    multipole_to_local_II!(target_weights, weights_tmp_1, weights_tmp_2, r, expansion_order, Ts, ηs_mag, Hs_π2, eimϕs, lamb_helmholtz, BE)

    return expansion_order, error_success
end

function multipole_to_local!(target_weights, target_branch, source_weights, source_branch, weights_tmp_1, weights_tmp_2, weights_tmp_3, Ts, eimϕs, ζs_mag, ηs_mag, Hs_π2, M̃, L̃, expansion_order, lamb_helmholtz::Val, error_method::PowerAbsolutePotential{ε,BE}) where {ε,BE}
    
    # determine expansion order to satisfy error tolerance
    r, error_success, expansion_order = multipole_to_local_PAP!(weights_tmp_1, weights_tmp_2, weights_tmp_3, source_weights, target_branch, source_branch, eimϕs, Ts, ζs_mag, Hs_π2, M̃, L̃, expansion_order, lamb_helmholtz, ε)

    # finish multipole-to-local translation
    multipole_to_local_II!(target_weights, weights_tmp_1, weights_tmp_2, r, expansion_order, Ts, ηs_mag, Hs_π2, eimϕs, lamb_helmholtz, BE)

    return expansion_order, error_success
end

function multipole_to_local!(target_weights, target_branch, source_weights, source_branch, weights_tmp_1, weights_tmp_2, weights_tmp_3, Ts, eimϕs, ζs_mag, ηs_mag, Hs_π2, M̃, L̃, expansion_order, lamb_helmholtz::Val, error_method::PowerRelativePotential{RET,AET,BE}) where {RET,AET,BE}

    # determine ε from the error method
    ε = max(target_branch.min_potential * RET, AET)

    # determine expansion order to satisfy error tolerance
    r, error_success, expansion_order = multipole_to_local_PAP!(weights_tmp_1, weights_tmp_2, weights_tmp_3, source_weights, target_branch, source_branch, eimϕs, Ts, ζs_mag, Hs_π2, M̃, L̃, expansion_order, lamb_helmholtz, ε)

    # finish multipole-to-local translation
    multipole_to_local_II!(target_weights, weights_tmp_1, weights_tmp_2, r, expansion_order, Ts, ηs_mag, Hs_π2, eimϕs, lamb_helmholtz, BE)

    return expansion_order, error_success
end

#--- rotated coefficients absolute gradient (RCAG) error method ---#

function multipole_to_local_RCAG!(weights_tmp_1, weights_tmp_2, source_weights, target_branch::Branch, source_branch::Branch, eimϕs, Ts, ζs_mag, Hs_π2, expansion_order, lamb_helmholtz::Val{LH}, ε) where LH

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
    for n in 1:expansion_order

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
        if ε_mp <= ε * 4 * pi

            # extract degree n, order 0-1 coefficients for error prediction
            # this function also performs the lamb-helmholtz transformation
            ϕn0_real, ϕn1_real, ϕn1_imag, χn1_real, χn1_imag = translate_multipole_to_local_z_m01_n(weights_tmp_2, r, rinv, lamb_helmholtz, n!_t_np1, n)

            # calculate local error
            ε_l = (abs(ϕn1_real) + abs(ϕn1_imag) + abs(ϕn0_real)) * r_l_nm1 * nm1!_inv
            if LH
                ε_l += sqrt(χn1_real * χn1_real + χn1_imag * χn1_imag) * r_l_nm1 * r_l * nm1!_inv
            end

            # check total error
            if ε_mp + ε_l * LOCAL_ERROR_SAFETY <= ε * 4 * pi
                expansion_order = n
                error_success = true
                break
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

    if !error_success && WARNING_FLAG_ERROR[]
        @warn "Error tolerance $ε not reached! Using max expansion order P=$(expansion_order).\n\tε_mp = $ε_mp, \n\tε_l = $ε_l"
        WARNING_FLAG_ERROR[] = false
    end

    return r, error_success, expansion_order
end

#--- multipole power absolute gradient (PAG) error method ---#

"""
Expects ζs_mag, ηs_mag, and Hs_π2 to be computed a priori.
"""
function multipole_to_local_PAG!(weights_tmp_1, weights_tmp_2, weights_tmp_3, source_weights, target_branch, source_branch, eimϕs, Ts, ζs_mag, Hs_π2, M̃, L̃, expansion_order, lamb_helmholtz::Val{LH}, ε) where LH

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

    # multipole power for n=0
    ϕn0_real, ϕn0_imag = source_weights[1,1,1], source_weights[2,1,1]
    M̃n0 = M̃[1]
    mp_power_ϕ = sqrt((ϕn0_real * ϕn0_real + ϕn0_imag * ϕn0_imag) * M̃n0 * M̃n0)

    if LH
        χn0_real, χn0_imag = source_weights[1,2,1], source_weights[2,2,1]
        mp_power_χ = sqrt((χn0_real * χn0_real + χn0_imag * χn0_imag) * M̃n0 * M̃n0)
    end

    #--- 0 < n <= n_crit and error predictions ---#

    # preallocate values to be available for recursion and debugging
    ε_mp, ε_l = zero(r), zero(r)
    error_success = false

    # n>0
    for n in 1:expansion_order

        # rotate about z axis
        eimϕ_real, eimϕ_imag, mp_power_ϕ_next, mp_power_χ_next, M̃n0_next = rotate_z_n_power!(weights_tmp_1, source_weights, eimϕs, eiϕ_real, eiϕ_imag, eimϕ_real, eimϕ_imag, M̃, lamb_helmholtz, n)

        # get y-axis Wigner rotation matrix
        _1_n = update_Ts_n!(Ts, Hs_π2, sβ, cβ, _1_n, n)

        # perform rotation about the y axis
        # NOTE: the method used here results in an additional rotatation of π about the new z axis
        i_ζ, i_T = _rotate_multipole_y_n!(weights_tmp_2, weights_tmp_1, Ts, ζs_mag, expansion_order, lamb_helmholtz, i_ζ, i_T, n)

        # calculate multipole error
        ε_mp = SQRT3 * mp_power_ϕ * r_mp_inv_np2 * np1! / M̃n0
        if LH
            ε_mp += SQRT3 * n * mp_power_χ_next * r_mp_inv_np2 * np1! / M̃n0_next
        end

        #--- check multipole error ---#

        if ε_mp <= ε * 4 * π

            #--- check local error ---#

            # translate order n multipole coefficients to local coefficients
            translate_multipole_to_local_z_n!(weights_tmp_3, weights_tmp_2, rinv, n!_t_np1, n, lamb_helmholtz)

            #--- check local error ---#

            # (note that all recursive quantities are updated for n here)
            l_power_ϕ, l_power_χ, L̃n0 = local_power(weights_tmp_3, r, n, L̃, lamb_helmholtz)
            ε_l = SQRT3 * l_power_ϕ * r_l_nm1 * nm1!_inv / L̃n0
            if LH
                ε_l += SQRT3 * n * l_power_χ * r_l_nm1 * r_l * nm1!_inv / (L̃n0 * n)
            end

            if ε_mp + ε_l * LOCAL_ERROR_SAFETY <= ε * 4 * π

                # tolerance satisfied so set expansion order
                expansion_order = n
                error_success = true
                break
            end
        end

        # recurse
        n!_t_np1 *= (n+1) * rinv
        r_l_nm1 *= r_l
        r_mp_inv_np2 *= r_mp_inv
        nm1!_inv /= n

        # multipole powers
        mp_power_ϕ = mp_power_ϕ_next
        if LH
            mp_power_χ = mp_power_χ_next
        end
        M̃n0 = M̃n0_next

        # increment (n+1)!
        np1! *= n+1
    end
    
    #--- warn if error tolerance is not reached ---#

    if !error_success && WARNING_FLAG_ERROR[]
        @warn "Error tolerance $ε not reached! Using max expansion order P=$(expansion_order).\n\tε_mp = $ε_mp, \n\tε_l = $ε_l"
        WARNING_FLAG_ERROR[] = false
    end

    return r, error_success, expansion_order
end

#--- multipole power absolute potential (PAP) error method ---#

function multipole_to_local_PAP!(weights_tmp_1, weights_tmp_2, weights_tmp_3, source_weights, target_branch, source_branch, eimϕs, Ts, ζs_mag, Hs_π2, M̃, L̃, expansion_order, lamb_helmholtz::Val{LH}, ε) where LH
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

    # multipole power for n=0
    ϕn0_real, ϕn0_imag = source_weights[1,1,1], source_weights[2,1,1]
    M̃n0 = M̃[1]
    mp_power_ϕ = sqrt((ϕn0_real * ϕn0_real + ϕn0_imag * ϕn0_imag) * M̃n0 * M̃n0)

    if LH
        χn0_real, χn0_imag = source_weights[1,2,1], source_weights[2,2,1]
        mp_power_χ = sqrt((χn0_real * χn0_real + χn0_imag * χn0_imag) * M̃n0 * M̃n0)
    end
    
    #--- n > 0 and error predictions ---#
    
    # preallocate values to be available for recursion and debugging
    ε_mp, ε_l = zero(r), zero(r)
    error_success = false
    
    # n > 0
    for n in 1:expansion_order

        # rotate about z axis
        eimϕ_real, eimϕ_imag, mp_power_ϕ_next, mp_power_χ_next, M̃n0_next = rotate_z_n_power!(weights_tmp_1, source_weights, eimϕs, eiϕ_real, eiϕ_imag, eimϕ_real, eimϕ_imag, M̃, lamb_helmholtz, n)

        # get y-axis Wigner rotation matrix
        _1_n = update_Ts_n!(Ts, Hs_π2, sβ, cβ, _1_n, n)

        # perform rotation about the y axis
        # NOTE: the method used here results in an additional rotatation of π about the new z axis
        i_ζ, i_T = _rotate_multipole_y_n!(weights_tmp_2, weights_tmp_1, Ts, ζs_mag, expansion_order, lamb_helmholtz, i_ζ, i_T, n)

        #--- check multipole error ---#

        # calculate multipole error
        ε_mp = SQRT3 * mp_power_ϕ_next * r_mp_inv_np2 * r_mp * np1! / (M̃n0 * (n+1))
        if LH
            ε_mp += SQRT3 * n * mp_power_χ_next * r_mp_inv_np2 * r_mp * np1! / (M̃n0_next * (n+1))
        end

        if ε_mp <= ε * 4 * π

            #--- check local error ---#

            # translate order n multipole coefficients to local coefficients
            translate_multipole_to_local_z_n!(weights_tmp_3, weights_tmp_2, rinv, n!_t_np1, n, lamb_helmholtz)

            #--- check local error ---#

            # (note that all recursive quantities are updated for n here)
            l_power_ϕ, l_power_χ, L̃n0 = local_power(weights_tmp_3, r, n, L̃, lamb_helmholtz)
            ε_l = SQRT3 * l_power_ϕ * r_l_nm1 * r_l * nm1!_inv / (L̃n0 * n)
            if LH
                ε_l += SQRT3 * n * l_power_χ * r_l_nm1 * r_l * nm1!_inv / (L̃n0 * n)
            end

            if ε_mp + ε_l * LOCAL_ERROR_SAFETY <= ε * 4 * π

                # tolerance satisfied so set expansion order
                expansion_order = n
                error_success = true
                break
            end
        end

        # recurse
        n!_t_np1 *= (n+1) * rinv
        r_l_nm1 *= r_l
        r_mp_inv_np2 *= r_mp_inv
        nm1!_inv /= n

        # multipole powers
        mp_power_ϕ = mp_power_ϕ_next
        if LH
            mp_power_χ = mp_power_χ_next
        end
        M̃n0 = M̃n0_next

        # increment (n+1)!
        np1! *= n+1
    end
    
    #--- warn if error tolerance is not reached ---#

    if !error_success && WARNING_FLAG_ERROR[]
        @warn "Error tolerance $ε not reached! Using max expansion order P=$(expansion_order).\n\tε_mp = $ε_mp, \n\tε_l = $ε_l"
        WARNING_FLAG_ERROR[] = false
    end

    return r, error_success, expansion_order
end

#--- finish multipole-to-local translation ---#

function multipole_to_local_II!(target_weights, weights_tmp_1, weights_tmp_2, r, expansion_order, Ts, ηs_mag, Hs_π2, eimϕs, lamb_helmholtz::Val{LH}, BE::Bool) where LH
    #--- translate along new z axis ---#

    translate_multipole_to_local_z!(weights_tmp_1, weights_tmp_2, r, expansion_order - 1 + BE, lamb_helmholtz)

    #--- transform Lamb-Helmholtz decomposition for the new center ---#

    LH && transform_lamb_helmholtz_local!(weights_tmp_1, r, expansion_order - 1 + BE)

    #--- back rotate coordinate system ---#

    # back rotate about y axis
    back_rotate_local_y!(weights_tmp_2, weights_tmp_1, Ts, Hs_π2, ηs_mag, expansion_order - 1 + BE, lamb_helmholtz)

    # back rotate about z axis and accumulate on target branch
    back_rotate_z!(target_weights, weights_tmp_2, eimϕs, expansion_order - 1 + BE, lamb_helmholtz)
end

"defaults to no error prediction"
multipole_to_local!(target_weights, target_branch, source_weights, source_branch, weights_tmp_1, weights_tmp_2, weights_tmp_3, Ts, eimϕs, ζs_mag, ηs_mag, Hs_π2, M̃, L̃, expansion_order, lamb_helmholtz) =
    multipole_to_local!(target_weights, target_branch, source_weights, source_branch, weights_tmp_1, weights_tmp_2, weights_tmp_3, Ts, eimϕs, ζs_mag, ηs_mag, Hs_π2, M̃, L̃, expansion_order, lamb_helmholtz, nothing)

# function dynamic_expansion_order!(weights_tmp_1, weights_tmp_2, Ts, eimϕs, ζs_mag, source_weights, Hs_π2, expansion_order, lamb_helmholtz::Val{LH}, r, θ, ϕ, ε, source_center, source_box, source_radius, target_center, target_box, target_radius; bonus_expansion::Bool=true) where LH

#     #--- distance information ---#

#     # multipole error location
#     Δx, Δy, Δz = minimum_distance(source_center, target_center, target_box)
#     r_mp = sqrt(Δx * Δx + Δy * Δy + Δz * Δz)

#     # local error location
#     r_l = target_radius

#     #--- initialize recursive values ---#

#     rinv = one(r) / r
#     r_mp_inv = one(r_mp) / r_mp
#     r_mp_inv_np2 = r_mp_inv * r_mp_inv * r_mp_inv
#     r_l_inv = one(r_l) / r_l
#     r_l_nm1 = one(r_l)

#     #--- n = 0 coefficients ---#

#     # rotate about z axis
#     eiϕ_real, eiϕ_imag, eimϕ_real, eimϕ_imag = rotate_z_0!(weights_tmp_1, source_weights, eimϕs, ϕ, expansion_order, lamb_helmholtz)

#     # get y-axis Wigner rotation matrix
#     sβ, cβ, _1_n = update_Ts_0!(Ts, Hs_π2, θ, expansion_order)

#     # perform rotation
#     i_ζ, i_T = _rotate_multipole_y_n!(weights_tmp_2, weights_tmp_1, Ts, ζs_mag, expansion_order, lamb_helmholtz, 0, 0, 0)

#     # (can't predict error yet, as we need degree n+1 coefficients)
#     # preallocate recursive values
#     n!_t_np1 = rinv * rinv
#     nm1!_inv = 1.0
#     np1! = 2.0

#     #--- n > 0 and error predictions ---#

#     # preallocate here for debugging only
#     ε_mp, ε_l = zero(r), zero(r)
#     # ϕn1_real, ϕn1_imag, ϕn0_real = zero(r), zero(r), zero(r)

#     # perform computation

#     #=
#     n = 0
#     # rotate about z axis
#     eimϕ_real, eimϕ_imag = rotate_z_n!(weights_tmp_1, source_weights, eimϕs, eiϕ_real, eiϕ_imag, eimϕ_real, eimϕ_imag, lamb_helmholtz, n)

#     # get y-axis Wigner rotation matrix
#     _1_n = update_Ts_n!(Ts, Hs_π2, sβ, cβ, _1_n, n)

#     # perform rotation about the y axis
#     # NOTE: the method used here results in an additional rotatation of π about the new z axis
#     i_ζ, i_T = _rotate_multipole_y_n!(weights_tmp_2, weights_tmp_1, Ts, ζs_mag, expansion_order, lamb_helmholtz, i_ζ, i_T, n)

#     # recurse
#     if n < expansion_order + 1 # if statement ensures that n == 1 + desired P
#                                # when tolerance is reached (breaks the loop early)
#                                # or that n == 1 + Pmax when tolerance is not reached
#         n!_t_np1 *= (n+1) * rinv
#         r_l_nm1 *= r_l
#         r_mp_inv_np2 *= r_mp_inv
#         nm1!_inv /= n

#         # increment n
#         n += 1
#         np1! *= n+1
#     end
#     =#

#     n = 1

#     # n>0
#     for _ in 1:expansion_order

#         # rotate about z axis
#         eimϕ_real, eimϕ_imag = rotate_z_n!(weights_tmp_1, source_weights, eimϕs, eiϕ_real, eiϕ_imag, eimϕ_real, eimϕ_imag, lamb_helmholtz, n)

#         # get y-axis Wigner rotation matrix
#         _1_n = update_Ts_n!(Ts, Hs_π2, sβ, cβ, _1_n, n)

#         # perform rotation about the y axis
#         # NOTE: the method used here results in an additional rotatation of π about the new z axis
#         i_ζ, i_T = _rotate_multipole_y_n!(weights_tmp_2, weights_tmp_1, Ts, ζs_mag, expansion_order, lamb_helmholtz, i_ζ, i_T, n)

#         # check multipole error
#         in0 = (n*(n+1))>>1 + 1
#         ϕn0 = weights_tmp_2[1,1,in0]
#         ϕn1_real = weights_tmp_2[1,1,in0+1]
#         ϕn1_imag = weights_tmp_2[2,1,in0+1]
#         if LH
#             χn1_real = weights_tmp_2[1,2,in0+1]
#             χn1_imag = weights_tmp_2[2,2,in0+1]
#         end

#         # calculate multipole error
#         ε_mp = (sqrt(ϕn1_real * ϕn1_real + ϕn1_imag * ϕn1_imag) + abs(ϕn0)) * np1! * r_mp_inv_np2
#         if LH
#             ε_mp += sqrt(χn1_real * χn1_real + χn1_imag * χn1_imag) * np1! * r_mp_inv_np2 * r_mp
#         end

#         # check local error if multipole error passes
#         if ε_mp <= ε

#             # extract degree n, order 0-1 coefficients for error prediction
#             # this function also performs the lamb-helmholtz transformation
#             ϕn0_real, ϕn1_real, ϕn1_imag, χn1_real, χn1_imag = translate_multipole_to_local_z_m01_n(weights_tmp_2, r, rinv, lamb_helmholtz, n!_t_np1, n)

#             # calculate local error
#             ε_l = (abs(ϕn1_real) + abs(ϕn1_imag) + abs(ϕn0_real)) * r_l_nm1 * nm1!_inv
#             if LH
#                 ε_l += sqrt(χn1_real * χn1_real + χn1_imag * χn1_imag) * r_l_nm1 * r_l * nm1!_inv
#             end

#             # check total error
#             if ε_mp + ε_l * LOCAL_ERROR_SAFETY <= ε
#                 return n-1+bonus_expansion, true
#             end
#         end

#         # recurse
#         if n < expansion_order # if statement ensures that n == desired P
#                                    # when tolerance is reached (breaks the loop early)
#                                    # or that n == Pmax when tolerance is not reached
#             n!_t_np1 *= (n+1) * rinv
#             r_l_nm1 *= r_l
#             r_mp_inv_np2 *= r_mp_inv
#             nm1!_inv /= n

#             # increment n
#             n += 1
#             np1! *= n+1
#         end
#     end

#     if WARNING_FLAG_ERROR[]
#         @warn "Error tolerance $(ε * ONE_OVER_4π) not reached! Using max expansion order P=$(n).\n\tε_mp = $ε_mp, \n\tε_l = $ε_l"
#         WARNING_FLAG_ERROR[] = false
#     end

#     return n, false
# end

# function dynamic_expansion_order!(weights_tmp_1, weights_tmp_2, Ts, eimϕs, ζs_mag, source_weights, Hs_π2, expansion_order, lamb_helmholtz::Val{LH}, r, θ, ϕ, ::AbsoluteGradient{ε}, source_center, source_box, source_radius, target_center, target_box, target_radius; bonus_expansion::Bool=true) where {LH, ε}

#     #--- distance information ---#

#     # multipole error location
#     Δx, Δy, Δz = minimum_distance(source_center, target_center, target_box)
#     r_mp = sqrt(Δx * Δx + Δy * Δy + Δz * Δz)

#     # local error location
#     r_l = target_radius

#     #--- initialize recursive values ---#

#     rinv = one(r) / r
#     r_mp_inv = one(r_mp) / r_mp
#     r_mp_inv_np2 = r_mp_inv * r_mp_inv * r_mp_inv
#     r_l_inv = one(r_l) / r_l
#     r_l_nm1 = one(r_l)

#     #--- n = 0 coefficients ---#

#     # rotate about z axis
#     eiϕ_real, eiϕ_imag, eimϕ_real, eimϕ_imag = rotate_z_0!(weights_tmp_1, source_weights, eimϕs, ϕ, expansion_order, lamb_helmholtz)

#     # get y-axis Wigner rotation matrix
#     sβ, cβ, _1_n = update_Ts_0!(Ts, Hs_π2, θ, expansion_order)

#     # perform rotation
#     i_ζ, i_T = _rotate_multipole_y_n!(weights_tmp_2, weights_tmp_1, Ts, ζs_mag, expansion_order, lamb_helmholtz, 0, 0, 0)

#     # (can't predict error yet, as we need degree n+1 coefficients)
#     # preallocate recursive values
#     n!_t_np1 = rinv * rinv
#     nm1!_inv = 1.0
#     np1! = 2.0

#     #--- n > 0 and error predictions ---#

#     n = 1
    
#     # preallocate here for debugging only
#     ε_mp, ε_l = zero(r), zero(r)

#     # n>0
#     for _ in 1:expansion_order

#         # rotate about z axis
#         eimϕ_real, eimϕ_imag = rotate_z_n!(weights_tmp_1, source_weights, eimϕs, eiϕ_real, eiϕ_imag, eimϕ_real, eimϕ_imag, lamb_helmholtz, n)

#         # get y-axis Wigner rotation matrix
#         _1_n = update_Ts_n!(Ts, Hs_π2, sβ, cβ, _1_n, n)

#         # perform rotation about the y axis
#         # NOTE: the method used here results in an additional rotatation of π about the new z axis
#         i_ζ, i_T = _rotate_multipole_y_n!(weights_tmp_2, weights_tmp_1, Ts, ζs_mag, expansion_order, lamb_helmholtz, i_ζ, i_T, n)

#         # check multipole error
#         in0 = (n*(n+1))>>1 + 1
#         ϕn0 = weights_tmp_2[1,1,in0]
#         ϕn1_real = weights_tmp_2[1,1,in0+1]
#         ϕn1_imag = weights_tmp_2[2,1,in0+1]
#         if LH
#             χn1_real = weights_tmp_2[1,2,in0+1]
#             χn1_imag = weights_tmp_2[2,2,in0+1]
#         end

#         # calculate multipole error
#         ε_mp = (sqrt(ϕn1_real * ϕn1_real + ϕn1_imag * ϕn1_imag) + abs(ϕn0)) * np1! * r_mp_inv_np2
#         if LH
#             ε_mp += sqrt(χn1_real * χn1_real + χn1_imag * χn1_imag) * np1! * r_mp_inv_np2 * r_mp
#         end

#         # check local error if multipole error passes
#         if ε_mp <= ε

#             # extract degree n, order 0-1 coefficients for error prediction
#             # this function also performs the lamb-helmholtz transformation
#             ϕn0_real, ϕn1_real, ϕn1_imag, χn1_real, χn1_imag = translate_multipole_to_local_z_m01_n(weights_tmp_2, r, rinv, lamb_helmholtz, n!_t_np1, n)

#             # calculate local error
#             ε_l = (abs(ϕn1_real) + abs(ϕn1_imag) + abs(ϕn0_real)) * r_l_nm1 * nm1!_inv
#             if LH
#                 ε_l += sqrt(χn1_real * χn1_real + χn1_imag * χn1_imag) * r_l_nm1 * r_l * nm1!_inv
#             end

#             # check total error
#             if ε_mp + ε_l * LOCAL_ERROR_SAFETY <= ε
#                 return n-1+bonus_expansion, true
#             end
#         end

#         # recurse
#         if n < expansion_order # if statement ensures that n == desired P
#                                    # when tolerance is reached (breaks the loop early)
#                                    # or that n == Pmax when tolerance is not reached
#             n!_t_np1 *= (n+1) * rinv
#             r_l_nm1 *= r_l
#             r_mp_inv_np2 *= r_mp_inv
#             nm1!_inv /= n

#             # increment n
#             n += 1
#             np1! *= n+1
#         end
#     end

#     if WARNING_FLAG_ERROR[]
#         @warn "Error tolerance $(ε * ONE_OVER_4π) not reached! Using max expansion order P=$(n).\n\tε_mp = $ε_mp, \n\tε_l = $ε_l"
#         WARNING_FLAG_ERROR[] = false
#     end

#     return n, false
# end

# function multipole_upper_bound(A, r, ρ, expansion_order, ET)
#     inv_r = 1 / r
#     rho_by_r = ρ * inv_r
#     base_factor = A * inv_r / ((r - ρ) * (r - ρ))

#     power_term = rho_by_r * rho_by_r # Start with the first power of (ρ / r)^(p+1)
#     for P in 1:expansion_order
#         linear_term = (P + 1) * ρ - (P + 2) * r
#         err_ub = base_factor * power_term * linear_term

#         if abs(err_ub) < ET
#             return P, true
#         end

#         # Update power_term for the next iteration (recursive multiplication)
#         power_term *= rho_by_r
#     end
#     return expansion_order, false
# end

# function local_upper_bound(A, r, ρ, expansion_order, ET)
#     inv_ρ = 1 / ρ
#     inv_diff = 1 / (ρ - r)
#     r_by_ρ = r * inv_ρ
#     base_factor = A * inv_diff
#     power_term = r_by_ρ  # Start with the first power of (r / ρ)

#     for P in 1:expansion_order
#         linear_term = (P + 1) * inv_ρ + inv_diff * r_by_ρ
#         err_ub = base_factor * power_term * linear_term

#         if err_ub < ET
#             return P, true
#         end

#         # Update power_term for the next iteration
#         power_term *= r_by_ρ
#     end
#     return expansion_order, false
# end

# function dynamic_expansion_order!(weights_tmp_1, weights_tmp_2, Ts, eimϕs, ζs_mag, source_weights, Hs_π2, expansion_order, lamb_helmholtz::Val{LH}, r, θ, ϕ, ::UpperBound{ε}, source_center, source_box, source_radius, target_center, target_box, target_radius; bonus_expansion::Bool=true) where {LH, ε}

#     #--- distance information ---#

#     # multipole error location
#     Δx, Δy, Δz = minimum_distance(source_center, target_center, target_box)
#     r_mp = sqrt(Δx * Δx + Δy * Δy + Δz * Δz)
#     ρ_max = source_radius

#     # local error location
#     r_l = target_radius
#     Δx, Δy, Δz = minimum_distance(target_center, source_center, source_box)
#     ρ_min = sqrt(Δx * Δx + Δy * Δy + Δz * Δz)

#     #--- differentiate Pringle's method to predict the norm velocity error ---#

#     A = abs(source_weights[1,1,1])
#     P_mp, error_success_mp = multipole_upper_bound(A, r_mp, ρ_max, expansion_order, ε)
#     P_l, error_success_l = local_upper_bound(A, r_l, ρ_min, expansion_order, ε)
#     P = max(P_mp, P_l)
#     error_success = error_success_mp && error_success_l

#     if !(error_success) && WARNING_FLAG_ERROR[]
#         @warn "Error tolerance $(ε * ONE_OVER_4π) not reached! Using max expansion order P=$(expansion_order)."
#         WARNING_FLAG_ERROR[] = false
#     end

#     #--- initialize recursive values ---#

#     rinv = one(r) / r
#     r_mp_inv = one(r_mp) / r_mp
#     r_mp_inv_np2 = r_mp_inv * r_mp_inv * r_mp_inv
#     r_l_inv = one(r_l) / r_l
#     r_l_nm1 = one(r_l)

#     #--- n = 0 coefficients ---#

#     # rotate about z axis
#     eiϕ_real, eiϕ_imag, eimϕ_real, eimϕ_imag = rotate_z_0!(weights_tmp_1, source_weights, eimϕs, ϕ, expansion_order, lamb_helmholtz)

#     # get y-axis Wigner rotation matrix
#     sβ, cβ, _1_n = update_Ts_0!(Ts, Hs_π2, θ, expansion_order)

#     # perform rotation
#     i_ζ, i_T = _rotate_multipole_y_n!(weights_tmp_2, weights_tmp_1, Ts, ζs_mag, expansion_order, lamb_helmholtz, 0, 0, 0)

#     # (can't predict error yet, as we need degree n+1 coefficients)
#     # preallocate recursive values
#     n!_t_np1 = rinv * rinv
#     nm1!_inv = 1.0
#     np1! = 2.0

#     #--- n > 0 and error predictions ---#

#     n = 1

#     # n>0
#     for _ in 1:P

#         # rotate about z axis
#         eimϕ_real, eimϕ_imag = rotate_z_n!(weights_tmp_1, source_weights, eimϕs, eiϕ_real, eiϕ_imag, eimϕ_real, eimϕ_imag, lamb_helmholtz, n)

#         # get y-axis Wigner rotation matrix
#         _1_n = update_Ts_n!(Ts, Hs_π2, sβ, cβ, _1_n, n)

#         # perform rotation about the y axis
#         # NOTE: the method used here results in an additional rotatation of π about the new z axis
#         i_ζ, i_T = _rotate_multipole_y_n!(weights_tmp_2, weights_tmp_1, Ts, ζs_mag, expansion_order, lamb_helmholtz, i_ζ, i_T, n)

#         # check multipole error
#         in0 = (n*(n+1))>>1 + 1
#         ϕn0 = weights_tmp_2[1,1,in0]
#         ϕn1_real = weights_tmp_2[1,1,in0+1]
#         ϕn1_imag = weights_tmp_2[2,1,in0+1]
#         if LH
#             χn1_real = weights_tmp_2[1,2,in0+1]
#             χn1_imag = weights_tmp_2[2,2,in0+1]
#         end

#         # recurse
#         if n < expansion_order # if statement ensures that n == desired P
#                                    # when tolerance is reached (breaks the loop early)
#                                    # or that n == Pmax when tolerance is not reached
#             n!_t_np1 *= (n+1) * rinv
#             r_l_nm1 *= r_l
#             r_mp_inv_np2 *= r_mp_inv
#             nm1!_inv /= n

#             # increment n
#             n += 1
#             np1! *= n+1
#         end
#     end

#     return P, error_success
# end

# """
# Expects ζs_mag, ηs_mag, and Hs_π2 to be computed a priori.
# """
# function multipole_to_local!(target_weights, target_branch, source_weights, source_branch, weights_tmp_1, weights_tmp_2, Ts, eimϕs, ζs_mag, ηs_mag, Hs_π2, expansion_order, lamb_helmholtz::Val{LH}, ε::AbsoluteVelocity) where LH

#     # translation vector
#     Δx = target_branch.center - source_branch.center
#     r, θ, ϕ = cartesian_to_spherical(Δx)

#     #------- rotate multipole coefficients (and determine P) -------#

#     expansion_order, error_success = dynamic_expansion_order!(weights_tmp_1, weights_tmp_2, Ts, eimϕs, ζs_mag, target_branch, source_branch, source_weights, Hs_π2, expansion_order, lamb_helmholtz, r, θ, ϕ, ε)

#     #--- translate along new z axis ---#

#     translate_multipole_to_local_z!(weights_tmp_1, weights_tmp_2, r, expansion_order, lamb_helmholtz)

#     #--- transform Lamb-Helmholtz decomposition for the new center ---#

#     LH && transform_lamb_helmholtz_local!(weights_tmp_1, r, expansion_order)

#     #--- back rotate coordinate system ---#

#     # back rotate about y axis
#     back_rotate_local_y!(weights_tmp_2, weights_tmp_1, Ts, Hs_π2, ηs_mag, expansion_order, lamb_helmholtz)

#     # back rotate about z axis and accumulate on target branch
#     back_rotate_z!(target_weights, weights_tmp_2, eimϕs, expansion_order, lamb_helmholtz)

#     return expansion_order, error_success
# end

#------- LOCAL TO LOCAL -------#

"""
Overwrites translated_weights
"""
function translate_local_z!(translated_weights, source_weights, t, expansion_order, ::Val{LH}) where LH
#function translate_local_z!(translated_weights, source_weights, t, ::Val{P}, ::Val{LH}) where {P,LH}
    _t = -t
    i = 1
    for n in 0:expansion_order
    	for m in 0:n
    		# inner summation
            val1_real = zero(eltype(translated_weights))
            val1_imag = zero(eltype(translated_weights))

            if LH
                val2_real = zero(eltype(translated_weights))
                val2_imag = zero(eltype(translated_weights))
            end
    		n_np = 1
    		_t_n_n! = one(t)
    		for np in n:expansion_order
                i_weight = harmonic_index(np,m)
                val1_real += _t_n_n! * source_weights[1,1,i_weight]
                val1_imag += _t_n_n! * source_weights[2,1,i_weight]
                if LH
                    val2_real += _t_n_n! * source_weights[1,2,i_weight]
                    val2_imag += _t_n_n! * source_weights[2,2,i_weight]
                end
    			_t_n_n! *= _t / n_np
    			n_np += 1
    		end

            # set translated coefficient
            translated_weights[1,1,i] = val1_real
            translated_weights[2,1,i] = val1_imag
            if LH
                translated_weights[1,2,i] = val2_real
                translated_weights[2,2,i] = val2_imag
            end

            # increment index
            i += 1
    	end
    end
end

"""
Expects ηs_mag and Hs_π2 to be precomputed. Ts and eimϕs are computed here.
"""
function local_to_local!(target_weights, target_branch, source_weights, source_branch, weights_tmp_1, weights_tmp_2, Ts, eimϕs, ηs_mag, Hs_π2, expansion_order, lamb_helmholtz::Val{LH}) where LH
    # translation vector
    Δx = target_branch.center - source_branch.center
    r, θ, ϕ = cartesian_to_spherical(Δx)

    #--- rotate coordinate system ---#

    # rotate about z axis (and compute eimϕs)
    rotate_z!(weights_tmp_1, source_weights, eimϕs, ϕ, expansion_order, lamb_helmholtz)

    # rotate about y axis (and compute Ts)
    # NOTE: the method used here results in an additional rotatation of π about the new z axis
    rotate_local_y!(weights_tmp_2, weights_tmp_1, Ts, Hs_π2, ηs_mag, θ, expansion_order, lamb_helmholtz)

    #--- translate along new z axis ---#

    translate_local_z!(weights_tmp_1, weights_tmp_2, r, expansion_order, lamb_helmholtz)

    #--- Lamb-Helmholtz transformation ---#

    LH && transform_lamb_helmholtz_local!(weights_tmp_1, r, expansion_order)

    #--- back rotate coordinate system ---#

    # back rotate about y axis
    back_rotate_local_y!(weights_tmp_2, weights_tmp_1, Ts, Hs_π2, ηs_mag, expansion_order, lamb_helmholtz)

    # back rotate about z axis and accumulate result to target branch
    back_rotate_z!(target_weights, weights_tmp_2, eimϕs, expansion_order, lamb_helmholtz)

end

