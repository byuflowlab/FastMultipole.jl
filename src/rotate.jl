#------- GENERAL FUNCTIONS -------#

#--- spherical coordinates ---#

@inline function cartesian_to_spherical(x; EPSILON=1e-10)
    return cartesian_to_spherical(x[1], x[2], x[3]; EPSILON)
end

@inline function cartesian_to_spherical(x::R, y::R, z::R; EPSILON=1e-10) where R
    x2y2 = x*x + y*y
    r2 = x2y2 + z*z
    r = iszero(r2) ? r2 : sqrt(r2)
    z_r = z/r
    if r > 0
        theta = x2y2 > 0 ? acos(z_r) : R(π) * (z < 0)
    else
        theta = zero(r)
    end
    phi = iszero(x2y2) ? zero(x2y2) : atan(y, x)
    return r, theta, phi
end

#--- z axis rotation matrices (diagonal) ---#

function update_eimϕs!(eimϕs, ϕ, P)
#function update_eimϕs!(eimϕs, ϕ, ::Val{expansion_order}) where expansion_order
    eiϕ_imag, eiϕ_real = sincos(ϕ)
    eimϕ_real, eimϕ_imag = one(ϕ), zero(ϕ)
    @inbounds for m in 0:P
        eimϕs[1,m+1] = eimϕ_real
        eimϕs[2,m+1] = eimϕ_imag
        eimϕ_real_tmp = eimϕ_real
        eimϕ_real = eimϕ_real * eiϕ_real - eimϕ_imag * eiϕ_imag
        eimϕ_imag = eimϕ_real_tmp * eiϕ_imag + eimϕ_imag * eiϕ_real
    end
end

"""
Returns eiϕ_real, eiϕ_imag, eimϕ_real, and eimϕ_imag appropriate for inputs for n=1 in update_eimϕs_n!
"""
function update_eimϕs_0!(eimϕs, ϕ)
    eiϕ_imag, eiϕ_real = sincos(ϕ)
    eimϕ_real, eimϕ_imag = one(ϕ), zero(ϕ)
    @inbounds eimϕs[1,1] = eimϕ_real
    @inbounds eimϕs[2,1] = eimϕ_imag
    return eiϕ_real, eiϕ_imag, eimϕ_real, eimϕ_imag
end

"""
Returns eimϕ_real and eimϕ_imag for n
"""
function update_eimϕs_n!(eimϕs, eiϕ_real, eiϕ_imag, eimϕ_real, eimϕ_imag, n)
    # calculate eimϕ
    eimϕ_real_tmp = eimϕ_real
    eimϕ_real = eimϕ_real * eiϕ_real - eimϕ_imag * eiϕ_imag
    eimϕ_imag = eimϕ_real_tmp * eiϕ_imag + eimϕ_imag * eiϕ_real

    # update container
    @inbounds eimϕs[1,n+1] = eimϕ_real
    @inbounds eimϕs[2,n+1] = eimϕ_imag

    return eimϕ_real, eimϕ_imag
end

#--- precompute y axis rotation matrices by π/2 ---#

function length_Hs(expansion_order)
    p = expansion_order
    n = p+1
    return div(n*(n+1)*(n+2), 6)
end

function length_H(n)
    return ((n+1)*(n+2))>>1
end

function get_H(Hs, n)
    if n==0
        return view(Hs,1:1)
    else
        i1 = length_Hs(n-1)+1
        i2 = length_Hs(n)
        return view(Hs,i1:i2)
    end
end

@inline function H_index(mp, m)
    return (m * (m+1))>>1 + mp + 1
end

@inline function Pnm_index(n, m)
    return (n * (n+1))>>1 + m + 1
end

function compute_Pnms(expansion_order, s, c)
    p = expansion_order
    Pnms = zeros(((p+1)*(p+2))>>1)
    Pn0, fact = 1.0, 1.0
    @inbounds for m in 0:p
        Pnm = Pn0
        i = Pnm_index(m, m)
        Pnms[i] = Pnm
        tmp1 = Pnm
        Pnm = c * (2 * m + 1) * tmp1
        for n in m+1:p
            i = Pnm_index(n, m)
            Pnms[i] = Pnm
            tmp2 = tmp1
            tmp1 = Pnm
            Pnm = (c * (2 * n + 1) * tmp1 - (n + m) * tmp2) / (n - m + 1)
        end
        Pn0 = -Pn0 * fact * s
        fact += 2
    end

    return Pnms
end

function compute_Hs!(Hs, β, n_start, n_end)
    @assert n_start != 0

    sβ, cβ = sincos(β)
    Pnms = compute_Pnms(n_end+1, sβ, cβ)
    H_np1 = zeros(n_end+2)

    # b
    bnms = zeros(2(n_end+1)+1)

    # d
    dnms = zeros(n_end+1)

    # Hs
    i_start = length_Hs(n_start-1) + 1
    i_end = i_start + length_H(n_start) - 1
    for n in n_start:n_end
        #H = get_H(Hs, n)
        H = view(Hs, i_start:i_end)

        # H_n_0m
        mp = 0
        _1_m, fact = 1, 1
        for m in 0:n
            # next value
            H[H_index(mp,m)] = _1_m * fact * Pnms[Pnm_index(n, m)]

            # recurse
            _1_m *= -1
            fact /= sqrt((n+m+1)*(n-m))
        end

        # H_np1_0m
        _1_m, fact = 1, 1
        for m in 0:n+1
            # next value
            H_np1[m+1] = _1_m * fact * Pnms[Pnm_index(n+1,m)]

            # recurse
            _1_m *= -1
            fact /= sqrt((n+1-m)*(n+m+2))
        end

        # b_np1_m
        b_n = n+1
        for m in -b_n:-1
            bnms[b_n + m + 1] = -sqrt((b_n-m-1)/(2b_n-1)*(b_n-m)/(2b_n+1))
        end
        for m in 0:b_n
            bnms[b_n + m + 1] = sqrt((b_n-m-1)/(2b_n-1)*(b_n-m)/(2b_n+1))
        end

        # H_n_1m
        one_m_cβ_2 = (1.0 - cβ)/2
        one_p_cβ_2 = (1.0 + cβ)/2
        for m in 1:n
            b_np1__mm1 = bnms[b_n-m]
            b_np1_mm1 = bnms[b_n+m]
            b_np1_0 = bnms[b_n+1]
            a_n_m = sqrt((n+1+m)/(2n+1)*(n+1-m)/(2n+3))
            H[H_index(1,m)] = (b_np1__mm1 * one_m_cβ_2 * H_np1[m+2] - b_np1_mm1 * one_p_cβ_2 * H_np1[m] - a_n_m * sβ * H_np1[m+1]) / b_np1_0
        end

        # update dnms
        for m in 0:n
            dnms[m+1] = sqrt((n-m)*(n+m+1)) / 2
        end

        # H_n_mp+1_m
        for mp in 1:n-1
            for m in mp+1:n-1
                val = dnms[mp] * H[H_index(mp-1,m)] - dnms[m] * H[H_index(mp,m-1)] + dnms[m+1] * H[H_index(mp,m+1)]
                H[H_index(mp+1,m)] = val / dnms[mp+1]
            end
            m = n
            val = dnms[mp] * H[H_index(mp-1,m)] - dnms[m] * H[H_index(mp,m-1)]
            H[H_index(mp+1,m)] = val / dnms[mp+1]
        end

        # recurse indices
        i_start = i_end + 1
        i_end = i_start + length_H(n+1) - 1

    end
end

function update_Hs_π2!(Hs_π2, expansion_order)
    l = length(Hs_π2)
    l_desired = length_Hs(expansion_order)

    if l_desired > l # we need to compute more matrices
        # determine how many we already have
        n_already, l_already = 0, 0
        for n in 0:expansion_order
            l_already += (n+1)^2
            l_already == l && (n_already = n)
        end

        # calculate the rest
        β = pi/2
        resize!(Hs_π2, l_desired)
        compute_Hs!(Hs_π2, β, n_already+1, expansion_order)

    end # otherwise, we already have enough, so do nothing
end

#--- use precomputed H matrices to compute rotations for arbitrary angle ---#

function length_Ts(expansion_order)
    p = expansion_order
    return div((p+1)*(p+2)*(2p+3),6)
end

function length_T(n)
    np1 = n+1
    return np1*np1
end

function get_T(Ts, n)
    if n == 0
        return view(Ts, 1:1)
    else
        i1 = length_Ts(n-1)+1
        i2 = length_Ts(n)
        return view(Ts,i1:i2)
    end
end

@inline function T_index(mp, m)
    return m*m + mp + m + 1
end

@inline function get_scalar(m_mp)
    isodd(m_mp) && (m_mp += 1)
    if iseven(m_mp>>1)
        return 1.0
    else
        return -1.0
    end
end

function update_Ts!(Ts, Hs_π2, β::TF, expansion_order) where TF
    p = expansion_order

    # initialize recursive indices
    i_H_start = length_Hs(0) + 1 # 2
    i_H_end = i_H_start + 2 # n=1 matrix contains 3 unique entries
    i_T_start = 2
    i_T_end = i_T_start + 3 # n=1 T matrix contains 4 unique entries

    # other recursive values
    sβ, cβ = sincos(β)
    _1_n = -1.0

    # rotate each expansion order
    Ts[1] = one(eltype(Ts))
    @inbounds for n in 1:p
        # Wigner matrix about y axis by π/2
        H_π2 = view(Hs_π2, i_H_start:i_H_end)

        # Wigner matrix by β
        T = view(Ts, i_T_start:i_T_end)

        for m in 0:n
            H_π2_n_m_0 = H_π2[H_index(0,m)]
            _1_mp_odd = 1.0
            m_mp = m
            _1_n_mp = _1_n

            for mp in 0:m
                H_π2_n_mp_0 = H_π2[H_index(0,mp)]
                val_positive_mp = zero(eltype(Ts))
                val_negative_mp = zero(eltype(Ts))
                s_νm1_β = 0.0
                c_νm1_β = 1.0
                m_mp_even = iseven(m_mp)
                scalar = get_scalar(m_mp)
                _1_n_mp_ν = -_1_n_mp

                for ν in 1:n
                    # sin and cos (νβ)
                    c_νβ = cβ * c_νm1_β - sβ * s_νm1_β
                    s_νβ = sβ * c_νm1_β + cβ * s_νm1_β

                    # alternating sine/cosine
                    sc_νβ_m_mp = m_mp_even ? scalar * c_νβ : scalar * s_νβ

                    # accumulate
                    i, j = minmax(mp,ν)
                    H_π2_n_mp_ν = H_π2[H_index(i,j)]
                    i, j = minmax(m,ν)
                    H_π2_n_m_ν = H_π2[H_index(i,j)]

                    # leverage symmetry to compute p/m mp
                    val = H_π2_n_mp_ν * H_π2_n_m_ν * sc_νβ_m_mp #
                    val_positive_mp += val
                    val_negative_mp += val * _1_n_mp_ν * _1_mp_odd #

                    # recurse
                    c_νm1_β = c_νβ
                    s_νm1_β = s_νβ
                    _1_n_mp_ν = -_1_n_mp_ν
                end

                val_positive_mp *= 2.0
                val_negative_mp *= 2.0
                scalar = m_mp_even ? scalar : 0.0
                val = H_π2_n_m_0 * H_π2_n_mp_0 * scalar
                val_positive_mp += val
                val_negative_mp += val * _1_n_mp * _1_mp_odd #
                T[T_index(mp,m)] = val_positive_mp
                T[T_index(-mp,m)] = val_negative_mp

                # recurse
                _1_n_mp = -_1_n_mp
                _1_mp_odd = -_1_mp_odd
                m_mp += 1
            end
        end

        # recurse
        i_H_start = i_H_end + 1
        i_H_end = i_H_start + length_H(n+1) - 1
        i_T_start = i_T_end + 1
        i_T_end = i_T_start + length_T(n+1) - 1
        _1_n = -_1_n
    end
end

"""
Returns sβ, cβ, and _1_n appropriate for calling update_Ts_n! for n=1
"""
function update_Ts_0!(Ts, Hs_π2, β::TF, expansion_order) where TF

    # rotate each expansion order
    Ts[1] = one(eltype(Ts))

    # other recursive values
    sβ, cβ = sincos(β)
    _1_n = -1.0

    return sβ, cβ, _1_n
end

function update_Ts_n!(Ts, Hs_π2, sβ, cβ, _1_n, n)

    # initialize recursive indices
    i_H_start = length_Hs(n-1) + 1
    i_H_end = i_H_start + length_H(n) - 1
    i_T_start = length_Ts(n-1)+1
    i_T_end = i_T_start + length_T(n) - 1

    # TODO: i_H_end and i_T_end both use the term (n+1)(n+2)/6;
    # could make this slightly more efficient (albeit less readable) by combining

    # Wigner matrix about y axis by π/2
    H_π2 = view(Hs_π2, i_H_start:i_H_end)

    # Wigner matrix by β
    T = view(Ts, i_T_start:i_T_end)

    @inbounds for m in 0:n
        H_π2_n_m_0 = H_π2[H_index(0,m)]
        _1_mp_odd = 1.0
        m_mp = m
        _1_n_mp = _1_n

        for mp in 0:m
            H_π2_n_mp_0 = H_π2[H_index(0,mp)]
            val_positive_mp = zero(eltype(Ts))
            val_negative_mp = zero(eltype(Ts))
            s_νm1_β = 0.0
            c_νm1_β = 1.0
            m_mp_even = iseven(m_mp)
            scalar = get_scalar(m_mp)
            _1_n_mp_ν = -_1_n_mp

            for ν in 1:n
                # sin and cos (νβ)
                c_νβ = cβ * c_νm1_β - sβ * s_νm1_β
                s_νβ = sβ * c_νm1_β + cβ * s_νm1_β

                # alternating sine/cosine
                sc_νβ_m_mp = m_mp_even ? scalar * c_νβ : scalar * s_νβ

                # accumulate
                i, j = minmax(mp,ν)
                H_π2_n_mp_ν = H_π2[H_index(i,j)]
                i, j = minmax(m,ν)
                H_π2_n_m_ν = H_π2[H_index(i,j)]

                # leverage symmetry to compute p/m mp
                val = H_π2_n_mp_ν * H_π2_n_m_ν * sc_νβ_m_mp #
                val_positive_mp += val
                val_negative_mp += val * _1_n_mp_ν * _1_mp_odd #

                # recurse
                c_νm1_β = c_νβ
                s_νm1_β = s_νβ
                _1_n_mp_ν = -_1_n_mp_ν
            end

            val_positive_mp *= 2.0
            val_negative_mp *= 2.0
            scalar = m_mp_even ? scalar : 0.0
            val = H_π2_n_m_0 * H_π2_n_mp_0 * scalar
            val_positive_mp += val
            val_negative_mp += val * _1_n_mp * _1_mp_odd #
            T[T_index(mp,m)] = val_positive_mp
            T[T_index(-mp,m)] = val_negative_mp

            # recurse
            _1_n_mp = -_1_n_mp
            _1_mp_odd = -_1_mp_odd
            m_mp += 1
        end
    end

    # recurse
    _1_n = -_1_n

    return _1_n
end

#--- functions to actually rotate ---#

"""
Performs a z-axis rotation of the supplied solid harmonic coefficients. Computes e^{imϕ} as well.
"""
function rotate_z!(rotated_weights, source_weights, eimϕs, ϕ, P, ::Val{LH}) where LH

    update_eimϕs!(eimϕs, ϕ, P)

    i_weight = 1

    # n = 0 (no change in the monopole term)
    @inbounds rotated_weights[1,1,i_weight] = source_weights[1,1,i_weight]
    @inbounds rotated_weights[2,1,i_weight] = source_weights[2,1,i_weight]
    if LH
        @inbounds rotated_weights[1,2,i_weight] = source_weights[1,2,i_weight]
        @inbounds rotated_weights[2,2,i_weight] = source_weights[2,2,i_weight]
    end
    i_weight += 1

    # n > 0
    @inbounds for n in 1:P
        for m in 0:n
            # get e^{imΔϕ}
            eimϕ_real, eimϕ_imag = eimϕs[1,m+1], eimϕs[2,m+1]

            # rotate coefficients
            Xnm_real, Xnm_imag = source_weights[1,1,i_weight], source_weights[2,1,i_weight]
            rotated_weights[1,1,i_weight] = Xnm_real * eimϕ_real - Xnm_imag * eimϕ_imag
            rotated_weights[2,1,i_weight] = Xnm_real * eimϕ_imag + Xnm_imag * eimϕ_real
            if LH
                Xnm_real, Xnm_imag = source_weights[1,2,i_weight], source_weights[2,2,i_weight]
                rotated_weights[1,2,i_weight] = Xnm_real * eimϕ_real - Xnm_imag * eimϕ_imag
                rotated_weights[2,2,i_weight] = Xnm_real * eimϕ_imag + Xnm_imag * eimϕ_real
            end

            # increment index
            i_weight += 1
        end
    end
end

function rotate_z_0!(rotated_weights, source_weights, eimϕs, ϕ, P, ::Val{LH}) where LH

    eiϕ_real, eiϕ_imag, eimϕ_real, eimϕ_imag = update_eimϕs_0!(eimϕs, ϕ)

    # n = 0 (no change in the monopole term)
    @inbounds rotated_weights[1,1,1] = source_weights[1,1,1]
    @inbounds rotated_weights[2,1,1] = source_weights[2,1,1]
    if LH
        @inbounds rotated_weights[1,2,1] = source_weights[1,2,1]
        @inbounds rotated_weights[2,2,1] = source_weights[2,2,1]
    end

    return eiϕ_real, eiϕ_imag, eimϕ_real, eimϕ_imag
end

"""
Performs a z-axis rotation of the supplied solid harmonic coefficients. Computes e^{imϕ} as well.
"""
function rotate_z_n!(rotated_weights, source_weights, eimϕs, eiϕ_real, eiϕ_imag, eimϕ_real, eimϕ_imag, ::Val{LH}, n) where LH

    eimϕ_real, eimϕ_imag = update_eimϕs_n!(eimϕs, eiϕ_real, eiϕ_imag, eimϕ_real, eimϕ_imag, n)

    i_weight = (n * (n+1)) >> 1 + 1

    @inbounds for m in 0:n
        # get e^{imΔϕ}
        eimϕ_real, eimϕ_imag = eimϕs[1,m+1], eimϕs[2,m+1]

        # rotate coefficients
        Xnm_real, Xnm_imag = source_weights[1,1,i_weight], source_weights[2,1,i_weight]
        rotated_weights[1,1,i_weight] = Xnm_real * eimϕ_real - Xnm_imag * eimϕ_imag
        rotated_weights[2,1,i_weight] = Xnm_real * eimϕ_imag + Xnm_imag * eimϕ_real
        if LH
            Xnm_real, Xnm_imag = source_weights[1,2,i_weight], source_weights[2,2,i_weight]
            rotated_weights[1,2,i_weight] = Xnm_real * eimϕ_real - Xnm_imag * eimϕ_imag
            rotated_weights[2,2,i_weight] = Xnm_real * eimϕ_imag + Xnm_imag * eimϕ_real
        end

        # increment index
        i_weight += 1
    end

    return eimϕ_real, eimϕ_imag
end

"""
Performs a z-axis rotation of the supplied solid harmonic coefficients. Computes e^{imϕ} and the multipole power at n as well.
"""
function rotate_z_n_power!(rotated_weights, source_weights, eimϕs, eiϕ_real, eiϕ_imag, eimϕ_real, eimϕ_imag, M̃L̃, ::Val{LH}, n) where LH

    eimϕ_real, eimϕ_imag = update_eimϕs_n!(eimϕs, eiϕ_real, eiϕ_imag, eimϕ_real, eimϕ_imag, n)

    i_weight = (n * (n+1)) >> 1 + 1

    # get M̃n0
    M̃L̃n0 = M̃L̃[i_weight]

    # initialize multipole power
    power_ϕ = zero(eltype(source_weights))
    power_χ = zero(eltype(source_weights))

    @inbounds for m in 0:n
        # get e^{imΔϕ}
        eimϕ_real, eimϕ_imag = eimϕs[1,m+1], eimϕs[2,m+1]

        # extract unrotated coefficients
        Xnm_real, Xnm_imag = source_weights[1,1,i_weight], source_weights[2,1,i_weight]
        
        # calculate the multipole power
        Ñ = M̃L̃[i_weight]
        power_ϕ += (Xnm_real * Xnm_real + Xnm_imag * Xnm_imag) * Ñ * Ñ * (1 + m>0)
        
        # rotate coefficients
        rotated_weights[1,1,i_weight] = Xnm_real * eimϕ_real - Xnm_imag * eimϕ_imag
        rotated_weights[2,1,i_weight] = Xnm_real * eimϕ_imag + Xnm_imag * eimϕ_real
        if LH
            Xnm_real, Xnm_imag = source_weights[1,2,i_weight], source_weights[2,2,i_weight]
            
            # calculate the multipole power
            power_χ += (Xnm_real * Xnm_real + Xnm_imag * Xnm_imag) * Ñ * Ñ * (1 + m>0)

            # rotate coefficients
            rotated_weights[1,2,i_weight] = Xnm_real * eimϕ_real - Xnm_imag * eimϕ_imag
            rotated_weights[2,2,i_weight] = Xnm_real * eimϕ_imag + Xnm_imag * eimϕ_real
        end

        # increment index
        i_weight += 1
    end
    # these two lines may be non-differentiable if the input to sqrt is zero.
    power_ϕ = sqrt(power_ϕ)
    power_χ = sqrt(power_χ)

    return eimϕ_real, eimϕ_imag, power_ϕ, power_χ, M̃L̃n0
end

"""
Assumes eimϕs have already been computed. DOES NOT overwrite rotated weights (unlike other rotate functions); rather, accumulates on top of it.
"""
function back_rotate_z!(rotated_weights, source_weights, eimϕs, P, ::Val{LH}) where LH
    i_weight = 1

    # n = 0 (no change in the monopole term)
    @inbounds rotated_weights[1,1,i_weight] += source_weights[1,1,i_weight]
    @inbounds rotated_weights[2,1,i_weight] += source_weights[2,1,i_weight]
    if LH
        @inbounds rotated_weights[1,2,i_weight] += source_weights[1,2,i_weight]
        @inbounds rotated_weights[2,2,i_weight] += source_weights[2,2,i_weight]
    end
    i_weight += 1

    # n > 0
    @inbounds for n in 1:P
        for m in 0:n
            # get e^{-imΔϕ}
            eimϕ_real, eimϕ_imag = eimϕs[1,m+1], -eimϕs[2,m+1]

            # rotate coefficients
            Xnm_real, Xnm_imag = source_weights[1,1,i_weight], source_weights[2,1,i_weight]
            rotated_weights[1,1,i_weight] += Xnm_real * eimϕ_real - Xnm_imag * eimϕ_imag
            rotated_weights[2,1,i_weight] += Xnm_real * eimϕ_imag + Xnm_imag * eimϕ_real
            if LH
                Xnm_real, Xnm_imag = source_weights[1,2,i_weight], source_weights[2,2,i_weight]
                rotated_weights[1,2,i_weight] += Xnm_real * eimϕ_real - Xnm_imag * eimϕ_imag
                rotated_weights[2,2,i_weight] += Xnm_real * eimϕ_imag + Xnm_imag * eimϕ_real
            end

            # increment index
            i_weight += 1
        end
    end
end

#------- MULTIPOLE ROTATIONS -------#

function length_ζs(expansion_order)
    p = expansion_order
    return div((p+1)*(p+2)*(2p+3),6)
end

function update_ζs_mag!(ζs_mag, expansion_order)
    l = length(ζs_mag)
    l_desired = length_ζs(expansion_order)

    if l_desired > l # we need to compute more matrices
        # determine how many we already have
        n_already, l_already = 0, 0
        for n in 0:expansion_order
            l_already += (n+1)^2
            l_already == l && (n_already = n)
        end

        # calculate the rest
        resize!(ζs_mag, l_desired)
        update_ζs_mag!(ζs_mag, n_already+1, expansion_order)

    end # otherwise, we already have enough, so do nothing
end

@inline function ζ_sign(magnitude, mp, m)
    mod = (abs(mp) - abs(m)) % 4
    mod < 0 && (mod += 4)
    if mod == 0
        return magnitude, 0.0
    elseif mod == 1
        return 0.0, magnitude
    elseif mod == 2
        return -magnitude, 0.0
    elseif mod == 3
        return 0.0, -magnitude
    end
end

function update_ζs_mag!(ζs, n_start, n_end)
    if n_start == 0
        ζs[1] = 1.0
        n_start = 1
    end
    for n in n_start:n_end
        ζ = get_ζ(ζs, n)
        i = 1
        mag_m = 1.0 # magnitude of ζ
        for m in 0:n
            mag_mp = mag_m

            for mp in 0:n
                # store magnitude (we'll compute the sign on the fly)
                ζ[i] = mag_mp

                # recurse over mp
                num = n + mp + 1
                denom = mp+1 < n ? n - mp : 1
                mag_mp *= sqrt(num/denom)

                # augment index
                i += 1
            end

            # recurse over m
            num = m < n - 1 ? n - m : 1
            denom = n+m+1
            mag_m *= sqrt(num/denom)
        end
    end
end

function get_ζ(ζs, n)
    i1 = length_ζs(n-1)+1
    i2 = length_ζs(n)
    return view(ζs, i1:i2)
end

#--- functions to actually rotate ---#

function _rotate_multipole_y!(rotated_weights, source_weights, Ts, ζs_mag, P, ::Val{LH}) where LH
    # reset container
    rotated_weights .= zero(eltype(rotated_weights))

    # rotate each order
    i_ζ = 0
    i_T = 0
    #println("=== START ===")
    @inbounds for n in 0:P
        #println("== n=$n ==")
        for m in 0:n
            #println("= m=$m =")
            val1_real = zero(eltype(Ts))
            val1_imag = zero(eltype(Ts))
            if LH
                val2_real = zero(eltype(Ts))
                val2_imag = zero(eltype(Ts))
            end
            i_ζ += n+1
            i_ζ_mp = i_ζ

            for mp in -n:0 # -m <= mp <= 0
                # T_n^{m',m} = T_n^{m,m'} = T_n^{-m,-m'}
                # @inline function T_index(mp, m)
                #     return m*m + mp + m + 1
                # end
                i1, i2 = minmax(-mp,m)
                T_mp_m = Ts[i_T + T_index(-i1,i2)]

                # ζ_n^{m',m} = β_n^{m'} / β_n^m
                ζ_real, ζ_imag = ζ_sign(ζs_mag[i_ζ_mp] * T_mp_m, mp, m)

                # current weights
                i_weight = harmonic_index(n, -mp)

                # sum contribution
                # (-1)^mp * complex conjugate for -mp
                M_n_mp_real, M_n_mp_imag = source_weights[1,1,i_weight], -source_weights[2,1,i_weight]
                if isodd(mp)
                    M_n_mp_real = -M_n_mp_real
                    M_n_mp_imag = -M_n_mp_imag
                end
                val1_real += M_n_mp_real * ζ_real - M_n_mp_imag * ζ_imag
                val1_imag += M_n_mp_real * ζ_imag + M_n_mp_imag * ζ_real

                if LH
                    # (-1)^mp * complex conjugate for -mp
                    M_n_mp_real, M_n_mp_imag = source_weights[1,2,i_weight], -source_weights[2,2,i_weight]
                    if isodd(mp)
                        M_n_mp_real = -M_n_mp_real
                        M_n_mp_imag = -M_n_mp_imag
                    end
                    val2_real += M_n_mp_real * ζ_real - M_n_mp_imag * ζ_imag
                    val2_imag += M_n_mp_real * ζ_imag + M_n_mp_imag * ζ_real
                end

                # decrement index
                i_ζ_mp -= 1
            end

            # augment index back to m'=1
            i_ζ_mp += 2

            #for mp in 1:m # 0 < mp <= m
            for mp in 1:n # mp > 0
                # T_mp_m = T[T_index(mp,m)]
                i1, i2 = minmax(mp,m)
                T_mp_m = Ts[i_T + T_index(i1,i2)]
                #@show T_index(mp,m)

                # ζ_n^{m',m} = β_n^{m'} / β_n^m
                ζ_real, ζ_imag = ζ_sign(ζs_mag[i_ζ_mp] * T_mp_m, mp, m)

                # current weights
                i_weight = harmonic_index(n, mp)

                # sum contribution
                M_n_mp_real, M_n_mp_imag = source_weights[1,1,i_weight], source_weights[2,1,i_weight]
                val1_real += M_n_mp_real * ζ_real - M_n_mp_imag * ζ_imag
                val1_imag += M_n_mp_real * ζ_imag + M_n_mp_imag * ζ_real
                if LH
                    M_n_mp_real, M_n_mp_imag = source_weights[1,2,i_weight], source_weights[2,2,i_weight]
                    val2_real += M_n_mp_real * ζ_real - M_n_mp_imag * ζ_imag
                    val2_imag += M_n_mp_real * ζ_imag + M_n_mp_imag * ζ_real
                end

                # increment index
                i_ζ_mp += 1
            end

            # update rotated_weights
            i = harmonic_index(n,m)
            rotated_weights[1,1,i] += val1_real
            rotated_weights[2,1,i] += val1_imag
            if LH
                rotated_weights[1,2,i] += val2_real
                rotated_weights[2,2,i] += val2_imag
            end
        end

        # next T matrix
        np1 = n+1
        i_T += np1*np1
    end
end

"""
The first time this function is called, it should receive i_ζ=0, i_T=0, and n=0. It will return i_ζ and i_T appropriate for n+1.
"""
function _rotate_multipole_y_n!(rotated_weights, source_weights, Ts, ζs_mag, P, ::Val{LH}, i_ζ, i_T, n) where LH
    # reset container
    in0 = (n * (n+1)) >> 1 + 1
    rotated_weights[:,:,in0:in0+n] .= zero(eltype(rotated_weights))

    for m in 0:n
        val1_real = zero(eltype(Ts))
        val1_imag = zero(eltype(Ts))
        if LH
            val2_real = zero(eltype(Ts))
            val2_imag = zero(eltype(Ts))
        end
        i_ζ += n+1
        i_ζ_mp = i_ζ

        for mp in -n:0 # -m <= mp <= 0
            # T_n^{m',m} = T_n^{m,m'} = T_n^{-m,-m'}
            # @inline function T_index(mp, m)
            #     return m*m + mp + m + 1
            # end
            i1, i2 = minmax(-mp,m)
            T_mp_m = Ts[i_T + T_index(-i1,i2)]

            # ζ_n^{m',m} = β_n^{m'} / β_n^m
            ζ_real, ζ_imag = ζ_sign(ζs_mag[i_ζ_mp] * T_mp_m, mp, m)

            # current weights
            i_weight = harmonic_index(n, -mp)

            # sum contribution
            # (-1)^mp * complex conjugate for -mp
            M_n_mp_real, M_n_mp_imag = source_weights[1,1,i_weight], -source_weights[2,1,i_weight]
            if isodd(mp)
                M_n_mp_real = -M_n_mp_real
                M_n_mp_imag = -M_n_mp_imag
            end
            val1_real += M_n_mp_real * ζ_real - M_n_mp_imag * ζ_imag
            val1_imag += M_n_mp_real * ζ_imag + M_n_mp_imag * ζ_real

            if LH
                # (-1)^mp * complex conjugate for -mp
                M_n_mp_real, M_n_mp_imag = source_weights[1,2,i_weight], -source_weights[2,2,i_weight]
                if isodd(mp)
                    M_n_mp_real = -M_n_mp_real
                    M_n_mp_imag = -M_n_mp_imag
                end
                val2_real += M_n_mp_real * ζ_real - M_n_mp_imag * ζ_imag
                val2_imag += M_n_mp_real * ζ_imag + M_n_mp_imag * ζ_real
            end

            # decrement index
            i_ζ_mp -= 1
        end

        # augment index back to m'=1
        i_ζ_mp += 2

        #for mp in 1:m # 0 < mp <= m
        for mp in 1:n # mp > 0
            # T_mp_m = T[T_index(mp,m)]
            i1, i2 = minmax(mp,m)
            T_mp_m = Ts[i_T + T_index(i1,i2)]
            if n==0 && DEBUG[]
                @show i_T + T_index(i1,i2)
            end
            #@show T_index(mp,m)

            # ζ_n^{m',m} = β_n^{m'} / β_n^m
            ζ_real, ζ_imag = ζ_sign(ζs_mag[i_ζ_mp] * T_mp_m, mp, m)

            # current weights
            i_weight = harmonic_index(n, mp)

            # sum contribution
            M_n_mp_real, M_n_mp_imag = source_weights[1,1,i_weight], source_weights[2,1,i_weight]
            val1_real += M_n_mp_real * ζ_real - M_n_mp_imag * ζ_imag
            val1_imag += M_n_mp_real * ζ_imag + M_n_mp_imag * ζ_real
            if LH
                M_n_mp_real, M_n_mp_imag = source_weights[1,2,i_weight], source_weights[2,2,i_weight]
                val2_real += M_n_mp_real * ζ_real - M_n_mp_imag * ζ_imag
                val2_imag += M_n_mp_real * ζ_imag + M_n_mp_imag * ζ_real
            end

            # increment index
            i_ζ_mp += 1
        end

        # update rotated_weights
        i = harmonic_index(n,m)
        rotated_weights[1,1,i] += val1_real
        rotated_weights[2,1,i] += val1_imag
        if LH
            rotated_weights[1,2,i] += val2_real
            rotated_weights[2,2,i] += val2_imag
        end
    end

    # next T matrix
    np1 = n+1
    i_T += np1*np1

    return i_ζ, i_T
end

"""
Rotate solid harmonic weights about the y axis by θ. Note that Hs_π2 and ζs_mag must be updated a priori, but Ts is updated en situ. Resets rotated_weights before computing.
"""
function rotate_multipole_y!(rotated_weights, source_weights, Ts, Hs_π2, ζs_mag, θ, P, lamb_helmholtz)
#function rotate_multipole_y!(rotated_weights, source_weights, Ts, Hs_π2, ζs_mag, θ, expansion_order, lamb_helmholtz)
    # get y-axis Wigner rotation matrix
    update_Ts!(Ts, Hs_π2, θ, P)

    # perform rotation
    _rotate_multipole_y!(rotated_weights, source_weights, Ts, ζs_mag, P, lamb_helmholtz)
end

"""
Assumes Ts, Hs_π2, and ζs_mag have all been precomputed. Resets target_weights.
"""
function back_rotate_multipole_y!(target_weights, rotated_weights, Ts, ζs_mag, P, lamb_helmholtz)
#function back_rotate_multipole_y!(target_weights, rotated_weights, Ts, ζs_mag, expansion_order, lamb_helmholtz)
    _rotate_multipole_y!(target_weights, rotated_weights, Ts, ζs_mag, P, lamb_helmholtz)
end

#------- LOCAL ROTATIONS -------#

function length_ηs(expansion_order)
    p = expansion_order
    return div((p+1)*(p+2)*(2p+3),6)
end

#function update_ηs_mag!(ηs_mag, P)
function update_ηs_mag!(ηs_mag, expansion_order)
    l = length(ηs_mag)
    l_desired = length_ηs(expansion_order)

    if l_desired > l # we need to compute more matrices
        # determine how many we already have
        n_already, l_already = 0, 0
        for n in 0:expansion_order
            l_already += (n+1)^2
            l_already == l && (n_already = n)
        end

        # calculate the rest
        resize!(ηs_mag, l_desired)
        update_ηs_mag!(ηs_mag, n_already+1, expansion_order)

    end # otherwise, we already have enough, so do nothing
end

@inline function η_sign(magnitude, mp, m)
    mod = (abs(m) - abs(mp)) % 4
    mod < 0 && (mod += 4)
    if mod == 0
        return magnitude, 0.0
    elseif mod == 1
        return 0.0, magnitude
    elseif mod == 2
        return -magnitude, 0.0
    elseif mod == 3
        return 0.0, -magnitude
    end
end

function update_ηs_mag!(ηs, n_start, n_end)
    if n_start == 0
        ηs[1] = 1.0
        n_start = 1
    end
    for n in n_start:n_end
        η = get_η(ηs, n)
        i = 1
        mag_m = 1.0 # magnitude of η
        for m in 0:n
            mag_mp = mag_m

            for mp in 0:n
                # store magnitude (we'll compute the sign on the fly)
                η[i] = mag_mp

                # recurse over mp
                num = mp+1 < n ? n - mp : 1
                denom = n + mp + 1
                mag_mp *= sqrt(num/denom)

                # augment index
                i += 1
            end

            # recurse over m
            num = n+m+1
            denom = m < n - 1 ? n - m : 1
            mag_m *= sqrt(num/denom)
        end
    end
end

function get_η(ηs, n)
    i1 = length_ηs(n-1)+1
    i2 = length_ηs(n)
    return view(ηs, i1:i2)
end

#--- functions to actually rotate ---#

function _rotate_local_y!(rotated_weights, source_weights, Ts, Hs_π2, ηs_mag, P, ::Val{LH}) where LH
    # reset container
    rotated_weights .= zero(eltype(rotated_weights))

    i_η = 0
    i_T = 0
    # rotate each order
    @inbounds for n in 0:P

        for m in 0:n
            val1_real = zero(eltype(Ts))
            val1_imag = zero(eltype(Ts))
            if LH
                val2_real = zero(eltype(Ts))
                val2_imag = zero(eltype(Ts))
            end

            i_η += n+1
            i_η_mp = i_η

            for mp in -n:0 # -n <= mp <= 0
                # T_n^{m',m} = T_n^{m,m'} = T_n^{-m,-m'}
                i1, i2 = minmax(-mp,m)
                T_mp_m = Ts[i_T + T_index(-i1,i2)]

                # η_n^{m',m} = β_n^{m'} / β_n^m
                η_real, η_imag = η_sign(ηs_mag[i_η_mp] * T_mp_m, mp, m)

                # current weights
                i_weight = harmonic_index(n, -mp)

                # sum contribution
                # (-1)^mp * complex conjugate for -mp
                M_n_mp_real, M_n_mp_imag = source_weights[1,1,i_weight], -source_weights[2,1,i_weight]
                if isodd(mp)
                    M_n_mp_real = -M_n_mp_real
                    M_n_mp_imag = -M_n_mp_imag
                end
                val1_real += M_n_mp_real * η_real - M_n_mp_imag * η_imag
                val1_imag += M_n_mp_real * η_imag + M_n_mp_imag * η_real

                if LH
                    # (-1)^mp * complex conjugate for -mp
                    M_n_mp_real, M_n_mp_imag = source_weights[1,2,i_weight], -source_weights[2,2,i_weight]
                    if isodd(mp)
                        M_n_mp_real = -M_n_mp_real
                        M_n_mp_imag = -M_n_mp_imag
                    end
                    val2_real += M_n_mp_real * η_real - M_n_mp_imag * η_imag
                    val2_imag += M_n_mp_real * η_imag + M_n_mp_imag * η_real
                end

                # decrement index
                i_η_mp -= 1
            end

            # augment index back to m'=1
            i_η_mp += 2

            for mp in 1:n # m < mp <= n
                # T_n^{m',m} = T_n^{m,m'}
                i1, i2 = minmax(mp,m)
                T_mp_m = Ts[i_T + T_index(i1,i2)]

                # η_n^{m',m} = β_n^{m'} / β_n^m
                η_real, η_imag = η_sign(ηs_mag[i_η_mp] * T_mp_m, mp, m)

                # current weights
                i_weight = harmonic_index(n, mp)

                # sum contribution
                M_n_mp_real, M_n_mp_imag = source_weights[1,1,i_weight], source_weights[2,1,i_weight]
                val1_real += M_n_mp_real * η_real - M_n_mp_imag * η_imag
                val1_imag += M_n_mp_real * η_imag + M_n_mp_imag * η_real

                if LH
                    M_n_mp_real, M_n_mp_imag = source_weights[1,2,i_weight], source_weights[2,2,i_weight]
                    val2_real += M_n_mp_real * η_real - M_n_mp_imag * η_imag
                    val2_imag += M_n_mp_real * η_imag + M_n_mp_imag * η_real
                end

                # increment index
                i_η_mp += 1
            end

            # update rotated_weights
            i = harmonic_index(n,m)
            rotated_weights[1,1,i] += val1_real
            rotated_weights[2,1,i] += val1_imag
            if LH
                rotated_weights[1,2,i] += val2_real
                rotated_weights[2,2,i] += val2_imag
            end
        end

        # next T matrix
        np1 = n+1
        i_T += np1*np1
    end
end

function rotate_local_y!(rotated_weights, source_weights, Ts, Hs_π2, ηs_mag, θ, expansion_order, lamb_helmholtz)
    # get y-axis Wigner rotation matrix
    update_Ts!(Ts, Hs_π2, θ, expansion_order)

    # perform rotation
    _rotate_local_y!(rotated_weights, source_weights, Ts, Hs_π2, ηs_mag, expansion_order, lamb_helmholtz)
end

"""
Assumes Ts, Hs_π2, and ηs_mag have all been precomputed. Resets target_weights.
"""
function back_rotate_local_y!(target_weights, rotated_weights, Ts, Hs_π2, ηs_mag, expansion_order, lamb_helmholtz)
    _rotate_local_y!(target_weights, rotated_weights, Ts, Hs_π2, ηs_mag, expansion_order, lamb_helmholtz)
end

