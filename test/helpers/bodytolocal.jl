function body_to_local_point!(::Type{Point{Source}}, local_coefficients, harmonics, Δx, strength, expansion_order)

    # invert the sign of the strength so v=∇ϕ instead of v=-∇ϕ
    # this will allow induced velocities to be added to those induced by vortex elements
    # we'll invert the potential again at the end
    strength = -strength

    # irregular harmonics
    ρ, θ, ϕ = FastMultipole.cartesian_to_spherical(Δx)
    FastMultipole.irregular_harmonics!(harmonics, ρ, θ, ϕ, expansion_order)

    # update coefficients
    _1_n = 1.0
    i = 1
    for n in 0:expansion_order
        _1_n_m = _1_n
        for m in 0:n
            local_coefficients[1,1,i] += harmonics[1,1,i] * _1_n_m * strength
            local_coefficients[2,1,i] -= harmonics[2,1,i] * _1_n_m * strength # Rnm*
            _1_n_m = -_1_n_m
            i += 1
        end
        _1_n = -_1_n
    end
end

function Snm(ρ,θ,ϕ,n,m)
    if abs(m) > n
        return 0.0 + 0im
    end
    return (1.0*im)^(-abs(m)) * Float64(factorial(big(n-abs(m)))) / ρ^(n+1) * Plm(cos(θ),n,abs(m)) * exp(im*m*ϕ)
end

function body_to_local_point!(::Type{Point{Vortex}}, local_coefficients, harmonics::AbstractArray{TF}, Δx, strength, expansion_order) where TF

    # extract strength
    ωx, ωy, ωz = strength

    # irregular harmonics
    ρ, θ, ϕ = FastMultipole.cartesian_to_spherical(Δx)
    FastMultipole.irregular_harmonics!(harmonics, ρ, θ, ϕ, expansion_order+1)

    # update ϕnm
    i = 2 # ϕ00 = 0
    _1_n = -1.0
    for n in 1:expansion_order
        _1_m = 1.0
        for m in 0:n
            # intermediate variables
            S_n_mp1_real = m < n ? -_1_m * harmonics[1,1,i+1] : zero(TF)
            S_n_mp1_imag = m < n ? _1_m * harmonics[2,1,i+1] : zero(TF)
            S_n_m_real = _1_m * harmonics[1,1,i]
            S_n_m_imag = -_1_m * harmonics[2,1,i]
            S_n_mm1_real = m == 0 ? -_1_m * S_n_mp1_real : -_1_m * harmonics[1,1,i-1]
            S_n_mm1_imag = m == 0 ? _1_m * S_n_mp1_imag : _1_m * harmonics[2,1,i-1]
            n_inv = 1/n

            # ϕnm
            local_coefficients[1,1,i] -= _1_n * n_inv * ((n-m)*0.5 * (ωx * S_n_mp1_real - ωy * S_n_mp1_imag) - (n+m)*0.5 * (ωx * S_n_mm1_real + ωy * S_n_mm1_imag) + ωz * m * S_n_m_imag)
            local_coefficients[2,1,i] -= _1_n * n_inv * ((n-m)*0.5 * (ωx * S_n_mp1_imag + ωy * S_n_mp1_real) - (n+m)*0.5 * (ωx * S_n_mm1_imag - ωy * S_n_mm1_real) - ωz * m * S_n_m_real)

            # recurse
            i += 1
            _1_m = -_1_m
        end
        _1_n = -_1_n
    end

    # update χnm
    i = 1
    _1_np1 = -1.0
    for n in 0:expansion_order
        _1_m = 1.0
        for m in 0:n
            # intermediate variables
            i_np1_m = i + n + 1
            S_np1_mp1_real = -_1_m * harmonics[1,1,i_np1_m+1]
            S_np1_mp1_imag = _1_m * harmonics[2,1,i_np1_m+1]
            S_np1_m_real = _1_m * harmonics[1,1,i_np1_m]
            S_np1_m_imag = -_1_m * harmonics[2,1,i_np1_m]
            S_np1_mm1_real = m == 0 ? -_1_m * S_np1_mp1_real : -_1_m * harmonics[1,1,i_np1_m-1]
            S_np1_mm1_imag = m == 0 ? _1_m * S_np1_mp1_imag : _1_m * harmonics[2,1,i_np1_m-1]
            np1_inv = 1/(n+1)

            # χnm
            local_coefficients[1,2,i] += _1_np1 * np1_inv * (0.5 * (ωy * S_np1_mm1_real - ωx * S_np1_mm1_imag) - 0.5 * (ωy * S_np1_mp1_real + ωx * S_np1_mp1_imag) - ωz * S_np1_m_real)
            local_coefficients[2,2,i] += _1_np1 * np1_inv * (0.5 * (ωy * S_np1_mm1_imag + ωx * S_np1_mm1_real) - 0.5 * (ωy * S_np1_mp1_imag - ωx * S_np1_mp1_real) - ωz * S_np1_m_imag)

            # recurse
            i += 1
            _1_m = -_1_m
        end
        _1_np1 = -_1_np1
    end

end
