@inline odd_or_even(n::Int) = (n & 1) == 1 ? -1 : 1

@inline ipow2n(n::Int) = n >= 0 ? 1 : odd_or_even(n);

@inline function harmonic_index(n, m)
    return (n * (n+1))>>1 + m + 1
end

#=
Gumerov's normalization:
(-1)^n * im^abs(m) * r^n * Plm(cos(θ),n,abs(m)) * exp(im*m*ϕ) / factorial(n+abs(m))
=#
# NOTE: every literal below must carry TF. Float64 literals here silently
# promote an otherwise-Float32 evaluation to Float64, which CUDA tolerates but
# Apple GPUs reject outright ("unsupported use of double value") -- the
# adaptive S2L/M2T device kernels call these. For TF=Float64 the typed forms
# are bit-identical to the previous literals, so the host path is unchanged.
function regular_harmonics!(harmonics, ρ::TF, θ::TF, ϕ::TF, expansion_order) where TF
    y, x = sincos(θ)
    fact = one(TF)
    pn = one(TF) # Legendre polynomial of degree n, order 0
    ρm = one(TF) # rho^n / (n+|m|)! * (-1)^n
    i_ei_imag, i_ei_real = sincos(ϕ + oftype(ϕ, π/2)) # i e^(iϕ) = e^[i(ϕ+π/2)]
    i_eim_real, i_eim_imag = one(TF), zero(TF) # i^m e^(i * m * phi) = e^[i m (ϕ+π/2)]

    # evaluate
    for m=0:expansion_order # n=m
        p = pn
        i = harmonic_index(m, m)
        ρm_p = ρm * p

        # set harmonics
        harmonics[1,1,i] = ρm_p * i_eim_real
        harmonics[2,1,i] = ρm_p * i_eim_imag

        p1 = p
        p = x * (2 * m + 1) * p1
        ρm *= ρ
        ρn = ρm
        for n=m+1:expansion_order # n>m in here
            i = harmonic_index(n, m)
            ρn /= -(n + m)
            ρn_p = ρn * p
            harmonics[1,1,i] = ρn_p * i_eim_real
            harmonics[2,1,i] = ρn_p * i_eim_imag

            # recurse
            p2 = p1
            p1 = p
            p = (x * (2 * n + 1) * p1 - (n + m) * p2) / (n - m + 1)
            ρn *= ρ
        end

        # recurse
        ρm /= -(2 * m + 2) * (2 * m + 1)
        pn = -pn * fact * y
        fact += 2
        i_eim_real_tmp = i_eim_real
        i_eim_imag_tmp = i_eim_imag
        i_eim_real = i_eim_real_tmp * i_ei_real - i_eim_imag_tmp * i_ei_imag
        i_eim_imag = i_eim_real_tmp * i_ei_imag + i_eim_imag_tmp * i_ei_real
    end
end

#=
Gumerov's normalization:
(-1)^m * im^abs(m) * r^(-n-1) * Plm(cos(θ),n,abs(m)) * exp(im*m*ϕ) * factorial(n-abs(m))
=#
function irregular_harmonics!(harmonics, ρ, θ, ϕ::TF, expansion_order) where TF
    y, x = sincos(θ)
    fact = one(TF) # 2m+1 (odd integers)
    pn = one(TF) # Legendre polynomial of degree n, order n
    one_over_ρ = one(TF) / ρ # NOTE: this should never be singular, as we only evaluate irregular harmonics far away
    ρm = one_over_ρ # (-1)^m / ρ^(n+1)
    i_ei_imag, i_ei_real = sincos(ϕ + oftype(ϕ, π/2)) # i e^(iϕ) = e^[i(ϕ+π/2)]
    i_eim_real, i_eim_imag = one(TF), zero(TF) # i^m e^(i * m * phi) = e^[i m (ϕ+π/2)]

    for m=0:expansion_order # n=m
        p = pn
        i = harmonic_index(m, m)
        ρm_p = ρm * p

        harmonics[1,1,i] = ρm_p * i_eim_real
        harmonics[2,1,i] = ρm_p * i_eim_imag

        p1 = p
        p = x * (2 * m + 1) * p1
        ρm *= -one_over_ρ
        ρn = -ρm
        for n=m+1:expansion_order # n>m
            i = harmonic_index(n, m)
            ρn_p = ρn * p
            harmonics[1,1,i] = ρn_p * i_eim_real
            harmonics[2,1,i] = ρn_p * i_eim_imag
            p2 = p1
            p1 = p
            p = (x * (2 * n + 1) * p1 - (n + m) * p2) / (n - m + 1)
            ρn *= one_over_ρ * (n - m + 1)
        end

        # recurse
        pn *= -fact * y
        fact += 2
        i_eim_real_tmp = i_eim_real
        i_eim_imag_tmp = i_eim_imag
        i_eim_real = i_eim_real_tmp * i_ei_real - i_eim_imag_tmp * i_ei_imag
        i_eim_imag = i_eim_real_tmp * i_ei_imag + i_eim_imag_tmp * i_ei_real
    end
end

