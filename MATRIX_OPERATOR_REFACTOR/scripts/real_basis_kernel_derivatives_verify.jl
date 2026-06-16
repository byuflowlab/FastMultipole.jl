using FastMultipole
using LinearAlgebra
using Printf
using StaticArrays

const OPERATOR_ATOL = 1.0e-12
const POINT_RTOL = 5.0e-7
const ONE_OVER_4PI = 1.0 / (4pi)
const DATA_DIR = joinpath(@__DIR__, "..", "data", "real_basis_kernel_derivatives")
const SUMMARY_PATH = joinpath(DATA_DIR, "verification_summary.md")

mode_index(n::Int, ::Val{:zero}) = n^2 + 1
mode_index(n::Int, m::Int, ::Val{:cos}) = n^2 + 2m
mode_index(n::Int, m::Int, ::Val{:sin}) = n^2 + 2m + 1
nreal(P::Int) = (P + 1)^2

active_components(::Val{LH}) where {LH} = LH ? 2 : 1

# ------------------------------------------------------------------ #
# Transforms and operator chain (ported from the approved task 008
# verification script; used here only to build the point-mass example).
# ------------------------------------------------------------------ #

function deterministic_complex(P::Int, layout; seed = 0.0, zero_m0_imag = true)
    weights = FastMultipole.initialize_expansion(P, Float64)
    components = active_components(layout)
    for component in 1:components
        for n in 0:P
            for m in 0:n
                i = FastMultipole.harmonic_index(n, m)
                base = seed + 0.071 * i + 0.113 * component
                weights[1, component, i] = sin(base) + 0.29 * cos(1.7 * base)
                weights[2, component, i] = cos(0.6 * base) - 0.19 * sin(2.1 * base)
                zero_m0_imag && m == 0 && (weights[2, component, i] = 0.0)
            end
        end
    end
    return weights
end

function complex_to_real(weights, P::Int, layout)
    components = active_components(layout)
    real_weights = zeros(Float64, nreal(P), components)
    for component in 1:components
        for n in 0:P
            i = FastMultipole.harmonic_index(n, 0)
            real_weights[mode_index(n, Val(:zero)), component] = weights[1, component, i]
            for m in 1:n
                i = FastMultipole.harmonic_index(n, m)
                real_weights[mode_index(n, m, Val(:cos)), component] = weights[1, component, i]
                real_weights[mode_index(n, m, Val(:sin)), component] = weights[2, component, i]
            end
        end
    end
    return real_weights
end

function real_to_complex(real_weights, P::Int, layout)
    weights = FastMultipole.initialize_expansion(P, Float64)
    components = active_components(layout)
    for component in 1:components
        for n in 0:P
            i = FastMultipole.harmonic_index(n, 0)
            weights[1, component, i] = real_weights[mode_index(n, Val(:zero)), component]
            weights[2, component, i] = 0.0
            for m in 1:n
                i = FastMultipole.harmonic_index(n, m)
                weights[1, component, i] = real_weights[mode_index(n, m, Val(:cos)), component]
                weights[2, component, i] = real_weights[mode_index(n, m, Val(:sin)), component]
            end
        end
    end
    return weights
end

function phase_vectors(P::Int, phi::Float64)
    ncoeff = FastMultipole.harmonic_index(P, P)
    C = zeros(Float64, ncoeff)
    S = zeros(Float64, ncoeff)
    for n in 0:P, m in 0:n
        i = FastMultipole.harmonic_index(n, m)
        S[i], C[i] = sincos(m * phi)
    end
    return C, S
end

function fused_rotate_z!(out, source, C, S, P::Int, layout)
    components = active_components(layout)
    for n in 0:P, m in 0:n
        i = FastMultipole.harmonic_index(n, m)
        c = C[i]; s = S[i]
        for component in 1:components
            real_in = source[1, component, i]
            imag_in = source[2, component, i]
            out[1, component, i] = c * real_in - s * imag_in
            out[2, component, i] = s * real_in + c * imag_in
        end
    end
    return out
end

function fused_back_rotate_z!(out, source, C, S, P::Int, layout)
    components = active_components(layout)
    for n in 0:P, m in 0:n
        i = FastMultipole.harmonic_index(n, m)
        c = C[i]; s = S[i]
        for component in 1:components
            real_in = source[1, component, i]
            imag_in = source[2, component, i]
            out[1, component, i] += c * real_in + s * imag_in
            out[2, component, i] += -s * real_in + c * imag_in
        end
    end
    return out
end

function axis_swap_update_Ts!(Ts, Hs_pi2, beta::Float64, P::Int)
    Ts[1] = 1.0
    sβ, cβ = sincos(beta)
    one_n = -1.0
    i_H_start = FastMultipole.length_Hs(0) + 1
    i_H_end = i_H_start + FastMultipole.length_H(1) - 1
    i_T_start = 2
    i_T_end = i_T_start + FastMultipole.length_T(1) - 1
    for n in 1:P
        H = view(Hs_pi2, i_H_start:i_H_end)
        T = view(Ts, i_T_start:i_T_end)
        for m in 0:n
            H_m_0 = H[FastMultipole.H_index(0, m)]
            mp_parity = 1.0
            m_plus_mp = m
            one_n_mp = one_n
            for mp in 0:m
                H_mp_0 = H[FastMultipole.H_index(0, mp)]
                positive_mp = 0.0
                negative_mp = 0.0
                s_prev = 0.0
                c_prev = 1.0
                even_m_mp = iseven(m_plus_mp)
                scalar = FastMultipole.get_scalar(m_plus_mp)
                one_n_mp_nu = -one_n_mp
                for nu in 1:n
                    c_nu = cβ * c_prev - sβ * s_prev
                    s_nu = sβ * c_prev + cβ * s_prev
                    z_phase = even_m_mp ? scalar * c_nu : scalar * s_nu
                    i, j = minmax(mp, nu)
                    H_mp_nu = H[FastMultipole.H_index(i, j)]
                    i, j = minmax(m, nu)
                    H_m_nu = H[FastMultipole.H_index(i, j)]
                    val = H_mp_nu * H_m_nu * z_phase
                    positive_mp += val
                    negative_mp += val * one_n_mp_nu * mp_parity
                    c_prev = c_nu
                    s_prev = s_nu
                    one_n_mp_nu = -one_n_mp_nu
                end
                positive_mp *= 2.0
                negative_mp *= 2.0
                zero_mode = even_m_mp ? scalar : 0.0
                val = H_m_0 * H_mp_0 * zero_mode
                positive_mp += val
                negative_mp += val * one_n_mp * mp_parity
                T[FastMultipole.T_index(mp, m)] = positive_mp
                T[FastMultipole.T_index(-mp, m)] = negative_mp
                one_n_mp = -one_n_mp
                mp_parity = -mp_parity
                m_plus_mp += 1
            end
        end
        i_H_start = i_H_end + 1
        i_H_end = i_H_start + FastMultipole.length_H(n + 1) - 1
        i_T_start = i_T_end + 1
        i_T_end = i_T_start + FastMultipole.length_T(n + 1) - 1
        one_n = -one_n
    end
    return Ts
end

function matrix_translate_m2m_z!(out, source, t, P::Int, layout)
    components = active_components(layout)
    for m in 0:P, n in m:P
        i_out = FastMultipole.harmonic_index(n, m)
        for component in 1:components
            real_acc = 0.0; imag_acc = 0.0
            coeff = one(t); factorial = one(t)
            for np in n:-1:m
                i_in = FastMultipole.harmonic_index(np, m)
                tmp = coeff / factorial
                real_acc += tmp * source[1, component, i_in]
                imag_acc += tmp * source[2, component, i_in]
                coeff *= -t
                factorial *= n - np + 1
            end
            out[1, component, i_out] = real_acc
            out[2, component, i_out] = imag_acc
        end
    end
    return out
end

function matrix_translate_l2l_z!(out, source, t, P::Int, layout)
    components = active_components(layout)
    for m in 0:P, n in m:P
        i_out = FastMultipole.harmonic_index(n, m)
        for component in 1:components
            real_acc = 0.0; imag_acc = 0.0
            coeff = one(t); factorial = one(t)
            for np in n:P
                i_in = FastMultipole.harmonic_index(np, m)
                tmp = coeff / factorial
                real_acc += tmp * source[1, component, i_in]
                imag_acc += tmp * source[2, component, i_in]
                coeff *= -t
                factorial *= np - n + 1
            end
            out[1, component, i_out] = real_acc
            out[2, component, i_out] = imag_acc
        end
    end
    return out
end

function matrix_translate_m2l_z!(out, source, t, P::Int, layout)
    components = active_components(layout)
    rho = inv(t)
    for m in 0:P
        row_seed = one(t)
        for k in 1:(2m)
            row_seed *= k * rho
        end
        row_seed *= rho
        for n in m:P
            i_out = FastMultipole.harmonic_index(n, m)
            for component in 1:components
                real_acc = 0.0; imag_acc = 0.0
                coeff = row_seed
                for np in m:P
                    i_in = FastMultipole.harmonic_index(np, m)
                    real_acc += coeff * source[1, component, i_in]
                    imag_acc += coeff * source[2, component, i_in]
                    coeff *= (n + np + 1) * rho
                end
                out[1, component, i_out] = real_acc
                out[2, component, i_out] = imag_acc
            end
            n < P && (row_seed *= (n + m + 1) * rho)
        end
    end
    return out
end

function branch(center)
    return FastMultipole.Branch(1:1, 0, 1:0, 0, 1, center, 0.0, SVector{3}(0.0, 0.0, 0.0))
end

function prealloc(P::Int)
    Hs_pi2 = [1.0]
    FastMultipole.update_Hs_π2!(Hs_pi2, P)
    zeta = zeros(Float64, FastMultipole.length_ζs(P))
    eta = zeros(Float64, FastMultipole.length_ηs(P))
    FastMultipole.update_ζs_mag!(zeta, 0, P)
    FastMultipole.update_ηs_mag!(eta, 0, P)
    return (;
        Hs_pi2, zeta, eta,
        Ts = zeros(Float64, FastMultipole.length_Ts(P)),
        tmp1 = FastMultipole.initialize_expansion(P, Float64),
        tmp2 = FastMultipole.initialize_expansion(P, Float64),
    )
end

function explicit_m2m!(target, target_branch, source, source_branch, P::Int, layout, cache)
    dx = target_branch.center - source_branch.center
    r, theta, phi = FastMultipole.cartesian_to_spherical(dx)
    C, S = phase_vectors(P, phi)
    fused_rotate_z!(cache.tmp1, source, C, S, P, layout)
    axis_swap_update_Ts!(cache.Ts, cache.Hs_pi2, theta, P)
    FastMultipole._rotate_multipole_y!(cache.tmp2, cache.tmp1, cache.Ts, cache.zeta, P, layout)
    matrix_translate_m2m_z!(cache.tmp1, cache.tmp2, r, P, layout)
    FastMultipole.back_rotate_multipole_y!(cache.tmp2, cache.tmp1, cache.Ts, cache.zeta, P, layout)
    fused_back_rotate_z!(target, cache.tmp2, C, S, P, layout)
    return target
end

function explicit_l2l!(target, target_branch, source, source_branch, P::Int, layout, cache)
    dx = target_branch.center - source_branch.center
    r, theta, phi = FastMultipole.cartesian_to_spherical(dx)
    C, S = phase_vectors(P, phi)
    fused_rotate_z!(cache.tmp1, source, C, S, P, layout)
    axis_swap_update_Ts!(cache.Ts, cache.Hs_pi2, theta, P)
    FastMultipole.rotate_local_y!(cache.tmp2, cache.tmp1, cache.Ts, cache.Hs_pi2, cache.eta, theta, P, layout)
    matrix_translate_l2l_z!(cache.tmp1, cache.tmp2, r, P, layout)
    FastMultipole.back_rotate_local_y!(cache.tmp2, cache.tmp1, cache.Ts, cache.Hs_pi2, cache.eta, P, layout)
    fused_back_rotate_z!(target, cache.tmp2, C, S, P, layout)
    return target
end

function explicit_m2l!(target, target_branch, source, source_branch, P::Int, layout, cache)
    dx = target_branch.center - source_branch.center
    r, theta, phi = FastMultipole.cartesian_to_spherical(dx)
    C, S = phase_vectors(P, phi)
    fused_rotate_z!(cache.tmp1, source, C, S, P, layout)
    axis_swap_update_Ts!(cache.Ts, cache.Hs_pi2, theta, P)
    FastMultipole._rotate_multipole_y!(cache.tmp2, cache.tmp1, cache.Ts, cache.zeta, P, layout)
    matrix_translate_m2l_z!(cache.tmp1, cache.tmp2, r, P, layout)
    FastMultipole.back_rotate_local_y!(cache.tmp2, cache.tmp1, cache.Ts, cache.Hs_pi2, cache.eta, P, layout)
    fused_back_rotate_z!(target, cache.tmp2, C, S, P, layout)
    return target
end

function apply_real_operator(kind::Symbol, target_real, target_branch, source_real, source_branch, P::Int, layout)
    target_complex = real_to_complex(target_real, P, layout)
    source_complex = real_to_complex(source_real, P, layout)
    if kind === :m2m
        explicit_m2m!(target_complex, target_branch, source_complex, source_branch, P, layout, prealloc(P))
    elseif kind === :m2l
        explicit_m2l!(target_complex, target_branch, source_complex, source_branch, P, layout, prealloc(P))
    elseif kind === :l2l
        explicit_l2l!(target_complex, target_branch, source_complex, source_branch, P, layout, prealloc(P))
    else
        error("unknown operator kind $kind")
    end
    return complex_to_real(target_complex, P, layout)
end

function source_point_multipole!(weights, point, center, strength::Float64, P::Int)
    dx = point - center
    rho, theta, phi = FastMultipole.cartesian_to_spherical(dx)
    harmonics = FastMultipole.initialize_harmonics(P)
    FastMultipole.regular_harmonics!(harmonics, rho, theta, phi, P)
    for n in 0:P, m in 0:n
        i = FastMultipole.harmonic_index(n, m)
        Rnm = harmonics[1, 1, i] + im * harmonics[2, 1, i]
        Mnm = -(-1)^(n + m) * strength * conj(Rnm)
        weights[1, 1, i] = real(Mnm)
        weights[2, 1, i] = imag(Mnm)
    end
    return weights
end

# ------------------------------------------------------------------ #
# Native real-basis kernel-derivative evaluation (the subject of 008e).
# ------------------------------------------------------------------ #

# read compressed-complex (a, b) lanes of degree N, order k from a single
# real-basis channel vector. Caller guarantees k <= N <= P.
@inline function read_ab(R, N::Int, k::Int)
    if k == 0
        return (R[mode_index(N, Val(:zero))], 0.0)
    else
        return (R[mode_index(N, k, Val(:cos))], R[mode_index(N, k, Val(:sin))])
    end
end

# Spatial-gradient coefficient operator G (chi-free). Field degree n is sourced
# from input degree n+1. Returns the three Cartesian field-coefficient channels
# as real-basis vectors.
function gradient_coeffs_scalar(R, P::Int)
    gx = zeros(Float64, nreal(P))
    gy = zeros(Float64, nreal(P))
    gz = zeros(Float64, nreal(P))
    for n in 0:(P - 1)
        N = n + 1
        a1, b1 = read_ab(R, N, 1)
        a0, _ = read_ab(R, N, 0)
        gx[mode_index(n, Val(:zero))] = -b1
        gy[mode_index(n, Val(:zero))] = -a1
        gz[mode_index(n, Val(:zero))] = -a0
        for m in 1:n
            am1, bm1 = read_ab(R, N, m - 1)
            ap1, bp1 = read_ab(R, N, m + 1)
            am, bm = read_ab(R, N, m)
            gx[mode_index(n, m, Val(:cos))] = -(bm1 + bp1) * 0.5
            gx[mode_index(n, m, Val(:sin))] =  (am1 + ap1) * 0.5
            gy[mode_index(n, m, Val(:cos))] =  (am1 - ap1) * 0.5
            gy[mode_index(n, m, Val(:sin))] =  (bm1 - bp1) * 0.5
            gz[mode_index(n, m, Val(:cos))] = -am
            gz[mode_index(n, m, Val(:sin))] = -bm
        end
    end
    return gx, gy, gz
end

# Lamb-Helmholtz gradient operator G_LH(phi, chi). phi part is the chi-free G;
# chi part (degree n, no shift) adds the documented curl coupling.
function gradient_coeffs_lh(Rphi, Rchi, P::Int)
    gx = zeros(Float64, nreal(P))
    gy = zeros(Float64, nreal(P))
    gz = zeros(Float64, nreal(P))
    for n in 0:P
        # m = 0
        vx0 = 0.0; vy0 = 0.0; vz0 = 0.0
        if n + 1 <= P
            a1, b1 = read_ab(Rphi, n + 1, 1)
            a0, _ = read_ab(Rphi, n + 1, 0)
            vx0 += -b1
            vy0 += -a1
            vz0 += -a0
        end
        if n >= 1
            ac1, bc1 = read_ab(Rchi, n, 1)
            vx0 += n * ac1
            vy0 += -n * bc1
        end
        gx[mode_index(n, Val(:zero))] = vx0
        gy[mode_index(n, Val(:zero))] = vy0
        gz[mode_index(n, Val(:zero))] = vz0
        for m in 1:n
            gxc = 0.0; gxs = 0.0; gyc = 0.0; gys = 0.0; gzc = 0.0; gzs = 0.0
            if n + 1 <= P
                am1, bm1 = read_ab(Rphi, n + 1, m - 1)
                ap1, bp1 = read_ab(Rphi, n + 1, m + 1)
                am, bm = read_ab(Rphi, n + 1, m)
                gxc += -(bm1 + bp1) * 0.5
                gxs +=  (am1 + ap1) * 0.5
                gyc +=  (am1 - ap1) * 0.5
                gys +=  (bm1 - bp1) * 0.5
                gzc += -am
                gzs += -bm
            end
            # chi contribution at degree n
            acm1, bcm1 = read_ab(Rchi, n, m - 1)
            acp1 = 0.0; bcp1 = 0.0
            if m < n
                acp1, bcp1 = read_ab(Rchi, n, m + 1)
            end
            # g^x += (1/2)((n-m) chi^{m+1} - (n+m) chi^{m-1})
            gxc += 0.5 * ((n - m) * acp1 - (n + m) * acm1)
            gxs += 0.5 * ((n - m) * bcp1 - (n + m) * bcm1)
            # g^y += (i/2)((n-m) chi^{m+1} + (n+m) chi^{m-1})
            Aa = (n - m) * acp1 + (n + m) * acm1
            Bb = (n - m) * bcp1 + (n + m) * bcm1
            gyc += -Bb * 0.5
            gys +=  Aa * 0.5
            # g^z += -i m chi^m
            acm, bcm = read_ab(Rchi, n, m)
            gzc += m * bcm
            gzs += -m * acm
            gx[mode_index(n, m, Val(:cos))] = gxc
            gx[mode_index(n, m, Val(:sin))] = gxs
            gy[mode_index(n, m, Val(:cos))] = gyc
            gy[mode_index(n, m, Val(:sin))] = gys
            gz[mode_index(n, m, Val(:cos))] = gzc
            gz[mode_index(n, m, Val(:sin))] = gzs
        end
    end
    return gx, gy, gz
end

# Scalar contraction Eval(R) = u_raw against precomputed regular harmonics.
function evaluate_field(R, harmonics, P::Int)
    u = 0.0
    for n in 0:P
        i_h = FastMultipole.harmonic_index(n, 0)
        u += harmonics[1, 1, i_h] * R[mode_index(n, Val(:zero))]
        for m in 1:n
            i_h = FastMultipole.harmonic_index(n, m)
            p = harmonics[1, 1, i_h]
            q = harmonics[2, 1, i_h]
            u += 2.0 * (p * R[mode_index(n, m, Val(:cos))] - q * R[mode_index(n, m, Val(:sin))])
        end
    end
    return u
end

function regular_harmonics_at(point, P::Int)
    rho, theta, phi = FastMultipole.cartesian_to_spherical(point)
    harmonics = FastMultipole.initialize_harmonics(P)
    FastMultipole.regular_harmonics!(harmonics, rho, theta, phi, P)
    return harmonics
end

function native_real_potential(Rphi, point, P::Int)
    harmonics = regular_harmonics_at(point, P)
    return evaluate_field(Rphi, harmonics, P) * ONE_OVER_4PI
end

function native_real_gradient(Rphi, Rchi, point, P::Int, layout)
    harmonics = regular_harmonics_at(point, P)
    gx, gy, gz = layout isa Val{true} ? gradient_coeffs_lh(Rphi, Rchi, P) : gradient_coeffs_scalar(Rphi, P)
    vx = evaluate_field(gx, harmonics, P)
    vy = evaluate_field(gy, harmonics, P)
    vz = evaluate_field(gz, harmonics, P)
    return SVector{3}(vx, vy, vz) * ONE_OVER_4PI
end

function native_real_hessian(Rphi, Rchi, point, P::Int, layout)
    harmonics = regular_harmonics_at(point, P)
    gx, gy, gz = layout isa Val{true} ? gradient_coeffs_lh(Rphi, Rchi, P) : gradient_coeffs_scalar(Rphi, P)
    channels = (gx, gy, gz)  # field channel beta in {x, y, z}
    # H[beta, alpha] = G_alpha(g^beta): row indexes the field channel, column the
    # derivative direction, matching production's SMatrix assembly order. Under
    # Lamb-Helmholtz the Jacobian is asymmetric (curl), so this order is load-bearing.
    H = zeros(Float64, 3, 3)
    for (beta, gbeta) in enumerate(channels)
        hx, hy, hz = gradient_coeffs_scalar(gbeta, P)  # chi-free second application
        H[beta, 1] = evaluate_field(hx, harmonics, P)
        H[beta, 2] = evaluate_field(hy, harmonics, P)
        H[beta, 3] = evaluate_field(hz, harmonics, P)
    end
    return SMatrix{3,3,Float64,9}(H) * ONE_OVER_4PI
end

# ------------------------------------------------------------------ #
# Production reference evaluation.
# ------------------------------------------------------------------ #

function production_eval(weights, point, P::Int, layout)
    harmonics = FastMultipole.initialize_harmonics(P)
    gradient_n_m = FastMultipole.initialize_gradient_n_m(P)
    u, grad, hess = FastMultipole.evaluate_local(
        point, harmonics, gradient_n_m, weights, P, layout,
        FastMultipole.DerivativesSwitch(true, true, true))
    return u, grad, hess
end

# ------------------------------------------------------------------ #
# Test cases.
# ------------------------------------------------------------------ #

function run_parity_case(P::Int, point::SVector{3,Float64}, layout)
    seed = layout isa Val{true} ? 3.1 : 2.4
    complex_weights = deterministic_complex(P, layout; seed = seed, zero_m0_imag = true)
    real_weights = complex_to_real(complex_weights, P, layout)
    Rphi = @view real_weights[:, 1]
    Rchi = layout isa Val{true} ? (@view real_weights[:, 2]) : zeros(Float64, nreal(P))

    u_prod, grad_prod, hess_prod = production_eval(complex_weights, point, P, layout)

    # scalar parity only for Val(false) (production phi potential is degenerate under LH)
    scalar_err = NaN
    if layout isa Val{false}
        u_native = native_real_potential(Rphi, point, P)
        scalar_err = abs(u_native - u_prod)
    end

    grad_native = native_real_gradient(Rphi, Rchi, point, P, layout)
    grad_err = maximum(abs.(grad_native .- grad_prod))

    hess_native = native_real_hessian(Rphi, Rchi, point, P, layout)
    hess_err = maximum(abs.(hess_native .- hess_prod))
    sym_err = maximum(abs.(hess_native .- transpose(hess_native)))
    trace_abs = abs(hess_native[1, 1] + hess_native[2, 2] + hess_native[3, 3])

    return (; P, point, layout = layout isa Val{true} ? "Val(true)" : "Val(false)",
        scalar_err, grad_err, hess_err, sym_err, trace_abs)
end

function point_mass_local_expansion(P::Int)
    # Approved unit point-mass M2M -> M2L -> L2L chain (task 008 example geometry).
    source_leaf_center = SVector{3}(0.04, -0.03, 0.02)
    source_point = source_leaf_center + SVector{3}(0.018, -0.012, 0.009)
    source_parent_center = SVector{3}(0.0, 0.0, 0.0)
    target_parent_center = SVector{3}(1.45, 1.1, -0.72)
    target_child_center = target_parent_center + SVector{3}(-0.05, 0.04, 0.03)
    target_point = target_child_center + SVector{3}(0.018, -0.015, 0.011)
    strength = 1.0
    layout = Val(false)

    source_leaf_complex = FastMultipole.initialize_expansion(P, Float64)
    source_point_multipole!(source_leaf_complex, source_point, source_leaf_center, strength, P)
    source_leaf_real = complex_to_real(source_leaf_complex, P, layout)

    source_parent_real = apply_real_operator(:m2m, zeros(Float64, nreal(P), 1),
        branch(source_parent_center), source_leaf_real, branch(source_leaf_center), P, layout)
    target_parent_real = apply_real_operator(:m2l, zeros(Float64, nreal(P), 1),
        branch(target_parent_center), source_parent_real, branch(source_parent_center), P, layout)
    target_child_real = apply_real_operator(:l2l, zeros(Float64, nreal(P), 1),
        branch(target_child_center), target_parent_real, branch(target_parent_center), P, layout)

    local_point = target_point - target_child_center
    return target_child_real, local_point, target_point, source_point, strength
end

function run_point_mass_case(P::Int)
    target_child_real, local_point, target_point, source_point, strength = point_mass_local_expansion(P)
    layout = Val(false)
    Rphi = @view target_child_real[:, 1]
    Rchi = zeros(Float64, nreal(P))

    # native real-basis derivatives, then production-to-analytic normalization
    u_native = native_real_potential(Rphi, local_point, P)
    grad_native = native_real_gradient(Rphi, Rchi, local_point, P, layout)
    hess_native = native_real_hessian(Rphi, Rchi, local_point, P, layout)

    pot = -4pi * u_native
    grad = -4pi * grad_native
    hess = -4pi * hess_native

    r = target_point - source_point
    s = norm(r)
    pot_analytic = strength / s
    grad_analytic = -strength * r / s^3
    I3 = SMatrix{3,3,Float64,9}(1, 0, 0, 0, 1, 0, 0, 0, 1)
    hess_analytic = strength * (3 * (r * transpose(r)) - s^2 * I3) / s^5

    pot_err = abs(pot - pot_analytic) / abs(pot_analytic)
    grad_err = norm(grad .- grad_analytic) / norm(grad_analytic)
    hess_err = maximum(abs.(hess .- hess_analytic)) / maximum(abs.(hess_analytic))

    # native-vs-production cross check on the same chain expansion
    target_child_complex = real_to_complex(target_child_real, P, layout)
    u_prod, grad_prod, hess_prod = production_eval(target_child_complex, local_point, P, layout)
    native_prod_err = max(abs(u_native - u_prod),
        maximum(abs.(grad_native .- grad_prod)),
        maximum(abs.(hess_native .- hess_prod)))

    return (; P, pot_err, grad_err, hess_err, native_prod_err)
end

function write_summary(parity_results, point_results)
    mkpath(DATA_DIR)
    max_scalar = maximum(r.scalar_err for r in parity_results if !isnan(r.scalar_err))
    max_grad = maximum(r.grad_err for r in parity_results)
    max_hess = maximum(r.hess_err for r in parity_results)
    # Symmetry is asserted only for the curl-free Val(false) case; under
    # Lamb-Helmholtz the field Jacobian has a curl (antisymmetric) part by design.
    max_sym = maximum(r.sym_err for r in parity_results if r.layout == "Val(false)")
    max_curl_asym = maximum(r.sym_err for r in parity_results if r.layout == "Val(true)")
    # Trace-free holds for both layouts (the scalar channel phi is harmonic).
    max_trace = maximum(r.trace_abs for r in parity_results)
    max_native_prod = maximum(r.native_prod_err for r in point_results)
    final = point_results[end]

    passed = max(max_scalar, max_grad, max_hess) <= OPERATOR_ATOL &&
        max(max_sym, max_trace) <= OPERATOR_ATOL &&
        max_native_prod <= OPERATOR_ATOL &&
        max(final.pot_err, final.grad_err, final.hess_err) <= POINT_RTOL

    open(SUMMARY_PATH, "w") do io
        println(io, "# Real-Basis Kernel Derivatives Verification Summary")
        println(io)
        println(io, "- Command: `julia --project=. MATRIX_OPERATOR_REFACTOR/scripts/real_basis_kernel_derivatives_verify.jl`")
        println(io, "- Parity tolerance: `atol <= $(OPERATOR_ATOL)`")
        println(io, "- Point-mass convergence target: final `rtol <= $(POINT_RTOL)` after `-4*pi` normalization")
        println(io, "- Status: `$(passed ? "PASS" : "FAIL")`")
        println(io, "- Max native-vs-production scalar error (`Val(false)`): `$(max_scalar)`")
        println(io, "- Max native-vs-production gradient error: `$(max_grad)`")
        println(io, "- Max native-vs-production Hessian error: `$(max_hess)`")
        println(io, "- Max Hessian asymmetry `|H - H^T|` (`Val(false)`, expected ~0): `$(max_sym)`")
        println(io, "- Max field-Jacobian curl asymmetry (`Val(true)`, expected nonzero): `$(max_curl_asym)`")
        println(io, "- Max Hessian trace `|tr H|` (both layouts, expected ~0): `$(max_trace)`")
        println(io, "- Max native-vs-production error on point-mass chain: `$(max_native_prod)`")
        println(io, "- Final point-mass potential/gradient/Hessian rel error: `$(final.pot_err)` / `$(final.grad_err)` / `$(final.hess_err)`")
        println(io)
        println(io, "## Derivative Parity")
        println(io)
        println(io, "Each case evaluates one representable local expansion with production `evaluate_local` (`DerivativesSwitch(true,true,true)`) and the same coefficients with the native real-basis evaluators at the identical local point. Scalar parity is reported only for `Val(false)`; the production scalar `phi` potential is intentionally degenerate under Lamb-Helmholtz. The asymmetry column is `|H - H^T|`: it is ~0 for the curl-free `Val(false)` Hessian and is the expected nonzero curl asymmetry for the `Val(true)` field Jacobian.")
        println(io)
        println(io, "| Layout | P | Point | Scalar err | Gradient err | Hessian err | Asymmetry | `|tr H|` |")
        println(io, "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |")
        for r in parity_results
            scalar_str = isnan(r.scalar_err) ? "n/a" : @sprintf("%.3e", r.scalar_err)
            @printf(io, "| `%s` | %d | `(%.4g, %.4g, %.4g)` | %s | %.3e | %.3e | %.3e | %.3e |\n",
                r.layout, r.P, r.point[1], r.point[2], r.point[3],
                scalar_str, r.grad_err, r.hess_err, r.sym_err, r.trace_abs)
        end
        println(io)
        println(io, "## Unit Point-Mass Convergence")
        println(io)
        println(io, "Native real-basis potential, gradient, and Hessian were evaluated on the approved unit point-mass M2M-M2L-L2L chain local expansion and multiplied by `-4*pi` before comparison to the analytic `1/r`, `grad(1/r) = -r/s^3`, and `Hess(1/r) = (3 r r^T - s^2 I)/s^5`.")
        println(io)
        println(io, "| P | Potential rel err | Gradient rel err | Hessian rel err | Native-vs-production err |")
        println(io, "| ---: | ---: | ---: | ---: | ---: |")
        for r in point_results
            @printf(io, "| %d | %.6e | %.6e | %.6e | %.3e |\n",
                r.P, r.pot_err, r.grad_err, r.hess_err, r.native_prod_err)
        end
    end

    return (; passed, max_scalar, max_grad, max_hess, max_sym, max_curl_asym, max_trace,
        max_native_prod, final_pot = final.pot_err, final_grad = final.grad_err,
        final_hess = final.hess_err)
end

function main()
    # gradient/Hessian require P >= 1: production evaluate_local reads degree n+1
    # coefficients, and its n=0 block assumes at least one higher degree exists.
    orders = (1, 3, 6, 9)
    layouts = (Val(false), Val(true))
    eval_points = (
        SVector{3}(0.04, -0.02, 0.03),
        SVector{3}(-0.03, 0.05, 0.02),
        SVector{3}(0.06, 0.01, -0.04),
        SVector{3}(-0.02, -0.05, 0.07),
    )
    parity_results = [run_parity_case(P, point, layout)
        for P in orders for point in eval_points for layout in layouts]
    point_results = [run_point_mass_case(P) for P in (1, 2, 3, 5, 7, 9, 12)]
    summary = write_summary(parity_results, point_results)

    println("real_basis_kernel_derivatives_verify: $(summary.passed ? "PASS" : "FAIL")")
    println("summary: $(SUMMARY_PATH)")
    println("max_scalar_parity_error: $(summary.max_scalar)")
    println("max_gradient_parity_error: $(summary.max_grad)")
    println("max_hessian_parity_error: $(summary.max_hess)")
    println("max_hessian_asymmetry_val_false: $(summary.max_sym)")
    println("max_curl_asymmetry_val_true: $(summary.max_curl_asym)")
    println("max_hessian_trace: $(summary.max_trace)")
    println("max_point_mass_native_vs_production_error: $(summary.max_native_prod)")
    println("final_point_mass_pot_rel_error: $(summary.final_pot)")
    println("final_point_mass_grad_rel_error: $(summary.final_grad)")
    println("final_point_mass_hess_rel_error: $(summary.final_hess)")

    summary.passed || exit(1)
end

main()
