using FastMultipole
using LinearAlgebra
using Printf
using StaticArrays

const EXACT_ATOL = 1.0e-14
const OPERATOR_ATOL = 1.0e-12
const POINT_RTOL = 5.0e-7
const DATA_DIR = joinpath(@__DIR__, "..", "data", "real_solid_harmonic")
const SUMMARY_PATH = joinpath(DATA_DIR, "verification_summary.md")

mode_index(n::Int, ::Val{:zero}) = n^2 + 1
mode_index(n::Int, m::Int, ::Val{:cos}) = n^2 + 2m
mode_index(n::Int, m::Int, ::Val{:sin}) = n^2 + 2m + 1
nreal(P::Int) = (P + 1)^2

function active_components(::Val{LH}) where {LH}
    return LH ? 2 : 1
end

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

function deterministic_real(P::Int, layout; seed = 0.0)
    components = active_components(layout)
    real_weights = zeros(Float64, nreal(P), components)
    for component in 1:components
        for n in 0:P
            i0 = mode_index(n, Val(:zero))
            base = seed + 0.053 * i0 + 0.17 * component
            real_weights[i0, component] = sin(base) - 0.21 * cos(1.3 * base)
            for m in 1:n
                ic = mode_index(n, m, Val(:cos))
                is = mode_index(n, m, Val(:sin))
                basec = seed + 0.053 * ic + 0.17 * component
                bases = seed + 0.053 * is + 0.17 * component
                real_weights[ic, component] = sin(basec) - 0.21 * cos(1.3 * basec)
                real_weights[is, component] = cos(0.7 * bases) + 0.18 * sin(1.9 * bases)
            end
        end
    end
    return real_weights
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
        c = C[i]
        s = S[i]
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
        c = C[i]
        s = S[i]
        for component in 1:components
            real_in = source[1, component, i]
            imag_in = source[2, component, i]
            out[1, component, i] += c * real_in + s * imag_in
            out[2, component, i] += -s * real_in + c * imag_in
        end
    end
    return out
end

function rotate_z_real(real_source, P::Int, phi::Float64, layout)
    complex_source = real_to_complex(real_source, P, layout)
    C, S = phase_vectors(P, phi)
    complex_out = FastMultipole.initialize_expansion(P, Float64)
    fused_rotate_z!(complex_out, complex_source, C, S, P, layout)
    return complex_to_real(complex_out, P, layout)
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
            real_acc = 0.0
            imag_acc = 0.0
            coeff = one(t)
            factorial = one(t)
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
            real_acc = 0.0
            imag_acc = 0.0
            coeff = one(t)
            factorial = one(t)
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
                real_acc = 0.0
                imag_acc = 0.0
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

function operator_lh_multipole!(out, source, r, P::Int)
    out .= source
    for m in 0:P, n in P:-1:max(m, 1)
        i = FastMultipole.harmonic_index(n, m)
        chi_re = source[1, 2, i]
        chi_im = source[2, 2, i]
        a = r * m / (n + 1)
        out[1, 1, i] = source[1, 1, i] + a * chi_im
        out[2, 1, i] = source[2, 1, i] - a * chi_re
        if m < n
            i_prev = FastMultipole.harmonic_index(n - 1, m)
            b = r / n
            out[1, 2, i] = chi_re + b * source[1, 2, i_prev]
            out[2, 2, i] = chi_im + b * source[2, 2, i_prev]
        end
    end
    return out
end

function operator_lh_local!(out, source, r, P::Int)
    out .= source
    for m in 0:P, n in m:P
        i = FastMultipole.harmonic_index(n, m)
        chi_re = source[1, 2, i]
        chi_im = source[2, 2, i]
        if n > 0
            a = r * m / n
            out[1, 1, i] = source[1, 1, i] - a * chi_im
            out[2, 1, i] = source[2, 1, i] + a * chi_re
        end
        if n < P
            i_next = FastMultipole.harmonic_index(n + 1, m)
            b = r / (n + 1)
            out[1, 2, i] = chi_re - b * source[1, 2, i_next]
            out[2, 2, i] = chi_im - b * source[2, 2, i_next]
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
        Hs_pi2,
        zeta,
        eta,
        Ts = zeros(Float64, FastMultipole.length_Ts(P)),
        tmp1 = FastMultipole.initialize_expansion(P, Float64),
        tmp2 = FastMultipole.initialize_expansion(P, Float64),
        tmp3 = FastMultipole.initialize_expansion(P, Float64),
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
    layout isa Val{true} && operator_lh_multipole!(cache.tmp1, copy(cache.tmp1), r, P)
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
    layout isa Val{true} && operator_lh_local!(cache.tmp1, copy(cache.tmp1), r, P)
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
    layout isa Val{true} && operator_lh_local!(cache.tmp1, copy(cache.tmp1), r, P)
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

function max_active_abs(a, b, layout)
    components = active_components(layout)
    return maximum(abs, @view(a[:, 1:components]) .- @view(b[:, 1:components]))
end

function max_complex_active_abs(a, b, layout)
    components = active_components(layout)
    return maximum(abs, @view(a[:, 1:components, :]) .- @view(b[:, 1:components, :]))
end

function run_index_case(P::Int)
    indices = Int[]
    for n in 0:P
        push!(indices, mode_index(n, Val(:zero)))
        for m in 1:n
            push!(indices, mode_index(n, m, Val(:cos)))
            push!(indices, mode_index(n, m, Val(:sin)))
        end
    end
    sorted = sort(indices)
    expected = collect(1:nreal(P))
    return (; P, count = length(indices), unique = length(unique(indices)) == nreal(P),
        contiguous = sorted == expected)
end

function run_transform_case(P::Int, layout)
    real_weights = deterministic_real(P, layout; seed = 0.2)
    real_roundtrip = complex_to_real(real_to_complex(real_weights, P, layout), P, layout)
    real_rt_error = max_active_abs(real_roundtrip, real_weights, layout)

    complex_weights = deterministic_complex(P, layout; seed = 0.5, zero_m0_imag = true)
    complex_roundtrip = real_to_complex(complex_to_real(complex_weights, P, layout), P, layout)
    complex_rt_error = max_complex_active_abs(complex_roundtrip, complex_weights, layout)

    projected = deterministic_complex(P, layout; seed = 0.8, zero_m0_imag = false)
    projected_roundtrip = real_to_complex(complex_to_real(projected, P, layout), P, layout)
    components = active_components(layout)
    m0_projection_error = 0.0
    represented_error = 0.0
    for component in 1:components, n in 0:P
        i = FastMultipole.harmonic_index(n, 0)
        m0_projection_error = max(m0_projection_error, abs(projected_roundtrip[2, component, i]))
    end
    for component in 1:components, n in 0:P, m in 0:n
        i = FastMultipole.harmonic_index(n, m)
        represented_error = max(represented_error, abs(projected_roundtrip[1, component, i] - projected[1, component, i]))
        if m > 0
            represented_error = max(represented_error, abs(projected_roundtrip[2, component, i] - projected[2, component, i]))
        end
    end
    original_m0_imag = maximum(abs(projected[2, component, FastMultipole.harmonic_index(n, 0)])
        for component in 1:components for n in 0:P)

    return (; P, layout = layout isa Val{true} ? "Val(true)" : "Val(false)",
        real_rt_error, complex_rt_error, m0_projection_error, represented_error, original_m0_imag)
end

function run_z_rotation_case(P::Int, phi::Float64, layout)
    real_source = deterministic_real(P, layout; seed = 1.3)
    real_rotated = rotate_z_real(real_source, P, phi, layout)

    complex_source = real_to_complex(real_source, P, layout)
    complex_rotated = FastMultipole.initialize_expansion(P, Float64)
    C, S = phase_vectors(P, phi)
    fused_rotate_z!(complex_rotated, complex_source, C, S, P, layout)
    expected_real = complex_to_real(complex_rotated, P, layout)
    error = max_active_abs(real_rotated, expected_real, layout)
    return (; P, phi, layout = layout isa Val{true} ? "Val(true)" : "Val(false)", error)
end

function evaluate_scalar_local(weights, point, P::Int)
    harmonics = FastMultipole.initialize_harmonics(P)
    gradient_n_m = FastMultipole.initialize_gradient_n_m(P)
    phi, _, _ = FastMultipole.evaluate_local(
        point, harmonics, gradient_n_m, weights,
        P, Val(false), FastMultipole.DerivativesSwitch(true, false, false))
    return phi
end

function evaluate_real_scalar_local(real_weights, point, P::Int)
    rho, theta, phi = FastMultipole.cartesian_to_spherical(point)
    harmonics = FastMultipole.initialize_harmonics(P)
    FastMultipole.regular_harmonics!(harmonics, rho, theta, phi, P)

    u_raw = 0.0
    for n in 0:P
        i_harmonic = FastMultipole.harmonic_index(n, 0)
        i_real = mode_index(n, Val(:zero))
        u_raw += harmonics[1, 1, i_harmonic] * real_weights[i_real, 1]
        for m in 1:n
            i_harmonic = FastMultipole.harmonic_index(n, m)
            i_cos = mode_index(n, m, Val(:cos))
            i_sin = mode_index(n, m, Val(:sin))
            p = harmonics[1, 1, i_harmonic]
            q = harmonics[2, 1, i_harmonic]
            u_raw += 2.0 * (p * real_weights[i_cos, 1] - q * real_weights[i_sin, 1])
        end
    end
    return u_raw / (4pi)
end

function run_evaluation_case(P::Int, point::SVector{3,Float64})
    layout = Val(false)
    complex_weights = deterministic_complex(P, layout; seed = 2.4, zero_m0_imag = true)
    real_weights = complex_to_real(complex_weights, P, layout)
    complex_value = evaluate_scalar_local(complex_weights, point, P)
    real_value = evaluate_real_scalar_local(real_weights, point, P)
    error = abs(real_value - complex_value)
    scale = max(abs(complex_value), eps(Float64))
    return (; P, point, complex_value, real_value, abs_error = error, rel_error = error / scale)
end

function run_operator_case(kind::Symbol, P::Int, label::String, offset::SVector{3,Float64}, layout)
    source_center = SVector{3}(0.2, -0.35, 0.11)
    target_center = source_center + offset
    source_branch = branch(source_center)
    target_branch = branch(target_center)
    source_real = complex_to_real(deterministic_complex(P, layout; seed = kind === :m2l ? 1.1 : 1.4), P, layout)
    target_real = complex_to_real(deterministic_complex(P, layout; seed = 1.8), P, layout)

    observed_real = apply_real_operator(kind, target_real, target_branch, source_real, source_branch, P, layout)

    source_complex = real_to_complex(source_real, P, layout)
    expected_complex = real_to_complex(target_real, P, layout)
    if kind === :m2l
        explicit_m2l!(expected_complex, target_branch, source_complex, source_branch, P, layout, prealloc(P))
    elseif kind === :m2m
        explicit_m2m!(expected_complex, target_branch, source_complex, source_branch, P, layout, prealloc(P))
    else
        explicit_l2l!(expected_complex, target_branch, source_complex, source_branch, P, layout, prealloc(P))
    end
    expected_real = complex_to_real(expected_complex, P, layout)
    error = max_active_abs(observed_real, expected_real, layout)
    return (; kind, P, label, offset, layout = layout isa Val{true} ? "Val(true)" : "Val(false)", error)
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

function real_point_chain_error(P::Int)
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

    source_parent_real = zeros(Float64, nreal(P), 1)
    source_parent_real = apply_real_operator(:m2m, source_parent_real,
        branch(source_parent_center), source_leaf_real, branch(source_leaf_center), P, layout)

    target_parent_real = zeros(Float64, nreal(P), 1)
    target_parent_real = apply_real_operator(:m2l, target_parent_real,
        branch(target_parent_center), source_parent_real, branch(source_parent_center), P, layout)

    target_child_real = zeros(Float64, nreal(P), 1)
    target_child_real = apply_real_operator(:l2l, target_child_real,
        branch(target_child_center), target_parent_real, branch(target_parent_center), P, layout)

    target_child_complex = real_to_complex(target_child_real, P, layout)
    harmonics = FastMultipole.initialize_harmonics(P)
    gradient_n_m = FastMultipole.initialize_gradient_n_m(P)
    phi_prod, _, _ = FastMultipole.evaluate_local(
        target_point - target_child_center, harmonics, gradient_n_m, target_child_complex,
        P, layout, FastMultipole.DerivativesSwitch(true, false, false))
    phi_one_over_r = -4pi * phi_prod
    analytic = strength / norm(target_point - source_point)

    return (; P, potential = phi_one_over_r, analytic, abs_error = abs(phi_one_over_r - analytic),
        rel_error = abs(phi_one_over_r - analytic) / abs(analytic))
end

function write_summary(index_results, transform_results, z_results, evaluation_results, operator_results, point_results)
    mkpath(DATA_DIR)
    index_passed = all(r.unique && r.contiguous for r in index_results)
    max_real_rt = maximum(r.real_rt_error for r in transform_results)
    max_complex_rt = maximum(r.complex_rt_error for r in transform_results)
    max_projected_m0 = maximum(r.m0_projection_error for r in transform_results)
    max_represented_projection = maximum(r.represented_error for r in transform_results)
    max_z = maximum(r.error for r in z_results)
    max_eval_abs = maximum(r.abs_error for r in evaluation_results)
    max_eval_rel = maximum(r.rel_error for r in evaluation_results)
    max_m2l = maximum(r.error for r in operator_results if r.kind === :m2l)
    max_chain_operator = maximum(r.error for r in operator_results)
    final_point_rel = point_results[end].rel_error
    passed = index_passed &&
        max(max_real_rt, max_complex_rt, max_projected_m0, max_represented_projection) <= EXACT_ATOL &&
        max_eval_abs <= OPERATOR_ATOL &&
        max(max_z, max_chain_operator) <= OPERATOR_ATOL &&
        final_point_rel <= POINT_RTOL

    open(SUMMARY_PATH, "w") do io
        println(io, "# Real Solid Harmonic Transform Verification Summary")
        println(io)
        println(io, "- Command: `julia --project=. MATRIX_OPERATOR_REFACTOR/scripts/real_solid_harmonic_transforms_verify.jl`")
        println(io, "- Transform/index tolerance: `atol <= $(EXACT_ATOL)`")
        println(io, "- Operator parity tolerance: `atol <= $(OPERATOR_ATOL)`")
        println(io, "- Point-mass chain convergence target: final `rtol <= $(POINT_RTOL)` in `1/r` normalization")
        println(io, "- Status: `$(passed ? "PASS" : "FAIL")`")
        println(io, "- Mode indices contiguous and unique: `$(index_passed)`")
        println(io, "- Max real-to-complex-to-real error: `$(max_real_rt)`")
        println(io, "- Max complex-to-real-to-complex representable error: `$(max_complex_rt)`")
        println(io, "- Max projected `m = 0` imaginary lane after round trip: `$(max_projected_m0)`")
        println(io, "- Max preserved-lane error during intentional projection: `$(max_represented_projection)`")
        println(io, "- Max real z-rotation parity error: `$(max_z)`")
        println(io, "- Max native real same-point scalar evaluation absolute error: `$(max_eval_abs)`")
        println(io, "- Max native real same-point scalar evaluation relative error: `$(max_eval_rel)`")
        println(io, "- Max real M2L parity error: `$(max_m2l)`")
        println(io, "- Max real M2M/M2L/L2L operator parity error: `$(max_chain_operator)`")
        println(io, "- Final point-chain relative error: `$(final_point_rel)`")
        println(io)
        println(io, "## Index Cases")
        println(io)
        println(io, "| P | Count | Unique | Contiguous |")
        println(io, "| ---: | ---: | --- | --- |")
        for r in index_results
            println(io, "| $(r.P) | $(r.count) | `$(r.unique)` | `$(r.contiguous)` |")
        end
        println(io)
        println(io, "## Transform Round Trips")
        println(io)
        println(io, "| Layout | P | R-C-R error | C-R-C error | Projected m=0 imag | Original m=0 imag | Preserved-lane error |")
        println(io, "| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for r in transform_results
            @printf(io, "| `%s` | %d | %.6e | %.6e | %.6e | %.6e | %.6e |\n",
                r.layout, r.P, r.real_rt_error, r.complex_rt_error,
                r.m0_projection_error, r.original_m0_imag, r.represented_error)
        end
        println(io)
        println(io, "## Z-Rotation Parity")
        println(io)
        println(io, "| Layout | P | phi | Max error |")
        println(io, "| --- | ---: | ---: | ---: |")
        for r in z_results
            @printf(io, "| `%s` | %d | %.16g | %.6e |\n", r.layout, r.P, r.phi, r.error)
        end
        println(io)
        println(io, "## Same-Point Scalar Evaluation")
        println(io)
        println(io, "Each case evaluates one representable compressed complex scalar local expansion with production `evaluate_local` and the same coefficients with the native real-basis scalar evaluator at identical local points.")
        println(io)
        println(io, "| P | Point | Complex value | Native real-basis value | Abs error | Rel error |")
        println(io, "| ---: | --- | ---: | ---: | ---: | ---: |")
        for r in evaluation_results
            @printf(io, "| %d | `(%.6g, %.6g, %.6g)` | %.16e | %.16e | %.6e | %.6e |\n",
                r.P, r.point[1], r.point[2], r.point[3],
                r.complex_value, r.real_value, r.abs_error, r.rel_error)
        end
        println(io)
        println(io, "## Real Operator Parity")
        println(io)
        println(io, "| Operator | Layout | Case | P | Offset | Max error |")
        println(io, "| --- | --- | --- | ---: | --- | ---: |")
        for r in operator_results
            @printf(io, "| `%s` | `%s` | %s | %d | `(%.6g, %.6g, %.6g)` | %.6e |\n",
                uppercase(String(r.kind)), r.layout, r.label, r.P,
                r.offset[1], r.offset[2], r.offset[3], r.error)
        end
        println(io)
        println(io, "## Unit Point-Mass Real M2M-M2L-L2L Chain")
        println(io)
        println(io, "Production scalar-potential results were multiplied by `-4*pi` before comparison so this table uses the theory `1/r` normalization.")
        println(io)
        println(io, "| P | Chain local evaluation | Analytic `1/r` | Abs error | Rel error |")
        println(io, "| ---: | ---: | ---: | ---: | ---: |")
        for r in point_results
            @printf(io, "| %d | %.16e | %.16e | %.6e | %.6e |\n",
                r.P, r.potential, r.analytic, r.abs_error, r.rel_error)
        end
    end

    return (; passed, index_passed, max_real_rt, max_complex_rt, max_projected_m0,
        max_represented_projection, max_z, max_eval_abs, max_eval_rel, max_m2l,
        max_chain_operator, final_point_rel)
end

function main()
    orders = (0, 1, 3, 6, 9)
    layouts = (Val(false), Val(true))
    index_results = [run_index_case(P) for P in orders]
    transform_results = [run_transform_case(P, layout) for P in orders for layout in layouts]
    z_results = [run_z_rotation_case(P, phi, layout)
        for (P, phi) in zip(orders, (0.0, 0.25, -1.125, pi / 3, 2.4))
        for layout in layouts]
    evaluation_points = (
        SVector{3}(0.0, 0.0, 0.0),
        SVector{3}(0.04, -0.02, 0.03),
        SVector{3}(-0.03, 0.05, 0.02),
        SVector{3}(0.06, 0.01, -0.04),
        SVector{3}(-0.02, -0.05, 0.07),
        SVector{3}(0.09, -0.08, 0.01),
        SVector{3}(-0.07, 0.03, -0.06),
    )
    evaluation_results = [run_evaluation_case(P, point) for P in orders for point in evaluation_points]
    operator_cases = [
        (0, "axis-aligned +z", SVector{3}(0.0, 0.0, 1.75)),
        (1, "axis-aligned -z", SVector{3}(0.0, 0.0, -2.25)),
        (3, "positive off-axis", SVector{3}(1.7, 0.8, 2.4)),
        (6, "negative off-axis", SVector{3}(-2.3, -0.9, 1.6)),
        (9, "positive off-axis", SVector{3}(2.8, -1.1, 3.3)),
    ]
    operator_results = [
        run_operator_case(kind, P, label, offset, layout)
        for kind in (:m2l, :m2m, :l2l)
        for (P, label, offset) in operator_cases
        for layout in layouts
    ]
    point_results = [real_point_chain_error(P) for P in (0, 1, 2, 3, 5, 7, 9, 12)]
    summary = write_summary(index_results, transform_results, z_results, evaluation_results, operator_results, point_results)

    println("real_solid_harmonic_transforms_verify: $(summary.passed ? "PASS" : "FAIL")")
    println("summary: $(SUMMARY_PATH)")
    println("index_passed: $(summary.index_passed)")
    println("max_real_roundtrip_error: $(summary.max_real_rt)")
    println("max_complex_roundtrip_error: $(summary.max_complex_rt)")
    println("max_projected_m0_imag: $(summary.max_projected_m0)")
    println("max_z_rotation_error: $(summary.max_z)")
    println("max_native_real_same_point_evaluation_abs_error: $(summary.max_eval_abs)")
    println("max_native_real_same_point_evaluation_rel_error: $(summary.max_eval_rel)")
    println("max_m2l_error: $(summary.max_m2l)")
    println("max_chain_operator_error: $(summary.max_chain_operator)")
    println("final_point_chain_rel_error: $(summary.final_point_rel)")

    summary.passed || exit(1)
end

main()
