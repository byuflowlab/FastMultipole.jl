using FastMultipole
using LinearAlgebra
using Printf
using StaticArrays

const ATOL = 1.0e-9
const RTOL = 2.0e-11
const BLOCK_RTOL = 1.0e-12
const POINT_RTOL = 5.0e-7
const DATA_DIR = joinpath(@__DIR__, "..", "data", "m2m_l2l")
const SUMMARY_PATH = joinpath(DATA_DIR, "verification_summary.md")

function deterministic_expansion(P::Int, ::Val{LH}; seed = 0.0) where {LH}
    weights = FastMultipole.initialize_expansion(P, Float64)
    ncoeff = size(weights, 3)
    components = LH ? 2 : 1

    for component in 1:components
        for i in 1:ncoeff
            base = seed + 0.043 * i + 0.101 * component
            weights[1, component, i] = sin(base) + 0.23 * cos(1.9 * base)
            weights[2, component, i] = cos(0.7 * base) - 0.17 * sin(2.3 * base)
        end
    end

    return weights
end

function deterministic_target(P::Int, ::Val{LH}; seed = 0.0) where {LH}
    weights = FastMultipole.initialize_expansion(P, Float64)
    ncoeff = size(weights, 3)
    components = LH ? 2 : 1

    for component in 1:components
        for i in 1:ncoeff
            base = seed + 0.061 * i + 0.137 * component
            weights[1, component, i] = -0.13 + 0.31 * sin(1.5 * base)
            weights[2, component, i] = 0.19 + 0.27 * cos(0.5 * base)
        end
    end

    return weights
end

function phase_vectors(P::Int, phi::Float64)
    ncoeff = FastMultipole.harmonic_index(P, P)
    C = zeros(Float64, ncoeff)
    S = zeros(Float64, ncoeff)

    for n in 0:P
        for m in 0:n
            i = FastMultipole.harmonic_index(n, m)
            S[i], C[i] = sincos(m * phi)
        end
    end

    return C, S
end

function fused_rotate_z!(out, source, C, S, P::Int, ::Val{LH}) where {LH}
    components = LH ? 2 : 1

    @inbounds for n in 0:P
        for m in 0:n
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
    end

    return out
end

function fused_back_rotate_z!(out, source, C, S, P::Int, ::Val{LH}) where {LH}
    components = LH ? 2 : 1

    @inbounds for n in 0:P
        for m in 0:n
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

    @inbounds for n in 1:P
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

function matrix_translate_m2m_z!(out, source, t, P::Int, ::Val{LH}) where {LH}
    components = LH ? 2 : 1

    for m in 0:P
        for n in m:P
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
    end

    return out
end

function matrix_translate_l2l_z!(out, source, t, P::Int, ::Val{LH}) where {LH}
    components = LH ? 2 : 1

    for m in 0:P
        for n in m:P
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
    end

    return out
end

function operator_lh_multipole!(out, source, r, P::Int)
    out .= source

    for m in 0:P
        for n in P:-1:max(m, 1)
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
    end

    return out
end

function operator_lh_local!(out, source, r, P::Int)
    out .= source

    for m in 0:P
        for n in m:P
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
    end

    return out
end

function matrix_translate_m2l_z!(out, source, t, P::Int, ::Val{LH}) where {LH}
    components = LH ? 2 : 1
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
        eimphis = zeros(Float64, 2, P + 1),
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

function production_m2m!(target, target_branch, source, source_branch, P::Int, layout, cache)
    FastMultipole.multipole_to_multipole!(
        target, target_branch, source, source_branch, cache.tmp1, cache.tmp2,
        cache.Ts, cache.eimphis, cache.zeta, cache.Hs_pi2, P, layout)
    return target
end

function production_l2l!(target, target_branch, source, source_branch, P::Int, layout, cache)
    FastMultipole.local_to_local!(
        target, target_branch, source, source_branch, cache.tmp1, cache.tmp2,
        cache.Ts, cache.eimphis, cache.eta, cache.Hs_pi2, P, layout)
    return target
end

function active_errors(a, b, ::Val{LH}) where {LH}
    components = LH ? 2 : 1
    max_abs = 0.0
    max_rel = 0.0

    @inbounds for i in axes(a, 3), component in 1:components, lane in 1:2
        err = abs(a[lane, component, i] - b[lane, component, i])
        scale = max(abs(b[lane, component, i]), eps(Float64))
        max_abs = max(max_abs, err)
        max_rel = max(max_rel, err / scale)
    end

    return max_abs, max_rel
end

function run_operator_case(kind::Symbol, P::Int, label::String, offset::SVector{3,Float64}, layout)
    source_center = SVector{3}(0.2, -0.35, 0.11)
    target_center = source_center + offset
    source_branch = branch(source_center)
    target_branch = branch(target_center)
    source = deterministic_expansion(P, layout; seed = kind === :m2m ? 0.0 : 0.7)
    initial = deterministic_target(P, layout; seed = kind === :m2m ? 0.2 : 1.1)

    expected = copy(initial)
    explicit = copy(initial)
    if kind === :m2m
        production_m2m!(expected, target_branch, source, source_branch, P, layout, prealloc(P))
        explicit_m2m!(explicit, target_branch, source, source_branch, P, layout, prealloc(P))
    else
        production_l2l!(expected, target_branch, source, source_branch, P, layout, prealloc(P))
        explicit_l2l!(explicit, target_branch, source, source_branch, P, layout, prealloc(P))
    end
    max_abs, max_rel = active_errors(explicit, expected, layout)

    return (; kind, P, label, offset, layout = layout isa Val{true} ? "Val(true)" : "Val(false)", max_abs, max_rel)
end

function m2m_block_coeff(n::Int, np::Int, t)
    np > n && return zero(t)
    k = n - np
    return (-t)^k / factorial(k)
end

function l2l_block_coeff(n::Int, np::Int, t)
    np < n && return zero(t)
    k = np - n
    return (-t)^k / factorial(k)
end

function run_block_case(P::Int, t::Float64)
    source = deterministic_expansion(P, Val(true); seed = 2.0)
    observed_m2m = FastMultipole.initialize_expansion(P, Float64)
    observed_l2l = FastMultipole.initialize_expansion(P, Float64)
    expected_m2m = FastMultipole.initialize_expansion(P, Float64)
    expected_l2l = FastMultipole.initialize_expansion(P, Float64)

    matrix_translate_m2m_z!(observed_m2m, source, t, P, Val(true))
    matrix_translate_l2l_z!(observed_l2l, source, t, P, Val(true))

    for m in 0:P
        for n in m:P
            i_out = FastMultipole.harmonic_index(n, m)
            for component in 1:2
                for np in m:P
                    i_in = FastMultipole.harmonic_index(np, m)
                    cm = m2m_block_coeff(n, np, t)
                    cl = l2l_block_coeff(n, np, t)
                    expected_m2m[1, component, i_out] += cm * source[1, component, i_in]
                    expected_m2m[2, component, i_out] += cm * source[2, component, i_in]
                    expected_l2l[1, component, i_out] += cl * source[1, component, i_in]
                    expected_l2l[2, component, i_out] += cl * source[2, component, i_in]
                end
            end
        end
    end

    m2m_abs, m2m_rel = active_errors(observed_m2m, expected_m2m, Val(true))
    l2l_abs, l2l_rel = active_errors(observed_l2l, expected_l2l, Val(true))
    return (; P, t, m2m_abs, m2m_rel, l2l_abs, l2l_rel)
end

function source_point_multipole!(weights, point, center, strength::Float64, P::Int)
    dx = point - center
    rho, theta, phi = FastMultipole.cartesian_to_spherical(dx)
    harmonics = FastMultipole.initialize_harmonics(P)
    FastMultipole.regular_harmonics!(harmonics, rho, theta, phi, P)

    for n in 0:P
        for m in 0:n
            i = FastMultipole.harmonic_index(n, m)
            Rnm = harmonics[1, 1, i] + im * harmonics[2, 1, i]
            Mnm = -(-1)^(n + m) * strength * conj(Rnm)
            weights[1, 1, i] = real(Mnm)
            weights[2, 1, i] = imag(Mnm)
        end
    end

    return weights
end

function point_chain_error(P::Int)
    source_leaf_center = SVector{3}(0.04, -0.03, 0.02)
    source_point = source_leaf_center + SVector{3}(0.018, -0.012, 0.009)
    source_parent_center = SVector{3}(0.0, 0.0, 0.0)
    target_parent_center = SVector{3}(1.45, 1.1, -0.72)
    target_child_center = target_parent_center + SVector{3}(-0.05, 0.04, 0.03)
    target_point = target_child_center + SVector{3}(0.018, -0.015, 0.011)
    strength = 1.0
    layout = Val(false)

    source_leaf = FastMultipole.initialize_expansion(P, Float64)
    source_point_multipole!(source_leaf, source_point, source_leaf_center, strength, P)

    source_parent = FastMultipole.initialize_expansion(P, Float64)
    explicit_m2m!(source_parent, branch(source_parent_center), source_leaf,
        branch(source_leaf_center), P, layout, prealloc(P))

    target_parent = FastMultipole.initialize_expansion(P, Float64)
    explicit_m2l!(target_parent, branch(target_parent_center), source_parent,
        branch(source_parent_center), P, layout, prealloc(P))

    target_child = FastMultipole.initialize_expansion(P, Float64)
    explicit_l2l!(target_child, branch(target_child_center), target_parent,
        branch(target_parent_center), P, layout, prealloc(P))

    harmonics = FastMultipole.initialize_harmonics(P)
    gradient_n_m = FastMultipole.initialize_gradient_n_m(P)
    phi_prod, _, _ = FastMultipole.evaluate_local(
        target_point - target_child_center, harmonics, gradient_n_m, target_child,
        P, layout, FastMultipole.DerivativesSwitch(true, false, false))
    phi_one_over_r = -4pi * phi_prod
    analytic = strength / norm(target_point - source_point)

    return (; P, potential = phi_one_over_r, analytic, abs_error = abs(phi_one_over_r - analytic),
        rel_error = abs(phi_one_over_r - analytic) / abs(analytic))
end

function write_summary(operator_results, block_results, point_results)
    mkpath(DATA_DIR)

    max_m2m_abs = maximum(r.max_abs for r in operator_results if r.kind === :m2m)
    max_m2m_rel = maximum(r.max_rel for r in operator_results if r.kind === :m2m)
    max_l2l_abs = maximum(r.max_abs for r in operator_results if r.kind === :l2l)
    max_l2l_rel = maximum(r.max_rel for r in operator_results if r.kind === :l2l)
    max_block_abs = maximum(max(r.m2m_abs, r.l2l_abs) for r in block_results)
    max_block_rel = maximum(max(r.m2m_rel, r.l2l_rel) for r in block_results)
    last_point_rel = point_results[end].rel_error
    passed = max_m2m_abs <= ATOL && max_m2m_rel <= RTOL &&
        max_l2l_abs <= ATOL && max_l2l_rel <= RTOL &&
        max_block_rel <= BLOCK_RTOL &&
        last_point_rel <= POINT_RTOL

    open(SUMMARY_PATH, "w") do io
        println(io, "# M2M and L2L Extension Verification Summary")
        println(io)
        println(io, "- Command: `julia --project=. MATRIX_OPERATOR_REFACTOR/scripts/m2m_l2l_verify.jl`")
        println(io, "- Operator tolerance: `atol <= $(ATOL)`, `rtol <= $(RTOL)`")
        println(io, "- Z-block tolerance: finite entries compare with `rtol <= $(BLOCK_RTOL)`; absolute errors are reported for scale context.")
        println(io, "- Point-mass chain convergence target: final `rtol <= $(POINT_RTOL)` in `1/r` normalization")
        println(io, "- Status: `$(passed ? "PASS" : "FAIL")`")
        println(io, "- Max M2M absolute error: `$(max_m2m_abs)`")
        println(io, "- Max M2M relative error: `$(max_m2m_rel)`")
        println(io, "- Max L2L absolute error: `$(max_l2l_abs)`")
        println(io, "- Max L2L relative error: `$(max_l2l_rel)`")
        println(io, "- Max z-block absolute error: `$(max_block_abs)`")
        println(io, "- Max z-block relative error: `$(max_block_rel)`")
        println(io, "- Final point-chain relative error: `$(last_point_rel)`")
        println(io, "- Coverage: axis-aligned `+z`, axis-aligned `-z`, positive off-axis, negative off-axis, layouts `Val(false)` and `Val(true)`, expansion orders `0`, `1`, `3`, `6`, `9`.")
        println(io)
        println(io, "## Operator Cases")
        println(io)
        println(io, "| Operator | Layout | Case | P | Offset | Max abs error | Max rel error |")
        println(io, "| --- | --- | --- | ---: | --- | ---: | ---: |")
        for r in operator_results
            @printf(io, "| `%s` | `%s` | %s | %d | `(%.6g, %.6g, %.6g)` | %.6e | %.6e |\n",
                uppercase(String(r.kind)), r.layout, r.label, r.P, r.offset[1], r.offset[2], r.offset[3], r.max_abs, r.max_rel)
        end
        println(io)
        println(io, "## Z-Translation Blocks")
        println(io)
        println(io, "| P | t | M2M abs | M2M rel | L2L abs | L2L rel |")
        println(io, "| ---: | ---: | ---: | ---: | ---: | ---: |")
        for r in block_results
            @printf(io, "| %d | %.16g | %.6e | %.6e | %.6e | %.6e |\n",
                r.P, r.t, r.m2m_abs, r.m2m_rel, r.l2l_abs, r.l2l_rel)
        end
        println(io)
        println(io, "## Unit Point-Mass M2M-M2L-L2L Chain")
        println(io)
        println(io, "Production scalar-potential results were multiplied by `-4π` before comparison so this table uses the theory `1/r` normalization.")
        println(io)
        println(io, "| P | Chain local evaluation | Analytic `1/r` | Abs error | Rel error |")
        println(io, "| ---: | ---: | ---: | ---: | ---: |")
        for r in point_results
            @printf(io, "| %d | %.16e | %.16e | %.6e | %.6e |\n",
                r.P, r.potential, r.analytic, r.abs_error, r.rel_error)
        end
    end

    return (; passed, max_m2m_abs, max_m2m_rel, max_l2l_abs, max_l2l_rel,
        max_block_abs, max_block_rel, last_point_rel)
end

function main()
    operator_cases = [
        (0, "axis-aligned +z", SVector{3}(0.0, 0.0, 1.75)),
        (1, "axis-aligned -z", SVector{3}(0.0, 0.0, -2.25)),
        (3, "positive off-axis", SVector{3}(1.7, 0.8, 2.4)),
        (6, "negative off-axis", SVector{3}(-2.3, -0.9, 1.6)),
        (9, "positive off-axis", SVector{3}(2.8, -1.1, 3.3)),
    ]
    layouts = (Val(false), Val(true))
    operator_results = [
        run_operator_case(kind, P, label, offset, layout)
        for kind in (:m2m, :l2l)
        for (P, label, offset) in operator_cases
        for layout in layouts
    ]
    block_results = [
        run_block_case(P, t)
        for P in (0, 1, 3, 6, 9) for t in (1.0e-3, 0.25, 2.0, 1.0e3)
    ]
    point_results = [point_chain_error(P) for P in (0, 1, 2, 3, 5, 7, 9, 12)]
    summary = write_summary(operator_results, block_results, point_results)

    println("m2m_l2l_verify: $(summary.passed ? "PASS" : "FAIL")")
    println("summary: $(SUMMARY_PATH)")
    println("max_m2m_abs_error: $(summary.max_m2m_abs)")
    println("max_m2m_rel_error: $(summary.max_m2m_rel)")
    println("max_l2l_abs_error: $(summary.max_l2l_abs)")
    println("max_l2l_rel_error: $(summary.max_l2l_rel)")
    println("max_z_block_abs_error: $(summary.max_block_abs)")
    println("max_z_block_rel_error: $(summary.max_block_rel)")
    println("final_point_chain_rel_error: $(summary.last_point_rel)")

    summary.passed || exit(1)
end

main()
