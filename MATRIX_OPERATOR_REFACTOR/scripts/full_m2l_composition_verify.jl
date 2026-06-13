using FastMultipole
using LinearAlgebra
using Printf
using StaticArrays

const ATOL = 1.0e-9
const RTOL = 2.0e-11
const SCALED_RTOL = 1.0e-12
const POINT_RTOL = 2.5e-7
const DATA_DIR = joinpath(@__DIR__, "..", "data", "full_m2l_composition")
const SUMMARY_PATH = joinpath(DATA_DIR, "verification_summary.md")

function deterministic_expansion(P::Int, ::Val{LH}) where {LH}
    weights = FastMultipole.initialize_expansion(P, Float64)
    ncoeff = size(weights, 3)
    components = LH ? 2 : 1

    for component in 1:components
        for i in 1:ncoeff
            base = 0.041 * i + 0.097 * component
            weights[1, component, i] = sin(base) + 0.31 * cos(1.7 * base)
            weights[2, component, i] = cos(0.8 * base) - 0.19 * sin(2.1 * base)
        end
    end

    return weights
end

function deterministic_target(P::Int, ::Val{LH}) where {LH}
    weights = FastMultipole.initialize_expansion(P, Float64)
    ncoeff = size(weights, 3)
    components = LH ? 2 : 1

    for component in 1:components
        for i in 1:ncoeff
            base = 0.059 * i + 0.151 * component
            weights[1, component, i] = -0.2 + 0.37 * sin(1.3 * base)
            weights[2, component, i] = 0.15 + 0.29 * cos(0.6 * base)
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
            coeff = row_seed
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

function branch(center)
    return FastMultipole.Branch(1:1, 0, 1:0, 0, 1, center, 0.0, SVector{3}(0.0, 0.0, 0.0))
end

function m2l_prealloc(P::Int)
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

function production_m2l!(target, target_branch, source, source_branch, P::Int, layout, cache)
    FastMultipole.multipole_to_local!(
        target, target_branch, source, source_branch, cache.tmp1, cache.tmp2, cache.tmp3,
        cache.Ts, cache.eimphis, cache.zeta, cache.eta, cache.Hs_pi2,
        FastMultipole.M̃, FastMultipole.L̃, P, layout)
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

function run_composition_case(P::Int, label::String, offset::SVector{3,Float64}, layout)
    source_center = SVector{3}(0.2, -0.35, 0.11)
    target_center = source_center + offset
    source_branch = branch(source_center)
    target_branch = branch(target_center)
    source = deterministic_expansion(P, layout)
    initial = deterministic_target(P, layout)

    expected = copy(initial)
    explicit = copy(initial)
    production_m2l!(expected, target_branch, source, source_branch, P, layout, m2l_prealloc(P))
    explicit_m2l!(explicit, target_branch, source, source_branch, P, layout, m2l_prealloc(P))
    max_abs, max_rel = active_errors(explicit, expected, layout)

    return (; P, label, offset, layout = layout isa Val{true} ? "Val(true)" : "Val(false)", max_abs, max_rel)
end

function unscaled_k(n::Int, np::Int, t)
    rho = inv(t)
    coeff = rho
    for k in 1:(n + np)
        coeff *= k * rho
    end
    return coeff
end

function scaled_k(n::Int, np::Int, t)
    D_L = factorial(big(n)) / big(t)^(n + 1)
    Khat = binomial(big(n + np), big(n))
    D_M = factorial(big(np)) / big(t)^np
    return Float64(D_L * Khat * D_M)
end

function run_scaled_case(P::Int, t::Float64)
    max_abs = 0.0
    max_rel = 0.0
    finite_count = 0

    for m in 0:P
        for n in m:P
            for np in m:P
                expected = unscaled_k(n, np, t)
                observed = scaled_k(n, np, t)
                if isfinite(expected) && isfinite(observed)
                    err = abs(observed - expected)
                    scale = max(abs(expected), eps(Float64))
                    max_abs = max(max_abs, err)
                    max_rel = max(max_rel, err / scale)
                    finite_count += 1
                end
            end
        end
    end

    return (; P, t, finite_count, max_abs, max_rel)
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

function point_potential_error(P::Int)
    source_center = SVector{3}(0.1, -0.15, 0.2)
    source_point = source_center + SVector{3}(0.04, -0.03, 0.02)
    target_center = SVector{3}(1.15, 0.85, -0.55)
    target_point = target_center + SVector{3}(0.05, -0.04, 0.03)
    strength = 1.0
    layout = Val(false)

    source = FastMultipole.initialize_expansion(P, Float64)
    source_point_multipole!(source, source_point, source_center, strength, P)
    local_expansion = FastMultipole.initialize_expansion(P, Float64)
    explicit_m2l!(local_expansion, branch(target_center), source, branch(source_center), P, layout, m2l_prealloc(P))

    harmonics = FastMultipole.initialize_harmonics(P)
    gradient_n_m = FastMultipole.initialize_gradient_n_m(P)
    phi_prod, _, _ = FastMultipole.evaluate_local(
        target_point - target_center, harmonics, gradient_n_m, local_expansion,
        P, layout, FastMultipole.DerivativesSwitch(true, false, false))
    phi_one_over_r = -4pi * phi_prod
    analytic = strength / norm(target_point - source_point)

    return (; P, potential = phi_one_over_r, analytic, abs_error = abs(phi_one_over_r - analytic),
        rel_error = abs(phi_one_over_r - analytic) / abs(analytic))
end

function write_summary(composition_results, scaled_results, point_results)
    mkpath(DATA_DIR)

    max_abs = maximum(r.max_abs for r in composition_results)
    max_rel = maximum(r.max_rel for r in composition_results)
    max_scaled_abs = maximum(r.max_abs for r in scaled_results)
    max_scaled_rel = maximum(r.max_rel for r in scaled_results)
    max_point_rel = maximum(r.rel_error for r in point_results)
    last_point_rel = point_results[end].rel_error
    passed = max_abs <= ATOL && max_rel <= RTOL &&
        max_scaled_rel <= SCALED_RTOL &&
        last_point_rel <= POINT_RTOL

    open(SUMMARY_PATH, "w") do io
        println(io, "# Full M2L Composition Verification Summary")
        println(io)
        println(io, "- Command: `julia --project=. MATRIX_OPERATOR_REFACTOR/scripts/full_m2l_composition_verify.jl`")
        println(io, "- Composition tolerance: `atol <= $(ATOL)`, `rtol <= $(RTOL)`")
        println(io, "- Scaled-block tolerance: finite entries compare with `rtol <= $(SCALED_RTOL)`; absolute errors are reported for scale context.")
        println(io, "- Point-mass convergence target: final `rtol <= $(POINT_RTOL)` in `1/r` normalization")
        println(io, "- Status: `$(passed ? "PASS" : "FAIL")`")
        println(io, "- Max composition absolute error: `$(max_abs)`")
        println(io, "- Max composition relative error: `$(max_rel)`")
        println(io, "- Max scaled-block absolute error: `$(max_scaled_abs)`")
        println(io, "- Max scaled-block relative error: `$(max_scaled_rel)`")
        println(io, "- Final point-mass relative error: `$(last_point_rel)`")
        println(io, "- Coverage: axis-aligned `+z`, axis-aligned `-z`, positive off-axis, negative off-axis, layouts `Val(false)` and `Val(true)`, expansion orders `0`, `1`, `3`, `6`, `9`.")
        println(io)
        println(io, "## Composition Cases")
        println(io)
        println(io, "| Layout | Case | P | Offset | Max abs error | Max rel error |")
        println(io, "| --- | --- | ---: | --- | ---: | ---: |")
        for r in composition_results
            @printf(io, "| `%s` | %s | %d | `(%.6g, %.6g, %.6g)` | %.6e | %.6e |\n",
                r.layout, r.label, r.P, r.offset[1], r.offset[2], r.offset[3], r.max_abs, r.max_rel)
        end
        println(io)
        println(io, "## Scaled M2L Blocks")
        println(io)
        println(io, "| P | t | Finite entries | Max abs error | Max rel error |")
        println(io, "| ---: | ---: | ---: | ---: | ---: |")
        for r in scaled_results
            @printf(io, "| %d | %.16g | %d | %.6e | %.6e |\n",
                r.P, r.t, r.finite_count, r.max_abs, r.max_rel)
        end
        println(io)
        println(io, "## Unit Point-Mass M2L Example")
        println(io)
        println(io, "Production scalar-potential results were multiplied by `-4π` before comparison so this table uses the theory `1/r` normalization.")
        println(io)
        println(io, "| P | M2L local evaluation | Analytic `1/r` | Abs error | Rel error |")
        println(io, "| ---: | ---: | ---: | ---: | ---: |")
        for r in point_results
            @printf(io, "| %d | %.16e | %.16e | %.6e | %.6e |\n",
                r.P, r.potential, r.analytic, r.abs_error, r.rel_error)
        end
    end

    return (; passed, max_abs, max_rel, max_scaled_abs, max_scaled_rel, max_point_rel, last_point_rel)
end

function main()
    composition_cases = [
        (0, "axis-aligned +z", SVector{3}(0.0, 0.0, 1.75)),
        (1, "axis-aligned -z", SVector{3}(0.0, 0.0, -2.25)),
        (3, "positive off-axis", SVector{3}(1.7, 0.8, 2.4)),
        (6, "negative off-axis", SVector{3}(-2.3, -0.9, 1.6)),
        (9, "positive off-axis", SVector{3}(2.8, -1.1, 3.3)),
    ]
    layouts = (Val(false), Val(true))
    composition_results = [
        run_composition_case(P, label, offset, layout)
        for (P, label, offset) in composition_cases for layout in layouts
    ]
    scaled_results = [
        run_scaled_case(P, t)
        for P in (0, 1, 3, 6, 9) for t in (1.0e-3, 0.25, 2.0, 1.0e3)
    ]
    point_results = [point_potential_error(P) for P in (0, 1, 2, 3, 5, 7, 9, 12)]
    summary = write_summary(composition_results, scaled_results, point_results)

    println("full_m2l_composition_verify: $(summary.passed ? "PASS" : "FAIL")")
    println("summary: $(SUMMARY_PATH)")
    println("max_composition_abs_error: $(summary.max_abs)")
    println("max_composition_rel_error: $(summary.max_rel)")
    println("max_scaled_abs_error: $(summary.max_scaled_abs)")
    println("max_scaled_rel_error: $(summary.max_scaled_rel)")
    println("final_point_mass_rel_error: $(summary.last_point_rel)")

    summary.passed || exit(1)
end

main()
