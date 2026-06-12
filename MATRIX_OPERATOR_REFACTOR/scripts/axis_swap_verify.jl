using FastMultipole
using Printf

const ATOL = 1.0e-12
const DATA_DIR = joinpath(@__DIR__, "..", "data", "axis_swap")
const SUMMARY_PATH = joinpath(DATA_DIR, "verification_summary.md")

function deterministic_expansion(P::Int, ::Val{LH}) where {LH}
    weights = FastMultipole.initialize_expansion(P, Float64)
    ncoeff = size(weights, 3)
    components = LH ? 2 : 1

    for component in 1:components
        for i in 1:ncoeff
            base = 0.071 * i + 0.113 * component
            weights[1, component, i] = sin(base) + 0.5cos(1.7base)
            weights[2, component, i] = cos(0.9base) - 0.25sin(2.3base)
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
            base = 0.037 * i + 0.19 * component
            weights[1, component, i] = -0.3 + sin(1.1base)
            weights[2, component, i] = 0.4 + cos(0.8base)
        end
    end

    return weights
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

function active_max_abs_diff(a, b, ::Val{LH}) where {LH}
    components = LH ? 2 : 1
    return maximum(abs, @view(a[:, 1:components, :]) .- @view(b[:, 1:components, :]))
end

function rotate_reference(source, theta::Float64, P::Int, Hs_pi2, signs, layout, kind::Symbol)
    out = fill(-71.0, size(source))
    Ts = zeros(Float64, FastMultipole.length_Ts(P))
    if kind === :multipole
        FastMultipole.rotate_multipole_y!(out, source, Ts, Hs_pi2, signs, theta, P, layout)
    else
        FastMultipole.rotate_local_y!(out, source, Ts, Hs_pi2, signs, theta, P, layout)
    end
    return out, Ts
end

function rotate_axis_swap(source, theta::Float64, P::Int, Hs_pi2, signs, layout, kind::Symbol)
    out = fill(53.0, size(source))
    Ts = zeros(Float64, FastMultipole.length_Ts(P))
    axis_swap_update_Ts!(Ts, Hs_pi2, theta, P)
    if kind === :multipole
        FastMultipole._rotate_multipole_y!(out, source, Ts, signs, P, layout)
    else
        FastMultipole._rotate_local_y!(out, source, Ts, Hs_pi2, signs, P, layout)
    end
    return out, Ts
end

function max_inactive_reset(weights, ::Val{LH}) where {LH}
    LH && return 0.0
    return maximum(abs, @view(weights[:, 2:2, :]))
end

function run_case(P::Int, theta::Real, label::String, layout)
    theta = Float64(theta)
    source = deterministic_expansion(P, layout)
    target = deterministic_target(P, layout)
    Hs_pi2 = [1.0]
    FastMultipole.update_Hs_π2!(Hs_pi2, P)

    zeta = zeros(Float64, FastMultipole.length_ζs(P))
    eta = zeros(Float64, FastMultipole.length_ηs(P))
    FastMultipole.update_ζs_mag!(zeta, 0, P)
    FastMultipole.update_ηs_mag!(eta, 0, P)

    ref_mp, prod_T_mp = rotate_reference(source, theta, P, Hs_pi2, zeta, layout, :multipole)
    axis_mp, axis_T = rotate_axis_swap(source, theta, P, Hs_pi2, zeta, layout, :multipole)
    ref_local, prod_T_local = rotate_reference(source, theta, P, Hs_pi2, eta, layout, :local)
    axis_local, _ = rotate_axis_swap(source, theta, P, Hs_pi2, eta, layout, :local)

    back_mp = copy(target)
    FastMultipole.back_rotate_multipole_y!(back_mp, axis_mp, axis_T, zeta, P, layout)
    reset_back_mp = FastMultipole.initialize_expansion(P, Float64)
    FastMultipole.back_rotate_multipole_y!(reset_back_mp, axis_mp, axis_T, zeta, P, layout)

    back_local = copy(target)
    FastMultipole.back_rotate_local_y!(back_local, axis_local, axis_T, Hs_pi2, eta, P, layout)
    reset_back_local = FastMultipole.initialize_expansion(P, Float64)
    FastMultipole.back_rotate_local_y!(reset_back_local, axis_local, axis_T, Hs_pi2, eta, P, layout)

    return (;
        label,
        P,
        theta,
        layout = layout isa Val{true} ? "Val(true)" : "Val(false)",
        multipole_error = active_max_abs_diff(axis_mp, ref_mp, layout),
        local_error = active_max_abs_diff(axis_local, ref_local, layout),
        T_multipole_error = maximum(abs, axis_T .- prod_T_mp),
        T_local_error = maximum(abs, axis_T .- prod_T_local),
        multipole_overwrite_inactive = max_inactive_reset(axis_mp, layout),
        local_overwrite_inactive = max_inactive_reset(axis_local, layout),
        multipole_reset_error = active_max_abs_diff(back_mp, reset_back_mp, layout),
        local_reset_error = active_max_abs_diff(back_local, reset_back_local, layout),
    )
end

function write_summary(results)
    mkpath(DATA_DIR)

    max_mp = maximum(r.multipole_error for r in results)
    max_local = maximum(r.local_error for r in results)
    max_T = maximum(max(r.T_multipole_error, r.T_local_error) for r in results)
    max_inactive = maximum(max(r.multipole_overwrite_inactive, r.local_overwrite_inactive) for r in results)
    max_reset = maximum(max(r.multipole_reset_error, r.local_reset_error) for r in results)
    passed = max(max_mp, max_local, max_T, max_inactive, max_reset) <= ATOL

    open(SUMMARY_PATH, "w") do io
        println(io, "# Axis-Swap Verification Summary")
        println(io)
        println(io, "- Command: `julia --project=. MATRIX_OPERATOR_REFACTOR/scripts/axis_swap_verify.jl`")
        println(io, "- Tolerance: `atol <= $(ATOL)`")
        println(io, "- Status: `$(passed ? "PASS" : "FAIL")`")
        println(io, "- Max multipole-axis-swap error: `$(max_mp)`")
        println(io, "- Max local-axis-swap error: `$(max_local)`")
        println(io, "- Max `T` reconstruction error: `$(max_T)`")
        println(io, "- Max inactive-channel reset error: `$(max_inactive)`")
        println(io, "- Max reset/back-rotation target-independence error: `$(max_reset)`")
        println(io, "- Layout coverage: `Val(false)`, `Val(true)`")
        println(io, "- Case coverage: axis-aligned `theta = 0`, `theta = pi`; off-axis positive and negative angles.")
        println(io)
        println(io, "| Layout | Case | P | theta | Multipole error | Local error | T error | Inactive reset | Reset/back target independence |")
        println(io, "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for r in results
            @printf(io, "| `%s` | %s | %d | %.16g | %.6e | %.6e | %.6e | %.6e | %.6e |\n",
                r.layout, r.label, r.P, r.theta, r.multipole_error, r.local_error,
                max(r.T_multipole_error, r.T_local_error),
                max(r.multipole_overwrite_inactive, r.local_overwrite_inactive),
                max(r.multipole_reset_error, r.local_reset_error))
        end
    end

    return (; passed, max_mp, max_local, max_T, max_inactive, max_reset)
end

function main()
    cases = [
        (0, 0.0, "axis-aligned +z"),
        (4, 0.0, "axis-aligned +z"),
        (5, pi, "axis-aligned -z"),
        (6, pi / 7, "off-axis positive"),
        (7, -2pi / 5, "off-axis negative"),
    ]
    layouts = (Val(false), Val(true))
    results = [run_case(P, theta, label, layout) for (P, theta, label) in cases for layout in layouts]
    summary = write_summary(results)

    println("axis_swap_verify: $(summary.passed ? "PASS" : "FAIL")")
    println("summary: $(SUMMARY_PATH)")
    println("max_multipole_axis_swap_error: $(summary.max_mp)")
    println("max_local_axis_swap_error: $(summary.max_local)")
    println("max_T_reconstruction_error: $(summary.max_T)")
    println("max_inactive_channel_reset_error: $(summary.max_inactive)")
    println("max_reset_back_rotation_target_independence_error: $(summary.max_reset)")

    summary.passed || exit(1)
end

main()
