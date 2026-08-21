using FastMultipole
using Printf

const ATOL = 1.0e-12
const RTOL = 1.0e-12
const DATA_DIR = joinpath(@__DIR__, "..", "data", "lamb_helmholtz_operator")
const SUMMARY_PATH = joinpath(DATA_DIR, "verification_summary.md")

function deterministic_expansion(P::Int)
    weights = FastMultipole.initialize_expansion(P, Float64)
    ncoeff = size(weights, 3)

    for component in 1:2
        for i in 1:ncoeff
            base = 0.11 * component + 0.017 * i
            weights[1, component, i] = sin(base) + 0.2 * cos(1.7 * base)
            weights[2, component, i] = cos(1.2 * base) - 0.15 * sin(0.4 * base)
        end
    end

    return weights
end

function operator_lh_multipole!(out, source, r, P::Int)
    out .= source

    for m in 0:P
        for n in max(m, 1):P
            i = FastMultipole.harmonic_index(n, m)
            chi_re = source[1, 2, i]
            chi_im = source[2, 2, i]

            a = r * m / (n + 1)
            out[1, 1, i] = source[1, 1, i] + a * chi_im
            out[2, 1, i] = source[2, 1, i] - a * chi_re

            b = r / n
            if m < n
                i_prev = FastMultipole.harmonic_index(n - 1, m)
                out[1, 2, i] = chi_re + b * source[1, 2, i_prev]
                out[2, 2, i] = chi_im + b * source[2, 2, i_prev]
            else
                out[1, 2, i] = chi_re
                out[2, 2, i] = chi_im
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

function max_errors(a, b)
    max_abs = 0.0
    max_rel = 0.0

    @inbounds for i in eachindex(a, b)
        err = abs(a[i] - b[i])
        scale = max(abs(b[i]), eps(Float64))
        max_abs = max(max_abs, err)
        max_rel = max(max_rel, err / scale)
    end

    return max_abs, max_rel
end

function run_case(P::Int, r::Float64)
    source = deterministic_expansion(P)

    expected_m = copy(source)
    operator_m = similar(source)
    FastMultipole.transform_lamb_helmholtz_multipole!(expected_m, r, P)
    operator_lh_multipole!(operator_m, source, r, P)
    m_abs, m_rel = max_errors(operator_m, expected_m)

    expected_l = copy(source)
    operator_l = similar(source)
    FastMultipole.transform_lamb_helmholtz_local!(expected_l, r, P)
    operator_lh_local!(operator_l, source, r, P)
    l_abs, l_rel = max_errors(operator_l, expected_l)

    return (; P, r, m_abs, m_rel, l_abs, l_rel)
end

function write_summary(results)
    mkpath(DATA_DIR)

    max_abs = maximum(max(r.m_abs, r.l_abs) for r in results)
    max_rel = maximum(max(r.m_rel, r.l_rel) for r in results)
    passed = max_abs <= ATOL && max_rel <= RTOL

    open(SUMMARY_PATH, "w") do io
        println(io, "# Lamb-Helmholtz Operator Verification Summary")
        println(io)
        println(io, "- Command: `julia --project=. MATRIX_OPERATOR_REFACTOR/scripts/lamb_helmholtz_operator_verify.jl`")
        println(io, "- Tolerance: `atol <= $(ATOL)`, `rtol <= $(RTOL)`")
        println(io, "- Status: `$(passed ? "PASS" : "FAIL")`")
        println(io, "- Max absolute error: `$(max_abs)`")
        println(io, "- Max relative error: `$(max_rel)`")
        println(io)
        println(io, "| P | r | M transform abs | M transform rel | L transform abs | L transform rel |")
        println(io, "| ---: | ---: | ---: | ---: | ---: | ---: |")
        for r in results
            @printf(io, "| %d | %.16g | %.6e | %.6e | %.6e | %.6e |\n",
                r.P, r.r, r.m_abs, r.m_rel, r.l_abs, r.l_rel)
        end
    end

    return passed, max_abs, max_rel
end

function main()
    cases = [(0, 0.75), (1, 1.1), (3, 2.25), (6, 4.0), (9, 7.5)]
    results = [run_case(P, r) for (P, r) in cases]
    passed, max_abs, max_rel = write_summary(results)

    println("lamb_helmholtz_operator_verify: $(passed ? "PASS" : "FAIL")")
    println("summary: $(SUMMARY_PATH)")
    println("max_abs_error: $(max_abs)")
    println("max_rel_error: $(max_rel)")

    passed || exit(1)
end

main()
