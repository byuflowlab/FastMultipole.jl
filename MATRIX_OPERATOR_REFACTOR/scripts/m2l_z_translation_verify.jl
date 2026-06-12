using FastMultipole
using Printf

const ATOL = 1.0e-12
const RTOL = 1.0e-12
const DATA_DIR = joinpath(@__DIR__, "..", "data", "m2l_z_translation")
const SUMMARY_PATH = joinpath(DATA_DIR, "verification_summary.md")

function deterministic_expansion(P::Int, ::Val{LH}) where {LH}
    weights = FastMultipole.initialize_expansion(P, Float64)
    ncoeff = size(weights, 3)
    components = LH ? 2 : 1

    for component in 1:components
        for i in 1:ncoeff
            base = 0.07 * component + 0.013 * i
            weights[1, component, i] = sin(base) + 0.25 * cos(2.0 * base)
            weights[2, component, i] = cos(1.3 * base) - 0.1 * sin(0.5 * base)
        end
    end

    return weights
end

function m2l_z_block(P::Int, m::Int, t)
    rho = inv(t)
    T = typeof(float(t))
    block = zeros(T, P - m + 1, P - m + 1)

    row_seed = one(T)
    for k in 1:(2m)
        row_seed *= k * rho
    end
    row_seed *= rho

    for n in m:P
        coeff = row_seed
        for np in m:P
            block[n - m + 1, np - m + 1] = coeff
            coeff *= (n + np + 1) * rho
        end
        n < P && (row_seed *= (n + m + 1) * rho)
    end

    return block
end

function matrix_translate_m2l_z!(out, source, t, P::Int, ::Val{LH}) where {LH}
    components = LH ? 2 : 1

    for m in 0:P
        block = m2l_z_block(P, m, t)
        for n in m:P
            i_out = FastMultipole.harmonic_index(n, m)
            for component in 1:components
                real_acc = zero(eltype(out))
                imag_acc = zero(eltype(out))
                for np in m:P
                    i_in = FastMultipole.harmonic_index(np, m)
                    k = block[n - m + 1, np - m + 1]
                    real_acc += k * source[1, component, i_in]
                    imag_acc += k * source[2, component, i_in]
                end
                out[1, component, i_out] = real_acc
                out[2, component, i_out] = imag_acc
            end
        end
    end

    return out
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

function untouched_inactive_error(out, sentinel, ::Val{LH}) where {LH}
    LH && return 0.0
    inactive = @view out[:, 2:2, :]
    return maximum(abs, inactive .- sentinel)
end

function run_case(P::Int, t::Float64, layout)
    source = deterministic_expansion(P, layout)
    expected = fill(-13.0, size(source))
    matrixed = fill(19.0, size(source))

    FastMultipole.translate_multipole_to_local_z!(expected, source, t, P, layout)
    matrix_translate_m2l_z!(matrixed, source, t, P, layout)

    max_abs, max_rel = active_errors(matrixed, expected, layout)
    inactive_error = untouched_inactive_error(matrixed, 19.0, layout)

    return (; P, t, layout = layout isa Val{true} ? "Val(true)" : "Val(false)",
        max_abs, max_rel, inactive_error)
end

function write_summary(results)
    mkpath(DATA_DIR)

    max_abs = maximum(r.max_abs for r in results)
    max_rel = maximum(r.max_rel for r in results)
    max_inactive = maximum(r.inactive_error for r in results)
    passed = max_abs <= ATOL && max_rel <= RTOL && max_inactive == 0.0

    open(SUMMARY_PATH, "w") do io
        println(io, "# M2L Z-Translation Verification Summary")
        println(io)
        println(io, "- Command: `julia --project=. MATRIX_OPERATOR_REFACTOR/scripts/m2l_z_translation_verify.jl`")
        println(io, "- Tolerance: `atol <= $(ATOL)`, `rtol <= $(RTOL)`")
        println(io, "- Status: `$(passed ? "PASS" : "FAIL")`")
        println(io, "- Max absolute error: `$(max_abs)`")
        println(io, "- Max relative error: `$(max_rel)`")
        println(io, "- Max inactive-channel overwrite error: `$(max_inactive)`")
        println(io)
        println(io, "| Layout | P | t | Max abs error | Max rel error | Inactive channel |")
        println(io, "| --- | ---: | ---: | ---: | ---: | ---: |")
        for r in results
            @printf(io, "| `%s` | %d | %.16g | %.6e | %.6e | %.6e |\n",
                r.layout, r.P, r.t, r.max_abs, r.max_rel, r.inactive_error)
        end
    end

    return passed, max_abs, max_rel, max_inactive
end

function main()
    cases = [(0, 1.75), (1, 2.25), (3, 4.5), (6, 8.0), (9, 13.0)]
    layouts = (Val(false), Val(true))
    results = [run_case(P, t, layout) for (P, t) in cases for layout in layouts]
    passed, max_abs, max_rel, max_inactive = write_summary(results)

    println("m2l_z_translation_verify: $(passed ? "PASS" : "FAIL")")
    println("summary: $(SUMMARY_PATH)")
    println("max_abs_error: $(max_abs)")
    println("max_rel_error: $(max_rel)")
    println("max_inactive_channel_error: $(max_inactive)")

    passed || exit(1)
end

main()
