using FastMultipole
using Printf

const ATOL = 1.0e-12
const DATA_DIR = joinpath(@__DIR__, "..", "data", "z_rotation")
const SUMMARY_PATH = joinpath(DATA_DIR, "verification_summary.md")

function deterministic_expansion(P::Int, ::Val{LH}) where {LH}
    weights = FastMultipole.initialize_expansion(P, Float64)
    ncoeff = size(weights, 3)
    components = LH ? 2 : 1

    for component in 1:components
        for i in 1:ncoeff
            base = 0.125 * component + 0.03125 * i
            weights[1, component, i] = sin(base) + cos(0.5 * base)
            weights[2, component, i] = cos(1.25 * base) - sin(0.75 * base)
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
            base = 0.2 * component + 0.017 * i
            weights[1, component, i] = 0.25 + sin(base)
            weights[2, component, i] = -0.5 + cos(1.7 * base)
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

function active_max_abs_diff(a, b, ::Val{LH}) where {LH}
    components = LH ? 2 : 1
    a_active = @view a[:, 1:components, :]
    b_active = @view b[:, 1:components, :]
    return maximum(abs, a_active .- b_active)
end

function max_m0_forward_error(rotated, source, P::Int, ::Val{LH}) where {LH}
    components = LH ? 2 : 1
    max_error = 0.0

    for n in 0:P
        i = FastMultipole.harmonic_index(n, 0)
        rotated_m0 = @view rotated[:, 1:components, i]
        source_m0 = @view source[:, 1:components, i]
        max_error = max(max_error, maximum(abs, rotated_m0 .- source_m0))
    end

    return max_error
end

function max_m0_back_error(back, target, source, P::Int, ::Val{LH}) where {LH}
    components = LH ? 2 : 1
    max_error = 0.0

    for n in 0:P
        i = FastMultipole.harmonic_index(n, 0)
        back_m0 = @view back[:, 1:components, i]
        target_m0 = @view target[:, 1:components, i]
        source_m0 = @view source[:, 1:components, i]
        expected = target_m0 .+ source_m0
        max_error = max(max_error, maximum(abs, back_m0 .- expected))
    end

    return max_error
end

function run_case(P::Int, phi::Float64, layout)
    source = deterministic_expansion(P, layout)
    C, S = phase_vectors(P, phi)

    expected_forward = fill(-17.0, size(source))
    fused_forward = fill(23.0, size(source))
    eimphis = zeros(Float64, 2, P + 1)
    FastMultipole.rotate_z!(expected_forward, source, eimphis, phi, P, layout)
    fused_rotate_z!(fused_forward, source, C, S, P, layout)

    target = deterministic_target(P, layout)
    expected_back = copy(target)
    fused_back = copy(target)
    FastMultipole.back_rotate_z!(expected_back, expected_forward, eimphis, P, layout)
    fused_back_rotate_z!(fused_back, expected_forward, C, S, P, layout)

    forward_error = active_max_abs_diff(fused_forward, expected_forward, layout)
    back_error = active_max_abs_diff(fused_back, expected_back, layout)
    m0_forward_error = max_m0_forward_error(fused_forward, source, P, layout)
    m0_back_error = max_m0_back_error(fused_back, target, expected_forward, P, layout)

    return (; P, phi, layout = layout isa Val{true} ? "Val(true)" : "Val(false)",
        forward_error, back_error, m0_forward_error, m0_back_error)
end

function write_summary(results)
    mkpath(DATA_DIR)

    max_forward = maximum(r.forward_error for r in results)
    max_back = maximum(r.back_error for r in results)
    max_m0_forward = maximum(r.m0_forward_error for r in results)
    max_m0_back = maximum(r.m0_back_error for r in results)
    passed = max(max_forward, max_back, max_m0_forward, max_m0_back) <= ATOL

    open(SUMMARY_PATH, "w") do io
        println(io, "# Z-Rotation Verification Summary")
        println(io)
        println(io, "- Command: `julia --project=. MATRIX_OPERATOR_REFACTOR/scripts/z_rotation_verify.jl`")
        println(io, "- Tolerance: `atol <= $(ATOL)`")
        println(io, "- Status: `$(passed ? "PASS" : "FAIL")`")
        println(io, "- Max forward error: `$(max_forward)`")
        println(io, "- Max back/inverse accumulation error: `$(max_back)`")
        println(io, "- Max `m = 0` forward identity error: `$(max_m0_forward)`")
        println(io, "- Max `m = 0` back accumulation error: `$(max_m0_back)`")
        println(io)
        println(io, "| Layout | P | phi | Forward max error | Back max error | m=0 forward | m=0 back |")
        println(io, "| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for r in results
            @printf(io, "| `%s` | %d | %.16g | %.6e | %.6e | %.6e | %.6e |\n",
                r.layout, r.P, r.phi, r.forward_error, r.back_error, r.m0_forward_error, r.m0_back_error)
        end
    end

    return passed, max_forward, max_back, max_m0_forward, max_m0_back
end

function main()
    cases = [(0, 0.0), (1, 0.25), (3, -1.125), (6, pi / 3), (9, 2.4)]
    layouts = (Val(false), Val(true))
    results = [run_case(P, phi, layout) for (P, phi) in cases for layout in layouts]
    passed, max_forward, max_back, max_m0_forward, max_m0_back = write_summary(results)

    println("z_rotation_verify: $(passed ? "PASS" : "FAIL")")
    println("summary: $(SUMMARY_PATH)")
    println("max_forward_error: $(max_forward)")
    println("max_back_error: $(max_back)")
    println("max_m0_forward_error: $(max_m0_forward)")
    println("max_m0_back_error: $(max_m0_back)")

    passed || exit(1)
end

main()
