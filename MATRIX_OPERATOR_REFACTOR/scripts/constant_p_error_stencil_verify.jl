using Printf

const DATA_DIR = normpath(joinpath(@__DIR__, "..", "data", "constant_p_error_stencil"))
const SUMMARY_PATH = joinpath(DATA_DIR, "verification_summary.md")

const Offset = NTuple{3,Int}

cell_radius(w::Float64) = w * sqrt(3.0)
offset_distance(w::Float64, d::Offset) = 2.0 * w * sqrt(sum(x -> x * x, d))
separation_ratio(d::Offset) = 2.0 * sqrt(sum(x -> x * x, d)) / sqrt(3.0)

function offset_domain(max_abs_offset::Int)
    return Offset[(i, j, k) for i in -max_abs_offset:max_abs_offset
        for j in -max_abs_offset:max_abs_offset
        for k in -max_abs_offset:max_abs_offset]
end

function scalar_bound(P::Int, d::Offset, A::Float64;
        w::Float64 = 1.0,
        production_normalized::Bool = false)
    rho = cell_radius(w)
    c = separation_ratio(d)
    c > 2.0 || return Inf
    bound = 2.0 * A / (rho * (c - 2.0)) * (1.0 / (c - 1.0))^(P + 1)
    production_normalized && (bound /= 4.0 * pi)
    return bound
end

function lh_bound(P::Int, d::Offset, A_phi::Float64, A_chi::Float64;
        w::Float64 = 1.0,
        production_normalized::Bool = false)
    B_phi = scalar_bound(P, d, A_phi; w, production_normalized)
    B_chi = scalar_bound(P, d, A_chi; w, production_normalized)
    isfinite(B_phi) && isfinite(B_chi) || return Inf
    R = offset_distance(w, d)
    return B_phi + (1.0 + 2.0 * R) * B_chi
end

accept_offset(::Val{false}, P::Int, d::Offset, epsilon::Float64;
        A::Float64,
        w::Float64 = 1.0,
        production_normalized::Bool = false) =
    (B = scalar_bound(P, d, A; w, production_normalized); isfinite(B) && B <= epsilon)

accept_offset(::Val{true}, P::Int, d::Offset, epsilon::Float64;
        A_phi::Float64,
        A_chi::Float64,
        w::Float64 = 1.0,
        production_normalized::Bool = false) =
    (B = lh_bound(P, d, A_phi, A_chi; w, production_normalized); isfinite(B) && B <= epsilon)

function scalar_stencil(P::Int, epsilon::Float64, max_abs_offset::Int;
        A::Float64,
        w::Float64 = 1.0,
        production_normalized::Bool = false)
    return Set(d for d in offset_domain(max_abs_offset)
        if accept_offset(Val(false), P, d, epsilon; A, w, production_normalized))
end

function lh_stencil(P::Int, epsilon::Float64, max_abs_offset::Int;
        A_phi::Float64,
        A_chi::Float64,
        w::Float64 = 1.0,
        production_normalized::Bool = false)
    return Set(d for d in offset_domain(max_abs_offset)
        if accept_offset(Val(true), P, d, epsilon; A_phi, A_chi, w, production_normalized))
end

function assert_condition(condition::Bool, message::String)
    condition || error(message)
end

function verify_geometry()
    w = 0.25
    samples = Offset[(0, 0, 0), (1, 0, 0), (1, 1, 0), (2, -1, 3)]
    for d in samples
        R = offset_distance(w, d)
        rho = cell_radius(w)
        expected_c = iszero(R) ? 0.0 : R / rho
        assert_condition(isapprox(separation_ratio(d), expected_c; atol = 8eps(Float64), rtol = 0.0),
            "separation-ratio mismatch for offset $(d)")
    end
    return true
end

function verify_rejection()
    P = 5
    A = 1.0
    rejected = [d for d in offset_domain(2) if separation_ratio(d) <= 2.0]
    assert_condition(!isempty(rejected), "expected at least one c <= 2 offset")
    for d in rejected
        assert_condition(!isfinite(scalar_bound(P, d, A)), "scalar bound should reject $(d)")
        assert_condition(!accept_offset(Val(false), P, d, 1.0; A), "Val(false) accepted $(d)")
        assert_condition(!accept_offset(Val(true), P, d, 1.0; A_phi = A, A_chi = A),
            "Val(true) accepted $(d)")
    end
    return true
end

function verify_stencil_agreement()
    max_abs_offset = 6
    P = 4
    epsilon = 1.0e-3
    A = 1.0
    A_phi = 0.8
    A_chi = 0.2

    scalar_expected = Set(d for d in offset_domain(max_abs_offset)
        if (B = scalar_bound(P, d, A); isfinite(B) && B <= epsilon))
    scalar_generated = scalar_stencil(P, epsilon, max_abs_offset; A)
    assert_condition(scalar_expected == scalar_generated,
        "scalar generated stencil did not match analytic predicate")

    lh_expected = Set(d for d in offset_domain(max_abs_offset)
        if (B = lh_bound(P, d, A_phi, A_chi); isfinite(B) && B <= epsilon))
    lh_generated = lh_stencil(P, epsilon, max_abs_offset; A_phi, A_chi)
    assert_condition(lh_expected == lh_generated,
        "Lamb-Helmholtz generated stencil did not match analytic predicate")

    prod_epsilon = epsilon / (4.0 * pi)
    scalar_prod = scalar_stencil(P, prod_epsilon, max_abs_offset; A,
        production_normalized = true)
    assert_condition(scalar_prod == scalar_generated,
        "production-normalized scalar stencil should match scaled analytic tolerance")

    return (;
        scalar_count = length(scalar_generated),
        lh_count = length(lh_generated),
        max_abs_offset,
        P,
        epsilon,
    )
end

function verify_monotonicity()
    max_abs_offset = 7
    A = 1.0
    loose_epsilon = 5.0e-4
    tight_epsilon = 1.0e-4
    low_P = 3
    high_P = 6

    base = scalar_stencil(low_P, tight_epsilon, max_abs_offset; A)
    higher_P = scalar_stencil(high_P, tight_epsilon, max_abs_offset; A)
    looser = scalar_stencil(low_P, loose_epsilon, max_abs_offset; A)

    assert_condition(issubset(base, higher_P), "larger P shrank scalar stencil")
    assert_condition(issubset(base, looser), "looser epsilon shrank scalar stencil")

    base_lh = lh_stencil(low_P, tight_epsilon, max_abs_offset; A_phi = 0.7, A_chi = 0.3)
    higher_P_lh = lh_stencil(high_P, tight_epsilon, max_abs_offset; A_phi = 0.7, A_chi = 0.3)
    looser_lh = lh_stencil(low_P, loose_epsilon, max_abs_offset; A_phi = 0.7, A_chi = 0.3)

    assert_condition(issubset(base_lh, higher_P_lh), "larger P shrank LH stencil")
    assert_condition(issubset(base_lh, looser_lh), "looser epsilon shrank LH stencil")

    return (;
        scalar_base = length(base),
        scalar_higher_P = length(higher_P),
        scalar_looser = length(looser),
        lh_base = length(base_lh),
        lh_higher_P = length(higher_P_lh),
        lh_looser = length(looser_lh),
    )
end

function write_summary(agreement, monotonicity)
    mkpath(DATA_DIR)
    open(SUMMARY_PATH, "w") do io
        println(io, "# Constant-`P` Error Stencil Verification Summary")
        println(io)
        println(io, "- Command: `julia --project=. MATRIX_OPERATOR_REFACTOR/scripts/constant_p_error_stencil_verify.jl`")
        println(io, "- Status: `PASS`")
        println(io, "- Geometry mapping: `PASS`")
        println(io, "- `c <= 2` rejection: `PASS`")
        println(io, "- Analytic/generated stencil agreement: `PASS`")
        println(io, "- Monotonicity: `PASS`")
        println(io, "- Paths covered: `Val(false)`, `Val(true)`")
        println(io)
        println(io, "## Agreement Case")
        println(io)
        println(io, "| P | Epsilon | Max abs offset | Scalar accepted | LH accepted |")
        println(io, "| ---: | ---: | ---: | ---: | ---: |")
        @printf(io, "| %d | %.6e | %d | %d | %d |\n",
            agreement.P, agreement.epsilon, agreement.max_abs_offset,
            agreement.scalar_count, agreement.lh_count)
        println(io)
        println(io, "## Monotonicity Case")
        println(io)
        println(io, "| Path | Base accepted | Larger P accepted | Looser epsilon accepted |")
        println(io, "| --- | ---: | ---: | ---: |")
        @printf(io, "| `Val(false)` | %d | %d | %d |\n",
            monotonicity.scalar_base, monotonicity.scalar_higher_P,
            monotonicity.scalar_looser)
        @printf(io, "| `Val(true)` | %d | %d | %d |\n",
            monotonicity.lh_base, monotonicity.lh_higher_P,
            monotonicity.lh_looser)
    end
    return true
end

function main()
    verify_geometry()
    verify_rejection()
    agreement = verify_stencil_agreement()
    monotonicity = verify_monotonicity()
    write_summary(agreement, monotonicity)
    println("constant_p_error_stencil_verify: PASS")
    println("summary: $(SUMMARY_PATH)")
end

main()
