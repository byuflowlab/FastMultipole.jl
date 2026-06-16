using FastMultipole
using LinearAlgebra
using Printf
using StaticArrays
using Statistics

const P_REF = 16
const P_VALUES = collect(4:2:12)
const DELTAS = (-1, 0, 1, 2)
const DATA_DIR = joinpath(@__DIR__, "..", "data", "lamb_helmholtz_accuracy_order")
const SUMMARY_PATH = joinpath(DATA_DIR, "verification_summary.md")

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
        eimphis = zeros(Float64, 2, P + 2),
        tmp1 = FastMultipole.initialize_expansion(P, Float64),
        tmp2 = FastMultipole.initialize_expansion(P, Float64),
        tmp3 = FastMultipole.initialize_expansion(P, Float64),
    )
end

function deterministic_source(P::Int, cloud_id::Int)
    source = FastMultipole.initialize_expansion(P, Float64)
    radius = 0.16 + 0.025 * cloud_id

    for n in 0:P
        radial = radius^n / float(factorial(big(n)))
        for m in 0:n
            i = FastMultipole.harmonic_index(n, m)
            phase = 0.37 * cloud_id + 0.41 * n + 0.23 * m
            source[1, 1, i] = radial * (sin(phase) + 0.25 * cos(1.7 * phase))
            source[2, 1, i] = radial * (cos(0.9 * phase) - 0.15 * sin(2.1 * phase))
            source[1, 2, i] = radial * (0.85 * cos(1.2 * phase) + 0.31 * sin(0.6 * phase))
            source[2, 2, i] = radial * (0.80 * sin(1.4 * phase) - 0.19 * cos(0.8 * phase))
        end
    end

    return source
end

function zero_channel_tail!(weights, component::Int, Pkeep::Int, Pmax::Int)
    Pkeep = min(Pkeep, Pmax)
    for n in max(Pkeep + 1, 0):Pmax
        for m in 0:n
            i = FastMultipole.harmonic_index(n, m)
            weights[1, component, i] = 0.0
            weights[2, component, i] = 0.0
        end
    end
    return weights
end

function truncate_channels!(weights, Pphi::Int, Pchi::Int, Pmax::Int)
    zero_channel_tail!(weights, 1, Pphi, Pmax)
    zero_channel_tail!(weights, 2, Pchi, Pmax)
    return weights
end

function production_m2l(source, source_center, target_center, P::Int)
    local_weights = FastMultipole.initialize_expansion(P, Float64)
    cache = m2l_prealloc(P)
    FastMultipole.multipole_to_local!(
        local_weights, branch(target_center), source, branch(source_center),
        cache.tmp1, cache.tmp2, cache.tmp3, cache.Ts, cache.eimphis,
        cache.zeta, cache.eta, cache.Hs_pi2, FastMultipole.M̃, FastMultipole.L̃,
        P, Val(true))
    return local_weights
end

function evaluate_gradient(local_weights, target_offset, P::Int)
    harmonics = FastMultipole.initialize_harmonics(P)
    gradient_n_m = FastMultipole.initialize_gradient_n_m(P)
    _, grad, _ = FastMultipole.evaluate_local(
        target_offset, harmonics, gradient_n_m, local_weights, P, Val(true),
        FastMultipole.DerivativesSwitch(false, true, false))
    return SVector{3,Float64}(grad)
end

function policy_gradient(source_ref, source_center, target_center, target_offset, Pphi::Int, Pchi::Int)
    source = copy(source_ref)
    truncate_channels!(source, Pphi, Pchi, P_REF)
    local_weights = production_m2l(source, source_center, target_center, P_REF)
    truncate_channels!(local_weights, Pphi, Pchi, P_REF)
    return evaluate_gradient(local_weights, target_offset, P_REF)
end

function rms(values)
    return sqrt(mean(abs2, values))
end

function slope(rows)
    xs = Float64[r.P for r in rows]
    ys = log.(max.(Float64[r.rms for r in rows], eps(Float64)))
    xbar = mean(xs)
    ybar = mean(ys)
    denom = sum((x - xbar)^2 for x in xs)
    return sum((x - xbar) * (y - ybar) for (x, y) in zip(xs, ys)) / denom
end

function aggregate(rows)
    grouped = Dict{Tuple{Symbol,Int},Vector{Float64}}()
    for row in rows
        push!(get!(grouped, (row.kind, row.P), Float64[]), row.err)
    end

    stats = NamedTuple[]
    for ((kind, P), errs) in sort(collect(grouped); by = x -> (String(x[1][1]), x[1][2]))
        push!(stats, (;
            kind,
            P,
            maxerr = maximum(errs),
            rms = rms(errs),
            median = median(errs),
            count = length(errs),
        ))
    end
    return stats
end

function run_study()
    rows = NamedTuple[]
    scalar_unchanged = true

    source_centers = (
        SVector{3}(0.00, 0.00, 0.00),
        SVector{3}(0.12, -0.07, 0.05),
    )
    directions = (
        normalize(SVector{3}(1.0, 0.35, 0.20)),
        normalize(SVector{3}(-0.45, 1.0, 0.55)),
        normalize(SVector{3}(0.30, -0.60, 1.0)),
    )
    separations = (2.7, 3.6, 4.8)
    target_offsets = (
        SVector{3}(0.015, -0.010, 0.012),
        SVector{3}(-0.020, 0.014, -0.006),
    )

    for cloud_id in 1:3
        source_ref = deterministic_source(P_REF, cloud_id)
        for source_center in source_centers, dir in directions, sep in separations
            target_center = source_center + sep * dir
            local_ref = production_m2l(source_ref, source_center, target_center, P_REF)
            for target_offset in target_offsets
                ref = evaluate_gradient(local_ref, target_offset, P_REF)

                for P in P_VALUES
                    phi_only_local = copy(local_ref)
                    truncate_channels!(phi_only_local, P, P_REF, P_REF)
                    phi_only = evaluate_gradient(phi_only_local, target_offset, P_REF)
                    push!(rows, (; kind = :phi_only, P, err = norm(phi_only - ref)))

                    chi_only_local = copy(local_ref)
                    truncate_channels!(chi_only_local, P_REF, P, P_REF)
                    chi_only = evaluate_gradient(chi_only_local, target_offset, P_REF)
                    push!(rows, (; kind = :chi_only, P, err = norm(chi_only - ref)))

                    for delta in DELTAS
                        Pchi = max(0, P + delta)
                        paired = policy_gradient(source_ref, source_center, target_center, target_offset, P, Pchi)
                        push!(rows, (; kind = Symbol("delta_", delta), P, err = norm(paired - ref)))
                    end
                end
            end
        end
    end

    scalar_source = copy(deterministic_source(P_REF, 1))
    scalar_source[:, 2, :] .= 0.0
    scalar_center = SVector{3}(0.0, 0.0, 0.0)
    scalar_target = SVector{3}(3.0, 1.0, 0.5)
    scalar_local_false = FastMultipole.initialize_expansion(P_REF, Float64)
    cache = m2l_prealloc(P_REF)
    FastMultipole.multipole_to_local!(
        scalar_local_false, branch(scalar_target), scalar_source, branch(scalar_center),
        cache.tmp1, cache.tmp2, cache.tmp3, cache.Ts, cache.eimphis,
        cache.zeta, cache.eta, cache.Hs_pi2, FastMultipole.M̃, FastMultipole.L̃,
        P_REF, Val(false))
    scalar_local_true = production_m2l(scalar_source, scalar_center, scalar_target, P_REF)
    scalar_unchanged = maximum(abs, scalar_local_false[:, 1, :] .- scalar_local_true[:, 1, :]) <= 1.0e-11

    return aggregate(rows), scalar_unchanged
end

function write_summary(stats, scalar_unchanged)
    mkpath(DATA_DIR)
    by_kind = Dict(kind => [row for row in stats if row.kind == kind] for kind in unique(row.kind for row in stats))
    slopes = Dict(kind => slope(rows) for (kind, rows) in by_kind)
    lastP = maximum(row.P for row in stats)
    at_last(kind) = only(row for row in stats if row.kind == kind && row.P == lastP)

    chosen = :delta_1
    delta0 = at_last(:delta_0)
    delta1 = at_last(:delta_1)
    delta2 = at_last(:delta_2)
    chi = at_last(:chi_only)

    pass = scalar_unchanged &&
        delta1.rms < 0.70 * delta0.rms &&
        delta2.rms > 0.75 * delta1.rms &&
        slopes[chosen] < slopes[:delta_0] &&
        slopes[chosen] < -0.6

    open(SUMMARY_PATH, "w") do io
        println(io, "# Lamb-Helmholtz Accuracy Order Verification Summary")
        println(io)
        println(io, "- Command: `julia --project=. MATRIX_OPERATOR_REFACTOR/scripts/lamb_helmholtz_accuracy_order_verify.jl`")
        println(io, "- Status: `$(pass ? "PASS" : "FAIL")`")
        println(io, "- Final rule: `P_chi = P_phi + 1` for `Val(true)` M2L/evaluation; `Val(false)` remains single-order `P`.")
        println(io, "- Tested offsets: `P_chi = P_phi + delta`, delta in `$(collect(DELTAS))`.")
        println(io, "- Configurations: 3 deterministic source coefficient clouds, 2 source centers, 3 directions, 3 separation ratios, 2 target offsets.")
        println(io, "- Scalar `Val(false)` unchanged behavior: `$(scalar_unchanged ? "PASS" : "FAIL")`")
        println(io)
        println(io, "## Convergence Slopes")
        println(io)
        println(io, "Slopes are least-squares slopes of `log(rms error)` versus `P_phi`; more negative is faster convergence.")
        println(io)
        println(io, "| Case | Slope |")
        println(io, "| --- | ---: |")
        for kind in sort(collect(keys(slopes)); by = String)
            @printf(io, "| `%s` | %.6f |\n", String(kind), slopes[kind])
        end
        println(io)
        println(io, "## Aggregate Errors")
        println(io)
        println(io, "| Case | P_phi | Count | Max error | RMS error | Median error |")
        println(io, "| --- | ---: | ---: | ---: | ---: | ---: |")
        for row in stats
            @printf(io, "| `%s` | %d | %d | %.6e | %.6e | %.6e |\n",
                String(row.kind), row.P, row.count, row.maxerr, row.rms, row.median)
        end
        println(io)
        println(io, "## Decision Checks")
        println(io)
        @printf(io, "- At `P_phi = %d`, `delta_1` RMS / `delta_0` RMS = `%.6e`.\n", lastP, delta1.rms / delta0.rms)
        @printf(io, "- At `P_phi = %d`, `delta_2` RMS / `delta_1` RMS = `%.6e`.\n", lastP, delta2.rms / delta1.rms)
        @printf(io, "- `delta_1` slope improvement over `delta_0`: `%.6e`.\n", slopes[:delta_0] - slopes[:delta_1])
        @printf(io, "- At `P_phi = %d`, isolated `chi_only(P_chi=P_phi)` RMS = `%.6e`.\n", lastP, chi.rms)
        println(io, "- The `phi_only` and `chi_only` rows are diagnostic isolated-tail rows; the pass decision is based on paired M2L policies.")
    end

    return pass
end

function main()
    stats, scalar_unchanged = run_study()
    passed = write_summary(stats, scalar_unchanged)
    println("lamb_helmholtz_accuracy_order_verify: $(passed ? "PASS" : "FAIL")")
    println("summary: $(SUMMARY_PATH)")
    println("scalar_val_false_unchanged: $(scalar_unchanged)")
    passed || exit(1)
end

main()
