using Printf

const ORDERS = (0, 1, 3, 6, 9)
const BATCH_COUNTS = (1, 2, 5)
const DATA_DIR = joinpath(@__DIR__, "..", "data", "coefficient_buffer_layout")
const SUMMARY_PATH = joinpath(DATA_DIR, "verification_summary.md")

ncomplex(P::Int) = div((P + 1) * (P + 2), 2)
complex_basis_dof(P::Int) = 2 * ncomplex(P)
harmonic_index(n::Int, m::Int) = div(n * (n + 1), 2) + m + 1
basis_index(n::Int, m::Int, reim::Int) = 2 * (harmonic_index(n, m) - 1) + reim

# Independent cross-check: the script's harmonic_index must agree with the
# production FastMultipole.harmonic_index so this spec cannot silently drift from
# the code it maps to. The import is guarded at top level so the script still
# runs (recording the check as skipped) when the package is not loadable.
const PROD_HARMONIC_INDEX = try
    @eval import FastMultipole
    getfield(FastMultipole, :harmonic_index)
catch
    nothing
end

function production_harmonic_index_check()
    PROD_HARMONIC_INDEX === nothing && return (; available = false, ok = false)
    ok = true
    for P in ORDERS, n in 0:P, m in 0:n
        # invokelatest avoids world-age errors against the freshly imported method
        ok &= Base.invokelatest(PROD_HARMONIC_INDEX, n, m) == harmonic_index(n, m)
    end
    return (; available = true, ok)
end

# Independent cross-check that the production flat_basis_index (task 017) agrees
# with this spec's basis_index, so the implemented buffer layout cannot drift from
# the approved layout this script encodes.
const PROD_FLAT_BASIS_INDEX = try
    @eval import FastMultipole
    getfield(FastMultipole, :flat_basis_index)
catch
    nothing
end

function production_flat_basis_index_check()
    PROD_FLAT_BASIS_INDEX === nothing && return (; available = false, ok = false)
    ok = true
    for P in ORDERS, n in 0:P, m in 0:n, reim in 1:2
        ok &= Base.invokelatest(PROD_FLAT_BASIS_INDEX, n, m, reim) == basis_index(n, m, reim)
    end
    return (; available = true, ok)
end

nreal(P::Int) = (P + 1)^2
mode_index(n::Int, ::Val{:zero}) = n^2 + 1
mode_index(n::Int, m::Int, ::Val{:cos}) = n^2 + 2m
mode_index(n::Int, m::Int, ::Val{:sin}) = n^2 + 2m + 1

active_channels(::Val{LH}) where {LH} = LH ? 2 : 1
layout_label(::Val{LH}) where {LH} = LH ? "Val(true)" : "Val(false)"

function deterministic_weights(P::Int, channels::Int, batch::Int)
    weights = Array{Float64}(undef, 2, channels, ncomplex(P))
    @inbounds for reim in 1:2, channel in 1:channels, harmonic in 1:ncomplex(P)
        weights[reim, channel, harmonic] =
            1000.0 * batch + 100.0 * channel + 10.0 * reim + harmonic / 1000.0
    end
    return weights
end

function pack_complex!(buffer, weights, batch::Int, P::Int, channels::Int)
    @inbounds for channel in 1:channels, n in 0:P, m in 0:n, reim in 1:2
        buffer[basis_index(n, m, reim), batch, channel] =
            weights[reim, channel, harmonic_index(n, m)]
    end
    return buffer
end

function unpack_complex!(weights, buffer, batch::Int, P::Int, channels::Int)
    @inbounds for channel in 1:channels, n in 0:P, m in 0:n, reim in 1:2
        weights[reim, channel, harmonic_index(n, m)] =
            buffer[basis_index(n, m, reim), batch, channel]
    end
    return weights
end

function verify_complex_case(P::Int, batch_count::Int, layout)
    channels = active_channels(layout)
    basis_dof = complex_basis_dof(P)
    buffer = Array{Float64}(undef, basis_dof, batch_count, channels)

    max_error = 0.0
    for batch in 1:batch_count
        weights = deterministic_weights(P, channels, batch)
        pack_complex!(buffer, weights, batch, P, channels)
        roundtrip = fill(NaN, size(weights))
        unpack_complex!(roundtrip, buffer, batch, P, channels)
        max_error = max(max_error, maximum(abs.(roundtrip .- weights)))
    end

    expected_indices = collect(1:basis_dof)
    actual_indices = [basis_index(n, m, reim) for n in 0:P for m in 0:n for reim in 1:2]
    indices_ok = sort(actual_indices) == expected_indices && length(unique(actual_indices)) == basis_dof

    slabs_dense = true
    slab_stride_1 = Int[]
    slab_stride_2 = Int[]
    for channel in 1:channels
        slab = @view buffer[:, :, channel]
        push!(slab_stride_1, stride(slab, 1))
        push!(slab_stride_2, stride(slab, 2))
        slabs_dense &= stride(slab, 1) == 1 && stride(slab, 2) == basis_dof
    end

    return (;
        P,
        layout = layout_label(layout),
        batch_count,
        channels,
        basis_dof,
        max_error,
        indices_ok,
        slabs_dense,
        slab_stride_1 = join(slab_stride_1, ","),
        slab_stride_2 = join(slab_stride_2, ","),
    )
end

function real_indices(P::Int)
    indices = Int[]
    for n in 0:P
        push!(indices, mode_index(n, Val(:zero)))
        for m in 1:n
            push!(indices, mode_index(n, m, Val(:cos)))
            push!(indices, mode_index(n, m, Val(:sin)))
        end
    end
    return indices
end

function verify_real_case(P::Int)
    basis_dof = nreal(P)
    indices = real_indices(P)
    contiguous_unique = sort(indices) == collect(1:basis_dof) && length(unique(indices)) == basis_dof
    degree_blocks_ok = true
    for n in 0:P
        block = Int[mode_index(n, Val(:zero))]
        for m in 1:n
            push!(block, mode_index(n, m, Val(:cos)))
            push!(block, mode_index(n, m, Val(:sin)))
        end
        degree_blocks_ok &= block == collect(n^2 + 1:(n + 1)^2)
    end
    return (; P, basis_dof, contiguous_unique, degree_blocks_ok)
end

function write_summary(complex_results, real_results, prod_check, flat_check)
    mkpath(DATA_DIR)

    max_roundtrip_error = maximum(r.max_error for r in complex_results)
    complex_passed = all(r.max_error == 0.0 && r.indices_ok && r.slabs_dense for r in complex_results)
    real_passed = all(r.contiguous_unique && r.degree_blocks_ok for r in real_results)
    # A failed production cross-check fails the run; an unavailable package is
    # recorded but does not fail (the script can run standalone).
    prod_passed = !prod_check.available || prod_check.ok
    flat_passed = !flat_check.available || flat_check.ok
    passed = complex_passed && real_passed && prod_passed && flat_passed
    prod_status = !prod_check.available ? "SKIPPED (FastMultipole not loadable)" :
        (prod_check.ok ? "PASS" : "FAIL")
    flat_status = !flat_check.available ? "SKIPPED (FastMultipole not loadable)" :
        (flat_check.ok ? "PASS" : "FAIL")

    open(SUMMARY_PATH, "w") do io
        println(io, "# Coefficient Buffer Layout Verification Summary")
        println(io)
        println(io, "- Command: `julia --project=. MATRIX_OPERATOR_REFACTOR/scripts/coefficient_buffer_layout_verify.jl`")
        println(io, "- Status: `$(passed ? "PASS" : "FAIL")`")
        println(io, "- Expansion orders: `$(join(ORDERS, "`, `"))`")
        println(io, "- Layouts: `Val(false)`, `Val(true)`")
        println(io, "- Batch counts: `$(join(BATCH_COUNTS, "`, `"))`")
        println(io, "- Max complex legacy/native round-trip error: `$(max_roundtrip_error)`")
        println(io, "- Fixed-channel slab requirement: `stride(view, 1) == 1`, `stride(view, 2) == basis_dof`")
        println(io, "- Production `harmonic_index` cross-check: `$(prod_status)`")
        println(io, "- Production `flat_basis_index` cross-check: `$(flat_status)`")
        println(io)
        println(io, "## Compressed Complex Cases")
        println(io)
        println(io, "| P | Layout | Batch count | Channels | Basis dof | Max round-trip error | Indices contiguous/unique | Slabs dense | Slab stride 1 | Slab stride 2 |")
        println(io, "| ---: | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |")
        for r in complex_results
            @printf(io, "| %d | `%s` | %d | %d | %d | %.1f | `%s` | `%s` | `%s` | `%s` |\n",
                r.P, r.layout, r.batch_count, r.channels, r.basis_dof, r.max_error,
                r.indices_ok, r.slabs_dense, r.slab_stride_1, r.slab_stride_2)
        end
        println(io)
        println(io, "## Real Basis Cases")
        println(io)
        println(io, "| P | Basis dof | Indices contiguous/unique | Degree blocks contiguous |")
        println(io, "| ---: | ---: | --- | --- |")
        for r in real_results
            @printf(io, "| %d | %d | `%s` | `%s` |\n",
                r.P, r.basis_dof, r.contiguous_unique, r.degree_blocks_ok)
        end
    end

    return (; passed, max_roundtrip_error)
end

function main()
    complex_results = [
        verify_complex_case(P, batch_count, layout)
        for P in ORDERS
        for layout in (Val(false), Val(true))
        for batch_count in BATCH_COUNTS
    ]
    real_results = [verify_real_case(P) for P in ORDERS]
    prod_check = production_harmonic_index_check()
    flat_check = production_flat_basis_index_check()
    summary = write_summary(complex_results, real_results, prod_check, flat_check)

    println("coefficient_buffer_layout_verify: $(summary.passed ? "PASS" : "FAIL")")
    println("summary: $(SUMMARY_PATH)")
    println("max_complex_roundtrip_error: $(summary.max_roundtrip_error)")

    summary.passed || exit(1)
end

main()
