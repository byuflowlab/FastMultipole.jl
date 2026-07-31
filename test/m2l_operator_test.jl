# Parity tests for the full M2L operator pipeline (Matrix Operator Refactor, task 014).
#
# The two swappable whole-M2L variants (MaterializedYRotationM2L and
# FactoredRotationM2L) are composed from the task 010-013c stages and validated
# side-by-side against the production multipole_to_local! (the side-by-side,
# parity-only first pass per the 008b re-plan). The pipeline runs uniformly at
# P_active. For Val(false), P_active = P and every row must match production at P.
# For Val(true), φ is physical only through P_phi while χ is carried at
# P_active = P_phi + 1 (theory/lamb-helmholtz-accuracy-order.md): φ above P_phi is
# zero-padded by the pipeline, so the result equals a production run at P_active on
# a source whose φ padding is zeroed, with no nonphysical φ output above P_phi.

@testset "M2L operator pipeline (task 014)" begin

# native flat coefficient buffer helpers (task 017): operators now consume
# FlatCoefficientBuffer; the parity references stay in the legacy [2,2,nh] layout.
isdefined(@__MODULE__, :to_flat_buffer) || include("flat_buffer_helpers.jl")

M2L_OFFSETS = (
    SVector{3}(0.0, 0.0, 3.0),       # +z axis  (θ = 0)
    SVector{3}(3.2, 0.0, 0.0),       # +x axis  (θ = π/2, ϕ = 0)
    SVector{3}(0.0, 2.7, 0.0),       # +y axis  (θ = π/2, ϕ = π/2)
    SVector{3}(1.5, -2.0, 2.5),      # general diagonal
    SVector{3}(-2.3, 1.1, -3.4),     # general, mixed signs
)

# Random *physical* coefficients [2,2,nh]: m = 0 rows are real (zero imaginary),
# as every real multipole/local expansion is. FactoredRotationM2L's rank-1 mode
# decomposition (task 013c) reproduces production only on this physical subspace;
# MaterializedYRotationM2L is a full linear operator (exact for any input).
function m2l_random_source(P, TF, ::Val{LH}) where LH
    src = FastMultipole.initialize_expansion(P, TF)
    for n in 0:P
        for m in 0:n
            i = FastMultipole.harmonic_index(n, m)
            src[1, 1, i] = randn(TF)
            src[2, 1, i] = m == 0 ? zero(TF) : randn(TF)
            if LH
                src[1, 2, i] = randn(TF)
                src[2, 2, i] = m == 0 ? zero(TF) : randn(TF)
            end
        end
    end
    return src
end

# Val(true) source carried in a P_active-sized buffer: φ physical through P_phi
# with a NONZERO sentinel in the φ padding rows (degree > P_phi) to prove the
# pipeline ignores/zeroes it; χ physical through P_active.
function m2l_lh_sentinel_source(P_phi, P_active, TF)
    src = FastMultipole.initialize_expansion(P_active, TF)
    for n in 0:P_active
        for m in 0:n
            i = FastMultipole.harmonic_index(n, m)
            src[1, 2, i] = randn(TF)
            src[2, 2, i] = m == 0 ? zero(TF) : randn(TF)
            if n <= P_phi
                src[1, 1, i] = randn(TF)
                src[2, 1, i] = m == 0 ? zero(TF) : randn(TF)
            else
                # distinctive sentinel; physical-form (m = 0 imaginary zero)
                src[1, 1, i] = TF(10) + randn(TF)
                src[2, 1, i] = m == 0 ? zero(TF) : TF(10) + randn(TF)
            end
        end
    end
    return src
end

# production M2L for a single source/offset pair at uniform order P (parity target)
function m2l_production(src, Δx, P, lamb_helmholtz)
    TF = eltype(src)
    box = SVector{3}(zero(TF), zero(TF), zero(TF))
    src_branch = FastMultipole.Branch(2:2, 0, 1:0, 0, 1, SVector{3}(zero(TF), zero(TF), zero(TF)), zero(TF), box)
    tgt_branch = FastMultipole.Branch(2:2, 0, 1:0, 0, 1, SVector{3}(Δx), zero(TF), box)

    Hs = TF[1.0]; FastMultipole.update_Hs_π2!(Hs, P)
    Ts = zeros(TF, FastMultipole.length_Ts(P))
    eimϕs = zeros(TF, 2, P + 1)
    w1 = FastMultipole.initialize_expansion(P, TF)
    w2 = FastMultipole.initialize_expansion(P, TF)
    w3 = FastMultipole.initialize_expansion(P, TF)
    ζ = zeros(TF, FastMultipole.length_ζs(P)); FastMultipole.update_ζs_mag!(ζ, 0, P)
    η = zeros(TF, FastMultipole.length_ηs(P)); FastMultipole.update_ηs_mag!(η, 0, P)

    local_exp = FastMultipole.initialize_expansion(P, TF)
    FastMultipole.multipole_to_local!(local_exp, tgt_branch, src, src_branch,
        w1, w2, w3, Ts, eimϕs, ζ, η, Hs, FastMultipole.M̃, FastMultipole.L̃, P, lamb_helmholtz)
    return local_exp
end

# Val(false): every harmonic of both lanes must match production at P.
function m2l_check_nolh!(target, ref, P, atol, rtol)
    nh = ((P + 1) * (P + 2)) >> 1
    for i in 1:nh
        @test isapprox(target[1, 1, i], ref[1, 1, i]; atol=atol, rtol=rtol)
        @test isapprox(target[2, 1, i], ref[2, 1, i]; atol=atol, rtol=rtol)
    end
end

# Val(true): φ physical through P_phi matches the φ-zero-padded production
# reference (sentinel ignored); χ through P_active matches (padding contributes);
# φ above P_phi is a clean zero (no nonphysical output row).
function m2l_check_lh!(target, ref, P_phi, P_active, atol, rtol)
    for n in 0:P_active
        for m in 0:n
            i = FastMultipole.harmonic_index(n, m)
            @test isapprox(target[1, 2, i], ref[1, 2, i]; atol=atol, rtol=rtol)
            @test isapprox(target[2, 2, i], ref[2, 2, i]; atol=atol, rtol=rtol)
            if n <= P_phi
                @test isapprox(target[1, 1, i], ref[1, 1, i]; atol=atol, rtol=rtol)
                @test isapprox(target[2, 1, i], ref[2, 1, i]; atol=atol, rtol=rtol)
            else
                @test isapprox(target[1, 1, i], zero(eltype(target)); atol=1e-13)
                @test isapprox(target[2, 1, i], zero(eltype(target)); atol=1e-13)
            end
        end
    end
end

TF = Float64
Random.seed!(140014)

# tolerances: the operator path reconstructs the y-rotation differently from
# production (cached blocks / factored modes), so parity is to round-off, not
# bit-for-bit. rtol keeps entries with magnitude tight; atol is the floor for
# structurally-zero entries (e.g. m = 0 imaginary χ), where the factored path
# yields exactly 0 while production accumulates ~1e-8 rotation round-off at high P.
M2L_ATOL = 1e-6
M2L_RTOL = 1e-7

#--- Val(false): uniform order P, all rows vs production at P ---#

# P = 1 covers the smallest order of the 019b always-dense decision (the dense
# operator path has no small-P recurrence fallback, so it must be exact there too).
@testset "parity vs production (Val(false)): $(variant)  P=$(P)" for
        variant in (MaterializedYRotationM2L(), FactoredRotationM2L()),
        P in (1, 2, 4, 6, 8)

    lh = Val(false)
    cache = OperatorInvariantCache(TF, P, lh)
    @test cache.basis_info.orders.P_active == P
    nbatch = length(M2L_OFFSETS)
    scratch = M2LOperatorScratch(TF, cache.basis_info, nbatch)
    nh = ((P + 1) * (P + 2)) >> 1

    sources = zeros(TF, 2, 2, nh, nbatch)
    refs = Vector{Array{TF,3}}(undef, nbatch)
    phis = zeros(TF, nbatch); thetas = zeros(TF, nbatch); rs = zeros(TF, nbatch)
    for j in 1:nbatch
        src = m2l_random_source(P, TF, lh)
        sources[:, :, :, j] .= src
        Δx = M2L_OFFSETS[j]
        r, θ, ϕ = FastMultipole.cartesian_to_spherical(Δx)
        rs[j] = r; thetas[j] = θ; phis[j] = ϕ
        refs[j] = m2l_production(src, Δx, P, lh)
    end

    sbuf = to_flat_buffer(sources, cache.basis_info)
    tbuf = FlatCoefficientBuffer(TF, cache.basis_info, nbatch)
    FastMultipole.m2l_operator_batch!(variant, tbuf, sbuf, phis, thetas, rs, cache, scratch, lh)
    targets = zeros(TF, 2, 2, nh, nbatch); from_flat_buffer!(targets, tbuf)
    for j in 1:nbatch
        m2l_check_nolh!(view(targets, :, :, :, j), refs[j], P, M2L_ATOL, M2L_RTOL)
    end

    # single-column path (batch width 1)
    let j = 4
        scratch1 = M2LOperatorScratch(TF, cache.basis_info, 1)
        src1 = zeros(TF, 2, 2, nh, 1); src1[:, :, :, 1] .= view(sources, :, :, :, j)
        sb1 = to_flat_buffer(src1, cache.basis_info)
        tb1 = FlatCoefficientBuffer(TF, cache.basis_info, 1)
        FastMultipole.m2l_operator_batch!(variant, tb1, sb1, [phis[j]], [thetas[j]], [rs[j]], cache, scratch1, lh)
        tgt1 = zeros(TF, 2, 2, nh, 1); from_flat_buffer!(tgt1, tb1)
        m2l_check_nolh!(view(tgt1, :, :, :, 1), refs[j], P, M2L_ATOL, M2L_RTOL)
    end
end

#--- Val(true): φ at P_phi, χ at P_active = P_phi + 1 ---#

@testset "parity vs production (Val(true)): $(variant)  P_phi=$(P_phi)" for
        variant in (MaterializedYRotationM2L(), FactoredRotationM2L()),
        P_phi in (1, 2, 4, 6, 8)

    lh = Val(true)
    cache = OperatorInvariantCache(TF, P_phi, lh)
    P_active = cache.basis_info.orders.P_active
    @test P_active == P_phi + 1
    nbatch = length(M2L_OFFSETS)
    scratch = M2LOperatorScratch(TF, cache.basis_info, nbatch)
    nh = ((P_active + 1) * (P_active + 2)) >> 1

    sources = zeros(TF, 2, 2, nh, nbatch)
    refs = Vector{Array{TF,3}}(undef, nbatch)
    phis = zeros(TF, nbatch); thetas = zeros(TF, nbatch); rs = zeros(TF, nbatch)
    for j in 1:nbatch
        src = m2l_lh_sentinel_source(P_phi, P_active, TF)
        sources[:, :, :, j] .= src
        Δx = M2L_OFFSETS[j]
        r, θ, ϕ = FastMultipole.cartesian_to_spherical(Δx)
        rs[j] = r; thetas[j] = θ; phis[j] = ϕ
        # reference: zero the φ padding (degree > P_phi), run production at P_active
        ref_src = copy(src)
        for n in (P_phi + 1):P_active
            for m in 0:n
                i = FastMultipole.harmonic_index(n, m)
                ref_src[1, 1, i] = zero(TF); ref_src[2, 1, i] = zero(TF)
            end
        end
        refs[j] = m2l_production(ref_src, Δx, P_active, lh)
    end

    sbuf = to_flat_buffer(sources, cache.basis_info)
    tbuf = FlatCoefficientBuffer(TF, cache.basis_info, nbatch)
    FastMultipole.m2l_operator_batch!(variant, tbuf, sbuf, phis, thetas, rs, cache, scratch, lh)
    targets = zeros(TF, 2, 2, nh, nbatch); from_flat_buffer!(targets, tbuf)
    for j in 1:nbatch
        m2l_check_lh!(view(targets, :, :, :, j), refs[j], P_phi, P_active, M2L_ATOL, M2L_RTOL)
    end

    # single-column path (batch width 1)
    let j = 4
        scratch1 = M2LOperatorScratch(TF, cache.basis_info, 1)
        src1 = zeros(TF, 2, 2, nh, 1); src1[:, :, :, 1] .= view(sources, :, :, :, j)
        sb1 = to_flat_buffer(src1, cache.basis_info)
        tb1 = FlatCoefficientBuffer(TF, cache.basis_info, 1)
        FastMultipole.m2l_operator_batch!(variant, tb1, sb1, [phis[j]], [thetas[j]], [rs[j]], cache, scratch1, lh)
        tgt1 = zeros(TF, 2, 2, nh, 1); from_flat_buffer!(tgt1, tb1)
        m2l_check_lh!(view(tgt1, :, :, :, 1), refs[j], P_phi, P_active, M2L_ATOL, M2L_RTOL)
    end
end

#--- Cross-variant parity: materialized vs factored agree (incl. padding) ---#

@testset "cross-variant parity: materialized vs factored  LH=$(LHbool)  P=$(P)" for
        LHbool in (false, true), P in (3, 6)

    lh = Val(LHbool)
    cache = OperatorInvariantCache(TF, P, lh)
    P_active = cache.basis_info.orders.P_active
    nbatch = length(M2L_OFFSETS)
    nh = ((P_active + 1) * (P_active + 2)) >> 1

    sources = zeros(TF, 2, 2, nh, nbatch)
    phis = zeros(TF, nbatch); thetas = zeros(TF, nbatch); rs = zeros(TF, nbatch)
    for j in 1:nbatch
        src = LHbool ? m2l_lh_sentinel_source(P, P_active, TF) : m2l_random_source(P_active, TF, lh)
        sources[:, :, :, j] .= src
        r, θ, ϕ = FastMultipole.cartesian_to_spherical(M2L_OFFSETS[j])
        rs[j] = r; thetas[j] = θ; phis[j] = ϕ
    end

    sm = M2LOperatorScratch(TF, cache.basis_info, nbatch)
    sf = M2LOperatorScratch(TF, cache.basis_info, nbatch)
    sbuf = to_flat_buffer(sources, cache.basis_info)
    tmbuf = FlatCoefficientBuffer(TF, cache.basis_info, nbatch)
    tfbuf = FlatCoefficientBuffer(TF, cache.basis_info, nbatch)
    FastMultipole.m2l_operator_batch!(MaterializedYRotationM2L(), tmbuf, sbuf, phis, thetas, rs, cache, sm, lh)
    FastMultipole.m2l_operator_batch!(FactoredRotationM2L(), tfbuf, sbuf, phis, thetas, rs, cache, sf, lh)
    tm = zeros(TF, 2, 2, nh, nbatch); from_flat_buffer!(tm, tmbuf)
    tf = zeros(TF, 2, 2, nh, nbatch); from_flat_buffer!(tf, tfbuf)

    for i in eachindex(tm)
        @test isapprox(tf[i], tm[i]; atol=M2L_ATOL, rtol=M2L_RTOL)
    end
end

end
