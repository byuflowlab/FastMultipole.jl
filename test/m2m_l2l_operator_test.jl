# Parity tests for the explicit M2M and L2L operator pipelines (task 016).

@testset "M2M/L2L operator pipelines (task 016)" begin

# native flat coefficient buffer helpers (task 017)
isdefined(@__MODULE__, :to_flat_buffer) || include("flat_buffer_helpers.jl")

using FastMultipole: harmonic_index, initialize_expansion,
    translate_multipole_z!, translate_local_z!,
    m2m_z_block_length, m2m_z_block_offset, m2m_z_blocks!, apply_m2m_z!,
    l2l_z_block_length, l2l_z_block_offset, l2l_z_blocks!, apply_l2l_z!

M2M_L2L_OFFSETS = (
    SVector{3}(0.0, 0.0, 2.0),
    SVector{3}(0.0, 0.0, -2.4),
    SVector{3}(2.8, 0.0, 0.0),
    SVector{3}(0.0, 1.9, 0.0),
    SVector{3}(1.5, -2.0, 2.5),
    SVector{3}(-2.3, 1.1, -3.4),
)

ncomplex_m2m_l2l(P) = ((P + 1) * (P + 2)) >> 1

function random_physical_expansion_016(P, TF, ::Val{LH}; sentinel_phi_above=-1) where LH
    src = initialize_expansion(P, TF)
    for n in 0:P
        for m in 0:n
            i = harmonic_index(n, m)
            src[1, 1, i] = randn(TF)
            src[2, 1, i] = m == 0 ? zero(TF) : randn(TF)
            if LH
                src[1, 2, i] = randn(TF)
                src[2, 2, i] = m == 0 ? zero(TF) : randn(TF)
            end
        end
    end
    if LH && sentinel_phi_above >= 0
        for n in (sentinel_phi_above + 1):P
            for m in 0:n
                i = harmonic_index(n, m)
                src[1, 1, i] = TF(10) + randn(TF)
                src[2, 1, i] = m == 0 ? zero(TF) : TF(10) + randn(TF)
            end
        end
    end
    return src
end

function zero_phi_padding_016!(src, P_phi, P_active)
    for n in (P_phi + 1):P_active
        for m in 0:n
            i = harmonic_index(n, m)
            src[1, 1, i] = zero(eltype(src))
            src[2, 1, i] = zero(eltype(src))
        end
    end
    return src
end

function production_m2m_016(src, Δx, P, lamb_helmholtz)
    TF = eltype(src)
    box = SVector{3}(zero(TF), zero(TF), zero(TF))
    source_branch = Branch(1:1, 0, 1:0, 0, 1, SVector{3}(zero(TF), zero(TF), zero(TF)), zero(TF), box)
    target_branch = Branch(1:1, 0, 1:0, 0, 1, SVector{3}(Δx), zero(TF), box)
    Hs = TF[1]; FastMultipole.update_Hs_π2!(Hs, P)
    Ts = zeros(TF, FastMultipole.length_Ts(P))
    eimphis = zeros(TF, 2, P + 1)
    zeta = zeros(TF, FastMultipole.length_ζs(P)); FastMultipole.update_ζs_mag!(zeta, 0, P)
    w1 = initialize_expansion(P, TF)
    w2 = initialize_expansion(P, TF)
    target = initialize_expansion(P, TF)
    FastMultipole.multipole_to_multipole!(
        target, target_branch, src, source_branch, w1, w2, Ts, eimphis,
        zeta, Hs, P, lamb_helmholtz,
    )
    return target
end

function production_l2l_016(src, Δx, P, lamb_helmholtz)
    TF = eltype(src)
    box = SVector{3}(zero(TF), zero(TF), zero(TF))
    source_branch = Branch(1:1, 0, 1:0, 0, 1, SVector{3}(zero(TF), zero(TF), zero(TF)), zero(TF), box)
    target_branch = Branch(1:1, 0, 1:0, 0, 1, SVector{3}(Δx), zero(TF), box)
    Hs = TF[1]; FastMultipole.update_Hs_π2!(Hs, P)
    Ts = zeros(TF, FastMultipole.length_Ts(P))
    eimphis = zeros(TF, 2, P + 1)
    eta = zeros(TF, FastMultipole.length_ηs(P)); FastMultipole.update_ηs_mag!(eta, 0, P)
    w1 = initialize_expansion(P, TF)
    w2 = initialize_expansion(P, TF)
    target = initialize_expansion(P, TF)
    FastMultipole.local_to_local!(
        target, target_branch, src, source_branch, w1, w2, Ts, eimphis,
        eta, Hs, P, lamb_helmholtz,
    )
    return target
end

function check_nolh_016!(target, ref, P, atol, rtol)
    for i in 1:ncomplex_m2m_l2l(P)
        @test isapprox(target[1, 1, i], ref[1, 1, i]; atol=atol, rtol=rtol)
        @test isapprox(target[2, 1, i], ref[2, 1, i]; atol=atol, rtol=rtol)
    end
end

function check_lh_016!(target, ref, P_phi, P_active, atol, rtol)
    for n in 0:P_active
        for m in 0:n
            i = harmonic_index(n, m)
            @test isapprox(target[1, 2, i], ref[1, 2, i]; atol=atol, rtol=rtol)
            @test isapprox(target[2, 2, i], ref[2, 2, i]; atol=atol, rtol=rtol)
            if n <= P_phi
                @test isapprox(target[1, 1, i], ref[1, 1, i]; atol=atol, rtol=rtol)
                @test isapprox(target[2, 1, i], ref[2, 1, i]; atol=atol, rtol=rtol)
            else
                @test isapprox(target[1, 1, i], zero(eltype(target)); atol=atol)
                @test isapprox(target[2, 1, i], zero(eltype(target)); atol=atol)
            end
        end
    end
end

Random.seed!(160016)

@testset "M2M/L2L z blocks" begin
    for P in 0:12
        @test m2m_z_block_length(P) == sum(k * (k + 1) ÷ 2 for k in 1:(P + 1))
        @test l2l_z_block_length(P) == m2m_z_block_length(P)
        @test m2m_z_block_offset(0, P) == 0
        @test l2l_z_block_offset(P + 1, P) == l2l_z_block_length(P)
    end

    for TF in (Float32, Float64), LHbool in (false, true), P in (0, 1, 3, 6, 9)
        lh = Val(LHbool)
        for t in (TF(0.75), TF(-1.2), TF(3.5))
            src = random_physical_expansion_016(P, TF, lh)

            ref_m2m = initialize_expansion(P, TF)
            translate_multipole_z!(ref_m2m, src, t, P, lh)
            m2m_blocks = Vector{TF}(undef, m2m_z_block_length(P))
            m2m_z_blocks!(m2m_blocks, t, P)
            out_m2m = initialize_expansion(P, TF)
            fill!(out_m2m, TF(-99))
            apply_m2m_z!(out_m2m, src, m2m_blocks, P, lh, Val(:overwrite))

            ref_l2l = initialize_expansion(P, TF)
            translate_local_z!(ref_l2l, src, t, P, lh)
            l2l_blocks = Vector{TF}(undef, l2l_z_block_length(P))
            l2l_z_blocks!(l2l_blocks, t, P)
            out_l2l = initialize_expansion(P, TF)
            fill!(out_l2l, TF(-99))
            apply_l2l_z!(out_l2l, src, l2l_blocks, P, lh, Val(:overwrite))

            for i in 1:ncomplex_m2m_l2l(P)
                z_atol = TF === Float32 ? 2f-5 : 1e-12
                z_rtol = TF === Float32 ? 2f-6 : 1e-12
                @test isapprox(out_m2m[1, 1, i], ref_m2m[1, 1, i]; atol=z_atol, rtol=z_rtol)
                @test isapprox(out_m2m[2, 1, i], ref_m2m[2, 1, i]; atol=z_atol, rtol=z_rtol)
                @test isapprox(out_l2l[1, 1, i], ref_l2l[1, 1, i]; atol=z_atol, rtol=z_rtol)
                @test isapprox(out_l2l[2, 1, i], ref_l2l[2, 1, i]; atol=z_atol, rtol=z_rtol)
                if LHbool
                    @test isapprox(out_m2m[1, 2, i], ref_m2m[1, 2, i]; atol=z_atol, rtol=z_rtol)
                    @test isapprox(out_m2m[2, 2, i], ref_m2m[2, 2, i]; atol=z_atol, rtol=z_rtol)
                    @test isapprox(out_l2l[1, 2, i], ref_l2l[1, 2, i]; atol=z_atol, rtol=z_rtol)
                    @test isapprox(out_l2l[2, 2, i], ref_l2l[2, 2, i]; atol=z_atol, rtol=z_rtol)
                end
            end
        end
    end
end

M2M_L2L_ATOL = 1e-6
M2M_L2L_RTOL = 1e-7
TF016 = Float64

@testset "M2M pipeline parity: $(variant) LH=$(LHbool) P=$(P)" for
        variant in (MaterializedYRotationM2M(), FactoredRotationM2M()),
        LHbool in (false, true), P in (2, 4, 6, 8)

    lh = Val(LHbool)
    cache = OperatorInvariantCache(TF016, P, lh)
    P_active = cache.basis_info.orders.P_active
    nbatch = length(M2M_L2L_OFFSETS)
    nh = ncomplex_m2m_l2l(P_active)
    scratch = M2MOperatorScratch(TF016, cache.basis_info, nbatch)
    sources = zeros(TF016, 2, 2, nh, nbatch)
    targets = zeros(TF016, 2, 2, nh, nbatch)
    refs = Vector{Array{TF016,3}}(undef, nbatch)
    phis = zeros(TF016, nbatch); thetas = zeros(TF016, nbatch); rs = zeros(TF016, nbatch)

    for j in 1:nbatch
        src = random_physical_expansion_016(P_active, TF016, lh; sentinel_phi_above=LHbool ? P : -1)
        sources[:, :, :, j] .= src
        ref_src = copy(src)
        LHbool && zero_phi_padding_016!(ref_src, P, P_active)
        Δx = M2M_L2L_OFFSETS[j]
        r, θ, ϕ = FastMultipole.cartesian_to_spherical(Δx)
        rs[j] = r; thetas[j] = θ; phis[j] = ϕ
        refs[j] = production_m2m_016(ref_src, Δx, P_active, lh)
    end

    sbuf = to_flat_buffer(sources, cache.basis_info)
    tbuf = FlatCoefficientBuffer(TF016, cache.basis_info, nbatch)
    FastMultipole.m2m_operator_batch!(variant, tbuf, sbuf, phis, thetas, rs, cache, scratch, lh)
    from_flat_buffer!(targets, tbuf)
    for j in 1:nbatch
        if LHbool
            check_lh_016!(view(targets, :, :, :, j), refs[j], P, P_active, M2M_L2L_ATOL, M2M_L2L_RTOL)
        else
            check_nolh_016!(view(targets, :, :, :, j), refs[j], P_active, M2M_L2L_ATOL, M2M_L2L_RTOL)
        end
    end
end

@testset "L2L pipeline parity: $(variant) LH=$(LHbool) P=$(P)" for
        variant in (MaterializedYRotationL2L(), FactoredRotationL2L()),
        LHbool in (false, true), P in (2, 4, 6, 8)

    lh = Val(LHbool)
    cache = OperatorInvariantCache(TF016, P, lh)
    P_active = cache.basis_info.orders.P_active
    nbatch = length(M2M_L2L_OFFSETS)
    nh = ncomplex_m2m_l2l(P_active)
    scratch = L2LOperatorScratch(TF016, cache.basis_info, nbatch)
    sources = zeros(TF016, 2, 2, nh, nbatch)
    targets = zeros(TF016, 2, 2, nh, nbatch)
    refs = Vector{Array{TF016,3}}(undef, nbatch)
    phis = zeros(TF016, nbatch); thetas = zeros(TF016, nbatch); rs = zeros(TF016, nbatch)

    for j in 1:nbatch
        src = random_physical_expansion_016(P_active, TF016, lh; sentinel_phi_above=LHbool ? P : -1)
        sources[:, :, :, j] .= src
        ref_src = copy(src)
        LHbool && zero_phi_padding_016!(ref_src, P, P_active)
        Δx = M2M_L2L_OFFSETS[j]
        r, θ, ϕ = FastMultipole.cartesian_to_spherical(Δx)
        rs[j] = r; thetas[j] = θ; phis[j] = ϕ
        refs[j] = production_l2l_016(ref_src, Δx, P_active, lh)
    end

    sbuf = to_flat_buffer(sources, cache.basis_info)
    tbuf = FlatCoefficientBuffer(TF016, cache.basis_info, nbatch)
    FastMultipole.l2l_operator_batch!(variant, tbuf, sbuf, phis, thetas, rs, cache, scratch, lh)
    from_flat_buffer!(targets, tbuf)
    for j in 1:nbatch
        if LHbool
            check_lh_016!(view(targets, :, :, :, j), refs[j], P, P_active, M2M_L2L_ATOL, M2M_L2L_RTOL)
        else
            check_nolh_016!(view(targets, :, :, :, j), refs[j], P_active, M2M_L2L_ATOL, M2M_L2L_RTOL)
        end
    end
end

end
