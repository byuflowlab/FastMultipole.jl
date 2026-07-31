# Resident batched-M2M GEMM strategies + GEMM-native buffer (Matrix Operator Refactor,
# task 022). Validates the array-generic degree-major buffer, its converters, and both
# swappable M2M strategies (SharedRotationM2M / DenseTranslationM2M) against the
# task-016 per-column `m2m_operator_batch!` oracle. CPU here exercises the same `mul!`
# code path used on the GPU (CuArray) lifecycle.

using FastMultipole
const FM = FastMultipole
using FastMultipole.StaticArrays
using LinearAlgebra, Random, Test

# --- helpers ---
_zero_im_m0!(slab, P) = (for n in 0:P; slab[FM.flat_basis_index(n, 0, 2), :] .= 0; end; slab)

function _random_flat_source(TF, binfo, N, rng, ::Val{LH}) where LH
    src = FM.FlatCoefficientBuffer(TF, binfo, N)
    src.phi .= TF.(randn(rng, size(src.phi)))
    _zero_im_m0!(src.phi, binfo.orders.P_phi)
    if LH
        src.chi .= TF.(randn(rng, size(src.chi)))
        _zero_im_m0!(src.chi, binfo.orders.P_active)
    end
    return src
end

function _oracle_m2m(TF, binfo, lhv::Val{LH}, src, phis, thetas, rs) where LH
    N = FM.flat_nbatch(src)
    cache = FM.OperatorInvariantCache(TF, binfo)
    scratch = FM.M2MOperatorScratch(TF, binfo, N)
    tgt = FM.FlatCoefficientBuffer(TF, binfo, N)
    fill!(tgt.phi, 0); LH && fill!(tgt.chi, 0)
    FM.m2m_operator_batch!(FM.MaterializedYRotationM2M(), tgt, src, phis, thetas, rs, cache, scratch, lhv)
    return tgt
end

function _oracle_m2l(TF, binfo, lhv::Val{LH}, src, phis, thetas, rs) where LH
    N = FM.flat_nbatch(src)
    cache = FM.OperatorInvariantCache(TF, binfo)
    scratch = FM.M2LOperatorScratch(TF, binfo, N)
    tgt = FM.FlatCoefficientBuffer(TF, binfo, N)
    fill!(tgt.phi, 0); LH && fill!(tgt.chi, 0)
    FM.m2l_operator_batch!(FM.MaterializedYRotationM2L(), tgt, src, phis, thetas, rs, cache, scratch, lhv)
    return tgt
end

function _oracle_l2l(TF, binfo, lhv::Val{LH}, src, phis, thetas, rs) where LH
    N = FM.flat_nbatch(src)
    cache = FM.OperatorInvariantCache(TF, binfo)
    scratch = FM.L2LOperatorScratch(TF, binfo, N)
    tgt = FM.FlatCoefficientBuffer(TF, binfo, N)
    fill!(tgt.phi, 0); LH && fill!(tgt.chi, 0)
    FM.l2l_operator_batch!(FM.MaterializedYRotationL2L(), tgt, src, phis, thetas, rs, cache, scratch, lhv)
    return tgt
end

function _assert_physical_m0(testbuf, P; atol)
    for n in 0:P
        @test maximum(abs.(testbuf[FM.flat_basis_index(n, 0, 2), :])) <= atol
    end
end

@testset "GEMM-native degree-major buffer roundtrip" begin
    for LH in (Val(false), Val(true)), TF in (Float64, Float32), P in (0, 1, 2, 4)
        binfo = FM.OperatorBasisInfo(FM.CompressedComplexBasis(), P, LH)
        N = 3
        src = _random_flat_source(TF, binfo, N, MersenneTwister(P), LH)
        gemm = FM.DegreeMajorRealBuffer(TF, binfo, N)
        FM.to_gemm_buffer!(gemm, src)
        @test size(gemm.phi, 1) == (binfo.orders.P_phi + 1)^2
        back = FM.FlatCoefficientBuffer(TF, binfo, N)
        FM.to_flat_buffer!(back, gemm)
        @test back.phi ≈ src.phi
        LH === Val(true) && @test back.chi ≈ src.chi
    end
    @test FM.degree_row_range(0) == 1:1
    @test length(FM.degree_row_range(3)) == 7
    @test FM.degree_major_dof(4) == 25
end

@testset "resident_m2l_batch! parity vs task-016 oracle" begin
    angle_sets = (
        (0.0, 0.0, 0.6),
        (0.0, pi, 0.6),
        (0.0, pi / 2, 0.7),
        (pi / 2, pi / 3, 0.8),
        (pi, pi / 4, 0.9),
        (1.234, 0.789, 1.1),
    )
    for LH in (Val(false), Val(true)), TF in (Float64, Float32), P in (0, 1, 2, 5)
        binfo = FM.OperatorBasisInfo(FM.CompressedComplexBasis(), P, LH)
        rng = MersenneTwister(2000 + P + (LH === Val(true) ? 100 : 0))
        for (phi0, theta0, r0) in angle_sets
            N = 5
            src = _random_flat_source(TF, binfo, N, rng, LH)
            phis = fill(TF(phi0), N)
            thetas = fill(TF(theta0), N)
            rs = fill(TF(r0), N)
            tgt = _oracle_m2l(TF, binfo, LH, src, phis, thetas, rs)

            cache = FM.OperatorInvariantCache(TF, binfo)
            sdm = FM.DegreeMajorRealBuffer(TF, binfo, N); FM.to_gemm_buffer!(sdm, src)
            tdm = FM.DegreeMajorRealBuffer(TF, binfo, N)
            FM.resident_m2l_batch!(FM.SharedRotationM2L(), tdm, sdm, phis, thetas, rs, cache, LH)
            got = FM.FlatCoefficientBuffer(TF, binfo, N); FM.to_flat_buffer!(got, tdm)

            rtol = TF === Float32 ? 5f-3 : 2e-9
            @test got.phi ≈ tgt.phi rtol=rtol atol=rtol
            _assert_physical_m0(got.phi, binfo.orders.P_phi; atol=rtol)
            if LH === Val(true)
                @test got.chi ≈ tgt.chi rtol=rtol atol=rtol
                _assert_physical_m0(got.chi, binfo.orders.P_active; atol=rtol)
            end
        end
    end
    # The dense parity surface deliberately accepts nonuniform geometry and builds
    # one complete materialized-y operator per column.
    for LH in (Val(false), Val(true)), TF in (Float64, Float32), P in (0, 1, 2, 5)
        binfo = FM.OperatorBasisInfo(FM.CompressedComplexBasis(), P, LH)
        N = length(angle_sets)
        rng = MersenneTwister(9000 + P + (LH === Val(true) ? 100 : 0))
        src = _random_flat_source(TF, binfo, N, rng, LH)
        phis = TF[first(a) for a in angle_sets]
        thetas = TF[a[2] for a in angle_sets]
        rs = TF[a[3] for a in angle_sets]
        oracle = _oracle_m2l(TF, binfo, LH, src, phis, thetas, rs)
        sdm = FM.DegreeMajorRealBuffer(TF, binfo, N); FM.to_gemm_buffer!(sdm, src)
        tdm = FM.DegreeMajorRealBuffer(TF, binfo, N)
        cache = FM.OperatorInvariantCache(TF, binfo)
        FM.resident_m2l_batch!(FM.DenseTranslationM2L(), tdm, sdm, phis,
            thetas, rs, cache, LH)
        got = FM.FlatCoefficientBuffer(TF, binfo, N); FM.to_flat_buffer!(got, tdm)
        tol = TF === Float32 ? 3f-3 : 1e-9
        @test got.phi ≈ oracle.phi rtol=tol atol=tol
        _assert_physical_m0(got.phi, binfo.orders.P_phi; atol=tol)
        if LH === Val(true)
            @test got.chi ≈ oracle.chi rtol=tol atol=tol
            _assert_physical_m0(got.chi, binfo.orders.P_active; atol=tol)
        end
    end
end

@testset "resident_l2l_batch! parity vs task-016 oracle" begin
    angle_sets = (
        (0.0, 0.0, 0.6),
        (0.0, pi, 0.6),
        (0.0, pi / 2, 0.7),
        (pi / 2, pi / 3, 0.8),
        (pi, pi / 4, 0.9),
        (1.234, 0.789, 1.1),
    )
    for LH in (Val(false), Val(true)), TF in (Float64, Float32), P in (0, 1, 2, 5)
        binfo = FM.OperatorBasisInfo(FM.CompressedComplexBasis(), P, LH)
        N = length(angle_sets)
        rng = MersenneTwister(3000 + P + (LH === Val(true) ? 100 : 0))
        src = _random_flat_source(TF, binfo, N, rng, LH)
        phis = TF[first(a) for a in angle_sets]
        thetas = TF[a[2] for a in angle_sets]
        rs = fill(TF(0.75), N)
        tgt = _oracle_l2l(TF, binfo, LH, src, phis, thetas, rs)

        cache = FM.OperatorInvariantCache(TF, binfo)
        sdm = FM.DegreeMajorRealBuffer(TF, binfo, N); FM.to_gemm_buffer!(sdm, src)
        tdm = FM.DegreeMajorRealBuffer(TF, binfo, N)
        FM.resident_l2l_batch!(tdm, sdm, phis, thetas, rs, cache, LH)
        got = FM.FlatCoefficientBuffer(TF, binfo, N); FM.to_flat_buffer!(got, tdm)

        rtol = TF === Float32 ? 5f-3 : 2e-9
        @test got.phi ≈ tgt.phi rtol=rtol atol=rtol
        _assert_physical_m0(got.phi, binfo.orders.P_phi; atol=rtol)
        if LH === Val(true)
            @test got.chi ≈ tgt.chi rtol=rtol atol=rtol
            _assert_physical_m0(got.chi, binfo.orders.P_active; atol=rtol)
        end
    end
end

@testset "resident_m2m_batch! parity vs task-016 oracle" begin
    for strat in (FM.SharedRotationM2M(), FM.DenseTranslationM2M())
        for LH in (Val(false), Val(true)), TF in (Float64, Float32), P in (0, 1, 2, 5)
            binfo = FM.OperatorBasisInfo(FM.CompressedComplexBasis(), P, LH)
            N = 7
            rng = MersenneTwister(P + (LH === Val(true) ? 100 : 0))
            src = _random_flat_source(TF, binfo, N, rng, LH)
            rs = strat isa FM.SharedRotationM2M ? fill(TF(0.7), N) : TF.(0.4 .+ rand(rng, N))
            thetas = TF.(pi .* rand(rng, N)); phis = TF.(2pi .* rand(rng, N))
            tgt = _oracle_m2m(TF, binfo, LH, src, phis, thetas, rs)

            cache = FM.OperatorInvariantCache(TF, binfo)
            sdm = FM.DegreeMajorRealBuffer(TF, binfo, N); FM.to_gemm_buffer!(sdm, src)
            tdm = FM.DegreeMajorRealBuffer(TF, binfo, N)   # zeroed -> accumulate from 0
            FM.resident_m2m_batch!(strat, tdm, sdm, phis, thetas, rs, cache, LH)
            got = FM.FlatCoefficientBuffer(TF, binfo, N); FM.to_flat_buffer!(got, tdm)

            rtol = TF === Float32 ? 3f-3 : 1e-9
            @test got.phi ≈ tgt.phi rtol=rtol atol=rtol
            LH === Val(true) && @test got.chi ≈ tgt.chi rtol=rtol atol=rtol
        end
    end
end
