#=
Parity tests for the explicit M2L z-translation blocks (Matrix Operator Refactor,
task 011) in src/translate_batched.jl. The materialize-then-apply path must
reproduce the production translate_multipole_to_local_z! behavior
(src/translate.jl) bit-for-bit, and the order-aware OperatorBasisInfo method must
carry the Lamb-Helmholtz χ channel at the padded order P_active = P_phi + 1.
=#

using FastMultipole
using Random
using Test
using FastMultipole: harmonic_index, initialize_expansion,
    translate_multipole_to_local_z!,
    transform_lamb_helmholtz_multipole!, transform_lamb_helmholtz_local!,
    m2l_z_block_length, m2l_z_block_offset, m2l_z_blocks!, apply_m2l_z!,
    lamb_helmholtz_multipole_coeffs!, lamb_helmholtz_local_coeffs!,
    apply_lamb_helmholtz_multipole!, apply_lamb_helmholtz_local!,
    OperatorBasisInfo, CompressedComplexBasis

const M2L_Z_ORDERS = (0, 1, 3, 6, 9)

ncomplex_z(P) = ((P + 1) * (P + 2)) >> 1

function random_expansion_z(P, TF, ::Val{LH}) where LH
    w = initialize_expansion(P, TF)
    nh = ncomplex_z(P)
    for i in 1:nh
        w[1,1,i] = randn(TF)
        w[2,1,i] = randn(TF)
        if LH
            w[1,2,i] = randn(TF)
            w[2,2,i] = randn(TF)
        end
    end
    return w
end

@testset "M2L z-translation blocks (batched)" begin

    # internal/non-exported, but callable as module members
    @test !(:m2l_z_blocks! in names(FastMultipole))
    @test !(:apply_m2l_z! in names(FastMultipole))
    @test FastMultipole.m2l_z_blocks! === m2l_z_blocks!
    @test FastMultipole.apply_m2l_z! === apply_m2l_z!

    # block-length accounting: sum_{k=1}^{P+1} k^2
    for P in 0:12
        @test m2l_z_block_length(P) == sum(k^2 for k in 1:(P+1))
        @test m2l_z_block_offset(0, P) == 0
        @test m2l_z_block_offset(P+1, P) == m2l_z_block_length(P)
    end

    Random.seed!(2011)

    #--- symmetric parity vs production (bit-for-bit) ---#

    for TF in (Float64, Float32)
        for LHbool in (false, true)
            lh = Val(LHbool)
            for P in M2L_Z_ORDERS
                blocks = Vector{TF}(undef, m2l_z_block_length(P))
                for t in (TF(5.4084748312255275), TF(1.3), TF(-2.7), TF(11.0))
                    src = random_expansion_z(P, TF, lh)

                    ref = initialize_expansion(P, TF)
                    translate_multipole_to_local_z!(ref, src, t, P, lh)

                    m2l_z_blocks!(blocks, t, P)
                    out = initialize_expansion(P, TF)
                    # preload sentinel to confirm overwrite (every stored coeff written)
                    fill!(out, TF(-99))
                    apply_m2l_z!(out, src, blocks, P, lh, Val(:overwrite))

                    nh = ncomplex_z(P)
                    for i in 1:nh
                        @test out[1,1,i] === ref[1,1,i]
                        @test out[2,1,i] === ref[2,1,i]
                        if LHbool
                            @test out[1,2,i] === ref[1,2,i]
                            @test out[2,2,i] === ref[2,2,i]
                        end
                        # overwrite: no sentinel survives in stored slots
                        @test out[1,1,i] != TF(-99)
                    end
                end
            end
        end
    end

    # Note: the canonical numeric datum (P=10, t=5.408...) is already covered
    # transitively: the symmetric block path is bit-for-bit identical to
    # translate_multipole_to_local_z!, which translate_multipole_to_local_test.jl
    # validates against the literal translated_weights_test vector.

    #--- Val(true) padded order via OperatorBasisInfo: χ carried at P_active = P+1 ---#

    for TF in (Float64, Float32)
        for P in M2L_Z_ORDERS
            lh = Val(true)
            basis_info = OperatorBasisInfo(CompressedComplexBasis(), P, lh)
            P_active = basis_info.orders.P_active
            @test basis_info.orders.P_phi == P
            @test P_active == P + 1

            t = TF(3.25)

            # source carried at the padded order P_active so χ has its extra degree
            src = random_expansion_z(P_active, TF, lh)

            blocks = Vector{TF}(undef, m2l_z_block_length(P_active))
            m2l_z_blocks!(blocks, t, P_active)

            out = initialize_expansion(P_active, TF)
            fill!(out, TF(-99))
            apply_m2l_z!(out, src, blocks, basis_info, Val(:overwrite))

            # reference φ at order P_phi=P (single channel)
            ref_phi = initialize_expansion(P, TF)
            src_phi = initialize_expansion(P, TF)
            for i in 1:ncomplex_z(P)
                src_phi[1,1,i] = src[1,1,i]
                src_phi[2,1,i] = src[2,1,i]
            end
            translate_multipole_to_local_z!(ref_phi, src_phi, t, P, Val(false))

            # reference χ at order P_active=P+1 (single channel), fed from component 2
            ref_chi = initialize_expansion(P_active, TF)
            src_chi = initialize_expansion(P_active, TF)
            for i in 1:ncomplex_z(P_active)
                src_chi[1,1,i] = src[1,2,i]
                src_chi[2,1,i] = src[2,2,i]
            end
            translate_multipole_to_local_z!(ref_chi, src_chi, t, P_active, Val(false))

            # φ output: rows n<=P_phi must match the P-order production result exactly
            for i in 1:ncomplex_z(P)
                @test out[1,1,i] === ref_phi[1,1,i]
                @test out[2,1,i] === ref_phi[2,1,i]
            end
            # χ output: rows n<=P_active must match the (P+1)-order production result
            for i in 1:ncomplex_z(P_active)
                @test out[1,2,i] === ref_chi[1,1,i]
                @test out[2,2,i] === ref_chi[2,1,i]
            end
        end
    end

end

@testset "Lamb-Helmholtz operators (batched)" begin

    # internal/non-exported, but callable as module members
    @test !(:lamb_helmholtz_multipole_coeffs! in names(FastMultipole))
    @test !(:apply_lamb_helmholtz_local! in names(FastMultipole))
    @test FastMultipole.apply_lamb_helmholtz_multipole! === apply_lamb_helmholtz_multipole!
    @test FastMultipole.apply_lamb_helmholtz_local! === apply_lamb_helmholtz_local!

    Random.seed!(2012)

    ncoeff(P) = ((P + 1) * (P + 2)) >> 1

    # full two-channel random expansion (both φ and χ populated through order P)
    function random_lh_expansion(P, TF)
        w = initialize_expansion(P, TF)
        for i in 1:ncoeff(P)
            w[1,1,i] = randn(TF); w[2,1,i] = randn(TF)
            w[1,2,i] = randn(TF); w[2,2,i] = randn(TF)
        end
        return w
    end

    #--- symmetric parity vs production (bit-for-bit), multipole and local ---#

    for TF in (Float64, Float32)
        for P in M2L_Z_ORDERS
            A = Vector{TF}(undef, ncoeff(P))
            B = Vector{TF}(undef, ncoeff(P))
            for r in (TF(0.7), TF(1.0), TF(-1.9), TF(3.25))
                src = random_lh_expansion(P, TF)

                # multipole side
                ref = copy(src)
                transform_lamb_helmholtz_multipole!(ref, r, P)
                lamb_helmholtz_multipole_coeffs!(A, B, r, P)
                out = initialize_expansion(P, TF); fill!(out, TF(-99))
                apply_lamb_helmholtz_multipole!(out, src, A, B, P, Val(:overwrite))
                for i in 1:ncoeff(P)
                    @test out[1,1,i] === ref[1,1,i]
                    @test out[2,1,i] === ref[2,1,i]
                    @test out[1,2,i] === ref[1,2,i]
                    @test out[2,2,i] === ref[2,2,i]
                    @test out[1,1,i] != TF(-99)   # overwrite: every slot written
                end

                # local side
                ref = copy(src)
                transform_lamb_helmholtz_local!(ref, r, P)
                lamb_helmholtz_local_coeffs!(A, B, r, P)
                out = initialize_expansion(P, TF); fill!(out, TF(-99))
                apply_lamb_helmholtz_local!(out, src, A, B, P, Val(:overwrite))
                for i in 1:ncoeff(P)
                    @test out[1,1,i] === ref[1,1,i]
                    @test out[2,1,i] === ref[2,1,i]
                    @test out[1,2,i] === ref[1,2,i]
                    @test out[2,2,i] === ref[2,2,i]
                    @test out[1,1,i] != TF(-99)
                end
            end
        end
    end

    #--- order-aware padded path: χ carried at P_active = P_phi + 1 ---#
    # The order-aware transform equals production run at the padded order P_active,
    # restricted to physical φ rows (n <= P_phi) and all χ rows (n <= P_active).

    for TF in (Float64, Float32)
        for P in M2L_Z_ORDERS
            basis_info = OperatorBasisInfo(CompressedComplexBasis(), P, Val(true))
            P_phi = basis_info.orders.P_phi
            P_active = basis_info.orders.P_active
            @test P_phi == P
            @test P_active == P + 1

            r = TF(2.3)
            src = random_lh_expansion(P_active, TF)   # carried at padded order
            A = Vector{TF}(undef, ncoeff(P_active))
            B = Vector{TF}(undef, ncoeff(P_active))

            #-- multipole order-aware --#
            ref = copy(src)
            transform_lamb_helmholtz_multipole!(ref, r, P_active)
            lamb_helmholtz_multipole_coeffs!(A, B, r, P_active)
            out = initialize_expansion(P_active, TF); fill!(out, TF(-99))
            apply_lamb_helmholtz_multipole!(out, src, A, B, basis_info, Val(:overwrite))
            for i in 1:ncoeff(P_phi)            # physical φ rows
                @test out[1,1,i] === ref[1,1,i]
                @test out[2,1,i] === ref[2,1,i]
            end
            for i in 1:ncoeff(P_active)         # all χ rows
                @test out[1,2,i] === ref[1,2,i]
                @test out[2,2,i] === ref[2,2,i]
            end

            #-- local order-aware --#
            ref = copy(src)
            transform_lamb_helmholtz_local!(ref, r, P_active)
            lamb_helmholtz_local_coeffs!(A, B, r, P_active)
            out = initialize_expansion(P_active, TF); fill!(out, TF(-99))
            apply_lamb_helmholtz_local!(out, src, A, B, basis_info, Val(:overwrite))
            for i in 1:ncoeff(P_phi)
                @test out[1,1,i] === ref[1,1,i]
                @test out[2,1,i] === ref[2,1,i]
            end
            for i in 1:ncoeff(P_active)
                @test out[1,2,i] === ref[1,2,i]
                @test out[2,2,i] === ref[2,2,i]
            end

            #-- the upper-neighbor χ_{P_phi+1} -> χ_{P_phi} row must be present:
            #   order-aware χ_{P_phi} must differ from same-order (truncated) χ_{P_phi}.
            ref_trunc = initialize_expansion(P, TF)
            for i in 1:ncoeff(P)
                ref_trunc[1,1,i] = src[1,1,i]; ref_trunc[2,1,i] = src[2,1,i]
                ref_trunc[1,2,i] = src[1,2,i]; ref_trunc[2,2,i] = src[2,2,i]
            end
            transform_lamb_helmholtz_local!(ref_trunc, r, P)   # truncates χ_{P+1}=0
            differs = false
            for m in 0:P_phi
                i = harmonic_index(P_phi, m)
                if out[1,2,i] != ref_trunc[1,2,i] || out[2,2,i] != ref_trunc[2,2,i]
                    differs = true
                end
            end
            @test differs   # padded path genuinely uses the extra χ degree
        end
    end

    #--- Val(false): no χ channel, φ copied through unchanged ---#

    for TF in (Float64, Float32)
        for P in M2L_Z_ORDERS
            basis_info = OperatorBasisInfo(CompressedComplexBasis(), P, Val(false))
            @test basis_info.orders.P_active == P
            r = TF(1.4)

            src = initialize_expansion(P, TF)
            for i in 1:ncoeff(P)
                src[1,1,i] = randn(TF); src[2,1,i] = randn(TF)
                src[1,2,i] = randn(TF); src[2,2,i] = randn(TF) # absent channel ignored
            end
            A = Vector{TF}(undef, ncoeff(P)); B = Vector{TF}(undef, ncoeff(P))

            lamb_helmholtz_multipole_coeffs!(A, B, r, P)
            out = initialize_expansion(P, TF); fill!(out, TF(-99))
            apply_lamb_helmholtz_multipole!(out, src, A, B, basis_info, Val(:overwrite))
            for i in 1:ncoeff(P)
                @test out[1,1,i] === src[1,1,i]
                @test out[2,1,i] === src[2,1,i]
                @test iszero(out[1,2,i])
                @test iszero(out[2,2,i])
            end

            lamb_helmholtz_local_coeffs!(A, B, r, P)
            out = initialize_expansion(P, TF); fill!(out, TF(-99))
            apply_lamb_helmholtz_local!(out, src, A, B, basis_info, Val(:overwrite))
            for i in 1:ncoeff(P)
                @test out[1,1,i] === src[1,1,i]
                @test out[2,1,i] === src[2,1,i]
                @test iszero(out[1,2,i])
                @test iszero(out[2,2,i])
            end
        end
    end

end
