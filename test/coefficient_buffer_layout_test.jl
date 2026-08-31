# Tests for the native flat coefficient buffer + typed views (Matrix Operator
# Refactor, task 017). Promotes the standalone
# MATRIX_OPERATOR_REFACTOR/scripts/coefficient_buffer_layout_verify.jl checks into
# the suite (basis-index contiguity, legacy<->flat round-trip, channel layouts,
# fixed-channel slab density, χ pruning), and adds a flat-vs-legacy kernel
# round-trip that anchors the relaid flat kernels to the proven legacy [2,2,nh]
# reference kernels.

isdefined(@__MODULE__, :to_flat_buffer) || include("flat_buffer_helpers.jl")

using FastMultipole: harmonic_index, flat_basis_index, _operator_ncomplex,
    OperatorBasisInfo, FlatCoefficientBuffer, phi_slab, chi_slab, phi_physical_view,
    initialize_expansion,
    m2l_z_blocks!, m2l_z_block_length, apply_m2l_z!, apply_m2l_z_flat!,
    m2m_z_blocks!, m2m_z_block_length, apply_m2m_z!, apply_m2m_z_flat!,
    l2l_z_blocks!, l2l_z_block_length, apply_l2l_z!, apply_l2l_z_flat!,
    lamb_helmholtz_local_coeffs!, lamb_helmholtz_multipole_coeffs!,
    apply_lamb_helmholtz_local!, apply_lamb_helmholtz_local_flat!,
    apply_lamb_helmholtz_multipole!, apply_lamb_helmholtz_multipole_flat!

@testset "flat coefficient buffer layout (task 017)" begin

ORDERS = (0, 1, 3, 6, 9)
TF = Float64

@testset "flat_basis_index contiguity/uniqueness P=$(P)" for P in ORDERS
    basis_dof = 2 * _operator_ncomplex(P)
    idx = [flat_basis_index(n, m, r) for n in 0:P for m in 0:n for r in 1:2]
    @test sort(idx) == collect(1:basis_dof)
    @test length(unique(idx)) == basis_dof
    # re/im of one harmonic are adjacent (interleaved), re even-then-odd offset
    for n in 0:P, m in 0:n
        @test flat_basis_index(n, m, 2) == flat_basis_index(n, m, 1) + 1
    end
end

@testset "buffer sizing + χ pruning + slab density: LH=$(LHbool) P=$(P)" for
        LHbool in (false, true), P in ORDERS, batch in (1, 2, 5)

    lh = Val(LHbool)
    info = OperatorBasisInfo(P, lh)
    buf = FlatCoefficientBuffer(TF, info, batch)

    @test size(phi_slab(buf)) == (info.basis_dof_phi, batch)
    @test info.basis_dof_phi == 2 * _operator_ncomplex(info.orders.P_phi)
    # fixed-channel slab is a dense GEMM target
    @test stride(phi_slab(buf), 1) == 1
    @test stride(phi_slab(buf), 2) == info.basis_dof_phi
    @test size(phi_physical_view(buf)) == (info.basis_dof_phi, batch)

    if LHbool
        @test info.channel_count == 2
        @test size(chi_slab(buf)) == (info.basis_dof_chi, batch)
        @test info.basis_dof_chi == 2 * _operator_ncomplex(info.orders.P_active)
        @test info.orders.P_active == P + 1   # χ carried at P+1 (008h)
        @test stride(chi_slab(buf), 1) == 1
        @test stride(chi_slab(buf), 2) == info.basis_dof_chi
    else
        @test info.channel_count == 1
        @test size(chi_slab(buf)) == (0, 0)   # dead χ channel pruned
        @test info.orders.P_active == P
    end
end

@testset "legacy<->flat round-trip: LH=$(LHbool) P=$(P)" for
        LHbool in (false, true), P in ORDERS

    lh = Val(LHbool)
    info = OperatorBasisInfo(P, lh)
    P_active = info.orders.P_active
    nh = _operator_ncomplex(P_active)
    nbatch = 3
    legacy = zeros(TF, 2, 2, nh, nbatch)
    for j in 1:nbatch, c in 1:(LHbool ? 2 : 1), i in 1:nh
        legacy[1, c, i, j] = 1000.0 * j + 100.0 * c + 10.0 + i / 1000.0
        legacy[2, c, i, j] = 1000.0 * j + 100.0 * c + 20.0 + i / 1000.0
    end
    buf = to_flat_buffer(legacy, info)
    rt = fill(TF(NaN), 2, 2, nh, nbatch)
    from_flat_buffer!(rt, buf)
    # φ physical through P_phi, χ through P_active must round-trip exactly
    maxerr = 0.0
    for j in 1:nbatch
        for n in 0:info.orders.P_phi, m in 0:n
            i = harmonic_index(n, m)
            maxerr = max(maxerr, abs(rt[1, 1, i, j] - legacy[1, 1, i, j]),
                                 abs(rt[2, 1, i, j] - legacy[2, 1, i, j]))
        end
        if LHbool
            for n in 0:P_active, m in 0:n
                i = harmonic_index(n, m)
                maxerr = max(maxerr, abs(rt[1, 2, i, j] - legacy[1, 2, i, j]),
                                     abs(rt[2, 2, i, j] - legacy[2, 2, i, j]))
            end
            # φ above P_phi reads back as a clean zero (ragged buffer has no such rows)
            for n in (info.orders.P_phi + 1):P_active, m in 0:n
                i = harmonic_index(n, m)
                @test rt[1, 1, i, j] == 0
                @test rt[2, 1, i, j] == 0
            end
        end
    end
    @test maxerr == 0.0
end

# Flat kernels reproduce the proven legacy [2,2,nh] order-aware kernels bit-for-bit
# (same arithmetic, same multiply order; only indexing moves). Single-column.
function random_physical_frame(P_active, LHbool)
    src = initialize_expansion(P_active, TF)
    for n in 0:P_active, m in 0:n
        i = harmonic_index(n, m)
        src[1, 1, i] = randn(); src[2, 1, i] = m == 0 ? 0.0 : randn()
        if LHbool
            src[1, 2, i] = randn(); src[2, 2, i] = m == 0 ? 0.0 : randn()
        end
    end
    return src
end

@testset "flat vs legacy kernel round-trip: LH=$(LHbool) P=$(P)" for
        LHbool in (false, true), P in (2, 4, 7)

    Random.seed!(170017 + P + (LHbool ? 100 : 0))
    lh = Val(LHbool)
    info = OperatorBasisInfo(P, lh)
    P_active = info.orders.P_active
    P_phi = info.orders.P_phi
    nh = _operator_ncomplex(P_active)
    r = 2.6
    KTOL = 1e-11

    cmp_physical(ref, flat) = begin
        for n in 0:P_phi, m in 0:n
            i = harmonic_index(n, m)
            @test isapprox(flat[1, 1, i], ref[1, 1, i]; atol=KTOL, rtol=KTOL)
            @test isapprox(flat[2, 1, i], ref[2, 1, i]; atol=KTOL, rtol=KTOL)
        end
        if LHbool
            for n in 0:P_active, m in 0:n
                i = harmonic_index(n, m)
                @test isapprox(flat[1, 2, i], ref[1, 2, i]; atol=KTOL, rtol=KTOL)
                @test isapprox(flat[2, 2, i], ref[2, 2, i]; atol=KTOL, rtol=KTOL)
            end
        end
    end

    mkcase() = begin
        src = random_physical_frame(P_active, LHbool)
        # The ragged flat φ matrix carries no rows above P_phi, so the reference
        # source must zero its φ padding to match (matters for the L2L φ gather,
        # which pulls from higher source degrees; the order-aware m2l/LH kernels
        # ignore φ > P_phi structurally, so this is a no-op for them).
        if LHbool
            for n in (P_phi + 1):P_active, m in 0:n
                i = harmonic_index(n, m); src[1, 1, i] = 0.0; src[2, 1, i] = 0.0
            end
        end
        ref = initialize_expansion(P_active, TF)
        sbuf = to_flat_buffer(reshape(src, 2, 2, nh, 1), info)
        obuf = FlatCoefficientBuffer(TF, info, 1)
        (src, ref, sbuf, obuf)
    end

    # M2L z-translation
    let (src, ref, sbuf, obuf) = mkcase()
        blocks = Vector{TF}(undef, m2l_z_block_length(P_active)); m2l_z_blocks!(blocks, r, P_active)
        apply_m2l_z!(ref, src, blocks, info, Val(:overwrite))
        apply_m2l_z_flat!(obuf, sbuf, 1, blocks, Val(:overwrite))
        out = fill(TF(NaN), 2, 2, nh, 1); from_flat_buffer!(out, obuf)
        cmp_physical(ref, view(out, :, :, :, 1))
    end

    # M2M z-translation (order-aware legacy uses uniform-P kernel; compare physical)
    let (src, ref, sbuf, obuf) = mkcase()
        blocks = Vector{TF}(undef, m2m_z_block_length(P_active)); m2m_z_blocks!(blocks, r, P_active)
        apply_m2m_z!(ref, src, blocks, P_active, lh, Val(:overwrite))
        apply_m2m_z_flat!(obuf, sbuf, 1, blocks, Val(:overwrite))
        out = fill(TF(NaN), 2, 2, nh, 1); from_flat_buffer!(out, obuf)
        cmp_physical(ref, view(out, :, :, :, 1))
    end

    # L2L z-translation
    let (src, ref, sbuf, obuf) = mkcase()
        blocks = Vector{TF}(undef, l2l_z_block_length(P_active)); l2l_z_blocks!(blocks, r, P_active)
        apply_l2l_z!(ref, src, blocks, P_active, lh, Val(:overwrite))
        apply_l2l_z_flat!(obuf, sbuf, 1, blocks, Val(:overwrite))
        out = fill(TF(NaN), 2, 2, nh, 1); from_flat_buffer!(out, obuf)
        cmp_physical(ref, view(out, :, :, :, 1))
    end

    if LHbool
        # Lamb-Helmholtz local (separate buffers)
        let (src, ref, sbuf, obuf) = mkcase()
            A = zeros(TF, nh); B = zeros(TF, nh); lamb_helmholtz_local_coeffs!(A, B, r, P_active)
            apply_lamb_helmholtz_local!(ref, src, A, B, info, Val(:overwrite))
            apply_lamb_helmholtz_local_flat!(obuf, sbuf, 1, A, B, Val(:overwrite))
            out = fill(TF(NaN), 2, 2, nh, 1); from_flat_buffer!(out, obuf)
            cmp_physical(ref, view(out, :, :, :, 1))
        end
        # Lamb-Helmholtz multipole (separate buffers)
        let (src, ref, sbuf, obuf) = mkcase()
            A = zeros(TF, nh); B = zeros(TF, nh); lamb_helmholtz_multipole_coeffs!(A, B, r, P_active)
            apply_lamb_helmholtz_multipole!(ref, src, A, B, info, Val(:overwrite))
            apply_lamb_helmholtz_multipole_flat!(obuf, sbuf, 1, A, B, Val(:overwrite))
            out = fill(TF(NaN), 2, 2, nh, 1); from_flat_buffer!(out, obuf)
            cmp_physical(ref, view(out, :, :, :, 1))
        end
    end
end

end
