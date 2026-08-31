@testset "real solid harmonic basis (task 018)" begin

using FastMultipole: CompressedComplexBasis, RealSolidHarmonicBasis,
    OperatorBasisInfo, FlatCoefficientBuffer, phi_slab, chi_slab,
    flat_basis_index, real_basis_index, _operator_ncomplex,
    _real_solid_harmonic_dof, complex_to_real_basis!, real_to_complex_basis!,
    flat_nbatch,
    harmonic_index, initialize_harmonics, initialize_gradient_n_m,
    apply_z_rotation_batch_flat!, m2l_z_blocks!, m2l_z_block_length,
    apply_m2l_z_flat!

REAL_BASIS_ORDERS = (0, 1, 3, 6, 9)
REAL_BASIS_TOL = 2.0e-11

function fill_physical_complex!(buf)
    info = buf.basis_info
    Random.seed!(180018 + info.orders.P_phi + (info.channel_count == 2 ? 100 : 0) + flat_nbatch(buf))
    ph = phi_slab(buf)
    for j in axes(ph, 2), n in 0:info.orders.P_phi, m in 0:n
        fr = flat_basis_index(n, m, 1)
        ph[fr, j] = randn()
        ph[fr + 1, j] = m == 0 ? 0.0 : randn()
    end
    if info.channel_count == 2
        ch = chi_slab(buf)
        for j in axes(ch, 2), n in 0:info.orders.P_active, m in 0:n
            fr = flat_basis_index(n, m, 1)
            ch[fr, j] = randn()
            ch[fr + 1, j] = m == 0 ? 0.0 : randn()
        end
    end
    return buf
end

function complex_flat_to_legacy(buf)
    info = buf.basis_info
    nh = _operator_ncomplex(info.orders.P_active)
    out = zeros(eltype(phi_slab(buf)), 2, 2, nh)
    ph = phi_slab(buf)
    for n in 0:info.orders.P_phi, m in 0:n
        i = harmonic_index(n, m)
        fr = flat_basis_index(n, m, 1)
        out[1, 1, i] = ph[fr, 1]
        out[2, 1, i] = ph[fr + 1, 1]
    end
    if info.channel_count == 2
        ch = chi_slab(buf)
        for n in 0:info.orders.P_active, m in 0:n
            i = harmonic_index(n, m)
            fr = flat_basis_index(n, m, 1)
            out[1, 2, i] = ch[fr, 1]
            out[2, 2, i] = ch[fr + 1, 1]
        end
    end
    return out
end

function assert_complex_buffers_close(a, b; atol=REAL_BASIS_TOL)
    info = a.basis_info
    @test b.basis_info.orders == info.orders
    ap = phi_slab(a); bp = phi_slab(b)
    for j in axes(ap, 2), n in 0:info.orders.P_phi, m in 0:n
        fr = flat_basis_index(n, m, 1)
        @test isapprox(ap[fr, j], bp[fr, j]; atol, rtol=atol)
        @test isapprox(ap[fr + 1, j], bp[fr + 1, j]; atol, rtol=atol)
    end
    if info.channel_count == 2
        ac = chi_slab(a); bc = chi_slab(b)
        for j in axes(ac, 2), n in 0:info.orders.P_active, m in 0:n
            fr = flat_basis_index(n, m, 1)
            @test isapprox(ac[fr, j], bc[fr, j]; atol, rtol=atol)
            @test isapprox(ac[fr + 1, j], bc[fr + 1, j]; atol, rtol=atol)
        end
    end
end

@testset "real-basis index and sizing P=$(P)" for P in REAL_BASIS_ORDERS
    idx = Int[]
    for n in 0:P
        push!(idx, real_basis_index(n, 0))
        for m in 1:n
            push!(idx, real_basis_index(n, m, Val(:cos)))
            push!(idx, real_basis_index(n, m, Val(:sin)))
        end
    end
    @test sort(idx) == collect(1:(P + 1)^2)
    @test length(unique(idx)) == (P + 1)^2
    @test _real_solid_harmonic_dof(P) == (P + 1)^2
end

@testset "transform round-trip and m=0 projection LH=$(LHbool)" for LHbool in (false, true)
    lh = Val(LHbool)
    P = 5
    batch = 4
    cinfo = OperatorBasisInfo(CompressedComplexBasis(), P, lh)
    rinfo = OperatorBasisInfo(RealSolidHarmonicBasis(), P, lh)
    csrc = fill_physical_complex!(FlatCoefficientBuffer(Float64, cinfo, batch))
    rbuf = FlatCoefficientBuffer(Float64, rinfo, batch)
    crt = FlatCoefficientBuffer(Float64, cinfo, batch)
    complex_to_real_basis!(rbuf, csrc)
    real_to_complex_basis!(crt, rbuf)
    assert_complex_buffers_close(csrc, crt; atol=0.0)

    cproj = fill_physical_complex!(FlatCoefficientBuffer(Float64, cinfo, batch))
    for j in 1:batch, n in 0:cinfo.orders.P_phi
        phi_slab(cproj)[flat_basis_index(n, 0, 2), j] = 10 + n + j / 10
    end
    if LHbool
        for j in 1:batch, n in 0:cinfo.orders.P_active
            chi_slab(cproj)[flat_basis_index(n, 0, 2), j] = -20 - n - j / 10
        end
    end
    complex_to_real_basis!(rbuf, cproj)
    real_to_complex_basis!(crt, rbuf)
    for j in 1:batch, n in 0:cinfo.orders.P_phi
        @test phi_slab(crt)[flat_basis_index(n, 0, 2), j] == 0.0
    end
    if LHbool
        for j in 1:batch, n in 0:cinfo.orders.P_active
            @test chi_slab(crt)[flat_basis_index(n, 0, 2), j] == 0.0
        end
    end
end

@testset "transform-sandwiched z operators LH=$(LHbool)" for LHbool in (false, true)
    lh = Val(LHbool)
    P = 4
    batch = 3
    phis = [-0.7, 0.25, 1.3]
    cinfo = OperatorBasisInfo(CompressedComplexBasis(), P, lh)
    rinfo = OperatorBasisInfo(RealSolidHarmonicBasis(), P, lh)
    csrc = fill_physical_complex!(FlatCoefficientBuffer(Float64, cinfo, batch))
    cout = FlatCoefficientBuffer(Float64, cinfo, batch)
    rbuf = FlatCoefficientBuffer(Float64, rinfo, batch)
    complex_to_real_basis!(rbuf, csrc)
    ctmp = FlatCoefficientBuffer(Float64, cinfo, batch)
    real_to_complex_basis!(ctmp, rbuf)
    apply_z_rotation_batch_flat!(cout, csrc, phis, Val(:overwrite))
    apply_z_rotation_batch_flat!(ctmp, ctmp, phis, Val(:overwrite))
    assert_complex_buffers_close(cout, ctmp)

    blocks = Vector{Float64}(undef, m2l_z_block_length(cinfo.orders.P_active))
    m2l_z_blocks!(blocks, 2.4, cinfo.orders.P_active)
    cz = FlatCoefficientBuffer(Float64, cinfo, batch)
    rz = FlatCoefficientBuffer(Float64, cinfo, batch)
    apply_m2l_z_flat!(cz, csrc, 2, blocks, Val(:overwrite))
    real_to_complex_basis!(ctmp, rbuf)
    apply_m2l_z_flat!(rz, ctmp, 2, blocks, Val(:overwrite))
    assert_complex_buffers_close(cz, rz)
end

@testset "real-basis evaluation parity LH=$(LHbool)" for LHbool in (false, true)
    lh = Val(LHbool)
    P = 5
    cinfo = OperatorBasisInfo(CompressedComplexBasis(), P, lh)
    rinfo = OperatorBasisInfo(RealSolidHarmonicBasis(), P, lh)
    csrc = fill_physical_complex!(FlatCoefficientBuffer(Float64, cinfo, 1))
    rbuf = FlatCoefficientBuffer(Float64, rinfo, 1)
    complex_to_real_basis!(rbuf, csrc)

    Δx = SVector{3}(0.31, -0.44, 0.72)
    P_eval = cinfo.orders.P_active
    h1 = initialize_harmonics(P_eval)
    h2 = initialize_harmonics(P_eval)
    g1 = initialize_gradient_n_m(P_eval)
    g2 = initialize_gradient_n_m(P_eval)
    sw = LHbool ? DerivativesSwitch(false, true, true) : DerivativesSwitch(true, true, true)

    ref = FastMultipole.evaluate_local(Δx, h1, g1, complex_flat_to_legacy(csrc), P_eval, lh, sw)
    got = FastMultipole.evaluate_local(Δx, h2, g2, rbuf, P, lh, sw)
    @test isapprox(got[1], ref[1]; atol=REAL_BASIS_TOL, rtol=REAL_BASIS_TOL)
    @test isapprox(got[2], ref[2]; atol=REAL_BASIS_TOL, rtol=REAL_BASIS_TOL)
    @test isapprox(got[3], ref[3]; atol=REAL_BASIS_TOL, rtol=REAL_BASIS_TOL)
end

end
