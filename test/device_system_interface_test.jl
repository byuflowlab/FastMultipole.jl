# Task 032 stage 1: generalized device-system interface, host-resident path.
# Covers the canonical all-rows packed layout (data_per_body > 5), the
# body_type trait with the Point{Vortex} B2M (φ + χ, Lamb-Helmholtz
# end-to-end), and the 9-component hessian output chosen at cache
# construction — all without CUDA. Device mirrors are exercised in
# test/cuda_radix_interface_test.jl.

using FastMultipole
using FastMultipole.StaticArrays
using FastMultipole.LinearAlgebra
using Random
using Test

if !isdefined(@__MODULE__, :generate_gravitational)
    include("gravitational.jl")
end
if !isdefined(@__MODULE__, :VortexParticles)
    include("vortex.jl")
end
if !isdefined(@__MODULE__, :ExtendedVortex)
    include("interface_test_systems.jl")
end

@testset "device-system interface stage 1 (task 032)" begin

    seed = 20260805

    #--- (a) scalar 9-component hessian vs analytic direct ---#

    n = 600
    sys8 = generate_gravitational(seed, n)
    u_ref, g_ref, H_ref = _interface_scalar_direct(sys8)
    cache8 = RadixFMMCache(sys8; expansion_order=8, ell=3, hessian=true,
        options=CUDARadixLifecycleOptions(; precision=Float64,
            m2l_strategy=FastMultipole.ConcatenatedFixedZM2L()))
    @test cache8.hessian
    fmm!(sys8, cache8; scalar_potential=true, gradient=true, hessian=true)
    @test maximum(abs.(sys8.potential[1, :] .- u_ref)) < 1e-6
    @test maximum(abs.(sys8.potential[5:7, :] .- g_ref)) < 1e-4
    @test maximum(abs.(sys8.potential[8:16, :] .- H_ref)) < 5e-3

    # P = 4 (standing rule): same-P parity below plus a loose accuracy check
    sys4 = generate_gravitational(seed, n)
    cache4 = RadixFMMCache(sys4; expansion_order=4, ell=3, hessian=true,
        options=CUDARadixLifecycleOptions(; precision=Float64,
            m2l_strategy=FastMultipole.ConcatenatedFixedZM2L()))
    fmm!(sys4, cache4; scalar_potential=true, gradient=true, hessian=true)
    @test maximum(abs.(sys4.potential[5:7, :] .- g_ref)) < 5e-2
    @test maximum(abs.(sys4.potential[8:16, :] .- H_ref)) < 2.0

    #--- (b) hessian=true cache leaves potential + gradient unchanged ---#

    for P in (4, 8)
        plain = generate_gravitational(seed, n)
        pcache = RadixFMMCache(plain; expansion_order=P, ell=3,
            options=CUDARadixLifecycleOptions(; precision=Float64,
                m2l_strategy=FastMultipole.ConcatenatedFixedZM2L()))
        fmm!(plain, pcache; scalar_potential=true, gradient=true)
        hsys = P == 8 ? sys8 : sys4
        @test maximum(abs.(plain.potential[1, :] .- hsys.potential[1, :])) < 1e-12
        @test maximum(abs.(plain.potential[5:7, :] .- hsys.potential[5:7, :])) < 1e-12
    end

    #--- (c) vortex end-to-end: B2M -> M2M -> M2L -> L2L -> L2B + nearfield ---#

    nv = 600
    vref = generate_vortex(seed, nv)
    FastMultipole.direct!(vref; gradient=true, hessian=true)
    g_scale = maximum(abs.(vref.gradient_stretching[1:3, :]))
    h_scale = maximum(abs.(vref.potential[5:13, :]))

    # Float64 at P = 8 (accuracy gate) and P = 4 (same-P coverage, loose)
    for (P, gtol, htol) in ((8, 1e-4, 1e-2), (4, 2e-3, 0.2))
        vsys = generate_vortex(seed, nv)
        vcache = RadixFMMCache(vsys; expansion_order=P, ell=3, hessian=true,
            options=CUDARadixLifecycleOptions(; precision=Float64,
                m2l_strategy=FastMultipole.ConcatenatedFixedZM2L()))
        @test vcache isa RadixFMMCache{Float64,true}
        @test vcache.state.options.body_type === Point{Vortex}
        fmm!(vsys, vcache; scalar_potential=false, gradient=true, hessian=true)
        @test maximum(abs.(vsys.gradient_stretching[1:3, :] .-
            vref.gradient_stretching[1:3, :])) < gtol
        @test maximum(abs.(vsys.potential[5:13, :] .- vref.potential[5:13, :])) < htol
        # chi channel is genuinely live
        @test any(!iszero, vcache.state.multipoles.chi)
        @test any(!iszero, vcache.state.locals.chi)
    end

    # Float32 at P = 4: relative accuracy floor (task 024: ~5e-4, essentially
    # P-independent), so gate loosely relative to the field scale
    v32 = generate_vortex(seed, nv)
    c32 = RadixFMMCache(v32; expansion_order=4, ell=3, hessian=true,
        options=CUDARadixLifecycleOptions(; precision=Float32,
            m2l_strategy=FastMultipole.ConcatenatedFixedZM2L()))
    fmm!(v32, c32; scalar_potential=false, gradient=true, hessian=true)
    @test maximum(abs.(v32.gradient_stretching[1:3, :] .-
        vref.gradient_stretching[1:3, :])) / g_scale < 2e-3
    @test maximum(abs.(v32.potential[5:13, :] .- vref.potential[5:13, :])) / h_scale < 2e-3

    #--- (d) packed-layout round trip with data_per_body > 5 ---#

    ext = ExtendedVortex(generate_vortex(seed, 300; radius_factor=0.1))
    eref = generate_vortex(seed, 300; radius_factor=0.1)
    FastMultipole.direct!(eref; gradient=true, hessian=true)
    ecache = RadixFMMCache(ext; expansion_order=8, ell=3, hessian=true,
        options=CUDARadixLifecycleOptions(; precision=Float64,
            m2l_strategy=FastMultipole.ConcatenatedFixedZM2L()))
    fmm!(ext, ecache; scalar_potential=false, gradient=true, hessian=true)
    @test maximum(abs.(ext.inner.gradient_stretching[1:3, :] .-
        eref.gradient_stretching[1:3, :])) < 1e-4
    # all data_per_body rows land in the packed matrix, including radius row 4
    st = ecache.state
    @test size(st.source_bodies, 1) == 9
    for sorted_i in 1:st.counts.n_bodies
        ibody = st.body_indices[st.body_perm[sorted_i]]
        @test st.source_bodies[4, sorted_i] == ext.inner.bodies[ibody].sigma
        @test st.source_bodies[8, sorted_i] == 10.0 + ibody
        @test st.source_bodies[9, sorted_i] == -Float64(ibody)
    end

    #--- (e) construction/validation error paths ---#

    plain = generate_gravitational(seed, 100)
    c4 = RadixFMMCache(plain; expansion_order=4, ell=2)
    @test !c4.hessian
    @test_throws ArgumentError fmm!(plain, c4; hessian=true)
    vplain = generate_vortex(seed, 100)
    @test_throws ArgumentError RadixFMMCache(vplain; expansion_order=4, ell=2,
        lamb_helmholtz=false)
    @test_throws ArgumentError RadixFMMCache((vplain, plain); expansion_order=4, ell=2)
end
