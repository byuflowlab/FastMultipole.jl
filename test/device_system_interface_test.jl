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

@testset "direct-kernel functor trait stage 2 (task 032)" begin

    seed = 20260805

    #--- (a) erf-free gaussianerf g/h vs an erf reference ---#

    A = sqrt(2 / pi)
    g_ref(rho) = _ref_erf(rho / sqrt(2)) - A * rho * exp(-rho^2 / 2)
    # the naive h = ρg′ − 3g cancels catastrophically at small ρ (the very §3
    # hazard), so the series-branch reference is evaluated in BigFloat
    function gh_big(rho)
        rb = big(rho)
        x = rb / sqrt(big(2))
        s = zero(BigFloat); term = x; n = 0
        while true
            add = term / (2n + 1); s += add; n += 1; term *= -x * x / n
            abs(add) <= eps(BigFloat) * max(abs(s), one(BigFloat)) && break
        end
        erfb = 2 / sqrt(big(pi)) * s
        Ab = sqrt(big(2) / big(pi))
        gb = erfb - Ab * rb * exp(-rb * rb / 2)
        hb = Ab * rb^3 * exp(-rb * rb / 2) - 3 * gb
        return Float64(gb), Float64(hb)
    end
    budget = 0.5e-3 * g_ref(2.0)          # 031a §6.2 absolute budget = 3.69e-4
    for (TF, rtol) in ((Float64, 1e-11), (Float32, 2e-6))
        emax_series = 0.0
        for rho in range(1e-3, 2.0, length=2001)
            g, h = FastMultipole._gaussianerf_g_h(TF(rho))
            gb, hb = gh_big(rho)
            emax_series = max(emax_series, abs(g / gb - 1), abs(h / hb - 1))
        end
        @test emax_series < rtol           # relative on the series branch
        emax_outer = 0.0
        for rho in range(2.0 + 1e-9, 40.0, length=2001)
            g, h = FastMultipole._gaussianerf_g_h(TF(rho))
            emax_outer = max(emax_outer, abs(Float64(g) - g_ref(rho)))
        end
        @test emax_outer < budget          # absolute on the outer branch
        # graceful singular limit
        g, h = FastMultipole._gaussianerf_g_h(TF(40))
        @test g == one(TF) && h == -TF(3)
    end

    #--- (b) functor kernels reproduce the hard-coded kernels exactly ---#

    n = 400
    sys = generate_gravitational(seed, n)
    cache = RadixFMMCache(sys; expansion_order=4, ell=3, hessian=true,
        options=CUDARadixLifecycleOptions(; precision=Float64,
            m2l_strategy=FastMultipole.ConcatenatedFixedZM2L()))
    @test cache.state.options.direct_kernel === SingularSource()
    fmm!(sys, cache; gradient=true, hessian=true)   # populates packed state
    st = cache.state
    args = (st.source_bodies, st.cell_ranges, st.direct_targets,
        st.direct_sources, st.counts.n_direct)
    ref = zeros(13, size(st.output, 2))
    got = zeros(13, size(st.output, 2))
    FastMultipole._host_direct_pairs_hessian_kernel!(ref, args...)
    FastMultipole._host_direct_pairs_functor_kernel!(SingularSource(), got, args...,
        Val(true))
    @test maximum(abs.(got .- ref)) == 0.0          # identical operation order
    fill!(ref, 0.0); fill!(got, 0.0)
    FastMultipole._host_direct_pairs_kernel!(view(ref, 1:4, :), args...)
    FastMultipole._host_direct_pairs_functor_kernel!(SingularSource(),
        view(got, 1:4, :), args..., Val(false))
    @test maximum(abs.(got .- ref)) == 0.0

    vsys = generate_vortex(seed, n)
    vcache = RadixFMMCache(vsys; expansion_order=4, ell=3, hessian=true,
        options=CUDARadixLifecycleOptions(; precision=Float64,
            m2l_strategy=FastMultipole.ConcatenatedFixedZM2L()))
    @test vcache.state.options.direct_kernel === SingularVortex()
    fmm!(vsys, vcache; gradient=true, hessian=true)
    vst = vcache.state
    vargs = (vst.source_bodies, vst.cell_ranges, vst.direct_targets,
        vst.direct_sources, vst.counts.n_direct)
    vref = zeros(13, size(vst.output, 2))
    vgot = zeros(13, size(vst.output, 2))
    FastMultipole._host_direct_pairs_vortex_kernel!(vref, vargs..., Val(true))
    FastMultipole._host_direct_pairs_functor_kernel!(SingularVortex(), vgot,
        vargs..., Val(true))
    scale = maximum(abs.(vref))
    @test maximum(abs.(vgot .- vref)) < 1e-13 * scale   # crss/a/b vs expanded form

    #--- (c) RegularizedVortex end-to-end vs erf-based regularized direct ---#

    nv = 400
    for (TF, gtol, htol) in ((Float64, 1e-3, 1e-3), (Float32, 3e-3, 3e-3))
        base = generate_vortex(seed, nv)
        sigma = 0.02 .+ 0.02 .* rand(MersenneTwister(seed), nv)   # σ_max = 0.04
        ssys = SmoothedVortex(base, sigma)
        U_ref, J_ref = _interface_regularized_direct(SmoothedVortex(base, sigma))
        scache = RadixFMMCache(ssys; expansion_order=8, ell=2, hessian=true,
            options=CUDARadixLifecycleOptions(; precision=TF,
                m2l_strategy=FastMultipole.ConcatenatedFixedZM2L()))
        @test scache.state.options.direct_kernel == RegularizedVortex(; sigma_row=8)
        fmm!(ssys, scache; scalar_potential=false, gradient=true, hessian=true)
        u_scale = maximum(abs.(U_ref)); j_scale = maximum(abs.(J_ref))
        @test maximum(abs.(base.gradient_stretching[1:3, :] .- U_ref)) / u_scale < gtol
        @test maximum(abs.(base.potential[5:13, :] .- J_ref)) / j_scale < htol
    end

    #--- (d) near-set adequacy gate: reject, don't enlarge ---#

    base = generate_vortex(seed, 400)
    big_sigma = fill(0.2, 400)          # cutoff ρ_t·σ ≈ 0.96 ≫ any leaf gap
    bad = SmoothedVortex(base, big_sigma)
    @test_throws ArgumentError RadixFMMCache(bad; expansion_order=4, ell=3,
        options=CUDARadixLifecycleOptions(; precision=Float64,
            m2l_strategy=FastMultipole.ConcatenatedFixedZM2L()))

    #--- (e) construction/validation error paths ---#

    # RegularizedVortex on a scalar body type
    plain = generate_gravitational(seed, 100)
    @test_throws ArgumentError RadixFMMCache(plain; expansion_order=4, ell=2,
        options=CUDARadixLifecycleOptions(; precision=Float64,
            m2l_strategy=FastMultipole.ConcatenatedFixedZM2L(),
            direct_kernel=RegularizedVortex(; sigma_row=8)))
    # sigma_row beyond the packed width
    base = generate_vortex(seed, 100)
    thin = SmoothedVortex(base, fill(0.01, 100))
    @test_throws ArgumentError RadixFMMCache(thin; expansion_order=4, ell=2,
        options=CUDARadixLifecycleOptions(; precision=Float64,
            m2l_strategy=FastMultipole.ConcatenatedFixedZM2L(),
            direct_kernel=RegularizedVortex(; sigma_row=9)))
    # invalid functor construction
    @test_throws ArgumentError RegularizedVortex(; sigma_row=4)
    @test_throws ArgumentError RegularizedVortex(; sigma_row=8, rho_t=0.0)
end

@testset "partitioned nearfield stage A (task 032a)" begin

    seed = 20260806
    rho_t = 4.789

    #--- (a) pair-level three-way parity: the partitioned kernel is bitwise the
    #    regularized kernel inside the cutoff and bitwise the singular kernel
    #    beyond it (same code paths, branch only) ---#

    for TF in (Float64, Float32)
        rk = RegularizedVortex(; sigma_row=8)
        pk = PartitionedVortex(; sigma_row=8)
        sk = SingularVortex()
        rng = MersenneTwister(seed)
        srcb = zeros(TF, 8, 1)
        for trial in 1:200
            sigma = TF(0.01 + 0.09 * rand(rng))
            # stay clear of the threshold so recomputed rho cannot straddle it
            inside = isodd(trial)
            rho = inside ? TF(0.05 + 4.4 * rand(rng)) : TF(5.0 + 15.0 * rand(rng))
            u = normalize(randn(rng, 3))
            dx, dy, dz = TF.(u .* Float64(rho * sigma))
            r2 = dx * dx + dy * dy + dz * dz
            invr = inv(sqrt(r2))
            srcb[5:7, 1] .= randn(rng, TF, 3)
            srcb[8, 1] = sigma
            want = inside ? FastMultipole._direct_pair_ugh(rk, dx, dy, dz, r2,
                invr, srcb, 1) :
                FastMultipole._direct_pair_ugh(sk, dx, dy, dz, r2, invr, srcb, 1)
            @test FastMultipole._direct_pair_ugh(pk, dx, dy, dz, r2, invr,
                srcb, 1) === want
            want_ug = inside ? FastMultipole._direct_pair_ug(rk, dx, dy, dz, r2,
                invr, srcb, 1) :
                FastMultipole._direct_pair_ug(sk, dx, dy, dz, r2, invr, srcb, 1)
            @test FastMultipole._direct_pair_ug(pk, dx, dy, dz, r2, invr,
                srcb, 1) === want_ug
        end
        # σ <= 0 padding falls back to singular, as for RegularizedVortex
        srcb[8, 1] = zero(TF)
        @test FastMultipole._direct_pair_ugh(pk, TF(0.1), TF(0), TF(0), TF(0.01),
            TF(10), srcb, 1) ===
            FastMultipole._direct_pair_ugh(sk, TF(0.1), TF(0), TF(0), TF(0.01),
            TF(10), srcb, 1)
    end

    #--- (b) end-to-end host resident A/B at identical geometry: partitioned vs
    #    regularized-everywhere vs the erf-based direct reference; P=8 and P=4 ---#

    nv = 400
    for P in (8, 4), (TF, gtol) in ((Float64, 1e-3), (Float32, 3e-3))
        tol = P == 4 ? 10 * gtol : gtol   # P=4 truncation dominates (task 032 record)
        base_r = generate_vortex(seed, nv)
        base_p = generate_vortex(seed, nv)
        sigma = 0.02 .+ 0.02 .* rand(MersenneTwister(seed), nv)
        rsys = SmoothedVortex(base_r, sigma)
        psys = PartitionedSmoothedVortex(SmoothedVortex(base_p, sigma))
        opts() = CUDARadixLifecycleOptions(; precision=TF,
            m2l_strategy=FastMultipole.ConcatenatedFixedZM2L())
        rcache = RadixFMMCache(rsys; expansion_order=P, ell=2, hessian=true,
            options=opts())
        pcache = RadixFMMCache(psys; expansion_order=P, ell=2, hessian=true,
            options=opts())
        @test pcache.state.options.direct_kernel == PartitionedVortex(; sigma_row=8)
        fmm!(rsys, rcache; scalar_potential=false, gradient=true, hessian=true)
        fmm!(psys, pcache; scalar_potential=false, gradient=true, hessian=true)
        U_ref, J_ref = _interface_regularized_direct(SmoothedVortex(
            generate_vortex(seed, nv), sigma))
        u_scale = maximum(abs.(U_ref)); j_scale = maximum(abs.(J_ref))
        # partitioned meets the same reference tolerance as regularized-everywhere
        @test maximum(abs.(base_p.gradient_stretching[1:3, :] .- U_ref)) / u_scale < tol
        @test maximum(abs.(base_p.potential[5:13, :] .- J_ref)) / j_scale < tol
        # and the two kernels differ only by the bounded beyond-cutoff tail
        @test maximum(abs.(base_p.gradient_stretching[1:3, :] .-
            base_r.gradient_stretching[1:3, :])) / u_scale < 5e-4
        @test maximum(abs.(base_p.potential[5:13, :] .-
            base_r.potential[5:13, :])) / j_scale < 5e-4
    end

    #--- (c) validation and error paths ---#

    # partitioned trait conflicts with an explicit different kernel
    base = generate_vortex(seed, 100)
    thin = SmoothedVortex(base, fill(0.01, 100))
    @test_throws ArgumentError RadixFMMCache(thin; expansion_order=4, ell=2,
        options=CUDARadixLifecycleOptions(; precision=Float64,
            m2l_strategy=FastMultipole.ConcatenatedFixedZM2L(),
            direct_kernel=PartitionedVortex(; sigma_row=8)))
    # scalar body type rejected
    plain = generate_gravitational(seed, 100)
    @test_throws ArgumentError RadixFMMCache(plain; expansion_order=4, ell=2,
        options=CUDARadixLifecycleOptions(; precision=Float64,
            m2l_strategy=FastMultipole.ConcatenatedFixedZM2L(),
            direct_kernel=PartitionedVortex(; sigma_row=8)))
    # adequacy gate applies identically to the partitioned kernel
    fat = PartitionedSmoothedVortex(SmoothedVortex(generate_vortex(seed, 400),
        fill(0.2, 400)))
    @test_throws ArgumentError RadixFMMCache(fat; expansion_order=4, ell=3,
        options=CUDARadixLifecycleOptions(; precision=Float64,
            m2l_strategy=FastMultipole.ConcatenatedFixedZM2L()))
    # constructor negatives
    @test_throws ArgumentError PartitionedVortex(; sigma_row=4)
    @test_throws ArgumentError PartitionedVortex(; sigma_row=8, rho_t=0.0)
end

@testset "stage 3 (task 032): recenter!, deprecated hooks" begin

    seed = 20260805
    opts64 = CUDARadixLifecycleOptions(; precision=Float64,
        m2l_strategy=FastMultipole.ConcatenatedFixedZM2L())

    #--- (a) deprecated device-buffer hooks are gone ---#

    @test !isdefined(FastMultipole, :source_system_to_device_buffer!)
    @test !isdefined(FastMultipole, :target_system_from_device_buffer!)

    #--- (b) recenter! parity vs a fresh cache at the same bounds ---#

    # hierarchical (ell = 3) and flat (ell = 2) policies, hessian on and off
    for (ell, hessian) in ((3, true), (3, false), (2, true))
        sys = generate_gravitational(seed, 500)
        ref_sys = generate_gravitational(seed, 500)
        cache = RadixFMMCache(sys; expansion_order=4, ell, hessian, options=opts64)
        fmm!(sys, cache; scalar_potential=true, gradient=true, hessian)

        # drift every body by a shift that leaves the old box
        shift = SVector(1.5, 0.25, -0.75)
        for s in (sys, ref_sys), i in eachindex(s.bodies)
            b = s.bodies[i]
            s.bodies[i] = Body(b.position + shift, b.radius, b.strength)
        end
        @test_throws ArgumentError fmm!(sys, cache; scalar_potential=true,
            gradient=true, hessian)      # out-of-box throws, cache stays usable

        new_bounds = (SVector(0.4, -0.85, -1.85), 2.3)
        recenter!(cache, sys; bounds=new_bounds)
        sys.potential .= 0
        fmm!(sys, cache; scalar_potential=true, gradient=true, hessian)

        fresh = RadixFMMCache(ref_sys; expansion_order=4, ell, hessian,
            bounds=new_bounds, options=opts64)
        fmm!(ref_sys, fresh; scalar_potential=true, gradient=true, hessian)
        @test maximum(abs.(sys.potential .- ref_sys.potential)) < 1e-12

        # derived bounds with padding also runs and stays accurate
        recenter!(cache, sys; padding=0.1)
        @test cache.h0 > 0
        sys.potential .= 0
        fmm!(sys, cache; scalar_potential=true, gradient=true, hessian)
        @test maximum(abs.(sys.potential[5:7, :] .- ref_sys.potential[5:7, :])) < 5e-3
    end

    #--- (c) recenter! validation errors leave the cache untouched ---#

    sys = generate_gravitational(seed, 200)
    cache = RadixFMMCache(sys; expansion_order=4, ell=2, options=opts64)
    x_min0, h00, step0 = cache.x_min, cache.h0, cache.step
    @test_throws ArgumentError recenter!(cache, sys; padding=-0.1)
    @test_throws ArgumentError recenter!(cache, sys; bounds=((0, 0, 0), -1.0))
    @test_throws ArgumentError recenter!(cache, sys; bounds=((NaN, 0, 0), 1.0))
    other = generate_gravitational(seed, 100)
    @test_throws ArgumentError recenter!(cache, (sys, other))   # changed system count
    # bounds that exclude the bodies: construction throws, cache unmodified
    @test_throws ArgumentError recenter!(cache, sys; bounds=((10.0, 10.0, 10.0), 1.0))
    @test cache.x_min == x_min0 && cache.h0 == h00 && cache.step == step0
    # still usable after every rejected call
    fmm!(sys, cache; gradient=true)
    @test cache.step == step0 + 1
end
