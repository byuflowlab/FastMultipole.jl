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
    rho_t = 4.252   # shipped split-kernel default since Checkpoint D (2026-08-07)

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
            rho = inside ? TF(0.05 + 4.1 * rand(rng)) : TF(5.0 + 15.0 * rand(rng))
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

# Float64 erf-based regularized and singular U/J for one pair (theory §1),
# used by the stage B tests below as the pair-level truth.
function _ref_pair_uj(d, G, sigma)
    A = sqrt(2 / pi)
    r2 = dot(d, d)
    r = sqrt(r2)
    rho = r / sigma
    g = _ref_erf(rho / sqrt2) - A * rho * exp(-rho^2 / 2)
    gp = A * rho^2 * exp(-rho^2 / 2)
    cr3 = 1 / (4pi * r2 * r)
    crss = ((d[3] * G[2] - d[2] * G[3]) * cr3,
            (d[1] * G[3] - d[3] * G[1]) * cr3,
            (d[2] * G[1] - d[1] * G[2]) * cr3)
    vals(gv, av, bv) = (gv * crss[1], gv * crss[2], gv * crss[3],
        av * crss[1] * d[1], av * crss[2] * d[1] - bv * G[3], av * crss[3] * d[1] + bv * G[2],
        av * crss[1] * d[2] + bv * G[3], av * crss[2] * d[2], av * crss[3] * d[2] - bv * G[1],
        av * crss[1] * d[3] - bv * G[2], av * crss[2] * d[3] + bv * G[1], av * crss[3] * d[3])
    reg = vals(g, (rho * gp - 3g) / r2, -g * cr3)
    sing = vals(1.0, -3 / r2, -cr3)
    return reg, sing
end

@testset "two-pass nearfield stage B (task 032a)" begin

    seed = 20260807
    rho_t = 4.252   # shipped split-kernel default since Checkpoint D (2026-08-07)
    rho_c = 2.0
    tk = TwoPassVortex(; sigma_row=8)
    rk = RegularizedVortex(; sigma_row=8)
    sk = SingularVortex()

    #--- (a) pair-level identity: pass 1 is bitwise regularized inside rho_c and
    #    bitwise singular beyond; singular + deficit reproduces the regularized
    #    kernel across the correction shell; the deficit vanishes off-shell ---#

    for (TF, shell_tol) in ((Float64, 1e-14), (Float32, 2e-6))
        rng = MersenneTwister(seed)
        srcb = zeros(TF, 8, 1)
        for trial in 1:300
            sigma = TF(0.01 + 0.09 * rand(rng))
            zone = mod1(trial, 3)   # 1: inside rho_c, 2: shell, 3: beyond rho_t
            rho = zone == 1 ? TF(0.05 + 1.85 * rand(rng)) :
                  zone == 2 ? TF(2.05 + 2.1 * rand(rng)) :
                              TF(5.0 + 15.0 * rand(rng))
            u = normalize(randn(rng, 3))
            dx, dy, dz = TF.(u .* Float64(rho * sigma))
            r2 = dx * dx + dy * dy + dz * dz
            invr = inv(sqrt(r2))
            srcb[5:7, 1] .= randn(rng, TF, 3)
            srcb[8, 1] = sigma
            p1 = FastMultipole._direct_pair_ugh(tk, dx, dy, dz, r2, invr, srcb, 1)
            rho_eval = r2 * invr / srcb[8, 1]
            ge, he = FastMultipole._twopass_deficit_gh(tk, rho_eval)
            if zone == 1
                @test p1 === FastMultipole._direct_pair_ugh(rk, dx, dy, dz, r2,
                    invr, srcb, 1)
                @test ge === zero(TF) && he === zero(TF)
            else
                @test p1 === FastMultipole._direct_pair_ugh(sk, dx, dy, dz, r2,
                    invr, srcb, 1)
                if zone == 2
                    d = FastMultipole._vortex_pair_ugh(dx, dy, dz, r2, invr,
                        srcb[5, 1], srcb[6, 1], srcb[7, 1], ge, he)
                    reg = FastMultipole._direct_pair_ugh(rk, dx, dy, dz, r2,
                        invr, srcb, 1)
                    scale = maximum(abs.(reg))
                    @test maximum(abs.((p1 .+ d) .- reg)) / scale < shell_tol
                else
                    @test ge === zero(TF) && he === zero(TF)
                end
            end
        end
    end

    #--- (b) end-to-end host two-pass vs the erf-based regularized reference and
    #    vs regularized-everywhere at identical geometry; P=8 and P=4, F64/F32 ---#

    nv = 400
    for P in (8, 4), (TF, gtol) in ((Float64, 1e-3), (Float32, 3e-3))
        tol = P == 4 ? 10 * gtol : gtol   # P=4 truncation dominates (task 032 record)
        base_r = generate_vortex(seed, nv)
        base_t = generate_vortex(seed, nv)
        sigma = 0.02 .+ 0.02 .* rand(MersenneTwister(seed), nv)
        rsys = SmoothedVortex(base_r, sigma)
        tsys = TwoPassSmoothedVortex(SmoothedVortex(base_t, sigma))
        opts() = CUDARadixLifecycleOptions(; precision=TF,
            m2l_strategy=FastMultipole.ConcatenatedFixedZM2L())
        rcache = RadixFMMCache(rsys; expansion_order=P, ell=2, hessian=true,
            options=opts())
        tcache = RadixFMMCache(tsys; expansion_order=P, ell=2, hessian=true,
            options=opts())
        @test tcache.state.options.direct_kernel == TwoPassVortex(; sigma_row=8)
        fmm!(rsys, rcache; scalar_potential=false, gradient=true, hessian=true)
        fmm!(tsys, tcache; scalar_potential=false, gradient=true, hessian=true)
        U_ref, J_ref = _interface_regularized_direct(SmoothedVortex(
            generate_vortex(seed, nv), sigma))
        u_scale = maximum(abs.(U_ref)); j_scale = maximum(abs.(J_ref))
        # two-pass meets the same reference tolerance as regularized-everywhere
        @test maximum(abs.(base_t.gradient_stretching[1:3, :] .- U_ref)) / u_scale < tol
        @test maximum(abs.(base_t.potential[5:13, :] .- J_ref)) / j_scale < tol
        # and differs from regularized-everywhere only by the bounded tail +
        # accumulator rounding (same bound as the stage A partitioned delta)
        @test maximum(abs.(base_t.gradient_stretching[1:3, :] .-
            base_r.gradient_stretching[1:3, :])) / u_scale < 5e-4
        @test maximum(abs.(base_t.potential[5:13, :] .-
            base_r.potential[5:13, :])) / j_scale < 5e-4
    end

    #--- (c) conditioning guard: the rho_c hybrid must NOT show the Float32
    #    small-rho amplification (031a §6.1 table: hybrid holds ~1e-7 where the
    #    plain F32 two-pass loses up to 16%) ---#

    # pair level: hybrid total vs Float64 erf truth at the table's rho values
    for rho in (0.01, 0.02, 0.05, 0.1, 0.5)
        sigma = 0.03
        d = (rho * sigma) .* (0.36, 0.48, 0.8)
        G = (0.4, -0.3, 0.6)
        reg, _ = _ref_pair_uj(d, G, sigma)
        srcb = zeros(Float32, 8, 1)
        srcb[5:7, 1] .= Float32.(G)
        srcb[8, 1] = Float32(sigma)
        dx, dy, dz = Float32.(d)
        r2 = dx * dx + dy * dy + dz * dz
        invr = inv(sqrt(r2))
        p1 = FastMultipole._direct_pair_ugh(tk, dx, dy, dz, r2, invr, srcb, 1)
        ge, he = FastMultipole._twopass_deficit_gh(tk, r2 * invr / srcb[8, 1])
        @test ge == 0.0f0 && he == 0.0f0   # below rho_c: pass 1 owns the pair
        scale = maximum(abs.(reg))
        @test maximum(abs.(Float64.(p1[2:13]) .- collect(reg))) / scale < 1e-5
        # contrast: the plain (non-hybrid) F32 singular + deficit for the same
        # pair shows the amplification the hybrid exists to remove
        if rho <= 0.02
            s32 = FastMultipole._direct_pair_ugh(sk, dx, dy, dz, r2, invr, srcb, 1)
            g32, h32 = FastMultipole._gaussianerf_g_h(Float32(rho))
            gbar32 = 1.0f0 - g32
            d32 = FastMultipole._vortex_pair_ugh(dx, dy, dz, r2, invr,
                srcb[5, 1], srcb[6, 1], srcb[7, 1], -gbar32,
                (h32 + 3.0f0 * g32) + 3.0f0 * gbar32)
            naive = s32 .+ d32
            @test maximum(abs.(Float64.(naive[2:13]) .- collect(reg))) / scale > 1e-3
        end
    end

    # end to end: clusters of near-coincident particles (rho = 0.02) in Float32
    # stay at the pipeline's F32 floor instead of the ~1e-1 amplification
    rng = MersenneTwister(seed + 1)
    nb = 60
    pos = 0.05 .+ 0.9 .* rand(rng, 3, nb)
    npair = 20
    partners = zeros(3, npair)
    for k in 1:npair
        dir = normalize(randn(rng, 3))
        partners[:, k] .= pos[:, k] .+ 6.0e-4 .* dir   # rho = 0.02 at sigma = 0.03
    end
    posc = hcat(pos, partners)
    strength = 0.02 .* randn(rng, 3, nb + npair)
    sigma_c = fill(0.03, nb + npair)
    base_c = VortexParticles(posc, strength)
    csys = TwoPassSmoothedVortex(SmoothedVortex(base_c, sigma_c))
    ccache = RadixFMMCache(csys; expansion_order=8, ell=2, hessian=true,
        bounds=(SVector(0.0, 0.0, 0.0), 1.0),
        options=CUDARadixLifecycleOptions(; precision=Float32,
            m2l_strategy=FastMultipole.ConcatenatedFixedZM2L()))
    fmm!(csys, ccache; scalar_potential=false, gradient=true, hessian=true)
    U_cref, J_cref = _interface_regularized_direct(SmoothedVortex(
        VortexParticles(copy(posc), copy(strength)), sigma_c))
    @test maximum(abs.(base_c.gradient_stretching[1:3, :] .- U_cref)) /
        maximum(abs.(U_cref)) < 3e-3
    @test maximum(abs.(base_c.potential[5:13, :] .- J_cref)) /
        maximum(abs.(J_cref)) < 3e-3

    #--- (d) pass-2 reach: a shell pair whose cells lie OUTSIDE the primary near
    #    set still receives its deficit (a missing shell would leave the exact
    #    singular tail error at the target) ---#

    # deterministic geometry: ell=3 on the unit box (h_leaf = 0.125),
    # near_radius2 = 3, so cell offset (2,0,0) is M2L; sigma = 0.05 puts the
    # designated pair (r = 0.2, rho = 4.0) in the correction shell while the
    # pass-1 gate needs only rho_c*sigma = 0.1 < g_min*h_leaf = 0.125
    n_bg = 24
    rng = MersenneTwister(seed + 2)
    posr = zeros(3, n_bg + 2)
    posr[:, 1] .= (0.115, 0.0625, 0.0625)    # target T, cell (0,0,0)
    posr[:, 2] .= (0.315, 0.0625, 0.0625)    # source S, cell (2,0,0), r = 0.2
    posr[:, 3:end] .= 0.70 .+ 0.25 .* rand(rng, 3, n_bg)   # far background
    strr = 0.02 .* randn(rng, 3, n_bg + 2)
    strr[:, 1] .= (1.0, 0.0, 0.0)
    strr[:, 2] .= (0.0, 0.0, 1.0)
    sigr = fill(0.05, n_bg + 2)
    kwargs = (expansion_order=8, ell=3, hessian=true,
        bounds=(SVector(0.0, 0.0, 0.0), 1.0), near_radius2=3)
    opts64 = () -> CUDARadixLifecycleOptions(; precision=Float64,
        m2l_strategy=FastMultipole.ConcatenatedFixedZM2L())
    base_tp = VortexParticles(copy(posr), copy(strr))
    tpsys = TwoPassSmoothedVortex(SmoothedVortex(base_tp, sigr))
    tpcache = RadixFMMCache(tpsys; kwargs..., options=opts64())
    # the designated pair is genuinely beyond the primary near set
    ctx = tpcache.state.interaction_list
    @test all(sum(abs2, o) <= 3 for o in ctx.tables.near_offsets)
    # ... which is exactly why the partitioned kernel rejects this geometry
    # while the two-pass rho_c gate admits it (gate dispatch)
    part_sys = PartitionedSmoothedVortex(SmoothedVortex(
        VortexParticles(copy(posr), copy(strr)), sigr))
    @test_throws ArgumentError RadixFMMCache(part_sys; kwargs..., options=opts64())
    fmm!(tpsys, tpcache; scalar_potential=false, gradient=true, hessian=true)
    # singular-kernel run on identical bodies: the difference at T is exactly
    # the accumulated pass-2 deficit (T has no direct neighbors, no pair below
    # rho_c, and the far field is common to both runs)
    base_sing = VortexParticles(copy(posr), copy(strr))
    singcache = RadixFMMCache(base_sing; kwargs..., options=opts64())
    fmm!(base_sing, singcache; scalar_potential=false, gradient=true, hessian=true)
    dU = base_tp.gradient_stretching[1:3, 1] .- base_sing.gradient_stretching[1:3, 1]
    dJ = base_tp.potential[5:13, 1] .- base_sing.potential[5:13, 1]
    d = Float64.(posr[:, 1] .- posr[:, 2])
    reg, sing = _ref_pair_uj(d, Float64.(strr[:, 2]), 0.05)
    dU_ana = collect(reg[1:3] .- sing[1:3])
    dJ_ana = collect(reg[4:12] .- sing[4:12])
    @test norm(dU_ana) > 0 && norm(dJ_ana) > 0
    @test norm(dU .- dU_ana) < 2e-3 * norm(dU_ana)
    @test norm(dJ .- dJ_ana) < 2e-3 * norm(dJ_ana)

    #--- (e) validation and error paths ---#

    # trait conflicts with an explicit different kernel
    thin = SmoothedVortex(generate_vortex(seed, 100), fill(0.01, 100))
    @test_throws ArgumentError RadixFMMCache(thin; expansion_order=4, ell=2,
        options=CUDARadixLifecycleOptions(; precision=Float64,
            m2l_strategy=FastMultipole.ConcatenatedFixedZM2L(),
            direct_kernel=TwoPassVortex(; sigma_row=8)))
    # scalar body type rejected
    plain = generate_gravitational(seed, 100)
    @test_throws ArgumentError RadixFMMCache(plain; expansion_order=4, ell=2,
        options=CUDARadixLifecycleOptions(; precision=Float64,
            m2l_strategy=FastMultipole.ConcatenatedFixedZM2L(),
            direct_kernel=TwoPassVortex(; sigma_row=8)))
    # pass-1 adequacy gate binds at rho_c: sigma large enough that even the
    # rho_c reach fails must throw
    fat = TwoPassSmoothedVortex(SmoothedVortex(generate_vortex(seed, 400),
        fill(0.2, 400)))
    @test_throws ArgumentError RadixFMMCache(fat; expansion_order=4, ell=3,
        options=CUDARadixLifecycleOptions(; precision=Float64,
            m2l_strategy=FastMultipole.ConcatenatedFixedZM2L()))
    # host-only until stage C: the device path must refuse the kernel rather
    # than silently skip pass 2
    dev = TwoPassSmoothedVortex(SmoothedVortex(generate_vortex(seed, 100),
        fill(0.01, 100)))
    @test_throws ArgumentError RadixFMMCache(dev; expansion_order=4, ell=2,
        device=true, options=CUDARadixLifecycleOptions(; precision=Float64,
            m2l_strategy=FastMultipole.ConcatenatedFixedZM2L()))
    # constructor negatives
    @test_throws ArgumentError TwoPassVortex(; sigma_row=4)
    @test_throws ArgumentError TwoPassVortex(; sigma_row=8, rho_t=0.0)
    @test_throws ArgumentError TwoPassVortex(; sigma_row=8, rho_c=0.0)
    @test_throws ArgumentError TwoPassVortex(; sigma_row=8, rho_c=1.0)
    @test_throws ArgumentError TwoPassVortex(; sigma_row=8, rho_t=2.0, rho_c=2.0)
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

@testset "binned nearfield pair stream stage C (task 032a)" begin

    seed = 20260807
    rho_t = 4.789

    #--- (a) three-way bucket rule is safe: a brute-force body-pair sweep over
    #    random cell pairs shows bucket 1 contains only beyond-cutoff pairs and
    #    bucket 2 only inside-cutoff pairs, for both precisions ---#

    for TF in (Float64, Float32)
        rng = MersenneTwister(seed)
        h = TF(0.05)
        cut = TF(rho_t)
        n_checked = zeros(Int, 3)
        for trial in 1:600
            # bias a third of the trials to adjacent offsets and fat uniform σ
            # so the pure-regularized bucket is genuinely sampled
            near = trial % 3 == 0
            span = near ? (-1:1) : (-6:6)
            o = (rand(rng, span), rand(rng, span), rand(rng, span))
            nb = 6
            # random σ per source body (occasionally nonpositive padding)
            sigmas = near ? fill(TF(0.03 + 0.05 * rand(rng)), nb) :
                TF[rand(rng) < 0.1 ? zero(TF) : TF(0.005 + 0.05 * rand(rng))
                    for _ in 1:nb]
            smax = maximum(sigmas)
            smin = minimum(sigmas)
            b = FastMultipole._nearfield_pair_bucket(o..., h, cut, smax, smin)
            n_checked[Int(b)] += 1
            # bodies anywhere inside their cells
            tpos = [h .* (rand(rng, TF, 3)) for _ in 1:nb]
            spos = [h .* (TF.(o) .+ rand(rng, TF, 3)) for _ in 1:nb]
            # boundary rounding: the classifier compares squared TF quantities,
            # so allow a few ulps of slack on the exact-cutoff comparison (a
            # misrouted boundary pair changes the result by <= the §4 tail)
            slack = 8 * eps(TF)
            for i in 1:nb, j in 1:nb
                d = Float64.(tpos[i]) .- Float64.(spos[j])
                r = sqrt(sum(abs2, d))
                sig = Float64(sigmas[j])
                if b == Int32(1)
                    @test !(sig > 0 && r <= Float64(cut) * sig * (1 - slack))
                elseif b == Int32(2)
                    @test sig > 0 && r <= Float64(cut) * sig * (1 + slack)
                end
            end
        end
        # the sweep genuinely exercised all three buckets
        @test all(n_checked .> 0)
    end

    #--- (b) class-split evaluation == unbinned partitioned evaluation on the
    #    production host pair list: classify with the shared bucket rule, run
    #    the three branch-free/mixed kernels, compare (P = 4 and P = 8) ---#

    nv = 400
    for P in (4, 8), TF in (Float64, Float32)
        base = generate_vortex(seed, nv)
        sigma = 0.02 .+ 0.02 .* rand(MersenneTwister(seed), nv)
        psys = PartitionedSmoothedVortex(SmoothedVortex(base, sigma))
        cache = RadixFMMCache(psys; expansion_order=P, ell=2, hessian=true,
            options=CUDARadixLifecycleOptions(; precision=TF,
                m2l_strategy=FastMultipole.ConcatenatedFixedZM2L()))
        fmm!(psys, cache; scalar_potential=false, gradient=true, hessian=true)
        st = cache.state
        n_direct = st.counts.n_direct
        @test n_direct > 0
        grid = st.grid
        h_leaf = TF(2 * cache.h0 / (1 << cache.ell))
        dk = st.options.direct_kernel
        cut = TF(FastMultipole._pass1_regularized_cutoff(dk))
        # per-cell σ extrema (host mirror of _cuda_cell_sigma_kernel!)
        ncell = st.counts.n_cells
        smax = zeros(TF, ncell)
        smin = zeros(TF, ncell)
        for c in 1:ncell
            f = st.cell_ranges[1, c]
            cnt = st.cell_ranges[2, c]
            vals = st.source_bodies[8, f:(f + cnt - 1)]
            smax[c] = cnt == 0 ? zero(TF) : maximum(vals)
            smin[c] = cnt == 0 ? zero(TF) : minimum(vals)
        end
        buckets = [Int[] for _ in 1:3]
        for p in 1:n_direct
            t = st.direct_targets[p]
            s = st.direct_sources[p]
            ct = FastMultipole.morton_decode(grid.cell_keys[t], cache.ell)
            cs = FastMultipole.morton_decode(grid.cell_keys[s], cache.ell)
            o = ct .- cs
            b = FastMultipole._nearfield_pair_bucket(o[1], o[2], o[3], h_leaf,
                cut, smax[s], smin[s])
            push!(buckets[Int(b)], p)
        end
        @test sum(length, buckets) == n_direct
        # mixed-bucket fraction sanity: with overlap-2 σ at ell=2 the near set
        # over-covers, so pure-singular pairs must exist
        @test !isempty(buckets[1])
        ref = zeros(TF, 13, size(st.output, 2))
        FastMultipole._host_direct_pairs_functor_kernel!(dk, ref,
            st.source_bodies, st.cell_ranges, st.direct_targets,
            st.direct_sources, n_direct, Val(true))
        got = zeros(TF, 13, size(st.output, 2))
        kernels = (SingularVortex(),
            RegularizedVortex(; sigma_row=8, rho_t=Float64(cut)),
            dk)
        for b in 1:3
            isempty(buckets[b]) && continue
            FastMultipole._host_direct_pairs_functor_kernel!(kernels[b], got,
                st.source_bodies, st.cell_ranges, st.direct_targets[buckets[b]],
                st.direct_sources[buckets[b]], length(buckets[b]), Val(true))
        end
        scale = maximum(abs.(ref))
        tol = TF == Float64 ? 1e-13 : 2e-5   # summation order differs
        @test maximum(abs.(got .- ref)) < tol * scale
    end

    #--- (c) two-pass offset ball: gap-ascending, complete, prefix-prunable ---#

    offsets, gap2 = FastMultipole._twopass_offset_ball(5.35)
    @test issorted(gap2)
    @test size(offsets, 2) == length(gap2)
    # completeness against a brute-force enumeration at several live reaches
    for reach in (0.9, 2.6, 4.0, 5.35)
        want = Set{NTuple{3,Int}}()
        R = floor(Int, reach) + 1
        for oz in -R:R, oy in -R:R, ox in -R:R
            g2 = max(abs(ox) - 1, 0)^2 + max(abs(oy) - 1, 0)^2 +
                max(abs(oz) - 1, 0)^2
            g2 <= reach^2 && push!(want, (ox, oy, oz))
        end
        live = Set{NTuple{3,Int}}()
        for k in 1:length(gap2)
            Float64(gap2[k]) <= reach^2 &&
                push!(live, (Int(offsets[1, k]), Int(offsets[2, k]),
                    Int(offsets[3, k])))
        end
        @test live == want
    end
    # includes the self offset first
    @test gap2[1] == 0
    @test any(k -> offsets[:, k] == Int32[0, 0, 0], 1:length(gap2))
    @test_throws ArgumentError FastMultipole._twopass_offset_ball(-1.0)

    #--- (d) gate-derived reach capacity dominates every admissible geometry:
    #    while the pass-1 gate passes, rho_t·σ_max/h_leaf < (rho_t/rho_c)·g_min ---#

    tk = TwoPassVortex(; sigma_row=8)
    base = generate_vortex(seed, 300)
    sigma = fill(0.03, 300)
    tsys = TwoPassSmoothedVortex(SmoothedVortex(base, sigma))
    tcache = RadixFMMCache(tsys; expansion_order=4, ell=2, hessian=true,
        options=CUDARadixLifecycleOptions(; precision=Float64,
            m2l_strategy=FastMultipole.ConcatenatedFixedZM2L()))
    fmm!(tsys, tcache; scalar_potential=false, gradient=true, hessian=true)
    g_min = FastMultipole._leaf_stencil_min_gap(tcache)
    h_leaf = 2 * Float64(tcache.h0) / (1 << tcache.ell)
    sigma_max = maximum(tcache.state.source_bodies[8, 1:tcache.state.counts.n_bodies])
    # the gate passed at construction, so the pass-1 reach inequality holds ...
    @test tk.rho_c * sigma_max < g_min * h_leaf
    # ... and the pass-2 reach is inside the construction ball capacity
    reach_cap = (tk.rho_t / tk.rho_c) * g_min
    @test tk.rho_t * sigma_max / h_leaf < reach_cap
    # host caches carry no bin context and the reach check no-ops
    @test FastMultipole._cache_nearfield_bin_ctx(tcache) === nothing
    @test FastMultipole._twopass_device_reach_check(tcache, tk,
        Float64(sigma_max), h_leaf) === nothing

    #--- (e) flat-policy device TwoPassVortex is refused (the constructor calls
    #    _assert_device_kernel_policy; end-to-end coverage with CUDA present is
    #    in test/cuda_radix_nearfield_binning_test.jl) ---#

    err = try
        FastMultipole._assert_device_kernel_policy(true, tk, false)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("hierarchical stencil", sprint(showerror, err))
    # every other combination is admissible
    @test FastMultipole._assert_device_kernel_policy(true, tk, true) === nothing
    @test FastMultipole._assert_device_kernel_policy(false, tk, false) === nothing
    @test FastMultipole._assert_device_kernel_policy(true,
        PartitionedVortex(; sigma_row=8), false) === nothing
    # a device construction attempt still fails on this CUDA-less host, before
    # reaching the policy check
    @test_throws ArgumentError RadixFMMCache(tsys; expansion_order=4, ell=2,
        device=true, options=CUDARadixLifecycleOptions(; precision=Float64,
            m2l_strategy=FastMultipole.ConcatenatedFixedZM2L()))

    #--- (f) sub-Morton key semantics (mechanism a, host mirror of the device
    #    key kernel): keys land in [0, 8^sub), and sorting a cell's bodies by
    #    key groups them by sub-octant ---#

    let rng = MersenneTwister(seed), ell = 3, sub = 3
        x_min = SVector{3,Float64}(0, 0, 0)
        h0 = 0.5   # box [0,1]^3
        Gs = 1 << (ell + sub)
        m = (1 << sub) - 1
        delta = 2 * h0 / Gs
        subkey(x) = begin
            c = ntuple(q -> clamp(Int(floor((x[q] - x_min[q]) / delta)), 0,
                Gs - 1) & m, 3)
            key = 0
            for bit in 0:(sub - 1)
                key |= ((c[1] >> bit) & 1) << (3 * bit)
                key |= ((c[2] >> bit) & 1) << (3 * bit + 1)
                key |= ((c[3] >> bit) & 1) << (3 * bit + 2)
            end
            key
        end
        cell_lo = SVector{3,Float64}(0.25, 0.5, 0.125)   # one ell=3 cell
        w = 2 * h0 / (1 << ell)
        pts = [cell_lo .+ w .* rand(rng, 3) for _ in 1:200]
        keys = subkey.(pts)
        @test all(0 .<= keys .< 8^sub)
        # bodies sharing a sub-octant share a key; distinct sub-octants of the
        # same parent octant stay adjacent under the Morton order
        order = sortperm(keys)
        sorted = pts[order]
        # spatial coherence: mean nearest-neighbor index distance in sorted
        # order is far below the random-order expectation
        d(i) = norm(sorted[i] .- sorted[i + 1])
        mean_sorted = sum(d, 1:199) / 199
        shuffled = pts[randperm(rng, 200)]
        ds(i) = norm(shuffled[i] .- shuffled[i + 1])
        mean_rand = sum(ds, 1:199) / 199
        @test mean_sorted < 0.7 * mean_rand
    end
end

@testset "cheapened g/h modes (task 037f)" begin
    # budgets and mode sizing: theory/nearfield-kernel-cheapening-budget.md
    # (data/kernel_splitting/fm037f_budget.csv); pointwise gates below are the
    # measured mode errors with ~2x margin

    seed = 20260814

    # BigFloat series reference (same construction as the stage-2 testset)
    function gh_big37f(rho)
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
    A37 = sqrt(2 / pi)
    g_ref37(rho) = _ref_erf(rho / sqrt(2)) - A37 * rho * exp(-rho^2 / 2)
    outer_budget = 0.5e-3 * g_ref37(2.0)     # 3.69e-4, the shipped outer gate

    #--- (a) defaults and validation ---#

    # default = :fp32 for Float64 configurations (user-approved 2026-08-14,
    # task 037f Work Record); bitwise no-op on Float32 configurations.
    # :shipped stays available as the control/opt-out (asserted below by the
    # bitwise-identity tests, which set the mode explicitly).
    @test FastMultipole.CUDA_NEARFIELD_GH_MODE[] === :fp32
    @test FastMultipole._validated_host_gh_mode() === :fp32
    @test :shipped in FastMultipole.NEARFIELD_GH_MODES
    FastMultipole.CUDA_NEARFIELD_GH_MODE[] = :bogus
    try
        @test_throws ArgumentError FastMultipole._validated_host_gh_mode()
    finally
        FastMultipole.CUDA_NEARFIELD_GH_MODE[] = :shipped
    end
    # the host reference path maps :lut -> :shipped (documented fallback)
    FastMultipole.CUDA_NEARFIELD_GH_MODE[] = :lut
    try
        @test FastMultipole._validated_host_gh_mode() === :shipped
    finally
        FastMultipole.CUDA_NEARFIELD_GH_MODE[] = :shipped
    end

    #--- (b) per-mode pointwise bounds vs the references ---#

    # :reduced — 12-term series (measured 4.9e-6 rel; gate 1e-5), outer
    # branch unchanged, singular limit exact
    for TF in (Float64, Float32)
        emax = 0.0
        for rho in range(1e-3, 2.0, length=501)
            g, h = FastMultipole._gaussianerf_g_h(TF(rho), Val(:reduced))
            gb, hb = gh_big37f(rho)
            emax = max(emax, abs(g / gb - 1), abs(h / hb - 1))
        end
        @test emax < 1e-5
        for rho in (2.5, 4.0, 40.0)
            @test FastMultipole._gaussianerf_g_h(TF(rho), Val(:reduced)) ===
                FastMultipole._gaussianerf_g_h(TF(rho))
        end
        g, h = FastMultipole._gaussianerf_g_h(TF(40), Val(:reduced))
        @test g == one(TF) && h == -TF(3)
    end

    # fp32 modes on Float64 arguments: Float32-quality series (shipped 13-term
    # gate 2e-6; reduced 1e-5), outer within the shipped absolute budget
    for (mode, rtol) in ((:fp32, 2e-6), (:reduced_fp32, 1e-5))
        emax_series = 0.0
        for rho in range(1e-3, 2.0 - 1e-6, length=501)
            g, h = FastMultipole._gaussianerf_g_h(Float64(rho), Val(mode))
            gb, hb = gh_big37f(rho)
            emax_series = max(emax_series, abs(g / gb - 1), abs(h / hb - 1))
        end
        @test emax_series < rtol
        emax_outer = 0.0
        for rho in range(2.001, 40.0, length=501)
            g, _ = FastMultipole._gaussianerf_g_h(Float64(rho), Val(mode))
            emax_outer = max(emax_outer, abs(g - g_ref37(rho)))
        end
        @test emax_outer < outer_budget
        g, h = FastMultipole._gaussianerf_g_h(40.0, Val(mode))
        @test g == 1.0 && h == -3.0
    end

    #--- (c) :lut table + interpolation: relative accuracy over the whole
    #    domain (normalized G/H preserves it near rho -> 0), exact singular
    #    beyond, branch continuity ---#

    for rho_t in (3.668, 4.252)
        tab = FastMultipole._build_gh_lut(rho_t)
        @test size(tab) == (2, FastMultipole._NF_GH_LUT_N)
        for TF in (Float64, Float32)
            # domain end exactly as the device computes it (T(rho_t)^2)
            x_max = TF(rho_t)^2
            emax = 0.0
            for rho in range(1e-3, rho_t - 1e-4, length=701)
                # delta vs shipped: the LUT cheapens the shipped evaluator
                g, h = FastMultipole._gh_from_lut(tab, TF(rho), x_max)
                g0, h0 = FastMultipole._gaussianerf_g_h(rho)
                # skip the one interpolation cell containing the rho = 2
                # branch discontinuity of the shipped evaluator itself
                abs(rho - 2.0) < 2 * Float64(x_max) / FastMultipole._NF_GH_LUT_N && continue
                emax = max(emax, abs(Float64(g) / g0 - 1), abs(Float64(h) / h0 - 1))
            end
            @test emax < 2e-5
            # half-open domain end: singular exactly at and beyond rho_t
            g, h = FastMultipole._gh_from_lut(tab, TF(rho_t), x_max)
            @test g == one(TF) && h == -TF(3)
        end
    end

    #--- (d) :shipped bitwise on the functor entry points, all kernels ---#

    rng = MersenneTwister(seed)
    for TF in (Float64, Float32)
        srcb = zeros(TF, 8, 1)
        for kernel in (RegularizedVortex(; sigma_row=8),
                PartitionedVortex(; sigma_row=8),
                TwoPassVortex(; sigma_row=8), SingularVortex())
            for trial in 1:50
                srcb[5:7, 1] .= randn(rng, TF, 3)
                srcb[8, 1] = TF(0.01 + 0.09 * rand(rng))
                d = TF.(0.3 .* randn(rng, 3))
                r2 = sum(abs2, d)
                invr = inv(sqrt(r2))
                @test FastMultipole._direct_pair_ugh(kernel, d[1], d[2], d[3],
                    r2, invr, srcb, 1, Val(:shipped)) ===
                    FastMultipole._direct_pair_ugh(kernel, d[1], d[2], d[3],
                    r2, invr, srcb, 1)
                @test FastMultipole._direct_pair_ug(kernel, d[1], d[2], d[3],
                    r2, invr, srcb, 1, Val(:shipped)) ===
                    FastMultipole._direct_pair_ug(kernel, d[1], d[2], d[3],
                    r2, invr, srcb, 1)
            end
        end
        # fp32 modes on Float32 configurations are the documented no-ops
        srcb[5:7, 1] .= randn(rng, TF, 3)
        srcb[8, 1] = TF(0.05)
        pk = PartitionedVortex(; sigma_row=8)
        d = TF.((0.06, 0.02, 0.01))
        r2 = sum(abs2, d); invr = inv(sqrt(r2))
        if TF === Float32
            @test FastMultipole._direct_pair_ugh(pk, d..., r2, invr, srcb, 1,
                Val(:fp32)) ===
                FastMultipole._direct_pair_ugh(pk, d..., r2, invr, srcb, 1)
            @test FastMultipole._direct_pair_ugh(pk, d..., r2, invr, srcb, 1,
                Val(:reduced_fp32)) ===
                FastMultipole._direct_pair_ugh(pk, d..., r2, invr, srcb, 1)
        end
    end

    #--- (e) host end-to-end threading: fmm! under each host mode vs :shipped
    #    (delta gated at the mapped budget scale; :lut host == :shipped) ---#

    nv = 300
    sigma = 0.02 .+ 0.02 .* rand(MersenneTwister(seed), nv)
    function run_mode37f(mode)
        sys = PartitionedSmoothedVortex(SmoothedVortex(generate_vortex(seed, nv),
            copy(sigma)))
        cache = RadixFMMCache(sys; expansion_order=4, ell=2, hessian=true,
            options=CUDARadixLifecycleOptions(; precision=Float64,
                m2l_strategy=FastMultipole.ConcatenatedFixedZM2L()))
        old = FastMultipole.CUDA_NEARFIELD_GH_MODE[]
        FastMultipole.CUDA_NEARFIELD_GH_MODE[] = mode
        try
            fmm!(sys, cache; scalar_potential=false, gradient=true, hessian=true)
        finally
            FastMultipole.CUDA_NEARFIELD_GH_MODE[] = old
        end
        inner = sys.smoothed.inner
        return copy(inner.gradient_stretching[1:3, :]), copy(inner.potential[5:13, :])
    end
    U0, J0 = run_mode37f(:shipped)
    u_scale = maximum(abs.(U0)); j_scale = maximum(abs.(J0))
    for (mode, tol) in ((:reduced, 1e-4), (:fp32, 1e-5), (:reduced_fp32, 1e-4))
        U, J = run_mode37f(mode)
        @test maximum(abs.(U .- U0)) / u_scale < tol
        @test maximum(abs.(J .- J0)) / j_scale < tol
    end
    Ul, Jl = run_mode37f(:lut)     # host falls back to :shipped
    @test Ul == U0 && Jl == J0
end

@testset "shipped nearfield defaults (task 032a Checkpoint D, 2026-08-07)" begin
    # user-approved defaults: PartitionedVortex is the recommended σ-carrying
    # vortex nearfield; the split kernels default to the §6.4 RMS radius
    # rho_t = 4.252 (confirmed on both Integration Phase cases in Stage D);
    # RegularizedVortex keeps the §4 per-pair radius as the fallback
    @test PartitionedVortex(; sigma_row=8).rho_t == 4.252
    @test TwoPassVortex(; sigma_row=8).rho_t == 4.252
    @test TwoPassVortex(; sigma_row=8).rho_c == 2.0
    @test RegularizedVortex(; sigma_row=8).rho_t == 4.789
    # the plain-vortex default is unchanged (no σ row exists to regularize on)
    @test FastMultipole._default_direct_kernel(Point{Vortex}) === SingularVortex()
    # trait-driven caches pick up the new default end-to-end (P = 4 and P = 8)
    for P in (4, 8)
        base = generate_vortex(20260807, 200)
        psys = PartitionedSmoothedVortex(SmoothedVortex(base, fill(0.02, 200)))
        cache = RadixFMMCache(psys; expansion_order=P, ell=2, hessian=true,
            options=CUDARadixLifecycleOptions(; precision=Float64,
                m2l_strategy=FastMultipole.ConcatenatedFixedZM2L()))
        dk = cache.state.options.direct_kernel
        @test dk isa PartitionedVortex && dk.rho_t == 4.252
        step0 = cache.step
        fmm!(psys, cache; scalar_potential=false, gradient=true, hessian=true)
        @test cache.step == step0 + 1
    end
end
