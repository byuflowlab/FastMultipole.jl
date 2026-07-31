using FastMultipole
using FastMultipole.StaticArrays
using Random
using Test

if !isdefined(@__MODULE__, :generate_gravitational)
    include("gravitational.jl")
end

function _radix_direct_reference(seed, n; kwargs...)
    ref = generate_gravitational(seed, n; kwargs...)
    FastMultipole.direct!(ref; scalar_potential=true, gradient=true)
    return ref
end

_radix_potential_error(sys, ref) = maximum(abs.(sys.potential[1, :] .- ref.potential[1, :]))
_radix_gradient_error(sys, ref) = maximum(abs.(sys.potential[5:7, :] .- ref.potential[5:7, :]))

@testset "radix fmm! production integration (task 023)" begin

    #--- (a) radix fmm! vs direct! through the public API ---#

    n = 2000
    seed = 20260714
    sys = generate_gravitational(seed, n)
    ref = _radix_direct_reference(seed, n)
    cache = RadixFMMCache(sys; expansion_order=8, ell=3)
    result = fmm!(sys, cache; scalar_potential=true, gradient=true)
    @test result === cache
    # tolerances scaled for constant-P truncation of the analytic stencil
    @test _radix_potential_error(sys, ref) < 1e-6
    @test _radix_gradient_error(sys, ref) < 1e-4

    #--- (b) radix vs legacy fmm! parity ---#

    legacy = generate_gravitational(seed, n)
    fmm!(legacy; expansion_order=10, scalar_potential=true, gradient=true, multipole_acceptance=0.4)
    # both approximate the same direct sums; their difference is bounded by the
    # sum of the two truncation errors
    @test maximum(abs.(sys.potential[1, :] .- legacy.potential[1, :])) < 2e-6
    @test maximum(abs.(sys.potential[5:7, :] .- legacy.potential[5:7, :])) < 2e-4

    #--- (c) point-mass 1/r convergence with monotone error decrease ---#

    n_far = 400
    far_seed = 20260701
    far_ref = _radix_direct_reference(far_seed, n_far)
    potential_errors = Float64[]
    gradient_errors = Float64[]
    for P in (4, 8, 12)
        far_sys = generate_gravitational(far_seed, n_far)
        far_cache = RadixFMMCache(far_sys; expansion_order=P, ell=3)
        fmm!(far_sys, far_cache; scalar_potential=true, gradient=true)
        push!(potential_errors, _radix_potential_error(far_sys, far_ref))
        push!(gradient_errors, _radix_gradient_error(far_sys, far_ref))
    end
    @test potential_errors[3] < potential_errors[2] < potential_errors[1]
    @test gradient_errors[3] < gradient_errors[2] < gradient_errors[1]

    #--- (d) error paths ---#

    err_sys = generate_gravitational(1, 100)
    err_cache = RadixFMMCache(err_sys; expansion_order=4, ell=2)
    @test_throws ArgumentError fmm!(err_sys, err_cache; hessian=true)
    @test_throws ArgumentError fmm!(err_sys, err_cache; lamb_helmholtz=true)
    other_sys = generate_gravitational(2, 100)
    @test_throws ArgumentError fmm!(other_sys, err_sys, err_cache)
    too_big = generate_gravitational(3, 150)
    @test_throws ArgumentError fmm!(too_big, err_cache)
    # bodies leaving the fixed box throw rather than silently rebuilding geometry
    box_sys = generate_gravitational(4, 100)
    box_cache = RadixFMMCache(box_sys; expansion_order=4, ell=2,
        bounds=(SVector(0.0, 0.0, 0.0), 1.0))
    box_sys.bodies[1] = Body(SVector(1.5, 0.5, 0.5), box_sys.bodies[1].radius,
        box_sys.bodies[1].strength)
    @test_throws ArgumentError fmm!(box_sys, box_cache)
    # constructor rejections
    @test_throws ArgumentError RadixFMMCache(err_sys; max_n_bodies=10)
    @test_throws ArgumentError RadixFMMCache(err_sys;
        options=CUDARadixLifecycleOptions(; m2l_strategy=FastMultipole.SharedRotationM2L()))
    @test_throws Exception RadixFMMCache(err_sys; device=true)   # no CUDA on test host

    #--- (e) Float32 and Lamb-Helmholtz variants ---#

    n32 = 800
    sys32 = generate_gravitational(seed, n32)
    ref32 = _radix_direct_reference(seed, n32)
    cache32 = RadixFMMCache(sys32; expansion_order=6, ell=3,
        options=CUDARadixLifecycleOptions(; precision=Float32,
            m2l_strategy=FastMultipole.ConcatenatedFixedZM2L()))
    fmm!(sys32, cache32; scalar_potential=true, gradient=true)
    @test _radix_potential_error(sys32, ref32) < 1f-3
    @test _radix_gradient_error(sys32, ref32) < 1f-1

    sys_lh = generate_gravitational(seed, n32)
    cache_lh = RadixFMMCache(sys_lh; expansion_order=8, ell=3, lamb_helmholtz=true)
    @test cache_lh isa RadixFMMCache{Float64,true}
    fmm!(sys_lh, cache_lh; scalar_potential=false, gradient=true)
    @test _radix_gradient_error(sys_lh, ref32) < 1e-4

    #--- (f) DEBUG[]-on factored-operator host run (016b guard) ---#

    saved_debug = FastMultipole.DEBUG[]
    try
        FastMultipole.DEBUG[] = true
        sys_f = generate_gravitational(seed, 500)
        ref_f = _radix_direct_reference(seed, 500)
        cache_f = RadixFMMCache(sys_f; expansion_order=8, ell=3,
            options=CUDARadixLifecycleOptions(; operator=FactoredRotationM2L(),
                m2l_strategy=FastMultipole.ConcatenatedFixedZM2L()))
        fmm!(sys_f, cache_f; scalar_potential=true, gradient=true)
        @test _radix_potential_error(sys_f, ref_f) < 1e-6
    finally
        FastMultipole.DEBUG[] = saved_debug
    end

    #--- (f2) genuine grouped factored resident M2L parity ---#

    for P in (4, 8), (TF, LH) in ((Float64, false), (Float64, true),
                                  (Float32, false), (Float32, true))
        nf = 300
        factored = generate_gravitational(seed + 9, nf)
        concat = generate_gravitational(seed + 9, nf)
        direct = _radix_direct_reference(seed + 9, nf)
        # The grouped-factored resident plan is a FLAT-path structure: under the
        # task-027 hierarchical default the factored selection deliberately routes
        # through the bounded concat engine (mirroring the host). Pin both caches
        # to the flat classifier so this block keeps comparing factored vs concat.
        factored_cache = RadixFMMCache(factored; expansion_order=P, ell=3,
            lamb_helmholtz=LH, stencil_epsilon=1e-4,
            options=CUDARadixLifecycleOptions(; precision=TF,
                operator=FactoredRotationM2L(),
                m2l_strategy=FastMultipole.ConcatenatedFixedZM2L()))
        concat_cache = RadixFMMCache(concat; expansion_order=P, ell=3,
            lamb_helmholtz=LH, stencil_epsilon=1e-4,
            options=CUDARadixLifecycleOptions(; precision=TF,
                operator=MaterializedYRotationM2L(),
                m2l_strategy=FastMultipole.ConcatenatedFixedZM2L()))
        @test factored_cache.state.scratch.m2l_concat isa FastMultipole.ResidentM2LFactoredPlan
        fmm!(factored, factored_cache; scalar_potential=!LH, gradient=true)
        fmm!(concat, concat_cache; scalar_potential=!LH, gradient=true)
        if P == 8
            # absolute accuracy only holds at the stencil's design order
            direct_p_tol = TF === Float64 ? 1e-6 : 1f-3
            direct_g_tol = TF === Float64 ? 1e-4 : 1f-1
            !LH && @test _radix_potential_error(factored, direct) < direct_p_tol
            @test _radix_gradient_error(factored, direct) < direct_g_tol
        end
        # factored-vs-concat parity shares the truncation error, so it holds at all P
        !LH && @test maximum(abs.(factored.potential[1, :] .- concat.potential[1, :])) < 2e-6
        @test maximum(abs.(factored.potential[5:7, :] .- concat.potential[5:7, :])) < 2e-4
    end

    #--- (f3) factored GEMM branch (023a crossover) parity + allocation ---#

    # force every degree block through the BLAS branch so the crossover code path is
    # covered regardless of measured default thresholds and observed class widths
    saved_cols = FastMultipole.FACTORED_Y_GEMM_MIN_COLS[]
    saved_dim = FastMultipole.FACTORED_Y_GEMM_MIN_DIM[]
    try
        for P in (4, 8), LH in (false, true)
            nf = 300
            gemm_direct = _radix_direct_reference(seed + 9, nf)
            mk_cache(sys) = RadixFMMCache(sys; expansion_order=P, ell=3,
                lamb_helmholtz=LH,
                options=CUDARadixLifecycleOptions(; operator=FactoredRotationM2L(),
                    m2l_strategy=FastMultipole.ConcatenatedFixedZM2L()))
            # scalar branch forced everywhere
            FastMultipole.FACTORED_Y_GEMM_MIN_COLS[] = typemax(Int)
            FastMultipole.FACTORED_Y_GEMM_MIN_DIM[] = typemax(Int)
            scalar_sys = generate_gravitational(seed + 9, nf)
            fmm!(scalar_sys, mk_cache(scalar_sys); scalar_potential=!LH, gradient=true)
            # BLAS branch forced everywhere
            FastMultipole.FACTORED_Y_GEMM_MIN_COLS[] = 1
            FastMultipole.FACTORED_Y_GEMM_MIN_DIM[] = 1
            gemm_sys = generate_gravitational(seed + 9, nf)
            gemm_cache = mk_cache(gemm_sys)
            fmm!(gemm_sys, gemm_cache; scalar_potential=!LH, gradient=true)
            # scalar and GEMM branches evaluate the same operators; they may differ
            # only by floating-point reassociation inside the BLAS products
            !LH && @test maximum(abs.(gemm_sys.potential[1, :] .-
                scalar_sys.potential[1, :])) < 1e-8
            @test maximum(abs.(gemm_sys.potential[5:7, :] .-
                scalar_sys.potential[5:7, :])) < 1e-7
            if P == 8
                !LH && @test _radix_potential_error(gemm_sys, gemm_direct) < 1e-6
                @test _radix_gradient_error(gemm_sys, gemm_direct) < 1e-4
            end
            FastMultipole._launch_resident_m2l!(gemm_cache.state)
            @test (@allocated FastMultipole._launch_resident_m2l!(gemm_cache.state)) <= 64 * 1024
        end
    finally
        FastMultipole.FACTORED_Y_GEMM_MIN_COLS[] = saved_cols
        FastMultipole.FACTORED_Y_GEMM_MIN_DIM[] = saved_dim
    end

    #--- (g) two-system tuple ---#

    sys_a = generate_gravitational(seed, 700)
    sys_b = generate_gravitational(seed + 1, 500)
    ref_a = generate_gravitational(seed, 700)
    ref_b = generate_gravitational(seed + 1, 500)
    FastMultipole.direct!((ref_a, ref_b), (ref_a, ref_b);
        scalar_potential=true, gradient=true)
    cache_ab = RadixFMMCache((sys_a, sys_b); expansion_order=8, ell=3)
    fmm!((sys_a, sys_b), (sys_a, sys_b), cache_ab; scalar_potential=true, gradient=true)
    @test _radix_potential_error(sys_a, ref_a) < 1e-6
    @test _radix_gradient_error(sys_a, ref_a) < 1e-4
    @test _radix_potential_error(sys_b, ref_b) < 1e-6
    @test _radix_gradient_error(sys_b, ref_b) < 1e-4
end
