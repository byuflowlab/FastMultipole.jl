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

    #--- (h) measured option defaults (tasks 024 / 028) ---#

    # Precision depends only on the expansion order; the strategy also gates on the
    # Lamb-Helmholtz channel, the platform, and the dense operator footprint.
    for (eo, TF) in ((1, Float32), (3, Float32), (4, Float64), (12, Float64))
        @test FastMultipole._default_radix_precision(eo) === TF
    end
    _sel(eo, LH, device; nclasses=874) = nameof(typeof(
        FastMultipole._default_radix_m2l_strategy(
            FastMultipole._default_radix_precision(eo), eo, LH, device, nclasses,
            FastMultipole._dense_m2m_dof(
                FastMultipole.OperatorBasisInfo(
                    FastMultipole.CompressedComplexBasis(), eo, Val(LH)), Val(LH)))))
    for device in (false, true)
        @test _sel(3, false, device) === :DenseTranslationM2L      # literature P = 4
        @test _sel(3, true, device) === :DenseTranslationM2L
        @test _sel(7, false, device) === :DenseTranslationM2L      # P = 8, LH off
        @test _sel(8, false, device) === :PrecomputedFactoredYM2L  # P >= 12
        @test _sel(11, true, device) === :PrecomputedFactoredYM2L
    end
    # P = 8 with Lamb-Helmholtz is the one measured platform split.
    @test _sel(7, true, false) === :DenseTranslationM2L
    @test _sel(7, true, true) === :PrecomputedFactoredYM2L
    # A dense operator payload over its gate falls back instead of throwing.
    @test _sel(7, false, true; nclasses=10^7) === :PrecomputedFactoredYM2L

    # The resolved choice must reach the cache, and each strategy must carry the
    # rotation operator its plan is built from.
    auto_sys = generate_gravitational(seed + 2, 400)
    auto_cache = RadixFMMCache(auto_sys; expansion_order=3, ell=4)
    @test auto_cache.state.options.precision === Float32
    @test auto_cache.state.options.m2l_strategy isa DenseTranslationM2L
    @test auto_cache.state.options.operator isa MaterializedYRotationM2L
    hi_sys = generate_gravitational(seed + 3, 400)
    hi_cache = RadixFMMCache(hi_sys; expansion_order=8, ell=3)
    @test hi_cache.state.options.precision === Float64
    @test hi_cache.state.options.m2l_strategy isa PrecomputedFactoredYM2L
    @test hi_cache.state.options.operator isa FactoredRotationM2L
    # Explicit options bypass the rules entirely.
    exp_sys = generate_gravitational(seed + 4, 400)
    exp_cache = RadixFMMCache(exp_sys; expansion_order=3, ell=4,
        options=CUDARadixLifecycleOptions(; precision=Float64,
            m2l_strategy=ConcatenatedFixedZM2L()))
    @test exp_cache.state.options.precision === Float64
    @test exp_cache.state.options.m2l_strategy isa ConcatenatedFixedZM2L

    # Float32 at literature P = 4 must cost no measurable accuracy against Float64
    # at the same geometry: the stencil truncation error dominates (task 028 §4.3).
    f32 = generate_gravitational(seed + 5, 1500)
    f64 = generate_gravitational(seed + 5, 1500)
    ref32 = _radix_direct_reference(seed + 5, 1500)
    c32 = RadixFMMCache(f32; expansion_order=3, ell=4)
    c64 = RadixFMMCache(f64; expansion_order=3, ell=4,
        options=CUDARadixLifecycleOptions(; precision=Float64,
            m2l_strategy=DenseTranslationM2L()))
    fmm!(f32, c32; scalar_potential=true, gradient=true)
    fmm!(f64, c64; scalar_potential=true, gradient=true)
    e32 = _radix_gradient_error(f32, ref32)
    e64 = _radix_gradient_error(f64, ref32)
    @test e32 <= 1.05 * e64
end

@testset "radix rectangular bounds (task 037 stages 1-2)" begin

    seed = 20260813
    origin = SVector(0.0, 0.0, 0.0)
    # stretch the unit cloud to x in [0, 4]: aspect 4:1:1
    stretch(bodies) = (bodies[1, :] .*= 4.0; bodies)

    #--- (a) cube-regression gate: vector (L, L, L) bounds == scalar L bounds ---#

    for P in (3, 8)   # literature P = 4 (standing rule) plus one higher order
        a = generate_gravitational(seed, 800)
        b = generate_gravitational(seed, 800)
        ca = RadixFMMCache(a; expansion_order=P, ell=3, bounds=(origin, 1.0))
        cb = RadixFMMCache(b; expansion_order=P, ell=3,
            bounds=(origin, (1.0, 1.0, 1.0)))
        @test ca.ell_axes == cb.ell_axes == SVector(3, 3, 3)
        @test ca.box_extent === cb.box_extent
        @test ca.h0 === cb.h0
        @test ca.x_min === cb.x_min
        @test ca.max_cells == cb.max_cells
        @test ca.max_nodes == cb.max_nodes
        @test ca.route_capacity == cb.route_capacity
        @test ca.direct_capacity == cb.direct_capacity
        sa, sb = ca.state, cb.state
        # n_routes on the hierarchical path is cumulative window telemetry and
        # the route buffers hold only the last generated window (with no valid-
        # prefix marker), so route parity is gated by the counts here and the
        # bitwise output equality below
        @test sa.counts.n_routes == sb.counts.n_routes
        @test sa.counts.n_direct == sb.counts.n_direct
        nd = sa.counts.n_direct
        @test sa.direct_targets[1:nd] == sb.direct_targets[1:nd]
        @test sa.direct_sources[1:nd] == sb.direct_sources[1:nd]
        fmm!(a, ca; scalar_potential=true, gradient=true)
        fmm!(b, cb; scalar_potential=true, gradient=true)
        @test a.potential == b.potential   # bitwise
    end

    #--- (b) rectangular host cache on an elongated cloud ---#

    nrect = 1200
    rect_bounds = (origin, (4.0, 1.0, 1.0))
    rect_ref = _radix_direct_reference(seed + 1, nrect; bodies_fun=stretch)
    for (P, p_tol, g_tol) in ((3, 2e-3, 2e-1), (8, 1e-6, 1e-4))
        rect = generate_gravitational(seed + 1, nrect; bodies_fun=stretch)
        rcache = RadixFMMCache(rect; expansion_order=P, ell=4, bounds=rect_bounds)
        @test rcache.ell_axes == SVector(4, 2, 2)
        @test rcache.box_extent == SVector(4.0, 1.0, 1.0)
        @test rcache.h0 == 2.0
        # capacity accounting: per-axis products, not the cubic 8^L bounds;
        # levels below the flat-top root R=2 are trimmed (task 037 stage 3)
        @test rcache.root_level == 2
        @test rcache.max_cells == 256          # 2^(4+2+2)
        @test rcache.max_nodes == 4 + 32 + 256 # levels 2:4 only
        fmm!(rect, rcache; scalar_potential=true, gradient=true)
        @test _radix_potential_error(rect, rect_ref) < p_tol
        @test _radix_gradient_error(rect, rect_ref) < g_tol
    end

    #--- (c) snap-up of a non-power-of-two aspect ---#

    snap = generate_gravitational(seed + 2, 300; bodies_fun=stretch)
    scache = RadixFMMCache(snap; expansion_order=3, ell=4,
        bounds=(origin, (4.0, 1.3, 1.0)))
    @test scache.ell_axes == SVector(4, 3, 2)
    @test scache.box_extent == SVector(4.0, 2.0, 1.0)

    #--- (d) per-axis out-of-box contract ---#

    oob = generate_gravitational(seed + 3, 100; bodies_fun=stretch)
    ocache = RadixFMMCache(oob; expansion_order=3, ell=4, bounds=rect_bounds)
    # inside the virtual cube [0, 4]^3 but outside the rectangular box in y
    oob.bodies[1] = Body(SVector(0.5, 2.5, 0.5), oob.bodies[1].radius,
        oob.bodies[1].strength)
    @test_throws ArgumentError fmm!(oob, ocache)

    #--- (e) rectangular host recenter! (task 037 stage 2) ---#

    # aspect ~ 4 : 0.8 : 0.8 so the tight-extent ratio sits comfortably inside
    # the (1/8, 1/4] band that resolves ell_axes = (4, 2, 2) at ell = 4 — the
    # derived-bounds recenter! below must reproduce it deterministically
    stretch2(bodies) = (bodies[1, :] .*= 4.0; bodies[2, :] .*= 0.8;
        bodies[3, :] .*= 0.8; bodies)
    nrec = 600
    rec = generate_gravitational(seed + 5, nrec; bodies_fun=stretch2)
    rec_direct = _radix_direct_reference(seed + 5, nrec; bodies_fun=stretch2)
    rec_cache = RadixFMMCache(rec; expansion_order=3, ell=4,
        bounds=(origin, (4.0, 0.8, 0.8)))
    @test rec_cache.ell_axes == SVector(4, 2, 2)
    @test rec_cache.box_extent == SVector(4.0, 1.0, 1.0)
    fmm!(rec, rec_cache; scalar_potential=true, gradient=true)
    @test _radix_potential_error(rec, rec_direct) < 2e-3
    @test _radix_gradient_error(rec, rec_direct) < 2e-1

    # derived per-axis bounds: same margin convention per axis, rectangularity
    # (and here the resolved ell_axes) preserved, step counts restart
    FastMultipole.recenter!(rec_cache, rec; padding=0.05)
    @test rec_cache.ell_axes == SVector(4, 2, 2)
    @test rec_cache.box_extent[2] == rec_cache.box_extent[3] <
        rec_cache.box_extent[1] / 2
    rec_lo = SVector(ntuple(a -> minimum(
        rec.bodies[i].position[a] for i in eachindex(rec.bodies)), 3))
    rec_hi = SVector(ntuple(a -> maximum(
        rec.bodies[i].position[a] for i in eachindex(rec.bodies)), 3))
    @test rec_cache.x_min + rec_cache.box_extent / 2 ≈ (rec_lo + rec_hi) / 2
    @test rec_cache.step == 1
    rec.potential .= 0
    fmm!(rec, rec_cache; scalar_potential=true, gradient=true)
    @test _radix_potential_error(rec, rec_direct) < 2e-3
    @test _radix_gradient_error(rec, rec_direct) < 2e-1
    # bitwise parity vs a fresh cache at the recentered (already snapped) bounds
    rec_fresh_sys = generate_gravitational(seed + 5, nrec; bodies_fun=stretch2)
    rec_fresh = RadixFMMCache(rec_fresh_sys; expansion_order=3, ell=4,
        bounds=(rec_cache.x_min, rec_cache.box_extent))
    fmm!(rec_fresh_sys, rec_fresh; scalar_potential=true, gradient=true)
    @test rec.potential == rec_fresh_sys.potential

    # explicit vector bounds after a drift out of the box: the fixed-box
    # contract throws, then recenter! restores service at the shifted box
    # components exactly representable in Float32 (the default precision at
    # this expansion order), so the x_min equality below is exact
    shift = SVector(0.5, 0.25, -0.25)
    for i in eachindex(rec.bodies)
        b = rec.bodies[i]
        rec.bodies[i] = Body(b.position + shift, b.radius, b.strength)
    end
    @test_throws ArgumentError fmm!(rec, rec_cache; scalar_potential=true,
        gradient=true)
    FastMultipole.recenter!(rec_cache, rec; bounds=(origin + shift, (4.0, 0.8, 0.8)))
    @test rec_cache.ell_axes == SVector(4, 2, 2)
    @test rec_cache.box_extent == SVector(4.0, 1.0, 1.0)
    @test rec_cache.x_min == origin + shift
    @test rec_cache.step == 1
    rec.potential .= 0
    fmm!(rec, rec_cache; scalar_potential=true, gradient=true)
    # a rigid shift of bodies and box leaves the direct sums unchanged
    @test _radix_potential_error(rec, rec_direct) < 2e-3
    @test _radix_gradient_error(rec, rec_direct) < 2e-1

    #--- (f) invalid vector bounds still fail loudly ---#

    dev_sys = generate_gravitational(seed + 4, 100)
    @test_throws ArgumentError RadixFMMCache(dev_sys;
        bounds=(origin, (1.0, -1.0, 1.0)))
end
