# Task 028: device-resident convection harness correctness.
#
# Validates the FM028 device system + device Euler step used by
# benchmark_028_feasibility.jl: the verdict-boundary loop (refresh + resident
# lifecycle + finalize + device Euler) keeps sampled-direct accuracy after the
# bodies move, with no per-step body H2D/D2H and flat transfer counters.
# Runs at expansion_order = 3 (P = 4) per the standing project rule.

using FastMultipole
using FastMultipole.StaticArrays
using Test

_cuda_convection_required() = get(ENV, "FASTMULTIPOLE_REQUIRE_CUDA_TESTS", "0") == "1"

# load + includes at top level so the FM028 method definitions are visible
# inside the testset (no world-age hazard)
const _CONV_LOADED = FastMultipole.load_cuda_radix_lifecycle!()
if _CONV_LOADED
    using CUDA
    include(joinpath(@__DIR__, "..", "MATRIX_OPERATOR_REFACTOR", "scripts",
        "fm028_device_system.jl"))
elseif _cuda_convection_required()
    error("FASTMULTIPOLE_REQUIRE_CUDA_TESTS=1 but CUDA radix lifecycle did not load: " *
          FastMultipole.cuda_radix_status())
else
    @info "CUDA radix lifecycle unavailable; skipping task 028 convection tests" FastMultipole.cuda_radix_status()
end

if _CONV_LOADED
    @testset "device convection lifecycle (task 028)" begin
        n = 2000
        ell = 3
        P = 3                       # expansion_order = 3, literature P = 4
        box_min = SVector(-0.01, -0.01, -0.01)
        box_size = 1.02
        dt = 1e-3
        indices = collect(1:n)

        for TF in (Float64, Float32)
            bodies = fm028_body_matrix(24025, n)
            sys = FM028DeviceSystem{TF}(bodies)
            opts = CUDARadixLifecycleOptions(; precision=TF,
                operator=MaterializedYRotationM2L(),
                m2l_strategy=DenseTranslationM2L())
            cache = RadixFMMCache(sys; expansion_order=P, ell, max_n_bodies=n,
                bounds=(box_min, box_size), device=true, options=opts,
                near_radius2=12, window_classes=8)

            fmm!(sys, cache; scalar_potential=true, gradient=true)

            # step-0 accuracy vs an on-device Float64 direct reference
            ref = fm028_direct_sample_reference(sys.positions, sys.strengths, indices)
            pot, grad = fm028_sampled_output(sys, indices)
            m0 = fm028_accuracy_metrics(pot, grad, ref[1, :], ref[2:4, :])
            @test m0.potential_rel_rms < 5e-3
            @test m0.gradient_rel_rms < 5e-3

            counters = cache.state.counters
            base = (counters.route_uploads, counters.operator_uploads,
                counters.body_uploads, counters.influence_downloads,
                counters.metadata_downloads)

            # verdict-boundary loop: fmm! (refresh + lifecycle + finalize) + Euler
            x0 = Array(sys.positions)
            for _ in 1:3
                fmm!(sys, cache; scalar_potential=true, gradient=true)
                fm028_euler!(sys, dt, 0.0, 1.0)
            end
            CUDA.synchronize()
            x3 = Array(sys.positions)
            @test maximum(abs.(x3 .- x0)) > 0        # bodies actually moved
            @test all(0.0 .<= x3 .<= 1.0)            # clamp kept them in the box

            # transfer counters flat across the convection steps (no per-step
            # body H2D/D2H on the device-resident verdict boundary)
            @test counters.route_uploads == base[1]
            @test counters.operator_uploads == base[2]
            @test counters.body_uploads == base[3]
            @test counters.influence_downloads == base[4]
            @test counters.metadata_downloads == base[5]
            @test counters.expansion_host_copies == 0

            # accuracy still holds at the moved positions (fresh reference)
            fmm!(sys, cache; scalar_potential=true, gradient=true)
            refk = fm028_direct_sample_reference(sys.positions, sys.strengths, indices)
            potk, gradk = fm028_sampled_output(sys, indices)
            mk = fm028_accuracy_metrics(potk, gradk, refk[1, :], refk[2:4, :])
            @test mk.potential_rel_rms < 5e-3
            @test mk.gradient_rel_rms < 5e-3
        end
    end

    # Task 028 lever 2: the fused dense M2L kernel packs several routes per
    # block and grid-strides over the rest, capped by DENSE_CUDA_FUSED_MAX_BLOCKS.
    # Squeezing the cap forces many grid-stride iterations and route groups that
    # run past n_routes, exercising the barrier-uniformity and shared-slice
    # logic that a single-wave launch never reaches. Results must be identical.
    #
    # This is also the only hierarchical coverage in the CUDA gate:
    # cuda_radix_lifecycle_test.jl never builds a hierarchical cache, so a
    # hierarchical-only kernel fault passes it (job 13000341 compiled to invalid
    # IR and the 215-test gate still went green).
    @testset "fused dense M2L grid-stride parity (task 028 lever 2)" begin
        n = 2000
        ell = 3
        P = 3
        box_min = SVector(-0.01, -0.01, -0.01)
        box_size = 1.02
        indices = collect(1:n)
        saved_cap = FastMultipole.DENSE_CUDA_FUSED_MAX_BLOCKS[]
        # This testset is about the FP32 fused kernel's launch decomposition; the
        # default tensor format would bypass that kernel entirely at Float32.
        saved_format = FastMultipole.DENSE_CUDA_TENSOR_FORMAT[]
        FastMultipole.DENSE_CUDA_TENSOR_FORMAT[] = :off
        try
            for TF in (Float64, Float32)
                results = map((typemax(Int), 3, 1)) do cap
                    FastMultipole.DENSE_CUDA_FUSED_MAX_BLOCKS[] = cap
                    bodies = fm028_body_matrix(24025, n)
                    sys = FM028DeviceSystem{TF}(bodies)
                    opts = CUDARadixLifecycleOptions(; precision=TF,
                        operator=MaterializedYRotationM2L(),
                        m2l_strategy=DenseTranslationM2L())
                    cache = RadixFMMCache(sys; expansion_order=P, ell,
                        max_n_bodies=n, bounds=(box_min, box_size), device=true,
                        options=opts, near_radius2=12, window_classes=8)
                    fmm!(sys, cache; scalar_potential=true, gradient=true)
                    fm028_sampled_output(sys, indices)
                end
                ref_pot, ref_grad = results[1]
                for (pot, grad) in results[2:end]
                    # same kernel, same inputs, different launch decomposition:
                    # atomics accumulate per target column in a different order,
                    # so allow a floating-point-reassociation tolerance only
                    tol = TF === Float64 ? 1e-10 : 1e-4
                    @test maximum(abs.(pot .- ref_pot)) <=
                        tol * max(1, maximum(abs.(ref_pot)))
                    @test maximum(abs.(grad .- ref_grad)) <=
                        tol * max(1, maximum(abs.(ref_grad)))
                end
            end
        finally
            FastMultipole.DENSE_CUDA_FUSED_MAX_BLOCKS[] = saved_cap
            FastMultipole.DENSE_CUDA_TENSOR_FORMAT[] = saved_format
        end
    end

    # Task 028 cycle 3: with overlap on, fill+nearfield run on a side stream
    # concurrently with the far-field chain, joined to L2B by a device event.
    # The arithmetic is identical to the serial path (the nearfield accumulates
    # atomically either way and L2B is ordered after it), so on/off must agree —
    # any divergence means a missed ordering edge (fill racing the previous
    # step's finalize, or L2B racing the nearfield). The convection loop below
    # stresses the cross-step begin-event edge.
    @testset "nearfield stream-overlap parity (task 028 cycle 3)" begin
        n = 2000
        ell = 3
        P = 3
        box_min = SVector(-0.01, -0.01, -0.01)
        box_size = 1.02
        indices = collect(1:n)
        saved_overlap = FastMultipole.CUDA_OVERLAP_NEARFIELD[]
        try
            for TF in (Float64, Float32)
                results = map((false, true)) do overlap
                    FastMultipole.CUDA_OVERLAP_NEARFIELD[] = overlap
                    bodies = fm028_body_matrix(24025, n)
                    sys = FM028DeviceSystem{TF}(bodies)
                    opts = CUDARadixLifecycleOptions(; precision=TF,
                        operator=MaterializedYRotationM2L(),
                        m2l_strategy=DenseTranslationM2L())
                    cache = RadixFMMCache(sys; expansion_order=P, ell,
                        max_n_bodies=n, bounds=(box_min, box_size), device=true,
                        options=opts, near_radius2=12, window_classes=8)
                    for _ in 1:3
                        fmm!(sys, cache; scalar_potential=true, gradient=true)
                        fm028_euler!(sys, 1e-3, 0.0, 1.0)
                    end
                    fmm!(sys, cache; scalar_potential=true, gradient=true)
                    fm028_sampled_output(sys, indices)
                end
                ref_pot, ref_grad = results[1]
                pot, grad = results[2]
                tol = TF === Float64 ? 1e-10 : 1e-4
                @test maximum(abs.(pot .- ref_pot)) <=
                    tol * max(1, maximum(abs.(ref_pot)))
                @test maximum(abs.(grad .- ref_grad)) <=
                    tol * max(1, maximum(abs.(ref_grad)))
            end
        finally
            FastMultipole.CUDA_OVERLAP_NEARFIELD[] = saved_overlap
        end
    end

    # Task 028 cycle 2: the operator-tiled dense M2L kernel must agree with the
    # plain fused kernel. Forcing MIN_ROUTES=0 routes every window (including
    # the tiny coarse levels) through the tiled kernel, and a squeezed blocks
    # cap forces large per-block chunks spanning many class segments, which
    # exercises the segment binary search and both WAR barriers.
    @testset "dense M2L operator-tile parity (task 028 cycle 2)" begin
        n = 2000
        ell = 3
        P = 3
        box_min = SVector(-0.01, -0.01, -0.01)
        box_size = 1.02
        indices = collect(1:n)
        # Same reason as the grid-stride testset: pin the FP32 kernel under test.
        saved_format = FastMultipole.DENSE_CUDA_TENSOR_FORMAT[]
        FastMultipole.DENSE_CUDA_TENSOR_FORMAT[] = :off
        saved_tiled = FastMultipole.DENSE_CUDA_TILED[]
        saved_min = FastMultipole.DENSE_CUDA_TILED_MIN_ROUTES[]
        saved_cap = FastMultipole.DENSE_CUDA_FUSED_MAX_BLOCKS[]
        try
            for TF in (Float64, Float32)
                configs = ((false, typemax(Int), typemax(Int)),  # fused reference
                           (true, 0, typemax(Int)),              # tiled, wide launch
                           (true, 0, 3),                         # tiled, deep chunks
                           (true, 0, 1))                         # tiled, one block
                results = map(configs) do (tiled, minroutes, cap)
                    FastMultipole.DENSE_CUDA_TILED[] = tiled
                    FastMultipole.DENSE_CUDA_TILED_MIN_ROUTES[] = minroutes
                    FastMultipole.DENSE_CUDA_FUSED_MAX_BLOCKS[] = cap
                    bodies = fm028_body_matrix(24025, n)
                    sys = FM028DeviceSystem{TF}(bodies)
                    opts = CUDARadixLifecycleOptions(; precision=TF,
                        operator=MaterializedYRotationM2L(),
                        m2l_strategy=DenseTranslationM2L())
                    cache = RadixFMMCache(sys; expansion_order=P, ell,
                        max_n_bodies=n, bounds=(box_min, box_size), device=true,
                        options=opts, near_radius2=12, window_classes=8)
                    fmm!(sys, cache; scalar_potential=true, gradient=true)
                    fm028_sampled_output(sys, indices)
                end
                ref_pot, ref_grad = results[1]
                for (pot, grad) in results[2:end]
                    tol = TF === Float64 ? 1e-10 : 1e-4
                    @test maximum(abs.(pot .- ref_pot)) <=
                        tol * max(1, maximum(abs.(ref_pot)))
                    @test maximum(abs.(grad .- ref_grad)) <=
                        tol * max(1, maximum(abs.(ref_grad)))
                end
            end
        finally
            FastMultipole.DENSE_CUDA_TENSOR_FORMAT[] = saved_format
            FastMultipole.DENSE_CUDA_TILED[] = saved_tiled
            FastMultipole.DENSE_CUDA_TILED_MIN_ROUTES[] = saved_min
            FastMultipole.DENSE_CUDA_FUSED_MAX_BLOCKS[] = saved_cap
        end
    end

    # Task 028 lever 1: _cuda_fast_rsqrt is new numerics — CUDA.rsqrt lowers to
    # rsqrt.approx (~1e-7 relative in Float64), and the Float64 method must
    # recover full precision through its two Newton steps.
    function _rsqrt_probe_kernel!(out, x)
        i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        i <= length(x) && (@inbounds out[i] = FastMultipole._cuda_fast_rsqrt(x[i]))
        return nothing
    end
    @testset "_cuda_fast_rsqrt accuracy (task 028 lever 1)" begin
        for (TF, tol) in ((Float64, 1e-14), (Float32, 5.0e-7))
            xs = TF.(exp10.(range(-30, 30; length=1001)))
            x = CuArray(xs)
            out = similar(x)
            @cuda threads=256 blocks=cld(length(x), 256) _rsqrt_probe_kernel!(out, x)
            rel = abs.(Float64.(Array(out)) .- 1 ./ sqrt.(Float64.(xs))) .*
                sqrt.(Float64.(xs))
            @test maximum(rel) < tol
        end
    end

    # Task 028 lever 1: the nearfield kernel is warp-per-pair with a grid-stride
    # capped by DIRECT_CUDA_MAX_BLOCKS. Squeezing the cap forces every warp
    # through many pairs (cap=1 runs all pairs through 4 warps), exercising the
    # stride logic a single-wave launch never reaches. n=2000 at ell=3 gives
    # ragged cells (empty, single-body, and target cells shared by many pairs).
    # Results must be identical up to atomic-reassociation rounding.
    @testset "nearfield warp-per-pair parity (task 028 lever 1)" begin
        ell = 3
        P = 3
        box_min = SVector(-0.01, -0.01, -0.01)
        box_size = 1.02
        saved_cap = FastMultipole.DIRECT_CUDA_MAX_BLOCKS[]
        try
            for TF in (Float64, Float32), n in (2000, 40)
                indices = collect(1:n)
                results = map((typemax(Int), 3, 1)) do cap
                    FastMultipole.DIRECT_CUDA_MAX_BLOCKS[] = cap
                    bodies = fm028_body_matrix(24025, n)
                    sys = FM028DeviceSystem{TF}(bodies)
                    opts = CUDARadixLifecycleOptions(; precision=TF,
                        operator=MaterializedYRotationM2L(),
                        m2l_strategy=DenseTranslationM2L())
                    cache = RadixFMMCache(sys; expansion_order=P, ell,
                        max_n_bodies=n, bounds=(box_min, box_size), device=true,
                        options=opts, near_radius2=12, window_classes=8)
                    fmm!(sys, cache; scalar_potential=true, gradient=true)
                    fm028_sampled_output(sys, indices)
                end
                ref_pot, ref_grad = results[1]
                for (pot, grad) in results[2:end]
                    tol = TF === Float64 ? 1e-10 : 1e-4
                    @test maximum(abs.(pot .- ref_pot)) <=
                        tol * max(1, maximum(abs.(ref_pot)))
                    @test maximum(abs.(grad .- ref_grad)) <=
                        tol * max(1, maximum(abs.(ref_grad)))
                end
            end
        finally
            FastMultipole.DIRECT_CUDA_MAX_BLOCKS[] = saved_cap
        end
    end
end
