# Host-only regression tests for core-path bugs: target metadata delivery,
# Float32 overflow/promotion, policy construction, and option resolution.

using FastMultipole
using FastMultipole.StaticArrays
using LinearAlgebra, Random, Test

const CORE_FM = FastMultipole
isdefined(Main, :generate_gravitational) || include("gravitational.jl")
isdefined(Main, :generate_vortex) || include("vortex.jl")

# a gravitational system carrying one controlled metadata row, recording the
# metadata row it sees in buffer_to_target_system!
struct MetaGravitational{TF}
    inner::Gravitational{TF}
    meta::Vector{TF}
    seen::Vector{TF}
end
MetaGravitational(inner::Gravitational{TF}, meta) where TF =
    MetaGravitational{TF}(inner, TF.(meta), fill(TF(NaN), length(meta)))
Base.eltype(::MetaGravitational{TF}) where TF = TF
CORE_FM.get_n_bodies(s::MetaGravitational) = CORE_FM.get_n_bodies(s.inner)
CORE_FM.get_position(s::MetaGravitational, i) = CORE_FM.get_position(s.inner, i)
CORE_FM.data_per_body(s::MetaGravitational) = CORE_FM.data_per_body(s.inner)
CORE_FM.strength_dims(s::MetaGravitational) = CORE_FM.strength_dims(s.inner)
CORE_FM.has_vector_potential(::MetaGravitational) = false
CORE_FM.source_system_to_buffer!(buffer, i_buffer, s::MetaGravitational, i_body) =
    CORE_FM.source_system_to_buffer!(buffer, i_buffer, s.inner, i_body)
CORE_FM.body_to_multipole!(s::MetaGravitational, args...) =
    CORE_FM.body_to_multipole!(Point{Source}, s, args...; scale_strength=-1.0)
CORE_FM.direct!(tb, ti, sw::CORE_FM.DerivativesSwitch, s::MetaGravitational, sb, si) =
    CORE_FM.direct!(tb, ti, sw, s.inner, sb, si)
CORE_FM.metadata_per_body(::MetaGravitational) = 1
CORE_FM.metadata_to_buffer!(buffer, switch, i_buffer, s::MetaGravitational, i_body) =
    (buffer[CORE_FM.metadata_index(switch, 1), i_buffer] = s.meta[i_body])
function CORE_FM.buffer_to_target_system!(s::MetaGravitational, i_target, switch,
        buffer, i_buffer)
    s.seen[i_target] = buffer[CORE_FM.metadata_index(switch, 1), i_buffer]
    return CORE_FM.buffer_to_target_system!(s.inner, i_target, switch, buffer, i_buffer)
end

_meta_system(seed, n) = MetaGravitational(generate_gravitational(seed, n),
    collect(1.0:n) .+ 0.5)

@testset "core regressions" begin

    @testset "FmmPlan keeps and refreshes target metadata" begin
        sys = _meta_system(11, 300)
        plan = CORE_FM.FmmPlan((sys,), (sys,); expansion_order=4, leaf_size_source=20)
        for call in 1:3
            sys.meta .= collect(1.0:300) .* call
            fill!(sys.seen, NaN)
            fmm!((sys,), (sys,), plan)
            @test sys.seen == sys.meta
        end
    end

    @testset "RadixFMMCache host path delivers metadata rows" begin
        sys = _meta_system(12, 300)
        cache = RadixFMMCache(sys; expansion_order=4, ell=3)
        for call in 1:2
            sys.meta .= collect(1.0:300) .+ 10call
            fill!(sys.seen, NaN)
            fmm!(sys, cache)
            @test sys.seen == sys.meta
        end
    end

    @testset "dense factorial matrix finite in Float32 at high P" begin
        for P in (18, 20)
            Z32 = CORE_FM._m2l_dense_factorial_matrix(zeros(Float32, 0, 0), Float32, P)
            @test eltype(Z32) == Float32
            @test all(isfinite, Z32)
        end
        # Float64 keeps the unscaled table, entry for entry
        for P in (4, 18, 20)
            Z64 = CORE_FM._m2l_dense_factorial_matrix(zeros(0, 0), Float64, P)
            fact = ones(2P + 1)
            for k in 1:2P
                fact[k + 1] = fact[k] * k
            end
            ref = zeros(size(Z64))
            for m in 0:P, k in (m == 0 ? (1:1) : (2m:(2m + 1))), np in m:P, n in m:P
                ref[CORE_FM.degree_row_offset(n) + k, CORE_FM.degree_row_offset(np) + k] =
                    fact[n + np + 1]
            end
            @test Z64 == ref
        end
    end

    @testset "Float32 concat M2L at P=18 matches Float64" begin
        results = Dict{DataType,Matrix{Float64}}()
        for TF in (Float32, Float64)
            sys = generate_gravitational(21, 200)
            cache = RadixFMMCache(sys; stencil_epsilon=1e-4, expansion_order=18, ell=3,
                options=RadixLifecycleOptions(; precision=TF,
                    m2l_strategy=ConcatenatedFixedZM2L()))
            @test cache.state.scratch.m2l_concat.nroutes > 0
            fmm!(sys, cache; scalar_potential=true, gradient=true)
            results[TF] = copy(sys.potential[[1, 5, 6, 7], :])
        end
        @test all(isfinite, results[Float32])
        @test maximum(abs.(results[Float32] .- results[Float64])) /
            maximum(abs.(results[Float64])) < 1e-4
    end

    @testset "Float32 third-derivative helpers stay Float32" begin
        P = 5
        nh = CORE_FM.harmonic_index(P, P)
        rng = MersenneTwister(3)
        for TF in (Float32, Float64)
            harmonics = TF.(randn(rng, 2, 2, nh))
            coeffs = TF.(randn(rng, 2, nh))
            g = CORE_FM._complex_gradient_contract(harmonics, coeffs, P)
            @test g isa NTuple{3,TF}
            scratch = zeros(TF, 2, 12, nh)
            scratch[:, 1:3, :] .= TF.(randn(rng, 2, 3, nh))
            for LH in (false, true)
                t = CORE_FM._third_derivative_from_gradient_coefficients!(
                    copy(scratch), harmonics, P, Val(LH))
                @test t isa SVector{18,TF}
            end
        end
        # Float32 agrees with the same inputs evaluated in Float64
        h64 = randn(rng, 2, 2, nh); c64 = randn(rng, 2, nh)
        g32 = CORE_FM._complex_gradient_contract(Float32.(h64), Float32.(c64), P)
        g64 = CORE_FM._complex_gradient_contract(Float32.(h64) .+ 0.0, Float32.(c64) .+ 0.0, P)
        @test all(isapprox.(g32, g64; rtol=1e-5))
    end

    @testset "Float32 cache keeps a Float32 policy" begin
        sys = generate_gravitational(31, 300)
        cache = RadixFMMCache(sys; expansion_order=3, ell=3)
        @test cache.state.options.precision == Float32
        @test cache.policy.config isa CORE_FM.ConstantPStencilConfig{Float32}
        recenter!(cache, sys)
        @test cache.policy.config isa CORE_FM.ConstantPStencilConfig{Float32}

        # all-direct demotion: Float32 policy, and the step is counted once
        base = generate_vortex(32, 200)
        isdefined(Main, :SmoothedVortex) || include("interface_test_systems.jl")
        bad = SmoothedVortex(base, fill(0.2, 200))
        fat = @test_logs (:warn, r"near-set adequacy failed") match_mode=:any RadixFMMCache(
            bad; expansion_order=4, ell=3,
            options=RadixLifecycleOptions(; precision=Float32,
                m2l_strategy=ConcatenatedFixedZM2L()))
        @test fat.ell == 2
        @test fat.policy.config isa CORE_FM.ConstantPStencilConfig{Float32}
        @test fat.step == 1
    end

    @testset "conflicting policy kwargs throw" begin
        sys = generate_gravitational(41, 200)
        @test_throws ArgumentError RadixFMMCache(sys; expansion_order=4, ell=1,
            near_radius2=12)
        @test_throws ArgumentError RadixFMMCache(sys; expansion_order=4, ell=1,
            level_radii2=(12,))
        # the non-conflicting forms still build; window_classes is a no-op there
        @test RadixFMMCache(sys; expansion_order=4, ell=1) isa RadixFMMCache
        @test RadixFMMCache(sys; expansion_order=4, ell=1, window_classes=64) isa RadixFMMCache
        @test RadixFMMCache(sys; expansion_order=4, ell=3, stencil_epsilon=1e-4) isa RadixFMMCache
    end

    @testset "sigma_row inside the strength rows throws" begin
        vort = generate_vortex(51, 100)            # 7 packed rows: 5:7 are strength
        opts(row) = RadixLifecycleOptions(; precision=Float64,
            m2l_strategy=ConcatenatedFixedZM2L(),
            direct_kernel=RegularizedVortex(; sigma_row=row))
        for row in 5:7
            err = try
                RadixFMMCache(vort; expansion_order=4, ell=2, options=opts(row))
                nothing
            catch e
                e
            end
            @test err isa ArgumentError
            @test occursin("strength rows", sprint(showerror, err))
        end
    end

    @testset "explicit direct kernel survives a body-type change" begin
        explicit = RadixLifecycleOptions(; direct_kernel=SingularSource())
        @test CORE_FM._options_with_body_type(explicit, Point{Vortex}).direct_kernel ===
            SingularSource()
        defaulted = RadixLifecycleOptions()
        @test defaulted.direct_kernel === SingularSource()
        @test CORE_FM._options_with_body_type(defaulted, Point{Vortex}).direct_kernel ===
            SingularVortex()
        # a trait-resolved cache still picks the body type's default kernel
        cache = RadixFMMCache(generate_vortex(52, 100); expansion_order=4, ell=2)
        @test cache.options.direct_kernel === SingularVortex()
    end

    @testset "estimate_nearfield_cache leaves buffers unchanged" begin
        sys = generate_gravitational(61, 300)
        plan = CORE_FM.FmmPlan((sys,), (sys,); expansion_order=4, leaf_size_source=20)
        fmm!((sys,), (sys,), plan)
        before = deepcopy(plan.target_tree.buffers)
        @test any(!iszero, before[1][CORE_FM.output_range(plan.derivatives_switches[1]), :])
        est = CORE_FM.estimate_nearfield_cache(plan.target_tree, plan.source_tree,
            plan.direct_list, plan.derivatives_switches, (sys,); sample=true)
        @test est.est_build_time >= 0
        @test plan.target_tree.buffers == before
    end

    @testset "threaded loops run nested" begin
        if Threads.nthreads() > 1
            sys = generate_gravitational(71, 400)
            plan = CORE_FM.FmmPlan((sys,), (sys,); expansion_order=4, leaf_size_source=20)
            cache = build_nearfield_cache!(plan, (sys,), (sys,))
            ok = Threads.Atomic{Int}(0)
            Threads.@threads for _ in 1:1
                nearfield_matvec!(plan.target_tree.buffers, cache,
                    plan.source_tree.buffers; n_threads=2)
                Threads.atomic_add!(ok, 1)
            end
            @test ok[] == 1
            built = Threads.Atomic{Int}(0)
            Threads.@threads for _ in 1:1
                RadixFMMCache(generate_gravitational(72, 150); stencil_epsilon=1e-4,
                    expansion_order=4, ell=3,
                    options=RadixLifecycleOptions(m2l_strategy=DenseTranslationM2L()))
                Threads.atomic_add!(built, 1)
            end
            @test built[] == 1
        else
            @test_skip "needs julia -t 2 or more"
        end
    end
end
