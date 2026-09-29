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

# rotation by `theta` about the unit axis `n`
_core_rot(n, theta) = (k = n / norm(n); K = SMatrix{3,3}(0, k[3], -k[2], -k[3], 0, k[1],
    k[2], -k[1], 0); SMatrix{3,3}(1.0I) + sin(theta) * K + (1 - cos(theta)) * K * K)

# point sources on a sphere whose solver influence is the gradient projected on
# the body's outward normal (rows 6:8 of the source buffer) plus a panel-like
# self term c*q: the adjoint double-layer equation -q/(2A) + dphi/dn = f of a
# panel method (A the area per body). Rigid-motion invariant.
struct FluxGravitational{TF}
    inner::Gravitational{TF}
    normals::Vector{SVector{3,TF}}
    flux::Vector{TF}
    c::TF
end
Base.eltype(::FluxGravitational{TF}) where TF = TF
CORE_FM.get_n_bodies(s::FluxGravitational) = CORE_FM.get_n_bodies(s.inner)
CORE_FM.get_position(s::FluxGravitational, i) = CORE_FM.get_position(s.inner, i)
CORE_FM.data_per_body(::FluxGravitational) = 8
CORE_FM.strength_dims(::FluxGravitational) = 1
CORE_FM.has_vector_potential(::FluxGravitational) = false
function CORE_FM.source_system_to_buffer!(buffer, i_buffer, s::FluxGravitational, i_body)
    CORE_FM.source_system_to_buffer!(buffer, i_buffer, s.inner, i_body)
    buffer[6:8, i_buffer] .= s.normals[i_body]
end
CORE_FM.body_to_multipole!(s::FluxGravitational, args...) =
    CORE_FM.body_to_multipole!(Point{Source}, s, args...; scale_strength=-1.0)
function CORE_FM.direct!(tb, ti, sw::CORE_FM.DerivativesSwitch, s::FluxGravitational, sb, si)
    CORE_FM.direct!(tb, ti, sw, s.inner, sb, si)
    r = CORE_FM.gradient_range(sw)
    for i in ti, j in si       # self term, stored along the normal
        if tb[1, i] == sb[1, j] && tb[2, i] == sb[2, j] && tb[3, i] == sb[3, j]
            for k in 1:3
                tb[r[k], i] += s.c * sb[5, j] * sb[5 + k, j]
            end
        end
    end
end
CORE_FM.buffer_to_target_system!(s::FluxGravitational, i, sw, buffer, ib) =
    CORE_FM.buffer_to_target_system!(s.inner, i, sw, buffer, ib)
function CORE_FM.influence!(influence, target_buffer, sw::CORE_FM.DerivativesSwitch,
        ::FluxGravitational, source_buffer)
    r = CORE_FM.gradient_range(sw)
    for i in eachindex(influence)
        influence[i] = target_buffer[r[1], i] * source_buffer[6, i] +
            target_buffer[r[2], i] * source_buffer[7, i] +
            target_buffer[r[3], i] * source_buffer[8, i]
    end
    return influence
end
function CORE_FM.target_influence_to_buffer!(target_buffer, i_buffer, sw::CORE_FM.DerivativesSwitch,
        s::FluxGravitational, i_target)
    target_buffer[CORE_FM.gradient_range(sw), i_buffer] .= -s.flux[i_target] .* s.normals[i_target]
end
CORE_FM.value_to_strength!(source_buffer, ::FluxGravitational, i_body, value) =
    (source_buffer[5, i_body] = value)
CORE_FM.strength_to_value(strength, ::FluxGravitational) = strength[1]
function CORE_FM.buffer_to_system_strength!(s::FluxGravitational, i_body, source_buffer, i_buffer)
    b = s.inner.bodies[i_body]
    s.inner.bodies[i_body] = typeof(b)(b.position, b.radius, source_buffer[5, i_buffer])
end

# n Fibonacci points on the unit sphere at x -> R*x + t, outward normals rotated
# with them, zero strengths, and a fixed random right-hand side
function _flux_system(seed, n, R, t)
    rng = MersenneTwister(seed)
    flux = 1 .+ 0.5 .* randn(rng, n)
    b = zeros(8, n)
    normals = Vector{SVector{3,Float64}}(undef, n)
    golden = pi * (3 - sqrt(5))
    for i in 1:n
        z = 1 - 2 * (i - 0.5) / n
        rho = sqrt(1 - z^2)
        p = SVector(rho * cos(golden * i), rho * sin(golden * i), z)
        normals[i] = R * p
        b[1:3, i] .= R * p + t
        b[4, i] = 1e-3
    end
    return FluxGravitational(Gravitational(b), normals, flux, -n / (8 * pi))
end

# rectangular-box leaf fill plus one extra body (8 x 2 x 2 unit leaves)
function _rect_face_system(extra)
    pts = SVector{3,Float64}[]
    for z in 0:1, y in 0:1, x in 0:7
        push!(pts, SVector(x + 0.5, y + 0.5, z + 0.5))
    end
    push!(pts, extra)
    n = length(pts); b = zeros(8, n)
    for (i, p) in enumerate(pts)
        b[1:3, i] .= p; b[4, i] = 1e-3; b[5, i] = 1.0 / n
    end
    return Gravitational(b)
end

# singular point source for direct_rectangular!, rows x y z q
struct CoreRectSource <: AbstractRectangularKernel end
CORE_FM.rect_source_rows(::CoreRectSource) = 4
@inline function CORE_FM.rect_pair(::CoreRectSource, target::SVector{3,T}, sources, q,
        ::Val{GRAD}, ::Val{POT}) where {T,GRAD,POT}
    @inbounds d = target - SVector{3,T}(sources[1, q], sources[2, q], sources[3, q])
    r2 = dot(d, d)
    u = iszero(r2) ? zero(SVector{3,T}) : sources[4, q] / (4 * T(pi) * r2 * sqrt(r2)) * d
    return u, zero(SMatrix{3,3,T,9}), zero(T)
end

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
        cache = RadixFMMCache(sys; expansion_order=3, ell=3,
            options=RadixLifecycleOptions(; precision=Float32))
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

    @testset "rectangular box: body on a short axis' upper face" begin
        @test CORE_FM.radix_cell_coord(SVector(0.0, 0.0, 0.0), 4.0, 3,
            SVector(3.3, 2.0, 1.2), SVector(3, 1, 1)) == SVector(3, 1, 1)
        for extra in (SVector(3.3, 2.0, 1.2), SVector(7.3, 1.2, 2.0))
            sys = _rect_face_system(extra)
            cache = RadixFMMCache(sys; expansion_order=6, ell=3,
                bounds=(SVector(0.0, 0.0, 0.0), (8.0, 2.0, 2.0)))
            fmm!(sys, cache; scalar_potential=true)
            @test cache.state.counts.n_nodes <= cache.max_nodes
            @test cache.state.counts.n_cells <= cache.max_cells
            ref = _rect_face_system(extra)
            direct!(ref; scalar_potential=true)
            @test maximum(abs.(sys.potential[1, :] .- ref.potential[1, :])) /
                maximum(abs, ref.potential[1, :]) < 1e-4
        end
    end

    @testset "Float32 M2L out of range throws for every plan" begin
        strategies = (
            RadixLifecycleOptions(; precision=Float32, m2l_strategy=ConcatenatedFixedZM2L()),
            RadixLifecycleOptions(; precision=Float32, m2l_strategy=ConcatenatedFixedZM2L(),
                operator=CORE_FM.FactoredRotationM2L()),
            RadixLifecycleOptions(; precision=Float32,
                m2l_strategy=CORE_FM.PrecomputedFactoredYM2L(),
                operator=CORE_FM.FactoredRotationM2L()),
            RadixLifecycleOptions(; precision=Float32,
                m2l_strategy=CORE_FM.DenseTranslationM2L()),
        )
        small(seed) = generate_gravitational(seed, 300; bodies_fun=b -> (b[1:4, :] .*= 0.01))
        for opt in strategies, kw in ((;), (; stencil_epsilon=1e-4))
            err = try
                RadixFMMCache(small(81); expansion_order=12, ell=3, options=opt, kw...)
                nothing
            catch e
                e
            end
            @test err isa ArgumentError
            msg = err === nothing ? "" : sprint(showerror, err)
            @test occursin("Float32 range", msg) || occursin("exceed the Float32", msg)
            @test occursin("Float64", msg) && occursin("expansion order", msg)
        end
        # the materialized z-translation factors alone overflow at r = 1/16, P = 9
        blocks = zeros(Float32, CORE_FM.m2l_z_block_length(9))
        CORE_FM.m2l_z_blocks!(blocks, 0.0625f0, 9)
        @test_throws ArgumentError CORE_FM._check_m2l_blocks_finite(blocks, 0.0625f0, 9,
            "FactoredRotationM2L")
        # the concat factorial ratios (n + np)!/np! overflow Float32 from P = 25
        @test_throws ArgumentError CORE_FM._m2l_dense_factorial_matrix(
            zeros(Float32, 0, 0), Float32, 25)
        @test all(isfinite, CORE_FM._m2l_dense_factorial_matrix(zeros(Float32, 0, 0),
            Float32, 24))
        # Float64 at the same small box still runs and stays finite
        sys = small(82)
        cache = RadixFMMCache(sys; expansion_order=12, ell=3,
            options=RadixLifecycleOptions(; precision=Float64,
                m2l_strategy=ConcatenatedFixedZM2L()))
        fmm!(sys, cache; scalar_potential=true)
        @test all(isfinite, sys.potential[1, :])
    end

    @testset "planned fmm! carries its docstring" begin
        @test occursin("Run the FMM using the precomputed", string(@doc fmm!))
    end

    @testset "repeated transform_tree! keeps boxes bounded" begin
        sys = generate_gravitational(91, 400)
        plan = CORE_FM.FmmPlan((sys,), (sys,); expansion_order=4, leaf_size_source=20)
        tree = plan.source_tree
        box0 = [b.box for b in tree.branches]
        R1 = _core_rot(SVector(0.0, 0.0, 1.0), deg2rad(1.0))
        # one call is abs.(R) * box
        CORE_FM.transform_tree!(tree, R1, zero(SVector{3,Float64}))
        @test all(isapprox(tree.branches[i].box, abs.(R1) * box0[i]; rtol=1e-14)
            for i in eachindex(box0))
        for _ in 2:360
            CORE_FM.transform_tree!(tree, R1, zero(SVector{3,Float64}))
        end
        @test all(all(tree.branches[i].box .<= sqrt(3) .* box0[i] .* (1 + 1e-12) .+ 1e-14)
            for i in eachindex(box0))
        # a full turn returns the reference boxes
        @test all(isapprox(tree.branches[i].box, box0[i]; rtol=1e-9, atol=1e-12)
            for i in eachindex(box0))
    end

    @testset "Float32 evaluate_local returns Float32" begin
        P = 5
        rng = MersenneTwister(5)
        sw = DerivativesSwitch(true, true, true; third_derivative=true)
        results = Dict{DataType,Any}()
        for TF in (Float32, Float64)
            rng = MersenneTwister(5)
            loc = CORE_FM.initialize_expansion(P, TF)
            loc .= TF.(randn(rng, size(loc)...))
            harm = CORE_FM.initialize_harmonics(P, TF)
            grad = CORE_FM.initialize_gradient_n_m(P, TF; third_derivative=true)
            for LH in (false, true)
                res = CORE_FM.evaluate_local(SVector{3,TF}(0.1, 0.2, -0.15), harm, grad,
                    loc, P, Val(LH), sw)
                @test res[1] isa TF
                @test eltype(res[2]) == TF
                @test eltype(res[3]) == TF
                @test eltype(CORE_FM.packed_data(res[4])) == TF
                results[TF] = res
            end
        end
        @test isapprox(results[Float32][2], results[Float64][2]; rtol=1e-4)
    end

    @testset "direct_rectangular! accepts reshaped views of host arrays" begin
        src = vcat(rand(MersenneTwister(1), 3, 40), rand(MersenneTwister(2), 1, 40))
        tgt = rand(MersenneTwister(3), 3, 30) .+ 2
        ref = direct_rectangular!(zeros(3, 30), tgt, CoreRectSource(), src)
        store = zeros(3 * 30)
        out = reshape(view(store, 1:(3 * 30)), 3, 30)
        direct_rectangular!(out, tgt, CoreRectSource(), src)
        @test out == ref
    end

    @testset "explicit options kernel conflicting with the trait throws" begin
        isdefined(Main, :SmoothedVortex) || include("interface_test_systems.jl")
        sv = SmoothedVortex(generate_vortex(92, 200), fill(0.01, 200))
        err = try
            RadixFMMCache(sv; expansion_order=4, ell=3,
                options=RadixLifecycleOptions(; direct_kernel=SingularVortex()))
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test err !== nothing && occursin("conflicts with the direct_kernel(system) trait",
            sprint(showerror, err))
        # the trait alone still resolves
        @test RadixFMMCache(sv; expansion_order=4, ell=3).options.direct_kernel ==
            RegularizedVortex(; sigma_row=8)
    end

    @testset "element kernel parameters are validated" begin
        for family in 1:3
            @test VortexFilamentKernel(; family).family == family
        end
        @test_throws ArgumentError VortexFilamentKernel(; family=0)
        @test_throws ArgumentError VortexFilamentKernel(; family=4)
        for order in 1:3
            @test VortexSheetPanelKernel(; order).order == order
        end
        @test_throws ArgumentError VortexSheetPanelKernel(; order=0)
        @test_throws ArgumentError VortexSheetPanelKernel(; order=4)
    end

    @testset "extra target systems receive their metadata rows" begin
        main = generate_gravitational(93, 300)
        cache = RadixFMMCache(main; expansion_order=4, ell=3)
        extra = _meta_system(94, 50)
        for call in 1:2
            extra.meta .= collect(1.0:50) .* (call + 1)
            fill!(extra.seen, NaN)
            fmm!((main, extra), (main,), cache; scalar_potential=true)
            @test extra.seen == extra.meta
        end
    end

    @testset "default solve! after transform_solver!" begin
        n, seed = 600, 95
        R = _core_rot(SVector(0.2, 1.0, -0.5), deg2rad(63.0))
        t = SVector(0.6, -0.3, 0.4)
        I3 = SMatrix{3,3}(1.0I)
        kw = (; expansion_order=8, multipole_acceptance=0.5, leaf_size=40)
        solve_kw = (; max_iterations=30, inner_iterations=1, tolerance=1e-10,
            final_update=false, verbose=false)
        strengths(s) = [b.strength for b in s.inner.bodies]

        # reference: built and solved at the original pose
        s0 = _flux_system(seed, n, I3, zero(t))
        f0 = FastGaussSeidel((s0,), (s0,); kw...)
        CORE_FM.solve!(s0, f0; solve_kw...)
        x0 = strengths(s0)
        @test all(isfinite, x0) && any(!iszero, x0)

        # built at the original pose, moved rigidly, transformed, then the
        # default (gradient=true) solve
        s1 = _flux_system(seed, n, I3, zero(t))
        f1 = FastGaussSeidel((s1,), (s1,); kw...)
        moved = _flux_system(seed, n, R, t)
        s1.inner.bodies .= moved.inner.bodies
        s1.normals .= moved.normals
        CORE_FM.transform_solver!(f1, (s1,), R, t)
        CORE_FM.solve!(s1, f1; solve_kw...)
        x1 = strengths(s1)
        @test norm(x1 - x0) / norm(x0) < 1e-8

        # a solver rebuilt at the transformed pose agrees to the FMM accuracy
        s2 = _flux_system(seed, n, R, t)
        f2 = FastGaussSeidel((s2,), (s2,); kw...)
        CORE_FM.solve!(s2, f2; solve_kw...)
        @test norm(x1 - strengths(s2)) / norm(strengths(s2)) < 1e-5
    end
end
