#=##############################################################################
transform_tree!: rigid-motion tree reuse (FLOWPanel BRAINSTORM 021).

Under rigid motion x -> R*x + t, everything a tree encodes about RELATIVE
geometry is invariant, so transforming the branch centers (and boxes) must
make the old tree exact for the moved system. A fresh tree on the rotated
positions has DIFFERENT axis-aligned topology, so fresh-vs-transformed
outputs differ at truncation level; the rtol-1e-12 equivalence is instead
EQUIVARIANCE: fmm! on the rotated system with the transformed tree must
reproduce the original run's outputs rotated (phi invariant, g -> R*g), which
holds only if the transformed tree is exactly as valid for the rotated system
as the original tree was for the original system. A fresh-tree cross-check
vs direct! guards absolute accuracy, and a rebuilt interaction list confirms
list invariance.

Target buffer positions are refreshed manually here (transform_plan!
encapsulates this for plans).
=###############################################################################

# Rodrigues rotation about unit axis n by angle theta
function _rodrigues(n::SVector{3,Float64}, theta::Float64)
    n = n / norm(n)
    K = SMatrix{3,3,Float64,9}(0, n[3], -n[2], -n[3], 0, n[1], n[2], -n[1], 0)
    return SMatrix{3,3,Float64,9}(I) + sin(theta) * K + (1 - cos(theta)) * K * K
end

@testset "transform_tree!: rigid-motion equivariance (point masses)" begin

    n_bodies = 2000
    fresh_kwargs = (; expansion_order=8, multipole_acceptance=0.4,
                    leaf_size_source=30, scalar_potential=true, gradient=true,
                    hessian=false)

    R = _rodrigues(SVector(0.3, -1.0, 0.7), 37.0 * pi / 180)
    t = SVector(0.3, -0.2, 0.5)

    sys = generate_gravitational(123, n_bodies)
    plan = FastMultipole.FmmPlan((sys,), (sys,); fresh_kwargs...)

    # premise guards: both far and near field exercised; nontrivial transform
    @test length(plan.m2l_list) > 0
    @test length(plan.direct_list) > 0
    @test norm(R - SMatrix{3,3,Float64,9}(I)) > 0.1
    @test norm(t) > 0.1

    sys.potential .= 0
    FastMultipole.fmm!((sys,), (sys,), plan)
    phi0 = copy(sys.potential[1, :])
    g0 = [SVector{3}(sys.potential[i_gradient, i]) for i in 1:n_bodies]
    @test any(!iszero, phi0)                        # non-vacuous
    @test any(g -> norm(g) > 0, g0)

    # pre-rotation truncation error vs direct (the accuracy class the
    # transformed tree must preserve); must be nonzero so the equivariance
    # test below is non-vacuous (far field genuinely truncating)
    sys_d0 = generate_gravitational(123, n_bodies)
    sys_d0.potential .= 0
    FastMultipole.direct!((sys_d0,), (sys_d0,);
        scalar_potential=true, gradient=true)
    phi_exact0 = sys_d0.potential[1, :]
    err_orig = norm(phi0 - phi_exact0) / norm(phi_exact0)
    @test err_orig > 1e-13

    # rigidly move the system and transform the plan's trees to follow
    for i in 1:n_bodies
        b = sys.bodies[i]
        sys.bodies[i] = typeof(b)(R * b.position + t, b.radius, b.strength)
    end
    transform_tree!(plan.target_tree, R, t)
    transform_tree!(plan.source_tree, R, t)
    # target buffer positions are frozen at plan build; refresh manually
    FastMultipole.target_to_buffer!(plan.target_tree.buffers, (sys,),
        plan.target_tree.sort_index_list, plan.derivatives_switches)

    sys.potential .= 0
    FastMultipole.fmm!((sys,), (sys,), plan)
    phi1 = sys.potential[1, :]
    g1 = [SVector{3}(sys.potential[i_gradient, i]) for i in 1:n_bodies]

    # equivariance at floating-point level: phi invariant, g rotates
    @test isapprox(phi1, phi0; rtol=1e-12)
    g0_rot = [R * g for g in g0]
    @test isapprox(norm(reduce(vcat, g1) - reduce(vcat, g0_rot)),
                   0.0; atol=1e-12 * norm(reduce(vcat, g0_rot)))

    # interaction lists rebuilt on the transformed tree are unchanged
    # (distances, radii, and MAC are all invariant); compare as sets since
    # the parallel builder's emission order is not contractual
    m2l2, direct2 = FastMultipole.build_interaction_lists(
        plan.target_tree.branches, plan.source_tree.branches,
        plan.leaf_size_source, plan.multipole_acceptance, true, true, true,
        plan.interaction_list_method)
    @test Set(Tuple.(m2l2)) == Set(Tuple.(plan.m2l_list))
    @test Set(Tuple.(direct2)) == Set(Tuple.(plan.direct_list))

    # absolute-accuracy guard: the transformed tree's error vs direct! on the
    # rotated system must match the ORIGINAL tree's error class on the
    # original system (2x headroom for rotated-arithmetic differences).
    # NOTE a fresh tree on the rotated positions is NOT a valid reference:
    # the moved cloud's axis-aligned subdivision can degenerate to zero m2l
    # pairs (measured here: 160 -> 0 at mac=0.4), i.e. all-direct and
    # machine-exact — the transformed tree instead RETAINS its far field.
    sys_direct = generate_gravitational(123, n_bodies)
    for i in 1:n_bodies
        b = sys_direct.bodies[i]
        sys_direct.bodies[i] = typeof(b)(R * b.position + t, b.radius, b.strength)
    end
    sys_direct.potential .= 0
    FastMultipole.direct!((sys_direct,), (sys_direct,);
        scalar_potential=true, gradient=true)
    phi_exact = sys_direct.potential[1, :]
    @test any(!iszero, phi_exact)

    err_transformed = norm(phi1 - phi_exact) / norm(phi_exact)
    @test err_transformed <= 2 * err_orig + 1e-14

end

# Self-contained triangular source-panel system: exercises the vertex-carrying
# buffer path (body_to_multipole! reads vertices from the source buffer) with
# a scalar kernel. NOTE the vortex-filament test system was tried first and
# rejected: its velocity output carries an origin-dependent Lamb-Helmholtz
# gauge term (a pure translation shifts direct! output by t_z/2-class terms),
# so it cannot serve an equivariance test.
struct TransformPanels{TF}
    x::Vector{SVector{3,SVector{3,TF}}}   # 3 vertices per panel
    strength::Vector{TF}
    potential::Vector{TF}
    gradient::Vector{SVector{3,TF}}
end

function generate_transform_panels(seed, n_panels; scale=0.02)
    Random.seed!(seed)
    x = Vector{SVector{3,SVector{3,Float64}}}(undef, n_panels)
    for i in 1:n_panels
        c = rand(SVector{3,Float64})
        e1 = (rand(SVector{3,Float64}) .- 0.5) * scale
        e2 = (rand(SVector{3,Float64}) .- 0.5) * scale
        x[i] = SVector(c, c + e1, c + e2)
    end
    strength = rand(n_panels) ./ n_panels
    return TransformPanels(x, strength, zeros(n_panels),
                           zeros(SVector{3,Float64}, n_panels))
end

Base.eltype(::TransformPanels{TF}) where TF = TF
FastMultipole.has_vector_potential(::TransformPanels) = false
FastMultipole.get_n_bodies(system::TransformPanels) = length(system.strength)
FastMultipole.strength_dims(::TransformPanels) = 1
FastMultipole.data_per_body(::TransformPanels) = 14
FastMultipole.get_position(system::TransformPanels, i) =
    (system.x[i][1] + system.x[i][2] + system.x[i][3]) / 3

function FastMultipole.source_system_to_buffer!(buffer, i_buffer,
        system::TransformPanels, i_body)
    v1, v2, v3 = system.x[i_body]
    c = (v1 + v2 + v3) / 3
    buffer[1:3, i_buffer] .= c
    buffer[4, i_buffer] = max(norm(v1 - c), norm(v2 - c), norm(v3 - c))
    buffer[5, i_buffer] = system.strength[i_body]
    buffer[6:8, i_buffer] .= v1
    buffer[9:11, i_buffer] .= v2
    buffer[12:14, i_buffer] .= v3
end

FastMultipole.body_to_multipole!(system::TransformPanels, args...) =
    FastMultipole.body_to_multipole!(Panel{Source}, system, args...)

function FastMultipole.reset!(system::TransformPanels{TF}) where TF
    system.potential .= zero(TF)
    system.gradient .= Ref(zero(SVector{3,TF}))
end

function FastMultipole.buffer_to_target_system!(target_system::TransformPanels,
        i_target, ::FastMultipole.DerivativesSwitch{PS,GS,HS}, target_buffer,
        i_buffer) where {PS,GS,HS}
    PS && (target_system.potential[i_target] +=
        FastMultipole.get_scalar_potential(target_buffer, i_buffer))
    GS && (target_system.gradient[i_target] +=
        FastMultipole.get_gradient(target_buffer, i_buffer))
end

function FastMultipole.direct!(target_system, target_index,
        derivatives_switch::FastMultipole.DerivativesSwitch{PS,GS,HS},
        source_system::TransformPanels, source_buffer, source_index) where {PS,GS,HS}
    for i_source in source_index
        v1 = FastMultipole.get_vertex(source_buffer, source_system, i_source, 1)
        v2 = FastMultipole.get_vertex(source_buffer, source_system, i_source, 2)
        v3 = FastMultipole.get_vertex(source_buffer, source_system, i_source, 3)
        vertices = SVector(v1, v2, v3)
        normal = FastMultipole.get_normal(source_buffer, source_system, i_source)
        strength = FastMultipole.get_strength(source_buffer, source_system, i_source)
        centroid = FastMultipole.get_position(source_buffer, i_source)
        for i_target in target_index
            xt = SVector{3}(FastMultipole.get_position(target_system, i_target))
            phi, v, _ = induced(xt, vertices, normal, strength, centroid,
                Panel{Source}, DerivativesSwitch(PS, GS, false))
            PS && FastMultipole.set_scalar_potential!(target_system,
                derivatives_switch, i_target, phi)
            GS && FastMultipole.set_gradient!(target_system,
                derivatives_switch, i_target, v)
        end
    end
end

@testset "transform_tree!: vertex-carrying system (source panels)" begin

    R = _rodrigues(SVector(1.0, 0.4, -0.2), 63.0 * pi / 180)
    t = SVector(-0.4, 0.15, 0.8)

    panels = generate_transform_panels(456, 2000)
    n = length(panels.strength)
    plan = FastMultipole.FmmPlan((panels,), (panels,);
        expansion_order=10, multipole_acceptance=0.4, leaf_size_source=30,
        scalar_potential=true, gradient=true, hessian=false)
    @test length(plan.m2l_list) > 0
    @test length(plan.direct_list) > 0

    FastMultipole.reset!(panels)
    FastMultipole.fmm!((panels,), (panels,), plan)
    phi0 = copy(panels.potential)
    u0 = copy(panels.gradient)
    @test any(!iszero, phi0)                        # non-vacuous
    @test any(u -> norm(u) > 0, u0)

    # rigid motion: the panel VERTICES rotate; the scalar strength is invariant
    for i in 1:n
        v1, v2, v3 = panels.x[i]
        panels.x[i] = SVector(R * v1 + t, R * v2 + t, R * v3 + t)
    end
    FastMultipole.reset!(panels)
    transform_tree!(plan.target_tree, R, t)
    transform_tree!(plan.source_tree, R, t)
    FastMultipole.target_to_buffer!(plan.target_tree.buffers, (panels,),
        plan.target_tree.sort_index_list, plan.derivatives_switches)

    FastMultipole.fmm!((panels,), (panels,), plan)

    @test isapprox(panels.potential, phi0; rtol=1e-12)
    u0_rot = reduce(vcat, [R * u for u in u0])
    @test isapprox(norm(reduce(vcat, panels.gradient) - u0_rot), 0.0;
                   atol=1e-12 * norm(u0_rot))

end

@testset "transform_tree!: non-rigid transforms refuse" begin

    sys = generate_gravitational(42, 200)
    plan = FastMultipole.FmmPlan((sys,), (sys,); expansion_order=4,
        multipole_acceptance=0.4, leaf_size_source=30,
        scalar_potential=true, gradient=false, hessian=false)
    t = SVector(0.0, 0.0, 0.0)

    scale = 2.0 * SMatrix{3,3,Float64,9}(I)
    @test_throws ArgumentError transform_tree!(plan.source_tree, scale, t)

    reflection = SMatrix{3,3,Float64,9}(1, 0, 0, 0, 1, 0, 0, 0, -1)
    @test_throws ArgumentError transform_tree!(plan.source_tree, reflection, t)

end
