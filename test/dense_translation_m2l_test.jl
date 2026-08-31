using FastMultipole
using FastMultipole.StaticArrays
using LinearAlgebra, Random, Test

const DENSE_FM = FastMultipole
isdefined(Main, :generate_gravitational) || include("gravitational.jl")

function _dense_random_physical_flat!(buf, rng, ::Val{LH}) where LH
    buf.phi .= randn(rng, eltype(buf.phi), size(buf.phi))
    for n in 0:buf.basis_info.orders.P_phi
        buf.phi[DENSE_FM.flat_basis_index(n, 0, 2), :] .= 0
    end
    if LH
        buf.chi .= randn(rng, eltype(buf.chi), size(buf.chi))
        for n in 0:buf.basis_info.orders.P_active
            buf.chi[DENSE_FM.flat_basis_index(n, 0, 2), :] .= 0
        end
    end
    return buf
end

function _dense_stack_flat(TF, binfo::DENSE_FM.OperatorBasisInfo{B,LH}, flat) where {B,LH}
    dm = DENSE_FM.DegreeMajorRealBuffer(TF, binfo, DENSE_FM.flat_nbatch(flat))
    DENSE_FM.to_gemm_buffer!(dm, flat)
    out = zeros(TF, DENSE_FM._dense_m2m_dof(binfo, Val(LH)),
        DENSE_FM.flat_nbatch(flat))
    DENSE_FM.stack_degree_major!(out, dm)
    return out
end

# Task 027 made HierarchicalRigidStencil the default policy. This file verifies the
# FLAT dense-translation resident plan (task 023e) -- per-class counts/starts,
# class capacities, and the per-class apply -- which the hierarchical windowed
# driver deliberately does not populate. Passing stencil_epsilon pins the flat
# classifier.

@testset "dense translation M2L (task 023e)" begin
    @test DenseTranslationM2L() isa DenseTranslationM2L
    @test_throws ArgumentError DenseTranslationM2L(max_persistent_bytes=0)
    @test_throws ArgumentError DenseTranslationM2L(max_persistent_bytes=big(typemax(Int)) + 1)
    @test_throws ArgumentError DenseTranslationM2L(apply_chunk=big(typemax(Int)) + 1)
    @test_throws ArgumentError DenseTranslationM2L(apply_chunk=-1)
    @test_throws ArgumentError DenseTranslationM2L(build_chunk=-1)
    @test_throws ArgumentError DenseTranslationM2L(apply_chunk=1.5)
    @test_throws ArgumentError CUDARadixLifecycleOptions(
        operator=FactoredRotationM2L(), m2l_strategy=DenseTranslationM2L())

    # CUDA-headroom keyword (task 023f): defaults, nonnegativity, Int-representability;
    # host numerical behavior is unchanged (the field is device-only).
    @test DenseTranslationM2L().cuda_headroom_bytes == 1 << 30
    @test DenseTranslationM2L(cuda_headroom_bytes=0).cuda_headroom_bytes == 0
    @test DenseTranslationM2L(cuda_headroom_bytes=2 << 30).cuda_headroom_bytes == 2 << 30
    @test_throws ArgumentError DenseTranslationM2L(cuda_headroom_bytes=-1)
    @test_throws ArgumentError DenseTranslationM2L(cuda_headroom_bytes=big(typemax(Int)) + 1)
    @test_throws ArgumentError DenseTranslationM2L(cuda_headroom_bytes=1.5)

    # Exact payload accounting and the inclusive persistent-memory gate on a small
    # synthetic displacement universe.
    bsmall = DENSE_FM.OperatorBasisInfo(DENSE_FM.CompressedComplexBasis(), 1, Val(false))
    invsmall = DENSE_FM.OperatorInvariantCache(Float64, bsmall)
    offsets_small = [SVector(2, 0, 0), SVector(-2, 1, 0)]
    caps = [min(7, 5, prod(8 .- abs.(Tuple(o)))) for o in offsets_small]
    aw = maximum(caps)
    fp = DENSE_FM._dense_m2l_footprint(Float64, bsmall, length(offsets_small), 7, aw, 2)
    exact = DenseTranslationM2L(max_persistent_bytes=fp.persistent_bytes,
        apply_chunk=aw, build_chunk=2)
    psmall = DENSE_FM.ResidentM2LDensePlan(Float64, bsmall, offsets_small, 0.25,
        7, 5, 8, exact, invsmall)
    @test psmall.persistent_bytes == fp.persistent_bytes
    @test psmall.operator_bytes == sum(sizeof, psmall.operators)
    @test psmall.scratch_bytes == sizeof(psmall.src_slab) + sizeof(psmall.dst_slab)
    actual_metadata = sizeof(psmall.route_class) + sizeof(psmall.class_counts) +
        sizeof(psmall.class_starts) + sizeof(psmall.class_capacities) +
        sizeof(psmall.packed_sources) + sizeof(psmall.packed_targets)
    @test psmall.route_metadata_bytes == actual_metadata
    @test psmall.persistent_bytes == psmall.operator_bytes + psmall.scratch_bytes + actual_metadata
    @test Base.summarysize(psmall) >= psmall.persistent_bytes
    limit_error = try
        DENSE_FM.ResidentM2LDensePlan(Float64, bsmall, offsets_small, 0.25,
            7, 5, 8, DenseTranslationM2L(
                max_persistent_bytes=fp.persistent_bytes - 1,
                apply_chunk=aw, build_chunk=2), invsmall)
        nothing
    catch err
        err
    end
    @test limit_error isa ArgumentError
    limit_message = sprint(showerror, limit_error)
    for fragment in ("D=", "classes=", "apply_width=", "build_width=",
            "operators=", "slabs=", "route metadata=", "persistent=",
            "construction peak=", "limit=", "Raise the limit", "lower P",
            "disable Lamb-Helmholtz", "reduce apply_chunk",
            "does not reduce operator storage")
        @test occursin(fragment, limit_message)
    end
    @test_throws ArgumentError DENSE_FM._dense_m2l_footprint(Float64, bsmall, 1,
        typemax(Int), 1, 1)

    # Float32 dense materialization at high order can overflow the combined
    # rotation*z-translation*rotation matrix. Explicit DenseTranslationM2L plans
    # reject those operators instead of carrying Inf/NaN into execution.
    for LH in (false, true)
        bbad = DENSE_FM.OperatorBasisInfo(DENSE_FM.CompressedComplexBasis(), 12, Val(LH))
        invbad32 = DENSE_FM.OperatorInvariantCache(Float32, bbad)
        bad32 = try
            DENSE_FM.ResidentM2LDensePlan(Float32, bbad, [SVector(2, 0, 0)], 0.15,
                8, 8, 8, DenseTranslationM2L(), invbad32)
            nothing
        catch err
            err
        end
        @test bad32 isa ArgumentError
        bad_msg = sprint(showerror, bad32)
        for fragment in ("DenseTranslationM2L", "non-finite", "precision=Float32",
                "P=12", "Lamb-Helmholtz=$(LH)", "displacement offset=[2, 0, 0]",
                "Use Float64", "lower P", "disable Lamb-Helmholtz",
                "PrecomputedFactoredYM2L", "factored", "concat")
            @test occursin(fragment, bad_msg)
        end

        invbad64 = DENSE_FM.OperatorInvariantCache(Float64, bbad)
        good64 = DENSE_FM.ResidentM2LDensePlan(Float64, bbad, [SVector(2, 0, 0)],
            0.15, 8, 8, 8, DenseTranslationM2L(), invbad64)
        @test all(op -> all(isfinite, op), good64.operators)
    end

    # Individual class matrices agree with the composed materialized-y oracle.
    rng = MersenneTwister(20260720)
    for cls in eachindex(psmall.operators)
        delta = 0.25 * SVector{3,Float64}(offsets_small[cls])
        r, theta, phi = DENSE_FM.cartesian_to_spherical(delta)
        src = DENSE_FM.FlatCoefficientBuffer(Float64, bsmall, 3)
        _dense_random_physical_flat!(src, rng, Val(false))
        tgt = DENSE_FM.FlatCoefficientBuffer(Float64, bsmall, 3)
        scratch = DENSE_FM.M2LOperatorScratch(Float64, bsmall, 3)
        DENSE_FM.m2l_operator_batch!(MaterializedYRotationM2L(), tgt, src,
            fill(phi, 3), fill(theta, 3), fill(r, 3), invsmall, scratch, Val(false))
        got = psmall.operators[cls] * _dense_stack_flat(Float64, bsmall, src)
        @test got ≈ _dense_stack_flat(Float64, bsmall, tgt) rtol=1e-10 atol=1e-10
    end

    # One-shot construction and execution use distinct represented offsets only.
    one_sys = generate_gravitational(20260720, 80)
    one_grid = RadixGrid(one_sys, 3)
    one_list = build_radix_interaction_list(
        LazyMaterializedBatches(1), ParentNeighborM2L(), one_grid)
    one = host_radix_state(one_sys, one_grid, one_list, 4;
        options=CUDARadixLifecycleOptions(m2l_strategy=
            DenseTranslationM2L(apply_chunk=8, build_chunk=8)))
    @test one.scratch.m2l_concat isa DENSE_FM.ResidentM2LDensePlan
    @test run_host_radix_lifecycle!(one) === one

    # Recurring parity, partial chunks, refresh identities, and stage allocation.
    for (TF, LH) in ((Float64, false), (Float64, true),
                     (Float32, false), (Float32, true))
        a = generate_gravitational(20260721, 120)
        b = generate_gravitational(20260721, 120)
        dense = RadixFMMCache(a; stencil_epsilon=1e-4, expansion_order=4, ell=3, lamb_helmholtz=LH,
            options=CUDARadixLifecycleOptions(; precision=TF,
                m2l_strategy=DenseTranslationM2L(apply_chunk=8, build_chunk=8)))
        concat = RadixFMMCache(b; stencil_epsilon=1e-4, expansion_order=4, ell=3, lamb_helmholtz=LH,
            options=CUDARadixLifecycleOptions(; precision=TF,
                m2l_strategy=ConcatenatedFixedZM2L()))
        plan = dense.state.scratch.m2l_concat
        @test sum(plan.class_counts) == dense.state.counts.n_routes
        @test all(plan.class_counts .<= plan.class_capacities)
        ids = (objectid(plan.route_class), objectid(plan.class_counts),
            objectid(plan.class_starts), objectid(plan.packed_sources),
            objectid(plan.packed_targets), objectid(plan.src_slab),
            objectid(plan.dst_slab), objectid(plan.operators), objectid.(plan.operators))
        fmm!(a, dense; scalar_potential=!LH, gradient=true)
        fmm!(b, concat; scalar_potential=!LH, gradient=true)
        tol = TF === Float64 ? 1e-9 : 3f-3
        @test a.potential ≈ b.potential rtol=tol atol=tol
        DENSE_FM.update_radix_state!(dense, (a,))
        @test (objectid(plan.route_class), objectid(plan.class_counts),
            objectid(plan.class_starts), objectid(plan.packed_sources),
            objectid(plan.packed_targets), objectid(plan.src_slab),
            objectid(plan.dst_slab), objectid(plan.operators),
            objectid.(plan.operators)) == ids
        DENSE_FM._launch_resident_m2l!(dense.state)
        @test (@allocated DENSE_FM._launch_resident_m2l!(dense.state)) <= 64 * 1024
    end

    # An isolated repeated target must receive the explicit sum of both products.
    rep_sys = generate_gravitational(20260722, 120)
    rep = RadixFMMCache(rep_sys; stencil_epsilon=1e-4, expansion_order=4, ell=3,
        options=CUDARadixLifecycleOptions(m2l_strategy=
            DenseTranslationM2L(apply_chunk=1, build_chunk=4)))
    rplan = rep.state.scratch.m2l_concat
    cls = findfirst(>=(2), rplan.class_capacities)
    @test cls !== nothing
    fill!(rplan.class_counts, 0)
    rplan.class_counts[cls] = 2
    cursor = 1
    for k in eachindex(rplan.class_counts)
        rplan.class_starts[k] = cursor
        cursor += rplan.class_counts[k]
    end
    rplan.class_starts[end] = cursor
    rplan.packed_sources[1:2] .= (1, 2)
    rplan.packed_targets[1:2] .= 3
    _dense_random_physical_flat!(rep.state.multipoles, rng, Val(false))
    phi_idx = rep.state.scratch.phi_flat_idx
    x = zeros(Float64, rplan.ndof, 2)
    for j in 1:2, i in eachindex(phi_idx)
        x[i, j] = rep.state.multipoles.phi[phi_idx[i], j]
    end
    expected = rplan.operators[cls] * (x[:, 1] + x[:, 2])
    DENSE_FM._launch_resident_m2l!(rep.state)
    @test rep.state.locals.phi[phi_idx, 3] ≈ expected rtol=1e-10 atol=1e-10

    # A genuine zero-route cache and launch leave locals zero.
    empty_sys = generate_gravitational(20260723, 1)
    empty_cache = RadixFMMCache(empty_sys; stencil_epsilon=1e-4, expansion_order=4, ell=2,
        bounds=(SVector(-0.1, -0.1, -0.1), 1.2),
        options=CUDARadixLifecycleOptions(m2l_strategy=DenseTranslationM2L()))
    @test empty_cache.state.counts.n_routes == 0
    DENSE_FM._launch_resident_m2l!(empty_cache.state)
    @test all(iszero, empty_cache.state.locals.phi)

    # Fixed-domain time stepping changes valid prefixes only and remains within the
    # task's warmed full-step allocation budget.
    full = generate_gravitational(20260724, 160)
    moving = RadixFMMCache(full; stencil_epsilon=1e-4, expansion_order=4, ell=3, max_n_bodies=160,
        bounds=(SVector(-0.1, -0.1, -0.1), 1.2),
        options=CUDARadixLifecycleOptions(m2l_strategy=
            DenseTranslationM2L(apply_chunk=8, build_chunk=8)))
    moving_plan = moving.state.scratch.m2l_concat
    moving_ids = (objectid(moving_plan.route_class), objectid(moving_plan.packed_sources),
        objectid(moving_plan.class_counts), objectid.(moving_plan.operators))
    for nbody in (73, 160, 91)
        step = Gravitational(copy(full.bodies[1:nbody]), zeros(16, nbody))
        fmm!(step, moving; scalar_potential=true, gradient=true)
        @test (objectid(moving_plan.route_class), objectid(moving_plan.packed_sources),
            objectid(moving_plan.class_counts), objectid.(moving_plan.operators)) == moving_ids
        @test sum(moving_plan.class_counts) == moving.state.counts.n_routes
    end
    fmm!(full, moving; scalar_potential=true, gradient=true)
    @test (@allocated fmm!(full, moving; scalar_potential=true, gradient=true)) < 512_000
end

@testset "dense translation M2L P=8 direct accuracy" begin
    seed = 20260725
    dense_sys = generate_gravitational(seed, 180)
    concat_sys = generate_gravitational(seed, 180)
    direct_sys = generate_gravitational(seed, 180)
    direct!(direct_sys; scalar_potential=true, gradient=true)
    dense_cache = RadixFMMCache(dense_sys; stencil_epsilon=1e-4, expansion_order=8, ell=3,
        options=CUDARadixLifecycleOptions(m2l_strategy=
            DenseTranslationM2L(apply_chunk=16, build_chunk=16)))
    concat_cache = RadixFMMCache(concat_sys; stencil_epsilon=1e-4, expansion_order=8, ell=3,
        options=CUDARadixLifecycleOptions(m2l_strategy=ConcatenatedFixedZM2L()))
    fmm!(dense_sys, dense_cache; scalar_potential=true, gradient=true)
    fmm!(concat_sys, concat_cache; scalar_potential=true, gradient=true)
    DENSE_FM._launch_resident_m2l!(dense_cache.state)
    @test (@allocated DENSE_FM._launch_resident_m2l!(dense_cache.state)) <= 64 * 1024
    @test maximum(abs.(dense_sys.potential .- concat_sys.potential)) < 1e-9
    @test maximum(abs.(dense_sys.potential[1, :] .- direct_sys.potential[1, :])) < 1e-6
    @test maximum(abs.(dense_sys.potential[5:7, :] .- direct_sys.potential[5:7, :])) < 1e-4
end
