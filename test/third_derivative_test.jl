_third_index(t) = t[2,3,2]
_third_construct(data) = ThirdDerivativeTensor(data)
_third_get(buffer, switch) = get_third_derivative(buffer, switch, 1)
_third_packed(t) = packed_data(t)
_third_set_tensor!(buffer, switch, t) = set_third_derivative!(buffer, switch, 1, t)
_third_set_packed!(buffer, switch, data) = set_third_derivative!(buffer, switch, 1, data)

@testset "packed third derivative API" begin
    data = SVector{18,Float64}(1:18)
    tensor = ThirdDerivativeTensor(data)
    @test size(tensor) == (3, 3, 3)
    @test axes(tensor) == (Base.OneTo(3), Base.OneTo(3), Base.OneTo(3))
    @test length(tensor) == 27
    @test IndexStyle(typeof(tensor)) == IndexCartesian()
    @test packed_data(tensor) === data
    for i in 1:3, j in 1:3, k in 1:3
        pair = ((1,1)=>1, (1,2)=>2, (1,3)=>3, (2,2)=>4, (2,3)=>5, (3,3)=>6)
        jk = j <= k ? (j, k) : (k, j)
        slot = only(p.second for p in pair if p.first == jk)
        @test tensor[i,j,k] == data[6(i-1) + slot]
    end
    @test Array(dense(tensor)) == Array(tensor)

    for ps in (false, true), gs in (false, true), hs in (false, true), ts in (false, true)
        switch = DerivativesSwitch(ps, gs, hs; third_derivative=ts)
        @test FastMultipole.target_buffer_rows(switch) == 3 + ps + 3gs + 9hs + 18ts
        @test length(third_derivative_range(switch)) == (ts ? 18 : 0)
    end

    switch = DerivativesSwitch(true, true, true; third_derivative=true)
    buffer = zeros(34, 1)
    set_third_derivative!(buffer, switch, 1, tensor)
    @test packed_data(get_third_derivative(buffer, switch, 1)) == data
    @test get_third_derivative(buffer, 1) == tensor

    _third_construct(data); _third_get(buffer, switch); _third_index(tensor)
    _third_packed(tensor); _third_set_tensor!(buffer, switch, tensor)
    _third_set_packed!(buffer, switch, data)
    @test @allocated(_third_construct(data)) == 0
    @test @allocated(_third_get(buffer, switch)) == 0
    @test @allocated(_third_index(tensor)) == 0
    @test @allocated(_third_packed(tensor)) == 0
    @test @allocated(_third_set_tensor!(buffer, switch, tensor)) == 0
    @test @allocated(_third_set_packed!(buffer, switch, data)) == 0
end

@testset "third derivative direct and preflight" begin
    system = generate_gravitational(818, 12)
    direct!(system; scalar_potential=false, gradient=false, hessian=false,
        third_derivative=true, n_threads=1)
    @test any(!iszero, system.potential[i_third_derivative, :])
    @test all(iszero, system.potential[i_POTENTIAL, :])
    @test_throws ArgumentError direct!(zeros(3, 1), zeros(3, 1);
        scalar_potential=false, gradient=false, hessian=false,
        third_derivative=true, n_threads=1)

    for ps in (false, true), gs in (false, true), hs in (false, true), ts in (false, true)
        switched = generate_gravitational(822, 6)
        direct!(switched; scalar_potential=ps, gradient=gs, hessian=hs,
            third_derivative=ts, n_threads=1)
        @test any(!iszero, switched.potential[i_POTENTIAL[1], :]) == ps
        @test any(!iszero, switched.potential[i_gradient, :]) == gs
        @test any(!iszero, switched.potential[i_hessian, :]) == hs
        @test any(!iszero, switched.potential[i_third_derivative, :]) == ts
    end
end

@testset "analytic direct third derivatives vs ForwardDiff" begin
    c = FastMultipole.ONE_OVER_4π
    x = SVector(0.7, -0.4, 1.2)
    source_position = SVector(-0.2, 0.3, 0.1)
    q = 1.7
    source_data = zeros(8, 1)
    source_data[1:3, 1] .= source_position
    source_data[5, 1] = q
    source = Gravitational(source_data)
    target_data = zeros(8, 1)
    target_data[1:3, 1] .= x
    target = Gravitational(target_data)
    direct!(target, source; scalar_potential=false, gradient=false, hessian=false,
        third_derivative=true, n_threads=1)
    scalar_hessian(z) = ForwardDiff.hessian(y -> q * c / norm(y - source_position), z)
    scalar_third = ForwardDiff.jacobian(z -> vec(scalar_hessian(z)), x)
    expected_scalar = SVector{18}(ntuple(Val(18)) do slot
        i = (slot - 1) ÷ 6 + 1
        pair = ((1,1), (1,2), (1,3), (2,2), (2,3), (3,3))[(slot - 1) % 6 + 1]
        j, k = pair
        scalar_third[i + 3(j - 1), k]
    end)
    @test target.potential[i_third_derivative, 1] ≈ expected_scalar rtol=2e-13

    Γ = SVector(0.3, -0.8, 1.1)
    vortex_source = VortexParticles(reshape(collect(source_position), 3, 1),
        reshape(collect(Γ), 3, 1))
    vortex_target = VortexParticles(reshape(collect(x), 3, 1), zeros(3, 1))
    direct!(vortex_target, vortex_source; scalar_potential=false, gradient=false,
        hessian=false, third_derivative=true, n_threads=1)
    velocity(z) = cross(Γ, z - source_position) * c / norm(z - source_position)^3
    expected_vortex = MVector{18,Float64}(undef)
    slot = 0
    for i in 1:3
        Hi = ForwardDiff.hessian(z -> velocity(z)[i], x)
        for (j, k) in ((1,1), (1,2), (1,3), (2,2), (2,3), (3,3))
            slot += 1
            expected_vortex[slot] = Hi[j,k]
        end
    end
    @test vortex_target.potential[i_THIRD_DERIVATIVE_vortex, 1] ≈ expected_vortex rtol=3e-13
end

@testset "third derivative FMM scalar and LH" begin
    scalar = generate_gravitational(819, 96)
    fmm!(scalar; scalar_potential=false, gradient=false, hessian=false,
        third_derivative=true, leaf_size=12, expansion_order=7)
    @test any(!iszero, scalar.potential[i_third_derivative, :])

    Random.seed!(820)
    vortex = VortexParticles(rand(3, 48), randn(3, 48))
    fmm!(vortex; scalar_potential=false, gradient=false, hessian=false,
        third_derivative=true, leaf_size=8, expansion_order=7)
    @test any(!iszero, vortex.potential[i_THIRD_DERIVATIVE_vortex, :])
end


@testset "third derivative FMM accuracy, multi-system, threading, dynamic P" begin
    # FMM TS converges to direct with expansion order through a REAL far field
    # (P-sweep 2026-09-07 at this geometry: 3.8e-8 at P=6, 3.2e-11 at P=12,
    # 3.5e-15 at P=20 — near machine precision, so accuracy is P-limited only)
    ref_sys = generate_gravitational(823, 2000)
    direct!(ref_sys; scalar_potential=false, gradient=false, hessian=false,
        third_derivative=true, n_threads=1)
    ref_ts = ref_sys.potential[i_third_derivative, :]
    @test any(!iszero, ref_ts)
    ts_error_at = function (P)
        fmm_sys = generate_gravitational(823, 2000)
        plan = FastMultipole.FmmPlan((fmm_sys,), (fmm_sys,); scalar_potential=false,
            gradient=false, hessian=false, third_derivative=true,
            leaf_size_source=30, expansion_order=P, multipole_acceptance=0.5)
        @test length(plan.m2l_list) > 0      # premise: real far field
        @test length(plan.direct_list) > 0   # premise: real near field
        fmm!((fmm_sys,), (fmm_sys,), plan)
        return norm(fmm_sys.potential[i_third_derivative, :] - ref_ts) / norm(ref_ts)
    end
    err_6 = ts_error_at(6)
    err_12 = ts_error_at(12)
    @test err_6 <= 1e-7
    @test err_12 <= 1e-10
    @test err_12 < err_6 / 100   # geometric convergence, not a shared floor

    # multi-system targets with per-system TS switches through the two-argument call
    grav_a = generate_gravitational(824, 40)
    grav_b = generate_gravitational(825, 40)
    source = (generate_gravitational(826, 60),)
    fmm!((grav_a, grav_b), source; scalar_potential=false, gradient=false,
        hessian=false, third_derivative=[true, false], leaf_size_source=8,
        expansion_order=8, silence_warnings=true)
    @test any(!iszero, grav_a.potential[i_third_derivative, :])
    @test all(iszero, grav_b.potential[i_third_derivative, :])

    # threaded direct! agrees with single-thread
    thr_1 = generate_gravitational(827, 90)
    thr_2 = generate_gravitational(827, 90)
    direct!(thr_1; scalar_potential=false, gradient=false, hessian=false,
        third_derivative=true, n_threads=1)
    direct!(thr_2; scalar_potential=false, gradient=false, hessian=false,
        third_derivative=true, n_threads=2)
    @test isapprox(thr_1.potential[i_third_derivative, :],
        thr_2.potential[i_third_derivative, :]; rtol=1e-13)

    # dynamic expansion order with TS requested still converges to direct
    # (same geometry as the convergence check above, so the far field is real)
    dyn_sys = generate_gravitational(823, 2000)
    fmm!(dyn_sys; scalar_potential=false, gradient=true, hessian=false,
        third_derivative=true, leaf_size=30, expansion_order=20,
        multipole_acceptance=0.5,
        error_tolerance=FastMultipole.PowerAbsoluteGradient(1e-9, false))
    @test any(!iszero, dyn_sys.potential[i_third_derivative, :])
    @test norm(dyn_sys.potential[i_third_derivative, :] - ref_ts) <=
        1e-3 * norm(ref_ts)
end

@testset "third derivative conditioning" begin
    plain = generate_gravitational(829, 60)
    conditioned = generate_gravitational(829, 60)
    direct!(plain; scalar_potential=false, gradient=false, hessian=false,
        third_derivative=true, n_threads=1)

    scale_strengths! = function (source_buffer, source_system, i_source_system,
            target_buffer, i_target_system)
        rows = 5:4+FastMultipole.strength_dims(source_system)
        source_buffer[rows, :] .*= 2.0
        return nothing
    end
    unscale_strengths! = function (source_buffer, source_system, i_source_system,
            target_buffer, i_target_system)
        rows = 5:4+FastMultipole.strength_dims(source_system)
        source_buffer[rows, :] .*= 0.5
        return nothing
    end
    rule = DirectConditioningRule(SelfPairs(), scale_strengths!, unscale_strengths!)

    switches = FastMultipole.DerivativesSwitch([false], [false], [false],
        (conditioned,); third_derivative=[true])
    source_buffers = FastMultipole.allocate_buffers((conditioned,), false, Float64, switches)
    FastMultipole.system_to_buffer!(source_buffers, (conditioned,))
    buffer_copies = deepcopy(source_buffers)
    direct!((conditioned,), (conditioned,); scalar_potential=[false], gradient=[false],
        hessian=[false], third_derivative=[true], n_threads=1, source_buffers,
        direct_conditioning=rule)

    # TS is linear in strength, so the doubled-strength conditioning doubles TS,
    # and the source buffers must be restored afterward
    @test isapprox(conditioned.potential[i_third_derivative, :],
        2.0 .* plain.potential[i_third_derivative, :]; rtol=1e-13)
    @test source_buffers[1] == buffer_copies[1]
end

@testset "third derivative nearfield cache and transformed plans" begin
    n_bodies = 2000
    plan_kwargs = (; expansion_order=8, multipole_acceptance=0.4, leaf_size_source=30,
        scalar_potential=false, gradient=false, hessian=false, third_derivative=true)

    sys_cached = generate_gravitational(830, n_bodies)
    sys_ref = generate_gravitational(830, n_bodies)
    plan_cached = FastMultipole.FmmPlan((sys_cached,), (sys_cached,); plan_kwargs...)
    plan_ref = FastMultipole.FmmPlan((sys_ref,), (sys_ref,); plan_kwargs...)
    @test length(plan_cached.direct_list) > 0
    @test length(plan_cached.m2l_list) > 0

    cache = build_nearfield_cache!(plan_cached, (sys_cached,), (sys_cached,))
    @test plan_cached.nearfield_cache[] === cache

    for trial in 1:2
        if trial > 1
            # strength-change reuse through the plan path
            for i in 1:n_bodies
                b = sys_cached.bodies[i]
                new_strength = b.strength * 1.5 + 0.001 * i
                sys_cached.bodies[i] = typeof(b)(b.position, b.radius, new_strength)
                r = sys_ref.bodies[i]
                sys_ref.bodies[i] = typeof(r)(r.position, r.radius, new_strength)
            end
        end
        sys_cached.potential .= 0
        sys_ref.potential .= 0
        fmm!((sys_cached,), (sys_cached,), plan_cached)
        fmm!((sys_ref,), (sys_ref,), plan_ref)
        @test any(!iszero, sys_ref.potential[i_third_derivative, :])
        @test isapprox(sys_cached.potential[i_third_derivative, :],
            sys_ref.potential[i_third_derivative, :]; rtol=1e-12)
    end

    # TS is direction-carrying: a stored cache refuses rigid-motion transforms
    R = SMatrix{3,3,Float64,9}(1, 0, 0, 0, 1, 0, 0, 0, 1)
    t = SVector(0.0, 0.0, 0.0)
    @test_throws ArgumentError FastMultipole.transform_plan!(plan_cached,
        (sys_cached,), R, t)

    # fmm cache built without TS rejects a TS request (switch layout mismatch)
    mismatch_sys = generate_gravitational(831, 100)
    gradient_switches = FastMultipole.DerivativesSwitch(false, true, false, (mismatch_sys,))
    fmm_cache = FastMultipole.Cache((mismatch_sys,), (mismatch_sys,), gradient_switches)
    @test_throws ArgumentError fmm!(mismatch_sys, fmm_cache; scalar_potential=false,
        gradient=true, hessian=false, third_derivative=true)
end

@testset "third derivative metadata, extra outputs, and legacy overloads" begin
    # TS rows sit between the Hessian and extra-output rows, after metadata
    switch = FastMultipole.DerivativesSwitch(true, true, true; third_derivative=true,
        extra_outputs=2, metadata=2)
    @test switch isa FastMultipole.DerivativesSwitch{true,true,true,2,2,true}
    @test FastMultipole.metadata_range(switch) == 4:5
    @test FastMultipole.scalar_potential_index(switch) == 6
    @test FastMultipole.gradient_range(switch) == 7:9
    @test FastMultipole.hessian_range(switch) == 10:18
    @test third_derivative_range(switch) == 19:36
    @test FastMultipole.extra_output_range(switch) == 37:38
    @test FastMultipole.target_buffer_rows(switch) == 38

    data = SVector{18,Float64}(1:18)
    buffer = zeros(38, 1)
    set_third_derivative!(buffer, switch, 1, data)
    @test buffer[19:36, 1] == collect(1.0:18.0)
    @test packed_data(get_third_derivative(buffer, switch, 1)) == data

    # legacy three-switch user kernel: ordinary requests work, TS fails preflight
    position = [3.0 1.0; 0.0 0.0; 0.0 0.0]
    strength = [1.0, 2.0]
    meta = [10.0 20.0; 11.0 21.0]
    legacy_target = MetadataSystem(copy(position), copy(strength), copy(meta),
        zeros(2), zeros(2, 2))
    legacy_source = MetadataSystem(copy(position), copy(strength), copy(meta),
        zeros(2), zeros(0, 2))
    @test !supports_third_derivative(legacy_target, legacy_source)
    # the MetadataSystem kernel always emits 2 extra outputs, so they must be requested
    direct!(legacy_target, legacy_source; scalar_potential=true, gradient=false,
        hessian=false, metadata=2, extra_outputs=2, n_threads=1)
    @test legacy_target.scalar == fill(sum(strength), 2)
    @test legacy_target.extra == meta .* sum(strength)
    @test_throws ArgumentError direct!(legacy_target, legacy_source;
        scalar_potential=true, gradient=false, hessian=false, metadata=2,
        extra_outputs=2, third_derivative=true, n_threads=1)
end

@testset "third derivative probes and plans" begin
    source = generate_gravitational(821, 20)
    probes = FastMultipole.ProbeSystem(3)
    probes.position .= [SVector(2.0, 2.0, 2.0), SVector(2.1, 2.0, 2.0),
        SVector(2.0, 2.1, 2.0)]
    direct!(probes, source; scalar_potential=false, gradient=false, hessian=false,
        third_derivative=true, n_threads=1)
    @test any(t -> any(!iszero, packed_data(t)), probes.third_derivative)

    fmm_probes = FastMultipole.ProbeSystem(40)
    Random.seed!(824)
    for i in 1:40
        fmm_probes.position[i] = SVector{3}(2.0 .+ 0.5 .* rand(3))
    end
    probe_plan = FastMultipole.FmmPlan((fmm_probes,), (source,);
        scalar_potential=false, gradient=false, hessian=false,
        third_derivative=true, leaf_size_source=8, expansion_order=8)
    @test length(probe_plan.m2l_list) > 0   # premise: probes reached via L2B
    fmm!((fmm_probes,), (source,), probe_plan)
    @test any(t -> any(!iszero, packed_data(t)), fmm_probes.third_derivative)

    planned = deepcopy(source)
    plan = FastMultipole.FmmPlan((planned,), (planned,); scalar_potential=false, gradient=false,
        hessian=false, third_derivative=true, leaf_size_source=8)
    FastMultipole.build_nearfield_cache!(plan, (planned,), (planned,))
    fmm!((planned,), (planned,), plan)
    @test any(!iszero, planned.potential[i_third_derivative, :])
end
@testset "LH third derivative FMM accuracy, symmetry, dynamic P" begin
    # FMM TS for a point-vortex (Lamb-Helmholtz) system converges to direct with
    # expansion order through a REAL far field (P-sweep 2026-09-07 at this geometry:
    # 3.0e-10 at P=6, 4.0e-13 at P=12, 8.4e-16 at P=20 with 537 M2L pairs)
    n_vort = 2000
    Random.seed!(833)
    vort_position = rand(3, n_vort)
    vort_strength = randn(3, n_vort) ./ n_vort
    ref_vort = VortexParticles(copy(vort_position), copy(vort_strength))
    direct!(ref_vort; scalar_potential=false, gradient=false, hessian=false,
        third_derivative=true, n_threads=1)
    ref_vort_ts = ref_vort.potential[i_THIRD_DERIVATIVE_vortex, :]
    @test any(!iszero, ref_vort_ts)

    # LH tensors are (j,k)-symmetric by construction but NOT (i,j)-symmetric;
    # assert the reference field actually exhibits that asymmetry so the packed
    # 18-slot layout is exercised beyond the scalar 10-component subspace
    max_asym = 0.0
    for b in 1:n_vort
        t = dense(ThirdDerivativeTensor(SVector{18}(ref_vort_ts[:, b])))
        a = maximum(abs(t[i,j,k] - t[j,i,k]) for i in 1:3, j in 1:3, k in 1:3)
        max_asym = max(max_asym, a / maximum(abs, t))
    end
    @test max_asym > 0.5

    lh_ts_error_at = function (P)
        sys = VortexParticles(copy(vort_position), copy(vort_strength))
        plan = FastMultipole.FmmPlan((sys,), (sys,); scalar_potential=false,
            gradient=false, hessian=false, third_derivative=true,
            leaf_size_source=30, expansion_order=P, multipole_acceptance=0.5)
        @test length(plan.m2l_list) > 0      # premise: real far field
        @test length(plan.direct_list) > 0   # premise: real near field
        fmm!((sys,), (sys,), plan)
        return norm(sys.potential[i_THIRD_DERIVATIVE_vortex, :] - ref_vort_ts) /
            norm(ref_vort_ts)
    end
    lh_err_6 = lh_ts_error_at(6)
    lh_err_12 = lh_ts_error_at(12)
    @test lh_err_6 <= 3e-9
    @test lh_err_12 <= 1e-11
    @test lh_err_12 < lh_err_6 / 100   # geometric convergence, not a shared floor

    # dynamic expansion order with TS requested still converges to direct
    dyn_vort = VortexParticles(copy(vort_position), copy(vort_strength))
    fmm!(dyn_vort; scalar_potential=false, gradient=true, hessian=false,
        third_derivative=true, leaf_size=30, expansion_order=20,
        multipole_acceptance=0.5,
        error_tolerance=FastMultipole.PowerAbsoluteGradient(1e-9, false))
    @test any(!iszero, dyn_vort.potential[i_THIRD_DERIVATIVE_vortex, :])
    @test norm(dyn_vort.potential[i_THIRD_DERIVATIVE_vortex, :] - ref_vort_ts) <=
        1e-10 * norm(ref_vort_ts)
end

@testset "LH third derivative threading and mixed pairs" begin
    # threaded direct! agrees with single-thread for vortons
    Random.seed!(836)
    thr_pos = rand(3, 90)
    thr_str = randn(3, 90)
    thr_1 = VortexParticles(copy(thr_pos), copy(thr_str))
    thr_2 = VortexParticles(copy(thr_pos), copy(thr_str))
    direct!(thr_1; scalar_potential=false, gradient=false, hessian=false,
        third_derivative=true, n_threads=1)
    direct!(thr_2; scalar_potential=false, gradient=false, hessian=false,
        third_derivative=true, n_threads=2)
    @test isapprox(thr_1.potential[i_THIRD_DERIVATIVE_vortex, :],
        thr_2.potential[i_THIRD_DERIVATIVE_vortex, :]; rtol=1e-13)

    # mixed scalar/LH target and source systems in one call, through a REAL far
    # field (calibrated 2026-09-07: 1399 M2L pairs, rel errors 7.0e-7 / 8.0e-7)
    Random.seed!(837)
    grav_t = generate_gravitational(834, 400)
    vort_t = VortexParticles(rand(3, 400), randn(3, 400) ./ 400)
    grav_s = generate_gravitational(835, 400)
    vort_s = VortexParticles(rand(3, 400), randn(3, 400) ./ 400)
    direct!((grav_t, vort_t), (grav_s, vort_s); scalar_potential=[false, false],
        gradient=[false, false], hessian=[false, false],
        third_derivative=[true, true], n_threads=1)
    mixed_ref_g = copy(grav_t.potential[i_third_derivative, :])
    mixed_ref_v = copy(vort_t.potential[i_THIRD_DERIVATIVE_vortex, :])
    @test any(!iszero, mixed_ref_g)
    @test any(!iszero, mixed_ref_v)
    grav_t.potential .= 0
    vort_t.potential .= 0
    mixed_plan = FastMultipole.FmmPlan((grav_t, vort_t), (grav_s, vort_s);
        scalar_potential=false, gradient=false, hessian=false,
        third_derivative=true, leaf_size_source=10, expansion_order=8,
        multipole_acceptance=0.6)
    @test length(mixed_plan.m2l_list) > 0    # premise: real far field
    @test length(mixed_plan.direct_list) > 0 # premise: real near field
    fmm!((grav_t, vort_t), (grav_s, vort_s), mixed_plan)
    @test isapprox(grav_t.potential[i_third_derivative, :], mixed_ref_g;
        rtol=1e-5)
    @test isapprox(vort_t.potential[i_THIRD_DERIVATIVE_vortex, :], mixed_ref_v;
        rtol=1e-5)
end

@testset "LH third derivative FLOWVPM-shaped usage" begin
    # FLOWVPM-style request: velocity + velocity gradient (stretching) + third
    # derivative in one lamb_helmholtz call, with plan reuse across a strength
    # update as in time stepping (calibrated 2026-09-07: 27 M2L pairs, rel errors
    # <= 6.9e-6 on both steps)
    n_vpm = 300
    Random.seed!(838)
    vpm_pos = rand(3, n_vpm)
    vpm_str = randn(3, n_vpm) ./ n_vpm
    vpm = VortexParticles(copy(vpm_pos), copy(vpm_str))
    vpm_ref = VortexParticles(copy(vpm_pos), copy(vpm_str))
    plan = FastMultipole.FmmPlan((vpm,), (vpm,); scalar_potential=false,
        gradient=true, hessian=true, third_derivative=true,
        leaf_size_source=10, expansion_order=10, multipole_acceptance=0.6)
    @test length(plan.m2l_list) > 0    # premise: real far field
    @test length(plan.direct_list) > 0 # premise: real near field
    for step in 1:2
        if step > 1
            # time-step-style strength update through plan reuse
            for i in 1:n_vpm
                b = vpm.bodies[i]
                vpm.bodies[i] = typeof(b)(b.position, b.strength * 1.1, b.sigma)
                r = vpm_ref.bodies[i]
                vpm_ref.bodies[i] = typeof(r)(r.position, r.strength * 1.1, r.sigma)
            end
        end
        reset!(vpm)
        reset!(vpm_ref)
        fmm!((vpm,), (vpm,), plan)
        direct!(vpm_ref; scalar_potential=false, gradient=true, hessian=true,
            third_derivative=true, n_threads=1)
        @test isapprox(vpm.gradient_stretching[i_gradient_vortex, :],
            vpm_ref.gradient_stretching[i_gradient_vortex, :]; rtol=1e-4)
        @test isapprox(vpm.potential[i_HESSIAN_vortex, :],
            vpm_ref.potential[i_HESSIAN_vortex, :]; rtol=1e-4)
        @test isapprox(vpm.potential[i_THIRD_DERIVATIVE_vortex, :],
            vpm_ref.potential[i_THIRD_DERIVATIVE_vortex, :]; rtol=1e-4)
        # downstream stretching update (ω·∇)v consumes the Hessian as in FLOWVPM
        update_gradient_stretching!(vpm)
        @test all(isfinite, vpm.gradient_stretching[i_STRETCHING_vortex, :])
        @test any(!iszero, vpm.gradient_stretching[i_STRETCHING_vortex, :])
    end
end
