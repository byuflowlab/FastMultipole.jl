@testset "direct conditioning" begin

    struct OffsetPairMatcher end
    FastMultipole.applies(::OffsetPairMatcher, i_source_system, i_target_system) = i_source_system + 1 == i_target_system

    function scale_strength_rows!(source_buffer, source_system, scale)
        rows = 5:4+FastMultipole.strength_dims(source_system)
        source_buffer[rows, :] .*= scale
        return nothing
    end

    grav = generate_gravitational(11, 16)
    vort = generate_vortex(12, 16)
    fils = generate_filament_field(16, 16^0.333, 13; strength_scale=1/16)
    systems = (grav, vort, fils)
    scalar_potential_switches = (false, false, true)
    gradient_switches = (true, true, true)
    hessian_switches = (false, false, false)

    switches = FastMultipole.DerivativesSwitch(false, true, false, systems)
    source_buffers = FastMultipole.allocate_buffers(systems, false, Float64, switches)
    FastMultipole.system_to_buffer!(source_buffers, systems)
    source_buffer_copies = deepcopy(source_buffers)

    events = Tuple{Symbol,Int,Int}[]
    before! = function (source_buffer, source_system, i_source_system, target_buffer, i_target_system)
        push!(events, (:before, i_source_system, i_target_system))
        scale_strength_rows!(source_buffer, source_system, 2.0)
    end
    after! = function (source_buffer, source_system, i_source_system, target_buffer, i_target_system)
        push!(events, (:after, i_source_system, i_target_system))
        scale_strength_rows!(source_buffer, source_system, 0.5)
    end

    rule = DirectConditioningRule(SelfPairs(), before!, after!)
    direct!(systems, systems; scalar_potential=scalar_potential_switches, gradient=gradient_switches, hessian=hessian_switches, n_threads=1, source_buffers, direct_conditioning=rule)

    @test events == [
        (:before, 1, 1), (:after, 1, 1),
        (:before, 2, 2), (:after, 2, 2),
        (:before, 3, 3), (:after, 3, 3),
    ]
    for i in eachindex(source_buffers)
        @test source_buffers[i] == source_buffer_copies[i]
    end

    empty!(events)
    pair_rule = DirectConditioningRule(PairSet(((1, 2), (3, 1))), before!, after!)
    direct!(systems, systems; scalar_potential=scalar_potential_switches, gradient=gradient_switches, hessian=hessian_switches, n_threads=1, source_buffers, direct_conditioning=(pair_rule,))
    @test events == [
        (:before, 1, 2), (:after, 1, 2),
        (:before, 3, 1), (:after, 3, 1),
    ]

    empty!(events)
    custom_rule = DirectConditioningRule(OffsetPairMatcher(), before!, after!)
    direct!(systems, systems; scalar_potential=scalar_potential_switches, gradient=gradient_switches, hessian=hessian_switches, n_threads=1, source_buffers, direct_conditioning=custom_rule)
    @test events == [
        (:before, 1, 2), (:after, 1, 2),
        (:before, 2, 3), (:after, 2, 3),
    ]

    empty!(events)
    direct!(systems, systems; scalar_potential=scalar_potential_switches, gradient=gradient_switches, hessian=hessian_switches, n_threads=2, source_buffers, direct_conditioning=rule)
    @test events == [
        (:before, 1, 1), (:after, 1, 1),
        (:before, 2, 2), (:after, 2, 2),
        (:before, 3, 3), (:after, 3, 3),
    ]

    empty!(events)
    fmm!(systems; scalar_potential=scalar_potential_switches, gradient=gradient_switches, hessian=hessian_switches, expansion_order=3, leaf_size=8, multipole_acceptance=0.2, direct_conditioning=rule, silence_warnings=true)
    @test events == [
        (:before, 1, 1), (:after, 1, 1),
        (:before, 2, 2), (:after, 2, 2),
        (:before, 3, 3), (:after, 3, 3),
    ]

    miss_rule = DirectConditioningRule(PairSet(((0, 0),)), before!, after!)
    perf_systems = (
        generate_gravitational(21, 48),
        generate_vortex(22, 48),
        generate_filament_field(48, 48^0.333, 23; strength_scale=1/48),
    )

    direct!(perf_systems, perf_systems; scalar_potential=scalar_potential_switches, gradient=gradient_switches, hessian=hessian_switches, n_threads=1)
    direct!(perf_systems, perf_systems; scalar_potential=scalar_potential_switches, gradient=gradient_switches, hessian=hessian_switches, n_threads=1, direct_conditioning=miss_rule)

    t_plain = @elapsed direct!(perf_systems, perf_systems; scalar_potential=scalar_potential_switches, gradient=gradient_switches, hessian=hessian_switches, n_threads=1)
    t_miss = @elapsed direct!(perf_systems, perf_systems; scalar_potential=scalar_potential_switches, gradient=gradient_switches, hessian=hessian_switches, n_threads=1, direct_conditioning=miss_rule)
    alloc_plain = @allocated direct!(perf_systems, perf_systems; scalar_potential=scalar_potential_switches, gradient=gradient_switches, hessian=hessian_switches, n_threads=1)
    alloc_miss = @allocated direct!(perf_systems, perf_systems; scalar_potential=scalar_potential_switches, gradient=gradient_switches, hessian=hessian_switches, n_threads=1, direct_conditioning=miss_rule)

    @test alloc_miss <= alloc_plain + 20_000
    @test t_miss <= max(0.02, 2.5 * t_plain)

end
