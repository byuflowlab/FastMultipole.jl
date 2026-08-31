using FastMultipole
using FastMultipole.StaticArrays
using Random
using Test

if !isdefined(@__MODULE__, :generate_gravitational)
    include("gravitational.jl")
end

# Collect every Array anywhere in the step-varying state (state fields, grid
# arrays, workspace group columns, concat-plan route classes) keyed by path.
function _radix_state_arrays(state)
    arrays = Dict{String,Any}()
    for field in fieldnames(typeof(state))
        value = getfield(state, field)
        value isa Array && (arrays["state.$field"] = value)
    end
    grid = state.grid
    for field in fieldnames(typeof(grid))
        value = getfield(grid, field)
        value isa Array && (arrays["grid.$field"] = value)
    end
    ws = state.scratch
    arrays["scratch.nonleaf_idx"] = ws.nonleaf_idx
    for (kind, groups) in (("m2m", ws.m2m_groups), ("l2l", ws.l2l_groups))
        for (gi, group) in enumerate(groups)
            arrays["scratch.$kind[$gi].source_idx"] = group.source_idx
            arrays["scratch.$kind[$gi].target_idx"] = group.target_idx
            arrays["scratch.$kind[$gi].phis"] = group.phis
            arrays["scratch.$kind[$gi].thetas"] = group.thetas
        end
    end
    plan = ws.m2l_concat
    arrays["scratch.m2l_plan.route_class"] = plan.route_class
    if plan isa FastMultipole.ResidentM2LFactoredPlan
        for (gi, group) in enumerate(plan.groups)
            arrays["scratch.m2l[$gi].source_idx"] = group.source_idx
            arrays["scratch.m2l[$gi].target_idx"] = group.target_idx
            arrays["scratch.m2l[$gi].phis"] = group.phis
            arrays["scratch.m2l[$gi].thetas"] = group.thetas
        end
    end
    arrays["state.multipoles.phi"] = state.multipoles.phi
    arrays["state.locals.phi"] = state.locals.phi
    return arrays
end


@testset "factored resident M2L lifecycle (task 023a)" begin
    seed = 2301
    full = generate_gravitational(seed, 400)
    opts = CUDARadixLifecycleOptions(; operator=FactoredRotationM2L(),
        m2l_strategy=FastMultipole.ConcatenatedFixedZM2L())
    # The grouped-factored resident plan and its `.groups` are FLAT-path structures:
    # under the task-027 hierarchical default the factored selection routes through
    # the bounded concat engine. Pin this testset to the flat classifier.
    cache = RadixFMMCache(full; expansion_order=8, ell=3, max_n_bodies=400,
        bounds=(SVector(-0.5, -0.5, -0.5), 2.0), options=opts,
        stencil_epsilon=1e-4)
    fmm!(full, cache; scalar_potential=true, gradient=true)
    FastMultipole._launch_resident_m2l!(cache.state) # warm stage specialization
    @test @allocated(FastMultipole._launch_resident_m2l!(cache.state)) <= 64 * 1024
    captured = _radix_state_arrays(cache.state)
    for (step, n) in enumerate((173, 400, 251, 320))
        bodies = [Body(clamp.(b.position .+ SVector(1e-3 * step, -5e-4 * step,
            7e-4 * step), -0.45, 1.45), b.radius, b.strength) for b in full.bodies[1:n]]
        sys = Gravitational(bodies, zeros(16, n))
        ref = Gravitational(copy(bodies), zeros(16, n))
        FastMultipole.direct!(ref; scalar_potential=true, gradient=true)
        fmm!(sys, cache; scalar_potential=true, gradient=true)
        @test maximum(abs.(sys.potential[1, :] .- ref.potential[1, :])) < 1e-6
        @test maximum(abs.(sys.potential[5:7, :] .- ref.potential[5:7, :])) < 1e-4
        current = _radix_state_arrays(cache.state)
        @test keys(current) == keys(captured)
        for (path, array) in captured
            @test current[path] === array
        end
        groups = cache.state.scratch.m2l_concat.groups
        @test sum(g.count[] for g in groups) == cache.state.counts.n_routes
        @test all(g.count[] <= cache.max_cells for g in groups)
    end
    @test @allocated(fmm!(full, cache; scalar_potential=true, gradient=true)) < 512_000

    # A low-order/small-grid stencil can have no accepted far-field offsets; the
    # factored launcher still clears locals and returns without rebuilding storage.
    empty_sys = generate_gravitational(seed + 1, 40)
    empty_cache = RadixFMMCache(empty_sys; expansion_order=4, ell=2, options=opts,
        stencil_epsilon=1e-4)
    @test empty_cache.state.counts.n_routes == 0
    @test isempty(empty_cache.state.scratch.m2l_concat.groups)
    @test FastMultipole._launch_resident_m2l!(empty_cache.state) === empty_cache.state
end

@testset "radix fmm! mock time stepping (task 023)" begin
    N = 1500
    P = 8
    seed = 4023
    rng = MersenneTwister(seed)
    full = generate_gravitational(seed, N)

    # fixed box covering the whole jittered trajectory
    cache = RadixFMMCache(full; expansion_order=P, ell=3, max_n_bodies=N,
        bounds=(SVector(-0.5, -0.5, -0.5), 2.0))
    @test cache.state.counts.n_bodies == N
    # warm up compilation so the timing/allocation checks measure steady state
    fmm!(full, cache; scalar_potential=true, gradient=true)

    captured = nothing
    warm_allocs = Int[]
    t_lifecycle = 0.0
    t_step = 0.0
    nsteps = 10
    for step in 1:nsteps
        # jitter positions/strengths and vary n in [N/2, N]
        n_step = step == 1 ? N : rand(rng, (N ÷ 2):N)
        bodies = Vector{Body{Float64}}(undef, n_step)
        for i in 1:n_step
            b = full.bodies[i]
            pos = clamp.(b.position .+ 0.02 .* (rand(rng, SVector{3,Float64}) .- 0.5), -0.45, 1.45)
            bodies[i] = Body(pos, b.radius, b.strength * (1 + 0.01 * randn(rng)))
            i <= length(full.bodies) && (full.bodies[i] = Body(pos, b.radius, bodies[i].strength))
        end
        sys = Gravitational(bodies, zeros(16, n_step))

        t0 = time()
        fmm!(sys, cache; scalar_potential=true, gradient=true)
        t_step += time() - t0

        # per-step accuracy vs direct!
        ref = Gravitational(copy(bodies), zeros(16, n_step))
        FastMultipole.direct!(ref; scalar_potential=true, gradient=true)
        @test maximum(abs.(sys.potential[1, :] .- ref.potential[1, :])) < 1e-6
        @test maximum(abs.(sys.potential[5:7, :] .- ref.potential[5:7, :])) < 1e-4
        @test cache.state.counts.n_bodies == n_step

        if step == 1
            captured = _radix_state_arrays(cache.state)
        else
            # the non-flaky zero-reallocation proof: every array identity survives
            current = _radix_state_arrays(cache.state)
            @test keys(current) == keys(captured)
            for (path, array) in captured
                @test current[path] === array
            end
        end
        if step > 3
            t1 = time()
            run_host_radix_lifecycle!(cache.state)
            t_lifecycle += time() - t1
            push!(warm_allocs, @allocated fmm!(sys, cache; scalar_potential=true, gradient=true))
        end
    end

    # small fixed allocation budget after warmup (views + dynamic dispatch only)
    @test maximum(warm_allocs) < 512_000
    # host path never counts transfers
    @test cache.state.counters.expansion_host_copies == 0
    @test cache.state.counters.route_uploads == 0
    @test cache.state.counters.operator_uploads == 0
    @test cache.state.counters.body_uploads == 0
    # loose timing sanity: a full step (update + lifecycle + finalize) should be
    # within a small factor of the bare lifecycle (hard guarantees are the
    # identity/counter checks above)
    avg_step = t_step / nsteps
    avg_lifecycle = t_lifecycle / length(warm_allocs)
    @test avg_step < 10 * avg_lifecycle
end
