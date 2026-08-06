# Task 029 cycle 1: occupancy-epoch window cache + graph-captured far-field
# chain.
#
# Validates, against the ungated per-step path (CUDA_CACHED_WINDOWS = false,
# CUDA_GRAPH_LIFECYCLE = false):
#   1. multi-step convection parity of (a) the cached-window path alone and
#      (b) cached windows + graph capture, at Float64/Float32 and at
#      expansion_order 3 (literature P=4, the standing project rule) and 4,
#      with both a small window chunk (K=8, multi-window concatenation) and
#      effectively-full windows (one window per level);
#   2. occupancy-change invalidation: a mirror-flip teleport of every body
#      forces a new occupancy epoch — the cached windows and the captured
#      graph must regenerate/re-record and agree with a fresh ungated cache
#      built at the moved positions;
#   3. the 023 transfer-counter contract stays flat across graphed steps;
#   4. the profile_stages fallback (graph-ineligible) still computes correct
#      results and reports cached-window telemetry.

using FastMultipole
using FastMultipole.StaticArrays
using Test

_cuda_graph_required() = get(ENV, "FASTMULTIPOLE_REQUIRE_CUDA_TESTS", "0") == "1"

const _GRAPH_LOADED = FastMultipole.load_cuda_radix_lifecycle!()
if _GRAPH_LOADED
    using CUDA
    include(joinpath(@__DIR__, "..", "MATRIX_OPERATOR_REFACTOR", "scripts",
        "fm028_device_system.jl"))
elseif _cuda_graph_required()
    error("FASTMULTIPOLE_REQUIRE_CUDA_TESTS=1 but CUDA radix lifecycle did not load: " *
          FastMultipole.cuda_radix_status())
else
    @info "CUDA radix lifecycle unavailable; skipping task 029 graph tests" FastMultipole.cuda_radix_status()
end

if _GRAPH_LOADED
    const FM = FastMultipole

    # build a cache + system at `bodies`, run `steps` verdict-boundary steps,
    # then one more evaluation; returns sampled outputs + the cache
    function _graph_run(bodies, ::Type{TF}, P, K; steps=3, dt=1e-3,
            teleport::Bool=false) where TF
        n = size(bodies, 2)
        sys = FM028DeviceSystem{TF}(bodies)
        opts = CUDARadixLifecycleOptions(; precision=TF,
            operator=MaterializedYRotationM2L(),
            m2l_strategy=DenseTranslationM2L())
        cache = RadixFMMCache(sys; expansion_order=P, ell=3, max_n_bodies=n,
            bounds=(SVector(-0.01, -0.01, -0.01), 1.02), device=true,
            options=opts, near_radius2=12, window_classes=K)
        for _ in 1:steps
            fmm!(sys, cache; scalar_potential=true, gradient=true)
            fm028_euler!(sys, dt, 0.0, 1.0)
        end
        if teleport
            # mirror-flip: stays inside [0,1]^3 but changes the occupied cell
            # set, forcing a new occupancy epoch mid-run
            sys.positions .= TF(1) .- sys.positions
        end
        # three motionless evaluations: under convection churn every step can
        # open a new occupancy epoch (whose first lifecycle runs uncaptured by
        # design), so a stable-epoch tail guarantees the graph path reaches
        # warm -> record -> replay before sampling
        for _ in 1:3
            fmm!(sys, cache; scalar_potential=true, gradient=true)
        end
        pot, grad = fm028_sampled_output(sys, collect(1:n))
        return pot, grad, sys, cache
    end

    function _with_flags(f, cached::Bool, graph::Bool)
        saved_c = FM.CUDA_CACHED_WINDOWS[]
        saved_g = FM.CUDA_GRAPH_LIFECYCLE[]
        FM.CUDA_CACHED_WINDOWS[] = cached
        FM.CUDA_GRAPH_LIFECYCLE[] = graph
        try
            return f()
        finally
            FM.CUDA_CACHED_WINDOWS[] = saved_c
            FM.CUDA_GRAPH_LIFECYCLE[] = saved_g
        end
    end

    _ptol(::Type{Float64}) = 1e-10
    _ptol(::Type{Float32}) = 1e-4

    @testset "cached-window + graph lifecycle parity (task 029 cycle 1)" begin
        n = 2000
        for TF in (Float64, Float32), P in (3, 4), K in (8, 10_000)
            bodies = fm028_body_matrix(24025, n)
            configs = ((false, false), (true, false), (true, true))
            results = map(configs) do (cached, graph)
                _with_flags(cached, graph) do
                    pot, grad, _, cache = _graph_run(copy(bodies), TF, P, K)
                    (pot, grad, cache)
                end
            end
            ref_pot, ref_grad, _ = results[1]
            for (i, (pot, grad, cache)) in enumerate(results[2:end])
                tol = _ptol(TF)
                @test maximum(abs.(pot .- ref_pot)) <=
                    tol * max(1, maximum(abs.(ref_pot)))
                @test maximum(abs.(grad .- ref_grad)) <=
                    tol * max(1, maximum(abs.(ref_grad)))
                hctx = cache.state.interaction_list
                @test hctx isa FM.DeviceHierarchicalM2LContext
                @test hctx.win_valid
                # the cached concatenation carries the whole-step route total
                @test hctx.total_routes == sum(hctx.win_level_counts)
                @test cache.state.counts.n_routes == hctx.total_routes
                if i == 2   # graphed run: the graph must have been recorded
                    @test hctx.graph_exec !== nothing
                    @test hctx.graph_epoch == hctx.epoch_id
                end
            end
        end
    end

    @testset "occupancy-epoch invalidation (task 029 cycle 1)" begin
        n = 2000
        for TF in (Float64, Float32)
            bodies = fm028_body_matrix(24025, n)
            pot_g, grad_g, sys_g, cache_g = _with_flags(true, true) do
                _graph_run(copy(bodies), TF, 3, 8; teleport=true)
            end
            hctx = cache_g.state.interaction_list
            # the teleport must actually have changed the occupancy epoch
            @test hctx.epoch_id >= 2
            @test hctx.graph_epoch == hctx.epoch_id
            # reference: fresh ungated cache built at the final positions
            moved = copy(bodies)
            moved_pos = Array(sys_g.positions)
            moved[1:3, :] .= Float64.(moved_pos)
            pot_r, grad_r, _, _ = _with_flags(false, false) do
                _graph_run(moved, TF, 3, 8; steps=0)
            end
            tol = _ptol(TF)
            @test maximum(abs.(pot_g .- pot_r)) <=
                tol * max(1, maximum(abs.(pot_r)))
            @test maximum(abs.(grad_g .- grad_r)) <=
                tol * max(1, maximum(abs.(grad_r)))
        end
    end

    @testset "transfer counters flat across graphed steps (task 029 cycle 1)" begin
        n = 2000
        TF = Float32
        bodies = fm028_body_matrix(24025, n)
        _with_flags(true, true) do
            sys = FM028DeviceSystem{TF}(bodies)
            opts = CUDARadixLifecycleOptions(; precision=TF,
                operator=MaterializedYRotationM2L(),
                m2l_strategy=DenseTranslationM2L())
            cache = RadixFMMCache(sys; expansion_order=3, ell=3, max_n_bodies=n,
                bounds=(SVector(-0.01, -0.01, -0.01), 1.02), device=true,
                options=opts, near_radius2=12, window_classes=8)
            # warm past capture (epoch step 1 = warm, step 2 = record)
            for _ in 1:3
                fmm!(sys, cache; scalar_potential=true, gradient=true)
            end
            counters = cache.state.counters
            base = (counters.route_uploads, counters.operator_uploads,
                counters.body_uploads, counters.influence_downloads,
                counters.metadata_downloads)
            for _ in 1:3
                fmm!(sys, cache; scalar_potential=true, gradient=true)
                fm028_euler!(sys, 1e-4, 0.0, 1.0)
            end
            @test counters.route_uploads == base[1]
            @test counters.operator_uploads == base[2]
            @test counters.body_uploads == base[3]
            @test counters.influence_downloads == base[4]
            @test counters.metadata_downloads == base[5]
            @test counters.expansion_host_copies == 0
        end
    end

    @testset "profile_stages fallback under caching (task 029 cycle 1)" begin
        n = 2000
        TF = Float32
        bodies = fm028_body_matrix(24025, n)
        _with_flags(true, true) do
            pot0, grad0, sys, cache = _graph_run(copy(bodies), TF, 3, 8; steps=2)
            hctx = cache.state.interaction_list
            hctx.profile_stages = true
            fmm!(sys, cache; scalar_potential=true, gradient=true)
            hctx.profile_stages = false
            # cached-window telemetry: per-level routes populated, per-step
            # generation cost zero, per-level apply timings measured
            @test sum(hctx.routes_per_level) == hctx.total_routes
            @test hctx.update_stage_ns[4] == 0
            @test any(>(0), hctx.m2l_level_ns)
            pot1, grad1 = fm028_sampled_output(sys, collect(1:n))
            tol = _ptol(TF)
            @test maximum(abs.(pot1 .- pot0)) <=
                tol * max(1, maximum(abs.(pot0)))
            @test maximum(abs.(grad1 .- grad0)) <=
                tol * max(1, maximum(abs.(grad0)))
        end
    end
end
