# Task 029 prototype P2: distributed (2-GPU) decomposition gates.
#
# Validates the script-side work-list-sliced two-GPU decomposition
# (MATRIX_OPERATOR_REFACTOR/scripts/fm029_p2_common.jl) against the single-GPU
# resident pipeline:
#   * slice coverage — the per-level M2L window slices and the nearfield pair
#     slices tile the production work lists exactly (host check);
#   * numerical agreement — each GPU's full result (partial compute + two
#     bitwise allreduce exchanges) vs an on-device Float64 direct reference,
#     and vs the single-GPU output;
#   * deterministic aggregation — the two allreduces are bitwise-symmetric
#     (IEEE a+b == b+a), so the mirrored body states must agree BITWISE after
#     every step, including across convection steps;
#   * per-device transfer-counter invariants on repeated steps; and
#   * per-GPU dual-graph capture (bodies A and B) of the custom lifecycle.
#
# Runs at expansion_order 3 AND 4 (P=4 literature rule). Skips (with a note)
# when fewer than two CUDA devices are visible unless
# FASTMULTIPOLE_REQUIRE_TWOGPU_TESTS=1.

using FastMultipole
using FastMultipole.StaticArrays
using Statistics
using Test

_twogpu_required() = get(ENV, "FASTMULTIPOLE_REQUIRE_TWOGPU_TESTS", "0") == "1"

const _P2_LOADED = FastMultipole.load_cuda_radix_lifecycle!()
if _P2_LOADED
    using CUDA
    include(joinpath(@__DIR__, "..", "MATRIX_OPERATOR_REFACTOR", "scripts",
        "fm028_device_system.jl"))
    include(joinpath(@__DIR__, "..", "MATRIX_OPERATOR_REFACTOR", "scripts",
        "fm029_p2_common.jl"))
elseif _twogpu_required()
    error("FASTMULTIPOLE_REQUIRE_TWOGPU_TESTS=1 but CUDA radix lifecycle did not load: " *
          FastMultipole.cuda_radix_status())
else
    @info "CUDA radix lifecycle unavailable; skipping P2 two-GPU tests" FastMultipole.cuda_radix_status()
end

const _P2_READY = _P2_LOADED && length(CUDA.devices()) >= 2
if _P2_LOADED && !_P2_READY
    msg = "two-GPU P2 tests need 2 CUDA devices; visible: $(length(CUDA.devices()))"
    _twogpu_required() ? error(msg) : @info "skipping: $msg"
end

if _P2_READY

const FM = FastMultipole

@testset "two-GPU work-list decomposition (task 029 P2)" begin
    n = 20000
    ell = 3
    box_min = SVector(-0.01, -0.01, -0.01)
    box_size = 1.02
    dt = 1e-4
    indices = collect(1:2000)
    bar = P2Barrier(2)
    seg = zeros(7, 2)

    # the decomposition drives its own per-device graphs; production graph +
    # nearfield-overlap singletons stay off, window cache on (the M2L slice
    # consumes the cached window stream)
    FM.CUDA_GRAPH_LIFECYCLE[] = false
    FM.CUDA_OVERLAP_NEARFIELD[] = false
    FM.CUDA_CACHED_WINDOWS[] = true

    for (TF, fmt, P) in ((Float64, :off, 3), (Float64, :off, 4), (Float32, :fp16, 3))
        @testset "TF=$TF fmt=$fmt P_exp=$P" begin
            println("[p2test] case TF=$TF fmt=$fmt P=$P: single reference"); flush(stdout)
            FM.DENSE_CUDA_TENSOR_FORMAT[] = fmt
            bodies = fm028_body_matrix(24025, n)
            opts = CUDARadixLifecycleOptions(; precision=TF,
                operator=MaterializedYRotationM2L(),
                m2l_strategy=DenseTranslationM2L())
            kwargs = (; near_radius2=12, window_classes=8)

            # single-GPU reference (device 0)
            CUDA.device!(0)
            ssys = FM028DeviceSystem{TF}(bodies)
            scache = RadixFMMCache(ssys; expansion_order=P, ell, max_n_bodies=n,
                bounds=(box_min, box_size), device=true, options=opts, kwargs...)
            fmm!(ssys, scache; scalar_potential=true, gradient=true)
            spot, sgrad = fm028_sampled_output(ssys, indices)
            dref = fm028_direct_sample_reference(ssys.positions, ssys.strengths, indices)
            ms = fm028_accuracy_metrics(spot, sgrad, dref[1, :], dref[2:4, :])
            @test ms.gradient_rel_rms < 5e-3

            # dual setup: mirrored caches, sliced work lists
            println("[p2test] dual setup"); flush(stdout)
            G = Vector{P2Gpu}(undef, 2)
            info = Vector{Any}(undef, 2)
            for g in 1:2
                G[g], info[g] = p2_setup_gpu!(g - 1, g, bodies, TF, P, ell,
                    kwargs, opts; max_n_bodies=n, bounds=(box_min, box_size),
                    use_graph=true)
            end
            @test G[1].part.c0 == 1 && G[2].part.c1 ==
                G[1].cache.state.counts.n_cells &&
                G[1].part.c1 + 1 == G[2].part.c0
            cover_ok, cover_detail = p2_slice_coverage(G)
            @test cover_ok
            @test info[1].routes == info[2].routes && info[1].routes > 0
            p2_require_peer_access!(G)

            # solo serialized recording, then concurrent replay (no motion)
            println("[p2test] record graphs (solo)"); flush(stdout)
            p2_record_graphs!(G)
            println("[p2test] concurrent replay steps"); flush(stdout)
            for _ in 1:2
                p2_step_pair!(G, bar; dt=0.0, do_euler=false, seg=seg)
            end
            @test G[1].slot.exec_a !== nothing && G[1].slot.exec_b !== nothing
            @test G[2].slot.exec_a !== nothing && G[2].slot.exec_b !== nothing

            # full-result accuracy on BOTH devices: vs direct reference and vs
            # the single-GPU output (not bitwise vs single — atomic ordering
            # differs — but tight)
            tol_ref = 5e-3
            tol_vs_single = TF === Float64 ? 1e-9 : 5e-3
            for g in 1:2
                CUDA.device!(G[g].dev)
                pot, grad = fm028_sampled_output(G[g].sys, indices)
                m = fm028_accuracy_metrics(pot, grad, dref[1, :], dref[2:4, :])
                @test m.gradient_rel_rms < tol_ref
                mvs = fm028_accuracy_metrics(pot, grad, spot, sgrad)
                @test mvs.gradient_rel_rms < tol_vs_single
            end
            # bitwise lockstep at step 0 (allreduce symmetry)
            CUDA.device!(G[1].dev); g1 = Array(G[1].sys.gradient)
            CUDA.device!(G[2].dev); g2 = Array(G[2].sys.gradient)
            @test g1 == g2

            # deterministic aggregation: bitwise lockstep across real steps
            println("[p2test] convection lockstep steps"); flush(stdout)
            base = [let c = G[g].cache.state.counters
                (c.route_uploads, c.operator_uploads, c.body_uploads,
                    c.influence_downloads, c.metadata_downloads,
                    c.expansion_host_copies)
            end for g in 1:2]
            for _ in 1:3
                p2_step_pair!(G, bar; dt, do_euler=true, seg=seg)
            end
            CUDA.device!(G[1].dev)
            p1 = Array(G[1].sys.positions)
            g1 = Array(G[1].sys.gradient)
            CUDA.device!(G[2].dev)
            p2v = Array(G[2].sys.positions)
            g2v = Array(G[2].sys.gradient)
            @test p1 == p2v
            @test g1 == g2v
            # moved-position accuracy vs a fresh direct reference
            p2_step_pair!(G, bar; dt=0.0, do_euler=false, seg=seg)
            CUDA.device!(G[1].dev)
            refc = fm028_direct_sample_reference(G[1].sys.positions,
                G[1].sys.strengths, indices)
            potc, gradc = fm028_sampled_output(G[1].sys, indices)
            mc = fm028_accuracy_metrics(potc, gradc, refc[1, :], refc[2:4, :])
            @test mc.gradient_rel_rms < tol_ref
            for g in 1:2
                c = G[g].cache.state.counters
                @test (c.route_uploads, c.operator_uploads, c.body_uploads,
                    c.influence_downloads, c.metadata_downloads,
                    c.expansion_host_copies) == base[g]
                @test G[g].slot.refilters == 0
            end

            # teardown
            G = nothing
            ssys = nothing
            scache = nothing
            GC.gc()
            for d in 0:1
                CUDA.device!(d)
                CUDA.reclaim()
            end
        end
    end
end

end # _P2_READY
