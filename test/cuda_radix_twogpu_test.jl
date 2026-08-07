# Task 029 prototype P2: distributed (2-GPU) decomposition gates.
#
# Validates the script-side mirrored-source / partitioned-target two-GPU
# decomposition (MATRIX_OPERATOR_REFACTOR/scripts/fm029_p2_common.jl) against
# the single-GPU resident pipeline:
#   * partition exactness — the per-GPU filtered M2L route windows and direct
#     pair lists exactly partition the single-cache sets (counts + xor
#     recombination, per level; no missing or double-counted interaction);
#   * partition-independent numerical agreement — union sampled output vs an
#     on-device Float64 direct reference, and vs the single-GPU output;
#   * deterministic aggregation — the mirrored body states stay in bitwise
#     lockstep across convection steps (the exchange is a bitwise copy);
#   * per-device transfer-counter invariants on repeated steps; and
#   * per-GPU graph capture of the custom lifecycle body.
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

@testset "two-GPU octant decomposition (task 029 P2)" begin
    n = 20000
    ell = 3
    box_min = SVector(-0.01, -0.01, -0.01)
    box_size = 1.02
    dt = 1e-4
    indices = collect(1:2000)
    bar = P2Barrier(2)
    seg = zeros(7, 2)

    # the decomposition drives its own per-device graphs; production graph +
    # nearfield-overlap singletons stay off, window cache on
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

            # single-GPU reference (device 0): full sets + output
            CUDA.device!(0)
            ssys = FM028DeviceSystem{TF}(bodies)
            scache = RadixFMMCache(ssys; expansion_order=P, ell, max_n_bodies=n,
                bounds=(box_min, box_size), device=true, options=opts, kwargs...)
            fmm!(ssys, scache; scalar_potential=true, gradient=true)
            sw = p2_window_signature(scache.state)
            sd = p2_direct_signature(scache.state)
            spot, sgrad = fm028_sampled_output(ssys, indices)
            dref = fm028_direct_sample_reference(ssys.positions, ssys.strengths, indices)
            ms = fm028_accuracy_metrics(spot, sgrad, dref[1, :], dref[2:4, :])
            @test ms.gradient_rel_rms < 5e-3

            # dual setup: mirrored caches, one target half each
            println("[p2test] dual setup"); flush(stdout)
            G = Vector{P2Gpu}(undef, 2)
            info = Vector{Any}(undef, 2)
            for g in 1:2
                G[g], info[g] = p2_setup_gpu!(g - 1, g, bodies, TF, P, ell,
                    kwargs, opts; max_n_bodies=n, bounds=(box_min, box_size),
                    use_graph=true)
            end
            # both mirrored caches saw the same full sets as the single cache
            @test info[1].full_windows.counts == sw.counts
            @test info[1].full_windows.xors == sw.xors
            @test info[1].full_direct.count == sd.count &&
                  info[1].full_direct.xor == sd.xor
            # forensics: where do the window targets actually live relative to
            # the partition's per-level node ranges? (job 13064752: the filter
            # was a no-op — g=1 kept all 80472 routes, g=2 kept zero, and the
            # count+xor gate is blind to that degenerate split)
            # frame check: do the three caches even share a body permutation?
            # (GPU counting-sort nondeterminism would break the sorted-frame
            # diagnostics AND the b0:b1 exchange-block body alignment)
            let hp1 = G[1].cache.state.host_body_perm,
                    hp2 = G[2].cache.state.host_body_perm,
                    hps = scache.state.host_body_perm
                println("[p2diag3] perm mismatches g1-vs-g2 = ", count(hp1 .!= hp2),
                    "  g1-vs-single = ", count(hp1 .!= hps), " of ", length(hp1))
            end
            println("[p2diag3] post-setup filtered route counts g=1,g=2 = ",
                sum(G[1].cache.state.interaction_list.win_level_counts), ", ",
                sum(G[2].cache.state.interaction_list.win_level_counts))
            CUDA.device!(0)
            FM.update_cuda_radix_state!(G[1].cache, (G[1].sys,))
            println("[p2diag3] post-update route count g=1 = ",
                sum(G[1].cache.state.interaction_list.win_level_counts),
                " (regenerated if it snapped back to the full count)")
            flush(stdout)
            let st1 = G[1].cache.state, hctx = st1.interaction_list
                println("[p2diag3] window-target forensics (g=1 cache, post-filter):")
                for L in 2:hctx.ell
                    s = hctx.win_level_starts[L + 1]
                    n = hctx.win_level_counts[L + 1]
                    n == 0 && (println("  L=$L n=0"); continue)
                    wtv = Array(view(hctx.win_targets, (s + 1):(s + n)))
                    lo1, hi1 = G[1].part.lo[L + 1], G[1].part.hi[L + 1]
                    lo2, hi2 = G[2].part.lo[L + 1], G[2].part.hi[L + 1]
                    c1 = count(t -> lo1 <= t <= hi1, wtv)
                    c2 = count(t -> lo2 <= t <= hi2, wtv)
                    println("  L=$L n=$n tgt_extrema=", extrema(wtv),
                        " own1=[$lo1,$hi1] in1=$c1 own2=[$lo2,$hi2] in2=$c2",
                        " neither=", n - c1 - c2)
                end
                println("[p2diag3] level_offsets=", collect(G[1].cache.level_offsets))
                flush(stdout)
            end
            # setup-time stock accuracy on the mirrored caches (fmm! ran inside
            # p2_setup_gpu! before any filtering; sys still holds its results)
            for g in 1:2
                CUDA.device!(G[g].dev)
                pot0, grad0 = fm028_sampled_output(G[g].sys, indices)
                m0 = fm028_accuracy_metrics(pot0, grad0, dref[1, :], dref[2:4, :])
                println("[p2diag3] setup fmm! g=$g grad relrms vs direct = ",
                    m0.gradient_rel_rms)
            end
            flush(stdout)
            ok, _ = p2_partition_exact(G, info[1].full_windows, info[1].full_direct,
                info[2].full_windows, info[2].full_direct)
            @test ok
            @test G[1].part.b0 == 1 && G[2].part.b1 == n &&
                  G[1].part.b1 + 1 == G[2].part.b0

            p2_require_peer_access!(G)

            # ---- [p2diag4] frame + user-order step-path bisection -----------
            # 13065445 paradox: stock fmm! is user-order perfect (2.9e-4) with
            # identical perms across all three caches, yet EVERY sorted-frame
            # comparison of state.output sits at ~0.4-0.5 — so the assumed
            # sorted<->user frame map must be verified empirically before any
            # sorted-frame number is trusted (the exchange block math depends
            # on the same map).
            let me = G[1], st1 = me.cache.state
                CUDA.device!(me.dev)
                r2(a, b) = sqrt(mean(abs2, Float64.(a) .- Float64.(b))) /
                    sqrt(mean(abs2, Float64.(b)))
                fmm!(me.sys, me.cache; scalar_potential=true, gradient=true)
                CUDA.synchronize()
                o1 = Array(st1.output)
                sg = Array(me.sys.gradient)
                nb = size(sg, 2)
                hidx = Array(st1.host_body_indices)[1:nb]
                d1 = r2(o1[2:4, 1:nb], sg[:, hidx])          # sorted j -> user hidx[j]
                inv = zeros(Int, nb); for j in 1:nb; inv[hidx[j]] = j; end
                d2 = r2(o1[2:4, inv], sg)                    # user u -> sorted inv[u]
                println("[p2diag4] frame d1 (out col j ~ user hidx[j]) = ", d1)
                println("[p2diag4] frame d2 (out col inv[u] ~ user u)  = ", d2)
                # user-order accuracy of MY step path (no exchange, no euler),
                # split into owned-user and complement samples
                sorted_ok = min(d1, d2) < 1e-3
                owned = sorted_ok ? Set(hidx[me.part.b0:me.part.b1]) : Set{Int}()
                selo = [k for (k, i) in enumerate(indices) if i in owned]
                selc = [k for (k, i) in enumerate(indices) if !(i in owned)]
                step_metrics(tag) = begin
                    CUDA.synchronize()
                    pot, grad = fm028_sampled_output(me.sys, indices)
                    mo = isempty(selo) ? NaN : fm028_accuracy_metrics(pot[selo],
                        grad[:, selo], dref[1, selo], dref[2:4, selo]).gradient_rel_rms
                    mcpl = isempty(selc) ? NaN : fm028_accuracy_metrics(pot[selc],
                        grad[:, selc], dref[1, selc], dref[2:4, selc]).gradient_rel_rms
                    println("[p2diag4] ", tag, " owned(", length(selo), ")=", mo,
                        " complement(", length(selc), ")=", mcpl)
                    flush(stdout)
                end
                my_finalize!() = FM.finalize_cuda_radix_output!(st1, (me.sys,);
                    derivatives_switches=me.switches,
                    host_output_staging=me.cache.device_ctx.host_output,
                    target_buffers=me.target_buffers,
                    device_target_buffers=me.cache.device_ctx.device_target_buffers)
                # (a) my worker path minus exchange/euler
                FM.update_cuda_radix_state!(me.cache, (me.sys,))
                p2_check_epoch!(me)
                p2_run_lifecycle!(me.slot, st1, me.part.c0, me.part.c1, me.side,
                    me.ev_begin, me.ev_done; use_graph=false)
                my_finalize!()
                step_metrics("a) my body + my finalize   ")
                # (b) production body + my finalize
                FM.update_cuda_radix_state!(me.cache, (me.sys,))
                FM.run_cuda_radix_lifecycle!(st1)
                my_finalize!()
                step_metrics("b) prod body + my finalize ")
                # (c) stock fmm! recovery sanity
                fmm!(me.sys, me.cache; scalar_potential=true, gradient=true)
                step_metrics("c) stock fmm! recovery     ")
            end

            # ---- solo isolation probes (device 0, owned half only) ----------
            # A: filtered body, uncaptured        -> filter/body correctness
            # C: filtered windows, FULL L2B range -> splits L2B restriction
            # D: regenerated FULL windows + full L2B -> custom body vs stock
            # B (after recording): solo graph replay -> capture/replay delta
            CUDA.device!(0)
            sout_probe = Array(scache.state.output)
            let me = G[1], st1 = me.cache.state
                rng = me.part.b0:me.part.b1
                den = sqrt(mean(abs2, Float64.(sout_probe[2:4, rng])))
                relrms(o) = sqrt(mean(abs2,
                    Float64.(o[2:4, rng]) .- Float64.(sout_probe[2:4, rng]))) / den
                CUDA.device!(me.dev)
                solo(c0, c1) = begin
                    FM.update_cuda_radix_state!(me.cache, (me.sys,))
                    p2_lifecycle_body!(st1, c0, c1, me.side, me.ev_begin, me.ev_done)
                    CUDA.synchronize()
                    relrms(Array(st1.output))
                end
                ncell = st1.counts.n_cells
                # R: stock fmm! on this mirrored cache — validates the sorted-
                # frame comparison itself (expect ~0 vs the single cache)
                fmm!(me.sys, me.cache; scalar_potential=true, gradient=true)
                CUDA.synchronize()
                println("[p2diag3] R stock fmm! on mirrored cache  = ",
                    relrms(Array(st1.output)))
                # E: production lifecycle body on the same cache/window state
                FM.update_cuda_radix_state!(me.cache, (me.sys,))
                FM.run_cuda_radix_lifecycle!(st1)
                CUDA.synchronize()
                println("[p2diag3] E production body, same windows = ",
                    relrms(Array(st1.output)))
                # F: my launcher sequence, inline nearfield, full L2B
                FM.update_cuda_radix_state!(me.cache, (me.sys,))
                FM._launch_cuda_nearfield_kernel!(st1)
                FM._launch_cuda_b2m!(st1)
                FM._launch_cuda_resident_m2m!(st1)
                FM._launch_cuda_resident_m2l!(st1)
                FM._launch_cuda_resident_l2l!(st1)
                p2_l2b_owned!(st1, 1, ncell)
                CUDA.synchronize()
                println("[p2diag3] F my sequence, inline NF        = ",
                    relrms(Array(st1.output)))
                println("[p2diag2] A filtered body, owned L2B      = ",
                    solo(me.part.c0, me.part.c1))
                println("[p2diag2] C filtered windows, full L2B    = ", solo(1, ncell))
                hctx1 = st1.interaction_list
                nw_filtered = sum(hctx1.win_level_counts)
                hctx1.win_valid = false            # force full regeneration
                println("[p2diag2] D full windows, full L2B        = ", solo(1, ncell))
                p2_filter_windows!(st1, me.part)   # restore the filtered state
                p2_filter_direct!(st1, me.part)
                println("[p2diag2] window counts filtered/full/refiltered = ",
                    nw_filtered, "/", hctx1.total_routes, "/",
                    sum(hctx1.win_level_counts))
                flush(stdout)
            end

            # solo serialized recording, then concurrent replay (no motion)
            println("[p2test] record graphs (solo)"); flush(stdout)
            p2_record_graphs!(G)
            let me = G[1], st1 = me.cache.state
                if me.slot.exec !== nothing
                    CUDA.device!(me.dev)
                    FM.update_cuda_radix_state!(me.cache, (me.sys,))
                    CUDA.launch(me.slot.exec::CUDA.CuGraphExec)
                    CUDA.synchronize()
                    o = Array(st1.output)
                    rng = me.part.b0:me.part.b1
                    den = sqrt(mean(abs2, Float64.(sout_probe[2:4, rng])))
                    println("[p2diag2] B solo graph replay, owned L2B  = ",
                        sqrt(mean(abs2, Float64.(o[2:4, rng]) .-
                            Float64.(sout_probe[2:4, rng]))) / den)
                    flush(stdout)
                end
            end
            println("[p2test] concurrent replay steps"); flush(stdout)
            for _ in 1:2
                p2_step_pair!(G, bar; dt=0.0, do_euler=false, seg=seg)
            end
            @test G[1].slot.exec !== nothing
            @test G[2].slot.exec !== nothing

            # union accuracy: vs direct reference and vs the single-GPU output
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

            # diagnostic: localize any union error to compute (self half) vs
            # exchange (peer half) in sorted coordinates against the single
            # cache's raw output (identical sorts => identical columns)
            CUDA.device!(0)
            sout = Array(scache.state.output)
            for g in 1:2
                CUDA.device!(G[g].dev)
                o = Array(G[g].cache.state.output)
                for h in 1:2
                    rng = G[h].part.b0:G[h].part.b1
                    num = sqrt(mean(abs2, Float64.(o[2:4, rng]) .- Float64.(sout[2:4, rng])))
                    den = sqrt(mean(abs2, Float64.(sout[2:4, rng])))
                    tag = h == g ? "self" : "peer"
                    println("[p2diag] dev g=$g half=$h ($tag) grad relrms vs single = ",
                        num / den)
                end
            end
            mepa = nothing
            for mod in (CUDA, isdefined(CUDA, :CUDACore) ? CUDA.CUDACore : CUDA)
                isdefined(mod, :maybe_enable_peer_access) &&
                    (mepa = getfield(mod, :maybe_enable_peer_access); break)
            end
            if mepa !== nothing
                CUDA.device!(0)
                println("[p2diag] peer access 0->1: ",
                    mepa(CUDA.CuDevice(0), CUDA.CuDevice(1)),
                    "  1->0: ", mepa(CUDA.CuDevice(1), CUDA.CuDevice(0)))
            else
                println("[p2diag] maybe_enable_peer_access not found")
            end
            flush(stdout)

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
