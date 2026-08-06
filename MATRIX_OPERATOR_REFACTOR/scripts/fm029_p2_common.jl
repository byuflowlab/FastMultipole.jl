# Task 029 prototype P2: shared 2-GPU octant-decomposition machinery.
#
# Included AFTER `using FastMultipole`, `using CUDA`, and
# fm028_device_system.jl, by prototype_029_p2_twogpu.jl and
# test/cuda_radix_twogpu_test.jl. Script/test side only — no production src/
# surface is modified; everything here drives the UNCHANGED cycle-1 resident
# pipeline through its public/internal launchers.
#
# Decomposition (documented in the 029 task file, P2 section):
#   * MIRRORED SOURCES, PARTITIONED TARGETS. Both GPUs hold the full body set
#     and construct byte-identical full-domain RadixFMMCaches (same bounds,
#     ell, policy; identical counting sorts => identical trees, node indexing,
#     and cached route windows). Precondition, asserted: full occupancy
#     (8^L nodes at every level), which makes the "first half of the nodes at
#     each level" an ancestor-closed set of 4 complete level-1 subtrees.
#   * GPU g owns target half g: its cached M2L route windows are filtered
#     in place to routes whose TARGET node lies in the owned half (every
#     level), its direct pairs to pairs whose TARGET leaf cell is owned, and
#     L2B is restricted to the owned (contiguous) leaf-cell range. B2M/M2M/
#     L2L and the refresh run on the full mirrored set (replicated work —
#     measured and reported as such; an 8-GPU production design would
#     partition sources too, see the task-file P2 notes).
#   * The far-field chain + side-stream nearfield is captured per GPU into a
#     script-owned CUDA graph (the production graph machinery's global
#     nearfield stream/event singletons are single-device; here each GPU gets
#     its own side stream + events, so production flags CUDA_GRAPH_LIFECYCLE
#     and CUDA_OVERLAP_NEARFIELD must be OFF while CUDA_CACHED_WINDOWS stays
#     ON). Filtered window counts and n_direct are baked at capture; any
#     occupancy-epoch change re-filters and re-records (counted, and its cost
#     lands in the measured step — the same recurrence rule as cycle 1).
#   * The ONLY recurring exchange: after the graphs complete, each GPU
#     P2P-pushes its owned contiguous sorted output block (rows 1:4 x owned
#     bodies) into the peer's output buffer, ordered by cross-device events
#     (no host sync inside the exchange). Both GPUs then run finalize + Euler
#     over the full body set on identical data, so the mirrored body states
#     advance in bitwise lockstep and positions never need exchanging.

using Printf

# ---- tiny two-party spin barrier (host orchestration, measured in the step) --

mutable struct P2Barrier
    const n::Int
    count::Threads.Atomic{Int}
    gen::Threads.Atomic{Int}
    abort::Threads.Atomic{Int}   # a failed worker flips this so its partner
                                 # errors out instead of spinning forever
end
P2Barrier(n::Int) = P2Barrier(n, Threads.Atomic{Int}(0), Threads.Atomic{Int}(0),
    Threads.Atomic{Int}(0))

function p2_barrier_wait!(b::P2Barrier; timeout_s::Float64=120.0)
    b.abort[] == 0 || error("P2 barrier: peer worker aborted")
    g = b.gen[]
    if Threads.atomic_add!(b.count, 1) == b.n - 1
        b.count[] = 0
        Threads.atomic_add!(b.gen, 1)
    else
        t0 = time()
        while b.gen[] == g
            if b.abort[] != 0
                error("P2 barrier: peer worker aborted")
            elseif time() - t0 > timeout_s
                b.abort[] = 1
                error("P2 barrier: timed out after $(timeout_s)s waiting for peer")
            end
            GC.safepoint()
            yield()          # keeps few-thread runs live; ~us-scale cost
        end
    end
    return nothing
end

# ---- partition ---------------------------------------------------------------

# Owned ranges for half `half` in (1, 2): per-level absolute node index range
# (levels 1:ell; level 0 owns nothing — no routes exist there), owned leaf-cell
# range c0:c1, and the owned sorted-body range b0:b1. Requires full occupancy.
function p2_partition(cache, half::Int)
    state = cache.state
    hctx = state.interaction_list
    ell = hctx.ell
    offs = cache.level_offsets
    lo = zeros(Int, ell + 1)
    hi = zeros(Int, ell + 1)
    for L in 0:ell
        nL = offs[L + 2] - offs[L + 1]
        nL == 8^L || error("P2 precondition violated: level $L has $nL of $(8^L) nodes")
        if L == 0
            lo[L + 1] = 1
            hi[L + 1] = 0
        else
            h = nL ÷ 2
            lo[L + 1] = offs[L + 1] + (half == 1 ? 1 : h + 1)
            hi[L + 1] = offs[L + 1] + (half == 1 ? h : nL)
        end
    end
    ncell = state.counts.n_cells
    h = ncell ÷ 2
    c0 = half == 1 ? 1 : h + 1
    c1 = half == 1 ? h : ncell
    first_c0 = Int(Array(view(state.cell_ranges, 1:1, c0:c0))[1])
    first_c1 = Int(Array(view(state.cell_ranges, 1:1, c1:c1))[1])
    count_c1 = Int(Array(view(state.cell_ranges, 2:2, c1:c1))[1])
    b0 = first_c0
    b1 = first_c1 + count_c1 - 1
    return (; half, ell, lo, hi, c0, c1, b0, b1)
end

# ---- route / direct-pair filtering (occupancy-epoch recurrence only) --------

_p2_xor(class, src, tgt) = hash((class, src, tgt))

# Per-level (count, xor-checksum) of the CURRENT window cache — order- and
# partition-independent identity used by the exactness gate.
function p2_window_signature(state)
    hctx = state.interaction_list
    counts = Int[]
    xors = UInt64[]
    for L in 2:hctx.ell
        s = hctx.win_level_starts[L + 1]
        n = hctx.win_level_counts[L + 1]
        x = UInt64(0)
        if n > 0
            rng = (s + 1):(s + n)
            wc = Array(view(hctx.win_class, rng))
            wsv = Array(view(hctx.win_sources, rng))
            wtv = Array(view(hctx.win_targets, rng))
            for i in 1:n
                x ⊻= _p2_xor(wc[i], wsv[i], wtv[i])
            end
        end
        push!(counts, n)
        push!(xors, x)
    end
    return (; counts, xors)
end

function p2_direct_signature(state)
    nd = state.counts.n_direct
    x = UInt64(0)
    if nd > 0
        dt = Array(view(state.direct_targets, 1:nd))
        ds = Array(view(state.direct_sources, 1:nd))
        for i in 1:nd
            x ⊻= hash((dt[i], ds[i]))
        end
    end
    return (; count=nd, xor=x)
end

# In-place filter of the cached per-level route windows to owned-target routes.
# Level starts stay fixed; only the per-level counts shrink (exactly what the
# cached M2L launcher and the graph capture read). Host round-trip is fine: it
# recurs only on occupancy-epoch change, like the window generation itself.
function p2_filter_windows!(state, part)
    hctx = state.interaction_list
    for L in 2:hctx.ell
        s = hctx.win_level_starts[L + 1]
        n = hctx.win_level_counts[L + 1]
        n == 0 && continue
        rng = (s + 1):(s + n)
        wc = Array(view(hctx.win_class, rng))
        wsv = Array(view(hctx.win_sources, rng))
        wtv = Array(view(hctx.win_targets, rng))
        lo = part.lo[L + 1]
        hi = part.hi[L + 1]
        keep = (wtv .>= lo) .& (wtv .<= hi)
        k = count(keep)
        if k > 0
            copyto!(view(hctx.win_class, (s + 1):(s + k)), wc[keep])
            copyto!(view(hctx.win_sources, (s + 1):(s + k)), wsv[keep])
            copyto!(view(hctx.win_targets, (s + 1):(s + k)), wtv[keep])
        end
        hctx.win_level_counts[L + 1] = k
        hctx.routes_per_level[L + 1] = k
    end
    hctx.total_routes = sum(hctx.win_level_counts)
    state.counts.n_routes = hctx.total_routes
    return state
end

# In-place filter of the direct pair list to owned-target-cell pairs. Both
# counts.n_direct (this step's launches) and hctx.epoch_n_direct (what the
# refresh restores while occupancy is unchanged) are set to the kept count.
function p2_filter_direct!(state, part)
    hctx = state.interaction_list
    nd = state.counts.n_direct
    nd == 0 && return state
    dt = Array(view(state.direct_targets, 1:nd))
    ds = Array(view(state.direct_sources, 1:nd))
    keep = (dt .>= part.c0) .& (dt .<= part.c1)
    k = count(keep)
    if k > 0
        copyto!(view(state.direct_targets, 1:k), dt[keep])
        copyto!(view(state.direct_sources, 1:k), ds[keep])
    end
    state.counts.n_direct = k
    hctx.epoch_n_direct = k
    return state
end

# ---- restricted L2B + custom captured lifecycle body ------------------------

# Warp-per-cell L2B over the owned contiguous leaf-cell range only. Same
# production kernel; the views re-base cell indexing while body indices
# (cell_ranges values), node indices (leaf_to_node values), and the output
# stay absolute. 4-row output (hessian off) asserted at setup.
function p2_l2b_owned!(state::FastMultipole.DeviceResidentRadixState{TF,B,LH},
        c0::Int, c1::Int) where {TF,B,LH}
    threads = 128
    orders = state.invariant_cache.basis_info.orders
    ncell = c1 - c0 + 1
    blocks = cld(ncell, threads ÷ 32)
    blocks > 0 && CUDA.@cuda threads=threads blocks=blocks FastMultipole._cuda_l2b_output_kernel!(
        state.output, state.source_bodies,
        view(state.cell_centers, :, c0:c1),
        view(state.cell_ranges, :, c0:c1),
        view(state.grid.leaf_to_node, c0:c1),
        state.locals.phi, state.locals.chi,
        orders.P_phi, orders.P_active, Val(LH), ncell)
    return state
end

# Mirror of the production _cuda_lifecycle_body! with (a) per-device side
# stream/events instead of the process-global singletons and (b) the L2B
# restricted to the owned leaf range. Sync-free and capacity-static within an
# occupancy epoch => capturable.
function p2_lifecycle_body!(state, c0::Int, c1::Int, side, ev_begin, ev_done)
    CUDA.record(ev_begin)                 # current (main) stream
    CUDA.wait(ev_begin, side)
    CUDA.stream!(side) do
        FastMultipole._launch_cuda_nearfield_kernel!(state)   # fill! + filtered pairs
    end
    CUDA.record(ev_done, side)
    FastMultipole._launch_cuda_b2m!(state)
    FastMultipole._launch_cuda_resident_m2m!(state)
    FastMultipole._launch_cuda_resident_m2l!(state)           # cached, filtered windows
    FastMultipole._launch_cuda_resident_l2l!(state)
    CUDA.wait(ev_done)
    p2_l2b_owned!(state, c0, c1)
    return state
end

# Script-owned per-GPU graph slot. IMPORTANT concurrency contract: stream
# capture runs in CUDA's GLOBAL capture mode, which makes most CUDA API use on
# OTHER threads capture-illegal for its duration (this wedged job 13061054:
# both workers captured concurrently, one died, its partner spun in the
# barrier). Recording therefore happens ONLY inside p2_record_graph!/
# p2_record_graphs! — serially, one device at a time, with the peer quiescent.
# The concurrent step path only replays a valid exec or runs the body
# uncaptured; it never captures.
mutable struct P2GraphSlot
    exec::Any
    epoch::Int
    filter_epoch::Int
    refilters::Int
    poisoned::Bool
end
P2GraphSlot() = P2GraphSlot(nothing, typemin(Int), typemin(Int), 0, false)

function p2_run_lifecycle!(slot::P2GraphSlot, state, c0, c1, side, ev_b, ev_d;
        use_graph::Bool=true)
    hctx = state.interaction_list
    if use_graph && !slot.poisoned && slot.exec !== nothing &&
            slot.epoch == hctx.epoch_id
        CUDA.launch(slot.exec::CUDA.CuGraphExec)
        return state
    end
    return p2_lifecycle_body!(state, c0, c1, side, ev_b, ev_d)
end

# ---- per-GPU bundle ----------------------------------------------------------
# (definitions below; p2_record_graph! needs the struct, see after it)

mutable struct P2Gpu
    dev::Int
    sys::Any
    cache::Any
    part::Any
    slot::P2GraphSlot
    side::Any
    ev_begin::Any
    ev_done::Any
    ev_seg0::Any
    ev_graph_done::Any
    ev_copy0::Any
    ev_copy1::Any
    ev_fin0::Any
    ev_fin1::Any
    switches::Any
    target_buffers::Any
    use_graph::Bool
end

# Construct system + cache on device `dev` for half `half`; run one stock
# (graph-off, overlap-off) step to build state/windows; filter to the owned
# half. Returns the bundle plus construction telemetry and the pre-filter
# window/direct signatures (for the exactness gate).
function p2_setup_gpu!(dev::Int, half::Int, bodies::Matrix{Float64}, ::Type{TF},
        P::Int, ell::Int, cache_kwargs, opts; max_n_bodies, bounds,
        use_graph::Bool=true) where TF
    CUDA.device!(dev)
    used0 = CUDA.total_memory() - CUDA.free_memory()
    t0 = time_ns()
    sys = FM028DeviceSystem{TF}(bodies)
    cache = RadixFMMCache(sys; expansion_order=P, ell, max_n_bodies,
        bounds, lamb_helmholtz=false, device=true, options=opts, cache_kwargs...)
    fmm!(sys, cache; scalar_potential=true, gradient=true)   # builds state + windows
    CUDA.synchronize()
    construction_ms = (time_ns() - t0) / 1e6
    persistent_bytes = (CUDA.total_memory() - CUDA.free_memory()) - used0
    state = cache.state
    size(state.output, 1) == 4 || error("P2 prototype expects the 4-row output (hessian off)")
    full_windows = p2_window_signature(state)
    full_direct = p2_direct_signature(state)
    part = p2_partition(cache, half)
    p2_filter_windows!(state, part)
    p2_filter_direct!(state, part)
    slot = P2GraphSlot()
    slot.filter_epoch = state.interaction_list.epoch_id
    side = CUDA.CuStream(; flags=CUDA.STREAM_NON_BLOCKING)
    switches = (FastMultipole.DerivativesSwitch(true, true, false, sys),)
    target_buffers = FastMultipole._radix_cache_target_buffers!(cache, switches)
    g = P2Gpu(dev, sys, cache, part, slot, side,
        CUDA.CuEvent(CUDA.EVENT_DISABLE_TIMING), CUDA.CuEvent(CUDA.EVENT_DISABLE_TIMING),
        CUDA.CuEvent(), CUDA.CuEvent(), CUDA.CuEvent(), CUDA.CuEvent(),
        CUDA.CuEvent(), CUDA.CuEvent(),
        switches, target_buffers, use_graph)
    return g, (; construction_ms, persistent_bytes, full_windows, full_direct)
end

# Re-filter + graph invalidation on occupancy-epoch change (real-recurrence
# rule: this cost lands inside the measured step whenever occupancy moves).
function p2_check_epoch!(me::P2Gpu)
    state = me.cache.state
    hctx = state.interaction_list
    if me.slot.filter_epoch != hctx.epoch_id
        p2_filter_windows!(state, me.part)
        p2_filter_direct!(state, me.part)
        me.slot.filter_epoch = hctx.epoch_id
        me.slot.exec = nothing
        me.slot.epoch = typemin(Int)
        me.slot.refilters += 1
        # NOTE: the step path never re-captures (see the P2GraphSlot contract);
        # after an epoch change, steps run uncaptured until the driver calls
        # p2_record_graphs! again. At the frozen workload this never triggers;
        # its cost would land inside the measured step either way.
    end
    return me
end

# Solo, serialized graph recording (the ONLY place capture runs). The caller
# guarantees no concurrent CUDA activity on any other thread. Runs the body
# once uncaptured (JIT/CUBLAS/staged-scalar warmth), then captures, then
# launches the instantiated graph once so the device state reflects a
# completed lifecycle.
function p2_record_graph!(me::P2Gpu)
    me.use_graph || return me
    CUDA.device!(me.dev)
    state = me.cache.state
    FastMultipole.update_cuda_radix_state!(me.cache, (me.sys,))
    p2_check_epoch!(me)
    p2_lifecycle_body!(state, me.part.c0, me.part.c1, me.side, me.ev_begin, me.ev_done)
    CUDA.synchronize()
    graph = try
        CUDA.capture(; throw_error=false) do
            p2_lifecycle_body!(state, me.part.c0, me.part.c1, me.side,
                me.ev_begin, me.ev_done)
        end
    catch err
        err isa CUDA.CuError || rethrow()
        @warn "P2 graph capture failed on device $(me.dev); running uncaptured" err
        me.slot.poisoned = true
        nothing
    end
    if graph !== nothing
        me.slot.exec = CUDA.instantiate(graph)
        me.slot.epoch = state.interaction_list.epoch_id
        CUDA.launch(me.slot.exec::CUDA.CuGraphExec)   # capture recorded without executing
        CUDA.synchronize()
    end
    return me
end

p2_record_graphs!(G::Vector{P2Gpu}) = (foreach(p2_record_graph!, G); G)

# The exchange's ordering contract: the D2D copy must be enqueued on the
# copying task's stream for the SOURCE device, so ev_copy1 (recorded on that
# stream) orders it for the peer. CUDACore only guarantees that on the direct
# peer-access path; its no-P2P fallback stages through host memory with the
# H2D enqueued on the DESTINATION context's stream, which ev_copy1 does not
# cover. Require direct peer access up front (H200 pairs: NVLink).
function p2_require_peer_access!(G::Vector{P2Gpu})
    f = nothing
    for mod in (CUDA, isdefined(CUDA, :CUDACore) ? CUDA.CUDACore : CUDA)
        if isdefined(mod, :maybe_enable_peer_access)
            f = getfield(mod, :maybe_enable_peer_access)
            break
        end
    end
    if f === nothing
        @warn "cannot find maybe_enable_peer_access; exchange stream ordering unverified"
        return G
    end
    for (a, b) in ((G[1].dev, G[2].dev), (G[2].dev, G[1].dev))
        CUDA.device!(a)
        f(CUDA.CuDevice(a), CUDA.CuDevice(b)) == 1 ||
            error("P2 exchange requires direct peer access $(a)->$(b) (no-P2P fallback is mis-ordered)")
    end
    return G
end

# ---- the complete 2-GPU verdict step ----------------------------------------

# Worker body for GPU g (1-based). seg rows, per GPU column:
#   1 refresh host-wall ms         5 GPU total wall ms
#   2 graph device ms (events)     6 pre-exchange barrier wait ms (imbalance)
#   3 exchange copy device ms      7 finalize+euler host wall ms incl. any
#   4 finalize+euler device ms       residual wait on the peer's copy + sync
function p2_worker_step!(G::Vector{P2Gpu}, g::Int, bar::P2Barrier;
        dt, clamp_lo, clamp_hi, do_euler::Bool=true, seg=nothing)
    try
        _p2_worker_step_inner!(G, g, bar; dt, clamp_lo, clamp_hi, do_euler, seg)
    catch
        bar.abort[] = 1     # release a partner parked at a barrier
        rethrow()
    end
end

function _p2_worker_step_inner!(G::Vector{P2Gpu}, g::Int, bar::P2Barrier;
        dt, clamp_lo, clamp_hi, do_euler::Bool=true, seg=nothing)
    me = G[g]
    peer = G[3 - g]
    CUDA.device!(me.dev)
    state = me.cache.state
    t0 = time_ns()
    FastMultipole.update_cuda_radix_state!(me.cache, (me.sys,))
    p2_check_epoch!(me)
    t1 = time_ns()
    CUDA.record(me.ev_seg0)
    p2_run_lifecycle!(me.slot, state, me.part.c0, me.part.c1, me.side,
        me.ev_begin, me.ev_done; use_graph=me.use_graph)
    CUDA.record(me.ev_graph_done)
    t2 = time_ns()
    p2_barrier_wait!(bar)              # peer's graph + event record are issued
    t3 = time_ns()
    CUDA.wait(peer.ev_graph_done)      # cross-device: peer's output is complete
    CUDA.record(me.ev_copy0)
    copyto!(view(peer.cache.state.output, :, me.part.b0:me.part.b1),
        view(state.output, :, me.part.b0:me.part.b1))
    CUDA.record(me.ev_copy1)
    p2_barrier_wait!(bar)              # peer's copy + record are issued
    CUDA.wait(peer.ev_copy1)           # incoming block visible before finalize
    t4 = time_ns()
    CUDA.record(me.ev_fin0)            # completes only after the peer copy wait
    FastMultipole.finalize_cuda_radix_output!(state, (me.sys,);
        derivatives_switches=me.switches,
        host_output_staging=me.cache.device_ctx.host_output,
        target_buffers=me.target_buffers,
        device_target_buffers=me.cache.device_ctx.device_target_buffers)
    do_euler && fm028_euler!(me.sys, dt, clamp_lo, clamp_hi)
    CUDA.record(me.ev_fin1)
    CUDA.synchronize()
    t5 = time_ns()
    if seg !== nothing
        seg[1, g] = (t1 - t0) / 1e6
        seg[2, g] = Float64(CUDA.elapsed(me.ev_seg0, me.ev_graph_done)) * 1e3
        seg[3, g] = Float64(CUDA.elapsed(me.ev_copy0, me.ev_copy1)) * 1e3
        seg[4, g] = Float64(CUDA.elapsed(me.ev_fin0, me.ev_fin1)) * 1e3
        seg[5, g] = (t5 - t0) / 1e6
        seg[6, g] = (t3 - t2) / 1e6
        seg[7, g] = (t5 - t4) / 1e6
    end
    return nothing
end

# One complete 2-GPU step (spawn + join measured — that IS the orchestration).
# Returns (wall_ms, seg).
function p2_step_pair!(G::Vector{P2Gpu}, bar::P2Barrier;
        dt=0.0, clamp_lo=0.0, clamp_hi=1.0, do_euler::Bool=true, seg=nothing)
    t = time_ns()
    @sync for g in 1:2
        Threads.@spawn p2_worker_step!(G, g, bar; dt, clamp_lo, clamp_hi,
            do_euler, seg)
    end
    return (time_ns() - t) / 1e6, seg
end

# ---- exactness gate ----------------------------------------------------------

# The two filtered route/pair sets must exactly partition the full sets: counts
# add up and xor checksums recombine, per level. `fw`/`fd` are the pre-filter
# signatures (identical caches => must agree between GPUs too).
function p2_partition_exact(G::Vector{P2Gpu}, fw1, fd1, fw2, fd2)
    ok = true
    fw1.counts == fw2.counts && fw1.xors == fw2.xors || (ok = false)
    fd1.count == fd2.count && fd1.xor == fd2.xor || (ok = false)
    s1 = p2_window_signature(G[1].cache.state)
    s2 = p2_window_signature(G[2].cache.state)
    for (i, (n, x)) in enumerate(zip(fw1.counts, fw1.xors))
        s1.counts[i] + s2.counts[i] == n || (ok = false)
        (s1.xors[i] ⊻ s2.xors[i]) == x || (ok = false)
    end
    d1 = p2_direct_signature(G[1].cache.state)
    d2 = p2_direct_signature(G[2].cache.state)
    d1.count + d2.count == fd1.count || (ok = false)
    (d1.xor ⊻ d2.xor) == fd1.xor || (ok = false)
    return ok, (; routes=(s1.counts, s2.counts), direct=(d1.count, d2.count))
end
