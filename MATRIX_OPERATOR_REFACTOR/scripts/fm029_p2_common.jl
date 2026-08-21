# Task 029 prototype P2: shared 2-GPU decomposition machinery.
#
# Included AFTER `using FastMultipole`, `using CUDA`, and
# fm028_device_system.jl, by prototype_029_p2_twogpu.jl and
# test/cuda_radix_twogpu_test.jl. Script/test side only — no production src/
# surface is modified; everything here drives the UNCHANGED cycle-1 resident
# pipeline through its public/internal launchers.
#
# Decomposition: WORK-LIST SLICING + ALLREDUCE (documented in the 029 task
# file, P2 section; replaces the earlier target-filtered design, which rested
# on unfounded assumptions about the hierarchical route-window semantics —
# jobs 13061265/13061495/13064752/13065445/13066053).
#   * MIRRORED SOURCES. Both GPUs hold the full body set and construct
#     full-domain RadixFMMCaches (same bounds, ell, policy). No assumption of
#     identical sorts or route streams is needed anywhere.
#   * SLICED WORK LISTS. The cached M2L route windows and the nearfield pair
#     list are pure work lists whose entries accumulate atomically into
#     locals/output — ANY positional split of them is exact. GPU g applies
#     the g-th half of every per-level window slice and of the pair list, and
#     runs L2B over the g-th contiguous leaf-cell half. B2M/M2M/L2L and the
#     refresh are replicated (measured and reported as such).
#   * TWO ALLREDUCE EXCHANGES per step, both bitwise-exact by IEEE add
#     commutativity (a+b == b+a), which makes the mirrored body states advance
#     in bitwise lockstep with NO position exchange:
#       1. after B2M/M2M/M2L-slice: each GPU pushes its partial `locals.phi`
#          into the peer's staging; both sides add — both now hold the SAME
#          two partial arrays summed (node numbering is key-sorted and thus
#          deterministic, so the locals frames agree across devices).
#       2. after L2L + nearfield-slice + L2B-half: same for `output`, but
#          PERM-AWARE: the device counting sort is not deterministic across
#          caches (within-cell body order comes from atomics — this scrambled
#          every sorted-frame diagnostic of the previous design at ~0.4-0.5
#          relrms), so each GPU pushes its output TOGETHER WITH its
#          `body_perm` slice, and the receiver scatter-adds through
#          inv(own perm) ∘ peer perm. Per USER body the sum is the same two
#          floats on both devices — still bitwise-symmetric.
#   * GRAPHS. The far-field is captured per GPU as TWO script-owned graphs
#     (A: fill+nearfield-slice on a side stream joined at the end, B2M, M2M,
#     M2L-slice; B: L2L + owned-half L2B), recorded solo and serialized
#     (GLOBAL capture mode outlaws concurrent CUDA API use on other threads —
#     job 13061054). Steps replay or run uncaptured; they never capture.
#     Production flags CUDA_GRAPH_LIFECYCLE / CUDA_OVERLAP_NEARFIELD must be
#     OFF while CUDA_CACHED_WINDOWS stays ON (the M2L slice consumes the
#     cached window stream).

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

# ---- partition (leaf cells only; no route semantics) -------------------------

# Contiguous leaf-cell half for L2B plus the corresponding sorted-body range
# (reporting only). Work-list slicing needs nothing else.
function p2_partition(cache, half::Int)
    state = cache.state
    ncell = state.counts.n_cells
    h = ncell ÷ 2
    c0 = half == 1 ? 1 : h + 1
    c1 = half == 1 ? h : ncell
    first_c0 = Int(Array(view(state.cell_ranges, 1:1, c0:c0))[1])
    first_c1 = Int(Array(view(state.cell_ranges, 1:1, c1:c1))[1])
    count_c1 = Int(Array(view(state.cell_ranges, 2:2, c1:c1))[1])
    return (; half, c0, c1, b0=first_c0, b1=first_c1 + count_c1 - 1)
end

_p2_slice(n::Int, half::Int) = half == 1 ? (1, n ÷ 2) : (n ÷ 2 + 1, n)

# ---- sliced work-list launches -----------------------------------------------

# The g-th half of every per-level cached M2L window slice. The fused-family
# kernels carry the operator class per route, so any positional split is exact;
# locals accumulate atomically.
function p2_m2l_sliced!(state::FastMultipole.DeviceResidentRadixState{TF,B,LH},
        half::Int) where {TF,B,LH}
    hctx = state.interaction_list
    hctx isa FastMultipole.DeviceHierarchicalM2LContext || throw(ArgumentError(
        "P2 M2L slicing requires the hierarchical device context"))
    (FastMultipole.CUDA_CACHED_WINDOWS[] && hctx.win_valid &&
        FastMultipole._cuda_windows_cacheable(hctx)) || throw(ArgumentError(
        "P2 M2L slicing requires a valid cacheable window stream"))
    ws = state.scratch
    plan = hctx.apply_plan
    fill!(state.locals.phi, zero(TF))
    LH && fill!(state.locals.chi, zero(TF))
    wc = hctx.win_class::CUDA.CuVector{Int32}
    wsrc = hctx.win_sources::CUDA.CuVector{Int}
    wtgt = hctx.win_targets::CUDA.CuVector{Int}
    for L in 2:hctx.ell
        n = hctx.win_level_counts[L + 1]
        s = hctx.win_level_starts[L + 1]
        lo, hi = _p2_slice(n, half)
        len = hi - lo + 1
        len > 0 && FastMultipole._cuda_hier_dense_apply_routes!(state, ws, plan,
            hctx, L, view(wc, (s + lo):(s + hi)), view(wsrc, (s + lo):(s + hi)),
            view(wtgt, (s + lo):(s + hi)), len)
    end
    state.counts.n_routes = hctx.total_routes   # telemetry parity with production
    return state
end

# fill! + the g-th half of the nearfield pair list (symmetric Newton pairs or
# plain functor pairs — whichever production would select). Output accumulates
# atomically, so the positional split is exact under the later allreduce.
function p2_nearfield_sliced!(state::FastMultipole.DeviceResidentRadixState{TF,B,LH},
        half::Int) where {TF,B,LH}
    fill!(state.output, zero(TF))
    size(state.output, 1) == 4 || throw(ArgumentError("P2 expects the 4-row output"))
    hctx = state.interaction_list
    dk = state.options.direct_kernel
    dk isa Union{FastMultipole.PartitionedVortex,FastMultipole.TwoPassVortex} &&
        throw(ArgumentError("split vortex nearfield is unsupported in the P2 slice"))
    hsv = Val(false)
    threads = 128
    symmetric = FastMultipole.CUDA_SYMMETRIC_NEARFIELD[] && !LH &&
        dk isa FastMultipole.SingularSource &&
        hctx isa FastMultipole.DeviceHierarchicalM2LContext &&
        !isempty(hctx.symmetric_targets)
    if symmetric
        np = hctx.n_symmetric_pairs
        lo, hi = _p2_slice(np, half)
        len = hi - lo + 1
        blocks = min(cld(len, threads ÷ 32), FastMultipole.DIRECT_CUDA_MAX_BLOCKS[])
        len > 0 && CUDA.@cuda threads=threads blocks=blocks FastMultipole._cuda_symmetric_pairs_output_kernel!(
            state.output, state.source_bodies, state.cell_ranges,
            view(hctx.symmetric_targets, lo:hi), view(hctx.symmetric_sources, lo:hi),
            len, hsv)
    else
        np = state.counts.n_direct
        lo, hi = _p2_slice(np, half)
        len = hi - lo + 1
        blocks = min(cld(len, threads ÷ 32), FastMultipole.DIRECT_CUDA_MAX_BLOCKS[])
        len > 0 && CUDA.@cuda threads=threads blocks=blocks FastMultipole._cuda_direct_pairs_functor_kernel!(
            dk, state.output, state.source_bodies, state.cell_ranges,
            view(state.direct_targets, lo:hi), view(state.direct_sources, lo:hi),
            len, hsv)
    end
    return state
end

# Warp-per-cell L2B over the owned contiguous leaf-cell range only. Same
# production kernel; the views re-base cell indexing while body indices
# (cell_ranges values), node indices (leaf_to_node values), and the output
# stay absolute.
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

# ---- perm-aware output allreduce kernels -------------------------------------

# inv[perm[j]] = j over the sorted prefix: user index -> my sorted column.
function _p2_inv_kernel!(inv, perm, n)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j > n && return nothing
    @inbounds inv[Int(perm[j])] = Int32(j)
    return nothing
end

# out[:, inv[idx[j]]] += blk[:, j]: fold the peer's partial output (in the
# peer's sorted frame, idx = peer's body_perm) into my sorted frame. idx is a
# permutation, so columns are written exactly once — no atomics needed.
function _p2_scatter_add_kernel!(out, inv, idx, blk, n)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j > n && return nothing
    @inbounds begin
        s = Int(inv[Int(idx[j])])
        for r in 1:4
            out[r, s] += blk[r, j]
        end
    end
    return nothing
end

# ---- the two captured lifecycle bodies ---------------------------------------

# Phase A: side-stream fill+nearfield-slice forked and JOINED inside the body
# (so both A and B stay single-graph capturable with no cross-graph events),
# plus the replicated upward pass and the M2L slice into partial locals.
function p2_body_a!(state, half::Int, side, ev_begin, ev_done)
    CUDA.record(ev_begin)                 # current (main) stream
    CUDA.wait(ev_begin, side)
    CUDA.stream!(side) do
        p2_nearfield_sliced!(state, half)
    end
    CUDA.record(ev_done, side)
    FastMultipole._launch_cuda_b2m!(state)
    FastMultipole._launch_cuda_resident_m2m!(state)
    p2_m2l_sliced!(state, half)
    CUDA.wait(ev_done)                    # join: A's completion covers the nearfield
    return state
end

# Phase B: L2L over the summed locals + owned-half L2B. No event operations —
# runs entirely on the main stream after the locals allreduce.
function p2_body_b!(state, c0::Int, c1::Int)
    FastMultipole._launch_cuda_resident_l2l!(state)
    p2_l2b_owned!(state, c0, c1)
    return state
end

# ---- per-GPU graph slots ------------------------------------------------------

# Stream capture runs in CUDA's GLOBAL capture mode, which makes most CUDA API
# use on OTHER threads capture-illegal for its duration (job 13061054 wedge).
# Recording happens ONLY inside p2_record_graph!/p2_record_graphs! — serially,
# one device at a time, with the peer quiescent. The concurrent step path only
# replays valid execs or runs the bodies uncaptured; it never captures.
mutable struct P2GraphSlot
    exec_a::Any
    exec_b::Any
    epoch::Int
    refilters::Int      # occupancy-epoch graph invalidations observed in-step
    poisoned::Bool
end
P2GraphSlot() = P2GraphSlot(nothing, nothing, typemin(Int), 0, false)

# ---- per-GPU bundle ----------------------------------------------------------

mutable struct P2Gpu
    dev::Int
    sys::Any
    cache::Any
    part::Any
    slot::P2GraphSlot
    side::Any
    ev_begin::Any       # body-A internal side-stream fork/join pair
    ev_done::Any
    ev_seg0::Any        # step start (timing)
    ev_a::Any           # body A complete (timing)
    ev_cl::Any          # locals push complete (peer waits this)
    ev_lb::Any          # body B complete (timing)
    ev_co::Any          # output push complete (peer waits this)
    ev_f0::Any          # finalize+euler bracket (timing)
    ev_f1::Any
    stage_lphi::Any     # peer-writable staging: partial locals.phi
    stage_out::Any      # peer-writable staging: partial output (peer's frame)
    stage_idx::Any      # peer-writable staging: peer's body_perm slice
    inv::Any            # my user-index -> sorted-column map (rebuilt per step)
    switches::Any
    target_buffers::Any
    use_graph::Bool
end

# Construct system + cache on device `dev` for half `half`; one stock
# (graph-off, overlap-off) fmm! builds state + the cached window stream.
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
    part = p2_partition(cache, half)
    slot = P2GraphSlot()
    side = CUDA.CuStream(; flags=CUDA.STREAM_NON_BLOCKING)
    switches = (FastMultipole.DerivativesSwitch(true, true, false, sys),)
    target_buffers = FastMultipole._radix_cache_target_buffers!(cache, switches)
    stage_lphi = CUDA.zeros(TF, size(state.locals.phi))
    stage_out = CUDA.zeros(TF, size(state.output))
    stage_idx = CUDA.zeros(eltype(state.body_perm), size(state.output, 2))
    inv = CUDA.zeros(Int32, size(state.output, 2))
    ev() = CUDA.CuEvent()
    evd() = CUDA.CuEvent(CUDA.EVENT_DISABLE_TIMING)
    g = P2Gpu(dev, sys, cache, part, slot, side,
        evd(), evd(),                       # ev_begin, ev_done
        ev(), ev(), ev(), ev(), ev(), ev(), ev(),   # seg0, a, cl, lb, co, f0, f1
        stage_lphi, stage_out, stage_idx, inv,
        switches, target_buffers, use_graph)
    hctx = state.interaction_list
    return g, (; construction_ms, persistent_bytes,
        routes=hctx.total_routes, n_direct=state.counts.n_direct)
end

# Slice coverage report: the per-level window slices and the pair-list slices
# must tile the production work lists exactly (a pure host check).
function p2_slice_coverage(G::Vector{P2Gpu})
    ok = true
    detail = Int[]
    for st in (G[1].cache.state,)
        hctx = st.interaction_list
        for L in 2:hctx.ell
            n = hctx.win_level_counts[L + 1]
            lo1, hi1 = _p2_slice(n, 1)
            lo2, hi2 = _p2_slice(n, 2)
            (lo1 == 1 && hi1 + 1 == lo2 && hi2 == n) || (ok = false)
            push!(detail, n)
        end
        nd = hctx isa FastMultipole.DeviceHierarchicalM2LContext &&
            FastMultipole.CUDA_SYMMETRIC_NEARFIELD[] &&
            !isempty(hctx.symmetric_targets) ?
            hctx.n_symmetric_pairs : st.counts.n_direct
        lo1, hi1 = _p2_slice(nd, 1)
        lo2, hi2 = _p2_slice(nd, 2)
        (lo1 == 1 && hi1 + 1 == lo2 && hi2 == nd) || (ok = false)
        push!(detail, nd)
    end
    return ok, detail
end

# Graph invalidation on occupancy-epoch change (real-recurrence rule: steps
# after a change run uncaptured until the driver re-records; at the frozen
# workload this never triggers).
function p2_check_epoch!(me::P2Gpu)
    hctx = me.cache.state.interaction_list
    if me.slot.epoch != hctx.epoch_id &&
            (me.slot.exec_a !== nothing || me.slot.exec_b !== nothing)
        me.slot.exec_a = nothing
        me.slot.exec_b = nothing
        me.slot.refilters += 1
    end
    return me
end

# Solo, serialized graph recording (the ONLY place capture runs). Runs both
# bodies once uncaptured (JIT/CUBLAS warmth), then captures each, then launches
# both once so the device state reflects a completed far-field.
function p2_record_graph!(me::P2Gpu)
    me.use_graph || return me
    CUDA.device!(me.dev)
    st = me.cache.state
    FastMultipole.update_cuda_radix_state!(me.cache, (me.sys,))
    p2_check_epoch!(me)
    half = me.part.half
    p2_body_a!(st, half, me.side, me.ev_begin, me.ev_done)
    p2_body_b!(st, me.part.c0, me.part.c1)
    CUDA.synchronize()
    cap(f) = try
        CUDA.capture(f; throw_error=false)
    catch err
        err isa CUDA.CuError || rethrow()
        @warn "P2 graph capture failed on device $(me.dev); running uncaptured" err
        me.slot.poisoned = true
        nothing
    end
    ga = cap(() -> p2_body_a!(st, half, me.side, me.ev_begin, me.ev_done))
    gb = me.slot.poisoned ? nothing :
        cap(() -> p2_body_b!(st, me.part.c0, me.part.c1))
    if ga !== nothing && gb !== nothing
        me.slot.exec_a = CUDA.instantiate(ga)
        me.slot.exec_b = CUDA.instantiate(gb)
        me.slot.epoch = st.interaction_list.epoch_id
        CUDA.launch(me.slot.exec_a::CUDA.CuGraphExec)   # capture records without executing
        CUDA.launch(me.slot.exec_b::CUDA.CuGraphExec)
        CUDA.synchronize()
    end
    return me
end

p2_record_graphs!(G::Vector{P2Gpu}) = (foreach(p2_record_graph!, G); G)

# The exchange's ordering contract: the D2D pushes must be enqueued on the
# pushing task's stream for the SOURCE device, so ev_cl/ev_co (recorded on that
# stream) order them for the peer. CUDACore only guarantees that on the direct
# peer-access path; its no-P2P fallback stages through host memory with the
# H2D enqueued on the DESTINATION context's stream, which the events do not
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

# cuCtxEnablePeerAccess maps only cuMemAlloc memory; stream-ordered POOL
# memory needs an explicit cuMemPoolSetAccess, which CUDACore leaves disabled
# (old NVIDIA bug workaround, memory.jl "XXX: disabled"). Without it the
# driver silently stages cross-device copies of pool-backed CuArrays at
# ~32 GB/s even on NV18 pairs (job 13066057: exchange 0.905 ms of a 1.003 ms
# comm+orch total). Granting each device's pool to the peer restores direct
# NVLink-rate P2P.
function p2_enable_pool_peer_access!(d0::Int, d1::Int)
    core = isdefined(CUDA, :CUDACore) ? CUDA.CUDACore : CUDA
    if !(isdefined(core, :pool_create) && isdefined(core, :access!))
        @warn "pool access API unavailable; exchange may run at staging bandwidth"
        return false
    end
    flags = core.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    try
        for (owner, accessor) in ((d0, d1), (d1, d0))
            CUDA.device!(owner)
            pool = core.pool_create(CUDA.CuDevice(owner))
            core.access!(pool, [CUDA.CuDevice(accessor)], flags)
        end
        return true
    catch err
        @warn "pool peer-access grant failed; exchange may run at staging bandwidth" err
        return false
    end
end

# ---- the complete 2-GPU verdict step ----------------------------------------

# Worker body for GPU g (1-based). seg rows, per GPU column:
#   1 refresh host-wall ms
#   2 far-field device ms (A + add/L2L/L2B, includes any peer-wait stall)
#   3 exchange push device ms (locals + output copies)
#   4 finalize+euler device ms
#   5 GPU total wall ms
#   6 pre-exchange barrier wait ms (imbalance)
#   7 finalize+euler host wall ms incl. residual peer waits + sync
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
    st = me.cache.state
    t0 = time_ns()
    FastMultipole.update_cuda_radix_state!(me.cache, (me.sys,))
    p2_check_epoch!(me)
    nb = st.counts.n_bodies
    CUDA.@cuda threads=256 blocks=cld(nb, 256) _p2_inv_kernel!(
        me.inv, st.body_perm, nb)
    t1 = time_ns()
    CUDA.record(me.ev_seg0)
    replay = me.use_graph && !me.slot.poisoned &&
        me.slot.epoch == st.interaction_list.epoch_id
    if replay && me.slot.exec_a !== nothing
        CUDA.launch(me.slot.exec_a::CUDA.CuGraphExec)
    else
        p2_body_a!(st, me.part.half, me.side, me.ev_begin, me.ev_done)
    end
    CUDA.record(me.ev_a)
    # allreduce 1: push my partial locals into the peer's staging
    copyto!(peer.stage_lphi, st.locals.phi)
    CUDA.record(me.ev_cl)
    t2 = time_ns()
    p2_barrier_wait!(bar)              # peer's push + record are issued
    t3 = time_ns()
    CUDA.wait(peer.ev_cl)              # incoming partial visible
    st.locals.phi .+= me.stage_lphi    # bitwise-symmetric: a+b == b+a
    if replay && me.slot.exec_b !== nothing
        CUDA.launch(me.slot.exec_b::CUDA.CuGraphExec)
    else
        p2_body_b!(st, me.part.c0, me.part.c1)
    end
    CUDA.record(me.ev_lb)
    # allreduce 2: push my partial output (nearfield slice + owned-half L2B)
    # together with my body_perm slice — the peer folds it through its own
    # inverse map (the sorts are not deterministic across caches)
    copyto!(view(peer.stage_out, :, 1:nb), view(st.output, :, 1:nb))
    copyto!(view(peer.stage_idx, 1:nb), view(st.body_perm, 1:nb))
    CUDA.record(me.ev_co)
    p2_barrier_wait!(bar)              # peer's push + record are issued
    CUDA.wait(peer.ev_co)
    t4 = time_ns()
    CUDA.record(me.ev_f0)
    CUDA.@cuda threads=256 blocks=cld(nb, 256) _p2_scatter_add_kernel!(
        st.output, me.inv, me.stage_idx, me.stage_out, nb)
    FastMultipole.finalize_cuda_radix_output!(st, (me.sys,);
        derivatives_switches=me.switches,
        host_output_staging=me.cache.device_ctx.host_output,
        target_buffers=me.target_buffers,
        device_target_buffers=me.cache.device_ctx.device_target_buffers)
    do_euler && fm028_euler!(me.sys, dt, clamp_lo, clamp_hi)
    CUDA.record(me.ev_f1)
    CUDA.synchronize()
    t5 = time_ns()
    if seg !== nothing
        seg[1, g] = (t1 - t0) / 1e6
        seg[2, g] = (Float64(CUDA.elapsed(me.ev_seg0, me.ev_a)) +
                     Float64(CUDA.elapsed(me.ev_cl, me.ev_lb))) * 1e3
        seg[3, g] = (Float64(CUDA.elapsed(me.ev_a, me.ev_cl)) +
                     Float64(CUDA.elapsed(me.ev_lb, me.ev_co))) * 1e3
        seg[4, g] = Float64(CUDA.elapsed(me.ev_f0, me.ev_f1)) * 1e3
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
