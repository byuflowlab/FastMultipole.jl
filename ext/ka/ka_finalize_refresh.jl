#------- output finalization -------#
#
# Backend-agnostic output finalize: de-permute the lifecycle's sorted-order
# `state.output` back into each target system's own body order and hand it to
# `buffer_to_target!`.
#
# Only one of the two branches contains anything device-specific. The
# host-resident branch is generic -- a prefix `copyto!` into the host staging,
# then `_copy_radix_output_to_host_target_buffer!` (src/resident/resident_finalize.jl)
# -- and downloads `host_output` once per call, shared across systems, bumping
# the `influence_downloads` counter. The device-resident branch runs the
# scatter kernel below.
#
# Row layout is the switch's, not the output's: `scalar_potential_index`,
# `gradient_range` and `hessian_range` decide where each of the output's 1 / 2:4
# / 5:13 rows lands, and a switch asking for hessian rows from a 4-row output
# throws rather than reading past the end.

@kernel function ka_scatter_output_to_target_buffer_kernel!(target_buffer, @Const(output),
        @Const(perm), @Const(body_system), @Const(body_index), isys, scalar_row,
        gradient_start, gradient_stop, hessian_start, hessian_stop, n_bodies)
    sorted_i = @index(Global)
    @inbounds if sorted_i <= n_bodies
        global_i = perm[sorted_i]
        if body_system[global_i] == isys
            ibody = body_index[global_i]
            if scalar_row > 0
                target_buffer[scalar_row, ibody] = output[1, sorted_i]
            end
            if gradient_start <= gradient_stop
                target_buffer[gradient_start, ibody] = output[2, sorted_i]
                target_buffer[gradient_start + 1, ibody] = output[3, sorted_i]
                target_buffer[gradient_start + 2, ibody] = output[4, sorted_i]
            end
            if hessian_start <= hessian_stop
                for k in 0:8
                    target_buffer[hessian_start + k, ibody] = output[5 + k, sorted_i]
                end
            end
        end
    end
end

"""
    ka_scatter_output_to_target_buffer!(target_buffer, output, body_perm,
        body_system_ids, body_indices, isys, derivatives_switch, n_bodies;
        workgroup=KA_AUTO_WORKGROUP)

Zero `target_buffer` and scatter the sorted-order `output` columns belonging to system `isys` into it at
the rows the derivatives switch selects.
"""
function ka_scatter_output_to_target_buffer!(target_buffer, output, body_perm,
        body_system_ids, body_indices, isys::Integer, derivatives_switch,
        n_bodies::Integer=size(output, 2); workgroup=KA_AUTO_WORKGROUP)
    fill!(target_buffer, zero(eltype(target_buffer)))
    hrange = FastMultipole.hessian_range(derivatives_switch)
    isempty(hrange) || size(output, 1) >= 13 ||
        throw(ArgumentError("hessian output requested but the radix output " *
            "carries potential + gradient only; construct RadixFMMCache(...; hessian=true)"))
    grange = FastMultipole.gradient_range(derivatives_switch)
    n_bodies == 0 && return target_buffer
    backend = KA.get_backend(output)
    kernel = _cached_kernel(ka_scatter_output_to_target_buffer_kernel!, backend, workgroup)
    kernel(target_buffer, output, body_perm, body_system_ids, body_indices, isys,
        FastMultipole.scalar_potential_index(derivatives_switch),
        isempty(grange) ? 1 : first(grange), isempty(grange) ? 0 : last(grange),
        isempty(hrange) ? 1 : first(hrange), isempty(hrange) ? 0 : last(hrange),
        n_bodies; ndrange=n_bodies)
    return target_buffer
end

# Per-system cached device scatter buffer: allocated undef and reused, since
# the scatter zero-fills it anyway. Capacity contract (found as a long-run leak): a
# shedding run changes `nb` every step, and an exact-size cache then
# reallocates every step -- the replaced device buffer survives a full step
# before dying, gets promoted, and no major GC ever runs because device bytes
# are invisible to the host GC heuristics, so ~rows*nb*8 bytes of dead pool
# blocks accumulate per step. The cache instead holds a grow-only capacity
# buffer (geometric headroom) and serves the live `nb` as a contiguous
# column-prefix view.
function _ka_cached_target_buffer(cache, backend, isys::Integer, ::Type{TF},
        rows::Integer, nb::Integer) where TF
    cache === nothing && return KA.allocate(backend, TF, rows, nb)
    buf = get(cache, isys, nothing)
    if !(buf isa AbstractMatrix{TF}) || size(buf, 1) != rows || size(buf, 2) < nb
        cap = buf isa AbstractMatrix{TF} && size(buf, 1) == rows ?
            max(nb, size(buf, 2) + cld(size(buf, 2), 4)) : nb
        buf = KA.allocate(backend, TF, rows, cap)
        cache[isys] = buf
    end
    buf = buf::AbstractMatrix{TF}
    return size(buf, 2) == nb ? buf : view(buf, :, 1:nb)
end

"""
    ka_finalize_radix_output!(state, target_systems; derivatives_switches,
        host_output_staging, target_buffers, device_target_buffers)

Backend-agnostic output finalize. Scatters `state.output` back
into the target systems, downloading it once per call into
`host_output_staging` (the valid column prefix only) when any target is host
resident, and going through `ka_scatter_output_to_target_buffer!` for
device-resident ones.

Metadata rows (`metadata_range(switch)`): a host-resident target's buffer has
them refilled from the system with `metadata_to_buffer!` every call, as on the
host radix path. A device-resident target's buffer is device memory, which the
per-body host hook cannot write, so there they are filled by the target's
`metadata_to_device_buffer!` overload (`fmm!` rejects a device-resident target
with metadata rows and no overload before any work).
"""
function ka_finalize_radix_output!(state, target_systems;
        derivatives_switches=nothing, host_output_staging=nothing,
        target_buffers=nothing, device_target_buffers=nothing)
    TF = eltype(state.output)
    systems = FastMultipole.to_tuple(target_systems)
    switches = derivatives_switches === nothing ?
        FastMultipole.to_tuple(FastMultipole.DerivativesSwitch(true, true, false, systems)) :
        FastMultipole.to_tuple(derivatives_switches)
    length(systems) == length(switches) ||
        throw(ArgumentError("target systems and derivatives switches must have the same length"))
    backend = KA.get_backend(state.output)

    host_output = nothing
    for (isys, target_system, switch) in zip(eachindex(systems), systems, switches)
        if FastMultipole.residency(target_system) isa FastMultipole.DeviceResident
            target_buffer = _ka_cached_target_buffer(device_target_buffers, backend,
                isys, TF, FastMultipole.target_buffer_rows(switch),
                FastMultipole.get_n_bodies(target_system))
            ka_scatter_output_to_target_buffer!(target_buffer, state.output,
                state.body_perm, state.body_system_ids, state.body_indices, isys,
                switch, state.counts.n_bodies)
            isempty(FastMultipole.metadata_range(switch)) ||
                FastMultipole.metadata_to_device_buffer!(target_buffer, switch, target_system)
            FastMultipole.buffer_to_target!(target_system, target_buffer, switch,
                1:FastMultipole.get_n_bodies(target_system))
        else
            if host_output === nothing
                if host_output_staging === nothing
                    host_output = Array(state.output)
                else
                    # recurring path: download only the valid column prefix into
                    # the preallocated staging
                    nb = state.counts.n_bodies
                    copyto!(host_output_staging, 1, state.output, 1,
                        size(state.output, 1) * nb)
                    host_output = host_output_staging
                end
                state.counters.influence_downloads += 1
            end
            target_buffer = target_buffers === nothing ?
                FastMultipole.allocate_target_buffer(TF, target_system, switch) :
                target_buffers[isys]
            # metadata rows, refilled as the host radix finalize fills them
            if !isempty(FastMultipole.metadata_range(switch))
                for i_body in 1:FastMultipole.get_n_bodies(target_system)
                    FastMultipole.metadata_to_buffer!(target_buffer, switch, i_body,
                        target_system, i_body)
                end
            end
            FastMultipole._copy_radix_output_to_host_target_buffer!(
                target_buffer, host_output, state.host_body_perm,
                state.host_body_system_ids, state.host_body_indices, isys, switch,
                state.counts.n_bodies,
            )
            FastMultipole.buffer_to_target!(target_system, target_buffer, switch,
                1:FastMultipole.get_n_bodies(target_system))
        end
    end
    return target_systems
end

#------- within-cell sub-Morton nearfield subsort -------#
#
# Compose a within-cell sub-Morton ordering into `grid.perm` after the sort and
# before body packing, so consecutive sorted bodies -- adjacent lanes in the
# nearfield kernel -- span a compact spatial sub-block of their cell.
#
# It is locality only: no cell key, cell range or node changes, and cells larger
# than the shared-memory sort capacity keep their unspecified order. But it does
# change the ORDER same-cell contributions are summed in, so it is the one
# post-tree stage whose absence moves results at roundoff level.
#
# Launch shape: one workgroup per cell, launched with exactly `n_cells` groups.
# The kernel keeps a grid-stride outer loop (`cell += n_groups`), but with
# `n_groups == n_cells` every group runs it once. A capped group count would
# make groups loop over a second cell with `@synchronize` inside that loop, and
# that deadlocks on the KA GPU backends. The sort itself is odd-even
# transposition in workgroup-local memory, capacity 1024.

const KA_SUBSORT_CAPACITY = 1024

@kernel function ka_subsort_keys_kernel!(subsort_keys, @Const(positions), @Const(perm),
        x_min, h0, ell, ell_axes, sub, n)
    p = @index(Global)
    @inbounds if p <= n
        b = perm[p]
        Gs = 1 << (ell + sub)
        m = Int32((1 << sub) - 1)
        T = eltype(positions)
        delta = (2 * h0) / T(Gs)
        # each axis clamps to its own 2^(ell_axes[a] + sub) sub-cells, matching
        # the per-axis clamp of the cell key on a rectangular box
        cx = min(max(unsafe_trunc(Int32, (positions[1, b] - x_min[1]) / delta),
            Int32(0)), (Int32(1) << ((ell_axes[1] + sub) % Int32)) - Int32(1)) & m
        cy = min(max(unsafe_trunc(Int32, (positions[2, b] - x_min[2]) / delta),
            Int32(0)), (Int32(1) << ((ell_axes[2] + sub) % Int32)) - Int32(1)) & m
        cz = min(max(unsafe_trunc(Int32, (positions[3, b] - x_min[3]) / delta),
            Int32(0)), (Int32(1) << ((ell_axes[3] + sub) % Int32)) - Int32(1)) & m
        key = UInt32(0)
        bit = 0
        while bit < sub
            key |= (UInt32((cx >> bit) & Int32(1)) << (3 * bit))
            key |= (UInt32((cy >> bit) & Int32(1)) << (3 * bit + 1))
            key |= (UInt32((cz >> bit) & Int32(1)) << (3 * bit + 2))
            bit += 1
        end
        subsort_keys[p] = key
    end
end

# Workgroup-per-cell odd-even transposition sort of the perm segment by sub-key
# in local memory. The `1 < cnt <= capacity` condition is uniform across the
# group, so the barriers are safe.
@kernel function ka_subsort_cell_sort_kernel!(perm, subsort_keys, @Const(cell_ranges),
        n_cells, n_groups, ::Type{TP}, ::Val{CAP}, ::Val{WG}) where {TP,CAP,WG}
    keys_sh = @localmem UInt32 CAP
    perm_sh = @localmem TP CAP
    cell = @index(Group)
    t = @index(Local)
    @inbounds while cell <= n_cells
        first = cell_ranges[1, cell]
        cnt = cell_ranges[2, cell]
        if 1 < cnt <= CAP
            idx = t
            while idx <= cnt
                keys_sh[idx] = subsort_keys[first + idx - 1]
                perm_sh[idx] = perm[first + idx - 1]
                idx += WG
            end
            @synchronize
            phase = 0
            while phase < cnt
                base = 1 + (phase & 1)
                idx = base + 2 * (t - 1)
                while idx <= cnt - 1
                    ka = keys_sh[idx]
                    kb = keys_sh[idx + 1]
                    if kb < ka
                        keys_sh[idx] = kb
                        keys_sh[idx + 1] = ka
                        pa = perm_sh[idx]
                        perm_sh[idx] = perm_sh[idx + 1]
                        perm_sh[idx + 1] = pa
                    end
                    idx += 2 * WG
                end
                @synchronize
                phase += 1
            end
            idx = t
            while idx <= cnt
                subsort_keys[first + idx - 1] = keys_sh[idx]
                perm[first + idx - 1] = perm_sh[idx]
                idx += WG
            end
            @synchronize
        end
        cell += n_groups
    end
end

"""
    ka_nearfield_subsort!(ctx, cache, n, n_cells; workgroup=256)

Compose a within-cell sub-Morton ordering into `ctx.grid.perm` and refresh
`invperm`. A no-op when the grid is already at the Morton depth cap
(`sub == 0`), when the grid is empty, or on the KA `CPU` backend.
"""
# Whether `ka_nearfield_subsort!` reorders (and re-inverts) the body perm.
# The cell sort's barriers sit inside a loop whose trip count is the cell's
# population, which the KA CPU backend cannot lower (it has no group index in
# uniform scope). The subsort only reorders bodies within a cell, so the CPU
# backend skips it.
_ka_subsort_runs(cache, backend, n::Int) =
    cache.options.direct_kernel isa FastMultipole.PartitionedVortex &&
    FastMultipole.RADIX_GRID_MAX_ELL - cache.ell > 0 && n > 0 && !(backend isa KA.CPU)

function ka_nearfield_subsort!(ctx, cache::FastMultipole.RadixFMMCache, n::Int,
        n_cells::Int; workgroup::Int=256)
    sub = min(3, FastMultipole.RADIX_GRID_MAX_ELL - cache.ell)
    (sub > 0 && n > 0 && n_cells > 0) || return nothing
    grid = ctx.grid
    backend = KA.get_backend(grid.perm)
    backend isa KA.CPU && return nothing
    kk = _cached_kernel(ka_subsort_keys_kernel!, backend, 128)
    kk(ctx.subsort_keys, ctx.positions, grid.perm, cache.x_min, cache.h0,
       cache.ell, cache.ell_axes, sub, n; ndrange=n)
    _utick!(:subsort_keys, backend)
    # ONE GROUP PER CELL, no grid-stride. A capped group count (e.g. 8192)
    # makes groups loop over a second cell, and `@synchronize` inside that loop
    # deadlocks on the KA GPU backends once n_cells exceeds the cap (seen on
    # both CUDA and Metal); n_groups = n_cells runs clean. KA has no
    # grid-dimension limit that needs the cap.
    n_groups = n_cells
    sk = _cached_kernel(ka_subsort_cell_sort_kernel!, backend, workgroup)
    sk(grid.perm, ctx.subsort_keys, grid.cell_ranges, n_cells, n_groups,
       eltype(grid.perm), Val(KA_SUBSORT_CAPACITY), Val(workgroup);
       ndrange=n_groups * workgroup)
    _utick!(:subsort_cells, backend)
    ka_fill_invperm!(grid.invperm, view(grid.perm, 1:n))
    _utick!(:subsort_invperm, backend)
    return nothing
end



#------- hierarchical refresh: direct pairs, window cache -------#
#
# The hierarchical branch of `ka_update_radix_state!`. This is the
# branch FLOWVPM actually takes: its cache is built with `window_classes`, so
# `RadixFMMCache` selects a `HierarchicalRigidStencil` and `hierarchical_ctx` is
# non-`nothing`.
#
# `ka_hier_refresh_occupancy!` (ka_hierarchical_m2l.jl) covers the per-level
# occupancy lookup; what is added here is the direct-pair generator and the
# generator that concatenates every level's windows into the epoch cache.
#
# Direct pairs differ from the flat path's in what they index: the flat kernels
# look up `cell_at` by decoded leaf Morton key, these look up the hierarchical
# per-level `node_at` by the leaf node's stored `node_coords` at `level_base_L`.
# Both chunk over the flag buffer with a running output base, for the same
# reason -- the flag/prefix buffers stay bounded by the scratch capacity rather
# than by `kn * n_cells`.

@kernel function ka_hier_direct_flags_kernel!(flags, @Const(node_at), @Const(node_coords),
        @Const(near_offsets), fbase, len, kn, leaf_base, level_base_L, ell)
    idx = @index(Global)
    @inbounds if idx <= len
        g = fbase + idx
        c = (g - 1) ÷ kn + 1
        k = (g - 1) % kn + 1
        G = 1 << ell
        target_node = leaf_base + c
        sx = node_coords[1, target_node] - near_offsets[1, k]
        sy = node_coords[2, target_node] - near_offsets[2, k]
        sz = node_coords[3, target_node] - near_offsets[3, k]
        src = Int32(0)
        if 0 <= sx < G && 0 <= sy < G && 0 <= sz < G
            src = node_at[level_base_L + (sx + G * (sy + G * sz)) + 1]
        end
        flags[idx] = src == Int32(0) ? Int32(0) : Int32(1)
    end
end

@kernel function ka_hier_direct_compact_kernel!(direct_targets, direct_sources,
        @Const(flags), @Const(prefix), @Const(node_at), @Const(node_coords),
        @Const(near_offsets), fbase, len, kn, leaf_base, level_base_L, ell, base)
    idx = @index(Global)
    @inbounds if idx <= len && flags[idx] == Int32(1)
        g = fbase + idx
        c = (g - 1) ÷ kn + 1
        k = (g - 1) % kn + 1
        G = 1 << ell
        target_node = leaf_base + c
        sx = node_coords[1, target_node] - near_offsets[1, k]
        sy = node_coords[2, target_node] - near_offsets[2, k]
        sz = node_coords[3, target_node] - near_offsets[3, k]
        src = Int(node_at[level_base_L + (sx + G * (sy + G * sz)) + 1])
        p = base + Int(prefix[idx])
        direct_targets[p] = c
        direct_sources[p] = src - leaf_base
    end
end

"""
    ka_hier_generate_direct_pairs!(ctx, hctx, grid, n_cells, leaf_base, ell;
                                   workgroup=KA_AUTO_WORKGROUP)

Flag/scan/compact the near-offset
neighbours of every occupied leaf cell into `ctx.direct_targets` /
`ctx.direct_sources`, chunked so the flag buffer bounds the working set.
Returns the pair count.
"""
function ka_hier_generate_direct_pairs!(ctx, hctx::FastMultipole.DeviceHierarchicalM2LContext,
        grid, n_cells::Int, leaf_base::Int, ell::Int; workgroup=KA_AUTO_WORKGROUP)
    kn = size(hctx.d_near_offsets, 2)
    (n_cells > 0 && kn > 0) || return 0
    backend = KA.get_backend(ctx.direct_flags)
    level_base_L = hctx.level_base[ell + 1]
    total = kn * n_cells
    capacity = length(ctx.direct_flags)
    capacity > 0 ||
        throw(AssertionError("device direct flag buffer has zero capacity"))
    flagk = _cached_kernel(ka_hier_direct_flags_kernel!, backend, workgroup)
    compactk = _cached_kernel(ka_hier_direct_compact_kernel!, backend, workgroup)
    n_direct = 0
    f0 = 0
    while f0 < total
        len = min(capacity, total - f0)
        flagk(ctx.direct_flags, hctx.node_at, grid.node_coords, hctx.d_near_offsets,
            f0, len, kn, leaf_base, level_base_L, ell; ndrange=len)
        accumulate!(+, view(ctx.direct_prefix, 1:len), view(ctx.direct_flags, 1:len))
        copyto!(ctx.host_scalar32, 1, ctx.direct_prefix, len, 1)
        chunk_total = Int(ctx.host_scalar32[1])
        if chunk_total > 0
            n_direct + chunk_total <= length(ctx.direct_targets) ||
                throw(AssertionError("device hierarchical direct pair buffer exceeded its capacity"))
            compactk(ctx.direct_targets, ctx.direct_sources, ctx.direct_flags,
                ctx.direct_prefix, hctx.node_at, grid.node_coords,
                hctx.d_near_offsets, f0, len, kn, leaf_base, level_base_L, ell,
                n_direct; ndrange=len)
        end
        n_direct += chunk_total
        f0 += len
    end
    return n_direct
end

# Grow the cached-window arrays to `needed`. The contents are not preserved:
# the caller regenerates every window into the fresh arrays. Growth happens only
# inside epoch regeneration, so this allocation recurs exactly with occupancy
# change.
function _ka_hier_win_ensure!(hctx, backend, needed::Int)
    old_class = hctx.win_class
    cap = old_class === nothing ? 0 : length(old_class)
    needed <= cap && return nothing
    newcap = max(needed, cap + cld(cap, 2), 1024)
    new_class = KA.allocate(backend, Int32, newcap)
    new_sources = KA.allocate(backend, Int, newcap)
    new_targets = KA.allocate(backend, Int, newcap)
    hctx.win_class = new_class
    hctx.win_sources = new_sources
    hctx.win_targets = new_targets
    return nothing
end

# Grow the window-scan flag/prefix scratch to `needed`; contents not preserved
# (every slot is rewritten by the flag kernels and the scan).
function _ka_hier_win_scan_ensure!(hctx, backend, needed::Int)
    cap = hctx.win_flags === nothing ? 0 : length(hctx.win_flags)
    needed <= cap && return nothing
    newcap = max(needed, cap + cld(cap, 2), 1024)
    hctx.win_flags = KA.allocate(backend, Int32, newcap)
    hctx.win_prefix = KA.allocate(backend, Int32, newcap)
    return nothing
end

"""
    ka_hier_cache_windows!(hctx, grid; workgroup=KA_AUTO_WORKGROUP)

Regenerate the complete per-level window concatenation for the current
occupancy epoch, in the same (level, offset-class window, class, source) order
as the host `build_hierarchical_routes_window!`. Runs inside the refresh, which
is legal because windows read node metadata only, never expansions.

Within a level the windows split the offset range `1:noffsets` into
consecutive blocks of `window_classes`, and each window enumerates
(offset, source) offset-major, so the windows of one level concatenated are
exactly one offset-major enumeration over `1:noffsets`: one flag launch and one
compact launch per level generate the stream, and only the route total is read
back.
"""
function ka_hier_cache_windows!(hctx::FastMultipole.DeviceHierarchicalM2LContext,
        grid; workgroup=KA_AUTO_WORKGROUP)
    scan = _ka_hier_windows_scan!(hctx, grid; workgroup)
    scan === nothing && return hctx
    backend = KA.get_backend(hctx.node_at)
    copyto!(hctx.win_host_total, 1, hctx.win_prefix, scan.total_used, 1)   # the one D2H sync
    _utick!(:win_sync_d2h, backend)
    return _ka_hier_windows_compact!(hctx, grid, scan, Int(hctx.win_host_total[1]);
        workgroup)
end

# Window cache phase 1: flag every level into one buffer and scan it. Returns
# `nothing` (cache already finalized as empty) when there is nothing to scan,
# else what phase 2 needs; the route total is `win_prefix[total_used]`.
function _ka_hier_windows_scan!(hctx, grid; workgroup=KA_AUTO_WORKGROUP)
    backend = KA.get_backend(hctx.node_at)
    noffsets = hctx.noffsets
    levels = hctx.first_m2l_level:hctx.ell
    _KA_UPDATE_TIMERS[] === nothing ||
        push!(get!(_KA_UPDATE_TIMERS[], :win_n_levels, Float64[]), length(levels))
    # level L's flags occupy bases[i]+1 : bases[i]+used[i]
    n_src(L) = hctx.level_offsets[L + 2] - hctx.level_offsets[L + 1]
    used = [n_src(L) * noffsets for L in levels]
    total_used = sum(used; init=0)
    # zero-M2L geometry (e.g. the all-direct fallback cache): no windows, no routes
    if total_used == 0
        hctx.total_routes = 0
        hctx.win_valid = true
        return nothing
    end
    bases = cumsum([0; used[1:end-1]])
    _ka_hier_win_scan_ensure!(hctx, backend, total_used)
    flags_all = hctx.win_flags
    flags_kernel = _cached_kernel(ka_hier_route_flags_kernel!, backend, workgroup)
    for (i, L) in enumerate(levels)
        used[i] > 0 || continue
        flags_kernel(view(flags_all, bases[i]+1:bases[i]+used[i]), hctx.node_at,
            grid.node_coords, hctx.d_push_offsets, hctx.d_class_of, hctx.level_base[L + 1],
            hctx.level_offsets[L + 1] + 1, n_src(L), 1, noffsets, L; ndrange=used[i])
    end
    accumulate!(+, view(hctx.win_prefix, 1:total_used), view(flags_all, 1:total_used))
    _utick!(:win_phase1_count, backend)
    return (; levels, used, bases, total_used)
end

# Window cache phase 2: compact the scanned flags straight into the cache.
function _ka_hier_windows_compact!(hctx, grid, scan, total_routes::Int;
        workgroup=KA_AUTO_WORKGROUP)
    backend = KA.get_backend(hctx.node_at)
    noffsets = hctx.noffsets
    (; levels, used, bases) = scan
    n_src(L) = hctx.level_offsets[L + 2] - hctx.level_offsets[L + 1]
    _ka_hier_win_ensure!(hctx, backend, total_routes)
    compact_kernel = _cached_kernel(ka_hier_route_compact_global_kernel!, backend, workgroup)
    for (i, L) in enumerate(levels)
        used[i] > 0 || continue
        class_base = (L - hctx.first_m2l_level) * noffsets
        compact_kernel(hctx.win_targets, hctx.win_sources, hctx.win_class,
            hctx.win_flags, hctx.win_prefix, bases[i], hctx.node_at, grid.node_coords,
            hctx.d_push_offsets, hctx.level_base[L + 1], hctx.level_offsets[L + 1] + 1,
            n_src(L), 1, noffsets, L, class_base; ndrange=used[i])
    end
    _utick!(:win_phase2_compact, backend)
    hctx.total_routes = total_routes
    hctx.win_valid = true
    return hctx
end

"""
    ka_hier_refresh_routes!(ctx, hctx, grid, n_cells, leaf_base, ell;
                            workgroup=KA_AUTO_WORKGROUP)

Epoch regeneration of the direct pairs and (concat plans) the M2L window
cache. When the direct pairs fit one flag chunk, both are flagged and scanned
first and their two totals come back in one transfer; otherwise this is
`ka_hier_generate_direct_pairs!` followed by `ka_hier_cache_windows!`. Returns
the direct pair count.
"""
function ka_hier_refresh_routes!(ctx, hctx::FastMultipole.DeviceHierarchicalM2LContext,
        grid, n_cells::Int, leaf_base::Int, ell::Int; workgroup=KA_AUTO_WORKGROUP)
    kn = size(hctx.d_near_offsets, 2)
    total = kn * n_cells
    windows = !hctx.win_valid && hctx.apply_plan isa FastMultipole.ResidentM2LConcatPlan
    if !windows || total == 0 || total > length(ctx.direct_flags)
        n_direct = ka_hier_generate_direct_pairs!(ctx, hctx, grid, n_cells, leaf_base,
            ell; workgroup)
        windows && ka_hier_cache_windows!(hctx, grid; workgroup)
        return n_direct
    end
    backend = KA.get_backend(ctx.direct_flags)
    level_base_L = hctx.level_base[ell + 1]
    flagk = _cached_kernel(ka_hier_direct_flags_kernel!, backend, workgroup)
    flagk(ctx.direct_flags, hctx.node_at, grid.node_coords, hctx.d_near_offsets,
        0, total, kn, leaf_base, level_base_L, ell; ndrange=total)
    accumulate!(+, view(ctx.direct_prefix, 1:total), view(ctx.direct_flags, 1:total))
    scan = _ka_hier_windows_scan!(hctx, grid; workgroup)
    # [direct pairs, routes] in one transfer
    fill!(ctx.route_scalars, Int32(0))
    copyto!(ctx.route_scalars, 1, ctx.direct_prefix, total, 1)
    scan === nothing ||
        copyto!(ctx.route_scalars, 2, hctx.win_prefix, scan.total_used, 1)
    copyto!(ctx.host_route_scalars, ctx.route_scalars)
    _utick!(:win_sync_d2h, backend)
    n_direct = Int(ctx.host_route_scalars[1])
    if n_direct > 0
        n_direct <= length(ctx.direct_targets) ||
            throw(AssertionError("device hierarchical direct pair buffer exceeded its capacity"))
        compactk = _cached_kernel(ka_hier_direct_compact_kernel!, backend, workgroup)
        compactk(ctx.direct_targets, ctx.direct_sources, ctx.direct_flags,
            ctx.direct_prefix, hctx.node_at, grid.node_coords,
            hctx.d_near_offsets, 0, total, kn, leaf_base, level_base_L, ell,
            0; ndrange=total)
    end
    scan === nothing ||
        _ka_hier_windows_compact!(hctx, grid, scan, Int(ctx.host_route_scalars[2]); workgroup)
    return n_direct
end
