#------- backend-agnostic device step (the whole uniform sfs=false lifecycle) -------#
#
# Device-resident state refresh for the branch FLOWVPM runs: a uniform grid
# with a `HierarchicalRigidStencil`, hence
# `ctx.hierarchical_ctx !== nothing`. Everything it calls is backend-agnostic --
# the four grid-rebuild stages, the hierarchical occupancy/direct-pair/
# window-cache refresh, the stage-group edges, the lifecycle body, and the
# output finalize.
#
# Notes:
#   * the within-cell sub-Morton reordering (always on with PartitionedVortex)
#     changes no cell key, cell range or node -- only the
#     order bodies are summed in -- so it is locality-only, never correctness.
#   * there is no graph capture/replay; the body is launched directly every step.
#   * a consumer near-field pass (`nearfield_pass`) runs between the lifecycle
#     body and the U/J finalize.

@kernel function ka_iota_kernel!(perm, invperm, n)
    i = @index(Global)
    @inbounds if i <= n
        perm[i] = i
        invperm[i] = i
    end
end

"Identity body permutation for the all-pairs direct arm, which does no sort."
function _ka_identity_perm!(perm, invperm, n::Int; workgroup=KA_AUTO_WORKGROUP)
    backend = KA.get_backend(perm)
    wg = resolve_workgroup(backend, workgroup)
    kern = _cached_kernel(ka_iota_kernel!, backend, wg)
    kern(perm, invperm, n; ndrange=cld(n, wg) * wg)
    return perm
end

# Optional per-stage timers for ka_update_radix_state!, for cost attribution:
# set `_KA_UPDATE_TIMERS[] = Dict{Symbol,Vector{Float64}}()` and every
# `_utick!` syncs the backend and records the time since the previous tick.
# `nothing` (the default) makes each tick a no-op.
const _KA_UPDATE_TIMERS = Ref{Any}(nothing)
const _KA_UPDATE_T0 = Ref{Float64}(0.0)
@inline function _utick!(name::Symbol, backend)
    d = _KA_UPDATE_TIMERS[]
    d === nothing && return nothing
    KA.synchronize(backend)
    t = time(); dt = t - _KA_UPDATE_T0[]
    push!(get!(d, name, Float64[]), dt * 1e3); _KA_UPDATE_T0[] = t
    return nothing
end


"""
    ka_update_radix_state!(cache, systems; workgroup=KA_AUTO_WORKGROUP, direct_only=false)

Device state refresh for a uniform, hierarchical `RadixFMMCache(device=true)` on
any KA backend: refresh the per-system source buffers, rebuild the grid in place
inside the cache's fixed Morton box, refresh the hierarchical occupancy / direct
pairs / cached M2L windows on occupancy change, and refresh the per-level
operator-group edges. `direct_only=true` skips the grid, tree and routes and
packs bodies in identity order for the all-pairs arm. Returns the cache with
`cache.state` built (first step) or refreshed in place.
"""
function ka_update_radix_state!(cache::FastMultipole.RadixFMMCache{TF,LH}, systems::Tuple;
        workgroup=KA_AUTO_WORKGROUP, direct_only::Bool=false) where {TF,LH}
    ctx = cache.device_ctx
    ctx === nothing &&
        throw(ArgumentError("ka_update_radix_state! requires a cache built with device=true"))
    length(systems) == cache.n_systems ||
        throw(ArgumentError("cache was built for $(cache.n_systems) source systems, got $(length(systems))"))
    n = FastMultipole.get_n_bodies(systems)
    n > 0 || throw(ArgumentError("ka_update_radix_state! requires at least one body"))
    n <= cache.max_n_bodies ||
        throw(ArgumentError("n=$n exceeds the cache capacity max_n_bodies=$(cache.max_n_bodies)"))
    grid = ctx.grid
    hctx = ctx.hierarchical_ctx
    counters = ctx.counters
    backend = KA.get_backend(ctx.positions)
    ell = cache.ell
    first_level = cache.root_level

    source_buffers = FastMultipole._radix_cache_refresh_source_buffers!(ctx, systems, TF)
    _KA_UPDATE_TIMERS[] === nothing || (KA.synchronize(backend); _KA_UPDATE_T0[] = time())
    if direct_only
        # all-pairs arm: no grid, no tree, no routes, no box check. Bodies are
        # packed in identity order -- the perm's only consumer on this arm is
        # `ka_finalize_radix_output!`, which scatters back through it.
        #
        # The system/index tags still have to be refreshed: the pack kernel
        # writes only columns whose tag matches, so bodies added since the
        # last tagging call were packed as zeros (position at the origin, no
        # strength) -- exact on a fixed field, wrong on a growing one.
        ka_collect_positions!(ctx.positions, grid.body_system, grid.body_index,
            source_buffers; workgroup)
        _ka_identity_perm!(grid.perm, grid.invperm, n; workgroup)
        n_cells = 0
        n_nodes = 0
        grid.n_bodies = n
        grid.n_cells = 0
    else
        ka_collect_positions!(ctx.positions, grid.body_system, grid.body_index,
            source_buffers; workgroup)

        # ---- grid rebuild, stages 1-4 ----
    _utick!(:collect_positions, backend)
        ka_radix_keys_checked!(view(ctx.keys, 1:n), ctx.oob_flag, ctx.host_oob,
            ctx.positions, cache.x_min, cache.box_extent, cache.h0, ell;
            ell_axes=cache.ell_axes, workgroup)
        kv = view(ctx.keys, 1:n)
        sk = view(ctx.sorted_keys, 1:n)
        # the bounded counting sort when the cache was built with a domain-sized
        # histogram (ell <= KA_COUNTING_SORT_MAX_ELL), else the stable sortperm
    _utick!(:keys, backend)
        ka_radix_sort_bodies!(view(grid.perm, 1:n), sk, grid.invperm, kv; workgroup,
            ell=ell, histogram=ctx.counting_histogram, prefix=ctx.counting_prefix,
            cursor=ctx.counting_cursor)
        n_cells = ka_radix_compress_cells!(grid.cell_keys, grid.cell_ranges, sk,
            view(ctx.body_flags, 1:n), view(ctx.body_prefix, 1:n), ctx.host_scalar;
            workgroup)
        n_cells <= cache.max_cells ||
            throw(AssertionError("device radix grid exceeded the cache cell capacity"))
        ckv = view(grid.cell_keys, 1:n_cells)

        # occupancy epoch: everything past this point is a pure function of the
        # occupied leaf-cell SET inside the fixed box
        track_epoch = length(ctx.epoch_cell_keys) > 0
    _utick!(:sort_compress, backend)
        # The epoch is keyed on the occupied-cell SET, not on the body count:
        # bodies added to already-occupied cells (a shedding solver, every step)
        # leave every route, window and stage group valid. Compare the keys
        # whenever the cell count matches; a count-only change must not force
        # the rebuild (measured at ~420 ms per call at 400 bodies on Metal,
        # against ~15 ms for the evaluation itself).
        occ_changed = true
        if track_epoch && ctx.epoch_have[] && ctx.epoch_prev_n_cells[] == n_cells
            occ_changed = ka_radix_occupancy_changed!(ctx.epoch_flag, ctx.host_epoch_flag,
                ckv, ctx.epoch_cell_keys, n_cells; workgroup)
        end
    _utick!(:occ_check, backend)
        if occ_changed
            ctx.epoch_id[] += 1
            # the snapshot is committed only once every epoch-derived stage has
            # succeeded (end of this function); until then a retry must rebuild
            ctx.epoch_have[] = false
            ka_radix_cell_centers!(grid.cell_centers, ctx.cell_coords, ckv, cache.x_min,
                cache.h0, ell, n_cells; workgroup)
            n_nodes, max_count = ka_radix_level_nodes!(grid.node_keys, cache.level_offsets,
                ctx.level_keys, ctx.level_flags, ctx.level_prefix, ctx.level_counts,
                ctx.host_level_counts, ctx.d_level_offsets, ckv, n_cells, ell,
                first_level, cache.max_nodes; workgroup)
            ka_radix_node_topology!(grid.node_levels, grid.node_coords, grid.node_centers,
                grid.parent_index, grid.child_ranges, view(grid.leaf_to_node, 1:n_cells),
                grid.node_keys, ctx.d_level_offsets, cache.level_offsets, cache.x_min,
                cache.h0, n_cells, ell, first_level, max_count; workgroup)
        end
        n_nodes = cache.level_offsets[end]
        grid.n_bodies = n
        grid.n_cells = n_cells
    end

    # optional within-cell sub-Morton ordering,
    # composed into the perm before packing (the sorted cell keys, cell ranges
    # and node metadata are unaffected).
    _utick!(:grid_rebuild, backend)
    if !direct_only && cache.options.direct_kernel isa FastMultipole.PartitionedVortex
        # NOT the refresh's `workgroup`: the local-memory sort's group size is
        # baked into the kernel (Val(WG) against a fixed capacity), so it is the
        # kernel's own constant, not a tuning surface.
        ka_nearfield_subsort!(ctx, cache, n, n_cells)
    end
    pack_sigma_row = _ka_kernel_sigma_row(cache.options.direct_kernel)
    # reciprocal-sigma row = the last row for a regularized kernel (convention
    # of the two allocators; see `_ka_nf_inv_sigma_row`)
    pack_inv_sigma_row = pack_sigma_row > 0 ? size(ctx.source_bodies, 1) : 0
    _utick!(:subsort, backend)
    for isys in eachindex(source_buffers)
        ka_pack_body_matrix!(ctx.source_bodies, source_buffers[isys],
            view(grid.perm, 1:n), grid.body_system, grid.body_index, n;
            isys, sigma_row=pack_sigma_row, inv_sigma_row=pack_inv_sigma_row,
            workgroup)
    end
    # the adequacy gate guards the M2L far field; on the all-pairs arm there is
    # none, so it is vacuous (same reasoning as the zero-M2L degenerate cache).
    # An inadequate hierarchical geometry demotes to the all-direct zero-M2L
    # cache and re-runs the refresh, as on the host; the rebuilt cache's
    # gate is vacuous, so the recursion terminates after one demotion.
    if !direct_only && FastMultipole._direct_kernel_geometry_gate!(cache,
            cache.options.direct_kernel, ctx.source_bodies, n) === :alldirect
        FastMultipole._alldirect_geometry_fallback!(cache, systems)
        return ka_update_radix_state!(cache, systems; workgroup)
    end

    # host mirrors serve host-resident target finalization only
    _utick!(:pack, backend)
    if FastMultipole._radix_any_host_resident(systems)
        KA.synchronize(backend)
        copyto!(ctx.host_perm, 1, grid.perm, 1, n)
        copyto!(ctx.host_body_system, 1, grid.body_system, 1, n)
        copyto!(ctx.host_body_index, 1, grid.body_index, 1, n)
        counters.metadata_downloads += 3
    end

    _utick!(:host_copy, backend)
    if direct_only
        n_routes = 0
        n_direct = 0
    else
        hctx === nothing && throw(ArgumentError(
            "ka_update_radix_state! covers the hierarchical stencil path only; " *
            "build the cache with window_classes (FLOWVPM's default)"))
        if occ_changed
            hctx.epoch_id += 1
            hctx.win_valid = false
            ka_hier_refresh_occupancy!(hctx, grid, cache.level_offsets; workgroup)
            n_direct = ka_hier_generate_direct_pairs!(ctx, hctx, grid, n_cells,
                cache.level_offsets[ell + 1], ell; workgroup)
            hctx.epoch_n_direct = n_direct
        else
            n_direct = hctx.epoch_n_direct
        end
    _utick!(:refresh_occ_direct_pairs, backend)
        # Occupancy-epoch window cache. The CONCAT apply consumes only
        # (class, source, target), which is exactly what the cache stores, and it
        # takes no level argument because the level is already baked into the
        # class, so the whole epoch collapses to one apply.
        if !hctx.win_valid && hctx.apply_plan isa FastMultipole.ResidentM2LConcatPlan
            ka_hier_cache_windows!(hctx, grid; workgroup)
        end
        n_routes = hctx.win_valid ? hctx.total_routes : 0

    _utick!(:cache_windows, backend)
        if occ_changed
            ka_refresh_resident_stage_groups!(ctx.workspace, grid, cache.level_offsets,
                ell, cache.root_level; workgroup)
        end
    end
    _utick!(:stage_groups, backend)
    KA.synchronize(backend)
    # occupancy-epoch snapshot, after the node rebuild, direct pairs, windows
    # and stage groups it stands for: a throw in any of them leaves no snapshot,
    # so the next call rebuilds instead of trusting half-written state
    if !direct_only && occ_changed && length(ctx.epoch_cell_keys) > 0
        copyto!(ctx.epoch_cell_keys, 1, grid.cell_keys, 1, n_cells)
        ctx.epoch_prev_n_cells[] = n_cells
        ctx.epoch_have[] = true
    end

    counts = ctx.counts
    counts.n_bodies = n
    counts.n_cells = n_cells
    counts.n_nodes = n_nodes
    counts.n_routes = n_routes
    counts.n_direct = n_direct

    if cache.state === nothing
        # every array here is persistent; the wrapper is built once and
        # refreshed in place on later steps
        cache.state = FastMultipole.DeviceResidentRadixState{TF,FastMultipole.CompressedComplexBasis,LH}(
            grid, hctx, ctx.source_bodies,
            grid.perm, grid.body_system, grid.body_index,
            ctx.host_perm, ctx.host_body_system, ctx.host_body_index, nothing,
            grid.cell_centers, grid.cell_ranges,
            ctx.m2m_parent_routes, ctx.m2m_child_routes,
            ctx.l2l_parent_routes, ctx.l2l_child_routes,
            ctx.multipoles, ctx.locals,
            # flat route staging: only the host lifecycle's flat M2L reads it;
            # the device M2L applies the epoch window cache (`hctx.win_*`)
            KA.zeros(backend, Int, 0), KA.zeros(backend, Int, 3, 0),
            KA.zeros(backend, Int, 0), KA.zeros(backend, Int, 0),
            ctx.direct_targets, ctx.direct_sources, ctx.output,
            ctx.invariant, ctx.workspace, counters, cache.options, counts;
        )
    end
    cache.step += 1
    return cache
end

ka_update_radix_state!(cache::FastMultipole.RadixFMMCache, systems; kwargs...) =
    ka_update_radix_state!(cache, FastMultipole.to_tuple(systems); kwargs...)
