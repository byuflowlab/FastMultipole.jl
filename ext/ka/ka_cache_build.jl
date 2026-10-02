#------- device-resident cache construction -------#
#
# `RadixFMMCache(...; device=true)` reaches this builder through the device
# backend registry. It produces a real `RadixFMMCache` whose `device_ctx` is
# backend-allocated, so `ka_radix_cache_device_step!` runs end to end and a
# full UJ can be gated against the host lifecycle.
#
# Scope, matching what the KA step actually implements (each omission is a guard,
# not a silent fallback):
#   * hierarchical stencil policy only -- `ka_update_radix_state!` refuses a flat
#     cache, and FLOWVPM's `window_classes` cache is hierarchical anyway
#     (see the header of ka_device_step.jl).
#   * concatenated M2L only -- `ka_radix_cache_workspace` pins
#     `ConcatenatedFixedZM2L`/`MaterializedYRotationM2L`. A dense or
#     precomputed-y strategy would silently get a different plan, so it throws.
#   * no built-in SFS pass; a consumer supplies one through `nearfield_pass`.
#
# The host mirrors are plain `Array`s (no pinned memory). `ka_launch_nearfield!`
# runs the unbinned functor kernel, so a `PartitionedVortex` cache needs no
# distance-binned scratch (binning is locality only).
function ka_radix_cache_device_build(backend, sources::Tuple, P::Int, ell::Int,
        x_min::SVector{3,TF}, h0::TF, maxn::Int,
        options::FastMultipole.RadixLifecycleOptions,
        stencil_policy, accepted::Vector{SVector{3,Int}},
        rejected::Vector{SVector{3,Int}}, max_cells::Int, max_nodes::Int,
        route_capacity::Int, direct_capacity::Int,
        basis_info::FastMultipole.OperatorBasisInfo{B,LH}, ::Val{LH};
        hierarchical_tables=nothing,
        hierarchical_level_class_of::Array{Int32,3}=Array{Int32}(undef, 0, 0, 0),
        hessian::Bool=false,
        ell_axes::SVector{3,Int}=SVector(ell, ell, ell),
        box_extent::SVector{3,TF}=SVector{3,TF}(2 * h0, 2 * h0, 2 * h0),
        root_level::Int=0, first_m2l_level::Int=2,
        direct_flag_capacity::Int=min(direct_capacity, 1 << 21),
        direct_pair_capacity::Int=min(direct_capacity, 1 << 16),
        workgroup=KA_AUTO_WORKGROUP) where {TF,B,LH}
    direct_flag_capacity > 0 || throw(ArgumentError("direct_flag_capacity must be positive"))
    0 < direct_pair_capacity <= max(direct_capacity, 1) || throw(ArgumentError(
        "direct_pair_capacity must be in 1:direct_capacity ($direct_capacity); got $direct_pair_capacity"))
    stencil_policy isa FastMultipole.HierarchicalRigidStencil || throw(ArgumentError(
        "ka_radix_cache_device_build covers the hierarchical stencil path only; " *
        "got $(typeof(stencil_policy))"))
    hierarchical_tables isa FastMultipole.RigidHierarchicalTables || throw(ArgumentError(
        "the hierarchical policy requires the rigid hierarchical tables"))
    # the KA lifecycle has no pass-2 deficit sweep: a TwoPassVortex cache would
    # run pass 1 only and report rows 2:13 without the rho_c..rho_t deficit
    options.direct_kernel isa FastMultipole.TwoPassVortex && throw(ArgumentError(
        "TwoPassVortex is not available on a KernelAbstractions device cache (no pass-2 deficit sweep); " *
        "use RegularizedVortex or PartitionedVortex, or build the cache with device=false"))
    options.m2l_strategy isa FastMultipole.ConcatenatedFixedZM2L || throw(ArgumentError(
        "the KA device cache builds the concatenated hierarchical plan; " *
        "m2l_strategy=$(typeof(options.m2l_strategy)) has no KA plan (DenseTranslationM2L is host-only)"))

    _z(T, dims...) = fill!(KA.allocate(backend, T, dims...), zero(T))

    counters = FastMultipole.RadixTransferCounters()
    multipoles = _ka_flat_buffer(backend, TF, basis_info, max_nodes)
    locals = _ka_flat_buffer(backend, TF, basis_info, max_nodes)
    invariant = FastMultipole.OperatorInvariantCache(TF, basis_info)
    workspace = ka_radix_cache_workspace(backend, TF, basis_info, ell, h0,
        max_cells, max_nodes, route_capacity, accepted, invariant;
        ell_axes, first_level=root_level, m2l_strategy=options.m2l_strategy)

    # capacity-sized persistent grid: counts bound the valid prefixes, so
    # recurring steps refresh these arrays in place and never reallocate
    grid = FastMultipole.DeviceRadixGrid(
        x_min, h0, ell, 0, 0,
        _z(Int, maxn), _z(Int, maxn),
        _z(UInt64, max_cells), _z(Int, 2, max_cells),
        _z(Int, maxn), _z(Int, maxn),
        _z(TF, 3, max_cells),
        _z(Int, max_nodes), _z(UInt64, max_nodes),
        _z(Int, 3, max_nodes), _z(TF, 3, max_nodes),
        _z(Int, max_nodes), _z(Int, 2, max_nodes),
        _z(Int, max_cells),
    )

    # per-system source staging: host-resident systems get a host buffer plus a
    # persistent device buffer (one upload per step); device-resident systems get
    # a persistent device buffer their `source_to_buffer!` overload fills in place
    host_stagings = Tuple(
        FastMultipole.residency(system) isa FastMultipole.HostResident ?
            Matrix{TF}(undef, FastMultipole.data_per_body(system), maxn) : nothing
        for system in sources)
    device_sources = Tuple(
        _z(TF, FastMultipole.data_per_body(system), maxn) for system in sources)

    # the hierarchical path keeps its stencil tables on the device hierarchical
    # context instead of a flat accepted/rejected pair, so these stay empty --
    # as do the leaf-only flat occupancy and route flag storage
    d_accepted = KA.zeros(backend, Int32, 3, 0)
    d_rejected = KA.zeros(backend, Int32, 3, 0)
    occupancy = FastMultipole.RadixLevelOccupancy(ell;
        max_bytes=stencil_policy.dense_occupancy_max_bytes,
        max_dense_ell=stencil_policy.dense_occupancy_max_ell)
    isempty(occupancy.node_at) && throw(ArgumentError(
        "the device hierarchical stencil requires the dense per-level occupancy " *
        "lookup, but ell=$ell exceeds the configured budget " *
        "(dense_occupancy_max_bytes=$(stencil_policy.dense_occupancy_max_bytes), " *
        "dense_occupancy_max_ell=$(stencil_policy.dense_occupancy_max_ell)); the " *
        "host Morton binary-search fallback has no device implementation"))
    # The concat plan is the apply plan (the dense strategy is refused above).
    # Classes are the UNSCALED push offsets built at the leaf cell width.
    apply_plan = workspace.m2l_concat
    # The M2L routes are compacted straight into the epoch window cache
    # (`hctx.win_*`, sized to the measured route total), so no per-window route
    # staging is allocated.
    hierarchical_ctx = ka_hierarchical_context(backend, hierarchical_tables,
        hierarchical_level_class_of, apply_plan, ell, first_m2l_level,
        occupancy; window_classes=stencil_policy.window_classes)

    dpb = maximum(FastMultipole.data_per_body(system) for system in sources)
    n_output_rows = hessian ? 13 : 4
    ctx = (;
        multipoles, locals, workspace, invariant, counters, grid,
        counts=FastMultipole.RadixStepCounts(0, 0, 0, 0, 0),
        # + 1 row of 1/sigma for a regularized kernel (see `_ka_nf_inv_sigma_row`)
        source_bodies=_z(TF, dpb + (_ka_kernel_sigma_row(options.direct_kernel) > 0), maxn),
        output=_z(TF, n_output_rows, maxn),
        cell_at=_z(Int32, 0, 0, 0),
        hierarchical_ctx,
        d_accepted, d_rejected, class_chunk=1,
        # near pairs: start at direct_pair_capacity and grow in place (resize!)
        # up to direct_capacity when an epoch needs more (`ka_hier_generate_direct_pairs!`);
        # the capacity bound is every (occupied cell, near offset) of the
        # full grid, of which a wake uses a few percent
        direct_targets=_z(Int, direct_pair_capacity),
        direct_sources=_z(Int, direct_pair_capacity),
        direct_capacity,
        # flag/scan scratch for the near-pair compaction, which walks the
        # (cell, near offset) candidates in chunks of this length: bounded, not
        # sized to the full pair capacity (`ka_hier_generate_direct_pairs!`)
        direct_flags=_z(Int32, direct_flag_capacity),
        direct_prefix=_z(Int32, direct_flag_capacity),
        # [direct pairs, M2L routes]: the epoch route regeneration readback
        route_scalars=_z(Int32, 2),
        host_route_scalars=zeros(Int32, 2),
        # grid-update scratch: persistent, so the recurring step allocates
        # nothing beyond the backend's own sort scratch
        positions=_z(TF, 3, maxn),
        keys=_z(UInt64, maxn),
        sorted_keys=_z(UInt64, maxn),
        # bounded counting-sort domain buffers: the full 2^(3ell) key domain
        # when the fast path is enabled for this `ell`, else length 1, which is
        # itself the signal `ka_counting_sort_ready` falls back on
        counting_histogram=_z(Int32, ka_counting_sort_enabled(ell) ? 1 << (3 * ell) : 1),
        counting_prefix=_z(Int32, ka_counting_sort_enabled(ell) ? 1 << (3 * ell) : 1),
        counting_cursor=_z(Int32, ka_counting_sort_enabled(ell) ? 1 << (3 * ell) : 1),
        body_flags=_z(Int, maxn),
        body_prefix=_z(Int, maxn),
        subsort_keys=_z(UInt32, maxn),
        cell_coords=_z(Int, 3, max_cells),
        level_keys=_z(UInt64, max_cells, ell + 1),
        level_flags=_z(Int, max_cells, ell + 1),
        level_prefix=_z(Int, max_cells, ell + 1),
        level_counts=_z(Int, ell + 1),
        d_level_offsets=_z(Int, ell + 2),
        oob_flag=_z(Int32, 1),
        # occupancy-epoch snapshot: the hierarchical cache compares the sorted
        # unique leaf keys against the previous step to skip node-metadata,
        # window and direct-pair regeneration
        epoch_cell_keys=_z(UInt64, max_cells),
        epoch_flag=_z(Int32, 1),
        host_epoch_flag=zeros(Int32, 1),
        # [n_cells, keys differ]: the compress readback with the epoch compare
        step_scalars=_z(Int, 2),
        host_step_scalars=zeros(Int, 2),
        epoch_prev_n_cells=Ref(0),
        epoch_have=Ref(false),
        epoch_id=Ref(0),                 # bumps whenever the occupied-cell set changes
        # extra tree sources prepared on the current epoch, by objectid (see
        # `_ka_extra_tree_prepared!`)
        extra_tree_cache=Dict{UInt,Any}(),
        extra_tree_hits=Ref(0), extra_tree_misses=Ref(0),   # reuse accounting, for profiles
        # tree-edge route arrays: fields of `DeviceResidentRadixState` that only
        # the host lifecycle reads; the KA M2M/L2L walk the stage-group edges
        m2m_parent_routes=_z(Int, 0),
        m2m_child_routes=_z(Int, 0),
        l2l_parent_routes=_z(Int, 0),
        l2l_child_routes=_z(Int, 0),
        host_stagings, device_sources,
        # host mirrors/staging for the step downloads
        host_oob=zeros(Int32, 1),
        host_scalar=zeros(Int, 1),
        host_scalar32=zeros(Int32, 1),
        host_level_counts=zeros(Int, ell + 1),
        host_perm=zeros(Int, maxn),
        host_body_system=zeros(Int, maxn),
        host_body_index=zeros(Int, maxn),
        # must track the output row count, or the prefix copyto! in
        # `ka_finalize_radix_output!` silently mis-strides
        host_output=zeros(TF, n_output_rows, maxn),
        device_target_buffers=Dict{Int,Any}(),
    )
    cache = FastMultipole.RadixFMMCache{TF,LH}(
        P, ell, x_min, h0, ell_axes, box_extent, root_level, maxn, true, hessian,
        options, stencil_policy,
        accepted, rejected, max_cells, max_nodes, route_capacity, direct_capacity,
        nothing, zeros(Int32, 0, 0, 0), SVector{3,Int}[], zeros(Int, ell + 2),
        UInt64[], Int[], Int[], Int[], nothing, nothing, ctx,
        length(sources), false, 0,
        FastMultipole.snapshot_locked_radix_settings(),
    )
    ka_update_radix_state!(cache, sources; workgroup)
    cache.built = true
    return cache
end

ka_radix_cache_device_build(backend, sources, args...; kwargs...) =
    ka_radix_cache_device_build(backend, FastMultipole.to_tuple(sources), args...;
        kwargs...)
