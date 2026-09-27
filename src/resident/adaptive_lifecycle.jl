#------- adaptive octree host resident lifecycle: state assembly + drivers -------#
#
# Runs the full host resident lifecycle (theory/adaptive-radix-octree.md §2.6)
# on the task-039 adaptive octree: B2M -> M2M -> V-list M2L -> X-list S2L ->
# L2L -> (U-list direct + L2B + W-list M2T). Everything except the M2T/S2L
# operators (translate_batched.jl) is the EXISTING resident machinery running
# over a DeviceRadixGrid mirror of the adaptive node table:
#
#   - the 039 level-major node layout matches the uniform level_offsets
#     convention, so the per-level M2M/L2L edge groups refresh verbatim
#     (_refresh_resident_stage_groups!; the parent-child radius at child level
#     L is the same sqrt(3) h0 / 2^L the groups bake in at construction);
#   - adaptive LEAVES are presented as the state's "cells"
#     (cell_ranges/cell_centers/leaf_to_node over leaf_index), so the B2M,
#     L2B, and direct-pair host kernels run unchanged; U node endpoints map to
#     leaf-cell slots through `leaf_slot_of`;
#   - V-list M2L consumes the existing resident window plans and 025
#     level-scaled operator content UNCHANGED (task constraint), fed from the
#     039 CSR class stream (_launch_adaptive_resident_m2l!).
#
# The adaptive body sort (full-depth keys at ell_max) differs from the uniform
# grid's ell-depth sort, so the lifecycle packs its own source_bodies/output
# slabs in adaptive sorted order; finalize_radix_output! works verbatim because
# the state carries the tree's permutation metadata. Sort unification remains a
# recorded 039 open item.
#
# Contracts: all capacities fixed at construction; per-step refresh and the
# lifecycle are allocation-free after warm-up; the 023 host counter contract
# (all transfer counters remain zero) is asserted around the pipeline.

function _allocate_adaptive_resident_lifecycle(::Type{TF},
        basis_info::OperatorBasisInfo{B,LH}, options::RadixLifecycleOptions{TF},
        tree::AdaptiveRadixTree{TF}, lists::AdaptiveInteractionLists,
        invariant::OperatorInvariantCache, dpb::Int, maxn::Int,
        hessian::Bool) where {TF,B,LH}
    ell = tree.policy.ell_max
    node_cap = tree.node_capacity
    leaf_cap = min(node_cap, maxn)
    window_cap = max(1, min(lists.v_capacity, 1 << 15))
    # DeviceRadixGrid mirror of the adaptive node table. node_centers/node_keys
    # and the body permutation arrays ALIAS the tree (refreshed by
    # update_adaptive_tree!); node_levels/parent_index are Int mirrors of the
    # tree's Int32 columns; the leaf-as-cell arrays are gathered per step.
    # node_coords/child_ranges are allocated empty: nothing on the adaptive
    # host path reads them (the occupancy/window generators never run here).
    grid = DeviceRadixGrid(
        tree.x_min, tree.h0, ell, 0, 0,
        tree.perm, tree.invperm,
        Vector{UInt64}(undef, leaf_cap),      # cell_keys: leaf own-level keys (diagnostic)
        Matrix{Int}(undef, 2, leaf_cap),      # cell_ranges (first, count)
        tree.body_system, tree.body_index,
        Matrix{TF}(undef, 3, leaf_cap),       # cell_centers
        Vector{Int}(undef, node_cap),         # node_levels (Int mirror)
        tree.node_keys,
        Matrix{Int}(undef, 3, 0),             # node_coords: unused on this path
        tree.node_centers,
        Vector{Int}(undef, node_cap),         # parent_index (Int mirror)
        Matrix{Int}(undef, 2, 0),             # child_ranges: unused on this path
        Vector{Int}(undef, leaf_cap),         # leaf_to_node
    )
    multipoles = _host_flat_buffer(TF, basis_info, node_cap)
    locals_buf = _host_flat_buffer(TF, basis_info, node_cap)
    source_bodies = Matrix{TF}(undef, dpb, maxn)
    output = zeros(TF, hessian ? 13 : 4, maxn)
    route_levels = Vector{Int}(undef, window_cap)
    route_offsets = Matrix{Int}(undef, 3, window_cap)
    route_targets = Vector{Int}(undef, window_cap)
    route_sources = Vector{Int}(undef, window_cap)
    direct_targets = Vector{Int}(undef, lists.u_capacity)
    direct_sources = Vector{Int}(undef, lists.u_capacity)
    edge_placeholder = Int[]
    # The adaptive lifecycle reuses the uniform host cache's workspace builder
    # verbatim: per-level M2M/L2L capacity groups over levels 0:ell_max and a
    # hierarchical-mode M2L window plan over the 039 class metadata
    # (effective_offsets ordering == the CSR global class numbering). Same
    # strategy substitution as the uniform hierarchical cache: concat engine
    # unless a specialized dense/precomputed-y plan was selected explicitly.
    specialized = options.m2l_strategy isa Union{PrecomputedFactoredYM2L,
        DenseTranslationM2L}
    ws_strategy = specialized ? options.m2l_strategy : ConcatenatedFixedZM2L()
    ws_operator = specialized ? options.operator : MaterializedYRotationM2L()
    workspace = _radix_cache_workspace(TF, basis_info, multipoles, ell,
        TF(tree.h0), leaf_cap, node_cap, window_cap, lists.effective_offsets,
        invariant, ws_strategy, ws_operator;
        hierarchical_noffsets=lists.noffsets,
        ell_axes=SVector(ell, ell, ell), first_level=0)
    state = DeviceResidentRadixState{TF,CompressedComplexBasis,LH}(
        grid, lists, source_bodies, source_bodies,
        tree.perm, tree.body_system, tree.body_index,
        tree.perm, tree.body_system, tree.body_index,
        grid.cell_centers,
        edge_placeholder, edge_placeholder, edge_placeholder, edge_placeholder,
        grid.node_levels, tree.node_centers,
        route_targets, route_sources,
        grid.cell_centers, grid.cell_ranges,
        edge_placeholder, edge_placeholder, edge_placeholder, edge_placeholder,
        multipoles, locals_buf, route_levels, route_offsets, route_targets,
        route_sources, direct_targets, direct_sources, output,
        invariant, workspace, RadixTransferCounters(), options,
        RadixStepCounts(0, 0, 0, 0, 0);
    )
    P_phi = basis_info.orders.P_phi
    nH = harmonic_index(P_phi + 2, P_phi + 2)
    harmonics = Array{TF,3}(undef, 2, 1, nH)
    return AdaptiveResidentLifecycle(state, zeros(Int32, node_cap), harmonics,
        window_cap, leaf_cap, 0)
end

# Per-step refresh of the lifecycle mirrors from the freshly rebuilt tree +
# lists. Zero allocation; called from _refresh_adaptive_radix! after the tree
# and list rebuild (source_buffers are already packed by update_radix_state!).
function _refresh_adaptive_lifecycle!(al::AdaptiveResidentLifecycle,
        tree::AdaptiveRadixTree, lists::AdaptiveInteractionLists, source_buffers)
    _refresh_adaptive_lifecycle_typed!(al, al.state::DeviceResidentRadixState,
        tree, lists, source_buffers)
    return al
end

function _refresh_adaptive_lifecycle_typed!(al::AdaptiveResidentLifecycle,
        state::DeviceResidentRadixState{TF,B,LH}, tree::AdaptiveRadixTree{TF},
        lists::AdaptiveInteractionLists, source_buffers) where {TF,B,LH}
    grid = state.grid::DeviceRadixGrid
    n = tree.n_bodies
    n_nodes = tree.n_nodes
    n_leaves = tree.n_leaves
    n_leaves <= al.leaf_capacity || throw(AssertionError(
        "adaptive lifecycle leaf capacity $(al.leaf_capacity) exceeded ($n_leaves)"))
    @inbounds for f in 1:n_nodes
        grid.node_levels[f] = Int(tree.node_levels[f])
        grid.parent_index[f] = Int(tree.parent_index[f])
        al.leaf_slot_of[f] = Int32(0)
    end
    @inbounds for c in 1:n_leaves
        f = Int(tree.leaf_index[c])
        grid.leaf_to_node[c] = f
        grid.cell_ranges[1, c] = tree.node_lo[f]
        grid.cell_ranges[2, c] = tree.node_hi[f] - tree.node_lo[f] + 1
        grid.cell_centers[1, c] = tree.node_centers[1, f]
        grid.cell_centers[2, c] = tree.node_centers[2, f]
        grid.cell_centers[3, c] = tree.node_centers[3, f]
        grid.cell_keys[c] = tree.node_keys[f]
        al.leaf_slot_of[f] = Int32(c)
    end
    grid.n_bodies = n
    grid.n_cells = n_leaves
    _pack_radix_source_bodies!(state.source_bodies, tree.perm, tree.body_system,
        tree.body_index, source_buffers, n)
    # U endpoints (flat node indices) -> leaf cell slots for the direct kernel
    n_u = lists.n_u
    @inbounds for k in 1:n_u
        ts = al.leaf_slot_of[lists.u_targets[k]]
        ss = al.leaf_slot_of[lists.u_sources[k]]
        (ts != Int32(0) && ss != Int32(0)) || throw(AssertionError(
            "adaptive U-list endpoints must be leaves"))
        state.direct_targets[k] = Int(ts)
        state.direct_sources[k] = Int(ss)
    end
    # per-level M2M/L2L edge-group columns (existing refresh over the mirror;
    # ws.nonleaf_idx is refreshed too but deliberately unused on this path —
    # see _launch_adaptive_resident_m2m!)
    _refresh_resident_stage_groups!(
        state.scratch::ResidentOperatorWorkspace{TF,B,LH}, grid,
        tree.level_offsets)
    counts = state.counts
    counts.n_bodies = n
    counts.n_cells = n_leaves
    counts.n_nodes = n_nodes
    counts.n_routes = lists.n_routes
    counts.n_direct = n_u
    al.step += 1
    return al
end

# X-list S2L stage: coarse source leaves accumulate directly into finer target
# cells' local expansions (before L2L, which carries them to the leaves).
function _launch_adaptive_s2l!(state::DeviceResidentRadixState{TF,B,LH},
        tree::AdaptiveRadixTree{TF}, lists::AdaptiveInteractionLists,
        al::AdaptiveResidentLifecycle) where {TF,B,LH}
    lists.n_x == 0 && return state
    orders = state.invariant_cache.basis_info.orders
    H = al.harmonics::Array{TF,3}
    if state.options.body_type <: Point{Vortex}
        _host_s2l_vortex_pairs_kernel!(phi_slab(state.locals),
            chi_slab(state.locals), state.source_bodies, tree.node_lo,
            tree.node_hi, tree.node_centers, lists.x_targets, lists.x_sources,
            lists.n_x, H, orders.P_phi, orders.P_active)
    elseif state.options.body_type <: Point{Source}
        _host_s2l_pairs_kernel!(phi_slab(state.locals), state.source_bodies,
            tree.node_lo, tree.node_hi, tree.node_centers, lists.x_targets,
            lists.x_sources, lists.n_x, H, orders.P_phi)
    else
        throw(ArgumentError("adaptive S2L supports Point{Source} and " *
            "Point{Vortex}; got $(state.options.body_type)"))
    end
    return state
end

# W-list M2T stage: finer source cells' multipoles evaluated directly at coarse
# target leaves' bodies (after L2B, accumulating into the same output slab).
function _launch_adaptive_m2t!(state::DeviceResidentRadixState{TF,B,LH},
        tree::AdaptiveRadixTree{TF}, lists::AdaptiveInteractionLists,
        al::AdaptiveResidentLifecycle) where {TF,B,LH}
    lists.n_w == 0 && return state
    orders = state.invariant_cache.basis_info.orders
    H = al.harmonics::Array{TF,3}
    hsv = size(state.output, 1) >= 13 ? Val(true) : Val(false)
    _host_m2t_pairs_kernel!(state.output, state.source_bodies, tree.node_lo,
        tree.node_hi, tree.node_centers, lists.w_targets, lists.w_sources,
        lists.n_w, phi_slab(state.multipoles), chi_slab(state.multipoles), H,
        orders.P_phi, orders.P_active, Val(LH), hsv)
    return state
end

"""
    run_adaptive_host_radix_lifecycle!(cache)

Execute the full host resident lifecycle on the cache's adaptive octree (task
040): B2M at every leaf, M2M over the occupied ancestor levels, V-list M2L
through the unchanged resident window plans, X-list S2L, L2L, then U-list
direct + L2B + W-list M2T into the adaptive output slab. Requires a cache
constructed with an `AdaptiveTreePolicy`; called by the host `fmm!` branch.
"""
function run_adaptive_host_radix_lifecycle!(cache::RadixFMMCache)
    al = cache.adaptive_state
    al isa AdaptiveResidentLifecycle || throw(ArgumentError(
        "run_adaptive_host_radix_lifecycle! requires a cache constructed with " *
        "an AdaptiveTreePolicy"))
    _run_adaptive_host_lifecycle_typed!(al, al.state::DeviceResidentRadixState,
        cache.adaptive_tree::AdaptiveRadixTree,
        cache.adaptive_lists::AdaptiveInteractionLists)
    return cache
end

function _run_adaptive_host_lifecycle_typed!(al::AdaptiveResidentLifecycle,
        state::DeviceResidentRadixState{TF,B,LH}, tree::AdaptiveRadixTree{TF},
        lists::AdaptiveInteractionLists) where {TF,B,LH}
    state.counters.expansion_host_copies == 0 || throw(AssertionError(
        "adaptive host radix lifecycle observed expansion host copies before execution"))
    _launch_host_b2m!(state)
    _launch_adaptive_resident_m2m!(state)
    _launch_adaptive_resident_m2l!(state, lists, al.route_window_capacity)
    _launch_adaptive_s2l!(state, tree, lists, al)
    _launch_resident_l2l!(state)
    _launch_host_l2b!(state)
    _launch_adaptive_m2t!(state, tree, lists, al)
    state.counters.expansion_host_copies == 0 || throw(AssertionError(
        "adaptive host radix lifecycle observed expansion host copies"))
    return state
end
