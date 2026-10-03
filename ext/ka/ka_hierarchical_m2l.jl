#------- hierarchical M2L window generation (KA) -------#
#
# The flag and compact kernels behind `ka_hier_cache_windows!`
# (ka_finalize_refresh.jl), which fills the epoch route cache
# `hctx.win_targets`/`win_sources`/`win_class`, level by level and within a
# level target by target.
#
# `DeviceHierarchicalM2LContext` is array-type generic (IV32/IM32/IA32/IV/SM
# type parameters), so it lives on any KA backend as is. The scan is
# `accumulate!`, which is backend-generic via GPUArrays.
#
# Both kernels are elementwise index math with no shared memory, so they carry
# no `@localmem` team-size coupling: both take `KA_AUTO_WORKGROUP`.

@kernel function ka_hier_route_flags_kernel!(flags, @Const(node_at),
        @Const(node_coords), @Const(push_offsets), @Const(class_of),
        level_base_L, first_source, n_sources, first_offset, kn, L)
    idx = @index(Global)
    @inbounds if idx <= kn * n_sources
        # target-major: the routes into one target are adjacent in the stream,
        # which the M2L scatter needs to sum them in a fixed order. The route
        # of offset k into `target` comes from the node at target - offset.
        t = (idx - 1) ÷ kn + 1
        kloc = (idx - 1) % kn + 1
        k = first_offset + kloc - 1
        target = first_source + t - 1
        G = 1 << L
        cx = node_coords[1, target] - push_offsets[1, k]
        cy = node_coords[2, target] - push_offsets[2, k]
        cz = node_coords[3, target] - push_offsets[3, k]
        hit = Int32(0)
        if 0 <= cx < G && 0 <= cy < G && 0 <= cz < G
            # the source's phase, same x/y/z bit convention as _rigid_phase_index
            phase = 1 + (cx & 1) + 2 * (cy & 1) + 4 * (cz & 1)
            if class_of[phase, k, L + 1] != Int32(0)
                linear = cx + G * (cy + G * cz)
                node_at[level_base_L + linear + 1] == Int32(0) || (hit = Int32(1))
            end
        end
        flags[idx] = hit
    end
end

# Compact for the concat window cache: with one scan over all windows,
# prefix[base+idx] IS the route's slot in the concatenated stream, so the
# compact writes the cache arrays directly -- no per-window staging, no
# device-to-device copies. Only the three arrays the concat apply reads.
@kernel function ka_hier_route_compact_global_kernel!(win_targets, win_sources, win_class,
        @Const(flags), @Const(prefix), base, @Const(node_at), @Const(node_coords),
        @Const(push_offsets), level_base_L, first_source, n_sources, first_offset, kn, L,
        class_base)
    idx = @index(Global)
    @inbounds if idx <= kn * n_sources && flags[base + idx] == Int32(1)
        t = (idx - 1) ÷ kn + 1
        kloc = (idx - 1) % kn + 1
        k = first_offset + kloc - 1
        target = first_source + t - 1
        G = 1 << L
        cx = node_coords[1, target] - push_offsets[1, k]
        cy = node_coords[2, target] - push_offsets[2, k]
        cz = node_coords[3, target] - push_offsets[3, k]
        linear = cx + G * (cy + G * cz)
        p = Int(prefix[base + idx])
        win_targets[p] = target
        win_sources[p] = Int(node_at[level_base_L + linear + 1])
        win_class[p] = Int32(class_base + k)
    end
end

#------- resident stage-group edge refresh (KA) -------#
#
# Rebuilds the per-level M2M/L2L edge columns -- (source, target) node index
# pairs plus the spherical angles of the parent-child displacement -- from the
# refreshed grid, once per occupancy change inside `ka_update_radix_state!`
# (the device counterpart of the host `_refresh_resident_stage_groups!`).
#
# `TF` is threaded in as a type argument rather than taken from `eltype(phis)`
# inside the kernel: the group fields are `Any`-typed, and an in-kernel
# `eltype` of such a field fails to resolve on Metal.
@kernel function ka_refresh_group_edges_kernel!(source_idx, target_idx, phis,
        thetas, @Const(parent_index), @Const(node_centers), first_child, n_edges,
        child_to_parent, ::Type{TF}) where {TF}
    i = @index(Global)
    @inbounds if i <= n_edges
        child = first_child + i - 1
        parent = parent_index[child]
        if child_to_parent
            dx = node_centers[1, parent] - node_centers[1, child]
            dy = node_centers[2, parent] - node_centers[2, child]
            dz = node_centers[3, parent] - node_centers[3, child]
            source_idx[i] = child
            target_idx[i] = parent
        else
            dx = node_centers[1, child] - node_centers[1, parent]
            dy = node_centers[2, child] - node_centers[2, parent]
            dz = node_centers[3, child] - node_centers[3, parent]
            source_idx[i] = parent
            target_idx[i] = child
        end
        x2y2 = dx * dx + dy * dy
        r2 = x2y2 + dz * dz
        eps2 = TF(1e-10) * TF(1e-10)
        r = sqrt(r2)
        theta = zero(TF)
        if r2 > eps2
            if x2y2 > eps2
                theta = acos(clamp(dz / r, -one(TF), one(TF)))
            else
                theta = TF(pi) * (dz < 0)
            end
        end
        phis[i] = iszero(x2y2) ? zero(TF) : atan(dy, dx)
        thetas[i] = theta
    end
end

function ka_refresh_group_edges!(group, grid, level_offsets::Vector{Int},
        child_level::Int, child_to_parent::Bool, kind::Symbol;
        workgroup=KA_AUTO_WORKGROUP)
    first_child = level_offsets[child_level + 1] + 1
    n_edges = level_offsets[child_level + 2] - level_offsets[child_level + 1]
    n_edges <= length(group.source_idx) || throw(AssertionError(
        "resident $kind group at child level $child_level exceeded its capacity"))
    group.count[] = n_edges
    n_edges > 0 || return group
    TF = eltype(group.phis)
    backend = KA.get_backend(group.phis)
    kernel = _cached_kernel(ka_refresh_group_edges_kernel!, backend, workgroup)
    kernel(group.source_idx, group.target_idx, group.phis, group.thetas,
        grid.parent_index, grid.node_centers, first_child, n_edges,
        child_to_parent, TF; ndrange=n_edges)
    return group
end

function ka_refresh_resident_stage_groups!(ws::FastMultipole.ResidentOperatorWorkspace,
        grid, level_offsets::Vector{Int}, ell::Int, first_level::Int=0;
        workgroup=KA_AUTO_WORKGROUP)
    length(ws.m2m_groups) == ell - first_level || throw(ArgumentError(
        "resident cache workspace does not match the trimmed level range"))
    for (gi, parent_level) in enumerate((ell - 1):-1:first_level)
        ka_refresh_group_edges!(ws.m2m_groups[gi], grid, level_offsets,
            parent_level + 1, true, :m2m; workgroup)
    end
    for (gi, child_level) in enumerate((first_level + 1):ell)
        ka_refresh_group_edges!(ws.l2l_groups[gi], grid, level_offsets,
            child_level, false, :l2l; workgroup)
    end
    return ws
end

#------- device source-position extraction (KA) -------#
#
# Elementwise gather of the xyz rows plus the (system, index) attribution of each
# body into the concatenated global order: no shared memory, `KA_AUTO_WORKGROUP`.
@kernel function ka_extract_source_positions_kernel!(positions, body_system,
        body_index, @Const(source_buffer), offset, isys, nb)
    i = @index(Global)
    @inbounds if i <= nb
        global_i = offset + i
        positions[1, global_i] = source_buffer[1, i]
        positions[2, global_i] = source_buffer[2, i]
        positions[3, global_i] = source_buffer[3, i]
        body_system[global_i] = isys
        body_index[global_i] = i
    end
end

# `source_buffers` are the per-system views `_radix_cache_refresh_source_buffers!`
# returns; the return value is the total body count.
function ka_collect_positions!(positions, body_system, body_index,
        source_buffers::Tuple; workgroup=KA_AUTO_WORKGROUP)
    offset = 0
    for isys in eachindex(source_buffers)
        buf = source_buffers[isys]
        nb = size(buf, 2)
        if nb > 0
            backend = KA.get_backend(positions)
            kernel = _cached_kernel(ka_extract_source_positions_kernel!, backend,
                workgroup)
            kernel(positions, body_system, body_index, buf, offset, isys, nb;
                ndrange=nb)
        end
        offset += nb
    end
    return offset
end

# `node_at` is zeroed at construction and refilled from the resident grid every
# time the occupied-node set changes; the window flag and compact kernels above
# read it.
# Elementwise scatter, no shared memory: `KA_AUTO_WORKGROUP`.
@kernel function ka_hier_node_at_scatter_kernel!(node_at, @Const(node_levels),
        @Const(node_coords), @Const(level_base), n_nodes)
    i = @index(Global)
    @inbounds if i <= n_nodes
        L = node_levels[i]
        G = 1 << L
        linear = node_coords[1, i] + G * (node_coords[2, i] + G * node_coords[3, i])
        node_at[level_base[L + 1] + linear + 1] = Int32(i)
    end
end

function ka_hier_refresh_occupancy!(hctx::FastMultipole.DeviceHierarchicalM2LContext,
        grid, level_offsets::Vector{Int}; workgroup=KA_AUTO_WORKGROUP)
    copyto!(hctx.level_offsets, level_offsets)
    n_nodes = level_offsets[end]
    n_nodes <= typemax(Int32) || throw(ArgumentError(
        "device hierarchical occupancy requires flat node indices to fit Int32; " *
        "got $n_nodes occupied nodes"))
    fill!(hctx.node_at, Int32(0))
    n_nodes == 0 && return hctx
    backend = KA.get_backend(hctx.node_at)
    kernel = _cached_kernel(ka_hier_node_at_scatter_kernel!, backend, workgroup)
    kernel(hctx.node_at, grid.node_levels, grid.node_coords, hctx.d_level_base,
        n_nodes; ndrange=n_nodes)
    return hctx
end

# Backend-generic builder of `DeviceHierarchicalM2LContext`, which is
# array-type generic: every buffer is a KA allocation on `backend`. It covers
# the concat plan.
function ka_hierarchical_context(backend, tables, level_class_of::Array{Int32,3},
        plan, ell::Int, first_m2l_level::Int, occupancy;
        window_classes::Int=typemax(Int))
    plan isa FastMultipole.ResidentM2LConcatPlan || throw(ArgumentError(
        "ka_hierarchical_context covers the ResidentM2LConcatPlan; got $(typeof(plan))"))
    isempty(occupancy.node_at) && throw(ArgumentError(
        "ka_hierarchical_context requires the dense per-level occupancy lookup"))
    noffsets = length(tables.push_offsets)
    K = max(min(window_classes, noffsets), 1)
    size(level_class_of) == (8, noffsets, ell + 1) || throw(ArgumentError(
        "invalid hierarchical per-level class table dimensions $(size(level_class_of))"))
    _dev(A) = KA.allocate(backend, eltype(A), size(A)...) |> d -> (copyto!(d, A); d)
    # 3 x n Int32 offsets matrix, inlined.
    _offsets_matrix(offsets) = (m = Matrix{Int32}(undef, 3, length(offsets));
        for (k, o) in enumerate(offsets); m[1, k] = Int32(o[1]);
            m[2, k] = Int32(o[2]); m[3, k] = Int32(o[3]); end; m)
    _zeros(T, n) = (z = KA.allocate(backend, T, n); fill!(z, zero(T)); z)
    return FastMultipole.DeviceHierarchicalM2LContext(
        plan, K, ell, first_m2l_level, noffsets,
        copy(occupancy.level_base), zeros(Int, ell + 2),
        _zeros(Int32, length(occupancy.node_at)),
        _dev(Vector{Int}(occupancy.level_base)),
        _dev(_offsets_matrix(tables.push_offsets)),
        _dev(level_class_of),
        _dev(_offsets_matrix(tables.near_offsets)),
        0,
        0, 0, false,
        nothing, nothing, nothing,
        nothing, nothing, zeros(Int32, 1),
    )
end

"""
    ka_hierarchical_m2l!(state, hctx, ws)

Hierarchical M2L on the device state. The occupancy-epoch window cache
(`ka_hier_cache_windows!`, run by `ka_update_radix_state!` on every occupancy
change) holds the whole route stream, so there is nothing to generate here and
the `(level, offset-class window)` loop collapses into a single concat apply
(the level rides in the class, not in an argument): the concat apply needs only
(class, source, target) and a count.

Concat plans only: the dense and factored strategies are host-only.
"""
function ka_hierarchical_m2l!(state::FastMultipole.DeviceResidentRadixState{TF,B,LH},
        hctx::FastMultipole.DeviceHierarchicalM2LContext, ws) where {TF,B,LH}
    plan = hctx.apply_plan
    plan isa FastMultipole.ResidentM2LConcatPlan || throw(ArgumentError(
        "ka_hierarchical_m2l! requires a ResidentM2LConcatPlan " *
        "(m2l_strategy = ConcatenatedFixedZM2L); got $(typeof(plan))"))
    hctx.win_valid || throw(AssertionError(
        "ka_hierarchical_m2l!: the M2L window cache is stale; ka_update_radix_state! " *
        "regenerates it on every occupancy change"))
    fill!(state.locals.phi, zero(TF))
    LH && fill!(state.locals.chi, zero(TF))
    n = hctx.total_routes
    state.counts.n_routes = n
    n == 0 && return state
    ka_resident_m2l_concat_apply!(state.locals, state.multipoles, ws,
        view(hctx.win_sources, 1:n), view(hctx.win_targets, 1:n), n;
        route_class=view(hctx.win_class, 1:n))
    return state
end
