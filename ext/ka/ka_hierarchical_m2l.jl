#------- hierarchical M2L window generation (KA) -------#
#
# KA port of `_cuda_hier_generate_window_core!`
# (src/translate_batched_cuda.jl:7350). This is the flag/scan/compact that fills
# `route_targets`/`route_sources`/`route_class` for one (level, offset-class)
# window. It was the last CUDA-only dependency inside `ka_hierarchical_m2l!`,
# and therefore the reason the hierarchical arm could not be gated on Metal.
#
# `DeviceHierarchicalM2LContext` is already array-type generic
# (containers.jl:782: IV32/IM32/IA32/IV/SM type parameters), so nothing here
# needs a CUDA-specific context mirror -- unlike the lifecycle, which needed
# `host_radix_state`. The scan reuses `_ka_scan_total!`'s `accumulate!`, which
# is backend-generic via GPUArrays.
#
# The three kernels are elementwise index math with no shared memory and no
# CUDA intrinsics, so they are direct translations and carry no `@localmem`
# team-size coupling: all three take `KA_AUTO_WORKGROUP`.

@kernel function ka_hier_route_flags_kernel!(flags, @Const(node_at),
        @Const(node_coords), @Const(push_offsets), @Const(class_of),
        level_base_L, first_source, n_sources, first_offset, kn, L)
    idx = @index(Global)
    @inbounds if idx <= kn * n_sources
        kloc = (idx - 1) ÷ n_sources + 1
        s = (idx - 1) % n_sources + 1
        k = first_offset + kloc - 1
        source = first_source + s - 1
        G = 1 << L
        cx = node_coords[1, source]
        cy = node_coords[2, source]
        cz = node_coords[3, source]
        # same x/y/z bit convention as _rigid_phase_index
        phase = 1 + (cx & 1) + 2 * (cy & 1) + 4 * (cz & 1)
        hit = Int32(0)
        if class_of[phase, k, L + 1] != Int32(0)
            tx = cx + push_offsets[1, k]
            ty = cy + push_offsets[2, k]
            tz = cz + push_offsets[3, k]
            if 0 <= tx < G && 0 <= ty < G && 0 <= tz < G
                linear = tx + G * (ty + G * tz)
                node_at[level_base_L + linear + 1] == Int32(0) || (hit = Int32(1))
            end
        end
        flags[idx] = hit
    end
end

# Per-class cumulative window counts read straight off the inclusive scan: class
# `kloc` ends at flat index `kloc * n_sources`.
@kernel function ka_hier_window_cum_kernel!(cum, @Const(prefix), n_sources, kn)
    i = @index(Global)
    @inbounds if i <= kn
        cum[i] = prefix[i * n_sources]
    end
end

# Compact one window into the start of the reusable route buffers. Offsets are
# the unscaled integer push offsets; endpoints are flat node indices.
# Base-offset variants for the concatenated-window scan in ka_hier_cache_windows!:
# every window's flags live at `base+1 : base+used` of one buffer, ONE inclusive
# scan runs over all of them, and a window's local prefix is
# prefix[base+idx] - prefix[base].
@kernel function ka_hier_window_cum_base_kernel!(cum, @Const(prefix), base, n_sources, kn)
    i = @index(Global)
    @inbounds if i <= kn
        pb = base > 0 ? prefix[base] : zero(eltype(prefix))
        cum[i] = prefix[base + i * n_sources] - pb
    end
end

@kernel function ka_hier_route_compact_base_kernel!(route_levels, route_offsets,
        route_targets, route_sources, route_class, @Const(flags), @Const(prefix), base,
        @Const(node_at), @Const(node_coords), @Const(push_offsets),
        level_base_L, first_source, n_sources, first_offset, kn, L, class_base)
    idx = @index(Global)
    @inbounds if idx <= kn * n_sources && flags[base + idx] == Int32(1)
        kloc = (idx - 1) ÷ n_sources + 1
        s = (idx - 1) % n_sources + 1
        k = first_offset + kloc - 1
        source = first_source + s - 1
        G = 1 << L
        ox = push_offsets[1, k]
        oy = push_offsets[2, k]
        oz = push_offsets[3, k]
        tx = node_coords[1, source] + ox
        ty = node_coords[2, source] + oy
        tz = node_coords[3, source] + oz
        linear = tx + G * (ty + G * tz)
        target = Int(node_at[level_base_L + linear + 1])
        pb = base > 0 ? prefix[base] : zero(eltype(prefix))
        p = Int(prefix[base + idx] - pb)
        route_levels[p] = L
        route_offsets[1, p] = Int(ox)
        route_offsets[2, p] = Int(oy)
        route_offsets[3, p] = Int(oz)
        route_targets[p] = target
        route_sources[p] = source
        route_class[p] = Int32(class_base + k)
    end
end

# Global-position variant for the concat window cache: with one scan over all
# windows, prefix[base+idx] IS the route's slot in the concatenated stream, so
# the compact writes the cache arrays directly -- no per-window staging, no
# device-to-device copies. Only the three arrays the concat apply reads.
@kernel function ka_hier_route_compact_global_kernel!(win_targets, win_sources, win_class,
        @Const(flags), @Const(prefix), base, @Const(node_at), @Const(node_coords),
        @Const(push_offsets), level_base_L, first_source, n_sources, first_offset, kn, L,
        class_base)
    idx = @index(Global)
    @inbounds if idx <= kn * n_sources && flags[base + idx] == Int32(1)
        kloc = (idx - 1) ÷ n_sources + 1
        s = (idx - 1) % n_sources + 1
        k = first_offset + kloc - 1
        source = first_source + s - 1
        G = 1 << L
        ox = push_offsets[1, k]
        oy = push_offsets[2, k]
        oz = push_offsets[3, k]
        tx = node_coords[1, source] + ox
        ty = node_coords[2, source] + oy
        tz = node_coords[3, source] + oz
        linear = tx + G * (ty + G * tz)
        p = Int(prefix[base + idx])
        win_targets[p] = Int(node_at[level_base_L + linear + 1])
        win_sources[p] = source
        win_class[p] = Int32(class_base + k)
    end
end

@kernel function ka_hier_route_compact_kernel!(route_levels, route_offsets,
        route_targets, route_sources, route_class, @Const(flags), @Const(prefix),
        @Const(node_at), @Const(node_coords), @Const(push_offsets),
        level_base_L, first_source, n_sources, first_offset, kn, L, class_base)
    idx = @index(Global)
    @inbounds if idx <= kn * n_sources && flags[idx] == Int32(1)
        kloc = (idx - 1) ÷ n_sources + 1
        s = (idx - 1) % n_sources + 1
        k = first_offset + kloc - 1
        source = first_source + s - 1
        G = 1 << L
        ox = push_offsets[1, k]
        oy = push_offsets[2, k]
        oz = push_offsets[3, k]
        tx = node_coords[1, source] + ox
        ty = node_coords[2, source] + oy
        tz = node_coords[3, source] + oz
        linear = tx + G * (ty + G * tz)
        target = Int(node_at[level_base_L + linear + 1])
        p = Int(prefix[idx])
        route_levels[p] = L
        route_offsets[1, p] = Int(ox)
        route_offsets[2, p] = Int(oy)
        route_offsets[3, p] = Int(oz)
        route_targets[p] = target
        route_sources[p] = source
        route_class[p] = Int32(class_base + k)
    end
end

#------- resident stage-group edge refresh (KA) -------#
#
# KA port of `_cuda_refresh_resident_stage_groups!`
# (src/translate_batched_cuda.jl:6120) and `_cuda_refresh_group_edges_kernel!`.
# Rebuilds the per-level M2M/L2L edge columns -- (source, target) node index
# pairs plus the spherical angles of the parent-child displacement -- from the
# refreshed grid, once per occupancy change inside `update_cuda_radix_state!`.
#
# Scope matches the CUDA function, not the host one: `_refresh_resident_stage_groups!`
# (translate_batched_resident.jl) also refills `ws.nonleaf_idx`, which is
# host-path-only storage (see the note at :3589) and untouched on device.
#
# `TF` is threaded in as a type argument rather than taken from `eltype(phis)`
# inside the kernel -- see [[reference-ka-localmem-eltype-metal]]; the group
# fields are `Any`-typed, so an in-kernel `eltype` is exactly the pattern that
# fails to resolve on Metal.
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
# KA port of `_cuda_extract_source_positions_kernel!`
# (src/translate_batched_cuda.jl:90) and its driver
# `_radix_cache_collect_positions!` (:6612). Unlike the three helpers beside it
# in the update path -- which were backend-agnostic code merely misfiled in the
# CUDA-only include and have been moved to translate_batched_resident.jl -- this
# one is a real kernel launch and needs a port.
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
# returns; the return value is the total body count, as on the CUDA side.
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

# KA port of `_cuda_hier_node_at_scatter_kernel!`
# (src/translate_batched_cuda.jl:7066) and its driver
# `_cuda_hier_refresh_occupancy!` (:7214). `node_at` is zeroed at construction on
# both backends and refilled from the resident grid every time the occupied-node
# set changes, so the window generator above cannot run off CUDA without it.
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
    @inbounds for level in 0:hctx.ell
        hctx.nodes_per_level[level + 1] =
            level_offsets[level + 2] - level_offsets[level + 1]
    end
    n_nodes == 0 && return hctx
    backend = KA.get_backend(hctx.node_at)
    kernel = _cached_kernel(ka_hier_node_at_scatter_kernel!, backend, workgroup)
    kernel(hctx.node_at, grid.node_levels, grid.node_coords, hctx.d_level_base,
        n_nodes; ndrange=n_nodes)
    return hctx
end

# Backend-generic mirror of `_build_cuda_hierarchical_context`
# (src/translate_batched_cuda.jl:8134). `DeviceHierarchicalM2LContext` is
# already array-type generic, so this is a pure array-type substitution: every
# `CUDA.zeros`/`CUDA.CuArray{T}` becomes a KA allocation on `backend`. It exists
# so `ka_hier_generate_window!` can be gated off CUDA; it covers the concat plan.
function ka_hierarchical_context(::Type{TF}, backend, tables, class_level,
        class_offset, effective_offsets, level_class_of::Array{Int32,3},
        level_radii2, plan, ell::Int, first_m2l_level::Int, max_level_nodes::Int,
        occupancy; window_classes::Int=typemax(Int), window_staging::Bool=true) where {TF}
    plan isa FastMultipole.ResidentM2LConcatPlan || throw(ArgumentError(
        "ka_hierarchical_context covers the ResidentM2LConcatPlan; got $(typeof(plan))"))
    isempty(occupancy.node_at) && throw(ArgumentError(
        "ka_hierarchical_context requires the dense per-level occupancy lookup"))
    noffsets = length(tables.push_offsets)
    K = max(min(window_classes, noffsets), 1)
    # Per-window flag/prefix scratch (K x max_level_nodes Int32 each) is only read
    # by the per-window generators; with the concat plan and cached windows
    # (`ka_hier_cache_windows!` allocates its own epoch-sized scratch) it is dead,
    # so `window_staging=false` shrinks it to one entry. At ell 5 that is 67 MB,
    # 8x per level (device memory accounting).
    flag_capacity = window_staging ? max(K * max_level_nodes, 1) : 1
    size(level_class_of) == (8, noffsets, ell + 1) || throw(ArgumentError(
        "invalid hierarchical per-level class table dimensions $(size(level_class_of))"))
    _dev(A) = KA.allocate(backend, eltype(A), size(A)...) |> d -> (copyto!(d, A); d)
    # `_radix_offsets_matrix` lives in translate_batched_cuda.jl, which is
    # `include`d only when CUDA is available -- calling it here would make this
    # builder CUDA-only at run time. Same five lines, inlined.
    _offsets_matrix(offsets) = (m = Matrix{Int32}(undef, 3, length(offsets));
        for (k, o) in enumerate(offsets); m[1, k] = Int32(o[1]);
            m[2, k] = Int32(o[2]); m[3, k] = Int32(o[3]); end; m)
    _zeros(T, n) = (z = KA.allocate(backend, T, n); fill!(z, zero(T)); z)
    # the concat plan reads no per-level scale column
    empty_scale = KA.allocate(backend, TF, 0, 0)
    src_scale, tgt_scale = (empty_scale, empty_scale)
    return FastMultipole.DeviceHierarchicalM2LContext(
        tables, level_radii2, class_level, class_offset, effective_offsets, plan,
        K, ell, first_m2l_level, noffsets,
        copy(occupancy.level_base), zeros(Int, ell + 2),
        _zeros(Int32, length(occupancy.node_at)),
        _dev(Vector{Int}(occupancy.level_base)),
        _dev(_offsets_matrix(tables.push_offsets)),
        _dev(level_class_of),
        _dev(_offsets_matrix(tables.near_offsets)),
        _zeros(Int, 0), _zeros(Int, 0),
        _zeros(Int32, flag_capacity), _zeros(Int32, flag_capacity),
        _zeros(Int32, max(K, 1)), zeros(Int32, max(K, 1)),
        src_scale, tgt_scale,
        0, zeros(Int, ell + 1), zeros(Int, ell + 1), 0, 0, 1, 0,
        false, zeros(UInt64, 5), zeros(UInt64, ell + 1),
        0, 0, false, zeros(Int, ell + 2), zeros(Int, ell + 2),
        nothing, nothing, nothing, nothing, -1, -1,
        nothing,
    )
end

function ka_hier_generate_window!(state::FastMultipole.DeviceResidentRadixState,
        hctx::FastMultipole.DeviceHierarchicalM2LContext, route_class, L::Int,
        first_offset::Int, last_offset::Int, class_base::Int;
        workgroup=KA_AUTO_WORKGROUP)
    return ka_hier_generate_window_core!(state.route_levels, state.route_offsets,
        state.route_targets, state.route_sources, state.grid, hctx, route_class,
        L, first_offset, last_offset, class_base; workgroup)
end

function ka_hier_generate_window_core!(route_levels, route_offsets, route_targets,
        route_sources, grid, hctx::FastMultipole.DeviceHierarchicalM2LContext,
        route_class, L::Int, first_offset::Int, last_offset::Int, class_base::Int;
        workgroup=KA_AUTO_WORKGROUP)
    first_source = hctx.level_offsets[L + 1] + 1
    n_sources = hctx.level_offsets[L + 2] - hctx.level_offsets[L + 1]
    kn = last_offset - first_offset + 1
    (n_sources > 0 && kn > 0) || return 0
    used = kn * n_sources
    used <= length(hctx.route_flags) || throw(AssertionError(
        "device hierarchical window flag buffer exceeded its capacity " *
        "($(length(hctx.route_flags)) < $used); reduce window_classes, or, if the " *
        "cache was built with the concat plan and cached windows (staging size 1), " *
        "set :CUDA_CACHED_WINDOWS=false BEFORE building the cache"))
    backend = KA.get_backend(hctx.route_flags)
    level_base_L = hctx.level_base[L + 1]

    flags_kernel = _cached_kernel(ka_hier_route_flags_kernel!, backend, workgroup)
    flags_kernel(hctx.route_flags, hctx.node_at, grid.node_coords,
        hctx.d_push_offsets, hctx.d_class_of, level_base_L, first_source,
        n_sources, first_offset, kn, L; ndrange=used)

    # Inclusive scan over the used prefix, then the per-class cumulative counts.
    accumulate!(+, view(hctx.route_prefix, 1:used), view(hctx.route_flags, 1:used))

    cum_kernel = _cached_kernel(ka_hier_window_cum_kernel!, backend, workgroup)
    cum_kernel(hctx.window_cum, hctx.route_prefix, n_sources, kn; ndrange=kn)

    # The `kn`-entry D2H is the one unavoidable sync point per window: the
    # compact launch needs `n_routes` on the host to bounds-check the route
    # buffers, exactly as the CUDA core does.
    KA.synchronize(backend)
    copyto!(hctx.host_window_cum, 1, hctx.window_cum, 1, kn)
    n_routes = Int(hctx.host_window_cum[kn])
    n_routes == 0 && return 0
    n_routes <= length(route_targets) || throw(AssertionError(
        "device hierarchical route window exceeded capacity " *
        "$(length(route_targets)); increase window storage or reduce window_classes " *
        "(size 1 means the cache was built for cached concat windows; set " *
        ":CUDA_CACHED_WINDOWS=false before building it to use per-window generation)"))

    compact_kernel = _cached_kernel(ka_hier_route_compact_kernel!, backend, workgroup)
    compact_kernel(route_levels, route_offsets, route_targets, route_sources,
        route_class, hctx.route_flags, hctx.route_prefix, hctx.node_at,
        grid.node_coords, hctx.d_push_offsets, level_base_L, first_source,
        n_sources, first_offset, kn, L, class_base; ndrange=used)
    return n_routes
end
# Two-phase window generation for the occupancy-epoch cache: `..._count!`
# runs flags/scan/cum for one window and stashes its route total on the
# device; the caller syncs ONCE for all windows, then `..._compact!` reruns
# the (cheap) flags/scan and compacts with the count already on the host. The
# single-window core above syncs per window -- one host readback per (level,
# offset class) -- which on a moving field runs every step; measured 39% of
# the step at np=16k on Metal, where a sync costs ~200 us.
function ka_hier_generate_window_count!(grid, hctx::FastMultipole.DeviceHierarchicalM2LContext,
        L::Int, first_offset::Int, last_offset::Int, win_totals, w::Int;
        workgroup=KA_AUTO_WORKGROUP)
    first_source = hctx.level_offsets[L + 1] + 1
    n_sources = hctx.level_offsets[L + 2] - hctx.level_offsets[L + 1]
    kn = last_offset - first_offset + 1
    (n_sources > 0 && kn > 0) || return nothing
    used = kn * n_sources
    used <= length(hctx.route_flags) || throw(AssertionError(
        "device hierarchical window flag buffer exceeded its capacity " *
        "($(length(hctx.route_flags)) < $used); reduce window_classes, or, if the " *
        "cache was built with the concat plan and cached windows (staging size 1), " *
        "set :CUDA_CACHED_WINDOWS=false BEFORE building the cache"))
    backend = KA.get_backend(hctx.route_flags)
    level_base_L = hctx.level_base[L + 1]
    flags_kernel = _cached_kernel(ka_hier_route_flags_kernel!, backend, workgroup)
    flags_kernel(hctx.route_flags, hctx.node_at, grid.node_coords,
        hctx.d_push_offsets, hctx.d_class_of, level_base_L, first_source,
        n_sources, first_offset, kn, L; ndrange=used)
    accumulate!(+, view(hctx.route_prefix, 1:used), view(hctx.route_flags, 1:used))
    cum_kernel = _cached_kernel(ka_hier_window_cum_kernel!, backend, workgroup)
    cum_kernel(hctx.window_cum, hctx.route_prefix, n_sources, kn; ndrange=kn)
    copyto!(win_totals, w, hctx.window_cum, kn, 1)   # device -> device, no sync
    return nothing
end

function ka_hier_generate_window_compact!(route_levels, route_offsets, route_targets,
        route_sources, grid, hctx::FastMultipole.DeviceHierarchicalM2LContext,
        route_class, L::Int, first_offset::Int, last_offset::Int, class_base::Int,
        n_routes::Int; workgroup=KA_AUTO_WORKGROUP)
    n_routes == 0 && return 0
    n_routes <= length(route_targets) || throw(AssertionError(
        "device hierarchical route window exceeded capacity " *
        "$(length(route_targets)); increase window storage or reduce window_classes " *
        "(size 1 means the cache was built for cached concat windows; set " *
        ":CUDA_CACHED_WINDOWS=false before building it to use per-window generation)"))
    first_source = hctx.level_offsets[L + 1] + 1
    n_sources = hctx.level_offsets[L + 2] - hctx.level_offsets[L + 1]
    kn = last_offset - first_offset + 1
    used = kn * n_sources
    backend = KA.get_backend(hctx.route_flags)
    level_base_L = hctx.level_base[L + 1]
    flags_kernel = _cached_kernel(ka_hier_route_flags_kernel!, backend, workgroup)
    flags_kernel(hctx.route_flags, hctx.node_at, grid.node_coords,
        hctx.d_push_offsets, hctx.d_class_of, level_base_L, first_source,
        n_sources, first_offset, kn, L; ndrange=used)
    accumulate!(+, view(hctx.route_prefix, 1:used), view(hctx.route_flags, 1:used))
    compact_kernel = _cached_kernel(ka_hier_route_compact_kernel!, backend, workgroup)
    compact_kernel(route_levels, route_offsets, route_targets, route_sources,
        route_class, hctx.route_flags, hctx.route_prefix, hctx.node_at,
        grid.node_coords, hctx.d_push_offsets, level_base_L, first_source,
        n_sources, first_offset, kn, L, class_base; ndrange=used)
    return n_routes
end

"""
    ka_hierarchical_m2l!(state, hctx, ws)

KA arm of `_launch_cuda_hierarchical_m2l!`: the same `(level, offset-class
window)` loop nest, with the per-window concat apply run by KA kernels instead
of `_launch_resident_m2l_concat!`.

**Window generation is now KA too.** `ka_hier_generate_window!` (above) is the
flag/scan/compact that fills `state.route_targets`/`route_sources` for one
window. It replaced the shared `_cuda_hier_generate_window!` call, so this
driver no longer reaches any `_cuda_*` function and the arm is backend-generic.

Historical note for reading older benchmarks: before the KA port the generator
was shared-native, which made every timing comparison built on this driver an
A/B of the M2L *apply* only (the rotation/translation math) and not of the
route bookkeeping. Job 13511158 and earlier numbers were produced under that
regime and must still be read that way.

Concat plans only: the dense and factored strategies are host-only.

`DeviceHierarchicalM2LContext` is array-type generic (containers.jl:782), so
with the generator ported this function is reachable on any KA backend and is
gated on Metal.
"""
function ka_hierarchical_m2l!(state::FastMultipole.DeviceResidentRadixState{TF,B,LH},
        hctx::FastMultipole.DeviceHierarchicalM2LContext, ws) where {TF,B,LH}
    plan = hctx.apply_plan
    plan isa FastMultipole.ResidentM2LConcatPlan || throw(ArgumentError(
        "ka_hierarchical_m2l! requires a ResidentM2LConcatPlan " *
        "(m2l_strategy = ConcatenatedFixedZM2L); got $(typeof(plan))"))
    # Steady state on the concat plan: the occupancy-epoch cache holds the whole
    # route stream, so there is nothing to generate and the level loop collapses
    # into a single apply (the level rides in the class, not in an argument).
    if hctx.win_valid && _ka_radix_setting(:CUDA_CACHED_WINDOWS, true)
        return ka_hierarchical_m2l_cached_concat!(state, hctx, ws)
    end
    fill!(state.locals.phi, zero(TF))
    LH && fill!(state.locals.chi, zero(TF))
    route_class = plan.route_class
    noffsets = hctx.noffsets
    K = hctx.window_classes
    total = 0
    fill!(hctx.routes_per_level, 0)
    for L in hctx.first_m2l_level:hctx.ell
        level_total = 0
        # dense classes are the unscaled push offsets, level enters through the
        # scale column only, so the level component drops out (cuda:7956)
        class_base = (L - hctx.first_m2l_level) * noffsets
        for first_offset in 1:K:noffsets
            last_offset = min(first_offset + K - 1, noffsets)
            n = ka_hier_generate_window!(state, hctx, route_class, L,
                first_offset, last_offset, class_base)
            hctx.last_window_routes = n
            state.counts.n_routes = n
            if n > 0
                ka_resident_m2l_concat_apply!(state.locals, state.multipoles, ws,
                    state.route_sources, state.route_targets, n)
            end
            level_total += n
        end
        hctx.routes_per_level[L + 1] = level_total
        total += level_total
    end
    hctx.total_routes = total
    state.counts.n_routes = total
    return state
end

# The cached counterpart of the loop above, for the concat plan only. KA-only:
# CUDA's cached path (`_launch_cuda_hierarchical_m2l_cached!`) is dense-fused,
# because its dense GEMM reference driver needs per-window class starts. The
# concat apply needs none of that -- (class, source, target) and a count is its
# entire input -- so the cached stream can be applied in one call, with no
# route generation, no per-window prefix D2H, and no per-window sync.
#
# Gated against the uncached loop, not against CUDA: CUDA has no concat cache
# to compare with. Route order is identical either way (the cache concatenates
# the same windows in the same order), so the two arms agree to within the
# reassociation of a longer chunk sequence.
function ka_hierarchical_m2l_cached_concat!(
        state::FastMultipole.DeviceResidentRadixState{TF,B,LH},
        hctx::FastMultipole.DeviceHierarchicalM2LContext, ws) where {TF,B,LH}
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

