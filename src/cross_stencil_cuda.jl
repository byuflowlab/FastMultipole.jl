#------- CROSS-PASS DEVICE PRODUCERS (task 052d Stage A) -------#
#
# Opt-in device code: included by load_cuda_radix_lifecycle!() after
# translate_batched_cuda.jl (whose CUDA/intrinsic bindings it shares). Builds,
# entirely on device each step (D1 device-native ruling, 2026-08-28), the
# two-occupancy cross-pass producer outputs the panels → particles pass
# consumes: per-level occupied-cell sets for BOTH body sets on the shared
# radix grid, the particle-side dense occupancy lookup, the cross permutation
# (perm at ell_x), and the three lists — M2L routes (far), guard-demoted
# direct blocks, and leaf near blocks.
#
# Kernel reuse: the hierarchical route kernels take the target occupancy
# (`node_at`) and the source coordinates (`node_coords`) as separate
# arguments, so the two-occupancy pass reuses `_cuda_hier_route_flags_kernel!`
# verbatim by passing PANEL coords with PARTICLE occupancy. Only two small
# kernels are new: node-range fills with a level offset, and a compact with a
# running output base (the self-pass compacts one window at a time; the cross
# pass concatenates all windows into persistent lists).
#
# v1 deviations (recorded in the plan doc): the particle sort at ell_x is a
# full-key CUDA.sortperm! rather than the segmented low-bits LSD sort — the
# sorted result is identical (Morton high bits are the ell prefix); the
# segmented kernel is a later optimization measured against this baseline.

"Per-level occupied-cell arrays of ONE body set on the shared grid."
mutable struct DeviceCrossOccupancy{TF}
    n::Int
    d_keys::CUDA.CuVector{UInt64}
    d_perm::CUDA.CuVector{Int}
    d_sorted_keys::CUDA.CuVector{UInt64}
    d_shift_keys::CUDA.CuVector{UInt64}
    d_flags::CUDA.CuVector{Int32}
    d_prefix::CUDA.CuVector{Int32}
    max_nodes::Int
    d_node_keys::CUDA.CuVector{UInt64}
    d_node_levels::CUDA.CuVector{Int32}
    d_node_coords::CUDA.CuMatrix{Int32}
    d_node_centers::CUDA.CuMatrix{TF}
    d_node_ranges::CUDA.CuMatrix{Int32}   # (first, count) in sorted body order
    d_parent_index::CUDA.CuVector{Int32}  # flat parent node (0 at the root level)
    d_body_node::CUDA.CuVector{Int32}     # leaf node of each sorted body
    level_offsets::Vector{Int}            # [L + 1] = nodes before level L
    d_level_offsets::CUDA.CuVector{Int}
    host_scalar32::Vector{Int32}
end

function DeviceCrossOccupancy{TF}(n::Int, ell_x::Int) where {TF}
    max_nodes = sum(min(1 << (3 * L), n) for L in 0:ell_x)
    return DeviceCrossOccupancy{TF}(n,
        CUDA.CuVector{UInt64}(undef, n),
        CUDA.CuVector{Int}(undef, n),
        CUDA.CuVector{UInt64}(undef, n),
        CUDA.CuVector{UInt64}(undef, n),
        CUDA.CuVector{Int32}(undef, n),
        CUDA.CuVector{Int32}(undef, n),
        max_nodes,
        CUDA.CuVector{UInt64}(undef, max_nodes),
        CUDA.CuVector{Int32}(undef, max_nodes),
        CUDA.CuMatrix{Int32}(undef, 3, max_nodes),
        CUDA.CuMatrix{TF}(undef, 3, max_nodes),
        CUDA.CuMatrix{Int32}(undef, 2, max_nodes),
        CUDA.CuVector{Int32}(undef, max_nodes),
        CUDA.CuVector{Int32}(undef, n),
        zeros(Int, ell_x + 2),
        CUDA.CuVector{Int}(undef, ell_x + 2),
        zeros(Int32, 1))
end

"""
Device cross-pass producer state: grid, uploaded stencil tables, panel and
particle occupancies, and the persistent list outputs. `refresh_cross_producers!`
rebuilds everything each step from current positions.
"""
mutable struct DeviceCrossProducerContext{TF}
    x_min::SVector{3,TF}
    h0::TF
    box_extent::SVector{3,TF}
    ell_x::Int
    ct::CrossStencilTables
    # uploaded once at construction
    d_push_offsets::CUDA.CuMatrix{Int32}
    d_near_offsets::CUDA.CuMatrix{Int32}
    d_class_far::CUDA.CuArray{Int32,3}
    d_class_demoted::CUDA.CuArray{Int32,3}
    d_near_class::CUDA.CuArray{Int32,3}
    # occupancies
    panels::DeviceCrossOccupancy{TF}
    particles::DeviceCrossOccupancy{TF}
    level_base::Vector{Int}
    d_level_base::CUDA.CuVector{Int}
    d_node_at::CUDA.CuVector{Int32}       # particle-side dense occupancy
    d_oob::CUDA.CuVector{Int32}
    # M2L route list (far)
    route_capacity::Int
    d_route_levels::CUDA.CuVector{Int}
    d_route_offsets::CUDA.CuMatrix{Int}
    d_route_targets::CUDA.CuVector{Int}
    d_route_sources::CUDA.CuVector{Int}
    d_route_class::CUDA.CuVector{Int32}
    n_routes::Int
    # direct blocks (guard-demoted routes ++ leaf near blocks)
    block_capacity::Int
    d_block_levels::CUDA.CuVector{Int}
    d_block_offsets::CUDA.CuMatrix{Int}
    d_block_targets::CUDA.CuVector{Int}
    d_block_sources::CUDA.CuVector{Int}
    d_block_class::CUDA.CuVector{Int32}
    n_demoted::Int
    n_blocks::Int
    # window scratch
    window_capacity::Int
    d_flags_w::CUDA.CuVector{Int32}
    d_prefix_w::CUDA.CuVector{Int32}
    host_scalar32::Vector{Int32}
    # 052h reverse leg (particles→panels): panel-side dense occupancy (panels
    # as route TARGETS) and the reversed route/block lists. All 0-length and
    # inert unless constructed with `build_reverse = true` — the forward-only
    # user pays nothing.
    build_reverse::Bool
    d_node_at_panels::CUDA.CuVector{Int32}
    rev_route_capacity::Int
    d_rev_route_levels::CUDA.CuVector{Int}
    d_rev_route_offsets::CUDA.CuMatrix{Int}
    d_rev_route_targets::CUDA.CuVector{Int}
    d_rev_route_sources::CUDA.CuVector{Int}
    d_rev_route_class::CUDA.CuVector{Int32}
    n_rev_routes::Int
    rev_block_capacity::Int
    d_rev_block_levels::CUDA.CuVector{Int}
    d_rev_block_offsets::CUDA.CuMatrix{Int}
    d_rev_block_targets::CUDA.CuVector{Int}
    d_rev_block_sources::CUDA.CuVector{Int}
    d_rev_block_class::CUDA.CuVector{Int32}
    n_rev_demoted::Int
    n_rev_blocks::Int
    # containment-failure signal: refresh_cross_producers! does NOT mutate the
    # grid in place (frozen h0 keys the cached operator tables AND the stencil
    # demotion masks); it reports the required union box here and the caller
    # must rebuild the full state (tables, masks, operators) around it.
    needs_rebuild::Bool
    rebuild_x_min::SVector{3,TF}
    rebuild_h0::TF
end

function device_cross_producer_context(ct::CrossStencilTables,
        x_min::SVector{3,TF}, h0::TF, n_panels::Int, n_particles::Int;
        route_capacity::Int=1 << 20, block_capacity::Int=1 << 20,
        window_capacity::Int=1 << 22, build_reverse::Bool=false,
        rev_route_capacity::Int=route_capacity,
        rev_block_capacity::Int=block_capacity) where {TF}
    ell_x = ct.ell_x
    ell_x <= 8 || throw(ArgumentError(
        "dense particle occupancy requires ell_x <= 8 (got $ell_x); the sorted " *
        "binary-search fallback of the p33 memo §3 is not yet ported"))
    level_base = zeros(Int, ell_x + 2)
    for L in 0:ell_x
        level_base[L + 2] = level_base[L + 1] + (1 << (3 * L))
    end
    return DeviceCrossProducerContext{TF}(x_min, h0,
        SVector{3,TF}(2h0, 2h0, 2h0), ell_x, ct,
        CUDA.CuArray{Int32}(_radix_offsets_matrix(ct.tables.push_offsets)),
        CUDA.CuArray{Int32}(_radix_offsets_matrix(ct.tables.near_offsets)),
        CUDA.CuArray{Int32}(ct.level_class_far),
        CUDA.CuArray{Int32}(ct.level_class_demoted),
        CUDA.CuArray{Int32}(ct.near_class),
        DeviceCrossOccupancy{TF}(n_panels, ell_x),
        DeviceCrossOccupancy{TF}(n_particles, ell_x),
        level_base,
        CUDA.CuArray{Int}(level_base),
        CUDA.zeros(Int32, level_base[end]),
        CUDA.zeros(Int32, 1),
        route_capacity,
        CUDA.CuVector{Int}(undef, route_capacity),
        CUDA.CuMatrix{Int}(undef, 3, route_capacity),
        CUDA.CuVector{Int}(undef, route_capacity),
        CUDA.CuVector{Int}(undef, route_capacity),
        CUDA.CuVector{Int32}(undef, route_capacity),
        0,
        block_capacity,
        CUDA.CuVector{Int}(undef, block_capacity),
        CUDA.CuMatrix{Int}(undef, 3, block_capacity),
        CUDA.CuVector{Int}(undef, block_capacity),
        CUDA.CuVector{Int}(undef, block_capacity),
        CUDA.CuVector{Int32}(undef, block_capacity),
        0, 0,
        window_capacity,
        CUDA.CuVector{Int32}(undef, window_capacity),
        CUDA.CuVector{Int32}(undef, window_capacity),
        zeros(Int32, 1),
        build_reverse,
        build_reverse ? CUDA.zeros(Int32, level_base[end]) :
            CUDA.CuVector{Int32}(undef, 0),
        build_reverse ? rev_route_capacity : 0,
        CUDA.CuVector{Int}(undef, build_reverse ? rev_route_capacity : 0),
        CUDA.CuMatrix{Int}(undef, 3, build_reverse ? rev_route_capacity : 0),
        CUDA.CuVector{Int}(undef, build_reverse ? rev_route_capacity : 0),
        CUDA.CuVector{Int}(undef, build_reverse ? rev_route_capacity : 0),
        CUDA.CuVector{Int32}(undef, build_reverse ? rev_route_capacity : 0),
        0,
        build_reverse ? rev_block_capacity : 0,
        CUDA.CuVector{Int}(undef, build_reverse ? rev_block_capacity : 0),
        CUDA.CuMatrix{Int}(undef, 3, build_reverse ? rev_block_capacity : 0),
        CUDA.CuVector{Int}(undef, build_reverse ? rev_block_capacity : 0),
        CUDA.CuVector{Int}(undef, build_reverse ? rev_block_capacity : 0),
        CUDA.CuVector{Int32}(undef, build_reverse ? rev_block_capacity : 0),
        0, 0,
        false, x_min, h0)
end

#------- new kernels (node ranges with offset; compact with base) -------#

function _cross_node_first_kernel!(node_ranges, flags, prefix, node_offset, n)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > n && return nothing
    @inbounds if flags[i] == Int32(1)
        node = node_offset + prefix[i]
        node_ranges[1, node] = Int32(i)
    end
    return nothing
end

function _cross_node_count_kernel!(node_ranges, flags, prefix, node_offset, n)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > n && return nothing
    # firsts are written by the prior launch; the launch boundary synchronizes
    @inbounds if i == n || flags[i + 1] == Int32(1)
        node = node_offset + prefix[i]
        node_ranges[2, node] = Int32(i) - node_ranges[1, node] + Int32(1)
    end
    return nothing
end

# `_cuda_hier_route_compact_kernel!` with a running output base, writing plain
# offset ids as the class (the cross pass owns its operator tables; Stage C
# assigns class bases when it builds them).
function _cross_route_compact_kernel!(route_levels, route_offsets, route_targets,
        route_sources, route_class, flags, prefix, node_at, node_coords,
        push_offsets, level_base_L, first_source, n_sources, first_offset, kn, L,
        base)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    idx > kn * n_sources && return nothing
    @inbounds begin
        flags[idx] == Int32(1) || return nothing
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
        p = base + Int(prefix[idx])
        route_levels[p] = L
        route_offsets[1, p] = Int(ox)
        route_offsets[2, p] = Int(oy)
        route_offsets[3, p] = Int(oz)
        route_targets[p] = target
        route_sources[p] = source
        route_class[p] = Int32(k)
    end
    return nothing
end

#------- per-step refresh -------#

function _cross_compute_keys!(ctx::DeviceCrossProducerContext,
        occ::DeviceCrossOccupancy, d_positions)
    n = occ.n
    size(d_positions, 2) == n || throw(ArgumentError(
        "cross producer positions have $(size(d_positions, 2)) columns; expected $n"))
    threads = 256
    blocks = cld(n, threads)
    CUDA.@cuda threads=threads blocks=blocks _cuda_radix_keys_checked_kernel!(
        occ.d_keys, ctx.d_oob, d_positions, ctx.x_min, ctx.box_extent, ctx.h0,
        ctx.ell_x)
    return nothing
end

"Cubic union root box over both position sets (containment-failure fallback)."
function _cross_union_root_box(ctx::DeviceCrossProducerContext{TF},
        d_panel_positions, d_particle_positions) where {TF}
    lo = MVector{3,TF}(undef)
    hi = MVector{3,TF}(undef)
    for a in 1:3
        lo[a] = min(minimum(view(d_panel_positions, a, :)),
            minimum(view(d_particle_positions, a, :)))
        hi[a] = max(maximum(view(d_panel_positions, a, :)),
            maximum(view(d_particle_positions, a, :)))
    end
    center = SVector{3,TF}((lo .+ hi) ./ 2)
    h0 = maximum(hi .- lo) / 2
    return center .- h0, h0
end

function _cross_build_levels!(occ::DeviceCrossOccupancy, ell_x::Int)
    n = occ.n
    threads = 256
    blocks = cld(n, threads)
    sortperm!(occ.d_perm, occ.d_keys)
    CUDA.@cuda threads=threads blocks=blocks _cuda_gather_sorted_keys_kernel!(
        occ.d_sorted_keys, occ.d_keys, occ.d_perm)
    occ.level_offsets[1] = 0
    for L in 0:ell_x
        CUDA.@cuda threads=threads blocks=blocks _cuda_leaf_ancestor_keys_kernel!(
            occ.d_shift_keys, occ.d_sorted_keys, ell_x, L)
        CUDA.@cuda threads=threads blocks=blocks _cuda_key_change_flags_kernel!(
            occ.d_flags, occ.d_shift_keys)
        accumulate!(+, occ.d_prefix, occ.d_flags)
        copyto!(occ.host_scalar32, 1, occ.d_prefix, n, 1)
        count = Int(occ.host_scalar32[1])
        offset = occ.level_offsets[L + 1]
        offset + count <= occ.max_nodes || throw(AssertionError(
            "cross occupancy node buffer exceeded capacity $(occ.max_nodes)"))
        CUDA.@cuda threads=threads blocks=blocks _cuda_fill_unique_keys_kernel!(
            occ.d_node_keys, occ.d_shift_keys, occ.d_flags, occ.d_prefix, offset)
        CUDA.@cuda threads=threads blocks=blocks _cross_node_first_kernel!(
            occ.d_node_ranges, occ.d_flags, occ.d_prefix, offset, n)
        CUDA.@cuda threads=threads blocks=blocks _cross_node_count_kernel!(
            occ.d_node_ranges, occ.d_flags, occ.d_prefix, offset, n)
        occ.level_offsets[L + 2] = offset + count
    end
    return nothing
end

function _cross_node_geometry!(occ::DeviceCrossOccupancy,
        x_min::SVector{3}, h0, ell_x::Int)
    copyto!(occ.d_level_offsets, occ.level_offsets)
    max_per_level = maximum(occ.level_offsets[L + 2] - occ.level_offsets[L + 1]
        for L in 0:ell_x)
    max_per_level == 0 && return nothing
    threads = 256
    blocks = (cld(max_per_level, threads), ell_x + 1)
    CUDA.@cuda threads=threads blocks=blocks _cuda_fill_node_geometry_kernel!(
        occ.d_node_levels, occ.d_node_coords, occ.d_node_centers, occ.d_node_keys,
        occ.d_level_offsets, x_min, h0, ell_x, 0)
    return nothing
end

"""
Append one (level, list-kind) sweep to an output list; returns the new count.
`src` supplies the explicit SOURCE node set and `d_node_at` the dense TARGET
occupancy: `(ctx.panels, ctx.d_node_at)` for the forward (panels→particles)
direction, `(ctx.particles, ctx.d_node_at_panels)` for the 052h reverse leg.
"""
function _cross_generate_level!(ctx::DeviceCrossProducerContext, L::Int,
        src::DeviceCrossOccupancy, d_node_at,
        d_offsets, d_class, out_levels, out_offsets, out_targets, out_sources,
        out_class, capacity::Int, base::Int, what::String)
    first_source = src.level_offsets[L + 1] + 1
    n_sources = src.level_offsets[L + 2] - src.level_offsets[L + 1]
    K = size(d_offsets, 2)
    (n_sources > 0 && K > 0) || return base
    threads = 256
    level_base_L = ctx.level_base[L + 1]
    kn_chunk = clamp(ctx.window_capacity ÷ n_sources, 1, K)
    first_offset = 1
    while first_offset <= K
        kn = min(kn_chunk, K - first_offset + 1)
        used = kn * n_sources
        used <= ctx.window_capacity || throw(AssertionError(
            "cross window scratch exceeded ($used > $(ctx.window_capacity))"))
        blocks = cld(used, threads)
        CUDA.@cuda threads=threads blocks=blocks _cuda_hier_route_flags_kernel!(
            ctx.d_flags_w, d_node_at, src.d_node_coords, d_offsets, d_class,
            level_base_L, first_source, n_sources, first_offset, kn, L)
        fv = view(ctx.d_flags_w, 1:used)
        pv = view(ctx.d_prefix_w, 1:used)
        accumulate!(+, pv, fv)
        copyto!(ctx.host_scalar32, 1, ctx.d_prefix_w, used, 1)
        chunk = Int(ctx.host_scalar32[1])
        if chunk > 0
            base + chunk <= capacity || throw(AssertionError(
                "cross $what list exceeded capacity $capacity"))
            CUDA.@cuda threads=threads blocks=blocks _cross_route_compact_kernel!(
                out_levels, out_offsets, out_targets, out_sources, out_class,
                ctx.d_flags_w, ctx.d_prefix_w, d_node_at, src.d_node_coords,
                d_offsets, level_base_L, first_source, n_sources, first_offset,
                kn, L, base)
            base += chunk
        end
        first_offset += kn
    end
    return base
end

"""
    refresh_cross_producers!(ctx, d_panel_positions, d_particle_positions)

Rebuild all cross-pass producer state from current positions (both `3 × n`
device matrices). A body outside the frozen root box does NOT mutate the grid
in place (the stencil demotion masks and cached operator tables are keyed on
`h0`; a partial refresh would misroute far/near work): the refresh sets
`ctx.needs_rebuild = true` with the required cubic union root box in
`ctx.rebuild_x_min` / `ctx.rebuild_h0`, empties the lists, and returns. The
caller must then reconstruct the FULL cross-pass state (stencil tables,
producer context, operator tables) around a box containing the new positions
and refresh again.
"""
function refresh_cross_producers!(ctx::DeviceCrossProducerContext{TF},
        d_panel_positions, d_particle_positions) where {TF}
    ctx.needs_rebuild = false
    fill!(ctx.d_oob, Int32(0))
    _cross_compute_keys!(ctx, ctx.panels, d_panel_positions)
    _cross_compute_keys!(ctx, ctx.particles, d_particle_positions)
    copyto!(ctx.host_scalar32, 1, ctx.d_oob, 1, 1)
    if ctx.host_scalar32[1] != Int32(0)
        x_min, h0 = _cross_union_root_box(ctx, d_panel_positions,
            d_particle_positions)
        ctx.needs_rebuild = true
        ctx.rebuild_x_min = x_min
        ctx.rebuild_h0 = h0
        ctx.n_routes = 0
        ctx.n_demoted = 0
        ctx.n_blocks = 0
        ctx.n_rev_routes = 0
        ctx.n_rev_demoted = 0
        ctx.n_rev_blocks = 0
        return ctx
    end
    ell_x = ctx.ell_x
    _cross_build_levels!(ctx.panels, ell_x)
    _cross_build_levels!(ctx.particles, ell_x)
    _cross_node_geometry!(ctx.panels, ctx.x_min, ctx.h0, ell_x)
    _cross_node_geometry!(ctx.particles, ctx.x_min, ctx.h0, ell_x)
    # particle-side dense occupancy
    fill!(ctx.d_node_at, Int32(0))
    n_nodes = ctx.particles.level_offsets[end]
    n_nodes <= typemax(Int32) || throw(AssertionError(
        "cross occupancy requires node indices to fit Int32"))
    threads = 256
    CUDA.@cuda threads=threads blocks=cld(n_nodes, threads) _cuda_hier_node_at_scatter_kernel!(
        ctx.d_node_at, ctx.particles.d_node_levels, ctx.particles.d_node_coords,
        ctx.d_level_base, n_nodes)
    # lists: far routes and demoted blocks at levels 2..ell_x, near at the leaf
    n_routes = 0
    n_blocks = 0
    for L in 2:ell_x
        n_routes = _cross_generate_level!(ctx, L, ctx.panels, ctx.d_node_at,
            ctx.d_push_offsets,
            ctx.d_class_far, ctx.d_route_levels, ctx.d_route_offsets,
            ctx.d_route_targets, ctx.d_route_sources, ctx.d_route_class,
            ctx.route_capacity, n_routes, "route")
        n_blocks = _cross_generate_level!(ctx, L, ctx.panels, ctx.d_node_at,
            ctx.d_push_offsets,
            ctx.d_class_demoted, ctx.d_block_levels, ctx.d_block_offsets,
            ctx.d_block_targets, ctx.d_block_sources, ctx.d_block_class,
            ctx.block_capacity, n_blocks, "direct-block")
    end
    ctx.n_demoted = n_blocks
    n_blocks = _cross_generate_level!(ctx, ell_x, ctx.panels, ctx.d_node_at,
        ctx.d_near_offsets,
        ctx.d_near_class, ctx.d_block_levels, ctx.d_block_offsets,
        ctx.d_block_targets, ctx.d_block_sources, ctx.d_block_class,
        ctx.block_capacity, n_blocks, "direct-block")
    ctx.n_routes = n_routes
    ctx.n_blocks = n_blocks
    if ctx.build_reverse
        # 052h reverse leg: panels become route TARGETS — scatter their dense
        # occupancy, then sweep with particle nodes as the explicit sources.
        # The stencil class masks are per-offset geometry, direction-agnostic.
        fill!(ctx.d_node_at_panels, Int32(0))
        n_pnodes = ctx.panels.level_offsets[end]
        n_pnodes <= typemax(Int32) || throw(AssertionError(
            "cross occupancy requires node indices to fit Int32"))
        CUDA.@cuda threads=threads blocks=cld(n_pnodes, threads) _cuda_hier_node_at_scatter_kernel!(
            ctx.d_node_at_panels, ctx.panels.d_node_levels,
            ctx.panels.d_node_coords, ctx.d_level_base, n_pnodes)
        n_routes = 0
        n_blocks = 0
        for L in 2:ell_x
            n_routes = _cross_generate_level!(ctx, L, ctx.particles,
                ctx.d_node_at_panels, ctx.d_push_offsets,
                ctx.d_class_far, ctx.d_rev_route_levels, ctx.d_rev_route_offsets,
                ctx.d_rev_route_targets, ctx.d_rev_route_sources,
                ctx.d_rev_route_class,
                ctx.rev_route_capacity, n_routes, "reverse-route")
            n_blocks = _cross_generate_level!(ctx, L, ctx.particles,
                ctx.d_node_at_panels, ctx.d_push_offsets,
                ctx.d_class_demoted, ctx.d_rev_block_levels,
                ctx.d_rev_block_offsets, ctx.d_rev_block_targets,
                ctx.d_rev_block_sources, ctx.d_rev_block_class,
                ctx.rev_block_capacity, n_blocks, "reverse-block")
        end
        ctx.n_rev_demoted = n_blocks
        n_blocks = _cross_generate_level!(ctx, ell_x, ctx.particles,
            ctx.d_node_at_panels, ctx.d_near_offsets,
            ctx.d_near_class, ctx.d_rev_block_levels, ctx.d_rev_block_offsets,
            ctx.d_rev_block_targets, ctx.d_rev_block_sources,
            ctx.d_rev_block_class,
            ctx.rev_block_capacity, n_blocks, "reverse-block")
        ctx.n_rev_routes = n_routes
        ctx.n_rev_blocks = n_blocks
    end
    return ctx
end

#------- Stage B: device panel B2M + upward M2M -------#
#
# Thread-per-panel B2M over the 17-row seam buffer (row 1 tag, row 2 nv,
# rows 3:14 vertices, rows 15:16 strengths): each thread runs the Gumerov-2023
# simplex recurrences (faithful device transcriptions of bodytomultipole.jl
# calculate_q!/calculate_pj!/calculate_ib!/source_to_dipole!) in a per-thread
# MArray scratch and atomically accumulates classic phi coefficients
# (row = 2*(harmonic_index - 1) + re/im) into its leaf node's column.
# Tag mapping mirrors the host shims: 1 → Source(s1); 2 → Dipole(s1);
# 3 → Dipole(s1) (a closed vortex ring enters as its dipole-panel equivalent,
# exactly the pure-VortexRing host body_to_multipole! overload,
# FLOWPanel_liftingbody.jl:804-910); 4/5 → Source(s1) + Dipole(s2) (the host
# Panel{SourceDipole} route). Unknown tags and nv < 3 (which includes tag-3
# nv=2 OPEN filaments — no closed ring, no dipole equivalent) are counted in
# `skipped` and contribute nothing; callers should treat a nonzero count as an
# error (the seam does). nv == 4 splits into triangles (1,2,3) and
# (1,3,4) with per-triangle normals, as the host quad path does.
#
# TE-wake arm (wake-partition ruling, session 5): when a wake matrix is
# supplied — 8 rows per BODY panel mirroring the FMM source buffer's trailing
# `end-7..end` block (idx1, Da, idx2, Db; FLOWPanel_liftingbody.jl
# additional_source_system_to_buffer!) — each processed panel with idx1 > 0
# also accumulates the two attached TE-wake dipole triangles of the
# RigidWakeBody body_to_multipole! overload (FLOWPanel_liftingbody.jl:739-786):
# (v[idx1], v[idx2], v[idx1]+Da) and (v[idx1]+Da, v[idx2], v[idx2]+Db), pure
# Dipole with the panel's dipole strength, into the PANEL's leaf node (host
# semantics: the wake rides with its panel; only the first wake row — rows
# behind it go to the wake-on-all step). The seam's appended wake COLUMNS
# (pack_panels!) serve the near-field direct functor and carry
# wake_strength_shift; they must NOT feed this B2M path.
#
# The upward pass applies dense per-(child level, octant) operators
# (cross_m2m_operators — probed from the production host M2M, so
# convention-exact) bottom-up with one atomic accumulation per (row, child).

const _CROSS_B2M_MAX_P = 8
const _CROSS_B2M_SCRATCH_H = ((_CROSS_B2M_MAX_P + 1) * (_CROSS_B2M_MAX_P + 2)) >> 1
# NOTE: the tag → B2M arm mapping `_cross_b2m_arms` the kernel consumes lives
# in cross_stencil_host.jl so the CPU tests cover it without CUDA.

# neighbor fetches at degree n-1 / n (device copies of bodytomultipole.jl
# get_nm1/get_n without the host @asserts)
@inline function _crossb2m_get_nm1(harm, ch, n, m, i_nm1_m, _1_m)
    z = zero(eltype(harm))
    if m > 0
        a_re = harm[1, ch, i_nm1_m - 1]; a_im = harm[2, ch, i_nm1_m - 1]
    elseif n > 1
        a_re = -_1_m * harm[1, ch, i_nm1_m + 1]; a_im = _1_m * harm[2, ch, i_nm1_m + 1]
    else
        a_re = z; a_im = z
    end
    if m < n
        b_re = harm[1, ch, i_nm1_m]; b_im = harm[2, ch, i_nm1_m]
    else
        b_re = z; b_im = z
    end
    if m + 1 < n
        c_re = harm[1, ch, i_nm1_m + 1]; c_im = harm[2, ch, i_nm1_m + 1]
    else
        c_re = z; c_im = z
    end
    return a_re, a_im, b_re, b_im, c_re, c_im
end

@inline function _crossb2m_q!(harm, ξ_re, ξ_im, η_re, η_im, z, P)
    harm[1, 1, 1] = one(eltype(harm))
    harm[2, 1, 1] = zero(eltype(harm))
    i = 2
    for n in 1:P
        i_nm1_m = i - n
        _1_m = 1.0
        for m in 0:n
            a_re, a_im, b_re, b_im, c_re, c_im =
                _crossb2m_get_nm1(harm, 1, n, m, i_nm1_m, _1_m)
            harm[1, 1, i] = (-(ξ_re * a_im + ξ_im * a_re) -
                (η_re * c_im + η_im * c_re) - z * b_re) / n
            harm[2, 1, i] = ((ξ_re * a_re - ξ_im * a_im) +
                (η_re * c_re - η_im * c_im) - z * b_im) / n
            i += 1; i_nm1_m += 1; _1_m = -_1_m
        end
    end
    return nothing
end

@inline function _crossb2m_pj!(harm, ξ_re, ξ_im, η_re, η_im, z, P)
    harm[1, 2, 1] = one(eltype(harm))
    harm[2, 2, 1] = zero(eltype(harm))
    i = 2
    for n in 1:P
        i_nm1_m = i - n
        _1_m = 1.0
        for m in 0:n
            a_re, a_im, b_re, b_im, c_re, c_im =
                _crossb2m_get_nm1(harm, 2, n, m, i_nm1_m, _1_m)
            q_re = harm[1, 1, i]; q_im = harm[2, 1, i]
            harm[1, 2, i] = (-(ξ_re * a_im + ξ_im * a_re) -
                (η_re * c_im + η_im * c_re) - z * b_re + q_re) / (n + 1)
            harm[2, 2, i] = ((ξ_re * a_re - ξ_im * a_im) +
                (η_re * c_re - η_im * c_im) - z * b_im + q_im) / (n + 1)
            i += 1; i_nm1_m += 1; _1_m = -_1_m
        end
    end
    return nothing
end

@inline function _crossb2m_ib!(harm, ξ_re, ξ_im, η_re, η_im, z, P)
    harm[1, 1, 1] = eltype(harm)(0.5)
    harm[2, 1, 1] = zero(eltype(harm))
    i = 2
    for n in 1:P
        i_nm1_m = i - n
        _1_m = 1.0
        for m in 0:n
            a_re, a_im, b_re, b_im, c_re, c_im =
                _crossb2m_get_nm1(harm, 1, n, m, i_nm1_m, _1_m)
            j_re = harm[1, 2, i]; j_im = harm[2, 2, i]
            harm[1, 1, i] = (-(ξ_re * a_im + ξ_im * a_re) -
                (η_re * c_im + η_im * c_re) - z * b_re + j_re) / (n + 2)
            harm[2, 1, i] = ((ξ_re * a_re - ξ_im * a_im) +
                (η_re * c_re - η_im * c_im) - z * b_im + j_im) / (n + 2)
            i += 1; i_nm1_m += 1; _1_m = -_1_m
        end
    end
    return nothing
end

@inline function _crossb2m_source_to_dipole!(harm, qx, qy, qz, P)
    harm[1, 2, 1] = zero(eltype(harm))
    harm[2, 2, 1] = zero(eltype(harm))
    i = 2
    for n in 1:P
        i_nm1_m = i - n
        _1_m = 1.0
        for m in 0:n
            a_re, a_im, b_re, b_im, c_re, c_im =
                _crossb2m_get_nm1(harm, 1, n, m, i_nm1_m, _1_m)
            harm[1, 2, i] = -qx * 0.5 * (c_im + a_im) + qy * 0.5 * (c_re - a_re) -
                qz * b_re
            harm[2, 2, i] = qx * 0.5 * (c_re + a_re) + qy * 0.5 * (c_im - a_im) -
                qz * b_im
            i += 1; i_nm1_m += 1; _1_m = -_1_m
        end
    end
    return nothing
end

# One triangle's Source and/or Dipole contribution, accumulated atomically into
# `multipoles[:, node]`. Mirrors body_to_multipole_panel! exactly: the shared
# q/j/i recurrences are computed once (the host Source and Dipole paths
# recompute identical values); the Source loop leaves the scratch untouched and
# source_to_dipole! reads only channel 1, so sharing is bit-safe.
@inline function _crossb2m_triangle!(multipoles, harm, node, v1, v2, v3, center,
        do_source, s_source, do_dipole, s_dipole, P)
    x0 = v1 - center
    xu = v2 - v1
    xv = v3 - v1
    ξ0_re, ξ0_im, η0_re, η0_im, z0 = x0[1] * 0.5, x0[2] * 0.5, x0[1] * 0.5,
        -x0[2] * 0.5, x0[3]
    ξu_re, ξu_im, ηu_re, ηu_im, zu = xu[1] * 0.5, xu[2] * 0.5, xu[1] * 0.5,
        -xu[2] * 0.5, xu[3]
    ξv_re, ξv_im, ηv_re, ηv_im, zv = xv[1] * 0.5, xv[2] * 0.5, xv[1] * 0.5,
        -xv[2] * 0.5, xv[3]
    _crossb2m_q!(harm, ξ0_re + ξu_re, ξ0_im + ξu_im, η0_re + ηu_re,
        η0_im + ηu_im, z0 + zu, P)
    _crossb2m_pj!(harm, ξ0_re + ξv_re, ξ0_im + ξv_im, η0_re + ηv_re,
        η0_im + ηv_im, z0 + zv, P)
    _crossb2m_ib!(harm, ξ0_re, ξ0_im, η0_re, η0_im, z0, P)
    nx = xu[2] * xv[3] - xu[3] * xv[2]
    ny = xu[3] * xv[1] - xu[1] * xv[3]
    nz = xu[1] * xv[2] - xu[2] * xv[1]
    Δx = sqrt(nx * nx + ny * ny + nz * nz)
    if do_source
        Jq = Δx * (-s_source)   # sign flip: v = ∇ϕ convention (host comment)
        i = 1
        _1_n = 1.0
        for n in 0:P
            _1_n_m = _1_n
            for m in 0:n
                CUDA.@atomic multipoles[2 * (i - 1) + 1, node] +=
                    Jq * _1_n_m * harm[1, 1, i]
                CUDA.@atomic multipoles[2 * (i - 1) + 2, node] +=
                    -(Jq * _1_n_m * harm[2, 1, i])
                i += 1; _1_n_m = -_1_n_m
            end
            _1_n = -_1_n
        end
    end
    if do_dipole
        inv_n = 1.0 / Δx
        qx = -nx * inv_n * s_dipole
        qy = -ny * inv_n * s_dipole
        qz = -nz * inv_n * s_dipole
        _crossb2m_source_to_dipole!(harm, qx, qy, qz, P)
        i = 1
        _1_n = 1.0
        for n in 0:P
            _1_n_m = _1_n
            for m in 0:n
                CUDA.@atomic multipoles[2 * (i - 1) + 1, node] +=
                    Δx * _1_n_m * harm[1, 2, i]
                CUDA.@atomic multipoles[2 * (i - 1) + 2, node] +=
                    -(Δx * _1_n_m * harm[2, 2, i])
                i += 1; _1_n_m = -_1_n_m
            end
            _1_n = -_1_n
        end
    end
    return nothing
end

function _cross_panel_b2m_kernel!(multipoles, panel_buffer, wake_buffer,
        has_wake, node_centers, body_node, perm, P, n_panels, skipped)
    t = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    t > n_panels && return nothing
    @inbounds begin
        node = Int(body_node[t])
        col = Int(perm[t])
        tag = Int(panel_buffer[1, col])
        nv = Int(panel_buffer[2, col])
        if !(1 <= tag <= 5) || nv < 3
            CUDA.@atomic skipped[1] += Int32(1)
            return nothing
        end
        s1 = panel_buffer[15, col]
        s2 = panel_buffer[16, col]
        do_source, do_dipole, s_source, s_dipole = _cross_b2m_arms(tag, s1, s2)
        center = SVector(node_centers[1, node], node_centers[2, node],
            node_centers[3, node])
        v1 = SVector(panel_buffer[3, col], panel_buffer[4, col], panel_buffer[5, col])
        v2 = SVector(panel_buffer[6, col], panel_buffer[7, col], panel_buffer[8, col])
        v3 = SVector(panel_buffer[9, col], panel_buffer[10, col], panel_buffer[11, col])
        harm = MArray{Tuple{2,2,_CROSS_B2M_SCRATCH_H},Float64}(undef)
        _crossb2m_triangle!(multipoles, harm, node, v1, v2, v3, center,
            do_source, s_source, do_dipole, s_dipole, P)
        if nv == 4
            v4 = SVector(panel_buffer[12, col], panel_buffer[13, col],
                panel_buffer[14, col])
            _crossb2m_triangle!(multipoles, harm, node, v1, v3, v4, center,
                do_source, s_source, do_dipole, s_dipole, P)
        end
        if has_wake
            idx1 = Int(wake_buffer[1, col])
            if idx1 > 0
                # host convention: vs = (x1, x2, x3); wake indices address the
                # first three vertices only (RigidWakeBody cells are triangles)
                idx2 = Int(wake_buffer[5, col])
                w1 = idx1 == 1 ? v1 : (idx1 == 2 ? v2 : v3)
                w2 = idx2 == 1 ? v1 : (idx2 == 2 ? v2 : v3)
                v1w = w1 + SVector(wake_buffer[2, col], wake_buffer[3, col],
                    wake_buffer[4, col])
                v2w = w2 + SVector(wake_buffer[6, col], wake_buffer[7, col],
                    wake_buffer[8, col])
                z = zero(s_dipole)
                _crossb2m_triangle!(multipoles, harm, node, w1, w2, v1w, center,
                    false, z, true, s_dipole, P)
                _crossb2m_triangle!(multipoles, harm, node, v1w, w2, v2w, center,
                    false, z, true, s_dipole, P)
            end
        end
    end
    return nothing
end

function _cross_body_node_kernel!(body_node, node_ranges, leaf_first, n_leaf)
    c = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    c > n_leaf && return nothing
    @inbounds begin
        node = leaf_first + c - 1
        first = Int(node_ranges[1, node])
        stop = first + Int(node_ranges[2, node]) - 1
        for s in first:stop
            body_node[s] = Int32(node)
        end
    end
    return nothing
end

function _cross_m2m_kernel!(multipoles, ops, parent_index, node_coords,
        first_child, n_children, Lc, D)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    idx > n_children * D && return nothing
    @inbounds begin
        c = (idx - 1) ÷ D + 1
        row = (idx - 1) % D + 1
        child = first_child + c - 1
        parent = Int(parent_index[child])
        parent == 0 && return nothing
        phase = 1 + (node_coords[1, child] & Int32(1)) +
            2 * (node_coords[2, child] & Int32(1)) +
            4 * (node_coords[3, child] & Int32(1))
        acc = zero(eltype(multipoles))
        for j in 1:D
            acc = muladd(ops[row, j, phase, Lc], multipoles[j, child], acc)
        end
        CUDA.@atomic multipoles[row, parent] += acc
    end
    return nothing
end

"""
Device cross-pass expansion state: classic phi-channel multipoles over the
panel node set (row = 2*(harmonic_index - 1) + re/im) plus the uploaded dense
per-(level, octant) M2M operators. Rebuilds the operators when the root box
changed (they depend on h0).
"""
mutable struct DeviceCrossExpansionState{TF}
    P::Int
    D::Int
    h0_ops::TF
    d_multipoles::CUDA.CuMatrix{TF}
    d_m2m_ops::CUDA.CuArray{TF,4}
    d_skipped::CUDA.CuVector{Int32}
    n_skipped::Int
    host_scalar32::Vector{Int32}
    d_empty_wake::CUDA.CuMatrix{TF}   # 8×0 placeholder for the no-wake kernel call
end

function device_cross_expansion_state(ctx::DeviceCrossProducerContext{TF},
        P::Integer) where {TF}
    TF === Float64 || throw(ArgumentError(
        "the cross-pass B2M kernel scratch is Float64-only; build the " *
        "producer context with Float64 positions (got $TF)"))
    P <= _CROSS_B2M_MAX_P || throw(ArgumentError(
        "cross panel B2M scratch is sized for P <= $_CROSS_B2M_MAX_P; got $P"))
    D = (Int(P) + 1) * (Int(P) + 2)
    ops = cross_m2m_operators(P, ctx.h0, ctx.ell_x)
    return DeviceCrossExpansionState{TF}(Int(P), D, ctx.h0,
        CUDA.zeros(TF, D, ctx.panels.max_nodes),
        CUDA.CuArray{TF}(ops),
        CUDA.zeros(Int32, 1), 0, zeros(Int32, 1),
        CUDA.zeros(TF, 8, 0))
end

"""
    refresh_cross_multipoles!(xs, ctx, d_panel_buffer, d_wake_buffer=nothing)

Stage-B per-step device pass: panel B2M at the cross leaf cells + dense
octant-class M2M up the panel node set. Requires `refresh_cross_producers!`
to have run this step (consumes its occupancy). `d_panel_buffer` is the
17-row seam buffer (device, columns = panels in original order).
`d_wake_buffer`, if given, is an `8 × n_panels` device matrix carrying each
BODY panel's attached-TE-wake block (idx1, Da, idx2, Db — the FMM source
buffer's trailing `end-7..end` rows); the kernel then adds the two TE-wake
dipole triangles per shedding panel exactly as the host RigidWakeBody
`body_to_multipole!` overload does.
"""
function refresh_cross_multipoles!(xs::DeviceCrossExpansionState{TF},
        ctx::DeviceCrossProducerContext{TF}, d_panel_buffer,
        d_wake_buffer=nothing) where {TF}
    occ = ctx.panels
    ell_x = ctx.ell_x
    if xs.h0_ops != ctx.h0   # defensive: ctx.h0 is frozen post-construction
        copyto!(xs.d_m2m_ops, cross_m2m_operators(xs.P, ctx.h0, ell_x))
        xs.h0_ops = ctx.h0
    end
    fill!(xs.d_multipoles, zero(TF))
    fill!(xs.d_skipped, Int32(0))
    threads = 256
    # leaf-node membership of each sorted panel
    leaf_first = occ.level_offsets[ell_x + 1] + 1
    n_leaf = occ.level_offsets[ell_x + 2] - occ.level_offsets[ell_x + 1]
    n_leaf > 0 || return xs
    CUDA.@cuda threads=threads blocks=cld(n_leaf, threads) _cross_body_node_kernel!(
        occ.d_body_node, occ.d_node_ranges, leaf_first, n_leaf)
    # parent links for the upward pass
    max_per_level = maximum(occ.level_offsets[L + 2] - occ.level_offsets[L + 1]
        for L in 0:ell_x)
    CUDA.@cuda threads=threads blocks=(cld(max_per_level, threads), ell_x + 1) _cuda_parent_index_kernel!(
        occ.d_parent_index, occ.d_node_keys, occ.d_level_offsets, ell_x, 0)
    # B2M at the leaves
    has_wake = d_wake_buffer !== nothing
    if has_wake
        (size(d_wake_buffer, 1) == 8 && size(d_wake_buffer, 2) == occ.n) ||
            throw(ArgumentError("wake buffer must be 8 × n_panels " *
                "(got $(size(d_wake_buffer)); n_panels = $(occ.n))"))
    end
    d_wake = has_wake ? d_wake_buffer : xs.d_empty_wake
    CUDA.@cuda threads=threads blocks=cld(occ.n, threads) _cross_panel_b2m_kernel!(
        xs.d_multipoles, d_panel_buffer, d_wake, has_wake, occ.d_node_centers,
        occ.d_body_node, occ.d_perm, xs.P, occ.n, xs.d_skipped)
    # upward: dense octant-class M2M, bottom-up
    for Lc in ell_x:-1:1
        first_child = occ.level_offsets[Lc + 1] + 1
        n_children = occ.level_offsets[Lc + 2] - occ.level_offsets[Lc + 1]
        n_children > 0 || continue
        CUDA.@cuda threads=threads blocks=cld(n_children * xs.D, threads) _cross_m2m_kernel!(
            xs.d_multipoles, xs.d_m2m_ops, occ.d_parent_index, occ.d_node_coords,
            first_child, n_children, Lc, xs.D)
    end
    copyto!(xs.host_scalar32, 1, xs.d_skipped, 1, 1)
    xs.n_skipped = Int(xs.host_scalar32[1])
    return xs
end

#------- Stage C: cross-M2L over the far route lists -------#
#
# Thread-per-(route, row) dense matvec: each far route (level L, source panel
# node, target particle node, offset class k) applies the slot-compacted
# reference-level M2L operator (cross_m2l_operators — probed from the
# production host M2L at level _CROSS_M2L_L_REF, classic-phi row basis) with
# the exact separable level rescaling f_L(n_row, n_col) =
# scale2[row, L] * pow2lvl[L] * scale2[col, L] (Stage-C header note in
# cross_stencil_host.jl), atomically accumulating into the target node's
# local expansion. Locals live on the PARTICLE node set at route levels
# 2..ell_x; Stage D pushes them down and evaluates at bodies.

function _cross_m2l_kernel!(locals, multipoles, ops, class_slot, scale2,
        pow2lvl, route_levels, route_targets, route_sources, route_class,
        n_routes, D)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    idx > n_routes * D && return nothing
    @inbounds begin
        r = (idx - 1) ÷ D + 1
        row = (idx - 1) % D + 1
        slot = Int(class_slot[route_class[r]])
        slot == 0 && return nothing   # defensively skip (far mask guarantees > 0)
        L1 = Int(route_levels[r]) + 1
        source = route_sources[r]
        target = route_targets[r]
        acc = zero(eltype(locals))
        for j in 1:D
            acc = muladd(ops[row, j, slot], multipoles[j, source] * scale2[j, L1],
                acc)
        end
        acc *= scale2[row, L1] * pow2lvl[L1]
        CUDA.@atomic locals[row, target] += acc
    end
    return nothing
end

#------- Stage D: downward L2L + leaf L2B over the particle node set -------#
#
# L2L applies the dense per-(child level, octant) operators
# (cross_l2l_operators — probed from the production host local_to_local!)
# top-down over the particle occupancy: child levels 3..ell_x (locals start
# at the coarsest route level 2). One thread per (child, row); parents and
# children live at different levels, so the read/modify of `locals` needs no
# atomics. L2B then evaluates each particle's leaf-node local at its position
# via `_resident_local_eval_flat` — its flat layout IS the classic-phi row
# basis (`flat_basis_index(n, m, reim) = 2*(harmonic_index - 1) + reim`,
# containers.jl:1224), verified against the host classic `evaluate_local` by
# the CPU test — writing (potential, gradient) rows through the particle
# permutation into original column order (no hessian).

function _cross_l2l_kernel!(locals, ops, parent_index, node_coords,
        first_child, n_children, Lc, D)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    idx > n_children * D && return nothing
    @inbounds begin
        c = (idx - 1) ÷ D + 1
        row = (idx - 1) % D + 1
        child = first_child + c - 1
        parent = Int(parent_index[child])
        parent == 0 && return nothing
        phase = 1 + (node_coords[1, child] & Int32(1)) +
            2 * (node_coords[2, child] & Int32(1)) +
            4 * (node_coords[3, child] & Int32(1))
        acc = zero(eltype(locals))
        for j in 1:D
            acc = muladd(ops[row, j, phase, Lc], locals[j, parent], acc)
        end
        locals[row, child] += acc
    end
    return nothing
end

function _cross_l2b_kernel!(out, positions, perm, body_node, node_centers,
        locals_phi, locals_chi, P, n, lhv)
    s = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    s > n && return nothing
    @inbounds begin
        node = Int(body_node[s])
        col = Int(perm[s])
        dx = positions[1, col] - node_centers[1, node]
        dy = positions[2, col] - node_centers[2, node]
        dz = positions[3, col] - node_centers[3, node]
        u, gx, gy, gz = _resident_local_eval_flat(locals_phi, locals_chi, node,
            dx, dy, dz, P, P, lhv)
        out[1, col] += u
        out[2, col] += gx
        out[3, col] += gy
        out[4, col] += gz
    end
    return nothing
end

"""
Device cross-pass local-expansion state: classic phi-channel locals over the
PARTICLE node set (row = 2*(harmonic_index - 1) + re/im), the uploaded
slot-compacted reference-level M2L operators and level-rescale tables, the
dense octant-class L2L operators, and the 4-row (potential, gradient) output
in ORIGINAL particle column order. Rebuilds the operators when the root box
changed (they depend on h0).
"""
mutable struct DeviceCrossLocalState{TF}
    P::Int
    D::Int
    h0_ops::TF
    d_locals::CUDA.CuMatrix{TF}
    d_m2l_ops::CUDA.CuArray{TF,3}
    d_class_slot::CUDA.CuVector{Int32}
    d_scale2::CUDA.CuMatrix{TF}
    d_pow2lvl::CUDA.CuVector{TF}
    d_l2l_ops::CUDA.CuArray{TF,4}
    d_out::CUDA.CuMatrix{TF}
    d_empty_wake::CUDA.CuMatrix{TF}    # 8×0 placeholder for no-wake near calls
    d_empty_shift::CUDA.CuVector{TF}   # 0-length placeholder likewise
end

function device_cross_local_state(ctx::DeviceCrossProducerContext{TF},
        P::Integer; m2l_tables=nothing, l2l_table=nothing) where {TF}
    D = (Int(P) + 1) * (Int(P) + 2)
    # m2l_tables / l2l_table: optional precomputed host tables from
    # cross_m2l_operators(P, ctx.h0, ctx.ct) / cross_l2l_operators — probing
    # is the expensive part of state construction, so callers that rebuild
    # state on particle-count growth (the seam) cache them host-side, keyed
    # on the frozen (P, h0).
    ops, class_slot = m2l_tables === nothing ?
        cross_m2l_operators(P, ctx.h0, ctx.ct) : m2l_tables
    l2l = l2l_table === nothing ?
        cross_l2l_operators(P, ctx.h0, ctx.ell_x) : l2l_table
    scale2, pow2lvl = cross_m2l_level_scales(P, ctx.ell_x)
    return DeviceCrossLocalState{TF}(Int(P), D, ctx.h0,
        CUDA.zeros(TF, D, ctx.particles.max_nodes),
        CUDA.CuArray{TF}(ops),
        CUDA.CuArray{Int32}(class_slot),
        CUDA.CuArray{TF}(scale2),
        CUDA.CuArray{TF}(pow2lvl),
        CUDA.CuArray{TF}(l2l),
        CUDA.zeros(TF, 4, ctx.particles.n),
        CUDA.zeros(TF, 8, 0),
        CUDA.zeros(TF, 0))
end

"""
    refresh_cross_locals!(ls, ctx, xs)

Stage-C per-step device pass: cross-M2L over the Stage-A far route lists,
consuming the Stage-B multipoles (`xs`) and accumulating classic-phi locals
on the particle node set. Requires `refresh_cross_producers!` and
`refresh_cross_multipoles!` to have run this step.
"""
function refresh_cross_locals!(ls::DeviceCrossLocalState{TF},
        ctx::DeviceCrossProducerContext{TF},
        xs::DeviceCrossExpansionState{TF}) where {TF}
    ls.P == xs.P || throw(ArgumentError(
        "local state P = $(ls.P) does not match expansion state P = $(xs.P)"))
    if ls.h0_ops != ctx.h0   # defensive: ctx.h0 is frozen post-construction
        ops, class_slot = cross_m2l_operators(ls.P, ctx.h0, ctx.ct)
        copyto!(ls.d_m2l_ops, ops)
        copyto!(ls.d_class_slot, class_slot)
        copyto!(ls.d_l2l_ops, cross_l2l_operators(ls.P, ctx.h0, ctx.ell_x))
        ls.h0_ops = ctx.h0
    end
    fill!(ls.d_locals, zero(TF))
    ctx.n_routes > 0 || return ls
    threads = 256
    CUDA.@cuda threads=threads blocks=cld(ctx.n_routes * ls.D, threads) _cross_m2l_kernel!(
        ls.d_locals, xs.d_multipoles, ls.d_m2l_ops, ls.d_class_slot,
        ls.d_scale2, ls.d_pow2lvl, ctx.d_route_levels, ctx.d_route_targets,
        ctx.d_route_sources, ctx.d_route_class, ctx.n_routes, ls.D)
    return ls
end

"""
    finish_cross_locals!(ls, ctx, d_particle_positions)

Stage-D per-step device pass: dense octant-class L2L top-down over the
particle node set (child levels `3:ell_x`), then leaf L2B into `ls.d_out`
(rows potential, gx, gy, gz; columns = particles in original order).
Requires `refresh_cross_locals!` to have run this step. `d_particle_positions`
must be the same `3 × n` device matrix the producers were refreshed with.
"""
function finish_cross_locals!(ls::DeviceCrossLocalState{TF},
        ctx::DeviceCrossProducerContext{TF}, d_particle_positions) where {TF}
    occ = ctx.particles
    ell_x = ctx.ell_x
    threads = 256
    fill!(ls.d_out, zero(TF))
    occ.n > 0 || return ls
    # parent links for the particle node set
    max_per_level = maximum(occ.level_offsets[L + 2] - occ.level_offsets[L + 1]
        for L in 0:ell_x)
    max_per_level > 0 || return ls
    CUDA.@cuda threads=threads blocks=(cld(max_per_level, threads), ell_x + 1) _cuda_parent_index_kernel!(
        occ.d_parent_index, occ.d_node_keys, occ.d_level_offsets, ell_x, 0)
    # L2L top-down: locals begin at the coarsest route level 2
    for Lc in 3:ell_x
        first_child = occ.level_offsets[Lc + 1] + 1
        n_children = occ.level_offsets[Lc + 2] - occ.level_offsets[Lc + 1]
        n_children > 0 || continue
        CUDA.@cuda threads=threads blocks=cld(n_children * ls.D, threads) _cross_l2l_kernel!(
            ls.d_locals, ls.d_l2l_ops, occ.d_parent_index, occ.d_node_coords,
            first_child, n_children, Lc, ls.D)
    end
    # leaf-node membership of each sorted particle, then evaluate
    leaf_first = occ.level_offsets[ell_x + 1] + 1
    n_leaf = occ.level_offsets[ell_x + 2] - occ.level_offsets[ell_x + 1]
    n_leaf > 0 || return ls
    CUDA.@cuda threads=threads blocks=cld(n_leaf, threads) _cross_body_node_kernel!(
        occ.d_body_node, occ.d_node_ranges, leaf_first, n_leaf)
    CUDA.@cuda threads=threads blocks=cld(occ.n, threads) _cross_l2b_kernel!(
        ls.d_out, d_particle_positions, occ.d_perm, occ.d_body_node,
        occ.d_node_centers, ls.d_locals, ls.d_locals, ls.P, occ.n, Val(false))
    return ls
end

#------- Stage E: block-sparse near field over the direct-block lists -------#
#
# One CUDA block per direct block (grid-strided): target range = the particle
# node's sorted range, source range = the panel node's sorted range (both via
# node_ranges — subtree ranges are contiguous at every level, so demoted
# blocks at coarse levels work unchanged). Panel columns are staged through
# shared memory in 26-row tiles (17 seam rows + 8 wake rows + 1 wake-shift
# row) and evaluated with the PRODUCTION pair math `_rect_panel_pair`
# (regularization code via Val — LineGauss = 4) plus `_rect_panel_potential`,
# accumulating (potential, velocity) atomically into the SAME `ls.d_out` the
# far field (Stage D) wrote — no hessian.
#
# TE-wake arm (near side of the wake-partition ruling): a shedding panel's
# two attached wake triangles are evaluated with the body's WAKE KERNEL tag
# (3 = tri vortex ring, 2 = tri doublet — get_wake_kernel), strength =
# panel dipole strength (+ optional per-panel wake_strength_shift, matching
# the production `_induced_wake`/pack_panels! near-field path, which — unlike
# the far-field B2M overload — DOES include the shift).

const _CROSS_NEAR_TILE = 128

function _cross_near_kernel!(out, panel_buffer, wake_buffer, has_wake,
        wake_shift, has_shift, wake_tag, positions, t_perm, s_perm,
        t_node_ranges, s_node_ranges, block_targets, block_sources, n_blocks,
        ::Val{REG}, ::Val{POT}) where {REG, POT}
    T = eltype(out)
    tid = threadIdx().x
    sh = CUDA.CuStaticSharedArray(T, (26, _CROSS_NEAR_TILE))
    b = blockIdx().x
    @inbounds while b <= n_blocks
        tnode = Int(block_targets[b])
        snode = Int(block_sources[b])
        tfirst = Int(t_node_ranges[1, tnode])
        tcount = Int(t_node_ranges[2, tnode])
        sfirst = Int(s_node_ranges[1, snode])
        scount = Int(s_node_ranges[2, snode])
        ntiles = cld(scount, _CROSS_NEAR_TILE)
        for tc in 1:cld(tcount, blockDim().x)
            ti = tfirst + (tc - 1) * blockDim().x + tid - 1
            active = ti <= tfirst + tcount - 1
            col = 1
            target = zero(SVector{3,T})
            if active
                col = Int(t_perm[ti])
                target = SVector{3,T}(positions[1, col], positions[2, col],
                    positions[3, col])
            end
            p = zero(T)
            u = zero(SVector{3,T})
            for t in 1:ntiles
                q0 = (t - 1) * _CROSS_NEAR_TILE
                q = q0 + tid
                if tid <= _CROSS_NEAR_TILE && q <= scount
                    scol = Int(s_perm[sfirst + q - 1])
                    for r in 1:17
                        sh[r, tid] = panel_buffer[r, scol]
                    end
                    if has_wake
                        for r in 1:8
                            sh[17 + r, tid] = wake_buffer[r, scol]
                        end
                        sh[26, tid] = has_shift ? wake_shift[scol] : zero(T)
                    end
                end
                CUDA.sync_threads()
                if active
                    for k in 1:min(_CROSS_NEAR_TILE, scount - q0)
                        tag = unsafe_trunc(Int, sh[1, k])
                        nv = unsafe_trunc(Int, sh[2, k])
                        (1 <= tag <= 5 && nv >= 3) || continue
                        v1 = SVector{3,T}(sh[3, k], sh[4, k], sh[5, k])
                        v2 = SVector{3,T}(sh[6, k], sh[7, k], sh[8, k])
                        v3 = SVector{3,T}(sh[9, k], sh[10, k], sh[11, k])
                        v4 = SVector{3,T}(sh[12, k], sh[13, k], sh[14, k])
                        s1 = sh[15, k]
                        s2 = sh[16, k]
                        koff = sh[17, k]
                        uq, _ = _rect_panel_pair(RectangularPanelInfluence(),
                            target, tag, nv, v1, v2, v3, v4, s1, s2, koff,
                            Val(false), Val(REG))
                        u += uq
                        if POT
                            p += _rect_panel_potential(target, tag, nv, v1, v2,
                                v3, s1, s2)
                        end
                        if has_wake
                            idx1 = unsafe_trunc(Int, sh[18, k])
                            if idx1 > 0
                                idx2 = unsafe_trunc(Int, sh[22, k])
                                w1 = idx1 == 1 ? v1 : (idx1 == 2 ? v2 : v3)
                                w2 = idx2 == 1 ? v1 : (idx2 == 2 ? v2 : v3)
                                v1w = w1 + SVector{3,T}(sh[19, k], sh[20, k],
                                    sh[21, k])
                                v2w = w2 + SVector{3,T}(sh[23, k], sh[24, k],
                                    sh[25, k])
                                # panel dipole strength: s1 for tags 2/3, s2
                                # for the combined tags 4/5 (_cross_b2m_arms)
                                mu = (tag == 2 || tag == 3 ? s1 : s2) + sh[26, k]
                                uw1, _ = _rect_panel_pair(
                                    RectangularPanelInfluence(), target,
                                    wake_tag, 3, w1, w2, v1w, v1w, mu, zero(T),
                                    koff, Val(false), Val(REG))
                                uw2, _ = _rect_panel_pair(
                                    RectangularPanelInfluence(), target,
                                    wake_tag, 3, v1w, w2, v2w, v2w, mu, zero(T),
                                    koff, Val(false), Val(REG))
                                u += uw1 + uw2
                                if POT
                                    p += _rect_panel_potential(target, wake_tag,
                                        3, w1, w2, v1w, mu, zero(T))
                                    p += _rect_panel_potential(target, wake_tag,
                                        3, v1w, w2, v2w, mu, zero(T))
                                end
                            end
                        end
                    end
                end
                CUDA.sync_threads()
            end
            if active
                if POT
                    CUDA.@atomic out[1, col] += p
                end
                CUDA.@atomic out[2, col] += u[1]
                CUDA.@atomic out[3, col] += u[2]
                CUDA.@atomic out[4, col] += u[3]
            end
        end
        b += gridDim().x
    end
    return nothing
end

"""
    apply_cross_near!(ls, ctx, d_panel_buffer, d_particle_positions;
        d_wake_buffer=nothing, d_wake_shift=nothing, wake_tag=3, reg=4,
        potential=true)

Stage-E per-step device pass: direct evaluation of the demoted + near blocks,
accumulating (potential, velocity) into `ls.d_out` on top of the Stage-D far
field. `d_wake_buffer` is the same `8 × n_panels` matrix as Stage B;
`d_wake_shift` (optional, requires the wake buffer) is a per-panel
`wake_strength_shift` added to the wake triangles' strength — near-field
only, matching the production `_induced_wake` path. `wake_tag` selects the
wake kernel (3 = tri vortex ring, 2 = tri doublet); `reg` is the filament
regularization code (4 = LineGauss). `potential=false` compiles out the
`_rect_panel_potential` arm and the row-1 atomic (the U-only production path;
oracle/XVERIFY paths keep the default `true`).
"""
function apply_cross_near!(ls::DeviceCrossLocalState{TF},
        ctx::DeviceCrossProducerContext{TF}, d_panel_buffer,
        d_particle_positions; d_wake_buffer=nothing, d_wake_shift=nothing,
        wake_tag::Int=3, reg::Int=4, potential::Bool=true) where {TF}
    ctx.n_blocks > 0 || return ls
    has_wake = d_wake_buffer !== nothing
    has_shift = d_wake_shift !== nothing
    has_shift && !has_wake && throw(ArgumentError(
        "d_wake_shift requires d_wake_buffer"))
    if has_wake
        (size(d_wake_buffer, 1) == 8 && size(d_wake_buffer, 2) == ctx.panels.n) ||
            throw(ArgumentError("wake buffer must be 8 × n_panels " *
                "(got $(size(d_wake_buffer)); n_panels = $(ctx.panels.n))"))
    end
    wake_tag in (2, 3) || throw(ArgumentError("wake_tag must be 2 or 3"))
    d_wake = has_wake ? d_wake_buffer : ls.d_empty_wake
    d_shift = has_shift ? d_wake_shift : ls.d_empty_shift
    threads = _CROSS_NEAR_TILE
    blocks = min(ctx.n_blocks, _RECT_MAX_BLOCKS)
    CUDA.@cuda threads=threads blocks=blocks _cross_near_kernel!(
        ls.d_out, d_panel_buffer, d_wake, has_wake, d_shift, has_shift,
        wake_tag, d_particle_positions, ctx.particles.d_perm, ctx.panels.d_perm,
        ctx.particles.d_node_ranges, ctx.panels.d_node_ranges,
        ctx.d_block_targets, ctx.d_block_sources, ctx.n_blocks, Val(reg),
        Val(potential))
    return ls
end

#------- 052h reverse leg: Stages B/C/D (particles→panels, LH dual channel) -------#
#
# The reverse leg carries point-vortex sources, so every expansion is
# Lamb-Helmholtz dual channel in the STACKED row basis of the LH operator
# tables (cross_stencil_host.jl 052h header): rows `1:D` = φ, `D+1:2D` = χ,
# `D = (P+1)(P+2)` per channel. Storage is stacked (the D-agnostic M2M/L2L
# kernels and the new LH M2L kernel run dense over `2D` rows); the radix
# vortex B2M kernel and the flat L2B evaluator see the channels through
# strided row views. Multipoles live on the PARTICLE node set, locals on the
# PANEL node set; the output is evaluated at panel control points.

"Gather sorted source rows (1:3 position, 5:7 strength) by the occupancy perm."
function _cross_gather_sorted_kernel!(dst, src, perm, n)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > n && return nothing
    @inbounds begin
        col = Int(perm[i])
        dst[1, i] = src[1, col]
        dst[2, i] = src[2, col]
        dst[3, i] = src[3, col]
        dst[5, i] = src[5, col]
        dst[6, i] = src[6, col]
        dst[7, i] = src[7, col]
    end
    return nothing
end

# `_cross_m2l_kernel!` with the LH shifted separable rescaling: distinct row
# and column scale tables (χ rows carry degree n+1, χ columns n-1 — see
# cross_m2l_level_scales_lh) instead of the single phi table.
function _cross_m2l_lh_kernel!(locals, multipoles, ops, class_slot, scale2_row,
        scale2_col, pow2lvl, route_levels, route_targets, route_sources,
        route_class, n_routes, D2)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    idx > n_routes * D2 && return nothing
    @inbounds begin
        r = (idx - 1) ÷ D2 + 1
        row = (idx - 1) % D2 + 1
        slot = Int(class_slot[route_class[r]])
        slot == 0 && return nothing   # defensively skip (far mask guarantees > 0)
        L1 = Int(route_levels[r]) + 1
        source = route_sources[r]
        target = route_targets[r]
        acc = zero(eltype(locals))
        for j in 1:D2
            acc = muladd(ops[row, j, slot],
                multipoles[j, source] * scale2_col[j, L1], acc)
        end
        acc *= scale2_row[row, L1] * pow2lvl[L1]
        CUDA.@atomic locals[row, target] += acc
    end
    return nothing
end

"""
Device reverse-leg state (052h): stacked LH multipoles over the PARTICLE node
set, stacked LH locals over the PANEL node set, the uploaded LH operator
tables (M2M octants, slot-compacted reference-level M2L + shifted separable
scales, L2L octants), the gathered sorted particle source buffer, and the
4-row (potential, gradient) output at panel control points in ORIGINAL panel
column order.
"""
mutable struct DeviceCrossReverseState{TF}
    P::Int
    D::Int   # per-channel packed length (P+1)(P+2); stacked arrays carry 2D
    h0_ops::TF
    # Stage B
    d_multipoles::CUDA.CuMatrix{TF}        # 2D × particles.max_nodes
    d_m2m_ops::CUDA.CuArray{TF,4}          # 2D × 2D × 8 × ell_x
    d_sorted_particles::CUDA.CuMatrix{TF}  # 7 × n_particles, sorted order
    d_leaf_to_node::CUDA.CuVector{Int32}
    # Stage C
    d_m2l_ops::CUDA.CuArray{TF,3}          # 2D × 2D × n_slots (ref level)
    d_class_slot::CUDA.CuVector{Int32}
    d_scale2_row::CUDA.CuMatrix{TF}
    d_scale2_col::CUDA.CuMatrix{TF}
    d_pow2lvl::CUDA.CuVector{TF}
    # Stage D
    d_l2l_ops::CUDA.CuArray{TF,4}          # 2D × 2D × 8 × ell_x
    d_locals::CUDA.CuMatrix{TF}            # 2D × panels.max_nodes
    d_out::CUDA.CuMatrix{TF}               # 4 × n_panels, original order
end

function device_cross_reverse_state(ctx::DeviceCrossProducerContext{TF},
        P::Integer) where {TF}
    ctx.build_reverse || throw(ArgumentError(
        "device_cross_reverse_state requires a producer context constructed " *
        "with build_reverse = true"))
    TF === Float64 || throw(ArgumentError(
        "the reverse-leg vortex B2M kernel scratch is Float64-only; build " *
        "the producer context with Float64 positions (got $TF)"))
    D = (Int(P) + 1) * (Int(P) + 2)
    m2l_ops, class_slot = cross_m2l_operators_lh(P, ctx.h0, ctx.ct)
    s2r, s2c, p2l = cross_m2l_level_scales_lh(P, ctx.ell_x)
    max_leaf = min(1 << (3 * ctx.ell_x), max(ctx.particles.n, 1))
    return DeviceCrossReverseState{TF}(Int(P), D, ctx.h0,
        CUDA.zeros(TF, 2 * D, ctx.particles.max_nodes),
        CUDA.CuArray{TF}(cross_m2m_operators_lh(P, ctx.h0, ctx.ell_x)),
        CUDA.zeros(TF, 7, ctx.particles.n),
        CUDA.zeros(Int32, max_leaf),
        CUDA.CuArray{TF}(m2l_ops),
        CUDA.CuArray{Int32}(class_slot),
        CUDA.CuArray{TF}(s2r),
        CUDA.CuArray{TF}(s2c),
        CUDA.CuArray{TF}(p2l),
        CUDA.CuArray{TF}(cross_l2l_operators_lh(P, ctx.h0, ctx.ell_x)),
        CUDA.zeros(TF, 2 * D, ctx.panels.max_nodes),
        CUDA.zeros(TF, 4, ctx.panels.n))
end

"""
    refresh_cross_reverse_multipoles!(rs, ctx, d_particle_buffer)

Reverse Stage-B per-step device pass: point-vortex B2M at the cross leaf
cells of the PARTICLE occupancy + dense octant-class LH M2M up the particle
node set. Requires `refresh_cross_producers!` to have run this step.
`d_particle_buffer` is a device matrix with ≥ 7 rows, columns = particles in
original order, rows 1:3 = position and 5:7 = vector strength (the standard
FMM source-buffer layout the radix vortex B2M consumes).
"""
function refresh_cross_reverse_multipoles!(rs::DeviceCrossReverseState{TF},
        ctx::DeviceCrossProducerContext{TF}, d_particle_buffer) where {TF}
    occ = ctx.particles
    ell_x = ctx.ell_x
    if rs.h0_ops != ctx.h0   # defensive: ctx.h0 is frozen post-construction
        copyto!(rs.d_m2m_ops, cross_m2m_operators_lh(rs.P, ctx.h0, ell_x))
        m2l_ops, _ = cross_m2l_operators_lh(rs.P, ctx.h0, ctx.ct)
        copyto!(rs.d_m2l_ops, m2l_ops)
        copyto!(rs.d_l2l_ops, cross_l2l_operators_lh(rs.P, ctx.h0, ell_x))
        rs.h0_ops = ctx.h0
    end
    (size(d_particle_buffer, 1) >= 7 && size(d_particle_buffer, 2) == occ.n) ||
        throw(ArgumentError("particle buffer must be ≥7 × n_particles " *
            "(got $(size(d_particle_buffer)); n_particles = $(occ.n))"))
    fill!(rs.d_multipoles, zero(TF))
    occ.n > 0 || return rs
    threads = 256
    leaf_first = occ.level_offsets[ell_x + 1] + 1
    n_leaf = occ.level_offsets[ell_x + 2] - occ.level_offsets[ell_x + 1]
    n_leaf > 0 || return rs
    # leaf-node membership + parent links (idempotent with the forward Stage B
    # equivalents on the OTHER occupancy; the reverse leg computes its own)
    CUDA.@cuda threads=threads blocks=cld(n_leaf, threads) _cross_body_node_kernel!(
        occ.d_body_node, occ.d_node_ranges, leaf_first, n_leaf)
    max_per_level = maximum(occ.level_offsets[L + 2] - occ.level_offsets[L + 1]
        for L in 0:ell_x)
    CUDA.@cuda threads=threads blocks=(cld(max_per_level, threads), ell_x + 1) _cuda_parent_index_kernel!(
        occ.d_parent_index, occ.d_node_keys, occ.d_level_offsets, ell_x, 0)
    # gather the sorted 7-row source buffer, then reuse the radix vortex B2M
    # kernel bit-for-bit (block per cell, body-parallel reduction)
    CUDA.@cuda threads=threads blocks=cld(occ.n, threads) _cross_gather_sorted_kernel!(
        rs.d_sorted_particles, d_particle_buffer, occ.d_perm, occ.n)
    copyto!(view(rs.d_leaf_to_node, 1:n_leaf),
        collect(Int32, leaf_first:leaf_first + n_leaf - 1))
    D = rs.D
    phi = view(rs.d_multipoles, 1:D, :)
    chi = view(rs.d_multipoles, D + 1:2 * D, :)
    CUDA.@cuda threads=CUDA_B2M_BLOCK blocks=n_leaf _cuda_b2m_vortex_leaf_nodes_kernel!(
        phi, chi, rs.d_sorted_particles,
        view(occ.d_node_centers, :, leaf_first:leaf_first + n_leaf - 1),
        view(occ.d_node_ranges, :, leaf_first:leaf_first + n_leaf - 1),
        rs.d_leaf_to_node, rs.P, rs.P, n_leaf)
    # upward: dense octant-class LH M2M, bottom-up (D-agnostic kernel, 2D rows)
    for Lc in ell_x:-1:1
        first_child = occ.level_offsets[Lc + 1] + 1
        n_children = occ.level_offsets[Lc + 2] - occ.level_offsets[Lc + 1]
        n_children > 0 || continue
        CUDA.@cuda threads=threads blocks=cld(n_children * 2 * D, threads) _cross_m2m_kernel!(
            rs.d_multipoles, rs.d_m2m_ops, occ.d_parent_index, occ.d_node_coords,
            first_child, n_children, Lc, 2 * D)
    end
    return rs
end

"""
    refresh_cross_reverse_locals!(rs, ctx)

Reverse Stage-C per-step device pass: LH M2L over the reverse far-route list
(particle nodes → panel nodes). Zero routes publishes zeroed locals and skips
every launch (052g discipline).
"""
function refresh_cross_reverse_locals!(rs::DeviceCrossReverseState{TF},
        ctx::DeviceCrossProducerContext{TF}) where {TF}
    fill!(rs.d_locals, zero(TF))
    ctx.n_rev_routes > 0 || return rs
    D2 = 2 * rs.D
    threads = 256
    CUDA.@cuda threads=threads blocks=cld(ctx.n_rev_routes * D2, threads) _cross_m2l_lh_kernel!(
        rs.d_locals, rs.d_multipoles, rs.d_m2l_ops, rs.d_class_slot,
        rs.d_scale2_row, rs.d_scale2_col, rs.d_pow2lvl,
        ctx.d_rev_route_levels, ctx.d_rev_route_targets,
        ctx.d_rev_route_sources, ctx.d_rev_route_class, ctx.n_rev_routes, D2)
    return rs
end

"""
    finish_cross_reverse_locals!(rs, ctx, d_panel_positions)

Reverse Stage-D per-step device pass: dense octant-class LH L2L top-down over
the PANEL node set (child levels `3:ell_x`), then leaf L2B at the panel
control points into `rs.d_out` (rows potential, gx, gy, gz; columns = panels
in original order). Requires `refresh_cross_reverse_locals!` this step.
`d_panel_positions` must be the same `3 × n` device matrix the producers were
refreshed with.
"""
function finish_cross_reverse_locals!(rs::DeviceCrossReverseState{TF},
        ctx::DeviceCrossProducerContext{TF}, d_panel_positions) where {TF}
    occ = ctx.panels
    ell_x = ctx.ell_x
    threads = 256
    D2 = 2 * rs.D
    fill!(rs.d_out, zero(TF))
    occ.n > 0 || return rs
    leaf_first = occ.level_offsets[ell_x + 1] + 1
    n_leaf = occ.level_offsets[ell_x + 2] - occ.level_offsets[ell_x + 1]
    n_leaf > 0 || return rs
    # panel-side body_node + parent links (idempotent; the reverse leg must
    # not assume the forward Stage B ran this step)
    CUDA.@cuda threads=threads blocks=cld(n_leaf, threads) _cross_body_node_kernel!(
        occ.d_body_node, occ.d_node_ranges, leaf_first, n_leaf)
    max_per_level = maximum(occ.level_offsets[L + 2] - occ.level_offsets[L + 1]
        for L in 0:ell_x)
    max_per_level > 0 || return rs
    CUDA.@cuda threads=threads blocks=(cld(max_per_level, threads), ell_x + 1) _cuda_parent_index_kernel!(
        occ.d_parent_index, occ.d_node_keys, occ.d_level_offsets, ell_x, 0)
    # L2L top-down: locals begin at the coarsest route level 2
    for Lc in 3:ell_x
        first_child = occ.level_offsets[Lc + 1] + 1
        n_children = occ.level_offsets[Lc + 2] - occ.level_offsets[Lc + 1]
        n_children > 0 || continue
        CUDA.@cuda threads=threads blocks=cld(n_children * D2, threads) _cross_l2l_kernel!(
            rs.d_locals, rs.d_l2l_ops, occ.d_parent_index, occ.d_node_coords,
            first_child, n_children, Lc, D2)
    end
    # leaf L2B at panel control points, dual channel
    CUDA.@cuda threads=threads blocks=cld(occ.n, threads) _cross_l2b_kernel!(
        rs.d_out, d_panel_positions, occ.d_perm, occ.d_body_node,
        occ.d_node_centers, view(rs.d_locals, 1:rs.D, :),
        view(rs.d_locals, rs.D + 1:D2, :), rs.P, occ.n, Val(true))
    return rs
end

#------- 052h reverse leg: Stage E (vortex near blocks) -------#

# `_cross_near_kernel!` with point-vortex sources: block-per-route over the
# REVERSE block list, tiling the PARTICLE sources (rows 1:3 position,
# 4:6 strength, 7 sigma in shared memory) against panel-control-point
# targets. Velocity only (the vortex kernel has no scalar potential).
# `REG` = true applies the gaussianerf regularization g(ρ = r/σ) exactly as
# the radix near field does; false is the singular kernel.
function _cross_near_vortex_kernel!(out, particle_buffer, sigma_row,
        positions, t_perm, s_perm, t_node_ranges, s_node_ranges,
        block_targets, block_sources, n_blocks, ::Val{REG}) where {REG}
    T = eltype(out)
    tid = threadIdx().x
    sh = CUDA.CuStaticSharedArray(T, (7, _CROSS_NEAR_TILE))
    b = blockIdx().x
    @inbounds while b <= n_blocks
        tnode = Int(block_targets[b])
        snode = Int(block_sources[b])
        tfirst = Int(t_node_ranges[1, tnode])
        tcount = Int(t_node_ranges[2, tnode])
        sfirst = Int(s_node_ranges[1, snode])
        scount = Int(s_node_ranges[2, snode])
        ntiles = cld(scount, _CROSS_NEAR_TILE)
        for tc in 1:cld(tcount, blockDim().x)
            ti = tfirst + (tc - 1) * blockDim().x + tid - 1
            active = ti <= tfirst + tcount - 1
            col = 1
            xt = zero(T); yt = zero(T); zt = zero(T)
            if active
                col = Int(t_perm[ti])
                xt = positions[1, col]
                yt = positions[2, col]
                zt = positions[3, col]
            end
            ux = zero(T); uy = zero(T); uz = zero(T)
            for t in 1:ntiles
                q0 = (t - 1) * _CROSS_NEAR_TILE
                q = q0 + tid
                if tid <= _CROSS_NEAR_TILE && q <= scount
                    scol = Int(s_perm[sfirst + q - 1])
                    sh[1, tid] = particle_buffer[1, scol]
                    sh[2, tid] = particle_buffer[2, scol]
                    sh[3, tid] = particle_buffer[3, scol]
                    sh[4, tid] = particle_buffer[5, scol]
                    sh[5, tid] = particle_buffer[6, scol]
                    sh[6, tid] = particle_buffer[7, scol]
                    sh[7, tid] = REG ? particle_buffer[sigma_row, scol] : one(T)
                end
                CUDA.sync_threads()
                if active
                    for k in 1:min(_CROSS_NEAR_TILE, scount - q0)
                        dx = xt - sh[1, k]
                        dy = yt - sh[2, k]
                        dz = zt - sh[3, k]
                        r2 = dx * dx + dy * dy + dz * dz
                        r2 > zero(T) || continue
                        invr = _cuda_fast_rsqrt(r2)
                        g = one(T)
                        if REG
                            rho = r2 * invr / sh[7, k]
                            g, _ = _gaussianerf_g_h(rho, Val(:shipped))
                        end
                        _, vx, vy, vz = _vortex_pair_ug(dx, dy, dz, invr,
                            sh[4, k], sh[5, k], sh[6, k], g)
                        ux += vx
                        uy += vy
                        uz += vz
                    end
                end
                CUDA.sync_threads()
            end
            if active
                CUDA.@atomic out[2, col] += ux
                CUDA.@atomic out[3, col] += uy
                CUDA.@atomic out[4, col] += uz
            end
        end
        b += gridDim().x
    end
    return nothing
end

"""
    apply_cross_reverse_near!(rs, ctx, d_particle_buffer, d_panel_positions;
        sigma_row=8, reg=true)

Reverse Stage-E per-step device pass: direct point-vortex evaluation of the
reverse demoted + near blocks, accumulating velocity into `rs.d_out` on top
of the Stage-D far field. `d_particle_buffer` is the same ≥7-row matrix as
reverse Stage B (rows 1:3 position, 5:7 strength); `sigma_row` names the
smoothing-radius row consumed when `reg = true` (gaussianerf, matching the
radix near field); `reg = false` is the singular kernel. Zero blocks skips
every launch (052g discipline).
"""
function apply_cross_reverse_near!(rs::DeviceCrossReverseState{TF},
        ctx::DeviceCrossProducerContext{TF}, d_particle_buffer,
        d_panel_positions; sigma_row::Int=8, reg::Bool=true) where {TF}
    ctx.n_rev_blocks > 0 || return rs
    (size(d_particle_buffer, 1) >= (reg ? sigma_row : 7) &&
        size(d_particle_buffer, 2) == ctx.particles.n) ||
        throw(ArgumentError("particle buffer must be ≥$(reg ? sigma_row : 7)" *
            " × n_particles (got $(size(d_particle_buffer)); n_particles = " *
            "$(ctx.particles.n))"))
    threads = _CROSS_NEAR_TILE
    blocks = min(ctx.n_rev_blocks, _RECT_MAX_BLOCKS)
    CUDA.@cuda threads=threads blocks=blocks _cross_near_vortex_kernel!(
        rs.d_out, d_particle_buffer, sigma_row, d_panel_positions,
        ctx.panels.d_perm, ctx.particles.d_perm,
        ctx.panels.d_node_ranges, ctx.particles.d_node_ranges,
        ctx.d_rev_block_targets, ctx.d_rev_block_sources, ctx.n_rev_blocks,
        Val(reg))
    return rs
end

#------- oracle download -------#

"""
    download_cross_lists(ctx)

Host copies of the producer outputs for the `PANEL_INFLUENCE_FMM_XVERIFY`
host-as-oracle bit-compare: route and block lists, node keys/ranges and level
offsets for both sets, and the two body permutations.
"""
function download_cross_lists(ctx::DeviceCrossProducerContext)
    nr = ctx.n_routes
    nb = ctx.n_blocks
    occ_host(occ) = (;
        node_keys = Array(occ.d_node_keys)[1:occ.level_offsets[end]],
        node_ranges = Array(occ.d_node_ranges)[:, 1:occ.level_offsets[end]],
        level_offsets = copy(occ.level_offsets),
        perm = Array(occ.d_perm))
    return (;
        routes = (; levels = Array(ctx.d_route_levels)[1:nr],
            offsets = Array(ctx.d_route_offsets)[:, 1:nr],
            targets = Array(ctx.d_route_targets)[1:nr],
            sources = Array(ctx.d_route_sources)[1:nr],
            class = Array(ctx.d_route_class)[1:nr]),
        blocks = (; levels = Array(ctx.d_block_levels)[1:nb],
            offsets = Array(ctx.d_block_offsets)[:, 1:nb],
            targets = Array(ctx.d_block_targets)[1:nb],
            sources = Array(ctx.d_block_sources)[1:nb],
            class = Array(ctx.d_block_class)[1:nb],
            n_demoted = ctx.n_demoted),
        rev_routes = (; levels = Array(ctx.d_rev_route_levels)[1:ctx.n_rev_routes],
            offsets = Array(ctx.d_rev_route_offsets)[:, 1:ctx.n_rev_routes],
            targets = Array(ctx.d_rev_route_targets)[1:ctx.n_rev_routes],
            sources = Array(ctx.d_rev_route_sources)[1:ctx.n_rev_routes],
            class = Array(ctx.d_rev_route_class)[1:ctx.n_rev_routes]),
        rev_blocks = (; levels = Array(ctx.d_rev_block_levels)[1:ctx.n_rev_blocks],
            offsets = Array(ctx.d_rev_block_offsets)[:, 1:ctx.n_rev_blocks],
            targets = Array(ctx.d_rev_block_targets)[1:ctx.n_rev_blocks],
            sources = Array(ctx.d_rev_block_sources)[1:ctx.n_rev_blocks],
            class = Array(ctx.d_rev_block_class)[1:ctx.n_rev_blocks],
            n_demoted = ctx.n_rev_demoted),
        panels = occ_host(ctx.panels),
        particles = occ_host(ctx.particles),
        x_min = ctx.x_min, h0 = ctx.h0,
        needs_rebuild = ctx.needs_rebuild)
end
