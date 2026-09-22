#------- extra target / source systems on the radix path -------#
#
# The resident radix lifecycle evaluates the cache's systems on themselves. A
# coupled solver (FLOWPanel, ActuatorLines) also needs the same call to deliver
# the field at a few hundred extra points (blade probes, wake-ring nodes) and
# to apply a few hundred extra sources (bound segments, ring filaments) to the
# resident bodies. Those systems are tiny, so no multipoles are built for them:
# every pairing that touches one is a direct rectangular evaluation.
#
# Given `fmm!(targets, sources, cache::RadixFMMCache)`:
#   main          = the cache's systems, the FIRST `n_systems` targets
#   extra targets = the remaining targets (evaluated from the main systems)
#   extra sources = sources that are not the main systems
#
# The main systems evaluate on themselves only when they ALSO appear in
# `sources` (all of them, at the front, in order). With the main systems absent
# from `sources` the lifecycle is skipped entirely and the call delivers the
# extra sources alone -- the "body on wake" direction of a coupled solve, where
# the self-induction was already evaluated earlier in the step and must not be
# recomputed over a field that has changed since.
#
#   main  <- main           resident FMM lifecycle (only when self-inducing)
#   main  <- extra sources  direct, accumulated into state.output in slot order
#                           BEFORE finalize (one write-back to the user system)
#   extra <- main           direct from the packed source bodies
#
# Extra sources are applied to the cache's systems ONLY. An extra target sees
# the resident field and nothing else: a coupled solver evaluates its own
# body-on-probe terms itself (with whatever self-exclusion its solve needs),
# so applying the extra sources there would double-count them. It follows that
# a non-self-inducing call (main systems absent from `sources`) evaluates no
# extra targets at all -- there is no resident field acting as a source.
#
# Contract for an extra SOURCE system: the usual `get_n_bodies`,
# `data_per_body`, `source_system_to_buffer!`, plus `direct_kernel(system)`
# returning an isbits functor with
#
#   _extra_pair_ug(kernel, tx, ty, tz, source_buffer, j)  -> (u, gx, gy, gz)
#   _extra_pair_ugh(kernel, tx, ty, tz, source_buffer, j) -> the 13-tuple
#
# where `(tx, ty, tz)` is the target position and `j` the packed column. The
# same functor runs on the host and on the device.
#
# Contract for an extra TARGET system: the usual target interface
# (`get_n_bodies`, `get_position`, `buffer_to_target_system!`).

struct RadixSystemSplit{MT<:Tuple,ET<:Tuple,ES<:Tuple}
    main::MT
    extra_targets::ET
    extra_sources::ES
    self_induce::Bool                # the main systems are sources as well
    main_index::Vector{Int}          # index of each main system within `targets`
    extra_target_index::Vector{Int}  # index of each extra target within `targets`
end

_is_in(x, tup::Tuple) = any(y -> y === x, tup)

function _split_radix_systems(n_systems::Integer, targets::Tuple, sources::Tuple)
    n = Int(n_systems)
    length(targets) >= n || throw(ArgumentError(
        "the radix fmm! path expects the cache's $n system(s) first in " *
        "target_systems; got $(length(targets)) target system(s)"))
    main = ntuple(i -> targets[i], n)
    n_in_sources = count(m -> _is_in(m, sources), main)
    self_induce = n_in_sources == n
    n_in_sources == 0 || self_induce || throw(ArgumentError(
        "the radix fmm! path requires the cache's systems to appear in " *
        "source_systems either all together (self-inducing) or not at all " *
        "(extra sources only); $(n_in_sources) of $n were found"))
    if self_induce
        all(sources[i] === main[i] for i in 1:n) || throw(ArgumentError(
            "the radix fmm! path requires the cache's systems to appear first " *
            "in source_systems, in the same order as in target_systems"))
    end
    extra_sources = Tuple(s for s in sources if !_is_in(s, main))
    return RadixSystemSplit(main, ntuple(i -> targets[n + i], length(targets) - n),
        extra_sources, self_induce, collect(1:n),
        collect(n+1:length(targets)))
end

_extra_pair_ug(kernel, tx, ty, tz, source_buffer, j) = throw(ArgumentError(
    "an extra source system on the radix path must define " *
    "FastMultipole._extra_pair_ug(::$(typeof(kernel)), tx, ty, tz, source_buffer, j); " *
    "return direct_kernel(system) as an isbits functor"))

_extra_pair_ugh(kernel, tx, ty, tz, source_buffer, j) = throw(ArgumentError(
    "hessian output from an extra source system requires " *
    "FastMultipole._extra_pair_ugh(::$(typeof(kernel)), tx, ty, tz, source_buffer, j)"))

"""
    _extra_pair_has_hessian(kernel) -> Bool

Whether an extra source's functor defines `_extra_pair_ugh`. Default `false`:
the source contributes velocity only, and the main systems' hessian rows are
left to the resident lifecycle even when the cache carries them. A functor
that defines `_extra_pair_ugh` overloads this to `true`. (A throwing default
inside a device kernel is a compile error, not a runtime one.)
"""
_extra_pair_has_hessian(kernel) = false

# Regularized vortex particles as an all-pairs extra source (FLOWVPM's oversize
# particles, 2026-09-21): the buffer is the particle source layout -- rows 1:3
# position, 5:7 strength, `sigma_row` the core -- so the resident near-field pair
# math applies verbatim. A coincident pair (r = 0: the source is also a target)
# contributes nothing, as in the near-field kernels.
@inline function _extra_pair_ug(kernel::Union{PartitionedVortex,RegularizedVortex},
        tx, ty, tz, source_buffer, j)
    T = typeof(tx)
    @inbounds dx = tx - source_buffer[1, j]
    @inbounds dy = ty - source_buffer[2, j]
    @inbounds dz = tz - source_buffer[3, j]
    r2 = dx * dx + dy * dy + dz * dz
    r2 > zero(T) || return zero(T), zero(T), zero(T), zero(T)
    return _direct_pair_ug(kernel, dx, dy, dz, r2, inv(sqrt(r2)), source_buffer, j)
end
@inline function _extra_pair_ugh(kernel::Union{PartitionedVortex,RegularizedVortex},
        tx, ty, tz, source_buffer, j)
    T = typeof(tx)
    @inbounds dx = tx - source_buffer[1, j]
    @inbounds dy = ty - source_buffer[2, j]
    @inbounds dz = tz - source_buffer[3, j]
    r2 = dx * dx + dy * dy + dz * dz
    z = zero(T)
    r2 > z || return z, z, z, z, z, z, z, z, z, z, z, z, z
    return _direct_pair_ugh(kernel, dx, dy, dz, r2, inv(sqrt(r2)), source_buffer, j)
end
_extra_pair_has_hessian(::Union{PartitionedVortex,RegularizedVortex}) = true

#------- host packing -------#

function _radix_extra_target_positions(::Type{TF}, system) where TF
    n = get_n_bodies(system)
    x = Matrix{TF}(undef, 3, n)
    @inbounds for i in 1:n
        p = get_position(system, i)
        x[1, i] = p[1]; x[2, i] = p[2]; x[3, i] = p[3]
    end
    return x
end

function _radix_extra_source_buffer(::Type{TF}, system) where TF
    n = get_n_bodies(system)
    buffer = zeros(TF, data_per_body(system), n)
    for i in 1:n
        source_system_to_buffer!(buffer, i, system, i)
    end
    return buffer
end

#------- host evaluators (also the reference for the device kernels) -------#

# targets at `xt` (3 x nt) from the packed main bodies `source_bodies[:, 1:n]`
# through the cache's nearfield functor. ACCUMULATES into `out` (4 or 13 rows).
function _host_extra_targets_from_main!(out::AbstractMatrix{TF}, kernel::AbstractDirectKernel,
        xt::AbstractMatrix, source_bodies, n::Integer, ::Val{HS},
        ghv::Val=Val(:shipped)) where {TF,HS}
    ep = _emits_potential(kernel)
    @inbounds for i in 1:size(xt, 2)
        xi = TF(xt[1, i]); yi = TF(xt[2, i]); zi = TF(xt[3, i])
        for j in 1:n
            dx = xi - source_bodies[1, j]
            dy = yi - source_bodies[2, j]
            dz = zi - source_bodies[3, j]
            r2 = dx * dx + dy * dy + dz * dz
            r2 == zero(TF) && continue
            invr = inv(sqrt(r2))
            if HS
                u, gx, gy, gz, h1, h2, h3, h4, h5, h6, h7, h8, h9 =
                    _direct_pair_ugh(kernel, dx, dy, dz, r2, invr, source_bodies, j, ghv)
                ep && (out[1, i] += u)
                out[2, i] += gx; out[3, i] += gy; out[4, i] += gz
                out[5, i] += h1; out[6, i] += h2; out[7, i] += h3
                out[8, i] += h4; out[9, i] += h5; out[10, i] += h6
                out[11, i] += h7; out[12, i] += h8; out[13, i] += h9
            else
                u, gx, gy, gz = _direct_pair_ug(kernel, dx, dy, dz, r2, invr,
                    source_bodies, j, ghv)
                ep && (out[1, i] += u)
                out[2, i] += gx; out[3, i] += gy; out[4, i] += gz
            end
        end
    end
    return out
end

# targets whose positions are rows 1:3 of `xt` (the packed main bodies in slot
# order) from an extra source's packed buffer. ACCUMULATES into `out`.
function _host_targets_from_extra_source!(out::AbstractMatrix{TF}, kernel,
        xt::AbstractMatrix, nt::Integer, source_buffer::AbstractMatrix,
        ::Val{HS}) where {TF,HS}
    ns = size(source_buffer, 2)
    @inbounds for i in 1:nt
        xi = xt[1, i]; yi = xt[2, i]; zi = xt[3, i]
        for j in 1:ns
            if HS
                u, gx, gy, gz, h1, h2, h3, h4, h5, h6, h7, h8, h9 =
                    _extra_pair_ugh(kernel, xi, yi, zi, source_buffer, j)
                out[1, i] += u
                out[2, i] += gx; out[3, i] += gy; out[4, i] += gz
                out[5, i] += h1; out[6, i] += h2; out[7, i] += h3
                out[8, i] += h4; out[9, i] += h5; out[10, i] += h6
                out[11, i] += h7; out[12, i] += h8; out[13, i] += h9
            else
                u, gx, gy, gz = _extra_pair_ug(kernel, xi, yi, zi, source_buffer, j)
                out[1, i] += u
                out[2, i] += gx; out[3, i] += gy; out[4, i] += gz
            end
        end
    end
    return out
end

#------- host drivers -------#

"""
    _radix_extra_sources_into_output!(state, extra_sources)

Host cache: apply every extra source system to the resident bodies, accumulating
into `state.output` in slot order (the slot positions are rows 1:3 of
`state.source_bodies`). Runs after the lifecycle body and before finalize.
"""
function _radix_extra_sources_into_output!(state::DeviceResidentRadixState{TF},
        extra_sources::Tuple) where TF
    isempty(extra_sources) && return state
    n = state.counts.n_bodies
    hs = size(state.output, 1) >= 13
    for system in extra_sources
        buffer = _radix_extra_source_buffer(TF, system)
        kernel = direct_kernel(system)
        _host_targets_from_extra_source!(state.output, kernel,
            state.source_bodies, n, buffer, Val(hs && _extra_pair_has_hessian(kernel)))
    end
    return state
end

# scatter a 4/13-row rectangular output into the target system through its
# derivatives switch, reusing the resident finalize copier with an identity
# permutation.
function _radix_scatter_extra_target!(::Type{TF}, system, switch, out::AbstractMatrix) where TF
    nt = size(out, 2)
    target_buffer = allocate_target_buffer(TF, system, switch)
    _copy_radix_output_to_host_target_buffer!(target_buffer, out, 1:nt,
        fill(1, nt), 1:nt, 1, switch, nt)
    buffer_to_target!(system, target_buffer, switch, 1:nt)
    return system
end

"""
    bin_resident_extra_targets(xt, grid, n_cells) -> (order, cell_ranges, loose)

Bin target positions `xt` (3 x n) onto the resident grid's OCCUPIED cells.
`order` lists the binned targets sorted by cell, `cell_ranges[:, c]` is
`(first, count)` into `order`, and `loose` holds the targets that fall outside
the box or in a cell no body occupies: those have no local expansion to read
and are summed all-pairs instead.
"""
function bin_resident_extra_targets(xt::AbstractMatrix, grid, n_cells::Integer)
    n = size(xt, 2)
    ell = grid.ell
    cell_keys = Array(grid.cell_keys)      # a device grid keeps its keys on the device
    nk = min(Int(n_cells), length(cell_keys))
    side = 1 << ell
    L = 2 * Float64(grid.h0)
    delta = L / side
    x0 = (Float64(grid.x_min[1]), Float64(grid.x_min[2]), Float64(grid.x_min[3]))
    cell_of = zeros(Int, n)
    loose = Int[]
    @inbounds for i in 1:n
        x = Float64(xt[1, i]); y = Float64(xt[2, i]); z = Float64(xt[3, i])
        # outside the box there is no cell whose expansion converges at the
        # point, so it is loose rather than clamped to an edge cell
        if !(x0[1] <= x < x0[1] + L && x0[2] <= y < x0[2] + L && x0[3] <= z < x0[3] + L)
            push!(loose, i); continue
        end
        ix = clamp(floor(Int, (x - x0[1]) / delta), 0, side - 1)
        iy = clamp(floor(Int, (y - x0[2]) / delta), 0, side - 1)
        iz = clamp(floor(Int, (z - x0[3]) / delta), 0, side - 1)
        key = morton_key(SVector{3,Int}(ix, iy, iz), ell)
        j = searchsortedfirst(view(cell_keys, 1:nk), key)
        if j <= nk && cell_keys[j] == key
            cell_of[i] = j
        else
            push!(loose, i)
        end
    end
    counts = zeros(Int, n_cells)
    @inbounds for i in 1:n
        c = cell_of[i]; c == 0 || (counts[c] += 1)
    end
    cell_ranges = zeros(Int, 2, n_cells)
    cursor = 1
    @inbounds for c in 1:n_cells
        cell_ranges[1, c] = cursor; cell_ranges[2, c] = counts[c]; cursor += counts[c]
    end
    order = zeros(Int, cursor - 1)
    fill!(counts, 0)
    @inbounds for i in 1:n
        c = cell_of[i]; c == 0 && continue
        order[cell_ranges[1, c] + counts[c]] = i; counts[c] += 1
    end
    return order, cell_ranges, loose
end

"""
    _host_extra_targets_tree!(out, state, xt, Val(HS))

Extra targets evaluated the way the resident bodies are: each binned target
reads its cell's local expansion and sums its cell's near source cells
directly; targets the grid cannot place are summed all-pairs. Replaces an
all-pairs sweep over every body, which at sixteen rotors was a third of the
step-start pass. ACCUMULATES into `out`.
"""
function _host_extra_targets_tree!(out::AbstractMatrix{TF},
        state::DeviceResidentRadixState{TF,B,LH}, xt::AbstractMatrix,
        ::Val{HS}) where {TF,B,LH,HS}
    n = Int(state.counts.n_bodies)
    n_cells = Int(state.counts.n_cells)
    kernel = state.options.direct_kernel
    order, t_ranges, loose = bin_resident_extra_targets(xt, state.grid, n_cells)
    orders = state.invariant_cache.basis_info.orders
    ph = phi_slab(state.locals); ch = chi_slab(state.locals)
    leaf_to_node = state.grid.leaf_to_node
    centers = state.cell_centers
    # far field: the cell's local expansion at the target (the L2B step, at a
    # point that is not a body)
    @inbounds for c in 1:n_cells
        cnt = t_ranges[2, c]; cnt == 0 && continue
        node = leaf_to_node[c]
        cx = centers[1, c]; cy = centers[2, c]; cz = centers[3, c]
        for k in t_ranges[1, c]:(t_ranges[1, c] + cnt - 1)
            i = order[k]
            dx = TF(xt[1, i]) - cx; dy = TF(xt[2, i]) - cy; dz = TF(xt[3, i]) - cz
            if HS
                vals = _resident_local_eval_flat_hessian(ph, ch, node, dx, dy, dz,
                    orders.P_phi, orders.P_active, Val(LH))
                for r in 1:13
                    out[r, i] += vals[r]
                end
            else
                sp, gx, gy, gz = _resident_local_eval_flat(ph, ch, node, dx, dy, dz,
                    orders.P_phi, orders.P_active, Val(LH))
                out[1, i] += sp; out[2, i] += gx; out[3, i] += gy; out[4, i] += gz
            end
        end
    end
    # near field: the resident near pairs, the target cell's targets against
    # the source cell's bodies
    bodies = state.source_bodies; b_ranges = state.cell_ranges
    dt = state.direct_targets; ds = state.direct_sources
    ep = _emits_potential(kernel)
    ghv = Val(:shipped)
    @inbounds for p in 1:Int(state.counts.n_direct)
        c = Int(dt[p]); s = Int(ds[p])
        tcnt = t_ranges[2, c]; tcnt == 0 && continue
        sfirst = Int(b_ranges[1, s]); scnt = Int(b_ranges[2, s]); scnt == 0 && continue
        for k in t_ranges[1, c]:(t_ranges[1, c] + tcnt - 1)
            i = order[k]
            xi = TF(xt[1, i]); yi = TF(xt[2, i]); zi = TF(xt[3, i])
            for j in sfirst:(sfirst + scnt - 1)
                dx = xi - bodies[1, j]; dy = yi - bodies[2, j]; dz = zi - bodies[3, j]
                r2 = dx * dx + dy * dy + dz * dz
                r2 == zero(TF) && continue
                invr = inv(sqrt(r2))
                if HS
                    u, gx, gy, gz, h1, h2, h3, h4, h5, h6, h7, h8, h9 =
                        _direct_pair_ugh(kernel, dx, dy, dz, r2, invr, bodies, j, ghv)
                    ep && (out[1, i] += u)
                    out[2, i] += gx; out[3, i] += gy; out[4, i] += gz
                    out[5, i] += h1; out[6, i] += h2; out[7, i] += h3
                    out[8, i] += h4; out[9, i] += h5; out[10, i] += h6
                    out[11, i] += h7; out[12, i] += h8; out[13, i] += h9
                else
                    u, gx, gy, gz = _direct_pair_ug(kernel, dx, dy, dz, r2, invr, bodies, j, ghv)
                    ep && (out[1, i] += u)
                    out[2, i] += gx; out[3, i] += gy; out[4, i] += gz
                end
            end
        end
    end
    if !isempty(loose)
        ol = zeros(TF, size(out, 1), length(loose))
        _host_extra_targets_from_main!(ol, kernel, xt[:, loose], bodies, n, Val(HS))
        out[:, loose] .+= ol
    end
    return out
end

"""
    _radix_extra_targets_evaluate!(state, extra_targets, switches)

Host cache: evaluate every extra target system from the resident bodies, then
write back through the target's switch.
"""
function _radix_extra_targets_evaluate!(state::DeviceResidentRadixState{TF},
        extra_targets::Tuple, switches::Tuple) where TF
    isempty(extra_targets) && return state
    n = state.counts.n_bodies
    for (system, switch) in zip(extra_targets, switches)
        HS = !isempty(hessian_range(switch))
        HS && size(state.output, 1) < 13 && throw(ArgumentError(
            "hessian output requested for an extra target system but the cache " *
            "was built with hessian=false"))
        xt = _radix_extra_target_positions(TF, system)
        out = zeros(TF, HS ? 13 : 4, size(xt, 2))
        if n > 0
            # the same default as the device step: all-pairs unless
            # :KA_EXTRA_TARGETS_GRID opts the probes into the grid path. The
            # host went through the grid unconditionally after fd15433 while
            # the device was reverted to all-pairs (d5bcb93), so the two arms
            # of a device gate differed by the grid's truncation (~1e-6 on
            # the coupled-rotor gate) and the extra-systems Metal suite
            # failed on the host arm.
            r = _radix_setting_ref(:KA_EXTRA_TARGETS_GRID)
            if r !== nothing && r[]
                _host_extra_targets_tree!(out, state, xt, Val(HS))
            else
                _host_extra_targets_from_main!(out, state.options.direct_kernel, xt,
                    state.source_bodies, n, Val(HS))
            end
        end
        _radix_scatter_extra_target!(TF, system, switch, out)
    end
    return state
end
