#=
Extra source systems carried by the resident tree.

An extra source system is otherwise summed directly against every resident
body, which costs `n_extra * n_bodies` and so grows one power of problem size
faster than the tree pass beside it. Binning the extra bodies into the SAME
grid the resident bodies already occupy lets them contribute to the leaf
multipoles, so their far field rides the existing M2M/M2L/L2B pipeline and
only the near cell pairs stay direct -- and those are the pairs the resident
near field already enumerates, so no new geometry is built.

Every (resident body, extra body) pair is still counted exactly once: by the
near sweep over `direct_targets`/`direct_sources` when the two cells are near,
and by the extra body's cell multipole otherwise. An extra body is held out of
the tree and summed against every resident body ("loose") when

  * its cell holds no resident body, so there is no node to carry a multipole,
  * or its own extent is not small against a cell, since a multipole about the
    cell center is only valid outside the source's extent.
=#

"""
    ResidentExtraSource(system, buffer, cell_ranges, loose)

An extra source system binned onto a resident cache's grid. `buffer` holds
every body packed and reordered so that the bodies of one cell are contiguous,
`cell_ranges[:, i]` is the `(first, count)` of cell `i` in the cache's own cell
order, and `loose` lists the columns of `buffer` that must be summed directly
against every resident body. See [`bin_resident_extra_source`](@ref).
"""
struct ResidentExtraSource{TF,S}
    system::S
    buffer::Matrix{TF}
    cell_ranges::Matrix{Int}
    loose::Vector{Int}
end

"""
    bin_resident_extra_source(TF, system, grid, n_cells; extent_fraction = 0.5)

Pack `system` and sort it onto `grid`'s cells, returning a
[`ResidentExtraSource`](@ref). A body goes to the loose set when its cell is
absent from the grid or when its radius (packed row 4) exceeds
`extent_fraction` of a cell width.
"""
function bin_resident_extra_source(::Type{TF}, system,
        grid::Union{RadixGrid,DeviceRadixGrid}, n_cells::Integer;
        extent_fraction = 0.5) where TF
    n = get_n_bodies(system)
    raw = zeros(TF, data_per_body(system), n)
    for i in 1:n
        source_system_to_buffer!(raw, i, system, i)
    end

    ell = grid.ell
    cell_keys = Array(grid.cell_keys)      # a device grid keeps its keys on the device
    side = 1 << ell
    delta = 2 * Float64(grid.h0) / side
    max_radius = extent_fraction * delta
    cell_of = zeros(Int, n)
    loose = Int[]
    @inbounds for i in 1:n
        if Float64(raw[4, i]) > max_radius
            push!(loose, i)
            continue
        end
        ix = clamp(floor(Int, (Float64(raw[1, i]) - Float64(grid.x_min[1])) / delta), 0, side - 1)
        iy = clamp(floor(Int, (Float64(raw[2, i]) - Float64(grid.x_min[2])) / delta), 0, side - 1)
        iz = clamp(floor(Int, (Float64(raw[3, i]) - Float64(grid.x_min[3])) / delta), 0, side - 1)
        key = morton_key(SVector{3,Int}(ix, iy, iz), ell)
        j = searchsortedfirst(view(cell_keys, 1:min(n_cells, length(cell_keys))), key)
        if j <= n_cells && j <= length(cell_keys) && cell_keys[j] == key
            cell_of[i] = j
        else
            push!(loose, i)
        end
    end

    # counting sort of the binned bodies by cell
    counts = zeros(Int, n_cells)
    @inbounds for i in 1:n
        c = cell_of[i]
        c == 0 || (counts[c] += 1)
    end
    cell_ranges = zeros(Int, 2, n_cells)
    cursor = 1
    @inbounds for c in 1:n_cells
        cell_ranges[1, c] = cursor
        cell_ranges[2, c] = counts[c]
        cursor += counts[c]
    end
    n_binned = cursor - 1
    buffer = zeros(TF, size(raw, 1), n_binned)
    fill_cursor = copy(view(cell_ranges, 1, :))
    @inbounds for i in 1:n
        c = cell_of[i]
        c == 0 && continue
        slot = fill_cursor[c]
        fill_cursor[c] = slot + 1
        for r in axes(raw, 1)
            buffer[r, slot] = raw[r, i]
        end
    end

    loose_buffer = zeros(TF, size(raw, 1), length(loose))
    @inbounds for (k, i) in enumerate(loose)
        for r in axes(raw, 1)
            loose_buffer[r, k] = raw[r, i]
        end
    end
    return ResidentExtraSource{TF,typeof(system)}(system, buffer, cell_ranges, Int[]),
           loose_buffer
end

"""
    resident_extra_b2m!(state, extra)

Accumulate the multipole of every binned extra body into its cell's leaf node,
on top of the resident bodies' own multipoles.
"""
function resident_extra_b2m!(state::DeviceResidentRadixState{TF,B,LH},
        extra::ResidentExtraSource) where {TF,B,LH}
    orders = state.invariant_cache.basis_info.orders
    _resident_extra_b2m_kernel!(phi_slab(state.multipoles), chi_slab(state.multipoles),
        extra.system, extra.buffer, extra.cell_ranges, state.cell_centers,
        state.grid.leaf_to_node, orders.P_phi, orders.P_active,
        Int(state.counts.n_cells), Val(LH))
    return state
end

"""
    _resident_extra_b2m_kernel!(ph, ch, system, buffer, cell_ranges, cell_centers,
                                leaf_to_node, P_phi, P_chi, n_cells, Val(LH))

Accumulate each cell's bodies into that cell's leaf node, in the resident slab
layout. The multipole itself comes from the same `body_to_multipole!` the host
octree uses, so any body type it supports is supported here; only the index
map from its `[reim, channel, harmonic]` layout to the slab's `[row, node]` is
new.
"""
function _resident_extra_b2m_kernel!(ph::AbstractMatrix{TF}, ch, system, buffer,
        cell_ranges, cell_centers, leaf_to_node, P_phi::Int, P_chi::Int,
        n_cells::Int, ::Val{LH}) where {TF,LH}
    P = max(P_phi, P_chi)
    coefficients = initialize_expansion(P, TF)
    harmonics = initialize_harmonics(P, TF)
    @inbounds for i_cell in 1:n_cells
        first = cell_ranges[1, i_cell]
        count = cell_ranges[2, i_cell]
        count == 0 && continue
        node = leaf_to_node[i_cell]
        center = SVector{3,TF}(cell_centers[1, i_cell], cell_centers[2, i_cell],
                               cell_centers[3, i_cell])
        coefficients .= zero(TF)
        # the element type leads: the system-first form is a fallback that warns
        # and writes nothing
        body_to_multipole!(body_type(system), system, coefficients, buffer, center,
            first:(first + count - 1), harmonics, P)
        for n in 0:P_phi, m in 0:n
            i = harmonic_index(n, m)
            row = flat_basis_index(n, m, 1)
            ph[row, node] += coefficients[1, 1, i]
            ph[row + 1, node] += coefficients[2, 1, i]
        end
        if LH
            for n in 1:P_chi, m in 0:n
                i = harmonic_index(n, m)
                row = flat_basis_index(n, m, 1)
                ch[row, node] += coefficients[1, 2, i]
                ch[row + 1, node] += coefficients[2, 2, i]
            end
        end
    end
    return ph
end

"""
    resident_extra_near!(state, extra, kernel)

Sum the binned extra bodies of every near source cell against the resident
bodies of the paired target cell, the same pair list the resident near field
walks.
"""
function resident_extra_near!(state::DeviceResidentRadixState{TF},
        extra::ResidentExtraSource, kernel::AbstractDirectKernel) where TF
    output = state.output
    bodies = state.source_bodies
    ranges = state.cell_ranges
    n_direct = Int(state.counts.n_direct)
    hs = size(output, 1) >= 13
    ep = _emits_potential(kernel)
    @inbounds for pair_i in 1:n_direct
        target_cell = state.direct_targets[pair_i]
        source_cell = state.direct_sources[pair_i]
        scount = extra.cell_ranges[2, source_cell]
        scount == 0 && continue
        sfirst = extra.cell_ranges[1, source_cell]
        tfirst = ranges[1, target_cell]
        tcount = ranges[2, target_cell]
        for i in tfirst:(tfirst + tcount - 1)
            xi = bodies[1, i]; yi = bodies[2, i]; zi = bodies[3, i]
            for j in sfirst:(sfirst + scount - 1)
                if hs && _extra_pair_has_hessian(kernel)
                    u, gx, gy, gz, h1, h2, h3, h4, h5, h6, h7, h8, h9 =
                        _extra_pair_ugh(kernel, xi, yi, zi, extra.buffer, j)
                    ep && (output[1, i] += u)
                    output[2, i] += gx; output[3, i] += gy; output[4, i] += gz
                    output[5, i] += h1; output[6, i] += h2; output[7, i] += h3
                    output[8, i] += h4; output[9, i] += h5; output[10, i] += h6
                    output[11, i] += h7; output[12, i] += h8; output[13, i] += h9
                else
                    u, gx, gy, gz = _extra_pair_ug(kernel, xi, yi, zi, extra.buffer, j)
                    ep && (output[1, i] += u)
                    output[2, i] += gx; output[3, i] += gy; output[4, i] += gz
                end
            end
        end
    end
    return state
end
