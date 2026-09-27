#------- HOST-MIRRORED RESIDENT RADIX LIFECYCLE (Matrix Operator Refactor) -------#
#
# CPU-only validation path for the device-resident CUDA lifecycle. This path uses
# ordinary Array storage but keeps the same resident state shape as the CUDA path:
# source bodies, radix metadata, expansion buffers, and output stay in the state
# across B2M, M2M, M2L, L2L, and L2B.

"""
    host_resident_radix_grid(grid::RadixGrid)

Expand a host [`RadixGrid`](@ref) into host-resident level and node metadata in
a [`DeviceRadixGrid`](@ref)-shaped container. The returned arrays are owned by
the new container; the input grid is not modified.
"""
function host_resident_radix_grid(grid::RadixGrid{TF}) where TF
    level_keys = [sort(unique(key >> (3 * (grid.ell - level)) for key in grid.cell_keys))
                  for level in 0:grid.ell]
    level_offsets = zeros(Int, grid.ell + 2)
    for level in 0:grid.ell
        level_offsets[level + 2] = level_offsets[level + 1] + length(level_keys[level + 1])
    end

    n_nodes = level_offsets[end]
    node_levels = Vector{Int}(undef, n_nodes)
    node_keys = Vector{UInt64}(undef, n_nodes)
    node_coords = Matrix{Int}(undef, 3, n_nodes)
    node_centers = Matrix{TF}(undef, 3, n_nodes)
    index_by_node = Dict{Tuple{Int,UInt64},Int}()

    for level in 0:grid.ell
        width = (2 * grid.h0) / (1 << level)
        for (local_i, key) in pairs(level_keys[level + 1])
            node = level_offsets[level + 1] + local_i
            coord = morton_decode(key, level)
            node_levels[node] = level
            node_keys[node] = key
            node_coords[:, node] .= coord
            node_centers[:, node] .= grid.x_min + width * (SVector{3,TF}(coord) .+ SVector{3,TF}(0.5, 0.5, 0.5))
            index_by_node[(level, key)] = node
        end
    end

    parent_index = zeros(Int, n_nodes)
    child_ranges = zeros(Int, 2, n_nodes)
    for node in 1:n_nodes
        level = node_levels[node]
        if level > 0
            parent_index[node] = index_by_node[(level - 1, node_keys[node] >> 3)]
        end
    end
    for node in 1:n_nodes
        children = findall(==(node), parent_index)
        if !isempty(children)
            child_ranges[1, node] = first(children)
            child_ranges[2, node] = length(children)
        end
    end

    leaf_to_node = level_offsets[grid.ell + 1] .+ collect(1:length(grid.cell_keys))
    cell_centers = Matrix{TF}(undef, 3, length(grid.cell_keys))
    for i_cell in eachindex(grid.cell_keys)
        cell_centers[:, i_cell] .= radix_cell_center(grid, i_cell)
    end

    return DeviceRadixGrid(
        grid.x_min, grid.h0, grid.ell, length(grid.perm), length(grid.cell_keys),
        copy(grid.perm), copy(grid.invperm), copy(grid.cell_keys), copy(grid.cell_ranges),
        copy(grid.body_system), copy(grid.body_index), cell_centers,
        node_levels, node_keys, node_coords, node_centers, parent_index, child_ranges,
        leaf_to_node,
    )
end

function _host_radix_tree_routes(grid::DeviceRadixGrid)
    n_edges = max(length(grid.parent_index) - 1, 0)
    m2m_parent = Vector{Int}(undef, n_edges)
    m2m_child = Vector{Int}(undef, n_edges)
    l2l_parent = Vector{Int}(undef, n_edges)
    l2l_child = Vector{Int}(undef, n_edges)
    @inbounds for edge in 1:n_edges
        node = edge + 1
        parent = grid.parent_index[node]
        m2m_parent[edge] = parent
        m2m_child[edge] = node
        l2l_parent[edge] = parent
        l2l_child[edge] = node
    end
    return m2m_parent, m2m_child, l2l_parent, l2l_child
end

function _host_radix_source_buffers(systems::Tuple, ::Type{TF}) where TF
    return map(systems) do system
        buffer = allocate_source_buffer(TF, system)
        source_to_buffer!(buffer, system, 1:get_n_bodies(system))
        buffer
    end
end

function _host_radix_body_matrix(grid::DeviceRadixGrid{TF}, source_buffers::Tuple) where TF
    # canonical all-rows packed layout (spec §3 decision (c)): every
    # source-buffer row is carried, including radius row 4; systems narrower than
    # the widest are zero-padded
    nrows = maximum(size(buffer, 1) for buffer in source_buffers)
    body = Matrix{TF}(undef, nrows, length(grid.perm))
    @inbounds for sorted_i in eachindex(grid.perm)
        global_i = grid.perm[sorted_i]
        isys = grid.body_system[global_i]
        ibody = grid.body_index[global_i]
        source = source_buffers[isys]
        nsys = size(source, 1)
        for row in 1:nsys
            body[row, sorted_i] = source[row, ibody]
        end
        for row in (nsys + 1):nrows
            body[row, sorted_i] = zero(TF)
        end
    end
    return body
end

function _host_radix_body_matrix(grid::DeviceRadixGrid{TF}, bodies::AbstractMatrix) where TF
    size(bodies, 1) >= 5 ||
        throw(ArgumentError("resident radix bodies must have at least 5 rows: x/y/z/output-placeholder/strength"))
    size(bodies, 2) == grid.n_bodies ||
        throw(ArgumentError("resident radix bodies must have one column per grid body"))
    return Matrix{TF}(bodies[:, grid.perm])
end

function _host_flat_buffer(::Type{TF}, basis_info::OperatorBasisInfo{B,LH}, batch::Integer) where {TF,B,LH}
    return FlatCoefficientBuffer(TF, basis_info, batch)
end

# Transcendental prologue of `_resident_regular_harmonic_coeff`, split out so a
# caller that needs MANY coefficients at the SAME offset pays the sqrt/acos/
# atan/2x-sincos once instead of once per (n,m). That was the dominant cost of
# the resident B2M and L2B kernels: both evaluate one coefficient per (n,m) pair
# and so re-derived (rho, theta, phi) for every one of them -- 63 times per body
# in L2B, and three times per (n,m) in the vortex B2M contributions.
#
# Bit-exactness is preserved by construction. These are the identical
# expressions the monolithic function computed, in the same order, and the
# recurrence that consumes them is untouched; only the point at which they are
# evaluated moves. The rho == 0 branch still short-circuits before any
# transcendental, and the recurrence re-tests it off the carried `rho`.
@inline function _resident_harmonic_setup(dx, dy, dz)
    rho = sqrt(dx * dx + dy * dy + dz * dz)
    if rho == zero(rho)
        z = zero(rho)
        return (rho, z, z, z, z)
    end
    theta = acos(clamp(dz / rho, -one(rho), one(rho)))
    phi = atan(dy, dx)
    y, x = sincos(theta)
    iei_imag, iei_real = sincos(phi + convert(typeof(rho), pi / 2))
    return (rho, x, y, iei_real, iei_imag)
end

@inline _resident_regular_harmonic_coeff(dx, dy, dz, nt, mt) =
    _resident_regular_harmonic_coeff(_resident_harmonic_setup(dx, dy, dz), nt, mt)

@inline function _resident_regular_harmonic_coeff(setup::NTuple{5,<:Any}, nt, mt)
    rho, x, y, iei_real, iei_imag = setup
    if rho == zero(rho)
        return nt == 0 && mt == 0 ? (one(rho), zero(rho)) : (zero(rho), zero(rho))
    end
    fact = one(rho)
    pn = one(rho)
    rhom = one(rho)
    ieim_real = one(rho)
    ieim_imag = zero(rho)
    for m in 0:nt
        p = pn
        rhom_p = rhom * p
        if m == mt && nt == m
            return rhom_p * ieim_real, rhom_p * ieim_imag
        end
        p1 = p
        p = x * (2m + 1) * p1
        rhom *= rho
        rhon = rhom
        for n in (m + 1):nt
            rhon /= -(n + m)
            rhon_p = rhon * p
            if m == mt && n == nt
                return rhon_p * ieim_real, rhon_p * ieim_imag
            end
            p2 = p1
            p1 = p
            p = (x * (2n + 1) * p1 - (n + m) * p2) / (n - m + 1)
            rhon *= rho
        end
        rhom /= -(2m + 2) * (2m + 1)
        pn = -pn * fact * y
        fact += 2
        tmp_re = ieim_real
        tmp_im = ieim_imag
        ieim_real = tmp_re * iei_real - tmp_im * iei_imag
        ieim_imag = tmp_re * iei_imag + tmp_im * iei_real
    end
    return zero(rho), zero(rho)
end

_launch_host_b2m!(state::DeviceResidentRadixState) =
    _launch_host_b2m!(state, state.options.body_type)

function _launch_host_b2m!(state::DeviceResidentRadixState{TF},
        ::Type{<:Point{Source}}) where TF
    fill!(state.multipoles.phi, zero(TF))
    fill!(state.multipoles.chi, zero(TF))
    P = state.invariant_cache.basis_info.orders.P_phi
    # Keep the scalar host loop behind a small array-specialized kernel.
    _host_b2m_kernel!(phi_slab(state.multipoles), state.source_bodies,
        state.cell_ranges, state.cell_centers, state.grid.leaf_to_node, P,
        state.counts.n_cells)
    return state
end

function _launch_host_b2m!(state::DeviceResidentRadixState{TF,B,LH},
        ::Type{<:Point{Vortex}}) where {TF,B,LH}
    LH || throw(ArgumentError(
        "Point{Vortex} sources require the Lamb-Helmholtz channel; construct the " *
        "cache with lamb_helmholtz=true"))
    fill!(state.multipoles.phi, zero(TF))
    fill!(state.multipoles.chi, zero(TF))
    orders = state.invariant_cache.basis_info.orders
    _host_b2m_vortex_kernel!(phi_slab(state.multipoles), chi_slab(state.multipoles),
        state.source_bodies, state.cell_ranges, state.cell_centers,
        state.grid.leaf_to_node, orders.P_phi, orders.P_active,
        state.counts.n_cells)
    return state
end

function _launch_host_b2m!(state::DeviceResidentRadixState{TF},
        ::Type{<:Point{Dipole}}) where TF
    fill!(state.multipoles.phi, zero(TF))
    fill!(state.multipoles.chi, zero(TF))
    P = state.invariant_cache.basis_info.orders.P_phi
    _host_b2m_dipole_kernel!(phi_slab(state.multipoles), state.source_bodies,
        state.cell_ranges, state.cell_centers, state.grid.leaf_to_node, P,
        state.counts.n_cells)
    return state
end

function _launch_host_b2m!(state::DeviceResidentRadixState{TF,B,LH},
        ::Type{<:Point{SourceVortex}}) where {TF,B,LH}
    LH || throw(ArgumentError(
        "Point{SourceVortex} sources require the Lamb-Helmholtz channel; construct the " *
        "cache with lamb_helmholtz=true"))
    fill!(state.multipoles.phi, zero(TF))
    fill!(state.multipoles.chi, zero(TF))
    orders = state.invariant_cache.basis_info.orders
    _host_b2m_sourcevortex_kernel!(phi_slab(state.multipoles), chi_slab(state.multipoles),
        state.source_bodies, state.cell_ranges, state.cell_centers,
        state.grid.leaf_to_node, orders.P_phi, orders.P_active,
        state.counts.n_cells)
    return state
end

# Elements (filaments and planar triangles): the shared per-body expansion of
# src/resident_elements.jl, run per cell into scratch and added into the slabs.
function _launch_host_b2m!(state::DeviceResidentRadixState{TF,B,LH},
        ::Type{BT}) where {TF,B,LH,BT<:Union{Filament,Panel}}
    ((BT <: Filament{Vortex} || BT <: Panel{3,Vortex}) && !LH) && throw(ArgumentError(
        "$BT sources require the Lamb-Helmholtz channel; construct the cache with lamb_helmholtz=true"))
    fill!(state.multipoles.phi, zero(TF))
    fill!(state.multipoles.chi, zero(TF))
    orders = state.invariant_cache.basis_info.orders
    _host_b2m_element_kernel!(BT, phi_slab(state.multipoles), chi_slab(state.multipoles),
        state.source_bodies, state.cell_ranges, state.cell_centers,
        state.grid.leaf_to_node, orders.P_phi, orders.P_active, state.counts.n_cells)
    return state
end

function _host_b2m_element_kernel!(::Type{BT}, ph::AbstractMatrix{TF}, ch, source_bodies,
        cell_ranges, cell_centers, leaf_to_node, P_phi::Int, P_chi::Int, n_cells::Int) where {BT,TF}
    P = max(P_phi, P_chi)
    coef = zeros(TF, 2, 2, harmonic_index(P, P))
    harmonics = zeros(TF, 2, 2, _res_element_harmonics_rows(P))
    ndof_phi = harmonic_index(P_phi, P_phi)
    # no chi slab without the Lamb-Helmholtz channel (P_active is passed as P_chi)
    ndof_chi = (P_chi >= 1 && size(ch, 2) > 0) ? harmonic_index(P_chi, P_chi) : 0
    sdv = Val(element_strength_dims(BT))
    @inbounds for i_cell in 1:n_cells
        first_body = cell_ranges[1, i_cell]
        count = cell_ranges[2, i_cell]
        node = leaf_to_node[i_cell]
        cx = cell_centers[1, i_cell]; cy = cell_centers[2, i_cell]; cz = cell_centers[3, i_cell]
        for k in first_body:(first_body + count - 1)
            fill!(coef, zero(TF))
            _res_element_b2m!(BT, coef, harmonics, source_bodies, k, cx, cy, cz, P, sdv)
            for i in 1:ndof_phi
                ph[2i - 1, node] += coef[1, 1, i]
                ph[2i, node] += coef[2, 1, i]
            end
            for i in 1:ndof_chi
                ch[2i - 1, node] += coef[1, 2, i]
                ch[2i, node] += coef[2, 2, i]
            end
        end
    end
    return ph
end
# the filament-only name the device suites call
_host_b2m_filament_kernel!(::Type{BT}, args...) where BT = _host_b2m_element_kernel!(BT, args...)

function _launch_host_b2m!(state::DeviceResidentRadixState, ::Type{BT}) where BT
    throw(ArgumentError("the resident radix lifecycle implements body-to-multipole for the " *
        "four Point types, the three Filament types and the four triangular Panel{3,TK} types; got body_type $BT"))
end


# Point{Dipole}: the dipole vector in rows 5:7; same sign/conjugation as the scalar B2M.
function _host_b2m_dipole_kernel!(ph::AbstractMatrix{TF}, source_bodies, cell_ranges,
        cell_centers, leaf_to_node, P::Int, n_cells::Int) where TF
    @inbounds for i_cell in 1:n_cells
        first_body = cell_ranges[1, i_cell]
        count = cell_ranges[2, i_cell]
        node = leaf_to_node[i_cell]
        cx = cell_centers[1, i_cell]
        cy = cell_centers[2, i_cell]
        cz = cell_centers[3, i_cell]
        for n in 0:P, m in 0:n
            acc_re = zero(TF)
            acc_im = zero(TF)
            sgn = isodd(n + m) ? -one(TF) : one(TF)
            for k in first_body:(first_body + count - 1)
                re, im = _resident_dipole_contrib(source_bodies[1, k] - cx,
                    source_bodies[2, k] - cy, source_bodies[3, k] - cz,
                    source_bodies[5, k], source_bodies[6, k], source_bodies[7, k], n, m)
                acc_re += re * sgn
                acc_im -= im * sgn
            end
            row = flat_basis_index(n, m, 1)
            ph[row, node] = acc_re
            ph[row + 1, node] = acc_im
        end
    end
    return ph
end

# Point{SourceVortex}: source strength in row 5 (scalar B2M) plus vortex strength
# in rows 6:8 (mirrored vortex B2M), on the same body.
function _host_b2m_sourcevortex_kernel!(ph::AbstractMatrix{TF}, ch::AbstractMatrix{TF},
        source_bodies, cell_ranges, cell_centers, leaf_to_node, P_phi::Int, P_chi::Int,
        n_cells::Int) where TF
    @inbounds for i_cell in 1:n_cells
        first_body = cell_ranges[1, i_cell]
        count = cell_ranges[2, i_cell]
        node = leaf_to_node[i_cell]
        cx = cell_centers[1, i_cell]
        cy = cell_centers[2, i_cell]
        cz = cell_centers[3, i_cell]
        for n in 0:P_phi, m in 0:n
            acc_re = zero(TF)
            acc_im = zero(TF)
            sgn = isodd(n + m) ? -one(TF) : one(TF)
            for k in first_body:(first_body + count - 1)
                dx = source_bodies[1, k] - cx
                dy = source_bodies[2, k] - cy
                dz = source_bodies[3, k] - cz
                rre, rim = _resident_regular_harmonic_coeff(dx, dy, dz, n, m)
                scale = sgn * source_bodies[5, k]
                acc_re += rre * scale
                acc_im -= rim * scale
                re, im = _resident_vortex_phi_contrib(-dx, -dy, -dz,
                    source_bodies[6, k], source_bodies[7, k], source_bodies[8, k], n, m)
                acc_re += re
                acc_im += im
            end
            row = flat_basis_index(n, m, 1)
            ph[row, node] = acc_re
            ph[row + 1, node] = acc_im
        end
        for n in 1:P_chi, m in 0:n
            acc_re = zero(TF)
            acc_im = zero(TF)
            for k in first_body:(first_body + count - 1)
                re, im = _resident_vortex_chi_contrib(cx - source_bodies[1, k],
                    cy - source_bodies[2, k], cz - source_bodies[3, k],
                    source_bodies[6, k], source_bodies[7, k], source_bodies[8, k], n, m)
                acc_re += re
                acc_im += im
            end
            row = flat_basis_index(n, m, 1)
            ch[row, node] = acc_re
            ch[row + 1, node] = acc_im
        end
    end
    return ch
end

function _host_b2m_kernel!(ph::AbstractMatrix{TF}, source_bodies, cell_ranges,
        cell_centers, leaf_to_node, P::Int, n_cells::Int) where TF
    @inbounds for i_cell in 1:n_cells
        first_body = cell_ranges[1, i_cell]
        count = cell_ranges[2, i_cell]
        node = leaf_to_node[i_cell]
        cx = cell_centers[1, i_cell]
        cy = cell_centers[2, i_cell]
        cz = cell_centers[3, i_cell]
        for n in 0:P, m in 0:n
            acc_re = zero(TF)
            acc_im = zero(TF)
            sgn = isodd(n + m) ? -one(TF) : one(TF)
            for k in first_body:(first_body + count - 1)
                dx = source_bodies[1, k] - cx
                dy = source_bodies[2, k] - cy
                dz = source_bodies[3, k] - cz
                q = source_bodies[5, k]
                rre, rim = _resident_regular_harmonic_coeff(dx, dy, dz, n, m)
                scale = sgn * q
                acc_re += rre * scale
                acc_im -= rim * scale
            end
            row = flat_basis_index(n, m, 1)
            ph[row, node] = acc_re
            ph[row + 1, node] = acc_im
        end
    end
    return ph
end

