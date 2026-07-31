#------- HOST-MIRRORED RESIDENT RADIX LIFECYCLE (Matrix Operator Refactor, task 022) -------#
#
# CPU-only validation path for the device-resident CUDA lifecycle. This path uses
# ordinary Array storage but keeps the same resident state shape as the CUDA path:
# source bodies, radix metadata, expansion buffers, and output stay in the state
# across B2M, M2M, M2L, L2L, and L2B.

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
    body = Matrix{TF}(undef, 5, length(grid.perm))
    @inbounds for sorted_i in eachindex(grid.perm)
        global_i = grid.perm[sorted_i]
        isys = grid.body_system[global_i]
        ibody = grid.body_index[global_i]
        source = source_buffers[isys]
        body[1:3, sorted_i] .= source[1:3, ibody]
        body[4, sorted_i] = zero(TF)
        body[5, sorted_i] = source[5, ibody]
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

@inline function _resident_regular_harmonic_coeff(dx, dy, dz, nt, mt)
    rho = sqrt(dx * dx + dy * dy + dz * dz)
    if rho == zero(rho)
        return nt == 0 && mt == 0 ? (one(rho), zero(rho)) : (zero(rho), zero(rho))
    end
    theta = acos(clamp(dz / rho, -one(rho), one(rho)))
    phi = atan(dy, dx)
    y, x = sincos(theta)
    fact = one(rho)
    pn = one(rho)
    rhom = one(rho)
    iei_imag, iei_real = sincos(phi + convert(typeof(rho), pi / 2))
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

function _launch_host_b2m!(state::DeviceResidentRadixState{TF}) where TF
    fill!(state.multipoles.phi, zero(TF))
    fill!(state.multipoles.chi, zero(TF))
    P = state.invariant_cache.basis_info.orders.P_phi
    # Keep the scalar host loop behind a small array-specialized kernel.
    _host_b2m_kernel!(phi_slab(state.multipoles), state.source_bodies,
        state.cell_ranges, state.cell_centers, state.grid.leaf_to_node, P,
        state.counts.n_cells)
    return state
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

function _launch_host_m2m!(state::DeviceResidentRadixState{TF,B,LH}) where {TF,B,LH}
    return _launch_resident_m2m!(state, state.options.m2m_strategy)
end

function _launch_host_m2l_flat_oracle!(state::DeviceResidentRadixState{TF,B,LH}) where {TF,B,LH}
    cache = state.invariant_cache
    op = state.options.operator isa FactoredRotationM2L ? state.options.operator : MaterializedYRotationM2L()
    scratch = M2LOperatorScratch(TF, cache.basis_info, max(length(state.route_targets), 1))
    fill!(state.locals.phi, zero(TF))
    LH && fill!(state.locals.chi, zero(TF))

    route_i = 0
    @inbounds for batch in state.interaction_list.m2l_batches
        nbatch = length(batch.targets)
        nbatch == 0 && continue
        targets = FlatCoefficientBuffer(TF, cache.basis_info, nbatch)
        sources = FlatCoefficientBuffer(TF, cache.basis_info, nbatch)
        phis = Vector{TF}(undef, nbatch)
        thetas = Vector{TF}(undef, nbatch)
        rs = Vector{TF}(undef, nbatch)
        for j in 1:nbatch
            route_i += 1
            target = state.route_targets[route_i]
            source = state.route_sources[route_i]
            sources.phi[:, j] .= state.multipoles.phi[:, source]
            LH && (sources.chi[:, j] .= state.multipoles.chi[:, source])
            dx = state.grid.node_centers[1, target] - state.grid.node_centers[1, source]
            dy = state.grid.node_centers[2, target] - state.grid.node_centers[2, source]
            dz = state.grid.node_centers[3, target] - state.grid.node_centers[3, source]
            r, theta, phi = cartesian_to_spherical(SVector{3,TF}(dx, dy, dz))
            rs[j] = TF(r)
            thetas[j] = TF(theta)
            phis[j] = TF(phi)
        end
        m2l_operator_batch!(op, targets, sources, phis, thetas, rs, cache, scratch, Val(LH))
        for j in 1:nbatch
            target = state.route_targets[route_i - nbatch + j]
            state.locals.phi[:, target] .+= targets.phi[:, j]
            LH && (state.locals.chi[:, target] .+= targets.chi[:, j])
        end
    end
    return state
end

function _launch_host_m2l!(state::DeviceResidentRadixState{TF,B,LH}) where {TF,B,LH}
    return _launch_resident_m2l!(state, state.options.m2l_strategy)
end

function _launch_host_l2l_flat_oracle!(state::DeviceResidentRadixState{TF,B,LH}) where {TF,B,LH}
    cache = state.invariant_cache
    op = state.options.operator isa FactoredRotationM2L ? FactoredRotationL2L() : MaterializedYRotationL2L()
    scratch = L2LOperatorScratch(TF, cache.basis_info, 8)
    @inbounds for level in 1:state.grid.ell
        parents = Int[]
        children = Int[]
        phis = TF[]
        thetas = TF[]
        rs = TF[]
        for edge in eachindex(state.l2l_parent_routes)
            parent = state.l2l_parent_routes[edge]
            child = state.l2l_child_routes[edge]
            parent == 0 && continue
            state.grid.node_levels[child] == level || continue
            dx = state.grid.node_centers[1, child] - state.grid.node_centers[1, parent]
            dy = state.grid.node_centers[2, child] - state.grid.node_centers[2, parent]
            dz = state.grid.node_centers[3, child] - state.grid.node_centers[3, parent]
            r, theta, phi = cartesian_to_spherical(SVector{3,TF}(dx, dy, dz))
            push!(parents, parent)
            push!(children, child)
            push!(rs, TF(r))
            push!(thetas, TF(theta))
            push!(phis, TF(phi))
        end
        isempty(children) && continue
        nbatch = length(children)
        targets = FlatCoefficientBuffer(TF, cache.basis_info, nbatch)
        sources = FlatCoefficientBuffer(TF, cache.basis_info, nbatch)
        for j in 1:nbatch
            sources.phi[:, j] .= state.locals.phi[:, parents[j]]
            LH && (sources.chi[:, j] .= state.locals.chi[:, parents[j]])
        end
        l2l_operator_batch!(op, targets, sources, phis, thetas, rs, cache, scratch, Val(LH))
        for j in 1:nbatch
            state.locals.phi[:, children[j]] .+= targets.phi[:, j]
            LH && (state.locals.chi[:, children[j]] .+= targets.chi[:, j])
        end
    end
    return state
end

function _launch_host_l2l!(state::DeviceResidentRadixState{TF,B,LH}) where {TF,B,LH}
    return _launch_resident_l2l!(state)
end

function _add_host_direct_pairs!(state::DeviceResidentRadixState{TF}) where TF
    # Iterate the flat pair arrays (bounded by counts) rather than the one-shot
    # interaction list so the recurring update path never rebuilds the list object;
    # function barrier as in _launch_host_b2m!.
    _host_direct_pairs_kernel!(state.output, state.source_bodies, state.cell_ranges,
        state.direct_targets, state.direct_sources, state.counts.n_direct)
    return state
end

function _host_direct_pairs_kernel!(output::AbstractMatrix{TF}, source_bodies,
        cell_ranges, direct_targets, direct_sources, n_direct::Int) where TF
    c = inv(TF(4) * TF(pi))
    @inbounds for pair_i in 1:n_direct
        target_cell = direct_targets[pair_i]
        source_cell = direct_sources[pair_i]
        tfirst = cell_ranges[1, target_cell]
        tcount = cell_ranges[2, target_cell]
        sfirst = cell_ranges[1, source_cell]
        scount = cell_ranges[2, source_cell]
        for i in tfirst:(tfirst + tcount - 1)
            xi = source_bodies[1, i]
            yi = source_bodies[2, i]
            zi = source_bodies[3, i]
            for j in sfirst:(sfirst + scount - 1)
                i == j && continue
                dx = xi - source_bodies[1, j]
                dy = yi - source_bodies[2, j]
                dz = zi - source_bodies[3, j]
                r2 = dx * dx + dy * dy + dz * dz
                r2 == zero(TF) && continue
                invr = inv(sqrt(r2))
                q = source_bodies[5, j] * c
                output[1, i] += q * invr
                invr3 = invr * invr * invr
                output[2, i] -= q * dx * invr3
                output[3, i] -= q * dy * invr3
                output[4, i] -= q * dz * invr3
            end
        end
    end
    return output
end

@inline _resident_flat_phi_re(ph, j, P, n, m) =
    (n < 0 || n > P || m < 0 || m > n) ? zero(eltype(ph)) : ph[flat_basis_index(n, m, 1), j]

@inline _resident_flat_phi_im(ph, j, P, n, m) =
    (n < 0 || n > P || m < 0 || m > n) ? zero(eltype(ph)) : ph[flat_basis_index(n, m, 2), j]

@inline _resident_flat_chi_re(ch, j, P, n, m) =
    (n < 0 || n > P || m < 0 || m > n) ? zero(eltype(ch)) : ch[flat_basis_index(n, m, 1), j]

@inline _resident_flat_chi_im(ch, j, P, n, m) =
    (n < 0 || n > P || m < 0 || m > n) ? zero(eltype(ch)) : ch[flat_basis_index(n, m, 2), j]

@inline function _resident_local_eval_flat(ph, ch, node, dx, dy, dz, P_phi, P_active, ::Val{LH}) where LH
    TF = eltype(ph)
    c = inv(TF(4) * TF(pi))
    u = zero(TF)
    vx = zero(TF); vy = zero(TF); vz = zero(TF)
    @inbounds for n in 0:P_active
        rre, rim = _resident_regular_harmonic_coeff(dx, dy, dz, n, 0)
        if n <= P_phi && (!LH || n == 0)
            u += rre * _resident_flat_phi_re(ph, node, P_phi, n, 0) -
                 rim * _resident_flat_phi_im(ph, node, P_phi, n, 0)
        end

        phi1c = _resident_flat_phi_re(ph, node, P_phi, n + 1, 1)
        phi1s = _resident_flat_phi_im(ph, node, P_phi, n + 1, 1)
        phi0c = _resident_flat_phi_re(ph, node, P_phi, n + 1, 0)
        phi0s = _resident_flat_phi_im(ph, node, P_phi, n + 1, 0)
        vxr = -phi1s; vxi = zero(TF)
        vyr = -phi1c; vyi = zero(TF)
        vzr = -phi0c; vzi = -phi0s
        if LH
            vxr += n * _resident_flat_chi_re(ch, node, P_active, n, 1)
            vyr -= n * _resident_flat_chi_im(ch, node, P_active, n, 1)
        end
        vx += vxr * rre - vxi * rim
        vy += vyr * rre - vyi * rim
        vz += vzr * rre - vzi * rim

        for m in 1:n
            rre, rim = _resident_regular_harmonic_coeff(dx, dy, dz, n, m)
            if n <= P_phi && !LH
                u += 2 * (rre * _resident_flat_phi_re(ph, node, P_phi, n, m) -
                          rim * _resident_flat_phi_im(ph, node, P_phi, n, m))
            end

            amm1 = _resident_flat_phi_re(ph, node, P_phi, n + 1, m - 1)
            bmm1 = _resident_flat_phi_im(ph, node, P_phi, n + 1, m - 1)
            amp1 = _resident_flat_phi_re(ph, node, P_phi, n + 1, m + 1)
            bmp1 = _resident_flat_phi_im(ph, node, P_phi, n + 1, m + 1)
            am = _resident_flat_phi_re(ph, node, P_phi, n + 1, m)
            bm = _resident_flat_phi_im(ph, node, P_phi, n + 1, m)

            vxr = -(bmm1 + bmp1) * TF(0.5)
            vxi = (amm1 + amp1) * TF(0.5)
            vyr = (amm1 - amp1) * TF(0.5)
            vyi = (bmm1 - bmp1) * TF(0.5)
            vzr = -am
            vzi = -bm

            if LH
                cmm1 = _resident_flat_chi_re(ch, node, P_active, n, m - 1)
                dmm1 = _resident_flat_chi_im(ch, node, P_active, n, m - 1)
                cmp1 = _resident_flat_chi_re(ch, node, P_active, n, m + 1)
                dmp1 = _resident_flat_chi_im(ch, node, P_active, n, m + 1)
                cm = _resident_flat_chi_re(ch, node, P_active, n, m)
                dm = _resident_flat_chi_im(ch, node, P_active, n, m)
                vxr += ((n - m) * cmp1 - (n + m) * cmm1) * TF(0.5)
                vxi += ((n - m) * dmp1 - (n + m) * dmm1) * TF(0.5)
                vyr -= ((n - m) * dmp1 + (n + m) * dmm1) * TF(0.5)
                vyi += ((n - m) * cmp1 + (n + m) * cmm1) * TF(0.5)
                vzr += m * dm
                vzi -= m * cm
            end

            vx += 2 * (vxr * rre - vxi * rim)
            vy += 2 * (vyr * rre - vyi * rim)
            vz += 2 * (vzr * rre - vzi * rim)
        end
    end
    return u * c, vx * c, vy * c, vz * c
end

function _launch_host_l2b!(state::DeviceResidentRadixState{TF,B,LH}) where {TF,B,LH}
    fill!(state.output, zero(TF))
    _add_host_direct_pairs!(state)
    P_phi = state.invariant_cache.basis_info.orders.P_phi
    P_active = state.invariant_cache.basis_info.orders.P_active
    _host_l2b_kernel!(state.output, state.source_bodies, state.cell_ranges,
        state.cell_centers, state.grid.leaf_to_node, phi_slab(state.locals),
        chi_slab(state.locals), P_phi, P_active, Val(LH), state.counts.n_cells)
    return state
end

function _host_l2b_kernel!(output::AbstractMatrix, source_bodies, cell_ranges,
        cell_centers, leaf_to_node, ph, ch, P_phi::Int, P_active::Int,
        lhv::Val{LH}, n_cells::Int) where LH
    @inbounds for cell in 1:n_cells
        node = leaf_to_node[cell]
        first_body = cell_ranges[1, cell]
        count = cell_ranges[2, cell]
        cx = cell_centers[1, cell]
        cy = cell_centers[2, cell]
        cz = cell_centers[3, cell]
        for i in first_body:(first_body + count - 1)
            scalar_potential, gx, gy, gz = _resident_local_eval_flat(
                ph, ch, node,
                source_bodies[1, i] - cx,
                source_bodies[2, i] - cy,
                source_bodies[3, i] - cz,
                P_phi, P_active, lhv,
            )
            output[1, i] += scalar_potential
            output[2, i] += gx
            output[3, i] += gy
            output[4, i] += gz
        end
    end
    return output
end

_radix_uses_factored_rotation(options::CUDARadixLifecycleOptions) =
    options.operator isa FactoredRotationM2L

function _launch_host_resident_operator_pipeline!(state::DeviceResidentRadixState)
    _launch_host_m2m!(state)
    # 016b watch item 2: the FactoredRotation* operators are exact only on the
    # physical subspace (m = 0 imaginary rows == 0); guard the upward-pass output
    # once per lifecycle run, DEBUG[]-gated (off in production).
    if DEBUG[] && _radix_uses_factored_rotation(state.options)
        _assert_factored_input_physical(state.multipoles)
    end
    _launch_host_m2l!(state)
    _launch_host_l2l!(state)
    _launch_host_l2b!(state)
    state.counters.expansion_host_copies == 0 ||
        throw(AssertionError("resident host radix lifecycle observed expansion host copies"))
    return state
end

function host_radix_state(systems, grid::RadixGrid, list::RadixInteractionList,
        P::Integer, lamb_helmholtz::Val{LH}=Val(false);
        options::CUDARadixLifecycleOptions=CUDARadixLifecycleOptions()) where LH
    return host_radix_state(systems, host_resident_radix_grid(grid), list, P, lamb_helmholtz; options)
end

function host_radix_state(systems, grid::DeviceRadixGrid, list::RadixInteractionList,
        P::Integer, lamb_helmholtz::Val{LH}=Val(false);
        options::CUDARadixLifecycleOptions=CUDARadixLifecycleOptions()) where LH
    TF = options.precision
    basis_info = OperatorBasisInfo(CompressedComplexBasis(), P, lamb_helmholtz)
    counters = CUDARadixTransferCounters()
    body_perm = grid.perm
    body_system_ids = grid.body_system
    body_indices = grid.body_index
    host_body_perm = grid.perm
    host_body_system_ids = grid.body_system
    host_body_indices = grid.body_index

    source_bodies_raw = systems isa AbstractMatrix ?
        _host_radix_body_matrix(grid, systems) :
        _host_radix_body_matrix(grid, _host_radix_source_buffers(to_tuple(systems), TF))
    source_bodies = Matrix{TF}(source_bodies_raw)

    m2m_parent_routes, m2m_child_routes, l2l_parent_routes, l2l_child_routes =
        _host_radix_tree_routes(grid)
    multipoles = _host_flat_buffer(TF, basis_info, length(grid.node_keys))
    locals = _host_flat_buffer(TF, basis_info, length(grid.node_keys))
    output = zeros(TF, 4, grid.n_bodies)

    levels, offsets, targets, sources = _flatten_radix_routes_host(list, grid)
    direct_targets, direct_sources = _flatten_radix_direct_pairs_host(list)
    cache = OperatorInvariantCache(TF, basis_info)
    scratch = ResidentOperatorWorkspace(
        TF, basis_info, multipoles, grid, list,
        m2m_parent_routes, m2m_child_routes, l2l_parent_routes, l2l_child_routes,
        grid.node_levels, grid.node_centers, targets, sources;
        m2l_strategy=options.m2l_strategy, operator=options.operator,
    )
    return DeviceResidentRadixState{TF,CompressedComplexBasis,LH}(
        grid, list, source_bodies, source_bodies,
        body_perm, body_system_ids, body_indices,
        host_body_perm, host_body_system_ids, host_body_indices,
        grid.cell_centers, m2m_parent_routes, m2m_child_routes,
        l2l_parent_routes, l2l_child_routes, grid.node_levels, grid.node_centers,
        targets, sources,
        grid.cell_centers, grid.cell_ranges,
        m2m_parent_routes, m2m_child_routes, l2l_parent_routes, l2l_child_routes,
        multipoles, locals, levels, offsets, targets, sources,
        direct_targets, direct_sources, output,
        cache, scratch, counters, options,
        RadixStepCounts(source_bodies, grid.cell_ranges, multipoles, targets, direct_targets),
    )
end

function _flatten_radix_routes_host(list::RadixInteractionList, grid::DeviceRadixGrid)
    nroute = sum((length(batch.targets) for batch in list.m2l_batches); init=0)
    levels = Vector{Int}(undef, nroute)
    offsets = Matrix{Int}(undef, 3, nroute)
    targets = Vector{Int}(undef, nroute)
    sources = Vector{Int}(undef, nroute)
    i = 0
    for batch in list.m2l_batches
        for j in eachindex(batch.targets)
            i += 1
            levels[i] = batch.level
            offsets[:, i] .= batch.offset
            targets[i] = grid.leaf_to_node[batch.targets[j]]
            sources[i] = grid.leaf_to_node[batch.sources[j]]
        end
    end
    return levels, offsets, targets, sources
end

function _flatten_radix_direct_pairs_host(list::RadixInteractionList)
    direct_targets = Vector{Int}(undef, length(list.direct_pairs))
    direct_sources = Vector{Int}(undef, length(list.direct_pairs))
    for (i, pair) in pairs(list.direct_pairs)
        direct_targets[i] = pair[1]
        direct_sources[i] = pair[2]
    end
    return direct_targets, direct_sources
end

#------- host output finalization (task 023) -------#
#
# The lifecycle output is a 4×n slab (scalar potential + gradient, sorted body
# order). These scatter it back to user target buffers/systems; hoisted from the
# CUDA-only file so the pure-host path can finalize without loading CUDA.

function _copy_radix_output_to_host_target_buffer!(target_buffer, output, grid::RadixGrid,
        isys::Integer, derivatives_switch)
    return _copy_radix_output_to_host_target_buffer!(
        target_buffer, output, grid.perm, grid.body_system, grid.body_index, isys, derivatives_switch,
    )
end

function _copy_radix_output_to_host_target_buffer!(target_buffer, output, body_perm,
        body_system_ids, body_indices, isys::Integer, derivatives_switch,
        n_bodies::Integer=size(output, 2))
    reset!(target_buffer)
    isempty(hessian_range(derivatives_switch)) ||
        throw(ArgumentError("radix output finalization does not provide hessian rows"))
    scalar_row = scalar_potential_index(derivatives_switch)
    grange = gradient_range(derivatives_switch)
    @inbounds for sorted_i in 1:n_bodies
        global_i = body_perm[sorted_i]
        body_system_ids[global_i] == isys || continue
        ibody = body_indices[global_i]
        if scalar_row > 0
            target_buffer[scalar_row, ibody] = output[1, sorted_i]
        end
        if !isempty(grange)
            target_buffer[grange, ibody] .= @view output[2:4, sorted_i]
        end
    end
    return target_buffer
end

function _copy_radix_output_to_host_target_buffer!(target_buffer, output,
        grid::DeviceRadixGrid, isys::Integer, derivatives_switch)
    throw(ArgumentError(
        "DeviceRadixGrid host finalization requires DeviceResidentRadixState host metadata mirrors; " *
        "call finalize_cuda_radix_output! or pass explicit host metadata",
    ))
end

"""
    finalize_radix_output!(state, target_systems; derivatives_switches, target_buffers)

Scatter a host-resident radix lifecycle output back into the user target systems:
de-permute `state.output` (4×n, sorted body order: scalar potential + gradient)
into per-system target buffers and call [`buffer_to_target!`](@ref). Hessian rows
are not available on the radix path; a derivatives switch requesting them throws.
Pass preallocated `target_buffers` (one per system) to keep recurring steps
allocation-free; otherwise buffers are allocated per call.
"""
function finalize_radix_output!(state::DeviceResidentRadixState{TF}, target_systems;
        derivatives_switches=DerivativesSwitch(true, true, false, to_tuple(target_systems)),
        target_buffers=nothing) where TF
    systems = to_tuple(target_systems)
    switches = to_tuple(derivatives_switches)
    length(systems) == length(switches) ||
        throw(ArgumentError("target systems and derivatives switches must have the same length"))
    state.output isa Array ||
        throw(ArgumentError("finalize_radix_output! requires a host-resident state; use finalize_cuda_radix_output! for device-resident states"))
    for (isys, target_system, switch) in zip(eachindex(systems), systems, switches)
        residency(target_system) isa HostResident ||
            throw(ArgumentError("finalize_radix_output! supports host-resident target systems only"))
        target_buffer = target_buffers === nothing ?
            allocate_target_buffer(TF, target_system, switch) : target_buffers[isys]
        _copy_radix_output_to_host_target_buffer!(
            target_buffer, state.output, state.host_body_perm,
            state.host_body_system_ids, state.host_body_indices, isys, switch,
            state.counts.n_bodies,
        )
        buffer_to_target!(target_system, target_buffer, switch, 1:get_n_bodies(target_system))
    end
    return target_systems
end

function run_host_radix_lifecycle!(state::DeviceResidentRadixState)
    state.counters.expansion_host_copies == 0 ||
        throw(AssertionError("resident host radix lifecycle observed expansion host copies before execution"))
    _launch_host_b2m!(state)
    _launch_host_resident_operator_pipeline!(state)
    return state
end

#------- RadixFMMCache: fixed-box recurring driver (task 023) -------#

function _assert_radix_targets_are_sources(targets::Tuple, sources::Tuple)
    length(targets) == length(sources) ||
        throw(ArgumentError("the radix fmm! path requires target_systems === source_systems (v1 restriction)"))
    for (t, s) in zip(targets, sources)
        t === s ||
            throw(ArgumentError("the radix fmm! path requires target_systems === source_systems (v1 restriction)"))
    end
    return nothing
end

function _assert_radix_positions_in_box(systems::Tuple, x_min::SVector{3,TF}, h0::TF) where TF
    x_max = x_min .+ 2 * h0
    for (isys, system) in enumerate(systems)
        for i_body in 1:get_n_bodies(system)
            x = get_position(system, i_body)
            if !(x_min[1] <= x[1] <= x_max[1] && x_min[2] <= x[2] <= x_max[2] &&
                 x_min[3] <= x[3] <= x_max[3])
                throw(ArgumentError(
                    "body $i_body of system $isys at $(Tuple(x)) lies outside the fixed " *
                    "RadixFMMCache box [$(Tuple(x_min)), $(Tuple(x_max))]; the box is part " *
                    "of the cache's invariant contract — construct a new cache (or pass " *
                    "explicit bounds=(x_min, box_size) covering the trajectory)"))
            end
        end
    end
    return nothing
end

# Measured window widths (task 027). The host default of 4 comes from the 026 host
# campaign; on the GPU the per-window flag/scan/compact carries a fixed ~50 us
# device-to-host round trip, so route generation scales as
# `(ell - 1) * ceil(noffsets / K)` and K = 4 spends 72-164 ms per step on latency
# alone. H200 job 12992039 measured route generation falling 109-193x from K = 4 to
# a whole-level window; K = 256 captures nearly all of that at <= 5x the K = 4
# device footprint (larger K grows the window buffers as `K * max_level_nodes`).
const RADIX_HOST_WINDOW_CLASSES = 4
const RADIX_DEVICE_WINDOW_CLASSES = 256

# Default separation policy (task 027). `HierarchicalRigidStencil` replaced the flat
# `ConstantPAnalyticStencil` as the production default: the flat classifier's
# accepted-offset set grows with `ell` (cell width shrinks at fixed epsilon), so its
# route count scales as `offsets(ell) x cells`, while the rigid stencil's offset set
# is level-invariant. An explicit `stencil_epsilon` still selects the flat policy, so
# existing callers keep their exact behavior.
function _default_radix_policy(policy, P::Int, ::Type{TF}, LH::Bool, h0, ell::Int,
        device::Bool, stencil_epsilon, near_radius2, window_classes) where TF
    if policy !== nothing
        (stencil_epsilon === nothing && near_radius2 === nothing &&
            window_classes === nothing) ||
            throw(ArgumentError("an explicit `policy` carries its own stencil " *
                "parameters; do not combine it with `stencil_epsilon`, " *
                "`near_radius2`, or `window_classes`"))
        return policy
    end
    K = window_classes === nothing ?
        (device ? RADIX_DEVICE_WINDOW_CLASSES : RADIX_HOST_WINDOW_CLASSES) :
        Int(window_classes)
    if stencil_epsilon !== nothing
        # explicit tolerance: the caller is asking for the flat analytic classifier
        near_radius2 === nothing ||
            throw(ArgumentError("`stencil_epsilon` selects the flat " *
                "ConstantPAnalyticStencil, which has no `near_radius2`; pass a " *
                "HierarchicalRigidStencil `policy` for an explicit tolerance " *
                "with a rigid near set"))
        return ConstantPAnalyticStencil(
            ConstantPStencilConfig(P, TF(stencil_epsilon); lamb_helmholtz=LH))
    end
    q = near_radius2 === nothing ? 12 : Int(near_radius2)
    if ell < 2
        # the first M2L level is 2; there is no hierarchy to walk below that
        return ConstantPAnalyticStencil(
            ConstantPStencilConfig(P, TF(1e-4); lamb_helmholtz=LH))
    end
    eps = rigid_stencil_epsilon(P, h0, ell, q; lamb_helmholtz=LH, TF)
    return HierarchicalRigidStencil(
        ConstantPStencilConfig(P, TF(eps); lamb_helmholtz=LH);
        near_radius2=q, window_classes=K)
end

"""
    RadixFMMCache(target_systems, source_systems=target_systems; kwargs...)

Construct the opt-in radix-grid / matrix-operator FMM cache (task 023). Eagerly
builds the capacity-sized resident state and every step-invariant operator table,
then runs the first state update from the systems' current positions — so the
first `fmm!(system, cache)` call is already the recurring fast path and no array
is reallocated over the cache's lifetime.

**Keyword arguments**

- `expansion_order::Int=4`: constant expansion order `P` on this path
- `ell::Int=4`: radix grid depth (leaf grid is `2^ell` cells per axis)
- `max_n_bodies::Int=n`: capacity bound; steps may use any `1 <= n <= max_n_bodies`
- `bounds=nothing`: `(x_min::SVector{3}, box_size)` fixed domain box; default derives
  a cube from the current positions inflated by `bounds_margin`
- `bounds_margin::Real=0.05`: relative margin applied to derived bounds
- `lamb_helmholtz=nothing`: override the `has_vector_potential` inference
- `device::Bool=false`: run the lifecycle device-resident (CUDA; requires
  `load_cuda_radix_lifecycle!()`)
- `options::CUDARadixLifecycleOptions`: operator strategies/precision; defaults to
  the task-022 tuned `m2l_strategy=ConcatenatedFixedZM2L()`
- `near_radius2`: rigid near set `{o : |o|^2 <= near_radius2}` of the default
  hierarchical policy (default `12`, the `024b` `theta=0.5` stencil; `3` is the
  classic FMM one)
- `window_classes`: route-window width; defaults to the measured
  `$(RADIX_DEVICE_WINDOW_CLASSES)` on device and `$(RADIX_HOST_WINDOW_CLASSES)` on host
- `stencil_epsilon::Real`: **selects the deprecated flat `ConstantPAnalyticStencil`**
  at this tolerance (task 027); omit it to get the hierarchical default
- `policy`: explicit `ConstantPAnalyticStencil` or `HierarchicalRigidStencil`
  (both run host- or device-resident). A policy carries its own stencil
  parameters, so combining it with `stencil_epsilon`, `near_radius2`, or
  `window_classes` throws; likewise `stencil_epsilon` (flat) rejects `near_radius2`.

Since task 027 the default policy is [`HierarchicalRigidStencil`](@ref) at
`near_radius2=12`, with a tolerance derived by [`rigid_stencil_epsilon`](@ref) so the
analytic accuracy gate is satisfied by construction. The flat
[`ConstantPAnalyticStencil`](@ref) is deprecated as a default but fully supported;
it is still used automatically when `ell < 2`, where there is no hierarchy to walk.

The domain box, `ell`, expansion order, and `max_n_bodies` are fixed for the
cache's lifetime; bodies leaving the box throw `ArgumentError` at the next step.
"""
function RadixFMMCache(target_systems, source_systems=target_systems;
        expansion_order::Integer=4,
        ell::Integer=4,
        max_n_bodies::Union{Nothing,Integer}=nothing,
        bounds=nothing,
        bounds_margin::Real=0.05,
        lamb_helmholtz::Union{Nothing,Bool}=nothing,
        device::Bool=false,
        options::Union{Nothing,CUDARadixLifecycleOptions}=nothing,
        stencil_epsilon::Union{Nothing,Real}=nothing,
        near_radius2::Union{Nothing,Integer}=nothing,
        window_classes::Union{Nothing,Integer}=nothing,
        policy::Union{Nothing,ConstantPAnalyticStencil,HierarchicalRigidStencil}=nothing)
    targets = to_tuple(target_systems)
    sources = to_tuple(source_systems)
    _assert_radix_targets_are_sources(targets, sources)
    LH = lamb_helmholtz === nothing ? has_vector_potential(sources) : Bool(lamb_helmholtz)
    if options === nothing
        options = CUDARadixLifecycleOptions(; m2l_strategy=ConcatenatedFixedZM2L())
    end
    TF = options.precision
    if device
        cuda_radix_available() ||
            throw(ArgumentError("RadixFMMCache(device=true) requires a functional CUDA " *
                "radix lifecycle; call load_cuda_radix_lifecycle!() first ($(cuda_radix_status()))"))
    end
    for system in sources
        data_per_body(system) >= 5 ||
            throw(ArgumentError("the radix path packs bodies as [x, y, z, radius, strength]; " *
                "data_per_body(system) must be >= 5"))
    end

    n0 = get_n_bodies(sources)
    n0 > 0 || throw(ArgumentError("RadixFMMCache requires at least one body"))
    maxn = max_n_bodies === nothing ? n0 : Int(max_n_bodies)
    maxn >= n0 || throw(ArgumentError("max_n_bodies=$maxn is smaller than the current body count $n0"))

    if bounds === nothing
        x_min_data, x_max_data = _radix_bounds(sources, TF)
        center = (x_min_data + x_max_data) * TF(0.5)
        box = (x_max_data - x_min_data) * TF(0.5)
        h0 = max(box[1], box[2], box[3]) * (1 + TF(bounds_margin))
        h0 > zero(TF) ||
            throw(ArgumentError("bodies are degenerate (zero extent); pass explicit bounds=(x_min, box_size)"))
        x_min = center - SVector{3,TF}(h0, h0, h0)
    else
        x_min = SVector{3,TF}(bounds[1])
        h0 = TF(bounds[2]) / 2
        h0 > zero(TF) || throw(ArgumentError("bounds box_size must be positive"))
    end

    # Validate the requested strategy before any policy-dependent substitution.
    # The hierarchical path routes the concatenated and grouped-factored selections
    # through the bounded concat engine, which would otherwise silently accept an
    # unsupported strategy that the flat path rejects (task 027).
    options.m2l_strategy isa Union{ConcatenatedFixedZM2L,PrecomputedFactoredYM2L,
        DenseTranslationM2L} || throw(ArgumentError(
        "RadixFMMCache supports ConcatenatedFixedZM2L, PrecomputedFactoredYM2L, " *
        "or DenseTranslationM2L; got $(typeof(options.m2l_strategy)) (the " *
        "SharedRotationM2L group layout is not refreshable in place)"))

    P = Int(expansion_order)
    stencil_policy = _default_radix_policy(policy, P, TF, LH, h0, Int(ell), device,
        stencil_epsilon, near_radius2, window_classes)
    hierarchical = stencil_policy isa HierarchicalRigidStencil
    hierarchical_tables = hierarchical ?
        RigidHierarchicalTables(stencil_policy.near_radius2) : nothing
    hierarchical && _verify_hierarchical_classifier!(h0, Int(ell),
        stencil_policy, hierarchical_tables)
    class_level, class_offset, effective_offsets = hierarchical ?
        _hierarchical_class_metadata(hierarchical_tables, Int(ell)) :
        (Int32[], Matrix{Int32}(undef, 3, 0), SVector{3,Int}[])
    accepted, rejected = hierarchical ?
        (effective_offsets, hierarchical_tables.near_offsets) :
        classify_radix_stencil_offsets(h0, Int(ell), stencil_policy.config)

    max_cells = _radix_level_node_capacity(Int(ell), maxn)
    max_nodes = sum(_radix_level_node_capacity(L, max_cells) for L in 0:Int(ell))
    max_level_nodes = Int(ell) >= 2 ? maximum(
        _radix_level_node_capacity(L, max_cells) for L in 2:Int(ell)) : 0
    route_capacity = hierarchical ?
        min(min(stencil_policy.window_classes,
                length(hierarchical_tables.push_offsets)) * max_level_nodes,
            max_level_nodes * max_level_nodes) :
        min(length(accepted), max_cells) * max_cells
    direct_capacity = max_cells * min(length(rejected), max_cells)

    basis_info = OperatorBasisInfo(CompressedComplexBasis(), P, Val(LH))

    if device
        cache = _radix_cache_device_build(sources, P, Int(ell), x_min, h0, maxn,
            options, stencil_policy, accepted, rejected, max_cells, max_nodes,
            route_capacity, direct_capacity, basis_info, Val(LH);
            hierarchical_tables, class_level, class_offset,
            max_level_nodes)
        cache.built = true
        return cache
    end

    grid = _allocate_host_radix_grid(TF, x_min, h0, Int(ell), maxn, max_cells, max_nodes)
    multipoles = _host_flat_buffer(TF, basis_info, max_nodes)
    locals = _host_flat_buffer(TF, basis_info, max_nodes)
    source_bodies = Matrix{TF}(undef, 5, maxn)
    output = zeros(TF, 4, maxn)
    route_levels = Vector{Int}(undef, route_capacity)
    route_offsets = Matrix{Int}(undef, 3, route_capacity)
    route_targets = Vector{Int}(undef, route_capacity)
    route_sources = Vector{Int}(undef, route_capacity)
    direct_targets = Vector{Int}(undef, direct_capacity)
    direct_sources = Vector{Int}(undef, direct_capacity)
    n_edges_capacity = max(max_nodes - 1, 0)
    m2m_parent_routes = Vector{Int}(undef, n_edges_capacity)
    m2m_child_routes = Vector{Int}(undef, n_edges_capacity)
    l2l_parent_routes = Vector{Int}(undef, n_edges_capacity)
    l2l_child_routes = Vector{Int}(undef, n_edges_capacity)

    invariant = OperatorInvariantCache(TF, basis_info)
    # The concatenated and grouped-factored selections share the bounded
    # per-column concatenated engine in hierarchical mode.  Precomputed-y and
    # dense retain their specialized host plans and refresh those plans once per
    # route window.
    hierarchical_specialized = hierarchical &&
        options.m2l_strategy isa Union{PrecomputedFactoredYM2L,DenseTranslationM2L}
    if hierarchical && !hierarchical_specialized &&
            options.operator isa FactoredRotationM2L
        @debug "hierarchical FactoredRotationM2L selection runs the shared " *
            "concatenated per-column engine under MaterializedYRotationM2L " *
            "(mathematically equivalent: concat is the factored composition " *
            "with a shared z-block); options.operator still reports " *
            "FactoredRotationM2L"
    end
    workspace_strategy = hierarchical ?
        (hierarchical_specialized ? options.m2l_strategy : ConcatenatedFixedZM2L()) :
        options.m2l_strategy
    workspace_operator = hierarchical ?
        (hierarchical_specialized ? options.operator : MaterializedYRotationM2L()) :
        options.operator
    scratch = _radix_cache_workspace(TF, basis_info, multipoles, Int(ell), h0,
        max_cells, max_nodes, route_capacity, accepted, invariant,
        workspace_strategy, workspace_operator;
        hierarchical_noffsets=hierarchical ? length(hierarchical_tables.push_offsets) : 0)
    counters = CUDARadixTransferCounters()
    occupancy = hierarchical ? RadixLevelOccupancy(Int(ell);
        max_bytes=stencil_policy.dense_occupancy_max_bytes,
        max_dense_ell=stencil_policy.dense_occupancy_max_ell) : nothing
    hierarchical_apply_plan = hierarchical ? scratch.m2l_concat : nothing
    hierarchical_ctx = hierarchical ? HostHierarchicalM2LContext(
        hierarchical_tables, occupancy, class_level, class_offset,
        effective_offsets, hierarchical_apply_plan, stencil_policy.window_classes,
        zeros(Int, Int(ell) + 2), 0, zeros(Int, Int(ell) + 1), 0,
        false, zeros(UInt64, 5), zeros(UInt64, Int(ell) + 1)) : nothing
    state = DeviceResidentRadixState{TF,CompressedComplexBasis,LH}(
        grid, hierarchical_ctx, source_bodies, source_bodies,
        grid.perm, grid.body_system, grid.body_index,
        grid.perm, grid.body_system, grid.body_index,
        grid.cell_centers, m2m_parent_routes, m2m_child_routes,
        l2l_parent_routes, l2l_child_routes, grid.node_levels, grid.node_centers,
        route_targets, route_sources,
        grid.cell_centers, grid.cell_ranges,
        m2m_parent_routes, m2m_child_routes, l2l_parent_routes, l2l_child_routes,
        multipoles, locals, route_levels, route_offsets, route_targets, route_sources,
        direct_targets, direct_sources, output,
        invariant, scratch, counters, options,
        RadixStepCounts(0, 0, 0, 0, 0),
    )

    G = 1 << Int(ell)
    source_buffers = Tuple(Matrix{TF}(undef, data_per_body(system), maxn) for system in sources)
    cache = RadixFMMCache{TF,LH}(
        P, Int(ell), x_min, h0, maxn, device, options, stencil_policy,
        accepted, rejected, max_cells, max_nodes, route_capacity, direct_capacity,
        state, hierarchical ? zeros(Int32, 0, 0, 0) : zeros(Int32, G, G, G),
        Vector{SVector{3,Int}}(undef, max_cells),
        zeros(Int, Int(ell) + 2), Vector{UInt64}(undef, maxn), Vector{Int}(undef, maxn),
        zeros(Int, 256), zeros(Int, 256), source_buffers, nothing, nothing,
        length(sources), false, 0,
    )
    update_radix_state!(cache, sources)
    cache.built = true
    return cache
end

# Refresh the per-level M2M/L2L group edge columns and the nonleaf index from the
# freshly updated grid. Nodes are level-major, so each level's children occupy one
# contiguous index block and the nonleaf prefix is exactly 1:level_offsets[ell + 1].
function _refresh_resident_stage_groups!(ws::ResidentOperatorWorkspace{TF},
        grid::DeviceRadixGrid, level_offsets::Vector{Int}) where TF
    ell = grid.ell
    length(ws.m2m_groups) == max(ell, 0) && length(ws.l2l_groups) == max(ell, 0) ||
        throw(ArgumentError("resident cache workspace does not match grid depth ell=$ell"))
    for (gi, parent_level) in enumerate((ell - 1):-1:0)
        _refresh_group_edges!(ws.m2m_groups[gi], grid, level_offsets, parent_level + 1, :m2m)
    end
    for (gi, child_level) in enumerate(1:ell)
        _refresh_group_edges!(ws.l2l_groups[gi], grid, level_offsets, child_level, :l2l)
    end
    n_nonleaf = level_offsets[ell + 1]
    resize!(ws.nonleaf_idx, n_nonleaf)
    @inbounds for i in 1:n_nonleaf
        ws.nonleaf_idx[i] = i
    end
    return ws
end

function _refresh_group_edges!(group::ResidentOperatorGroup, grid::DeviceRadixGrid{TF},
        level_offsets::Vector{Int}, child_level::Integer, kind::Symbol) where TF
    first_child = level_offsets[child_level + 1] + 1
    last_child = level_offsets[child_level + 2]
    n = last_child - first_child + 1
    n <= length(group.source_idx) ||
        throw(AssertionError("resident $kind group at child level $child_level exceeded its capacity"))
    group.count[] = n
    # function barrier: group fields are Any-typed
    _refresh_group_edges_kernel!(group.source_idx, group.target_idx, group.phis,
        group.thetas, grid.parent_index, grid.node_centers, first_child, last_child,
        kind === :m2m)
    return group
end

function _refresh_group_edges_kernel!(source_idx, target_idx, phis::AbstractVector{TF},
        thetas, parent_index, node_centers, first_child::Int, last_child::Int,
        child_to_parent::Bool) where TF
    i = 0
    @inbounds for child in first_child:last_child
        parent = parent_index[child]
        i += 1
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
        _, theta, phi = cartesian_to_spherical(SVector{3,TF}(dx, dy, dz))
        phis[i] = TF(phi)
        thetas[i] = TF(theta)
    end
    return nothing
end

function _refresh_radix_coords!(coords::Vector{SVector{3,Int}}, cell_keys,
        n_cells::Int, ell::Int)
    @inbounds for cell in 1:n_cells
        coords[cell] = morton_decode(cell_keys[cell], ell)
    end
    return coords
end

function _refresh_radix_tree_routes!(m2m_parent::Vector{Int}, m2m_child::Vector{Int},
        l2l_parent::Vector{Int}, l2l_child::Vector{Int}, parent_index, n_edges::Int)
    @inbounds for edge in 1:n_edges
        node = edge + 1
        parent = parent_index[node]
        m2m_parent[edge] = parent
        m2m_child[edge] = node
        l2l_parent[edge] = parent
        l2l_child[edge] = node
    end
    return nothing
end

function _pack_radix_source_bodies!(source_bodies::AbstractMatrix{TF}, perm, body_system,
        body_index, source_buffers::Tuple, n::Int) where TF
    @inbounds for sorted_i in 1:n
        global_i = perm[sorted_i]
        isys = body_system[global_i]
        ibody = body_index[global_i]
        src = source_buffers[isys]
        source_bodies[1, sorted_i] = src[1, ibody]
        source_bodies[2, sorted_i] = src[2, ibody]
        source_bodies[3, sorted_i] = src[3, ibody]
        source_bodies[4, sorted_i] = zero(TF)
        source_bodies[5, sorted_i] = src[5, ibody]
    end
    return source_bodies
end

function _refresh_hierarchical_route_telemetry!(state,
        ctx::HostHierarchicalM2LContext)
    noffsets = length(ctx.tables.push_offsets)
    total = 0
    fill!(ctx.routes_per_level, 0)
    last_count = 0
    for level in 2:state.grid.ell
        level_total = 0
        for first_offset in 1:ctx.window_classes:noffsets
            last_offset = min(first_offset + ctx.window_classes - 1, noffsets)
            last_count = build_hierarchical_routes_window!(
                state.route_levels, state.route_offsets, state.route_targets,
                state.route_sources, ctx.apply_plan.route_class,
                ctx, state.grid, level, first_offset, last_offset)
            level_total += last_count
        end
        ctx.routes_per_level[level + 1] = level_total
        total += level_total
    end
    ctx.total_routes = total
    ctx.last_window_routes = last_count
    return total
end

"""
    update_radix_state!(cache, systems)

Refresh every step-varying part of the cache's resident state from the systems'
current positions and strengths: grid (fixed Morton domain), packed source
bodies, occupancy map, M2L routes + direct pairs, tree edges, per-level operator
group columns, and the step counts. No array is reallocated. Returns the cache.
"""
function update_radix_state!(cache::RadixFMMCache{TF,LH}, systems::Tuple) where {TF,LH}
    length(systems) == cache.n_systems ||
        throw(ArgumentError("cache was built for $(cache.n_systems) source systems, got $(length(systems))"))
    state = cache.state
    grid = state.grid
    ctx = state.interaction_list
    hierarchical = ctx isa HostHierarchicalM2LContext
    profiling = hierarchical && ctx.profile_stages
    profiling && fill!(ctx.update_stage_ns, 0)
    n = get_n_bodies(systems)
    n > 0 || throw(ArgumentError("update_radix_state! requires at least one body"))
    n <= cache.max_n_bodies ||
        throw(ArgumentError("n=$n exceeds the cache capacity max_n_bodies=$(cache.max_n_bodies)"))
    _assert_radix_positions_in_box(systems, cache.x_min, cache.h0)

    t_stage = profiling ? time_ns() : UInt64(0)
    update_radix_grid!(grid, systems, cache.body_keys, cache.sort_scratch,
        cache.sort_counts, cache.sort_offsets, cache.level_offsets)
    profiling && (ctx.update_stage_ns[1] = time_ns() - t_stage)
    n_cells = grid.n_cells
    n_nodes = cache.level_offsets[end]

    for (isys, system) in enumerate(systems)
        source_to_buffer!(cache.source_buffers[isys], system, 1:get_n_bodies(system))
    end
    _pack_radix_source_bodies!(state.source_bodies, grid.perm, grid.body_system,
        grid.body_index, cache.source_buffers, n)

    resize!(cache.coords, n_cells)
    _refresh_radix_coords!(cache.coords, grid.cell_keys, n_cells, grid.ell)
    if hierarchical
        t_stage = profiling ? time_ns() : UInt64(0)
        copyto!(ctx.level_offsets, cache.level_offsets)
        refresh_radix_level_occupancy!(ctx.occupancy, grid, cache.level_offsets)
        profiling && (ctx.update_stage_ns[2] = time_ns() - t_stage)
    else
        refresh_cell_at!(cache.cell_at, grid.cell_keys, n_cells, grid.ell)
    end

    plan = state.scratch.m2l_concat
    if hierarchical
        t_stage = profiling ? time_ns() : UInt64(0)
        n_direct = build_hierarchical_direct_pairs!(state.direct_targets,
            state.direct_sources, ctx, grid, n_cells)
        profiling && (ctx.update_stage_ns[3] = time_ns() - t_stage)
        t_stage = profiling ? time_ns() : UInt64(0)
        n_routes = _refresh_hierarchical_route_telemetry!(state, ctx)
        profiling && (ctx.update_stage_ns[4] = time_ns() - t_stage)
    else
        n_routes, n_direct = build_radix_routes!(
            state.route_levels, state.route_offsets, state.route_targets,
            state.route_sources, plan === nothing ? nothing : plan.route_class,
            state.direct_targets, state.direct_sources,
            cache.accepted_offsets, cache.rejected_offsets, cache.cell_at, cache.coords,
            grid.leaf_to_node, grid.ell, n_cells,
        )
    end

    !hierarchical && plan isa ResidentM2LFactoredPlan && _refresh_factored_m2l_routes!(plan,
        state.route_sources::Vector{Int}, state.route_targets::Vector{Int}, n_routes)
    !hierarchical && plan isa ResidentM2LPrecomputedYPlan && _refresh_precomputed_y_m2l_routes!(plan,
        state.route_sources::Vector{Int}, state.route_targets::Vector{Int}, n_routes)
    !hierarchical && plan isa ResidentM2LDensePlan && _refresh_dense_m2l_routes!(plan,
        state.route_sources::Vector{Int}, state.route_targets::Vector{Int}, n_routes)

    t_stage = profiling ? time_ns() : UInt64(0)
    n_edges = max(n_nodes - 1, 0)
    resize!(state.m2m_parent_routes, n_edges)
    resize!(state.m2m_child_routes, n_edges)
    resize!(state.l2l_parent_routes, n_edges)
    resize!(state.l2l_child_routes, n_edges)
    _refresh_radix_tree_routes!(state.m2m_parent_routes, state.m2m_child_routes,
        state.l2l_parent_routes, state.l2l_child_routes, grid.parent_index, n_edges)

    _refresh_resident_stage_groups!(state.scratch, grid, cache.level_offsets)
    profiling && (ctx.update_stage_ns[5] = time_ns() - t_stage)

    counts = state.counts
    counts.n_bodies = n
    counts.n_cells = n_cells
    counts.n_nodes = n_nodes
    counts.n_routes = n_routes
    counts.n_direct = n_direct
    cache.step += 1
    return cache
end

function _refresh_precomputed_y_m2l_routes!(plan::ResidentM2LPrecomputedYPlan,
        route_sources::Vector{Int}, route_targets::Vector{Int}, n_routes::Int)
    fill!(plan.offset_counts, 0)
    fill!(plan.angle_counts, 0)
    @inbounds for i in 1:n_routes
        offset = Int(plan.route_class[i])
        plan.offset_counts[offset] += 1
        plan.angle_counts[plan.offset_to_angle[offset]] += 1
    end
    # Prefixes follow the immutable angle-major / offset-minor order.
    cursor = 1
    @inbounds for angle in eachindex(plan.angle_counts)
        plan.angle_starts[angle] = cursor
        for p in plan.angle_offset_starts[angle]:(plan.angle_offset_starts[angle + 1] - 1)
            offset = plan.angle_offsets[p]
            plan.offset_starts[offset] = cursor
            cursor += plan.offset_counts[offset]
        end
    end
    plan.angle_starts[end] = cursor
    plan.offset_starts[end] = cursor
    cursor - 1 == n_routes || throw(AssertionError("precomputed-y route histogram mismatch"))

    # Reuse offset_starts as insertion cursors.  Traversing the original route
    # prefix makes ordering stable within every offset range.
    @inbounds for i in 1:n_routes
        offset = Int(plan.route_class[i])
        dst = plan.offset_starts[offset]
        plan.packed_sources[dst] = route_sources[i]
        plan.packed_targets[dst] = route_targets[i]
        plan.packed_phis[dst] = plan.offset_phis[offset]
        plan.offset_starts[offset] = dst + 1
    end
    # Restore the public starts in place; array identity never changes.
    cursor = 1
    @inbounds for angle in eachindex(plan.angle_counts)
        for p in plan.angle_offset_starts[angle]:(plan.angle_offset_starts[angle + 1] - 1)
            offset = plan.angle_offsets[p]
            plan.offset_starts[offset] = cursor
            cursor += plan.offset_counts[offset]
        end
    end
    plan.offset_starts[end] = cursor
    return plan
end

function _refresh_dense_m2l_routes!(plan::ResidentM2LDensePlan,
        route_sources::Vector{Int}, route_targets::Vector{Int}, n_routes::Int)
    0 <= n_routes <= length(plan.route_class) || throw(ArgumentError(
        "dense M2L route count $n_routes exceeds plan capacity $(length(plan.route_class))"))
    fill!(plan.class_counts, 0)
    @inbounds for i in 1:n_routes
        cls = Int(plan.route_class[i])
        1 <= cls <= length(plan.class_counts) || throw(AssertionError(
            "dense M2L route $i has invalid class $cls"))
        plan.class_counts[cls] += 1
    end
    cursor = 1
    @inbounds for cls in eachindex(plan.class_counts)
        count = plan.class_counts[cls]
        count <= plan.class_capacities[cls] || throw(AssertionError(
            "dense M2L class $cls count $count exceeds capacity $(plan.class_capacities[cls])"))
        plan.class_starts[cls] = cursor
        cursor += count
    end
    plan.class_starts[end] = cursor
    cursor - 1 == n_routes || throw(AssertionError("dense M2L route histogram mismatch"))

    # Reuse starts as stable insertion cursors, then restore the public prefixes.
    @inbounds for i in 1:n_routes
        cls = Int(plan.route_class[i])
        dst = plan.class_starts[cls]
        plan.packed_sources[dst] = route_sources[i]
        plan.packed_targets[dst] = route_targets[i]
        plan.class_starts[cls] = dst + 1
    end
    cursor = 1
    @inbounds for cls in eachindex(plan.class_counts)
        plan.class_starts[cls] = cursor
        cursor += plan.class_counts[cls]
    end
    plan.class_starts[end] = cursor
    return plan
end

function _refresh_factored_m2l_routes!(plan::ResidentM2LFactoredPlan{R,G},
        route_sources::Vector{Int}, route_targets::Vector{Int}, n_routes::Int) where {R,G}
    for group in plan.groups
        group.count[] = 0
    end
    route_class = plan.route_class::Vector{Int32}
    @inbounds for i in 1:n_routes
        group = plan.groups[route_class[i]]
        j = group.count[] + 1
        (group.source_idx::Vector{Int})[j] = route_sources[i]
        (group.target_idx::Vector{Int})[j] = route_targets[i]
        group.count[] = j
    end
    return plan
end

update_radix_state!(cache::RadixFMMCache, systems) =
    update_radix_state!(cache, to_tuple(systems))

# Preallocated per-switch-layout scatter buffers for the recurring finalize; the
# buffers are rebuilt only when the requested derivatives layout changes (rare).
function _radix_cache_target_buffers!(cache::RadixFMMCache{TF}, switches::Tuple) where TF
    tb = cache.target_buffers
    if tb === nothing || tb.switches != switches
        buffers = Tuple(zeros(TF, target_buffer_rows(switch), cache.max_n_bodies)
                        for switch in switches)
        cache.target_buffers = (; switches, buffers)
    end
    return cache.target_buffers.buffers
end

# Device-resident construction/step; redefined by translate_batched_cuda.jl (task
# 023 step 7) once load_cuda_radix_lifecycle!() has run.
function _radix_cache_device_build(args...)
    throw(CUDARadixUnavailable(cuda_radix_status()))
end

# Device-resident dense M2L plan construction (task 023f); redefined by
# translate_batched_cuda.jl. Reached only from the device-mode branch of
# _radix_cache_workspace, so on a CPU-only build this stub never runs, but keep it
# defined so the host method resolves.
function _build_cuda_dense_m2l_plan(args...)
    throw(CUDARadixUnavailable(cuda_radix_status()))
end

function _radix_cache_device_step!(cache::RadixFMMCache, targets::Tuple, switches::Tuple)
    throw(CUDARadixUnavailable(cuda_radix_status()))
end
