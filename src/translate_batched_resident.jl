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
    # canonical all-rows packed layout (task 032, spec §3 decision (c)): every
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

#------- vortex B2M (task 032) -------#
#
# Verbatim port of the legacy `mirrored_source_to_vortex!` (bodytomultipole.jl)
# to the resident flat-buffer layout: regular harmonics of the *mirrored*
# offset `-Δx` evaluated on the fly (matching the scalar resident B2M's
# per-(n,m) recurrence style), with the legacy `get_n`/`get_nm1` negative-m
# conjugate-symmetry rules folded into `_resident_vortex_q`. No sign changes:
# the legacy chain `evaluate_local ∘ multipole_to_local! ∘ vortex B2M` was
# verified machine-exact against the analytic Biot-Savart field for an
# off-center vorton, and the resident M2L and L2B were verified numerically
# identical to those legacy stages — so the legacy vortex coefficients are the
# physical convention here. (The legacy *Point{Source}* B2M's strength negation
# is a legacy-pipeline quirk the resident scalar B2M deliberately omits; it has
# no analogue for the vortex.)

@inline function _resident_vortex_q(mdx, mdy, mdz, n, m)
    TF = typeof(mdx)
    if m < 0
        # conjugate symmetry per legacy get_n/get_nm1: Q_{n,-1} = -conj(Q_{n,1})
        (m == -1 && n >= 1) || return zero(TF), zero(TF)
        qre, qim = _resident_regular_harmonic_coeff(mdx, mdy, mdz, n, 1)
        return -qre, qim
    end
    (m > n || n < 0) && return zero(TF), zero(TF)
    return _resident_regular_harmonic_coeff(mdx, mdy, mdz, n, m)
end

@inline function _resident_vortex_phi_contrib(mdx, mdy, mdz, vx, vy, vz, n, m)
    TF = typeof(mdx)
    qmm1_re, qmm1_im = _resident_vortex_q(mdx, mdy, mdz, n, m - 1)
    qm_re, qm_im = _resident_vortex_q(mdx, mdy, mdz, n, m)
    qmp1_re, qmp1_im = _resident_vortex_q(mdx, mdy, mdz, n, m + 1)
    nmmp1_2 = TF(n - m + 1) * TF(0.5)
    npmp1_2 = TF(n + m + 1) * TF(0.5)
    _1_np1 = inv(TF(n + 1))
    _1_m = isodd(m) ? -one(TF) : one(TF)
    re = _1_m * ((-vx * qmm1_re + vy * qmm1_im) * nmmp1_2 +
                 (vx * qmp1_re + vy * qmp1_im) * npmp1_2 - vz * m * qm_im) * _1_np1
    im = _1_m * ((vx * qmm1_im + vy * qmm1_re) * nmmp1_2 +
                 (-vx * qmp1_im + vy * qmp1_re) * npmp1_2 - vz * m * qm_re) * _1_np1
    return re, im
end

@inline function _resident_vortex_chi_contrib(mdx, mdy, mdz, vx, vy, vz, n, m)
    TF = typeof(mdx)
    qmm1_re, qmm1_im = _resident_vortex_q(mdx, mdy, mdz, n - 1, m - 1)
    qm_re, qm_im = _resident_vortex_q(mdx, mdy, mdz, n - 1, m)
    qmp1_re, qmp1_im = _resident_vortex_q(mdx, mdy, mdz, n - 1, m + 1)
    # legacy get_nm1 zeroes (n-1, m) for m == n and (n-1, m+1) for m+1 >= n;
    # _resident_vortex_q's m > n-1 bound check reproduces both
    _1_over_n = inv(TF(n))
    _1_m = isodd(m) ? -one(TF) : one(TF)
    re = -_1_m * _1_over_n * (TF(0.5) * (-vy * qmm1_re - vx * qmm1_im +
        vy * qmp1_re - vx * qmp1_im) - vz * qm_re)
    im = -_1_m * _1_over_n * (TF(0.5) * (vy * qmm1_im - vx * qmm1_re -
        vy * qmp1_im - vx * qmp1_re) + vz * qm_im)
    return re, im
end

function _host_b2m_vortex_kernel!(ph::AbstractMatrix{TF}, ch, source_bodies,
        cell_ranges, cell_centers, leaf_to_node, P_phi::Int, P_chi::Int,
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
            for k in first_body:(first_body + count - 1)
                mdx = cx - source_bodies[1, k]
                mdy = cy - source_bodies[2, k]
                mdz = cz - source_bodies[3, k]
                vx = source_bodies[5, k]
                vy = source_bodies[6, k]
                vz = source_bodies[7, k]
                re, im = _resident_vortex_phi_contrib(mdx, mdy, mdz, vx, vy, vz, n, m)
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
                mdx = cx - source_bodies[1, k]
                mdy = cy - source_bodies[2, k]
                mdz = cz - source_bodies[3, k]
                vx = source_bodies[5, k]
                vy = source_bodies[6, k]
                vz = source_bodies[7, k]
                re, im = _resident_vortex_chi_contrib(mdx, mdy, mdz, vx, vy, vz, n, m)
                acc_re += re
                acc_im += im
            end
            row = flat_basis_index(n, m, 1)
            ch[row, node] = acc_re
            ch[row + 1, node] = acc_im
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

# Iterate the flat pair arrays (bounded by counts) rather than the one-shot
# interaction list so the recurring update path never rebuilds the list object;
# function barrier as in _launch_host_b2m!. Since stage 2 the pair math comes
# from the `direct_kernel` functor stamped into the options at construction
# (compile-time specialization; the legacy hard-coded kernels below remain as
# the functor-abstraction benchmark reference).
function _add_host_direct_pairs!(state::DeviceResidentRadixState)
    hsv = size(state.output, 1) >= 13 ? Val(true) : Val(false)
    # task 037f: cheapened g/h mode (host maps :lut -> :shipped, see
    # _validated_host_gh_mode); Val() barrier specializes the loop per mode
    _host_direct_pairs_functor_kernel!(state.options.direct_kernel, state.output,
        state.source_bodies, state.cell_ranges, state.direct_targets,
        state.direct_sources, state.counts.n_direct, hsv,
        Val(_validated_host_gh_mode()))
    return state
end

function _host_direct_pairs_functor_kernel!(kernel::AbstractDirectKernel,
        output::AbstractMatrix{TF}, source_bodies, cell_ranges, direct_targets,
        direct_sources, n_direct::Int, ::Val{HS},
        ghv::Val=Val(:shipped)) where {TF,HS}
    ep = _emits_potential(kernel)
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
                if HS
                    u, gx, gy, gz, h1, h2, h3, h4, h5, h6, h7, h8, h9 =
                        _direct_pair_ugh(kernel, dx, dy, dz, r2, invr,
                            source_bodies, j, ghv)
                    ep && (output[1, i] += u)
                    output[2, i] += gx
                    output[3, i] += gy
                    output[4, i] += gz
                    output[5, i] += h1
                    output[6, i] += h2
                    output[7, i] += h3
                    output[8, i] += h4
                    output[9, i] += h5
                    output[10, i] += h6
                    output[11, i] += h7
                    output[12, i] += h8
                    output[13, i] += h9
                else
                    u, gx, gy, gz = _direct_pair_ug(kernel, dx, dy, dz, r2, invr,
                        source_bodies, j, ghv)
                    ep && (output[1, i] += u)
                    output[2, i] += gx
                    output[3, i] += gy
                    output[4, i] += gz
                end
            end
        end
    end
    return output
end

#------- two-pass additive-correction deficit sweep (task 032a stage B) -------#
#
# Pass 2 of the TwoPassVortex hybrid: after the unmodified pass 1 (singular far
# field + the rho_c-partitioned direct nearfield above), add the 031a §6.1
# deficit for every pair with rho_c < ρ = r/σ_src ≤ rho_t, wherever pass 1
# routed that pair (direct or M2L — the deficit is additive, so no exact-once
# bookkeeping exists to get wrong).
#
# Traversal geometry: instead of a second stored direct route list, the sweep
# enumerates the offset ball arithmetically — for each occupied leaf cell all
# integer offsets with Chebyshev radius ≤ R = ⌊rho_t·σ_max/h_leaf⌋ + 1, pruned
# per offset by the minimum-gap test gap(o)·h_leaf ≤ rho_t·σ_max (gap(o) is the
# closest approach of two cells at lattice offset o), each resolved to an
# occupied cell by binary search on the sorted leaf Morton keys. Because R is
# recomputed from the live σ_max every evaluation, pass-2 reach covers
# rho_t·σ_max by construction — offsets just beyond R have gap ≥ R·h_leaf >
# rho_t·σ_max — so the reach requirement the Stage-A gate enforces for the
# single-pass kernels holds here without a gate, and the primary near set only
# needs the rho_c adequacy (_direct_kernel_geometry_gate! dispatch below).
# Zero per-step allocation: loop bounds and binary searches only. The stage-C
# CUDA mirror can materialize the same pruned ball as a compacted class list.
_add_host_twopass_deficit!(state::DeviceResidentRadixState) =
    _host_twopass_deficit_dispatch!(state, state.options.direct_kernel)

_host_twopass_deficit_dispatch!(state::DeviceResidentRadixState, ::AbstractDirectKernel) = state

function _host_twopass_deficit_dispatch!(state::DeviceResidentRadixState,
        kernel::TwoPassVortex)
    hsv = size(state.output, 1) >= 13 ? Val(true) : Val(false)
    _host_twopass_deficit_kernel!(kernel, state.output, state.source_bodies,
        state.cell_ranges, state.grid.cell_keys, state.counts.n_cells,
        state.counts.n_bodies, state.grid.ell, Float64(state.grid.h0), hsv)
    return state
end

# Occupied-leaf lookup by Morton key over the sorted prefix cell_keys[1:n_cells]
# (cells are emitted in key order by the radix sort); returns 0 when the coord
# is outside the grid or the cell is empty.
@inline function _twopass_cell_lookup(cell_keys, n_cells::Int, ell::Int,
        x::Int, y::Int, z::Int)
    G = 1 << ell
    (0 <= x < G && 0 <= y < G && 0 <= z < G) || return 0
    key = morton_key(SVector{3,Int}(x, y, z), ell)
    lo, hi = 1, n_cells
    @inbounds while lo <= hi
        mid = (lo + hi) >>> 1
        k = cell_keys[mid]
        if k < key
            lo = mid + 1
        elseif k > key
            hi = mid - 1
        else
            return mid
        end
    end
    return 0
end

function _host_twopass_deficit_kernel!(kernel::TwoPassVortex,
        output::AbstractMatrix{TF}, source_bodies, cell_ranges, cell_keys,
        n_cells::Int, n_bodies::Int, ell::Int, h0::Float64,
        ::Val{HS}) where {TF,HS}
    sigma_row = kernel.sigma_row
    # live sigma_max (allocation-free; mirrors the adequacy gate's reduction)
    sigma_max = zero(TF)
    @inbounds for j in 1:n_bodies
        s = source_bodies[sigma_row, j]
        s > sigma_max && (sigma_max = s)
    end
    sigma_max > zero(TF) || return output
    h_leaf = 2 * h0 / (1 << ell)
    reach = kernel.rho_t * Float64(sigma_max)
    R = floor(Int, reach / h_leaf) + 1
    reach2 = reach * reach
    hl2 = h_leaf * h_leaf
    rho_c = TF(kernel.rho_c)
    rho_t = TF(kernel.rho_t)
    @inbounds for target_cell in 1:n_cells
        tfirst = cell_ranges[1, target_cell]
        tcount = cell_ranges[2, target_cell]
        tcount > 0 || continue
        tcoord = morton_decode(cell_keys[target_cell], ell)
        for oz in -R:R, oy in -R:R, ox in -R:R
            # minimum-gap pruning: cells at offset o cannot hold a pair inside
            # the reach when their closest approach already exceeds it
            gx = max(abs(ox) - 1, 0)
            gy = max(abs(oy) - 1, 0)
            gz = max(abs(oz) - 1, 0)
            (gx * gx + gy * gy + gz * gz) * hl2 > reach2 && continue
            source_cell = _twopass_cell_lookup(cell_keys, n_cells, ell,
                tcoord[1] + ox, tcoord[2] + oy, tcoord[3] + oz)
            source_cell == 0 && continue
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
                    sigma = source_bodies[sigma_row, j]
                    sigma > zero(TF) || continue
                    invr = inv(sqrt(r2))
                    rho = r2 * invr / sigma
                    (rho_c < rho <= rho_t) || continue
                    gbar, rhogp = _gaussianerf_gbar_rhogp(rho)
                    ge = -gbar
                    he = muladd(TF(3), gbar, rhogp)
                    gsx = source_bodies[5, j]
                    gsy = source_bodies[6, j]
                    gsz = source_bodies[7, j]
                    if HS
                        _, ux, uy, uz, h1, h2, h3, h4, h5, h6, h7, h8, h9 =
                            _vortex_pair_ugh(dx, dy, dz, r2, invr,
                                gsx, gsy, gsz, ge, he)
                        output[2, i] += ux
                        output[3, i] += uy
                        output[4, i] += uz
                        output[5, i] += h1
                        output[6, i] += h2
                        output[7, i] += h3
                        output[8, i] += h4
                        output[9, i] += h5
                        output[10, i] += h6
                        output[11, i] += h7
                        output[12, i] += h8
                        output[13, i] += h9
                    else
                        _, ux, uy, uz = _vortex_pair_ug(dx, dy, dz, invr,
                            gsx, gsy, gsz, ge)
                        output[2, i] += ux
                        output[3, i] += uy
                        output[4, i] += uz
                    end
                end
            end
        end
    end
    return output
end

#------- gaussianerf g/h evaluation and direct-kernel pair functors (032 stage 2) -------#
#
# Erf-free evaluation of the gaussianerf regularization factor
#   g(ρ) = erf(ρ/√2) − Aρe^{−ρ²/2},  A = √(2/π),
# and the combined Jacobian numerator h(ρ) = ρg′(ρ) − 3g(ρ), used by the
# `RegularizedVortex` nearfield (`theory/kernel-splitting-nearfield.md` §3, §6.2).
#
# Below ρ = 2: the cancellation-safe alternating Horner series
#   g = Aρ³ Σ_k (−1)^k ρ^{2k}/((2k+3) 2^k k!),  h = Aρ⁵ Σ_k (−1)^{k+1} ρ^{2k}/((2k+5) 2^k k!),
# with term counts measured against a 256-bit reference over the whole branch
# (scripts/fit_032_nearfield_g.jl → data/kernel_splitting/nearfield_g_eval.csv):
# 13 terms hold ≤ 6.8e-7 relative in Float32 and 19 terms ≤ 1.7e-12 in Float64
# (the theory-§3 6/10-term counts are valid only to its ρ = 0.5 partitioning
# switch; the erf-free design runs the series to ρ = 2, hence the re-measured
# counts). Above ρ = 2: ḡ = e^{−ρ²/2}(Aρ + s), with s ≈ degree-3 polynomial in
# u = 1/ρ² least-squares fitted on [2, 4.789]; g = 1 − ḡ holds ≤ 2.1e-4 absolute
# against the 3.69e-4 budget (031a §6.2: absolute tolerance suffices where the
# retained result is O(1)), decaying like e^{−ρ²/2} beyond ρ_t, with the correct
# singular limit (g→1, h→−3). One hardware exp, no erf, both branches GPU-safe.
const _GAUSSERF_A = 0.7978845608028654           # √(2/π)
const _GAUSSERF_G_COEFFS = Tuple(Float64((isodd(k) ? -1 : 1) //
    ((2k + 3) * BigInt(2)^k * factorial(BigInt(k)))) for k in 0:18)
const _GAUSSERF_H_COEFFS = Tuple(Float64((iseven(k) ? -1 : 1) //
    ((2k + 5) * BigInt(2)^k * factorial(BigInt(k)))) for k in 0:18)
const _GAUSSERF_G_COEFFS32 = Tuple(Float32(c) for c in _GAUSSERF_G_COEFFS[1:13])
const _GAUSSERF_H_COEFFS32 = Tuple(Float32(c) for c in _GAUSSERF_H_COEFFS[1:13])
# degree-3 fit of s(u) on [ρ_c = 2, ρ_t = 4.789] (fit_032_nearfield_g.jl)
const _GAUSSERF_S_COEFFS = (0.082593826443677007, 2.0801015208954681,
    -6.8585923004500211, 10.482822830062796)

@inline _gausserf_series_g(z::Float64) = evalpoly(z, _GAUSSERF_G_COEFFS)
@inline _gausserf_series_h(z::Float64) = evalpoly(z, _GAUSSERF_H_COEFFS)
@inline _gausserf_series_g(z::Float32) = evalpoly(z, _GAUSSERF_G_COEFFS32)
@inline _gausserf_series_h(z::Float32) = evalpoly(z, _GAUSSERF_H_COEFFS32)

# Outer-branch (ρ > 2) complement pair: ḡ = e^{−ρ²/2}(Aρ + s(1/ρ²)) with the
# 031a §6.2 degree-3 fit of s, and ρg′ = Aρ³e^{−ρ²/2} reusing the same
# exponential. Factored out of `_gaussianerf_g_h` so the two-pass deficit
# (task 032a stage B) consumes ḡ directly to absolute tolerance instead of
# reconstructing it as 1 − g.
@inline function _gaussianerf_gbar_rhogp(rho::T) where T<:AbstractFloat
    z = rho * rho
    e = exp(-z / 2)
    u = inv(z)
    s = muladd(muladd(muladd(T(_GAUSSERF_S_COEFFS[4]), u, T(_GAUSSERF_S_COEFFS[3])),
        u, T(_GAUSSERF_S_COEFFS[2])), u, T(_GAUSSERF_S_COEFFS[1]))
    gbar = e * muladd(T(_GAUSSERF_A), rho, s)
    rhogp = T(_GAUSSERF_A) * rho * z * e
    return gbar, rhogp
end

@inline function _gaussianerf_g_h(rho::T) where T<:AbstractFloat
    z = rho * rho
    if rho <= T(2)
        g = T(_GAUSSERF_A) * rho * z * _gausserf_series_g(z)
        h = T(_GAUSSERF_A) * rho * z * z * _gausserf_series_h(z)
        return g, h
    end
    gbar, rhogp = _gaussianerf_gbar_rhogp(rho)
    g = one(T) - gbar
    h = rhogp - 3 * g
    return g, h
end

# Per-pair math behind the `direct_kernel` functor trait. Shared verbatim by the
# host loops and the CUDA pair kernels (arithmetic + exp only); the caller
# computes `invr` with its preferred reciprocal sqrt and skips r2 == 0 pairs.
# Conventions (theory §1 / legacy direct!): dx,dy,dz = target − source;
# crss_i = −(Δx×Γ)_i/(4πr³); U = g·crss; J[i,j] = ∂u_i/∂x_j column-major with
# a = h/r², b = −g/(4πr³).

@inline function _direct_pair_ug(::SingularSource, dx, dy, dz, r2, invr,
        source_bodies, j)
    T = typeof(r2)
    @inbounds q = source_bodies[5, j] * inv(T(4) * T(π))
    u = q * invr
    invr3 = invr * invr * invr
    return u, -q * dx * invr3, -q * dy * invr3, -q * dz * invr3
end

@inline function _direct_pair_ugh(::SingularSource, dx, dy, dz, r2, invr,
        source_bodies, j)
    T = typeof(r2)
    @inbounds q = source_bodies[5, j] * inv(T(4) * T(π))
    u = q * invr
    invr2 = invr * invr
    invr3 = invr * invr2
    q3invr5 = 3 * q * invr3 * invr2
    qinvr3 = q * invr3
    return (u, -q * dx * invr3, -q * dy * invr3, -q * dz * invr3,
        q3invr5 * dx * dx - qinvr3, q3invr5 * dx * dy, q3invr5 * dx * dz,
        q3invr5 * dy * dx, q3invr5 * dy * dy - qinvr3, q3invr5 * dy * dz,
        q3invr5 * dz * dx, q3invr5 * dz * dy, q3invr5 * dz * dz - qinvr3)
end

# Shared vortex U/J assembly for a given regularization pair (g, h); g = 1,
# h = −3 reproduces the singular Biot-Savart kernel exactly.
@inline function _vortex_pair_ug(dx, dy, dz, invr, gsx, gsy, gsz, g)
    T = typeof(dx)
    cr3 = inv(T(4) * T(π)) * invr * invr * invr
    ux = (dz * gsy - dy * gsz) * cr3
    uy = (dx * gsz - dz * gsx) * cr3
    uz = (dy * gsx - dx * gsy) * cr3
    return zero(T), g * ux, g * uy, g * uz
end

@inline function _vortex_pair_ugh(dx, dy, dz, r2, invr, gsx, gsy, gsz, g, h)
    T = typeof(dx)
    cr3 = inv(T(4) * T(π)) * invr * invr * invr
    crss1 = (dz * gsy - dy * gsz) * cr3
    crss2 = (dx * gsz - dz * gsx) * cr3
    crss3 = (dy * gsx - dx * gsy) * cr3
    a = h * invr * invr
    b = -g * cr3
    return (zero(T), g * crss1, g * crss2, g * crss3,
        a * crss1 * dx, a * crss2 * dx - b * gsz, a * crss3 * dx + b * gsy,
        a * crss1 * dy + b * gsz, a * crss2 * dy, a * crss3 * dy - b * gsx,
        a * crss1 * dz - b * gsy, a * crss2 * dz + b * gsx, a * crss3 * dz)
end

@inline function _direct_pair_ug(::SingularVortex, dx, dy, dz, r2, invr,
        source_bodies, j)
    T = typeof(r2)
    @inbounds gsx = source_bodies[5, j]
    @inbounds gsy = source_bodies[6, j]
    @inbounds gsz = source_bodies[7, j]
    return _vortex_pair_ug(dx, dy, dz, invr, gsx, gsy, gsz, one(T))
end

@inline function _direct_pair_ugh(::SingularVortex, dx, dy, dz, r2, invr,
        source_bodies, j)
    T = typeof(r2)
    @inbounds gsx = source_bodies[5, j]
    @inbounds gsy = source_bodies[6, j]
    @inbounds gsz = source_bodies[7, j]
    return _vortex_pair_ugh(dx, dy, dz, r2, invr, gsx, gsy, gsz, one(T), -T(3))
end

@inline function _direct_pair_ug(kernel::RegularizedVortex, dx, dy, dz, r2, invr,
        source_bodies, j)
    T = typeof(r2)
    @inbounds gsx = source_bodies[5, j]
    @inbounds gsy = source_bodies[6, j]
    @inbounds gsz = source_bodies[7, j]
    @inbounds sigma = source_bodies[kernel.sigma_row, j]
    g = one(T)
    if sigma > zero(T)
        g, _ = _gaussianerf_g_h(r2 * invr / sigma)
    end
    return _vortex_pair_ug(dx, dy, dz, invr, gsx, gsy, gsz, g)
end

@inline function _direct_pair_ugh(kernel::RegularizedVortex, dx, dy, dz, r2, invr,
        source_bodies, j)
    T = typeof(r2)
    @inbounds gsx = source_bodies[5, j]
    @inbounds gsy = source_bodies[6, j]
    @inbounds gsz = source_bodies[7, j]
    @inbounds sigma = source_bodies[kernel.sigma_row, j]
    g = one(T)
    h = -T(3)
    if sigma > zero(T)
        g, h = _gaussianerf_g_h(r2 * invr / sigma)
    end
    return _vortex_pair_ugh(dx, dy, dz, r2, invr, gsx, gsy, gsz, g, h)
end

# Partitioned replacement (task 032a candidate 2, theory §2): stable regularized
# g/h inside the smoothing cutoff, exact singular limits beyond it — the branch
# selects HOW a direct pair is evaluated, never WHICH pairs are direct (the
# adequacy gate guarantees every cutoff pair is in the direct set).
#
# The two-pass hybrid's pass 1 (task 032a stage B, candidate 3) is the same
# math with the branch at rho_c instead of rho_t: stable regularized U/J for
# ρ ≤ rho_c, exact singular beyond, with the pass-2 deficit sweep
# (_add_host_twopass_deficit!) supplying the correction on (rho_c, rho_t].
@inline _pass1_regularized_cutoff(kernel::PartitionedVortex) = kernel.rho_t
@inline _pass1_regularized_cutoff(kernel::TwoPassVortex) = kernel.rho_c

@inline function _direct_pair_ug(kernel::Union{PartitionedVortex,TwoPassVortex},
        dx, dy, dz, r2, invr, source_bodies, j)
    T = typeof(r2)
    @inbounds gsx = source_bodies[5, j]
    @inbounds gsy = source_bodies[6, j]
    @inbounds gsz = source_bodies[7, j]
    @inbounds sigma = source_bodies[kernel.sigma_row, j]
    g = one(T)
    if sigma > zero(T)
        rho = r2 * invr / sigma
        if rho <= T(_pass1_regularized_cutoff(kernel))
            g, _ = _gaussianerf_g_h(rho)
        end
    end
    return _vortex_pair_ug(dx, dy, dz, invr, gsx, gsy, gsz, g)
end

@inline function _direct_pair_ugh(kernel::Union{PartitionedVortex,TwoPassVortex},
        dx, dy, dz, r2, invr, source_bodies, j)
    T = typeof(r2)
    @inbounds gsx = source_bodies[5, j]
    @inbounds gsy = source_bodies[6, j]
    @inbounds gsz = source_bodies[7, j]
    @inbounds sigma = source_bodies[kernel.sigma_row, j]
    g = one(T)
    h = -T(3)
    if sigma > zero(T)
        rho = r2 * invr / sigma
        if rho <= T(_pass1_regularized_cutoff(kernel))
            g, h = _gaussianerf_g_h(rho)
        end
    end
    return _vortex_pair_ugh(dx, dy, dz, r2, invr, gsx, gsy, gsz, g, h)
end

# Two-pass deficit coefficients (031a §6.1) as an effective (g, h) pair for the
# shared vortex U/J assembly: with g_e = −ḡ and h_e = ρg′ + 3ḡ,
# `_vortex_pair_ug(h)` yields exactly ΔU = −ḡC, Δa = h_e/r² = (ρg′+3ḡ)/r², and
# Δb = −g_e/(4πr³) = ḡ/(4πr³), and (singular) + (deficit) = (regularized)
# identically: (1 − ḡ, ρg′ + 3ḡ − 3) = (g, ρg′ − 3g). Pairs outside the shell
# (rho_c, rho_t] contribute nothing (ρ ≤ rho_c is fully handled by pass 1's
# regularized branch; beyond rho_t the tail is inside the §4 error budget).
@inline function _twopass_deficit_gh(kernel::TwoPassVortex, rho::T) where T<:AbstractFloat
    (T(kernel.rho_c) < rho <= T(kernel.rho_t)) || return zero(T), zero(T)
    gbar, rhogp = _gaussianerf_gbar_rhogp(rho)
    return -gbar, muladd(T(3), gbar, rhogp)
end

#------- cheapened g/h evaluation modes (task 037f) -------#
#
# `CUDA_NEARFIELD_GH_MODE` selects how the regularized-family pair functors
# evaluate the gaussianerf (g, h) — budgets, sizing, and the pointwise ->
# delivered-error mapping in `theory/nearfield-kernel-cheapening-budget.md`
# (derivation script `scripts/fm037f_error_budget.jl`):
#
#   :shipped       the unmodified evaluation above (default; every other call
#                  path is bitwise-identical to pre-037f code);
#   :reduced       12-term series in both precisions (from 19/13), outer
#                  branch unchanged (deg-2 s(u) fails the mapped budget);
#   :fp32          Float64 configurations only: the shipped Float32 math
#                  (13-term series + deg-3 outer), pair U/J assembled in
#                  Float32, accumulated in Float64.  On Float32
#                  configurations this is the shipped path (documented no-op);
#   :reduced_fp32  :fp32 with the 12-term reduced series;
#   :lut           device-only shared-memory lookup table (see
#                  translate_batched_cuda.jl).  The HOST reference path falls
#                  back to :shipped under :lut — host/device parity for :lut
#                  is gated at its budgeted pointwise error, not bitwise.
#
# Like the stage-C mechanism Refs, the CUDA side reads this inside the
# lifecycle body, so the selection is baked into a captured CUDA graph at
# record time: flip it only BEFORE cache construction (or force a new epoch),
# or a replayed graph keeps the old mode silently.
#
# DEFAULT = :fp32 (user-approved flip, 2026-08-14, task 037f): on Float64
# configurations the g/h transcendental (and functor-path assembly) runs in
# Float32 with Float64 accumulation — measured +6.8-10.2% end-to-end U/J on
# cube and +7.4-7.8% on the wake at delivered-error deltas of ~1e-8 relative
# RMS (fm037f_screen.csv / fm037f_decomposition.csv, H200 job 13170769). On
# Float32 configurations :fp32 is bitwise the shipped path (documented
# no-op), so this default changes nothing there. :shipped remains the
# control/opt-out.
const NEARFIELD_GH_MODES = (:shipped, :reduced, :fp32, :reduced_fp32, :lut)
const CUDA_NEARFIELD_GH_MODE = Ref{Symbol}(:fp32)

# 12-term truncations of the exact series (task 037f sizing: delta vs shipped
# <= 4.9e-6 relative on the series branch, >= 7x under the coherent-tier
# delivered budget B = 2.66e-4; fm037f_budget.csv)
const _GAUSSERF_G_COEFFS_R = _GAUSSERF_G_COEFFS[1:12]
const _GAUSSERF_H_COEFFS_R = _GAUSSERF_H_COEFFS[1:12]
const _GAUSSERF_G_COEFFS32_R = _GAUSSERF_G_COEFFS32[1:12]
const _GAUSSERF_H_COEFFS32_R = _GAUSSERF_H_COEFFS32[1:12]

@inline _gausserf_series_g_r(z::Float64) = evalpoly(z, _GAUSSERF_G_COEFFS_R)
@inline _gausserf_series_h_r(z::Float64) = evalpoly(z, _GAUSSERF_H_COEFFS_R)
@inline _gausserf_series_g_r(z::Float32) = evalpoly(z, _GAUSSERF_G_COEFFS32_R)
@inline _gausserf_series_h_r(z::Float32) = evalpoly(z, _GAUSSERF_H_COEFFS32_R)

@inline function _gaussianerf_g_h_reduced(rho::T) where T<:AbstractFloat
    z = rho * rho
    if rho <= T(2)
        g = T(_GAUSSERF_A) * rho * z * _gausserf_series_g_r(z)
        h = T(_GAUSSERF_A) * rho * z * z * _gausserf_series_h_r(z)
        return g, h
    end
    gbar, rhogp = _gaussianerf_gbar_rhogp(rho)
    g = one(T) - gbar
    return g, rhogp - 3 * g
end

# scalar mode dispatch (:lut resolves at the kernel level on the device and
# falls back to :shipped here; the fp32 modes narrow rho when the caller has
# not already narrowed the whole pair computation)
@inline _gaussianerf_g_h(rho::T, ::Val{:shipped}) where T<:AbstractFloat =
    _gaussianerf_g_h(rho)
@inline _gaussianerf_g_h(rho::T, ::Val{:reduced}) where T<:AbstractFloat =
    _gaussianerf_g_h_reduced(rho)
@inline _gaussianerf_g_h(rho::T, ::Val{:lut}) where T<:AbstractFloat =
    _gaussianerf_g_h(rho)
@inline _gaussianerf_g_h(rho::Float32, ::Val{:fp32}) = _gaussianerf_g_h(rho)
@inline _gaussianerf_g_h(rho::Float32, ::Val{:reduced_fp32}) =
    _gaussianerf_g_h_reduced(rho)
@inline function _gaussianerf_g_h(rho::Float64, ::Val{:fp32})
    g, h = _gaussianerf_g_h(Float32(rho))
    return Float64(g), Float64(h)
end
@inline function _gaussianerf_g_h(rho::Float64, ::Val{:reduced_fp32})
    g, h = _gaussianerf_g_h_reduced(Float32(rho))
    return Float64(g), Float64(h)
end

# (g, h) under a mode with the kernel's branch structure: RegularizedVortex is
# regularized everywhere; the split kernels switch to singular beyond the
# pass-1 cutoff.  sigma <= 0 padding falls back to singular as shipped.
@inline function _pair_gh_mode(::RegularizedVortex, r2::T, invr::T, sigma::T,
        mv::Val) where T
    sigma > zero(T) || return one(T), -T(3)
    return _gaussianerf_g_h(r2 * invr / sigma, mv)
end
@inline function _pair_gh_mode(kernel::Union{PartitionedVortex,TwoPassVortex},
        r2::T, invr::T, sigma::T, mv::Val) where T
    g = one(T)
    h = -T(3)
    if sigma > zero(T)
        rho = r2 * invr / sigma
        if rho <= T(_pass1_regularized_cutoff(kernel))
            g, h = _gaussianerf_g_h(rho, mv)
        end
    end
    return g, h
end

@inline _widen_pair(::Type{T}, v::NTuple{N,Float32}) where {T,N} =
    ntuple(i -> T(v[i]), Val(N))

# Mode-threaded functor entry points.  The generic fallback ignores the mode,
# so consumer functors keep the documented 8-argument contract and the
# singular kernels are mode-independent; :shipped routes straight to the
# 8-argument methods (bitwise-identical code paths).
@inline _direct_pair_ug(kernel::AbstractDirectKernel, dx, dy, dz, r2, invr,
        source_bodies, j, ::Val) =
    _direct_pair_ug(kernel, dx, dy, dz, r2, invr, source_bodies, j)
@inline _direct_pair_ugh(kernel::AbstractDirectKernel, dx, dy, dz, r2, invr,
        source_bodies, j, ::Val) =
    _direct_pair_ugh(kernel, dx, dy, dz, r2, invr, source_bodies, j)

@inline function _direct_pair_ug(kernel::AbstractRegularizedVortex, dx, dy, dz,
        r2, invr, source_bodies, j, ::Val{GH}) where GH
    T = typeof(r2)
    if GH === :shipped || GH === :lut || (GH === :fp32 && T === Float32)
        return _direct_pair_ug(kernel, dx, dy, dz, r2, invr, source_bodies, j)
    elseif (GH === :fp32 || GH === :reduced_fp32) && T === Float64
        @inbounds gsx = Float32(source_bodies[5, j])
        @inbounds gsy = Float32(source_bodies[6, j])
        @inbounds gsz = Float32(source_bodies[7, j])
        @inbounds sigma = Float32(source_bodies[kernel.sigma_row, j])
        r232 = Float32(r2)
        invr32 = Float32(invr)
        g, _ = _pair_gh_mode(kernel, r232, invr32, sigma,
            GH === :fp32 ? Val(:shipped) : Val(:reduced))
        v = _vortex_pair_ug(Float32(dx), Float32(dy), Float32(dz), invr32,
            gsx, gsy, gsz, g)
        return _widen_pair(T, v)
    else # :reduced (also :reduced_fp32 on a Float32 configuration)
        @inbounds gsx = source_bodies[5, j]
        @inbounds gsy = source_bodies[6, j]
        @inbounds gsz = source_bodies[7, j]
        @inbounds sigma = source_bodies[kernel.sigma_row, j]
        g, _ = _pair_gh_mode(kernel, r2, invr, sigma, Val(:reduced))
        return _vortex_pair_ug(dx, dy, dz, invr, gsx, gsy, gsz, g)
    end
end

@inline function _direct_pair_ugh(kernel::AbstractRegularizedVortex, dx, dy, dz,
        r2, invr, source_bodies, j, ::Val{GH}) where GH
    T = typeof(r2)
    if GH === :shipped || GH === :lut || (GH === :fp32 && T === Float32)
        return _direct_pair_ugh(kernel, dx, dy, dz, r2, invr, source_bodies, j)
    elseif (GH === :fp32 || GH === :reduced_fp32) && T === Float64
        @inbounds gsx = Float32(source_bodies[5, j])
        @inbounds gsy = Float32(source_bodies[6, j])
        @inbounds gsz = Float32(source_bodies[7, j])
        @inbounds sigma = Float32(source_bodies[kernel.sigma_row, j])
        r232 = Float32(r2)
        invr32 = Float32(invr)
        g, h = _pair_gh_mode(kernel, r232, invr32, sigma,
            GH === :fp32 ? Val(:shipped) : Val(:reduced))
        v = _vortex_pair_ugh(Float32(dx), Float32(dy), Float32(dz), r232,
            invr32, gsx, gsy, gsz, g, h)
        return _widen_pair(T, v)
    else # :reduced (also :reduced_fp32 on a Float32 configuration)
        @inbounds gsx = source_bodies[5, j]
        @inbounds gsy = source_bodies[6, j]
        @inbounds gsz = source_bodies[7, j]
        @inbounds sigma = source_bodies[kernel.sigma_row, j]
        g, h = _pair_gh_mode(kernel, r2, invr, sigma, Val(:reduced))
        return _vortex_pair_ugh(dx, dy, dz, r2, invr, gsx, gsy, gsz, g, h)
    end
end

# Host-side :lut table builder (task 037f; also the construction source of the
# device table).  Linear interpolation in x = rho^2 over [0, rho_t^2] of the
# NORMALIZED functions G(x) = g/rho^3 and H(x) = h/rho^5 — analytic in x with
# G(0) = A/3 != 0, so the table preserves relative accuracy down to rho -> 0.
# Values sample the shipped Float64 evaluator, so the LUT inherits (never adds
# to) the shipped outer-fit error; Float32 storage; N = 1024 sized in
# fm037f_budget.csv (interp delta <= 3.0e-6 relative, 6x under budget).
const _NF_GH_LUT_N = 1024

function _build_gh_lut(rho_t::Float64)
    x_max = rho_t * rho_t
    tab = Matrix{Float32}(undef, 2, _NF_GH_LUT_N)
    tab[1, 1] = Float32(_GAUSSERF_A / 3)
    tab[2, 1] = Float32(-_GAUSSERF_A / 5)
    for i in 2:_NF_GH_LUT_N
        x = x_max * (i - 1) / (_NF_GH_LUT_N - 1)
        rho = sqrt(x)
        g, h = _gaussianerf_g_h(rho)
        tab[1, i] = Float32(g / rho^3)
        tab[2, i] = Float32(h / rho^5)
    end
    return tab
end

# Shared LUT lookup (host mirror of the device math; `tab` is any 2 x N
# indexable).  Returns the singular (1, -3) for x >= x_max — the partitioned
# cutoff itself (half-open boundary, measure zero vs the shipped `<=`).
@inline function _gh_from_lut(tab, rho::T, x_max::T) where T
    x = rho * rho
    x >= x_max && return one(T), -T(3)
    t = x * (T(_NF_GH_LUT_N - 1) / x_max)
    i0 = unsafe_trunc(Int32, t)
    f = t - T(i0)
    i1 = i0 + Int32(1)
    @inbounds G0 = T(tab[1, i1])
    @inbounds G1 = T(tab[1, i1 + Int32(1)])
    @inbounds H0 = T(tab[2, i1])
    @inbounds H1 = T(tab[2, i1 + Int32(1)])
    G = muladd(f, G1 - G0, G0)
    H = muladd(f, H1 - H0, H0)
    return rho * x * G, rho * x * x * H
end

# Validated host-side mode (the host reference path maps :lut -> :shipped)
function _validated_host_gh_mode()
    m = radix_setting(:CUDA_NEARFIELD_GH_MODE)
    m in NEARFIELD_GH_MODES || throw(ArgumentError(
        "CUDA_NEARFIELD_GH_MODE must be one of $(NEARFIELD_GH_MODES); got $m"))
    return m === :lut ? :shipped : m
end

# Singular Biot-Savart direct kernel for Point{Vortex} sources (task 032 stage 1):
# U = -Δx×Γ/(4πr³), J per theory §1 with g→1 (transcribed from the legacy
# test-reference vortex direct!). No scalar potential is produced. Retained
# verbatim (with the scalar kernels below) as the hard-coded reference for the
# stage-2 functor-abstraction benchmark; production dispatch now routes through
# `_host_direct_pairs_functor_kernel!`.
function _host_direct_pairs_vortex_kernel!(output::AbstractMatrix{TF}, source_bodies,
        cell_ranges, direct_targets, direct_sources, n_direct::Int,
        ::Val{HS}) where {TF,HS}
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
                gx = source_bodies[5, j]
                gy = source_bodies[6, j]
                gz = source_bodies[7, j]
                invr = inv(sqrt(r2))
                invr2 = invr * invr
                denom = c * invr * invr2
                output[2, i] += (dz * gy - dy * gz) * denom
                output[3, i] += (dx * gz - dz * gx) * denom
                output[4, i] += (dy * gx - dx * gy) * denom
                if HS
                    denom *= invr2
                    output[5, i] += -3 * dx * (gy * dz - gz * dy) * denom
                    output[6, i] += (-3 * dx * (gz * dx - gx * dz) + gz * r2) * denom
                    output[7, i] += (-3 * dx * (gx * dy - gy * dx) - gy * r2) * denom
                    output[8, i] += (-3 * dy * (gy * dz - gz * dy) - gz * r2) * denom
                    output[9, i] += -3 * dy * (gz * dx - gx * dz) * denom
                    output[10, i] += (-3 * dy * (gx * dy - gy * dx) + gx * r2) * denom
                    output[11, i] += (-3 * dz * (gy * dz - gz * dy) + gy * r2) * denom
                    output[12, i] += (-3 * dz * (gz * dx - gx * dz) - gx * r2) * denom
                    output[13, i] += -3 * dz * (gx * dy - gy * dx) * denom
                end
            end
        end
    end
    return output
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

# Singular scalar direct kernel with the 9-component hessian (task 032):
# u = qc/r, g = -qc·Δx/r³, H = qc·(3ΔxΔxᵀ/r⁵ - I/r³) — symmetric, so the
# column-major linear order equals the row-major one.
function _host_direct_pairs_hessian_kernel!(output::AbstractMatrix{TF}, source_bodies,
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
                invr2 = invr * invr
                invr3 = invr * invr2
                output[2, i] -= q * dx * invr3
                output[3, i] -= q * dy * invr3
                output[4, i] -= q * dz * invr3
                q3invr5 = 3 * q * invr3 * invr2
                qinvr3 = q * invr3
                output[5, i] += q3invr5 * dx * dx - qinvr3
                output[6, i] += q3invr5 * dx * dy
                output[7, i] += q3invr5 * dx * dz
                output[8, i] += q3invr5 * dy * dx
                output[9, i] += q3invr5 * dy * dy - qinvr3
                output[10, i] += q3invr5 * dy * dz
                output[11, i] += q3invr5 * dz * dx
                output[12, i] += q3invr5 * dz * dy
                output[13, i] += q3invr5 * dz * dz - qinvr3
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

# Per-(n, m) local-expansion gradient coefficient (task 032): the values the
# legacy `evaluate_local` stores in its `gradient_n_m` scratch, recomputed on
# the fly from the flat zero-padded accessors so the hessian pass needs no
# per-thread coefficient array. Must stay in lockstep with the coefficient
# blocks inside `_resident_local_eval_flat` above.
@inline function _resident_gradient_coeff(ph, ch, node, P_phi, P_active, n, m,
        ::Val{LH}) where LH
    TF = eltype(ph)
    if m == 0
        phi1c = _resident_flat_phi_re(ph, node, P_phi, n + 1, 1)
        phi1s = _resident_flat_phi_im(ph, node, P_phi, n + 1, 1)
        phi0c = _resident_flat_phi_re(ph, node, P_phi, n + 1, 0)
        phi0s = _resident_flat_phi_im(ph, node, P_phi, n + 1, 0)
        vxr = -phi1s
        vyr = -phi1c
        vzr = -phi0c
        vzi = -phi0s
        if LH
            vxr += n * _resident_flat_chi_re(ch, node, P_active, n, 1)
            vyr -= n * _resident_flat_chi_im(ch, node, P_active, n, 1)
        end
        return vxr, zero(TF), vyr, zero(TF), vzr, vzi
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
    return vxr, vxi, vyr, vyi, vzr, vzi
end

# Hessian-emitting variant of `_resident_local_eval_flat` (task 032): identical
# potential/gradient math (via `_resident_gradient_coeff`), plus a second pass
# porting the `evaluate_expansions.jl` hessian recurrences — each velocity
# component's coefficient field is differentiated with the same operator pattern
# as the first pass. Returns
# `(u, vx, vy, vz, hxx, hxy, hxz, hyx, hyy, hyz, hzx, hzy, hzz)` in the legacy
# `SMatrix{3,3}` column-major linear order (`set_hessian!` order).
@inline function _resident_local_eval_flat_hessian(ph, ch, node, dx, dy, dz,
        P_phi, P_active, lhv::Val{LH}) where LH
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
        vxr, vxi, vyr, vyi, vzr, vzi =
            _resident_gradient_coeff(ph, ch, node, P_phi, P_active, n, 0, lhv)
        vx += vxr * rre - vxi * rim
        vy += vyr * rre - vyi * rim
        vz += vzr * rre - vzi * rim
        for m in 1:n
            rre, rim = _resident_regular_harmonic_coeff(dx, dy, dz, n, m)
            if n <= P_phi && !LH
                u += 2 * (rre * _resident_flat_phi_re(ph, node, P_phi, n, m) -
                          rim * _resident_flat_phi_im(ph, node, P_phi, n, m))
            end
            vxr, vxi, vyr, vyi, vzr, vzi =
                _resident_gradient_coeff(ph, ch, node, P_phi, P_active, n, m, lhv)
            vx += 2 * (vxr * rre - vxi * rim)
            vy += 2 * (vyr * rre - vyi * rim)
            vz += 2 * (vzr * rre - vzi * rim)
        end
    end

    hxx = zero(TF); hxy = zero(TF); hxz = zero(TF)
    hyx = zero(TF); hyy = zero(TF); hyz = zero(TF)
    hzx = zero(TF); hzy = zero(TF); hzz = zero(TF)
    @inbounds for n in 0:(P_active - 1)
        rre, rim = _resident_regular_harmonic_coeff(dx, dy, dz, n, 0)
        # gradient coefficients at (n+1, 0) and (n+1, 1)
        g0x_r, g0x_i, g0y_r, g0y_i, g0z_r, g0z_i =
            _resident_gradient_coeff(ph, ch, node, P_phi, P_active, n + 1, 0, lhv)
        g1x_r, g1x_i, g1y_r, g1y_i, g1z_r, g1z_i =
            _resident_gradient_coeff(ph, ch, node, P_phi, P_active, n + 1, 1, lhv)
        hxx += -g1x_i * rre
        hyx += -g1x_r * rre
        hzx += -g0x_r * rre + g0x_i * rim
        hxy += -g1y_i * rre
        hyy += -g1y_r * rre
        hzy += -g0y_r * rre + g0y_i * rim
        hxz += -g1z_i * rre
        hyz += -g1z_r * rre
        hzz += -g0z_r * rre + g0z_i * rim
        for m in 1:n
            rre, rim = _resident_regular_harmonic_coeff(dx, dy, dz, n, m)
            amx_r, amx_i, amy_r, amy_i, amz_r, amz_i =
                _resident_gradient_coeff(ph, ch, node, P_phi, P_active, n + 1, m - 1, lhv)
            bmx_r, bmx_i, bmy_r, bmy_i, bmz_r, bmz_i =
                _resident_gradient_coeff(ph, ch, node, P_phi, P_active, n + 1, m, lhv)
            cmx_r, cmx_i, cmy_r, cmy_i, cmz_r, cmz_i =
                _resident_gradient_coeff(ph, ch, node, P_phi, P_active, n + 1, m + 1, lhv)
            # x column: ∂x, ∂y, ∂z of vx
            tr = -(amx_i + cmx_i) * TF(0.5); ti = (amx_r + cmx_r) * TF(0.5)
            hxx += 2 * (tr * rre - ti * rim)
            tr = (amx_r - cmx_r) * TF(0.5); ti = (amx_i - cmx_i) * TF(0.5)
            hyx += 2 * (tr * rre - ti * rim)
            hzx += 2 * (-bmx_r * rre + bmx_i * rim)
            # y column
            tr = -(amy_i + cmy_i) * TF(0.5); ti = (amy_r + cmy_r) * TF(0.5)
            hxy += 2 * (tr * rre - ti * rim)
            tr = (amy_r - cmy_r) * TF(0.5); ti = (amy_i - cmy_i) * TF(0.5)
            hyy += 2 * (tr * rre - ti * rim)
            hzy += 2 * (-bmy_r * rre + bmy_i * rim)
            # z column
            tr = -(amz_i + cmz_i) * TF(0.5); ti = (amz_r + cmz_r) * TF(0.5)
            hxz += 2 * (tr * rre - ti * rim)
            tr = (amz_r - cmz_r) * TF(0.5); ti = (amz_i - cmz_i) * TF(0.5)
            hyz += 2 * (tr * rre - ti * rim)
            hzz += 2 * (-bmz_r * rre + bmz_i * rim)
        end
    end
    return u * c, vx * c, vy * c, vz * c,
        hxx * c, hxy * c, hxz * c, hyx * c, hyy * c, hyz * c, hzx * c, hzy * c, hzz * c
end

function _launch_host_l2b!(state::DeviceResidentRadixState{TF,B,LH}) where {TF,B,LH}
    fill!(state.output, zero(TF))
    _add_host_direct_pairs!(state)
    _add_host_twopass_deficit!(state)
    P_phi = state.invariant_cache.basis_info.orders.P_phi
    P_active = state.invariant_cache.basis_info.orders.P_active
    if size(state.output, 1) >= 13
        _host_l2b_hessian_kernel!(state.output, state.source_bodies, state.cell_ranges,
            state.cell_centers, state.grid.leaf_to_node, phi_slab(state.locals),
            chi_slab(state.locals), P_phi, P_active, Val(LH), state.counts.n_cells)
    else
        _host_l2b_kernel!(state.output, state.source_bodies, state.cell_ranges,
            state.cell_centers, state.grid.leaf_to_node, phi_slab(state.locals),
            chi_slab(state.locals), P_phi, P_active, Val(LH), state.counts.n_cells)
    end
    return state
end

function _host_l2b_hessian_kernel!(output::AbstractMatrix, source_bodies, cell_ranges,
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
            vals = _resident_local_eval_flat_hessian(
                ph, ch, node,
                source_bodies[1, i] - cx,
                source_bodies[2, i] - cy,
                source_bodies[3, i] - cz,
                P_phi, P_active, lhv,
            )
            for row in 1:13
                output[row, i] += vals[row]
            end
        end
    end
    return output
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
    hrange = hessian_range(derivatives_switch)
    isempty(hrange) || size(output, 1) >= 13 ||
        throw(ArgumentError("hessian output requested but the radix output carries " *
            "potential + gradient only; construct RadixFMMCache(...; hessian=true)"))
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
        if !isempty(hrange)
            target_buffer[hrange, ibody] .= @view output[5:13, sorted_i]
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
de-permute `state.output` (sorted body order: scalar potential + gradient, plus
the 9-component hessian when the cache was built with `hessian=true`) into
per-system target buffers and call [`buffer_to_target!`](@ref). A derivatives
switch requesting hessian rows from a 4-row output throws.
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

#------- SFS (subfilter-scale vortex stretching) pass, host mirror (task 048) -------#
#
# Mirrors the device SFS pass in translate_batched_cuda.jl: after the lifecycle
# completes U/J in `state.output`, (a) precompute per body T = op(J)Γ and zero
# the ζ-accumulators, (b) sweep the FULL direct pair list accumulating
# Ω_i = Σ_j ζ_σj(r_ij) Γ_j and Q_i = Σ_j ζ_σj(r_ij) T_j (self pair skipped —
# it cancels exactly in E), then (c) at finalize form
# E_i = op(J_i) Ω_i − Q_i (041b §1.2 reordered transposed/classic scheme) and
# scatter sorted -> global through the permutation metadata to
# `sfs_to_target!`. Γ is packed rows 5:7, raw σ row 8 (source σ convention),
# J is output rows 5:13 in FLOWVPM's J[(j-1)*3 + i] order.
#
# ζ_σ(r) = K1 exp(-ρ²/2)/σ³ with ρ = r/σ and K1 = (2π)^{-3/2} — the
# gaussianerf regularization's ζ. The saturation cutoff ρ² ≤ rc² drops
# contributions below ~1e-9 (F32) / ~1e-18 (F64) of ζ(0); it exists so the
# device kernel's exp is never fed huge arguments and host/device agree.

const _SFS_ZETA_K1 = 0.06349363593424097  # (2π)^(-3/2)
@inline _sfs_saturation_rc2(::Type{Float32}) = 42.25f0
@inline _sfs_saturation_rc2(::Type{Float64}) = 81.0

# persistent host accumulators (3 x capacity each) + the baked scheme flag
_host_sfs_context(::Type{TF}, maxn::Int, transposed::Bool,
        active_row::Int=0) where TF =
    (; tg=zeros(TF, 3, maxn), om=zeros(TF, 3, maxn), q=zeros(TF, 3, maxn),
       transposed, active_row)

@inline function _sfs_apply_op(J5, J6, J7, J8, J9, J10, J11, J12, J13,
        v1, v2, v3, transposed::Bool)
    if transposed
        return (J5 * v1 + J6 * v2 + J7 * v3,
                J8 * v1 + J9 * v2 + J10 * v3,
                J11 * v1 + J12 * v2 + J13 * v3)
    else
        return (J5 * v1 + J8 * v2 + J11 * v3,
                J6 * v1 + J9 * v2 + J12 * v3,
                J7 * v1 + J10 * v2 + J13 * v3)
    end
end

function _host_sfs_tg_and_zero!(tg, om, q, output::AbstractMatrix{TF},
        source_bodies, transposed::Bool, n::Int) where TF
    @inbounds for i in 1:n
        g1 = source_bodies[5, i]
        g2 = source_bodies[6, i]
        g3 = source_bodies[7, i]
        t1, t2, t3 = _sfs_apply_op(
            output[5, i], output[6, i], output[7, i], output[8, i],
            output[9, i], output[10, i], output[11, i], output[12, i],
            output[13, i], g1, g2, g3, transposed)
        tg[1, i] = t1; tg[2, i] = t2; tg[3, i] = t3
        om[1, i] = zero(TF); om[2, i] = zero(TF); om[3, i] = zero(TF)
        q[1, i] = zero(TF); q[2, i] = zero(TF); q[3, i] = zero(TF)
    end
    return tg
end

function _host_sfs_zeta_pairs!(om::AbstractMatrix{TF}, q, tg, source_bodies,
        cell_ranges, direct_targets, direct_sources, n_direct::Int,
        active_row::Int=0) where TF
    rc2 = _sfs_saturation_rc2(TF)
    K1 = TF(_SFS_ZETA_K1)
    half = TF(0.5)
    @inbounds for pair_i in 1:n_direct
        target_cell = direct_targets[pair_i]
        source_cell = direct_sources[pair_i]
        tfirst = cell_ranges[1, target_cell]
        tcount = cell_ranges[2, target_cell]
        sfirst = cell_ranges[1, source_cell]
        scount = cell_ranges[2, source_cell]
        for i in tfirst:(tfirst + tcount - 1)
            active_row != 0 && iszero(source_bodies[active_row, i]) && continue
            xi = source_bodies[1, i]
            yi = source_bodies[2, i]
            zi = source_bodies[3, i]
            o1 = zero(TF); o2 = zero(TF); o3 = zero(TF)
            q1 = zero(TF); q2 = zero(TF); q3 = zero(TF)
            for j in sfirst:(sfirst + scount - 1)
                i == j && continue
                active_row != 0 && iszero(source_bodies[active_row, j]) && continue
                dx = xi - source_bodies[1, j]
                dy = yi - source_bodies[2, j]
                dz = zi - source_bodies[3, j]
                r2 = dx * dx + dy * dy + dz * dz
                sigma = source_bodies[8, j]
                rho2 = r2 / (sigma * sigma)
                if rho2 <= rc2
                    z = K1 * exp(-half * rho2) / (sigma * sigma * sigma)
                    o1 += z * source_bodies[5, j]
                    o2 += z * source_bodies[6, j]
                    o3 += z * source_bodies[7, j]
                    q1 += z * tg[1, j]
                    q2 += z * tg[2, j]
                    q3 += z * tg[3, j]
                end
            end
            om[1, i] += o1; om[2, i] += o2; om[3, i] += o3
            q[1, i] += q1; q[2, i] += q2; q[3, i] += q3
        end
    end
    return om
end

"""
    _run_host_radix_sfs!(state)

Host mirror of the device SFS pass (task 048): TG precompute + ζ pair sweep
over the full direct list. Requires a state built with `sfs=true` (13-row
output). Call after `run_host_radix_lifecycle!` (U/J complete), before
`finalize_radix_sfs_output!`.
"""
function _run_host_radix_sfs!(state::DeviceResidentRadixState)
    sfs = state.sfs
    sfs === nothing && throw(ArgumentError(
        "sfs=true evaluation requires a RadixFMMCache built with sfs=true"))
    size(state.output, 1) >= 13 || throw(AssertionError(
        "the SFS pass requires the 13-row (hessian) output"))
    n = state.counts.n_bodies
    _host_sfs_tg_and_zero!(sfs.tg, sfs.om, sfs.q, state.output,
        state.source_bodies, sfs.transposed, n)
    _host_sfs_zeta_pairs!(sfs.om, sfs.q, sfs.tg, state.source_bodies,
        state.cell_ranges, state.direct_targets, state.direct_sources,
        state.counts.n_direct, sfs.active_row)
    return state
end

# E-formation into `tg` (dead after the pair sweep) in sorted body order
function _host_sfs_form_e!(tg, om, q, output::AbstractMatrix,
        transposed::Bool, n::Int)
    @inbounds for i in 1:n
        e1, e2, e3 = _sfs_apply_op(
            output[5, i], output[6, i], output[7, i], output[8, i],
            output[9, i], output[10, i], output[11, i], output[12, i],
            output[13, i], om[1, i], om[2, i], om[3, i], transposed)
        tg[1, i] = e1 - q[1, i]
        tg[2, i] = e2 - q[2, i]
        tg[3, i] = e3 - q[3, i]
    end
    return tg
end

# sorted -> global permute of the 3-row E slab into a per-system buffer
function _scatter_sfs_host!(buf::AbstractMatrix, e, perm, body_system,
        body_index, isys::Int, n::Int)
    fill!(buf, zero(eltype(buf)))
    @inbounds for sorted_i in 1:n
        global_i = perm[sorted_i]
        body_system[global_i] == isys || continue
        ibody = body_index[global_i]
        buf[1, ibody] = e[1, sorted_i]
        buf[2, ibody] = e[2, sorted_i]
        buf[3, ibody] = e[3, sorted_i]
    end
    return buf
end

"""
    finalize_radix_sfs_output!(state, target_systems; sfs_buffers=nothing)

Form E = op(J)Ω − Q from the host SFS accumulators, de-permute into per-system
`3 x n_bodies` global-order buffers, and deliver through
[`sfs_to_target!`](@ref). Pass preallocated `sfs_buffers` (one 3-row matrix
per system) to keep recurring steps allocation-free.
"""
function finalize_radix_sfs_output!(state::DeviceResidentRadixState{TF},
        target_systems; sfs_buffers=nothing) where TF
    sfs = state.sfs
    sfs === nothing && throw(ArgumentError(
        "sfs=true evaluation requires a RadixFMMCache built with sfs=true"))
    systems = to_tuple(target_systems)
    n = state.counts.n_bodies
    _host_sfs_form_e!(sfs.tg, sfs.om, sfs.q, state.output, sfs.transposed, n)
    for (isys, target_system) in enumerate(systems)
        nb = get_n_bodies(target_system)
        buf_full = sfs_buffers === nothing ? Matrix{TF}(undef, 3, nb) :
            sfs_buffers[isys]
        # capacity-sized cached buffers: deliver the live-body prefix
        buf = size(buf_full, 2) == nb ? buf_full : view(buf_full, :, 1:nb)
        _scatter_sfs_host!(buf, sfs.tg, state.host_body_perm,
            state.host_body_system_ids, state.host_body_indices, isys, n)
        sfs_to_target!(target_system, buf, 1:nb)
    end
    return target_systems
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

#------- near-set adequacy gate for regularized nearfield kernels (032 stage 2) -------#
#
# The FMM far field is singular under every nearfield strategy, so the direct
# geometry must contain every pair inside the smoothing cutoff r/σ_src ≤ ρ_t or
# the accuracy gate is silently missed (spec §5, theory §5.1-§5.2). The binding
# quantity is the smallest AABB gap the stencil leaves to M2L: adequacy is
# g_min·h_leaf > ρ_t·σ_max, evaluated per step from the live geometry (σ may
# grow, e.g. under core spreading). The box-filling n/8^ℓ form is design-time
# sizing only and must NOT be asserted here — it mis-ranks clustered fields by
# one to two levels. Per user decision (2026-08-05) an inadequate configuration
# is REJECTED with the measured ratio and admissible depth; enlarging the
# deepest-level near set is row 032a.

@inline _offset_gap2(o) =
    Float64(max(0, abs(o[1]) - 1)^2 + max(0, abs(o[2]) - 1)^2 + max(0, abs(o[3]) - 1)^2)

# min over {o : |o|² > q} of the AABB gap in cell units (√5 for q = 12, 1 for
# q = 3..5); brute force over the finite shell just outside the ball.
function _ball_stencil_min_gap(q::Int)
    reach = ceil(Int, sqrt(q)) + 2
    best = Inf
    for oz in -reach:reach, oy in -reach:reach, ox in -reach:reach
        ox * ox + oy * oy + oz * oz <= q && continue
        best = min(best, _offset_gap2((ox, oy, oz)))
    end
    return sqrt(best)
end

# Leaf-level minimum M2L gap in units of the leaf cell size. The constraint
# binds only at the leaf level (theory §5.1: ρ_tσ/h halves with each level up
# while the coarse-level gap in leaf units doubles), so coarser levels of the
# hierarchical schedule need no separate check.
function _leaf_stencil_min_gap(policy, accepted_offsets)
    policy isa HierarchicalRigidStencil && return _ball_stencil_min_gap(
        isempty(policy.level_radii2) ? policy.near_radius2 : policy.level_radii2[end])
    # flat analytic stencil: the direct set is `rejected_offsets`, so the binding
    # M2L pair is the closest accepted offset class
    best = Inf
    for o in accepted_offsets
        best = min(best, _offset_gap2(o))
    end
    return sqrt(best)
end

_leaf_stencil_min_gap(cache::RadixFMMCache) =
    _leaf_stencil_min_gap(cache.policy, cache.accepted_offsets)

#------- distance-binned nearfield pair stream (task 032a stage C) -------#
#
# Shared bucket rule for the §6.3 class-level pre-split (mechanism c): a direct
# cell pair at integer offset (ox, oy, oz) is classified from its AABB distance
# extrema against the source cell's σ extrema. The pure buckets are exactly the
# pairs whose *every* body pair falls on one side of the ρ = r/σ_src cutoff, so
# the branch-free kernels applied to them are bitwise the split kernel's own
# branch outcome; the mixed bucket keeps the predicated (or queued) kernel.
# Pure Julia arithmetic — shared verbatim by the host reference/tests and the
# CUDA classification kernel.
#
#   1 = pure singular:    d_min > rho_cut · σ_max(source cell)   (or σ_max ≤ 0)
#   2 = pure regularized: d_max ≤ rho_cut · σ_min(source cell) and σ_min > 0
#   3 = mixed
#
# d_min/d_max are the exact axis-aligned cell-AABB distance extrema at lattice
# offset o with cube cell size h: per axis max(|o_q|−1, 0)·h and (|o_q|+1)·h.
@inline function _nearfield_pair_bucket(ox::Integer, oy::Integer, oz::Integer,
        h_leaf::T, rho_cut::T, sigma_max_s::T, sigma_min_s::T) where T<:AbstractFloat
    gx = T(max(abs(ox) - 1, 0)); mx = T(abs(ox) + 1)
    gy = T(max(abs(oy) - 1, 0)); my = T(abs(oy) + 1)
    gz = T(max(abs(oz) - 1, 0)); mz = T(abs(oz) + 1)
    h2 = h_leaf * h_leaf
    dmin2 = (gx * gx + gy * gy + gz * gz) * h2
    dmax2 = (mx * mx + my * my + mz * mz) * h2
    if !(sigma_max_s > zero(T))
        return Int32(1)
    end
    cmax = rho_cut * sigma_max_s
    dmin2 > cmax * cmax && return Int32(1)
    if sigma_min_s > zero(T)
        cmin = rho_cut * sigma_min_s
        dmax2 <= cmin * cmin && return Int32(2)
    end
    return Int32(3)
end

# Task 037e: exact target-point/source-cell-AABB reachability predicate for the
# mixed-bucket pair-level fast path (`CUDA_NEARFIELD_PAIR_AABB`). Returns true
# when the point (xi, yi, zi) can reach the source-cell AABB
# [slo, slo + h_leaf]^3 within rho_cut·σ_max(source cell), i.e. when at least
# one source body in that cell COULD satisfy the regularized branch predicate
# ρ = r/σ ≤ rho_cut. The point-to-AABB gap arithmetic mirrors the 037a-validated
# `_cuda_twopass_deficit_kernel!` qx/near2 form exactly. Pure Julia scalar
# arithmetic — shared verbatim by the CUDA kernels, the host unit tests, and
# the fm037e scoping script.
@inline function _nearfield_point_aabb_reach(xi::T, yi::T, zi::T, slo_x::T,
        slo_y::T, slo_z::T, h_leaf::T, rho_cut::T,
        sigma_max_s::T) where T<:AbstractFloat
    sigma_max_s > zero(T) || return false
    shi_x = slo_x + h_leaf
    shi_y = slo_y + h_leaf
    shi_z = slo_z + h_leaf
    qx = xi < slo_x ? slo_x - xi : (xi > shi_x ? xi - shi_x : zero(T))
    qy = yi < slo_y ? slo_y - yi : (yi > shi_y ? yi - shi_y : zero(T))
    qz = zi < slo_z ? slo_z - zi : (zi > shi_z ? zi - shi_z : zero(T))
    near2 = qx * qx + qy * qy + qz * qz
    return near2 <= (rho_cut * sigma_max_s)^2
end

# Construction-sized compacted offset ball for the device TwoPassVortex pass-2
# sweep (Stage-B open risk): every integer offset with lattice gap
# gap(o) = √Σ max(|o_q|−1, 0)² ≤ reach_cap_cells, sorted gap-ascending so the
# live ball {o : gap(o)·h_leaf ≤ rho_t·σ_max} is always a prefix-selectable
# subset (the deficit kernel tests gap²·h² ≤ (rho_t·σ_max)² per entry from the
# device σ_max scalar — no per-step list rebuild). The capacity reach is
# gate-derived: pass-1 adequacy asserts rho_c·σ_max < g_min·h_leaf per step,
# so rho_t·σ_max < (rho_t/rho_c)·g_min·h_leaf ≡ reach_cap_cells·h_leaf always.
function _twopass_offset_ball(reach_cap_cells::Float64)
    reach_cap_cells >= 0 || throw(ArgumentError("reach_cap_cells must be nonnegative"))
    R = floor(Int, reach_cap_cells) + 1
    offs = NTuple{3,Int}[]
    gap2s = Int[]
    cap2 = reach_cap_cells * reach_cap_cells
    for oz in -R:R, oy in -R:R, ox in -R:R
        g2 = max(abs(ox) - 1, 0)^2 + max(abs(oy) - 1, 0)^2 + max(abs(oz) - 1, 0)^2
        g2 <= cap2 || continue
        push!(offs, (ox, oy, oz))
        push!(gap2s, g2)
    end
    p = sortperm(gap2s)
    K = length(p)
    offsets = Matrix{Int32}(undef, 3, K)
    gap2 = Vector{Int32}(undef, K)
    for (k, idx) in enumerate(p)
        offsets[1, k] = Int32(offs[idx][1])
        offsets[2, k] = Int32(offs[idx][2])
        offsets[3, k] = Int32(offs[idx][3])
        gap2[k] = Int32(gap2s[idx])
    end
    return offsets, gap2
end

_direct_kernel_geometry_gate!(cache::RadixFMMCache, ::AbstractDirectKernel,
    source_bodies, n::Int) = nothing

# The reach the primary direct near set must cover, in units of sigma. The
# single-pass regularized kernels evaluate every cutoff pair directly, so they
# need the full rho_t; the two-pass hybrid's pass 1 only evaluates ρ ≤ rho_c
# regularized (its self-sizing pass-2 sweep covers the (rho_c, rho_t] shell on
# its own, see _host_twopass_deficit_kernel!), so its gate binds at rho_c.
@inline _gate_reach_rho(kernel::AbstractRegularizedVortex) = (kernel.rho_t, "rho_t")
@inline _gate_reach_rho(kernel::TwoPassVortex) = (kernel.rho_c, "rho_c (pass-1 hybrid switch)")

function _direct_kernel_geometry_gate!(cache::RadixFMMCache,
        kernel::AbstractRegularizedVortex, source_bodies, n::Int)
    n > 0 || return nothing
    # Zero-M2L degenerate cache (task 052c): with no accepted offset class the
    # direct list covers every pair at every offset, so no pair can fall to
    # the singular far field — the adequacy gate is vacuous.
    isempty(cache.accepted_offsets) && return nothing
    # works for Matrix and CuMatrix alike (device reduction + scalar download)
    sigma_max = Float64(maximum(view(source_bodies, kernel.sigma_row, 1:n)))
    sigma_max > 0 || return nothing
    g_min = _leaf_stencil_min_gap(cache)
    h_leaf = 2 * Float64(cache.h0) / (1 << cache.ell)
    rho_reach, rho_name = _gate_reach_rho(kernel)
    cutoff = rho_reach * sigma_max
    if g_min * h_leaf > cutoff
        _twopass_device_reach_check(cache, kernel, sigma_max, h_leaf)
        return nothing
    end
    x = g_min * 2 * Float64(cache.h0) / cutoff   # admissible 2^ℓ bound
    ell_max = floor(Int, log2(x))
    2.0^ell_max < x || (ell_max -= 1)
    depth_msg = ell_max >= 0 ? "the admissible depth at this geometry is ell <= $ell_max" :
        "no tree depth is admissible at this geometry (the box itself is inside the cutoff)"
    throw(ArgumentError(
        "regularized nearfield near-set adequacy failed: the direct stencil leaves " *
        "an M2L gap of g_min*h_leaf = $(round(g_min * h_leaf, sigdigits=4)) but the " *
        "smoothing cutoff needs $rho_name*sigma_max = $(round(cutoff, sigdigits=4)) " *
        "(ratio $(round(g_min * h_leaf / cutoff, sigdigits=4)), g_min = " *
        "$(round(g_min, sigdigits=4)), sigma_max = $(round(sigma_max, sigdigits=4)), " *
        "ell = $(cache.ell)); $depth_msg. Pairs inside the cutoff would be handled " *
        "by the singular far field and silently lose the regularization. Reduce ell, " *
        "shrink sigma, or use a larger near set (row 032a)."))
end

# Defensive pass-2 capacity assertion for device TwoPassVortex caches (task 032a
# stage C): the construction-sized offset ball covers (rho_t/rho_c)·g_min cells,
# and the pass-1 gate just passed rho_c·σ_max < g_min·h_leaf, so this can only
# fire on an internal-consistency bug — never on user geometry.
_twopass_device_reach_check(cache::RadixFMMCache, ::AbstractDirectKernel,
    sigma_max, h_leaf) = nothing

function _twopass_device_reach_check(cache::RadixFMMCache, kernel::TwoPassVortex,
        sigma_max::Float64, h_leaf::Float64)
    cache.device || return nothing
    nfctx = _cache_nearfield_bin_ctx(cache)
    nfctx === nothing && return nothing
    reach_cells = kernel.rho_t * sigma_max / h_leaf
    reach_cells <= nfctx.twopass_reach_cap_cells * (1 + 1e-12) ||
        throw(AssertionError(
            "TwoPassVortex device pass-2 offset ball capacity exceeded: live " *
            "reach $(reach_cells) cells > capacity $(nfctx.twopass_reach_cap_cells) " *
            "cells despite a passing pass-1 gate (internal inconsistency)"))
    return nothing
end

# 032a stage C: the device pass-2 deficit sweep (and the §6.3 binned pair
# stream) live on the hierarchical device context; a flat-policy device cache
# would run pass 1 but silently skip the pass-2 deficit sweep, so it is refused
# at construction.
function _assert_device_kernel_policy(device::Bool, dk, hierarchical::Bool)
    device && dk isa TwoPassVortex && !hierarchical && throw(ArgumentError(
        "TwoPassVortex on a device cache requires the hierarchical stencil " *
        "policy (the flat-policy device path has no pass-2 deficit sweep); " *
        "use the default HierarchicalRigidStencil, build the cache with " *
        "device=false, or select another nearfield kernel"))
    return nothing
end

# The stage-C nearfield bin context lives on the hierarchical device context;
# flat-policy or host caches have none.
function _cache_nearfield_bin_ctx(cache::RadixFMMCache)
    cache.device || return nothing
    ctx = cache.device_ctx
    ctx === nothing && return nothing
    hctx = ctx.hierarchical_ctx
    hctx === nothing && return nothing
    return hctx.nearfield
end

# Legacy cubic arity; the per-axis method below is the contractual check
# (task 037: cubic caches pass box_extent = (2h0, 2h0, 2h0), same values).
_assert_radix_positions_in_box(systems::Tuple, x_min::SVector{3,TF}, h0::TF) where TF =
    _assert_radix_positions_in_box(systems, x_min, SVector{3,TF}(2 * h0, 2 * h0, 2 * h0))

function _assert_radix_positions_in_box(systems::Tuple, x_min::SVector{3,TF},
        box_extent::SVector{3,TF}) where TF
    x_max = x_min .+ box_extent
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

# Resolve the rectangular geometry contract (task 037) from the bounds box size.
# Scalar sizes reproduce the legacy cubic contract bit-for-bit. Vector sizes
# embed the box in a virtual cube of half-width h0 = maximum(box_size)/2 whose
# leaf width Δ = 2h0/2^ell tiles every axis: axis a spans 2^ell_axes[a] leaf
# cells with its extent snapped up to Δ * 2^ell_axes[a] (never below the
# requested extent).
function _resolve_radix_ell_axes(box_size::Real, ell::Int, ::Type{TF}) where TF
    h0 = TF(box_size) / 2
    h0 > zero(TF) || throw(ArgumentError("bounds box_size must be positive"))
    return SVector(ell, ell, ell), h0, SVector{3,TF}(2 * h0, 2 * h0, 2 * h0)
end

function _resolve_radix_ell_axes(box_size, ell::Int, ::Type{TF}) where TF
    L = SVector{3,TF}(box_size)
    (L[1] > zero(TF) && L[2] > zero(TF) && L[3] > zero(TF)) || throw(ArgumentError(
        "bounds box_size must be positive on every axis; got $(Tuple(L))"))
    h0 = max(L[1], L[2], L[3]) / 2
    delta = (2 * h0) / (1 << ell)
    function resolve_axis(a)
        la = clamp(ceil(Int, log2(Float64(L[a]) / Float64(delta))), 0, ell)
        # fp guard: log2/ceil may land one level short of covering the extent
        while la < ell && delta * (1 << la) < L[a]
            la += 1
        end
        return la
    end
    ell_axes = SVector(resolve_axis(1), resolve_axis(2), resolve_axis(3))
    box_extent = SVector{3,TF}(delta * (1 << ell_axes[1]),
        delta * (1 << ell_axes[2]), delta * (1 << ell_axes[3]))
    return ell_axes, h0, box_extent
end

# Measured window widths (task 027). The host default of 4 comes from the 026 host
# campaign; on the GPU the per-window flag/scan/compact carries a fixed ~50 us
# device-to-host round trip, so route generation scales as
# `(ell - 1) * ceil(noffsets / K)` and K = 4 spends 72-164 ms per step on latency
# alone. H200 job 12992039 measured route generation falling 109-193x from K = 4 to
# a whole-level window.
#
# Task 028 Phase A then measured the residual at n = 1e6: K = 256 still spent
# 24.19 ms per step in route generation against 2.13 ms for a whole-level window,
# so the device default is now larger than any supported shell's offset count and
# every level is generated in one window. The cost is route-buffer memory, which
# grows as `min(K, noffsets) * max_level_nodes` (2.0 GB persistent at n = 1e6,
# ell = 5 on the default policy); pass a smaller `window_classes` on
# memory-constrained devices.
const RADIX_HOST_WINDOW_CLASSES = 4
const RADIX_DEVICE_WINDOW_CLASSES = 4096

# Measured defaults for `RadixFMMCache(...; options=nothing)` (tasks 024 and 028).
# Both choices are made from the expansion order, the Lamb-Helmholtz channel, the
# platform, and the dense operator footprint — the selectors 024 found sufficient —
# and both are overridden by passing an explicit `options`.
#
# Precision. Task 024 measured Float32 max gradient error at 5.33e-4 (CPU) / 5.34e-4
# (H200) essentially independently of `P`, against 5.25e-6 for Float64: Float32 has an
# accuracy floor, not an accuracy cost proportional to the order. At literature P = 4
# that floor sits under the stencil's own truncation error (task 028 measured 3.186e-4
# Float32 vs 3.185e-4 Float64 at n = 1e6, +0.03%), so Float32 is free accuracy-wise and
# worth 1.14x; above P = 4 the user is paying for accuracy Float32 would discard.
_default_radix_precision(expansion_order::Int) =
    expansion_order <= 3 ? Float32 : Float64

# Strategy. Task 024's recurring-step rules: dense wins every measured P = 4 case on
# both platforms and every P = 8 case with LH off; with LH on at P = 8 the platforms
# split (H200 precomputed-y, CPU dense); precomputed-y wins P = 12 everywhere, where
# dense is either unsupported (Float32) or over its memory gate. Task 028 confirmed
# dense over precomputed-y on the hierarchical path at n = 1e6 (106.3 vs 172.3 ms).
# Concat/factored won no steady-state case on either platform, so the historical
# `ConcatenatedFixedZM2L` default was not the measured choice at any order.
#
# Dense trades construction for steady state (~20 s build and ~300-370 break-even
# steps at the task-028 target), which suits the repeated-step cache this is, but not
# one-shot evaluation: pass `PrecomputedFactoredYM2L()` explicitly for that.
function _default_radix_m2l_strategy(::Type{TF}, expansion_order::Int, LH::Bool,
        device::Bool, nclasses::Int, ndof::Int) where TF
    dense = DenseTranslationM2L()
    # 024 found the operator payload is dense's binding constraint; keep a margin
    # under its own gate so the auto choice never construction-errors on storage.
    dense_bytes = nclasses * ndof * ndof * sizeof(TF)
    dense_bytes <= (dense.max_persistent_bytes * 3) ÷ 4 || return PrecomputedFactoredYM2L()
    expansion_order <= 3 && return dense                  # literature P <= 4
    expansion_order <= 7 || return PrecomputedFactoredYM2L()  # literature P >= 12
    LH || return dense                                    # P = 8, LH off
    return device ? PrecomputedFactoredYM2L() : dense     # P = 8, LH on: platform split
end

_default_radix_options(::Type{TF}, expansion_order::Int, LH::Bool, device::Bool,
        nclasses::Int, ndof::Int) where TF =
    _radix_options_for(TF, _default_radix_m2l_strategy(TF, expansion_order, LH,
        device, nclasses, ndof))

# Each resident strategy is bound to the rotation operator its plan is built from.
_radix_options_for(::Type{TF}, m2l_strategy::PrecomputedFactoredYM2L) where TF =
    CUDARadixLifecycleOptions(; precision=TF, operator=FactoredRotationM2L(),
        m2l_strategy)
_radix_options_for(::Type{TF}, m2l_strategy) where TF =
    CUDARadixLifecycleOptions(; precision=TF, operator=MaterializedYRotationM2L(),
        m2l_strategy)

# Default separation policy (task 027). `HierarchicalRigidStencil` replaced the flat
# `ConstantPAnalyticStencil` as the production default: the flat classifier's
# accepted-offset set grows with `ell` (cell width shrinks at fixed epsilon), so its
# route count scales as `offsets(ell) x cells`, while the rigid stencil's offset set
# is level-invariant. An explicit `stencil_epsilon` still selects the flat policy, so
# existing callers keep their exact behavior.
function _default_radix_policy(policy, P::Int, ::Type{TF}, LH::Bool, h0, ell::Int,
        device::Bool, stencil_epsilon, near_radius2, window_classes,
        level_radii2=nothing) where TF
    if policy !== nothing
        (stencil_epsilon === nothing && near_radius2 === nothing &&
            window_classes === nothing && level_radii2 === nothing) ||
            throw(ArgumentError("an explicit `policy` carries its own stencil " *
                "parameters; do not combine it with `stencil_epsilon`, " *
                "`near_radius2`, `level_radii2`, or `window_classes`"))
        return policy
    end
    K = window_classes === nothing ?
        (device ? RADIX_DEVICE_WINDOW_CLASSES : RADIX_HOST_WINDOW_CLASSES) :
        Int(window_classes)
    if stencil_epsilon !== nothing
        # explicit tolerance: the caller is asking for the flat analytic classifier
        (near_radius2 === nothing && level_radii2 === nothing) ||
            throw(ArgumentError("`stencil_epsilon` selects the flat " *
                "ConstantPAnalyticStencil, which has no `near_radius2` or " *
                "`level_radii2`; pass a HierarchicalRigidStencil `policy` for " *
                "an explicit tolerance with a rigid near set"))
        return ConstantPAnalyticStencil(
            ConstantPStencilConfig(P, TF(stencil_epsilon); lamb_helmholtz=LH))
    end
    q = near_radius2 === nothing ? RADIX_DEFAULT_NEAR_RADIUS2 : Int(near_radius2)
    if ell < 2
        # the first M2L level is 2; there is no hierarchy to walk below that
        return ConstantPAnalyticStencil(
            ConstantPStencilConfig(P, TF(1e-4); lamb_helmholtz=LH))
    end
    # Only the untouched default carries the task-028 Stage 7 level schedule: an
    # explicit `near_radius2` is honored as the uniform geometry the caller asked
    # for. The schedule covers M2L levels 2:ell and is non-increasing with depth.
    qs = if level_radii2 !== nothing
        Tuple(Int(x) for x in level_radii2)
    elseif near_radius2 === nothing && ell >= 3
        (RADIX_DEFAULT_COARSE_NEAR_RADIUS2,
            ntuple(_ -> RADIX_DEFAULT_NEAR_RADIUS2, ell - 2)...)
    else
        ()
    end
    eps = rigid_stencil_epsilon(P, h0, ell, q; lamb_helmholtz=LH, TF)
    return HierarchicalRigidStencil(
        ConstantPStencilConfig(P, TF(eps); lamb_helmholtz=LH);
        near_radius2=q, level_radii2=qs, window_classes=K)
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
  a cube from the current positions inflated by `bounds_margin`. `box_size` may be a
  scalar (cubic, legacy) or a 3-vector/`NTuple{3}` of per-axis extents (task 037):
  the rectangular box is embedded in a virtual cube of half-width
  `h0 = maximum(box_size)/2`, per-axis extents snap up to whole leaf cells
  (readable as `cache.ell_axes` / `cache.box_extent`), and the per-axis in-box
  contract is enforced each step, on host and device caches alike.
- `bounds_margin::Real=0.05`: relative margin applied to derived bounds
- `lamb_helmholtz=nothing`: override the `has_vector_potential` inference
- `hessian::Bool=false`: allocate the 13-row output (potential + gradient +
  9-component hessian) and enable `fmm!(...; hessian=true)`. Off by default so
  the scalar path's output bandwidth is unchanged (task 032).
- `sfs::Bool=false`: allocate the subfilter-scale vortex-stretching pass
  (task 048) and enable `fmm!(...; sfs=true)` delivery through
  [`sfs_to_target!`](@ref). Requires `hessian=true` and the raw smoothing
  radius σ in packed row 8 (vortex couplings). `sfs_transposed::Bool=true`
  bakes the FLOWVPM transposed-scheme convention.
- `device::Bool=false`: run the lifecycle device-resident (CUDA; requires
  `load_cuda_radix_lifecycle!()`)
- `options::CUDARadixLifecycleOptions`: operator strategies/precision. Omitted, both
  are selected from the measured 024/028 rules (see below); passed explicitly, it is
  used verbatim. The resolved choice is readable as `cache.state.options`.
- `near_radius2`: rigid leaf near set `{o : |o|^2 <= near_radius2}` of the default
  hierarchical policy (default `$(RADIX_DEFAULT_NEAR_RADIUS2)`; `12` is the `024b`
  `theta=0.5` stencil and `3` the classic FMM one). Passing it explicitly also
  selects the *uniform* geometry, i.e. it drops the default level schedule below.
- `level_radii2`: per-M2L-level near radii, coarse to fine; must be
  non-increasing and end at `near_radius2` (task 028 Stage 7). Anchored to
  levels `2:ell` (legacy length `ell - 1`) or — task 037, rectangular caches
  with trimmed coarse levels — to the active M2L levels
  `first_m2l_level:ell`; the legacy anchoring is sliced to the active range,
  which is the identity on cubic caches
- `window_classes`: route-window width; defaults to the measured
  `$(RADIX_DEVICE_WINDOW_CLASSES)` on device and `$(RADIX_HOST_WINDOW_CLASSES)` on host
- `stencil_epsilon::Real`: **selects the deprecated flat `ConstantPAnalyticStencil`**
  at this tolerance (task 027); omit it to get the hierarchical default
- `policy`: explicit `ConstantPAnalyticStencil` or `HierarchicalRigidStencil`
  (both run host- or device-resident). A policy carries its own stencil
  parameters, so combining it with `stencil_epsilon`, `near_radius2`,
  `level_radii2`, or `window_classes` throws; likewise `stencil_epsilon` (flat)
  rejects `near_radius2`.

Since task 027 the default policy is [`HierarchicalRigidStencil`](@ref); since task
028 Stage 7 its default geometry is `near_radius2=$(RADIX_DEFAULT_NEAR_RADIUS2)` with
the level schedule `($(RADIX_DEFAULT_COARSE_NEAR_RADIUS2), $(RADIX_DEFAULT_NEAR_RADIUS2), ...)`,
the fastest configuration inside task 028's `P = 4` accuracy gate. Its tolerance is
derived by [`rigid_stencil_epsilon`](@ref) so the analytic accuracy gate is satisfied
by construction; pass `near_radius2=12` for the previous, more accurate and slower
default. The flat
[`ConstantPAnalyticStencil`](@ref) is deprecated as a default but fully supported;
it is still used automatically when `ell < 2`, where there is no hierarchy to walk.

When no `options` are passed, precision and M2L strategy follow the measured rules
of tasks 024 and 028:

| selector | precision | M2L strategy |
|---|---|---|
| `expansion_order <= 3` (literature `P <= 4`) | `Float32` | dense |
| `expansion_order <= 7`, no Lamb-Helmholtz | `Float64` | dense |
| `expansion_order <= 7`, Lamb-Helmholtz | `Float64` | precomputed-y on device, dense on host |
| `expansion_order >= 8` (literature `P >= 12`) | `Float64` | precomputed-y |
| dense operator payload over its gate | unchanged | precomputed-y |

`Float32` is selected only where task 024 measured its accuracy floor (~5.3e-4 max
gradient error, essentially independent of `P`) to sit below the stencil's own
truncation error; above `P = 4` it would discard accuracy the higher order was paid
for. Dense trades a large construction cost for the best steady state (~300-370
break-even steps at the task-028 target), which suits this repeated-step cache; pass
`options=CUDARadixLifecycleOptions(; m2l_strategy=PrecomputedFactoredYM2L(),
operator=FactoredRotationM2L())` for one-shot evaluation, or any explicit `options`
to bypass the rules entirely.

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
        hessian::Bool=false,
        sfs::Bool=false,
        sfs_transposed::Bool=true,
        sfs_active_row::Integer=0,
        device::Bool=false,
        options::Union{Nothing,CUDARadixLifecycleOptions}=nothing,
        stencil_epsilon::Union{Nothing,Real}=nothing,
        near_radius2::Union{Nothing,Integer}=nothing,
        level_radii2=nothing,
        window_classes::Union{Nothing,Integer}=nothing,
        policy::Union{Nothing,ConstantPAnalyticStencil,HierarchicalRigidStencil}=nothing,
        adaptive::Union{Nothing,AdaptiveTreePolicy}=nothing)
    targets = to_tuple(target_systems)
    sources = to_tuple(source_systems)
    _assert_radix_targets_are_sources(targets, sources)
    # task 048: the SFS pass reads the 9-component J from the 13-row output
    # and the raw smoothing radius sigma from packed row 8
    sfs && !hessian && throw(ArgumentError(
        "RadixFMMCache(sfs=true) requires hessian=true (the SFS pass reads " *
        "the velocity Jacobian from the 13-row output)"))
    if sfs
        for system in sources
            data_per_body(system) >= 8 || throw(ArgumentError(
                "RadixFMMCache(sfs=true) requires the raw smoothing radius " *
                "sigma in packed row 8; data_per_body must be >= 8 " *
                "(got $(data_per_body(system)) for $(typeof(system)))"))
        end
        sfs_active_row >= 0 || throw(ArgumentError(
            "sfs_active_row must be zero (all bodies active) or a positive packed row"))
        sfs_active_row == 0 ||
            all(data_per_body(system) >= sfs_active_row for system in sources) ||
            throw(ArgumentError("sfs_active_row=$sfs_active_row exceeds data_per_body for an SFS source system"))
    end
    LH = lamb_helmholtz === nothing ? has_vector_potential(sources) : Bool(lamb_helmholtz)
    # B2M element resolution (task 032): one shared body type per cache, checked
    # here so a Point{Vortex} system with the χ channel off fails at construction
    # rather than inside a kernel (the LH=false chi buffer is 0×0).
    BT = body_type(first(sources))
    for system in sources
        body_type(system) === BT || throw(ArgumentError(
            "all source systems sharing a RadixFMMCache must report the same " *
            "body_type; got $(body_type(system)) and $BT"))
        strength_dims(system) == strength_dims(first(sources)) || throw(ArgumentError(
            "all source systems sharing a RadixFMMCache must report the same " *
            "strength_dims (the packed strength rows 5:4+strength_dims are shared)"))
    end
    if BT <: Point{Vortex} && !LH
        throw(ArgumentError(
            "Point{Vortex} sources require the Lamb-Helmholtz channel; construct " *
            "the cache with lamb_helmholtz=true (or leave it to be inferred from " *
            "has_vector_potential)"))
    end
    # Nearfield kernel resolution (task 032 stage 2): one shared functor per
    # cache, resolved from the trait like body_type above; validated after the
    # options carry the final choice (below).
    dk_trait = direct_kernel(first(sources))
    for system in sources
        direct_kernel(system) == dk_trait || throw(ArgumentError(
            "all source systems sharing a RadixFMMCache must report the same " *
            "direct_kernel; got $(direct_kernel(system)) and $dk_trait"))
    end
    # Measured defaults (024/028). Precision depends only on the expansion order and
    # is needed for the bounds and stencil tolerance below; the strategy also depends
    # on the class count, so it is resolved once the policy is built.
    auto_options = options === nothing
    if auto_options
        options = CUDARadixLifecycleOptions(;
            precision=_default_radix_precision(Int(expansion_order)),
            m2l_strategy=ConcatenatedFixedZM2L())
    end
    TF = options.precision
    if device
        cuda_radix_available() || radix_device_backend_available() ||
            throw(ArgumentError("RadixFMMCache(device=true) requires a functional CUDA " *
                "radix lifecycle, or a registered non-CUDA device backend; call " *
                "load_cuda_radix_lifecycle!() first, or load a backend extension " *
                "($(cuda_radix_status()))"))
    end
    for system in sources
        data_per_body(system) >= 4 + strength_dims(system) ||
            throw(ArgumentError("the radix path packs bodies as [x, y, z, radius, " *
                "strength..., extras...]; data_per_body(system) must be >= " *
                "4 + strength_dims(system)"))
    end
    dpb = maximum(data_per_body(system) for system in sources)

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
        ell_axes = SVector(Int(ell), Int(ell), Int(ell))
        box_extent = SVector{3,TF}(2 * h0, 2 * h0, 2 * h0)
    else
        x_min = SVector{3,TF}(bounds[1])
        ell_axes, h0, box_extent = _resolve_radix_ell_axes(bounds[2], Int(ell), TF)
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
        stencil_epsilon, near_radius2, window_classes, level_radii2)
    hierarchical = stencil_policy isa HierarchicalRigidStencil
    # Active-level trimming (task 037 stage 3): hierarchical caches retain node
    # levels root_level:ell and run M2L on levels first_m2l_level:ell (the
    # flat-top root level plus the task-025 transition levels). Cubic caches
    # degenerate to root_level = 1 with an empty flat-top (first_m2l_level = 2,
    # the legacy schedule); flat-policy caches stay untrimmed (root_level = 0).
    if hierarchical
        hierarchical_tables, hierarchical_level_class_of, hierarchical_level_radii2,
            root_level, first_m2l_level =
            _hierarchical_scheduled_tables(stencil_policy, Int(ell), ell_axes)
        _verify_hierarchical_classifier!(h0, Int(ell), stencil_policy,
            hierarchical_tables, ell_axes, root_level, first_m2l_level,
            hierarchical_level_radii2)
        class_level, class_offset, effective_offsets =
            _hierarchical_class_metadata(hierarchical_tables, Int(ell),
                first_m2l_level)
        accepted, rejected = effective_offsets, hierarchical_tables.near_offsets
    else
        hierarchical_tables = nothing
        hierarchical_level_class_of = Array{Int32}(undef, 0, 0, 0)
        hierarchical_level_radii2 = Int[]
        root_level, first_m2l_level = 0, 2
        class_level, class_offset, effective_offsets =
            Int32[], Matrix{Int32}(undef, 3, 0), SVector{3,Int}[]
        accepted, rejected =
            classify_radix_stencil_offsets(h0, Int(ell), stencil_policy.config)
    end

    max_cells = _radix_level_node_capacity(Int(ell), ell_axes, Int(ell), maxn)
    max_nodes = sum(_radix_level_node_capacity(L, ell_axes, Int(ell), max_cells)
        for L in root_level:Int(ell))
    # init=0 covers the zero-M2L degenerate hierarchy (first_m2l_level == ell+1,
    # empty range — task 052c): no M2L level, so no per-level node bound needed.
    max_level_nodes = Int(ell) >= 2 ? maximum(
        (_radix_level_node_capacity(L, ell_axes, Int(ell), max_cells)
         for L in (hierarchical ? first_m2l_level : 2):Int(ell)); init=0) : 0
    route_capacity = hierarchical ?
        min(min(stencil_policy.window_classes,
                length(hierarchical_tables.push_offsets)) * max_level_nodes,
            max_level_nodes * max_level_nodes) :
        min(length(accepted), max_cells) * max_cells
    direct_capacity = max_cells * min(length(rejected), max_cells)

    basis_info = OperatorBasisInfo(CompressedComplexBasis(), P, Val(LH))

    if auto_options
        options = _default_radix_options(TF, P, LH, device, length(accepted),
            _dense_m2m_dof(basis_info, Val(LH)))
    end
    options = _options_with_body_type(options, BT)
    if dk_trait != _default_direct_kernel(BT)
        # explicit trait choice; a conflicting explicit options choice is an error
        (options.direct_kernel == _default_direct_kernel(BT) ||
            options.direct_kernel == dk_trait) || throw(ArgumentError(
            "options.direct_kernel=$(options.direct_kernel) conflicts with the " *
            "direct_kernel(system) trait $dk_trait"))
        options = _options_with_direct_kernel(options, dk_trait)
    end
    dk = options.direct_kernel
    isbits(dk) || throw(ArgumentError(
        "direct_kernel must be an isbits functor (GPU-compilable, no references); " *
        "got $(typeof(dk))"))
    if dk isa AbstractRegularizedVortex
        kname = nameof(typeof(dk))
        BT <: Point{Vortex} || throw(ArgumentError(
            "$kname requires body_type Point{Vortex}; got $BT"))
        for system in sources
            dk.sigma_row <= data_per_body(system) || throw(ArgumentError(
                "$kname sigma_row=$(dk.sigma_row) exceeds " *
                "data_per_body=$(data_per_body(system)) for $(typeof(system)); " *
                "every source system must carry the smoothing radius σ in packed " *
                "row sigma_row"))
        end
    end
    _assert_device_kernel_policy(device, dk, hierarchical)

    # Opt-in adaptive octree (tasks 039/040): host-only, cubic-domain-only.
    # Refreshed alongside the uniform structures by update_radix_state!; with
    # the policy armed the host fmm! branch runs the ADAPTIVE lifecycle (task
    # 040) instead of the uniform one. Production defaults are unchanged.
    if adaptive !== nothing
        # task 041: the device mirror supports the same lifecycle surface as
        # the host path. The S2L stage supports Point{Source}/Point{Vortex}
        # only — guard at construction (040 approval note) instead of the
        # former runtime throw alone.
        options.body_type <: Union{Point{Source},Point{Vortex}} ||
            throw(ArgumentError(
                "the adaptive octree S2L stage supports Point{Source} and " *
                "Point{Vortex} body types; got $(options.body_type)"))
        device && adaptive.split_veto && throw(ArgumentError(
            "the adaptive device path does not implement the §5.4 split veto " *
            "(default OFF, pending user ratification); construct with " *
            "split_veto=false or run host-resident"))
        ell_axes == SVector(Int(ell), Int(ell), Int(ell)) || throw(ArgumentError(
            "the adaptive octree requires a cubic Morton domain in task 039; " *
            "rectangular ell_axes support is a recorded deferral"))
        if adaptive.sigma_row > 0
            for system in sources
                adaptive.sigma_row <= data_per_body(system) || throw(ArgumentError(
                    "AdaptiveTreePolicy sigma_row=$(adaptive.sigma_row) exceeds " *
                    "data_per_body=$(data_per_body(system)) for $(typeof(system))"))
            end
        end
        # Task 040 lifecycle guards.
        if dk isa Union{TwoPassVortex,PartitionedVortex}
            throw(ArgumentError(
                "the adaptive octree lifecycle does not support " *
                "$(nameof(typeof(dk))) (task 040 deferral: the two-pass/" *
                "partitioned nearfield assumes the uniform leaf lattice); use " *
                "RegularizedVortex or a singular kernel"))
        end
        if dk isa AbstractRegularizedVortex
            rho_reach, rho_name = _gate_reach_rho(dk)
            (adaptive.rho_t >= rho_reach && adaptive.sigma_row == dk.sigma_row) ||
                throw(ArgumentError(
                "a regularized nearfield on the adaptive octree requires the " *
                "per-cell sigma gate (theory §5): AdaptiveTreePolicy(rho_t >= " *
                "$rho_name = $rho_reach, sigma_row = $(dk.sigma_row)); got " *
                "rho_t=$(adaptive.rho_t), sigma_row=$(adaptive.sigma_row). The " *
                "global geometry gate is replaced by sticky per-cell demotion " *
                "on the adaptive path, so the gate must be armed."))
        end
        if hessian && LH
            throw(ArgumentError(
                "hessian output on the adaptive lifecycle with " *
                "lamb_helmholtz=true is a task-040 deferral (the W-list M2T " *
                "Lamb-Helmholtz hessian is not implemented); use hessian=false " *
                "or the uniform path"))
        end
    end

    if device
        cache = _radix_cache_device_build(sources, P, Int(ell), x_min, h0, maxn,
            options, stencil_policy, accepted, rejected, max_cells, max_nodes,
            route_capacity, direct_capacity, basis_info, Val(LH);
            hierarchical_tables, class_level, class_offset,
            hierarchical_level_class_of, hierarchical_level_radii2,
            max_level_nodes, hessian, sfs, sfs_transposed,
            sfs_active_row=Int(sfs_active_row), ell_axes, box_extent,
            root_level, first_m2l_level,
            adaptive_policy=adaptive, dpb_adaptive=dpb)
        cache.built = true
        return cache
    end

    grid = _allocate_host_radix_grid(TF, x_min, h0, Int(ell), maxn, max_cells, max_nodes)
    multipoles = _host_flat_buffer(TF, basis_info, max_nodes)
    locals = _host_flat_buffer(TF, basis_info, max_nodes)
    source_bodies = Matrix{TF}(undef, dpb, maxn)
    output = zeros(TF, hessian ? 13 : 4, maxn)
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
        hierarchical_noffsets=hierarchical ? length(hierarchical_tables.push_offsets) : 0,
        ell_axes, first_level=root_level)
    counters = CUDARadixTransferCounters()
    occupancy = hierarchical ? RadixLevelOccupancy(Int(ell);
        max_bytes=stencil_policy.dense_occupancy_max_bytes,
        max_dense_ell=stencil_policy.dense_occupancy_max_ell) : nothing
    hierarchical_apply_plan = hierarchical ? scratch.m2l_concat : nothing
    hierarchical_ctx = hierarchical ? HostHierarchicalM2LContext(
        hierarchical_tables, hierarchical_level_class_of, occupancy,
        class_level, class_offset,
        effective_offsets, hierarchical_apply_plan, stencil_policy.window_classes,
        first_m2l_level,
        zeros(Int, Int(ell) + 2), 0, zeros(Int, Int(ell) + 1), 0,
        false, zeros(UInt64, 5), zeros(UInt64, Int(ell) + 1)) : nothing
    sfs_ctx = sfs ? _host_sfs_context(TF, maxn, sfs_transposed,
        Int(sfs_active_row)) : nothing
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
        RadixStepCounts(0, 0, 0, 0, 0);
        sfs=sfs_ctx,
    )

    G = 1 << Int(ell)
    source_buffers = Tuple(Matrix{TF}(undef, data_per_body(system), maxn) for system in sources)
    adaptive_tree = adaptive === nothing ? nothing :
        _allocate_adaptive_radix_tree(TF, x_min, h0, adaptive, maxn)
    adaptive_lists = adaptive === nothing ? nothing :
        AdaptiveInteractionLists(adaptive_tree)
    adaptive_state = adaptive === nothing ? nothing :
        _allocate_adaptive_resident_lifecycle(TF, basis_info, options,
            adaptive_tree, adaptive_lists, invariant, dpb, maxn, hessian;
            sfs_ctx=sfs ? _host_sfs_context(TF, maxn, sfs_transposed,
                Int(sfs_active_row)) : nothing)
    cache = RadixFMMCache{TF,LH}(
        P, Int(ell), x_min, h0, ell_axes, box_extent, root_level, maxn, device,
        hessian, options, stencil_policy,
        accepted, rejected, max_cells, max_nodes, route_capacity, direct_capacity,
        state, hierarchical ? zeros(Int32, 0, 0, 0) : zeros(Int32, G, G, G),
        Vector{SVector{3,Int}}(undef, max_cells),
        zeros(Int, Int(ell) + 2), Vector{UInt64}(undef, maxn), Vector{Int}(undef, maxn),
        zeros(Int, 256), zeros(Int, 256), source_buffers, nothing, nothing,
        length(sources), false, 0,
        adaptive, adaptive_tree, adaptive_lists, adaptive_state,
        snapshot_locked_radix_settings(),
        sfs, sfs_transposed, nothing,
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
    # the workspace group count encodes the trimmed level range (task 037
    # stage 3): groups cover levels first_level:ell only
    first_level = ell - length(ws.m2m_groups)
    0 <= first_level <= ell && length(ws.l2l_groups) == length(ws.m2m_groups) ||
        throw(ArgumentError("resident cache workspace does not match grid depth ell=$ell"))
    for (gi, parent_level) in enumerate((ell - 1):-1:first_level)
        _refresh_group_edges!(ws.m2m_groups[gi], grid, level_offsets, parent_level + 1, :m2m)
    end
    for (gi, child_level) in enumerate((first_level + 1):ell)
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
        l2l_parent::Vector{Int}, l2l_child::Vector{Int}, parent_index, n_edges::Int,
        n_root_nodes::Int=1)
    # edges are the children of the retained levels (task 037 stage 3): the
    # first n_root_nodes nodes are roots with parent_index 0 and carry no edge
    @inbounds for edge in 1:n_edges
        node = edge + n_root_nodes
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
    # all data_per_body rows are carried, including radius row 4 (task 032);
    # systems narrower than the packed matrix are zero-padded
    nrows = size(source_bodies, 1)
    @inbounds for sorted_i in 1:n
        global_i = perm[sorted_i]
        isys = body_system[global_i]
        ibody = body_index[global_i]
        src = source_buffers[isys]
        nsys = min(size(src, 1), nrows)
        for row in 1:nsys
            source_bodies[row, sorted_i] = src[row, ibody]
        end
        for row in (nsys + 1):nrows
            source_bodies[row, sorted_i] = zero(TF)
        end
    end
    return source_bodies
end

function _refresh_hierarchical_route_telemetry!(state,
        ctx::HostHierarchicalM2LContext)
    noffsets = length(ctx.tables.push_offsets)
    total = 0
    fill!(ctx.routes_per_level, 0)
    last_count = 0
    for level in ctx.first_m2l_level:state.grid.ell
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
    recenter!(cache::RadixFMMCache, systems; bounds=nothing, padding=0.05)

Re-anchor the cache's fixed domain box (task 032, spec §4). The box is part of
the cache's invariant contract: bodies leaving it make the next `fmm!` throw,
and `fmm!` never recenters implicitly. When the physical domain should move or
resize, the consumer calls `recenter!` explicitly between evaluations, before
the next `fmm!`.

- `bounds = (x_min, box_size)` is the deterministic fast path (recommended for
  consumers that already track their domain); caller-supplied bounds are final
  and are **not** padded. `box_size` may be a scalar (cubic rebuild) or a
  3-vector (rectangular rebuild, task 037), regardless of the cache's current
  shape.
- With `bounds = nothing`, the union bounds of all live bodies are derived:
  host-resident systems through `get_position`, device-resident systems by a
  device min/max reduction over their persistent packed source buffers
  (refilled via `source_to_buffer!` first; only six extrema scalars reach the
  host). `padding` is a nonnegative fraction of the tight cube's side added on
  each face: `x_min = lo - padding*L_tight`, `L = (1 + 2*padding)*L_tight`.
  A rectangular cache (non-uniform `ell_axes`) applies the same convention per
  axis and then pads each shorter extent symmetrically to the power-of-two leaf
  count required by the shared cubic leaf width. Thus the snapped rectangular
  box remains centered on the tight cloud; rectangularity (and, for a
  similar-shaped cloud, the resolved `ell_axes`) is preserved.

Validation errors (`ArgumentError`) — an empty system, non-finite or
nonpositive bounds, negative padding, a changed system count, or a live count
above `max_n_bodies` — leave the cache unmodified and usable.

Implementation (geometry-rebuild fallback, user decision 2026-08-05): the
geometry-dependent state — stencil classification, operator tables, grid
keying, device geometry — is re-derived by re-running the construction path at
the new bounds with the cache's own parameters (`expansion_order`, `ell`,
`max_n_bodies`, `hessian`, `options`, and the cache's policy re-anchored to the
new box: a `HierarchicalRigidStencil` gets its box-derived tolerance re-derived
via `rigid_stencil_epsilon` at the new `h0` — the rigid near set and level
schedule are preserved exactly — while a flat `ConstantPAnalyticStencil` keeps
its tolerance and re-classifies; a hierarchical policy carrying custom
source/chi strengths incompatible with the re-derived tolerance fails the
construction accuracy gate loudly rather than running wrong), then swapped into
the existing cache object in place; the object identity consumers hold remains
valid, and
step-count prefixes restart so no stale state is trusted. Consequences to plan
around: `recenter!` costs about as much as cache construction, transiently
holds a second set of buffers (device caches: transiently ~2x device memory),
and restarts the transfer counters (a construction-equivalent event — route and
operator uploads recur here, never in ordinary steps). The zero-cost
alternative — normalized unit-cube internal coordinates making the operator
tables box-size-invariant — is recorded as a task 035 lever.
"""
function recenter!(cache::RadixFMMCache{TF,LH}, systems;
        bounds=nothing, padding::Real=0.05) where {TF,LH}
    systems_tuple = to_tuple(systems)
    length(systems_tuple) == cache.n_systems || throw(ArgumentError(
        "recenter! got $(length(systems_tuple)) systems for a cache built with " *
        "$(cache.n_systems); the system set is part of the cache contract"))
    padding >= 0 || throw(ArgumentError("recenter! padding must be nonnegative"))
    rectangular = cache.ell_axes != SVector(cache.ell, cache.ell, cache.ell)
    n = get_n_bodies(systems_tuple)
    n <= cache.max_n_bodies || throw(ArgumentError(
        "recenter! live body count n=$n exceeds the cache capacity " *
        "max_n_bodies=$(cache.max_n_bodies)"))
    if bounds === nothing
        lo, hi = _recenter_union_bounds(cache, systems_tuple)
        (all(isfinite, lo) && all(isfinite, hi)) || throw(ArgumentError(
            "recenter! derived non-finite body bounds; check body positions"))
        if rectangular
            # task 037 stage 2: a rectangular cache keeps per-axis tight extents
            # (same margin convention as the cube, applied per axis), so the
            # rebuild resolves vector bounds and rectangularity is preserved
            ext_tight = hi .- lo
            (ext_tight[1] > zero(TF) && ext_tight[2] > zero(TF) &&
                ext_tight[3] > zero(TF)) || throw(ArgumentError(
                "recenter! on a rectangular cache derived a degenerate " *
                "(zero-extent) axis; pass explicit bounds=(x_min, box_size)"))
            x_min_new = lo .- TF(padding) .* ext_tight
            L_new = (1 + 2 * TF(padding)) .* ext_tight
            # `_resolve_radix_ell_axes` pads short axes upward to power-of-two
            # leaf counts. Apply that padding equally on both faces for derived
            # bounds; retaining the raw lower face shifts the leaf lattice and
            # can inflate occupied/direct/M2L counts for a centered cloud.
            center_new = (lo + hi) / 2
            _, _, snapped_extent =
                _resolve_radix_ell_axes(L_new, cache.ell, TF)
            x_min_new = center_new - snapped_extent / 2
            L_new = snapped_extent
        else
            L_tight = max(hi[1] - lo[1], hi[2] - lo[2], hi[3] - lo[3])
            L_tight > zero(TF) || throw(ArgumentError(
                "recenter! derived a degenerate (zero-extent) body cloud; pass " *
                "explicit bounds=(x_min, box_size)"))
            x_min_new = lo .- TF(padding) * L_tight
            L_new = (1 + 2 * TF(padding)) * L_tight
        end
    else
        x_min_new = SVector{3,TF}(bounds[1])
        # caller-supplied bounds are final: a scalar box_size rebuilds cubic, a
        # 3-vector rebuilds rectangular, regardless of the cache's current shape
        L_new = bounds[2] isa Real ? TF(bounds[2]) : SVector{3,TF}(bounds[2])
        all(isfinite, x_min_new) && all(isfinite, L_new) || throw(ArgumentError(
            "recenter! bounds must be finite"))
        all(>(zero(TF)), L_new) ||
            throw(ArgumentError("recenter! box_size must be positive"))
    end
    # Build the replacement first: any failure (empty system, body outside the
    # requested bounds, capacity) leaves the original cache untouched.
    # the SFS configuration lives on the state, not the cache fields; dropping
    # it here would strip sfs from the rebuilt cache and the next
    # fmm!(...; sfs=true) evaluation throws (052 stage-d regression)
    old_sfs = cache.state.sfs
    fresh = RadixFMMCache(systems_tuple, systems_tuple;
        expansion_order=cache.expansion_order, ell=cache.ell,
        max_n_bodies=cache.max_n_bodies, bounds=(x_min_new, L_new),
        lamb_helmholtz=LH, hessian=cache.hessian, device=cache.device,
        sfs=old_sfs !== nothing,
        sfs_transposed=old_sfs === nothing ? true : old_sfs.transposed,
        sfs_active_row=old_sfs === nothing ? 0 : old_sfs.active_row,
        options=cache.options,
        policy=_recentered_policy(cache.policy, cache.expansion_order,
            maximum(L_new) / 2, cache.ell, TF, LH),
        adaptive=cache.adaptive)
    for f in fieldnames(RadixFMMCache)
        setfield!(cache, f, getfield(fresh, f))
    end
    return cache
end

_config_normalization(::ConstantPStencilConfig{TF,LH,N}) where {TF,LH,N} = N

# The flat analytic stencil keeps its tolerance and re-classifies at the new
# box; the hierarchical rigid stencil keeps its near set and level schedule
# exactly and re-derives the box-scaled tolerance those sets realize (the
# construction-time _verify_hierarchical_classifier! gate requires it).
_recentered_policy(policy::ConstantPAnalyticStencil, P, h0_new, ell, ::Type, LH) = policy

function _recentered_policy(policy::HierarchicalRigidStencil, P, h0_new, ell,
        ::Type{TF}, LH) where TF
    cfg = policy.config
    eps_new = rigid_stencil_epsilon(cfg.P_phi, h0_new, ell, policy.near_radius2;
        lamb_helmholtz=LH, TF)
    config = ConstantPStencilConfig(cfg.P_phi, eps_new, cfg.source_strength;
        chi_strength=cfg.chi_strength, lamb_helmholtz=LH,
        normalization=_config_normalization(cfg))
    return HierarchicalRigidStencil(config;
        near_radius2=policy.near_radius2, level_radii2=policy.level_radii2,
        window_classes=policy.window_classes,
        dense_occupancy_max_bytes=policy.dense_occupancy_max_bytes,
        dense_occupancy_max_ell=policy.dense_occupancy_max_ell)
end

function _recenter_union_bounds(cache::RadixFMMCache{TF}, systems::Tuple) where TF
    lox = TF(Inf); loy = TF(Inf); loz = TF(Inf)
    hix = -TF(Inf); hiy = -TF(Inf); hiz = -TF(Inf)
    for (isys, system) in enumerate(systems)
        n_sys = get_n_bodies(system)
        n_sys > 0 || throw(ArgumentError(
            "recenter! requires at least one live body in every system " *
            "(system $isys is empty)"))
        if residency(system) isa DeviceResident
            cache.device || throw(ArgumentError(
                "DeviceResident system $isys requires a device=true cache"))
            buf = cache.device_ctx.device_sources[isys]
            _fill_device_source_buffer!(view(buf, :, 1:n_sys), system)
            lox = min(lox, TF(minimum(view(buf, 1, 1:n_sys))))
            loy = min(loy, TF(minimum(view(buf, 2, 1:n_sys))))
            loz = min(loz, TF(minimum(view(buf, 3, 1:n_sys))))
            hix = max(hix, TF(maximum(view(buf, 1, 1:n_sys))))
            hiy = max(hiy, TF(maximum(view(buf, 2, 1:n_sys))))
            hiz = max(hiz, TF(maximum(view(buf, 3, 1:n_sys))))
        else
            for i in 1:n_sys
                x = get_position(system, i)
                lox = min(lox, TF(x[1])); loy = min(loy, TF(x[2])); loz = min(loz, TF(x[3]))
                hix = max(hix, TF(x[1])); hiy = max(hiy, TF(x[2])); hiz = max(hiz, TF(x[3]))
            end
        end
    end
    return SVector{3,TF}(lox, loy, loz), SVector{3,TF}(hix, hiy, hiz)
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
    _assert_radix_positions_in_box(systems, cache.x_min, cache.box_extent)

    t_stage = profiling ? time_ns() : UInt64(0)
    update_radix_grid!(grid, systems, cache.body_keys, cache.sort_scratch,
        cache.sort_counts, cache.sort_offsets, cache.level_offsets,
        cache.root_level)
    profiling && (ctx.update_stage_ns[1] = time_ns() - t_stage)
    n_cells = grid.n_cells
    n_nodes = cache.level_offsets[end]

    for (isys, system) in enumerate(systems)
        source_to_buffer!(cache.source_buffers[isys], system, 1:get_n_bodies(system))
    end
    _pack_radix_source_bodies!(state.source_bodies, grid.perm, grid.body_system,
        grid.body_index, cache.source_buffers, n)
    # Task 040: with the adaptive lifecycle armed, the global geometry gate is
    # replaced by the theory §5 per-cell sticky demotion gate (construction
    # requires it armed for regularized kernels), so the global throw is
    # skipped — one locally fat sigma must not force a globally shallow tree.
    cache.adaptive === nothing && _direct_kernel_geometry_gate!(cache,
        state.options.direct_kernel, state.source_bodies, n)

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
            grid.leaf_to_node, grid.ell, n_cells, RadixRouteSelection(),
        )
    end

    !hierarchical && plan isa ResidentM2LFactoredPlan && _refresh_factored_m2l_routes!(plan,
        state.route_sources::Vector{Int}, state.route_targets::Vector{Int}, n_routes)
    !hierarchical && plan isa ResidentM2LPrecomputedYPlan && _refresh_precomputed_y_m2l_routes!(plan,
        state.route_sources::Vector{Int}, state.route_targets::Vector{Int}, n_routes)
    !hierarchical && plan isa ResidentM2LDensePlan && _refresh_dense_m2l_routes!(plan,
        state.route_sources::Vector{Int}, state.route_targets::Vector{Int}, n_routes)

    t_stage = profiling ? time_ns() : UInt64(0)
    # multi-root tree edges (task 037 stage 3): every node at root_level is a
    # root, so the edge count is n_nodes - n_root_nodes (legacy: one level-0
    # root, n_nodes - 1)
    n_root_nodes = cache.level_offsets[cache.root_level + 2]
    n_edges = max(n_nodes - n_root_nodes, 0)
    resize!(state.m2m_parent_routes, n_edges)
    resize!(state.m2m_child_routes, n_edges)
    resize!(state.l2l_parent_routes, n_edges)
    resize!(state.l2l_child_routes, n_edges)
    _refresh_radix_tree_routes!(state.m2m_parent_routes, state.m2m_child_routes,
        state.l2l_parent_routes, state.l2l_child_routes, grid.parent_index, n_edges,
        n_root_nodes)

    _refresh_resident_stage_groups!(state.scratch, grid, cache.level_offsets)
    profiling && (ctx.update_stage_ns[5] = time_ns() - t_stage)

    counts = state.counts
    counts.n_bodies = n
    counts.n_cells = n_cells
    counts.n_nodes = n_nodes
    counts.n_routes = n_routes
    counts.n_direct = n_direct

    # opt-in adaptive octree refresh (task 039): rebuilds the adaptive tree and
    # its U/V/W/X lists in place within capacity; no-op unless the cache was
    # constructed with an AdaptiveTreePolicy. The uniform structures above are
    # unaffected.
    cache.adaptive_tree === nothing || _refresh_adaptive_radix!(cache, systems)

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

# task 048: lazy per-system 3-row host SFS scatter buffers (allocated once at
# the first sfs=true evaluation, sized to capacity so live-count changes reuse
# them; the finalize writes only the per-system body prefix)
function _radix_cache_sfs_buffers!(cache::RadixFMMCache{TF}, targets::Tuple) where TF
    sb = cache.sfs_target_buffers
    if !(sb isa Tuple) || length(sb) != length(targets)
        sb = Tuple(zeros(TF, 3, cache.max_n_bodies) for _ in targets)
        cache.sfs_target_buffers = sb
    end
    return sb
end

#------- backend-agnostic device source-buffer plumbing -------#
#
# These four lived in translate_batched_cuda.jl, which is `include`d only when
# CUDA is available, but none of them contains a CUDA type or a CUDA launch:
# they are `source_to_buffer!` dispatch, a `copyto!`, and a residency query, all
# generic over the array type. Their placement made them CUDA-only at run time
# — the same defect as `_radix_offsets_matrix` — and
# `_recenter_union_bounds` (above) already calls `_fill_device_source_buffer!`
# from generic code. Moved here so a KA `_radix_cache_device_step!` can reach
# them; behavior is unchanged and the CUDA path resolves the same methods.

function _has_device_source_to_buffer_method(device_buffer, system, sort_index)
    sig = Tuple{typeof(device_buffer),typeof(system),typeof(sort_index)}
    return hasmethod(source_to_buffer!, sig)
end

# identity permutation: a range, matching the documented `sort_index` default in
# compatibility.jl. `collect` here allocated an 8 MB Vector{Int} every step at
# n=1e6 (14% of per-step host allocation, task 028).
function _fill_device_source_buffer!(device_buffer, system)
    sort_index = Base.OneTo(get_n_bodies(system))
    _has_device_source_to_buffer_method(device_buffer, system, sort_index) ||
        throw(ArgumentError(
            "DeviceResident CUDA source systems must overload FastMultipole.source_to_buffer!(device_buffer, system, sort_index)",
        ))
    source_to_buffer!(device_buffer, system, sort_index)
    return device_buffer
end

_radix_any_host_resident(systems::Tuple) =
    any(residency(system) isa HostResident for system in systems)

# Refresh the persistent per-system device source buffers. Host-resident systems
# repack into their pinned staging and upload the valid column prefix (one upload
# per system per step); device-resident systems fill the valid prefix of their
# persistent buffer in place through their source_to_buffer! overload (no
# transfer, no allocation — task 032 gap-5 fix).
function _radix_cache_refresh_source_buffers!(ctx, systems::Tuple, ::Type{TF}) where TF
    return ntuple(length(systems)) do isys
        system = systems[isys]
        n_sys = get_n_bodies(system)
        device_buffer = ctx.device_sources[isys]
        if residency(system) isa HostResident
            staging = ctx.host_stagings[isys]
            source_to_buffer!(staging, system, 1:n_sys)
            # linear-prefix copy: the first n_sys columns are contiguous
            copyto!(device_buffer, 1, staging, 1, size(staging, 1) * n_sys)
            ctx.counters.body_uploads += 1
        else
            _fill_device_source_buffer!(view(device_buffer, :, 1:n_sys), system)
        end
        view(device_buffer, :, 1:n_sys)
    end
end

# Device-resident construction/step; redefined by translate_batched_cuda.jl (task
# 023 step 7) once load_cuda_radix_lifecycle!() has run.
function _radix_cache_device_build(args...; kwargs...)
    hook = _RADIX_DEVICE_BUILD_HOOK[]
    hook === nothing && throw(CUDARadixUnavailable(cuda_radix_status()))
    return hook(args...; kwargs...)
end

# Device-resident dense M2L plan construction (task 023f); redefined by
# translate_batched_cuda.jl. Reached only from the device-mode branch of
# _radix_cache_workspace, so on a CPU-only build this stub never runs, but keep it
# defined so the host method resolves.
function _build_cuda_dense_m2l_plan(args...)
    throw(CUDARadixUnavailable(cuda_radix_status()))
end

function _radix_cache_device_step!(cache::RadixFMMCache, targets::Tuple, switches::Tuple;
        sfs::Bool=false)
    hook = _RADIX_DEVICE_STEP_HOOK[]
    hook === nothing && throw(CUDARadixUnavailable(cuda_radix_status()))
    return hook(cache, targets, switches; sfs)
end

#------- adaptive octree host resident lifecycle: state assembly + drivers (task 040) -------#
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
        basis_info::OperatorBasisInfo{B,LH}, options::CUDARadixLifecycleOptions{TF},
        tree::AdaptiveRadixTree{TF}, lists::AdaptiveInteractionLists,
        invariant::OperatorInvariantCache, dpb::Int, maxn::Int,
        hessian::Bool; sfs_ctx=nothing) where {TF,B,LH}
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
        invariant, workspace, CUDARadixTransferCounters(), options,
        RadixStepCounts(0, 0, 0, 0, 0);
        sfs=sfs_ctx,
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
