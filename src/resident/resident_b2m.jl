#------- vortex B2M -------#
#
# Transcription of the octree path's `mirrored_source_to_vortex!` (bodytomultipole.jl)
# to the resident flat-buffer layout: regular harmonics of the *mirrored*
# offset `-Δx` evaluated on the fly (matching the scalar resident B2M's
# per-(n,m) recurrence style), with the octree path's `get_n`/`get_nm1` negative-m
# conjugate-symmetry rules folded into `_resident_vortex_q`. No sign changes:
# the octree chain `evaluate_local ∘ multipole_to_local! ∘ vortex B2M` matches
# the analytic Biot-Savart field of an off-center vorton to machine precision,
# and the resident M2L and L2B match those octree stages, so the octree vortex
# coefficients are the physical convention here. (The octree *Point{Source}* B2M's strength negation
# is an octree-pipeline quirk the resident scalar B2M deliberately omits; it has
# no analogue for the vortex.)

@inline _resident_vortex_q(mdx, mdy, mdz, n, m) =
    _resident_vortex_q(_resident_harmonic_setup(mdx, mdy, mdz), n, m)

@inline function _resident_vortex_q(setup::NTuple{5,<:Any}, n, m)
    TF = typeof(setup[1])
    if m < 0
        # conjugate symmetry per the octree get_n/get_nm1: Q_{n,-1} = -conj(Q_{n,1})
        (m == -1 && n >= 1) || return zero(TF), zero(TF)
        qre, qim = _resident_regular_harmonic_coeff(setup, n, 1)
        return -qre, qim
    end
    (m > n || n < 0) && return zero(TF), zero(TF)
    return _resident_regular_harmonic_coeff(setup, n, m)
end

# The three `_resident_vortex_q` lookups below share one offset, so the
# transcendental prologue is computed once here rather than inside each.
@inline function _resident_vortex_phi_contrib(mdx, mdy, mdz, vx, vy, vz, n, m)
    TF = typeof(mdx)
    setup = _resident_harmonic_setup(mdx, mdy, mdz)
    qmm1_re, qmm1_im = _resident_vortex_q(setup, n, m - 1)
    qm_re, qm_im = _resident_vortex_q(setup, n, m)
    qmp1_re, qmp1_im = _resident_vortex_q(setup, n, m + 1)
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

# Point dipole phi contribution: the octree path's `source_to_dipole!` recurrence
# (bodytomultipole.jl) on the resident regular harmonics of the offset x − c
# (the SOURCE convention, no mirroring): order-n coefficient from the order
# n−1 harmonics at m−1, m, m+1, with `_resident_vortex_q`'s bound and
# conjugate-symmetry rules standing in for the octree `get_nm1`. Returns the term
# BEFORE the scalar B2M's (−1)^(n+m) sign and conjugation, which the caller
# applies exactly as for a source. No strength negation, matching the resident
# scalar B2M. n = 0 contributes nothing.
@inline function _resident_dipole_contrib(dx, dy, dz, px, py, pz, n, m)
    TF = typeof(dx)
    n == 0 && return zero(TF), zero(TF)
    setup = _resident_harmonic_setup(dx, dy, dz)
    pmm1_re, pmm1_im = _resident_vortex_q(setup, n - 1, m - 1)
    pm_re, pm_im = _resident_vortex_q(setup, n - 1, m)
    pmp1_re, pmp1_im = _resident_vortex_q(setup, n - 1, m + 1)
    re = -px * TF(0.5) * (pmp1_im + pmm1_im) + py * TF(0.5) * (pmp1_re - pmm1_re) - pz * pm_re
    im = px * TF(0.5) * (pmp1_re + pmm1_re) + py * TF(0.5) * (pmp1_im - pmm1_im) - pz * pm_im
    return re, im
end

@inline function _resident_vortex_chi_contrib(mdx, mdy, mdz, vx, vy, vz, n, m)
    TF = typeof(mdx)
    setup = _resident_harmonic_setup(mdx, mdy, mdz)
    qmm1_re, qmm1_im = _resident_vortex_q(setup, n - 1, m - 1)
    qm_re, qm_im = _resident_vortex_q(setup, n - 1, m)
    qmp1_re, qmp1_im = _resident_vortex_q(setup, n - 1, m + 1)
    # the octree get_nm1 zeroes (n-1, m) for m == n and (n-1, m+1) for m+1 >= n;
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
    return _launch_resident_m2m!(state)
end

function _launch_host_m2l!(state::DeviceResidentRadixState{TF,B,LH}) where {TF,B,LH}
    return _launch_resident_m2l!(state, state.options.m2l_strategy)
end

function _launch_host_l2l!(state::DeviceResidentRadixState{TF,B,LH}) where {TF,B,LH}
    return _launch_resident_l2l!(state)
end

# Iterate the flat pair arrays (bounded by counts) rather than the one-shot
# interaction list so the recurring update path never rebuilds the list object;
# function barrier as in _launch_host_b2m!. The pair math comes from the
# `direct_kernel` functor stamped into the options at construction (compile-time
# specialization). The hard-coded `_host_direct_pairs_*_kernel!` functions in
# resident_pair_kernels.jl are not called here; the tests use them as the
# reference the functor path is checked against.
function _add_host_direct_pairs!(state::DeviceResidentRadixState)
    hsv = size(state.output, 1) >= 13 ? Val(true) : Val(false)
    # cheapened g/h mode (see _validated_host_gh_mode); Val() barrier
    # specializes the loop per mode
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

#------- two-pass additive-correction deficit sweep -------#
#
# Pass 2 of the TwoPassVortex hybrid: after the unmodified pass 1 (singular far
# field + the rho_c-partitioned direct nearfield above), add the regularization
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
# rho_t·σ_max — so the reach requirement the geometry gate enforces for the
# single-pass kernels holds here without a gate, and the primary near set only
# needs the rho_c adequacy (the _direct_kernel_geometry_gate! dispatch in
# radix_cache.jl).
# Zero per-step allocation: loop bounds and binary searches only. The KA device
# backend has no pass-2 sweep and refuses TwoPassVortex at cache build.
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
