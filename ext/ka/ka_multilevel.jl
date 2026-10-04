#------- multilevel insertion of the masked bodies (KA) -------#
#
# src/radix_multilevel.jl has the rule and the exactly-once argument. Per step:
# the levels come from the host (radix_multilevel_levels), one kernel finds each
# body's slot, leaf, ancestor node at its level and that node's center; the host
# computes one multipole column per node (a masked set is small) and the
# lifecycle adds them after B2M, before M2M. The direct part runs level by level,
# one thread per target: its cell at the level, the stencil offsets, a dense
# per-level map from level cell to the masked bodies sorted by (cell, index).
# Fixed order, no atomics: deterministic.

@inline function _ka_morton_decode3(key::UInt64, l::Int)
    i = 0; j = 0; k = 0
    for bit in 0:(l - 1)
        i |= Int((key >> (3 * bit)) & UInt64(1)) << bit
        j |= Int((key >> (3 * bit + 1)) & UInt64(1)) << bit
        k |= Int((key >> (3 * bit + 2)) & UInt64(1)) << bit
    end
    return i, j, k
end

# each masked body: its ancestor node at its level, its level cell (linear) and the node's center
@kernel function _ml_locate_kernel!(node_out, key_out, center_out, @Const(mslot), @Const(levels),
        @Const(cell_ranges), @Const(cell_keys), @Const(leaf_to_node), @Const(parent_index),
        @Const(node_centers), n_cells, ell, K)
    j = @index(Global)
    @inbounds if j <= K
        s = Int(mslot[j]); l = Int(levels[j])
        if s > 0 && l > 0
            c = _ka_cell_of(cell_ranges, n_cells, s)
            node = Int(leaf_to_node[c])
            for _ in 1:(ell - l)
                node = Int(parent_index[node])
            end
            node_out[j] = Int32(node)
            x, y, z = _ka_morton_decode3(cell_keys[c] >> (3 * (ell - l)), l)
            key_out[j] = Int32(x + (y + z << l) << l)    # linear index x + G(y + Gz)
            center_out[1, j] = node_centers[1, node]
            center_out[2, j] = node_centers[2, node]
            center_out[3, j] = node_centers[3, node]
        else
            node_out[j] = Int32(0)
        end
    end
end

# offs[k + 1] = first position (1-based) of linear level cell k in the sorted cells
@kernel function _ml_offsets_kernel!(offs, @Const(keys), nk, ncell)
    k = @index(Global)
    @inbounds if k <= ncell + 1
        c = Int32(k - 1)
        lo = 1; hi = nk + 1
        while lo < hi
            mid = (lo + hi) >>> 1
            if keys[mid] < c
                lo = mid + 1
            else
                hi = mid
            end
        end
        offs[k] = Int32(lo)
    end
end

@kernel function _ml_near_kernel!(kernel, out, @Const(bodies), @Const(cell_ranges), @Const(cell_keys),
        @Const(offs), @Const(buf), n_cells, n_bodies, shift, l, q, r,
        ::Type{T}, ::Val{HS}, ::Val{EP}) where {T,HS,EP}
    i = @index(Global)
    @inbounds if i <= n_bodies
        c = _ka_cell_of(cell_ranges, n_cells, i)
        cx, cy, cz = _ka_morton_decode3(cell_keys[c] >> shift, l)
        G = 1 << l
        xi = bodies[1, i]; yi = bodies[2, i]; zi = bodies[3, i]
        u = zero(T); gx = zero(T); gy = zero(T); gz = zero(T)
        h1 = zero(T); h2 = zero(T); h3 = zero(T)
        h4 = zero(T); h5 = zero(T); h6 = zero(T)
        h7 = zero(T); h8 = zero(T); h9 = zero(T)
        for oz in -r:r, oy in -r:r, ox in -r:r
            x = cx + ox; y = cy + oy; z = cz + oz
            if ox * ox + oy * oy + oz * oz <= q && 0 <= x < G && 0 <= y < G && 0 <= z < G
                k = x + (y + z * G) * G
                for jj in Int(offs[k + 1]):(Int(offs[k + 2]) - 1)
                    if HS
                        du, dgx, dgy, dgz, dh1, dh2, dh3, dh4, dh5, dh6, dh7, dh8, dh9 =
                            FastMultipole._extra_pair_ugh(kernel, xi, yi, zi, buf, jj)
                        u += du; gx += dgx; gy += dgy; gz += dgz
                        h1 += dh1; h2 += dh2; h3 += dh3
                        h4 += dh4; h5 += dh5; h6 += dh6
                        h7 += dh7; h8 += dh8; h9 += dh9
                    else
                        du, dgx, dgy, dgz = FastMultipole._extra_pair_ug(kernel, xi, yi, zi, buf, jj)
                        u += du; gx += dgx; gy += dgy; gz += dgz
                    end
                end
            end
        end
        if EP
            out[1, i] += u
        end
        out[2, i] += gx; out[3, i] += gy; out[4, i] += gz
        if HS
            out[5, i] += h1; out[6, i] += h2; out[7, i] += h3
            out[8, i] += h4; out[9, i] += h5; out[10, i] += h6
            out[11, i] += h7; out[12, i] += h8; out[13, i] += h9
        end
    end
end

_ka_state_lh(::FastMultipole.DeviceResidentRadixState{TF,B,LH}) where {TF,B,LH} = LH

_ka_is_multilevel(s) = s isa FastMultipole.MaskedBodies

"""
    ka_multilevel_prepare(cache, mb) -> NamedTuple

Locate the masked bodies on the device and build what the lifecycle (`tree`:
multipole columns per node, added after B2M) and [`ka_multilevel_near!`](@ref)
(the per-level direct sets and the all-pairs remainder) need.
"""
function ka_multilevel_prepare(cache, mb)
    state = cache.state; grid = state.grid
    TF = eltype(state.output); LH = _ka_state_lh(state)
    backend = KA.get_backend(state.output)
    up(A) = (d = KA.allocate(backend, eltype(A), size(A)...); copyto!(d, A); d)
    K = size(mb.buffer, 2)
    levels = FastMultipole.radix_multilevel_levels(cache, mb)
    nodes = zeros(Int32, K); keys = zeros(Int32, K); centers = zeros(TF, 3, K)
    if any(>(0), levels)
        nf = FastMultipole.radix_nearfield(cache)
        idx = mb.idx; korder = sortperm(idx)
        mslot = KA.zeros(backend, Int32, K)
        _masked_slots_kernel!(backend, 256)(mslot, nf.body_perm, nf.body_system_ids, nf.body_indices,
            up(Int.(idx[korder])), up(korder), K, nf.n_bodies; ndrange = cld(nf.n_bodies, 256) * 256)
        nd = KA.zeros(backend, Int32, K); kd = KA.zeros(backend, Int32, K)
        cd = KA.zeros(backend, TF, 3, K)
        _ml_locate_kernel!(backend, 64)(nd, kd, cd, mslot, up(Int32.(levels)), state.cell_ranges,
            grid.cell_keys, grid.leaf_to_node, grid.parent_index, grid.node_centers,
            nf.n_cells, cache.ell, K; ndrange = cld(K, 64) * 64)
        copyto!(nodes, nd); copyto!(keys, kd); copyto!(centers, cd)
    end
    levels[nodes .== 0] .= 0                         # not resident: all-pairs
    buffer = TF.(mb.buffer)
    # one multipole column per node, the bodies grouped by node (stable)
    ins = findall(>(0), levels)
    order = ins[sortperm(nodes[ins])]
    starts = [k for k in eachindex(order) if k == 1 || nodes[order[k]] != nodes[order[k - 1]]]
    gnodes = nodes[order[starts]]; gcenters = centers[:, order[starts]]
    granges = vcat(starts', diff(vcat(starts, length(order) + 1))')
    orders = state.invariant_cache.basis_info.orders
    rows_phi = size(FastMultipole.phi_slab(state.multipoles), 1)
    rows_chi = LH ? size(FastMultipole.chi_slab(state.multipoles), 1) : rows_phi
    G = length(gnodes)
    phi = zeros(TF, rows_phi, G); chi = zeros(TF, rows_chi, G)
    # the node groups are independent (distinct columns): one chunk per thread
    sorted_buffer = buffer[:, order]
    Threads.@threads for chunk in collect(Iterators.partition(1:G, cld(G, Threads.nthreads())))
        FastMultipole._resident_extra_b2m_kernel!(phi, chi, mb, sorted_buffer, granges[:, chunk],
            gcenters[:, chunk], chunk, orders.P_phi, orders.P_active, length(chunk), Val(LH))
    end
    tree = (; phi = up(phi), chi = up(chi), nodes = up(gnodes))
    # the direct sets, one per level: bodies sorted by (level key, index)
    sets = map(sort!(unique(levels[ins]))) do l
        js = ins[levels[ins] .== l]
        js = js[sortperm(keys[js])]                  # sortperm is stable: index order within a key
        ncell = 1 << (3 * l)
        kd = up(keys[js]); offs = KA.allocate(backend, Int32, ncell + 1)
        _ml_offsets_kernel!(backend, 256)(offs, kd, length(js), ncell; ndrange = cld(ncell + 1, 256) * 256)
        (; l, q = FastMultipole._radix_level_radius2(cache.policy, l), offs, buf = up(buffer[:, js]))
    end
    fallback = findall(==(0), levels)
    return (; tree, sets, loose = buffer[:, fallback], kernel = mb.kernel)
end

"""
    ka_multilevel_near!(state, cache, prepared; workgroup)

The direct part of the multilevel insertion, after the lifecycle: each target
sums the masked bodies whose level cell is near its own, level by level, and the
bodies no level admits all-pairs.
"""
function ka_multilevel_near!(state::FastMultipole.DeviceResidentRadixState{TF}, cache, p;
        workgroup=KA_AUTO_WORKGROUP) where TF
    n = Int(state.counts.n_bodies)
    n == 0 && return state
    backend = KA.get_backend(state.output)
    wg = resolve_workgroup(backend, workgroup)
    dkernel = _ka_device_direct_kernel(p.kernel, TF, 0)
    hs = size(state.output, 1) >= 13
    ep = FastMultipole._extra_emits_potential(p.kernel)
    hsv = Val(hs && FastMultipole._extra_pair_has_hessian(dkernel))
    kern = _cached_kernel(_ml_near_kernel!, backend, wg)
    for s in p.sets
        kern(dkernel, state.output, state.source_bodies, state.cell_ranges, state.grid.cell_keys,
             s.offs, s.buf, Int(state.counts.n_cells), n, 3 * (cache.ell - s.l), s.l, s.q, isqrt(s.q),
             TF, hsv, Val(ep); ndrange = cld(n, wg) * wg)
    end
    _ka_launch_extra_buffer!(backend, wg, state.output, state.source_bodies, n, p.loose, dkernel, TF, hs, ep)
    return state
end
