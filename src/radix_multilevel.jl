#------- multilevel insertion of the masked bodies -------#
#
# A body the oversize policy took out of the tree (its core reach exceeds what the
# leaf stencil admits) is put back at the deepest level whose stencil does admit it:
# its multipole joins that level's node (after B2M, so the upward pass carries it
# coarser), and every target whose cell at that level is near the body's cell gets
# it directly with the regularized kernel. Every other target receives it once
# through M2L at that level or coarser, beyond its reach. Why that covers each pair
# exactly once: the hierarchy's M2L at level L pairs cells far at L whose parents
# were near; with a non-increasing radius schedule and q >= 3, near at a level
# implies near at every coarser one, so the pairs not covered by M2L down to level
# `ls` are exactly those near at `ls`. A body too large for the coarsest M2L level
# stays all-pairs. (Checked combinatorially on 33M pairs, 2026-10-03.)

# the near radius the cache's hierarchical stencil uses at `level`
function _radix_level_radius2(policy::HierarchicalRigidStencil, level::Int)
    isempty(policy.level_radii2) && return policy.near_radius2
    return policy.level_radii2[level - 1]          # schedule covers levels 2:ell
end

# the coarsest level that carries M2L classes (FLOWVPM's validation convention)
function _radix_first_m2l_level(cache)
    R, L_allnear = _radix_root_level(cache.ell_axes, cache.ell, cache.policy.near_radius2)
    return R == L_allnear ? R + 1 : R
end

# sorted leaf cell holding slot `s` (cell_ranges: first slot and count per cell)
function _radix_cell_of_slot(cell_ranges, n_cells::Int, s::Int)
    lo = 1; hi = n_cells
    @inbounds while lo < hi
        mid = (lo + hi + 1) >>> 1
        cell_ranges[1, mid] <= s ? (lo = mid) : (hi = mid - 1)
    end
    return lo
end

"""
    radix_multilevel_plan(cache, mb::MaskedBodies, masked_idx) -> NamedTuple

Where each masked body goes: its level `ls` (0: all-pairs), the node at that
level holding it, the node's center and the body's coarse cell coordinates; the
bodies grouped by node for the multipole insertion; the all-pairs remainder.
`masked_idx` are the bodies' global indices into system 1 (their slots locate them
in the grid's own cells). Host cache.
"""
function radix_multilevel_plan(cache, mb, masked_idx::AbstractVector{<:Integer})
    state = cache.state; grid = state.grid
    ell = cache.ell; K = size(mb.buffer, 2)
    levels = zeros(Int, K); nodes = zeros(Int, K); coords = zeros(Int, 3, K)
    cubic = cache.ell_axes == SVector(ell, ell, ell) && cache.policy isa HierarchicalRigidStencil
    if cubic && K > 0
        first_level = _radix_first_m2l_level(cache)
        n_cells = Int(state.counts.n_cells)
        perm = state.host_body_perm; sys = state.host_body_system_ids; bidx = state.host_body_indices
        slot_of = Dict{Int,Int}()
        for s in 1:Int(state.counts.n_bodies)
            g = perm[s]
            sys[g] == 1 && (slot_of[bidx[g]] = s)
        end
        margin = mb.evaluator.margin
        for j in 1:K
            reach = margin * _extra_regularization_reach(mb.kernel, mb.buffer, j)
            s = get(slot_of, Int(masked_idx[j]), 0)
            s == 0 && continue
            for l in (ell - 1):-1:first_level
                h = 2 * Float64(cache.h0) / (1 << l)
                if _ball_stencil_min_gap(_radix_level_radius2(cache.policy, l)) * h >= reach
                    levels[j] = l; break
                end
            end
            levels[j] == 0 && continue
            c = _radix_cell_of_slot(state.cell_ranges, n_cells, s)
            node = Int(grid.leaf_to_node[c])
            for _ in 1:(ell - levels[j])
                node = Int(grid.parent_index[node])
            end
            nodes[j] = node
            coords[:, j] .= morton_decode(UInt64(grid.cell_keys[c]) >> (3 * (ell - levels[j])), levels[j])
        end
    end
    inserted = findall(>(0), levels)
    # group the inserted bodies by node (stable), one multipole column per node
    order = inserted[sortperm(nodes[inserted])]
    group_nodes = Int[]; group_ranges = zeros(Int, 2, 0)
    for (k, j) in enumerate(order)
        if isempty(group_nodes) || group_nodes[end] != nodes[j]
            push!(group_nodes, nodes[j]); group_ranges = hcat(group_ranges, [k, 0])
        end
        group_ranges[2, end] += 1
    end
    return (; levels, nodes, coords, order, group_nodes, group_ranges,
              fallback = findall(==(0), levels), mb)
end

"""
    _radix_multilevel_b2m!(state, plan)

Add the inserted bodies' multipoles to their nodes (host), on top of what B2M put
there; run before the upward pass.
"""
function _radix_multilevel_b2m!(state::DeviceResidentRadixState{TF,B,LH}, plan) where {TF,B,LH}
    isempty(plan.order) && return state
    grid = state.grid; mb = plan.mb
    orders = state.invariant_cache.basis_info.orders
    buf = mb.buffer[:, plan.order]
    centers = Matrix{TF}(grid.node_centers[:, plan.group_nodes])
    _resident_extra_b2m_kernel!(phi_slab(state.multipoles), chi_slab(state.multipoles),
        mb, buf, plan.group_ranges, centers, plan.group_nodes,
        orders.P_phi, orders.P_active, length(plan.group_nodes), Val(LH))
    return state
end

"""
    _radix_multilevel_near!(state, plan)

The direct part (host): each inserted body against every target in the cells near
its cell at its level, with the regularized kernel; the remainder all-pairs.
Accumulates into `state.output`.
"""
function _radix_multilevel_near!(state::DeviceResidentRadixState{TF}, cache, plan) where TF
    mb = plan.mb; kernel = mb.kernel
    out = state.output; bodies = state.source_bodies
    keys = state.grid.cell_keys; ranges = state.cell_ranges
    n_cells = Int(state.counts.n_cells); ell = cache.ell
    hs = size(out, 1) >= 13 && _extra_pair_has_hessian(kernel)
    ep = _extra_emits_potential(kernel)
    for j in plan.order
        l = plan.levels[j]; q = _radix_level_radius2(cache.policy, l)
        r = isqrt(q); G = 1 << l; d = 3 * (ell - l)
        c0 = SVector{3,Int}(plan.coords[1, j], plan.coords[2, j], plan.coords[3, j])
        for oz in -r:r, oy in -r:r, ox in -r:r
            ox * ox + oy * oy + oz * oz <= q || continue
            c = c0 + SVector{3,Int}(ox, oy, oz)
            all(0 .<= c .< G) || continue
            klo = UInt64(morton_key(c, l)) << d; khi = klo + (UInt64(1) << d)
            a = searchsortedfirst(view(keys, 1:n_cells), klo)
            b = searchsortedfirst(view(keys, 1:n_cells), khi) - 1
            a <= b || continue
            s1 = ranges[1, a]; s2 = ranges[1, b] + ranges[2, b] - 1
            @inbounds for i in s1:s2
                xi = bodies[1, i]; yi = bodies[2, i]; zi = bodies[3, i]
                if hs
                    u, gx, gy, gz, h1, h2, h3, h4, h5, h6, h7, h8, h9 =
                        _extra_pair_ugh(kernel, xi, yi, zi, mb.buffer, j)
                    ep && (out[1, i] += u)
                    out[2, i] += gx; out[3, i] += gy; out[4, i] += gz
                    out[5, i] += h1; out[6, i] += h2; out[7, i] += h3
                    out[8, i] += h4; out[9, i] += h5; out[10, i] += h6
                    out[11, i] += h7; out[12, i] += h8; out[13, i] += h9
                else
                    u, gx, gy, gz = _extra_pair_ug(kernel, xi, yi, zi, mb.buffer, j)
                    ep && (out[1, i] += u)
                    out[2, i] += gx; out[3, i] += gy; out[4, i] += gz
                end
            end
        end
    end
    if !isempty(plan.fallback)
        _host_targets_from_extra_source!(out, kernel, bodies, Int(state.counts.n_bodies),
            mb.buffer[:, plan.fallback], Val(size(out, 1) >= 13 && _extra_pair_has_hessian(kernel)))
    end
    return state
end
