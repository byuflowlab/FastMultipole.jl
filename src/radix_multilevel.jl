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

"""
    radix_multilevel_levels(cache, mb::MaskedBodies) -> Vector{Int}

Each masked body's insertion level: the deepest level, from `ell - 1` up to the
coarsest M2L level, whose stencil gap admits its margin-scaled regularization
reach; 0 (all-pairs) when none does or the grid is not a cubic hierarchical one.
"""
function radix_multilevel_levels(cache, mb)
    ell = cache.ell; K = size(mb.buffer, 2)
    levels = zeros(Int, K)
    (cache.ell_axes == SVector(ell, ell, ell) && cache.policy isa HierarchicalRigidStencil) || return levels
    first_level = _radix_first_m2l_level(cache)
    margin = mb.margin
    # the largest reach each level admits, deepest first
    admits = [(l, _ball_stencil_min_gap(_radix_level_radius2(cache.policy, l)) * 2 * Float64(cache.h0) / (1 << l))
              for l in (ell - 1):-1:first_level]
    for j in 1:K
        reach = margin * _extra_regularization_reach(mb.kernel, mb.buffer, j)
        for (l, a) in admits
            if a >= reach
                levels[j] = l; break
            end
        end
    end
    return levels
end
