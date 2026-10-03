#------- KA LIFECYCLE DRIVER -------#
#
# The UNIFORM radix lifecycle body -- the one FLOWVPM actually runs: it builds
# its `RadixFMMCache` with `ell`/`near_radius2`/`window_classes`.
#
# The uniform per-step body is only three stages and rebuilds no tree (the
# lattice is fixed at cache construction):
#     nearfield -> B2M -> [M2M -> M2L -> L2L -> L2B]
#
# This driver calls the ext's `ka_*` stage drivers, so "KA" is unambiguous on
# every backend.
#
# No `KA.synchronize` per stage: a sync per kernel costs 1.37-2.7x on this
# code. The driver issues none at all; every stage is ordered on one queue and
# `ka_radix_cache_device_step!` syncs once before returning.

"""
    ka_lifecycle_body!(state; extra_tree=())

Run the uniform radix lifecycle over `state` entirely with KA kernels. `state`
may be resident on any KA backend. The B2M and L2B team sizes are the drivers'
own defaults (128 and 64): each is the per-cell team size its kernel's work
split is built on (for B2M also the `@localmem` reduction extent), not a
tuning surface.
"""
function ka_lifecycle_body!(state::FastMultipole.DeviceResidentRadixState{TF,B,LH};
        extra_tree::Tuple=()) where {TF,B,LH}
    ws = state.scratch
    ws isa FastMultipole.ResidentOperatorWorkspace || throw(ArgumentError(
        "ka_lifecycle_body! requires a ResidentOperatorWorkspace in state.scratch"))

    backend = KA.get_backend(state.output)
    # 1. nearfield (clears state.output first)
    ka_launch_nearfield!(state; clear=true)   # workgroup from _nf_config
    _utick!(:lc_near, backend)

    # 2. B2M, then any extra source system the tree carries (before M2M, so the
    #    upward pass picks it up)
    ka_launch_b2m!(state)
    _utick!(:lc_b2m, backend)
    for prepared in extra_tree
        ka_extra_tree_b2m!(state, prepared)
    end
    isempty(extra_tree) || _utick!(:lc_extra_b2m, backend)

    # 3. far field: M2M -> M2L -> L2L, then L2B. No non-leaf zeroing: every
    #    ka_launch_b2m! method zero-fills the whole multipole buffer, and the
    #    extra-tree B2M adds into leaf columns only.
    for group in ws.m2m_groups
        ka_resident_stage_group_apply!(state.multipoles, state.multipoles, group, ws, :m2m)
    end
    _utick!(:lc_m2m, backend)
    ka_launch_m2l!(state, ws)
    _utick!(:lc_m2l, backend)
    for group in ws.l2l_groups
        ka_resident_stage_group_apply!(state.locals, state.locals, group, ws, :l2l)
    end
    _utick!(:lc_l2l, backend)
    ka_launch_l2b!(state)
    _utick!(:lc_l2b, backend)
    return state
end


"""
    ka_launch_m2l!(state, ws)

M2L stage of [`ka_lifecycle_body!`](@ref): applies the epoch window cache of
the state's `DeviceHierarchicalM2LContext` (every KA device cache is
hierarchical; a device cache allocates its flat route arrays empty).
"""
function ka_launch_m2l!(state::FastMultipole.DeviceResidentRadixState{TF,B,LH},
        ws) where {TF,B,LH}
    hctx = state.interaction_list
    hctx isa FastMultipole.DeviceHierarchicalM2LContext || throw(ArgumentError(
        "ka_launch_m2l! requires a DeviceHierarchicalM2LContext; got $(typeof(hctx))"))
    return ka_hierarchical_m2l!(state, hctx, ws)
end
