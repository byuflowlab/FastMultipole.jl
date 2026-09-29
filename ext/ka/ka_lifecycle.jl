#------- KA LIFECYCLE DRIVER (step 7a) -------#
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
# One `KA.synchronize` at the END of the driver, never per stage: per-kernel
# syncs cost 1.37-2.7x in earlier measurements on this code.

"""
    ka_lifecycle_body!(state; extra_tree=())

Run the uniform radix lifecycle over `state` entirely with KA kernels. `state`
may be resident on any KA backend. The B2M and L2B team sizes are the drivers'
own defaults (128 and 64): they are the per-cell team size the kernels'
`@localmem` extents are declared against, not a tuning surface.
"""
function ka_lifecycle_body!(state::FastMultipole.DeviceResidentRadixState{TF,B,LH};
        extra_tree::Tuple=()) where {TF,B,LH}
    ws = state.scratch
    ws isa FastMultipole.ResidentOperatorWorkspace || throw(ArgumentError(
        "ka_lifecycle_body! requires a ResidentOperatorWorkspace in state.scratch"))

    backend = KA.get_backend(state.output)
    # 1. nearfield (clears state.output, as CUDA's fill+nearfield does)
    ka_launch_nearfield!(state; clear=true)   # shape/workgroup from _nf_config
    _utick!(:lc_near, backend)

    # 2. B2M, then any extra source system the tree carries (before M2M, so the
    #    upward pass picks it up)
    ka_launch_b2m!(state)
    _utick!(:lc_b2m, backend)
    for prepared in extra_tree
        ka_extra_tree_b2m!(state, prepared)
    end
    isempty(extra_tree) || _utick!(:lc_extra_b2m, backend)

    # 3. far field: M2M -> M2L -> L2L, then L2B
    FastMultipole._zero_resident_nonleaf_multipoles!(state)
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

    KA.synchronize(backend)
    return state
end


"""
    ka_launch_m2l!(state, ws)

M2L stage of [`ka_lifecycle_body!`](@ref), branching on the resident
interaction context: a
`DeviceHierarchicalM2LContext` applies its epoch window cache here, and
anything else (the `host_radix_state` mirror used by the lifecycle gate) is the
flat whole-route concat apply over `state.route_sources`/`route_targets`.

The branch is not optional: a device cache allocates its flat route arrays
empty, while `state.counts.n_routes` holds the cached-stream total.
"""
function ka_launch_m2l!(state::FastMultipole.DeviceResidentRadixState{TF,B,LH},
        ws) where {TF,B,LH}
    hctx = state.interaction_list
    hctx isa FastMultipole.DeviceHierarchicalM2LContext &&
        return ka_hierarchical_m2l!(state, hctx, ws)
    fill!(state.locals.phi, zero(TF))
    LH && fill!(state.locals.chi, zero(TF))
    nroutes = state.counts.n_routes
    nroutes > 0 && ka_resident_m2l_concat_apply!(state.locals, state.multipoles, ws,
        state.route_sources, state.route_targets, nroutes)
    return state
end

