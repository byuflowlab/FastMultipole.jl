#------- KA LIFECYCLE DRIVER (step 7a) -------#
#
# Mirror of `_cuda_lifecycle_body!` (src/translate_batched_cuda.jl) for the
# UNIFORM radix lifecycle -- the one FLOWVPM actually runs.
#
# Why uniform and not adaptive: `_radix_cache_device_step!` branches on
# `cache.adaptive === nothing`, and FLOWVPM builds its `RadixFMMCache` with
# `ell`/`near_radius2`/`window_classes` and NO adaptive policy, so it takes
# `run_cuda_radix_lifecycle!`. It could not take the adaptive path anyway: that
# lifecycle rejects `PartitionedVortex` (FLOWVPM's shipped default kernel, a
# task-040 deferral) and the Lamb-Helmholtz M2T hessian throws outright.
#
# The uniform per-step body is only three stages and rebuilds no tree (the
# lattice is fixed at cache construction):
#     nearfield -> B2M -> [M2M -> M2L -> L2L -> L2B]
#
# This driver deliberately calls the ext's STANDALONE `ka_*` stage drivers
# rather than FastMultipole's generic ones. The generic drivers dispatch their
# primitives on array type, which means they run KA kernels on Metal but NATIVE
# CUDA kernels on CuArrays -- correct for production, useless for a KA-vs-native
# A/B on one GPU. Going through the standalone drivers makes "KA" unambiguous on
# every backend, so the same state can be run both ways and compared.
#
# One `KA.synchronize` at the END of the driver, never per stage: per-kernel
# syncs cost 1.37-2.7x in earlier measurements on this code.

"""
    ka_lifecycle_body!(state; workgroup_b2m=128, workgroup=64, sync=true)

`workgroup` here is *not* the auto-resolved occupancy knob: it reaches
`ka_launch_nearfield!` and `ka_launch_l2b!`, where it is the per-pair/per-cell
team size their `@localmem` extents are declared against. It stays explicit.

Run the uniform radix lifecycle over `state` entirely with KA kernels. `state`
may be resident on any KA backend, including a `CuArray` state built by the
existing `RadixFMMCache(device=true)` -- which is how the KA-vs-native
comparison runs both arms over identical data with no second cache build.
"""
function ka_lifecycle_body!(state::FastMultipole.DeviceResidentRadixState{TF,B,LH};
        workgroup_b2m::Int=128, workgroup::Int=64, sync::Bool=true,
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
    ka_launch_b2m!(state; workgroup=workgroup_b2m)
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
    ka_launch_l2b!(state; workgroup=workgroup)
    _utick!(:lc_l2b, backend)

    sync && KA.synchronize(KA.get_backend(state.output))
    return state
end


"""
    ka_launch_m2l!(state, ws)

M2L stage of [`ka_lifecycle_body!`](@ref), branching on the resident
interaction context exactly as `_launch_cuda_resident_m2l!` does: a
`DeviceHierarchicalM2LContext` generates and applies route windows here, and
anything else is the flat whole-route concat apply.

The branch is not optional. FLOWVPM's `RadixFMMCache` carries the hierarchical
policy, and on that context `state.counts.n_routes` holds only the LAST
window's route count -- a flat apply would silently translate a fraction of the
V list and be wrong rather than merely slow.
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

