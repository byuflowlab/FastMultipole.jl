#------- host output finalization -------#
#
# The lifecycle output is a 4×n slab (scalar potential + gradient, sorted body
# order). These scatter it back to user target buffers/systems; hoisted from the
# CUDA-only file so the pure-host path can finalize without loading CUDA.

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
        throw(ArgumentError("finalize_radix_output! requires a host-resident state; device-resident states are finalized by the backend extension"))
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

"""
    run_host_radix_lifecycle!(state)

Run B2M, M2M, M2L, L2L, and L2B in place for a host-resident radix `state`.
The source bodies and routes must already be current. Returns `state`; use
[`finalize_radix_output!`](@ref) to scatter its output to user systems.
"""
function run_host_radix_lifecycle!(state::DeviceResidentRadixState)
    state.counters.expansion_host_copies == 0 ||
        throw(AssertionError("resident host radix lifecycle observed expansion host copies before execution"))
    _launch_host_b2m!(state)
    _launch_host_resident_operator_pipeline!(state)
    return state
end

