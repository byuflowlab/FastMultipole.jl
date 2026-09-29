#------- backend-agnostic device source-buffer plumbing -------#
#
# `source_to_buffer!` dispatch, a `copyto!`, and a residency query, all generic
# over the array type, so they live in the base package: the KA extension's
# `_radix_cache_device_step!` and the generic `_recenter_union_bounds`
# (radix_cache.jl) both call them.

function _has_device_source_to_buffer_method(device_buffer, system, sort_index)
    sig = Tuple{typeof(device_buffer),typeof(system),typeof(sort_index)}
    return hasmethod(source_to_buffer!, sig)
end

# identity permutation: a range, matching the documented `sort_index` default in
# compatibility.jl, so the per-step fill allocates no index vector.
function _fill_device_source_buffer!(device_buffer, system)
    sort_index = Base.OneTo(get_n_bodies(system))
    _has_device_source_to_buffer_method(device_buffer, system, sort_index) ||
        throw(ArgumentError(
            "DeviceResident source systems must overload FastMultipole.source_to_buffer!(device_buffer, system, sort_index)",
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
# transfer, no allocation).
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

# Device-resident construction/step; provided by the registered backend
# extension (register_radix_device_backend!).
function _radix_cache_device_build(args...; kwargs...)
    hook = _RADIX_DEVICE_BUILD_HOOK[]
    hook === nothing && throw(RadixDeviceUnavailable(radix_device_status()))
    return hook(args...; kwargs...)
end

function _radix_cache_device_step!(cache::RadixFMMCache, targets::Tuple, switches::Tuple;
        nearfield_pass=nothing, extra_targets::Tuple=(),
        extra_target_switches::Tuple=(), extra_sources::Tuple=(),
        extra_tree_sources::Tuple=(), self_induce::Bool=true)
    hook = _RADIX_DEVICE_STEP_HOOK[]
    hook === nothing && throw(RadixDeviceUnavailable(radix_device_status()))
    return hook(cache, targets, switches; nearfield_pass, extra_targets,
        extra_target_switches, extra_sources, extra_tree_sources, self_induce)
end
