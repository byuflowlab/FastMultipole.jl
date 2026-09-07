"""
    direct!(systems; derivatives_switches)

Applies all interactions of `systems` acting on itself without multipole acceleration.

# Arguments

- `systems`: either

    - a system object for which compatibility functions have been overloaded, or
    - a tuple of system objects for which compatibility functions have been overloaded
    - `(target_systems, source_systems)` where `target_systems` and `source_systems` are each either a system or tuple of systems for which compatibility functions have been overloaded

# Optional Arguments

- `scalar_potential::Bool`: either a `::Bool` or a `::AbstractVector{Bool}` of length `length(target_systems)` indicating whether each system should receive a scalar potential from `source_systems`
- `gradient::Bool`: either a `::Bool` or a `::AbstractVector{Bool}` of length `length(target_systems)` indicating whether each system should receive a vector field from `source_systems`
- `hessian::Bool`: either a `::Bool` or a `::AbstractVector{Bool}` of length `length(target_systems)` indicating whether each system should receive a vector gradient from `source_systems`
- `third_derivative::Bool`: either a `::Bool` or target-wise booleans requesting the
  packed second spatial derivative of that vector field
- `extra_outputs::Int`: number of extra accumulated target output rows; defaults to `0`
- `metadata::Union{Nothing,Int}`: number of target metadata rows carried with positions; `nothing` infers [`metadata_per_body`](@ref)
- `n_threads::Int`: the number of threads to use for parallelization; defaults to `Threads.nthreads()`
- `direct_conditioning`: a `DirectConditioningRule` or tuple of rules used to temporarily condition source buffers for selected source-target system pairs

"""
function direct!(systems::Tuple; args...)
    direct!(systems, systems; args...)
end

function direct!(system; args...)
    direct!((system,); args...)
end

function direct!(target_system, source_system; args...)
    target_system = to_tuple(target_system)
    source_system = to_tuple(source_system)
    _direct!(target_system, source_system; args...)
end

function _direct!(target_system, source_system; n_threads=Threads.nthreads(), args...)
    if n_threads > 1
        return direct_multithread!(target_system, source_system, n_threads; args...)
    else
        return direct_singlethread!(target_system, source_system; args...)
    end
end


function direct_singlethread!(target_systems::Tuple, source_systems::Tuple; target_buffers=nothing, source_buffers=nothing, scalar_potential=fill(false, length(target_systems)), gradient=fill(true, length(target_systems)), hessian=fill(false, length(target_systems)), third_derivative=fill(false, length(target_systems)), extra_outputs=0, metadata=nothing, direct_conditioning=(), nearfield_cache=nothing)

    # get float type
    TF = get_type(target_systems, source_systems)

    # derivatives switches
    scalar_potential = to_vector(scalar_potential, length(target_systems))
    gradient = to_vector(gradient, length(target_systems))
    hessian = to_vector(hessian, length(target_systems))
    third_derivative = to_vector(third_derivative, length(target_systems))
    derivatives_switches = DerivativesSwitch(scalar_potential, gradient, hessian, target_systems; third_derivative, extra_outputs, metadata)
    _check_third_derivative_support(target_systems, source_systems, derivatives_switches)

    # set up target buffers
    if isnothing(target_buffers)
        target_buffers = allocate_buffers(target_systems, true, TF, derivatives_switches)
        target_to_buffer!(target_buffers, target_systems, SVector{length(target_systems)}([1:get_n_bodies(system) for system in target_systems]), derivatives_switches)
    end

    # set up source buffers
    if isnothing(source_buffers)
        source_buffers = allocate_buffers(source_systems, false, TF, derivatives_switches)
        system_to_buffer!(source_buffers, source_systems)
    end

    direct_conditioning = normalize_direct_conditioning(direct_conditioning)

    if !isnothing(nearfield_cache)
        _refuse_conditioning(direct_conditioning, "standalone direct! evaluation")
        nearfield_matvec!(target_buffers, nearfield_cache, source_buffers; n_threads=1)
    elseif has_direct_conditioning(direct_conditioning)
        for (i_source_system, (source_system, source_buffer)) in enumerate(zip(source_systems, source_buffers))
            for (i_target_system, (target_system, target_buffer, derivatives_switch)) in enumerate(zip(target_systems, target_buffers, derivatives_switches))
                with_direct_conditioning!(direct_conditioning, source_buffer, source_system, i_source_system, target_buffer, i_target_system) do
                    direct!(target_buffer, 1:get_n_bodies(target_system), derivatives_switch, source_system, source_buffer, 1:get_n_bodies(source_system))
                end
            end
        end
    else
        for (source_system, source_buffer) in zip(source_systems, source_buffers)
            for (target_system, target_buffer, derivatives_switch) in zip(target_systems, target_buffers, derivatives_switches)
                direct!(target_buffer, 1:get_n_bodies(target_system), derivatives_switch, source_system, source_buffer, 1:get_n_bodies(source_system))
            end
        end
    end

    # update target systems
    buffer_to_target!(target_systems, target_buffers, derivatives_switches)

end

function direct_multithread!(target_systems::Tuple, source_systems::Tuple, n_threads; target_buffers=nothing, source_buffers=nothing, scalar_potential=fill(false, length(target_systems)), gradient=fill(true, length(target_systems)), hessian=fill(false, length(target_systems)), third_derivative=fill(false, length(target_systems)), extra_outputs=0, metadata=nothing, direct_conditioning=(), nearfield_cache=nothing)

    # get float type
    TF = get_type(target_systems, source_systems)

    # ensure derivative switch information is a vector
    scalar_potential = to_vector(scalar_potential, length(target_systems))
    gradient = to_vector(gradient, length(target_systems))
    hessian = to_vector(hessian, length(target_systems))
    third_derivative = to_vector(third_derivative, length(target_systems))
    derivatives_switches = DerivativesSwitch(scalar_potential, gradient, hessian, target_systems; third_derivative, extra_outputs, metadata)
    _check_third_derivative_support(target_systems, source_systems, derivatives_switches)

    # set up target buffers
    if isnothing(target_buffers)
        target_buffers = allocate_buffers(target_systems, true, TF, derivatives_switches)
        target_to_buffer!(target_buffers, target_systems, SVector{length(target_systems)}([1:get_n_bodies(system) for system in target_systems]), derivatives_switches)
    end

    # set up source buffers
    if isnothing(source_buffers)
        source_buffers = allocate_buffers(source_systems, false, TF, derivatives_switches)
        system_to_buffer!(source_buffers, source_systems)
    end

    direct_conditioning = normalize_direct_conditioning(direct_conditioning)

    if !isnothing(nearfield_cache)
        _refuse_conditioning(direct_conditioning, "standalone direct! evaluation")
        nearfield_matvec!(target_buffers, nearfield_cache, source_buffers; n_threads)
    elseif has_direct_conditioning(direct_conditioning)
        for (i_source_system, (source_system, source_buffer)) in enumerate(zip(source_systems, source_buffers))
            n_source_bodies = get_n_bodies(source_system)
            for (i_target_system, (target_system, target_buffer, derivatives_switch)) in enumerate(zip(target_systems, target_buffers, derivatives_switches))
                with_direct_conditioning!(direct_conditioning, source_buffer, source_system, i_source_system, target_buffer, i_target_system) do
                    direct_multithread_pair!(target_buffer, target_system, derivatives_switch, source_system, source_buffer, n_source_bodies, n_threads)
                end
            end
        end
    else
        for (source_system, source_buffer) in zip(source_systems, source_buffers)
            n_source_bodies = get_n_bodies(source_system)
            for (target_system, target_buffer, derivatives_switch) in zip(target_systems, target_buffers, derivatives_switches)
                direct_multithread_pair!(target_buffer, target_system, derivatives_switch, source_system, source_buffer, n_source_bodies, n_threads)
            end
        end
    end

    # update target systems
    buffer_to_target!(target_systems, target_buffers, derivatives_switches)

end

function direct_multithread_pair!(target_buffer, target_system, derivatives_switch, source_system, source_buffer, n_source_bodies, n_threads)
    # load balance
    n_target_bodies = get_n_bodies(target_system)
    n_per_thread, rem = divrem(n_target_bodies, n_threads)
    rem > 0 && (n_per_thread += 1)
    n_per_thread = max(n_per_thread, MIN_NPT_NF)
    Threads.@threads :static for i_start in 1:n_per_thread:n_target_bodies
        direct!(target_buffer, i_start:min(i_start+n_per_thread-1, n_target_bodies), derivatives_switch, source_system, source_buffer, 1:n_source_bodies)
    end
end
