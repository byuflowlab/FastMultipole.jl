"""
    direct!(systems; derivatives_switches)

Applies all interactions of `systems` acting on itself without multipole acceleration.

# Arguments

- `systems`: either

    - a system object for which compatibility functions have been overloaded, or
    - a tuple of system objects for which compatibility functions have been overloaded

# Optional Arguments

- `scalar_potential::Bool`: either a `::Bool` or a `::AbstractVector{Bool}` of length `length(target_systems)` indicating whether each system should receive a scalar potential from `source_systems`
- `velocity::Bool`: either a `::Bool` or a `::AbstractVector{Bool}` of length `length(target_systems)` indicating whether each system should receive a velocity from `source_systems`
- `velocity_gradient::Bool`: either a `::Bool` or a `::AbstractVector{Bool}` of length `length(target_systems)` indicating whether each system should receive a velocity gradient from `source_systems`

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
        return direct!(target_system, source_system; args...)
    end
end


function direct!(target_systems::Tuple, source_systems::Tuple; target_buffers=nothing, source_buffers=nothing, scalar_potential=fill(true, length(target_systems)), velocity=fill(true, length(target_systems)), velocity_gradient=fill(true, length(target_systems)))

    # set up target buffers
    if isnothing(target_buffers)
        target_buffers = allocate_buffers(target_systems, true)
        target_to_buffer!(target_buffers, target_systems)
    end
    # set up source buffers
    if isnothing(source_buffers)
        source_buffers = allocate_buffers(source_systems, false)
        system_to_buffer!(source_buffers, source_systems)
    end

    # ensure derivative switch information is a vector
    scalar_potential = to_vector(scalar_potential, length(target_systems))
    velocity = to_vector(velocity, length(target_systems))
    velocity_gradient = to_vector(velocity_gradient, length(target_systems))
    derivatives_switches = DerivativesSwitch(scalar_potential, velocity, velocity_gradient)

    #check_derivs(target_systems[1].particles[4:6, :]; label="after passing derivatives back")
    check_derivs(target_buffers[1][1:3, :]; label="after passing derivatives back")
    for (source_system, source_buffer) in zip(source_systems, source_buffers)
        for (target_system, target_buffer, derivatives_switch) in zip(target_systems, target_buffers, derivatives_switches)
            direct!(target_buffer, 1:get_n_bodies(target_system), derivatives_switch, source_system, source_buffer, 1:get_n_bodies(source_system))
        end
    end
    #check_derivs(target_systems[1].particles[4:6, :]; label="before passing derivatives back")
    check_derivs(target_buffers[1][1:3, :]; label="before passing derivatives back")

    # update target systems
    buffer_to_target!(target_systems, target_buffers, derivatives_switches)

end

function direct_multithread!(target_systems::Tuple, source_systems::Tuple, n_threads; target_buffers=nothing, source_buffers=nothing, scalar_potential=fill(true, length(target_systems)), velocity=fill(true, length(target_systems)), velocity_gradient=fill(true, length(target_systems)))
    # set up target buffers
    if isnothing(target_buffers)
        target_buffers = allocate_buffers(target_systems, true)
        target_to_buffer!(target_buffers, target_systems)
    end

    # set up source buffers
    if isnothing(source_buffers)
        source_buffers = allocate_buffers(source_systems, false)
        system_to_buffer!(source_buffers, source_systems)
    end

    # ensure derivative switch information is a vector
    scalar_potential = to_vector(scalar_potential, length(target_systems))
    velocity = to_vector(velocity, length(target_systems))
    velocity_gradient = to_vector(velocity_gradient, length(target_systems))
    derivatives_switches = DerivativesSwitch(scalar_potential, velocity, velocity_gradient)

    for (source_system, source_buffer) in zip(source_systems, source_buffers)
        for (target_system, target_buffer, derivatives_switch) in zip(target_systems, target_buffers, derivatives_switches)
            
            # load balance
            n_target_bodies = get_n_bodies(target_system)
            n_per_thread, rem = divrem(n_target_bodies, n_threads)
            rem > 0 && (n_per_thread += 1)
            n_per_thread = max(n_per_thread, MIN_NPT_NF)
            Threads.@threads for i_start in 1:n_per_thread:n_target_bodies
                direct!(target_buffer, i_start:min(i_start+n_per_thread-1, n_source_bodies), derivatives_switch, source_system, source_buffer, 1:get_n_bodies(source_system))
            end
        end
    end

    # update target systems
    buffer_to_target!(target_systems, target_buffers, derivatives_switches)

end

