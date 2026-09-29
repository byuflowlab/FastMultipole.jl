# TrackedArrays do not support in-place assignment. It's technically possible to just write a new version of setindex! that works,
#    but that would be really inefficient - every assignment has an associated jvp that is one-hot.
#    So assigning all elements of an array elementwise would be as inefficient as storing the Jacobian of assigning all elements of the array at once.
#    The memory cost scales with the size (number of elements) of the original array squared, which is really bad.
#    And the Jacobian/jvp entries would also just be the identity matrix.
#    Long story short, it's better to write explicit pullbacks for array assignments instead of a general rule.

#=function target_to_buffer!(buffer::Matrix{<:ReverseDiff.TrackedReal}, system, sort_index=1:get_n_bodies(system))
    error("needs revisions if ever called")
    buffer_val_star = ReverseDiff.value.(buffer)
    tp = ReverseDiff.tape(system)
    for i_body in 1:get_n_bodies(system)
        for j=1:3
            buffer[j, i_body].value = get_position(system, sort_index[i_body])[j].value
        end
        #buffer.value[1:3, i_body] .= ReverseDiff.value.(get_position(system, sort_index[i_body]))
    end

    ReverseDiff.record!(tp,
                        ReverseDiff.SpecialInstruction,
                        target_to_buffer!,
                        (buffer, system, sort_index),
                        nothing,
                        buffer_val_star)
    return nothing

end

@noinline function ReverseDiff.special_reverse_exec!(instruction::ReverseDiff.SpecialInstruction{typeof(target_to_buffer!)})

    buffer, system, sort_index = instruction.input
    buffer_val_star = instruction.cache
    ReverseDiff.value!.(buffer, buffer_val_star)
    #zeroR = zero(eltype(buffer.deriv))
    for i_body in 1:get_n_bodies(system)
        # ReverseDiff._add_to_deriv!.(get_position(system, sort_index[i_body]), buffer.value[1:3, i_body])
        get_position_pullback!(system, sort_index[i_body], buffer[1:3, i_body])
    end
    return nothing

end

@noinline function ReverseDiff.special_forward_exec!(instruction::ReverseDiff.SpecialInstruction{typeof(target_to_buffer!)})

    buffer, system, sort_index = instruction.input
    instruction.cache = ReverseDiff.value.(buffer)
    for i_body in 1:get_n_bodies(system)
        for j=1:3
            buffer[j, i_body].value = get_position(system, sort_index[i_body])[j].value
        end
    end
    return nothing
    
end=#

function source_system_to_buffer_pullback!(buffer, i_body, system, sort_index_i_body)
    throw("source_system_to_buffer_pullback! not overloaded for type $(typeof(system))")
end

function get_position_pullback!(system, i, buffer)
    throw("get_position_pullback! not overloaded for type $(typeof(system))")
end

function get_previous_influence_pullback!(system, i, buffer)
    throw("get_previous_influence_pullback! not overloaded for type $(typeof(system))")
end

#=
function system_to_buffer!(buffer::ReverseDiff.TrackedArray, system, sort_index=1:get_n_bodies(system))
    for i_body in 1:get_n_bodies(system)
        source_system_to_buffer!(buffer, i_body, system, sort_index[i_body])
    end
end
=#

check_derivs(x;label=nothing) = x
check_derivs_trackedarray() = error()
check_derivs_array_of_trackedreals() = error()
function check_derivs(x::ReverseDiff.TrackedArray; label=nothing)

    println("ready to check derivs of TrackedArray")
    tp = ReverseDiff.tape(x)

    ReverseDiff.record!(tp,
                        ReverseDiff.SpecialInstruction,
                        check_derivs_trackedarray,
                        (x),
                        x,
                        label)
    return x

end

function check_derivs(x::AbstractArray{<:ReverseDiff.TrackedReal}; label=nothing)

    println("ready to check derivs of array of TrackedReals")
    tp = ReverseDiff.tape(x)

    ReverseDiff.record!(tp,
                        ReverseDiff.SpecialInstruction,
                        check_derivs_array_of_trackedreals,
                        (x),
                        x,
                        label)
    return x

end

@noinline function ReverseDiff.special_reverse_exec!(instruction::ReverseDiff.SpecialInstruction{typeof(check_derivs_array_of_trackedreals)})

    if instruction.cache === nothing
        println("sum of derivatives: $(sum(ReverseDiff.deriv.(instruction.input)))")
    else
        println("sum of derivatives of $(instruction.cache): $(sum(ReverseDiff.deriv.(instruction.input)))")
    end
    return nothing

end

@noinline function ReverseDiff.special_reverse_exec!(instruction::ReverseDiff.SpecialInstruction{typeof(check_derivs_trackedarray)})

    if instruction.cache === nothing
        println("sum of derivatives: $(sum(ReverseDiff.deriv(instruction.input)))")
    else
        println("sum of derivatives of $(instruction.cache): $(sum(ReverseDiff.deriv(instruction.input)))")
    end
    return nothing

end

@noinline function ReverseDiff.special_forward_exec!(instruction::ReverseDiff.SpecialInstruction{typeof(check_derivs)})
    return nothing
end

##### function for checking that an array of tracked reals is properly allocated.

check_deriv_allocation(x; label=nothing) = x
check_deriv_allocation_trackedarray() = error() # dummy function
check_deriv_allocation_array_of_trackedreals() = error() # dummy function
function check_deriv_allocation(x::ReverseDiff.TrackedArray; label=nothing)

    ϵ = 1e-6
    tp = ReverseDiff.tape(x)
    s = sum(x.value)
    one_x_val = one(eltype(x.value))
    for xi in x.value
        xi += one_x_val
    end
    s2 = sum(x.value)
    for xi in x.value
        xi -= one_x_val
    end
    s3 = sum(x.value)
    if abs(s-s3) > ϵ ; error("Initial sum of values $s is not equal to final sum of values $(s3)!"); end
    if abs(s2-s - length(x.value)) > ϵ; error("Perturbation check failed! Initial sum of values is $s, final sum is $s2, and the length of the array is $(length(x)). Difference: $(s2 - length(x.value))"); end
    label === nothing ? println("value of TrackedArray is properly allocated!") : println("value of TrackedArray $label is properly allocated!")

    tp = ReverseDiff.tape(x)

    if length(tp) == 0
        label === nothing ? error("tape has length zero!") : error("tape of $label has length zero!")
    end

    ReverseDiff.record!(tp,
                        ReverseDiff.SpecialInstruction,
                        check_deriv_allocation_trackedarray,
                        (x),
                        x,
                        label)
    return x

end

function check_deriv_allocation(x::AbstractArray{<:ReverseDiff.TrackedReal}; label=nothing)

    ϵ = 1e-6
    tp = ReverseDiff.tape(x)
    s = sum(ReverseDiff.value.(x))
    one_x_val = one(eltype(x[1].value))
    for xi in x
        xi.value += one_x_val
    end
    s2 = sum(ReverseDiff.value.(x))
    for xi in x
        xi.value -= one_x_val
    end
    s3 = sum(ReverseDiff.value.(x))
    if abs(s-s3 > ϵ); error("Initial sum of values $s is not equal to final sum of values $(s3)!"); end
    if abs(s2-s - length(x)) > ϵ ; error("Perturbation check failed! Initial sum of values is $s, final sum is $s2, and the length of the array is $(length(x)). Difference: $(s2 - s - length(x))"); end
    label === nothing ? println("value of array of TrackedReals is properly allocated!") : println("value of array of TrackedReals $label is properly allocated!")

    tp = ReverseDiff.tape(x)

    if length(tp) == 0
        label === nothing ? error("tape has length zero!") : error("tape of $label has length zero!")
    end

    ReverseDiff.record!(tp,
                        ReverseDiff.SpecialInstruction,
                        check_deriv_allocation_array_of_trackedreals,
                        (x),
                        x,
                        label)
    return x

end
@noinline function ReverseDiff.special_reverse_exec!(instruction::ReverseDiff.SpecialInstruction{typeof(check_deriv_allocation_trackedarray)})

    
    return nothing

end
@noinline function ReverseDiff.special_reverse_exec!(instruction::ReverseDiff.SpecialInstruction{typeof(check_deriv_allocation_array_of_trackedreals)})

    x = instruction.input
    label = instruction.cache
    ϵ = 1e-6
    tp = ReverseDiff.tape(x)
    s = sum(ReverseDiff.deriv.(x))
    @show s
    one_x_deriv = one(eltype(x[1].deriv))
    for xi in x
        xi.deriv += one_x_deriv
    end
    s2 = sum(ReverseDiff.deriv.(x))
    for xi in x
        xi.deriv -= one_x_deriv
    end
    s3 = sum(ReverseDiff.deriv.(x))
    if abs(s-s3 > ϵ); error("Initial sum of derivs $s is not equal to final sum of derivs $(s3)!"); end
    if abs(s2-s - length(x)) > ϵ ; error("Perturbation check failed! Initial sum of derivs is $s, final sum is $s2, and the length of the array is $(length(x)). Difference: $(s2 - s)"); end
    label === nothing ? println("derivative of array of TrackedReals is properly allocated!") : println("derivative of array of TrackedReals $label is properly allocated!")

    if length(tp) == 0
        label === nothing ? error("tape has length zero!") : error("tape of $label has length zero!")
    end
    return nothing

end

@noinline function ReverseDiff.special_forward_exec!(instruction::ReverseDiff.SpecialInstruction{typeof(check_deriv_allocation_trackedarray)})
    return nothing
end

init_rd_array!(arr, tp) = error("attempted to initialize non-tracked array!")
function init_rd_array!(arr::AbstractArray{<:ReverseDiff.TrackedReal}, tp)

    for idx in CartesianIndices(arr)
        arr[idx] = ReverseDiff.track(arr[idx].value, tp)
    end
    return nothing

end

#=
function target_to_buffer!(buffer::Matrix, system, sort_index=1:get_n_bodies(system), switch=DerivativesSwitch(true, true, true, system))
    if Threads.nthreads() > 1 && get_n_bodies(system) > MIN_BODIES
        target_to_buffer_multithread!(buffer, system, sort_index, switch)
    else
        for i_body in 1:get_n_bodies(system)
            i_sorted = sort_index[i_body]
            buffer[1:3, i_body] .= get_position(system, i_sorted)
            metadata_to_buffer!(buffer, switch, i_body, system, i_sorted)
        end
    end
end
=#

function target_to_buffer!(buffer::Matrix{<:ReverseDiff.TrackedReal}, system, sort_index=1:get_n_bodies(system), switch=DerivativesSwitch(true, true, true, system))
    
    if Threads.nthreads() > 1 && get_n_bodies(system) > MIN_BODIES
        error("multithreading case not implemented for ReverseDiff yet.")
        target_to_buffer_multithread!(buffer, system, sort_index, switch)
    else
        buffer_star = zeros(ReverseDiff.valtype(eltype(buffer)), 3 + metadata_per_body(system), get_n_bodies(system))
        for i_body in 1:get_n_bodies(system)
            i_sorted = sort_index[i_body]
            # since get_position is defined outside FastMultipole, we will also require the forward-mode value-only version and the reverse-mode pullback to be defined outside FastMultipole
            position_to_buffer__value!(buffer, i_body, system, i_sorted, buffer_star)
            # same reasoning as position_to_buffer__value!
            metadata_to_buffer__value!(buffer, switch, i_body, system, i_sorted, buffer_star)
        end
    end
    tp = ReverseDiff.tape(buffer)
    ReverseDiff.record!(tp,
                        ReverseDiff.SpecialInstruction,
                        target_to_buffer2!,
                        (buffer, system, sort_index, switch),
                        nothing,
                        buffer_star)
    return nothing
end

target_to_buffer2!() = error() # dummy function to disambiguate two different target_to_buffer! calls

function ReverseDiff.special_reverse_exec!(instruction::ReverseDiff.SpecialInstruction{typeof(target_to_buffer2!)})

    buffer, system, sort_index, switch = instruction.input
    buffer_star = instruction.cache
    
    if Threads.nthreads() > 1 && get_n_bodies(system) > MIN_BODIES
        error("multithreading case not implemented for ReverseDiff yet.")
        target_to_buffer_multithread!(buffer, system, sort_index, switch)
    else
        for i_body in 1:get_n_bodies(system)
            i_sorted = sort_index[i_body]
            # we require user definitions for these
            position_to_buffer__pullback!(buffer, i_body, system, i_sorted, buffer_star)
            metadata_to_buffer__pullback!(buffer, switch, i_body, system, i_sorted, buffer_star)
        end
    end
    return nothing

end

function ReverseDiff.special_forward_exec!(instruction::ReverseDiff.SpecialInstruction{typeof(target_to_buffer2!)})

    buffer, system, sort_index, switch = instruction.input
    buffer_star = instruction.cache

    if Threads.nthreads() > 1 && get_n_bodies(system) > MIN_BODIES
        error("multithreading case not implemented for ReverseDiff yet.")
        target_to_buffer_multithread!(buffer, system, sort_index, switch)
    else
        for i_body in 1:get_n_bodies(system)
            i_sorted = sort_index[i_body]
            # since get_position is defined outside FastMultipole, we will also require the forward-mode value-only version and the reverse-mode pullback to be defined outside FastMultipole
            position_to_buffer__value!(buffer, i_body, system, i_sorted, buffer_star)
            # same reasoning as position_to_buffer__value!
            metadata_to_buffer__value!(buffer, switch, i_body, system, i_sorted, buffer_star)
        end
    end
    return nothing

end

position_to_buffer__value!(buffer, i_body, system, i_sorted, buffer_star) = error("position to buffer forward pass not defined")
metadata_to_buffer__value!(buffer, switch, i_body, system, i_sorted, buffer_star) = error("metadata to buffer forward pass not defined")

position_to_buffer__pullback!(buffer, i_body, system, i_sorted, buffer_star) = error("position to buffer reverse pass not defined")
metadata_to_buffer__pullback!(buffer, switch, i_buffer, system, i_body, buffer_star) = extra_target_data_to_buffer__pullback!(buffer, i_buffer, system, i_body, buffer_star)
extra_target_data_to_buffer__pullback!(buffer, i_body, system, i_sorted, buffer_star) = error("metadata to buffer reverse pass (or extra target data to buffer reverse pass) not defined")