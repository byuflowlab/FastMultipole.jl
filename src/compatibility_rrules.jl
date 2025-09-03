# TrackedArrays do not support in-place assignment. It's technically possible to just write a new version of setindex! that works,
#    but that would be really inefficient - every assignment has an associated jvp that is one-hot.
#    So assigning all elements of an array elementwise would be as inefficient as storing the Jacobian of assigning all elements of the array at once.
#    The memory cost scales with the size (number of elements) of the original array squared, which is really bad.
#    And the Jacobian/jvp entries would also just be the identity matrix.
#    Long story short, it's better to write explicit pullbacks for array assignments instead of a general rule.

function target_to_buffer!(buffer::ReverseDiff.TrackedArray, system, sort_index=1:get_n_bodies(system))
    buffer_val_star = deepcopy(buffer.value)
    tp = ReverseDiff.tape(system)
    for i_body in 1:get_n_bodies(system)
        for j=1:3
            buffer.value[j, i_body] = get_position(system, sort_index[i_body])[j].value
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
    ReverseDiff.value!(buffer, buffer_val_star)
    #zeroR = zero(eltype(buffer.deriv))
    for i_body in 1:get_n_bodies(system)
        # ReverseDiff._add_to_deriv!.(get_position(system, sort_index[i_body]), buffer.value[1:3, i_body])
        get_position_pullback!(system, sort_index[i_body], buffer[1:3, i_body])
    end
    return nothing

end

@noinline function ReverseDiff.special_forward_exec!(instruction::ReverseDiff.SpecialInstruction{typeof(target_to_buffer!)})

    error("TODO: implement forward pass")

end

function source_system_to_buffer_pullback!(buffer, i_body, system, sort_index_i_body)
    throw("source_system_to_buffer_pullback! not overloaded for type $(typeof(system))")
end

function get_position_pullback!(system, i, buffer)
    throw("get_position_pullback! not overloaded for type $(typeof(system))")
end

function system_to_buffer!(buffer::ReverseDiff.TrackedArray, system, sort_index=1:get_n_bodies(system))
    for i_body in 1:get_n_bodies(system)
        source_system_to_buffer!(buffer, i_body, system, sort_index[i_body])
    end
end

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