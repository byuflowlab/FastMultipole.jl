"""

@inline function max_xyz(x_min, x_max, y_min, y_max, z_min, z_max, x, y, z)
    x_min = min(x_min, x)
    x_max = max(x_max, x)
    y_min = min(y_min, y)
    y_max = max(y_max, y)
    z_min = min(z_min, z)
    z_max = max(z_max, z)

    return x_min, x_max, y_min, y_max, z_min, z_max
end

"""

# This pullback is technically wrong - min/max are not differentiable when the inputs have the same value.
# However, a special case occurs in the FMM where the inputs are the same object. In that case, we want to
#    just pass the incoming cotangent along to one (but not both) of the actually-identical inputs anway.
# The default ReverseDiff behavior, unfortunately, is to silently break the AD chain when calling min/max
#    at non-differentiable poi

function ChainRulesCore.rrule(::typeof(max_xyz), x_min, x_max, y_min, y_max, z_min, z_max, x, y, z)
    function max_xyz_pullback(xxyyzz_out_bar)
        xmin_out_bar, xmax_out_bar, ymin_out_bar, ymax_out_bar, zmin_out_bar, zmax_out_bar = xxyyzz_out_bar # unpack incoming cotangent
        
        xbar = zero(x)
        xminbar = zero(x)
        xmaxbar = zero(x)
        ybar = zero(y)
        yminbar = zero(y)
        ymaxbar = zero(y)
        zbar = zero(z)
        zminbar = zero(z)
        zmaxbar = zero(z)

        x_min < x ? xminbar = xmin_out_bar : xbar += xmin_out_bar # the plusequals addresses the special case of x == xmin == xmax
        x_max > x ? xmaxbar = xmax_out_bar : xbar += xmax_out_bar
        
        y_min < y ? yminbar = ymin_out_bar : ybar += ymin_out_bar
        y_max > y ? ymaxbar = ymax_out_bar : ybar += ymax_out_bar

        z_min < z ? zminbar = zmin_out_bar : zbar += zmin_out_bar
        z_max > z ? zmaxbar = zmax_out_bar : zbar += zmax_out_bar

        return NoTangent(), xminbar, xmaxbar, yminbar, ymaxbar, zminbar, zmaxbar, xbar, ybar, zbar

    end
    return max_xyz(x_min, x_max, y_min, y_max, z_min, z_max, x, y, z), max_xyz_pullback

end


function sort_bodies!(buffer::Matrix{<:ReverseDiff.TrackedReal}, small_buffer::Matrix{<:ReverseDiff.TrackedReal}, sort_index, octant_indices::AbstractVector, sort_index_buffer, bodies_index::UnitRange, center, target::Bool)

    # sort indices
    for i_body in bodies_index
        # identify octant
        i_octant = get_octant(get_position(buffer, i_body), center)
        this_i = octant_indices[i_octant]

        # update small buffer
        small_buffer[1,this_i].value = buffer[1, i_body].value
        small_buffer[2,this_i].value = buffer[2, i_body].value
        small_buffer[3,this_i].value = buffer[3, i_body].value
        if target
            small_buffer[4,this_i].value = buffer[17, i_body].value # copy influence from the buffer to the small buffer
            small_buffer[5,this_i].value = buffer[18, i_body].value # copy influence from the buffer to the small buffer
        end
        # tmp = system[i_body, Body()]
        # buffer[this_i] = tmp

        # update sort index
        sort_index_buffer[octant_indices[i_octant]] = sort_index[i_body]

        # increment octant census
        octant_indices[i_octant] += 1
    end

    # place buffers
    for i_body in bodies_index
        buffer[1, i_body].value = small_buffer[1, i_body].value
        buffer[2, i_body].value = small_buffer[2, i_body].value
        buffer[3, i_body].value = small_buffer[3, i_body].value
    end
    if target
        for i_body in bodies_index
            buffer[17, i_body].value = small_buffer[4, i_body].value
            buffer[18, i_body].value = small_buffer[5, i_body].value
        end
    end

    for i in bodies_index
        sort_index[i] = sort_index_buffer[i]
    end

    tp = ReverseDiff.tape(buffer)
    ReverseDiff.record!(tp,
                        ReverseDiff.SpecialInstruction,
                        sort_bodies!,
                        (buffer, small_buffer, sort_index, octant_indices, sort_index_buffer, bodies_index, center, target),
                        nothing)

end

function ReverseDiff.special_reverse_exec!(instruction::ReverseDiff.SpecialInstruction{typeof(sort_bodies!)})

    buffer, small_buffer, sort_index, octant_indices, sort_index_buffer, bodies_index, center, target = instruction.input

    # We need to unsort everything. Luckily this is pretty easy - we just reverse all the assignments.
    # We also need to map derivatives in the inverse sorting order.

    for i_body in bodies_index
        for j = 1:3
            small_buffer[j, i_body].value = buffer[j, i_body].value
            small_buffer[j, i_body].deriv = buffer[j, i_body].deriv
        end
    end
    if target
        for i_body in bodies_index
            for j=1:2
                small_buffer[3+j, i_body].value = buffer[16+j, i_body].value
                small_buffer[3+j, i_body].deriv = buffer[16+j, i_body].deriv
            end
        end
    end

    for i in bodies_index
        sort_index_buffer[i] = sort_index[i]
    end

    for i_body in bodies_index

        # decrement octant census back to where it was
        i_octant = get_octant(get_position(buffer, i_body), center)
        octant_indices[i_octant] -= 1
        this_i = octant_indices[i_octant]

        # update small buffer
        for j=1:3
            buffer[j, i_body].value = small_buffer[j,this_i].value
            buffer[j, i_body].deriv = small_buffer[j,this_i].deriv
        end
        if target
            for j=1:2
                buffer[16+j, i_body].value = small_buffer[3+j,this_i].value
                buffer[16+j, i_body].deriv = small_buffer[3+j,this_i].deriv
            end
        end
        # tmp = system[i_body, Body()]
        # buffer[this_i] = tmp

        # update sort index
        sort_index[i_body] = sort_index_buffer[octant_indices[i_octant]]

    end

    return nothing
    
end