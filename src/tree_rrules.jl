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

#@grad_from_chainrules_multiple_returns max_xyz(x_min::ReverseDiff.TrackedReal, x_max::ReverseDiff.TrackedReal, y_min::ReverseDiff.TrackedReal, y_max::ReverseDiff.TrackedReal, z_min::ReverseDiff.TrackedReal, z_max::ReverseDiff.TrackedReal, x::ReverseDiff.TrackedReal, y::ReverseDiff.TrackedReal, z::ReverseDiff.TrackedReal)
#ReverseDiff.@grad_from_chainrules max_xyz(x_min::ReverseDiff.TrackedReal, x_max::ReverseDiff.TrackedReal, y_min::ReverseDiff.TrackedReal, y_max::ReverseDiff.TrackedReal, z_min::ReverseDiff.TrackedReal, z_max::ReverseDiff.TrackedReal, x::ReverseDiff.TrackedReal, y::ReverseDiff.TrackedReal, z::ReverseDiff.TrackedReal)

#=
function Branch(n_bodies::SVector{<:Any,Int64}, bodies_index, n_branches, branch_index, i_parent::Int, i_leaf_index, source_center::AbstractArray{<:ReverseDiff.TrackedReal}, target_center::AbstractArray{<:ReverseDiff.TrackedReal}, source_radius::ReverseDiff.TrackedReal, target_radius::ReverseDiff.TrackedReal, source_box::AbstractArray{<:ReverseDiff.TrackedReal}, target_box::AbstractArray{<:ReverseDiff.TrackedReal})

    TF = eltype(source_center)
    for I in (target_center, source_radius, target_radius, source_box, target_box)
        TF = promote_type(TF, eltype(I))
    end
    return Branch(n_bodies, bodies_index, n_branches, branch_index, i_parent, i_leaf_index, TF.(source_center), TF.(target_center), TF(source_radius), TF(target_radius), TF.(source_box), TF.(target_box), ReentrantLock(), zero(TF))
end

"""
struct Tree{TF,N}
    branches::Vector{Branch{TF,N}}        # a vector of `Branch` objects composing the tree
    expansions::Array{TF,4}
    levels_index::Vector{UnitRange{Int64}}
    leaf_index::Vector{Int}
    sort_index_list::NTuple{N,Vector{Int}}
    inverse_sort_index_list::NTuple{N,Vector{Int}}
    buffers::NTuple{N,Matrix{TF}}
    small_buffers::Vector{Matrix{TF}}
    expansion_order::Int64
    leaf_size::SVector{N,Int64}    # max number of bodies in a leaf
    # cost_parameters::MultiCostParameters{N}
    # cost_parameters::SVector{N,Float64}
end
"""

function Tree(branches::Vector{Branch{<:ReverseDiff.TrackedReal, N}}, expansions::Array{<:ReverseDiff.TrackedReal, 4}, levels_index, leaf_index, sort_index_list::NTuple{N,Vector{Int}}, inverse_sort_index_list::NTuple{N,Vector{Int}}, buffers::NTuple{N, Matrix{<:ReverseDiff.TrackedReal}}, small_buffers::Vector{Matrix{<:ReverseDiff.TrackedReal}}, expansion_order, leaf_size::SVector{N,Int64}) where {N}
    TF = promote_type(eltype(branches), eltype(expansions), eltype(buffers), eltype(small_buffers))
    return Tree{TF, N}(TF.(branches), TF.(expansions), levels_index, leaf_index, sort_index_list, inverse_sort_index_list, TF.(buffers), TF.(small_buffers), expansion_order, leaf_size)
end
=#