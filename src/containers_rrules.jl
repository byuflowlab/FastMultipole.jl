

#=

struct Branch{TF,N}
    n_bodies::SVector{N,Int64}
    bodies_index::SVector{N,UnitRange{Int64}}
    n_branches::Int64
    branch_index::UnitRange{Int64}
    i_parent::Int64
    i_leaf::Int64
    source_center::SVector{3,TF}   # center of the branch
    target_center::SVector{3,TF}   # center of the branch
    source_radius::TF
    target_radius::TF
    source_box::SVector{3,TF} # x, y, and z half widths of the box encapsulating all sources
    target_box::SVector{3,TF} # x, y, and z half widths of the box encapsulating all sources
    # multipole_expansion::Array{TF,3} # multipole expansion coefficients
    # local_expansion::Array{TF,3}     # local expansion coefficients
    # harmonics::Array{TF,3}
    lock::ReentrantLock
    max_influence::TF
end

=#

function Branch(n_bodies::SVector{<:Any,Int64}, bodies_index, n_branches, branch_index, i_parent::Int, i_leaf_index, source_center, target_center, source_radius::ReverseDiff.TrackedReal, target_radius, source_box, target_box)
    @show typeof(source_center) typeof(target_center) typeof(source_radius) typeof(target_radius) typeof(source_box) typeof(target_box)
    return Branch(n_bodies, bodies_index, n_branches, branch_index, i_parent, i_leaf_index, source_center, target_center, source_radius, target_radius, source_box, target_box, ReentrantLock(), zero(target_radius))
end