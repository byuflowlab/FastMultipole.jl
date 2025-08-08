#--- create interaction lists ---#

function build_interaction_lists(target_branches, source_branches, source_leaf_size, multipole_acceptance, farfield, nearfield, self_induced, method::InteractionListMethod=SelfTuning())

    # prepare containers
    m2l_list = Vector{SVector{2,Int32}}(undef,0)
    direct_list = Vector{SVector{2,Int32}}(undef,0)

    # populate lists
    if length(target_branches) > 0 && length(source_branches) > 0
        build_interaction_lists!(m2l_list, direct_list, Int32(1), Int32(1), target_branches, source_branches, source_leaf_size, multipole_acceptance, Val(farfield), Val(nearfield), Val(self_induced), method)
    end

    return m2l_list, direct_list
end

mean(x) = sum(x) / length(x)

function build_interaction_lists!(m2l_list, direct_list, i_target, j_source, target_branches, source_branches, source_leaf_size, multipole_acceptance, farfield::Val{ff}, nearfield::Val{nf}, self_induced::Val{si}, method::Barba) where {ff,nf,si}
    # unpack
    source_branch = source_branches[j_source]
    target_branch = target_branches[i_target]

    # branch center separation distance
    Δx, Δy, Δz = target_branch.center - source_branch.center
    separation_distance_squared = Δx*Δx + Δy*Δy + Δz*Δz

    # decide whether or not to accept the multipole expansion
    summed_radii = source_branch.radius + target_branch.radius
    # summed_radii = sqrt(3) * mean(source_branch.box) + sqrt(3) * mean(target_branch.box)

    if separation_distance_squared * multipole_acceptance * multipole_acceptance > summed_radii * summed_radii
    #if ρ_max <= multipole_acceptance * r_min && r_max <= multipole_acceptance * ρ_min # exploring a new criterion
        if ff
            push!(m2l_list, SVector{2}(i_target, j_source))
        end
        return nothing
    end

    # both are leaves, so direct!
    if source_branch.n_branches == target_branch.n_branches == 0
        (nf || (i_target==j_source && si)) && (i_target!=j_source || si) && push!(direct_list, SVector{2}(i_target, j_source))
        return nothing
    end

    # source is a leaf OR target is not a leaf and is bigger or the same size, so subdivide target branch
    if source_branch.n_branches == 0 || (target_branch.radius >= source_branch.radius && target_branch.n_branches != 0)

        for i_child in target_branch.branch_index
            build_interaction_lists!(m2l_list, direct_list, i_child, j_source, target_branches, source_branches, source_leaf_size, multipole_acceptance, farfield, nearfield, self_induced, method)
        end

    # source is not a leaf AND target is a leaf or is smaller, so subdivide source branch
    else
        for j_child in source_branch.branch_index
            build_interaction_lists!(m2l_list, direct_list, i_target, j_child, target_branches, source_branches, source_leaf_size, multipole_acceptance, farfield, nearfield, self_induced, method)
        end
    end

end

function build_interaction_lists!(m2l_list, direct_list, i_target, j_source, target_branches, source_branches, source_leaf_size, multipole_acceptance, farfield::Val{ff}, nearfield::Val{nf}, self_induced::Val{si}, method::SelfTuning) where {ff,nf,si}
    # unpack
    source_branch = source_branches[j_source]
    target_branch = target_branches[i_target]

    # predict if direct cost is less than multipole cost
    fraction = 0.0
    for i_sys in 1:length(source_branch.bodies_index)
        fraction += source_branch.n_bodies[i_sys] / (source_leaf_size[i_sys] * source_leaf_size[i_sys])
    end
    n_targets = sum(target_branch.n_bodies)
    fraction *= n_targets

    # cost of direct is less than M2L, or both branches are leaves, so no need to continue subdividing
    if fraction < 1.0 || source_branch.n_branches == target_branch.n_branches == 0
        nf && (i_target!=j_source || si) && push!(direct_list, SVector{2}(i_target, j_source))
        return nothing
    end

    # branch center separation distance
    Δx, Δy, Δz = target_branch.center - source_branch.center
    separation_distance_squared = Δx*Δx + Δy*Δy + Δz*Δz

    # decide whether or not to accept the multipole expansion
    summed_radii = source_branch.radius + target_branch.radius

    # distance is greater than multipole threshold, perform M2L
    if separation_distance_squared * multipole_acceptance * multipole_acceptance > summed_radii * summed_radii
    #if ρ_max <= multipole_acceptance * r_min && r_max <= multipole_acceptance * ρ_min # exploring a new criterion
        if ff
            push!(m2l_list, SVector{2}(i_target, j_source))
        end
        return nothing
    end

    # count number of sources
    n_sources = sum(source_branch.n_bodies)

    # too close for M2L, and source is a leaf OR target is not a leaf and is bigger or the same size, so subdivide targets
    if source_branch.n_branches == 0 || (n_targets >= n_sources && target_branch.n_branches != 0)
    # if source_branch.n_branches == 0 || (target_branch.radius >= source_branch.radius && target_branch.n_branches != 0)

        for i_child in target_branch.branch_index
            build_interaction_lists!(m2l_list, direct_list, i_child, j_source, target_branches, source_branches, source_leaf_size, multipole_acceptance, farfield, nearfield, self_induced, method)
        end

    # source is not a leaf AND target is a leaf or is smaller, so subdivide source
    else

        for j_child in source_branch.branch_index
            build_interaction_lists!(m2l_list, direct_list, i_target, j_child, target_branches, source_branches, source_leaf_size, multipole_acceptance, farfield, nearfield, self_induced, method)
        end

    end
end

function build_interaction_lists!(m2l_list, direct_list, i_target, j_source, target_branches, source_branches, source_leaf_size, multipole_acceptance, farfield::Val{ff}, nearfield::Val{nf}, self_induced::Val{si}, method::SelfTuningTreeStop) where {ff,nf,si}
    # unpack
    source_branch = source_branches[j_source]
    target_branch = target_branches[i_target]

    # both are leaves, so direct!
    if source_branch.n_branches == target_branch.n_branches == 0
        (nf || (i_target==j_source && si)) && (i_target!=j_source || si) && push!(direct_list, SVector{2}(i_target, j_source))
        return nothing
    end

    # branch center separation distance
    Δx, Δy, Δz = target_branch.center - source_branch.center
    separation_distance_squared = Δx*Δx + Δy*Δy + Δz*Δz

    # decide whether or not to accept the multipole expansion
    summed_radii = source_branch.radius + target_branch.radius
    # summed_radii = sqrt(3) * mean(source_branch.box) + sqrt(3) * mean(target_branch.box)

    if separation_distance_squared * multipole_acceptance * multipole_acceptance > summed_radii * summed_radii
    #if ρ_max <= multipole_acceptance * r_min && r_max <= multipole_acceptance * ρ_min # exploring a new criterion
        if ff
            push!(m2l_list, SVector{2}(i_target, j_source))
        end
        return nothing
    end

    # count number of sources
    n_targets = sum(target_branch.n_bodies)
    n_sources = sum(source_branch.n_bodies)

    # source is a leaf OR target is not a leaf and has more bodies, so subdivide target branch
    if source_branch.n_branches == 0 || (n_targets >= n_sources && target_branch.n_branches != 0)

        for i_child in target_branch.branch_index
            build_interaction_lists!(m2l_list, direct_list, i_child, j_source, target_branches, source_branches, source_leaf_size, multipole_acceptance, farfield, nearfield, self_induced, method)
        end

    # source is not a leaf AND target is a leaf or has fewer bodies, so subdivide source branch
    else
        for j_child in source_branch.branch_index
            build_interaction_lists!(m2l_list, direct_list, i_target, j_child, target_branches, source_branches, source_leaf_size, multipole_acceptance, farfield, nearfield, self_induced, method)
        end
    end
end

@inline preallocate_bodies_index(T::Type{<:Branch{<:Any,NT}}, n) where NT = Tuple(Vector{UnitRange{Int64}}(undef, n) for _ in 1:NT)
# @inline preallocate_bodies_index(T::Type{<:SingleBranch}, n) = Vector{UnitRange{Int64}}(undef, n)

# function sort_by(list, target_branches, source_branches, ::InteractionListMethod{SortBySource()})
#     return sort_by_source(list, source_branches)
# end

# function sort_by(list, target_branches, source_branches, ::InteractionListMethod{SortByTarget()})
#     return sort_by_target(list, target_branches)
# end

# function sort_by(list, target_branches, source_branches, ::InteractionListMethod{nothing})
#     return list
# end

function sort_by_target(list, target_branches::Vector{<:Branch})
    # count cardinality of each target leaf in list
    target_counter = zeros(Int32, 2, length(target_branches))
    for (i,j) in list
        target_counter[1,i] += 1
    end

    # cumsum cardinality to obtain an index map
    target_counter[2,1] = Int32(1)
    for i in 2:size(target_counter,2)
        target_counter[2,i] = target_counter[2,i-1] + target_counter[1,i-1]
    end

    # preallocate sorted list
    sorted_list = similar(list)

    # sort list by source
    for ij in list
        # get source branch index
        i = ij[1]

        # get and update target destination index for this branch
        i_dest = target_counter[2,i]
        target_counter[2,i] += Int32(1)

        # place target-source pair in the sorted list
        sorted_list[i_dest] = ij
    end

    return sorted_list
end

function sort_by_source(list, source_branches::Vector{<:Branch})
    # count cardinality of each source leaf in list
    source_counter = zeros(Int32, 2, length(source_branches))
    for (i,j) in list
        source_counter[1,j] += Int32(1)
    end

    # cumsum cardinality to obtain an index map
    source_counter[2,1] = Int32(1)
    for i in 2:size(source_counter,2)
        source_counter[2,i] = source_counter[2,i-1] + source_counter[1,i-1]
    end

    # preallocate sorted list
    sorted_list = similar(list)

    # sort list by source
    for ij in list
        # get source branch index
        j = ij[2]

        # get and update target destination index for this branch
        i_dest = source_counter[2,j]
        source_counter[2,j] += Int32(1)

        # place target-source pair in the sorted list
        sorted_list[i_dest] = ij
    end

    return sorted_list
end

@inline function update_direct_bodies!(direct_bodies::Vector{<:UnitRange}, leaf_index, bodies_index::UnitRange)
    direct_bodies[leaf_index] = bodies_index
end

@inline function update_direct_bodies!(direct_bodies_list, leaf_index, bodies_indices::AbstractVector{<:UnitRange})
    for (direct_bodies, bodies_index) in zip(direct_bodies_list, bodies_indices)
        update_direct_bodies!(direct_bodies, leaf_index, bodies_index)
    end
end

function InteractionList(direct_list, target_systems, target_tree::Tree, source_systems, source_tree::Tree{TF}, derivatives_switches) where TF
    # unpack tree
    leaf_index = source_tree.leaf_index

    # preallocate containers
    influence_matrices = Vector{Matrix{TF}}(undef, length(leaf_index))

    # determine strength dimensions
    d = strength_dims(source_systems)

    # add influence matrices
    for (i_matrix,i_source_branch) in enumerate(leaf_index)
        add_influence_matrix!(influence_matrices, i_matrix, target_systems, target_tree.branches, source_systems, source_tree.branches, i_source_branch, d, direct_list, derivatives_switches)
    end

    # create largest needed storage strength and influence vectors
    n_cols_max = 0
    n_rows_max = 0
    for influence_matrix in influence_matrices
        n_rows, n_cols = size(influence_matrix)
        n_cols_max = max(n_cols_max, n_cols)
        n_rows_max = max(n_rows_max, n_rows)
    end
    strengths = zeros(TF,n_cols_max)
    influence = zeros(TF,n_rows_max)

    return InteractionList{TF}(influence_matrices, strengths, influence, direct_list)
end

