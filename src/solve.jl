#------- matrix storage -------#

function Matrices(sizes::Vector{Tuple{Int,Int}}, TF=Float64)
    # preallocate matrix storage
    n_matrix = sum(m * n for (m, n) in sizes)
    n_rhs = sum(m for (m,_) in sizes)
    data = Vector{TF}(undef, n_matrix)
    rhs = Vector{TF}(undef, n_rhs)
    
    # offsets
    matrix_offsets = Vector{Int}(undef, length(sizes))
    rhs_offsets = Vector{Int}(undef, length(sizes))
    matrix_offset = 1
    rhs_offset = 1
    for i in 1:length(sizes)
        matrix_offsets[i] = matrix_offset
        rhs_offsets[i] = rhs_offset

        m, n = sizes[i]
        matrix_offset += m * n
        rhs_offset += m
    end

    return Matrices{TF}(data, rhs, sizes, matrix_offsets, rhs_offsets)
end

function EmptyMatrices(TF=Float64)
    # preallocate empty matrix storage
    return Matrices{TF}(zeros(TF, 0), zeros(TF, 0), Tuple{Int,Int}[], Int[], Int[])
end

@inline function get_matrix_range(ms::Matrices, k::Int, m, n)
    # get the range of values corresponding to the k-th matrix
    m, n = ms.sizes[k]
    matrix_offset = ms.matrix_offsets[k]
    return matrix_offset:matrix_offset + m*n - 1
end

@inline function get_gradient_range(ms::Matrices, k::Int, m)
    # get the range of values corresponding to the k-th rhs vector
    rhs_offset = ms.rhs_offsets[k]
    return rhs_offset:rhs_offset + m - 1
end

function get_matrix_vector(ms::Matrices, k::Int)
    m, n = ms.sizes[k]
    vrange = get_gradient_range(ms, k, m)
    mrange = get_matrix_range(ms, k, m, n)
    mat = @view ms.data[mrange]
    return reshape(mat, m, n), view(ms.rhs, vrange)
end

function set_unit_strength!(source_buffers::Tuple, source_systems::Tuple)
    for i_source_system in eachindex(source_systems)
        source_system = source_systems[i_source_system]
        source_buffer = source_buffers[i_source_system]
        
        set_unit_strength!(source_buffer, source_system)
    end
end

function set_unit_strength!(source_buffer::AbstractMatrix{TF}, source_system) where TF
    # set the source strength to unit
    n_bodies = get_n_bodies(source_system)
    unit_strength = one(TF)
    dim = strength_dims(source_system)
    for j in 1:n_bodies
        value_to_strength!(source_buffer, source_system, j, one(TF))
    end
end

function save_strengths(source_buffers::Tuple, source_systems::Tuple)
    # save the strengths of the source systems
    TF = promote_type(eltype.(source_buffers)...)
    old_strengths = Tuple(Matrix{TF}(undef, strength_dims(source_systems[k]), get_n_bodies(source_systems[k])) for k in eachindex(source_systems))
    for k in eachindex(source_systems)
        source_system = source_systems[k]
        source_buffer = source_buffers[k]
        old_strengths[k] .= view(source_buffer, 5:5 + strength_dims(source_system) - 1, :)
    end

    return old_strengths
end

function restore_strengths!(source_buffers::Tuple, source_systems::Tuple, old_strengths)
    # restore the strengths of the source systems
    for k in eachindex(source_systems)
        source_system = source_systems[k]
        source_buffer = source_buffers[k]
        source_buffer[5:5 + strength_dims(source_system) - 1, :] .= old_strengths[k]
    end
end

function nonself_influence_matrices(target_buffers::Tuple, source_buffers::Tuple, source_systems::Tuple, target_tree::Tree{TF,<:Any}, source_tree::Tree, direct_list, derivatives_switches) where TF
    
    #--- sort by source ---#

    source_branches = source_tree.branches
    target_branches = target_tree.branches
    sorted_list = sort_by_source(direct_list, source_branches)

    #--- pre-allocate influence matrices ---#
    
    if length(sorted_list) > 0
        
        # preallocate sizes
        sizes = Vector{Tuple{Int,Int}}(undef, length(source_tree.leaf_index))

        # number of non-empty matrices
        this_source = 0
        n_matrices = 0
        for (i_target,j_source) in sorted_list
            if j_source != this_source
                n_matrices += 1
                this_source = j_source
            end
        end

        # generate matrix map
        matrix_map = Vector{Int}(undef, n_matrices)
        i_leaf = 1
        i_matrix = 1
        this_source = 0
        for (_, j_source) in sorted_list

            # found a non-empty matrix
            if j_source != this_source
                # upper bound of number of leaves we've skipped
                n_missing_ub = j_source - source_tree.leaf_index[i_leaf]

                # increment i_leaf until we line up with j_source
                for i_plus in 1:n_missing_ub
                    # i_leaf is empty
                    sizes[i_leaf] = (0,0)

                    # move on to the next leaf
                    i_leaf += 1

                    # check if we've reached j_source
                    if source_tree.leaf_index[i_leaf] == j_source
                        break
                    end
                end

                # update matrix_map
                matrix_map[i_matrix] = i_leaf

                # recurse
                i_leaf += 1
                i_matrix += 1
                this_source = j_source
            end
        end

        # fill in the rest with zeros
        for i in i_leaf:length(source_tree.leaf_index)
            sizes[i] = (0,0)
        end
        
        # populate sizes
        this_leaf = source_tree.leaf_index[1]
        this_source = sorted_list[1][2]
        i_matrix = 1
        n = 0
        for (i_target, j_source) in sorted_list
            
            # just finished a matrix
            if j_source != this_source
                
                # save this size
                m = get_n_bodies(source_branches[this_source].bodies_index)
                sizes[matrix_map[i_matrix]] = (n,m)

                # increment indices for the next
                i_matrix += 1
                this_source = j_source
                n = 0
            end
            
            # accumulate n
            n += get_n_bodies(target_branches[i_target].bodies_index)
        end

        # add the final matrix
        this_source = sorted_list[end][2]
        m = get_n_bodies(source_branches[this_source].bodies_index)
        sizes[matrix_map[i_matrix]] = (n,m)

        # construct influence matrices
        matrices = Matrices(sizes, TF)

        #--- populate influence matrices ---#

        # store strengths for later
        old_strengths = save_strengths(source_buffers, source_systems)

        # set sources to unit strength
        set_unit_strength!(source_buffers, source_systems)

        # loop over direct list
        this_source = 0
        i_matrix = 0
        i_target_start = 1
        i_source_start = 1
        matrix, influence = get_matrix_vector(matrices, 1)

        for (i_target,j_source) in sorted_list
                
            # start on the next matrix
            if j_source != this_source
                i_matrix += 1
                this_source = j_source
                i_source_start = 1
                i_target_start = 1
                matrix, influence = get_matrix_vector(matrices, matrix_map[i_matrix])
            end

            # loop over source systems
            this_i_source_start = i_source_start

            for i_source_system in eachindex(source_systems)
                
                # get view of matrix corresponding to this source system
                source_index = source_branches[j_source].bodies_index[i_source_system]
                n_sources = length(source_index)

                # unpack source buffer
                source_system = source_systems[i_source_system]
                source_buffer = source_buffers[i_source_system]
                # this_source_buffer = view(source_buffer, :, source_index)

                # loop over target systems
                for i_target_system in eachindex(target_buffers)
                    
                    # get view of matrix corresponding to this source and target system
                    target_index = target_branches[i_target].bodies_index[i_target_system]
                    n_targets = length(target_index)
                    this_matrix = view(matrix, i_target_start:i_target_start + n_targets - 1, this_i_source_start:this_i_source_start + n_sources - 1)
                    this_influence = view(influence, i_target_start:i_target_start + n_targets - 1)

                    # unpack target buffer
                    target_buffer = target_buffers[i_target_system]
                    this_target_buffer = view(target_buffer, :, target_index)
                    target_source_buffer = source_buffers[i_target_system]
                    this_source_buffer = view(target_source_buffer, :, target_index)
                    derivatives_switch = derivatives_switches[i_target_system]

                    # loop over source bodies
                    for (isb,i_source_body) in enumerate(source_index)
                    
                        # reset targets
                        reset!(target_buffer, target_index)

                        # direct influence on these targets
                        direct!(target_buffer, target_index, derivatives_switch, source_system, source_buffer, i_source_body:i_source_body)

                        # compute influences
                        influence!(this_influence, this_target_buffer, source_system, this_source_buffer)

                        # update matrix
                        this_matrix[:,isb] .= this_influence

                    end

                    # update target starting index
                    i_target_start += n_targets
                end

                # update source starting index
                this_i_source_start += length(source_index)
            end
        end

        # restore old strengths
        restore_strengths!(source_buffers, source_systems, old_strengths)

        # zero rhs
        matrices.rhs .= zero(TF)

    else
        matrices = EmptyMatrices(TF)
    end

    return matrices, sorted_list
end

"""
    index_by_source(sorted_list::Vector{SVector{2,Int}}, leaf_index::Vector{Int})

Constructs an index map that maps each leaf to the range of indices in the sorted list where it acts as a source. It is possible for some leaves to never act as sources, in which case the range will be empty.
"""
function index_by_source(sorted_list::Vector{SVector{2,Int32}}, leaf_index::Vector{Int})
    
    # preallocate index map
    # index_map = Vector{UnitRange{Int}}(undef, length(leaf_index))
    index_map = fill(1:0, length(leaf_index)) # initialize with empty ranges

    # check if sorted_list is empty
    if length(sorted_list) > 0
    
        # loop over direct list
        this_source = 0
        i_start = 1
        i_leaf = 1
        this_source = sorted_list[1][2] # start with the first source
        for (i_list, (i_target, j_source)) in enumerate(sorted_list)

            # finished a non-empty matrix
            if j_source != this_source

                # upper bound of number of leaves we've skipped
                n_missing_ub = this_source - leaf_index[i_leaf]

                # increment i_leaf until we line up with j_source
                for i_plus in 1:n_missing_ub

                    # no interactions for this leaf
                    # index_map[i_leaf] = i_start:i_start - 1

                    # move on to the next leaf
                    i_leaf += 1

                    # check if we've reached j_source
                    if leaf_index[i_leaf] == this_source
                        break
                    end
                end

                # store leaf index range
                index_map[i_leaf] = i_start:i_list-1

                # recurse
                i_leaf += 1
                this_source = j_source
                i_start = i_list
            end
        end

        #--- last index ---#

        # upper bound of number of leaves we've skipped
        n_missing_ub = this_source - leaf_index[i_leaf]

        # increment i_leaf until we line up with j_source
        for i_plus in 1:n_missing_ub

            # no interactions for this leaf
            # index_map[i_leaf] = i_start:i_start - 1

            # move on to the next leaf
            i_leaf += 1

            # check if we've reached j_source
            if leaf_index[i_leaf] == this_source
                break
            end
        end

        # store leaf index range
        index_map[i_leaf] = i_start:length(sorted_list)
        i_start = length(sorted_list) + 1

        #--- all remaining leaves do not act as sources ---#

        # for i in i_leaf + 1:length(leaf_index)
        #     index_map[i] = i_start:i_start - 1
        # end
    end

    return index_map
end

"""
    self_influence_matrices(target_buffers, source_buffers, source_systems, target_tree, source_tree, derivatives_switches)

Constructs influence matrices for all leaves of the tree. (Assumes source tree and target trees are identical.)
"""
function self_influence_matrices(target_buffers, source_buffers, source_systems, target_tree::Tree{TF,<:Any}, source_tree, derivatives_switches) where TF
    
    #--- pre-allocate influence matrices ---#

    # get sizes
    sizes = Vector{Tuple{Int,Int}}(undef,length(source_tree.leaf_index))
    for (i, i_branch) in enumerate(source_tree.leaf_index)
        n_sources = get_n_bodies(source_tree.branches[i_branch].bodies_index)
        sizes[i] = (n_sources, n_sources)
    end
    
    # construct influence matrices
    matrices = Matrices(sizes, TF)

    #--- populate influence matrices ---#

    # store strengths for later
    old_strengths = save_strengths(source_buffers, source_systems)

    # set sources to unit strength
    set_unit_strength!(source_buffers, source_systems)

    # populate influence matrices
    for (i_matrix, i_leaf) in enumerate(source_tree.leaf_index)
        # get branch
        branch = source_tree.branches[i_leaf]

        # get view of matrix corresponding to this branch
        matrix, influence = get_matrix_vector(matrices, i_matrix)

        # loop over source systems
        i_source_start = 0
        for i_source_system in eachindex(source_systems)

            # unpack source system and buffer
            source_system = source_systems[i_source_system]
            source_buffer = source_buffers[i_source_system]
            source_bodies_index = branch.bodies_index[i_source_system]

            # loop over target systems
            i_target_start = 1
            for i_target_system in eachindex(target_buffers)

                # unpack target buffer
                target_buffer = target_buffers[i_target_system]
                derivatives_switch = derivatives_switches[i_target_system]
                target_bodies_index = branch.bodies_index[i_target_system]

                # view of influence corresponding to these targets
                n_targets = length(target_bodies_index)
                this_influence = view(influence, i_target_start:i_target_start + n_targets - 1)

                # loop over bodies in this branch
                for (isb, i_source_body) in enumerate(source_bodies_index)

                    # reset targets
                    reset!(target_buffer, target_bodies_index)

                    # direct influence on these targets
                    direct!(target_buffer, target_bodies_index, derivatives_switch, source_system, source_buffer, i_source_body:i_source_body)

                    # compute influences
                    influence!(this_influence, view(target_buffer, :, target_bodies_index), source_system, view(source_buffer, :, source_bodies_index))

                    # update matrix
                    matrix[i_target_start:i_target_start + n_targets - 1, isb + i_source_start] .= this_influence

                end

                # update target starting index
                i_target_start += length(branch.bodies_index[i_target_system])
            end
            
            # update source starting index
            i_source_start += length(source_bodies_index)
        end
    end
    
    # restore old strengths
    restore_strengths!(source_buffers, source_systems, old_strengths)

    # zero rhs
    matrices.rhs .= zero(TF)

    return matrices
end

"""
    map_by_leaf(source_tree::Tree)

Constructs a mapping from each leaf to the range of indices in the strengths vector that correspond to the bodies in that leaf.
"""
function map_by_leaf(source_tree::Tree)
    strengths_by_leaf = Vector{UnitRange{Int}}(undef, length(source_tree.leaf_index))
    i_start = 1
    for (i_leaf, i_branch) in enumerate(source_tree.leaf_index)

        # get bodies index
        bodies_index = source_tree.branches[i_branch].bodies_index
        
        # get number of bodies in this branch
        n_bodies = get_n_bodies(bodies_index)
        
        # store strength index
        strengths_by_leaf[i_leaf] = i_start:i_start + n_bodies - 1
        i_start += n_bodies

    end
    return strengths_by_leaf
end

"""
    map_by_branch(target_tree::Tree)

Constructs a mapping from each branch to the range of indices in the targets vector that correspond to the bodies in that branch.

This relies on the fact that `tree.leaf_index` is sorted in order of increasing branch index.

"""
function map_by_branch(target_tree::Tree)
    targets_by_branch = fill(1:0, length(target_tree.branches)) # initialize with empty ranges
    i_start = 1
    for i_branch in target_tree.leaf_index

        # get bodies index
        bodies_index = target_tree.branches[i_branch].bodies_index
        
        # get number of bodies in this branch
        n_bodies = get_n_bodies(bodies_index)
        
        # store target index
        targets_by_branch[i_branch] = i_start:i_start + n_bodies - 1
        i_start += n_bodies
    end
    return targets_by_branch
end

function add_self_interactions(direct_list::Vector{SVector{2,Int32}}, source_tree::Tree)
    # prepare container
    full_direct_list = copy(direct_list)
    i_start = length(full_direct_list)
    resize!(full_direct_list, length(full_direct_list) + length(source_tree.leaf_index))
    
    # add leaf-on-self interactions
    for (i_leaf,i_branch) in enumerate(source_tree.leaf_index)
        full_direct_list[i_start + i_leaf] = SVector{2,Int32}(i_branch, i_branch)
    end

    # sort by target
    full_direct_list = sort_by_target(full_direct_list, source_tree.branches)

    return full_direct_list
end

FastGaussSeidel(system; optargs...) = FastGaussSeidel((system,); optargs...)

FastGaussSeidel(systems::Tuple; optargs...) = FastGaussSeidel(systems, systems; optargs...)

FastGaussSeidel(target_systems, source_systems; optargs...) = FastGaussSeidel((target_systems,), (source_systems,); optargs...)

function FastGaussSeidel(target_systems::Tuple, source_systems::Tuple; 
    expansion_order=4, multipole_acceptance=0.5, leaf_size=30,
    interaction_list_method=Barba(), shrink_recenter=true,
    derivatives_switches=DerivativesSwitch(true, true, false, target_systems)
)

    #--- identical source and target trees ---#

    @assert target_systems === source_systems "different sources and targets are not yet supported for FastGaussSeidel"
    @assert interaction_list_method == Barba() "only the Barba() interaction_list_method is currently supported for FastGaussSeidel"

    #--- generate octree ---#

    # promote leaf_size to vector
    leaf_size = to_vector(leaf_size, length(source_systems))

    # create trees
    TF = promote_type(eltype.(target_systems)...)
    target_tree = Tree(target_systems, true; expansion_order, leaf_size, shrink_recenter, interaction_list_method)
    source_tree = Tree(source_systems, false; expansion_order, leaf_size, shrink_recenter, interaction_list_method)

    #--- ensure no leaves have fewer than 2 bodies ---#

    # n_bodies_min = 1000000
    # for i_branch in source_tree.leaf_index
    #     n_bodies = get_n_bodies(source_tree.branches[i_branch].bodies_index)
    #     n_bodies_min = min(n_bodies_min, n_bodies)
    # end
    # @assert n_bodies_min >= 2 "The smallest leaf has only $n_bodies_min bodies, but at least 2 are required for FastGaussSeidel.\nConsider increasing the leaf_size or using a different interaction_list_method."

    #--- build interaction lists ---#

    farfield, nearfield, self_induced = true, true, false # self-induced interactions accounted for in self-influence matrices
    m2l_list, direct_list = build_interaction_lists(target_tree.branches, source_tree.branches, leaf_size, multipole_acceptance, farfield, nearfield, self_induced, interaction_list_method)

    #--- build non-self influence matrices ---#

    nonself_matrices, sorted_list = nonself_influence_matrices(target_tree.buffers, source_tree.buffers, source_systems, target_tree, source_tree, direct_list, derivatives_switches)
    old_influence_storage = similar(nonself_matrices.rhs)

    #--- full direct list includes leaf-on-self interactions ---#

    full_direct_list = add_self_interactions(direct_list, source_tree)
    
    #--- index by source ---#

    index_map = index_by_source(sorted_list, source_tree.leaf_index)

    #--- build self-influence matrices ---#

    self_matrices = self_influence_matrices(target_tree.buffers, source_tree.buffers, source_systems, target_tree, source_tree, derivatives_switches)

    #--- source strength vector ---#

    strengths = zeros(TF, get_n_bodies(source_systems))
    strengths_by_leaf = map_by_leaf(source_tree)
    
    #--- mapping to targets by branch ---#
    
    targets_by_branch = map_by_branch(target_tree)
    
    #--- external right-hand side vector ---#
    
    extra_right_hand_side = Vector{TF}(undef, get_n_bodies(target_systems))
    
    #--- influences per system ---#
    
    influences_per_system = Vector{Vector{TF}}(undef, length(target_systems))
    for i_target_system in eachindex(target_systems)
        # get number of bodies in this target system
        n_bodies = get_n_bodies(target_systems[i_target_system])
        
        # create vector for influences
        influences_per_system[i_target_system] = zeros(TF, n_bodies)
    end

    #--- residual vector ---#

    # get max number of strengths in a leaf
    n_max = 0
    for i_leaf in eachindex(self_matrices.sizes)
        n_sources = self_matrices.sizes[i_leaf][1]
        n_max = max(n_max, n_sources)
    end

    # construct residual vector
    residual_vector = Vector{TF}(undef, n_max)

    return FastGaussSeidel{TF,length(source_systems),typeof(interaction_list_method)}(
        self_matrices,
        nonself_matrices,
        index_map,
        m2l_list,
        sorted_list,
        full_direct_list,
        interaction_list_method,
        multipole_acceptance,
        strengths,
        strengths_by_leaf,
        targets_by_branch,
        source_tree,
        target_tree,
        old_influence_storage,
        extra_right_hand_side,
        influences_per_system,
        residual_vector
    )
end

function reset!(v::Vector{TF}) where TF
    # reset vector to zero
    v .= zero(TF)
end

function reset!(influences_per_system::Vector{<:AbstractVector})
    for i in eachindex(influences_per_system)
        reset!(influences_per_system[i])
    end
end

function update_by_leaf!(strengths::Vector, strengths_by_leaf::Vector{UnitRange{Int}}, source_systems::Tuple, source_buffers::Tuple{<:Matrix}, source_tree::Tree)
    # update strengths by leaf
    for (i_leaf, i_branch) in enumerate(source_tree.leaf_index)
        # get bodies index
        bodies_index = source_tree.branches[i_branch].bodies_index

        if get_n_bodies(bodies_index) > 0
            # get strengths for this leaf
            strength_index = strengths_by_leaf[i_leaf]
            these_strengths = view(strengths, strength_index)

            # loop over source systems
            i_strength = 1
            for i_source_system in eachindex(source_systems)
                # unpack source system and buffer
                source_system = source_systems[i_source_system]
                source_buffer = source_buffers[i_source_system]

                # update strengths
                for i_body in bodies_index[i_source_system]
                    # set strength value
                    these_strengths[i_strength] = strength_to_value(source_buffer, source_system, i_body)

                    # increment index
                    i_strength += 1
                end
            end
            @assert i_strength - 1 == get_n_bodies(bodies_index) "number of strengths does not match number of bodies in leaf $(i_leaf)"
        end
    end
end

function update_by_leaf!(source_buffers::Tuple{<:Matrix}, source_systems::Tuple, strengths::Vector, strengths_by_leaf::Vector{UnitRange{Int}}, source_tree::Tree)
    # update strengths by leaf
    for (i_leaf, i_branch) in enumerate(source_tree.leaf_index)
        # get bodies index
        bodies_index = source_tree.branches[i_branch].bodies_index

        if get_n_bodies(bodies_index) > 0
            # get strengths for this leaf
            strength_index = strengths_by_leaf[i_leaf]
            these_strengths = view(strengths, strength_index)

            # loop over source systems
            i_strength = 1
            for i_source_system in eachindex(source_systems)
                # unpack source system and buffer
                source_system = source_systems[i_source_system]
                source_buffer = source_buffers[i_source_system]

                # update strengths
                for i_body in bodies_index[i_source_system]
                    # set strength value
                    value_to_strength!(source_buffer, source_system, i_body, these_strengths[i_strength])

                    # increment index
                    i_strength += 1
                end
            end
            @assert i_strength - 1 == get_n_bodies(bodies_index) "number of strengths does not match number of bodies in leaf $(i_leaf)"
        end
    end
end

"""
    update_nonself_influence!(right_hand_side, nonself_matrices::Matrices, old_influence_storage::Vector, source_tree::Tree, target_tree::Tree, strengths_by_leaf::Vector{UnitRange{Int}}, index_map::Vector{UnitRange{Int}}, direct_list::Vector{SVector{2,Int32}}, targets_by_branch::Vector{UnitRange{Int}})

Updates the right-hand side vector based on the non-self influence matrices and the current strengths of the source systems. Does this by removing the old influence and adding the new.

"""
function update_nonself_influence!(right_hand_side, strengths::Vector, nonself_matrices::Matrices, old_influence_storage::Vector, source_tree::Tree, target_tree::Tree, strengths_by_leaf::Vector{UnitRange{Int}}, index_map::Vector{UnitRange{Int}}, direct_list::Vector{SVector{2,Int32}}, targets_by_branch::Vector{UnitRange{Int}})
    
    # check if there are any direct interactions
    if length(direct_list) > 0

        # loop over source leaves
        for (i_leaf, i_branch) in enumerate(source_tree.leaf_index)
            update_nonself_influence!(right_hand_side, strengths, nonself_matrices, old_influence_storage, i_leaf, source_tree, target_tree, strengths_by_leaf, index_map, direct_list, targets_by_branch)
        end
    end
end

function update_nonself_influence!(right_hand_side, strengths::Vector, nonself_matrices::Matrices, old_influence_storage, i_leaf::Int, source_tree::Tree, target_tree::Tree, strengths_by_leaf::Vector{UnitRange{Int}}, index_map::Vector{UnitRange{Int}}, direct_list::Vector{SVector{2,Int32}}, targets_by_branch::Vector{UnitRange{Int}})

    # unpack influence matrix and right-hand side
    mat, target_influence = get_matrix_vector(nonself_matrices, i_leaf)
    m = size(mat, 1)
    rhs_offset = nonself_matrices.rhs_offsets[i_leaf]
    old_influence = view(old_influence_storage, rhs_offset:rhs_offset + m - 1)

    if length(target_influence) > 0

        #--- compute the updated influence ---#

        # unpack strengths
        leaf_strengths = view(strengths, strengths_by_leaf[i_leaf])

        # copy the old influence for later
        old_influence .= target_influence

        # compute the influence
        mul!(target_influence, mat, leaf_strengths)

        #--- move to right-hand side ---#

        # determine which target branches this leaf influences
        direct_list_indices = index_map[i_leaf]
        i_influence_start = 1
        for index in direct_list_indices

            # determine which target leaf
            i_target, _ = direct_list[index]

            # how many targets
            n_targets = get_n_bodies(target_tree.branches[i_target].bodies_index)

            # influence on this target leaf
            this_influence = view(target_influence, i_influence_start:i_influence_start + n_targets - 1)
            this_old_influence = view(old_influence, i_influence_start:i_influence_start + n_targets - 1)
            i_influence_start += n_targets

            # get influence at the appropriate target
            this_rhs = view(right_hand_side, targets_by_branch[i_target])

            # remove old influence from right-hand side
            this_rhs .+= this_old_influence

            # add influence to the right-hand side
            this_rhs .-= this_influence
        end
    end
end

solve!(system, solver::FastGaussSeidel; optargs...) = solve!((system,), solver; optargs...)

solve!(systems::Tuple, solver::FastGaussSeidel; optargs...) = solve!(systems, systems, solver; optargs...)

solve!(target_systems, source_systems, solver::FastGaussSeidel; optargs...) = solve!((target_systems,), (source_systems,), solver; optargs...)

function solve!(target_systems::Tuple, source_systems::Tuple, solver::FastGaussSeidel{TF,N}; 
    derivatives_switches=DerivativesSwitch(true, true, false, target_systems),
    max_iterations=10, tolerance=1e-3,
) where {TF,N}

    #--- unpack containers ---#

    source_tree = solver.source_tree
    target_tree = solver.target_tree
    source_buffers = source_tree.buffers
    target_buffers = target_tree.buffers
    self_matrices = solver.self_matrices
    nonself_matrices = solver.nonself_matrices
    index_map = solver.index_map
    m2l_list = solver.m2l_list
    direct_list = solver.direct_list
    full_direct_list = solver.full_direct_list
    interaction_list_method = solver.interaction_list_method
    multipole_acceptance = solver.multipole_acceptance
    strengths = solver.strengths
    strengths_by_leaf = solver.strengths_by_leaf
    targets_by_branch = solver.targets_by_branch
    influences_per_system = solver.influences_per_system
    old_influence_storage = solver.old_influence_storage
    right_hand_side = self_matrices.rhs
    extra_right_hand_side = solver.extra_right_hand_side
    residual_vector = solver.residual_vector

    #--- external right-hand side based on current influence ---#

    # reset and update buffers
    target_influence_to_buffer!(target_buffers, target_systems, derivatives_switches, target_tree.sort_index_list)

    # run influence function on buffers
    reset!(extra_right_hand_side)
    influence!(extra_right_hand_side, influences_per_system, target_buffers, source_systems, source_buffers, source_tree)

    # set the right-hand side to the external influence
    # NOTE: this only happens once as it is not reset in the iterations
    right_hand_side .= extra_right_hand_side

    #--- update strengths ---#

    update_by_leaf!(strengths, strengths_by_leaf, source_systems, source_buffers, source_tree)

    #--- non-self influence ---#

    # add nonself influence to the right-hand side
    nonself_matrices.rhs .= zero(TF) # reset rhs
    update_nonself_influence!(right_hand_side, strengths, nonself_matrices, old_influence_storage, source_tree, target_tree, strengths_by_leaf, index_map, direct_list, targets_by_branch)

    #--- fast gauss seidel iterations ---#

    # prepare inputs
    empty_direct_list = Vector{Tuple{Int,Int}}(undef, 0)
    mse = one(TF) * 100000
    mse_best = mse

    # begin iterations
    for iteration in 1:max_iterations

        #--- farfield influence ---#
        
        # fmm call
        reset!(target_buffers)
        fmm!(target_systems, target_tree, source_systems, source_tree, source_tree.leaf_size, m2l_list, empty_direct_list, derivatives_switches, interaction_list_method;
            source_tree.expansion_order, error_tolerance=nothing,
            upward_pass=true, horizontal_pass=true, downward_pass=true,
            # horizontal_pass_verbose::Bool=false,
            reset_target_tree=true, reset_source_tree=true,
            # nearfield_device::Bool=false,
            tune=false, update_target_systems=false, multipole_acceptance,
            # t_source_tree=0.0, t_target_tree=0.0, t_lists=0.0,
            # silence_warnings=false,
        )

        # move farfield influence to the right-hand side
        reset!(extra_right_hand_side)
        influence!(extra_right_hand_side, influences_per_system, target_buffers, source_systems, source_buffers, source_tree)
        right_hand_side .+= extra_right_hand_side

        #--- check residual ---#

        # note that `right_hand_side` now contains external, nonself, and farfield influence
        mse = residual!(residual_vector, self_matrices, strengths, strengths_by_leaf)
        
        println("Iteration $(iteration): MSE = $(mse)")
        
        # if mse > mse_best * 10 # stop if mse begins increasing
        #     @warn "FastGaussSeidel stopped early at iteration $(iteration) with MSE = $(mse) (previous was $(mse_best))"
        #     break
        # end
        # mse_best = min(mse_best, mse)

        if mse <= tolerance
            @info "FastGaussSeidel converged after $(iteration-1) iterations with MSE = $(mse)"
            break
        end

        #--- nearfield influence and solve ---#

        for (i_leaf, i_branch) in enumerate(source_tree.leaf_index)
            
            # unpack influence matrix and right-hand side
            mat, rhs = get_matrix_vector(self_matrices, i_leaf)

            # unpack strengths
            leaf_strengths = view(strengths, strengths_by_leaf[i_leaf])

            # solve for strengths
            leaf_strengths .= mat \ rhs

            # update non-self influence
            length(direct_list) > 0 && update_nonself_influence!(right_hand_side, strengths, nonself_matrices, old_influence_storage, i_leaf, source_tree, target_tree, strengths_by_leaf, index_map, direct_list, targets_by_branch)

        end

        #--- update strengths in buffers ---#

        update_by_leaf!(source_buffers, source_systems, strengths, strengths_by_leaf, source_tree)

        #--- restore right hand side to exclude farfield influence ---#

        right_hand_side .-= extra_right_hand_side

    end

    #--- check if we converged ---#
    if mse > tolerance
        @warn "FastGaussSeidel did not converge after $(max_iterations) iterations with MSE = $(mse)"
    end

    #--- final update of systems ---#

    # use new strengths to get the full influence (farfield was already computed)
    fmm!(target_systems, target_tree, source_systems, source_tree, source_tree.leaf_size, m2l_list, full_direct_list, derivatives_switches, interaction_list_method;
            expansion_order=source_tree.expansion_order, error_tolerance=nothing,
            upward_pass=false, horizontal_pass=false, downward_pass=false, # just nearfield influence
            # horizontal_pass_verbose::Bool=false,
            reset_target_tree=false, reset_source_tree=false, # false now
            # nearfield_device::Bool=false,
            tune=false, update_target_systems=true, multipole_acceptance, # update_target_systems is `true` now
            # t_source_tree=0.0, t_target_tree=0.0, t_lists=0.0,
            # silence_warnings=false,
        )

    # update source system strengths
    buffer_to_system_strength!(source_systems, source_tree)

end

"""
    influence!(sorted_influences, influences_per_system, target_buffers, source_systems, source_buffers, source_tree)

Evaluate the influence as pertains to the boundary element influence matrix and subtracts it from `sorted_influences` (which would act like the RHS of a linear system). Based on the current state of the `target_buffers` and `source_buffers`. Note that `source_systems` is provided solely for dispatch. Note also that `influences_per_system` is overwritten each time.

* `sorted_influences::Vector{Float64}`: single vector containing the influence for every body in the target buffers, sorted by source branch in the direct interaction list
* `influences_per_system::Vector{Vector{Float64}}`: vector of vectors containing the influence for each target system, sorted the same way as the buffers
* `target_buffers::NTuple{N,Matrix{Float64}}`: target buffers used to compute the influence
* `source_systems::NTuple{N,<:{UserDefinedSystem}}`: system objects used for dispatch
* `source_buffers::NTuple{N,Matrix{Float64}}`: source buffers used to compute the influence

"""
function influence!(sorted_influences::Vector{TF}, influences_per_system::Vector{Vector{TF}}, target_buffers::Tuple, source_systems::Tuple, source_buffers::Tuple, source_tree::Tree) where TF
    @assert length(target_buffers) == length(source_buffers) == length(source_systems)

    #--- evaluate influences ---#

    for i_system in eachindex(target_buffers)
        # unpack containers
        influence = influences_per_system[i_system]
        target_buffer = target_buffers[i_system]
        source_buffer = source_buffers[i_system]
        source_system = source_systems[i_system]

        # evaluate influence
        influence!(influence, target_buffer, source_system, source_buffer)
    end

    #--- sort by source ---#

    i_influence = 1
    for i_leaf in source_tree.leaf_index
        # unpack bodies index
        bodies_index = source_tree.branches[i_leaf].bodies_index

        for i_system in eachindex(source_systems)
            # unpack containers
            index = bodies_index[i_system]
            influences = influences_per_system[i_system]

            # update influences
            sorted_influences[i_influence:i_influence + length(index) - 1] .-= view(influences, index)
            i_influence += length(index)
        end
    end
end

function residual!(residual_vector, self_matrices::Matrices, strengths::Vector, strengths_by_leaf::Vector{UnitRange{Int}})

    # loop over self influence matrices
    for i_leaf in eachindex(self_matrices.rhs_offsets)
        # get matrix and rhs
        mat, rhs = get_matrix_vector(self_matrices, i_leaf)

        # unpack strengths
        leaf_strengths = view(strengths, strengths_by_leaf[i_leaf])

        # compute the residual
        vrange = 1:length(rhs)
        this_residual = view(residual_vector, vrange)
        this_residual .= rhs
        mul!(this_residual, mat, leaf_strengths, 1.0, -1.0)
    end

    # sum of the squared residuals
    mse = zero(eltype(residual_vector))
    for i in eachindex(residual_vector)
        r = residual_vector[i]
        mse += r * r
    end

    # compute the mean squared error
    mse /= length(residual_vector)

    return mse
end