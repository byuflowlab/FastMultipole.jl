#------- direct interactions -------#

function nearfield_singlethread!(target_buffers, target_branches, source_systems, source_buffers, source_branches, derivatives_switches, direct_list)
    # loop over sources
    t_nf = @MVector zeros(length(source_systems))
    for i_source_system in eachindex(source_systems)
        source_system = source_systems[i_source_system]
        source_buffer = source_buffers[i_source_system]

        # perform direct interactions
        t_elapsed = @elapsed nearfield_loop!(target_buffers, target_branches, source_system, source_buffer, i_source_system, source_branches, direct_list, derivatives_switches)
        t_nf[i_source_system] = t_elapsed
    end

    return t_nf
end

function nearfield_loop!(target_buffers, target_branches, source_system, source_buffer, i_source_system, source_branches, direct_list, derivatives_switches)
    # loop over target systems
    for (i_target_system, target_system) in enumerate(target_buffers)

        # extract derivatives switch
        derivatives_switch = derivatives_switches[i_target_system]

        # loop over direct list
        for (i_target, i_source) in direct_list

            # identify sources
            source_index = source_branches[i_source].bodies_index[i_source_system]

            # identify targets
            target_index = target_branches[i_target].bodies_index[i_target_system]

            # compute interaction
            direct!(target_system, target_index, derivatives_switch, source_system, source_buffer, source_index)

        end

    end
end

function get_n_interactions(i_target_system, target_branches, i_source_system, source_branches::Vector{<:Branch}, direct_list)
    n_interactions = 0
    for (i_target, i_source) in direct_list
        n_interactions += target_branches[i_target].n_bodies[i_target_system] * source_branches[i_source].n_bodies[i_source_system]
    end

    return n_interactions
end

"""
    make_direct_assignments!(assignments, i_target_system, target_branches, i_source_system, source_branches, direct_list, n_threads, n_per_thread, interaction_list_method)

Assumes `direct_list` is sorted by target branch index. Assigns ranges of interactions to each thread.

"""
function make_direct_assignments!(assignments, i_target_system, target_branches, i_source_system, source_branches, direct_list, n_threads, n_per_thread, interaction_list_method)
    i_start = 1
    i_end = 1
    i_thread = 1
    n_interactions = 0

    if length(direct_list) > 0

        # prepare recursive quantities
        i_target, i_source = direct_list[1]

        # loop over interaction list
        for (i_target_next, i_source_next) in view(direct_list, 2:length(direct_list))
            # update number of interactions in the current assignment
            n_interactions += target_branches[i_target].n_bodies[i_target_system] * source_branches[i_source].n_bodies[i_source_system]

            # if sorting by target, make sure we finish off the current target
            ready = i_target_next != i_target

            # if we exceed n_per_thread, finish this assignment and reset counters
            if n_interactions >= n_per_thread && ready
                assignments[i_thread] = i_start:i_end
                i_start = i_end + 1
                i_thread += 1
                n_interactions = 0
            end

            # recurse
            i_end += 1
            i_target, i_source = i_target_next, i_source_next
        end
        
        # get the last assignment
        assignments[i_thread] = i_start:i_end

    end
end

function execute_assignment!(target_buffer, i_target_buffer, target_branches, derivatives_switch, source_system, source_buffer, i_source_system, source_branches, direct_list, assignment, interaction_list_method)
    for i_interaction in assignment
        i_target, i_source = direct_list[i_interaction]
        
        # identify sources
        source_branch = source_branches[i_source]
        source_index = source_branch.bodies_index[i_source_system]
        
        # identify targets
        target_branch = target_branches[i_target]
        target_index = target_branch.bodies_index[i_target_buffer]

        # compute interaction
        direct!(target_buffer, target_index, derivatives_switch, source_system, source_buffer, source_index)
    end
end

function nearfield_multithread!(target_systems, target_branches, source_systems::Tuple, source_buffers, source_branches, derivatives_switches, direct_list, interaction_list_method, n_threads)
    # benchmark for auto-tuning
    t_nf = @MVector zeros(length(source_systems))

    for (i_source_system, source_system) in enumerate(source_systems)
        source_buffer = source_buffers[i_source_system]
        for (i_target_buffer, target_buffer) in enumerate(target_systems)
            t = @elapsed nearfield_multithread!(target_buffer, i_target_buffer, target_branches, source_system, source_buffer, i_source_system, source_branches, derivatives_switches[i_target_buffer], direct_list, interaction_list_method, n_threads)
            t_nf[i_source_system] += t
        end
    end

    return t_nf
end

function nearfield_multithread!(target_buffer, i_target_buffer, target_branches, source_system, source_buffer, i_source_system, source_branches, derivatives_switch, direct_list, interaction_list_method, n_threads)

    #--- load balance ---#

    # total number of interactions
    n_interactions = get_n_interactions(i_target_buffer, target_branches, i_source_system, source_branches, direct_list)

    # interactions per thread
    n_per_thread, rem = divrem(n_interactions, n_threads)
    rem > 0 && (n_per_thread += 1)

    # if there are too many threads, we'll actually hurt performance
    n_per_thread < MIN_NPT_NF && (n_per_thread = MIN_NPT_NF)

    # create assignments
    assignments = Vector{UnitRange{Int64}}(undef,n_threads)
    for i in eachindex(assignments)
        assignments[i] = 1:0
    end
    make_direct_assignments!(assignments, i_target_buffer, target_branches, i_source_system, source_branches, direct_list, n_threads, n_per_thread, interaction_list_method)

    # # rule out datarace conditions
    # copies = Tuple(zeros(size(target_buffer)) for _ in 1:n_threads)
    # pos = view(target_buffer, 1:3, :)
    # for i_copy in 1:n_threads
    #     copies[i_copy][1:3,:] .= pos
    # end

    # execute tasks
    Threads.@threads :static for i_task in eachindex(assignments)
        assignment = assignments[i_task]
        # this_buffer = copies[i_task]
        # execute_assignment!(this_buffer, i_target_buffer, target_branches, derivatives_switch, source_system, source_buffer, i_source_system, source_branches, direct_list, assignment, interaction_list_method)
        execute_assignment!(target_buffer, i_target_buffer, target_branches, derivatives_switch, source_system, source_buffer, i_source_system, source_branches, direct_list, assignment, interaction_list_method)
    end

    # if DEBUG[]
    #     test_buffer = deepcopy(target_buffer)
    #     test_buffer[4:16,:] .= zero(eltype(target_buffer))
    #     rs = 900:920
    #     @show target_buffer[4,rs]
    #     for (i,b) in enumerate(copies)
    #         println("i=$i")
    #         @show b[4,rs]
    #         test_buffer[4,:] .+= view(b, 4, :)
    #     end
    #    @show findmax(abs.(test_buffer[4,:] - target_buffer[4,:]))
    # end

    # consolidate buffers
    # for b in copies
    #     target_buffer[4:end,:] .+= view(b, 4:16, :)
    # end

end

"""
    nearfield_device!(target_systems, target_tree, derivatives_switches, source_systems, source_tree, direct_list)

User-defined function used to offload nearfield calculations to a device, such as GPU.

# Arguments

* `target_systems`: user-defined system on which `source_system` acts
* `target_tree::Tree`: octree object used to sort `target_systems`
* `derivatives_switches::Union{DerivativesSwitch, NTuple{N,DerivativesSwitch}}`: determines whether the scalar potential, vector field, and or vector gradient should be calculated
* `source_systems`: user-defined system acting on `target_system`
* `source_tree::Tree`: octree object used to sort `target_systems`
* `direct_list::Vector{SVector{2,Int32}}`: each element `[i,j]` maps nearfield interaction from `source_tree.branches[j]` on `target_tree.branches[i]`

"""
function nearfield_device!(target_systems, target_tree::Tree, derivatives_switches, source_systems, source_tree::Tree, direct_list)
    @warn "nearfield_device! was called but hasn't been overloaded by the user"
end

"""
    nearfield_device!(target_systems, derivatives_switches, source_systems)

Dispatches `nearfield_device!` without having to build a `::Tree`. Performs all interactions.

# Arguments

* `target_systems`: user-defined system on which `source_system` acts
* `derivatives_switches::Union{DerivativesSwitch, NTuple{N,DerivativesSwitch}}`: determines whether the scalar potential, vector field, and or vector gradient should be calculated
* `source_systems`: user-defined system acting on `target_system`

"""
function nearfield_device!(target_systems, derivatives_switches, source_systems)

    # get type
    x_source, _, _ = first_body_position(source_systems)
    TF_source = typeof(x_source)
    x_target, _, _ = first_body_position(target_systems)
    TF_target = typeof(x_target)
    TF = promote_type(TF_source, TF_target)

    # build target tree
    target_bodies_index = get_bodies_index(target_systems)
    n_branches, branch_index, i_parent, i_leaf_index = 0, 1, -1, 1
    center, radius, = SVector{3,TF}(0.0,0,0), zero(TF)
    box, expansion_order = SVector{3,TF}(0.0,0,0), 0
    target_branch = Branch(target_bodies_index, n_branches, branch_index, i_parent, i_leaf_index, center, radius, box, zero(radius))
    levels_index, leaf_index, sort_index, inverse_sort_index, leaf_size = [1:1], [1], dummy_sort_index(target_systems), dummy_sort_index(target_systems), full_leaf_size(target_systems)
    target_tree = Tree([target_branch], levels_index, leaf_index, sort_index, inverse_sort_index, buffer, Val(expansion_order), leaf_size)

    if target_systems === source_systems
        source_tree = target_tree
    else
        # build source tree
        source_bodies_index = get_bodies_index(source_systems)
        source_branch = Branch(source_bodies_index, n_branches, branch_index, i_parent, i_leaf_index, center, radius, box, expansion_order)
        sort_index, inverse_sort_index, leaf_size = dummy_sort_index(source_systems), dummy_sort_index(source_systems), full_leaf_size(source_systems)
        source_tree = Tree([source_branch], levels_index, leaf_index, sort_index, inverse_sort_index, buffer, expansion_order, leaf_size)
    end

    # build direct_list
    direct_list = [SVector{2,Int32}(1,1)]

    # call user-defined function
    nearfield_device!(target_systems, target_tree, derivatives_switches, source_systems, source_tree, direct_list)

end

#------- UPWARD PASS -------#

function upward_pass_singlethread_1!(tree::Tree{TF, <:Any}, systems, expansion_order) where TF

    harmonics = initialize_harmonics(expansion_order, TF)

    # body_to_multipole
    for (i_system, system) in enumerate(systems)
        body_to_multipole!(tree, harmonics, system, i_system, expansion_order)
    end
end

function upward_pass_singlethread_2!(tree::Tree{TF,<:Any}, expansion_order, lamb_helmholtz) where TF

    # try preallocating one container to be reused
    Ts = zeros(TF, length_Ts(expansion_order))
    eimϕs = zeros(TF, 2, expansion_order+1)
    weights_tmp_1 = initialize_expansion(expansion_order, TF)
    weights_tmp_2 = initialize_expansion(expansion_order, TF)

    # loop over branches
    for i_branch in length(tree.branches):-1:1 # no need to create a multipole expansion at the very top level
        branch = tree.branches[i_branch]
        expansion = view(tree.expansions, :, :, :, i_branch)

        if branch.n_branches !== 0 # branch is not a leaf
            # iterate over children
            for i_child in branch.branch_index
                child_branch = tree.branches[i_child]
                child_expansion = view(tree.expansions, :, :, :, i_child)
                multipole_to_multipole!(expansion, branch, child_expansion, child_branch, weights_tmp_1, weights_tmp_2, Ts, eimϕs, ζs_mag, Hs_π2, expansion_order, lamb_helmholtz)
            end
        end
    end
end

function upward_pass_singlethread!(tree::Tree, systems, expansion_order, lamb_helmholtz)
    upward_pass_singlethread_1!(tree, systems, expansion_order)
    upward_pass_singlethread_2!(tree, expansion_order, lamb_helmholtz)
end

function upward_pass_multithread_1!(source_tree::Tree, systems::Tuple, expansion_order, n_threads)
    
    #--- load balance ---#

    # extract containers
    leaf_index = source_tree.leaf_index
    branches = source_tree.branches

    leaf_assignments = fill(1:0, length(systems), n_threads)
    for (i_system, system) in enumerate(systems)

        # total number of bodies
        n_bodies = get_n_bodies(system)

        # number of bodies per thread
        n_per_thread, rem = divrem(n_bodies, n_threads)
        rem > 0 && (n_per_thread += 1)

        # if there are too many threads, we'll actually hurt performance
        n_per_thread < MIN_NPT_B2M && (n_per_thread = MIN_NPT_B2M)

        # create chunks
        i_start = 1
        i_thread = 1
        n_bodies = 0
        for (i_end,i_leaf) in enumerate(leaf_index)
            n_bodies += length(branches[i_leaf].bodies_index[i_system])
            if n_bodies >= n_per_thread
                leaf_assignments[i_system,i_thread] = i_start:i_end
                i_start = i_end+1
                i_thread += 1
                n_bodies = 0
            end
        end
        i_thread <= n_threads && (leaf_assignments[i_system,i_thread] = i_start:length(leaf_index))
    end

    #--- preallocate memory ---#

    harmonics = [initialize_harmonics(expansion_order) for _ in 1:n_threads]

    #--- compute multipole expansion coefficients ---#

    for (i_system, system) in enumerate(systems)
        buffer = source_tree.buffers[i_system]
        Threads.@threads :static for i_thread in 1:n_threads
            leaf_assignment = leaf_assignments[i_system,i_thread]
            for i_task in leaf_assignment
                i_branch = leaf_index[i_task]
                branch = branches[i_branch]
                multipole_coefficients = view(source_tree.expansions, :, :, :, i_branch)
                body_to_multipole!(system, multipole_coefficients, buffer, branch.center, branch.bodies_index[i_system], harmonics[i_thread], expansion_order)
            end
        end
    end
end

function assign_m2m!(assignments, branches, level_index, n_per_thread, n_threads)
    i_start = level_index[1]
    i_end = i_start
    i_thread = 1
    n_interactions = 0

    # reset assignments
    for i in eachindex(assignments)
        assignments[i] = 1:0
    end

    # loop over interaction list
    for i_branch in level_index

        # extract branch
        branch = branches[i_branch]

        # update number of interactions in the current assignment
        n_interactions += branch.n_branches

        # if we exceed n_per_thread, finish this assignment and reset counters
        if n_interactions >= n_per_thread
            assignments[i_thread] = i_start:i_end
            i_start = i_end + 1
            i_thread += 1
            n_interactions = 0
        end

        i_end += 1
    end

    i_thread <= n_threads && (assignments[i_thread] = i_start:level_index[end])
end

function execute_m2m!(expansions, branches, assignment, weights_tmp_1, weights_tmp_2, Ts, eimϕs, ζs_mag, Hs_π2, expansion_order, lamb_helmholtz)
    for i_branch in assignment
        # extract branch
        parent_branch = branches[i_branch]
        parent_expansion = view(expansions, :, :, :, i_branch)

        # loop over children
        for i_child in parent_branch.branch_index
            child_branch = branches[i_child]
            child_expansion = view(expansions, :, :, :, i_child)
            multipole_to_multipole!(parent_expansion, parent_branch, child_expansion, child_branch, weights_tmp_1, weights_tmp_2, Ts, eimϕs, ζs_mag, Hs_π2, expansion_order, lamb_helmholtz)
        end
    end
end

function upward_pass_multithread_2!(tree::Tree{TF,N}, expansion_order, lamb_helmholtz, n_threads) where {TF,N}

    # extract containers
    branches = tree.branches
    levels_index = tree.levels_index
    expansions = tree.expansions

    # try preallocating one set of containers to be reused
    Ts = [zeros(TF, length_Ts(expansion_order)) for _ in 1:n_threads]
    eimϕs = [zeros(TF, 2, expansion_order+1) for _ in 1:n_threads]
    weights_tmp_1 = [initialize_expansion(expansion_order, TF) for _ in 1:n_threads]
    weights_tmp_2 = [initialize_expansion(expansion_order, TF) for _ in 1:n_threads]
    assignments = fill(1:0, n_threads)

    # iterate over levels
    for i_level in length(levels_index)-1:-1:1
        level_index = levels_index[i_level]

        # load balance
        n_interactions = 0
        for i_branch in level_index
            n_interactions += branches[i_branch].n_branches
        end
        n_per_thread, rem = divrem(n_interactions, n_threads)
        rem > 0 && (n_per_thread += 1)

        # if there are too many threads, we'll actually hurt performance
        n_per_thread < MIN_NPT_M2M && (n_per_thread = MIN_NPT_M2M)

        # how many threads are actually needed
        assign_m2m!(assignments, branches, level_index, n_per_thread, n_threads)

        # assign thread start branches
        Threads.@threads :static for i_task in 1:n_threads
            # get assignment
            assignment = assignments[i_task]

            # execute assignment
            execute_m2m!(expansions, branches, assignment, weights_tmp_1[i_task], weights_tmp_2[i_task], Ts[i_task], eimϕs[i_task], ζs_mag, Hs_π2, expansion_order, lamb_helmholtz)
        end
    end
end

function upward_pass_multithread!(source_tree, source_systems, expansion_order, lamb_helmholtz, n_threads)

    # create multipole expansions
    upward_pass_multithread_1!(source_tree, source_systems, expansion_order, n_threads)

    # m2m translation
    upward_pass_multithread_2!(source_tree, expansion_order, lamb_helmholtz, n_threads)
end

#------- direct interaction matrix -------#

# TODO: add influence matrix approach to direct interactions

#------- horizontal pass -------#

function horizontal_pass_singlethread!(target_tree::Tree{TF1,<:Any}, source_tree::Tree{TF2,<:Any}, m2l_list, lamb_helmholtz, expansion_order, error_tolerance; verbose=false) where {TF1,TF2}

    TF = promote_type(TF1, TF2)

    # preallocate containers to be reused
    weights_tmp_1 = initialize_expansion(expansion_order, TF)
    weights_tmp_2 = initialize_expansion(expansion_order, TF)
    weights_tmp_3 = initialize_expansion(expansion_order, TF)
    Ts = zeros(TF, length_Ts(expansion_order))
    eimϕs = zeros(TF, 2, expansion_order + 1)

    Pmax = 0
    error_success = true
    # Ps = zeros(length(m2l_list))
    for (i,(i_target, j_source)) in enumerate(m2l_list)
        target_branch = target_tree.branches[i_target]
        target_expansion = view(target_tree.expansions, :, :, :, i_target)
        source_branch = source_tree.branches[j_source]
        source_expansion = view(source_tree.expansions, :, :, :, j_source)
        P, this_error_success = multipole_to_local!(target_expansion, target_branch, source_expansion, source_branch, weights_tmp_1, weights_tmp_2, weights_tmp_3, Ts, eimϕs, ζs_mag, ηs_mag, Hs_π2, M̃, L̃, expansion_order, lamb_helmholtz, error_tolerance)
        Pmax = max(P, Pmax)
        # Ps[i] = P
        error_success = error_success && this_error_success
    end

    # if verbose
    #     println("\n------- M2L Stats: -------")
    #     println("\n\tmean: ", mean(Ps))
    #     println("\tstd:  ", std(Ps))
    #     println("\tmax:  ", maximum(Ps))
    #     println("\tmin:  ", minimum(Ps))
    #     println("\n--------------------------\n")
    # end

    return Pmax, error_success
end

function assign_m2l!(assignments, m2l_list, n_threads, n_per_thread, interaction_list_method)
    i_start = 1
    i_end = 1
    i_thread = 1
    n_interactions = 0

    if length(m2l_list) > 0

        i_target, i_source = m2l_list[1]

        # loop over interaction list
        for (i_target_next, i_source_next) in view(m2l_list, 2:length(m2l_list))
            # update number of interactions in the current assignment
            n_interactions += 1

            # if sorting by target, make sure we finish off the current target
            ready = i_target != i_target_next

            # if we exceed n_per_thread, finish this assignment and reset counters
            if n_interactions >= n_per_thread && ready
                assignments[i_thread] = i_start:i_end
                i_start = i_end + 1
                i_thread += 1
                n_interactions = 0
            end

            i_end += 1
            i_target, i_source = i_target_next, i_source_next
        end

        # get the last assignment
        assignments[i_thread] = i_start:i_end

    end
end

# function execute_m2l!(target_expansions, target_branches, source_expansions, source_branches, m2l_list, assignment, weights_tmp_1, weights_tmp_2, weights_tmp_3, Ts, eimϕs, ζs_mag, ηs_mag, Hs_π2, M̃, L̃, expansion_order, lamb_helmholtz, error_tolerance, ::InteractionListMethod{SortByTarget()})
#     Pmax = 0
#     error_success = true
#     for i in assignment
#         i_target, j_source = m2l_list[i]
#         target_expansion = view(target_expansions, :, :, :, i_target)
#         source_expansion = view(source_expansions, :, :, :, j_source)
#         target_branch = target_branches[i_target]
#         source_branch = source_branches[j_source]
#         P, this_error_success = multipole_to_local!(target_expansion, target_branch, source_expansion, source_branch, weights_tmp_1, weights_tmp_2, weights_tmp_3, Ts, eimϕs, ζs_mag, ηs_mag, Hs_π2, M̃, L̃, expansion_order, lamb_helmholtz, error_tolerance)
#         Pmax = max(P, Pmax)
#         error_success = error_success && this_error_success
#     end    

#     return Pmax, error_success
# end

function execute_m2l!(target_expansions, target_branches, source_expansions, source_branches, m2l_list, assignment, weights_tmp_1, weights_tmp_2, weights_tmp_3, Ts, eimϕs, ζs_mag, ηs_mag, Hs_π2, M̃, L̃, expansion_order, lamb_helmholtz, error_tolerance, ::InteractionListMethod)
    Pmax = 0
    error_success = true
    for i in assignment
        i_target, j_source = m2l_list[i]
        target_expansion = view(target_expansions, :, :, :, i_target)
        source_expansion = view(source_expansions, :, :, :, j_source)
        target_branch = target_branches[i_target]
        source_branch = source_branches[j_source]
        P, this_error_success = multipole_to_local!(target_expansion, target_branch, source_expansion, source_branch, weights_tmp_1, weights_tmp_2, weights_tmp_3, Ts, eimϕs, ζs_mag, ηs_mag, Hs_π2, M̃, L̃, expansion_order, lamb_helmholtz, error_tolerance)
        Pmax = max(P, Pmax)
        error_success = error_success && this_error_success
    end    

    return Pmax, error_success
end

function horizontal_pass_multithread!(target_tree::Tree{TF1,<:Any}, source_tree::Tree{TF2,<:Any}, m2l_list, lamb_helmholtz, expansion_order, error_tolerance, interaction_list_method, n_threads; verbose=false) where {TF1, TF2}

    TF = promote_type(TF1, TF2)

    # number of translations per thread
    n_per_thread, rem = divrem(length(m2l_list),n_threads)
    rem > 0 && (n_per_thread += 1)
    assignments = fill(1:0, n_threads)

    # make assignments 
    assign_m2l!(assignments, m2l_list, n_threads, n_per_thread, interaction_list_method)

    # preallocate containers
    Ts = [zeros(TF, length_Ts(expansion_order)) for _ in 1:n_threads]
    eimϕs = [zeros(TF, 2, expansion_order + 1) for _ in 1:n_threads]
    weights_tmp_1 = [initialize_expansion(expansion_order, TF) for _ in 1:n_threads]
    weights_tmp_2 = [initialize_expansion(expansion_order, TF) for _ in 1:n_threads]
    weights_tmp_3 = [initialize_expansion(expansion_order, TF) for _ in 1:n_threads]
    Pmax = @MVector zeros(Int, n_threads)
    error_success = @MVector zeros(Bool, n_threads)
    target_expansions = target_tree.expansions
    target_branches = target_tree.branches
    source_expansions = source_tree.expansions
    source_branches = source_tree.branches

    # execute tasks
    Threads.@threads :static for i_thread in 1:n_threads
        this_Pmax, this_error_success = execute_m2l!(target_expansions, target_branches, source_expansions, source_branches, m2l_list, assignments[i_thread], weights_tmp_1[i_thread], weights_tmp_2[i_thread], weights_tmp_3[i_thread], Ts[i_thread], eimϕs[i_thread], ζs_mag, ηs_mag, Hs_π2, M̃, L̃, expansion_order, lamb_helmholtz, error_tolerance, interaction_list_method)
        Pmax[i_thread] = this_Pmax
        error_success[i_thread] = this_error_success
    end

    Pmax = maximum(Pmax)
    error_success = prod(error_success)

    return Pmax, error_success
end

#------- DOWNWARD PASS -------#

function downward_pass_singlethread_1!(tree::Tree{TF,<:Any}, expansion_order, lamb_helmholtz) where TF

    # try preallocating one container to be reused
    Ts = zeros(TF, length_Ts(expansion_order))
    eimϕs = zeros(TF, 2, expansion_order+1)
    weights_tmp_1 = initialize_expansion(expansion_order, TF)
    weights_tmp_2 = initialize_expansion(expansion_order, TF)

    # loop over branches
    for i_branch in 1:length(tree.branches)
        branch = tree.branches[i_branch]
        if branch.n_branches > 0 # if branch is a non-leaf target
            for i_child_branch in branch.branch_index
                child_branch = tree.branches[i_child_branch]
                child_expansion = view(tree.expansions, :, :, :, i_child_branch)
                branch_expansion = view(tree.expansions, :, :, :, i_branch)
                local_to_local!(child_expansion, child_branch, branch_expansion, branch, weights_tmp_1, weights_tmp_2, Ts, eimϕs, ηs_mag, Hs_π2, expansion_order, lamb_helmholtz)
            end
        end
    end
end

function downward_pass_singlethread_2!(tree::Tree{TF,<:Any}, systems, expansion_order, lamb_helmholtz, derivatives_switches, gradient_n_m) where TF

    harmonics = initialize_harmonics(expansion_order)
    # loop over systems
    for (i_system, system) in enumerate(systems)
        evaluate_local!(system, i_system, tree, harmonics, gradient_n_m, expansion_order, lamb_helmholtz, derivatives_switches)
    end

end

@inline function downward_pass_singlethread!(tree, systems, expansion_order, lamb_helmholtz, derivatives_switches, gradient_n_m)

    downward_pass_singlethread_1!(tree, expansion_order, lamb_helmholtz)

    downward_pass_singlethread_2!(tree, systems, expansion_order, lamb_helmholtz, derivatives_switches, gradient_n_m)

end

assign_l2l!(assignments, branches, level_index, n_per_thread, n_threads) = assign_m2m!(assignments, branches, level_index, n_per_thread, n_threads)

function execute_l2l!(expansions, branches, assignment, weights_tmp_1, weights_tmp_2, Ts, eimϕs, ηs_mag, Hs_π2, expansion_order, lamb_helmholtz)
    for i_branch in assignment
        # extract branch
        parent_branch = branches[i_branch]
        parent_expansion = view(expansions, :, :, :, i_branch)

        # loop over children
        for i_child in parent_branch.branch_index
            child_branch = branches[i_child]
            child_expansion = view(expansions, :, :, :, i_child)
            local_to_local!(child_expansion, child_branch, parent_expansion, parent_branch, weights_tmp_1, weights_tmp_2, Ts, eimϕs, ηs_mag, Hs_π2, expansion_order, lamb_helmholtz)
        end
    end
end

function downward_pass_multithread_1!(tree::Tree{TF,<:Any}, expansion_order, lamb_helmholtz, n_threads) where TF

    # extract containers
    branches = tree.branches
    levels_index = tree.levels_index
    expansions = tree.expansions

    # try preallocating one set of containers to be reused
    Ts = [zeros(TF, length_Ts(expansion_order)) for _ in 1:n_threads]
    eimϕs = [zeros(TF, 2, expansion_order+1) for _ in 1:n_threads]
    weights_tmp_1 = [initialize_expansion(expansion_order, TF) for _ in 1:n_threads]
    weights_tmp_2 = [initialize_expansion(expansion_order, TF) for _ in 1:n_threads]
    assignments = fill(1:0, n_threads)

    # iterate over levels
    for i_level in 1:length(levels_index)
        level_index = levels_index[i_level]

        # load balance
        n_interactions = 0
        for i_branch in level_index
            n_interactions += branches[i_branch].n_branches
        end
        n_per_thread, rem = divrem(n_interactions, n_threads)
        rem > 0 && (n_per_thread += 1)

        # if there are too many threads, we'll actually hurt performance
        n_per_thread < MIN_NPT_L2L && (n_per_thread = MIN_NPT_L2L)

        # how many threads are actually needed
        assign_l2l!(assignments, branches, level_index, n_per_thread, n_threads)

        # assign thread start branches
        Threads.@threads :static for i_task in 1:n_threads
            
            # get assignment
            assignment = assignments[i_task]

            # execute assignment
            execute_l2l!(expansions, branches, assignment, weights_tmp_1[i_task], weights_tmp_2[i_task], Ts[i_task], eimϕs[i_task], ηs_mag, Hs_π2, expansion_order, lamb_helmholtz) 
        end
    end
end

function downward_pass_multithread_2!(tree::Tree{TF,<:Any}, systems, derivatives_switches, expansion_order, lamb_helmholtz, n_threads) where TF
    
    #--- load balance ---#

    leaf_index = tree.leaf_index
    branches = tree.branches

    leaf_assignments = fill(1:0, length(systems), n_threads)
    for (i_system, system) in enumerate(systems)

        # total number of bodies
        n_bodies = get_n_bodies(system)

        # number of bodies per thread
        n_per_thread, rem = divrem(n_bodies, n_threads)
        rem > 0 && (n_per_thread += 1)

        # if there are too many threads, we'll actually hurt performance
        n_per_thread < MIN_NPT_L2B && (n_per_thread = MIN_NPT_L2B)

        # create chunks
        i_start = 1
        i_thread = 1
        n_bodies = 0
        for (i_end,i_leaf) in enumerate(leaf_index)
            n_bodies += length(branches[i_leaf].bodies_index[i_system])
            if n_bodies >= n_per_thread
                leaf_assignments[i_system,i_thread] = i_start:i_end
                i_start = i_end+1
                i_thread += 1
                n_bodies = 0
            end
        end
        i_thread <= n_threads && (leaf_assignments[i_system,i_thread] = i_start:length(leaf_index))
    end

    #--- preallocate memory ---#

    harmonics = [initialize_harmonics(expansion_order, TF) for _ in 1:n_threads]
    gradient_n_m = [initialize_gradient_n_m(expansion_order, TF) for _ in 1:n_threads]

    #--- compute multipole expansion coefficients ---#

    for (i_system, system) in enumerate(systems)
        Threads.@threads :static for i_thread in 1:n_threads
            leaf_assignment = leaf_assignments[i_system,i_thread]
            these_harmonics = harmonics[i_thread]
            these_gradient_n_m = gradient_n_m[i_thread]
            for i_leaf in leaf_assignment
                i_branch = leaf_index[i_leaf]
                evaluate_local!(system, i_system, tree, i_branch, these_harmonics, these_gradient_n_m, expansion_order, lamb_helmholtz, derivatives_switches)
            end
        end
    end
end

function downward_pass_multithread!(tree, systems, derivatives_switch, expansion_order, lamb_helmholtz, n_threads)

    # m2m translation
    downward_pass_multithread_1!(tree, expansion_order, lamb_helmholtz, n_threads)

    # local to body interaction
    downward_pass_multithread_2!(tree, systems, derivatives_switch, expansion_order, lamb_helmholtz, n_threads)

end

#--- running FMM ---#

# warn if lamb_helmholtz = true and scalar_potential=true

function warn_scalar_potential_with_lh(switch::DerivativesSwitch{PS,<:Any,<:Any}, lamb_helmholtz) where PS
    success = !(PS && lamb_helmholtz)
end

function warn_scalar_potential_with_lh(derivatives_switches::Tuple, lamb_helmholtz::Bool)
    success = true
    for switch in derivatives_switches
        success = success && warn_scalar_potential_with_lh(switch, lamb_helmholtz)
    end
    if WARNING_FLAG_LH_POTENTIAL[] && !success
        @warn "\nScalar potential was requested for a source system inducing a vector potential; this may result in nonsensical scalar potential predictions.\nIf you really need the scalar_potential, check which system results in has_vector_potential(system)==true, and remove it."
    end
end

@inline function to_tuple(input::Tuple)
    return input
end

@inline function to_tuple(input)
    return (input,)
end

@inline function to_vector(input::AbstractVector, n)
    return SVector{n}(input)
end

@inline function to_vector(input::Number, n)
    return SVector{n}(input for _ in 1:n)
end

@inline function to_vector(input::Tuple, n)
    return SVector{n}(input...)
end

fmm!(system, cache::Cache=Cache(to_tuple(system), to_tuple(system)); leaf_size=20, optargs...) = fmm!(system, system, cache; leaf_size_source=leaf_size, leaf_size_target=leaf_size, optargs...)

function fmm!(target_systems, source_systems, cache::Cache=Cache(to_tuple(target_systems), to_tuple(source_systems)); optargs...)
    # promote arguments to Tuples
    target_systems = to_tuple(target_systems)
    source_systems = to_tuple(source_systems)

    return fmm!(target_systems, source_systems, cache; optargs...)
end

"""
    fmm!(target_systems::Tuple, source_systems::Tuple; optargs...)

Dispatches `fmm!` with automatic tree creation.

**Arguments**

- `target_systems::Union{Tuple, {UserDefinedSystem}}`: either a system object for which compatibility functions have been overloaded, or a tuple of system objects for which compatibility functions have been overloaded
- `source_systems::Union{Tuple, {UserDefinedSystem}}`: either a system object for which compatibility functions have been overloaded, or a tuple of system objects for which compatibility functions have been overloaded

Note: a convenience function `fmm!(system)` is provided, which is equivalent to `fmm!(system, system)`. This is for situations where all system(s) act on all other systems, including themselves.

**Optional Arguments: Allocation**

- `target_buffers::Vector{<:Any}`: buffers for target systems; if not provided, buffers are allocated using [`FastMultipole.allocate_buffers`](@ref)
- `target_small_buffers::Vector{<:Any}`: small buffers for target systems; if not provided, small buffers are allocated using [`FastMultipole.allocate_small_buffers`](@ref)
- `source_buffers::Vector{<:Any}`: buffers for source systems; if not provided, buffers are allocated using [`FastMultipole.allocate_buffers`](@ref)
- `source_small_buffers::Vector{<:Any}`: small buffers for source systems; if not provided, small buffers are allocated using [`FastMultipole.allocate_small_buffers`](@ref)

**Optional Arguments: Tuning Parameters**

- `expansion_order::Int`: order of multipole expansions; default is 5
- `multipole_acceptance::Float64`: acceptance criterion for multipole expansions; default is 0.4
- `leaf_size_target::Union{Nothing,Int}`: leaf size for target systems; if not provided, the minimum of the source leaf sizes is used
- `leaf_size_source::Union{Nothing,Int}`: leaf size for source systems; if not provided, the default leaf size is used
- `error_tolerance::Union{Nothing,ErrorMethod}`: error tolerance for multipole to local translations; if not provided, no error treatment is performed

**Optional Arguments: Tree Options**

- `shrink_recenter::Bool`: whether to shrink and recenter branches around their bodies, accounting for finite body radius; default is `true`
- `interaction_list_method::InteractionListMethod`: method for building interaction lists; default is `SelfTuningTreeStop()`

**Optional Arguments: Additional Options**

- `farfield::Bool`: whether to compute farfield interactions; default is `true`
- `nearfield::Bool`: whether to compute nearfield interactions; default is `true`
- `self_induced::Bool`: whether to compute self-induced interactions; default is `true`
- `upward_pass::Bool`: whether to perform the upward pass; default is `true`
- `horizontal_pass::Bool`: whether to perform the horizontal pass; default is `true`
- `downward_pass::Bool`: whether to perform the downward pass; default is `true`
- `scalar_potential::Union{Bool,AbstractVector{Bool}}`: whether to compute the scalar potential; default is `false`
- `gradient::Union{Bool,AbstractVector{Bool}}`: whether to compute the vector field; default is `true`
- `hessian::Union{Bool,AbstractVector{Bool}}`: whether to compute the vector gradient; default is `false`

"""
function fmm!(target_systems::Tuple, source_systems::Tuple, cache::Cache=Cache(target_systems, source_systems);
    leaf_size_target=nothing,
    leaf_size_source=default_leaf_size(source_systems),
    expansion_order=5,
    error_tolerance=nothing,
    shrink_recenter=true,
    interaction_list_method::InteractionListMethod=SelfTuningTreeStop(),
    optargs...
)

    # get float type
    TF = get_type(target_systems, source_systems)

    # promote leaf_size to vector
    leaf_size_source = to_vector(leaf_size_source, length(source_systems))
    leaf_size_target = to_vector(isnothing(leaf_size_target) ? minimum(leaf_size_source) : leaf_size_target, length(target_systems))

    # create trees
    t_target_tree = @elapsed target_tree = Tree(target_systems, true, TF; buffers=cache.target_buffers, small_buffers=cache.target_small_buffers, expansion_order, leaf_size=leaf_size_target, shrink_recenter, interaction_list_method)
    t_source_tree = @elapsed source_tree = Tree(source_systems, false, TF; buffers=cache.source_buffers, small_buffers=cache.source_small_buffers, expansion_order, leaf_size=leaf_size_source, shrink_recenter, interaction_list_method)

    # println("  Target tree: $t_target_tree")
    # println("  Source tree: $t_source_tree")

    return fmm!(target_systems, target_tree, source_systems, source_tree; expansion_order, leaf_size_source, error_tolerance, t_source_tree, t_target_tree, interaction_list_method, optargs...)
end

function fmm!(target_systems::Tuple, target_tree::Tree, source_systems::Tuple, source_tree::Tree;
    leaf_size_source=default_leaf_size(source_systems), multipole_acceptance=0.4,
    scalar_potential=false, gradient=true, hessian=false,
    farfield=true, nearfield=true, self_induced=true,
    interaction_list_method::InteractionListMethod=SelfTuningTreeStop(),
    t_source_tree=0.0, t_target_tree=0.0,
    optargs...
)

    # promote derivative arguments to a vector
    scalar_potential = to_vector(scalar_potential, length(target_systems))
    gradient = to_vector(gradient, length(target_systems))
    hessian = to_vector(hessian, length(target_systems))
    
    # assemble derivatives switch
    derivatives_switches = DerivativesSwitch(scalar_potential, gradient, hessian, target_systems)

    # create interaction lists
    build_list_t = @elapsed m2l_list, direct_list = build_interaction_lists(target_tree.branches, source_tree.branches, leaf_size_source, multipole_acceptance, farfield, nearfield, self_induced, interaction_list_method)
    t_lists = @elapsed begin
        m2l_list = sort_by_target(m2l_list, target_tree.branches)
        direct_list = sort_by_target(direct_list, target_tree.branches)
    end
    println("  Build interaction lists: $build_list_t")
    println("  Sort Interaction lists: $t_lists")

    error()

    # run fmm
    return fmm!(target_systems, target_tree, source_systems, source_tree, leaf_size_source, m2l_list, direct_list, derivatives_switches, interaction_list_method; multipole_acceptance, t_source_tree, t_target_tree, t_lists, optargs...)
end

function fmm!(target_systems::Tuple, target_tree::Tree, source_systems::Tuple, source_tree::Tree, leaf_size_source, m2l_list, direct_list, derivatives_switches::Tuple, interaction_list_method::InteractionListMethod;
    expansion_order=5, error_tolerance=nothing,
    upward_pass::Bool=true, horizontal_pass::Bool=true, downward_pass::Bool=true,
    horizontal_pass_verbose::Bool=false,
    reset_target_tree::Bool=true, reset_source_tree::Bool=true,
    nearfield_device::Bool=false,
    tune=false, update_target_systems=true, multipole_acceptance=0.5,
    t_source_tree=0.0, t_target_tree=0.0, t_lists=0.0,
    silence_warnings=false,
)

    #--- check if lamb-helmholtz decomposition is required ---#

    lamb_helmholtz = has_vector_potential(source_systems)

    #--- check for datarace condition ---#

    # if Threads.nthreads() > 1 && typeof(interaction_list_method) <: Union{InteractionListMethod{SortBySource}, SelfTuning}
    #     throw(ArgumentError("InteractionListMethod $interaction_list_method is not thread-safe; use <:InteractionListMethod{SortByTarget} instead, and avoid `SelfTuning`."))
    # end

    #--- silence warnings ---#

    if silence_warnings
        WARNING_FLAG_LEAF_SIZE[] = false
        WARNING_FLAG_PMAX[] = false
        WARNING_FLAG_ERROR[] = false
        WARNING_FLAG_SCALAR_POTENTIAL[] = false
        WARNING_FLAG_VECTOR_POTENTIAL[] = false
        WARNING_FLAG_gradient[] = false
        WARNING_FLAG_hessian[] = false
        WARNING_FLAG_STRENGTH[] = false
        WARNING_FLAG_B2M[] = false
        WARNING_FLAG_DIRECT[] = false
        WARNING_FLAG_LH_POTENTIAL[] = false
    else
        WARNING_FLAG_LEAF_SIZE[] = true
        WARNING_FLAG_PMAX[] = true
        WARNING_FLAG_ERROR[] = true
        WARNING_FLAG_SCALAR_POTENTIAL[] = true
        WARNING_FLAG_VECTOR_POTENTIAL[] = true
        WARNING_FLAG_gradient[] = true
        WARNING_FLAG_hessian[] = true
        WARNING_FLAG_STRENGTH[] = true
        WARNING_FLAG_B2M[] = true
        WARNING_FLAG_DIRECT[] = true
        WARNING_FLAG_LH_POTENTIAL[] = true
    end

    #--- estimate influence for relative error tolerance ---#

    # estimate_influence!(target_systems, target_tree, source_systems, source_tree, error_tolerance; nearfield_device)

    # check if systems are empty
    n_target_bodies = get_n_bodies(target_systems)
    n_source_bodies = get_n_bodies(source_systems)

    if n_target_bodies > 0 && n_source_bodies > 0

        # check that lamb_helmholtz and ScalarPotential are not both true
        warn_scalar_potential_with_lh(derivatives_switches, lamb_helmholtz)

        # wrap lamb_helmholtz in Val
        lamb_helmholtz = Val(lamb_helmholtz)

        # increment the expansion order if error_tolerance !== nothing
        # error_check = !(isnothing(error_tolerance))

        # precompute y-axis rotation by π/2 matrices (if not already done)
        update_Hs_π2!(Hs_π2, expansion_order)

        # precompute y-axis Wigner matrix normalization (if not already done)
        update_ζs_mag!(ζs_mag, expansion_order)
        update_ηs_mag!(ηs_mag, expansion_order)

        # precompute error prediction normalization (if not already done)
        update_M̃!(M̃, expansion_order)
        update_L̃!(L̃, expansion_order)

        # available threads
        n_threads = Threads.nthreads()

        # reset trees
        if reset_target_tree
            reset_expansions!(target_tree)
        end
        if reset_source_tree
            reset_expansions!(source_tree)
        end

        # declare error success
        error_success = true

        # begin FMM
        if nearfield_device # use GPU

            # allow nearfield_device! to be called concurrently with upward and horizontal passes
            t1 = Threads.@spawn nearfield && nearfield_device!(target_systems, target_tree, derivatives_switches, source_systems, source_tree, direct_list)
            n_threads_multipole = n_threads == 1 ? n_threads : n_threads - 1
            t2 = Threads.@spawn begin
                    upward_pass && upward_pass_multithread!(source_tree.branches, source_systems, expansion_order, lamb_helmholtz, source_tree.levels_index, source_tree.leaf_index, n_threads_multipole)
                    horizontal_pass && length(m2l_list) > 0 && horizontal_pass_multithread!(target_tree.branches, source_tree.branches, m2l_list, lamb_helmholtz, expansion_order, error_tolerance, n_threads_multipole)
	                downward_pass && downward_pass_multithread_1!(target_tree.branches, expansion_order, lamb_helmholtz, target_tree.levels_index, n_threads_multipole)
                end

            fetch(t1)
            fetch(t2)

            # local to body interaction
            downward_pass && downward_pass_multithread_2!(target_tree.branches, target_systems, derivatives_switches, Pmax, lamb_helmholtz, tree.leaf_index, n_threads)

        else # use CPU

            # single threaded
            if n_threads == 1

                # perform nearfield calculations
                t_direct = nearfield_singlethread!(target_tree.buffers, target_tree.branches, source_systems, source_tree.buffers, source_tree.branches, derivatives_switches, direct_list)

                # check number of interactions
                if tune
                    n_interactions = 0
                    for i_source_system in eachindex(source_systems)
                        for (i_target, i_source) in direct_list
                            source_branch = source_tree.branches[i_source]
                            target_branch = target_tree.branches[i_target]
                            n_interactions += source_branch.n_bodies[i_source_system] * sum(target_branch.n_bodies)
                        end
                        t_direct[i_source_system] /= n_interactions
                    end
                end

                # farfield computations
                t_up = 0.0
                if upward_pass
                    t_up = @elapsed upward_pass_singlethread!(source_tree, source_systems, expansion_order, lamb_helmholtz)
                end

                t_m2l = 0.0
                Pmax = 0
                if horizontal_pass
                    t_m2l = @elapsed Pmax, error_success = horizontal_pass_singlethread!(target_tree, source_tree, m2l_list, lamb_helmholtz, expansion_order, error_tolerance; verbose=horizontal_pass_verbose)
                end
                if !error_success
                    Pmax += 1
                end

                # @time downward_pass && downward_pass_singlethread!(tree.branches, tree.leaf_index, systems, expansion_order, lamb_helmholtz, derivatives_switches)
                t_dp = 0.0
                if downward_pass
                    t_dp = @elapsed downward_pass_singlethread_1!(target_tree, expansion_order, lamb_helmholtz)
                    t_dp += @elapsed gradient_n_m = initialize_gradient_n_m(expansion_order, eltype(target_tree.branches[1]))
                    t_dp += @elapsed downward_pass_singlethread_2!(target_tree, target_tree.buffers, expansion_order, lamb_helmholtz, derivatives_switches, gradient_n_m)
                end

                # copy results to target systems
                update_target_systems && buffer_to_target!(target_systems, target_tree, derivatives_switches)

                # finish autotuning
                if tune

                    if length(m2l_list) > 0
                        #--- compute optimal leaf_size_source ---#

                        # t per m2l transformation
                        # t_m2l += t_up + t_dp + t_target_tree + t_source_tree + t_lists
                        t_m2l /= length(m2l_list)

                        # t_per_interaction * LS^2 = t_per_m2l
                        leaf_size_source = SVector{length(source_systems),Int}(Int(ceil(sqrt(t_m2l / t_direct[i]))) for i in eachindex(source_systems))
                    else
                        # make leaf size smaller so that some m2l operations exist
                        leaf_size_source = max.(leaf_size_source .>> 1, Ref(1))
                    end
                    expansion_order = Pmax
                end

            # multithreaded
            else

                # perform nearfield calculations
                t_direct = nearfield_multithread!(target_tree.buffers, target_tree.branches, source_systems, source_tree.buffers, source_tree.branches, derivatives_switches, direct_list, interaction_list_method, n_threads)
                # println("  Direct interactions: ", t_direct[1])
                # check number of interactions
                if tune
                    for i_source_system in eachindex(source_systems)
                        n_interactions_local = zeros(Int, n_threads)

                        n_points = length(direct_list)
                        n_per_thread, rem = divrem(n_points,n_threads)
                        n = n_per_thread + (rem > 0)
                        assignments = 1:n:n_points

                        Threads.@threads :static for i_assignment in length(assignments)
                            i_start = assignments[i_assignment]
                            i_end = min(i_start + n - 1, n_points)

                            for idx in i_start:i_end
                                i_target, i_source = direct_list[idx]
                                source_branch = source_tree.branches[i_source]
                                target_branch = target_tree.branches[i_target]
                                n_interactions_local[i_assignment] += source_branch.n_bodies[i_source_system] * sum(target_branch.n_bodies)
                            end
                        end

                        n_interactions = sum(n_interactions_local)
                        t_direct[i_source_system] /= n_interactions
                    end
                end

                # farfield computations
                t_up = 0.0
                if upward_pass
                    t_up = @elapsed upward_pass_multithread!(source_tree, source_systems, expansion_order, lamb_helmholtz, n_threads)
                    # println("  Upward pass: $t_up")
                end

                Pmax = 0
                if horizontal_pass
                    t_m2l = @elapsed Pmax, error_success = horizontal_pass_multithread!(target_tree, source_tree, m2l_list, lamb_helmholtz, expansion_order, error_tolerance, interaction_list_method, n_threads)
                    # println("  Horizontal pass: $t_m2l")
                end
                if !error_success
                    Pmax += 1
                end
                
                # @time downward_pass && downward_pass_singlethread!(tree.branches, tree.leaf_index, systems, expansion_order, lamb_helmholtz, derivatives_switches)
                t_dp = 0.0
                if downward_pass
                    t_dp = @elapsed downward_pass_multithread!(target_tree, target_tree.buffers, derivatives_switches, expansion_order, lamb_helmholtz, n_threads)
                    # println("  Downward pass: $t_dp")
                end
                
                # copy results to target systems
                update_target_systems && buffer_to_target!(target_systems, target_tree, derivatives_switches)

                # finish autotuning
                if tune

                    if length(m2l_list) > 0
                        #--- compute optimal leaf_size_source ---#

                        # t per m2l transformation
                        # t_m2l += t_up + t_dp + t_target_tree + t_source_tree + t_lists
                        t_m2l /= length(m2l_list)

                        # t_per_interaction * LS^2 = t_per_m2l
                        leaf_size_source = SVector{length(source_systems),Int}(Int(ceil(sqrt(t_m2l / t_direct[i]))) for i in eachindex(source_systems))
                    else
                        # make leaf size smaller so that some m2l operations exist
                        leaf_size_source = max.(leaf_size_source .>> 1, Ref(1))
                    end
                    expansion_order = Pmax
                end
            end

        end

    else

        @warn "fmm! called but either sources or targets are empty; foregoing calculation"
        error_success = true

    end

    # pack up optimal arguments for next fmm! call
    optimized_args = (
                       leaf_size_source = leaf_size_source,
                       expansion_order = max(expansion_order, 1),
                       multipole_acceptance = multipole_acceptance,
                      )

    cache = Cache(;
             target_buffers = target_tree.buffers,
             target_small_buffers = target_tree.small_buffers,
             source_buffers = source_tree.buffers,
             source_small_buffers = source_tree.small_buffers,
            )

    return optimized_args, cache, target_tree, source_tree, m2l_list, direct_list, derivatives_switches, error_success
end

#--- estimate influence for relative error tolerance ---#

# @inline function get_influence(system::Matrix, j, ::Union{PowerRelativeGradient, RotatedCoefficientsRelativeGradient})
#     vx, vy, vz = get_gradient(system, j)
#     return sqrt(vx*vx + vy*vy + vz*vz)
# end

# @inline function get_influence(system::Matrix, j, ::PowerRelativePotential)
#     return get_scalar_potential(system, j)
# end

# function estimate_influence!(target_systems, target_tree, source_systems, source_tree, error_tolerance::Union{Nothing, AbsoluteError}; optargs...)
#     return nothing
# end

# function estimate_influence!(target_systems, target_tree, source_systems, source_tree, error_tolerance::RelativeError; nearfield_device=false, shrink_recenter=true)

#     #--- low-order estimate for relative error tolerance ---#

#     _, _, estimate_tree, _ = fmm!(target_systems, source_systems; 
#         scalar_potential = true, gradient = true, hessian = false,
#         leaf_size_source = to_vector(5, length(source_systems)),
#         expansion_order = 3, multipole_acceptance = 0.6,
#         error_tolerance = nothing, shrink_recenter, nearfield_device,
#         update_target_systems = false,
#         silence_warnings = true
#     )

#     #--- update target buffers ---#

#     for (i_system, (buffer, estimate)) in enumerate(zip(target_tree.buffers, estimate_tree.buffers))
        
#         # loop over estimate bodies
#         for j_estimate in 1:get_n_bodies(buffer)

#             # get estimate body index
#             j_system = sorted_index_2_unsorted_index(j_estimate, i_system, estimate_tree)

#             # get buffer index
#             j_buffer = unsorted_index_2_sorted_index(j_system, i_system, target_tree)

#             # check that we have the right body
#             @assert get_position(buffer, j_buffer) == get_position(estimate, j_estimate)

#             # update influence
#             set_scalar_potential!(buffer, j_buffer, get_scalar_potential(estimate, j_estimate))
#             set_gradient!(buffer, j_buffer, get_gradient(estimate, j_estimate))

#         end        
#     end

#     #--- update target branches ---#

#     # loop over target buffers
#     for (i_system, system) in enumerate(target_tree.buffers)

#         # loop over target branches
#         for (i, branch) in enumerate(target_tree.branches)
            
#             # extract branch info
#             n_bodies = branch.n_bodies
#             bodies_index = branch.bodies_index
#             n_branches = branch.n_branches
#             branch_index = branch.branch_index
#             i_parent = branch.i_parent
#             i_leaf = branch.i_leaf
#             source_center = branch.source_center
#             target_center = branch.target_center
#             source_radius = branch.source_radius
#             target_radius = branch.target_radius
#             source_box = branch.source_box
#             target_box = branch.target_box
#             max_influence = branch.max_influence

#             # loop over bodies 
#             for j in branch.bodies_index[i_system]
#                 influence = get_influence(system, j, error_tolerance)
#                 max_influence = max(max_influence, influence)
#             end

#             # replace branch
#             target_tree.branches[i] = typeof(branch)(n_bodies, bodies_index, n_branches, branch_index, i_parent, i_leaf, 
#                 source_center, target_center, source_radius, target_radius, source_box, target_box, max_influence)
#         end
#     end

#     #--- reset buffers ---#

#     reset!(target_tree.buffers)
#     reset!(target_tree.small_buffers)

#     if DEBUG[]
#         maxinf = [target_tree.branches[j].max_influence * (j in target_tree.leaf_index) for j in 1:length(target_tree.branches)]
#         i_max = findfirst(x -> x == maximum(maxinf), maxinf)
#         @info "Max influence found at branch $(i_max): bodies index: $(target_tree.branches[i_max].bodies_index), max_influence: $(maximum(maxinf))"

#         @show mean(maxinf) maximum(maxinf) minimum(maxinf)
#     end
# end