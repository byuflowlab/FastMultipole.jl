"""
    `extra_farfield!(target_tree::Tree, source_tree::Tree, source_systems::Tuple, m2l_list::Vector{NTuple{2,Int32}}, derivatives_switches::Tuple)`

Perform extra farfield calculations. Designed with semi-infinite panels in mind, where it doesn't make sense to generate multipole expansions for them, but their strengths are tied to another system.

**Arguments**

- `target_tree::Tree`: octree for targets
- `source_tree::Tree`: octree for sources
- `source_systems::Tuple`: tuple containing the source system data
- `m2l_list::Vector{NTuple{2,Int32}}`: vector of tuples (`(i_target, i_source)`) indicating which target and source branches interact in the horizontal pass
- `derivatives_switches::Tuple`: tuple of `DerivativesSwitch` objects indicating which derivatives to compute

"""
function extra_farfield!(target_tree::Tree, source_tree::Tree, source_systems::Tuple, m2l_list, derivatives_switches::Tuple)
    # loop over source systems
    for (i_source_system, source_system) in enumerate(source_systems)
        
        # extract source system buffer
        source_buffer = source_tree.buffers[i_source_system]

        # loop over target systems
        for (i_target_buffer, target_buffer) in enumerate(target_tree.buffers)

            # extract derivatives switch
            switch = derivatives_switches[i_target_buffer]

            # call the extra farfield function for this source system
            for (i_target_branch, i_source_branch) in m2l_list

                # extract source branch
                source_branch = source_tree.branches[i_source_branch]
                
                # check if source system has bodies in this interaction
                source_bodies_index = source_branch.bodies_index[i_source_system]
                if length(source_bodies_index) > 0
                    
                    # check if there any targets
                    target_branch = target_tree.branches[i_target_branch]
                    target_bodies_index = target_branch.bodies_index[i_target_buffer]
                    if length(target_bodies_index) > 0

                        # perform extra farfield calculations
                        extra_farfield!(target_buffer, target_bodies_index, source_system, source_buffer, source_bodies_index, switch)
                    end
                end
            end
        end
    end
end 

"""
    `extra_farfield!(target_buffer::Matrix{Float64}, target_bodies_index::UnitRange, source_system, source_buffer::Matrix{Float64}, source_bodies_index, switch::DerivativesSwitch)`

Perform extra farfield calculations for a specific source system. Should be overloaded for each user-defined source system for which extra farfield influence is desired.

**Arguments**

- `target_buffer::Matrix{Float64}`: buffer for target data
- `target_bodies_index`: indices of the bodies in the target system to consider
- `source_system`: the user-defined source system object
- `source_buffer::Matrix{Float64}`: buffer for source data
- `source_bodies_index`: indices of the bodies in the source system to consider
- `switch::DerivativesSwitch`: which derivatives should be for this target system

"""
function extra_farfield!(target_buffer, target_bodies_index, source_system, source_buffer, source_bodies_index, switch)
    return nothing
end