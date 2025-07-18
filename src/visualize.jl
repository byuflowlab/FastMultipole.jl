# function visualize_bodies(name, system, tree; i_system=nothing)
#     n_bodies = get_n_bodies(system)
#     body_locations = zeros(3,n_bodies,1,1)
#     body_radii = zeros(n_bodies,1,1)
#     scalar_strength = zeros(n_bodies,1,1)
#     vector_strength = zeros(3,n_bodies,1,1)
#     scalar_potential = zeros(n_bodies,1,1)
#     vector_potential = zeros(3,n_bodies,1,1)
#     velocity = zeros(3,n_bodies,1,1)
#     body_indices = zeros(n_bodies,1,1)
#     unsorted_body_indices = zeros(n_bodies,1,1)
#     for i in 1:n_bodies
#         body_locations[:,i,1,1] .= system[i,Position()]
#         body_radii[i,1,1] = system[i,Radius()]
#         if typeof(system[i,Strength()]) <: AbstractArray
#             vector_strength[:,i,1,1] .= system[i,Strength()]
#         else
#             scalar_strength[i,1,1] = system[i,Strength()]
#         end
#         scalar_potential[i,1,1] = system[i,ScalarPotential()]
#         velocity[:,i,1,1] .= system[i,Velocity()]
#         unsorted_body_indices[i,1,1] = Float64(i)
#         if isnothing(i_system)
#             body_indices[i,1,1] = Float64(unsorted_index_2_sorted_index(i, tree))
#         else
#             body_indices[i,1,1] = Flaot64(unsorted_index_2_sorted_index(i, i_system, tree))
#         end
#     end
#     vtk_grid(name*"_bodies", body_locations) do vtk
#         vtk["radius"] = body_radii
#         if typeof(system[1,Strength()]) <: AbstractArray
#             vtk["vector strength"] = vector_strength
#         else
#             vtk["scalar strength"] = scalar_strength
#         end
#         vtk["scalar potential"] = scalar_potential
#         vtk["velocity"] = velocity
#         vtk["body index"] = body_indices
#         vtk["unsorted body index"] = unsorted_body_indices
#     end
# end

function visualize_bodies(name, system, tree, ::Nothing; i_system=nothing)
    visualize_bodies(name, system, tree; i_system)
end

function visualize_bodies(name, system, tree, probe_indices; i_system=nothing)
    visualize_bodies(name, system, tree; i_system)

    #####
    ##### probes
    #####
    n_probes = length(probe_indices)
    if n_probes > 0
        body_locations = zeros(3,n_probes,1,1)
        body_radii = zeros(n_probes,1,1)
        scalar_strength = zeros(n_probes,1,1)
        vector_strength = zeros(3,n_probes,1,1)
        for (i_probe,i_body) in enumerate(probe_indices)
            body_locations[:,i_probe,1,1] .= system[i_body,POSITION]
            body_radii[i_probe,1,1] = system[i_body,RADIUS]
            scalar_strength[i_probe,1,1] = system[i_body,SCALAR_STRENGTH]
            vector_strength[:,i_probe,1,1] .= system[i_body,VECTOR_STRENGTH]
        end
        vtk_grid(name*"_probes", body_locations) do vtk
            vtk["radius"] = body_radii
            vtk["scalar strength"] = scalar_strength
            vtk["vector strength"] = vector_strength
            vtk["index"] = probe_indices
        end
    end
end

function visualize_bodies(name_list, systems::Tuple, tree::Tree, probe_indices_list)
    for (i_system,(name, system, probe_indices)) in enumerate(zip(name_list, systems, probe_indices_list))
        visualize_bodies(name, system, tree, probe_indices; i_system)
    end
end

function visualize_bodies(name_list, systems::Tuple, tree::Tree, probe_indices_list::Nothing)
    for (i_system, (name, system)) in enumerate(zip(name_list, systems))
        visualize_bodies(name, system, tree; i_system)
    end
end

function visualize(name, system, tree; probe_indices=nothing, toggle_branches=true, toggle_bodies=false)
    #####
    ##### source branches
    #####
    if toggle_branches
        branches = tree.branches
        n_branches = length(branches)

        # source branches
        branch_locations = Array{Float64,4}(undef,3,n_branches,1,1)
        branch_radii = Array{Float64,3}(undef,n_branches,1,1)
        branch_indices = Array{Float64,3}(undef,n_branches,1,1)
        for (i,branch) in enumerate(branches)
            branch_locations[:,i,1,1] .= branch.center
            branch_radii[i,1,1] = branch.radius
            branch_indices[i,1,1] = Float64(i)
        end
        vtk_grid(name*"_source_branches", branch_locations) do vtk
            vtk["radius"] = branch_radii
            vtk["branch index"] = branch_indices
        end

        # target branches
        branch_locations = Array{Float64,4}(undef,3,n_branches,1,1)
        branch_radii = Array{Float64,3}(undef,n_branches,1,1)
        branch_indices = Array{Float64,3}(undef,n_branches,1,1)
        for (i,branch) in enumerate(branches)
            branch_locations[:,i,1,1] .= branch.center
            branch_radii[i,1,1] = branch.radius
            branch_indices[i,1,1] = Float64(i)
        end
        vtk_grid(name*"_target_branches", branch_locations) do vtk
            vtk["radius"] = branch_radii
            vtk["branch index"] = branch_indices
        end
    end

    #####
    ##### bodies
    #####
    if toggle_bodies
        visualize_bodies(name, system, tree, probe_indices)
    end

    return nothing
end

