# using Profile
# using ProfileView

include("../test/gravitational.jl")

function generate_gravitational(seed, n_bodies; radius_factor=0.1)
    Random.seed!(123)
    bodies = rand(8,n_bodies)
    # bodies[1:3,3] .=  0.811770914672987, 0.15526131946379113, 0.30656077208169424
    # bodies[1:3,3] .=   0.7427186184997012, 0.2351893322824516, 0.3380666354208596
    bodies[4,:] ./= (n_bodies^(1/3)*2)
    bodies[4,:] .*= radius_factor
    system = Gravitational(bodies)
end

# generate system
n_bodies = 1_000_000
leaf_size = 10
sys = generate_gravitational(123, n_bodies)

# direct!(sys; scalar_potential=true)
# phi_direct = sys.potential[1,:]

# sys.potential .= 0.0 # reset potential
# fmm!(sys; scalar_potential=true, leaf_size=leaf_size, interaction_list_method=FastMultipole.SelfTuningTreeStop())
# phi_fmm = sys.potential[1,:]

# diff = phi_direct - phi_fmm
# println("Max difference: ", maximum(abs.(diff)))

# profile tree creation
# @profview tree = Tree(sys, true)
# @profview tree = Tree(sys, true)
# FastMultipole.DEBUG_COUNTER[1] = 0
# tree = Tree(sys, true; leaf_size=SVector{1}(1));
@time tree = Tree(sys, true; leaf_size=SVector{1}(leaf_size));#, interaction_list_method=FastMultipole.SelfTuningTreeStop());
println("\n\n")
# sys.potential .= 0.0 # reset potential
@time tree = Tree(sys, true; leaf_size=SVector{1}(leaf_size));#, interaction_list_method=FastMultipole.SelfTuningTreeStop());
println("done.")

function test_tree_creation(tree, n_bodies)
    for (i, levels_index) in enumerate(tree.levels_index)
        println("Level $i: ", length(levels_index), " branches")
        bodies = zeros(Bool, n_bodies)
        for branch in tree.branches[levels_index]
            for i_body in branch.bodies_index[1]
                @assert bodies[i_body] == false "Body $i_body already counted in branch $branch"
                bodies[i_body] = true
            end
        end
    end
end

function test_tree_creation_2(tree)
    for i in 2:length(tree.branches)
        branch = tree.branches[i]
        parent_branch = tree.branches[branch.i_parent]
        for i_body in branch.bodies_index[1]
            @assert i_body in parent_branch.bodies_index[1] "Body $i_body in branch $branch not found in parent branch $parent_branch"
        end
    end
end

function test_tree_creation_3(tree, n_bodies)
    bodies = zeros(Bool, n_bodies)
    for i_leaf in tree.leaf_index
        branch = tree.branches[i_leaf]
        for i_body in branch.bodies_index[1]
            @assert bodies[i_body] == false "Body $i_body already counted in branch $branch"
            bodies[i_body] = true
        end
    end
    @assert sum(bodies) == n_bodies "Not all bodies counted in leaves"
end

test_tree_creation(tree, n_bodies)
test_tree_creation_2(tree)
test_tree_creation_3(tree, n_bodies)