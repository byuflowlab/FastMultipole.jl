# using FastMultipole
# using Random
# using LinearAlgebra
# using Test

# include("gravitational.jl")

#--- define influence function ---#

# just the scalar potential
function FastMultipole.influence!(influence, target_buffer, ::Gravitational, source_buffer)
    influence .= view(target_buffer, 4, :)
end

function FastMultipole.target_influence_to_buffer!(target_buffer, i_buffer, derivatives_switch, target_system::Gravitational, i_target)
    target_buffer[4, i_buffer] = target_system.potential[1, i_target]
end

function FastMultipole.value_to_strength!(source_buffer, ::Gravitational, i_body, value)
    source_buffer[5, i_body] = value
end

function FastMultipole.strength_to_value(strength, ::Gravitational)
    return strength[1]
end

function FastMultipole.buffer_to_system_strength!(system::Gravitational, i_body, source_buffer, i_buffer)
    position = system.bodies[i_body].position
    radius = system.bodies[i_body].radius
    strength = source_buffer[5, i_buffer]
    system.bodies[i_body] = eltype(system.bodies)(position, radius, strength)
end

@testset "Fast Gauss Seidel: self influence" begin

#--- create system ---#

n_bodies = 10
seed = 1234
system = generate_gravitational(seed, n_bodies)

direct!(system; scalar_potential=true, gradient=false)
phi_desired = system.potential[1, :]

#--- create FGS solver ---#

fgs = FastMultipole.FastGaussSeidel((system,), (system,); expansion_order=4, multipole_acceptance=0.5, leaf_size=30)

#--- check influence matrix ---#

influence_matrix_check = zeros(Float64, n_bodies, n_bodies)
source_buffer = fgs.source_tree.buffers[1]
for i in 1:n_bodies
    for j in 1:n_bodies
        r = norm(FastMultipole.get_position(source_buffer, i) .- FastMultipole.get_position(source_buffer, j))
        if r > 0.0
            influence_matrix_check[i, j] = 1.0 / (r*4*pi)
        end
    end
end

influence_matrix_fgs, _ = FastMultipole.get_matrix_vector(fgs.self_matrices, 1)
for i in 1:n_bodies
    for j in 1:n_bodies
        @test influence_matrix_fgs[i, j] ≈ influence_matrix_check[i, j] atol=1e-6
    end
end

end

@testset "Fast Gauss Seidel: all influence matrices" begin

#--- larger system ---#

n_bodies = 1000
seed = 1234
system = generate_gravitational(seed, n_bodies)

direct!(system; scalar_potential=true, gradient=false)
system.potential[1, :] .*= -1.0

#--- create FGS solver ---#

fgs = FastMultipole.FastGaussSeidel((system,), (system,); expansion_order=4, multipole_acceptance=0.5, leaf_size=100) # try with leaf_size=3 for sources with no non-self influence

#--- check self influence matrices ---#

for (i, i_leaf) in enumerate(fgs.source_tree.leaf_index)

    bodies_index = fgs.source_tree.branches[i_leaf].bodies_index[1]
    n_bodies = length(bodies_index)
    influence_matrix_check = zeros(Float64, n_bodies, n_bodies)
    source_buffer = fgs.source_tree.buffers[1]
    for (i,i_body) in enumerate(bodies_index)
        for (j,j_body) in enumerate(bodies_index)
            r = norm(FastMultipole.get_position(source_buffer, i_body) .- FastMultipole.get_position(source_buffer, j_body))
            if r > 0.0
                influence_matrix_check[i, j] = 1.0 / (r*4*pi)
            end
        end
    end

    influence_matrix_fgs, _ = FastMultipole.get_matrix_vector(fgs.self_matrices, i)
    for (i,i_body) in enumerate(bodies_index)
        for (j,j_body) in enumerate(bodies_index)
            @test influence_matrix_fgs[i, j] ≈ influence_matrix_check[i, j] atol=1e-6
        end
    end

end

#--- check non-self influence matrices ---#

direct_list = fgs.direct_list
target_buffer = fgs.target_tree.buffers[1]
source_buffer = fgs.source_tree.buffers[1]
source_list = [dl[2] for dl in direct_list]

for i_leaf in 1:length(fgs.source_tree.leaf_index)
    this_source = fgs.source_tree.leaf_index[i_leaf]
    if this_source in source_list
        mat, rhs = FastMultipole.get_matrix_vector(fgs.nonself_matrices, i_leaf)

        # build influence matrix manually to check against FGS
        source_indices = fgs.source_tree.branches[this_source].bodies_index[1]
        target_indices = Int[]
        for (i_target, j_source) in direct_list
            if j_source == this_source
                target_bodies_index = fgs.target_tree.branches[i_target].bodies_index[1]
                target_indices = vcat(target_indices, collect(target_bodies_index))
            end
        end

        # create test influence matrix
        influence_matrix_check = zeros(Float64, length(target_indices), length(source_indices))
        for (i_target, i_target_body) in enumerate(target_indices)
            for (j_source, j_source_body) in enumerate(source_indices)
                r = norm(FastMultipole.get_position(source_buffer, j_source_body) .- FastMultipole.get_position(target_buffer, i_target_body))
                if r > 0.0
                    influence_matrix_check[i_target, j_source] = 1.0 / (r*4*pi)
                end
            end
        end

        # test
        for (i_target, i_target_body) in enumerate(target_indices)
            for (j_source, j_source_body) in enumerate(source_indices)
                @test mat[i_target, j_source] ≈ influence_matrix_check[i_target, j_source] atol=1e-6
            end
        end
    else
        @assert fgs.nonself_matrices.sizes[i_leaf][1] == 0 "this influence matrix should be empty"
    end
end

end

@testset "Fast Gauss Seidel: various functions" begin

#--- generate system ---#

n_bodies = 1000
seed = 1234
system = generate_gravitational(seed, n_bodies)
derivatives_switches = FastMultipole.DerivativesSwitch(true, false, false, (system,))

direct!(system; scalar_potential=true, gradient=false)
phi_desired = system.potential[1, :]
system.potential[1, :] .*= -1.0 # invert potential to compel FGS to compute strengths

#--- create FGS solver ---#

fgs = FastMultipole.FastGaussSeidel((system,), (system,); expansion_order=4, multipole_acceptance=0.5, leaf_size=100) # try with leaf_size=3 for sources with no non-self influence

#--- unpack containers ---#

source_tree = fgs.source_tree
target_tree = fgs.target_tree
source_buffers = source_tree.buffers
target_buffers = target_tree.buffers
self_matrices = fgs.self_matrices
nonself_matrices = fgs.nonself_matrices
index_map = fgs.index_map
m2l_list = fgs.m2l_list
direct_list = fgs.direct_list
multipole_acceptance = fgs.multipole_acceptance
strengths = fgs.strengths
strengths_by_leaf = fgs.strengths_by_leaf
targets_by_branch = fgs.targets_by_branch
influences_per_system = fgs.influences_per_system
old_influence_storage = fgs.old_influence_storage
right_hand_side = self_matrices.rhs
extra_right_hand_side = fgs.extra_right_hand_side

#--- external right-hand side based on current influence ---#

# reset and update buffers
FastMultipole.target_influence_to_buffer!(target_buffers, (system,), derivatives_switches, target_tree.sort_index_list)

# run influence function on buffers
FastMultipole.reset!(extra_right_hand_side)
FastMultipole.influence!(extra_right_hand_side, influences_per_system, target_buffers, (system,), source_buffers, source_tree)

#--- check influence function ---#

i_body = 1
for i_branch in source_tree.leaf_index
    branch = source_tree.branches[i_branch]
    bodies_index = branch.bodies_index[1]
    for i in bodies_index
        @test isapprox(extra_right_hand_side[i_body], phi_desired[FastMultipole.sorted_index_2_unsorted_index(i, 1, source_tree)]; atol=1e-6)
        i_body += 1
    end
end

#--- check strengths_by_leaf ---#

for (i_leaf, i_branch) in enumerate(source_tree.leaf_index)
    branch = source_tree.branches[i_branch]
    bodies_index = branch.bodies_index[1]
    n_bodies = length(bodies_index)
    @test length(strengths_by_leaf[i_leaf]) == n_bodies
end

#--- update strengths ---#

FastMultipole.update_by_leaf!(strengths, strengths_by_leaf, (system,), source_buffers, source_tree)

#--- check strengths ---#

i_body = 1
for i_branch in source_tree.leaf_index
    branch = source_tree.branches[i_branch]
    bodies_index = branch.bodies_index[1]
    for i in bodies_index
        @test isapprox(strengths[i_body], system.bodies[FastMultipole.sorted_index_2_unsorted_index(i, 1, source_tree)].strength; atol=1e-6)
        i_body += 1
    end
end

#--- nonself influence function: first matrix ---#

# just the first nonself matrix
i_branch = direct_list[1][2]
i_leaf = findfirst(x -> source_tree.leaf_index[x] == i_branch, 1:length(source_tree.leaf_index)) # i_leaf = 3
right_hand_side .= zero(eltype(right_hand_side))
old_influence_storage .= zero(eltype(old_influence_storage))
FastMultipole.update_nonself_influence!(right_hand_side, strengths, nonself_matrices, old_influence_storage, i_leaf, source_tree, target_tree, strengths_by_leaf, index_map, direct_list, targets_by_branch)

# updated branch influence
i_target = direct_list[1][1] # i_target = 82
branch_influence = right_hand_side[targets_by_branch[i_target]]

# manual influence
FastMultipole.reset!(target_buffers)
source_index = source_tree.branches[i_branch].bodies_index[1]
target_index = target_tree.branches[i_target].bodies_index[1]
direct!(target_buffers[1], target_index, derivatives_switches[1], system, source_buffers[1], source_index)
test_influence = zero(extra_right_hand_side)
FastMultipole.influence!(test_influence, influences_per_system, target_buffers, (system,), source_buffers, source_tree)
manual_influence = test_influence[targets_by_branch[i_target]]

@test isapprox(branch_influence, manual_influence; atol=1e-6)

#--- nonself influence function: all matrices ---#

# zero RHS
right_hand_side .= zero(eltype(right_hand_side))

# zero nonself rhs
nonself_matrices.rhs .= zero(eltype(nonself_matrices.rhs))

# update nonself influence
FastMultipole.update_nonself_influence!(right_hand_side, strengths, nonself_matrices, old_influence_storage, source_tree, target_tree, strengths_by_leaf, index_map, direct_list, targets_by_branch)

# #--- check nonself influence function ---#

FastMultipole.reset!(target_buffers)

# all nonself nearfield interactions
FastMultipole.nearfield_singlethread!(target_buffers, target_tree.branches, (system,), source_buffers, source_tree.branches, derivatives_switches, direct_list)

# get influence from target buffers
test_influence = similar(extra_right_hand_side)
test_influence .= zero(eltype(test_influence))
FastMultipole.influence!(test_influence, influences_per_system, target_buffers, (system,), source_buffers, source_tree)

# check that the influence is the same as the right-hand side
@test isapprox(test_influence, right_hand_side; atol=1e-6)

#--- update strengths ---#

strengths .= rand(length(strengths)) # randomize strengths
FastMultipole.update_by_leaf!(source_buffers, (system,), strengths, strengths_by_leaf, source_tree)
FastMultipole.buffer_to_system_strength!((system,), source_tree)

#--- check strengths ---#

i_body = 1
for i_branch in source_tree.leaf_index
    branch = source_tree.branches[i_branch]
    bodies_index = branch.bodies_index[1]
    for i in bodies_index
        @test isapprox(strengths[i_body], system.bodies[FastMultipole.sorted_index_2_unsorted_index(i, 1, source_tree)].strength; atol=1e-6)
        i_body += 1
    end
end

end

@testset "Fast Gauss Seidel: full solve" begin

#--- generate system ---#

n_bodies = 100
seed = 123
system = generate_gravitational(seed, n_bodies)
derivatives_switches = FastMultipole.DerivativesSwitch(true, false, false, (system,))

direct!(system; scalar_potential=true, gradient=false)
strengths_desired = [b.strength for b in system.bodies]
phi_desired = system.potential[1, :]
system.potential[1, :] .*= -1.0 # invert external potential to compel FGS to compute the original strengths

# perturb strengths slightly
for i in eachindex(system.bodies)
    position = system.bodies[i].position
    radius = system.bodies[i].radius
    strength = system.bodies[i].strength
    system.bodies[i] = eltype(system.bodies)(position, radius, strength + round(strength, sigdigits=1)*100*(rand()-0.5))
end

#--- create FGS solver ---#

fgs = FastMultipole.FastGaussSeidel((system,), (system,); expansion_order=4, multipole_acceptance=0.5, leaf_size=n_bodies, shrink_recenter=false) # try with leaf_size=3 for sources with no non-self influence

#--- test solve! ---#

FastMultipole.solve!(system, fgs; max_iterations=10, tolerance=1e-3)

#--- check strengths ---#

i_body = 1
for i_branch in fgs.source_tree.leaf_index
    branch = fgs.source_tree.branches[i_branch]
    bodies_index = branch.bodies_index[1]
    for i in bodies_index
        @test isapprox(system.bodies[FastMultipole.sorted_index_2_unsorted_index(i, 1, fgs.source_tree)].strength, strengths_desired[i_body]; atol=1e-3)
        i_body += 1
    end
end

end

@testset "Fast Gauss Seidel: update system strengths" begin

#--- generate system ---#

n_bodies = 10000
seed = 123
system = generate_gravitational(seed, n_bodies)
derivatives_switches = FastMultipole.DerivativesSwitch(true, false, false, (system,))

#--- generate buffers ---#

systems = (system,)
target = false
TF = eltype(system)
source_tree = Tree(systems, target; buffers=FastMultipole.allocate_buffers(systems, target, TF), small_buffers = FastMultipole.allocate_small_buffers(systems, TF), expansion_order=4, leaf_size=SVector{1}(20), n_divisions=20, shrink_recenter=false, interaction_list_method=Barba())

#--- modify the buffer strengths ---#

for i in 1:FastMultipole.get_n_bodies(system)
    source_tree.buffers[1][5,i] = rand()
end

buffer_strengths = source_tree.buffers[1][5, :]
buffer_positions = source_tree.buffers[1][1:3, :]

#--- update system strengths ---#

FastMultipole.buffer_to_system_strength!(systems, source_tree)

#--- check strengths ---#

# sort system by position
system_positions = zeros(3, FastMultipole.get_n_bodies(system))
for i in 1:FastMultipole.get_n_bodies(system)
    system_positions[:, i] = system.bodies[i].position
end
system_strengths = [system.bodies[i].strength for i in 1:FastMultipole.get_n_bodies(system)]

system_p = sortperm(collect(eachcol(system_positions)))
sorted_system_positions = system_positions[:, system_p]
sorted_system_strengths = system_strengths[system_p]

# sort buffer by position
buffer_p = sortperm(collect(eachcol(buffer_positions)))
sorted_buffer_positions = buffer_positions[:, buffer_p]
sorted_buffer_strengths = buffer_strengths[buffer_p]

# test that the sorted positions and strengths are the same
for i in 1:FastMultipole.get_n_bodies(system)
    @test isapprox(sorted_system_positions[:, i], sorted_buffer_positions[:, i]; atol=1e-6)
    @test isapprox(sorted_system_strengths[i], sorted_buffer_strengths[i]; atol=1e-6)
end

end