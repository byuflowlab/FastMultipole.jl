# using FastMultipole
# using Random
# using LinearAlgebra
# using Test

# include("gravitational.jl")

#--- define influence function ---#

# just the scalar potential
function FastMultipole.influence!(influence, target_buffer, derivatives_switch::FastMultipole.DerivativesSwitch, ::Gravitational, source_buffer)
    influence .= view(target_buffer, FastMultipole.scalar_potential_index(derivatives_switch), :)
end

function FastMultipole.target_influence_to_buffer!(target_buffer, i_buffer, derivatives_switch::FastMultipole.DerivativesSwitch{PS}, target_system::Gravitational, i_target) where PS
    PS && (target_buffer[FastMultipole.scalar_potential_index(derivatives_switch), i_buffer] = target_system.potential[1, i_target])
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
FastMultipole.influence!(extra_right_hand_side, influences_per_system, target_buffers, (system,), source_buffers, source_tree, derivatives_switches)

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
FastMultipole.influence!(test_influence, influences_per_system, target_buffers, (system,), source_buffers, source_tree, derivatives_switches)
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
FastMultipole.influence!(test_influence, influences_per_system, target_buffers, (system,), source_buffers, source_tree, derivatives_switches)

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

fgs = FastMultipole.FastGaussSeidel((system,), (system,); expansion_order=4, multipole_acceptance=0.5, leaf_size=n_bodies, shrink=false, recenter=false) # try with leaf_size=3 for sources with no non-self influence

#--- test solve! ---#

FastMultipole.solve!(system, fgs; scalar_potential=true, gradient=false, max_iterations=10, tolerance=1e-3)

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
switches = DerivativesSwitch(true, true, true, systems)
source_tree = Tree(systems, target, switches; buffers=FastMultipole.allocate_buffers(systems, target, TF, switches), small_buffers = FastMultipole.allocate_small_buffers(systems, TF), expansion_order=4, leaf_size=SVector{1}(20), n_divisions=20, shrink=false, recenter=false, interaction_list_method=Barba())

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

@testset "Fast Gauss Seidel: shared source/target tree topology" begin

#--- system whose body radii split independently built source/target trees ---#

n_bodies = 300
seed = 42
leaf_size = 10
expansion_order = 4

# large body radii make the source tree stop subdividing early (child radius < max body
# radius) while a target tree keeps splitting to leaf_size
system = generate_gravitational(seed, n_bodies; radius_factor=4.0)

# premise guard: independent builds MUST diverge, else this test is vacuous
switches = FastMultipole.DerivativesSwitch(true, true, true, (system,))
independent_target_tree = FastMultipole.Tree((system,), true, switches; expansion_order, leaf_size=SVector{1}(leaf_size), shrink=true, recenter=false, interaction_list_method=FastMultipole.Barba())
independent_source_tree = FastMultipole.Tree((system,), false, switches; expansion_order, leaf_size=SVector{1}(leaf_size), shrink=true, recenter=false, interaction_list_method=FastMultipole.Barba())
@test length(independent_target_tree.branches) != length(independent_source_tree.branches)

#--- FGS constructor must produce structurally identical trees ---#

fgs = FastMultipole.FastGaussSeidel((system,), (system,); expansion_order, multipole_acceptance=0.5, leaf_size)

@test length(fgs.target_tree.branches) == length(fgs.source_tree.branches)
@test all(t.bodies_index == s.bodies_index && t.branch_index == s.branch_index for (t, s) in zip(fgs.target_tree.branches, fgs.source_tree.branches))
FastMultipole.assert_shared_topology(fgs.target_tree, fgs.source_tree) # must not throw

# target-role shrink keeps target leaf radii tight (source radii include body radii)
@test all(fgs.target_tree.branches[i].radius <= fgs.source_tree.branches[i].radius + 1e-12 for i in fgs.source_tree.leaf_index)

#--- one solve must run without BoundsError ---#

direct!(system; scalar_potential=true, gradient=false)
system.potential[1, :] .*= -1.0 # invert external potential so FGS solves for strengths
FastMultipole.solve!(system, fgs; scalar_potential=true, gradient=false, max_iterations=20, tolerance=1e-3)
@test all(isfinite(b.strength) for b in system.bodies)

end

@testset "Fast Gauss Seidel: threaded M2L repeatability" begin

# Multiple leaves and a nonempty M2L list are premise guards: without both,
# the owner-partitioning race fixed by canonical target/source ordering is not
# exercised.  Keep the solve cold and fixed-iteration so threshold behavior
# cannot hide an early bit difference.
system = generate_gravitational(20260815, 800)
direct!(system; scalar_potential=true, gradient=false)
system.potential[1, :] .*= -1.0

fgs = FastMultipole.FastGaussSeidel((system,), (system,);
    expansion_order=4, multipole_acceptance=0.5, leaf_size=40,
    shrink=true, recenter=false)
@test length(fgs.source_tree.leaf_index) > 1
@test !isempty(fgs.m2l_list)
@test issorted(fgs.m2l_list; by=ij -> (ij[1], ij[2]))

function cold_fixed_solve!()
    for i in eachindex(system.bodies)
        body = system.bodies[i]
        system.bodies[i] = typeof(body)(body.position, body.radius, 0.0)
    end
    residuals = Float64[]
    FastMultipole.solve!(system, fgs; scalar_potential=true, gradient=false,
        max_iterations=6, inner_iterations=2, tolerance=-1.0,
        reverse_pass=false, final_update=false, verbose=false,
        callback=(_, residual) -> push!(residuals, residual))
    strengths = [body.strength for body in system.bodies]
    return collect(reinterpret(UInt64, residuals)),
           collect(reinterpret(UInt64, strengths))
end

reference_residuals, reference_strengths = cold_fixed_solve!()
for _ in 1:3
    residuals, strengths = cold_fixed_solve!()
    @test residuals == reference_residuals
    @test strengths == reference_strengths
end

end

@testset "Fast Gauss Seidel: cached leaf LU factorizations" begin
    # This deterministic tree has multiple leaves and no one-body gravitational
    # leaf (whose deliberately zero self term would make that block singular).
    system = generate_gravitational(20260815, 800)
    direct!(system; scalar_potential=true, gradient=false)
    system.potential[1, :] .*= -1.0

    original_strengths = [body.strength for body in system.bodies]
    cached = FastMultipole.FastGaussSeidel((system,), (system,);
        expansion_order=4, multipole_acceptance=0.5, leaf_size=40,
        shrink=true, recenter=false)
    self_data = copy(cached.self_matrices.data)
    cache = cached.leaf_lu_cache

    @test cached.cache_leaf_lu
    @test cache !== nothing
    @test length(cache.factorizations) == length(cached.self_matrices.sizes) > 1
    @test cache.data !== cached.self_matrices.data
    @test cache.data != cached.self_matrices.data
    @test cached.self_matrices.data == self_data
    @test cache.build_time >= 0.0
    @test cache.bytes == sizeof(cache.data) + sum(sizeof(F.ipiv) for F in cache.factorizations)
    @test all(parent(parent(F.factors)) === cache.data for F in cache.factorizations)

    uncached = FastMultipole.FastGaussSeidel((system,), (system,);
        expansion_order=4, multipole_acceptance=0.5, leaf_size=40,
        shrink=true, recenter=false, cache_leaf_lu=false)
    @test !uncached.cache_leaf_lu
    @test uncached.leaf_lu_cache === nothing
    @test uncached.self_matrices.data == cached.self_matrices.data

    for i_leaf in eachindex(cached.self_matrices.sizes)
        _, rhs = FastMultipole.get_matrix_vector(cached.self_matrices, i_leaf)
        rhs .= sin.(eachindex(rhs))
        x_cached = similar(rhs)
        x_uncached = similar(rhs)
        FastMultipole.solve_leaf!(x_cached, cached.self_matrices, cache, i_leaf)
        FastMultipole.solve_leaf!(x_uncached, cached.self_matrices, nothing, i_leaf)
        @test x_cached ≈ x_uncached rtol=1e-12 atol=1e-12
        @test cached.self_matrices.data == self_data
    end

    function cached_path_cold_solve!(system, solver, original_strengths;
                                     reverse_pass)
        for (i, body) in enumerate(system.bodies)
            system.bodies[i] = typeof(body)(body.position, body.radius,
                                             original_strengths[i])
        end
        residuals = Float64[]
        FastMultipole.solve!(system, solver; scalar_potential=true, gradient=false,
            max_iterations=5, inner_iterations=2, tolerance=-1.0,
            reverse_pass, final_update=false, verbose=false,
            callback=(_, residual) -> push!(residuals, residual))
        return residuals, [body.strength for body in system.bodies]
    end

    for reverse_pass in (false, true), repetition in 1:2
        residuals_cached, strengths_cached = cached_path_cold_solve!(
            system, cached, original_strengths; reverse_pass)
        residuals_uncached, strengths_uncached = cached_path_cold_solve!(
            system, uncached, original_strengths; reverse_pass)
        @test residuals_cached ≈ residuals_uncached rtol=1e-11 atol=1e-12
        @test strengths_cached ≈ strengths_uncached rtol=1e-11 atol=1e-12
        @test cached.self_matrices.data == self_data
    end

    singular = FastMultipole.Matrices([(2, 2)])
    singular.data .= [1.0, 2.0, 2.0, 4.0]
    @test_throws LinearAlgebra.SingularException FastMultipole.build_leaf_lu_cache(singular)
end
