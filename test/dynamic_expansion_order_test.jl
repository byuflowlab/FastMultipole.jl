# @testset "dynamic expansion order: absolute and relative rotated coefficients, point source" begin

# expansion_order, leaf_size_source, multipole_acceptance = 20, SVector{1}(10), 0.5
# n_bodies = 10000

# shrink = recenter = true
# seed = 123
# validation_system = generate_gravitational(seed, n_bodies; radius_factor=0.0)
# FastMultipole.direct!(validation_system)
# validation_potential = validation_system.potential[1,:]

# validation_system2 = generate_gravitational(seed, n_bodies; radius_factor=0.1)
# FastMultipole.direct!(validation_system2)
# validation_potential2 = validation_system2.potential[1,:]

# @assert validation_potential == validation_potential2

# ε = 1e-5
# error_tolerance = FastMultipole.RotatedCoefficientsAbsoluteGradient(ε, false)
# # error_tolerance = nothing
# system = generate_gravitational(seed, n_bodies; radius_factor=0.0)
# system2 = generate_gravitational(seed, n_bodies; radius_factor=0.1)

# # println("\n===== radius factor = 0.0 =====\n")

# gradient_null = system.potential[5:7,:]
# optimized_args, cache, target_tree, source_tree, m2l_list, direct_list, derivatives_switches, error_success = FastMultipole.fmm!(system; expansion_order, leaf_size_source, multipole_acceptance, nearfield=true, farfield=true, shrink, recenter, error_tolerance)

# gradient_fmm = system.potential[5:7,:]
# gradient_err = [norm(system.potential[5:7,i] - validation_system.potential[5:7,i]) for i in 1:size(system.potential,2)]

# @test ε * 0.1 < maximum(gradient_err) < ε * 30

# # println("\n===== radius factor = 0.1 =====\n")

# FastMultipole.fmm!(system2; expansion_order, leaf_size_source, multipole_acceptance, nearfield=true, farfield=true, shrink, recenter, error_tolerance)

# gradient_err = [norm(system2.potential[5:7,i] - validation_system.potential[5:7,i]) for i in 1:size(system.potential,2)]

# @test ε * 0.1 < maximum(gradient_err) < ε * 30

# # relative error tolerance
# error_tolerance = FastMultipole.RotatedCoefficientsRelativeGradient(ε, eps(), false)
# FastMultipole.fmm!(system; expansion_order, leaf_size_source, multipole_acceptance, nearfield=true, farfield=true, shrink, recenter, error_tolerance)
# gradient_err = [norm(system.potential[5:7,i] - validation_system.potential[5:7,i]) / norm(validation_system.potential[5:7,i]) for i in 1:size(system.potential,2)]

# @test ε * 0.1 < maximum(gradient_err) < ε * 30

# end

# @testset "dynamic expansion order: absolute and relative rotated coefficients, point vortex" begin

# expansion_order, leaf_size_source, multipole_acceptance = 20, SVector{1}(10), 0.5
# n_bodies = 10000

# shrink = recenter = true
# seed = 123
# validation_system = generate_vortex(seed, n_bodies; radius_factor=0.0)
# FastMultipole.direct!(validation_system)
# validation_potential = validation_system.gradient_stretching[1:3,:]

# validation_system2 = generate_vortex(seed, n_bodies; radius_factor=0.1)
# FastMultipole.direct!(validation_system2)
# validation_potential2 = validation_system2.gradient_stretching[1:3,:]

# @assert validation_potential == validation_potential2

# ε = 1e-5
# error_tolerance = FastMultipole.RotatedCoefficientsAbsoluteGradient(ε, false)
# # error_tolerance = nothing
# system = generate_vortex(seed, n_bodies; radius_factor=0.0)
# system2 = generate_vortex(seed, n_bodies; radius_factor=0.1)

# # println("\n===== radius factor = 0.0 =====\n")

# FastMultipole.fmm!(system; expansion_order, leaf_size_source, multipole_acceptance, shrink, recenter, error_tolerance)

# gradient_err = [norm(system.gradient_stretching[1:3,i] - validation_system.gradient_stretching[1:3,i]) for i in 1:size(system.gradient_stretching,2)]

# @test ε * 0.1 < maximum(gradient_err) < ε * 100

# # println("\n===== radius factor = 0.1 =====\n")

# FastMultipole.fmm!(system2; expansion_order, leaf_size_source, multipole_acceptance, shrink, recenter, error_tolerance)

# gradient_err = [norm(system2.gradient_stretching[1:3,i] - validation_system.gradient_stretching[1:3,i]) for i in 1:size(system.gradient_stretching,2)]

# @test ε * 0.1 < maximum(gradient_err) < ε * 100

# # relative error tolerance
# error_tolerance = FastMultipole.RotatedCoefficientsRelativeGradient(ε, eps(), false)
# optimized_args, cache, target_tree, source_tree, _ = FastMultipole.fmm!(system; expansion_order, leaf_size_source, multipole_acceptance, shrink, recenter, error_tolerance)
# gradient_err = [norm(system.gradient_stretching[1:3,i] - validation_system.gradient_stretching[1:3,i]) / norm(validation_system.gradient_stretching[1:3,i]) for i in 1:size(system.gradient_stretching,2)]

# @test ε * 0.1 < maximum(gradient_err) < ε * 30

# end

@testset "dynamic expansion order: absolute and relative multipole power, point source" begin

expansion_order, leaf_size_source, multipole_acceptance = 20, SVector{1}(10), 0.5
n_bodies = 10000

shrink = recenter = true
seed = 123
validation_system = generate_gravitational(seed, n_bodies; radius_factor=0.0)
FastMultipole.direct!(validation_system; scalar_potential=true)
validation_potential = validation_system.potential[1,:]

validation_system2 = generate_gravitational(seed, n_bodies; radius_factor=0.1)
FastMultipole.direct!(validation_system2; scalar_potential=true)
validation_potential2 = validation_system2.potential[1,:]

@assert validation_potential == validation_potential2

ε = 1e-5
error_tolerance = FastMultipole.PowerAbsoluteGradient(ε, false)
system = generate_gravitational(seed, n_bodies; radius_factor=0.0)
system2 = generate_gravitational(seed, n_bodies; radius_factor=0.1)

# println("\n===== radius factor = 0.0 =====\n")

FastMultipole.fmm!(system; expansion_order, leaf_size_source, multipole_acceptance, nearfield=true, farfield=true, shrink, recenter, error_tolerance)

gradient_err = [norm(system.potential[5:7,i] - validation_system.potential[5:7,i]) for i in 1:size(system.potential,2)]

@test ε * 0.1 < maximum(gradient_err) < ε * 10

# println("\n===== radius factor = 0.1 =====\n")

FastMultipole.fmm!(system2; expansion_order, leaf_size_source, multipole_acceptance, nearfield=true, farfield=true, shrink, recenter, error_tolerance)

gradient_err = [norm(system2.potential[5:7,i] - validation_system.potential[5:7,i]) for i in 1:size(system.potential,2)]

@test ε * 0.1 < maximum(gradient_err) < ε * 10

# relative error tolerance
error_tolerance = FastMultipole.PowerRelativeGradient(ε, eps(), false)
FastMultipole.fmm!(system; expansion_order, leaf_size_source, multipole_acceptance, nearfield=true, farfield=true, shrink, recenter, error_tolerance)
gradient_err = [norm(system.potential[5:7,i] - validation_system.potential[5:7,i]) / norm(validation_system.potential[5:7,i]) for i in 1:size(system.potential,2)]

@test ε * 0.1 < maximum(gradient_err) < ε * 10

# absolute potential
error_tolerance = FastMultipole.PowerAbsolutePotential(ε, false)
FastMultipole.fmm!(system; expansion_order, leaf_size_source, multipole_acceptance, nearfield=true, farfield=true, shrink, recenter, error_tolerance, scalar_potential=true)
gradient_err = [(system.potential[1,i] - validation_system.potential[1,i]) for i in 1:size(system.potential,2)]

@test ε * 0.1 < maximum(gradient_err) < ε * 10

# relative error tolerance
error_tolerance = FastMultipole.PowerRelativePotential(ε, eps(), false)
FastMultipole.fmm!(system; expansion_order, leaf_size_source, multipole_acceptance, nearfield=true, farfield=true, shrink, recenter, error_tolerance, scalar_potential=true)
gradient_err = [(system.potential[1,i] - validation_system.potential[1,i]) / validation_system.potential[1,i] for i in 1:size(system.potential,2)]

@test ε * 0.1 < maximum(gradient_err) < ε * 10

end

@testset "dynamic expansion order: absolute and relative multipole power, point vortex" begin

expansion_order, leaf_size_source, multipole_acceptance = 20, SVector{1}(10), 0.5
n_bodies = 10000

shrink = recenter = true
seed = 12345
validation_system = generate_vortex(seed, n_bodies; radius_factor=0.0)
FastMultipole.direct!(validation_system)
validation_potential = validation_system.potential

validation_system2 = generate_vortex(seed, n_bodies; radius_factor=0.1)
FastMultipole.direct!(validation_system2)
validation_potential2 = validation_system2.potential

@assert validation_potential == validation_potential2

ε = 1e-5
error_tolerance = FastMultipole.PowerAbsoluteGradient(ε, false)
# error_tolerance = nothing
system = generate_vortex(seed, n_bodies; radius_factor=0.0)
system2 = generate_vortex(seed, n_bodies; radius_factor=0.1)

# println("\n===== radius factor = 0.0 =====\n")

tree, m2l_list, direct_list, derivatives_switches = FastMultipole.fmm!(system; expansion_order, leaf_size_source, multipole_acceptance, shrink, recenter, error_tolerance)

gradient_err = [norm(system.gradient_stretching[1:3,i] - validation_system.gradient_stretching[1:3,i]) for i in 1:size(system.potential,2)]

@test ε * 0.1 < maximum(gradient_err) < ε * 10

# println("\n===== radius factor = 0.1 =====\n")

tree2, m2l_list2, direct_list2, derivatives_switches2 = FastMultipole.fmm!(system2; expansion_order, leaf_size_source, multipole_acceptance, shrink, recenter, error_tolerance)

gradient_err = [norm(system2.gradient_stretching[1:3,i] - validation_system.gradient_stretching[1:3,i]) for i in 1:size(system.potential,2)]

@test ε * 0.1 < maximum(gradient_err) < ε * 10

# relative error tolerance
error_tolerance = FastMultipole.PowerRelativeGradient(ε, eps(), false)
optimized_args, cache, target_tree, source_tree, _ = FastMultipole.fmm!(system; expansion_order, leaf_size_source, multipole_acceptance, shrink, recenter, error_tolerance)
gradient_err = [norm(system.gradient_stretching[1:3,i] - validation_system.gradient_stretching[1:3,i]) / norm(validation_system.gradient_stretching[1:3,i]) for i in 1:size(system.gradient_stretching,2)]

@test ε * 0.1 < maximum(gradient_err) < ε * 10

end




# @testset "dynamic expansion order: influence estimate, point source" begin

# expansion_order, leaf_size_source, multipole_acceptance = 20, SVector{1}(100), 0.5
# n_bodies = 10000

# shrink = recenter = true
# seed = 123
# validation_system = generate_gravitational(seed, n_bodies; radius_factor=0.0)
# FastMultipole.direct!(validation_system)
# validation_potential = validation_system.potential[1,:]
# validation_vector = validation_system.potential[5:7,:]

# validation_system2 = generate_gravitational(seed, n_bodies; radius_factor=0.1)
# fmm!(validation_system2;
#     scalar_potential = true, gradient = true, hessian = false,
#     leaf_size_source = FastMultipole.to_vector(5, 1),
#     expansion_order = 3, multipole_acceptance = 0.6,
#     error_tolerance = nothing, shrink = recenter = true, nearfield_device = false,
#     update_target_systems = true,
#     silence_warnings = true
# )
# validation_potential2 = validation_system2.potential[1,:]
# validation_vector2 = validation_system2.potential[5:7,:]

# gradient = [norm(validation_gradient[1:3,i]) for i in 1:size(validation_gradient,2)]
# gradient_err = [norm(validation_gradient[1:3,i] - validation_gradient2[1:3,i]) for i in 1:size(validation_gradient2,2)]
# relative_err = gradient_err ./ gradient

# @test maximum(relative_err) < 1e-1

# end







# @testset "dynamic expansion order: relative rotated coefficients, point source" begin

# expansion_order, leaf_size_source, multipole_acceptance = 20, SVector{1}(100), 0.5
# n_bodies = 10000

# shrink = recenter = true
# seed = 123
# validation_system = generate_gravitational(seed, n_bodies; radius_factor=0.0)
# FastMultipole.direct!(validation_system)
# validation_potential = validation_system.potential[1,:]

# validation_system2 = generate_gravitational(seed, n_bodies; radius_factor=0.1)
# FastMultipole.direct!(validation_system2)
# validation_potential2 = validation_system2.potential[1,:]

# @assert validation_potential == validation_potential2

# ε = 1e-4
# error_tolerance = FastMultipole.RotatedCoefficientsRelativeGradient(ε, false)
# # error_tolerance = nothing
# system = generate_gravitational(seed, n_bodies; radius_factor=0.0)
# system2 = generate_gravitational(seed, n_bodies; radius_factor=0.1)

# # println("\n===== radius factor = 0.0 =====\n")

# FastMultipole.DEBUG[] = true
# _, _, target_tree, _ = FastMultipole.fmm!(system; expansion_order, leaf_size_source, multipole_acceptance, nearfield=true, farfield=true, shrink, recenter, error_tolerance)
# FastMultipole.DEBUG[] = false

# gradient = [norm(validation_system.potential[5:7,i]) for i in 1:size(validation_system.potential,2)]
# gradient_err = [norm(system.potential[5:7,i] - validation_system.potential[5:7,i]) for i in 1:size(system.potential,2)]
# relative_err = gradient_err ./ gradient

# i_worst = findfirst((x) -> x == maximum(relative_err), relative_err)
# @show i_worst FastMultipole.unsorted_index_2_sorted_index(i_worst, 1, target_tree)

# @show mean(relative_err) std(relative_err) maximum(relative_err)
# @show mean(gradient_err) std(gradient_err) maximum(gradient_err)
# @show mean(gradient) std(gradient) maximum(gradient)
# @show sum(relative_err .> ε) / length(relative_err)

# @test ε * 0.1 < maximum(relative_err) < ε * 10

# # println("\n===== radius factor = 0.1 =====\n")

# FastMultipole.fmm!(system2; expansion_order, leaf_size_source, multipole_acceptance, nearfield=true, farfield=true, shrink, recenter, error_tolerance)

# gradient = [norm(system2.potential[5:7,:]) for i in 1:size(system2.potential,2)]
# gradient_err = [norm(system2.potential[5:7,i] - validation_system.potential[5:7,i]) for i in 1:size(system2.potential,2)]
# relative_err = gradient_err ./ gradient

# @test ε * 0.1 < maximum(relative_err) < ε * 10

# end

# @testset "dynamic expansion order: relative rotated coefficients, point vortex" begin

# expansion_order, leaf_size_source, multipole_acceptance = 20, SVector{1}(100), 0.5
# n_bodies = 10000

# shrink = recenter = true
# seed = 123
# validation_system = generate_vortex(seed, n_bodies; radius_factor=0.0)
# FastMultipole.direct!(validation_system)
# validation_potential = validation_system.gradient_stretching[1:3,:]

# validation_system2 = generate_vortex(seed, n_bodies; radius_factor=0.1)
# FastMultipole.direct!(validation_system2)
# validation_potential2 = validation_system2.gradient_stretching[1:3,:]

# @assert validation_potential == validation_potential2

# ε = 1e-4
# error_tolerance = FastMultipole.RotatedCoefficientsRelativeGradient(ε, false)
# lamb_helmholtz = true
# # error_tolerance = nothing
# system = generate_vortex(seed, n_bodies; radius_factor=0.0)
# system2 = generate_vortex(seed, n_bodies; radius_factor=0.1)

# # println("\n===== radius factor = 0.0 =====\n")

# FastMultipole.fmm!(system; expansion_order, leaf_size_source, multipole_acceptance, shrink, recenter, error_tolerance)

# gradient = [norm(system.gradient_stretching[1:3,i]) for i in 1:size(system.potential,2)]
# gradient_err = [norm(system.gradient_stretching[1:3,i] - validation_system.gradient_stretching[1:3,i]) for i in 1:size(system.potential,2)]
# relative_err = gradient_err ./ gradient

# @test ε * 0.1 < maximum(gradient_err) < ε * 60

# # println("\n===== radius factor = 0.1 =====\n")

# FastMultipole.fmm!(system2; expansion_order, leaf_size_source, multipole_acceptance, shrink, recenter, error_tolerance)

# gradient = [norm(system2.gradient_stretching[1:3,i]) for i in 1:size(system2.potential,2)]
# gradient_err = [norm(system2.gradient_stretching[1:3,i] - validation_system.gradient_stretching[1:3,i]) for i in 1:size(system2.potential,2)]
# relative_err = gradient_err ./ gradient

# @test ε * 0.1 < maximum(gradient_err) < ε * 60

# end

# @testset "dynamic expansion order: relative multipole power, point source" begin

# expansion_order, leaf_size_source, multipole_acceptance = 20, SVector{1}(100), 0.5
# n_bodies = 10000

# shrink = recenter = true
# seed = 123
# validation_system = generate_gravitational(seed, n_bodies; radius_factor=0.0)
# FastMultipole.direct!(validation_system)
# validation_potential = validation_system.potential[1,:]

# validation_system2 = generate_gravitational(seed, n_bodies; radius_factor=0.1)
# FastMultipole.direct!(validation_system2)
# validation_potential2 = validation_system2.potential[1,:]

# @assert validation_potential == validation_potential2

# ε = 1e-4
# error_tolerance = FastMultipole.PowerRelativeGradient(ε, false)
# system = generate_gravitational(seed, n_bodies; radius_factor=0.0)
# system2 = generate_gravitational(seed, n_bodies; radius_factor=0.1)

# # println("\n===== radius factor = 0.0 =====\n")

# FastMultipole.fmm!(system; expansion_order, leaf_size_source, multipole_acceptance, nearfield=true, farfield=true, shrink, recenter, error_tolerance)

# gradient = [norm(system.potential[5:7,:]) for i in 1:size(system.potential,2)]
# gradient_err = [norm(system.potential[5:7,i] - validation_system.potential[5:7,i]) for i in 1:size(system.potential,2)]
# relative_err = gradient_err ./ gradient

# @test ε * 0.1 < maximum(gradient_err) < ε * 10

# # println("\n===== radius factor = 0.1 =====\n")

# FastMultipole.fmm!(system2; expansion_order, leaf_size_source, multipole_acceptance, nearfield=true, farfield=true, shrink, recenter, error_tolerance)

# gradient = [norm(system2.potential[5:7,:]) for i in 1:size(system2.potential,2)]
# gradient_err = [norm(system2.potential[5:7,i] - validation_system.potential[5:7,i]) for i in 1:size(system2.potential,2)]
# relative_err = gradient_err ./ gradient

# @test ε * 0.1 < maximum(relative_err) < ε * 10

# end

# @testset "dynamic expansion order: relative multipole power, point vortex" begin

# expansion_order, leaf_size_source, multipole_acceptance = 20, SVector{1}(100), 0.5
# n_bodies = 10000
# lamb_helmholtz = true

# shrink = recenter = true
# seed = 123
# validation_system = generate_vortex(seed, n_bodies; radius_factor=0.0)
# FastMultipole.direct!(validation_system)
# validation_potential = validation_system.potential

# validation_system2 = generate_vortex(seed, n_bodies; radius_factor=0.1)
# FastMultipole.direct!(validation_system2)
# validation_potential2 = validation_system2.potential

# @assert validation_potential == validation_potential2

# ε = 1e-4
# error_tolerance = FastMultipole.PowerRelativeGradient(ε, false)
# system = generate_vortex(seed, n_bodies; radius_factor=0.0)
# system2 = generate_vortex(seed, n_bodies; radius_factor=0.1)

# # println("\n===== radius factor = 0.0 =====\n")

# FastMultipole.fmm!(system; expansion_order, leaf_size_source, multipole_acceptance, shrink, recenter, error_tolerance)

# gradient = [norm(system.gradient_stretching[1:3,i]) for i in 1:size(system.potential,2)]
# gradient_err = [norm(system.gradient_stretching[1:3,i] - validation_system.gradient_stretching[1:3,i]) for i in 1:size(system.potential,2)]
# relative_err = gradient_err ./ gradient

# @test ε * 0.1 < maximum(gradient_err) < ε * 10

# # println("\n===== radius factor = 0.1 =====\n")

# FastMultipole.fmm!(system2; expansion_order, leaf_size_source, multipole_acceptance, shrink, recenter, error_tolerance)

# gradient = [norm(system2.gradient_stretching[1:3,i]) for i in 1:size(system2.potential,2)]
# gradient_err = [norm(system2.gradient_stretching[1:3,i] - validation_system.gradient_stretching[1:3,i]) for i in 1:size(system2.potential,2)]
# relative_err = gradient_err ./ gradient

# @test ε * 0.1 < maximum(gradient_err) < ε * 30

# end
