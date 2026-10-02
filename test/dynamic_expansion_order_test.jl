@testset "dynamic expansion order: absolute and relative multipole power, point source" begin

expansion_order, leaf_size_source, multipole_acceptance = 20, SVector{1}(10), 0.5
n_bodies = 10000

shrink = recenter = true
seed = 123
validation_system = generate_gravitational(seed, n_bodies; radius_factor=0.0)
FastMultipole.direct!(validation_system; scalar_potential=true)

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
