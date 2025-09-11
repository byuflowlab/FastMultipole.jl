# import FLOWUnsteady as uns
# using VSPGeom
# using BSON:@load,@save
using StaticArrays
# import FLOWPanel as pnl
# import FLOWVPM as vpm
import FastMultipole as fmm
using LinearAlgebra
using Random
using Statistics

println("*********************")

include("../test/vortex.jl")
include("../test/gravitational.jl")
include("../test/source_vortex.jl")

n_tests = 2
n_bodies = 10000
seed = 123
expansion_order = 8
ε_abs = 1e-4

#------- combining kernels for coincident elements -------#

#=
position = rand(3,n_bodies)
vector_strength = (rand(3, n_bodies) .* 2 .- 1) .* 4.5/n_bodies
scalar_strength = (rand(1, n_bodies) .* 2 .- 1) .* 12.7/n_bodies

vortons = VortexParticles(deepcopy(position), deepcopy(vector_strength))
sources = Gravitational(vcat(deepcopy(position), zeros(1,n_bodies), deepcopy(scalar_strength)))

source_vortons = SourceVortons(deepcopy(position), vcat(deepcopy(scalar_strength), deepcopy(vector_strength)))


leaf_size = 100
t_source = @elapsed fmm.fmm!(sources; hessian=false, leaf_size, lamb_helmholtz=false, expansion_order, ε_abs)

t_vortex = @elapsed fmm.fmm!(vortons; hessian=false, leaf_size, lamb_helmholtz=true, expansion_order, ε_abs)

t_source_vortex = @elapsed fmm.fmm!(source_vortons; hessian=false, leaf_size, lamb_helmholtz=true, expansion_order, ε_abs)

println("==== BENCHMARKS: ====")
println("\tt_source:          $t_source")
println("\tt_vortex:          $t_vortex")
println("\tt_separate:        $(t_source + t_vortex)")
println("\tt_combined:        $t_source_vortex")
println("done.")
=#
#------- combining disparate elements under the same expansions -------#

n_bodies = 10000
# particle_system = generate_gravitational(seed, 5*n_bodies; strength_factor=12.672128785485043/n_bodies)
# vortex_system = generate_vortex(seed, n_bodies; strength_scale=4.521392295809281/n_bodies)

system1 = generate_gravitational(seed, n_bodies; strength_factor=12.672128785485043/n_bodies)
system2 = generate_gravitational(seed*2, n_bodies; strength_factor=12.672128785485043/n_bodies)
system3 = generate_gravitational(seed*3, n_bodies; strength_factor=12.672128785485043/n_bodies)
system4 = generate_vortex(seed*4, n_bodies; strength_scale=4.521392295809281/n_bodies)
system5 = generate_vortex(seed*5, n_bodies; strength_scale=4.521392295809281/n_bodies)
system6 = generate_vortex(seed*6, n_bodies; strength_scale=4.521392295809281/n_bodies)

system1_times = zeros(n_tests)
system2_times = zeros(n_tests)
system3_times = zeros(n_tests)
system4_times = zeros(n_tests)
system5_times = zeros(n_tests)
system6_times = zeros(n_tests)
scalar_times = zeros(n_tests)
vector_times = zeros(n_tests)

combined_times = zeros(n_tests)

expansion_order = 8
# fmm tuning
lsv = 130
lsm = 130
leaf_size_target = SVector{6}(lsm,lsm,lsm,lsv,lsv,lsv)
leaf_size_source = lsm
println("starting tests 1")

for i = 1:n_tests
    system1_times[i] += @elapsed fmm.fmm!((system1, system2, system3, system4, system5, system6), system1; hessian=false, leaf_size_target, leaf_size_source, lamb_helmholtz=false, expansion_order, ε_abs=1e-4)
end

println("starting tests 2")
for i in 1:n_tests
    system2_times[i] += @elapsed fmm.fmm!((system1, system2, system3, system4, system5, system6), system2; hessian=false, leaf_size_target, leaf_size_source, lamb_helmholtz=false, expansion_order, ε_abs=1e-4)
end

println("starting tests 3")
for i in 1:n_tests
    system3_times[i] += @elapsed fmm.fmm!((system1, system2, system3, system4, system5, system6), system3; hessian=false, leaf_size_target, leaf_size_source, lamb_helmholtz=false, expansion_order, ε_abs=1e-4)
end

# fmm tuning
lsv = 130
lsm = 130
leaf_size_target = SVector{6}(lsm,lsm,lsm,lsv,lsv,lsv)
leaf_size_source = lsv

println("starting tests 4")
for i in 1:n_tests
    system4_times[i] += @elapsed fmm.fmm!((system1, system2, system3, system4, system5, system6), system4; hessian=false, leaf_size_target, leaf_size_source, lamb_helmholtz=true, expansion_order, ε_abs=1e-4)
end


println("starting tests 5")
# leaf_size_source = 18
for i in 1:n_tests
    system5_times[i] += @elapsed fmm.fmm!((system1, system2, system3, system4, system5, system6), system5; hessian=false, leaf_size_target, leaf_size_source, lamb_helmholtz=true, expansion_order, ε_abs=1e-4)
end

println("starting tests 6")
for i in 1:n_tests
    system6_times[i] += @elapsed fmm.fmm!((system1, system2, system3, system4, system5, system6), system6; hessian=false, leaf_size_target, leaf_size_source, lamb_helmholtz=true, expansion_order, ε_abs=1e-4)
end

# fmm tuning
lsv = 112
lsm = 115
leaf_size_target = SVector{6}(lsm,lsm,lsm,lsm,lsv,lsv)
leaf_size_source = SVector{4}(lsm,lsm,lsm,lsm)

println("starting tests 7")
for i in 1:n_tests
    scalar_times[i] += @elapsed fmm.fmm!((system1, system2, system3, system4, system5, system6), (system1, system2, system3, system4); hessian=false, leaf_size_target, leaf_size_source, lamb_helmholtz=false, expansion_order, ε_abs=1e-4)
end

# fmm tuning
lsv = 120
lsm = 120
leaf_size_target = SVector{6}(lsm,lsm,lsm,lsm,lsv,lsv)
leaf_size_source = SVector{2}(lsv,lsv)

println("starting tests 8")
for i in 1:n_tests
    vector_times[i] += @elapsed fmm.fmm!((system1, system2, system3, system4, system5, system6), (system5, system6); hessian=false, leaf_size_target, leaf_size_source, lamb_helmholtz=true, expansion_order, ε_abs=1e-4)
end

# fmm tuning
lsv = 110
lsm = 110
leaf_size = SVector{6}(lsm,lsm,lsm,lsm,lsv,lsv)

println("starting tests 9")
for i in 1:n_tests
    combined_times[i] += @elapsed fmm.fmm!((system1, system2, system3, system4, system5, system6); hessian=false, leaf_size, lamb_helmholtz=true, expansion_order, ε_abs=1e-4)
end

@show mean(system1_times[2:end]) mean(system2_times[2:end]) mean(system3_times[2:end]) mean(system4_times[2:end]) mean(system5_times[2:end]) mean(system6_times[2:end]) mean(scalar_times[2:end]) mean(vector_times[2:end]) mean(combined_times[2:end])
