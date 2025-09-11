using StaticArrays
using Random

include("../test/vortex.jl")

n_bodies = 10_000
system = generate_vortex(123, n_bodies; strength_scale=1/0.22/n_bodies)

expansion_order = 13
leaf_size_source = 50
multipole_acceptance = 0.5
ε_abs = 1e-6

# optargs, _ = fmm!(system; expansion_order, ε_abs, leaf_size_source=20, multipole_acceptance=0.5, scalar_potential=false, tune=true)
# @time fmm!(system; ε_abs, scalar_potential=false, optargs...)
# @time fmm!(system; ε_abs, scalar_potential=false, optargs...)

# check direct cost
println("\n\ndirect cost")
@time direct!(system; scalar_potential=false)
velocity = system.velocity_stretching[1:3,:]

println("\n\nwith manually tuned arguments")
optargs, _ = fmm!(system; expansion_order, ε_abs, leaf_size_source, multipole_acceptance, scalar_potential=false, tune=false)
@time fmm!(system; expansion_order, ε_abs, leaf_size_source, multipole_acceptance, scalar_potential=false, tune=false, optargs...)
@time fmm!(system; expansion_order, ε_abs, leaf_size_source, multipole_acceptance, scalar_potential=false, tune=false, optargs...)
@profview fmm!(system; expansion_order, ε_abs, leaf_size_source, multipole_acceptance, scalar_potential=false, tune=false)

println("\n\nwith autotuned arguments")
optargs = tune_fmm!(system; ε_abs, scalar_potential=false)
@time fmm!(system; ε_abs, scalar_potential=false, tune=false, optargs...)
@time fmm!(system; ε_abs, scalar_potential=false, tune=false, optargs...)
@time fmm!(system; ε_abs, scalar_potential=false, tune=false, optargs...)
@profview fmm!(system; ε_abs, scalar_potential=false, optargs...)

println("done")