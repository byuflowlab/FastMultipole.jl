using FastMultipole
using Statistics
using Random

include("../test/gravitational.jl")
include("../test/vortex.jl")

n_bodies = 50_000
# masses = generate_gravitational(123, n_bodies)
vort = generate_vortex(123, n_bodies; strength_scale=4.5/n_bodies)


tuned_params, cache = tune_fmm(vort; lamb_helmholtz=true, ε_abs=1e-6)

@time fmm!(vort; lamb_helmholtz=true, ε_abs=1e-6, cache, tuned_params...)
@time fmm!(vort; lamb_helmholtz=true, ε_abs=1e-6, cache, tuned_params...)
@time fmm!(vort; lamb_helmholtz=true, ε_abs=1e-6, cache, tuned_params...)
println("done")