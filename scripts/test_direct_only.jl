using FastMultipole
using Random
using Statistics

include("../test/gravitational.jl")
include("../test/vortex.jl")

n_bodies = 10_000
ε_abs = 1e-3
lamb_helmholtz = true
# system = generate_gravitational(123, n_bodies; strength_scale=1/n_bodies/0.07891333941819026)
system = generate_vortex(123, n_bodies; strength_scale=1/n_bodies/0.07891333941819026)
direct!(system)
v_true = system.velocity_stretching[1:3,:]

# @time tuned_params, cache, t_opt = tune_fmm!(system; ε_abs, lamb_helmholtz, verbose=true)

# @time tuned_params, cache = FastMultipole.tune_fmm!(system; ε_abs, lamb_helmholtz, verbose=true)
reset!(system)
optimized_args, cache, target_tree, source_tree, m2l_list, direct_list, derivatives_switches, error_success = fmm!(system; lamb_helmholtz, tune=true, farfield=false, nearfield=true, self_induced=true, multipole_acceptance=0.1, leaf_size_source=SVector{1}(10), shrink_recenter=false)
v_direct_only = system.velocity_stretching[1:3,:]

diff = v_true - v_direct_only
diff2 = diff .* diff
diffnorm = sqrt.(sum(diff2, dims=1))

mag_true = v_true .* v_true
mag_true = sqrt.(sum(mag_true, dims=1))

@show mean(mag_true)
@show std(mag_true)
@show maximum(mag_true)

@show mean(diffnorm)
@show std(diffnorm)
@show maximum(diffnorm)
@show minimum(diffnorm)

println("done")
