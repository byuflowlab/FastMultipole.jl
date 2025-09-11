using FastMultipole
using Random
using Statistics

include("../test/gravitational.jl")
include("../test/vortex.jl")

function check_error(system, velocity_true, ε_abs;
        lamb_helmholtz,
        expansion_order, leaf_size_source, multipole_acceptance
    )

    return quantile(errs, (0.0, 0.25, 0.5, 0.75, 1.0))
end

n_bodies = 30_000
ε_abs = 1e-3
lamb_helmholtz = true
# system = generate_gravitational(123, n_bodies; strength_scale=1/n_bodies/0.07891333941819026)
system = generate_vortex(123, n_bodies; strength_scale=1/n_bodies/0.07891333941819026)
direct!(system)
v_true = system.potential[5:7,:]

# @time tuned_params, cache, t_opt = tune_fmm!(system; ε_abs, lamb_helmholtz, verbose=true)

@time tuned_params, cache = FastMultipole.tune_fmm!(system; ε_abs, lamb_helmholtz, verbose=true)

# optargs, cache, _ = fmm!(system; ε_abs, lamb_helmholtz, tune=true)

# fmm!(system; ε_abs, lamb_helmholtz, cache, optargs...)
# @profview fmm!(system; ε_abs, lamb_helmholtz, cache, optargs...)

# expansion_order, leaf_size_source, multipole_acceptance = 14, SVector{1}(123), 0.6

println("done")
