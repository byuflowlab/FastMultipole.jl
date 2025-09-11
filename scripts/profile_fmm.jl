using FastMultipole
using Random
using Statistics
using PProf

include("../test/gravitational.jl")
include("../test/vortex.jl")

function grav_system(n_bodies; rand_seed=123, expansion_order=5, leaf_size_source=40, multipole_acceptance=0.5)
    system = generate_gravitational(rand_seed, n_bodies)

    # optimal args and cache
    lamb_helmholtz = false
    optargs, cache, _ = fmm!(system; tune=true, expansion_order, leaf_size_source, multipole_acceptance, lamb_helmholtz)
    optargs, cache, _ = fmm!(system; tune=true, optargs..., lamb_helmholtz)

    return system, optargs, cache, lamb_helmholtz
end

function vort_system(n_bodies; rand_seed=123)
    system = generate_vortex(rand_seed, n_bodies)

    # optimal args and cache
    lamb_helmholtz = false
    optargs, cache, _ = fmm!(system; tune=true, expansion_order, leaf_size_source, multipole_acceptance, lamb_helmholtz)
    optargs, cache, _ = fmm!(system; tune=true, optargs..., lamb_helmholtz)

    return system, optargs, cache, lamb_helmholtz
end

#--- create systems ---#

# n_bodies = 10000
# n_bodies = 10000
n_bodies = 262144
system, optargs, cache, lamb_helmholtz = grav_system(n_bodies)

direct() = direct!(system)
fmm_prof() = fmm!(system; optargs..., cache, lamb_helmholtz)

#--- VS Code Profiler
# direct()
# @profview direct()
fmm_prof()
@profview fmm_prof()

#--- PProf Profiler
# using Profile
# using PProf

# @profile fmm!(system; optargs..., cache, lamb_helmholtz)
# Profile.clear()
# @profile fmm!(system; optargs..., cache, lamb_helmholtz)
# pprof()
