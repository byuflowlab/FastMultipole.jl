using FastMultipole
using Random
using Statistics
# using PProf

include("../test/gravitational.jl")
include("../test/vortex.jl")

function grav_system(n_bodies; rand_seed=123, expansion_order=5, leaf_size_source=40, multipole_acceptance=0.5, interaction_list_method=FastMultipole.SelfTuningTreeStop())
    system = generate_gravitational(rand_seed, n_bodies)

    # optimal args and cache
    @time optargs, cache, _ = fmm!(system; tune=true, expansion_order, leaf_size_source, multipole_acceptance, interaction_list_method)
    @time optargs, cache, _ = fmm!(system, cache; tune=true, expansion_order=optargs.expansion_order, leaf_size_source=optargs.leaf_size_source, multipole_acceptance=optargs.multipole_acceptance, interaction_list_method)
    @time optargs, cache, _ = fmm!(system, cache; tune=true, expansion_order=optargs.expansion_order, leaf_size_source=optargs.leaf_size_source, multipole_acceptance=optargs.multipole_acceptance, interaction_list_method)
    @time optargs, cache, _ = fmm!(system, cache; tune=true, expansion_order=optargs.expansion_order, leaf_size_source=optargs.leaf_size_source, multipole_acceptance=optargs.multipole_acceptance, interaction_list_method)
    @time optargs, cache, _ = fmm!(system, cache; tune=true, expansion_order=optargs.expansion_order, leaf_size_source=optargs.leaf_size_source, multipole_acceptance=optargs.multipole_acceptance, interaction_list_method)

    return system, optargs, cache
end

function vort_system(n_bodies; rand_seed=123)
    system = generate_vortex(rand_seed, n_bodies)

    # optimal args and cache
    optargs, cache, _ = fmm!(system; tune=true, expansion_order, leaf_size_source, multipole_acceptance)
    optargs, cache, _ = fmm!(system, cache; tune=true, optargs...)
    optargs, cache, _ = fmm!(system, cache; tune=true, optargs...)
    optargs, cache, _ = fmm!(system, cache; tune=true, optargs...)

    return system, optargs, cache
end

#--- create systems ---#

# n_bodies = 10000
# n_bodies = 10000
# n_bodies = 262144
n_bodies = 10_000
system, optargs, cache = grav_system(n_bodies; interaction_list_method=FastMultipole.SelfTuningTargetStop());
@show optargs

# direct() = direct!(system)
# fmm_prof() = fmm!(system; optargs..., cache, lamb_helmholtz)


println("done")



#--- VS Code Profiler
# direct()
# @profview direct()
# fmm_prof()
# @profview fmm_prof()

#--- PProf Profiler
# using Profile
# using PProf

# @profile fmm!(system; optargs..., cache, lamb_helmholtz)
# Profile.clear()
# @profile fmm!(system; optargs..., cache, lamb_helmholtz)
# pprof()
