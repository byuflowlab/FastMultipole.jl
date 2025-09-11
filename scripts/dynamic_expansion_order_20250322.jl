using FastMultipole
using Random
using Statistics
using LinearAlgebra
using LaTeXStrings
using PythonPlot

include("../test/gravitational.jl")
include("../test/vortex.jl")

function get_gradient(system::Gravitational)
    return system.potential[5:7,:]
end

function get_gradient(system::VortexParticles)
    return system.velocity_stretching[1:3,:]
end

function get_potential(system::Gravitational)
    return system.potential[1,:]
end

function check_error(system, v_true; optargs...)
    
    # using expansions
    reset!(system)
    _, _, target_tree, source_tree, m2l_list, direct_list, _ = fmm!(system; optargs...)
    v_fmm = get_gradient(system)

    # velocity error
    errs_v = v_true - v_fmm
    errs_v_mag = sqrt.(sum(errs_v.^2, dims=1))
    min_err, q25, q50, q75, max_err = quantile(errs_v_mag, (0.0, 0.25, 0.5, 0.75, 1.0))
    @show min_err, q25, q50, q75, max_err

    return min_err, q25, q50, q75, max_err, target_tree, source_tree
end

function test_dynamic_p(system, εs_abs; optargs...)
    # get true velocity
    reset!(system)
    direct!(system)
    v_true = get_gradient(system)

    # get errors
    errs_ub = zeros(length(εs_abs), 5)
    errs_ts = zeros(length(εs_abs), 5)
    for (i, ε_abs) in enumerate(εs_abs)
        min_err, q25, q50, q75, max_err, target_tree, source_tree = check_error(system, v_true; optargs..., ε_abs=FastMultipole.UpperBound{ε_abs}())
        errs_ub[i,:] = [min_err, q25, q50, q75, max_err]
        
        min_err, q25, q50, q75, max_err, target_tree, source_tree = check_error(system, v_true; optargs..., ε_abs)
        errs_ts[i,:] = [min_err, q25, q50, q75, max_err]
    end
    
    return errs_ub, errs_ts
end

n_bodies = 10_000
system = generate_gravitational(123, n_bodies; strength_scale=1/n_bodies)
# system = generate_vortex(123, n_bodies; strength_scale=1/n_bodies/0.07891333941819026)

εs_abs = 10.0 .^ (-3:0.5:-1)
errs_ub, errs_ts = test_dynamic_p(system, εs_abs; lamb_helmholtz=typeof(system)<:VortexParticles, leaf_size_source=SVector{1}(50), multipole_acceptance=0.5)


fig = figure("dynamic P")
fig.clear()
ax = fig.add_subplot(111, xlabel="absolute error tolerance", ylabel="measured error")
ax.boxplot(
    [errs_ub[:, i] for i in 1:5],
    positions=1:length(εs_abs),
    widths=0.3,
    patch_artist=true,
    boxprops=Dict("facecolor" => "blue", "alpha" => 0.5),
    medianprops=Dict("color" => "black")
)
ax.boxplot(
    [errs_ts[:, i] for i in 1:5],
    positions=(1:length(εs_abs)) .+ 0.4,
    widths=0.3,
    patch_artist=true,
    boxprops=Dict("facecolor" => "orange", "alpha" => 0.5),
    medianprops=Dict("color" => "black")
)
ax.set_xticks(1:length(εs_abs))
ax.set_xticklabels(string.(εs_abs))
ax.legend(["Upper Bound", "True Error"])
ax.set_yscale("log")
fig.savefig("dynamic_p.png")
println("done")