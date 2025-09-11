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

function check_error(system, ϕ_true, velocity_true; optargs...)

    reset!(system)
    optargs, _ = fmm!(system; tune=true, optargs...)

    # potential error
    ϕ_fmm = get_potential(system)
    errs_ϕ = norm(ϕ_true - ϕ_fmm, 2)
    errs_ϕ_rel = errs_ϕ ./ ϕ_true
    l2_err_ϕ = norm(errs_ϕ_rel)
    max_err_ϕ_rel = maximum(errs_ϕ_rel)

    # velocity error
    velocity_fmm = get_gradient(system)
    diff = velocity_true - velocity_fmm
    errs2 = diff .* diff
    mag2 = velocity_true .* velocity_true
    rel_err = sqrt(sum(errs2) / sum(mag2))
    max_err_v_rel = maximum(sqrt.(sum(errs2, dims=1) ./ sum(mag2, dims=1)))

    return optargs, (mean(errs_ϕ), maximum(errs_ϕ), mean(sqrt.(sum(errs2,dims=1))), maximum(sqrt.(sum(errs2,dims=1))), l2_err_ϕ, max_err_ϕ_rel, rel_err, max_err_v_rel)
end

n_bodies = 30_000
system = generate_gravitational(123, n_bodies; strength_scale=1/n_bodies)
# system = generate_vortex(123, n_bodies; strength_scale=1/n_bodies/0.07891333941819026)
direct!(system)
ϕ_true = get_potential(system)
v_true = get_gradient(system)


function test_error_ub(system, ϕ_true, v_true, theta)
    println("\n==== begin theta = $theta =====\n")
    ps = 1:16
    res = zeros(8, length(ps))
    optargs, cache, _ = fmm!(system; lamb_helmholtz=false, tune=true, expansion_order=1, multipole_acceptance=theta)
    leaf_size_source = optargs.leaf_size_source
    for p in ps
        println("P = $p")
        # mean_phi, max_phi, mean_v, max_v, l2_phi_rel, max_phi_rel, l2_v_rel, max_v_rel
        @show leaf_size_source
        optargs, stuff = check_error(system, ϕ_true, v_true; leaf_size_source, expansion_order=p, multipole_acceptance=theta, lamb_helmholtz=false, scalar_potential=true, cache)
        leaf_size_source = optargs.leaf_size_source
        for i in 1:8
            res[i,p] = stuff[i]
        end
    end

    ϕ_ub = theta .^ Float64.(ps)

    fig = figure("error")
    fig.clear()
    fig.add_subplot(111, xlabel="expansion order", ylabel="error", title=L"\theta="*"$theta")
    ax = fig.get_axes()[0]
    ax.plot(res[7,:])
    ax.plot(res[8,:])
    ax.plot(ϕ_ub)
    ax.legend(["Yokota error", "max relative velocity magnitude", "traditional upper bound"])
    ax.set_yscale("log")
    tight_layout()

    fig.savefig("yokota_ryan_theta$theta.png")
end

for theta in 0.3:0.1:0.8
    test_error_ub(system, ϕ_true, v_true, theta)
end

println("done")
