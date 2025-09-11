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

function check_error(target_system, source_system; optargs...)
    
    # using expansions
    reset!(target_system)
    leaf_size_source = SVector{1}(get_n_bodies(source_system) * 1000)
    _, _, target_tree, source_tree, m2l_list, direct_list, _ = fmm!(target_system, source_system; leaf_size_source, scalar_potential=true, optargs...)
    @assert length(target_tree.branches) == 1
    @assert length(source_tree.branches) == 1
    # @assert length(m2l_list) == 1
    # @assert length(direct_list) == 0 "diret list has $(length(direct_list)) elements"
     ϕ_fmm = get_potential(target_system)

    # direct
    reset!(target_system)
    direct!(target_system, source_system; scalar_potential=true)
    ϕ_true = get_potential(target_system)

    # potential error
    errs_ϕ = ϕ_true - ϕ_fmm
    max_err = maximum(abs.(errs_ϕ))

    return errs_ϕ, max_err, target_tree, source_tree
end

"""
    vector_to_closest_point(point, center, box)

Finds the vector from a point to the closest point on a rectangular region defined by its center and half-dimensions (box).
"""
function vector_to_closest_point(point, center, box)
    return clamp.(point - center, -box, box) + center - point
end

function error_ub(target_system, target_tree, source_system, source_tree, A, expansion_order)
    @assert length(target_tree.branches) == 1
    @assert length(source_tree.branches) == 1

    # cluster parameters
    target_center = target_tree.branches[1].target_center
    target_box = target_tree.branches[1].target_box
    source_center = source_tree.branches[1].source_center
    
    # multipole error
    a = source_tree.branches[1].source_radius
    P = expansion_order
    dx_multipole = vector_to_closest_point(source_center, target_center, target_box)
    r = norm(dx_multipole)
    c = r / a
    err_multipole = A / ((c - 1) * a) * (1 / c)^(P + 1)
    
    # local error
    a = target_tree.branches[1].target_radius
    ρ = norm(vector_to_closest_point(target_center, source_center, source_tree.branches[1].source_box))
    c = ρ / a - 1
    err_local = A / ((c - 1) * a) * (1 / c)^(P + 1)

    # total error
    err_ub = err_multipole + err_local

    return err_ub
end

function test_error_ub(target_system, source_system; expansion_order, optargs...)
    # get actual FMM error
    errs, max_err, target_tree, source_tree = check_error(target_system, source_system; expansion_order, optargs...)
    
    # get error upper bound
    A = abs(source_tree.expansions[1,1,1,1])
    err_ub = error_ub(target_system, target_tree, source_system, source_tree, A, expansion_order)

    return errs, max_err, err_ub
end

n_bodies = 100
source_system = generate_gravitational(123, n_bodies; strength_scale=1/n_bodies)
target_system = generate_gravitational(123, n_bodies; strength_scale=1/n_bodies, xshift=SVector{3}(3.0,0,0))
# system = generate_vortex(123, n_bodies; strength_scale=1/n_bodies/0.07891333941819026)

max_errs = []
err_ubs = []
expansion_orders = 1:20
for expansion_order = expansion_orders
    errs, max_err, err_ub = test_error_ub(target_system, source_system; expansion_order, multipole_acceptance=5.0, lamb_helmholtz=false)
    push!(max_errs, max_err)
    push!(err_ubs, err_ub)
end

fig = figure("error")
fig.clear()
ax = fig.add_subplot(111, xlabel="expansion order", ylabel="error")
ax.plot(expansion_orders, max_errs)
ax.plot(expansion_orders, err_ubs)
fig.savefig("error_ub_vs_fmm_error.png")
ax.legend(["fmm error", "Pringle upper bound"])
ax.set_yscale("log")
