#=
this file is used to generate Figure 5 in the paper (examples_verify_prediction_source.pdf and examples_verify_prediction_vortex.pdf)
=#

using FastMultipole
using Statistics
using Random
using BSON
using LinearAlgebra
using PythonPlot

include("../test/evaluate_multipole.jl")
include("../test/bodytolocal.jl")
include("../test/gravitational.jl")
include("../test/vortex.jl")

function get_gradient(system::Gravitational)
    return system.potential[5:7,:]
end

function get_gradient(system::VortexParticles)
    return system.velocity_stretching[1:3,:]
end

function velocity_err!(v_fmm, v_direct, bodies_index)
    # difference
    for d in 1:3
        for (i,ib) in enumerate(bodies_index)
            v_fmm[d, i] -= v_direct[d+4, ib]
        end
    end

    # square
    for d in 1:3
        for i in eachindex(bodies_index)
            v_fmm[d, i] *= v_fmm[d, i]
        end
    end

    # sum to d=1
    for d in 2:3
        for i in eachindex(bodies_index)
            v_fmm[1, i] += v_fmm[d, i]
        end
    end
    
    # sqrt
    for i in eachindex(bodies_index)
        v_fmm[1, i] = sqrt(v_fmm[1, i])
    end
end

function get_potential(system::Gravitational)
    return system.potential[1,:]
end
    
function get_potential(system::VortexParticles)
    return system.potential[1,:]
end

function potential_err!(ϕ_fmm, ϕ_direct, bodies_index)
    # difference
    for i in eachindex(bodies_index)
        ϕ_fmm[i] = abs(ϕ_direct[4, bodies_index[i]] - ϕ_fmm[1,i])
    end   
end

function test_accuracy(source_systems::Tuple, r_l_over_r_mp_list, expansion_order, max_expansion_order, multipole_acceptance, lamb_helmholtz, dx_vec)

    # wrap val
    lamb_helmholtz_bool = lamb_helmholtz
    lamb_helmholtz = Val(lamb_helmholtz_bool)

    # create source tree
    leaf_size_source = SVector{1}(get_n_bodies(source_systems))
    source_tree = Tree(source_systems, false; expansion_order=max_expansion_order, leaf_size=leaf_size_source, shrink_recenter=false)
    source_buffers = source_tree.buffers

    # upward pass
    FastMultipole.upward_pass_singlethread!(source_tree, source_systems, max_expansion_order, lamb_helmholtz)

    # outputs
    max_errs_list = Vector{Vector{Float64}}()
    max_v_errs_list = Vector{Vector{Float64}}()
    max_mp_errs_list = Vector{Vector{Float64}}()
    max_v_mp_errs_list = Vector{Vector{Float64}}()
    max_l_errs_list = Vector{Vector{Float64}}()
    max_v_l_errs_list = Vector{Vector{Float64}}()
    ε_mp_hat_list = Vector{Vector{Float64}}()
    ε_l_hat_list = Vector{Vector{Float64}}()
    ε_v_mp_hat_list = Vector{Vector{Float64}}()
    ε_v_l_hat_list = Vector{Vector{Float64}}()

    # preallocate containers to be reused
    weights_tmp_1 = initialize_expansion(max_expansion_order, Float64)
    weights_tmp_2 = initialize_expansion(max_expansion_order, Float64)
    weights_tmp_3 = initialize_expansion(max_expansion_order, Float64)
    Ts = zeros(Float64, FastMultipole.length_Ts(max_expansion_order))
    eimϕs = zeros(Float64, 2, max_expansion_order + 1)
    harmonics = initialize_harmonics(max_expansion_order)
    velocity_n_m = FastMultipole.initialize_gradient_n_m(max_expansion_order, Float64)
    derivatives_switches = DerivativesSwitch(true, true, false, (nothing,))

    for r_l_over_r_mp in r_l_over_r_mp_list

        # determine target system size
        r_mp = source_tree.branches[1].source_radius
        r_l = r_l_over_r_mp * r_mp
        target_center_distance = (r_mp + r_l) / multipole_acceptance
        source_center = source_tree.branches[1].source_center
        # target_center = SVector{3}(source_center[1] + target_center_distance, source_center[2], source_center[3])
        target_center = source_center + dx_vec * target_center_distance
        # target_x = SVector{3}(target_center[1] - r_l, target_center[2], target_center[3])
        target_x = target_center - dx_vec * r_l

        # println("r_l/r_mp = $r_l_over_r_mp")
        # @show source_center, target_center, target_x

        # create target system
        target_bodies = zeros(5, 1)
        target_bodies[1:3, 1] = target_x
        target_system = Gravitational(target_bodies)
        target_systems = (target_system,)

        # create target branch
        radius = norm(target_x - target_center)
        box = abs.(SVector{3}(target_x - target_center))
        target_branch = FastMultipole.Branch(SVector{1}([1:1]), 0, 1:0, -1, 1, target_center, target_center, radius, radius, box, box)
        
        # create expansions
        expansions = FastMultipole.initialize_expansions(max_expansion_order, 1, Float64)

        # generate buffers
        target_buffers = FastMultipole.allocate_buffers((target_system,), true)
        FastMultipole.target_to_buffer!(target_buffers, (target_system,))
        small_buffers = FastMultipole.allocate_small_buffers((target_system,))

        # create target tree
        target_tree = Tree([target_branch], expansions, [1:1], [1], ([1],), ([1],), target_buffers, small_buffers, max_expansion_order, SVector{1}(1))

        # get m2l list
        m2l_list, _ = FastMultipole.build_interaction_lists(target_tree.branches, source_tree.branches, leaf_size_source, 1.0, true, true, true, SelfTuning())
        m2l_list = [SVector{2}(Int32(1), Int32(1))]

        # target velocity container
        target_potential = Tuple(zeros(Float64, 1, get_n_bodies(target_systems[i])) for i in 1:length(target_systems))
        target_velocity = Tuple(zeros(Float64, 3, get_n_bodies(target_systems[i])) for i in 1:length(target_systems))
        multipole_potential = Tuple(zeros(Float64, 1, get_n_bodies(target_systems[i])) for i in 1:length(target_systems))
        multipole_velocity = Tuple(zeros(Float64, 3, get_n_bodies(target_systems[i])) for i in 1:length(target_systems))
        local_potential = Tuple(zeros(Float64, 1, get_n_bodies(target_systems[i])) for i in 1:length(target_systems))
        local_velocity = Tuple(zeros(Float64, 3, get_n_bodies(target_systems[i])) for i in 1:length(target_systems))

        # outputs
        max_errs = zeros(length(m2l_list))
        max_v_errs = zeros(length(m2l_list))
        max_mp_errs = zeros(length(m2l_list))
        max_v_mp_errs = zeros(length(m2l_list))
        max_l_errs = zeros(length(m2l_list))
        max_v_l_errs = zeros(length(m2l_list))
        ε_mp_hat = zeros(length(m2l_list))
        ε_l_hat = zeros(length(m2l_list))
        ε_v_mp_hat = zeros(length(m2l_list))
        ε_v_l_hat = zeros(length(m2l_list))

        # manual horizontal pass
        # println("\n\t#--- Manual Horizontal Pass ---#\n")
        for (i,(i_target, j_source)) in enumerate(m2l_list)
            # if i % 1000 == 0
            #     println("\t\ti = $i / $(length(m2l_list))")
            # end

            #--- expansions ---#
            
            # perform M2L
            target_branch = target_tree.branches[i_target]
            target_expansion = view(target_tree.expansions, :, :, :, i_target)
            target_expansion .= 0.0
            source_branch = source_tree.branches[j_source]
            source_expansion = view(source_tree.expansions, :, :, :, j_source)
            FastMultipole.multipole_to_local!(target_expansion, target_branch, source_expansion, source_branch, weights_tmp_1, weights_tmp_2, weights_tmp_3, Ts, eimϕs, FastMultipole.ζs_mag, FastMultipole.ηs_mag, FastMultipole.Hs_π2, FastMultipole.M̃, FastMultipole.L̃, expansion_order, lamb_helmholtz, nothing)

            # evaluate local expansion
            FastMultipole.reset!(target_buffers)
            for (i_system, system) in enumerate(target_buffers)
                FastMultipole.evaluate_local!(system, target_branch.bodies_index[i_system], harmonics, velocity_n_m, target_expansion, target_branch.target_center, expansion_order, lamb_helmholtz, derivatives_switches[i_system])
            end
            
            # save velocity
            for i_target_system in 1:length(target_buffers)
                target_potential[i_target_system][:, 1:length(target_branch.bodies_index[i_target_system])] .= target_buffers[i_target_system][4, target_branch.bodies_index[i_target_system]]
                target_velocity[i_target_system][:, 1:length(target_branch.bodies_index[i_target_system])] .= target_buffers[i_target_system][5:7, target_branch.bodies_index[i_target_system]]
            end
            
            # evaluate multipole expansion
            FastMultipole.reset!(target_buffers)
            for (i_system, system) in enumerate(target_buffers)
                evaluate_multipole!(system, target_branch.bodies_index[i_system], harmonics, source_expansion, source_branch.source_center, expansion_order, lamb_helmholtz, derivatives_switches[i_system])
            end
            
            # save velocity
            for i_target_system in 1:length(target_buffers)
                multipole_potential[i_target_system][:, 1:length(target_branch.bodies_index[i_target_system])] .= target_buffers[i_target_system][4, target_branch.bodies_index[i_target_system]]
                multipole_velocity[i_target_system][:, 1:length(target_branch.bodies_index[i_target_system])] .= target_buffers[i_target_system][5:7, target_branch.bodies_index[i_target_system]]
            end
            
            #--- evaluate local expansion ---#

            target_expansion .= 0.0
            FastMultipole.reset!(target_buffers)

            # compute local coefficients
            for (i_system, system) in enumerate(source_systems)
                source_buffer = source_buffers[i_system]
                source_index = source_tree.branches[j_source].bodies_index[i_system]
                for i_body in source_index
                    Δx = FastMultipole.get_position(source_buffer, i_body) - target_branch.target_center
                    strength = FastMultipole.get_strength(source_buffer, system, i_body)
                    body_to_local_point!(Point{Source}, target_expansion, harmonics, Δx, strength[1], expansion_order+1)
                end
            end

            # evaluate local expansion
            FastMultipole.reset!(target_buffers)
            for (i_system, system) in enumerate(target_buffers)
                FastMultipole.evaluate_local!(system, target_branch.bodies_index[i_system], harmonics, velocity_n_m, target_expansion, target_branch.target_center, expansion_order, lamb_helmholtz, derivatives_switches[i_system])
            end
            
            # save velocity
            for i_target_system in 1:length(target_buffers)
                local_potential[i_target_system][:, 1:length(target_branch.bodies_index[i_target_system])] .= target_buffers[i_target_system][4, target_branch.bodies_index[i_target_system]]
                local_velocity[i_target_system][:, 1:length(target_branch.bodies_index[i_target_system])] .= target_buffers[i_target_system][5:7, target_branch.bodies_index[i_target_system]]
            end

            #--- predict error ---#

            ε_mp, ε_l = FastMultipole.predict_error(target_branch, source_expansion, source_branch, weights_tmp_1, weights_tmp_2, weights_tmp_3, Ts, eimϕs, FastMultipole.ζs_mag, FastMultipole.Hs_π2, FastMultipole.M̃, FastMultipole.L̃, expansion_order, lamb_helmholtz, FastMultipole.PowerAbsolutePotential{1.0,false}())
            ε_v_mp, ε_v_l = FastMultipole.predict_error(target_branch, source_expansion, source_branch, weights_tmp_1, weights_tmp_2, weights_tmp_3, Ts, eimϕs, FastMultipole.ζs_mag, FastMultipole.Hs_π2, FastMultipole.M̃, FastMultipole.L̃, expansion_order, lamb_helmholtz, FastMultipole.PowerAbsoluteVelocity{1.0,false}())
            ε_mp_hat[i] = ε_mp
            ε_l_hat[i] = ε_l
            ε_v_mp_hat[i] = ε_v_mp
            ε_v_l_hat[i] = ε_v_l

            #--- direct ---#
            
            # reset systems
            FastMultipole.reset!(target_buffers)

            # loop over source systems
            for i_source_system in eachindex(source_systems)
                source_system = source_systems[i_source_system]
                source_buffer = source_buffers[i_source_system]
        
                # loop over target systems
                for (i_target_system, target_system) in enumerate(target_buffers)

                    # extract derivatives switch
                    derivatives_switch = derivatives_switches[i_target_system]

                    # identify sources
                    source_index = source_tree.branches[j_source].bodies_index[i_source_system]

                    # identify targets
                    target_index = target_tree.branches[i_target].bodies_index[i_target_system]

                    # compute interaction
                    # @show target_buffers[1] target_potential[1]
                    direct!(target_system, target_index, derivatives_switch, source_system, source_buffer, source_index)

                end
            end

            # calculate full expansion error
            for i_target_system in 1:length(target_buffers)
                potential_err!(target_potential[i_target_system], target_buffers[i_target_system], target_branch.bodies_index[i_target_system])
                max_errs[i] = max(max_errs[i], maximum(view(target_potential[i_target_system],1,1:length(target_branch.bodies_index[i_target_system]))))
                velocity_err!(target_velocity[i_target_system], target_buffers[i_target_system], target_branch.bodies_index[i_target_system])
                max_v_errs[i] = max(max_v_errs[i], maximum(view(target_velocity[i_target_system],1,1:length(target_branch.bodies_index[i_target_system]))))
            end

            # just multipole error
            for i_target_system in 1:length(target_buffers)
                potential_err!(multipole_potential[i_target_system], target_buffers[i_target_system], target_branch.bodies_index[i_target_system])
                max_mp_errs[i] = max(max_mp_errs[i], maximum(view(multipole_potential[i_target_system],1,1:length(target_branch.bodies_index[i_target_system]))))
                velocity_err!(multipole_velocity[i_target_system], target_buffers[i_target_system], target_branch.bodies_index[i_target_system])
                max_v_mp_errs[i] = max(max_v_mp_errs[i], maximum(view(multipole_velocity[i_target_system],1,1:length(target_branch.bodies_index[i_target_system]))))
            end

            # just local error
            for i_target_system in 1:length(target_buffers)
                potential_err!(local_potential[i_target_system], target_buffers[i_target_system], target_branch.bodies_index[i_target_system])
                max_l_errs[i] = max(max_l_errs[i], maximum(view(local_potential[i_target_system],1,1:length(target_branch.bodies_index[i_target_system]))))
                velocity_err!(local_velocity[i_target_system], target_buffers[i_target_system], target_branch.bodies_index[i_target_system])
                max_v_l_errs[i] = max(max_v_l_errs[i], maximum(view(local_velocity[i_target_system],1,1:length(target_branch.bodies_index[i_target_system]))))
            end
        end

        push!(max_errs_list, max_errs)
        push!(max_v_errs_list, max_v_errs)
        push!(max_mp_errs_list, max_mp_errs)
        push!(max_v_mp_errs_list, max_v_mp_errs)
        push!(max_l_errs_list, max_l_errs)
        push!(max_v_l_errs_list, max_v_errs - max_v_mp_errs)
        # push!(max_v_l_errs_list, max_v_l_errs)
        push!(ε_mp_hat_list, ε_mp_hat)
        push!(ε_l_hat_list, ε_l_hat)
        push!(ε_v_mp_hat_list, ε_v_mp_hat)
        push!(ε_v_l_hat_list, ε_v_l_hat)

    end

    return max_errs_list, max_v_errs_list, max_mp_errs_list, max_v_mp_errs_list, max_l_errs_list, max_v_l_errs_list, ε_mp_hat_list, ε_l_hat_list, ε_v_mp_hat_list, ε_v_l_hat_list
end

function compile(max_errs, max_v_errs, max_mp_errs, max_v_mp_errs, max_l_errs, max_v_l_errs, ε_mp_hat, ε_l_hat, ε_v_mp_hat, ε_v_l_hat, i_target_systems)
    # compile
    max_errs = map(x -> x[i_target_systems], max_errs)
    max_v_errs = map(x -> x[i_target_systems], max_v_errs)
    max_mp_errs = map(x -> x[i_target_systems], max_mp_errs)
    max_v_mp_errs = map(x -> x[i_target_systems], max_v_mp_errs)
    max_l_errs = map(x -> x[i_target_systems], max_l_errs)
    max_v_l_errs = map(x -> x[i_target_systems], max_v_l_errs)
    ε_mp_hat = map(x -> x[i_target_systems], ε_mp_hat)
    ε_l_hat = map(x -> x[i_target_systems], ε_l_hat)
    ε_v_mp_hat = map(x -> x[i_target_systems], ε_v_mp_hat)
    ε_v_l_hat = map(x -> x[i_target_systems], ε_v_l_hat)

    # # save to BSON
    # filename = "prediction_accuracy_$(n_bodies).bson"
    # BSON.@save filename max_errs max_mp_errs ε_mp_hat ε_l_hat

    return max_errs, max_v_errs, max_mp_errs, max_v_mp_errs, max_l_errs, max_v_l_errs, ε_mp_hat, ε_l_hat, ε_v_mp_hat, ε_v_l_hat
end

#--- create systems ---#

println("#----- Creating systems -----#")

# n_bodies = 50_000
# source = generate_gravitational(123, n_bodies; strength_scale=1.0/3798.955768976926)
# vortex = generate_vortex(123, n_bodies; strength_scale=1.0/10566.33461495282)

n_bodies_source = 300
function bodies_fun2(bodies) 
    bodies[1,:] .*= bodies[1,:]
    bodies[1,:] .*= bodies[1,:]
end
bodies_fun = (x) -> nothing
# source_systems = (generate_gravitational(123, n_bodies_source; strength_scale=1, bodies_fun=bodies_fun),)
source_systems = (generate_vortex(123, n_bodies_source; strength_scale=1), )

expansion_order = 4
max_expansion_order = 20
lamb_helmholtz = typeof(source_systems[1]) <: VortexParticles
@show lamb_helmholtz

multipole_acceptance = 0.8
r_l_over_r_mp = [1.0]
ψs = range(0, stop=pi, length=100)

prediction_over_measured_mp = zeros(length(ψs), length(r_l_over_r_mp))
measured_mp = similar(prediction_over_measured_mp)
predicted_mp = similar(prediction_over_measured_mp)
measured_mp_v = similar(prediction_over_measured_mp)
predicted_mp_v = similar(prediction_over_measured_mp)
measured_l = similar(prediction_over_measured_mp)
predicted_l = similar(prediction_over_measured_mp)
measured_l_v = similar(prediction_over_measured_mp)
predicted_l_v = similar(prediction_over_measured_mp)
measured_total = similar(prediction_over_measured_mp)
measured_total_v = similar(prediction_over_measured_mp)

ξ = pi/4
for (iψ, ψ) in enumerate(ψs)
    # dx_vec = SVector{3}(0.0, 0.0, 1.0)
    # dx_vec = SVector{3}(1.0, 0.0, 0.0)
    dx_vec = SVector{3}(sin(ψ), 0.0, cos(ψ))
    dx_vec = SVector{3}(sin(ψ) * cos(ξ), sin(ψ) * sin(ξ), cos(ψ))
    max_errs, max_v_errs, max_mp_errs, max_v_mp_errs, max_l_errs, max_v_l_errs, ε_mp_hat, ε_l_hat, ε_v_mp_hat, ε_v_l_hat = test_accuracy(source_systems, r_l_over_r_mp, expansion_order, max_expansion_order, multipole_acceptance, lamb_helmholtz, dx_vec)
    max_errs, max_v_errs, max_mp_errs, max_v_mp_errs, max_l_errs, max_v_l_errs, ε_mp_hat, ε_l_hat, ε_v_mp_hat, ε_v_l_hat = compile(max_errs, max_v_errs, max_mp_errs, max_v_mp_errs, max_l_errs, max_v_l_errs, ε_mp_hat, ε_l_hat, ε_v_mp_hat, ε_v_l_hat, 1)

    prediction_over_measured_mp[iψ,:] .= ε_mp_hat ./ max_mp_errs
    measured_mp[iψ,:] .= max_mp_errs
    predicted_mp[iψ,:] .= ε_mp_hat
    measured_mp_v[iψ,:] .= max_v_mp_errs
    predicted_mp_v[iψ,:] .= ε_v_mp_hat
    measured_l[iψ,:] .= max_l_errs
    predicted_l[iψ,:] .= ε_l_hat
    measured_l_v[iψ,:] .= max_v_l_errs
    predicted_l_v[iψ,:] .= ε_v_l_hat
    measured_total[iψ,:] .= max_errs
    measured_total_v[iψ,:] .= max_v_errs
end
predicted_total = predicted_mp .+ predicted_l
predicted_total_v = predicted_mp_v .+ predicted_l_v

# make plots

fig = figure("error", figsize=(10, 5))
fig.clear()
fig.add_subplot(121, xlabel="translation angle "*L"\psi", ylabel="predicted / measured error")
ax = fig.get_axes()[0]
for i_rr in eachindex(r_l_over_r_mp)
    ax.plot(ψs, prediction_over_measured_mp[:,i_rr], label="r_l/r_mp = $(r_l_over_r_mp[i_rr])")
end
ax.set_yscale("log")
ax.legend(["r_l/r_mp = $(r_l_over_r_mp[i_rr])" for i_rr in eachindex(r_l_over_r_mp)])

fig.add_subplot(122, xlabel="translation angle "*L"\psi", ylabel="measured error")
ax = fig.get_axes()[1]
for i_rr in eachindex(r_l_over_r_mp)
    ax.plot(ψs, measured_mp[:,i_rr], label="r_l/r_mp = $(r_l_over_r_mp[i_rr])")
end
# ax.legend(["r_l/r_mp = $(r_l_over_r_mp[i_rr])" for i_rr in eachindex(r_l_over_r_mp)])
ax.set_yscale("log")
tight_layout()

i_rr = 1
fig2 = figure("error2", figsize=(6,5))
fig2.clear()
fig2.add_subplot(111, xlabel="translation angle "*L"\psi", ylabel="multipole error")
ax2 = fig2.get_axes()[0]
ax2.plot(ψs, measured_mp[:,i_rr])
ax2.plot(ψs, predicted_mp[:,i_rr])
ax2.set_yscale("log")
ax2.legend(["measured", "predicted"])
tight_layout()

fig3 = figure("error3", figsize=(6,5))
fig3.clear()
fig3.add_subplot(111, xlabel="translation angle "*L"\psi", ylabel="local error")
ax3 = fig3.get_axes()[0]
ax3.plot(ψs, measured_l[:,i_rr])
ax3.plot(ψs, predicted_l[:,i_rr])
ax3.set_yscale("log")
ax3.legend(["measured", "predicted"])
tight_layout()

i_rr = 1
fig4 = figure("error_v_mp", figsize=(6,5))
fig4.clear()
fig4.add_subplot(111, xlabel="translation angle "*L"\psi", ylabel="multipole error")
ax4 = fig4.get_axes()[0]
ax4.plot(ψs, measured_mp_v[:,i_rr])
ax4.plot(ψs, predicted_mp_v[:,i_rr])
ax4.set_yscale("log")
ax4.legend(["measured", "predicted"])
tight_layout()

fig5 = figure("error_v_l", figsize=(6,5))
fig5.clear()
fig5.add_subplot(111, xlabel="translation angle "*L"\psi", ylabel="local error")
ax5 = fig5.get_axes()[0]
ax5.plot(ψs, measured_l_v[:,i_rr])
ax5.plot(ψs, predicted_l_v[:,i_rr])
ax5.set_yscale("log")
ax5.legend(["measured", "predicted"])
tight_layout()

fig6 = figure("error_v_total", figsize=(6,5))
fig6.clear()
fig6.add_subplot(111, xlabel="translation angle "*L"\psi", ylabel="local error")
ax6 = fig6.get_axes()[0]
ax6.plot(ψs, measured_total_v[:,i_rr])
ax6.plot(ψs, predicted_total_v[:,i_rr])
ax6.set_yscale("log")
ax6.legend(["measured", "predicted"])
tight_layout()