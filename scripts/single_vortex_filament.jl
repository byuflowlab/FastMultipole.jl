using FastMultipole
using Statistics
using StaticArrays
using LinearAlgebra
using WriteVTK

include("../test/vortex_filament.jl")
include("../test/vortex.jl")
include("rotor.jl")

# create a single vortex filament

function generate_filament(center, l, strength)
    x = zeros(SVector{3,Float64}, 2, 1)
    strength_vec = zeros(SVector{3,Float64}, 1)
    dx = strength / norm(strength) * l * 0.5
    x[1,1] = center - dx
    x[2,1] = center + dx
    strength_vec[1] = strength

    # create filaments
    potential = zeros(length(strength_vec))
    force = zeros(SVector{3,Float64}, length(strength_vec))
    gradient = zeros(SMatrix{3,3,Float64,9}, length(strength_vec))

    return VortexFilaments(x, strength_vec, potential, force, gradient)
end

function generate_two_filaments(centers, ls, strengths)
    x = zeros(SVector{3,Float64}, 2, 2)
    strength_vec = zeros(SVector{3,Float64}, 2)
    for i in 1:2
        center = centers[i]
        dx = strengths[i] / norm(strengths[i]) * ls[i] * 0.5
        x[1,i] = center - dx
        x[2,i] = center + dx
        strength_vec[i] = strengths[i]
    end

    # create filaments
    core_size = fill(1e-2, size(x, 2))
    error_tolerance = fill(1e-4, size(x, 2))
    potential = zeros(length(strength_vec))
    force = zeros(SVector{3,Float64}, length(strength_vec))
    gradient = zeros(SMatrix{3,3,Float64,9}, length(strength_vec))

    return VortexFilaments(x, strength_vec, core_size, error_tolerance, potential, force, gradient)
end

function generate_probes_xz(xspan, zspan, phi)
    n_bodies = length(xspan) * length(zspan)
    position = zeros(3, n_bodies)
    strength = zeros(3, n_bodies)
    sϕ, cϕ = sincos(phi)
    R = SMatrix{3,3}(cϕ, 0.0, -sϕ, 0.0, 1.0, 0.0, sϕ, 0.0, cϕ)

    i = 1
    for x in xspan
        for z in zspan
            position[:,i] .= R * SVector{3}(x,0.0,z)
            i += 1
        end
    end

    return VortexParticles(position, strength)
end

function generate_probes_octant(xspan, yspan, zspan)
    n_bodies = length(xspan) * length(yspan) * length(zspan)
    position = zeros(3, n_bodies)
    strength = zeros(3, n_bodies)

    i = 1
    for x in xspan
        for y in yspan
            for z in zspan
                position[:,i] .= SVector{3}(x,y,z)
                i += 1
            end
        end
    end

    return VortexParticles(position, strength)
end

#=
#------- test single vortex filament -------#

# generate filament
center = SVector{3}(0.0,0.0,0.0)
l = 1.0e-2
strength = SVector{3}(1.0,0.0,1.0)
fil = generate_filament(center, l, strength)

# generate probes
xspan = range(0, stop=10.0, length=100)
zspan = range(-5.0, stop=5.0, length=100)
phi = pi/5
# probes = generate_probes_xz(xspan, zspan, phi)
probes = generate_probes_octant(xspan)

# direct velocity
direct!(probes, fil)
# optargs, _ = fmm!(probes, fil; expansion_order=20, scalar_potential=false)
v_direct = probes.velocity_stretching[1:3,:]
reset!(probes)

# fmm velocity
optargs, tt, st, m2l_list, direct_list, _ = fmm!(probes, fil; expansion_order=10, scalar_potential=false, tune=true)
v_fmm = probes.velocity_stretching[1:3,:]
@show optargs.leaf_size_source

diff = v_direct - v_fmm
norms = sqrt.(sum(diff .* diff; dims=1))

@show maximum(norms) minimum(norms) mean(norms) std(norms)
=#


# #------- test dual vortex filament -------#

# # generate filament
# centers = [SVector{3}(-0.1e1,0.0,0.0), SVector{3}(0.1e1,0,0)]
# ls = [1.0e-2, 1.0e-2]
# strengths = [SVector{3}(0.0,0.0,1.0), SVector{3}(0.0,0.0,1.0)]
# fil = generate_two_filaments(centers, ls, strengths)

# # generate probes
# xspan = range(0, stop=10.0, length=100)
# zspan = range(-5.0, stop=5.0, length=100)
# phi = pi/5
# # probes = generate_probes_xz(xspan, zspan, phi)
# probes = generate_probes_octant(xspan, xspan, zspan)

# # direct velocity
# direct!(probes, fil)
# # optargs, _ = fmm!(probes, fil; expansion_order=20, scalar_potential=false)
# v_direct = probes.velocity_stretching[1:3,:]
# reset!(probes)

# # # fmm velocity
# # optargs, _, tt, st, m2l_list, direct_list, _ = fmm!(probes, fil; expansion_order=10, scalar_potential=false, tune=true, leaf_size_source=1)
# # v_fmm = probes.velocity_stretching[1:3,:]
# # @show optargs.leaf_size_source

# # diff = v_direct - v_fmm
# # norms = sqrt.(sum(diff .* diff; dims=1))

# # @show maximum(norms) minimum(norms) mean(norms) std(norms)

# # filaments on themselves
# reset!(fil)
# direct!(fil)
# v_direct_r = deepcopy(fil.force)
# reset!(fil)
# _, _, tt, st, m2l_list, direct_list, _ = fmm!(fil; leaf_size_source=1, multipole_acceptance=0.99999, expansion_order=10, lamb_helmholtz=true, error_tolerance=nothing)
# v_fmm_r = deepcopy(fil.force)
# diff_r = v_fmm_r - v_direct_r
# norms_r = [sqrt(sum(diff_r[i] .* diff_r[i])) for i in 1:length(diff_r)]
# @show maximum(norms_r) minimum(norms_r) mean(norms_r) std(norms_r)




#------- rotor on probes -------#

# viz_name_prefix = "one_blade_"
viz_name_prefix = "one_filament_"

# generate rotor
filaments = generate_rotor_blade()

# filaments = generate_rotor_blade_filament(; i_blade=1, i_filament=1)

#=
# from unit test
# x1 = SVector{3,Float64}(0.0,0.29,0.0)
x1 = SVector{3}(0.0027,0.004,0.006)
x2 = x1 + SVector{3,Float64}(0.0,0.03,0.0)
x = zeros(SVector{3,Float64},2,1)
q = SVector{3}(0,1.0,0)
x[1,1] = x1
x[2,1] = x2
filaments = VortexFilaments(x, [q]) # -q breaks it!
=#



# filaments = generate_rotor()
l = total_length(filaments)
n = size(filaments.x, 2)
refined_filaments = refine_filaments(filaments, l / n / 20)

# generate probes
xspan = range(0.0, stop=0.2, length=40)
yspan = range(0.0, stop=0.05, length=10)
zspan = range(0.0, stop=0.2, length=40)
probes = generate_probes_octant(xspan, yspan, zspan)

direct!(probes, filaments)
v_direct = probes.velocity_stretching[1:3,:]

save_vtk(viz_name_prefix * "v_direct", probes)

# reset!(filaments)
# direct!(filaments, refined_filaments)
# v_direct_r = deepcopy(filaments.force)
# @show maximum(norm.(v_direct - v_direct_r))

reset!(probes)
optargs, target_tree, source_tree, m2l_list, direct_list, derivatives_switches, error_success = fmm!(probes, filaments; leaf_size_source=20, multipole_acceptance=0.4, expansion_order=10, lamb_helmholtz=true, error_tolerance=nothing)

save_vtk(viz_name_prefix * "v_fmm", probes)

v_fmm = probes.velocity_stretching[1:3,:]

probes.velocity_stretching[1:3,:] .-= v_direct

save_vtk(viz_name_prefix * "v_error", probes)

diff = v_fmm - v_direct
norms = sqrt.(sum(diff .* diff; dims=1))
@show maximum(norms) minimum(norms) mean(norms) std(norms)

val, i = findmax(norms)

viz(viz_name_prefix * "filament", filaments)

# filaments on themselves
reset!(filaments)
direct!(filaments)
v_direct_r = deepcopy(filaments.force)

reset!(filaments)
_, _, _, _, m2l_list, direct_list, _ = fmm!(filaments; leaf_size_source=1, multipole_acceptance=0.4, expansion_order=20, lamb_helmholtz=true, error_tolerance=nothing)
v_fmm_r = deepcopy(filaments.force)
diff_r = v_fmm_r - v_direct_r
norms_r = [sqrt(sum(diff_r[i] .* diff_r[i])) for i in 1:length(diff_r)]
@show maximum(norms_r) minimum(norms_r) mean(norms_r) std(norms_r)

# mysterious case
BSON.@load "/Users/ryan/Downloads/vortices.bson" vortex_system wing_system rotor_system
reset!(vortex_system)
direct!(vortex_system)
v_direct_vs = deepcopy(vortex_system.force)
reset!(vortex_system)
_, _, _, _, m2l_list, direct_list, _ = fmm!(vortex_system; leaf_size_source=1, multipole_acceptance=0.4, expansion_order=20, lamb_helmholtz=true, error_tolerance=nothing)
v_fmm_vs = deepcopy(vortex_system.force)
diff_vs = v_fmm_vs - v_direct_vs
norms_vs = [sqrt(sum(diff_vs[i] .* diff_vs[i])) for i in 1:length(diff_vs)]
@show maximum(norms_vs) minimum(norms_vs) mean(norms_vs) std(norms_vs)
