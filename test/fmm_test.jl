function psi(target_x, source_x, source_gamma)
    dx = target_x - source_x
    dx_norm = sqrt(dx' * dx)
    return source_gamma ./ dx_norm * ONE_OVER_4PI
end

function dpsidx(target_x, source_x, source_gamma)
    dx = target_x - source_x
    dx_norm = sqrt(dx' * dx)
    x, y, z = dx
    jacobian = [
        -x*source_gamma[1] -x*source_gamma[2] -x*source_gamma[3];
        -y*source_gamma[1] -y*source_gamma[2] -y*source_gamma[3];
        -z*source_gamma[1] -z*source_gamma[2] -z*source_gamma[3];
    ] ./ dx_norm^3 * ONE_OVER_4PI
    return jacobian
end

function d2psidx2(target_x, source_x, source_gamma)
    dx = target_x - source_x
    dx_norm = sqrt(dx' * dx)
    x, y, z = dx
    hessian = zeros(3,3,3)
    d2dr2 = [
        2x^2-y^2-z^2 3x*y 3x*z;
        3x*y 2y^2-x^2-z^2 3y*z;
        3x*z 3y*z 2z^2-x^2-y^2
    ] / dx_norm^5
    hessian[:,:,1] = d2dr2 * source_gamma[1] * ONE_OVER_4PI
    hessian[:,:,2] = d2dr2 * source_gamma[2] * ONE_OVER_4PI
    hessian[:,:,3] = d2dr2 * source_gamma[3] * ONE_OVER_4PI
    return hessian
end

function u(target_x, source_x, source_gamma)
    dx = target_x  - source_x
    dx_norm = sqrt(dx' * dx)
    return 1/4/pi/dx_norm^3 * [
        -dx[2]*source_gamma[3] + dx[3]*source_gamma[2],
        -dx[3]*source_gamma[1] + dx[1]*source_gamma[3],
        -dx[1]*source_gamma[2] + dx[2]*source_gamma[1]
    ]
end

function duidxj_fd_fun(target_x, source_x, source_gamma; h=1e-8)
    duidx = (u(target_x+[h,0,0], source_x, source_gamma) - u(target_x,source_x,source_gamma))/h
    duidy = (u(target_x+[0,h,0], source_x, source_gamma) - u(target_x,source_x,source_gamma))/h
    duidz = (u(target_x+[0,0,h], source_x, source_gamma) - u(target_x,source_x,source_gamma))/h
    duidxj_res = hcat(duidx, duidy, duidz) .* 4 * pi
    return duidxj_res
end

function stretching(target_x, source_x, target_gamma, source_gamma)
    dx = target_x - source_x
    x, y, z = dx
    xy = x*y
    yz = y*z
    xz = x*z
    gx, gy, gz = source_gamma
    dx_norm = sqrt(dx' * dx)
    duidxj = [
        (3xy*gz-3xz*gy) ((2y^2-x^2-z^2)*gz-3yz*gy) (3yz*gz-(2z^2-x^2-y^2)*gy);
        (3xz*gx-(2x^2-y^2-z^2)*gz) (3yz*gx-3xy*gz) ((2z^2-x^2-y^2)*gx-3xz*gz);
        ((2x^2-y^2-z^2)*gy-3xy*gx) (3xy*gy-(2y^2-x^2-z^2)*gx) (3xz*gy-3yz*gx)
    ]/dx_norm^5
    stretch = 1/4/pi*duidxj*target_gamma
    return stretch
end

@testset "fmm: scalar potential" begin

xs = [
    1.2 1.1 0.8;
    0.8 0.9 0.2;
    0.1 0.2 0.9;
    0.1 0.3 0.2;
    0.2 0.25 0.4
]

ms = [
    0.8,
    1.1,
    2.2,
    0.5,
    1.9
]

bodies = zeros(8,length(ms))
for i in 1:length(ms)
    bodies[1:3,i] .= xs[i,1:3]
    bodies[5,i] = ms[i]
end
masses = Gravitational(bodies)

expansion_order = 24
multipole_acceptance = 0.5

fmm!(masses; expansion_order, multipole_acceptance, scalar_potential=true)
u_fmm = masses.potential[1,:]

masses.potential .= 0.0
direct!(masses; scalar_potential=true)
u_direct = masses.potential[1,:]

for i in eachindex(u_fmm)
    @test isapprox(u_fmm[i], u_direct[i]; atol=1e-12)
end

end

@testset "fmm: vector potential: 2D" begin

# 2-D vortex ring
xs = [
    0 0
    0.5 -0.5
    0 0
]

Gammas = [
    0 0;
    0 0;
    1 -1.0
]

vortexparticles = VortexParticles(xs, Gammas)
source_buffer = FastMultipole.system_to_buffer(vortexparticles)
target_buffer = FastMultipole.target_to_buffer(vortexparticles, true)

# using direct method
direct!((vortexparticles,); hessian=true)
update_gradient_stretching!(vortexparticles)

@test isapprox(vortexparticles.gradient_stretching[1,1], 1/4/pi; atol=1e-10)
@test isapprox(vortexparticles.gradient_stretching[1,2], 1/4/pi; atol=1e-10)
@test isapprox(vortexparticles.gradient_stretching[2,1], 0; atol=1e-10)
@test isapprox(vortexparticles.gradient_stretching[2,2], 0; atol=1e-10)
@test isapprox(vortexparticles.gradient_stretching[3,1], 0; atol=1e-10)
@test isapprox(vortexparticles.gradient_stretching[3,2], 0; atol=1e-10)

vorton_potential_check = zeros(4,2)
vorton_potential_check[i_POTENTIAL_VECTOR,:] = deepcopy(vortexparticles.potential[i_POTENTIAL_VECTOR,1:2])

vorton_gradient_check = deepcopy(vortexparticles.gradient_stretching[i_gradient_vortex,:])

# reset vortons
vortexparticles.potential .*= 0
vortexparticles.gradient_stretching .*= 0

# manually build tree for testing
expansion_order = 9
leaf_size = SVector{1}(1)
x_branch_1 = SVector{3}([0.0,0,0])
bounding_box = SVector{3}(0.0,0,0)

branch_1 = FastMultipole.Branch(SVector{1}(2), SVector{1}([1:2]), 2, 2:3, 0, -1, x_branch_1, 1/8, bounding_box)
x_branch_2 = FastMultipole.SVector{3}(xs[:,1] .+ [0.01, 0.02, -0.03])
branch_2 = FastMultipole.Branch(SVector{1}(1), SVector{1}([1:1]), 0, 3:2, 1, 1, x_branch_2, 1/8, bounding_box)
multipole_coefficients_2 = FastMultipole.initialize_expansion(expansion_order)
x_branch_3 = FastMultipole.SVector{3}(xs[:,2] .+ [0.02, -0.04, 0.01])
branch_3 = FastMultipole.Branch(SVector{1}(1), SVector{1}([2:2]), 0, 3:2, 1, 2, x_branch_3, 1/8, bounding_box)
multipole_coefficients_3 = FastMultipole.initialize_expansion(expansion_order)

# using FMM
dummy_index = (zeros(Int64,FastMultipole.get_n_bodies(vortexparticles)),)
dummy_leaf_index = collect(1:3)
expansions = FastMultipole.initialize_expansions(expansion_order, 3)
tree = FastMultipole.Tree([branch_1, branch_2, branch_3], expansions, [1:1,2:3], dummy_leaf_index, dummy_index, dummy_index, (source_buffer,), FastMultipole.allocate_small_buffers((vortexparticles,), eltype(vortexparticles)), expansion_order, leaf_size)#, dummy_cost_parameter)
multipole_coefficients_2 = FastMultipole.initialize_expansion(expansion_order)
multipole_coefficients_3 = FastMultipole.initialize_expansion(expansion_order)
local_coefficients_2 = view(expansions, :, :, :, 2)
local_coefficients_3 = view(expansions, :, :, :, 3)
harmonics = initialize_harmonics(expansion_order)

FastMultipole.body_to_multipole!(vortexparticles, multipole_coefficients_2, source_buffer, branch_2.center, branch_2.bodies_index[1], harmonics, expansion_order)
FastMultipole.body_to_multipole!(vortexparticles, multipole_coefficients_3, source_buffer, branch_3.center, branch_3.bodies_index[1], harmonics, expansion_order)

m2l_harmonics = initialize_harmonics(expansion_order)
# L = zeros(eltype(tree.branches[1].local_expansion), 2, 4)
lamb_helmholtz = Val(true)

# preallocate containers
Hs_π2 = [1.0]
FastMultipole.update_Hs_π2!(Hs_π2, expansion_order)
Ts = zeros(FastMultipole.length_Ts(expansion_order))
eimϕs = zeros(2, expansion_order+2)
weights_tmp_1 = initialize_expansion(expansion_order, eltype(Ts))
weights_tmp_2 = initialize_expansion(expansion_order, eltype(Ts))
weights_tmp_3 = initialize_expansion(expansion_order, eltype(Ts))

# normalization
ζs_mag = zeros(FastMultipole.length_ζs(expansion_order))
FastMultipole.update_ζs_mag!(ζs_mag, 0, expansion_order)
ηs_mag = zeros(FastMultipole.length_ηs(expansion_order))
FastMultipole.update_ηs_mag!(ηs_mag, 0, expansion_order)

FastMultipole.multipole_to_local!(local_coefficients_2, tree.branches[2], multipole_coefficients_3, tree.branches[3], weights_tmp_1, weights_tmp_2, weights_tmp_3, Ts, eimϕs, ζs_mag, ηs_mag, Hs_π2, FastMultipole.M̃, FastMultipole.L̃, expansion_order, lamb_helmholtz)
FastMultipole.multipole_to_local!(local_coefficients_3, tree.branches[3], multipole_coefficients_2, tree.branches[2], weights_tmp_1, weights_tmp_2, weights_tmp_3, Ts, eimϕs, ζs_mag, ηs_mag, Hs_π2, FastMultipole.M̃, FastMultipole.L̃, expansion_order, lamb_helmholtz)

gradient_n_m = FastMultipole.initialize_gradient_n_m(expansion_order)

FastMultipole.evaluate_local!(target_buffer, 1, tree, 2, harmonics, gradient_n_m, expansion_order, lamb_helmholtz, (FastMultipole.DerivativesSwitch(),))
FastMultipole.evaluate_local!(target_buffer, 1, tree, 3, harmonics, gradient_n_m, expansion_order, lamb_helmholtz, (FastMultipole.DerivativesSwitch(),))

# update system
FastMultipole.buffer_to_target!(vortexparticles, target_buffer, DerivativesSwitch(false, true, true), 1:FastMultipole.get_n_bodies(vortexparticles))

update_gradient_stretching!(vortexparticles)

for i in 1:2
    for ind in 1:4
        @test isapprox(vortexparticles.potential[ind,i], vorton_potential_check[ind,i]; atol=1e-12)
    end
    for dim in 1:3
        @test isapprox(vortexparticles.gradient_stretching[dim,i], vorton_gradient_check[dim,i]; atol=1e-12)
    end
end

end

@testset "fmm: vector potential: 2 particles in 3D" begin

bodies = [
    0.4 0.1
    0.1 -0.5
    -0.3 0.2
    1/8 1/8
    0.3 -0.4
    -0.1 -0.2
    0.08 0.5
]

vortex_particles = VortexParticles(bodies)
source_buffer = FastMultipole.system_to_buffer(vortex_particles)
target_buffer = FastMultipole.target_to_buffer(vortex_particles, true)

#####
##### obtain psi, u, and stretching analytically
#####

psis = zeros(3,2)
psis[:,1] = psi(bodies[1:3,1], bodies[1:3,2], bodies[5:7,2])
psis[:,2] = psi(bodies[1:3,2], bodies[1:3,1], bodies[5:7,1])
hessians = zeros(3,3,3,2)
hessians[:,:,:,1] = d2psidx2(bodies[1:3,1], bodies[1:3,2], bodies[5:7,2])
hessians[:,:,:,2] = d2psidx2(bodies[1:3,2], bodies[1:3,1], bodies[5:7,1])
us = zeros(3,2)
us[:,1] = u(bodies[1:3,1], bodies[1:3,2], bodies[5:7,2])
us[:,2] = u(bodies[1:3,2], bodies[1:3,1], bodies[5:7,1])
ss = zeros(3,2)
ss[:,1] = stretching(bodies[1:3,1], bodies[1:3,2], bodies[5:7,1], bodies[5:7,2])
ss[:,2] = stretching(bodies[1:3,2], bodies[1:3,1], bodies[5:7,2], bodies[5:7,1])

#--- use direct method ---#

FastMultipole.direct!((vortex_particles,); gradient=true, hessian=true)
potential_direct = deepcopy(vortex_particles.potential)
update_gradient_stretching!(vortex_particles)

us_direct = vortex_particles.gradient_stretching[1:3,:]
for i in 1:length(us)
    @test isapprox(us_direct, us;atol=1e-12)
end
ss_direct = vortex_particles.gradient_stretching[4:6,:]
for i in 1:length(ss)
    @test isapprox(ss_direct[i], ss[i];atol=1e-12)
end

#--- fmm, manual tree ---#

# reset potential
vortex_particles.potential .*= 0
vortex_particles.gradient_stretching .*= 0

# set up fmm call
expansion_order = 20
leaf_size_source = SVector{1}(1)

# manually build tree
x_branch_1 = SVector{3}((bodies[1:3,1] + bodies[1:3,2])/2)
bounding_box = SVector{3}(0.0,0,0)

branch_1 = FastMultipole.Branch(SVector{1}(2), SVector{1}([1:2]), 2, 2:3, 0, -1, x_branch_1, 1/8, bounding_box)
x_branch_2 = SVector{3}(bodies[1:3,1])# .+ [0.01, 0.02, -0.03])
multipole_coefficients_2 = FastMultipole.initialize_expansion(expansion_order)
branch_2 = FastMultipole.Branch(SVector{1}(1), SVector{1}([1:1]), 0, 3:2, 1, 1, x_branch_2, 1/8, bounding_box)
x_branch_3 = SVector{3}(bodies[1:3,2])# .+ [0.02, -0.04, 0.01])
multipole_coefficients_3 = FastMultipole.initialize_expansion(expansion_order)
branch_3 = FastMultipole.Branch(SVector{1}(1), SVector{1}([2:2]), 0, 3:2, 1, 2, x_branch_3, 1/8, bounding_box)

dummy_index = (zeros(Int,length(vortex_particles.bodies)),)
dummy_leaf_index = collect(1:3)
# dummy_cost_parameter = FastMultipole.dummy_direct_cost_estimate((vortex_particles,), leaf_size)
expansions = FastMultipole.initialize_expansions(expansion_order, 3)
tree = FastMultipole.Tree([branch_1, branch_2, branch_3], expansions, [1:1,2:3], dummy_leaf_index, dummy_index, dummy_index, (source_buffer,), FastMultipole.allocate_small_buffers((vortex_particles,), eltype(vortex_particles)), expansion_order, leaf_size_source)#, dummy_cost_parameter)
local_coefficients_2 = view(expansions, :, :, :, 2)
local_coefficients_3 = view(expansions, :, :, :, 3)
harmonics = FastMultipole.initialize_harmonics(expansion_order)

# manually compute multipole coefficients
FastMultipole.body_to_multipole!(vortex_particles, multipole_coefficients_2, source_buffer, branch_2.center, branch_2.bodies_index[1], harmonics, expansion_order)
FastMultipole.body_to_multipole!(vortex_particles, multipole_coefficients_3, source_buffer, branch_3.center, branch_3.bodies_index[1], harmonics, expansion_order)

# preallocate containers
Hs_π2 = [1.0]
FastMultipole.update_Hs_π2!(Hs_π2, expansion_order)
Ts = zeros(FastMultipole.length_Ts(expansion_order))
eimϕs = zeros(2, expansion_order+2)
weights_tmp_1 = initialize_expansion(expansion_order, eltype(Ts))
weights_tmp_2 = initialize_expansion(expansion_order, eltype(Ts))
weights_tmp_3 = initialize_expansion(expansion_order, eltype(Ts))

# normalization
ζs_mag = zeros(FastMultipole.length_ζs(expansion_order))
FastMultipole.update_ζs_mag!(ζs_mag, 0, expansion_order)
ηs_mag = zeros(FastMultipole.length_ηs(expansion_order))
FastMultipole.update_ηs_mag!(ηs_mag, 0, expansion_order)
lamb_helmholtz = Val(true)

# translate multipoles
FastMultipole.multipole_to_local!(local_coefficients_2, branch_2, multipole_coefficients_3, branch_3, weights_tmp_1, weights_tmp_2, weights_tmp_3, Ts, eimϕs, ζs_mag, ηs_mag, Hs_π2, FastMultipole.M̃, FastMultipole.L̃, expansion_order, lamb_helmholtz)
FastMultipole.multipole_to_local!(local_coefficients_3, branch_3, multipole_coefficients_2, branch_2, weights_tmp_1, weights_tmp_2, weights_tmp_3, Ts, eimϕs, ζs_mag, ηs_mag, Hs_π2, FastMultipole.M̃, FastMultipole.L̃, expansion_order, lamb_helmholtz)

# evaluate multipoles
gradient_n_m = FastMultipole.initialize_gradient_n_m(expansion_order)
harmonics = FastMultipole.initialize_harmonics(expansion_order)
derivatives_switches = DerivativesSwitch(true, true, true, (vortex_particles,))

FastMultipole.evaluate_local!(target_buffer, 1, tree, 2, harmonics, gradient_n_m, expansion_order, lamb_helmholtz, (FastMultipole.DerivativesSwitch(),))
FastMultipole.evaluate_local!(target_buffer, 1, tree, 3, harmonics, gradient_n_m, expansion_order, lamb_helmholtz, (FastMultipole.DerivativesSwitch(),))

FastMultipole.buffer_to_target!(vortex_particles, target_buffer, DerivativesSwitch(false, true, true), 1:FastMultipole.get_n_bodies(vortex_particles))

update_gradient_stretching!(vortex_particles)
# hessians_fmm.= deepcopy(reshape(vortex_particles.potential[i_POTENTIAL_HESSIAN[10:end],:],3,3,3,2))
# for i in 1:length(hessians)
#     @test isapprox(hessians_fmm.i], hessians[i]; atol=1e-8)
# end
us_fmm = vortex_particles.gradient_stretching[1:3,:]
for i in eachindex(us_fmm)
    @test isapprox(us_fmm[i], us[i];atol=1e-12)
end
ss_fmm = vortex_particles.gradient_stretching[4:6,:]
for i in 1:length(ss)
    @test isapprox(ss_fmm[i], ss[i];atol=1e-12)
end

#--- fmm, automated tree ---#

# reset potential
vortex_particles.potential .*= 0
vortex_particles.gradient_stretching .*= 0

# run fmm
multipole_acceptance = 0.7
tree, m2l_list, direct_list, derivatives_switches = fmm!((vortex_particles,); expansion_order, leaf_size_source, multipole_acceptance, shrink_recenter=true, scalar_potential=false, hessian=true)
update_gradient_stretching!(vortex_particles)

# test vector
us_fmm_2 = vortex_particles.gradient_stretching[1:3,:]
for i in eachindex(us_fmm_2)
    @test isapprox(us_fmm_2[i], us[i];atol=1e-12)
end

# test stretching
ss_fmm_2 = vortex_particles.gradient_stretching[4:6,:]
for i in 1:length(ss)
    @test isapprox(ss_fmm_2[i], ss[i];atol=1e-12)
end

end

@testset "fmm: vector potential: 3 particles in 3D" begin

bodies = [
    0.4 0.1 -0.1
    0.1 -0.5 0.25
    -0.3 0.2 0.1
    1/8 1/8 1/8
    0.3 -0.4 0.2
    -0.1 -0.2 0.5
    0.08 0.5 1.1
]

vortex_particles = VortexParticles(bodies)
source_buffer = FastMultipole.system_to_buffer(vortex_particles)
target_buffer = FastMultipole.target_to_buffer(vortex_particles, true)

psis = zeros(3,3)
psis[:,1] = psi(bodies[1:3,1], bodies[1:3,2], bodies[5:7,2]) + psi(bodies[1:3,1], bodies[1:3,3], bodies[5:7,3])
psis[:,2] = psi(bodies[1:3,2], bodies[1:3,1], bodies[5:7,1]) + psi(bodies[1:3,2], bodies[1:3,3], bodies[5:7,3])
psis[:,3] = psi(bodies[1:3,3], bodies[1:3,1], bodies[5:7,1]) + psi(bodies[1:3,3], bodies[1:3,2], bodies[5:7,2])
jacobians = zeros(3,3,3)
jacobians[:,:,1] = dpsidx(bodies[1:3,1], bodies[1:3,2], bodies[5:7,2]) + dpsidx(bodies[1:3,1], bodies[1:3,3], bodies[5:7,3])
jacobians[:,:,2] = dpsidx(bodies[1:3,2], bodies[1:3,1], bodies[5:7,1]) + dpsidx(bodies[1:3,2], bodies[1:3,3], bodies[5:7,3])
jacobians[:,:,3] = dpsidx(bodies[1:3,3], bodies[1:3,1], bodies[5:7,1]) + dpsidx(bodies[1:3,3], bodies[1:3,2], bodies[5:7,2])
hessians = zeros(3,3,3,3)
hessians[:,:,:,1] = d2psidx2(bodies[1:3,1], bodies[1:3,2], bodies[5:7,2]) + d2psidx2(bodies[1:3,1], bodies[1:3,3], bodies[5:7,3])
hessians[:,:,:,2] = d2psidx2(bodies[1:3,2], bodies[1:3,1], bodies[5:7,1]) + d2psidx2(bodies[1:3,2], bodies[1:3,3], bodies[5:7,3])
hessians[:,:,:,3] = d2psidx2(bodies[1:3,3], bodies[1:3,1], bodies[5:7,1]) + d2psidx2(bodies[1:3,3], bodies[1:3,2], bodies[5:7,2])
us = zeros(3,3)
us[:,1] = u(bodies[1:3,1], bodies[1:3,2], bodies[5:7,2]) + u(bodies[1:3,1], bodies[1:3,3], bodies[5:7,3])
us[:,2] = u(bodies[1:3,2], bodies[1:3,1], bodies[5:7,1]) + u(bodies[1:3,2], bodies[1:3,3], bodies[5:7,3])
us[:,3] = u(bodies[1:3,3], bodies[1:3,1], bodies[5:7,1]) + u(bodies[1:3,3], bodies[1:3,2], bodies[5:7,2])
ss = zeros(3,3)
ss[:,1] = stretching(bodies[1:3,1], bodies[1:3,2], bodies[5:7,1], bodies[5:7,2]) + stretching(bodies[1:3,1], bodies[1:3,3], bodies[5:7,1], bodies[5:7,3])
ss[:,2] = stretching(bodies[1:3,2], bodies[1:3,1], bodies[5:7,2], bodies[5:7,1]) + stretching(bodies[1:3,2], bodies[1:3,3], bodies[5:7,2], bodies[5:7,3])
ss[:,3] = stretching(bodies[1:3,3], bodies[1:3,1], bodies[5:7,3], bodies[5:7,1]) + stretching(bodies[1:3,3], bodies[1:3,2], bodies[5:7,3], bodies[5:7,2])

#--- use direct method ---#

FastMultipole.direct!((vortex_particles,); hessian=true)
update_gradient_stretching!(vortex_particles)

#psis_direct = deepcopy(vortex_particles.potential[2:4,:])
#for i in 1:length(psis_direct)
#    @test isapprox(psis_direct[i], psis[i]; atol=1e-10)
#end
us_direct = deepcopy(vortex_particles.gradient_stretching[1:3,:])
for i in 1:length(us)
    @test isapprox(us_direct[i], us[i];atol=1e-12)
end
ss_direct = deepcopy(vortex_particles.gradient_stretching[4:6,:])
for i in 1:length(ss)
    @test isapprox(ss_direct[i], ss[i];atol=1e-12)
end

#--- use fmm ---#

# reset potential
vortex_particles.potential .*= 0
vortex_particles.gradient_stretching .*= 0

expansion_order = 20
leaf_size = SVector{1}(1)

source_tree = FastMultipole.Tree((vortex_particles,), false; expansion_order, leaf_size, shrink_recenter=false)
target_tree = FastMultipole.Tree((vortex_particles,), true; expansion_order, leaf_size, shrink_recenter=false)

m2l_list, direct_list, derivatives_switches = FastMultipole.fmm!((vortex_particles,), target_tree, (vortex_particles,), source_tree; expansion_order, multipole_acceptance=0.8, scalar_potential=false, hessian=true)

update_gradient_stretching!(vortex_particles)

#psis_fmm = deepcopy(vortex_particles.potential[2:4,:])
#for i in 1:length(psis_fmm)
#    @test isapprox(psis_fmm[i], psis[i]; rtol=1e-12)
#end
us_fmm = deepcopy(vortex_particles.gradient_stretching[1:3,:])
for i in 1:length(us)
    @test isapprox(us_fmm[i], us[i]; atol=1e-12)
end
ss_fmm = deepcopy(vortex_particles.gradient_stretching[4:6,:])
for i in 1:length(ss)
    @test isapprox(ss_fmm[i], ss[i]; atol=1e-12)
end

end

@testset "single/dual tree, single/multi branch" begin

expansion_order, leaf_size_source, multipole_acceptance = 12, SVector{1}(1), 0.99
n_bodies = 4

shrink_recenter = true
seed = 2
system = generate_gravitational(seed, n_bodies; radius_factor=0.1)
FastMultipole.direct!(system; scalar_potential=true)
validation_potential = system.potential[1,:]

system.potential .= zero(eltype(system.potential))
optargs, _ = FastMultipole.fmm!(system; expansion_order, leaf_size_source, multipole_acceptance, shrink_recenter, scalar_potential=true)
potential3 = system.potential[1,:]
@test isapprox(maximum(abs.(potential3 - validation_potential)), 0.0; atol=1e-10)

system.potential .= zero(eltype(system.potential))
FastMultipole.fmm!((system,); expansion_order, leaf_size_source, multipole_acceptance, nearfield=true, farfield=true, shrink_recenter, scalar_potential=true)
potential4 = system.potential[1,:]
@test isapprox(maximum(abs.(potential4 - validation_potential)), 0.0; atol=1e-10)

system.potential .= zero(eltype(system.potential))
FastMultipole.fmm!(system, system; expansion_order, leaf_size_source, multipole_acceptance, nearfield=true, farfield=true, shrink_recenter, scalar_potential=true)
potential5 = system.potential[1,:]
@test isapprox(maximum(abs.(potential5 - validation_potential)), 0.0; atol=1e-10)

system.potential .= zero(eltype(system.potential))
FastMultipole.fmm!((system,), system; expansion_order, leaf_size_source, multipole_acceptance, shrink_recenter, scalar_potential=true)
potential6 = system.potential[1,:]
@test isapprox(maximum(abs.(potential6 - validation_potential)), 0.0; atol=1e-10)

system.potential .= zero(eltype(system.potential))
FastMultipole.fmm!((system,), (system,); expansion_order, leaf_size_source, multipole_acceptance, shrink_recenter, scalar_potential=true)
potential7 = system.potential[1,:]
@test isapprox(maximum(abs.(potential7 - validation_potential)), 0.0; atol=1e-10)

end

