# module VPM

import FastMultipole as fmm
import FastMultipole.WriteVTK
using FastMultipole
using SpecialFunctions:erf

#------- classic vortex particle method -------#

const ONE_OVER_4PI = 1/4/pi
const sqrt2 = sqrt(2)
const i_POSITION_vortex = 1:3
const i_STRENGTH_vortex = 4:6
const i_POTENTIAL_SCALAR = 1:1
const i_POTENTIAL_VECTOR = 2:4
const i_HESSIAN_vortex = 5:13
const i_gradient_vortex = 1:3
const i_STRETCHING_vortex = 4:6

struct Vorton{TF}
    position::SVector{3,TF}
    strength::SVector{3,TF}
    sigma::TF
end

struct VortexParticles{TF}
    bodies::Vector{Vorton{TF}}
    potential::Matrix{TF}
    gradient_stretching::Matrix{TF}
end

function generate_vortex(seed, n_bodies; strength_scale=1/n_bodies, radius_factor=0.0)
    Random.seed!(seed)
    position = rand(3, n_bodies)
    radius = rand(n_bodies) .* radius_factor
    strength = 2 .* rand(3, n_bodies) .- 1.0
    strength .-= 0.5
    strength .*= 2
    strength .*= strength_scale
    return VortexParticles(position, strength, radius)
end

function reset!(system::VortexParticles{TF}) where TF
    system.potential .= zero(TF)
    system.gradient_stretching .= zero(TF)
end

abstract type IntegrationScheme end

struct Euler{TF} <: IntegrationScheme
    dt::TF
end

#------- overload compatibility functions -------#

function FastMultipole.source_system_to_buffer!(buffer, i_buffer, system::VortexParticles, i_body)
    buffer[1:3, i_buffer] .= system.bodies[i_body].position
    buffer[4, i_buffer] = system.bodies[i_body].sigma
    buffer[5:7, i_buffer] .= system.bodies[i_body].strength
end

function FastMultipole.data_per_body(system::VortexParticles)
    return 7
end

function FastMultipole.get_position(system::VortexParticles, i_body)
    return system.bodies[i_body].position
end

function FastMultipole.strength_dims(system::VortexParticles)
    return 3
end

function FastMultipole.get_n_bodies(system::VortexParticles)
    return length(system.bodies)
end

fmm.body_to_multipole!(system::VortexParticles, args...) = body_to_multipole!(Point{Vortex}, system, args...)

function fmm.direct!(target_system::Matrix{TF}, target_index, ::FastMultipole.DerivativesSwitch{S,V,VG}, source_system::VortexParticles, source_buffer, source_index) where {TF,S,V,VG}
    for j_source in source_index
        x_source = FastMultipole.get_position(source_buffer, j_source)
        Γx, Γy, Γz = FastMultipole.get_strength(source_buffer,  source_system, j_source)
        for i_target in target_index
            x_target = FastMultipole.get_position(target_system, i_target)
            dx, dy, dz = x_target - x_source
            r2 = dx * dx + dy * dy + dz * dz

            if r2 > 0
                # distance away
                r = sqrt(r2)
                rinv = 1 / r
                rinv2 = rinv * rinv

                # useful denominator
                denom = FastMultipole.ONE_OVER_4π * rinv * rinv2

                # induced vector field
                vx, vy, vz = zero(r2), zero(r2), zero(r2)
                if V
                    vx = (dz * Γy - dy * Γz) * denom
                    vy = (dx * Γz - dz * Γx) * denom
                    vz = (dy * Γx - dx * Γy) * denom
                    FastMultipole.set_gradient!(target_system, i_target, SVector{3,TF}(vx,vy,vz))
                end

                # vector gradient
                if VG
                    denom *= rinv2
                    vxx = -3 * dx * (Γy * dz - Γz * dy) * denom
                    vxy = (-3 * dx * (Γz * dx - Γx * dz) + Γz * r2) * denom
                    vxz = (-3 * dx * (Γx * dy - Γy * dx) - Γy * r2) * denom
                    vyx = (-3 * dy * (Γy * dz - Γz * dy) - Γz * r2) * denom
                    vyy = -3 * dy * (Γz * dx - Γx * dz) * denom
                    vyz = (-3 * dy * (Γx * dy - Γy * dx) + Γx * r2) * denom
                    vzx = (-3 * dz * (Γy * dz - Γz * dy) + Γy * r2) * denom
                    vzy = (-3 * dz * (Γz * dx - Γx * dz) - Γx * r2) * denom
                    vzz = -3 * dz * (Γx * dy - Γy * dx) * denom
                    FastMultipole.set_hessian!(target_system, i_target, SMatrix{3,3,TF,9}(vxx, vxy, vxz, vyx, vyy, vyz, vzx, vzy, vzz))
                end

            end
        end
    end
end

function FastMultipole.buffer_to_target_system!(target_system::VortexParticles, i_target, ::FastMultipole.DerivativesSwitch{PS,GS,HS}, target_buffer, i_buffer) where {PS,GS,HS}
    # retrieve fields
    TF = eltype(target_system)
    gradient = GS ? FastMultipole.get_gradient(target_buffer, i_buffer) : zero(SVector{3,TF})
    hessian = HS ? FastMultipole.get_hessian(target_buffer, i_buffer) : zero(SMatrix{3,3,TF,9})

    # update system
    if GS
        target_system.gradient_stretching[i_gradient_vortex, i_target] .= gradient
    end
    if HS
        target_system.potential[i_HESSIAN_vortex, i_target] .= reshape(hessian, 9)
    end
end

function FastMultipole.get_previous_influence(system::VortexParticles, i)
    phi_last = zero(eltype(system))
    gradient_last = SVector{3}(system.gradient_stretching[1,i], system.gradient_stretching[2,i], system.gradient_stretching[3,i])

    return phi_last, norm(gradient_last)
end

function FastMultipole.has_vector_potential(system::VortexParticles)
    return true
end

Base.eltype(::VortexParticles{TF}) where TF = TF

#------- additional functions -------#

function flatten_derivatives!(jacobian, hessian, derivatives_switch::DerivativesSwitch{PS,GS,HS}) where {PS,GS,HS}
    if GS
        # vector field
        jacobian[1,1] = -jacobian[1,1] + jacobian[2,4] - jacobian[3,3]
        jacobian[2,1] = -jacobian[2,1] + jacobian[3,2] - jacobian[1,4]
        jacobian[3,1] = -jacobian[3,1] + jacobian[1,3] - jacobian[2,2]
    end

    if HS
        # vector gradient
        hessian[1,1,1] = -hessian[1,1,1] + hessian[2,1,4] - hessian[3,1,3]
        hessian[2,1,1] = -hessian[2,1,1] + hessian[3,1,2] - hessian[1,1,4]
        hessian[3,1,1] = -hessian[3,1,1] + hessian[1,1,3] - hessian[2,1,2]
        hessian[1,2,1] = -hessian[1,2,1] + hessian[2,2,4] - hessian[3,2,3]
        hessian[2,2,1] = -hessian[2,2,1] + hessian[3,2,2] - hessian[1,2,4]
        hessian[3,2,1] = -hessian[3,2,1] + hessian[1,2,3] - hessian[2,2,2]
        hessian[1,3,1] = -hessian[1,3,1] + hessian[2,3,4] - hessian[3,3,3]
        hessian[2,3,1] = -hessian[2,3,1] + hessian[3,3,2] - hessian[1,3,4]
        hessian[3,3,1] = -hessian[3,3,1] + hessian[1,3,3] - hessian[2,3,2]
    end
end


function VortexParticles(position, strength, radius=zeros(size(position,2));
    N = size(position)[2],
    potential = zeros(i_HESSIAN_vortex[end],N),
    gradient_stretching = zeros(3+3,N),
)
    @assert size(position)[1] == 3
    @assert size(strength)[1] == 3
    bodies = [Vorton(SVector{3}(position[:,i]), SVector{3}(strength[:,i]), radius[i]) for i in 1:size(position)[2]]
    return VortexParticles(bodies, potential, gradient_stretching)
end

function VortexParticles(bodies;
    N = size(bodies)[2],
    potential = zeros(i_HESSIAN_vortex[end],N),
    gradient_stretching = zeros(3+3,N)
)
    bodies = [Vorton(SVector{3}(bodies[1:3,i]), SVector{3}(bodies[5:7,i]), bodies[4,i]) for i in 1:size(bodies)[2]]
    return VortexParticles(bodies, potential, gradient_stretching)
end

@inline function update_gradient_stretching!(system, i_body)
    # vorticity = @SVector [
    #     jacobian[2,3] - jacobian[3,2],
    #     jacobian[3,1] - jacobian[1,3],
    #     jacobian[1,2] - jacobian[2,1]
    # ]
    # stretching term (omega dot nabla)

    # total derivative of the strength due to vortex stretching
    duidxj = reshape(view(system.potential,i_HESSIAN_vortex,i_body),3,3)
    # @show duidxj system.bodies[i_body].strength
    fmm.mul!(view(system.gradient_stretching,i_STRETCHING_vortex,i_body), duidxj, system.bodies[i_body].strength)

    return nothing
end

function update_gradient_stretching!(vortex_particles::VortexParticles)
    for i_body in 1:fmm.get_n_bodies(vortex_particles)
        update_gradient_stretching!(vortex_particles, i_body)
    end
end

@inline function update_gradient_stretching_new!(system, i_body)
    # vorticity = @SVector [
    #     jacobian[2,3] - jacobian[3,2],
    #     jacobian[3,1] - jacobian[1,3],
    #     jacobian[1,2] - jacobian[2,1]
    # ]
    # stretching term (omega dot nabla)

    hessian = reshape(view(system.potential,i_HESSIAN_vortex,i_body),3,3)

    system.gradient_stretching[4] = dot(hessian[1,1:3], system.bodies[i_body].strength)
    system.gradient_stretching[5] = dot(hessian[2,1:3], system.bodies[i_body].strength)
    system.gradient_stretching[6] = dot(hessian[3,1:3], system.bodies[i_body].strength)

    return nothing
end

function update_gradient_stretching_new!(vortex_particles::VortexParticles)
    for i_body in 1:fmm.get_n_bodies(vortex_particles)
        update_gradient_stretching_new!(vortex_particles, i_body)
    end
end

"Assumes velocities and potential and derivatives are already calculated."
function (euler::Euler)(vortex_particles::VortexParticles, fmm_options, direct)
    # reset potential
    vortex_particles.potential .*= 0
    # reset vector field and stretching
    vortex_particles.gradient_stretching .*= 0

    # calculate influences
    if direct
        fmm.direct!(vortex_particles)
    else
        fmm.fmm!((vortex_particles,); fmm_options...)
    end

    # convect bodies
    bodies = vortex_particles.bodies
    potential = vortex_particles.potential
    gradient_stretching = vortex_particles.gradient_stretching
    update_gradient_stretching!(vortex_particles)
    for i_body in 1:fmm.get_n_bodies(vortex_particles)
        vortex_particles[i_body, Position()] = vortex_particles[i_body, Position()] + vortex_particles.gradient_stretching[i_gradient_vortex,i_body] * euler.dt
        vortex_particles[i_body, Strength()] = vortex_particles[i_body, Strength()] + vortex_particles.gradient_stretching[i_STRETCHING_vortex] * euler.dt
    end
end

function convect!(vortex_particles::VortexParticles, nsteps;
        # integration options
        integrate::IntegrationScheme=Euler(1.0),
        # fmm options
        fmm_p=4, fmm_ncrit=50, fmm_multipole_acceptance=0.5,
        direct::Bool=false,
        # save options
        save::Bool=true, filename::String="default", compress::Bool=false,
    )
    fmm_options = (; expansion_order=fmm_p, leaf_size=fmm_ncrit, multipole_acceptance=fmm_multipole_acceptance, lamb_helmholtz=true)
    save && save_vtk(filename, vortex_particles; compress)
    for istep in 1:nsteps
        integrate(vortex_particles, fmm_options, direct)
        save && save_vtk(filename, vortex_particles, istep; compress)
    end
    return nothing
end

function save_vtk(filename, vortex_particles::VortexParticles, nt=0; compress=false)

    n_bodies = length(vortex_particles.bodies)

    positions = reshape([FastMultipole.get_position(vortex_particles, j)[i] for i in 1:3, j in 1:n_bodies], 3, n_bodies, 1, 1)
    vectorstrength = reshape([vortex_particles.bodies[j].strength[i] for i in 1:3, j in 1:n_bodies], 3, n_bodies, 1, 1)
    gradient = reshape(vortex_particles.gradient_stretching[i_gradient_vortex,:],3,n_bodies,1,1)
    stretching = reshape(vortex_particles.gradient_stretching[i_STRETCHING_vortex,:],3,n_bodies,1,1)

    WriteVTK.vtk_grid(filename*"."*string(nt)*".vts", positions; compress) do vtk
        vtk["vector strength"] = vectorstrength
        vtk["gradient"] = gradient
        vtk["stretching"] = stretching
    end

end
