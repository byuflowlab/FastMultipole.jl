import FastMultipole
using FastMultipole
using FastMultipole.WriteVTK
import Base: getindex, setindex!
using FastMultipole.StaticArrays
using FastMultipole.LinearAlgebra
using Random
const i_POSITION = 1:3
const i_RADIUS = 4:4
const i_STRENGTH = 5:5
const i_POTENTIAL = 1:4
const i_gradient = 5:7
const i_hessian = 8:16

#------- gravitational kernel and mass elements -------#

struct Body{TF}
    position::SVector{3,TF}
    radius::TF
    strength::TF
end

struct Gravitational{TF}
    bodies::Vector{Body{TF}}
    potential::Matrix{TF}
end

function Gravitational(bodies::Matrix)
    nbodies = size(bodies)[2]
    bodies2 = [Body(SVector{3}(bodies[1:3,i]), bodies[4,i], bodies[5,i]) for i in 1:nbodies]
    potential = zeros(eltype(bodies), 16, nbodies)
    return Gravitational(bodies2, potential)
end

function generate_gravitational(seed, n_bodies; radius_factor=0.1, strength_scale=1/n_bodies, bodies_fun=(x)->x)
    Random.seed!(seed)
    bodies = rand(8,n_bodies)
    bodies[4,:] ./= (n_bodies^(1/3)*2)
    bodies[4,:] .*= radius_factor
    bodies[5,:] .*= strength_scale

    bodies_fun(bodies)

    system = Gravitational(bodies)
end

#------- FastMultipole compatibility functions -------#

Base.eltype(::Gravitational{TF}) where TF = TF

function FastMultipole.source_system_to_buffer!(buffer, i_buffer, system::Gravitational, i_body)
    x, y, z = system.bodies[i_body].position
    buffer[1, i_buffer] = x
    buffer[2, i_buffer] = y
    buffer[3, i_buffer] = z
    buffer[4, i_buffer] = system.bodies[i_body].radius
    buffer[5, i_buffer] = system.bodies[i_body].strength
end

function FastMultipole.data_per_body(system::Gravitational)
    return 5
end

function reset!(system::Gravitational{TF}) where TF
    system.potential .= zero(TF)
end

function FastMultipole.get_position(g::Gravitational, i)
    return g.bodies[i].position
end

function FastMultipole.strength_dims(system::Gravitational)
    return 1
end

FastMultipole.get_n_bodies(g::Gravitational) = length(g.bodies)

FastMultipole.body_to_multipole!(system::Gravitational, args...) = FastMultipole.body_to_multipole!(Point{Source}, system, args...)

function FastMultipole.has_vector_potential(system::Gravitational)
    return false
end

function FastMultipole.get_previous_influence(system::Gravitational, i)
    phi_last = system.potential[1,i]
    gradient_last = SVector{3}(system.potential[5,i], system.potential[6,i], system.potential[7,i])

    return phi_last, norm(gradient_last)
end

function FastMultipole.direct!(target_system, target_index, derivatives_switch::FastMultipole.DerivativesSwitch{PS,GS,HS}, source_system::Gravitational, source_buffer, source_index) where {PS,GS,HS}
    @inbounds for j_target in target_index
        target_x, target_y, target_z = FastMultipole.get_position(target_system, j_target)
        d_potential = zero(eltype(target_system))
        d_gradient = zero(SVector{3,eltype(target_system)})
        d_hessian = zero(SMatrix{3,3,eltype(target_system),9})
        @inbounds for i_source in source_index
            source_x, source_y, source_z = FastMultipole.get_position(source_buffer, i_source)
            source_strength = FastMultipole.get_strength(source_buffer, source_system, i_source)[1]
            dx = target_x - source_x
            dy = target_y - source_y
            dz = target_z - source_z
            r2 = dx*dx + dy*dy + dz*dz
            if r2 > 0
                r = sqrt(r2)
                if PS
                    d_potential += source_strength / r * FastMultipole.ONE_OVER_4π
                end
                if GS
                    d_gradient += SVector{3}(dx,dy,dz) * d_potential / r2
                end
            end
        end
        if PS
            FastMultipole.set_scalar_potential!(target_system, j_target, d_potential)
        end
        if GS
            FastMultipole.set_gradient!(target_system, j_target, d_gradient)
        end
    end
end

function FastMultipole.buffer_to_target_system!(target_system::Gravitational, i_target, ::FastMultipole.DerivativesSwitch{PS,GS,HS}, target_buffer, i_buffer) where {PS,GS,HS}
    # get values
    TF = eltype(target_buffer)
    scalar_potential = PS ? FastMultipole.get_scalar_potential(target_buffer, i_buffer) : zero(TF)
    gradient = GS ? FastMultipole.get_gradient(target_buffer, i_buffer) : zero(SVector{3,TF})
    hessian = HS ? FastMultipole.get_hessian(target_buffer, i_buffer) : zero(SMatrix{3,3,TF,9})

    # update system
    target_system.potential[i_POTENTIAL[1], i_target] = scalar_potential
    target_system.potential[i_gradient, i_target] .= gradient
    for (jj,j) in enumerate(i_hessian)
        target_system.potential[j, i_target] = hessian[jj]
    end
end
