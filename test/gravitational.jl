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
const i_third_derivative = 17:34

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
    potential = zeros(eltype(bodies), 34, nbodies)
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

FastMultipole.body_to_multipole!(system::Gravitational, args...) = FastMultipole.body_to_multipole!(Point{Source}, system, args...; scale_strength=-1.0)

function FastMultipole.has_vector_potential(system::Gravitational)
    return false
end

FastMultipole.supports_third_derivative(target_system, source_system::Gravitational) = true

FastMultipole.metadata_per_body(system::Gravitational) = 2
FastMultipole.previous_potential_metadata_index(system::Gravitational) = 1
FastMultipole.previous_gradient_metadata_index(system::Gravitational) = 2

function FastMultipole.metadata_to_buffer!(buffer, switch, i_buffer, system::Gravitational, i_body)
    previous_potential = system.potential[1, i_body]
    previous_gradient = norm(SVector{3}(system.potential[5, i_body], system.potential[6, i_body], system.potential[7, i_body]))
    buffer[FastMultipole.metadata_index(switch, 1), i_buffer] = previous_potential
    buffer[FastMultipole.metadata_index(switch, 2), i_buffer] = previous_gradient
end

function FastMultipole.direct!(target_system, target_index, derivatives_switch::FastMultipole.DerivativesSwitch{PS,GS,HS,NO,NM,TS}, source_system::Gravitational, source_buffer, source_index) where {PS,GS,HS,NO,NM,TS}
    @inbounds for j_target in target_index
        target_x, target_y, target_z = FastMultipole.get_position(target_system, j_target)
        dϕ = zero(eltype(target_system))
        d∇ϕ = zero(SVector{3,eltype(target_system)})
        dH = zero(SMatrix{3,3,eltype(target_system),9})
        dT = zero(MVector{18,eltype(target_system)})
        @inbounds for i_source in source_index
            source_x, source_y, source_z = FastMultipole.get_position(source_buffer, i_source)
            source_strength = FastMultipole.get_strength(source_buffer, source_system, i_source)[1]
            dx = target_x - source_x
            dy = target_y - source_y
            dz = target_z - source_z
            r2 = dx*dx + dy*dy + dz*dz
            if r2 > 0
                r = sqrt(r2)
                tmp = source_strength / r * FastMultipole.ONE_OVER_4π
                if PS
                    dϕ += tmp
                end 
                if GS
                    d∇ϕ -= SVector{3}(dx,dy,dz) * tmp / r2
                end
                if HS || TS
                    q5 = source_strength * FastMultipole.ONE_OVER_4π / (r2 * r2 * r)
                    x = SVector(dx, dy, dz)
                    if HS
                        dH += SMatrix{3,3}(ntuple(Val(9)) do n
                            ii = (n - 1) % 3 + 1
                            jj = (n - 1) ÷ 3 + 1
                            q5 * (3x[ii] * x[jj] - (ii == jj ? r2 : zero(r2)))
                        end)
                    end
                    if TS
                        q7 = q5 / r2
                        slot = 0
                        for ii in 1:3, (jj, kk) in ((1,1), (1,2), (1,3), (2,2), (2,3), (3,3))
                            slot += 1
                            delta_terms = (ii == jj ? x[kk] : zero(r2)) +
                                (ii == kk ? x[jj] : zero(r2)) +
                                (jj == kk ? x[ii] : zero(r2))
                            dT[slot] += q7 * (3r2 * delta_terms - 15x[ii] * x[jj] * x[kk])
                        end
                    end
                end
            end
        end
        PS && FastMultipole.set_scalar_potential!(target_system, derivatives_switch, j_target, dϕ)
        GS && FastMultipole.set_gradient!(target_system, derivatives_switch, j_target, d∇ϕ)
        HS && FastMultipole.set_hessian!(target_system, derivatives_switch, j_target, dH)
        TS && FastMultipole.set_third_derivative!(target_system, derivatives_switch, j_target, SVector(dT))
    end
end

function FastMultipole.buffer_to_target_system!(target_system::Gravitational, i_target, derivatives_switch::FastMultipole.DerivativesSwitch{PS,GS,HS,NO,NM,TS}, target_buffer, i_buffer) where {PS,GS,HS,NO,NM,TS}
    # get values
    TF = eltype(target_buffer)
    scalar_potential = PS ? FastMultipole.get_scalar_potential(target_buffer, derivatives_switch, i_buffer) : zero(TF)
    gradient = GS ? FastMultipole.get_gradient(target_buffer, derivatives_switch, i_buffer) : zero(SVector{3,TF})
    hessian = HS ? FastMultipole.get_hessian(target_buffer, derivatives_switch, i_buffer) : zero(SMatrix{3,3,TF,9})
    third_derivative = TS ? FastMultipole.get_third_derivative(target_buffer, derivatives_switch, i_buffer) : nothing

    # update system
    target_system.potential[i_POTENTIAL[1], i_target] = scalar_potential
    target_system.potential[i_gradient, i_target] .= gradient
    for (jj,j) in enumerate(i_hessian)
        target_system.potential[j, i_target] = hessian[jj]
    end
    TS && (target_system.potential[i_third_derivative, i_target] .= FastMultipole.packed_data(third_derivative))
end
