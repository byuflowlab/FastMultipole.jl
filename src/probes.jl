function ProbeSystem(n_bodies, TF=Float64)
    position = zeros(SVector{3,TF}, n_bodies)
    scalar_potential = zeros(TF, n_bodies)
    gradient = zeros(SVector{3,TF}, n_bodies)
    hessian = zeros(SMatrix{3,3,TF,9}, n_bodies)
    return ProbeSystemStatic{TF}(position, scalar_potential, gradient, hessian)
end

function ProbeSystemArray(n_bodies, TF=Float64)
    position = zeros(TF, 3, n_bodies)
    scalar_potential = zeros(TF, n_bodies)
    gradient = zeros(TF, 3, n_bodies)
    hessian = zeros(TF, 3, 3, n_bodies)
    return ProbeSystemArray{TF}(position, scalar_potential, gradient, hessian)
end

function reset!(system::ProbeSystem{TF}) where TF
    system.scalar_potential .= zero(TF)
    for i in eachindex(system.gradient)
        system.gradient[i] = zero(SVector{3,TF})
    end
    for i in eachindex(system.hessian)
        system.hessian[i] = zero(SMatrix{3,3,TF,9})
    end
end

#------- FastMultipole compatibility functions -------#

Base.eltype(::ProbeSystem{TF}) where TF = TF

function FastMultipole.source_system_to_buffer!(buffer, i_buffer, system::ProbeSystem, i_body)
    throw("ProbeSystem cannot be used as a source system")
end

function FastMultipole.data_per_body(system::ProbeSystem)
    return 3
end

function FastMultipole.get_position(system::ProbeSystemStatic, i)
    return system.position[i]
end

function FastMultipole.get_position(system::ProbeSystemArray, i)
    return system.position[1,i], system.position[2,i], system.position[3,i]
end

function FastMultipole.strength_dims(system::ProbeSystem)
    return 0
end

FastMultipole.get_n_bodies(system::ProbeSystemStatic) = length(system.position)
FastMultipole.get_n_bodies(system::ProbeSystemArray) = size(system.position, 2)

function FastMultipole.body_to_multipole!(system::ProbeSystem, args...)
    return nothing
end

function FastMultipole.direct!(target_system, target_index, derivatives_switch, source_system::ProbeSystem, source_buffer, source_index)
    return nothing
end

function FastMultipole.buffer_to_target_system!(target_system::ProbeSystemStatic, i_target, ::FastMultipole.DerivativesSwitch{PS,GS,HS}, target_buffer, i_buffer) where {PS,GS,HS}
    if PS
        scalar_potential = FastMultipole.get_scalar_potential(target_buffer, i_buffer)
        target_system.scalar_potential[i_target] += scalar_potential
    end
    if GS
        gradient = FastMultipole.get_gradient(target_buffer, i_buffer)
        target_system.gradient[i_target] += gradient
    end
    if HS        
        hessian = FastMultipole.get_hessian(target_buffer, i_buffer)
        target_system.hessian[i_target] += hessian
    end
end

function FastMultipole.buffer_to_target_system!(target_system::ProbeSystemArray, i_target, ::FastMultipole.DerivativesSwitch{PS,GS,HS}, target_buffer, i_buffer) where {PS,GS,HS}
    if PS
        scalar_potential = FastMultipole.get_scalar_potential(target_buffer, i_buffer)
        target_system.scalar_potential[i_target] += scalar_potential
    end
    if GS
        gradient = FastMultipole.get_gradient(target_buffer, i_buffer)
        target_system.gradient[:, i_target] .+= gradient
    end
    if HS        
        hessian = FastMultipole.get_hessian(target_buffer, i_buffer)
        target_system.hessian[:, :, i_target] .+= hessian
    end
end
