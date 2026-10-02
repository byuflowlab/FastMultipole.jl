# A point-source system that carries two metadata rows and emits two extra
# outputs (metadata_extra_test.jl, third_derivative_test.jl).
struct MetadataSystem{TF}
    position::Matrix{TF}
    strength::Vector{TF}
    metadata::Matrix{TF}
    scalar::Vector{TF}
    extra::Matrix{TF}
end

Base.eltype(::MetadataSystem{TF}) where TF = TF
FastMultipole.get_n_bodies(system::MetadataSystem) = length(system.strength)
FastMultipole.get_position(system::MetadataSystem{TF}, i) where TF = SVector{3,TF}(system.position[1, i], system.position[2, i], system.position[3, i])
FastMultipole.has_vector_potential(system::MetadataSystem) = false
FastMultipole.data_per_body(system::MetadataSystem) = 5
FastMultipole.strength_dims(system::MetadataSystem) = 1

function FastMultipole.source_system_to_buffer!(buffer, i_buffer, system::MetadataSystem, i_body)
    buffer[1:3, i_buffer] .= FastMultipole.get_position(system, i_body)
    buffer[4, i_buffer] = 0
    buffer[5, i_buffer] = system.strength[i_body]
end

FastMultipole.metadata_per_body(system::MetadataSystem) = 2

function FastMultipole.metadata_to_buffer!(buffer, switch, i_buffer, system::MetadataSystem, i_body)
    buffer[FastMultipole.metadata_index(switch, 1), i_buffer] = system.metadata[1, i_body]
    buffer[FastMultipole.metadata_index(switch, 2), i_buffer] = system.metadata[2, i_body]
end

function FastMultipole.direct!(target_buffer, target_index, switch::FastMultipole.DerivativesSwitch{PS,GS,HS}, source_system::MetadataSystem, source_buffer, source_index) where {PS,GS,HS}
    for j_target in target_index
        metadata_1 = target_buffer[FastMultipole.metadata_index(switch, 1), j_target]
        metadata_2 = target_buffer[FastMultipole.metadata_index(switch, 2), j_target]
        scalar = zero(eltype(target_buffer))
        extra_1 = zero(eltype(target_buffer))
        extra_2 = zero(eltype(target_buffer))
        for i_source in source_index
            strength = source_buffer[5, i_source]
            scalar += strength
            extra_1 += metadata_1 * strength
            extra_2 += metadata_2 * strength
        end
        PS && FastMultipole.set_scalar_potential!(target_buffer, switch, j_target, scalar)
        FastMultipole.set_extra_output!(target_buffer, switch, j_target, 1, extra_1)
        FastMultipole.set_extra_output!(target_buffer, switch, j_target, 2, extra_2)
    end
end

function FastMultipole.buffer_to_target_system!(target_system::MetadataSystem, i_target, switch::FastMultipole.DerivativesSwitch{PS,GS,HS}, target_buffer, i_buffer) where {PS,GS,HS}
    PS && (target_system.scalar[i_target] += FastMultipole.get_scalar_potential(target_buffer, switch, i_buffer))
    target_system.extra[:, i_target] .+= FastMultipole.extra_output_view(target_buffer, switch, i_buffer)
end
