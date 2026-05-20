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

@testset "metadata and extra outputs" begin
    switch_old = FastMultipole.DerivativesSwitch(true, true, false)
    @test switch_old isa FastMultipole.DerivativesSwitch{true,true,false,0,0}

    switch = FastMultipole.DerivativesSwitch(true, false, false; extra_outputs=3, metadata=2)
    @test switch isa FastMultipole.DerivativesSwitch{true,false,false,3,2}
    @test FastMultipole.metadata_range(switch) == 4:5
    @test FastMultipole.scalar_potential_index(switch) == 6
    @test FastMultipole.gradient_range(switch) == 7:6
    @test FastMultipole.hessian_range(switch) == 7:6
    @test FastMultipole.standard_output_range(switch) == 6:6
    @test FastMultipole.extra_output_range(switch) == 7:9
    @test FastMultipole.output_range(switch) == 6:9

    position = [3.0 1.0 2.0 4.0; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0]
    strength = [1.0, 2.0, 3.0, 4.0]
    metadata = [10.0 20.0 30.0 40.0; 11.0 21.0 31.0 41.0]
    target = MetadataSystem(copy(position), copy(strength), copy(metadata), zeros(4), zeros(2, 4))
    switches = FastMultipole.DerivativesSwitch(false, false, false, (target,); extra_outputs=2, metadata=2)

    buffer = FastMultipole.allocate_buffers((target,), true, Float64, switches)[1]
    @test size(buffer, 1) == 7

    tree = FastMultipole.Tree((target,), true, switches; leaf_size=SVector{1}(1), shrink=false)
    sorted_metadata = tree.buffers[1][FastMultipole.metadata_range(switches[1]), :]
    for i_buffer in axes(sorted_metadata, 2)
        i_body = tree.sort_index_list[1][i_buffer]
        @test sorted_metadata[:, i_buffer] == metadata[:, i_body]
    end
    @test all(iszero, tree.buffers[1][FastMultipole.output_range(switches[1]), :])

    source = MetadataSystem(copy(position), copy(strength), copy(metadata), zeros(4), zeros(2, 4))
    FastMultipole.direct!(target, source; scalar_potential=true, gradient=false, hessian=false, extra_outputs=2, metadata=2, n_threads=1)
    total_strength = sum(strength)
    @test target.scalar == fill(total_strength, 4)
    @test target.extra[1, :] == metadata[1, :] .* total_strength
    @test target.extra[2, :] == metadata[2, :] .* total_strength
end

@testset "compact target buffer allocation" begin
    system = MetadataSystem(zeros(3, 2), ones(2), zeros(2, 2), zeros(2), zeros(0, 2))

    switches = FastMultipole.DerivativesSwitch(false, false, false, (system,); metadata=0, extra_outputs=0)
    @test size(FastMultipole.allocate_buffers((system,), true, Float64, switches)[1], 1) == 3

    switches = FastMultipole.DerivativesSwitch(false, true, false, (system,); metadata=0, extra_outputs=0)
    @test size(FastMultipole.allocate_buffers((system,), true, Float64, switches)[1], 1) == 6

    switches = FastMultipole.DerivativesSwitch(true, false, false, (system,); metadata=2, extra_outputs=0)
    @test size(FastMultipole.allocate_buffers((system,), true, Float64, switches)[1], 1) == 6

    switches = FastMultipole.DerivativesSwitch(false, false, true, (system,); metadata=2, extra_outputs=0)
    @test size(FastMultipole.allocate_buffers((system,), true, Float64, switches)[1], 1) == 14

    switches = FastMultipole.DerivativesSwitch(true, true, true, (system,); metadata=2, extra_outputs=4)
    @test size(FastMultipole.allocate_buffers((system,), true, Float64, switches)[1], 1) == 22
end

@testset "disabled compact target outputs throw" begin
    switch = FastMultipole.DerivativesSwitch(false, false, false; metadata=0, extra_outputs=0)
    buffer = zeros(3, 1)

    @test FastMultipole.scalar_potential_index(switch) == 0
    @test FastMultipole.gradient_range(switch) == 4:3
    @test FastMultipole.hessian_range(switch) == 4:3
    @test FastMultipole.standard_output_range(switch) == 4:3
    @test FastMultipole.extra_output_range(switch) == 4:3
    @test FastMultipole.output_range(switch) == 4:3

    @test_throws ArgumentError FastMultipole.get_scalar_potential(buffer, switch, 1)
    @test_throws ArgumentError FastMultipole.get_gradient(buffer, switch, 1)
    @test_throws ArgumentError FastMultipole.get_hessian(buffer, switch, 1)
    @test_throws ArgumentError FastMultipole.set_scalar_potential!(buffer, switch, 1, 1.0)
    @test_throws ArgumentError FastMultipole.set_gradient!(buffer, switch, 1, SVector(1.0, 2.0, 3.0))
    @test_throws ArgumentError FastMultipole.set_hessian!(buffer, switch, 1, SMatrix{3,3,Float64,9}(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0))
end

@testset "fmm cache switch layout mismatch" begin
    system = generate_gravitational(123, 8)
    gradient_switches = FastMultipole.DerivativesSwitch(false, true, false, (system,))
    cache = FastMultipole.Cache((system,), (system,), gradient_switches)

    @test_throws ArgumentError FastMultipole.fmm!(system, cache; scalar_potential=false, gradient=true, hessian=true)
end
