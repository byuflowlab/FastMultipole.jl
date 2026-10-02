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

@testset "switchless accessors throw" begin
    buffer = zeros(34, 1)

    @test_throws ArgumentError FastMultipole.get_scalar_potential(buffer, 1)
    @test_throws ArgumentError FastMultipole.get_gradient(buffer, 1)
    @test_throws ArgumentError FastMultipole.get_hessian(buffer, 1)
    @test_throws ArgumentError FastMultipole.set_scalar_potential!(buffer, 1, 1.0)
    @test_throws ArgumentError FastMultipole.set_gradient!(buffer, 1, SVector(1.0, 2.0, 3.0))
    @test_throws ArgumentError FastMultipole.set_hessian!(buffer, 1, SMatrix{3,3,Float64,9}(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0))
end

@testset "fmm cache switch layout mismatch" begin
    system = generate_gravitational(123, 8)
    gradient_switches = FastMultipole.DerivativesSwitch(false, true, false, (system,))
    cache = FastMultipole.Cache((system,), (system,), gradient_switches)

    @test_throws ArgumentError FastMultipole.fmm!(system, cache; scalar_potential=false, gradient=true, hessian=true)
end

@testset "threaded fmm extra_farfield" begin
    if Threads.nthreads() == 1
        @test_skip "requires multiple Julia threads"
    else
        n_bodies = FastMultipole.MIN_BODIES ÷ 2 + 1
        target = generate_gravitational(123, n_bodies)
        source = generate_gravitational(456, n_bodies)

        @test begin
            FastMultipole.fmm!(
                target, source;
                scalar_potential=false,
                gradient=false,
                hessian=false,
                expansion_order=1,
                leaf_size_source=n_bodies,
                leaf_size_target=n_bodies,
                multipole_acceptance=0.0,
                upward_pass=false,
                horizontal_pass=false,
                downward_pass=false,
                nearfield=false,
                update_target_systems=false,
                extra_farfield=true,
                silence_warnings=true,
            )
            true
        end
    end
end
