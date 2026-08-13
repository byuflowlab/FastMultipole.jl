using FastMultipole
using FastMultipole.StaticArrays
using Random
using Test

struct CUDARadixCPUScalarSystem{TF}
    positions::Matrix{TF}
    radii::Vector{TF}
    strengths::Vector{TF}
    potential::Vector{TF}
    gradient::Matrix{TF}
end

CUDARadixCPUScalarSystem(positions::Matrix{TF}, radii::Vector{TF}, strengths::Vector{TF}) where TF =
    CUDARadixCPUScalarSystem(positions, radii, strengths, zeros(TF, length(strengths)), zeros(TF, 3, length(strengths)))

struct CUDARadixDeviceScalarSystem{TF,A,B,C}
    host::CUDARadixCPUScalarSystem{TF}
    positions::A
    radii::B
    strengths::C
end

Base.eltype(::CUDARadixCPUScalarSystem{TF}) where TF = TF
FastMultipole.get_n_bodies(system::CUDARadixCPUScalarSystem) = size(system.positions, 2)
FastMultipole.data_per_body(::CUDARadixCPUScalarSystem) = 5
FastMultipole.strength_dims(::CUDARadixCPUScalarSystem) = 1
FastMultipole.get_position(system::CUDARadixCPUScalarSystem{TF}, i) where TF =
    SVector{3,TF}(system.positions[1, i], system.positions[2, i], system.positions[3, i])

function FastMultipole.source_system_to_buffer!(buffer, i_buffer, system::CUDARadixCPUScalarSystem, i_body)
    buffer[1:3, i_buffer] .= system.positions[:, i_body]
    buffer[4, i_buffer] = system.radii[i_body]
    buffer[5, i_buffer] = system.strengths[i_body]
end

Base.eltype(::CUDARadixDeviceScalarSystem{TF}) where TF = TF
FastMultipole.get_n_bodies(system::CUDARadixDeviceScalarSystem) =
    FastMultipole.get_n_bodies(system.host)
FastMultipole.data_per_body(system::CUDARadixDeviceScalarSystem) =
    FastMultipole.data_per_body(system.host)
FastMultipole.strength_dims(system::CUDARadixDeviceScalarSystem) =
    FastMultipole.strength_dims(system.host)
FastMultipole.get_position(system::CUDARadixDeviceScalarSystem, i) =
    FastMultipole.get_position(system.host, i)
FastMultipole.source_system_to_buffer!(buffer, i_buffer,
        system::CUDARadixDeviceScalarSystem, i_body) =
    FastMultipole.source_system_to_buffer!(buffer, i_buffer, system.host, i_body)
FastMultipole.residency(::CUDARadixDeviceScalarSystem) = DeviceResident()

function FastMultipole.source_to_buffer!(device_buffer,
        system::CUDARadixDeviceScalarSystem, sort_index)
    device_buffer[1:3, :] .= system.positions[:, sort_index]
    device_buffer[4, :] .= system.radii[sort_index]
    device_buffer[5, :] .= system.strengths[sort_index]
    return device_buffer
end

function FastMultipole.buffer_to_target_system!(target_system::CUDARadixCPUScalarSystem,
        i_target, derivatives_switch, target_buffer, i_buffer)
    target_system.potential[i_target] +=
        FastMultipole.get_scalar_potential(target_buffer, derivatives_switch, i_buffer)
    target_system.gradient[:, i_target] .+=
        FastMultipole.get_gradient(target_buffer, derivatives_switch, i_buffer)
end

function _cuda_radix_expected_body(grid, system::CUDARadixCPUScalarSystem{TF}) where TF
    source_buffer = FastMultipole.source_to_buffer(system)
    body = zeros(TF, size(source_buffer, 1), length(grid.perm))
    for sorted_i in eachindex(grid.perm)
        global_i = grid.perm[sorted_i]
        ibody = grid.body_index[global_i]
        # all data_per_body rows are carried, including radius row 4 (task 032)
        body[:, sorted_i] .= source_buffer[:, ibody]
    end
    return body
end

function _expected_cuda_radix_nodes(grid::RadixGrid{TF}) where TF
    level_keys = [sort(unique(key >> (3 * (grid.ell - level)) for key in grid.cell_keys))
                  for level in 0:grid.ell]
    offsets = zeros(Int, grid.ell + 2)
    for level in 0:grid.ell
        offsets[level + 2] = offsets[level + 1] + length(level_keys[level + 1])
    end
    n_nodes = offsets[end]
    levels = Vector{Int}(undef, n_nodes)
    keys = Vector{UInt64}(undef, n_nodes)
    coords = Matrix{Int}(undef, 3, n_nodes)
    centers = Matrix{TF}(undef, 3, n_nodes)
    index_by_node = Dict{Tuple{Int,UInt64},Int}()
    for level in 0:grid.ell
        Δ = (2 * grid.h0) / (1 << level)
        for (local_i, key) in pairs(level_keys[level + 1])
            node = offsets[level + 1] + local_i
            coord = FastMultipole.morton_decode(key, level)
            levels[node] = level
            keys[node] = key
            coords[:, node] .= coord
            centers[:, node] .= grid.x_min + Δ * (SVector{3,TF}(coord) .+ SVector{3,TF}(0.5, 0.5, 0.5))
            index_by_node[(level, key)] = node
        end
    end
    parent_index = zeros(Int, n_nodes)
    child_ranges = zeros(Int, 2, n_nodes)
    for node in 1:n_nodes
        level = levels[node]
        if level > 0
            parent_index[node] = index_by_node[(level - 1, keys[node] >> 3)]
        end
    end
    for node in 1:n_nodes
        children = findall(==(node), parent_index)
        if !isempty(children)
            child_ranges[1, node] = first(children)
            child_ranges[2, node] = length(children)
        end
    end
    leaf_to_node = offsets[grid.ell + 1] .+ collect(1:length(grid.cell_keys))
    return (; levels, keys, coords, centers, parent_index, child_ranges, leaf_to_node)
end

function _expected_root_multipole(grid, source_bodies, P)
    TF = eltype(source_bodies)
    root_center = grid.node_centers[:, 1]
    phi = zeros(TF, 2 * ((P + 1) * (P + 2) ÷ 2))
    for n in 0:P, m in 0:n
        acc_re = zero(TF)
        acc_im = zero(TF)
        sgn = isodd(n + m) ? -one(TF) : one(TF)
        for k in axes(source_bodies, 2)
            dx = source_bodies[1, k] - root_center[1]
            dy = source_bodies[2, k] - root_center[2]
            dz = source_bodies[3, k] - root_center[3]
            q = source_bodies[5, k]
            rre, rim = FastMultipole._resident_regular_harmonic_coeff(dx, dy, dz, n, m)
            scale = sgn * q
            acc_re += rre * scale
            acc_im -= rim * scale
        end
        row = FastMultipole.flat_basis_index(n, m, 1)
        phi[row] = acc_re
        phi[row + 1] = acc_im
    end
    return phi
end

function _expected_direct_output(source_bodies)
    TF = eltype(source_bodies)
    output = zeros(TF, 4, size(source_bodies, 2))
    c = inv(TF(4) * TF(pi))
    for i in axes(source_bodies, 2)
        xi = source_bodies[1, i]
        yi = source_bodies[2, i]
        zi = source_bodies[3, i]
        for j in axes(source_bodies, 2)
            i == j && continue
            dx = xi - source_bodies[1, j]
            dy = yi - source_bodies[2, j]
            dz = zi - source_bodies[3, j]
            r2 = dx * dx + dy * dy + dz * dz
            r2 == zero(TF) && continue
            invr = inv(sqrt(r2))
            q = source_bodies[5, j] * c
            output[1, i] += q * invr
            invr3 = invr * invr * invr
            output[2, i] -= q * dx * invr3
            output[3, i] -= q * dy * invr3
            output[4, i] -= q * dz * invr3
        end
    end
    return output
end

function _assert_resident_l2b_eval_matches_production(::Type{TF}, P, lh::Val{LH}) where {TF,LH}
    basis_info = OperatorBasisInfo(CompressedComplexBasis(), P, lh)
    locals = FlatCoefficientBuffer(TF, basis_info, 1)
    @inbounds for i in axes(locals.phi, 1)
        locals.phi[i, 1] = TF(sin(0.17 * i) + cos(0.11 * i))
    end
    if LH
        @inbounds for i in axes(locals.chi, 1)
            locals.chi[i, 1] = TF(cos(0.13 * i) - sin(0.07 * i))
        end
    end
    dx = SVector{3,TF}(TF(0.07), TF(-0.04), TF(0.05))
    scalar_potential, gx, gy, gz = FastMultipole._resident_local_eval_flat(
        locals.phi, locals.chi, 1, dx[1], dx[2], dx[3],
        basis_info.orders.P_phi, basis_info.orders.P_active, lh,
    )
    legacy = FastMultipole.initialize_expansion(basis_info.orders.P_active, TF)
    FastMultipole._pack_flat_column!(legacy, locals, 1, basis_info.orders.P_phi, basis_info.orders.P_active, lh)
    harmonics = FastMultipole.initialize_harmonics(basis_info.orders.P_active, TF)
    gradient_n_m = FastMultipole.initialize_gradient_n_m(basis_info.orders.P_active, TF)
    ref_u, ref_g, _ = FastMultipole.evaluate_local(
        dx, harmonics, gradient_n_m, legacy, P, lh, DerivativesSwitch(true, true, false),
    )
    rtol = TF === Float32 ? 5f-5 : (LH ? 2e-7 : 1e-12)
    atol = TF === Float32 ? 5f-6 : (LH ? 5e-8 : 1e-12)
    @test scalar_potential ≈ ref_u rtol=rtol atol=atol
    @test SVector(gx, gy, gz) ≈ ref_g rtol=rtol atol=atol
    return nothing
end

function _farfield_test_system(::Type{TF}=Float64) where TF
    rng = MersenneTwister(20260701)
    n = 400
    positions = TF.(rand(rng, 3, n))
    radii = fill(TF(0.002), n)
    strengths = TF.(randn(rng, n))
    return CUDARadixCPUScalarSystem(positions, radii, strengths)
end

function _max_channel_errors(output, reference)
    potential_error = maximum(abs.(output[1, :] .- reference[1, :]))
    gradient_error = maximum(abs.(output[2:4, :] .- reference[2:4, :]))
    return potential_error, gradient_error
end

function _flatten_test_routes(list)
    levels = Int[]
    targets = Int[]
    sources = Int[]
    for batch in list.m2l_batches
        for j in eachindex(batch.targets)
            push!(levels, batch.level)
            push!(targets, batch.targets[j])
            push!(sources, batch.sources[j])
        end
    end
    return levels, targets, sources
end

_cuda_required_tests() = get(ENV, "FASTMULTIPOLE_REQUIRE_CUDA_TESTS", "0") == "1"

@testset "CUDA radix lifecycle gate (task 022)" begin
    require_cuda_tests = _cuda_required_tests()

    @test FastMultipole.residency(CUDARadixCPUScalarSystem(zeros(3, 0), Float64[], Float64[])) isa HostResident
    # the deprecated device-buffer hooks were removed in task 032 stage 3
    @test !isdefined(FastMultipole, :source_system_to_device_buffer!)
    @test !isdefined(FastMultipole, :target_system_from_device_buffer!)
    @test FastMultipole.cuda_radix_available() == false
    @test occursin("not loaded", FastMultipole.cuda_radix_status())

    counters = CUDARadixTransferCounters()
    @test counters.body_uploads == 0
    @test counters.influence_downloads == 0
    @test counters.expansion_host_copies == 0

    opts64 = CUDARadixLifecycleOptions()
    @test isconcretetype(typeof(opts64))
    @test fieldtype(typeof(opts64), :operator) === MaterializedYRotationM2L
    @test fieldtype(typeof(opts64), :m2m_strategy) === SharedRotationM2M
    @test fieldtype(typeof(opts64), :m2l_strategy) === SharedRotationM2L
    @test CUDARadixLifecycleOptions{Float64}(
        Float64, MaterializedYRotationM2L(), SharedRotationM2M(),
        SharedRotationM2L()) == opts64
    @test CUDARadixLifecycleOptions{Float64}() == opts64
    @test opts64.precision === Float64
    @test opts64.operator isa MaterializedYRotationM2L
    @test opts64.m2m_strategy isa FastMultipole.SharedRotationM2M
    @test opts64.m2l_strategy isa FastMultipole.SharedRotationM2L
    opts32 = CUDARadixLifecycleOptions(; precision=Float32)
    @test opts32.precision === Float32
    @test_throws ArgumentError CUDARadixLifecycleOptions(; precision=Float16)
    @test_throws ArgumentError CUDARadixLifecycleOptions(; operator=:invalid)
    @test_throws ArgumentError CUDARadixLifecycleOptions(; m2l_strategy=FastMultipole.SharedRotationM2M())

    _assert_resident_l2b_eval_matches_production(Float64, 4, Val(false))
    _assert_resident_l2b_eval_matches_production(Float64, 4, Val(true))
    _assert_resident_l2b_eval_matches_production(Float32, 3, Val(false))

    positions = Float64[
        0.03 0.29 0.56 0.91 0.12
        0.08 0.61 0.43 0.19 0.88
        0.14 0.37 0.79 0.52 0.24
    ]
    radii = fill(0.01, size(positions, 2))
    strengths = Float64.(1:size(positions, 2))
    cpu_system = CUDARadixCPUScalarSystem(positions, radii, strengths)
    grid = RadixGrid(cpu_system, 3)
    host_grid = host_resident_radix_grid(grid)
    expected_nodes = _expected_cuda_radix_nodes(grid)
    @test host_grid isa DeviceRadixGrid{Float64}
    @test host_grid.x_min ≈ grid.x_min
    @test host_grid.h0 ≈ grid.h0
    @test host_grid.ell == grid.ell
    @test host_grid.perm == grid.perm
    @test host_grid.invperm == grid.invperm
    @test host_grid.cell_keys == grid.cell_keys
    @test host_grid.cell_ranges == grid.cell_ranges
    @test host_grid.node_levels == expected_nodes.levels
    @test host_grid.node_keys == expected_nodes.keys
    @test host_grid.node_coords == expected_nodes.coords
    @test host_grid.node_centers ≈ expected_nodes.centers
    @test host_grid.parent_index == expected_nodes.parent_index
    @test host_grid.child_ranges == expected_nodes.child_ranges
    @test host_grid.leaf_to_node == expected_nodes.leaf_to_node

    list = build_radix_interaction_list(LazyMaterializedBatches(1), ParentNeighborM2L(), grid)
    host_state = host_radix_state(cpu_system, grid, list, 2)
    expected_body = _cuda_radix_expected_body(grid, cpu_system)
    @test host_state.source_bodies ≈ expected_body
    @test host_state.counters.body_uploads == 0
    @test host_state.counters.expansion_host_copies == 0
    @test run_host_radix_lifecycle!(host_state) === host_state
    @test host_state.counters.expansion_host_copies == 0
    @test host_state.multipoles.phi[:, 1] ≈ _expected_root_multipole(host_grid, expected_body, 2)
    @test size(host_state.output) == (4, length(grid.perm))

    host_state_lh = host_radix_state(cpu_system, grid, list, 1, Val(true))
    @test size(host_state_lh.multipoles.chi, 1) == OperatorBasisInfo(CompressedComplexBasis(), 1, Val(true)).basis_dof_chi
    @test run_host_radix_lifecycle!(host_state_lh) === host_state_lh
    @test host_state_lh.counters.expansion_host_copies == 0

    host_state32 = host_radix_state(cpu_system, grid, list, 1; options=CUDARadixLifecycleOptions(; precision=Float32))
    @test eltype(host_state32.source_bodies) === Float32
    @test run_host_radix_lifecycle!(host_state32) === host_state32
    @test host_state32.counters.expansion_host_copies == 0

    m2l_oracle_state = host_radix_state(cpu_system, grid, list, 2)
    m2l_resident_state = host_radix_state(cpu_system, grid, list, 2)
    FastMultipole._launch_host_b2m!(m2l_oracle_state)
    FastMultipole._launch_host_m2m!(m2l_oracle_state)
    FastMultipole._launch_host_m2l_flat_oracle!(m2l_oracle_state)
    FastMultipole._launch_host_b2m!(m2l_resident_state)
    FastMultipole._launch_host_m2m!(m2l_resident_state)
    FastMultipole._launch_host_m2l!(m2l_resident_state)
    @test m2l_resident_state.locals.phi ≈ m2l_oracle_state.locals.phi rtol=1e-9 atol=1e-10
    @test m2l_resident_state.counters.expansion_host_copies == 0

    l2l_oracle_state = host_radix_state(cpu_system, grid, list, 2)
    l2l_resident_state = host_radix_state(cpu_system, grid, list, 2)
    FastMultipole._launch_host_b2m!(l2l_oracle_state)
    FastMultipole._launch_host_m2m!(l2l_oracle_state)
    FastMultipole._launch_host_m2l!(l2l_oracle_state)
    FastMultipole._launch_host_l2l_flat_oracle!(l2l_oracle_state)
    FastMultipole._launch_host_b2m!(l2l_resident_state)
    FastMultipole._launch_host_m2m!(l2l_resident_state)
    FastMultipole._launch_host_m2l!(l2l_resident_state)
    FastMultipole._launch_host_l2l!(l2l_resident_state)
    @test l2l_resident_state.locals.phi ≈ l2l_oracle_state.locals.phi rtol=1e-9 atol=1e-10
    @test l2l_resident_state.counters.expansion_host_copies == 0

    l2l_oracle_state32_lh = host_radix_state(
        cpu_system, grid, list, 1, Val(true);
        options=CUDARadixLifecycleOptions(; precision=Float32),
    )
    l2l_resident_state32_lh = host_radix_state(
        cpu_system, grid, list, 1, Val(true);
        options=CUDARadixLifecycleOptions(; precision=Float32),
    )
    FastMultipole._launch_host_b2m!(l2l_oracle_state32_lh)
    FastMultipole._launch_host_m2m!(l2l_oracle_state32_lh)
    FastMultipole._launch_host_m2l_flat_oracle!(l2l_oracle_state32_lh)
    FastMultipole._launch_host_l2l_flat_oracle!(l2l_oracle_state32_lh)
    FastMultipole._launch_host_b2m!(l2l_resident_state32_lh)
    FastMultipole._launch_host_m2m!(l2l_resident_state32_lh)
    FastMultipole._launch_host_m2l!(l2l_resident_state32_lh)
    FastMultipole._launch_host_l2l!(l2l_resident_state32_lh)
    @test l2l_resident_state32_lh.locals.phi ≈ l2l_oracle_state32_lh.locals.phi rtol=5f-3 atol=5f-4
    @test l2l_resident_state32_lh.locals.chi ≈ l2l_oracle_state32_lh.locals.chi rtol=5f-3 atol=5f-4
    @test l2l_resident_state32_lh.counters.expansion_host_copies == 0

    direct_grid = RadixGrid(cpu_system, 0; h0_fallback=1.0)
    direct_list = build_radix_interaction_list(LazyMaterializedBatches(1), ParentNeighborM2L(), direct_grid)
    @test isempty(direct_list.m2l_batches)
    @test length(direct_list.direct_pairs) == 1

    for lh in (Val(false), Val(true))
        direct_state = host_radix_state(cpu_system, direct_grid, direct_list, 2, lh)
        @test run_host_radix_lifecycle!(direct_state) === direct_state
        @test direct_state.output ≈ _expected_direct_output(direct_state.source_bodies) rtol=1e-12 atol=1e-12
        @test direct_state.counters.expansion_host_copies == 0
    end

    direct_state32 = host_radix_state(
        cpu_system, direct_grid, direct_list, 2, Val(false);
        options=CUDARadixLifecycleOptions(; precision=Float32),
    )
    @test run_host_radix_lifecycle!(direct_state32) === direct_state32
    @test direct_state32.output ≈ _expected_direct_output(direct_state32.source_bodies) rtol=1f-5 atol=1f-6
    @test direct_state32.counters.expansion_host_copies == 0

    far_system = _farfield_test_system(Float64)
    far_grid = RadixGrid(far_system, 3)
    far_list = build_radix_interaction_list(LazyMaterializedBatches(1), ParentNeighborM2L(), far_grid)
    @test !isempty(far_list.m2l_batches)
    far_reference_state = host_radix_state(far_system, far_grid, far_list, 4)
    far_reference = _expected_direct_output(far_reference_state.source_bodies)
    potential_errors = Float64[]
    gradient_errors = Float64[]
    for P in (4, 8, 12)
        far_state = host_radix_state(far_system, far_grid, far_list, P)
        @test run_host_radix_lifecycle!(far_state) === far_state
        @test far_state.counters.expansion_host_copies == 0
        potential_error, gradient_error = _max_channel_errors(far_state.output, far_reference)
        push!(potential_errors, potential_error)
        push!(gradient_errors, gradient_error)
    end
    @test potential_errors[3] < potential_errors[2] < potential_errors[1]
    @test gradient_errors[3] < gradient_errors[2] < gradient_errors[1]

    far_lh_state = host_radix_state(far_system, far_grid, far_list, 4, Val(true))
    @test run_host_radix_lifecycle!(far_lh_state) === far_lh_state
    @test all(isfinite, far_lh_state.output)
    @test far_lh_state.counters.expansion_host_copies == 0

    far32_system = _farfield_test_system(Float32)
    far32_grid = RadixGrid(far32_system, 3)
    far32_list = build_radix_interaction_list(LazyMaterializedBatches(1), ParentNeighborM2L(), far32_grid)
    far32_state = host_radix_state(
        far32_system, far32_grid, far32_list, 4, Val(false);
        options=CUDARadixLifecycleOptions(; precision=Float32),
    )
    @test run_host_radix_lifecycle!(far32_state) === far32_state
    @test all(isfinite, far32_state.output)
    @test far32_state.counters.expansion_host_copies == 0

    saved_retry_test = get(ENV, "FASTMULTIPOLE_CUDA_RETRY_TEST", nothing)
    saved_loaded = FastMultipole._CUDA_RADIX_LIFECYCLE_LOADED[]
    saved_error = FastMultipole._CUDA_RADIX_LIFECYCLE_LOAD_ERROR[]
    try
        @eval FastMultipole function _cuda_radix_preflight_error()
            retry_test = get(ENV, "FASTMULTIPOLE_CUDA_RETRY_TEST", "")
            retry_test == "fail-first" && return "retry test first failure"
            retry_test == "fail-second" && return "retry test second failure"
            get(ENV, "FASTMULTIPOLE_FORCE_CUDA_LOAD", "0") == "1" && return nothing
            Sys.isapple() && return "CUDA is not available on macOS in this runtime"
            if Sys.islinux() && !ispath("/dev/nvidiactl") && !ispath("/proc/driver/nvidia/version")
                return "no NVIDIA device nodes detected; set FASTMULTIPOLE_FORCE_CUDA_LOAD=1 to force CUDA.jl loading"
            end
            return nothing
        end
        FastMultipole._CUDA_RADIX_LIFECYCLE_LOADED[] = false
        FastMultipole._CUDA_RADIX_LIFECYCLE_LOAD_ERROR[] = :stale_error
        ENV["FASTMULTIPOLE_CUDA_RETRY_TEST"] = "fail-first"
        @test !FastMultipole.load_cuda_radix_lifecycle!()
        @test !FastMultipole._CUDA_RADIX_LIFECYCLE_LOADED[]
        @test occursin("retry test first failure", FastMultipole.cuda_radix_status())

        ENV["FASTMULTIPOLE_CUDA_RETRY_TEST"] = "fail-second"
        @test !FastMultipole.load_cuda_radix_lifecycle!()
        @test !FastMultipole._CUDA_RADIX_LIFECYCLE_LOADED[]
        @test occursin("retry test second failure", FastMultipole.cuda_radix_status())
    finally
        @eval FastMultipole function _cuda_radix_preflight_error()
            get(ENV, "FASTMULTIPOLE_FORCE_CUDA_LOAD", "0") == "1" && return nothing
            Sys.isapple() && return "CUDA is not available on macOS in this runtime"
            if Sys.islinux() && !ispath("/dev/nvidiactl") && !ispath("/proc/driver/nvidia/version")
                return "no NVIDIA device nodes detected; set FASTMULTIPOLE_FORCE_CUDA_LOAD=1 to force CUDA.jl loading"
            end
            return nothing
        end
        if saved_retry_test === nothing
            delete!(ENV, "FASTMULTIPOLE_CUDA_RETRY_TEST")
        else
            ENV["FASTMULTIPOLE_CUDA_RETRY_TEST"] = saved_retry_test
        end
        FastMultipole._CUDA_RADIX_LIFECYCLE_LOADED[] = saved_loaded
        FastMultipole._CUDA_RADIX_LIFECYCLE_LOAD_ERROR[] = saved_error
    end

    loaded = FastMultipole.load_cuda_radix_lifecycle!()
    if !loaded
        @test !FastMultipole.cuda_radix_available()
        @test !isempty(FastMultipole.cuda_radix_status())
        @test_throws FastMultipole.CUDARadixUnavailable FastMultipole.cuda_radix_grid(zeros(Float64, 3, 0), 1)
        @test_throws FastMultipole.CUDARadixUnavailable FastMultipole.run_cuda_radix_lifecycle!(
            DeviceResidentRadixState{Float64,CompressedComplexBasis,false}(
                nothing, nothing, nothing, nothing,
                FlatCoefficientBuffer(Float64, OperatorBasisInfo(0, Val(false)), 0),
                FlatCoefficientBuffer(Float64, OperatorBasisInfo(0, Val(false)), 0),
                nothing, nothing, nothing, nothing, nothing, nothing, nothing,
                counters, opts64,
            ),
        )
        if require_cuda_tests
            error(
                "FASTMULTIPOLE_REQUIRE_CUDA_TESTS=1 but CUDA radix lifecycle did not load: " *
                FastMultipole.cuda_radix_status(),
            )
        end
    else
        @test FastMultipole.cuda_radix_available()
        @test occursin("CUDA functional", FastMultipole.cuda_radix_status())

        @eval using CUDA
        @eval function FastMultipole.buffer_to_target!(target_system::CUDARadixDeviceScalarSystem,
                device_output_buffer::CUDA.AnyCuArray, derivatives_switch, sort_index)
            target_system.host.potential .= Array(device_output_buffer[FastMultipole.scalar_potential_index(derivatives_switch), :])
            target_system.host.gradient .= Array(device_output_buffer[FastMultipole.gradient_range(derivatives_switch), :])
            return target_system
        end

        positions = Float64[
            0.10 0.90 0.20 0.70
            0.10 0.90 0.20 0.70
            0.10 0.90 0.20 0.70
        ]
        radii = fill(0.01, size(positions, 2))
        strengths = [1.0, -2.0, 3.0, -4.0]
        cpu_system = CUDARadixCPUScalarSystem(positions, radii, strengths)
        device_system = CUDARadixDeviceScalarSystem(
            cpu_system, CUDA.CuArray(positions), CUDA.CuArray(radii), CUDA.CuArray(strengths),
        )
        grid = RadixGrid(cpu_system, 1)
        host_grid = FastMultipole.host_resident_radix_grid(grid)
        device_grid = cuda_radix_grid(device_system, 1)
        @test device_grid isa DeviceRadixGrid{Float64}
        @test !(device_grid isa DeviceRadixGrid{Float64,Any})
        for field in (
                :perm, :invperm, :cell_keys, :cell_ranges, :body_system, :body_index,
                :cell_centers, :node_levels, :node_keys, :node_coords, :node_centers,
                :parent_index, :child_ranges, :leaf_to_node)
            @test getfield(device_grid, field) isa CUDA.CuArray
        end
        @test device_grid.x_min ≈ grid.x_min
        @test device_grid.h0 ≈ grid.h0
        @test device_grid.ell == grid.ell
        @test Array(device_grid.perm) == grid.perm
        @test Array(device_grid.invperm) == grid.invperm
        @test Array(device_grid.cell_keys) == grid.cell_keys
        @test Array(device_grid.cell_ranges) == grid.cell_ranges
        @test Array(device_grid.body_system) == grid.body_system
        @test Array(device_grid.body_index) == grid.body_index
        @test Array(device_grid.cell_centers) ≈ hcat([FastMultipole.radix_cell_center(grid, i) for i in 1:length(grid)]...)
        expected_nodes = _expected_cuda_radix_nodes(grid)
        @test Array(device_grid.node_levels) == expected_nodes.levels
        @test Array(device_grid.node_keys) == expected_nodes.keys
        @test Array(device_grid.node_coords) == expected_nodes.coords
        @test Array(device_grid.node_centers) ≈ expected_nodes.centers
        @test Array(device_grid.parent_index) == expected_nodes.parent_index
        @test Array(device_grid.child_ranges) == expected_nodes.child_ranges
        @test Array(device_grid.leaf_to_node) == expected_nodes.leaf_to_node
        @test count(==(0), Array(device_grid.node_levels)) == 1
        @test all(Array(device_grid.node_levels)[Array(device_grid.leaf_to_node)] .== grid.ell)

        for ell_test in (0, 3)
            test_positions = ell_test == 0 ? positions : Float64[
                0.03 0.29 0.56 0.91 0.12
                0.08 0.61 0.43 0.19 0.88
                0.14 0.37 0.79 0.52 0.24
            ]
            test_system = CUDARadixCPUScalarSystem(
                test_positions, fill(0.01, size(test_positions, 2)),
                Float64.(1:size(test_positions, 2)),
            )
            test_device_system = CUDARadixDeviceScalarSystem(
                test_system, CUDA.CuArray(test_positions), CUDA.CuArray(test_system.radii),
                CUDA.CuArray(test_system.strengths),
            )
            test_grid = RadixGrid(test_system, ell_test)
            test_device_grid = cuda_radix_grid(test_device_system, ell_test)
            test_expected_nodes = _expected_cuda_radix_nodes(test_grid)
            @test Array(test_device_grid.node_levels) == test_expected_nodes.levels
            @test Array(test_device_grid.node_keys) == test_expected_nodes.keys
            @test Array(test_device_grid.node_coords) == test_expected_nodes.coords
            @test Array(test_device_grid.node_centers) ≈ test_expected_nodes.centers
            @test Array(test_device_grid.parent_index) == test_expected_nodes.parent_index
            @test Array(test_device_grid.child_ranges) == test_expected_nodes.child_ranges
            @test Array(test_device_grid.leaf_to_node) == test_expected_nodes.leaf_to_node
        end

        prepacked_device_grid = cuda_radix_grid(CUDA.CuArray(_cuda_radix_expected_body(grid, cpu_system)), 1)
        @test Array(prepacked_device_grid.perm) == collect(1:length(grid.perm))
        @test Array(prepacked_device_grid.cell_keys) == grid.cell_keys
        @test Array(prepacked_device_grid.cell_ranges) == grid.cell_ranges

        same_cell_positions = Float64[
            0.5 0.5 0.5 0.5
            0.5 0.5 0.5 0.5
            0.5 0.5 0.5 0.5
        ]
        same_cell_system = CUDARadixCPUScalarSystem(
            same_cell_positions, fill(0.01, size(same_cell_positions, 2)),
            Float64[4, 3, 2, 1],
        )
        same_cell_device_system = CUDARadixDeviceScalarSystem(
            same_cell_system, CUDA.CuArray(same_cell_positions),
            CUDA.CuArray(same_cell_system.radii), CUDA.CuArray(same_cell_system.strengths),
        )
        same_cell_grid = RadixGrid(same_cell_system, 3; h0_fallback=1.0)
        same_cell_device_grid = cuda_radix_grid(same_cell_device_system, 3; h0_fallback=1.0)
        @test length(same_cell_grid.cell_keys) == 1
        @test Array(same_cell_device_grid.perm) == same_cell_grid.perm
        @test Array(same_cell_device_grid.cell_ranges) == same_cell_grid.cell_ranges
        @test Array(same_cell_device_grid.body_system) == same_cell_grid.body_system
        @test Array(same_cell_device_grid.body_index) == same_cell_grid.body_index

        list = build_radix_interaction_list(LazyMaterializedBatches(1), ParentNeighborM2L(), grid)
        expected_body = _cuda_radix_expected_body(grid, cpu_system)

        dense_cuda_options = CUDARadixLifecycleOptions(
            m2l_strategy=DenseTranslationM2L())
        dense_cuda_error = try
            cuda_radix_state(cpu_system, grid, list, 1; options=dense_cuda_options)
            nothing
        catch err
            err
        end
        # DenseTranslationM2L is supported on the recurring device cache (task 023f);
        # only the one-shot cuda_radix_state builders reject it.
        @test dense_cuda_error isa ArgumentError
        @test occursin("one-shot", sprint(showerror, dense_cuda_error))
        @test occursin("023f", sprint(showerror, dense_cuda_error))

        fallback_state = cuda_radix_state(cpu_system, grid, list, 1)
        @test fallback_state.counters.body_uploads == 1
        @test Array(fallback_state.source_bodies) ≈ expected_body
        @test_throws ArgumentError cuda_radix_state(
            cpu_system, grid, list, 1;
            options=CUDARadixLifecycleOptions(; operator=FactoredRotationM2L()),
        )

        device_state = cuda_radix_state(device_system, grid, list, 1)
        @test device_state.counters.body_uploads == 0
        @test Array(device_state.source_bodies) ≈ expected_body
        @test run_cuda_radix_lifecycle!(device_state) === device_state
        @test device_state.counters.body_uploads == 0
        @test device_state.counters.influence_downloads == 0
        @test device_state.counters.expansion_host_copies == 0
        @test device_state.multipoles.phi isa CUDA.CuArray
        @test device_state.locals.phi isa CUDA.CuArray
        @test device_state.output isa CUDA.CuArray
        @test Array(device_state.multipoles.phi[:, 1]) ≈
              _expected_root_multipole(FastMultipole.host_resident_radix_grid(grid), expected_body, 1)

        @test_throws ArgumentError cuda_radix_state(device_system, device_grid, list, 1)
        device_grid_state = cuda_radix_state(device_system, device_grid, list, 1; host_grid)
        @test device_grid_state.counters.body_uploads == 0
        @test device_grid_state.counters.route_uploads <= device_state.counters.route_uploads
        @test Array(device_grid_state.source_bodies) ≈ expected_body
        @test length(device_grid_state.m2m_parent_routes) == length(device_grid.parent_index) - 1
        @test length(device_grid_state.l2l_child_routes) == length(device_grid.parent_index) - 1
        @test !isempty(device_grid_state.m2m_parent_routes)
        @test Array(device_grid_state.m2m_parent_routes) == Array(device_grid.parent_index)[2:end]
        @test Array(device_grid_state.m2m_child_routes) == collect(2:length(device_grid.parent_index))
        @test Array(device_grid_state.l2l_parent_routes) == Array(device_grid.parent_index)[2:end]
        @test Array(device_grid_state.l2l_child_routes) == collect(2:length(device_grid.parent_index))

        route_positions = Float64[
            0.03 0.29 0.56 0.91 0.12
            0.08 0.61 0.43 0.19 0.88
            0.14 0.37 0.79 0.52 0.24
        ]
        route_system = CUDARadixCPUScalarSystem(
            route_positions, fill(0.01, size(route_positions, 2)),
            Float64.(1:size(route_positions, 2)),
        )
        route_device_system = CUDARadixDeviceScalarSystem(
            route_system, CUDA.CuArray(route_positions), CUDA.CuArray(route_system.radii),
            CUDA.CuArray(route_system.strengths),
        )
        route_grid = RadixGrid(route_system, 3)
        route_host_grid = FastMultipole.host_resident_radix_grid(route_grid)
        route_device_grid = cuda_radix_grid(route_device_system, 3)
        route_list = build_radix_interaction_list(LazyMaterializedBatches(1), ParentNeighborM2L(), route_grid)
        route_state = cuda_radix_state(route_device_system, route_device_grid, route_list, 1; host_grid=route_host_grid)
        route_levels, route_target_cells, route_source_cells = _flatten_test_routes(route_list)
        @test !isempty(route_target_cells)
        route_leaf_to_node = Array(route_device_grid.leaf_to_node)
        expected_route_targets = route_leaf_to_node[route_target_cells]
        expected_route_sources = route_leaf_to_node[route_source_cells]
        @test Array(route_state.route_levels) == route_levels
        @test Array(route_state.route_targets) == expected_route_targets
        @test Array(route_state.route_sources) == expected_route_sources
        route_node_levels = Array(route_device_grid.node_levels)
        @test all(route_node_levels[Array(route_state.route_targets)] .== route_grid.ell)
        @test all(route_node_levels[Array(route_state.route_sources)] .== route_grid.ell)

        regression_positions = Float64[
            0.031 0.287 0.564 0.913 0.118 0.742 0.394 0.836
            0.083 0.617 0.431 0.197 0.881 0.324 0.756 0.529
            0.141 0.373 0.797 0.523 0.247 0.684 0.452 0.919
        ]
        regression_strengths = Float64[1.25, -2.5, 3.75, -1.5, 0.625, -0.875, 2.125, -3.25]
        regression_system = CUDARadixCPUScalarSystem(
            regression_positions, fill(0.01, size(regression_positions, 2)),
            regression_strengths,
        )
        regression_device_system = CUDARadixDeviceScalarSystem(
            regression_system, CUDA.CuArray(regression_positions),
            CUDA.CuArray(regression_system.radii), CUDA.CuArray(regression_system.strengths),
        )
        regression_grid = RadixGrid(regression_system, 3)
        regression_host_grid = FastMultipole.host_resident_radix_grid(regression_grid)
        regression_device_grid = cuda_radix_grid(regression_device_system, 3)
        regression_list = build_radix_interaction_list(
            LazyMaterializedBatches(1), ParentNeighborM2L(), regression_grid,
        )
        @test !isempty(regression_list.m2l_batches)
        @test !isempty(regression_list.direct_pairs)

        host_m2l_state = host_radix_state(regression_system, regression_grid, regression_list, 2)
        device_m2l_state = cuda_radix_state(
            regression_device_system, regression_device_grid, regression_list, 2;
            host_grid=regression_host_grid,
        )
        FastMultipole._launch_host_b2m!(host_m2l_state)
        FastMultipole._launch_host_m2m!(host_m2l_state)
        FastMultipole._launch_host_m2l!(host_m2l_state)
        FastMultipole._launch_cuda_b2m!(device_m2l_state)
        FastMultipole._launch_cuda_resident_m2m!(device_m2l_state)
        FastMultipole._launch_cuda_resident_m2l!(device_m2l_state)
        @test Array(device_m2l_state.locals.phi) ≈ host_m2l_state.locals.phi rtol=1e-9 atol=1e-10
        @test device_m2l_state.counters.expansion_host_copies == 0
        @test device_m2l_state.scratch.phis isa CUDA.CuArray
        @test device_m2l_state.scratch.m2l_targets.phi isa CUDA.CuArray
        # task 019: M2M/L2L staging buffers were replaced by the shared aphi..cchi
        # pool plus the StackedYChannel scratch — assert those are device-resident.
        @test device_m2l_state.scratch.aphi isa CUDA.CuArray
        @test device_m2l_state.scratch.ystk_phi.G isa CUDA.CuArray
        @test device_m2l_state.scratch.ystk_phi.mult_Ur isa CUDA.CuArray

        host_full_state = host_radix_state(regression_system, regression_grid, regression_list, 2)
        device_full_state = cuda_radix_state(
            regression_device_system, regression_device_grid, regression_list, 2;
            host_grid=regression_host_grid,
        )
        @test run_host_radix_lifecycle!(host_full_state) === host_full_state
        @test run_cuda_radix_lifecycle!(device_full_state) === device_full_state
        @test Array(device_full_state.output) ≈ host_full_state.output rtol=1e-9 atol=1e-10
        @test device_full_state.counters.body_uploads == 0
        @test device_full_state.counters.influence_downloads == 0
        @test device_full_state.counters.expansion_host_copies == 0

        prepacked_state = cuda_radix_state(CUDA.CuArray(expected_body), grid, list, 1)
        @test prepacked_state.counters.body_uploads == 0
        @test Array(prepacked_state.source_bodies) ≈ expected_body
        route_uploads_after_state = prepacked_state.counters.route_uploads

        host_output = zeros(Float64, size(prepacked_state.output))
        copy_cuda_radix_output!(host_output, prepacked_state)
        @test prepacked_state.counters.influence_downloads == 1
        @test host_output ≈ Array(prepacked_state.output)

        device_output = CUDA.zeros(Float64, size(prepacked_state.output))
        copy_cuda_radix_output!(device_output, prepacked_state)
        @test prepacked_state.counters.influence_downloads == 1
        @test Array(device_output) ≈ Array(prepacked_state.output)

        prepacked_state.output .= CUDA.CuArray(Float64[
            10 20 30 40
            1 2 3 4
            5 6 7 8
            9 10 11 12
        ])
        target_switch = DerivativesSwitch(true, true, false, cpu_system)

        # `output` is stored in sorted body order; finalize_cuda_radix_output!
        # de-permutes it back to original body order via grid.perm/body_index.
        sorted_potential = Float64[10, 20, 30, 40]
        sorted_gradient = Float64[
            1 2 3 4
            5 6 7 8
            9 10 11 12
        ]
        expected_potential = zeros(Float64, 4)
        expected_gradient = zeros(Float64, 3, 4)
        for sorted_i in 1:4
            ibody = grid.body_index[grid.perm[sorted_i]]
            expected_potential[ibody] = sorted_potential[sorted_i]
            expected_gradient[:, ibody] .= sorted_gradient[:, sorted_i]
        end

        host_target = CUDARadixCPUScalarSystem(positions, radii, strengths)
        finalize_cuda_radix_output!(prepacked_state, host_target; derivatives_switches=target_switch)
        @test prepacked_state.counters.influence_downloads == 2
        @test prepacked_state.counters.route_uploads == route_uploads_after_state
        @test host_target.potential ≈ expected_potential
        @test host_target.gradient ≈ expected_gradient

        device_target = CUDARadixDeviceScalarSystem(
            CUDARadixCPUScalarSystem(positions, radii, strengths),
            CUDA.CuArray(positions), CUDA.CuArray(radii), CUDA.CuArray(strengths),
        )
        finalize_cuda_radix_output!(prepacked_state, device_target; derivatives_switches=target_switch)
        @test prepacked_state.counters.influence_downloads == 2
        @test prepacked_state.counters.route_uploads == route_uploads_after_state
        @test device_target.host.potential ≈ expected_potential
        @test device_target.host.gradient ≈ host_target.gradient

        #--- rectangular device cache (task 037 stage 2) ---#

        # elongated cloud in [0, 4] x [0, 0.8]^2; vector bounds (4, 1, 1)
        # resolve ell_axes = (4, 2, 2) on host and device alike
        rect_rng = MersenneTwister(37)
        rect_n = 600
        rect_positions = rand(rect_rng, 3, rect_n)
        rect_positions[1, :] .*= 4.0
        rect_positions[2:3, :] .*= 0.8
        rect_radii = fill(0.01, rect_n)
        rect_strengths = Float64.(1:rect_n) ./ rect_n
        rect_host_sys = CUDARadixCPUScalarSystem(rect_positions, rect_radii, rect_strengths)
        rect_dev_sys = CUDARadixCPUScalarSystem(copy(rect_positions), copy(rect_radii),
            copy(rect_strengths))
        rect_bounds = (SVector(0.0, 0.0, 0.0), (4.0, 1.0, 1.0))
        rect_opts = CUDARadixLifecycleOptions(; precision=Float64,
            m2l_strategy=FastMultipole.ConcatenatedFixedZM2L())
        # lamb_helmholtz passed explicitly: CUDARadixCPUScalarSystem does not
        # overload has_vector_potential, so the constructor cannot derive it
        rect_hc = RadixFMMCache(rect_host_sys; expansion_order=3, ell=4,
            bounds=rect_bounds, options=rect_opts, lamb_helmholtz=false)
        rect_dc = RadixFMMCache(rect_dev_sys; expansion_order=3, ell=4,
            bounds=rect_bounds, options=rect_opts, device=true,
            lamb_helmholtz=false)
        @test rect_dc.ell_axes == rect_hc.ell_axes == SVector(4, 2, 2)
        @test rect_dc.box_extent == rect_hc.box_extent
        @test rect_dc.max_cells == rect_hc.max_cells
        @test rect_dc.max_nodes == rect_hc.max_nodes
        fmm!(rect_host_sys, rect_hc; scalar_potential=true, gradient=true)
        fmm!(rect_dev_sys, rect_dc; scalar_potential=true, gradient=true)
        # host route buffers are windowed scratch (037 stage 1 note): route and
        # direct parity is the telemetry counts plus the output parity below
        @test rect_dc.state.counts.n_cells == rect_hc.state.counts.n_cells
        @test rect_dc.state.counts.n_nodes == rect_hc.state.counts.n_nodes
        @test rect_dc.state.counts.n_routes == rect_hc.state.counts.n_routes
        @test rect_dc.state.counts.n_direct == rect_hc.state.counts.n_direct
        @test maximum(abs.(rect_dev_sys.potential .- rect_host_sys.potential)) < 1e-10
        @test maximum(abs.(rect_dev_sys.gradient .- rect_host_sys.gradient)) < 1e-9
        # per-axis out-of-box contract on device: inside the virtual cube but
        # outside the rectangular box in z
        rect_dev_sys.positions[3, 1] = 2.5
        @test_throws ArgumentError fmm!(rect_dev_sys, rect_dc;
            scalar_potential=true, gradient=true)
    end
end

@testset "ConcatenatedFixedZM2L host parity (task 022 throughput repair)" begin
    rng = Random.MersenneTwister(4022)
    n = 400
    positions = rand(rng, 3, n)
    bodies = vcat(positions, zeros(1, n), reshape(randn(rng, n), 1, n))

    # the P = 1 and P = 2 rows cover the 019b always-dense decision: the concat
    # dense path serves every order with no small-P recurrence fallback
    for (policy, P) in ((ParentNeighborM2L(), 1), (ParentNeighborM2L(), 4),
            (ConstantPAnalyticStencil(2, 1e-2), 2), (ConstantPAnalyticStencil(4, 1e-4), 4))
        grid = RadixGrid(bodies, 3)
        list = build_radix_interaction_list(LazyMaterializedBatches(8), policy, grid)
        base = host_radix_state(bodies, grid, list, P)
        run_host_radix_lifecycle!(base)
        # chunk smaller than the route count exercises multi-chunk boundaries and
        # cross-chunk scatter accumulation into repeated targets
        for chunk in (typemax(Int) >> 1, 37)
            concat = host_radix_state(bodies, grid, list, P;
                options=CUDARadixLifecycleOptions(; m2l_strategy=ConcatenatedFixedZM2L(chunk)))
            @test isempty(concat.scratch.m2l_groups)
            @test concat.scratch.m2l_concat isa FastMultipole.ResidentM2LConcatPlan
            run_host_radix_lifecycle!(concat)
            @test concat.output ≈ base.output rtol=1e-10 atol=1e-12
            @test concat.counters.expansion_host_copies == 0
        end
    end

    # Lamb-Helmholtz chi channel: scalar-source B2M leaves chi = 0, so drive the
    # M2L stage directly with synthetic multipoles to exercise the chi arithmetic.
    grid = RadixGrid(bodies, 3)
    list = build_radix_interaction_list(LazyMaterializedBatches(8), ParentNeighborM2L(), grid)
    for P_lh in (1, 3)   # P_lh = 1 covers the 019b always-dense decision under LH
        base_lh = host_radix_state(bodies, grid, list, P_lh, Val(true))
        concat_lh = host_radix_state(bodies, grid, list, P_lh, Val(true);
            options=CUDARadixLifecycleOptions(; m2l_strategy=ConcatenatedFixedZM2L(41)))
        mphi = randn(rng, size(base_lh.multipoles.phi))
        mchi = randn(rng, size(base_lh.multipoles.chi))
        for st in (base_lh, concat_lh)
            st.multipoles.phi .= mphi
            st.multipoles.chi .= mchi
        end
        FastMultipole._launch_resident_m2l!(base_lh, SharedRotationM2L())
        FastMultipole._launch_resident_m2l!(concat_lh, ConcatenatedFixedZM2L(41))
        @test concat_lh.locals.phi ≈ base_lh.locals.phi rtol=1e-10 atol=1e-12
        @test concat_lh.locals.chi ≈ base_lh.locals.chi rtol=1e-10 atol=1e-12
    end

    # Float32 smoke
    bodies32 = Float32.(bodies)
    grid32 = RadixGrid(bodies32, 3)
    list32 = build_radix_interaction_list(LazyMaterializedBatches(8), ParentNeighborM2L(), grid32)
    base32 = host_radix_state(bodies32, grid32, list32, 4;
        options=CUDARadixLifecycleOptions(; precision=Float32))
    run_host_radix_lifecycle!(base32)
    concat32 = host_radix_state(bodies32, grid32, list32, 4;
        options=CUDARadixLifecycleOptions(; precision=Float32, m2l_strategy=ConcatenatedFixedZM2L()))
    run_host_radix_lifecycle!(concat32)
    @test concat32.output ≈ base32.output rtol=2f-3 atol=1f-4
end
