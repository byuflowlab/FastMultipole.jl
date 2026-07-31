using FastMultipole
using FastMultipole.StaticArrays
using Random
using Printf
using Test

struct CUDA022ScalarSystem{TF}
    positions::Matrix{TF}
    radii::Vector{TF}
    strengths::Vector{TF}
    potential::Vector{TF}
    gradient::Matrix{TF}
end

CUDA022ScalarSystem(positions::Matrix{TF}, radii::Vector{TF}, strengths::Vector{TF}) where TF =
    CUDA022ScalarSystem(positions, radii, strengths, zeros(TF, length(strengths)), zeros(TF, 3, length(strengths)))

struct CUDA022DeviceSystem{TF,A,B,C}
    host::CUDA022ScalarSystem{TF}
    positions::A
    radii::B
    strengths::C
end

Base.eltype(::CUDA022ScalarSystem{TF}) where TF = TF
FastMultipole.get_n_bodies(system::CUDA022ScalarSystem) = size(system.positions, 2)
FastMultipole.data_per_body(::CUDA022ScalarSystem) = 5
FastMultipole.strength_dims(::CUDA022ScalarSystem) = 1
FastMultipole.get_position(system::CUDA022ScalarSystem{TF}, i) where TF =
    SVector{3,TF}(system.positions[1, i], system.positions[2, i], system.positions[3, i])

function FastMultipole.source_system_to_buffer!(buffer, i_buffer, system::CUDA022ScalarSystem, i_body)
    buffer[1:3, i_buffer] .= system.positions[:, i_body]
    buffer[4, i_buffer] = system.radii[i_body]
    buffer[5, i_buffer] = system.strengths[i_body]
    return buffer
end

function FastMultipole.buffer_to_target_system!(target_system::CUDA022ScalarSystem,
        i_target, derivatives_switch, target_buffer, i_buffer)
    target_system.potential[i_target] +=
        FastMultipole.get_scalar_potential(target_buffer, derivatives_switch, i_buffer)
    target_system.gradient[:, i_target] .+=
        FastMultipole.get_gradient(target_buffer, derivatives_switch, i_buffer)
    return target_system
end

Base.eltype(::CUDA022DeviceSystem{TF}) where TF = TF
FastMultipole.get_n_bodies(system::CUDA022DeviceSystem) = FastMultipole.get_n_bodies(system.host)
FastMultipole.data_per_body(system::CUDA022DeviceSystem) = FastMultipole.data_per_body(system.host)
FastMultipole.strength_dims(system::CUDA022DeviceSystem) = FastMultipole.strength_dims(system.host)
FastMultipole.get_position(system::CUDA022DeviceSystem, i) = FastMultipole.get_position(system.host, i)
FastMultipole.residency(::CUDA022DeviceSystem) = DeviceResident()

function FastMultipole.source_to_buffer!(device_buffer, system::CUDA022DeviceSystem, sort_index)
    device_buffer[1:3, :] .= system.positions[:, sort_index]
    device_buffer[4, :] .= system.radii[sort_index]
    device_buffer[5, :] .= system.strengths[sort_index]
    return device_buffer
end

function FastMultipole.buffer_to_target!(target_system::CUDA022DeviceSystem,
        device_output_buffer, derivatives_switch, sort_index)
    target_system.host.potential .= Array(device_output_buffer[FastMultipole.scalar_potential_index(derivatives_switch), :])
    target_system.host.gradient .= Array(device_output_buffer[FastMultipole.gradient_range(derivatives_switch), :])
    return target_system
end

function make_system(::Type{TF}, n; seed=22022) where TF
    rng = MersenneTwister(seed)
    positions = TF.(rand(rng, 3, n))
    strengths = TF.(randn(rng, n))
    radii = fill(TF(0.01), n)
    return CUDA022ScalarSystem(positions, radii, strengths)
end

function device_system(system::CUDA022ScalarSystem)
    return CUDA022DeviceSystem(
        system,
        CUDA.CuArray(system.positions),
        CUDA.CuArray(system.radii),
        CUDA.CuArray(system.strengths),
    )
end

function run_case(::Type{TF}, lh::Val{LH}; n=96, ell=3, P=2) where {TF,LH}
    cpu_system = make_system(TF, n; seed=LH ? 22023 : 22022)
    dev_system = device_system(cpu_system)
    grid = RadixGrid(cpu_system, ell)
    host_grid = FastMultipole.host_resident_radix_grid(grid)
    device_grid = cuda_radix_grid(dev_system, ell)
    list = build_radix_interaction_list(LazyMaterializedBatches(8), ParentNeighborM2L(), grid)

    host_state = host_radix_state(cpu_system, grid, list, P, lh; options=CUDARadixLifecycleOptions(; precision=TF))
    run_host_radix_lifecycle!(host_state)

    rtol = TF === Float32 ? 5f-4 : 5e-10
    atol = TF === Float32 ? 5f-5 : 5e-11

    host_origin_state = cuda_radix_state(cpu_system, device_grid, list, P, lh;
        options=CUDARadixLifecycleOptions(; precision=TF), host_grid)
    device_origin_state = cuda_radix_state(dev_system, device_grid, list, P, lh;
        options=CUDARadixLifecycleOptions(; precision=TF), host_grid)
    concat_state = cuda_radix_state(cpu_system, device_grid, list, P, lh;
        options=CUDARadixLifecycleOptions(; precision=TF, m2l_strategy=ConcatenatedFixedZM2L()),
        host_grid)

    run_cuda_radix_lifecycle!(host_origin_state)
    run_cuda_radix_lifecycle!(device_origin_state)
    run_cuda_radix_lifecycle!(concat_state)

    host_output = Array(host_origin_state.output)
    device_output = Array(device_origin_state.output)
    concat_output = Array(concat_state.output)
    @test host_output ≈ host_state.output rtol=rtol atol=atol
    @test device_output ≈ host_state.output rtol=rtol atol=atol
    @test concat_output ≈ host_state.output rtol=rtol atol=atol
    @test concat_state.counters.expansion_host_copies == 0

    downloaded = similar(host_state.output)
    copy_cuda_radix_output!(downloaded, host_origin_state)
    @test host_origin_state.counters.body_uploads == 1
    @test host_origin_state.counters.influence_downloads == 1
    @test host_origin_state.counters.expansion_host_copies == 0
    @test device_origin_state.counters.body_uploads == 0
    @test device_origin_state.counters.influence_downloads == 0
    @test device_origin_state.counters.expansion_host_copies == 0

    return (; precision=string(TF), lamb_helmholtz=LH, n, ell, P,
        m2l_routes=length(host_state.route_targets),
        max_batch=maximum((length(batch.targets) for batch in list.m2l_batches); init=0),
        host_body_uploads=host_origin_state.counters.body_uploads,
        host_downloads=host_origin_state.counters.influence_downloads,
        device_body_uploads=device_origin_state.counters.body_uploads,
        expansion_host_copies=device_origin_state.counters.expansion_host_copies)
end

function make_body_matrix(n; seed=22024)
    rng = MersenneTwister(seed)
    return vcat(rand(rng, 3, n), zeros(1, n), reshape(randn(rng, n), 1, n))
end

function _cuda_stage_times(state)
    CUDA.synchronize()
    t_b2m = @elapsed (FastMultipole._launch_cuda_b2m!(state); CUDA.synchronize())
    t_m2m = @elapsed (FastMultipole._launch_cuda_resident_m2m!(state); CUDA.synchronize())
    t_m2l = @elapsed (FastMultipole._launch_cuda_resident_m2l!(state); CUDA.synchronize())
    t_l2l = @elapsed (FastMultipole._launch_cuda_resident_l2l!(state); CUDA.synchronize())
    t_l2b = @elapsed (FastMultipole._launch_cuda_resident_l2b!(state); CUDA.synchronize())
    return (; t_b2m, t_m2m, t_m2l, t_l2l, t_l2b)
end

# Throughput protocol (022 repair, 2026-07-10): state construction and lifecycle
# execution are timed separately; execution is the minimum of `reps` re-runs of a
# warmed, reused state. The ConstantPAnalyticStencil configs are the wide-batch
# cases the 008c break-even targets; the tiny ParentNeighborM2L case is kept for
# continuity with the earlier group-looping numbers. The host mirror time is a
# single-thread reference of the same operator path, not the tuned legacy CPU FMM.
function throughput_sweep()
    rows = NamedTuple[]
    configs = (
        (; n=128, ell=3, P=2, policy=:parent, strategies=(:shared, :concat), reps=3, host_ref=true),
        (; n=10_000, ell=4, P=4, policy=:constp, strategies=(:concat,), reps=3, host_ref=true),
        (; n=100_000, ell=4, P=4, policy=:constp, strategies=(:concat,), reps=3, host_ref=false),
        (; n=100_000, ell=4, P=8, policy=:constp, strategies=(:concat,), reps=3, host_ref=false),
    )
    for cfg in configs
        bodies = make_body_matrix(cfg.n)
        grid = RadixGrid(bodies, cfg.ell)
        host_grid = FastMultipole.host_resident_radix_grid(grid)
        dev_bodies = CUDA.CuArray(bodies)
        device_grid = cuda_radix_grid(dev_bodies, cfg.ell)
        policy = cfg.policy === :parent ? ParentNeighborM2L() :
            ConstantPAnalyticStencil(cfg.P, cfg.P >= 8 ? 1e-8 : 1e-4)
        list_time = @elapsed list = build_radix_interaction_list(
            LazyMaterializedBatches(32), policy, grid)
        routes = sum((length(b.targets) for b in list.m2l_batches); init=0)

        host_time = NaN
        if cfg.host_ref
            host_state = host_radix_state(bodies, grid, list, cfg.P;
                options=CUDARadixLifecycleOptions(; m2l_strategy=ConcatenatedFixedZM2L()))
            run_host_radix_lifecycle!(host_state)
            host_time = @elapsed run_host_radix_lifecycle!(host_state)
        end

        for strat_name in cfg.strategies
            strat = strat_name === :shared ? SharedRotationM2L() : ConcatenatedFixedZM2L()
            options = CUDARadixLifecycleOptions(; m2l_strategy=strat)
            build_time = @elapsed begin
                state = cuda_radix_state(dev_bodies, device_grid, list, cfg.P;
                    options, host_grid)
                CUDA.synchronize()
            end
            run_cuda_radix_lifecycle!(state)   # warm/JIT
            CUDA.synchronize()
            exec_time = minimum(
                (@elapsed (run_cuda_radix_lifecycle!(state); CUDA.synchronize()))
                for _ in 1:cfg.reps
            )
            stages = _cuda_stage_times(state)
            if cfg.host_ref
                gpu_out = Array(state.output)
                @test gpu_out ≈ host_state.output rtol=1e-9 atol=1e-10
            end
            push!(rows, (;
                cfg.n, cfg.ell, cfg.P, policy=cfg.policy, strategy=strat_name,
                routes, direct_pairs=length(list.direct_pairs),
                list_time, build_time, exec_time, host_time, stages...,
            ))
        end
    end
    return rows
end

function run_all!(rows)
    for TF in (Float64, Float32), lh in (Val(false), Val(true))
        push!(rows, run_case(TF, lh))
    end
    return throughput_sweep()
end

# The lifecycle include happens at runtime, so everything after the load must be
# driven from top level (each top-level statement sees the latest world); a
# `main()` function invoked before the load would run in a frozen world and
# dispatch to the pre-load fallback methods.
if !FastMultipole.load_cuda_radix_lifecycle!()
    println("CUDA_022_CASES")
    println("CUDA_022_THROUGHPUT")
    println("CUDA_022_RESULT")
    println("FAIL")
    println("CUDA radix lifecycle did not load: ", FastMultipole.cuda_radix_status())
    exit(1)
end

using CUDA
CUDA.versioninfo()
CUDA.allowscalar(false)
println("device_name=", CUDA.name(CUDA.device()))
println("cuda_status=", FastMultipole.cuda_radix_status())

ok = false
rows = Any[]
sweep_rows = Any[]
failure = nothing
try
    global sweep_rows = run_all!(rows)
    global ok = true
catch err
    global failure = (err, catch_backtrace())
end

println("CUDA_022_CASES")
for row in rows
    println(row)
end
println("CUDA_022_THROUGHPUT")
for row in sweep_rows
    @printf("n=%d ell=%d P=%d policy=%s strategy=%s routes=%d direct=%d list_t=%.3e build_t=%.3e exec_t=%.3e host_t=%.3e b2m=%.3e m2m=%.3e m2l=%.3e l2l=%.3e l2b=%.3e\n",
        row.n, row.ell, row.P, row.policy, row.strategy, row.routes, row.direct_pairs,
        row.list_time, row.build_time, row.exec_time, row.host_time,
        row.t_b2m, row.t_m2m, row.t_m2l, row.t_l2l, row.t_l2b)
end
println("CUDA_022_RESULT")
if ok
    println("PASS")
else
    println("FAIL")
    showerror(stdout, failure[1], failure[2])
    println()
    exit(1)
end
