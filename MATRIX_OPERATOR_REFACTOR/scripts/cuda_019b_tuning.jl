# Task 019b GPU exploratory benchmark: the small-P / tiny-batch corner the 019
# tuning data does not cover. Measures, per (config, P, LH):
#   * GPU resident lifecycle exec time + per-stage min-of-reps timings for the
#     dense concat M2L strategy (and the shared-rotation strategy at the tiny
#     parent-neighbor config, whose per-route batches are the tiny-batch case),
#   * host concat lifecycle wall time, and
#   * a per-route legacy-recurrence M2L stage time on the same host state (the
#     exact shape a small-P recurrence fallback would take on CPU).
# Protocol follows cuda_019_tuning.jl (min-of-reps, CSV under a marker line).
using FastMultipole
using FastMultipole.StaticArrays
using Random
using Printf
using Test

const FM = FastMultipole

function make_body_matrix(n; seed=190219)
    rng = MersenneTwister(seed)
    return vcat(rand(rng, 3, n), zeros(1, n), reshape(randn(rng, n), 1, n))
end

# ---- per-route legacy recurrence M2L over a host radix state (019b CPU shape) ----
struct RouteRecurrenceScratch{TF}
    P_run::Int
    src::Array{TF,3}
    dst::Array{TF,3}
    w1::Array{TF,3}
    w2::Array{TF,3}
    w3::Array{TF,3}
    Ts::Vector{TF}
    eimϕs::Matrix{TF}
    ζ::Vector{TF}
    η::Vector{TF}
    Hs::Vector{TF}
end

function RouteRecurrenceScratch(::Type{TF}, P_run) where TF
    Hs = TF[1.0]; FM.update_Hs_π2!(Hs, P_run)
    ζ = zeros(TF, FM.length_ζs(P_run)); FM.update_ζs_mag!(ζ, 0, P_run)
    η = zeros(TF, FM.length_ηs(P_run)); FM.update_ηs_mag!(η, 0, P_run)
    return RouteRecurrenceScratch{TF}(P_run,
        FM.initialize_expansion(P_run, TF), FM.initialize_expansion(P_run, TF),
        FM.initialize_expansion(P_run, TF), FM.initialize_expansion(P_run, TF),
        FM.initialize_expansion(P_run, TF),
        zeros(TF, FM.length_Ts(P_run)), zeros(TF, 2, P_run + 1), ζ, η, Hs)
end

function route_recurrence_m2l!(state, rs::RouteRecurrenceScratch{TF}, ::Val{LH}) where {TF,LH}
    orders = state.invariant_cache.basis_info.orders
    P_phi = orders.P_phi
    P_active = orders.P_active
    lh = Val(LH)
    box = SVector{3}(zero(TF), zero(TF), zero(TF))
    fill!(state.locals.phi, zero(TF))
    LH && fill!(state.locals.chi, zero(TF))
    @inbounds for route_i in eachindex(state.route_targets)
        target = state.route_targets[route_i]
        source = state.route_sources[route_i]
        FM._pack_flat_column!(rs.src, state.multipoles, source, P_phi, P_active, lh)
        fill!(rs.dst, zero(TF))
        src_center = SVector{3,TF}(state.grid.node_centers[1, source],
                                   state.grid.node_centers[2, source],
                                   state.grid.node_centers[3, source])
        tgt_center = SVector{3,TF}(state.grid.node_centers[1, target],
                                   state.grid.node_centers[2, target],
                                   state.grid.node_centers[3, target])
        src_branch = FM.Branch(2:2, 0, 1:0, 0, 1, src_center, zero(TF), box)
        tgt_branch = FM.Branch(2:2, 0, 1:0, 0, 1, tgt_center, zero(TF), box)
        FM.multipole_to_local!(rs.dst, tgt_branch, rs.src, src_branch,
            rs.w1, rs.w2, rs.w3, rs.Ts, rs.eimϕs, rs.ζ, rs.η, rs.Hs, FM.M̃, FM.L̃, rs.P_run, lh)
        FM._unpack_flat_column_accumulate!(state.locals, rs.dst, target, P_phi, P_active, lh)
    end
    return state
end

function _cuda_stage_times(state; reps=5)
    stage_fns = (
        t_b2m = FM._launch_cuda_b2m!,
        t_m2m = FM._launch_cuda_resident_m2m!,
        t_m2l = FM._launch_cuda_resident_m2l!,
        t_l2l = FM._launch_cuda_resident_l2l!,
        t_l2b = FM._launch_cuda_resident_l2b!,
    )
    vals = map(stage_fns) do f
        f(state); CUDA.synchronize()   # warm
        minimum((@elapsed (f(state); CUDA.synchronize())) for _ in 1:reps)
    end
    return vals
end

const CSV_HEADER = "config,n,ell,P,lamb_helmholtz,policy,strategy,routes,direct_pairs,list_time,build_time,exec_time,exec_max,host_time,host_concat_m2l,host_recur_m2l,t_b2m,t_m2m,t_m2l,t_l2l,t_l2b"

function print_row(row; prefix="")
    @printf("%s%s,%d,%d,%d,%s,%s,%s,%d,%d,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e\n",
        prefix, row.config, row.n, row.ell, row.P, string(row.lamb_helmholtz), row.policy, row.strategy,
        row.routes, row.direct_pairs, row.list_time, row.build_time, row.exec_time, row.exec_max,
        row.host_time, row.host_concat_m2l, row.host_recur_m2l,
        row.t_b2m, row.t_m2m, row.t_m2l, row.t_l2l, row.t_l2b)
    flush(stdout)
end

function smallp_sweep()
    rows = NamedTuple[]
    # host_reps / host_recur: the host mirror at medium_constp walks ~12M routes per
    # call, which blew the first submission's 1 h wall time; the per-route recurrence
    # evidence at scale is already covered by the fm019bcpu job, so the medium config
    # keeps only a single-rep host concat reference for GPU parity/anchor.
    #
    # shared_reps: SharedRotationM2L is the launch-bound per-batch path and runs
    # 50-160 s per M2L execution at ~100k routes (measured, fm019b-12746806) — the
    # second submission spent its whole 3 h wall time timing it at reps=7. The
    # shared strategy is therefore (a) measured only at the 2k-route tiny_parent
    # config, where it already loses to concat by ~500-1000x, and (b) timed with
    # fewer reps — millisecond precision is irrelevant at that margin.
    configs = (
        (; name="tiny_parent", n=128, ell=2, policy=:parent, P_list=(1, 2, 3, 4),
           strategies=(:shared, :concat), reps=7, shared_reps=2, host_reps=3, host_recur=true),
        (; name="tiny_parent3", n=512, ell=3, policy=:parent, P_list=(1, 2, 3, 4),
           strategies=(:concat,), reps=7, shared_reps=2, host_reps=3, host_recur=true),
        (; name="small_constp", n=1_000, ell=3, policy=:constp, P_list=(2, 3, 4),
           strategies=(:concat,), reps=7, shared_reps=2, host_reps=3, host_recur=true),
        (; name="medium_constp", n=10_000, ell=4, policy=:constp, P_list=(2, 3, 4),
           strategies=(:concat,), reps=5, shared_reps=2, host_reps=1, host_recur=false),
    )
    for cfg in configs
        bodies = make_body_matrix(cfg.n)
        grid = RadixGrid(bodies, cfg.ell)
        host_grid = FM.host_resident_radix_grid(grid)
        dev_bodies = CUDA.CuArray(bodies)
        device_grid = cuda_radix_grid(dev_bodies, cfg.ell)
        for P in cfg.P_list, LHbool in (false, true)
            lh = Val(LHbool)
            policy = cfg.policy === :parent ? ParentNeighborM2L() :
                ConstantPAnalyticStencil(P, P >= 8 ? 1e-8 : (P >= 4 ? 1e-4 : 1e-2))
            list_time = @elapsed list = build_radix_interaction_list(
                LazyMaterializedBatches(32), policy, grid)
            routes = sum((length(b.targets) for b in list.m2l_batches); init=0)
            routes == 0 && continue

            # host references: concat lifecycle (+ per-route recurrence M2L stage
            # where enabled — see the config comment above)
            host_state = host_radix_state(bodies, grid, list, P, lh;
                options=CUDARadixLifecycleOptions(; m2l_strategy=ConcatenatedFixedZM2L()))
            run_host_radix_lifecycle!(host_state)
            host_time = minimum((@elapsed run_host_radix_lifecycle!(host_state)) for _ in 1:cfg.host_reps)
            FM._launch_host_b2m!(host_state)
            FM._launch_host_m2m!(host_state)
            host_recur_m2l = NaN
            if cfg.host_recur
                rscratch = RouteRecurrenceScratch(Float64,
                    host_state.invariant_cache.basis_info.orders.P_active)
                route_recurrence_m2l!(host_state, rscratch, lh)   # warm
                host_recur_m2l = minimum(
                    (@elapsed route_recurrence_m2l!(host_state, rscratch, lh)) for _ in 1:cfg.host_reps)
            end
            host_concat_m2l = minimum(
                (@elapsed FM._launch_host_m2l!(host_state)) for _ in 1:cfg.host_reps)
            # clean host reference values for the GPU parity check
            run_host_radix_lifecycle!(host_state)

            for strat_name in cfg.strategies
                strat = strat_name === :shared ? SharedRotationM2L() : ConcatenatedFixedZM2L()
                reps = strat_name === :shared ? cfg.shared_reps : cfg.reps
                options = CUDARadixLifecycleOptions(; m2l_strategy=strat)
                build_time = @elapsed begin
                    state = cuda_radix_state(dev_bodies, device_grid, list, P, lh;
                        options, host_grid)
                    CUDA.synchronize()
                end
                run_cuda_radix_lifecycle!(state)   # warm/JIT
                CUDA.synchronize()
                exec_samples = [(@elapsed (run_cuda_radix_lifecycle!(state); CUDA.synchronize()))
                    for _ in 1:reps]
                exec_time = minimum(exec_samples)
                exec_max = maximum(exec_samples)
                stages = _cuda_stage_times(state; reps)
                run_cuda_radix_lifecycle!(state)   # clean pass before parity check
                CUDA.synchronize()
                gpu_out = Array(state.output)
                @test gpu_out ≈ host_state.output rtol=1e-9 atol=1e-10
                row = (;
                    config=cfg.name, cfg.n, cfg.ell, P, lamb_helmholtz=LHbool,
                    policy=cfg.policy, strategy=strat_name,
                    routes, direct_pairs=length(list.direct_pairs),
                    list_time, build_time, exec_time, exec_max,
                    host_time, host_concat_m2l, host_recur_m2l, stages...,
                )
                push!(rows, row)
                # stream each row immediately so a wall-time kill keeps partial data
                print_row(row; prefix="CUDA_019B_ROW,")
            end
        end
        CUDA.reclaim()
    end
    return rows
end

if !FM.load_cuda_radix_lifecycle!()
    println("CUDA_019B_CSV")
    println("CUDA_019B_RESULT")
    println("FAIL")
    println("CUDA radix lifecycle did not load: ", FM.cuda_radix_status())
    exit(1)
end

using CUDA
CUDA.versioninfo()
CUDA.allowscalar(false)
println("device_name=", CUDA.name(CUDA.device()))
println("cuda_status=", FM.cuda_radix_status())
println("CUDA_019B_ROW_HEADER,", CSV_HEADER)
flush(stdout)

ok = false
rows = Any[]
failure = nothing
try
    global rows = smallp_sweep()
    global ok = true
catch err
    global failure = (err, catch_backtrace())
end

println("CUDA_019B_CSV")
println(CSV_HEADER)
for row in rows
    print_row(row)
end
println("CUDA_019B_RESULT")
if ok
    println("PASS")
else
    println("FAIL")
    showerror(stdout, failure[1], failure[2])
    println()
    exit(1)
end
