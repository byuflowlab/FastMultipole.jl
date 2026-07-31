using FastMultipole
using FastMultipole.StaticArrays
using FastMultipole.LinearAlgebra
using Dates
using Random
using Statistics

include(joinpath(@__DIR__, "benchmark_024_common.jl"))

const SCHEMA026 = "026.2"
const N026 = parse(Int, get(ENV, "FM026_N", "20000"))
const ELL026 = parse(Int, get(ENV, "FM026_ELL", "4"))
const P026 = parse(Int, get(ENV, "FM026_P", "4"))
const TF026 = get(ENV, "FM026_TF", "Float64") == "Float32" ? Float32 : Float64
const LH026 = _envbool("FM026_LH")
const REPS026 = parse(Int, get(ENV, "FM026_REPS", "7"))
const WARMUPS026 = parse(Int, get(ENV, "FM026_WARMUPS", "2"))
const WINDOW026 = parse(Int, get(ENV, "FM026_WINDOW_CLASSES", "8"))
const POLICY026 = get(ENV, "FM026_POLICY", "hierarchical_12")
const STRATEGY026 = get(ENV, "FM026_STRATEGY", "concat")
const DISTRIBUTION026 = get(ENV, "FM026_DISTRIBUTION", "uniform")
const SEED026 = parse(Int, get(ENV, "FM026_SEED",
    string(26026000 + N026 + 101ELL026 + 17Int(LH026) +
        (TF026 === Float32 ? 1 : 0))))
const ACCURACY026 = _envbool("FM026_ACCURACY")
const PERFORMANCE026 = _envbool("FM026_PERFORMANCE", true)
const STAGE_DETAIL026 = _envbool("FM026_STAGE_DETAIL")
const STEP_TIMING026 = _envbool("FM026_STEP_TIMING", true)
const ALLOC_REPS026 = parse(Int, get(ENV, "FM026_ALLOC_REPS", "1"))
const ACCURACY_TARGETS026 = parse(Int, get(ENV, "FM026_ACCURACY_TARGETS", "256"))
const BLAS_THREADS026 = parse(Int, get(ENV, "FM026_BLAS_THREADS",
    string(BLAS.get_num_threads())))
const OUT026 = get(ENV, "FM026_OUT", joinpath(@__DIR__, "..", "data",
    "hierarchical_m2l_host", "raw",
    "host_$(gethostname())_$(Dates.format(now(), "yyyymmdd-HHMMSS")).csv"))
const LEVEL_OUT026 = get(ENV, "FM026_LEVEL_OUT",
    replace(OUT026, r"\.csv$" => "_levels.csv"))

BLAS.set_num_threads(BLAS_THREADS026)
BLAS.get_num_threads() == BLAS_THREADS026 ||
    error("requested BLAS=$BLAS_THREADS026, got $(BLAS.get_num_threads())")

function epsilon026(::Type{TF}, P, q, h0, ell, lh) where TF
    q == 3 && return TF(1.0e12)
    probe = ConstantPStencilConfig(P, one(TF); lamb_helmholtz=lh)
    upper = constant_p_stencil_bound(TF(h0), ell, probe, SVector(2, 2, 2))
    lower = constant_p_stencil_bound(TF(h0), ell, probe, SVector(3, 2, 0))
    return (upper + lower) / 2
end

function strategy026(label)
    row = only(filter(x -> x.label == label, FM024_STRATEGIES))
    return CUDARadixLifecycleOptions(; precision=TF026, operator=row.operator,
        m2l_strategy=row.strategy)
end

function policy026(label, h0)
    q = endswith(label, "_3") ? 3 : 12
    eps = epsilon026(TF026, P026, q, h0, ELL026, LH026)
    config = ConstantPStencilConfig(P026, eps; lamb_helmholtz=LH026)
    startswith(label, "flat") && return ConstantPAnalyticStencil(config)
    startswith(label, "hierarchical") || error("unknown FM026_POLICY=$label")
    return HierarchicalRigidStencil(config; near_radius2=q,
        window_classes=WINDOW026)
end

function samples026(prep, f)
    for _ in 1:WARMUPS026
        prep()
        f()
    end
    values = Float64[]
    for _ in 1:REPS026
        prep()
        t0 = time_ns()
        f()
        push!(values, (time_ns() - t0) / 1e6)
    end
    return fm024_stats_ms(values)
end

function allocated026(f)
    f()
    best = typemax(Int)
    for _ in 1:ALLOC_REPS026
        best = min(best, @allocated f())
    end
    return best
end

function sampled_errors026(sys, ref, seed)
    n = size(sys.potential, 2)
    ns = min(ACCURACY_TARGETS026, n)
    rng = MersenneTwister(seed + 991)
    ids = sort!(randperm(rng, n)[1:ns])
    perr = LH026 ? NaN : maximum(abs.(
        Float64.(sys.potential[1, ids]) .- ref.potential[1, ids]))
    gerr = maximum(abs.(
        Float64.(sys.potential[5:7, ids]) .- ref.potential[5:7, ids]))
    return ns, perr, gerr
end

function class_occupancy026(state)
    ctx = state.interaction_list
    if !(ctx isa FastMultipole.HostHierarchicalM2LContext)
        plan = state.scratch.m2l_concat
        route_class = plan !== nothing && hasproperty(plan, :route_class) ?
            getproperty(plan, :route_class) : nothing
        nclasses = route_class === nothing || state.counts.n_routes == 0 ? 0 :
            maximum(@view route_class[1:state.counts.n_routes])
        counts = zeros(Int, nclasses)
        if plan !== nothing && hasproperty(plan, :route_class) &&
                getproperty(plan, :route_class) !== nothing
            @inbounds for i in 1:state.counts.n_routes
                c = Int(route_class[i])
                c > 0 && (counts[c] += 1)
            end
        end
        return [(level=state.grid.ell, routes=state.counts.n_routes,
            occupied_nodes=state.counts.n_cells,
            stats=fm024_occupancy(filter(>(0), counts)))]
    end

    noffsets = length(ctx.tables.push_offsets)
    plan = ctx.apply_plan
    route_class = plan.route_class
    rows = NamedTuple[]
    for level in 2:state.grid.ell
        counts = zeros(Int, noffsets)
        routes = 0
        for first_offset in 1:ctx.window_classes:noffsets
            last_offset = min(first_offset + ctx.window_classes - 1, noffsets)
            n = FastMultipole.build_hierarchical_routes_window!(
                state.route_levels, state.route_offsets, state.route_targets,
                state.route_sources, route_class, ctx, state.grid, level,
                first_offset, last_offset)
            routes += n
            for i in 1:n
                local_class =
                    Int(route_class[i]) - (level - 2) * noffsets
                1 <= local_class <= noffsets || throw(AssertionError(
                    "route class $(route_class[i]) at level $level maps to " *
                    "invalid local class $local_class (noffsets=$noffsets)"))
                counts[local_class] += 1
            end
        end
        lo = ctx.level_offsets[level + 1] + 1
        hi = ctx.level_offsets[level + 2]
        push!(rows, (level, routes, occupied_nodes=max(hi - lo + 1, 0),
            stats=fm024_occupancy(filter(>(0), counts))))
    end
    return rows
end

function measure026()
    base = fm024_system(SEED026, N026, DISTRIBUTION026)
    h0 = 0.51
    policy = policy026(POLICY026, h0)
    options = strategy026(STRATEGY026)

    sys = fm024_copy_system(base)
    t0 = time_ns()
    cache = RadixFMMCache(sys; expansion_order=P026, ell=ELL026,
        max_n_bodies=N026,
        bounds=(SVector(-0.01, -0.01, -0.01), 1.02),
        lamb_helmholtz=LH026, policy, options)
    construction_ms = (time_ns() - t0) / 1e6
    state = cache.state
    ctx = state.interaction_list
    hierarchical = ctx isa FastMultipole.HostHierarchicalM2LContext
    hierarchical && (ctx.profile_stages = true)

    fmm!(sys, cache; scalar_potential=!LH026, gradient=true)
    accuracy_samples = 0
    potential_error = NaN
    gradient_error = NaN
    if ACCURACY026
        ref = fm024_copy_system(base)
        direct!(ref; scalar_potential=!LH026, gradient=true)
        accuracy_samples, potential_error, gradient_error =
            sampled_errors026(sys, ref, SEED026)
    end

    empty_stats = fm024_stats_ms(Float64[])
    b2m = STAGE_DETAIL026 ? samples026(() -> nothing,
        () -> FastMultipole._launch_host_b2m!(state)) : empty_stats
    m2m = STAGE_DETAIL026 ?
        samples026(() -> FastMultipole._launch_host_b2m!(state),
            () -> FastMultipole._launch_host_m2m!(state)) : empty_stats
    level_samples = [Float64[] for _ in 1:(ELL026 + 1)]
    m2l = PERFORMANCE026 ? samples026(
        () -> begin
            FastMultipole._launch_host_b2m!(state)
            FastMultipole._launch_host_m2m!(state)
        end,
        () -> begin
            FastMultipole._launch_host_m2l!(state)
            if hierarchical
                for level in 2:ELL026
                    push!(level_samples[level + 1],
                        Float64(ctx.m2l_level_ns[level + 1]) / 1e6)
                end
            end
        end) : empty_stats
    l2l = STAGE_DETAIL026 ? samples026(
        () -> begin
            FastMultipole._launch_host_b2m!(state)
            FastMultipole._launch_host_m2m!(state)
            FastMultipole._launch_host_m2l!(state)
        end,
        () -> FastMultipole._launch_host_l2l!(state)) : empty_stats
    l2b = STAGE_DETAIL026 ? samples026(
        () -> begin
            FastMultipole._launch_host_b2m!(state)
            FastMultipole._launch_host_m2m!(state)
            FastMultipole._launch_host_m2l!(state)
            FastMultipole._launch_host_l2l!(state)
        end,
        () -> FastMultipole._launch_host_l2b!(state)) : empty_stats

    update_samples = Float64[]
    update_stage_samples = [Float64[] for _ in 1:5]
    for step in 1:(WARMUPS026 + REPS026)
        fm024_jitter!(sys, SEED026, step)
        t_update = time_ns()
        FastMultipole.update_radix_state!(cache, (sys,))
        elapsed = (time_ns() - t_update) / 1e6
        if step > WARMUPS026
            push!(update_samples, elapsed)
            if hierarchical
                for i in 1:5
                    push!(update_stage_samples[i],
                        Float64(ctx.update_stage_ns[i]) / 1e6)
                end
            end
        end
    end
    update_stats = fm024_stats_ms(update_samples)
    update_parts = map(fm024_stats_ms, update_stage_samples)

    step_stats = PERFORMANCE026 && STEP_TIMING026 ?
        samples026(() -> nothing,
            () -> fmm!(sys, cache; scalar_potential=!LH026, gradient=true)) :
        empty_stats
    m2l_alloc = allocated026(
        () -> FastMultipole._launch_resident_m2l!(state))
    step_alloc = allocated026(
        () -> fmm!(sys, cache; scalar_potential=!LH026, gradient=true))
    update_alloc = allocated026(
        () -> FastMultipole.update_radix_state!(cache, (sys,)))

    level_rows = class_occupancy026(state)
    memory = fm024_memory(state)
    routes_per_level = hierarchical ? join(ctx.routes_per_level, ';') :
        string(state.counts.n_routes)
    nclasses = hierarchical ? length(ctx.class_level) :
        length(cache.accepted_offsets)
    occupancy_bytes = hierarchical ? sizeof(ctx.occupancy.node_at) :
        sizeof(cache.cell_at)
    source_hash = get(ENV, "FM026_SOURCE_HASH", "unknown")
    job_id = get(ENV, "SLURM_JOB_ID", "none")

    row = (; schema_version=SCHEMA026, source_hash, slurm_job_id=job_id,
        host=gethostname(), julia_version=string(VERSION),
        blas_vendor=string(BLAS.vendor()), blas_threads=BLAS.get_num_threads(),
        julia_threads=Threads.nthreads(), policy=POLICY026,
        strategy=STRATEGY026, distribution=DISTRIBUTION026,
        precision=string(TF026), lh=LH026, n=N026, ell=ELL026, P=P026,
        seed=SEED026, window_classes=WINDOW026,
        cells=state.counts.n_cells, nodes=state.counts.n_nodes,
        routes=hierarchical ? ctx.total_routes : state.counts.n_routes,
        routes_per_level, direct_pairs=state.counts.n_direct, nclasses,
        construction_ms,
        grid_refresh_ms=update_parts[1].median,
        occupancy_refresh_ms=update_parts[2].median,
        direct_generation_ms=update_parts[3].median,
        scan_compact_ms=update_parts[4].median,
        tree_group_refresh_ms=update_parts[5].median,
        update_ms_median=update_stats.median,
        update_ms_min=update_stats.minimum, update_ms_iqr=update_stats.iqr,
        b2m_ms_median=b2m.median, m2m_ms_median=m2m.median,
        m2l_ms_median=m2l.median, m2l_ms_min=m2l.minimum,
        m2l_ms_iqr=m2l.iqr, l2l_ms_median=l2l.median,
        l2b_ms_median=l2b.median, step_ms_median=step_stats.median,
        step_ms_min=step_stats.minimum, step_ms_iqr=step_stats.iqr,
        m2l_alloc_bytes=m2l_alloc, update_alloc_bytes=update_alloc,
        step_alloc_bytes=step_alloc, occupancy_bytes,
        operator_bytes=memory.operator_bytes, metadata_bytes=memory.metadata_bytes,
        expansion_bytes=memory.expansion_bytes, scratch_bytes=memory.scratch_bytes,
        persistent_bytes=memory.persistent_bytes, peak_bytes=memory.peak_bytes,
        accuracy_samples, potential_error, gradient_error,
        expansion_host_copies=state.counters.expansion_host_copies,
        route_uploads=state.counters.route_uploads,
        operator_uploads=state.counters.operator_uploads)

    common = (; schema_version=SCHEMA026, source_hash,
        slurm_job_id=job_id, host=gethostname(),
        blas_threads=BLAS.get_num_threads(), policy=POLICY026,
        strategy=STRATEGY026, distribution=DISTRIBUTION026,
        precision=string(TF026), lh=LH026, n=N026, ell=ELL026, P=P026,
        seed=SEED026, window_classes=WINDOW026, accuracy_samples)
    expanded_levels = NamedTuple[]
    for lr in level_rows
        timed = hierarchical && lr.level + 1 <= length(level_samples) ?
            fm024_stats_ms(level_samples[lr.level + 1]) :
            fm024_stats_ms(Float64[])
        push!(expanded_levels, merge(common, (; level=lr.level,
            occupied_nodes=lr.occupied_nodes, routes=lr.routes,
            nonempty_classes=lr.stats.nonempty_classes,
            mean_occupancy=lr.stats.mean_occupancy,
            p50_occupancy=lr.stats.p50_occupancy,
            p90_occupancy=lr.stats.p90_occupancy,
            p95_occupancy=lr.stats.p95_occupancy,
            p99_occupancy=lr.stats.p99_occupancy,
            max_occupancy=lr.stats.max_occupancy,
            occupancy_skew=lr.stats.occupancy_skew,
            m2l_ms_median=timed.median, m2l_ms_min=timed.minimum,
            m2l_ms_iqr=timed.iqr)))
    end
    return row, expanded_levels
end

row, level_rows = measure026()
fm024_write_rows(OUT026, [row])
fm024_write_rows(LEVEL_OUT026, level_rows)
