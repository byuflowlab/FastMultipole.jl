# Task 024 definitive host resident-M2L case driver.
include(joinpath(@__DIR__, "benchmark_024_common.jl"))

const N = parse(Int, get(ENV, "FM024_N", "150"))
const P = parse(Int, get(ENV, "FM024_P", "4"))
const ELL = parse(Int, get(ENV, "FM024_ELL", "3"))
const TF = get(ENV, "FM024_TF", "Float64") == "Float32" ? Float32 : Float64
const LH = _envbool("FM024_LH")
const DISTRIBUTION = get(ENV, "FM024_DISTRIBUTION", "uniform")
const SEED = parse(Int, get(ENV, "FM024_SEED",
    string(240000 + N + 101P + 17Int(LH) + (TF === Float32 ? 1 : 0))))
const OUT = get(ENV, "FM024_OUT", joinpath(FM024_REPO,
    "MATRIX_OPERATOR_REFACTOR", "data", "operator_ab_benchmark",
    "raw", "cpu_$(gethostname())_$(Dates.format(now(), "yyyymmdd-HHMMSS")).csv"))
const BLAS_THREADS = parse(Int, get(ENV, "FM024_BLAS_THREADS",
    string(BLAS.get_num_threads())))
BLAS.set_num_threads(BLAS_THREADS)
BLAS.get_num_threads() == BLAS_THREADS ||
    error("requested BLAS=$BLAS_THREADS, got $(BLAS.get_num_threads())")

options(v) = CUDARadixLifecycleOptions(; precision=TF, operator=v.operator,
    m2l_strategy=v.strategy)
newcache(sys, v) = RadixFMMCache(sys; expansion_order=P, ell=ELL,
    max_n_bodies=N, bounds=(SVector(-0.01, -0.01, -0.01), 1.02),
    lamb_helmholtz=LH, options=options(v))

base = fm024_system(SEED, N, DISTRIBUTION)
direct_ref = fm024_copy_system(base)
FastMultipole.direct!(direct_ref; scalar_potential=true, gradient=true)

# Correctness gate: same bodies/routes, direct result, every resident strategy,
# and the full reconstructed-Ts coefficient oracle.  Full oracle coverage is
# intentionally stronger than route sampling for the large campaign cases.
gate = Dict{String,NamedTuple}()
resident_outputs = Dict{String,Matrix{Float64}}()
oracle_sys = fm024_copy_system(base)
oracle_cache = newcache(oracle_sys, FM024_STRATEGIES[1])
oracle_state = oracle_cache.state
FastMultipole._launch_host_b2m!(oracle_state)
FastMultipole._launch_host_m2m!(oracle_state)
fm024_reconstructed_ts_oracle!(oracle_state)
oracle_phi = copy(oracle_state.locals.phi)
oracle_chi = copy(oracle_state.locals.chi)
for v in FM024_STRATEGIES
    if v.label == "dense" && TF === Float32 && P == 12
        gate[v.label] = (status="infeasible",
            note="unsupported Float32/P=12 dense materialization",
            potential_error=NaN, gradient_error=NaN,
            coefficient_error=NaN, correctness_pass=false)
        continue
    end
    try
        sys = fm024_copy_system(base)
        cache = newcache(sys, v)
        fmm!(sys, cache; scalar_potential=!LH, gradient=true)
        perr, gerr = fm024_direct_errors(sys, direct_ref, LH)
        state = cache.state
        FastMultipole._launch_host_b2m!(state)
        FastMultipole._launch_host_m2m!(state)
        FastMultipole._launch_host_m2l!(state)
        phi = copy(state.locals.phi)
        chi = copy(state.locals.chi)
        coeff_ok, coeff_err = fm024_coeff_pass(TF, phi, oracle_phi, chi, oracle_chi)
        resident_outputs[v.label] = Float64.(sys.potential)
        direct_ok = fm024_direct_pass("cpu", TF, LH, perr, gerr)
        gate[v.label] = (status=(direct_ok && coeff_ok ? "eligible" : "disqualified"),
            note=(direct_ok && coeff_ok ? "" : "correctness tolerance failure"),
            potential_error=perr, gradient_error=gerr,
            coefficient_error=coeff_err, correctness_pass=direct_ok && coeff_ok)
    catch err
        gate[v.label] = (status="infeasible", note=fm024_infeasible_note(err),
            potential_error=NaN, gradient_error=NaN,
            coefficient_error=NaN, correctness_pass=false)
    end
end

# Cross-strategy output parity is part of the gate, independent of the oracle.
if !isempty(resident_outputs)
    first_output = first(values(resident_outputs))
    rtol, atol = TF === Float32 ? (5e-3, 5e-4) : (1e-9, 1e-10)
    for (label, output) in resident_outputs
        if !isapprox(output, first_output; rtol, atol)
            g = gate[label]
            gate[label] = merge(g, (status="disqualified",
                note="resident output parity failure", correctness_pass=false))
        end
    end
end

rows = NamedTuple[]
meta = fm024_metadata("cpu")
for v in FM024_STRATEGIES
    g = gate[v.label]
    if g.status != "eligible"
        empty_stats = fm024_stats_ms(Float64[])
        push!(rows, merge(meta, (; distribution=DISTRIBUTION, precision=string(TF),
            p=P, lh=LH, n=N, ell=ELL, seed=SEED, strategy=v.label,
            status=g.status, note=g.note, correctness_pass=g.correctness_pass,
            direct_scope="all_bodies", oracle_scope="all_routes",
            potential_error=g.potential_error, gradient_error=g.gradient_error,
            coefficient_error=g.coefficient_error, routes=0, direct_pairs=0,
            nonempty_classes=0, mean_occupancy=0.0, max_occupancy=0,
            p50_occupancy=0.0, p90_occupancy=0.0, p95_occupancy=0.0,
            p99_occupancy=0.0, occupancy_skew=0.0,
            construction_ms=NaN, construction_uploads=0, operator_bytes=0,
            metadata_bytes=0, expansion_bytes=0, scratch_bytes=0,
            persistent_bytes=0, peak_bytes=0, launch_count=-1, gemm_count=-1,
            b2m_ms_median=NaN, b2m_ms_min=NaN, b2m_ms_iqr=NaN, b2m_alloc_bytes=0,
            m2m_ms_median=NaN, m2m_ms_min=NaN, m2m_ms_iqr=NaN, m2m_alloc_bytes=0,
            m2l_ms_median=NaN, m2l_ms_min=NaN, m2l_ms_iqr=NaN, m2l_alloc_bytes=0,
            l2l_ms_median=NaN, l2l_ms_min=NaN, l2l_ms_iqr=NaN, l2l_alloc_bytes=0,
            l2b_ms_median=NaN, l2b_ms_min=NaN, l2b_ms_iqr=NaN, l2b_alloc_bytes=0,
            update_ms_median=NaN, update_ms_min=NaN, update_ms_iqr=NaN,
            update_alloc_bytes=0, lifecycle_ms_median=NaN, lifecycle_ms_min=NaN,
            lifecycle_ms_iqr=NaN, lifecycle_alloc_bytes=0,
            finalize_ms_median=NaN, finalize_ms_min=NaN, finalize_ms_iqr=NaN,
            finalize_alloc_bytes=0, recurring_ms_median=NaN,
            recurring_ms_min=NaN, recurring_ms_iqr=NaN, recurring_alloc_bytes=0,
            body_uploads=0, route_uploads=0, operator_uploads=0,
            expansion_host_copies=0, influence_downloads=0,
            concat_scalar_staging_observed=false)))
        continue
    end

    # Gate construction above warmed all specializations.  Measure construction
    # exactly once in this isolated process/case/strategy.
    GC.gc()
    sys = fm024_copy_system(base)
    t0 = time_ns()
    cache = newcache(sys, v)
    construction_ms = (time_ns() - t0) / 1e6
    state = cache.state
    counters0 = deepcopy(state.counters)
    fmm!(sys, cache; scalar_potential=!LH, gradient=true)

    stage_specs = (
        b2m=FastMultipole._launch_host_b2m!,
        m2m=FastMultipole._launch_host_m2m!,
        m2l=FastMultipole._launch_host_m2l!,
        l2l=FastMultipole._launch_host_l2l!,
        l2b=FastMultipole._launch_host_l2b!,
    )
    stage_stats = Dict{Symbol,NamedTuple}()
    stage_alloc = Dict{Symbol,Int}()
    for (name, f) in pairs(stage_specs)
        stage_stats[name] = fm024_samples(() -> f(state))
        stage_alloc[name] = @allocated f(state)
    end
    update_i = Ref(0)
    update_f = () -> begin
        update_i[] += 1
        fm024_jitter!(sys, SEED, update_i[])
        update_radix_state!(cache, (sys,))
    end
    update_stats = fm024_samples(update_f)
    update_alloc = @allocated update_radix_state!(cache, (sys,))
    lifecycle_stats = fm024_samples(() -> run_host_radix_lifecycle!(state))
    lifecycle_alloc = @allocated run_host_radix_lifecycle!(state)
    switches = DerivativesSwitch((!LH,), (true,), (false,), (sys,))
    finalize_f = () -> finalize_radix_output!(state, (sys,);
        derivatives_switches=switches)
    finalize_stats = fm024_samples(finalize_f)
    finalize_alloc = @allocated finalize_f()
    recurring_stats = fm024_samples(() ->
        fmm!(sys, cache; scalar_potential=!LH, gradient=true))
    recurring_alloc = @allocated fmm!(sys, cache;
        scalar_potential=!LH, gradient=true)

    state.counters.route_uploads == counters0.route_uploads ||
        error("host route_uploads changed")
    state.counters.operator_uploads == counters0.operator_uploads ||
        error("host operator_uploads changed")
    state.counters.expansion_host_copies == 0 ||
        error("host expansion copies observed")
    occ = fm024_occupancy(fm024_class_counts(state.scratch.m2l_concat))
    mem = fm024_memory(state)
    c = state.counters
    push!(rows, merge(meta, (; distribution=DISTRIBUTION, precision=string(TF),
        p=P, lh=LH, n=N, ell=ELL, seed=SEED, strategy=v.label,
        status="eligible", note="", correctness_pass=true,
        direct_scope="all_bodies", oracle_scope="all_routes",
        potential_error=g.potential_error, gradient_error=g.gradient_error,
        coefficient_error=g.coefficient_error, routes=state.counts.n_routes,
        direct_pairs=state.counts.n_direct, occ...,
        construction_ms, construction_uploads=0, mem..., launch_count=-1,
        gemm_count=-1,
        b2m_ms_median=stage_stats[:b2m].median,
        b2m_ms_min=stage_stats[:b2m].minimum, b2m_ms_iqr=stage_stats[:b2m].iqr,
        b2m_alloc_bytes=stage_alloc[:b2m],
        m2m_ms_median=stage_stats[:m2m].median,
        m2m_ms_min=stage_stats[:m2m].minimum, m2m_ms_iqr=stage_stats[:m2m].iqr,
        m2m_alloc_bytes=stage_alloc[:m2m],
        m2l_ms_median=stage_stats[:m2l].median,
        m2l_ms_min=stage_stats[:m2l].minimum, m2l_ms_iqr=stage_stats[:m2l].iqr,
        m2l_alloc_bytes=stage_alloc[:m2l],
        l2l_ms_median=stage_stats[:l2l].median,
        l2l_ms_min=stage_stats[:l2l].minimum, l2l_ms_iqr=stage_stats[:l2l].iqr,
        l2l_alloc_bytes=stage_alloc[:l2l],
        l2b_ms_median=stage_stats[:l2b].median,
        l2b_ms_min=stage_stats[:l2b].minimum, l2b_ms_iqr=stage_stats[:l2b].iqr,
        l2b_alloc_bytes=stage_alloc[:l2b],
        update_ms_median=update_stats.median, update_ms_min=update_stats.minimum,
        update_ms_iqr=update_stats.iqr, update_alloc_bytes=update_alloc,
        lifecycle_ms_median=lifecycle_stats.median,
        lifecycle_ms_min=lifecycle_stats.minimum,
        lifecycle_ms_iqr=lifecycle_stats.iqr,
        lifecycle_alloc_bytes=lifecycle_alloc,
        finalize_ms_median=finalize_stats.median,
        finalize_ms_min=finalize_stats.minimum, finalize_ms_iqr=finalize_stats.iqr,
        finalize_alloc_bytes=finalize_alloc,
        recurring_ms_median=recurring_stats.median,
        recurring_ms_min=recurring_stats.minimum,
        recurring_ms_iqr=recurring_stats.iqr,
        recurring_alloc_bytes=recurring_alloc,
        body_uploads=c.body_uploads, route_uploads=c.route_uploads,
        operator_uploads=c.operator_uploads,
        expansion_host_copies=c.expansion_host_copies,
        influence_downloads=c.influence_downloads,
        concat_scalar_staging_observed=v.label == "concat" &&
            stage_alloc[:m2l] > 0)))
end

fm024_write_rows(OUT, rows)
