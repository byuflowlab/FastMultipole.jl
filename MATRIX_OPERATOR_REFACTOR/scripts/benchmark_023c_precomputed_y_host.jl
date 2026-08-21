using FastMultipole
using Statistics
using Dates
using LinearAlgebra

include(joinpath(@__DIR__, "..", "..", "test", "gravitational.jl"))

const NS = parse.(Int, split(get(ENV, "FM023C_N", "150,2000,20000"), ','))
const PS = parse.(Int, split(get(ENV, "FM023C_P", "4,8,12"), ','))
const REPS = parse(Int, get(ENV, "FM023C_REPS", "3"))
const BLAS_THREADS = parse(Int, get(ENV, "FM023C_BLAS_THREADS", string(BLAS.get_num_threads())))
const SWEEP_COLS = [parse(Int, s) for s in
    split(get(ENV, "FM023C_SWEEP_COLS", "4,8,12,16,24"), ',') if !isempty(s)]
const OUT = get(ENV, "FM023C_OUT", joinpath(@__DIR__, "..", "data",
    "precomputed_y_resident_m2l_host",
    "host_$(gethostname())_$(Dates.format(now(), "yyyymmdd-HHMMSS")).csv"))
BLAS.set_num_threads(BLAS_THREADS)

readcmd(cmd, fallback="unknown") = try
    strip(read(cmd, String))
catch
    fallback
end

const HOST = gethostname()
const KERNEL = readcmd(`uname -sr`)
const CPU_MODEL = readcmd(`sh -c "lscpu 2>/dev/null | sed -n 's/^Model name:[[:space:]]*//p' | head -1"`,
    Sys.CPU_NAME)
const GIT_COMMIT = haskey(ENV, "FM023C_GIT_COMMIT") ?
    ENV["FM023C_GIT_COMMIT"] : readcmd(`git rev-parse HEAD`)
const GIT_WORKTREE = if haskey(ENV, "FM023C_GIT_WORKTREE")
    ENV["FM023C_GIT_WORKTREE"]
else
    success(pipeline(`git diff --quiet`; stdout=devnull, stderr=devnull)) &&
        success(pipeline(`git diff --cached --quiet`; stdout=devnull, stderr=devnull)) ?
        "clean" : "dirty"
end
const PRODUCTION_THRESHOLD = FastMultipole.PRECOMPUTED_Y_GEMM_MIN_COLS[]

csvfield(x) = begin
    s = string(x)
    occursin(r"[\",\n\r]", s) ? '"' * replace(s, '"' => "\"\"") * '"' : s
end

function warm_specializations(P, LH, strategy)
    sys = generate_gravitational(923023 + P + Int(LH), 32)
    opts = CUDARadixLifecycleOptions(; operator=FactoredRotationM2L(),
        m2l_strategy=strategy)
    cache = RadixFMMCache(sys; expansion_order=P, ell=3, max_n_bodies=32,
        lamb_helmholtz=LH, options=opts)
    fmm!(sys, cache; scalar_potential=!LH, gradient=true)
    FastMultipole._launch_resident_m2l!(cache.state)
    return nothing
end

function measure(P, LH, N, variant; candidate_threshold=nothing)
    saved = FastMultipole.PRECOMPUTED_Y_GEMM_MIN_COLS[]
    try
        if variant == "precomputed_scalar"
            strategy = PrecomputedFactoredYM2L(); threshold = typemax(Int)
        elseif variant == "precomputed_gemm"
            strategy = PrecomputedFactoredYM2L(); threshold = 1
        elseif variant == "precomputed_production"
            strategy = PrecomputedFactoredYM2L(); threshold = saved
        elseif variant == "precomputed_candidate"
            strategy = PrecomputedFactoredYM2L(); threshold = something(candidate_threshold)
        elseif variant == "factored_023a"
            strategy = ConcatenatedFixedZM2L(); threshold = saved
        else
            error("unknown variant $variant")
        end
        FastMultipole.PRECOMPUTED_Y_GEMM_MIN_COLS[] = threshold
        sys = generate_gravitational(23023 + P + Int(LH), N)
        opts = CUDARadixLifecycleOptions(; operator=FactoredRotationM2L(),
            m2l_strategy=strategy)
        t0 = time_ns()
        cache = RadixFMMCache(sys; expansion_order=P, ell=3, max_n_bodies=N,
            lamb_helmholtz=LH, options=opts)
        construction_ms = (time_ns() - t0) / 1e6
        fmm!(sys, cache; scalar_potential=!LH, gradient=true)
        FastMultipole._launch_resident_m2l!(cache.state)
        stage_alloc = @allocated FastMultipole._launch_resident_m2l!(cache.state)
        stage_times = [1e3 * @elapsed(FastMultipole._launch_resident_m2l!(cache.state)) for _ in 1:REPS]
        full_alloc = @allocated fmm!(sys, cache; scalar_potential=!LH, gradient=true)
        full_times = [1e3 * @elapsed(fmm!(sys, cache; scalar_potential=!LH, gradient=true)) for _ in 1:REPS]
        plan = cache.state.scratch.m2l_concat
        if plan isa FastMultipole.ResidentM2LPrecomputedYPlan
            ac = filter(>(0), plan.angle_counts)
            oc = filter(>(0), plan.offset_counts)
            operator_bytes = Base.summarysize((plan.y_mult, plan.y_loc, plan.z_phi,
                plan.z_chi, plan.lh_phi_rows, plan.lh_chi_rows))
            scratch_bytes = Base.summarysize((plan.packed_sources, plan.packed_targets,
                plan.packed_phis, plan.scratch))
        else
            oc = [g.count[] for g in plan.groups if g.count[] > 0]
            ac = oc
            operator_bytes = Base.summarysize((plan.ym_flat, plan.z_flat, plan.groups))
            scratch_bytes = Base.summarysize((plan.route_class, cache.state.scratch.aphi,
                cache.state.scratch.achi))
        end
        return (host=HOST, kernel=KERNEL, cpu_model=CPU_MODEL,
            julia_version=VERSION, cpu_threads=Sys.CPU_THREADS,
            julia_threads=Threads.nthreads(), blas_vendor=BLAS.vendor(),
            blas_config=BLAS.get_config(), blas_threads=BLAS.get_num_threads(),
            git_commit=GIT_COMMIT, git_worktree=GIT_WORKTREE,
            P=P, lh=LH, n=N, variant=variant,
            routes=cache.state.counts.n_routes, occupied_angles=length(ac),
            max_angle_occupancy=maximum(ac; init=0), mean_angle_occupancy=isempty(ac) ? 0.0 : mean(ac),
            occupied_offsets=length(oc), max_offset_occupancy=maximum(oc; init=0),
            construction_ms=construction_ms, operator_bytes=operator_bytes,
            scratch_bytes=scratch_bytes, stage_ms=median(stage_times),
            stage_alloc_bytes=stage_alloc, full_step_ms=median(full_times),
            full_step_alloc_bytes=full_alloc, precomputed_gemm_min_cols=threshold,
            production_gemm_min_cols=PRODUCTION_THRESHOLD)
    finally
        FastMultipole.PRECOMPUTED_Y_GEMM_MIN_COLS[] = saved
    end
end

rows = NamedTuple[]
for P in PS, LH in (false, true)
    warm_specializations(P, LH, ConcatenatedFixedZM2L())
    warm_specializations(P, LH, PrecomputedFactoredYM2L())
end
for N in NS, P in PS, LH in (false, true)
    for variant in ("factored_023a", "precomputed_scalar", "precomputed_gemm",
                    "precomputed_production")
        push!(rows, measure(P, LH, N, variant))
    end
    for threshold in SWEEP_COLS
        threshold == PRODUCTION_THRESHOLD && continue
        push!(rows, measure(P, LH, N, "precomputed_candidate";
            candidate_threshold=threshold))
    end
    GC.gc()
end
mkpath(dirname(OUT))
open(OUT, "w") do io
    names = propertynames(first(rows))
    println(io, join(names, ','))
    for row in rows
        println(io, join((csvfield(getproperty(row, name)) for name in names), ','))
    end
end
println(OUT)
