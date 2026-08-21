using FastMultipole
using Statistics
using Dates
using LinearAlgebra

include(joinpath(@__DIR__, "..", "..", "test", "gravitational.jl"))

const N = parse(Int, get(ENV, "FM023E_N", "2000"))
const P = parse(Int, get(ENV, "FM023E_P", "8"))
const TF = get(ENV, "FM023E_TF", "Float64") == "Float32" ? Float32 : Float64
const LH = lowercase(get(ENV, "FM023E_LH", "false")) in ("1", "true", "yes")
const VARIANT = get(ENV, "FM023E_VARIANT", "dense")
const LABEL = get(ENV, "FM023E_LABEL", "functional_baseline")
const REPS = parse(Int, get(ENV, "FM023E_REPS", "3"))
const BLAS_THREADS = parse(Int, get(ENV, "FM023E_BLAS_THREADS", string(BLAS.get_num_threads())))
const APPLY_CHUNK = parse(Int, get(ENV, "FM023E_APPLY_CHUNK", "0"))
const BUILD_CHUNK = parse(Int, get(ENV, "FM023E_BUILD_CHUNK", "0"))
const MAX_BYTES = parse(Int, get(ENV, "FM023E_MAX_PERSISTENT_BYTES", string(4 << 30)))
const OUT = get(ENV, "FM023E_OUT", joinpath(@__DIR__, "..", "data",
    "dense_translation_m2l_host",
    "$(LABEL)_$(gethostname())_$(Dates.format(now(), "yyyymmdd-HHMMSS"))_$(VARIANT).csv"))
BLAS.set_num_threads(BLAS_THREADS)

readcmd(cmd, fallback="unknown") = try strip(read(cmd, String)) catch; fallback end
const HOST = gethostname()
const KERNEL = readcmd(`uname -sr`)
const CPU_MODEL = readcmd(`sh -c "lscpu 2>/dev/null | sed -n 's/^Model name:[[:space:]]*//p' | head -1"`, Sys.CPU_NAME)
const GIT_COMMIT = get(ENV, "FM023E_GIT_COMMIT", readcmd(`git rev-parse HEAD`))
const GIT_TREE = get(ENV, "FM023E_GIT_TREE", readcmd(`git rev-parse "HEAD^{tree}"`))
const GIT_WORKTREE = get(ENV, "FM023E_GIT_WORKTREE",
    success(pipeline(`git diff --quiet`; stdout=devnull, stderr=devnull)) &&
    success(pipeline(`git diff --cached --quiet`; stdout=devnull, stderr=devnull)) ? "clean" : "dirty")

function variant_options()
    if VARIANT == "dense"
        return MaterializedYRotationM2L(), DenseTranslationM2L(
            max_persistent_bytes=MAX_BYTES, apply_chunk=APPLY_CHUNK,
            build_chunk=BUILD_CHUNK)
    elseif VARIANT == "materialized_concat"
        return MaterializedYRotationM2L(), ConcatenatedFixedZM2L()
    elseif VARIANT == "factored"
        return FactoredRotationM2L(), ConcatenatedFixedZM2L()
    elseif VARIANT == "precomputed_y"
        return FactoredRotationM2L(), PrecomputedFactoredYM2L()
    end
    error("unknown FM023E_VARIANT=$VARIANT")
end

op, strategy = variant_options()
sys = generate_gravitational(23023 + N + P + Int(LH), N)
opts = CUDARadixLifecycleOptions(; precision=TF, operator=op, m2l_strategy=strategy)
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
if plan isa FastMultipole.ResidentM2LDensePlan
    counts = filter(>(0), plan.class_counts)
    operator_bytes = plan.operator_bytes
    scratch_bytes = plan.scratch_bytes
    route_metadata_bytes = plan.route_metadata_bytes
    persistent_bytes = plan.persistent_bytes
    construction_peak_bytes = plan.construction_peak_bytes
    plan_summary_bytes = Base.summarysize(plan)
    effective_apply_width = plan.width
else
    counts = if plan isa FastMultipole.ResidentM2LPrecomputedYPlan
        filter(>(0), plan.offset_counts)
    elseif plan isa FastMultipole.ResidentM2LFactoredPlan
        [g.count[] for g in plan.groups if g.count[] > 0]
    else
        Int[]
    end
    operator_bytes = missing
    scratch_bytes = missing
    route_metadata_bytes = missing
    persistent_bytes = missing
    construction_peak_bytes = missing
    plan_summary_bytes = Base.summarysize(plan)
    effective_apply_width = missing
end

row = (; label=LABEL, host=HOST, kernel=KERNEL, cpu_model=CPU_MODEL,
    julia_version=VERSION, cpu_threads=Sys.CPU_THREADS,
    julia_threads=Threads.nthreads(), blas_vendor=BLAS.vendor(),
    blas_config=BLAS.get_config(), blas_threads=BLAS.get_num_threads(),
    git_commit=GIT_COMMIT, git_tree=GIT_TREE, git_worktree=GIT_WORKTREE,
    variant=VARIANT, precision=TF, lh=LH, n=N, P=P,
    max_persistent_bytes=MAX_BYTES, apply_chunk=APPLY_CHUNK,
    build_chunk=BUILD_CHUNK, effective_apply_width,
    routes=cache.state.counts.n_routes, occupied_classes=length(counts),
    max_class_occupancy=maximum(counts; init=0),
    mean_class_occupancy=isempty(counts) ? 0.0 : mean(counts),
    construction_ms, operator_bytes, scratch_bytes, route_metadata_bytes,
    persistent_bytes, construction_peak_bytes, plan_summary_bytes,
    stage_ms=median(stage_times), stage_alloc_bytes=stage_alloc,
    full_step_ms=median(full_times), full_step_alloc_bytes=full_alloc,
    process_peak_rss_raw=try Sys.maxrss() catch; missing end)

csvfield(x) = begin
    s = string(x)
    occursin(r"[\",\n\r]", s) ? '"' * replace(s, '"' => "\"\"") * '"' : s
end
mkpath(dirname(OUT))
open(OUT, "w") do io
    names = propertynames(row)
    println(io, join(names, ','))
    println(io, join((csvfield(getproperty(row, name)) for name in names), ','))
end
println(OUT)
