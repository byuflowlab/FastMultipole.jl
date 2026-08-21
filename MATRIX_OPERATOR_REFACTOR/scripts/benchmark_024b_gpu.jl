# Task 024b: one resident-radix GPU scaling case per Julia process.
# Set FM024B_DEVICE=false for local host-only plumbing smoke tests.

using FastMultipole
using FastMultipole.StaticArrays
using Dates
using Sockets
using Statistics

const REPO = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(REPO, "test", "gravitational.jl"))
include(joinpath(@__DIR__, "benchmark_024b_common.jl"))

envbool(name, default) =
    lowercase(get(ENV, name, string(default))) in ("1", "true", "yes", "on")

const N = parse(Int, get(ENV, "FM024B_N", "2000"))
const ELL = parse(Int, get(ENV, "FM024B_ELL", "3"))
const SEED = parse(Int, get(ENV, "FM024B_SEED", "24025"))
const WARMUPS = parse(Int, get(ENV, "FM024B_WARMUPS", "2"))
const SAMPLES = parse(Int, get(ENV, "FM024B_SAMPLES", "7"))
const DEVICE = envbool("FM024B_DEVICE", true)
const TF = let name = get(ENV, "FM024B_TF", "Float64")
    name == "Float64" ? Float64 : name == "Float32" ? Float32 :
        error("FM024B_TF must be Float64 or Float32")
end
const REQUESTED_STRATEGY = get(ENV, "FM024B_STRATEGY", "dense")
const EXPANSION_ORDER = 3
const P_LITERATURE = 4
const STENCIL_EPSILON = 0.19542385331034917 * 2.0^(ELL - 4)
const OUT = get(ENV, "FM024B_OUT",
    joinpath(REPO, "MATRIX_OPERATOR_REFACTOR", "data", "cpu_gpu_scaling",
        "gpu_$(gethostname()).csv"))
const REFERENCE_DIR = get(ENV, "FM024B_REFERENCE_DIR",
    joinpath(REPO, "MATRIX_OPERATOR_REFACTOR", "data", "cpu_gpu_scaling",
        "references"))

SAMPLES >= 5 || error("FM024B_SAMPLES must be at least 5")
WARMUPS >= 2 || error("FM024B_WARMUPS must be at least 2")
ELL <= FastMultipole.RADIX_GRID_MAX_ELL ||
    error("FM024B_ELL=$ELL exceeds RADIX_GRID_MAX_ELL=$(FastMultipole.RADIX_GRID_MAX_ELL)")

if DEVICE
    @eval using CUDA
    FastMultipole.load_cuda_radix_lifecycle!() ||
        error("failed to load CUDA radix lifecycle: $(FastMultipole.cuda_radix_status())")
end

function is_oom(err)
    text = lowercase(sprint(showerror, err))
    return occursin("out of memory", text) || occursin("outofmemory", text) ||
           occursin("cuda_error_out_of_memory", text) ||
           occursin("max_persistent_bytes", text) ||
           occursin("device persistent payload exceeds", text) ||
           occursin("estimated device peak", text)
end

function strategy_options(label)
    if label == "dense"
        return CUDARadixLifecycleOptions(; precision=TF,
            operator=MaterializedYRotationM2L(),
            m2l_strategy=DenseTranslationM2L())
    elseif label == "precomputed_y"
        return CUDARadixLifecycleOptions(; precision=TF,
            operator=FactoredRotationM2L(),
            m2l_strategy=PrecomputedFactoredYM2L())
    end
    error("FM024B_STRATEGY must be dense or precomputed_y")
end

function build_cache(sys, label)
    options = strategy_options(label)
    started = time_ns()
    cache = RadixFMMCache(sys; expansion_order=EXPANSION_ORDER, ell=ELL,
        max_n_bodies=N, bounds=(SVector(-0.01, -0.01, -0.01), 1.02),
        device=DEVICE, options, stencil_epsilon=STENCIL_EPSILON)
    DEVICE && CUDA.synchronize()
    return cache, (time_ns() - started) / 1e9
end

function main()
    reference_path = fm024b_reference_path(REFERENCE_DIR, N)
    reference = fm024b_read_reference(reference_path, N)
    sys = generate_gravitational(SEED, N)
    strategy = REQUESTED_STRATEGY
    oom_fallback = false
    cache = nothing
    construct_seconds = NaN

    try
        cache, construct_seconds = build_cache(sys, strategy)
    catch err
        if DEVICE && strategy == "dense" && is_oom(err)
            @warn "dense construction OOM; falling back to precomputed_y" n=N ell=ELL
            CUDA.reclaim()
            strategy = "precomputed_y"
            oom_fallback = true
            cache, construct_seconds = build_cache(sys, strategy)
        else
            rethrow()
        end
    end

    run_step(; scalar_potential=false) =
        fmm!(sys, cache; gradient=true, scalar_potential)
    sync_device() = DEVICE ? CUDA.synchronize() : nothing
    for _ in 1:WARMUPS
        run_step()
        sync_device()
    end
    samples = Float64[]
    for _ in 1:SAMPLES
        started = time_ns()
        run_step()
        sync_device()
        push!(samples, (time_ns() - started) / 1e9)
    end

    run_step(; scalar_potential=true)
    sync_device()
    metrics = fm024b_accuracy_metrics(sys, reference)
    all(isfinite, metrics) || error("024b GPU accuracy metrics are not finite")

    header = ("mode", "tf", "n", "ell", "strategy", "expansion_order",
        "P_literature", "stencil_epsilon", "construct_seconds",
        "step_seconds_min", "step_seconds_median", "err_potential_abs_rms",
        "err_potential_rel_rms", "err_gradient_abs_rms",
        "err_gradient_rel_rms", "err_gradient_max", "reference_samples",
        "reference_checksum", "oom_fallback", "host", "timestamp")
    row = ("gpu", string(TF), N, ELL, strategy, EXPANSION_ORDER, P_LITERATURE,
        STENCIL_EPSILON, construct_seconds, minimum(samples), median(samples),
        metrics.potential_abs_rms, metrics.potential_rel_rms,
        metrics.gradient_abs_rms, metrics.gradient_rel_rms,
        metrics.gradient_max, reference.samples, reference.checksum,
        oom_fallback, gethostname(),
        Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SSZ"))
    fm024b_append_row(OUT, header, row)
    println("024b $(DEVICE ? "GPU" : "host smoke") wrote $OUT: tf=$TF n=$N " *
        "ell=$ELL strategy=$strategy min=$(minimum(samples)) s " *
        "grad_rel=$(metrics.gradient_rel_rms)")
end

main()
