# Task 024b: one fixed-MAC legacy-octree CPU case per Julia process.

using FastMultipole
using Dates
using Sockets
using Statistics

const REPO = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(REPO, "test", "gravitational.jl"))
include(joinpath(@__DIR__, "benchmark_024b_common.jl"))

const N = parse(Int, get(ENV, "FM024B_N", "2000"))
const SEED = parse(Int, get(ENV, "FM024B_SEED", "24025"))
const FINAL_WARMUPS = parse(Int, get(ENV, "FM024B_WARMUPS", "2"))
const FINAL_SAMPLES = parse(Int, get(ENV, "FM024B_SAMPLES", "7"))
const EXPANSION_ORDER = 3
const P_LITERATURE = 4
const MULTIPOLE_ACCEPTANCE = 0.5
const OUT = get(ENV, "FM024B_OUT",
    joinpath(REPO, "MATRIX_OPERATOR_REFACTOR", "data", "cpu_gpu_scaling",
        "cpu_$(gethostname()).csv"))
const REFERENCE_DIR = get(ENV, "FM024B_REFERENCE_DIR",
    joinpath(REPO, "MATRIX_OPERATOR_REFACTOR", "data", "cpu_gpu_scaling",
        "references"))
const AUDIT_HEADER = ("phase", "attempt", "n", "threads", "leaf_size", "warmups",
    "samples", "step_seconds_min", "step_seconds_median", "host", "timestamp")

FINAL_WARMUPS >= 2 || error("FM024B_WARMUPS must be at least 2")
FINAL_SAMPLES >= 7 || error("FM024B_SAMPLES must be at least 7")
Threads.nthreads() in (1, 64) ||
    @warn "024b production modes are 1 and 64 Julia threads" threads=Threads.nthreads()

function leaf_candidates()
    if haskey(ENV, "FM024B_LEAF_CANDIDATES")
        values = parse.(Int, split(ENV["FM024B_LEAF_CANDIDATES"], ','))
    else
        values = [5, 10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120, N]
    end
    return unique(sort(clamp.(values, 1, N)))
end

function timing_cache(sys)
    switch = FastMultipole.DerivativesSwitch(false, true, false, (sys,))
    return FastMultipole.Cache((sys,), (sys,), switch)
end

function existing_audit_stats(phase)
    stats = Dict{Int,NamedTuple}()
    dir = dirname(OUT)
    isdir(dir) || return stats
    suffix = "_t$(Threads.nthreads())_n$(N).csv"
    paths = filter(path -> begin
        name = basename(path)
        startswith(name, "cpu_leaf_search_") && endswith(name, suffix)
    end, readdir(dir; join=true))
    for path in paths
        lines = readlines(path)
        length(lines) >= 2 || continue
        header = split(lines[1], ',')
        columns = Dict(name => i for (i, name) in enumerate(header))
        all(haskey(columns, name) for name in
            ("phase", "n", "threads", "leaf_size",
             "step_seconds_min", "step_seconds_median")) || continue
        for line in @view lines[2:end]
            fields = split(line, ',')
            length(fields) == length(header) || continue
            fields[columns["phase"]] == phase || continue
            tryparse(Int, fields[columns["n"]]) == N || continue
            tryparse(Int, fields[columns["threads"]]) == Threads.nthreads() || continue
            leaf = tryparse(Int, fields[columns["leaf_size"]])
            minimum = tryparse(Float64, fields[columns["step_seconds_min"]])
            median = tryparse(Float64, fields[columns["step_seconds_median"]])
            (isnothing(leaf) || isnothing(minimum) || isnothing(median)) && continue
            isfinite(minimum) && minimum > 0 && isfinite(median) && median > 0 ||
                continue
            record = (; minimum, median)
            (!haskey(stats, leaf) || median < stats[leaf].median) &&
                (stats[leaf] = record)
        end
    end
    return stats
end

function accuracy_cache(sys)
    switch = FastMultipole.DerivativesSwitch(true, true, false, (sys,))
    return FastMultipole.Cache((sys,), (sys,), switch)
end

function run_step!(sys, cache, leaf; scalar_potential=false)
    return fmm!(sys, cache; leaf_size=leaf,
        multipole_acceptance=MULTIPOLE_ACCEPTANCE,
        expansion_order=EXPANSION_ORDER, error_tolerance=nothing, tune=false,
        gradient=true, scalar_potential)
end

function measure_leaf!(audit_rows, sys, cache, leaf, phase, attempt;
        warmups, samples)
    for _ in 1:warmups
        run_step!(sys, cache, leaf)
    end
    times = [@elapsed(run_step!(sys, cache, leaf)) for _ in 1:samples]
    push!(audit_rows, (phase, attempt, N, Threads.nthreads(), leaf, warmups,
        samples, minimum(times), median(times), gethostname(),
        Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SSZ")))
    fm024b_append_row(audit_path(), AUDIT_HEADER, audit_rows[end])
    return (; minimum=minimum(times), median=median(times))
end

function search_leaf!(sys, cache)
    started = time_ns()
    audit_rows = Tuple[]
    coarse = leaf_candidates()
    coarse_stats = existing_audit_stats("coarse")
    for leaf in sort!(collect(keys(coarse_stats)))
        println("024b CPU resume: reuse coarse n=$N threads=$(Threads.nthreads()) leaf=$leaf")
    end
    attempt = 1

    while true
        for leaf in coarse
            haskey(coarse_stats, leaf) && continue
            coarse_stats[leaf] = measure_leaf!(audit_rows, sys, cache, leaf,
                "coarse", attempt; warmups=1, samples=3)
        end
        ordered = sort(collect(keys(coarse_stats)))
        best = argmin(leaf -> coarse_stats[leaf].median, ordered)
        best_i = findfirst(==(best), ordered)
        if (best_i > 1 && best_i < length(ordered)) || best in (1, N)
            coarse = ordered
            break
        elseif best_i == 1
            new_leaf = max(1, best ÷ 2)
            new_leaf == best && (new_leaf = 1)
            push!(coarse, new_leaf)
        else
            new_leaf = min(N, max(best + 1, 2best))
            push!(coarse, new_leaf)
        end
        coarse = unique(sort(coarse))
        attempt += 1
        attempt <= 32 || error("failed to bracket coarse leaf-size optimum")
    end

    coarse_best = argmin(leaf -> coarse_stats[leaf].median, coarse)
    coarse_i = findfirst(==(coarse_best), coarse)
    lo_i = coarse_best == 1 ? coarse_i : coarse_i - 1
    hi_i = coarse_best == N ? coarse_i : coarse_i + 1
    lo = coarse[lo_i]
    hi = coarse[hi_i]
    refine_stats = existing_audit_stats("refine")
    for leaf in sort!(collect(keys(refine_stats)))
        println("024b CPU resume: reuse refine n=$N threads=$(Threads.nthreads()) leaf=$leaf")
    end
    refine_attempt = 1

    while true
        candidates = unique(sort(vcat(lo:5:hi, lo, hi)))
        for leaf in candidates
            haskey(refine_stats, leaf) && continue
            refine_stats[leaf] = measure_leaf!(audit_rows, sys, cache, leaf,
                "refine", refine_attempt; warmups=1, samples=5)
        end
        tested = sort(collect(keys(refine_stats)))
        eligible = filter(leaf -> lo <= leaf <= hi, tested)
        winner = argmin(leaf -> refine_stats[leaf].median, eligible)
        if winner != lo && winner != hi || winner in (1, N)
            leaf_search_seconds = (time_ns() - started) / 1e9
            return winner, leaf_search_seconds, audit_rows
        elseif winner == lo
            coarse_pos = searchsortedfirst(coarse, lo)
            coarse_pos > 1 || begin
                lo = 1
                continue
            end
            lo = coarse[coarse_pos - 1]
        else
            coarse_pos = searchsortedfirst(coarse, hi)
            coarse_pos < length(coarse) || begin
                hi = N
                continue
            end
            hi = coarse[coarse_pos + 1]
        end
        refine_attempt += 1
        refine_attempt <= length(coarse) + 4 ||
            error("failed to bracket refined leaf-size optimum")
    end
end

function audit_path()
    return get(ENV, "FM024B_LEAF_OUT",
        joinpath(dirname(OUT), "cpu_leaf_search_$(gethostname())_t$(Threads.nthreads())_n$(N).csv"))
end

function main()
    reference_path = fm024b_reference_path(REFERENCE_DIR, N)
    reference = fm024b_read_reference(reference_path, N)
    sys = generate_gravitational(SEED, N)
    cache = timing_cache(sys)

    leaf, leaf_search_seconds, audit_rows = search_leaf!(sys, cache)
    isempty(audit_rows) &&
        @warn "024b CPU leaf search reused its complete persisted audit"

    for _ in 1:FINAL_WARMUPS
        run_step!(sys, cache, leaf)
    end
    samples = [@elapsed(run_step!(sys, cache, leaf)) for _ in 1:FINAL_SAMPLES]

    acache = accuracy_cache(sys)
    run_step!(sys, acache, leaf; scalar_potential=true)
    metrics = fm024b_accuracy_metrics(sys, reference)
    all(isfinite, metrics) || error("024b CPU accuracy metrics are not finite")

    threads = Threads.nthreads()
    mode = threads == 1 ? "cpu1" : threads == 64 ? "cpu64" : "cpu$(threads)"
    header = ("mode", "n", "threads", "leaf_size", "multipole_acceptance",
        "expansion_order", "P_literature", "leaf_search_seconds",
        "step_seconds_min", "step_seconds_median", "err_potential_abs_rms",
        "err_potential_rel_rms", "err_gradient_abs_rms",
        "err_gradient_rel_rms", "err_gradient_max", "reference_samples",
        "reference_checksum", "host", "timestamp")
    row = (mode, N, threads, leaf, MULTIPOLE_ACCEPTANCE, EXPANSION_ORDER,
        P_LITERATURE, leaf_search_seconds, minimum(samples), median(samples),
        metrics.potential_abs_rms, metrics.potential_rel_rms,
        metrics.gradient_abs_rms, metrics.gradient_rel_rms,
        metrics.gradient_max, reference.samples, reference.checksum, gethostname(),
        Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SSZ"))
    fm024b_append_row(OUT, header, row)
    println("024b CPU wrote $OUT: mode=$mode n=$N leaf=$leaf " *
        "min=$(minimum(samples)) s grad_rel=$(metrics.gradient_rel_rms)")
end

main()
