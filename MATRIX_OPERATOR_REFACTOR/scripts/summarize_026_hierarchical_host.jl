using Statistics

const ROOT026 = normpath(joinpath(@__DIR__, "..", "data",
    "hierarchical_m2l_host"))
const RAW026 = length(ARGS) >= 1 ? abspath(ARGS[1]) : joinpath(ROOT026, "raw")
const SUMMARY026 = length(ARGS) >= 2 ? abspath(ARGS[2]) :
    joinpath(ROOT026, "summary")

function read026(path)
    lines = readlines(path)
    isempty(lines) && error("empty task-026 CSV: $path")
    names = split(lines[1], ','; keepempty=true)
    return [Dict(names .=> split(line, ','; keepempty=true)) for line in lines[2:end]
            if !isempty(strip(line))]
end

function write026(path, rows, names)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, join(names, ','))
        for row in rows
            println(io, join((get(row, name, "") for name in names), ','))
        end
    end
end

num(row, key) = parse(Float64, row[key])
intnum(row, key) = parse(Int, row[key])
iszero_counters(row) = intnum(row, "expansion_host_copies") == 0 &&
    intnum(row, "route_uploads") == 0 && intnum(row, "operator_uploads") == 0
eligible(row) = iszero_counters(row) &&
    (!startswith(row["policy"], "hierarchical") ||
     (intnum(row, "m2l_alloc_bytes") <= 64 << 10 &&
      intnum(row, "step_alloc_bytes") < 512 << 10))

case_paths = filter(path -> endswith(path, ".csv") &&
    !endswith(path, "_levels.csv"), readdir(RAW026; join=true))
level_paths = filter(path -> endswith(path, "_levels.csv"),
    readdir(RAW026; join=true))
isempty(case_paths) && error("no task-026 raw case CSVs in $RAW026")
cases = reduce(vcat, read026.(case_paths))
levels = isempty(level_paths) ? Dict[] : reduce(vcat, read026.(level_paths))
case_id(row) = join((row[key] for key in ("blas_threads", "policy", "strategy",
    "n", "ell", "window_classes", "precision", "lh", "distribution",
    "accuracy_samples")), '|')
length(unique(case_id.(cases))) == length(cases) ||
    error("task-026 raw directory contains duplicate cases")

case_names = collect(keys(first(cases)))
# Preserve the driver's schema order rather than Dict iteration order.
case_names = split(first(readlines(first(case_paths))), ',')
level_names = isempty(levels) ? String[] :
    split(first(readlines(first(level_paths))), ',')
write026(joinpath(SUMMARY026, "cases.csv"), cases, case_names)
!isempty(levels) && write026(joinpath(SUMMARY026, "levels.csv"), levels, level_names)

window_rows = filter(cases) do row
    startswith(row["policy"], "hierarchical") &&
    row["distribution"] == "uniform" && row["precision"] == "Float64" &&
    row["lh"] == "false" && intnum(row, "n") == 20_000 &&
    intnum(row, "ell") == 4
end
scores = Dict{Int,Float64}()
best_score = NaN
selected = if isempty(window_rows)
    haskey(ENV, "FM026_SELECTED_WINDOW") ||
        error("FM026_SELECTED_WINDOW is required when summarizing a phase without tuning rows")
    parse(Int, ENV["FM026_SELECTED_WINDOW"])
else
    common_windows = sort!(collect(intersect([
        Set(intnum(row, "window_classes") for row in window_rows
            if row["policy"] == policy) for policy in
            ("hierarchical_12", "hierarchical_3")]...)))
    # Completeness is judged against the matrix actually swept (a strategy-
    # filtered re-measurement has fewer than the full campaign's 16 keys).
    nexpect = length(Set((row["policy"], row["strategy"], row["blas_threads"])
        for row in window_rows))
    for width in common_windows
        group = filter(row -> intnum(row, "window_classes") == width, window_rows)
        keys_seen = Set((row["policy"], row["strategy"], row["blas_threads"])
            for row in group if eligible(row))
        length(keys_seen) == nexpect || continue
        scores[width] = exp(mean(log(num(row, "m2l_ms_median"))
            for row in group if eligible(row)))
    end
    isempty(scores) && error("no window width passed every task-026 gate")
    best_score = minimum(values(scores))
    minimum(width for (width, score) in scores
        if score <= 1.02 * best_score)
end
mkpath(SUMMARY026)
write(joinpath(SUMMARY026, "selected_window.txt"), string(selected, '\n'))
window_names = vcat(case_names, ["eligible", "geomean_ms", "relative_to_best"])
window_output = Dict[]
for row in window_rows
    width = intnum(row, "window_classes")
    extra = Dict{String,String}("eligible" => string(eligible(row)),
        "geomean_ms" => string(get(scores, width, NaN)),
        "relative_to_best" => string(get(scores, width, NaN) / best_score))
    push!(window_output, merge(Dict{String,String}(row), extra))
end
write026(joinpath(SUMMARY026, "window_tuning.csv"), window_output, window_names)

accuracy = filter(row -> intnum(row, "accuracy_samples") > 0, cases)
write026(joinpath(SUMMARY026, "accuracy.csv"), accuracy, case_names)

scaling_ns = Dict(3 => Set((64, 128, 256, 512, 1024, 2048)),
    4 => Set((128, 256, 512, 1024, 2048, 4096, 8192)),
    5 => Set((256, 512, 1024, 2048, 4096)))
scaling_base = filter(cases) do row
    ell = intnum(row, "ell")
    row["distribution"] == "uniform" && row["precision"] == "Float64" &&
    row["lh"] == "false" && intnum(row, "window_classes") == selected &&
    intnum(row, "accuracy_samples") == 0 &&
    haskey(scaling_ns, ell) && intnum(row, "n") in scaling_ns[ell]
end
flat_key(row) = (row["strategy"], row["blas_threads"], row["ell"], row["n"])
flat_times = Dict(flat_key(row) => num(row, "m2l_ms_median") for row in scaling_base
    if row["policy"] == "flat")
flat_routes = Dict(flat_key(row) => num(row, "routes") for row in scaling_base
    if row["policy"] == "flat")
scaling = Dict[]
for row in scaling_base
    key = flat_key(row)
    c = num(row, "cells")
    extra = Dict{String,String}(
        "route_ratio_c2" => string(num(row, "routes") / max(c * c, 1)),
        "m2l_ratio_flat" => string(num(row, "m2l_ms_median") /
            get(flat_times, key, NaN)),
        "route_ratio_flat" => string(num(row, "routes") /
            get(flat_routes, key, NaN)))
    push!(scaling, merge(Dict{String,String}(row), extra))
end
scaling_names = vcat(case_names,
    ["route_ratio_c2", "m2l_ratio_flat", "route_ratio_flat"])
write026(joinpath(SUMMARY026, "scaling.csv"), scaling, scaling_names)

function slope026(xs, ys)
    length(xs) >= 3 || return NaN
    lx, ly = log.(Float64.(xs)), log.(Float64.(ys))
    denom = sum(abs2, lx .- mean(lx))
    denom == 0 && return NaN
    return sum((lx .- mean(lx)) .* (ly .- mean(ly))) / denom
end

scaling_summary = Dict[]
for policy in ("flat", "hierarchical_12", "hierarchical_3"),
        strategy in ("concat", "factored", "precomputed_y", "dense"),
        bt in ("1", "64"), ell in ("3", "4", "5")
    group = sort(filter(row -> row["policy"] == policy &&
        row["strategy"] == strategy && row["blas_threads"] == bt &&
        row["ell"] == ell, scaling_base); by=row -> intnum(row, "n"))
    isempty(group) && continue
    # Exclude points at more than 90% of the level's maximum cell count.
    cap = (1 << parse(Int, ell))^3
    fit = filter(row -> intnum(row, "cells") < 0.9cap, group)
    n_min = isempty(fit) ? 0 : minimum(intnum(row, "n") for row in fit)
    n_max = isempty(fit) ? 0 : maximum(intnum(row, "n") for row in fit)
    push!(scaling_summary, Dict(
        "policy" => policy, "strategy" => strategy, "blas_threads" => bt,
        "ell" => ell, "npoints" => string(length(fit)),
        "n_min" => string(n_min), "n_max" => string(n_max),
        "route_exponent" => string(slope026(
            [intnum(row, "n") for row in fit],
            [num(row, "routes") for row in fit])),
        "m2l_exponent" => string(slope026(
            [intnum(row, "n") for row in fit],
            [num(row, "m2l_ms_median") for row in fit]))))
end
write026(joinpath(SUMMARY026, "scaling_exponents.csv"), scaling_summary,
    ["policy", "strategy", "blas_threads", "ell", "npoints", "n_min",
     "n_max", "route_exponent", "m2l_exponent"])

crossover_rows = Dict[]
for policy in ("hierarchical_12", "hierarchical_3"),
        strategy in ("concat", "factored", "precomputed_y", "dense"),
        bt in ("1", "64"), ell in ("3", "4", "5")
    group = sort(filter(row -> row["policy"] == policy &&
        row["strategy"] == strategy && row["blas_threads"] == bt &&
        row["ell"] == ell, scaling); by=row -> intnum(row, "n"))
    isempty(group) && continue
    winner = 0
    for i in 1:max(length(group) - 1, 0)
        if num(group[i], "m2l_ratio_flat") < 1 &&
                num(group[i + 1], "m2l_ratio_flat") < 1
            winner = i
            break
        end
    end
    verdict = winner == 0 ? "not_observed" : "observed"
    nlo = winner <= 1 ? 0 : intnum(group[winner - 1], "n")
    nhi = winner == 0 ? 0 : intnum(group[winner], "n")
    push!(crossover_rows, Dict("policy" => policy, "strategy" => strategy,
        "blas_threads" => bt, "ell" => ell, "verdict" => verdict,
        "n_lower" => string(nlo), "n_upper" => string(nhi)))
end
write026(joinpath(SUMMARY026, "crossovers.csv"), crossover_rows,
    ["policy", "strategy", "blas_threads", "ell", "verdict", "n_lower", "n_upper"])

source_hashes = unique(row["source_hash"] for row in cases)
tuning_hashes = unique(row["source_hash"] for row in window_rows)
final_hashes = unique(row["source_hash"] for row in cases if !(row in window_rows))
required_gate_rows = filter(cases) do row
    !(row in window_rows) ||
        intnum(row, "window_classes") == selected
end
bad_gates = count(row -> !eligible(row), required_gate_rows)
bad_accuracy = count(accuracy) do row
    tf32 = row["precision"] == "Float32"
    ptol, gtol = tf32 ? (1e-3, 1e-1) : (1e-6, 1e-4)
    # Radius 3 is the deliberately less-accurate classic FMM policy. Record
    # its accuracy cost and apply only a 20x sanity bound; radius 12 remains
    # subject to the standard host correctness tolerance.
    endswith(row["policy"], "_3") && ((ptol, gtol) = (20ptol, 20gtol))
    (!isnan(num(row, "potential_error")) &&
        num(row, "potential_error") >= ptol) ||
        num(row, "gradient_error") >= gtol
end
open(joinpath(SUMMARY026, "validation_summary.md"), "w") do io
    println(io, "# Task 026 Campaign Validation")
    println(io)
    println(io, "- Raw cases: ", length(cases))
    println(io, "- Level rows: ", length(levels))
    println(io, "- Source hashes: `", join(source_hashes, "`, `"), "`")
    println(io, "- Selected window: ", selected)
    println(io, "- Allocation/counter gate failures: ", bad_gates)
    flat_overruns = count(cases) do row
        !startswith(row["policy"], "hierarchical") &&
        (intnum(row, "m2l_alloc_bytes") > 64 << 10 ||
         intnum(row, "step_alloc_bytes") >= 512 << 10)
    end
    println(io, "- Recorded flat-oracle allocation overruns: ", flat_overruns)
    println(io, "- Accuracy gate failures: ", bad_accuracy)
end

length(tuning_hashes) <= 1 ||
    error("task-026 tuning phase mixed source hashes")
length(final_hashes) <= 1 ||
    error("task-026 scaling/accuracy phases mixed source hashes")
bad_gates == 0 || error("$bad_gates task-026 cases failed allocation/counter gates")
bad_accuracy == 0 || error("$bad_accuracy task-026 cases failed accuracy gates")
expected_phase = get(ENV, "FM026_EXPECTED_PHASE", "none")
if expected_phase in ("tuning", "all")
    length(window_rows) == 128 ||
        error("expected 128 window-tuning cases, found $(length(window_rows))")
end
if expected_phase == "all"
    length(cases) == 776 ||
        error("expected 776 total task-026 cases, found $(length(cases))")
    length(accuracy) == 192 ||
        error("expected 192 accuracy cases, found $(length(accuracy))")
    length(scaling_base) == 432 ||
        error("expected 432 uniform scaling cases, found $(length(scaling_base))")
    "unknown" in source_hashes &&
        error("task-026 final campaign has an unknown source hash")
end
level_case_ids = Set(case_id(row) for row in levels)
all(case_id(row) in level_case_ids for row in cases) ||
    error("one or more task-026 cases have no per-level rows")
println("selected_window=", selected)
println("wrote ", SUMMARY026)
