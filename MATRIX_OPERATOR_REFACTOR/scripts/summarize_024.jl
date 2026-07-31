#!/usr/bin/env julia
# Dependency-free task-024 artifact validator and summarizer.
#
# Usage:
#   julia summarize_024.jl [raw_dir] [summary_dir]
# Set FM024_REQUIRE_COMPLETE=1 after both cluster jobs have been fetched to
# enforce the complete CPU(BLAS 1/64) + H200 matrix.

using Dates
using Statistics

const RAW = length(ARGS) >= 1 ? abspath(ARGS[1]) :
    abspath(joinpath(@__DIR__, "..", "data", "operator_ab_benchmark", "raw"))
const OUT = length(ARGS) >= 2 ? abspath(ARGS[2]) :
    abspath(joinpath(@__DIR__, "..", "data", "operator_ab_benchmark", "summary"))
const REQUIRE_COMPLETE =
    lowercase(get(ENV, "FM024_REQUIRE_COMPLETE", "0")) in ("1", "true", "yes")

function parse_csv_line(line)
    fields = String[]
    io = IOBuffer()
    quoted = false
    i = firstindex(line)
    while i <= lastindex(line)
        c = line[i]
        if c == '"'
            ni = nextind(line, i)
            if quoted && ni <= lastindex(line) && line[ni] == '"'
                write(io, '"')
                i = ni
            else
                quoted = !quoted
            end
        elseif c == ',' && !quoted
            push!(fields, String(take!(io)))
        else
            write(io, c)
        end
        i = nextind(line, i)
    end
    quoted && error("unterminated CSV quote")
    push!(fields, String(take!(io)))
    return fields
end

function read_csv(path)
    lines = readlines(path)
    isempty(lines) && error("empty CSV: $path")
    header = parse_csv_line(first(lines))
    rows = Dict{String,String}[]
    for (line_index, line) in enumerate(lines[2:end])
        lineno = line_index + 1
        isempty(strip(line)) && continue
        values = parse_csv_line(line)
        length(values) == length(header) ||
            error("$path:$lineno has $(length(values)) fields, expected $(length(header))")
        push!(rows, Dict(zip(header, values)))
    end
    return header, rows
end

csvfield(x) = begin
    s = replace(string(x), '\n' => ' ', '\r' => ' ')
    occursin(r"[\",]", s) ? "\"" * replace(s, '"' => "\"\"") * "\"" : s
end

function write_csv(path, names, rows)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, join(names, ','))
        for row in rows
            println(io, join((csvfield(get(row, n, "")) for n in names), ','))
        end
    end
end

num(row, key) = try parse(Float64, row[key]) catch; NaN end
integer(row, key) = try parse(Int, row[key]) catch; 0 end
finite(row, key) = isfinite(num(row, key))

const REQUIRED = [
    "schema_version", "campaign", "platform", "host", "precision", "p", "lh",
    "n", "ell", "distribution", "seed", "strategy", "status", "note",
    "correctness_pass", "direct_scope", "oracle_scope", "potential_error",
    "gradient_error", "coefficient_error", "routes", "direct_pairs",
    "nonempty_classes", "mean_occupancy", "max_occupancy", "p50_occupancy",
    "p90_occupancy", "p95_occupancy", "p99_occupancy", "occupancy_skew",
    "construction_ms", "construction_uploads", "operator_bytes",
    "metadata_bytes", "expansion_bytes", "scratch_bytes", "persistent_bytes",
    "peak_bytes", "b2m_ms_median", "m2m_ms_median", "m2l_ms_median",
    "l2l_ms_median", "l2b_ms_median", "update_ms_median",
    "lifecycle_ms_median", "finalize_ms_median", "recurring_ms_median",
    "body_uploads", "route_uploads", "operator_uploads",
    "expansion_host_copies", "git_commit", "git_worktree", "source_manifest",
]
const TIMINGS = [
    "construction_ms", "b2m_ms_median", "m2m_ms_median", "m2l_ms_median",
    "l2l_ms_median", "l2b_ms_median", "update_ms_median",
    "lifecycle_ms_median", "finalize_ms_median", "recurring_ms_median",
]
const STRATEGIES = Set(["concat", "factored", "precomputed_y", "dense"])

files = sort(filter(f -> endswith(f, ".csv"), readdir(RAW; join=true)))
isempty(files) && error("no raw task-024 CSV files found in $RAW")
allrows = Dict{String,String}[]
indexrows = Dict{String,String}[]
for file in files
    header, rows = read_csv(file)
    missing = setdiff(REQUIRED, header)
    isempty(missing) || error("$(basename(file)) missing required columns: $(join(missing, ", "))")
    isempty(rows) && error("$(basename(file)) contains no rows")
    for row in rows
        row["_file"] = basename(file)
        push!(allrows, row)
    end
    push!(indexrows, Dict(
        "file" => basename(file), "rows" => string(length(rows)),
        "platform" => join(sort(unique(r["platform"] for r in rows)), ";"),
        "host" => join(sort(unique(r["host"] for r in rows)), ";"),
        "commit" => join(sort(unique(r["git_commit"] for r in rows)), ";"),
        "worktree" => join(sort(unique(r["git_worktree"] for r in rows)), ";"),
        "source_manifest" => join(sort(unique(r["source_manifest"] for r in rows)), ";"),
        "schema_version" => join(sort(unique(r["schema_version"] for r in rows)), ";"),
    ))
end

casekeys = ["platform", "host", "blas_threads", "distribution", "precision",
    "lh", "p", "n", "ell", "seed"]
function key(row; strategy=true)
    fields = strategy ? [casekeys; "strategy"] : casekeys
    return join((get(row, k, "") for k in fields), '\0')
end
seen = Set{String}()
for row in allrows
    row["campaign"] == "resident_m2l_024" ||
        error("non-024 campaign row in $(row["_file"])")
    row["strategy"] in STRATEGIES ||
        error("unknown strategy $(row["strategy"]) in $(row["_file"])")
    k = key(row)
    k in seen && error("duplicate task-024 case/strategy: $(replace(k, '\0' => '/'))")
    push!(seen, k)
    row["status"] in ("eligible", "disqualified", "infeasible") ||
        error("invalid status $(row["status"])")
    if row["status"] == "eligible"
        row["correctness_pass"] == "true" ||
            error("eligible row failed correctness: $(row["_file"]) $(row["strategy"])")
        for t in TIMINGS
            finite(row, t) || error("non-finite $t in eligible row $(row["_file"]) $(row["strategy"])")
        end
        integer(row, "expansion_host_copies") == 0 ||
            error("expansion_host_copies invariant failed")
    else
        isempty(strip(row["note"])) &&
            error("$(row["status"]) row lacks an explanatory note")
    end
end

groups = Dict{String,Vector{Dict{String,String}}}()
for row in allrows
    push!(get!(groups, key(row; strategy=false), Dict{String,String}[]), row)
end
for (k, rows) in groups
    Set(r["strategy"] for r in rows) == STRATEGIES ||
        error("case lacks exactly four strategy rows: $(replace(k, '\0' => '/'))")
end

if REQUIRE_COMPLETE
    for platform in ("cpu", "cuda")
        for distribution in ("uniform", "clustered")
            tfs = distribution == "uniform" ? ("Float32", "Float64") : ("Float64",)
            ns = distribution == "uniform" ? (150, 2000, 20000) : (20000,)
            ell = distribution == "uniform" ? 3 : 4
            blas = platform == "cpu" ? ("1", "64") : ("",)
            for bt in blas, tf in tfs, lh in ("false", "true"), p in (4, 8, 12), n in ns
                matches = filter(allrows) do r
                    r["platform"] == platform && r["distribution"] == distribution &&
                    r["precision"] == tf && r["lh"] == lh &&
                    integer(r, "p") == p && integer(r, "n") == n &&
                    integer(r, "ell") == ell &&
                    (platform == "cuda" || r["blas_threads"] == bt)
                end
                length(matches) == 4 || error(
                    "missing complete coverage: $platform blas=$bt $distribution $tf LH=$lh P=$p N=$n ell=$ell")
            end
        end
    end
end

ranking_rows = Dict{String,String}[]
crossover = Dict("cpu" => Dict{String,String}[], "cuda" => Dict{String,String}[])
amortization_rows = Dict{String,String}[]
memory_rows = Dict{String,String}[]
notes = String[]
for groupkey in sort(collect(keys(groups)))
    rows = groups[groupkey]
    eligible = filter(r -> r["status"] == "eligible", rows)
    ordered_m2l = sort(eligible; by=r -> num(r, "m2l_ms_median"))
    ordered_step = sort(eligible; by=r -> num(r, "recurring_ms_median"))
    m2lrank = Dict(r["strategy"] => i for (i, r) in enumerate(ordered_m2l))
    steprank = Dict(r["strategy"] => i for (i, r) in enumerate(ordered_step))
    for row in rows
        push!(ranking_rows, merge(Dict(k => get(row, k, "") for k in casekeys),
            Dict("strategy" => row["strategy"], "status" => row["status"],
                "m2l_rank" => string(get(m2lrank, row["strategy"], 0)),
                "step_rank" => string(get(steprank, row["strategy"], 0)),
                "m2l_ms_median" => row["m2l_ms_median"],
                "recurring_ms_median" => row["recurring_ms_median"],
                "construction_ms" => row["construction_ms"])))
        push!(memory_rows, merge(Dict(k => get(row, k, "") for k in casekeys),
            Dict("strategy" => row["strategy"], "feasible" => string(row["status"] != "infeasible"),
                "status" => row["status"], "note" => row["note"],
                "operator_bytes" => row["operator_bytes"],
                "metadata_bytes" => row["metadata_bytes"],
                "expansion_bytes" => row["expansion_bytes"],
                "scratch_bytes" => row["scratch_bytes"],
                "persistent_bytes" => row["persistent_bytes"],
                "peak_bytes" => row["peak_bytes"])))
    end
    isempty(eligible) && (push!(notes, "No eligible strategy: " *
        join((get(first(rows), k, "") for k in casekeys), ", ")); continue)
    winner = first(ordered_m2l)
    stepwinner = first(ordered_step)
    push!(crossover[winner["platform"]],
        merge(Dict(k => get(winner, k, "") for k in casekeys),
            Dict("m2l_winner" => winner["strategy"],
                "m2l_ms_median" => winner["m2l_ms_median"],
                "step_winner" => stepwinner["strategy"],
                "step_ms_median" => stepwinner["recurring_ms_median"],
                "routes" => winner["routes"],
                "nonempty_classes" => winner["nonempty_classes"],
                "mean_occupancy" => winner["mean_occupancy"],
                "p95_occupancy" => winner["p95_occupancy"],
                "max_occupancy" => winner["max_occupancy"],
                "occupancy_skew" => winner["occupancy_skew"])))
    for competitor in eligible
        competitor === stepwinner && continue
        Δsteady = num(competitor, "recurring_ms_median") -
            num(stepwinner, "recurring_ms_median")
        Δconstruct = num(stepwinner, "construction_ms") -
            num(competitor, "construction_ms")
        break_even = Δsteady > 0 ? max(ceil(Int, Δconstruct / Δsteady), 0) : -1
        push!(amortization_rows,
            merge(Dict(k => get(stepwinner, k, "") for k in casekeys),
                Dict("steady_winner" => stepwinner["strategy"],
                    "competitor" => competitor["strategy"],
                    "winner_construction_ms" => stepwinner["construction_ms"],
                    "competitor_construction_ms" => competitor["construction_ms"],
                    "winner_step_ms" => stepwinner["recurring_ms_median"],
                    "competitor_step_ms" => competitor["recurring_ms_median"],
                    "break_even_steps" => string(break_even))))
    end
    for row in eligible
        relnoise = num(row, "m2l_ms_iqr") / max(num(row, "m2l_ms_median"), eps())
        relnoise > 0.10 && push!(notes,
            "Noisy M2L (>10% IQR/median): $(row["platform"]) $(row["strategy"]) " *
            "$(row["distribution"]) $(row["precision"]) LH=$(row["lh"]) " *
            "P=$(row["p"]) N=$(row["n"]) ratio=$(round(relnoise; digits=3)).")
    end
    for row in rows
        row["status"] != "eligible" && push!(notes,
            "$(uppercasefirst(row["status"])): $(row["platform"]) $(row["strategy"]) " *
            "$(row["distribution"]) $(row["precision"]) LH=$(row["lh"]) " *
            "P=$(row["p"]) N=$(row["n"]): $(row["note"])")
    end
end

mkpath(OUT)
write_csv(joinpath(OUT, "raw_data_index.csv"),
    ["file", "rows", "platform", "host", "commit", "worktree",
        "source_manifest", "schema_version"], indexrows)
rank_names = [casekeys; "strategy"; "status"; "m2l_rank"; "step_rank";
    "m2l_ms_median"; "recurring_ms_median"; "construction_ms"]
write_csv(joinpath(OUT, "case_rankings.csv"), rank_names, ranking_rows)
cross_names = [casekeys; "m2l_winner"; "m2l_ms_median"; "step_winner";
    "step_ms_median"; "routes"; "nonempty_classes"; "mean_occupancy";
    "p95_occupancy"; "max_occupancy"; "occupancy_skew"]
write_csv(joinpath(OUT, "cpu_crossovers.csv"), cross_names, crossover["cpu"])
write_csv(joinpath(OUT, "gpu_crossovers.csv"), cross_names, crossover["cuda"])
amort_names = [casekeys; "steady_winner"; "competitor";
    "winner_construction_ms"; "competitor_construction_ms";
    "winner_step_ms"; "competitor_step_ms"; "break_even_steps"]
write_csv(joinpath(OUT, "construction_amortization.csv"), amort_names, amortization_rows)
memory_names = [casekeys; "strategy"; "feasible"; "status"; "note";
    "operator_bytes"; "metadata_bytes"; "expansion_bytes"; "scratch_bytes";
    "persistent_bytes"; "peak_bytes"]
write_csv(joinpath(OUT, "memory_feasibility.csv"), memory_names, memory_rows)
open(joinpath(OUT, "unresolved_notes.md"), "w") do io
    println(io, "# Task 024 unresolved and noisy regimes\n")
    if isempty(notes)
        println(io, "No unresolved or noisy regimes were detected by the mechanical checks.")
    else
        for note in sort(unique(notes))
            println(io, "- ", note)
        end
    end
end
open(joinpath(OUT, "README.md"), "w") do io
    println(io, "# Task 024 resident-M2L benchmark summary\n")
    println(io, "Generated: ", now())
    println(io, "\nValidated raw files: ", length(files))
    println(io, "\nValidated rows: ", length(allrows))
    println(io, "\nValidated cases: ", length(groups))
    println(io, "\nEligible rows: ", count(r -> r["status"] == "eligible", allrows))
    println(io, "\nDisqualified rows: ", count(r -> r["status"] == "disqualified", allrows))
    println(io, "\nInfeasible rows: ", count(r -> r["status"] == "infeasible", allrows))
    println(io, "\nThe CSV tables in this directory are the dependency-free machine-readable ",
        "rankings, crossover, construction-amortization, and memory-feasibility outputs.")
end
println("validated $(length(allrows)) rows from $(length(files)) files; wrote $OUT")
