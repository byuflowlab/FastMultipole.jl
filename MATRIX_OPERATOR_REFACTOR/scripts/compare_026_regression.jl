# Join a new task-026 host-campaign `cases.csv` against one or more baseline
# summaries and report per-cell m2l_ms_median ratios (new / baseline).
#
#   julia --project MATRIX_OPERATOR_REFACTOR/scripts/compare_026_regression.jl \
#       <new_summary_dir> <baseline_summary_dir> [<baseline_summary_dir> ...]
#
# Baselines are merged in argument order; the LAST directory that supplies a given
# key wins, so pass the strategy-specific reruns after the broad baseline.
#
# Criterion (concretely-typed resident struct refactor, verification step c;
# extended for the task-027 mandatory old/new gate, which additionally requires an
# explicit warmed FULL-LIFECYCLE verdict, not only the warmed M2L stage):
#   * flag  cells > 5% slower
#   * block cells > 10% slower
#   * geomean of ratios must be <= 1.00
# applied independently to `m2l_ms_median` (warmed M2L stage) and, when the column
# is present in both files, `step_ms_median` (warmed full lifecycle). Both metrics
# must pass for the overall verdict to be PASS.
#   * m2l_alloc_bytes / update_alloc_bytes / step_alloc_bytes <= baseline everywhere
#   * zero counter violations (expansion_host_copies / route_uploads /
#     operator_uploads must stay at their baseline values)
# construction_ms is reported but never gates.

const KEY_COLS = ("policy", "strategy", "blas_threads", "ell", "n", "window_classes",
    "distribution", "precision", "lh", "P")
const METRIC_COLS = ("m2l_ms_median", "step_ms_median")
const ALLOC_COLS = ("m2l_alloc_bytes", "update_alloc_bytes", "step_alloc_bytes")
const COUNTER_COLS = ("expansion_host_copies", "route_uploads", "operator_uploads")

# Minimal RFC4180-free CSV reader: these files are plain comma-separated, no quotes.
function read_cases(path)
    lines = readlines(path)
    isempty(lines) && error("empty cases file: $path")
    header = split(lines[1], ',')
    rows = Dict{String,String}[]
    for line in @view lines[2:end]
        isempty(strip(line)) && continue
        fields = split(line, ',')
        length(fields) == length(header) || error(
            "$path: expected $(length(header)) fields, got $(length(fields))")
        push!(rows, Dict(String(h) => String(f) for (h, f) in zip(header, fields)))
    end
    return rows
end

casekey(row) = join((row[c] for c in KEY_COLS), '|')

num(row, col) = (v = get(row, col, ""); v == "" ? NaN : something(tryparse(Float64, v), NaN))

function main(args)
    length(args) >= 2 || error("usage: compare_026_regression.jl <new_dir> <baseline_dir>...")
    newrows = read_cases(joinpath(args[1], "cases.csv"))
    baseline = Dict{String,Dict{String,String}}()
    for dir in args[2:end], row in read_cases(joinpath(dir, "cases.csv"))
        baseline[casekey(row)] = row
    end

    matched = 0
    unmatched = String[]
    ratios = Dict(m => Float64[] for m in METRIC_COLS)
    flagged = Dict(m => Tuple{String,Float64}[] for m in METRIC_COLS)   # > 5%
    blockers = Dict(m => Tuple{String,Float64}[] for m in METRIC_COLS)  # > 10%
    alloc_violations = String[]
    counter_violations = String[]
    construction = Tuple{String,Float64}[]

    for row in newrows
        key = casekey(row)
        base = get(baseline, key, nothing)
        if base === nothing
            push!(unmatched, key)
            continue
        end
        matched += 1
        for metric in METRIC_COLS
            bm = num(base, metric)
            nm = num(row, metric)
            (isfinite(bm) && isfinite(nm) && bm > 0) || continue
            r = nm / bm
            push!(ratios[metric], r)
            r > 1.10 ? push!(blockers[metric], (key, r)) :
                r > 1.05 && push!(flagged[metric], (key, r))
        end
        for col in ALLOC_COLS
            b, n = num(base, col), num(row, col)
            isfinite(b) && isfinite(n) && n > b &&
                push!(alloc_violations, "$key  $col: $(Int(b)) -> $(Int(n))")
        end
        for col in COUNTER_COLS
            b, n = num(base, col), num(row, col)
            isfinite(b) && isfinite(n) && n != b &&
                push!(counter_violations, "$key  $col: $(Int(b)) -> $(Int(n))")
        end
        bc, nc = num(base, "construction_ms"), num(row, "construction_ms")
        isfinite(bc) && isfinite(nc) && bc > 0 && push!(construction, (key, nc / bc))
    end

    geo = Dict(m => (isempty(ratios[m]) ? NaN :
        exp(sum(log, ratios[m]) / length(ratios[m]))) for m in METRIC_COLS)
    println("new cases:            ", length(newrows))
    println("matched to baseline:  ", matched)
    println("unmatched (no base):  ", length(unmatched))
    for m in METRIC_COLS
        if isempty(ratios[m])
            println(rpad(m, 16), " ratios: ABSENT (column missing in one of the files)")
            continue
        end
        println(rpad(m, 16), " ratios: n=", length(ratios[m]),
            "  geomean=", round(geo[m], digits=4),
            "  min=", round(minimum(ratios[m]), digits=4),
            "  max=", round(maximum(ratios[m]), digits=4))
    end
    if !isempty(construction)
        cr = last.(construction)
        println("construction_ms ratio (report-only): geomean=",
            round(exp(sum(log, cr) / length(cr)), digits=4),
            "  max=", round(maximum(cr), digits=4))
    end

    function dump(title, items; limit=40)
        println("\n", title, " (", length(items), ")")
        for x in first(items, limit)
            println("  ", x isa Tuple ? "$(x[1])  ratio=$(round(x[2], digits=4))" : x)
        end
        length(items) > limit && println("  ... ", length(items) - limit, " more")
    end

    for m in METRIC_COLS
        sort!(blockers[m]; by=last, rev=true)
        sort!(flagged[m]; by=last, rev=true)
        dump("BLOCKERS  >10% slower  [$m]", blockers[m])
        dump("FLAGGED   >5% slower   [$m]", flagged[m])
    end
    dump("ALLOCATION regressions", alloc_violations)
    dump("COUNTER violations", counter_violations)
    isempty(unmatched) || dump("UNMATCHED keys", unmatched)

    metric_ok = Dict(m => isempty(blockers[m]) &&
        (isnan(geo[m]) || geo[m] <= 1.00) for m in METRIC_COLS)
    for m in METRIC_COLS
        println(rpad(m, 16), " verdict: ",
            isempty(ratios[m]) ? "NOT EVALUATED" : (metric_ok[m] ? "PASS" : "FAIL"))
    end
    nflag = sum(length(flagged[m]) for m in METRIC_COLS)
    ok = all(values(metric_ok)) && isempty(alloc_violations) &&
        isempty(counter_violations)
    println("\nVERDICT: ", ok ? "PASS" : "FAIL",
        nflag == 0 ? "" : "  (with $nflag cells to investigate)")
    return ok ? 0 : 1
end

exit(main(ARGS))
