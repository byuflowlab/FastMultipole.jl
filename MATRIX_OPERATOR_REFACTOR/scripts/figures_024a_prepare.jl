#!/usr/bin/env julia
#
# Task 024a -- benchmark figure data preparation.
#
# Reads the committed benchmark CSVs under MATRIX_OPERATOR_REFACTOR/data/ and
# writes one tidy, wide, pgfplots-ready CSV per figure panel into
# MATRIX_OPERATOR_REFACTOR/data/figures/tables/.  The .tex figure sources under
# data/figures/ consume those tables with \addplot table and contain no data
# filtering logic of their own.
#
# Dependencies: Julia stdlib only.  Nothing here may be added to the package or
# test environments, so CSV.jl / DataFrames.jl / DelimitedFiles are deliberately
# not used (DelimitedFiles is no longer a stdlib as of Julia 1.9).
#
# Usage (from the repository root or anywhere):
#     julia MATRIX_OPERATOR_REFACTOR/scripts/figures_024a_prepare.jl
#
# Missing sources, unexpected headers and empty selections are hard errors: a
# silently empty panel is the failure mode this script exists to prevent.

const SCRIPT_DIR = @__DIR__
const REFACTOR_DIR = normpath(joinpath(SCRIPT_DIR, ".."))
const DATA_DIR = joinpath(REFACTOR_DIR, "data")
const OUT_DIR = joinpath(DATA_DIR, "figures", "tables")

# ---------------------------------------------------------------------------
# Minimal CSV reader / writer
# ---------------------------------------------------------------------------

struct Table
    path::String
    header::Vector{String}
    rows::Vector{Vector{String}}
end

"""
    readtable(relpath) -> Table

Read a comma-separated file under `data/`.  None of the campaign CSVs contain
quoted fields or embedded commas (verified across all sources), so a plain split
is sufficient and keeps this script stdlib-only.
"""
function readtable(relpath::AbstractString)
    path = joinpath(DATA_DIR, relpath)
    isfile(path) || error("024a: missing source CSV: $path")
    lines = filter(!isempty, strip.(readlines(path)))
    length(lines) >= 2 || error("024a: source CSV has no data rows: $path")
    header = String.(split(lines[1], ','))
    rows = Vector{Vector{String}}(undef, length(lines) - 1)
    for (i, line) in enumerate(lines[2:end])
        fields = String.(split(line, ','))
        length(fields) == length(header) || error(
            "024a: $path line $(i+1) has $(length(fields)) fields, header has $(length(header))")
        rows[i] = fields
    end
    return Table(path, header, rows)
end

function colindex(t::Table, name::AbstractString)
    i = findfirst(==(name), t.header)
    i === nothing && error("024a: column '$name' not in $(t.path); header = $(join(t.header, ','))")
    return i
end

"Raw string field."
sget(t::Table, row, name) = row[colindex(t, name)]

"Numeric field; empty strings and 'NaN' become NaN."
function fget(t::Table, row, name)
    s = sget(t, row, name)
    (isempty(s) || s == "NaN" || s == "nan") && return NaN
    v = tryparse(Float64, s)
    v === nothing && error("024a: cannot parse '$s' as a number ($name in $(t.path))")
    return v
end

iget(t::Table, row, name) = Int(round(fget(t, row, name)))

"""
    select(t; kwargs...) -> Vector{row}

Rows whose named columns equal the given values (compared as strings; numbers
are stringified).  Errors when the selection is empty.
"""
function select(t::Table; kwargs...)
    idx = [(colindex(t, String(k)), v isa AbstractString ? String(v) : string(v)) for (k, v) in kwargs]
    out = filter(r -> all(((i, v),) -> r[i] == v, idx), t.rows)
    isempty(out) && error("024a: empty selection on $(t.path) for $(kwargs)")
    return out
end

"Distinct values of a column, in first-seen order."
function uniquevals(t::Table, name::AbstractString, rows = t.rows)
    i = colindex(t, name)
    out = String[]
    for r in rows
        r[i] in out || push!(out, r[i])
    end
    return out
end

const WRITTEN = String[]

"""
    writewide(name, xname, xs, colnames, cols; comment)

Write a wide table: one x column plus one y column per series.  Non-finite
values are written as `nan`; the figures set `unbounded coords=jump` so those
points are skipped rather than drawn at zero.  Column names carry their units.
"""
function writewide(name::AbstractString, xname::AbstractString, xs::Vector,
                   colnames::Vector{String}, cols::Vector{<:Vector};
                   comment::AbstractString = "")
    length(colnames) == length(cols) || error("024a: $name column/series count mismatch")
    all(c -> length(c) == length(xs), cols) || error("024a: $name series length mismatch")
    isempty(xs) && error("024a: $name has no rows")
    mkpath(OUT_DIR)
    open(joinpath(OUT_DIR, name), "w") do io
        isempty(comment) || println(io, "# ", comment)
        println(io, xname, ",", join(colnames, ","))
        for i in eachindex(xs)
            vals = map(c -> fmtnum(c[i]), cols)
            println(io, fmtnum(xs[i]), ",", join(vals, ","))
        end
    end
    push!(WRITTEN, name)
    return nothing
end

fmtnum(x::Integer) = string(x)
fmtnum(x::AbstractString) = x
function fmtnum(x::Real)
    isfinite(x) || return "nan"
    x == round(x) && abs(x) < 1e15 && return string(Int(round(x)))
    return string(x)
end

"Geometric mean over the finite, strictly positive entries; NaN when none."
function geomean(v)
    good = filter(x -> isfinite(x) && x > 0, v)
    isempty(good) && return NaN
    return exp(sum(log, good) / length(good))
end

# ---------------------------------------------------------------------------
# fig01 -- small-P crossover curves (019b isolated per-column operators)
#   Source: smallp_fallback_layout/m12-1-7/crossover_isolated_blas{1,64}.csv
#   EPYC 7763, Float64, batch = 64.
# ---------------------------------------------------------------------------

const FIG01_BATCH = 64
const ISO_VARIANTS = ("production_recurrence", "materialized", "factored")

function fig01()
    tabs = Dict(
        "1"  => readtable("smallp_fallback_layout/m12-1-7/crossover_isolated_blas1.csv"),
        "64" => readtable("smallp_fallback_layout/m12-1-7/crossover_isolated_blas64.csv"),
    )
    # per-expansion seconds keyed by (blas, lh, variant, P)
    vals = Dict{Tuple{String,String,String,Int},Float64}()
    Ps = Int[]
    for (blas, t) in tabs
        for r in select(t; batch = FIG01_BATCH, precision = "Float64")
            variant = sget(t, r, "variant")
            variant in ISO_VARIANTS || error("024a: unexpected variant '$variant' in $(t.path)")
            P = iget(t, r, "P")
            lh = sget(t, r, "lamb_helmholtz")
            vals[(blas, lh, variant, P)] = fget(t, r, "seconds_per_expansion")
            P in Ps || push!(Ps, P)
        end
    end
    sort!(Ps)

    for (lh, tag) in (("false", "phi"), ("true", "lh"))
        get1(blas, v, P) = get(vals, (blas, lh, v, P), NaN)
        # absolute per-expansion time, single-thread BLAS, microseconds
        writewide("fig01a_abs_$(tag).csv", "P", Ps,
            ["recurrence_us", "materialized_us", "factored_us"],
            [[1e6 * get1("1", "production_recurrence", P) for P in Ps],
             [1e6 * get1("1", "materialized", P) for P in Ps],
             [1e6 * get1("1", "factored", P) for P in Ps]];
            comment = "019b crossover_isolated, m12-1-7 EPYC 7763, BLAS=1, Float64, batch=$FIG01_BATCH, lamb_helmholtz=$lh")
        # speedup over production recurrence (>1 means the operator variant wins)
        ratio(blas, v) = [get1(blas, "production_recurrence", P) / get1(blas, v, P) for P in Ps]
        writewide("fig01b_ratio_$(tag).csv", "P", Ps,
            ["mat_blas1", "fac_blas1", "mat_blas64", "fac_blas64"],
            [ratio("1", "materialized"), ratio("1", "factored"),
             ratio("64", "materialized"), ratio("64", "factored")];
            comment = "019b crossover_isolated speedup over production recurrence, m12-1-7, Float64, batch=$FIG01_BATCH, lamb_helmholtz=$lh")
    end

    # Whole-slab crossover on the same host.  Keep this in a separate row from
    # the isolated operators because the stage harness includes route execution
    # while the isolated harness measures one operator column.
    stage = Dict(
        "1"  => readtable("smallp_fallback_layout/m12-1-7/crossover_stage_blas1.csv"),
        "64" => readtable("smallp_fallback_layout/m12-1-7/crossover_stage_blas64.csv"),
    )
    for (lh, tag) in (("false", "phi"), ("true", "lh"))
        Ps_stage = sort(unique(iget(stage["1"], r, "P") for r in
            select(stage["1"]; config = "tiny_parent", lamb_helmholtz = lh)))
        function stage_us(blas, form)
            t = stage[blas]
            [1e6 * fget(t, only(select(t; config = "tiny_parent", lamb_helmholtz = lh,
                                        form = form, P = P)), "seconds_per_route") for P in Ps_stage]
        end
        writewide("fig01c_wholeslab_$(tag).csv", "P", Ps_stage,
            ["recurrence_blas1_us", "concat_blas1_us", "concat_blas64_us"],
            [stage_us("1", "route_recurrence"), stage_us("1", "concat_host"),
             stage_us("64", "concat_host")];
            comment = "019b crossover_stage tiny_parent, m12-1-7 EPYC 7763, Float64, " *
                      "routes=3096, lamb_helmholtz=$lh; microseconds per routed expansion. " *
                      "Kept separate from isolated per-column timings because harness scope differs.")
    end
end

# ---------------------------------------------------------------------------
# fig02 -- speedup summaries
#   (a) whole-slab concat host stage vs per-route legacy recurrence (019b)
#   (b) H200 device lifecycle / M2L stage vs host (019b gpu_smallp)
# ---------------------------------------------------------------------------

const STAGE_CONFIGS = ("tiny_parent", "small_constp", "medium_constp")

function fig02a()
    for blas in ("1", "64")
        t = readtable("smallp_fallback_layout/m12-1-7/crossover_stage_blas$(blas).csv")
        # The P=1 rows are duplicated in these files; key-based reduction dedups.
        vals = Dict{Tuple{String,String,String,Int},Float64}()
        Ps = Int[]
        for r in t.rows
            cfg = sget(t, r, "config")
            cfg in STAGE_CONFIGS || error("024a: unexpected config '$cfg' in $(t.path)")
            form = sget(t, r, "form")
            form in ("concat_host", "route_recurrence") || error("024a: unexpected form '$form'")
            P = iget(t, r, "P")
            key = (cfg, sget(t, r, "lamb_helmholtz"), form, P)
            v = fget(t, r, "seconds_per_route")
            if haskey(vals, key)
                isapprox(vals[key], v; rtol = 1e-9) || error("024a: conflicting duplicate row $key in $(t.path)")
            end
            vals[key] = v
            P in Ps || push!(Ps, P)
        end
        sort!(Ps)
        names = String[]
        cols = Vector{Vector{Float64}}()
        for cfg in STAGE_CONFIGS, (lh, tag) in (("false", "phi"), ("true", "lh"))
            push!(names, replace(cfg, "_" => "") * "_" * tag)
            push!(cols, [get(vals, (cfg, lh, "route_recurrence", P), NaN) /
                         get(vals, (cfg, lh, "concat_host", P), NaN) for P in Ps])
        end
        writewide("fig02a_concat_speedup_blas$(blas).csv", "P", Ps, names, cols;
            comment = "019b crossover_stage, m12-1-7 EPYC 7763, BLAS=$blas, Float64; " *
                      "per-route legacy recurrence / whole-slab concat host (>1 = operators win)")
    end
end

function fig02b()
    t = readtable("smallp_fallback_layout/m13h-1-1/gpu_smallp.csv")
    rows = select(t; strategy = "concat", P = 4)
    # one point per (n, lamb_helmholtz); configs differ in ell, recorded in README
    ns = sort(unique(iget(t, r, "n") for r in rows))
    pick(n, lh, col) = begin
        sel = filter(r -> iget(t, r, "n") == n && sget(t, r, "lamb_helmholtz") == lh, rows)
        isempty(sel) ? NaN : fget(t, sel[1], col)
    end
    life(lh) = [pick(n, lh, "host_time") / pick(n, lh, "exec_time") for n in ns]
    m2lcat(lh) = [pick(n, lh, "host_concat_m2l") / pick(n, lh, "t_m2l") for n in ns]
    m2lrec(lh) = [pick(n, lh, "host_recur_m2l") / pick(n, lh, "t_m2l") for n in ns]
    writewide("fig02b_gpu_speedup.csv", "n", ns,
        ["lifecycle_phi", "lifecycle_lh", "m2l_vs_host_concat_phi", "m2l_vs_host_recurrence_phi"],
        [life("false"), life("true"), m2lcat("false"), m2lrec("false")];
        comment = "019b gpu_smallp, m13h-1-1 H200, P=4, ConcatenatedFixedZM2L; " *
                  "host time / device time (>1 = GPU faster). host_recur_m2l is NaN at n=10000")
end

function fig02c()
    tabs = Dict(
        "1"  => readtable("smallp_fallback_layout/m12-1-7/crossover_stage_blas1.csv"),
        "64" => readtable("smallp_fallback_layout/m12-1-7/crossover_stage_blas64.csv"),
    )
    routes = sort(unique(iget(tabs["1"], r, "routes") for r in
        select(tabs["1"]; P = 4, form = "concat_host")))
    function speedup(blas, lh)
        t = tabs[blas]
        [begin
            rows = filter(r -> iget(t, r, "P") == 4 && iget(t, r, "routes") == nroutes &&
                               sget(t, r, "lamb_helmholtz") == lh, t.rows)
            concat = filter(r -> sget(t, r, "form") == "concat_host", rows)
            recur = filter(r -> sget(t, r, "form") == "route_recurrence", rows)
            (isempty(concat) || isempty(recur)) ? NaN :
                fget(t, recur[1], "seconds_per_route") / fget(t, concat[1], "seconds_per_route")
        end for nroutes in routes]
    end
    writewide("fig02c_concat_speedup_routes.csv", "routes", routes,
        ["blas1_phi", "blas1_lh", "blas64_phi", "blas64_lh"],
        [speedup("1", "false"), speedup("1", "true"),
         speedup("64", "false"), speedup("64", "true")];
        comment = "019b crossover_stage, m12-1-7 EPYC 7763, Float64 P=4; " *
                  "per-route recurrence / whole-slab concat vs measured route count. " *
                  "The P=4 slice only, in both BLAS regimes; fig02c_routes_* below covers " *
                  "every P at BLAS=1.")

    # Full route-count scatter: every (config, P) point, not just the P=4 slice.
    # `routes` is non-monotonic in P (small_constp runs 8, 59334, 144290, 31254,
    # 144290, 15492 for P=1,2,3,4,6,8), so this is a scatter answering "does route
    # count predict the win?" -- not a curve in P.  One table per config so the
    # figure can carry config in the mark and channel in the colour.
    for cfg in STAGE_CONFIGS
        t = tabs["1"]
        pts = Dict{String,Vector{Tuple{Int,Float64}}}()
        for lh in ("false", "true")
            rows = filter(r -> sget(t, r, "config") == cfg &&
                               sget(t, r, "lamb_helmholtz") == lh, t.rows)
            isempty(rows) && error("024a: no $cfg rows for lamb_helmholtz=$lh in $(t.path)")
            for P in sort(unique(iget(t, r, "P") for r in rows))
                atP = filter(r -> iget(t, r, "P") == P, rows)
                concat = filter(r -> sget(t, r, "form") == "concat_host", atP)
                recur = filter(r -> sget(t, r, "form") == "route_recurrence", atP)
                (isempty(concat) || isempty(recur)) && continue
                nroutes = iget(t, concat[1], "routes")
                iget(t, recur[1], "routes") == nroutes ||
                    error("024a: $cfg P=$P route counts differ between forms in $(t.path)")
                push!(get!(pts, lh, Tuple{Int,Float64}[]),
                      (nroutes, fget(t, recur[1], "seconds_per_route") /
                                fget(t, concat[1], "seconds_per_route")))
            end
        end
        # phi and LH share the same (config, P) grid, hence the same route counts
        rs = sort(unique(n for v in values(pts) for (n, _) in v))
        col(lh) = [begin
            hit = filter(((n, _),) -> n == nroutes, pts[lh])
            isempty(hit) ? NaN : maximum(x -> x[2], hit)
        end for nroutes in rs]
        writewide("fig02c_routes_$(replace(cfg, "_" => "")).csv", "routes", rs,
            ["phi", "lh"], [col("false"), col("true")];
            comment = "019b crossover_stage_blas1, m12-1-7 EPYC 7763, Float64, config=$cfg; " *
                      "per-route recurrence / whole-slab concat against the measured route count " *
                      "over every P this config ran. Route count is non-monotonic in P, so this " *
                      "is a scatter, not a curve. Where two P values share a route count " *
                      "(small_constp P=2 and P=3 both report 144290) the better speedup is kept.")
    end
end

function fig02d()
    t = readtable("operator_ab_benchmark/summary/case_rankings.csv")
    # Best eligible resident strategy on each platform, matched on the complete
    # 024 case key.  CPU uses the process-start BLAS=1 regime.
    ns = sort(unique(iget(t, r, "n") for r in t.rows if
        sget(t, r, "distribution") == "uniform" && sget(t, r, "precision") == "Float64" &&
        iget(t, r, "p") == 4))
    function best(platform, lh, n)
        rows = filter(r -> sget(t, r, "platform") == platform &&
                           (platform != "cpu" || sget(t, r, "blas_threads") == "1") &&
                           sget(t, r, "distribution") == "uniform" &&
                           sget(t, r, "precision") == "Float64" &&
                           sget(t, r, "lh") == lh && iget(t, r, "p") == 4 &&
                           iget(t, r, "n") == n && sget(t, r, "status") == "eligible", t.rows)
        isempty(rows) && return NaN
        return minimum(fget(t, r, "recurring_ms_median") for r in rows)
    end
    ratio(lh) = [best("cpu", lh, n) / best("cuda", lh, n) for n in ns]
    writewide("fig02d_integrated_gpu_speedup.csv", "n", ns,
        ["best_resident_phi", "best_resident_lh"], [ratio("false"), ratio("true")];
        comment = "024 case_rankings, uniform Float64 P=4; best eligible CPU resident recurring " *
                  "step at BLAS=1 divided by best eligible H200 resident recurring step, matched by n")
end

# ---------------------------------------------------------------------------
# fig03 -- H200 lifecycle stage breakdown and roofline gap (019 / 022 / 023)
# ---------------------------------------------------------------------------

const STAGE_COLS = ("t_b2m", "t_m2m", "t_m2l", "t_l2l", "t_l2b")
const STAGE_LABELS = ("B2M", "M2M", "M2L", "L2L", "L2B")
const FIG03_KEY = (n = 100000, ell = 4, P = 4, policy = "constp", strategy = "concat", chunk = 131072)

function fig03()
    base = readtable("operator_performance_tuning/cuda_022_baseline.csv")
    abd = readtable("operator_performance_tuning/m13h-1-1/cuda_019_phaseABD_throughput.csv")
    e = readtable("operator_performance_tuning/m13h-1-1/cuda_019_phaseE_throughput.csv")
    one(t) = only(select(t; FIG03_KEY...))

    rb, ra, re = one(base), one(abd), one(e)
    # (a) per-stage device time, milliseconds, log axis
    writewide("fig03a_stages.csv", "stage_index", collect(1:length(STAGE_COLS)),
        ["baseline_022_ms", "post_abd_ms", "post_e_ms"],
        [[1e3 * fget(base, rb, c) for c in STAGE_COLS],
         [1e3 * fget(abd, ra, c) for c in STAGE_COLS],
         [1e3 * fget(e, re, c) for c in STAGE_COLS]];
        comment = "019/022, m13h-1-1 H200, n=1e5 ell=4 P=4 constp concat chunk=2^17; " *
                  "stage order = $(join(STAGE_LABELS, '/')); baseline is single-shot, post-* are min-of-3")

    # (b) 019 evaluation beside the then-one-shot host setup costs. Task 023
    # subsequently removes list/state construction from the recurring loop.
    writewide("fig03b_lifecycle.csv", "phase_index", collect(1:3),
        ["baseline_022_s", "post_abd_s", "post_e_s"],
        [[fget(base, rb, "exec_time"), fget(base, rb, "list_time"), fget(base, rb, "build_time")],
         [fget(abd, ra, "exec_time"), fget(abd, ra, "list_time"), fget(abd, ra, "build_time")],
         [fget(e, re, "exec_time"), fget(e, re, "list_time"), fget(e, re, "build_time")]];
        comment = "019/022, m13h-1-1 H200, same case; phase order = device exec / host list build / state build. " *
                  "List/state build are setup costs, not 023 recurring-step costs.")

    # (c) 023 production-integration per-step split at the same n, P
    t023 = readtable("production_integration/benchmark_023_m13h-1-1_20260714-233159.csv")
    phases = ("update", "lifecycle", "finalize")
    ns023 = (10000, 100000)
    cols = Vector{Vector{Float64}}()
    for n in ns023
        push!(cols, [fget(t023, only(select(t023; backend = "gpu", path = "radix", n = n, P = 4, phase = ph)), "seconds")
                     for ph in phases])
    end
    writewide("fig03c_step_split.csv", "phase_index", collect(1:length(phases)),
        ["n10000_s", "n100000_s"], cols;
        comment = "023 benchmark_023_m13h-1-1, H200 resident radix step split; " *
                  "phase order = $(join(phases, '/')) (sums to the recurring step)")

    # (d) one-shot cache construction beside the recurring step it enables.
    # The gpu construct value at n=1e4 is JIT-contaminated: that case runs first
    # in the 023 script, so it absorbs one-time CUDA kernel compilation.  Both
    # values are plotted and the figure annotates which one is representative.
    one023(backend, n, phase) = fget(t023,
        only(select(t023; backend = backend, path = "radix", n = n, P = 4, phase = phase)), "seconds")
    writewide("fig03d_oneshot_vs_recurring.csv", "n_index", collect(1:length(ns023)),
        ["cpu_construct_s", "cpu_step_s", "gpu_construct_s", "gpu_step_s"],
        [[one023("cpu", n, "construct") for n in ns023], [one023("cpu", n, "step") for n in ns023],
         [one023("gpu", n, "construct") for n in ns023], [one023("gpu", n, "step") for n in ns023]];
        comment = "023 benchmark_023_m13h-1-1, radix path P=4 ell=4; one-shot RadixFMMCache " *
                  "construction vs the recurring step it enables. n order = " *
                  "$(join(ns023, '/')). gpu construct at n=1e4 (23.7 s) is dominated by one-time " *
                  "CUDA kernel compilation -- the n=1e5 value (0.126 s) is the representative " *
                  "setup cost; m13h-2-1 reads 24.7 s / 0.030 s for the same pair.")
end

# ---------------------------------------------------------------------------
# fig04 -- storage and allocation
# ---------------------------------------------------------------------------

function fig04a()
    t = readtable("smallp_fallback_layout/m12-1-7/layout_storage.csv")
    Ps = sort(unique(iget(t, r, "P_phi") for r in t.rows))
    frac(buf) = [100 * fget(t, only(select(t; buffer = buf, P_phi = P)), "padded_overhead_frac") for P in Ps]
    writewide("fig04a_layout_overhead.csv", "P_phi", Ps,
        ["flat_pct", "degreemajor_pct"], [frac("FlatCoefficientBuffer"), frac("DegreeMajorRealBuffer")];
        comment = "019b layout_storage (analytic, host-independent); padded-vs-ragged chi storage " *
                  "overhead per column, Float64, P_chi = P_phi + 1")
end

const STRATEGIES = ("concat", "factored", "precomputed_y", "dense")

function fig04b()
    t = readtable("operator_ab_benchmark/summary/memory_feasibility.csv")
    Ps = sort(unique(iget(t, r, "p") for r in t.rows))
    # dense is the only strategy with a materialized per-class operator payload
    function dense_mib(precision, lh, dist, n)
        out = Float64[]
        for P in Ps
            sel = filter(r -> sget(t, r, "strategy") == "dense" && sget(t, r, "platform") == "cuda" &&
                              sget(t, r, "precision") == precision && sget(t, r, "lh") == lh &&
                              sget(t, r, "distribution") == dist && iget(t, r, "n") == n &&
                              iget(t, r, "p") == P, t.rows)
            push!(out, isempty(sel) ? NaN : fget(t, sel[1], "operator_bytes") / 2^20)
        end
        return out
    end
    writewide("fig04b_dense_operator_mib.csv", "P", Ps,
        ["f64_phi_mib", "f64_lh_mib", "f32_phi_mib", "f32_lh_mib", "clustered_f64_phi_mib", "clustered_f64_lh_mib"],
        [dense_mib("Float64", "false", "uniform", 20000), dense_mib("Float64", "true", "uniform", 20000),
         dense_mib("Float32", "false", "uniform", 20000), dense_mib("Float32", "true", "uniform", 20000),
         dense_mib("Float64", "false", "clustered", 20000), dense_mib("Float64", "true", "clustered", 20000)];
        comment = "024 memory_feasibility, cuda rows, n=20000; DenseTranslationM2L operator payload. " *
                  "Points above the 12 GiB gate are the infeasible cases; Float32/P=12 dense is a " *
                  "hard-coded skip. concat/factored/precomputed_y report operator_bytes = 0 (no " *
                  "materialized per-class operator).")

    # device peak is a per-strategy delta on CUDA (CPU peak is process-wide maxrss -- excluded)
    peak(strategy) = begin
        out = Float64[]
        for P in Ps
            sel = filter(r -> sget(t, r, "strategy") == strategy && sget(t, r, "platform") == "cuda" &&
                              sget(t, r, "precision") == "Float64" && sget(t, r, "lh") == "false" &&
                              sget(t, r, "distribution") == "uniform" && iget(t, r, "n") == 20000 &&
                              iget(t, r, "p") == P, t.rows)
            push!(out, isempty(sel) ? NaN : fget(t, sel[1], "peak_bytes") / 2^20)
        end
        out
    end
    writewide("fig04c_cuda_peak_mib.csv", "P", Ps,
        ["concat_mib", "factored_mib", "precomputed_y_mib", "dense_mib"],
        [peak(s) for s in STRATEGIES];
        comment = "024 memory_feasibility, cuda rows, Float64 phi-only uniform n=20000; " *
                  "peak_bytes is a per-strategy device-memory delta (comparable across strategies, " *
                  "unlike the process-wide CPU maxrss)")
end

function fig04d()
    t = readtable("operator_performance_tuning/local_macos_allocations_storage.csv")
    # concat-plan route geometry: analytic payload before vs measured payload now
    geo(metric, precision) = fget(t, only(select(t; category = "storage", metric = metric,
                                                 precision = precision)), "bytes_per_count")
    writewide("fig04d_geometry_bytes.csv", "precision_index", collect(1:2),
        ["before_analytic_bytes", "current_measured_bytes"],
        [[geo("geometry_before_analytic", p) for p in ("Float64", "Float32")],
         [geo("geometry_current_measured", p) for p in ("Float64", "Float32")]];
        comment = "019 local_macos_allocations_storage; per-route concat-plan geometry payload, " *
                  "precision order = Float64/Float32 (n=2000, ell=3, P=4, constant-P)")

    cache(metric) = begin
        rows = select(t; category = "storage", metric = metric)
        Ps = sort([iget(t, r, "P") for r in rows])
        Ps, [fget(t, only(filter(r -> iget(t, r, "P") == P, rows)), "total_bytes") / 1024 for P in Ps]
    end
    Ps, spos = cache("cache_S_pos_S_neg")
    Ps2, ymodes = cache("cache_factored_y_modes")
    Ps == Ps2 || error("024a: invariant-cache metrics cover different P sets")
    writewide("fig04e_cache_kib.csv", "P", Ps,
        ["S_pos_S_neg_kib", "factored_y_modes_kib"], [spos, ymodes];
        comment = "019 local_macos_allocations_storage; measured invariant-cache payload, Float64")

    # Warmed host allocations after task-019 tuning.  The launch unit differs
    # by stage (chunk for M2L, group for M2M/L2L), so preserve the recorded
    # per-unit normalization and state it in the table and figure.
    metrics = ("concat_m2l", "resident_m2m", "resident_l2l")
    alloc(lh) = [fget(t, only(select(t; category = "allocation", metric = metric,
                                   precision = "Float64", lamb_helmholtz = lh)),
                           "bytes_per_count") / 1024 for metric in metrics]
    writewide("fig04g_warmed_allocations_kib.csv", "stage_index", collect(1:length(metrics)),
        ["phi_kib_per_unit", "lh_kib_per_unit"], [alloc("false"), alloc("true")];
        comment = "019 local_macos_allocations_storage, warmed host launches, Float64 P=4 n=2000 ell=3; " *
                  "stage order = concat M2L / resident M2M / resident L2L; unit = chunk / group / group")
end

function fig04e()
    t = readtable("operator_ab_benchmark/summary/memory_feasibility.csv")
    rows = select(t; platform = "cuda", strategy = "concat", precision = "Float64",
                  distribution = "uniform", n = 20000)
    Ps = sort(unique(iget(t, r, "p") for r in rows))
    field(lh, name) = [fget(t, only(filter(r -> iget(t, r, "p") == P &&
                                               sget(t, r, "lh") == lh, rows)), name) / 2^20
                       for P in Ps]
    writewide("fig04f_concat_buffers_mib.csv", "P", Ps,
        ["expansion_phi_mib", "expansion_lh_mib", "scratch_phi_mib", "scratch_lh_mib"],
        [field("false", "expansion_bytes"), field("true", "expansion_bytes"),
         field("false", "scratch_bytes"), field("true", "scratch_bytes")];
        comment = "024 memory_feasibility, H200 concat plan, uniform Float64 n=20000; " *
                  "plan-reported expansion and scratch footprints. This is a within-strategy " *
                  "order/channel comparison, not a cross-strategy scratch comparison.")
end

# ---------------------------------------------------------------------------
# fig05 -- four-strategy regime selection (024)
# ---------------------------------------------------------------------------

"Case key: everything except the strategy."
casekey(t, r) = Tuple(sget(t, r, c) for c in
    ("platform", "host", "blas_threads", "distribution", "precision", "lh", "p", "n", "ell", "seed"))

function fig05ab()
    t = readtable("operator_ab_benchmark/summary/case_rankings.csv")
    # group eligible rows by case
    cases = Dict{Any,Dict{String,Float64}}()
    for r in t.rows
        sget(t, r, "status") == "eligible" || continue
        s = sget(t, r, "strategy")
        s in STRATEGIES || error("024a: unexpected strategy '$s' in $(t.path)")
        get!(cases, casekey(t, r), Dict{String,Float64}())[s] = fget(t, r, "recurring_ms_median")
    end
    isempty(cases) && error("024a: no eligible 024 cases")

    Ps = sort(unique(parse(Int, k[7]) for k in keys(cases)))
    for (platform, tag) in (("cpu", "cpu"), ("cuda", "gpu"))
        names = String[]
        cols = Vector{Vector{Float64}}()
        wins = zeros(Int, length(STRATEGIES))
        for (si, s) in enumerate(STRATEGIES)
            push!(names, s * "_rel")
            col = Float64[]
            for P in Ps
                ratios = Float64[]
                for (k, v) in cases
                    k[1] == platform && parse(Int, k[7]) == P || continue
                    best = minimum(values(v))
                    haskey(v, s) && push!(ratios, v[s] / best)
                end
                push!(col, geomean(ratios))
            end
            push!(cols, col)
            for (k, v) in cases
                k[1] == platform || continue
                argmin_s = STRATEGIES[argmin([get(v, x, Inf) for x in STRATEGIES])]
                argmin_s == s && (wins[si] += 1)
            end
        end
        writewide("fig05a_relstep_$(tag).csv", "P", Ps, names, cols;
            comment = "024 case_rankings, $platform; geometric mean of recurring-step time " *
                      "normalized to the per-case best strategy (1.0 = winner). Strategies " *
                      "infeasible in a case are omitted from that case's mean.")
        writewide("fig05b_wins_$(tag).csv", "strategy_index", collect(1:length(STRATEGIES)),
            ["wins"], [wins];
            comment = "024 case_rankings, $platform; complete-recurring-step wins per strategy " *
                      "out of $(count(k -> k[1] == platform, keys(cases))) cases. " *
                      "Strategy order = $(join(STRATEGIES, '/'))")
    end
end

"""
Regime map: y is a regime slot (BLAS / precision / channel / distribution / n),
x is P, and the mark identifies the winning strategy.  Tick labels are emitted as
a generated \\input snippet because pgfplots cannot read symbolic tick labels from
a data table.
"""
function fig05cd()
    for (file, platform, tag, macroname) in (
            ("operator_ab_benchmark/summary/cpu_crossovers.csv", "cpu", "cpu", "FigFiveCpuTicks"),
            ("operator_ab_benchmark/summary/gpu_crossovers.csv", "cuda", "gpu", "FigFiveGpuTicks"))
        t = readtable(file)
        Ps = sort(unique(iget(t, r, "p") for r in t.rows))
        regimes = String[]        # ordered regime labels
        slot = Dict{String,Int}()
        pts = Dict{String,Vector{Tuple{Int,Int}}}()
        for r in t.rows
            blas = sget(t, r, "blas_threads")
            prec = sget(t, r, "precision") == "Float64" ? "F64" : "F32"
            chan = sget(t, r, "lh") == "true" ? "LH" : "phi"
            dist = sget(t, r, "distribution") == "clustered" ? "clust" : "unif"
            label = "b$(blas) $(prec) $(chan) $(dist) n=$(sget(t, r, "n"))"
            if !haskey(slot, label)
                push!(regimes, label)
                slot[label] = length(regimes)
            end
            w = sget(t, r, "step_winner")
            w in STRATEGIES || error("024a: unexpected step_winner '$w' in $(t.path)")
            push!(get!(pts, w, Tuple{Int,Int}[]), (iget(t, r, "p"), slot[label]))
        end
        # stable regime order: BLAS, precision, channel, distribution, n
        order = sortperm(regimes; by = l -> begin
            parts = split(l, ' ')
            (parts[1], parts[2], parts[3] == "phi" ? 1 : 2, parts[4],
             parse(Int, split(parts[5], '=')[2]))
        end)
        remap = Dict(old => new for (new, old) in enumerate(order))
        for w in keys(pts)
            pts[w] = [(p, remap[s]) for (p, s) in pts[w]]
        end
        for w in STRATEGIES
            haskey(pts, w) || continue
            sorted = sort(pts[w])
            writewide("fig05$(tag)_win_$(w).csv", "P", [p for (p, _) in sorted],
                ["regime_slot"], [[s for (_, s) in sorted]];
                comment = "024 $(basename(file)); cases whose complete recurring step is won by " *
                          "$(w). regime_slot indexes the generated tick list.")
        end
        # Emitted as a pgfplots style rather than bare \def macros: a style is
        # expanded correctly wherever it appears in an axis option list, which a
        # macro holding a key list is not.
        mkpath(OUT_DIR)
        open(joinpath(OUT_DIR, "fig05$(tag)_ticks.tex"), "w") do io
            println(io, "% generated by figures_024a_prepare.jl -- regime axis for $platform")
            println(io, "\\pgfplotsset{", macroname, "/.style={%")
            println(io, "  ytick={", join(1:length(regimes), ","), "},%")
            println(io, "  yticklabels={{", join(regimes[order], "},{"), "}},%")
            println(io, "  ymin=0.4, ymax=", length(regimes), ".6,%")
            println(io, "  xtick={", join(Ps, ","), "}}}")
        end
        push!(WRITTEN, "fig05$(tag)_ticks.tex")
    end
end

# ---------------------------------------------------------------------------
# fig06 -- tradeoffs and limitations
# ---------------------------------------------------------------------------

function fig06a()
    t = readtable("operator_ab_benchmark/summary/case_rankings.csv")
    vals = Dict{Any,Float64}()
    for r in t.rows
        sget(t, r, "status") == "eligible" || continue
        vals[(sget(t, r, "platform"), sget(t, r, "blas_threads"), sget(t, r, "distribution"),
              sget(t, r, "precision"), sget(t, r, "lh"), iget(t, r, "p"), sget(t, r, "n"),
              sget(t, r, "strategy"))] = fget(t, r, "recurring_ms_median")
    end
    Ps = sort(unique(k[6] for k in keys(vals)))
    for (platform, tag) in (("cpu", "cpu"), ("cuda", "gpu"))
        cols = Vector{Vector{Float64}}()
        for s in STRATEGIES
            col = Float64[]
            for P in Ps
                ratios = Float64[]
                for k in keys(vals)
                    k[1] == platform && k[6] == P && k[5] == "false" && k[8] == s || continue
                    khi = (k[1], k[2], k[3], k[4], "true", k[6], k[7], k[8])
                    haskey(vals, khi) && push!(ratios, vals[khi] / vals[k])
                end
                push!(col, geomean(ratios))
            end
            push!(cols, col)
        end
        writewide("fig06a_lh_ratio_$(tag).csv", "P", Ps, [s * "_ratio" for s in STRATEGIES], cols;
            comment = "024 case_rankings, $platform; geometric mean of Lamb-Helmholtz / phi-only " *
                      "recurring-step time at matched (BLAS, precision, distribution, n)")
    end
end

function fig06b()
    load(blas) = begin
        t = readtable("smallp_fallback_layout/m12-1-7/crossover_stage_blas$(blas).csv")
        d = Dict{Tuple{String,String,Int},Float64}()
        for r in select(t; form = "concat_host")
            d[(sget(t, r, "config"), sget(t, r, "lamb_helmholtz"), iget(t, r, "P"))] =
                fget(t, r, "seconds_per_route")
        end
        d
    end
    b1, b64 = load("1"), load("64")
    Ps = sort(unique(k[3] for k in keys(b1)))
    names = String[]
    cols = Vector{Vector{Float64}}()
    for cfg in STAGE_CONFIGS, (lh, tag) in (("false", "phi"), ("true", "lh"))
        push!(names, replace(cfg, "_" => "") * "_" * tag)
        push!(cols, [get(b1, (cfg, lh, P), NaN) / get(b64, (cfg, lh, P), NaN) for P in Ps])
    end
    writewide("fig06b_blas_degradation.csv", "P", Ps, names, cols;
        comment = "019b crossover_stage, m12-1-7 EPYC 7763; BLAS=1 time / BLAS=64 time for the " *
                  "whole-slab concat host stage (<1 = 64-thread BLAS is slower)")
end

function fig06c()
    t = readtable("operator_ab_benchmark/summary/construction_amortization.csv")
    # dense-vs-precomputed_y pairs only: the decision 024's GPU default rests on
    out = Dict{String,Vector{Tuple{Int,Float64}}}()
    for r in t.rows
        w, c = sget(t, r, "steady_winner"), sget(t, r, "competitor")
        Set((w, c)) == Set(("dense", "precomputed_y")) || continue
        be = fget(t, r, "break_even_steps")
        isfinite(be) && be > 0 || continue
        push!(get!(out, sget(t, r, "platform"), Tuple{Int,Float64}[]), (iget(t, r, "p"), be))
    end
    isempty(out) && error("024a: no dense/precomputed_y construction-amortization pairs")
    Ps = sort(unique(p for v in values(out) for (p, _) in v))
    stat(platform, f) = [begin
        vs = [be for (p, be) in get(out, platform, Tuple{Int,Float64}[]) if p == P]
        isempty(vs) ? NaN : f(vs)
    end for P in Ps]
    writewide("fig06c_break_even.csv", "P", Ps,
        ["cpu_min", "cpu_max", "gpu_min", "gpu_max"],
        [stat("cpu", minimum), stat("cpu", maximum), stat("cuda", minimum), stat("cuda", maximum)];
        comment = "024 construction_amortization; steps of recurring work needed to amortize the " *
                  "dense-vs-precomputed_y construction difference (min/max over all cases at each P)")
end

function fig06d()
    dir = joinpath(DATA_DIR, "operator_ab_benchmark", "raw")
    isdir(dir) || error("024a: missing 024 raw directory: $dir")
    files = sort(filter(f -> endswith(f, ".csv"), readdir(dir)))
    isempty(files) && error("024a: no raw 024 CSVs in $dir")
    # (recurring step ms, potential error, gradient error) keyed by precision and P
    pts = Dict{Tuple{String,Int},Vector{NTuple{3,Float64}}}()
    for f in files
        t = readtable(joinpath("operator_ab_benchmark", "raw", f))
        for r in t.rows
            sget(t, r, "status") == "eligible" || continue
            sget(t, r, "platform") == "cuda" || continue
            sget(t, r, "correctness_pass") == "true" ||
                error("024a: eligible row with correctness_pass=false in $f")
            key = (sget(t, r, "precision"), iget(t, r, "p"))
            push!(get!(pts, key, NTuple{3,Float64}[]),
                  (fget(t, r, "recurring_ms_median"), fget(t, r, "potential_error"),
                   fget(t, r, "gradient_error")))
        end
    end
    isempty(pts) && error("024a: no accuracy rows recovered from 024 raw")
    for (kind, col) in (("potential", 2), ("gradient", 3))
        for prec in ("Float64", "Float32")
            Ps = sort(unique(k[2] for k in keys(pts) if k[1] == prec))
            for P in Ps
                v = sort(pts[(prec, P)])
                writewide("fig06d_$(kind)_$(lowercase(prec))_p$(P).csv", "step_ms",
                    [x[1] for x in v], ["error"], [[x[col] for x in v]];
                    comment = "024 raw, cuda H200 eligible rows, $prec P=$P; max direct " *
                              "$(kind) error vs complete recurring-step time. Single fixed " *
                              "ConstantPAnalyticStencil tolerance -- no tolerance sweep exists.")
            end
        end
    end
end

# ---------------------------------------------------------------------------
# fig07 -- 008c baseline projections vs what 015 realized
# ---------------------------------------------------------------------------

const BASE_STAGES = ("axis_swap", "m2l_z_translation", "m2m_z_translation", "l2l_z_translation")
const FIG07_P = 20

function fig07a()
    for (relpath, tag, host) in (
            ("impl_performance_baseline/m13h-1-1/dense_vs_loop_blas1.csv", "blas1", "m13h-1-1 Xeon 8568Y+, BLAS=1"),
            ("impl_performance_baseline/m12-2-5/dense_vs_loop_blas72.csv", "blas72", "m12-2-5 EPYC 7763, BLAS=72"))
        t = readtable(relpath)
        batches = sort(unique(iget(t, r, "batch") for r in t.rows))
        cols = Vector{Vector{Float64}}()
        for stage in BASE_STAGES
            col = Float64[]
            for b in batches
                sel(form) = filter(r -> sget(t, r, "stage") == stage && sget(t, r, "form") == form &&
                                        sget(t, r, "precision") == "Float64" &&
                                        iget(t, r, "P") == FIG07_P && iget(t, r, "batch") == b, t.rows)
                rd, rr = sel("dense"), sel("recurrence")
                push!(col, (isempty(rd) || isempty(rr)) ? NaN :
                    fget(t, rr[1], "seconds_per_expansion") / fget(t, rd[1], "seconds_per_expansion"))
            end
            push!(cols, col)
        end
        writewide("fig07a_dense_speedup_$(tag).csv", "batch", batches,
            [s * "_speedup" for s in BASE_STAGES], cols;
            comment = "008c dense_vs_loop, $host, Float64 P=$FIG07_P; recurrence / dense " *
                      "per-expansion time. Large-batch recurrence rows are projected " *
                      "(scaled_from_per_expansion=true).")
    end
end

function fig07b()
    t = readtable("impl_performance_baseline/m13h-1-1/dense_gpu.csv")
    batches = sort(unique(iget(t, r, "batch") for r in t.rows))
    series = (("m2l_z_translation", "device_resident"), ("m2l_z_translation", "with_transfer"),
              ("axis_swap", "device_resident"), ("axis_swap", "with_transfer"))
    cols = Vector{Vector{Float64}}()
    for (stage, tv) in series
        push!(cols, [begin
            sel = filter(r -> sget(t, r, "stage") == stage && sget(t, r, "transfer_variant") == tv &&
                              sget(t, r, "launch_strategy") == "fused_kernel" &&
                              sget(t, r, "precision") == "Float64" && iget(t, r, "P") == FIG07_P &&
                              iget(t, r, "batch") == b, t.rows)
            isempty(sel) ? NaN : 1e9 * fget(t, sel[1], "seconds_per_expansion")
        end for b in batches])
    end
    writewide("fig07b_gpu_transfer_floor.csv", "batch", batches,
        ["m2lz_resident_ns", "m2lz_with_transfer_ns", "axisswap_resident_ns", "axisswap_with_transfer_ns"],
        cols;
        comment = "008c dense_gpu, m13h-1-1 H200, hand-written fused scalar kernel (not cuBLAS), " *
                  "Float64 P=$FIG07_P; nanoseconds per expansion")

    # best CPU dense on the same node, as the reference the GPU has to beat
    tc = readtable("impl_performance_baseline/m13h-1-1/dense_vs_loop_blas1.csv")
    best(stage) = 1e9 * minimum(fget(tc, r, "seconds_per_expansion") for r in
        select(tc; stage = stage, form = "dense", precision = "Float64", P = FIG07_P))
    # replicated across the batch axis so the figure can draw it as a flat rule
    writewide("fig07b_cpu_reference.csv", "batch", batches,
        ["m2lz_best_cpu_dense_ns", "axisswap_best_cpu_dense_ns"],
        [fill(best("m2l_z_translation"), length(batches)),
         fill(best("axis_swap"), length(batches))];
        comment = "008c dense_vs_loop blas1, m13h-1-1, Float64 P=$FIG07_P; best (min over batch) " *
                  "single-thread CPU dense per-expansion time, replicated at every batch width " *
                  "so it plots as a reference rule")
end

function fig07c()
    for blas in ("1", "8")
        t = readtable("axis_swap/tmpfac-126-17.et.byu.edu/m2l_variants_blas$(blas).csv")
        batch = maximum(iget(t, r, "batch") for r in t.rows)
        Ps = sort(unique(iget(t, r, "P") for r in t.rows))
        get1(variant, lh, P) = fget(t, only(select(t; variant = variant, lamb_helmholtz = lh,
                                                  P = P, batch = batch)), "seconds_per_expansion")
        names = String[]
        cols = Vector{Vector{Float64}}()
        for (lh, tag) in (("false", "phi"), ("true", "lh")), v in ("materialized", "factored")
            push!(names, "$(v)_$(tag)")
            push!(cols, [get1("production_recurrence", lh, P) / get1(v, lh, P) for P in Ps])
        end
        writewide("fig07c_axisswap_speedup_blas$(blas).csv", "P", Ps, names, cols;
            comment = "015 m2l_variants, tmpfac-126-17 Apple M2, BLAS=$blas, Float64, batch=$batch; " *
                      "production recurrence / variant per-expansion time (>1 = variant wins)")
    end
end

# ---------------------------------------------------------------------------
# fig08 -- radix path vs the shipping legacy octree fmm! (023)
#   The CSVs record backend/path/n but not the Julia thread count, which is the
#   single largest non-comparability between the two paths.  Threads come from
#   the sidecar .md files and are carried in the case labels below.
# ---------------------------------------------------------------------------

# (relative CSV path, short host label, julia threads) -- threads read from the
# sidecar .md of each run, which the CSV itself omits.
const FIG08_HOSTS = (
    ("production_integration/benchmark_023_m13h-1-1_20260714-233159.csv", "m13h-1-1", 8),
    ("production_integration/benchmark_023_m13h-2-1_20260715-070804.csv", "m13h-2-1", 8),
    ("production_integration/benchmark_023_mecsrs-MacBook-Pro-188.local_20260714-231506.csv", "mac", 1),
)
const FIG08_NS = (10000, 100000)

function fig08()
    labels = String[]
    radix_host, legacy, direct, radix_gpu = Float64[], Float64[], Float64[], Float64[]
    for (relpath, host, threads) in FIG08_HOSTS
        t = readtable(relpath)
        # thread count is not in the CSV; assert the sidecar value we recorded
        side = replace(joinpath(DATA_DIR, relpath), ".csv" => ".md")
        isfile(side) || error("024a: missing 023 sidecar: $side")
        occursin("threads=$threads", read(side, String)) ||
            error("024a: $side does not record threads=$threads")
        for n in FIG08_NS
            step(backend, path) = begin
                sel = filter(r -> sget(t, r, "backend") == backend && sget(t, r, "path") == path &&
                                  iget(t, r, "n") == n && iget(t, r, "P") == 4 &&
                                  sget(t, r, "phase") == "step", t.rows)
                isempty(sel) ? NaN : fget(t, sel[1], "seconds")
            end
            push!(labels, "$(host) t=$(threads) n=$(n)")
            push!(radix_host, step("cpu", "radix"))
            push!(legacy, step("cpu", "legacy"))
            push!(direct, step("cpu", "direct"))
            push!(radix_gpu, step("gpu", "radix"))
        end
    end
    all(isfinite, radix_host) && all(isfinite, legacy) ||
        error("024a: fig08 is missing a radix or legacy step time")
    idx = collect(1:length(labels))
    writewide("fig08a_abs_step.csv", "case_index", idx,
        ["radix_host_s", "legacy_s", "direct_s", "radix_gpu_s"],
        [radix_host, legacy, direct, radix_gpu];
        comment = "023 production_integration, all three hosts, P=4 ell=4; per-step wall time. " *
                  "case order = $(join(labels, " | ")). Legacy timing includes both octree builds " *
                  "and the interaction-list build every call; the radix step excludes its one-shot " *
                  "cache construction (see fig03d). direct! is a single un-repeated timing and was " *
                  "not run at n=1e5; the Mac has no GPU.")
    ratio(v) = [v[i] > 0 && isfinite(v[i]) ? legacy[i] / v[i] : NaN for i in idx]
    writewide("fig08b_speedup_vs_legacy.csv", "case_index", idx,
        ["radix_host_x", "radix_gpu_x", "direct_x"],
        [ratio(radix_host), ratio(radix_gpu), ratio(direct)];
        comment = "023 production_integration; legacy octree fmm! step / the compared path " *
                  "(>1 = faster than the shipping default). case order = $(join(labels, " | "))")
end

# ---------------------------------------------------------------------------
# fig09 -- fixed-P end-user scaling: CPU64 and resident H200 vs CPU1 (024b)
# ---------------------------------------------------------------------------

const FIG09_NS = [1000, 3162, 10000, 31623, 100000, 316228, 1000000]
const FIG09_ALLOW_PARTIAL =
    lowercase(get(ENV, "FM024B_ALLOW_PARTIAL", "false")) in ("1", "true", "yes", "on")

function fig09()
    dir = joinpath(DATA_DIR, "cpu_gpu_scaling")
    isdir(dir) || error("024a: missing 024b campaign directory: $dir")
    failure_files = sort(filter(f -> begin
        b = basename(f)
        endswith(b, ".csv") && startswith(b, "gpu_failures_")
    end, readdir(dir; join=true)))
    files = sort(filter(f -> begin
        b = basename(f)
        endswith(b, ".csv") &&
            ((startswith(b, "cpu_") && !startswith(b, "cpu_leaf_search_")) ||
             (startswith(b, "gpu_") && !startswith(b, "gpu_failures_")))
    end, readdir(dir; join=true)))
    isempty(files) && error("024a: no 024b CSVs in $dir")

    cpu = Dict{Tuple{String,Int},NamedTuple}()
    gpu = Dict{Tuple{Int,String,Int},NamedTuple}()
    gpu_failures = Set{Tuple{Int,String,Int}}()
    for path in files
        t = readtable(relpath(path, DATA_DIR))
        "mode" in t.header || error("024a: 024b file has no mode column: $path")
        for row in t.rows
            mode = sget(t, row, "mode")
            n = iget(t, row, "n")
            seconds = fget(t, row, "step_seconds_min")
            isfinite(seconds) && seconds > 0 ||
                error("024a: invalid 024b step_seconds_min in $path at n=$n")
            if mode in ("cpu1", "cpu64")
                iget(t, row, "expansion_order") == 3 ||
                    error("024a: 024b CPU expansion_order must be 3 in $path")
                iget(t, row, "P_literature") == 4 ||
                    error("024a: 024b CPU P_literature must be 4 in $path")
                isapprox(fget(t, row, "multipole_acceptance"), 0.5; rtol=0, atol=0) ||
                    error("024a: 024b CPU multipole_acceptance must be 0.5 in $path")
                expected_threads = mode == "cpu1" ? 1 : 64
                iget(t, row, "threads") == expected_threads ||
                    error("024a: 024b $mode row has wrong thread count in $path")
                err = fget(t, row, "err_gradient_rel_rms")
                isfinite(err) && err >= 0 ||
                    error("024a: invalid 024b CPU gradient error in $path")
                key = (mode, n)
                record = (; seconds, error=err, leaf=iget(t, row, "leaf_size"),
                    checksum=sget(t, row, "reference_checksum"))
                (!haskey(cpu, key) || seconds < cpu[key].seconds) &&
                    (cpu[key] = record)
            elseif mode == "gpu"
                tf = sget(t, row, "tf")
                tf in ("Float64", "Float32") ||
                    error("024a: unexpected 024b GPU precision '$tf' in $path")
                iget(t, row, "expansion_order") == 3 ||
                    error("024a: 024b GPU expansion_order must be 3 in $path")
                iget(t, row, "P_literature") == 4 ||
                    error("024a: 024b GPU P_literature must be 4 in $path")
                ell = iget(t, row, "ell")
                expected_epsilon = 0.19542385331034917 * 2.0^(ell - 4)
                isapprox(fget(t, row, "stencil_epsilon"), expected_epsilon;
                    rtol=5eps(Float64), atol=0) ||
                    error("024a: wrong 024b GPU stencil_epsilon at n=$n ell=$ell in $path")
                err = fget(t, row, "err_gradient_rel_rms")
                isfinite(err) && err >= 0 ||
                    error("024a: invalid 024b GPU gradient error in $path")
                key = (n, tf, ell)
                record = (; seconds, error=err,
                    checksum=sget(t, row, "reference_checksum"))
                (!haskey(gpu, key) || seconds < gpu[key].seconds) &&
                    (gpu[key] = record)
            else
                error("024a: unexpected 024b mode '$mode' in $path")
            end
        end
    end

    for path in failure_files
        t = readtable(relpath(path, DATA_DIR))
        for required in ("mode", "tf", "n", "ell", "status")
            required in t.header ||
                error("024a: 024b GPU failure ledger has no $required column: $path")
        end
        for row in t.rows
            sget(t, row, "mode") == "gpu" ||
                error("024a: unexpected mode in GPU failure ledger: $path")
            # Only device-capacity failures may be published as infeasible points;
            # any other failure status is an unexplained error that must surface.
            sget(t, row, "status") == "failed_after_fallback" ||
                error("024a: GPU failure ledger has a non-capacity status " *
                      "'$(sget(t, row, "status"))' in $path; investigate that case " *
                      "instead of publishing it as hardware-infeasible")
            push!(gpu_failures,
                (iget(t, row, "n"), sget(t, row, "tf"), iget(t, row, "ell")))
        end
    end

    expected_ells(n) = n <= 3162 ? (2, 3, 4) :
        n <= 31623 ? (3, 4, 5) : (4, 5)
    expected_failures = Set((n, tf, ell)
        for (n, ells) in ((100000, (6,)), (316228, (6,)),
                          (1000000, (6, 7)))
        for tf in ("Float64", "Float32") for ell in ells)
    if !FIG09_ALLOW_PARTIAL
        length(cpu) == 14 ||
            error("024a: fig09 requires 14 unique CPU cases, found $(length(cpu))")
        length(gpu) == 36 ||
            error("024a: fig09 requires 36 unique feasible GPU cases, found $(length(gpu))")
        gpu_failures == expected_failures ||
            error("024a: fig09 GPU capacity-failure ledger mismatch: " *
                  "found $(sort!(collect(gpu_failures))), expected " *
                  "$(sort!(collect(expected_failures)))")
        for n in FIG09_NS
            haskey(cpu, ("cpu1", n)) || error("024a: fig09 missing cpu1 n=$n")
            haskey(cpu, ("cpu64", n)) || error("024a: fig09 missing cpu64 n=$n")
            for tf in ("Float64", "Float32"), ell in expected_ells(n)
                haskey(gpu, (n, tf, ell)) ||
                    error("024a: fig09 missing GPU $tf n=$n ell=$ell")
            end
        end
    end

    available_ells(n, tf) =
        [ell for ell in expected_ells(n) if haskey(gpu, (n, tf, ell))]
    ns = if FIG09_ALLOW_PARTIAL
        [n for n in FIG09_NS
            if haskey(cpu, ("cpu1", n)) && haskey(cpu, ("cpu64", n)) &&
               !isempty(available_ells(n, "Float64")) &&
               !isempty(available_ells(n, "Float32"))]
    else
        copy(FIG09_NS)
    end
    isempty(ns) && error("024a: fig09 has no comparable CPU1/CPU64/GPU particle counts")

    best_gpu(n, tf) = begin
        candidates = [(ell, gpu[(n, tf, ell)]) for ell in available_ells(n, tf)]
        candidates[argmin(last(x).seconds for x in candidates)]
    end
    best64 = [best_gpu(n, "Float64") for n in ns]
    best32 = [best_gpu(n, "Float32") for n in ns]
    baseline = [cpu[("cpu1", n)].seconds for n in ns]
    speedup_cpu64 = [baseline[i] / cpu[("cpu64", n)].seconds
        for (i, n) in enumerate(ns)]
    speedup_gpu64 = [baseline[i] / best64[i][2].seconds for i in eachindex(ns)]
    speedup_gpu32 = [baseline[i] / best32[i][2].seconds for i in eachindex(ns)]
    best_ell64 = first.(best64)
    best_ell32 = first.(best32)
    err_cpu1 = [cpu[("cpu1", n)].error for n in ns]
    err_cpu64 = [cpu[("cpu64", n)].error for n in ns]
    err_gpu64 = [best64[i][2].error for i in eachindex(ns)]
    err_gpu32 = [best32[i][2].error for i in eachindex(ns)]
    error_ratio(a, b) = max(max(a, 1e-15) / max(b, 1e-15),
                            max(b, 1e-15) / max(a, 1e-15))
    ratio64_cpu1 = error_ratio.(err_gpu64, err_cpu1)
    ratio64_cpu64 = error_ratio.(err_gpu64, err_cpu64)
    ratio32_cpu1 = error_ratio.(err_gpu32, err_cpu1)
    ratio32_cpu64 = error_ratio.(err_gpu32, err_cpu64)
    for (i, n) in enumerate(ns)
        cpu[("cpu1", n)].leaf < n && ratio64_cpu1[i] > 10 &&
            error("024a: fig09 Float64 accuracy audit failed at n=$n: " *
                  "GPU/CPU1=$(ratio64_cpu1[i])")
        cpu[("cpu64", n)].leaf < n && ratio64_cpu64[i] > 10 &&
            error("024a: fig09 Float64 accuracy audit failed at n=$n: " *
                  "GPU/CPU64=$(ratio64_cpu64[i])")
        checksums = (cpu[("cpu1", n)].checksum,
            cpu[("cpu64", n)].checksum,
            best64[i][2].checksum, best32[i][2].checksum)
        length(unique(checksums)) == 1 ||
            error("024a: fig09 reference checksum mismatch at n=$n")
    end

    writewide("fig09_speedup_vs_n.csv", "n", ns,
        ["speedup_cpu64", "speedup_gpu_f64", "speedup_gpu_f32",
         "best_ell_f64", "best_ell_f32", "err_grad_rel_cpu1",
         "err_grad_rel_cpu64", "err_grad_rel_gpu_f64", "err_grad_rel_gpu_f32",
         "error_ratio_f64_cpu1", "error_ratio_f64_cpu64",
         "error_ratio_f32_cpu1", "error_ratio_f32_cpu64"],
        [speedup_cpu64, speedup_gpu64, speedup_gpu32, best_ell64, best_ell32,
         err_cpu1, err_cpu64, err_gpu64, err_gpu32, ratio64_cpu1,
         ratio64_cpu64, ratio32_cpu1, ratio32_cpu64];
        comment = "024b $(FIG09_ALLOW_PARTIAL ? "PROVISIONAL " : "")literature P=4 " *
                  "(expansion_order=3) steady-state scaling; denominator " *
                  "is fixed-MAC legacy octree CPU1 minimum step time. CPU leaf size is manually " *
                  "searched per (n,threads); each GPU precision selects the minimum measured " *
                  "step over ell. Float64 GPU/CPU relative-gradient RMS ratios are hard-gated <=10.")

    status_path = joinpath(OUT_DIR, "fig09_status.tex")
    open(status_path, "w") do io
        if FIG09_ALLOW_PARTIAL
            println(io, "\\def\\fmfigstatusnine{\\quad\\textbf{PROVISIONAL: $(length(ns))/$(length(FIG09_NS)) sizes}}")
            println(io, "\\def\\fmfigstatusninenote{\\textbf{Provisional snapshot:} campaign jobs were still running; only particle counts with CPU1, CPU64, Float64 GPU, and Float32 GPU measurements are shown.}")
        else
            println(io, "\\def\\fmfigstatusnine{}")
            println(io, "\\def\\fmfigstatusninenote{}")
        end
    end
    push!(WRITTEN, "fig09_status.tex")
end

# ---------------------------------------------------------------------------

function main()
    mkpath(OUT_DIR)
    for f in readdir(OUT_DIR)
        rm(joinpath(OUT_DIR, f))
    end
    fig01(); fig02a(); fig02b(); fig02c(); fig02d(); fig03()
    fig04a(); fig04b(); fig04d(); fig04e()
    fig05ab(); fig05cd()
    fig06a(); fig06b(); fig06c(); fig06d()
    fig07a(); fig07b(); fig07c()
    fig08(); fig09()
    println("024a: wrote $(length(WRITTEN)) files to $(relpath(OUT_DIR, REFACTOR_DIR))")
    for f in sort(WRITTEN)
        println("  ", f)
    end
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    main()
end
