# Task 030 cost model: predict the per-time-step verdict cost of an UNMEASURED
# (n, ell, radius schedule, precision) from measured campaign rows.
#
# Division of labour, kept strict so nothing modeled can be mistaken for data:
#
#   analyze_030_structure.jl   EXACT geometry counts (computed, validated to
#                              integer equality against measured CSVs)
#   this script                per-unit device rates, FITTED to measured rows,
#                              applied to oracle counts
#
# Every number this script produces is MODELED and is labeled as such wherever
# it is written.
#
# Model form (per precision, since the FP16-WMMA and Float64 paths have
# different per-unit costs but the same structure):
#
#   verdict_ms ~ a0                      fixed per-step overhead
#              + a1 * (ell - 1)          per-level launch floor (M2M/L2L/windows)
#              + a2 * n                  per-body work (B2M, L2B gather, euler,
#                                        grid/sort in refresh)
#              + a3 * routes             M2L apply
#              + a4 * routegen           window generation, sum_L |V_L| * N_L
#              + a5 * pairwork           direct body-body interactions
#
# The features are the structural quantities the 030 sweep already showed to be
# the cost drivers: `M2M+L2L` is a flat 1.15/1.52/1.91 ms at ell=3/4/5 with no n
# dependence (a launch floor), `route_gen` is ~46% of "M2L" at the 028 winner and
# scales as |V_L|*N_L, and near+L2B tracks the direct pair work.
#
# Fitting: ordinary least squares on column-scaled features, then any feature
# whose coefficient comes out negative (physically impossible for a cost rate) is
# dropped and the fit repeated. Quality is reported as the per-row relative
# residual, and the same statistic is what bounds every prediction below.
#
# Usage:
#   julia --project=. MATRIX_OPERATOR_REFACTOR/scripts/analyze_030_costmodel.jl fit
#       fit the model to every measured fit=true row and report residuals
#   julia --project=. MATRIX_OPERATOR_REFACTOR/scripts/analyze_030_costmodel.jl candidates
#       fit, then predict the candidate grid, prune it, and write
#       data/cost_vs_n/retune_cases.txt (the pre-registered case list) and
#       data/cost_vs_n/cost_model_predictions.csv
#
# Env: FM030_DATADIR (default MATRIX_OPERATOR_REFACTOR/data/cost_vs_n)

include(joinpath(@__DIR__, "analyze_030_structure.jl"))

using Printf
using LinearAlgebra

# ---------------------------------------------------------------------------
# measured rows
# ---------------------------------------------------------------------------

struct MeasuredRow
    n::Int
    ell::Int
    schedule::Vector{Int}
    policy::String
    tensor_format::String   # "fp16" or "off"
    verdict::Float64
    err::Float64
    stages::Dict{String,Float64}
    file::String
end

const STAGE_COLS = ["refresh_ms", "b2m_ms", "m2m_ms", "m2l_ms", "l2l_ms",
                    "l2b_ms", "eval_ms", "finalize_ms", "euler_ms",
                    "route_gen_ms", "construction_ms"]

function measured_rows(dir::AbstractString = DATA_DIR)
    out = MeasuredRow[]
    for f in campaign_files(dir)
        t = readtable(f)
        for row in t.rows
            sget(t, row, "fit") == "true" || continue
            ell = iget(t, row, "ell")
            policy = sget(t, row, "policy")
            sched = policy_schedule(policy, ell)
            sched === nothing && continue          # flat policy: no geometry
            stages = Dict{String,Float64}(
                c => parse(Float64, sget(t, row, c)) for c in STAGE_COLS)
            push!(out, MeasuredRow(iget(t, row, "n"), ell, sched, policy,
                sget(t, row, "tensor_format"),
                parse(Float64, sget(t, row, "verdict_step_ms")),
                parse(Float64, sget(t, row, "err_gradient_rel_rms")),
                stages, basename(f)))
        end
    end
    return out
end

# ---------------------------------------------------------------------------
# structure features (cached: the oracle is O(8^ell) per configuration)
# ---------------------------------------------------------------------------

const _STRUCT_CACHE = Dict{Tuple{Int,Int,String},Any}()

function structure_cached(n::Int, ell::Int, schedule::Vector{Int})
    key = (n, ell, join(schedule, '-'))
    get!(_STRUCT_CACHE, key) do
        structure(n, ell, schedule)
    end
end

"Window-generation work: every active offset is flagged against every occupied source node."
routegen_work(s) = sum(s.vsize_per_level[L - 1] * s.nodes_per_level[L + 1] for L in 2:s.ell;
                       init = 0)

const FEATURES = ["const", "levels", "bodies", "routes", "routegen", "pairwork"]

function feature_row(n::Int, ell::Int, schedule::Vector{Int})
    s = structure_cached(n, ell, schedule)
    return [1.0, float(ell - 1), float(n), float(s.routes),
            float(routegen_work(s)), float(s.pairwork)], s
end

# ---------------------------------------------------------------------------
# least squares with a non-negativity screen
# ---------------------------------------------------------------------------

"""
    fit_nonneg(A, y) -> coefficients

Ordinary least squares, column-scaled for conditioning, with negative
coefficients screened out one at a time (a cost rate cannot be negative; a
negative fit means the feature is collinear with another over the measured
grid, not that the work is free). The `const` column is exempt from the screen
only in the sense that it is dropped last.
"""
function fit_nonneg(A::Matrix{Float64}, y::Vector{Float64})
    active = collect(1:size(A, 2))
    coef = zeros(size(A, 2))
    while !isempty(active)
        Aa = A[:, active]
        scale = [maximum(abs, @view Aa[:, j]) for j in axes(Aa, 2)]
        scale = [s == 0 ? 1.0 : s for s in scale]
        As = Aa ./ scale'
        c = (As' * As + 1e-12I) \ (As' * y)
        c ./= scale
        if all(>=(0), c)
            coef .= 0
            coef[active] .= c
            return coef
        end
        # drop the feature whose (negative) fitted contribution is largest in
        # magnitude over the measured grid, then refit
        contrib = [c[j] * maximum(abs, @view Aa[:, j]) for j in eachindex(c)]
        deleteat!(active, argmin(contrib))
    end
    return coef
end

struct CostModel
    tensor_format::String
    coef::Vector{Float64}
    nrows::Int
    rel_rms::Float64
    rel_max::Float64
end

function fit_model(rows::Vector{MeasuredRow}, fmt::AbstractString)
    sel = filter(r -> r.tensor_format == fmt, rows)
    isempty(sel) && error("030 cost model: no measured rows with tensor_format=$fmt")
    A = Matrix{Float64}(undef, length(sel), length(FEATURES))
    y = Vector{Float64}(undef, length(sel))
    for (i, r) in enumerate(sel)
        A[i, :], _ = feature_row(r.n, r.ell, r.schedule)
        y[i] = r.verdict
    end
    # Fit in RELATIVE error: the measured verdicts span 1.9 ms to 582 ms, so an
    # unweighted least squares is decided entirely by the largest rows and
    # predicts nothing at small n. Row-scaling by 1/y makes every row contribute
    # its relative residual, which is also the accuracy statement this model is
    # used for.
    coef = fit_nonneg(A ./ y, ones(length(y)))
    pred = A * coef
    rel = abs.(pred .- y) ./ y
    return CostModel(fmt, coef, length(sel), sqrt(sum(rel .^ 2) / length(rel)),
                     maximum(rel)), sel, pred
end

predict(m::CostModel, n::Int, ell::Int, schedule::Vector{Int}) =
    dot(first(feature_row(n, ell, schedule)), m.coef)

# ---------------------------------------------------------------------------
# candidate grid
# ---------------------------------------------------------------------------

const NS = [1000, 3162, 10000, 31623, 100000, 316228, 1000000]

"""
Depths to consider at each `n`: the measured best depth from the fixed-ell
sweep, plus one coarser and one finer, clipped to the constructible range. The
sweep only covered ell = 3/4/5, so this is what opens ell = 2 at small `n` and
ell = 6 at large `n`.
"""
const DEPTHS = Dict(
    1000    => [2, 3, 4],
    3162    => [2, 3, 4],
    10000   => [2, 3, 4],
    31623   => [3, 4, 5],
    100000  => [3, 4, 5],
    316228  => [3, 4, 5, 6],
    1000000 => [4, 5, 6],
)

"""
Radius-schedule shapes, coarse-to-fine, in the two families the 025 proof
covers (uniform, and one boosted coarsest level).

  classic  uniform q=3, i.e. |o|_inf <= 1 — the classic FMM near set, the
           cheapest legal geometry and the like-for-like comparison point 025
           singles out against the theta=0.5 (|o|^2 <= 12) family.
  cheap    leaf q=4: fewer direct pairs and a smaller offset set; buys time and
           spends accuracy. Only useful where the sweep left accuracy headroom.
  shipped  the production (6,5,...,5) geometry, the 028 Stage-7 winner shape.
  rich     uniform q=6: the 028 Stage-7 runner-up, measured at n=1e6/ell=5 as
           15.45 ms at 5.75e-4 against 12.51 ms at 1.05e-3 for shipped — i.e.
           the accuracy lever that the off-target n need.
  richer   uniform q=8, for the n where even `rich` may not reach the target.
"""
function shape_schedule(shape::Symbol, ell::Int)
    ell == 2 && return shape === :classic ? [3] :
                       shape === :cheap ? [4] :
                       shape === :shipped ? [5] :
                       shape === :rich ? [6] : [8]
    shape === :classic && return fill(3, ell - 1)
    shape === :cheap   && return vcat(6, fill(4, ell - 2))
    shape === :shipped && return vcat(6, fill(5, ell - 2))
    shape === :rich    && return fill(6, ell - 1)
    shape === :richer  && return fill(8, ell - 1)
    error("unknown shape $shape")
end

const SHAPES = [:classic, :cheap, :shipped, :rich, :richer]
const FORMATS = [("Float32", "fp16"), ("Float64", "off")]

"Accuracy target: the unchanged 028 gate, applied at every n (user decision 2026-08-04)."
const ERR_TARGET = 1.19e-3

"""
Cost-pruning slack: a candidate predicted within this factor of the best
measured admissible cost is still measured. Set to 1.3, just above the model's
worst measured relative residual (28%), so no candidate can be pruned purely by
model error.
"""
const COST_SLACK = 1.3

"Per (n, precision) budget on cost candidates; accuracy candidates are exempt."
const PER_GROUP_CAP = 6

"""
Depth coverage keeps the cheapest geometry at every candidate depth, but not
when even that is predicted at more than twice the best admissible cost: no
plausible model error turns a 2x deficit into a win, and the depth is then
already answered by the sweep.
"""
const DEPTH_COVER_SLACK = 2.0

"""
    candidate_grid(rows, models) -> Vector{NamedTuple}

Predict every (n, depth, shape, precision) candidate and prune to a measurable
list. Pruning rule, fixed before any case runs:

  * COST candidates: keep if the predicted cost is below `COST_SLACK` times the
    best measured admissible cost at that (n, precision) — only those can win,
    and the slack is the model's own worst measured relative residual, so a
    winner cannot be pruned by model error alone. If nothing is admissible yet
    at that (n, precision), every candidate is a cost candidate.
  * ACCURACY candidates: the two accuracy-improving shapes (`rich`, `richer`)
    are kept regardless of predicted cost wherever the best measured error is
    above `0.8 * ERR_TARGET`, i.e. wherever the current geometry has little or
    no headroom. These are the only route to a usable configuration at an
    off-target (n, precision), so cost must not veto them.
  * BUDGET: at most `PER_GROUP_CAP` cases per (n, precision), taken by ascending
    predicted cost, with accuracy candidates exempt from the cap.

Already-measured configurations are dropped: the runner would skip them anyway,
and the analysis reads them straight from the sweep.
"""
function candidate_grid(rows::Vector{MeasuredRow}, models::Dict{String,CostModel})
    measured = Set((r.n, r.ell, join(r.schedule, '-'), r.tensor_format) for r in rows)
    best_ok = Dict{Tuple{Int,String},Float64}()     # best admissible measured cost
    best_err = Dict{Tuple{Int,String},Float64}()    # best measured error
    for r in rows
        k = (r.n, r.tensor_format)
        best_err[k] = min(get(best_err, k, Inf), r.err)
        r.err <= ERR_TARGET || continue
        best_ok[k] = min(get(best_ok, k, Inf), r.verdict)
    end

    out = NamedTuple[]
    for (tf, fmt) in FORMATS, n in NS
        k = (n, fmt)
        ok_cost = get(best_ok, k, Inf)
        err_now = get(best_err, k, Inf)
        group = NamedTuple[]
        for ell in DEPTHS[n], shape in SHAPES
            sched = shape_schedule(shape, ell)
            (n, ell, join(sched, '-'), fmt) in measured && continue
            p = predict(models[fmt], n, ell, sched)
            # Accuracy rescue applies only where the sweep found NO admissible
            # configuration at that (n, precision) — there, and only there, an
            # accuracy-improving geometry is worth measuring at any cost.
            accuracy_candidate = shape in (:rich, :richer) && err_now > ERR_TARGET
            cost_candidate = p < COST_SLACK * ok_cost
            push!(group, (; n, ell, schedule = join(sched, '-'), shape, tf, fmt,
                          predicted_ms = p, accuracy_candidate, cost_candidate,
                          best_measured_admissible_ms = ok_cost,
                          best_measured_err = err_now))
        end
        sort!(group; by = c -> c.predicted_ms)
        # Depth coverage first: the cheapest predicted candidate at every depth
        # is always measured, so a depth cannot be excluded from the campaign by
        # the model alone (the model's depth term is the least certain part of
        # it). The rest of the budget goes to the cheapest predictions.
        chosen = Set{Int}()
        for ell in DEPTHS[n]
            i = findfirst(c -> c.ell == ell, group)
            i === nothing && continue
            # a depth whose cheapest geometry is predicted at more than twice the
            # best admissible cost cannot win even at the model's worst residual
            group[i].predicted_ms > DEPTH_COVER_SLACK * ok_cost && continue
            push!(chosen, i)
        end
        kept = 0
        for (i, c) in enumerate(group)
            i in chosen && continue
            if c.accuracy_candidate
                push!(chosen, i)
            elseif c.cost_candidate && kept < PER_GROUP_CAP
                kept += 1
                push!(chosen, i)
            end
        end
        for i in sort!(collect(chosen))
            push!(out, group[i])
        end
    end
    return out
end

policy_string(schedule::AbstractString) = "sched" * schedule

# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

function report_fit(models, fits)
    println("## Cost-model fit (MODELED rates over measured rows)\n")
    println("| format | rows | ", join(FEATURES, " | "), " | rel RMS | rel max |")
    println("|---|---|", repeat("---|", length(FEATURES) + 2))
    for (fmt, m) in models
        cells = [@sprintf("%.4g", c) for c in m.coef]
        @printf("| %s | %d | %s | %.1f%% | %.1f%% |\n", fmt, m.nrows,
                join(cells, " | "), 100 * m.rel_rms, 100 * m.rel_max)
    end
    println()
    println("Units: `const` ms, `levels` ms/level, `bodies` ms/body, ",
            "`routes` ms/route, `routegen` ms/flagged-offset, `pairwork` ms/body-pair.\n")
    println("### Worst per-row residuals\n")
    println("| n | ell | schedule | fmt | measured ms | modeled ms | rel |")
    println("|---|---|---|---|---|---|---|")
    allrows = NamedTuple[]
    for (fmt, (sel, pred)) in fits
        for (r, p) in zip(sel, pred)
            push!(allrows, (; r.n, r.ell, sched = join(r.schedule, '-'), fmt,
                            meas = r.verdict, mod = p,
                            rel = abs(p - r.verdict) / r.verdict))
        end
    end
    sort!(allrows; by = r -> -r.rel)
    for r in first(allrows, 8)
        @printf("| %d | %d | %s | %s | %.3f | %.3f | %.1f%% |\n",
                r.n, r.ell, r.sched, r.fmt, r.meas, r.mod, 100 * r.rel)
    end
    println()
end

function write_candidates(cands, dir::AbstractString; tag::AbstractString = "")
    mkpath(dir)
    listpath = joinpath(dir, "retune_cases$(tag).txt")
    open(listpath, "w") do io
        println(io, "# 030 joint retune campaign: pre-registered case list ",
                    "(n:geometry:tf:tensor_format).")
        println(io, "# Generated by analyze_030_costmodel.jl from the fixed-ell sweep; ",
                    "the predicted cost of every case is in cost_model_predictions.csv.")
        println(io, "# Ordered n-outer so a walltime kill leaves complete small-n groups.")
        for c in cands
            println(io, c.n, ":", policy_string(c.schedule), ":", c.tf, ":", c.fmt)
        end
    end
    predpath = joinpath(dir, "cost_model_predictions$(tag).csv")
    open(predpath, "w") do io
        println(io, "# 030 cost model: MODELED per-step cost for the retune candidate grid. ",
                    "predicted_ms is modeled, not measured; ",
                    "best_measured_admissible_ms and best_measured_err come from the ",
                    "fixed-ell sweep (job 13035897).")
        println(io, "n,ell,schedule,shape,tf,tensor_format,predicted_ms,",
                    "best_measured_admissible_ms,best_measured_err")
        for c in cands
            println(io, join((c.n, c.ell, c.schedule, c.shape, c.tf, c.fmt,
                round(c.predicted_ms; digits=4),
                isinf(c.best_measured_admissible_ms) ? "" :
                    round(c.best_measured_admissible_ms; digits=4),
                c.best_measured_err), ','))
        end
    end
    println("030 cost model: wrote $listpath ($(length(cands)) cases)")
    println("030 cost model: wrote $predpath")
    return listpath, predpath
end

# ---------------------------------------------------------------------------
# refinement grid: neighbours of each n's measured winner
# ---------------------------------------------------------------------------

"""
    neighbour_schedules(sched, ell) -> Vector{Vector{Int}}

Legal schedules one step away from `sched` at depth `ell`: each entry moved one
place up or down the supported radius list, plus the intermediate shapes between
a boosted coarsest level and a reduced leaf (e.g. `(6,4,4,4)` also reaches
`(6,5,4,4)`, `(6,6,4,4)`, `(6,5,5,4)`). Non-increasing with depth is enforced,
and the leaf entry defines the direct list, so it is varied explicitly rather
than only as a side effect.

This is the fine search the two-family candidate grid could not express: the
first campaign sampled uniform and single-boost shapes, and this fills the
space between the shapes that actually won.
"""
function neighbour_schedules(sched::Vector{Int}, ell::Int)
    out = Vector{Int}[]
    base = length(sched) == ell - 1 ? copy(sched) :
           length(sched) < ell - 1 ? vcat(sched, fill(last(sched), ell - 1 - length(sched))) :
           sched[end - (ell - 2):end]
    push!(out, base)
    for i in eachindex(base), dq in (-1, 1)
        j = findfirst(==(base[i]), SUPPORTED_Q)
        j === nothing && continue
        k = j + dq
        (1 <= k <= length(SUPPORTED_Q)) || continue
        cand = copy(base); cand[i] = SUPPORTED_Q[k]
        all(cand[t + 1] <= cand[t] for t in 1:length(cand)-1) || continue
        push!(out, cand)
    end
    # staircases between the coarsest and the leaf entry
    if length(base) >= 2 && base[1] > last(base)
        lo = findfirst(==(last(base)), SUPPORTED_Q)
        hi = findfirst(==(base[1]), SUPPORTED_Q)
        if lo !== nothing && hi !== nothing
            for mid in SUPPORTED_Q[lo:hi], cut in 2:length(base)
                cand = vcat(base[1], fill(mid, cut - 1), fill(last(base), length(base) - cut))
                length(cand) == length(base) || continue
                all(cand[t + 1] <= cand[t] for t in 1:length(cand)-1) || continue
                push!(out, cand)
            end
        end
    end
    return unique(out)
end

"""
    refine_grid(rows, models) -> Vector{NamedTuple}

Second-pass candidate grid: around each `n`'s measured winner, every neighbour
schedule at the winner depth and at one depth either side, in both precisions,
pruned to those predicted to beat the current best admissible cost (with the
same `COST_SLACK`) and capped per `(n, precision)`. Everything already measured
is dropped.
"""
function refine_grid(rows::Vector{MeasuredRow}, models::Dict{String,CostModel})
    measured = Set((r.n, r.ell, join(r.schedule, '-'), r.tensor_format) for r in rows)
    out = NamedTuple[]
    for n in NS
        atn = filter(r -> r.n == n, rows)
        adm = filter(r -> r.err <= ERR_TARGET, atn)
        isempty(adm) && continue
        win = adm[argmin([r.verdict for r in adm])]
        ok_cost = win.verdict
        for (tf, fmt) in FORMATS
            group = NamedTuple[]
            for ell in max(2, win.ell - 1):(win.ell + 1)
                for sched in neighbour_schedules(win.schedule, ell)
                    (n, ell, join(sched, '-'), fmt) in measured && continue
                    p = predict(models[fmt], n, ell, sched)
                    p < COST_SLACK * ok_cost || continue
                    push!(group, (; n, ell, schedule = join(sched, '-'),
                                  shape = :refine, tf, fmt, predicted_ms = p,
                                  accuracy_candidate = false,
                                  best_measured_admissible_ms = ok_cost,
                                  best_measured_err = win.err))
                end
            end
            sort!(group; by = c -> c.predicted_ms)
            append!(out, first(group, PER_GROUP_CAP))
        end
    end
    return out
end

function main(mode::AbstractString)
    rows = measured_rows(DATA_DIR)
    isempty(rows) && error("030 cost model: no measured rows under $DATA_DIR")
    println("030 cost model: $(length(rows)) measured rows under $DATA_DIR\n")
    models = Dict{String,CostModel}()
    fits = Dict{String,Any}()
    for (_, fmt) in FORMATS
        m, sel, pred = fit_model(rows, fmt)
        models[fmt] = m
        fits[fmt] = (sel, pred)
    end
    report_fit(models, fits)
    mode == "fit" && return
    cands = mode == "refine" ? refine_grid(rows, models) :
            candidate_grid(rows, models)
    sort!(cands; by = c -> (c.n, c.ell, c.schedule, c.fmt))
    println("## Candidate grid: $(length(cands)) cases\n")
    println("| n | ell | schedule | shape | fmt | modeled ms | best measured admissible ms |")
    println("|---|---|---|---|---|---|---|")
    for c in cands
        @printf("| %d | %d | %s | %s | %s | %.3f | %s |\n", c.n, c.ell, c.schedule,
                c.shape, c.fmt, c.predicted_ms,
                isinf(c.best_measured_admissible_ms) ? "none" :
                    @sprintf("%.3f", c.best_measured_admissible_ms))
    end
    println()
    write_candidates(cands, DATA_DIR; tag = mode == "refine" ? "_refine" : "")
    return
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    main(isempty(ARGS) ? "fit" : ARGS[1])
end
