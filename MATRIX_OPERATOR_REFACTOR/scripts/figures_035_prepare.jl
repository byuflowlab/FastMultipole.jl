#!/usr/bin/env julia
#
# Task 035 -- final-report figure data preparation (024a conventions).
#
# Reads the committed campaign CSVs under data/flowvpm_gpu_campaign/ and
# writes pgfplots-ready tables into data/figures/tables/.  Julia stdlib only;
# missing sources, unexpected labels and empty selections are hard errors.
#
# Usage:  julia MATRIX_OPERATOR_REFACTOR/scripts/figures_035_prepare.jl
#
# Tables produced:
#   fig09_ladder_f32.csv / fig09_ladder_f64.csv
#       shipped-default U/J-solve trajectory per case/scale across the
#       optimization cycles (034 shipped -> cycle1 -> cycle2 -> cycle3D).
#       Missing entries (not measured at that cycle) are written as nan.
#   fig10_stages_<point>.csv  (point in cube1e5, wake1e5, cube1e6, wake1e6)
#       isolated CUDA-event stage medians (F32): shipped-034 profile where
#       measured, the pre-3D shipped defaults (cy3d P4 anchors), and the
#       final cycle-3D auto defaults.
#   fig11_error_decomposition.csv
#       velocity cutoff/FMM error components, observed total, and the
#       conservative triangle bound per config (Float32 rows; Float64
#       differs in the last digits -- see fm035_error_decomposition.csv).

const SCRIPT_DIR = @__DIR__
const REFACTOR_DIR = normpath(joinpath(SCRIPT_DIR, ".."))
const DATA_DIR = joinpath(REFACTOR_DIR, "data")
const CAMPAIGN = joinpath(DATA_DIR, "flowvpm_gpu_campaign")
const OUT_DIR = joinpath(DATA_DIR, "figures", "tables")

struct Table
    path::String
    header::Vector{String}
    rows::Vector{Vector{String}}
end

function readtable(path::AbstractString)
    isfile(path) || error("035 figures: missing source CSV: $path")
    lines = filter(!isempty, strip.(readlines(path)))
    header = String.(split(lines[1], ','))
    rows = [String.(split(l, ',')) for l in lines[2:end]]
    for (i, r) in enumerate(rows)
        length(r) == length(header) || error(
            "035 figures: $path line $(i + 1): $(length(r)) fields vs $(length(header))")
    end
    return Table(path, header, rows)
end

function colindex(t::Table, name)
    i = findfirst(==(name), t.header)
    i === nothing && error("035 figures: $(t.path) lacks column $name")
    return i
end

"Fetch column `col` of the unique row with label `lab`; error if absent/dup."
function cell(t::Table, lab::AbstractString, col::AbstractString)
    li = colindex(t, "label")
    hits = [r for r in t.rows if r[li] == lab]
    length(hits) == 1 || error("035 figures: $(t.path): $(length(hits)) rows " *
        "labelled $lab (need exactly 1)")
    return hits[1][colindex(t, col)]
end

function writetable(name::AbstractString, header::Vector{String},
                    rows::Vector{Vector{String}})
    isempty(rows) && error("035 figures: refusing to write empty table $name")
    mkpath(OUT_DIR)
    open(joinpath(OUT_DIR, name), "w") do io
        println(io, join(header, ','))
        foreach(r -> println(io, join(r, ',')), rows)
    end
    println("wrote $(joinpath(OUT_DIR, name)) ($(length(rows)) rows)")
end

sweep = readtable(joinpath(CAMPAIGN, "fm035_sweep.csv"))
cy1 = readtable(joinpath(CAMPAIGN, "fm035_cycle1.csv"))
cy2 = readtable(joinpath(CAMPAIGN, "fm035_cycle2.csv"))
cy3d = readtable(joinpath(CAMPAIGN, "fm035_cycle3d.csv"))
decomp = readtable(joinpath(CAMPAIGN, "fm035_error_decomposition.csv"))

# ---------------------------------------------------------------- fig09 ----
# label maps: (table, label) per cycle stage; "-" = not measured then.
const POINTS = ["cube1e5", "cube1e6", "wake1e5", "wake1e6"]
ladder = Dict(
    ("cube1e5", "F32") => [(sweep, "c5_reg_auto_f32"), (cy1, "cy1_c5_auto_f32"),
        (cy2, "cy2_c5_auto_f32"), (cy3d, "cy3d_c5_p5_auto_f32")],
    ("cube1e6", "F32") => [(sweep, "c6_reg_auto_f32"), (cy1, "cy1_c6_auto_f32"),
        (cy2, "cy2_c6_auto_f32"), (cy3d, "cy3d_c6_p5_auto_f32")],
    ("wake1e5", "F32") => [(sweep, "w5_reg_auto_f32"), (cy1, "cy1_w5_auto_f32"),
        (cy2, "cy2_w5_auto_f32"), (cy3d, "cy3d_w5_p5_auto_f32")],
    ("wake1e6", "F32") => [(sweep, "w6_reg_auto_f32"), (cy1, "cy1_w6_auto_f32"),
        (cy2, "cy2_w6_auto_f32"), (cy3d, "cy3d_w6_p5_auto_f32")],
    ("cube1e5", "F64") => [(sweep, "c5_reg_auto_f64"), (cy1, "cy1_c5_auto_f64"),
        (cy2, "cy2_c5_auto_f64"), (cy3d, "cy3d_c5_p5_auto_f64")],
    ("wake1e5", "F64") => [(sweep, "w5_reg_auto_f64"), (cy1, "cy1_w5_auto_f64"),
        (cy2, "cy2_w5_auto_f64"), (cy3d, "cy3d_w5_p5_auto_f64")],
    # F64 at 1e6 was first measured under the cycle-2 defaults inside the 3D
    # job (the P4 anchors); earlier cycles are nan.
    ("cube1e6", "F64") => [nothing, nothing,
        (cy3d, "cy3d_c6_p4_anchor_f64"), (cy3d, "cy3d_c6_p5_auto_f64")],
    ("wake1e6", "F64") => [nothing, nothing,
        (cy3d, "cy3d_w6_p4_anchor_f64"), (cy3d, "cy3d_w6_p5_auto_f64")],
)
for tf in ("F32", "F64")
    rows = Vector{Vector{String}}()
    for (pi, p) in enumerate(POINTS)
        entries = ladder[(p, tf)]
        vals = [e === nothing ? "nan" : cell(e[1], e[2], "uj_ms_median")
                for e in entries]
        push!(rows, [string(pi); p; vals])
    end
    writetable("fig09_ladder_$(lowercase(tf)).csv",
        ["point_index", "point", "shipped034", "cycle1", "cycle2", "cycle3"],
        rows)
end

# ---------------------------------------------------------------- fig10 ----
const STAGES = ["b2m_ms", "m2m_ms", "m2l_ms", "l2l_ms", "l2b_ms", "refresh_ms"]
const STAGE_NAMES = ["B2M", "M2M", "M2L", "L2L", "NF+L2B", "refresh"]
stage_sets = Dict(
    "cube1e5" => [("shipped034", sweep, "c5_reg_auto_f32"),
                  ("pre3D", cy3d, "cy3d_c5_p4_anchor_f32"),
                  ("cycle3D", cy3d, "cy3d_c5_p5_auto_f32")],
    "wake1e5" => [("shipped034", sweep, "w5_reg_auto_f32"),
                  ("pre3D", cy3d, "cy3d_w5_p4_anchor_f32"),
                  ("cycle3D", cy3d, "cy3d_w5_p5_auto_f32")],
    # the 034-shipped 1e6 rows were not profiled (no stage medians recorded)
    "cube1e6" => [("pre3D", cy3d, "cy3d_c6_p4_anchor_f32"),
                  ("cycle3D", cy3d, "cy3d_c6_p5_auto_f32")],
    "wake1e6" => [("pre3D", cy3d, "cy3d_w6_p4_anchor_f32"),
                  ("cycle3D", cy3d, "cy3d_w6_p5_auto_f32")],
)
for p in POINTS
    sets = stage_sets[p]
    header = ["stage_index"; "stage"; String[s[1] for s in sets]]
    rows = Vector{Vector{String}}()
    for (si, s) in enumerate(STAGES)
        vals = String[]
        for (_, t, lab) in sets
            v = cell(t, lab, s)
            isempty(v) && error("035 figures: empty stage $s for $lab")
            push!(vals, v)
        end
        push!(rows, [string(si); STAGE_NAMES[si]; vals])
    end
    writetable("fig10_stages_$p.csv", header, rows)
end

# ---------------------------------------------------------------- fig11 ----
li = colindex(decomp, "label")
rows11 = Vector{Vector{String}}()
for r in decomp.rows
    r[colindex(decomp, "tf")] == "Float32" || continue
    cfg = r[li]                              # p4 | p5
    case = r[colindex(decomp, "case")]
    n = r[colindex(decomp, "n")]
    scale = n == "100000" ? "1e5" : n == "1000000" ? "1e6" :
        error("035 figures: unexpected n=$n in decomposition CSV")
    push!(rows11, [
        "$(case)$(scale)-$(uppercase(cfg))",
        r[colindex(decomp, "u_cutoff_rel")],
        r[colindex(decomp, "u_fmm_rel")],
        r[colindex(decomp, "u_total_rel")],
        r[colindex(decomp, "u_triangle_rel")],
    ])
end
length(rows11) == 8 || error("035 figures: expected 8 Float32 decomposition " *
    "rows, got $(length(rows11))")
sort!(rows11, by=first)
rows11 = [[string(i); r] for (i, r) in enumerate(rows11)]
writetable("fig11_error_decomposition.csv",
    ["config_index", "config", "cutoff", "fmm", "total", "bound"], rows11)

println("035 figure tables complete")
