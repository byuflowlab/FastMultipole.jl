# 037e/037f screen analysis: same-job anchor -> candidate speedups, the
# pre-registered promotion-gate checks, 037e pair-AABB telemetry, and (for
# 037f) the oracle budget check of each mode's delivered-error delta against
# the signed-off allowance table. Stdlib only.
#
#   julia MATRIX_OPERATOR_REFACTOR/scripts/analyze_037ef_screen.jl \
#         <screen.csv> [decomposition.csv]
#
# Screen anchors are keyed (case, n, tf) — every candidate row is compared to
# the shipped-default anchor of its own block (the 037b convention, refined
# per-precision). Promotion gate (identical for both rows; any default change
# additionally needs explicit user approval): u_rel_rms <= 1e-3 everywhere;
# >= 5% end-to-end U/J speedup on a material wake or rotor case; no > 3%
# regression on any other measured case.

using Printf

const DATADIR = joinpath(@__DIR__, "..", "data", "flowvpm_gpu_campaign")
screen_path = length(ARGS) >= 1 ? ARGS[1] : joinpath(DATADIR, "fm037e_screen.csv")
decomp_path = length(ARGS) >= 2 ? ARGS[2] : ""

# per-case delivered-error allowances (theory/nearfield-kernel-cheapening-
# budget.md §1, data/kernel_splitting/fm037f_budget.csv): (1e-3 - u_total)/1.1
const BUDGET = Dict(
    "cube|100000" => 2.904e-4, "cube|1000000" => 2.656e-4,
    "wake|100000" => 6.092e-4, "wake|1000000" => 6.373e-4,
    "rotor|100000" => 3.258e-4, "rotor|1000000" => 2.758e-4)

function read_csv(path)
    lines = readlines(path)
    header = split(lines[1], ',')
    rows = Dict{String,String}[]
    for line in lines[2:end]
        isempty(strip(line)) && continue
        f = split(line, ',')
        length(f) == length(header) || error("ragged row in $path: $line")
        push!(rows, Dict(string(h) => string(v) for (h, v) in zip(header, f)))
    end
    return rows
end

num(r, k) = (v = get(r, k, ""); isempty(v) || v == "n/a" ? NaN : parse(Float64, v))

screen = read_csv(screen_path)
decomp = isempty(decomp_path) ? Dict{String,String}[] :
    (isfile(decomp_path) ? read_csv(decomp_path) : Dict{String,String}[])

akey(r) = join((r["case"], r["n"], r["tf"]), "|")
anchors = Dict{String,Dict{String,String}}()
for r in screen
    endswith(r["label"], "_anchor") && r["status"] == "ok" && (anchors[akey(r)] = r)
end

# decomposition: index by (case, n, tf, gh_mode); anchor mode = shipped
dmap = Dict{String,Dict{String,String}}()
for r in decomp
    dmap[join((r["case"], r["n"], r["tf"], get(r, "gh_mode", "shipped")), "|")] = r
end

println("screen: $screen_path")
@printf("%-24s %-6s %8s %8s %8s | %9s %9s %6s | %s\n",
    "label", "tf", "uj_ms", "anch_ms", "speedup", "u_rms", "anchor_u", "gate",
    "telemetry (mode / paabb tested:skipped)")
println("-"^130)
material = Float64[]          # wake/rotor candidate speedups
regressions = Tuple{String,Float64}[]
for r in screen
    if r["status"] != "ok"
        println(rpad(r["label"], 24), " STATUS=", r["status"], " ",
            first(get(r, "message", ""), 80))
        continue
    end
    endswith(r["label"], "_anchor") && continue
    a = get(anchors, akey(r), nothing)
    uj = num(r, "uj_ms_median")
    anch = a === nothing ? NaN : num(a, "uj_ms_median")
    sp = anch / uj
    urms = num(r, "u_rel_rms")
    aurms = a === nothing ? NaN : num(a, "u_rel_rms")
    gate = urms <= 1e-3 ? "PASS" : "FAIL"
    tel = get(r, "gh_mode", "shipped") != "shipped" ?
        "gh_mode=" * r["gh_mode"] :
        (get(r, "pair_aabb", "false") in ("true", "1") ?
            @sprintf("paabb %s:%s of %s", get(r, "pair_aabb_tested", "?"),
                get(r, "pair_aabb_skipped", "?"), get(r, "nf_mixed_pairs", "?")) : "")
    @printf("%-24s %-6s %8.3f %8.3f %7.3fx | %9.3e %9.3e %6s | %s\n",
        r["label"], first(r["tf"] == "Float32" ? "F32" : "F64", 3), uj, anch,
        sp, urms, aurms, gate, tel)
    if !isnan(sp)
        r["case"] in ("wake", "rotor") && push!(material, sp)
        sp < 1.0 && push!(regressions, (r["label"], sp))
    end
end

println()
if !isempty(material)
    best = maximum(material)
    @printf("promotion: best material (wake/rotor) speedup = %.2f%% (need >= 5%%)\n",
        (best - 1) * 100)
end
worst = isempty(regressions) ? nothing :
    regressions[argmin(last.(regressions))]
if worst === nothing
    println("promotion: no regressions")
else
    @printf("promotion: worst regression = %s at %.2f%% (limit 3%%)\n",
        worst[1], (1 - worst[2]) * 100)
end

if !isempty(decomp)
    println()
    println("oracle budget check (candidate u_total - shipped-anchor u_total <= allowance):")
    @printf("%-24s %-6s %10s %10s %10s %10s %6s\n",
        "label", "tf", "u_total", "anchor", "delta", "allow", "ok")
    for r in decomp
        get(r, "gh_mode", "shipped") == "shipped" && continue
        a = get(dmap, join((r["case"], r["n"], r["tf"], "shipped"), "|"), nothing)
        ut = num(r, "u_total_rel")
        au = a === nothing ? NaN : num(a, "u_total_rel")
        delta = ut - au
        allow = get(BUDGET, r["case"] * "|" * r["n"], NaN)
        ok = !isnan(delta) && delta <= allow ? "OK" : "CHECK"
        @printf("%-24s %-6s %10.3e %10.3e %10.3e %10.3e %6s\n",
            r["label"], first(r["tf"] == "Float32" ? "F32" : "F64", 3),
            ut, au, delta, allow, ok)
    end
end
