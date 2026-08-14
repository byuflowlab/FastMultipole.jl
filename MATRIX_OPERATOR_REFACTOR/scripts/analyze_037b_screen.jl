# 037b screen analysis: join the timing screen (fm037b_screen.csv) with the
# exact decomposition mirror (fm037b_error_decomposition.csv), apply the
# pre-registered conservative-sum gate, and report same-job anchor->candidate
# speedups with pair-count and stage attribution. Stdlib only.
#
#   julia MATRIX_OPERATOR_REFACTOR/scripts/analyze_037b_screen.jl \
#         [screen.csv] [decomposition.csv]

using Printf

const DATADIR = joinpath(@__DIR__, "..", "data", "flowvpm_gpu_campaign")
screen_path = length(ARGS) >= 1 ? ARGS[1] : joinpath(DATADIR, "fm037b_screen.csv")
decomp_path = length(ARGS) >= 2 ? ARGS[2] : joinpath(DATADIR, "fm037b_error_decomposition.csv")

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

num(r, k) = (v = get(r, k, ""); isempty(v) ? NaN : parse(Float64, v))

screen = read_csv(screen_path)
decomp = isfile(decomp_path) ? read_csv(decomp_path) : Dict{String,String}[]

# accuracy-config key (ell explicit in decomp; screen may carry auto -> skip join)
dkey(r) = join((r["case"], r["n"], r["expansion_order"], r["ell"], r["q"],
                r["kernel"], r["rho_t"], get(r, "rho_c", "")), "|")
# decomposition rows exist per (config, tf); gate on the row's own tf,
# fall back to Float64 (components are precision-insensitive per 3C).
dmap = Dict{String,Dict{String,String}}()
for r in decomp
    dmap[dkey(r) * "|" * r["tf"]] = r
end
function decomp_for(r)
    k = dkey(r)
    return get(dmap, k * "|" * r["tf"], get(dmap, k * "|Float64", nothing))
end

anchors = Dict{String,Dict{String,String}}()
for r in screen
    endswith(r["label"], "_anchor") && (anchors[r["case"]*"|"*r["n"]] = r)
end

@printf("%-26s %8s %8s %7s | %9s %9s %9s %6s | %10s %10s %10s\n",
    "label", "uj_ms", "anch_ms", "speedup", "u_rms", "csum", "fmm_comp",
    "gate", "direct_prs", "tp_cand", "tp_shell")
println("-"^140)
for r in screen
    r["status"] == "ok" || (println(rpad(r["label"], 26), " STATUS=", r["status"], " ", get(r, "message", "")); continue)
    a = get(anchors, r["case"]*"|"*r["n"], nothing)
    uj = num(r, "uj_ms_median")
    anch = a === nothing ? NaN : num(a, "uj_ms_median")
    sp = anch / uj
    d = decomp_for(r)
    cut = d === nothing ? NaN : num(d, "u_cutoff_rel")
    fmm = d === nothing ? NaN : num(d, "u_fmm_rel")
    csum = cut + fmm
    urms = num(r, "u_rel_rms")
    # pre-registered gate: sampled u AND conservative sum (when decomposed)
    gate = urms <= 1e-3 && (d === nothing || csum <= 1e-3) ? "PASS" : "FAIL"
    @printf("%-26s %8.3f %8.3f %6.2fx | %9.3e %9.3e %9.3e %6s | %10.3g %10.3g %10.3g\n",
        r["label"], uj, anch, sp, urms, csum, fmm, gate,
        num(r, "direct_body_pairs"), num(r, "twopass_candidate_pairs"),
        num(r, "twopass_shell_pairs"))
end

println()
println("Stage medians (ms): label: eval | b2m m2m m2l l2l l2b (isolated; production overlaps NF with far field)")
for r in screen
    r["status"] == "ok" || continue
    @printf("%-26s %8.3f | %7.3f %7.3f %7.3f %7.3f %7.3f  ell=%s leaf_q=%s cells=%s\n",
        r["label"], num(r, "eval_ms"), num(r, "b2m_ms"), num(r, "m2m_ms"),
        num(r, "m2l_ms"), num(r, "l2l_ms"), num(r, "l2b_ms"),
        get(r, "ell", "?"), get(r, "leaf_q", "?"), get(r, "n_cells", "?"))
end
