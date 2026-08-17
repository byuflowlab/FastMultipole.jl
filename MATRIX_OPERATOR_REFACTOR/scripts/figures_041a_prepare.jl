# figures_041a_prepare.jl — task 041a figure data preparation (stdlib only).
#
# Reads the 041a CSVs of record:
#   data/fm041a_gpu_widen.csv      (job 13184013)
#   data/fm041a_pweep.csv          (job 13184013)
#   data/fm041a_leafpop.csv        (job 13184013)
#   data/fm041a_gpu_contrast.csv   (job 13184014)
#   data/fm041a_contrast_leafpop.csv (job 13184014)
#   data/fm041a_gpu_stages.csv     (job 13184014)
#   data/fm041a_gpu_sigma.csv      (job 13184014)
#   data/fm041a_host_widen.csv     (job 13184015)
# and the 039-041 CSVs of record where cited (fm040_lifecycle_cost.csv for
# the host ell{5,6}/K{64,128} record rows; ratios always use same-job rows).
#
# Emits one tidy pgfplots table per series into the per-figure data
# directories fig14_adaptive_contrast/ ... fig20_adaptive_sigma/ under
# data/figures/, plus figures_041a_summary.txt with the headline numbers
# quoted by the report. Errors out on missing sources or unexpected headers.
#
# Usage: julia MATRIX_OPERATOR_REFACTOR/scripts/figures_041a_prepare.jl

const DATA = normpath(joinpath(@__DIR__, "..", "data"))
const FIGS = joinpath(DATA, "figures")

function read_csv(path)
    isfile(path) || error("missing source: $path")
    lines = readlines(path)
    isempty(lines) && error("empty source: $path")
    header = split(lines[1], ",")
    rows = Vector{Dict{String,String}}()
    for ln in lines[2:end]
        isempty(strip(ln)) && continue
        vals = split(ln, ",")
        d = Dict{String,String}()
        for (i, h) in enumerate(header)
            d[h] = i <= length(vals) ? String(vals[i]) : ""
        end
        push!(rows, d)
    end
    return rows
end

num(r, k) = parse(Float64, r[k])
isok(r) = r["status"] == "ok"
getf(r, k, default=NaN) = (v = get(r, k, ""); isempty(v) || startswith(v, "fail") ?
    default : tryparse(Float64, v) === nothing ? default : parse(Float64, v))

function write_table(dir, name, header, rows)
    mkpath(dir)
    open(joinpath(dir, name), "w") do io
        println(io, header)
        for r in rows
            println(io, r)
        end
    end
end

fmt(x) = x === NaN ? "nan" : x == round(x) && abs(x) < 1e15 ?
    string(Int(round(x))) : string(x)

const SUMMARY = String[]
note(s) = (push!(SUMMARY, s); println(s))

#--- load sources ---#
widen = read_csv(joinpath(DATA, "fm041a_gpu_widen.csv"))
pweep = read_csv(joinpath(DATA, "fm041a_pweep.csv"))
contrast = read_csv(joinpath(DATA, "fm041a_gpu_contrast.csv"))
cleaf = read_csv(joinpath(DATA, "fm041a_contrast_leafpop.csv"))
stages = read_csv(joinpath(DATA, "fm041a_gpu_stages.csv"))
sigma = read_csv(joinpath(DATA, "fm041a_gpu_sigma.csv"))
hostw = read_csv(joinpath(DATA, "fm041a_host_widen.csv"))

const CASES = ["unitcube", "wake", "multiscale100"]

#=========== fig14: time vs cluster contrast (GPU, n=1e6, F64 dense) ==========#
let dir = joinpath(FIGS, "fig14_adaptive_contrast")
    clist = sort(unique(num(r, "contrast") for r in contrast))
    for (structure, params) in (("uniform", [4, 5, 6, 7]), ("adaptive", [64]))
        for p in params
            rows = String[]
            for c in clist
                sel = [r for r in contrast if num(r, "contrast") == c &&
                    r["structure"] == structure && num(r, "param") == p]
                @assert length(sel) == 1 "fig14: expected 1 row for c=$c $structure $p"
                r = sel[1]
                t = isok(r) ? num(r, "t_step_ms") : NaN
                rel = isok(r) ? num(r, "vel_rel_rms") : NaN
                pm = isok(r) ? num(r, "popmax") : NaN
                mem = isok(r) ? num(r, "mem_gb") : NaN
                push!(rows, join(fmt.([c, t, rel, pm, mem]), ","))
            end
            write_table(dir, "$(structure)_$(p).csv",
                "contrast,t_step_ms,vel_rel_rms,popmax,mem_gb", rows)
        end
    end
    # best-uniform envelope per contrast (gate-passing rows only)
    rows = String[]
    for c in clist
        sel = [r for r in contrast if num(r, "contrast") == c &&
            r["structure"] == "uniform" && isok(r) && num(r, "vel_rel_rms") <= 1e-3]
        if isempty(sel)
            push!(rows, join(fmt.([c, NaN, NaN]), ","))
        else
            r = sel[argmin([num(r, "t_step_ms") for r in sel])]
            push!(rows, join(fmt.([c, num(r, "t_step_ms"), num(r, "param")]), ","))
        end
    end
    write_table(dir, "uniform_best.csv", "contrast,t_step_ms,ell", rows)
    # headline: adaptive vs best uniform at extremes
    for c in (clist[1], 100.0, clist[end])
        ad = [r for r in contrast if num(r, "contrast") == c &&
            r["structure"] == "adaptive" && isok(r)]
        un = [r for r in contrast if num(r, "contrast") == c &&
            r["structure"] == "uniform" && isok(r) && num(r, "vel_rel_rms") <= 1e-3]
        if !isempty(ad) && !isempty(un)
            tu = minimum(num(r, "t_step_ms") for r in un)
            ta = num(ad[1], "t_step_ms")
            note("fig14 c=$c: adaptive $(ta) ms vs best-uniform $(tu) ms -> $(round(tu / ta; digits=2))x")
        end
    end
end

#============ fig15: time vs n per case (GPU widen + host widen) ==============#
let dir = joinpath(FIGS, "fig15_adaptive_time_vs_n")
    nlist = sort(unique(num(r, "n") for r in widen))
    for case in CASES
        for (label, pred) in (
            ("gpu_adaptive", r -> r["structure"] == "adaptive"),
            ("gpu_uniform", r -> r["structure"] == "uniform"))
            rows = String[]
            for n in nlist
                sel = [r for r in widen if r["case"] == case && num(r, "n") == n &&
                    r["tf"] == "Float64" && pred(r) && isok(r) &&
                    num(r, "vel_rel_rms") <= 1e-3]
                isempty(sel) && (push!(rows, join(fmt.([n, NaN, NaN]), ",")); continue)
                r = sel[argmin([num(r, "t_step_ms") for r in sel])]
                push!(rows, join(fmt.([n, num(r, "t_step_ms"), num(r, "param")]), ","))
            end
            write_table(dir, "$(case)_$(label).csv", "n,t_step_ms,param", rows)
        end
        for (label, structure) in (("host_adaptive", "adaptive"),
                                   ("host_uniform", "uniform"))
            rows = String[]
            for n in sort(unique(num(r, "n") for r in hostw))
                sel = [r for r in hostw if r["case"] == case && num(r, "n") == n &&
                    r["structure"] == structure && isok(r) &&
                    num(r, "vel_rel_rms") <= 1e-3]
                isempty(sel) && continue
                r = sel[argmin([num(r, "t_step_ms") for r in sel])]
                push!(rows, join(fmt.([n, num(r, "t_step_ms"), num(r, "param")]), ","))
            end
            write_table(dir, "$(case)_$(label).csv", "n,t_step_ms,param", rows)
        end
        # headline at n=1e6 (GPU best-vs-best F64)
        for (plat, src) in (("GPU", widen), ("host", hostw))
            ad = [r for r in src if r["case"] == case && num(r, "n") == 1e6 &&
                get(r, "tf", "Float64") == "Float64" && r["structure"] == "adaptive" &&
                isok(r) && num(r, "vel_rel_rms") <= 1e-3]
            un = [r for r in src if r["case"] == case && num(r, "n") == 1e6 &&
                get(r, "tf", "Float64") == "Float64" && r["structure"] == "uniform" &&
                isok(r) && num(r, "vel_rel_rms") <= 1e-3]
            if !isempty(ad) && !isempty(un)
                ta, ka = findmin([num(r, "t_step_ms") for r in ad])
                tu, ku = findmin([num(r, "t_step_ms") for r in un])
                note("fig15 $case $plat n=1e6: adaptive best $(ta) ms (param=$(ad[ka]["param"])) vs uniform best $(tu) ms (ell=$(un[ku]["param"])) -> $(round(tu / ta; digits=2))x")
            end
        end
    end
end

#=================== fig16: per-stage breakdown (GPU n=1e6) ===================#
let dir = joinpath(FIGS, "fig16_adaptive_stages")
    rows = String[]
    idx = 0
    for case in CASES
        for r in stages
            r["case"] == case || continue
            isok(r) || continue
            idx += 1
            lbl = r["structure"] == "adaptive" ? "adapt-K$(r["param"])" :
                "unif-l$(r["param"])"
            push!(rows, join([string(idx), "$(case) $(lbl)",
                fmt.([getf(r, "t_b2m_ms"), getf(r, "t_m2m_ms"), getf(r, "t_m2l_ms"),
                    getf(r, "t_s2l_ms"), getf(r, "t_l2l_ms"), getf(r, "t_near_ms"),
                    getf(r, "t_l2b_ms"), getf(r, "t_m2t_ms"),
                    getf(r, "t_update_warm_ms"),
                    getf(r, "t_life_graph_overlap_ms"),
                    getf(r, "t_life_nograph_serial_ms")])...], ","))
        end
    end
    write_table(dir, "stages.csv",
        "idx,label,b2m,m2m,m2l,s2l,l2l,near,l2b,m2t,update,life_shipped,life_serial",
        rows)
end

#============= fig17: memory vs depth/K and vs contrast (device) ==============#
let dir = joinpath(FIGS, "fig17_adaptive_memory")
    for case in CASES
        for structure in ("uniform", "adaptive")
            rows = String[]
            sel = [r for r in widen if r["case"] == case && num(r, "n") == 1e6 &&
                r["tf"] == "Float64" && r["structure"] == structure]
            for r in sort(sel; by=r -> num(r, "param"))
                mem = isok(r) ? num(r, "mem_gb") : NaN
                push!(rows, join(fmt.([num(r, "param"), mem]), ","))
            end
            write_table(dir, "$(case)_$(structure).csv", "param,mem_gb", rows)
        end
    end
    # note failed uniform depths (capacity/OOM) as data
    for r in widen
        if !isok(r) && r["structure"] == "uniform"
            note("fig17 FAIL row: $(r["case"]) n=$(r["n"]) $(r["tf"]) ell=$(r["param"]): $(r["status"])")
        end
    end
end

#================= fig18: accuracy-cost frontier (P + geometry) ===============#
let dir = joinpath(FIGS, "fig18_adaptive_accuracy_cost")
    for case in ("wake", "multiscale100")
        # P-sweep curves per config (P in {2,3,4,6,8}; P=4 from widen same-job?
        # the pweep job re-ran P=4? No: P=4 rows come from the widen grid rows
        # of the SAME job 13184013 (same process, same anchors).
        for (label, structure, param) in (("adaptive_k64", "adaptive", 64),
                                          ("uniform_l6", "uniform", 6),
                                          ("uniform_l7", "uniform", 7))
            rows = Tuple{Float64,Float64,Float64}[]
            for r in pweep
                r["case"] == case || continue
                r["structure"] == structure && num(r, "param") == param || continue
                isok(r) || continue
                push!(rows, (num(r, "P"), num(r, "t_step_ms"), num(r, "vel_rel_rms")))
            end
            for r in widen
                r["case"] == case || continue
                num(r, "n") == 1e6 && r["tf"] == "Float64" || continue
                r["structure"] == structure && num(r, "param") == param || continue
                isok(r) || continue
                push!(rows, (num(r, "P"), num(r, "t_step_ms"), num(r, "vel_rel_rms")))
            end
            sort!(rows)
            write_table(dir, "$(case)_$(label).csv", "P,t_step_ms,vel_rel_rms",
                [join(fmt.(collect(t)), ",") for t in rows])
        end
        # geometry sweep at P=4 (all params, both structures) as scatter
        for structure in ("adaptive", "uniform")
            rows = String[]
            for r in widen
                r["case"] == case || continue
                num(r, "n") == 1e6 && r["tf"] == "Float64" || continue
                r["structure"] == structure && isok(r) || continue
                push!(rows, join(fmt.([num(r, "param"), num(r, "t_step_ms"),
                    num(r, "vel_rel_rms")]), ","))
            end
            write_table(dir, "$(case)_geom_$(structure).csv",
                "param,t_step_ms,vel_rel_rms", rows)
        end
    end
end

#================ fig19: leaf-population CCDF (contrast case) =================#
let dir = joinpath(FIGS, "fig19_adaptive_leafpop")
    for c in (100.0, 1000.0)
        for (structure, param) in (("adaptive", 64), ("uniform", 5), ("uniform", 6))
            sel = [r for r in cleaf if num(r, "contrast") == c &&
                r["structure"] == structure && num(r, "param") == param]
            isempty(sel) && continue
            pops = [(num(r, "pop"), num(r, "count")) for r in sel]
            sort!(pops)
            total = sum(p[2] for p in pops)
            # CCDF: fraction of leaves with population >= pop
            rows = String[]
            remaining = total
            for (p, cnt) in pops
                push!(rows, join(fmt.([p, remaining / total]), ","))
                remaining -= cnt
            end
            write_table(dir, "c$(Int(c))_$(structure)_$(param).csv",
                "pop,ccdf", rows)
        end
    end
end

#===================== fig20: sigma-heterogeneous variant =====================#
let dir = joinpath(FIGS, "fig20_adaptive_sigma")
    slist = sort(unique(num(r, "spread") for r in sigma))
    for (structure, params) in (("adaptive", [64]), ("uniform", [2, 3, 4, 5, 6]))
        for p in params
            rows = String[]
            for s in slist
                sel = [r for r in sigma if num(r, "spread") == s &&
                    r["structure"] == structure && num(r, "param") == p]
                @assert length(sel) == 1 "fig20: expected 1 row for s=$s $structure $p"
                r = sel[1]
                t = isok(r) ? num(r, "t_step_ms") : NaN
                rel = isok(r) ? num(r, "vel_rel_rms") : NaN
                gate = isok(r) && num(r, "vel_rel_rms") <= 1e-3 ? 1.0 : 0.0
                push!(rows, join(fmt.([s, t, rel, gate]), ","))
            end
            write_table(dir, "$(structure)_$(p).csv",
                "spread,t_step_ms,vel_rel_rms,gate_pass", rows)
        end
    end
    # overlay: every ok row that misses the 1e-3 velocity gate (hollow marks)
    gf = String[]
    for r in sigma
        if isok(r) && num(r, "vel_rel_rms") > 1e-3
            push!(gf, join(fmt.([num(r, "spread"), num(r, "t_step_ms")]), ","))
        end
    end
    isempty(gf) && push!(gf, "nan,nan")
    write_table(dir, "gatefail.csv", "spread,t_step_ms", gf)
    for r in sigma
        isok(r) || note("fig20 FAIL row: spread=$(r["spread"]) $(r["structure"]) " *
            "param=$(r["param"]): $(first(r["status"], 100))")
    end
    for r in sigma
        if isok(r) && num(r, "vel_rel_rms") > 1e-3
            note("fig20 GATE-FAIL row: spread=$(r["spread"]) $(r["structure"]) " *
                "param=$(r["param"]) rel=$(r["vel_rel_rms"])")
        end
    end
end

open(joinpath(FIGS, "figures_041a_summary.txt"), "w") do io
    for s in SUMMARY
        println(io, s)
    end
end
println("figures_041a_prepare: done")
