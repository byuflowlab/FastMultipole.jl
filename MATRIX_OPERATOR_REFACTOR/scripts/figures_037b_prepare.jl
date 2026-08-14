# 037b figure table prep (stdlib only, 024a conventions): writes the CSV
# tables backing fig12_037b_deficit_curves and fig13_037b_codesign_screen.
#   julia MATRIX_OPERATOR_REFACTOR/scripts/figures_037b_prepare.jl

const DATADIR = joinpath(@__DIR__, "..", "data", "flowvpm_gpu_campaign")
const FIGDIR = joinpath(@__DIR__, "..", "data", "figures")

function read_csv(path)
    lines = readlines(path)
    header = split(lines[1], ',')
    [Dict(string(h) => string(v) for (h, v) in zip(header, split(l, ',')))
     for l in lines[2:end] if !isempty(strip(l))]
end
num(r, k) = (v = get(r, k, ""); isempty(v) ? NaN : parse(Float64, v))

# ---- fig12: truncated-deficit U error vs rho_x, per case x n ----
curves = read_csv(joinpath(DATADIR, "fm037b_deficit_curves.csv"))
dir12 = joinpath(FIGDIR, "fig12_037b_deficit_curves"); mkpath(dir12)
for case in ("cube", "wake", "rotor"), n in ("100000", "1000000")
    rows = sort([r for r in curves if r["case"] == case && r["n"] == n],
                by=r -> num(r, "rho_t"))
    isempty(rows) && continue
    open(joinpath(dir12, "$(case)_n$(n).csv"), "w") do io
        println(io, "rho,u_deficit")
        for r in rows
            println(io, num(r, "rho_t"), ",", num(r, "u_cutoff_rel"))
        end
    end
end

# ---- fig13: co-design screen speedups vs same-job anchor ----
screen = read_csv(joinpath(DATADIR, "fm037b_screen.csv"))
decomp = read_csv(joinpath(DATADIR, "fm037b_error_decomposition.csv"))
dkey(r) = join((r["case"], r["n"], r["expansion_order"], r["ell"], r["q"],
                r["kernel"], r["rho_t"], get(r, "rho_c", "")), "|")
dsum = Dict{String,Float64}()
for r in decomp
    r["tf"] == "Float32" || continue
    dsum[dkey(r)] = num(r, "u_cutoff_rel") + num(r, "u_fmm_rel")
end
anchors = Dict(r["case"]*"|"*r["n"] => num(r, "uj_ms_median")
               for r in screen if endswith(r["label"], "_anchor"))
open(joinpath(FIGDIR, "fig13_037b_codesign_screen", "screen.csv") |>
     (p -> (mkpath(dirname(p)); p)), "w") do io
    println(io, "idx,label,case,n,kernel,speedup,gatepass,uj_ms")
    idx = 0
    for r in screen
        r["status"] == "ok" || continue
        endswith(r["label"], "_anchor") && continue
        endswith(r["label"], "_noaabb") && continue
        uj = num(r, "uj_ms_median")
        sp = anchors[r["case"]*"|"*r["n"]] / uj
        cs = get(dsum, dkey(r), NaN)
        gate = num(r, "u_rel_rms") <= 1e-3 && (isnan(cs) || cs <= 1e-3)
        idx += 1
        short = replace(replace(r["label"], "037b_" => ""), "_" => "-")
        println(io, idx, ",", short, ",", r["case"], ",", r["n"], ",",
                r["kernel"], ",", round(sp, digits=3), ",", gate ? 1 : 0, ",",
                round(uj, digits=2))
    end
end
# pass/fail split tables (plotted as separate ybar series; avoids pgfplots
# row filters, which interact badly with ybar+log axes)
let dir13 = joinpath(FIGDIR, "fig13_037b_codesign_screen")
    rows = read_csv(joinpath(dir13, "screen.csv"))
    for (name, flag) in (("pass", "1"), ("fail", "0"))
        open(joinpath(dir13, "$(name).csv"), "w") do io
            println(io, "idx,speedup")
            for r in rows
                r["gatepass"] == flag &&
                    println(io, r["idx"], ",", r["speedup"])
            end
        end
    end
end
println("fig12/fig13 tables written")
