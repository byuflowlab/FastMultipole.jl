# Post-process ka_vs_cuda_proof.jl output: one markdown report from OUT.
#
#   julia ka_vs_cuda_proof_report.jl <OUT dir> [> report.md]
#
# For every (tf, case, np) it joins the ka and cuda rows of every run and reports
#   * speed: median seconds of each arm, KA/CUDA ratio for fmm and dir, and the
#     run-to-run spread of the KA fmm median (the reproducibility check)
#   * accuracy: relerr of each fmm vs the CPU Float64 reference, and the
#     ELEMENTWISE KA-vs-CUDA difference of the fmm U vectors (max and L2,
#     normalised by max|U_cpu|), computed from the dumped .bin files.
# No packages beyond Base + Printf + Statistics.

using Printf, Statistics

const OUT = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "logs", "proof")
const CSV = joinpath(OUT, "proof.csv")

function readcsv(path)
    lines = readlines(path)
    hdr = split(lines[1], ',')
    rows = NamedTuple[]
    for l in lines[2:end]
        isempty(strip(l)) && continue
        f = split(l, ',')
        d = Dict(Symbol(hdr[i]) => f[i] for i in eachindex(hdr))
        num(k) = parse(Float64, d[k])
        push!(rows, (; run=d[:run], mode=d[:mode], tf=d[:tf], case=d[:case],
                       np=parse(Int, d[:np]), ell=parse(Int, d[:ell]),
                       ncells=parse(Int, d[:ncells]), ndirect=parse(Int, d[:ndirect]),
                       nroutes=parse(Int, d[:nroutes]),
                       cpu=num(:cpu_med), fmm_min=num(:fmm_min), fmm=num(:fmm_med),
                       fmm_max=num(:fmm_max), dir_min=num(:dir_min), dir=num(:dir_med),
                       dir_max=num(:dir_max), e_max=num(:fmm_vs_cpu_max),
                       e_l2=num(:fmm_vs_cpu_l2), dir_vs_fmm=num(:dir_vs_fmm_max)))
    end
    return rows
end

readbin(path) = isfile(path) ? reinterpret(Float64, read(path)) : nothing

rows = readcsv(CSV)
runs = sort(unique(r.run for r in rows))
keys_ = sort(unique((r.tf, r.case, r.np) for r in rows); by=k -> (k[1], k[2], k[3]))

pick(tf, case, np, mode, run) = (i = findfirst(r -> r.tf == tf && r.case == case &&
    r.np == np && r.mode == mode && r.run == run, rows); i === nothing ? nothing : rows[i])

fmt(x) = isnan(x) ? "--" : @sprintf("%.4f", x)
ratio(a, b) = (isnan(a) || isnan(b) || b == 0) ? "--" : @sprintf("%.3f", a / b)
pct(a, b) = (isnan(a) || isnan(b) || b == 0) ? "--" : @sprintf("%+.1f%%", 100 * (a / b - 1))

println("# KA vs native CUDA: acceptance report\n")
println("Source: `$(CSV)`; runs: ", join(runs, ", "),
        ". Times are MEDIAN wall seconds over the timed calls; `dir` is all-pairs ",
        "with the same per-pair kernel as the FMM nearfield.\n")

for tf in unique(k[1] for k in keys_)
    println("## $(tf)\n")
    println("### Speed (run-wise medians; KA/CUDA < 1 means KA faster)\n")
    println("| case | np | ell | run | cpu s | cuda fmm s | ka fmm s | ka/cuda fmm | cuda dir s | ka dir s | ka/cuda dir | dir/fmm (KA) |")
    println("|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    for (t, case, np) in keys_
        t == tf || continue
        for run in runs
            k = pick(t, case, np, "ka", run); c = pick(t, case, np, "cuda", run)
            k === nothing && c === nothing && continue
            g(r, f) = r === nothing ? NaN : getfield(r, f)
            @printf("| %s | %d | %d | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n",
                    case, np, (k === nothing ? c : k).ell, run,
                    fmt(g(k, :cpu)), fmt(g(c, :fmm)), fmt(g(k, :fmm)),
                    pct(g(k, :fmm), g(c, :fmm)),
                    fmt(g(c, :dir)), fmt(g(k, :dir)), pct(g(k, :dir), g(c, :dir)),
                    ratio(g(k, :dir), g(k, :fmm)))
        end
    end
    println()
    println("### Run-to-run reproducibility of the KA FMM median\n")
    println("| case | np | " * join(("run $(r)" for r in runs), " | ") * " | max spread |")
    println("|---|--:|" * join(("--:" for _ in runs), "|") * "|--:|")
    for (t, case, np) in keys_
        t == tf || continue
        v = [(k = pick(t, case, np, "ka", r); k === nothing ? NaN : k.fmm) for r in runs]
        vv = filter(!isnan, v)
        spread = length(vv) >= 2 ? @sprintf("%.1f%%", 100 * (maximum(vv) / minimum(vv) - 1)) : "--"
        println("| $(case) | $(np) | " * join(fmt.(v), " | ") * " | $(spread) |")
    end
    println()
    println("### Accuracy (relative to the CPU Float64 FMM; KA-vs-CUDA is elementwise on U)\n")
    println("| case | np | run | cuda fmm max / L2 | ka fmm max / L2 | KA-CUDA max / L2 | dir vs fmm (cuda) | dir vs fmm (ka) |")
    println("|---|--:|--:|--:|--:|--:|--:|--:|")
    for (t, case, np) in keys_
        t == tf || continue
        ucpu = readbin(joinpath(OUT, "U_$(case)_$(np)_cpu.bin"))
        for run in runs
            k = pick(t, case, np, "ka", run); c = pick(t, case, np, "cuda", run)
            k === nothing && c === nothing && continue
            uk = readbin(joinpath(OUT, "U_$(case)_$(np)_ka_$(t)_run$(run)_fmm.bin"))
            uc = readbin(joinpath(OUT, "U_$(case)_$(np)_cuda_$(t)_run$(run)_fmm.bin"))
            kc = if uk !== nothing && uc !== nothing && ucpu !== nothing
                s = maximum(abs, ucpu)
                @sprintf("%.2e / %.2e", maximum(abs.(uk .- uc)) / s,
                         sqrt(sum(abs2, uk .- uc) / sum(abs2, ucpu)))
            else
                "--"
            end
            e(r) = r === nothing ? "--" : @sprintf("%.2e / %.2e", r.e_max, r.e_l2)
            d(r) = r === nothing ? "--" : @sprintf("%.2e", r.dir_vs_fmm)
            println("| $(case) | $(np) | $(run) | $(e(c)) | $(e(k)) | $(kc) | $(d(c)) | $(d(k)) |")
        end
    end
    println()
end

println("### Tree statistics (from the KA run 1 rows; the CUDA lifecycle builds the same tree)\n")
println("| tf | case | np | ell | cells | direct pairs | M2L routes | same in cuda? |")
println("|---|---|--:|--:|--:|--:|--:|---|")
for (t, case, np) in keys_
    k = pick(t, case, np, "ka", runs[1]); c = pick(t, case, np, "cuda", runs[1])
    k === nothing && continue
    same = c === nothing ? "--" :
        (k.ell == c.ell && k.ncells == c.ncells && k.ndirect == c.ndirect &&
         k.nroutes == c.nroutes) ? "yes" : "NO"
    println("| $(t) | $(case) | $(np) | $(k.ell) | $(k.ncells) | $(k.ndirect) | $(k.nroutes) | $(same) |")
end
