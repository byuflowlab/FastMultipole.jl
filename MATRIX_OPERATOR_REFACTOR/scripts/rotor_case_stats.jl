# Task 037b: geometry/occupancy statistics for the DJI 9443 rotor-wake case,
# with the existing wake-cylinder case at the same n for contrast (multi-scale
# density evidence for task 038). Stdlib only + the 033 common include.
#
# Writes MATRIX_OPERATOR_REFACTOR/data/rotor_wake/rotor_case_stats.csv with one
# row per (case, n): sigma stats (min/mean/max/p05/p95), bounding-box extents,
# aspect ratio (z extent / max transverse extent), total |Gamma|, and uniform
# cubic-grid occupancy at levels ell=4..7 over the bounding CUBE (side = max
# extent, anchored at the bbox min corner): occupied cells, mean/max/p99 bodies
# per occupied cell, and the fraction of bodies in the top-1% densest cells.
#
# Usage:
#   julia --project=<env with FLOWVPM> MATRIX_OPERATOR_REFACTOR/scripts/rotor_case_stats.jl
# Env: FM033_STATS_NS (comma list, default "100000,1000000").

include(joinpath(@__DIR__, "benchmark_033_common.jl"))
using Printf

const STATS_NS = parse.(Int, split(get(ENV, "FM033_STATS_NS", "100000,1000000"), ','))
const STATS_LEVELS = 4:7
const OUTPATH = joinpath(@__DIR__, "..", "data", "rotor_wake", "rotor_case_stats.csv")

function quantile_sorted(sorted, p)
    n = length(sorted)
    n == 1 && return float(sorted[1])
    h = (n - 1) * p + 1
    lo = clamp(floor(Int, h), 1, n)
    hi = min(lo + 1, n)
    return sorted[lo] + (h - lo) * (sorted[hi] - sorted[lo])
end

# Occupancy of a 2^ell uniform cubic grid over the bounding cube.
function occupancy(X, lo, side, ell)
    ncell = 1 << ell
    counts = Dict{Int,Int}()
    for x in X
        i = clamp(floor(Int, (x[1] - lo[1]) / side * ncell), 0, ncell - 1)
        j = clamp(floor(Int, (x[2] - lo[2]) / side * ncell), 0, ncell - 1)
        k = clamp(floor(Int, (x[3] - lo[3]) / side * ncell), 0, ncell - 1)
        key = (i * ncell + j) * ncell + k
        counts[key] = get(counts, key, 0) + 1
    end
    occ = sort!(collect(values(counts)); rev=true)
    ntop = max(1, ceil(Int, 0.01 * length(occ)))
    return (occupied=length(occ),
        mean=sum(occ) / length(occ),
        max=occ[1],
        p99=quantile_sorted(reverse(occ), 0.99),
        topfrac=sum(occ[1:ntop]) / sum(occ))
end

function case_stats(case, n)
    pfield = fm033_build(case, n)
    np = vpm.get_np(pfield)
    np == n || error("$case built $np != $n particles")
    X = [Tuple(vpm.get_X(pfield, i)) for i in 1:np]
    sig = sort!([vpm.get_sigma(pfield, i)[] for i in 1:np])
    gtot = sum(sqrt(sum(abs2, vpm.get_Gamma(pfield, i))) for i in 1:np)
    lo = ntuple(d -> minimum(x -> x[d], X), 3)
    hi = ntuple(d -> maximum(x -> x[d], X), 3)
    ext = hi .- lo
    side = maximum(ext)
    ar = ext[3] / max(ext[1], ext[2])
    occ = [occupancy(X, lo, side, ell) for ell in STATS_LEVELS]
    return (; case, n,
        sigma_min=sig[1], sigma_mean=sum(sig) / np, sigma_max=sig[end],
        sigma_p05=quantile_sorted(sig, 0.05), sigma_p95=quantile_sorted(sig, 0.95),
        ext, ar, gtot, occ)
end

rows = [case_stats(case, n) for n in STATS_NS, case in ("rotor", "wake")]

mkpath(dirname(OUTPATH))
open(OUTPATH, "w") do io
    print(io, "case,n,sigma_min,sigma_mean,sigma_max,sigma_p05,sigma_p95,")
    print(io, "extent_x,extent_y,extent_z,aspect_ratio,total_abs_gamma")
    for ell in STATS_LEVELS
        print(io, ",ell$(ell)_occupied,ell$(ell)_mean,ell$(ell)_max,ell$(ell)_p99,ell$(ell)_top1pct_frac")
    end
    println(io)
    for r in rows
        print(io, r.case, ',', r.n, ',', r.sigma_min, ',', r.sigma_mean, ',',
            r.sigma_max, ',', r.sigma_p05, ',', r.sigma_p95, ',',
            r.ext[1], ',', r.ext[2], ',', r.ext[3], ',', r.ar, ',', r.gtot)
        for o in r.occ
            print(io, ',', o.occupied, ',', o.mean, ',', o.max, ',', o.p99, ',', o.topfrac)
        end
        println(io)
    end
end
println("wrote $OUTPATH")

for r in rows
    @printf("%-6s n=%-8d sigma[min/p05/mean/p95/max] = %.3e/%.3e/%.3e/%.3e/%.3e\n",
        r.case, r.n, r.sigma_min, r.sigma_p05, r.sigma_mean, r.sigma_p95, r.sigma_max)
    @printf("       extents = (%.4f, %.4f, %.4f)  AR = %.2f  total|Gamma| = %.4e\n",
        r.ext..., r.ar, r.gtot)
    for (ell, o) in zip(STATS_LEVELS, r.occ)
        frac_cells = o.occupied / (1 << (3ell))
        @printf("       ell=%d occ=%7d (%.3f%% of cells) bodies/cell mean=%8.1f max=%7d p99=%8.1f top1%%frac=%.3f\n",
            ell, o.occupied, 100frac_cells, o.mean, o.max, o.p99, o.topfrac)
    end
end
