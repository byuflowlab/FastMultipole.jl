# P3.2d — measure the regularized-vs-singular kernel tail and the
# near-surface particle population, to set the guard radius R_guard for the
# hybrid (physical-radius) cross-stencil schedule.
#
# Part A: per-pair relative velocity difference |U_reg - U_sing| / |U_sing|
#   for single panels probed at controlled distances r (log-spaced 2mm..12cm),
#   200 random panels x 24 distances x 4 directions. Reports per-distance
#   median/p90/max and a local log-log slope. R* = distance where p90 tail
#   first drops below 1e-4 (and 3e-5 for margin).
# Part B: fraction of the 241,986 particles within R of ANY panel centroid,
#   for R in {1,2,3,4,5,6,8} cm (grid-bucketed), with the implied dense
#   near-field cost at the measured 2.695e9 pairs/s A100 rate.
#
# Run: JULIA_NUM_THREADS=4 julia --project=../FLOWPanel.jl p32d_tail.jl

import FLOWPanel as pnl
import FastMultipole
const FM = FastMultipole
using StaticArrays, Random, Statistics, Printf, LinearAlgebra

const SNAPDIR = "/private/tmp/claude-502/-Users-ryan-Dropbox-research-projects-FastMultipole/1a9c539a-6d87-44ee-87f7-d4e1c17d2793/scratchpad/snapshot472"

read3(f) = collect(reshape(reinterpret(Float64, read(joinpath(SNAPDIR, f))), 3, :))
positions = read3("particle_positions_3xN_f64.bin")
vpts = read3("panel_vertices_3xM_f64.bin")
conn = collect(reinterpret(Int64, read(joinpath(SNAPDIR, "panel_connectivity_i64.bin"))))
cells = collect(reshape(conn, 3, :))
nt = size(positions, 2)
ns = size(cells, 2)
centroids = [SVector{3,Float64}((vpts[:, cells[1, k]] .+ vpts[:, cells[2, k]] .+
              vpts[:, cells[3, k]]) ./ 3) for k in 1:ns]

function make_body(cst)
    kernel = Union{pnl.ConstantSource, pnl.VortexRing}
    b = pnl.RigidWakeBody{kernel}(vpts, cells, pnl.noshedding; watertight=true,
        DBC=true, core_size_panel=0.12e-10, core_size_targets=cst)
    pnl.calc_normals!(b)
    pnl.calc_controlpoints!(b)
    pnl._set_core_sizes!((b,), :core_size_targets)
    Random.seed!(472)
    b.strength .= 0.1 .* randn(b.ncells, size(b.strength, 2))
    return b
end
body = make_body(1e-3)
body_s = make_body(1e-12)
sbuf = FM.system_to_buffer(body)
sbuf_s = FM.system_to_buffer(body_s)
ds = FM.DerivativesSwitch(false, true, false)
try
    println("filament regularization family: ", pnl.FILAMENT_REGULARIZATION[])
catch e
    println("filament regularization family lookup failed: ", typeof(e))
end
println("body core_size = ", body.core_size, " | singular control core = ", body_s.core_size)

println("\n=== A. per-pair kernel tail (200 panels x 24 distances x 4 dirs) ===")
rng = MersenneTwister(7)
panel_ids = rand(rng, 1:ns, 200)
dists = exp10.(range(log10(2e-3), log10(0.12); length=24))
dirs = [normalize(randn(rng, SVector{3,Float64})) for _ in 1:4]
@printf("%10s | %10s %10s %10s | %s\n", "r (m)", "median", "p90", "max", "r/sigma")
p90s = Float64[]
for r in dists
    rels = Float64[]
    for k in panel_ids, d in dirs
        x = centroids[k] + r * d
        tb = zeros(7, 1); tb[1:3, 1] .= x
        FM.direct!(tb, 1:1, ds, body, sbuf, k:k)
        Ur = SVector{3,Float64}(tb[4, 1], tb[5, 1], tb[6, 1])
        tb2 = zeros(7, 1); tb2[1:3, 1] .= x
        FM.direct!(tb2, 1:1, ds, body_s, sbuf_s, k:k)
        Us = SVector{3,Float64}(tb2[4, 1], tb2[5, 1], tb2[6, 1])
        nUs = norm(Us)
        nUs > 0 && push!(rels, norm(Ur - Us) / nUs)
    end
    push!(p90s, quantile(rels, 0.9))
    @printf("%10.4e | %10.3e %10.3e %10.3e | %6.1f\n", r, quantile(rels, 0.5),
            quantile(rels, 0.9), maximum(rels), r / 1e-3)
end
for i in 2:length(dists)
    if p90s[i - 1] > 0 && p90s[i] > 0
        s = log(p90s[i] / p90s[i - 1]) / log(dists[i] / dists[i - 1])
        i % 4 == 0 && @printf("local p90 slope near r=%.3e : %.2f\n", dists[i], s)
    end
end
for tol in (1e-4, 3e-5)
    i = findfirst(<=(tol), p90s)
    println("R*(p90 <= $tol) = ", i === nothing ? "NOT REACHED by 0.12 m" : @sprintf("%.4e m", dists[i]))
end

println("\n=== B. near-surface particle population + dense-cost model ===")
# bucket panel centroids on a 1 cm grid; particle within R of a centroid tested
# against buckets in ceil(R/w)-ring
const WB = 0.01
lo = SVector{3,Float64}(minimum(positions[1, :]), minimum(positions[2, :]), minimum(positions[3, :]))
key(p) = (floor(Int, (p[1] - lo[1]) / WB), floor(Int, (p[2] - lo[2]) / WB), floor(Int, (p[3] - lo[3]) / WB))
buckets = Dict{NTuple{3,Int},Vector{Int}}()
for k in 1:ns
    push!(get!(Vector{Int}, buckets, key(centroids[k])), k)
end
mind2 = fill(Inf, nt)
Threads.@threads for i in 1:nt
    p = SVector{3,Float64}(positions[:, i])
    kx, ky, kz = key(p)
    best = Inf
    ring = 0
    # expand rings until the found minimum is certain (ring covers best)
    while true
        found_any = false
        for dz in -ring:ring, dy in -ring:ring, dx in -ring:ring
            max(abs(dx), abs(dy), abs(dz)) == ring || continue
            b = get(buckets, (kx + dx, ky + dy, kz + dz), nothing)
            b === nothing && continue
            found_any = true
            for k in b
                d2 = sum(abs2, p - centroids[k])
                d2 < best && (best = d2)
            end
        end
        # stop when the next ring cannot contain anything closer, or we're far out
        if best < ((ring) * WB)^2 || ring > 12
            break
        end
        ring += 1
    end
    mind2[i] = best
end
mind = sqrt.(mind2)
DENSE_RATE = 2.695e9
for R in (0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08)
    n_in = count(<=(R), mind)
    frac = n_in / nt
    t = frac * ns * nt / DENSE_RATE
    @printf("R=%4.2f m : particles within R of surface = %7d (%5.2f%%) -> dense guard cost ~ %.3f s\n",
            R, n_in, 100 * frac, t)
end
println("DONE")
