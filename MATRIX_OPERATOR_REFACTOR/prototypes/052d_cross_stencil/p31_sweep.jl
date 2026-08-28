# P3.1 — two-occupancy rigid-stencil list-statistics sweep at production
# step-472 shape (36,752 tri-panel centroids -> 241,986 particles), 052d.
#
# Sweeps ell_x in ELLXS x q in QS on the SHARED grid (particle/self-pass root
# box, panel containment asserted per ruling R1). Uniform-q schedule, first
# M2L level 2. Per config: occupied cell counts, M2L route counts per level,
# near-field pair counts, per-offset-class census, device cost model, and
# exact-once certification:
#   (1) count identity over all 8.89e9 pairs,
#   (2) brute-force per-pair coverage for 400 sampled particles x ALL panels,
#   (3) the prototype's full coverage_counts matrix on a 2000x3000 subset
#       driven by the materialized lists.
#
# Run: JULIA_NUM_THREADS=4 julia --project=<FastMultipole> p31_sweep.jl

include(joinpath(@__DIR__, "CrossStencil.jl"))
using .CrossStencil
using StaticArrays, Printf, Random

const SNAPDIR = "/private/tmp/claude-502/-Users-ryan-Dropbox-research-projects-FastMultipole/1a9c539a-6d87-44ee-87f7-d4e1c17d2793/scratchpad/snapshot472"
const ELLXS = 4:9
const QS = (3, 5, 12)
const DENSE_TIME = 3.3          # s, measured dense A100 panels->particles
const DEV_THROUGHPUT = 1.0e12   # flop/s effective (same model as production_run.jl)
const GATE = 0.6                # s/step
m2l_flops(P) = Float64((2P + 1)^3)

read_i64(f) = reinterpret(Int64, read(joinpath(SNAPDIR, f)))
read_3xn(f) = reshape(reinterpret(Float64, read(joinpath(SNAPDIR, f))), 3, :)
pos = read_3xn("particle_positions_3xN_f64.bin")
verts = read_3xn("panel_vertices_3xM_f64.bin")
conn = read_i64("panel_connectivity_i64.bin"); offs = read_i64("panel_offsets_i64.bin")
panels = let lo = 1
    [begin
         hi = offs[k]
         c = sum(SVector(verts[1, conn[j]], verts[2, conn[j]], verts[3, conn[j]])
                 for j in lo:hi) / (hi - lo + 1)
         lo = hi + 1
         c
     end for k in 1:length(offs)]
end
particles = [SVector(pos[1, i], pos[2, i], pos[3, i]) for i in 1:size(pos, 2)]
ns, nt = length(panels), length(particles)
dense_total = Int128(ns) * Int128(nt)

g = CrossGrid(particles)   # SELF-PASS (particle) root box — ruling R1 default
println("Geometry: REAL step-472 snapshot | ns=$ns panels, nt=$nt particles, dense=$dense_total")
@printf("Shared grid (particle box): x_min=(%.5f, %.5f, %.5f) h0=%.6f\n",
        g.x_min[1], g.x_min[2], g.x_min[3], g.h0)

# R1 containment assert for panels (no silent clamping)
let maxexc = 0.0
    for p in panels
        _, e = CrossStencil.level_coords(g, p, 1)
        maxexc = max(maxexc, e)
    end
    @printf("Panel containment in particle box: max excursion = %.3g cell widths (level 1)\n", maxexc)
    maxexc == 0.0 || error("R1 CONTAINMENT ASSERT FAILED: panels exceed the particle root box; union-box fallback required")
end
println()

rng = MersenneTwister(42)
sample_js = rand(rng, 1:nt, 400)
# fixed subset for the full coverage-matrix check
sub_pan = panels[randperm(rng, ns)[1:2000]]
sub_par = particles[randperm(rng, nt)[1:3000]]

@printf("%-4s %-5s | %8s %8s | %9s %14s | %8s %13s %8s | %7s %7s | %9s %9s %9s | %s\n",
        "q", "ellx", "panCell", "parCell", "n_M2L", "m2l_inter", "n_near",
        "near_inter", "frac%", "offCls", "lvlCls", "t_near", "tM2L_P4", "tM2L_P8", "checks")
results = []
for q in QS
    tq = CrossStencil.UniformQTables(q)
    for ell_x in ELLXS
        t0 = time()
        src_levels, _, _ = build_level_cells(g, panels, ell_x)
        tgt_levels, _, _ = build_level_cells(g, particles, ell_x)
        stats = sweep_config(tq, src_levels, tgt_levels, ell_x)
        t_build = time() - t0

        covered = stats.m2l_interactions + stats.near_interactions
        ident = covered == dense_total
        bad = brute_coverage(tq, g, panels, particles, ell_x, sample_js)

        # subset full coverage matrix via materialized lists
        sg = g   # same shared grid
        ssrc, _, _ = build_level_cells(sg, sub_pan, ell_x)
        stgt, _, _ = build_level_cells(sg, sub_par, ell_x)
        _, m2l_list, near_list = sweep_config(tq, ssrc, stgt, ell_x; materialize=true)
        miss, dup = coverage_counts_subset(ssrc, stgt, m2l_list, near_list,
                                           length(sub_pan), length(sub_par), ell_x)

        frac = Float64(stats.near_interactions) / Float64(dense_total)
        t_near = frac * DENSE_TIME
        t4 = stats.n_m2l * m2l_flops(4) / DEV_THROUGHPUT
        t8 = stats.n_m2l * m2l_flops(8) / DEV_THROUGHPUT
        ok = ident && bad == 0 && miss == 0 && dup == 0
        checks = ok ? "PASS" : "FAIL(id=$ident bad=$bad miss=$miss dup=$dup)"
        @printf("%-4d %-5d | %8d %8d | %9d %14d | %8d %13d %8.3f | %7d %7d | %9.4f %9.4f %9.4f | %s  [build %.2fs]\n",
                q, ell_x, stats.src_cells_per_level[end], stats.tgt_cells_per_level[end],
                stats.n_m2l, Int(stats.m2l_interactions), stats.near_pairs,
                Int(stats.near_interactions), 100 * frac, stats.nclasses_used,
                stats.lo_classes, t_near, t4, t8, checks, t_build)
        push!(results, (; q, ell_x, stats, frac, t_near, t4, t8, ok))
    end
    # per-level route/cell profile for this q at the deepest ell_x
    r = last(results)
    println("  per-level profile q=$q ell_x=$(r.ell_x):")
    for L in 0:r.ell_x
        @printf("    L=%d: panCells=%6d parCells=%7d routes=%9d\n", L,
                r.stats.src_cells_per_level[L + 1], r.stats.tgt_cells_per_level[L + 1],
                L >= 2 ? r.stats.routes_per_level[L + 1] : 0)
    end
end

println()
println("Cost-model note: t_near scales the measured 3.3 s dense A100 rate by the")
println("near-field fraction; tM2L_P assumes (2P+1)^3 flop/route at 1e12 flop/s")
println("effective. Gate = $(GATE) s/step. B2M/L2B/M2M/L2L unmodeled (small).")
for (label, sel) in (("q=3", r -> r.q == 3), ("q=5", r -> r.q == 5), ("q=12", r -> r.q == 12))
    rs = filter(sel, results)
    best = argmin(r -> r.t_near + r.t4, rs)
    @printf("Best %-4s (P=4 model): ell_x=%d  total=%.4f s  (near %.4f + M2L %.4f)  margin %.0fx\n",
            label, best.ell_x, best.t_near + best.t4, best.t_near, best.t4,
            GATE / (best.t_near + best.t4))
end
allok = all(r -> r.ok, results)
println(allok ? "\nALL EXACT-ONCE CERTIFICATIONS PASSED" : "\nCERTIFICATION FAILURES PRESENT")
exit(allok ? 0 : 1)
