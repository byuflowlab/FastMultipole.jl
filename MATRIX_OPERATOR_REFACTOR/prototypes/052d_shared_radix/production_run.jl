# Production-shape list build + device cost model for the shared-radix
# dual-tree design (052d Phase 2b-revised).
#
# Sources = panel centroids from the step-472 snapshot; targets = particles.
# Sweeps leaf_size x theta, reporting list build wall time, list sizes,
# near-field pair-interaction fraction of the dense total, and a modeled
# device cost against the 0.6 s/step gate.
#
# Run: JULIA_NUM_THREADS=4 julia --project=<FastMultipole> production_run.jl
#
# If the snapshot directory is missing, generates a synthetic stand-in
# (annular-wake-like 242k particles + 9k source points on a small disc)
# and clearly labels the output as SYNTHETIC.

include(joinpath(@__DIR__, "SharedRadix.jl"))
using .SharedRadix
using StaticArrays, Printf, Random

const SNAPDIR = "/private/tmp/claude-502/-Users-ryan-Dropbox-research-projects-FastMultipole/1a9c539a-6d87-44ee-87f7-d4e1c17d2793/scratchpad/snapshot472"

const DENSE_TIME = 3.3          # s, measured dense A100 all-pairs panels->particles
const P = 4                     # expansion order for the M2L flops estimate
const M2L_FLOPS = Float64((2P + 1)^3)   # flops-order per M2L pair (rotation-trick O(p^3))
const DEV_THROUGHPUT = 1.0e12   # flop/s assumed EFFECTIVE device throughput
                                # (~5% of A100 FP32 peak 19.5 TF -- conservative for
                                #  small-batch, memory-bound translation kernels)
const GATE = 0.6                # s/step

function load_snapshot()
    read_i64(f) = reinterpret(Int64, read(joinpath(SNAPDIR, f)))
    read_3xn(f) = reshape(reinterpret(Float64, read(joinpath(SNAPDIR, f))), 3, :)
    pos = read_3xn("particle_positions_3xN_f64.bin")
    verts = read_3xn("panel_vertices_3xM_f64.bin")
    conn = read_i64("panel_connectivity_i64.bin")   # 1-based, verified
    offs = read_i64("panel_offsets_i64.bin")        # cumulative ends (VTK style)
    npan = length(offs)
    centroids = Vector{SVector{3,Float64}}(undef, npan)
    lo = 1
    for k in 1:npan
        hi = offs[k]
        c = SVector(0.0, 0.0, 0.0)
        for j in lo:hi
            v = conn[j]
            c += SVector(verts[1, v], verts[2, v], verts[3, v])
        end
        centroids[k] = c / (hi - lo + 1)
        lo = hi + 1
    end
    targets = [SVector(pos[1, i], pos[2, i], pos[3, i]) for i in 1:size(pos, 2)]
    return centroids, targets, "REAL step-472 snapshot"
end

function synthetic_standin()
    rng = MersenneTwister(2026)
    # annular-wake-like particle cloud: helical annulus stretched along +x
    nt = 242_000
    targets = Vector{SVector{3,Float64}}(undef, nt)
    for i in 1:nt
        x = 0.42 * rand(rng)^0.8
        r = 0.09 + 0.06 * randn(rng) * 0.3 + 0.5 * x * 0.15
        phi = 2pi * rand(rng)
        targets[i] = SVector(x, r * cos(phi), r * sin(phi))
    end
    # 9k source points on a disc ~1/5 the extent, near x=0
    ns = 9_000
    sources = Vector{SVector{3,Float64}}(undef, ns)
    for i in 1:ns
        r = 0.09 * sqrt(rand(rng))
        phi = 2pi * rand(rng)
        sources[i] = SVector(0.002 * randn(rng), r * cos(phi), r * sin(phi))
    end
    return sources, targets, "SYNTHETIC STAND-IN (snapshot unavailable)"
end

sources, targets, label = isdir(SNAPDIR) && isfile(joinpath(SNAPDIR, "particle_positions_3xN_f64.bin")) ?
    load_snapshot() : synthetic_standin()

ns, nt = length(sources), length(targets)
dense_total = Float64(ns) * Float64(nt)
println("Geometry: $label")
println("  n_sources (panel centroids) = $ns, n_targets (particles) = $nt")
println("  dense pair total = ", @sprintf("%.4g", dense_total))
println("  threads = ", Threads.nthreads())
println()

g = shared_grid(sources, targets)
println("Shared grid: center=", g.center, " halfwidth=", @sprintf("%.5f", g.halfwidth))
println()

# warm-up (JIT) on a small subset
let s = sources[1:min(500, ns)], t = targets[1:min(500, nt)]
    gs = shared_grid(s, t)
    a = build_tree(gs, s; leaf_size=32); b = build_tree(gs, t; leaf_size=32)
    dual_traversal(a, b; theta=0.5)
end

@printf("%-9s %-6s | %9s %9s | %8s %8s %10s | %12s %9s | %9s %9s %9s | %s\n",
        "leaf", "theta", "t_tree(s)", "t_trav(s)", "n_M2L", "n_near", "nearpairs",
        "frac_dense", "t_near(s)", "t_M2L(s)", "total(s)", "gate", "verdict")
results = []
for leaf_size in (32, 64, 128, 256), theta in (0.4, 0.5, 0.6)
    t0 = time()
    stree = build_tree(g, sources; leaf_size)
    ttree = build_tree(g, targets; leaf_size)
    t_tree = time() - t0
    t0 = time()
    m2l, near = dual_traversal(stree, ttree; theta)
    t_trav = time() - t0
    nearpairs = sum(length(stree.cells[si].range) * length(ttree.cells[ti].range)
                    for (si, ti) in near; init=0)
    frac = nearpairs / dense_total
    t_near = frac * DENSE_TIME
    t_m2l = length(m2l) * M2L_FLOPS / DEV_THROUGHPUT
    total = t_near + t_m2l
    # distinct (dlevel, offset) translation classes (device operator-cache size)
    classes = Set{NTuple{4,Int}}()
    for (si, ti) in m2l
        S = stree.cells[si]; T = ttree.cells[ti]
        cs = cell_center(g, S.level, S.code); ct = cell_center(g, T.level, T.code)
        h = cell_halfwidth(g, max(S.level, T.level))
        d = ct - cs
        push!(classes, (Int(S.level) - Int(T.level),
                        round(Int, d[1] / h), round(Int, d[2] / h), round(Int, d[3] / h)))
    end
    verdict = total < GATE ? "PASS" : "FAIL"
    @printf("%-9d %-6.2f | %9.3f %9.3f | %8d %8d %10d | %12.5f %9.4f | %9.4f %9.4f %9.2f | %s (classes=%d)\n",
            leaf_size, theta, t_tree, t_trav, length(m2l), length(near), nearpairs,
            frac, t_near, t_m2l, t_near + t_m2l, GATE, verdict, length(classes))
    push!(results, (; leaf_size, theta, t_tree, t_trav, n_m2l=length(m2l),
                    n_near=length(near), nearpairs, frac, t_near, t_m2l, total,
                    nclasses=length(classes)))
end

best = argmin(r -> r.total, results)
println()
@printf("Best modeled total: %.4f s (leaf=%d theta=%.2f) vs gate %.1f s -> margin %.1fx\n",
        best.total, best.leaf_size, best.theta, GATE, GATE / best.total)
