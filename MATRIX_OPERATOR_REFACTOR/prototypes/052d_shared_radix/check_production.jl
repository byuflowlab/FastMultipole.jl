# Production-scale partition checks (the full 8.9e9 coverage matrix is
# infeasible, so two complementary checks):
#   1. COUNT IDENTITY: sum over all list pairs of |S| x |T| must equal
#      exactly n_src * n_tgt (necessary for a partition; combined with the
#      no-overlap sampled check below it is also sufficient in practice).
#   2. SAMPLED EXACT COVERAGE: for 400 random target points, accumulate the
#      per-source coverage vector over every list pair containing that target;
#      all n_src entries must equal exactly 1.
# Also re-verifies MAC validity for every accepted M2L pair at this scale.
# Run: JULIA_NUM_THREADS=4 julia --project=<FastMultipole> check_production.jl

include(joinpath(@__DIR__, "SharedRadix.jl"))
using .SharedRadix
using StaticArrays, Random, Printf
using LinearAlgebra: norm

const SNAPDIR = "/private/tmp/claude-502/-Users-ryan-Dropbox-research-projects-FastMultipole/1a9c539a-6d87-44ee-87f7-d4e1c17d2793/scratchpad/snapshot472"

read_i64(f) = reinterpret(Int64, read(joinpath(SNAPDIR, f)))
read_3xn(f) = reshape(reinterpret(Float64, read(joinpath(SNAPDIR, f))), 3, :)
pos = read_3xn("particle_positions_3xN_f64.bin")
verts = read_3xn("panel_vertices_3xM_f64.bin")
conn = read_i64("panel_connectivity_i64.bin"); offs = read_i64("panel_offsets_i64.bin")
sources = let lo = 1
    [begin
         hi = offs[k]
         c = sum(SVector(verts[1, conn[j]], verts[2, conn[j]], verts[3, conn[j]]) for j in lo:hi) / (hi - lo + 1)
         lo = hi + 1
         c
     end for k in 1:length(offs)]
end
targets = [SVector(pos[1, i], pos[2, i], pos[3, i]) for i in 1:size(pos, 2)]
ns, nt = length(sources), length(targets)

allok = true
for (leaf_size, theta) in ((32, 0.5), (128, 0.4), (256, 0.6))
    g = shared_grid(sources, targets)
    st = build_tree(g, sources; leaf_size)
    tt = build_tree(g, targets; leaf_size)
    m2l, near = dual_traversal(st, tt; theta)

    covered = sum(Int64(length(st.cells[si].range)) * length(tt.cells[ti].range) for (si, ti) in m2l; init=Int64(0)) +
              sum(Int64(length(st.cells[si].range)) * length(tt.cells[ti].range) for (si, ti) in near; init=Int64(0))
    ident = covered == Int64(ns) * Int64(nt)

    # exact integer-arithmetic MAC (non-strict; see exact_mac_leq docstring —
    # shared-grid quantization makes exact boundary ties possible)
    q = rationalize(theta)
    macbad = 0
    for (si, ti) in m2l
        exact_mac_leq(st.cells[si], tt.cells[ti], numerator(q), denominator(q)) || (macbad += 1)
    end

    rng = MersenneTwister(42)
    sample = rand(rng, 1:nt, 400)          # sorted-order target indices
    bad_samples = 0
    cover = zeros(Int32, ns)
    for j in sample
        fill!(cover, Int32(0))
        for lists in (m2l, near), (si, ti) in lists
            if j in tt.cells[ti].range
                cover[st.cells[si].range] .+= Int32(1)
            end
        end
        all(==(Int32(1)), cover) || (bad_samples += 1)
    end

    ok = ident && macbad == 0 && bad_samples == 0
    global allok &= ok
    @printf("leaf=%3d theta=%.1f : count_identity=%s (covered=%d, dense=%d) macbad=%d bad_samples=%d/400 -> %s\n",
            leaf_size, theta, ident, covered, Int64(ns) * Int64(nt), macbad, bad_samples, ok ? "PASS" : "FAIL")
end
println(allok ? "PRODUCTION-SCALE CHECKS PASSED" : "PRODUCTION-SCALE CHECKS FAILED")
exit(allok ? 0 : 1)
