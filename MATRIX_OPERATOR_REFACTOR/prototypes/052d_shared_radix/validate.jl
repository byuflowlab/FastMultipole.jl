# Correctness validation for the shared-radix dual-tree list machinery.
# Checks, per configuration:
#   (a) PAIR-PARTITION: every (source point, target point) pair is covered
#       EXACTLY once by the union of M2L cell pairs (via subtree ranges) and
#       near-field leaf pairs. Zero misses, zero double counts.
#   (b) MAC VALIDITY: every accepted M2L pair satisfies the MAC recomputed
#       from on-the-fly (level, code) geometry.
#   (c) DETERMINISM: the lists (as sets of grid cell identities) are invariant
#       under permutation of the input points.
# Run: JULIA_NUM_THREADS=4 julia --project=<FastMultipole> validate.jl

include(joinpath(@__DIR__, "SharedRadix.jl"))
using .SharedRadix
using StaticArrays, Random, Printf
using LinearAlgebra: norm

const LEAF_SIZE = 32
const THETA = 0.5

rand3(rng, n, lo::SVector{3,Float64}, hi::SVector{3,Float64}) =
    [lo .+ rand(rng, SVector{3,Float64}) .* (hi .- lo) for _ in 1:n]

"Exhaustive pair-coverage matrix in sorted-index space (bijective relabeling)."
function coverage_counts(src, tgt, m2l, near)
    ns, nt = length(src.order), length(tgt.order)
    cover = zeros(Int32, ns, nt)
    for (si, ti) in m2l
        S = src.cells[si]; T = tgt.cells[ti]
        for j in T.range, i in S.range
            cover[i, j] += Int32(1)
        end
    end
    for (si, ti) in near
        S = src.cells[si]; T = tgt.cells[ti]
        for j in T.range, i in S.range
            cover[i, j] += Int32(1)
        end
    end
    return cover
end

"""
Independent MAC recheck in EXACT integer arithmetic (non-strict: accepted
pairs must satisfy r_S + r_T <= theta * dist; shared-grid quantization makes
exact boundary ties genuinely possible, and a tie still delivers the theta
error bound).
"""
function mac_violations(src, tgt, m2l; theta=THETA)
    q = rationalize(theta)
    bad = 0
    for (si, ti) in m2l
        S = src.cells[si]; T = tgt.cells[ti]
        exact_mac_leq(S, T, numerator(q), denominator(q)) || (bad += 1)
    end
    return bad
end

"Lists as an order-independent set of grid-cell-identity pairs."
list_ids(src, tgt, pairs) =
    sort!([(cell_id(src.cells[si]), cell_id(tgt.cells[ti])) for (si, ti) in pairs])

function run_case(name, src_pts, tgt_pts; leaf_size=LEAF_SIZE, theta=THETA, rng=MersenneTwister(0))
    g = shared_grid(src_pts, tgt_pts)
    src = build_tree(g, src_pts; leaf_size)
    tgt = build_tree(g, tgt_pts; leaf_size)
    m2l, near = dual_traversal(src, tgt; theta)

    cover = coverage_counts(src, tgt, m2l, near)
    miss = count(==(Int32(0)), cover)
    dup = count(>(Int32(1)), cover)
    macbad = mac_violations(src, tgt, m2l; theta)

    # determinism under point permutation
    ps = randperm(rng, length(src_pts)); pt = randperm(rng, length(tgt_pts))
    src2 = build_tree(g, src_pts[ps]; leaf_size)
    tgt2 = build_tree(g, tgt_pts[pt]; leaf_size)
    m2l2, near2 = dual_traversal(src2, tgt2; theta)
    det_ok = list_ids(src, tgt, m2l) == list_ids(src2, tgt2, m2l2) &&
             list_ids(src, tgt, near) == list_ids(src2, tgt2, near2)

    npair_near = sum(length(src.cells[si].range) * length(tgt.cells[ti].range)
                     for (si, ti) in near; init=0)
    pass = miss == 0 && dup == 0 && macbad == 0 && det_ok
    @printf("%-34s %4s | nsrc=%5d ntgt=%5d | M2L=%6d near=%6d nearpairs=%9d | miss=%d dup=%d macbad=%d det=%s\n",
            name, pass ? "PASS" : "FAIL", length(src_pts), length(tgt_pts),
            length(m2l), length(near), npair_near, miss, dup, macbad, det_ok)
    return pass
end

allpass = true

# --- random clustered-source / wide-target configurations, several seeds ---
for seed in 1:4
    rng = MersenneTwister(seed)
    # small off-center source cluster (mimics the rotor inside the wake cloud)
    src = rand3(rng, 2000, SVector(0.55, 0.55, 0.55), SVector(0.75, 0.75, 0.75))
    # targets spread wide
    tgt = rand3(rng, 3000, SVector(-1.0, -1.0, -1.0), SVector(1.0, 1.0, 1.0))
    global allpass &= run_case("random seed $seed", src, tgt; rng)
end

# --- pathological cases ---
rng = MersenneTwister(99)
# all points coincident (single finest cell; forced oversized leaves at MAX_LEVEL)
p0 = SVector(0.3, 0.3, 0.3)
global allpass &= run_case("all-in-one-cell (coincident)", fill(p0, 200), fill(p0, 300); rng)
# all points in one deep cell but distinct
eps_pts(rng, n) = [p0 .+ 1e-9 .* rand(rng, SVector{3,Float64}) for _ in 1:n]
global allpass &= run_case("all-in-one-cell (tiny ball)", eps_pts(rng, 200), eps_pts(rng, 300); rng)
# collinear points
line(rng, n, a, b) = [SVector(a + (b - a) * rand(rng), 0.0, 0.0) for _ in 1:n]
global allpass &= run_case("collinear same line", line(rng, 500, -1.0, 1.0), line(rng, 700, -1.0, 1.0); rng)
global allpass &= run_case("collinear disjoint segments", line(rng, 500, -1.0, -0.5), line(rng, 700, 0.5, 1.0); rng)
# coincident boxes: source and target sets drawn from identical distribution
src = rand3(rng, 1500, SVector(-1.0, -1.0, -1.0), SVector(1.0, 1.0, 1.0))
tgt = rand3(rng, 1500, SVector(-1.0, -1.0, -1.0), SVector(1.0, 1.0, 1.0))
global allpass &= run_case("coincident boxes", src, tgt; rng)
# identical point sets (self-like cross pass)
global allpass &= run_case("identical point sets", src, copy(src); rng)
# strong scale separation: source cluster 1e-3 of the target extent
src = rand3(rng, 2000, SVector(0.7, 0.7, 0.7), SVector(0.701, 0.701, 0.701))
tgt = rand3(rng, 3000, SVector(-1.0, -1.0, -1.0), SVector(1.0, 1.0, 1.0))
global allpass &= run_case("deep-level source cluster", src, tgt; rng)
# leaf_size / theta variations on one config
for ls in (8, 64), th in (0.4, 0.6)
    global allpass &= run_case("random ls=$ls theta=$th", rand3(rng, 2000, SVector(0.55,0.55,0.55), SVector(0.75,0.75,0.75)),
                        rand3(rng, 3000, SVector(-1.0,-1.0,-1.0), SVector(1.0,1.0,1.0));
                        leaf_size=ls, theta=th, rng)
end

println(allpass ? "\nALL VALIDATION CHECKS PASSED" : "\nVALIDATION FAILURES PRESENT")
exit(allpass ? 0 : 1)
