# Independent reviewer adversarial tests for SharedRadix (052d review, 2026-08-28).
# NOT part of the original prototype. Targets suspected weaknesses:
#   T1  points exactly ON cell boundaries (lattice coordinates), several theta
#   T2  engineered exact MAC boundary ties (unit-test exact_mac_leq at the tie,
#       and count exact ties among FP-accepted production-style pairs)
#   T3  maximum level disparity: 1e-15-extent source cloud (forced MAX_LEVEL
#       oversized leaf) inside a wide target cloud; single-point source; empty src
#   T4  validator-of-the-validator: corrupt the lists (drop a pair / duplicate a
#       pair / swap an M2L pair's child for its parent) and confirm the coverage
#       checker actually flags it
#   T5  identical src/tgt sets WITH internal duplicates, far-off-origin coords
#   T6  FP-accept vs exact-reject: any accepted pair that STRICTLY violates the
#       exact MAC (not a tie) would be a real bug
# Run: JULIA_NUM_THREADS=4 julia --project=<FastMultipole> reviewer_tests.jl

include(joinpath(@__DIR__, "SharedRadix.jl"))
using .SharedRadix
using StaticArrays, Random, Printf

failures = String[]
check(name, ok) = (ok || push!(failures, name);
                   @printf("%-58s %s\n", name, ok ? "PASS" : "FAIL"); ok)

function coverage(src, tgt, m2l, near)
    ns, nt = length(src.order), length(tgt.order)
    cover = zeros(Int32, ns, nt)
    for lists in (m2l, near), (si, ti) in lists
        S = src.cells[si]; T = tgt.cells[ti]
        for j in T.range, i in S.range
            cover[i, j] += Int32(1)
        end
    end
    return count(==(Int32(0)), cover), count(>(Int32(1)), cover)
end

# exact-MAC audit of accepted list: (strict violations, exact ties)
function mac_audit(src, tgt, m2l, theta)
    q = rationalize(theta); p = numerator(q); qd = denominator(q)
    viol = 0; ties = 0
    for (si, ti) in m2l
        S = src.cells[si]; T = tgt.cells[ti]
        exact_mac_leq(S, T, p, qd) || (viol += 1)
        # tie test: <= holds AND >= holds  <=>  equality
        if exact_mac_leq(S, T, p, qd)
            # recompute equality directly
            L = max(S.level, T.level)
            ixs, iys, izs = morton_decode(S.code); ixt, iyt, izt = morton_decode(T.code)
            ss = Int128(1) << (L - S.level); st = Int128(1) << (L - T.level)
            dx = (2Int128(ixs)+1)*ss - (2Int128(ixt)+1)*st
            dy = (2Int128(iys)+1)*ss - (2Int128(iyt)+1)*st
            dz = (2Int128(izs)+1)*ss - (2Int128(izt)+1)*st
            s = ss + st
            3*Int128(qd)^2*s^2 == Int128(p)^2*(dx^2+dy^2+dz^2) && (ties += 1)
        end
    end
    return viol, ties
end

function full_case(name, spts, tpts; leaf_size=32, theta=0.5)
    g = shared_grid(spts, tpts)
    src = build_tree(g, spts; leaf_size)
    tgt = build_tree(g, tpts; leaf_size)
    m2l, near = dual_traversal(src, tgt; theta)
    miss, dup = coverage(src, tgt, m2l, near)
    viol, ties = mac_audit(src, tgt, m2l, theta)
    check("$name (miss=$miss dup=$dup viol=$viol ties=$ties)", miss == 0 && dup == 0 && viol == 0)
    return src, tgt, m2l, near, ties
end

rng = MersenneTwister(2028)

# ---------- T1: points exactly on cell boundaries ----------
# Lattice at multiples of 1/8 in [0,1]^3 -> after the 1e-6 grid inflation these
# are near (not exactly on) internal boundaries; ALSO build a case where points
# land exactly on floor()-boundaries of the fine lattice by using the grid's own
# lo + k*w coordinates.
lat = [SVector(i/8, j/8, k/8) for i in 0:8 for j in 0:8 for k in 0:8]
for th in (0.4, 0.5, 0.6, 1.0)
    full_case("T1 lattice 9^3 src=tgt theta=$th", lat, copy(lat); leaf_size=16, theta=th)
end
# points exactly at grid cell lower corners of the implied fine lattice:
let g = shared_grid(lat, lat)
    n = UInt64(1) << MAX_LEVEL
    w = 2 * g.halfwidth / n
    lo = g.center .- g.halfwidth
    exact = [SVector(lo[1] + (k * 2^15) * w, lo[2] + (k * 2^15) * w, lo[3]) for k in 0:60]
    full_case("T1b points on exact fine-cell corners", exact, reverse(exact); leaf_size=4, theta=0.5)
end

# ---------- T2: engineered exact MAC boundary tie ----------
# theta = 3/5, Delta-level 3: s = 2^3 + 2^0 = 9, need 3*25*s^2 == 9*|D|^2
# -> |D|^2 = 675 = 15^2+15^2+15^2. Construct the two cells directly.
let
    # S at level L-3 (coarse), T at level L (fine). Choose L=10, S code (0,0,0),
    # T integer coords so center offset in fine half-widths is (15,15,15):
    # center_S = (2*0+1)*8 = 8 (each axis, fine halfwidth units), so
    # center_T = 8+15 = 23 = 2*i+1 -> i = 11.
    S = SharedRadix.Cell(Int32(7), SharedRadix.morton_encode(UInt64(0), UInt64(0), UInt64(0)), 1:1, 1:0)
    T = SharedRadix.Cell(Int32(10), SharedRadix.morton_encode(UInt64(11), UInt64(11), UInt64(11)), 1:1, 1:0)
    tie   = exact_mac_leq(S, T, 3, 5)                      # tie must satisfy non-strict MAC
    # nudge one axis by one fine half-width-equivalent (i=12 -> offset 17? no:
    # 2*12+1=25 -> offset 17) -> farther, must pass; i=10 -> offset 5,13? use
    # (2*10+1)=21 -> offset 13 -> closer, must fail.
    Tfar  = SharedRadix.Cell(Int32(10), SharedRadix.morton_encode(UInt64(12), UInt64(11), UInt64(11)), 1:1, 1:0)
    Tnear = SharedRadix.Cell(Int32(10), SharedRadix.morton_encode(UInt64(10), UInt64(11), UInt64(11)), 1:1, 1:0)
    check("T2 exact tie satisfies non-strict MAC", tie)
    check("T2 one-cell-farther passes", exact_mac_leq(S, Tfar, 3, 5))
    check("T2 one-cell-nearer fails", !exact_mac_leq(S, Tnear, 3, 5))
end
# and confirm ties genuinely occur in a full run at theta=0.6 on a config with
# cross-level pairs (clustered source in wide target):
let
    spts = [SVector(0.5, 0.5, 0.5) .+ 0.02 .* rand(rng, SVector{3,Float64}) for _ in 1:3000]
    tpts = [SVector(-1.0, -1.0, -1.0) .+ 2.0 .* rand(rng, SVector{3,Float64}) for _ in 1:6000]
    _, _, _, _, ties = full_case("T2b cluster-in-wide theta=0.6 leaf=8", spts, tpts; leaf_size=8, theta=0.6)
    println("        (exact ties observed among accepted pairs: $ties)")
end

# ---------- T3: maximum level disparity ----------
spts = [SVector(0.25, 0.25, 0.25) .+ 1e-15 .* rand(rng, SVector{3,Float64}) for _ in 1:500]
tpts = [SVector(-1000.0, -1000.0, -1000.0) .+ 2000.0 .* rand(rng, SVector{3,Float64}) for _ in 1:3000]
full_case("T3 1e-15 src cloud in 2000-wide tgt", spts, tpts; leaf_size=32, theta=0.5)
full_case("T3b same, reversed roles", tpts, spts; leaf_size=32, theta=0.5)
full_case("T3c single source point", [SVector(0.1, 0.2, 0.3)], tpts; leaf_size=32, theta=0.5)
let g = shared_grid(tpts, tpts)  # empty source set: traversal must return empty lists
    es = build_tree(g, SVector{3,Float64}[]; leaf_size=32)
    tt = build_tree(g, tpts; leaf_size=32)
    m2l, near = dual_traversal(es, tt; theta=0.5)
    check("T3d empty source -> empty lists", isempty(m2l) && isempty(near))
end

# ---------- T4: does the coverage checker catch corrupted lists? ----------
let
    spts = [rand(rng, SVector{3,Float64}) for _ in 1:800]
    tpts = [rand(rng, SVector{3,Float64}) for _ in 1:900]
    g = shared_grid(spts, tpts)
    src = build_tree(g, spts; leaf_size=16); tgt = build_tree(g, tpts; leaf_size=16)
    m2l, near = dual_traversal(src, tgt; theta=0.5)
    miss0, dup0 = coverage(src, tgt, m2l, near)
    check("T4 baseline clean", miss0 == 0 && dup0 == 0)
    # (a) drop one near pair -> misses must appear
    miss_a, _ = coverage(src, tgt, m2l, near[1:end-1])
    check("T4a dropped near pair detected", miss_a > 0)
    # (b) duplicate one M2L pair -> dups must appear
    miss_b, dup_b = coverage(src, tgt, vcat(m2l, m2l[1:1]), near)
    check("T4b duplicated M2L pair detected", dup_b > 0 && miss_b == 0)
    # (c) replace an M2L pair's source cell by its PARENT (classic double-count
    #     via overlapping subtree): find a pair whose src cell has a parent
    parent_of = Dict{Int,Int}()
    for (ci, c) in enumerate(src.cells), ch in c.children
        parent_of[ch] = ci
    end
    idx = findfirst(((si, ti),) -> haskey(parent_of, si), m2l)
    if idx === nothing
        check("T4c parent-substitution detected (no candidate)", false)
    else
        (si, ti) = m2l[idx]
        m2l_bad = copy(m2l); m2l_bad[idx] = (parent_of[si], ti)
        miss_c, dup_c = coverage(src, tgt, m2l_bad, near)
        check("T4c parent-substitution detected", dup_c > 0)
    end
end

# ---------- T5: identical sets with internal duplicates, off-origin ----------
base = [SVector(1e6, -1e6, 5e5) .+ 0.001 .* rand(rng, SVector{3,Float64}) for _ in 1:600]
withdup = vcat(base, base[1:200])            # 200 exact internal duplicates
full_case("T5 identical sets w/ duplicates, 1e6 offset", withdup, copy(withdup); leaf_size=16, theta=0.5)
full_case("T5b theta=0.3", withdup, copy(withdup); leaf_size=16, theta=0.3)

# ---------- T6: production-snapshot FP-accept vs exact-strict audit ----------
const SNAPDIR = "/private/tmp/claude-502/-Users-ryan-Dropbox-research-projects-FastMultipole/1a9c539a-6d87-44ee-87f7-d4e1c17d2793/scratchpad/snapshot472"
if isdir(SNAPDIR)
    read_i64(f) = reinterpret(Int64, read(joinpath(SNAPDIR, f)))
    read_3xn(f) = reshape(reinterpret(Float64, read(joinpath(SNAPDIR, f))), 3, :)
    pos = read_3xn("particle_positions_3xN_f64.bin")
    verts = read_3xn("panel_vertices_3xM_f64.bin")
    conn = read_i64("panel_connectivity_i64.bin"); offs = read_i64("panel_offsets_i64.bin")
    srcs = let lo = 1
        [begin
             hi = offs[k]
             c = sum(SVector(verts[1, conn[j]], verts[2, conn[j]], verts[3, conn[j]]) for j in lo:hi) / (hi - lo + 1)
             lo = hi + 1; c
         end for k in 1:length(offs)]
    end
    tgts = [SVector(pos[1, i], pos[2, i], pos[3, i]) for i in 1:size(pos, 2)]
    for (ls, th) in ((256, 0.6), (32, 0.6))
        g = shared_grid(srcs, tgts)
        st = build_tree(g, srcs; leaf_size=ls); tt = build_tree(g, tgts; leaf_size=ls)
        m2l, near = dual_traversal(st, tt; theta=th)
        viol, ties = mac_audit(st, tt, m2l, th)
        check("T6 production leaf=$ls theta=$th strict-viol=0 (viol=$viol, ties=$ties)", viol == 0)
    end
else
    println("T6 SKIPPED: snapshot dir missing")
end

println()
if isempty(failures)
    println("REVIEWER TESTS: ALL PASSED")
else
    println("REVIEWER TESTS: FAILURES -> ", failures)
end
exit(isempty(failures) ? 0 : 1)
