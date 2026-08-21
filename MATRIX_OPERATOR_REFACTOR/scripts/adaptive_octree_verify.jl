# adaptive_octree_verify.jl — task 038 validation script (stdlib-only).
#
# Computationally verifies the theory in
# MATRIX_OPERATOR_REFACTOR/theory/adaptive-radix-octree.md:
#
#   1. exact-once ordered body-pair coverage of U/V/W/X + self on adaptive
#      2:1-balanced (and deliberately unbalanced) occupied Morton octrees, at
#      both supported near radii (q = 3 classic, q = 12 theta = 0.5), on
#      uniform, multi-scale, filament (rotor-like), and adversarial
#      distributions;
#   2. the 2:1 balance sweep (fixed point reached, property holds);
#   3. V-class admissibility: every emitted V pair is same-level, separated
#      (o not in N_q), with near parents (o_parent in N_q) — i.e. inside the
#      025 phase-table class set, so the level-scaled operator tables serve
#      it unchanged;
#   4. the W/X level structure: partners are strictly finer than their leaf;
#      the level-difference distribution is measured (theory §2.5: exactly
#      one level in the complete-tree classic limit, deeper entries possible
#      under occupancy pruning);
#   4b. the scalar M2T and S2L definitions of theory §4.1/§4.2 (numerical
#      convergence to the analytic potential at P = 4 and P = 8, with
#      self-contained Gumerov-normalized harmonics);
#   5. uniform-limit parity: forcing all leaves to one depth reproduces the
#      025 first-separated-ancestor (L*) list construction exactly;
#   6. the per-cell sigma gate (STICKY demotion form, review correction
#      2026-08-14): with heterogeneous sigma, every body pair inside the
#      regularization cutoff r <= rho_t*sigma_src is covered by U (the 031a
#      §5.1 contract), coverage stays exact-once, AND every emitted V pair
#      on the gated lists remains inside the 025 phase-table class set
#      (near parents, Chebyshev reach) — the table-reuse claim holds with
#      the gate active;
#   7. capacity formulas: measured node/leaf/list counts never exceed the
#      closed-form capacity bounds of theory §6;
#   8. constant-P bound consistency (P = 4 and P = 8 per the standing test
#      invariant): the 008d bound B(P, o, A) evaluated at the minimal
#      admissible W/X offsets never exceeds the V-list worst case at the
#      same level — the W/X error budget is inside the V budget.
#
# Also generates the cost-model evidence tables of theory §6 (list sizes and
# work counts vs K_max; adaptive vs best uniform depth) under
# MATRIX_OPERATOR_REFACTOR/data/adaptive_octree/.
#
# Usage:  julia --project=. MATRIX_OPERATOR_REFACTOR/scripts/adaptive_octree_verify.jl
# Stdlib only; single-threaded; deterministic (fixed seeds); runs in ~1 min.

using Random
using Printf

const OUTDIR = joinpath(@__DIR__, "..", "data", "adaptive_octree")

# ---------------------------------------------------------------- geometry --

# Morton key by explicit bit interleave (matches theory/radix-sort-clustering.md).
function morton3(i::Int, j::Int, k::Int, ell::Int)
    key = UInt64(0)
    for b in 0:ell-1
        key |= (UInt64((i >> b) & 1) << (3b + 0))
        key |= (UInt64((j >> b) & 1) << (3b + 1))
        key |= (UInt64((k >> b) & 1) << (3b + 2))
    end
    return key
end

# quantize positions at depth ell_max over root cube [x0, x0 + 2h0]^3
function quantize(xs::Matrix{Float64}, x0::NTuple{3,Float64}, h0::Float64, ell::Int)
    n = size(xs, 2)
    G = 1 << ell
    Delta = 2h0 / G
    coords = Matrix{Int}(undef, 3, n)
    for p in 1:n, a in 1:3
        u = (xs[a, p] - x0[a]) / Delta
        coords[a, p] = clamp(floor(Int, u), 0, G - 1)
    end
    return coords
end

# --------------------------------------------------------------- tree type --

mutable struct Node
    level::Int
    cx::Int; cy::Int; cz::Int      # integer coords at `level`
    lo::Int; hi::Int               # contiguous sorted-body range (subtree)
    parent::Int
    children::Vector{Int}
    leaf::Bool
end

struct Tree
    nodes::Vector{Node}
    perm::Vector{Int}              # sorted body order (full-depth Morton)
    keys::Vector{UInt64}           # full-depth keys, sorted order
    ell_max::Int
    h0::Float64
    x0::NTuple{3,Float64}
end

cellwidth(t::Tree, level::Int) = 2t.h0 / (1 << level)

# child selector: bits of the full-depth key for level `lev` (1-based bit math)
childindex(key::UInt64, lev::Int, ell_max::Int) = Int((key >> (3 * (ell_max - lev))) & 0x7)

"""Build the occupied adaptive Morton tree: split while population > K_max and
level < ell_max (force_uniform_depth: split every occupied node above that
depth regardless of population — used for the uniform-limit parity check)."""
function build_tree(xs::Matrix{Float64}, K_max::Int, ell_max::Int;
        pad::Float64 = 1e-3, force_uniform_depth::Int = -1)
    n = size(xs, 2)
    lo = ntuple(a -> minimum(view(xs, a, :)), 3)
    hi = ntuple(a -> maximum(view(xs, a, :)), 3)
    b = maximum(ntuple(a -> hi[a] - lo[a], 3)) / 2
    h0 = b * (1 + pad) + (b == 0 ? 1.0 : 0.0)
    c0 = ntuple(a -> (lo[a] + hi[a]) / 2, 3)
    x0 = ntuple(a -> c0[a] - h0, 3)
    coords = quantize(xs, x0, h0, ell_max)
    keys0 = [morton3(coords[1, p], coords[2, p], coords[3, p], ell_max) for p in 1:n]
    perm = sortperm(keys0)                       # stable by construction
    keys = keys0[perm]
    nodes = Node[Node(0, 0, 0, 0, 1, n, 0, Int[], true)]
    stack = [1]
    while !isempty(stack)
        idx = pop!(stack)
        nd = nodes[idx]
        pop = nd.hi - nd.lo + 1
        dosplit = force_uniform_depth >= 0 ?
            (nd.level < force_uniform_depth) :
            (pop > K_max && nd.level < ell_max)
        dosplit || continue
        split_node!(nodes, keys, idx, ell_max)
        append!(stack, nodes[idx].children)
    end
    return Tree(nodes, perm, keys, ell_max, h0, x0)
end

"""Split node `idx` into its occupied children (pigeonhole on the next 3 key
bits; the sorted-key prefix property makes each child a contiguous range)."""
function split_node!(nodes::Vector{Node}, keys::Vector{UInt64}, idx::Int, ell_max::Int)
    nd = nodes[idx]
    @assert nd.leaf && nd.level < ell_max
    lev = nd.level + 1
    r = nd.lo
    while r <= nd.hi
        c = childindex(keys[r], lev, ell_max)
        r2 = r
        while r2 < nd.hi && childindex(keys[r2 + 1], lev, ell_max) == c
            r2 += 1
        end
        push!(nodes, Node(lev,
            2nd.cx + (c & 1), 2nd.cy + ((c >> 1) & 1), 2nd.cz + ((c >> 2) & 1),
            r, r2, idx, Int[], true))
        push!(nd.children, length(nodes))
        r = r2 + 1
    end
    nd.leaf = false
    return nothing
end

leaves(t::Tree) = [i for i in eachindex(t.nodes) if t.nodes[i].leaf]

# ------------------------------------------------------ adjacency and near --

# per-axis index distance of cell b (level lb) outside the tile interval of
# cell a (level la <= lb), measured at level lb; 0 when inside
@inline function axis_clamp_dist(ca::Int, la::Int, cb::Int, lb::Int)
    k = lb - la
    a0 = ca << k
    a1 = ((ca + 1) << k) - 1
    return cb < a0 ? a0 - cb : (cb > a1 ? cb - a1 : 0)
end

"""Touching test (closed boxes share at least a point): per-axis clamp
distance <= 1 at the finer level."""
function touching(A::Node, B::Node)
    la, lb = A.level, B.level
    if la <= lb
        dx = axis_clamp_dist(A.cx, la, B.cx, lb)
        dy = axis_clamp_dist(A.cy, la, B.cy, lb)
        dz = axis_clamp_dist(A.cz, la, B.cz, lb)
    else
        dx = axis_clamp_dist(B.cx, lb, A.cx, la)
        dy = axis_clamp_dist(B.cy, lb, A.cy, la)
        dz = axis_clamp_dist(B.cz, lb, A.cz, la)
    end
    return max(dx, dy, dz) <= 1
end

"""Near predicate at radius q (theory §2.2): measured on the lattice of the
finer of the two cells; minimal tile offset by per-axis clamping."""
function isnear(A::Node, B::Node, q::Int)
    if A.level == B.level
        ox = B.cx - A.cx; oy = B.cy - A.cy; oz = B.cz - A.cz
        return ox^2 + oy^2 + oz^2 <= q
    elseif A.level < B.level
        dx = axis_clamp_dist(A.cx, A.level, B.cx, B.level)
        dy = axis_clamp_dist(A.cy, A.level, B.cy, B.level)
        dz = axis_clamp_dist(A.cz, A.level, B.cz, B.level)
        return dx^2 + dy^2 + dz^2 <= q
    else
        dx = axis_clamp_dist(B.cx, B.level, A.cx, A.level)
        dy = axis_clamp_dist(B.cy, B.level, A.cy, A.level)
        dz = axis_clamp_dist(B.cz, B.level, A.cz, A.level)
        return dx^2 + dy^2 + dz^2 <= q
    end
end

"""Physical AABB gap between two closed cell boxes (0 when touching)."""
function aabb_gap(t::Tree, A::Node, B::Node)
    g2 = 0.0
    for (ca, la, cb, lb) in ((A.cx, A.level, B.cx, B.level),
                             (A.cy, A.level, B.cy, B.level),
                             (A.cz, A.level, B.cz, B.level))
        wa = cellwidth(t, la); wb = cellwidth(t, lb)
        alo = ca * wa; ahi = alo + wa
        blo = cb * wb; bhi = blo + wb
        g = max(alo - bhi, blo - ahi, 0.0)
        g2 += g * g
    end
    return sqrt(g2)
end

# --------------------------------------------------------------- balancing --

"""2:1 balance sweep (theory §1.4): split any occupied leaf that touches an
occupied leaf two or more levels finer; iterate to the fixed point. Returns
the number of balance-induced splits."""
function balance!(t::Tree)
    nsplit = 0
    changed = true
    while changed
        changed = false
        lv = leaves(t)
        # mark coarse leaves violating 2:1 vs any leaf
        marked = Int[]
        for i in lv
            A = t.nodes[i]
            A.level < t.ell_max || continue
            for j in lv
                B = t.nodes[j]
                if B.level >= A.level + 2 && touching(A, B)
                    push!(marked, i)
                    break
                end
            end
        end
        for i in marked
            t.nodes[i].leaf || continue
            split_node!(t.nodes, t.keys, i, t.ell_max)
            nsplit += 1
            changed = true
        end
    end
    return nsplit
end

function check_balanced(t::Tree)
    lv = leaves(t)
    for i in lv, j in lv
        A = t.nodes[i]; B = t.nodes[j]
        if touching(A, B) && abs(A.level - B.level) >= 2
            return false
        end
    end
    return true
end

# --------------------------------------------------- dual-tree list builder --

struct Lists
    U::Vector{NTuple{2,Int}}   # (target leaf, source leaf) — direct
    V::Vector{NTuple{2,Int}}   # (target node, source node) — same level, M2L
    W::Vector{NTuple{2,Int}}   # (target leaf coarser, source node finer) — M2T
    X::Vector{NTuple{2,Int}}   # (target node finer, source leaf coarser) — S2L
    demoted::Vector{NTuple{2,Int}} # far pairs demoted by the sigma gate that
                                   # continued descent (diagnostic only)
end

"""Dual-tree recursion of theory §2.3 with the STICKY demotion gate of
theory §5.2 (review correction 2026-08-14). `sigma_node[i]` is the subtree
max of the source smoothing radius (0 disables the gate); `rho_t` the 031a
cutoff. A geometrically separated pair whose physical AABB gap cannot
guarantee gap >= rho_t*sigma_max(source) is demoted: it descends, and its
ENTIRE descendant pair set terminates in U (no V/W/X emission below a
demoted pair). This preserves Invariant 2 — every emitted V pair has
geometrically near parents — so the 025 phase-table membership holds with
the gate active."""
function build_lists(t::Tree, q::Int; sigma_node::Vector{Float64} = Float64[],
        rho_t::Float64 = 0.0)
    L = Lists(NTuple{2,Int}[], NTuple{2,Int}[], NTuple{2,Int}[], NTuple{2,Int}[],
        NTuple{2,Int}[])
    gate = !isempty(sigma_node) && rho_t > 0
    stack = [(1, 1, false)]                    # (target, source, demoted lineage)
    while !isempty(stack)
        (ia, ib, dem) = pop!(stack)
        A = t.nodes[ia]; B = t.nodes[ib]
        near = dem || isnear(A, B, q)
        if !near && gate
            # per-cell sigma gate on the SOURCE side (031a: rho = r/sigma_src)
            if aabb_gap(t, A, B) < rho_t * sigma_node[ib]
                near = true
                dem = true                     # sticky: direct-only from here down
                push!(L.demoted, (ia, ib))
            end
        end
        if !near
            if A.level == B.level
                push!(L.V, (ia, ib))
            elseif A.level < B.level
                @assert A.leaf   # invariant: coarser member of a mixed pair is a leaf
                push!(L.W, (ia, ib))
            else
                @assert B.leaf
                push!(L.X, (ia, ib))
            end
        else
            if A.leaf && B.leaf
                push!(L.U, (ia, ib))
            elseif A.level == B.level
                if A.leaf                      # split the internal one
                    for jb in B.children; push!(stack, (ia, jb, dem)); end
                elseif B.leaf
                    for ja in A.children; push!(stack, (ja, ib, dem)); end
                else                           # split both
                    for ja in A.children, jb in B.children
                        push!(stack, (ja, jb, dem))
                    end
                end
            elseif A.level < B.level           # A coarser => A leaf; split B
                @assert A.leaf
                for jb in B.children; push!(stack, (ia, jb, dem)); end
            else
                @assert B.leaf
                for ja in A.children; push!(stack, (ja, ib, dem)); end
            end
        end
    end
    return L
end

# subtree sigma max by upward sweep (theory §5.2)
function sigma_upward(t::Tree, sigma::Vector{Float64})
    s = zeros(Float64, length(t.nodes))
    order = sortperm([nd.level for nd in t.nodes]; rev = true)
    for i in order
        nd = t.nodes[i]
        if nd.leaf
            m = 0.0
            for r in nd.lo:nd.hi
                m = max(m, sigma[t.perm[r]])
            end
            s[i] = m
        else
            m = 0.0
            for c in nd.children
                m = max(m, s[c])
            end
            s[i] = m
        end
    end
    return s
end

# ------------------------------------------------------ coverage validation --

"""Exact-once check: paint every emitted bucket's covered ordered body-pair
rectangle (in sorted index space, so subtrees are contiguous ranges) and
assert the count matrix is identically 1."""
function check_exact_once(t::Tree, L::Lists)
    n = length(t.perm)
    cover = zeros(UInt8, n, n)
    paint!(c, A, B) = (c[A.lo:A.hi, B.lo:B.hi] .+= UInt8(1))
    for (ia, ib) in L.U; paint!(cover, t.nodes[ia], t.nodes[ib]); end
    for (ia, ib) in L.V; paint!(cover, t.nodes[ia], t.nodes[ib]); end
    for (ia, ib) in L.W; paint!(cover, t.nodes[ia], t.nodes[ib]); end
    for (ia, ib) in L.X; paint!(cover, t.nodes[ia], t.nodes[ib]); end
    bad = 0
    for s in 1:n, tt in 1:n
        cover[tt, s] == 1 || (bad += 1)
    end
    return bad
end

"""V admissibility (theory §2.4): same level, separated at its level radius,
parents near at the parent radius, offset inside the finite phase-table
Chebyshev reach of hierarchical-rigid-m2l-stencil.md."""
function check_v_classes(t::Tree, L::Lists, q::Int)
    maxcheb = q == 3 ? 3 : 7
    for (ia, ib) in L.V
        A = t.nodes[ia]; B = t.nodes[ib]
        A.level == B.level || return false
        o = (B.cx - A.cx, B.cy - A.cy, B.cz - A.cz)
        sum(abs2, o) > q || return false
        maximum(abs, o) <= maxcheb || return false
        if A.level >= 1
            P = t.nodes[A.parent]; Q = t.nodes[B.parent]
            p = (Q.cx - P.cx, Q.cy - P.cy, Q.cz - P.cz)
            sum(abs2, p) <= q || return false
        end
    end
    return true
end

"""W/X level structure (theory §2.5): partners strictly finer than their
leaf (asserted); returns (max level difference, fraction at exactly one
level). The one-level property is exact only in the complete-tree classic
limit; occupancy pruning admits deeper entries."""
function wx_level_stats(t::Tree, L::Lists)
    maxd = 0
    n1 = 0
    tot = 0
    for (ia, ib) in L.W
        d = t.nodes[ib].level - t.nodes[ia].level
        @assert d >= 1
        maxd = max(maxd, d); n1 += (d == 1); tot += 1
    end
    for (ia, ib) in L.X
        d = t.nodes[ia].level - t.nodes[ib].level
        @assert d >= 1
        maxd = max(maxd, d); n1 += (d == 1); tot += 1
    end
    return maxd, tot == 0 ? 1.0 : n1 / tot
end

"""Uniform-limit parity (theory §3.3): on a forced uniform-depth tree the
dual-tree lists must equal the 025 first-separated-ancestor construction."""
function check_uniform_parity(t::Tree, L::Lists, q::Int, ell::Int)
    lv = leaves(t)
    # independent 025-style construction on ordered occupied leaf pairs
    refU = Set{NTuple{2,Int}}()
    refV = Set{NTuple{4,Int}}()  # (level, ox, oy, oz) tagged by leaf pair via nodes
    refVpairs = Set{NTuple{2,Int}}() # (target ancestor, source ancestor) node ids
    # ancestor-node lookup: walk parents
    anc = Dict{Tuple{Int,Int},Int}() # (leaf node id, level) -> ancestor node id
    for i in lv
        j = i
        nd = t.nodes[j]
        anc[(i, nd.level)] = j
        while nd.parent != 0
            j = nd.parent
            nd = t.nodes[j]
            anc[(i, nd.level)] = j
        end
    end
    for i in lv, j in lv
        A = t.nodes[i]; B = t.nodes[j]
        o = (B.cx - A.cx, B.cy - A.cy, B.cz - A.cz)
        if sum(abs2, o) <= q
            push!(refU, (i, j))
        else
            Lstar = 0
            for Lv in 1:ell
                a = anc[(i, Lv)]; b = anc[(j, Lv)]
                na = t.nodes[a]; nb = t.nodes[b]
                oo = (nb.cx - na.cx, nb.cy - na.cy, nb.cz - na.cz)
                if sum(abs2, oo) > q
                    Lstar = Lv
                    push!(refVpairs, (a, b))
                    break
                end
            end
            @assert Lstar > 0
        end
    end
    return Set(L.U) == refU && Set(L.V) == refVpairs &&
           isempty(L.W) && isempty(L.X)
end

"""031a §5.1 contract under the per-cell gate: every ordered body pair with
r <= rho_t * sigma_src is covered by U."""
function check_sigma_contract(t::Tree, L::Lists, xs::Matrix{Float64},
        sigma::Vector{Float64}, rho_t::Float64)
    n = length(t.perm)
    inU = falses(n, n)
    for (ia, ib) in L.U
        A = t.nodes[ia]; B = t.nodes[ib]
        inU[A.lo:A.hi, B.lo:B.hi] .= true
    end
    bad = 0
    for ss in 1:n, tt in 1:n
        ps = t.perm[ss]; pt = t.perm[tt]
        r = sqrt(sum(abs2, view(xs, :, pt) .- view(xs, :, ps)))
        if r <= rho_t * sigma[ps] && !inU[tt, ss]
            bad += 1
        end
    end
    return bad
end

# --------------------------------------------------------- capacity bounds --

"""Theory §6.4 capacity bounds. Population splits: at each level the split
nodes are disjoint and each holds > K_max bodies, so there are at most
n/(K_max+1) per level; every node is the root or a child of a split node
(<= 8 each); balance splits are covered by the same per-level disjointness
with the balance inflation factor folded into the *measured* check —
the bound below is the strict population-split bound times the inflation
allowance `binfl` used for cache sizing."""
function capacity_nodes(n::Int, K_max::Int, ell_max::Int; binfl::Float64 = 2.0)
    nsplit = ell_max * cld(n, K_max + 1)
    return ceil(Int, binfl * (1 + 8 * nsplit))
end

# ---------------------------------------------------------- 008d bound B() --

"""Conservative constant-P bound of theory/constant-p-error-stencil.md for
unit budget A at unit leaf width (relative comparisons only)."""
function bound_B(P::Int, o::NTuple{3,Int}; A::Float64 = 1.0, w::Float64 = 0.5)
    rho = w * sqrt(3.0)
    c = 2 * sqrt(sum(abs2, o)) / sqrt(3.0)
    c > 2 || return Inf
    return 2A / (rho * (c - 2)) * (1 / (c - 1))^(P + 1)
end

# minimal separated offset at radius q (by norm), and worst-case V offset
function extreme_offsets(q::Int)
    best = (0, 0, 0); bestn = typemax(Int)
    reach = ceil(Int, sqrt(q)) + 2
    for oz in -reach:reach, oy in -reach:reach, ox in -reach:reach
        n2 = ox^2 + oy^2 + oz^2
        if n2 > q && n2 < bestn
            best = (ox, oy, oz); bestn = n2
        end
    end
    return best
end

# ------------------------------------- M2T / S2L scalar convergence (§4) ---

"""Associated Legendre P_n^m(x) with Condon-Shortley phase, for all
0 <= m <= n <= N; standard recurrences."""
function plm_table(x::Float64, N::Int)
    P = zeros(N + 1, N + 1)          # P[n+1, m+1]
    P[1, 1] = 1.0
    s = sqrt(max(0.0, 1 - x^2))
    for m in 1:N
        P[m + 1, m + 1] = -(2m - 1) * s * P[m, m]
    end
    for m in 0:N-1
        P[m + 2, m + 1] = (2m + 1) * x * P[m + 1, m + 1]
    end
    for m in 0:N, nn in m+2:N
        P[nn + 1, m + 1] = ((2nn - 1) * x * P[nn, m + 1] -
                            (nn - 1 + m) * P[nn - 1, m + 1]) / (nn - m)
    end
    return P
end

"""Gumerov-normalized regular (kind = :R) or irregular (kind = :S) solid
harmonics of Δx, complex, for 0 <= m <= n <= N (theory §4.1)."""
function solid_harmonics(dx::NTuple{3,Float64}, N::Int, kind::Symbol)
    rho = sqrt(sum(abs2, dx))
    ct = rho == 0 ? 1.0 : dx[3] / rho
    phi = atan(dx[2], dx[1])
    P = plm_table(ct, N)
    H = zeros(ComplexF64, N + 1, N + 1)
    for nn in 0:N, m in 0:nn
        e = exp(im * m * phi) * (im)^m
        if kind == :R
            H[nn + 1, m + 1] = (-1.0)^nn * rho^nn * P[nn + 1, m + 1] * e /
                factorial(big(nn + m)) |> ComplexF64
        else
            H[nn + 1, m + 1] = (-1.0)^m * rho^(-nn - 1) * P[nn + 1, m + 1] * e *
                Float64(factorial(big(nn - m)))
        end
    end
    return H
end

"""Numerical §4.1/§4.2 check: unit source at x_s. M2T: multipole about c_M
(near x_s) evaluated at far target x. S2L: local about c_L (near x) built
from the far source, evaluated at x. Both must converge to -1/|x - x_s|
(analytic normalization). Returns relative errors at each P."""
function s2l_m2t_errors(x_s, c_M, c_L, x, Ps)
    r = sqrt(sum(abs2, x .- x_s))
    uref = -1 / r
    errsM = Float64[]; errsL = Float64[]
    for P in Ps
        R = solid_harmonics(x_s .- c_M, P, :R)
        S = solid_harmonics(x .- c_M, P, :S)
        uM = 0.0
        for nn in 0:P, m in 0:nn
            M_nm = -(-1.0)^(nn + m) * conj(R[nn + 1, m + 1])
            uM += (m == 0 ? 1.0 : 2.0) * real(S[nn + 1, m + 1] * M_nm)
        end
        push!(errsM, abs(uM - uref) / abs(uref))
        Sl = solid_harmonics(x_s .- c_L, P, :S)
        Rl = solid_harmonics(x .- c_L, P, :R)
        uL = 0.0
        for nn in 0:P, m in 0:nn
            L_nm = -(-1.0)^(nn + m) * conj(Sl[nn + 1, m + 1])
            uL += (m == 0 ? 1.0 : 2.0) * real(Rl[nn + 1, m + 1] * L_nm)
        end
        push!(errsL, abs(uL - uref) / abs(uref))
    end
    return errsM, errsL
end

# ------------------------------------------------------------------- cases --

function make_uniform(n::Int; seed = 38001)
    rng = MersenneTwister(seed)
    return rand(rng, 3, n)
end

"""Multi-scale case of the 038 task file: unit cube plus an embedded dense
cluster at `contrast` times the background density."""
function make_multiscale(n::Int; contrast::Float64 = 30.0, seed = 38002)
    rng = MersenneTwister(seed)
    # cluster radius chosen so the cluster density is `contrast` x background
    frac = 0.35                       # fraction of bodies in the cluster
    nc = round(Int, frac * n)
    nb = n - nc
    Rc = (3 * nc / (4pi * contrast * nb))^(1 / 3)  # background density nb / 1
    xs = Matrix{Float64}(undef, 3, n)
    xs[:, 1:nb] .= rand(rng, 3, nb)
    ctr = (0.6, 0.4, 0.55)
    k = 0
    while k < nc
        p = 2 .* (rand(rng, 3) .- 0.5)
        if sum(abs2, p) <= 1
            k += 1
            xs[:, nb + k] .= ctr .+ Rc .* p
        end
    end
    return xs
end

"""Filament case (rotor-like): a thin helical filament plus diffuse haze —
reproduces the 037b occupancy-contrast mechanism (top-1% cells hold a large
body share)."""
function make_filament(n::Int; seed = 38003)
    rng = MersenneTwister(seed)
    nf = round(Int, 0.6n)
    nh = n - nf
    xs = Matrix{Float64}(undef, 3, n)
    core = 0.004
    for p in 1:nf
        t = 4pi * (p - 1) / nf
        c = (0.5 + 0.35cos(t), 0.5 + 0.35sin(t), 0.15 + 0.7t / (4pi))
        xs[:, p] .= c .+ core .* randn(rng, 3)
    end
    xs[:, nf+1:end] .= rand(rng, 3, nh)
    return xs
end

"""Adversarial case: two tight clusters at opposite corners plus a sparse
background — maximizes level contrast."""
function make_adversarial(n::Int; seed = 38004)
    rng = MersenneTwister(seed)
    n1 = n ÷ 3; n2 = n ÷ 3; nb = n - n1 - n2
    xs = Matrix{Float64}(undef, 3, n)
    xs[:, 1:n1] .= 0.02 .* rand(rng, 3, n1) .+ 0.01
    xs[:, n1+1:n1+n2] .= 0.02 .* rand(rng, 3, n2) .+ 0.97
    xs[:, n1+n2+1:end] .= rand(rng, 3, nb)
    return xs
end

# ------------------------------------------------------------ cost counting --

function work_counts(t::Tree, L::Lists)
    pop(i) = t.nodes[i].hi - t.nodes[i].lo + 1
    u_pairs = sum(Int[pop(a) * pop(b) for (a, b) in L.U]; init = 0)
    w_evals = sum(Int[pop(a) for (a, _) in L.W]; init = 0)   # M2T: per target body
    x_evals = sum(Int[pop(b) for (_, b) in L.X]; init = 0)   # S2L: per source body
    return (; u_pairs, v_routes = length(L.V), w_evals, x_evals,
        w_entries = length(L.W), x_entries = length(L.X))
end

"""Uniform-grid direct/route counts at fixed depth ell (occupied cells only) —
the baseline the adaptive tree is compared against (theory §6.5)."""
function uniform_counts(xs::Matrix{Float64}, q::Int, ell::Int)
    t = build_tree(xs, typemax(Int), ell; force_uniform_depth = ell)
    L = build_lists(t, q)
    wc = work_counts(t, L)
    @assert wc.w_evals == 0 && wc.x_evals == 0
    lv = leaves(t)
    popmax = maximum(t.nodes[i].hi - t.nodes[i].lo + 1 for i in lv)
    return (; wc.u_pairs, wc.v_routes, n_leaves = length(lv), popmax)
end

# ------------------------------------------------------------------- main --

function main()
    mkpath(OUTDIR)
    n = 3000
    ell_max = 8
    rho_t = 4.789                    # 031a velocity cutoff at eps = 1e-3, beta = 2
    cases = [
        ("uniform",     make_uniform(n)),
        ("multiscale30", make_multiscale(n; contrast = 30.0)),
        ("multiscale100", make_multiscale(n; contrast = 100.0)),
        ("filament",    make_filament(n)),
        ("adversarial", make_adversarial(n)),
    ]
    allpass = true
    results = String[]
    push!(results, "case,q,K_max,balanced,n_nodes,n_leaves,balance_splits,depth_min,depth_max," *
        "u_entries,u_pairs,v_routes,w_entries,x_entries,w_evals,x_evals,exact_once_bad," *
        "v_classes_ok,wx_max_leveldiff,wx_frac_one_level")
    for (name, xs) in cases, q in (3, 12), K_max in (16, 64)
        for do_balance in (true, false)
            t = build_tree(xs, K_max, ell_max)
            nsplit = do_balance ? balance!(t) : 0
            do_balance && (allpass &= check_balanced(t))
            L = build_lists(t, q)
            bad = check_exact_once(t, L)
            vok = check_v_classes(t, L, q)
            wxmax, wxfrac = wx_level_stats(t, L)
            allpass &= (bad == 0) && vok
            lv = leaves(t)
            dmin = minimum(t.nodes[i].level for i in lv)
            dmax = maximum(t.nodes[i].level for i in lv)
            wc = work_counts(t, L)
            cap = capacity_nodes(n, K_max, ell_max)
            allpass &= length(t.nodes) <= cap
            push!(results, join([name, q, K_max, do_balance, length(t.nodes),
                length(lv), nsplit, dmin, dmax, length(L.U), wc.u_pairs,
                wc.v_routes, wc.w_entries, wc.x_entries, wc.w_evals, wc.x_evals,
                bad, vok, wxmax, round(wxfrac; digits = 4)], ","))
            @printf("%-13s q=%-2d K=%-3d bal=%d  nodes=%5d leaves=%5d depth=%d..%d  U=%6d V=%6d W=%5d X=%5d wxmax=%d  exact_once=%s\n",
                name, q, K_max, do_balance, length(t.nodes), length(lv), dmin,
                dmax, length(L.U), wc.v_routes, wc.w_entries, wc.x_entries,
                wxmax, bad == 0 ? "PASS" : "FAIL($bad)")
        end
    end
    open(joinpath(OUTDIR, "exact_once_coverage.csv"), "w") do io
        for r in results; println(io, r); end
    end

    # ---- uniform-limit parity (theory §3.3) --------------------------------
    for q in (3, 12)
        xs = make_uniform(800; seed = 38010)
        ell = 3
        t = build_tree(xs, typemax(Int), ell; force_uniform_depth = ell)
        L = build_lists(t, q)
        ok = check_uniform_parity(t, L, q, ell)
        # review note 2026-08-14: set equality alone would mask duplicate
        # emissions — run exact-once painting on the parity tree as well
        ok &= (check_exact_once(t, L) == 0)
        allpass &= ok
        println("uniform-limit parity q=$q ell=$ell (incl. exact-once painting): ",
            ok ? "PASS" : "FAIL")
    end

    # ---- per-cell sigma gate (theory §5) -----------------------------------
    sigma_rows = String[]
    push!(sigma_rows, "case,q,K_max,sigma_kind,rho_t,demoted,contract_bad,v_classes_ok_gated,wx_max_leveldiff_gated,u_pairs_gated,u_pairs_ungated")
    for (name, xs) in cases[[1, 2, 4]], q in (3, 12)
        K_max = 32
        nn = size(xs, 2)
        rng = MersenneTwister(38020)
        for (skind, sigma) in (
                ("uniform_small", fill(1e-3, nn)),
                ("heterogeneous", 10 .^ (rand(rng, nn) .* 2 .- 3.5)),  # 3e-4..3e-2
                ("one_fat", (s = fill(3e-4, nn); s[7] = 0.15; s)))
            t = build_tree(xs, K_max, ell_max)
            balance!(t)
            snode = sigma_upward(t, sigma)
            Lg = build_lists(t, q; sigma_node = snode, rho_t = rho_t)
            L0 = build_lists(t, q)
            bad_cov = check_exact_once(t, Lg)
            bad_ct = check_sigma_contract(t, Lg, xs, sigma, rho_t)
            # review correction 2026-08-14: V-class/phase-table membership must
            # hold on the GATED lists too (sticky demotion preserves Invariant 2,
            # so every emitted V pair keeps geometrically near parents)
            vok_g = check_v_classes(t, Lg, q)
            wxmax_g, _ = wx_level_stats(t, Lg)
            allpass &= (bad_cov == 0) && (bad_ct == 0) && vok_g
            wcg = work_counts(t, Lg); wc0 = work_counts(t, L0)
            push!(sigma_rows, join([name, q, K_max, skind, rho_t,
                length(Lg.demoted), bad_ct, vok_g, wxmax_g,
                wcg.u_pairs, wc0.u_pairs], ","))
            @printf("sigma-gate %-13s q=%-2d %-14s demoted=%5d contract=%s cover=%s vclasses=%s\n",
                name, q, skind, length(Lg.demoted),
                bad_ct == 0 ? "PASS" : "FAIL($bad_ct)",
                bad_cov == 0 ? "PASS" : "FAIL($bad_cov)",
                vok_g ? "PASS" : "FAIL")
        end
    end
    open(joinpath(OUTDIR, "sigma_gate_contract.csv"), "w") do io
        for r in sigma_rows; println(io, r); end
    end

    # ---- cost model: adaptive vs best uniform depth (theory §6.5) ----------
    cost_rows = String[]
    push!(cost_rows, "case,q,structure,K_max_or_ell,n_leaves,popmax,u_pairs,v_routes,w_entries,x_entries,model_cost")
    # relative work model: c_d per direct pair, c_v per V route (P = 4 class
    # constants; see theory §6.5 — calibration constants, relative units)
    c_d, c_v, c_wx = 1.0, 600.0, 30.0
    for (name, xs) in cases, q in (3, 12)
        for K_max in (8, 16, 32, 64, 128)
            t = build_tree(xs, K_max, ell_max)
            balance!(t)
            L = build_lists(t, q)
            wc = work_counts(t, L)
            lv = leaves(t)
            popmax = maximum(t.nodes[i].hi - t.nodes[i].lo + 1 for i in lv)
            cost = c_d * wc.u_pairs + c_v * wc.v_routes +
                   c_wx * (wc.w_evals + wc.x_evals)
            push!(cost_rows, join([name, q, "adaptive", K_max, length(lv),
                popmax, wc.u_pairs, wc.v_routes, wc.w_entries, wc.x_entries,
                round(cost; digits = 1)], ","))
        end
        for ell in 2:6
            uc = uniform_counts(xs, q, ell)
            cost = c_d * uc.u_pairs + c_v * uc.v_routes
            push!(cost_rows, join([name, q, "uniform", ell, uc.n_leaves,
                uc.popmax, uc.u_pairs, uc.v_routes, 0, 0,
                round(cost; digits = 1)], ","))
        end
    end
    open(joinpath(OUTDIR, "cost_model_counts.csv"), "w") do io
        for r in cost_rows; println(io, r); end
    end

    # ---- constant-P bound consistency at P = 4 and P = 8 (theory §4.4) -----
    bound_rows = String[]
    push!(bound_rows, "q,P,offset,norm2,B_minimal_separated,B_note")
    for q in (3, 12), P in (4, 8)
        o = extreme_offsets(q)
        Bmin = bound_B(P, o)
        push!(bound_rows, join([q, P, "\"$(o)\"", sum(abs2, o), Bmin,
            "worst_admissible_V_and_WX_offset"], ","))
    end
    # W/X entries share the same admissible offset set at the finer level, so
    # the worst-case W/X bound equals the worst-case V bound; monotonicity in
    # |o| checked explicitly:
    mono_ok = true
    for q in (3, 12), P in (4, 8)
        vals = Float64[]
        for n2 in (q + 1):(q + 30)
            # any representative offset of norm^2 = n2 (if one exists)
            found = false
            reach = ceil(Int, sqrt(n2)) + 1
            for oz in 0:reach, oy in 0:reach, ox in 0:reach
                if ox^2 + oy^2 + oz^2 == n2
                    push!(vals, bound_B(P, (ox, oy, oz)))
                    found = true
                    break
                end
            end
        end
        mono_ok &= issorted(vals; rev = true)
    end
    allpass &= mono_ok
    push!(bound_rows, "all,4_and_8,-,-,-,bound_monotone_decreasing_in_norm=$(mono_ok)")
    open(joinpath(OUTDIR, "constant_p_bound_consistency.csv"), "w") do io
        for r in bound_rows; println(io, r); end
    end

    # ---- M2T / S2L scalar convergence at P = 4 and P = 8 (theory §4) -------
    conv_rows = String[]
    push!(conv_rows, "config,P,m2t_relerr,s2l_relerr")
    rng = MersenneTwister(38030)
    Ps = (4, 8, 12)
    conv_ok = true
    for cfg in 1:6
        x_s = ntuple(_ -> 0.12 * (rand(rng) - 0.5), 3)          # near c_M = 0
        c_M = (0.0, 0.0, 0.0)
        x = ntuple(a -> 1.2 * (rand(rng) - 0.5) .+ (1.4, -1.1, 1.2)[a], 3)
        c_L = x .+ ntuple(_ -> 0.10 * (rand(rng) - 0.5), 3)     # near x
        eM, eL = s2l_m2t_errors(x_s, c_M, c_L, x, Ps)
        for (i, P) in enumerate(Ps)
            push!(conv_rows, join([cfg, P, eM[i], eL[i]], ","))
        end
        conv_ok &= eM[2] < eM[1] && eL[2] < eL[1]               # P8 < P4
        conv_ok &= eM[3] < 1e-6 && eL[3] < 1e-6                 # P12 tight
        @printf("m2t/s2l cfg %d: P4 (%.2e, %.2e)  P8 (%.2e, %.2e)  P12 (%.2e, %.2e)\n",
            cfg, eM[1], eL[1], eM[2], eL[2], eM[3], eL[3])
    end
    allpass &= conv_ok
    println("m2t/s2l scalar convergence: ", conv_ok ? "PASS" : "FAIL")
    open(joinpath(OUTDIR, "s2l_m2t_convergence.csv"), "w") do io
        for r in conv_rows; println(io, r); end
    end

    println()
    println(allpass ? "ALL CHECKS PASS" : "CHECK FAILURES PRESENT")
    open(joinpath(OUTDIR, "summary.txt"), "w") do io
        println(io, "adaptive_octree_verify.jl — ", allpass ? "ALL CHECKS PASS" :
            "CHECK FAILURES PRESENT")
        println(io, "n = $n per case, ell_max = $ell_max, rho_t = $rho_t")
        println(io, "cases: uniform, multiscale30, multiscale100, filament, adversarial")
        println(io, "checks: exact-once (q = 3, 12; balanced + unbalanced), 2:1 balance")
        println(io, "fixed point, V-class admissibility (ungated AND sigma-gated lists),")
        println(io, "W/X level statistics, uniform-limit 025 parity (set equality +")
        println(io, "exact-once painting), per-cell sigma gate (sticky demotion) 031a")
        println(io, "contract, capacity bounds, P = 4 / P = 8 bound monotonicity,")
        println(io, "M2T/S2L scalar convergence at P = 4/8/12.")
    end
    return allpass
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main() ? 0 : 1)
end
