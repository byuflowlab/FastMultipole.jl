"""
SharedRadix — host-side prototype of the shared-radix dual-tree list machinery
for the Phase 2b-revised device FMM design (052d, 2026-08-27).

All octrees are sparse subsets of ONE implied global radix grid: a shared root
box and shared per-level spacing. A cell is identified by (level, Morton code);
its center and half-width are computed on the fly by index arithmetic — no
stored cell geometry. Cross-tree M2L and near-field lists are produced by an
adaptive dual traversal using only (level, code) arithmetic for geometry.

Plain Julia: Base + StaticArrays only.
"""
module SharedRadix

using StaticArrays
using Base.Threads
using LinearAlgebra: norm

export Grid, Tree, Cell, cell_center, cell_halfwidth, build_tree, shared_grid,
       dual_traversal, cell_id, morton_encode, morton_decode, MAX_LEVEL,
       exact_mac_leq

# ------------------------------------------------------------------
# 1. Implied global grid + Morton arithmetic
# ------------------------------------------------------------------

"Maximum tree depth: 21 bits per dimension -> 63-bit Morton codes in UInt64."
const MAX_LEVEL = 21

"Spread the low 21 bits of x so they occupy every 3rd bit."
@inline function spread3(x::UInt64)
    x &= 0x00000000001fffff
    x = (x | x << 32) & 0x001f00000000ffff
    x = (x | x << 16) & 0x001f0000ff0000ff
    x = (x | x << 8)  & 0x100f00f00f00f00f
    x = (x | x << 4)  & 0x10c30c30c30c30c3
    x = (x | x << 2)  & 0x1249249249249249
    return x
end

"Inverse of spread3."
@inline function compact3(x::UInt64)
    x &= 0x1249249249249249
    x = (x | x >> 2)  & 0x10c30c30c30c30c3
    x = (x | x >> 4)  & 0x100f00f00f00f00f
    x = (x | x >> 8)  & 0x001f0000ff0000ff
    x = (x | x >> 16) & 0x001f00000000ffff
    x = (x | x >> 32) & 0x00000000001fffff
    return x
end

"Interleave three 21-bit integer coordinates into a 63-bit Morton code."
@inline morton_encode(ix::UInt64, iy::UInt64, iz::UInt64) =
    spread3(ix) | (spread3(iy) << 1) | (spread3(iz) << 2)

"Recover (ix, iy, iz) from a Morton code with 3*nbits significant bits."
@inline morton_decode(code::UInt64) =
    (compact3(code), compact3(code >> 1), compact3(code >> 2))

"""
    Grid{T}

The one implied global radix grid: root box center + half-width. Level-l cells
have half-width `halfwidth/2^l`; the level-l lattice is 2^l cells per side.
"""
struct Grid{T}
    center::SVector{3,T}
    halfwidth::T
end

"""
    shared_grid(pointsets...) -> Grid

Cubic root box covering the union of all point sets, inflated slightly so no
point sits exactly on the upper boundary.
"""
function shared_grid(pointsets::Vararg{AbstractVector{SVector{3,T}}}) where {T}
    lo = SVector{3,T}(Inf, Inf, Inf)
    hi = SVector{3,T}(-Inf, -Inf, -Inf)
    for pts in pointsets, p in pts
        lo = min.(lo, p)
        hi = max.(hi, p)
    end
    c = (lo .+ hi) ./ 2
    h = maximum(hi .- lo) / 2
    h = h > 0 ? h * (1 + T(1e-6)) : one(T)   # degenerate (all coincident) -> unit box
    return Grid{T}(c, h)
end

"Half-width of any level-`level` cell — pure arithmetic, no storage."
@inline cell_halfwidth(g::Grid, level::Integer) = g.halfwidth / (1 << level)

"""
    cell_center(g, level, code) -> SVector{3}

Center of the level-`level` cell with level-local Morton code `code`
(i.e. the leading 3*level bits of a full-depth code). Pure index arithmetic.
"""
@inline function cell_center(g::Grid{T}, level::Integer, code::UInt64) where {T}
    ix, iy, iz = morton_decode(code)
    h = cell_halfwidth(g, level)
    lo = g.center .- g.halfwidth
    return SVector{3,T}(lo[1] + (2 * ix + 1) * h,
                        lo[2] + (2 * iy + 1) * h,
                        lo[3] + (2 * iz + 1) * h)
end

"Full-depth (MAX_LEVEL) Morton code of a point (clamped into the root box)."
@inline function point_code(g::Grid{T}, p::SVector{3,T}) where {T}
    n = UInt64(1) << MAX_LEVEL
    w = 2 * g.halfwidth / n            # finest cell width
    lo = g.center .- g.halfwidth
    ix = clamp(floor(Int64, (p[1] - lo[1]) / w), 0, Int64(n) - 1) % UInt64
    iy = clamp(floor(Int64, (p[2] - lo[2]) / w), 0, Int64(n) - 1) % UInt64
    iz = clamp(floor(Int64, (p[3] - lo[3]) / w), 0, Int64(n) - 1) % UInt64
    return morton_encode(ix, iy, iz)
end

# ------------------------------------------------------------------
# 2. Sparse adaptive tree on the shared grid
# ------------------------------------------------------------------

"""
    Cell

One sparse cell: (level, level-local code) identifies it on the implied global
grid; `range` is the contiguous slice of the tree's sorted point order lying in
this cell (which equals its full subtree's points, since children partition the
parent range); `children` is a range of cell indices (empty -> leaf).
"""
struct Cell
    level::Int32
    code::UInt64
    range::UnitRange{Int}
    children::UnitRange{Int}
end

@inline isleaf(c::Cell) = isempty(c.children)
"Grid identity of a cell — used for cross-run list comparison."
@inline cell_id(c::Cell) = (c.level, c.code)

"""
    Tree{T}

Sparse adaptive octree over one point set, on the shared grid `grid`.
`order[k]` is the original index of the k-th point in Morton-sorted order;
`codes` are the full-depth codes in sorted order. No cell geometry is stored.
"""
struct Tree{T}
    grid::Grid{T}
    cells::Vector{Cell}
    order::Vector{Int}
    codes::Vector{UInt64}
end

"""
    build_tree(grid, points; leaf_size) -> Tree

Morton-sort the points, then split every cell holding more than `leaf_size`
points (breadth-first; empty children are skipped; splitting stops at
MAX_LEVEL, so coincident points force an oversized leaf there). Cell 1 is the
root (level 0, code 0). Trees built for different point sets on the same
`grid` share the identical implied lattice by construction.
"""
function build_tree(grid::Grid{T}, points::AbstractVector{SVector{3,T}};
                    leaf_size::Int=32) where {T}
    n = length(points)
    codes_unsorted = Vector{UInt64}(undef, n)
    @threads for i in 1:n
        codes_unsorted[i] = point_code(grid, points[i])
    end
    order = sortperm(codes_unsorted)
    codes = codes_unsorted[order]

    cells = Cell[]
    push!(cells, Cell(0, UInt64(0), 1:n, 1:0))
    i = 1
    while i <= length(cells)
        c = cells[i]
        if length(c.range) > leaf_size && c.level < MAX_LEVEL
            clevel = c.level + Int32(1)
            shift = 3 * (MAX_LEVEL - Int(clevel))
            first_child = length(cells) + 1
            lo = first(c.range)
            hi = last(c.range)
            while lo <= hi
                pref = codes[lo] >> shift            # child code of this run
                # find end of the run of equal child prefixes (codes sorted)
                hi_run = searchsortedlast(codes, ((pref + 1) << shift) - 1, lo, hi,
                                          Base.Order.Forward)
                push!(cells, Cell(clevel, pref, lo:hi_run, 1:0))
                lo = hi_run + 1
            end
            cells[i] = Cell(c.level, c.code, c.range, first_child:length(cells))
        end
        i += 1
    end
    return Tree{T}(grid, cells, order, codes)
end

# ------------------------------------------------------------------
# 3. Cross-tree adaptive dual traversal
# ------------------------------------------------------------------

"""
    dual_traversal(src::Tree, tgt::Tree; theta=0.5) -> (m2l, near)

Adaptive dual-tree traversal over two sparse subsets of the same grid, using
ONLY (level, code) arithmetic + on-the-fly centers for geometry.

For a candidate pair (S, T):
- Accept as M2L if the Barba-style MAC holds: `r_S + r_T < theta * dist`,
  with `r = sqrt(3) * cell_halfwidth(level)` (circumscribed-sphere radius of
  the raw grid box — no shrinking) and `dist = |center_S - center_T|`.
- Else if both are leaves: near-field pair.
- Else DESCEND THE CELL WITH THE LARGER GRID HALF-WIDTH, i.e. the one at the
  SHALLOWER LEVEL (ties descend the source); if that cell is a leaf, descend
  the other. This is the level-heterogeneity rule: a tiny deep-level source
  cloud embedded in a wide target cloud is handled by repeatedly refining the
  coarse side until levels are commensurate or the MAC accepts, so accepted
  pairs may sit at DIFFERENT levels and the recursion still partitions
  points(S_root) × points(T_root) exactly (each step replaces one side by the
  disjoint cover of its children).

Overlapping/identical cells give dist small or 0 -> MAC fails -> descent, so
the shared root (both trees start at level 0, code 0) is handled naturally.

Returns vectors of (src_cell_index, tgt_cell_index) pairs: `m2l` (any levels)
and `near` (both leaves).
"""
function dual_traversal(src::Tree{T}, tgt::Tree{T}; theta::Real=0.5) where {T}
    g = src.grid
    @assert g.center == tgt.grid.center && g.halfwidth == tgt.grid.halfwidth "trees must share the grid"
    m2l = Tuple{Int,Int}[]
    near = Tuple{Int,Int}[]
    isempty(src.codes) && return m2l, near
    isempty(tgt.codes) && return m2l, near
    sqrt3 = sqrt(T(3))
    stack = [(1, 1)]
    while !isempty(stack)
        (si, ti) = pop!(stack)
        S = src.cells[si]
        Tc = tgt.cells[ti]
        hs = cell_halfwidth(g, S.level)
        ht = cell_halfwidth(g, Tc.level)
        d = norm(cell_center(g, S.level, S.code) - cell_center(g, Tc.level, Tc.code))
        if sqrt3 * (hs + ht) < theta * d
            push!(m2l, (si, ti))
        elseif isleaf(S) && isleaf(Tc)
            push!(near, (si, ti))
        elseif isleaf(Tc) || (!isleaf(S) && hs >= ht)
            for ci in S.children
                push!(stack, (ci, ti))
            end
        else
            for ci in Tc.children
                push!(stack, (si, ci))
            end
        end
    end
    return m2l, near
end

"""
    exact_mac_leq(S::Cell, T::Cell, θ_num, θ_den) -> Bool

EXACT (integer-arithmetic) evaluation of the non-strict Barba MAC
`r_S + r_T <= θ * dist` with `r = sqrt(3) * halfwidth` and rational
`θ = θ_num/θ_den`. On the shared grid every cell center is an odd-integer
multiple of the level-L fine half-width (L = max of the two levels), so with
integer center offsets Δ and integer half-width sum s = 2^(L-l_S) + 2^(L-l_T),
the MAC is `3 θ_den² s² <= θ_num² |Δ|²` — no floating point, no boundary-tie
ambiguity. (Distances on a shared grid are quantized, so exact ties
`r_S + r_T = θ·dist` genuinely occur; the FP traversal may classify a tie
either way, which only moves the pair between the M2L and near/descend
branches and never breaks the partition.)
"""
function exact_mac_leq(S::Cell, T::Cell, θ_num::Integer, θ_den::Integer)
    L = max(S.level, T.level)
    ixs, iys, izs = morton_decode(S.code)
    ixt, iyt, izt = morton_decode(T.code)
    ss = Int128(1) << (L - S.level)   # S half-width in fine (level-L) half-widths
    st = Int128(1) << (L - T.level)
    # center coordinate in fine half-widths: (2i+1) * 2^(L-l)
    dx = (2 * Int128(ixs) + 1) * ss - (2 * Int128(ixt) + 1) * st
    dy = (2 * Int128(iys) + 1) * ss - (2 * Int128(iyt) + 1) * st
    dz = (2 * Int128(izs) + 1) * ss - (2 * Int128(izt) + 1) * st
    s = ss + st
    return 3 * Int128(θ_den)^2 * s^2 <= Int128(θ_num)^2 * (dx^2 + dy^2 + dz^2)
end

end # module
