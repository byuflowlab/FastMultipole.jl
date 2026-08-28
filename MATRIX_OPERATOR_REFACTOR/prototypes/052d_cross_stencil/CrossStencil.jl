"""
CrossStencil — two-occupancy uniform-q rigid-stencil list machinery for the
052d cross pass (panels → particles) on the shared radix grid (P3.1).

Reuses the shipped classifier tables (`FastMultipole.RigidHierarchicalTables`)
and the SharedRadix Morton arithmetic. Lists follow the production convention
(interaction_list_batched.jl:578-642): routes are SOURCE-major, target cell =
source coord + offset, push-set membership keyed by the SOURCE cell's parity
phase; near field at the leaf level `ell_x` is the `|o|^2 <= q` shell.

Uniform-q schedule, first M2L level = 2 (level-1 offsets satisfy |o|^2 <= 3
<= q, so the level-2 emission — parent near, child far — needs nothing above
it). Exact-once rests on near-set monotonicity: |fld(c+phi,2)| <= |c| per
component, so the set of levels at which a pair's offset is near is a prefix
from the root, and coverage count = [near at leaf] + #{level: far here, near
at parent} = 1 always (the P3.3 argument).
"""
module CrossStencil

using StaticArrays
using FastMultipole: RigidHierarchicalTables, _rigid_phase_index, _rigid_near

include(joinpath(@__DIR__, "..", "052d_shared_radix", "SharedRadix.jl"))
using .SharedRadix: morton_encode, morton_decode

export CrossGrid, LevelCells, build_level_cells, sweep_config, brute_coverage,
       coverage_counts_subset

"Shared grid: production formula (tree_batched.jl:35-40) — cubic box from the
PARTICLE bounds (self-pass box), x_min = center - h0."
struct CrossGrid{T}
    x_min::SVector{3,T}
    h0::T
end

function CrossGrid(particles::AbstractVector{SVector{3,T}}) where {T}
    lo = SVector{3,T}(Inf, Inf, Inf); hi = -lo
    for p in particles
        lo = min.(lo, p); hi = max.(hi, p)
    end
    c = (lo .+ hi) ./ 2
    h0 = maximum(hi .- lo) / 2
    return CrossGrid{T}(c .- h0, h0)
end

"Integer lattice coords at level `ell`; returns (coords, excursion) where
excursion > 0 means the point sat outside the box by that many cell widths
before clamping (R1 containment assert uses it)."
@inline function level_coords(g::CrossGrid{T}, p::SVector{3,T}, ell::Int) where {T}
    G = 1 << ell
    w = 2 * g.h0 / G
    exc = 0.0
    ix = MVector{3,Int}(0, 0, 0)
    for a in 1:3
        f = (p[a] - g.x_min[a]) / w
        i = floor(Int, f)
        i < 0 && (exc = max(exc, -f); i = 0)
        i > G - 1 && (exc = max(exc, f - G); i = G - 1)
        ix[a] = i
    end
    return SVector{3,Int}(ix), exc
end

"""
Per-level occupied cells of one body set. `codes` sorted unique level-local
Morton codes; body k in sorted order occupies cell i iff starts[i] <= k <
starts[i+1]. `order` = body permutation into Morton-sorted order (leaf level;
shared by all levels since level codes are prefixes).
"""
struct LevelCells
    codes::Vector{UInt64}
    starts::Vector{Int}
end

ncells(lc::LevelCells) = length(lc.codes)
cellcount(lc::LevelCells, i::Int) = lc.starts[i + 1] - lc.starts[i]

"Binary search for a level code; 0 if unoccupied."
@inline function cell_index(lc::LevelCells, code::UInt64)
    i = searchsortedfirst(lc.codes, code)
    return (i <= length(lc.codes) && lc.codes[i] == code) ? i : 0
end

"Build LevelCells for levels 0..ell from leaf Morton codes (sorted with perm).
Returns (levels::Vector{LevelCells} indexed [level+1], order)."
function build_level_cells(g::CrossGrid, pts::AbstractVector{<:SVector{3}}, ell::Int)
    n = length(pts)
    leaf = Vector{UInt64}(undef, n)
    max_exc = 0.0
    for i in 1:n
        c, exc = level_coords(g, pts[i], ell)
        max_exc = max(max_exc, exc)
        leaf[i] = morton_encode(UInt64(c[1]), UInt64(c[2]), UInt64(c[3]))
    end
    order = sortperm(leaf)
    sorted = leaf[order]
    levels = Vector{LevelCells}(undef, ell + 1)
    for L in ell:-1:0
        shift = 3 * (ell - L)
        codes = UInt64[]; starts = Int[]
        prev = ~UInt64(0)
        for k in 1:n
            c = sorted[k] >> shift
            if c != prev
                push!(codes, c); push!(starts, k); prev = c
            end
        end
        push!(starts, n + 1)
        levels[L + 1] = LevelCells(codes, starts)
    end
    return levels, order, max_exc
end

"Per-phase push-offset views + near set from the shipped tables (uniform q)."
struct UniformQTables
    q::Int
    near_offsets::Vector{SVector{3,Int}}
    push_offsets::Vector{SVector{3,Int}}
    by_phase::NTuple{8,Vector{Int}}      # indices into push_offsets
end

function UniformQTables(q::Integer)
    t = RigidHierarchicalTables(q)
    by_phase = ntuple(p -> [Int(k) for k in
        t.phase_index[t.phase_starts[p]:t.phase_starts[p+1]-1]], 8)
    # cross-check: table membership == (child far, source-phase parent near)
    for phase in 0:7
        u = SVector(phase & 1, (phase >> 1) & 1, (phase >> 2) & 1)
        want = Set{SVector{3,Int}}()
        ext = 2 * isqrt(Int(q)) + 1
        for z in -ext:ext, y in -ext:ext, x in -ext:ext
            o = SVector(x, y, z)
            !_rigid_near(o, Int(q)) && _rigid_near(fld.(o .+ u, 2), Int(q)) &&
                push!(want, o)
        end
        got = Set(t.push_offsets[k] for k in by_phase[phase + 1])
        got == want || error("push-table mismatch at q=$q phase=$phase")
    end
    return UniformQTables(Int(q), t.near_offsets, t.push_offsets, by_phase)
end

"""
    sweep_config(tq, src_levels, tgt_levels, ell_x; materialize=false)

Count the two-occupancy uniform-q lists. Returns a NamedTuple of statistics;
with `materialize=true` also returns the route/near cell-pair lists
(src_cell_i, tgt_cell_i, level) for the coverage_counts subset check.
"""
function sweep_config(tq::UniformQTables, src_levels::Vector{LevelCells},
        tgt_levels::Vector{LevelCells}, ell_x::Int; materialize::Bool=false)
    q = tq.q
    routes_per_level = zeros(Int, ell_x + 1)     # [level+1]
    m2l_interactions = Int128(0)
    class_census = zeros(Int, length(tq.push_offsets))
    m2l_list = NTuple{3,Int}[]                    # (si, ti, level)
    for L in 2:ell_x
        G = 1 << L
        slc = src_levels[L + 1]; tlc = tgt_levels[L + 1]
        for si in 1:ncells(slc)
            code = slc.codes[si]
            cx, cy, cz = morton_decode(code)
            phase = _rigid_phase_index(cx, cy, cz)
            ns = cellcount(slc, si)
            for k in tq.by_phase[phase]
                o = tq.push_offsets[k]
                tx = Int(cx) + o[1]; ty = Int(cy) + o[2]; tz = Int(cz) + o[3]
                (0 <= tx < G && 0 <= ty < G && 0 <= tz < G) || continue
                ti = cell_index(tlc, morton_encode(UInt64(tx), UInt64(ty), UInt64(tz)))
                ti == 0 && continue
                routes_per_level[L + 1] += 1
                class_census[k] += 1
                m2l_interactions += Int128(ns) * cellcount(tlc, ti)
                materialize && push!(m2l_list, (si, ti, L))
            end
        end
    end
    # near field at the leaf level
    G = 1 << ell_x
    slc = src_levels[ell_x + 1]; tlc = tgt_levels[ell_x + 1]
    near_pairs = 0
    near_interactions = Int128(0)
    near_list = NTuple{3,Int}[]
    for si in 1:ncells(slc)
        cx, cy, cz = morton_decode(slc.codes[si])
        ns = cellcount(slc, si)
        for o in tq.near_offsets
            tx = Int(cx) + o[1]; ty = Int(cy) + o[2]; tz = Int(cz) + o[3]
            (0 <= tx < G && 0 <= ty < G && 0 <= tz < G) || continue
            ti = cell_index(tlc, morton_encode(UInt64(tx), UInt64(ty), UInt64(tz)))
            ti == 0 && continue
            near_pairs += 1
            near_interactions += Int128(ns) * cellcount(tlc, ti)
            materialize && push!(near_list, (si, ti, ell_x))
        end
    end
    nclasses_used = count(>(0), class_census)
    # distinct (level, offset) classes actually emitted
    lo_classes = 0
    for L in 2:ell_x
        lo_classes += _emitted_offsets_at_level(tq, src_levels, tgt_levels, L)
    end
    stats = (; routes_per_level, n_m2l = sum(routes_per_level),
        m2l_interactions, near_pairs, near_interactions,
        class_census, nclasses_used, lo_classes,
        src_cells_per_level = [ncells(src_levels[L + 1]) for L in 0:ell_x],
        tgt_cells_per_level = [ncells(tgt_levels[L + 1]) for L in 0:ell_x])
    return materialize ? (stats, m2l_list, near_list) : stats
end

"Count distinct offset classes emitted at one level (for the device
per-(level,offset) operator-table size)."
function _emitted_offsets_at_level(tq, src_levels, tgt_levels, L)
    G = 1 << L
    slc = src_levels[L + 1]; tlc = tgt_levels[L + 1]
    used = falses(length(tq.push_offsets))
    for si in 1:ncells(slc)
        cx, cy, cz = morton_decode(slc.codes[si])
        phase = _rigid_phase_index(cx, cy, cz)
        for k in tq.by_phase[phase]
            used[k] && continue
            o = tq.push_offsets[k]
            tx = Int(cx) + o[1]; ty = Int(cy) + o[2]; tz = Int(cz) + o[3]
            (0 <= tx < G && 0 <= ty < G && 0 <= tz < G) || continue
            cell_index(tlc, morton_encode(UInt64(tx), UInt64(ty), UInt64(tz))) != 0 &&
                (used[k] = true)
        end
    end
    return count(used)
end

"""
    brute_coverage(tq, g, panels, particles, ell_x, sample_js)

Independent exact-once check (no lists): for each sampled particle j, for
EVERY panel i, count covering assignments directly from the rule —
[|o_leaf|^2 <= q] + #{L in 2..ell_x : |o_L|^2 > q and |o_{L-1}|^2 <= q} —
and require exactly 1. Returns number of bad (pair-coverage != 1) pairs.
"""
function brute_coverage(tq::UniformQTables, g::CrossGrid,
        panels::AbstractVector{<:SVector{3}}, particles::AbstractVector{<:SVector{3}},
        ell_x::Int, sample_js::AbstractVector{<:Integer})
    q = tq.q
    # precompute panel coords at all levels
    np = length(panels)
    pc = Array{Int32}(undef, 3, np, ell_x + 1)
    for i in 1:np
        c, _ = level_coords(g, panels[i], ell_x)
        for L in ell_x:-1:0, a in 1:3
            pc[a, i, L + 1] = Int32(c[a] >> (ell_x - L))
        end
    end
    bad = 0
    for j in sample_js
        tcl, _ = level_coords(g, particles[j], ell_x)
        tcoords = [SVector{3,Int}(tcl[1] >> (ell_x - L), tcl[2] >> (ell_x - L),
                                  tcl[3] >> (ell_x - L)) for L in 0:ell_x]
        for i in 1:np
            cov = 0
            near_prev = true   # level-1 (and 0) offsets always near for q >= 3
            for L in 2:ell_x
                t = tcoords[L + 1]
                dx = t[1] - Int(pc[1, i, L + 1]); dy = t[2] - Int(pc[2, i, L + 1])
                dz = t[3] - Int(pc[3, i, L + 1])
                near_here = dx * dx + dy * dy + dz * dz <= q
                (!near_here && near_prev) && (cov += 1)
                near_prev = near_here
            end
            near_prev && (cov += 1)   # leaf-level near field
            cov == 1 || (bad += 1)
        end
    end
    return bad
end

"""
    coverage_counts_subset(...) — the prototype's full coverage matrix
(validate.jl:24-40) driven by the MATERIALIZED stencil lists on a subset
geometry: cover[i,j] over sorted body orders must be all-ones.
Returns (miss, dup).
"""
function coverage_counts_subset(src_levels, tgt_levels, m2l_list, near_list,
        ns::Int, nt::Int, ell_x::Int)
    cover = zeros(Int32, ns, nt)
    for (list, ) in ((m2l_list,), (near_list,))
        for (si, ti, L) in list
            slc = src_levels[L + 1]; tlc = tgt_levels[L + 1]
            for jj in tlc.starts[ti]:tlc.starts[ti + 1] - 1
                for ii in slc.starts[si]:slc.starts[si + 1] - 1
                    cover[ii, jj] += Int32(1)
                end
            end
        end
    end
    return count(==(Int32(0)), cover), count(>(Int32(1)), cover)
end

end # module
