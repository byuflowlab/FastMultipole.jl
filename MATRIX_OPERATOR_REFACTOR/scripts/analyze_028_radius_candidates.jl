# Deterministic geometry/work audit for task-028 intermediate rigid radii.
# No FastMultipole production helper is used: this independently enumerates the
# task-025 near family, phase V-list, downward-monotonicity condition, and exact
# directed work on a fully occupied ell=5 grid.

const OUT = normpath(joinpath(@__DIR__, "..", "data", "feasibility_1m_10ms",
    "radius_candidates.csv"))
const SCHEDULE_OUT = normpath(joinpath(@__DIR__, "..", "data",
    "feasibility_1m_10ms", "radius_schedule_candidates.csv"))
const STAGE7_OUT = normpath(joinpath(@__DIR__, "..", "data",
    "feasibility_1m_10ms", "stage7_schedule_coverage.csv"))
const ELL = 5

norm2(o) = o[1]^2 + o[2]^2 + o[3]^2
near(o, q) = norm2(o) <= q

function geometry(q)
    r = isqrt(q)
    near_offsets = NTuple{3,Int}[]
    for z in -r:r, y in -r:r, x in -r:r
        o = (x, y, z)
        near(o, q) && push!(near_offsets, o)
    end

    extent = 2r + 1
    by_phase = [NTuple{3,Int}[] for _ in 1:8]
    union_offsets = Set{NTuple{3,Int}}()
    for phase in 0:7
        u = (phase & 1, (phase >> 1) & 1, (phase >> 2) & 1)
        for z in -extent:extent, y in -extent:extent, x in -extent:extent
            o = (x, y, z)
            near(o, q) && continue
            p = (fld(u[1] + x, 2), fld(u[2] + y, 2), fld(u[3] + z, 2))
            near(p, q) || continue
            push!(by_phase[phase + 1], o)
            push!(union_offsets, o)
        end
    end
    all(length(v) == length(first(by_phase)) for v in by_phase) ||
        error("phase cardinalities differ for q=$q")
    return near_offsets, by_phase, union_offsets
end

function minimum_child_norm2(q)
    b = isqrt(q) + 2
    result = typemax(Int)
    for pz in -b:b, py in -b:b, px in -b:b
        p = (px, py, pz)
        near(p, q) && continue
        for uz in 0:1, uy in 0:1, ux in 0:1,
                vz in 0:1, vy in 0:1, vx in 0:1
            o = (2px + vx - ux, 2py + vy - uy, 2pz + vz - uz)
            result = min(result, norm2(o))
        end
    end
    return result
end

# Number of x in 0:G-1 with phase x%2=u and translated coordinate in bounds.
parity_extent(G, o, u) = count(x -> (x & 1) == u && 0 <= x + o < G, 0:G-1)

function exact_work(q, ell)
    near_offsets, by_phase, union_offsets = geometry(q)
    G = 1 << ell
    direct = sum((G - abs(o[1])) * (G - abs(o[2])) * (G - abs(o[3]))
        for o in near_offsets)
    routes_per_level = Int[]
    for L in 2:ell
        g = 1 << L
        count_L = 0
        for phase in 0:7
            ux, uy, uz = phase & 1, (phase >> 1) & 1, (phase >> 2) & 1
            for o in by_phase[phase + 1]
                count_L += parity_extent(g, o[1], ux) *
                    parity_extent(g, o[2], uy) * parity_extent(g, o[3], uz)
            end
        end
        push!(routes_per_level, count_L)
    end
    shell_norms = sort!(unique(norm2(o) for o in Iterators.product(
        -G:G, -G:G, -G:G)))
    max_near = maximum(v for v in shell_norms if v <= q)
    min_far = minimum(v for v in shell_norms if v > q)
    min_child = minimum_child_norm2(q)
    min_child > q || error("downward monotonicity fails for q=$q")
    return (; q, near_offsets=length(near_offsets),
        phase_offsets=length(first(by_phase)), union_offsets=length(union_offsets),
        max_near_norm2=max_near, min_far_norm2=min_far,
        minimum_child_norm2=min_child, downward_monotone=true,
        direct_cell_pairs=direct, routes_total=sum(routes_per_level),
        routes_per_level=join(routes_per_level, ' '))
end

# One decreasing-radius transition: the parent level uses q_parent and its child
# level uses q_child.  q_child <= q_parent preserves exact-once coverage because
# the fixed-q downward-monotonicity result gives
# parent far at q_parent => child far at q_parent => child far at q_child.
function transition_work(q_parent, q_child, level)
    q_child <= q_parent || error("radius must not increase with depth")
    extent = 2isqrt(q_parent) + 1
    by_phase = [NTuple{3,Int}[] for _ in 1:8]
    union_offsets = Set{NTuple{3,Int}}()
    for phase in 0:7
        u = (phase & 1, (phase >> 1) & 1, (phase >> 2) & 1)
        for z in -extent:extent, y in -extent:extent, x in -extent:extent
            o = (x, y, z)
            near(o, q_child) && continue
            p = (fld(u[1] + x, 2), fld(u[2] + y, 2), fld(u[3] + z, 2))
            near(p, q_parent) || continue
            push!(by_phase[phase + 1], o)
            push!(union_offsets, o)
        end
    end
    all(length(v) == length(first(by_phase)) for v in by_phase) ||
        error("transition phase cardinalities differ for $q_parent->$q_child")
    g = 1 << level
    routes = 0
    for phase in 0:7
        ux, uy, uz = phase & 1, (phase >> 1) & 1, (phase >> 2) & 1
        for o in by_phase[phase + 1]
            routes += parity_extent(g, o[1], ux) *
                parity_extent(g, o[2], uy) * parity_extent(g, o[3], uz)
        end
    end
    return (; phase_offsets=length(first(by_phase)),
        union_offsets=length(union_offsets), routes)
end

function transition_geometry(q_parent, q_child)
    extent = 2isqrt(q_parent) + 1
    by_phase = [Set{NTuple{3,Int}}() for _ in 1:8]
    for phase in 0:7
        u = (phase & 1, (phase >> 1) & 1, (phase >> 2) & 1)
        for z in -extent:extent, y in -extent:extent, x in -extent:extent
            o = (x, y, z)
            near(o, q_child) && continue
            p = (fld(u[1] + x, 2), fld(u[2] + y, 2), fld(u[3] + z, 2))
            near(p, q_parent) && push!(by_phase[phase + 1], o)
        end
    end
    return by_phase
end

sub3(a, b) = (a[1] - b[1], a[2] - b[2], a[3] - b[3])
ancestor(c, shift) = (c[1] >> shift, c[2] >> shift, c[3] >> shift)
phaseof(c) = 1 + (c[1] & 1) + 2(c[2] & 1) + 4(c[3] & 1)

function sparse_exact_once(coords, qs)
    transitions = [transition_geometry(j == 1 ? qs[j] : qs[j-1], qs[j])
                   for j in eachindex(qs)]
    ell = length(qs) + 1
    for target in coords, source in coords
        hits = near(sub3(target, source), last(qs)) ? 1 : 0
        for L in 2:ell
            shift = ell - L
            s = ancestor(source, shift)
            t = ancestor(target, shift)
            o = sub3(t, s)
            o in transitions[L - 1][phaseof(s)] && (hits += 1)
        end
        hits == 1 || return false
    end
    return true
end

function dense_schedule_work(qs)
    ell = length(qs) + 1
    routes = Int[]
    for L in 2:ell
        qp = L == 2 ? qs[1] : qs[L - 2]
        push!(routes, transition_work(qp, qs[L - 1], L).routes)
    end
    G = 1 << ell
    near_offsets = first(geometry(last(qs)))
    direct = sum((G - abs(o[1])) * (G - abs(o[2])) * (G - abs(o[3]))
                 for o in near_offsets)
    represented = direct + sum(routes[L - 1] * (1 << (ell - L))^6 for L in 2:ell)
    return direct, routes, represented == G^6
end

rows = [exact_work(q, ELL) for q in 3:12]
mkpath(dirname(OUT))
open(OUT, "w") do io
    println(io, join(string.(keys(first(rows))), ','))
    for row in rows
        println(io, join(string.(values(row)), ','))
    end
end
println("wrote $OUT")
for row in rows
    println(row)
end

# A compact two-stage family for follow-up triage: q_coarse is used through
# level ell-1, then q_leaf at the leaf.  These are geometry/work projections,
# not accuracy claims or production-supported policies.
distinct_q = [3, 4, 5, 6, 8, 9, 10, 11, 12]
row_by_q = Dict(row.q => row for row in rows)
schedule_rows = NamedTuple[]
for q_coarse in distinct_q, q_leaf in distinct_q
    q_leaf <= q_coarse || continue
    coarse = row_by_q[q_coarse]
    leaf = row_by_q[q_leaf]
    uniform_routes = parse.(Int, split(coarse.routes_per_level))
    transition = transition_work(q_coarse, q_leaf, ELL)
    routes_per_level = [uniform_routes[1:end-1]; transition.routes]
    exact_once = minimum_child_norm2(q_coarse) > q_leaf
    exact_once || error("cross-level monotonicity fails for $q_coarse->$q_leaf")
    push!(schedule_rows, (; q_coarse, q_leaf,
        leaf_phase_offsets=transition.phase_offsets,
        leaf_union_offsets=transition.union_offsets,
        direct_cell_pairs=leaf.direct_cell_pairs,
        routes_total=sum(routes_per_level),
        routes_per_level=join(routes_per_level, ' '), exact_once))
end
open(SCHEDULE_OUT, "w") do io
    println(io, join(string.(keys(first(schedule_rows))), ','))
    for row in schedule_rows
        println(io, join(string.(values(row)), ','))
    end
end
println("wrote $SCHEDULE_OUT")

# Complete adjacent-shell frontier required by Stage 7. Dense coverage is
# counted analytically with each level route weighted by its descendant leaf
# pairs; sparse and boundary-truncated grids are brute-force ordered-pair audits.
frontier = ((5,5,5,5), (6,5,5,5), (6,6,5,5), (6,6,6,5), (6,6,6,6))
sparse = [(0,0,0), (3,1,0), (31,31,31), (16,25,7), (1,31,19)]
boundary = [(0,0,0), (1,0,0), (31,31,31), (30,31,31), (0,31,15)]
stage7_rows = NamedTuple[]
for qs in frontier
    direct, routes, dense_ok = dense_schedule_work(qs)
    sparse_ok = sparse_exact_once(sparse, qs)
    boundary_ok = sparse_exact_once(boundary, qs)
    dense_ok && sparse_ok && boundary_ok || error("schedule coverage failed for $qs")
    push!(stage7_rows, (; schedule=join(qs, '-'), direct_cell_pairs=direct,
        routes_total=sum(routes), routes_per_level=join(routes, ' '),
        dense_exact_once=dense_ok, sparse_exact_once=sparse_ok,
        boundary_exact_once=boundary_ok,
        downward_monotone=all(qs[i+1] <= qs[i] for i in 1:length(qs)-1)))
end
open(STAGE7_OUT, "w") do io
    println(io, join(string.(keys(first(stage7_rows))), ','))
    for row in stage7_rows
        println(io, join(string.(values(row)), ','))
    end
end
println("wrote $STAGE7_OUT")
