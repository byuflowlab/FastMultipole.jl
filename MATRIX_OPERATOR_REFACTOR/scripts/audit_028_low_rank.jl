# Task 028 Stage 8: Float64 singular-spectrum audit of the selected hierarchical
# dense M2L classes. A production factor path is permitted only when the
# route-weighted rank has >=20% arithmetic headroom against the r=8 break-even.

using FastMultipole
using FastMultipole.StaticArrays
using LinearAlgebra
using Dates

const FM = FastMultipole
const ELL = parse(Int, get(ENV, "FM028_ELL", "5"))
const P = parse(Int, get(ENV, "FM028_P", "3"))
const SCHEDULE = parse.(Int, split(get(ENV, "FM028_SCHEDULE", "6-6-6-6"), '-'))
const GATE = parse(Float64, get(ENV, "FM028_ACCURACY_GATE", "1.19e-3"))
const OPERATOR_BUDGET = 0.2 * GATE
const OUT = get(ENV, "FM028_LOWRANK_OUT", joinpath(@__DIR__, "..", "data",
    "feasibility_1m_10ms", "stage8_singular_spectra_" *
    Dates.format(now(), "yyyymmdd-HHMMSS") * ".csv"))

length(SCHEDULE) == ELL - 1 || error("schedule must have ell-1 entries")
base = HierarchicalRigidStencil(P, rigid_stencil_epsilon(P, 0.51, ELL,
    last(SCHEDULE)); near_radius2=last(SCHEDULE), window_classes=typemax(Int))
policy = FM._hierarchical_stencil_with_schedule(base, SCHEDULE)
tables, level_class_of, qs = FM._hierarchical_scheduled_tables(policy, ELL)
class_level, class_offset, effective = FM._hierarchical_class_metadata(tables, ELL)
basis = FM.OperatorBasisInfo(FM.CompressedComplexBasis(), P, Val(false))
invariant = FM.OperatorInvariantCache(Float64, basis)
plan = FM.ResidentM2LDensePlan(Float64, basis, effective, 1.02 / (1 << ELL),
    1, 1, 0, DenseTranslationM2L(apply_chunk=1, build_chunk=8), invariant;
    hierarchical_noffsets=length(tables.push_offsets))

function route_count(L, k)
    G = 1 << L
    o = tables.push_offsets[k]
    n = 0
    parity_extent(d, u) = count(x -> (x & 1) == u && 0 <= x + d < G, 0:G-1)
    for phase in 0:7
        level_class_of[phase + 1, k, L + 1] == 0 && continue
        ux, uy, uz = phase & 1, (phase >> 1) & 1, (phase >> 2) & 1
        n += parity_extent(o[1], ux) * parity_extent(o[2], uy) *
            parity_extent(o[3], uz)
    end
    return n
end

rows = NamedTuple[]
weighted_rank = 0.0
total_routes = 0
noffsets = length(tables.push_offsets)
for L in 2:ELL, k in 1:noffsets
    routes = route_count(L, k)
    routes == 0 && continue
    cls = (L - 2) * noffsets + k
    A = Diagonal(plan.target_scale[:, cls]) * plan.operators[k] *
        Diagonal(plan.source_scale[:, cls])
    s = svdvals(A)
    denom = norm(s)
    rank = length(s)
    for r in 0:length(s)
        tail = r == length(s) ? 0.0 : norm(@view s[r+1:end]) / denom
        if tail <= OPERATOR_BUDGET
            rank = r
            break
        end
    end
    global weighted_rank += routes * rank
    global total_routes += routes
    push!(rows, (; level=L, q_parent=L == 2 ? qs[1] : qs[L - 2],
        q_child=qs[L - 1], offset=join(Tuple(tables.push_offsets[k]), ':'),
        orbit=join(FM._rigid_orbit_key(tables.push_offsets[k]), ':'), routes,
        admissible_rank=rank, sigma=join(s, ':'),
        relative_tail_at_rank7=norm(@view s[min(8, length(s)+1):end]) / denom))
end
weighted_rank /= total_routes
modeled_fraction = weighted_rank / 8
prototype_allowed = weighted_rank < 8 && modeled_fraction <= 0.8

mkpath(dirname(OUT))
open(OUT, "w") do io
    println(io, join(string.(keys(first(rows))), ','))
    for row in rows
        println(io, join(string.(values(row)), ','))
    end
end
println("schedule=", join(SCHEDULE, '-'), " routes=", total_routes,
    " weighted_rank=", weighted_rank, " modeled_fraction=", modeled_fraction,
    " prototype_allowed=", prototype_allowed,
    " operator_relative_budget=", OPERATOR_BUDGET)
println("wrote ", OUT)
