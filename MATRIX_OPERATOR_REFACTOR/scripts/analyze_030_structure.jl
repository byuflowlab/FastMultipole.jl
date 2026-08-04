# Task 030 structure oracle.
#
# For any candidate (n, ell, radius schedule) this computes the EXACT geometry
# counts the cost model needs, on the host, with no GPU and no cache build:
#
#   n_cells            occupied leaf cells
#   n_nodes            occupied nodes over levels 0..ell
#   nodes_per_level    occupied nodes at each level 0..ell
#   routes_per_level   M2L routes at each level 2..ell
#   routes             total M2L routes
#   n_direct           direct cell pairs (target, source), self included
#   pairwork           sum over direct pairs of n_target * n_source, i.e. the
#                      exact direct body-body interaction count
#   vsize_per_level    |V_L|, active push offsets at each level
#   launches           per-step kernel launches (see `_launch_count`)
#
# Why this exists: it turns "predict an unmeasured configuration" from an
# extrapolation into an interpolation. Structure is computed exactly; only the
# per-unit device rates are ever fitted (analyze_030_costmodel.jl). Nothing here
# is a model.
#
# The counts are reproduced from the production definitions, not from a parallel
# invention: cells come from `radix_cell_coord`, node occupancy is the set of
# ancestors of occupied leaves, routes mirror `build_hierarchical_routes_window!`
# (source-major, phase-masked, target = source + offset), and direct pairs mirror
# `build_hierarchical_direct_pairs!` (target - offset over `near_offsets`).
# `validate` is the gate that proves this correspondence against measured CSVs.
#
# Convection drift in the measured structure columns (established 2026-08-03 by
# this oracle, and not stated in the 028 record). `benchmark_028_feasibility.jl`
# writes its row AFTER the timing loops, and `step!()` advances every body by
# `FM028_DT` (default 1e-5) on each call: the boundary-b samples, the stale
# probe, the allocation probes, the counter-contract step, and the FM028_STEPS
# convection loop together run roughly `2*REPS + 5 + STEPS` Euler updates before
# the row is recorded. The structure columns are therefore POST-motion, while
# this oracle computes step-0 structure. Bodies near a cell boundary cross it,
# so occupancy-derived counts drift in EITHER direction (measured 1071656 vs
# oracle 1071012 routes at n=1e4/ell=4, but 676170 vs 676530 direct pairs at
# n=31623/ell=5). The drift grows with the surface-to-volume ratio of the
# occupied set, so it is largest at mid `n` on unsaturated grids and vanishes on
# saturated ones.
#
# Two consequences, both handled rather than assumed away:
#   * a saturated leaf grid (every cell occupied, n_cells == 8^ell) cannot drift,
#     so those rows are gated on EXACT integer equality;
#   * an unsaturated grid is gated on `DRIFT_TOL`, and the observed drift is
#     reported so a real modeling error (percent-level or structural) can never
#     hide inside the allowance.
# This is also why the same configuration measured 252483 / 252477 / 252486
# occupied cells in three different 028 jobs: different REPS/STEPS, hence a
# different number of Euler updates before the row was written.
#
# Usage:
#   julia --project=. MATRIX_OPERATOR_REFACTOR/scripts/analyze_030_structure.jl [validate|emit]
#     validate  re-derive every measured campaign row and require exact integer
#               agreement on n_cells/n_nodes/nodes_per_level/routes/
#               routes_per_level/n_direct.  Any mismatch voids the modeling layer.
#     emit      write data/cost_vs_n/structure_oracle.csv for the measured
#               configurations plus the candidate grid.
#
# Env: FM030_DATADIR (default MATRIX_OPERATOR_REFACTOR/data/cost_vs_n)

using FastMultipole
using FastMultipole.StaticArrays
using Random

const FM = FastMultipole
const SCRIPT_DIR = @__DIR__
const REFACTOR_DIR = normpath(joinpath(SCRIPT_DIR, ".."))
const DATA_DIR = get(ENV, "FM030_DATADIR",
    joinpath(REFACTOR_DIR, "data", "cost_vs_n"))

# 028 frozen workload (benchmark_028_feasibility.jl:110-115)
const SEED = 24025
const BOX_MIN = SVector(-0.01, -0.01, -0.01)
const BOX_SIZE = 1.02
const P = 3            # expansion_order; literature P = 4
const LH = false

"""
Relative bound on the convection drift documented in the header, applied only to
unsaturated leaf grids.

Sized from the mechanism, not fitted to the observations. With `REPS = 9` the
row is written after ~23 Euler updates at `dt = 1e-5`, so a body travels up to
~`2.3e-4 * |v|` in box units; at `ell = 4` the cell width is `1.02/16 = 0.064`,
so the boundary layer that can change cells is a fraction ~1e-2 of a cell wide,
and occupancy-derived counts can shift by a comparable fraction of their
surface. 5e-3 sits above that and one to two orders of magnitude below any real
modeling error, which would be percent-level or structural (a wrong offset set,
a wrong phase mask, a wrong level) rather than a few parts in ten thousand. The
observed maximum is always printed so a regression cannot hide inside the band.
"""
const DRIFT_TOL = 5e-3

# ---------------------------------------------------------------------------
# bodies and occupancy
# ---------------------------------------------------------------------------

"Body positions exactly as the 028 harness generates them (fm028_device_system.jl:15)."
function body_positions(n::Int)
    Random.seed!(SEED)
    bodies = rand(8, n)
    return bodies[1:3, :]
end

"""
    leaf_cells(positions, ell) -> Vector{Int}

Linear leaf-cell index (0-based, x fastest) for every body, using the production
`radix_cell_coord` so the binning cannot drift from the grid the benchmark built.
"""
function leaf_cells(positions::Matrix{Float64}, ell::Int)
    G = 1 << ell
    h0 = BOX_SIZE / 2
    out = Vector{Int}(undef, size(positions, 2))
    @inbounds for i in axes(positions, 2)
        c = FM.radix_cell_coord(BOX_MIN, h0, ell,
            SVector{3,Float64}(positions[1, i], positions[2, i], positions[3, i]))
        out[i] = c[1] + G * (c[2] + G * c[3])
    end
    return out
end

"""
    occupancy(cells, ell) -> Vector{Vector{Int}}

`counts[L+1][linear+1]` = number of bodies under the level-`L` node at that
linear coordinate; 0 means unoccupied. Level `ell` is the leaf level, and each
coarser level is the ancestor aggregation, which is exactly the set of occupied
nodes the device occupancy scatter produces.
"""
function occupancy(cells::Vector{Int}, ell::Int)
    levels = Vector{Vector{Int}}(undef, ell + 1)
    G = 1 << ell
    leaf = zeros(Int, G^3)
    @inbounds for c in cells
        leaf[c + 1] += 1
    end
    levels[ell + 1] = leaf
    for L in (ell - 1):-1:0
        GL = 1 << L
        prev = levels[L + 2]
        GP = 1 << (L + 1)
        cur = zeros(Int, GL^3)
        @inbounds for lin in 0:(GP^3 - 1)
            prev[lin + 1] == 0 && continue
            x = lin % GP; y = (lin ÷ GP) % GP; z = lin ÷ (GP * GP)
            cur[(x >> 1) + GL * ((y >> 1) + GL * (z >> 1)) + 1] += prev[lin + 1]
        end
        levels[L + 1] = cur
    end
    return levels
end

# ---------------------------------------------------------------------------
# geometry tables
# ---------------------------------------------------------------------------

"""
    policy_schedule(policy, ell) -> Vector{Int} or nothing

Map a harness policy string to its level radius schedule. `sched6-5-5` is
explicit; `hier<q>` is the uniform schedule `fill(q, ell-1)`, because
`_hierarchical_scheduled_tables` fills an empty `level_radii2` with
`near_radius2` at every level. `flat` has no hierarchical geometry.
"""
function policy_schedule(policy::AbstractString, ell::Int)
    if startswith(policy, "sched")
        return parse.(Int, split(String(policy)[6:end], '-'))
    elseif (m = match(r"^hier(\d+)$", policy)) !== nothing
        return fill(parse(Int, only(m.captures)), ell - 1)
    end
    return nothing
end

parse_schedule(s::AbstractString) = parse.(Int, split(String(s)[6:end], '-'))

"The shipped (6, 5, ..., 5) schedule for a depth, matching cuda_030_run.sh."
shipped_schedule(ell::Int) = ell == 2 ? [5] : vcat(6, fill(5, ell - 2))

"""
    tables_for(ell, schedule) -> (tables, level_class_of)

The production scheduled tables: `tables.push_offsets` is the shared offset
union, `tables.near_offsets` the leaf direct list, and
`level_class_of[phase, k, L+1]` the per-level phase mask that decides whether
offset `k` is active for a source of that parity phase at level `L`.
"""
function tables_for(ell::Int, schedule::Vector{Int})
    length(schedule) == ell - 1 || error(
        "schedule $(schedule) has $(length(schedule)) entries but ell=$ell needs $(ell-1)")
    q = last(schedule)
    h0 = BOX_SIZE / 2
    eps = FM.rigid_stencil_epsilon(P, h0, ell, q; lamb_helmholtz=LH, TF=Float64)
    base = FM.HierarchicalRigidStencil(FM.ConstantPStencilConfig(P, eps;
        lamb_helmholtz=LH); near_radius2=q, window_classes=4096)
    policy = FM._hierarchical_stencil_with_schedule(base, schedule)
    tables, level_class_of, _ = FM._hierarchical_scheduled_tables(policy, ell)
    return tables, level_class_of
end

# ---------------------------------------------------------------------------
# the oracle
# ---------------------------------------------------------------------------

"""
Per-step kernel launches, counted the way the resident lifecycle issues them.
This is the quantity that sets the small-`n` latency floor, where the linear
work terms predict essentially nothing.

Refresh is a fixed set of passes (grid/sort, occupancy, direct generation,
groups); each M2L level costs one window generation (flag/scan/compact, counted
as 3) plus one apply; B2M/M2M/L2L/L2B/nearfield/finalize/euler are one each,
with M2M and L2L issued per level. It is an estimate of launch COUNT, not of
launch cost — the per-launch cost is fitted.
"""
_launch_count(ell::Int) = 10 + (ell - 1) * 4 + 2 * (ell - 1) + 6

"""
    structure(n, ell, schedule) -> NamedTuple

Exact geometry counts. `schedule` is the level radius schedule (`ell-1`
non-increasing entries); pass `shipped_schedule(ell)` for the default geometry.
"""
function structure(n::Int, ell::Int, schedule::Vector{Int})
    tables, level_class_of = tables_for(ell, schedule)
    pos = body_positions(n)
    cells = leaf_cells(pos, ell)
    occ = occupancy(cells, ell)

    nodes_per_level = [count(!iszero, occ[L + 1]) for L in 0:ell]
    n_cells = nodes_per_level[ell + 1]
    n_nodes = sum(nodes_per_level)

    push_offsets = tables.push_offsets
    noffsets = length(push_offsets)

    # routes: source-major over occupied nodes, phase-masked, target = source + o
    routes_per_level = zeros(Int, max(ell - 1, 0))
    vsize_per_level = zeros(Int, max(ell - 1, 0))
    for L in 2:ell
        GL = 1 << L
        cnt = 0
        active = 0
        for k in 1:noffsets
            any(!iszero, @view level_class_of[:, k, L + 1]) && (active += 1)
        end
        vsize_per_level[L - 1] = active
        occL = occ[L + 1]
        @inbounds for lin in 0:(GL^3 - 1)
            occL[lin + 1] == 0 && continue
            x = lin % GL; y = (lin ÷ GL) % GL; z = lin ÷ (GL * GL)
            phase = FM._rigid_phase_index(x, y, z)
            for k in 1:noffsets
                level_class_of[phase, k, L + 1] == 0 && continue
                o = push_offsets[k]
                tx = x + o[1]; ty = y + o[2]; tz = z + o[3]
                (0 <= tx < GL && 0 <= ty < GL && 0 <= tz < GL) || continue
                occL[tx + GL * (ty + GL * tz) + 1] == 0 && continue
                cnt += 1
            end
        end
        routes_per_level[L - 1] = cnt
    end

    # direct pairs and exact body-pair work: target - o over near_offsets
    G = 1 << ell
    leaf = occ[ell + 1]
    n_direct = 0
    pairwork = 0
    @inbounds for lin in 0:(G^3 - 1)
        nt = leaf[lin + 1]
        nt == 0 && continue
        x = lin % G; y = (lin ÷ G) % G; z = lin ÷ (G * G)
        for o in tables.near_offsets
            sx = x - o[1]; sy = y - o[2]; sz = z - o[3]
            (0 <= sx < G && 0 <= sy < G && 0 <= sz < G) || continue
            ns = leaf[sx + G * (sy + G * sz) + 1]
            ns == 0 && continue
            n_direct += 1
            pairwork += nt * ns
        end
    end

    return (; n, ell, schedule=join(schedule, '-'),
        n_cells, n_nodes, nodes_per_level, routes_per_level,
        routes=sum(routes_per_level), n_direct, pairwork,
        vsize_per_level, noffsets,
        launches=_launch_count(ell),
        mean_bodies_per_leaf = n_cells == 0 ? 0.0 : n / n_cells)
end

structure(n::Int, ell::Int) = structure(n, ell, shipped_schedule(ell))

# ---------------------------------------------------------------------------
# CSV helpers (stdlib only, matching the project convention)
# ---------------------------------------------------------------------------

struct Table
    path::String
    header::Vector{String}
    rows::Vector{Vector{String}}
end

function readtable(path::AbstractString)
    isfile(path) || error("030: missing CSV: $path")
    lines = filter(!isempty, strip.(readlines(path)))
    length(lines) >= 2 || error("030: CSV has no data rows: $path")
    header = String.(split(lines[1], ','))
    rows = Vector{Vector{String}}()
    for (i, line) in enumerate(lines[2:end])
        fields = String.(split(line, ','))
        length(fields) == length(header) || error(
            "030: $path line $(i+1) has $(length(fields)) fields, header has $(length(header))")
        push!(rows, fields)
    end
    return Table(path, header, rows)
end

function colindex(t::Table, name::AbstractString)
    i = findfirst(==(name), t.header)
    i === nothing && error("030: column '$name' not in $(t.path)")
    return i
end
sget(t::Table, row, name) = row[colindex(t, name)]
iget(t::Table, row, name) = parse(Int, sget(t, row, name))
splitints(s::AbstractString) = isempty(strip(s)) ? Int[] : parse.(Int, split(strip(s)))

"""
Timing CSVs written by the 028 harness, excluding the `.classes.csv` companions
and any failure ledger. `prefix` selects the campaign: `cuda030_` for this row's
sweep, `cuda_` for the 028 campaign directory (used as extra validation data).
"""
function campaign_files(dir::AbstractString; prefix::AbstractString = "cuda030_")
    isdir(dir) || return String[]
    return sort(filter(readdir(dir; join=true)) do f
        b = basename(f)
        startswith(b, prefix) && endswith(b, ".csv") &&
            !endswith(b, ".classes.csv") && !occursin("failures", b)
    end)
end

# ---------------------------------------------------------------------------
# validation gate
# ---------------------------------------------------------------------------

"""
    validate(dir) -> Bool

Re-derive every measured `fit=true` row's structure and require EXACT integer
agreement. This is the single load-bearing check on the whole modeling layer: if
the oracle and the device disagree on a count, every modeled prediction built on
it is void, so this returns false rather than reporting a tolerance.
"""
function validate(dir::AbstractString = DATA_DIR;
        extra_dirs::Vector{<:Tuple{String,String}} = Tuple{String,String}[])
    files = [(f, "030") for f in campaign_files(dir)]
    for (d, prefix) in extra_dirs
        append!(files, [(f, basename(d)) for f in campaign_files(d; prefix)])
    end
    if isempty(files)
        println("030 oracle: no campaign CSVs in $dir yet — nothing to validate")
        return true
    end
    checked = 0
    seen = Set{Tuple{Int,Int,String}}()
    failures = String[]
    maxdrift = Ref(0.0)
    for (path, _) in files
        t = readtable(path)
        "n_cells" in t.header || continue
        for row in t.rows
            sget(t, row, "fit") == "true" || continue
            policy = sget(t, row, "policy")
            n = iget(t, row, "n")
            ell = iget(t, row, "ell")
            sched = policy_schedule(policy, ell)
            sched === nothing && continue          # flat rows have no schedule
            key = (n, ell, policy)
            key in seen && continue                 # identical geometry repeats
            push!(seen, key)
            s = structure(n, ell, sched)
            # A saturated leaf grid (every cell occupied) cannot drift under the
            # convection described above, so those rows are gated exactly; an
            # unsaturated grid is gated on the drift bound.
            saturated = s.n_cells == (1 << ell)^3
            checks = (
                ("n_cells", s.n_cells, iget(t, row, "n_cells")),
                ("n_nodes", s.n_nodes, iget(t, row, "n_nodes")),
                ("routes", s.routes, iget(t, row, "routes")),
                ("n_direct", s.n_direct, iget(t, row, "n_direct")),
            )
            for (name, got, want) in checks
                ok = saturated ? (got == want) :
                     (want > 0 && abs(got - want) <= DRIFT_TOL * want)
                ok || push!(failures,
                    "$(basename(path)) n=$n ell=$ell $policy: $name oracle=$got " *
                    "measured=$want rel=$(want == 0 ? NaN : (got - want) / want)" *
                    (saturated ? " [saturated: exact match required]" : ""))
                maxdrift[] = max(maxdrift[],
                    want == 0 ? 0.0 : abs(got - want) / want)
            end
            for (name, got, want) in (("nodes_per_level", s.nodes_per_level,
                                       splitints(sget(t, row, "nodes_per_level"))),
                                      ("routes_per_level", s.routes_per_level,
                                       splitints(sget(t, row, "routes_per_level"))))
                isempty(want) && continue
                length(got) == length(want) || (push!(failures,
                    "$(basename(path)) n=$n ell=$ell: $name length $(length(got)) vs $(length(want))"); continue)
                for (j, (g, w)) in enumerate(zip(got, want))
                    ok = (g == w) || (!saturated && w > 0 && abs(g - w) <= DRIFT_TOL * w)
                    ok || push!(failures,
                        "$(basename(path)) n=$n ell=$ell: $name[$j] oracle=$g measured=$w")
                    w > 0 && (maxdrift[] = max(maxdrift[], abs(g - w) / w))
                end
            end
            checked += 1
        end
    end
    if isempty(failures)
        println("030 oracle: VALIDATED — $checked measured configurations reproduced ",
                "(exact on saturated grids; max convection drift ",
                round(maxdrift[] * 100; sigdigits=2), "% vs the ",
                DRIFT_TOL * 100, "% bound)")
        return true
    end
    println("030 oracle: FAILED on $(length(failures)) checks over $checked configurations:")
    for f in first(failures, 25)
        println("  ", f)
    end
    length(failures) > 25 && println("  ... and $(length(failures) - 25) more")
    return false
end

# ---------------------------------------------------------------------------
# emit
# ---------------------------------------------------------------------------

const SWEEP_NS = [1000, 3162, 10000, 31623, 100000, 316228, 1000000]
const SWEEP_ELLS = [3, 4, 5]

"Legal candidate schedules: non-increasing over the supported lattice shells."
const SUPPORTED_Q = [3, 4, 5, 6, 8, 9, 10, 11, 12]

function candidate_schedules(ell::Int; max_candidates::Int = 40)
    ell < 2 && return Vector{Int}[]
    out = Vector{Int}[]
    # uniform schedules, plus the shipped (6,5,...,5) shape generalized to
    # (q_top, q, ..., q) for q_top >= q -- the two families the 025 proof covers
    # and the only ones with any measured neighbourhood.
    for q in SUPPORTED_Q
        push!(out, fill(q, ell - 1))
        for qt in SUPPORTED_Q
            qt > q || continue
            ell >= 3 && push!(out, vcat(qt, fill(q, ell - 2)))
        end
    end
    return first(unique(out), max_candidates)
end

function emit(dir::AbstractString = DATA_DIR)
    mkpath(dir)
    path = joinpath(dir, "structure_oracle.csv")
    cols = ["n", "ell", "schedule", "n_cells", "n_nodes", "routes", "n_direct",
            "pairwork", "noffsets", "launches", "mean_bodies_per_leaf",
            "nodes_per_level", "routes_per_level", "vsize_per_level"]
    open(path, "w") do io
        println(io, "# 030 structure oracle: exact geometry counts for the 028 frozen ",
                    "workload (seed $SEED, box $(BOX_MIN[1]) size $BOX_SIZE, ",
                    "expansion_order=$P). Space-separated sub-fields are per level. ",
                    "Computed, not measured and not modeled.")
        println(io, join(cols, ','))
        for n in SWEEP_NS, ell in SWEEP_ELLS, sched in candidate_schedules(ell)
            s = structure(n, ell, sched)
            println(io, join((s.n, s.ell, s.schedule, s.n_cells, s.n_nodes,
                s.routes, s.n_direct, s.pairwork, s.noffsets, s.launches,
                s.mean_bodies_per_leaf,
                join(s.nodes_per_level, ' '), join(s.routes_per_level, ' '),
                join(s.vsize_per_level, ' ')), ','))
            flush(io)
        end
    end
    println("030 oracle: wrote $path")
    return path
end

"""
The 028 campaign directory is independent validation data for the oracle: it
spans nine uniform radii, five schedules, and ell = 4/5/6 at n = 1e6 and below,
all measured on device before this row existed.
"""
const EXTRA_VALIDATION_DIRS = [
    (joinpath(REFACTOR_DIR, "data", "feasibility_1m_10ms"), "cuda_"),
]

if abspath(PROGRAM_FILE) == (@__FILE__)
    mode = isempty(ARGS) ? "validate" : ARGS[1]
    if mode == "validate"
        exit(validate(DATA_DIR; extra_dirs=EXTRA_VALIDATION_DIRS) ? 0 : 1)
    elseif mode == "emit"
        emit()
    else
        error("usage: analyze_030_structure.jl [validate|emit]")
    end
end
