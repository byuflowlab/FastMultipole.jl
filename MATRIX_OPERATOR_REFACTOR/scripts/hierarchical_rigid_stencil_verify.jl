#!/usr/bin/env julia

# Task 025: deterministic enumeration and production-operator verification for the
# source-major hierarchical rigid M2L stencil.

using FastMultipole
using FastMultipole.StaticArrays
using LinearAlgebra
using Random
using Printf

const FM = FastMultipole
const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const OUTDIR = joinpath(ROOT, "MATRIX_OPERATOR_REFACTOR", "data",
    "hierarchical_rigid_stencil")
const QVALUES = (3, 4, 5, 6, 8, 9, 10, 11, 12)
const PHASES = [SVector{3,Int}(i, j, k) for k in 0:1 for j in 0:1 for i in 0:1]

norm2(o) = sum(abs2, o)
near(o, q) = norm2(o) <= q
phase(o) = SVector(mod(o[1], 2), mod(o[2], 2), mod(o[3], 2))
parent_offset(u, o) = SVector(fld(u[1] + o[1], 2),
    fld(u[2] + o[2], 2), fld(u[3] + o[3], 2))

function near_offsets(q)
    b = isqrt(q)
    return [SVector{3,Int}(i, j, k) for k in -b:b for j in -b:b for i in -b:b
        if i * i + j * j + k * k <= q]
end

function push_list(q, u)
    b = 2 * isqrt(q) + 1
    result = SVector{3,Int}[]
    for k in -b:b, j in -b:b, i in -b:b
        o = SVector(i, j, k)
        !near(o, q) && near(parent_offset(u, o), q) && push!(result, o)
    end
    sort!(result; by=o -> (o[3], o[2], o[1]))
    return result
end

function next_lattice_shell(q)
    shell = q + 1
    while true
        b = isqrt(shell)
        any(i*i + j*j + k*k == shell for k in -b:b, j in -b:b, i in -b:b) &&
            return shell
        shell += 1
    end
end

function pull_equivalent(q, u, o)
    v = phase(u + o)                       # target phase
    reverse_parent = parent_offset(v, -o) # source parent - target parent
    return !near(-o, q) && near(reverse_parent, q)
end

csvfield(x) = begin
    s = string(x)
    occursin(r"[\",\n]", s) ? "\"" * replace(s, "\"" => "\"\"") * "\"" : s
end

function writecsv(path, header, rows)
    open(path, "w") do io
        println(io, join(header, ','))
        for row in rows
            println(io, join(csvfield.(row), ','))
        end
    end
end

function stencil_rows()
    rows = Tuple[]
    for q in QVALUES
        N = near_offsets(q)
        lists = Dict(u => push_list(q, u) for u in PHASES)
        union_offsets = union((Set(v) for v in values(lists))...)
        far_q = next_lattice_shell(q)
        theta_lo = sqrt(3 / far_q)
        theta_hi = sqrt(3 / q)
        for ell in 2:7, (pid, u) in enumerate(PHASES)
            V = lists[u]
            push_ok = all(o -> pull_equivalent(q, u, o), V)
            reverse_ok = all(o -> begin
                v = phase(u + o)
                -o in lists[v]
            end, V)
            push!(rows, (q, ell, pid - 1, u[1], u[2], u[3], length(N),
                length(V), length(union_offsets), maximum(maximum(abs, o) for o in V),
                minimum(norm2, V), push_ok && reverse_ok,
                @sprintf("%.17g", theta_lo), @sprintf("%.17g", theta_hi),
                "sqrt(3/$far_q) < theta <= sqrt(3/$q)"))
        end
    end
    return rows
end

function level_rows()
    rows = Tuple[]
    for q in QVALUES
        # Level 1 is a 2x2x2 grid. Every in-box offset has norm2 <= 3,
        # so neither radius admits a separated pair.
        level1_separated = count(begin
            o = SVector(tx-sx, ty-sy, tz-sz)
            !near(o, q)
        end for sz in 0:1 for sy in 0:1 for sx in 0:1
            for tz in 0:1 for ty in 0:1 for tx in 0:1)
        @assert level1_separated == 0
        push!(rows, (q, "level_branch", 1, 64, level1_separated, true, 0,
            "exhaustive 8x8 ordered cells; all level-1 offsets are near"))

        G = 4
        checks = 0
        failures = 0
        for sz in 0:G-1, sy in 0:G-1, sx in 0:G-1,
                tz in 0:G-1, ty in 0:G-1, tx in 0:G-1
            S = SVector(sx, sy, sz)
            T = SVector(tx, ty, tz)
            o = T - S
            near(o, q) && continue
            checks += 1
            p_formula = parent_offset(phase(S), o)
            p_geometry = SVector(fld.(T, 2)) - SVector(fld.(S, 2))
            (p_formula == p_geometry && near(p_formula, q)) || (failures += 1)
        end
        @assert failures == 0
        push!(rows, (q, "level_branch_geometric", 2, checks, failures,
            failures == 0, 0,
            "all separated ordered level-2 cell pairs have near geometric parents"))

        if q == 12
            offsets = [SVector(i, j, k) for k in -(G-1):(G-1),
                j in -(G-1):(G-1), i in -(G-1):(G-1) if i*i+j*j+k*k > q]
            formula_checks = length(offsets) * length(PHASES)
            formula_failures = count(!near(parent_offset(u, o), q)
                for o in offsets for u in PHASES)
            @assert formula_checks == 1312 && formula_failures == 0
            push!(rows, (q, "level2_formula_range_preliminary", 2,
                formula_checks, formula_failures, true, 0,
                "164 separated bounded offsets x 8 phases; stronger q=12 audit"))
        end

        b = isqrt(q) + 2
        min_child = typemax(Int)
        checked_children = 0
        inequality_ok = true
        for pk in -b:b, pj in -b:b, pi in -b:b
            p = SVector(pi, pj, pk)
            near(p, q) && continue
            for uz in 0:1, uy in 0:1, ux in 0:1
                # o = 2p + target_phase - source_phase; enumerating both child
                # phases independently covers every geometric child offset.
                for vz in 0:1, vy in 0:1, vx in 0:1
                    o = 2p + SVector(vx-ux, vy-uy, vz-uz)
                    checked_children += 1
                    min_child = min(min_child, norm2(o))
                    for d in 1:3
                        abs(o[d]) >= 2abs(p[d]) - 1 || (inequality_ok = false)
                    end
                end
            end
        end
        @assert inequality_ok && min_child > q
        push!(rows, (q, "downward_monotonicity", 0, checked_children, 0, true,
            min_child, "|o_k| >= 2|p_k|-1 for every enumerated child"))
    end
    return rows
end

coord_key(c) = (c[1], c[2], c[3])

function occupied_nodes(leaves, depth, level)
    shift = depth - level
    nodes = Dict{NTuple{3,Int},Vector{Int}}()
    for (i, c) in enumerate(leaves)
        a = SVector(c[1] >> shift, c[2] >> shift, c[3] >> shift)
        push!(get!(Vector{Int}, nodes, coord_key(a)), i)
    end
    return nodes
end

function verify_coverage(name, leaves, depth, q)
    n = length(leaves)
    counts = zeros(UInt8, n, n) # target, source
    direct_pairs = 0
    route_pairs = 0

    # Generator A: direct leaf work plus source-major node V-lists expanded to
    # descendant leaf pairs. Bounds truncation happens through dictionary lookup.
    for s in 1:n, t in 1:n
        if near(leaves[t] - leaves[s], q)
            counts[t, s] += 1
            direct_pairs += 1
        end
    end
    for level in 2:depth
        nodes = occupied_nodes(leaves, depth, level)
        for (skey, sbodies) in nodes
            S = SVector(skey)
            for o in push_list(q, phase(S))
                tbodies = get(nodes, coord_key(S + o), nothing)
                tbodies === nothing && continue
                for s in sbodies, t in tbodies
                    counts[t, s] += 1
                    route_pairs += 1
                end
            end
        end
    end

    # Generator B: independent pairwise first-separated-ancestor partition.
    expected_direct = 0
    expected_route = 0
    first_levels = zeros(Int, depth + 1)
    wrong_kind = 0
    for s in 1:n, t in 1:n
        leaf_o = leaves[t] - leaves[s]
        if near(leaf_o, q)
            expected_direct += 1
            counts[t, s] == 1 || (wrong_kind += 1)
            continue
        end
        first = 0
        for level in 2:depth
            shift = depth - level
            os = SVector(leaves[s][1] >> shift, leaves[s][2] >> shift,
                leaves[s][3] >> shift)
            ot = SVector(leaves[t][1] >> shift, leaves[t][2] >> shift,
                leaves[t][3] >> shift)
            if !near(ot - os, q)
                first = level
                break
            end
        end
        first > 0 || error("$name q=$q: separated leaf pair lacks first ancestor")
        first_levels[first] += 1
        expected_route += 1
        counts[t, s] == 1 || (wrong_kind += 1)
    end
    duplicates = count(>(1), counts)
    missing = count(==(0), counts)
    @assert direct_pairs == expected_direct
    @assert route_pairs == expected_route
    @assert duplicates == 0 && missing == 0 && wrong_kind == 0
    return (q, name, depth, n, n * n, direct_pairs, route_pairs,
        join(first_levels[2:depth], ';'), missing, duplicates, wrong_kind, true)
end

function coverage_rows()
    cases = [
        ("dense_depth3", 3, [SVector(i,j,k) for k in 0:7 for j in 0:7 for i in 0:7]),
        ("dense_depth4", 4, [SVector(i,j,k) for k in 0:15 for j in 0:15 for i in 0:15]),
        ("boundary_truncated", 4, [SVector(i,j,k) for k in 0:5 for j in 9:15
            for i in 0:4 if (i + 2j + 3k) % 4 != 0]),
        ("sparse_fixed", 4, [SVector(i,j,k) for k in (0,2,7,15), j in (0,5,14),
            i in (0,3,9,15) if (i + j + k) % 3 != 1]),
    ]
    return [verify_coverage(name, leaves, depth, q)
        for q in QVALUES for (name, depth, leaves) in cases]
end

function degree_vector(binfo, lh)
    phi = reduce(vcat, [fill(n, 2n + 1) for n in 0:binfo.orders.P_phi])
    if lh
        chi = reduce(vcat, [fill(n, 2n + 1) for n in 0:binfo.orders.P_active])
        return phi, chi
    end
    return phi, Int[]
end

function scaling_rows()
    rows = Tuple[]
    P = 3
    offsets = [("axial_pos", SVector(0,0,2)), ("axial_neg", SVector(0,0,-2)),
        ("equatorial", SVector(2,1,0)), ("generic", SVector(2,-3,1))]
    for lh in (false, true)
        binfo = FM.OperatorBasisInfo(FM.CompressedComplexBasis(), P, Val(lh))
        phi_deg, chi_deg = degree_vector(binfo, lh)
        target_exp = lh ? vcat(phi_deg, chi_deg .+ 1) : phi_deg
        source_exp = lh ? vcat(phi_deg, chi_deg .- 1) : phi_deg
        for (label, o) in offsets, s in (0.25, 0.5, 2.0, 4.0)
            r, theta, phi = FM.cartesian_to_spherical(SVector{3,Float64}(o))
            K0 = FM.build_dense_m2l_operator(Float64, binfo, r, theta, phi, Val(lh))
            Ks = FM.build_dense_m2l_operator(Float64, binfo, s*r, theta, phi, Val(lh))
            Lt = Diagonal(s .^ (-target_exp))
            Ls = Diagonal(s .^ (-source_exp))
            predicted = s^(-1) * Lt * K0 * Ls
            rel = norm(Ks - predicted) / max(norm(Ks), eps())
            @assert rel <= 1e-13
            push!(rows, (lh ? "lamb_helmholtz" : "scalar", label, o[1], o[2],
                o[3], s, size(K0, 1), maximum(abs, Ks - predicted), rel, 1e-13,
                rel <= 1e-13, lh ? "target=(n_phi,n_chi+1);source=(n_phi,n_chi-1)" :
                "target=source=n"))
        end
    end
    return rows
end

# Exact occupied-cell counts for the 024b input generator. Calling rand(8,n)
# preserves the campaign's column-major random stream without depending on test/.
function campaign_occupancies(n, ell)
    Random.seed!(24025)
    bodies = rand(8, n)
    G = 1 << ell
    cells = Set{NTuple{3,Int}}()
    for j in 1:n
        c = ntuple(d -> clamp(floor(Int, (bodies[d,j] + 0.01) / 1.02 * G), 0, G-1), 3)
        push!(cells, c)
    end
    counts = zeros(Int, ell + 1)
    counts[ell + 1] = length(cells)
    current = cells
    for level in (ell - 1):-1:0
        current = Set((c[1] >> 1, c[2] >> 1, c[3] >> 1) for c in current)
        counts[level + 1] = length(current)
    end
    return counts
end

function expected_pairs(C, G, offsets; phased=false)
    M = G^3
    p2 = M == 1 ? 0.0 : C * (C - 1) / (M * (M - 1))
    p1 = C / M
    total = 0.0
    for item in offsets
        u, o = phased ? item : (nothing, item)
        ways = 1
        for d in 1:3
            if phased
                # source x has parity u and both x and x+o must be in the box.
                lo = max(0, -o[d]); hi = min(G-1, G-1-o[d])
                first = lo + mod(u[d] - lo, 2)
                ways *= first <= hi ? fld(hi-first, 2) + 1 : 0
            else
                ways *= G - abs(o[d])
            end
        end
        total += ways * (iszero(o) ? p1 : p2)
    end
    return total
end

function cost_rows()
    rows = Tuple[]
    for q in QVALUES
        N = near_offsets(q)
        V = [(u, o) for u in PHASES for o in push_list(q, u)]
        vphase = length(push_list(q, PHASES[1]))
        vunion = length(union((Set(push_list(q, u)) for u in PHASES)...))
        crossover = vphase * 8 / 7
        push!(rows, ("constant", q, 0, 0, "analytic", "not_applicable",
            length(N), vphase, vunion, crossover, NaN, NaN, NaN, "",
            "hierarchical linear-work crossover C=8|V|/7"))
    end

    # Fully occupied dense-lattice anchors are exact.
    q = 12
    for ell in 5:7
        G = 1 << ell
        flat_classes = (2G - 1)^3 - length(near_offsets(q))
        hier_classes = (ell - 1) * 1740
        flat_routes = G^6 - expected_pairs(G^3, G, near_offsets(q))
        level_routes = 0.0
        for level in 2:ell
            g = 1 << level
            phased = [(u,o) for u in PHASES for o in push_list(q,u)]
            level_routes += expected_pairs(g^3, g, phased; phased=true)
        end
        push!(rows, ("flat_vs_hierarchical", q, ell, G^3, "fully_occupied_dense_lattice",
            "exact", length(near_offsets(q)), 1253, 1740, 8*1253/7,
            flat_classes / hier_classes, flat_routes / level_routes, G^3,
            join((1 << (3level) for level in 0:ell), ';'),
            "occupancy-independent dense-lattice anchor"))
    end

    # The timing CSV has no occupancy/routes columns. Reconstruct exact campaign
    # leaf occupancies, then report a transparent uniform-without-replacement
    # route expectation conditioned on those occupancies.
    for (n, ell) in ((31623,5), (316228,6), (1000000,7))
        G = 1 << ell
        occupancies = campaign_occupancies(n, ell)
        C = occupancies[ell + 1]
        flat_offsets = [SVector(i,j,k) for k in -(G-1):(G-1),
            j in -(G-1):(G-1), i in -(G-1):(G-1) if i*i+j*j+k*k > 12]
        flat_routes = expected_pairs(C, G, flat_offsets)
        hier_routes = 0.0
        hier_classes = 0
        for level in 2:ell
            g = 1 << level
            Cl = occupancies[level + 1]
            phased = [(u,o) for u in PHASES for o in push_list(12,u)]
            hier_routes += expected_pairs(Cl, g, phased; phased=true)
            hier_classes += 1740
        end
        flat_classes = length(flat_offsets)
        push!(rows, ("campaign_audit", 12, ell, n,
            "measured_leaf_occupancy_reconstructed_seed24025",
            "routes_uniform_expectation_not_measured", 179, 1253, 1740, 8*1253/7,
            flat_classes / hier_classes, flat_routes / hier_routes, C,
            join(occupancies, ';'),
            "024b CSV lacks occupancy/routes; preliminary ratios are projections"))
    end
    return rows
end

function epsilon_rows()
    rows = Tuple[]
    P = 3
    h0 = 0.51
    bound(ell, m2) = begin
        c = 2sqrt(float(m2)) / sqrt(3.0)
        c > 2 || return Inf
        rho = h0 / 2^ell * sqrt(3.0)
        2 / (rho * (c - 2)) * (1 / (c - 1))^(P + 1)
    end
    for q in QVALUES, ell in 2:7
        shell = next_lattice_shell(q)
        lower = bound(ell, shell)
        upper = bound(ell, q)
        push!(rows, (q, "epsilon_interval", ell, shell, 0, true, 0,
            @sprintf("accepted iff epsilon >= %.17g and epsilon < %s", lower,
                isfinite(upper) ? @sprintf("%.17g", upper) : "Inf")))
    end
    # Compare the smallest epsilon that accepts the classic first-far shell
    # (norm2=4) with the smallest compatible q=12 epsilon (first-far shell 13).
    ratio = bound(4, 4) / bound(4, 13)
    @assert isapprox(ratio, 238.2; rtol=5e-4)
    push!(rows, (3, "classic_epsilon_price", 0, 4, 0, true, 0,
        @sprintf("%.17g times q=12 lower compatible epsilon endpoint", ratio)))
    return rows
end

function main()
    mkpath(OUTDIR)
    srows = stencil_rows()
    lrows = vcat(level_rows(), epsilon_rows())
    crows = coverage_rows()
    orows = scaling_rows()
    krows = cost_rows()

    @assert all(r -> r[7] == length(near_offsets(r[1])), srows)
    @assert all(r -> r[8] == length(push_list(r[1], PHASES[r[3] + 1])), srows)
    @assert all(r -> r[9] == length(RigidHierarchicalTables(r[1]).push_offsets), srows)

    writecsv(joinpath(OUTDIR, "stencil_counts.csv"),
        ("near_radius2","ell","phase_id","ux","uy","uz","near_count",
         "phase_v_count","union_count","max_norm_inf","min_norm2","push_pull_ok",
         "theta_lower_open","theta_upper_closed","theta_interval"), srows)
    writecsv(joinpath(OUTDIR, "level_monotonicity_checks.csv"),
        ("near_radius2","check","ell","cases","failures","pass","minimum_child_norm2",
         "detail"), lrows)
    writecsv(joinpath(OUTDIR, "coverage_cases.csv"),
        ("near_radius2","case","depth","occupied_leaves","ordered_pairs",
         "direct_pairs","m2l_pairs","first_separated_counts_levels_2_to_depth",
         "missing","duplicates","wrong_kind","pass"), crows)
    writecsv(joinpath(OUTDIR, "operator_scaling.csv"),
        ("mode","offset_case","ox","oy","oz","scale","operator_dof","max_abs_error",
         "relative_error","tolerance","pass","diagonal_exponents"), orows)
    writecsv(joinpath(OUTDIR, "cost_occupancy_comparisons.csv"),
        ("comparison","near_radius2","ell","n_or_dense_cells","occupancy_basis",
         "route_basis","near_count","phase_v_count","union_count","crossover_cells",
         "class_reduction","route_reduction","occupied_leaf_cells",
         "occupied_cells_levels_0_to_ell","note"), krows)

    open(joinpath(OUTDIR, "verification_summary.md"), "w") do io
        println(io, "# Hierarchical rigid stencil verification")
        println(io)
        println(io, "Generated deterministically by `hierarchical_rigid_stencil_verify.jl`.")
        println(io)
        println(io, "- supported shells: `$(join(QVALUES, ", "))`; near/V/union counts are recorded per shell")
        println(io, "- level-1/2 and downward monotonicity checks: PASS for every supported shell")
        println(io, "- exact-once ordered-pair coverage: $(length(crows)) cases PASS")
        println(io, "- production dense scalar and Lamb–Helmholtz scaling: $(length(orows)) cases PASS at rtol 1e-13")
        println(io, "- campaign audit: occupancy reconstructed exactly from seed 24025; route comparisons are labeled model estimates because 024b CSVs contain no route telemetry")
    end
    println("task 025 verification PASS; wrote $(OUTDIR)")
end

main()
