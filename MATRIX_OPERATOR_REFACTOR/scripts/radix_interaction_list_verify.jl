using Printf
using Random

# Standalone verifier for the radix-path M2L interaction-list construction
# (task 008g). No production FastMultipole imports. No expansion translations are
# performed: only cell/pair identification and coverage are checked.
#
# Geometry / clustering helpers are reproduced locally to match
# scripts/radix_sort_clustering_verify.jl (008f). The stencil predicate is
# reproduced locally to match theory/constant-p-error-stencil.md (008d).

const DATA_DIR = joinpath(@__DIR__, "..", "data", "radix_interaction_list")
const SUMMARY_PATH = joinpath(DATA_DIR, "verification_summary.md")

# ----------------------------------------------------------------------------
# 008f geometry / clustering helpers (reproduced)
# ----------------------------------------------------------------------------

struct CellRecord
    coord::NTuple{3,Int}
    range::UnitRange{Int}
end

grid_resolution(level::Int) = 1 << level
cell_width(root_half_width::Float64, level::Int) = 2.0 * root_half_width / grid_resolution(level)
cell_half_width(root_half_width::Float64, level::Int) = root_half_width / grid_resolution(level)
cell_radius(root_half_width::Float64, level::Int) = cell_half_width(root_half_width, level) * sqrt(3.0)

function morton_key(coord::NTuple{3,Int}, level::Int)
    level <= 21 || error("UInt64 Morton keys require level <= 21")
    x, y, z = coord
    key = UInt64(0)
    for bit in 0:(level - 1)
        key |= UInt64((x >> bit) & 1) << (3 * bit)
        key |= UInt64((y >> bit) & 1) << (3 * bit + 1)
        key |= UInt64((z >> bit) & 1) << (3 * bit + 2)
    end
    return key
end

function quantize_point(point::NTuple{3,Float64}, root_center::NTuple{3,Float64},
        root_half_width::Float64, level::Int)
    G = grid_resolution(level)
    Delta = cell_width(root_half_width, level)
    return ntuple(3) do axis
        lower = root_center[axis] - root_half_width
        u = (point[axis] - lower) / Delta
        clamp(floor(Int, u), 0, G - 1)
    end
end

function cell_center(coord::NTuple{3,Int}, root_center::NTuple{3,Float64},
        root_half_width::Float64, level::Int)
    Delta = cell_width(root_half_width, level)
    return ntuple(3) do axis
        lower = root_center[axis] - root_half_width
        lower + Delta * (coord[axis] + 0.5)
    end
end

offset_class(target_coord::NTuple{3,Int}, source_coord::NTuple{3,Int}) =
    ntuple(axis -> target_coord[axis] - source_coord[axis], 3)

# Cluster points into occupied cells. Returns the occupied-cell table and a
# coord -> occupied-cell-index lookup. The source and target grids are the same
# grid (shared root, depth, width, quantization), so a single offset stencil is
# well-defined.
function cluster_points(points::Vector{NTuple{3,Float64}}, level::Int;
        root_center::NTuple{3,Float64} = (0.0, 0.0, 0.0),
        root_half_width::Float64 = 1.0)
    coords = [quantize_point(p, root_center, root_half_width, level) for p in points]
    keys = [morton_key(c, level) for c in coords]
    perm = sortperm(eachindex(points), by = i -> (keys[i], i), alg = MergeSort)

    cells = CellRecord[]
    lookup = Dict{NTuple{3,Int},Int}()
    if !isempty(perm)
        start = 1
        while start <= length(perm)
            key = keys[perm[start]]
            stop = start
            while stop < length(perm) && keys[perm[stop + 1]] == key
                stop += 1
            end
            coord = coords[perm[start]]
            push!(cells, CellRecord(coord, start:stop))
            lookup[coord] = length(cells)
            start = stop + 1
        end
    end

    return (; coords, keys, perm, cells, lookup, root_center, root_half_width, level)
end

# ----------------------------------------------------------------------------
# 008d conservative constant-P stencil (reproduced)
# ----------------------------------------------------------------------------

separation_ratio(d::NTuple{3,Int}) = 2.0 * sqrt(d[1]^2 + d[2]^2 + d[3]^2) / sqrt(3.0)

# Scalar bound B(P, d, A) = 2A / (rho (c - 2)) * (1/(c-1))^(P+1), valid for c > 2.
function scalar_bound(P::Int, d::NTuple{3,Int}, A::Float64, rho::Float64)
    c = separation_ratio(d)
    c > 2.0 || return Inf
    return 2.0 * A / (rho * (c - 2.0)) * (1.0 / (c - 1.0))^(P + 1)
end

# Lamb-Helmholtz combined bound:
# B_LH = B(P,d,A_phi) + (1 + 2R) * B(P,d,A_chi), R = 2 w norm(d).
function lh_bound(P::Int, d::NTuple{3,Int}, A_phi::Float64, A_chi::Float64,
        rho::Float64, w::Float64)
    c = separation_ratio(d)
    c > 2.0 || return Inf
    R = 2.0 * w * sqrt(d[1]^2 + d[2]^2 + d[3]^2)
    return scalar_bound(P, d, A_phi, rho) + (1.0 + 2.0 * R) * scalar_bound(P, d, A_chi, rho)
end

# Build the accepted-offset set over the finite offset domain of a depth-`level`
# grid (offsets span -(G-1):(G-1) per axis). d = target - source, d != 0.
function accepted_offsets(P::Int, eps::Float64, level::Int, root_half_width::Float64;
        lamb_helmholtz::Bool = false, A::Float64 = 1.0,
        A_phi::Float64 = 1.0, A_chi::Float64 = 1.0)
    G = grid_resolution(level)
    rho = cell_radius(root_half_width, level)
    w = cell_half_width(root_half_width, level)
    accepted = Set{NTuple{3,Int}}()
    for dx in -(G - 1):(G - 1), dy in -(G - 1):(G - 1), dz in -(G - 1):(G - 1)
        d = (dx, dy, dz)
        d == (0, 0, 0) && continue
        bound = lamb_helmholtz ? lh_bound(P, d, A_phi, A_chi, rho, w) :
            scalar_bound(P, d, A, rho)
        if isfinite(bound) && bound <= eps
            push!(accepted, d)
        end
    end
    return accepted
end

# ----------------------------------------------------------------------------
# 008g interaction-list construction
# ----------------------------------------------------------------------------

# Sweep construction: for each occupied target and each accepted offset d, the
# source coordinate is t - d (008d/008f convention d = target - source). If that
# coord is occupied, enqueue (target, source) into the batch for d.
function build_m2l_batches(cluster, accepted::Set{NTuple{3,Int}})
    G = grid_resolution(cluster.level)
    batches = Dict{NTuple{3,Int},Vector{Tuple{Int,Int}}}()
    for (t_idx, target) in pairs(cluster.cells)
        for d in accepted
            s = ntuple(axis -> target.coord[axis] - d[axis], 3)
            all(c -> 0 <= c <= G - 1, s) || continue
            s_idx = get(cluster.lookup, s, 0)
            s_idx == 0 && continue
            push!(get!(batches, d, Tuple{Int,Int}[]), (t_idx, s_idx))
        end
    end
    return batches
end

# ----------------------------------------------------------------------------
# Verification of one case
# ----------------------------------------------------------------------------

function verify_case(name::String, points::Vector{NTuple{3,Float64}}, level::Int,
        P::Int, eps::Float64; lamb_helmholtz::Bool = false,
        root_center::NTuple{3,Float64} = (0.0, 0.0, 0.0),
        root_half_width::Float64 = 1.0)
    cluster = cluster_points(points, level; root_center, root_half_width)
    accepted = accepted_offsets(P, eps, level, root_half_width; lamb_helmholtz)
    batches = build_m2l_batches(cluster, accepted)

    ncells = length(cluster.cells)
    nbodies = length(points)

    # --- (1) Cell-pair partition over all ordered occupied pairs ---
    far_pairs = Set{Tuple{Int,Int}}()
    near_pairs = Set{Tuple{Int,Int}}()
    self_pairs = Set{Tuple{Int,Int}}()
    for ti in 1:ncells, si in 1:ncells
        d = offset_class(cluster.cells[ti].coord, cluster.cells[si].coord)
        if d == (0, 0, 0)
            push!(self_pairs, (ti, si))
        elseif d in accepted
            push!(far_pairs, (ti, si))
        else
            push!(near_pairs, (ti, si))
        end
    end
    total_pairs = ncells * ncells
    disjoint = isempty(intersect(far_pairs, near_pairs)) &&
        isempty(intersect(far_pairs, self_pairs)) &&
        isempty(intersect(near_pairs, self_pairs))
    union_complete = length(far_pairs) + length(near_pairs) + length(self_pairs) == total_pairs
    cell_partition_ok = disjoint && union_complete

    # --- Stencil/offset agreement: sweep batches == classified far set ---
    swept_far = Set{Tuple{Int,Int}}()
    batch_offsets_ok = true
    for (d, pairs_d) in batches
        for (ti, si) in pairs_d
            push!(swept_far, (ti, si))
            # every enqueued pair actually has offset d and d is accepted
            actual = offset_class(cluster.cells[ti].coord, cluster.cells[si].coord)
            batch_offsets_ok &= (actual == d) && (d in accepted)
        end
    end
    stencil_agreement_ok = (swept_far == far_pairs) && batch_offsets_ok

    # --- (2) Body-pair coverage: every ordered (i,j) covered exactly once ---
    counts = zeros(Int, nbodies, nbodies)
    function mark(pair_set)
        for (ti, si) in pair_set
            for sp in cluster.perm[cluster.cells[ti].range]      # target bodies
                for ss in cluster.perm[cluster.cells[si].range]  # source bodies
                    counts[sp, ss] += 1
                end
            end
        end
    end
    mark(far_pairs); mark(near_pairs); mark(self_pairs)
    body_coverage_ok = all(==(1), counts)

    # --- Grid-sharing invariant (single shared grid here, trivially true) ---
    Delta = cell_width(root_half_width, level)
    w = cell_half_width(root_half_width, level)
    rho = cell_radius(root_half_width, level)
    grid_sharing_ok = isapprox(2w, Delta; atol = 0.0) && isapprox(rho, w * sqrt(3.0); atol = 0.0)

    passed = cell_partition_ok && stencil_agreement_ok && body_coverage_ok && grid_sharing_ok

    return (;
        name, lamb_helmholtz, n_points = nbodies, level, P, eps,
        occupied_cells = ncells,
        accepted_offsets = length(accepted),
        m2l_batches = length(batches),
        far = length(far_pairs), near = length(near_pairs), self = length(self_pairs),
        cell_partition_ok, stencil_agreement_ok, body_coverage_ok, grid_sharing_ok,
        passed,
    )
end

# ----------------------------------------------------------------------------
# Test point sets (reuse 008f-style occupancy patterns)
# ----------------------------------------------------------------------------

function grid_aligned_points(level::Int)
    root_center = (0.0, 0.0, 0.0)
    root_half_width = 1.0
    coords = [(0, 0, 0), (0, 0, 4), (4, 0, 0), (7, 7, 7), (4, 4, 4), (0, 7, 0)]
    return [cell_center(c, root_center, root_half_width, level) for c in coords]
end

function clustered_points()
    return [
        (-0.95, -0.95, -0.95), (-0.94, -0.95, -0.95), (-0.95, -0.94, -0.95),
        (0.92, 0.92, 0.92), (0.93, 0.92, 0.92),
        (-0.95, 0.93, -0.95), (0.92, -0.95, 0.93),
    ]
end

function sparse_points()
    return [(-0.97, -0.97, -0.97), (-0.97, 0.97, 0.97),
            (0.97, -0.97, 0.97), (0.97, 0.97, -0.97)]
end

function random_points()
    rng = MersenneTwister(8675309)
    return [Tuple(rand(rng, 3) .* 2 .- 1) for _ in 1:64]
end

# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------

function write_summary(results)
    mkpath(DATA_DIR)
    passed = all(r.passed for r in results)
    open(SUMMARY_PATH, "w") do io
        println(io, "# Radix-Path M2L Interaction-List Verification Summary")
        println(io)
        println(io, "- Command: `julia --project=. MATRIX_OPERATOR_REFACTOR/scripts/radix_interaction_list_verify.jl`")
        println(io, "- Status: `$(passed ? "PASS" : "FAIL")`")
        println(io, "- No expansion translations performed; cell/pair identification and coverage only.")
        println(io, "- Root domain: center `(0, 0, 0)`, half-width `1`; source and target share one grid.")
        println(io)
        println(io, "## Cases")
        println(io)
        println(io, "| Case | LH | Points | Level | P | eps | Cells | Accepted offsets | M2L batches | Far | Near | Self | Cell partition | Stencil agreement | Body coverage ==1 | Grid sharing | Status |")
        println(io, "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- |")
        for r in results
            @printf(io, "| %s | `%s` | %d | %d | %d | %.3g | %d | %d | %d | %d | %d | %d | `%s` | `%s` | `%s` | `%s` | `%s` |\n",
                r.name, r.lamb_helmholtz, r.n_points, r.level, r.P, r.eps,
                r.occupied_cells, r.accepted_offsets, r.m2l_batches,
                r.far, r.near, r.self,
                r.cell_partition_ok, r.stencil_agreement_ok, r.body_coverage_ok,
                r.grid_sharing_ok, r.passed ? "PASS" : "FAIL")
        end
        println(io)
        println(io, "Coverage target: for each case the far/near/self sets partition every ordered")
        println(io, "occupied cell pair exactly once, the swept M2L batches equal the classified far")
        println(io, "set, and every ordered body pair is covered exactly once (count == 1).")
    end
    return (; passed)
end

function main()
    P = 4
    results = [
        verify_case("grid-aligned", grid_aligned_points(3), 3, P, 1.0),
        verify_case("clustered", clustered_points(), 4, P, 1.0),
        verify_case("sparse occupancy", sparse_points(), 3, P, 1.0),
        verify_case("fixed-seed random", random_points(), 3, P, 1.0),
        verify_case("fixed-seed random (LH)", random_points(), 3, P, 1.0; lamb_helmholtz = true),
        verify_case("grid-aligned (LH)", grid_aligned_points(3), 3, P, 1.0; lamb_helmholtz = true),
    ]

    summary = write_summary(results)
    println("radix_interaction_list_verify: $(summary.passed ? "PASS" : "FAIL")")
    println("summary: $(SUMMARY_PATH)")
    summary.passed || exit(1)
end

main()
