using Printf
using Random

const DATA_DIR = joinpath(@__DIR__, "..", "data", "radix_sort_clustering")
const SUMMARY_PATH = joinpath(DATA_DIR, "verification_summary.md")

struct CellRecord
    key::UInt64
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

function cluster_points(points::Vector{NTuple{3,Float64}}, level::Int;
        root_center::NTuple{3,Float64} = (0.0, 0.0, 0.0),
        root_half_width::Float64 = 1.0)
    coords = [quantize_point(p, root_center, root_half_width, level) for p in points]
    keys = [morton_key(c, level) for c in coords]
    perm = sortperm(eachindex(points), by = i -> (keys[i], i), alg = MergeSort)

    invperm = similar(perm)
    for (sorted_index, original_index) in pairs(perm)
        invperm[original_index] = sorted_index
    end

    cells = CellRecord[]
    if !isempty(perm)
        start = 1
        while start <= length(perm)
            key = keys[perm[start]]
            stop = start
            while stop < length(perm) && keys[perm[stop + 1]] == key
                stop += 1
            end
            push!(cells, CellRecord(key, coords[perm[start]], start:stop))
            start = stop + 1
        end
    end

    return (; coords, keys, perm, invperm, cells, root_center, root_half_width, level)
end

offset_class(target_coord::NTuple{3,Int}, source_coord::NTuple{3,Int}) =
    ntuple(axis -> target_coord[axis] - source_coord[axis], 3)

function assert_condition(condition::Bool, message::String)
    condition || error(message)
end

function verify_key_construction()
    level = 3
    samples = Dict(
        (0, 0, 0) => UInt64(0),
        (1, 0, 0) => UInt64(1),
        (0, 1, 0) => UInt64(2),
        (0, 0, 1) => UInt64(4),
        (3, 5, 6) => UInt64(427),
        (7, 7, 7) => UInt64(511),
    )
    for (coord, expected) in samples
        assert_condition(morton_key(coord, level) == expected,
            "Morton key mismatch for $(coord)")
    end
    return true
end

function verify_case(name::String, points::Vector{NTuple{3,Float64}}, level::Int;
        root_center::NTuple{3,Float64} = (0.0, 0.0, 0.0),
        root_half_width::Float64 = 1.0)
    result = cluster_points(points, level; root_center, root_half_width)
    sorted_keys = result.keys[result.perm]

    keys_sorted = issorted(sorted_keys)
    stable_ties = true
    for cell in result.cells
        originals = result.perm[cell.range]
        stable_ties &= originals == sort(originals)
    end

    compressed_ranges = true
    for cell in result.cells
        range_keys = result.keys[result.perm[cell.range]]
        compressed_ranges &= all(==(cell.key), range_keys)
        before_ok = first(cell.range) == 1 || result.keys[result.perm[first(cell.range) - 1]] != cell.key
        after_ok = last(cell.range) == length(result.perm) ||
            result.keys[result.perm[last(cell.range) + 1]] != cell.key
        compressed_ranges &= before_ok && after_ok
    end

    inverse_roundtrip = result.perm[result.invperm] == collect(eachindex(points))

    Delta = cell_width(root_half_width, level)
    w = cell_half_width(root_half_width, level)
    rho = cell_radius(root_half_width, level)
    geometry_ok = isapprox(2w, Delta; atol = 0.0, rtol = 0.0) &&
        isapprox(rho, w * sqrt(3.0); atol = 0.0, rtol = 0.0)

    offset_ok = true
    for target in result.cells, source in result.cells
        d = offset_class(target.coord, source.coord)
        ct = cell_center(target.coord, root_center, root_half_width, level)
        cs = cell_center(source.coord, root_center, root_half_width, level)
        physical = ntuple(axis -> ct[axis] - cs[axis], 3)
        expected = ntuple(axis -> Delta * d[axis], 3)
        offset_ok &= all(axis -> isapprox(physical[axis], expected[axis]; atol = 4eps(Float64), rtol = 0.0), 1:3)
    end

    passed = keys_sorted && stable_ties && compressed_ranges && inverse_roundtrip &&
        geometry_ok && offset_ok

    return (;
        name,
        n_points = length(points),
        level,
        occupied_cells = length(result.cells),
        keys_sorted,
        stable_ties,
        compressed_ranges,
        inverse_roundtrip,
        geometry_ok,
        offset_ok,
        passed,
    )
end

function grid_aligned_points(level::Int)
    root_center = (0.0, 0.0, 0.0)
    root_half_width = 1.0
    coords = [(0, 0, 0), (1, 2, 3), (2, 1, 0), (3, 3, 3)]
    return [cell_center(c, root_center, root_half_width, level) for c in coords]
end

function boundary_tie_points()
    return [
        (-1.0, -1.0, -1.0),
        (1.0, 1.0, 1.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (nextfloat(0.0), 0.0, 0.0),
        (1.0, -1.0, 1.0),
    ]
end

function clustered_points()
    return [
        (-0.74, -0.73, -0.72),
        (-0.739, -0.731, -0.721),
        (-0.738, -0.732, -0.722),
        (0.61, 0.62, 0.63),
        (0.611, 0.621, 0.631),
        (0.0, 0.0, 0.0),
    ]
end

function sparse_points()
    return [
        (-0.99, -0.99, -0.99),
        (-0.99, 0.99, 0.99),
        (0.99, -0.99, 0.99),
        (0.99, 0.99, -0.99),
    ]
end

function random_points()
    rng = MersenneTwister(8675309)
    return [Tuple(rand(rng, 3) .* 2 .- 1) for _ in 1:64]
end

function verify_quantization_boundaries()
    level = 2
    root_center = (0.0, 0.0, 0.0)
    root_half_width = 1.0
    G = grid_resolution(level)
    lower = (-1.0, -1.0, -1.0)
    upper = (1.0, 1.0, 1.0)
    middle = (0.0, 0.0, 0.0)

    assert_condition(quantize_point(lower, root_center, root_half_width, level) == (0, 0, 0),
        "lower root boundary did not quantize to first cell")
    assert_condition(quantize_point(upper, root_center, root_half_width, level) == (G - 1, G - 1, G - 1),
        "upper root boundary did not clamp to last cell")
    assert_condition(quantize_point(middle, root_center, root_half_width, level) == (2, 2, 2),
        "interior half-open boundary did not quantize to upper adjacent cell")
    return true
end

function write_summary(results)
    mkpath(DATA_DIR)
    passed = all(r.passed for r in results)
    open(SUMMARY_PATH, "w") do io
        println(io, "# Radix-Sort Clustering Verification Summary")
        println(io)
        println(io, "- Command: `julia --project=. MATRIX_OPERATOR_REFACTOR/scripts/radix_sort_clustering_verify.jl`")
        println(io, "- Status: `$(passed ? "PASS" : "FAIL")`")
        println(io, "- Key construction: `PASS`")
        println(io, "- Boundary quantization: `PASS`")
        println(io, "- Root domain: center `(0, 0, 0)`, half-width `1`")
        println(io)
        println(io, "## Cases")
        println(io)
        println(io, "| Case | Points | Level | Occupied cells | Keys sorted | Stable ties | Ranges compressed | Inverse round trip | Geometry | Offsets | Status |")
        println(io, "| --- | ---: | ---: | ---: | --- | --- | --- | --- | --- | --- | --- |")
        for r in results
            @printf(io, "| %s | %d | %d | %d | `%s` | `%s` | `%s` | `%s` | `%s` | `%s` | `%s` |\n",
                r.name, r.n_points, r.level, r.occupied_cells, r.keys_sorted,
                r.stable_ties, r.compressed_ranges, r.inverse_roundtrip,
                r.geometry_ok, r.offset_ok, r.passed ? "PASS" : "FAIL")
        end
    end
    return (; passed)
end

function main()
    verify_key_construction()
    verify_quantization_boundaries()

    results = [
        verify_case("grid-aligned", grid_aligned_points(2), 2),
        verify_case("boundary/tie", boundary_tie_points(), 2),
        verify_case("clustered", clustered_points(), 4),
        verify_case("sparse occupancy", sparse_points(), 5),
        verify_case("fixed-seed random", random_points(), 4),
    ]

    summary = write_summary(results)
    println("radix_sort_clustering_verify: $(summary.passed ? "PASS" : "FAIL")")
    println("summary: $(SUMMARY_PATH)")
    summary.passed || exit(1)
end

main()
