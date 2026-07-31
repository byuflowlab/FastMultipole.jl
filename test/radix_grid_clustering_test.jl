using FastMultipole
using FastMultipole.StaticArrays
using Test

@testset "radix grid clustering" begin
    @test FastMultipole.morton_key(0, 0, 0, 2) == UInt64(0)
    @test FastMultipole.morton_key(1, 0, 0, 2) == UInt64(1)
    @test FastMultipole.morton_key(0, 1, 0, 2) == UInt64(2)
    @test FastMultipole.morton_key(0, 0, 1, 2) == UInt64(4)
    @test FastMultipole.morton_key(1, 1, 1, 2) == UInt64(7)
    @test FastMultipole.morton_key(2, 0, 0, 2) == UInt64(8)
    @test FastMultipole.morton_key(3, 2, 1, 2) == UInt64(29)

    for ell in 0:5
        G = 1 << ell
        for coord in (SVector(0, 0, 0), SVector(G - 1, 0, 0), SVector(0, G - 1, 0), SVector(0, 0, G - 1), SVector(G - 1, G - 1, G - 1))
            key = FastMultipole.morton_key(coord, ell)
            @test FastMultipole.morton_decode(key, ell) == coord
        end
    end

    boundary_positions = [
        0.0 1.0 0.25 0.75
        0.0 0.0 0.0 0.0
        0.0 0.0 0.0 0.0
    ]
    boundary_grid = RadixGrid(boundary_positions, 2)
    @test FastMultipole.radix_resolution(boundary_grid) == 4
    @test FastMultipole.radix_cell_coord(boundary_grid, SVector(0.0, 0.0, 0.0))[1] == 0
    @test FastMultipole.radix_cell_coord(boundary_grid, SVector(1.0, 0.0, 0.0))[1] == 3
    @test FastMultipole.radix_cell_coord(boundary_grid, SVector(0.25, 0.0, 0.0))[1] == 1

    positions = [
        0.10 0.90 0.10 0.60 0.90 0.60
        0.10 0.90 0.10 0.40 0.90 0.40
        0.10 0.90 0.10 0.40 0.90 0.40
    ]
    grid = RadixGrid(positions, 1)
    host_grid = radix_grid(positions, 1; sort=HostRadixSort())
    auto_grid = radix_grid(positions, 1; sort=AutoRadixSort(min_device_bodies=0))

    @test grid.perm == [1, 3, 4, 6, 2, 5]
    @test host_grid.perm == grid.perm
    @test auto_grid.perm == grid.perm
    @test host_grid.cell_keys == grid.cell_keys
    @test auto_grid.cell_ranges == grid.cell_ranges
    @test_throws ArgumentError radix_grid(positions, 1; sort=DeviceRadixSort())
    @test all(grid.invperm[grid.perm[s]] == s for s in eachindex(grid.perm))
    @test all(grid.perm[grid.invperm[i]] == i for i in 1:size(positions, 2))
    @test grid.cell_keys == sort(grid.cell_keys)
    @test grid.cell_ranges == [1 3 5; 2 2 2]
    @test vcat([collect(FastMultipole.radix_body_range(grid, i)) for i in 1:length(grid)]...) == collect(1:size(positions, 2))
    @test sort(grid.perm) == collect(1:size(positions, 2))
    @test length(unique(grid.perm)) == size(positions, 2)
    @test grid.body_system == fill(1, size(positions, 2))
    @test grid.body_index == collect(1:size(positions, 2))
    @test FastMultipole.radix_body_system(grid, 4) == 1
    @test FastMultipole.radix_body_index(grid, 4) == 4
    @test FastMultipole.radix_body_ref(grid, 4) == SVector(1, 4)

    @test FastMultipole.radix_cell_half_width(grid) == grid.h0 / 2
    @test FastMultipole.radix_cell_width(grid) == grid.h0
    @test FastMultipole.radix_cell_radius(grid) ≈ FastMultipole.radix_cell_half_width(grid) * sqrt(3)

    c000 = FastMultipole.radix_cell_center(grid, SVector(0, 0, 0))
    c111 = FastMultipole.radix_cell_center(grid, SVector(1, 1, 1))
    @test c111 - c000 ≈ SVector(FastMultipole.radix_cell_width(grid), FastMultipole.radix_cell_width(grid), FastMultipole.radix_cell_width(grid))

    i000 = FastMultipole.radix_cell_index(grid, SVector(0, 0, 0))
    i111 = FastMultipole.radix_cell_index(grid, SVector(1, 1, 1))
    @test i000 == 1
    @test i111 == 3
    @test FastMultipole.radix_cell_index(grid, SVector(0, 1, 0)) == 0
    @test FastMultipole.radix_cell_index(grid, SVector(2, 0, 0)) == 0
    @test FastMultipole.radix_cell_index(grid, SVector(-1, 0, 0)) == 0
    @test FastMultipole.radix_cell_index(grid, SVector(0, 2, 0)) == 0
    @test FastMultipole.radix_cell_index(grid, SVector(0, -1, 0)) == 0
    @test FastMultipole.radix_cell_index(grid, SVector(0, 0, 2)) == 0
    @test FastMultipole.radix_cell_index(grid, SVector(0, 0, -1)) == 0
    @test collect(FastMultipole.radix_body_indices(grid, i000)) == [1, 3]
    @test FastMultipole.radix_offset(grid, i111, i000) == SVector(1, 1, 1)
    @test FastMultipole.radix_displacement(grid, i111, i000) ≈ SVector(FastMultipole.radix_cell_width(grid), FastMultipole.radix_cell_width(grid), FastMultipole.radix_cell_width(grid))

    empty_grid = RadixGrid(zeros(3, 0), 1)
    @test isempty(empty_grid.perm)
    @test isempty(empty_grid.invperm)
    @test isempty(empty_grid.cell_keys)
    @test size(empty_grid.cell_ranges) == (2, 0)
    @test isempty(empty_grid.body_system)
    @test isempty(empty_grid.body_index)

    degenerate_positions = [
        2.0 2.0 2.0
        3.0 3.0 3.0
        4.0 4.0 4.0
    ]
    degenerate_grid = RadixGrid(degenerate_positions, 3; h0_fallback=2.0)
    @test degenerate_grid.h0 == 2.0
    @test degenerate_grid.x_min == SVector(0.0, 1.0, 2.0)
    @test length(degenerate_grid) == 1
    @test degenerate_grid.perm == [1, 2, 3]
    @test FastMultipole.radix_cell_coord(degenerate_grid, SVector(2.0, 3.0, 4.0)) == SVector(4, 4, 4)

    duplicate_positions = [
        0.1 0.1 0.1 0.1 0.9 0.9 0.9
        0.1 0.1 0.1 0.1 0.9 0.9 0.9
        0.1 0.1 0.1 0.1 0.9 0.9 0.9
    ]
    duplicate_grid = RadixGrid(duplicate_positions, 1; sort=HostRadixSort())
    @test duplicate_grid.perm == collect(1:7)
    @test duplicate_grid.cell_ranges == [1 5; 4 3]

    tuple_a = [
        0.10 0.60 0.90
        0.10 0.40 0.90
        0.10 0.40 0.90
    ]
    tuple_b = Float32[
        0.10 0.60 1.20
        0.10 0.40 1.20
        0.10 0.40 1.20
    ]
    tuple_grid = RadixGrid((tuple_a, tuple_b), 1)
    n_tuple = FastMultipole.get_n_bodies((tuple_a, tuple_b))
    @test tuple_grid isa RadixGrid{Float64}
    @test length(tuple_grid.perm) == n_tuple
    @test tuple_grid.x_min ≈ SVector(0.10, 0.10, 0.10)
    @test tuple_grid.x_min + SVector(2 * tuple_grid.h0, 2 * tuple_grid.h0, 2 * tuple_grid.h0) ≈ SVector(1.20, 1.20, 1.20) atol=1e-6
    @test tuple_grid.h0 ≈ 0.55 atol=1e-6
    @test sort(tuple_grid.perm) == collect(1:n_tuple)
    @test length(unique(tuple_grid.perm)) == n_tuple
    @test all(tuple_grid.invperm[tuple_grid.perm[s]] == s for s in eachindex(tuple_grid.perm))
    @test all(tuple_grid.perm[tuple_grid.invperm[i]] == i for i in 1:n_tuple)
    @test tuple_grid.cell_keys == sort(tuple_grid.cell_keys)
    @test vcat([collect(FastMultipole.radix_body_indices(tuple_grid, i)) for i in 1:length(tuple_grid)]...) == tuple_grid.perm
    @test sort(vcat([collect(FastMultipole.radix_body_indices(tuple_grid, i)) for i in 1:length(tuple_grid)]...)) == collect(1:n_tuple)
    @test [FastMultipole.radix_body_ref(tuple_grid, i) for i in 1:n_tuple] ==
        [SVector(1, 1), SVector(1, 2), SVector(1, 3), SVector(2, 1), SVector(2, 2), SVector(2, 3)]

    mixed_cell = FastMultipole.radix_cell_index(tuple_grid, FastMultipole.radix_cell_coord(tuple_grid, SVector(0.10, 0.10, 0.10)))
    @test collect(FastMultipole.radix_body_indices(tuple_grid, mixed_cell)) == [1, 2, 4, 5]
    @test [FastMultipole.radix_body_ref(tuple_grid, i) for i in FastMultipole.radix_body_indices(tuple_grid, mixed_cell)] ==
        [SVector(1, 1), SVector(1, 2), SVector(2, 1), SVector(2, 2)]
end
