# Correctness gate for stage 2 of the in-place device grid rebuild:
# `ka_radix_occupancy_changed!` and `ka_radix_cell_centers!`
# (ext/FastMultipoleKAExt.jl), the KA port of the occupancy-epoch check and the
# cell-center block of `_cuda_update_radix_grid_in_place!`
# (src/translate_batched_cuda.jl:6591), directly after the leaf-cell
# compression gated by ka_grid_keys_cells_correctness.jl.
#
# Oracle: the cell-center loop of `_refresh_radix_grid!` (src/tree_batched.jl),
# which is what `_radix_grid` runs on the CPU, plus `morton_decode` for the
# integer cell coords (the host grid does not store them; `ctx.cell_coords` is
# device-side only). Cell keys come from the stage-1 KA path, so the two stages
# are gated on the same data the driver will hand between them.
#
# The epoch check is exercised in both directions and at its edges: an
# unchanged key set must report false, and a single perturbed key must report
# true whether it sits first, last or in the middle -- a scan that misses the
# boundary lanes would pass an interior-only test.
include("ka_backend.jl")
using FastMultipole, Random, StaticArrays, Test

const FM = FastMultipole
const ext = Base.get_extension(FastMultipole, :FastMultipoleKAExt)

println("Starting KA grid epoch/centers correctness test on $DEV_NAME...")
if !dev_functional()
    println("$DEV_NAME not functional; skipping")
    exit(0)
end

_z(T, dims...) = (a = KernelAbstractions.allocate(DEV_BACKEND, T, dims...);
    fill!(a, zero(T)); a)

function run_case(TF, seed, n, ell, cluster)
    rng = MersenneTwister(seed)
    h0 = TF(1)
    x_min = SVector{3,TF}(-1, -1, -1)
    centers_c = [SVector{3,TF}(rand(rng, TF, 3) .* TF(1.8) .- TF(0.9)) for _ in 1:cluster]
    positions = Matrix{TF}(undef, 3, n)
    for i in 1:n
        c = centers_c[rand(rng, 1:cluster)]
        positions[:, i] .= clamp.(c .+ TF(0.02) .* randn(rng, TF, 3), TF(-1), TF(1))
    end

    # ---- occupied leaf cells, host side ----
    ref_keys = Vector{UInt64}(undef, n)
    for i in 1:n
        coord = FM.radix_cell_coord(x_min, h0, ell, SVector{3}(positions[:, i]))
        ref_keys[i] = FM.morton_key(coord, ell)
    end
    ref_perm = FM._host_radix_sort_permutation(ref_keys)
    ref_cell_keys, _ = FM._compress_radix_cells(ref_keys, ref_perm)
    n_cells_ref = length(ref_cell_keys)

    # host reference centers/coords, straight out of `_refresh_radix_grid!`
    delta = (2 * h0) / (1 << ell)
    ref_centers = Matrix{TF}(undef, 3, n_cells_ref)
    ref_coords = Matrix{Int}(undef, 3, n_cells_ref)
    for cell in 1:n_cells_ref
        coord = FM.morton_decode(ref_cell_keys[cell], ell)
        for d in 1:3
            ref_coords[d, cell] = coord[d]
            ref_centers[d, cell] = x_min[d] + delta * (TF(coord[d]) + TF(0.5))
        end
    end

    # ---- device: cell keys produced by the stage-1 KA path ----
    cap = n + 5
    max_cells = n_cells_ref + 3
    d_pos = devarray(positions)
    d_keys = _z(UInt64, cap); d_perm = _z(Int, cap); d_sorted = _z(UInt64, cap)
    d_invperm = _z(Int, cap); d_flags = _z(Int, cap); d_prefix = _z(Int, cap)
    d_cell_keys = _z(UInt64, max_cells)
    d_cell_ranges = _z(Int, 2, max_cells)
    kv = view(d_keys, 1:n)
    ext.ka_radix_keys_checked!(kv, _z(Int32, 1), zeros(Int32, 1), d_pos, x_min,
        SVector{3,TF}(2, 2, 2), h0, ell)
    sv = view(d_sorted, 1:n)
    ext.ka_radix_sort_bodies!(view(d_perm, 1:n), sv, view(d_invperm, 1:n), kv)
    n_cells = ext.ka_radix_compress_cells!(d_cell_keys, d_cell_ranges, sv,
        view(d_flags, 1:n), view(d_prefix, 1:n), zeros(Int, 1))
    n_cells == n_cells_ref ||
        error("n_cells $n_cells != $n_cells_ref (seed=$seed ell=$ell)")

    # ---- cell centers ----
    d_centers = _z(TF, 3, max_cells)
    d_coords = _z(Int, 3, max_cells)
    ext.ka_radix_cell_centers!(d_centers, d_coords, view(d_cell_keys, 1:n_cells),
        x_min, h0, ell, n_cells)
    c = 1:n_cells
    Array(d_coords)[:, c] == ref_coords ||
        error("cell_coords mismatch (seed=$seed ell=$ell)")
    Array(d_centers)[:, c] == ref_centers ||
        error("cell_centers mismatch (seed=$seed ell=$ell)")

    # ---- epoch check, both directions ----
    flag = _z(Int32, 1); host_flag = zeros(Int32, 1)
    snap = _z(UInt64, max_cells)
    copyto!(snap, 1, d_cell_keys, 1, n_cells)
    ckv = view(d_cell_keys, 1:n_cells)
    ext.ka_radix_occupancy_changed!(flag, host_flag, ckv, snap, n_cells) &&
        error("identical key set reported changed (seed=$seed ell=$ell)")
    # perturb first, middle and last slot in turn: a scan that drops the
    # boundary lanes would still pass an interior-only test
    host_snap = Array(snap)[1:n_cells]
    for slot in unique((1, cld(n_cells, 2), n_cells))
        bad = copy(host_snap)
        bad[slot] = bad[slot] ⊻ UInt64(1) << 40
        copyto!(snap, 1, devarray(bad), 1, n_cells)
        ext.ka_radix_occupancy_changed!(flag, host_flag, ckv, snap, n_cells) ||
            error("perturbed slot $slot reported unchanged (seed=$seed ell=$ell)")
        copyto!(snap, 1, devarray(host_snap), 1, n_cells)
    end
    return n_cells
end

cases = (
    (1, 1000,  3, 1),      # every body in one cell
    (2, 1000,  3, 8),
    (3, 5000,  4, 40),
    (4, 20000, 5, 200),
    (5, 1,     3, 1),      # single body, single cell
    (6, 4096,  2, 64),     # ell=2: 64 cells, fully occupied
)

for (seed, n, ell, cluster) in cases
    nc = run_case(Float32, seed, n, ell, cluster)
    println("✓ seed=$seed n=$n ell=$ell clusters=$cluster: $nc cells, " *
            "coords/centers exact, epoch check exact in both directions")
end

println("\n✓✓✓ KA grid epoch/centers gate passed on $DEV_NAME ✓✓✓")
