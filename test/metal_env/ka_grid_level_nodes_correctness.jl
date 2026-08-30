# Correctness gate for stage 3 of the in-place device grid rebuild:
# `ka_radix_level_nodes!` (ext/FastMultipoleKAExt.jl), the KA port of the
# per-level unique-node block of `_cuda_update_radix_grid_in_place!`
# (src/translate_batched_cuda.jl:6591) -- the first block behind the stage-2
# occupancy-epoch check.
#
# Oracle: `_refresh_radix_nodes!` (src/tree_batched.jl), the host node builder
# `_radix_grid` runs, called on a host-side `DeviceRadixGrid` over plain
# Vectors. It computes far more than stage 3 does (geometry, parents, child
# ranges -- stage 4); this gate compares only what stage 3 owns, `level_offsets`
# and the level-major `node_keys`, plus the derived `max_count` that sets the
# stage-4 launch geometry.
#
# Cell keys come from the stage-1 KA path, so the stages are gated on the same
# data the driver hands between them. Cases sweep `ell` and cluster count (how
# many levels collapse to a single node) and `first_level` (task 037 active-
# level trimming: the trimmed columns are never keyed, so `level_counts` reads
# garbage there and the host loop must zero it before it reaches an offset).
include("ka_backend.jl")
using FastMultipole, Random, StaticArrays, Test

const FM = FastMultipole
const ext = Base.get_extension(FastMultipole, :FastMultipoleKAExt)

println("Starting KA grid level-nodes correctness test on $DEV_NAME...")
if !dev_functional()
    println("$DEV_NAME not functional; skipping")
    exit(0)
end

_z(T, dims...) = (a = KernelAbstractions.allocate(DEV_BACKEND, T, dims...);
    fill!(a, zero(T)); a)

# host `DeviceRadixGrid` over plain Vectors, sized to hold the oracle's output
function host_grid(TF, x_min, h0, ell, cell_keys, cap_nodes, n_cells)
    return FM.DeviceRadixGrid(
        x_min, h0, ell, 0, n_cells,
        Int[], Int[], copy(cell_keys), zeros(Int, 2, n_cells),
        Int[], Int[], zeros(TF, 3, n_cells),
        Int[], UInt64[], zeros(Int, 3, cap_nodes), zeros(TF, 3, cap_nodes),
        Int[], zeros(Int, 2, cap_nodes), Int[],
    )
end

function run_case(TF, seed, n, ell, cluster, first_level)
    rng = MersenneTwister(seed)
    h0 = TF(1)
    x_min = SVector{3,TF}(-1, -1, -1)
    ctrs = [SVector{3,TF}(rand(rng, TF, 3) .* TF(1.8) .- TF(0.9)) for _ in 1:cluster]
    positions = Matrix{TF}(undef, 3, n)
    for i in 1:n
        c = ctrs[rand(rng, 1:cluster)]
        positions[:, i] .= clamp.(c .+ TF(0.02) .* randn(rng, TF, 3), TF(-1), TF(1))
    end

    # ---- device: cell keys via the stage-1 KA path ----
    cap = n + 5
    d_pos = devarray(positions)
    d_keys = _z(UInt64, cap); d_perm = _z(Int, cap); d_sorted = _z(UInt64, cap)
    d_invperm = _z(Int, cap); d_flags = _z(Int, cap); d_prefix = _z(Int, cap)
    max_cells = n + 5
    d_cell_keys = _z(UInt64, max_cells)
    d_cell_ranges = _z(Int, 2, max_cells)
    kv = view(d_keys, 1:n)
    ext.ka_radix_keys_checked!(kv, _z(Int32, 1), zeros(Int32, 1), d_pos, x_min,
        SVector{3,TF}(2, 2, 2), h0, ell)
    sv = view(d_sorted, 1:n)
    ext.ka_radix_sort_bodies!(view(d_perm, 1:n), sv, view(d_invperm, 1:n), kv)
    n_cells = ext.ka_radix_compress_cells!(d_cell_keys, d_cell_ranges, sv,
        view(d_flags, 1:n), view(d_prefix, 1:n), zeros(Int, 1))

    # ---- host reference: the real node builder ----
    ref_cell_keys = Array(d_cell_keys)[1:n_cells]
    cap_nodes = 2 * n_cells * (ell + 1) + 8
    hgrid = host_grid(TF, x_min, h0, ell, ref_cell_keys, cap_nodes, n_cells)
    ref_offsets = zeros(Int, ell + 2)
    ref_nodes = FM._refresh_radix_nodes!(hgrid, ref_offsets, n_cells, first_level)
    ref_max_count = maximum(ref_offsets[l + 2] - ref_offsets[l + 1]
                            for l in first_level:ell)

    # ---- stage 3 ----
    level_keys = _z(UInt64, max_cells, ell + 1)
    level_flags = _z(Int, max_cells, ell + 1)
    level_prefix = _z(Int, max_cells, ell + 1)
    level_counts = _z(Int, ell + 1)
    host_level_counts = zeros(Int, ell + 1)
    d_level_offsets = _z(Int, ell + 2)
    d_node_keys = _z(UInt64, cap_nodes)
    offsets = zeros(Int, ell + 2)
    n_nodes, max_count = ext.ka_radix_level_nodes!(d_node_keys, offsets,
        level_keys, level_flags, level_prefix, level_counts, host_level_counts,
        d_level_offsets, view(d_cell_keys, 1:n_cells), n_cells, ell, first_level,
        cap_nodes)

    tag = "(seed=$seed ell=$ell first_level=$first_level)"
    n_nodes == ref_nodes || error("n_nodes $n_nodes != $ref_nodes $tag")
    offsets == ref_offsets || error("level_offsets mismatch $tag")
    Array(d_level_offsets) == ref_offsets || error("d_level_offsets mismatch $tag")
    max_count == ref_max_count || error("max_count $max_count != $ref_max_count $tag")
    Array(d_node_keys)[1:n_nodes] == hgrid.node_keys[1:n_nodes] ||
        error("node_keys mismatch $tag")
    return n_cells, n_nodes
end

cases = (
    # (seed, n, ell, cluster centers, first_level)
    (1, 1000,  3, 1,   0),   # every body in one cell: one node per level
    (2, 1000,  3, 8,   0),
    (3, 5000,  4, 40,  0),
    (4, 20000, 5, 200, 0),
    (5, 1,     3, 1,   0),   # single body
    (6, 4096,  2, 64,  0),   # ell=2, densely occupied
    (7, 5000,  4, 40,  2),   # trimmed: levels 0-1 never keyed
    (8, 20000, 5, 200, 3),   # trimmed deeper
    (9, 5000,  4, 40,  4),   # first_level == ell: leaves are roots
)

for (seed, n, ell, cluster, first_level) in cases
    nc, nn = run_case(Float32, seed, n, ell, cluster, first_level)
    println("✓ seed=$seed n=$n ell=$ell clusters=$cluster first_level=$first_level: " *
            "$nc cells -> $nn nodes, level_offsets/node_keys/max_count exact")
end

println("\n✓✓✓ KA grid level-nodes gate passed on $DEV_NAME ✓✓✓")
