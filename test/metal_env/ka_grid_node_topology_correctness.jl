# Correctness gate for stage 4 of the in-place device grid rebuild:
# `ka_radix_node_topology!` (ext/FastMultipoleKAExt.jl), the KA port of the node
# geometry / parent-index / child-range trio plus `leaf_to_node` -- the last
# block of `_cuda_update_radix_grid_in_place!`
# (src/translate_batched_cuda.jl:6591).
#
# Oracle: `_refresh_radix_nodes!` (src/tree_batched.jl), the same host node
# builder stage 3 was gated against. Stage 3 compared the part of its output
# stage 3 owns (`level_offsets`, `node_keys`); this gate compares the rest --
# `node_levels`, `node_coords`, `node_centers`, `parent_index`, `child_ranges`
# and `leaf_to_node`. The two agreeing on all of it is the whole grid rebuild
# matching the CPU builder.
#
# The comparison is meaningful only because the device resolves parents and
# children by *binary search* into the adjacent level's block while the host
# walks a sorted merge: same answer, different algorithm, so an elementwise
# match is a real cross-check and not a transcription.
#
# Cases sweep `ell`, cluster count and `first_level`. `first_level` matters
# twice here: the trimmed levels must stay untouched, and nodes at
# `first_level` are roots whose `parent_index` must be 0 rather than a
# binary-search hit in a block that does not exist.
include("ka_backend.jl")
using FastMultipole, Random, StaticArrays, Test

const FM = FastMultipole
const ext = Base.get_extension(FastMultipole, :FastMultipoleKAExt)

println("Starting KA grid node-topology correctness test on $DEV_NAME...")
if !dev_functional()
    println("$DEV_NAME not functional; skipping")
    exit(0)
end

_z(T, dims...) = (a = KernelAbstractions.allocate(DEV_BACKEND, T, dims...);
    fill!(a, zero(T)); a)

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

    # ---- stages 1 and 3 on device ----
    cap = n + 5
    max_cells = n + 5
    d_pos = devarray(positions)
    d_keys = _z(UInt64, cap); d_perm = _z(Int, cap); d_sorted = _z(UInt64, cap)
    d_invperm = _z(Int, cap); d_flags = _z(Int, cap); d_prefix = _z(Int, cap)
    d_cell_keys = _z(UInt64, max_cells); d_cell_ranges = _z(Int, 2, max_cells)
    kv = view(d_keys, 1:n)
    ext.ka_radix_keys_checked!(kv, _z(Int32, 1), zeros(Int32, 1), d_pos, x_min,
        SVector{3,TF}(2, 2, 2), h0, ell)
    sv = view(d_sorted, 1:n)
    ext.ka_radix_sort_bodies!(view(d_perm, 1:n), sv, view(d_invperm, 1:n), kv)
    n_cells = ext.ka_radix_compress_cells!(d_cell_keys, d_cell_ranges, sv,
        view(d_flags, 1:n), view(d_prefix, 1:n), zeros(Int, 1))

    cap_nodes = 2 * n_cells * (ell + 1) + 8
    d_node_keys = _z(UInt64, cap_nodes)
    offsets = zeros(Int, ell + 2)
    n_nodes, max_count = ext.ka_radix_level_nodes!(d_node_keys, offsets,
        _z(UInt64, max_cells, ell + 1), _z(Int, max_cells, ell + 1),
        _z(Int, max_cells, ell + 1), _z(Int, ell + 1), zeros(Int, ell + 1),
        _z(Int, ell + 2), view(d_cell_keys, 1:n_cells), n_cells, ell,
        first_level, cap_nodes)
    d_level_offsets = devarray(offsets)

    # ---- host reference ----
    hgrid = host_grid(TF, x_min, h0, ell, Array(d_cell_keys)[1:n_cells],
        cap_nodes, n_cells)
    ref_offsets = zeros(Int, ell + 2)
    FM._refresh_radix_nodes!(hgrid, ref_offsets, n_cells, first_level)
    offsets == ref_offsets || error("stage-3 offsets diverged (seed=$seed)")

    # ---- stage 4 ----
    d_levels = _z(Int, cap_nodes)
    d_coords = _z(Int, 3, cap_nodes)
    d_centers = _z(TF, 3, cap_nodes)
    d_parent = _z(Int, cap_nodes)
    d_child = _z(Int, 2, cap_nodes)
    d_leaf_to_node = _z(Int, max_cells)
    ext.ka_radix_node_topology!(d_levels, d_coords, d_centers, d_parent, d_child,
        d_leaf_to_node, d_node_keys, d_level_offsets, offsets, x_min, h0,
        n_cells, ell, first_level, max_count)

    tag = "(seed=$seed ell=$ell first_level=$first_level)"
    v = 1:n_nodes
    Array(d_levels)[v] == hgrid.node_levels[v] || error("node_levels mismatch $tag")
    Array(d_coords)[:, v] == hgrid.node_coords[:, v] || error("node_coords mismatch $tag")
    Array(d_centers)[:, v] == hgrid.node_centers[:, v] || error("node_centers mismatch $tag")
    Array(d_parent)[v] == hgrid.parent_index[v] || error("parent_index mismatch $tag")
    Array(d_child)[:, v] == hgrid.child_ranges[:, v] || error("child_ranges mismatch $tag")
    Array(d_leaf_to_node)[1:n_cells] == hgrid.leaf_to_node[1:n_cells] ||
        error("leaf_to_node mismatch $tag")

    # roots must be roots: every node at first_level has no parent, and every
    # node above the deepest level must own the children pointing back at it
    hp = Array(d_parent); hc = Array(d_child)
    all(hp[i] == 0 for i in 1:offsets[first_level + 2]) ||
        error("a node at first_level has a nonzero parent $tag")
    for node in (offsets[first_level + 2] + 1):n_nodes
        p = hp[node]
        p != 0 || error("non-root node $node has no parent $tag")
        hc[1, p] <= node < hc[1, p] + hc[2, p] ||
            error("node $node outside its parent's child range $tag")
    end
    return n_cells, n_nodes
end

cases = (
    # (seed, n, ell, cluster centers, first_level)
    (1, 1000,  3, 1,   0),   # one cell: a single chain of nodes
    (2, 1000,  3, 8,   0),
    (3, 5000,  4, 40,  0),
    (4, 20000, 5, 200, 0),
    (5, 1,     3, 1,   0),   # single body
    (6, 4096,  2, 64,  0),   # ell=2, densely occupied
    (7, 5000,  4, 40,  2),   # trimmed: levels 0-1 never built
    (8, 20000, 5, 200, 3),   # trimmed deeper
    (9, 5000,  4, 40,  4),   # first_level == ell: every node is both root and leaf
)

for (seed, n, ell, cluster, first_level) in cases
    nc, nn = run_case(Float32, seed, n, ell, cluster, first_level)
    println("✓ seed=$seed n=$n ell=$ell clusters=$cluster first_level=$first_level: " *
            "$nc cells -> $nn nodes, levels/coords/centers/parents/children/" *
            "leaf_to_node exact")
end

println("\n✓✓✓ KA grid node-topology gate passed on $DEV_NAME ✓✓✓")
