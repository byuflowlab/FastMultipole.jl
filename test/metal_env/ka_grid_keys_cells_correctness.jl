# Correctness gate for stage 1 of the in-place device grid rebuild:
# `ka_radix_keys_checked!`, `ka_radix_sort_bodies!` and
# `ka_radix_compress_cells!` (ext/FastMultipoleKAExt.jl), the KA port of the
# first two stages of `_cuda_update_radix_grid_in_place!`
# (src/translate_batched_cuda.jl:6591) -- the last CUDA-only block on the
# uniform, `sfs=false` path of `update_cuda_radix_state!`.
#
# Oracle: the host builder's own three steps, `_radix_fill_body_data!`,
# `_host_radix_sort_permutation` and `_compress_radix_cells`
# (src/tree_batched.jl), which is what `_radix_grid` runs on the CPU. Both the
# host radix sort and the device `sortperm!` are stable, so `perm` is compared
# elementwise, not as a per-cell set.
#
# Cases sweep the two things that break here: `ell` (key width, and whether the
# grid is sparsely or fully occupied) and the body distribution (clustered
# positions put many bodies in one cell, which is what exercises the
# flag/scan/compact cell compression). The out-of-bounds guard -- the reason
# this needed a second key kernel rather than reusing `ka_radix_keys!` -- is
# checked on its own at the end.
include("ka_backend.jl")
using FastMultipole, Random, StaticArrays, Test

const FM = FastMultipole
const ext = Base.get_extension(FastMultipole, :FastMultipoleKAExt)

println("Starting KA grid keys/sort/cells correctness test on $DEV_NAME...")
if !dev_functional()
    println("$DEV_NAME not functional; skipping")
    exit(0)
end

# host key fill, straight out of `_radix_fill_body_data!` but from a position
# matrix rather than a system tuple (the device stage reads `ctx.positions`)
function host_keys(positions, x_min, h0, ell)
    n = size(positions, 2)
    keys = Vector{UInt64}(undef, n)
    for i in 1:n
        coord = FM.radix_cell_coord(x_min, h0, ell, SVector{3}(positions[:, i]))
        keys[i] = FM.morton_key(coord, ell)
    end
    return keys
end

function run_case(TF, seed, n, ell, cluster)
    rng = MersenneTwister(seed)
    h0 = TF(1)
    x_min = SVector{3,TF}(-1, -1, -1)
    box_extent = SVector{3,TF}(2, 2, 2)
    # `cluster` cluster centers, so many bodies share a leaf cell
    centers = [SVector{3,TF}(rand(rng, TF, 3) .* TF(1.8) .- TF(0.9)) for _ in 1:cluster]
    positions = Matrix{TF}(undef, 3, n)
    for i in 1:n
        c = centers[rand(rng, 1:cluster)]
        positions[:, i] .= clamp.(c .+ TF(0.02) .* randn(rng, TF, 3), TF(-1), TF(1))
    end

    # ---- host reference ----
    ref_keys = host_keys(positions, x_min, h0, ell)
    ref_perm = FM._host_radix_sort_permutation(ref_keys)
    ref_invperm = Vector{Int}(undef, n)
    for i in eachindex(ref_perm); ref_invperm[ref_perm[i]] = i; end
    ref_sorted = ref_keys[ref_perm]
    ref_cell_keys, ref_cell_ranges = FM._compress_radix_cells(ref_keys, ref_perm)
    n_cells_ref = length(ref_cell_keys)

    # ---- device: capacity-sized storage, as in the cache ----
    cap = n + 5
    max_cells = n_cells_ref + 3
    _z(T, dims...) = (a = KernelAbstractions.allocate(DEV_BACKEND, T, dims...);
        fill!(a, zero(T)); a)
    d_pos = devarray(positions)
    d_keys = _z(UInt64, cap)
    d_perm = _z(Int, cap)
    d_sorted = _z(UInt64, cap)
    d_invperm = _z(Int, cap)
    d_flags = _z(Int, cap)
    d_prefix = _z(Int, cap)
    d_cell_keys = _z(UInt64, max_cells)
    d_cell_ranges = _z(Int, 2, max_cells)
    oob_flag = _z(Int32, 1)
    host_oob = zeros(Int32, 1)
    host_scalar = zeros(Int, 1)

    kv = view(d_keys, 1:n)
    ext.ka_radix_keys_checked!(kv, oob_flag, host_oob, d_pos, x_min, box_extent,
        h0, ell)
    Array(d_keys)[1:n] == ref_keys || error("keys mismatch (seed=$seed ell=$ell)")

    pv = view(d_perm, 1:n)
    sv = view(d_sorted, 1:n)
    ext.ka_radix_sort_bodies!(pv, sv, view(d_invperm, 1:n), kv)
    got_sorted = Array(d_sorted)[1:n]
    got_sorted == ref_sorted || error("sorted_keys mismatch (seed=$seed ell=$ell)")
    Array(d_perm)[1:n] == ref_perm || error("perm mismatch (seed=$seed ell=$ell)")
    Array(d_invperm)[1:n] == ref_invperm || error("invperm mismatch (seed=$seed ell=$ell)")

    n_cells = ext.ka_radix_compress_cells!(d_cell_keys, d_cell_ranges, sv,
        view(d_flags, 1:n), view(d_prefix, 1:n), host_scalar)
    n_cells == n_cells_ref ||
        error("n_cells $n_cells != $n_cells_ref (seed=$seed ell=$ell)")
    c = 1:n_cells
    Array(d_cell_keys)[c] == ref_cell_keys ||
        error("cell_keys mismatch (seed=$seed ell=$ell)")
    Array(d_cell_ranges)[:, c] == ref_cell_ranges ||
        error("cell_ranges mismatch (seed=$seed ell=$ell)")
    return n_cells
end

cases = (
    # (seed, n, ell, cluster centers)
    (1, 1000,  3, 1),      # every body in one cell
    (2, 1000,  3, 8),
    (3, 5000,  4, 40),
    (4, 20000, 5, 200),
    (5, 1,     3, 1),      # single body
    (6, 4096,  2, 64),     # ell=2: 64 cells, fully occupied
)

for (seed, n, ell, cluster) in cases
    nc = run_case(Float32, seed, n, ell, cluster)
    println("✓ seed=$seed n=$n ell=$ell clusters=$cluster: keys/perm/invperm/" *
            "sorted_keys exact, $nc cells compressed exactly")
end

# out-of-bounds guard: one body pushed just outside the fixed box must throw
let TF = Float32, n = 64, ell = 3
    h0 = TF(1)
    x_min = SVector{3,TF}(-1, -1, -1)
    box_extent = SVector{3,TF}(2, 2, 2)
    positions = zeros(TF, 3, n)
    positions[1, 7] = TF(1.5)
    d_keys = devarray(zeros(UInt64, n))
    oob = devarray(zeros(Int32, 1))
    threw = false
    try
        ext.ka_radix_keys_checked!(d_keys, oob, zeros(Int32, 1),
            devarray(positions), x_min, box_extent, h0, ell)
    catch e
        threw = e isa ArgumentError
    end
    threw || error("out-of-bounds body did not throw ArgumentError")
    println("✓ out-of-bounds guard throws ArgumentError")
end

println("\n✓✓✓ KA grid keys/sort/cells gate passed on $DEV_NAME ✓✓✓")
