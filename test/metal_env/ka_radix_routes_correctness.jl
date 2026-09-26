# Correctness gate for `ka_generate_radix_routes!` (ext/FastMultipoleKAExt.jl),
# the KA port of `_cuda_generate_radix_routes!`
# (src/translate_batched_cuda.jl:6174) -- the flat/uniform arm of
# `update_cuda_radix_state!`'s route stage, and one of the two remaining
# CUDA-only blocks on the uniform, `sfs=false` path.
#
# Oracle: `build_radix_routes!` (src/interaction_list_batched.jl:966), the host
# CPU builder. It emits M2L routes offset-class-major/cell-minor and direct
# pairs target-major/offset-minor -- exactly the two flat index decompositions
# the device kernels use -- so the comparison is elementwise, not set-wise.
#
# Setup is synthetic rather than cache-derived, deliberately: the generator is
# agnostic to where the offset sets came from (it reads `d_accepted` /
# `d_rejected` as plain Int32 matrices), and driving it from a random occupied
# cell set plus a Chebyshev accept/reject split lets the cases sweep the two
# things that actually break it -- the chunked class loop (`class_chunk` smaller
# than the accepted count, so the flag/prefix buffer is reused across chunks
# with a running `n_routes` base) and the chunked direct loop (`direct_flags`
# capacity smaller than `nreject * n_cells`).
include("ka_backend.jl")
using FastMultipole, Random, StaticArrays, Test

const FM = FastMultipole
const ext = Base.get_extension(FastMultipole, :FastMultipoleKAExt)

println("Starting KA flat radix route generation test on $DEV_NAME...")
if !dev_functional()
    println("$DEV_NAME not functional; skipping")
    exit(0)
end

offsets_matrix(offs) = (m = Matrix{Int32}(undef, 3, length(offs));
    for (k, o) in enumerate(offs); m[1, k] = o[1]; m[2, k] = o[2]; m[3, k] = o[3]; end; m)

# Chebyshev split over the offset ball of radius `radius`: |d|inf <= 1 is the
# near/self complement (direct), the rest is accepted (M2L). Accepted is sorted
# by (z, y, x) to match the RadixInteractionList batch order the host builder
# documents.
function offset_sets(radius)
    accepted = SVector{3,Int}[]
    rejected = SVector{3,Int}[]
    for dz in -radius:radius, dy in -radius:radius, dx in -radius:radius
        d = SVector{3,Int}(dx, dy, dz)
        push!(max(abs(dx), abs(dy), abs(dz)) <= 1 ? rejected : accepted, d)
    end
    return accepted, rejected
end

function run_case(seed, ell, n_cells, radius, class_chunk, direct_cap_frac)
    rng = MersenneTwister(seed)
    G = 1 << ell
    n_cells <= G^3 || error("case asks for more cells than the grid has")
    cell_keys = sort!(collect(UInt64.(randperm(rng, G^3)[1:n_cells] .- 1)))
    coords = [FM.morton_decode(k, ell) for k in cell_keys]
    accepted, rejected = offset_sets(radius)
    leaf_offset = 37   # arbitrary nonzero node offset of the leaf level
    leaf_to_node = [leaf_offset + c for c in 1:n_cells]

    # ---- host reference ----
    cap_routes = length(accepted) * n_cells
    cap_direct = length(rejected) * n_cells
    ref_levels = fill(-1, cap_routes)
    ref_offsets = fill(-1, 3, cap_routes)
    ref_targets = fill(-1, cap_routes)
    ref_sources = fill(-1, cap_routes)
    ref_class = fill(Int32(-1), cap_routes)
    ref_dtargets = fill(-1, cap_direct)
    ref_dsources = fill(-1, cap_direct)
    cell_at_host = zeros(Int32, G, G, G)
    FM.refresh_cell_at!(cell_at_host, cell_keys, n_cells, ell)
    n_routes_ref, n_direct_ref = FM.build_radix_routes!(ref_levels, ref_offsets,
        ref_targets, ref_sources, ref_class, ref_dtargets, ref_dsources,
        accepted, rejected, cell_at_host, coords, leaf_to_node, ell, n_cells)

    # ---- device context: only the fields the generator reads ----
    cc = min(class_chunk, length(accepted))
    flag_capacity = cc * n_cells
    direct_flag_capacity = max(cld(cap_direct, direct_cap_frac), 1)
    _z(T, dims...) = (a = KernelAbstractions.allocate(DEV_BACKEND, T, dims...);
        fill!(a, zero(T)); a)
    ctx = (;
        cell_at = _z(Int32, G, G, G),
        class_chunk = cc,
        d_accepted = devarray(offsets_matrix(accepted)),
        d_rejected = devarray(offsets_matrix(rejected)),
        route_flags = _z(Int32, flag_capacity),
        route_prefix = _z(Int32, flag_capacity),
        direct_flags = _z(Int32, direct_flag_capacity),
        direct_prefix = _z(Int32, direct_flag_capacity),
        host_scalar32 = Vector{Int32}(undef, 1),
        route_levels = devarray(fill(-1, cap_routes)),
        route_offsets = devarray(fill(-1, 3, cap_routes)),
        route_targets = devarray(fill(-1, cap_routes)),
        route_sources = devarray(fill(-1, cap_routes)),
        direct_targets = devarray(fill(-1, cap_direct)),
        direct_sources = devarray(fill(-1, cap_direct)),
    )
    route_class = devarray(fill(Int32(-1), cap_routes))
    grid = (; cell_keys = devarray(cell_keys))

    n_routes, n_direct = ext.ka_generate_radix_routes!(ctx, grid, n_cells,
        leaf_offset, ell, route_class)

    tag = "seed=$seed ell=$ell n_cells=$n_cells r=$radius chunk=$cc"
    n_routes == n_routes_ref ||
        error("n_routes $n_routes != $n_routes_ref ($tag)")
    n_direct == n_direct_ref ||
        error("n_direct $n_direct != $n_direct_ref ($tag)")

    # occupancy scatter checked on its own, before the routes that consume it
    Array(ctx.cell_at) == cell_at_host || error("cell_at mismatch ($tag)")

    r = 1:n_routes
    Array(ctx.route_levels)[r] == ref_levels[r] || error("route_levels mismatch ($tag)")
    Array(ctx.route_offsets)[:, r] == ref_offsets[:, r] || error("route_offsets mismatch ($tag)")
    Array(ctx.route_targets)[r] == ref_targets[r] || error("route_targets mismatch ($tag)")
    Array(ctx.route_sources)[r] == ref_sources[r] || error("route_sources mismatch ($tag)")
    Array(route_class)[r] == ref_class[r] || error("route_class mismatch ($tag)")
    d = 1:n_direct
    Array(ctx.direct_targets)[d] == ref_dtargets[d] || error("direct_targets mismatch ($tag)")
    Array(ctx.direct_sources)[d] == ref_dsources[d] || error("direct_sources mismatch ($tag)")
    return n_routes, n_direct
end

cases = (
    # (seed, ell, n_cells, radius, class_chunk, direct flag capacity divisor)
    (1, 3, 64,  2, 10^6, 1),   # single class chunk, single direct chunk
    (2, 3, 200, 2, 7,     1),  # chunked class loop, running n_routes base
    (3, 4, 500, 2, 10^6,  4),  # chunked direct loop
    (4, 4, 300, 3, 5,     3),  # both loops chunked, wider offset ball
    (5, 2, 1,   2, 10^6,  1),  # single occupied cell: self-pair only
    (6, 3, 512, 2, 3,     5),  # fully occupied grid, both loops chunked
)

for c in cases
    nr, nd = run_case(c...)
    println("✓ seed=$(c[1]) ell=$(c[2]) n_cells=$(c[3]) r=$(c[4]) chunk=$(c[5]): " *
            "$nr routes, $nd direct pairs match the host builder")
end

println("\n✓✓✓ KA flat radix route gate passed on $DEV_NAME ✓✓✓")
