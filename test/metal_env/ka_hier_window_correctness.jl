# Correctness gate for the KA hierarchical M2L window generator
# (`ka_hier_generate_window_core!`, ext/FastMultipoleKAExt.jl), the KA port of
# `_cuda_hier_generate_window_core!` (src/translate_batched_cuda.jl:7350) and the
# last CUDA-only dependency inside `ka_hierarchical_m2l!`.
#
# Oracle: `build_hierarchical_routes_window!`
# (src/interaction_list_batched.jl:578), the host CPU builder. It emits the same
# arrays in the same class-major/source-major order as the device flat index
# (idx -> kloc = (idx-1)÷n_sources+1, s = (idx-1)%n_sources+1), so the comparison
# is elementwise, not set-wise.
#
# Setup: a plain CPU `RadixFMMCache` with the shipped hierarchical policy gives
# the grid, the dense per-level occupancy, and a populated
# `HostHierarchicalM2LContext`. The device side rebuilds that context on the KA
# backend via `ka_hierarchical_context` and drives the core directly with a
# NamedTuple grid (the core reads `grid.node_coords` only), so no
# `DeviceResidentRadixState` -- and therefore no resident lifecycle -- is needed.
#
# `hctx.node_at` is zeroed at construction on both backends and refilled from the
# resident grid by `ka_hier_refresh_occupancy!` (the KA port of
# `_cuda_hier_refresh_occupancy!`, :7214). This gate drives that port too, and
# checks the resulting device lookup against the host occupancy the CPU cache
# refreshed before using it -- so a scatter bug surfaces as itself rather than as
# a downstream window mismatch.
include("ka_backend.jl")
include("../gravitational.jl")
using FastMultipole, Random, Test

const FM = FastMultipole
const ext = Base.get_extension(FastMultipole, :FastMultipoleKAExt)

println("Starting KA hierarchical window correctness test on $DEV_NAME...")
if !dev_functional()
    println("$DEV_NAME not functional; skipping")
    exit(0)
end

function run_case(seed, n_bodies, ell, P; window_classes=8)
    sys = generate_gravitational(seed, n_bodies)
    # Pin the concat strategy: the shipped default measures its way to a dense
    # plan, and the route-class ids compared below are the concat convention
    # (level-true), the same pin cuda_radix_hierarchical_test.jl makes.
    cache = RadixFMMCache(sys; expansion_order=P, ell=ell,
        options=CUDARadixLifecycleOptions(; m2l_strategy=ConcatenatedFixedZM2L()))
    state = cache.state
    ctx = state.interaction_list
    ctx isa FM.HostHierarchicalM2LContext ||
        error("expected a hierarchical host cache; got $(typeof(ctx))")
    grid = state.grid
    occ = ctx.occupancy
    isempty(occ.node_at) && error("case needs the dense occupancy lookup")

    noffsets = length(ctx.tables.push_offsets)
    K = max(min(ctx.window_classes, noffsets), 1)
    lo = ctx.first_m2l_level
    # widest per-level node count -- bounds the device flag/prefix buffers
    max_level_nodes = maximum(diff(ctx.level_offsets[1:(grid.ell + 2)]))

    # ---- device context ----
    hctx = ext.ka_hierarchical_context(Float32, DEV_BACKEND, ctx.tables,
        ctx.class_level, ctx.class_offset, ctx.effective_offsets,
        ctx.level_class_of, Int[], ctx.apply_plan, Int(grid.ell),
        ctx.first_m2l_level, Int(max_level_nodes), occ;
        window_classes=ctx.window_classes)
    # Only the first `n_nodes` columns of the host grid are initialized; the
    # tail is undefined memory, so convert the valid prefix and leave the rest 0.
    n_nodes = ctx.level_offsets[grid.ell + 2]
    coords32 = zeros(Int32, 3, size(grid.node_coords, 2))
    coords32[:, 1:n_nodes] .= Int32.(grid.node_coords[:, 1:n_nodes])
    dgrid = (node_coords = devarray(coords32),
             node_levels = devarray(Int32.(grid.node_levels[1:n_nodes])))

    ext.ka_hier_refresh_occupancy!(hctx, dgrid, Vector{Int}(ctx.level_offsets))
    Array(hctx.node_at) == Vector{Int32}(occ.node_at) ||
        error("node_at scatter mismatch against the host occupancy lookup")

    cap = length(state.route_sources)
    d_levels = devarray(zeros(Int, cap))
    d_offsets = devarray(zeros(Int, 3, cap))
    d_targets = devarray(zeros(Int, cap))
    d_sources = devarray(zeros(Int, cap))
    d_class = devarray(zeros(Int32, cap))

    h_class = zeros(Int32, cap)
    nwin = 0
    for L in lo:Int(grid.ell), first in 1:K:noffsets
        last = min(first + K - 1, noffsets)

        n_host = FM.build_hierarchical_routes_window!(state.route_levels,
            state.route_offsets, state.route_targets, state.route_sources,
            h_class, ctx, grid, L, first, last)

        n_dev = ext.ka_hier_generate_window_core!(d_levels, d_offsets, d_targets,
            d_sources, dgrid, hctx, d_class, L, first, last,
            (L - lo) * noffsets)

        n_dev == n_host || error("L=$L window $first:$last: n_routes " *
            "device=$n_dev host=$n_host")
        n_host == 0 && continue

        r = 1:n_host
        Array(d_levels)[r]     == state.route_levels[r]   || error("L=$L $first:$last route_levels mismatch")
        Array(d_offsets)[:, r] == state.route_offsets[:, r] || error("L=$L $first:$last route_offsets mismatch")
        Array(d_targets)[r]    == state.route_targets[r]  || error("L=$L $first:$last route_targets mismatch")
        Array(d_sources)[r]    == state.route_sources[r]  || error("L=$L $first:$last route_sources mismatch")
        Array(d_class)[r]      == h_class[r]              || error("L=$L $first:$last route_class mismatch")
        nwin += 1
    end
    return nwin
end

total = 0
for (seed, n, ell, P) in ((26031, 500, 3, 4), (26032, 2000, 3, 4),
                          (26033, 2000, 4, 4), (26034, 5000, 4, 8))
    nwin = run_case(seed, n, ell, P)
    global total += nwin
    println("✓ seed=$seed n=$n ell=$ell P=$P: $nwin nonempty windows match the host builder")
end

println("\n✓✓✓ KA hierarchical window gate passed on $DEV_NAME ($total nonempty windows) ✓✓✓")
