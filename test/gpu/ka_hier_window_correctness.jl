# Correctness gate for the KA hierarchical M2L window cache
# (`ka_hier_cache_windows!`, ext/ka/ka_finalize_refresh.jl), which fills the
# epoch route stream `hctx.win_targets/win_sources/win_class` that
# `ka_hierarchical_m2l!` applies in one call.
#
# Oracle: `build_hierarchical_routes_window!` (src/interaction_list_batched.jl),
# the host CPU builder, run window by window. It emits a level's routes
# class-major/source-major; the device stream is target-major within a level
# (so the M2L scatter can sum a target's routes in a fixed order), so each
# level's host routes are sorted by (target, class) -- unique within a level --
# and the comparison stays elementwise, not set-wise.
#
# Setup: a plain CPU `RadixFMMCache` with the default hierarchical policy gives
# the grid, the dense per-level occupancy, and a populated
# `HostHierarchicalM2LContext`. The device side rebuilds that context on the KA
# backend via `ka_hierarchical_context` and drives the cache generator directly
# with a NamedTuple grid (it reads `grid.node_coords` only), so no
# `DeviceResidentRadixState` -- and therefore no resident lifecycle -- is needed.
#
# `hctx.node_at` is zeroed at construction on both backends and refilled from the
# resident grid by `ka_hier_refresh_occupancy!`. This gate drives that
# refresh too, and
# checks the resulting device lookup against the host occupancy the CPU cache
# refreshed before using it -- so a scatter bug surfaces as itself rather than as
# a downstream window mismatch.
include("ka_backend.jl")
include("../helpers/gravitational.jl")
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
    # Pin the concat strategy: the default measures its way to a dense
    # plan, and the route-class ids compared below are the concat convention
    # (level-true).
    cache = RadixFMMCache(sys; expansion_order=P, ell=ell,
        options=RadixLifecycleOptions(; m2l_strategy=ConcatenatedFixedZM2L()))
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

    # ---- device context ----
    hctx = ext.ka_hierarchical_context(DEV_BACKEND, ctx.tables,
        ctx.level_class_of, ctx.apply_plan, Int(grid.ell),
        ctx.first_m2l_level, occ; window_classes=ctx.window_classes)
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

    # host oracle: every window, concatenated in generation order
    cap = length(state.route_sources)
    h_class = zeros(Int32, cap)
    want_targets = Int[]; want_sources = Int[]; want_class = Int32[]
    nwin = 0
    for L in lo:Int(grid.ell)
        lt = Int[]; ls = Int[]; lc = Int32[]
        for first in 1:K:noffsets
            last = min(first + K - 1, noffsets)
            n_host = FM.build_hierarchical_routes_window!(state.route_levels,
                state.route_offsets, state.route_targets, state.route_sources,
                h_class, ctx, grid, L, first, last)
            n_host == 0 && continue
            append!(lt, state.route_targets[1:n_host])
            append!(ls, state.route_sources[1:n_host])
            append!(lc, h_class[1:n_host])
            nwin += 1
        end
        o = sortperm(collect(zip(lt, lc)))
        append!(want_targets, lt[o]); append!(want_sources, ls[o]); append!(want_class, lc[o])
    end

    # device: the epoch window cache the production step builds
    ext.ka_hier_cache_windows!(hctx, dgrid)
    hctx.win_valid || error("window cache not marked valid")
    n = hctx.total_routes
    n == length(want_targets) || error("total routes device=$n host=$(length(want_targets))")
    if n > 0
        Array(hctx.win_targets)[1:n] == want_targets || error("win_targets mismatch")
        Array(hctx.win_sources)[1:n] == want_sources || error("win_sources mismatch")
        Array(hctx.win_class)[1:n]   == want_class   || error("win_class mismatch")
    end
    return nwin
end

total = 0
for (seed, n, ell, P) in ((26031, 500, 3, 4), (26032, 2000, 3, 4),
                          (26033, 2000, 4, 4), (26034, 5000, 4, 8))
    nwin = run_case(seed, n, ell, P)
    global total += nwin
    println("✓ seed=$seed n=$n ell=$ell P=$P: cached stream of $nwin nonempty windows matches the host builder")
end

println("\n✓✓✓ KA hierarchical window gate passed on $DEV_NAME ($total nonempty windows) ✓✓✓")
