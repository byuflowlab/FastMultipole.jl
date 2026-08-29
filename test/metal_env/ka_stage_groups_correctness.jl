# Correctness gate for `ka_refresh_resident_stage_groups!` /
# `ka_refresh_group_edges_kernel!` (ext/FastMultipoleKAExt.jl), the KA port of
# `_cuda_refresh_resident_stage_groups!` (src/translate_batched_cuda.jl:6120).
# This is the stage of `update_cuda_radix_state!` that rebuilds the per-level
# M2M/L2L edge columns -- the (source, target) node index pairs and the
# spherical angles of each parent-child displacement -- after an occupancy change.
#
# Oracle: `_refresh_resident_stage_groups!` (translate_batched_resident.jl), the
# host CPU refresh, run over the same grid. Both walk levels in the same order
# and emit one edge per child node in ascending flat node index, so the
# comparison is elementwise per group.
#
# The host function additionally refills `ws.nonleaf_idx`; that is host-path-only
# storage (see the note at translate_batched_resident.jl:3589) and neither the
# CUDA nor the KA refresh touches it, so it is deliberately not compared.
include("ka_backend.jl")
include("../gravitational.jl")
using FastMultipole, Test

const FM = FastMultipole
const ext = Base.get_extension(FastMultipole, :FastMultipoleKAExt)

println("Starting KA stage-group refresh test on $DEV_NAME...")
if !dev_functional()
    println("$DEV_NAME not functional; skipping")
    exit(0)
end

# Float32 throughout: Metal has no Float64, so the host oracle runs at the same
# precision and the angle columns compare to a Float32 tolerance rather than a
# mixed-precision one.
const TF = Float32
const RTOL = 1f-6

function run_case(seed, n_bodies, ell, P)
    sys = generate_gravitational(seed, n_bodies)
    cache = RadixFMMCache(sys; expansion_order=P, ell=ell,
        options=CUDARadixLifecycleOptions(; precision=TF,
            m2l_strategy=ConcatenatedFixedZM2L()))
    state = cache.state
    grid = state.grid
    ws_host = state.scratch
    level_offsets = Vector{Int}(cache.level_offsets)
    ell_i = Int(cache.ell)
    first_level = Int(cache.root_level)

    FM._refresh_resident_stage_groups!(ws_host, grid, level_offsets)

    ws_dev = ext.ka_radix_cache_workspace(DEV_BACKEND, TF,
        state.invariant_cache.basis_info, ell_i, TF(cache.h0), Int(cache.max_cells),
        Int(cache.max_nodes), Int(cache.route_capacity), cache.accepted_offsets,
        state.invariant_cache; ell_axes=cache.ell_axes, first_level=first_level)

    n_nodes = level_offsets[end]
    dgrid = (parent_index = devarray(Vector{Int32}(grid.parent_index[1:n_nodes])),
             node_centers = devarray(Matrix{TF}(grid.node_centers[:, 1:n_nodes])))

    ext.ka_refresh_resident_stage_groups!(ws_dev, dgrid, level_offsets, ell_i,
        first_level)

    ngroups = 0
    nedges = 0
    for (kind, hg, dg) in Iterators.flatten((
            ((:m2m, h, d) for (h, d) in zip(ws_host.m2m_groups, ws_dev.m2m_groups)),
            ((:l2l, h, d) for (h, d) in zip(ws_host.l2l_groups, ws_dev.l2l_groups))))
        n = hg.count[]
        dg.count[] == n || error("$kind group count device=$(dg.count[]) host=$n")
        n == 0 && continue
        r = 1:n
        Array(dg.source_idx)[r] == Vector{Int}(hg.source_idx[r]) ||
            error("$kind group source_idx mismatch")
        Array(dg.target_idx)[r] == Vector{Int}(hg.target_idx[r]) ||
            error("$kind group target_idx mismatch")
        for (name, hv, dv) in ((:phis, hg.phis, dg.phis), (:thetas, hg.thetas, dg.thetas))
            h = Vector{TF}(hv[r])
            d = Array(dv)[r]
            scale = max(maximum(abs, h), one(TF))
            err = maximum(abs.(d .- h)) / scale
            err < RTOL || error("$kind group $name relerr=$err >= $RTOL")
        end
        ngroups += 1
        nedges += n
    end
    return ngroups, nedges
end

for (seed, n, ell, P) in ((26041, 500, 3, 4), (26042, 2000, 3, 4),
                          (26043, 2000, 4, 4), (26044, 8000, 5, 6))
    ngroups, nedges = run_case(seed, n, ell, P)
    println("✓ seed=$seed n=$n ell=$ell P=$P: $ngroups nonempty groups, $nedges edges match the host refresh")
end

println("\n✓✓✓ KA stage-group refresh gate passed on $DEV_NAME ✓✓✓")
