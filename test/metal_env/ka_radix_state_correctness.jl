# Correctness check for step (v) of the dispatch-wiring plan: ka_radix_state,
# the entry that hands the tree in `actx.grid` to a `DeviceResidentRadixState`.
#
# Gated on Metal against a CPU oracle, not on CUDA. `DeviceResidentRadixState`
# carries no CUDA in its type (every array field is separately parameterized),
# and the packing/route helpers the CUDA path validates against --
# `_host_radix_body_matrix` and `_host_radix_tree_routes` -- are
# backend-independent CPU functions over a `DeviceRadixGrid`. So the oracle here
# is: download `actx.grid`, truncate it to its logical extents, run FastMultipole's
# OWN host helpers on it, and compare against what the device state holds. That
# checks exactly what step (v) is -- the plumbing -- with no lifecycle execution.
#
# What this deliberately does NOT check, because step (v) does not do it:
# running the lifecycle (B2M/L2B have no KA port), the interaction list, and the
# operator workspace. Those fields are asserted to be `nothing`/empty so a later
# step that populates them has to update this file rather than silently drift.
include("ka_backend.jl")
using FastMultipole, Random, Test

const FM = FastMultipole

# Rebuild a host-side DeviceRadixGrid from the device grid, truncated to the
# logical extents. This is the oracle's input: FastMultipole's host helpers take
# a DeviceRadixGrid, so feeding them the downloaded tree runs the real reference
# implementation rather than a reimplementation of it.
function host_grid_from_device(grid, n::Int, n_cells::Int, n_nodes::Int)
    return FM.DeviceRadixGrid(
        grid.x_min, grid.h0, grid.ell, n, n_cells,
        Array(grid.perm)[1:n], Array(grid.invperm)[1:n],
        Array(grid.cell_keys)[1:n_cells], Array(grid.cell_ranges)[:, 1:n_cells],
        Array(grid.body_system)[1:n], Array(grid.body_index)[1:n],
        Array(grid.cell_centers)[:, 1:n_cells],
        Array(grid.node_levels)[1:n_nodes], Array(grid.node_keys)[1:n_nodes],
        Array(grid.node_coords)[:, 1:n_nodes], Array(grid.node_centers)[:, 1:n_nodes],
        Array(grid.parent_index)[1:n_nodes], Array(grid.child_ranges)[:, 1:n_nodes],
        Array(grid.leaf_to_node)[1:n_cells],
    )
end

if !dev_functional()
    println("$(DEV_NAME) not functional; skipping")
    exit(0)
end

ext = Base.get_extension(FastMultipole, :FastMultipoleKAExt)
ext !== nothing || error("FastMultipoleKAExt did not load")

const CASES = [
    # (n, K_max, ell_max, P, lamb_helmholtz, dpb, balance)
    (   50,  4, 4,  2, false, 5, true),
    (  400,  8, 5,  4, false, 8, true),
    (  400,  8, 5,  4,  true, 8, true),
    ( 2000, 16, 6,  4, false, 8, true),
    ( 2000,  4, 6,  2, false, 5, false),
    (    1,  4, 4,  2, false, 5, true),
]

npass = 0
for (case_i, (n, K_max, ell_max, P, lh, dpb, balance)) in pairs(CASES)
    Random.seed!(4700 + case_i)
    positions = rand(Float32, 3, n)
    source_buffer = rand(Float32, dpb, n)
    x_min = (0.0f0, 0.0f0, 0.0f0)
    h0 = 1.0f0

    nl_estimate = max(2 * n ÷ K_max, 16)
    node_capacity = 100 * nl_estimate + 256
    leaf_capacity = 10 * nl_estimate + 256
    frontier_capacity = 16 * leaf_capacity

    actx = ext.ka_allocate_adaptive_context(DEV_BACKEND, Float32, n;
        leaf_capacity, frontier_capacity, node_capacity)
    build = ext.ka_build_adaptive_tree!(actx, devarray(positions), ell_max, K_max,
        balance, x_min, h0)

    options = FM.CUDARadixLifecycleOptions(; precision=Float32)
    state = ext.ka_radix_state(actx, build, devarray(source_buffer), P, Val(lh);
        options=options)

    n_nodes = build.n_nodes
    n_cells = build.n_leaves
    hgrid = host_grid_from_device(actx.grid, n, n_cells, n_nodes)

    # --- the grid goes in by reference, not as a copy ---
    @test state.grid === actx.grid

    # --- logical extents live in counts, not in array lengths ---
    @test state.counts.n_bodies == n
    @test state.counts.n_cells == n_cells
    @test state.counts.n_nodes == n_nodes
    @test length(state.grid.perm) == actx.maxn        # still capacity-sized
    @test length(state.grid.node_keys) == node_capacity

    # --- body packing vs FastMultipole's own host packer ---
    ref_bodies = FM._host_radix_body_matrix(hgrid, (source_buffer,))
    @test size(ref_bodies) == (dpb, n)
    # `source_bodies` carries exactly the dpb rows CUDA packs, so it mirrors the
    # host packer row for row.
    @test size(state.source_bodies, 1) == dpb
    @test Array(state.source_bodies)[1:dpb, 1:n] == ref_bodies
    @test state.target_bodies === state.source_bodies

    # --- M2M/L2L edge list vs FastMultipole's own host route builder ---
    ref_m2m_p, ref_m2m_c, ref_l2l_p, ref_l2l_c = FM._host_radix_tree_routes(hgrid)
    n_edges = n_nodes - 1
    @test length(ref_m2m_p) == n_edges
    @test Array(state.m2m_parent_routes)[1:n_edges] == ref_m2m_p
    @test Array(state.m2m_child_routes)[1:n_edges] == ref_m2m_c
    @test Array(state.l2l_parent_routes)[1:n_edges] == ref_l2l_p
    @test Array(state.l2l_child_routes)[1:n_edges] == ref_l2l_c

    # --- expansion buffers: node-capacity columns, zeroed, LH-correct width ---
    bi = state.multipoles.basis_info
    @test size(state.multipoles.phi) == (bi.basis_dof_phi, node_capacity)
    @test size(state.multipoles.chi) == (lh ? (bi.basis_dof_chi, node_capacity) : (0, 0))
    @test size(state.locals.phi) == size(state.multipoles.phi)
    @test all(iszero, Array(state.multipoles.phi))
    @test all(iszero, Array(state.locals.phi))
    @test all(iszero, Array(state.multipoles.chi))
    @test size(state.output) == (4, actx.maxn)
    @test all(iszero, Array(state.output))

    # --- aliases into the grid, not copies ---
    @test state.body_perm === actx.grid.perm
    @test state.cell_ranges === actx.grid.cell_ranges
    @test state.cell_centers === actx.grid.cell_centers

    # --- host body mirrors are downloaded and correct for this build ---
    @test state.host_body_perm == Array(actx.grid.perm)[1:n]
    @test state.host_body_system_ids == Array(actx.grid.body_system)[1:n]
    @test state.host_body_indices == Array(actx.grid.body_index)[1:n]

    # --- not yet wired; a later step must update these assertions ---
    @test state.interaction_list === nothing
    @test state.scratch === nothing
    @test state.host_node_levels === nothing
    @test state.host_route_targets === nothing
    @test length(state.route_targets) == 0
    @test length(state.direct_targets) == 0
    @test state.counts.n_routes == 0
    @test state.counts.n_direct == 0

    global npass += 1
    println("✓ n=$n K_max=$K_max ell_max=$ell_max P=$P lh=$lh dpb=$dpb balance=$balance " *
            "-> nodes=$n_nodes cells=$n_cells edges=$n_edges")
end

println("\nStep (v) ka_radix_state: $npass/$(length(CASES)) cases passed")
println("\n✓✓✓ All KA resident-state handoff correctness tests passed on $(DEV_NAME)! ✓✓✓")
