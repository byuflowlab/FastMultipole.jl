# Local gate for `ka_lifecycle_body!` -- the whole uniform radix lifecycle
# (nearfield -> B2M -> M2M -> M2L -> L2L -> L2B), run end to end on a KA
# backend and compared against FastMultipole's OWN host resident lifecycle.
#
# Why this file exists: the session-3 note recorded `ka_lifecycle_body!` as
# "reachable only on a CUDA-resident state", and the acceptance gate was
# therefore aimed straight at H200. That is not true. `host_radix_state`
# (src/translate_batched_resident.jl) builds a complete
# `DeviceResidentRadixState` with no CUDA anywhere -- host body matrix, host
# tree routes, flat coefficient buffers and a real `ResidentOperatorWorkspace`
# -- and `run_host_radix_lifecycle!` runs every stage of it on the CPU. The
# state type carries no array type in its parameters, so the same state is
# constructible on any KA backend. That makes the whole lifecycle body gateable
# locally, against an exact CPU oracle, before any HPC round trip.
#
# The oracle is `run_host_radix_lifecycle!` over host `Array`s: B2M, then the
# operator pipeline M2M -> M2L -> L2L -> L2B, with `_add_host_direct_pairs!`
# folded into L2B. `ka_lifecycle_body!` runs nearfield first (clearing output)
# and folds direct into that stage instead; the stage ORDER differs but the sum
# does not, because output accumulates and starts zeroed on both sides.
#
# What this deliberately does NOT cover, so no one reads it as the acceptance
# gate:
#   1. `ka_hierarchical_m2l!`. `host_radix_state` stores a flat
#      `RadixInteractionList`, so `ka_launch_m2l!` takes its flat branch here.
#      The hierarchical branch reuses `_cuda_hier_generate_window!` for window
#      generation, which is CUDA-only by construction, so it stays an H200 gate.
#   2. The cache-level call. `_radix_cache_device_step!` is a stub that throws
#      `CUDARadixUnavailable` until translate_batched_cuda.jl redefines it, so
#      a full `UJ_fmm` through a `RadixFMMCache` cannot run off CUDA.
#   3. Timing. Metal-vs-CPU numbers say nothing about KA-vs-native on one GPU.
#
# What it DOES cover is every stage FLOWVPM's uniform per-step lifecycle runs,
# plus the sequencing between them -- which is where a silent wrong answer of
# the kind that bit the M2L branch would show up.
#
# Body type: `Point{Vortex}` with Lamb-Helmholtz on. Not a preference -- it is
# the only type with a KA B2M port (`ka_launch_b2m!` has exactly one body-type
# method, ext:2684), and it is what FLOWVPM runs. A `Point{Source}` case throws
# a `MethodError` at the B2M stage; that gap is deliberate and out of scope
# here, since the migration targets FLOWVPM's vortex lifecycle.
include("ka_backend.jl")
using FastMultipole, Random, Test
using FastMultipole.StaticArrays

const FM = FastMultipole

# the repo's own vortex system + generator (correct traits, body_type Point{Vortex})
include(joinpath(@__DIR__, "..", "vortex.jl"))

if !dev_functional()
    println("$(DEV_NAME) not functional; skipping")
    exit(0)
end
ext = Base.get_extension(FastMultipole, :FastMultipoleKAExt)
ext !== nothing || error("FastMultipoleKAExt did not load")

relerr(a, b) = (d = maximum(abs.(Array(a) .- Array(b))); s = maximum(abs.(Array(b)));
                s == 0 ? d : d / s)

#------- host state -> device state -------#
#
# `DeviceRadixGrid` and `DeviceResidentRadixState` are both fully generic in
# their array types, so the mirror is a field-by-field upload. The `host_*`
# fields of the state stay host `Array`s by contract (that is what they are
# for); everything else moves. `invariant_cache` stays host too: the KA stage
# drivers read only `basis_info.orders` off it, which is host scalars.

to_dev(x::AbstractArray) = devarray(x)
to_dev(x) = x

function dev_grid(g::FM.DeviceRadixGrid)
    return FM.DeviceRadixGrid(
        g.x_min, g.h0, g.ell, g.n_bodies, g.n_cells,
        to_dev(g.perm), to_dev(g.invperm), to_dev(g.cell_keys), to_dev(g.cell_ranges),
        to_dev(g.body_system), to_dev(g.body_index), to_dev(g.cell_centers),
        to_dev(g.node_levels), to_dev(g.node_keys), to_dev(g.node_coords),
        to_dev(g.node_centers), to_dev(g.parent_index), to_dev(g.child_ranges),
        to_dev(g.leaf_to_node))
end

# Rebuild the flat coefficient buffer on device rather than uploading it: the
# buffer carries its `basis_info` and the chi slab is 0x0 when LH is off, and
# `_ka_flat_buffer` is the ext's own allocator for exactly this shape.
function dev_flat(backend, buf::FM.FlatCoefficientBuffer{TF}) where TF
    b = ext._ka_flat_buffer(backend, TF, buf.basis_info, size(buf.phi, 2))
    copyto!(b.phi, buf.phi)
    size(b.chi, 2) > 0 && copyto!(b.chi, buf.chi)
    return b
end

function dev_state(hs::FM.DeviceResidentRadixState{TF,B,LH}, backend) where {TF,B,LH}
    g = dev_grid(hs.grid)
    src = to_dev(hs.source_bodies)
    mult = dev_flat(backend, hs.multipoles)
    locs = dev_flat(backend, hs.locals)
    # The workspace must be built through the SAME constructor `host_radix_state`
    # uses, off the concrete grid and list -- it is backend-generic, deciding
    # every array type from the `exemplar` buffer, so a device exemplar puts the
    # whole workspace on the backend. The cache-shaped `ka_radix_cache_workspace`
    # is NOT interchangeable here: it builds stage groups from capacity plus an
    # accepted-offset set rather than from this grid's node table, so its
    # m2m/l2l groups do not address the same nodes and M2M silently fills
    # nothing.
    ws = FM.ResidentOperatorWorkspace(TF, hs.invariant_cache.basis_info, mult,
        hs.grid, hs.interaction_list,
        hs.host_m2m_parent_routes, hs.host_m2m_child_routes,
        hs.host_l2l_parent_routes, hs.host_l2l_child_routes,
        hs.host_node_levels, hs.host_node_centers,
        hs.host_route_targets, hs.host_route_sources;
        m2l_strategy=FM.ConcatenatedFixedZM2L(),
        operator=hs.options.operator)
    return FM.DeviceResidentRadixState{TF,B,LH}(
        g, hs.interaction_list, src, src,
        to_dev(hs.body_perm), to_dev(hs.body_system_ids), to_dev(hs.body_indices),
        hs.host_body_perm, hs.host_body_system_ids, hs.host_body_indices,
        hs.host_cell_centers, hs.host_m2m_parent_routes, hs.host_m2m_child_routes,
        hs.host_l2l_parent_routes, hs.host_l2l_child_routes, hs.host_node_levels,
        hs.host_node_centers, hs.host_route_targets, hs.host_route_sources,
        to_dev(hs.cell_centers), to_dev(hs.cell_ranges),
        to_dev(hs.m2m_parent_routes), to_dev(hs.m2m_child_routes),
        to_dev(hs.l2l_parent_routes), to_dev(hs.l2l_child_routes),
        mult, locs,
        to_dev(hs.route_levels), to_dev(hs.route_offsets),
        to_dev(hs.route_targets), to_dev(hs.route_sources),
        to_dev(hs.direct_targets), to_dev(hs.direct_sources), to_dev(hs.output),
        hs.invariant_cache, ws, FM.CUDARadixTransferCounters(), hs.options, hs.counts,
    )
end

#------- cases -------#

# (P, ell, n) -- Float32 throughout, the precision Metal supports and the one
# the CUDA arm runs for FLOWVPM.
const CASES = [
    (2, 3,   64),
    (4, 3,  256),
    (4, 4, 1024),
    (6, 3,  256),
    (8, 4, 2048),
]

const TOL = 2e-4  # Float32 whole-lifecycle accumulation; per-stage suites run 1e-4/1e-5

npass = Ref(0); nfail = Ref(0)

for (ci, (P, ell, n)) in pairs(CASES)
    t_case = time()
    TF = Float32
    # Float32 all the way down: `generate_vortex` is Float64, which makes
    # `RadixGrid` Float64 and the grid upload throw on Metal even though
    # options.precision is Float32.
    Random.seed!(4400 + ci)
    system = VortexParticles(rand(TF, 3, n), (randn(TF, 3, n) ./ TF(n)),
        zeros(TF, n); potential=zeros(TF, 13, n),
        gradient_stretching=zeros(TF, 6, n))

    grid = FM.RadixGrid(system, ell)
    list = FM.build_radix_interaction_list(FM.LazyMaterializedBatches(1),
        FM.ParentNeighborM2L(), grid)
    opts = FM.CUDARadixLifecycleOptions(; precision=TF,
        m2l_strategy=FM.ConcatenatedFixedZM2L(), body_type=FM.Point{FM.Vortex})

    hs = FM.host_radix_state(system, grid, list, P, Val(true); options=opts)

    local ds
    try
        ds = dev_state(hs, DEV_BACKEND)
    catch err
        nfail[] += 1
        println("case $ci (P=$P, ell=$ell, n=$n): device state build FAILED: $err")
        continue
    end

    FM.run_host_radix_lifecycle!(hs)
    try
        ext.ka_lifecycle_body!(ds)
    catch err
        nfail[] += 1
        println("case $ci (P=$P, ell=$ell, n=$n): ka_lifecycle_body! THREW: $err")
        continue
    end

    e_mul = relerr(ds.multipoles.phi, hs.multipoles.phi)
    e_loc = relerr(ds.locals.phi, hs.locals.phi)
    e_out = relerr(ds.output, hs.output)
    ok = e_mul < TOL && e_loc < TOL && e_out < TOL
    ok ? (npass[] += 1) : (nfail[] += 1)
    println("[$(round(Int, time() - t_case))s] case $ci (P=$P, ell=$ell, n=$n): ", ok ? "PASS" : "FAIL",
        "  multipoles=", e_mul, "  locals=", e_loc, "  output=", e_out)
end

println("\nka_lifecycle_body! vs run_host_radix_lifecycle!: $(npass[])/$(length(CASES)) pass")
nfail[] == 0 || error("$(nfail[]) case(s) failed")
