# Timing instrument for the whole uniform radix lifecycle (`ka_lifecycle_body!`).
#
# Why this exists: `ka_m2m_benchmark.jl` drives ONE resident stage group per
# rep, so it cannot see the cost of a per-group barrier -- the barrier lands
# next to the harness boundary either way. The lifecycle runs every M2M group
# and every L2L group back to back, which is the only place a per-group
# `KA.synchronize` actually serialises anything. Session 6 removed those two
# barriers from `ka_resident_stage_group_apply!` and
# `ka_resident_m2l_concat_apply!`; this file is what measures that, and what
# any future launch/sync work should be measured against.
#
# Production semantics: each rep calls `ka_lifecycle_body!(ds)` with its
# default `sync=true`, so the reported time is one complete, host-visible
# lifecycle -- not a pipelined lower bound.

include("ka_backend.jl")
using FastMultipole, Random

# one host-visible barrier; `ka_backend.jl` deliberately exposes no sync helper
dev_synchronize() = KernelAbstractions.synchronize(DEV_BACKEND)
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
#
# Deeper trees than the correctness suite: group count (hence barrier count)
# grows with `ell`, and that is the axis under test.
const CASES = [
    (4, 3,   1024),
    (4, 4,   8192),
    (6, 4,   8192),
    (6, 5,  32768),
    (8, 5,  32768),
]

const NREPS  = parse(Int, get(ENV, "KA_BENCH_NREPS",  "200"))
const NWARM  = parse(Int, get(ENV, "KA_BENCH_NWARMUP", "20"))

println("lifecycle benchmark on $(DEV_NAME): nreps=$NREPS warmup=$NWARM")
println(rpad("P",4), rpad("ell",5), rpad("n",8), rpad("m2m_grp",9),
        rpad("l2l_grp",9), rpad("ms/step",11))

for (ci, (P, ell, n)) in pairs(CASES)
    TF = Float32
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
        println("case $ci (P=$P, ell=$ell, n=$n): device state build FAILED: $err")
        continue
    end

    ws = ds.scratch
    n_m2m = length(ws.m2m_groups); n_l2l = length(ws.l2l_groups)

    try
        for _ in 1:NWARM; ext.ka_lifecycle_body!(ds); end
        dev_synchronize()
        t = @elapsed begin
            for _ in 1:NREPS; ext.ka_lifecycle_body!(ds); end
            dev_synchronize()
        end
        ms = 1000 * t / NREPS
        println(rpad(P,4), rpad(ell,5), rpad(n,8), rpad(n_m2m,9), rpad(n_l2l,9),
                rpad(round(ms; digits=4), 11))
    catch err
        println("case $ci (P=$P, ell=$ell, n=$n): THREW: $err")
    end
end
