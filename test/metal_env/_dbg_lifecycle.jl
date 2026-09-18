# scratch: isolate which stage of ka_lifecycle_body! diverges
include("ka_backend.jl")
using FastMultipole, Random, Test
using FastMultipole.StaticArrays
const FM = FastMultipole
include(joinpath(@__DIR__, "..", "vortex.jl"))
ext = Base.get_extension(FastMultipole, :FastMultipoleKAExt)

relerr(a, b) = (d = maximum(abs.(Array(a) .- Array(b))); s = maximum(abs.(Array(b)));
                s == 0 ? d : d / s)
to_dev(x::AbstractArray) = devarray(x)
to_dev(x) = x

function dev_grid(g::FM.DeviceRadixGrid)
    FM.DeviceRadixGrid(g.x_min, g.h0, g.ell, g.n_bodies, g.n_cells,
        to_dev(g.perm), to_dev(g.invperm), to_dev(g.cell_keys), to_dev(g.cell_ranges),
        to_dev(g.body_system), to_dev(g.body_index), to_dev(g.cell_centers),
        to_dev(g.node_levels), to_dev(g.node_keys), to_dev(g.node_coords),
        to_dev(g.node_centers), to_dev(g.parent_index), to_dev(g.child_ranges),
        to_dev(g.leaf_to_node))
end
function dev_flat(backend, buf::FM.FlatCoefficientBuffer{TF}) where TF
    b = ext._ka_flat_buffer(backend, TF, buf.basis_info, size(buf.phi, 2))
    copyto!(b.phi, buf.phi)
    size(b.chi, 2) > 0 && copyto!(b.chi, buf.chi)
    b
end
function dev_state(hs::FM.DeviceResidentRadixState{TF,B,LH}, backend, acc, ell, h0,
        mc, mn, rc) where {TF,B,LH}
    mult = dev_flat(backend, hs.multipoles)
    ws = FM.ResidentOperatorWorkspace(TF, hs.invariant_cache.basis_info, mult,
        hs.grid, hs.interaction_list,
        hs.host_m2m_parent_routes, hs.host_m2m_child_routes,
        hs.host_l2l_parent_routes, hs.host_l2l_child_routes,
        hs.host_node_levels, hs.host_node_centers,
        hs.host_route_targets, hs.host_route_sources;
        m2l_strategy=FM.ConcatenatedFixedZM2L(), operator=hs.options.operator)
    src = to_dev(hs.source_bodies)
    FM.DeviceResidentRadixState{TF,B,LH}(
        dev_grid(hs.grid), hs.interaction_list, src, src,
        to_dev(hs.body_perm), to_dev(hs.body_system_ids), to_dev(hs.body_indices),
        hs.host_body_perm, hs.host_body_system_ids, hs.host_body_indices,
        hs.host_cell_centers, hs.host_m2m_parent_routes, hs.host_m2m_child_routes,
        hs.host_l2l_parent_routes, hs.host_l2l_child_routes, hs.host_node_levels,
        hs.host_node_centers, hs.host_route_targets, hs.host_route_sources,
        to_dev(hs.cell_centers), to_dev(hs.cell_ranges),
        to_dev(hs.m2m_parent_routes), to_dev(hs.m2m_child_routes),
        to_dev(hs.l2l_parent_routes), to_dev(hs.l2l_child_routes),
        mult, dev_flat(backend, hs.locals),
        to_dev(hs.route_levels), to_dev(hs.route_offsets),
        to_dev(hs.route_targets), to_dev(hs.route_sources),
        to_dev(hs.direct_targets), to_dev(hs.direct_sources), to_dev(hs.output),
        hs.invariant_cache, ws, FM.CUDARadixTransferCounters(), hs.options, hs.counts)
end

TF = Float32; P = 4; ell = 3; n = 256
Random.seed!(4402)
system = VortexParticles(rand(TF, 3, n), randn(TF, 3, n) ./ TF(n), zeros(TF, n);
    potential=zeros(TF, 13, n), gradient_stretching=zeros(TF, 6, n))
grid = FM.RadixGrid(system, ell)
list = FM.build_radix_interaction_list(FM.LazyMaterializedBatches(1),
    FM.ParentNeighborM2L(), grid)
opts = FM.CUDARadixLifecycleOptions(; precision=TF,
    m2l_strategy=FM.ConcatenatedFixedZM2L(), body_type=FM.Point{FM.Vortex})
hs = FM.host_radix_state(system, grid, list, P, Val(true); options=opts)
acc = unique(SVector{3,Int}[b.offset for b in list.m2l_batches])
g = hs.grid
ds = dev_state(hs, DEV_BACKEND, acc, g.ell, g.h0, g.n_cells, length(g.node_keys),
    max(length(hs.route_targets), 1))

println("counts.n_cells=", ds.counts.n_cells, " n_bodies=", ds.counts.n_bodies,
        " n_routes=", ds.counts.n_routes, " n_direct=", ds.counts.n_direct)
println("grid.n_cells=", g.n_cells, " nodes=", length(g.node_keys))
println("host source_bodies rows=", size(hs.source_bodies, 1))

FM.run_host_radix_lifecycle!(hs)
try
    ext.ka_lifecycle_body!(ds)
    println("multipoles=", relerr(ds.multipoles.phi, hs.multipoles.phi))
    println("locals=", relerr(ds.locals.phi, hs.locals.phi))
    println("output=", relerr(ds.output, hs.output))
catch err
    for l in first(stacktrace(catch_backtrace()), 14)
        println("  ", l)
    end
    rethrow()
end
