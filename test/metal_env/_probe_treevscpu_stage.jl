# Stage-timed reproduction of ka_tree_vs_cpu_correctness.jl CASE 1 ONLY.
# Every stage prints with an explicit flush so a stall is attributable.
include(joinpath(@__DIR__, "ka_backend.jl"))
using FastMultipole, Random, Printf, Test
using FastMultipole.StaticArrays
using KernelAbstractions
const FM = FastMultipole
include(joinpath(@__DIR__, "..", "vortex.jl"))

t0 = time()
mark(s) = (@printf("[%7.1fs] %s\n", time() - t0, s); flush(stdout))

mark("loaded packages")
dev_functional() || (println("no device"); exit(0))
ext = Base.get_extension(FastMultipole, :FastMultipoleKAExt)
mark("extension = $(ext !== nothing)")

n = length(ARGS)>0 ? parse(Int,ARGS[1]) : 512
ell = length(ARGS)>1 ? parse(Int,ARGS[2]) : 2
TF = Float32; P = 4
mark("PARAMS n=$n ell=$ell")
mkfield() = (Random.seed!(4242); pos = rand(TF, 3, n);
    VortexParticles(pos, (randn(TF, 3, n) ./ TF(n)), zeros(TF, n);
        potential=zeros(TF, 13, n), gradient_stretching=zeros(TF, 6, n)))
sys_h = mkfield(); sys_d = mkfield(); sys_r = mkfield()
mark("fields built")

opts = FM.CUDARadixLifecycleOptions(; precision=TF,
    m2l_strategy=FM.ConcatenatedFixedZM2L(), body_type=FM.Point{FM.Vortex})
hcache = RadixFMMCache(sys_h; expansion_order=P, ell=ell, window_classes=256, options=opts)
mark("RadixFMMCache built")
fmm!(sys_h, hcache)
mark("host fmm! done")

sp = hcache.policy
tables, level_class_of, level_radii2, root_level, first_m2l_level =
    FM._hierarchical_scheduled_tables(sp, ell, hcache.ell_axes)
class_level, class_offset, _ = FM._hierarchical_class_metadata(tables, ell, first_m2l_level)
max_level_nodes = ell >= 2 ? maximum(
    (FM._radix_level_node_capacity(L, hcache.ell_axes, ell, hcache.max_cells)
     for L in first_m2l_level:ell); init=0) : 0
mark("device_build_args: root_level=$root_level first_m2l_level=$first_m2l_level max_level_nodes=$max_level_nodes")

LH = typeof(hcache).parameters[2]
dcache = ext.ka_radix_cache_device_build(DEV_BACKEND, (sys_d,),
    hcache.expansion_order, ell, hcache.x_min, hcache.h0, hcache.max_n_bodies,
    hcache.options, hcache.policy, hcache.accepted_offsets,
    hcache.rejected_offsets, hcache.max_cells, hcache.max_nodes,
    hcache.route_capacity, hcache.direct_capacity,
    hcache.state.multipoles.basis_info, Val(LH);
    hierarchical_tables=tables, class_level=class_level,
    class_offset=class_offset, hierarchical_level_class_of=level_class_of,
    hierarchical_level_radii2=level_radii2,
    max_level_nodes=max_level_nodes, hessian=hcache.hessian,
    ell_axes=hcache.ell_axes, box_extent=hcache.box_extent,
    root_level=root_level, first_m2l_level=first_m2l_level)
mark("device cache built")

ext.ka_update_radix_state!(dcache, (sys_d,))
KernelAbstractions.synchronize(DEV_BACKEND)
mark("ka_update_radix_state! done")

hg = hcache.state.grid; dg = dcache.state.grid
nb = hg.n_bodies; nc = hg.n_cells; nn = hcache.level_offsets[end]
mark("grids: nb=$nb nc=$nc nn=$nn")
ctx = dcache.device_ctx
mark("counting_sort_ready = $(ext.ka_counting_sort_ready(ctx.counting_histogram, ell))")
Array(dg.perm); Array(dg.cell_ranges); Array(dg.node_keys)
mark("arm1 device copies OK")

# --- the suspect: classic CPU octree ---
switches = FM.DerivativesSwitch(true, true, false, (sys_r,))
mark("switches built; entering FM.Tree ...")
tr = FM.Tree((sys_r,), false, switches; expansion_order=4, leaf_size=SVector{1,Int}(64))
mark("FM.Tree built: $(length(tr.branches)) branches, $(length(tr.leaf_index)) leaves")
mark("DONE")
