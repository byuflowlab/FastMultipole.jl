# Correctness gate for the KA bounded-key counting sort -- the port of CUDA's
# `_cuda_counting_sort_into!` fast path (src/translate_batched_cuda.jl:223).
#
# This path is UNSTABLE BY DESIGN, matching CUDA: the scatter claims slots with
# an atomic cursor, so bodies sharing a cell come out in a run-dependent order.
# That makes the usual gate -- elementwise `perm` against the stable host sort --
# the wrong assertion. It would fail on correct code.
#
# What is actually invariant, and what this checks:
#   1. `perm` is a genuine permutation of 1:n (nothing lost, nothing doubled).
#   2. `sorted_keys` is nondecreasing and is the same MULTISET as the input keys.
#   3. `sorted_keys[j] == keys[perm[j]]` -- perm and the keys agree.
#   4. Cell membership matches the host exactly: for every cell, the SET of
#      bodies in it is identical to the host's, even though the order within it
#      is not. This is the real claim, because it is what everything downstream
#      consumes.
#   5. `cell_keys`, `cell_ranges`, and the whole node table (`node_levels`,
#      `node_keys`, `node_coords`, `parent_index`, `child_ranges`,
#      `leaf_to_node`) match the host BIT FOR BIT -- they are pure functions of
#      the occupied-cell set, so instability upstream must not reach them.
#   6. Re-running the same input reproduces (4) and (5) exactly, while (1)-(3)
#      stay valid -- i.e. the instability is confined to within-cell order.
include(joinpath(@__DIR__, "ka_backend.jl"))
using FastMultipole, Random, Printf, Test
using FastMultipole.StaticArrays
using KernelAbstractions
const FM = FastMultipole
include(joinpath(@__DIR__, "..", "vortex.jl"))

dev_functional() || (println("$(DEV_NAME) not functional; skipping"); exit(0))
ext = Base.get_extension(FastMultipole, :FastMultipoleKAExt)
ext !== nothing || error("FastMultipoleKAExt did not load")

make_field(seed, n, TF; clustered=false) = begin
    Random.seed!(seed)
    pos = clustered ?
        hcat((rand(TF, 3) .* TF(0.05) .+ (i % 8) / TF(8) for i in 0:(n-1))...) :
        rand(TF, 3, n)
    VortexParticles(pos, (randn(TF, 3, n) ./ TF(n)), zeros(TF, n);
        potential=zeros(TF, 13, n), gradient_stretching=zeros(TF, 6, n))
end

function device_build_args(hcache)
    sp = hcache.policy; ell = hcache.ell
    tables, level_class_of, level_radii2, root_level, first_m2l_level =
        FM._hierarchical_scheduled_tables(sp, ell, hcache.ell_axes)
    class_level, class_offset, _ =
        FM._hierarchical_class_metadata(tables, ell, first_m2l_level)
    max_level_nodes = ell >= 2 ? maximum(
        (FM._radix_level_node_capacity(L, hcache.ell_axes, ell, hcache.max_cells)
         for L in first_m2l_level:ell); init=0) : 0
    return (; tables, level_class_of, level_radii2, root_level, first_m2l_level,
        class_level, class_offset, max_level_nodes)
end

nfail = Ref(0)
fail(msg) = (nfail[] += 1; println("    FAIL ", msg))

# body identity for sorted slot k, as a set per cell
cellsets(grid, nc, nb) = begin
    cr = Array(grid.cell_ranges)[:, 1:nc]
    bidx = Array(grid.body_index)[1:nb]
    [Set(bidx[cr[1, c]:(cr[1, c] + cr[2, c] - 1)]) for c in 1:nc]
end

function check_case(n, ell, TF; clustered=false)
    @printf("case n=%d ell=%d %s\n", n, ell, clustered ? "clustered" : "uniform")
    sys_h = make_field(77, n, TF; clustered)
    sys_d = make_field(77, n, TF; clustered)
    opts = FM.CUDARadixLifecycleOptions(; precision=TF,
        m2l_strategy=FM.ConcatenatedFixedZM2L(), body_type=FM.Point{FM.Vortex})
    hcache = RadixFMMCache(sys_h; expansion_order=4, ell=ell,
        window_classes=256, options=opts)
    fmm!(sys_h, hcache)
    a = device_build_args(hcache); LH = typeof(hcache).parameters[2]
    dcache = ext.ka_radix_cache_device_build(DEV_BACKEND, (sys_d,),
        hcache.expansion_order, ell, hcache.x_min, hcache.h0, hcache.max_n_bodies,
        hcache.options, hcache.policy, hcache.accepted_offsets,
        hcache.rejected_offsets, hcache.max_cells, hcache.max_nodes,
        hcache.route_capacity, hcache.direct_capacity,
        hcache.state.multipoles.basis_info, Val(LH);
        hierarchical_tables=a.tables, class_level=a.class_level,
        class_offset=a.class_offset, hierarchical_level_class_of=a.level_class_of,
        hierarchical_level_radii2=a.level_radii2,
        max_level_nodes=a.max_level_nodes, hessian=hcache.hessian,
        ell_axes=hcache.ell_axes, box_extent=hcache.box_extent,
        root_level=a.root_level, first_m2l_level=a.first_m2l_level)

    ctx = dcache.device_ctx
    want = ext.ka_counting_sort_enabled(ell) ? 1 << (3 * ell) : 1
    length(ctx.counting_histogram) == want ||
        fail("histogram sized $(length(ctx.counting_histogram)), expected $want")
    ext.ka_counting_sort_ready(ctx.counting_histogram, ell) ||
        fail("counting sort gate is OFF at ell=$ell -- this case tests nothing")

    hg = hcache.state.grid
    nb = hg.n_bodies; nc = hg.n_cells; nn = hcache.level_offsets[end]
    hsets = cellsets(hg, nc, nb)

    prev = nothing
    for rep in 1:3
        ext.ka_update_radix_state!(dcache, (sys_d,))
        KernelAbstractions.synchronize(DEV_BACKEND)
        dg = dcache.state.grid
        A(x) = Array(x)

        # 1-3: the sort itself
        perm = A(dg.perm)[1:nb]
        sort(perm) == collect(1:nb) || fail("rep$rep perm is not a permutation of 1:$nb")
        skeys = A(ctx.sorted_keys)[1:nb]
        issorted(skeys) || fail("rep$rep sorted_keys not nondecreasing")
        rawk = A(ctx.keys)[1:nb]
        sort(skeys) == sort(rawk) || fail("rep$rep sorted_keys is a different multiset")
        all(skeys[j] == rawk[perm[j]] for j in 1:nb) ||
            fail("rep$rep sorted_keys[j] != keys[perm[j]]")

        # 4: cell membership matches the host as SETS
        dg.n_cells == nc || fail("rep$rep n_cells $(dg.n_cells) vs host $nc")
        if dg.n_cells == nc
            dsets = cellsets(dg, nc, nb)
            nbad = count(i -> dsets[i] != hsets[i], 1:nc)
            nbad == 0 || fail("rep$rep $nbad/$nc cells have different body SETS than host")
        end

        # 5: everything downstream bit-exact against the host
        for (name, h, d) in (
            ("cell_keys",    hg.cell_keys[1:nc],       A(dg.cell_keys)[1:nc]),
            ("cell_ranges",  hg.cell_ranges[:, 1:nc],  A(dg.cell_ranges)[:, 1:nc]),
            ("leaf_to_node", hg.leaf_to_node[1:nc],    A(dg.leaf_to_node)[1:nc]),
            ("node_levels",  hg.node_levels[1:nn],     A(dg.node_levels)[1:nn]),
            ("node_keys",    hg.node_keys[1:nn],       A(dg.node_keys)[1:nn]),
            ("node_coords",  hg.node_coords[:, 1:nn],  A(dg.node_coords)[:, 1:nn]),
            ("parent_index", hg.parent_index[1:nn],    A(dg.parent_index)[1:nn]),
            ("child_ranges", hg.child_ranges[:, 1:nn], A(dg.child_ranges)[:, 1:nn]),
        )
            h == d || fail("rep$rep $name differs from host ($(count(h .!= d)) entries)")
        end

        # 6: instability is confined to within-cell order
        if prev !== nothing
            same = prev == perm
            rep == 2 && @printf("    perm identical across runs: %s%s\n", same,
                same ? " (no same-cell ties exercised, or sort was stable)" : " (expected: unstable)")
        end
        prev = perm
    end
    return nothing
end

println("device = $DEV_NAME\n")
# n=8192 ell=4 uniform was dropped: same branch as the 4096 case, 10 s for no new coverage
for (n, ell, cl) in ((1024, 3, false), (4096, 4, false), (4096, 3, true))
    before = nfail[]
    t_case = time()
    check_case(n, ell, Float32; clustered=cl)
    @printf("  -> %s [%.0fs]\n\n", nfail[] == before ? "PASS" : "FAIL", time() - t_case)
end
println(nfail[] == 0 ? "gate passed: all counting-sort cases PASS" :
    "FAILURES: $(nfail[]) failing checks")
