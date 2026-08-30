# End-to-end tree/grid accuracy gate: device vs CPU, on real particle fields.
#
# The existing ka_grid_*_correctness.jl suites gate the four grid-rebuild stages
# INDIVIDUALLY, each against the matching host builder stage and each fed
# hand-built inputs. This gates the whole thing end to end: take a particle
# field, run it through both production front doors, and compare the trees that
# come out.
#
# TWO ARMS, and the distinction is the point:
#
#   ARM 1 -- device DeviceRadixGrid vs host DeviceRadixGrid.
#     Same algorithm, same code in src/, different backend. The host
#     RadixFMMCache builds a DeviceRadixGrid over plain Arrays, so this is a
#     field-for-field comparison of one struct. Every load-bearing field is an
#     INTEGER (Morton keys, perm, cell ranges, node tables), so the gate is
#     bit-equality, not a tolerance. Only cell_centers/node_centers are float,
#     and they are derived from integer coords by the same formula, so they get
#     a few-ulp allowance for FMA/association differences.
#
#     The failure this is really hunting: key generation quantizes a position
#     to a cell via floor((x - x_min)/dx). If the device contracts that
#     differently from the host, a body near a cell boundary lands in a
#     DIFFERENT CELL -- a discrete jump that silently rewrites every downstream
#     interaction list while every float in sight still looks fine. That is why
#     perm is compared elementwise WHEN the stable sortperm path is active.
#     On the production path (CUDA's bounded counting sort, now ported) the
#     scatter is unstable by design, so perm is compared as per-cell SETS
#     instead; ka_counting_sort_correctness.jl is the full gate for that path.
#
#   ARM 2 -- device DeviceRadixGrid vs the classic CPU threaded octree (Tree).
#     DIFFERENT ALGORITHMS: uniform depth-ell Morton grid vs adaptive
#     leaf_size subdivision with shrink-to-fit boxes. Different node counts,
#     different partitions, different centers. There is no elementwise
#     comparison to make here and expecting one would be a category error.
#     What IS checkable, and is checked: both are valid partitions of the SAME
#     body multiset, and every body lies geometrically inside the cell/box it
#     was assigned to. That catches a tree that is self-inconsistent, which is
#     the only cross-algorithm claim available.
include(joinpath(@__DIR__, "ka_backend.jl"))
using FastMultipole, Random, Printf, Test
using FastMultipole.StaticArrays
using KernelAbstractions
const FM = FastMultipole
include(joinpath(@__DIR__, "..", "vortex.jl"))

dev_functional() || (println("$(DEV_NAME) not functional; skipping"); exit(0))
ext = Base.get_extension(FastMultipole, :FastMultipoleKAExt)
ext !== nothing || error("FastMultipoleKAExt did not load")

#------- particle fields: the distributions that break tree builders ----------#
# uniform   -- the easy case, cells uniformly occupied
# clustered -- gaussian blobs; most cells empty, a few very deep -> exercises
#              the flag/scan/compact cell compression and the sparse node table
# planar    -- degenerate, all z equal; a whole axis collapses
# onboundary-- positions placed EXACTLY on cell boundaries, which is the only
#              way to actually trigger the quantization-rounding failure above
function make_field(kind::Symbol, seed, n, TF)
    Random.seed!(seed)
    pos = if kind === :uniform
        rand(TF, 3, n)
    elseif kind === :clustered
        nb = 8; c = rand(TF, 3, nb)
        # NOTE: written as a preallocated loop, not hcat(gen...). Splatting or
        # reduce(hcat, ...) over n columns folds n successively larger SMatrix
        # types -- one Julia specialization per body -- and costs minutes.
        p = Matrix{TF}(undef, 3, n)
        for i in 0:(n-1)
            p[:, i+1] .= c[:, 1 + (i % nb)] .+ TF(0.02) .* randn(TF, 3)
        end
        p
    elseif kind === :planar
        p = rand(TF, 3, n); p[3, :] .= TF(0.5); p
    elseif kind === :onboundary
        p = rand(TF, 3, n)
        # snap a third of the bodies onto exact multiples of 1/16
        for i in 1:3:n, d in 1:3
            p[d, i] = round(p[d, i] * TF(16)) / TF(16)
        end
        p
    else
        error("unknown field $kind")
    end
    return VortexParticles(pos, (randn(TF, 3, n) ./ TF(n)), zeros(TF, n);
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

function build_pair(sys_h, sys_d, P, ell, TF)
    opts = FM.CUDARadixLifecycleOptions(; precision=TF,
        m2l_strategy=FM.ConcatenatedFixedZM2L(), body_type=FM.Point{FM.Vortex})
    hcache = RadixFMMCache(sys_h; expansion_order=P, ell=ell,
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
    ext.ka_update_radix_state!(dcache, (sys_d,))
    KernelAbstractions.synchronize(DEV_BACKEND)
    return hcache, dcache
end

#------- ARM 1: device grid vs host grid, field for field --------------------#
nfail = Ref(0)
function cmp_int(label, h, d, name)
    ok = h == d
    ok || (nfail[] += 1;
        bad = findall(h .!= d);
        @printf("    FAIL %-14s %d/%d differ, first at %s\n",
            name, length(bad), length(h), string(first(bad))))
    return ok
end
function cmp_float(label, h, d, name; ulps=4)
    dif = abs.(Float64.(h) .- Float64.(d))
    scale = max.(abs.(Float64.(h)), 1e-30)
    rel = dif ./ scale
    worst = isempty(rel) ? 0.0 : maximum(rel)
    tol = ulps * eps(Float64(one(eltype(h))))
    ok = worst <= tol
    ok || (nfail[] += 1;
        @printf("    FAIL %-14s max rel %.3e > %.3e (%d ulp)\n", name, worst, tol, ulps))
    return (ok, worst)
end

function arm1(label, hcache, dcache)
    hg = hcache.state.grid
    dg = dcache.state.grid
    nb = hg.n_bodies; nc = hg.n_cells
    nn = hcache.level_offsets[end]
    @printf("  arm1 device-vs-host-radix: n_bodies=%d n_cells=%d n_nodes=%d\n", nb, nc, nn)
    dg.n_bodies == nb || (nfail[] += 1; @printf("    FAIL n_bodies %d vs %d\n", nb, dg.n_bodies))
    dg.n_cells  == nc || (nfail[] += 1; @printf("    FAIL n_cells %d vs %d\n", nc, dg.n_cells))
    A(x) = Array(x)
    # `perm`/`invperm` are elementwise-comparable ONLY on the stable sortperm
    # path. The production path now takes CUDA's bounded counting sort, whose
    # atomic-cursor scatter is unstable by design, so within-cell body order is
    # run-dependent and an elementwise comparison would fail on correct code.
    # There, the invariant claim is cell membership as SETS -- checked below and
    # gated in full by ka_counting_sort_correctness.jl.
    ctx = dcache.device_ctx
    unstable = ext.ka_counting_sort_ready(ctx.counting_histogram, hcache.ell)
    if unstable
        perm = A(dg.perm)[1:nb]
        sort(perm) == collect(1:nb) ||
            (nfail[] += 1; @printf("    FAIL perm is not a permutation of 1:%d\n", nb))
        # slot -> global body is `perm`; `body_index` is indexed BY the global
        # id (see _pack_radix_source_bodies!, translate_batched_resident.jl:2893),
        # so indexing it by a sorted slot is a category error.
        hs = [Set(hg.perm[hg.cell_ranges[1,c]:(hg.cell_ranges[1,c]+hg.cell_ranges[2,c]-1)])
              for c in 1:nc]
        dcr = A(dg.cell_ranges)[:, 1:nc]; dbi = A(dg.perm)[1:nb]
        ds = [Set(dbi[dcr[1,c]:(dcr[1,c]+dcr[2,c]-1)]) for c in 1:nc]
        nbad = count(i -> hs[i] != ds[i], 1:nc)
        nbad == 0 || (nfail[] += 1;
            @printf("    FAIL %d/%d cells have different body SETS than host\n", nbad, nc))
        @printf("    perm: counting-sort path (unstable) -- compared as cell sets, %d/%d cells match\n",
            nc - nbad, nc)
    else
        cmp_int(label, hg.perm[1:nb],    A(dg.perm)[1:nb],    "perm")
        cmp_int(label, hg.invperm[1:nb], A(dg.invperm)[1:nb], "invperm")
    end
    cmp_int(label, hg.body_system[1:nb],  A(dg.body_system)[1:nb],  "body_system")
    cmp_int(label, hg.body_index[1:nb],   A(dg.body_index)[1:nb],   "body_index")
    cmp_int(label, hg.cell_keys[1:nc],    A(dg.cell_keys)[1:nc],    "cell_keys")
    cmp_int(label, hg.cell_ranges[:,1:nc],A(dg.cell_ranges)[:,1:nc],"cell_ranges")
    cmp_int(label, hg.leaf_to_node[1:nc], A(dg.leaf_to_node)[1:nc], "leaf_to_node")
    cmp_int(label, hg.node_levels[1:nn],  A(dg.node_levels)[1:nn],  "node_levels")
    cmp_int(label, hg.node_keys[1:nn],    A(dg.node_keys)[1:nn],    "node_keys")
    cmp_int(label, hg.node_coords[:,1:nn],A(dg.node_coords)[:,1:nn],"node_coords")
    cmp_int(label, hg.parent_index[1:nn], A(dg.parent_index)[1:nn], "parent_index")
    cmp_int(label, hg.child_ranges[:,1:nn],A(dg.child_ranges)[:,1:nn],"child_ranges")
    _, wc = cmp_float(label, hg.cell_centers[:,1:nc], A(dg.cell_centers)[:,1:nc], "cell_centers")
    _, wn = cmp_float(label, hg.node_centers[:,1:nn], A(dg.node_centers)[:,1:nn], "node_centers")
    @printf("    centers: worst rel cell %.2e node %.2e\n", wc, wn)
    return nothing
end

#------- ARM 2: structural validity + containment, both trees ----------------#
function arm2(label, sys, hcache, dcache, ell, TF)
    hg = hcache.state.grid
    dg = dcache.state.grid
    nb = hg.n_bodies; nc = hg.n_cells
    # see the NOTE in make_field: reduce(hcat, ...) over a generator of
    # SVector{3} is a compile-time explosion, not an allocation nuisance.
    nbod = length(sys.bodies)
    pos = Matrix{TF}(undef, 3, nbod)
    for i in 1:nbod
        q = sys.bodies[i].position
        pos[1, i] = q[1]; pos[2, i] = q[2]; pos[3, i] = q[3]
    end

    # (a) radix perm is a genuine permutation
    perm = Array(dg.perm)[1:nb]
    sort(perm) == collect(1:nb) ||
        (nfail[] += 1; @printf("    FAIL radix perm is not a permutation of 1:%d\n", nb))

    # (b) cell_ranges partition the sorted slots exactly once, contiguously
    cr = Array(dg.cell_ranges)[:, 1:nc]
    covered = zeros(Int, nb)
    for c in 1:nc
        lo, cnt = cr[1, c], cr[2, c]
        for k in lo:(lo + cnt - 1)
            (1 <= k <= nb) || (nfail[] += 1; @printf("    FAIL cell %d slot %d out of range\n", c, k); break)
            covered[k] += 1
        end
    end
    all(covered .== 1) ||
        (nfail[] += 1; @printf("    FAIL cell_ranges cover: %d slots uncovered, %d doubled\n",
            count(covered .== 0), count(covered .> 1)))

    # (c) every body lies inside the cell it was assigned to
    dx = 2 * hg.h0 / (1 << ell)
    cc = Array(dg.cell_centers)[:, 1:nc]
    # slot -> global body id (see the note in arm1)
    bi = Array(dg.perm)[1:nb]
    worst = 0.0
    for c in 1:nc, k in cr[1, c]:(cr[1, c] + cr[2, c] - 1)
        p = pos[:, bi[k]]
        for d in 1:3
            worst = max(worst, abs(Float64(p[d]) - Float64(cc[d, c])) / (dx / 2))
        end
    end
    ok_contain = worst <= 1.0 + 1e-5
    if !ok_contain
        nfail[] += 1
        # diagnose: is the assumed cell width wrong, or is a body really in the
        # wrong cell? Print the worst offender and the grid's own geometry.
        bc, bk, bw = 0, 0, 0.0
        for c in 1:nc, k in cr[1, c]:(cr[1, c] + cr[2, c] - 1)
            q = pos[:, bi[k]]
            for d in 1:3
                w = abs(Float64(q[d]) - Float64(cc[d, c])) / (dx / 2)
                w > bw && ((bc, bk, bw) = (c, k, w))
            end
        end
        q = pos[:, bi[bk]]
        @printf("    diag: h0=%.6g ell=%d dx=%.6g ell_axes=%s box_extent=%s x_min=%s\n",
            Float64(hg.h0), ell, dx, string(hcache.ell_axes),
            string(hcache.box_extent), string(hcache.x_min))
        @printf("    diag: worst cell %d slot %d body %d  pos=%s center=%s  key=%d\n",
            bc, bk, bi[bk], string(Float64.(q)), string(Float64.(cc[:, bc])),
            Int(Array(dg.cell_keys)[bc]))
    end
    @printf("  arm2 radix: perm ok, partition ok, containment worst=%.6f of half-cell %s\n",
        worst, ok_contain ? "PASS" : "FAIL")

    # (d) classic CPU threaded octree: same body multiset, valid partition,
    #     bodies inside their leaf boxes
    switches = FM.DerivativesSwitch(true, true, false, (sys,))
    tree = FM.Tree((sys,), false, switches; expansion_order=4, leaf_size=SVector{1,Int}(64))
    sidx = tree.sort_index_list[1]
    sort(copy(sidx)) == collect(1:nb) ||
        (nfail[] += 1; @printf("    FAIL octree sort_index_list is not a permutation\n"))
    leafpop = Int[]
    wbox = 0.0
    for il in tree.leaf_index
        b = tree.branches[il]
        rng = b.bodies_index[1]
        push!(leafpop, length(rng))
        for k in rng
            p = pos[:, sidx[k]]
            for d in 1:3
                wbox = max(wbox, (abs(Float64(p[d]) - Float64(b.center[d]))) /
                    max(Float64(b.box[d]), 1e-30))
            end
        end
    end
    sum(leafpop) == nb ||
        (nfail[] += 1; @printf("    FAIL octree leaves cover %d of %d bodies\n", sum(leafpop), nb))
    ok_box = wbox <= 1.0 + 1e-4
    ok_box || (nfail[] += 1)
    radixpop = cr[2, 1:nc]
    @printf("  arm2 octree: %d leaves, pop min/med/max %d/%d/%d, containment worst=%.6f of box %s\n",
        length(leafpop), minimum(leafpop), sort(leafpop)[(end+1)÷2], maximum(leafpop),
        wbox, ok_box ? "PASS" : "FAIL")
    @printf("  arm2 radix : %d cells,  pop min/med/max %d/%d/%d  (uniform grid, not comparable elementwise)\n",
        nc, minimum(radixpop), sort(radixpop)[(end+1)÷2], maximum(radixpop))
    return nothing
end

#------- cases ---------------------------------------------------------------#
const CASES = [
    (:uniform,     512, 2),
    (:uniform,    1024, 3),
    (:clustered,  1024, 3),
    (:planar,     1024, 3),
    (:onboundary, 1024, 3),
    (:uniform,    4096, 4),
    (:onboundary, 4096, 4),
]

println("device = $DEV_NAME  threads = $(Threads.nthreads())\n")
for (kind, n, ell) in CASES
    TF = Float32
    @printf("case %s n=%d ell=%d\n", kind, n, ell); flush(stdout)
    sys_h = make_field(kind, 4242, n, TF)
    sys_d = make_field(kind, 4242, n, TF)
    sys_r = make_field(kind, 4242, n, TF)
    before = nfail[]
    hcache, dcache = build_pair(sys_h, sys_d, 4, ell, TF)
    arm1(kind, hcache, dcache)
    arm2(kind, sys_r, hcache, dcache, ell, TF)
    @printf("  -> %s\n\n", nfail[] == before ? "PASS" : "FAIL ($(nfail[] - before) checks)")
end
@printf("%s: %d failing checks over %d cases\n",
    nfail[] == 0 ? "all cases PASS" : "FAILURES", nfail[], length(CASES))
