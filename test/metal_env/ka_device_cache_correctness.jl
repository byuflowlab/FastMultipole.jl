# End-to-end gate for `ka_radix_cache_device_build` + `ka_radix_cache_device_step!`
# -- a whole `RadixFMMCache` built and stepped on a KA backend, compared against FastMultipole's OWN host cache running `fmm!`.
#
# Why this suite exists. The per-stage suites gate each part of the device
# lifecycle (grid rebuild stages 1-4, hierarchical occupancy/direct pairs/
# windows, the lifecycle body, finalize) against its own host oracle in
# isolation. The `ka_update_radix_state!` / `ka_radix_cache_device_step!`
# drivers have no entry point other than a device-resident `RadixFMMCache`, so
# this suite builds one and runs a real UJ through it.
#
# Oracle: a second `RadixFMMCache` over an identical system with device=false
# and the SAME stencil policy, stepped with `fmm!`. That is the host resident
# lifecycle -- a genuinely independent implementation of every stage, not a
# transcription -- so an elementwise match on the scattered velocity/potential
# is the end-to-end acceptance check behind the per-stage suites.
#
# Both caches are hierarchical (`window_classes` set), which is the policy a
# vortex particle method builds and the only one the KA step implements.
#
# Body type `Point{Vortex}` with Lamb-Helmholtz on: the configuration a vortex
# particle method runs.
include("ka_backend.jl")
using FastMultipole, Random, Test, Printf
using FastMultipole.StaticArrays

const FM = FastMultipole
include(joinpath(@__DIR__, "..", "vortex.jl"))

if !dev_functional()
    println("$(DEV_NAME) not functional; skipping")
    exit(0)
end
ext = Base.get_extension(FastMultipole, :FastMultipoleKAExt)
ext !== nothing || error("FastMultipoleKAExt did not load")

relerr(a, b) = (d = maximum(abs.(Array(a) .- Array(b))); s = maximum(abs.(Array(b)));
                s == 0 ? d : d / s)

make_system(seed, n, TF) = (Random.seed!(seed);
    VortexParticles(rand(TF, 3, n), (randn(TF, 3, n) ./ TF(n)), zeros(TF, n);
        potential=zeros(TF, 13, n), gradient_stretching=zeros(TF, 6, n)))

# Re-derive the construction arguments the host constructor computes between its
# policy selection and its `device` branch (src/resident/radix_cache.jl). Everything else comes off the host cache's own fields, so the two
# caches are guaranteed to be built for the same geometry, capacities and
# stencil -- a divergence here would make the comparison meaningless rather
# than merely failing.
function device_build_args(hcache)
    sp = hcache.policy
    sp isa FM.HierarchicalRigidStencil ||
        error("expected a hierarchical host cache; got $(typeof(sp))")
    ell = hcache.ell
    tables, level_class_of, _, root_level, first_m2l_level =
        FM._hierarchical_scheduled_tables(sp, ell, hcache.ell_axes)
    return (; tables, level_class_of, root_level, first_m2l_level)
end

# (P, ell, n, window_classes)
# Case list kept deliberately narrow in DISTINCT expansion orders, not in cases.
# The hierarchical M2L kernels specialize on P, and a first Metal compile of that
# stage measured 82 s (83% compilation) against ~1 s for every other stage
# combined -- so each new P costs more than every other axis put together. Three
# P=4 cases sweep ell, n and window_classes for free on one compile; the single
# P=6 case is what proves the sweep is not P-specific.
# A unit run takes the short case list; FM_FULL_SWEEP=1 takes the full one.
# The sweep is a robustness study, not a check: it belongs in a debugging pass,
# not in every run.
const CASES_FULL = [
    (4, 3,  256, 8),
    (4, 4, 1024, 8),
    (4, 4, 4096, 8),
    (4, 4, 4096, 256),
    (6, 4, 1024, 4),
]
const CASES_SHORT = [
    (4, 3, 256, 8),
    (4, 4, 512, 64),   # deeper tree and wider window classes
]
const CASES = haskey(ENV, "FM_FULL_SWEEP") ? CASES_FULL : CASES_SHORT

const TOL = 3e-4   # Float32 whole-lifecycle accumulation, as in ka_lifecycle_body

npass = Ref(0); nfail = Ref(0)

for (ci, (P, ell, n, wc)) in pairs(CASES)
    TF = Float32
    @printf("case %d (P=%d, ell=%d, n=%d, wc=%d): start\n", ci, P, ell, n, wc); flush(stdout)
    t_case = time()
    sys_h = make_system(5100 + ci, n, TF)
    sys_d = make_system(5100 + ci, n, TF)

    opts = FM.RadixLifecycleOptions(; precision=TF,
        m2l_strategy=FM.ConcatenatedFixedZM2L(), body_type=FM.Point{FM.Vortex})
    hcache = RadixFMMCache(sys_h; expansion_order=P, ell=ell, window_classes=wc,
        options=opts)
    fmm!(sys_h, hcache)

    a = device_build_args(hcache)
    LH = typeof(hcache).parameters[2]
    basis_info = hcache.state.multipoles.basis_info

    local dcache
    try
        dcache = ext.ka_radix_cache_device_build(DEV_BACKEND, (sys_d,),
            hcache.expansion_order, ell, hcache.x_min, hcache.h0,
            hcache.max_n_bodies, hcache.options, hcache.policy,
            hcache.accepted_offsets, hcache.rejected_offsets,
            hcache.max_cells, hcache.max_nodes, hcache.route_capacity,
            hcache.direct_capacity, basis_info, Val(LH);
            hierarchical_tables=a.tables,
            hierarchical_level_class_of=a.level_class_of, hessian=hcache.hessian,
            ell_axes=hcache.ell_axes, box_extent=hcache.box_extent,
            root_level=a.root_level, first_m2l_level=a.first_m2l_level)
    catch err
        nfail[] += 1
        println("case $ci (P=$P, ell=$ell, n=$n, wc=$wc): device cache build FAILED: ",
            sprint(showerror, err))
        flush(stdout); continue
    end

    # the device cache must be a real one, not a host cache in disguise
    @assert dcache.device && dcache.built
    @assert dcache.max_cells == hcache.max_cells
    @assert dcache.max_nodes == hcache.max_nodes
    @assert !(dcache.device_ctx.grid.cell_keys isa Array)

    switches = FM.DerivativesSwitch(FM.to_vector(false, 1), FM.to_vector(true, 1),
        FM.to_vector(false, 1), (sys_d,))
    try
        ext.ka_radix_cache_device_step!(dcache, (sys_d,), switches)
    catch err
        nfail[] += 1
        println("case $ci (P=$P, ell=$ell, n=$n, wc=$wc): device step THREW: ",
            sprint(showerror, err))
        flush(stdout); continue
    end

    # grid agreement first: a topology divergence explains a field mismatch, and
    # a field match with a topology mismatch would be a coincidence worth seeing
    hs, ds = hcache.state, dcache.state
    same_counts = ds.counts.n_bodies == hs.counts.n_bodies &&
        ds.counts.n_cells == hs.counts.n_cells &&
        ds.counts.n_nodes == hs.counts.n_nodes

    e_vel = relerr(sys_d.gradient_stretching[1:3, :], sys_h.gradient_stretching[1:3, :])
    ok = same_counts && e_vel < TOL
    ok ? (npass[] += 1) : (nfail[] += 1)
    @printf("  [%.0fs] ", time() - t_case)
    println("case $ci (P=$P, ell=$ell, n=$n, wc=$wc): ", ok ? "PASS" : "FAIL",
        "  velocity=", e_vel,
        "  counts=(", ds.counts.n_bodies, ",", ds.counts.n_cells, ",",
        ds.counts.n_nodes, ") host=(", hs.counts.n_bodies, ",",
        hs.counts.n_cells, ",", hs.counts.n_nodes, ")")
    flush(stdout)

    # a second step on the same cache must reproduce the first: the recurring
    # path takes the occupancy-epoch fast branch (identical positions => the
    # leaf-cell set is unchanged), which the first step never exercises
    if ok
        fill!(sys_d.gradient_stretching, zero(TF))
        fill!(sys_d.potential, zero(TF))
        ext.ka_radix_cache_device_step!(dcache, (sys_d,), switches)
        e2 = relerr(sys_d.gradient_stretching[1:3, :], sys_h.gradient_stretching[1:3, :])
        e2 < TOL || (nfail[] += 1; npass[] -= 1;
            println("  case $ci second step (epoch fast path) FAILED: velocity=", e2))
    end
end

# The caller's M2L chunk reaches the device plan (it was replaced by the default
# 2^17, so the M2L scratch was sized to the full route capacity whatever the
# caller asked for), and a chunk far below the route count -- many pieces per
# apply -- still matches the host cache. Same for the near-pair flag/scan
# scratch: a bound far below the pair count, so the compaction runs in many chunks;
# and the pair buffers start far below the pair count, so they grow in place;
# and the M2M/L2L scratch is far narrower than the level groups, so they run in chunks.
let (P, ell, n, wc) = (4, 4, 512, 64), chunk = 64, nflag = 97, npair0 = 50, sbatch = 8, TF = Float32
    sys_h = make_system(5199, n, TF); sys_d = make_system(5199, n, TF)
    opts = FM.RadixLifecycleOptions(; precision=TF,
        m2l_strategy=FM.ConcatenatedFixedZM2L(chunk), body_type=FM.Point{FM.Vortex})
    hcache = RadixFMMCache(sys_h; expansion_order=P, ell=ell, window_classes=wc, options=opts)
    fmm!(sys_h, hcache)
    a = device_build_args(hcache)
    LH = typeof(hcache).parameters[2]
    dcache = ext.ka_radix_cache_device_build(DEV_BACKEND, (sys_d,),
        hcache.expansion_order, ell, hcache.x_min, hcache.h0,
        hcache.max_n_bodies, hcache.options, hcache.policy,
        hcache.accepted_offsets, hcache.rejected_offsets,
        hcache.max_cells, hcache.max_nodes, hcache.route_capacity,
        hcache.direct_capacity, hcache.state.multipoles.basis_info, Val(LH);
        hierarchical_tables=a.tables,
        hierarchical_level_class_of=a.level_class_of, hessian=hcache.hessian,
        ell_axes=hcache.ell_axes, box_extent=hcache.box_extent,
        root_level=a.root_level, first_m2l_level=a.first_m2l_level,
        direct_flag_capacity=nflag, direct_pair_capacity=npair0, stage_batch=sbatch)
    switches = FM.DerivativesSwitch(FM.to_vector(false, 1), FM.to_vector(true, 1),
        FM.to_vector(false, 1), (sys_d,))
    ext.ka_radix_cache_device_step!(dcache, (sys_d,), switches)
    plan = dcache.state.interaction_list.apply_plan
    e_vel = relerr(sys_d.gradient_stretching[1:3, :], sys_h.gradient_stretching[1:3, :])
    nroutes = dcache.state.interaction_list.total_routes
    ndirect = dcache.state.interaction_list.epoch_n_direct
    ws = dcache.state.scratch
    gmax = maximum(g.count[] for g in vcat(ws.m2m_groups, ws.l2l_groups))
    ok = plan.chunk == min(chunk, hcache.route_capacity) && size(plan.aphi, 2) == plan.chunk &&
        nroutes > chunk && length(dcache.device_ctx.direct_flags) == nflag && ndirect > nflag &&
        length(dcache.state.direct_targets) >= ndirect > npair0 &&
        dcache.state.direct_targets === dcache.device_ctx.direct_targets &&
        size(ws.aphi, 2) == sbatch && gmax > sbatch && length(plan.route_class) == chunk &&
        e_vel < TOL
    ok ? (npass[] += 1) : (nfail[] += 1)
    println("m2l chunk + direct flag bound (chunk=$chunk, routes=$nroutes; flags=$nflag, pairs=$ndirect): ",
        ok ? "PASS" : "FAIL", "  plan.chunk=", plan.chunk, "  scratch cols=", size(plan.aphi, 2),
        "  flag length=", length(dcache.device_ctx.direct_flags),
        "  pair buffer ", npair0, " -> ", length(dcache.state.direct_targets),
        "  stage scratch ", size(ws.aphi, 2), " cols for groups up to ", gmax, "  velocity=", e_vel)
    flush(stdout)
end

println("\nKA device cache build+step vs host RadixFMMCache fmm!: ",
    "$(npass[])/$(length(CASES) + 1) pass")
nfail[] == 0 || error("$(nfail[]) case(s) failed")
