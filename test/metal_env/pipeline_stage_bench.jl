# Top-to-bottom pipeline bench, built up ONE STAGE AT A TIME.
#
# Driven by a real FLOWUnsteady rotor wake (see pipeline_field.jl), at the
# geometry FLOWVPM's own `_radix_auto_geometry` picks for that wake -- not the
# hardcoded ell / sigma=0 / unit-cube of the ka_*_correctness suites.
#
# Each stage reports wall time, allocation count and allocated bytes, plus its
# accuracy check against the host oracle. Stages are added in CUDA's order:
# stage 0 (entry) first, then 1, and so on.

include("ka_backend.jl")
include("pipeline_field.jl")

using FastMultipole, Printf, Test
using FastMultipole.StaticArrays
const FM = FastMultipole
const V = FLOWVPM

dev_functional() || (println("$(DEV_NAME) not functional; skipping"); exit(0))
ext = Base.get_extension(FastMultipole, :FastMultipoleKAExt)
ext !== nothing || error("FastMultipoleKAExt did not load")

include(joinpath(@__DIR__, "pipeline_device_args.jl"))

# Time EVERY call, never just the second one. On Metal the first TWO calls pay
# MPSGraph/kernel-cache population -- call 2 is routinely the most expensive of
# the run -- so a warm-up-once-then-time-the-next helper reports a figure that
# does not exist as a per-step cost. See the 2026-08-30 session-19 note and
# [[feedback-two-call-gpu-warmup]]. `measured` returns the LAST call (the
# plateau) plus every call, so the ramp stays visible in the log.
"Run `f` `calls` times; return (value, rows) with rows[i] = (seconds, allocs, bytes)."
function measured(f; calls::Int=5)
    rows = Tuple{Float64,Int,Int}[]
    local v
    for _ in 1:calls
        GC.gc()
        st = Base.gc_num(); t0 = time_ns()
        v = f()
        t1 = time_ns(); d = Base.GC_Diff(Base.gc_num(), st)
        push!(rows, ((t1 - t0) / 1e9, Base.gc_alloc_count(d), d.allocd))
    end
    return v, rows
end

function report(label, rows; extra="")
    t, a, b = rows[end]
    ramp = length(rows) > 1 ?
        "ramp " * join((@sprintf("%.3f", r[1]) for r in rows), "/") : ""
    @printf("  %-34s %8.4f s  %10d allocs  %10.3f MiB   %s %s\n",
        label, t, a, b/2^20, extra, ramp)
end

const STEP = 36
const SIZES = (512, 2048, 8192)

# np => steady-state (allocs, bytes) of one ka_fmm! step, for the sweep table.
const STEP_ALLOCS = Tuple{Int,Int,Int}[]

println("=== pipeline stage bench: real wake $(WAKE_CASE) step $(STEP) ===")
flush(stdout)

for np in SIZES
    pf_h = load_wake(STEP; np=np, TF=Float64)
    n = V.get_np(pf_h)

    # --- production config, from FLOWVPM's OWN builder --------------------
    #
    # Do NOT hand-roll `RadixFMMCache(...)` here: `_build_radix_fmm_cache`
    # derives padded (and optionally center-snapped) bounds and feeds THOSE to
    # the cache, so consulting `_radix_auto_geometry` separately and then
    # building a cache without the same bounds picks an `ell` for one geometry
    # and validates it against another -- which trips the nearfield adequacy
    # gate (src/translate_batched_resident.jl:2104).
    # m2l_strategy: FLOWVPM's DEFAULT is :dense (DenseTranslationM2L,
    # FLOWVPM_fmm_radix.jl:110). KA has no plan for it --
    # `ka_radix_cache_device_build` throws "m2l_strategy=DenseTranslationM2L has
    # no KA plan" (ext:5248) -- so this bench pins :concat, which is the one
    # strategy KA implements. DEVIATION FROM PRODUCTION DEFAULT, recorded here
    # rather than hidden: closing it is its own task.
    st = V.RadixFMMSettings(; precision=Float32, window_classes=256,
        m2l_strategy=:concat)
    (hcache, rows) = measured(() -> V._build_radix_fmm_cache(pf_h, st); calls=1)
    ell = hcache.ell
    @printf("\nnp=%d  ell=%d  P=%d  wc=256  ell_axes=%s  max_cells=%d\n",
        n, ell, hcache.expansion_order, string(Tuple(hcache.ell_axes)),
        hcache.max_cells); flush(stdout)
    report("host cache build", rows)

    # --- host oracle -------------------------------------------------------
    (_, rows) = measured(() -> fmm!(pf_h, hcache); calls=3)
    report("host oracle  fmm!", rows)

    # --- KA cache (front end still supplied by the host cache: stages 2-6) --
    pf_d = load_wake(STEP; np=np, TF=Float64)
    c = production_caller_args(pf_h, st, typeof(hcache).parameters[1])
    c.ell == ell || error("caller-arg ell=$(c.ell); the host cache was built at $ell")
    dcache = build_ka_cache(ext, hcache, pf_d, c)

    # --- STAGE 1: argument + trait validation ------------------------------
    # `build_ka_cache` already ran it and asserted its LH against the host
    # cache; this call is the timing/allocation row for the stage itself.
    (v1, rows) = measured(() -> ext.ka_validate_radix_arguments(DEV_BACKEND, (pf_d,);
        expansion_order=hcache.expansion_order, options=hcache.options,
        hessian=hcache.hessian, max_n_bodies=hcache.max_n_bodies); calls=3)
    report("stage 1      validate", rows;
        extra=@sprintf("LH=%s BT=%s dk=%s TF=%s n0=%d",
            v1.LH, nameof(v1.BT), nameof(typeof(v1.dk_trait)), v1.TF, v1.n0))
    for (name, got, want) in (("LH", v1.LH, typeof(hcache).parameters[2]),
                              ("TF", v1.TF, typeof(hcache).parameters[1]),
                              ("n0", v1.n0, n),
                              ("maxn", v1.maxn, hcache.max_n_bodies))
        got == want || error("stage 1 $name = $got; host cache says $want")
    end
    println("  stage 1      accuracy                     all resolved traits match the host cache")

    # --- STAGE 2: root geometry -------------------------------------------
    # `build_ka_cache` already ran it and asserted all four outputs against the
    # host cache; this call is the timing/allocation row for the stage itself.
    (g2, rows) = measured(() -> ext.ka_radix_geometry((pf_d,), v1.TF, ell;
        bounds=c.bounds); calls=3)
    report("stage 2      geometry", rows;
        extra=@sprintf("h0=%.4g ell_axes=%s", g2.h0, string(Tuple(g2.ell_axes))))
    println("  stage 2      accuracy                     x_min/h0/ell_axes/box_extent match the host cache")

    # --- STAGE 5: stencil policy + hierarchical tables ---------------------
    (s5, rows) = measured(() -> ext.ka_radix_stencil_policy(v1, ell, g2.h0,
        g2.ell_axes; near_radius2=c.near_radius2, window_classes=c.window_classes,
        level_radii2=c.level_radii2); calls=3)
    report("stage 5      policy+tables", rows;
        extra=@sprintf("root=%d first_m2l=%d |acc|=%d |rej|=%d",
            s5.root_level, s5.first_m2l_level, length(s5.accepted),
            length(s5.rejected)))
    println("  stage 5      accuracy                     policy/accepted/rejected/options match the host cache")

    # --- STAGE 6: capacity sizing -----------------------------------------
    (c6, rows) = measured(() -> ext.ka_radix_capacities(v1, ell, g2.ell_axes, s5); calls=3)
    report("stage 6      capacities", rows;
        extra=@sprintf("cells=%d nodes=%d lvl=%d route=%d direct=%d",
            c6.max_cells, c6.max_nodes, c6.max_level_nodes, c6.route_capacity,
            c6.direct_capacity))
    println("  stage 6      accuracy                     all five capacities match the host cache")

    # --- STAGE 0: ka_fmm! entry -------------------------------------------
    (_, rows) = measured(() -> ext.ka_fmm!(pf_d, dcache; hessian=true); calls=5)
    report("stage 0      ka_fmm!", rows)
    push!(STEP_ALLOCS, (n, rows[end][2], rows[end][3]))

    # --- accuracy against the oracle --------------------------------------
    #
    # `fmm!` ACCUMULATES into the particle field, so the timing loops above --
    # 3 host calls against 5 KA calls -- leave the two fields scaled by their
    # own call counts (a 2/3 relerr, all bookkeeping, no numerics). Reset both
    # and score ONE call each.
    V._reset_particles(pf_h); V._reset_particles(pf_d)
    fmm!(pf_h, hcache)
    ext.ka_fmm!(pf_d, dcache; hessian=true)
    Uh = [V.get_U(V.get_particle(pf_h, i))[k] for i in 1:n, k in 1:3]
    Ud = [V.get_U(V.get_particle(pf_d, i))[k] for i in 1:n, k in 1:3]
    relerr = maximum(abs.(Ud .- Uh)) / max(maximum(abs.(Uh)), eps())
    @printf("  %-34s relerr(U) = %.3e\n", "stage 0      accuracy", relerr)
    flush(stdout)
end

# --- residual per-step allocation sweep -----------------------------------
#
# The ~45k host allocations a KA step makes are Metal per-launch host-side
# objects (ObjC autorelease pools, MPSGraph dispatch), so their count should be
# set by the number of kernel launches -- fixed by `ell` and the group
# structure -- and NOT by particle count. FLAT across np means it is a constant
# against a step that grows with n, and can be ignored permanently; GROWTH
# means it is real and worth chasing. This table settles it.
println("\n=== residual per-step allocations vs np (steady state, stage 0) ===")
@printf("  %8s %14s %14s %12s\n", "np", "allocs", "MiB", "allocs/np")
for (n, a, b) in STEP_ALLOCS
    @printf("  %8d %14d %14.3f %12.2f\n", n, a, b/2^20, a/n)
end
if length(STEP_ALLOCS) >= 2
    lo, hi = STEP_ALLOCS[1][2], STEP_ALLOCS[end][2]
    nlo, nhi = STEP_ALLOCS[1][1], STEP_ALLOCS[end][1]
    @printf("  np x%.0f -> allocs x%.2f  (%s)\n", nhi/nlo, hi/lo,
        hi/lo < 1.3 ? "FLAT: launch-bound, ignore permanently" :
                      "GROWTH: per-body host allocation, real")
end
flush(stdout)
