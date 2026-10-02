# Extra target / source systems on the radix path (src/radix_extra_systems.jl):
#
#   fmm!((main, probes), (main, segments), cache)
#
# with `main` a vortex particle system on the resident lifecycle, `probes` a
# pure target (FastMultipole.ProbeSystem) and `segments` a pure source (a
# straight vortex filament, defined here with the extra-source contract).
#
# Gates, per case: the device cache against the host cache (Float32 lifecycle
# TOL) for every output of three calls -- particles + probes, + segments as an
# extra source, and extra-sources-ONLY fmm!((sys,), (segs,), cache), where the
# lifecycle is skipped and the particles get the segments alone. The host cache
# itself is checked against Float64 references in test/extra_systems_test.jl.
#
# The segment physics is not independently gated here; it is a test kernel.
# ActuatorLines gates its own filament functor against its pairwise loops.

include("ka_backend.jl")
using FastMultipole, Random, Test, Printf, LinearAlgebra
using FastMultipole.StaticArrays
const FM = FastMultipole
include(joinpath(@__DIR__, "..", "vortex.jl"))
include(joinpath(@__DIR__, "..", "extra_systems_test_systems.jl"))

if !dev_functional()
    println("$(DEV_NAME) not functional; skipping")
    exit(0)
end
ext = Base.get_extension(FastMultipole, :FastMultipoleKAExt)
ext !== nothing || error("FastMultipoleKAExt did not load")


FM.device_backend(::VortexParticles) = DEV_BACKEND
# A unit run takes the short case list; FM_FULL_SWEEP=1 takes the full one.
# The sweep is a robustness study, not a check: it belongs in a debugging pass,
# not in every run.
const CASES_FULL = [
    # P, ell, n, wc, nprobe, nseg
    (4, 3,  256,  8, 16,  8),
    (4, 4, 4096,  8, 64, 32),
    (4, 4, 4096, 64, 128, 64),   # P stays 4: the extras kernels are P-independent, and a P=6 Metal compile alone cost 40 s
]
const CASES_SHORT = [
    (4, 3, 256, 8, 16, 8),
    (4, 4, 512, 64, 64, 32),   # more probes and segments than cells
]
const CASES = haskey(ENV, "FM_FULL_SWEEP") ? CASES_FULL : CASES_SHORT
const TOL_DEV  = 3e-4   # device vs host, both Float32 (ka_device_cache_correctness)

npass = Ref(0); nfail = Ref(0)
for (ci, (P, ell, n, wc, nprobe, nseg)) in pairs(CASES)
    TF = Float32
    @printf("case %d (P=%d, ell=%d, n=%d, probes=%d, segs=%d): start\n",
        ci, P, ell, n, nprobe, nseg); flush(stdout)
    t_case = time()
    sys_h = make_system(6100 + ci, n, TF)
    sys_d = make_system(6100 + ci, n, TF)
    segs = make_segments(6200 + ci, nseg, TF)
    opts = FM.RadixLifecycleOptions(; precision=TF,
        m2l_strategy=FM.ConcatenatedFixedZM2L(), body_type=FM.Point{FM.Vortex})
    local hcache, dcache
    try
        hcache = RadixFMMCache(sys_h; expansion_order=P, ell=ell, window_classes=wc,
            options=opts)
        dcache = RadixFMMCache(sys_d; expansion_order=P, ell=ell, window_classes=wc,
            options=opts, device=true)
    catch err
        nfail[] += 1
        println("  cache build FAILED: ", sprint(showerror, err)[1:min(end, 400)])
        flush(stdout); continue
    end
    ok = true
    # --- call 1: particles on themselves + probes (no extra sources) ---
    pr_h = make_probes(6300 + ci, nprobe, TF)
    pr_d = make_probes(6300 + ci, nprobe, TF)
    try
        fmm!((sys_h, pr_h), (sys_h,), hcache)
        fmm!((sys_d, pr_d), (sys_d,), dcache)
    catch err
        nfail[] += 1
        println("  call 1 THREW: ", sprint(showerror, err)[1:min(end, 600)])
        flush(stdout); continue
    end
    u1_h = copy(sys_h.gradient_stretching[1:3, :])
    u1_d = copy(sys_d.gradient_stretching[1:3, :])
    e_probe_d = relerr(probe_velocity(pr_d), probe_velocity(pr_h))
    e_self_d = relerr(u1_d, u1_h)
    # --- call 2: + segments as an extra source ---
    pr2_h = make_probes(6300 + ci, nprobe, TF)
    pr2_d = make_probes(6300 + ci, nprobe, TF)
    try
        fmm!((sys_h, pr2_h), (sys_h, segs), hcache)
        fmm!((sys_d, pr2_d), (sys_d, segs), dcache)
    catch err
        nfail[] += 1
        println("  call 2 THREW: ", sprint(showerror, err)[1:min(end, 600)])
        flush(stdout); continue
    end
    e_segp_d = relerr(probe_velocity(pr2_d), probe_velocity(pr2_h))
    e_segs_d = relerr(sys_d.gradient_stretching[1:3, :], sys_h.gradient_stretching[1:3, :])
    # --- call 3: extra sources ONLY (no self-induction, lifecycle skipped) ---
    for s in (sys_h, sys_d)
        fill!(s.gradient_stretching, zero(TF)); fill!(s.potential, zero(TF))
    end
    local e_only_d
    try
        fmm!((sys_h,), (segs,), hcache)
        fmm!((sys_d,), (segs,), dcache)
    catch err
        nfail[] += 1
        println("  call 3 (sources only) THREW: ", sprint(showerror, err)[1:min(end, 600)])
        flush(stdout); continue
    end
    e_only_d = relerr(sys_d.gradient_stretching[1:3, :], sys_h.gradient_stretching[1:3, :])

    ok = e_only_d < TOL_DEV &&
         e_probe_d < TOL_DEV && e_self_d < TOL_DEV && e_segp_d < TOL_DEV && e_segs_d < TOL_DEV
    ok ? (npass[] += 1) : (nfail[] += 1)
    @printf("  [%.0fs] %s  device vs host: self=%.2e probes=%.2e probes2=%.2e particles2=%.2e sources-only=%.2e\n",
        time() - t_case, ok ? "PASS" : "FAIL",
        e_self_d, e_probe_d, e_segp_d, e_segs_d, e_only_d)
    flush(stdout)
end
println("\nradix extra target/source systems, $(DEV_NAME) vs host: ",
    "$(npass[])/$(length(CASES)) pass")
nfail[] == 0 || error("$(nfail[]) case(s) failed")
