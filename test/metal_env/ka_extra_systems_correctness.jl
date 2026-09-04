# Extra target / source systems on the radix path (src/radix_extra_systems.jl):
#
#   fmm!((main, probes), (main, segments), cache)
#
# with `main` a vortex particle system on the resident lifecycle, `probes` a
# pure target (FastMultipole.ProbeSystem) and `segments` a pure source (a
# straight vortex filament, defined here with the extra-source contract).
#
# Gates, per case:
#   1. host cache, probes <- particles   vs FastMultipole's generic direct!
#   2. host cache, probes unchanged by the segments (extra sources hit main only)
#   3. host cache, particles <- segments vs a Float64 host loop of the same functor
#   4. host and device, extra-sources-ONLY call fmm!((sys,), (segs,), cache):
#      the lifecycle is skipped and the particles get the segments alone
#   5. device cache: all outputs vs the host cache (Float32 lifecycle TOL)
#
# The segment physics is not independently gated here; it is a test kernel.
# ActuatorLines gates its own filament functor against its pairwise loops.

include("ka_backend.jl")
using FastMultipole, Random, Test, Printf, LinearAlgebra
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

FM.device_backend(::VortexParticles) = DEV_BACKEND
# every generator draws in Float64 and casts, so the Float32 systems and their
# Float64 references share positions (the Float32/Float64 random streams differ)
make_system(seed, n, TF) = (Random.seed!(seed);
    VortexParticles(TF.(rand(3, n)), TF.(randn(3, n) ./ n), zeros(TF, n);
        potential=zeros(TF, 13, n), gradient_stretching=zeros(TF, 6, n)))

#------- a straight vortex segment as an extra source -------#

struct TestSegments{TF}
    r1::Vector{SVector{3,TF}}
    r2::Vector{SVector{3,TF}}
    gamma::Vector{TF}
end
function make_segments(seed, ns, TF)
    Random.seed!(seed)
    r1 = [SVector{3,TF}(rand(3)) for _ in 1:ns]
    r2 = [r1[i] + SVector{3,TF}(0.05 .* randn(3)) for i in 1:ns]
    TestSegments(r1, r2, TF.(randn(ns) ./ ns))
end
FM.get_n_bodies(s::TestSegments) = length(s.gamma)
FM.data_per_body(::TestSegments) = 12
FM.strength_dims(::TestSegments) = 1
FM.has_vector_potential(::TestSegments) = true
FM.get_position(s::TestSegments, i) = (s.r1[i] + s.r2[i]) / 2
function FM.source_system_to_buffer!(buffer, ib, s::TestSegments, i)
    c = (s.r1[i] + s.r2[i]) / 2
    buffer[1:3, ib] .= c
    buffer[4, ib] = norm(s.r2[i] - s.r1[i]) / 2
    buffer[5, ib] = s.gamma[i]
    buffer[6:8, ib] .= s.r1[i]
    buffer[9:11, ib] .= s.r2[i]
    buffer[12, ib] = 0
end
struct SegmentKernel <: FM.AbstractDirectKernel end
FM.direct_kernel(::TestSegments) = SegmentKernel()

# singular straight segment, Biot-Savart
@inline function FM._extra_pair_ug(::SegmentKernel, tx, ty, tz, buf, j)
    T = typeof(tx)
    @inbounds begin
        g = buf[5, j]
        ax = tx - buf[6, j]; ay = ty - buf[7, j]; az = tz - buf[8, j]
        bx = tx - buf[9, j]; by = ty - buf[10, j]; bz = tz - buf[11, j]
    end
    cx = ay * bz - az * by; cy = az * bx - ax * bz; cz = ax * by - ay * bx
    c2 = cx * cx + cy * cy + cz * cz
    na = sqrt(ax * ax + ay * ay + az * az); nb = sqrt(bx * bx + by * by + bz * bz)
    if c2 <= zero(T) || na <= zero(T) || nb <= zero(T)
        return (zero(T), zero(T), zero(T), zero(T))
    end
    r0x = ax - bx; r0y = ay - by; r0z = az - bz
    s = (r0x * ax + r0y * ay + r0z * az) / na - (r0x * bx + r0y * by + r0z * bz) / nb
    k = g * s / (T(4) * T(pi) * c2)
    return (zero(T), k * cx, k * cy, k * cz)
end

# Float64 reference loop through the same functor
function segments_on_points(segs::TestSegments, pts::AbstractMatrix)
    buf = FM._radix_extra_source_buffer(Float64, segs)
    out = zeros(Float64, 4, size(pts, 2))
    FM._host_targets_from_extra_source!(out, SegmentKernel(), Float64.(pts), size(pts, 2),
        buf, Val(false))
    return out[2:4, :]
end

particle_positions(sys) = reduce(hcat, [Vector(b.position) for b in sys.bodies])
probe_velocity(p) = reduce(hcat, [Vector(g) for g in p.gradient])

function make_probes(seed, nprobe, TF)
    Random.seed!(seed)
    p = FM.ProbeSystem(nprobe, TF)
    for i in 1:nprobe
        p.position[i] = SVector{3,TF}(rand(3))
    end
    p
end

const CASES = [
    # P, ell, n, wc, nprobe, nseg
    (4, 3,  256,  8, 16,  8),
    (4, 4, 4096,  8, 64, 32),
    (4, 4, 4096, 64, 128, 64),   # P stays 4: the extras kernels are P-independent, and a P=6 Metal compile alone cost 40 s
]
const TOL_HOST = 2e-3   # Float32 host lifecycle vs Float64 references
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
    opts = FM.CUDARadixLifecycleOptions(; precision=TF,
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
    pr_ref = make_probes(6300 + ci, nprobe, Float64)
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
    sys_ref = make_system(6100 + ci, n, Float64)
    FM.direct!((pr_ref,), (sys_ref,); gradient=true)
    e_probe_h = relerr(probe_velocity(pr_h), probe_velocity(pr_ref))
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
    seg_on_particles_ref = segments_on_points(segs, particle_positions(sys_ref))
    e_segp_h = relerr(probe_velocity(pr2_h), probe_velocity(pr_h))   # must be untouched
    e_segs_h = relerr(sys_h.gradient_stretching[1:3, :] .- u1_h, seg_on_particles_ref)
    e_segp_d = relerr(probe_velocity(pr2_d), probe_velocity(pr2_h))
    e_segs_d = relerr(sys_d.gradient_stretching[1:3, :], sys_h.gradient_stretching[1:3, :])
    # --- call 3: extra sources ONLY (no self-induction, lifecycle skipped) ---
    for s in (sys_h, sys_d)
        fill!(s.gradient_stretching, zero(TF)); fill!(s.potential, zero(TF))
    end
    local e_only_h, e_only_d
    try
        fmm!((sys_h,), (segs,), hcache)
        fmm!((sys_d,), (segs,), dcache)
    catch err
        nfail[] += 1
        println("  call 3 (sources only) THREW: ", sprint(showerror, err)[1:min(end, 600)])
        flush(stdout); continue
    end
    e_only_h = relerr(sys_h.gradient_stretching[1:3, :], seg_on_particles_ref)
    e_only_d = relerr(sys_d.gradient_stretching[1:3, :], sys_h.gradient_stretching[1:3, :])

    ok = e_probe_h < TOL_HOST && e_segp_h < TOL_HOST && e_segs_h < TOL_HOST &&
         e_only_h < TOL_HOST && e_only_d < TOL_DEV &&
         e_probe_d < TOL_DEV && e_self_d < TOL_DEV && e_segp_d < TOL_DEV && e_segs_d < TOL_DEV
    ok ? (npass[] += 1) : (nfail[] += 1)
    @printf("  [%.0fs] %s  host: probes<-particles=%.2e probes-unchanged=%.2e particles<-segs=%.2e sources-only=%.2e | device vs host: self=%.2e probes=%.2e probes2=%.2e particles2=%.2e sources-only=%.2e\n",
        time() - t_case, ok ? "PASS" : "FAIL", e_probe_h, e_segp_h, e_segs_h, e_only_h,
        e_self_d, e_probe_d, e_segp_d, e_segs_d, e_only_d)
    flush(stdout)
end
println("\nradix extra target/source systems, host and $(DEV_NAME): ",
    "$(npass[])/$(length(CASES)) pass")
nfail[] == 0 || error("$(nfail[]) case(s) failed")
