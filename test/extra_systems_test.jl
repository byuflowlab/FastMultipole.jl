# Extra target / source systems on the host radix path (src/radix_extra_systems.jl):
#
#   fmm!((main, probes), (main, segments), cache)
#
# with `main` a vortex particle system on the resident lifecycle, `probes` a
# pure target (FastMultipole.ProbeSystem) and `segments` a pure source.
#
# Gates, per case, on a Float32 host cache:
#   1. probes <- particles   vs FastMultipole's generic direct!
#   2. probes unchanged by the segments (extra sources hit main only)
#   3. particles <- segments vs a Float64 host loop of the same functor
#   4. extra-sources-ONLY fmm!((sys,), (segs,), cache): the lifecycle is
#      skipped and the particles get the segments alone
# The device cache is checked against this host cache in
# gpu/ka_extra_systems_correctness.jl.
if !@isdefined(FM) const FM = FastMultipole end
@isdefined(VortexParticles) || include(joinpath(@__DIR__, "helpers", "vortex.jl"))
@isdefined(TestSegments) || include(joinpath(@__DIR__, "helpers", "extra_systems_test_systems.jl"))

@testset "extra target/source systems on a host cache" begin
    TOL_HOST = 2e-3    # Float32 host lifecycle vs Float64 references
    # Probes are summed all-pairs against the resident bodies, so this
    # tolerance is deliberately loose.
    TOL_PROBE = 5e-3
    # P, ell, n, wc, nprobe, nseg
    for (ci, (P, ell, n, wc, nprobe, nseg)) in pairs(((4, 3, 256, 8, 16, 8),
            (4, 4, 512, 64, 64, 32)))   # more probes and segments than cells
        TF = Float32
        sys = make_system(6100 + ci, n, TF)
        segs = make_segments(6200 + ci, nseg, TF)
        opts = FastMultipole.RadixLifecycleOptions(; precision=TF,
            m2l_strategy=FastMultipole.ConcatenatedFixedZM2L(),
            body_type=FastMultipole.Point{FastMultipole.Vortex})
        cache = RadixFMMCache(sys; expansion_order=P, ell=ell, window_classes=wc,
            options=opts)
        # call 1: particles on themselves + probes (no extra sources)
        pr = make_probes(6300 + ci, nprobe, TF)
        pr_ref = make_probes(6300 + ci, nprobe, Float64)
        fmm!((sys, pr), (sys,), cache)
        u1 = copy(sys.gradient_stretching[1:3, :])
        sys_ref = make_system(6100 + ci, n, Float64)
        FastMultipole.direct!((pr_ref,), (sys_ref,); gradient=true)
        @test relerr(probe_velocity(pr), probe_velocity(pr_ref)) < TOL_PROBE
        # call 2: + segments as an extra source
        pr2 = make_probes(6300 + ci, nprobe, TF)
        fmm!((sys, pr2), (sys, segs), cache)
        seg_on_particles_ref = segments_on_points(segs, particle_positions(sys_ref))
        @test relerr(probe_velocity(pr2), probe_velocity(pr)) < TOL_HOST   # untouched
        @test relerr(sys.gradient_stretching[1:3, :] .- u1, seg_on_particles_ref) < TOL_HOST
        # call 3: extra sources ONLY (no self-induction, lifecycle skipped)
        fill!(sys.gradient_stretching, zero(TF)); fill!(sys.potential, zero(TF))
        fmm!((sys,), (segs,), cache)
        @test relerr(sys.gradient_stretching[1:3, :], seg_on_particles_ref) < TOL_HOST
    end
end
