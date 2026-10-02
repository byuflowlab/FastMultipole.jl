# Extra source systems carried by the resident tree (src/resident_extra_tree.jl),
# host path:
#   1. the multipole a generic body type writes into the resident slab must
#      match the trusted resident kernel coefficient for coefficient, which is
#      checkable for Point{Vortex} because both paths exist,
#   2. at a depth where every cell pair is near, the tree carries nothing and
#      the result must equal the all-direct reference to roundoff; deeper, the
#      far field must be carried.
# The device path is checked against this host path in
# gpu/ka_extra_tree_correctness.jl.
if !@isdefined(FM) const FM = FastMultipole end
@isdefined(VortexParticles) || include(joinpath(@__DIR__, "vortex.jl"))
@isdefined(Segs) || include(joinpath(@__DIR__, "extra_tree_test_systems.jl"))

let TF = Float64
@testset "extra tree: slab layout against the resident Point{Vortex} kernel" begin
let n = 40, P = 6
    Random.seed!(3)
    sys = PV([SVector{3,TF}(rand(3)) for _ in 1:n], [SVector{3,TF}(randn(3) ./ n) for _ in 1:n])
    buffer = zeros(TF, 8, n)
    for i in 1:n; FM.source_system_to_buffer!(buffer, i, sys, i); end
    ranges = reshape([1, n], 2, 1); centers = reshape(TF[0.5, 0.5, 0.5], 3, 1)
    # ragged, as the real slabs are under Lamb-Helmholtz: P_chi = P_active = P_phi + 1
    P_chi = P + 1
    rows_phi = 2 * FM.harmonic_index(P, P)
    rows_chi = 2 * FM.harmonic_index(P_chi, P_chi)
    ph_ref = zeros(TF, rows_phi, 1); ch_ref = zeros(TF, rows_chi, 1)
    FM._host_b2m_vortex_kernel!(ph_ref, ch_ref, buffer, ranges, centers, [1], P, P_chi, 1)
    ph = zeros(TF, rows_phi, 1); ch = zeros(TF, rows_chi, 1)
    FM._resident_extra_b2m_kernel!(ph, ch, sys, buffer, ranges, centers, [1], P, P_chi, 1, Val(true))
    tol = 1e-12 * max(maximum(abs, ph_ref), maximum(abs, ch_ref))
    @test maximum(abs.(ph .- ph_ref)) <= tol && maximum(abs.(ch .- ch_ref)) <= tol
    # a chi slab sized to P_phi must be rejected, not written past: that
    # overflow corrupted the heap before the ragged fix
    short = zeros(TF, rows_phi, 1)
    threw = try
        FM._resident_extra_b2m_kernel!(zeros(TF, rows_phi, 1), short, sys, buffer,
            ranges, centers, [1], P, P_chi, 1, Val(true))
        false
    catch err
        err isa DimensionMismatch
    end
    @test threw
    # the compact device columns must be ragged the same way
    _, cphi, cchi = FM.resident_extra_multipole_columns(TF, sys, buffer, ranges,
        centers, [1], P, P_chi, 1, Val(true), rows_phi, rows_chi)
    @test (size(cphi, 1) == rows_phi && size(cchi, 1) == rows_chi &&
          maximum(abs.(cchi[:, 1] .- ch_ref[:, 1])) <= tol)
end
end

@testset "extra tree: filament sources, near-only exact and the far field carried" begin
let n = 800, ns = 120
    Random.seed!(1)
    sys0() = VortexParticles(TF.(rand(3, n)), TF.(randn(3, n) ./ n), fill(TF(0.01), n);
        potential = zeros(TF, 13, n), gradient_stretching = zeros(TF, 6, n))
    Random.seed!(2)
    r1 = [SVector{3,TF}(rand(3)) for _ in 1:ns]
    ex = Segs(r1, [r1[i] + SVector{3,TF}(0.01 .* randn(3)) for i in 1:ns],
              TF.(randn(ns) ./ ns), fill(TF(0.005), ns))
    opts = FM.RadixLifecycleOptions(; precision = TF,
        m2l_strategy = FM.ConcatenatedFixedZM2L(), body_type = FM.Point{FM.Vortex})
    # `near` is the tree path with the multipoles left out: the far field is
    # then missing entirely, which is what the third check must reject
    function arms(P, ell)
        Random.seed!(1); sys = sys0()
        cache = RadixFMMCache(sys; expansion_order = P, ell = ell, window_classes = 64,
                              options = opts, hessian = true)
        FM.update_radix_state!(cache, (sys,)); st = cache.state
        nc = Int(st.counts.n_cells); nb = Int(st.counts.n_bodies)
        g(o) = copy(view(Array(o), 2:4, 1:nb))
        FM.run_host_radix_lifecycle!(st); base = g(st.output)
        FM.update_radix_state!(cache, (sys,)); FM.run_host_radix_lifecycle!(st)
        FM._radix_extra_sources_into_output!(st, (ex,)); ref = g(st.output) .- base
        binned, loose = FM.bin_resident_extra_source(TF, ex, st.grid, nc)
        # the shipped driver, which also sums the bodies held out of the tree
        FM.update_radix_state!(cache, (sys,))
        FM.run_host_radix_lifecycle_with_extra_tree!(st, (ex,))
        tree = g(st.output) .- base
        FM.update_radix_state!(cache, (sys,)); FM.run_host_radix_lifecycle!(st)
        FM.resident_extra_near!(st, binned, FM.direct_kernel(ex))
        near = g(st.output) .- base
        scale = maximum(abs, ref)
        return maximum(abs.(tree .- ref)) / scale, maximum(abs.(near .- ref)) / scale, size(loose, 2)
    end
    e_tree1, _, loose1 = arms(4, 1)
    @test e_tree1 < 1e-12
    e_tree, e_near, _ = arms(4, 3)
    # with the far field missing, or delivered with the wrong sign, the tree
    # path is no better than dropping it; it must beat that by a wide margin
    @test e_tree < e_near / 20
end
end
end
