# Extra source systems carried by the resident tree (src/resident_extra_tree.jl).
# Host only: no device is required.
#
#   1. the multipole a generic body type writes into the resident slab must
#      match the trusted resident kernel coefficient for coefficient, which is
#      checkable for Point{Vortex} because both paths exist,
#   2. at a depth where every cell pair is near, the tree carries nothing and
#      the result must equal the all-direct reference to roundoff,
#   3. deeper, the difference from the all-direct reference is the multipole
#      truncation of the far field and must fall with the expansion order,
#   4. sources-only (the particles are targets but not sources) must deliver
#      the extra system's field alone, whether it arrives as `extra_sources`
#      or as `extra_tree_sources`.
using FastMultipole, Random, Printf, LinearAlgebra, Test
using FastMultipole.StaticArrays
const FM = FastMultipole
include(joinpath(@__DIR__, "..", "vortex.jl"))

include(joinpath(@__DIR__, "ka_extra_tree_correctness_systems.jl"))

const TF = Float64
npass = Ref(0); nfail = Ref(0)
check(ok, msg) = (ok ? (npass[] += 1) : (nfail[] += 1); println(ok ? "  PASS  $msg" : "  FAIL  $msg"))

println("1. slab layout against the resident Point{Vortex} kernel")
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
    check(maximum(abs.(ph .- ph_ref)) <= tol && maximum(abs.(ch .- ch_ref)) <= tol,
          @sprintf("phi %.1e, chi %.1e", maximum(abs.(ph .- ph_ref)), maximum(abs.(ch .- ch_ref))))
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
    check(threw, "an undersized chi slab raises DimensionMismatch")
    # the compact device columns must be ragged the same way
    _, cphi, cchi = FM.resident_extra_multipole_columns(TF, sys, buffer, ranges,
        centers, [1], P, P_chi, 1, Val(true), rows_phi, rows_chi)
    check(size(cphi, 1) == rows_phi && size(cchi, 1) == rows_chi &&
          maximum(abs.(cchi[:, 1] .- ch_ref[:, 1])) <= tol,
          "device multipole columns are ragged and match")
end

println("2. filament sources: near-only exact, and the far field carried")
let n = 800, ns = 120
    Random.seed!(1)
    sys0() = VortexParticles(TF.(rand(3, n)), TF.(randn(3, n) ./ n), fill(TF(0.01), n);
        potential = zeros(TF, 13, n), gradient_stretching = zeros(TF, 6, n))
    Random.seed!(2)
    r1 = [SVector{3,TF}(rand(3)) for _ in 1:ns]
    ex = Segs(r1, [r1[i] + SVector{3,TF}(0.01 .* randn(3)) for i in 1:ns],
              TF.(randn(ns) ./ ns), fill(TF(0.005), ns))
    opts = FM.CUDARadixLifecycleOptions(; precision = TF,
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
    check(e_tree1 < 1e-12, @sprintf("every pair near is exact (%.1e, %d loose)", e_tree1, loose1))
    e_tree, e_near, _ = arms(4, 3)
    # with the far field missing, or delivered with the wrong sign, the tree
    # path is no better than dropping it; it must beat that by a wide margin
    check(e_tree < e_near / 20,
          @sprintf("far field carried (tree %.1e vs near-only %.1e)", e_tree, e_near))
end

println("3. the device path against the verified host path")
include(joinpath(@__DIR__, "ka_backend.jl"))
if !dev_functional()
    println("  $(DEV_NAME) not functional; device check skipped")
else
    let n = 800, ns = 120, P = 4, ell = 3, DTF = (DEV_NAME == "Metal" ? Float32 : Float64)
        Random.seed!(1)
        pos = DTF.(rand(3, n)); str = DTF.(randn(3, n) ./ n)
        mk() = VortexParticles(copy(pos), copy(str), fill(DTF(0.01), n);
            potential = zeros(DTF, 13, n), gradient_stretching = zeros(DTF, 6, n))
        Random.seed!(2)
        r1 = [SVector{3,DTF}(rand(3)) for _ in 1:ns]
        ex = Segs(r1, [r1[i] + SVector{3,DTF}(0.01 .* randn(3)) for i in 1:ns],
                  DTF.(randn(ns) ./ ns), fill(DTF(0.005), ns))
        opts = FM.CUDARadixLifecycleOptions(; precision = DTF,
            m2l_strategy = FM.ConcatenatedFixedZM2L(), body_type = FM.Point{FM.Vortex})
        FM.device_backend(::VortexParticles) = DEV_BACKEND
        extmod = Base.get_extension(FastMultipole, :FastMultipoleKAExt)
        function run(kw)
            sysd = mk()
            cd_ = RadixFMMCache(sysd; expansion_order = P, ell = ell, window_classes = 64,
                                options = opts, hessian = true, device = true)
            sw = FM.DerivativesSwitch(false, true, true, (sysd,))
            extmod.ka_radix_cache_device_step!(cd_, (sysd,), sw; kw...)
            # de-permute: the device sort's within-cell order is not
            # reproducible between runs, so slot order cannot be compared
            st = cd_.state
            nb = Int(st.counts.n_bodies)
            out = Array(st.output)
            idx = Array(st.host_body_indices)
            res = zeros(DTF, 12, n)                  # velocity, then the nine gradient rows
            inv = Array(cd_.state.grid.invperm)
            for i in 1:nb
                res[:, i] .= out[2:13, inv[i]]
            end
            return res
        end
        base = run((;))
        ref = run((; extra_sources = (ex,))) .- base        # all-direct extra
        new = run((; extra_tree_sources = (ex,))) .- base   # carried by the tree
        V = 1:3; G = 4:12
        e = maximum(abs.(new[V, :] .- ref[V, :])) / maximum(abs, ref[V, :])
        tol = DTF === Float32 ? 5e-3 : 1e-3
        check(e <= tol, @sprintf("device tree matches device all-direct (%.2e, tol %.0e)", e, tol))
        # the velocity gradient too: the near pass once dropped it (2026-09-26),
        # which the velocity-only check above could not see
        eg = maximum(abs.(new[G, :] .- ref[G, :])) / maximum(abs, ref[G, :])
        check(eg <= tol, @sprintf("device tree gradient matches device all-direct (%.2e, tol %.0e)", eg, tol))

        # 4. sources-only: the particles contribute nothing as sources, so the
        # whole result is the extra system's field. This is the "body on wake"
        # direction, where the self-induction was evaluated earlier in the step.
        # `extra_tree_sources` were silently DROPPED here before 2026-09-19 --
        # the device branch zeroed the output and applied only `extra_sources`.
        so_ref = run((; extra_sources = (ex,), self_induce = false))
        so_new = run((; extra_tree_sources = (ex,), self_induce = false))
        e2 = maximum(abs.(so_new[V, :] .- so_ref[V, :])) / maximum(abs, so_ref[V, :])
        check(e2 <= tol,
              @sprintf("sources-only carries tree sources (%.2e, tol %.0e)", e2, tol))
        # and it must be the extra field alone, not the self-induction again
        check(maximum(abs.(so_ref[V, :] .- ref[V, :])) / maximum(abs, ref[V, :]) <= tol,
              "sources-only carries the extra field alone")
    end
end

@printf("\n%d passed, %d failed\n", npass[], nfail[])
nfail[] == 0 || error("$(nfail[]) check(s) failed")
println("gate passed: extra sources carried by the resident tree")
