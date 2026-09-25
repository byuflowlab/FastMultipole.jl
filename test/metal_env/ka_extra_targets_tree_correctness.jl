# Extra TARGETS evaluated through the resident grid (src/radix_extra_systems.jl
# `_host_extra_targets_tree!`, ext `ka_extra_targets_evaluate!`): each target
# reads its cell's local expansion and sums its near source cells directly,
# instead of an all-pairs sweep over every body.
#
#   1. host: tree targets vs all-pairs targets, velocity, at the far-field
#      truncation of the lifecycle itself; targets outside the box and in
#      empty cells fall back to all-pairs and must be exact,
#   2. host: the same with the velocity gradient (13 rows),
#   3. device: the same call against the host result,
#   4. host: the probe error falls with the expansion order and is never worse
#      than the particles' own lifecycle error at the same setting.
using FastMultipole, Random, Printf, LinearAlgebra, Test
using FastMultipole.StaticArrays
const FM = FastMultipole
include(joinpath(@__DIR__, "..", "vortex.jl"))

const TF = Float64
npass = Ref(0); nfail = Ref(0)
check(ok, msg) = (ok ? (npass[] += 1) : (nfail[] += 1); println(ok ? "  PASS  $msg" : "  FAIL  $msg"))
relerr(a, b) = maximum(abs.(a .- b)) / maximum(abs, b)

# targets: most among the particles, a few outside the unit box (loose)
function targets(nt, TFt)
    Random.seed!(7)
    xt = TFt.(rand(3, nt))
    xt[:, end-3:end] .= TFt.(1.5 .* rand(3, 4) .+ 1.0)
    return xt
end

println("1./2. host: tree targets against all-pairs targets")
let n = 3000, nt = 200, P = 5, ell = 3
    Random.seed!(1)
    sys = VortexParticles(TF.(rand(3, n)), TF.(randn(3, n) ./ n), fill(TF(0.01), n);
        potential = zeros(TF, 13, n), gradient_stretching = zeros(TF, 6, n))
    opts = FM.CUDARadixLifecycleOptions(; precision = TF,
        m2l_strategy = FM.ConcatenatedFixedZM2L(), body_type = FM.Point{FM.Vortex})
    cache = RadixFMMCache(sys; expansion_order = P, ell = ell, window_classes = 64,
                          options = opts, hessian = true)
    FM.update_radix_state!(cache, (sys,)); st = cache.state
    FM.run_host_radix_lifecycle!(st)
    xt = targets(nt, TF)
    order, ranges, loose = FM.bin_resident_extra_targets(xt, st.grid, Int(st.counts.n_cells))
    check(length(loose) >= 4 && length(order) + length(loose) == nt,
          "binning: $(length(order)) binned, $(length(loose)) loose")
    for (HS, rows, label) in ((false, 4, "velocity"), (true, 13, "velocity + gradient"))
        ref = zeros(TF, rows, nt)
        FM._host_extra_targets_from_main!(ref, st.options.direct_kernel, xt, st.source_bodies,
            Int(st.counts.n_bodies), Val(HS))
        out = zeros(TF, rows, nt)
        FM._host_extra_targets_tree!(out, st, xt, Val(HS))
        e = relerr(out[2:rows, :], ref[2:rows, :])
        check(e <= 2e-3, @sprintf("%s: tree vs all-pairs %.2e (tol 2e-03)", label, e))
        el = relerr(out[2:rows, loose], ref[2:rows, loose])
        check(el <= 1e-12, @sprintf("%s: loose targets exact (%.1e)", label, el))
        # a missing near pair would show as an O(1) error on some target
        worst = maximum(abs.(out[2:4, order] .- ref[2:4, order])) / maximum(abs, ref[2:4, :])
        check(worst <= 2e-3, @sprintf("%s: worst binned target %.2e", label, worst))
    end
end

println("4. host: the probe error falls with the expansion order and stays below the particles' own")
let n = 256, ell = 3, nt = 16
    errs = Float64[]; perrs = Float64[]
    for P in 4:7
        Random.seed!(6101)
        sys = VortexParticles(TF.(rand(3, n)), TF.(randn(3, n) ./ n), fill(TF(0.01), n);
            potential = zeros(TF, 13, n), gradient_stretching = zeros(TF, 6, n))
        opts = FM.CUDARadixLifecycleOptions(; precision = TF,
            m2l_strategy = FM.ConcatenatedFixedZM2L(), body_type = FM.Point{FM.Vortex})
        cache = RadixFMMCache(sys; expansion_order = P, ell = ell, window_classes = 64, options = opts)
        FM.update_radix_state!(cache, (sys,)); st = cache.state
        FM.run_host_radix_lifecycle!(st)
        nb = Int(st.counts.n_bodies)
        Random.seed!(99); xt = TF.(rand(3, nt))
        ref = zeros(TF, 4, nt)
        FM._host_extra_targets_from_main!(ref, st.options.direct_kernel, xt, st.source_bodies, nb, Val(false))
        out = zeros(TF, 4, nt)
        FM._host_extra_targets_tree!(out, st, xt, Val(false))
        push!(errs, relerr(out[2:4, :], ref[2:4, :]))
        # the particles' own lifecycle error against the exact sum at their positions
        xb = Array(view(st.source_bodies, 1:3, 1:nb))
        refb = zeros(TF, 4, nb)
        FM._host_extra_targets_from_main!(refb, st.options.direct_kernel, xb, st.source_bodies, nb, Val(false))
        push!(perrs, relerr(Array(view(st.output, 2:4, 1:nb)), refb[2:4, :]))
    end
    @printf("      P=4..7 probe %s\n      P=4..7 body  %s\n",
            join((@sprintf("%.1e", e) for e in errs), " "), join((@sprintf("%.1e", e) for e in perrs), " "))
    check(errs[end] < errs[1] / 4, "probe error falls with P")
    check(all(errs .<= perrs), "probe error at or below the particles' own at every P")
end

println("3. device against host")
include(joinpath(@__DIR__, "ka_backend.jl"))
if !dev_functional()
    println("  $(DEV_NAME) not functional; device check skipped")
else
    let n = 3000, nt = 200, P = 5, ell = 3, DTF = (DEV_NAME == "Metal" ? Float32 : Float64)
        Random.seed!(1)
        pos = DTF.(rand(3, n)); str = DTF.(randn(3, n) ./ n)
        mk() = VortexParticles(copy(pos), copy(str), fill(DTF(0.01), n);
            potential = zeros(DTF, 13, n), gradient_stretching = zeros(DTF, 6, n))
        opts = FM.CUDARadixLifecycleOptions(; precision = DTF,
            m2l_strategy = FM.ConcatenatedFixedZM2L(), body_type = FM.Point{FM.Vortex})
        FM.device_backend(::VortexParticles) = DEV_BACKEND
        extmod = Base.get_extension(FastMultipole, :FastMultipoleKAExt)
        extmod._KA_SETTING_OVERRIDES[:KA_EXTRA_TARGETS_GRID] = true   # the grid path is opt-in
        xt = targets(nt, DTF)
        mkp() = (p = FM.ProbeSystem(nt, DTF); for i in 1:nt; p.position[i] = SVector{3,DTF}(xt[:, i]); end; p)
        probes_h = mkp(); probes_d = mkp()
        sysh = mk()
        ch = RadixFMMCache(sysh; expansion_order = P, ell = ell, window_classes = 64,
                           options = opts, hessian = true)
        FM.fmm!((sysh, probes_h), (sysh,), ch; scalar_potential = false, gradient = true, hessian = (true, false))
        sysd = mk()
        cd_ = RadixFMMCache(sysd; expansion_order = P, ell = ell, window_classes = 64,
                            options = opts, hessian = true, device = true)
        FM.fmm!((sysd, probes_d), (sysd,), cd_; scalar_potential = false, gradient = true, hessian = (true, false))
        # `fmm!` sums extra targets all-pairs on the host, so `probes_h` is the
        # exact sum, not the grid path; the like-for-like oracle for the
        # device grid path is the host grid path on the host cache's own state.
        gh_exact = reduce(hcat, probes_h.gradient); gd = reduce(hcat, probes_d.gradient)
        gh_tree = zeros(DTF, 4, nt)
        FM._host_extra_targets_tree!(gh_tree, ch.state, xt, Val(false))
        e = relerr(gd, gh_tree[2:4, :])
        tol = DTF === Float32 ? 3e-4 : 1e-8      # summation order differs between the two
        check(e <= tol, @sprintf("device tree targets match the host grid path (%.2e, tol %.0e)", e, tol))
        ex = relerr(gd, gh_exact)
        check(ex <= 2e-3, @sprintf("device tree targets vs the exact sum %.2e (tol 2e-03)", ex))

        # 5. the all-pairs direct arm runs no lifecycle, so its probes must not
        # read local expansions or near lists (they are stale there); they are
        # summed all-pairs and must match the exact host sum to roundoff
        sysa = mk(); probes_a = mkp()
        ca = RadixFMMCache(sysa; expansion_order = P, ell = ell, window_classes = 64,
                           options = opts, hessian = true, device = true)
        FM.set_radix_setting!(:RADIX_DIRECT_ARM, true)
        try
            FM.fmm!((sysa, probes_a), (sysa,), ca; scalar_potential = false, gradient = true, hessian = (true, false))
        finally
            FM.set_radix_setting!(:RADIX_DIRECT_ARM, false)
        end
        ga = reduce(hcat, probes_a.gradient)
        sysr = mk(); probes_r = mkp()
        FM.direct!((probes_r,), (sysr,); gradient = true)      # generic exact sum, the system's own kernel
        ea = relerr(ga, reduce(hcat, probes_r.gradient))
        tol_a = 5e-4                                            # the partitioned kernel's regularization tail
        check(ea <= tol_a, @sprintf("direct-arm probes are the all-pairs sum (%.2e, tol %.0e)", ea, tol_a))
    end
end

@printf("\n%d passed, %d failed\n", npass[], nfail[])
nfail[] == 0 || error("$(nfail[]) check(s) failed")
println("gate passed: extra targets evaluated through the resident grid")
