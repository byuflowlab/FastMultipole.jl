# Extra source systems carried by the resident tree (src/resident_extra_tree.jl),
# device path: the device tree against the device all-direct arm, including
# sources-only (the particles are targets but not sources), whether the extra
# system arrives as `extra_sources` or as `extra_tree_sources`. The host path
# (slab layout, near-only exactness, far field carried) is test/extra_tree_test.jl.
using FastMultipole, Random, Printf, LinearAlgebra, Test
using FastMultipole.StaticArrays
const FM = FastMultipole
include(joinpath(@__DIR__, "..", "helpers", "vortex.jl"))

include(joinpath(@__DIR__, "..", "helpers", "extra_tree_test_systems.jl"))

const TF = Float64
npass = Ref(0); nfail = Ref(0)
check(ok, msg) = (ok ? (npass[] += 1) : (nfail[] += 1); println(ok ? "  PASS  $msg" : "  FAIL  $msg"))

println("the device path against the verified host path")
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
        opts = FM.RadixLifecycleOptions(; precision = DTF,
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

        # 5. the all-pairs direct arm must carry tree sources too: it dropped
        # them silently before 2026-09-26
        FM.set_radix_settings!((; RADIX_DIRECT_ARM = true))
        try
            da_base = run((;)); da_tree = run((; extra_tree_sources = (ex,))) .- da_base
            e3 = maximum(abs.(da_tree[V, :] .- ref[V, :])) / maximum(abs, ref[V, :])
            check(e3 <= tol, @sprintf("direct arm carries tree sources (%.2e, tol %.0e)", e3, tol))
        finally
            FM.set_radix_settings!((; RADIX_DIRECT_ARM = false))
        end
    end
end

@printf("\n%d passed, %d failed\n", npass[], nfail[])
nfail[] == 0 || error("$(nfail[]) check(s) failed")
println("gate passed: extra sources carried by the resident tree")
