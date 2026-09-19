# Are two identical device steps bit-identical? Particles (de-permuted) and
# probes, with and without the tree-carried extra source.
using FastMultipole, Random, Printf, LinearAlgebra
using FastMultipole.StaticArrays
const FM = FastMultipole
include(joinpath(@__DIR__, "..", "vortex.jl"))
include(joinpath(@__DIR__, "ka_extra_tree_correctness_systems.jl"))
include(joinpath(@__DIR__, "ka_backend.jl"))
dev_functional() || (println("no device"); exit(0))
ext = Base.get_extension(FastMultipole, :FastMultipoleKAExt)
let n = 20000, nt = 300, ns = 200, P = 4, ell = 4, DTF = Float32
    Random.seed!(1); pos = DTF.(rand(3, n)); str = DTF.(randn(3, n) ./ n)
    mk() = VortexParticles(copy(pos), copy(str), fill(DTF(0.01), n);
        potential = zeros(DTF, 13, n), gradient_stretching = zeros(DTF, 6, n))
    Random.seed!(2); r1 = [SVector{3,DTF}(rand(3)) for _ in 1:ns]
    ex = Segs(r1, [r1[i] + SVector{3,DTF}(0.01 .* randn(3)) for i in 1:ns], DTF.(randn(ns) ./ ns), fill(DTF(0.005), ns))
    Random.seed!(3); xt = DTF.(rand(3, nt))
    mkp() = (p = FM.ProbeSystem(nt, DTF); for i in 1:nt; p.position[i] = SVector{3,DTF}(xt[:, i]); end; p)
    opts = FM.CUDARadixLifecycleOptions(; precision = DTF, m2l_strategy = FM.ConcatenatedFixedZM2L(), body_type = FM.Point{FM.Vortex})
    FM.device_backend(::VortexParticles) = DEV_BACKEND
    function run(tree)
        sys = mk(); pr = mkp()
        c = RadixFMMCache(sys; expansion_order = P, ell = ell, window_classes = 64, options = opts, hessian = true, device = true)
        sw = FM.DerivativesSwitch(false, true, true, (sys,))
        ext.ka_radix_cache_device_step!(c, (sys,), sw; extra_targets = (pr,), extra_target_switches = FM.DerivativesSwitch(false, true, false, (pr,)),
                                        extra_tree_sources = tree ? (ex,) : ())
        st = c.state; nb = Int(st.counts.n_bodies); out = Array(st.output); inv = Array(st.grid.invperm)
        u = [out[2:4, inv[i]] for i in 1:nb]
        return u, copy(pr.gradient)
    end
    for tree in (false, true)
        u1, g1 = run(tree); u2, g2 = run(tree)
        du = maximum(maximum(abs.(a .- b)) for (a, b) in zip(u1, u2)); dg = maximum(maximum(abs.(a .- b)) for (a, b) in zip(g1, g2))
        @printf("tree extra source=%s: particles max |diff| %.1e   probes max |diff| %.1e\n", tree, du, dg)
    end
end
