# Debugging companion to ka_extra_tree_correctness.jl: run this when that gate
# fails. It sweeps the expansion order and the tree depth on a field large
# enough for the far field to dominate, and splits the error into its parts,
# which is what tells a wrong multipole from a wrong near/far split.
#
#   julia --project=test/metal_env test/metal_env/debug/extra_tree_convergence.jl
#
# What the numbers mean:
#   * "near only" exact at ell = 1: every cell pair is near, so the tree
#     carries nothing and any error is in the near sweep itself.
#   * error falling with the expansion order: the far field is genuinely
#     carried. An error that is FLAT in the order is not truncation -- if it
#     is about twice the far field, the source kernel's sign convention is
#     opposite FastMultipole's; if it equals the far field, the multipoles are
#     not reaching the tree at all (the 7-argument body_to_multipole! is a
#     fallback that warns and writes nothing: the element type must lead).
#   * "held out" counts the bodies too large for a cell or in a cell with no
#     resident body; those are summed directly and must not be forgotten.
using FastMultipole, Random, Printf, LinearAlgebra
using FastMultipole.StaticArrays
const FM = FastMultipole
include(joinpath(@__DIR__, "..", "..", "vortex.jl"))
include(joinpath(@__DIR__, "..", "ka_extra_tree_correctness_systems.jl"))

const TF = Float64
n = parse(Int, get(ENV, "N_BODIES", "4000"))
ns = parse(Int, get(ENV, "N_SEGMENTS", "400"))
println("$n particles, $ns segments\n")
@printf("%4s %4s %7s %7s %12s %12s %12s\n", "P", "ell", "cells", "heldout",
        "tree", "near only", "tree/near")
for (P, ell) in ((4, 1), (4, 2), (4, 3), (6, 3), (8, 3), (10, 3))
    Random.seed!(1)
    sys = VortexParticles(TF.(rand(3, n)), TF.(randn(3, n) ./ n), fill(TF(0.01), n);
        potential = zeros(TF, 13, n), gradient_stretching = zeros(TF, 6, n))
    Random.seed!(2)
    r1 = [SVector{3,TF}(rand(3)) for _ in 1:ns]
    ex = Segs(r1, [r1[i] + SVector{3,TF}(0.01 .* randn(3)) for i in 1:ns],
              TF.(randn(ns) ./ ns), fill(TF(0.005), ns))
    opts = FM.CUDARadixLifecycleOptions(; precision = TF,
        m2l_strategy = FM.ConcatenatedFixedZM2L(), body_type = FM.Point{FM.Vortex})
    cache = RadixFMMCache(sys; expansion_order = P, ell = ell, window_classes = 64,
                          options = opts, hessian = true)
    FM.update_radix_state!(cache, (sys,)); st = cache.state
    nc = Int(st.counts.n_cells); nb = Int(st.counts.n_bodies)
    g(o) = copy(view(Array(o), 2:4, 1:nb))
    FM.run_host_radix_lifecycle!(st); base = g(st.output)
    FM.update_radix_state!(cache, (sys,)); FM.run_host_radix_lifecycle!(st)
    FM._radix_extra_sources_into_output!(st, (ex,)); ref = g(st.output) .- base
    binned, loose = FM.bin_resident_extra_source(TF, ex, st.grid, nc)
    FM.update_radix_state!(cache, (sys,))
    FM.run_host_radix_lifecycle_with_extra_tree!(st, (ex,))
    tree = g(st.output) .- base
    FM.update_radix_state!(cache, (sys,)); FM.run_host_radix_lifecycle!(st)
    FM.resident_extra_near!(st, binned, FM.direct_kernel(ex))
    near = g(st.output) .- base
    scale = maximum(abs, ref)
    et = maximum(abs.(tree .- ref)) / scale
    en = maximum(abs.(near .- ref)) / scale
    @printf("%4d %4d %7d %7d %12.3e %12.3e %12.2f\n", P, ell, nc, size(loose, 2), et, en, et / en)
end
