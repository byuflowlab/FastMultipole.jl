# FastMultipole's own straight vortex filament (Filament{Vortex} body type,
# VortexFilamentKernel) as a radix extra source. Host only. Until 2026-09-26
# the near pairs of such a source threw (no _extra_pair_ug/_ugh methods).
#   1. singular core: the radix extra path equals the test segment kernel
#      (which is Biot-Savart at zero core) to roundoff off the segments' lines;
#   2. regularized core: the extra path equals a plain loop over the same pair
#      function (packing, dx convention, hessian rows), all finite;
#   3. the tree-carried path accepts the source (near pairs through the same
#      methods) and matches the all-direct extra path at a depth where every
#      pair is near.
using FastMultipole, Random, Printf, LinearAlgebra, Test
using FastMultipole.StaticArrays
const FM = FastMultipole
include(joinpath(@__DIR__, "..", "helpers", "vortex.jl"))
include(joinpath(@__DIR__, "..", "helpers", "extra_tree_test_systems.jl"))
const TF = Float64
npass = Ref(0); nfail = Ref(0)
check(ok, msg) = (ok ? (npass[] += 1) : (nfail[] += 1); println(ok ? "  PASS  $msg" : "  FAIL  $msg"))

# the same packing as Segs (rows 1:3 midpoint, 5:7 strength, 8:13 endpoints,
# 14 core), served by the built-in kernel instead of the test's
struct BuiltinSegs{TF}
    s::Segs{TF}
    core_row::Int
end
FM.get_n_bodies(b::BuiltinSegs) = FM.get_n_bodies(b.s)
FM.data_per_body(b::BuiltinSegs) = 15
FM.strength_dims(b::BuiltinSegs) = 3
FM.has_vector_potential(b::BuiltinSegs) = true
FM.body_type(b::BuiltinSegs) = FM.Filament{FM.Vortex}
FM.get_position(b::BuiltinSegs, i) = FM.get_position(b.s, i)
FM.source_system_to_buffer!(buf, ib, b::BuiltinSegs, i) = FM.source_system_to_buffer!(buf, ib, b.s, i)
FM.direct_kernel(b::BuiltinSegs) = FM.VortexFilamentKernel(; core_row = b.core_row, family = 1)

n = 600; ns = 80
Random.seed!(1)
sys0() = VortexParticles(TF.(rand(3, n)), TF.(randn(3, n) ./ n), fill(TF(0.01), n);
    potential = zeros(TF, 13, n), gradient_stretching = zeros(TF, 6, n))
Random.seed!(2)
r1 = [SVector{3,TF}(rand(3)) for _ in 1:ns]
r2 = [r1[i] + SVector{3,TF}(0.02 .* randn(3)) for i in 1:ns]
gam = TF.(randn(ns) ./ ns)
opts = FM.RadixLifecycleOptions(; precision = TF, m2l_strategy = FM.ConcatenatedFixedZM2L(), body_type = FM.Point{FM.Vortex})

function extra_field(ex; ell = 1, tree = false)
    Random.seed!(1); sys = sys0()
    cache = RadixFMMCache(sys; expansion_order = 4, ell, window_classes = 64, options = opts, hessian = true)
    FM.update_radix_state!(cache, (sys,)); st = cache.state
    nb = Int(st.counts.n_bodies)
    FM.run_host_radix_lifecycle!(st); base = copy(view(Array(st.output), 2:13, 1:nb))
    FM.update_radix_state!(cache, (sys,))
    if tree
        FM.run_host_radix_lifecycle_with_extra_tree!(st, (ex,))
    else
        FM.run_host_radix_lifecycle!(st); FM._radix_extra_sources_into_output!(st, (ex,))
    end
    return copy(view(Array(st.output), 2:13, 1:nb)) .- base, sys
end

println("1. singular core: built-in kernel against the segment kernel (Biot-Savart at zero core)")
let
    seg = Segs(r1, r2, gam, zeros(TF, ns))
    bi = BuiltinSegs(seg, 0)
    ref, _ = extra_field(seg); got, _ = extra_field(bi)
    V = 1:3
    e = maximum(abs.(got[V, :] .- ref[V, :])) / maximum(abs, ref[V, :])
    check(e <= 1e-10, @sprintf("velocity %.1e", e))
end

println("2. regularized core: the extra path equals a loop over the pair function; finite")
let
    seg = Segs(r1, r2, gam, fill(TF(0.01), ns))
    bi = BuiltinSegs(seg, 14)
    got, sys = extra_field(bi)
    kern = FM.direct_kernel(bi)
    buf = zeros(TF, 15, ns); for j in 1:ns; FM.source_system_to_buffer!(buf, j, bi, j); end
    ref = zeros(TF, 12, n)
    for i in 1:n
        x = sys.bodies[i].position
        for j in 1:ns
            dx = x[1] - buf[1, j]; dy = x[2] - buf[2, j]; dz = x[3] - buf[3, j]; rr = dx*dx + dy*dy + dz*dz
            v = FM._direct_pair_ugh(kern, dx, dy, dz, rr, inv(sqrt(rr)), buf, j)
            for k in 1:12; ref[k, i] += v[k + 1]; end
        end
    end
    # the radix output orders the particles by its sort; compare as multisets per row via sorting
    e = maximum(abs.(sort(got[1, :]) .- sort(ref[1, :]))) / maximum(abs, ref[1, :])
    eg = maximum(abs.(sort(vec(got[4:12, :])) .- sort(vec(ref[4:12, :])))) / maximum(abs, ref[4:12, :])
    check(all(isfinite, got), "all rows finite")
    check(e <= 1e-12 && eg <= 1e-12, @sprintf("velocity %.1e, gradient %.1e against the pair loop", e, eg))
end

println("3. tree-carried filament source accepted; equals all-direct where every pair is near")
let
    seg = Segs(r1, r2, gam, fill(TF(0.01), ns))
    bi = BuiltinSegs(seg, 14)
    direct, _ = extra_field(bi; ell = 1)
    tree, _ = extra_field(bi; ell = 1, tree = true)
    e = maximum(abs.(tree .- direct)) / maximum(abs, direct)
    check(e <= 1e-12, @sprintf("tree vs all-direct extra %.1e", e))
end
println(nfail[] == 0 ? "all $(npass[]) checks passed" : "FAILURES: $(nfail[])")
nfail[] == 0 || error("built-in filament extra source failed")
