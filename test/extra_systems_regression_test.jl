#------- extra source / target systems on the host radix path -------#
#
# Regression tests for tree-carried extra sources (`fmm!(...; tree_sources)`)
# and extra targets on a RadixFMMCache: far-field sign, Lamb-Helmholtz
# refusal, the two-pass kernel on extra targets, regularization reach in the
# binning, and contract-only (non-AbstractDirectKernel) source functors.

using FastMultipole
using FastMultipole.StaticArrays
using LinearAlgebra
using Random
using Test

# packed rows: 1:3 position, 4 radius, 5:(4 + sdims) strength, then any extra
# state (the core size of a regularized vortex sits in row 8)
struct XSRBodies{BT,TF,K}
    data::Matrix{TF}
    sdims::Int
    kernel::K
end
XSRBodies{BT}(data::Matrix{TF}, sdims, kernel=nothing) where {BT,TF} =
    XSRBodies{BT,TF,typeof(kernel)}(data, sdims, kernel)

Base.eltype(::XSRBodies{BT,TF}) where {BT,TF} = TF
FastMultipole.get_n_bodies(s::XSRBodies) = size(s.data, 2)
FastMultipole.data_per_body(s::XSRBodies) = size(s.data, 1)
FastMultipole.strength_dims(s::XSRBodies) = s.sdims
FastMultipole.get_position(s::XSRBodies{BT,TF}, i) where {BT,TF} =
    SVector{3,TF}(s.data[1, i], s.data[2, i], s.data[3, i])
FastMultipole.body_type(::XSRBodies{BT}) where BT = BT
FastMultipole.has_vector_potential(::XSRBodies{BT}) where BT =
    BT <: FastMultipole.Point{<:Union{FastMultipole.Vortex,FastMultipole.SourceVortex}}
FastMultipole.direct_kernel(s::XSRBodies{BT,TF,Nothing}) where {BT,TF} =
    FastMultipole._default_direct_kernel(BT)
FastMultipole.direct_kernel(s::XSRBodies) = s.kernel
function FastMultipole.source_system_to_buffer!(buffer, i_buffer, s::XSRBodies, i_body)
    buffer[1:size(s.data, 1), i_buffer] .= view(s.data, :, i_body)
end

# evaluation points for the extra-target path
struct XSRProbes{TF}
    x::Matrix{TF}
    out::Matrix{TF}   # 13 x n: u, gradient, hessian
end
XSRProbes(x::Matrix{TF}) where TF = XSRProbes{TF}(x, zeros(TF, 13, size(x, 2)))
FastMultipole.get_n_bodies(s::XSRProbes) = size(s.x, 2)
FastMultipole.get_position(s::XSRProbes{TF}, i) where TF =
    SVector{3,TF}(s.x[1, i], s.x[2, i], s.x[3, i])
function FastMultipole.buffer_to_target!(s::XSRProbes, buffer, switch, sort_index)
    g = FastMultipole.gradient_range(switch)
    for (k, i) in enumerate(sort_index)
        isempty(g) || (s.out[2:4, i] .= view(buffer, g, k))
    end
    return s
end

# a shipped pair kernel exposed through the extra-source contract only: an
# isbits functor with `_extra_pair_ug`, NOT an AbstractDirectKernel
struct XSRPair{K}
    inner::K
end
@inline function FastMultipole._extra_pair_ug(f::XSRPair, tx, ty, tz, b, j)
    T = typeof(tx)
    dx = tx - b[1, j]; dy = ty - b[2, j]; dz = tz - b[3, j]
    r2 = dx * dx + dy * dy + dz * dz
    r2 > zero(T) || return zero(T), zero(T), zero(T), zero(T)
    return FastMultipole._direct_pair_ug(f.inner, dx, dy, dz, r2, inv(sqrt(r2)), b, j)
end

function xsr_points(rng, n, sdims, extra_rows=0; scale=1.0)
    data = zeros(5 + sdims - 1 + extra_rows, n)
    data[1:3, :] .= rand(rng, 3, n)
    data[4, :] .= 1e-4
    data[5:(4 + sdims), :] .= scale .* (rand(rng, sdims, n) .- 0.5) ./ n
    return data
end

# (tree, all-pairs reference, near-only baseline) contributions of `ex` to the
# resident bodies, each with the resident self field subtracted
function xsr_tree_vs_direct(cache, main, ex)
    FM = FastMultipole
    st = cache.state
    grab() = copy(st.output[1:4, 1:Int(st.counts.n_bodies)])
    FM.update_radix_state!(cache, (main,)); FM.run_host_radix_lifecycle!(st)
    base = grab()
    FM.update_radix_state!(cache, (main,)); FM.run_host_radix_lifecycle!(st)
    FM._radix_extra_sources_into_output!(st, (ex,))
    ref = grab() .- base
    FM.update_radix_state!(cache, (main,))
    FM.run_host_radix_lifecycle_with_extra_tree!(st, (ex,))
    tree = grab() .- base
    FM.update_radix_state!(cache, (main,)); FM.run_host_radix_lifecycle!(st)
    n = Int(st.counts.n_bodies)
    binned, loose = FM.bin_resident_extra_source(Float64, ex, st.grid, Int(st.counts.n_cells))
    k = FM.direct_kernel(ex)
    FM.resident_extra_near!(st, binned, k)
    FM._host_targets_from_extra_source!(st.output, k, st.source_bodies, n, loose, Val(false))
    near = grab() .- base
    return tree, ref, near, size(loose, 2)
end

function xsr_check_far_field(tree, ref, near; rows=2:4)
    t = tree[rows, :]; r = ref[rows, :]; nr = near[rows, :]
    s = maximum(abs, r)
    err_tree = maximum(abs.(t .- r)) / s
    err_near = maximum(abs.(nr .- r)) / s
    far_ref = vec(r .- nr); far_tree = vec(t .- nr)
    cosine = dot(far_ref, far_tree) / (norm(far_ref) * norm(far_tree))
    return err_tree, err_near, cosine
end

@testset "extra systems: tree-carried sources match direct" begin
    FM = FastMultipole
    n, ns = 2000, 300
    cases = (
        ("Point{Source} on a source cache", FM.Point{FM.Source}, 1,
            FM.Point{FM.Source}, 1, FM.SingularSource(), false),
        ("Point{Dipole} on a dipole cache", FM.Point{FM.Dipole}, 3,
            FM.Point{FM.Dipole}, 3, FM.SingularDipole(), false),
        ("Point{Vortex} on a vortex cache", FM.Point{FM.Vortex}, 3,
            FM.Point{FM.Vortex}, 3, FM.SingularVortex(), true),
        ("Point{Source} on a vortex cache", FM.Point{FM.Vortex}, 3,
            FM.Point{FM.Source}, 1, FM.SingularSource(), true),
    )
    for (label, MBT, msd, EBT, esd, inner, LH) in cases
        @testset "$label" begin
            rng = MersenneTwister(20260929)
            main = XSRBodies{MBT}(xsr_points(rng, n, msd), msd)
            ex = XSRBodies{EBT}(xsr_points(rng, ns, esd), esd, XSRPair(inner))
            cache = RadixFMMCache(main; expansion_order=8, ell=3, lamb_helmholtz=LH)
            tree, ref, near, n_loose = xsr_tree_vs_direct(cache, main, ex)
            err_tree, err_near, cosine = xsr_check_far_field(tree, ref, near)
            @test n_loose < ns ÷ 10   # most bodies ride the tree
            @test cosine > 0.99
            @test err_tree < err_near
            @test err_tree < 1e-3
            if EBT <: FM.Point{FM.Source}
                # the near sweep writes the potential row like the all-pairs path
                pe_tree, pe_near, pcos = xsr_check_far_field(tree, ref, near; rows=1:1)
                @test pcos > 0.99
                @test pe_tree < pe_near
            end
        end
    end
end

@testset "extra systems: vortex tree source needs Lamb-Helmholtz" begin
    FM = FastMultipole
    rng = MersenneTwister(1)
    main = XSRBodies{FM.Point{FM.Source}}(xsr_points(rng, 200, 1), 1)
    ex = XSRBodies{FM.Point{FM.Vortex}}(xsr_points(rng, 20, 3), 3, XSRPair(FM.SingularVortex()))
    cache = RadixFMMCache(main; expansion_order=4, ell=2, lamb_helmholtz=false)
    FM.update_radix_state!(cache, (main,))
    @test_throws ArgumentError FM.run_host_radix_lifecycle_with_extra_tree!(cache.state, (ex,))
end

@testset "extra systems: regularized tree body wider than the near set" begin
    FM = FastMultipole
    rng = MersenneTwister(7)
    n, ns = 1500, 40
    main = XSRBodies{FM.Point{FM.Vortex}}(xsr_points(rng, n, 3), 3)
    edata = xsr_points(rng, ns, 3, 1)
    # rho_t * sigma ~ 0.48: several leaf widths at ell = 3 on the unit box
    edata[8, :] .= 0.1
    ex = XSRBodies{FM.Point{FM.Vortex}}(edata, 3, FM.RegularizedVortex(; sigma_row=8))
    cache = RadixFMMCache(main; expansion_order=8, ell=3)
    tree, ref, near, n_loose = xsr_tree_vs_direct(cache, main, ex)
    @test n_loose == ns
    @test maximum(abs.(tree[2:4, :] .- ref[2:4, :])) / maximum(abs, ref[2:4, :]) < 1e-12
end

@testset "extra systems: two-pass extra target equals the resident body" begin
    FM = FastMultipole
    rng = MersenneTwister(11)
    n = 400
    data = xsr_points(rng, n, 3, 1; scale=100.0)
    data[8, :] .= 0.04 .+ 0.01 .* rand(rng, n)
    main = XSRBodies{FM.Point{FM.Vortex}}(data, 3, FM.TwoPassVortex(; sigma_row=8))
    cache = RadixFMMCache(main; expansion_order=16, ell=2)
    st = cache.state
    FM.update_radix_state!(cache, (main,)); FM.run_host_radix_lifecycle!(st)
    nb = Int(st.counts.n_bodies)
    resident = copy(st.output[2:4, 1:nb])
    probes = XSRProbes(copy(st.source_bodies[1:3, 1:nb]))
    FM._radix_extra_targets_evaluate!(st, (probes,), (DerivativesSwitch(false, true, false),))
    err = maximum(abs.(probes.out[2:4, :] .- resident)) / maximum(abs, resident)
    @test err < 1e-5
end
