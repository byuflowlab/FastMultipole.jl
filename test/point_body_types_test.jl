# Point{Dipole} and Point{SourceVortex} on the resident radix lifecycle (host):
# the far field built from the new B2M contributions must agree with the
# all-pairs sum of the matching pair kernel, and the dipole pair kernel must be
# the source-position derivative of the source kernel.
using LinearAlgebra, Random
using FastMultipole.StaticArrays

# minimal host systems: packed rows [x y z r strength...] kept as a matrix
struct PointBodies{TF,BT}
    data::Matrix{TF}          # data_per_body x n
    potential::Matrix{TF}     # 13 x n: u, g(3), h(9), the pair-sum layout
end
PointBodies{BT}(data::Matrix{TF}) where {TF,BT} = PointBodies{TF,BT}(data, zeros(TF, 13, size(data, 2)))
Base.eltype(::PointBodies{TF}) where TF = TF
FastMultipole.get_n_bodies(s::PointBodies) = size(s.data, 2)
FastMultipole.data_per_body(s::PointBodies) = size(s.data, 1)
FastMultipole.strength_dims(s::PointBodies) = size(s.data, 1) - 4
FastMultipole.get_position(s::PointBodies{TF}, i) where TF = SVector{3,TF}(s.data[1, i], s.data[2, i], s.data[3, i])
FastMultipole.body_type(::PointBodies{TF,BT}) where {TF,BT} = BT
FastMultipole.has_vector_potential(::PointBodies{TF,BT}) where {TF,BT} = BT <: FastMultipole.Point{FastMultipole.SourceVortex}
function FastMultipole.source_system_to_buffer!(buffer, i_buffer, s::PointBodies, i_body)
    buffer[1:size(s.data, 1), i_buffer] .= view(s.data, :, i_body)
end
function FastMultipole.buffer_to_target!(s::PointBodies, buffer, switch, sort_index)
    spi = FastMultipole.scalar_potential_index(switch)
    for (k, i) in enumerate(sort_index)
        spi > 0 && (s.potential[1, i] = buffer[spi, k])
        g = FastMultipole.gradient_range(switch)
        isempty(g) || (s.potential[2:4, i] .= view(buffer, g, k))
        h = FastMultipole.hessian_range(switch)
        isempty(h) || (s.potential[5:13, i] .= view(buffer, h, k))
    end
    return s
end

pair_sum(kernel, xt, bodies) = (out = zeros(eltype(bodies), 13, size(xt, 2));
    FastMultipole._host_extra_targets_from_main!(out, kernel, xt, bodies, size(bodies, 2), Val(true)); out)

@testset "point dipole and source-vortex bodies on the resident lifecycle" begin
    Random.seed!(20260925)
    n = 1500
    xs = rand(3, n); r = fill(1e-3, n)

    # --- the dipole pair kernel is the source-position derivative of the source kernel ---
    bodies_s = vcat(xs[:, 1:1], r[1:1]', [1.0;;])
    p = [0.3, -0.5, 0.8]; eps = 1e-6
    xt = rand(3, 5)
    plus = copy(bodies_s); plus[1:3, 1] .+= eps .* p
    minus = copy(bodies_s); minus[1:3, 1] .-= eps .* p
    fd = (pair_sum(SingularSource(), xt, plus) .- pair_sum(SingularSource(), xt, minus)) ./ (2eps)
    bodies_d = vcat(xs[:, 1:1], r[1:1]', p)
    dp = pair_sum(SingularDipole(), xt, bodies_d)
    @test maximum(abs.(dp[1, :] .- fd[1, :])) < 1e-6 * maximum(abs.(fd[1, :]))
    @test maximum(abs.(dp[2:4, :] .- fd[2:4, :])) < 1e-5 * maximum(abs.(fd[2:4, :]))
    @test maximum(abs.(dp[5:13, :] .- fd[5:13, :])) < 1e-4 * maximum(abs.(fd[5:13, :]))

    # --- dipoles: resident lifecycle (host) vs all-pairs ---
    data = vcat(xs, r', (rand(3, n) .- 0.5) ./ n)
    sys = PointBodies{FastMultipole.Point{FastMultipole.Dipole}}(data)
    ref = pair_sum(SingularDipole(), xs, data)
    cache = RadixFMMCache(sys; expansion_order=8, ell=3, hessian=true)
    @test cache.state.options.direct_kernel isa SingularDipole
    fmm!(sys, cache; scalar_potential=true, gradient=true, hessian=true)
    # self term excluded on both sides: the pair sum skips r2 == 0, the lifecycle too
    @test maximum(abs.(sys.potential[1, :] .- ref[1, :])) < 5e-5 * maximum(abs.(ref[1, :]))
    @test maximum(abs.(sys.potential[2:4, :] .- ref[2:4, :])) < 1e-4 * maximum(abs.(ref[2:4, :]))
    @test maximum(abs.(sys.potential[5:13, :] .- ref[5:13, :])) < 5e-3 * maximum(abs.(ref[5:13, :]))

    # --- source-vortex: resident lifecycle vs all-pairs, and vs source + vortex separately ---
    q = rand(n) ./ n; G = (rand(3, n) .- 0.5) ./ n
    data_sv = vcat(xs, r', q', G)
    sys_sv = PointBodies{FastMultipole.Point{FastMultipole.SourceVortex}}(data_sv)
    ref_sv = pair_sum(SingularSourceVortex(), xs, data_sv)
    ref_s = pair_sum(SingularSource(), xs, vcat(xs, r', q'))
    ref_v = pair_sum(SingularVortex(), xs, vcat(xs, r', G))
    @test maximum(abs.(ref_sv .- (ref_s .+ ref_v))) < 1e-12 * maximum(abs.(ref_sv))
    cache_sv = RadixFMMCache(sys_sv; expansion_order=8, ell=3, hessian=true)
    @test cache_sv.state.options.direct_kernel isa SingularSourceVortex
    fmm!(sys_sv, cache_sv; scalar_potential=true, gradient=true, hessian=true)
    # Under the Lamb-Helmholtz channel the resident local evaluation keeps only
    # the monopole term of the phi channel for the scalar potential (the phi
    # channel is not the physical potential once vortices are in it), so the
    # potential of a source-vortex body is not delivered; velocity and its
    # gradient are.
    @test maximum(abs.(sys_sv.potential[2:4, :] .- ref_sv[2:4, :])) < 1e-4 * maximum(abs.(ref_sv[2:4, :]))
    @test maximum(abs.(sys_sv.potential[5:13, :] .- ref_sv[5:13, :])) < 5e-3 * maximum(abs.(ref_sv[5:13, :]))
end
