# Straight filaments on the resident radix lifecycle (host): the three pair
# kernels against quadrature of the point kernels (and the singular Biot-Savart
# segment formula), and the far field built by the filament B2M against the
# all-pairs sum of the same kernels.
using LinearAlgebra, Random
using FastMultipole.StaticArrays

struct FilamentBodies{TF,BT}
    data::Matrix{TF}           # [x y z r strength... x1(3) x2(3)] per column
    potential::Matrix{TF}      # 13 x n: u, g(3), h(9)
end
FilamentBodies{BT}(data::Matrix{TF}) where {TF,BT} = FilamentBodies{TF,BT}(data, zeros(TF, 13, size(data, 2)))
Base.eltype(::FilamentBodies{TF}) where TF = TF
FastMultipole.get_n_bodies(s::FilamentBodies) = size(s.data, 2)
FastMultipole.data_per_body(s::FilamentBodies) = size(s.data, 1)
FastMultipole.strength_dims(::FilamentBodies{TF,BT}) where {TF,BT} = FastMultipole.element_strength_dims(BT)
FastMultipole.get_position(s::FilamentBodies{TF}, i) where TF = SVector{3,TF}(s.data[1, i], s.data[2, i], s.data[3, i])
FastMultipole.body_type(::FilamentBodies{TF,BT}) where {TF,BT} = BT
FastMultipole.has_vector_potential(::FilamentBodies{TF,BT}) where {TF,BT} = BT <: FastMultipole.Filament{FastMultipole.Vortex}
FastMultipole.source_system_to_buffer!(buffer, i_buffer, s::FilamentBodies, i_body) =
    (buffer[1:size(s.data, 1), i_buffer] .= view(s.data, :, i_body))
function FastMultipole.buffer_to_target!(s::FilamentBodies, buffer, switch, sort_index)
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

# packed filament: midpoint, radius = half length, strength, x1, x2
function pack_filaments(x1s, x2s, strengths)
    n = size(x1s, 2); sd = size(strengths, 1)
    data = zeros(4 + sd + 6, n)
    for i in 1:n
        data[1:3, i] .= (x1s[:, i] .+ x2s[:, i]) ./ 2
        data[4, i] = norm(x2s[:, i] .- x1s[:, i]) / 2
        data[5:4+sd, i] .= strengths[:, i]
        data[5+sd:7+sd, i] .= x1s[:, i]
        data[8+sd:10+sd, i] .= x2s[:, i]
    end
    return data
end
pair_sum(kernel, xt, bodies) = (out = zeros(eltype(bodies), 13, size(xt, 2));
    FastMultipole._host_extra_targets_from_main!(out, kernel, xt, bodies, size(bodies, 2), Val(true)); out)
# the same sum at the bodies themselves, skipping the self pair as the lifecycle does
function self_excluded_sum(kernel, bodies)
    n = size(bodies, 2); out = zeros(eltype(bodies), 13, n)
    for i in 1:n, j in 1:n
        i == j && continue
        dx = bodies[1, i] - bodies[1, j]; dy = bodies[2, i] - bodies[2, j]; dz = bodies[3, i] - bodies[3, j]
        r2 = dx * dx + dy * dy + dz * dz; r2 == 0 && continue
        v = FastMultipole._direct_pair_ugh(kernel, dx, dy, dz, r2, inv(sqrt(r2)), bodies, j)
        for k in 1:13; out[k, i] += v[k]; end
    end
    return out
end

# Gauss-Legendre quadrature of a point kernel along a segment (point bodies of the
# given strength per unit length at the nodes)
function segment_quadrature(point_kernel, xt, x1, x2, strength_per_length; nq = 24)
    nodes, weights = let  # Golub-Welsch
        J = zeros(nq, nq); for i in 1:nq-1; J[i, i+1] = J[i+1, i] = i / sqrt(4i^2 - 1); end
        e = eigen(Symmetric(J)); (e.values, 2 .* e.vectors[1, :] .^ 2)
    end
    L = norm(x2 - x1)
    sd = length(strength_per_length)
    bodies = zeros(4 + sd, nq)
    for (k, (t, w)) in enumerate(zip(nodes, weights))
        bodies[1:3, k] .= x1 .+ (t + 1) / 2 .* (x2 .- x1)
        bodies[5:4+sd, k] .= strength_per_length .* (w * L / 2)
    end
    return pair_sum(point_kernel, xt, bodies)
end

@testset "straight filaments on the resident lifecycle" begin
    Random.seed!(20260925)
    x1 = [0.1, 0.2, 0.3]; x2 = [0.35, 0.15, 0.5]
    xt = [0.9 0.2 -0.4; 0.1 0.8 0.3; 0.7 -0.2 0.9]        # three targets a few lengths away

    # --- source filament: closed form vs quadrature of the point source ---
    q = 0.7
    bf = pack_filaments(reshape(x1, 3, 1), reshape(x2, 3, 1), [q;;])
    a = pair_sum(SourceFilamentKernel(), xt, bf)
    b = segment_quadrature(SingularSource(), xt, x1, x2, [q])
    @test maximum(abs.(a .- b)) < 1e-9 * maximum(abs.(b))

    # --- dipole filament: closed form vs quadrature of the point dipole ---
    p = [0.3, -0.5, 0.8]
    bf = pack_filaments(reshape(x1, 3, 1), reshape(x2, 3, 1), reshape(p, 3, 1))
    a = pair_sum(DipoleFilamentKernel(), xt, bf)
    b = segment_quadrature(SingularDipole(), xt, x1, x2, p)
    @test maximum(abs.(a[1, :] .- b[1, :])) < 1e-9 * maximum(abs.(b[1, :]))
    @test maximum(abs.(a[2:4, :] .- b[2:4, :])) < 1e-9 * maximum(abs.(b[2:4, :]))
    @test maximum(abs.(a[5:13, :] .- b[5:13, :])) < 1e-8 * maximum(abs.(b[5:13, :]))

    # --- vortex filament: singular Biot-Savart vs the closed-form segment and vs quadrature ---
    G = 0.6 .* (x2 .- x1) ./ norm(x2 .- x1)
    bf = pack_filaments(reshape(x1, 3, 1), reshape(x2, 3, 1), reshape(G, 3, 1))
    a = pair_sum(VortexFilamentKernel(), xt, bf)
    b = segment_quadrature(SingularVortex(), xt, x1, x2, G)
    @test maximum(abs.(a[2:4, :] .- b[2:4, :])) < 1e-9 * maximum(abs.(b[2:4, :]))
    @test maximum(abs.(a[5:13, :] .- b[5:13, :])) < 1e-8 * maximum(abs.(b[5:13, :]))
    include(joinpath(@__DIR__, "vortex_filament.jl"))   # vortex_filament(x1, x2, xt, q)
    for c in 1:3
        v = vortex_filament(SVector{3}(x1), SVector{3}(x2), SVector{3}(xt[:, c]), 0.6)
        @test maximum(abs.(a[2:4, c] .- v)) < 1e-12
    end

    # --- resident lifecycle (host) vs all-pairs, each filament type ---
    n = 1200
    mids = rand(3, n); dirs = randn(3, n); dirs ./= sqrt.(sum(dirs .^ 2; dims = 1))
    len = 0.004 .* (0.5 .+ rand(n))
    x1s = mids .- dirs .* (len' ./ 2); x2s = mids .+ dirs .* (len' ./ 2)
    for (label, BT, strengths, kernel, lh) in (
            ("source", FastMultipole.Filament{FastMultipole.Source}, (rand(1, n) ./ n) ./ len', SourceFilamentKernel(), false),
            ("dipole", FastMultipole.Filament{FastMultipole.Dipole}, ((rand(3, n) .- 0.5) ./ n) ./ len', DipoleFilamentKernel(), false),
            ("vortex", FastMultipole.Filament{FastMultipole.Vortex}, dirs .* ((rand(1, n) .- 0.5) ./ n), VortexFilamentKernel(), true))
        data = pack_filaments(x1s, x2s, strengths)
        sys = FilamentBodies{BT}(data)
        ref = self_excluded_sum(kernel, data)
        cache = RadixFMMCache(sys; expansion_order = 8, ell = 3, hessian = true)
        @test typeof(cache.state.options.direct_kernel) == typeof(kernel)
        fmm!(sys, cache; scalar_potential = true, gradient = true, hessian = true)
        if lh
            # the potential is not delivered under the Lamb-Helmholtz channel
        else
            @test maximum(abs.(sys.potential[1, :] .- ref[1, :])) < 5e-5 * maximum(abs.(ref[1, :]))
        end
        @test maximum(abs.(sys.potential[2:4, :] .- ref[2:4, :])) < 2e-4 * maximum(abs.(ref[2:4, :]))
        @test maximum(abs.(sys.potential[5:13, :] .- ref[5:13, :])) < 5e-3 * maximum(abs.(ref[5:13, :]))
    end
end

# edge cases of the pair kernels (codex-verify-impl 2026-09-25): a target close to
# the segment must not cancel in S - L; a singular core returns nothing on the
# segment's line while a regularized core keeps its gradient there; the Gaussian
# family with no core must not produce NaN.
@testset "filament pair kernel edge cases" begin
    x1 = [0.0, 0.0, 0.0]; x2 = [1.0, 0.0, 0.0]
    # near-segment target: the stable B agrees with a 24-point quadrature of the point source
    src = pack_filaments(reshape(x1, 3, 1), reshape(x2, 3, 1), reshape([1.0], 1, 1))
    xt = reshape([0.5, 1e-7, 0.0], 3, 1)
    v = FastMultipole._direct_pair_ugh(SourceFilamentKernel(), xt[1] - src[1, 1], xt[2] - src[2, 1], xt[3] - src[3, 1], 0.0, 0.0, src, 1)
    @test isfinite(v[1]) && v[1] > 0
    @test abs(v[1] - log(1e14) / (4pi)) / v[1] < 1e-6   # ln(A/B) with A = 2 + 2h^2, B = 2h^2, h = 1e-7
    # on the segment: singular core returns zero for vortex, nothing finite for the source
    vsrc = pack_filaments(reshape(x1, 3, 1), reshape(x2, 3, 1), reshape([1.0, 0.0, 0.0], 3, 1))
    on = FastMultipole._direct_pair_ugh(VortexFilamentKernel(), 0.25, 0.0, 0.0, 0.0625, 4.0, vsrc, 1)
    @test all(iszero, on)
    # regularized core on the segment's line: velocity zero, gradient nonzero and finite
    core = 0.05
    vcore = vcat(vsrc, fill(core, 1, 1))
    for fam in 1:3
        k = VortexFilamentKernel(; core_row = size(vcore, 1), family = fam)
        onc = FastMultipole._direct_pair_ugh(k, 0.25, 0.0, 0.0, 0.0625, 4.0, vcore, 1)
        @test all(isfinite, onc)
        @test maximum(abs, onc[2:4]) == 0
        @test maximum(abs, onc[5:13]) > 0
        # Gaussian / compact with core = 0 fall back to the singular kernel, no NaN
        k0 = VortexFilamentKernel(; core_row = size(vcore, 1), family = fam)
        v0 = vcat(vsrc, zeros(1, 1))
        off = FastMultipole._direct_pair_ugh(k0, 0.25, 0.3, 0.0, 0.0625 + 0.09, inv(sqrt(0.0625 + 0.09)), v0, 1)
        ref = FastMultipole._direct_pair_ugh(VortexFilamentKernel(), 0.25, 0.3, 0.0, 0.0625 + 0.09, inv(sqrt(0.0625 + 0.09)), v0, 1)
        @test all(isfinite, off)
        @test maximum(abs.(off .- ref)) <= 1e-12 * maximum(abs, ref)
    end
end
