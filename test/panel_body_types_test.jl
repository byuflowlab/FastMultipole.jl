# Planar triangular panels on the resident radix lifecycle (host): the pair
# kernels against triangle quadrature of the point kernels, and the far field
# built by the panel B2M against the self-excluded all-pairs sum.
using LinearAlgebra, Random
using FastMultipole.StaticArrays

struct PanelBodies{TF,BT}
    data::Matrix{TF}           # [x y z r strength... v1(3) v2(3) v3(3)] per column
    potential::Matrix{TF}      # 13 x n: u, g(3), h(9)
end
PanelBodies{BT}(data::Matrix{TF}) where {TF,BT} = PanelBodies{TF,BT}(data, zeros(TF, 13, size(data, 2)))
Base.eltype(::PanelBodies{TF}) where TF = TF
FastMultipole.get_n_bodies(s::PanelBodies) = size(s.data, 2)
FastMultipole.data_per_body(s::PanelBodies) = size(s.data, 1)
FastMultipole.strength_dims(::PanelBodies{TF,BT}) where {TF,BT} = FastMultipole.element_strength_dims(BT)
FastMultipole.get_position(s::PanelBodies{TF}, i) where TF = SVector{3,TF}(s.data[1, i], s.data[2, i], s.data[3, i])
FastMultipole.body_type(::PanelBodies{TF,BT}) where {TF,BT} = BT
FastMultipole.has_vector_potential(::PanelBodies{TF,BT}) where {TF,BT} = BT <: FastMultipole.Panel{3,FastMultipole.Vortex}
FastMultipole.source_system_to_buffer!(buffer, i_buffer, s::PanelBodies, i_body) =
    (buffer[1:size(s.data, 1), i_buffer] .= view(s.data, :, i_body))
function FastMultipole.buffer_to_target!(s::PanelBodies, buffer, switch, sort_index)
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

# packed panel: centroid, radius = farthest vertex from the centroid, strength, v1, v2, v3
function pack_panels(v1s, v2s, v3s, strengths)
    n = size(v1s, 2); sd = size(strengths, 1)
    data = zeros(4 + sd + 9, n)
    for i in 1:n
        c = (v1s[:, i] .+ v2s[:, i] .+ v3s[:, i]) ./ 3
        data[1:3, i] .= c
        data[4, i] = maximum(norm(v .- c) for v in (v1s[:, i], v2s[:, i], v3s[:, i]))
        data[5:4+sd, i] .= strengths[:, i]
        data[5+sd:7+sd, i] .= v1s[:, i]; data[8+sd:10+sd, i] .= v2s[:, i]; data[11+sd:13+sd, i] .= v3s[:, i]
    end
    return data
end
pair_sum(kernel, xt, bodies) = (out = zeros(eltype(bodies), 13, size(xt, 2));
    FastMultipole._host_extra_targets_from_main!(out, kernel, xt, bodies, size(bodies, 2), Val(true)); out)
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
# product Gauss-Legendre quadrature over a triangle (collapsed square), point bodies at the nodes
function triangle_quadrature(point_kernel, xt, v1, v2, v3, strength_per_area; nq = 20)
    nodes, weights = let
        J = zeros(nq, nq); for i in 1:nq-1; J[i, i+1] = J[i+1, i] = i / sqrt(4i^2 - 1); end
        e = eigen(Symmetric(J)); (e.values, 2 .* e.vectors[1, :] .^ 2)
    end
    area = norm(cross(v2 .- v1, v3 .- v1)) / 2
    sd = length(strength_per_area)
    bodies = zeros(4 + sd, nq * nq); k = 0
    for (a, wa) in zip(nodes, weights), (b, wb) in zip(nodes, weights)
        s = (a + 1) / 2; t = (b + 1) / 2          # collapse: xi = s, eta = t (1 - s)
        xi = s; eta = t * (1 - s); jac = (1 - s) / 4
        k += 1
        bodies[1:3, k] .= v1 .+ xi .* (v2 .- v1) .+ eta .* (v3 .- v1)
        bodies[5:4+sd, k] .= strength_per_area .* (wa * wb * jac * 2 * area)
    end
    return pair_sum(point_kernel, xt, bodies)
end

@testset "planar triangular panels on the resident lifecycle" begin
    Random.seed!(20260925)
    v1 = [0.1, 0.2, 0.3]; v2 = [0.35, 0.15, 0.5]; v3 = [0.2, 0.45, 0.4]
    nrm = cross(v2 .- v1, v3 .- v1); nrm ./= norm(nrm)
    xt = [0.9 0.2 -0.4; 0.1 0.8 0.3; 0.7 -0.2 0.9]

    # --- source panel: closed form vs quadrature of the point source ---
    q = 0.7
    bp = pack_panels(reshape(v1, 3, 1), reshape(v2, 3, 1), reshape(v3, 3, 1), [q;;])
    a = pair_sum(SourcePanelKernel(), xt, bp)
    b = triangle_quadrature(SingularSource(), xt, v1, v2, v3, [q])
    @test maximum(abs.(a[1, :] .- b[1, :])) < 1e-8 * maximum(abs.(b[1, :]))
    @test maximum(abs.(a[2:4, :] .- b[2:4, :])) < 1e-8 * maximum(abs.(b[2:4, :]))
    @test maximum(abs.(a[5:13, :] .- b[5:13, :])) < 1e-6 * maximum(abs.(b[5:13, :]))

    # --- dipole panel: closed form vs quadrature of the point dipole along the normal ---
    mu = 0.4
    bp = pack_panels(reshape(v1, 3, 1), reshape(v2, 3, 1), reshape(v3, 3, 1), [mu;;])
    a = pair_sum(DipolePanelKernel(), xt, bp)
    b = triangle_quadrature(SingularDipole(), xt, v1, v2, v3, mu .* nrm)
    @test maximum(abs.(a[1, :] .- b[1, :])) < 1e-8 * maximum(abs.(b[1, :]))
    @test maximum(abs.(a[2:4, :] .- b[2:4, :])) < 1e-8 * maximum(abs.(b[2:4, :]))
    @test maximum(abs.(a[5:13, :] .- b[5:13, :])) < 1e-6 * maximum(abs.(b[5:13, :]))

    # --- source-dipole panel: the sum ---
    bp = pack_panels(reshape(v1, 3, 1), reshape(v2, 3, 1), reshape(v3, 3, 1), [q; mu;;])
    a = pair_sum(SourceDipolePanelKernel(), xt, bp)
    b = triangle_quadrature(SingularSource(), xt, v1, v2, v3, [q]) .+ triangle_quadrature(SingularDipole(), xt, v1, v2, v3, mu .* nrm)
    @test maximum(abs.(a .- b)) < 1e-6 * maximum(abs.(b))

    # --- vortex sheet: 7-point rule vs a fine quadrature of the point vortex, far targets ---
    G = [0.3, -0.2, 0.5]; G .-= (G' * nrm) .* nrm       # in-plane sheet vorticity
    bp = pack_panels(reshape(v1, 3, 1), reshape(v2, 3, 1), reshape(v3, 3, 1), reshape(G, 3, 1))
    a = pair_sum(VortexSheetPanelKernel(), xt, bp)
    b = triangle_quadrature(SingularVortex(), xt, v1, v2, v3, G)
    @test maximum(abs.(a[2:4, :] .- b[2:4, :])) < 2e-3 * maximum(abs.(b[2:4, :]))
    @test maximum(abs.(a[5:13, :] .- b[5:13, :])) < 1e-2 * maximum(abs.(b[5:13, :]))

    # --- resident lifecycle (host) vs all-pairs, each panel type ---
    n = 1000
    cs = rand(3, n)
    e1 = randn(3, n); e2 = randn(3, n)
    e1 ./= sqrt.(sum(e1 .^ 2; dims = 1)); e2 ./= sqrt.(sum(e2 .^ 2; dims = 1))
    h = 0.003
    v1s = cs .- h .* e1; v2s = cs .+ h .* e1; v3s = cs .+ h .* e2
    areas = [norm(cross(v2s[:, i] .- v1s[:, i], v3s[:, i] .- v1s[:, i])) / 2 for i in 1:n]
    normals = reduce(hcat, [normalize(cross(v2s[:, i] .- v1s[:, i], v3s[:, i] .- v1s[:, i])) for i in 1:n])
    Gs = randn(3, n); Gs .-= sum(Gs .* normals; dims = 1) .* normals; Gs ./= (n .* areas')
    for (label, BT, strengths, kernel, lh) in (
            ("source", FastMultipole.Panel{3,FastMultipole.Source}, (rand(1, n) ./ n) ./ areas', SourcePanelKernel(), false),
            ("dipole", FastMultipole.Panel{3,FastMultipole.Dipole}, ((rand(1, n) .- 0.5) ./ n) ./ areas', DipolePanelKernel(), false),
            ("source-dipole", FastMultipole.Panel{3,FastMultipole.SourceDipole}, vcat(rand(1, n) ./ n, (rand(1, n) .- 0.5) ./ n) ./ areas', SourceDipolePanelKernel(), false),
            ("vortex sheet", FastMultipole.Panel{3,FastMultipole.Vortex}, Gs, VortexSheetPanelKernel(), true))
        data = pack_panels(v1s, v2s, v3s, strengths)
        sys = PanelBodies{BT}(data)
        ref = self_excluded_sum(kernel, data)
        cache = RadixFMMCache(sys; expansion_order = 8, ell = 3, hessian = true)
        @test typeof(cache.state.options.direct_kernel) == typeof(kernel)
        fmm!(sys, cache; scalar_potential = true, gradient = true, hessian = true)
        lh || @test maximum(abs.(sys.potential[1, :] .- ref[1, :])) < 5e-5 * maximum(abs.(ref[1, :]))
        @test maximum(abs.(sys.potential[2:4, :] .- ref[2:4, :])) < 5e-4 * maximum(abs.(ref[2:4, :]))
        @test maximum(abs.(sys.potential[5:13, :] .- ref[5:13, :])) < 1e-2 * maximum(abs.(ref[5:13, :]))
    end
end

# Float32 panels on the resident path: the closed forms carry guards scaled at
# 1e-12 relative (inert below Float64), so check that a Float32 evaluation of
# the pair kernels tracks Float64 for targets in general position, on the panel
# plane outside the triangle, and near the plane on both sides.
@testset "panel pair kernels in Float32" begin
    Random.seed!(11)
    v1 = [0.0, 0.0, 0.0]; v2 = [0.01, 0.002, 0.0]; v3 = [0.003, 0.009, 0.0]
    c = (v1 .+ v2 .+ v3) ./ 3
    targets = ([0.004, 0.003, 0.006], [0.03, -0.01, 0.0], [-0.02, 0.05, 0.0],
               c .+ [0.0, 0.0, 1e-4], c .- [0.0, 0.0, 1e-4], [0.02, 0.01, 1e-6])
    for (label, kernel, strengths) in (("source", SourcePanelKernel(), reshape([1.0], 1, 1)),
                                       ("dipole", DipolePanelKernel(), reshape([1.0], 1, 1)),
                                       ("source-dipole", SourceDipolePanelKernel(), reshape([1.0, -0.5], 2, 1)),
                                       ("vortex sheet", VortexSheetPanelKernel(), reshape([1.0, 0.5, 0.0], 3, 1)))
        d64 = pack_panels(reshape(v1, 3, 1), reshape(v2, 3, 1), reshape(v3, 3, 1), strengths)
        d32 = Float32.(d64)
        for xt in targets
            dx, dy, dz = xt .- d64[1:3, 1]
            r2 = dx * dx + dy * dy + dz * dz
            v64 = FastMultipole._direct_pair_ugh(kernel, dx, dy, dz, r2, inv(sqrt(r2)), d64, 1)
            v32 = FastMultipole._direct_pair_ugh(kernel, Float32(dx), Float32(dy), Float32(dz), Float32(r2), inv(sqrt(Float32(r2))), d32, 1)
            @test all(isfinite, v32)
            scale = max(maximum(abs, v64[2:4]), 1e-30)
            @test maximum(abs.(v32[2:4] .- v64[2:4])) <= 2e-3 * scale     # velocity, 2e-3 relative
            kernel isa VortexSheetPanelKernel || @test abs(v32[1] - v64[1]) <= 2e-3 * max(abs(v64[1]), 1e-30)
        end
    end
end

# Dunavant rules: exact for monomials up to their degree (1, 5, 7) on the
# reference triangle, where ∫ x^a y^b = a! b! / (a+b+2)!; the vortex sheet
# kernel at order 3 agrees with order 2 to expansion accuracy away from the panel
@testset "Dunavant rules and vortex sheet order 3" begin
    for (order, degree) in ((1, 1), (2, 5), (3, 7))
        rule = FastMultipole._dunavant(Val(order), Float64)
        @test isapprox(sum(w for (_, w) in rule), 1.0; atol = 1e-14)
        for a in 0:degree, b in 0:(degree - a)
            exact = factorial(a) * factorial(b) / factorial(a + b + 2)
            # barycentric (l1, l2, l3) -> (x, y) = (l2, l3) on the reference triangle, area 1/2
            q = 0.5 * sum(w * lam[2]^a * lam[3]^b for (lam, w) in rule)
            @test isapprox(q, exact; atol = 1e-13)
        end
    end
    v1 = [0.0, 0.0, 0.0]; v2 = [0.01, 0.002, 0.0]; v3 = [0.003, 0.009, 0.0]
    data = pack_panels(reshape(v1, 3, 1), reshape(v2, 3, 1), reshape(v3, 3, 1), reshape([1.0, 0.5, 0.0], 3, 1))
    for xt in ([0.05, 0.02, 0.03], [0.004, 0.003, 0.02])
        dx, dy, dz = xt .- data[1:3, 1]; r2 = dx * dx + dy * dy + dz * dz
        v2q = FastMultipole._direct_pair_ugh(VortexSheetPanelKernel(; order = 2), dx, dy, dz, r2, inv(sqrt(r2)), data, 1)
        v3q = FastMultipole._direct_pair_ugh(VortexSheetPanelKernel(; order = 3), dx, dy, dz, r2, inv(sqrt(r2)), data, 1)
        @test maximum(abs.(v3q[2:13] .- v2q[2:13])) <= 1e-4 * maximum(abs.(v2q[2:13]))   # order-2 error at ~2 panel sizes
    end
end
