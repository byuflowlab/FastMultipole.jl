# direct_rectangular!: the generic interface, exercised through a kernel type
# defined here (as a consumer package would define its own).

using Test
using FastMultipole
using FastMultipole.StaticArrays
using FastMultipole.LinearAlgebra: norm
using ForwardDiff
import Random

# singular point source, rows x y z q: phi = q/(4π r), u = q d/(4π r³), d = target − source
struct RectTestSource <: AbstractRectangularKernel end
FastMultipole.rect_source_rows(::RectTestSource) = 4
FastMultipole.rect_has_potential(::RectTestSource) = true
FastMultipole.rect_check_sources(::RectTestSource, sources; scalar_potential=false) =
    all(isfinite, view(sources, 4, :)) || throw(ArgumentError("non-finite source strength"))

@inline function FastMultipole.rect_pair(::RectTestSource, target::SVector{3,T}, sources, q,
        ::Val{GRAD}, ::Val{POT}) where {T,GRAD,POT}
    @inbounds d = target - SVector{3,T}(sources[1, q], sources[2, q], sources[3, q])
    @inbounds s = sources[4, q]
    r2 = d[1]*d[1] + d[2]*d[2] + d[3]*d[3]
    iszero(r2) && return zero(SVector{3,T}), zero(SMatrix{3,3,T,9}), zero(T)
    r = sqrt(r2)
    c = s / (4 * T(pi) * r2 * r)
    u = c * d
    g = GRAD ? c * (SMatrix{3,3,T,9}(1, 0, 0, 0, 1, 0, 0, 0, 1) - 3 * d * transpose(d) / r2) :
        zero(SMatrix{3,3,T,9})
    p = POT ? s / (4 * T(pi) * r) : zero(T)
    return u, g, p
end

function _rect_test_velocity(x, sources)
    u = zero(x)
    for q in axes(sources, 2)
        d = x - sources[1:3, q]
        u += sources[4, q] * d / (4pi * norm(d)^3)
    end
    return u
end

@testset "direct rectangular: generic interface" begin
    Random.seed!(3901)
    n_src, n_tgt = 37, 23
    src = vcat(rand(3, n_src), rand(1, n_src) .- 0.5)
    tgt = vcat(rand(3, n_tgt) .+ 1.5, rand(2, n_tgt))   # extra target rows are ignored
    k = RectTestSource()

    out = zeros(13, n_tgt)
    direct_rectangular!(out, tgt, k, src; gradient=true, scalar_potential=true)
    for i in 1:n_tgt
        x = tgt[1:3, i]
        u = _rect_test_velocity(x, src)
        J = ForwardDiff.jacobian(y -> _rect_test_velocity(y, src), x)
        phi = sum(src[4, q] / (4pi * norm(x - src[1:3, q])) for q in 1:n_src)
        @test isapprox(out[1:3, i], u; rtol=1e-13)
        @test isapprox(out[4:12, i], vec(J); rtol=1e-12)   # out[3 + (j-1)*3 + i] = du_i/dx_j
        @test isapprox(out[13, i], phi; rtol=1e-13)
    end

    # velocity only, potential in row 4; results accumulate
    out2 = zeros(4, n_tgt)
    direct_rectangular!(out2, tgt, k, src; scalar_potential=true)
    @test out2[1:3, :] ≈ out[1:3, :] && out2[4, :] ≈ out[13, :]
    direct_rectangular!(out2, tgt, k, src; scalar_potential=true)
    @test out2 ≈ 2 .* vcat(out[1:3, :], out[13:13, :])

    # a coincident source contributes nothing, and Float32 runs end to end
    out3 = zeros(3, 1)
    direct_rectangular!(out3, src[1:3, 1:1], k, src[:, 1:1])
    @test out3 == zeros(3, 1)
    out32 = zeros(Float32, 12, n_tgt)
    direct_rectangular!(out32, Float32.(tgt), k, Float32.(src); gradient=true)
    @test isapprox(out32, out[1:12, :]; rtol=1e-4)
end

struct RectTestNoPotential <: AbstractRectangularKernel end
FastMultipole.rect_source_rows(::RectTestNoPotential) = 4

@testset "direct rectangular: argument validation" begin
    k = RectTestSource()
    src = rand(4, 3); tgt = rand(3, 5)
    @test_throws ArgumentError direct_rectangular!(zeros(3, 5), rand(2, 5), k, src)
    @test_throws ArgumentError direct_rectangular!(zeros(3, 5), tgt, k, rand(3, 3))
    @test_throws ArgumentError direct_rectangular!(zeros(3, 4), tgt, k, src)
    @test_throws ArgumentError direct_rectangular!(zeros(3, 5), tgt, k, src; gradient=true)
    @test_throws ArgumentError direct_rectangular!(zeros(12, 5), tgt, k, src; gradient=true, scalar_potential=true)
    @test_throws ArgumentError direct_rectangular!(zeros(4, 5), tgt, RectTestNoPotential(), src; scalar_potential=true)
    bad = copy(src); bad[4, 2] = NaN
    @test_throws ArgumentError direct_rectangular!(zeros(3, 5), tgt, k, bad)   # rect_check_sources
end
