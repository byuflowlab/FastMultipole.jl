using ForwardDiff
using LinearAlgebra
pushfirst!(LOAD_PATH, normpath(joinpath(@__DIR__, "..", "..")))
using FastMultipole
using FastMultipole.StaticArrays

const C = 1 / (4pi)
const PAIRS = ((1,1), (1,2), (1,3), (2,2), (2,3), (3,3))

function scalar_formula(x, q)
    r2 = dot(x, x)
    r = sqrt(r2)
    return SVector{18}(ntuple(Val(18)) do slot
        i = (slot - 1) ÷ 6 + 1
        j, k = PAIRS[(slot - 1) % 6 + 1]
        d = (i == j ? x[k] : zero(r2)) + (i == k ? x[j] : zero(r2)) +
            (j == k ? x[i] : zero(r2))
        C * q * (3r2 * d - 15x[i] * x[j] * x[k]) / (r2^3 * r)
    end)
end

function vortex_formula(x, gamma)
    A = cross(gamma, x)
    B = SMatrix{3,3}(0, gamma[3], -gamma[2], -gamma[3], 0, gamma[1],
        gamma[2], -gamma[1], 0)
    r2 = dot(x, x)
    r = sqrt(r2)
    return SVector{18}(ntuple(Val(18)) do slot
        i = (slot - 1) ÷ 6 + 1
        j, k = PAIRS[(slot - 1) % 6 + 1]
        C * (15A[i] * x[j] * x[k] - 3r2 *
            (A[i] * (j == k) + B[i,j] * x[k] + B[i,k] * x[j])) / (r2^3 * r)
    end)
end

for scale in (1e-3, 1.0, 1e3)
    x = scale * SVector(0.7, -0.4, 1.2)
    q = 1.7
    scalar_ad = ForwardDiff.jacobian(z -> vec(ForwardDiff.hessian(y -> C*q/norm(y), z)), x)
    scalar_expected = SVector{18}(ntuple(Val(18)) do slot
        i = (slot - 1) ÷ 6 + 1
        j, k = PAIRS[(slot - 1) % 6 + 1]
        scalar_ad[i + 3(j-1), k]
    end)
    @assert isapprox(scalar_formula(x, q), scalar_expected; rtol=2e-12)

    gamma = SVector(0.3, -0.8, 1.1)
    velocity(z) = C * cross(gamma, z) / norm(z)^3
    vortex_expected = SVector{18}(ntuple(Val(18)) do slot
        i = (slot - 1) ÷ 6 + 1
        j, k = PAIRS[(slot - 1) % 6 + 1]
        ForwardDiff.hessian(z -> velocity(z)[i], x)[j,k]
    end)
    @assert isapprox(vortex_formula(x, gamma), vortex_expected; rtol=3e-12)
end

println("PASS: scalar and point-vortex packed-18 formulas agree with ForwardDiff")
