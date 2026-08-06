# Shared systems/references for the task 032 device-system interface tests
# (host: device_system_interface_test.jl; CUDA: cuda_radix_interface_test.jl).
# Requires gravitational.jl and vortex.jl to be included first.

using FastMultipole.LinearAlgebra: dot, I

# stdlib-only reference erf (so standalone `julia --project=. test/...` runs work
# without the test env): Maclaurin series below x = 2 (alternating, converged to
# eps), continued-fraction erfc above — the same construction the 031a/032
# fit scripts validate against 256-bit references.
function _ref_erf(x::Float64)
    ax = abs(x)
    if ax < 2.0
        s = 0.0
        term = ax
        n = 0
        while true
            add = term / (2n + 1)
            s += add
            n += 1
            term *= -ax * ax / n
            abs(add) <= eps() * max(abs(s), 1.0) && break
        end
        return sign(x) * (2 / sqrt(pi)) * s
    end
    u = 1 / (2 * ax * ax)
    cf = 1.0
    for k in 60:-1:1
        cf = 1 + k * u / cf
    end
    return sign(x) * (1 - exp(-ax * ax) / (ax * sqrt(pi)) / cf)
end

# analytic O(N²) scalar reference including the hessian (the Gravitational
# test system's direct! computes potential + gradient only)
function _interface_scalar_direct(sys)
    n = FastMultipole.get_n_bodies(sys)
    u = zeros(n); g = zeros(3, n); H = zeros(9, n)
    c = 1 / (4pi)
    for i in 1:n
        xi = FastMultipole.get_position(sys, i)
        for j in 1:n
            i == j && continue
            d = xi - FastMultipole.get_position(sys, j)
            q = sys.bodies[j].strength * c
            r2 = dot(d, d)
            invr = 1 / sqrt(r2); invr3 = invr / r2; invr5 = invr3 / r2
            u[i] += q * invr
            g[:, i] .-= q .* d .* invr3
            H[:, i] .+= vec(q .* (3 .* (d * d') .* invr5 .- invr3 .* Matrix(I, 3, 3)))
        end
    end
    return u, g, H
end

# data_per_body > strength rows: a vortex system carrying two extra state rows
# (rows 8:9) through the packed matrix, opaque to the far field (task 032
# packed-layout round trip)
struct ExtendedVortex{TF}
    inner::VortexParticles{TF}
end
function FastMultipole.source_system_to_buffer!(buffer, i_buffer, system::ExtendedVortex, i_body)
    FastMultipole.source_system_to_buffer!(buffer, i_buffer, system.inner, i_body)
    buffer[8, i_buffer] = 10.0 + i_body
    buffer[9, i_buffer] = -Float64(i_body)
    return nothing
end
FastMultipole.data_per_body(::ExtendedVortex) = 9
FastMultipole.get_position(system::ExtendedVortex, i) = FastMultipole.get_position(system.inner, i)
FastMultipole.strength_dims(::ExtendedVortex) = 3
FastMultipole.get_n_bodies(system::ExtendedVortex) = FastMultipole.get_n_bodies(system.inner)
FastMultipole.has_vector_potential(::ExtendedVortex) = true
FastMultipole.body_type(::ExtendedVortex) = Point{Vortex}
FastMultipole.buffer_to_target_system!(system::ExtendedVortex, i_target, switch, buffer, i_buffer) =
    FastMultipole.buffer_to_target_system!(system.inner, i_target, switch, buffer, i_buffer)

# Vortex system with a per-body smoothing radius in packed extra-state row 8,
# selecting the regularized-everywhere gaussianerf nearfield (task 032 stage 2).
# Row 4 stays the (distinct) MAC radius from the wrapped system, mirroring the
# FLOWVPM convention of an inflated MAC radius plus a raw σ extra state.
struct SmoothedVortex{TF}
    inner::VortexParticles{TF}
    sigma::Vector{TF}
end
function FastMultipole.source_system_to_buffer!(buffer, i_buffer, system::SmoothedVortex, i_body)
    FastMultipole.source_system_to_buffer!(buffer, i_buffer, system.inner, i_body)
    buffer[8, i_buffer] = system.sigma[i_body]
    return nothing
end
FastMultipole.data_per_body(::SmoothedVortex) = 8
FastMultipole.get_position(system::SmoothedVortex, i) = FastMultipole.get_position(system.inner, i)
FastMultipole.strength_dims(::SmoothedVortex) = 3
FastMultipole.get_n_bodies(system::SmoothedVortex) = FastMultipole.get_n_bodies(system.inner)
FastMultipole.has_vector_potential(::SmoothedVortex) = true
FastMultipole.body_type(::SmoothedVortex) = Point{Vortex}
FastMultipole.direct_kernel(::SmoothedVortex) = RegularizedVortex(; sigma_row=8)
FastMultipole.buffer_to_target_system!(system::SmoothedVortex, i_target, switch, buffer, i_buffer) =
    FastMultipole.buffer_to_target_system!(system.inner, i_target, switch, buffer, i_buffer)

# The same smoothed system with the task-032a partitioned-replacement trait:
# delegates everything to an inner SmoothedVortex but selects PartitionedVortex,
# so the end-to-end A/B runs on identical bodies/σ with only the kernel changed.
struct PartitionedSmoothedVortex{TF}
    smoothed::SmoothedVortex{TF}
end
FastMultipole.source_system_to_buffer!(buffer, i_buffer, system::PartitionedSmoothedVortex, i_body) =
    FastMultipole.source_system_to_buffer!(buffer, i_buffer, system.smoothed, i_body)
FastMultipole.data_per_body(::PartitionedSmoothedVortex) = 8
FastMultipole.get_position(system::PartitionedSmoothedVortex, i) =
    FastMultipole.get_position(system.smoothed, i)
FastMultipole.strength_dims(::PartitionedSmoothedVortex) = 3
FastMultipole.get_n_bodies(system::PartitionedSmoothedVortex) =
    FastMultipole.get_n_bodies(system.smoothed)
FastMultipole.has_vector_potential(::PartitionedSmoothedVortex) = true
FastMultipole.body_type(::PartitionedSmoothedVortex) = Point{Vortex}
FastMultipole.direct_kernel(::PartitionedSmoothedVortex) = PartitionedVortex(; sigma_row=8)
FastMultipole.buffer_to_target_system!(system::PartitionedSmoothedVortex, i_target, switch, buffer, i_buffer) =
    FastMultipole.buffer_to_target_system!(system.smoothed, i_target, switch, buffer, i_buffer)

# O(N²) regularized gaussianerf U/J reference (Float64, stdlib _ref_erf),
# theory §1 formulas with the source σ. Returns (U 3×n, J 9×n column-major).
function _interface_regularized_direct(system::SmoothedVortex)
    n = FastMultipole.get_n_bodies(system)
    U = zeros(3, n); J = zeros(9, n)
    A = sqrt(2 / pi)
    for i in 1:n
        xi = FastMultipole.get_position(system, i)
        for j in 1:n
            i == j && continue
            d = xi - FastMultipole.get_position(system, j)
            G = system.inner.bodies[j].strength
            sigma = Float64(system.sigma[j])
            r2 = dot(d, d)
            r2 == 0 && continue
            r = sqrt(r2)
            rho = r / sigma
            g = _ref_erf(rho / sqrt2) - A * rho * exp(-rho^2 / 2)
            gp = A * rho^2 * exp(-rho^2 / 2)
            cr3 = 1 / (4pi * r2 * r)
            crss = (
                (d[3] * G[2] - d[2] * G[3]) * cr3,
                (d[1] * G[3] - d[3] * G[1]) * cr3,
                (d[2] * G[1] - d[1] * G[2]) * cr3,
            )
            a = (rho * gp - 3g) / r2
            b = -g * cr3
            U[1, i] += g * crss[1]
            U[2, i] += g * crss[2]
            U[3, i] += g * crss[3]
            J[1, i] += a * crss[1] * d[1]
            J[2, i] += a * crss[2] * d[1] - b * G[3]
            J[3, i] += a * crss[3] * d[1] + b * G[2]
            J[4, i] += a * crss[1] * d[2] + b * G[3]
            J[5, i] += a * crss[2] * d[2]
            J[6, i] += a * crss[3] * d[2] - b * G[1]
            J[7, i] += a * crss[1] * d[3] - b * G[2]
            J[8, i] += a * crss[2] * d[3] + b * G[1]
            J[9, i] += a * crss[3] * d[3]
        end
    end
    return U, J
end
