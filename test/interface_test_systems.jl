# Shared systems/references for the task 032 device-system interface tests
# (host: device_system_interface_test.jl; CUDA: cuda_radix_interface_test.jl).
# Requires gravitational.jl and vortex.jl to be included first.

using FastMultipole.LinearAlgebra: dot, I

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
