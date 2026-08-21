# Task 028 shared harness pieces: a device-resident scalar source/target system,
# the device Euler convection step, and an on-device Float64 sampled direct
# reference. Included by benchmark_028_feasibility.jl and
# test/cuda_radix_convection_test.jl AFTER `using FastMultipole` and `using CUDA`.
#
# Test/benchmark side only — no production `src/` surface. The bodies replicate
# `generate_gravitational(seed, n)` exactly (test/gravitational.jl) so the 024b
# checksummed direct references (seed 24025, sampler seed 24026) apply verbatim.

using Random
using Statistics

# Same body stream as test/gravitational.jl generate_gravitational, without the
# Gravitational wrapper: rows 1:3 position in [0,1]^3, 4 radius, 5 strength.
function fm028_body_matrix(seed, n; radius_factor=0.1, strength_scale=1 / n)
    Random.seed!(seed)
    bodies = rand(8, n)
    bodies[4, :] ./= (n^(1 / 3) * 2)
    bodies[4, :] .*= radius_factor
    bodies[5, :] .*= strength_scale
    return bodies
end

# Device-resident scalar system: positions/strengths live on the GPU and are
# advanced there by fm028_euler!; `host_positions` is a construction-time mirror
# (get_position) that goes stale once convection starts — the resident lifecycle
# never reads it after cache construction.
struct FM028DeviceSystem{TF,PM,VV}
    host_positions::Matrix{TF}
    positions::PM      # CuMatrix{TF} 3 x n
    radii::VV          # CuVector{TF}
    strengths::VV      # CuVector{TF}
    potential::VV      # CuVector{TF}, overwritten each step
    gradient::PM       # CuMatrix{TF} 3 x n, overwritten each step
end

function FM028DeviceSystem{TF}(bodies::Matrix{Float64}) where TF
    n = size(bodies, 2)
    host_positions = Matrix{TF}(bodies[1:3, :])
    return FM028DeviceSystem(host_positions,
        CUDA.CuArray(host_positions),
        CUDA.CuArray(Vector{TF}(bodies[4, :])),
        CUDA.CuArray(Vector{TF}(bodies[5, :])),
        CUDA.zeros(TF, n),
        CUDA.zeros(TF, 3, n))
end

Base.eltype(::FM028DeviceSystem{TF}) where TF = TF
FastMultipole.get_n_bodies(sys::FM028DeviceSystem) = size(sys.positions, 2)
FastMultipole.data_per_body(::FM028DeviceSystem) = 5
FastMultipole.strength_dims(::FM028DeviceSystem) = 1
FastMultipole.get_position(sys::FM028DeviceSystem{TF}, i) where TF =
    SVector{3,TF}(sys.host_positions[1, i], sys.host_positions[2, i],
        sys.host_positions[3, i])
FastMultipole.residency(::FM028DeviceSystem) = DeviceResident()
FastMultipole.has_vector_potential(::FM028DeviceSystem) = false

# The recurring device refresh always passes the identity sort index
# (_canonical_cuda_source_buffer), so this is a straight device-side repack.
function FastMultipole.source_to_buffer!(device_buffer::CUDA.AnyCuArray,
        sys::FM028DeviceSystem, sort_index)
    (first(sort_index) == 1 && last(sort_index) == size(sys.positions, 2)) ||
        error("FM028DeviceSystem expects the identity sort index")
    device_buffer[1:3, :] .= sys.positions
    device_buffer[4, :] .= sys.radii
    device_buffer[5, :] .= sys.strengths
    return device_buffer
end

# Overwrite (not accumulate): each step's finalize delivers the full influence,
# which is what a time stepper consumes.
function FastMultipole.buffer_to_target!(sys::FM028DeviceSystem,
        device_output_buffer::CUDA.AnyCuArray, derivatives_switch, sort_index)
    spi = FastMultipole.scalar_potential_index(derivatives_switch)
    spi > 0 && (sys.potential .= vec(view(device_output_buffer, spi, :)))
    grange = FastMultipole.gradient_range(derivatives_switch)
    isempty(grange) || (sys.gradient .= view(device_output_buffer, grange, :))
    return sys
end

# Device Euler convection proxy (task 028 per-step workload): x .+= dt * v with
# v = the computed gradient, clamped so no body ever exits the cache's fixed
# Morton box (out-of-box refresh throws ArgumentError by contract).
function fm028_euler!(sys::FM028DeviceSystem{TF}, dt, lo, hi) where TF
    sys.positions .= clamp.(sys.positions .+ TF(dt) .* sys.gradient, TF(lo), TF(hi))
    return sys
end

# ---- on-device Float64 sampled direct reference ------------------------------
#
# Same kernel convention as the production CUDA direct stage
# (_cuda_direct_source_output_kernel!): u += q/(4pi r), g -= q dx/(4pi r^3),
# self-interaction excluded by r2 > 0. Accumulation is Float64 regardless of the
# system precision, so this is a valid reference at the *current* positions —
# used for the post-convection accuracy re-check and for any n without a 024b
# reference CSV. One block per sample; strided threads + atomic reduction.
function _fm028_direct_sample_kernel!(out, sample_idx, positions, strengths, n)
    s = blockIdx().x
    t = threadIdx().x
    i = sample_idx[s]
    xi = Float64(positions[1, i])
    yi = Float64(positions[2, i])
    zi = Float64(positions[3, i])
    u = 0.0
    gx = 0.0
    gy = 0.0
    gz = 0.0
    c = inv(4.0 * pi)
    j = t
    @inbounds while j <= n
        dx = xi - Float64(positions[1, j])
        dy = yi - Float64(positions[2, j])
        dz = zi - Float64(positions[3, j])
        r2 = dx * dx + dy * dy + dz * dz
        if r2 > 0.0
            invr = inv(sqrt(r2))
            q = Float64(strengths[j]) * c
            u += q * invr
            invr3 = invr * invr * invr
            gx -= q * dx * invr3
            gy -= q * dy * invr3
            gz -= q * dz * invr3
        end
        j += blockDim().x
    end
    CUDA.@atomic out[1, s] += u
    CUDA.@atomic out[2, s] += gx
    CUDA.@atomic out[3, s] += gy
    CUDA.@atomic out[4, s] += gz
    return nothing
end

# positions: 3 x n (any real eltype, device); strengths: n (device);
# indices: host Vector{Int} of sample body ids. Returns 4 x nsamples Float64
# host matrix (row 1 potential, rows 2:4 gradient).
function fm028_direct_sample_reference(positions::CUDA.AnyCuArray,
        strengths::CUDA.AnyCuArray, indices::Vector{Int})
    n = size(positions, 2)
    nsamples = length(indices)
    out = CUDA.zeros(Float64, 4, nsamples)
    d_idx = CUDA.CuArray(indices)
    CUDA.@cuda threads=256 blocks=nsamples _fm028_direct_sample_kernel!(
        out, d_idx, positions, strengths, n)
    CUDA.synchronize()
    return Array(out)
end

# ---- accuracy metrics (024b five-column contract) ----------------------------

# potential/gradient: sampled FMM values (nsamples, and 3 x nsamples);
# ref_potential/ref_gradient the same shapes from the reference. Returns the
# 024b metric names (fm024b_accuracy_metrics equivalent, array-based).
function fm028_accuracy_metrics(potential, gradient, ref_potential, ref_gradient)
    nsamples = length(ref_potential)
    pe2 = 0.0
    pr2 = 0.0
    ge2 = 0.0
    gr2 = 0.0
    gmax = 0.0
    for k in 1:nsamples
        dp = Float64(potential[k]) - Float64(ref_potential[k])
        pe2 += dp * dp
        pr2 += Float64(ref_potential[k])^2
        e2 = 0.0
        for d in 1:3
            dg = Float64(gradient[d, k]) - Float64(ref_gradient[d, k])
            e2 += dg * dg
            gr2 += Float64(ref_gradient[d, k])^2
        end
        ge2 += e2
        gmax = max(gmax, sqrt(e2))
    end
    return (potential_abs_rms=sqrt(pe2 / nsamples),
        potential_rel_rms=sqrt(pe2 / max(pr2, eps(Float64))),
        gradient_abs_rms=sqrt(ge2 / nsamples),
        gradient_rel_rms=sqrt(ge2 / max(gr2, eps(Float64))),
        gradient_max=gmax)
end

# Download sampled FMM output from a device system (outside any timed region).
function fm028_sampled_output(sys::FM028DeviceSystem, indices::Vector{Int})
    pot = Array(sys.potential)[indices]
    grad = Array(sys.gradient)[:, indices]
    return pot, grad
end
