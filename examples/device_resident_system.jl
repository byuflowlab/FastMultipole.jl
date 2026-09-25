#=
Device-resident system worked example: how an external code that keeps its
particle state on the GPU connects to the resident radix FMM lifecycle through
the KernelAbstractions extension.

  * a device-resident regularized vortex-particle system with all trait
    overloads, including `device_backend` (which GPU the cache is built on),
  * device methods for source_to_buffer! and buffer_to_target!, written on
    GPUArraysCore's abstract types so the same code runs on CUDA and Metal,
  * RadixFMMCache construction with max_n_bodies capacity pre-sizing,
  * a 3-step convection loop that asserts the zero-transfer counter contract
    (zero per-step body H2D/D2H, zero expansion host copies).

The device lifecycle covers vortex bodies (`Point{Vortex}`, three-component
strength, the Lamb-Helmholtz channel on) with the `RegularizedVortex` nearfield
kernel; that is the body a vortex particle method needs, and the only one the
extension implements today.

Companion documentation: docs/src/gpu.md and docs/src/device_interface.md.

Run with:  julia --project=<env> examples/device_resident_system.jl
where <env> has FastMultipole, KernelAbstractions, GPUArraysCore, StaticArrays
and CUDA (NVIDIA) or Metal (Apple). Loading KernelAbstractions next to
FastMultipole activates the extension; the script exits with a message when no
functional GPU is found.
=#
using FastMultipole
using FastMultipole.StaticArrays
using Random
using KernelAbstractions
using GPUArraysCore: AbstractGPUMatrix, AbstractGPUVector

# Metal on Apple, CUDA elsewhere: the two are mutually exclusive by package
# availability, so `using` the absent one must not even be attempted.
@static if Sys.isapple()
    using Metal
    const DEV_BACKEND = Metal.MetalBackend()
    devarray(x) = Metal.MtlArray(x)
    dev_functional() = Metal.functional()
    dev_synchronize() = Metal.synchronize()
else
    using CUDA
    const DEV_BACKEND = CUDABackend()
    devarray(x) = CUDA.CuArray(x)
    dev_functional() = CUDA.functional()
    dev_synchronize() = CUDA.synchronize()
end
const DEV_TF = Float32          # Metal has no Float64; the H200 runs either

if !dev_functional()
    @info "no functional GPU; skipping the device-resident example"
else

#===============================================================================
A device-resident vortex-particle system
===============================================================================#
# Positions, strengths (Gamma) and cores (sigma) live on the GPU and are
# advanced there by the time stepper; the velocity the FMM delivers lands on
# the device too. `host_positions` is a construction-time mirror used only by
# `get_position`: the framework reads positions on the host solely to derive
# the domain box at cache construction. It goes stale once convection starts;
# the resident lifecycle never reads it after construction.
struct VortexBlobs{TF,PM<:AbstractGPUMatrix{TF},VV<:AbstractGPUVector{TF}}
    host_positions::Matrix{TF}
    positions::PM        # 3 x n
    strengths::PM        # 3 x n, Gamma
    sigma::VV            # n, regularization core
    velocity::PM         # 3 x n, output
end

function VortexBlobs{TF}(positions::Matrix, strengths::Matrix, sigma::Vector) where TF
    n = size(positions, 2)
    host_positions = Matrix{TF}(positions)
    return VortexBlobs(host_positions, devarray(host_positions), devarray(Matrix{TF}(strengths)),
                       devarray(Vector{TF}(sigma)), devarray(zeros(TF, 3, n)))
end

#===============================================================================
Trait overloads
===============================================================================#
Base.eltype(::VortexBlobs{TF}) where TF = TF

# Live body count. It may vary per step (1 <= n <= max_n_bodies): a consumer
# that adds/removes particles just updates its arrays and this count; the
# framework tracks a valid prefix of its capacity-sized buffers and never
# reallocates.
FastMultipole.get_n_bodies(sys::VortexBlobs) = size(sys.positions, 2)

# Packed rows per body: [x, y, z, radius, Gx, Gy, Gz, sigma] = 8. The contract
# requires data_per_body >= 4 + strength_dims; sigma travels in an extra-state
# row (>= 5), never in the MAC-radius row 4.
FastMultipole.data_per_body(::VortexBlobs) = 8
FastMultipole.strength_dims(::VortexBlobs) = 3

# Host-side position accessor: construction bounds derivation only.
FastMultipole.get_position(sys::VortexBlobs{TF}, i) where TF =
    SVector{3,TF}(sys.host_positions[1, i], sys.host_positions[2, i], sys.host_positions[3, i])

# THE key trait: opt into the device-resident lifecycle. To fall back to the
# transfer-based host path (adequate at n <~ 1e4 or step budgets >> 100 ms),
# delete this method (HostResident() is the default) and provide the host
# hooks `source_system_to_buffer!` / host `buffer_to_target!` instead; the
# rest of the surface (traits, cache construction, fmm! call) is identical.
FastMultipole.residency(::VortexBlobs) = FastMultipole.DeviceResident()

# Which GPU the cache is built on. The extension reads this from the first
# source system that names a backend, before it touches any buffer.
FastMultipole.device_backend(::VortexBlobs) = DEV_BACKEND

# Vortex sources carry the vector potential (the Lamb-Helmholtz channel is
# then required and the cache infers it), and the B2M kernel is Point{Vortex}.
FastMultipole.has_vector_potential(::VortexBlobs) = true
FastMultipole.body_type(::VortexBlobs) = FastMultipole.Point{FastMultipole.Vortex}

# Nearfield kernel functor: the regularized vortex with sigma in row 8.
FastMultipole.direct_kernel(::VortexBlobs) = FastMultipole.RegularizedVortex(; sigma_row = 8)

#===============================================================================
Per-step device hooks
===============================================================================#
# Called at the top of every fmm!(sys, cache) step. `device_buffer` is the
# FRAMEWORK-OWNED persistent per-system device buffer (data_per_body x
# max_n_bodies; only the live-prefix columns matter). The consumer fills the
# packed column layout:
#   rows 1:3   position
#   row  4     radius (multipole-acceptance use): a few cores here
#   rows 5:7   strength
#   row  8     sigma
# `sort_index` is the identity on the device-resident path (the framework
# sorts internally); we assert that rather than silently mis-pack.
# Steady-state allocation-free: broadcasts into the existing buffer only.
const RHO_OVER_SIGMA = 4.0f0
function FastMultipole.source_to_buffer!(device_buffer::AbstractGPUMatrix,
        sys::VortexBlobs, sort_index)
    (first(sort_index) == 1 && last(sort_index) == size(sys.positions, 2)) ||
        error("VortexBlobs expects the identity sort index")
    device_buffer[1:3, :] .= sys.positions
    device_buffer[4, :] .= RHO_OVER_SIGMA .* sys.sigma
    device_buffer[5:7, :] .= sys.strengths
    device_buffer[8, :] .= sys.sigma
    return device_buffer
end

# Called at the end of every fmm!(sys, cache) step with the framework-owned
# per-system device output buffer, which is SWITCH-RELATIVE: it carries only
# the channels the DerivativesSwitch requested, so always address it through
# the accessors (gradient_range / hessian_range), never hard-coded rows. For a
# vortex system `gradient` is the induced velocity.
#
# The framework always delivers the TOTAL influence of this evaluation (its
# accumulators are zeroed each step). Overwrite vs accumulate into your own
# state is the consumer's choice: this example OVERWRITES (`.=`); FLOWVPM
# ACCUMULATES (`.+=`) because it sums several contributions.
function FastMultipole.buffer_to_target!(sys::VortexBlobs,
        device_output_buffer::AbstractGPUMatrix, derivatives_switch, sort_index)
    grange = FastMultipole.gradient_range(derivatives_switch)
    isempty(grange) || (sys.velocity .= view(device_output_buffer, grange, :))
    # A consumer needing the velocity gradient (cache built with hessian=true
    # and fmm!(...; hessian=true)) would additionally read
    # FastMultipole.hessian_range(derivatives_switch) here.
    return sys
end

#===============================================================================
A device time stepper
===============================================================================#
# Euler convection entirely on the device. Positions are clamped so no body
# ever exits the cache's fixed domain box: an out-of-box body throws
# ArgumentError at the next step by contract (no silent geometry rebuild; use
# recenter! when the box must actually move).
function convect!(sys::VortexBlobs{TF}, dt, lo, hi) where TF
    sys.positions .= clamp.(sys.positions .+ TF(dt) .* sys.velocity, TF(lo), TF(hi))
    return sys
end

#===============================================================================
Driver: construct, step, and verify the zero-transfer counter contract
===============================================================================#
function run_example(; n = 2000, ell = 3, expansion_order = 4, TF = DEV_TF, dt = 1e-3)
    # A random cloud of vortex blobs in [0,1]^3. The core must satisfy the
    # near-set adequacy rule the cache enforces at construction: the smoothing
    # cutoff rho_t * sigma (rho_t = 4.789 for RegularizedVortex) has to fit
    # inside the gap the direct stencil leaves to the first M2L cell, about
    # 0.13 of the box at ell = 3, so sigma stays under 0.027 here.
    Random.seed!(24025)
    positions = rand(3, n)
    strengths = (rand(3, n) .- 0.5) ./ n
    sigma = fill(0.2 / n^(1 / 3), n)
    sys = VortexBlobs{TF}(positions, strengths, sigma)

    # Capacity contract: pre-size ONCE with max_n_bodies (here = n; a consumer
    # expecting particle growth passes its known maximum). The domain box,
    # ell, expansion_order, max_n_bodies and precision are fixed for the
    # cache's lifetime; every buffer and operator table is allocated here at
    # capacity, and the recurring steps below reallocate nothing.
    box_min = SVector{3,TF}(-0.01, -0.01, -0.01)
    box_size = TF(1.02)
    # Lifecycle options the device path needs stated: the working precision
    # (the default is Float64 from expansion order 4 up, which Metal cannot
    # run) and an M2L strategy the extension builds a device plan for
    # (concatenated here; dense is the other; the host default is neither).
    options = FastMultipole.CUDARadixLifecycleOptions(; precision = TF,
        m2l_strategy = FastMultipole.ConcatenatedFixedZM2L())
    cache = RadixFMMCache(sys; expansion_order, ell, max_n_bodies = n,
        bounds = (box_min, box_size), device = true, options)

    # First step (already the recurring fast path: construction ran the
    # initial state update).
    fmm!(sys, cache; scalar_potential = false, gradient = true)

    # Snapshot the transfer counters AFTER the first step: construction and
    # step 1 legitimately upload operators/routes once; the resident contract
    # is that the recurring steps add nothing.
    counters = cache.state.counters
    base_body_uploads = counters.body_uploads
    base_influence_downloads = counters.influence_downloads
    base_route_uploads = counters.route_uploads
    base_operator_uploads = counters.operator_uploads

    # 3-step convection loop: fmm! (in-place refresh + resident lifecycle +
    # device finalize) then a device Euler step. No host/device body traffic.
    for _ in 1:3
        fmm!(sys, cache; scalar_potential = false, gradient = true)
        convect!(sys, dt, 0.0, 1.0)
    end
    dev_synchronize()

    # Counter contract: flat across recurring steps.
    @assert counters.body_uploads == base_body_uploads "per-step body H2D detected"
    @assert counters.influence_downloads == base_influence_downloads "per-step result D2H detected"
    @assert counters.route_uploads == base_route_uploads "per-step route upload detected"
    @assert counters.operator_uploads == base_operator_uploads "per-step operator upload detected"
    @assert counters.expansion_host_copies == 0 "expansion left the device"

    # Results live on the device in the consumer's own arrays.
    U = Array(sys.velocity)
    println("device-resident example on $(nameof(typeof(DEV_BACKEND))): n = $n, TF = $TF, ell = $ell, P = $(expansion_order)")
    println("  velocity[:, 1] = ", U[:, 1])
    println("  |velocity| mean = ", sum(abs, U) / (3n))
    println("  counter contract satisfied: body_uploads/influence_downloads/",
        "route_uploads/operator_uploads flat, expansion_host_copies == 0")
    return sys, cache
end

run_example()

end # dev_functional()
