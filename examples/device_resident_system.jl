# Device-resident system worked example (task 032, promoted from the task 028
# benchmark harness MATRIX_OPERATOR_REFACTOR/scripts/fm028_device_system.jl).
#
# Demonstrates how an external code that keeps its particle state on the GPU
# connects to the resident radix FMM lifecycle:
#
#   * a device-resident scalar point-source system with all trait overloads,
#   * device methods for source_to_buffer! and buffer_to_target!,
#   * RadixFMMCache construction with max_n_bodies capacity pre-sizing,
#   * a 3-step convection loop that asserts the task 023 counter contract
#     (zero per-step body H2D/D2H, zero expansion host copies).
#
# Companion documentation: docs/src/device_interface.md.
#
# Run with:  julia --project=. examples/device_resident_system.jl
# Requires CUDA.jl and an NVIDIA GPU; exits gracefully with a message when the
# CUDA radix lifecycle is unavailable.

using FastMultipole
using FastMultipole.StaticArrays
using Random

# `load_cuda_radix_lifecycle!()` is the opt-in gate for the CUDA resident path.
# It returns `false` (rather than throwing) when CUDA.jl or a functional GPU is
# missing; `cuda_radix_status()` explains why. The tests use the same gate.
const _CUDA_LOADED = FastMultipole.load_cuda_radix_lifecycle!()
if !_CUDA_LOADED
    @info "CUDA radix lifecycle unavailable; skipping the device-resident example." status = FastMultipole.cuda_radix_status()
end

if _CUDA_LOADED

using CUDA

#===============================================================================
The consumer system type
===============================================================================#

# A device-resident scalar system: positions and strengths live on the GPU and
# are advanced there by the time stepper. `host_positions` is a
# construction-time mirror used only by `get_position` — the framework reads
# positions on the host solely to derive the domain box at cache construction
# (and, for `bounds=nothing` recenter! calls, host systems only). It goes stale
# once convection starts; the resident lifecycle never reads it after
# construction.
struct DeviceResidentPoints{TF,PM<:CUDA.CuMatrix{TF},VV<:CUDA.CuVector{TF}}
    host_positions::Matrix{TF}
    positions::PM      # 3 x n, device
    radii::VV          # n, device (MAC/error radius, packed row 4)
    strengths::VV      # n, device (scalar strength q, packed row 5)
    potential::VV      # n, device output (overwritten each step)
    gradient::PM       # 3 x n, device output (overwritten each step)
end

function DeviceResidentPoints{TF}(bodies::Matrix{Float64}) where TF
    n = size(bodies, 2)
    host_positions = Matrix{TF}(bodies[1:3, :])
    return DeviceResidentPoints(host_positions,
        CUDA.CuArray(host_positions),
        CUDA.CuArray(Vector{TF}(bodies[4, :])),
        CUDA.CuArray(Vector{TF}(bodies[5, :])),
        CUDA.zeros(TF, n),
        CUDA.zeros(TF, 3, n))
end

#===============================================================================
Trait overloads (the consumer surface; see docs/src/device_interface.md)

All of these must be allocation-free — they are called every step (or at
construction) on the hot path.
===============================================================================#

Base.eltype(::DeviceResidentPoints{TF}) where TF = TF

# Live body count. It may vary per step (1 <= n <= max_n_bodies): a consumer
# that adds/removes particles just updates its arrays and this count — the
# framework tracks a valid prefix of its capacity-sized buffers and never
# reallocates.
FastMultipole.get_n_bodies(sys::DeviceResidentPoints) = size(sys.positions, 2)

# Packed rows per body: [x, y, z, radius, q] = 5 for a scalar point source with
# no extra states. A regularized vortex consumer would add extra-state rows
# here (e.g. sigma) after the strength; the contract requires
# data_per_body >= 4 + strength_dims (checked at construction).
FastMultipole.data_per_body(::DeviceResidentPoints) = 5

# Scalar strength (a vortex system with vector strength Gamma would return 3).
FastMultipole.strength_dims(::DeviceResidentPoints) = 1

# Host-side position accessor — construction/recenter bounds derivation only.
FastMultipole.get_position(sys::DeviceResidentPoints{TF}, i) where TF =
    SVector{3,TF}(sys.host_positions[1, i], sys.host_positions[2, i],
        sys.host_positions[3, i])

# THE key trait: opt into the device-resident lifecycle. To fall back to the
# transfer-based host path (adequate at n <~ 1e4 or step budgets >> 100 ms;
# see docs §"Resident vs transfer-based coupling"), simply delete this method
# (HostResident() is the default) and provide the host hooks
# `source_system_to_buffer!` / host `buffer_to_target!` instead — the rest of
# the surface (traits, cache construction, fmm! call) is identical.
FastMultipole.residency(::DeviceResidentPoints) = FastMultipole.DeviceResident()

# Scalar source, no Lamb-Helmholtz channel. A vortex consumer returns `true`
# here and `Point{Vortex}` from `body_type` (the chi channel is then required
# and checked at construction).
FastMultipole.has_vector_potential(::DeviceResidentPoints) = false

# B2M kernel selection (default shown explicitly for clarity). All source
# systems sharing one cache must agree on this.
FastMultipole.body_type(::DeviceResidentPoints) = FastMultipole.Point{FastMultipole.Source}

# Nearfield kernel functor (default for Point{Source} shown explicitly).
# A regularized vortex consumer would return
# RegularizedVortex(; sigma_row=8) — sigma always travels in an extra-state
# row (>= 5), never MAC-radius row 4.
FastMultipole.direct_kernel(::DeviceResidentPoints) = FastMultipole.SingularSource()

#===============================================================================
Per-step device hooks
===============================================================================#

# Called at the top of every fmm!(sys, cache) step. `device_buffer` is the
# FRAMEWORK-OWNED persistent per-system device buffer (data_per_body x
# max_n_bodies; only the live-prefix columns matter). The consumer fills the
# packed column layout:
#   rows 1:3                 position
#   row  4                   radius (MAC/error use)
#   rows 5:4+strength_dims   strength
#   remaining rows           extra states (none here)
# `sort_index` is the identity on the device-resident path (the framework
# sorts internally); we assert that rather than silently mis-pack.
# Steady-state allocation-free: broadcasts into the existing buffer only.
function FastMultipole.source_to_buffer!(device_buffer::CUDA.AnyCuArray,
        sys::DeviceResidentPoints, sort_index)
    (first(sort_index) == 1 && last(sort_index) == size(sys.positions, 2)) ||
        error("DeviceResidentPoints expects the identity sort index")
    device_buffer[1:3, :] .= sys.positions
    device_buffer[4, :] .= sys.radii
    device_buffer[5, :] .= sys.strengths
    return device_buffer
end

# Called at the end of every fmm!(sys, cache) step with the framework-owned
# per-system device output buffer, which is SWITCH-RELATIVE: it carries only
# the channels the DerivativesSwitch requested, so always address it through
# the accessors (scalar_potential_index / gradient_range / hessian_range),
# never hard-coded rows.
#
# DELIVERY SEMANTICS: the framework always delivers the TOTAL influence of
# this evaluation (its accumulators are zeroed each step). Overwrite vs
# accumulate into your own state is YOUR choice:
#   * this example OVERWRITES (`.=`) — a time stepper consumes the fresh field;
#   * FLOWVPM ACCUMULATES (`.+=`) — it zeroes its own U/J before each
#     evaluation and may sum several contributions.
function FastMultipole.buffer_to_target!(sys::DeviceResidentPoints,
        device_output_buffer::CUDA.AnyCuArray, derivatives_switch, sort_index)
    spi = FastMultipole.scalar_potential_index(derivatives_switch)
    spi > 0 && (sys.potential .= vec(view(device_output_buffer, spi, :)))
    grange = FastMultipole.gradient_range(derivatives_switch)
    isempty(grange) || (sys.gradient .= view(device_output_buffer, grange, :))
    # A consumer needing the 9-component hessian (cache built with
    # hessian=true and fmm!(...; hessian=true)) would additionally read
    # hrange = FastMultipole.hessian_range(derivatives_switch) here.
    return sys
end

#===============================================================================
A device time stepper
===============================================================================#

# Euler convection proxy entirely on the device: x .+= dt * v with v = the
# computed gradient. Positions are clamped so no body ever exits the cache's
# fixed domain box — an out-of-box body throws ArgumentError at the next step
# by contract (no silent geometry rebuild; use recenter!, task 032 stage 3,
# when the box must actually move).
function convect!(sys::DeviceResidentPoints{TF}, dt, lo, hi) where TF
    sys.positions .= clamp.(sys.positions .+ TF(dt) .* sys.gradient, TF(lo), TF(hi))
    return sys
end

#===============================================================================
Driver: construct, step, and verify the zero-transfer counter contract
===============================================================================#

function run_example(; n=2000, ell=3, expansion_order=3, TF=Float32, dt=1e-3)
    # Random point sources in [0,1]^3 (same stream as test/gravitational.jl).
    Random.seed!(24025)
    bodies = rand(8, n)
    bodies[4, :] .*= 0.1 / (n^(1 / 3) * 2)   # small finite radii
    bodies[5, :] ./= n                        # strengths ~ 1/n
    sys = DeviceResidentPoints{TF}(bodies)

    # Capacity contract: pre-size ONCE with max_n_bodies (here = n; a consumer
    # expecting particle growth passes its known maximum). The domain box,
    # ell, expansion_order, max_n_bodies, and precision are fixed for the
    # cache's lifetime; every buffer and operator table is allocated here at
    # capacity, and the recurring steps below reallocate nothing.
    box_min = SVector{3,Float64}(-0.01, -0.01, -0.01)
    box_size = 1.02
    cache = RadixFMMCache(sys; expansion_order, ell, max_n_bodies=n,
        bounds=(box_min, box_size), device=true)

    # First step (already the recurring fast path — construction ran the
    # initial state update).
    fmm!(sys, cache; scalar_potential=true, gradient=true)

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
        fmm!(sys, cache; scalar_potential=true, gradient=true)
        convect!(sys, dt, 0.0, 1.0)
    end
    CUDA.synchronize()

    # Counter contract (task 023): flat across recurring steps.
    @assert counters.body_uploads == base_body_uploads "per-step body H2D detected"
    @assert counters.influence_downloads == base_influence_downloads "per-step result D2H detected"
    @assert counters.route_uploads == base_route_uploads "per-step route upload detected"
    @assert counters.operator_uploads == base_operator_uploads "per-step operator upload detected"
    @assert counters.expansion_host_copies == 0 "expansion left the device"

    # Results live on the device in the consumer's own arrays.
    println("device-resident example: n = $n, TF = $TF, ell = $ell, P = $(expansion_order)")
    println("  potential[1:3] = ", Array(sys.potential)[1:3])
    println("  |gradient| mean = ", sum(abs.(Array(sys.gradient))) / (3n))
    println("  counter contract satisfied: body_uploads/influence_downloads/",
        "route_uploads/operator_uploads flat, expansion_host_copies == 0")
    return sys, cache
end

run_example()

end # _CUDA_LOADED
