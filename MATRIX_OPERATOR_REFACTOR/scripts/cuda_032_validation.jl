# Task 032 Stage 4: H200 validation of the generalized device-resident system
# interface (deliverable 5) with a DEVICE-RESIDENT vortex consumer.
#
# What this script does, per precision (Float64, Float32):
#   1. Builds a device-resident regularized-vortex system (the fm028 device
#      pattern generalized to vector Γ, per-source σ in packed extra-state row 8,
#      `Point{Vortex}` + Lamb-Helmholtz, `RegularizedVortex(sigma_row=8)`,
#      13-row hessian output) at adequate geometry: uniform σ = β·n^{-1/3} and
#      the deepest ell satisfying the near-set adequacy inequality
#      g_min·h_leaf > ρ_t·σ_max (the margin is computed and printed up front;
#      the cache constructor independently enforces the same gate).
#   2. Runs a multi-step device Euler convection loop and asserts the 023
#      residency counter contract for a device-resident consumer:
#      body_uploads == 0, expansion_host_copies == 0, route/operator uploads
#      frozen at their post-construction values, influence/metadata downloads
#      flat across steps; steady-state device/host allocation per step is
#      measured and flagged if above FM032V_ALLOC_WARN.
#   3. Accuracy: a host-side Float64 erf-based REGULARIZED Biot-Savart sampled
#      direct reference (independent of the shipped erf-free g/h evaluation) at
#      the current device positions, at step 0 and again after the convection
#      loop. Sampled relative RMS of the velocity U gates at <= 1e-3 for
#      Float64 (the script errors after writing the CSV if the gate fails);
#      the 9-component J relative RMS is reported as a diagnostic only (spec:
#      J does not gate).
#   4. Writes one CSV row per (precision, step) with the per-step wall time
#      (CUDA.@elapsed around the full fmm! + Euler step), counters, allocation,
#      and accuracy columns, into FM032V_OUTDIR.
#
# Scalar no-regression (deliverable 5's second half) is intentionally NOT in
# this script: the shipped-028 scalar configuration rerun uses
# benchmark_028_feasibility.jl UNCHANGED (cuda_030_run.sh pattern) so the
# before/after comparison is against the identical harness. This script is
# vortex-only.
#
# Env knobs (FM032V_*, defaults in brackets):
#   FM032V_N            body count                          [100000]
#   FM032V_BETA         overlap β, σ = β·n^{-1/3}           [2.0]
#   FM032V_P            expansion order (P_literature - 1)  [3]
#   FM032V_ELL          radix depth; "auto" = deepest ell
#                       passing the adequacy margin         [auto]
#   FM032V_Q            near_radius2 (ball stencil)         [12]
#   FM032V_K            window_classes                      [256]
#   FM032V_TF           comma list of Float64,Float32       [Float64,Float32]
#   FM032V_STEPS        convection steps                    [5]
#   FM032V_WARMUP       warmup steps before the counter
#                       baseline snapshot (part of STEPS)   [2]
#   FM032V_DT           Euler dt                            [1e-5]
#   FM032V_SAMPLES      reference sample count              [200]
#   FM032V_SEED         body seed                           [24025]
#   FM032V_SAMPLER_SEED sample-index seed                   [24026]
#   FM032V_ALLOC_WARN   per-step host-alloc warn threshold,
#                       bytes (device threshold is 0)       [16384]
#   FM032V_OUTDIR       output directory
#                       [MATRIX_OPERATOR_REFACTOR/data/feasibility_1m_10ms]

using FastMultipole
using FastMultipole.StaticArrays
using Statistics
using LinearAlgebra
using Dates
using Printf
using Random

FastMultipole.load_cuda_radix_lifecycle!() ||
    error("CUDA radix lifecycle failed to load: $(FastMultipole.cuda_radix_status())")
using CUDA

const FM = FastMultipole

const N = parse(Int, get(ENV, "FM032V_N", "100000"))
const BETA = parse(Float64, get(ENV, "FM032V_BETA", "2.0"))
const P = parse(Int, get(ENV, "FM032V_P", "3"))
const ELL_SPEC = get(ENV, "FM032V_ELL", "auto")
const Q = parse(Int, get(ENV, "FM032V_Q", "12"))
const K = parse(Int, get(ENV, "FM032V_K", "256"))
const TFS = [t == "Float32" ? Float32 : Float64
             for t in split(get(ENV, "FM032V_TF", "Float64,Float32"), ',')]
const STEPS = parse(Int, get(ENV, "FM032V_STEPS", "5"))
const WARMUP = parse(Int, get(ENV, "FM032V_WARMUP", "2"))
const DT = parse(Float64, get(ENV, "FM032V_DT", "1e-5"))
const SAMPLES = parse(Int, get(ENV, "FM032V_SAMPLES", "200"))
const SEED = parse(Int, get(ENV, "FM032V_SEED", "24025"))
const SAMPLER_SEED = parse(Int, get(ENV, "FM032V_SAMPLER_SEED", "24026"))
const ALLOC_WARN = parse(Int, get(ENV, "FM032V_ALLOC_WARN", "16384"))
const OUTDIR = get(ENV, "FM032V_OUTDIR", joinpath(@__DIR__, "..", "data",
    "feasibility_1m_10ms"))
const STAMP = Dates.format(now(), "yyyymmdd-HHMMSS")
const JOBID = get(ENV, "SLURM_JOB_ID", "")
const CSVPATH = joinpath(OUTDIR, "cuda032v_$(gethostname())_$(STAMP).csv")
const U_GATE = 1e-3

WARMUP < STEPS || error("FM032V_WARMUP ($WARMUP) must be < FM032V_STEPS ($STEPS)")

# 028/024b box conventions: bodies in [0,1]^3, Morton box (-0.01, 1.02)
const BOX_MIN64 = SVector(-0.01, -0.01, -0.01)
const BOX_SIZE = 1.02
const CLAMP_LO = 0.0
const CLAMP_HI = 1.0

# ---- geometry: adequacy margin (mirrors _direct_kernel_geometry_gate!) ------

const RHO_T = RegularizedVortex(; sigma_row=8).rho_t   # 4.789 at eps = 1e-3
const SIGMA = BETA * Float64(N)^(-1 / 3)               # uniform, so sigma_max = SIGMA
const G_MIN = FM._ball_stencil_min_gap(Q)              # sqrt(5) for q = 12

function _auto_ell()
    # deepest ell with g_min * BOX_SIZE / 2^ell > rho_t * sigma (strict)
    x = G_MIN * BOX_SIZE / (RHO_T * SIGMA)
    ell = floor(Int, log2(x))
    2.0^ell < x || (ell -= 1)
    ell >= 2 || error("no admissible tree depth (ell >= 2) at n=$N beta=$BETA q=$Q")
    return ell
end

const ELL = ELL_SPEC == "auto" ? _auto_ell() : parse(Int, ELL_SPEC)
const H_LEAF = BOX_SIZE / (1 << ELL)
const MARGIN = G_MIN * H_LEAF / (RHO_T * SIGMA)

@printf("geometry: n=%d beta=%.3f sigma=%.6f q=%d g_min=%.4f ell=%d h_leaf=%.6f\n",
    N, BETA, SIGMA, Q, G_MIN, ELL, H_LEAF)
@printf("adequacy margin g_min*h_leaf / (rho_t*sigma_max) = %.4f (must be > 1)\n", MARGIN)
MARGIN > 1 || error("chosen geometry fails the near-set adequacy inequality; " *
    "reduce FM032V_ELL, N^{-1/3} overlap, or beta")

# ---- device-resident regularized-vortex system -------------------------------
#
# fm028_device_system.jl generalized: packed layout rows 1:3 position, 4 MAC
# radius (FLOWVPM-style inflated rho_t*sigma, distinct from sigma), 5:7 Gamma,
# 8 sigma. Output consumed on device: velocity U (gradient rows) and the
# 9-component velocity gradient J (hessian rows). `host_positions` is a
# construction-time mirror for get_position; it goes stale under convection and
# the resident lifecycle never reads it after cache construction.
struct FM032VDeviceVortex{TF,PM,VV}
    host_positions::Matrix{TF}
    positions::PM      # CuMatrix{TF} 3 x n
    mac_radii::VV      # CuVector{TF}, rho_t * sigma
    gamma::PM          # CuMatrix{TF} 3 x n
    sigma::VV          # CuVector{TF}
    velocity::PM       # CuMatrix{TF} 3 x n, overwritten each step
    jacobian::PM       # CuMatrix{TF} 9 x n, overwritten each step
end

function FM032VDeviceVortex{TF}(positions::Matrix{Float64},
        gamma::Matrix{Float64}, sigma::Vector{Float64}) where TF
    n = size(positions, 2)
    host_positions = Matrix{TF}(positions)
    return FM032VDeviceVortex(host_positions,
        CUDA.CuArray(host_positions),
        CUDA.CuArray(Vector{TF}(RHO_T .* sigma)),
        CUDA.CuArray(Matrix{TF}(gamma)),
        CUDA.CuArray(Vector{TF}(sigma)),
        CUDA.zeros(TF, 3, n),
        CUDA.zeros(TF, 9, n))
end

Base.eltype(::FM032VDeviceVortex{TF}) where TF = TF
FM.get_n_bodies(sys::FM032VDeviceVortex) = size(sys.positions, 2)
FM.data_per_body(::FM032VDeviceVortex) = 8
FM.strength_dims(::FM032VDeviceVortex) = 3
FM.get_position(sys::FM032VDeviceVortex{TF}, i) where TF =
    SVector{3,TF}(sys.host_positions[1, i], sys.host_positions[2, i],
        sys.host_positions[3, i])
FM.residency(::FM032VDeviceVortex) = DeviceResident()
FM.has_vector_potential(::FM032VDeviceVortex) = true
FM.body_type(::FM032VDeviceVortex) = Point{Vortex}
FM.direct_kernel(::FM032VDeviceVortex) = RegularizedVortex(; sigma_row=8)

# recurring device refresh passes the identity sort index
# (_canonical_cuda_source_buffer): straight device-side repack, no host traffic
function FM.source_to_buffer!(device_buffer::CUDA.AnyCuArray,
        sys::FM032VDeviceVortex, sort_index)
    (first(sort_index) == 1 && last(sort_index) == size(sys.positions, 2)) ||
        error("FM032VDeviceVortex expects the identity sort index")
    device_buffer[1:3, :] .= sys.positions
    device_buffer[4, :] .= sys.mac_radii
    device_buffer[5:7, :] .= sys.gamma
    device_buffer[8, :] .= sys.sigma
    return device_buffer
end

# overwrite (not accumulate): each step's finalize delivers the full influence
function FM.buffer_to_target!(sys::FM032VDeviceVortex,
        device_output_buffer::CUDA.AnyCuArray, derivatives_switch, sort_index)
    grange = FM.gradient_range(derivatives_switch)
    isempty(grange) || (sys.velocity .= view(device_output_buffer, grange, :))
    hrange = FM.hessian_range(derivatives_switch)
    isempty(hrange) || (sys.jacobian .= view(device_output_buffer, hrange, :))
    return sys
end

# device Euler convection: x .+= dt * U, clamped inside the fixed Morton box
function fm032v_euler!(sys::FM032VDeviceVortex{TF}, dt) where TF
    sys.positions .= clamp.(sys.positions .+ TF(dt) .* sys.velocity,
        TF(CLAMP_LO), TF(CLAMP_HI))
    return sys
end

# ---- host Float64 erf-based regularized sampled direct reference -------------
#
# Independent of the shipped erf-free _gaussianerf_g_h: stdlib-only reference
# erf (Maclaurin below x = 2, continued-fraction erfc above; the
# test/interface_test_systems.jl construction) and the theory-§1 gaussianerf
# U/J formulas with the per-source sigma. Evaluated on host in Float64 from
# downloaded (current) positions — O(samples * n), cheap at the default sizes.
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

# positions 3 x n, gamma 3 x n, sigma n (Float64, host); indices: sample body
# ids. Returns (U 3 x S, J 9 x S), self-pairs excluded.
function fm032v_reference(positions, gamma, sigma, indices)
    S = length(indices)
    n = size(positions, 2)
    U = zeros(3, S)
    J = zeros(9, S)
    A = sqrt(2 / pi)
    Threads.@threads for k in 1:S
        i = indices[k]
        xi = SVector{3,Float64}(positions[1, i], positions[2, i], positions[3, i])
        for j in 1:n
            j == i && continue
            d = xi - SVector{3,Float64}(positions[1, j], positions[2, j],
                positions[3, j])
            r2 = dot(d, d)
            r2 == 0 && continue
            G = SVector{3,Float64}(gamma[1, j], gamma[2, j], gamma[3, j])
            sig = sigma[j]
            r = sqrt(r2)
            rho = r / sig
            e = exp(-rho^2 / 2)
            g = _ref_erf(rho / sqrt(2.0)) - A * rho * e
            gp = A * rho^2 * e
            cr3 = 1 / (4pi * r2 * r)
            crss = SVector{3,Float64}(
                (d[3] * G[2] - d[2] * G[3]) * cr3,
                (d[1] * G[3] - d[3] * G[1]) * cr3,
                (d[2] * G[1] - d[1] * G[2]) * cr3)
            a = (rho * gp - 3g) / r2
            b = -g * cr3
            U[1, k] += g * crss[1]
            U[2, k] += g * crss[2]
            U[3, k] += g * crss[3]
            J[1, k] += a * crss[1] * d[1]
            J[2, k] += a * crss[2] * d[1] - b * G[3]
            J[3, k] += a * crss[3] * d[1] + b * G[2]
            J[4, k] += a * crss[1] * d[2] + b * G[3]
            J[5, k] += a * crss[2] * d[2]
            J[6, k] += a * crss[3] * d[2] - b * G[1]
            J[7, k] += a * crss[1] * d[3] - b * G[2]
            J[8, k] += a * crss[2] * d[3] + b * G[1]
            J[9, k] += a * crss[3] * d[3]
        end
    end
    return U, J
end

_rel_rms(x, ref) = sqrt(sum(abs2, Float64.(x) .- ref) /
    max(sum(abs2, ref), eps(Float64)))

# download current device state (outside any timed region) and compute the
# sampled U/J relative RMS against a fresh Float64 reference
function fm032v_accuracy(sys::FM032VDeviceVortex, indices)
    pos = Float64.(Array(sys.positions))
    gam = Float64.(Array(sys.gamma))
    sig = Float64.(Array(sys.sigma))
    U_ref, J_ref = fm032v_reference(pos, gam, sig, indices)
    U = Array(sys.velocity)[:, indices]
    J = Array(sys.jacobian)[:, indices]
    return _rel_rms(U, U_ref), _rel_rms(J, J_ref)
end

# ---- CSV ---------------------------------------------------------------------

function write_csv(path, rows)
    isempty(rows) && return
    open(path, "w") do io
        println(io, join(string.(keys(first(rows))), ','))
        for row in rows
            println(io, join(string.(values(row)), ','))
        end
    end
end

# ---- one precision -----------------------------------------------------------

function measure(::Type{TF}) where TF
    rng = MersenneTwister(SEED)
    positions = rand(rng, 3, N)
    gamma = randn(rng, 3, N) ./ N
    sigma = fill(SIGMA, N)
    indices = sort!(randperm(MersenneTwister(SAMPLER_SEED), N)[1:SAMPLES])

    sys = FM032VDeviceVortex{TF}(positions, gamma, sigma)
    opts = CUDARadixLifecycleOptions(; precision=TF,
        m2l_strategy=FM.ConcatenatedFixedZM2L())
    GC.gc(); CUDA.reclaim()
    t0 = time_ns()
    cache = RadixFMMCache(sys; expansion_order=P, ell=ELL, max_n_bodies=N,
        bounds=(SVector{3,TF}(BOX_MIN64), TF(BOX_SIZE)), hessian=true,
        near_radius2=Q, window_classes=K, device=true, options=opts)
    CUDA.synchronize()
    construction_ms = (time_ns() - t0) / 1e6

    counters = cache.state.counters
    step!() = (fmm!(sys, cache; scalar_potential=false, gradient=true,
        hessian=true); fm032v_euler!(sys, DT))

    # warm evaluation + step-0 accuracy (positions have not moved yet)
    fmm!(sys, cache; scalar_potential=false, gradient=true, hessian=true)
    u0, j0 = fm032v_accuracy(sys, indices)
    @printf("%s step0 accuracy: u_rel_rms=%.3e j_rel_rms=%.3e\n", TF, u0, j0)

    # 023 counter contract, part 1: a device-resident consumer never uploads
    # bodies or copies expansions through the host — at any point in its life
    counters.body_uploads == 0 ||
        error("body_uploads = $(counters.body_uploads) on a device-resident consumer")
    counters.expansion_host_copies == 0 ||
        error("expansion_host_copies = $(counters.expansion_host_copies)")

    base = (route=counters.route_uploads, operator=counters.operator_uploads,
        infl=counters.influence_downloads, meta=counters.metadata_downloads)

    rows = NamedTuple[]
    common = (; job=JOBID, host=gethostname(), gpu=CUDA.name(CUDA.device()),
        julia=string(VERSION), cuda=string(CUDA.runtime_version()),
        precision=string(TF), n=N, ell=ELL, near_radius2=Q, window_classes=K,
        expansion_order=P, p_literature=P + 1, beta=BETA, sigma=SIGMA,
        rho_t=RHO_T, g_min=G_MIN, h_leaf=H_LEAF, adequacy_margin=MARGIN,
        dt=DT, samples=SAMPLES, seed=SEED, sampler_seed=SAMPLER_SEED,
        construction_ms)
    push!(rows, merge(common, (; step=0, t_step_ms=NaN,
        step_device_alloc_bytes=-1, step_host_alloc_bytes=-1,
        body_uploads=counters.body_uploads, route_uploads=counters.route_uploads,
        operator_uploads=counters.operator_uploads,
        expansion_host_copies=counters.expansion_host_copies,
        influence_downloads=counters.influence_downloads,
        metadata_downloads=counters.metadata_downloads,
        u_rel_rms=u0, j_rel_rms=j0)))

    # convection loop: WARMUP steps establish steady state, then the counter
    # baseline must stay frozen for every subsequent step
    step_ms = Float64[]
    for s in 1:STEPS
        t = Float64(CUDA.@elapsed step!()) * 1e3
        push!(step_ms, t)
        counters.body_uploads == 0 || error("body upload at step $s")
        counters.expansion_host_copies == 0 || error("expansion host copy at step $s")
        if s == WARMUP
            base = (route=counters.route_uploads, operator=counters.operator_uploads,
                infl=counters.influence_downloads, meta=counters.metadata_downloads)
        elseif s > WARMUP
            counters.route_uploads == base.route || error("route upload at step $s")
            counters.operator_uploads == base.operator ||
                error("operator upload at step $s")
            counters.influence_downloads == base.infl ||
                error("influence download at step $s")
            counters.metadata_downloads == base.meta ||
                error("metadata download at step $s")
        end
        push!(rows, merge(common, (; step=s, t_step_ms=t,
            step_device_alloc_bytes=-1, step_host_alloc_bytes=-1,
            body_uploads=counters.body_uploads,
            route_uploads=counters.route_uploads,
            operator_uploads=counters.operator_uploads,
            expansion_host_copies=counters.expansion_host_copies,
            influence_downloads=counters.influence_downloads,
            metadata_downloads=counters.metadata_downloads,
            u_rel_rms=NaN, j_rel_rms=NaN)))
        @printf("%s step %d: %.3f ms\n", TF, s, t)
    end

    # steady-state allocation probe (two extra steps outside the timed loop)
    dev_alloc = CUDA.@allocated step!()
    host_alloc = @allocated step!()
    dev_alloc > 0 && @printf("WARNING: %s steady-state device allocation %d B > 0\n",
        TF, dev_alloc)
    host_alloc > ALLOC_WARN &&
        @printf("WARNING: %s steady-state host allocation %d B > %d B\n",
            TF, host_alloc, ALLOC_WARN)

    # post-convection accuracy: re-evaluate at the moved positions, fresh reference
    fmm!(sys, cache; scalar_potential=false, gradient=true, hessian=true)
    uc, jc = fm032v_accuracy(sys, indices)
    @printf("%s post-convection accuracy (%d steps): u_rel_rms=%.3e j_rel_rms=%.3e\n",
        TF, STEPS, uc, jc)
    push!(rows, merge(common, (; step=STEPS + 2, t_step_ms=NaN,
        step_device_alloc_bytes=dev_alloc, step_host_alloc_bytes=host_alloc,
        body_uploads=counters.body_uploads, route_uploads=counters.route_uploads,
        operator_uploads=counters.operator_uploads,
        expansion_host_copies=counters.expansion_host_copies,
        influence_downloads=counters.influence_downloads,
        metadata_downloads=counters.metadata_downloads,
        u_rel_rms=uc, j_rel_rms=jc)))

    steady = step_ms[(WARMUP + 1):end]
    @printf("%s median steady step: %.3f ms (min %.3f, max %.3f); dev alloc %d B, host alloc %d B\n",
        TF, median(steady), minimum(steady), maximum(steady), dev_alloc, host_alloc)

    gate_err = max(u0, uc)
    cache = nothing
    sys = nothing
    GC.gc(); CUDA.reclaim()
    return rows, gate_err
end

# ---- sweep -------------------------------------------------------------------

all_rows = NamedTuple[]
gate_failures = String[]
for TF in TFS
    rows, gate_err = measure(TF)
    append!(all_rows, rows)
    if TF === Float64 && !(gate_err <= U_GATE)
        push!(gate_failures,
            "Float64 velocity gate FAILED: u_rel_rms $gate_err > $U_GATE")
    end
end

mkpath(OUTDIR)
write_csv(CSVPATH, all_rows)
println("wrote ", CSVPATH)

for msg in gate_failures
    println(msg)
end
isempty(gate_failures) || error(join(gate_failures, "; "))
println("032 stage-4 validation PASSED")
