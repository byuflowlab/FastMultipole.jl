# fm041k_direct_bruteforce.jl — task 041k: brute-force direct UJ(+SFS) ceiling
# on one H200. Naive all-pairs O(N²) evaluation of the realistic FLOWVPM
# workload, N = round(10^e), e = 2.0:0.5:7.5, per series stopping after the
# first N whose median wall time exceeds 10 s.
#
# Series: {uj, ujsfs} × {naive, tiled} × {Float64, Float32(+rsqrt)}.
#   uj    = one all-pairs pass: gaussianerf-regularized velocity U(3) + full
#           Jacobian J(9), transcribed from ../FLOWVPM.jl/src/FLOWVPM_fmm.jl:132-198
#           with g_dgdr from FLOWVPM_kernel.jl:54-57.
#   ujsfs = uj pass + O(N) per-source T_q(Γ_q) precompute + fused ζ pass
#           accumulating Ω = Σ ζ Γ_q and Q = Σ ζ T_q(Γ_q) (041b §1.2 factorized
#           identity; ζ from FLOWVPM_kernel.jl:51, Estr transposed scheme from
#           FLOWVPM_subfilterscale_models.jl:16-41). E_p = T_p(Ω_p) − Q_p is
#           formed on host, O(N), excluded from timing. Self-pair kept in the
#           ζ pass (its contribution to E cancels identically); self-pair
#           skipped in the UJ pass as in production.
#
# Timing: median of 5 synchronized reps after 2 warm-ups; if the first warm
# rep exceeds 15 s, 1 warm-up + 2 reps (recorded in the `reps` column).
# H2D/D2H measured separately. Accuracy: at N = 1e3 and 1e4, tiled GPU vs a
# threaded Float64 CPU reference of the identical formulas; F64 gate
# max rel err ≤ 1e-11 per output block; the CPU reference also cross-checks
# the 041b reordered E against the original pairwise Estr form.
#
# Bodies (pre-registered): MersenneTwister(123), positions uniform in the
# unit cube, Γ components U(-1,1)/N, uniform σ = 2 N^(-1/3).
#
# CUDA-dependent code lives in fm041k_direct_bruteforce_gpu.jl, included only
# when CUDA loads and is functional; otherwise this script runs the CPU
# reference + 041b-identity self-test only (local smoke mode).
#
# Usage: julia --project=<env> -t <threads> fm041k_direct_bruteforce.jl [outdir]

using Random
using Statistics
using Printf

include(joinpath(@__DIR__, "fm041k_erf_vendored.jl"))

const OUTDIR = length(ARGS) >= 1 ? ARGS[1] :
    joinpath(@__DIR__, "..", "data", "direct_bruteforce_ceiling")
mkpath(OUTDIR)
const JOBID = get(ENV, "SLURM_JOB_ID", "local")

const TIME_CAP_S = 10.0
const EXPS = 2.0:0.5:7.5
const TILE = 128
const ACC_NS = (1_000, 10_000)

# kernel variants to sweep; optional ARGS[2] = comma list (e.g. "tiled,opt")
# restricts the sweep and redirects it to sweep_<list>.csv so a partial rerun
# never clobbers the full sweep.csv
const ALL_VARIANTS = (:naive, :tiled, :opt)
const VARIANTS = length(ARGS) >= 2 ? Tuple(Symbol.(split(ARGS[2], ","))) : ALL_VARIANTS
const SWEEP_CSV = VARIANTS == ALL_VARIANTS ? "sweep.csv" :
    "sweep_" * join(String.(VARIANTS), "_") * ".csv"

# gaussianerf constants (FLOWVPM.jl definitions of const1/const2/const4/sqr2)
const K1 = 1 / (2π)^1.5      # ζ normalization
const K2 = sqrt(2 / π)
const K4 = 1 / (4π)
const SQR2 = sqrt(2.0)

# Nominal non-transcendental flops per pair (add/mul counted 1 each, from the
# pair functions below); the one exp (+ one erf for uj) counted separately.
const FLOPS_UJ = 78
const FLOPS_ZETA = 26

# --- pair math (shared verbatim by CPU reference and device kernels) --------

@inline _rinv_cpu(r2) = one(r2) / sqrt(r2)

# FLOWVPM_kernel.jl:54-57 (g_dgdr_gauserf), typed for F32 kernel stability
@inline function g_dgdr(rho::T) where T
    aux = T(K2) * rho * exp(-rho * rho / 2)
    return custom_erf(rho / T(SQR2)) - aux, rho * aux
end

# FLOWVPM_fmm.jl:148-192 per-pair U and J (column-major J: du_i/dx_j at
# J[(j-1)*3+i]); rinv passed in so CPU/GPU share everything else
@inline function uj_pair(dx::T, dy::T, dz::T, r2::T, rinv::T,
                         gx::T, gy::T, gz::T, sigma_inv::T) where T
    r = r2 * rinv
    g, dgdr = g_dgdr(r * sigma_inv)
    return uj_tail(dx, dy, dz, r2, rinv, g, dgdr, gx, gy, gz, sigma_inv)
end

# tail shared by the reference path (g_dgdr above) and the opt kernels'
# far-field-switched path (g = 1, dgdr = 0 beyond the saturation cutoff)
@inline function uj_tail(dx::T, dy::T, dz::T, r2::T, rinv::T, g::T, dgdr::T,
                         gx::T, gy::T, gz::T, sigma_inv::T) where T
    r3inv = rinv * rinv * rinv
    c = -T(K4) * r3inv
    crss1 = c * (dy * gz - dz * gy)
    crss2 = c * (dz * gx - dx * gz)
    crss3 = c * (dx * gy - dy * gx)
    ux = g * crss1
    uy = g * crss2
    uz = g * crss3
    aux = dgdr * sigma_inv * rinv - 3 * g * rinv * rinv
    aux2 = c * g
    j11 = aux * crss1 * dx
    j21 = aux * crss2 * dx - aux2 * gz
    j31 = aux * crss3 * dx + aux2 * gy
    j12 = aux * crss1 * dy + aux2 * gz
    j22 = aux * crss2 * dy
    j32 = aux * crss3 * dy - aux2 * gx
    j13 = aux * crss1 * dz - aux2 * gy
    j23 = aux * crss2 * dz + aux2 * gx
    j33 = aux * crss3 * dz
    return ux, uy, uz, j11, j21, j31, j12, j22, j32, j13, j23, j33
end

# ζ_σ(r) from r² only (no sqrt needed): ζ(ρ) σ⁻³ with ρ² = r² σ⁻²
@inline zeta_sgm_r2(r2::T, si::T, si3::T) where T = T(K1) * exp(-r2 * si * si / 2) * si3

# --- body generation --------------------------------------------------------

function gen_bodies(n)
    rng = MersenneTwister(123)
    P = Matrix{Float64}(rand(rng, n, 3)')            # 3×n positions, unit cube
    G = Matrix{Float64}((rand(rng, n, 3)' .- 0.5) .* (2.0 / n))
    sigma = 2.0 * n^(-1 / 3)
    return P, G, sigma
end

# --- CPU Float64 reference (threaded) ---------------------------------------

function cpu_uj!(U, J, P, G, si)
    n = size(P, 2)
    Threads.@threads for i in 1:n
        xi, yi, zi = P[1, i], P[2, i], P[3, i]
        u1 = u2 = u3 = 0.0
        j1 = j2 = j3 = j4 = j5 = j6 = j7 = j8 = j9 = 0.0
        for q in 1:n
            dx = xi - P[1, q]; dy = yi - P[2, q]; dz = zi - P[3, q]
            r2 = dx * dx + dy * dy + dz * dz
            if r2 > 0.0
                ux, uy, uz, a1, a2, a3, a4, a5, a6, a7, a8, a9 =
                    uj_pair(dx, dy, dz, r2, _rinv_cpu(r2), G[1, q], G[2, q], G[3, q], si)
                u1 += ux; u2 += uy; u3 += uz
                j1 += a1; j2 += a2; j3 += a3; j4 += a4; j5 += a5
                j6 += a6; j7 += a7; j8 += a8; j9 += a9
            end
        end
        U[1, i], U[2, i], U[3, i] = u1, u2, u3
        J[1, i], J[2, i], J[3, i] = j1, j2, j3
        J[4, i], J[5, i], J[6, i] = j4, j5, j6
        J[7, i], J[8, i], J[9, i] = j7, j8, j9
    end
end

# transposed scheme: T(Γ)_k = Σ_i J[3(k-1)+i] Γ_i  (FLOWVPM_subfilterscale_models.jl:24-26)
function cpu_tg!(TG, J, G)
    n = size(G, 2)
    for i in 1:n
        g1, g2, g3 = G[1, i], G[2, i], G[3, i]
        TG[1, i] = J[1, i] * g1 + J[2, i] * g2 + J[3, i] * g3
        TG[2, i] = J[4, i] * g1 + J[5, i] * g2 + J[6, i] * g3
        TG[3, i] = J[7, i] * g1 + J[8, i] * g2 + J[9, i] * g3
    end
end

function cpu_zeta!(OM, Q, P, G, TG, si)
    n = size(P, 2)
    si3 = si^3
    Threads.@threads for i in 1:n
        xi, yi, zi = P[1, i], P[2, i], P[3, i]
        o1 = o2 = o3 = q1 = q2 = q3 = 0.0
        for q in 1:n
            dx = xi - P[1, q]; dy = yi - P[2, q]; dz = zi - P[3, q]
            z = zeta_sgm_r2(dx * dx + dy * dy + dz * dz, si, si3)
            o1 += z * G[1, q]; o2 += z * G[2, q]; o3 += z * G[3, q]
            q1 += z * TG[1, q]; q2 += z * TG[2, q]; q3 += z * TG[3, q]
        end
        OM[1, i], OM[2, i], OM[3, i] = o1, o2, o3
        Q[1, i], Q[2, i], Q[3, i] = q1, q2, q3
    end
end

# E_p = T_p(Ω_p) − Q_p (041b §1.2 reordered form, transposed scheme)
function form_E(J, OM, Q)
    n = size(OM, 2)
    E = zeros(Float64, 3, n)
    for i in 1:n
        o1, o2, o3 = OM[1, i], OM[2, i], OM[3, i]
        E[1, i] = J[1, i] * o1 + J[2, i] * o2 + J[3, i] * o3 - Q[1, i]
        E[2, i] = J[4, i] * o1 + J[5, i] * o2 + J[6, i] * o3 - Q[2, i]
        E[3, i] = J[7, i] * o1 + J[8, i] * o2 + J[9, i] * o3 - Q[3, i]
    end
    return E
end

# original pairwise Estr form (FLOWVPM_subfilterscale_models.jl:16-41),
# identity cross-check against form_E
function cpu_estr_pairwise(P, G, J, si)
    n = size(P, 2)
    si3 = si^3
    E = zeros(3, n)
    Threads.@threads for i in 1:n
        e1 = e2 = e3 = 0.0
        for q in 1:n
            dx = P[1, i] - P[1, q]; dy = P[2, i] - P[2, q]; dz = P[3, i] - P[3, q]
            r2 = dx * dx + dy * dy + dz * dz
            z = zeta_sgm_r2(r2, si, si3)
            g1, g2, g3 = G[1, q], G[2, q], G[3, q]
            s1 = (J[1, i] - J[1, q]) * g1 + (J[2, i] - J[2, q]) * g2 + (J[3, i] - J[3, q]) * g3
            s2 = (J[4, i] - J[4, q]) * g1 + (J[5, i] - J[5, q]) * g2 + (J[6, i] - J[6, q]) * g3
            s3 = (J[7, i] - J[7, q]) * g1 + (J[8, i] - J[8, q]) * g2 + (J[9, i] - J[9, q]) * g3
            e1 += z * s1; e2 += z * s2; e3 += z * s3
        end
        E[1, i], E[2, i], E[3, i] = e1, e2, e3
    end
    return E
end

function relerr(A, R)
    s = maximum(abs, R)
    s == 0 && return (0.0, 0.0)
    d = Float64.(A) .- R
    return maximum(abs, d) / s, sqrt(mean(abs2, d)) / s
end

# One CPU Float64 reference set (all blocks) for a given n
struct RefBlocks
    n::Int
    P::Matrix{Float64}; G::Matrix{Float64}; si::Float64
    U::Matrix{Float64}; J::Matrix{Float64}; TG::Matrix{Float64}
    OM::Matrix{Float64}; Q::Matrix{Float64}; E::Matrix{Float64}
end

function build_reference(n)
    P, G, sigma = gen_bodies(n)
    si = 1.0 / sigma
    U = zeros(3, n); J = zeros(9, n); TG = zeros(3, n)
    OM = zeros(3, n); Q = zeros(3, n)
    cpu_uj!(U, J, P, G, si)
    cpu_tg!(TG, J, G)
    cpu_zeta!(OM, Q, P, G, TG, si)
    E = form_E(J, OM, Q)
    E2 = cpu_estr_pairwise(P, G, J, si)
    return RefBlocks(n, P, G, si, U, J, TG, OM, Q, E), E2
end

# --- conditional GPU load ---------------------------------------------------

const HAVE_CUDA = try
    @eval using CUDA
    true
catch err
    @warn "CUDA.jl unavailable; CPU self-test only" err
    false
end
const GPU_READY = HAVE_CUDA && CUDA.functional()
GPU_READY && include(joinpath(@__DIR__, "fm041k_direct_bruteforce_gpu.jl"))

# --- main -------------------------------------------------------------------

function main()
    @printf("fm041k brute-force ceiling  jobid=%s  threads=%d  gpu=%s\n",
            JOBID, Threads.nthreads(), string(GPU_READY))
    GPU_READY && println(CUDA.name(CUDA.device()))
    gate_ok = true
    open(joinpath(OUTDIR, "accuracy.csv"), "w") do io
        println(io, "n,precision,block,max_rel_err,rms_rel_err")
        for n in ACC_NS
            ref, E2 = build_reference(n)
            me, re = relerr(E2, ref.E)
            @printf(io, "%d,f64,identity_E,%.3e,%.3e\n", n, me, re)
            @printf("identity check n=%d: reordered-vs-pairwise E max rel %.3e\n", n, me)
            me <= 1e-11 || (gate_ok = false)
            if GPU_READY
                gate_ok &= Base.invokelatest(gpu_accuracy!, io, ref)
            end
        end
    end
    if !GPU_READY
        println("no functional GPU: accuracy/identity self-test finished (smoke mode)")
        return
    end
    gate_ok || error("041k F64 accuracy gate FAILED (max rel err > 1e-11); see accuracy.csv")
    println("F64 accuracy gate passed; starting sweep: ", join(String.(VARIANTS), ","))
    open(io -> Base.invokelatest(run_sweep, io, VARIANTS), joinpath(OUTDIR, SWEEP_CSV), "w")
    println("done")
end

# run only as a script; fm041k_crosscheck_flowvpm.jl includes this file as a library
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
