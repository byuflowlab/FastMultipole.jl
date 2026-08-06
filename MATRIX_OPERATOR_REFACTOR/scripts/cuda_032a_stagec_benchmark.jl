# Task 032a Stage C: H200 mechanism-selection benchmark for the distance-binned
# nearfield pair stream (031a §6.3) and the TwoPassVortex pass-2 deficit sweep.
#
# Measures, at overlap-2 uniform-cube operating points with adequate near-set
# geometry, for both precisions:
#   - RegularizedVortex (032 regularized-everywhere baseline)
#   - PartitionedVortex × binning mechanism {unbinned, classsplit, ballot,
#     classsplit_ballot} × subsort {0, 1}
#   - TwoPassVortex × {unbinned, classsplit} × pass-2 {predicated, queued}
# reporting steady-state full-step time, isolated nearfield-stage time (fill +
# pass kernels, overlap disabled), achieved warp homogeneity (all/mixed/shell),
# bucket occupancies, and a sampled-direct Float64 relative RMS error per
# configuration (u gate diagnostic).
#
# Output: one CSV per operating point under FM032A_OUTDIR (default
# MATRIX_OPERATOR_REFACTOR/data/split_nearfield). Env knobs:
#   FM032A_POINTS   comma list among a,b,c   (default a,b,c)
#   FM032A_REPS     timing reps              (default 20)
#   FM032A_SAMPLES  sampled-direct targets   (default 200)

using FastMultipole
using FastMultipole.StaticArrays
using Random
using Printf
using Statistics

const FM = FastMultipole
FM.load_cuda_radix_lifecycle!() || error("CUDA radix lifecycle failed to load: " *
    FM.cuda_radix_status())
using CUDA

const TESTDIR = joinpath(dirname(@__DIR__), "..", "test")
isdefined(@__MODULE__, :VortexParticles) || include(joinpath(TESTDIR, "vortex.jl"))
isdefined(@__MODULE__, :ExtendedVortex) ||
    include(joinpath(TESTDIR, "interface_test_systems.jl"))

const OUTDIR = get(ENV, "FM032A_OUTDIR",
    joinpath(dirname(@__DIR__), "data", "split_nearfield"))
mkpath(OUTDIR)
const REPS = parse(Int, get(ENV, "FM032A_REPS", "20"))
const NSAMPLE = parse(Int, get(ENV, "FM032A_SAMPLES", "200"))
const POINTS = split(get(ENV, "FM032A_POINTS", "a,b,c"), ",")

# operating points: overlap-2 uniform cube, adequate rigid near sets
# (031a §5.1/§5.2; q = near_radius2). ell/q per the 032/032a records:
#   a: the Stage-D cube geometry (n = 1e5, ell = 3, q = 16)
#   b: deep point at n = 1e6 (ell = 4 is the deepest adequate depth at
#      supported radii; ell = 5 needs |o|² ≤ 22 > 20)
#   c: mid point with the largest constructible regularized fraction
const POINT_DEFS = Dict(
    "a" => (n=100_000, ell=3, q=16),
    "b" => (n=1_000_000, ell=4, q=12),
    "c" => (n=200_000, ell=4, q=20),
)

function make_system(ctor, seed, n)
    sigma_u = 2.0 * (1.0 / n)^(1 / 3)   # overlap 2 vs mean spacing, unit cube
    base = generate_vortex(seed, n)
    sm = SmoothedVortex(base, fill(sigma_u, n))
    return ctor === SmoothedVortex ? sm : ctor(sm)
end

# Float64 erf-based regularized U/J at sampled targets against all sources
# (the kernel-independent physics truth all three candidates approximate)
function sampled_reference(sys::SmoothedVortex, sample::Vector{Int})
    n = FM.get_n_bodies(sys)
    U = zeros(3, length(sample))
    J = zeros(9, length(sample))
    A = sqrt(2 / pi)
    pos = sys.inner.bodies
    Threads.@threads for si in eachindex(sample)
        i = sample[si]
        xi = FM.get_position(sys, i)
        for j in 1:n
            i == j && continue
            d = xi - FM.get_position(sys, j)
            r2 = d[1]^2 + d[2]^2 + d[3]^2
            r2 == 0 && continue
            G = pos[j].strength
            sigma = Float64(sys.sigma[j])
            r = sqrt(r2)
            rho = r / sigma
            g = _ref_erf(rho / sqrt(2)) - A * rho * exp(-rho^2 / 2)
            gp = A * rho^2 * exp(-rho^2 / 2)
            cr3 = 1 / (4pi * r2 * r)
            c1 = (d[3] * G[2] - d[2] * G[3]) * cr3
            c2 = (d[1] * G[3] - d[3] * G[1]) * cr3
            c3 = (d[2] * G[1] - d[1] * G[2]) * cr3
            a = (rho * gp - 3g) / r2
            b = -g * cr3
            U[1, si] += g * c1; U[2, si] += g * c2; U[3, si] += g * c3
            J[1, si] += a * c1 * d[1]
            J[2, si] += a * c2 * d[1] - b * G[3]
            J[3, si] += a * c3 * d[1] + b * G[2]
            J[4, si] += a * c1 * d[2] + b * G[3]
            J[5, si] += a * c2 * d[2]
            J[6, si] += a * c3 * d[2] - b * G[1]
            J[7, si] += a * c1 * d[3] - b * G[2]
            J[8, si] += a * c2 * d[3] + b * G[1]
            J[9, si] += a * c3 * d[3]
        end
    end
    return U, J
end

rel_rms(got, ref) = sqrt(mean(abs2, got .- ref)) / sqrt(mean(abs2, ref))

inner_of(sys::SmoothedVortex) = sys.inner
inner_of(sys::PartitionedSmoothedVortex) = sys.smoothed.inner
inner_of(sys::TwoPassSmoothedVortex) = sys.smoothed.inner

function run_config!(io, point, def, TF, ctor, kernel_name, mode, subsort,
        pass2_queued, seed, sample, Uref, Jref)
    FM.CUDA_NEARFIELD_BINNING[] = mode
    FM.CUDA_NEARFIELD_SUBSORT[] = subsort
    FM.CUDA_TWOPASS_PASS2_QUEUED[] = pass2_queued
    sys = make_system(ctor, seed, def.n)
    opts = CUDARadixLifecycleOptions(; precision=TF,
        m2l_strategy=FM.ConcatenatedFixedZM2L())
    cache = RadixFMMCache(sys; expansion_order=3, ell=def.ell,
        near_radius2=def.q, hessian=true, device=true, options=opts)
    step!() = fmm!(sys, cache; scalar_potential=false, gradient=true, hessian=true)
    # warm (JIT + epoch + graph record)
    step!(); step!(); step!()
    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:REPS
        step!()
    end
    CUDA.synchronize()
    step_ms = (time_ns() - t0) / 1e6 / REPS
    # isolated nearfield stage (fill + σ/bin/pass kernels + pass 2), overlapped
    # stream disabled so the measurement is the stage itself
    st = cache.state
    FM._launch_cuda_nearfield_kernel!(st)
    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:REPS
        FM._launch_cuda_nearfield_kernel!(st)
    end
    CUDA.synchronize()
    nearfield_ms = (time_ns() - t0) / 1e6 / REPS
    # restore a clean output for the error sample
    step!()
    CUDA.synchronize()
    inner = inner_of(sys)
    Ugot = Float64.(inner.gradient_stretching[1:3, sample])
    Jgot = Float64.(inner.potential[5:13, sample])
    u_rms = rel_rms(Ugot, Uref)
    j_rms = rel_rms(Jgot, Jref)
    # homogeneity telemetry (diagnostic launches)
    hom_all = hom_mixed = hom_shell = reg_frac = NaN
    b1 = b2 = b3 = -1
    if ctor !== SmoothedVortex
        h = FM.cuda_nearfield_homogeneity(st; stream=:all)
        hom_all = h.homogeneous_fraction
        reg_frac = h.regularized_fraction
        hm = FM.cuda_nearfield_homogeneity(st; stream=:mixed)
        hom_mixed = hm.instants == 0 ? 1.0 : hm.homogeneous_fraction
        nfctx = FM._cache_nearfield_bin_ctx(cache)
        counts = Array(nfctx.bin_counts)   # valid after the :mixed classification
        b1, b2, b3 = Int.(counts)
        if ctor === TwoPassSmoothedVortex
            hs = FM.cuda_twopass_shell_homogeneity(st)
            hom_shell = hs.instants == 0 ? 1.0 : hs.homogeneous_fraction
        end
    end
    n_direct = st.counts.n_direct
    @printf(io, "%s,%d,%d,%d,%s,%s,%s,%d,%d,%d,%.6f,%.6f,%.4e,%.4e,%.4f,%.4f,%.4f,%.4f,%d,%d,%d\n",
        point, def.n, def.ell, def.q, TF === Float64 ? "F64" : "F32",
        kernel_name, String(mode), subsort ? 1 : 0, pass2_queued ? 1 : 0,
        n_direct, step_ms, nearfield_ms, u_rms, j_rms, hom_all, hom_mixed,
        hom_shell, reg_frac, b1, b2, b3)
    flush(io)
    @printf("  %-12s %-18s sub=%d p2q=%d %s: step %.3f ms  nearfield %.3f ms  u_rms %.2e  hom(all/mixed/shell) %.3f/%.3f/%.3f\n",
        kernel_name, String(mode), subsort ? 1 : 0, pass2_queued ? 1 : 0,
        TF === Float64 ? "F64" : "F32", step_ms, nearfield_ms, u_rms, hom_all,
        hom_mixed, hom_shell)
    CUDA.reclaim()
    return nothing
end

host = gethostname()
job = get(ENV, "SLURM_JOB_ID", "manual")
for point in POINTS
    haskey(POINT_DEFS, point) || (println("unknown point $point"); continue)
    def = POINT_DEFS[point]
    seed = 20260807
    println("=== point $point: n=$(def.n) ell=$(def.ell) q=$(def.q)")
    ref_sys = make_system(SmoothedVortex, seed, def.n)
    sample = unique(round.(Int, range(1, def.n; length=min(NSAMPLE, def.n))))
    print("  sampled-direct reference ($(length(sample)) targets)... ")
    tref = @elapsed Uref, Jref = sampled_reference(ref_sys, sample)
    @printf("%.1f s\n", tref)
    out = joinpath(OUTDIR, "cuda032a_stagec_$(point)_n$(def.n)_$(host)_$(job).csv")
    open(out, "w") do io
        println(io, "point,n,ell,q,tf,kernel,mode,subsort,pass2_queued,n_direct," *
            "step_ms,nearfield_ms,u_rel_rms,j_rel_rms,hom_all,hom_mixed," *
            "hom_shell,reg_frac,bucket_singular,bucket_regularized,bucket_mixed")
        for TF in (Float32, Float64)
            # baseline: regularized everywhere (no binning applies)
            run_config!(io, point, def, TF, SmoothedVortex, "regularized",
                :unbinned, false, false, seed, sample, Uref, Jref)
            for mode in (:unbinned, :classsplit, :ballot, :classsplit_ballot),
                    subsort in (false, true)
                run_config!(io, point, def, TF, PartitionedSmoothedVortex,
                    "partitioned", mode, subsort, false, seed, sample, Uref, Jref)
            end
            for mode in (:unbinned, :classsplit), p2q in (false, true)
                run_config!(io, point, def, TF, TwoPassSmoothedVortex,
                    "twopass", mode, false, p2q, seed, sample, Uref, Jref)
            end
        end
    end
    println("  wrote $out")
end
println("cuda_032a_stagec_benchmark complete")
