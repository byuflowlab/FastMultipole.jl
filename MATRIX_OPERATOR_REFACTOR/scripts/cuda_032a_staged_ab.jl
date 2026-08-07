# Task 032a Stage D: H200 A/B ladder of the three resident vortex-nearfield
# strategies at FIXED adequate geometry per case (opposite depth trends, 031a
# §6.1 λ* table — never per-strategy optima):
#
#   - RegularizedVortex        (032 regularized-everywhere baseline)
#   - PartitionedVortex        (Stage-C-tuned: classsplit + sub-Morton sort)
#   - TwoPassVortex            (Stage-C-tuned: classsplit pass 1, predicated
#                               pass 2, sub-Morton sort)
#
# per case (overlap-2 unit cube; 033 helical wake cylinder) and precision
# (F32/F64), plus:
#   - the §6.4 RMS-radius lever rho_t = 4.252 for both split strategies
#     (deliverable 4; adopt only if sampled-direct confirms on BOTH cases);
#   - the user-endorsed rider: a partitioned unbinned-vs-classsplit mechanism
#     spot check on the WAKE geometry (Stage C measured cubes only);
#   - inline contract gates per configuration: flat 023 transfer counters and
#     steady-state device-allocation stability.
#
# Per row: steady-state step time, isolated nearfield-stage time, sampled
# erf-based Float64 relative RMS errors (u gates the winner at 1e-3; J is
# diagnostic), and homogeneity for the rider rows. Geometry adequacy ladder:
# if the BASELINE F64 u_rel_rms misses the 1e-3 gate, escalate q (16 -> 20)
# and then reduce ell, so every recorded comparison sits at a passing
# geometry (Stage-C note: q=12 at n=1e6 measures 1.49e-3 — never used here).
#
# Sentinel policy (task file deliverable 6): the primary ladder runs the two
# n≈1e5 cases; n≈{1e3,1e6} sentinel cases run only when FM032A_SENTINELS=1
# (submitted as a follow-up if the primary strategies land within 10% or a
# crossover is modeled).
#
# Output: cuda032a_staged_<case>_*.csv under FM032A_OUTDIR
# (default MATRIX_OPERATOR_REFACTOR/data/split_nearfield).

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
const SENTINELS = get(ENV, "FM032A_SENTINELS", "0") == "1"
const SEED = 20260807

# --- case geometry (033 conventions; wake per benchmark_033_common.jl) -------

const WAKE_R = 0.5
const WAKE_LEN = 5.0
const WAKE_PITCH = 2 * WAKE_R
wake_volume() = pi * WAKE_R^2 * WAKE_LEN
wake_sigma(n) = 2.0 * (wake_volume() / n)^(1 / 3)
cube_sigma(n) = 2.0 * (1.0 / n)^(1 / 3)

# cube positions in [0,1]^3 (VortexParticles convention); wake centred at the
# origin, axis z, uniform in the cylinder volume, helix-tangent strengths with
# tip-weighted magnitude |Γ| ∝ r (033 wake amendment)
function build_case(case::String, n::Int)
    rng = MersenneTwister(SEED + n + (case == "wake" ? 7919 : 0))
    if case == "cube"
        base = generate_vortex(SEED, n)   # rand positions in [0,1]^3
        return SmoothedVortex(base, fill(cube_sigma(n), n))
    end
    position = Matrix{Float64}(undef, 3, n)
    strength = Matrix{Float64}(undef, 3, n)
    for i in 1:n
        r = WAKE_R * sqrt(rand(rng))
        theta = 2pi * rand(rng)
        z = WAKE_LEN * (rand(rng) - 0.5)
        position[1, i] = r * cos(theta)
        position[2, i] = r * sin(theta)
        position[3, i] = z
        t = (-r * sin(theta), r * cos(theta), WAKE_PITCH / (2pi))
        tnrm = sqrt(sum(abs2, t))
        mag = (r / WAKE_R) / n
        strength[1, i] = mag * t[1] / tnrm
        strength[2, i] = mag * t[2] / tnrm
        strength[3, i] = mag * t[3] / tnrm
    end
    base = VortexParticles(position, strength, zeros(n))
    return SmoothedVortex(base, fill(wake_sigma(n), n))
end

# primary cases + optional sentinels; (ell, q) start points are the adequate
# geometries (q=16 everywhere per the Stage-C accuracy note), escalated by the
# ladder below only if the baseline misses the gate
# wc: window_classes override — at ell=6 the default whole-level windows size
# the route-capacity arrays at min(K·max_level_nodes, ·) ≈ 22 GB per cache
# (job 13065443's free-memory rejection); K = 64 keeps every wake1e6 cache a
# few GB with identical steady-state math (windows only batch generation)
const CASE_DEFS = Dict(
    "cube1e5" => (case="cube", n=100_000, ell=3, q=16, wc=nothing),
    "wake1e5" => (case="wake", n=100_000, ell=5, q=16, wc=nothing),
    "cube1e3" => (case="cube", n=1_000, ell=1, q=16, wc=nothing),
    "cube1e6" => (case="cube", n=1_000_000, ell=4, q=16, wc=nothing),
    "wake1e3" => (case="wake", n=1_000, ell=3, q=16, wc=nothing),
    "wake1e6" => (case="wake", n=1_000_000, ell=6, q=16, wc=64),
)
const PRIMARY = ("cube1e5", "wake1e5")
const SENTINEL = ("cube1e3", "cube1e6", "wake1e3", "wake1e6")

# --- sampled erf-based Float64 reference (kernel-independent truth) ----------

function sampled_reference(sys::SmoothedVortex, sample::Vector{Int})
    n = FM.get_n_bodies(sys)
    U = zeros(3, length(sample))
    J = zeros(9, length(sample))
    A = sqrt(2 / pi)
    bodies = sys.inner.bodies
    Threads.@threads for si in eachindex(sample)
        i = sample[si]
        xi = FM.get_position(sys, i)
        for j in 1:n
            i == j && continue
            d = xi - FM.get_position(sys, j)
            r2 = d[1]^2 + d[2]^2 + d[3]^2
            r2 == 0 && continue
            G = bodies[j].strength
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

# --- strategy wrappers -------------------------------------------------------

struct StratDef
    name::String
    wrap                # SmoothedVortex -> system
    mode::Symbol
    subsort::Bool
    rho_t::Float64
end

wrap_reg(sm) = sm
wrap_part(rho_t) = sm -> PartitionedSmoothedVortexRT(sm, rho_t)
wrap_two(rho_t) = sm -> TwoPassSmoothedVortexRT(sm, rho_t)

# rho_t-parameterized trait wrappers (the test systems fix rho_t = 4.789)
struct PartitionedSmoothedVortexRT{TF}
    smoothed::SmoothedVortex{TF}
    rho_t::Float64
end
struct TwoPassSmoothedVortexRT{TF}
    smoothed::SmoothedVortex{TF}
    rho_t::Float64
end
for S in (:PartitionedSmoothedVortexRT, :TwoPassSmoothedVortexRT)
    @eval begin
        FM.source_system_to_buffer!(buffer, i_buffer, system::$S, i_body) =
            FM.source_system_to_buffer!(buffer, i_buffer, system.smoothed, i_body)
        FM.data_per_body(::$S) = 8
        FM.get_position(system::$S, i) = FM.get_position(system.smoothed, i)
        FM.strength_dims(::$S) = 3
        FM.get_n_bodies(system::$S) = FM.get_n_bodies(system.smoothed)
        FM.has_vector_potential(::$S) = true
        FM.body_type(::$S) = FM.Point{FM.Vortex}
        FM.buffer_to_target_system!(system::$S, i_target, switch, buffer, i_buffer) =
            FM.buffer_to_target_system!(system.smoothed, i_target, switch, buffer, i_buffer)
    end
end
FM.direct_kernel(s::PartitionedSmoothedVortexRT) =
    PartitionedVortex(; sigma_row=8, rho_t=s.rho_t)
FM.direct_kernel(s::TwoPassSmoothedVortexRT) =
    TwoPassVortex(; sigma_row=8, rho_t=s.rho_t)

inner_of(sys::SmoothedVortex) = sys.inner
inner_of(sys) = sys.smoothed.inner

function strategy_matrix(case::String)
    rt = 4.789
    rms = 4.252
    strats = StratDef[
        StratDef("regularized", wrap_reg, :classsplit, true, rt),
        StratDef("partitioned", wrap_part(rt), :classsplit, true, rt),
        StratDef("twopass", wrap_two(rt), :classsplit, true, rt),
        StratDef("partitioned_rms", wrap_part(rms), :classsplit, true, rms),
        StratDef("twopass_rms", wrap_two(rms), :classsplit, true, rms),
    ]
    if case == "wake"
        # rider: mechanism spot check on the wake geometry
        push!(strats, StratDef("partitioned_unbinned", wrap_part(rt), :unbinned, false, rt))
        push!(strats, StratDef("partitioned_cs_nosub", wrap_part(rt), :classsplit, false, rt))
    end
    return strats
end

# --- one configuration -------------------------------------------------------

function run_strategy!(io, label, def, ell, q, TF, sd::StratDef, sample, Uref, Jref)
    FM.CUDA_NEARFIELD_BINNING[] = sd.mode
    FM.CUDA_NEARFIELD_SUBSORT[] = sd.subsort
    FM.CUDA_TWOPASS_PASS2_QUEUED[] = false
    sys = sd.wrap(build_case(def.case, def.n))
    opts = CUDARadixLifecycleOptions(; precision=TF,
        m2l_strategy=FM.DenseTranslationM2L(apply_chunk=64, build_chunk=8))
    cache = RadixFMMCache(sys; expansion_order=3, ell=ell, near_radius2=q,
        window_classes=def.wc, hessian=true, device=true, options=opts)
    step!() = fmm!(sys, cache; scalar_potential=false, gradient=true, hessian=true)
    step!(); step!(); step!()
    CUDA.synchronize()
    # 023 counter gate: route/operator uploads constant, no expansion copies
    counters = cache.state.counters
    r0, o0 = counters.route_uploads, counters.operator_uploads
    step!()
    CUDA.synchronize()
    counters.route_uploads == r0 && counters.operator_uploads == o0 &&
        counters.expansion_host_copies == 0 ||
        error("023 counter contract violated for $(sd.name) at $label/$TF")
    # allocation stability gate
    a1 = @eval CUDA.@allocated fmm!($sys, $cache; scalar_potential=false,
        gradient=true, hessian=true)
    a2 = @eval CUDA.@allocated fmm!($sys, $cache; scalar_potential=false,
        gradient=true, hessian=true)
    a1 == a2 || error("steady-state allocation unstable ($a1 vs $a2) for $(sd.name)")
    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:REPS
        step!()
    end
    CUDA.synchronize()
    step_ms = (time_ns() - t0) / 1e6 / REPS
    st = cache.state
    FM._launch_cuda_nearfield_kernel!(st)
    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:REPS
        FM._launch_cuda_nearfield_kernel!(st)
    end
    CUDA.synchronize()
    nearfield_ms = (time_ns() - t0) / 1e6 / REPS
    step!()
    CUDA.synchronize()
    inner = inner_of(sys)
    u_rms = rel_rms(Float64.(inner.gradient_stretching[1:3, sample]), Uref)
    j_rms = rel_rms(Float64.(inner.potential[5:13, sample]), Jref)
    hom_all = hom_mixed = NaN
    # flat-policy fallback caches (e.g. the degenerate ell=1 sentinel) carry no
    # bin context: the split kernels run unbinned there and homogeneity
    # telemetry is unavailable
    if sd.name != "regularized" && FM._cache_nearfield_bin_ctx(cache) !== nothing
        h = FM.cuda_nearfield_homogeneity(st; stream=:all)
        hom_all = h.homogeneous_fraction
        hm = FM.cuda_nearfield_homogeneity(st; stream=:mixed)
        hom_mixed = hm.instants == 0 ? 1.0 : hm.homogeneous_fraction
    end
    @printf(io, "%s,%s,%d,%d,%d,%s,%s,%s,%d,%.4f,%.6f,%.6f,%.4e,%.4e,%.4f,%.4f,%d\n",
        label, def.case, def.n, ell, q, TF === Float64 ? "F64" : "F32",
        sd.name, String(sd.mode), sd.subsort ? 1 : 0, sd.rho_t, step_ms,
        nearfield_ms, u_rms, j_rms, hom_all, hom_mixed, st.counts.n_direct)
    flush(io)
    @printf("  %-22s %s: step %8.3f ms  nearfield %8.3f ms  u_rms %.3e  j_rms %.3e\n",
        sd.name, TF === Float64 ? "F64" : "F32", step_ms, nearfield_ms, u_rms, j_rms)
    cache = nothing; sys = nothing
    GC.gc()
    CUDA.reclaim()
    return (; step_ms, nearfield_ms, u_rms, j_rms)
end

# baseline-gated geometry ladder: escalate q 16 -> 20, then ell - 1 at q = 16,
# until the F64 baseline passes the 1e-3 velocity gate
function resolve_geometry(label, def, sample, Uref, Jref)
    candidates = [(def.ell, def.q), (def.ell, 20), (def.ell - 1, 16), (def.ell - 1, 20)]
    for (ell, q) in candidates
        ell >= 1 || continue
        println("  geometry probe: ell=$ell q=$q")
        res = try
            mktemp() do _, tio
                run_strategy!(tio, label, def, ell, q, Float64,
                    StratDef("regularized", wrap_reg, :classsplit, true, 4.789),
                    sample, Uref, Jref)
            end
        catch err
            println("    rejected: ", sprint(showerror, err)[1:min(end, 200)])
            continue
        end
        if res.u_rms <= 1e-3
            return ell, q
        end
        println("    u_rms $(res.u_rms) misses the 1e-3 gate; escalating")
    end
    error("no adequate geometry found for $label")
end

host = gethostname()
job = get(ENV, "SLURM_JOB_ID", "manual")
# case-list separator is ":" — sbatch --export treats "," as its own list
# separator and silently drops trailing cases (job 13065443)
labels = let sel = get(ENV, "FM032A_CASES", "")
    isempty(sel) ? (SENTINELS ? SENTINEL : PRIMARY) : Tuple(split(sel, r"[:,]"))
end
for label in labels
    try
        # previous cases' device caches are function-local garbage but not yet
        # collected; without this the dense free-memory preflight sees a
        # starved pool (wake1e6 rejection, job 13065376)
        GC.gc()
        CUDA.reclaim()
        def = CASE_DEFS[label]
        println("=== case $label: $(def.case) n=$(def.n) (start ell=$(def.ell) q=$(def.q))")
        ref_sys = build_case(def.case, def.n)
        sample = unique(round.(Int, range(1, def.n; length=min(NSAMPLE, def.n))))
        print("  sampled-direct reference ($(length(sample)) targets)... ")
        tref = @elapsed Uref, Jref = sampled_reference(ref_sys, sample)
        @printf("%.1f s\n", tref)
        ell, q = resolve_geometry(label, def, sample, Uref, Jref)
        println("  selected geometry: ell=$ell q=$q")
        out = joinpath(OUTDIR, "cuda032a_staged_$(label)_$(host)_$(job).csv")
        open(out, "w") do io
            println(io, "label,case,n,ell,q,tf,strategy,mode,subsort,rho_t,step_ms," *
                "nearfield_ms,u_rel_rms,j_rel_rms,hom_all,hom_mixed,n_direct")
            for TF in (Float32, Float64), sd in strategy_matrix(def.case)
                try
                    run_strategy!(io, label, def, ell, q, TF, sd, sample, Uref, Jref)
                catch err
                    # e.g. TwoPassVortex refused on a flat-fallback (ell=1)
                    # cache — keep the remaining rows
                    println("  CONFIG $(sd.name)/$TF FAILED: ",
                        sprint(showerror, err)[1:min(end, 300)])
                end
            end
        end
        println("  wrote $out")
    catch err
        # a degenerate sentinel geometry (e.g. n=1e3 forcing ell<=1) must not
        # abort the remaining cases; record and continue
        println("  CASE $label FAILED: ", sprint(showerror, err)[1:min(end, 400)])
    end
end
println("cuda_032a_staged_ab complete")
