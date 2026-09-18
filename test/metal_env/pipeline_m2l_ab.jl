# Dense vs concat m2l_strategy, interleaved A/B, many warm trials.
#
# Session 23 left this open: dense was steady at ~75 ms and concat swung
# 41-393 ms over six trials, which cannot separate them. Both fields are built
# once and the two strategies are then alternated call-for-call, so any drift
# or contention hits both arms equally.
#
# Timing is per call with an explicit device synchronize, warmups discarded
# ([[feedback-two-call-gpu-warmup]]); the report is the distribution, not a
# single number ([[feedback-gpu-benchmark-trial-counts]]).

include("ka_backend.jl")
include("pipeline_field.jl")

using FastMultipole, Printf, Statistics
const FM = FastMultipole
const V = FLOWVPM

dev_functional() || (println("$(DEV_NAME) not functional; skipping"); exit(0))
Base.get_extension(FastMultipole, :FastMultipoleKAExt) === nothing &&
    error("the KA extension is not loaded")

const TF = Float32
const STEP = 36
const P = 5
const NP = parse(Int, get(ENV, "AB_NP", "8192"))
const NWARM = parse(Int, get(ENV, "AB_WARM", "10"))
const NTRIAL = parse(Int, get(ENV, "AB_TRIALS", "120"))

function device_field(np, strategy)
    host = load_wake(STEP; np, TF, P)
    d = V.ParticleField(host.maxparticles, TF; arraytype=devmatrix, np=host.np,
        fmm=V.FMM(; p=P + 1, autotune_p=false, autotune_ncrit=false,
                    autotune_reg_error=false, default_rho_over_sigma=1.0))
    d.particles .= devarray(host.particles)
    V.radix_fmm_settings!(d; m2l_strategy=strategy)
    return d
end

timed!(d) = (t = time(); V.UJ_fmm(d); KernelAbstractions.synchronize(DEV_BACKEND);
             (time() - t) * 1e3)

function report(name, v)
    s = sort(v)
    n = length(s)
    q(p) = s[clamp(round(Int, p * n), 1, n)]
    @printf("%-7s n=%d  min %7.2f  p25 %7.2f  med %7.2f  p75 %7.2f  p95 %7.2f  max %8.2f  mean %7.2f\n",
            name, n, s[1], q(0.25), q(0.50), q(0.75), q(0.95), s[end], mean(s))
end

println("=== dense vs concat, np=$NP, $NWARM warmup + $NTRIAL interleaved trials ===")
flush(stdout)

fields = Dict(s => device_field(NP, s) for s in (:dense, :concat))
println("fields built"); flush(stdout)

for s in (:dense, :concat), _ in 1:NWARM
    timed!(fields[s])
end
println("warmup done"); flush(stdout)

t = Dict(s => Float64[] for s in (:dense, :concat))
for i in 1:NTRIAL
    for s in (:dense, :concat)          # same order every trial; interleaved
        push!(t[s], timed!(fields[s]))
    end
    if i % 20 == 0
        @printf("  trial %d: dense %.1f ms  concat %.1f ms\n", i, t[:dense][end], t[:concat][end])
        flush(stdout)
    end
end

println()
report("dense", t[:dense])
report("concat", t[:concat])
@printf("\nmedian ratio concat/dense = %.3f\n", median(t[:concat]) / median(t[:dense]))
flush(stdout)
