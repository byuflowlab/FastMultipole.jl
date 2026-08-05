# Task 033: baseline CPU benchmark of FLOWVPM (e2bd487) for one (case, n).
# Times the UJ_fmm evaluation and a full RK3 time step at fixed FMM settings
# (p=4, ncrit=50, theta=0.4, autotune off), logs sampled-direct accuracy for
# every row, and optionally captures a Profile breakdown.
#
# Env: FM033_CASE (cube|wake), FM033_N (target n from the 024b grid),
#      FM033_OUT (csv path), FM033_REFERENCE_DIR, FM033_PROFILE (0|1).

include(joinpath(@__DIR__, "benchmark_033_common.jl"))
using Dates
using Profile

const CASE = ENV["FM033_CASE"]
const N_TARGET = parse(Int, ENV["FM033_N"])
const OUT = ENV["FM033_OUT"]
const REFDIR = ENV["FM033_REFERENCE_DIR"]
const DO_PROFILE = get(ENV, "FM033_PROFILE", "0") == "1"

const BENCH_HEADER = (
    "case", "mode", "n_target", "n_actual", "sigma", "p", "ncrit", "theta",
    "uj_reps", "t_build", "t_uj_median", "t_uj_min", "t_reset_median",
    "step_reps", "t_step_median", "t_step_min",
    "u_rel_rms", "u_abs_rms", "u_max_err", "j_rel_rms",
    "threads", "host", "jobid", "julia_version", "timestamp",
)

threads = Threads.nthreads()
mode = "cpu$(threads)"
host = gethostname()
jobid = get(ENV, "SLURM_JOB_ID", "local")
println("=== 033 benchmark case=$CASE n=$N_TARGET mode=$mode host=$host")

t_build = @elapsed pfield = fm033_build(CASE, N_TARGET)
n_actual = vpm.get_np(pfield)
sigma = fm033_sigma(CASE, N_TARGET)
settings = pfield.fmm
println("built n_actual=$n_actual sigma=$sigma in $(round(t_build, digits=2))s")

reference = fm033_read_reference(
    fm033_reference_path(REFDIR, CASE, N_TARGET), CASE, N_TARGET, n_actual)

# --- UJ_fmm evaluation timing (includes its internal particle reset) --------
uj_reps = N_TARGET >= 316228 ? 3 : 5
vpm.UJ_fmm(pfield; autotune=false)  # warmup / JIT
t_uj = [@elapsed vpm.UJ_fmm(pfield; autotune=false) for _ in 1:uj_reps]
println("t_uj = ", join(round.(t_uj; sigdigits=4), ", "))

# Accuracy of the final evaluation (UJ_fmm resets before accumulating, so the
# field now holds the pure FMM result).
acc = fm033_accuracy_metrics(pfield, reference)
println("accuracy: u_rel_rms=$(acc.u_rel_rms) j_rel_rms=$(acc.j_rel_rms)")

t_reset = [@elapsed vpm._reset_particles(pfield) for _ in 1:5]

# --- full-step timing (RK3 low-storage; relaxation applied as in run_vpm!) --
# dt is tiny so the configuration stays effectively frozen across reps.
dt = 1e-6
step_reps = N_TARGET >= 316228 ? 2 : 3
vpm.nextstep(pfield, dt; relax=true)  # warmup
t_step = [@elapsed vpm.nextstep(pfield, dt; relax=true) for _ in 1:step_reps]
println("t_step = ", join(round.(t_step; sigdigits=4), ", "))

fm033_append_row(OUT, BENCH_HEADER, (
    CASE, mode, N_TARGET, n_actual, sigma,
    settings.p, settings.ncrit, settings.theta,
    uj_reps, t_build, median(t_uj), minimum(t_uj), median(t_reset),
    step_reps, median(t_step), minimum(t_step),
    acc.u_rel_rms, acc.u_abs_rms, acc.u_max_err, acc.j_rel_rms,
    threads, host, jobid, string(VERSION),
    Dates.format(Dates.now(), dateformat"yyyy-mm-ddTHH:MM:SS"),
))
println("appended row to $OUT")

# --- optional profile capture (bottleneck attribution) ----------------------
if DO_PROFILE
    Profile.clear()
    Profile.@profile vpm.UJ_fmm(pfield; autotune=false)
    profpath = joinpath(dirname(OUT), "profile_$(CASE)_n$(N_TARGET)_$(mode).txt")
    open(profpath, "w") do io
        Profile.print(IOContext(io, :displaysize => (200, 250));
            format=:tree, C=false, mincount=10, maxdepth=40)
    end
    println("wrote $profpath")
end
