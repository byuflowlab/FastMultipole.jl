# Re-measure the premises of the three candidate speedup levers on the CURRENT
# depth rule (post cost-based ell selection, post 9a7ec2e's reciprocal-row
# revert). Every premise below was fitted before those two commits.
#
#   lever 1  route count at fixed depth  -> hctx.total_routes, windows/step
#   lever 2  M2L small-GEMM floor        -> ka_launch_m2l! time, us/route
#   lever 3  nearfield per-pair divide   -> ka_launch_nearfield! time, share
#
# Smallest real case only (step 36, full np). Scale up only on command.

include("ka_backend.jl")
include("pipeline_field.jl")

using FastMultipole, Printf, Statistics, KernelAbstractions
const KA = KernelAbstractions
const FM = FastMultipole
const V = FLOWVPM
const ext = Base.get_extension(FastMultipole, :FastMultipoleKAExt)

dev_functional() || (println("$(DEV_NAME) not functional; skipping"); exit(0))
ext === nothing && error("the KA extension is not loaded")

const TF = Float32
const STEP = parse(Int, get(ENV, "UJ_STEP", "36"))
const NP = (s = get(ENV, "PROF_NP", ""); isempty(s) ? nothing : parse(Int, s))
const P = 5
const NTRIAL = parse(Int, get(ENV, "PROF_TRIALS", "30"))

med_ms(f, n, backend) = begin
    ts = Float64[]
    for _ in 1:n
        t = time(); f(); KA.synchronize(backend); push!(ts, (time() - t) * 1e3)
    end
    median(ts)
end

host = load_wake(STEP; np=NP, TF=Float64, P=P)
d = V.ParticleField(host.maxparticles, TF; arraytype=devmatrix, np=host.np,
    fmm=V.FMM(; p=P + 1, autotune_p=false, autotune_ncrit=false,
                autotune_reg_error=false, default_rho_over_sigma=1.0))
d.particles .= devarray(TF.(Array(host.particles)))
V.radix_fmm_settings!(d; m2l_strategy=:concat)
for _ in 1:6; V._reset_particles(d); V.UJ_fmm(d); end   # Metal ramp: 5+ calls

cache = V._radix_fmm_coupling!(d).cache
state = cache.state
hctx  = state.interaction_list
ws    = state.scratch
backend = KA.get_backend(state.output)
NPP = V.get_np(d)

# ---- structure ----
ranges = Array(state.cell_ranges)
dt = Array(state.direct_targets); ds = Array(state.direct_sources)
npairs = state.counts.n_direct
ncell  = state.counts.n_cells
interactions = sum(Int(ranges[2, dt[k]]) * Int(ranges[2, ds[k]]) for k in 1:npairs)
nwin = sum(length(1:hctx.window_classes:hctx.noffsets)
           for L in hctx.first_m2l_level:hctx.ell)

# ---- timing ----
t_step = med_ms(() -> (V._reset_particles(d); V.UJ_fmm(d)), NTRIAL, backend)
t_m2l  = med_ms(() -> ext.ka_launch_m2l!(state, ws), NTRIAL, backend)
t_near = med_ms(() -> ext.ka_launch_nearfield!(state), NTRIAL, backend)

println("=== lever premise re-measurement (current depth rule) ===")
@printf("np=%d  ell=%d  levels %d:%d  n_cells=%d\n",
        NPP, hctx.ell, hctx.first_m2l_level, hctx.ell, ncell)
@printf("L1  routes=%d  noffsets=%d  K=%d  windows/step=%d  routes/cell=%.1f\n",
        hctx.total_routes, hctx.noffsets, hctx.window_classes, nwin,
        hctx.total_routes / ncell)
@printf("L3  direct pairs=%d  pairs/cell=%.1f  interactions=%.3e\n",
        npairs, npairs / ncell, Float64(interactions))
@printf("step %8.3f ms\n", t_step)
@printf("L2  M2L       %8.3f ms (%5.1f%% of step)  %.3f us/route\n",
        t_m2l, 100t_m2l / t_step, 1e3 * t_m2l / max(hctx.total_routes, 1))
@printf("L3  nearfield %8.3f ms (%5.1f%% of step)  %.3f ns/interaction\n",
        t_near, 100t_near / t_step, 1e6 * t_near / max(interactions, 1))
flush(stdout)
