# Where does a UJ step actually go? Session 39 left 54% of the step
# unattributed by M2L + nearfield. Time every top-level segment of
# `ka_radix_cache_device_step!` (ext:5886) plus each stage of
# `ka_lifecycle_body!`, on the same warm cache, and check the parts against the
# whole ([[feedback-ablations-must-isolate-one-variable]]).
#
#   update  ka_update_radix_state!   repack + grid rebuild + sort + lists
#   near / b2m / m2m / m2l / l2l / l2b   the lifecycle body, in its own order
#   final   ka_finalize_radix_output!  D2H + scatter into FLOWVPM's buffers
# residue = UJ_fmm - (update + body + final)  is host-side FLOWVPM overhead.

include("ka_backend.jl")
include("pipeline_field.jl")

using FastMultipole, Printf, Statistics, KernelAbstractions
const KA = KernelAbstractions
const FM = FastMultipole
const V = FLOWVPM
const ext = Base.get_extension(FastMultipole, :FastMultipoleKAExt)

dev_functional() || (println("$(DEV_NAME) not functional; skipping"); exit(0))
ext === nothing && error("the KA extension is not loaded")

const TF = getfield(Base, Symbol(get(ENV, "DEV_TF", "Float32")))
const STEP = parse(Int, get(ENV, "UJ_STEP", "36"))
const NP = (s = get(ENV, "PROF_NP", ""); isempty(s) ? nothing : parse(Int, s))
const P = 5
const NTRIAL = parse(Int, get(ENV, "PROF_TRIALS", "30"))
# PROF_MOVE=1: jitter every position by ~1% of the leaf width before each
# timed call so the occupancy changes and the rebuild path (occupancy check,
# lists, M2L window generation) runs, as it does on every step of a real
# simulation. The warm probe never exercises it.
const MOVE = get(ENV, "PROF_MOVE", "0") == "1"
# PROF_SUBSORT=0: disable the within-cell nearfield sub-sort (locality only)
get(ENV, "PROF_SUBSORT", "1") == "0" && (ext._KA_SETTING_OVERRIDES[:CUDA_NEARFIELD_SUBSORT] = false)

med_ms(f, n, backend; pre=nothing) = begin
    ts = Float64[]
    for _ in 1:n
        pre === nothing || (pre(); KA.synchronize(backend))
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
for _ in 1:6; V._reset_particles(d); V.UJ_fmm(d); end

coupling = V._radix_fmm_coupling!(d)
cache   = coupling.cache
state   = cache.state
ws      = state.scratch
backend = KA.get_backend(state.output)
# targets/switches as `fmm!` builds them (src/fmm.jl:893)
targets  = FM.to_tuple(d)
switches = FM.DerivativesSwitch(
    FM.to_vector(false, length(targets)),
    FM.to_vector(true,  length(targets)),
    FM.to_vector(true,  length(targets)), targets)

jitter = if MOVE
    h = 2 * Float64(state.grid.h0) / (1 << state.interaction_list.ell)
    noise = devarray(TF.(0.01 * h .* randn(3, d.np)))
    () -> (view(d.particles, V.X_INDEX, 1:d.np) .+= noise; noise .*= -one(TF))  # alternate sign: bounded drift
else
    nothing
end
t_step   = med_ms(() -> (V._reset_particles(d); V.UJ_fmm(d)), NTRIAL, backend; pre=jitter)
t_update = med_ms(() -> ext.ka_update_radix_state!(cache, targets), NTRIAL, backend; pre=jitter)
# sub-stage attribution of the update (timers inside ka_update_radix_state!)
ext._KA_UPDATE_TIMERS[] = Dict{Symbol,Vector{Float64}}()
med_ms(() -> ext.ka_update_radix_state!(cache, targets), NTRIAL, backend; pre=jitter)
update_stages = ext._KA_UPDATE_TIMERS[]; ext._KA_UPDATE_TIMERS[] = nothing
t_body   = med_ms(() -> ext.ka_lifecycle_body!(state), NTRIAL, backend)
t_near   = med_ms(() -> ext.ka_launch_nearfield!(state; clear=true), NTRIAL, backend)   # production config (_nf_config)
t_b2m    = med_ms(() -> ext.ka_launch_b2m!(state; workgroup=128), NTRIAL, backend)
t_m2m    = med_ms(NTRIAL, backend) do
    FM._zero_resident_nonleaf_multipoles!(state)
    for g in ws.m2m_groups
        ext.ka_resident_stage_group_apply!(state.multipoles, state.multipoles, g, ws, :m2m)
    end
end
t_m2l    = med_ms(() -> ext.ka_launch_m2l!(state, ws), NTRIAL, backend)
t_l2l    = med_ms(NTRIAL, backend) do
    for g in ws.l2l_groups
        ext.ka_resident_stage_group_apply!(state.locals, state.locals, g, ws, :l2l)
    end
end
t_l2b    = med_ms(() -> ext.ka_launch_l2b!(state; workgroup=64), NTRIAL, backend)
t_final  = med_ms(NTRIAL, backend) do
    ext.ka_finalize_radix_output!(state, targets; derivatives_switches=switches,
        host_output_staging=cache.device_ctx.host_output,
        target_buffers=FM._radix_cache_target_buffers!(cache, switches),
        device_target_buffers=cache.device_ctx.device_target_buffers)
end

println("=== UJ step attribution: $(DEV_NAME) $(TF) step=$(STEP) move=$(MOVE) ===")
@printf("np=%d  ell=%d  n_cells=%d  trials=%d\n",
        V.get_np(d), state.interaction_list.ell, state.counts.n_cells, NTRIAL)
@printf("%-10s %9s %8s\n", "segment", "ms", "% step")
row(n, t) = @printf("%-10s %9.3f %7.1f%%\n", n, t, 100t / t_step)
row("UJ_fmm", t_step)
row("update", t_update)
row("body", t_body)
for (n, t) in (("  near", t_near), ("  b2m", t_b2m), ("  m2m", t_m2m),
               ("  m2l", t_m2l), ("  l2l", t_l2l), ("  l2b", t_l2b))
    row(n, t)
end
row("final", t_final)
println("update sub-stages (ms, median; synced per stage):")
for (k, v) in sort(collect(update_stages); by=kv -> -median(kv[2]))
    @printf("  %-26s %8.3f\n", k, median(v))
end
stages = t_near + t_b2m + t_m2m + t_m2l + t_l2l + t_l2b
row("body-parts", stages)
row("residue", t_step - (t_update + t_body + t_final))
flush(stdout)
