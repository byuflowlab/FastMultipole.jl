# How much of the KA M2L stage is per-step WINDOW GENERATION?
#
# `ka_hierarchical_m2l!` regenerates every (level, offset-window) every step:
# flag -> scan -> compact, plus a per-window prefix D2H. CUDA skips all of that
# in steady state by consuming an occupancy-epoch window cache
# (`_launch_cuda_hierarchical_m2l_cached!`). Porting that is only worth it if
# generation is a real share of the stage, so measure it first.
#
# Three timings per strategy, each over the SAME warm cache:
#   full   -- ka_launch_m2l! (generate + apply, what runs today)
#   gen    -- the same loop with the applies removed
#   step   -- the whole UJ_fmm, for context
# `gen/full` is the ceiling on what a cached-window apply could remove.

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
const STEP = 36
const P = 5
const NP = parse(Int, get(ENV, "PROF_NP", "8192"))
const NTRIAL = parse(Int, get(ENV, "PROF_TRIALS", "30"))

function device_field(np, strategy)
    host = load_wake(STEP; np, TF, P)
    d = V.ParticleField(host.maxparticles, TF; arraytype=devmatrix, np=host.np,
        fmm=V.FMM(; p=P + 1, autotune_p=false, autotune_ncrit=false,
                    autotune_reg_error=false, default_rho_over_sigma=1.0))
    d.particles .= devarray(host.particles)
    V.radix_fmm_settings!(d; m2l_strategy=strategy)
    return d
end

# `ka_hierarchical_m2l!` with every apply removed: the generation half only.
function generate_only!(state, hctx)
    plan = hctx.apply_plan
    dense = plan isa FM.ResidentM2LDenseCUDAPlan
    route_class = plan.route_class
    noffsets = hctx.noffsets
    K = hctx.window_classes
    total = 0
    for L in hctx.first_m2l_level:hctx.ell
        class_base = dense ? 0 : (L - hctx.first_m2l_level) * noffsets
        for first_offset in 1:K:noffsets
            last_offset = min(first_offset + K - 1, noffsets)
            total += ext.ka_hier_generate_window!(state, hctx, route_class, L,
                first_offset, last_offset, class_base)
        end
    end
    return total
end

med_ms(f, n, backend) = begin
    ts = Float64[]
    for _ in 1:n
        t = time(); f(); KA.synchronize(backend); push!(ts, (time() - t) * 1e3)
    end
    median(ts)
end

println("=== M2L generation share, np=$NP, $NTRIAL trials ==="); flush(stdout)

for strategy in (:dense, :concat)
    d = device_field(NP, strategy)
    V.UJ_fmm(d); V.UJ_fmm(d)            # warm: compile + steady state
    cache = V._radix_fmm_coupling!(d).cache
    state = cache.state
    hctx = state.interaction_list
    ws = state.scratch
    backend = KA.get_backend(state.output)

    nwin = 0
    for L in hctx.first_m2l_level:hctx.ell
        nwin += length(1:hctx.window_classes:hctx.noffsets)
    end

    t_step = med_ms(() -> V.UJ_fmm(d), NTRIAL, backend)
    t_full = med_ms(() -> ext.ka_launch_m2l!(state, ws), NTRIAL, backend)
    t_gen  = med_ms(() -> generate_only!(state, hctx), NTRIAL, backend)

    @printf("%-7s  levels %d:%d  K=%d  noffsets=%d  windows/step=%d  routes=%d\n",
            strategy, hctx.first_m2l_level, hctx.ell, hctx.window_classes,
            hctx.noffsets, nwin, hctx.total_routes)
    @printf("         step %7.2f ms   M2L full %7.2f ms (%4.1f%% of step)   gen %7.2f ms (%4.1f%% of M2L)\n\n",
            t_step, t_full, 100 * t_full / t_step, t_gen, 100 * t_gen / t_full)
    flush(stdout)
end
