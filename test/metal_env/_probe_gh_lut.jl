# Does the task-037f `:lut` g/h mode buy the nearfield time the ablation
# promised, and what does it cost in accuracy?
#
# The ablation (`_probe_nearfield_cost.jl`) bounded the prize by REPLACING the
# 13-term series with the singular (1,-3): 35.7% of the nearfield at np=248714.
# `:lut` is the real mechanism -- a 1024-entry Float32 table of the NORMALIZED
# G = g/rho^3, H = h/rho^5, linearly interpolated in x = rho^2 -- so it should
# recover most of that while staying inside its error budget.
#
# Accuracy is scored as the delivered ||F_lut - F_shipped|| / ||F_shipped||,
# which is exactly the quantity task 037f budgets: 2.656e-4 allowed (the
# binding cube-1e6 row), 4.2e-5 predicted coherent, 4.2e-6 measured on the
# n=1e4 smoke case. The host reference maps :lut -> :shipped by design, so a
# bitwise host gate is not available and is not the right gate.

include("ka_backend.jl")
include("pipeline_field.jl")

using FastMultipole, Printf
const FM = FastMultipole
const V = FLOWVPM
import KernelAbstractions as KA

dev_functional() || (println("$(DEV_NAME) not functional; skipping"); exit(0))
ext = Base.get_extension(FastMultipole, :FastMultipoleKAExt)
ext === nothing && error("the KA extension is not loaded")

const STEP  = parse(Int, get(ENV, "UJ_STEP", "720"))
const P     = 5
const CALLS = 7

host = load_wake(STEP; TF=Float64, P=P)
NP = V.get_np(host)

function make_dev()
    d = V.ParticleField(host.maxparticles, Float32; arraytype=devmatrix, np=host.np,
        fmm=V.FMM(; p=P+1, autotune_p=false, autotune_ncrit=false,
                    autotune_reg_error=false, default_rho_over_sigma=1.0))
    d.particles .= devarray(Float32.(Array(host.particles)))
    V.radix_fmm_settings!(d; m2l_strategy=:concat)
    d
end

# time the nearfield stage alone, and return the full-step U/J field
function run_mode(mode::Symbol; lutmem::Symbol=:global)
    FM.set_radix_setting!(:CUDA_NEARFIELD_GH_MODE, mode)
    ext._KA_GH_LUT_MEM[] = lutmem
    empty!(ext._KA_GH_LUT_CACHE)
    dev = make_dev()
    V.UJ_fmm(dev)                       # warm: compile + build the cache
    st = V._radix_fmm_couplings[dev]
    state = st.cache.state
    backend = KA.get_backend(state.output)

    ts = Float64[]
    for _ in 1:CALLS
        t0 = time_ns()
        ext.ka_launch_nearfield!(state)
        KA.synchronize(backend)
        push!(ts, (time_ns() - t0) / 1e9)
    end
    tnf = minimum(ts[cld(CALLS,2):end])

    # a clean full step for the field comparison
    dev2 = make_dev()
    V.UJ_fmm(dev2)
    U = Array(dev2.particles)[10:12, 1:NP]
    lut_on = ext._ka_gh_lut(backend, ext._ka_device_direct_kernel(
        state.options.direct_kernel, eltype(state.output))) !== nothing
    return (; tnf, U, lut_on, npairs=state.counts.n_direct)
end

println("=== :lut g/h mode -- step $(STEP), np=$(NP) ===")
base = run_mode(:fp32)      # the shipped path on a Float32 config (documented no-op)
lutg = run_mode(:lut; lutmem=:global)
lutl = run_mode(:lut; lutmem=:local)
lut  = lutg.tnf <= lutl.tnf ? lutg : lutl
FM.set_radix_setting!(:CUDA_NEARFIELD_GH_MODE, :fp32)

@printf("n_direct pairs = %d\n", base.npairs)
@printf("lut table actually engaged: %s\n\n", lut.lut_on)

num = sqrt(sum(abs2, lut.U .- base.U))
den = sqrt(sum(abs2, base.U))
@printf("nearfield  :shipped  %.5f s\n", base.tnf)
@printf("nearfield  :lut/glob %.5f s   (%.1f%% faster)\n",
        lutg.tnf, 100*(base.tnf - lutg.tnf)/base.tnf)
@printf("nearfield  :lut/local %.5f s   (%.1f%% faster)\n",
        lutl.tnf, 100*(base.tnf - lutl.tnf)/base.tnf)
@printf("\ndelivered ||U_lut - U_shipped|| / ||U_shipped|| = %.3e\n", num/den)
@printf("  task 037f budget (binding row)                = 2.656e-4  %s\n",
        num/den <= 2.656e-4 ? "PASS" : "FAIL")
@printf("  predicted coherent                            = 4.2e-5    %s\n",
        num/den <= 4.2e-5 ? "under" : "OVER")
