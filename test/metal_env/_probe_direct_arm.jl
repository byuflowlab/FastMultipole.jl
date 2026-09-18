# All-pairs direct arm (:RADIX_DIRECT_ARM) vs the FMM lifecycle: accuracy
# against the CPU reference and end-to-end time, across np.
include("ka_backend.jl")
include("pipeline_field.jl")
using FastMultipole, Printf
const FM = FastMultipole
const V = FLOWVPM
import KernelAbstractions as KA
dev_functional() || (println("skipping"); exit(0))
const P = 5
timeit(f; calls=10) = (ts=Float64[]; for _ in 1:calls; t0=time_ns(); f(); push!(ts,(time_ns()-t0)/1e9); end; minimum(ts))

function device_copy_of(h)
    d = V.ParticleField(h.maxparticles, Float32; arraytype=devmatrix, np=h.np,
        fmm=V.FMM(; p=P+1, autotune_p=false, autotune_ncrit=false,
                    autotune_reg_error=false, default_rho_over_sigma=1.0))
    d.particles .= devarray(Float32.(Array(h.particles)))
    return d
end

const CASES = [(36,512),(36,2048),(36,8192),(36,nothing),(72,nothing),(144,nothing)]
@printf("%8s | %9s %9s | %9s %9s | %8s\n", "np","fmm s","fmm rel","dir s","dir rel","speedup")
for (step, np) in CASES
    host = load_wake(step; np=np, TF=Float64, P=P)
    NP = V.get_np(host); V.UJ_fmm(host; autotune=false)
    U_ref = copy(Array(host.particles)[V.U_INDEX, 1:NP])
    res = Tuple{Float64,Float64}[]
    for arm in (false, true)
        FM.set_radix_setting!(:RADIX_DIRECT_ARM, arm)
        dev = device_copy_of(load_wake(step; np=np, TF=Float64, P=P))
        V.radix_fmm_settings!(dev; m2l_strategy=:concat)
        V.UJ_fmm(dev); V.UJ_fmm(dev)
        t = timeit(() -> V.UJ_fmm(dev))
        U = Array(dev.particles)[V.U_INDEX, 1:NP]
        rel = maximum(abs.(U .- U_ref))/maximum(abs.(U_ref))
        push!(res, (t, rel))
        V.clear_radix_fmm_cache!(dev); delete!(V._radix_fmm_couplings, dev)
        GC.gc(); GC.gc()
    end
    FM.set_radix_setting!(:RADIX_DIRECT_ARM, false)
    @printf("%8d | %9.5f %9.2e | %9.5f %9.2e | %7.2fx\n",
            NP, res[1][1], res[1][2], res[2][1], res[2][2], res[1][1]/res[2][1])
    flush(stdout)
end
