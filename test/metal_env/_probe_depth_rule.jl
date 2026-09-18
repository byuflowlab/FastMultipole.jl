# Validate cost-based depth selection: what does the NEW rule pick, what did
# the OLD rule pick, and what does each cost end to end on the real wake?
include("ka_backend.jl")
include("pipeline_field.jl")
using FastMultipole, Printf
const FM = FastMultipole
const V = FLOWVPM
import KernelAbstractions as KA
dev_functional() || (println("skipping"); exit(0))
ext = Base.get_extension(FastMultipole, :FastMultipoleKAExt)
const P = 5
timeit(f; calls=10) = (ts=Float64[]; for _ in 1:calls; GC.gc(); t0=time_ns(); f(); push!(ts,(time_ns()-t0)/1e9); end; minimum(ts))

function device_copy_of(h)
    d = V.ParticleField(h.maxparticles, Float32; arraytype=devmatrix, np=h.np,
        fmm=V.FMM(; p=P+1, autotune_p=false, autotune_ncrit=false,
                    autotune_reg_error=false, default_rho_over_sigma=1.0))
    d.particles .= devarray(Float32.(Array(h.particles)))
    return d
end

# what the OLD rule would have returned: deepest admissible
function legacy_ell(host, settings_q, sigma_max, L, np)
    qs = sort!([Int(q) for q in FM._SUPPORTED_RIGID_NEAR_RADII2 if q >= settings_q])
    reach = 1.03 * 4.789 * sigma_max
    for ell in max(2, floor(Int, log2(max(np,8))/3)):-1:2, q in qs
        FM._ball_stencil_min_gap(q) * (L/2^ell) >= reach && return (ell, q)
    end
    return (2, settings_q)
end

const CASES = [(36,2048),(36,8192),(36,nothing),(72,nothing),(144,nothing),(288,nothing)]
@printf("%8s | %4s %9s | %4s %9s | %8s | %9s\n",
        "np","old","old s","new","new s","speedup","relerr")
for (step, np) in CASES
    host = load_wake(step; np=np, TF=Float64, P=P)
    NP = V.get_np(host); V.UJ_fmm(host; autotune=false)
    U_ref = copy(Array(host.particles)[V.U_INDEX, 1:NP])
    bounds = V._radix_derive_bounds(host, 0.1; rectangular=false)
    L = bounds[2] isa Real ? Float64(bounds[2]) : Float64(maximum(bounds[2]))
    oell, oq = legacy_ell(host, 6, Float64(V._radix_sigma_max(host)), L, NP)
    res = Tuple{Int,Float64,Float64}[]
    for (tag, ell, q) in (("old", oell, oq), ("new", nothing, nothing))
        dev = device_copy_of(load_wake(step; np=np, TF=Float64, P=P))
        V.radix_fmm_settings!(dev; m2l_strategy=:concat,
            (ell === nothing ? (;) : (; ell=ell, near_radius2=q))...)
        V.UJ_fmm(dev)
        t = timeit(() -> (V._reset_particles(dev); V.UJ_fmm(dev)))
        U = Array(dev.particles)[V.U_INDEX, 1:NP]
        rel = maximum(abs.(U .- U_ref))/maximum(abs.(U_ref))
        push!(res, (V._radix_fmm_couplings[dev].cache.ell, t, rel))
        V.clear_radix_fmm_cache!(dev); delete!(V._radix_fmm_couplings, dev)
        GC.gc(); GC.gc()
    end
    @printf("%8d | %4d %9.5f | %4d %9.5f | %7.2fx | %9.2e\n",
            NP, res[1][1], res[1][2], res[2][1], res[2][2],
            res[1][2]/res[2][2], res[2][3]); flush(stdout)
end
