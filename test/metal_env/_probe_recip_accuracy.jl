# Delivered accuracy of the reciprocal-sigma nearfield against the divide.
# Same state, same everything: run the real functor kernel with inv_sigma_row
# wired (production) and with it forced to 0 (the divide), compare outputs.
include("ka_backend.jl"); include("pipeline_field.jl")
using FastMultipole, Printf
const FM = FastMultipole; const V = FLOWVPM
import KernelAbstractions as KA
ext = Base.get_extension(FastMultipole, :FastMultipoleKAExt)
STEP = parse(Int, get(ENV,"UJ_STEP","720")); P = 5
host = load_wake(STEP; TF=Float64, P=P)
dev = V.ParticleField(host.maxparticles, Float32; arraytype=devmatrix, np=host.np,
    fmm=V.FMM(; p=P+1, autotune_p=false, autotune_ncrit=false,
                autotune_reg_error=false, default_rho_over_sigma=1.0))
dev.particles .= devarray(Float32.(Array(host.particles)))
V.radix_fmm_settings!(dev; m2l_strategy=:concat)
V.UJ_fmm(dev)
st = V._radix_fmm_couplings[dev]; state = st.cache.state
backend = KA.get_backend(state.output); TF = eltype(state.output)
npairs = state.counts.n_direct; HS = size(state.output,1) >= 13
dk = state.options.direct_kernel
isr = ext._ka_inv_sigma_row(dk, state.source_bodies)
@printf("np=%d  sigma_row=%d  inv_sigma_row=%d  rows=%d\n",
        V.get_np(dev), ext._ka_kernel_sigma_row(dk), isr, size(state.source_bodies,1))
function run(row, wg=64)
    k = ext._ka_device_direct_kernel(dk, TF, row)
    kern = ext._cached_kernel(ext.ka_direct_pairs_functor_kernel!, backend, wg)
    fill!(state.output, zero(TF))
    kern(k, state.output, state.source_bodies, state.cell_ranges,
         state.direct_targets, state.direct_sources, npairs, TF, Val(HS), Val(wg);
         ndrange=npairs*wg)
    KA.synchronize(backend); Array(state.output)
end
recip = run(isr); divd = run(0)
den = sqrt(sum(abs2, divd))
@printf("||recip - div|| / ||div||  = %.3e   (Float32 eps = %.2e)\n",
        sqrt(sum(abs2, recip .- divd))/den, eps(Float32))
@printf("max |recip - div| / max|div| = %.3e\n",
        maximum(abs, recip .- divd)/maximum(abs, divd))

# ---- delivered nearfield time: production (reciprocal) vs the divide ----
const CALLS = 12
function time_row(row, wg)
    k = ext._ka_device_direct_kernel(dk, TF, row)
    kern = ext._cached_kernel(ext.ka_direct_pairs_functor_kernel!, backend, wg)
    f = () -> begin
        fill!(state.output, zero(TF))
        kern(k, state.output, state.source_bodies, state.cell_ranges,
             state.direct_targets, state.direct_sources, npairs, TF, Val(HS), Val(wg);
             ndrange=npairs*wg)
        KA.synchronize(backend)
    end
    f(); ts = Float64[]
    for _ in 1:CALLS; t0=time_ns(); f(); push!(ts,(time_ns()-t0)/1e9); end
    sort(ts[cld(CALLS,2)+1:end])[1]
end
println()
@printf("  %-5s %11s %11s %9s\n", "WG", "divide s", "recip s", "gain")
for wg in (64, 128)
    td = time_row(0, wg); tr = time_row(isr, wg)
    @printf("  %-5d %11.5f %11.5f %8.1f%%\n", wg, td, tr, 100*(td-tr)/td)
    flush(stdout)
end
