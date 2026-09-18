include("ka_backend.jl"); include("pipeline_field.jl")
using FastMultipole, Printf
const FM = FastMultipole; const V = FLOWVPM
dev_functional() || exit(0)
const P = 5; const WG = 64
function device_copy_of(h)
    d = V.ParticleField(h.maxparticles, Float32; arraytype=devmatrix, np=h.np,
        fmm=V.FMM(; p=P+1, autotune_p=false, autotune_ncrit=false,
                    autotune_reg_error=false, default_rho_over_sigma=1.0))
    d.particles .= devarray(Float32.(Array(h.particles))); d
end
dev = device_copy_of(load_wake(36; np=nothing, TF=Float64, P=P))
V.radix_fmm_settings!(dev; m2l_strategy=:concat); V.UJ_fmm(dev)
st = V._radix_fmm_couplings[dev].cache.state
rr = Array(st.cell_ranges); dt = Array(st.direct_targets); ds = Array(st.direct_sources)
np_ = st.counts.n_direct
# bucket pairs by target-cell occupancy; report true work, padded work, waste
buckets = [(1,15),(16,31),(32,63),(64,127),(128,191),(192,255),(256,10^9)]
tot_true = Ref(0); tot_pad = Ref(0)
rows = [Any[] for _ in buckets]
acc = [[0,0,0] for _ in buckets]   # npairs, true, padded
for k in 1:np_
    nt = Int(rr[2, dt[k]]); ns = Int(rr[2, ds[k]])
    tr = nt*ns; pd = cld(nt,WG)*WG*ns
    tot_true[] += tr; tot_pad[] += pd
    for (b,(lo,hi)) in enumerate(buckets)
        if lo <= nt <= hi; acc[b][1]+=1; acc[b][2]+=tr; acc[b][3]+=pd; break; end
    end
end
@printf("np=%d  cells=%d  pairs=%d   true=%.4g  padded=%.4g  (%.2fx)\n\n",
        V.get_np(dev), st.counts.n_cells, np_, tot_true[], tot_pad[], tot_pad[]/tot_true[])
@printf("%14s %8s %11s %11s %8s %14s\n","target occ","pairs","true work","padded","factor","% of ALL waste")
for (b,(lo,hi)) in enumerate(buckets)
    n,tr,pd = acc[b]; n==0 && continue
    @printf("%7d-%-6s %8d %11.4g %11.4g %8.2fx %13.1f%%\n",
            lo, hi>10^8 ? "up" : string(hi), n, tr, pd, tr==0 ? 0 : pd/tr,
            100*(pd-tr)/(tot_pad[]-tot_true[]))
end
