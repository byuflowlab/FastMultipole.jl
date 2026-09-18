# Why is the FMM nearfield more expensive PER PAIR than the all-pairs arm, when
# both call the same _direct_pair_ug with the same Val(:shipped) series?
#
# Two candidate costs that are NOT the physics:
#   padding -- ka_direct_pairs_functor_kernel! gives a workgroup one CELL PAIR
#              and strides WG lanes over the target cell. A target cell of
#              occupancy n_t costs ceil(n_t/WG)*WG lane-slots, not n_t, and the
#              waste is charged once per SOURCE cell paired with it.
#   atomics -- the pair shape shares targets across pairs, so every accumulation
#              is atomic; the all-pairs shape owns its target and uses stores.
#              Count is 13 (or 4) atomics per target per pair, independent of n_s.
#
# This probe counts both against the real direct list, so the per-pair timing gap
# can be attributed instead of guessed.
include("ka_backend.jl")
include("pipeline_field.jl")
using FastMultipole, Printf
const FM = FastMultipole
const V = FLOWVPM
import KernelAbstractions as KA
dev_functional() || (println("$(DEV_NAME) not functional; skipping"); exit(0))
ext = Base.get_extension(FastMultipole, :FastMultipoleKAExt)

const WG = 64
const P  = 5

function device_copy_of(h)
    d = V.ParticleField(h.maxparticles, Float32; arraytype=devmatrix, np=h.np,
        fmm=V.FMM(; p=P+1, autotune_p=false, autotune_ncrit=false,
                    autotune_reg_error=false, default_rho_over_sigma=1.0))
    d.particles .= devarray(Float32.(Array(h.particles)))
    return d
end

for (step, np) in [(36, 2048), (36, 8192), (36, nothing)]
    dev = device_copy_of(load_wake(step; np=np, TF=Float64, P=P))
    V.radix_fmm_settings!(dev; m2l_strategy=:concat)
    V.UJ_fmm(dev)
    c = V._radix_fmm_couplings[dev].cache; st = c.state
    NP = V.get_np(dev)
    rr = Array(st.cell_ranges)
    dt = Array(st.direct_targets); ds = Array(st.direct_sources)
    npair = st.counts.n_direct
    rows  = size(st.output, 1) >= 13 ? 13 : 4

    ninter = 0; lanework = 0; natomic = 0; occ = Int[]
    for k in 1:npair
        nt = Int(rr[2, dt[k]]); ns = Int(rr[2, ds[k]])
        ninter   += nt * ns
        lanework += cld(nt, WG) * WG * ns
        natomic  += nt * rows
        push!(occ, nt)
    end
    pad = lanework / ninter
    # all-pairs arm for reference: ndrange = NP, inner loop NP
    lw_dir = cld(NP, WG) * WG * NP
    @printf("\nnp=%d  ell=%d  cells=%d  pairs=%d  target-cell occupancy: min %d  med %d  max %d\n",
            NP, c.ell, st.counts.n_cells, npair,
            minimum(occ), sort(occ)[cld(end,2)], maximum(occ))
    @printf("  true interactions   %.4g\n", ninter)
    @printf("  padded lane-slots   %.4g   -> %.2fx padding waste (%.1f%% of lanes idle)\n",
            lanework, pad, 100*(1 - 1/pad))
    @printf("  atomic ops          %.4g   = %.3f per true interaction\n",
            natomic, natomic/ninter)
    @printf("  direct arm padding  %.4gx  (lane-slots %.4g vs %.4g pairs)\n",
            lw_dir/float(NP)^2, lw_dir, float(NP)^2)
    V.clear_radix_fmm_cache!(dev); delete!(V._radix_fmm_couplings, dev)
    GC.gc(); GC.gc()
end
