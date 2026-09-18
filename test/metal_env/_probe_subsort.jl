# Did stage 19 actually run? Compare perm with the subsort gate on vs off, and
# check the within-cell sub-Morton skey are nondecreasing for covered cells.
include("ka_backend.jl")
include("pipeline_field.jl")
using FastMultipole, Printf
const FM = FastMultipole
const V = FLOWVPM
const ext = Base.get_extension(FastMultipole, :FastMultipoleKAExt)
const TF = Float32

function permof(np, on)
    host = load_wake(36; np, TF, P=5)
    d = V.ParticleField(host.maxparticles, TF; arraytype=devmatrix, np=host.np,
        fmm=V.FMM(; p=6, autotune_p=false, autotune_ncrit=false,
                    autotune_reg_error=false, default_rho_over_sigma=1.0))
    d.particles .= devarray(host.particles)
    V.radix_fmm_settings!(d; m2l_strategy=:concat)
    V.UJ_fmm(d)
    c = V._radix_fmm_coupling!(d).cache
    g = c.device_ctx.grid
    n = c.state.counts.n_bodies
    return Array(view(g.perm, 1:n)), Array(view(g.cell_ranges, :, 1:g.n_cells)),
           Array(view(c.device_ctx.subsort_keys, 1:n)),
           c.options.direct_kernel, c.ell
end

np = 2048
p_on, cr, skey, dk, ell = permof(np, true)
println("direct_kernel = ", typeof(dk), "   ell = ", ell,
        "   sub = ", min(3, FM.RADIX_GRID_MAX_ELL - ell))
bad = Ref(0); covered = Ref(0)
for c in 1:size(cr, 2)
    f = cr[1, c]; n = cr[2, c]
    (1 < n <= 1024) || continue
    covered[] += 1
    for i in f:(f + n - 2)
        skey[i] <= skey[i + 1] || (bad[] += 1)
    end
end
@printf("cells covered by the sort: %d; out-of-order key pairs: %d\n", covered[], bad[])
