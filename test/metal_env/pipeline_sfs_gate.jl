# Gate for the KA SFS pass (task 048).
#
# Two checks, in increasing scope:
#
# (A) ISOLATED. After a device step with `sfs=true`, download the state the
#     pass consumed -- `output` (the finished J), `source_bodies`, the cell
#     ranges and the direct pair list -- and re-run the HOST mirror
#     (`_host_sfs_tg_and_zero!` + `_host_sfs_zeta_pairs!` + `_host_sfs_form_e!`)
#     on those exact arrays. Same J, same pair list, so any disagreement is the
#     port's traversal / striding / atomics, not the FMM feeding it. This is the
#     tight check: tolerance is float reassociation only.
#
# (B) END TO END. FLOWVPM's own `UJ_fmm_gpu!(sfs=true)` on a Metal-backed field
#     vs the host FMM's SFS on the same wake, compared on `SFS_INDEX`. Here the
#     two arms run DIFFERENT FMMs, so the tolerance is the U/J relerr the port
#     already carries (~5e-5), not the isolated one -- this check is about
#     delivery (E formation, the sorted->global scatter, `sfs_to_target!`),
#     which (A) does not exercise.

include("ka_backend.jl")
include("pipeline_field.jl")

using FastMultipole, Printf, LinearAlgebra
const FM = FastMultipole
const V = FLOWVPM

dev_functional() || (println("$(DEV_NAME) not functional; skipping"); exit(0))
ext = Base.get_extension(FastMultipole, :FastMultipoleKAExt)
ext === nothing && error("the KA extension is not loaded")

const TF = Float32
const STEP = 36
const P = 5
const SIZES = (512, 2048, 8192)

relerr(a, b) = (s = maximum(abs.(b)); d = maximum(abs.(a .- b)); s == 0 ? d : d / s)

function device_copy_of(host)
    d = V.ParticleField(host.maxparticles, TF; arraytype=devmatrix, np=host.np,
        fmm=V.FMM(; p=P + 1, autotune_p=false, autotune_ncrit=false,
                    autotune_reg_error=false, default_rho_over_sigma=1.0))
    d.particles .= devarray(host.particles)
    return d
end

# (A) host mirror over the DEVICE state's own inputs
function host_e_from_state(state)
    n = state.counts.n_bodies
    nd = state.counts.n_direct
    out = Array(state.output)
    sb = Array(state.source_bodies)
    cr = Array(state.cell_ranges)
    dt = Array(state.direct_targets)
    ds = Array(state.direct_sources)
    sfs = state.sfs
    tg = zeros(TF, 3, n); om = zeros(TF, 3, n); q = zeros(TF, 3, n)
    FM._host_sfs_tg_and_zero!(tg, om, q, out, sb, sfs.transposed, n)
    FM._host_sfs_zeta_pairs!(om, q, tg, sb, cr, dt, ds, nd, sfs.active_row)
    e = zeros(TF, 3, n)
    FM._host_sfs_form_e!(e, om, q, out, sfs.transposed, n)
    return e, n
end

npass = Ref(0); nfail = Ref(0)
check(name, got, tol) = (ok = got <= tol; ok ? npass[] += 1 : nfail[] += 1;
    @printf("  %-28s %.3e  (tol %.1e)  %s\n", name, got, tol, ok ? "PASS" : "FAIL");
    flush(stdout))

for np in SIZES
    println("=== np = $np ==="); flush(stdout)

    # host arm: FLOWVPM's own CPU radix FMM with SFS
    host = load_wake(STEP; np, TF, P)
    V.UJ_fmm_gpu!(host; reset=true, reset_sfs=true, sfs=true)
    E_host = Array(host.particles)[V.SFS_INDEX, 1:host.np]

    # device arm
    fresh = load_wake(STEP; np, TF, P)
    dev = device_copy_of(fresh)
    V.UJ_fmm_gpu!(dev; reset=true, reset_sfs=true, sfs=true)
    E_dev = Array(dev.particles)[V.SFS_INDEX, 1:dev.np]

    # (A) isolated: re-run the host mirror on the device state's own inputs
    cache = V._radix_fmm_coupling!(dev).cache
    e_ref, n = host_e_from_state(cache.state)
    e_ka = Array(cache.state.sfs.tg)[:, 1:n]
    check("A: E (sorted, same J+pairs)", relerr(e_ka, e_ref), 1e-5)

    # (B) end to end through sfs_to_target!
    check("B: E_str delivered", relerr(E_dev, E_host), 5e-3)
    @printf("     |E_host|max = %.4e   |E_dev|max = %.4e\n",
            maximum(abs.(E_host)), maximum(abs.(E_dev)))
    flush(stdout)
end

println()
println(nfail[] == 0 ? "ALL PASS ($(npass[]) checks)" :
        "FAIL: $(nfail[]) of $(npass[] + nfail[]) checks")
