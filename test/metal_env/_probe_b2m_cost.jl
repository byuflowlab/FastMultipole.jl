# Why is B2M 18.6% of the GPU step when the CPU's whole upward pass is 0.3%?
#
# Hypothesis: `ka_b2m_vortex_leaf_nodes_kernel!` (ext:2677) runs one full
# WG-lane halving-tree reduction, with ~log2(WG)+2 `@synchronize` barriers, PER
# (n,m) PAIR. With ~104 bodies per cell and WG=128 each thread owns at most one
# body, so the barriers -- not the physics -- are the whole cost.
#
# Discriminator: sweep the workgroup size. Total BODY work is identical at every
# WG (each body is visited once per (n,m) either way). If the kernel is
# body-work bound the times are flat; if it is reduction bound the time tracks
# the lane count / barrier count.

include("ka_backend.jl")
include("pipeline_field.jl")

using FastMultipole, Printf
const FM = FastMultipole
const V = FLOWVPM
import KernelAbstractions as KA

dev_functional() || (println("$(DEV_NAME) not functional; skipping"); exit(0))
ext = Base.get_extension(FastMultipole, :FastMultipoleKAExt)
ext === nothing && error("the KA extension is not loaded")

const STEP = 36
const P = 5
const CALLS = 12

host = load_wake(STEP; TF=Float64, P=P)
NP = V.get_np(host)
dev = V.ParticleField(host.maxparticles, Float32; arraytype=devmatrix, np=host.np,
    fmm=V.FMM(; p=P+1, autotune_p=false, autotune_ncrit=false,
                autotune_reg_error=false, default_rho_over_sigma=1.0))
dev.particles .= devarray(Float32.(Array(host.particles)))
V.radix_fmm_settings!(dev; m2l_strategy=:concat)
V.UJ_fmm(dev)

st = V._radix_fmm_couplings[dev]
state = st.cache.state
orders = state.invariant_cache.basis_info.orders
backend = KA.get_backend(state.multipoles.phi)

P_phi, P_chi = orders.P_phi, orders.P_active
nm_phi = sum(n -> n + 1, 0:P_phi)
nm_chi = P_chi >= 1 ? sum(n -> n + 1, 1:P_chi) : 0
ncell = state.counts.n_cells

println("=== B2M cost probe ===")
@printf("np=%d  n_cells=%d  P_phi=%d  P_chi(P_active)=%d\n", NP, ncell, P_phi, P_chi)
@printf("(n,m) pairs: phi=%d  chi=%d  total=%d\n", nm_phi, nm_chi, nm_phi + nm_chi)

counts = Array(state.cell_ranges)[2, 1:ncell]
@printf("bodies/cell: min=%d  median=%d  mean=%.1f  max=%d  sum=%d\n",
        minimum(counts), sort(counts)[cld(ncell,2)], sum(counts)/ncell,
        maximum(counts), sum(counts))
println()

# Total per-cell (n,m)x body evaluations -- the irreducible physics work.
evals = (nm_phi + nm_chi) * sum(counts)
@printf("irreducible body-x-(n,m) evaluations: %d\n", evals)
@printf("barriers at WG=w: %d groups x %d (n,m) x (log2(w)+2)\n\n",
        ncell, nm_phi + nm_chi)

function time_b2m(wg)
    f() = (ext.ka_launch_b2m!(state; workgroup=wg); KA.synchronize(backend))
    f()  # compile
    ts = Float64[]
    for _ in 1:CALLS
        t0 = time_ns(); f(); push!(ts, (time_ns() - t0)/1e9)
    end
    sort(ts[cld(CALLS,2)+1:end])[1]
end

@printf("  %-8s %10s %10s %12s\n", "WG", "sec", "barriers", "vs WG=128")
base = Ref{Union{Nothing,Float64}}(nothing)
for wg in (16, 32, 64, 128, 256)
    t = try
        time_b2m(wg)
    catch err
        @printf("  %-8d  FAILED: %s\n", wg, sprint(showerror, err)[1:min(end,80)])
        continue
    end
    wg == 128 && (base[] = t)
    nbar = ncell * (nm_phi + nm_chi) * (Int(log2(wg)) + 2)
    @printf("  %-8d %10.5f %10d %12s\n", wg, t, nbar,
            base[] === nothing ? "-" : @sprintf("%.2fx", t/base[]))
    flush(stdout)
end
