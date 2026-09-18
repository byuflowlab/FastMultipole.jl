# Q2: is the far-field overhead at small np reducible by choosing a shallower
#     grid?  Sweep `ell` at fixed np and price the far field against the near.
# Q3: how ragged is the direct list?  The nearfield kernel gives each target
#     cell one workgroup of WG lanes, so a target cell holding fewer than WG
#     bodies runs idle lanes.  Measure the occupancy distribution directly.
include("ka_backend.jl")
include("pipeline_field.jl")
using FastMultipole, Printf, Statistics
const FM = FastMultipole
const V = FLOWVPM
import KernelAbstractions as KA

dev_functional() || (println("$(DEV_NAME) not functional; skipping"); exit(0))
ext = Base.get_extension(FastMultipole, :FastMultipoleKAExt)
const P = 5
const WG = 64
const CALLS = 10

function timeit(f; calls=CALLS)
    ts = Float64[]
    for _ in 1:calls; GC.gc(); t0 = time_ns(); f(); push!(ts, (time_ns()-t0)/1e9); end
    return minimum(ts)
end

function device_copy_of(h)
    d = V.ParticleField(h.maxparticles, Float32; arraytype=devmatrix, np=h.np,
        fmm=V.FMM(; p=P+1, autotune_p=false, autotune_ncrit=false,
                    autotune_reg_error=false, default_rho_over_sigma=1.0))
    d.particles .= devarray(Float32.(Array(h.particles)))
    return d
end

const CASES = [(36, 2048), (36, 8192), (36, nothing), (72, nothing), (144, nothing), (288, nothing)]
const ELLS  = [nothing, 2, 3, 4]      # `nothing` = whatever the auto rule picks

wake_sigma(pf) = [V.get_sigma(V.get_particle(pf, i))[] for i in 1:V.get_np(pf)]
function L_geo(pf)
    np = V.get_np(pf)
    X = reduce(hcat, [collect(V.get_X(V.get_particle(pf, i))) for i in 1:np])
    ext_ = vec(maximum(X; dims=2) .- minimum(X; dims=2))
    return maximum(ext_) * 1.1        # matches settings.padding = 0.1
end

println("=== ell sweep + direct-list occupancy (WG = $WG lanes) ===")
@printf("%7s %4s | %9s %9s %9s | %8s %9s | %8s %8s %8s | %9s\n",
        "np","ell","total s","near s","far s","routes","near%n2",
        "occ mean","occ med","waste","relerr")
flush(stdout)

for (step, np) in CASES
    host = load_wake(step; np=np, TF=Float64, P=P)
    NP = V.get_np(host)
    V.UJ_fmm(host; autotune=false)
    U_ref = copy(Array(host.particles)[V.U_INDEX, 1:NP])
    println("-"^118)
    for ell in ELLS
        dev = device_copy_of(load_wake(step; np=np, TF=Float64, P=P))
        ok = true
        # the auto rule searches `q` upward at each depth to satisfy sigma
        # adequacy; a forced ell must do the same or it fails on a technicality
        qsel = nothing
        if ell !== nothing
            reach = 1.03 * 4.789 * maximum(wake_sigma(host))
            h = L_geo(host) / 2^ell
            for q in FM._SUPPORTED_RIGID_NEAR_RADII2
                q >= 6 || continue
                if FM._ball_stencil_min_gap(q) * h >= reach; qsel = q; break; end
            end
            qsel === nothing && (@printf("%7d %4d | no admissible near_radius2\n", NP, ell); ok = false)
        end
        ok && try
            V.radix_fmm_settings!(dev; m2l_strategy=:concat, ell=ell,
                                  near_radius2=something(qsel, 6))
            V.UJ_fmm(dev)
        catch e
            @printf("%7d %4s | rejected: %s\n", NP, string(ell),
                    first(split(sprint(showerror, e), '\n')))
            ok = false
        end
        if ok
            ttot = timeit(() -> (V._reset_particles(dev); V.UJ_fmm(dev)))
            U_dev = Array(dev.particles)[V.U_INDEX, 1:NP]
            rel = maximum(abs.(U_dev .- U_ref)) / maximum(abs.(U_ref))
            dcache = V._radix_fmm_couplings[dev].cache
            st = dcache.state
            backend = KA.get_backend(st.output)
            tnf = timeit(() -> (ext.ka_launch_nearfield!(st); KA.synchronize(backend)))
            rr = Array(st.cell_ranges)
            dt = Array(st.direct_targets); ds = Array(st.direct_sources)
            nd = st.counts.n_direct
            tocc = [Int(rr[2, dt[k]]) for k in 1:nd]      # target bodies per workgroup
            ninter = sum(Int(rr[2, dt[k]]) * Int(rr[2, ds[k]]) for k in 1:nd)
            # a workgroup processes ceil(occ/WG) lane-passes; waste = idle lane fraction
            lanes = sum(cld(o, WG) * WG for o in tocc)
            waste = 1 - sum(tocc) / lanes
            # the work the kernel actually issues: each target cell gets
            # ceil(occ_t/WG) full WG-lane passes, each sweeping all occ_s sources
            lanework = sum(cld(Int(rr[2, dt[k]]), WG) * WG * Int(rr[2, ds[k]])
                           for k in 1:nd)
            @printf("%7d %4s | %9.5f %9.5f %9.5f | %8d %8.2f%% | %8.1f %8d %7.1f%% | %9.2e\n",
                    NP, string(dcache.ell, ell === nothing ? "*" : ""),
                    ttot, tnf, ttot - tnf, st.counts.n_routes,
                    100*ninter/float(NP)^2, mean(tocc), median(tocc),
                    100*waste, rel)
            @printf("        ^ n_cells=%d  n_direct=%d  ninter=%.6g  lanework=%.6g\n",
                    st.counts.n_cells, nd, float(ninter), float(lanework))
            flush(stdout)
        end
        haskey(V._radix_fmm_couplings, dev) && V.clear_radix_fmm_cache!(dev)
        delete!(V._radix_fmm_couplings, dev)
        GC.gc(); GC.gc()
    end
end
