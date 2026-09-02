# Are the seven radix cost-model constants (FLOWVPM_fmm_radix.jl:493-499) still
# ordering (ell, q) candidates correctly on THIS device?
#
# They were fit over 16 configurations topping out at np=115455, the motivating
# measurement in that comment is Metal, and we now run H200 at np=249k --
# extrapolating routes ~ n_cells^2.22 well past the calibration set. The model
# picks ell/q at every cache build, so a mis-ordering is a live cost.
#
# For each case: enumerate every admissible (ell, q) exactly as
# _radix_auto_geometry does, MEASURE the median device UJ_fmm at each, print the
# model's predicted cost beside it, and report whether the model's argmin is the
# measured argmin. Median, not minimum: the launch-queue jitter finding (job
# 13561994) showed min-of-N misreporting the Float64 top rung by 1.7x.
#
# Env: DEV_TF  CASE=wake|ring  NPS=comma list  CALLS

include("ka_backend.jl")
include("pipeline_field.jl")
using FastMultipole, Printf, Statistics
const FM = FastMultipole
const V = FLOWVPM
import KernelAbstractions as KA

dev_functional() || (println("$(DEV_NAME) not functional; skipping"); exit(0))
ext = Base.get_extension(FastMultipole, :FastMultipoleKAExt)
ext === nothing && error("the KA extension is not loaded")

const DEV_TF = getfield(Base, Symbol(get(ENV, "DEV_TF", "Float32")))
const CASE   = Symbol(get(ENV, "CASE", "wake"))
const NPS    = [parse(Int, s) for s in split(get(ENV, "NPS", "15984,63936,115455,248714"), ",")]
const CALLS  = parse(Int, get(ENV, "CALLS", "9"))
const P      = parse(Int, get(ENV, "P", "5"))

settings(; kw...) = V.RadixFMMSettings(; m2l_strategy=:concat, kw...)

function host_field(np)
    if CASE === :ring
        src = make_wake(np; TF=Float64); n = V.get_np(src)
        pf = V.ParticleField(n, Float64; fmm=V.FMM(; p=P+1, autotune_p=false,
                autotune_ncrit=false, autotune_reg_error=false,
                default_rho_over_sigma=1.0))
        A = src.particles
        for i in 1:n
            V.add_particle(pf, (A[V.X_INDEX[1],i], A[V.X_INDEX[2],i], A[V.X_INDEX[3],i]),
                           (A[V.GAMMA_INDEX[1],i], A[V.GAMMA_INDEX[2],i], A[V.GAMMA_INDEX[3],i]),
                           A[V.SIGMA_INDEX,i]; vol=A[V.VOL_INDEX,i])
        end
        return pf
    end
    for s in (36, 72, 144, 288, 720)
        n_all = h5open(h -> length(read(h["sigma"])), wake_path(s))
        n_all >= np && return load_wake(s; np=(n_all == np ? nothing : np), TF=Float64, P=P)
    end
    error("no wake dump holds $np particles")
end

device_copy_of(h) = begin
    d = V.ParticleField(h.maxparticles, DEV_TF; arraytype=devmatrix, np=h.np,
            fmm=V.FMM(; p=P+1, autotune_p=false, autotune_ncrit=false,
                        autotune_reg_error=false, default_rho_over_sigma=1.0))
    d.particles .= devarray(DEV_TF.(Array(h.particles)))
    d
end

# every (ell, q) the auto rule would consider: smallest admissible q per depth,
# down from the occupancy cap, mirroring _radix_auto_geometry
function candidates(host, bounds)
    st = settings()
    reach = st.accuracy_margin * V._radix_primary_reach(V._radix_direct_kernel(st)) *
            Float64(V._radix_sigma_max(host))
    L = bounds[2] isa Real ? Float64(bounds[2]) : Float64(maximum(bounds[2]))
    qs = sort!([Int(q) for q in FM._SUPPORTED_RIGID_NEAR_RADII2 if q >= st.near_radius2])
    out = Tuple{Int,Int}[]
    for ell in max(2, floor(Int, log2(max(host.np, 8)) / 3)):-1:2
        h = L / 2^ell
        for q in qs
            if FM._ball_stencil_min_gap(q) * h >= reach; push!(out, (ell, q)); break; end
        end
    end
    return out
end

println("=== (ell,q) cost-model calibration: $(DEV_NAME) $(DEV_TF) $(CASE) P=$P ===")
println("measured = median of $CALLS device UJ_fmm calls; predicted = _radix_cost_estimate")
flush(stdout)

for np in NPS
    host = host_field(np); NP = V.get_np(host)
    bounds = V._radix_derive_bounds(host, 0.1)
    cands = candidates(host, bounds)
    model_pick = V._radix_auto_geometry(
        bounds[2] isa Real ? Float64(bounds[2]) : Float64(maximum(bounds[2])),
        Float64(V._radix_sigma_max(host)), NP, settings().near_radius2,
        V._radix_primary_reach(V._radix_direct_kernel(settings())),
        settings().accuracy_margin; cost_field=host, cost_bounds=bounds)
    println("-"^96)
    @printf("np=%d  admissible=%s  model picks (ell,q)=%s\n", NP, cands, model_pick)
    @printf("  %4s %4s | %9s %9s | %9s %9s | %8s %12s | %9s\n",
            "ell","q","meas med","meas min","pred s","pred/meas","cells","ninter","relerr")
    flush(stdout)
    best_meas = (Inf, nothing); best_pred = (Inf, nothing)
    meas = Dict{Tuple{Int,Int},Float64}()
    V.UJ_fmm(host; autotune=false)
    U_ref = copy(Array(host.particles)[V.U_INDEX, 1:NP])
    for (ell, q) in cands
        dev = device_copy_of(host)
        ok = true
        try
            V.radix_fmm_settings!(dev; m2l_strategy=:concat, ell=ell, near_radius2=q,
                                  expansion_order=P)
            V.UJ_fmm(dev)
        catch e
            @printf("  %4d %4d | rejected: %s\n", ell, q,
                    first(split(sprint(showerror, e), '\n'))); ok = false
        end
        if ok
            ts = Float64[]
            for _ in 1:CALLS
                V._reset_particles(dev); t0 = time_ns(); V.UJ_fmm(dev)
                push!(ts, (time_ns()-t0)/1e9)
            end
            U_dev = Array(dev.particles)[V.U_INDEX, 1:NP]
            rel = maximum(abs.(U_dev .- U_ref)) / maximum(abs.(U_ref))
            n_cells, ninter = V._radix_grid_counts(host, bounds, ell, q)
            pred = V._radix_cost_estimate(NP, n_cells, ninter)
            m = median(ts); meas[(ell, q)] = m
            m < best_meas[1] && (best_meas = (m, (ell, q)))
            pred < best_pred[1] && (best_pred = (pred, (ell, q)))
            @printf("  %4d %4d | %9.5f %9.5f | %9.5f %9.2f | %8d %12.4g | %9.2e\n",
                    ell, q, m, minimum(ts), pred, pred/m, n_cells, ninter, rel)
            flush(stdout)
        end
        haskey(V._radix_fmm_couplings, dev) && V.clear_radix_fmm_cache!(dev)
        delete!(V._radix_fmm_couplings, dev); GC.gc(); GC.gc()
    end
    agree = best_pred[2] == best_meas[2]
    @printf("  VERDICT np=%d: measured fastest %s (%.5f s), model argmin %s, auto rule %s -> %s\n",
            NP, best_meas[2], best_meas[1], best_pred[2], model_pick,
            agree ? "AGREE" : "DISAGREE")
    if !agree && haskey(meas, model_pick)
        @printf("  cost of disagreement: the auto rule's %s runs %.2fx the measured best\n",
                model_pick, meas[model_pick] / best_meas[1])
    end
    flush(stdout)
end
println("=== done ===")
