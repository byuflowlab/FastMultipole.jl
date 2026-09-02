# Is one level past the occupancy cap admissible, faster, and lower-floored?
#
# Job 13562318 showed the deepest admissible depth winning at every rung on
# the H200, so the cap ell = floor(log2 np / 3) is what binds, and nothing has
# measured ell = cap + 1. Bodies per occupied cell sets the Float32 roundoff
# floor (jobs 13562137/13562204/13562289), and one more level cuts it ~8x.
#
# For each case/np and ell in (cap, cap+1): search q for sigma adequacy exactly
# as _radix_auto_geometry does (report the shortfall if none passes), then
# measure median UJ_fmm time at P=5 and the error against the device all-pairs
# sum at each P in PS, so the run answers speed and floor together.
#
# Env: DEV_TF  CASE=wake|ring  NPS  PS  CALLS

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
const NPS    = [parse(Int, s) for s in split(get(ENV, "NPS", "63936,248714"), ",")]
const PS     = [parse(Int, s) for s in split(get(ENV, "PS", "4,6,8,10"), ",")]
const CALLS  = parse(Int, get(ENV, "CALLS", "9"))
const PTIME  = 5

fmm_settings(P) = V.FMM(; p=P+1, autotune_p=false, autotune_ncrit=false,
                          autotune_reg_error=false, default_rho_over_sigma=1.0)

function host_field(np, P)
    if CASE === :ring
        src = make_wake(np; TF=Float64); n = V.get_np(src)
        pf = V.ParticleField(n, Float64; fmm=fmm_settings(P)); A = src.particles
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

function device_copy_of(h, P)
    d = V.ParticleField(h.maxparticles, DEV_TF; arraytype=devmatrix, np=h.np, fmm=fmm_settings(P))
    d.particles .= devarray(DEV_TF.(Array(h.particles)))
    return d
end

# smallest supported q passing sigma adequacy at this depth, or the shortfall
function admissible_q(host, bounds, ell)
    st = V.RadixFMMSettings(; m2l_strategy=:concat)
    reach = st.accuracy_margin * V._radix_primary_reach(V._radix_direct_kernel(st)) *
            Float64(V._radix_sigma_max(host))
    L = bounds[2] isa Real ? Float64(bounds[2]) : Float64(maximum(bounds[2]))
    h = L / 2^ell
    best_gap = 0.0
    for q in sort!([Int(q) for q in FM._SUPPORTED_RIGID_NEAR_RADII2 if q >= st.near_radius2])
        g = FM._ball_stencil_min_gap(q) * h
        g >= reach && return (q, g / reach)
        best_gap = max(best_gap, g)
    end
    return (nothing, best_gap / reach)
end

function all_pairs_arm(state, chunk, wg)
    TF = eltype(state.output); n = size(state.source_bodies, 2); ng = cld(n, chunk)
    CR = typeof(state.cell_ranges); IT = eltype(state.cell_ranges)
    h = zeros(IT, 2, ng + 1)
    for g in 1:ng
        fb = (g - 1) * chunk + 1; h[1, g] = fb; h[2, g] = min(chunk, n - fb + 1)
    end
    h[1, ng+1] = 1; h[2, ng+1] = n
    cr = CR(undef, 2, ng + 1); copyto!(cr, h)
    DT = typeof(state.direct_targets); JT = eltype(state.direct_targets)
    tg = DT(undef, ng); copyto!(tg, JT.(1:ng))
    sr = DT(undef, ng); copyto!(sr, fill(JT(ng + 1), ng))
    hs = size(state.output, 1) >= 13
    b = KA.get_backend(state.output)
    kern = ext._cached_kernel(ext.ka_direct_pairs_functor_kernel!, b, wg)
    dk = ext._ka_device_direct_kernel(state.options.direct_kernel, TF, 0)
    return function ()
        fill!(state.output, zero(TF))
        kern(dk, state.output, state.source_bodies, cr, tg, sr, ng, TF,
             Val(hs), Val(wg); ndrange=ng * wg)
        KA.synchronize(b); return nothing
    end
end

function fmm_vs_allpairs(dev)
    dcache = V._radix_fmm_couplings[dev].cache; dstate = dcache.state
    dsw = FM.DerivativesSwitch(FM.to_vector(false, 1), FM.to_vector(true, 1),
                               FM.to_vector(true, 1), (dev,))
    ext.ka_radix_cache_device_step!(dcache, (dev,), dsw); KA.synchronize(KA.get_backend(dstate.output))
    fmm_out = Float64.(Array(dstate.output)[2:4, :])
    all_pairs_arm(dstate, 64, 64)()
    ref = Float64.(Array(dstate.output)[2:4, :])
    s = maximum(abs, ref)
    return (maximum(abs.(fmm_out .- ref))/s, sqrt(sum(abs2, fmm_out .- ref)/sum(abs2, ref)),
            Int(dstate.counts.n_cells))
end

function drop!(dev)
    haskey(V._radix_fmm_couplings, dev) && V.clear_radix_fmm_cache!(dev)
    delete!(V._radix_fmm_couplings, dev); GC.gc(); GC.gc()
end

println("=== ell past the occupancy cap: $(DEV_NAME) $(DEV_TF) $(CASE) ===")
println("time = median of $CALLS UJ_fmm calls at P=$PTIME; error = device FMM vs device all-pairs")
flush(stdout)

for np in NPS
    host = host_field(np, PTIME); NP = V.get_np(host)
    bounds = V._radix_derive_bounds(host, 0.1)
    cap = max(2, floor(Int, log2(max(NP, 8)) / 3))
    println("-"^96)
    @printf("np=%d  cap ell=%d\n", NP, cap)
    rows = Dict{Int,Float64}()
    for ell in (cap, cap + 1)
        q, margin = admissible_q(host, bounds, ell)
        if q === nothing
            @printf("  ell=%d: NOT ADMISSIBLE -- largest supported q gives %.2fx the required gap\n", ell, margin)
            flush(stdout); continue
        end
        @printf("  ell=%d q=%d (gap margin %.2fx)\n", ell, q, margin)
        # speed at the shared P
        dev = device_copy_of(host, PTIME); ok = true
        try
            V.radix_fmm_settings!(dev; m2l_strategy=:concat, ell=ell, near_radius2=q, expansion_order=PTIME)
            V.UJ_fmm(dev)
        catch e
            @printf("    rejected: %s\n", first(split(sprint(showerror, e), '\n'))); ok = false
        end
        if ok
            ts = Float64[]
            for _ in 1:CALLS
                V._reset_particles(dev); t0 = time_ns(); V.UJ_fmm(dev); push!(ts, (time_ns()-t0)/1e9)
            end
            n_cells = Int(V._radix_fmm_couplings[dev].cache.state.counts.n_cells)
            rows[ell] = median(ts)
            @printf("    P=%d time: median %.5f  min %.5f  | cells %d  bodies/cell %.0f\n",
                    PTIME, median(ts), minimum(ts), n_cells, NP / n_cells)
            flush(stdout)
        end
        drop!(dev)
        ok || continue
        # floor at this depth
        @printf("    %3s | %10s %10s\n", "P", "max relerr", "L2 relerr")
        for P in PS
            h2 = host_field(np, P); dev = device_copy_of(h2, P)
            V.radix_fmm_settings!(dev; m2l_strategy=:concat, ell=ell, near_radius2=q, expansion_order=P)
            V.UJ_fmm(dev)
            mx, l2, _ = fmm_vs_allpairs(dev)
            @printf("    %3d | %10.3e %10.3e\n", P, mx, l2); flush(stdout)
            drop!(dev)
        end
    end
    if haskey(rows, cap) && haskey(rows, cap + 1)
        @printf("  VERDICT np=%d: ell=%d is %.2fx the time of ell=%d (%s)\n", NP, cap + 1,
                rows[cap+1] / rows[cap], cap, rows[cap+1] < rows[cap] ? "FASTER" : "SLOWER")
    end
    flush(stdout)
end
println("=== done ===")
