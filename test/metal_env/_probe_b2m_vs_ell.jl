# Does the depth rule's cost model mis-order depths because it ignores B2M/L2B?
#
# The suspicion in [[project-b2m-l2b-are-the-real-cost]] was that the
# session-37 cost model (far = a + b*routes + c*np ; near = d + e*ninter) has
# no B2M or L2B term, so it might be picking a depth those stages punish.
# B2M's BODY work is ell-invariant (np x 63 (n,m) either way) -- but its GROUP
# count is n_cells, and each group pays 63 reductions, so the reduction half
# does scale with depth. Same for L2B's group count. Measure the two stages
# per forced depth and see whether including them would move the argmin.

include("ka_backend.jl")
include("pipeline_field.jl")
using FastMultipole, Printf, Statistics
const FM = FastMultipole
const V = FLOWVPM
import KernelAbstractions as KA
dev_functional() || (println("skipping"); exit(0))
const ext = Base.get_extension(FastMultipole, :FastMultipoleKAExt)

const P = 5
const STEP = parse(Int, get(ENV, "UJ_STEP", "36"))
const NTRIAL = 15

med_ms(f, backend) = begin
    f(); KA.synchronize(backend)
    ts = Float64[]
    for _ in 1:NTRIAL
        t = time_ns(); f(); KA.synchronize(backend); push!(ts, (time_ns()-t)/1e6)
    end
    median(ts)
end

host = load_wake(STEP; TF=Float64, P=P)
NP = V.get_np(host)
bounds = V._radix_derive_bounds(host, 0.1; rectangular=false)
L = bounds[2] isa Real ? Float64(bounds[2]) : Float64(maximum(bounds[2]))
sigma_max = Float64(V._radix_sigma_max(host))
reach = 1.03 * 4.789 * sigma_max
qs = sort!([Int(q) for q in FM._SUPPORTED_RIGID_NEAR_RADII2 if q >= 6])
# smallest admissible q at each depth, exactly as _radix_auto_geometry searches
admissible = Tuple{Int,Int}[]
for ell in 2:max(2, floor(Int, log2(max(NP,8))/3))
    for q in qs
        if FM._ball_stencil_min_gap(q) * (L/2^ell) >= reach
            push!(admissible, (ell, q)); break
        end
    end
end

@printf("np=%d  admissible (ell,q): %s\n\n", NP, admissible)
@printf("%4s %3s %8s %9s %9s %9s %9s %9s %9s\n",
        "ell","q","n_cells","step","near","b2m","m2l","l2b","b2m+l2b")
for (ell, q) in admissible
    d = V.ParticleField(host.maxparticles, Float32; arraytype=devmatrix, np=host.np,
        fmm=V.FMM(; p=P+1, autotune_p=false, autotune_ncrit=false,
                    autotune_reg_error=false, default_rho_over_sigma=1.0))
    d.particles .= devarray(Float32.(Array(host.particles)))
    V.radix_fmm_settings!(d; m2l_strategy=:concat, ell=ell, near_radius2=q)
    local state, backend
    try
        for _ in 1:4; V._reset_particles(d); V.UJ_fmm(d); end
        state = V._radix_fmm_couplings[d].cache.state
        ws = state.scratch
        backend = KA.get_backend(state.output)
        t_step = med_ms(() -> (V._reset_particles(d); V.UJ_fmm(d)), backend)
        t_near = med_ms(() -> ext.ka_launch_nearfield!(state; workgroup=64, clear=true), backend)
        t_b2m  = med_ms(() -> ext.ka_launch_b2m!(state; workgroup=128), backend)
        t_m2l  = med_ms(() -> ext.ka_launch_m2l!(state, ws), backend)
        t_l2b  = med_ms(() -> ext.ka_launch_l2b!(state; workgroup=64), backend)
        @printf("%4d %3d %8d %9.3f %9.3f %9.3f %9.3f %9.3f %9.3f\n",
                ell, q, state.counts.n_cells, t_step, t_near, t_b2m, t_m2l,
                t_l2b, t_b2m + t_l2b)
    catch err
        @printf("%4d %3d  FAILED: %s\n", ell, q, sprint(showerror, err)[1:min(end,90)])
    end
    flush(stdout)
    haskey(V._radix_fmm_couplings, d) && (V.clear_radix_fmm_cache!(d);
        delete!(V._radix_fmm_couplings, d))
    GC.gc(); GC.gc()
end
