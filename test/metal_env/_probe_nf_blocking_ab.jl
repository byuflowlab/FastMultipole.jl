# Q3 direct-list re-blocking: A/B of the two nearfield kernel shapes.
#
#   pairs    `ka_direct_pairs_functor_kernel!` -- one workgroup per U-list pair,
#            lanes striding the target bodies. Idle lanes when a leaf holds
#            fewer bodies than the workgroup, which is the common case.
#   blocked  `ka_direct_blocked_functor_kernel!` -- the U list flattened into
#            (pair, target body) work items, one lane each. No idle lanes; costs
#            a prefix sum over the pair list and a binary search per lane.
#
# Reports the nearfield STAGE in isolation (both shapes driven through
# `ka_launch_nearfield!`, so each gets its real launch path including the index
# build) and the END-TO-END `UJ_fmm` step, since the whole point of the memo
# figure was that the stage share and the idle-lane share fight each other.
#
# `relerr` between the two output buffers is load-bearing: the packing must be
# accuracy-neutral by construction, so anything above roundoff is a bug.

include("ka_backend.jl")
include("pipeline_field.jl")

using FastMultipole, Printf
const FM = FastMultipole
const V = FLOWVPM
import KernelAbstractions as KA

dev_functional() || (println("$(DEV_NAME) not functional; skipping"); exit(0))
ext = Base.get_extension(FastMultipole, :FastMultipoleKAExt)
ext === nothing && error("the KA extension is not loaded")

const P     = 5
const CALLS = 12
const LADDER = [(36, 2048), (36, 8192), (36, nothing), (72, nothing),
                (144, nothing), (288, nothing)]
const CASES = haskey(ENV, "CASES") ?
    [(parse(Int, split(c, ':')[1]),
      length(split(c, ':')) > 1 && split(c, ':')[2] != "" ?
        parse(Int, split(c, ':')[2]) : nothing) for c in split(ENV["CASES"], ",")] :
    LADDER

function device_copy_of(h)
    d = V.ParticleField(h.maxparticles, Float32; arraytype=devmatrix, np=h.np,
        fmm=V.FMM(; p=P+1, autotune_p=false, autotune_ncrit=false,
                    autotune_reg_error=false, default_rho_over_sigma=1.0))
    d.particles .= devarray(Float32.(Array(h.particles)))
    return d
end

function timeit(f; calls::Int=CALLS)
    f()
    ts = Float64[]
    for _ in 1:calls
        t0 = time_ns(); f(); push!(ts, (time_ns() - t0)/1e9)
    end
    minimum(ts)
end

function run_case(step, np)
    dev = device_copy_of(load_wake(step; np=np, TF=Float64, P=P))
    V.radix_fmm_settings!(dev; m2l_strategy=:concat)
    V.UJ_fmm(dev)
    st = V._radix_fmm_couplings[dev]
    state = st.cache.state
    backend = KA.get_backend(state.output)
    TF = eltype(state.output)
    NP = V.get_np(dev)
    npairs = state.counts.n_direct
    ranges = Array(state.cell_ranges)
    dt = Array(state.direct_targets)
    nitems = sum(ranges[2, dt[k]] for k in 1:npairs)
    occ = nitems / npairs                       # mean targets per pair
    @printf("np=%-7d ell=%d n_cells=%-6d n_direct=%-7d mean target occ=%.1f  idle lanes=%.1f%%\n",
            NP, st.cache.ell, state.counts.n_cells, npairs, occ,
            100 * max(0.0, 1 - occ / 64))

    #--- stage in isolation ---
    stage(b) = timeit(() -> (ext.ka_launch_nearfield!(state; clear=true, blocked=b);
                             KA.synchronize(backend)))
    t_pairs = stage(false)
    ext.ka_launch_nearfield!(state; clear=true, blocked=false)
    KA.synchronize(backend); out_pairs = Array(state.output)
    t_blocked = stage(true)
    ext.ka_launch_nearfield!(state; clear=true, blocked=true)
    KA.synchronize(backend); out_blocked = Array(state.output)
    den = maximum(abs.(out_pairs))
    rel = den > 0 ? maximum(abs.(out_blocked .- out_pairs)) / den : 0.0
    @printf("  stage    pairs %8.5f s   blocked %8.5f s   %.2fx   relerr=%.2e\n",
            t_pairs, t_blocked, t_pairs / t_blocked, rel)

    #--- end to end ---
    function uj(b)
        ext.KA_NEARFIELD_BLOCKED[] = b
        t = timeit(() -> (V._reset_particles(dev); V.UJ_fmm(dev)))
        ext.KA_NEARFIELD_BLOCKED[] = true
        t
    end
    e_pairs = uj(false)
    e_blocked = uj(true)
    @printf("  UJ_fmm   pairs %8.5f s   blocked %8.5f s   %.2fx   (stage share %.0f%% -> %.0f%%)\n\n",
            e_pairs, e_blocked, e_pairs / e_blocked,
            100 * t_pairs / e_pairs, 100 * t_blocked / e_blocked)
    flush(stdout)
end

println("=== nearfield re-blocking A/B (", DEV_NAME, ", P=", P, ", :concat) ===\n")
for (s, n) in CASES
    run_case(s, n)
end
