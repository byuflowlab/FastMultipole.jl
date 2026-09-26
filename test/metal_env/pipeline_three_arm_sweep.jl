# Threaded CPU FMM vs KA FMM vs KA all-pairs DIRECT, over a size ladder.
#
# Three arms, same real particle field, same body ordering:
#
#   cpu     Array-backed field, FLOWVPM's own `UJ_fmm` -> fmm.fmm!(pfield; ...)
#           Threaded, but note FastMultipole forces n_threads=1 below
#           MIN_BODIES (src/FastMultipole.jl:35): sizes under that are
#           SINGLE-threaded and are flagged `st` in the output.
#   ka_fmm  device-backed field, `UJ_fmm` -> UJ_fmm_gpu! -> _radix_fmm_evaluate!
#   ka_dir  the PRODUCTION pair kernel `ka_direct_pairs_functor_kernel!` driven
#           with an all-pairs list (targets chunked so there is parallelism, one
#           source cell spanning all n). Per-pair math is byte-identical to the
#           nearfield stage, so this is the O(n^2) cost of the same physics, not
#           a second kernel. `direct_gpu!` is a dead export and cannot be used.
#
# `relerr` columns are load-bearing: all three arms write U over the same body
# ordering, so agreement proves each arm computes the same physics and the
# timing gap is real.
#
# The direct arm is skipped once n^2 exceeds MAX_PAIRS so the top of the ladder
# does not stall for minutes on an O(n^2) arm whose scaling is already clean.

include("ka_backend.jl")
include("pipeline_field.jl")

using FastMultipole, Printf
using FastMultipole.StaticArrays
const FM = FastMultipole
const V = FLOWVPM
import KernelAbstractions as KA

dev_functional() || (println("$(DEV_NAME) not functional; skipping"); exit(0))
ext = Base.get_extension(FastMultipole, :FastMultipoleKAExt)
ext === nothing && error("the KA extension is not loaded")

# MODE=ka   (default) the KA extension drives the device lifecycle, via the
#           _RADIX_DEVICE_*_HOOK registry. All three arms run.
# MODE=cuda load_cuda_radix_lifecycle!() REDEFINES _radix_cache_device_build /
#           _radix_cache_device_step!, so the CUDA path wins over the registered
#           KA hook for the rest of the process. One process cannot host both
#           arms; run the script twice. The KA-internal probes (nearfield alone,
#           all-pairs direct) do not apply and are skipped.
const MODE = Symbol(get(ENV, "MODE", "ka"))
MODE in (:ka, :cuda) || error("MODE must be ka or cuda, got $(MODE)")
const IS_KA = MODE === :ka
const FMM_ARM = IS_KA ? "ka_fmm " : "cufmm  "
if !IS_KA
    FM.load_cuda_radix_lifecycle!() || error("load_cuda_radix_lifecycle!() failed: " *
                                             FM.cuda_radix_status())
end

const P         = 5
const STRATEGY  = :concat
const GPU_CALLS = 12      # Metal needs ~5 calls of kernel-cache ramp
const CPU_CALLS = 5       # flat after call 1
const DIR_CALLS = 5
const DEV_TF = getfield(Base, Symbol(get(ENV, "DEV_TF", "Float32")))  # device field type; Int width follows it
haskey(ENV, "GH_MODE") && FastMultipole.set_radix_setting!(:CUDA_NEARFIELD_GH_MODE, Symbol(ENV["GH_MODE"]))  # native nearfield g/h precision
const MAX_PAIRS = parse(Float64, get(ENV, "MAX_PAIRS", "1e11"))

# (step, np) -- np=nothing means the whole dump. Smallest first, always.
const LADDER = [(36, 2048), (36, 8192), (36, nothing), (72, nothing),
                (144, nothing), (288, nothing), (720, nothing)]
const CASES = haskey(ENV, "CASES") ?
    [(parse(Int, split(c, ':')[1]),
      length(split(c, ':')) > 1 && split(c, ':')[2] != "" ?
        parse(Int, split(c, ':')[2]) : nothing) for c in split(ENV["CASES"], ",")] :
    LADDER

"Minimum and median wall time in seconds over `calls` timed calls."
function timeit(f; calls::Int)
    ts = Float64[]
    for _ in 1:calls
        GC.gc(); t0 = time_ns(); f(); push!(ts, (time_ns() - t0)/1e9)
    end
    s = sort(ts)
    return (min=s[1], med=s[(length(s)+1) ÷ 2], ramp=ts)
end

ramps(ts) = join((@sprintf("%.4f", t) for t in ts), "/")

function device_copy_of(h)
    d = V.ParticleField(h.maxparticles, DEV_TF; arraytype=devmatrix, np=h.np,
        fmm=V.FMM(; p=P+1, autotune_p=false, autotune_ncrit=false,
                    autotune_reg_error=false, default_rho_over_sigma=1.0))
    d.particles .= devarray(DEV_TF.(Array(h.particles)))
    return d
end

# ---- native CUDA all-pairs arm (MODE=cuda only) ----------------------------
#
# `ka_direct_all_pairs_kernel!` is the KA arm; there is no CUDA equivalent in
# src/ (`:RADIX_DIRECT_ARM` is read only in the KA extension, and `direct_gpu!`
# is a dead export). This is a probe-local mirror of that kernel written with
# CUDA.jl directly, so the CUDA side of the ladder has a direct arm to compare.
#
# It calls the SAME `FastMultipole._direct_pair_ug/_ugh` with the SAME
# `Val(:shipped)` series and the same plain `inv(sqrt(r2))`, so per-pair the
# physics is identical to both the KA arm and the nearfield stage; only the
# launch is CUDA-native. Scored against the FMM output like every other arm.
@static if !HAS_METAL
function cu_all_pairs_kernel!(kernel, output, source_bodies, nbodies, ::Type{T},
                              ::Val{HS}) where {T,HS}
    i = (CUDA.blockIdx().x - Int32(1)) * CUDA.blockDim().x + CUDA.threadIdx().x
    ep = FastMultipole._emits_potential(kernel)
    ghv = Val(:shipped)
    @inbounds if i <= nbodies
        xi = source_bodies[1, i]; yi = source_bodies[2, i]; zi = source_bodies[3, i]
        u = zero(T); gx = zero(T); gy = zero(T); gz = zero(T)
        h1 = zero(T); h2 = zero(T); h3 = zero(T)
        h4 = zero(T); h5 = zero(T); h6 = zero(T)
        h7 = zero(T); h8 = zero(T); h9 = zero(T)
        for j in 1:nbodies
            if i != j
                dx = xi - source_bodies[1, j]
                dy = yi - source_bodies[2, j]
                dz = zi - source_bodies[3, j]
                r2 = dx * dx + dy * dy + dz * dz
                if r2 > zero(r2)
                    invr = inv(sqrt(r2))
                    if HS
                        du, dgx, dgy, dgz, dh1, dh2, dh3, dh4, dh5, dh6, dh7, dh8, dh9 =
                            FastMultipole._direct_pair_ugh(kernel, dx, dy, dz, r2, invr,
                                source_bodies, j, ghv)
                        u += du; gx += dgx; gy += dgy; gz += dgz
                        h1 += dh1; h2 += dh2; h3 += dh3
                        h4 += dh4; h5 += dh5; h6 += dh6
                        h7 += dh7; h8 += dh8; h9 += dh9
                    else
                        du, dgx, dgy, dgz = FastMultipole._direct_pair_ug(kernel,
                            dx, dy, dz, r2, invr, source_bodies, j, ghv)
                        u += du; gx += dgx; gy += dgy; gz += dgz
                    end
                end
            end
        end
        if ep; output[1, i] = u; end
        output[2, i] = gx; output[3, i] = gy; output[4, i] = gz
        if HS
            output[5, i]  = h1; output[6, i]  = h2; output[7, i]  = h3
            output[8, i]  = h4; output[9, i]  = h5; output[10, i] = h6
            output[11, i] = h7; output[12, i] = h8; output[13, i] = h9
        end
    end
    return nothing
end

function make_cuda_direct_arm(state, threads::Int=256)
    TF = eltype(state.output)
    n  = Int(state.counts.n_bodies)
    hs = size(state.output, 1) >= 13
    dkernel = ext._ka_device_direct_kernel(state.options.direct_kernel, TF, 0)
    blocks = cld(n, threads)
    return function ()
        CUDA.@cuda threads=threads blocks=blocks cu_all_pairs_kernel!(
            dkernel, state.output, state.source_bodies, n, TF, Val(hs))
        CUDA.synchronize()
        return nothing
    end
end
end

# All-pairs launcher over the device state's own body array: `chunk` target
# bodies per workgroup, one source cell spanning all n. The kernel's own
# `i != j` guard handles self-interaction, as in the nearfield stage.
function make_direct_arm(state, chunk::Int, workgroup::Int)
    TF = eltype(state.output)
    n  = size(state.source_bodies, 2)
    ng = cld(n, chunk)
    CR = typeof(state.cell_ranges); IT = eltype(state.cell_ranges)
    h_cr = zeros(IT, 2, ng + 1)
    for g in 1:ng
        first_b = (g - 1) * chunk + 1
        h_cr[1, g] = first_b
        h_cr[2, g] = min(chunk, n - first_b + 1)
    end
    h_cr[1, ng+1] = 1; h_cr[2, ng+1] = n            # the all-source cell
    cr = CR(undef, 2, ng + 1); copyto!(cr, h_cr)
    DT = typeof(state.direct_targets); JT = eltype(state.direct_targets)
    tg = DT(undef, ng); copyto!(tg, JT.(1:ng))
    sr = DT(undef, ng); copyto!(sr, fill(JT(ng + 1), ng))
    hs = size(state.output, 1) >= 13
    backend = KA.get_backend(state.output)
    kern = ext._cached_kernel(ext.ka_direct_pairs_functor_kernel!, backend, workgroup)
    # rebuild the functor with TF cutoffs + the reciprocal-sigma row exactly as
    # the production launcher `ka_launch_nearfield!` does, so the per-pair math
    # really is byte-identical to the nearfield stage
    dkernel = ext._ka_device_direct_kernel(state.options.direct_kernel, TF,
        0)
    return function ()
        fill!(state.output, zero(TF))
        kern(dkernel, state.output, state.source_bodies,
             cr, tg, sr, ng, TF, Val(hs), Val(workgroup); ndrange=ng * workgroup)
        KA.synchronize(backend)
        return nothing
    end
end

println("=== three-arm speed sweep: $(WAKE_CASE) ===")
println("julia threads = ", Threads.nthreads(),
        "   FastMultipole MIN_BODIES = ", FM.MIN_BODIES)
println("device = ", DEV_NAME, "   P = ", P, "   m2l_strategy = ", STRATEGY)
@printf("calls: cpu %d, ka_fmm %d, ka_dir %d   direct skipped above %.3g pairs\n",
        CPU_CALLS, GPU_CALLS, DIR_CALLS, MAX_PAIRS)
Threads.nthreads() > 1 || @warn "julia started with 1 thread; the CPU arm is not threaded"
flush(stdout)

const SUMMARY = NamedTuple[]

function run_case(step::Int, np)
    host = load_wake(step; np=np, TF=Float64, P=P)
    NP = V.get_np(host)
    wk = wake_stats(host)
    println("\n", "="^96)
    @printf("step %d  np = %d   box L = %.4g   sigma in [%.4g, %.4g]%s\n",
            step, NP, wk.L, wk.sigma_min, wk.sigma_max,
            NP >= FM.MIN_BODIES ? "" : "   [CPU arm is SINGLE-threaded: np < MIN_BODIES]")
    flush(stdout)

    #--- arm 1: threaded CPU FMM -------------------------------------------
    V.UJ_fmm(host; autotune=false)                 # warm compile, discard
    V._reset_particles(host)
    tc = timeit(() -> (V._reset_particles(host); V.UJ_fmm(host; autotune=false));
                calls=CPU_CALLS)
    U_ref = copy(Array(host.particles)[V.U_INDEX, 1:NP])
    @printf("  cpu     %9.5f s min  (ramp %s)\n", tc.min, ramps(tc.ramp)); flush(stdout)

    #--- arm 2: KA FMM ------------------------------------------------------
    dev = device_copy_of(load_wake(step; np=np, TF=Float64, P=P))
    V.radix_fmm_settings!(dev; m2l_strategy=STRATEGY)
    V.UJ_fmm(dev)                                  # warm compile, discard
    tg = timeit(() -> (V._reset_particles(dev); V.UJ_fmm(dev)); calls=GPU_CALLS)
    U_dev = Array(dev.particles)[V.U_INDEX, 1:NP]
    rel_fmm = maximum(abs.(U_dev .- U_ref)) / maximum(abs.(U_ref))
    @printf("  %s %9.5f s min  (ramp %s)   relerr(U) vs cpu = %.3e\n",
            FMM_ARM, tg.min, ramps(tg.ramp), rel_fmm); flush(stdout)

    dcache = V._radix_fmm_couplings[dev].cache
    # both lifecycles hang a DeviceResidentRadixState off the cache, but only
    # assume it under MODE=ka -- the CUDA arm only needs the timing.
    dstate = hasproperty(dcache, :state) ? dcache.state : nothing
    nroutes = dstate === nothing ? -1 : Int(dstate.counts.n_routes)
    if dstate !== nothing
        @printf("  [%s]  ell=%d  n_cells=%d  n_direct=%d  n_routes=%d\n",
                MODE, dcache.ell, dstate.counts.n_cells, dstate.counts.n_direct,
                dstate.counts.n_routes); flush(stdout)
    else
        @printf("  [%s]  ell=%d\n", MODE, dcache.ell); flush(stdout)
    end

    tnf = (; min=NaN)
    td = nothing; rel_dir = NaN; frac = NaN
    if IS_KA && dstate !== nothing
        #--- how much all-pairs work did the FMM actually remove? --------------
        # The direct list is CELL pairs; what the nearfield kernel costs is the
        # INTERACTIONS they span. ninter/np^2 is the fraction of the all-pairs work
        # the FMM still does in its near field -- i.e. the part it did NOT remove.
        backend0 = KA.get_backend(dstate.output)
        rr = Array(dstate.cell_ranges)
        dt = Array(dstate.direct_targets); ds = Array(dstate.direct_sources)
        ninter = sum(Int(rr[2, dt[k]]) * Int(rr[2, ds[k]]) for k in 1:dstate.counts.n_direct)
        frac = ninter / float(NP)^2
        # nearfield stage ALONE, the production launcher
        nf! = () -> (ext.ka_launch_nearfield!(dstate); KA.synchronize(backend0))
        nf!(); nf!()
        tnf = timeit(nf!; calls=DIR_CALLS)
        @printf("  [ka]    direct interactions = %.4g  = %.2f%% of all-pairs (%.4g)\n",
                ninter, 100*frac, float(NP)^2)
        @printf("  ka_nf   %9.5f s min  = %.1f%% of ka_fmm  (rest = FMM overhead %.5f s)\n",
                tnf.min, 100*tnf.min/tg.min, tg.min - tnf.min); flush(stdout)

        #--- arm 3: KA all-pairs direct ----------------------------------------
        pairs = float(NP)^2
        td = nothing; rel_dir = NaN
        if pairs > MAX_PAIRS
            @printf("  ka_dir  skipped (%.3g pairs > MAX_PAIRS)\n", pairs)
        else
            dswitches = FM.DerivativesSwitch(FM.to_vector(false, 1), FM.to_vector(true, 1),
                                             FM.to_vector(true, 1), (dev,))
            backend = KA.get_backend(dstate.output)
            # score direct against the FMM in the SAME output buffer / body ordering
            ext.ka_radix_cache_device_step!(dcache, (dev,), dswitches); KA.synchronize(backend)
            out_fmm = Array(dstate.output)[2:4, :]
            direct! = make_direct_arm(dstate, 64, 64)
            direct!()
            out_dir = Array(dstate.output)[2:4, :]
            s = maximum(abs.(out_dir))
            rel_dir = s == 0 ? maximum(abs.(out_fmm .- out_dir)) :
                               maximum(abs.(out_fmm .- out_dir)) / s
            direct!(); direct!()                       # ramp
            td = timeit(direct!; calls=DIR_CALLS)
            @printf("  ka_dir  %9.5f s min  (ramp %s)   relerr(grad) vs ka_fmm = %.3e\n",
                    td.min, ramps(td.ramp), rel_dir); flush(stdout)
        end
    end

    #--- arm 3b: native CUDA all-pairs direct (MODE=cuda) -------------------
    @static if !HAS_METAL
    if !IS_KA && dstate !== nothing && float(NP)^2 <= MAX_PAIRS
        dswitches = FM.DerivativesSwitch(FM.to_vector(false, 1), FM.to_vector(true, 1),
                                         FM.to_vector(true, 1), (dev,))
        FM._radix_cache_device_step!(dcache, (dev,), dswitches); CUDA.synchronize()
        out_fmm = Array(dstate.output)[2:4, :]
        direct! = make_cuda_direct_arm(dstate)
        direct!()
        out_dir = Array(dstate.output)[2:4, :]
        sc = maximum(abs.(out_dir))
        rel_dir = sc == 0 ? maximum(abs.(out_fmm .- out_dir)) :
                            maximum(abs.(out_fmm .- out_dir)) / sc
        direct!(); direct!()
        td = timeit(direct!; calls=DIR_CALLS)
        @printf("  cu_dir  %9.5f s min  (ramp %s)   relerr(grad) vs cufmm = %.3e\n",
                td.min, ramps(td.ramp), rel_dir); flush(stdout)
    end
    end

    push!(SUMMARY, (; step, np=NP, ell=dcache.ell, threaded=NP >= FM.MIN_BODIES,
                    cpu=tc.min, kafmm=tg.min,
                    kadir=(td === nothing ? NaN : td.min),
                    nf=tnf.min, frac, nroutes,
                    rel_fmm, rel_dir))

    V.clear_radix_fmm_cache!(dev)
    delete!(V._radix_fmm_couplings, dev)
    GC.gc(); GC.gc()
    return nothing
end

for (s, n) in CASES
    run_case(s, n)
end

println("\n", "="^96)
println("LADDER SUMMARY   (times are min-of-N wall seconds for one full U/J evaluation)")
@printf("  %8s %4s | %10s %10s %10s | %9s | %9s %8s %8s %9s\n",
        "np", "ell", "cpu s", "ka_fmm s", "ka_dir s", "dir/kafmm",
        "ka_nf s", "nf%fmm", "near%n2", "routes")
for r in SUMMARY
    dstr = isnan(r.kadir) ? "        --" : @sprintf("%10.5f", r.kadir)
    sd   = isnan(r.kadir) ? "       --" : @sprintf("%8.2fx", r.kadir/r.kafmm)
    @printf("  %8d %4d | %10.5f %10.5f %s | %s | %9.5f %7.1f%% %7.1f%% %9d\n",
            r.np, r.ell, r.cpu, r.kafmm, dstr, sd,
            r.nf, 100*r.nf/r.kafmm, 100*r.frac, r.nroutes)
end
flush(stdout)
