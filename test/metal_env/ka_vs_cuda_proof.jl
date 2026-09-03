# KA-vs-native-CUDA acceptance sweep: one process = one (MODE, DEV_TF, CASE).
#
# Arms, all on the SAME body ordering and scored against the threaded CPU
# Float64 FMM (`cpu`):
#   fmm    the device radix FMM: MODE=ka via the KA extension, MODE=cuda via
#          `load_cuda_radix_lifecycle!()` (the pure-CUDA branch).
#   dir    all-pairs direct with the same per-pair kernel: the production KA
#          pair kernel (MODE=ka) or a CUDA.jl mirror of it (MODE=cuda).
#
# Differences from pipeline_three_arm_sweep.jl (which this derives from):
#   * timings are MEDIAN with min/max over N calls after a warm-up, not min-of-N
#   * two cases: CASE=wake (the NREL wake dumps) and CASE=ring (synthetic vortex
#     rings from pipeline_field.jl's make_wake, no data files needed)
#   * the device FMM's U (particle order) is dumped to OUT as Float64 so that a
#     separate process can compute the KA-vs-CUDA difference ELEMENTWISE
#     (KA and CUDA lifecycles cannot share a process); direct is scored
#     elementwise against the fmm in the state's own buffer
#   * one CSV row per rung appended to OUT/proof.csv
#
# Env: MODE=ka|cuda  DEV_TF=Float32|Float64  CASE=wake|ring  OUT=<dir>
#      NP=<comma list of sizes>  (default 15984,62792,115455,248714 -- the
#      wake rung sizes, so the ring case matches them)  CALLS=12  WARM=3
#      MAX_PAIRS=1e11  RUN=<tag written into the CSV>

include("ka_backend.jl")
include("pipeline_field.jl")

using FastMultipole, Printf, Statistics
using FastMultipole.StaticArrays
const FM = FastMultipole
const V = FLOWVPM
import KernelAbstractions as KA

dev_functional() || (println("$(DEV_NAME) not functional; skipping"); exit(0))
ext = Base.get_extension(FastMultipole, :FastMultipoleKAExt)
ext === nothing && error("the KA extension is not loaded")

const MODE = Symbol(get(ENV, "MODE", "ka"))
MODE in (:ka, :cuda) || error("MODE must be ka or cuda, got $(MODE)")
const IS_KA = MODE === :ka
IS_KA || error("MODE=cuda: the native CUDA lifecycle was removed after the KA port " *
               "reached parity with it (H200 job 13563393); only MODE=ka remains")
const CASE      = Symbol(get(ENV, "CASE", "wake"))
CASE in (:wake, :ring) || error("CASE must be wake or ring")
const DEV_TF    = getfield(Base, Symbol(get(ENV, "DEV_TF", "Float32")))
const OUT       = get(ENV, "OUT", joinpath(@__DIR__, "logs", "proof"))
const RUN       = get(ENV, "RUN", "1")
const P         = 5
const STRATEGY  = :concat
const CALLS     = parse(Int, get(ENV, "CALLS", "12"))
const WARM      = parse(Int, get(ENV, "WARM", "3"))
const CPU_CALLS = 5
const MAX_PAIRS = parse(Float64, get(ENV, "MAX_PAIRS", "1e11"))
const NPS = [parse(Int, s) for s in split(get(ENV, "NP", "15984,62792,115455,248714"), ",")]
# wake dumps: which step holds (at least) np particles; smallest step first
const WAKE_STEPS = [36, 72, 144, 288, 720]
mkpath(OUT)

function timeit(f; calls::Int, warm::Int)
    for _ in 1:warm; f(); end
    ts = Float64[]
    for _ in 1:calls
        GC.gc(); t0 = time_ns(); f(); push!(ts, (time_ns() - t0)/1e9)
    end
    return (min=minimum(ts), med=median(ts), max=maximum(ts), ramp=ts)
end
ramps(ts) = join((@sprintf("%.4f", t) for t in ts), "/")

fmm_settings() = V.FMM(; p=P+1, autotune_p=false, autotune_ncrit=false,
                         autotune_reg_error=false, default_rho_over_sigma=1.0)

"Synthetic ring train with the radix-compatible FMM settings (make_wake lacks the kwarg)."
function make_ring(np::Int; TF=Float64)
    src = make_wake(np; TF=TF)
    n = V.get_np(src)
    pf = V.ParticleField(n, TF; fmm=fmm_settings())
    A = src.particles
    for i in 1:n
        V.add_particle(pf, (A[V.X_INDEX[1],i], A[V.X_INDEX[2],i], A[V.X_INDEX[3],i]),
                       (A[V.GAMMA_INDEX[1],i], A[V.GAMMA_INDEX[2],i], A[V.GAMMA_INDEX[3],i]),
                       A[V.SIGMA_INDEX,i]; vol=A[V.VOL_INDEX,i])
    end
    return pf
end

function load_host(np::Int)
    if CASE === :ring
        return make_ring(np)
    end
    for s in WAKE_STEPS
        n_all = h5open(h -> length(read(h["sigma"])), wake_path(s))
        n_all >= np && return load_wake(s; np=(n_all == np ? nothing : np), TF=Float64, P=P)
    end
    error("no wake dump holds $np particles")
end

function device_copy_of(h)
    d = V.ParticleField(h.maxparticles, DEV_TF; arraytype=devmatrix, np=h.np,
                        fmm=fmm_settings())
    d.particles .= devarray(DEV_TF.(Array(h.particles)))
    return d
end

# ---- native CUDA all-pairs arm (MODE=cuda), mirror of the KA pair kernel ----
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

# ---- KA all-pairs arm: the production pair kernel over an all-pairs list ----
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
    h_cr[1, ng+1] = 1; h_cr[2, ng+1] = n
    cr = CR(undef, 2, ng + 1); copyto!(cr, h_cr)
    DT = typeof(state.direct_targets); JT = eltype(state.direct_targets)
    tg = DT(undef, ng); copyto!(tg, JT.(1:ng))
    sr = DT(undef, ng); copyto!(sr, fill(JT(ng + 1), ng))
    hs = size(state.output, 1) >= 13
    backend = KA.get_backend(state.output)
    kern = ext._cached_kernel(ext.ka_direct_pairs_functor_kernel!, backend, workgroup)
    dkernel = ext._ka_device_direct_kernel(state.options.direct_kernel, TF, 0)
    return function ()
        fill!(state.output, zero(TF))
        kern(dkernel, state.output, state.source_bodies,
             cr, tg, sr, ng, TF, Val(hs), Val(workgroup); ndrange=ng * workgroup)
        KA.synchronize(backend)
        return nothing
    end
end

relmax(a, b) = maximum(abs.(a .- b)) / maximum(abs.(b))
rell2(a, b)  = sqrt(sum(abs2, a .- b) / sum(abs2, b))
dump(name, A) = open(io -> write(io, Float64.(A)), joinpath(OUT, name * ".bin"), "w")

const CSV = joinpath(OUT, "proof.csv")
isfile(CSV) || open(io -> println(io, "run,mode,tf,case,np,ell,ncells,ndirect,nroutes," *
    "cpu_med,fmm_min,fmm_med,fmm_max,dir_min,dir_med,dir_max," *
    "fmm_vs_cpu_max,fmm_vs_cpu_l2,dir_vs_fmm_max,jac_vs_cpu_max,jac_vs_cpu_l2"), CSV, "w")

println("=== KA-vs-CUDA proof: MODE=$(MODE) DEV_TF=$(DEV_TF) CASE=$(CASE) RUN=$(RUN) ===")
println("device = ", DEV_NAME, "  julia threads = ", Threads.nthreads(),
        "  P = ", P, "  calls = ", CALLS, " (+", WARM, " warm)")
flush(stdout)

function run_case(np::Int)
    host = load_host(np)
    NP = V.get_np(host)
    tag = "$(CASE)_$(NP)_$(MODE)_$(DEV_TF)_run$(RUN)"
    println("\n", "="^88); @printf("%s  np = %d\n", CASE, NP); flush(stdout)

    #--- CPU reference (Float64, threaded) ---------------------------------
    V.UJ_fmm(host; autotune=false); V._reset_particles(host)
    tc = timeit(() -> (V._reset_particles(host); V.UJ_fmm(host; autotune=false));
                calls=CPU_CALLS, warm=1)
    U_ref = copy(Array(host.particles)[V.U_INDEX, 1:NP])
    J_ref = copy(Array(host.particles)[V.J_INDEX, 1:NP])   # velocity gradient (9 rows)
    RUN == "1" && MODE === :ka && DEV_TF === Float32 && dump("U_$(CASE)_$(NP)_cpu", U_ref)
    @printf("  cpu   med %9.5f s  (min %.5f max %.5f)\n", tc.med, tc.min, tc.max); flush(stdout)

    #--- device FMM ----------------------------------------------------------
    dev = device_copy_of(host)
    V.radix_fmm_settings!(dev; m2l_strategy=STRATEGY)
    V.UJ_fmm(dev)
    tg = timeit(() -> (V._reset_particles(dev); V.UJ_fmm(dev)); calls=CALLS, warm=WARM)
    U_dev = Float64.(Array(dev.particles)[V.U_INDEX, 1:NP])
    J_dev = Float64.(Array(dev.particles)[V.J_INDEX, 1:NP])
    dump("U_$(tag)_fmm", U_dev)
    e_fmm = (relmax(U_dev, U_ref), rell2(U_dev, U_ref))
    e_jac = (relmax(J_dev, J_ref), rell2(J_dev, J_ref))
    @printf("  fmm   med %9.5f s  (min %.5f max %.5f)  relerr vs cpu max %.3e L2 %.3e\n",
            tg.med, tg.min, tg.max, e_fmm...)
    @printf("  jac   relerr vs cpu max %.3e L2 %.3e   (velocity gradient, 9 rows)\n", e_jac...)
    println("        ramp ", ramps(tg.ramp)); flush(stdout)

    dcache = V._radix_fmm_couplings[dev].cache
    dstate = hasproperty(dcache, :state) ? dcache.state : nothing
    counts = dstate === nothing ? (n_cells=-1, n_direct=-1, n_routes=-1) :
        (n_cells=Int(dstate.counts.n_cells), n_direct=Int(dstate.counts.n_direct),
         n_routes=Int(dstate.counts.n_routes))
    @printf("  [%s] ell=%d cells=%d direct=%d routes=%d\n", MODE, dcache.ell,
            counts.n_cells, counts.n_direct, counts.n_routes); flush(stdout)

    #--- all-pairs direct, scored against cpu AND against the fmm ------------
    td = (min=NaN, med=NaN, max=NaN); dir_vs_fmm = NaN
    if dstate !== nothing && float(NP)^2 <= MAX_PAIRS
        dswitches = FM.DerivativesSwitch(FM.to_vector(false, 1), FM.to_vector(true, 1),
                                         FM.to_vector(true, 1), (dev,))
        if IS_KA
            backend = KA.get_backend(dstate.output)
            ext.ka_radix_cache_device_step!(dcache, (dev,), dswitches); KA.synchronize(backend)
            direct! = make_direct_arm(dstate, 64, 64)
        else
            @static if !HAS_METAL
                FM._radix_cache_device_step!(dcache, (dev,), dswitches); CUDA.synchronize()
                direct! = make_cuda_direct_arm(dstate)
            end
        end
        out_fmm = Float64.(Array(dstate.output)[2:4, :])
        direct!()
        out_dir = Float64.(Array(dstate.output)[2:4, :])
        dir_vs_fmm = relmax(out_fmm, out_dir)
        # state output is in DEVICE body order (not the particle order of U_ref),
        # so direct is scored elementwise against the fmm in the same buffer;
        # the chain cpu <-> fmm <-> dir closes the comparison.
        td = timeit(direct!; calls=CALLS, warm=WARM)
        @printf("  dir   med %9.5f s  (min %.5f max %.5f)  relerr(grad) vs fmm %.3e\n",
                td.med, td.min, td.max, dir_vs_fmm)
        println("        ramp ", ramps(td.ramp)); flush(stdout)
    end

    open(CSV, "a") do io
        @printf(io, "%s,%s,%s,%s,%d,%d,%d,%d,%d,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.4e,%.4e,%.4e,%.4e,%.4e\n",
                RUN, MODE, DEV_TF, CASE, NP, dcache.ell, counts.n_cells, counts.n_direct,
                counts.n_routes, tc.med, tg.min, tg.med, tg.max, td.min, td.med, td.max,
                e_fmm..., dir_vs_fmm, e_jac...)
    end
    V.clear_radix_fmm_cache!(dev)
    delete!(V._radix_fmm_couplings, dev)
    GC.gc(); GC.gc()
    return nothing
end

for np in NPS
    run_case(np)
end
println("\n=== done: $(CSV) ===")
