# How much accuracy is on the table from the expansion order, versus the
# run-to-run accumulation floor?
#
# For each P: device FMM error against a FLOAT64 device all-pairs sum on the
# same bodies, plus the wall time.
#
# The reference MUST be Float64 whatever DEV_TF is. A same-precision all-pairs
# sum is an n-term Float32 sum whose own roundoff (1.8e-5 at 16k, 1.7e-4 at
# 249k, 2.1e-4 on the cancelling ring) reproduced every "accuracy plateau"
# reported on 2026-09-02 to four digits; the Float32 FMM itself reaches ~6e-7
# (job 13562395, _probe_floor_reference.jl). Each device state sorts bodies its
# own way, so outputs are mapped to original order by exact Float32 position.
#
# Env: DEV_TF  CASE=wake|ring  NP  PS=comma list

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
const NP     = parse(Int, get(ENV, "NP", "15984"))
const PS     = [parse(Int, s) for s in split(get(ENV, "PS", "3,4,5,6,7,8"), ",")]
const CALLS  = parse(Int, get(ENV, "CALLS", "8"))

settings(P) = V.FMM(; p=P+1, autotune_p=false, autotune_ncrit=false,
                      autotune_reg_error=false, default_rho_over_sigma=1.0)

function host_field(P)
    if CASE === :ring
        src = make_wake(NP; TF=Float64); n = V.get_np(src)
        pf = V.ParticleField(n, Float64; fmm=settings(P)); A = src.particles
        for i in 1:n
            V.add_particle(pf, (A[V.X_INDEX[1],i], A[V.X_INDEX[2],i], A[V.X_INDEX[3],i]),
                           (A[V.GAMMA_INDEX[1],i], A[V.GAMMA_INDEX[2],i], A[V.GAMMA_INDEX[3],i]),
                           A[V.SIGMA_INDEX,i]; vol=A[V.VOL_INDEX,i])
        end
        return pf
    end
    for s in (36, 72, 144, 288, 720)
        n_all = h5open(h -> length(read(h["sigma"])), wake_path(s))
        n_all >= NP && return load_wake(s; np=(n_all == NP ? nothing : NP), TF=Float64, P=P)
    end
    error("no wake dump holds $NP particles")
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

println("=== P sweep: $(DEV_NAME) $(DEV_TF) $(CASE) np=$(NP) ===")
println("error is the device FMM against the FLOAT64 device all-pairs sum on the same bodies")
@printf("  %3s | %10s | %10s %10s | %8s\n", "P", "time s", "max relerr", "L2 relerr", "cells")
println("  ", "-"^56)

function device_copy_of(h, TF, P)
    d = V.ParticleField(h.maxparticles, TF; arraytype=devmatrix, np=h.np, fmm=settings(P))
    d.particles .= devarray(TF.(Array(h.particles)))
    return d
end

# state-order -> original-order permutation, by exact Float32 position match
function state_perm(state, keys, n)
    Xs = Array(state.source_bodies)[1:3, 1:n]
    return [keys[(Float32(Xs[1,j]), Float32(Xs[2,j]), Float32(Xs[3,j]))] for j in 1:n]
end

function fmm_and_allpairs(dev, keys, n)
    dcache = V._radix_fmm_couplings[dev].cache; st = dcache.state
    dsw = FM.DerivativesSwitch(FM.to_vector(false, 1), FM.to_vector(true, 1),
                               FM.to_vector(true, 1), (dev,))
    ext.ka_radix_cache_device_step!(dcache, (dev,), dsw); KA.synchronize(KA.get_backend(st.output))
    perm = state_perm(st, keys, n)
    fmm = zeros(Float64, 3, n); fmm[:, perm] .= Float64.(Array(st.output)[2:4, 1:n])
    all_pairs_arm(st, 64, 64)()
    ap = zeros(Float64, 3, n); ap[:, perm] .= Float64.(Array(st.output)[2:4, 1:n])
    return fmm, ap, Int(st.counts.n_cells), dcache.ell
end

function drop!(dev)
    haskey(V._radix_fmm_couplings, dev) && V.clear_radix_fmm_cache!(dev)
    delete!(V._radix_fmm_couplings, dev); GC.gc()
end

# Float64 reference once per run
let host = host_field(PS[1])
    n = V.get_np(host)
    X = Array(host.particles)[V.X_INDEX, 1:n]
    global KEYS = Dict{NTuple{3,Float32},Int}()
    for i in 1:n
        k = (Float32(X[1,i]), Float32(X[2,i]), Float32(X[3,i]))
        haskey(KEYS, k) && error("duplicate Float32 position at bodies $(KEYS[k]) and $i")
        KEYS[k] = i
    end
    dev64 = device_copy_of(host, Float64, PS[1])
    V.radix_fmm_settings!(dev64; m2l_strategy=:concat, expansion_order=PS[1]); V.UJ_fmm(dev64)
    _, ref, _, _ = fmm_and_allpairs(dev64, KEYS, n)
    global REF64 = ref
    drop!(dev64)
end

for P in PS
    host = host_field(P); n = V.get_np(host)
    dev = device_copy_of(host, DEV_TF, P)
    V.radix_fmm_settings!(dev; m2l_strategy=:concat, expansion_order=P)
    V.UJ_fmm(dev)
    ts = Float64[]
    for _ in 1:CALLS
        V._reset_particles(dev); t0 = time_ns(); V.UJ_fmm(dev); push!(ts, (time_ns()-t0)/1e9)
    end
    fmm_out, _, n_cells, ell = fmm_and_allpairs(dev, KEYS, n)
    s = maximum(abs, REF64)
    @printf("  %3d | %10.5f | %10.3e %10.3e | %8d  [cache P=%d ell=%d, ref Float64]\n", P, median(ts),
            maximum(abs.(fmm_out .- REF64))/s,
            sqrt(sum(abs2, fmm_out .- REF64)/sum(abs2, REF64)), n_cells, P, ell)
    flush(stdout)
    drop!(dev)
end
