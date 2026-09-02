# Does the Float32 "accuracy floor" belong to the FMM or to the reference?
#
# Every floor so far (jobs 13562137/204/289/379) was device Float32 FMM vs the
# device Float32 ALL-PAIRS sum. That reference is itself an n-term Float32 sum
# whose roundoff grows with n and with cancellation, and the floor turned out
# to be independent of tree depth (job 13562379) -- the signature of an error
# that lives outside the tree. Discriminate by measuring against a Float64
# all-pairs sum on the same bodies (Float64 ordering noise is ~1e-16):
#   ref32 vs ref64          the reference's own error
#   FMM32(P) vs ref64       the FMM's true error
#   FMM32(P) vs ref32       should reproduce the old plateau
#   FMM64(P_max) vs ref64   mapping sanity check (expect ~1e-8)
# Outputs are mapped back to ORIGINAL body order by matching Float32 positions,
# since each device state sorts bodies its own way.
#
# Env: CASE=wake|ring  NPS  PS

include("ka_backend.jl")
include("pipeline_field.jl")
using FastMultipole, Printf, Statistics
const FM = FastMultipole
const V = FLOWVPM
import KernelAbstractions as KA

dev_functional() || (println("$(DEV_NAME) not functional; skipping"); exit(0))
ext = Base.get_extension(FastMultipole, :FastMultipoleKAExt)
ext === nothing && error("the KA extension is not loaded")

const CASE = Symbol(get(ENV, "CASE", "ring"))
const NPS  = [parse(Int, s) for s in split(get(ENV, "NPS", "63936"), ",")]
const PS   = [parse(Int, s) for s in split(get(ENV, "PS", "4,6,8,10"), ",")]

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

function device_copy_of(h, TF, P)
    d = V.ParticleField(h.maxparticles, TF; arraytype=devmatrix, np=h.np, fmm=fmm_settings(P))
    d.particles .= devarray(TF.(Array(h.particles)))
    return d
end

function all_pairs!(state, chunk, wg)
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
    fill!(state.output, zero(TF))
    kern(dk, state.output, state.source_bodies, cr, tg, sr, ng, TF, Val(hs), Val(wg); ndrange=ng * wg)
    KA.synchronize(b)
    return nothing
end

# state-order -> original-order permutation, by exact Float32 position match
function state_perm(state, key_of_original::Dict{NTuple{3,Float32},Int}, n)
    Xs = Array(state.source_bodies)[1:3, 1:n]
    perm = Vector{Int}(undef, n)
    for j in 1:n
        k = (Float32(Xs[1, j]), Float32(Xs[2, j]), Float32(Xs[3, j]))
        perm[j] = key_of_original[k]   # throws if a position fails to match
    end
    return perm
end

# run the device FMM step on a live coupling, return (fmm U, all-pairs U) in original order
function fmm_and_allpairs(dev, keys, n)
    dcache = V._radix_fmm_couplings[dev].cache; st = dcache.state
    dsw = FM.DerivativesSwitch(FM.to_vector(false, 1), FM.to_vector(true, 1),
                               FM.to_vector(true, 1), (dev,))
    ext.ka_radix_cache_device_step!(dcache, (dev,), dsw); KA.synchronize(KA.get_backend(st.output))
    perm = state_perm(st, keys, n)
    fmm = zeros(Float64, 3, n); fmm[:, perm] .= Float64.(Array(st.output)[2:4, 1:n])
    all_pairs!(st, 64, 64)
    ap = zeros(Float64, 3, n); ap[:, perm] .= Float64.(Array(st.output)[2:4, 1:n])
    return fmm, ap
end

relmax(a, b) = maximum(abs.(a .- b)) / maximum(abs.(b))
rell2(a, b)  = sqrt(sum(abs2, a .- b) / sum(abs2, b))

function drop!(dev)
    haskey(V._radix_fmm_couplings, dev) && V.clear_radix_fmm_cache!(dev)
    delete!(V._radix_fmm_couplings, dev); GC.gc(); GC.gc()
end

println("=== floor attribution: $(DEV_NAME) $(CASE) ===")
for np in NPS
    host = host_field(np, PS[end]); n = V.get_np(host)
    X = Array(host.particles)[V.X_INDEX, 1:n]
    keys = Dict{NTuple{3,Float32},Int}()
    for i in 1:n
        k = (Float32(X[1,i]), Float32(X[2,i]), Float32(X[3,i]))
        haskey(keys, k) && error("duplicate Float32 position at bodies $(keys[k]) and $i")
        keys[k] = i
    end
    println("-"^80); @printf("np=%d\n", n)

    # Float64 device: reference + a sanity FMM at the largest P
    dev64 = device_copy_of(host, Float64, PS[end])
    V.radix_fmm_settings!(dev64; m2l_strategy=:concat, expansion_order=PS[end]); V.UJ_fmm(dev64)
    fmm64, ref64 = fmm_and_allpairs(dev64, keys, n)
    @printf("  FMM64(P=%d) vs ref64 (mapping check): max %.3e  L2 %.3e\n", PS[end],
            relmax(fmm64, ref64), rell2(fmm64, ref64))
    drop!(dev64)

    # Float32 reference's own error
    dev32 = device_copy_of(host, Float32, PS[1])
    V.radix_fmm_settings!(dev32; m2l_strategy=:concat, expansion_order=PS[1]); V.UJ_fmm(dev32)
    _, ref32 = fmm_and_allpairs(dev32, keys, n)
    drop!(dev32)
    @printf("  ref32 vs ref64 (the Float32 reference's own error): max %.3e  L2 %.3e\n",
            relmax(ref32, ref64), rell2(ref32, ref64))
    flush(stdout)

    @printf("  %3s | %10s %10s | %10s %10s\n", "P", "vs64 max", "vs64 L2", "vs32 max", "vs32 L2")
    for P in PS
        h = host_field(np, P); dev = device_copy_of(h, Float32, P)
        V.radix_fmm_settings!(dev; m2l_strategy=:concat, expansion_order=P); V.UJ_fmm(dev)
        fmm32, _ = fmm_and_allpairs(dev, keys, n)
        @printf("  %3d | %10.3e %10.3e | %10.3e %10.3e\n", P,
                relmax(fmm32, ref64), rell2(fmm32, ref64), relmax(fmm32, ref32), rell2(fmm32, ref32))
        flush(stdout); drop!(dev)
    end
end
println("=== done ===")
