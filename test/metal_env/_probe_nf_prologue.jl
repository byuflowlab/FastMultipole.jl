# One-variable ablation: does the per-(pair,target) PROLOGUE explain why the
# FMM nearfield costs more per interaction than the all-pairs direct arm?
#
# The pair kernel pays, once per (pair, target body): 3 position loads, 13
# accumulator zeros, and 13 atomic stores. The all-pairs arm pays that once per
# TARGET. At np=15984 that is ~9.8e5 target-visits vs 15984 -- 62x -- amortized
# over an inner loop of ~124 sources instead of 15984.
#
# Everything else is held FIXED. Over n bodies, build T target cells x S source
# cells, all cells sized so that:
#
#   true interactions  = n*n            (every pair, every configuration)
#   lane-slots issued  = n*n            (cell sizes are multiples of WG: no padding)
#   workgroups         = T*S = G        (constant: same GPU occupancy)
#   target visits      = n*S            (THE ONLY VARIABLE, swept 1x .. 256x)
#
# So a flat time vs S kills the hypothesis; a rising time prices the prologue
# directly, in ns per target-visit.
include("ka_backend.jl")
include("pipeline_field.jl")
using FastMultipole, Printf, Statistics
const FM = FastMultipole
const V = FLOWVPM
import KernelAbstractions as KA
dev_functional() || (println("$(DEV_NAME) not functional; skipping"); exit(0))
const ext = Base.get_extension(FastMultipole, :FastMultipoleKAExt)

const WG    = 64
const P     = 5
const N     = 12288          # 2^12*3: every T,S split below lands on a WG multiple
const G     = 192            # workgroups, held constant
const CALLS = 12

function device_copy_of(h)
    d = V.ParticleField(h.maxparticles, Float32; arraytype=devmatrix, np=h.np,
        fmm=V.FMM(; p=P+1, autotune_p=false, autotune_ncrit=false,
                    autotune_reg_error=false, default_rho_over_sigma=1.0))
    d.particles .= devarray(Float32.(Array(h.particles)))
    return d
end
function timeit(f; calls::Int)
    ts = Float64[]
    for _ in 1:calls; t = time(); f(); push!(ts, time() - t); end
    (min=minimum(ts), ramp=ts)
end

# real wake bodies, so the physics (and its branch behaviour) is the real one
host = load_wake(36; np=N, TF=Float64, P=P)
dev  = device_copy_of(host)
V.radix_fmm_settings!(dev; m2l_strategy=:concat)
V.UJ_fmm(dev)
st = V._radix_fmm_couplings[dev].cache.state
n  = Int(V.get_np(dev))
n == N || @warn "np=$n != N=$N"

TF      = eltype(st.output)
rows    = size(st.output, 1) >= 13 ? 13 : 4
hs      = rows >= 13
backend = KA.get_backend(st.output)
kern    = ext._cached_kernel(ext.ka_direct_pairs_functor_kernel!, backend, WG)
dkernel = ext._ka_device_direct_kernel(st.options.direct_kernel, TF, 0)
CR = typeof(st.cell_ranges); DT = typeof(st.direct_targets); JT = eltype(st.direct_targets)

# S source cells x T target cells, T*S = G, target cell = n/T, source cell = n/S
function build(S::Int)
    T = G ÷ S
    tsz = n ÷ T; ssz = n ÷ S
    (tsz % WG == 0 && ssz % WG == 0) || return nothing
    h = zeros(Int, 2, T + S)
    for t in 1:T; h[1, t] = (t-1)*tsz + 1; h[2, t] = tsz; end
    for s in 1:S; h[1, T+s] = (s-1)*ssz + 1; h[2, T+s] = ssz; end
    cr = CR(undef, 2, T + S); copyto!(cr, h)
    tg = Vector{JT}(undef, T*S); sr = Vector{JT}(undef, T*S)
    k = 0
    for t in 1:T, s in 1:S; k += 1; tg[k] = t; sr[k] = T + s; end
    dtg = DT(undef, T*S); copyto!(dtg, tg)
    dsr = DT(undef, T*S); copyto!(dsr, sr)
    npair = T*S
    run! = function ()
        fill!(st.output, zero(TF))
        kern(dkernel, st.output, st.source_bodies, cr, dtg, dsr,
             npair, TF, Val(hs), Val(WG); ndrange = npair * WG)
        KA.synchronize(backend)
    end
    return (; T, tsz, ssz, npair, run!)
end

@printf("n=%d  WG=%d  workgroups G=%d  rows=%d  device=%s\n", n, WG, G, rows, DEV_NAME)
@printf("held fixed: interactions=%.4g  lane-slots=%.4g  workgroups=%d\n\n",
        float(n)^2, float(n)^2, G)
@printf("%4s %5s %7s %7s %9s %12s %12s\n",
        "S", "T", "tcell", "scell", "visits", "time_ms", "ns/visit_vs_S1")
base_t = NaN; base_v = NaN
for S in [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 192]
    b = build(S); b === nothing && continue
    b.run!(); b.run!(); b.run!()                 # ramp
    t = timeit(b.run!; calls=CALLS)
    visits = float(n) * S
    if S == 1; global base_t = t.min; global base_v = visits; end
    slope = (t.min - base_t) / (visits - base_v)
    @printf("%4d %5d %7d %7d %9.4g %12.4f %12s\n", S, b.T, b.tsz, b.ssz, visits,
            1e3*t.min, isnan(slope) || S == 1 ? "--" : @sprintf("%.4f", 1e9*slope))
    flush(stdout)
end
