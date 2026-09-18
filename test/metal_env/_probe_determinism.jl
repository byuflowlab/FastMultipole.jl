# Where does the run-to-run difference in U come from?
#
# Two candidate sources, both documented in the extension:
#   S1  the counting-sort refresh scatters with an atomic cursor, so WITHIN-CELL
#       body order varies run to run (ext comment "deliberately UNSTABLE" at
#       ka_counting_sort_into!), which reorders the same-cell nearfield sum.
#   S2  the nearfield / L2B / M2L scatters accumulate with KA.@atomic across
#       workgroups, so even at a FIXED body order the sum happens in whatever
#       order the hardware schedules.
#
# The discriminator does not need the (CUDA-only) counting-sort setting: build
# the device state ONCE, then
#   A  repeat the whole UJ_fmm            -> S1 + S2, and report whether the
#                                            device body order actually moved
#   B  repeat single stages on that state -> body order pinned, so S2 alone
#      B1 nearfield (pair shape, atomic accumulation)
#      B2 all-pairs direct (one workitem owns a target, plain stores) -- the
#         extension claims this shape is deterministic; this checks the claim
#      B3 B2M + M2M + M2L + L2L + L2B, the far-field chain
#
# Env: DEV_TF  CASE=wake|ring  NP  REPS

include("ka_backend.jl")
include("pipeline_field.jl")

using FastMultipole, Printf
const FM = FastMultipole
const V = FLOWVPM
import KernelAbstractions as KA

dev_functional() || (println("$(DEV_NAME) not functional; skipping"); exit(0))
ext = Base.get_extension(FastMultipole, :FastMultipoleKAExt)
ext === nothing && error("the KA extension is not loaded")

const DEV_TF = getfield(Base, Symbol(get(ENV, "DEV_TF", "Float32")))
const CASE   = Symbol(get(ENV, "CASE", "wake"))
const NP     = parse(Int, get(ENV, "NP", "15984"))
const REPS   = parse(Int, get(ENV, "REPS", "5"))
const P      = 5

fmm_settings() = V.FMM(; p=P+1, autotune_p=false, autotune_ncrit=false,
                         autotune_reg_error=false, default_rho_over_sigma=1.0)

function host_field()
    if CASE === :ring
        src = make_wake(NP; TF=Float64); n = V.get_np(src)
        pf = V.ParticleField(n, Float64; fmm=fmm_settings()); A = src.particles
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

"max|x_k - x_1| / max|x_1| over repeats of `f`, which must return a host array."
function spread(f, reps)
    a = f()
    s = maximum(abs, a); s = s == 0 ? one(s) : s
    d = [maximum(abs.(f() .- a)) / s for _ in 2:reps]
    return maximum(d), all(iszero, d)
end

println("=== determinism probe: $(DEV_NAME) $(DEV_TF) $(CASE) np=$(NP) reps=$(REPS) ===")

host = host_field(); n = V.get_np(host)
dev = V.ParticleField(host.maxparticles, DEV_TF; arraytype=devmatrix, np=host.np,
                      fmm=fmm_settings())
dev.particles .= devarray(DEV_TF.(Array(host.particles)))
V.radix_fmm_settings!(dev; m2l_strategy=:concat)
V.UJ_fmm(dev)                                     # build + warm
dcache = V._radix_fmm_couplings[dev].cache
dstate = dcache.state
backend = KA.get_backend(dstate.output)
dtargets = (dev,)
dswitches = FM.DerivativesSwitch(FM.to_vector(false, 1), FM.to_vector(true, 1),
                                 FM.to_vector(true, 1), dtargets)

# ---- A: the whole evaluation (S1 + S2) -------------------------------------
eu, _ = spread(REPS) do
    V._reset_particles(dev); V.UJ_fmm(dev)
    Float64.(Array(dev.particles)[V.U_INDEX, 1:n])
end
eo, order_stable = spread(REPS) do
    V._reset_particles(dev); V.UJ_fmm(dev)
    Float64.(Array(dstate.source_bodies)[1:3, 1:n])
end
@printf("A  full UJ_fmm            U spread %.3e    device body order identical: %s\n",
        eu, order_stable)

# ---- B: single stages on the state built above (body order pinned) ---------
V._reset_particles(dev); V.UJ_fmm(dev)            # settle one order, keep it

e1, _ = spread(REPS) do
    ext.ka_launch_nearfield!(dstate; workgroup=64, clear=true); KA.synchronize(backend)
    Float64.(Array(dstate.output)[2:4, :])
end
@printf("B1 nearfield (pair, atomic)          spread %.3e\n", e1)

# all-pairs shape: one workitem owns one target, plain stores
if float(n)^2 <= 1e10
    ng = cld(n, 64); CR = typeof(dstate.cell_ranges); IT = eltype(dstate.cell_ranges)
    h_cr = zeros(IT, 2, ng + 1)
    for g in 1:ng
        fb = (g - 1) * 64 + 1; h_cr[1, g] = fb; h_cr[2, g] = min(64, n - fb + 1)
    end
    h_cr[1, ng+1] = 1; h_cr[2, ng+1] = n
    cr = CR(undef, 2, ng + 1); copyto!(cr, h_cr)
    DT = typeof(dstate.direct_targets); JT = eltype(dstate.direct_targets)
    tg = DT(undef, ng); copyto!(tg, JT.(1:ng))
    sr = DT(undef, ng); copyto!(sr, fill(JT(ng + 1), ng))
    hs = size(dstate.output, 1) >= 13
    TF = eltype(dstate.output)
    kern = ext._cached_kernel(ext.ka_direct_pairs_functor_kernel!, backend, 64)
    dk = ext._ka_device_direct_kernel(dstate.options.direct_kernel, TF, 0)
    e2, _ = spread(REPS) do
        fill!(dstate.output, zero(TF))
        kern(dk, dstate.output, dstate.source_bodies, cr, tg, sr, ng, TF,
             Val(hs), Val(64); ndrange=ng * 64)
        KA.synchronize(backend)
        Float64.(Array(dstate.output)[2:4, :])
    end
    @printf("B2 all-pairs (same pair kernel)      spread %.3e\n", e2)
end

dws = dstate.scratch
e3, _ = spread(REPS) do
    fill!(dstate.output, zero(eltype(dstate.output)))
    ext.ka_launch_b2m!(dstate; workgroup=128)
    FM._zero_resident_nonleaf_multipoles!(dstate)
    for g in dws.m2m_groups
        ext.ka_resident_stage_group_apply!(dstate.multipoles, dstate.multipoles, g, dws, :m2m)
    end
    ext.ka_launch_m2l!(dstate, dws)
    for g in dws.l2l_groups
        ext.ka_resident_stage_group_apply!(dstate.locals, dstate.locals, g, dws, :l2l)
    end
    ext.ka_launch_l2b!(dstate; workgroup=64)
    KA.synchronize(backend)
    Float64.(Array(dstate.output)[2:4, :])
end
@printf("B3 far field B2M..L2B                spread %.3e\n", e3)

println()
println("body order moved between full runs: ", !order_stable)
println("with the order pinned, nearfield spread = ", e1, ", far field = ", e3)
