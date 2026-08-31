# P4.8 — fit the production-vs-dump INPUT difference (052d, 2026-08-29).
#
# Job 13510464 proved: device kernels are exact (1e-13) and a fresh cross pass
# on the dumped inputs is accurate (5e-5, ≤2.5e-5 at the worst particles), so
# the dumped deviceU/hostU were computed from inputs that differ from the
# dumped srcmat/wakemat. Model the difference as per-panel strength
# perturbations and per-station ruling scalings:
#   delta(t) ≈ Σ_k a_k B1_k(t) + Σ_k b_k B2_k(t) + Σ_st c_st B3_st(t)
#   B1_k: unit s1 (source-slot) field of panel k
#   B2_k: unit s2 (ring-slot) field of panel k INCLUDING its arm pair
#   B3_st: ∂(arm-pair field)/∂(ruling scale) at station st (FD)
# over the 400 nearest-TE targets (np=3544), for delta_hfm (host, cleaner) and
# delta_dev. Panels restricted to those within RCUT of any of the 13 worst
# particles. Reports residuals per basis combination and coefficient patterns.
#
# Run: JULIA_NUM_THREADS=4 DUMPDIR=<dir> \
#      julia --project=../../../../FLOWPanel.jl p48_input_delta_fit.jl

include(joinpath(@__DIR__, "CrossStencil.jl"))
using .CrossStencil
import FastMultipole
const FM = FastMultipole
using StaticArrays, Printf, LinearAlgebra, Statistics
using Base.Threads

const DUMPDIR = get(ENV, "DUMPDIR", "")
isempty(DUMPDIR) && error("set DUMPDIR")
const REG = 4
const WAKE_TAG = 3
const RCUT = 0.08

readmat(f, r) = collect(reshape(reinterpret(Float64, read(f)), r, :))

function panel_u(tag, nv, v1, v2, v3, v4, s1, s2, core, t)
    uq, _ = FM._rect_panel_pair(FM.RectangularPanelInfluence(), t,
        tag, nv, v1, v2, v3, v4, s1, s2, core, Val(false), Val(REG))
    return uq
end

np = 3544
pre = joinpath(DUMPDIR, "dump_np$np")
positions = readmat(pre * "_positions_3xN_f64.bin", 3)
deviceU = readmat(pre * "_deviceU_3xN_f64.bin", 3)
hostU = readmat(pre * "_hostU_3xN_f64.bin", 3)
srcmat = readmat(pre * "_body1_srcmat_17xS_f64.bin", 17)
cent = readmat(pre * "_body1_cent_3xS_f64.bin", 3)
wakemat = readmat(pre * "_body1_wakemat_8xS_f64.bin", 8)
ns = size(srcmat, 2)
particles = [SVector{3,Float64}(positions[:, i]) for i in 1:np]

verts(k) = (SVector(srcmat[3, k], srcmat[4, k], srcmat[5, k]),
            SVector(srcmat[6, k], srcmat[7, k], srcmat[8, k]),
            SVector(srcmat[9, k], srcmat[10, k], srcmat[11, k]))

struct St
    k::Int; w1::SVector{3,Float64}; w2::SVector{3,Float64}
    Da::SVector{3,Float64}; Db::SVector{3,Float64}; core::Float64
end
stmap = Dict{Int,St}()
for k in 1:ns
    idx1 = Int(wakemat[1, k]); idx1 > 0 || continue
    idx2 = Int(wakemat[5, k])
    vs = verts(k)
    stmap[k] = St(k, vs[idx1], vs[idx2],
        SVector(wakemat[2, k], wakemat[3, k], wakemat[4, k]),
        SVector(wakemat[6, k], wakemat[7, k], wakemat[8, k]), srcmat[17, k])
end

"arm pair field with ruling scaled by g, unit ring strength"
function armpair_u(st::St, g, t)
    v1w = st.w1 + g * st.Da; v2w = st.w2 + g * st.Db
    panel_u(WAKE_TAG, 3, st.w1, st.w2, v1w, v1w, 1.0, 0.0, st.core, t) +
        panel_u(WAKE_TAG, 3, v1w, st.w2, v2w, v2w, 1.0, 0.0, st.core, t)
end

# dense baseline over pool
shedc = [SVector{3,Float64}(cent[:, k]) for k in keys(stmap)]
dTE = [minimum(c -> norm(particles[i] - c), shedc) for i in 1:np]
pool = sortperm(dTE)[1:400]
tgts = [particles[i] for i in pool]
nt = length(tgts)
worst = [particles[i] for i in np-12:np]

Udense = zeros(3, nt)
@threads for s in 1:nt
    t = tgts[s]
    u = zero(SVector{3,Float64})
    for k in 1:ns
        v1, v2, v3 = verts(k)
        u += panel_u(Int(srcmat[1, k]), 3, v1, v2, v3, v3,
            srcmat[15, k], srcmat[16, k], srcmat[17, k], t)
    end
    for (k, st) in stmap
        u += srcmat[16, k] * armpair_u(st, 1.0, t)
    end
    Udense[:, s] .= u
end
ddev = vec(deviceU[:, pool] .- Udense)
dhfm = vec(hostU[:, pool] .- Udense)
@printf("pool=%d | ||ddev||=%.3e ||dhfm||=%.3e\n", nt, norm(ddev), norm(dhfm))

# candidate panels: strongest ring-slot influence at the worst particles
score = zeros(ns)
@threads for k in 1:ns
    v1, v2, v3 = verts(k)
    s = 0.0
    for w in worst
        s = max(s, norm(panel_u(4, 3, v1, v2, v3, v3, 0.0, 1.0, srcmat[17, k], w)))
    end
    score[k] = s
end
candk = sort(sortperm(score; rev=true)[1:300])
candst = [k for k in candk if haskey(stmap, k)]
@printf("candidates: %d panels (%d shedding), score range %.2e..%.2e\n",
    length(candk), length(candst), score[candk[argmin(score[candk])]],
    maximum(score))

# build bases
function build_cols(kind)
    ks = kind == :ruling ? candst : candk
    A = zeros(3nt, length(ks))
    @threads for s in 1:nt
        t = tgts[s]
        for (c, k) in enumerate(ks)
            v1, v2, v3 = verts(k)
            u = if kind == :src
                panel_u(4, 3, v1, v2, v3, v3, 1.0, 0.0, srcmat[17, k], t)
            elseif kind == :ring
                ur = panel_u(4, 3, v1, v2, v3, v3, 0.0, 1.0, srcmat[17, k], t)
                haskey(stmap, k) ? ur + armpair_u(stmap[k], 1.0, t) : ur
            else # :ruling — d(field)/d(scale) at station, scaled by mu
                st = stmap[k]
                srcmat[16, k] * (armpair_u(st, 1.05, t) - armpair_u(st, 0.95, t)) / 0.1
            end
            A[3s-2, c] = u[1]; A[3s-1, c] = u[2]; A[3s, c] = u[3]
        end
    end
    return ks, A
end

ks_src, Asrc = build_cols(:src)
ks_ring, Aring = build_cols(:ring)
ks_rul, Arul = build_cols(:ruling)

function fit(nm, A, d, ks)
    x = qr(A) \ d
    res = norm(A * x - d) / norm(d)
    @printf("  %-14s res=%.3f  med|x|=%.3e max|x|=%.3e", nm, res,
        median(abs.(x)), maximum(abs.(x)))
    if !isempty(ks)
        imax = argmax(abs.(x))
        @printf("  argmax panel=%d shed=%s s2=%.4g x=%.3e", ks[imax],
            haskey(stmap, ks[imax]), srcmat[16, ks[imax]], x[imax])
    end
    println()
    return res
end

for (dn, d) in (("delta_hfm", dhfm), ("delta_dev", ddev))
    println(dn, ":")
    fit("src-slot", Asrc, d, ks_src)
    fit("ring-slot", Aring, d, ks_ring)
    fit("ruling", Arul, d, ks_rul)
    fit("ring+ruling", hcat(Aring, Arul), d, Int[])
    fit("src+ring", hcat(Asrc, Aring), d, Int[])
    fit("all", hcat(Asrc, Aring, Arul), d, Int[])
end
println("DONE")
