# P4.4 — least-squares attribution of the production-vs-dense delta onto
# per-shedding-station field bases (052d, 2026-08-29).
#
# delta_dev(t) = deviceU(t) - dense(t) and delta_hfm(t) analogously, over the
# 400 nearest-TE particles (np=3544 dump). Fit delta ≈ A x where A's columns
# are per-station unit fields under three candidate mechanisms:
#   B1 arm-pair:   unit-strength attached wake pair (mu=1)  -> x = strength
#                  perturbation per station (e.g. wake_strength_shift)
#   B2 TE filament: unit-circulation LineGauss segment w1->w2 (the bound TE
#                  vortex)                                   -> x = Gamma_TE
#   B3 outer filament: unit-circulation segment v1w->v2w (free wake row edge)
# Reports residual reduction, coefficient stats, and x_dev vs x_hfm agreement.
#
# Run: JULIA_NUM_THREADS=4 DUMPDIR=<dir> \
#      julia --project=../../../../FLOWPanel.jl p44_basis_fit.jl

include(joinpath(@__DIR__, "CrossStencil.jl"))
using .CrossStencil
import FastMultipole
const FM = FastMultipole
import FLOWPanel
using StaticArrays, Random, Statistics, Printf, LinearAlgebra
using Base.Threads

const DUMPDIR = get(ENV, "DUMPDIR", "")
isempty(DUMPDIR) && error("set DUMPDIR")
const REG = 4
const WAKE_TAG = 3

readmat(f, r) = collect(reshape(reinterpret(Float64, read(f)), r, :))

@inline function colverts(E, j)
    (SVector(E[3, j], E[4, j], E[5, j]), SVector(E[6, j], E[7, j], E[8, j]),
     SVector(E[9, j], E[10, j], E[11, j]), SVector(E[12, j], E[13, j], E[14, j]))
end

function panel_u(tag, nv, v1, v2, v3, v4, s1, s2, core, t)
    uq, _ = FM._rect_panel_pair(FM.RectangularPanelInfluence(), t,
        tag, nv, v1, v2, v3, v4, s1, s2, core, Val(false), Val(REG))
    return uq
end

"unit-circulation LineGauss filament velocity at t for segment a->b"
filament_u(a, b, core, t) =
    FLOWPanel._bound_vortex_velocity(t - a, t - b, true, core,
        Val(FLOWPanel.LineGaussRegularization))

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

# stations
struct Station
    k::Int
    w1::SVector{3,Float64}
    w2::SVector{3,Float64}
    v1w::SVector{3,Float64}
    v2w::SVector{3,Float64}
    mu::Float64
    core::Float64
end
stations = Station[]
for k in 1:ns
    idx1 = Int(wakemat[1, k])
    idx1 > 0 || continue
    idx2 = Int(wakemat[5, k])
    tag = Int(srcmat[1, k])
    vs = (SVector(srcmat[3, k], srcmat[4, k], srcmat[5, k]),
          SVector(srcmat[6, k], srcmat[7, k], srcmat[8, k]),
          SVector(srcmat[9, k], srcmat[10, k], srcmat[11, k]))
    w1 = vs[idx1]; w2 = vs[idx2]
    v1w = w1 + SVector(wakemat[2, k], wakemat[3, k], wakemat[4, k])
    v2w = w2 + SVector(wakemat[6, k], wakemat[7, k], wakemat[8, k])
    mu = (tag == 2 || tag == 3) ? srcmat[15, k] : srcmat[16, k]
    push!(stations, Station(k, w1, w2, v1w, v2w, mu, srcmat[17, k]))
end
nst = length(stations)

# pool: 400 nearest-TE particles
shedc = [SVector{3,Float64}(cent[:, st.k]) for st in stations]
dTE = [minimum(c -> norm(particles[i] - c), shedc) for i in 1:np]
pool = sortperm(dTE)[1:400]
tgts = [particles[i] for i in pool]
nt = length(tgts)

# dense baseline at pool (all columns incl arms)
function expand_columns(srcmat, wakemat)
    nshed = count(k -> wakemat[1, k] > 0, 1:ns)
    E = zeros(17, ns + 2 * nshed)
    E[:, 1:ns] .= srcmat
    col = ns
    for st in stations
        for (a, b, c) in ((st.w1, st.w2, st.v1w), (st.v1w, st.w2, st.v2w))
            col += 1
            E[1, col] = WAKE_TAG; E[2, col] = 3
            E[3:5, col] .= a; E[6:8, col] .= b; E[9:11, col] .= c
            E[12:14, col] .= c
            E[15, col] = st.mu; E[16, col] = 0.0; E[17, col] = st.core
        end
    end
    return E
end
E = expand_columns(srcmat, wakemat)
Udense = zeros(3, nt)
@threads for s in 1:nt
    u = zero(SVector{3,Float64})
    @inbounds for j in 1:size(E, 2)
        tag = Int(E[1, j]); nv = Int(E[2, j])
        (1 <= tag <= 5 && nv >= 3) || continue
        v1, v2, v3, v4 = colverts(E, j)
        u += panel_u(tag, nv, v1, v2, v3, v4, E[15, j], E[16, j], E[17, j], tgts[s])
    end
    Udense[:, s] .= u
end
ddev = vec(deviceU[:, pool] .- Udense)
dhfm = vec(hostU[:, pool] .- Udense)
@printf("pool: %d targets, %d stations | ||ddev||=%.3e ||dhfm||=%.3e (relative to ||U||=%.3e)\n",
    nt, nst, norm(ddev), norm(dhfm), norm(Udense))

# bases
function basis(kind)
    A = zeros(3 * nt, nst)
    @threads for s in 1:nt
        t = tgts[s]
        for (c, st) in enumerate(stations)
            u = if kind == :armpair
                panel_u(WAKE_TAG, 3, st.w1, st.w2, st.v1w, st.v1w, 1.0, 0.0, st.core, t) +
                panel_u(WAKE_TAG, 3, st.v1w, st.w2, st.v2w, st.v2w, 1.0, 0.0, st.core, t)
            elseif kind == :tefil
                filament_u(st.w1, st.w2, st.core, t)
            else # :outerfil
                filament_u(st.v1w, st.v2w, st.core, t)
            end
            A[3s-2, c] = u[1]; A[3s-1, c] = u[2]; A[3s, c] = u[3]
        end
    end
    return A
end

mus = [st.mu for st in stations]
for kind in (:armpair, :tefil, :outerfil)
    A = basis(kind)
    F = qr(A)
    for (nm, d) in (("dev", ddev), ("hfm", dhfm))
        x = F \ d
        res = norm(A * x - d) / norm(d)
        # coefficient scale vs mu
        ratio = x ./ mus
        @printf("%-9s %-4s | res=%.3f  ||x||=%.3e  med|x|=%.3e  med(x/mu)=%+.4f  iqr(x/mu)=[%+.4f,%+.4f]\n",
            String(kind), nm, res, norm(x), median(abs.(x)),
            median(ratio), quantile(ratio, 0.25), quantile(ratio, 0.75))
    end
    # dev vs hfm coefficient agreement
    xd = F \ ddev; xh = F \ dhfm
    @printf("%-9s      | corr(x_dev, x_hfm)=%.3f  ||x_dev - x_hfm||/||x_dev||=%.3f\n",
        String(kind), cor(xd, xh), norm(xd - xh) / max(norm(xd), eps()))
end
println("DONE")
