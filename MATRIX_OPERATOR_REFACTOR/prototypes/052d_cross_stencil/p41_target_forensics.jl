# P4.1 — per-target forensics on the worst relU targets (052d, 2026-08-29).
#
# p40 showed the device-vs-dense deviation on dumped production states is
# carried by ~10 targets (50% of ||rdev||^2 at np=3544, 79% at np=12776),
# concentrated near shedding panels, with dev and hfmm agreeing with each
# other better than with dense there. This script prints, for the top-12
# targets by |deviceU - denseU| at each dumped np:
#   - geometry: distance to nearest shedding-panel centroid, nearest TE edge
#     segment (w1-w2), nearest wake arm filament segment (w1-v1w, w2-v2w,
#     v1w-v2w), nearest panel vertex, nearest OTHER particle;
#   - fields: |U| dense total / body-only / arms-only, and the deltas
#     |dev-dns|, |hfmm-dns|, |dev-hfmm| (magnitudes).
# If these targets sit ON the wake sheet (dist << core_size=1e-3 to an arm),
# the discrepancy is an on-sheet evaluation semantics difference, not FMM
# truncation.
#
# Run: JULIA_NUM_THREADS=4 DUMPDIR=<dir> \
#      julia --project=../../../../FLOWPanel.jl p41_target_forensics.jl

include(joinpath(@__DIR__, "CrossStencil.jl"))
using .CrossStencil
import FastMultipole
const FM = FastMultipole
using StaticArrays, Random, Statistics, Printf, LinearAlgebra
using Base.Threads

const DUMPDIR = get(ENV, "DUMPDIR", "")
isempty(DUMPDIR) && error("set DUMPDIR")
const REG = 4
const WAKE_TAG = 3

readmat(f, r) = collect(reshape(reinterpret(Float64, read(f)), r, :))

# ---- helpers copied VERBATIM from p39 (validated) ----
function expand_columns(srcmat, wakemat, cent)
    ns = size(srcmat, 2)
    nshed = wakemat === nothing ? 0 :
        count(k -> wakemat[1, k] > 0, 1:ns)
    E = zeros(17, ns + 2 * nshed)
    E[:, 1:ns] .= srcmat
    sortpos = Vector{SVector{3,Float64}}(undef, ns + 2 * nshed)
    for k in 1:ns
        sortpos[k] = SVector(cent[1, k], cent[2, k], cent[3, k])
    end
    col = ns
    wakemat === nothing && return E, sortpos
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
        koff = srcmat[17, k]
        for (a, b, c) in ((w1, w2, v1w), (v1w, w2, v2w))
            col += 1
            E[1, col] = WAKE_TAG; E[2, col] = 3
            E[3:5, col] .= a; E[6:8, col] .= b; E[9:11, col] .= c
            E[12:14, col] .= c
            E[15, col] = mu; E[16, col] = 0.0; E[17, col] = koff
            sortpos[col] = sortpos[k]
        end
    end
    return E, sortpos
end

@inline function colverts(E, j)
    (SVector(E[3, j], E[4, j], E[5, j]), SVector(E[6, j], E[7, j], E[8, j]),
     SVector(E[9, j], E[10, j], E[11, j]), SVector(E[12, j], E[13, j], E[14, j]))
end

function direct_cols(E, js, target::SVector{3,Float64})
    u = zero(SVector{3,Float64})
    @inbounds for j in js
        tag = Int(E[1, j]); nv = Int(E[2, j])
        (1 <= tag <= 5 && nv >= 3) || continue
        v1, v2, v3, v4 = colverts(E, j)
        uq, _ = FM._rect_panel_pair(FM.RectangularPanelInfluence(), target,
            tag, nv, v1, v2, v3, v4, E[15, j], E[16, j], E[17, j],
            Val(false), Val(REG))
        u += uq
    end
    return u
end
# ---- end verbatim ----

"distance from point p to segment ab"
function segdist(p, a, b)
    ab = b - a
    t = clamp(dot(p - a, ab) / max(dot(ab, ab), eps()), 0.0, 1.0)
    return norm(p - (a + t * ab))
end

for np in (3544, 12776, 28389)
    pre = joinpath(DUMPDIR, "dump_np$np")
    isfile(pre * "_meta.txt") || continue
    positions = readmat(pre * "_positions_3xN_f64.bin", 3)
    hostU = readmat(pre * "_hostU_3xN_f64.bin", 3)
    deviceU = readmat(pre * "_deviceU_3xN_f64.bin", 3)
    srcmat = readmat(pre * "_body1_srcmat_17xS_f64.bin", 17)
    cent = readmat(pre * "_body1_cent_3xS_f64.bin", 3)
    wakemat = readmat(pre * "_body1_wakemat_8xS_f64.bin", 8)
    E, _ = expand_columns(srcmat, wakemat, cent)
    ns = size(srcmat, 2)
    particles = [SVector{3,Float64}(positions[:, i]) for i in 1:np]

    # wake arm segments: TE edge (w1,w2), sides (w1,v1w), (w2,v2w), outer (v1w,v2w)
    shed = [k for k in 1:ns if wakemat[1, k] > 0]
    arms = Vector{NTuple{4,SVector{3,Float64}}}()
    for k in shed
        idx1 = Int(wakemat[1, k]); idx2 = Int(wakemat[5, k])
        vs = (SVector(srcmat[3, k], srcmat[4, k], srcmat[5, k]),
              SVector(srcmat[6, k], srcmat[7, k], srcmat[8, k]),
              SVector(srcmat[9, k], srcmat[10, k], srcmat[11, k]))
        w1 = vs[idx1]; w2 = vs[idx2]
        v1w = w1 + SVector(wakemat[2, k], wakemat[3, k], wakemat[4, k])
        v2w = w2 + SVector(wakemat[6, k], wakemat[7, k], wakemat[8, k])
        push!(arms, (w1, w2, v1w, v2w))
    end

    # sample must match p39/p40 so target ids map to the same dense values
    Random.seed!(39)
    idx = np <= 5000 ? collect(1:np) : sort(Random.shuffle(1:np)[1:5000])
    nsmp = length(idx)
    Udense = zeros(3, nsmp); Uarm = zeros(3, nsmp)
    @threads for s in eachindex(idx)
        Udense[:, s] .= direct_cols(E, 1:size(E, 2), particles[idx[s]])
        Uarm[:, s] .= direct_cols(E, ns+1:size(E, 2), particles[idx[s]])
    end
    Uh = hostU[:, idx]; Ud = deviceU[:, idx]
    rdev = Ud .- Udense
    pt_err = [norm(view(rdev, :, s)) for s in 1:nsmp]
    ord = sortperm(pt_err; rev=true)

    relL2(a, b) = sqrt(sum(abs2, a .- b)) / max(sqrt(sum(abs2, b)), eps())
    println("\n================ np=$np  (dev_vs_dns=", @sprintf("%.3e", relL2(Ud, Udense)),
        ", top-12 shown) ================")
    @printf("%-7s %-9s | %9s %9s %9s %9s | %9s %9s %9s | %9s %9s %9s\n",
        "pid", "err_rank", "d_TE", "d_armfil", "d_armout", "d_part",
        "|Udns|", "|Ubody|", "|Uarm|", "|dev-dns|", "|hfm-dns|", "|dev-hfm|")
    for r in 1:12
        s = ord[r]
        p = particles[idx[s]]
        d_te = minimum(a -> segdist(p, a[1], a[2]), arms)
        d_fil = minimum(a -> min(segdist(p, a[1], a[3]), segdist(p, a[2], a[4])), arms)
        d_out = minimum(a -> segdist(p, a[3], a[4]), arms)
        d_part = minimum(j -> j == idx[s] ? Inf : norm(p - particles[j]), 1:np)
        udns = view(Udense, :, s); uarm = view(Uarm, :, s)
        @printf("%-7d %-9d | %9.5f %9.5f %9.5f %9.5f | %9.3e %9.3e %9.3e | %9.3e %9.3e %9.3e\n",
            idx[s], r, d_te, d_fil, d_out, d_part,
            norm(udns), norm(udns .- uarm), norm(uarm),
            norm(view(Ud, :, s) .- udns), norm(view(Uh, :, s) .- udns),
            norm(view(Ud, :, s) .- view(Uh, :, s)))
    end
    # context: overall medians
    med_te = median([minimum(a -> segdist(particles[idx[s]], a[1], a[2]), arms)
        for s in 1:min(nsmp, 1000)])
    @printf("median d_TE over first 1000 sampled targets: %.5f ; core_size=%.4g\n",
        med_te, srcmat[17, 1])
end
println("DONE")
