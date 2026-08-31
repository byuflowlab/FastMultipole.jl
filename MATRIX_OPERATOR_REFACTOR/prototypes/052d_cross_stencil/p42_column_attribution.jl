# P4.2 — per-column attribution of the production-vs-dense delta (052d).
#
# p41 showed that at several just-shed particles BOTH production routes agree
# to ~5e-6 while differing from the p38-recipe dense by up to 3e-3 — i.e. some
# specific source column(s) are treated differently by production than by the
# dense reconstruction. For each such "oracle" target (dev-hfmm < 2e-5), this
# script computes every column's individual field u_j at the target and fits
#   delta ≈ α u_j          (single column, scalar α)
#   delta ≈ α (u_j1 + u_j2) (per-shedding-station arm pair)
# reporting the candidates with the smallest residual. α = -1 means production
# OMITS the column; α = +1 doubled; other α = strength scaling.
#
# Run: JULIA_NUM_THREADS=4 DUMPDIR=<dir> \
#      julia --project=../../../../FLOWPanel.jl p42_column_attribution.jl

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

# ---- verbatim p39 helpers ----
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
    parent = zeros(Int, ns + 2 * nshed)
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
            parent[col] = k
        end
    end
    return E, sortpos, parent
end

@inline function colverts(E, j)
    (SVector(E[3, j], E[4, j], E[5, j]), SVector(E[6, j], E[7, j], E[8, j]),
     SVector(E[9, j], E[10, j], E[11, j]), SVector(E[12, j], E[13, j], E[14, j]))
end

function col_field(E, j, target::SVector{3,Float64})
    tag = Int(E[1, j]); nv = Int(E[2, j])
    (1 <= tag <= 5 && nv >= 3) || return zero(SVector{3,Float64})
    v1, v2, v3, v4 = colverts(E, j)
    uq, _ = FM._rect_panel_pair(FM.RectangularPanelInfluence(), target,
        tag, nv, v1, v2, v3, v4, E[15, j], E[16, j], E[17, j],
        Val(false), Val(REG))
    return uq
end
# ---- end ----

for np in (3544, 12776, 28389)
    pre = joinpath(DUMPDIR, "dump_np$np")
    isfile(pre * "_meta.txt") || continue
    positions = readmat(pre * "_positions_3xN_f64.bin", 3)
    hostU = readmat(pre * "_hostU_3xN_f64.bin", 3)
    deviceU = readmat(pre * "_deviceU_3xN_f64.bin", 3)
    srcmat = readmat(pre * "_body1_srcmat_17xS_f64.bin", 17)
    cent = readmat(pre * "_body1_cent_3xS_f64.bin", 3)
    wakemat = readmat(pre * "_body1_wakemat_8xS_f64.bin", 8)
    E, _, parent = expand_columns(srcmat, wakemat, cent)
    ns = size(srcmat, 2); ncol = size(E, 2)
    particles = [SVector{3,Float64}(positions[:, i]) for i in 1:np]

    # oracle targets: top |dev - dense| where dev and hfmm agree tightly.
    # (recompute dense only at a candidate pool: top-200 by |dev-hfmm agreement
    # is unknown before dense; use all particles' dev/host first)
    agree = [norm(deviceU[:, i] .- hostU[:, i]) for i in 1:np]
    # dense at the 400 particles with the largest |dev| deviation potential:
    # cheap proxy — closest to shedding panels
    shedc = [SVector{3,Float64}(cent[:, k]) for k in 1:ns if wakemat[1, k] > 0]
    dTE = [minimum(c -> norm(particles[i] - c), shedc) for i in 1:np]
    pool = sortperm(dTE)[1:min(400, np)]
    Upool = zeros(3, length(pool))
    @threads for s in eachindex(pool)
        i = pool[s]
        u = zero(SVector{3,Float64})
        for j in 1:ncol
            u += col_field(E, j, particles[i])
        end
        Upool[:, s] .= u
    end
    devdel = [norm(deviceU[:, pool[s]] .- Upool[:, s]) for s in eachindex(pool)]
    ord = sortperm(devdel; rev=true)
    oracles = [pool[s] for s in ord if agree[pool[s]] < 2e-5][1:min(3, end)]
    isempty(oracles) && (println("np=$np: no tight-agreement oracle targets in pool"); continue)

    println("\n================ np=$np ================")
    for i in oracles
        s = findfirst(==(i), pool)
        tgt = particles[i]
        delta = SVector{3,Float64}(deviceU[:, i] .- Upool[:, s])
        @printf("target pid=%d  |delta|=%.3e  |U|=%.3e  dev-hfmm=%.2e\n",
            i, norm(delta), norm(view(Upool, :, s)), agree[i])
        # per-column fields
        u = Vector{SVector{3,Float64}}(undef, ncol)
        @threads for j in 1:ncol
            u[j] = col_field(E, j, tgt)
        end
        # single-column fits
        best = Tuple{Float64,Int,Float64}[]  # (residual, j, alpha)
        for j in 1:ncol
            nj2 = dot(u[j], u[j])
            nj2 > 1e-12 || continue
            α = dot(delta, u[j]) / nj2
            res = norm(delta - α * u[j]) / norm(delta)
            push!(best, (res, j, α))
        end
        sort!(best)
        println("  single-column fits (residual, col, alpha, tag, kind, |u_j|):")
        for (res, j, α) in best[1:min(6, end)]
            kind = j <= ns ? "panel" : "arm(parent=$(parent[j]))"
            @printf("    res=%.3f  col=%-6d α=%+.4f  tag=%d  %s  |u|=%.3e\n",
                res, j, α, Int(E[1, j]), kind, norm(u[j]))
        end
        # arm-pair (per shedding station) fits
        pairs = Dict{Int,Vector{Int}}()
        for j in ns+1:ncol
            push!(get!(Vector{Int}, pairs, parent[j]), j)
        end
        bestp = Tuple{Float64,Int,Float64}[]
        for (k, js) in pairs
            up = sum(u[j] for j in js)
            np2 = dot(up, up)
            np2 > 1e-12 || continue
            α = dot(delta, up) / np2
            res = norm(delta - α * up) / norm(delta)
            push!(bestp, (res, k, α))
        end
        sort!(bestp)
        println("  arm-pair fits (residual, parent panel, alpha, |u_pair|):")
        for (res, k, α) in bestp[1:min(4, end)]
            up = sum(u[j] for j in pairs[k])
            @printf("    res=%.3f  panel=%-6d α=%+.4f  |u|=%.3e  s1=%.4g s2=%.4g\n",
                res, k, α, norm(up), srcmat[15, k], srcmat[16, k])
        end
        # combined: subtract best arm-pair, then refit singles (two-term)
        if !isempty(bestp)
            res1, k1, α1 = bestp[1]
            d2 = delta - α1 * sum(u[j] for j in pairs[k1])
            best2 = Tuple{Float64,Int,Float64}[]
            for (kk, js) in pairs
                kk == k1 && continue
                up = sum(u[j] for j in js)
                np2 = dot(up, up)
                np2 > 1e-12 || continue
                α = dot(d2, up) / np2
                push!(best2, (norm(d2 - α * up) / norm(delta), kk, α))
            end
            sort!(best2)
            if !isempty(best2)
                res, k, α = best2[1]
                @printf("  two-station fit: station %d (α=%+.3f) + station %d (α=%+.3f) → res=%.3f\n",
                    k1, α1, k, α, res)
            end
        end
    end
end
println("DONE")
