# P4.6 — cell-consistent routing fit (052d, 2026-08-29).
#
# Fixes p45's flaw: a per-target fit of a 3-vector delta on many candidate
# cell-error vectors is underdetermined. Routing is a property of the TARGET
# CELL, so here we take the L-4 cell containing the worst just-shed particles,
# gather ALL dumped particles in that cell, and fit ONE routing perturbation
#   delta(t) ≈ Σ_c s_c ec_c(t),   s_c ∈ {−1,+1}, one set for all t in cell
# (s_c = −1: production far-routes cell c where dense is direct; +1: the
# opposite) by matching pursuit over candidate source cells at L=4 and L=5.
# 3×(#targets) equations per pick make the selection overdetermined.
#
# Run: JULIA_NUM_THREADS=4 DUMPDIR=<dir> \
#      julia --project=../../../../FLOWPanel.jl p46_cellwise_routing_fit.jl

include(joinpath(@__DIR__, "CrossStencil.jl"))
using .CrossStencil
import FastMultipole
const FM = FastMultipole
using StaticArrays, Random, Statistics, Printf, LinearAlgebra
using Base.Threads

const DUMPDIR = get(ENV, "DUMPDIR", "")
isempty(DUMPDIR) && error("set DUMPDIR")
const Q = 12
const ELLX = 5
const P = 6
const REG = 4
const WAKE_TAG = 3
const lhv = Val(false)
const ds = FM.DerivativesSwitch(false, true, false)

readmat(f, r) = collect(reshape(reinterpret(Float64, read(f)), r, :))

function expand_columns(srcmat, wakemat, cent)
    ns = size(srcmat, 2)
    nshed = count(k -> wakemat[1, k] > 0, 1:ns)
    E = zeros(17, ns + 2 * nshed)
    E[:, 1:ns] .= srcmat
    sortpos = Vector{SVector{3,Float64}}(undef, ns + 2 * nshed)
    for k in 1:ns
        sortpos[k] = SVector(cent[1, k], cent[2, k], cent[3, k])
    end
    col = ns
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

function b2m_col!(e, harmonics, E, j, ctr)
    tag = Int(E[1, j]); nv = Int(E[2, j])
    (1 <= tag <= 5 && nv >= 3) || return
    v1, v2, v3, v4 = colverts(E, j)
    tris = nv == 4 ? ((v1, v2, v3), (v1, v3, v4)) : ((v1, v2, v3),)
    s1 = E[15, j]; s2 = E[16, j]
    for (t1, t2, t3) in tris
        x0 = t1 - ctr; xu = t2 - t1; xv = t3 - t1
        nrm = normalize(cross(xu, xv))
        if tag == 1
            FM.body_to_multipole_panel!(FM.Panel{FM.Source}, e, harmonics,
                x0, xu, xv, nrm, SVector(s1), P)
        elseif tag == 2 || tag == 3
            FM.body_to_multipole_panel!(FM.Panel{FM.Dipole}, e, harmonics,
                x0, xu, xv, nrm, SVector(s1), P)
        else
            FM.body_to_multipole_panel!(FM.Panel{FM.SourceDipole}, e, harmonics,
                x0, xu, xv, nrm, SVector(s1, s2), P)
        end
    end
    return
end

cellcenter(g, c::SVector{3,Int}, L) = begin
    h = g.h0 / (1 << L)
    SVector(g.x_min[1] + (2c[1] + 1) * h, g.x_min[2] + (2c[2] + 1) * h,
            g.x_min[3] + (2c[3] + 1) * h)
end
decode3(code) = SVector{3,Int}(Int.(CrossStencil.SharedRadix.morton_decode(code))...)
dummy_branch(center) = FM.Branch(1:1, 0, 1:0, 0, 1, center, 0.0, SVector(0.0, 0.0, 0.0))

np = 3544
pre = joinpath(DUMPDIR, "dump_np$np")
positions = readmat(pre * "_positions_3xN_f64.bin", 3)
deviceU = readmat(pre * "_deviceU_3xN_f64.bin", 3)
hostU = readmat(pre * "_hostU_3xN_f64.bin", 3)
srcmat = readmat(pre * "_body1_srcmat_17xS_f64.bin", 17)
cent = readmat(pre * "_body1_cent_3xS_f64.bin", 3)
wakemat = readmat(pre * "_body1_wakemat_8xS_f64.bin", 8)
E, sortpos = expand_columns(srcmat, wakemat, cent)
ns = size(srcmat, 2); ncol = size(E, 2)
particles = [SVector{3,Float64}(positions[:, i]) for i in 1:np]

lo = zeros(3); hi = zeros(3)
for a in 1:3
    pv = extrema(view(positions, a, :))
    sv = extrema(vcat(vec(srcmat[2 + a, :]), vec(srcmat[5 + a, :]),
        vec(srcmat[8 + a, :])))
    lo[a] = min(pv[1], sv[1]); hi[a] = max(pv[2], sv[2])
end
h0 = 0.75 * maximum(hi .- lo) + 1e-9
g = CrossGrid{Float64}(SVector{3,Float64}((lo .+ hi) ./ 2 .- h0), h0)

FM.update_Hs_π2!(FM.Hs_π2, P)
FM.update_ζs_mag!(FM.ζs_mag, P); FM.update_ηs_mag!(FM.ηs_mag, P)
FM.update_M̃!(FM.M̃, P); FM.update_L̃!(FM.L̃, P)
harmonics = FM.initialize_harmonics(P)
gnm = FM.initialize_gradient_n_m(P)
w1e = FM.initialize_expansion(P); w2e = FM.initialize_expansion(P)
w3e = FM.initialize_expansion(P)
Ts = zeros(FM.length_Ts(P)); eimϕs = zeros(2, P + 1)

function dsum(js, t)
    u = zero(SVector{3,Float64})
    @inbounds for j in js
        tag = Int(E[1, j]); nv = Int(E[2, j])
        (1 <= tag <= 5 && nv >= 3) || continue
        v1, v2, v3, v4 = colverts(E, j)
        uq, _ = FM._rect_panel_pair(FM.RectangularPanelInfluence(), t,
            tag, nv, v1, v2, v3, v4, E[15, j], E[16, j], E[17, j],
            Val(false), Val(REG))
        u += uq
    end
    return u
end

"route field: B2M(js at src ctr) -> M2L(to tgt ctr) -> evaluate at t"
function route_u(mexp, src_ctr, tgt_ctr, t)
    bl = FM.initialize_expansion(P)
    FM.multipole_to_local!(bl, dummy_branch(tgt_ctr), mexp, dummy_branch(src_ctr),
        w1e, w2e, w3e, Ts, eimϕs, FM.ζs_mag, FM.ηs_mag, FM.Hs_π2, FM.M̃, FM.L̃,
        P, lhv, nothing)
    _, grad, _ = FM.evaluate_local(t - tgt_ctr, harmonics, gnm, bl, P, lhv, ds)
    return SVector{3,Float64}(grad)
end

# target cell: the L4 cell of the worst particle (pid 3544)
for L in (4, 5)
    tc0, _ = CrossStencil.level_coords(g, particles[3544], L)
    tcell = SVector{3,Int}(Int.(tc0)...)
    tctr = cellcenter(g, tcell, L)
    tids = [i for i in 1:np if
        SVector{3,Int}(Int.(CrossStencil.level_coords(g, particles[i], L)[1])...) == tcell]
    nt = length(tids)
    @printf("\n==== L=%d target cell (%d,%d,%d): %d particles ====\n",
        L, tcell[1], tcell[2], tcell[3], nt)

    Udns = zeros(3, nt)
    @threads for s in 1:nt
        Udns[:, s] .= dsum(1:ncol, particles[tids[s]])
    end
    Ddev = deviceU[:, tids] .- Udns
    Dhfm = hostU[:, tids] .- Udns
    @printf("||Ddev||=%.3e ||Dhfm||=%.3e (%d eqs)\n", norm(Ddev), norm(Dhfm), 3nt)

    # candidate source cells at this level within 6 cell-widths
    cells = Dict{UInt64,Vector{Int}}()
    for j in 1:ncol
        c, _ = CrossStencil.level_coords(g, sortpos[j], L)
        code = CrossStencil.SharedRadix.morton_encode(UInt64(c[1]), UInt64(c[2]), UInt64(c[3]))
        push!(get!(Vector{Int}, cells, code), j)
    end
    w = 2 * g.h0 / (1 << L)
    cand = UInt64[]
    for (code, js) in cells
        ctr = cellcenter(g, decode3(code), L)
        (norm(ctr - tctr) < 6w && decode3(code) != tcell) && push!(cand, code)
    end
    # ec matrix: per candidate, per target: direct - route
    ECs = Dict{UInt64,Matrix{Float64}}()
    for code in cand
        js = cells[code]
        ctr = cellcenter(g, decode3(code), L)
        mexp = FM.initialize_expansion(P)
        for j in js
            b2m_col!(mexp, harmonics, E, j, ctr)
        end
        M = zeros(3, nt)
        for s in 1:nt
            t = particles[tids[s]]
            M[:, s] .= dsum(js, t) .- route_u(mexp, ctr, tctr, t)
        end
        ECs[code] = M
    end
    @printf("%d candidate source cells\n", length(cand))

    for (nm, D0) in (("dev", copy(Ddev)), ("hfm", copy(Dhfm)))
        D = copy(D0)
        picks = String[]
        used = Set{UInt64}()
        for _ in 1:6
            best = (norm(D), UInt64(0), 1.0)
            for code in cand
                code in used && continue
                M = ECs[code]
                for sgn in (1.0, -1.0)
                    nr = norm(D .- sgn .* M)
                    nr < best[1] * 0.999 && (best = (nr, code, sgn))
                end
            end
            best[2] == 0 && break
            push!(used, best[2])
            D .-= best[3] .* ECs[best[2]]
            c3 = decode3(best[2])
            off = c3 - tcell
            js = cells[best[2]]
            narm = count(>(ns), js)
            push!(picks, @sprintf("%soff(%+d,%+d,%+d)|ec|=%.1e[%dp+%da]",
                best[3] > 0 ? "+" : "-", off[1], off[2], off[3],
                norm(ECs[best[2]]), length(js) - narm, narm))
        end
        @printf("%s: res %.3e -> %.3e (%.2f)  picks: %s\n",
            nm, norm(D0), norm(D), norm(D) / norm(D0), join(picks, " "))
    end
end
println("DONE")
