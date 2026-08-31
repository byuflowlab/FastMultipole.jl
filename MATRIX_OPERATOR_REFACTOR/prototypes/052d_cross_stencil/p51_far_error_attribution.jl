# P5.1 — far-field error attribution at the worst just-shed pids (052d, step 5f).
# Given p50's finding (near path + lists exact at reg 3), the whole 9.058e-4
# relU must be far-field error of the cross pass. This script, at the EXACT
# production box and pair set:
#   (a) computes the true far field per pid (direct over the complement of the
#       production near set) with reg 3 AND reg 4 — their difference is the
#       regularization sensitivity of the cell-far sources (nonzero only if
#       far-binned geometry reaches within ~6 sigma of the target);
#   (b) compares the production device farU against both — the residual vs
#       reg 3 is the true production far error per pid;
#   (c) lists which far columns pass within 6 mm of each worst pid (arm vs
#       panel counts + min distances) to identify the escaping geometry.
#
# Run: JULIA_NUM_THREADS=4 DUMPDIR=<relU_dumps_5f dir> \
#      julia --project=../../../../FLOWPanel.jl p51_far_error_attribution.jl

include(joinpath(@__DIR__, "CrossStencil.jl"))
using .CrossStencil
import FastMultipole
const FM = FastMultipole
using StaticArrays, Printf, LinearAlgebra
using Base.Threads

const DUMPDIR = get(ENV, "DUMPDIR", "")
isempty(DUMPDIR) && error("set DUMPDIR")
const NP = parse(Int, get(ENV, "NP", "3544"))
const WAKE_TAG = 3

readmat(f, r) = collect(reshape(reinterpret(Float64, read(f)), r, :))

function read_lists(pre)
    d = Dict{String,Any}()
    for ln in readlines(pre * "_lists_meta.txt")
        if occursin(" = ", ln)
            k, v = split(ln, " = ", limit=2)
            d[strip(k)] = strip(v)
        else
            name, et, sz = split(ln)
            T = getfield(Base, Symbol(et))
            dims = parse.(Int, split(sz, "x"))
            d[String(name)] = collect(reshape(reinterpret(T,
                read(pre * "_lists_" * name * ".bin")), dims...))
        end
    end
    return d
end

function expand_columns(srcmat, wakemat, cent)
    ns = size(srcmat, 2)
    nshed = count(k -> wakemat[1, k] > 0, 1:ns)
    E = zeros(17, ns + 2 * nshed)
    E[:, 1:ns] .= srcmat
    owner = collect(1:ns + 2 * nshed)
    armcols = Dict{Int,Tuple{Int,Int}}()
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
        armcols[k] = (col + 1, col + 2)
        for (a, b, c) in ((w1, w2, v1w), (v1w, w2, v2w))
            col += 1
            E[1, col] = WAKE_TAG; E[2, col] = 3
            E[3:5, col] .= a; E[6:8, col] .= b; E[9:11, col] .= c
            E[12:14, col] .= c
            E[15, col] = mu; E[16, col] = 0.0; E[17, col] = koff
            owner[col] = k
        end
    end
    return E, owner, armcols
end

@inline function colverts(E, j)
    (SVector(E[3, j], E[4, j], E[5, j]), SVector(E[6, j], E[7, j], E[8, j]),
     SVector(E[9, j], E[10, j], E[11, j]), SVector(E[12, j], E[13, j], E[14, j]))
end

function direct_cols(E, js, target::SVector{3,Float64}, reg::Int)
    u = zero(SVector{3,Float64})
    regv = Val(reg)
    @inbounds for j in js
        tag = Int(E[1, j]); nv = Int(E[2, j])
        (1 <= tag <= 5 && nv >= 3) || continue
        v1, v2, v3, v4 = colverts(E, j)
        uq, _ = FM._rect_panel_pair(FM.RectangularPanelInfluence(), target,
            tag, nv, v1, v2, v3, v4, E[15, j], E[16, j], E[17, j],
            Val(false), regv)
        u += uq
    end
    return u
end

# min distance from point to segment
function seg_dist(p, a, b)
    ab = b - a
    t = clamp(dot(p - a, ab) / max(dot(ab, ab), eps()), 0.0, 1.0)
    return norm(p - (a + t * ab))
end
# min distance from target to any edge of column j's geometry
function col_dist(E, j, p)
    v1, v2, v3, v4 = colverts(E, j)
    nv = Int(E[2, j])
    d = min(seg_dist(p, v1, v2), seg_dist(p, v2, v3))
    d = min(d, seg_dist(p, v3, nv == 4 ? v4 : v1))
    nv == 4 && (d = min(d, seg_dist(p, v4, v1)))
    return d
end

pre = joinpath(DUMPDIR, "ateval_np$(NP)")
positions = readmat(pre * "_positions.bin", 3)
srcmat = readmat(pre * "_srcmat.bin", 17)
wakemat = readmat(pre * "_wakemat.bin", 8)
cent = readmat(pre * "_cent.bin", 3)
farU = readmat(pre * "_farU.bin", 3)
np = size(positions, 2)
ns = size(srcmat, 2)
L = read_lists(pre)
E, owner, armcols = expand_columns(srcmat, wakemat, cent)
ncol = size(E, 2)
particles = [SVector{3,Float64}(positions[:, i]) for i in 1:np]

# production near set per particle (panel ids), from the dumped block lists
tr = L["particles_node_ranges"]; tp = L["particles_perm"]
sr = L["panels_node_ranges"];    sp = L["panels_perm"]
bt = L["blocks_targets"];        bs = L["blocks_sources"]
near_panels = [Set{Int}() for _ in 1:np]
for i in eachindex(bt)
    tn = Int(bt[i]); sn = Int(bs[i])
    sids = Int.(sp[Int(sr[1, sn]):Int(sr[1, sn]) + Int(sr[2, sn]) - 1])
    for j in Int(tr[1, tn]):Int(tr[1, tn]) + Int(tr[2, tn]) - 1
        union!(near_panels[Int(tp[j])], sids)
    end
end
# far E columns per particle = all cols whose OWNER panel is not in near set
farcols_of(t) = [j for j in 1:ncol if !(owner[j] in near_panels[t])]

pids = collect(np - 12:np)
@printf("%-6s %12s %12s %12s %12s | %6s %6s %10s\n", "pid", "|far3-far4|",
    "|farU-far3|", "|farU-far4|", "|far3|", "arms<6", "pnls<6", "mindist")
agg3 = 0.0; agg4 = 0.0; aggd = 0.0
results = Vector{NTuple{7,Float64}}(undef, length(pids))
@threads for ii in eachindex(pids)
    t = pids[ii]
    fc = farcols_of(t)
    u3 = direct_cols(E, fc, particles[t], 3)
    u4 = direct_cols(E, fc, particles[t], 4)
    fp = SVector{3,Float64}(farU[:, t])
    # geometry scan: far cols passing within 6 mm
    na = 0; npn = 0; dmin = Inf
    for j in fc
        d = col_dist(E, j, particles[t])
        d < dmin && (dmin = d)
        if d < 0.006
            j > ns ? (na += 1) : (npn += 1)
        end
    end
    results[ii] = (norm(u3 - u4), norm(fp - u3), norm(fp - u4), norm(u3),
        Float64(na), Float64(npn), dmin)
end
for (ii, t) in enumerate(pids)
    r = results[ii]
    @printf("%-6d %12.3e %12.3e %12.3e %12.3e | %6d %6d %10.4g\n", t,
        r[1], r[2], r[3], r[4], Int(r[5]), Int(r[6]), r[7])
end
println("DONE")
