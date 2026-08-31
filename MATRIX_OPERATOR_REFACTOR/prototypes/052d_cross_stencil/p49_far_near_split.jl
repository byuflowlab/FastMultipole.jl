# P4.9 — production far/near split vs prototype at the EXACT production box
# (052d, 2026-08-29). Uses job 13510533's eval-time dumps: bitwise-identical
# inputs plus the production frozen box (meta) and the far-only device output
# (ateval_np3544_farU.bin). Splits the proto pipeline the same way (far =
# local-expansion evaluation only; near = near shell + demoted direct) and
# attributes the production delta to far vs near, per-target at the worst
# particles and in aggregate.
#
# Run: JULIA_NUM_THREADS=4 DUMPDIR=<relU_dumps_5e dir> \
#      julia --project=../../../../FLOWPanel.jl p49_far_near_split.jl

include(joinpath(@__DIR__, "CrossStencil.jl"))
using .CrossStencil
import FastMultipole
const FM = FastMultipole
using StaticArrays, Printf, LinearAlgebra, Statistics
using Base.Threads

const DUMPDIR = get(ENV, "DUMPDIR", "")
isempty(DUMPDIR) && error("set DUMPDIR")
const Q = 12
const ELLX = 5
const P = 6
const RG = 0.006
const REG = 4
const WAKE_TAG = 3
const lhv = Val(false)
const ds = FM.DerivativesSwitch(false, true, false)

# production box from ateval meta (job 13510533)
const XMIN = SVector(-0.18023760190087498, -0.17793772306714212, -0.17793772306714215)
const H0 = 0.17793772306714215

readmat(f, r) = collect(reshape(reinterpret(Float64, read(f)), r, :))

# ---- verbatim p39 helpers ----
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
@inline boxgap(o::SVector{3,Int}, w) =
    w * norm(SVector(max(abs(o[1]) - 1, 0), max(abs(o[2]) - 1, 0), max(abs(o[3]) - 1, 0)))

"run_cross split: returns (Ufar, Unear, n_m2l, n_dem)"
function run_cross_split(E, sortpos, g, particles, idx)
    tq = CrossStencil.UniformQTables(Q)
    npush = length(tq.push_offsets)
    member = falses(8, npush)
    for ph in 1:8, k in tq.by_phase[ph]
        member[ph, k] = true
    end
    src_levels, sorder, _ = build_level_cells(g, sortpos, ELLX)
    anc = [Dict{UInt64,Vector{Int}}() for _ in 0:ELLX]
    for (s, j) in enumerate(idx)
        c, _ = CrossStencil.level_coords(g, particles[j], ELLX)
        code = CrossStencil.SharedRadix.morton_encode(UInt64(c[1]), UInt64(c[2]), UInt64(c[3]))
        for L in 0:ELLX
            push!(get!(Vector{Int}, anc[L + 1], code >> (3 * (ELLX - L))), s)
        end
    end
    FM.update_Hs_π2!(FM.Hs_π2, P)
    FM.update_ζs_mag!(FM.ζs_mag, P); FM.update_ηs_mag!(FM.ηs_mag, P)
    FM.update_M̃!(FM.M̃, P); FM.update_L̃!(FM.L̃, P)
    harmonics = FM.initialize_harmonics(P)
    gnm = FM.initialize_gradient_n_m(P)
    w1 = FM.initialize_expansion(P); w2 = FM.initialize_expansion(P)
    w3 = FM.initialize_expansion(P)
    Ts = zeros(FM.length_Ts(P)); eimϕs = zeros(2, P + 1)

    mult = [Dict{UInt64,Array{Float64,3}}() for _ in 0:ELLX]
    leaf = src_levels[ELLX + 1]
    for ci in 1:length(leaf.codes)
        e = get!(() -> FM.initialize_expansion(P), mult[ELLX + 1], leaf.codes[ci])
        ctr = cellcenter(g, decode3(leaf.codes[ci]), ELLX)
        for k in leaf.starts[ci]:leaf.starts[ci + 1] - 1
            b2m_col!(e, harmonics, E, sorder[k], ctr)
        end
    end
    for L in ELLX-1:-1:0
        for (ccode, ce) in mult[L + 2]
            pcode = ccode >> 3
            pe = get!(() -> FM.initialize_expansion(P), mult[L + 1], pcode)
            pb = dummy_branch(cellcenter(g, decode3(pcode), L))
            cb = dummy_branch(cellcenter(g, decode3(ccode), L + 1))
            FM.multipole_to_multipole!(pe, pb, ce, cb, w1, w2, Ts, eimϕs,
                FM.ζs_mag, FM.Hs_π2, P, lhv)
        end
    end
    demoted = Tuple{UnitRange{Int},Vector{Int}}[]
    locals_ = [Dict{UInt64,Array{Float64,3}}() for _ in 0:ELLX]
    n_m2l = 0; n_dem = 0
    for L in 2:ELLX
        G = 1 << L
        w = 2 * g.h0 / G
        slc = src_levels[L + 1]
        for (bcode, ss) in anc[L + 1]
            bl = get!(() -> FM.initialize_expansion(P), locals_[L + 1], bcode)
            bc = decode3(bcode)
            bb = dummy_branch(cellcenter(g, bc, L))
            if L > 2
                pl = get(locals_[L], bcode >> 3, nothing)
                if pl !== nothing
                    pb = dummy_branch(cellcenter(g, decode3(bcode >> 3), L - 1))
                    FM.local_to_local!(bl, bb, pl, pb, w1, w2, Ts, eimϕs,
                        FM.ηs_mag, FM.Hs_π2, P, lhv)
                end
            end
            for k in 1:npush
                o = tq.push_offsets[k]
                ax = bc[1] - o[1]; ay = bc[2] - o[2]; az = bc[3] - o[3]
                (0 <= ax < G && 0 <= ay < G && 0 <= az < G) || continue
                member[1 + (ax & 1) + 2 * (ay & 1) + 4 * (az & 1), k] || continue
                acode = CrossStencil.SharedRadix.morton_encode(UInt64(ax), UInt64(ay), UInt64(az))
                ci = CrossStencil.cell_index(slc, acode)
                ci == 0 && continue
                if boxgap(o, w) < RG
                    push!(demoted, (slc.starts[ci]:slc.starts[ci + 1] - 1, ss))
                    n_dem += 1
                    continue
                end
                ab = dummy_branch(cellcenter(g, SVector(ax, ay, az), L))
                FM.multipole_to_local!(bl, bb, mult[L + 1][acode], ab, w1, w2, w3,
                    Ts, eimϕs, FM.ζs_mag, FM.ηs_mag, FM.Hs_π2, FM.M̃, FM.L̃, P, lhv, nothing)
                n_m2l += 1
            end
        end
    end
    Ufar = zeros(3, length(idx))
    for (bcode, ss) in anc[ELLX + 1]
        bl = get(locals_[ELLX + 1], bcode, nothing)
        bl === nothing && continue
        ctr = cellcenter(g, decode3(bcode), ELLX)
        for s in ss
            Δx = particles[idx[s]] - ctr
            _, grad, _ = FM.evaluate_local(Δx, harmonics, gnm, bl, P, lhv, ds)
            Ufar[:, s] .= grad
        end
    end
    Unear = zeros(3, length(idx))
    G = 1 << ELLX
    for (bcode, ss) in anc[ELLX + 1]
        bc = decode3(bcode)
        srcids = Int[]
        for o in tq.near_offsets
            ax = bc[1] - o[1]; ay = bc[2] - o[2]; az = bc[3] - o[3]
            (0 <= ax < G && 0 <= ay < G && 0 <= az < G) || continue
            acode = CrossStencil.SharedRadix.morton_encode(UInt64(ax), UInt64(ay), UInt64(az))
            ci = CrossStencil.cell_index(src_levels[ELLX + 1], acode)
            ci == 0 && continue
            lc = src_levels[ELLX + 1]
            append!(srcids, (sorder[k] for k in lc.starts[ci]:lc.starts[ci + 1] - 1))
        end
        isempty(srcids) && continue
        for s in ss
            Unear[:, s] .+= direct_cols(E, srcids, particles[idx[s]])
        end
    end
    for (srange, ss) in demoted
        srcids = [sorder[k] for k in srange]
        for s in ss
            Unear[:, s] .+= direct_cols(E, srcids, particles[idx[s]])
        end
    end
    return Ufar, Unear, n_m2l, n_dem
end
# ---- end ----

relL2(a, b) = sqrt(sum(abs2, a .- b)) / max(sqrt(sum(abs2, b)), eps())

np = 3544
positions = readmat(joinpath(DUMPDIR, "ateval_np$(np)_positions.bin"), 3)
srcmat = readmat(joinpath(DUMPDIR, "ateval_np$(np)_srcmat.bin"), 17)
wakemat = readmat(joinpath(DUMPDIR, "ateval_np$(np)_wakemat.bin"), 8)
cent = readmat(joinpath(DUMPDIR, "ateval_np$(np)_cent.bin"), 3)
farU = readmat(joinpath(DUMPDIR, "ateval_np$(np)_farU.bin"), 3)
deviceU = readmat(joinpath(DUMPDIR, "dump_np$(np)_deviceU_3xN_f64.bin"), 3)
hostU = readmat(joinpath(DUMPDIR, "dump_np$(np)_hostU_3xN_f64.bin"), 3)
nearU = deviceU .- farU

E, sortpos = expand_columns(srcmat, wakemat, cent)
ncol = size(E, 2)
particles = [SVector{3,Float64}(positions[:, i]) for i in 1:np]
idx = collect(1:np)
g = CrossGrid{Float64}(XMIN, H0)

Ufar, Unear, n_m2l, n_dem = run_cross_split(E, sortpos, g, particles, idx)
Udense = zeros(3, np)
@threads for s in 1:np
    Udense[:, s] .= direct_cols(E, 1:ncol, particles[s])
end
Uproto = Ufar .+ Unear

@printf("proto (production box): n_m2l=%d n_dem=%d\n", n_m2l, n_dem)
@printf("proto_vs_dns %.3e | proddev_vs_dns %.3e | proto_vs_prodev %.3e\n",
    relL2(Uproto, Udense), relL2(deviceU, Udense), relL2(Uproto, deviceU))
@printf("FAR : prod_vs_proto %.3e   (||prodfar||=%.3e ||protofar||=%.3e)\n",
    relL2(farU, Ufar), norm(farU), norm(Ufar))
@printf("NEAR: prod_vs_proto %.3e   (||prodnear||=%.3e ||protonear||=%.3e)\n",
    relL2(nearU, Unear), norm(nearU), norm(Unear))
println("\nper-target (13 worst just-shed):")
@printf("%-6s %11s %11s %11s %11s\n", "pid", "|dfar|", "|dnear|", "|dev-dns|", "|hfm-dns|")
for pid in np-12:np
    @printf("%-6d %11.3e %11.3e %11.3e %11.3e\n", pid,
        norm(farU[:, pid] .- Ufar[:, pid]), norm(nearU[:, pid] .- Unear[:, pid]),
        norm(deviceU[:, pid] .- Udense[:, pid]), norm(hostU[:, pid] .- Udense[:, pid]))
end
println("DONE")
