# P5.0 — production near PAIR-SET compare (052d, 2026-08-29, step 5f).
# Uses the eval-time list dump (ateval_np*_lists_*.bin from the 5f job) to
# (1) replay the production near field on the host from the EXACT production
#     block lists (same direct kernel as the dense mirror) and compare it
#     against the dumped device near field (deviceU - farU): tells whether the
#     block lists + occupancy fully determine the production near output;
# (2) bit-compare the production per-target near source sets against the proto
#     near shell (panel-id sets AND multiplicity);
# (3) compute the direct field of the pair-set difference and match it against
#     dnear per target at the worst just-shed pids.
#
# Run: JULIA_NUM_THREADS=4 DUMPDIR=<relU_dumps_5e dir> \
#      julia --project=../../../../FLOWPanel.jl p50_pairset_compare.jl

include(joinpath(@__DIR__, "CrossStencil.jl"))
using .CrossStencil
import FastMultipole
const FM = FastMultipole
using StaticArrays, Printf, LinearAlgebra, Statistics
using Base.Threads

const DUMPDIR = get(ENV, "DUMPDIR", "")
isempty(DUMPDIR) && error("set DUMPDIR")
const NP = parse(Int, get(ENV, "NP", "3544"))
const Q = 12
const ELLX = 5
const RG = 0.006
const REG = parse(Int, get(ENV, "REG", "4"))
const WAKE_TAG = 3

readmat(f, r) = collect(reshape(reinterpret(Float64, read(f)), r, :))

# ---- lists reader (matches the wdump format in FLOWPanel_gpu_influence.jl) ----
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
            A = reshape(reinterpret(T, read(pre * "_lists_" * name * ".bin")),
                dims...)
            d[String(name)] = collect(A)
        end
    end
    return d
end

# ---- verbatim p39/p49 helpers ----
function expand_columns(srcmat, wakemat, cent)
    ns = size(srcmat, 2)
    nshed = count(k -> wakemat[1, k] > 0, 1:ns)
    E = zeros(17, ns + 2 * nshed)
    E[:, 1:ns] .= srcmat
    sortpos = Vector{SVector{3,Float64}}(undef, ns + 2 * nshed)
    owner = collect(1:ns + 2 * nshed)          # E column -> owning panel id
    armcols = Dict{Int,Tuple{Int,Int}}()       # panel id -> its two arm columns
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
        armcols[k] = (col + 1, col + 2)
        for (a, b, c) in ((w1, w2, v1w), (v1w, w2, v2w))
            col += 1
            E[1, col] = WAKE_TAG; E[2, col] = 3
            E[3:5, col] .= a; E[6:8, col] .= b; E[9:11, col] .= c
            E[12:14, col] .= c
            E[15, col] = mu; E[16, col] = 0.0; E[17, col] = koff
            sortpos[col] = sortpos[k]
            owner[col] = k
        end
    end
    return E, sortpos, owner, armcols
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

relL2(a, b) = sqrt(sum(abs2, a .- b)) / max(sqrt(sum(abs2, b)), eps())

# ---- load dumps ----
pre = joinpath(DUMPDIR, "ateval_np$(NP)")
positions = readmat(pre * "_positions.bin", 3)
srcmat = readmat(pre * "_srcmat.bin", 17)
wakemat = readmat(pre * "_wakemat.bin", 8)
cent = readmat(pre * "_cent.bin", 3)
farU = readmat(pre * "_farU.bin", 3)
deviceU = readmat(joinpath(DUMPDIR, "dump_np$(NP)_deviceU_3xN_f64.bin"), 3)
nearU = deviceU .- farU                       # production device near field
np = size(positions, 2)
ns = size(srcmat, 2)
np == NP || error("np mismatch")
L = read_lists(pre)
meta = Dict(split(ln, " = ", limit=2)[1] => strip(split(ln, " = ", limit=2)[2])
    for ln in readlines(pre * "_lists_meta.txt") if occursin(" = ", ln))
XMIN = SVector{3,Float64}(eval(Meta.parse(meta["x_min"])))
H0 = parse(Float64, meta["h0"])
n_demoted = parse(Int, meta["blocks_n_demoted"])
E, sortpos, owner, armcols = expand_columns(srcmat, wakemat, cent)
particles = [SVector{3,Float64}(positions[:, i]) for i in 1:np]
@printf("np=%d ns=%d ncolE=%d n_blocks=%d n_demoted=%d h0=%.12g\n",
    np, ns, size(E, 2), length(L["blocks_targets"]), n_demoted, H0)

# panel id -> E columns (panel + its wake arms, matching the device kernel)
cols_of(k::Int) = haskey(armcols, k) ? (k, armcols[k]...) : (k,)

# ---- (1) host replay of production near from the block lists ----
tr = L["particles_node_ranges"]; tp = L["particles_perm"]
sr = L["panels_node_ranges"];    sp = L["panels_perm"]
bt = L["blocks_targets"];        bs = L["blocks_sources"]
tgt_srcs = [Int[] for _ in 1:np]              # per-particle panel ids, WITH multiplicity
for i in eachindex(bt)
    tn = Int(bt[i]); sn = Int(bs[i])
    sids = Int.(sp[Int(sr[1, sn]):Int(sr[1, sn]) + Int(sr[2, sn]) - 1])
    for j in Int(tr[1, tn]):Int(tr[1, tn]) + Int(tr[2, tn]) - 1
        append!(tgt_srcs[Int(tp[j])], sids)
    end
end
prodNear = zeros(3, np)
@threads for t in 1:np
    js = Int[]
    for k in tgt_srcs[t]
        append!(js, cols_of(k))
    end
    prodNear[:, t] .= direct_cols(E, js, particles[t])
end
@printf("\n(1) host replay of production block lists vs device near:\n")
@printf("    relL2(prodNearReplay, deviceNear) = %.3e\n", relL2(prodNear, nearU))
@printf("    (||deviceNear||=%.3e ||replay||=%.3e)\n", norm(nearU), norm(prodNear))

# ---- (2) proto near shell per-target sets (panel ids) at the SAME box ----
g = CrossGrid{Float64}(XMIN, H0)
tq = CrossStencil.UniformQTables(Q)
src_levels, sorder, _ = build_level_cells(g, sortpos, ELLX)
leaf = src_levels[ELLX + 1]
anc = Dict{UInt64,Vector{Int}}()
for i in 1:np
    c, _ = CrossStencil.level_coords(g, particles[i], ELLX)
    code = CrossStencil.SharedRadix.morton_encode(UInt64(c[1]), UInt64(c[2]), UInt64(c[3]))
    push!(get!(Vector{Int}, anc, code), i)
end
decode3(code) = SVector{3,Int}(Int.(CrossStencil.SharedRadix.morton_decode(code))...)
@inline boxgap(o::SVector{3,Int}, w) =
    w * norm(SVector(max(abs(o[1]) - 1, 0), max(abs(o[2]) - 1, 0), max(abs(o[3]) - 1, 0)))
G = 1 << ELLX
w_leaf = 2 * H0 / G
proto_srcs = [Int[] for _ in 1:np]            # per-particle panel ids (near shell)
n_proto_dem = 0
for (bcode, ss) in anc
    bc = decode3(bcode)
    srcids = Int[]
    for o in tq.near_offsets
        ax = bc[1] - o[1]; ay = bc[2] - o[2]; az = bc[3] - o[3]
        (0 <= ax < G && 0 <= ay < G && 0 <= az < G) || continue
        acode = CrossStencil.SharedRadix.morton_encode(UInt64(ax), UInt64(ay), UInt64(az))
        ci = CrossStencil.cell_index(leaf, acode)
        ci == 0 && continue
        append!(srcids, (sorder[k] for k in leaf.starts[ci]:leaf.starts[ci + 1] - 1))
    end
    # E columns -> owning panel ids, arms dropped (re-added via cols_of);
    # count each panel once per appearance of ITS OWN column (arms excluded)
    pids = [owner[j] for j in srcids if j <= ns]
    for s in ss
        proto_srcs[s] = pids
    end
end
# demoted M2L pairs (RG guard) also land in proto near — count them
# (production meta says n_demoted; proto should agree)
# NOTE: full demoted handling matches p49; with n_demoted=0 both sides skip it.
n_demoted == 0 || @warn "n_demoted != 0 — demoted pairs NOT compared here"

nset_diff = 0; nmult_diff = 0; npair_extra = 0; npair_missing = 0
diff_pids = Int[]
for t in 1:np
    a = sort(tgt_srcs[t]); b = sort(proto_srcs[t])
    if a != b
        if Set(a) != Set(b)
            nset_diff += 1
        else
            nmult_diff += 1
        end
        push!(diff_pids, t)
        ca = Dict{Int,Int}(); for k in a; ca[k] = get(ca, k, 0) + 1; end
        cb = Dict{Int,Int}(); for k in b; cb[k] = get(cb, k, 0) + 1; end
        for (k, c) in ca; npair_extra += max(0, c - get(cb, k, 0)); end
        for (k, c) in cb; npair_missing += max(0, c - get(ca, k, 0)); end
    end
end
@printf("\n(2) pair-set compare (production blocks vs proto near shell):\n")
@printf("    targets with different SET: %d, same set but diff multiplicity: %d (of %d)\n",
    nset_diff, nmult_diff, np)
@printf("    extra pairs (prod - proto): %d, missing pairs (proto - prod): %d\n",
    npair_extra, npair_missing)

# ---- (3) direct field of the pair-set difference vs dnear ----
protoNear = zeros(3, np)
@threads for t in 1:np
    js = Int[]
    for k in proto_srcs[t]
        append!(js, cols_of(k))
    end
    protoNear[:, t] .= direct_cols(E, js, particles[t])
end
dnear = nearU .- protoNear                    # production near - proto near
ddiff = prodNear .- protoNear                 # pair-set difference direct field
@printf("\n(3) pair-set difference field vs near delta:\n")
@printf("    relL2(deviceNear, protoNear) = %.3e\n", relL2(nearU, protoNear))
@printf("    relL2(ddiff, dnear)          = %.3e  (1.0-match means lists explain dnear)\n",
    relL2(ddiff, dnear))
@printf("    ||dnear||=%.3e ||ddiff||=%.3e\n", norm(dnear), norm(ddiff))
println("\nper-target (13 worst just-shed + any set-diff pids):")
@printf("%-6s %11s %11s %11s %6s %6s\n", "pid", "|dnear|", "|ddiff|",
    "|dnear-ddiff|", "nprod", "nprot")
shown = sort(unique(vcat(collect(np - 12:np), diff_pids[1:min(end, 20)])))
for t in shown
    @printf("%-6d %11.3e %11.3e %11.3e %6d %6d\n", t,
        norm(dnear[:, t]), norm(ddiff[:, t]),
        norm(dnear[:, t] .- ddiff[:, t]), length(tgt_srcs[t]), length(proto_srcs[t]))
end
println("DONE")
