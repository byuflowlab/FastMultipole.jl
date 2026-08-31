# P3.9 — xverify relU attribution (052d, 2026-08-29).
#
# Job 13507743's Stage-F xverify printed relU (device cross pass vs host fmm!)
# up to 9.2e-4 at small np. Both routes are approximations, so relU sums their
# truncation errors. This script splits the blame on dumped production states
# (PANEL_FMM_DUMP_DIR hook in FLOWPanel_gpu_influence.jl): for each dump it
# computes, on CPU,
#   - a dense regularized reference (exact; the p38 per-column evaluator over
#     body columns + TE-wake arm columns),
#   - the host CrossStencil guarded pipeline at the PRODUCTION operating point
#     (q=12, ell_x=5, P=6, R_guard=0.006, LineGauss reg 4, production padded
#     root box; certified == device to ~1e-15 by p35/p38),
#   - and reads the dumped host-fmm! result,
# then reports relL2 of {cross, hostfmm} vs dense and cross vs hostfmm (the
# latter should reproduce the cluster relU at matching np).
#
# Run: JULIA_NUM_THREADS=4 DUMPDIR=<dir> \
#      julia --project=../../../../FLOWPanel.jl p39_relU_attribution.jl

include(joinpath(@__DIR__, "CrossStencil.jl"))
using .CrossStencil
import FastMultipole
const FM = FastMultipole
using StaticArrays, Random, Statistics, Printf, LinearAlgebra
using Base.Threads

const DUMPDIR = get(ENV, "DUMPDIR", "")
isempty(DUMPDIR) && error("set DUMPDIR to the PANEL_FMM_DUMP_DIR directory")
const Q = 12
const ELLX = 5
const P = 6
const RG = 0.006
const REG = parse(Int, get(ENV, "REG", "4"))
# NOTE (2026-08-29, p50 finding): production _gpu_filament_reg() is 3
# (Gaussian, the FILAMENT_REGULARIZATION default) — NOT 4 (LineGauss).
# Run with REG=3 for a production-faithful dense reference.
const WAKE_TAG = 3         # production: VortexRing wake -> tag 3
const NSAMPLE = 5000
const lhv = Val(false)     # cross pass is phi-only
const ds = FM.DerivativesSwitch(false, true, false)

readmat(f, r) = collect(reshape(reinterpret(Float64, read(f)), r, :))

"Expanded columns: body columns + 2 arm columns per shedding panel (device
Stage-B/E arm recipe; arm strength mu = tag in (2,3) ? s1 : s2). Returns
(E 17 x ncol, sortpos::Vector{SVector} = parent centroid per column)."
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

"Regularized direct velocity of columns `js` at `target` (p38 recipe)."
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

"Packed-column B2M into expansion `e` centered at `ctr` (p36 host-ref mapping)."
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

"Guarded cross pipeline on expanded columns (p32e run_guarded, column-driven)."
function run_cross(E, sortpos, g, particles, idx)
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
    U = zeros(3, length(idx))
    for (bcode, ss) in anc[ELLX + 1]
        bl = get(locals_[ELLX + 1], bcode, nothing)
        bl === nothing && continue
        ctr = cellcenter(g, decode3(bcode), ELLX)
        for s in ss
            Δx = particles[idx[s]] - ctr
            _, grad, _ = FM.evaluate_local(Δx, harmonics, gnm, bl, P, lhv, ds)
            U[:, s] .= grad
        end
    end
    # near shell + demoted routes: regularized direct over column ids
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
            U[:, s] .+= direct_cols(E, srcids, particles[idx[s]])
        end
    end
    for (srange, ss) in demoted
        srcids = [sorder[k] for k in srange]
        for s in ss
            U[:, s] .+= direct_cols(E, srcids, particles[idx[s]])
        end
    end
    return U, n_m2l, n_dem
end

relL2(a, b) = sqrt(sum(abs2, a .- b)) / max(sqrt(sum(abs2, b)), eps())

metas = sort(filter(f -> occursin(r"^dump_np\d+_meta\.txt$", f), readdir(DUMPDIR));
    by = f -> parse(Int, match(r"np(\d+)", f)[1]))
isempty(metas) && error("no dump_np*_meta.txt found in $DUMPDIR")
println("dumps: ", join(metas, ", "), " | config q=$Q ell_x=$ELLX P=$P ",
    "Rg=$RG reg=$REG wake_tag=$WAKE_TAG | threads=$(nthreads())")

@printf("\n%-8s %-6s %-6s | %12s %12s %12s %12s %12s | %7s %6s %8s\n",
    "np", "ncol", "nsamp", "dev_vs_dns", "hfmm_vs_dns", "dev_vs_hfmm",
    "proto_vs_dns", "proto_vs_dev", "n_m2l", "n_dem", "t_dense")
for meta in metas
    np = parse(Int, match(r"np(\d+)", meta)[1])
    pre = joinpath(DUMPDIR, "dump_np$np")
    positions = readmat(pre * "_positions_3xN_f64.bin", 3)
    hostU = readmat(pre * "_hostU_3xN_f64.bin", 3)
    devfile = pre * "_deviceU_3xN_f64.bin"
    deviceU = isfile(devfile) ? readmat(devfile, 3) : nothing
    @assert size(positions, 2) == np && size(hostU, 2) == np
    srcmat = readmat(pre * "_body1_srcmat_17xS_f64.bin", 17)
    cent = readmat(pre * "_body1_cent_3xS_f64.bin", 3)
    wakefile = pre * "_body1_wakemat_8xS_f64.bin"
    wakemat = isfile(wakefile) ? readmat(wakefile, 8) : nothing
    E, sortpos = expand_columns(srcmat, wakemat, cent)

    # production padded root box (_cross_root_box: panel verts + particles,
    # h0 = 0.75 * max extent + 1e-9)
    lo = zeros(3); hi = zeros(3)
    for a in 1:3
        pv = extrema(view(positions, a, :))
        sv = extrema(vcat(vec(srcmat[2 + a, :]), vec(srcmat[5 + a, :]),
            vec(srcmat[8 + a, :])))
        lo[a] = min(pv[1], sv[1]); hi[a] = max(pv[2], sv[2])
    end
    h0 = 0.75 * maximum(hi .- lo) + 1e-9
    g = CrossGrid{Float64}(SVector{3,Float64}((lo .+ hi) ./ 2 .- h0), h0)

    particles = [SVector{3,Float64}(positions[:, i]) for i in 1:np]
    Random.seed!(39)
    idx = np <= NSAMPLE ? collect(1:np) : sort(shuffle(1:np)[1:NSAMPLE])

    Udense = zeros(3, length(idx))
    t_dense = @elapsed @threads for s in eachindex(idx)
        Udense[:, s] .= direct_cols(E, 1:size(E, 2), particles[idx[s]])
    end
    Ucross, n_m2l, n_dem = run_cross(E, sortpos, g, particles, idx)
    Uh = hostU[:, idx]
    Ud = deviceU === nothing ? nothing : deviceU[:, idx]

    @printf("%-8d %-6d %-6d | %12s %12.3e %12s %12.3e %12s | %7d %6d %8.1f\n",
        np, size(E, 2), length(idx),
        Ud === nothing ? "-" : @sprintf("%.3e", relL2(Ud, Udense)),
        relL2(Uh, Udense),
        Ud === nothing ? "-" : @sprintf("%.3e", relL2(Ud, Uh)),
        relL2(Ucross, Udense),
        Ud === nothing ? "-" : @sprintf("%.3e", relL2(Ucross, Ud)),
        n_m2l, n_dem, t_dense)
    flush(stdout)
end
println("DONE")
