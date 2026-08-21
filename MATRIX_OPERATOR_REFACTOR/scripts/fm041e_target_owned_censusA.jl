#!/usr/bin/env julia
#
# Task 041e Stage A: read-only census and roofline for the target-owned fused
# U/J nearfield kernel.  Local only (<= 4 threads), no production changes, no
# hardware runs.  Outputs bound five components (de-double-counted) of the
# 39-60%-of-ceiling residual and record a CONTINUE/KILL decision:
#   (1) lane underfill recoverable by repacking / source-splitting
#   (2) mixed-predicate divergence under lanes=targets vs lanes=sources
#   (3) per-edge setup / accumulator retirement
#   (4) source/target load reuse (bounded control; DRAM 2-12% of peak)
#   (5) irreducible pair arithmetic (remainder)
#
# Run: julia --project=../../../FLOWVPM.jl --threads=4 \
#          MATRIX_OPERATOR_REFACTOR/scripts/fm041e_target_owned_censusA.jl
# (FLOWVPM project needed only for the fm033 rotor generator.)
#
# REGISTRATION (predeclared): cases cube/wake/sigma_multiscale proxies
# (041c/041d seeds) + real DJI-9443 rotor; n=1e5 exact standalone replay,
# n=1e6 rotor attempted with a time guard (else the 041a leafpop record
# carries the 1e6 occupancy evidence); K_max in {64, 256}; near_radius2=5;
# sticky sigma demotion with rho_t=4.789 (031a J cutoff, the production
# partitioned predicate).  Sampled predicate replay: <=200 seeded target
# leaves per case (seed 41011).  Op-cost model: F_pair regularized = 1.0
# (normalized), singular path cost c_s in {0.4, 0.5, 0.6} of regularized
# (037b measured erf-free regularized ~1.5x singular; brackets opt/nom/pess);
# per-edge per-target-lane setup+retirement overhead in {30, 50, 80} flops
# vs ~130-flop fused U+J pair math (bounded attribution, not exact
# accounting — Review Amendment item 4).  Warp width 32.
# The already-falsified atomic-only store lever (0.13%) is excluded.

using Random
using Statistics
using Printf
using SHA

Threads.nthreads() <= 4 || error("041e Stage A must run on <= 4 threads")

include(joinpath(@__DIR__, "adaptive_octree_verify.jl"))

const HAVE_ROTOR = Ref(false)
try
    include(joinpath(@__DIR__, "benchmark_033_common.jl"))
    HAVE_ROTOR[] = true
catch err
    @warn "FLOWVPM project unavailable; rotor case skipped" err
end

const OUTDIR = joinpath(@__DIR__, "..", "data", "target_owned_nearfield")
mkpath(OUTDIR)

const RHO_T = 4.789
const NEAR_Q = 5
const WARP = 32
const CS_BRK = (0.4, 0.5, 0.6)      # singular path cost / regularized
const OVH_BRK = (30.0, 50.0, 80.0)  # per-edge per-lane setup+retire flops
const F_PAIR = 130.0                # fused U+J regularized pair flops (model)
const SAMPLE_LEAVES = 200
const SEED_SAMPLE = 41011

function case_arrays(name::String, n::Int)
    if name == "cube"
        return make_uniform(n; seed=41003), fill(2.0 * n^(-1 / 3), n)
    elseif name == "wake"
        return make_filament(n; seed=41004), fill(3.15 * n^(-1 / 3), n)
    elseif name == "sigma_multiscale"
        X = make_multiscale(n; contrast=100.0, seed=41006)
        c = (0.6, 0.4, 0.55)
        r = [sqrt(sum((X[d, i] - c[d])^2 for d in 1:3)) for i in 1:n]
        med = median(r)
        return X, [0.08 * n^(-1 / 3) * exp(log(18.0) * (ri <= med)) for ri in r]
    elseif name == "rotor"
        HAVE_ROTOR[] || error("rotor requires FLOWVPM project")
        X = Matrix{Float64}(undef, 3, n)
        σ = Vector{Float64}(undef, n)
        rng = MersenneTwister(FM033_SEED + FM033_ROTOR_SEED_OFFSET + n)
        i = Ref(0)
        fm033_rotor_foreach(n; rng) do x, _, s
            i[] += 1
            X[1, i[]] = x[1]; X[2, i[]] = x[2]; X[3, i[]] = x[3]
            σ[i[]] = s
        end
        @assert i[] == n
        return X, σ
    end
    error("unknown case $name")
end

quantiles_str(v) = isempty(v) ? "0,0,0,0,0" :
    join([@sprintf("%.3g", quantile(float.(v), q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)], ",")

leaf_rows = String[]
pred_rows = String[]
comp_rows = String[]
report_lines = String[]

const CASELIST = HAVE_ROTOR[] ?
    [("cube", 100_000, 64), ("wake", 100_000, 256), ("sigma_multiscale", 100_000, 64),
     ("rotor", 100_000, 64), ("rotor", 100_000, 256), ("rotor", 1_000_000, 64)] :
    [("cube", 100_000, 64), ("wake", 100_000, 256), ("sigma_multiscale", 100_000, 64)]

for (cname, n, K) in CASELIST
    X, σ = case_arrays(cname, n)
    t0 = time()
    t = build_tree(X, K, 12)
    balance!(t)
    snode = sigma_upward(t, σ)
    L = build_lists(t, NEAR_Q; sigma_node=snode, rho_t=RHO_T)
    tbuild = time() - t0
    @printf("[%s n=%d K=%d] tree+lists %.1fs  U=%d V=%d W=%d X=%d\n",
        cname, n, K, tbuild, length(L.U), length(L.V), length(L.W), length(L.X))

    pop(i) = t.nodes[i].hi - t.nodes[i].lo + 1
    # target CSR: group ordered U edges by target leaf
    tgt = Dict{Int,Vector{Int}}()
    for (a, b) in L.U
        push!(get!(tgt, a, Int[]), b)
    end
    leaves = sort!(collect(keys(tgt)))
    nt_v = Int[]; deg_v = Int[]; srcsum_v = Int[]; waves_v = Int[]
    lost = 0.0; slots = 0.0; pairs = 0.0
    edge_lanes = 0.0                    # sum over edges of 32*ceil(nt/32) lane rows
    tgt_loads_rep = 0.0                 # repeated target loads (edges-1 per leaf x nt)
    src_visits = Dict{Int,Int}()        # source leaf -> #target leaves visiting it
    for a in leaves
        na = pop(a); d = length(tgt[a])
        ss = sum(pop(b) for b in tgt[a])
        push!(nt_v, na); push!(deg_v, d); push!(srcsum_v, ss)
        push!(waves_v, cld(na, WARP))
        lost += (WARP * cld(na, WARP) - na) * ss
        slots += WARP * cld(na, WARP) * ss
        pairs += na * ss
        edge_lanes += d * WARP * cld(na, WARP)
        tgt_loads_rep += (d - 1) * na
        for b in tgt[a]
            src_visits[b] = get(src_visits, b, 0) + 1
        end
    end
    underfill = lost / slots
    src_reuse = mean(collect(values(src_visits)))
    push!(leaf_rows, @sprintf("%s,%d,%d,%d,%d,%s,%s,%s,%.4f,%.3f,%.3f,%.1f",
        cname, n, K, length(leaves), length(L.U), quantiles_str(nt_v),
        quantiles_str(deg_v), quantiles_str(srcsum_v), underfill,
        mean(waves_v), tgt_loads_rep / pairs, src_reuse))

    # --- sampled predicate replay under both lane mappings -----------------
    rng = MersenneTwister(SEED_SAMPLE)
    sample = length(leaves) <= SAMPLE_LEAVES ? leaves :
             leaves[sort!(randperm(rng, length(leaves))[1:SAMPLE_LEAVES])]
    # accumulators per cost bracket: base ops and divergence-extra ops
    baseT = zeros(3); extraT = zeros(3)      # lanes=targets (shipped)
    baseS = zeros(3); extraS = zeros(3)      # lanes=sources, r/sigma-sorted
    mixedT = 0.0; totT = 0.0; mixedS = 0.0; totS = 0.0
    spairs = 0.0; sregular = 0.0
    for a in sample
        ta = t.nodes[a]
        tb = [t.perm[i] for i in ta.lo:ta.hi]        # target body ids
        srcs = Int[]
        for b in tgt[a], i in t.nodes[b].lo:t.nodes[b].hi
            push!(srcs, t.perm[i])
        end
        isempty(srcs) && continue
        # per (target body, source body) predicate: regularized iff r <= rho_t*sigma_s
        # lanes=targets: warps over targets in tree order; serial over sources
        nwT = cld(length(tb), WARP)
        predmat = falses(length(tb), length(srcs))
        for (js, s) in enumerate(srcs)
            cut2 = (RHO_T * σ[s])^2
            for (jt, tt) in enumerate(tb)
                r2 = (X[1, tt] - X[1, s])^2 + (X[2, tt] - X[2, s])^2 + (X[3, tt] - X[3, s])^2
                predmat[jt, js] = (r2 <= cut2) && (r2 > 0)
            end
        end
        spairs += length(tb) * length(srcs)
        sregular += count(predmat)
        for w in 1:nwT
            lo = (w - 1) * WARP + 1; hi = min(w * WARP, length(tb))
            for js in eachindex(srcs)
                anyr = any(@view predmat[lo:hi, js])
                anys = !all(@view predmat[lo:hi, js])
                totT += 1
                for br in 1:3
                    cs = CS_BRK[br]
                    baseT[br] += anyr && anys ? 1 + cs : (anyr ? 1.0 : cs)
                    extraT[br] += anyr && anys ? min(1.0, cs) : 0.0
                end
                mixedT += (anyr && anys) ? 1 : 0
            end
        end
        # lanes=sources: per target, sources sorted by r/sigma_s (monotone runs)
        for (jt, tt) in enumerate(tb)
            ord = sortperm([((X[1, tt] - X[1, s])^2 + (X[2, tt] - X[2, s])^2 +
                            (X[3, tt] - X[3, s])^2) / σ[s]^2 for s in srcs])
            nwS = cld(length(srcs), WARP)
            for w in 1:nwS
                lo = (w - 1) * WARP + 1; hi = min(w * WARP, length(srcs))
                seg = @view predmat[jt, ord[lo:hi]]
                anyr = any(seg); anys = !all(seg)
                totS += 1
                for br in 1:3
                    cs = CS_BRK[br]
                    baseS[br] += anyr && anys ? 1 + cs : (anyr ? 1.0 : cs)
                    extraS[br] += anyr && anys ? min(1.0, cs) : 0.0
                end
                mixedS += (anyr && anys) ? 1 : 0
            end
        end
    end
    divT = [extraT[br] / max(baseT[br], 1e-300) for br in 1:3]
    divS = [extraS[br] / max(baseS[br], 1e-300) for br in 1:3]
    push!(pred_rows, @sprintf("%s,%d,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f",
        cname, n, K, length(sample), mixedT / max(totT, 1), mixedS / max(totS, 1),
        divT[1], divT[2], divT[3], divS[1], divS[2], divS[3],
        sregular / max(spairs, 1)))

    # --- components (fractions of current nearfield pair-op cost) ----------
    for br in 1:3
        cs = CS_BRK[br]; ovh = OVH_BRK[br]
        # current cost model per bracket (normalized flop units):
        # pairs*mixture + underfilled lane slots + per-edge overhead
        mix = sregular / max(spairs, 1)
        pair_ops = pairs * (mix + (1 - mix) * cs) * F_PAIR
        under_ops = lost * (mix + (1 - mix) * cs) * F_PAIR * (1 + divT[br])
        div_ops = pair_ops * divT[br]
        edge_ops = edge_lanes * ovh
        cur = pair_ops + under_ops + div_ops + edge_ops
        # shape-4 (lanes=sources, sorted, per-target): underfill on source axis,
        # divergence divS, one edge-setup per target (not per edge)
        u4 = sum(WARP * cld(ss, WARP) - ss for ss in srcsum_v) /
             max(sum(WARP .* cld.(srcsum_v, WARP)), 1)
        shape4 = pairs * (mix + (1 - mix) * cs) * F_PAIR * (1 + divS[br]) / (1 - u4) +
                 length(leaves) * mean(nt_v) * ovh
        # de-double-counted combined ceiling: components 1+2+3 together
        comb = max(0.0, 1 - shape4 / cur)
        c1 = under_ops / cur
        c2 = div_ops / cur
        c3 = edge_ops * (1 - 1 / max(mean(deg_v), 1)) / cur
        c4 = tgt_loads_rep * 12 * 4 / max(cur, 1) # bytes proxy, bounded control
        push!(comp_rows, @sprintf("%s,%d,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f",
            cname, n, K, br, c1, c2, c3, c4, max(0.0, 1 - c1 - c2 - c3), comb))
        br == 1 && push!(report_lines, @sprintf(
            "%s n=%d K=%d: OPT ceiling (1)+(2)+(3) de-double-counted = %.1f%% (c1=%.1f%% c2=%.1f%% c3=%.1f%%)",
            cname, n, K, 100comb, 100c1, 100c2, 100c3))
    end
end

writecsv(path, header, rows) = open(path, "w") do io
    println(io, header)
    foreach(r -> println(io, r), rows)
end

writecsv(joinpath(OUTDIR, "censusA_leaf.csv"),
    "case,n,K,target_leaves,u_edges,nt_q05,nt_q25,nt_q50,nt_q75,nt_q95,deg_q05,deg_q25,deg_q50,deg_q75,deg_q95,srcsum_q05,srcsum_q25,srcsum_q50,srcsum_q75,srcsum_q95,underfill_frac,mean_waves,repeated_tgt_load_per_pair,src_leaf_reuse",
    leaf_rows)
writecsv(joinpath(OUTDIR, "censusA_predicate.csv"),
    "case,n,K,sampled_leaves,mixed_warpslot_frac_T,mixed_warpslot_frac_S,divT_opt,divT_nom,divT_pess,divS_opt,divS_nom,divS_pess,regularized_pair_frac",
    pred_rows)
writecsv(joinpath(OUTDIR, "censusA_components.csv"),
    "case,n,K,bracket,c1_underfill,c2_divergence,c3_edge_overhead,c4_load_reuse_proxy,c5_irreducible,combined_ceiling_123",
    comp_rows)
writecsv(joinpath(OUTDIR, "censusA_manifest.csv"), "key,value",
    ["registration,see script header (predeclared brackets/seeds/op model)",
     "rho_t,$(RHO_T)", "near_q,$(NEAR_Q)", "sample_leaves,$(SAMPLE_LEAVES)",
     "seed,$(SEED_SAMPLE)", "have_rotor,$(HAVE_ROTOR[])",
     "threads,$(Threads.nthreads())", "julia,$(VERSION)"])

open(joinpath(OUTDIR, "censusA_report.txt"), "w") do io
    println(io, "041e Stage A read-only census (2026-08-18)")
    println(io, "Continue rule: optimistic de-double-counted (1)+(2)+(3) >= 10% nearfield")
    println(io, "on a material case (rotor primary; nearfield share 92-96% per 041b => 5%")
    println(io, "complete-solve follows from ~>=5.4% nearfield).")
    println(io, "")
    foreach(l -> println(io, l), report_lines)
    println(io, "")
    println(io, "Notes: op model normalized to F_pair=130 regularized-pair flops;")
    println(io, "singular path bracket $(CS_BRK); per-edge lane overhead bracket $(OVH_BRK);")
    println(io, "atomic-store-only lever excluded (028: 0.13%). c4 is a bytes-based")
    println(io, "bounded control, not additive with c1-c3. 1e6 occupancy evidence for")
    println(io, "cube/wake/multiscale lives in data/fm041a_leafpop.csv (041a).")
end

open(joinpath(OUTDIR, "censusA_checksums.sha256"), "w") do io
    for f in sort(readdir(OUTDIR))
        (startswith(f, "censusA_") && f != "censusA_checksums.sha256") || continue
        println(io, "$(bytes2hex(open(sha256, joinpath(OUTDIR, f))))  $f")
    end
end

println("done -> $OUTDIR")
