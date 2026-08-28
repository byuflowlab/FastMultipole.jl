# P3.2b — root-cause diagnostic for the P-independent error floor seen in
# p32_harness.jl (no config cleared 1e-4; error grows with ell_x, flat in P).
#
# Hypothesis: PANEL EXTENT vs cell width — panels overhanging their grid cell
# make the innermost rigid-stencil shells marginal/divergent (finite body
# reach, same mechanism as the task-032 sigma-adequacy gate), producing a
# P-independent floor that worsens as cells shrink.
#
# Three parts:
#  A. Geometry: panel circumradius + per-cell overhang stats vs cell width.
#  B. Forensics: per-target error percentiles for two configs; worst targets'
#     distance to nearest panel centroid in cell widths.
#  C. Control: same pipeline with panels SHRUNK 50x about their centroids
#     (same centroids/strengths; dense reference recomputed on the shrunk
#     body). If P-convergence reappears, extent is the root cause.
#
# Run: JULIA_NUM_THREADS=4 julia --project=../FLOWPanel.jl p32b_diag.jl

include(joinpath(@__DIR__, "CrossStencil.jl"))
using .CrossStencil
import FLOWPanel as pnl
import FastMultipole
const FM = FastMultipole
using StaticArrays, Random, Statistics, Printf, LinearAlgebra

const SNAPDIR = "/private/tmp/claude-502/-Users-ryan-Dropbox-research-projects-FastMultipole/1a9c539a-6d87-44ee-87f7-d4e1c17d2793/scratchpad/snapshot472"
const NSAMPLE = 5000
const SHRINK = 0.02

read3(f) = collect(reshape(reinterpret(Float64, read(joinpath(SNAPDIR, f))), 3, :))
positions = read3("particle_positions_3xN_f64.bin")
vpts0 = read3("panel_vertices_3xM_f64.bin")
conn = collect(reinterpret(Int64, read(joinpath(SNAPDIR, "panel_connectivity_i64.bin"))))
cells = collect(reshape(conn, 3, :))
nt = size(positions, 2)
ns = size(cells, 2)
particles = [SVector{3,Float64}(positions[:, i]) for i in 1:nt]

function make_body(vpts)
    kernel = Union{pnl.ConstantSource, pnl.VortexRing}
    b = pnl.RigidWakeBody{kernel}(vpts, cells, pnl.noshedding; watertight=true,
        DBC=true, core_size_panel=0.12e-10, core_size_targets=1e-3)
    pnl.calc_normals!(b)
    pnl.calc_controlpoints!(b)
    pnl._set_core_sizes!((b,), :core_size_targets)
    Random.seed!(472)
    b.strength .= 0.1 .* randn(b.ncells, size(b.strength, 2))
    return b
end

centroids = [SVector{3,Float64}((vpts0[:, cells[1, k]] .+ vpts0[:, cells[2, k]] .+
              vpts0[:, cells[3, k]]) ./ 3) for k in 1:ns]
circumr = [maximum(norm(SVector{3,Float64}(vpts0[:, cells[v, k]]) - centroids[k])
           for v in 1:3) for k in 1:ns]
g = CrossGrid(particles)

# ---------------- Part A: geometry ----------------
println("=== A. panel extent vs cell width ===")
@printf("panel circumradius: max=%.3e p99=%.3e p90=%.3e median=%.3e\n",
        maximum(circumr), quantile(circumr, 0.99), quantile(circumr, 0.9),
        quantile(circumr, 0.5))
for ell_x in (5, 6, 7, 8)
    w = 2 * g.h0 / (1 << ell_x)
    # per-panel overhang beyond its cell: reach of farthest vertex past faces
    over = zeros(ns)
    for k in 1:ns
        c, _ = CrossStencil.level_coords(g, centroids[k], ell_x)
        lo = SVector(g.x_min[1] + c[1] * w, g.x_min[2] + c[2] * w, g.x_min[3] + c[3] * w)
        o = 0.0
        for v in 1:3
            p = SVector{3,Float64}(vpts0[:, cells[v, k]])
            for a in 1:3
                o = max(o, lo[a] - p[a], p[a] - (lo[a] + w))
            end
        end
        over[k] = o
    end
    @printf("ell_x=%d w=%.3e | overhang/w: max=%.2f p99=%.2f frac>0=%.2f  (circumr/w max=%.2f p99=%.2f)\n",
            ell_x, w, maximum(over) / w, quantile(over, 0.99) / w,
            count(>(0), over) / ns, maximum(circumr) / w, quantile(circumr, 0.99) / w)
end
println()

# ---------------- shared pipeline (copy of p32_harness core) ----------------
cellcenter(g, c::SVector{3,Int}, L) = begin
    h = g.h0 / (1 << L)
    SVector(g.x_min[1] + (2c[1] + 1) * h, g.x_min[2] + (2c[2] + 1) * h,
            g.x_min[3] + (2c[3] + 1) * h)
end
decode3(code) = SVector{3,Int}(Int.(CrossStencil.SharedRadix.morton_decode(code))...)
dummy_branch(center) = FM.Branch(1:1, 0, 1:0, 0, 1, center, 0.0, SVector(0.0, 0.0, 0.0))
const ds = FM.DerivativesSwitch(false, true, false)

"Returns per-target 3xN FMM velocity for the sampled targets."
function run_config(body, sbuf, cents, q, ell_x, P, idx, lhv)
    tq = CrossStencil.UniformQTables(q)
    npush = length(tq.push_offsets)
    member = falses(8, npush)
    for ph in 1:8, k in tq.by_phase[ph]
        member[ph, k] = true
    end
    src_levels, sorder, _ = build_level_cells(g, cents, ell_x)
    anc = [Dict{UInt64,Vector{Int}}() for _ in 0:ell_x]
    for (s, j) in enumerate(idx)
        c, _ = CrossStencil.level_coords(g, particles[j], ell_x)
        code = CrossStencil.SharedRadix.morton_encode(UInt64(c[1]), UInt64(c[2]), UInt64(c[3]))
        for L in 0:ell_x
            push!(get!(Vector{Int}, anc[L + 1], code >> (3 * (ell_x - L))), s)
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

    mult = [Dict{UInt64,Array{Float64,3}}() for _ in 0:ell_x]
    leaf = src_levels[ell_x + 1]
    for ci in 1:length(leaf.codes)
        e = get!(() -> FM.initialize_expansion(P), mult[ell_x + 1], leaf.codes[ci])
        ctr = cellcenter(g, decode3(leaf.codes[ci]), ell_x)
        for k in leaf.starts[ci]:leaf.starts[ci + 1] - 1
            i = sorder[k]
            FM.body_to_multipole!(body, e, sbuf, ctr, i:i, harmonics, P)
        end
    end
    for L in ell_x-1:-1:0
        for (ccode, ce) in mult[L + 2]
            pcode = ccode >> 3
            pe = get!(() -> FM.initialize_expansion(P), mult[L + 1], pcode)
            pb = dummy_branch(cellcenter(g, decode3(pcode), L))
            cb = dummy_branch(cellcenter(g, decode3(ccode), L + 1))
            FM.multipole_to_multipole!(pe, pb, ce, cb, w1, w2, Ts, eimϕs,
                FM.ζs_mag, FM.Hs_π2, P, lhv)
        end
    end
    locals_ = [Dict{UInt64,Array{Float64,3}}() for _ in 0:ell_x]
    for L in 2:ell_x
        G = 1 << L
        for (bcode, _) in anc[L + 1]
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
                ae = get(mult[L + 1], acode, nothing)
                ae === nothing && continue
                ab = dummy_branch(cellcenter(g, SVector(ax, ay, az), L))
                FM.multipole_to_local!(bl, bb, ae, ab, w1, w2, w3, Ts, eimϕs,
                    FM.ζs_mag, FM.ηs_mag, FM.Hs_π2, FM.M̃, FM.L̃, P, lhv, nothing)
            end
        end
    end
    U = zeros(3, length(idx))
    for (bcode, ss) in anc[ell_x + 1]
        bl = get(locals_[ell_x + 1], bcode, nothing)
        bl === nothing && continue
        ctr = cellcenter(g, decode3(bcode), ell_x)
        for s in ss
            Δx = particles[idx[s]] - ctr
            _, grad, _ = FM.evaluate_local(Δx, harmonics, gnm, bl, P, lhv, ds)
            U[:, s] .= grad
        end
    end
    tb = zeros(7, length(idx))
    tb[1:3, :] .= positions[:, idx]
    G = 1 << ell_x
    for (bcode, ss) in anc[ell_x + 1]
        bc = decode3(bcode)
        srcids = Int[]
        for o in tq.near_offsets
            ax = bc[1] - o[1]; ay = bc[2] - o[2]; az = bc[3] - o[3]
            (0 <= ax < G && 0 <= ay < G && 0 <= az < G) || continue
            acode = CrossStencil.SharedRadix.morton_encode(UInt64(ax), UInt64(ay), UInt64(az))
            ci = CrossStencil.cell_index(src_levels[ell_x + 1], acode)
            ci == 0 && continue
            lc = src_levels[ell_x + 1]
            append!(srcids, (sorder[k] for k in lc.starts[ci]:lc.starts[ci + 1] - 1))
        end
        isempty(srcids) && continue
        FM.direct!(tb, ss, ds, body, sbuf, srcids)
    end
    U .+= tb[4:6, :]
    return U
end

function dense_ref(body, idx)
    sub = FM.ProbeSystemArray(length(idx))
    sub.position .= positions[:, idx]
    sub.gradient .= 0
    FM.direct!((sub,), (body,); scalar_potential=false, gradient=true, hessian=false)
    return copy(sub.gradient)
end

Random.seed!(99)
idx = sort(shuffle(1:nt)[1:NSAMPLE])
const CONFIGS = ((12, 5, 3), (12, 5, 8), (3, 7, 3), (3, 7, 8))

# ---------------- Part B: real panels, forensics ----------------
println("=== B. real panels: error percentiles + worst-target forensics ===")
body = make_body(vpts0)
lhv = Val(FM.has_vector_potential((body,)))
sbuf = FM.system_to_buffer(body)
Uref = dense_ref(body, idx)
refnorm = sqrt(mean(abs2, Uref))
for (q, ell_x, P) in CONFIGS
    U = run_config(body, sbuf, centroids, q, ell_x, P, idx, lhv)
    e = vec(sqrt.(sum(abs2, U .- Uref; dims=1)))
    relrms = sqrt(mean(abs2, U .- Uref)) / refnorm
    @printf("q=%-3d ellx=%d P=%d : relRMS=%.3e | per-target |err| max=%.3e p99=%.3e p90=%.3e med=%.3e\n",
            q, ell_x, P, relrms, maximum(e), quantile(e, 0.99), quantile(e, 0.9),
            quantile(e, 0.5))
    # error concentration: what fraction of sum-sq error is in the top 10/100 targets?
    es = sort(e; rev=true)
    tot = sum(abs2, e)
    @printf("    error concentration: top10=%.1f%% top100=%.1f%% of sum-sq\n",
            100 * sum(abs2, es[1:10]) / tot, 100 * sum(abs2, es[1:100]) / tot)
    w = 2 * g.h0 / (1 << ell_x)
    ord = sortperm(e; rev=true)
    print("    worst 8 targets (dist to nearest panel centroid, in cell widths): ")
    for t in ord[1:8]
        d = minimum(norm(particles[idx[t]] - c) for c in centroids)
        @printf("%.2f ", d / w)
    end
    println()
end
println()

# ---------------- Part C: shrunk-panel control ----------------
println("=== C. shrunk panels (factor $SHRINK about centroids; same centroids/strengths) ===")
vpts_s = copy(vpts0)
# shrink every panel about its centroid: vertices are shared between panels in
# the mesh, so build an UNSHARED vertex array instead (3 verts per panel)
vpts_s = zeros(3, 3 * ns)
cells_s = reshape(collect(1:3ns), 3, :)
for k in 1:ns, v in 1:3
    p = SVector{3,Float64}(vpts0[:, cells[v, k]])
    vpts_s[:, cells_s[v, k]] .= centroids[k] .+ SHRINK .* (p - centroids[k])
end
kernel = Union{pnl.ConstantSource, pnl.VortexRing}
body_s = pnl.RigidWakeBody{kernel}(vpts_s, cells_s, pnl.noshedding; watertight=false,
    DBC=true, core_size_panel=0.12e-10, core_size_targets=1e-3)
pnl.calc_normals!(body_s)
pnl.calc_controlpoints!(body_s)
pnl._set_core_sizes!((body_s,), :core_size_targets)
Random.seed!(472)
body_s.strength .= 0.1 .* randn(body_s.ncells, size(body_s.strength, 2))
sbuf_s = FM.system_to_buffer(body_s)
Uref_s = dense_ref(body_s, idx)
refnorm_s = sqrt(mean(abs2, Uref_s))
@printf("shrunk dense ref rms|U|=%.4e (real was %.4e)\n", refnorm_s, refnorm)
for (q, ell_x, P) in CONFIGS
    U = run_config(body_s, sbuf_s, centroids, q, ell_x, P, idx, lhv)
    relrms = sqrt(mean(abs2, U .- Uref_s)) / refnorm_s
    @printf("SHRUNK q=%-3d ellx=%d P=%d : relRMS=%.3e\n", q, ell_x, P, relrms)
end
println("DONE")
