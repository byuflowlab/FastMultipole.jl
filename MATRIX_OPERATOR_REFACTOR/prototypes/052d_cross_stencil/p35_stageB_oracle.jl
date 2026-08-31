# P3.5 — Stage-B device panel-B2M + M2M oracle (052d Step 4, D1 device-native).
#
# Parity-checks the DEVICE cross-pass expansion pass (src/cross_stencil_cuda.jl
# refresh_cross_multipoles!: thread-per-panel B2M off the 17-row buffer +
# dense octant-class M2M) against an independent HOST reference:
# body_to_multipole_panel!(Panel{SourceDipole}, ...) per panel per leaf cell,
# then multipole_to_multipole! walked up the code tree (the p32 pattern) —
# NOT via cross_m2m_operators, so the dense-operator path is validated
# end-to-end. Pass criterion: per-node relative coefficient error <= 1e-10
# (device atomics reorder the panel sum; recurrences are order-identical).
#
# Case 1: real step-472 snapshot, tag 4 (Source+VortexRing->SourceDipole),
#         seeded strengths, expect skipped == 0.
# Case 2: synthetic mixed buffer — all five tags (tag 3 = pure vortex ring,
#         processed through the dipole arm since the Step-4 review fix), tris +
#         quads, plus one tag-3 nv=2 open-filament column that must be SKIPPED
#         and excluded host-side.
# Case 3: TE-wake dipole triangles — a small finite-wake RigidWakeBody strip;
#         the HOST REFERENCE IS THE RigidWakeBody body_to_multipole! OVERLOAD
#         ITSELF (FLOWPanel_liftingbody.jl:712-782) called per leaf node over
#         the real FMM source buffer, so the device wake arm is checked against
#         production semantics, not a transcription. REQUIRES FLOWPanel in the
#         environment (the seam depot: JULIA_DEPOT_PATH=/private/tmp/
#         flowpanel-052b-depot:... with the FLOWPanel.jl project active).
#
# GPU-only: runs in the Step-5 sbatch. Exit code 0 iff all checks pass.

include(joinpath(@__DIR__, "CrossStencil.jl"))
using .CrossStencil
using FastMultipole
const FM = FastMultipole
using StaticArrays, Printf, Random, LinearAlgebra

const SNAPDIR = get(ENV, "SNAPDIR",
    "/private/tmp/claude-502/-Users-ryan-Dropbox-research-projects-FastMultipole/1a9c539a-6d87-44ee-87f7-d4e1c17d2793/scratchpad/snapshot472")
const P = 6
const ELL_X = 5
const Q = 12
const RG = 0.006
const TOL = 1e-10

FM.load_cuda_radix_lifecycle!() ||
    error("CUDA radix lifecycle failed to load: $(FM.cuda_radix_status())")
const CUDA = FM.CUDA

npass = 0; nfail = 0
function check(name, ok)
    global npass, nfail
    ok ? (npass += 1) : (nfail += 1)
    @printf("  %-58s %s\n", name, ok ? "PASS" : "FAIL")
    return ok
end

"Node center from its morton key (occupancy download convention)."
function host_center_of(occ, x_min, h0, node, L)
    cx, cy, cz = CrossStencil.SharedRadix.morton_decode(occ.node_keys[node])
    delta = (2 * h0) / (1 << L)
    return SVector(x_min[1] + delta * (Int(cx) + 0.5),
        x_min[2] + delta * (Int(cy) + 0.5), x_min[3] + delta * (Int(cz) + 0.5))
end

"Shared host scaffold: fresh per-node expansions + a leaf-B2M callback, then
the production multipole_to_multipole! walked up the code tree (p32 pattern)."
function host_expansions(leaf_b2m!, lists, x_min, h0, ell_x)
    occ = lists.panels
    FM.update_Hs_π2!(FM.Hs_π2, P)
    FM.update_ζs_mag!(FM.ζs_mag, P)
    harmonics = FM.initialize_harmonics(P)
    w1 = FM.initialize_expansion(P); w2 = FM.initialize_expansion(P)
    Ts = zeros(FM.length_Ts(P)); eimϕs = zeros(2, P + 1)
    n_nodes = occ.level_offsets[end]
    exps = [FM.initialize_expansion(P) for _ in 1:n_nodes]
    center_of(node, L) = host_center_of(occ, x_min, h0, node, L)
    # B2M at the leaves
    leaf_first = occ.level_offsets[ell_x + 1] + 1
    for node in leaf_first:occ.level_offsets[ell_x + 2]
        ctr = center_of(node, ell_x)
        r0 = Int(occ.node_ranges[1, node])
        cols = [Int(occ.perm[s]) for s in r0:r0 + Int(occ.node_ranges[2, node]) - 1]
        leaf_b2m!(exps[node], ctr, cols, harmonics)
    end
    # M2M up the code tree
    for Lc in ell_x:-1:1
        for child in occ.level_offsets[Lc + 1] + 1:occ.level_offsets[Lc + 2]
            pkey = occ.node_keys[child] >> 3
            prange = occ.level_offsets[Lc] + 1:occ.level_offsets[Lc + 1]
            parent = prange[searchsortedfirst(view(occ.node_keys, prange), pkey)]
            cb = FM._cross_dummy_branch(center_of(child, Lc))
            pb = FM._cross_dummy_branch(center_of(parent, Lc - 1))
            FM.multipole_to_multipole!(exps[parent], pb, exps[child], cb, w1, w2,
                Ts, eimϕs, FM.ζs_mag, FM.Hs_π2, P, Val(false))
        end
    end
    return exps
end

"Host reference: per-panel B2M + code-tree M2M over the DOWNLOADED occupancy."
function host_reference(pbuf, lists, x_min, h0, ell_x)
    return host_expansions(lists, x_min, h0, ell_x) do exp_node, ctr, cols, harmonics
        for col in cols
            tag = Int(pbuf[1, col]); nv = Int(pbuf[2, col])
            1 <= tag <= 5 && nv >= 3 || continue
            s1 = pbuf[15, col]; s2 = pbuf[16, col]
            vs = [SVector(pbuf[3 + 3 * (k - 1), col], pbuf[4 + 3 * (k - 1), col],
                pbuf[5 + 3 * (k - 1), col]) for k in 1:4]
            tris = nv == 4 ? ((vs[1], vs[2], vs[3]), (vs[1], vs[3], vs[4])) :
                ((vs[1], vs[2], vs[3]),)
            for (t1, t2, t3) in tris
                x0 = t1 - ctr; xu = t2 - t1; xv = t3 - t1
                nrm = normalize(cross(xu, xv))
                if tag == 1
                    FM.body_to_multipole_panel!(FM.Panel{FM.Source}, exp_node,
                        harmonics, x0, xu, xv, nrm, SVector(s1), P)
                elseif tag == 2 || tag == 3
                    # tag 3: closed vortex ring == dipole panel of strength s1
                    # (the pure-VortexRing body_to_multipole! overload)
                    FM.body_to_multipole_panel!(FM.Panel{FM.Dipole}, exp_node,
                        harmonics, x0, xu, xv, nrm, SVector(s1), P)
                else
                    FM.body_to_multipole_panel!(FM.Panel{FM.SourceDipole}, exp_node,
                        harmonics, x0, xu, xv, nrm, SVector(s1, s2), P)
                end
            end
        end
    end
end

function compare(name, d_mult, exps, expected_skipped, got_skipped)
    H = ((P + 1) * (P + 2)) >> 1
    n_nodes = length(exps)
    worst = 0.0; worst_node = 0
    for node in 1:n_nodes
        ref = [exps[node][2 - (c & 1), 1, (c + 1) >> 1] for c in 1:2H]
        dev = d_mult[1:2H, node]
        scale = max(norm(ref), 1e-300)
        err = norm(dev - ref) / scale
        err > worst && (worst = err; worst_node = node)
    end
    @printf("  worst per-node rel err: %.3e (node %d of %d)\n", worst, worst_node,
        n_nodes)
    check("$name: per-node coefficients (tol $TOL)", worst <= TOL)
    check("$name: skipped == $expected_skipped", got_skipped == expected_skipped)
end

function run_case(name, pbuf, particles_mat, expected_skipped;
        wakemat=nothing, hostref=host_reference)
    ns = size(pbuf, 2); nt = size(particles_mat, 2)
    particles = [SVector(particles_mat[1, i], particles_mat[2, i],
        particles_mat[3, i]) for i in 1:nt]
    g = CrossGrid(particles)
    ct = CrossStencilTables(Q, ELL_X, g.h0, RG)
    ctx = FM.device_cross_producer_context(ct, SVector{3,Float64}(g.x_min), g.h0,
        ns, nt)
    d_pbuf = CUDA.CuArray{Float64}(pbuf)
    d_part = CUDA.CuArray{Float64}(particles_mat)
    # panel "positions" for keying = centroids (first nv vertices averaged)
    cent = zeros(3, ns)
    for i in 1:ns
        nv = clamp(Int(pbuf[2, i]), 1, 4)
        for k in 1:nv, a in 1:3
            cent[a, i] += pbuf[2 + 3 * (k - 1) + a, i] / nv
        end
    end
    d_cent = CUDA.CuArray{Float64}(cent)
    d_wake = wakemat === nothing ? nothing : CUDA.CuArray{Float64}(wakemat)
    FM.refresh_cross_producers!(ctx, d_cent, d_part)
    xs = FM.device_cross_expansion_state(ctx, P)
    FM.refresh_cross_multipoles!(xs, ctx, d_pbuf, d_wake)
    CUDA.synchronize()
    t = CUDA.@elapsed begin
        FM.refresh_cross_multipoles!(xs, ctx, d_pbuf, d_wake)
        CUDA.synchronize()
    end
    @printf("\n== %s: ns=%d nt=%d | B2M+M2M %.4f s ==\n", name, ns, nt, t)
    lists = FM.download_cross_lists(ctx)
    exps = hostref(pbuf, lists, ctx.x_min, ctx.h0, ELL_X)
    compare(name, Array(xs.d_multipoles), exps, expected_skipped, xs.n_skipped)
end

# ---- case 1: step-472 snapshot, tag 4 ----
read_i64(f) = reinterpret(Int64, read(joinpath(SNAPDIR, f)))
read_3xn(f) = reshape(reinterpret(Float64, read(joinpath(SNAPDIR, f))), 3, :)
pos = read_3xn("particle_positions_3xN_f64.bin")
verts = read_3xn("panel_vertices_3xM_f64.bin")
conn = read_i64("panel_connectivity_i64.bin"); offs = read_i64("panel_offsets_i64.bin")
Random.seed!(472)
ns = length(offs)
pbuf = zeros(17, ns)
let lo = 1
    for k in 1:ns
        hi = offs[k]
        nv = hi - lo + 1
        pbuf[1, k] = 4.0
        pbuf[2, k] = nv
        for j in 1:4
            v = conn[min(lo + j - 1, hi)]   # repeat last vertex, as pack_panels! does
            pbuf[2 + 3 * (j - 1) + 1, k] = verts[1, v]
            pbuf[2 + 3 * (j - 1) + 2, k] = verts[2, v]
            pbuf[2 + 3 * (j - 1) + 3, k] = verts[3, v]
        end
        pbuf[15, k] = randn(); pbuf[16, k] = randn()
        pbuf[17, k] = 1e-3
        lo = hi + 1
    end
end
run_case("case 1 (snapshot, tag 4)", pbuf, pos, 0)

# ---- case 2: synthetic mixed tags (all five, incl. tag-3 vortex rings) +
# quads + one nv=2 open-filament column that must be SKIPPED ----
Random.seed!(99)
ns2 = 500
pbuf2 = zeros(17, ns2)
for k in 1:ns2
    c = randn(3) * 0.3
    tag = (1, 2, 3, 4, 5)[mod1(k, 5)]
    nv = isodd(k) ? 3 : 4
    if k == 13     # tag-3 nv=2 open filament: no dipole equivalent, skipped
        tag = 3
        nv = 2
    end
    pbuf2[1, k] = tag; pbuf2[2, k] = nv
    # planar-ish quad: random triangle + fourth point in the plane
    v1 = c + randn(3) * 0.02; v2 = c + randn(3) * 0.02; v3 = c + randn(3) * 0.02
    v4 = v1 + (v3 - v2)
    for (j, v) in enumerate((v1, v2, v3, nv == 4 ? v4 : v3))
        pbuf2[2 + 3 * (j - 1) + 1, k] = v[1]
        pbuf2[2 + 3 * (j - 1) + 2, k] = v[2]
        pbuf2[2 + 3 * (j - 1) + 3, k] = v[3]
    end
    pbuf2[15, k] = randn(); pbuf2[16, k] = randn(); pbuf2[17, k] = 1e-3
end
run_case("case 2 (synthetic mixed)", pbuf2, randn(3, 20_000) * 0.4, 1)

# ---- case 3: TE-wake dipole triangles (host reference = the overload itself) ----
# A finite-wake RigidWakeBody strip: 2m triangles over an m-segment quad strip
# in [0,1]x[0,m*dy]; every even cell carries the TE edge and sheds. The seam
# inputs (17-row pbuf + 8-row wake matrix) are sliced from the REAL FMM source
# buffer filled by source_system_to_buffer!, and the host reference calls the
# production RigidWakeBody body_to_multipole! overload per leaf node.
import FLOWPanel
const pnl = FLOWPanel

Random.seed!(52)
m = 12
nn = m + 1
wing_nodes = zeros(3, 2nn)
for j in 1:nn
    y = (j - 1) / m * 1.5
    wing_nodes[:, j] .= (0.0, y, 0.05 * randn())        # LE row
    wing_nodes[:, nn + j] .= (1.0, y, 0.05 * randn())   # TE row
end
wing_cells = zeros(Int, 3, 2m)
for j in 1:m
    a, b = j, j + 1              # LE_j, LE_{j+1}
    d, c = nn + j, nn + j + 1    # TE_j, TE_{j+1}
    wing_cells[:, 2j - 1] .= (a, b, d)
    wing_cells[:, 2j] .= (b, c, d)   # carries the TE edge (c, d)
end
# shedding column: [upper cell, TE1 local idx, TE2 local idx, lower cell(-1), -1, -1]
wing_shedding = [Matrix{Int}(reshape(vcat(([2j, 2, 3, -1, -1, -1] for j in 1:m)...), 6, m))]
wing = pnl.RigidWakeBody{Union{pnl.ConstantSource, pnl.VortexRing}}(
    wing_nodes, wing_cells, wing_shedding;
    check_mesh=false, watertight=false, ensure_winding=false,
    semiinfinite_wake=false)
for D in wing.Das
    D .= 0.25 .+ 0.15 .* rand(size(D)...)   # varied, nonzero first wake row
end
pnl.calc_normals!(wing); pnl.calc_controlpoints!(wing)
wing.strength .= randn(size(wing.strength))

ncells3 = wing.ncells
wingbuf = zeros(FM.data_per_body(wing), ncells3)
for i in 1:ncells3
    FM.source_system_to_buffer!(wingbuf, i, wing, i)
end
pbuf3 = zeros(17, ncells3)
for i in 1:ncells3
    pbuf3[1, i] = 4.0; pbuf3[2, i] = 3.0
    for r in 1:9
        pbuf3[2 + r, i] = wingbuf[6 + r, i]   # vertices (rows 7:15, ns=2)
    end
    pbuf3[12:14, i] .= pbuf3[9:11, i]         # repeat last vertex (slot 4)
    pbuf3[15, i] = wingbuf[5, i]; pbuf3[16, i] = wingbuf[6, i]
    pbuf3[17, i] = 1e-3
end
wakemat3 = wingbuf[end-7:end, :]
n_shed = count(>(0), wakemat3[1, :])
check("case 3 setup: $m shedding + $(ncells3 - m) non-shedding panels",
    n_shed == m && ncells3 == 2m)
hostref_wake(pbuf_, lists, x_min, h0, ell_x) =
    host_expansions(lists, x_min, h0, ell_x) do exp_node, ctr, cols, harmonics
        FM.body_to_multipole!(wing, exp_node, wingbuf, ctr, cols, harmonics, P)
    end
run_case("case 3 (TE wake, overload reference)", pbuf3, rand(3, 20_000) .* 4 .- 1.0,
    0; wakemat=wakemat3, hostref=hostref_wake)

# regression: same body WITHOUT the wake matrix must match a wake-suppressed
# overload reference (proves the no-wake path is untouched by the new arm)
hostref_nowake(pbuf_, lists, x_min, h0, ell_x) = begin
    wing.suppress_attached_wake[] = true
    exps = host_expansions(lists, x_min, h0, ell_x) do exp_node, ctr, cols, harmonics
        FM.body_to_multipole!(wing, exp_node, wingbuf, ctr, cols, harmonics, P)
    end
    wing.suppress_attached_wake[] = false
    return exps
end
run_case("case 3b (no wake matrix)", pbuf3, rand(3, 20_000) .* 4 .- 1.0, 0;
    hostref=hostref_nowake)

@printf("\nP3.5 Stage-B oracle: %d PASS, %d FAIL\n", npass, nfail)
exit(nfail == 0 ? 0 : 1)
