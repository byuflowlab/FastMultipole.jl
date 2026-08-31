# P3.7 — Stage-D device L2L + L2B oracle (052d Step 4, D1 device-native).
#
# Parity-checks the DEVICE downward pass (src/cross_stencil_cuda.jl
# finish_cross_locals!: dense octant-class L2L over the particle node set +
# leaf L2B via _resident_local_eval_flat through perm) against an independent
# HOST reference: host multipoles (p35 scaffold) -> production
# multipole_to_local! per downloaded far route -> production local_to_local!
# walked down the particle code tree -> production classic evaluate_local per
# particle. This closes the expansion chain B2M -> M2M -> M2L -> L2L -> L2B
# end-to-end on device (near-field/direct blocks are Stage E).
# Pass criterion: potential and gradient relative RMS <= 1e-10 over particles
# with a nonzero reference.
#
# Case 1: real step-472 snapshot, tag 4 buffer, seeded strengths.
# Case 2: synthetic mixed buffer (p35 case-2 construction).
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

function center_of(occ, x_min, h0, node, L)
    cx, cy, cz = CrossStencil.SharedRadix.morton_decode(occ.node_keys[node])
    delta = (2 * h0) / (1 << L)
    return SVector(x_min[1] + delta * (Int(cx) + 0.5),
        x_min[2] + delta * (Int(cy) + 0.5), x_min[3] + delta * (Int(cz) + 0.5))
end

"Host multipoles over the downloaded panel occupancy (p35 scaffold)."
function host_multipoles(pbuf, lists, x_min, h0, ell_x)
    occ = lists.panels
    FM.update_Hs_π2!(FM.Hs_π2, P)
    FM.update_ζs_mag!(FM.ζs_mag, P)
    harmonics = FM.initialize_harmonics(P)
    w1 = FM.initialize_expansion(P); w2 = FM.initialize_expansion(P)
    Ts = zeros(FM.length_Ts(P)); eimϕs = zeros(2, P + 1)
    n_nodes = occ.level_offsets[end]
    exps = [FM.initialize_expansion(P) for _ in 1:n_nodes]
    leaf_first = occ.level_offsets[ell_x + 1] + 1
    for node in leaf_first:occ.level_offsets[ell_x + 2]
        ctr = center_of(occ, x_min, h0, node, ell_x)
        r0 = Int(occ.node_ranges[1, node])
        for s in r0:r0 + Int(occ.node_ranges[2, node]) - 1
            col = occ.perm[s]
            tag = Int(pbuf[1, col]); nv = Int(pbuf[2, col])
            (1 <= tag <= 5) && nv >= 3 || continue
            s1 = pbuf[15, col]; s2 = pbuf[16, col]
            vs = [SVector(pbuf[3 + 3 * (k - 1), col], pbuf[4 + 3 * (k - 1), col],
                pbuf[5 + 3 * (k - 1), col]) for k in 1:4]
            tris = nv == 4 ? ((vs[1], vs[2], vs[3]), (vs[1], vs[3], vs[4])) :
                ((vs[1], vs[2], vs[3]),)
            for (t1, t2, t3) in tris
                x0 = t1 - ctr; xu = t2 - t1; xv = t3 - t1
                nrm = normalize(cross(xu, xv))
                if tag == 1
                    FM.body_to_multipole_panel!(FM.Panel{FM.Source}, exps[node],
                        harmonics, x0, xu, xv, nrm, SVector(s1), P)
                elseif tag == 2 || tag == 3
                    # tag 3: closed vortex ring == dipole panel of strength s1
                    FM.body_to_multipole_panel!(FM.Panel{FM.Dipole}, exps[node],
                        harmonics, x0, xu, xv, nrm, SVector(s1), P)
                else
                    FM.body_to_multipole_panel!(FM.Panel{FM.SourceDipole}, exps[node],
                        harmonics, x0, xu, xv, nrm, SVector(s1, s2), P)
                end
            end
        end
    end
    for Lc in ell_x:-1:1
        for child in occ.level_offsets[Lc + 1] + 1:occ.level_offsets[Lc + 2]
            pkey = occ.node_keys[child] >> 3
            prange = occ.level_offsets[Lc] + 1:occ.level_offsets[Lc + 1]
            parent = prange[searchsortedfirst(view(occ.node_keys, prange), pkey)]
            cb = FM._cross_dummy_branch(center_of(occ, x_min, h0, child, Lc))
            pb = FM._cross_dummy_branch(center_of(occ, x_min, h0, parent, Lc - 1))
            FM.multipole_to_multipole!(exps[parent], pb, exps[child], cb, w1, w2,
                Ts, eimϕs, FM.ζs_mag, FM.Hs_π2, P, Val(false))
        end
    end
    return exps
end

"Host locals: M2L per route, then L2L walked down the particle code tree."
function host_locals_full(mult_exps, lists, x_min, h0, ell_x)
    pocc = lists.panels
    tocc = lists.particles
    FM.update_ηs_mag!(FM.ηs_mag, P)
    FM.update_M̃!(FM.M̃, P)
    FM.update_L̃!(FM.L̃, P)
    w1 = FM.initialize_expansion(P); w2 = FM.initialize_expansion(P)
    w3 = FM.initialize_expansion(P)
    Ts = zeros(FM.length_Ts(P)); eimϕs = zeros(2, P + 1)
    n_nodes = tocc.level_offsets[end]
    locs = [FM.initialize_expansion(P) for _ in 1:n_nodes]
    r = lists.routes
    for i in eachindex(r.levels)
        L = Int(r.levels[i])
        sb = FM._cross_dummy_branch(center_of(pocc, x_min, h0, Int(r.sources[i]), L))
        tb = FM._cross_dummy_branch(center_of(tocc, x_min, h0, Int(r.targets[i]), L))
        FM.multipole_to_local!(locs[Int(r.targets[i])], tb,
            mult_exps[Int(r.sources[i])], sb, w1, w2, w3, Ts, eimϕs,
            FM.ζs_mag, FM.ηs_mag, FM.Hs_π2, FM.M̃, FM.L̃, P, Val(false), nothing)
    end
    for Lc in 3:ell_x
        for child in tocc.level_offsets[Lc + 1] + 1:tocc.level_offsets[Lc + 2]
            pkey = tocc.node_keys[child] >> 3
            prange = tocc.level_offsets[Lc] + 1:tocc.level_offsets[Lc + 1]
            parent = prange[searchsortedfirst(view(tocc.node_keys, prange), pkey)]
            cb = FM._cross_dummy_branch(center_of(tocc, x_min, h0, child, Lc))
            pb = FM._cross_dummy_branch(center_of(tocc, x_min, h0, parent, Lc - 1))
            FM.local_to_local!(locs[child], cb, locs[parent], pb, w1, w2,
                Ts, eimϕs, FM.ηs_mag, FM.Hs_π2, P, Val(false))
        end
    end
    return locs
end

"Host per-particle outputs: classic evaluate_local at each particle's leaf."
function host_outputs(locs, lists, particles_mat, x_min, h0, ell_x)
    tocc = lists.particles
    nt = size(particles_mat, 2)
    out = zeros(4, nt)
    harmonics = FM.initialize_harmonics(P)
    gradient_n_m = FM.initialize_gradient_n_m(P)
    ds = FM.DerivativesSwitch(true, true, false)
    leaf_first = tocc.level_offsets[ell_x + 1] + 1
    for node in leaf_first:tocc.level_offsets[ell_x + 2]
        ctr = center_of(tocc, x_min, h0, node, ell_x)
        r0 = Int(tocc.node_ranges[1, node])
        for s in r0:r0 + Int(tocc.node_ranges[2, node]) - 1
            col = tocc.perm[s]
            Δx = SVector(particles_mat[1, col], particles_mat[2, col],
                particles_mat[3, col]) - ctr
            u, g, _ = FM.evaluate_local(Δx, harmonics, gradient_n_m, locs[node],
                P, Val(false), ds)
            out[1, col] = u
            out[2, col] = g[1]; out[3, col] = g[2]; out[4, col] = g[3]
        end
    end
    return out
end

function rel_rms(dev, ref)
    num = 0.0; den = 0.0
    for i in eachindex(ref)
        num += (dev[i] - ref[i])^2
        den += ref[i]^2
    end
    return den == 0.0 ? (num == 0.0 ? 0.0 : Inf) : sqrt(num / den)
end

function run_case(name, pbuf, particles_mat)
    ns = size(pbuf, 2); nt = size(particles_mat, 2)
    particles = [SVector(particles_mat[1, i], particles_mat[2, i],
        particles_mat[3, i]) for i in 1:nt]
    g = CrossGrid(particles)
    ct = CrossStencilTables(Q, ELL_X, g.h0, RG)
    ctx = FM.device_cross_producer_context(ct, SVector{3,Float64}(g.x_min), g.h0,
        ns, nt)
    d_pbuf = CUDA.CuArray{Float64}(pbuf)
    d_part = CUDA.CuArray{Float64}(particles_mat)
    cent = zeros(3, ns)
    for i in 1:ns
        nv = clamp(Int(pbuf[2, i]), 1, 4)
        for k in 1:nv, a in 1:3
            cent[a, i] += pbuf[2 + 3 * (k - 1) + a, i] / nv
        end
    end
    d_cent = CUDA.CuArray{Float64}(cent)
    FM.refresh_cross_producers!(ctx, d_cent, d_part)
    xs = FM.device_cross_expansion_state(ctx, P)
    FM.refresh_cross_multipoles!(xs, ctx, d_pbuf)
    ls = FM.device_cross_local_state(ctx, P)
    FM.refresh_cross_locals!(ls, ctx, xs)
    FM.finish_cross_locals!(ls, ctx, d_part)
    CUDA.synchronize()
    t = CUDA.@elapsed begin
        FM.refresh_cross_locals!(ls, ctx, xs)
        FM.finish_cross_locals!(ls, ctx, d_part)
        CUDA.synchronize()
    end
    @printf("\n== %s: ns=%d nt=%d routes=%d | M2L+L2L+L2B %.4f s ==\n",
        name, ns, nt, ctx.n_routes, t)
    lists = FM.download_cross_lists(ctx)
    mult_exps = host_multipoles(pbuf, lists, ctx.x_min, ctx.h0, ELL_X)
    locs = host_locals_full(mult_exps, lists, ctx.x_min, ctx.h0, ELL_X)
    ref = host_outputs(locs, lists, particles_mat, ctx.x_min, ctx.h0, ELL_X)
    dev = Array(ls.d_out)
    eu = rel_rms(view(dev, 1, :), view(ref, 1, :))
    eg = rel_rms(view(dev, 2:4, :), view(ref, 2:4, :))
    @printf("  relRMS potential %.3e | gradient %.3e\n", eu, eg)
    check("$name: potential relRMS (tol $TOL)", eu <= TOL)
    check("$name: gradient relRMS (tol $TOL)", eg <= TOL)
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
            v = conn[min(lo + j - 1, hi)]
            pbuf[2 + 3 * (j - 1) + 1, k] = verts[1, v]
            pbuf[2 + 3 * (j - 1) + 2, k] = verts[2, v]
            pbuf[2 + 3 * (j - 1) + 3, k] = verts[3, v]
        end
        pbuf[15, k] = randn(); pbuf[16, k] = randn()
        pbuf[17, k] = 1e-3
        lo = hi + 1
    end
end
run_case("case 1 (snapshot, tag 4)", pbuf, pos)

# ---- case 2: synthetic mixed tags + quads + skip columns ----
Random.seed!(99)
ns2 = 500
pbuf2 = zeros(17, ns2)
for k in 1:ns2
    c = randn(3) * 0.3
    tag = (1, 2, 4, 5)[mod1(k, 4)]
    nv = isodd(k) ? 3 : 4
    if k == 7
        tag = 3
    elseif k == 13
        nv = 2
    end
    pbuf2[1, k] = tag; pbuf2[2, k] = nv
    v1 = c + randn(3) * 0.02; v2 = c + randn(3) * 0.02; v3 = c + randn(3) * 0.02
    v4 = v1 + (v3 - v2)
    for (j, v) in enumerate((v1, v2, v3, nv == 4 ? v4 : v3))
        pbuf2[2 + 3 * (j - 1) + 1, k] = v[1]
        pbuf2[2 + 3 * (j - 1) + 2, k] = v[2]
        pbuf2[2 + 3 * (j - 1) + 3, k] = v[3]
    end
    pbuf2[15, k] = randn(); pbuf2[16, k] = randn(); pbuf2[17, k] = 1e-3
end
run_case("case 2 (synthetic mixed)", pbuf2, randn(3, 20_000) * 0.4)

@printf("\nP3.7 Stage-D oracle: %d PASS, %d FAIL\n", npass, nfail)
exit(nfail == 0 ? 0 : 1)
