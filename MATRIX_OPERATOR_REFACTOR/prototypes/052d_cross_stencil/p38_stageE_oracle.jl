# P3.8 — Stage-E block-sparse near-field oracle (052d Step 4).
#
# Parity-checks the DEVICE near pass (src/cross_stencil_cuda.jl
# apply_cross_near!: one CUDA block per direct block — demoted ++ near —
# 26-row shared tiles, production _rect_panel_pair/_rect_panel_potential pair
# math, LineGauss Val(4)) against a HOST reference that loops the downloaded
# block lists with the SAME host pair functions (summation order differs →
# tol 1e-10 relRMS). The far field is left zeroed so only the near pass is
# compared.
#
# Case 1: real step-472 snapshot, tag 4, no wake.
# Case 2: synthetic mixed buffer + wake matrix + per-panel shift, wake_tag=3
#         (tri vortex ring) — the production Union{Source,VortexRing} path.
# Case 3: same inputs with wake_tag=2 (tri doublet wake kernel).
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
const REG = 4   # LineGauss

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

"Host near-field reference over the downloaded block lists."
function host_near(pbuf, wakemat, shift, wake_tag, lists, particles_mat)
    pocc = lists.panels
    tocc = lists.particles
    nt = size(particles_mat, 2)
    out = zeros(4, nt)
    b = lists.blocks
    for i in eachindex(b.levels)
        tnode = Int(b.targets[i]); snode = Int(b.sources[i])
        tfirst = Int(tocc.node_ranges[1, tnode])
        tcount = Int(tocc.node_ranges[2, tnode])
        sfirst = Int(pocc.node_ranges[1, snode])
        scount = Int(pocc.node_ranges[2, snode])
        for ti in tfirst:tfirst + tcount - 1
            col = Int(tocc.perm[ti])
            target = SVector{3,Float64}(particles_mat[1, col],
                particles_mat[2, col], particles_mat[3, col])
            p = 0.0
            u = zero(SVector{3,Float64})
            for si in sfirst:sfirst + scount - 1
                scol = Int(pocc.perm[si])
                tag = Int(pbuf[1, scol]); nv = Int(pbuf[2, scol])
                (1 <= tag <= 5 && nv >= 3) || continue
                v1 = SVector{3,Float64}(pbuf[3, scol], pbuf[4, scol], pbuf[5, scol])
                v2 = SVector{3,Float64}(pbuf[6, scol], pbuf[7, scol], pbuf[8, scol])
                v3 = SVector{3,Float64}(pbuf[9, scol], pbuf[10, scol], pbuf[11, scol])
                v4 = SVector{3,Float64}(pbuf[12, scol], pbuf[13, scol], pbuf[14, scol])
                s1 = pbuf[15, scol]; s2 = pbuf[16, scol]; koff = pbuf[17, scol]
                uq, _ = FM._rect_panel_pair(FM.RectangularPanelInfluence(),
                    target, tag, nv, v1, v2, v3, v4, s1, s2, koff,
                    Val(false), Val(REG))
                u += uq
                p += FM._rect_panel_potential(target, tag, nv, v1, v2, v3, s1, s2)
                if wakemat !== nothing
                    idx1 = Int(wakemat[1, scol])
                    if idx1 > 0
                        idx2 = Int(wakemat[5, scol])
                        vs = (v1, v2, v3)
                        w1 = vs[idx1]; w2 = vs[idx2]
                        v1w = w1 + SVector(wakemat[2, scol], wakemat[3, scol],
                            wakemat[4, scol])
                        v2w = w2 + SVector(wakemat[6, scol], wakemat[7, scol],
                            wakemat[8, scol])
                        mu = (tag == 2 ? s1 : s2) +
                            (shift === nothing ? 0.0 : shift[scol])
                        for (a, bb, cc) in ((w1, w2, v1w), (v1w, w2, v2w))
                            uw, _ = FM._rect_panel_pair(
                                FM.RectangularPanelInfluence(), target,
                                wake_tag, 3, a, bb, cc, cc, mu, 0.0, koff,
                                Val(false), Val(REG))
                            u += uw
                            p += FM._rect_panel_potential(target, wake_tag, 3,
                                a, bb, cc, mu, 0.0)
                        end
                    end
                end
            end
            out[1, col] += p
            out[2, col] += u[1]; out[3, col] += u[2]; out[4, col] += u[3]
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

function run_case(name, pbuf, particles_mat; wakemat=nothing, shift=nothing,
        wake_tag=3)
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
    ls = FM.device_cross_local_state(ctx, P)   # d_out starts zeroed; far skipped
    d_wake = wakemat === nothing ? nothing : CUDA.CuArray{Float64}(wakemat)
    d_shift = shift === nothing ? nothing : CUDA.CuArray{Float64}(shift)
    FM.apply_cross_near!(ls, ctx, d_pbuf, d_part; d_wake_buffer=d_wake,
        d_wake_shift=d_shift, wake_tag=wake_tag, reg=REG)
    CUDA.synchronize()
    fill!(ls.d_out, 0.0)
    t = CUDA.@elapsed begin
        FM.apply_cross_near!(ls, ctx, d_pbuf, d_part; d_wake_buffer=d_wake,
            d_wake_shift=d_shift, wake_tag=wake_tag, reg=REG)
        CUDA.synchronize()
    end
    @printf("\n== %s: ns=%d nt=%d blocks=%d (demoted %d) | near %.4f s ==\n",
        name, ns, nt, ctx.n_blocks, ctx.n_demoted, t)
    lists = FM.download_cross_lists(ctx)
    ref = host_near(pbuf, wakemat, shift, wake_tag, lists, particles_mat)
    dev = Array(ls.d_out)
    eu = rel_rms(view(dev, 1, :), view(ref, 1, :))
    eg = rel_rms(view(dev, 2:4, :), view(ref, 2:4, :))
    @printf("  relRMS potential %.3e | velocity %.3e\n", eu, eg)
    check("$name: potential relRMS (tol $TOL)", eu <= TOL)
    check("$name: velocity relRMS (tol $TOL)", eg <= TOL)
end

# ---- case 1: step-472 snapshot, tag 4, no wake ----
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
run_case("case 1 (snapshot, tag 4, no wake)", pbuf, pos)

# ---- cases 2/3: synthetic mixed + wake matrix + shift ----
Random.seed!(99)
ns2 = 500
pbuf2 = zeros(17, ns2)
wakemat2 = zeros(8, ns2)
shift2 = zeros(ns2)
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
    if nv == 3 && k % 3 == 0    # a third of the tris shed a wake
        wakemat2[1, k] = rand(1:3)
        wakemat2[5, k] = mod1(Int(wakemat2[1, k]) + 1, 3)
        wakemat2[2:4, k] .= 0.1 .+ 0.1 .* rand(3)
        wakemat2[6:8, k] .= 0.1 .+ 0.1 .* rand(3)
        shift2[k] = 0.1 * randn()
    else
        wakemat2[1, k] = -1.0
        wakemat2[5, k] = -1.0
    end
end
part2 = randn(3, 20_000) * 0.4
run_case("case 2 (mixed + wake, ring tag 3)", pbuf2, part2;
    wakemat=wakemat2, shift=shift2, wake_tag=3)
run_case("case 3 (mixed + wake, doublet tag 2)", pbuf2, part2;
    wakemat=wakemat2, shift=shift2, wake_tag=2)

@printf("\nP3.8 Stage-E oracle: %d PASS, %d FAIL\n", npass, nfail)
exit(nfail == 0 ? 0 : 1)
