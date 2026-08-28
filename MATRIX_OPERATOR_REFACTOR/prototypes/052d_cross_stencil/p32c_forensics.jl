# P3.2c — error decomposition for the P-independent floor (follow-up to
# p32b_diag.jl, which REFUTED the panel-extent hypothesis).
#
# Decomposition, exact by construction (near-field pairs identical between
# harness and reference, so they cancel):
#   U_far_reg  = Uref - U_near            (regularized far field, from ref)
#   U_far_sing = Uref_sing - U_near_sing  (singular-kernel far field, from a
#                                          body with core_size ~ 0)
#   total far error   = U_fmm_far - U_far_reg
#   kernel mismatch   = U_far_sing - U_far_reg   (P-independent by nature)
#   truncation        = U_fmm_far - U_far_sing   (must -> 0 with P if the
#                                                 harness is correct)
# Configs: (q=12, ell_x=5) and (q=3, ell_x=7), P in {3, 8}.
# Also prints the top-10 worst targets' per-component errors.
#
# Run: JULIA_NUM_THREADS=4 julia --project=../FLOWPanel.jl p32c_forensics.jl

include(joinpath(@__DIR__, "CrossStencil.jl"))
using .CrossStencil
import FLOWPanel as pnl
import FastMultipole
const FM = FastMultipole
using StaticArrays, Random, Statistics, Printf, LinearAlgebra

const SNAPDIR = "/private/tmp/claude-502/-Users-ryan-Dropbox-research-projects-FastMultipole/1a9c539a-6d87-44ee-87f7-d4e1c17d2793/scratchpad/snapshot472"
const NSAMPLE = 5000

read3(f) = collect(reshape(reinterpret(Float64, read(joinpath(SNAPDIR, f))), 3, :))
positions = read3("particle_positions_3xN_f64.bin")
vpts = read3("panel_vertices_3xM_f64.bin")
conn = collect(reinterpret(Int64, read(joinpath(SNAPDIR, "panel_connectivity_i64.bin"))))
cells = collect(reshape(conn, 3, :))
nt = size(positions, 2)
ns = size(cells, 2)
particles = [SVector{3,Float64}(positions[:, i]) for i in 1:nt]
centroids = [SVector{3,Float64}((vpts[:, cells[1, k]] .+ vpts[:, cells[2, k]] .+
              vpts[:, cells[3, k]]) ./ 3) for k in 1:ns]
g = CrossGrid(particles)

function make_body(cst)
    kernel = Union{pnl.ConstantSource, pnl.VortexRing}
    b = pnl.RigidWakeBody{kernel}(vpts, cells, pnl.noshedding; watertight=true,
        DBC=true, core_size_panel=0.12e-10, core_size_targets=cst)
    pnl.calc_normals!(b)
    pnl.calc_controlpoints!(b)
    pnl._set_core_sizes!((b,), :core_size_targets)
    Random.seed!(472)
    b.strength .= 0.1 .* randn(b.ncells, size(b.strength, 2))
    return b
end

cellcenter(g, c::SVector{3,Int}, L) = begin
    h = g.h0 / (1 << L)
    SVector(g.x_min[1] + (2c[1] + 1) * h, g.x_min[2] + (2c[2] + 1) * h,
            g.x_min[3] + (2c[3] + 1) * h)
end
decode3(code) = SVector{3,Int}(Int.(CrossStencil.SharedRadix.morton_decode(code))...)
dummy_branch(center) = FM.Branch(1:1, 0, 1:0, 0, 1, center, 0.0, SVector(0.0, 0.0, 0.0))
const ds = FM.DerivativesSwitch(false, true, false)

"FMM far field only (no near add) + the near-field matrix for `body`."
function run_far_and_near(body, sbuf, q, ell_x, P, idx, lhv)
    tq = CrossStencil.UniformQTables(q)
    npush = length(tq.push_offsets)
    member = falses(8, npush)
    for ph in 1:8, k in tq.by_phase[ph]
        member[ph, k] = true
    end
    src_levels, sorder, _ = build_level_cells(g, centroids, ell_x)
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
    n_m2l = 0
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
                n_m2l += 1
            end
        end
    end
    Ufar = zeros(3, length(idx))
    for (bcode, ss) in anc[ell_x + 1]
        bl = get(locals_[ell_x + 1], bcode, nothing)
        bl === nothing && continue
        ctr = cellcenter(g, decode3(bcode), ell_x)
        for s in ss
            Δx = particles[idx[s]] - ctr
            _, grad, _ = FM.evaluate_local(Δx, harmonics, gnm, bl, P, lhv, ds)
            Ufar[:, s] .= grad
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
    return Ufar, tb[4:6, :]
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

body = make_body(1e-3)          # production regularized
body_s = make_body(1e-12)       # effectively singular kernel
lhv = Val(FM.has_vector_potential((body,)))
sbuf = FM.system_to_buffer(body)
sbuf_s = FM.system_to_buffer(body_s)
Uref = dense_ref(body, idx)
Uref_s = dense_ref(body_s, idx)
refnorm = sqrt(mean(abs2, Uref))
@printf("dense refs done: rms|U| reg=%.4e sing=%.4e | global reg-vs-sing relRMS=%.3e\n\n",
        refnorm, sqrt(mean(abs2, Uref_s)), sqrt(mean(abs2, Uref_s .- Uref)) / refnorm)

rel(X) = sqrt(mean(abs2, X)) / refnorm
for (q, ell_x) in ((12, 5), (3, 7))
    for P in (3, 8)
        Ufar, Unear = run_far_and_near(body, sbuf, q, ell_x, P, idx, lhv)
        _, Unear_s = run_far_and_near(body_s, sbuf_s, q, ell_x, min(P, 3), idx, lhv)
        # (far pass of the singular body is not needed; reuse near pairs only —
        #  P of that call is irrelevant to Unear_s, use 3 to save time)
        Ufar_reg = Uref .- Unear
        Ufar_sing = Uref_s .- Unear_s
        e_tot = rel(Ufar .- Ufar_reg)
        e_kernel = rel(Ufar_sing .- Ufar_reg)
        e_trunc = rel(Ufar .- Ufar_sing)
        @printf("q=%-3d ellx=%d P=%d : farErr=%.3e | kernel_mismatch=%.3e | truncation=%.3e\n",
                q, ell_x, P, e_tot, e_kernel, e_trunc)
        if P == 8
            e = vec(sqrt.(sum(abs2, Ufar .- Ufar_reg; dims=1)))
            ord = sortperm(e; rev=true)
            println("  top-10 worst targets (P=8): |farErr|  |kernel|  |trunc|  dist_to_nearest_centroid")
            for t in ord[1:10]
                dk = norm(Ufar_sing[:, t] .- Ufar_reg[:, t])
                dt = norm(Ufar[:, t] .- Ufar_sing[:, t])
                d = minimum(norm(particles[idx[t]] - c) for c in centroids)
                @printf("    %.3e  %.3e  %.3e  %.4e\n", e[t], dk, dt, d)
            end
        end
        flush(stdout)
    end
end
println("DONE")
