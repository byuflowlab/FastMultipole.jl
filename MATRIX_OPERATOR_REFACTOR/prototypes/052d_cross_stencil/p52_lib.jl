# P5.2 — 052h reverse-leg host oracle (CPU-only, no CUDA).
#
# End-to-end semantic check of the particles→panels pipeline: random vortex
# cloud (sources) + random target points, routed by the untouched CrossStencil
# host prototype in the REVERSE direction (particles = explicit sources,
# targets looked up densely on the device — here just the swapped sweep), with
# expansions run through the EXACT math the device reverse leg uses:
#
#   B2M   host body_to_multipole_point!(Point{Vortex}) per leaf cell (LH φ+χ)
#   M2M   cross_m2m_operators_lh stacked matvec (octant class per level)
#   M2L   cross_m2l_operators_lh at the ref level + shifted separable
#         rescaling (cross_m2l_level_scales_lh: χ rows n+1, χ cols n-1)
#   L2L   cross_l2l_operators_lh stacked matvec
#   L2B   _resident_local_eval_flat(phi, chi, ..., Val(true))
#
# plus a hand direct near-field (convention self-calibrated against the
# production fmm.direct! on the full fixture to < 1e-12 first).
#
# PASS criteria: (1) hand-direct convention calibrates; (2) route+near pair
# coverage == ns·nt; (3) fmm(far)+direct(near) matches the production direct
# velocity to expansion accuracy (report max/mean relU; assert < RELU_TOL).
#
# Run: julia --project=<FastMultipole> p52_reverse_host_oracle.jl
#
# The fixture + host pipeline live in p52_lib.jl, shared with the device
# replay p53_reverse_device_replay.jl (GPU-only).

include(joinpath(@__DIR__, "CrossStencil.jl"))
using .CrossStencil
const morton_decode = CrossStencil.SharedRadix.morton_decode
using FastMultipole
using FastMultipole: initialize_expansion, initialize_harmonics,
    body_to_multipole_point!, Point, Vortex,
    cross_m2m_operators_lh, cross_m2l_operators_lh, cross_m2l_level_scales_lh,
    cross_l2l_operators_lh, _resident_local_eval_flat
using StaticArrays, Printf
using LinearAlgebra: norm, cross
import Random

include(joinpath(@__DIR__, "..", "..", "..", "test", "gravitational.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "test", "vortex.jl"))

const Q = 3
const ELL = 4

npass = 0; nfail = 0
function check(name, ok)
    global npass, nfail
    ok ? (npass += 1) : (nfail += 1)
    @printf("  %-58s %s\n", name, ok ? "PASS" : "FAIL")
    return ok
end

# ---- fixture ----
Random.seed!(52)
ns, nt = 500, 400
src_pos = rand(3, ns)
src_str = randn(3, ns) ./ ns
tgt_pos = rand(3, nt)
src_sys = VortexParticles(src_pos, src_str, zeros(ns))
tgt_sys = Gravitational(vcat(tgt_pos, zeros(1, nt), zeros(1, nt)))  # massless probes

# ---- production reference: direct velocity at the targets ----
fmm.direct!(tgt_sys, src_sys)
U_ref = [SVector{3,Float64}(tgt_sys.potential[5:7, i]) for i in 1:nt]

# ---- hand direct kernel, self-calibrated sign ----
u_hand(x, y, Γ, s) = begin
    r = x - y
    nr = norm(r)
    s * cross(Γ, r) / (4π * nr^3)
end
src_x = [SVector{3,Float64}(src_pos[:, j]) for j in 1:ns]
src_G = [SVector{3,Float64}(src_str[:, j]) for j in 1:ns]
tgt_x = [SVector{3,Float64}(tgt_pos[:, i]) for i in 1:nt]
Uh_p = [sum(u_hand(tgt_x[i], src_x[j], src_G[j], +1.0) for j in 1:ns) for i in 1:nt]
Uh_m = -Uh_p
dev(U) = maximum(norm(U[i] - U_ref[i]) / norm(U_ref[i]) for i in 1:nt)
SIGN = dev(Uh_p) < dev(Uh_m) ? +1.0 : -1.0
check("hand direct kernel calibrates to fmm.direct! (< 1e-12)",
    dev(SIGN > 0 ? Uh_p : Uh_m) < 1e-12)

# ---- reverse routing on the shared grid ----
allpts = vcat(src_x, tgt_x)
g = CrossGrid(allpts)
src_levels, src_order, exc_s = build_level_cells(g, src_x, ELL)
tgt_levels, tgt_order, exc_t = build_level_cells(g, tgt_x, ELL)
check("containment (both sets)", exc_s == 0.0 && exc_t == 0.0)
tq = CrossStencil.UniformQTables(Q)
_, m2l_list, near_list = sweep_config(tq, src_levels, tgt_levels, ELL;
    materialize = true)

# pair coverage identity
paircount(list) = isempty(list) ? Int128(0) :
    sum(Int128(CrossStencil.cellcount(src_levels[L + 1], si)) *
        Int128(CrossStencil.cellcount(tgt_levels[L + 1], ti))
        for (si, ti, L) in list)
cov = paircount(m2l_list) + paircount(near_list)
check("pair coverage Σ|A||B| == ns·nt", cov == Int128(ns) * Int128(nt))

function far_field(P::Int)
    # ---- LH tables ----
    H = ((P + 1) * (P + 2)) >> 1
    D = 2 * H
    ct = CrossStencilTables(Q, ELL, g.h0, 0.0)
    ops_m2m = cross_m2m_operators_lh(P, g.h0, ELL)
    ops_m2l, class_slot = cross_m2l_operators_lh(P, g.h0, ct)
    s2r, s2c, p2l = cross_m2l_level_scales_lh(P, ELL)
    ops_l2l = cross_l2l_operators_lh(P, g.h0, ELL)
    off2k = Dict(SVector{3,Int}(o) => k for (k, o) in enumerate(ct.tables.push_offsets))

    cellcoords(lc, i, L) = begin
        cx, cy, cz = morton_decode(lc.codes[i])
        SVector{3,Int}(Int(cx), Int(cy), Int(cz))
    end
    cellcenter(coords, L) = begin
        w = 2 * g.h0 / (1 << L)
        g.x_min .+ (SVector{3,Float64}(coords) .+ 0.5) .* w
    end

    "pack a (2,2,H) expansion into the stacked LH vector"
    function lh_pack(e)
        v = zeros(2 * D)
        for ch in 1:2, c in 1:D
            v[(ch - 1) * D + c] = e[2 - (c & 1), ch, (c + 1) >> 1]
        end
        return v
    end

    # ---- Stage B: leaf B2M + LH M2M up the SOURCE cells ----
    M = [zeros(2 * D, CrossStencil.ncells(src_levels[L + 1])) for L in 0:ELL]
    harmonics = initialize_harmonics(P)
    lc_leaf = src_levels[ELL + 1]
    for i in 1:CrossStencil.ncells(lc_leaf)
        center = cellcenter(cellcoords(lc_leaf, i, ELL), ELL)
        e = initialize_expansion(P)
        for k in lc_leaf.starts[i]:lc_leaf.starts[i + 1] - 1
            j = src_order[k]
            body_to_multipole_point!(Point{Vortex}, e, harmonics,
                src_x[j] - center, src_G[j], P)
        end
        M[ELL + 1][:, i] .= lh_pack(e)
    end
    for Lc in ELL:-1:1
        lc_c = src_levels[Lc + 1]
        lc_p = src_levels[Lc]
        for i in 1:CrossStencil.ncells(lc_c)
            co = cellcoords(lc_c, i, Lc)
            phase = 1 + (co[1] & 1) + 2 * (co[2] & 1) + 4 * (co[3] & 1)
            pi_ = CrossStencil.cell_index(lc_p, lc_c.codes[i] >> 3)
            pi_ != 0 || error("parent cell missing")
            M[Lc][:, pi_] .+= ops_m2m[:, :, phase, Lc] * M[Lc + 1][:, i]
        end
    end

    # ---- Stage C: LH M2L over the reverse far routes ----
    Lo = [zeros(2 * D, CrossStencil.ncells(tgt_levels[L + 1])) for L in 0:ELL]
    for (si, ti, L) in m2l_list
        sco = cellcoords(src_levels[L + 1], si, L)
        tco = cellcoords(tgt_levels[L + 1], ti, L)
        k = get(off2k, tco - sco, 0)
        k != 0 || error("route offset $(tco - sco) not in push set")
        slot = class_slot[k]
        slot != 0 || error("route class $k has no slot")
        Lo[L + 1][:, ti] .+= (ops_m2l[:, :, slot] *
            (M[L + 1][:, si] .* s2c[:, L + 1])) .* s2r[:, L + 1] .* p2l[L + 1]
    end

    # ---- Stage D: LH L2L down the TARGET cells + L2B ----
    for Lc in 3:ELL
        lc_c = tgt_levels[Lc + 1]
        lc_p = tgt_levels[Lc]
        for i in 1:CrossStencil.ncells(lc_c)
            co = cellcoords(lc_c, i, Lc)
            phase = 1 + (co[1] & 1) + 2 * (co[2] & 1) + 4 * (co[3] & 1)
            pi_ = CrossStencil.cell_index(lc_p, lc_c.codes[i] >> 3)
            pi_ != 0 || error("parent cell missing")
            Lo[Lc + 1][:, i] .+= ops_l2l[:, :, phase, Lc] * Lo[Lc][:, pi_]
        end
    end
    lt_leaf = tgt_levels[ELL + 1]
    flat_phi = Lo[ELL + 1][1:D, :]
    flat_chi = Lo[ELL + 1][D + 1:2 * D, :]
    U_far = [zero(SVector{3,Float64}) for _ in 1:nt]
    for i in 1:CrossStencil.ncells(lt_leaf)
        center = cellcenter(cellcoords(lt_leaf, i, ELL), ELL)
        for k in lt_leaf.starts[i]:lt_leaf.starts[i + 1] - 1
            b = tgt_order[k]
            Δ = tgt_x[b] - center
            _, gx, gy, gz = _resident_local_eval_flat(flat_phi, flat_chi, i,
                Δ[1], Δ[2], Δ[3], P, P, Val(true))
            U_far[b] = SVector(gx, gy, gz)
        end
    end

    return U_far
end

# ---- Stage E: hand direct over the near list ----
U_near = [zero(SVector{3,Float64}) for _ in 1:nt]
for (si, ti, L) in near_list
    L == ELL || error("near pair not at leaf level")
    ls = src_levels[L + 1]; lt = tgt_levels[L + 1]
    for kt in lt.starts[ti]:lt.starts[ti + 1] - 1
        b = tgt_order[kt]
        acc = zero(SVector{3,Float64})
        for ks in ls.starts[si]:ls.starts[si + 1] - 1
            j = src_order[ks]
            acc += u_hand(tgt_x[b], src_x[j], src_G[j], SIGN)
        end
        U_near[b] += acc
    end
end

