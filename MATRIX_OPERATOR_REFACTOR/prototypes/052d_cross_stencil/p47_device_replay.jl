# P4.7 — device-side dense + cross-pass replay on dumped production states
# (052d, 2026-08-29; runs on the H200 node, --project=$HOME/fm052env-h200).
#
# Inputs: DUMPDIR (relU_dumps_13509236). For each np in (3544, 12776):
#   A. CPU dense (16 threads, _rect_panel_pair reg 4) over expanded columns —
#      same recipe as local p39.
#   B. DEVICE dense: all-pairs CUDA kernel over the same expanded columns with
#      the same _rect_panel_pair — device-kernel parity on production inputs.
#   C. DEVICE cross-pass replay at production cfg (q=12, ell_x=5, P=6,
#      R_guard=0.006, wake_tag=3, reg=4) on the dumped srcmat/wakemat/cent/
#      positions with a fresh p39-formula box; plus box variants (shifted,
#      scaled) for sensitivity.
# Outputs: relL2 of {devdense, cross variants} vs CPU dense, vs dumped
# deviceU/hostU, plus per-target rows for the 13 worst just-shed particles.

import FastMultipole
const FM = FastMultipole
FM.load_cuda_radix_lifecycle!() ||
    error("CUDA radix lifecycle failed to load: $(FM.cuda_radix_status())")
const CUDA = FM.CUDA
using StaticArrays, Printf, LinearAlgebra, Statistics
using Base.Threads

const DUMPDIR = get(ENV, "DUMPDIR", "")
isempty(DUMPDIR) && error("set DUMPDIR")
const REG = 4
const WAKE_TAG = 3
const Q = 12
const ELLX = 5
const P = 6
const RG = 0.006

readmat(f, r) = collect(reshape(reinterpret(Float64, read(f)), r, :))

function expand_columns(srcmat, wakemat)
    ns = size(srcmat, 2)
    nshed = count(k -> wakemat[1, k] > 0, 1:ns)
    E = zeros(17, ns + 2 * nshed)
    E[:, 1:ns] .= srcmat
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
        end
    end
    return E
end

function cpu_dense(E, positions)
    nt = size(positions, 2)
    U = zeros(3, nt)
    @threads for s in 1:nt
        t = SVector{3,Float64}(positions[1, s], positions[2, s], positions[3, s])
        u = zero(SVector{3,Float64})
        @inbounds for j in 1:size(E, 2)
            tag = Int(E[1, j]); nv = Int(E[2, j])
            (1 <= tag <= 5 && nv >= 3) || continue
            v1 = SVector(E[3, j], E[4, j], E[5, j])
            v2 = SVector(E[6, j], E[7, j], E[8, j])
            v3 = SVector(E[9, j], E[10, j], E[11, j])
            v4 = SVector(E[12, j], E[13, j], E[14, j])
            uq, _ = FM._rect_panel_pair(FM.RectangularPanelInfluence(), t,
                tag, nv, v1, v2, v3, v4, E[15, j], E[16, j], E[17, j],
                Val(false), Val(REG))
            u += uq
        end
        U[:, s] .= u
    end
    return U
end

function _dev_dense_kernel!(out, E, pos, ncol, nt)
    s = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    s > nt && return nothing
    @inbounds begin
        t = SVector{3,Float64}(pos[1, s], pos[2, s], pos[3, s])
        u = zero(SVector{3,Float64})
        for j in 1:ncol
            tag = unsafe_trunc(Int, E[1, j])
            nv = unsafe_trunc(Int, E[2, j])
            (1 <= tag <= 5 && nv >= 3) || continue
            v1 = SVector(E[3, j], E[4, j], E[5, j])
            v2 = SVector(E[6, j], E[7, j], E[8, j])
            v3 = SVector(E[9, j], E[10, j], E[11, j])
            v4 = SVector(E[12, j], E[13, j], E[14, j])
            uq, _ = FM._rect_panel_pair(FM.RectangularPanelInfluence(), t,
                tag, nv, v1, v2, v3, v4, E[15, j], E[16, j], E[17, j],
                Val(false), Val(REG))
            u += uq
        end
        out[1, s] = u[1]; out[2, s] = u[2]; out[3, s] = u[3]
    end
    return nothing
end

function device_dense(E, positions)
    nt = size(positions, 2)
    d_E = CUDA.CuArray(E)
    d_pos = CUDA.CuArray(positions)
    d_out = CUDA.zeros(Float64, 3, nt)
    CUDA.@cuda threads=128 blocks=cld(nt, 128) _dev_dense_kernel!(
        d_out, d_E, d_pos, size(E, 2), nt)
    CUDA.synchronize()
    return Array(d_out)
end

"full device cross pass on dumped inputs with box (x_min, h0)"
function device_cross(srcmat, wakemat, cent, positions, x_min, h0)
    ns = size(srcmat, 2)
    np = size(positions, 2)
    ct = FM.CrossStencilTables(Q, ELLX, h0, RG)
    ctx = FM.device_cross_producer_context(ct, x_min, h0, ns, np)
    xs = FM.device_cross_expansion_state(ctx, P)
    m2l = FM.cross_m2l_operators(P, h0, ct)
    l2l = FM.cross_l2l_operators(P, h0, ct.ell_x)
    ls = FM.device_cross_local_state(ctx, P; m2l_tables=m2l, l2l_table=l2l)
    d_src = CUDA.CuArray(srcmat)
    d_cent = CUDA.CuArray(cent)
    d_wake = CUDA.CuArray(wakemat)
    pos_d = CUDA.CuArray(positions)
    FM.refresh_cross_producers!(ctx, d_cent, pos_d)
    ctx.needs_rebuild && return nothing, :needs_rebuild
    FM.refresh_cross_multipoles!(xs, ctx, d_src, d_wake)
    xs.n_skipped == 0 || return nothing, :skipped
    FM.refresh_cross_locals!(ls, ctx, xs)
    FM.finish_cross_locals!(ls, ctx, pos_d)
    FM.apply_cross_near!(ls, ctx, d_src, pos_d; d_wake_buffer=d_wake,
        wake_tag=WAKE_TAG, reg=REG, potential=false)
    return Array(ls.d_out)[2:4, :], :ok
end

relL2(a, b) = sqrt(sum(abs2, a .- b)) / max(sqrt(sum(abs2, b)), eps())

for np in (3544, 12776)
    pre = joinpath(DUMPDIR, "dump_np$np")
    isfile(pre * "_meta.txt") || continue
    positions = readmat(pre * "_positions_3xN_f64.bin", 3)
    deviceU = readmat(pre * "_deviceU_3xN_f64.bin", 3)
    hostU = readmat(pre * "_hostU_3xN_f64.bin", 3)
    srcmat = readmat(pre * "_body1_srcmat_17xS_f64.bin", 17)
    cent = readmat(pre * "_body1_cent_3xS_f64.bin", 3)
    wakemat = readmat(pre * "_body1_wakemat_8xS_f64.bin", 8)
    E = expand_columns(srcmat, wakemat)
    println("\n================ np=$np (ncol=$(size(E,2))) ================")

    tA = @elapsed Udense = cpu_dense(E, positions)
    @printf("A. cpu dense: %.1fs | vs deviceU %.3e | vs hostU %.3e\n",
        tA, relL2(deviceU, Udense), relL2(hostU, Udense))

    tB = @elapsed Udd = device_dense(E, positions)
    @printf("B. device dense: %.1fs | vs cpu dense %.3e | max abs diff %.3e | vs deviceU %.3e\n",
        tB, relL2(Udd, Udense), maximum(abs.(Udd .- Udense)), relL2(Udd, deviceU))

    # p39-formula box
    lo = zeros(3); hi = zeros(3)
    for a in 1:3
        pv = extrema(view(positions, a, :))
        sv = extrema(vcat(vec(srcmat[2 + a, :]), vec(srcmat[5 + a, :]),
            vec(srcmat[8 + a, :])))
        lo[a] = min(pv[1], sv[1]); hi[a] = max(pv[2], sv[2])
    end
    h0 = 0.75 * maximum(hi .- lo) + 1e-9
    ctr = (lo .+ hi) ./ 2
    wleaf = 2 * h0 / (1 << ELLX)
    boxes = [
        ("fresh", SVector{3,Float64}(ctr .- h0), h0),
        ("shift+0.37w", SVector{3,Float64}(ctr .- h0 .+ 0.37 * wleaf), h0),
        ("shift-0.61w", SVector{3,Float64}(ctr .- h0 .- 0.61 * wleaf), h0),
        ("scale1.25", SVector{3,Float64}(ctr .- 1.25 * h0), 1.25 * h0),
    ]
    for (nm, xm, h) in boxes
        t = @elapsed (Uc, status) = device_cross(srcmat, wakemat, cent,
            positions, xm, h)
        if status != :ok
            @printf("C. cross[%s]: %s\n", nm, status)
            continue
        end
        @printf("C. cross[%-11s]: %.1fs | vs cpu dense %.3e | vs deviceU %.3e | vs hostU %.3e\n",
            nm, t, relL2(Uc, Udense), relL2(Uc, deviceU), relL2(Uc, hostU))
        if nm == "fresh"
            println("   per-target (13 worst just-shed):")
            @printf("   %-6s %10s %10s %10s\n", "pid", "|cr-dns|", "|cr-dev|", "|dev-dns|")
            for pid in np-12:np
                cr = SVector{3,Float64}(Uc[:, pid])
                dn = SVector{3,Float64}(Udense[:, pid])
                dv = SVector{3,Float64}(deviceU[:, pid])
                @printf("   %-6d %10.3e %10.3e %10.3e\n", pid,
                    norm(cr - dn), norm(cr - dv), norm(dv - dn))
            end
        end
        flush(stdout)
    end
end
println("DONE")
