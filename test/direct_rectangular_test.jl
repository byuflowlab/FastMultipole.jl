# direct_rectangular_test.jl — task 051 stage 1: rectangular (distinct
# source-set -> distinct target-set) direct evaluation, host reference path.
#
# Layers:
#   1. point functor vs a plain all-pairs loop (independent transcription of
#      the gaussianerf pair math, SpecialFunctions.erf) — 1e-13 relative
#      (erf implementations differ at the ulp level).
#   2. point functor vs FLOWVPM's OWN pair math (kernel_gaussianerf.g_dgdr)
#      when FLOWVPM resolves in the active environment — 1e-14 relative.
#   3. panel functor analytic single-panel gates (FLOWPanel-free):
#      square-ring center velocity, source-panel self/surface limits and
#      far-field monopole, doublet == vortex-ring equivalence.
#   4. panel functor vs FLOWPanel's `induced` (the math `direct!` at
#      FLOWPanel_abstractbody.jl:1260 sums) on a ~56-panel sphere x 200
#      targets — 1e-12 relative — when FLOWPanel resolves. Skips cleanly
#      when absent so FastMultipole CI does not depend on it.

using Test
using FastMultipole
using FastMultipole.StaticArrays
using FastMultipole.LinearAlgebra: norm
import Random

const FMR = FastMultipole

# Optional cross-check providers. SpecialFunctions supplies an INDEPENDENT
# erf (openlibm) for the transcription gate and the cross-erf sum gate;
# FLOWVPM supplies the exact production pair math. Both are optional so the
# file runs in any of: FastMultipole test env, FLOWPanel/FLOWVPM dev env,
# bare FastMultipole env.
const _HAVE_SPECIALFUNCTIONS = try
    @eval import SpecialFunctions
    true
catch
    false
end

# all-pairs reference for the point functor (independent transcription of
# FLOWVPM_fmm.jl:144-218), parameterized by the g/dgdr provider so the same
# loop serves the structural gate (same erf as the implementation), the
# cross-erf gate (SpecialFunctions.erf), and the FLOWVPM gate (production
# g_dgdr)
function _point_reference(g_dgdr::F, tgt, src) where F
    n_tgt = size(tgt, 2)
    n_src = size(src, 2)
    ref = zeros(12, n_tgt)
    for i in 1:n_tgt, q in 1:n_src
        dx = tgt[1, i] - src[1, q]; dy = tgt[2, i] - src[2, q]; dz = tgt[3, i] - src[3, q]
        r2 = dx^2 + dy^2 + dz^2
        iszero(r2) && continue
        r = sqrt(r2)
        sigma = src[7, q]
        g, dgdr = g_dgdr(r / sigma)
        r3inv = 1 / (r2 * r)
        c4 = 1 / (4pi)
        gx, gy, gz = src[4, q], src[5, q], src[6, q]
        crss1 = -c4 * r3inv * (dy*gz - dz*gy)
        crss2 = -c4 * r3inv * (dz*gx - dx*gz)
        crss3 = -c4 * r3inv * (dx*gy - dy*gx)
        ref[1, i] += g * crss1; ref[2, i] += g * crss2; ref[3, i] += g * crss3
        aux1 = dgdr / (sigma*r) - 3g / r2
        aux2 = -c4 * g * r3inv
        ref[4, i] += aux1*crss1*dx
        ref[5, i] += aux1*crss2*dx - aux2*gz
        ref[6, i] += aux1*crss3*dx + aux2*gy
        ref[7, i] += aux1*crss1*dy + aux2*gz
        ref[8, i] += aux1*crss2*dy
        ref[9, i] += aux1*crss3*dy - aux2*gx
        ref[10, i] += aux1*crss1*dz - aux2*gy
        ref[11, i] += aux1*crss2*dz + aux2*gx
        ref[12, i] += aux1*crss3*dz
    end
    return ref
end

# g/dgdr from an erf function (FLOWVPM_kernel.jl:54-57 form)
function _g_dgdr_from_erf(erf_fn::F) where F
    sqrt2opi = sqrt(2 / pi)
    return rho -> begin
        aux = sqrt2opi * rho * exp(-rho^2 / 2)
        (erf_fn(rho / sqrt(2)) - aux, rho * aux)
    end
end

@testset "direct rectangular: point functor" begin
    Random.seed!(51051)
    T = Float64
    n_src = 300
    n_tgt = 140
    src = rand(T, 7, n_src) .- 0.5
    src[7, :] .= 0.02 .+ 0.08 .* rand(n_src)          # sigma > 0
    tgt = zeros(T, 3, n_tgt)
    tgt[:, 1:100] .= rand(T, 3, 100) .- 0.5           # inside the cloud
    tgt[:, 101:110] .= src[1:3, 1:10]                 # coincident with sources
    tgt[:, 111:140] .= 50.0 .* (rand(T, 3, 30) .- 0.5)  # far away
    # also make some targets EXTREMELY close (but not equal) to sources: the
    # CPU semantics keep these pairs (no absolute eps2 cutoff)
    tgt[:, 1:5] .= src[1:3, 11:15] .+ 1e-8 .* randn(3, 5)

    out = zeros(T, 12, n_tgt)
    direct_rectangular!(out, tgt, RectangularGaussianErfVortex(), src; gradient=true)
    @test all(isfinite, out)

    relerr(a, b) = maximum(abs.(a .- b)) / maximum(abs.(b))

    # layer 1a (STRUCTURAL, always, tight): all-pairs reference using the
    # implementation's own g/dgdr, so the gate isolates the pair-sum
    # structure (cross products, aux1/aux2 assembly, accumulation,
    # threading) with zero erf-implementation confound.
    ref = _point_reference(FastMultipole._rect_g_dgdr_gauserf, tgt, src)
    @test relerr(out[1:3, :], ref[1:3, :]) < 1e-14
    @test relerr(out[4:12, :], ref[4:12, :]) < 1e-14

    # layer 1b (CROSS-ERF, when SpecialFunctions resolves): the same sums
    # referenced with an INDEPENDENT erf (openlibm). The two erfs agree to
    # 1 ulp, but g = erf(rho/sqrt2) - aux cancels catastrophically at tiny
    # rho: g ~ sqrt(2/pi) rho^3/3 while the erf value is ~sqrt(2/pi) rho, so
    # a 1-ulp erf difference is amplified to a RELATIVE g (hence J) error of
    # up to ~3 eps/rho^2 — measured up to 5.3e-2 at rho = 1e-7. The
    # deliberately near-coincident columns 1:5 (offset 1e-8, rho ~ 2e-7 to
    # 1e-6) therefore CANNOT meet an ulp-level gate against a different erf;
    # they get a per-column amplification-scaled gate instead. Regular
    # columns keep a tight gate: their smallest rho is O(1e-2), where the
    # amplification 3 eps/rho^2 is ~1e-11 on the pair with the smallest g —
    # and that pair's J contribution is ~rho^3 of the column scale, keeping
    # the column-normalized error at the low end of 1e-13; gate 1e-12 with
    # margin.
    if _HAVE_SPECIALFUNCTIONS
        # erf transcription gate (hypothesis-(a) guard): the vendored fdlibm
        # erf must match openlibm to 1 ulp absolutely everywhere and 1 ulp
        # RELATIVELY at small arguments (the cancellation-critical regime)
        worst_abs = 0.0
        for x in vcat(0.0, 10.0 .^ (-12:0.05:0.8))
            for s in (x, -x)
                worst_abs = max(worst_abs,
                    abs(FastMultipole._rect_erf(s) - SpecialFunctions.erf(s)))
            end
        end
        @test worst_abs < 5e-16
        worst_rel = 0.0
        for x in 10.0 .^ (-10:0.01:-4)
            worst_rel = max(worst_rel,
                abs(FastMultipole._rect_erf(x) - SpecialFunctions.erf(x)) /
                    SpecialFunctions.erf(x))
        end
        @test worst_rel < 1e-15

        ref_sf = _point_reference(_g_dgdr_from_erf(SpecialFunctions.erf), tgt, src)
        @test relerr(out[1:3, 6:end], ref_sf[1:3, 6:end]) < 1e-12
        @test relerr(out[4:12, 6:end], ref_sf[4:12, 6:end]) < 1e-12
        # near-coincident columns: per-column gate at the derived
        # amplification bound 3 eps/rho_min^2 (x16 safety for multiple
        # contributing near pairs), floored at 1e-12
        for i in 1:5
            rho_min = Inf
            for q in 1:n_src
                r = sqrt((tgt[1, i] - src[1, q])^2 + (tgt[2, i] - src[2, q])^2 +
                         (tgt[3, i] - src[3, q])^2)
                r > 0 && (rho_min = min(rho_min, r / src[7, q]))
            end
            gate = max(1e-12, 16 * 3 * eps(1.0) / rho_min^2)
            @test relerr(out[1:3, i:i], ref_sf[1:3, i:i]) < gate
            @test relerr(out[4:12, i:i], ref_sf[4:12, i:i]) < gate
        end
    else
        @info "SpecialFunctions not available in this environment; skipping the cross-erf gates"
    end

    # U-only call matches the U rows of the gradient call
    out_u = zeros(T, 3, n_tgt)
    direct_rectangular!(out_u, tgt, RectangularGaussianErfVortex(), src; gradient=false)
    @test out_u == out[1:3, :]

    # accumulation semantics: second call doubles
    out2 = copy(out)
    direct_rectangular!(out2, tgt, RectangularGaussianErfVortex(), src; gradient=true)
    @test out2 ≈ 2 .* out rtol=1e-14

    # layer 2: FLOWVPM's own pair math (exact custom_erf match, 1e-14)
    flowvpm_loaded = try
        @eval import FLOWVPM
        true
    catch
        false
    end
    if flowvpm_loaded
        # production pair math wholesale: FLOWVPM's g_dgdr uses the SAME
        # fdlibm erf (custom_erf) the implementation vendors, so this gate is
        # cancellation-free and validates the implementation's hardcoded
        # constants (sqrt(2/pi), sqrt 2) against FLOWVPM's computed ones
        refv = _point_reference(FLOWVPM.kernel_gaussianerf.g_dgdr, tgt, src)
        @test relerr(out[1:3, :], refv[1:3, :]) < 1e-14
        @test relerr(out[4:12, :], refv[4:12, :]) < 1e-14
    else
        @info "FLOWVPM not available in this environment; skipping the exact pair-math cross-check"
    end

    # Float32 path (F32 variant of the point kernel)
    src32 = Float32.(src); tgt32 = Float32.(tgt)
    out32 = zeros(Float32, 12, n_tgt)
    direct_rectangular!(out32, tgt32, RectangularGaussianErfVortex(), src32; gradient=true)
    @test all(isfinite, out32)
    # columns 1:5 are 1e-8 from a source — that separation rounds away in F32
    # (positions O(0.5), eps(F32) ~ 6e-8), so compare well-separated targets
    @test relerr(Float64.(out32[1:3, 6:end]), ref[1:3, 6:end]) < 1e-4
end

# panel-source packing helper (row layout documented on RectangularPanelInfluence)
function _pack_panel!(A, q, tag, verts, s1, s2, koff)
    A[1, q] = tag
    A[2, q] = length(verts)
    for (iv, v) in enumerate(verts)
        A[3 + 3*(iv-1), q] = v[1]
        A[4 + 3*(iv-1), q] = v[2]
        A[5 + 3*(iv-1), q] = v[3]
    end
    A[15, q] = s1
    A[16, q] = s2
    A[17, q] = koff
    return A
end

@testset "direct rectangular: panel functor analytic gates" begin
    T = Float64
    koff = 1e-8

    # --- gate 1: square vortex ring, velocity at center ---
    # Singular square ring side a: |U(center)| = 2*sqrt(2)*Gamma/(pi*a), along
    # +z for CCW vertex ordering in the z=0 plane. Vatistas core rc=1e-8 at
    # standoff a/2 perturbs at O((2 rc/a)^4) ~ 1e-32.
    a = 1.0
    ring = zeros(T, 17, 1)
    _pack_panel!(ring, 1, 3, (SVector(0.0, 0.0, 0.0), SVector(a, 0.0, 0.0),
        SVector(a, a, 0.0), SVector(0.0, a, 0.0)), 1.0, 0.0, koff)
    center = reshape(T[a/2, a/2, 0.0], 3, 1)
    outr = zeros(T, 3, 1)
    direct_rectangular!(outr, center, RectangularPanelInfluence(), ring; gradient=false)
    @test isapprox(outr[3, 1], 2*sqrt(2)/(pi*a); rtol=1e-12)
    @test abs(outr[1, 1]) < 1e-14 && abs(outr[2, 1]) < 1e-14

    # --- gate 2: constant-source equilateral tri: self/surface limits + monopole ---
    s3 = sqrt(3.0)
    v1 = SVector(0.0, 0.0, 0.0); v2 = SVector(1.0, 0.0, 0.0); v3 = SVector(0.5, s3/2, 0.0)
    area = s3/4
    centroid = (v1 + v2 + v3) / 3
    sigma = 1.7
    srcp = zeros(T, 17, 1)
    _pack_panel!(srcp, 1, 1, (v1, v2, v3), sigma, 0.0, koff)
    # (a) exact self pair: u = sigma/2 * n (tangential PV vanishes by symmetry)
    tgt_self = reshape(collect(centroid), 3, 1)
    outs = zeros(T, 3, 1)
    direct_rectangular!(outs, tgt_self, RectangularPanelInfluence(), srcp; gradient=false)
    @test isapprox(outs[3, 1], sigma/2; atol=1e-12)   # n = +z for CCW ordering
    @test abs(outs[1, 1]) < 1e-12 && abs(outs[2, 1]) < 1e-12
    # (b) near-surface exterior/interior normal-velocity jump
    for (dz, sgn) in ((1e-7, +1.0), (-1e-7, -1.0))
        tgt_near = reshape([centroid[1], centroid[2], dz], 3, 1)
        outn = zeros(T, 3, 1)
        direct_rectangular!(outn, tgt_near, RectangularPanelInfluence(), srcp; gradient=false)
        @test isapprox(outn[3, 1], sgn*sigma/2; rtol=1e-4)
    end
    # (c) far-field monopole: U -> sigma*A/(4 pi r^2) rhat
    rfar = 250.0
    dir = SVector(0.3, -0.5, 0.81); dir = dir / norm(dir)
    tgt_far = reshape(collect(centroid + rfar*dir), 3, 1)
    outf = zeros(T, 12, 1)
    direct_rectangular!(outf, tgt_far, RectangularPanelInfluence(), srcp; gradient=true)
    U_far = SVector(outf[1, 1], outf[2, 1], outf[3, 1])
    @test isapprox(U_far, sigma*area/(4pi*rfar^2) * dir; rtol=1e-4)

    # --- gate 3: constant doublet == vortex ring (same tri, same strength) ---
    mu = 0.83
    dbl = zeros(T, 17, 1)
    _pack_panel!(dbl, 1, 2, (v1, v2, v3), mu, 0.0, 1e-12)
    rng3 = zeros(T, 17, 1)
    _pack_panel!(rng3, 1, 3, (v1, v2, v3), mu, 0.0, 1e-12)
    Random.seed!(5151)
    for _ in 1:5
        p = centroid + SVector{3}(2 .* randn(3))
        tgtp = reshape(collect(p), 3, 1)
        od = zeros(T, 3, 1); og = zeros(T, 3, 1)
        direct_rectangular!(od, tgtp, RectangularPanelInfluence(), dbl; gradient=false)
        direct_rectangular!(og, tgtp, RectangularPanelInfluence(), rng3; gradient=false)
        @test isapprox(od, og; rtol=1e-6, atol=1e-10)
    end

    # --- gate 4a: open filament (tag 3, nv=2), straight-segment analytic ---
    # U at perpendicular distance d from the midpoint of a length-L segment:
    # |U| = Gamma L / (4 pi d sqrt(d^2 + L^2/4)); Vatistas rc=1e-8 at d=0.5
    # perturbs at O((rc/d)^4). Segment along +x, target at +y => U along +z.
    L = 2.0; d = 0.5; gam = 1.3
    fil = zeros(T, 17, 1)
    _pack_panel!(fil, 1, 3, (SVector(-L/2, 0.0, 0.0), SVector(L/2, 0.0, 0.0)),
        gam, 0.0, 1e-8)
    tgtf = reshape(T[0.0, d, 0.0], 3, 1)
    outfil = zeros(T, 3, 1)
    direct_rectangular!(outfil, tgtf, RectangularPanelInfluence(), fil; gradient=false)
    @test isapprox(outfil[3, 1], gam*L/(4pi*d*sqrt(d^2 + L^2/4)); rtol=1e-12)
    @test abs(outfil[1, 1]) < 1e-14 && abs(outfil[2, 1]) < 1e-14

    # --- gate 4b: closed tri ring == sum of its three open segments, all
    # families (same segment functions, only the Gamma grouping differs) ---
    rc = 0.05
    vA = SVector(0.0, 0.0, 0.0); vB = SVector(1.0, 0.1, 0.0); vC = SVector(0.4, 0.9, 0.2)
    ring3f = zeros(T, 17, 1)
    _pack_panel!(ring3f, 1, 3, (vA, vB, vC), 0.77, 0.0, rc)
    segs = zeros(T, 17, 3)
    _pack_panel!(segs, 1, 3, (vA, vB), 0.77, 0.0, rc)
    _pack_panel!(segs, 2, 3, (vB, vC), 0.77, 0.0, rc)
    _pack_panel!(segs, 3, 3, (vC, vA), 0.77, 0.0, rc)
    tgts4 = randn(3, 8) .* 0.8
    for fam in (:vatistas, :compact, :gaussian)
        kernf = RectangularPanelInfluence(fam)
        o1 = zeros(T, 12, 8); o2 = zeros(T, 12, 8)
        direct_rectangular!(o1, tgts4, kernf, ring3f; gradient=true)
        direct_rectangular!(o2, tgts4, kernf, segs; gradient=true)
        @test isapprox(o1, o2; rtol=1e-12, atol=1e-13)
    end
end

@testset "direct rectangular: panel functor vs FLOWPanel" begin
    flowpanel_loaded = try
        @eval import FLOWPanel
        true
    catch
        false
    end
    if !flowpanel_loaded
        @info "FLOWPanel not available in this environment; skipping the FLOWPanel parity layer"
    else
        pnl = FLOWPanel
        # this file's math is transcribed from FLOWPanel commit 75b45c7, whose
        # filament kernel is the Vatistas n=2 core; if the loaded (WIP)
        # FLOWPanel carries the selectable-family upgrade, pin it to Vatistas
        # for the comparison.
        if isdefined(pnl, :set_filament_regularization!)
            pnl.set_filament_regularization!(:vatistas)
        end

        # small closed sphere mesh (pattern: FLOWPanel test/runtests_unit_added_mass.jl)
        function sphere_mesh(R, n_theta, n_phi)
            np = n_phi
            nodes = zeros(3, 2 + (n_theta - 1) * np)
            nodes[:, 1] .= (0.0, 0.0, R)
            nodes[:, 2] .= (0.0, 0.0, -R)
            node_id(j, k) = 2 + (j - 1) * np + mod(k - 1, np) + 1
            for j in 1:n_theta-1, k in 1:np
                th = j * pi / n_theta
                ph = 2pi * (k - 1) / np
                nodes[:, node_id(j, k)] .= (R*sin(th)*cos(ph), R*sin(th)*sin(ph), R*cos(th))
            end
            ncells = 2np + 2np*(n_theta - 2)
            cells = zeros(Int, 3, ncells)
            c = 0
            for k in 1:np
                cells[:, c += 1] .= (1, node_id(1, k), node_id(1, k + 1))
            end
            for j in 1:n_theta-2, k in 1:np
                uk, ukp = node_id(j, k), node_id(j, k + 1)
                lk, lkp = node_id(j + 1, k), node_id(j + 1, k + 1)
                cells[:, c += 1] .= (uk, lk, lkp)
                cells[:, c += 1] .= (uk, lkp, ukp)
            end
            for k in 1:np
                cells[:, c += 1] .= (2, node_id(n_theta - 1, k + 1), node_id(n_theta - 1, k))
            end
            return nodes, cells
        end

        Random.seed!(151)
        nodes, cells = sphere_mesh(1.0, 5, 7)              # 56 tri panels
        koff = 1e-3                                        # CORE_SIZE_TARGETS scale
        relerr(a, b) = maximum(abs.(a .- b)) / maximum(abs.(b))

        # 200 targets: shell around the body + a few centroids (self pairs)
        n_tgt = 200
        tgt = zeros(3, n_tgt)
        for i in 1:n_tgt
            d = randn(3); d ./= norm(d)
            tgt[:, i] .= (1.0 + 2.5*rand()) .* d           # r in [1, 3.5]
        end

        for (label, E, tag, nk) in (
                ("source+vortexring (018 body set)", Union{pnl.ConstantSource, pnl.VortexRing}, 4, 2),
                ("source+doublet", Union{pnl.ConstantSource, pnl.ConstantDoublet}, 5, 2),
                ("constant source", Union{pnl.ConstantSource}, 1, 1),
            )
            body = pnl.NonLiftingBody{E}(copy(nodes), copy(cells); core_size=koff)
            body.strength .= randn(body.ncells, nk)
            # centroids for self-pair targets
            tgt_local = copy(tgt)
            for (slot, ic) in enumerate((3, 20, 41))
                va = SVector{3}(body.nodes[:, body.cells[1, ic]])
                vb = SVector{3}(body.nodes[:, body.cells[2, ic]])
                vc = SVector{3}(body.nodes[:, body.cells[3, ic]])
                tgt_local[:, slot] .= (va + vb + vc) ./ 3
            end

            # pack the body into the rectangular source layout
            srcp = zeros(17, body.ncells)
            for ic in 1:body.ncells
                verts = (SVector{3}(body.nodes[:, body.cells[1, ic]]),
                         SVector{3}(body.nodes[:, body.cells[2, ic]]),
                         SVector{3}(body.nodes[:, body.cells[3, ic]]))
                s1 = body.strength[ic, 1]
                s2 = nk == 2 ? body.strength[ic, 2] : 0.0
                _pack_panel!(srcp, ic, tag, verts, s1, s2, koff)
            end

            out = zeros(13, n_tgt)
            direct_rectangular!(out, tgt_local, RectangularPanelInfluence(), srcp;
                gradient=true, scalar_potential=true)

            # FLOWPanel reference: the same sum FLOWPanel's direct! performs
            # (FLOWPanel_abstractbody.jl:1260), via the index-path `induced`
            switch = FastMultipole.DerivativesSwitch(true, true, true)
            ref = zeros(13, n_tgt)
            for i in 1:n_tgt
                target = SVector{3}(tgt_local[:, i])
                for ic in 1:body.ncells
                    phi, U, H = pnl.induced(target, body, ic, switch; core_size=koff)
                    ref[1:3, i] .+= U
                    for j in 1:3, k in 1:3
                        ref[3 + (j-1)*3 + k, i] += H[k, j]
                    end
                    ref[13, i] += phi
                end
            end
            @test relerr(out[1:3, :], ref[1:3, :]) < 1e-12
            @test relerr(out[4:12, :], ref[4:12, :]) < 1e-12
            @test relerr(out[13:13, :], ref[13:13, :]) < 1e-12
            @info "FLOWPanel parity ($label)" relerr_U=relerr(out[1:3, :], ref[1:3, :]) relerr_H=relerr(out[4:12, :], ref[4:12, :]) relerr_phi=relerr(out[13:13, :], ref[13:13, :])
        end

        # --- filament-regularization families (051 stage 2): run the
        # ring-containing element sets once per family, selecting the family
        # on BOTH sides (set_filament_regularization! for the FLOWPanel
        # reference, filament_reg on the rectangular functor). Requires the
        # working-tree FLOWPanel with the selectable-family upgrade; on HEAD
        # FLOWPanel only the vatistas case above runs.
        if isdefined(pnl, :set_filament_regularization!)
            fam0 = pnl.FILAMENT_REGULARIZATION[]
            for (fam, regi) in ((:vatistas, 1), (:compact, 2), (:gaussian, 3))
                pnl.set_filament_regularization!(fam)
                for (label, E, tag, nk) in (
                        ("source+vortexring", Union{pnl.ConstantSource, pnl.VortexRing}, 4, 2),
                        ("pure vortexring", pnl.VortexRing, 3, 1),
                    )
                    body = pnl.NonLiftingBody{E}(copy(nodes), copy(cells); core_size=koff)
                    body.strength .= randn(body.ncells, nk)
                    tgt_local = copy(tgt)
                    for (slot, ic) in enumerate((3, 20, 41))
                        va = SVector{3}(body.nodes[:, body.cells[1, ic]])
                        vb = SVector{3}(body.nodes[:, body.cells[2, ic]])
                        vc = SVector{3}(body.nodes[:, body.cells[3, ic]])
                        tgt_local[:, slot] .= (va + vb + vc) ./ 3
                    end
                    srcp = zeros(17, body.ncells)
                    for ic in 1:body.ncells
                        verts = (SVector{3}(body.nodes[:, body.cells[1, ic]]),
                                 SVector{3}(body.nodes[:, body.cells[2, ic]]),
                                 SVector{3}(body.nodes[:, body.cells[3, ic]]))
                        s1 = body.strength[ic, 1]
                        s2 = nk == 2 ? body.strength[ic, 2] : 0.0
                        _pack_panel!(srcp, ic, tag, verts, s1, s2, koff)
                    end
                    out = zeros(12, n_tgt)
                    direct_rectangular!(out, tgt_local,
                        RectangularPanelInfluence(fam), srcp; gradient=true)
                    switch = FastMultipole.DerivativesSwitch(false, true, true)
                    ref = zeros(12, n_tgt)
                    for i in 1:n_tgt
                        target = SVector{3}(tgt_local[:, i])
                        for ic in 1:body.ncells
                            _, U, H = pnl.induced(target, body, ic, switch; core_size=koff)
                            ref[1:3, i] .+= U
                            for j in 1:3, k in 1:3
                                ref[3 + (j-1)*3 + k, i] += H[k, j]
                            end
                        end
                    end
                    @test relerr(out[1:3, :], ref[1:3, :]) < 1e-12
                    @test relerr(out[4:12, :], ref[4:12, :]) < 1e-12
                    @info "FLOWPanel parity ($fam, $label)" relerr_U=relerr(out[1:3, :], ref[1:3, :]) relerr_H=relerr(out[4:12, :], ref[4:12, :])
                end

                # open filament columns (tag 3, nv=2): the FilamentWrapper
                # direct! sum (FLOWPanel_wake.jl:2874-2907) — one
                # _bound_vortex_velocity/_gradient segment per column
                nfil = 10
                cs = 2e-2
                filsrc = zeros(17, nfil)
                fverts = Vector{NTuple{2,SVector{3,Float64}}}(undef, nfil)
                fgam = randn(nfil)
                for q in 1:nfil
                    p1 = SVector{3}(randn(3))
                    p2 = p1 + SVector{3}(0.3 .* randn(3))
                    fverts[q] = (p1, p2)
                    _pack_panel!(filsrc, q, 3, (p1, p2), fgam[q], 0.0, cs)
                end
                outfl = zeros(12, n_tgt)
                direct_rectangular!(outfl, tgt, RectangularPanelInfluence(fam),
                    filsrc; gradient=true)
                reff = zeros(12, n_tgt)
                famval = Val(pnl.FILAMENT_REGULARIZATION[])
                for i in 1:n_tgt
                    target = SVector{3}(tgt[:, i])
                    for q in 1:nfil
                        p1, p2 = fverts[q]
                        U = pnl._bound_vortex_velocity(target - p1, target - p2,
                            true, cs, famval) * fgam[q]
                        G = pnl._bound_vortex_gradient(p1 - target, p2 - target,
                            true, cs, famval) * fgam[q]
                        reff[1:3, i] .+= U
                        for j in 1:3, k in 1:3
                            reff[3 + (j-1)*3 + k, i] += G[k, j]
                        end
                    end
                end
                @test relerr(outfl[1:3, :], reff[1:3, :]) < 1e-12
                @test relerr(outfl[4:12, :], reff[4:12, :]) < 1e-12
                @info "FLOWPanel parity ($fam, open filament)" relerr_U=relerr(outfl[1:3, :], reff[1:3, :]) relerr_H=relerr(outfl[4:12, :], reff[4:12, :])
            end
            pnl.set_filament_regularization!(fam0)
        else
            @info "loaded FLOWPanel lacks set_filament_regularization!; family parity limited to vatistas"
        end
    end
end

# packs a random tri (occasionally a quad or an nv=2 open filament for tag 3)
# around a base point; keeps panels O(0.1)-sized so targets shifted +2 in x
# are safely off every edge
function _pack_random_panel!(A, q, base)
    tag = rand(1:5)
    r = rand()
    nv = tag == 3 ? (r < 0.25 ? 2 : (r < 0.5 ? 4 : 3)) : 3
    verts = ntuple(nv) do iv
        SVector{3,Float64}(base .+ 0.15 .* randn(3))
    end
    _pack_panel!(A, q, tag, verts, 0.5 + rand(), 0.5 + rand(), 1e-3)
    return A
end

@testset "direct rectangular: argument validation" begin
    src = rand(7, 4) .- 0.5
    src[7, :] .= 0.05
    tgt = rand(3, 3)
    # too-short out for gradient=true
    @test_throws ArgumentError direct_rectangular!(zeros(3, 3), tgt,
        RectangularGaussianErfVortex(), src; gradient=true)
    # column mismatch
    @test_throws ArgumentError direct_rectangular!(zeros(3, 2), tgt,
        RectangularGaussianErfVortex(), src)
    # panel functor: non-integral / out-of-range tag rows must throw, not
    # silently contribute zero (they feed unchecked Int truncation on device)
    psrc = zeros(17, 2)
    _pack_panel!(psrc, 1, 1, (SVector(0.0, 0.0, 0.0), SVector(1.0, 0.0, 0.0),
        SVector(0.0, 1.0, 0.0)), 1.0, 0.0, 1e-3)
    _pack_panel!(psrc, 2, 1, (SVector(0.0, 0.0, 1.0), SVector(1.0, 0.0, 1.0),
        SVector(0.0, 1.0, 1.0)), 1.0, 0.0, 1e-3)
    out17 = zeros(12, 3)
    direct_rectangular!(out17, tgt, RectangularPanelInfluence(), psrc)  # sane baseline
    bad = copy(psrc); bad[1, 2] = 7.0
    @test_throws ArgumentError direct_rectangular!(out17, tgt,
        RectangularPanelInfluence(), bad)
    bad = copy(psrc); bad[1, 2] = 1.5
    @test_throws ArgumentError direct_rectangular!(out17, tgt,
        RectangularPanelInfluence(), bad)
    bad = copy(psrc); bad[2, 1] = 5.0
    @test_throws ArgumentError direct_rectangular!(out17, tgt,
        RectangularPanelInfluence(), bad)
    # nv == 2 (open filament) is legal ONLY for tag 3
    filok = copy(psrc); filok[1, 2] = 3.0; filok[2, 2] = 2.0
    direct_rectangular!(out17, tgt, RectangularPanelInfluence(), filok)
    bad = copy(psrc); bad[2, 2] = 2.0   # tag 1 with nv == 2
    @test_throws ArgumentError direct_rectangular!(out17, tgt,
        RectangularPanelInfluence(), bad)
    # panel functor is F64-only: its absolute singularity guards are inert in F32
    @test_throws ArgumentError direct_rectangular!(zeros(Float32, 12, 3),
        Float32.(tgt), RectangularPanelInfluence(), Float32.(psrc))
end

@testset "direct rectangular: CUDA device parity" begin
    cuda_ok = try
        @eval import CUDA
        CUDA.functional()
    catch
        false
    end
    if !cuda_ok
        @info "CUDA not functional; skipping the device parity layer"
    else
        FMR.load_cuda_radix_lifecycle!() || error(
            "CUDA is functional but load_cuda_radix_lifecycle!() failed")
        Random.seed!(51151)
        # worst per-target relative error: a single branch-flipped target must
        # fail the gate rather than be diluted by a global norm
        function per_target_relerr(a, b)
            m = 0.0
            for i in axes(b, 2)
                nb = norm(view(b, :, i))
                m = max(m, norm(view(a, :, i) .- view(b, :, i)) / max(nb, eps()))
            end
            return m
        end

        # --- points: tile-boundary sweep around _RECT_TILE_POINTS = 256 ---
        # (last-partial-tile, single-tile, multi-tile, and 1-target edges).
        # Targets sit 2 units off the source cloud: gates are the
        # well-separated FMA/libdevice bounds, not the rho->0-amplified ones.
        for n_src in (1, 127, 128, 129, 255, 256, 257, 512),
                n_tgt in (1, 255, 256, 257)
            src = rand(7, n_src) .- 0.5
            src[7, :] .= 0.02 .+ 0.08 .* rand(n_src)
            tgt = rand(3, n_tgt) .- 0.5
            tgt[1, :] .+= 2.0
            ref = zeros(12, n_tgt)
            direct_rectangular!(ref, tgt, RectangularGaussianErfVortex(), src;
                gradient=true)
            d_out = CUDA.zeros(Float64, 12, n_tgt)
            direct_rectangular!(d_out, CUDA.CuMatrix(tgt),
                RectangularGaussianErfVortex(), CUDA.CuMatrix(src); gradient=true)
            out = Array(d_out)
            @test per_target_relerr(out[1:3, :], ref[1:3, :]) <= 1e-13
            @test per_target_relerr(out[4:12, :], ref[4:12, :]) <= 1e-12
        end

        # --- points F32 (opt-in path) ---
        let n_src = 300, n_tgt = 140
            src = rand(Float32, 7, n_src) .- 0.5f0
            src[7, :] .= 0.02f0 .+ 0.08f0 .* rand(Float32, n_src)
            tgt = rand(Float32, 3, n_tgt) .- 0.5f0
            tgt[1, :] .+= 2.0f0
            ref = zeros(Float32, 12, n_tgt)
            direct_rectangular!(ref, tgt, RectangularGaussianErfVortex(), src;
                gradient=true)
            d_out = CUDA.zeros(Float32, 12, n_tgt)
            direct_rectangular!(d_out, CUDA.CuMatrix(tgt),
                RectangularGaussianErfVortex(), CUDA.CuMatrix(src); gradient=true)
            @test per_target_relerr(Array(d_out)[1:3, :], ref[1:3, :]) <= 1e-4
        end

        # --- panels: tile-boundary sweep around _RECT_TILE_PANELS = 128 ---
        for n_src in (1, 127, 128, 129, 257), n_tgt in (1, 255, 257)
            src = zeros(17, n_src)
            for q in 1:n_src
                _pack_random_panel!(src, q, randn(3) .* 0.3)
            end
            tgt = rand(3, n_tgt) .- 0.5
            tgt[1, :] .+= 2.0
            ref = zeros(12, n_tgt)
            direct_rectangular!(ref, tgt, RectangularPanelInfluence(), src;
                gradient=true)
            d_out = CUDA.zeros(Float64, 12, n_tgt)
            direct_rectangular!(d_out, CUDA.CuMatrix(tgt),
                RectangularPanelInfluence(), CUDA.CuMatrix(src); gradient=true)
            out = Array(d_out)
            @test per_target_relerr(out[1:3, :], ref[1:3, :]) <= 1e-11
            @test per_target_relerr(out[4:12, :], ref[4:12, :]) <= 1e-10
        end

        # Combined panel scalar-potential + velocity output used by the
        # FLOWPanel block cross-influence route.
        let n_src = 129, n_tgt = 257
            src = zeros(17, n_src)
            for q in 1:n_src
                base = randn(3) .* 0.3
                verts = (SVector{3}(base),
                    SVector{3}(base .+ [0.12, 0.01, 0.0]),
                    SVector{3}(base .+ [0.02, 0.11, 0.03]))
                _pack_panel!(src, q, 5, verts, randn(), randn(), 1e-3)
            end
            tgt = rand(3, n_tgt)
            tgt[1, :] .+= 2.0
            ref = zeros(4, n_tgt)
            direct_rectangular!(ref, tgt, RectangularPanelInfluence(), src;
                scalar_potential=true)
            d_out = CUDA.zeros(Float64, 4, n_tgt)
            direct_rectangular!(d_out, CUDA.CuMatrix(tgt),
                RectangularPanelInfluence(), CUDA.CuMatrix(src);
                scalar_potential=true)
            out = Array(d_out)
            @test per_target_relerr(out[1:3, :], ref[1:3, :]) <= 1e-11
            @test per_target_relerr(out[4:4, :], ref[4:4, :]) <= 1e-11
        end

        # --- panels: ON-SURFACE targets at p018 scale (job 13309844 regime) --
        # The failing pass-3 seam config evaluates body self-influence: targets
        # are control points ON the source sheet, so every pair near the target
        # runs the near-singular branches (_rect_is_self_pair, edge PV limits,
        # solid-angle sign) that the well-separated sweeps above never touch on
        # device -- and device FMA contraction can flip guards that host math
        # does not. Structured wavy sheet, tag-4 (source+ring) columns like the
        # p018 packing, targets = centroids of a subsample + near-plane probes.
        let nu = 96, nv = 192                # 2*95*191 = 36290 panels ~ p018 scale
            xs = range(0.0, 1.0; length=nu)
            ys = range(0.0, 2.0; length=nv)
            zfun(x, y) = 0.05 * sin(3x) * cos(2y)
            nid(i, j) = (j - 1) * nu + i
            P = Matrix{Float64}(undef, 3, nu * nv)
            for j in 1:nv, i in 1:nu
                P[:, nid(i, j)] .= (xs[i], ys[j], zfun(xs[i], ys[j]))
            end
            tris = Vector{NTuple{3,Int}}()
            for j in 1:nv-1, i in 1:nu-1
                push!(tris, (nid(i, j), nid(i + 1, j), nid(i + 1, j + 1)))
                push!(tris, (nid(i, j), nid(i + 1, j + 1), nid(i, j + 1)))
            end
            n_src = length(tris)
            src = zeros(17, n_src)
            rng_vals = [sin(0.7q) + 1.1 for q in 1:n_src]
            for (q, (a, b, c)) in enumerate(tris)
                src[1, q] = 4.0                       # tag 4: source + ring
                src[2, q] = 3.0
                src[3:5, q] .= P[:, a]
                src[6:8, q] .= P[:, b]
                src[9:11, q] .= P[:, c]
                src[12:14, q] .= P[:, a]              # v4 unused for nv=3
                src[15, q] = rng_vals[q]              # sigma
                src[16, q] = sin(1.3q) + 0.2          # Gamma
                src[17, q] = 1.19e-11                 # p018 core_size_panel
            end
            # targets: centroids of every 5th panel (on-surface, incl. exact
            # self pairs) + the same points nudged 1e-8 off-plane
            sel = 1:5:n_src
            cps = Matrix{Float64}(undef, 3, length(sel))
            for (t, q) in enumerate(sel)
                a, b, c = tris[q]
                cps[:, t] .= (P[:, a] .+ P[:, b] .+ P[:, c]) ./ 3
            end
            nudge = [1e-8 * sin(0.37 * (3 * (t - 1) + d)) for d in 1:3, t in 1:length(sel)]
            tgt = hcat(cps, cps .+ nudge)
            n_tgt = size(tgt, 2)
            ref = zeros(12, n_tgt)
            direct_rectangular!(ref, tgt, RectangularPanelInfluence(), src;
                gradient=true)
            d_out = CUDA.zeros(Float64, 12, n_tgt)
            direct_rectangular!(d_out, CUDA.CuMatrix(tgt),
                RectangularPanelInfluence(), CUDA.CuMatrix(src); gradient=true)
            out = Array(d_out)
            @test per_target_relerr(out[1:3, :], ref[1:3, :]) <= 1e-11
            @test per_target_relerr(out[4:12, :], ref[4:12, :]) <= 1e-10
        end

        # --- device accumulate contract (+=): second call doubles ---
        let
            src = rand(7, 64) .- 0.5
            src[7, :] .= 0.05
            tgt = rand(3, 32) .+ 2.0
            d_src = CUDA.CuMatrix(src)
            d_tgt = CUDA.CuMatrix(tgt)
            d_out = CUDA.zeros(Float64, 3, 32)
            direct_rectangular!(d_out, d_tgt, RectangularGaussianErfVortex(), d_src)
            once = Array(d_out)
            direct_rectangular!(d_out, d_tgt, RectangularGaussianErfVortex(), d_src)
            @test Array(d_out) ≈ 2 .* once rtol=1e-14
        end

        # --- degenerate sizes ---
        let
            d_src = CUDA.CuMatrix(rand(7, 8))
            @test direct_rectangular!(CUDA.zeros(Float64, 3, 0),
                CUDA.zeros(Float64, 3, 0), RectangularGaussianErfVortex(),
                d_src) isa CUDA.CuMatrix
            d_out = CUDA.zeros(Float64, 3, 5)
            direct_rectangular!(d_out, CUDA.CuMatrix(rand(3, 5)),
                RectangularGaussianErfVortex(), CUDA.zeros(Float64, 7, 0))
            @test all(iszero, Array(d_out))
        end
    end
end
