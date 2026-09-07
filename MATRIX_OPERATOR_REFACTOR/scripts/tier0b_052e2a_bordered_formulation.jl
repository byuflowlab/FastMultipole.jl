# 052e.2a Tier 0B — bordered :area_mean formulation proof (stage 1 only)
# Pre-registration (LOCKED 2026-09-07):
#   MATRIX_OPERATOR_REFACTOR/052e2a-tier0b-preregistration-2026-09-07.md
# Run:  julia --project=/Users/ryan/Dropbox/research/projects/FLOWPanel.jl \
#             --threads=4 \
#             MATRIX_OPERATOR_REFACTOR/scripts/tier0b_052e2a_bordered_formulation.jl
#
# System under test (production route, FLOWPanel_formulation.jl):
#   sigma -> _source_potential! -> _build_green_solve_state(body,:area_mean)
#         -> _green_solve_q!  (bordered (I-B)q + a*lambda = S*sigma, a'q = 0)
#   telemetry: _green_B_product! residual, gauge defect, surface_hodge_trace!
# Oracle (independent of the production influence machinery): per-panel sums of
#   the Tier 0A-validated public wrapper pnl.induced(x, oracle_body, i, switch)
#   over the wake's nonzero-strength panels (A10 route).
#
# Registered implementation notes (recorded, not gate edits):
# - sigma convention: sigma_i = -dot(n_i, u_wake(x_i))  (FLOWPanel source
#   density sigma = -g; theory note section 2). Flux metric is sign-invariant.
# - The mesh spans y in [-b/2, +b/2], so the locked elliptic distribution
#   (max at midspan, zero at tips) is mu(y) = MU0*sqrt(1-(2y/b)^2).
# - The Kutta (live) row is realized as wake row 1 with exactly zero strength:
#   zero-strength constant-doublet panels induce nothing, so the oracle's
#   singular support starts one row length (0.5c) behind the TE, honoring the
#   prereg support condition without touching live_rows Route-B state.
# - Hodge diagnostics (B6) run on L1-L3 only: surface_hodge_trace! builds a
#   dense (m+1) x N least-squares system, ~9 GB at L4. Recorded as a
#   diagnostic-coverage limitation; B6 is a trend record, not a finest gate.

import FLOWPanel as pnl
using StaticArrays, Printf, Statistics, LinearAlgebra, SHA

const HERE = @__DIR__
include(joinpath(HERE, "..", "..", "..", "FLOWPanel.jl", "examples", "pitching_wing.jl"))

# TIER0B_SMOKE=1 runs a tiny UNREGISTERED configuration (mechanical/API check
# only; outputs go to a temp dir; its gate values are meaningless and must not
# be interpreted). The registered run uses the defaults below.
const SMOKE = get(ENV, "TIER0B_SMOKE", "0") == "1"

const DATADIR = SMOKE ? mktempdir() : joinpath(HERE, "..", "data", "052e2a-tier0b")
mkpath(DATADIR)

# ---------------- locked fixture constants -----------------------------------

const C_FIX = 0.76               # chord [m]
const B_FIX = 2.7                # span [m]
const THICK = 0.12               # NACA0012
const MU0 = 1.0
const NFREE = SMOKE ? 6 : 40     # wake rows; row 1 = zero-strength Kutta row
const ROWLEN = 0.5 * C_FIX       # uniform row length -> 20c total
const CORE_ORACLE = 1e-8
const AOAS = SMOKE ? (3.0,) : (0.0, 7.0)   # cases C1, C2 (wake incl., deg)
const LEVELS = SMOKE ? [(41, 5, 3), (61, 6, 4)] :
    [(81, 7, 5), (121, 10, 7), (181, 15, 11), (271, 22, 15)]
const HODGE_MAX_LEVEL = SMOKE ? 2 : 3

const SW_PV = pnl.FastMultipole.DerivativesSwitch(true, true, false)
const DIRECT = pnl.DirectBackend()

# ---------------- fixture construction ---------------------------------------

function make_body(lev)
    n_airfoil, n_span, n_endcap = LEVELS[lev]
    body = build_pitching_wing_body(C_FIX, B_FIX; n_span, n_airfoil, n_endcap,
        thickness=THICK, semiinfinite_wake=false)
    # FLOWPanel fills these lazily (solver does it in solve()); without them
    # controlpoints/normals are all-zero and sigma degenerates to -0.0
    pnl.calc_normals!(body)
    pnl.calc_controlpoints!(body)
    return body
end

"Prescribed flat wake: row 1 zero-strength (Kutta), rows 2:NFREE elliptic."
function make_wake(body, wakedir; mu0=MU0)
    wake = pnl.PanelWake(body; nwakerows=NFREE, include_final_filament=false)
    pnl.update_TE!(wake, body)
    step = wakedir ./ norm(wakedir) .* ROWLEN
    for nodes in wake.nodes
        first_row = copy(view(nodes, :, 1, :))
        for row in 1:NFREE+1
            view(nodes, :, row, :) .= first_row .+ (row - 1) .* step
        end
    end
    for (nodes, strength) in zip(wake.nodes, wake.strength)
        for j in axes(strength, 3)   # spanwise strips
            ymid = (nodes[2, 1, j] + nodes[2, 1, j+1]) / 2
            mu = mu0 * sqrt(max(0.0, 1.0 - (2ymid / B_FIX)^2))
            strength[1, 1, j] = 0.0             # Kutta row: zero strength
            for row in 2:NFREE
                strength[1, row, j] = mu
            end
        end
    end
    wake.nwakes[] = NFREE
    return wake
end

"Oracle body: NonLiftingBody{ConstantDoublet} from the nonzero wake panels."
function make_oracle(wake)
    verts = Vector{Float64}[]
    cells = Vector{Int}[]
    mus = Float64[]
    for (nodes, strength) in zip(wake.nodes, wake.strength)
        nspan = size(nodes, 3)
        idx = Dict{Tuple{Int,Int},Int}()
        node_id(r, j) = get!(idx, (r, j)) do
            push!(verts, nodes[:, r, j]); length(verts)
        end
        # two triangles per (planar) wake quad — exact for constant doublets
        # by solid-angle additivity; FLOWPanel bodies are triangle meshes
        for r in 2:NFREE, j in 1:nspan-1
            n11, n21 = node_id(r, j), node_id(r + 1, j)
            n22, n12 = node_id(r + 1, j + 1), node_id(r, j + 1)
            push!(cells, [n11, n21, n22])
            push!(cells, [n11, n22, n12])
            push!(mus, strength[1, r, j])
            push!(mus, strength[1, r, j])
        end
    end
    onodes = reduce(hcat, verts)
    ocells = reduce(hcat, cells)
    obody = pnl.NonLiftingBody{pnl.ConstantDoublet}(onodes, ocells;
        core_size=CORE_ORACLE)
    pnl.calc_normals!(obody)
    pnl.calc_controlpoints!(obody)
    obody.strength[:, 1] .= mus
    return obody
end

"Direct oracle evaluation (A10 wrapper route) at targets (3 x N)."
function oracle_eval(obody, targets)
    N = size(targets, 2)
    phi = zeros(N)
    vel = zeros(3, N)
    Threads.@threads for i in 1:N
        x = SVector{3}(targets[1, i], targets[2, i], targets[3, i])
        p = 0.0
        u = SVector(0.0, 0.0, 0.0)
        for s in 1:obody.ncells
            ps, us, _ = pnl.induced(x, obody, s, SW_PV; core_size=CORE_ORACLE)
            p += ps
            u += us
        end
        phi[i] = p
        vel[:, i] .= u
    end
    return phi, vel
end

# ---------------- geometry helpers -------------------------------------------

"Exact point-to-triangle distance (Ericson, Real-Time Collision Detection)."
function dist_point_tri(p, a, b, c)
    ab = b - a; ac = c - a; ap = p - a
    d1 = dot(ab, ap); d2 = dot(ac, ap)
    (d1 <= 0 && d2 <= 0) && return norm(p - a)
    bp = p - b; d3 = dot(ab, bp); d4 = dot(ac, bp)
    (d3 >= 0 && d4 <= d3) && return norm(p - b)
    vc = d1 * d4 - d3 * d2
    if vc <= 0 && d1 >= 0 && d3 <= 0
        v = d1 / (d1 - d3); return norm(p - (a + v * ab))
    end
    cp = p - c; d5 = dot(ab, cp); d6 = dot(ac, cp)
    (d6 >= 0 && d5 <= d6) && return norm(p - c)
    vb = d5 * d2 - d1 * d6
    if vb <= 0 && d2 >= 0 && d6 <= 0
        w = d2 / (d2 - d6); return norm(p - (a + w * ac))
    end
    va = d3 * d6 - d5 * d4
    if va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return norm(p - (b + w * (c - b)))
    end
    den = 1 / (va + vb + vc)
    v = vb * den; w = vc * den
    return norm(p - (a + ab * v + ac * w))
end

"Min distance from each control point to any nonzero-strength oracle panel."
function min_clearance(obody, targets)
    N = size(targets, 2)
    dmin = fill(Inf, N)
    Threads.@threads for i in 1:N
        p = SVector{3}(targets[1, i], targets[2, i], targets[3, i])
        best = Inf
        for s in 1:obody.ncells
            cell = view(obody.cells, :, s)
            v = ntuple(k -> SVector{3}(view(obody.nodes, :, cell[k])), 3)
            best = min(best, dist_point_tri(p, v[1], v[2], v[3]))
        end
        dmin[i] = best
    end
    return minimum(dmin)
end

"Body panels with any vertex on the TE line x = c (upper+lower rows, cap tips)."
function te_adjacent(body)
    tol = max(100 * eps() * C_FIX, 1e-8 * C_FIX)
    [any(abs(body.nodes[1, body.cells[k, j]] - C_FIX) <= tol
         for k in axes(body.cells, 1)) for j in axes(body.cells, 2)]
end

# ---------------- metrics (locked definitions) -------------------------------

align(q, a) = q .- dot(a, q) / sum(a)
rmsA(q, a) = sqrt(sum(a .* q .^ 2) / sum(a))

function trace_metrics(q, qref, a, temask)
    qt, rt = align(q, a), align(qref, a)
    scale = rmsA(rt, a)
    E_q = sqrt(sum(a .* (qt .- rt) .^ 2) / sum(a)) / scale
    E_inf_all = maximum(abs.(qt .- rt)) / scale
    keep = .!temask
    E_inf_ex = maximum(abs.(qt[keep] .- rt[keep])) / scale
    return E_q, E_inf_all, E_inf_ex
end

# ---------------- per-case / per-level run -----------------------------------

struct LevelResult
    N::Int
    E_q::Float64
    E_inf_all::Float64
    E_inf_ex::Float64
    flux::Float64
    lambda::Float64
    residual::Float64
    gauge_defect::Float64
    hodge_defect::Float64      # NaN if skipped
    hodge_mismatch::Float64    # NaN if skipped
    clearance::Float64
    lin_err::Float64           # NaN except finest level
    xcheck_pot::Float64        # influence!-route vs oracle (report only)
    xcheck_vel::Float64
end

function run_level(case_id, aoa, lev; linearity::Bool)
    body = make_body(lev)
    N = body.ncells
    wakedir = [cosd(aoa), 0.0, sind(aoa)]
    set_wake_Das!(body, wakedir)
    wake = make_wake(body, wakedir)
    obody = make_oracle(wake)

    qref, uw = oracle_eval(obody, body.controlpoints)
    sigma = [-dot(view(uw, :, i), view(body.normals, :, i)) for i in 1:N]
    a = pnl._panel_areas(body)
    temask = te_adjacent(body)
    clearance = min_clearance(obody, body.controlpoints)
    clearance > 0 || error("case $case_id L$lev: control point ON wake surface")

    # cross-check (report-only): production PanelWake -> influence! route
    qf = zeros(N)
    pnl._wake_potential!(qf, body, (wake,), DIRECT)
    uf = zeros(3, N)
    pnl._wake_panel_velocity!(uf, body, (wake,), DIRECT)
    xp = norm(qf .- qref) / max(norm(qref), eps())
    xv = norm(uf .- uw) / max(norm(uw), eps())

    # ---- system under test: production bordered :area_mean route ----
    Ssigma = zeros(N)
    pnl._source_potential!(Ssigma, body, sigma, DIRECT)
    gs = pnl._build_green_solve_state(body, :area_mean)
    q = copy(pnl._green_solve_q!(gs, Ssigma))
    lambda = gs.sol_b[end]

    Bq = similar(q)
    pnl._with_green_scratch(body) do
        pnl._green_B_product!(Bq, body, q, DIRECT)
    end
    residual = norm(q .- Bq .+ a .* lambda .- Ssigma) / max(norm(Ssigma), eps())
    gauge_defect = abs(dot(a, q)) / max(norm(a) * norm(q), eps())

    # B8 linearity at the finest level: reused factorization + kernel spot check
    lin_err = NaN
    if linearity
        q2 = copy(pnl._green_solve_q!(gs, 2 .* Ssigma))
        e_solve = norm(q2 .- 2 .* q) / max(norm(2 .* q), eps())
        obody2 = make_oracle(make_wake(body, wakedir; mu0=2.0))
        nspot = min(32, N)
        spots = round.(Int, range(1, N, length=nspot))
        p2, _ = oracle_eval(obody2, body.controlpoints[:, spots])
        e_kern = maximum(abs.(p2 .- 2 .* qref[spots])) /
                 max(maximum(abs.(2 .* qref[spots])), eps())
        lin_err = max(e_solve, e_kern)
    end

    # B6 Hodge diagnostics (L1..HODGE_MAX_LEVEL)
    hodge_defect = NaN
    hodge_mismatch = NaN
    qh = fill(NaN, N)
    if lev <= HODGE_MAX_LEVEL
        qh .= 0.0
        hodge_defect = pnl.surface_hodge_trace!(qh, body, uw)
        qt, ht = align(q, a), align(qh, a)
        hodge_mismatch = rmsA(qt .- ht, a) / rmsA(align(qref, a), a)
    end

    flux = abs(sum(a .* sigma)) / sum(a .* abs.(sigma))
    E_q, E_inf_all, E_inf_ex = trace_metrics(q, qref, a, temask)

    # per-level CSV of all sampled values
    open(joinpath(DATADIR, "trace_$(case_id)_L$(lev).csv"), "w") do io
        println(io, "x,y,z,area,te_adjacent,sigma,q_ref,q_green,q_hodge")
        for i in 1:N
            @printf(io, "%.9e,%.9e,%.9e,%.9e,%d,%.9e,%.9e,%.9e,%.9e\n",
                body.controlpoints[1, i], body.controlpoints[2, i],
                body.controlpoints[3, i], a[i], temask[i], sigma[i],
                qref[i], q[i], qh[i])
        end
    end

    @printf("  %s L%d N=%d: E_q=%.3e Einf(ex)=%.3e F=%.2e lam=%.2e res=%.1e gd=%.1e clr=%.3f xchk(p,v)=(%.1e,%.1e)\n",
        case_id, lev, N, E_q, E_inf_ex, flux, lambda, residual, gauge_defect,
        clearance, xp, xv)

    return LevelResult(N, E_q, E_inf_all, E_inf_ex, flux, lambda, residual,
        gauge_defect, hodge_defect, hodge_mismatch, clearance, lin_err, xp, xv)
end

# ---------------- gates ------------------------------------------------------

results_txt = String[]
gate(id, ok, val, lim) = push!(results_txt,
    @sprintf("%-8s %-4s  value=%s  gate=%s", id, ok ? "PASS" : "FAIL",
        string(val), string(lim)))

monotone_dec(v) = all(diff(v) .< 0)
orders(E, N) = [log(E[i] / E[i+1]) / log(sqrt(N[i+1] / N[i]))
                for i in 1:length(E)-1]

all_pass = Bool[]
case_results = Dict{String,Vector{LevelResult}}()

for (ci, aoa) in enumerate(AOAS)
    case_id = "C$ci"
    println("== case $case_id (AOA=$(aoa) deg) ==")
    levs = [run_level(case_id, aoa, lev; linearity=(lev == length(LEVELS)))
            for lev in 1:length(LEVELS)]
    case_results[case_id] = levs
    fin = levs[end]
    Es = [l.E_q for l in levs]
    Ns = [l.N for l in levs]

    b1 = fin.E_q <= 1e-2
    gate("B1/$case_id", b1, fin.E_q, 1e-2)
    b2 = monotone_dec(Es)
    ordstr = join([@sprintf("%.2f", o) for o in orders(Es, Ns)], ",")
    push!(results_txt, @sprintf("%-8s %-4s  E_q=%s orders=[%s] (order recorded, not gated)",
        "B2/$case_id", b2 ? "PASS" : "FAIL",
        join([@sprintf("%.3e", e) for e in Es], ","), ordstr))
    fluxes = [l.flux for l in levs]
    b3 = all(fluxes .<= 1e-3) && monotone_dec(fluxes)
    gate("B3/$case_id", b3, maximum(fluxes), 1e-3)
    push!(results_txt, @sprintf("%-8s REC   lambda=[%s]", "B4/$case_id",
        join([@sprintf("%.3e", l.lambda) for l in levs], ",")))
    b5 = fin.residual <= 1e-10 && fin.gauge_defect <= 1e-12
    gate("B5/$case_id", b5, (fin.residual, fin.gauge_defect), (1e-10, 1e-12))
    hd = [l.hodge_defect for l in levs if !isnan(l.hodge_defect)]
    hm = [l.hodge_mismatch for l in levs if !isnan(l.hodge_mismatch)]
    b6 = monotone_dec(hd) && monotone_dec(hm)
    push!(results_txt, @sprintf("%-8s %-4s  defect=[%s] mismatch=[%s] (L1-L%d; L4 skipped: memory)",
        "B6/$case_id", b6 ? "PASS" : "FAIL",
        join([@sprintf("%.3e", x) for x in hd], ","),
        join([@sprintf("%.3e", x) for x in hm], ","), HODGE_MAX_LEVEL))
    b7 = fin.E_inf_ex <= 3e-2
    push!(results_txt, @sprintf("%-8s %-4s  value=%.3e gate=3.0e-02 (all-panel %.3e recorded)",
        "B7/$case_id", b7 ? "PASS" : "FAIL", fin.E_inf_ex, fin.E_inf_all))
    b8 = fin.lin_err <= 1e-12
    gate("B8/$case_id", b8, fin.lin_err, 1e-12)

    append!(all_pass, [b1, b2, b3, b5, b6, b7, b8])

    # structural kill rule (advisory print; ruling is Ryan's)
    if fin.E_q > 0.2 || !b2
        push!(results_txt,
            "KILL-RULE/$case_id TRIGGERED: analytic-normal-velocity reconstruction " *
            "exceeds 20% at finest mesh or fails to improve under refinement")
    end
end

b9 = all(all_pass)
gate("B9", b9, b9 ? "all cases pass" : "failures above", "required")

# ---------------- provenance + summary ---------------------------------------

function gitinfo(path)
    sha = strip(read(`git -C $path rev-parse HEAD`, String))
    dirty = !isempty(strip(read(`git -C $path status --porcelain`, String)))
    dh = dirty ? bytes2hex(sha256(read(`git -C $path diff`)))[1:12] : ""
    "$sha $(dirty ? "DIRTY tracked-diff=$dh" : "clean")"
end
script_sha = bytes2hex(sha256(read(@__FILE__)))[1:12]

open(joinpath(DATADIR, "gates.txt"), "w") do io
    println(io, "date=$(chomp(read(`date +%Y-%m-%d`, String))) julia=$(VERSION) threads=$(Threads.nthreads()) script_sha256=$(script_sha)")
    println(io, "FLOWPanel: $(gitinfo(joinpath(HERE, "..", "..", "..", "FLOWPanel.jl")))")
    println(io, "FastMultipole: $(gitinfo(joinpath(HERE, "..", "..")))")
    println(io, "call chain: pnl.induced (oracle) | _source_potential! -> _build_green_solve_state(:area_mean) -> _green_solve_q! | _green_B_product! residual | surface_hodge_trace!")
    println(io, "sigma convention: sigma = -n.u_wake ; elliptic mu(y)=MU0*sqrt(1-(2y/b)^2), y in [-b/2,b/2]")
    for (cid, levs) in sort(collect(case_results); by=first)
        for (lev, l) in enumerate(levs)
            @printf(io, "%s L%d N=%d E_q=%.6e Einf_all=%.6e Einf_ex=%.6e flux=%.3e lambda=%.6e res=%.3e gd=%.3e hodge=(%.3e,%.3e) clr=%.4f lin=%.3e xchk=(%.3e,%.3e)\n",
                cid, lev, l.N, l.E_q, l.E_inf_all, l.E_inf_ex, l.flux, l.lambda,
                l.residual, l.gauge_defect, l.hodge_defect, l.hodge_mismatch,
                l.clearance, l.lin_err, l.xcheck_pot, l.xcheck_vel)
        end
    end
    foreach(l -> println(io, l), results_txt)
end

println("\n===== TIER 0B GATES =====")
foreach(println, results_txt)
println("=========================")
