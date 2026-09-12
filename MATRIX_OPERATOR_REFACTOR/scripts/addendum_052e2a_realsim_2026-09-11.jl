# 052e.2a realsim addendum — cross-formulation convergence test (three routes)
# Pre-registration (LOCKED 2026-09-09):
#   MATRIX_OPERATOR_REFACTOR/052e2a-addendum-realsim-preregistration-2026-09-08.md
# Smoke (unregistered, tiny mesh, temp outputs):
#   ADDENDUM_SMOKE=1 julia --project=/Users/ryan/Dropbox/research/projects/FLOWPanel.jl \
#       --threads=4 MATRIX_OPERATOR_REFACTOR/scripts/addendum_052e2a_realsim_2026-09-09.jl
# Registered run: same command without ADDENDUM_SMOKE, nohup-detached; logs+CSVs
#   under MATRIX_OPERATOR_REFACTOR/data/052e2a-addendum-realsim/.
#
# Harness-realization choices RECORDED PRE-LAUNCH (prereg leaves these to the
# harness; none alters a locked value):
# R1. Phase A ("wake strengths coupled to the solve through the standard
#     rigid-wake/Kutta machinery" on frozen flat geometry) is realized as a
#     fixed-point iteration inside ONE simulate! call: the maneuver! hook
#     (which runs before update_TE! and the solve each step) re-prescribes the
#     flat 20c wake geometry every step and sets every wake row's strength per
#     strip to the previous solve's TE jump Gamma_j. Step 0 starts from a
#     zero-strength wake. Convergence delta = max|dGamma|/max|Gamma| must reach
#     TOL_A within NITER_A steps (else G1 fails).
# R2. Phase B production defaults recorded pre-launch: dt = 0.5*c/U (the
#     example's c_per_dt=0.5), NSAMP_B=61 solve samples (60 sheds ~ 30c wake);
#     W1 PanelWake retains all rows (nwakerows=NSAMP_B+2,
#     include_final_filament=false); W2 PanelParticleWake with production
#     shedding OverlapPPS(1.3,2) trailing+unsteady, nwakerows=4 (short panel
#     buffer so the wake is predominantly particles; warm-up >= 4 steps),
#     max_particles=200_000 (headroom; example default is 20_000).
#     Steadiness recorded (not gated): |Gtot drift| over last 5 steps.
# R3. C_L is the Kutta-Joukowski scalar summary CL = 2*sum(Gamma_j*dy_j)/(U*b*c)
#     (identical definition across routes; a pressure-based CL is not part of
#     the registered measurements).
# R4. G3 (Green residual <= 1e-10, gauge defect <= 1e-11) is gated at the
#     measurement solve of each R-GR run (Phase A: final iteration; Phase B:
#     final step), tier0b B5 precedent; the full per-step history is recorded.
# R5. G4 is gated on the Phase B W1 R-GR oracle sequence (the converged solve
#     of the real simulation); the Phase A oracle sequence is computed and
#     recorded alongside it.
# R6. G2 is gated on the in-solve wake-node hash (callback fires post-solve,
#     pre-convection) at every Phase A step: all steps and all routes must see
#     bit-identical wake nodes per level. Pre-run prescription hashes recorded.
# R7. TE-adjacent mask (for E_inf exclusion) = body panels sharing a node with
#     any shedding edge (the pitched body has no fixed x=c TE line).
# R8. Kutta closure is the default legacy pair (RigidTransitionAttachment +
#     JumpKutta): closure c == 0 by construction inside the solve
#     (FLOWPanel_kutta.jl:68-70); the recorded "Kutta residual" is the
#     cross-check max_j |Gamma_harness_j - (mu_i - mu_j)| against FLOWPanel's
#     own _get_wakestrength_mu at the measurement solve (0 up to roundoff
#     evidences the identical closure convention).
# R9. Frozen-particle Phase A W2 variant: NOT RUN (allowed by lock L2).
#     Reason: placing a particle set whose induced field matches the prescribed
#     panel sheet at matched discretization is its own study; P4 uses the
#     prereg fallback (matched Phase B final-step snapshot, same t_range/dt/
#     level, R-GR, aligned traces).
# R10. sigma convention sigma = -n.u_wake; metrics (align, rms_A, E_q, E_inf)
#     and oracle machinery are the stage-1 locked definitions, reused from the
#     frozen tier0b script (read, not edited). Oracle includes ALL active wake
#     rows at their solve-time strengths (in-sim wake attaches at the TE, so
#     min clearance ~ 0 there; clearance recorded over all and non-TE points,
#     not gated — the stage-1 clearance>0 guard was for the manufactured
#     fixture, meaningless for an attached wake).
# R11. Das (TE wake-attachment vector) follows the production convention of
#     prepare_pitching_wing (examples/pitching_wing.jl:953-966): frames are
#     built FIRST (pitching_wing_frame rotates body AND Das), then
#     set_wake_Das!(body, VINF; magnitude=0.05*c) — freestream direction,
#     das_chord_fraction=0.05 production default. The L4 probes' seed pattern
#     (unit-magnitude Das set before frames, hence rotated with the body) put
#     wake row 1 one meter behind the TE along the chord line; found and
#     corrected at smoke stage (mechanical harness fix, no locked value).
# R12. SUPERSESSION (Ryan 2026-09-11): W2 shedding OverlapPPS(1.3,2) ->
#     OverlapPPS(2.4,2) both arms, all levels, after the registered
#     2026-09-10 run G1-failed (BW2-VTS-L4 vortex-stretching runaway;
#     see 052e2a-addendum-realsim-supersession-2026-09-11.md). Outputs
#     go to data/052e2a-addendum-realsim-v2 to preserve the failed
#     run's evidence. No other change from the 2026-09-09 harness.

import FLOWPanel as pnl
using StaticArrays, Printf, Statistics, LinearAlgebra, SHA

const HERE = @__DIR__
include(joinpath(HERE, "..", "..", "..", "FLOWPanel.jl", "examples", "pitching_wing.jl"))

const SMOKE = get(ENV, "ADDENDUM_SMOKE", "0") == "1"
const DATADIR = SMOKE ?
    get(ENV, "ADDENDUM_SMOKE_DIR", joinpath(tempdir(), "addendum_052e2a_smoke")) :
    joinpath(HERE, "..", "data", "052e2a-addendum-realsim-v2")
mkpath(DATADIR)

# ---------------- locked fixture constants -----------------------------------
const C_FIX = 0.76
const B_FIX = 2.7
const THICK = 0.12
const AOA = 30.0                       # deg (lock L1)
const UMAG = 1.0                       # lock L1
const VINF = SVector{3}(UMAG, 0.0, 0.0)
Uinf_f(t) = VINF
noman!(frames, systems, wakes, t) = nothing

const LEVELS = SMOKE ? [(41, 5, 3), (61, 6, 4)] :
    [(81, 7, 5), (121, 10, 7), (181, 15, 11), (271, 22, 15)]   # stage-1 L1-L4
const NFREE = SMOKE ? 6 : 40           # Phase A wake rows (40 x 0.5c = 20c)
const ROWLEN = 0.5 * C_FIX
const DT = 0.5 * C_FIX / UMAG          # production c_per_dt = 0.5
const NITER_A = SMOKE ? 50 : 80
const TOL_A = 1e-8
const NSAMP_B = SMOKE ? 6 : 61
const W2_NWAKEROWS = SMOKE ? 2 : 4
const W2_MAXPART = SMOKE ? 5000 : 200_000
const CORE_ORACLE = 1e-8
const DAS = 0.05 * C_FIX               # production das_chord_fraction=0.05 (R11)

const DIRECT = pnl.DirectBackend()
const SW_PV = pnl.FastMultipole.DerivativesSwitch(true, true, false)
const PIVOT = SVector{3}(0.25 * C_FIX, 0.0, 0.0)

# ---------------- bodies (L4-verified construction paths) --------------------
function make_capped_wing(lev)
    n_airfoil, n_span, n_endcap = LEVELS[lev]
    wing = build_pitching_wing_body(C_FIX, B_FIX; n_span, n_airfoil, n_endcap,
        thickness=THICK, semiinfinite_wake=false)
    pnl.calc_normals!(wing)
    pnl.calc_controlpoints!(wing)
    return wing
end

function make_uncapped_neumann(lev)
    n_airfoil, n_span, n_endcap = LEVELS[lev]
    nodes, cells = pitching_wing_mesh(C_FIX, B_FIX; n_span, n_airfoil,
        n_endcap, caps=false, thickness=THICK)
    bodytype = pnl.RigidWakeBody{pnl.ConstantDoublet, 1, Float64, false}
    opts = (; core_size=1e-6 * C_FIX, kernelcutoff=1e-12 * C_FIX,
        semiinfinite_wake=false, watertight=false)
    base = bodytype(nodes, cells, zeros(Int, 6, 0); opts...)
    shedding = calc_pitching_wing_shedding(base.nodes, base.cells, C_FIX)
    body = bodytype(copy(base.nodes), copy(base.cells), [shedding]; opts...)
    pnl.calc_normals!(body)
    pnl.calc_controlpoints!(body)
    return body
end

const ROUTES = (
    (id="VTS", neumann=false, gr=false,
     form=() -> pnl.VelocityThroughSources(require_outer_convergence=true)),
    (id="GR", neumann=false, gr=true,
     form=() -> pnl.GreenReconstruction(gauge=:area_mean)),
    (id="NEU", neumann=true, gr=false,
     form=() -> pnl.VelocityThroughSources(require_outer_convergence=true)),
)

# ---------------- shedding / circulation helpers -----------------------------
struct ShedInfo
    isurf::Vector{Int}
    icol::Vector{Int}
    p_up::Vector{Int}
    p_lo::Vector{Int}
    y::Vector{Float64}
    dy::Vector{Float64}
    te_nodes::Set{Int}
end

function shed_info(body)
    isurfs = Int[]; icols = Int[]; pups = Int[]; plos = Int[]
    ys = Float64[]; dys = Float64[]
    tenodes = Set{Int}()
    for (isurf, sh) in enumerate(body.shedding)
        for i in axes(sh, 2)
            pi = sh[1, i]
            n1 = body.cells[sh[2, i], pi]
            n2 = body.cells[sh[3, i], pi]
            push!(ys, (body.nodes[2, n1] + body.nodes[2, n2]) / 2)
            push!(dys, abs(body.nodes[2, n2] - body.nodes[2, n1]))
            push!(tenodes, n1, n2)
            pj = sh[4, i]
            if pj != -1
                push!(tenodes, body.cells[sh[5, i], pj], body.cells[sh[6, i], pj])
            end
            push!(isurfs, isurf); push!(icols, i); push!(pups, pi); push!(plos, pj)
        end
    end
    return ShedInfo(isurfs, icols, pups, plos, ys, dys, tenodes)
end

"Gamma_j = mu_upper - mu_lower (doublet column = last strength column)."
gamma_of(strength, info::ShedInfo) =
    [strength[info.p_up[k], end] -
     (info.p_lo[k] == -1 ? 0.0 : strength[info.p_lo[k], end])
     for k in eachindex(info.p_up)]

gtot_of(G, info) = sum(G .* info.dy)
cl_of(G, info) = 2 * gtot_of(G, info) / (UMAG * B_FIX * C_FIX)

te_mask(body, info) = [any(body.cells[k, j] in info.te_nodes
                           for k in axes(body.cells, 1))
                       for j in axes(body.cells, 2)]

"Cross-check our Gamma indexing against FLOWPanel's own TE-jump helper (R8)."
function kutta_xcheck(body, info, strength_cap)
    saved = copy(body.strength)
    body.strength .= strength_cap
    err = 0.0
    try
        for k in eachindex(info.p_up)
            si, sj = pnl._get_wakestrength_mu(body, info.icol[k], info.isurf[k])
            Gk = strength_cap[info.p_up[k], end] -
                 (info.p_lo[k] == -1 ? 0.0 : strength_cap[info.p_lo[k], end])
            err = max(err, abs(Gk - (si - sj)))
        end
    catch
        err = NaN
    end
    body.strength .= saved
    return err
end

# ---------------- wake prescription (Phase A) --------------------------------
"Flat wake along VINF from the (update_TE!-set) row-1 nodes; all rows = Gam."
function prescribe_wake!(wake, Gam, info)
    step = VINF ./ norm(VINF) .* ROWLEN
    for nodes in wake.nodes
        fr = copy(view(nodes, :, 1, :))
        for r in axes(nodes, 2)
            view(nodes, :, r, :) .= fr .+ (r - 1) .* step
        end
    end
    for k in eachindex(info.icol)
        s = wake.strength[info.isurf[k]]
        for r in 1:NFREE
            s[1, r, info.icol[k]] = Gam[k]
        end
    end
    wake.nwakes[] = NFREE
    return wake
end

wakehash(wake) = bytes2hex(sha256(reinterpret(UInt8,
    vec(reduce(hcat, [reshape(n, 3, :) for n in wake.nodes])))))[1:16]

function phaseA_maneuver(body, wake, info)
    return (frames, systems, wakes, t) -> begin
        Gam = gamma_of(body.strength, info)
        pnl.update_TE!(wake, body)
        prescribe_wake!(wake, Gam, info)
        nothing
    end
end

# ---------------- telemetry capture ------------------------------------------
mutable struct RunStore
    steps::Vector{Any}
    gammas::Vector{Vector{Float64}}
    gr::Vector{Any}
    cap::Any
end
RunStore() = RunStore([], Vector{Float64}[], [], nothing)

function state_scalars(st)
    parts = String[]
    for fn in fieldnames(typeof(st))
        v = getfield(st, fn)
        if v isa Number
            push!(parts, "$fn=$v")
        elseif v isa Base.RefValue && v[] isa Number
            push!(parts, "$fn=$(v[])")
        end
    end
    return join(parts, ";")
end

np_of(wake) = hasproperty(wake, :pfield) ?
    (try Int(pnl.FLOWVPM.get_np(wake.pfield)) catch; -1 end) : -1

function make_callback(body, wake, info, store::RunStore;
                       gr::Bool, capture_at::Int, do_hash::Bool)
    return (nt) -> begin
        st = nt.formulation_state
        G = gamma_of(body.strength, info)
        delta = isempty(store.gammas) ? NaN :
            maximum(abs.(G .- store.gammas[end])) / max(maximum(abs.(G)), eps())
        push!(store.gammas, G)
        push!(store.steps, (i=nt.i_step, delta, gtot=gtot_of(G, info),
            maxs=maximum(abs, body.strength),
            finite=all(isfinite, body.strength),
            np=np_of(wake),
            hash=(do_hash ? wakehash(wake) : ""),
            scal=state_scalars(st)))
        if gr
            push!(store.gr, (i=nt.i_step, q=copy(st.green.q),
                sigma=copy(st.sigma),
                lambda=pnl._green_lambda(st.green),
                last_recompute=st.last_recompute[]))
        end
        if nt.i_step == capture_at
            store.cap = (strength=copy(body.strength),
                nodes=(hasproperty(wake, :nodes) ?
                       [copy(n) for n in wake.nodes] : nothing),
                wstrength=(hasproperty(wake, :strength) ?
                           [copy(s) for s in wake.strength] : nothing),
                nwakes=(hasproperty(wake, :nwakes) ? wake.nwakes[] : -1),
                np=np_of(wake))
        end
        nothing
    end
end

# ---------------- stage-1 locked metrics (tier0b definitions) ----------------
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

# ---------------- oracle (tier0b machinery, generalized to active rows) ------
"NonLiftingBody{ConstantDoublet} from all nonzero-strength active wake panels."
function make_oracle(cap)
    verts = Vector{Float64}[]
    cells = Vector{Int}[]
    mus = Float64[]
    for (nodes, strength) in zip(cap.nodes, cap.wstrength)
        nspan = size(nodes, 3)
        nrows = min(cap.nwakes, size(strength, 2))
        idx = Dict{Tuple{Int,Int},Int}()
        node_id(r, j) = get!(idx, (r, j)) do
            push!(verts, nodes[:, r, j]); length(verts)
        end
        for r in 1:nrows, j in 1:nspan-1
            mu = strength[1, r, j]
            abs(mu) < 1e-300 && continue
            n11, n21 = node_id(r, j), node_id(r + 1, j)
            n22, n12 = node_id(r + 1, j + 1), node_id(r, j + 1)
            push!(cells, [n11, n21, n22])
            push!(cells, [n11, n22, n12])
            push!(mus, mu); push!(mus, mu)
        end
    end
    isempty(cells) && return nothing
    obody = pnl.NonLiftingBody{pnl.ConstantDoublet}(reduce(hcat, verts),
        reduce(hcat, cells); core_size=CORE_ORACLE)
    pnl.calc_normals!(obody)
    pnl.calc_controlpoints!(obody)
    obody.strength[:, 1] .= mus
    return obody
end

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

function min_clearance(obody, targets, mask)
    dmin = fill(Inf, size(targets, 2))
    Threads.@threads for i in 1:size(targets, 2)
        mask[i] || continue
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

# ---------------- Green-route diagnostics (tier0b pattern, post-run) ---------
function green_diags(body, a, gr_steps)
    N = length(a)
    rows = []
    Ssig = zeros(N)
    Bq = zeros(N)
    for s in gr_steps
        fill!(Ssig, 0.0)
        pnl._source_potential!(Ssig, body, s.sigma, DIRECT)
        fill!(Bq, 0.0)
        pnl._with_green_scratch(body) do
            pnl._green_B_product!(Bq, body, s.q, DIRECT)
        end
        res = norm(s.q .- Bq .+ a .* s.lambda .- Ssig) / max(norm(Ssig), eps())
        gd = abs(dot(a, s.q)) / max(norm(a) * norm(s.q), eps())
        fsc = sum(a .* abs.(s.sigma))
        fl = abs(sum(a .* s.sigma)) / max(fsc, eps())
        push!(rows, (i=s.i, lambda=s.lambda, residual=res, gauge=gd,
            flux=fl, flux_scale=fsc, last_recompute=s.last_recompute))
    end
    return rows
end

# ---------------- wake restore (for post-run xchecks/oracle) -----------------
function restore_wake!(wake, cap)
    for (n, cn) in zip(wake.nodes, cap.nodes)
        n .= cn
    end
    for (s, cs) in zip(wake.strength, cap.wstrength)
        s .= cs
    end
    wake.nwakes[] = cap.nwakes
    return wake
end

# ---------------- one run ----------------------------------------------------
function run_case(route, lev; phase::Symbol, arm::Symbol)
    runid = "$(phase)$(arm == :W1 ? "W1" : "W2")-$(route.id)-L$lev"
    body = route.neumann ? make_uncapped_neumann(lev) : make_capped_wing(lev)
    info = shed_info(body)
    frames = pitching_wing_frame(body, PIVOT, deg2rad(AOA))
    set_wake_Das!(body, VINF; magnitude=DAS)   # after frames: production convention (R11)
    local wake, man, trange, capat, hash0
    if phase == :A
        wake = pnl.PanelWake(body; nwakerows=NFREE, include_final_filament=false)
        pnl.update_TE!(wake, body)
        prescribe_wake!(wake, zeros(length(info.y)), info)
        hash0 = wakehash(wake)
        man = phaseA_maneuver(body, wake, info)
        trange = range(0.0, step=DT, length=NITER_A)
        capat = NITER_A - 1
    else
        if arm == :W1
            wake = pnl.PanelWake(body; nwakerows=NSAMP_B + 2,
                include_final_filament=false)
            pnl.update_TE!(wake, body)
        else
            wake = pnl.PanelParticleWake(body; nwakerows=W2_NWAKEROWS,
                max_particles=W2_MAXPART,
                method_trailing=pnl.OverlapPPS(2.4, 2),
                method_unsteady=pnl.OverlapPPS(2.4, 2))
        end
        hash0 = ""
        man = noman!
        trange = range(0.0, step=DT, length=NSAMP_B)
        capat = NSAMP_B - 1
    end
    store = RunStore()
    cb = make_callback(body, wake, info, store; gr=route.gr,
        capture_at=capat, do_hash=(phase == :A))
    t_el = @elapsed pnl.simulate!((body,), (wake,), frames, man, Uinf_f, trange;
        body_solvers=(pnl.Backslash(body),), backend=DIRECT, path=nothing,
        formulation=route.form(), step_telemetry_callback=cb, verbose=false)

    cap = store.cap
    cap === nothing && error("$runid: capture step never fired")
    G = gamma_of(cap.strength, info)
    all(isfinite, G) || error("$runid: non-finite Gamma")
    all(s.finite for s in store.steps) || error("$runid: non-finite strengths mid-run")

    # Phase A fixed-point convergence
    deltaA = NaN
    if phase == :A
        g1, g0 = store.gammas[end], store.gammas[end-1]
        deltaA = maximum(abs.(g1 .- g0)) / max(maximum(abs.(g1)), eps())
    end
    # Phase B steadiness (recorded, not gated)
    drift = NaN
    if phase == :B && length(store.steps) >= 6
        gts = [s.gtot for s in store.steps]
        drift = abs(gts[end] - gts[end-5]) / max(abs(gts[end]), eps())
    end

    grrows = route.gr ? green_diags(body, pnl._panel_areas(body), store.gr) : []
    kx = kutta_xcheck(body, info, cap.strength)

    # ---- CSVs ----
    open(joinpath(DATADIR, "gamma_$(runid).csv"), "w") do io
        println(io, "y,dy,Gamma")
        for k in eachindex(G)
            @printf(io, "%.9e,%.9e,%.9e\n", info.y[k], info.dy[k], G[k])
        end
    end
    open(joinpath(DATADIR, "steps_$(runid).csv"), "w") do io
        println(io, "i,delta_Gamma,Gamma_tot,maxabs_strength,np,wakehash,state_scalars")
        for s in store.steps
            @printf(io, "%d,%.9e,%.9e,%.9e,%d,%s,\"%s\"\n",
                s.i, s.delta, s.gtot, s.maxs, s.np, s.hash, s.scal)
        end
    end
    if route.gr
        open(joinpath(DATADIR, "grdiag_$(runid).csv"), "w") do io
            println(io, "i,lambda,residual,gauge_defect,flux,flux_scale,last_recompute")
            for r in grrows
                @printf(io, "%d,%.9e,%.9e,%.9e,%.9e,%.9e,%d\n",
                    r.i, r.lambda, r.residual, r.gauge, r.flux,
                    r.flux_scale, r.last_recompute)
            end
        end
    end

    @printf("  %-14s N=%-6d Gtot=%+.6e CL=%+.5f dA=%.1e drift=%.1e np=%d kx=%.1e t=%.0fs\n",
        runid, body.ncells, gtot_of(G, info), cl_of(G, info), deltaA, drift,
        cap.np, kx, t_el)
    if phase == :A
        ds = [s.delta for s in store.steps[2:end]]
        sel = unique(clamp.(round.(Int, range(1, length(ds), length=min(8, length(ds)))),
            1, length(ds)))
        println("    PhaseA delta history: ",
            join([@sprintf("%d:%.1e", k, ds[k]) for k in sel], " "))
    end
    flush(stdout)

    return (; runid, body, wake, info, store, cap, G,
        Gtot=gtot_of(G, info), CL=cl_of(G, info),
        deltaA, drift, grrows, kutta_x=kx, hash0,
        insolve_hashes=[s.hash for s in store.steps],
        N=body.ncells, t_el)
end

# ---------------- oracle check for an R-GR W1 run ----------------------------
function oracle_check(rec, tag)
    body, cap, info = rec.body, rec.cap, rec.info
    a = pnl._panel_areas(body)
    temask = te_mask(body, info)
    obody = make_oracle(cap)
    obody === nothing && error("$(rec.runid): empty oracle (all-zero wake)")
    qref, uw = oracle_eval(obody, body.controlpoints)
    qhat = rec.store.gr[end].q
    E_q, E_inf_all, E_inf_ex = trace_metrics(qhat, qref, a, temask)
    # cross-checks: production wake-influence route vs oracle; sigma consistency
    restore_wake!(rec.wake, cap)
    N = body.ncells
    qf = zeros(N)
    pnl._wake_potential!(qf, body, (rec.wake,), DIRECT)
    uf = zeros(3, N)
    pnl._wake_panel_velocity!(uf, body, (rec.wake,), DIRECT)
    xp = norm(qf .- qref) / max(norm(qref), eps())
    xv = norm(uf .- uw) / max(norm(uw), eps())
    sig_prod = [-dot(view(uf, :, i), view(body.normals, :, i)) for i in 1:N]
    sx = norm(sig_prod .- rec.store.gr[end].sigma) /
         max(norm(rec.store.gr[end].sigma), eps())
    clr_all = min_clearance(obody, body.controlpoints, trues(N))
    clr_nte = min_clearance(obody, body.controlpoints, .!temask)
    open(joinpath(DATADIR, "trace_$(tag).csv"), "w") do io
        println(io, "x,y,z,area,te_adjacent,sigma,q_ref,q_green")
        sig = rec.store.gr[end].sigma
        for i in 1:N
            @printf(io, "%.9e,%.9e,%.9e,%.9e,%d,%.9e,%.9e,%.9e\n",
                body.controlpoints[1, i], body.controlpoints[2, i],
                body.controlpoints[3, i], a[i], temask[i], sig[i],
                qref[i], qhat[i])
        end
    end
    @printf("    oracle %-12s E_q=%.3e Einf(all)=%.3e Einf(ex)=%.3e xchk=(%.1e,%.1e) sigx=%.1e clr=(%.4f,%.4f)\n",
        tag, E_q, E_inf_all, E_inf_ex, xp, xv, sx, clr_all, clr_nte)
    flush(stdout)
    return (; N=rec.N, E_q, E_inf_all, E_inf_ex, xchk_pot=xp, xchk_vel=xv,
        sigma_x=sx, clr_all, clr_nte)
end

# ---------------- Gamma-gap metrics ------------------------------------------
function interp_lin(y_src, g_src, y_tgt)
    p = sortperm(y_src)
    ys, gs = y_src[p], g_src[p]
    map(y_tgt) do y
        y <= ys[1] && return gs[1]
        y >= ys[end] && return gs[end]
        k = searchsortedlast(ys, y)
        k == length(ys) && return gs[end]
        gs[k] + (gs[k+1] - gs[k]) * (y - ys[k]) / (ys[k+1] - ys[k])
    end
end

"dy-weighted rms and max gap of route A minus route B, on B's stations, /B-scale."
function gamma_gap(recA, recB)
    yB, dyB, GB = recB.info.y, recB.info.dy, recB.G
    GA = interp_lin(recA.info.y, recA.G, yB)
    scale = sqrt(sum(dyB .* GB .^ 2) / sum(dyB))
    grms = sqrt(sum(dyB .* (GA .- GB) .^ 2) / sum(dyB)) / max(scale, eps())
    gmax = maximum(abs.(GA .- GB)) / max(maximum(abs.(GB)), eps())
    return (; grms, gmax, dGtot=recA.Gtot - recB.Gtot, dCL=recA.CL - recB.CL)
end

# ---------------- P4 ---------------------------------------------------------
function p4_metric(recW1, recW2, tag)
    a = pnl._panel_areas(recW1.body)
    q1 = align(recW1.store.gr[end].q, a)
    q2 = align(recW2.store.gr[end].q, a)
    ratio = rmsA(q2 .- q1, a) / max(rmsA(q1, a), eps())
    open(joinpath(DATADIR, "p4_$(tag).csv"), "w") do io
        println(io, "area,q_W1_aligned,q_W2_aligned")
        for i in eachindex(a)
            @printf(io, "%.9e,%.9e,%.9e\n", a[i], q1[i], q2[i])
        end
    end
    return ratio
end

# ---------------- main -------------------------------------------------------
gate_lines = String[]
gate(id, ok, val, lim) = push!(gate_lines,
    @sprintf("%-6s %-4s  value=%s  gate=%s", id, ok ? "PASS" : "FAIL",
        string(val), string(lim)))
monotone_dec(v) = all(diff(v) .< 0)

g1_ok = Ref(true)
g2_ok = Ref(true)
lvl = Dict{Int,Dict{String,Any}}()     # lvl[lev][key] => record
oracleA = []; oracleB = []             # per-level oracle metrics
p4s = Float64[]
summary_rows = String[]

println("== 052e2a realsim addendum $(SMOKE ? "(SMOKE, unregistered)" : "(REGISTERED)") ==")
println("levels=$(LEVELS) NFREE=$NFREE DT=$DT NITER_A=$NITER_A NSAMP_B=$NSAMP_B")
println("datadir=$DATADIR threads=$(Threads.nthreads())")
flush(stdout)

try
    for lev in 1:length(LEVELS)
        println("---- level L$lev $(LEVELS[lev]) ----"); flush(stdout)
        lvl[lev] = Dict{String,Any}()
        for route in ROUTES
            for (phase, arm) in ((:A, :W1), (:B, :W1), (:B, :W2))
                rec = run_case(route, lev; phase, arm)
                key = "$(phase)$(arm == :W1 ? "W1" : "W2")-$(route.id)"
                lvl[lev][key] = rec
                if phase == :A
                    (isfinite(rec.deltaA) && rec.deltaA <= TOL_A) ||
                        (g1_ok[] = false;
                         push!(gate_lines, "G1 FAIL $(rec.runid): Phase A fixed point delta=$(rec.deltaA) > $TOL_A"))
                end
            end
            GC.gc()
        end
        # G2: in-solve wake-node hashes across routes and steps (Phase A)
        hs = reduce(vcat, [filter(!isempty, lvl[lev]["AW1-$(r.id)"].insolve_hashes)
                           for r in ROUTES])
        h0s = join(unique([lvl[lev]["AW1-$(r.id)"].hash0 for r in ROUTES]), ",")
        if length(unique(hs)) != 1
            g2_ok[] = false
            push!(gate_lines, "G2 FAIL L$lev: in-solve hashes not identical: $(unique(hs))")
        end
        push!(summary_rows, "L$lev hashes: insolve=$(unique(hs)) prescribed=$h0s")
        # oracle checks (P3/G4): Phase B W1 gated, Phase A recorded
        push!(oracleA, oracle_check(lvl[lev]["AW1-GR"], "A_L$lev"))
        push!(oracleB, oracle_check(lvl[lev]["BW1-GR"], "B_L$lev"))
        # P4
        push!(p4s, p4_metric(lvl[lev]["BW1-GR"], lvl[lev]["BW2-GR"], "L$lev"))
        @printf("    P4 L%d: rmsA(qW2-qW1)/rmsA(qW1) = %.3e\n", lev, p4s[end])
        # drop heavy objects we no longer need (keep records for gap tables)
        GC.gc()
    end

    # ---------------- gates ----------------
    gate("G1", g1_ok[], g1_ok[] ? "all solves converged (hard-fail on; Phase A fixed points converged)" :
        "failures listed above", "required")
    gate("G2", g2_ok[], g2_ok[] ? "wake-node hashes identical across routes/steps, all levels" :
        "mismatch listed above", "required")
    g3v = [(r.runid, r.grrows[end].residual, r.grrows[end].gauge)
           for lev in 1:length(LEVELS)
           for r in (lvl[lev]["AW1-GR"], lvl[lev]["BW1-GR"], lvl[lev]["BW2-GR"])]
    g3 = all(x -> x[2] <= 1e-10 && x[3] <= 1e-11, g3v)
    gate("G3", g3, @sprintf("max residual=%.2e max gauge=%.2e (measurement solves)",
        maximum(x -> x[2], g3v), maximum(x -> x[3], g3v)), "res<=1e-10, gd<=1e-11")
    EB = [o.E_q for o in oracleB]
    g4 = EB[end] <= 1e-2 && (length(EB) < 2 || monotone_dec(EB))
    gate("G4", g4, @sprintf("PhaseB E_q=[%s]", join([@sprintf("%.3e", e) for e in EB], ",")),
        "finest<=1e-2, monotone (Phase A recorded: [" *
        join([@sprintf("%.3e", o.E_q) for o in oracleA], ",") * "])")

    # ---------------- evidence tables (G5: recorded, Ryan rules) ----------------
    push!(summary_rows, "", "P1/P2 Gamma-gap vs R-NEU (dy-weighted rms / max, dGtot, dCL):")
    for lev in 1:length(LEVELS), (phase, arm) in (("A", "W1"), ("B", "W1"), ("B", "W2"))
        rn = lvl[lev]["$(phase)$(arm)-NEU"]
        for rid in ("GR", "VTS")
            g = gamma_gap(lvl[lev]["$(phase)$(arm)-$(rid)"], rn)
            push!(summary_rows, @sprintf(
                "  %s%s L%d  %-3s-NEU: grms=%.4e gmax=%.4e dGtot=%+.4e dCL=%+.5f",
                phase, arm, lev, rid, g.grms, g.gmax, g.dGtot, g.dCL))
        end
    end
    push!(summary_rows, "", "Scalars per run (Gtot, CL, deltaA/drift, np, kutta_x):")
    for lev in 1:length(LEVELS), key in sort(collect(keys(lvl[lev])))
        r = lvl[lev][key]
        push!(summary_rows, @sprintf(
            "  %-16s N=%-6d Gtot=%+.6e CL=%+.5f dA=%.2e drift=%.2e np=%d kx=%.2e",
            r.runid, r.N, r.Gtot, r.CL, r.deltaA, r.drift, r.cap.np, r.kutta_x))
    end
    push!(summary_rows, "", "P3 oracle (R-GR, W1):")
    for (tag, os) in (("PhaseA", oracleA), ("PhaseB", oracleB))
        for (lev, o) in enumerate(os)
            push!(summary_rows, @sprintf(
                "  %s L%d N=%-6d E_q=%.4e Einf_all=%.4e Einf_ex=%.4e xchk=(%.2e,%.2e) sigx=%.2e clr=(%.4f,%.4f)",
                tag, lev, o.N, o.E_q, o.E_inf_all, o.E_inf_ex,
                o.xchk_pot, o.xchk_vel, o.sigma_x, o.clr_all, o.clr_nte))
        end
    end
    push!(summary_rows, "", "P4 (R-GR, W2 vs W1, matched Phase B final step):")
    for (lev, p) in enumerate(p4s)
        push!(summary_rows, @sprintf("  L%d ratio=%.4e", lev, p))
    end
    push!(summary_rows, "", "Phase A W2 frozen-particle variant: NOT RUN (R9; allowed by lock L2)")

finally
    # ---------------- provenance + summary ----------------
    function gitinfo(path)
        sha = strip(read(`git -C $path rev-parse HEAD`, String))
        dirty = !isempty(strip(read(`git -C $path status --porcelain`, String)))
        dh = dirty ? bytes2hex(sha256(read(`git -C $path diff`)))[1:12] : ""
        "$sha $(dirty ? "DIRTY tracked-diff=$dh" : "clean")"
    end
    script_sha = bytes2hex(sha256(read(@__FILE__)))[1:12]
    open(joinpath(DATADIR, "gates.txt"), "w") do io
        println(io, "052e2a realsim addendum $(SMOKE ? "SMOKE (UNREGISTERED — values meaningless)" : "REGISTERED")")
        println(io, "date=$(chomp(read(`date +%Y-%m-%d`, String))) julia=$(VERSION) threads=$(Threads.nthreads()) script_sha256=$script_sha")
        println(io, "FLOWPanel: $(gitinfo(joinpath(HERE, "..", "..", "..", "FLOWPanel.jl")))")
        println(io, "FastMultipole: $(gitinfo(joinpath(HERE, "..", "..")))")
        println(io, "fixture: AOA=$(AOA)deg |U|=$UMAG levels=$(LEVELS) NFREE=$NFREE ROWLEN=$ROWLEN DT=$DT NITER_A=$NITER_A TOL_A=$TOL_A NSAMP_B=$NSAMP_B W2_rows=$W2_NWAKEROWS W2_maxpart=$W2_MAXPART")
        println(io, "realization: R1 PhaseA fixed-point via maneuver!; R2 PhaseB prod defaults; R3 CL=Kutta-Joukowski; R4 G3 at measurement solve; R5 G4 on PhaseB W1 sequence; R6 G2 on in-solve hashes; R7 TE mask from shedding nodes; R8 JumpKutta c==0 by construction + _get_wakestrength_mu xcheck; R9 PhaseA-W2 NOT RUN; R10 stage-1 metrics/oracle")
        println(io, "call chain: simulate!/Backslash/DirectBackend | GR: state.green.q + _green_lambda | telemetry via step_telemetry_callback | G3 by harness: _source_potential! + _green_B_product! | oracle: pnl.induced sums")
        for l in gate_lines
            println(io, l)
        end
        for l in summary_rows
            println(io, l)
        end
    end
    println("\n===== 052e2a ADDENDUM $(SMOKE ? "SMOKE" : "GATES") =====")
    foreach(println, gate_lines)
    foreach(println, summary_rows)
    println("outputs: $DATADIR")
    println("==========================================")
end
