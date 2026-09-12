# L4 pre-lock MECHANICAL verification probe for 052e.2a realsim addendum.
# API acceptance only — produces NO registered values. Checks A-D per handoff.
import FLOWPanel as pnl
using StaticArrays, LinearAlgebra, SHA, Printf, Logging

const FPDIR = "/Users/ryan/Dropbox/research/projects/FLOWPanel.jl"
include(joinpath(FPDIR, "examples", "pitching_wing.jl"))

const C = 0.76
const B = 2.7
const THICK = 0.12
const NA, NS, NE = 41, 5, 3   # tiny stage-1 smoke-level mesh
const DT = 0.1
const VINF = SVector{3}(1.0, 0.0, 0.0)
Uinf_f(t) = VINF
noman!(frames, systems, wakes, t) = nothing
const BACKEND = pnl.DirectBackend()

results = String[]
function check(f::Function, name::String)
    try
        msg = f()
        push!(results, "PASS  $name" * (msg === nothing ? "" : " -- $msg"))
    catch e
        io = IOBuffer()
        showerror(io, e)
        s = replace(String(take!(io)), r"\s+" => " ")
        push!(results, "FAIL  $name -- $(first(s, 600))")
    end
    flush(stdout)
end

function make_capped_wing()
    wing = build_pitching_wing_body(C, B; n_span=NS, n_airfoil=NA, n_endcap=NE,
        thickness=THICK, semiinfinite_wake=false)
    pnl.calc_normals!(wing)
    pnl.calc_controlpoints!(wing)
    set_wake_Das!(wing, VINF)
    return wing
end

function make_uncapped_neumann()
    nodes, cells = pitching_wing_mesh(C, B; n_span=NS, n_airfoil=NA,
        n_endcap=NE, caps=false, thickness=THICK)
    bodytype = pnl.RigidWakeBody{pnl.ConstantDoublet, 1, Float64, false}
    opts = (; core_size=1e-6 * C, kernelcutoff=1e-12 * C,
        semiinfinite_wake=false, watertight=false)
    base = bodytype(nodes, cells, zeros(Int, 6, 0); opts...)
    shedding = calc_pitching_wing_shedding(base.nodes, base.cells, C)
    body = bodytype(copy(base.nodes), copy(base.cells), [shedding]; opts...)
    pnl.calc_normals!(body)
    pnl.calc_controlpoints!(body)
    set_wake_Das!(body, VINF)
    return body
end

"Prescribe a flat wake: overwrite node rows along VINF, elliptic strengths, mark rows active."
function prescribe_flat!(wake; nactive=4)
    step = VINF ./ norm(VINF) .* (0.5 * C)
    for nodes in wake.nodes
        first_row = copy(view(nodes, :, 1, :))
        for row in axes(nodes, 2)
            view(nodes, :, row, :) .= first_row .+ (row - 1) .* step
        end
    end
    for (nodes, strength) in zip(wake.nodes, wake.strength)
        for j in axes(strength, 3)
            ymid = (nodes[2, 1, j] + nodes[2, 1, j+1]) / 2
            mu = sqrt(max(0.0, 1.0 - (2ymid / B)^2))
            strength[1, 1, j] = 0.0
            for row in 2:nactive
                strength[1, row, j] = mu
            end
        end
    end
    wake.nwakes[] = nactive
    return wake
end

wakehash(wake) = bytes2hex(sha256(reinterpret(UInt8,
    vec(reduce(hcat, [reshape(n, 3, :) for n in wake.nodes])))))[1:16]

t2 = range(0.0, step=DT, length=2)   # two solve steps; step i_step=0 sees the prescribed wake
t3 = range(0.0, step=DT, length=6)

# ---------------- Check A: GreenReconstruction(:area_mean) acceptance --------
println("== A =="); flush(stdout)
stateA = Ref{Any}(nothing)
check("A1 simulate! GreenReconstruction(:area_mean) capped Dirichlet PanelWake Backslash default-Kutta") do
    wing = make_capped_wing()
    frames = pitching_wing_frame(wing, SVector{3}(0.25C, 0.0, 0.0), deg2rad(5.0))
    wake = pnl.PanelWake(wing; nwakerows=6, include_final_filament=false)
    pnl.update_TE!(wake, wing)
    solver = pnl.Backslash(wing)
    cb(nt) = (stateA[] = nt.formulation_state)
    pnl.simulate!((wing,), (wake,), frames, noman!, Uinf_f, t2;
        body_solvers=(solver,), backend=BACKEND, path=nothing,
        formulation=pnl.GreenReconstruction(gauge=:area_mean),
        step_telemetry_callback=cb, verbose=false)
    all(isfinite, wing.strength) || error("non-finite strengths after solve")
    "green state = $(nameof(typeof(stateA[].green)))"
end
check("A2 formulation state is GreenReconstructionState w/ Householder green state") do
    st = stateA[]
    st isa pnl.GreenReconstructionState || error("state is $(typeof(st))")
    occursin("Householder", string(typeof(st.green))) ||
        error("green state is $(typeof(st.green)), not Householder")
    nothing
end
check("A3 _green_lambda accessible and finite") do
    lam = pnl._green_lambda(stateA[].green)
    isfinite(lam) || error("lambda = $lam")
    "lambda finite"
end

# ---------------- Check B: Neumann uncapped referee route --------------------
println("== B =="); flush(stdout)
neu_logs = String[]
check("B1 uncapped caps=false doublet-only DBC=false RigidWakeBody constructs (watertight=false)") do
    body = make_uncapped_neumann()
    body.watertight && error("body reports watertight=true for caps=false mesh")
    "ncells=$(body.ncells)"
end
check("B2 steady! Neumann solve, Backslash, default Kutta pair, no rank-deficiency warning") do
    body = make_uncapped_neumann()
    logger = Base.CoreLogging.SimpleLogger(IOBuffer(), Logging.Warn)
    testio = IOBuffer()
    with_logger(ConsoleLogger(testio, Logging.Warn)) do
        pnl.steady!((body,), pitching_wing_frame(body,
                SVector{3}(0.25C, 0.0, 0.0), deg2rad(5.0)), VINF;
            body_solvers=(pnl.Backslash(body),), backend=BACKEND, path=nothing)
    end
    warns = String(take!(testio))
    occursin("rank-deficient", warns) &&
        error("rank-deficiency warning fired: $(first(warns, 200))")
    all(isfinite, body.strength) || error("non-finite strengths")
    maximum(abs, body.strength) > 0 || error("all-zero strengths")
    nothing
end

# ---------------- Check C: particle wake feeds GR sigma path -----------------
println("== C =="); flush(stdout)
check("C1 simulate! PanelParticleWake + GreenReconstruction; u_prewake/sigma populated; recompute ran") do
    wing = make_capped_wing()
    frames = pitching_wing_frame(wing, SVector{3}(0.25C, 0.0, 0.0), deg2rad(5.0))
    wake = pnl.PanelParticleWake(wing; nwakerows=2, max_particles=5000,
        method_trailing=pnl.OverlapPPS(1.3, 2),
        method_unsteady=pnl.OverlapPPS(1.3, 2))
    solver = pnl.Backslash(wing)
    snaps = []
    cb(nt) = push!(snaps, (i=nt.i_step,
        u_prewake_max=maximum(abs, nt.formulation_state.u_prewake),
        sigma_max=maximum(abs, nt.formulation_state.sigma),
        finite=all(isfinite, nt.formulation_state.u_prewake) &&
               all(isfinite, nt.formulation_state.sigma),
        last_recompute=nt.formulation_state.last_recompute[]))
    pnl.simulate!((wing,), (wake,), frames, noman!, Uinf_f, t3;
        body_solvers=(solver,), backend=BACKEND, path=nothing,
        formulation=pnl.GreenReconstruction(gauge=:area_mean),
        step_telemetry_callback=cb, verbose=false)
    isempty(snaps) && error("telemetry callback never fired")
    last_s = snaps[end]
    last_s.finite || error("non-finite u_prewake/sigma")
    last_s.u_prewake_max > 0 || error("u_prewake all zero (snapshot not populated)")
    last_s.last_recompute == last_s.i || error(
        "reconstruction did not run at step $(last_s.i) (last_recompute=$(last_s.last_recompute))")
    np = pnl.FLOWVPM.get_np(wake.pfield)
    np > 0 || error("no particles shed after $(length(snaps)) steps")
    "steps=$(length(snaps)), np=$np, sigma_max nonzero at last step: $(last_s.sigma_max > 0)"
end

# ---------------- Check D: prescribed wake via public API, all 3 routes ------
println("== D =="); flush(stdout)
hashes = Dict{String,String}()
check("D1 R-VTS prescribed-wake first-step solve (simulate!, default formulation)") do
    wing = make_capped_wing()
    frames = pitching_wing_frame(wing, SVector{3}(0.25C, 0.0, 0.0), deg2rad(5.0))
    wake = pnl.PanelWake(wing; nwakerows=6, include_final_filament=false)
    pnl.update_TE!(wake, wing)
    prescribe_flat!(wake)
    hashes["VTS"] = wakehash(wake)
    pnl.simulate!((wing,), (wake,), frames, noman!, Uinf_f, t2;
        body_solvers=(pnl.Backslash(wing),), backend=BACKEND, path=nothing,
        formulation=pnl.VelocityThroughSources(require_outer_convergence=true),
        verbose=false)
    all(isfinite, wing.strength) || error("non-finite strengths")
    nothing
end
check("D2 R-GR prescribed-wake first-step solve") do
    wing = make_capped_wing()
    frames = pitching_wing_frame(wing, SVector{3}(0.25C, 0.0, 0.0), deg2rad(5.0))
    wake = pnl.PanelWake(wing; nwakerows=6, include_final_filament=false)
    pnl.update_TE!(wake, wing)
    prescribe_flat!(wake)
    hashes["GR"] = wakehash(wake)
    pnl.simulate!((wing,), (wake,), frames, noman!, Uinf_f, t2;
        body_solvers=(pnl.Backslash(wing),), backend=BACKEND, path=nothing,
        formulation=pnl.GreenReconstruction(gauge=:area_mean), verbose=false)
    all(isfinite, wing.strength) || error("non-finite strengths")
    nothing
end
check("D3 R-NEU prescribed-wake first-step solve (uncapped, default formulation)") do
    body = make_uncapped_neumann()
    frames = pitching_wing_frame(body, SVector{3}(0.25C, 0.0, 0.0), deg2rad(5.0))
    wake = pnl.PanelWake(body; nwakerows=6, include_final_filament=false)
    pnl.update_TE!(wake, body)
    prescribe_flat!(wake)
    hashes["NEU"] = wakehash(wake)
    pnl.simulate!((body,), (wake,), frames, noman!, Uinf_f, t2;
        body_solvers=(pnl.Backslash(body),), backend=BACKEND, path=nothing,
        verbose=false)
    all(isfinite, body.strength) || error("non-finite strengths")
    nothing
end
check("D4 wake node geometry hashable; capped vs uncapped wake-node hash comparison") do
    length(hashes) == 3 || error("only $(length(hashes)) wakes hashed")
    same = hashes["VTS"] == hashes["GR"]
    neu_same = hashes["NEU"] == hashes["VTS"]
    "VTS==GR: $same; NEU==VTS(capped): $neu_same; hashes=$(hashes)"
end

println("\n===== L4 PROBE RESULTS =====")
foreach(println, results)
println("============================")
