# L4 follow-up C2: does production VelocityThroughSources consume the
# PanelParticleWake unchanged? Mechanical probe only — no registered values.
import FLOWPanel as pnl
using StaticArrays, LinearAlgebra, Printf

const FPDIR = "/Users/ryan/Dropbox/research/projects/FLOWPanel.jl"
include(joinpath(FPDIR, "examples", "pitching_wing.jl"))

const C = 0.76
const B = 2.7
const VINF = SVector{3}(1.0, 0.0, 0.0)
Uinf_f(t) = VINF
noman!(frames, systems, wakes, t) = nothing

try
    wing = build_pitching_wing_body(C, B; n_span=5, n_airfoil=41, n_endcap=3,
        thickness=0.12, semiinfinite_wake=false)
    pnl.calc_normals!(wing)
    pnl.calc_controlpoints!(wing)
    set_wake_Das!(wing, VINF)
    frames = pitching_wing_frame(wing, SVector{3}(0.25C, 0.0, 0.0), deg2rad(5.0))
    wake = pnl.PanelParticleWake(wing; nwakerows=2, max_particles=5000,
        method_trailing=pnl.OverlapPPS(1.3, 2),
        method_unsteady=pnl.OverlapPPS(1.3, 2))
    solver = pnl.Backslash(wing)
    nsolves = Ref(0)
    finite_each_step = Ref(true)
    cb(nt) = begin
        nsolves[] += 1
        finite_each_step[] &= all(isfinite, wing.strength)
    end
    pnl.simulate!((wing,), (wake,), frames, noman!, Uinf_f,
        range(0.0, step=0.1, length=6);
        body_solvers=(solver,), backend=pnl.DirectBackend(), path=nothing,
        formulation=pnl.VelocityThroughSources(require_outer_convergence=true),
        step_telemetry_callback=cb, verbose=false)
    np = pnl.FLOWVPM.get_np(wake.pfield)
    ok = nsolves[] == 6 && finite_each_step[] &&
         all(isfinite, wing.strength) && maximum(abs, wing.strength) > 0 &&
         np > 0
    if ok
        println("PASS  C2 VTS + PanelParticleWake -- solves=$(nsolves[]), " *
            "strengths finite each step, np=$np after warm-up")
    else
        println("FAIL  C2 VTS + PanelParticleWake -- solves=$(nsolves[]), " *
            "finite_each_step=$(finite_each_step[]), np=$np, " *
            "maxabs_strength=$(maximum(abs, wing.strength))")
    end
catch e
    io = IOBuffer()
    showerror(io, e)
    println("FAIL  C2 VTS + PanelParticleWake -- " *
        first(replace(String(take!(io)), r"\s+" => " "), 600))
end
