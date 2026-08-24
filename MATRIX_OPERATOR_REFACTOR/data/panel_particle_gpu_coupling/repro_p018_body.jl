# Local repro attempt for the pass-3 seam-vs-exact deviation (job 13309844:
# seam 1.235e-2, cpu-fmm 1.1e-10). Builds the p018 rotor body from the local
# mesh (driver construction, noshedding variant -> NO attached wake columns),
# deterministic strengths, then compares exact DirectBackend vs the :host rect
# seam. If the deviation reproduces here, it is functor math (not device) and
# not the attached wake; root-cause locally.
import FLOWPanel as pnl
import FastMultipole
using Printf

const FPROOT = "/Users/ryan/Dropbox/research/projects/FLOWPanel.jl"
include(joinpath(FPROOT, "benchmark", "fm051_pass3_attribution.jl"))

R = 0.119
msh_file = joinpath(FPROOT, "examples", "data", "dji9443_20260725_45_185_capped_captess4.msh")
core_size_panel = R * 1e-10
core_size_targets = 1e-3
kernelcutoff = R * 1e-13

msh = pnl.read_gmsh(msh_file)
nodes, cells = pnl.meshes2nodes_cells(msh)
nodes .*= R / maximum(nodes[2, :])          # radial_dimension = 2 for dji9443
rotor = pnl.RigidWakeBody{Union{pnl.ConstantSource,pnl.VortexRing}}(nodes, cells,
    pnl.noshedding;
    core_size=core_size_panel, core_size_panel, core_size_targets, kernelcutoff,
    semiinfinite_wake=false, watertight=true, DBC=true)
rotor.needs_velocity_gradient[] = false
pnl.calc_normals!(rotor)
pnl.calc_controlpoints!(rotor)
println("rotor: $(rotor.ncells) panels")
for j in axes(rotor.strength, 2), i in axes(rotor.strength, 1)
    rotor.strength[i, j] = sin(0.7 * i + 1.3 * j) + 0.1
end

backend = pnl.FastMultipoleBackend(8, 0.4, 20)   # p018 body backend

function run_arm!(body, bk; seam::Symbol=:off)
    body.velocity .= 0.0
    pnl._set_core_sizes!((body,), :core_size_targets)
    pnl.set_gpu_influence!(seam)
    h0 = pnl.GPU_INFLUENCE_HITS[]
    t = @elapsed pnl._sa_body_influence!((body,), (body,), bk;
        needs_induced_vorticity=false, body_on_wake=true,
        body_hessian_to_particles=false, body_gradient_core_size=NaN)
    hits = pnl.GPU_INFLUENCE_HITS[] - h0
    pnl.set_gpu_influence!(:off)
    return copy(body.velocity), t, hits
end

dexact, t_e, _ = run_arm!(rotor, pnl.DirectBackend())
@printf("exact direct: %.1f s\n", t_e)
dhost, t_h, hits_h = run_arm!(rotor, backend; seam=:host)
@printf("host seam:    %.1f s (hits %d)\n", t_h, hits_h)
hits_h > 0 || error("host seam did not accept -- vacuous")
dcpu, t_c, _ = run_arm!(rotor, backend)
@printf("cpu fmm:      %.1f s\n", t_c)

r = attr_report(dexact, dcpu, dhost; gate_seam=1e-10, label="p018-body local (noshedding)")
println("worst host-seam-vs-exact offender control points:")
for (i, rel, dn, rn) in r.seam.top
    ii = Int(i)
    cp = rotor.controlpoints[:, ii]
    @printf("  target %-8d rel %.3e |diff| %.3e  cp % .5f % .5f % .5f\n",
        ii, rel, dn, cp[1], cp[2], cp[3])
end
