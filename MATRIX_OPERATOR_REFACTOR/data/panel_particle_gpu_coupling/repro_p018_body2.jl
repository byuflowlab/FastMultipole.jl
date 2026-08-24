# Variant 2 of the local pass-3 repro: SAME as repro_p018_body.jl but with the
# driver's real trailing-edge shedding (both blades, root clip 0.1) and nonzero
# Das, so the body packs ATTACHED-WAKE columns exactly like the failing cluster
# config (37072 columns vs 36752 panels). Host seam vs exact direct.
import FLOWPanel as pnl
import FastMultipole
using Printf

const FPROOT = "/Users/ryan/Dropbox/research/projects/FLOWPanel.jl"
include(joinpath(FPROOT, "benchmark", "fm051_pass3_attribution.jl"))
include(joinpath(FPROOT, "examples", "dji9443_trailing_edge.jl"))

R = 0.119
rdim = 2
msh_file = joinpath(FPROOT, "examples", "data", "dji9443_20260725_45_185_capped_captess4.msh")
core_size_panel = R * 1e-10
core_size_targets = 1e-3
kernelcutoff = R * 1e-13

msh = pnl.read_gmsh(msh_file)
nodes, cells = pnl.meshes2nodes_cells(msh)
nodes .*= R / maximum(nodes[rdim, :])
kernel = Union{pnl.ConstantSource,pnl.VortexRing}
rotor = pnl.RigidWakeBody{kernel}(nodes, cells, pnl.noshedding;
    core_size=core_size_panel, core_size_panel, core_size_targets, kernelcutoff,
    semiinfinite_wake=false, watertight=true, DBC=true)

te1, te2 = find_dji9443_trailing_edge_indices(msh_file; watertight=true)

function make_shedding_bbox(nodes, seed_nodes, radial_dimension, R, shedding_r_over_R)
    radial_midpoint = sum(nodes[radial_dimension, seed_nodes]) / length(seed_nodes)
    radial_sign = sign(radial_midpoint)
    lower = [minimum(nodes[i, :]) for i in 1:size(nodes, 1)]
    upper = [maximum(nodes[i, :]) for i in 1:size(nodes, 1)]
    padding = max(sqrt(eps(eltype(nodes))) * R, R * 1e-6)
    lower .-= padding; upper .+= padding
    radial_cutoff = shedding_r_over_R * R
    if radial_sign > 0
        lower[radial_dimension] = radial_cutoff - padding
    else
        upper[radial_dimension] = -radial_cutoff + padding
    end
    return (pnl.SVector{3}(lower...), pnl.SVector{3}(upper...))
end

function clip_shedding_root(nodes, shedding, cells, radial_dimension, R, clip_r_over_R)
    keep = Int[]
    for j in axes(shedding, 2)
        p, nia, nib = shedding[1, j], shedding[2, j], shedding[3, j]
        na, nb = cells[nia, p], cells[nib, p]
        mid = (nodes[radial_dimension, na] + nodes[radial_dimension, nb]) / 2
        abs(mid) / R >= clip_r_over_R && push!(keep, j)
    end
    return shedding[:, keep]
end

sheds = map(((te, tag),) -> begin
    bbox = make_shedding_bbox(rotor.nodes, te[1:2], rdim, R, 0.0)
    full = pnl.calc_shedding_from_seed(rotor.nodes, rotor.cells, te[1], te[2];
        bbox=bbox, end_node=(length(te) >= 3 ? te[3] : nothing),
        normal_jump_tol=0.2, max_turn_angle=pi/3, debug=false)
    clip_shedding_root(rotor.nodes, full, rotor.cells, rdim, R, 0.1)
end, ((te1, 1), (te2, 2)))
println("shedding edges: blade1 $(size(sheds[1], 2))  blade2 $(size(sheds[2], 2))")

rotor = pnl.RigidWakeBody{kernel}(rotor.nodes, rotor.cells, [sheds[1], sheds[2]];
    core_size=core_size_panel, core_size_panel, core_size_targets, kernelcutoff,
    semiinfinite_wake=false, watertight=true, ensure_winding=true, DBC=true)
rotor.needs_velocity_gradient[] = false
pnl.calc_normals!(rotor)
pnl.calc_controlpoints!(rotor)
for k in eachindex(rotor.Das)
    rotor.Das[k] .= repeat([0.005, 0.0, 0.0], 1, size(rotor.Das[k], 2))
end
println("rotor: $(rotor.ncells) panels, $(length(rotor.Das)) shedding sets")
for j in axes(rotor.strength, 2), i in axes(rotor.strength, 1)
    rotor.strength[i, j] = sin(0.7 * i + 1.3 * j) + 0.1
end

backend = pnl.FastMultipoleBackend(8, 0.4, 20)

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

st = attr_class_stats(dexact, dhost)
@printf("host-seam vs exact: diff/scale %.3e  worst rel %.3e (target %d, |diff| %.3e, |ref| %.3e)  over-gate %d divergent %d\n",
    st.diff_to_scale, st.worst_rel, st.iw, st.dn_w, st.rn_w, st.n_over, st.n_div)
for (i, rel, dn, rn) in st.top
    ii = Int(i)
    cp = rotor.controlpoints[:, ii]
    @printf("  target %-8d rel %.3e |diff| %.3e  cp % .5f % .5f % .5f\n",
        ii, rel, dn, cp[1], cp[2], cp[3])
end
