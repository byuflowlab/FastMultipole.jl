# fm049_extract_snapshot.jl — task 049 rotor-field snapshots 710:719.
#
# LOCAL extraction of the p018 rotor-wake particle snapshots (steps 710:719,
# variable n) into raw binary dumps consumable by the cluster driver
# (FLOWVPM.jl/scripts/fm049_rotor_verify.jl) without ReadVTK/FLOWPanel on the
# cluster.
#
# Run from anywhere with FLOWPanel.jl's environment (it has ReadVTK and dev'd
# FLOWVPM/FLOWPanel):
#
#   julia --project=$HOME/Dropbox/research/projects/FLOWPanel.jl \
#       FastMultipole/MATRIX_OPERATOR_REFACTOR/scripts/fm049_extract_snapshot.jl
#
# Loads the VTP through the exact field mapping of FLOWPanel's
# `_load_panel_particle_wake_vtk!` (FLOWPanel.jl/src/FLOWPanel_warmstart.jl:238)
# into a bare FLOWVPM.ParticleField constructed the way PanelParticleWake's
# constructor does (FLOWPanel_wake.jl:1884 — viscous=Inviscid, fmm autotune
# off, gaussianerf default kernel). The full loader is not called directly
# because it first restores the *panel* wake (`_load_panel_wake_vtk!`), which
# needs the body/panel VTP series and a constructed lifting body; only the
# particle block (verbatim mapping below) is relevant here.
#
# Output (little-endian Float64, column-major):
#   MATRIX_OPERATOR_REFACTOR/data/rotor_field_gpu_verification/
#     p018_<step>_particles.bin: Int64 nrows(=46), Int64 np, then
#                                particles[1:46, 1:np] as Float64
#     manifest.csv             : file, step, np, sha256
#
# Loaded VTP rows (identical to the FLOWPanel loader):
#   X=1:3 <- points, GAMMA=4:6 <- "gamma", SIGMA=7 <- "sigma",
#   VOL=8 <- "vol", CIRCULATION=9 <- "circulation", U=10:12 <- "velocity",
#   VORTICITY=13:15 <- "vorticity", J=16:24 <- "velocity_gradient" (9 x np),
#   C=37:39 <- "C", SFS=40:42 <- "SFS". All other rows zero.

import ReadVTK
import FLOWVPM
using SHA: sha256

# ------------------------------------------------------------------ config
const SNAP_PATH = joinpath(homedir(), "p018_L1_ov3_paraview")
const WAKE_NAME = "p018_L1_ov3_wake1"
function parse_steps(spec)
    if occursin(':', spec)
        bounds = parse.(Int, split(spec, ':'))
        length(bounds) == 2 || error("FM049_STEPS range must be first:last")
        return collect(bounds[1]:bounds[2])
    end
    return parse.(Int, split(spec, ','))
end
const STEP_IDXS = parse_steps(get(ENV, "FM049_STEPS", "710:719"))
const OUT_DIR   = normpath(joinpath(@__DIR__, "..", "data",
                                    "rotor_field_gpu_verification"))
const MANIFEST = joinpath(OUT_DIR, "manifest.csv")

function extract_step(step_idx)
vtp_path = joinpath(SNAP_PATH, WAKE_NAME * "_particles",
                    "$(WAKE_NAME)_particles.$(step_idx).vtp")
isfile(vtp_path) || error("Particles VTP not found: $(vtp_path)")
vtk = ReadVTK.VTKFile(vtp_path)
np = vtk.n_points
@info "loaded VTP" vtp_path np step_idx

# pfield constructed as PanelParticleWake's constructor does (defaults:
# gaussianerf kernel, rVPM formulation, transposed=true)
pfield = FLOWVPM.ParticleField(np, Float64;
    viscous=FLOWVPM.Inviscid(),
    fmm=FLOWVPM.FMM(autotune_reg_error=false),
    SFS=FLOWVPM.SFS_default,
    integration=FLOWVPM.euler,
    relaxation=FLOWVPM.relaxation_correctedpedrizzetti)
pf = pfield

# particle block of _load_panel_particle_wake_vtk!, verbatim mapping
point_data = ReadVTK.get_point_data(vtk)
required_fields = ("gamma", "sigma", "vol", "circulation", "velocity",
                   "vorticity", "C", "SFS", "velocity_gradient")
missing_fields = filter(field -> !(field in keys(point_data)), collect(required_fields))
isempty(missing_fields) || throw(ArgumentError(
    "VTP is missing required field(s): $(join(missing_fields, ", "))."))

pf.particles[:, :] .= zero(eltype(pf.particles))
pf.np = 0

points = ReadVTK.get_points(vtk)  # 3 x np
pf.particles[FLOWVPM.X_INDEX, 1:np] .= points
pf.particles[FLOWVPM.GAMMA_INDEX, 1:np] .= ReadVTK.get_data(point_data["gamma"])
pf.particles[FLOWVPM.SIGMA_INDEX, 1:np] .= ReadVTK.get_data(point_data["sigma"])
pf.particles[FLOWVPM.VOL_INDEX, 1:np] .= ReadVTK.get_data(point_data["vol"])
pf.particles[FLOWVPM.CIRCULATION_INDEX, 1:np] .= ReadVTK.get_data(point_data["circulation"])
pf.particles[FLOWVPM.U_INDEX, 1:np] .= ReadVTK.get_data(point_data["velocity"])
pf.particles[FLOWVPM.VORTICITY_INDEX, 1:np] .= ReadVTK.get_data(point_data["vorticity"])
pf.particles[FLOWVPM.C_INDEX, 1:np] .= ReadVTK.get_data(point_data["C"])
pf.particles[FLOWVPM.SFS_INDEX, 1:np] .= ReadVTK.get_data(point_data["SFS"])
J_arr = ReadVTK.get_data(point_data["velocity_gradient"])
pf.particles[FLOWVPM.J_INDEX, 1:np] .= reshape(J_arr, 9, np)
pf.np = np

# ------------------------------------------------------------------ validate
nrows = size(pf.particles, 1)
nrows == 46 || error("unexpected particle matrix row count $(nrows) != 46")
P = pf.particles[:, 1:np]
all(isfinite, P) || error("non-finite values in loaded particle matrix")
sig = P[FLOWVPM.SIGMA_INDEX, :]
all(>(0), sig) || error("non-positive sigma found (min $(minimum(sig)))")
@info "spot values" np nrows sigma_min=minimum(sig) sigma_max=maximum(sig) x_extent=(extrema(P[1, :]), extrema(P[2, :]), extrema(P[3, :])) gamma_norm_max=maximum(sqrt.(sum(abs2, P[FLOWVPM.GAMMA_INDEX, :]; dims=1)))

# ------------------------------------------------------------------ write
mkpath(OUT_DIR)
out_bin = joinpath(OUT_DIR, "p018_$(step_idx)_particles.bin")
open(out_bin, "w") do io
    write(io, Int64(nrows))
    write(io, Int64(np))
    write(io, Float64.(P))       # column-major 46 x np, little-endian
end
digest = bytes2hex(open(sha256, out_bin))
sz = filesize(out_bin)
expected = 16 + 8 * nrows * np
sz == expected || error("output size $(sz) != expected $(expected)")
@info "wrote snapshot dump" out_bin size_bytes=sz sha256=digest
return (file=basename(out_bin), step=step_idx, np=np, sha256=digest)
end

mkpath(OUT_DIR)
rows = map(extract_step, STEP_IDXS)
open(MANIFEST, "w") do io
    println(io, "file,step,np,sha256")
    foreach(r -> println(io, "$(r.file),$(r.step),$(r.np),$(r.sha256)"), rows)
end
println("DONE steps=$(join(STEP_IDXS, ','))")
