# 052e.0 Tier 0A — constant-doublet kernel convention test
# Pre-registration: MATRIX_OPERATOR_REFACTOR/052e0-tier0a-preregistration-2026-09-07.md
# Run:  julia --project=/Users/ryan/Dropbox/research/projects/FLOWPanel.jl \
#             MATRIX_OPERATOR_REFACTOR/scripts/tier0a_052e_doublet_convention.jl
# Exercises the same kernel HybridWakePotential uses for wake-panel potential:
# _induced(..., ConstantDoublet, ...) in FLOWPanel_elements_fmm.jl.

import FLOWPanel as pnl
using StaticArrays, Printf, Statistics, LinearAlgebra, SHA

const DATADIR = joinpath(@__DIR__, "..", "data", "052e0-tier0a")
mkpath(DATADIR)

const R_DISK = 1.0
const MU = 1.0
const CORE = 1e-8
const SW_P = pnl.FastMultipole.DerivativesSwitch(true, false, false)
const SW_V = pnl.FastMultipole.DerivativesSwitch(false, true, false)

# ---------------- disk triangulation (normal +z <=> CCW seen from +z) --------

struct Tri
    verts::NTuple{3,SVector{3,Float64}}
    centroid::SVector{3,Float64}
    R::SMatrix{3,3,Float64,9}
end

function make_tri(a, b, c)
    Rm, _, _, _ = pnl.rotate_to_panel(a[1],a[2],a[3], b[1],b[2],b[3], c[1],c[2],c[3])
    Tri((a,b,c), (a+b+c)/3, SMatrix{3,3,Float64,9}(Rm))
end

flip(t::Tri) = make_tri(t.verts[1], t.verts[3], t.verts[2])

function disk_mesh(ntheta::Int, nr::Int; R=R_DISK)
    node(j, k) = SVector(R*j/nr*cos(2pi*k/ntheta), R*j/nr*sin(2pi*k/ntheta), 0.0)
    tris = Tri[]
    for k in 0:ntheta-1                      # hub fan
        push!(tris, make_tri(SVector(0.0,0.0,0.0), node(1,k), node(1,k+1)))
    end
    for j in 1:nr-1, k in 0:ntheta-1         # ring quads -> 2 triangles
        a, b, c, d = node(j,k), node(j+1,k), node(j+1,k+1), node(j,k+1)
        push!(tris, make_tri(a, b, c))
        push!(tris, make_tri(a, c, d))
    end
    tris
end

# ---------------- kernel-under-test evaluation -------------------------------

function phi_impl(x, tris; mu=MU)
    s = 0.0
    for t in tris
        p, _ = pnl._induced(x, t.verts, t.centroid, SVector{1}(mu),
                            pnl.ConstantDoublet, CORE, t.R, SW_P)
        s += p
    end
    s
end

function vel_impl(x, tris; mu=MU)
    v = SVector(0.0, 0.0, 0.0)
    for t in tris
        _, u = pnl._induced(x, t.verts, t.centroid, SVector{1}(mu),
                            pnl.ConstantDoublet, CORE, t.R, SW_V)
        v += u
    end
    v
end

# ---------------- independent references -------------------------------------

# Van Oosterom & Strackee signed solid angle of a triangle seen from x.
function omega_tri(x, t::Tri)
    r1, r2, r3 = t.verts[1]-x, t.verts[2]-x, t.verts[3]-x
    n1, n2, n3 = norm(r1), norm(r2), norm(r3)
    num = dot(r1, cross(r2, r3))
    den = n1*n2*n3 + dot(r1,r2)*n3 + dot(r2,r3)*n1 + dot(r3,r1)*n2
    -2*atan(num, den)   # sign: Omega > 0 on the +normal (CCW) side (prereg v2)
end
omega_poly(x, tris) = sum(omega_tri(x, t) for t in tris)
phi_ref_poly(x, tris; mu=MU) = -mu*omega_poly(x, tris)/(4pi)     # plan convention

omega_axis(z; R=R_DISK) = 2pi*(sign(z) - z/sqrt(z^2 + R^2))     # zero-at-infinity
phi_axis(z; mu=MU) = -mu*omega_axis(z)/(4pi)                    # plan convention

# ---------------- point sets (locked in prereg) ------------------------------

const Z_ON = [s*z for z in (0.05,0.1,0.25,0.5,1.0,2.0,5.0,10.0,20.0), s in (1,-1)][:]
onaxis_pt(z) = SVector(0.0, 0.0, z)
const OFFAXIS = [SVector(0.5*cos(a), 0.5*sin(a), z)
                 for a in (0.0, pi/4), z in (0.25, 1.0, -0.25, -1.0)][:]
const Z_FAR = (5.0, 10.0, 20.0)

# ---------------- gauss-legendre line integral -------------------------------

function gl_nodes(n)  # nodes/weights on [-1,1] via eigen of Jacobi matrix
    b = [k/sqrt(4k^2 - 1) for k in 1:n-1]
    T = SymTridiagonal(zeros(n), b)
    E = eigen(T)
    E.values, 2 .* (E.vectors[1, :] .^ 2)
end

function line_integral_u_dl(pA, pB, tris, npan, nord)
    xs, ws = gl_nodes(nord)
    acc = 0.0
    d = (pB - pA)/npan
    for i in 0:npan-1
        a = pA + i*d
        for (x, w) in zip(xs, ws)
            p = a + (x+1)/2*d
            acc += w/2 * dot(vel_impl(p, tris), d)
        end
    end
    acc
end

function circulation(tris; npan=48, nord=8)
    th0 = pi/256   # loop azimuthal plane midway between finest-mesh node lines
    er = SVector(cos(th0), sin(th0), 0.0)
    X, Z, x0 = 25.0, 25.0, 0.5
    c1 = x0*er - SVector(0,0,Z); c2 = x0*er + SVector(0,0,Z)
    c3 = X*er + SVector(0,0,Z);  c4 = X*er - SVector(0,0,Z)
    (line_integral_u_dl(c1, c2, tris, npan, nord) +   # up through disk interior
     line_integral_u_dl(c2, c3, tris, npan, nord) +
     line_integral_u_dl(c3, c4, tris, npan, nord) +
     line_integral_u_dl(c4, c1, tris, npan, nord))
end

# ---------------- run --------------------------------------------------------

results = String[]
gate(id, ok, val, lim) = push!(results,
    @sprintf("%-4s %-4s  value=%.3e  gate=%.1e", id, ok ? "PASS" : "FAIL", val, lim))

meshes = Dict(nt => disk_mesh(nt, max(nt ÷ 4, 2)) for nt in (32, 64, 128, 256))
fine = meshes[256]
@printf("mesh sizes: %s\n", join(["$nt->$(length(meshes[nt]))tris" for nt in sort(collect(keys(meshes)))], ", "))

# --- sign + branch identification against polygon-exact reference (finest) ---
probe = [onaxis_pt.(Z_ON); OFFAXIS]
impl = [phi_impl(x, fine) for x in probe]
refp = [phi_ref_poly(x, fine) for x in probe]
s_id = sum(impl .* refp) >= 0 ? 1.0 : -1.0
resid = impl .- s_id .* refp
side = [x[3] > 0 ? :+ : :- for x in probe]
cpos = mean(resid[side .== :+]); cneg = mean(resid[side .== :-])
snap(c) = argmin(abs.(c .- [0.0, MU, -MU]))  # 1->0, 2->+mu, 3->-mu
cpos_s = [0.0, MU, -MU][snap(cpos)]; cneg_s = [0.0, MU, -MU][snap(cneg)]
@printf("identified: s=%+.0f  c+=%.2e (snap %.1f)  c-=%.2e (snap %.1f)\n",
        s_id, cpos, cpos_s, cneg, cneg_s)

# A7: branch constants uniform per side and in {0, +-mu}
a7 = maximum(abs.(resid .- [sd == :+ ? cpos_s : cneg_s for sd in side]))
gate("A7", a7 <= 1e-6*MU, a7, 1e-6)

# A9: kernel vs independent polygon-exact implementation, identical geometry
a9 = maximum(abs.(impl .- (s_id .* refp .+ [sd == :+ ? cpos_s : cneg_s for sd in side])))
gate("A9", a9 <= 1e-6*MU, a9, 1e-6)

corr(x) = (phi_impl(onaxis_pt(x), fine) - (x > 0 ? cpos_s : cneg_s))/s_id  # to plan convention

# A1: jump across sheet at z = +-0.05R vs analytic
jump_i = corr(0.05) - corr(-0.05)
jump_a = phi_axis(0.05) - phi_axis(-0.05)
a1 = abs(jump_i - jump_a)/abs(jump_a)
gate("A1", a1 <= 1e-3, a1, 1e-3)

# A2: on-axis branch match, both sides, finest mesh
errs_on = [abs(corr(z) - phi_axis(z)) for z in Z_ON]
a2 = maximum(errs_on)
gate("A2", a2 <= 1e-3*MU, a2, 1e-3)

# A3: linearity mu=2 vs 2*(mu=1)
a3 = maximum(abs(phi_impl(x, fine; mu=2.0) - 2*phi_impl(x, fine)) /
             max(abs(2*phi_impl(x, fine)), 1e-300) for x in probe)
gate("A3", a3 <= 1e-12, a3, 1e-12)

# A4: orientation reversal negates phi (absolute gate per prereg v2)
fine_flip = [flip(t) for t in fine]
a4 = maximum(abs(phi_impl(x, fine_flip) + phi_impl(x, fine)) for x in probe)
gate("A4", a4 <= 1e-12*MU, a4, 1e-12)

# A5: far-field decay slope both sides
function slope(zs, side)
    lv = [log(abs(phi_impl(onaxis_pt(side*z), fine))) for z in zs]
    lz = log.(collect(zs))
    (length(lz)*sum(lz.*lv) - sum(lz)*sum(lv)) / (length(lz)*sum(lz.^2) - sum(lz)^2)
end
sp, sm = slope(Z_FAR, 1), slope(Z_FAR, -1)
a5 = max(abs(sp + 2), abs(sm + 2))
gate("A5", a5 <= 0.05, a5, 5e-2)
@printf("far-field slopes: +z %.4f, -z %.4f\n", sp, sm)

# A6: circulation via velocity loop integral, with quadrature-doubling check.
# Loop orientation: interior leg ascends +z through the disk, positively
# linking the +z normal, so the registered expectation is Gamma = +mu.
g1 = circulation(fine; npan=96, nord=8)
g2 = circulation(fine; npan=192, nord=8)
a6 = abs(g2 - MU)/abs(MU)
gate("A6", a6 <= 1e-3 && abs(g2 - g1) < 0.1*1e-3*MU, a6, 1e-3)
@printf("circulation: G1=%.8f  G2=%.8f  (quad delta %.2e), sign %+d\n",
        g1, g2, abs(g2-g1), Int(sign(g2)))

# A8: refinement of the A2 error (max over ALL on-axis points) per mesh
e_ref = Float64[]
for nt in (32, 64, 128, 256)
    tris = meshes[nt]
    push!(e_ref, maximum(
        abs((phi_impl(onaxis_pt(z), tris) - (z > 0 ? cpos_s : cneg_s))/s_id -
            phi_axis(z)) for z in Z_ON))
end
orders = [log2(e_ref[i]/e_ref[i+1]) for i in 1:length(e_ref)-1]
a8ok = all(diff(e_ref) .< 0) && minimum(orders) >= 1.5
gate("A8", a8ok, minimum(orders), 1.5)
# A10: wrapper parity — public induced(body) route vs low-level _induced
wp_nodes = Float64[0.3 1.1 0.4; 0.1 0.2 0.9; 0.0 0.0 0.0]
wp_cells = reshape(Int[1, 2, 3], 3, 1)
wp_body = pnl.NonLiftingBody{pnl.ConstantDoublet}(wp_nodes, wp_cells; core_size=CORE)
pnl.calc_normals!(wp_body); pnl.calc_controlpoints!(wp_body)
wp_body.strength[1, 1] = MU
wp_tri = make_tri(SVector{3}(wp_nodes[:, 1]), SVector{3}(wp_nodes[:, 2]),
                  SVector{3}(wp_nodes[:, 3]))
wp_x = SVector(0.7, -0.4, 0.9)
wp_pub, _, _ = pnl.induced(wp_x, wp_body, 1, SW_P; core_size=CORE)
wp_low, _ = pnl._induced(wp_x, wp_tri.verts, wp_tri.centroid, SVector{1}(MU),
                         pnl.ConstantDoublet, CORE, wp_tri.R, SW_P)
a10 = abs(wp_pub - wp_low)
gate("A10", a10 <= 1e-12*MU, a10, 1e-12)

@printf("A8 errors vs ntheta: %s ; orders %s\n",
        join([@sprintf("%.3e", e) for e in e_ref], ", "),
        join([@sprintf("%.2f", o) for o in orders], ", "))

# ---------------- outputs ----------------------------------------------------

open(joinpath(DATADIR, "onaxis_finest.csv"), "w") do io
    println(io, "z,phi_impl,phi_analytic,abs_err")
    for z in sort(Z_ON)
        println(io, "$z,$(phi_impl(onaxis_pt(z), fine)),$(phi_axis(z)),$(abs(corr(z)-phi_axis(z)))")
    end
end
open(joinpath(DATADIR, "refinement_z0p5.csv"), "w") do io
    println(io, "ntheta,abs_err")
    for (nt, e) in zip((32,64,128,256), e_ref)
        println(io, "$nt,$e")
    end
end
function gitinfo(path)
    sha = strip(read(`git -C $path rev-parse HEAD`, String))
    dirty = !isempty(strip(read(`git -C $path status --porcelain`, String)))
    dh = dirty ? bytes2hex(sha256(read(`git -C $path diff`)))[1:12] : ""
    "$sha $(dirty ? "DIRTY tracked-diff=$dh" : "clean")"
end
script_sha = bytes2hex(sha256(read(@__FILE__)))[1:12]
open(joinpath(DATADIR, "gates.txt"), "w") do io
    println(io, "date=2026-09-07 julia=$(VERSION) threads=$(Threads.nthreads()) script_sha256=$(script_sha)")
    println(io, "FLOWPanel: $(gitinfo("/Users/ryan/Dropbox/research/projects/FLOWPanel.jl"))")
    println(io, "FastMultipole: $(gitinfo("/Users/ryan/Dropbox/research/projects/FastMultipole"))")
    println(io, "identified convention: s=$(s_id), c+=$(cpos_s), c-=$(cneg_s)")
    foreach(l -> println(io, l), results)
end

println("\n===== TIER 0A GATES =====")
println("identified convention: phi_impl = s*(-mu*Omega/4pi) + c;  s=$(s_id), c+=$(cpos_s), c-=$(cneg_s)")
foreach(println, results)
println("=========================")
