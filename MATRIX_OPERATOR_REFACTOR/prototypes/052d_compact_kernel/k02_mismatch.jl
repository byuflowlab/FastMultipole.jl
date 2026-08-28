# k02 — far-field kernel-mismatch curves vs exclusion radius R at the
# production step-472 snapshot, for three filament regularizations:
#   gauss     : shipped GaussianRegularization (anchor; must reproduce p32e/f)
#   compact   : shipped CompactRegularization  (cheap A/B, Ryan question)
#   linegauss : the 052d line-convolved Gaussian candidate (this dir)
# Protocol identical to p32e_guard.jl part 1: per (target, panel) pair
# accumulate (U_fam - U_sing) into distance bins; suffix sums give
# relRMS[mismatch beyond R]. Source-panel contribution is included via the
# full shipped `induced` call (same for all families); only the VortexRing
# edge kernel differs.
#
# Env: P32_STRENGTHS=random|solved (default random), NKMISS (default 2000).
#      K02_SNAPSHOT_DIR overrides the production step-472 snapshot directory.
# Run: JULIA_NUM_THREADS=4 julia --project=../FLOWPanel.jl k02_mismatch.jl

include(joinpath(@__DIR__, "linegauss.jl"))
include(joinpath(@__DIR__, "guard_utils.jl"))
using .LineGauss
import FLOWPanel as pnl
import FastMultipole
const FM = FastMultipole
using StaticArrays, Random, Statistics, Printf, LinearAlgebra
using Base.Threads

const DEFAULT_SNAPDIR = "/private/tmp/claude-502/-Users-ryan-Dropbox-research-projects-FastMultipole/1a9c539a-6d87-44ee-87f7-d4e1c17d2793/scratchpad/snapshot472"
const SNAPDIR = get(ENV, "K02_SNAPSHOT_DIR", DEFAULT_SNAPDIR)
const NKMISS = parse(Int, get(ENV, "NKMISS", "2000"))

read3(f) = collect(reshape(reinterpret(Float64, read(joinpath(SNAPDIR, f))), 3, :))
const positions = read3("particle_positions_3xN_f64.bin")
const vpts = read3("panel_vertices_3xM_f64.bin")
const conn = collect(reinterpret(Int64, read(joinpath(SNAPDIR, "panel_connectivity_i64.bin"))))
const cells = collect(reshape(conn, 3, :))
const nt = size(positions, 2)
const ns = size(cells, 2)
const particles = [SVector{3,Float64}(positions[:, i]) for i in 1:nt]
const centroids = [SVector{3,Float64}((vpts[:, cells[1, k]] .+ vpts[:, cells[2, k]] .+
              vpts[:, cells[3, k]]) ./ 3) for k in 1:ns]

function make_body(cst)
    kernel = Union{pnl.ConstantSource, pnl.VortexRing}
    b = pnl.RigidWakeBody{kernel}(vpts, cells, pnl.noshedding; watertight=true,
        DBC=true, core_size_panel=0.12e-10, core_size_targets=cst)
    pnl.calc_normals!(b)
    pnl.calc_controlpoints!(b)
    pnl._set_core_sizes!((b,), :core_size_targets)
    Random.seed!(472)
    b.strength .= 0.1 .* randn(b.ncells, size(b.strength, 2))
    return b
end
const body = make_body(1e-3)
const body_s = make_body(1e-12)
if get(ENV, "P32_STRENGTHS", "random") == "solved"
    import ReadVTK
    vtu = ReadVTK.VTKFile(joinpath(SNAPDIR, "fm052d_gpu_1080_body1.472.vtu"))
    cd = ReadVTK.get_cell_data(vtu)
    sig = ReadVTK.get_data(cd["sigma"])
    gam = ReadVTK.get_data(cd["gamma"])
    @assert length(sig) == ns && length(gam) == ns
    for b in (body, body_s)
        b.strength[:, 1] .= vec(sig)
        b.strength[:, 2] .= vec(gam)
    end
    println("USING SOLVED STRENGTHS (production step-472 sigma/gamma)")
else
    println("USING RANDOM STRENGTHS (seed 472)")
end
const sbuf = FM.system_to_buffer(body)
const sbuf_s = FM.system_to_buffer(body_s)
const ds = FM.DerivativesSwitch(false, true, false)
const rc = body.core_size
println("core size rc = $rc; NKMISS = $NKMISS")

Random.seed!(99)
const idx = sort(shuffle(1:nt)[1:5000])
const kidx = idx[1:NKMISS]

const FAMS = (:gauss, :compact, :linegauss)
const Rgrid = vcat(collect(0.001:0.001:0.020), collect(0.025:0.005:0.10))
const nb = length(Rgrid)

const binacc = Dict(f => [zeros(3, nb + 1) for _ in 1:NKMISS] for f in FAMS)
const Uacc = zeros(3, NKMISS)

# ring-edge sum with a swappable per-edge kernel
@inline function ring_edges(kernelfun, x, v1, v2, v3)
    kernelfun(v1 - x, v2 - x) + kernelfun(v2 - x, v3 - x) + kernelfun(v3 - x, v1 - x)
end

function mismatch_pass!(binacc, Uacc, kidx)
    fam_g = Val(pnl.GaussianRegularization)
    fam_c = Val(pnl.CompactRegularization)
    @threads for s in eachindex(kidx)
        x = particles[kidx[s]]
        acc_g = binacc[:gauss][s]
        acc_c = binacc[:compact][s]
        acc_l = binacc[:linegauss][s]
        for k in 1:ns
            d = norm(x - centroids[k])
            b = searchsortedfirst(Rgrid, d)
            _, Ug, _ = pnl.induced(x, body, sbuf, k, ds, fam_g; core_size=rc)
            _, Uc, _ = pnl.induced(x, body, sbuf, k, ds, fam_c; core_size=rc)
            _, Us, _ = pnl.induced(x, body_s, sbuf_s, k, ds, fam_g;
                                   core_size=body_s.core_size)
            _, v1, v2, v3 = pnl.rotate_to_panel(body, sbuf, k)
            γ = sbuf[6, k]
            ring_g = γ * ring_edges((a, c) ->
                pnl._bound_vortex_velocity(a, c, true, rc, fam_g), x, v1, v2, v3)
            ring_l = γ * ring_edges((a, c) -> lg_velocity(a, c, rc), x, v1, v2, v3)
            Ul = Ug - ring_g + ring_l
            for a in 1:3
                acc_g[a, b] += Ug[a] - Us[a]
                acc_c[a, b] += Uc[a] - Us[a]
                acc_l[a, b] += Ul[a] - Us[a]
                Uacc[a, s] += Ug[a]
            end
        end
    end
end

t1 = @elapsed mismatch_pass!(binacc, Uacc, kidx)
refnorm = sqrt(mean(abs2, Uacc))
@printf("pair pass: %.1f s, rms|U_gauss| = %.4e\n", t1, refnorm)

println("\nfar-mismatch relRMS beyond R (rows: R in m):")
@printf("%7s | %11s %11s %11s\n", "R", "gauss", "compact", "linegauss")
mism = Dict(f => zeros(nb) for f in FAMS)
for (j, R) in enumerate(Rgrid)
    for f in FAMS
        tot = 0.0
        for s in 1:NKMISS
            acc = binacc[f][s]
            dx = 0.0; dy = 0.0; dz = 0.0
            for b in (j + 1):(nb + 1)
                dx += acc[1, b]; dy += acc[2, b]; dz += acc[3, b]
            end
            tot += dx * dx + dy * dy + dz * dz
        end
        mism[f][j] = sqrt(tot / (3 * NKMISS)) / refnorm
    end
    @printf("%7.3f | %11.3e %11.3e %11.3e\n", R, mism[:gauss][j],
            mism[:compact][j], mism[:linegauss][j])
end

for thr in (3e-5, 1e-5)
    for f in FAMS
        jg = first_suffix_safe(mism[f], thr)
        println("R_guard(<=$(thr), $f) = ",
                jg === nothing ? "> $(Rgrid[end]) m (NOT REACHED)" : "$(Rgrid[jg]) m")
    end
end
println("DONE")
