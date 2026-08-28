# P3.2e — guard-radius selection + guarded accuracy harness.
#
# Part 1: aggregate far-field kernel mismatch vs exclusion radius R:
#   for 2000 sampled targets, accumulate (U_reg - U_sing) per (target, panel)
#   pair into log-spaced distance bins; suffix sums give
#   relRMS[ mismatch of all pairs beyond R ] as a function of R.
#   R_guard = smallest grid R with mismatch <= 3e-5 (margin under the 1e-4
#   ceiling, ruling R4).
# Part 2: guarded pipeline (p32_harness with physical-distance route
#   demotion): any M2L route whose cell boxes are closer than R_guard is
#   evaluated DIRECT (same pair set -> exact-once trivially preserved).
#   Sweep q in {3,12} x ell_x in {5,7} x P in {3,4,6} at R_guard and
#   1.5*R_guard; report relRMS, margin, demoted pair-interaction counts.
#
# Run: JULIA_NUM_THREADS=4 julia --project=../FLOWPanel.jl p32e_guard.jl

include(joinpath(@__DIR__, "CrossStencil.jl"))
using .CrossStencil
import FLOWPanel as pnl
import FastMultipole
const FM = FastMultipole
using StaticArrays, Random, Statistics, Printf, LinearAlgebra
using Base.Threads

const SNAPDIR = "/private/tmp/claude-502/-Users-ryan-Dropbox-research-projects-FastMultipole/1a9c539a-6d87-44ee-87f7-d4e1c17d2793/scratchpad/snapshot472"
const NSAMPLE = 5000
const NKMISS = 2000     # targets for the part-1 mismatch curve

read3(f) = collect(reshape(reinterpret(Float64, read(joinpath(SNAPDIR, f))), 3, :))
positions = read3("particle_positions_3xN_f64.bin")
vpts = read3("panel_vertices_3xM_f64.bin")
conn = collect(reinterpret(Int64, read(joinpath(SNAPDIR, "panel_connectivity_i64.bin"))))
cells = collect(reshape(conn, 3, :))
nt = size(positions, 2)
ns = size(cells, 2)
particles = [SVector{3,Float64}(positions[:, i]) for i in 1:nt]
centroids = [SVector{3,Float64}((vpts[:, cells[1, k]] .+ vpts[:, cells[2, k]] .+
              vpts[:, cells[3, k]]) ./ 3) for k in 1:ns]
g = CrossGrid(particles)

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
body = make_body(1e-3)
body_s = make_body(1e-12)
# P32_STRENGTHS=solved : replace the random strengths with the SOLVED per-panel
# sigma/gamma cell data from the step-472 body .vtu (production fingerprint).
if get(ENV, "P32_STRENGTHS", "random") == "solved"
    import ReadVTK
    vtu = ReadVTK.VTKFile(joinpath(SNAPDIR, "fm052d_gpu_1080_body1.472.vtu"))
    cd = ReadVTK.get_cell_data(vtu)
    sig = ReadVTK.get_data(cd["sigma"])
    gam = ReadVTK.get_data(cd["gamma"])
    println("solved strengths: sigma size=", size(sig), " gamma size=", size(gam))
    @assert length(sig) == ns && length(gam) == ns
    for b in (body, body_s)
        b.strength[:, 1] .= vec(sig)
        b.strength[:, 2] .= vec(gam)
    end
    println("USING SOLVED STRENGTHS (production step-472 sigma/gamma)")
else
    println("USING RANDOM STRENGTHS (seed 472)")
end
lhv = Val(FM.has_vector_potential((body,)))
sbuf = FM.system_to_buffer(body)
sbuf_s = FM.system_to_buffer(body_s)
const ds = FM.DerivativesSwitch(false, true, false)
const fam = Val(pnl.FILAMENT_REGULARIZATION[])

Random.seed!(99)
idx = sort(shuffle(1:nt)[1:NSAMPLE])

# ---------------- Part 1: aggregate mismatch vs exclusion radius ----------------
println("=== 1. aggregate far mismatch vs exclusion radius (", NKMISS, " targets) ===")
Rgrid = collect(0.005:0.005:0.10)
nb = length(Rgrid)
kidx = idx[1:NKMISS]
binacc = [zeros(3, nb + 1) for _ in 1:NKMISS]   # per-target, per-bin sum of dU
Uacc = zeros(3, NKMISS)                          # regularized total (norm ref)
function mismatch_pass!(binacc, Uacc, kidx, particles, centroids, body, sbuf,
        body_s, sbuf_s, ds, fam, Rgrid, ns)
    @threads for s in eachindex(kidx)
        x = particles[kidx[s]]
        acc = binacc[s]
        for k in 1:ns
            d = norm(x - centroids[k])
            _, Ur, _ = pnl.induced(x, body, sbuf, k, ds, fam; core_size=body.core_size)
            _, Us, _ = pnl.induced(x, body_s, sbuf_s, k, ds, fam; core_size=body_s.core_size)
            b = searchsortedfirst(Rgrid, d)  # pairs beyond R_j live in bins > j
            for a in 1:3
                acc[a, b] += Ur[a] - Us[a]
                Uacc[a, s] += Ur[a]
            end
        end
    end
end
t1 = @elapsed mismatch_pass!(binacc, Uacc, kidx, particles, centroids, body, sbuf,
    body_s, sbuf_s, ds, fam, Rgrid, ns)
refnorm1 = sqrt(mean(abs2, Uacc))
@printf("pair pass: %.1f s, rms|U_reg|=%.4e\n", t1, refnorm1)
mism = zeros(nb)
for (j, R) in enumerate(Rgrid)
    # mismatch from pairs with d > R : bins j+1 .. nb+1
    tot = 0.0
    for s in 1:NKMISS
        acc = binacc[s]
        dx = 0.0; dy = 0.0; dz = 0.0
        for b in (j + 1):(nb + 1)
            dx += acc[1, b]; dy += acc[2, b]; dz += acc[3, b]
        end
        tot += dx * dx + dy * dy + dz * dz
    end
    mism[j] = sqrt(tot / (3 * NKMISS)) / refnorm1
    @printf("  R=%5.3f m : far-mismatch relRMS = %.3e\n", R, mism[j])
end
jg = findfirst(<=(3e-5), mism)
R_guard = jg === nothing ? 0.10 : Rgrid[jg]
println("R_guard(<=3e-5) = $R_guard m", jg === nothing ? "  (NOT REACHED — capped)" : "")
println()

# ---------------- Part 2: guarded pipeline ----------------
cellcenter(g, c::SVector{3,Int}, L) = begin
    h = g.h0 / (1 << L)
    SVector(g.x_min[1] + (2c[1] + 1) * h, g.x_min[2] + (2c[2] + 1) * h,
            g.x_min[3] + (2c[3] + 1) * h)
end
decode3(code) = SVector{3,Int}(Int.(CrossStencil.SharedRadix.morton_decode(code))...)
dummy_branch(center) = FM.Branch(1:1, 0, 1:0, 0, 1, center, 0.0, SVector(0.0, 0.0, 0.0))

"Minimum distance between two same-level cell boxes at integer offset o."
@inline boxgap(o::SVector{3,Int}, w) =
    w * norm(SVector(max(abs(o[1]) - 1, 0), max(abs(o[2]) - 1, 0), max(abs(o[3]) - 1, 0)))

function run_guarded(q, ell_x, P, Rg)
    tq = CrossStencil.UniformQTables(q)
    npush = length(tq.push_offsets)
    member = falses(8, npush)
    for ph in 1:8, k in tq.by_phase[ph]
        member[ph, k] = true
    end
    src_levels, sorder, _ = build_level_cells(g, centroids, ell_x)
    anc = [Dict{UInt64,Vector{Int}}() for _ in 0:ell_x]
    for (s, j) in enumerate(idx)
        c, _ = CrossStencil.level_coords(g, particles[j], ell_x)
        code = CrossStencil.SharedRadix.morton_encode(UInt64(c[1]), UInt64(c[2]), UInt64(c[3]))
        for L in 0:ell_x
            push!(get!(Vector{Int}, anc[L + 1], code >> (3 * (ell_x - L))), s)
        end
    end
    FM.update_Hs_π2!(FM.Hs_π2, P)
    FM.update_ζs_mag!(FM.ζs_mag, P); FM.update_ηs_mag!(FM.ηs_mag, P)
    FM.update_M̃!(FM.M̃, P); FM.update_L̃!(FM.L̃, P)
    harmonics = FM.initialize_harmonics(P)
    gnm = FM.initialize_gradient_n_m(P)
    w1 = FM.initialize_expansion(P); w2 = FM.initialize_expansion(P)
    w3 = FM.initialize_expansion(P)
    Ts = zeros(FM.length_Ts(P)); eimϕs = zeros(2, P + 1)

    mult = [Dict{UInt64,Array{Float64,3}}() for _ in 0:ell_x]
    leaf = src_levels[ell_x + 1]
    for ci in 1:length(leaf.codes)
        e = get!(() -> FM.initialize_expansion(P), mult[ell_x + 1], leaf.codes[ci])
        ctr = cellcenter(g, decode3(leaf.codes[ci]), ell_x)
        for k in leaf.starts[ci]:leaf.starts[ci + 1] - 1
            i = sorder[k]
            FM.body_to_multipole!(body, e, sbuf, ctr, i:i, harmonics, P)
        end
    end
    for L in ell_x-1:-1:0
        for (ccode, ce) in mult[L + 2]
            pcode = ccode >> 3
            pe = get!(() -> FM.initialize_expansion(P), mult[L + 1], pcode)
            pb = dummy_branch(cellcenter(g, decode3(pcode), L))
            cb = dummy_branch(cellcenter(g, decode3(ccode), L + 1))
            FM.multipole_to_multipole!(pe, pb, ce, cb, w1, w2, Ts, eimϕs,
                FM.ζs_mag, FM.Hs_π2, P, lhv)
        end
    end
    # demoted routes: (panel sorted-order range at level L, sample list)
    demoted = Tuple{UnitRange{Int},Vector{Int}}[]
    locals_ = [Dict{UInt64,Array{Float64,3}}() for _ in 0:ell_x]
    n_m2l = 0
    n_demoted = 0
    for L in 2:ell_x
        G = 1 << L
        w = 2 * g.h0 / G
        slc = src_levels[L + 1]
        for (bcode, ss) in anc[L + 1]
            bl = get!(() -> FM.initialize_expansion(P), locals_[L + 1], bcode)
            bc = decode3(bcode)
            bb = dummy_branch(cellcenter(g, bc, L))
            if L > 2
                pl = get(locals_[L], bcode >> 3, nothing)
                if pl !== nothing
                    pb = dummy_branch(cellcenter(g, decode3(bcode >> 3), L - 1))
                    FM.local_to_local!(bl, bb, pl, pb, w1, w2, Ts, eimϕs,
                        FM.ηs_mag, FM.Hs_π2, P, lhv)
                end
            end
            for k in 1:npush
                o = tq.push_offsets[k]
                ax = bc[1] - o[1]; ay = bc[2] - o[2]; az = bc[3] - o[3]
                (0 <= ax < G && 0 <= ay < G && 0 <= az < G) || continue
                member[1 + (ax & 1) + 2 * (ay & 1) + 4 * (az & 1), k] || continue
                acode = CrossStencil.SharedRadix.morton_encode(UInt64(ax), UInt64(ay), UInt64(az))
                ci = CrossStencil.cell_index(slc, acode)
                ci == 0 && continue
                if boxgap(o, w) < Rg
                    push!(demoted, (slc.starts[ci]:slc.starts[ci + 1] - 1, ss))
                    n_demoted += 1
                    continue
                end
                ab = dummy_branch(cellcenter(g, SVector(ax, ay, az), L))
                FM.multipole_to_local!(bl, bb, mult[L + 1][acode], ab, w1, w2, w3,
                    Ts, eimϕs, FM.ζs_mag, FM.ηs_mag, FM.Hs_π2, FM.M̃, FM.L̃, P, lhv, nothing)
                n_m2l += 1
            end
        end
    end
    U = zeros(3, NSAMPLE)
    for (bcode, ss) in anc[ell_x + 1]
        bl = get(locals_[ell_x + 1], bcode, nothing)
        bl === nothing && continue
        ctr = cellcenter(g, decode3(bcode), ell_x)
        for s in ss
            Δx = particles[idx[s]] - ctr
            _, grad, _ = FM.evaluate_local(Δx, harmonics, gnm, bl, P, lhv, ds)
            U[:, s] .= grad
        end
    end
    # near field (stencil near shell) + demoted routes, both direct/regularized
    tb = zeros(7, NSAMPLE)
    tb[1:3, :] .= positions[:, idx]
    near_int = 0
    G = 1 << ell_x
    for (bcode, ss) in anc[ell_x + 1]
        bc = decode3(bcode)
        srcids = Int[]
        for o in tq.near_offsets
            ax = bc[1] - o[1]; ay = bc[2] - o[2]; az = bc[3] - o[3]
            (0 <= ax < G && 0 <= ay < G && 0 <= az < G) || continue
            acode = CrossStencil.SharedRadix.morton_encode(UInt64(ax), UInt64(ay), UInt64(az))
            ci = CrossStencil.cell_index(src_levels[ell_x + 1], acode)
            ci == 0 && continue
            lc = src_levels[ell_x + 1]
            append!(srcids, (sorder[k] for k in lc.starts[ci]:lc.starts[ci + 1] - 1))
        end
        isempty(srcids) && continue
        near_int += length(srcids) * length(ss)
        FM.direct!(tb, ss, ds, body, sbuf, srcids)
    end
    dem_int = 0
    for (srange, ss) in demoted
        srcids = [sorder[k] for k in srange]
        dem_int += length(srcids) * length(ss)
        FM.direct!(tb, ss, ds, body, sbuf, srcids)
    end
    U .+= tb[4:6, :]
    return U, n_m2l, n_demoted, near_int, dem_int
end

function dense_ref(b, idx)
    sub = FM.ProbeSystemArray(length(idx))
    sub.position .= positions[:, idx]
    sub.gradient .= 0
    FM.direct!((sub,), (b,); scalar_potential=false, gradient=true, hessian=false)
    return copy(sub.gradient)
end
Uref = dense_ref(body, idx)
refnorm = sqrt(mean(abs2, Uref))

println("=== 2. guarded accuracy sweep (R_guard=$R_guard and 1.5x) ===")
@printf("%-6s %-4s %-5s %-3s | %9s %7s | %7s %7s %9s %10s\n",
        "Rg", "q", "ellx", "P", "relRMS_U", "margin", "n_m2l", "n_dem", "near_int", "dem_int")
# 052e step 1: LineGauss guard certification — fixed guard radii (6 mm, 12 mm)
Rg_list = (0.006, 0.012)
for Rg in Rg_list
    for q in (3, 12), ell_x in (5, 7), P in (3, 4, 6)
        U, n_m2l, n_dem, near_int, dem_int = run_guarded(q, ell_x, P, Rg)
        relrms = sqrt(mean(abs2, U .- Uref)) / refnorm
        @printf("%-6.3f %-4d %-5d %-3d | %9.3e %7.2f | %7d %7d %9d %10d\n",
                Rg, q, ell_x, P, relrms, 1e-4 / relrms, n_m2l, n_dem, near_int, dem_int)
        flush(stdout)
    end
end
println("DONE")
