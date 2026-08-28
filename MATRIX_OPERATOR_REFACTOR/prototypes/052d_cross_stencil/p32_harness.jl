# P3.2 — host accuracy/cost harness for the 052d cross pass at production
# step-472 shape. Full honest pipeline on the two-occupancy uniform-q rigid
# stencil lists: per-panel B2M at leaf cells (TIMED, 36,752 panels) -> M2M up
# -> cross-M2L per route (host translate ops, Lamb-Helmholtz on) -> L2L down
# -> U-only L2B at sampled particles + near-field direct (production
# regularized mixed ConstantSource+VortexRing kernels). Velocity relRMS vs a
# dense direct! reference on a 5000-particle sample, vs the 1e-4 relU ceiling
# (ruling R4: report MARGIN per config).
#
# Sweep: q in {3,5,12} x ell_x in {5,6,7,8} x P in {3,4,6,8}.
# Run: JULIA_NUM_THREADS=4 julia --project=<FastMultipole> p32_harness.jl

include(joinpath(@__DIR__, "CrossStencil.jl"))
using .CrossStencil
import FLOWPanel as pnl
import FastMultipole
const FM = FastMultipole
using StaticArrays, Random, Statistics, Printf, LinearAlgebra

const SNAPDIR = "/private/tmp/claude-502/-Users-ryan-Dropbox-research-projects-FastMultipole/1a9c539a-6d87-44ee-87f7-d4e1c17d2793/scratchpad/snapshot472"
const QS = (3, 5, 12)
const ELLXS = (5, 6, 7, 8)
const PS = (3, 4, 6, 8)
const NSAMPLE = 5000
const CEIL = 1e-4

read3(f) = collect(reshape(reinterpret(Float64, read(joinpath(SNAPDIR, f))), 3, :))
positions = read3("particle_positions_3xN_f64.bin")
vpts = read3("panel_vertices_3xM_f64.bin")
conn = collect(reinterpret(Int64, read(joinpath(SNAPDIR, "panel_connectivity_i64.bin"))))
@assert maximum(conn) == size(vpts, 2) && minimum(conn) >= 1
cells = collect(reshape(conn, 3, :))
nt = size(positions, 2)

# --- production body construction (profile_fmm.jl:28-38, validated route) ---
kernel = Union{pnl.ConstantSource, pnl.VortexRing}
body = pnl.RigidWakeBody{kernel}(vpts, cells, pnl.noshedding; watertight=true,
    DBC=true, core_size_panel=0.12e-10, core_size_targets=1e-3)
pnl.calc_normals!(body)
pnl.calc_controlpoints!(body)
pnl._set_core_sizes!((body,), :core_size_targets)
Random.seed!(472)
body.strength .= 0.1 .* randn(body.ncells, size(body.strength, 2))
ns = body.ncells
lh = FM.has_vector_potential((body,))
lhv = Val(lh)
println("body ncells=$ns lamb_helmholtz=$lh core_size=$(body.core_size) threads=$(Threads.nthreads())")

centroids = [SVector{3,Float64}((vpts[:, cells[1, k]] .+ vpts[:, cells[2, k]] .+
              vpts[:, cells[3, k]]) ./ 3) for k in 1:ns]
particles = [SVector{3,Float64}(positions[:, i]) for i in 1:nt]
g = CrossGrid(particles)

sbuf = FM.system_to_buffer(body)
ds = FM.DerivativesSwitch(false, true, false)

# --- dense reference on the sample (regularized production kernels) ---
Random.seed!(99)
idx = sort(shuffle(1:nt)[1:NSAMPLE])
sub = FM.ProbeSystemArray(NSAMPLE)
sub.position .= positions[:, idx]
sub.gradient .= 0
td = @elapsed FM.direct!((sub,), (body,); scalar_potential=false, gradient=true, hessian=false)
Uref = copy(sub.gradient)
refnorm = sqrt(mean(abs2, Uref))
@printf("dense reference: %d targets, %.1f s, rms|U|=%.4e\n\n", NSAMPLE, td, refnorm)

cellcenter(g, c::SVector{3,Int}, L) = begin
    h = g.h0 / (1 << L)
    SVector(g.x_min[1] + (2c[1] + 1) * h, g.x_min[2] + (2c[2] + 1) * h,
            g.x_min[3] + (2c[3] + 1) * h)
end
decode3(code) = SVector{3,Int}(Int.(CrossStencil.SharedRadix.morton_decode(code))...)
# test/bodytomultipole_test.jl:361 pattern — only .center is load-bearing
dummy_branch(center) = FM.Branch(1:1, 0, 1:0, 0, 1, center, 0.0, SVector(0.0, 0.0, 0.0))

# --- JIT warmup: exercise every op once on a tiny problem (P value-typed, so
# --- no per-P recompiles; Val(lh) fixed) ---
let P = 3
    FM.update_Hs_π2!(FM.Hs_π2, P)
    FM.update_ζs_mag!(FM.ζs_mag, P); FM.update_ηs_mag!(FM.ηs_mag, P)
    FM.update_M̃!(FM.M̃, P); FM.update_L̃!(FM.L̃, P)
    h = FM.initialize_harmonics(P); gnm = FM.initialize_gradient_n_m(P)
    e1 = FM.initialize_expansion(P); e2 = FM.initialize_expansion(P)
    w1 = FM.initialize_expansion(P); w2 = FM.initialize_expansion(P)
    w3 = FM.initialize_expansion(P)
    Ts = zeros(FM.length_Ts(P)); eim = zeros(2, P + 1)
    c1 = SVector(0.0, 0.0, 0.0); c2 = SVector(0.05, 0.0, 0.0)
    b1 = dummy_branch(c1); b2 = dummy_branch(c2)
    FM.body_to_multipole!(body, e1, sbuf, c1, 1:1, h, P)
    FM.multipole_to_multipole!(e2, b2, e1, b1, w1, w2, Ts, eim, FM.ζs_mag, FM.Hs_π2, P, lhv)
    FM.multipole_to_local!(e2, b2, e1, b1, w1, w2, w3, Ts, eim, FM.ζs_mag, FM.ηs_mag,
        FM.Hs_π2, FM.M̃, FM.L̃, P, lhv, nothing)
    FM.local_to_local!(e1, b1, e2, b2, w1, w2, Ts, eim, FM.ηs_mag, FM.Hs_π2, P, lhv)
    FM.evaluate_local(SVector(0.01, 0.0, 0.0), h, gnm, e1, P, lhv, ds)
    wtb = zeros(7, 1); wtb[1:3, 1] .= (0.1, 0.0, 0.0)
    FM.direct!(wtb, 1:1, ds, body, sbuf, 1:2)
    println("warmup done")
end

@printf("%-4s %-5s %-3s | %9s | %8s %8s %8s | %10s %8s %8s %8s | %s\n",
        "q", "ellx", "P", "relRMS_U", "margin", "n_m2l", "nearInt",
        "t_b2m(s)", "t_m2m", "t_m2l", "t_near", "verdict")
results = []
for q in QS
    tq = CrossStencil.UniformQTables(q)
    npush = length(tq.push_offsets)
    member = falses(8, npush)
    for ph in 1:8, k in tq.by_phase[ph]
        member[ph, k] = true
    end
    for ell_x in ELLXS
        src_levels, sorder, _ = build_level_cells(g, centroids, ell_x)
        # sampled particles: coords at every level + leaf grouping
        pcoords = [CrossStencil.level_coords(g, particles[j], ell_x)[1] for j in idx]
        anc = [Dict{UInt64,Vector{Int}}() for _ in 0:ell_x]  # level -> code -> sample indices
        for (s, c) in enumerate(pcoords)
            code = CrossStencil.SharedRadix.morton_encode(UInt64(c[1]), UInt64(c[2]), UInt64(c[3]))
            for L in 0:ell_x
                push!(get!(Vector{Int}, anc[L + 1], code >> (3 * (ell_x - L))), s)
            end
        end
        for P in PS
            FM.update_Hs_π2!(FM.Hs_π2, P)
            FM.update_ζs_mag!(FM.ζs_mag, P); FM.update_ηs_mag!(FM.ηs_mag, P)
            FM.update_M̃!(FM.M̃, P); FM.update_L̃!(FM.L̃, P)
            harmonics = FM.initialize_harmonics(P)
            gnm = FM.initialize_gradient_n_m(P)
            w1 = FM.initialize_expansion(P); w2 = FM.initialize_expansion(P)
            w3 = FM.initialize_expansion(P)
            Ts = zeros(FM.length_Ts(P)); eimϕs = zeros(2, P + 1)

            # ---- upward: per-panel B2M at leaf cells (timed), then M2M ----
            mult = [Dict{UInt64,Array{Float64,3}}() for _ in 0:ell_x]
            leaf = src_levels[ell_x + 1]
            t_b2m = @elapsed for ci in 1:length(leaf.codes)
                e = get!(() -> FM.initialize_expansion(P), mult[ell_x + 1], leaf.codes[ci])
                ctr = cellcenter(g, decode3(leaf.codes[ci]), ell_x)
                for k in leaf.starts[ci]:leaf.starts[ci + 1] - 1
                    i = sorder[k]
                    FM.body_to_multipole!(body, e, sbuf, ctr, i:i, harmonics, P)
                end
            end
            t_m2m = @elapsed for L in ell_x-1:-1:0
                for (ccode, ce) in mult[L + 2]
                    pcode = ccode >> 3
                    pe = get!(() -> FM.initialize_expansion(P), mult[L + 1], pcode)
                    pb = dummy_branch(cellcenter(g, decode3(pcode), L))
                    cb = dummy_branch(cellcenter(g, decode3(ccode), L + 1))
                    FM.multipole_to_multipole!(pe, pb, ce, cb, w1, w2, Ts, eimϕs,
                        FM.ζs_mag, FM.Hs_π2, P, lhv)
                end
            end

            # ---- downward on sample-ancestor cells: L2L + M2L per level ----
            locals_ = [Dict{UInt64,Array{Float64,3}}() for _ in 0:ell_x]
            n_m2l = 0
            t_m2l = @elapsed for L in 2:ell_x
                slc = src_levels[L + 1]
                G = 1 << L
                for (bcode, _) in anc[L + 1]
                    bl = get!(() -> FM.initialize_expansion(P), locals_[L + 1], bcode)
                    bc = decode3(bcode)
                    bb = dummy_branch(cellcenter(g, bc, L))
                    # L2L from parent (parent locals exist for L > 2)
                    if L > 2
                        pl = get(locals_[L], bcode >> 3, nothing)
                        if pl !== nothing
                            pb = dummy_branch(cellcenter(g, decode3(bcode >> 3), L - 1))
                            FM.local_to_local!(bl, bb, pl, pb, w1, w2, Ts, eimϕs,
                                FM.ηs_mag, FM.Hs_π2, P, lhv)
                        end
                    end
                    # M2L: enumerate candidate sources A = B - o
                    for k in 1:npush
                        o = tq.push_offsets[k]
                        ax = bc[1] - o[1]; ay = bc[2] - o[2]; az = bc[3] - o[3]
                        (0 <= ax < G && 0 <= ay < G && 0 <= az < G) || continue
                        member[1 + (ax & 1) + 2 * (ay & 1) + 4 * (az & 1), k] || continue
                        acode = CrossStencil.SharedRadix.morton_encode(UInt64(ax), UInt64(ay), UInt64(az))
                        ae = get(mult[L + 1], acode, nothing)
                        ae === nothing && continue
                        ab = dummy_branch(cellcenter(g, SVector(ax, ay, az), L))
                        FM.multipole_to_local!(bl, bb, ae, ab, w1, w2, w3, Ts, eimϕs,
                            FM.ζs_mag, FM.ηs_mag, FM.Hs_π2, FM.M̃, FM.L̃, P, lhv, nothing)
                        n_m2l += 1
                    end
                end
            end

            # ---- L2B at sampled particles (U only) ----
            U = zeros(3, NSAMPLE)
            leaflocals = locals_[ell_x + 1]
            for (bcode, ss) in anc[ell_x + 1]
                bl = get(leaflocals, bcode, nothing)
                bl === nothing && continue
                ctr = cellcenter(g, decode3(bcode), ell_x)
                for s in ss
                    Δx = particles[idx[s]] - ctr
                    _, grad, _ = FM.evaluate_local(Δx, harmonics, gnm, bl, P, lhv, ds)
                    U[:, s] .= grad
                end
            end

            # ---- near field: production regularized direct kernel ----
            tb = zeros(7, NSAMPLE)
            tb[1:3, :] .= positions[:, idx]
            near_int = 0
            t_near = @elapsed for (bcode, ss) in anc[ell_x + 1]
                bc = decode3(bcode)
                G = 1 << ell_x
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
            U .+= tb[4:6, :]   # gradient_range with PS=false, NM=0 is rows 4:6

            err = U .- Uref
            relrms = sqrt(mean(abs2, err)) / refnorm
            margin = CEIL / relrms
            verdict = relrms <= CEIL ? "PASS" : "fail"
            @printf("%-4d %-5d %-3d | %9.3e | %8.2f %8d %8d | %10.3f %8.3f %8.3f %8.3f | %s\n",
                    q, ell_x, P, relrms, margin, n_m2l, near_int, t_b2m, t_m2m,
                    t_m2l, t_near, verdict)
            push!(results, (; q, ell_x, P, relrms, margin, n_m2l, near_int, t_b2m))
            flush(stdout)
        end
    end
end

println()
pass = filter(r -> r.relrms <= CEIL, results)
if !isempty(pass)
    println("Configs clearing 1e-4, sorted by (P, q, ell_x) cost proxy:")
    for r in sort(pass; by=r -> (r.P, r.q, r.ell_x))
        @printf("  q=%-3d ell_x=%d P=%d : relRMS=%.3e margin=%.1fx b2m=%.2fs\n",
                r.q, r.ell_x, r.P, r.relrms, r.margin, r.t_b2m)
    end
else
    println("NO CONFIG CLEARED 1e-4")
end
println("DONE")
