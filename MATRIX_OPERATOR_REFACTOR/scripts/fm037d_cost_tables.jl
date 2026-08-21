# fm037d_cost_tables.jl -- 037d paper cost model tables
#
# Generates the cost tables for theory/fourier-nearfield-cost-model.md from
# (a) the measured error-model spot check (data/fm037d_error_spotcheck.csv),
# (b) the 037b same-job anchors (data/flowvpm_gpu_campaign/fm037b_screen.csv
#     and fm037b_confirm.csv), and
# (c) recorded hardware/throughput assumptions (all defined below, next to
#     their use, per the 037d task-file constraint).
#
# No hardware runs; stdlib only. Output: data/fm037d_cost_tables.csv and
# data/fm037d_ewald_rotor.csv.
#
# ------------------------- PRICED ASSUMPTIONS (each used where marked) -----
# A1  H200 HBM3e peak bandwidth 4.8 TB/s; effective fractions for large
#     bandwidth-bound kernels: optimistic 0.70, nominal 0.50, pessimistic 0.30.
# A2  3D C2C FFT cost model: bandwidth-bound, 3 passes (one per dimension,
#     transposes folded into the effective fraction), read+write per pass:
#     48 bytes/point/transform at F32 complex (8 B), 96 at F64. R2C symmetry
#     savings are ignored (conservative).
# A3  Transforms per step: 6 (3 forward vorticity + 3 inverse U). J is
#     obtained by analytic B-spline-derivative interpolation of the U meshes
#     (validated by fm037d_error_spotcheck.jl; J is a campaign diagnostic,
#     not a gate). The kernel spectrum K_hat is tabulated at construction;
#     per-step re-tabulation on geometry drift is priced into the pessimistic
#     band as +3 transforms.
# A4  k-space multiply + sinc^(2p) deconvolution: one sweep reading 3 omega_hat
#     + 3 K_hat and writing 3 U_hat = 72 B/point (F32), 144 (F64).
# A5  Spread rate (order p=4, 3 components, Morton-sorted particles with
#     shared-memory tiling -- the sort already exists in the radix path):
#     optimistic 2e9, nominal 8e8, pessimistic 3e8 particles/s; scaled by
#     (4/p)^3 for other orders. Calibration: cuFINUFFT-class spreaders reach
#     ~1e9 pts/s at width 4 on A100-class parts; pessimistic assumes
#     unsorted-atomics-like behavior.
# A6  Interpolate U + J rate: 0.5x the spread rate (gather of 3 fields with
#     value+derivative weights, 12 outputs), same (4/p)^3 scaling.
# A7  Fixed launch/orchestration overhead per evaluation: 0.5 / 1.0 / 2.0 ms
#     (opt/nom/pess); calibrated from this project's measured ~50 us/launch
#     H200 latency and a ~10-25-kernel pipeline.
# A8  h/sigma selection from the spot check: largest measured h/sigma whose
#     sampled u_rel_rms <= 1.0e-3 (opt) or <= 7e-4 (nom); pessimistic takes
#     the TIGHTEST measured h/sigma (0.40), because the spot check's
#     sinc-deconvolution error floor (~5e-4) means no measured point clears
#     3.5e-4 -- see the floor note in fm037d_error_spotcheck.jl. The gate is
#     1e-3 and full-VIC replaces the FMM entirely, so the whole budget is
#     available; nom/pess hold margin. p chosen per band to minimize total
#     modeled time (p in {4,6}).
# A9  Free-space handling: Hockney doubling (x2 per axis), each padded dim
#     rounded up to the next 5-smooth FFT size. Interior dims include +p
#     spread-support margin.
# A10 F64 variant: all bandwidth terms x2, spread/interp rates x0.5
#     (bandwidth-dominated), same h/sigma (spot check is F64 arithmetic;
#     mesh truncation dominates rounding).
# A11 Ewald split (rotor): real-space cutoff r_c = 3.5*alpha (the 037b
#     truncation wall, rho_x >= 3.2-3.668, rounded to 3.5); neighbor count
#     anchored to the MEASURED rotor pair counts:
#     ell8 q6 -> 1600 neighbors within sqrt(6)*w8 = 1.151e-2 m, and
#     ell6 q6 -> 26679 within 4.605e-2 m, giving measured density exponent
#     d = log(26679/1600)/log(4) = 2.03. Band d = 1.7/2.0/2.4.
#     Pair rate: measured rotor fused NF <= 7.8e10 pairs/s (F32); Ewald pair
#     is priced at rate/1.3 (one-exp outer form, theory 6.2). Band
#     1e11/7e10/4e10 pairs/s before the 1.3 factor.
# A12 Post-038 comparison bar: cube/wake anchors unchanged (auto-geometry
#     already optimal there); rotor uses the measured pinned-depth
#     partitioned winners (F32 7.006/33.971 ms, F64 11.491/60.867 ms) as a
#     stand-in lower bound for the adaptive octree's benefit.
# ---------------------------------------------------------------------------

using Printf

const DATA = joinpath(@__DIR__, "..", "data")

# ---------------------------------------------------------------- CSV utils
function read_csv(path)
    lines = readlines(path)
    hdr = split(lines[1], ',')
    rows = [split(l, ',') for l in lines[2:end] if !isempty(l)]
    idx = Dict(String(h) => i for (i, h) in enumerate(hdr))
    return idx, rows
end

# ------------------------------------------------- anchors from 037b CSVs
function load_anchors()
    anchors = Dict{Tuple{String,Int,String},Float64}()  # (case,n,tf)=>uj_ms
    idx, rows = read_csv(joinpath(DATA, "flowvpm_gpu_campaign", "fm037b_screen.csv"))
    for r in rows
        if endswith(r[idx["label"]], "_anchor")
            anchors[(String(r[idx["case"]]), parse(Int, r[idx["n"]]),
                     String(r[idx["tf"]]))] = parse(Float64, r[idx["uj_ms_median"]])
        end
    end
    idx, rows = read_csv(joinpath(DATA, "flowvpm_gpu_campaign", "fm037b_confirm.csv"))
    for r in rows
        lbl = r[idx["label"]]
        if occursin("_anchor_", lbl)
            anchors[(String(r[idx["case"]]), parse(Int, r[idx["n"]]),
                     String(r[idx["tf"]]))] = parse(Float64, r[idx["uj_ms_median"]])
        end
    end
    return anchors
end

# Post-038 bar (A12): rotor pinned-depth measured winners.
const POST038 = Dict(
    ("rotor", 100000, "Float32") => 7.006, ("rotor", 1000000, "Float32") => 33.971,
    ("rotor", 100000, "Float64") => 11.491, ("rotor", 1000000, "Float64") => 60.867)

# ------------------------------------------------- error-model h/sigma picks
function load_hsigma()
    idx, rows = read_csv(joinpath(DATA, "fm037d_error_spotcheck.csv"))
    # per p: sorted (h/sigma, err) descending in h/sigma
    tbl = Dict{Int,Vector{Tuple{Float64,Float64}}}()
    for r in rows
        p = parse(Int, r[idx["p"]])
        push!(get!(tbl, p, Tuple{Float64,Float64}[]),
              (parse(Float64, r[idx["h_over_sigma"]]),
               parse(Float64, r[idx["u_rel_rms"]])))
    end
    for v in values(tbl); sort!(v; rev=true); end
    # thr > 0: largest measured h/sigma with u_rel_rms <= thr.
    # thr == 0 (pessimistic, A8): the tightest measured h/sigma for this p --
    # the spot check's sinc-deconvolution floor (~5e-4) prevents an unmeasured
    # 3.5e-4 threshold from being meaningful; the tightest measured point is
    # the conservative, evidence-backed stand-in.
    pick(p, thr) = begin
        thr == 0.0 && return minimum(first.(tbl[p]))
        for (hs, e) in tbl[p]
            e <= thr && return hs
        end
        return NaN  # no measured point passes; caller must handle
    end
    return pick
end

# ---------------------------------------------------------- case geometry
# Boxes/sigmas from benchmark_033_common.jl and rotor_case_stats.csv.
struct CaseGeom
    box::NTuple{3,Float64}
    sigma_min::Float64
    sigma_max::Float64
end
geom(case, n) = begin
    if case == "cube"
        s = 2.0 * (1.0 / n)^(1 / 3)
        CaseGeom((1.0, 1.0, 1.0), s, s)
    elseif case == "wake"
        s = 2.0 * ((pi / 4) * 5.0 / n)^(1 / 3)
        CaseGeom((1.0, 1.0, 5.0), s, s)
    else # rotor (rotor_case_stats.csv)
        n == 100000 ?
            CaseGeom((0.23205, 0.20065, 1.20334), 1.7323e-4, 3.1081e-3) :
            CaseGeom((0.23205, 0.20022, 1.20214), 1.7323e-5, 3.1082e-4)
    end
end

# next 5-smooth integer >= m (A9)
function next_5smooth(m::Int)
    while true
        r = m
        for f in (2, 3, 5)
            while r % f == 0; r ÷= f; end
        end
        r == 1 && return m
        m += 1
    end
end

# ------------------------------------------------------------- band params
struct Band
    name::String
    bw_frac::Float64      # A1
    spread_rate::Float64  # A5 (particles/s at p=4)
    launch_ms::Float64    # A7
    err_thr::Float64      # A8
    extra_transforms::Int # A3 (pess: kernel re-tab)
end
const BANDS = [Band("optimistic", 0.70, 2e9, 0.5, 1.0e-3, 0),
               Band("nominal", 0.50, 8e8, 1.0, 7.0e-4, 0),
               Band("pessimistic", 0.30, 3e8, 2.0, 0.0, 3)]  # 0.0 => tightest measured
const PEAK_BW = 4.8e12  # A1, bytes/s

# ------------------------------------------------------------ VIC pricing
# Returns nothing if infeasible (mesh too large), else a NamedTuple.
function price_vic(case, n, tf, band, pick)
    gm = geom(case, n)
    bw = PEAK_BW * band.bw_frac
    bpc = tf == "Float32" ? 8 : 16        # bytes/complex (A2)
    fbytes(Np) = 3 * 2 * bpc * Np         # per transform (A2)
    best = nothing
    for p in (4, 6)
        hs = pick(p, band.err_thr)
        isnan(hs) && continue
        h = hs * gm.sigma_min             # sigma_min sets the mesh (Sec. 4)
        dims = ntuple(d -> next_5smooth(2 * (ceil(Int, gm.box[d] / h) + p)), 3) # A9
        Np = prod(dims)
        mem_gb = Np * (6 * (bpc)) / 1e9   # 3 omega + 3 U complex fields (K_hat extra at construction)
        Np > 3e10 && continue             # > H200 memory by any layout: infeasible
        nT = 6 + band.extra_transforms    # A3
        t_fft = nT * fbytes(Np) / bw * 1e3
        t_kmul = (tf == "Float32" ? 72 : 144) * Np / bw * 1e3   # A4
        srate = band.spread_rate * (4 / p)^3 * (tf == "Float32" ? 1.0 : 0.5) # A5,A10
        t_spread = n / srate * 1e3
        t_interp = n / (0.5 * srate) * 1e3                       # A6
        t = band.launch_ms + t_fft + t_kmul + t_spread + t_interp # A7; serial chain
        cand = (p=p, hs=hs, dims=dims, Np=Np, mem_gb=mem_gb, t_fft=t_fft,
                t_kmul=t_kmul, t_spread=t_spread, t_interp=t_interp,
                t_launch=band.launch_ms, total=t)
        (best === nothing || t < best.total) && (best = cand)
    end
    return best
end

# ------------------------------------------------- Ewald split (rotor, A11)
function price_ewald_rotor(n, tf, band, pick)
    gm = geom("rotor", n)
    bw = PEAK_BW * band.bw_frac
    bpc = tf == "Float32" ? 8 : 16
    # measured neighbor anchor at n=1e6 (A11); assume neighbor count within a
    # fixed metric radius scales linearly with n (same geometry, n-proportional
    # particle count along the same filament structures)
    nb_ref = 1600.0 * (n / 1e6); r_ref = 1.151e-2
    dexp = band.name == "optimistic" ? 1.7 : band.name == "nominal" ? 2.0 : 2.4
    prate = (band.name == "optimistic" ? 1e11 : band.name == "nominal" ? 7e10 : 4e10) /
            1.3 * (tf == "Float32" ? 1.0 : 0.5)
    p = 4
    hs = pick(p, band.err_thr)
    isnan(hs) && return nothing
    best = nothing
    for alpha in exp10.(range(log10(gm.sigma_max), log10(50 * gm.sigma_max); length=60))
        h = hs * alpha
        dims = ntuple(d -> next_5smooth(2 * (ceil(Int, gm.box[d] / h) + p)), 3)
        Np = prod(dims)
        Np > 3e10 && continue
        r_c = 3.5 * alpha
        # neighbors per particle within a fixed metric radius scale ~linearly
        # with n (same filament geometry, n-proportional density); anchored at
        # the measured n=1e6 point (A11):
        nb = nb_ref * (r_c / r_ref)^dexp
        pairs = n * nb
        t_pairs = pairs / prate * 1e3
        srate = band.spread_rate * (tf == "Float32" ? 1.0 : 0.5)
        nT = 6 + band.extra_transforms
        t_mesh = nT * 3 * 2 * bpc * Np / bw * 1e3 +
                 (tf == "Float32" ? 72 : 144) * Np / bw * 1e3 +
                 n / srate * 1e3 + n / (0.5 * srate) * 1e3
        t = band.launch_ms + t_mesh + t_pairs
        cand = (alpha=alpha, dims=dims, Np=Np, pairs=pairs, t_mesh=t_mesh,
                t_pairs=t_pairs, total=t)
        (best === nothing || t < best.total) && (best = cand)
    end
    return best
end

# ------------------------------------------------------------------ driver
anchors = load_anchors()
pick = load_hsigma()

open(joinpath(DATA, "fm037d_cost_tables.csv"), "w") do io
    println(io, "case,n,tf,band,p,h_over_sigma,mesh_dims,mesh_points,mem_gb," *
                "t_spread_ms,t_fft_ms,t_kmul_ms,t_interp_ms,t_launch_ms," *
                "t_total_ms,anchor_ms,speedup_vs_anchor,post038_ms,speedup_vs_post038,feasible")
    for case in ("cube", "wake", "rotor"), n in (100000, 1000000),
        tf in ("Float32", "Float64")
        anc = get(anchors, (case, n, tf), NaN)
        isnan(anc) && tf == "Float64" && (anc = NaN)  # some F64 anchors absent
        p38 = get(POST038, (case, n, tf), anc)
        for band in BANDS
            r = price_vic(case, n, tf, band, pick)
            if r === nothing
                gmx = geom(case, n)
                hsx = pick(4, band.err_thr)
                hx = hsx * gmx.sigma_min
                Npx = prod(ntuple(d -> 2 * ceil(Int, gmx.box[d] / hx), 3))
                @printf(io, "%s,%d,%s,%s,4,%.2f,,%e,%.1f,,,,,,,%.3f,,%.3f,,INFEASIBLE\n",
                        case, n, tf, band.name, hsx, Npx,
                        Npx * 6 * (tf == "Float32" ? 8 : 16) / 1e9, anc, p38)
            else
                @printf(io, "%s,%d,%s,%s,%d,%.2f,%dx%dx%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.2f,%.3f,%.2f,ok\n",
                        case, n, tf, band.name, r.p, r.hs, r.dims..., r.Np,
                        r.mem_gb, r.t_spread, r.t_fft, r.t_kmul, r.t_interp,
                        r.t_launch, r.total, anc, anc / r.total, p38, p38 / r.total)
            end
        end
    end
end

open(joinpath(DATA, "fm037d_ewald_rotor.csv"), "w") do io
    println(io, "case,n,tf,band,alpha_m,alpha_over_sigmamax,mesh_dims,mesh_points," *
                "pairs,t_mesh_ms,t_pairs_ms,t_total_ms,anchor_ms,post038_ms,speedup_vs_post038")
    for n in (100000, 1000000), tf in ("Float32", "Float64")
        anc = get(anchors, ("rotor", n, tf), NaN)
        p38 = get(POST038, ("rotor", n, tf), anc)
        gm = geom("rotor", n)
        for band in BANDS
            r = price_ewald_rotor(n, tf, band, pick)
            r === nothing && continue
            @printf(io, "rotor,%d,%s,%s,%.4e,%.2f,%dx%dx%d,%d,%.3e,%.3f,%.3f,%.3f,%.3f,%.3f,%.2f\n",
                    n, tf, band.name, r.alpha, r.alpha / gm.sigma_max, r.dims...,
                    r.Np, r.pairs, r.t_mesh, r.t_pairs,
                    band.launch_ms + r.t_mesh + r.t_pairs, anc, p38,
                    p38 / (band.launch_ms + r.t_mesh + r.t_pairs))
        end
    end
end
println("wrote fm037d_cost_tables.csv and fm037d_ewald_rotor.csv")
