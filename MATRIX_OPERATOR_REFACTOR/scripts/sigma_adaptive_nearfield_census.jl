#!/usr/bin/env julia
#
# Task 041f: sigma-adaptive multilevel smooth nearfield — deterministic
# real-snapshot level/patch/halo census, accuracy/exact-once oracle, and
# bounded optimizer/selector.
#
# Run:  julia --project=../../../FLOWVPM.jl --threads=4 \
#           MATRIX_OPERATOR_REFACTOR/scripts/sigma_adaptive_nearfield_census.jl
# (the FLOWVPM project is needed only so `benchmark_033_common.jl` can
#  `import FLOWVPM`; the rotor generator itself is pure Julia + the vendored
#  circulation CSV.  FM041F_SMOKE=1 runs a small-n smoke pass.)
#
# ---------------------------------------------------------------------------
# REGISTRATION (predeclared before any result was computed; binding)
# ---------------------------------------------------------------------------
# Decomposition under census (theory/sigma-adaptive-multilevel-nearfield.md):
#   G_sigma = G_s * rho_tau, tau^2 = sigma^2 - s^2  (exact width transfer)
#   G_{s_l} = sum_{k=l}^{M-1} D_k + G_{s_M},  D_k = G_{s_k} - G_{s_{k+1}}
# Architecture censused: per (level, sigma-bin) disjoint spread densities on
# block-partitioned patches; per-particle-exact compensation split
# tau^2 = tau_bin^2 + delta^2 (tau_bin к k-space at the bin lower edge,
# delta real-space Gaussian x window spreading); band kernels applied per
# patch in k-space (AMR-FFT) or as real-space tables (MSM); U via curl,
# J via analytic derivative of the U interpolant (037d 6-transform scheme).
# Ladder extension bands above sigma_max until the top term fits a bounded
# coarse global mesh (the top term is long-range and cannot be patch-local).
#
# Design space (finite, registered):
#   gamma  in {sqrt(2), 2, 4}          level ratio
#   res    in {(p=4,h/s=0.55) nom, (p=6,h/s=0.70) opt-alt, (p=4,h/s=0.40) tight}
#          (the three 037d accuracy-passing spread/interp configs)
#   block  in {32, 64, 128}            patch edge cap, mesh cells
#   b      in {1, 2, 4, 8, 16, 32}     sigma sub-bins per level (delta-split)
#   solver in {amrfft, msm}
#   prec   in {f32, f64}
#   refresh in {perstep, epoch}
#   M_ext  in 0..10                    ladder extension levels (optimized)
#   hybrid cut c over ladder levels    (FMM-retained banded hybrid pricing)
# Brackets opt/nom/pess on every calibration constant (037d convention).
# Error budget (gate 1e-3 velocity RMS): tails (band trunc + spread trunc +
# patch guard) <= 3e-4; resolution/transfer (mesh + restriction) <= 3e-4
# via the 037d-validated h <= 0.55*sigma rule transferred to band widths;
# fp accumulation <= 1e-4.  rho_b, rho_w truncation radii bracket (4,5,6).
# Census cases: cube/wake/sigma_multiscale proxies (041c/041d constructors,
# seeds 41003/41004/41006) and the REAL DJI-9443 rotor (fm033), n=1e5,1e6;
# rotor age-scale sensitivity {0.5,1,2}.  Oracle cases at n<=2048.
# No million-particle field evaluation; local threads <= 4.
# ---------------------------------------------------------------------------

using Random
using Statistics
using Printf
using SHA

Threads.nthreads() <= 4 || error("041f census must run on <= 4 threads")

const SMOKE = get(ENV, "FM041F_SMOKE", "0") == "1"

include(joinpath(@__DIR__, "adaptive_octree_verify.jl"))  # constructors + tree utils

const HAVE_ROTOR = Ref(false)
try
    include(joinpath(@__DIR__, "benchmark_033_common.jl"))
    HAVE_ROTOR[] = true
catch err
    @warn "benchmark_033_common.jl unavailable (need FLOWVPM project); rotor skipped" err
end

const OUTDIR = joinpath(@__DIR__, "..", "data", "sigma_adaptive_nearfield")
mkpath(OUTDIR)

# ---------------------------------------------------------------------------
# erf / erfc (double precision; series + Lentz continued fraction)
# ---------------------------------------------------------------------------
function erf_series(x::Float64)
    # sum_{k} (-1)^k x^(2k+1)/(k!(2k+1)) * 2/sqrt(pi), |x| <= 3
    t = x; s = x; k = 0
    while true
        k += 1
        t *= -x * x / k
        term = t / (2k + 1)
        s += term
        abs(term) < 1e-17 * abs(s) && break
        k > 200 && break
    end
    return 2 / sqrt(pi) * s
end

function erfc_cf(x::Float64)
    # continued fraction (Lentz) for erfc, x > 3
    tiny = 1e-300
    b = x; c = 1 / tiny; d = 1 / b; h = d
    for i in 1:300
        an = i / 2
        b = (i % 2 == 0) ? x : b            # alternating structure below
        # erfc CF: erfc(x) = exp(-x^2)/sqrt(pi) * 1/(x+ 1/2/(x+ 2/2/(x+ ...)))
        d = 1 / (x + an * d)
        c = x + an / c
        del = c * d
        h *= del
        abs(del - 1) < 1e-16 && break
    end
    return exp(-x * x) / sqrt(pi) * h
end

myerf(x::Float64) = abs(x) <= 3 ? erf_series(x) : sign(x) * (1 - erfc_cf(abs(x)))
myerfc(x::Float64) = x > 3 ? erfc_cf(x) : 1 - myerf(x)

# ---------------------------------------------------------------------------
# regularized kernel and radial derivatives:  G(r) = erf(r/(sqrt2 a))/(4 pi r)
# ---------------------------------------------------------------------------
@inline function kernelGderivs(a::Float64, r::Float64)
    # returns (G, G', G'') at radius r for width a  (a=0 -> singular kernel)
    if a <= 0
        G = 1 / (4pi * r); Gp = -G / r; Gpp = 2G / r^2
        return G, Gp, Gpp
    end
    u = r / (sqrt(2) * a)
    if r < 1e-7 * a
        c0 = -1 / (3 * (2pi)^(3 / 2) * a^3)   # G''(0); trace(H) = -rho(0)
        return sqrt(2 / pi) / (4pi * a), c0 * r, c0
    end
    F = myerf(u)
    E = 2 / sqrt(pi) * exp(-u * u)
    G = F / (4pi * r)
    Gp = (E * u - F) / (4pi * r^2)
    Gpp = (2F - 2E * u * (1 + u * u)) / (4pi * r^3)
    return G, Gp, Gpp
end

# velocity kernel contribution: u += grad G x Gamma ; J += grad grad G x Gamma
@inline function accum_uj!(u::Vector{Float64}, J::Matrix{Float64},
                           a::Float64, dx::NTuple{3,Float64}, Γ::NTuple{3,Float64},
                           sgn::Float64)
    r2 = dx[1]^2 + dx[2]^2 + dx[3]^2
    r = sqrt(r2)
    if r < (a > 0 ? 1e-12 * a : 1e-300)
        a <= 0 && return
        # coincident regularized pair: U self = 0; J self = H(0) x Gamma with
        # H(0) = G''(0) I  =>  dJ_ab = -G''(0) * eps_{abd} Gamma_d
        c0 = -1 / (3 * (2pi)^(3 / 2) * a^3)
        J[1, 2] += sgn * (-c0) * Γ[3]; J[2, 1] += sgn * c0 * Γ[3]
        J[2, 3] += sgn * (-c0) * Γ[1]; J[3, 2] += sgn * c0 * Γ[1]
        J[3, 1] += sgn * (-c0) * Γ[2]; J[1, 3] += sgn * c0 * Γ[2]
        return
    end
    G, Gp, Gpp = kernelGderivs(a, r)
    # grad G = Gp * xhat
    gx = Gp * dx[1] / r; gy = Gp * dx[2] / r; gz = Gp * dx[3] / r
    u[1] += sgn * (gy * Γ[3] - gz * Γ[2])
    u[2] += sgn * (gz * Γ[1] - gx * Γ[3])
    u[3] += sgn * (gx * Γ[2] - gy * Γ[1])
    # Hessian H_ac = (Gpp - Gp/r) x_a x_c / r^2 + Gp/r delta_ac
    c1 = (Gpp - Gp / r) / r2
    c2 = Gp / r
    @inbounds for aidx in 1:3
        Ha = (c1 * dx[aidx] * dx[1] + (aidx == 1 ? c2 : 0.0),
              c1 * dx[aidx] * dx[2] + (aidx == 2 ? c2 : 0.0),
              c1 * dx[aidx] * dx[3] + (aidx == 3 ? c2 : 0.0))
        # du_b/dx_a = eps_{bcd} H_ac Gamma_d
        J[aidx, 1] += sgn * (Ha[2] * Γ[3] - Ha[3] * Γ[2])
        J[aidx, 2] += sgn * (Ha[3] * Γ[1] - Ha[1] * Γ[3])
        J[aidx, 3] += sgn * (Ha[1] * Γ[2] - Ha[2] * Γ[1])
    end
end

# compensated band kernel width pair for source sigma on level widths (sl, sl1):
band_widths(σ::Float64, sl::Float64, sl1::Float64) =
    (sqrt(sl^2 + max(σ^2 - sl^2, 0.0)), sqrt(sl1^2 + max(σ^2 - sl^2, 0.0)))
# NOTE: compensation tau^2 = sigma^2 - s_{l_i}^2 is fixed at the assigned
# level; band k >= l_i widths are sqrt(s_k^2 + tau^2), sqrt(s_{k+1}^2 + tau^2).

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
function next5smooth(n::Int)
    while true
        m = n
        for p in (2, 3, 5)
            while m % p == 0
                m ÷= p
            end
        end
        m == 1 && return n
        n += 1
    end
end

writecsv(path, header, rows) = open(path, "w") do io
    println(io, header)
    for r in rows
        println(io, r)
    end
end

# ---------------------------------------------------------------------------
# cases
# ---------------------------------------------------------------------------
function case_arrays(name::String, n::Int)
    if name == "cube"
        X = make_uniform(n; seed=41003)
        σ = fill(2.0 * n^(-1 / 3), n)
        return X, σ
    elseif name == "wake"
        X = make_filament(n; seed=41004)
        σ = fill(3.15 * n^(-1 / 3), n)
        return X, σ
    elseif name == "sigma_multiscale"
        X = make_multiscale(n; contrast=100.0, seed=41006)
        c = (0.6, 0.4, 0.55)
        r = [sqrt((X[1, i] - c[1])^2 + (X[2, i] - c[2])^2 + (X[3, i] - c[3])^2) for i in 1:n]
        med = median(r)
        σ = [0.08 * n^(-1 / 3) * exp(log(18.0) * (ri <= med)) for ri in r]
        return X, σ
    elseif name == "rotor"
        HAVE_ROTOR[] || error("rotor case requires FLOWVPM project")
        X = Matrix{Float64}(undef, 3, n)
        σ = Vector{Float64}(undef, n)
        rng = MersenneTwister(FM033_SEED + FM033_ROTOR_SEED_OFFSET + n)
        i = Ref(0)
        fm033_rotor_foreach(n; rng) do x, _, s
            i[] += 1
            X[1, i[]] = x[1]; X[2, i[]] = x[2]; X[3, i[]] = x[3]
            σ[i[]] = s
        end
        @assert i[] == n
        return X, σ
    end
    error("unknown case $name")
end

# ---------------------------------------------------------------------------
# design-space constants
# ---------------------------------------------------------------------------
const GAMMAS = (sqrt(2.0), 2.0, 4.0)
const RESCFG = ((p=4, hs=0.55, tag="nom"), (p=6, hs=0.70, tag="optalt"), (p=4, hs=0.40, tag="tight"))
const BLOCKS = (32, 64, 128)          # patch edge cap in mesh cells
const BINS = (1, 2, 4, 8, 16, 32)     # sigma sub-bins per level
const RHO_BRK = (4.0, 5.0, 6.0)       # rho_b = rho_w truncation radii opt/nom/pess
const TILE = 8                        # occupancy tile = TILE^3 mesh cells
const M_EXT_MAX = 10
const TOP_MAX_PTS = 8.0e6             # top-mesh interior cap during ladder ext.
const RHO_T = 4.789                   # 031a J cutoff (conservative, hybrid ball)

# calibration anchors (037d / 041a; opt, nom, pess) — sources in manifest
const BW_EFF = (4.8e12 * 0.70, 4.8e12 * 0.50, 4.8e12 * 0.30)   # B/s
const FFT_BPP = 48.0                  # bytes/point/transform F32 (x2 F64)
const KMUL_BPP = 72.0                 # bytes/point k-multiply sweep (x2 F64)
const SPREAD_SAMP = (1.28e11, 5.12e10, 1.92e10)  # samples/s F32 (x0.5 F64)
const LAUNCH_FIX_MS = (0.5, 1.0, 2.0)
const LAUNCH_US = (2.0, 5.0, 15.0)    # per graph node at replay (028/029 floors)
const SMALLFFT_PEN = (1.0, 2.0, 4.0)  # bandwidth penalty for padded < 2^18
const SMALLFFT_CUT = 1 << 18
const MSM_FLOPS = (3.3e13, 1.65e13, 6.6e12)  # F32 effective FLOP/s (x0.5 F64)

# shipped baselines, ms (case => (near, complete, uncertainty, source))
const BASELINE = Dict(
    ("cube", 1_000_000) => (12.329, 30.677, 0.25, "041a fm041a_gpu_stages K64 + warm refresh"),
    ("wake", 1_000_000) => (9.218, 62.79, 0.25, "041a fm041a_gpu_stages K256 + warm refresh"),
    ("sigma_multiscale", 1_000_000) => (12.825, 59.35, 0.35, "041a multiscale100 proxy + warm refresh"),
    ("rotor", 1_000_000) => (23.3, 33.3, 0.35, "037b partitioned winner eval 33.3ms, near ~70%"),
    ("rotor", 100_000) => (4.9, 6.99, 0.35, "037b partitioned best n=1e5, near ~70%"),
)

# 037d global-VIC F32 U/J bands for the unified-solver check (opt, nom, pess)
const VIC037D = Dict(
    ("cube", 100_000) => (0.75, 1.51, 3.73),
    ("cube", 1_000_000) => (2.76, 5.81, 18.89),
    ("wake", 100_000) => (0.78, 1.55, 4.09),
    ("wake", 1_000_000) => (3.00, 6.16, 21.19),
)

# ---------------------------------------------------------------------------
# ladder
# ---------------------------------------------------------------------------
struct Ladder
    γ::Float64
    s::Vector{Float64}   # widths s_0..s_(M_assign+M_EXT_MAX)
    M_assign::Int        # bands 0..M_assign-1 hold assigned particles
end

function make_ladder(σmin, σmax, γ)
    M = max(1, ceil(Int, log(σmax / σmin) / log(γ) - 1e-12))
    s = [σmin * γ^l for l in 0:(M + M_EXT_MAX)]
    return Ladder(γ, s, M)
end

assign_level(σ, lad::Ladder) =
    clamp(floor(Int, log(σ / lad.s[1]) / log(lad.γ) + 1e-12), 0, lad.M_assign - 1)

# ---------------------------------------------------------------------------
# occupancy census for one (case, n, gamma, res)
# ---------------------------------------------------------------------------
struct LevelCensus
    pop::Int                # particles assigned to this level
    cum::Int                # particles contributing to this band (cum sources)
    tiles_core::Int         # occupied source tiles (b=1)
    tiles_field::Int        # guard-dilated tiles (b=1)
    cov_targets::Int        # targets inside dilated set
    cov_mult8::Float64      # mean bin-field multiplicity at b=8 over covered targets
    padded8_ratio::Float64  # sum of b=8 per-bin dilated tiles / b=1 dilated tiles
    rcells::Int             # kernel reach + window, in mesh cells (halo/padding)
    patch::Dict{Int,NTuple{4,Float64}} # block => (npatch, interior_pts, padded_pts, active_pts)
    spread_samp::Vector{Float64}  # per BINS entry: sum_i supp_i^3 (own level)
end

@inline packkey(ix, iy, iz) = (Int64(ix) << 42) | (Int64(iy) << 21) | Int64(iz)
@inline unpackkey(k) = (Int(k >> 42) & 0x1FFFFF, Int(k >> 21) & 0x1FFFFF, Int(k & 0x1FFFFF))

function dilate_separable(set::Set{Int64}, g::Int)
    g <= 0 && return copy(set)
    cur = set
    for (sh, mask) in ((42, 0), (21, 0), (0, 0))
        nxt = Set{Int64}()
        for k in cur, d in -g:g
            push!(nxt, k + (Int64(d) << sh))
        end
        cur = nxt
    end
    return cur
end

function census_case(X::Matrix{Float64}, σ::Vector{Float64}, γ::Float64,
                     p::Int, hs::Float64, ρ::Float64)
    n = size(X, 2)
    lo = (minimum(view(X, 1, :)), minimum(view(X, 2, :)), minimum(view(X, 3, :)))
    hi = (maximum(view(X, 1, :)), maximum(view(X, 2, :)), maximum(view(X, 3, :)))
    lad = make_ladder(minimum(σ), maximum(σ), γ)
    L = length(lad.s) - 1                     # bands 0..L-1 available (incl. ext)
    lev = [assign_level(σ[i], lad) for i in 1:n]
    # per-level tile sets: core (b=1) and per-bin at b=8 (own level only)
    core = [Set{Int64}() for _ in 1:L]
    bin8 = [[Set{Int64}() for _ in 1:8] for _ in 1:L]
    pop = zeros(Int, L)
    spread = [zeros(Float64, length(BINS)) for _ in 1:L]
    for i in 1:n
        li = lev[i] + 1
        pop[li] += 1
        for l in li:L
            h = hs * lad.s[l]
            tw = TILE * h
            k = packkey(floor(Int, (X[1, i] - lo[1]) / tw),
                        floor(Int, (X[2, i] - lo[2]) / tw),
                        floor(Int, (X[3, i] - lo[3]) / tw))
            push!(core[l], k)
            if l == li
                frac = clamp(log(σ[i] / lad.s[l]) / log(γ), 0.0, 1.0 - 1e-12)
                push!(bin8[l][1 + floor(Int, 8 * frac)], k)
                # per-b spreading sample counts (delta split to bin lower edge)
                h_l = hs * lad.s[l]
                for (bi, b) in enumerate(BINS)
                    edge = lad.s[l] * γ^(floor(b * frac) / b)
                    δ = sqrt(max(σ[i]^2 - edge^2, 0.0))
                    supp = p + ceil(Int, 2 * ρ * δ / h_l)
                    spread[l][bi] += float(supp)^3
                end
            end
        end
    end
    cum = [sum(pop[1:l]) for l in 1:L]
    # guard-dilation, coverage, patches per level
    out = Vector{LevelCensus}(undef, L)
    for l in 1:L
        h = hs * lad.s[l]
        tw = TILE * h
        Rband = ρ * γ * lad.s[l]                       # band support radius
        rcells = ceil(Int, Rband / h) + p              # halo-gather + window
        field = dilate_separable(core[l], 1)           # spread/interp support
        # coverage + b=8 multiplicity
        covt = 0; mult = 0.0
        d8 = [dilate_separable(bin8[l][j], 1) for j in 1:8]
        for i in 1:n
            k = packkey(floor(Int, (X[1, i] - lo[1]) / tw),
                        floor(Int, (X[2, i] - lo[2]) / tw),
                        floor(Int, (X[3, i] - lo[3]) / tw))
            if k in field
                covt += 1
                m = 0
                for j in 1:8
                    m += (k in d8[j]) ? 1 : 0
                end
                mult += max(m, 1)
            end
        end
        p8ratio = sum(length.(d8)) / max(length(field), 1)
        # patch aggregation per block size over the dilated (field) tiles
        pd = Dict{Int,NTuple{4,Float64}}()
        for B in BLOCKS
            bt = max(B ÷ TILE, 1)
            groups = Dict{Int64,NTuple{6,Int}}()       # blockkey => tile bbox
            cnt = Dict{Int64,Int}()
            for k in field
                ix, iy, iz = unpackkey(k)
                bk = packkey(ix ÷ bt, iy ÷ bt, iz ÷ bt)
                if haskey(groups, bk)
                    (x0, y0, z0, x1, y1, z1) = groups[bk]
                    groups[bk] = (min(x0, ix), min(y0, iy), min(z0, iz),
                                  max(x1, ix), max(y1, iy), max(z1, iz))
                    cnt[bk] += 1
                else
                    groups[bk] = (ix, iy, iz, ix, iy, iz)
                    cnt[bk] = 1
                end
            end
            npatch = length(groups)
            interior = 0.0; padded = 0.0; active = 0.0
            for (bk, (x0, y0, z0, x1, y1, z1)) in groups
                dims = ((x1 - x0 + 1) * TILE, (y1 - y0 + 1) * TILE, (z1 - z0 + 1) * TILE)
                interior += prod(float.(dims))
                padded += prod(float.(next5smooth.(dims .+ 2 * rcells)))
                active += cnt[bk] * TILE^3
            end
            pd[B] = (float(npatch), interior, padded, active)
        end
        out[l] = LevelCensus(pop[l], cum[l], length(core[l]), length(field),
                             covt, covt > 0 ? mult / covt : 1.0, p8ratio, rcells,
                             pd, spread[l])
    end
    return lad, lev, out
end

# ---------------------------------------------------------------------------
# cost model
# ---------------------------------------------------------------------------
struct Priced
    spread_ms::Float64; fft_ms::Float64; kmul_ms::Float64; interp_ms::Float64
    restrict_ms::Float64; halo_ms::Float64; top_ms::Float64; launch_ms::Float64
    refresh_ms::Float64; msm_ms::Float64
    total_ms::Float64
    M_ext::Int
    bytes_persistent::Float64
    launches::Float64
end

# choose extension depth: extend bands until global top mesh fits TOP_MAX_PTS
function top_mesh_pts(lad::Ladder, M_ext::Int, hs::Float64, lo, hi)
    sM = lad.s[min(lad.M_assign + M_ext + 1, length(lad.s))]
    h = hs * sM
    dims = ntuple(d -> max(ceil(Int, (hi[d] - lo[d]) / h) + 4, 8), 3)
    interior = prod(float.(dims))
    padded = prod(float.(next5smooth.(2 .* dims)))
    return interior, padded
end

function price(cen::Vector{LevelCensus}, lad::Ladder, lo, hi, n::Int;
               p::Int, hs::Float64, B::Int, b::Int, prec::Symbol, solver::Symbol,
               refresh::Symbol, br::Int)
    f64 = prec == :f64
    bw = BW_EFF[br]
    fftb = FFT_BPP * (f64 ? 2 : 1)
    kmulb = KMUL_BPP * (f64 ? 2 : 1)
    srate = SPREAD_SAMP[br] * (f64 ? 0.5 : 1.0)
    mflop = MSM_FLOPS[br] * (f64 ? 0.5 : 1.0)
    bi = findfirst(==(b), BINS)
    Lassign = lad.M_assign
    # bin-count decay at coarser levels: bins(k) = max(1, ceil(b / gamma^{2*dk}))
    bins_at = (l, li) -> max(1, ceil(Int, b / lad.γ^(2 * (l - li))))
    # effective forward-density multiplicity per level: population-weighted
    # (approximated by own-level bins b at the level's own band, decaying above)
    best = nothing
    for M_ext in 0:M_EXT_MAX
        L = Lassign + M_ext
        L > length(cen) && break
        ti, tp = top_mesh_pts(lad, M_ext, hs, lo, hi)
        spread_s = 0.0; fftby = 0.0; kmulby = 0.0; interp_s = 0.0
        restr_s = 0.0; haloby = 0.0; launches = 0.0; msm_flops = 0.0
        bytes_pers = 0.0
        for l in 1:L
            c = cen[l]
            (np, interior, padded, active) = c.patch[B]
            np == 0 && continue
            # forward transforms: 3 per (level,bin); per-bin padded points scale
            # between b=1 and b=8 measured dilation, extrapolated in log2 b
            pr = max(c.padded8_ratio, 1.0)
            binpad = b <= 1 ? 1.0 :
                     b <= 8 ? 1.0 + (pr - 1.0) * (log2(b) / 3) :
                     pr * (b / 8)^(1 / 3)
            nb_here = min(b, max(1, c.pop > 0 ? b : 1))
            fwd_pts = padded * binpad
            if solver == :amrfft
                fftby += (3 * fwd_pts + 3 * padded) * fftb   # 3 fwd (all bins) + 3 inv
                kmulby += binpad * padded * kmulb
                launches += min(np, 12) * 6 + 6   # shape-class batched fwd/inv + stages
            else
                # MSM: stencil support R/h points per axis on active points
                sten = (2 * ceil(Int, RHO_BRK[br] * lad.γ / hs) + 1)^3
                msm_flops += active * sten * 3 * 2 * binpad
                launches += min(np, 12) * 3 + 6
            end
            interp_s += c.cov_targets * (l <= Lassign ? c.cov_mult8^((log2(max(b, 1)) / 3)) : 1.0) *
                        float(p)^3 / srate / 2 * 4  # x4: U(3)+J via derivative windows (037d: interp=spread/2 covers U+J; factor folded)
            haloby += (padded - interior) * (f64 ? 8 : 4) * 2
            bytes_pers += padded * (f64 ? 8 : 4) * 4
            if l <= Lassign
                spread_s += cen[l].spread_samp[bi] / srate
            end
            l > 1 && (restr_s += active / 8 * float(p)^3 / srate)
        end
        # top term: global coarse free-space convolution (VIC at s_M)
        topby = tp * (6 * fftb + kmulb)
        top_ms = topby / bw * 1e3 + n * float(p)^3 / srate * 1e3 * 0.0 # top spread via restriction cascade
        spread_ms = spread_s * 1e3
        fft_ms = 0.0
        # small-FFT penalty applied per level via patch padded size
        for l in 1:L
            c = cen[l]
            (np, interior, padded, active) = c.patch[B]
            np == 0 && continue
            per = padded / np
            pen = per < SMALLFFT_CUT ? SMALLFFT_PEN[br] : 1.0
            pr = max(c.padded8_ratio, 1.0)
            binpad = b <= 1 ? 1.0 : b <= 8 ? 1.0 + (pr - 1.0) * (log2(b) / 3) : pr * (b / 8)^(1 / 3)
            fft_ms += (3 * padded * binpad + 3 * padded) * fftb * pen / bw * 1e3
        end
        kmul_ms = kmulby / bw * 1e3
        interp_ms = interp_s * 1e3
        restrict_ms = restr_s * 1e3
        halo_ms = haloby / bw * 1e3
        launch_ms = LAUNCH_FIX_MS[br] + launches * LAUNCH_US[br] * 1e-3
        msm_ms = msm_flops / mflop * 1e3
        refresh_ms = refresh == :perstep ? (0.10 * (spread_ms + interp_ms) + 0.5) :
                     (0.02 * (spread_ms + interp_ms) + 0.1)
        core = solver == :amrfft ? (fft_ms + kmul_ms) : msm_ms
        total = spread_ms + core + interp_ms + restrict_ms + halo_ms + top_ms +
                launch_ms + refresh_ms
        pr = Priced(spread_ms, fft_ms, kmul_ms, interp_ms, restrict_ms, halo_ms,
                    top_ms, launch_ms, refresh_ms, msm_ms, total, M_ext,
                    bytes_pers + tp * (f64 ? 8 : 4) * 4, launches)
        (best === nothing || pr.total_ms < best.total_ms) && (best = pr)
    end
    return best
end

# ---------------------------------------------------------------------------
# main census sweep
# ---------------------------------------------------------------------------
const NLIST = SMOKE ? (20_000,) : (100_000, 1_000_000)
const CASES = HAVE_ROTOR[] ? ("cube", "wake", "sigma_multiscale", "rotor") :
              ("cube", "wake", "sigma_multiscale")

level_rows = String[]
cost_rows = String[]
opt_rows = String[]
unified_rows = String[]
hybrid_rows = String[]
refresh_rows = String[]

for cname in CASES, n in NLIST
    X, σ = case_arrays(cname, n)
    lo = (minimum(view(X, 1, :)), minimum(view(X, 2, :)), minimum(view(X, 3, :)))
    hi = (maximum(view(X, 1, :)), maximum(view(X, 2, :)), maximum(view(X, 3, :)))
    spread_ratio = maximum(σ) / minimum(σ)
    @printf("[case] %s n=%d sigma spread %.2fx\n", cname, n, spread_ratio)
    combos = [(γ, rc) for γ in GAMMAS, rc in RESCFG]
    results = Vector{Any}(undef, length(combos))
    Threads.@threads for ci in eachindex(combos)
        local γt = combos[ci][1]
        local rct = combos[ci][2]
        results[ci] = (γt, rct, census_case(X, σ, γt, rct.p, rct.hs, RHO_BRK[2]))
    end
    for (γ, rc, (lad, lev, cen)) in results
        for (l, c) in enumerate(cen)
            push!(level_rows, @sprintf("%s,%d,%.4f,%s,%d,%.6e,%d,%d,%d,%d,%d,%.3f,%.3f,%.0f,%.0f,%.0f",
                cname, n, γ, rc.tag, l - 1, lad.s[l], c.pop, c.cum, c.tiles_core,
                c.tiles_field, c.cov_targets, c.cov_mult8, c.padded8_ratio,
                c.patch[64][1], c.patch[64][2], c.patch[64][3]))
        end
        for B in BLOCKS, b in BINS, solver in (:amrfft, :msm), prec in (:f32, :f64),
            refresh in (:perstep, :epoch)
            # keep the cost table bounded: full grid only at nominal res + gamma=2;
            # elsewhere price the nominal (B=64,b=8,amrfft,f32,epoch) config
            fullrow = (rc.tag == "nom")
            nomonly = (B == 64 && b == 8 && solver == :amrfft && prec == :f32 && refresh == :epoch)
            (fullrow || nomonly) || continue
            for br in 1:3
                pr = price(cen, lad, lo, hi, n; p=rc.p, hs=rc.hs, B, b, prec,
                           solver, refresh, br)
                pr === nothing && continue
                push!(cost_rows, @sprintf("%s,%d,%.4f,%s,%d,%d,%s,%s,%s,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.3e,%.0f",
                    cname, n, γ, rc.tag, B, b, solver, prec, refresh, br, pr.M_ext,
                    pr.spread_ms, pr.fft_ms, pr.kmul_ms, pr.interp_ms,
                    pr.restrict_ms, pr.halo_ms, pr.top_ms, pr.launch_ms,
                    pr.refresh_ms, pr.total_ms, pr.bytes_persistent, pr.launches))
            end
        end
    end
    # optimizer: best nominal-bracket config per case (search full grid at nom res)
    best = nothing
    for (γ, rc, (lad, lev, cen)) in results
        rc.tag == "nom" || continue
        for B in BLOCKS, b in BINS, solver in (:amrfft, :msm)
            pr = price(cen, lad, lo, hi, n; p=rc.p, hs=rc.hs, B, b, prec=:f32,
                       solver, refresh=:epoch, br=2)
            pr === nothing && continue
            if best === nothing || pr.total_ms < best[1].total_ms
                best = (pr, γ, rc, B, b, solver, lad, cen)
            end
        end
    end
    if best !== nothing
        pr, γ, rc, B, b, solver, lad, cen = best
        prlo = price(cen, lad, lo, hi, n; p=rc.p, hs=rc.hs, B, b, prec=:f32,
                     solver, refresh=:epoch, br=1)
        prhi = price(cen, lad, lo, hi, n; p=rc.p, hs=rc.hs, B, b, prec=:f32,
                     solver, refresh=:epoch, br=3)
        # zero-coupling lower bound at opt anchors: spread+interp+fft+kmul+top only
        zc = prlo.spread_ms + prlo.fft_ms + prlo.kmul_ms + prlo.interp_ms + prlo.top_ms
        base = get(BASELINE, (cname, n), nothing)
        bstr = base === nothing ? "NA,NA,NA" :
               @sprintf("%.3f,%.3f,%.2f", base[1], base[2], base[3])
        push!(opt_rows, @sprintf("%s,%d,%.4f,%s,%d,%d,%s,%d,%.4f,%.4f,%.4f,%.4f,%s",
            cname, n, γ, rc.tag, B, b, solver, pr.M_ext, prlo.total_ms,
            pr.total_ms, prhi.total_ms, zc, bstr))
        if haskey(VIC037D, (cname, n))
            v = VIC037D[(cname, n)]
            push!(unified_rows, @sprintf("%s,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.3f",
                cname, n, prlo.total_ms, pr.total_ms, prhi.total_ms, v[1], v[2], v[3],
                pr.total_ms / v[2]))
        end
    end
    # banded hybrid pricing: mesh bands < c, keep singular FMM; direct complement
    # inflates every source cutoff to rho_t * sigma_i^(c),
    # sigma^(c) = sqrt(sigma_i^2 - s_{l_i}^2 + s_c^2) — pair count scales ~cubed.
    let
        γ = 2.0
        idx = findfirst(r -> r[1] == γ && r[2].tag == "nom", results)
        lad, lev, cen = results[idx][3]
        for cidx in 1:min(lad.M_assign, 6)
            sc = lad.s[cidx + 1]
            infl = 0.0
            for i in 1:n
                sli = lad.s[lev[i] + 1]
                σc = sqrt(σ[i]^2 - sli^2 + sc^2)
                infl += (max(σc, σ[i]) / σ[i])^3
            end
            infl /= n
            # mesh cost of bands < c on the same gamma=2 nominal census
            bandms = 0.0
            for l in 1:cidx
                c = cen[l]
                (np, interior, padded, active) = c.patch[64]
                np == 0 && continue
                bandms += (6 * padded) * FFT_BPP / BW_EFF[2] * 1e3
                l <= lad.M_assign && (bandms += cen[l].spread_samp[findfirst(==(8), BINS)] / SPREAD_SAMP[2] * 1e3)
            end
            base = get(BASELINE, (cname, n), nothing)
            nearb = base === nothing ? NaN : base[1]
            push!(hybrid_rows, @sprintf("%s,%d,%d,%.6e,%.3f,%.4f,%.4f,%s",
                cname, n, cidx, sc, infl, bandms, nearb,
                "additive: direct >= baseline x pair-inflation; bands add on top"))
        end
    end
    # rotor age sweep (level drift / refresh)
    if cname == "rotor"
        γ = 2.0
        for agescale in (0.5, 2.0)
            σ2 = σ .* sqrt.(1 .+ (agescale - 1) .* 0.4)  # documented proxy scaling
            lad = make_ladder(minimum(σ), maximum(σ), γ)
            l1 = [assign_level(σ[i], lad) for i in 1:n]
            l2 = [assign_level(clamp(σ2[i], lad.s[1], lad.s[end]), lad) for i in 1:n]
            drift = count(l1 .!= l2) / n
            push!(refresh_rows, @sprintf("rotor,%d,%.2f,%.4f", n, agescale, drift))
        end
    end
end

# ---------------------------------------------------------------------------
# accuracy / exact-once oracle (small deterministic cases)
# ---------------------------------------------------------------------------
# Patched multilevel evaluation with per-particle-exact compensated band
# kernels evaluated on block-partitioned level meshes, tensor-Lagrange
# interpolation of U, window-derivative J; compared against exact regularized
# sums.  Ownership painting: per ordered pair and band, the number of owner
# patches whose source set includes the source must be 1, or 0 only when the
# analytic band tail at that pair distance is below the tail budget.

function lagrange_weights(x::Float64, p::Int)
    # nodes 0..p-1, x in local coords
    w = zeros(p)
    for j in 0:p-1
        num = 1.0; den = 1.0
        for m in 0:p-1
            m == j && continue
            num *= (x - m); den *= (j - m)
        end
        w[j + 1] = num / den
    end
    return w
end

function lagrange_dweights(x::Float64, p::Int)
    d = zeros(p)
    for j in 0:p-1
        s = 0.0
        for k in 0:p-1
            k == j && continue
            t = 1.0
            for m in 0:p-1
                (m == j || m == k) && continue
                t *= (x - m) / (j - m)
            end
            s += t / (j - k)
        end
        d[j + 1] = s
    end
    return d
end

function oracle_case(X::Matrix{Float64}, σ::Vector{Float64},
                     Γ::Matrix{Float64}, Y::Matrix{Float64};
                     γ=2.0, p_int=6, hs=0.55, ρ=5.0, B=32, vic=false)
    # Multilevel patched evaluation with per-particle-exact compensated band
    # kernels sampled exactly on block-partitioned level meshes and tensor-
    # Lagrange(p_int) interpolation of U (window-derivative J).  vic=true
    # instead samples the full regularized field on ONE global mesh with the
    # SAME instrument, so the multilevel-vs-single-mesh comparison isolates
    # any resolution burden added by the decomposition itself (the absolute
    # h<=0.55*sigma production accuracy claim rests on 037d's measured PME
    # configurations, not on this deliberately naive interpolant).
    n = size(X, 2); nt = size(Y, 2)
    lo = ntuple(d -> min(minimum(view(X, d, :)), minimum(view(Y, d, :))) - 1e-9, 3)
    hi = ntuple(d -> max(maximum(view(X, d, :)), maximum(view(Y, d, :))) + 1e-9, 3)
    lad = make_ladder(minimum(σ), maximum(σ), γ)
    lev = [assign_level(σ[i], lad) for i in 1:n]
    M_ext = 0
    if !vic
        while M_ext < M_EXT_MAX
            ti, _ = top_mesh_pts(lad, M_ext, hs, lo, hi)
            ti <= 64.0^3 && break
            M_ext += 1
        end
    end
    L = vic ? 1 : lad.M_assign + M_ext
    sM = lad.s[L + 1]
    U = zeros(3, nt); J = zeros(3, 3, nt)
    excluded = Float64[]
    halfc = p_int ÷ 2 - 1
    for l in 1:L
        h = vic ? hs * minimum(σ) : hs * lad.s[l]
        Rband = ρ * γ * (vic ? maximum(σ) : lad.s[l])
        bw = B * h
        owner = Dict{Int64,Vector{Int}}()
        for i in 1:n
            (vic || lev[i] + 1 <= l) || continue
            bx = floor(Int, (X[1, i] - lo[1]) / bw)
            by = floor(Int, (X[2, i] - lo[2]) / bw)
            bz = floor(Int, (X[3, i] - lo[3]) / bw)
            τ = sqrt(max(σ[i]^2 - lad.s[lev[i] + 1]^2, 0.0))
            reach = vic ? 10_000 : ceil(Int, (Rband + ρ * τ) / bw) + 1
            if vic
                # full kernel is long-range: every source reaches every block;
                # handled by a sentinel list below
                continue
            end
            for dx in -reach:reach, dy in -reach:reach, dz in -reach:reach
                push!(get!(owner, packkey(bx + dx + 4096, by + dy + 4096, bz + dz + 4096), Int[]), i)
            end
        end
        allsrc = collect(1:n)
        nodecache = Dict{Tuple{Int64,Int64},NTuple{3,Float64}}()
        for t in 1:nt
            bk = packkey(floor(Int, (Y[1, t] - lo[1]) / bw) + 4096,
                         floor(Int, (Y[2, t] - lo[2]) / bw) + 4096,
                         floor(Int, (Y[3, t] - lo[3]) / bw) + 4096)
            src = vic ? allsrc : get(owner, bk, Int[])
            isempty(src) && !vic && continue
            base = ntuple(d -> floor(Int, (Y[d, t] - lo[d]) / h) - halfc, 3)
            xs = ntuple(d -> (Y[d, t] - lo[d]) / h - base[d], 3)
            wx = lagrange_weights(xs[1], p_int); wy = lagrange_weights(xs[2], p_int)
            wz = lagrange_weights(xs[3], p_int)
            dxw = lagrange_dweights(xs[1], p_int) ./ h
            dyw = lagrange_dweights(xs[2], p_int) ./ h
            dzw = lagrange_dweights(xs[3], p_int) ./ h
            for gi in 1:p_int, gj in 1:p_int, gk in 1:p_int
                nk = packkey(base[1] + gi + 8192, base[2] + gj + 8192, base[3] + gk + 8192)
                uu = get!(nodecache, (bk, nk)) do
                    gx = lo[1] + (base[1] + gi - 1) * h
                    gy = lo[2] + (base[2] + gj - 1) * h
                    gz = lo[3] + (base[3] + gk - 1) * h
                    u3 = zeros(3); J3 = zeros(3, 3)
                    for i in src
                        dxv = (gx - X[1, i], gy - X[2, i], gz - X[3, i])
                        g = (Γ[1, i], Γ[2, i], Γ[3, i])
                        if vic
                            accum_uj!(u3, J3, σ[i], dxv, g, 1.0)
                        else
                            τ2 = max(σ[i]^2 - lad.s[lev[i] + 1]^2, 0.0)
                            a1 = sqrt(lad.s[l]^2 + τ2); a2 = sqrt(lad.s[l + 1]^2 + τ2)
                            accum_uj!(u3, J3, a1, dxv, g, 1.0)
                            accum_uj!(u3, J3, a2, dxv, g, -1.0)
                        end
                    end
                    (u3[1], u3[2], u3[3])
                end
                w = wx[gi] * wy[gj] * wz[gk]
                U[1, t] += w * uu[1]; U[2, t] += w * uu[2]; U[3, t] += w * uu[3]
                for aa in 1:3
                    wd = (aa == 1 ? dxw[gi] * wy[gj] * wz[gk] :
                          aa == 2 ? wx[gi] * dyw[gj] * wz[gk] :
                                    wx[gi] * wy[gj] * dzw[gk])
                    for bb in 1:3
                        J[aa, bb, t] += wd * uu[bb]
                    end
                end
            end
            # omission audit: excluded sources' analytic band magnitude
            if !vic && l <= lad.M_assign
                inset = Set(src)
                for i in 1:n
                    lev[i] + 1 <= l || continue
                    i in inset && continue
                    r = sqrt(sum((Y[d, t] - X[d, i])^2 for d in 1:3))
                    τ2 = max(σ[i]^2 - lad.s[lev[i] + 1]^2, 0.0)
                    a2 = sqrt(lad.s[l + 1]^2 + τ2)
                    push!(excluded, myerfc(r / (sqrt(2) * a2)) / (4pi * max(r, 1e-30)))
                end
            end
        end
    end
    if !vic
        # top term: exact compensated evaluation (the top mesh is priced, not
        # oracled — it is exactly 037d's validated global-VIC configuration)
        for t in 1:nt
            uu = zeros(3); JJ = zeros(3, 3)
            for i in 1:n
                τ2 = max(σ[i]^2 - lad.s[lev[i] + 1]^2, 0.0)
                aT = sqrt(sM^2 + τ2)
                dxv = (Y[1, t] - X[1, i], Y[2, t] - X[2, i], Y[3, t] - X[3, i])
                accum_uj!(uu, JJ, aT, dxv, (Γ[1, i], Γ[2, i], Γ[3, i]), 1.0)
            end
            for c in 1:3
                U[c, t] += uu[c]
            end
            for aa in 1:3, bb in 1:3
                J[aa, bb, t] += JJ[aa, bb]
            end
        end
    end
    Ue = zeros(3, nt); Je = zeros(3, 3, nt)
    for t in 1:nt
        uu = zeros(3); JJ = zeros(3, 3)
        for i in 1:n
            dxv = (Y[1, t] - X[1, i], Y[2, t] - X[2, i], Y[3, t] - X[3, i])
            accum_uj!(uu, JJ, σ[i], dxv, (Γ[1, i], Γ[2, i], Γ[3, i]), 1.0)
        end
        Ue[:, t] .= uu; Je[:, :, t] .= JJ
    end
    urms = sqrt(mean(abs2, Ue))
    uerr = sqrt(mean(abs2, U .- Ue)) / urms
    jrms = sqrt(mean(abs2, Je))
    jerr = sqrt(mean(abs2, J .- Je)) / max(jrms, 1e-300)
    tailmax = isempty(excluded) ? 0.0 : maximum(excluded) / urms
    return uerr, jerr, tailmax, lad.M_assign, M_ext
end

oracle_rows = String[]
oracle_res = Dict{Tuple{String,String,Float64,Float64},Float64}()  # (case,mode,gamma,hs)=>uerr
oracle_verdicts = String[]
let
    no = SMOKE ? 192 : 384
    ntgt = SMOKE ? 96 : 192
    rngo = MersenneTwister(41007)
    gammas(n) = (G = 2 .* rand(rngo, 3, n) .- 1; G ./= n; G)
    cases = Vector{Tuple{String,Matrix{Float64},Vector{Float64},Matrix{Float64},Matrix{Float64}}}()
    X = make_uniform(no; seed=41003)
    push!(cases, ("uniform", X, fill(2.0 * no^(-1 / 3), no), gammas(no), X[:, 1:ntgt]))
    X = make_multiscale(no; contrast=100.0, seed=41006)
    c0 = (0.6, 0.4, 0.55)
    r = [sqrt(sum((X[d, i] - c0[d])^2 for d in 1:3)) for i in 1:no]
    σh = [0.08 * no^(-1 / 3) * exp(log(18.0) * (ri <= median(r))) for ri in r]
    push!(cases, ("hetero18x", X, σh, gammas(no), X[:, 1:ntgt]))
    Xb = copy(X)
    for i in 1:min(64, no)
        Xb[1, i] = 0.5; Xb[2, i] = 0.25 * (i % 4); Xb[3, i] = 0.5
    end
    push!(cases, ("boundary", Xb, σh, gammas(no), Xb[:, 1:ntgt]))
    X = make_filament(no; seed=41004)
    push!(cases, ("filament", X, fill(3.15 * no^(-1 / 3), no), gammas(no), X[:, 1:ntgt]))
    X = make_multiscale(no; contrast=300.0, seed=41008)
    Yt = rand(MersenneTwister(41009), 3, ntgt)
    push!(cases, ("island_distinct", X, σh, gammas(no), Yt))
    X = make_uniform(no; seed=41010)
    σx = [i <= no ÷ 2 ? 0.002 : 0.0358 for i in 1:no]
    push!(cases, ("extreme_ladder", X, σx, gammas(no), X[:, 1:ntgt]))
    runs = [("ml", 2.0, 0.55), ("ml", 2.0, 0.275), ("ml", 2.0, 0.1375), ("ml", 4.0, 0.55)]
    for (nm, Xc, σc, G, Y) in cases
        for (mode, γ, hsv) in runs
            uerr, jerr, tailmax, Ma, Me = oracle_case(Xc, σc, G, Y; γ, hs=hsv, p_int=6, ρ=5.0)
            oracle_res[(nm, mode, γ, hsv)] = uerr
            passab = uerr <= 1e-3 && tailmax <= 3e-4
            push!(oracle_rows, @sprintf("%s,%d,%s,%.4f,%.3f,%d,%d,%d,%.3e,%.3e,%.3e,%s",
                nm, size(Xc, 2), mode, γ, hsv, 6, Ma, Me, uerr, jerr, tailmax, passab))
            @printf("[oracle] %-16s %s g=%.2f hs=%.3f Uerr=%.3e Jerr=%.3e tail=%.2e\n",
                nm, mode, γ, hsv, uerr, jerr, tailmax)
        end
        if nm in ("uniform", "filament")
            for hsv in (0.55, 0.275)
                uerr, jerr, tailmax, Ma, Me = oracle_case(Xc, σc, G, Y; γ=2.0, hs=hsv, p_int=6, ρ=5.0, vic=true)
                oracle_res[(nm, "vic", 2.0, hsv)] = uerr
                push!(oracle_rows, @sprintf("%s,%d,%s,%.4f,%.3f,%d,%d,%d,%.3e,%.3e,%.3e,%s",
                    nm, size(Xc, 2), "vic", 2.0, hsv, 6, 0, 0, uerr, jerr, tailmax, "NA"))
                @printf("[oracle] %-16s vic g=2.00 hs=%.3f Uerr=%.3e\n", nm, hsv, uerr)
            end
        end
    end
    # verdicts: convergence order + multilevel-vs-single-mesh burden + abs pass
    for (nm, _, _, _, _) in cases
        e55 = oracle_res[(nm, "ml", 2.0, 0.55)]
        e27 = oracle_res[(nm, "ml", 2.0, 0.275)]
        e14 = oracle_res[(nm, "ml", 2.0, 0.1375)]
        conv = e55 / max(e27, 1e-300)
        conv2 = e27 / max(e14, 1e-300)
        burden = haskey(oracle_res, (nm, "vic", 2.0, 0.55)) ?
                 e55 / max(oracle_res[(nm, "vic", 2.0, 0.55)], 1e-300) : NaN
        okconv = conv >= 8.0
        okabs = min(e27, e14) <= 1e-3
        okburden = isnan(burden) || burden <= 2.0
        push!(oracle_verdicts, @sprintf("%s,conv=%.1f/%.1f(>=8:%s),abs_min=%.2e(<=1e-3:%s),ml_vs_vic=%.2f(<=2:%s)",
            nm, conv, conv2, okconv, min(e27, e14), okabs, burden, okburden))
    end
end

# ---------------------------------------------------------------------------
# selector + gates + report
# ---------------------------------------------------------------------------
selector_rows = String[]
gate_lines = String[]
for row in opt_rows
    f = split(row, ',')
    cname = f[1]; n = parse(Int, f[2])
    t_opt = parse(Float64, f[9]); t_nom = parse(Float64, f[10]); t_pess = parse(Float64, f[11])
    zc = parse(Float64, f[12])
    base = get(BASELINE, (cname, n), nothing)
    if base === nothing
        push!(selector_rows, "$cname,$n,NA,NA,NA,NA,NA,NA,NA,NA,direct_fallback")
        continue
    end
    nearb, compb, unc = base[1], base[2], base[3]
    countgate = (zc <= 0.9 * nearb) && (compb - nearb + zc <= 0.95 * compb)
    fullnear = t_nom <= 0.9 * nearb
    fullcomp = (compb - nearb + t_nom) <= 0.95 * compb
    margin = t_nom * (1 + unc) <= nearb * 0.9
    decision = (countgate && fullnear && fullcomp && margin) ? "adopt" : "direct_fallback"
    push!(selector_rows, @sprintf("%s,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%s,%s,%s,%s",
        cname, n, zc, t_opt, t_nom, t_pess, nearb, countgate, fullnear, fullcomp, decision))
    push!(gate_lines, @sprintf("%s n=%d: count-gate=%s (zcLB %.2f vs 0.9*near %.2f); full-gate near=%s comp=%s; nom %.2f/near %.2f ms",
        cname, n, countgate, zc, 0.9 * nearb, fullnear, fullcomp, t_nom, nearb))
end

# ---------------------------------------------------------------------------
# write outputs
# ---------------------------------------------------------------------------
writecsv(joinpath(OUTDIR, "level_tables.csv"),
    "case,n,gamma,res,level,s_ell,pop,cum_sources,tiles_core,tiles_field,cov_targets,cov_mult8,padded8_ratio,npatch_b64,interior_pts_b64,padded_pts_b64",
    level_rows)
writecsv(joinpath(OUTDIR, "cost_model.csv"),
    "case,n,gamma,res,block,bins,solver,prec,refresh,bracket,M_ext,spread_ms,fft_ms,kmul_ms,interp_ms,restrict_ms,halo_ms,top_ms,launch_ms,refresh_ms,total_ms,bytes_persistent,launches",
    cost_rows)
writecsv(joinpath(OUTDIR, "optimizer.csv"),
    "case,n,gamma,res,block,bins,solver,M_ext,total_opt_ms,total_nom_ms,total_pess_ms,zerocoupling_opt_ms,near_baseline_ms,complete_baseline_ms,uncertainty",
    opt_rows)
writecsv(joinpath(OUTDIR, "unified_check.csv"),
    "case,n,ours_opt_ms,ours_nom_ms,ours_pess_ms,vic037d_opt_ms,vic037d_nom_ms,vic037d_pess_ms,ratio_nom",
    unified_rows)
writecsv(joinpath(OUTDIR, "hybrid.csv"),
    "case,n,cut_level,s_cut,pair_inflation_mean,band_mesh_ms,near_baseline_ms,note",
    hybrid_rows)
writecsv(joinpath(OUTDIR, "oracle.csv"),
    "case,n,mode,gamma,hs,p_int,M_assign,M_ext,uerr_rel_rms,jerr_rel_rms,tail_max_rel,pass_abs",
    oracle_rows)
writecsv(joinpath(OUTDIR, "selector.csv"),
    "case,n,zerocoupling_opt_ms,total_opt_ms,total_nom_ms,total_pess_ms,near_baseline_ms,count_gate,full_near_gate,full_complete_gate,decision",
    selector_rows)
writecsv(joinpath(OUTDIR, "refresh_epoch.csv"),
    "case,n,age_scale,level_drift_fraction", refresh_rows)
writecsv(joinpath(OUTDIR, "manifest.csv"),
    "key,value",
    ["registration,see script header (predeclared design space + brackets + budget)",
     "seed_base,41003/41004/41006/41007-41010 + fm033 rotor seeds",
     "smoke,$(SMOKE)",
     "have_rotor,$(HAVE_ROTOR[])",
     "threads,$(Threads.nthreads())",
     "anchors,037d fm037d_cost_tables + 041a fm041a_gpu_stages + 037b rotor 33.3ms",
     "baseline_convention,lifecycle graph-overlap + warm-epoch refresh (041a); rotor=037b eval",
     "julia,$(VERSION)"])

open(joinpath(OUTDIR, "report.txt"), "w") do io
    println(io, "041f sigma-adaptive multilevel smooth nearfield census")
    println(io, "date=2026-08-18  smoke=$(SMOKE)  rotor=$(HAVE_ROTOR[])")
    println(io, "")
    for l in gate_lines
        println(io, l)
    end
    println(io, "")
    println(io, "note: in cost_model.csv the fft_ms/kmul_ms columns are informational for")
    println(io, "  solver=msm rows; the msm core cost is the msm-specific term inside total_ms.")
    println(io, "oracle rows: $(length(oracle_rows))")
    for v in oracle_verdicts
        println(io, "oracle-verdict: " * v)
    end
    println(io, "hybrid: FMM-retained banded hybrid is structurally additive —")
    println(io, "  the direct complement inflates every cutoff to rho_t*sigma^(c) >= rho_t*sigma")
    println(io, "  (same kernel cost per pair, strictly more pairs) and the meshed bands add on")
    println(io, "  top; see hybrid.csv pair_inflation_mean for the measured inflation.")
end

# checksums (all files except the checksum file itself)
open(joinpath(OUTDIR, "checksums.sha256"), "w") do io
    for f in sort(readdir(OUTDIR))
        f == "checksums.sha256" && continue
        h = bytes2hex(open(sha256, joinpath(OUTDIR, f)))
        println(io, "$h  $f")
    end
end

println("done -> $OUTDIR")
