# Task 037e E0: nearfield direct-list pruning scoping (pure host, no CUDA).
#
# For each production-shaped config this measures, from the REAL host route +
# bucket machinery (`update_radix_state!` -> `build_hierarchical_direct_pairs!`,
# `_nearfield_pair_bucket`, `_nearfield_point_aabb_reach` — the exact predicates
# the CUDA kernels share):
#   (a) direct cell pairs and body pairs by bucket (1 = pure singular,
#       2 = pure regularized, 3 = mixed);
#   (b) the E1 ceiling: fraction of mixed cell pairs where NO target body can
#       reach the source-cell AABB within rho_t*sigma_max(src) (pair-level =
#       whole-warp skippable), plus the same at 32-lane-block and 16-lane
#       (half-warp) granularity, and the mixed body-pair share those skips
#       remove at the implemented 32-lane granularity;
#   (c) per-cell sigma_max spread (min/median/max over occupied cells) and the
#       E2 re-route ceiling: body pairs in bucket-1 (pure-singular) cell pairs
#       whose offset class is M2L-valid per the shipped minimum near set
#       (|o|^2 > 3, the smallest admissible rigid-stencil radius) — direct
#       pairs a sigma-directed router could hand to leaf M2L;
#   (d) a decision table (stdout, and CSV when FM037E_OUT is set).
#
# Case geometry comes from the 033 builders (benchmark_033_common.jl
# fm033_build) when FLOWVPM is available in the project (the cluster fm034env);
# without FLOWVPM the cube and wake cases are rebuilt with the exact same RNG
# stream (positions/sigma bitwise identical — the builders' Gamma draws are
# consumed to keep the stream aligned) and the rotor case is skipped (it needs
# the vendored circulation CSV pipeline inside the FLOWVPM builder).
#
# Usage:
#   FM037E_SMOKE=1 julia --project=. --threads=1 \
#       MATRIX_OPERATOR_REFACTOR/scripts/fm037e_scoping.jl     # local smoke, n=1e4
#   julia --project=$ENVDIR MATRIX_OPERATOR_REFACTOR/scripts/fm037e_scoping.jl
#                                                # full grid (cluster CPU node)
# Env: FM037E_SMOKE=1 (cube n=1e4 only), FM037E_OUT (optional CSV path),
#      FM037E_RHOT (default 3.668, the FLOWVPM production cutoff).

using Random
using Statistics
using Printf
using FastMultipole
using FastMultipole.StaticArrays

const FM037E_SMOKE = get(ENV, "FM037E_SMOKE", "0") == "1"
const FM037E_RHOT = parse(Float64, get(ENV, "FM037E_RHOT", "3.668"))
const FM037E_OUT = get(ENV, "FM037E_OUT", "")
const FM037E_QMIN_M2L = 3   # smallest admissible rigid near set (M2L validity)

# ---------------------------------------------------------------- builders ----

const _HAVE_FLOWVPM = Base.find_package("FLOWVPM") !== nothing

if _HAVE_FLOWVPM
    include(joinpath(@__DIR__, "benchmark_033_common.jl"))
    # positions 3xn and per-body sigma from the FLOWVPM particle matrix
    function fm037e_positions_sigma(case::String, n::Int)
        pfield = fm033_build(case, n)
        na = vpm.get_np(pfield)
        na == n || error("builder returned n=$na, wanted $n")
        P = Array(pfield.particles)
        return Matrix{Float64}(P[1:3, 1:n]), Vector{Float64}(P[7, 1:n])
    end
else
    # stdlib replicas of fm033_build_cube / fm033_build_wake: identical
    # MersenneTwister streams (Gamma draws consumed), identical sigma laws.
    const FM033_SEED = 33025
    const FM033_WAKE_SEED_OFFSET = 7919
    const FM033_WAKE_R = 0.5
    const FM033_WAKE_LEN = 5.0
    function fm037e_positions_sigma(case::String, n::Int)
        X = Matrix{Float64}(undef, 3, n)
        if case == "cube"
            rng = MersenneTwister(FM033_SEED + n)
            for i in 1:n
                X[:, i] .= rand(rng, 3)
                rand(rng, 3)                      # Gamma draw (stream parity)
            end
            return X, fill(2.0 * (1.0 / n)^(1 / 3), n)
        elseif case == "wake"
            rng = MersenneTwister(FM033_SEED + FM033_WAKE_SEED_OFFSET + n)
            for i in 1:n
                r = FM033_WAKE_R * sqrt(rand(rng))
                theta = 2pi * rand(rng)
                z = FM033_WAKE_LEN * (rand(rng) - 0.5)
                X[1, i] = r * cos(theta)
                X[2, i] = r * sin(theta)
                X[3, i] = z
            end
            vol = pi * FM033_WAKE_R^2 * FM033_WAKE_LEN
            return X, fill(2.0 * (vol / n)^(1 / 3), n)
        else
            error("case $case needs FLOWVPM in the project (rotor builder)")
        end
    end
end

# ------------------------------------------------- minimal source system ----
# Positions + per-body sigma only (row 8, the SmoothedVortex convention);
# strengths are zero — this script never evaluates the field, it only drives
# the route refresh. The kernel trait carries the production rho_t.

struct FM037EBodies
    X::Matrix{Float64}      # 3 x n
    sigma::Vector{Float64}
end
function FastMultipole.source_system_to_buffer!(buffer, i_buffer,
        s::FM037EBodies, i_body)
    buffer[1, i_buffer] = s.X[1, i_body]
    buffer[2, i_buffer] = s.X[2, i_body]
    buffer[3, i_buffer] = s.X[3, i_body]
    buffer[4, i_buffer] = 0.0
    buffer[5, i_buffer] = 0.0
    buffer[6, i_buffer] = 0.0
    buffer[7, i_buffer] = 0.0
    buffer[8, i_buffer] = s.sigma[i_body]
    return nothing
end
FastMultipole.data_per_body(::FM037EBodies) = 8
FastMultipole.get_position(s::FM037EBodies, i) =
    SVector{3,Float64}(s.X[1, i], s.X[2, i], s.X[3, i])
FastMultipole.strength_dims(::FM037EBodies) = 3
FastMultipole.get_n_bodies(s::FM037EBodies) = size(s.X, 2)
FastMultipole.has_vector_potential(::FM037EBodies) = true
FastMultipole.body_type(::FM037EBodies) = FastMultipole.Point{FastMultipole.Vortex}
FastMultipole.direct_kernel(::FM037EBodies) =
    FastMultipole.PartitionedVortex(; sigma_row=8, rho_t=FM037E_RHOT)
FastMultipole.buffer_to_target_system!(s::FM037EBodies, i_target, switch,
    buffer, i_buffer) = nothing

# ------------------------------------------------------------------ metrics ----

function fm037e_scope(case::String, n::Int, ell::Int, q::Int)
    X, sigma = fm037e_positions_sigma(case, n)
    sys = FM037EBodies(X, sigma)
    cache = FastMultipole.RadixFMMCache(sys; expansion_order=4, ell=ell,
        near_radius2=q,
        options=FastMultipole.CUDARadixLifecycleOptions(; precision=Float64,
            m2l_strategy=FastMultipole.ConcatenatedFixedZM2L()))
    FastMultipole.update_radix_state!(cache, (sys,))
    state = cache.state
    n_cells = state.counts.n_cells
    n_direct = state.counts.n_direct
    ranges = state.cell_ranges
    coords = cache.coords
    bodies = state.source_bodies       # packed sorted; row 8 = sigma
    h_leaf = 2 * Float64(cache.h0) / (1 << cache.ell)
    x_min = cache.x_min
    rho_t = FM037E_RHOT

    # per-cell sigma extrema over the packed rows
    smax = zeros(n_cells); smin = fill(Inf, n_cells)
    for c in 1:n_cells
        f = ranges[1, c]; cnt = ranges[2, c]
        for i in f:(f + cnt - 1)
            s = bodies[8, i]
            smax[c] = max(smax[c], s)
            smin[c] = min(smin[c], s)
        end
        cnt == 0 && (smin[c] = 0.0)
    end

    cp = zeros(Int, 3)                 # cell pairs by bucket
    bp = zeros(Int64, 3)               # body pairs by bucket
    mixed_pairs = 0
    pair_skip = 0                      # whole-pair skippable (E1 pair ceiling)
    blk32 = 0; blk32_skip = 0          # 32-lane blocks tested / skippable
    blk16 = 0; blk16_skip = 0          # half-warp granularity
    skip_bp32 = Int64(0)               # mixed body pairs removed at 32-lane blocks
    mixed_bp = Int64(0)
    e2_bp = Int64(0)                   # E2 ceiling body pairs
    for p in 1:n_direct
        t = state.direct_targets[p]
        s = state.direct_sources[p]
        o = coords[t] - coords[s]
        b = FastMultipole._nearfield_pair_bucket(o[1], o[2], o[3], h_leaf,
            rho_t, smax[s], smin[s])
        tsz = Int64(ranges[2, t]); ssz = Int64(ranges[2, s])
        cp[b] += 1
        bp[b] += tsz * ssz
        if b == Int32(1) && (o[1]^2 + o[2]^2 + o[3]^2) > FM037E_QMIN_M2L
            e2_bp += tsz * ssz
        end
        b == Int32(3) || continue
        mixed_pairs += 1
        mixed_bp += tsz * ssz
        slo_x = x_min[1] + coords[s][1] * h_leaf
        slo_y = x_min[2] + coords[s][2] * h_leaf
        slo_z = x_min[3] + coords[s][3] * h_leaf
        tf = ranges[1, t]; tcnt = ranges[2, t]
        reach = Vector{Bool}(undef, tcnt)
        for k in 1:tcnt
            i = tf + k - 1
            reach[k] = FastMultipole._nearfield_point_aabb_reach(
                bodies[1, i], bodies[2, i], bodies[3, i],
                slo_x, slo_y, slo_z, h_leaf, rho_t, smax[s])
        end
        any(reach) || (pair_skip += 1)
        for (W, cnt_skip) in ((32, :b32), (16, :b16))
            nblk = cld(tcnt, W)
            for blk in 0:(nblk - 1)
                lo = blk * W + 1
                hi = min(lo + W - 1, tcnt)
                sk = !any(view(reach, lo:hi))
                if W == 32
                    blk32 += 1
                    if sk
                        blk32_skip += 1
                        skip_bp32 += Int64(hi - lo + 1) * ssz
                    end
                else
                    blk16 += 1
                    sk && (blk16_skip += 1)
                end
            end
        end
    end

    occ_smax = smax[1:n_cells]
    tot_bp = sum(bp)
    return (; case, n, ell, q, n_cells, n_direct,
        cp1=cp[1], cp2=cp[2], cp3=cp[3],
        bp1=bp[1], bp2=bp[2], bp3=bp[3], tot_bp,
        mixed_pairs, pair_skip,
        pair_skip_frac=mixed_pairs == 0 ? 0.0 : pair_skip / mixed_pairs,
        blk32_skip_frac=blk32 == 0 ? 0.0 : blk32_skip / blk32,
        blk16_skip_frac=blk16 == 0 ? 0.0 : blk16_skip / blk16,
        skip_bp32,
        skip_bp32_mixed_share=mixed_bp == 0 ? 0.0 : skip_bp32 / mixed_bp,
        skip_bp32_total_share=tot_bp == 0 ? 0.0 : skip_bp32 / tot_bp,
        sigma_max_min=minimum(occ_smax), sigma_max_med=median(occ_smax),
        sigma_max_max=maximum(occ_smax),
        e2_bp, e2_share=tot_bp == 0 ? 0.0 : e2_bp / tot_bp,
        status="ok")
end

# ------------------------------------------------------------------- driver ----

configs = FM037E_SMOKE ?
    [("cube", 10_000, 2, 12)] :
    [("cube", 100_000, 4, 12), ("cube", 1_000_000, 5, 12),
     ("wake", 100_000, 5, 6), ("wake", 1_000_000, 6, 6),
     ("rotor", 100_000, 5, 6), ("rotor", 1_000_000, 6, 6)]

const COLS = ["case", "n", "ell", "q", "status", "n_cells", "n_direct",
    "cp1", "cp2", "cp3", "bp1", "bp2", "bp3", "tot_bp",
    "mixed_pairs", "pair_skip", "pair_skip_frac", "blk32_skip_frac",
    "blk16_skip_frac", "skip_bp32", "skip_bp32_mixed_share",
    "skip_bp32_total_share", "sigma_max_min", "sigma_max_med", "sigma_max_max",
    "e2_bp", "e2_share"]

rows = []
for (case, n, ell, q) in configs
    print("[fm037e] $case n=$n ell=$ell q=$q rho_t=$FM037E_RHOT ... ")
    flush(stdout)
    r = try
        t = @elapsed row = fm037e_scope(case, n, ell, q)
        println("done ($(round(t, digits=1))s)")
        row
    catch err
        msg = replace(first(sprint(showerror, err), 200), r"[,\n]" => ";")
        println("FAILED: $msg")
        (; case, n, ell, q, status="fail: $msg")
    end
    push!(rows, r)
end

println("\n===== fm037e E0 decision table (rho_t=$FM037E_RHOT) =====")
@printf("%-6s %8s %4s %3s | %9s %9s | %-24s | %-30s | %s\n",
    "case", "n", "ell", "q", "cells", "cpairs",
    "bodypair frac b1/b2/b3", "E1 skip: pair/blk32/blk16 bp32", "E2 share")
for r in rows
    if r.status != "ok"
        @printf("%-6s %8d %4d %3d | %s\n", r.case, r.n, r.ell, r.q, r.status)
        continue
    end
    @printf("%-6s %8d %4d %3d | %9d %9d | %.3f/%.3f/%.3f          | %.3f %.3f %.3f bp %.3f/%.3f | %.4f\n",
        r.case, r.n, r.ell, r.q, r.n_cells, r.n_direct,
        r.bp1 / r.tot_bp, r.bp2 / r.tot_bp, r.bp3 / r.tot_bp,
        r.pair_skip_frac, r.blk32_skip_frac, r.blk16_skip_frac,
        r.skip_bp32_mixed_share, r.skip_bp32_total_share, r.e2_share)
end
println("\nE1 ceiling = pair/blk32 skip fractions of the MIXED bucket ",
    "(bp shares are of mixed / of all direct body pairs).")
println("E2 go/no-go: build the sigma-directed re-route only if e2_share >= ",
    "0.10 on some case.")
println("sigma_max spread per config (min/med/max over occupied cells):")
for r in rows
    r.status == "ok" || continue
    @printf("  %-6s n=%-8d  %.4e / %.4e / %.4e\n", r.case, r.n,
        r.sigma_max_min, r.sigma_max_med, r.sigma_max_max)
end

if !isempty(FM037E_OUT)
    open(FM037E_OUT, "w") do io
        println(io, join(COLS, ','))
        for r in rows
            println(io, join((string(hasproperty(r, Symbol(c)) ?
                getproperty(r, Symbol(c)) : "") for c in COLS), ','))
        end
    end
    println("\nCSV written: $FM037E_OUT")
end
