#------- radix geometry policy: box, depth and near stencil, chosen and kept current -------#
#
# Moved from FLOWVPM (src/FLOWVPM_fmm_radix.jl, 2026-10-03) so the decisions about
# what the uniform grid can admit live next to the grid. A consumer describes its
# field with `radix_geometry_source` and holds an `AbstractRadixGeometry`; the uniform
# radix grid's rule is `AutoUniformGeometry`. Another method (an adaptive tree) would
# be another policy behind the same calls.

"""
    AbstractRadixGeometry

How a radix cache's box, depth `ell` and leaf near radius `q` are chosen
([`radix_choose_geometry`](@ref)) and when the live field has outgrown them
([`radix_depth_outgrown!`](@ref), [`radix_sigma_outgrown!`](@ref)).
"""
abstract type AbstractRadixGeometry end

"""
    AbstractOversizePolicy

Which bodies a uniform grid takes out of its tree because their core reach
exceeds what the chosen geometry admits: [`NoOversize`](@ref),
[`FixedOversize`](@ref) or [`AdaptiveOversize`](@ref). The masked bodies keep
their exact influence through an all-pairs extra source ([`MaskedBodies`](@ref)).
"""
abstract type AbstractOversizePolicy end
"Mask nothing."
struct NoOversize <: AbstractOversizePolicy end
"Mask the `count` largest cores (when the field has more than `8 count` bodies and they stand out)."
struct FixedOversize <: AbstractOversizePolicy
    count::Int
end
"""
    AdaptiveOversize(fraction)

Mask every core the geometry chosen for the field without its tail cannot
admit, at most `fraction` of the live count (see [`radix_oversize_threshold`](@ref)).
"""
struct AdaptiveOversize <: AbstractOversizePolicy
    fraction::Float64
end

"Default memory cap on the auto depth: the dense per-level node table is 8^ell Int32 entries (~64 MB at 8)."
const RADIX_AUTO_MAX_ELL = 8

"""
    AutoUniformGeometry(; reach, near_radius2=6, accuracy_margin=1.03, ell=nothing,
                        padding=0.1, rectangular=false, bounds=nothing,
                        rebuild_growth=1.5, max_ell=RADIX_AUTO_MAX_ELL,
                        oversize=AdaptiveOversize(0.02))

The uniform radix grid's rule. `reach` is the direct kernel's primary reach in
units of the core ([`radix_primary_reach`](@ref)); `near_radius2` is the floor on
the leaf near radius; `ell` fixes the depth (a promise: never rebuilt); `bounds`
fixes the box (a promise: never recentered); otherwise the box is derived from the
field, padded by `padding` of its tight extent per face. `oversize` decides which
cores leave the tree ([`radix_oversize_select`](@ref)).
"""
Base.@kwdef struct AutoUniformGeometry <: AbstractRadixGeometry
    reach::Float64
    near_radius2::Int = 6
    accuracy_margin::Float64 = 1.03
    ell::Union{Nothing,Int} = nothing
    padding::Float64 = 0.1
    rectangular::Bool = false
    bounds::Union{Nothing,Tuple} = nothing
    rebuild_growth::Float64 = 1.5
    max_ell::Int = RADIX_AUTO_MAX_ELL
    oversize::AbstractOversizePolicy = AdaptiveOversize(0.02)
end

"""
    radix_primary_reach(kernel) -> Float64

The direct kernel's regularization reach in units of the core: `rho_t`, or
`rho_c` for the two-pass kernel.
"""
radix_primary_reach(kernel) = Float64(kernel.rho_t)
radix_primary_reach(kernel::TwoPassVortex) = Float64(kernel.rho_c)

"""
    radix_geometry_source(system) -> (; P, x_rows, core_row, n)

What the geometry rule reads from a field: the matrix `P` holding one body per
column (host or device), the (contiguous) range of the three coordinate rows, the row of the core
size, and the live count `n`. Defined by the consumer for its system type.
"""
radix_geometry_source(system) = throw(ArgumentError(
    "radix_geometry_source is not defined for $(typeof(system)); a consumer of the radix geometry rule must define it"))

_geom_row_extrema(src, row::Int) = _device_row_extrema(src.P, row, src.n)
_geom_core_max(src) = _geom_row_extrema(src, src.core_row)[2]

"""
    radix_derive_bounds(src, padding; rectangular=false) -> (x_min::SVector{3}, box_size)

Domain bounds covering the live bodies, padded by `padding` of the tight
extent on each face (the `recenter!` convention). Cubic mode (the default)
returns a scalar `box_size` from the maximum tight span; rectangular mode
keeps per-axis tight extents and returns a 3-vector `box_size`,
each axis padded by the same per-face convention
(`L_a = (1 + 2*padding)*ext_a`, centered). In both modes degenerate extents
are inflated to `4*sigma_max` (per axis in rectangular mode) so a
near-singleton field still yields a valid box.
"""
function radix_derive_bounds(src, padding::Real; rectangular::Bool=false)
    lo1, hi1 = _geom_row_extrema(src, src.x_rows[1])
    lo2, hi2 = _geom_row_extrema(src, src.x_rows[2])
    lo3, hi3 = _geom_row_extrema(src, src.x_rows[3])
    cx = (lo1 + hi1) / 2
    cy = (lo2 + hi2) / 2
    cz = (lo3 + hi3) / 2
    floor4s = 4 * _geom_core_max(src)
    if !rectangular
        span = max(hi1 - lo1, hi2 - lo2, hi3 - lo3)
        L_tight = max(span, floor4s)
        L_tight > 0 || error("cannot derive radix FMM bounds: degenerate particle field")
        L = (1 + 2 * padding) * L_tight
        x_min = SVector{3,Float64}(cx - L / 2, cy - L / 2, cz - L / 2)
        return (x_min, Float64(L))
    end
    ex = max(hi1 - lo1, floor4s)
    ey = max(hi2 - lo2, floor4s)
    ez = max(hi3 - lo3, floor4s)
    (ex > 0 && ey > 0 && ez > 0) ||
        error("cannot derive radix FMM bounds: degenerate particle field")
    Lx = (1 + 2 * padding) * ex
    Ly = (1 + 2 * padding) * ey
    Lz = (1 + 2 * padding) * ez
    x_min = SVector{3,Float64}(cx - Lx / 2, cy - Ly / 2, cz - Lz / 2)
    return (x_min, SVector{3,Float64}(Lx, Ly, Lz))
end

"""
    radix_center_snapped_bounds(bounds, ell) -> (x_min, box_extent)

Center the power-of-two rectangular embedding around the center of
automatically derived tight bounds. The longest extent and leaf
width are unchanged; shorter extents are padded symmetrically to whole
power-of-two leaf-cell counts. Explicit user bounds do not use this helper and
therefore retain their caller-owned `x_min` anchor.
"""
function radix_center_snapped_bounds(bounds, ell::Integer)
    x_min = SVector{3,Float64}(bounds[1])
    L = SVector{3,Float64}(bounds[2])
    delta = maximum(L) / (1 << Int(ell))
    function snapped_axis(a)
        la = clamp(ceil(Int, log2(L[a] / delta)), 0, Int(ell))
        while la < ell && delta * (1 << la) < L[a]
            la += 1
        end
        return delta * (1 << la)
    end
    snapped = SVector{3,Float64}(
        snapped_axis(1), snapped_axis(2), snapped_axis(3))
    center = x_min + L / 2
    return (center - snapped / 2, snapped)
end

"""
    radix_occupancy_sums(src, bounds, ell_top) -> Dict{Int,Tuple{Int,Float64}}

For every level `2:ell_top`, the number of OCCUPIED cells and the sum over
cells of (bodies in the cell)^2, from one host sort of the bodies' Morton
keys at `ell_top` (a level-`ell` key is the finest key shifted by
`3*(ell_top - ell)`). The squared sum is the expected number of near-field
pairs per stencil offset; with the stencil size it ranks admissible depths by
the near-field work they cost, which is the term that dominates the device
step. O(np log np) once per rebuild, on the host.
"""
function radix_occupancy_sums(src, bounds, ell_top::Int)
    np = src.n
    out = Dict{Int,Tuple{Int,Float64}}()
    np == 0 && return out
    x_min, L = bounds
    n = 1 << ell_top
    hx, hy, hz = L isa Real ? (L / n, L / n, L / n) : (L[1] / n, L[2] / n, L[3] / n)
    # one host copy: a device-backed field must not be indexed elementwise
    X = Array(view(src.P, src.x_rows, 1:np))
    keys = Vector{UInt64}(undef, np)
    @inbounds for i in 1:np
        ix = clamp(floor(Int, (X[1, i] - x_min[1]) / hx), 0, n - 1)
        iy = clamp(floor(Int, (X[2, i] - x_min[2]) / hy), 0, n - 1)
        iz = clamp(floor(Int, (X[3, i] - x_min[3]) / hz), 0, n - 1)
        keys[i] = UInt64(morton_key(SVector{3,Int}(ix, iy, iz), ell_top))
    end
    sort!(keys)
    for ell in 2:ell_top
        sh = 3 * (ell_top - ell)
        n_occ = 0; sumsq = 0.0
        run = 1
        @inbounds for i in 2:np
            if (keys[i] >> sh) == (keys[i - 1] >> sh)
                run += 1
            else
                n_occ += 1; sumsq += Float64(run)^2; run = 1
            end
        end
        n_occ += 1; sumsq += Float64(run)^2
        out[ell] = (n_occ, sumsq)
    end
    return out
end

# integer offsets within a rigid stencil of squared radius q
radix_stencil_size(q::Int) = (r = isqrt(q); count(ox * ox + oy * oy + oz * oz <= q
    for ox in -r:r, oy in -r:r, oz in -r:r))

"""
    radix_auto_geometry(L, sigma_max, np, q_floor, rho_t, margin;
                        ell_fixed=nothing, occupancy=nothing, max_ell=RADIX_AUTO_MAX_ELL) -> (ell, q)

Joint depth/leaf-radius rule. Every depth `ell` for which some supported leaf
near radius `q >= q_floor` satisfies the margin-guarded inequality
`g_min(q) * h_leaf >= margin * rho_t * sigma_max` (`h_leaf = L / 2^ell`) is
admissible, with the smallest passing `q` at that depth; the cap is only the
memory bound `max_ell`. Among them: with occupancy counts, the fewest expected
near-field pairs (stencil size times the sum of squared cell occupancies);
without, the deepest. The margin buys regularization-deficit accuracy headroom
over the bare adequacy gate (`margin = 1` reproduces adequacy-only selection).
Errors loudly when no depth `>= 2` is admissible.
"""
function radix_auto_geometry(L::Real, sigma_max::Real, np::Int, q_floor::Int,
                             rho_t::Real, margin::Real; ell_fixed=nothing,
                             occupancy=nothing, max_ell::Int=RADIX_AUTO_MAX_ELL)
    reach = margin * rho_t * sigma_max
    qs = sort!([Int(q) for q in _SUPPORTED_RIGID_NEAR_RADII2 if q >= q_floor])
    isempty(qs) && error("near_radius2=$q_floor exceeds every supported rigid " *
        "near radius $(_SUPPORTED_RIGID_NEAR_RADII2)")
    gaps = Dict(q => _ball_stencil_min_gap(q) for q in qs)
    # The only cap is memory: the dense per-level node table is 8^ell Int32
    # entries (~64 MB at 8). An occupancy heuristic of ~n^(1/3) cells per
    # side used to sit here; it assumes a uniformly filled box, and a wake is
    # a thin structure in a mostly empty one. Checked against an exact sum
    # (LiftingLines test/gpu/al_depth_rule.jl, 2026-09-19): sixteen rotors at 363k
    # particles are at 1.3e-4..1.6e-4 for every depth 2..8, four rotors at
    # 4.6e-5..6.3e-5 for 2..6, Metal 1e-5 for 2..4 -- no degradation with
    # depth -- and the deepest is the fastest (16 rotors: 73 ms at 8 vs 171
    # at the cap's 6). Which admissible depth is used is decided below by
    # the near-pair count, not by "deepest".
    ell_top = max_ell
    # a fixed `ell` still takes the smallest adequate stencil at that depth
    # (near_radius2 is a floor on the auto path too)
    ells = ell_fixed === nothing ? (ell_top:-1:2) : (Int(ell_fixed):Int(ell_fixed))
    # every admissible depth, with the smallest near set that satisfies it
    admissible = Tuple{Int,Int}[]
    for ell in ells
        h = L / 2^ell
        for q in qs
            if gaps[q] * h >= reach
                push!(admissible, (ell, q))
                break
            end
        end
    end
    if !isempty(admissible)
        # Without occupancy counts: deepest admissible. With them: the
        # admissible (ell, q) with the fewest expected near-field pairs,
        # stencil size times the sum of squared cell occupancies. That is
        # the term that dominates the device step (M2L is a few percent), and
        # it is what "deepest admissible" gets wrong when the smallest
        # adequate stencil at the deepest level is wide: the NREL 5MW spent
        # an epoch at 8 s/step, four times its other epochs, on such a pick.
        # Exact counts, no calibration.
        occupancy === nothing && return first(admissible)
        best = first(admissible); best_cost = Inf
        for (ell, q) in admissible
            haskey(occupancy, ell) || continue
            c = radix_stencil_size(q) * occupancy[ell][2]
            if c < best_cost
                best_cost = c; best = (ell, q)
            end
        end
        return best
    end
    error("no admissible radix depth (need ell >= 2): the margin-guarded " *
        "near-set inequality requires g_min(q)*L/2^ell >= " *
        "margin*rho_t*sigma_max = $reach, but even ell = 2 with the largest " *
        "supported q >= $q_floor gives $(maximum(gaps[q] for q in qs) * L / 4). " *
        "Reduce the smoothing overlap, enlarge the domain box, or use more " *
        "particles.")
end

_geom_L(bounds) = bounds[2] isa Real ? Float64(bounds[2]) : Float64(maximum(bounds[2]))

"""
    radix_choose_geometry(g::AutoUniformGeometry, src; verbose=false) -> (bounds, ell, q)

The box, depth and leaf near radius a cache built now should have: the fixed or
derived bounds, then [`radix_auto_geometry`](@ref) on the field's largest core
and occupancy; an auto rectangular box is center-snapped to the chosen depth.
"""
function radix_choose_geometry(g::AutoUniformGeometry, src; verbose::Bool=false)
    bounds = g.bounds === nothing ?
        radix_derive_bounds(src, g.padding; rectangular=g.rectangular) : g.bounds
    sigma_max = Float64(_geom_core_max(src))
    # The auto-geometry rule is shape-independent: L = max extent
    # drives ell and q via sigma-adequacy exactly as in cubic mode, so the leaf
    # width L/2^ell is identical — rectangularity only trims per-axis counts.
    L_geo = _geom_L(bounds)
    occupancy = radix_occupancy_sums(src, bounds, g.max_ell)
    ell, q = radix_auto_geometry(L_geo, sigma_max, src.n, g.near_radius2,
        g.reach, g.accuracy_margin; ell_fixed = g.ell, occupancy, max_ell = g.max_ell)
    if verbose
        # one line per (re)build: what the depth rule saw and what it chose
        occ = join((string(l, ":", occupancy[l][1], "/", round(Int, occupancy[l][2]))
                    for l in sort!(collect(keys(occupancy)))), " ")
        println("radix build: np=$(src.n) L=$(round(L_geo; sigdigits=4)) sigma_max=$(round(sigma_max; sigdigits=4)) -> ell=$ell q=$q  [ell:cells/sum_sq  $occ]")
        flush(stdout)
    end
    if g.bounds === nothing && g.rectangular
        bounds = radix_center_snapped_bounds(bounds, ell)
    end
    return bounds, ell, q
end

"""
    radix_recenter_bounds(g::AutoUniformGeometry, src, ell) -> bounds

The bounds a cache of depth `ell` recenters to when the field leaves its box
(derived, padded, center-snapped when rectangular).
"""
function radix_recenter_bounds(g::AutoUniformGeometry, src, ell::Integer)
    bounds = radix_derive_bounds(src, g.padding; rectangular=g.rectangular)
    g.rectangular && (bounds = radix_center_snapped_bounds(bounds, ell))
    return bounds
end

"""
    radix_sigma_limit(g::AutoUniformGeometry, cache) -> Float64

Largest `sigma_max` the cached grid can serve. FastMultipole's runtime adequacy
gate refuses to evaluate when `g_min*h_leaf <= rho_reach*sigma_max`; cores grow
between builds, so a geometry picked with headroom at build time can become
inadmissible mid-run. The limit divides out the SAME `accuracy_margin` the auto
rule applies at build, so a rebuild triggers while the bare gate still holds.
`Inf` for a zero-M2L degenerate cache (the gate is vacuous there).
"""
function radix_sigma_limit(g::AutoUniformGeometry, cache)
    isempty(cache.accepted_offsets) && return Inf
    g_min = _leaf_stencil_min_gap(cache)
    h_leaf = 2 * Float64(cache.h0) / (1 << cache.ell)
    return g_min * h_leaf / (Float64(g.accuracy_margin) * Float64(g.reach))
end

# The same limit for a hierarchical geometry given as (ell, box, q, level_radii2), before
# its cache exists (a checkpoint restore builds the cache from it): `box` is the cubic
# edge or the rectangular extents, whose largest is the virtual cube's 2 h0.
function radix_sigma_limit(g::AutoUniformGeometry, ell::Integer, box, q::Integer, level_radii2=())
    g_min = _ball_stencil_min_gap(isempty(level_radii2) ? Int(q) : Int(level_radii2[end]))
    h_leaf = Float64(maximum(box)) / (1 << ell)
    return g_min * h_leaf / (Float64(g.accuracy_margin) * Float64(g.reach))
end

"""
    radix_depth_outgrown!(g::AutoUniformGeometry, src, cache, q_cached, np_checked, evals;
                          verbose=false) -> Bool

`true` when the auto rule at the CURRENT count, bounds and cores would pick a
different `(ell, q)` than the cached ones, so the caller rebuilds. A fixed `ell`
is a promise and is never outgrown. Checked when the count has grown by
`rebuild_growth` since the last check (`np_checked`, a `Ref`) or after 60
evaluations (`evals`, a `Ref`): a convecting wake's box grows without the count
doubling. If no admissible geometry exists at the grown shape the cache is kept.
"""
function radix_depth_outgrown!(g::AutoUniformGeometry, src, cache, q_cached,
                               np_checked, evals; verbose::Bool=false)
    g.ell === nothing || return false
    np = src.n
    # Two triggers: the count has grown by `rebuild_growth`, or 60 evaluations
    # (~10 RK3 steps) have passed -- the box of a convecting wake grows without
    # the count doubling (NREL 5MW: L 1417 -> 1840 m between 531k and 705k, the
    # 705k geometry one level deeper and 25% faster; 2026-09-21). The check
    # itself costs ~0.02 s and rebuilds only when (ell, q) would change.
    evals[] += 1
    (np > g.rebuild_growth * np_checked[] || evals[] >= 60) || return false
    evals[] = 0
    np_checked[] = np
    t0 = time()
    verbose && (println("radix depth check: np=$np (cached ell=$(cache.ell)) ..."); flush(stdout))
    # The admissible depth grows with the box (a wake convects), so the
    # geometry is re-derived whenever the count has grown by rebuild_growth.
    bounds = g.bounds === nothing ?
        radix_derive_bounds(src, g.padding; rectangular=g.rectangular) : g.bounds
    sigma_max = Float64(_geom_core_max(src))
    L_geo = _geom_L(bounds)
    t1 = time()
    occupancy = radix_occupancy_sums(src, bounds, g.max_ell)
    t2 = time()
    ell, q = try
        radix_auto_geometry(L_geo, sigma_max, np, g.near_radius2, g.reach,
            g.accuracy_margin; occupancy, max_ell = g.max_ell)
    catch
        verbose && (println("radix depth check: no admissible geometry, cache kept ($(round(time() - t0; digits=2)) s)"); flush(stdout))
        return false
    end
    verbose && (println("radix depth check: L=$(round(L_geo; sigdigits=4)) sigma_max=$(round(sigma_max; sigdigits=4)) -> ell=$ell (bounds+sigma $(round(t1 - t0; digits=2)) s, occupancy $(round(t2 - t1; digits=2)) s, total $(round(time() - t0; digits=2)) s)"); flush(stdout))
    # The cheapest geometry can move either way as the wake spreads, and the
    # stencil radius matters as much as the depth: a box that grew 1.8x at the
    # same depth kept a q sized for the old, smaller cells (NREL 5MW, 2026-09-21).
    # Compare both.
    return ell != cache.ell || q != q_cached
end

"""
    radix_sigma_outgrown!(g::AutoUniformGeometry, src, cache; verbose=false) -> Bool

Companion to [`radix_depth_outgrown!`](@ref) in the opposite direction: `true`
when the LIVE largest core exceeds the cached geometry's admissible limit
([`radix_sigma_limit`](@ref)) and an admissible geometry exists at the grown
core, so the caller rebuilds (shallower `ell` and/or a larger near set). Checked
every call (one O(n) device row reduction). A fixed `ell` is never rebuilt
(the runtime gate's error propagates).
"""
function radix_sigma_outgrown!(g::AutoUniformGeometry, src, cache; verbose::Bool=false)
    g.ell === nothing || return false
    sigma_max = Float64(_geom_core_max(src))
    # the live limit: `recenter!` changes the cache's box (and so its limit)
    limit = radix_sigma_limit(g, cache)
    sigma_max > limit || return false
    verbose && (println("radix sigma outgrown: np=$(src.n) sigma_max=$(round(sigma_max; sigdigits=4)) > limit $(round(limit; sigdigits=4))"); flush(stdout))
    bounds = g.bounds === nothing ?
        radix_derive_bounds(src, g.padding; rectangular=g.rectangular) : g.bounds
    try
        radix_auto_geometry(_geom_L(bounds), sigma_max, src.n, g.near_radius2,
            g.reach, g.accuracy_margin; max_ell = g.max_ell)
    catch
        return false
    end
    return true
end

#------- bodies whose core the grid cannot admit (moved from FLOWVPM, 2026-10-03) -------#
#
# A tail of large cores would force the whole field onto a coarser grid (the
# geometry must admit the largest core's reach). Instead those bodies are taken
# out of the tree for the evaluation: the consumer zeroes their strength and core
# in its own storage (`radix_mask_bodies!`, which returns their packed columns),
# the tree sees the rest, and the masked bodies go back in through a `MaskedBodies`
# extra source, each at the coarser level its reach admits (radix_multilevel.jl);
# `radix_unmask_bodies!` restores them.

"""
    MaskedBodies(buffer, kernel, strength_dims; idx=Int[], bodytype=nothing, margin=1.0)

The masked bodies of one evaluation as an extra source: `buffer` holds their packed
columns in the consumer's source layout (one column per body, as its
`source_system_to_buffer!` writes them), `kernel` is the direct kernel, `idx` their
global indices into system 1, `bodytype` their element type (for the multipole).
On a device (KA) cache each body goes back into the tree at the deepest level whose
stencil admits `margin` times its regularization reach: direct to the targets near
it at that level, the far field through the tree, all-pairs only when no level
admits it (radix_multilevel.jl). The host radix path, a GPU test oracle, applies
them all-pairs.
"""
struct MaskedBodies{TF,K}
    buffer::Matrix{TF}
    kernel::K
    strength_dims::Int
    idx::Vector{Int}
    bodytype::Any
    margin::Float64
end
MaskedBodies(buffer, kernel, strength_dims::Integer; idx=Int[], bodytype=nothing, margin=1.0) =
    MaskedBodies(buffer, kernel, Int(strength_dims), collect(Int, idx), bodytype, Float64(margin))
body_type(o::MaskedBodies) = o.bodytype === nothing ?
    throw(ArgumentError("MaskedBodies needs `bodytype` for a multipole")) : o.bodytype
get_n_bodies(o::MaskedBodies) = size(o.buffer, 2)
data_per_body(o::MaskedBodies) = size(o.buffer, 1)
get_position(o::MaskedBodies, i) = SVector{3}(o.buffer[1, i], o.buffer[2, i], o.buffer[3, i])
strength_dims(o::MaskedBodies) = o.strength_dims
direct_kernel(o::MaskedBodies) = o.kernel
function source_system_to_buffer!(buffer, i_buffer, o::MaskedBodies, i_body)
    @inbounds for r in 1:size(o.buffer, 1)
        buffer[r, i_buffer] = o.buffer[r, i_body]
    end
    return nothing
end

"""
    radix_mask_bodies!(system, idx) -> buffer::Matrix

Consumer hook: return the packed source columns of bodies `idx` (as
`source_system_to_buffer!` would write them) and zero their strength and core in
the system's own storage, so the next packing leaves them out of the tree.
"""
radix_mask_bodies!(system, idx) = throw(ArgumentError(
    "radix_mask_bodies! is not defined for $(typeof(system)); masking oversize bodies needs it"))

"""
    radix_unmask_bodies!(system, idx, buffer)

Consumer hook: restore the strength and core of bodies `idx` from the columns
[`radix_mask_bodies!`](@ref) returned.
"""
radix_unmask_bodies!(system, idx, buffer) = throw(ArgumentError(
    "radix_unmask_bodies! is not defined for $(typeof(system))"))

"""
    radix_rows_above(P, row, n, thr, cap) -> Vector{Int}

The columns `1:n` of `P` whose `row` exceeds `thr`, at most `cap` (the largest),
sorted. Host and device (the KA extension) give the same list.
"""
function radix_rows_above(P::Matrix, row::Int, np::Int, thr, cap::Int)
    sig = view(P, row, 1:np)
    idx = findall(>(thr), sig)
    length(idx) > cap && (idx = idx[partialsortperm(view(sig, idx), 1:cap; rev=true)])
    return idx
end

"""
    radix_rows_top(P, row, n, K) -> Vector{Int}

Of the `K` largest values of `row` over columns `1:n`, those that stand clear of
the `(K+1)`-th (by 2%); empty when the row is flat.
"""
function radix_rows_top(P::Matrix, row::Int, np::Int, K::Int)
    sig = view(P, row, 1:np)
    top = partialsortperm(sig, 1:(K + 1); rev=true)
    sigma_ref = sig[top[K + 1]]
    return [i for i in view(top, 1:K) if sig[i] > 1.02 * sigma_ref]
end

radix_oversize_kmax(p::AdaptiveOversize, np::Int) = max(32, round(Int, p.fraction * np))

"""
    radix_oversize_threshold(g::AutoUniformGeometry, src, rec; verbose=false) -> (thr, rec)

The core size above which bodies leave the tree (`Inf`: nothing to mask), for an
[`AdaptiveOversize`](@ref) policy. Derived from the field: take the
`(K_max+1)`-th largest core as the largest core the field would have without its
tail, ask the auto-geometry rule (occupancy included) which `(ell, q)` it would
choose, and return that geometry's adequacy limit `g_min(q) * L/2^ell / (margin *
reach)`. Every core above it is masked, at most `K_max` of them by construction.
`rec` (`nothing` or `(; thr, np, evals::Ref)`) caches it: refreshed when the live
count has grown 5% or after 60 evaluations; the returned record is a new object
only when it was refreshed.
"""
function radix_oversize_threshold(g::AutoUniformGeometry, src, rec; verbose::Bool=false)
    np = src.n
    np > 256 || return Inf, rec
    if rec !== nothing && np <= 1.05 * rec.np && rec.evals[] < 60
        rec.evals[] += 1
        return rec.thr, rec
    end
    K_max = radix_oversize_kmax(g.oversize, np)
    sig = Array(view(src.P, src.core_row, 1:np))
    sigma_top = Float64(maximum(sig))
    sigma_q = Float64(partialsort(sig, K_max + 1; rev=true))
    thr = Inf
    if sigma_q < sigma_top
        bounds = g.bounds === nothing ?
            radix_derive_bounds(src, g.padding; rectangular=g.rectangular) : g.bounds
        L_geo = bounds[2] isa Real ? Float64(bounds[2]) : Float64(maximum(bounds[2]))
        rho_t = g.reach
        ell, q = try
            radix_auto_geometry(L_geo, sigma_q, np, g.near_radius2, rho_t,
                g.accuracy_margin; ell_fixed=g.ell,
                occupancy=radix_occupancy_sums(src, bounds, g.max_ell), max_ell=g.max_ell)
        catch
            (0, 0)
        end
        if ell > 0
            lim = _ball_stencil_min_gap(q) * (L_geo / 2^ell) / (g.accuracy_margin * rho_t)
            lim < sigma_top && (thr = max(lim, sigma_q))
        end
        verbose && (println(
            "radix oversize threshold: np=$np K_max=$K_max sigma_q=$(round(sigma_q; sigdigits=4)) " *
            "sigma_max=$(round(sigma_top; sigdigits=4)) -> ell=$ell q=$q thr=$(round(thr; sigdigits=4))"); flush(stdout))
    end
    return thr, (; thr, np, evals=Ref(0))
end

"""
    radix_oversize_select(g::AutoUniformGeometry, src, rec, cache; verbose=false) -> (idx, rec)

The bodies to take out of the tree for this evaluation (sorted global indices) under
`g.oversize`, never above what the cached grid (`cache`, or `nothing` before the
first build) admits. Adaptive: every core above [`radix_oversize_threshold`](@ref),
at most `K_max`; a longer tail drops the threshold record (`rec = nothing`) so the
next call re-derives it.
"""
function radix_oversize_select(g::AutoUniformGeometry, src, rec, cache; verbose::Bool=false)
    p = g.oversize
    p isa NoOversize && return Int[], rec
    np = src.n
    if p isa FixedOversize
        K = p.count
        (K > 0 && np > 8 * K) || return Int[], rec
        return radix_rows_top(src.P, src.core_row, np, K), rec
    end
    thr, rec = radix_oversize_threshold(g, src, rec; verbose)
    # Never above what the cached grid admits: the threshold is derived for a
    # geometry the cache may not have, so on its own it let cores through that
    # the cached geometry cannot serve
    cache === nothing || (thr = min(thr, radix_sigma_limit(g, cache)))
    isfinite(thr) || return Int[], rec
    K_max = radix_oversize_kmax(p, np)
    idx = radix_rows_above(src.P, src.core_row, np, thr, K_max + 64)
    length(idx) > K_max && (rec = nothing)
    return idx, rec
end
