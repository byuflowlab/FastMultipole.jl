#------- RadixFMMCache: fixed-box recurring driver -------#

function _assert_radix_targets_are_sources(targets::Tuple, sources::Tuple)
    length(targets) == length(sources) ||
        throw(ArgumentError("the radix fmm! path requires target_systems === source_systems (v1 restriction)"))
    for (t, s) in zip(targets, sources)
        t === s ||
            throw(ArgumentError("the radix fmm! path requires target_systems === source_systems (v1 restriction)"))
    end
    return nothing
end

#------- near-set adequacy gate for regularized nearfield kernels -------#
#
# The FMM far field is singular under every nearfield strategy, so the direct
# geometry must contain every pair inside the smoothing cutoff r/σ_src ≤ ρ_t or
# the accuracy gate is silently missed. The binding
# quantity is the smallest AABB gap the stencil leaves to M2L: adequacy is
# g_min·h_leaf > ρ_t·σ_max, evaluated per step from the live geometry (σ may
# grow, e.g. under core spreading). The box-filling n/8^ℓ form is design-time
# sizing only and must NOT be asserted here — it mis-ranks clustered fields by
# one to two levels. An inadequate configuration is REJECTED with the
# measured ratio and admissible depth.

@inline _offset_gap2(o) =
    Float64(max(0, abs(o[1]) - 1)^2 + max(0, abs(o[2]) - 1)^2 + max(0, abs(o[3]) - 1)^2)

# min over {o : |o|² > q} of the AABB gap in cell units (√5 for q = 12, 1 for
# q = 3..5); brute force over the finite shell just outside the ball.
function _ball_stencil_min_gap(q::Int)
    reach = ceil(Int, sqrt(q)) + 2
    best = Inf
    for oz in -reach:reach, oy in -reach:reach, ox in -reach:reach
        ox * ox + oy * oy + oz * oz <= q && continue
        best = min(best, _offset_gap2((ox, oy, oz)))
    end
    return sqrt(best)
end

# Leaf-level minimum M2L gap in units of the leaf cell size. The constraint
# binds only at the leaf level (ρ_tσ/h halves with each level up
# while the coarse-level gap in leaf units doubles), so coarser levels of the
# hierarchical schedule need no separate check.
function _leaf_stencil_min_gap(policy, accepted_offsets)
    policy isa HierarchicalRigidStencil && return _ball_stencil_min_gap(
        isempty(policy.level_radii2) ? policy.near_radius2 : policy.level_radii2[end])
    # flat analytic stencil: the direct set is `rejected_offsets`, so the binding
    # M2L pair is the closest accepted offset class
    best = Inf
    for o in accepted_offsets
        best = min(best, _offset_gap2(o))
    end
    return sqrt(best)
end

_leaf_stencil_min_gap(cache::RadixFMMCache) =
    _leaf_stencil_min_gap(cache.policy, cache.accepted_offsets)

_direct_kernel_geometry_gate!(cache::RadixFMMCache, ::AbstractDirectKernel,
    source_bodies, n::Int) = nothing

# The reach the primary direct near set must cover, in units of sigma. The
# single-pass regularized kernels evaluate every cutoff pair directly, so they
# need the full rho_t; the two-pass hybrid's pass 1 only evaluates ρ ≤ rho_c
# regularized (its self-sizing pass-2 sweep covers the (rho_c, rho_t] shell on
# its own, see _host_twopass_deficit_kernel!), so its gate binds at rho_c.
@inline _gate_reach_rho(kernel::AbstractRegularizedVortex) = (kernel.rho_t, "rho_t")
@inline _gate_reach_rho(kernel::TwoPassVortex) = (kernel.rho_c, "rho_c (pass-1 hybrid switch)")

"""
    _device_row_extrema(A, row, n) -> (lo, hi)
    _device_row_max(A, row, n)

`extrema(A[row, 1:n])` for a host OR device matrix. The host default is the
plain view reduction. The KA extension overrides the device case with a
FIXED-geometry kernel: a GPUArrays `maximum` over an `n`-length strided view
compiles one reduction kernel per distinct `n` (measured 140 ms each on
Metal), and a shedding solver changes `n` every step.
"""
function _device_row_extrema(A::AbstractMatrix, row::Integer, n::Integer)
    v = view(A, row, 1:n)
    return minimum(v), maximum(v)
end
_device_row_max(A, row::Integer, n::Integer) = _device_row_extrema(A, row, n)[2]

function _direct_kernel_geometry_gate!(cache::RadixFMMCache,
        kernel::AbstractRegularizedVortex, source_bodies, n::Int)
    n > 0 || return nothing
    # Zero-M2L degenerate cache: with no accepted offset class the
    # direct list covers every pair at every offset, so no pair can fall to
    # the singular far field — the adequacy gate is vacuous.
    isempty(cache.accepted_offsets) && return nothing
    # works for host and device matrices alike (device reduction + scalar download)
    sigma_max = Float64(_device_row_max(source_bodies, kernel.sigma_row, n))
    sigma_max > 0 || return nothing
    g_min = _leaf_stencil_min_gap(cache)
    h_leaf = 2 * Float64(cache.h0) / (1 << cache.ell)
    rho_reach, rho_name = _gate_reach_rho(kernel)
    cutoff = rho_reach * sigma_max
    g_min * h_leaf > cutoff && return nothing
    x = g_min * 2 * Float64(cache.h0) / cutoff   # admissible 2^ℓ bound
    ell_max = floor(Int, log2(x))
    2.0^ell_max < x || (ell_max -= 1)
    depth_msg = ell_max >= 0 ? "the admissible depth at this geometry is ell <= $ell_max" :
        "no tree depth is admissible at this geometry (the box itself is inside the cutoff)"
    msg = "regularized nearfield near-set adequacy failed: the direct stencil leaves " *
        "an M2L gap of g_min*h_leaf = $(round(g_min * h_leaf, sigdigits=4)) but the " *
        "smoothing cutoff needs $rho_name*sigma_max = $(round(cutoff, sigdigits=4)) " *
        "(ratio $(round(g_min * h_leaf / cutoff, sigdigits=4)), g_min = " *
        "$(round(g_min, sigdigits=4)), sigma_max = $(round(sigma_max, sigdigits=4)), " *
        "ell = $(cache.ell)); $depth_msg. Pairs inside the cutoff would be handled " *
        "by the singular far field and silently lose the regularization."
    # a hierarchical cache does not
    # throw here. sigma can outgrow every admissible stencil geometry mid-run
    # (core spreading + merging fatten sigma_max monotonically), so the caller
    # demotes the cache to the all-direct zero-M2L geometry instead:
    # every pair is evaluated by the regularized direct kernel on the same
    # arrays/device, and no pair can reach the singular far field. The
    # demotion is terminal for the cache — the degenerate geometry has no
    # accepted offsets, so this gate goes vacuous and never fires again.
    # TwoPassVortex is excluded: its pass-2 deficit sweep is sized from the
    # gate-passing geometry, so the degenerate grid is not admissible for it.
    if cache.policy isa HierarchicalRigidStencil && !(kernel isa TwoPassVortex)
        @warn "$msg Falling back to the all-direct zero-M2L geometry: the " *
            "cache is rebuilt at ell = 2 with a full-grid near ball (q = 27) " *
            "and every pair runs the regularized direct kernel." maxlog = 4
        return :alldirect
    end
    throw(ArgumentError(msg))
end

# all-direct demotion for a cache whose sigma_max outgrew every
# admissible stencil geometry. Mirrors the recenter! rebuild-and-swap idiom
# (same bounds, same capacities, same options) but forces
# ell = 2 with a full-grid near ball: at ell = 2 every leaf offset satisfies
# |o|^2 <= 27, so _radix_root_level reports L_allnear = ell, the scheduled
# tables degenerate to the zero-M2L form, and the whole evaluation is
# the regularized direct near field on the original arrays/device. Cost mirrors
# recenter!: one construction-equivalent rebuild (device caches transiently
# ~2x device memory), after which the adequacy gate is vacuous forever.
function _alldirect_geometry_fallback!(cache::RadixFMMCache{TF,LH},
        systems::Tuple) where {TF,LH}
    policy = cache.policy
    policy isa HierarchicalRigidStencil || throw(ArgumentError(
        "the all-direct adequacy fallback requires a HierarchicalRigidStencil " *
        "policy; got $(typeof(policy))"))
    # re-derive the box-scaled tolerance the q = 27 ball realizes at ell = 2
    # (the construction-time _verify_hierarchical_classifier! gate requires the
    # epsilon and the near set to agree exactly, same as _recentered_policy)
    cfg = policy.config
    eps_new = rigid_stencil_epsilon(cfg.P_phi, maximum(cache.box_extent) / 2, 2, 27;
        lamb_helmholtz=LH, TF)
    newconfig = ConstantPStencilConfig(cfg.P_phi, TF(eps_new), cfg.source_strength;
        chi_strength=cfg.chi_strength, lamb_helmholtz=LH,
        normalization=_config_normalization(cfg))
    newpolicy = HierarchicalRigidStencil(newconfig;
        near_radius2=27, level_radii2=(),
        window_classes=policy.window_classes,
        dense_occupancy_max_bytes=policy.dense_occupancy_max_bytes,
        dense_occupancy_max_ell=policy.dense_occupancy_max_ell)
    fresh = RadixFMMCache(systems, systems;
        expansion_order=cache.expansion_order, ell=2,
        max_n_bodies=cache.max_n_bodies,
        bounds=(cache.x_min, cache.box_extent),
        lamb_helmholtz=LH, hessian=cache.hessian, device=cache.device,
        options=cache.options,
        policy=newpolicy)
    for f in fieldnames(RadixFMMCache)
        setfield!(cache, f, getfield(fresh, f))
    end
    return cache
end

function _assert_radix_positions_in_box(systems::Tuple, x_min::SVector{3,TF},
        box_extent::SVector{3,TF}) where TF
    x_max = x_min .+ box_extent
    for (isys, system) in enumerate(systems)
        for i_body in 1:get_n_bodies(system)
            x = get_position(system, i_body)
            if !(x_min[1] <= x[1] <= x_max[1] && x_min[2] <= x[2] <= x_max[2] &&
                 x_min[3] <= x[3] <= x_max[3])
                throw(ArgumentError(
                    "body $i_body of system $isys at $(Tuple(x)) lies outside the fixed " *
                    "RadixFMMCache box [$(Tuple(x_min)), $(Tuple(x_max))]; the box is part " *
                    "of the cache's invariant contract — construct a new cache (or pass " *
                    "explicit bounds=(x_min, box_size) covering the trajectory)"))
            end
        end
    end
    return nothing
end

# Resolve the rectangular geometry contract from the bounds box size.
# A scalar size gives a cube of that side. Vector sizes
# embed the box in a virtual cube of half-width h0 = maximum(box_size)/2 whose
# leaf width Δ = 2h0/2^ell tiles every axis: axis a spans 2^ell_axes[a] leaf
# cells with its extent snapped up to Δ * 2^ell_axes[a] (never below the
# requested extent).
function _resolve_radix_ell_axes(box_size::Real, ell::Int, ::Type{TF}) where TF
    h0 = TF(box_size) / 2
    h0 > zero(TF) || throw(ArgumentError("bounds box_size must be positive"))
    return SVector(ell, ell, ell), h0, SVector{3,TF}(2 * h0, 2 * h0, 2 * h0)
end

function _resolve_radix_ell_axes(box_size, ell::Int, ::Type{TF}) where TF
    L = SVector{3,TF}(box_size)
    (L[1] > zero(TF) && L[2] > zero(TF) && L[3] > zero(TF)) || throw(ArgumentError(
        "bounds box_size must be positive on every axis; got $(Tuple(L))"))
    h0 = max(L[1], L[2], L[3]) / 2
    delta = (2 * h0) / (1 << ell)
    function resolve_axis(a)
        la = clamp(ceil(Int, log2(Float64(L[a]) / Float64(delta))), 0, ell)
        # fp guard: log2/ceil may land one level short of covering the extent
        while la < ell && delta * (1 << la) < L[a]
            la += 1
        end
        return la
    end
    ell_axes = SVector(resolve_axis(1), resolve_axis(2), resolve_axis(3))
    box_extent = SVector{3,TF}(delta * (1 << ell_axes[1]),
        delta * (1 << ell_axes[2]), delta * (1 << ell_axes[3]))
    return ell_axes, h0, box_extent
end

# Measured window widths. The host default of 4 is the fastest measured host
# width. On the GPU the per-window flag/scan/compact carries a fixed ~50 us
# device-to-host round trip, so route generation scales as
# `(ell - 1) * ceil(noffsets / K)` and K = 4 spends 72-164 ms per step on latency
# alone. On an H200, route generation fell 109-193x from K = 4 to
# a whole-level window.
#
# At n = 1e6, K = 256 still spent 24.19 ms per step in route generation against
# 2.13 ms for a whole-level window, so the device default is larger than any
# supported shell's offset count and every level is generated in one window. The cost is route-buffer memory, which
# grows as `min(K, noffsets) * max_level_nodes` (2.0 GB persistent at n = 1e6,
# ell = 5 on the default policy); pass a smaller `window_classes` on
# memory-constrained devices.
const RADIX_HOST_WINDOW_CLASSES = 4
const RADIX_DEVICE_WINDOW_CLASSES = 4096

# Measured defaults for `RadixFMMCache(...; options=nothing)`.
# Both choices are made from the expansion order, the Lamb-Helmholtz channel, the
# platform, and the dense operator footprint — measured to be sufficient
# selectors — and both are overridden by passing an explicit `options`.
#
# Precision. Host caches always default to Float64: the CPU gains nothing from
# Float32. A device cache defaults to Float32 only for literature P <= 4
# (expansion_order <= 3). There the stencil's own truncation error dominates:
# at n = 1e6 the max gradient error was 3.186e-4 in Float32 vs 3.185e-4 in
# Float64 (+0.03%), and Float32 was 1.14x faster. Above P = 4 the default is
# Float64. This is a chosen default, not a Float32 accuracy floor: measured
# against a Float64 all-pairs reference, the Float32 FMM error keeps falling with
# P (to ~6e-7 on a wake case at P ≈ 8-10), so a Float32 cache at higher order is
# a legitimate explicit choice.
_default_radix_precision(expansion_order::Int, device::Bool) =
    device && expansion_order <= 3 ? Float32 : Float64

# Host strategy, from the measured recurring-step rules: dense wins every measured
# P = 4 and P = 8 case; precomputed-y wins P = 12, where dense is either unsupported
# (Float32) or over its memory gate. Dense also beat precomputed-y on the
# hierarchical path at n = 1e6 (106.3 vs 172.3 ms). A device cache always takes
# `ConcatenatedFixedZM2L`: it is the only plan the KernelAbstractions build has.
#
# Dense trades construction for steady state (~20 s build and ~300-370 break-even
# steps in the benchmarked configuration), which suits the repeated-step cache this is, but not
# one-shot evaluation: pass `PrecomputedFactoredYM2L()` explicitly for that.
function _default_radix_m2l_strategy(::Type{TF}, expansion_order::Int, LH::Bool,
        device::Bool, nclasses::Int, ndof::Int) where TF
    # The device lifecycle is the KernelAbstractions extension, which builds the
    # concatenated hierarchical plan only (dense and factored are host-only).
    device && return ConcatenatedFixedZM2L()
    dense = DenseTranslationM2L()
    fallback = PrecomputedFactoredYM2L()
    # the operator payload is dense's binding constraint; keep a margin
    # under its own gate so the auto choice never construction-errors on storage.
    dense_bytes = nclasses * ndof * ndof * sizeof(TF)
    dense_bytes <= (dense.max_persistent_bytes * 3) ÷ 4 || return fallback
    expansion_order <= 3 && return dense                  # literature P <= 4
    expansion_order <= 7 || return fallback               # literature P >= 12
    return dense                                          # P = 8
end

_default_radix_options(::Type{TF}, expansion_order::Int, LH::Bool, device::Bool,
        nclasses::Int, ndof::Int) where TF =
    _radix_options_for(TF, _default_radix_m2l_strategy(TF, expansion_order, LH,
        device, nclasses, ndof))

# Each resident strategy is bound to the rotation operator its plan is built from.
_radix_options_for(::Type{TF}, m2l_strategy::PrecomputedFactoredYM2L) where TF =
    RadixLifecycleOptions(; precision=TF, operator=FactoredRotationM2L(),
        m2l_strategy)
_radix_options_for(::Type{TF}, m2l_strategy) where TF =
    RadixLifecycleOptions(; precision=TF, operator=MaterializedYRotationM2L(),
        m2l_strategy)

# Default separation policy: `HierarchicalRigidStencil`. The flat
# `ConstantPAnalyticStencil` classifier's accepted-offset set grows with `ell` (cell
# width shrinks at fixed epsilon), so its route count scales as
# `offsets(ell) x cells`, while the rigid stencil's offset set is level-invariant.
# An explicit `stencil_epsilon` selects the flat policy.
function _default_radix_policy(policy, P::Int, ::Type{TF}, LH::Bool, h0, ell::Int,
        device::Bool, stencil_epsilon, near_radius2, window_classes,
        level_radii2=nothing) where TF
    if policy !== nothing
        (stencil_epsilon === nothing && near_radius2 === nothing &&
            window_classes === nothing && level_radii2 === nothing) ||
            throw(ArgumentError("an explicit `policy` carries its own stencil " *
                "parameters; do not combine it with `stencil_epsilon`, " *
                "`near_radius2`, `level_radii2`, or `window_classes`"))
        return policy
    end
    K = window_classes === nothing ?
        (device ? RADIX_DEVICE_WINDOW_CLASSES : RADIX_HOST_WINDOW_CLASSES) :
        Int(window_classes)
    if stencil_epsilon !== nothing
        # explicit tolerance: the caller is asking for the flat analytic classifier
        (near_radius2 === nothing && level_radii2 === nothing) ||
            throw(ArgumentError("`stencil_epsilon` selects the flat " *
                "ConstantPAnalyticStencil, which has no `near_radius2` or " *
                "`level_radii2`; pass a HierarchicalRigidStencil `policy` for " *
                "an explicit tolerance with a rigid near set"))
        return ConstantPAnalyticStencil(
            ConstantPStencilConfig(P, TF(stencil_epsilon); lamb_helmholtz=LH))
    end
    q = near_radius2 === nothing ? RADIX_DEFAULT_NEAR_RADIUS2 : Int(near_radius2)
    if ell < 2
        # the first M2L level is 2; there is no hierarchy to walk below that
        (near_radius2 === nothing && level_radii2 === nothing) ||
            throw(ArgumentError("ell=$ell < 2 selects the flat " *
                "ConstantPAnalyticStencil, which has no `near_radius2` or " *
                "`level_radii2`; use ell >= 2 for the hierarchical policy"))
        return ConstantPAnalyticStencil(
            ConstantPStencilConfig(P, TF(1e-4); lamb_helmholtz=LH))
    end
    # Only the untouched default carries the level schedule: an
    # explicit `near_radius2` is honored as the uniform geometry the caller asked
    # for. The schedule covers M2L levels 2:ell and is non-increasing with depth.
    qs = if level_radii2 !== nothing
        Tuple(Int(x) for x in level_radii2)
    elseif near_radius2 === nothing && ell >= 3
        (RADIX_DEFAULT_COARSE_NEAR_RADIUS2,
            ntuple(_ -> RADIX_DEFAULT_NEAR_RADIUS2, ell - 2)...)
    else
        ()
    end
    eps = rigid_stencil_epsilon(P, h0, ell, q; lamb_helmholtz=LH, TF)
    return HierarchicalRigidStencil(
        ConstantPStencilConfig(P, TF(eps); lamb_helmholtz=LH);
        near_radius2=q, level_radii2=qs, window_classes=K)
end

"""
    RadixFMMCache(target_systems, source_systems=target_systems; kwargs...)

Construct the opt-in radix-grid / matrix-operator FMM cache. Eagerly
builds the capacity-sized resident state and every step-invariant operator table,
then runs the first state update from the systems' current positions — so the
first `fmm!(system, cache)` call is already the recurring fast path and no array
is reallocated over the cache's lifetime.

**Keyword arguments**

- `expansion_order::Int=4`: constant expansion order `P` on this path
- `ell::Int=4`: radix grid depth (leaf grid is `2^ell` cells per axis)
- `max_n_bodies::Int=n`: capacity bound; steps may use any `1 <= n <= max_n_bodies`
- `bounds=nothing`: `(x_min::SVector{3}, box_size)` fixed domain box; default derives
  a cube from the current positions inflated by `bounds_margin`. `box_size` may be a
  scalar (cubic) or a 3-vector/`NTuple{3}` of per-axis extents:
  the rectangular box is embedded in a virtual cube of half-width
  `h0 = maximum(box_size)/2`, per-axis extents snap up to whole leaf cells
  (readable as `cache.ell_axes` / `cache.box_extent`), and the per-axis in-box
  contract is enforced each step, on host and device caches alike.
- `bounds_margin::Real=0.05`: relative margin applied to derived bounds
- `lamb_helmholtz=nothing`: override the `has_vector_potential` inference
- `hessian::Bool=false`: allocate the 13-row output (potential + gradient +
  9-component hessian) and enable `fmm!(...; hessian=true)`. Off by default so
  the scalar path's output bandwidth is unchanged.
- `device::Bool=false`: run the lifecycle device-resident (requires a registered
  device backend, i.e. the KernelAbstractions extension)
- `options::RadixLifecycleOptions`: operator strategies/precision. Omitted, both
  are selected from measured rules (see below); passed explicitly, it is
  used verbatim. The resolved choice is readable as `cache.state.options`.
- `near_radius2`: rigid leaf near set `{o : |o|^2 <= near_radius2}` of the default
  hierarchical policy (default `$(RADIX_DEFAULT_NEAR_RADIUS2)`; `12` is the
  `theta=0.5` stencil and `3` the classic FMM one). Passing it explicitly also
  selects the *uniform* geometry, i.e. it drops the default level schedule below.
- `level_radii2`: per-M2L-level near radii, coarse to fine; must be
  non-increasing and end at `near_radius2`. Anchored either to levels `2:ell`
  (length `ell - 1`) or, for rectangular caches with trimmed coarse levels, to
  the active M2L levels `first_m2l_level:ell`; a `2:ell` schedule is sliced to
  the active range, which is the identity on cubic caches
- `window_classes`: route-window width of the hierarchical policy; defaults to the
  measured `$(RADIX_DEVICE_WINDOW_CLASSES)` on device and `$(RADIX_HOST_WINDOW_CLASSES)`
  on host. The flat policy (`stencil_epsilon`, or `ell < 2`) has no route windows
  and ignores it
- `stencil_epsilon::Real`: **selects the deprecated flat `ConstantPAnalyticStencil`**
  at this tolerance; omit it to get the hierarchical default
- `policy`: explicit `ConstantPAnalyticStencil` or `HierarchicalRigidStencil`
  (both run host- or device-resident). A policy carries its own stencil
  parameters, so combining it with `stencil_epsilon`, `near_radius2`,
  `level_radii2`, or `window_classes` throws; likewise `stencil_epsilon` (flat)
  and `ell < 2` (flat) reject `near_radius2` and `level_radii2`.

The default policy is [`HierarchicalRigidStencil`](@ref); its default geometry is `near_radius2=$(RADIX_DEFAULT_NEAR_RADIUS2)` with
the level schedule `($(RADIX_DEFAULT_COARSE_NEAR_RADIUS2), $(RADIX_DEFAULT_NEAR_RADIUS2), ...)`,
the fastest measured configuration inside the `P = 4` accuracy tolerance. Its tolerance is
derived by [`rigid_stencil_epsilon`](@ref) so the analytic accuracy gate is satisfied
by construction; pass `near_radius2=12` for a more accurate, slower
geometry. The flat
[`ConstantPAnalyticStencil`](@ref) is deprecated as a default but fully supported;
it is still used automatically when `ell < 2`, where there is no hierarchy to walk.

When no `options` are passed, precision and M2L strategy follow measured rules:

| selector | precision | M2L strategy |
|---|---|---|
| `expansion_order <= 3` (literature `P <= 4`) | `Float32` on a device cache, `Float64` on the host | dense |
| `expansion_order <= 7`, no Lamb-Helmholtz | `Float64` | dense |
| `expansion_order <= 7`, Lamb-Helmholtz | `Float64` | dense (host; a device cache always builds `ConcatenatedFixedZM2L`) |
| `expansion_order >= 8` (literature `P >= 12`) | `Float64` | precomputed-y |
| dense operator payload over its gate | unchanged | precomputed-y |

A host cache always defaults to `Float64`. A device cache defaults to `Float32`
only at literature `P <= 4`, where the stencil's own truncation error dominates
and Float32 matched Float64 accuracy (+0.03%) while running 1.14x faster; above
that the default is `Float64`. Pass explicit `options`
to run Float32 at a higher order. Dense trades a large construction cost for the
best steady state (~300-370
break-even steps in the benchmarked configuration), which suits this repeated-step cache; pass
`options=RadixLifecycleOptions(; m2l_strategy=PrecomputedFactoredYM2L(),
operator=FactoredRotationM2L())` for one-shot evaluation, or any explicit `options`
to bypass the rules entirely.

The domain box, `ell`, expansion order, and `max_n_bodies` are fixed for the
cache's lifetime; bodies leaving the box throw `ArgumentError` at the next step.
"""
function RadixFMMCache(target_systems, source_systems=target_systems;
        expansion_order::Integer=4,
        ell::Integer=4,
        max_n_bodies::Union{Nothing,Integer}=nothing,
        bounds=nothing,
        bounds_margin::Real=0.05,
        lamb_helmholtz::Union{Nothing,Bool}=nothing,
        hessian::Bool=false,
        device::Bool=false,
        options::Union{Nothing,RadixLifecycleOptions}=nothing,
        stencil_epsilon::Union{Nothing,Real}=nothing,
        near_radius2::Union{Nothing,Integer}=nothing,
        level_radii2=nothing,
        window_classes::Union{Nothing,Integer}=nothing,
        policy::Union{Nothing,ConstantPAnalyticStencil,HierarchicalRigidStencil}=nothing)
    targets = to_tuple(target_systems)
    sources = to_tuple(source_systems)
    _assert_radix_targets_are_sources(targets, sources)
    0 <= ell <= RADIX_GRID_MAX_ELL || throw(ArgumentError(
        "RadixFMMCache ell=$ell must lie in 0:$(RADIX_GRID_MAX_ELL) (64-bit Morton keys)"))
    _check_expansion_order(expansion_order)
    LH = lamb_helmholtz === nothing ? has_vector_potential(sources) : Bool(lamb_helmholtz)
    # B2M element resolution: one shared body type per cache, checked
    # here so a Point{Vortex} system with the χ channel off fails at construction
    # rather than inside a kernel (the LH=false chi buffer is 0×0).
    BT = body_type(first(sources))
    for system in sources
        body_type(system) === BT || throw(ArgumentError(
            "all source systems sharing a RadixFMMCache must report the same " *
            "body_type; got $(body_type(system)) and $BT"))
        strength_dims(system) == strength_dims(first(sources)) || throw(ArgumentError(
            "all source systems sharing a RadixFMMCache must report the same " *
            "strength_dims (the packed strength rows 5:4+strength_dims are shared)"))
    end
    if BT <: Point{Vortex} && !LH
        throw(ArgumentError(
            "Point{Vortex} sources require the Lamb-Helmholtz channel; construct " *
            "the cache with lamb_helmholtz=true (or leave it to be inferred from " *
            "has_vector_potential)"))
    end
    # Nearfield kernel resolution: one shared functor per
    # cache, resolved from the trait like body_type above; validated after the
    # options carry the final choice (below).
    dk_trait = direct_kernel(first(sources))
    for system in sources
        direct_kernel(system) == dk_trait || throw(ArgumentError(
            "all source systems sharing a RadixFMMCache must report the same " *
            "direct_kernel; got $(direct_kernel(system)) and $dk_trait"))
    end
    # Measured defaults. Precision depends only on the expansion order and
    # is needed for the bounds and stencil tolerance below; the strategy also depends
    # on the class count, so it is resolved once the policy is built.
    auto_options = options === nothing
    if auto_options
        options = RadixLifecycleOptions(;
            precision=_default_radix_precision(Int(expansion_order), device),
            m2l_strategy=ConcatenatedFixedZM2L())
    end
    TF = options.precision
    if device
        radix_device_backend_available() ||
            throw(ArgumentError("RadixFMMCache(device=true) requires a registered " *
                "device radix backend; load a backend extension " *
                "($(radix_device_status()))"))
    end
    for system in sources
        data_per_body(system) >= 4 + strength_dims(system) ||
            throw(ArgumentError("the radix path packs bodies as [x, y, z, radius, " *
                "strength..., extras...]; data_per_body(system) must be >= " *
                "4 + strength_dims(system)"))
    end
    dpb = maximum(data_per_body(system) for system in sources)

    n0 = get_n_bodies(sources)
    n0 > 0 || throw(ArgumentError("RadixFMMCache requires at least one body"))
    maxn = max_n_bodies === nothing ? n0 : Int(max_n_bodies)
    maxn >= n0 || throw(ArgumentError("max_n_bodies=$maxn is smaller than the current body count $n0"))

    if bounds === nothing
        x_min_data, x_max_data = _radix_bounds(sources, TF)
        center = (x_min_data + x_max_data) * TF(0.5)
        box = (x_max_data - x_min_data) * TF(0.5)
        h0 = max(box[1], box[2], box[3]) * (1 + TF(bounds_margin))
        h0 > zero(TF) ||
            throw(ArgumentError("bodies are degenerate (zero extent); pass explicit bounds=(x_min, box_size)"))
        x_min = center - SVector{3,TF}(h0, h0, h0)
        ell_axes = SVector(Int(ell), Int(ell), Int(ell))
        box_extent = SVector{3,TF}(2 * h0, 2 * h0, 2 * h0)
    else
        x_min = SVector{3,TF}(bounds[1])
        ell_axes, h0, box_extent = _resolve_radix_ell_axes(bounds[2], Int(ell), TF)
    end

    P = Int(expansion_order)
    stencil_policy = _default_radix_policy(policy, P, TF, LH, h0, Int(ell), device,
        stencil_epsilon, near_radius2, window_classes, level_radii2)
    hierarchical = stencil_policy isa HierarchicalRigidStencil
    # Active-level trimming: hierarchical caches retain node
    # levels root_level:ell and run M2L on levels first_m2l_level:ell (the
    # flat-top root level plus the transition levels). Cubic caches normally
    # degenerate to root_level = 1 with an empty flat-top (first_m2l_level = 2,
    # the untrimmed schedule); flat-policy caches stay untrimmed (root_level = 0).
    if hierarchical
        hierarchical_tables, hierarchical_level_class_of, hierarchical_level_radii2,
            root_level, first_m2l_level =
            _hierarchical_scheduled_tables(stencil_policy, Int(ell), ell_axes)
        _verify_hierarchical_classifier!(h0, Int(ell), stencil_policy,
            hierarchical_tables, ell_axes, root_level, first_m2l_level,
            hierarchical_level_radii2)
        class_level, class_offset, effective_offsets =
            _hierarchical_class_metadata(hierarchical_tables, Int(ell),
                first_m2l_level)
        accepted, rejected = effective_offsets, hierarchical_tables.near_offsets
    else
        hierarchical_tables = nothing
        hierarchical_level_class_of = Array{Int32}(undef, 0, 0, 0)
        hierarchical_level_radii2 = Int[]
        root_level, first_m2l_level = 0, 2
        class_level, class_offset, effective_offsets =
            Int32[], Matrix{Int32}(undef, 3, 0), SVector{3,Int}[]
        accepted, rejected =
            classify_radix_stencil_offsets(h0, Int(ell), stencil_policy.config)
    end

    max_cells = _radix_level_node_capacity(Int(ell), ell_axes, Int(ell), maxn)
    max_nodes = sum(_radix_level_node_capacity(L, ell_axes, Int(ell), max_cells)
        for L in root_level:Int(ell))
    # init=0 covers the zero-M2L degenerate hierarchy (first_m2l_level == ell+1,
    # empty range): no M2L level, so no per-level node bound needed.
    max_level_nodes = Int(ell) >= 2 ? maximum(
        (_radix_level_node_capacity(L, ell_axes, Int(ell), max_cells)
         for L in (hierarchical ? first_m2l_level : 2):Int(ell)); init=0) : 0
    route_capacity = hierarchical ?
        min(min(stencil_policy.window_classes,
                length(hierarchical_tables.push_offsets)) * max_level_nodes,
            max_level_nodes * max_level_nodes) :
        min(length(accepted), max_cells) * max_cells
    direct_capacity = max_cells * min(length(rejected), max_cells)

    basis_info = OperatorBasisInfo(CompressedComplexBasis(), P, Val(LH))

    if auto_options
        options = _default_radix_options(TF, P, LH, device, length(accepted),
            _dense_m2m_dof(basis_info, Val(LH)))
    end
    options = _options_with_body_type(options, BT)
    if dk_trait != _default_direct_kernel(BT)
        # explicit trait choice; a conflicting explicit options choice is an error
        (!options.direct_kernel_explicit ||
            options.direct_kernel == dk_trait) || throw(ArgumentError(
            "options.direct_kernel=$(options.direct_kernel) conflicts with the " *
            "direct_kernel(system) trait $dk_trait"))
        options = _options_with_direct_kernel(options, dk_trait)
    end
    dk = options.direct_kernel
    isbits(dk) || throw(ArgumentError(
        "direct_kernel must be an isbits functor (GPU-compilable, no references); " *
        "got $(typeof(dk))"))
    if dk isa AbstractRegularizedVortex
        kname = nameof(typeof(dk))
        BT <: Point{Vortex} || throw(ArgumentError(
            "$kname requires body_type Point{Vortex}; got $BT"))
        first_extra_row = 5 + element_strength_dims(BT)
        dk.sigma_row >= first_extra_row || throw(ArgumentError(
            "$kname sigma_row=$(dk.sigma_row) points inside the packed $BT " *
            "strength rows 5:$(first_extra_row - 1); the smoothing radius σ " *
            "must sit in an extra-state row >= $first_extra_row"))
        for system in sources
            dk.sigma_row <= data_per_body(system) || throw(ArgumentError(
                "$kname sigma_row=$(dk.sigma_row) exceeds " *
                "data_per_body=$(data_per_body(system)) for $(typeof(system)); " *
                "every source system must carry the smoothing radius σ in packed " *
                "row sigma_row"))
        end
    end
    if device
        cache = _radix_cache_device_build(sources, P, Int(ell), x_min, h0, maxn,
            options, stencil_policy, accepted, rejected, max_cells, max_nodes,
            route_capacity, direct_capacity, basis_info, Val(LH);
            hierarchical_tables, hierarchical_level_class_of, hessian,
            ell_axes, box_extent, root_level, first_m2l_level)
        cache.built = true
        return cache
    end

    grid = _allocate_host_radix_grid(TF, x_min, h0, Int(ell), maxn, max_cells, max_nodes)
    multipoles = _host_flat_buffer(TF, basis_info, max_nodes)
    locals = _host_flat_buffer(TF, basis_info, max_nodes)
    source_bodies = Matrix{TF}(undef, dpb, maxn)
    output = zeros(TF, hessian ? 13 : 4, maxn)
    route_levels = Vector{Int}(undef, route_capacity)
    route_offsets = Matrix{Int}(undef, 3, route_capacity)
    route_targets = Vector{Int}(undef, route_capacity)
    route_sources = Vector{Int}(undef, route_capacity)
    direct_targets = Vector{Int}(undef, direct_capacity)
    direct_sources = Vector{Int}(undef, direct_capacity)
    n_edges_capacity = max(max_nodes - 1, 0)
    m2m_parent_routes = Vector{Int}(undef, n_edges_capacity)
    m2m_child_routes = Vector{Int}(undef, n_edges_capacity)
    l2l_parent_routes = Vector{Int}(undef, n_edges_capacity)
    l2l_child_routes = Vector{Int}(undef, n_edges_capacity)

    invariant = OperatorInvariantCache(TF, basis_info)
    # The concatenated and grouped-factored selections share the bounded
    # per-column concatenated engine in hierarchical mode.  Precomputed-y and
    # dense retain their specialized host plans and refresh those plans once per
    # route window.
    hierarchical_specialized = hierarchical &&
        options.m2l_strategy isa Union{PrecomputedFactoredYM2L,DenseTranslationM2L}
    if hierarchical && !hierarchical_specialized &&
            options.operator isa FactoredRotationM2L
        @debug "hierarchical FactoredRotationM2L selection runs the shared " *
            "concatenated per-column engine under MaterializedYRotationM2L " *
            "(mathematically equivalent: concat is the factored composition " *
            "with a shared z-block); options.operator still reports " *
            "FactoredRotationM2L"
    end
    workspace_strategy = hierarchical ?
        (hierarchical_specialized ? options.m2l_strategy : ConcatenatedFixedZM2L()) :
        options.m2l_strategy
    workspace_operator = hierarchical ?
        (hierarchical_specialized ? options.operator : MaterializedYRotationM2L()) :
        options.operator
    scratch = _radix_cache_workspace(TF, basis_info, multipoles, Int(ell), h0,
        max_cells, max_nodes, route_capacity, accepted, invariant,
        workspace_strategy, workspace_operator;
        hierarchical_noffsets=hierarchical ? length(hierarchical_tables.push_offsets) : 0,
        ell_axes, first_level=root_level)
    counters = RadixTransferCounters()
    occupancy = hierarchical ? RadixLevelOccupancy(Int(ell);
        max_bytes=stencil_policy.dense_occupancy_max_bytes,
        max_dense_ell=stencil_policy.dense_occupancy_max_ell) : nothing
    hierarchical_apply_plan = hierarchical ? scratch.m2l_concat : nothing
    hierarchical_ctx = hierarchical ? HostHierarchicalM2LContext(
        hierarchical_tables, hierarchical_level_class_of, occupancy,
        class_level, class_offset,
        effective_offsets, hierarchical_apply_plan, stencil_policy.window_classes,
        first_m2l_level,
        zeros(Int, Int(ell) + 2), 0, zeros(Int, Int(ell) + 1), 0,
        false, zeros(UInt64, 5), zeros(UInt64, Int(ell) + 1)) : nothing
    state = DeviceResidentRadixState{TF,CompressedComplexBasis,LH}(
        grid, hierarchical_ctx, source_bodies,
        grid.perm, grid.body_system, grid.body_index,
        grid.perm, grid.body_system, grid.body_index, grid.node_levels,
        grid.cell_centers, grid.cell_ranges,
        m2m_parent_routes, m2m_child_routes, l2l_parent_routes, l2l_child_routes,
        multipoles, locals, route_levels, route_offsets, route_targets, route_sources,
        direct_targets, direct_sources, output,
        invariant, scratch, counters, options,
        RadixStepCounts(0, 0, 0, 0, 0);
    )

    G = 1 << Int(ell)
    source_buffers = Tuple(Matrix{TF}(undef, data_per_body(system), maxn) for system in sources)
    cache = RadixFMMCache{TF,LH}(
        P, Int(ell), x_min, h0, ell_axes, box_extent, root_level, maxn, device,
        hessian, options, stencil_policy,
        accepted, rejected, max_cells, max_nodes, route_capacity, direct_capacity,
        state, hierarchical ? zeros(Int32, 0, 0, 0) : zeros(Int32, G, G, G),
        Vector{SVector{3,Int}}(undef, max_cells),
        zeros(Int, Int(ell) + 2), Vector{UInt64}(undef, maxn), Vector{Int}(undef, maxn),
        zeros(Int, 256), zeros(Int, 256), source_buffers, nothing, nothing,
        length(sources), false, 0,
        snapshot_locked_radix_settings(),
    )
    _update_host_radix_state!(cache, to_tuple(sources))
    cache.built = true
    return cache
end

# Refresh the per-level M2M/L2L group edge columns from the freshly updated grid.
# Nodes are level-major, so each level's children occupy one contiguous index block.
function _refresh_resident_stage_groups!(ws::ResidentOperatorWorkspace{TF},
        grid::DeviceRadixGrid, level_offsets::Vector{Int}) where TF
    ell = grid.ell
    # the workspace group count encodes the trimmed level range: groups cover
    # levels first_level:ell only
    first_level = ell - length(ws.m2m_groups)
    0 <= first_level <= ell && length(ws.l2l_groups) == length(ws.m2m_groups) ||
        throw(ArgumentError("resident cache workspace does not match grid depth ell=$ell"))
    for (gi, parent_level) in enumerate((ell - 1):-1:first_level)
        _refresh_group_edges!(ws.m2m_groups[gi], grid, level_offsets, parent_level + 1, :m2m)
    end
    for (gi, child_level) in enumerate((first_level + 1):ell)
        _refresh_group_edges!(ws.l2l_groups[gi], grid, level_offsets, child_level, :l2l)
    end
    return ws
end

function _refresh_group_edges!(group::ResidentOperatorGroup, grid::DeviceRadixGrid{TF},
        level_offsets::Vector{Int}, child_level::Integer, kind::Symbol) where TF
    first_child = level_offsets[child_level + 1] + 1
    last_child = level_offsets[child_level + 2]
    n = last_child - first_child + 1
    n <= length(group.source_idx) ||
        throw(AssertionError("resident $kind group at child level $child_level exceeded its capacity"))
    group.count[] = n
    # function barrier: group fields are Any-typed
    _refresh_group_edges_kernel!(group.source_idx, group.target_idx, group.phis,
        group.thetas, grid.parent_index, grid.node_centers, first_child, last_child,
        kind === :m2m)
    return group
end

function _refresh_group_edges_kernel!(source_idx, target_idx, phis::AbstractVector{TF},
        thetas, parent_index, node_centers, first_child::Int, last_child::Int,
        child_to_parent::Bool) where TF
    i = 0
    @inbounds for child in first_child:last_child
        parent = parent_index[child]
        i += 1
        if child_to_parent
            dx = node_centers[1, parent] - node_centers[1, child]
            dy = node_centers[2, parent] - node_centers[2, child]
            dz = node_centers[3, parent] - node_centers[3, child]
            source_idx[i] = child
            target_idx[i] = parent
        else
            dx = node_centers[1, child] - node_centers[1, parent]
            dy = node_centers[2, child] - node_centers[2, parent]
            dz = node_centers[3, child] - node_centers[3, parent]
            source_idx[i] = parent
            target_idx[i] = child
        end
        _, theta, phi = cartesian_to_spherical(SVector{3,TF}(dx, dy, dz))
        phis[i] = TF(phi)
        thetas[i] = TF(theta)
    end
    return nothing
end

function _refresh_radix_coords!(coords::Vector{SVector{3,Int}}, cell_keys,
        n_cells::Int, ell::Int)
    @inbounds for cell in 1:n_cells
        coords[cell] = morton_decode(cell_keys[cell], ell)
    end
    return coords
end

function _refresh_radix_tree_routes!(m2m_parent::Vector{Int}, m2m_child::Vector{Int},
        l2l_parent::Vector{Int}, l2l_child::Vector{Int}, parent_index, n_edges::Int,
        n_root_nodes::Int=1)
    # edges are the children of the retained levels: the
    # first n_root_nodes nodes are roots with parent_index 0 and carry no edge
    @inbounds for edge in 1:n_edges
        node = edge + n_root_nodes
        parent = parent_index[node]
        m2m_parent[edge] = parent
        m2m_child[edge] = node
        l2l_parent[edge] = parent
        l2l_child[edge] = node
    end
    return nothing
end

function _pack_radix_source_bodies!(source_bodies::AbstractMatrix{TF}, perm, body_system,
        body_index, source_buffers::Tuple, n::Int) where TF
    # all data_per_body rows are carried, including radius row 4;
    # systems narrower than the packed matrix are zero-padded
    nrows = size(source_bodies, 1)
    @inbounds for sorted_i in 1:n
        global_i = perm[sorted_i]
        isys = body_system[global_i]
        ibody = body_index[global_i]
        src = source_buffers[isys]
        nsys = min(size(src, 1), nrows)
        for row in 1:nsys
            source_bodies[row, sorted_i] = src[row, ibody]
        end
        for row in (nsys + 1):nrows
            source_bodies[row, sorted_i] = zero(TF)
        end
    end
    return source_bodies
end

"""
    recenter!(cache::RadixFMMCache, systems; bounds=nothing, padding=0.05)

Re-anchor the cache's fixed domain box. The box is part of
the cache's invariant contract: bodies leaving it make the next `fmm!` throw,
and `fmm!` never recenters implicitly. When the physical domain should move or
resize, the consumer calls `recenter!` explicitly between evaluations, before
the next `fmm!`.

- `bounds = (x_min, box_size)` is the deterministic fast path (recommended for
  consumers that already track their domain); caller-supplied bounds are final
  and are **not** padded. `box_size` may be a scalar (cubic rebuild) or a
  3-vector (rectangular rebuild), regardless of the cache's current
  shape.
- With `bounds = nothing`, the union bounds of all live bodies are derived:
  host-resident systems through `get_position`, device-resident systems by a
  device min/max reduction over their persistent packed source buffers
  (refilled via `source_to_buffer!` first; only six extrema scalars reach the
  host). `padding` is a nonnegative fraction of the tight cube's side added on
  each face: `x_min = lo - padding*L_tight`, `L = (1 + 2*padding)*L_tight`.
  A rectangular cache (non-uniform `ell_axes`) applies the same convention per
  axis and then pads each shorter extent symmetrically to the power-of-two leaf
  count required by the shared cubic leaf width. Thus the snapped rectangular
  box remains centered on the tight cloud; rectangularity (and, for a
  similar-shaped cloud, the resolved `ell_axes`) is preserved.

Validation errors (`ArgumentError`) — an empty system when the bounds are
derived (with explicit `bounds`, only a system set with no live bodies at all),
non-finite or nonpositive bounds, negative padding, a changed system count, a
live count above `max_n_bodies`, or a body outside explicit bounds — leave the
cache unmodified and usable.

Implementation (geometry-rebuild fallback): the
geometry-dependent state — stencil classification, operator tables, grid
keying, device geometry — is re-derived by re-running the construction path at
the new bounds with the cache's own parameters (`expansion_order`, `ell`,
`max_n_bodies`, `hessian`, `options`, and the cache's policy re-anchored to the
new box: a `HierarchicalRigidStencil` gets its box-derived tolerance re-derived
via `rigid_stencil_epsilon` at the new `h0` — the rigid near set and level
schedule are preserved exactly — while a flat `ConstantPAnalyticStencil` keeps
its tolerance and re-classifies; a hierarchical policy carrying custom
source/chi strengths incompatible with the re-derived tolerance fails the
construction accuracy gate loudly rather than running wrong), then swapped into
the existing cache object in place; the object identity consumers hold remains
valid, and
step-count prefixes restart so no stale state is trusted. Consequences to plan
around: `recenter!` costs about as much as cache construction, transiently
holds a second set of buffers (device caches: transiently ~2x device memory),
and restarts the transfer counters (a construction-equivalent event — route and
operator uploads recur here, never in ordinary steps). The zero-cost
alternative — normalized unit-cube internal coordinates making the operator
tables box-size-invariant — is not implemented.
"""
function recenter!(cache::RadixFMMCache{TF,LH}, systems;
        bounds=nothing, padding::Real=0.05) where {TF,LH}
    systems_tuple = to_tuple(systems)
    length(systems_tuple) == cache.n_systems || throw(ArgumentError(
        "recenter! got $(length(systems_tuple)) systems for a cache built with " *
        "$(cache.n_systems); the system set is part of the cache contract"))
    padding >= 0 || throw(ArgumentError("recenter! padding must be nonnegative"))
    rectangular = cache.ell_axes != SVector(cache.ell, cache.ell, cache.ell)
    n = get_n_bodies(systems_tuple)
    n <= cache.max_n_bodies || throw(ArgumentError(
        "recenter! live body count n=$n exceeds the cache capacity " *
        "max_n_bodies=$(cache.max_n_bodies)"))
    if bounds === nothing
        lo, hi = _recenter_union_bounds(cache, systems_tuple)
        (all(isfinite, lo) && all(isfinite, hi)) || throw(ArgumentError(
            "recenter! derived non-finite body bounds; check body positions"))
        if rectangular
            # a rectangular cache keeps per-axis tight extents
            # (same margin convention as the cube, applied per axis), so the
            # rebuild resolves vector bounds and rectangularity is preserved
            ext_tight = hi .- lo
            (ext_tight[1] > zero(TF) && ext_tight[2] > zero(TF) &&
                ext_tight[3] > zero(TF)) || throw(ArgumentError(
                "recenter! on a rectangular cache derived a degenerate " *
                "(zero-extent) axis; pass explicit bounds=(x_min, box_size)"))
            L_new = (1 + 2 * TF(padding)) .* ext_tight
            # `_resolve_radix_ell_axes` pads short axes upward to power-of-two
            # leaf counts. Apply that padding equally on both faces for derived
            # bounds; keeping the raw lower face would shift the leaf lattice and
            # can inflate occupied/direct/M2L counts for a centered cloud.
            center_new = (lo + hi) / 2
            _, _, snapped_extent =
                _resolve_radix_ell_axes(L_new, cache.ell, TF)
            x_min_new = center_new - snapped_extent / 2
            L_new = snapped_extent
        else
            L_tight = max(hi[1] - lo[1], hi[2] - lo[2], hi[3] - lo[3])
            L_tight > zero(TF) || throw(ArgumentError(
                "recenter! derived a degenerate (zero-extent) body cloud; pass " *
                "explicit bounds=(x_min, box_size)"))
            x_min_new = lo .- TF(padding) * L_tight
            L_new = (1 + 2 * TF(padding)) * L_tight
        end
    else
        x_min_new = SVector{3,TF}(bounds[1])
        # caller-supplied bounds are final: a scalar box_size rebuilds cubic, a
        # 3-vector rebuilds rectangular, regardless of the cache's current shape
        L_new = bounds[2] isa Real ? TF(bounds[2]) : SVector{3,TF}(bounds[2])
        all(isfinite, x_min_new) && all(isfinite, L_new) || throw(ArgumentError(
            "recenter! bounds must be finite"))
        all(>(zero(TF)), L_new) ||
            throw(ArgumentError("recenter! box_size must be positive"))
    end
    # Build the replacement first: any failure (empty system, body outside the
    # requested bounds, capacity) leaves the original cache untouched.
    fresh = RadixFMMCache(systems_tuple, systems_tuple;
        expansion_order=cache.expansion_order, ell=cache.ell,
        max_n_bodies=cache.max_n_bodies, bounds=(x_min_new, L_new),
        lamb_helmholtz=LH, hessian=cache.hessian, device=cache.device,
        options=cache.options,
        policy=_recentered_policy(cache.policy, cache.expansion_order,
            maximum(L_new) / 2, cache.ell, TF, LH))
    for f in fieldnames(RadixFMMCache)
        setfield!(cache, f, getfield(fresh, f))
    end
    return cache
end

_config_normalization(::ConstantPStencilConfig{TF,LH,N}) where {TF,LH,N} = N

# The flat analytic stencil keeps its tolerance and re-classifies at the new
# box; the hierarchical rigid stencil keeps its near set and level schedule
# exactly and re-derives the box-scaled tolerance those sets realize (the
# construction-time _verify_hierarchical_classifier! gate requires it).
_recentered_policy(policy::ConstantPAnalyticStencil, P, h0_new, ell, ::Type, LH) = policy

function _recentered_policy(policy::HierarchicalRigidStencil, P, h0_new, ell,
        ::Type{TF}, LH) where TF
    cfg = policy.config
    eps_new = rigid_stencil_epsilon(cfg.P_phi, h0_new, ell, policy.near_radius2;
        lamb_helmholtz=LH, TF)
    config = ConstantPStencilConfig(cfg.P_phi, TF(eps_new), cfg.source_strength;
        chi_strength=cfg.chi_strength, lamb_helmholtz=LH,
        normalization=_config_normalization(cfg))
    return HierarchicalRigidStencil(config;
        near_radius2=policy.near_radius2, level_radii2=policy.level_radii2,
        window_classes=policy.window_classes,
        dense_occupancy_max_bytes=policy.dense_occupancy_max_bytes,
        dense_occupancy_max_ell=policy.dense_occupancy_max_ell)
end

function _recenter_union_bounds(cache::RadixFMMCache{TF}, systems::Tuple) where TF
    lox = TF(Inf); loy = TF(Inf); loz = TF(Inf)
    hix = -TF(Inf); hiy = -TF(Inf); hiz = -TF(Inf)
    for (isys, system) in enumerate(systems)
        n_sys = get_n_bodies(system)
        n_sys > 0 || throw(ArgumentError(
            "recenter! requires at least one live body in every system " *
            "(system $isys is empty)"))
        if residency(system) isa DeviceResident
            cache.device || throw(ArgumentError(
                "DeviceResident system $isys requires a device=true cache"))
            buf = cache.device_ctx.device_sources[isys]
            _fill_device_source_buffer!(view(buf, :, 1:n_sys), system)
            # fixed-geometry reductions (see _device_row_extrema): a view
            # reduction compiles one kernel per distinct n_sys on Metal
            x_lo, x_hi = _device_row_extrema(buf, 1, n_sys)
            y_lo, y_hi = _device_row_extrema(buf, 2, n_sys)
            z_lo, z_hi = _device_row_extrema(buf, 3, n_sys)
            lox = min(lox, TF(x_lo)); loy = min(loy, TF(y_lo)); loz = min(loz, TF(z_lo))
            hix = max(hix, TF(x_hi)); hiy = max(hiy, TF(y_hi)); hiz = max(hiz, TF(z_hi))
        else
            for i in 1:n_sys
                x = get_position(system, i)
                lox = min(lox, TF(x[1])); loy = min(loy, TF(x[2])); loz = min(loz, TF(x[3]))
                hix = max(hix, TF(x[1])); hiy = max(hiy, TF(x[2])); hiz = max(hiz, TF(x[3]))
            end
        end
    end
    return SVector{3,TF}(lox, loy, loz), SVector{3,TF}(hix, hiy, hiz)
end

"""
    update_radix_state!(cache, systems)

Refresh every step-varying part of the cache's resident state from the systems'
current positions and strengths: grid (fixed Morton domain), packed source
bodies, occupancy map, M2L routes + direct pairs, tree edges, per-level operator
group columns, and the step counts. No array is reallocated. Returns the cache.
Under the hierarchical policy the M2L routes are generated window by window
inside the M2L stage instead, which also sets `counts.n_routes` and the
context's route totals; until then `counts.n_routes` is 0.
A device cache is refreshed by the backend extension (`_RADIX_DEVICE_UPDATE_HOOK`);
a host cache by `_update_host_radix_state!`.
"""
function update_radix_state!(cache::RadixFMMCache, systems)
    if cache.device
        hook = _RADIX_DEVICE_UPDATE_HOOK[]
        hook === nothing && throw(RadixDeviceUnavailable(radix_device_status()))
        return hook(cache, to_tuple(systems))
    end
    return _update_host_radix_state!(cache, to_tuple(systems))
end

function _update_host_radix_state!(cache::RadixFMMCache{TF,LH}, systems::Tuple) where {TF,LH}
    length(systems) == cache.n_systems ||
        throw(ArgumentError("cache was built for $(cache.n_systems) source systems, got $(length(systems))"))
    state = cache.state
    grid = state.grid
    ctx = state.interaction_list
    hierarchical = ctx isa HostHierarchicalM2LContext
    profiling = hierarchical && ctx.profile_stages
    profiling && fill!(ctx.update_stage_ns, 0)
    n = get_n_bodies(systems)
    n > 0 || throw(ArgumentError("update_radix_state! requires at least one body"))
    n <= cache.max_n_bodies ||
        throw(ArgumentError("n=$n exceeds the cache capacity max_n_bodies=$(cache.max_n_bodies)"))
    _assert_radix_positions_in_box(systems, cache.x_min, cache.box_extent)

    t_stage = profiling ? time_ns() : UInt64(0)
    update_radix_grid!(grid, systems, cache.body_keys, cache.sort_scratch,
        cache.sort_counts, cache.sort_offsets, cache.level_offsets,
        cache.root_level; ell_axes=cache.ell_axes)
    profiling && (ctx.update_stage_ns[1] = time_ns() - t_stage)
    n_cells = grid.n_cells
    n_nodes = cache.level_offsets[end]

    for (isys, system) in enumerate(systems)
        source_to_buffer!(cache.source_buffers[isys], system, 1:get_n_bodies(system))
    end
    _pack_radix_source_bodies!(state.source_bodies, grid.perm, grid.body_system,
        grid.body_index, cache.source_buffers, n)
    # an inadequate hierarchical geometry demotes to the all-direct
    # zero-M2L cache; its constructor already ran the full refresh from these
    # systems, so nothing is left to do for this step.
    if _direct_kernel_geometry_gate!(cache,
            state.options.direct_kernel, state.source_bodies, n) === :alldirect
        return _alldirect_geometry_fallback!(cache, systems)
    end

    resize!(cache.coords, n_cells)
    _refresh_radix_coords!(cache.coords, grid.cell_keys, n_cells, grid.ell)
    if hierarchical
        t_stage = profiling ? time_ns() : UInt64(0)
        copyto!(ctx.level_offsets, cache.level_offsets)
        refresh_radix_level_occupancy!(ctx.occupancy, grid, cache.level_offsets)
        profiling && (ctx.update_stage_ns[2] = time_ns() - t_stage)
    else
        refresh_cell_at!(cache.cell_at, grid.cell_keys, n_cells, grid.ell)
    end

    plan = state.scratch.m2l_concat
    if hierarchical
        t_stage = profiling ? time_ns() : UInt64(0)
        n_direct = build_hierarchical_direct_pairs!(state.direct_targets,
            state.direct_sources, ctx, grid, n_cells)
        profiling && (ctx.update_stage_ns[3] = time_ns() - t_stage)
        n_routes = 0    # counted by the M2L stage, which generates the windows
    else
        n_routes, n_direct = build_radix_routes!(
            state.route_levels, state.route_offsets, state.route_targets,
            state.route_sources, plan === nothing ? nothing : plan.route_class,
            state.direct_targets, state.direct_sources,
            cache.accepted_offsets, cache.rejected_offsets, cache.cell_at, cache.coords,
            grid.leaf_to_node, grid.ell, n_cells, RadixRouteSelection(),
        )
    end

    !hierarchical && plan isa ResidentM2LFactoredPlan && _refresh_factored_m2l_routes!(plan,
        state.route_sources::Vector{Int}, state.route_targets::Vector{Int}, n_routes)
    !hierarchical && plan isa ResidentM2LPrecomputedYPlan && _refresh_precomputed_y_m2l_routes!(plan,
        state.route_sources::Vector{Int}, state.route_targets::Vector{Int}, n_routes)
    !hierarchical && plan isa ResidentM2LDensePlan && _refresh_dense_m2l_routes!(plan,
        state.route_sources::Vector{Int}, state.route_targets::Vector{Int}, n_routes)

    t_stage = profiling ? time_ns() : UInt64(0)
    # multi-root tree edges: every node at root_level is a
    # root, so the edge count is n_nodes - n_root_nodes (an untrimmed tree has
    # one level-0 root, n_nodes - 1)
    n_root_nodes = cache.level_offsets[cache.root_level + 2]
    n_edges = max(n_nodes - n_root_nodes, 0)
    resize!(state.m2m_parent_routes, n_edges)
    resize!(state.m2m_child_routes, n_edges)
    resize!(state.l2l_parent_routes, n_edges)
    resize!(state.l2l_child_routes, n_edges)
    _refresh_radix_tree_routes!(state.m2m_parent_routes, state.m2m_child_routes,
        state.l2l_parent_routes, state.l2l_child_routes, grid.parent_index, n_edges,
        n_root_nodes)

    _refresh_resident_stage_groups!(state.scratch, grid, cache.level_offsets)
    profiling && (ctx.update_stage_ns[5] = time_ns() - t_stage)

    counts = state.counts
    counts.n_bodies = n
    counts.n_cells = n_cells
    counts.n_nodes = n_nodes
    counts.n_routes = n_routes
    counts.n_direct = n_direct

    cache.step += 1
    return cache
end

function _refresh_precomputed_y_m2l_routes!(plan::ResidentM2LPrecomputedYPlan,
        route_sources::Vector{Int}, route_targets::Vector{Int}, n_routes::Int)
    fill!(plan.offset_counts, 0)
    fill!(plan.angle_counts, 0)
    @inbounds for i in 1:n_routes
        offset = Int(plan.route_class[i])
        plan.offset_counts[offset] += 1
        plan.angle_counts[plan.offset_to_angle[offset]] += 1
    end
    # Prefixes follow the immutable angle-major / offset-minor order.
    cursor = 1
    @inbounds for angle in eachindex(plan.angle_counts)
        plan.angle_starts[angle] = cursor
        for p in plan.angle_offset_starts[angle]:(plan.angle_offset_starts[angle + 1] - 1)
            offset = plan.angle_offsets[p]
            plan.offset_starts[offset] = cursor
            cursor += plan.offset_counts[offset]
        end
    end
    plan.angle_starts[end] = cursor
    plan.offset_starts[end] = cursor
    cursor - 1 == n_routes || throw(AssertionError("precomputed-y route histogram mismatch"))

    # Reuse offset_starts as insertion cursors.  Traversing the original route
    # prefix makes ordering stable within every offset range.
    @inbounds for i in 1:n_routes
        offset = Int(plan.route_class[i])
        dst = plan.offset_starts[offset]
        plan.packed_sources[dst] = route_sources[i]
        plan.packed_targets[dst] = route_targets[i]
        plan.packed_phis[dst] = plan.offset_phis[offset]
        plan.offset_starts[offset] = dst + 1
    end
    # Restore the public starts in place; array identity never changes.
    cursor = 1
    @inbounds for angle in eachindex(plan.angle_counts)
        for p in plan.angle_offset_starts[angle]:(plan.angle_offset_starts[angle + 1] - 1)
            offset = plan.angle_offsets[p]
            plan.offset_starts[offset] = cursor
            cursor += plan.offset_counts[offset]
        end
    end
    plan.offset_starts[end] = cursor
    return plan
end

function _refresh_dense_m2l_routes!(plan::ResidentM2LDensePlan,
        route_sources::Vector{Int}, route_targets::Vector{Int}, n_routes::Int)
    0 <= n_routes <= length(plan.route_class) || throw(ArgumentError(
        "dense M2L route count $n_routes exceeds plan capacity $(length(plan.route_class))"))
    fill!(plan.class_counts, 0)
    @inbounds for i in 1:n_routes
        cls = Int(plan.route_class[i])
        1 <= cls <= length(plan.class_counts) || throw(AssertionError(
            "dense M2L route $i has invalid class $cls"))
        plan.class_counts[cls] += 1
    end
    cursor = 1
    @inbounds for cls in eachindex(plan.class_counts)
        count = plan.class_counts[cls]
        count <= plan.class_capacities[cls] || throw(AssertionError(
            "dense M2L class $cls count $count exceeds capacity $(plan.class_capacities[cls])"))
        plan.class_starts[cls] = cursor
        cursor += count
    end
    plan.class_starts[end] = cursor
    cursor - 1 == n_routes || throw(AssertionError("dense M2L route histogram mismatch"))

    # Reuse starts as stable insertion cursors, then restore the public prefixes.
    @inbounds for i in 1:n_routes
        cls = Int(plan.route_class[i])
        dst = plan.class_starts[cls]
        plan.packed_sources[dst] = route_sources[i]
        plan.packed_targets[dst] = route_targets[i]
        plan.class_starts[cls] = dst + 1
    end
    cursor = 1
    @inbounds for cls in eachindex(plan.class_counts)
        plan.class_starts[cls] = cursor
        cursor += plan.class_counts[cls]
    end
    plan.class_starts[end] = cursor
    return plan
end

function _refresh_factored_m2l_routes!(plan::ResidentM2LFactoredPlan{R,G},
        route_sources::Vector{Int}, route_targets::Vector{Int}, n_routes::Int) where {R,G}
    for group in plan.groups
        group.count[] = 0
    end
    route_class = plan.route_class::Vector{Int32}
    @inbounds for i in 1:n_routes
        group = plan.groups[route_class[i]]
        j = group.count[] + 1
        (group.source_idx::Vector{Int})[j] = route_sources[i]
        (group.target_idx::Vector{Int})[j] = route_targets[i]
        group.count[] = j
    end
    return plan
end

# Preallocated per-switch-layout scatter buffers for the recurring finalize,
# one set per distinct layout. A caller that alternates layouts every step
# (a self-induction pass with probe systems as extra targets, then a
# sources-only pass on the main system alone) keeps both sets instead of
# reallocating capacity-sized buffers at each switch.
function _radix_cache_target_buffers!(cache::RadixFMMCache{TF}, switches::Tuple) where TF
    tb = cache.target_buffers
    if !(tb isa Dict)
        tb = Dict{Any,Any}(); cache.target_buffers = tb
    end
    return get!(tb, switches) do
        Tuple(zeros(TF, target_buffer_rows(switch), cache.max_n_bodies) for switch in switches)
    end
end
