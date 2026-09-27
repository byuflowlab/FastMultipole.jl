#------- stage 0: user entry (port of `fmm!`, src/fmm.jl:873-899) -------#
#
# CUDA's stage 0 is `fmm!(targets, sources, cache::RadixFMMCache)`
# (src/fmm.jl:873), whose device branch calls `_radix_cache_device_step!`
# (src/fmm.jl:898). That call resolves to CUDA's definition only because
# `translate_batched_cuda.jl` is runtime-`include`d INTO the FastMultipole
# module, overwriting the throwing stub at
# `src/translate_batched_resident.jl:3425`.
#
# DEVIATION (forced, entry symbol only). A package extension cannot use that
# mechanism: defining `FastMultipole._radix_cache_device_step!` here with the
# same signature is method piracy over a method the parent module already owns,
# and Julia's own diagnostic for it is "incremental compilation may be fatally
# broken". So this entry point carries a different NAME. Everything inside it is
# a statement-for-statement port of `fmm!`'s prologue, in CUDA's order, with the
# same argument checks, the same error text, and the same `DerivativesSwitch`
# construction. Wiring real `fmm!` dispatch requires a backend trait hook in
# `src/` (the stub would ask the cache for its backend instead of throwing);
# that is a `src/` change and is deliberately NOT made here.
#
# Two checks from `fmm!` are absent, both because the KA cache cannot reach the
# state they guard:
#   * (the former SFS check is gone with the SFS pass; consumers run their own
#     near-field pass through `nearfield_pass`, which
#     reads, and throws CUDA's message.
#   * the adaptive branch (src/fmm.jl:900-916) -- `ka_update_radix_state!`
#     throws on a non-`nothing` `cache.adaptive`.
"""
    ka_fmm!(target_systems, source_systems, cache; kwargs...)
    ka_fmm!(systems, cache; kwargs...)

KA stage 0: the backend-agnostic entry point corresponding to
`fmm!(targets, sources, cache::RadixFMMCache)` (`src/fmm.jl:873`). Validates
arguments, builds the `DerivativesSwitch` tuple, and dispatches to
`ka_radix_cache_device_step!`.
"""
function ka_fmm!(target_systems, source_systems,
        cache::FastMultipole.RadixFMMCache{TF,LH};
        scalar_potential::Bool=false, gradient::Bool=true, hessian=false,
        lamb_helmholtz::Union{Nothing,Bool}=nothing,
        workgroup=KA_AUTO_WORKGROUP) where {TF,LH}
    targets = FastMultipole.to_tuple(target_systems)
    sources = FastMultipole.to_tuple(source_systems)
    split = FastMultipole._split_radix_systems(cache.n_systems, targets, sources)
    hessian_v = FastMultipole.to_vector(hessian, length(targets))
    any(hessian_v) && !cache.hessian && throw(ArgumentError(
        "hessian output requested but this RadixFMMCache was built with " *
        "hessian=false (4-row output); construct RadixFMMCache(...; hessian=true)"))
    lamb_helmholtz === nothing || Bool(lamb_helmholtz) == LH || throw(ArgumentError(
        "lamb_helmholtz=$(lamb_helmholtz) conflicts with the cache's lamb_helmholtz=$LH; " *
        "the Lamb-Helmholtz channel is fixed at cache construction"))
    !FastMultipole.has_vector_potential(split.main) || LH || throw(ArgumentError(
        "source systems carry a vector potential but the cache was built with " *
        "lamb_helmholtz=false; rebuild the cache with lamb_helmholtz=true"))
    cache.device || throw(ArgumentError(
        "ka_fmm! requires a device-resident cache built by ka_radix_cache_device_build"))

    all_switches = FastMultipole.DerivativesSwitch(
        FastMultipole.to_vector(scalar_potential, length(targets)),
        FastMultipole.to_vector(gradient, length(targets)),
        hessian_v, targets)
    switches = Tuple(all_switches[i] for i in split.main_index)
    extra_switches = Tuple(all_switches[i] for i in split.extra_target_index)
    ka_radix_cache_device_step!(cache, split.main, switches; workgroup,
        extra_targets=split.extra_targets, extra_target_switches=extra_switches,
        extra_sources=split.extra_sources, self_induce=split.self_induce)
    return cache
end

ka_fmm!(systems, cache::FastMultipole.RadixFMMCache; kwargs...) =
    ka_fmm!(systems, systems, cache; kwargs...)


#------- stage 1: argument + trait validation -------#
#
# CUDA's stage 1 is the prologue of `RadixFMMCache`
# (src/translate_batched_resident.jl:2445-2519 plus the strategy check at
# :2549) -- everything the constructor decides BEFORE it looks at where the
# bodies are. It resolves four traits off the source systems (Lamb-Helmholtz,
# body type, strength dims, direct kernel), rejects every argument combination
# the radix path cannot represent, and picks the first-pass options so that
# `TF` exists for the geometry that follows.
#
# This is a statement-for-statement port in CUDA's own order, with the same
# error text. It returns the resolved values instead of assigning them into a
# constructor's local scope, because KA's stages 2-6 are not written yet and
# the caller has to thread them by hand until they are.
#
# THREE DEVIATIONS, all forced:
#
#  1. Device availability. CUDA checks `cuda_radix_available()` behind
#     `device=true` (:2510). The KA analogue is that a `KernelAbstractions`
#     backend was actually passed; a KA cache is device-resident by
#     construction, so there is no `device=false` branch to guard.
#
#  2. (RESOLVED, the port KA port) SFS used to be refused here because
#     `ka_radix_cache_device_build` had no `sfs_ctx` and no SFS pass. Both now
#     exist, so this is CUDA's validation (:2449-2464) statement for statement:
#     the hessian requirement, the packed-row-8 sigma requirement, and the
#     `sfs_active_row` bounds. Only two deviations remain (1 and 3).
#
#  3. The trait tail is NOT here. CUDA's `dk` checks at :2600-2626 (isbits,
#     the `AbstractRegularizedVortex` sigma_row bound, and
#     `_assert_device_kernel_policy`) read `options.direct_kernel` AFTER the
#     policy-dependent options substitution, so they belong to stage 5, not
#     stage 1. What is portable without the policy -- the per-system
#     `direct_kernel` trait agreement -- is done here, exactly as CUDA does it
#     at :2492-2498.
#
# What stage 1 does NOT do, on purpose: derive `ell`. There is no auto-`ell`
# rule in FastMultipole on either side (PIPELINE.md stage 4); it is a kwarg
# defaulting to 4, and FLOWVPM's own rule lives in the caller.

"""
    ka_validate_radix_arguments(backend, target_systems, source_systems; kwargs...)

KA stage 1: the argument and trait validation `RadixFMMCache` performs before
it computes any geometry (`src/translate_batched_resident.jl:2445-2519`).

Returns a `NamedTuple` with the resolved
`(; targets, sources, LH, BT, dk_trait, options, auto_options, TF, P, dpb, n0,
maxn, hessian)`, which the
remaining (unported) construction stages consume. Throws the same
`ArgumentError`s, with the same messages, that the host constructor throws.
"""
function ka_validate_radix_arguments(backend, target_systems, source_systems=target_systems;
        expansion_order::Integer=4,
        max_n_bodies::Union{Nothing,Integer}=nothing,
        lamb_helmholtz::Union{Nothing,Bool}=nothing,
        hessian::Bool=false,
        options::Union{Nothing,FastMultipole.RadixLifecycleOptions}=nothing)
    targets = FastMultipole.to_tuple(target_systems)
    sources = FastMultipole.to_tuple(source_systems)
    FastMultipole._assert_radix_targets_are_sources(targets, sources)


    LH = lamb_helmholtz === nothing ?
        FastMultipole.has_vector_potential(sources) : Bool(lamb_helmholtz)

    # B2M element resolution: one shared body type per cache.
    BT = FastMultipole.body_type(first(sources))
    for system in sources
        FastMultipole.body_type(system) === BT || throw(ArgumentError(
            "all source systems sharing a RadixFMMCache must report the same " *
            "body_type; got $(FastMultipole.body_type(system)) and $BT"))
        FastMultipole.strength_dims(system) == FastMultipole.strength_dims(first(sources)) ||
            throw(ArgumentError(
            "all source systems sharing a RadixFMMCache must report the same " *
            "strength_dims (the packed strength rows 5:4+strength_dims are shared)"))
    end
    if (BT <: FastMultipole.Point{FastMultipole.Vortex} ||
            BT <: FastMultipole.Point{FastMultipole.SourceVortex}) && !LH
        throw(ArgumentError(
            "$BT sources require the Lamb-Helmholtz channel; construct " *
            "the cache with lamb_helmholtz=true (or leave it to be inferred from " *
            "has_vector_potential)"))
    end

    # Nearfield kernel resolution (): one shared functor per
    # cache. The post-options checks on it are stage 5 -- DEVIATION 3.
    dk_trait = FastMultipole.direct_kernel(first(sources))
    for system in sources
        FastMultipole.direct_kernel(system) == dk_trait || throw(ArgumentError(
            "all source systems sharing a RadixFMMCache must report the same " *
            "direct_kernel; got $(FastMultipole.direct_kernel(system)) and $dk_trait"))
    end

    # Measured defaults (024/028). Precision depends only on the expansion order
    # and is needed for the bounds and stencil tolerance in stage 2; the strategy
    # also depends on the class count, so it is re-resolved once the policy is
    # built (stage 5).
    auto_options = options === nothing
    if auto_options
        options = FastMultipole.RadixLifecycleOptions(;
            precision=FastMultipole._default_radix_precision(Int(expansion_order)),
            m2l_strategy=FastMultipole.ConcatenatedFixedZM2L())
    end
    TF = options.precision

    # DEVIATION 1 (see above): the KA analogue of `cuda_radix_available()`.
    backend isa KA.Backend || throw(ArgumentError(
        "the KA radix path requires a KernelAbstractions backend; got " *
        "$(typeof(backend))"))

    for system in sources
        FastMultipole.data_per_body(system) >= 4 + FastMultipole.strength_dims(system) ||
            throw(ArgumentError("the radix path packs bodies as [x, y, z, radius, " *
                "strength..., extras...]; data_per_body(system) must be >= " *
                "4 + strength_dims(system)"))
    end
    dpb = maximum(FastMultipole.data_per_body(system) for system in sources)

    n0 = FastMultipole.get_n_bodies(sources)
    n0 > 0 || throw(ArgumentError("RadixFMMCache requires at least one body"))
    maxn = max_n_bodies === nothing ? n0 : Int(max_n_bodies)
    maxn >= n0 || throw(ArgumentError(
        "max_n_bodies=$maxn is smaller than the current body count $n0"))

    # Out of source order (CUDA runs this at :2549, after the geometry block)
    # but argument-only: the hierarchical path routes the concatenated and
    # grouped-factored selections through the bounded concat engine, which would
    # otherwise silently accept a strategy the flat path rejects.
    #
    # DEVIATION. CUDA also accepts `PrecomputedFactoredYM2L` here; KA does not.
    # That plan's per-step refresh (`_cuda_refresh_precomputed_y_m2l_routes!`,
    # cuda:4047) has no KA counterpart and `ka_radix_cache_workspace` pins
    # `ConcatenatedFixedZM2L`, so no `ResidentM2LPrecomputedYPlan` is ever
    # constructible on this path. It was refused only at
    # `ka_radix_cache_device_build`, several stages downstream, which let a
    # `:precomputed_y` setting (a legal `RadixFMMSettings` value in FLOWVPM,
    # FLOWVPM_fmm_radix.jl:243) past the stage that is meant to state KA's
    # envelope. The builder keeps its own guard for callers that bypass
    # validation.
    options.m2l_strategy isa FastMultipole.ConcatenatedFixedZM2L ||
        throw(ArgumentError(
        "the KA RadixFMMCache supports ConcatenatedFixedZM2L; got " *
        "$(typeof(options.m2l_strategy)) (DenseTranslationM2L and the factored " *
        "strategies are host-only)"))

    return (; targets, sources, LH, BT, dk_trait, options, auto_options, TF,
        P=Int(expansion_order), dpb, n0, maxn, hessian)
end



#------- stage 2: root geometry -------#
#
# CUDA's stage 2 is the geometry block of `RadixFMMCache`
# (src/translate_batched_resident.jl:2521-2535): the first point at which the
# constructor looks at WHERE the bodies are. It turns the caller's `bounds`
# (or, absent them, the body extent) into the four root-box quantities every
# later stage indexes against -- `x_min`, `h0`, `ell_axes` and `box_extent`.
#
# This is a statement-for-statement port in CUDA's own order, sharing CUDA's
# own helpers: `_radix_bounds` (src/tree_batched.jl:117/137) and
# `_resolve_radix_ell_axes` (:2190/:2196) are backend-independent host code
# that computes host scalars, so a KA copy of them would be duplication, not a
# port. Only the surrounding branch is reproduced here.
#
# ONE DEVIATION, and it is a subtraction: CUDA's derived branch also names
# `center` and `box`, which are consumed only by the two lines below them and
# are dead at the end of the block. They are not returned.
#
# WHY THERE IS NO DEVICE REDUCTION. The `bounds === nothing` branch walks the
# bodies one at a time through `get_position`, which is scalar indexing if the
# system is backed by device arrays. That is CUDA's behavior too, not a KA
# regression, and the production path never reaches it: FLOWVPM always passes
# an explicit `bounds` (its own padded, optionally center-snapped box --
# FLOWVPM_fmm_radix.jl:521-523, :539, :560-562), so the constructor takes the
# `else` branch, which touches no body data at all. A device min/max reduction
# would be a new capability on a path production does not use; if the derived
# branch ever becomes hot for a device-resident field, that is its own task.

"""
    ka_radix_geometry(sources, TF, ell; bounds=nothing, bounds_margin=0.05)

KA stage 2: the root-box geometry `RadixFMMCache` derives immediately after
argument validation (`src/translate_batched_resident.jl:2521-2535`).

With `bounds === nothing` the root cube is the body bounding box inflated by
`bounds_margin`; otherwise `bounds = (x_min, box_size)` is taken as given and
`box_size` is resolved into per-axis depths. Returns a `NamedTuple`
`(; x_min, h0, ell_axes, box_extent)`. `TF` and `ell` come from stage 1 and the
caller respectively.
"""
function ka_radix_geometry(sources, ::Type{TF}, ell::Integer;
        bounds=nothing, bounds_margin::Real=0.05) where TF
    if bounds === nothing
        x_min_data, x_max_data = FastMultipole._radix_bounds(sources, TF)
        center = (x_min_data + x_max_data) * TF(0.5)
        box = (x_max_data - x_min_data) * TF(0.5)
        h0 = max(box[1], box[2], box[3]) * (1 + TF(bounds_margin))
        h0 > zero(TF) || throw(ArgumentError(
            "bodies are degenerate (zero extent); pass explicit bounds=(x_min, box_size)"))
        x_min = center - SVector{3,TF}(h0, h0, h0)
        ell_axes = SVector(Int(ell), Int(ell), Int(ell))
        box_extent = SVector{3,TF}(2 * h0, 2 * h0, 2 * h0)
    else
        x_min = SVector{3,TF}(bounds[1])
        ell_axes, h0, box_extent =
            FastMultipole._resolve_radix_ell_axes(bounds[2], Int(ell), TF)
    end
    return (; x_min, h0, ell_axes, box_extent)
end


#------- stage 5: stencil policy, hierarchical tables, options finalization -------#
#
# CUDA's stage 5 is the constructor's policy block
# (src/translate_batched_resident.jl:2546-2578) plus the options/`dk` tail
# (:2594-2626): it picks the stencil policy, builds and verifies the
# hierarchical level schedule, produces the accepted/rejected offset sets every
# later stage sizes and routes against, and only then -- once the class count
# exists -- finalizes `options` and runs the kernel checks stage 1 had to defer.
#
# It consumes stage 1's NamedTuple and stage 2's geometry directly, because
# that is what the constructor's local scope hands these statements.
#
# THE ONE REORDERING. CUDA runs the capacity block (:2580-2592, KA stage 6)
# BETWEEN the tables and the options tail. The two are independent -- the
# capacities read `ell`, `ell_axes`, `maxn`, `stencil_policy.window_classes`
# and accepted/rejected, none of which the options tail touches, and the
# options tail reads `length(accepted)` and `basis_info`, neither of which the
# capacities produce -- so KA runs the tail here and leaves the capacities
# whole for stage 6. Nothing observable depends on the order.
#
# TWO DEVIATIONS, both refusals in the style of stage 1's `sfs`:
#
#  1. Adaptive octree. CUDA's `adaptive !== nothing` block (:2628-2679) guards
#     a host- and CUDA-only lifecycle that KA does not implement at all.
#     `adaptive` is refused outright rather than validated.
#
#  2. `_assert_device_kernel_policy` is called with `device=true`
#     unconditionally: a KA cache is device-resident by construction, exactly
#     as in stage 1's deviation 1. The `device` argument to
#     `_default_radix_policy` is `true` for the same reason, which is what
#     selects `RADIX_DEVICE_WINDOW_CLASSES` as the default window width.
#
# `_default_radix_policy`, `_hierarchical_scheduled_tables`,
# `_verify_hierarchical_classifier!`, `_hierarchical_class_metadata`,
# `classify_radix_stencil_offsets` and `_default_radix_options` are shared host
# code producing host tables -- as in stage 2, a KA copy would be duplication
# rather than a port, so they are called, not reimplemented.

"""
    ka_radix_stencil_policy(v, ell, h0, ell_axes; kwargs...)

KA stage 5: the stencil policy, hierarchical level schedule and options
finalization `RadixFMMCache` performs after the geometry
(`src/translate_batched_resident.jl:2546-2578`, `:2594-2626`).

`v` is stage 1's NamedTuple; `h0` and `ell_axes` come from stage 2. The
`policy` / `stencil_epsilon` / `near_radius2` / `window_classes` /
`level_radii2` kwargs are the constructor's own, with the constructor's
meanings. Returns a `NamedTuple` carrying the policy, the hierarchical tables
and their metadata, the accepted/rejected offset sets, `basis_info`, and the
finalized `options` / `dk`.
"""
function ka_radix_stencil_policy(v, ell::Integer, h0, ell_axes;
        policy=nothing, stencil_epsilon=nothing, near_radius2=nothing,
        window_classes=nothing, level_radii2=nothing, adaptive=nothing)
    TF, LH, BT = v.TF, v.LH, v.BT
    P = v.P

    # DEVIATION 1 (see above): KA has no adaptive octree lifecycle.
    adaptive === nothing || throw(ArgumentError(
        "the KA radix path has no adaptive octree lifecycle (tasks 039-041 are " *
        "host- and CUDA-only); construct with adaptive=nothing"))

    # DEVIATION 2: device=true, a KA cache being device-resident by construction.
    stencil_policy = FastMultipole._default_radix_policy(policy, P, TF, LH, h0,
        Int(ell), true, stencil_epsilon, near_radius2, window_classes, level_radii2)
    hierarchical = stencil_policy isa FastMultipole.HierarchicalRigidStencil
    # Active-level trimming (): hierarchical caches retain node
    # levels root_level:ell and run M2L on levels first_m2l_level:ell. Cubic
    # caches degenerate to root_level = 1 with an empty flat-top; flat-policy
    # caches stay untrimmed (root_level = 0).
    if hierarchical
        hierarchical_tables, hierarchical_level_class_of, hierarchical_level_radii2,
            root_level, first_m2l_level =
            FastMultipole._hierarchical_scheduled_tables(stencil_policy, Int(ell), ell_axes)
        FastMultipole._verify_hierarchical_classifier!(h0, Int(ell), stencil_policy,
            hierarchical_tables, ell_axes, root_level, first_m2l_level,
            hierarchical_level_radii2)
        class_level, class_offset, effective_offsets =
            FastMultipole._hierarchical_class_metadata(hierarchical_tables, Int(ell),
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
            FastMultipole.classify_radix_stencil_offsets(h0, Int(ell), stencil_policy.config)
    end

    basis_info = FastMultipole.OperatorBasisInfo(
        FastMultipole.CompressedComplexBasis(), P, Val(LH))

    options = v.options
    if v.auto_options
        options = FastMultipole._default_radix_options(TF, P, LH, true,
            length(accepted), FastMultipole._dense_m2m_dof(basis_info, Val(LH)))
    end
    options = FastMultipole._options_with_body_type(options, BT)
    if v.dk_trait != FastMultipole._default_direct_kernel(BT)
        # explicit trait choice; a conflicting explicit options choice is an error
        (options.direct_kernel == FastMultipole._default_direct_kernel(BT) ||
            options.direct_kernel == v.dk_trait) || throw(ArgumentError(
            "options.direct_kernel=$(options.direct_kernel) conflicts with the " *
            "direct_kernel(system) trait $(v.dk_trait)"))
        options = FastMultipole._options_with_direct_kernel(options, v.dk_trait)
    end
    dk = options.direct_kernel
    isbits(dk) || throw(ArgumentError(
        "direct_kernel must be an isbits functor (GPU-compilable, no references); " *
        "got $(typeof(dk))"))
    if dk isa FastMultipole.AbstractRegularizedVortex
        kname = nameof(typeof(dk))
        BT <: FastMultipole.Point{FastMultipole.Vortex} || throw(ArgumentError(
            "$kname requires body_type Point{Vortex}; got $BT"))
        for system in v.sources
            dk.sigma_row <= FastMultipole.data_per_body(system) || throw(ArgumentError(
                "$kname sigma_row=$(dk.sigma_row) exceeds " *
                "data_per_body=$(FastMultipole.data_per_body(system)) for " *
                "$(typeof(system)); every source system must carry the smoothing " *
                "radius sigma in packed row sigma_row"))
        end
    end
    FastMultipole._assert_device_kernel_policy(true, dk, hierarchical)

    return (; stencil_policy, hierarchical, hierarchical_tables,
        hierarchical_level_class_of, hierarchical_level_radii2,
        root_level, first_m2l_level, class_level, class_offset,
        accepted, rejected, basis_info, options, dk)
end


#------- stage 6: capacity sizing -------#
#
# CUDA's stage 6 is the constructor's capacity block
# (src/translate_batched_resident.jl:2580-2592): the five numbers that fix
# every persistent device allocation the cache will ever make. They are sized
# to `maxn` (the capacity contract: live `np` may vary below it with no
# reallocation), not to the live body count.
#
# A statement-for-statement port with NO deviations. It reads stage 1's `maxn`,
# stage 2's `ell_axes` and stage 5's policy/offset sets, and computes nothing
# device-side -- these are host integers.
#
# This is the stage the pipeline note called ungated by construction: until it
# existed, every suite handed KA known-good capacities copied off a host cache,
# so no gate could catch a KA sizing bug because KA did no sizing.

"""
    ka_radix_capacities(v, ell, ell_axes, s)

KA stage 6: the five persistent capacities `RadixFMMCache` derives from the
resolved policy (`src/translate_batched_resident.jl:2580-2592`).

`v` is stage 1's NamedTuple, `ell_axes` stage 2's, and `s` stage 5's. Returns
`(; max_cells, max_nodes, max_level_nodes, route_capacity, direct_capacity)`.
"""
function ka_radix_capacities(v, ell::Integer, ell_axes, s)
    L_max = Int(ell)
    max_cells = FastMultipole._radix_level_node_capacity(L_max, ell_axes, L_max, v.maxn)
    max_nodes = sum(FastMultipole._radix_level_node_capacity(L, ell_axes, L_max, max_cells)
        for L in s.root_level:L_max)
    # init=0 covers the zero-M2L degenerate hierarchy (first_m2l_level == ell+1,
    # empty range -- the port): no M2L level, so no per-level node bound needed.
    max_level_nodes = L_max >= 2 ? maximum(
        (FastMultipole._radix_level_node_capacity(L, ell_axes, L_max, max_cells)
         for L in (s.hierarchical ? s.first_m2l_level : 2):L_max); init=0) : 0
    route_capacity = s.hierarchical ?
        min(min(s.stencil_policy.window_classes,
                length(s.hierarchical_tables.push_offsets)) * max_level_nodes,
            max_level_nodes * max_level_nodes) :
        min(length(s.accepted), max_cells) * max_cells
    direct_capacity = max_cells * min(length(s.rejected), max_cells)
    return (; max_cells, max_nodes, max_level_nodes, route_capacity, direct_capacity)
end


