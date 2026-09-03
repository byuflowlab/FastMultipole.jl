#=##############################################################################
Task 047: consolidated settings surface for the GPU/radix production path.

Every mechanism tunable of the radix lifecycle is a process-global `Ref`
(most defined in the lazily-include'd translate_batched_cuda.jl). This file
provides the single documented, validated access surface plus the
construction-lock contract:

- `radix_settings()`            -> NamedTuple of current values (defined Refs only)
- `radix_setting(name)`         -> current value
- `set_radix_setting!(name, v)` -> validated write (throws on unknown name,
                                   bad type, or out-of-domain value)
- `radix_setting_lock(name)`    -> :construction | :runtime

Lock contract (047): `:construction` settings are read at cache/device-context
construction or inside the graph-captured lifecycle body (baked at record
time; a re-record happens only on an occupancy epoch), so flipping them after
a `RadixFMMCache` is built used to silently keep the old mechanism — the
documented hazard. Both cache constructors now snapshot the
construction-locked settings (`snapshot_locked_radix_settings`), and the
device step entry verifies the snapshot (`verify_locked_radix_settings`),
throwing a loud, actionable error on drift. `:runtime` settings are read
per-step outside capture and may be flipped freely.

Read-timing classification per tunable is from the 047 audit (recorded in
MATRIX_OPERATOR_REFACTOR/047-impl-production-settings-hardening.md).
=###############################################################################

struct RadixSettingSpec
    lock::Symbol            # :construction or :runtime
    validate::Function      # value -> nothing (throws ArgumentError on bad value)
    doc::String
end

_rs_bool(v) = v isa Bool ? nothing : throw(ArgumentError("expected Bool, got $(typeof(v))"))
_rs_posint(v) = (v isa Integer && !(v isa Bool) && v >= 1 && v <= typemax(Int)) ? nothing :
    throw(ArgumentError("expected a non-Bool Integer in 1:$(typemax(Int)), got $(repr(v))"))
_rs_nonnegint(v) = (v isa Integer && !(v isa Bool) && v >= 0 && v <= typemax(Int)) ? nothing :
    throw(ArgumentError("expected a non-Bool Integer in 0:$(typemax(Int)), got $(repr(v))"))
_rs_cuda_threads(v) = (v isa Int && 32 <= v <= 1024 && v % 32 == 0) ? nothing :
    throw(ArgumentError("expected Int warp multiple in 32:1024, got $(repr(v))"))
_rs_enum(vals) = v -> v in vals ? nothing : throw(ArgumentError("expected one of $(vals), got $(repr(v))"))
_rs_ka_workgroup(v) = (v isa Int && (v == 0 || (32 <= v <= 1024 && v % 32 == 0))) ? nothing :
    throw(ArgumentError("expected 0 (auto) or an Int warp multiple in 32:1024, got $(repr(v))"))

# KA arm of the uniform lifecycle. Unlike every other tunable here this one has
# no CUDA-side Ref to reference, so it lives in always-loaded src: the whole
# point is to select a NON-CUDA implementation of the per-step body, and the
# selection has to be readable whether or not translate_batched_cuda.jl was
# ever included.
const RADIX_KA_LIFECYCLE = Ref(false)

# Workgroup size for the auto-tuned KA launches, for the same reason as
# RADIX_KA_LIFECYCLE above: a KA tunable has to be settable on a build where
# translate_batched_cuda.jl never loads. `0` means "let the backend decide" --
# FastMultipoleKAExt.resolve_workgroup then picks per backend (256 on
# CUDA/ROCm/oneAPI, 64 on Metal and CPU), which is the point: 64 suits Metal's
# small threadgroups but wastes scheduler slots on an A100. Does not affect the
# launches where `workgroup` is a team size the kernel's @localmem extents are
# declared against (b2m, l2b, nearfield, adaptive m2t/s2l).
const KA_WORKGROUP = Ref(0)

# All-pairs direct arm (task 053). Swaps `ka_lifecycle_body!` for a single
# O(np^2) kernel and skips the grid/route refresh entirely. Off by default and
# selected only by an explicit write here: below some np the arm is faster
# (np ~ 6e4 on Metal for one wake), but that crossover has not been measured
# on CUDA and no automatic dispatch rule is built on it. Lives here for the
# same reason as
# RADIX_KA_LIFECYCLE: it selects a non-CUDA implementation and must be
# readable on a build where translate_batched_cuda.jl never loads.
const RADIX_DIRECT_ARM = Ref(false)

"""
    ka_radix_lifecycle!(state)

Run the uniform per-step lifecycle over `state` with KernelAbstractions
kernels. Overloaded by `FastMultipoleKAExt`; this stub is what a caller hits
when `:RADIX_KA_LIFECYCLE` was set without `KernelAbstractions` loaded.
"""
function ka_radix_lifecycle!(state)
    throw(ArgumentError(
        "radix setting :RADIX_KA_LIFECYCLE is on but the KernelAbstractions " *
        "extension is not loaded; `using KernelAbstractions` (plus the backend " *
        "package) before the first device step, or set it back to false"))
end

const RADIX_SETTING_SPECS = Dict{Symbol,RadixSettingSpec}(
    # ---- tree/refresh --------------------------------------------------------
    :RADIX_CUDA_COUNTING_SORT => RadixSettingSpec(:construction, _rs_bool,
        "Use the device counting-sort refresh when ell <= RADIX_CUDA_COUNTING_SORT_MAX_ELL (histogram sized at construction)."),
    :RADIX_CUDA_COUNTING_SORT_MAX_ELL => RadixSettingSpec(:construction, _rs_nonnegint,
        "Max uniform depth for the counting-sort refresh (8^ell histogram allocated at construction)."),
    # ---- nearfield -----------------------------------------------------------
    :CUDA_NEARFIELD_GH_MODE => RadixSettingSpec(:construction,
        _rs_enum((:shipped, :reduced, :fp32, :reduced_fp32, :lut)),
        "g/h evaluation mode for the regularized nearfield (037f; :fp32 default; :lut needs the construction-built table)."),
    :CUDA_NEARFIELD_BINNING => RadixSettingSpec(:construction,
        _rs_enum((:unbinned, :classsplit, :ballot, :classsplit_ballot)),
        "Pair-class binning of the split vortex nearfield kernels (baked into the captured graph)."),
    :CUDA_NEARFIELD_SHAPE => RadixSettingSpec(:construction,
        _rs_enum((:pairs, :fused_cta, :fused_srclanes, :fused_packed)),
        "Nearfield kernel shape (041e target-owned CSR shapes need construction-time arming: U-CSR buffers are sized 0 under :pairs)."),
    :CUDA_NEARFIELD_FUSED_MIN_BODIES => RadixSettingSpec(:construction, _rs_nonnegint,
        "Body-count threshold above which the fused nearfield shapes engage."),
    :CUDA_NEARFIELD_SUBSORT => RadixSettingSpec(:runtime, _rs_bool,
        "Sub-Morton ordering inside cells during the (uncaptured) host refresh; flippable per step."),
    :CUDA_TWOPASS_PASS2_QUEUED => RadixSettingSpec(:construction, _rs_bool,
        "TwoPassVortex pass-2 ballot queue (captured)."),
    :CUDA_TWOPASS_TARGET_AABB_PRUNE => RadixSettingSpec(:construction, _rs_bool,
        "TwoPassVortex target-AABB pruning (captured)."),
    :CUDA_NEARFIELD_PAIR_AABB => RadixSettingSpec(:construction, _rs_bool,
        "Per-pair AABB gap predicate in the split nearfield (captured)."),
    :CUDA_SYMMETRIC_NEARFIELD => RadixSettingSpec(:construction, _rs_bool,
        "Symmetric nearfield pair walk (buffers sized at construction; throws at launch if armed late)."),
    :SYMMETRIC_CUDA_MAX_CELL_BODIES => RadixSettingSpec(:runtime, _rs_posint,
        "Cell-size cap for symmetric-pair eligibility (host refresh; flippable per step)."),
    :DIRECT_CUDA_MAX_BLOCKS => RadixSettingSpec(:construction, _rs_posint,
        "Grid cap for the direct nearfield kernels (captured)."),
    # ---- M2L strategies ------------------------------------------------------
    :FACTORED_CUDA_WHOLE_PASS => RadixSettingSpec(:runtime, _rs_bool,
        "Whole-pass chunked launch for the factored M2L (never graph-captured; flippable)."),
    :FACTORED_CUDA_CHUNK => RadixSettingSpec(:construction, _rs_posint,
        "Route chunk of the factored whole-pass scratch (sized at construction)."),
    :PRECOMPUTED_CUDA_WHOLE_PASS => RadixSettingSpec(:runtime, _rs_bool,
        "Whole-pass chunked launch for the precomputed-y M2L (never graph-captured; flippable)."),
    :PRECOMPUTED_CUDA_CHUNK => RadixSettingSpec(:construction, _rs_posint,
        "Route chunk of the precomputed-y whole-pass scratch (sized at construction)."),
    :DENSE_CUDA_WHOLE_PASS => RadixSettingSpec(:construction, _rs_bool,
        "Whole-pass launch for the dense M2L (captured)."),
    :DENSE_CUDA_CHUNK => RadixSettingSpec(:construction, _rs_posint,
        "Dense M2L slab capacity (allocated at construction)."),
    :DENSE_CUDA_FUSED => RadixSettingSpec(:runtime, _rs_bool,
        "Fused per-route dense M2L kernel. Runtime-flippable, but it also gates graph eligibility: flipping it OFF disables capture; window-cache validity follows it."),
    :DENSE_CUDA_FUSED_MAX_BLOCKS => RadixSettingSpec(:construction, _rs_posint,
        "Grid cap of the fused dense kernel (captured)."),
    :DENSE_CUDA_TILED => RadixSettingSpec(:construction, _rs_bool,
        "Tiled variant of the fused dense kernel (captured)."),
    :DENSE_CUDA_TILED_MIN_ROUTES => RadixSettingSpec(:construction, _rs_nonnegint,
        "Route-count threshold for the tiled dense kernel (captured)."),
    :DENSE_CUDA_TILED_THREADS => RadixSettingSpec(:construction, _rs_cuda_threads,
        "Block size of the tiled dense kernel (captured)."),
    :DENSE_CUDA_TILED_MAX_BLOCKS => RadixSettingSpec(:construction, _rs_posint,
        "Grid cap of the tiled dense kernel (captured)."),
    :DENSE_CUDA_TENSOR_FORMAT => RadixSettingSpec(:construction,
        _rs_enum((:off, :fp16, :bf16)),
        "Tensor-core operator format for the dense M2L (low-precision operator copies allocated at construction)."),
    # ---- lifecycle/orchestration --------------------------------------------
    :CUDA_CACHED_WINDOWS => RadixSettingSpec(:runtime, _rs_bool,
        "Occupancy-epoch window caching (checked per step; also gates graph eligibility)."),
    :CUDA_GRAPH_LIFECYCLE => RadixSettingSpec(:runtime, _rs_bool,
        "CUDA-graph capture/replay of the lifecycle (checked per step at entry)."),
    :CUDA_OVERLAP_NEARFIELD => RadixSettingSpec(:construction, _rs_bool,
        "Nearfield side-stream overlap (first statement of the captured body)."),
    :RADIX_KA_LIFECYCLE => RadixSettingSpec(:runtime, _rs_bool,
        "Run the uniform per-step lifecycle with KernelAbstractions kernels instead of the native CUDA ones (checked per step at entry; requires the KA extension loaded)."),
    :RADIX_DIRECT_ARM => RadixSettingSpec(:runtime, _rs_bool,
        "Evaluate the step as a single all-pairs O(np^2) direct kernel instead of the FMM lifecycle, skipping the grid and route refresh (KA device path only; checked per step at entry). Opt-in: nothing selects it automatically."),
    :KA_WORKGROUP => RadixSettingSpec(:runtime, _rs_ka_workgroup,
        "Workgroup size for auto-tuned KA launches; 0 (default) resolves per backend. Team-size launches (b2m/l2b/nearfield/adaptive) are unaffected."),
    # ---- host GEMM thresholds ------------------------------------------------
    :FACTORED_Y_GEMM_MIN_COLS => RadixSettingSpec(:runtime, _rs_nonnegint,
        "Host factored-y GEMM column threshold."),
    :FACTORED_Y_GEMM_MIN_DIM => RadixSettingSpec(:runtime, _rs_nonnegint,
        "Host factored-y GEMM dimension threshold."),
    :PRECOMPUTED_Y_GEMM_MIN_COLS => RadixSettingSpec(:runtime, _rs_nonnegint,
        "Host precomputed-y GEMM column threshold."),
)

"Return the `Ref` behind a setting name, or `nothing` if its defining file
(a device backend extension) has not been loaded yet."
function _radix_setting_ref(name::Symbol)
    haskey(RADIX_SETTING_SPECS, name) ||
        throw(ArgumentError("unknown radix setting $(repr(name)); known: $(sort!(collect(keys(RADIX_SETTING_SPECS))))"))
    isdefined(FastMultipole, name) || return nothing
    return getfield(FastMultipole, name)::Base.RefValue
end

"""
    radix_setting(name::Symbol)

Current value of a radix-lifecycle tunable. Throws on unknown names and on
device-only settings before a backend extension has loaded.
"""
function radix_setting(name::Symbol)
    r = _radix_setting_ref(name)
    r === nothing && throw(ArgumentError(
        "radix setting $(repr(name)) is defined by a device backend that is not loaded; load a backend extension first"))
    return r[]
end

"""
    set_radix_setting!(name::Symbol, value)

Validated write to a radix-lifecycle tunable. Construction-locked settings
(`radix_setting_lock(name) == :construction`) must be set BEFORE constructing
a `RadixFMMCache`; a later flip is caught at the next device step by
`verify_locked_radix_settings` with a loud error.
"""
function set_radix_setting!(name::Symbol, value)
    haskey(RADIX_SETTING_SPECS, name) ||
        throw(ArgumentError("unknown radix setting $(repr(name)); known: $(sort!(collect(keys(RADIX_SETTING_SPECS))))"))
    spec = RADIX_SETTING_SPECS[name]
    try
        spec.validate(value)
    catch err
        err isa ArgumentError && throw(ArgumentError("invalid value for radix setting $(repr(name)): $(err.msg)"))
        rethrow()
    end
    r = _radix_setting_ref(name)
    r === nothing && throw(ArgumentError(
        "radix setting $(repr(name)) is defined by a device backend that is not loaded; load a backend extension first"))
    r[] = value
    return value
end

"""
    set_radix_settings!(settings::NamedTuple)

Validate and apply a group of radix settings atomically. Every name, value,
and lazy-load precondition is checked before the first write. If an unexpected
assignment failure occurs, already-written settings are restored.
"""
function set_radix_settings!(settings::NamedTuple)
    isempty(settings) && return settings
    refs = Pair{Symbol,Base.RefValue}[]
    converted = Pair{Symbol,Any}[]
    old = Pair{Symbol,Any}[]
    for (name, value) in pairs(settings)
        haskey(RADIX_SETTING_SPECS, name) || throw(ArgumentError(
            "unknown radix setting $(repr(name)); known: $(sort!(collect(keys(RADIX_SETTING_SPECS))))"))
        spec = RADIX_SETTING_SPECS[name]
        try
            spec.validate(value)
        catch err
            err isa ArgumentError && throw(ArgumentError(
                "invalid value for radix setting $(repr(name)): $(err.msg)"))
            rethrow()
        end
        r = _radix_setting_ref(name)
        r === nothing && throw(ArgumentError(
            "radix setting $(repr(name)) is defined by a device backend that is not loaded; load a backend extension first"))
        T = typeof(r[])
        value_t = try
            convert(T, value)
        catch err
            throw(ArgumentError("invalid value for radix setting $(repr(name)): cannot convert $(typeof(value)) to $T"))
        end
        push!(refs, name => r)
        push!(converted, name => value_t)
        push!(old, name => r[])
    end
    written = 0
    try
        for i in eachindex(refs)
            refs[i].second[] = converted[i].second
            written = i
        end
    catch
        for i in 1:written
            refs[i].second[] = old[i].second
        end
        rethrow()
    end
    return settings
end

"Lock class of a setting: `:construction` (baked at cache construction /
graph record — locked once a cache exists) or `:runtime` (flippable per step)."
function radix_setting_lock(name::Symbol)
    haskey(RADIX_SETTING_SPECS, name) ||
        throw(ArgumentError("unknown radix setting $(repr(name))"))
    return RADIX_SETTING_SPECS[name].lock
end

"""
    radix_settings()

NamedTuple of all currently-defined radix settings (CUDA-only settings appear
after a backend extension loads). See `RADIX_SETTING_SPECS` for lock
classes and docs.
"""
function radix_settings()
    names = sort!(collect(keys(RADIX_SETTING_SPECS)))
    defined = [n for n in names if _radix_setting_ref(n) !== nothing]
    return NamedTuple{Tuple(defined)}(Tuple(radix_setting(n) for n in defined))
end

"""
    snapshot_locked_radix_settings() -> Vector{Pair{Symbol,Any}}

Snapshot of every currently-defined construction-locked setting. Taken by
both `RadixFMMCache` constructors and verified at each device step.
"""
function snapshot_locked_radix_settings()
    out = Pair{Symbol,Any}[]
    for name in sort!(collect(keys(RADIX_SETTING_SPECS)))
        RADIX_SETTING_SPECS[name].lock === :construction || continue
        r = _radix_setting_ref(name)
        r === nothing && continue
        push!(out, name => r[])
    end
    return out
end

"""
    verify_locked_radix_settings(snapshot::Vector{Pair{Symbol,Any}})

Throw a loud error if any construction-locked setting drifted from the value
it had when the cache was built. Called at device-step entry (047 contract):
previously a late flip silently kept the old mechanism (the value is baked
into constructed buffers or the captured CUDA graph).
"""
function verify_locked_radix_settings(snapshot::Vector{Pair{Symbol,Any}})
    for (name, locked) in snapshot
        r = _radix_setting_ref(name)
        r === nothing && continue
        current = r[]
        if current != locked
            error(
                "radix setting $(name) is construction-locked but was changed after the RadixFMMCache was built " *
                "(built with $(repr(locked)), now $(repr(current))). The value is baked into construction-sized " *
                "buffers or the captured CUDA graph, so the flip would be silently ignored. Either restore " *
                "FastMultipole.set_radix_setting!($(repr(name)), $(repr(locked))) or rebuild the cache " *
                "(construct a new RadixFMMCache; from FLOWVPM use radix_fmm_settings!/clear_radix_fmm_cache!).")
        end
    end
    return nothing
end

verify_locked_radix_settings(::Nothing) = nothing
