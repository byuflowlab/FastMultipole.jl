#=##############################################################################
Consolidated settings surface for the GPU/radix production path.

Every mechanism tunable of the radix lifecycle is a process-global `Ref`
defined in this file. This file provides the single documented, validated
access surface plus the construction-lock contract:

- `radix_settings()`            -> NamedTuple of current values (defined Refs only)
- `radix_setting(name)`         -> current value
- `set_radix_setting!(name, v)` -> validated write (throws on unknown name,
                                   bad type, or out-of-domain value)

Each spec carries a lock class, `:construction` or `:runtime`.
Lock contract: a `:construction` setting would be read at cache/device-context
construction (it sizes buffers or selects the mechanism the cache is built
around), so flipping it after a `RadixFMMCache` is built would silently keep
the old mechanism. Both cache constructors therefore snapshot the
construction-locked settings (`snapshot_locked_radix_settings`), and the
device step entry verifies the snapshot (`verify_locked_radix_settings`),
throwing an actionable error on drift. `:runtime` settings are read per step
and may be flipped freely. Every setting registered below is currently
`:runtime`, so the snapshot is empty; the mechanism is kept for settings
that need it.
=###############################################################################

struct RadixSettingSpec
    lock::Symbol            # :construction or :runtime
    validate::Function      # value -> nothing (throws ArgumentError on bad value)
    doc::String
end

_rs_bool(v) = v isa Bool ? nothing : throw(ArgumentError("expected Bool, got $(typeof(v))"))
_rs_nonnegint(v) = (v isa Integer && !(v isa Bool) && v >= 0 && v <= typemax(Int)) ? nothing :
    throw(ArgumentError("expected a non-Bool Integer in 0:$(typemax(Int)), got $(repr(v))"))
_rs_enum(vals) = v -> v in vals ? nothing : throw(ArgumentError("expected one of $(vals), got $(repr(v))"))

# All-pairs direct arm. Swaps `ka_lifecycle_body!` for a single
# O(np^2) kernel and skips the grid/route refresh entirely. Off by default and
# selected only by an explicit write here: below some np the arm is faster
# (np ~ 6e4 on Metal for one wake), but that crossover has not been measured
# on CUDA and no automatic dispatch rule is built on it.
const RADIX_DIRECT_ARM = Ref(false)

const RADIX_SETTING_SPECS = Dict{Symbol,RadixSettingSpec}(
    # ---- nearfield -----------------------------------------------------------
    :CUDA_NEARFIELD_GH_MODE => RadixSettingSpec(:runtime,
        _rs_enum((:shipped, :reduced, :fp32, :reduced_fp32)),
        "g/h evaluation mode for the regularized nearfield on the HOST radix path (:fp32 default); read at every host step, so a flip takes effect on the next step. The KernelAbstractions kernels always evaluate the shipped series in the field precision and ignore this setting."),
    # ---- lifecycle/orchestration --------------------------------------------
    :RADIX_DIRECT_ARM => RadixSettingSpec(:runtime, _rs_bool,
        "Evaluate the step as a single all-pairs O(np^2) direct kernel instead of the FMM lifecycle, skipping the grid and route refresh (KA device path only; checked per step at entry). Opt-in: nothing selects it automatically."),
    # ---- host GEMM thresholds ------------------------------------------------
    :FACTORED_Y_GEMM_MIN_DIM => RadixSettingSpec(:runtime, _rs_nonnegint,
        "Host factored-y GEMM dimension threshold."),
    :PRECOMPUTED_Y_GEMM_MIN_COLS => RadixSettingSpec(:runtime, _rs_nonnegint,
        "Host precomputed-y GEMM column threshold."),
)

"Return the `Ref` behind a setting name (every registered setting has one),
or `nothing` for a registered name whose `Ref` is not defined."
function _radix_setting_ref(name::Symbol)
    haskey(RADIX_SETTING_SPECS, name) ||
        throw(ArgumentError("unknown radix setting $(repr(name)); known: $(sort!(collect(keys(RADIX_SETTING_SPECS))))"))
    isdefined(FastMultipole, name) || return nothing
    return getfield(FastMultipole, name)::Base.RefValue
end

"""
    radix_setting(name::Symbol)

Current value of a radix-lifecycle tunable. Throws on unknown names.
"""
function radix_setting(name::Symbol)
    r = _radix_setting_ref(name)
    r === nothing && throw(ArgumentError(
        "radix setting $(repr(name)) has no backing Ref"))
    return r[]
end

"""
    set_radix_setting!(name::Symbol, value)

Validated write to a radix-lifecycle tunable. Construction-locked settings
(`RADIX_SETTING_SPECS[name].lock == :construction`) must be set BEFORE constructing
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
        "radix setting $(repr(name)) has no backing Ref"))
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
            "radix setting $(repr(name)) has no backing Ref"))
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

"""
    radix_settings()

NamedTuple of all radix settings. See `RADIX_SETTING_SPECS` for lock classes
and docs.
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

Throw an error if any construction-locked setting drifted from the value it
had when the cache was built. Called at device-step entry, because such a
value is baked into constructed buffers and a late flip would otherwise be
silently ignored.
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
                "buffers, so the flip would be silently ignored. Either restore " *
                "FastMultipole.set_radix_setting!($(repr(name)), $(repr(locked))) or rebuild the cache " *
                "(construct a new RadixFMMCache, or have the package that owns it rebuild it).")
        end
    end
    return nothing
end

verify_locked_radix_settings(::Nothing) = nothing
