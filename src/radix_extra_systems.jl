#------- extra target / source systems on the radix path -------#
#
# The resident radix lifecycle evaluates the cache's systems on themselves. A
# coupled solver (FLOWPanel, ActuatorLines) also needs the same call to deliver
# the field at a few hundred extra points (blade probes, wake-ring nodes) and
# to apply a few hundred extra sources (bound segments, ring filaments) to the
# resident bodies. Those systems are tiny, so no multipoles are built for them:
# every pairing that touches one is a direct rectangular evaluation.
#
# Given `fmm!(targets, sources, cache::RadixFMMCache)`:
#   main          = the cache's systems, the FIRST `n_systems` targets
#   extra targets = the remaining targets (evaluated from the main systems)
#   extra sources = sources that are not the main systems
#
# The main systems evaluate on themselves only when they ALSO appear in
# `sources` (all of them, at the front, in order). With the main systems absent
# from `sources` the lifecycle is skipped entirely and the call delivers the
# extra sources alone -- the "body on wake" direction of a coupled solve, where
# the self-induction was already evaluated earlier in the step and must not be
# recomputed over a field that has changed since.
#
#   main  <- main           resident FMM lifecycle (only when self-inducing)
#   main  <- extra sources  direct, accumulated into state.output in slot order
#                           BEFORE finalize (one write-back to the user system)
#   extra <- main           direct from the packed source bodies
#
# Extra sources are applied to the cache's systems ONLY. An extra target sees
# the resident field and nothing else: a coupled solver evaluates its own
# body-on-probe terms itself (with whatever self-exclusion its solve needs),
# so applying the extra sources there would double-count them. It follows that
# a non-self-inducing call (main systems absent from `sources`) evaluates no
# extra targets at all -- there is no resident field acting as a source.
#
# Contract for an extra SOURCE system: the usual `get_n_bodies`,
# `data_per_body`, `source_system_to_buffer!`, plus `direct_kernel(system)`
# returning an isbits functor with
#
#   _extra_pair_ug(kernel, tx, ty, tz, source_buffer, j)  -> (u, gx, gy, gz)
#   _extra_pair_ugh(kernel, tx, ty, tz, source_buffer, j) -> the 13-tuple
#
# where `(tx, ty, tz)` is the target position and `j` the packed column. The
# same functor runs on the host and on the device.
#
# Contract for an extra TARGET system: the usual target interface
# (`get_n_bodies`, `get_position`, `buffer_to_target_system!`).

struct RadixSystemSplit{MT<:Tuple,ET<:Tuple,ES<:Tuple}
    main::MT
    extra_targets::ET
    extra_sources::ES
    self_induce::Bool                # the main systems are sources as well
    main_index::Vector{Int}          # index of each main system within `targets`
    extra_target_index::Vector{Int}  # index of each extra target within `targets`
end

_is_in(x, tup::Tuple) = any(y -> y === x, tup)

function _split_radix_systems(n_systems::Integer, targets::Tuple, sources::Tuple)
    n = Int(n_systems)
    length(targets) >= n || throw(ArgumentError(
        "the radix fmm! path expects the cache's $n system(s) first in " *
        "target_systems; got $(length(targets)) target system(s)"))
    main = ntuple(i -> targets[i], n)
    n_in_sources = count(m -> _is_in(m, sources), main)
    self_induce = n_in_sources == n
    n_in_sources == 0 || self_induce || throw(ArgumentError(
        "the radix fmm! path requires the cache's systems to appear in " *
        "source_systems either all together (self-inducing) or not at all " *
        "(extra sources only); $(n_in_sources) of $n were found"))
    if self_induce
        all(sources[i] === main[i] for i in 1:n) || throw(ArgumentError(
            "the radix fmm! path requires the cache's systems to appear first " *
            "in source_systems, in the same order as in target_systems"))
    end
    extra_sources = Tuple(s for s in sources if !_is_in(s, main))
    return RadixSystemSplit(main, ntuple(i -> targets[n + i], length(targets) - n),
        extra_sources, self_induce, collect(1:n),
        collect(n+1:length(targets)))
end

_extra_pair_ug(kernel, tx, ty, tz, source_buffer, j) = throw(ArgumentError(
    "an extra source system on the radix path must define " *
    "FastMultipole._extra_pair_ug(::$(typeof(kernel)), tx, ty, tz, source_buffer, j); " *
    "return direct_kernel(system) as an isbits functor"))

_extra_pair_ugh(kernel, tx, ty, tz, source_buffer, j) = throw(ArgumentError(
    "hessian output from an extra source system requires " *
    "FastMultipole._extra_pair_ugh(::$(typeof(kernel)), tx, ty, tz, source_buffer, j)"))

"""
    _extra_pair_has_hessian(kernel) -> Bool

Whether an extra source's functor defines `_extra_pair_ugh`. Default `false`:
the source contributes velocity only, and the main systems' hessian rows are
left to the resident lifecycle even when the cache carries them. A functor
that defines `_extra_pair_ugh` overloads this to `true`. (A throwing default
inside a device kernel is a compile error, not a runtime one.)
"""
_extra_pair_has_hessian(kernel) = false

#------- host packing -------#

function _radix_extra_target_positions(::Type{TF}, system) where TF
    n = get_n_bodies(system)
    x = Matrix{TF}(undef, 3, n)
    @inbounds for i in 1:n
        p = get_position(system, i)
        x[1, i] = p[1]; x[2, i] = p[2]; x[3, i] = p[3]
    end
    return x
end

function _radix_extra_source_buffer(::Type{TF}, system) where TF
    n = get_n_bodies(system)
    buffer = zeros(TF, data_per_body(system), n)
    for i in 1:n
        source_system_to_buffer!(buffer, i, system, i)
    end
    return buffer
end

#------- host evaluators (also the reference for the device kernels) -------#

# targets at `xt` (3 x nt) from the packed main bodies `source_bodies[:, 1:n]`
# through the cache's nearfield functor. ACCUMULATES into `out` (4 or 13 rows).
function _host_extra_targets_from_main!(out::AbstractMatrix{TF}, kernel::AbstractDirectKernel,
        xt::AbstractMatrix, source_bodies, n::Integer, ::Val{HS},
        ghv::Val=Val(:shipped)) where {TF,HS}
    ep = _emits_potential(kernel)
    @inbounds for i in 1:size(xt, 2)
        xi = TF(xt[1, i]); yi = TF(xt[2, i]); zi = TF(xt[3, i])
        for j in 1:n
            dx = xi - source_bodies[1, j]
            dy = yi - source_bodies[2, j]
            dz = zi - source_bodies[3, j]
            r2 = dx * dx + dy * dy + dz * dz
            r2 == zero(TF) && continue
            invr = inv(sqrt(r2))
            if HS
                u, gx, gy, gz, h1, h2, h3, h4, h5, h6, h7, h8, h9 =
                    _direct_pair_ugh(kernel, dx, dy, dz, r2, invr, source_bodies, j, ghv)
                ep && (out[1, i] += u)
                out[2, i] += gx; out[3, i] += gy; out[4, i] += gz
                out[5, i] += h1; out[6, i] += h2; out[7, i] += h3
                out[8, i] += h4; out[9, i] += h5; out[10, i] += h6
                out[11, i] += h7; out[12, i] += h8; out[13, i] += h9
            else
                u, gx, gy, gz = _direct_pair_ug(kernel, dx, dy, dz, r2, invr,
                    source_bodies, j, ghv)
                ep && (out[1, i] += u)
                out[2, i] += gx; out[3, i] += gy; out[4, i] += gz
            end
        end
    end
    return out
end

# targets whose positions are rows 1:3 of `xt` (the packed main bodies in slot
# order) from an extra source's packed buffer. ACCUMULATES into `out`.
function _host_targets_from_extra_source!(out::AbstractMatrix{TF}, kernel,
        xt::AbstractMatrix, nt::Integer, source_buffer::AbstractMatrix,
        ::Val{HS}) where {TF,HS}
    ns = size(source_buffer, 2)
    @inbounds for i in 1:nt
        xi = xt[1, i]; yi = xt[2, i]; zi = xt[3, i]
        for j in 1:ns
            if HS
                u, gx, gy, gz, h1, h2, h3, h4, h5, h6, h7, h8, h9 =
                    _extra_pair_ugh(kernel, xi, yi, zi, source_buffer, j)
                out[1, i] += u
                out[2, i] += gx; out[3, i] += gy; out[4, i] += gz
                out[5, i] += h1; out[6, i] += h2; out[7, i] += h3
                out[8, i] += h4; out[9, i] += h5; out[10, i] += h6
                out[11, i] += h7; out[12, i] += h8; out[13, i] += h9
            else
                u, gx, gy, gz = _extra_pair_ug(kernel, xi, yi, zi, source_buffer, j)
                out[1, i] += u
                out[2, i] += gx; out[3, i] += gy; out[4, i] += gz
            end
        end
    end
    return out
end

#------- host drivers -------#

"""
    _radix_extra_sources_into_output!(state, extra_sources)

Host cache: apply every extra source system to the resident bodies, accumulating
into `state.output` in slot order (the slot positions are rows 1:3 of
`state.source_bodies`). Runs after the lifecycle body and before finalize.
"""
function _radix_extra_sources_into_output!(state::DeviceResidentRadixState{TF},
        extra_sources::Tuple) where TF
    isempty(extra_sources) && return state
    n = state.counts.n_bodies
    hs = size(state.output, 1) >= 13
    for system in extra_sources
        buffer = _radix_extra_source_buffer(TF, system)
        kernel = direct_kernel(system)
        _host_targets_from_extra_source!(state.output, kernel,
            state.source_bodies, n, buffer, Val(hs && _extra_pair_has_hessian(kernel)))
    end
    return state
end

# scatter a 4/13-row rectangular output into the target system through its
# derivatives switch, reusing the resident finalize copier with an identity
# permutation.
function _radix_scatter_extra_target!(::Type{TF}, system, switch, out::AbstractMatrix) where TF
    nt = size(out, 2)
    target_buffer = allocate_target_buffer(TF, system, switch)
    _copy_radix_output_to_host_target_buffer!(target_buffer, out, 1:nt,
        fill(1, nt), 1:nt, 1, switch, nt)
    buffer_to_target!(system, target_buffer, switch, 1:nt)
    return system
end

"""
    _radix_extra_targets_evaluate!(state, extra_targets, switches)

Host cache: evaluate every extra target system from the resident bodies, then
write back through the target's switch.
"""
function _radix_extra_targets_evaluate!(state::DeviceResidentRadixState{TF},
        extra_targets::Tuple, switches::Tuple) where TF
    isempty(extra_targets) && return state
    n = state.counts.n_bodies
    for (system, switch) in zip(extra_targets, switches)
        HS = !isempty(hessian_range(switch))
        HS && size(state.output, 1) < 13 && throw(ArgumentError(
            "hessian output requested for an extra target system but the cache " *
            "was built with hessian=false"))
        xt = _radix_extra_target_positions(TF, system)
        out = zeros(TF, HS ? 13 : 4, size(xt, 2))
        _host_extra_targets_from_main!(out, state.options.direct_kernel, xt,
            state.source_bodies, n, Val(HS))
        _radix_scatter_extra_target!(TF, system, switch, out)
    end
    return state
end
