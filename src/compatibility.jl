#------- functions that should be overloaded for each user-defined system for use in the FMM -------#

"""
    residency(system)

Return whether a system's canonical FastMultipole buffers live on the host or on
the active device. Systems are host-resident by default. Device-backed systems
may opt into CUDA device-native materialization by overloading this method to
return [`DeviceResident()`](@ref).
"""
residency(system) = HostResident()

"""
    supports_third_derivative(target_system, source_system) -> Bool

Opt-in trait for a target/source pair whose direct interaction and target writeback
implement packed third-derivative output. It defaults to `false` so legacy kernels cannot
silently return incomplete near-field results.
"""
supports_third_derivative(target_system, source_system) = false

@inline _requests_third_derivative(::DerivativesSwitch{PS,GS,HS,NO,NM,TS}) where {PS,GS,HS,NO,NM,TS} = TS

function _check_third_derivative_support(target_systems, source_systems, switches)
    for (target, switch) in zip(target_systems, switches)
        _requests_third_derivative(switch) || continue
        for source in source_systems
            supports_third_derivative(target, source) || throw(ArgumentError(
                "third_derivative=true requires supports_third_derivative(target, source) == true; " *
                "unsupported pair: $(typeof(target)) <- $(typeof(source))"))
        end
    end
    return nothing
end

"""
    body_type(system)

Return the element type used to form multipole expansions from `system` on the
radix/resident path (task 032), e.g. `Point{Source}` (default) or
`Point{Vortex}`. The returned value is the element *type* itself, matching the
`body_to_multipole!(Point{Vortex}, system, args...)` convention of the legacy
path. All source systems sharing one `RadixFMMCache` must return the same body
type; `Point{Vortex}` requires `has_vector_potential(system) == true` (the
Lamb-Helmholtz χ channel), which is checked at cache construction.
"""
body_type(system) = Point{Source}

"""
    direct_kernel(system)

Return the nearfield direct-interaction kernel functor used for `system` on the
radix/resident path (task 032 stage 2). Defaults follow [`body_type`](@ref):
`SingularSource()` for `Point{Source}` and `SingularVortex()` for
`Point{Vortex}`. Overload to select [`RegularizedVortex`](@ref) (regularized
Biot-Savart, `gaussianerf`) or a custom kernel. All source systems sharing one
`RadixFMMCache` must return equal kernels; the functor must be `isbits` and, for
`device=true` caches, GPU-compilable. It is stamped into the cache options at
construction, so the pair kernels specialize on it at compile time (one kernel
instantiation per functor type, no runtime branch in the pair loop).

Custom kernels subtype `AbstractDirectKernel` and implement (with
`kernel = direct_kernel(system)`):

- `_direct_pair_ug(kernel, dx, dy, dz, r2, source_bodies, j)` returning
  `(u, gx, gy, gz)`, and
- `_direct_pair_ugh(kernel, dx, dy, dz, r2, source_bodies, j)` returning
  `(u, gx, gy, gz, h1, ..., h9)` (hessian in column-major 3×3 order), and
- `_emits_potential(kernel)::Bool` — whether `u` is meaningful (row 1 written).

Here `dx, dy, dz = target - source`, `r2 = dx^2+dy^2+dz^2 > 0` (self/coincident
pairs are skipped by the caller), and `source_bodies[:, j]` is the packed source
column (`[x, y, z, radius, strength..., extras...]`), giving the kernel access
to per-source extra states such as a smoothing radius. This flat-argument form
deviates from the spec §5 column-view signature so the same code compiles as a
CUDA device function without constructing a view per pair.
"""
direct_kernel(system) = _default_direct_kernel(body_type(system))

#--- buffer functions ---#

"""
    source_system_to_buffer!(buffer::Matrix, i_buffer, system::{UserDefinedSystem}, i_body)

Compatibility function used to sort source systems. It should be overloaded for each system (where `{UserDefinedSystem}` is replaced with the type of the user-defined system) to be used as a source and should behave as follows. For the `i_body`th body contained inside of `system`,

* `buffer[1:3, i_buffer]` should be set to the x, y, and z coordinates of the body position used for sorting into the octree
* `buffer[4, i_buffer]` should be set to the radius beyond which a multipole expansion is allowed to be evaluated (e.g. for panels, or other bodies of finite area/volume)
* `buffer[5:4+strength_dims, i_buffer]` should be set to the body strength, which is a vector of length `strength_dims`

Any additional information required for either forming multipole expansions or computing direct interactions should be stored in the rest of the column.

If a body contains vertices that are required for, e.g. computing multipole coefficients of dipole panels, these must be stored immediately following the body strength, and should be listed in a counter-clockwise order. For example, if I am using vortex tri-panels with `strength_dims=3`, I would set `buffer[8:10,i] .= v1`, `buffer[11:13,i] .= v2`, and `bufer[14:16,i] .= v3`, where `v1`, `v2`, and `v3` are listed according to the right-hand-rule with thumb aligned with the panel normal vector.

Note that any system acting only as a target need not overload `source_system_to_buffer!`.

"""
function source_system_to_buffer!(buffer, i_buffer, system, i_body)
    throw("source_system_to_buffer! not overloaded for type $(typeof(system))")
end

"""
    data_per_body(system::{UserDefinedSystem})

Returns the number of values used to represent a single body in a source system. Should be overloaded for each user-defined system object (where `{UserDefinedSystem}` is replaced with the type of the user-defined system).

"""
function data_per_body(system)
    throw("data_per_body not overloaded for type $(typeof(system))")
end

#--- getters ---#

"""
    get_position(system::{UserDefinedSystem}, i)

Returns a (static) vector of length 3 containing the x, y, and z coordinates of the position of the `i`th body. Should be overloaded for each user-defined system object (where `{UserDefinedSystem}` is replaced with the type of the user-defined system).

"""
function get_position(system, i)
    throw("get_position not overloaded for type $(typeof(system))")
end

"""
    strength_dims(system::{UserDefinedSystem})

Returns the cardinality of the vector used to define the strength of each body inside `system`. E.g., a point mass would return 1, and a point dipole would return 3. Should be overloaded for each user-defined system object (where `{UserDefinedSystem}` is replaced with the type of the user-defined system).

"""
function strength_dims(system)
    throw("strength_dims() not overloaded for type $(typeof(system))")
end

"""
    get_normal(source_buffer, source_system::{UserDefinedSystem}, i)

**OPTIONAL OVERLOAD:**

Returns the unit normal vector for the `i`th body of `source_buffer`. May be (optionally) overloaded for a user-defined system object (where `{UserDefinedSystem}` is replaced with the type of the user-defined system); otherwise, the default behavior assumes counter-clockwise ordered vertices. Note that whatever method is used should match `source_system_to_buffer!` for each system.

"""
function get_normal(source_buffer, source_system, i_body)
    v1 = get_vertex(source_buffer, source_system, i_body, 1)
    v2 = get_vertex(source_buffer, source_system, i_body, 2)
    v3 = get_vertex(source_buffer, source_system, i_body, 3)
    normal = cross(v2-v1, v3-v1)

    return normal / norm(normal)
end

"""
    get_n_bodies(system::{UserDefinedSystem})

Returns the number of bodies contained inside `system`. Should be overloaded for each user-defined system object (where `{UserDefinedSystem}` is replaced with the type of the user-defined system).
"""
get_n_bodies(system) = throw("FastMultipole.get_n_bodies() not overloaded for type $(typeof(system))")

"""
    body_to_multipole!(system::{UserDefinedSystem}, multipole_coefficients, buffer, expansion_center, bodies_index, harmonics, expansion_order)

Calculates the multipole coefficients due to the bodies contained in `buffer[:,bodies_index]` and accumulates them in `multipole_coefficients`. Should be overloaded for each user-defined system object (where `{UserDefinedSystem}` is replaced with the type of the user-defined system).

Typically, this is done using one of the convience functions contained within FastMultipole in one line, as

```julia
body_to_multipole!(system::MySystem, args...) = body_to_multipole!(Point{Vortex}, system, args...)
```

"""
function body_to_multipole!(system, multipole_coefficients, buffer, expansion_center, bodies_index, harmonics, expansion_order)
    if WARNING_FLAG_B2M[]
        @warn "body_to_multipole! not overloaded for type $(typeof(system)); multipole expansions from this system ignored"
        WARNING_FLAG_B2M[] = false
    end
    return nothing
end

"""
    direct!(target_buffer, target_index, derivatives_switch::DerivativesSwitch{PS,GS,HS}, ::{UserDefinedSystem}, source_buffer, source_index) where {PS,GS,HS}

Calculates direct (nearfield) interactions of `source_system` on `target_buffer`. Should be overloaded or each user-defined system object (where `{UserDefinedSystem}` is replaced with the type of the user-defined system), for all source bodies in `source_index`, at all target bodies in `target_index`, as follows:

```julia
# loop over source bodies
for i_source in source_index

    # extract source body information here...

    # loop over target bodies
    for i_target in target_index

        # get target position
        target_position = get_position(target_buffer, i_target)

        # evaluate influence here...

        # update appropriate quantities
        if PS
            set_scalar_potential!(target_buffer, i_target, scalar_potential)
        end
        if GS
            set_gradient!(target_buffer, i_target, gradient)
        end
        if HS
            set_hessian!(target_buffer, i_target, hessian)
        end

    end
end
```

Note that `::{UserDefinedSystem}` is used purely for overloading the method for the appropriate system, and should NOT be accessed in this function, since it will NOT be indexed according to `source_index`. Rather, `source_buffer`, which is updated using `source_system_to_buffer!`, should be accessed.

The following convenience getter functions are available for accessing the source system:

* `get_position(source_system::{UserDefinedSystem}, i_body::Int)`: returns an SVector of length 3 containing the position of the `i_body` body
* `get_strength(source_buffer::Matrix, source_system::{UserDefinedSystem}, i_body::Int)`: returns an SVector containing the strength of the `i_body` body
* `get_vertex(source_buffer::Matrix, source_system::{UserDefinedSystem}, i_body::Int, i_vertex::Int)`: returns an SVector containing the x, y, and z coordinates of the `i_vertex` vertex of the `i_body` body

Note also that the compile time parameters `PS`, `GS`, and `HS` are used to determine whether the scalar potential and vector field should be computed, respectively. This allows us to skip unnecessary calculations and improve performance.

"""
function direct!(target_buffer, target_index, derivatives_switch, source_system, source_buffer, source_index)
    if WARNING_FLAG_DIRECT[]
        @warn "direct! not overloaded for type $(typeof(source_system)); interaction ignored"
        WARNING_FLAG_DIRECT[] = false
    end
    return nothing
end

"""
    buffer_to_target_system!(target_system::{UserDefinedSystem}, i_target, ::DerivativesSwitch{PS,GS,HS}, target_buffer, i_buffer) where {PS,GS,HS}

Compatibility function used to update target systems. It should be overloaded for each system (where `{UserDefinedSystem}` is replaced with the type of the user-defined system) to be a target and should behave as follows. For the `i_body`th body contained inside of `target_system`,

* `get_scalar_potential(target_buffer, derivatives_switch, i_buffer)` contains the scalar potential influence to be added to the `i_target` body of `target_system`, when scalar potential output is enabled
* `get_gradient(target_buffer, derivatives_switch, i_buffer)` contains the vector field influence to be added to the `i_target` body of `target_system`, when gradient output is enabled
* `get_hessian(target_buffer, derivatives_switch, i_buffer)` contains the vector field gradient to be added to the `i_target` body of `target_system`, when hessian output is enabled

Note that any system acting only as a source (and not as a target) need not overload `buffer_to_target_system!`.

Target buffers are compact when metadata or disabled outputs are involved. The no-switch matrix getters still read the legacy rows `4`, `5:7`, and `8:16`, but new target-buffer code should use the switch-aware getter functions.

For some slight performance improvements, the booleans `PS`, `GS`, and `HS` can be used as a switch to indicate whether the scalar potential, vector field, and vector gradient are to be stored, respectively. Since they are compile-time parameters, `if` statements relying on them will not incur a runtime cost.

"""
function buffer_to_target_system!(target_system, i_target, derivatives_switch, target_buffer, i_buffer)
    throw("buffer_to_target_system! not overloaded for type $(typeof(target_system))")
end

"""
    target_influence_to_buffer!(target_buffer, i_buffer, ::DerivativesSwitch{PS,GS,HS}, target_system::{UserDefinedSystem}, i_target) where {PS,GS,HS}

**NOTE:** this function is primarily used for the boundary element solver, and is not required for the FMM.

Updates the `target_buffer` with influences from `target_system` for the `i_target`th body. Should be overloaded for each user-defined system object (where `{UserDefinedSystem}` is replaced with the type of the user-defined system) to be used as a target, assuming the target buffer positions have already been set. It should behave as follows:

* `set_scalar_potential!(target_buffer, derivatives_switch, i_buffer, scalar_potential)` should be used when scalar potential output is enabled
* `set_gradient!(target_buffer, derivatives_switch, i_buffer, gradient)` should be used when gradient output is enabled
* `set_hessian!(target_buffer, derivatives_switch, i_buffer, hessian)` should be used when hessian output is enabled

The no-switch matrix setters still write legacy rows `4`, `5:7`, and `8:16`; switch-aware setters are required for compact target buffers.

"""
function target_influence_to_buffer!(target_buffer, i_buffer, derivatives_switch, target_system, i_target)
    throw("target_influence_to_buffer! not overloaded for type $(typeof(target_system))")
end

"""
    strength_to_value(strength, source_system)

**NOTE:** this function is primarily used for the boundary element solver, and is not required for the FMM.

Converts the strength of a body in `source_system` to a scalar value. Should be overloaded for each user-defined system object used with the boundary element solver.

**Arguments:**

* `strength::SVector{dim, Float64}`: the strength of the body, where `dim` is the number of components in the strength vector (e.g., 1 for a point source, 3 for a point vortex, etc.)
* `source_system::{UserDefinedSystem}`: the user-defined system object, used solely for dispatch

"""
function strength_to_value(strength, source_system)
    throw("strength_to_value not overloaded for type $(typeof(source_system))")
end

function strength_to_value(source_buffer::Matrix, source_system, i_body)
    return strength_to_value(get_strength(source_buffer, source_system, i_body), source_system)
end

"""
    value_to_strength!(source_buffer, source_system, i_body, value)

**NOTE:** this function is primarily used for the boundary element solver, and is not required for the FMM.

Converts a scalar value to a vector strength of a body in `source_system`. Should be overloaded for each user-defined system object used with the boundary element solver.

**Arguments:**

* `source_buffer::Matrix{Float64}`: the source buffer containing the body information
* `source_system::{UserDefinedSystem}`: the user-defined system object, used solely for dispatch
* `i_body::Int`: the index of the body in `source_buffer` to set the strength for
* `value::Float64`: the scalar value used to set the strength

The following convenience function may be helpful when accessing the buffer:

* `get_strength(source_buffer, source_system, i_body)`: returns the strength of the `i_body` body in `source_buffer`, formatted as a vector (e.g. `strength::SVector{dim,Float64}`)

"""
function value_to_strength!(source_buffer, source_system, i_body, value)
    throw("value_to_strength! not overloaded for type $(typeof(source_system))")
end

"Ignore relaxation factor by default, unless overloaded by the user."
function value_to_strength!(source_buffer, source_system, i_body, value, rlx)
    value_to_strength!(source_buffer, source_system, i_body, value)
end

"""
    buffer_to_system_strength!(system::{UserDefinedSystem}, source_buffer::Matrix{Float64}, i_body::Int)

**NOTE:** this function is primarily used for the boundary element solver, and is not required for the FMM.

Updates the strength in `system` for the `i_body`th body using the strength information contained in `source_buffer`. Should be overloaded for each user-defined system object used with the boundary element solver.

**Arguments:**

* `system::{UserDefinedSystem}`: the user-defined system object
* `i_body::Int`: the index of the body in `source_system` whose strength is to be set
* `source_buffer::Matrix{Float64}`: the source buffer containing the body information
* `i_buffer::Int`: the index of the body in `source_buffer` whose strength is to be set

"""
function buffer_to_system_strength!(system, i_body, source_buffer, i_buffer)
    throw("buffer_to_system_strength! not overloaded for type $(typeof(system))")
end

"""
    influence!(influence, target_buffer, derivatives_switch, source_system, source_buffer)

**NOTE:** `source_system` is provided solely for dispatch; it's member bodies will be out of order and should not be referenced.

**NOTE:** This function is primarily used for the boundary element solver, and is not required for the FMM.

Evaluate the influence as pertains to the boundary element influence matrix and overwrites it to `influence` (which would need to be subtracted for it to act like the RHS of a linear system). Based on the current state of the `target_buffer` and `source_buffer`. Should be overloaded for each system type that is used in the boundary element solver.

**Arguments:**

* `influence::AbstractVector{TF}`: vector containing the influence for every body in the target buffer
* `target_buffer::Matrix{TF}`: target buffer used to compute the influence
* `derivatives_switch::DerivativesSwitch`: target-buffer layout used to locate enabled outputs
* `source_system::{UserDefinedSystem}`: system object used solely for dispatch
* `source_buffer::Matrix{TF}`: source buffer used to compute the influence

"""
function influence!(influence, target_buffer, derivatives_switch::DerivativesSwitch, source_system, source_buffer)
    error("influence! not overloaded for systems of type $(typeof(source_system))")
end

"""
    has_vector_potential(system::{UserDefinedSystem})

Returns `true` if the system induces a vector potential, `false` otherwise. Should be overloaded for each user-defined system object (where `{UserDefinedSystem}` is replaced with the type of the user-defined system).
"""
function has_vector_potential(system)
    error("has_vector_potential not overloaded for type $(typeof(system))")
end

#------- internal functions -------#

#--- lamb-helmholtz ---#

function has_vector_potential(systems::Tuple)
    not_lh = true
    for system in systems
        not_lh = not_lh && !has_vector_potential(system)
    end
    return !not_lh
end

#--- source_buffer getters ---#

# function get_position(source_buffer, source_system, i_body::Int)
#     return SVector{3}(view(source_buffer, 1:3, i_body))
# end

function get_radius(source_buffer::Matrix, i_body)
    return source_buffer[4,i_body]
end

function get_strength(source_buffer, source_system, i_body::Int)
    dim = strength_dims(source_system)
    strength = SVector{dim, eltype(source_buffer)}(source_buffer[4+i, i_body] for i in 1:dim)
    return strength
end

function get_vertex(source_buffer, source_system, i_body::Int, i_vertex::Int)
    i_offset = 3 * (i_vertex - 1)
    vertex = SVector{3}(view(source_buffer, 5+strength_dims(source_system)+i_offset:7+strength_dims(source_system)+i_offset, i_body))
    return vertex
end

#--- system/buffer setters ---#

function buffer_to_target!(target_systems::Tuple, target_tree::Tree, derivatives_switches=DerivativesSwitch(true, true, true, target_systems))
    buffer_to_target!(target_systems, target_tree.buffers, derivatives_switches, target_tree.sort_index_list)
end

"""
    buffer_to_target!(target_system, target_buffer, derivatives_switch, sort_index, ...)

Deliver an evaluation's results from the **framework-owned** output buffer to
the consumer's own state. Called by the framework at the end of every
evaluation; rows are switch-relative (`scalar_potential_index`,
`gradient_range`, `hessian_range` of the `DerivativesSwitch`), and the call
must be steady-state allocation-free.

**Delivery semantics**: the buffer always holds the **total influence of this
evaluation** — the framework zeroes its accumulators each step. Whether the
consumer overwrites its state or accumulates into it (`.=` vs `.+=`) inside
this call is the consumer's choice; both are correct (a time stepper typically
overwrites, FLOWVPM-style resets accumulate).

Host systems get this behavior for free by overloading
[`buffer_to_target_system!`](@ref); `DeviceResident` systems overload
`buffer_to_target!(system, device_output_buffer, derivatives_switch,
sort_index)` for their device buffer type and consume it with device-to-device
operations.
"""
function buffer_to_target!(target_systems::Tuple, target_buffers, derivatives_switches, sort_index_list=Tuple(1:get_n_bodies(system) for system in target_systems), buffer_index_list=Tuple(1:get_n_bodies(system) for system in target_systems))
    for (target_system, target_buffer, derivatives_switch, sort_index, buffer_index) in zip(target_systems, target_buffers, derivatives_switches, sort_index_list, buffer_index_list)
        buffer_to_target!(target_system, target_buffer, derivatives_switch, sort_index, buffer_index)
    end
end

function buffer_to_target!(target_system, target_buffer, derivatives_switch, sort_index=1:get_n_bodies(target_system), buffer_index=1:get_n_bodies(target_system))
    if get_n_bodies(target_system) > MIN_BODIES
        Threads.@threads for i_body in buffer_index
            buffer_to_target_system!(target_system, sort_index[i_body], derivatives_switch, target_buffer, i_body)
        end
    else
        for i_body in buffer_index
            buffer_to_target_system!(target_system, sort_index[i_body], derivatives_switch, target_buffer, i_body)
        end
    end
end

"""
    sfs_to_target!(target_system, sfs_buffer, sort_index=1:get_n_bodies(target_system))

Deliver the SFS (subfilter-scale vortex-stretching) result of an evaluation to
the consumer: `sfs_buffer` is a **framework-owned** `3 x n_bodies` matrix in
**global (unsorted) body order** holding `E_str` for every body of
`target_system` (device caches pass a device matrix to `DeviceResident`
systems, a host matrix otherwise). Same delivery semantics as
[`buffer_to_target!`](@ref): the buffer holds the total influence of this
evaluation; overwrite vs accumulate is the consumer's choice, and the call
must be steady-state allocation-free. Only consumers evaluated with
`fmm!(...; sfs=true)` on an `sfs=true` [`RadixFMMCache`](@ref) need this
overload (task 048).
"""
function sfs_to_target!(target_system, sfs_buffer,
        sort_index=1:get_n_bodies(target_system))
    throw(ArgumentError(
        "target systems evaluated with sfs=true must overload " *
        "FastMultipole.sfs_to_target!(target_system, sfs_buffer, sort_index) " *
        "for $(typeof(target_system))"))
end

"""
    extra_target_data_to_buffer!(buffer, i_body, system, i_sorted)

Deprecated compatibility hook. New code should overload [`metadata_to_buffer!`](@ref)
instead. This function is still called by the default `metadata_to_buffer!`.

**NOTE:** Exactly `n=extra_target_data_per_body(system)` rows should be added by this function in `target_buffer[end-n-1:end-2, i_body]`.

"""
function extra_target_data_to_buffer!(buffer, i_body, system, i_sorted)
    return nothing
end

"""
    extra_target_data_per_body(system)

Deprecated compatibility hook. New code should overload [`metadata_per_body`](@ref)
instead. This function is still used as the default fallback for
`metadata_per_body`.
"""
function extra_target_data_per_body(system)
    return 0
end

"""
    metadata_to_buffer!(buffer, switch, i_buffer, system, i_body)

Copies target metadata for `i_body` into the target buffer column `i_buffer`.
Metadata rows occupy `metadata_range(switch)` and are sorted with positions.
Outputs are stored after metadata and should not be written by this function.
"""
function metadata_to_buffer!(buffer, switch, i_buffer, system, i_body)
    extra_target_data_to_buffer!(buffer, i_buffer, system, i_body)
end

"""
    metadata_per_body(system)

Returns the number of per-target metadata rows that should be carried through
target tree sorting. Defaults to `extra_target_data_per_body(system)` for
compatibility, which defaults to `0`.
"""
function metadata_per_body(system)
    return extra_target_data_per_body(system)
end

"""
    previous_potential_metadata_index(system)

Returns the 1-based metadata row used as the previous scalar-potential estimate
for relative error prediction, or `0` if unavailable.
"""
previous_potential_metadata_index(system) = 0

"""
    previous_gradient_metadata_index(system)

Returns the 1-based metadata row used as the previous gradient-magnitude
estimate for relative error prediction, or `0` if unavailable.
"""
previous_gradient_metadata_index(system) = 0

target_uses_previous_influence_metadata(system) = previous_potential_metadata_index(system) > 0 && previous_gradient_metadata_index(system) > 0
target_uses_previous_influence_metadata(system, ::DerivativesSwitch{PS,GS,HS,NO,NM}) where {PS,GS,HS,NO,NM} =
    target_uses_previous_influence_metadata(system) &&
    previous_potential_metadata_index(system) <= NM &&
    previous_gradient_metadata_index(system) <= NM

metadata_value(buffer, j, i_body) = j == 0 ? zero(eltype(buffer)) : buffer[3 + j, i_body]

function warn_missing_previous_influence_metadata(target_systems::Tuple, error_tolerance)
    isnothing(error_tolerance) && return nothing
    error_tolerance isa RelativeErrorMethod || return nothing
    for system in target_systems
        if !target_uses_previous_influence_metadata(system) && WARNING_FLAG_MAX_INFLUENCE[]
            @warn "relative error prediction requested but previous influence metadata indices are unavailable for type $(typeof(system)); falling back to absolute tolerance behavior"
            WARNING_FLAG_MAX_INFLUENCE[] = false
            return nothing
        end
    end
    return nothing
end

function metadata_vector(metadata, target_systems::Tuple)
    isnothing(metadata) && return [metadata_per_body(system) for system in target_systems]
    return to_vector(metadata, length(target_systems))
end

function target_to_buffer!(buffers, systems::Tuple, sort_index_list=SVector{length(systems)}([1:get_n_bodies(system) for system in systems]), switches=DerivativesSwitch(true, true, true, systems))
    @assert length(switches) == length(systems) "target switches must match target systems"
    for (buffer, system, sort_index, switch) in zip(buffers, systems, sort_index_list, switches)
        target_to_buffer!(buffer, system, sort_index, switch)
    end
end

function target_to_buffer!(buffer::Matrix, system, sort_index=1:get_n_bodies(system), switch=DerivativesSwitch(true, true, true, system))
    if Threads.nthreads() > 1 && get_n_bodies(system) > MIN_BODIES
        target_to_buffer_multithread!(buffer, system, sort_index, switch)
    else
        for i_body in 1:get_n_bodies(system)
            i_sorted = sort_index[i_body]
            buffer[1:3, i_body] .= get_position(system, i_sorted)
            metadata_to_buffer!(buffer, switch, i_body, system, i_sorted)
        end
    end
end

function target_to_buffer_multithread!(buffer::Matrix, system, sort_index=1:get_n_bodies(system), switch=DerivativesSwitch(true, true, true, system))
    Threads.@threads for i_body in 1:get_n_bodies(system)
        i_sorted = sort_index[i_body]
        buffer[1:3, i_body] .= get_position(system, i_sorted)
        metadata_to_buffer!(buffer, switch, i_body, system, i_sorted)
    end
end

"""
    source_to_buffer!(buffer, system, sort_index=1:get_n_bodies(system))

Pack `system`'s live bodies into the **framework-owned** packed source buffer:
column `i` holds body `sort_index[i]` as `[x, y, z, radius,
strength (rows 5:4+strength_dims), extras...]`. Called by the framework every
evaluation (and by [`recenter!`](@ref) when deriving bounds); the consumer
never allocates or retains the buffer, and the call must be steady-state
allocation-free. Host systems get this behavior for free by overloading
[`source_system_to_buffer!`](@ref); `DeviceResident` systems overload this
method for their device buffer type (the framework passes a view of the valid
column prefix of a persistent device buffer, with the identity `sort_index`)
and fill it with device-to-device operations.
"""
function source_to_buffer!(buffers, systems::Tuple, sort_index_list=SVector{length(systems)}([1:get_n_bodies(system) for system in systems]))
    for (buffer, system, sort_index) in zip(buffers, systems, sort_index_list)
        source_to_buffer!(buffer, system, sort_index)
    end
end

function source_to_buffer!(buffer::Matrix, system, sort_index=1:get_n_bodies(system))
    if Threads.nthreads() > 1 && get_n_bodies(system) > MIN_BODIES
        source_to_buffer_multithread!(buffer, system, sort_index)
    else
        for i_body in 1:get_n_bodies(system)
            source_system_to_buffer!(buffer, i_body, system, sort_index[i_body])
        end
    end
end

function source_to_buffer_multithread!(buffer::Matrix, system, sort_index=1:get_n_bodies(system))
    Threads.@threads for i_body in 1:get_n_bodies(system)
        source_system_to_buffer!(buffer, i_body, system, sort_index[i_body])
    end
end

function target_to_buffer(systems::Tuple, sort_index_list=SVector{length(systems)}([1:get_n_bodies(system) for system in systems]), switches=DerivativesSwitch(true, true, true, systems))
    buffers = allocate_buffers(systems, true, get_type(systems), switches)
    target_to_buffer!(buffers, systems, sort_index_list, switches)
    return buffers
end

function target_to_buffer(system, switch::DerivativesSwitch, sort_index=1:get_n_bodies(system))
    buffer = allocate_target_buffer(numtype(system), system, switch)
    target_to_buffer!(buffer, system, sort_index, switch)
    return buffer
end

function source_to_buffer(systems::Tuple, sort_index_list=SVector{length(systems)}([1:get_n_bodies(system) for system in systems]))
    buffers = allocate_buffers(systems, false, get_type(systems), DerivativesSwitch(false, false, false, systems))
    source_to_buffer!(buffers, systems, sort_index_list)
    return buffers
end

function source_to_buffer(system, sort_index=1:get_n_bodies(system))
    buffer = allocate_source_buffer(numtype(system), system)
    source_to_buffer!(buffer, system, sort_index)
    return buffer
end

function target_influence_to_buffer!(target_buffers, target_systems::Tuple, derivatives_switches::Tuple, sort_index_list=SVector{length(target_systems)}([1:get_n_bodies(system) for system in target_systems]))
    for (target_buffer, target_system, derivatives_switch, sort_index) in zip(target_buffers, target_systems, derivatives_switches, sort_index_list)
        reset!(target_buffer)
        target_influence_to_buffer!(target_buffer, target_system, derivatives_switch, sort_index)
    end
end

function target_influence_to_buffer!(target_buffer::Matrix, target_system, derivatives_switch, sort_index=1:get_n_bodies(target_system))
    for i_body in 1:get_n_bodies(target_system)
        target_influence_to_buffer!(target_buffer, i_body, derivatives_switch, target_system, sort_index[i_body])
    end
end

function system_to_buffer!(buffers, systems::Tuple, sort_index_list=SVector{length(systems)}([1:get_n_bodies(system) for system in systems]))
    if Threads.nthreads() > 1 && get_n_bodies(systems) > MIN_BODIES
        for (buffer, system, sort_index) in zip(buffers, systems, sort_index_list)
            system_to_buffer_multithread!(buffer, system, sort_index)
        end
    else
        # single-threaded fallback
        for (buffer, system, sort_index) in zip(buffers, systems, sort_index_list)
            system_to_buffer!(buffer, system, sort_index)
        end
    end
end

function system_to_buffer!(buffer::Matrix, system, sort_index=1:get_n_bodies(system))
    for i_body in 1:get_n_bodies(system)
        source_system_to_buffer!(buffer, i_body, system, sort_index[i_body])
    end
end

function system_to_buffer_multithread!(buffer::Matrix, system, sort_index=1:get_n_bodies(system))
    Threads.@threads for i_body in 1:get_n_bodies(system)
        source_system_to_buffer!(buffer, i_body, system, sort_index[i_body])
    end
end

function system_to_buffer(systems::Tuple, sort_index_list=SVector{length(systems)}([1:get_n_bodies(system) for system in systems]))
    buffers = allocate_buffers(systems, false, get_type(systems), DerivativesSwitch(false, false, false, systems))
    system_to_buffer!(buffers, systems, sort_index_list)
    return buffers
end

function system_to_buffer(system, sort_index=1:get_n_bodies(system))
    buffer = allocate_source_buffer(numtype(system), system)
    system_to_buffer!(buffer, system, sort_index)
    return buffer
end

function buffer_to_system_strength!(source_systems::Tuple, source_tree::Tree)
    buffer_to_system_strength!(source_systems, source_tree.buffers, source_tree.sort_index_list)
end

function buffer_to_system_strength!(source_systems::Tuple, source_buffers::AbstractVector{<:Matrix}, sort_index_list=SVector{length(source_systems)}([1:get_n_bodies(system) for system in source_systems]), buffer_index_list=SVector{length(source_systems)}([1:get_n_bodies(system) for system in source_systems]))
    for (source_system, source_buffer, sort_index, buffer_index) in zip(source_systems, source_buffers, sort_index_list, buffer_index_list)
        buffer_to_system_strength_range!(source_system, source_buffer, sort_index, buffer_index)
    end
end

function buffer_to_system_strength_range!(source_system, source_buffer::Matrix, sort_index::AbstractVector=1:get_n_bodies(source_system), buffer_index::AbstractVector=1:get_n_bodies(source_system))
    for i_buffer in buffer_index
        buffer_to_system_strength!(source_system, sort_index[i_buffer], source_buffer, i_buffer)
    end
end

#--- auxilliary functions ---#

@inline function get_n_bodies(systems::Union{Tuple, AbstractVector{<:Matrix}})
    n_bodies = 0
    for system in systems
        n_bodies += get_n_bodies(system)
    end
    return n_bodies
end

#------- access functions for use with a matrix of targets used as input to direct! -------#

#--- getters ---#

# function get_position(source_buffer, source_system, i_body::Int)
#     return SVector{3}(view(source_buffer, 1:3, i_body))
# end

function get_position(system::AbstractMatrix{TF}, i) where TF
    @inbounds val = SVector{3,TF}(system[1, i], system[2, i], system[3, i])
    return val
end

get_scalar_potential(system::AbstractMatrix, i) = @inbounds system[4, i]
get_scalar_potential(system::AbstractMatrix, switch::DerivativesSwitch{true,<:Any,<:Any}, i) = @inbounds system[scalar_potential_index(switch), i]
get_scalar_potential(system::AbstractMatrix, ::DerivativesSwitch{false,<:Any,<:Any}, i) =
    throw(ArgumentError("scalar potential output is disabled for this target buffer"))

get_gradient(system::AbstractMatrix{TF}, i) where TF = @inbounds SVector{3,TF}(system[5,i], system[6,i], system[7,i])
get_gradient(system::AbstractMatrix{TF}, switch::DerivativesSwitch{<:Any,true,<:Any}, i) where TF = @inbounds SVector{3,TF}(system[gradient_range(switch)[1],i], system[gradient_range(switch)[2],i], system[gradient_range(switch)[3],i])
get_gradient(system::AbstractMatrix, ::DerivativesSwitch{<:Any,false,<:Any}, i) =
    throw(ArgumentError("gradient output is disabled for this target buffer"))

get_hessian(system::AbstractMatrix{TF}, i) where TF =
    @inbounds SMatrix{3,3,TF,9}(system[8, i], system[9, i], system[10, i],
    system[11, i], system[12, i], system[13, i],
    system[14, i], system[15, i], system[16, i])
function get_hessian(system::AbstractMatrix{TF}, switch::DerivativesSwitch{<:Any,<:Any,true}, i) where TF
    r = hessian_range(switch)
    return @inbounds SMatrix{3,3,TF,9}(system[r[1], i], system[r[2], i], system[r[3], i],
    system[r[4], i], system[r[5], i], system[r[6], i],
    system[r[7], i], system[r[8], i], system[r[9], i])
end
get_hessian(system::AbstractMatrix, ::DerivativesSwitch{<:Any,<:Any,false}, i) =
    throw(ArgumentError("hessian output is disabled for this target buffer"))

"""
    get_third_derivative(target_buffer, i_body)
    get_third_derivative(target_buffer, derivatives_switch, i_body)

Returns the third derivative `T[i,j,k] = ∂H[i,j]/∂x[k]` induced at the `i_body`th body of
`target_buffer` as a [`ThirdDerivativeTensor`](@ref). The two-argument form assumes the
default layout (rows 17:34, no metadata or extra outputs); the switch-aware form reads the
rows given by [`third_derivative_range`](@ref) and throws an `ArgumentError` if the switch
did not request third derivatives.
"""
function get_third_derivative(system::AbstractMatrix{TF}, i) where TF
    return ThirdDerivativeTensor(SVector{18,TF}(ntuple(n -> @inbounds(system[16 + n, i]), Val(18))))
end
function get_third_derivative(system::AbstractMatrix{TF}, switch::DerivativesSwitch{<:Any,<:Any,<:Any,<:Any,<:Any,true}, i) where TF
    first_row = first(third_derivative_range(switch))
    return ThirdDerivativeTensor(SVector{18,TF}(ntuple(n -> @inbounds(system[first_row + n - 1, i]), Val(18))))
end
get_third_derivative(system::AbstractMatrix, ::DerivativesSwitch{<:Any,<:Any,<:Any,<:Any,<:Any,false}, i) =
    throw(ArgumentError("third-derivative output is disabled for this target buffer"))

get_n_bodies(sys::AbstractMatrix) = size(sys, 2)

#--- setters ---#

"""
    set_scalar_potential!(target_buffer, i_body, scalar_potential)

Accumulates `scalar_potential` to `target_buffer`.

"""
function set_scalar_potential!(system::Matrix, i, scalar_potential)
    @inbounds system[4, i] += scalar_potential
end
function set_scalar_potential!(system::Matrix, switch::DerivativesSwitch{true,<:Any,<:Any}, i, scalar_potential)
    @inbounds system[scalar_potential_index(switch), i] += scalar_potential
end
set_scalar_potential!(system::Matrix, ::DerivativesSwitch{false,<:Any,<:Any}, i, scalar_potential) =
    throw(ArgumentError("scalar potential output is disabled for this target buffer"))

"""
    set_gradient!(target_buffer, i_body, gradient)

Accumulates `gradient` to `target_buffer`.

"""
function set_gradient!(system::Matrix, i, gradient)
    @inbounds system[5,i] += gradient[1]
    @inbounds system[6,i] += gradient[2]
    @inbounds system[7,i] += gradient[3]
end
function set_gradient!(system::Matrix, switch::DerivativesSwitch{<:Any,true,<:Any}, i, gradient)
    r = gradient_range(switch)
    @inbounds system[r[1],i] += gradient[1]
    @inbounds system[r[2],i] += gradient[2]
    @inbounds system[r[3],i] += gradient[3]
end
set_gradient!(system::Matrix, ::DerivativesSwitch{<:Any,false,<:Any}, i, gradient) =
    throw(ArgumentError("gradient output is disabled for this target buffer"))

"""
    set_hessian!(target_buffer, i_body, hessian)

Accumulates `hessian` to `target_buffer`.

"""
function set_hessian!(system::Matrix, i, hessian)
    @inbounds system[8, i] += hessian[1]
    @inbounds system[9, i] += hessian[2]
    @inbounds system[10, i] += hessian[3]
    @inbounds system[11, i] += hessian[4]
    @inbounds system[12, i] += hessian[5]
    @inbounds system[13, i] += hessian[6]
    @inbounds system[14, i] += hessian[7]
    @inbounds system[15, i] += hessian[8]
    @inbounds system[16, i] += hessian[9]
end
function set_hessian!(system::Matrix, switch::DerivativesSwitch{<:Any,<:Any,true}, i, hessian)
    r = hessian_range(switch)
    @inbounds system[r[1], i] += hessian[1]
    @inbounds system[r[2], i] += hessian[2]
    @inbounds system[r[3], i] += hessian[3]
    @inbounds system[r[4], i] += hessian[4]
    @inbounds system[r[5], i] += hessian[5]
    @inbounds system[r[6], i] += hessian[6]
    @inbounds system[r[7], i] += hessian[7]
    @inbounds system[r[8], i] += hessian[8]
    @inbounds system[r[9], i] += hessian[9]
end
set_hessian!(system::Matrix, ::DerivativesSwitch{<:Any,<:Any,false}, i, hessian) =
    throw(ArgumentError("hessian output is disabled for this target buffer"))

@inline function _set_third_derivative_packed!(system::Matrix, first_row, i, data::SVector{18})
    @inbounds for n in 1:18
        system[first_row + n - 1, i] += data[n]
    end
    return nothing
end

"""
    set_third_derivative!(target_buffer, i_body, value)
    set_third_derivative!(target_buffer, derivatives_switch, i_body, value)

Accumulates the packed third derivative `value` — a [`ThirdDerivativeTensor`](@ref) or an
`SVector{18}` in the packed `(xx,xy,xz,yy,yz,zz)`-per-component order — into the 18
third-derivative rows for the `i_body`th body of `target_buffer`. The three-argument form
assumes the default layout (rows 17:34); the switch-aware form uses
[`third_derivative_range`](@ref) and throws an `ArgumentError` if the switch did not
request third derivatives.
"""
set_third_derivative!(system::Matrix, i, tensor::ThirdDerivativeTensor) =
    _set_third_derivative_packed!(system, 17, i, packed_data(tensor))
set_third_derivative!(system::Matrix, i, data::SVector{18}) =
    _set_third_derivative_packed!(system, 17, i, data)
set_third_derivative!(system::Matrix, switch::DerivativesSwitch{<:Any,<:Any,<:Any,<:Any,<:Any,true}, i, tensor::ThirdDerivativeTensor) =
    _set_third_derivative_packed!(system, first(third_derivative_range(switch)), i, packed_data(tensor))
set_third_derivative!(system::Matrix, switch::DerivativesSwitch{<:Any,<:Any,<:Any,<:Any,<:Any,true}, i, data::SVector{18}) =
    _set_third_derivative_packed!(system, first(third_derivative_range(switch)), i, data)
set_third_derivative!(system::Matrix, ::DerivativesSwitch{<:Any,<:Any,<:Any,<:Any,<:Any,false}, i, value) =
    throw(ArgumentError("third-derivative output is disabled for this target buffer"))

#--- auxilliary functions ---#

function reset!(systems::Union{Tuple, AbstractVector{<:Matrix}})
    for system in systems
        reset!(system)
    end
end

function reset_small_buffers!(small_buffers::Vector{<:Matrix})
    for buffer in small_buffers
        buffer .= 0.0
    end
end

function reset!(system::Matrix, indices=1:size(system, 2))
    system[4:size(system, 1), indices] .= zero(eltype(system))
end

function reset_outputs!(system::Matrix, switch::DerivativesSwitch, indices=1:size(system, 2))
    system[output_range(switch), indices] .= zero(eltype(system))
end

function reset_outputs!(systems::Union{Tuple, AbstractVector{<:Matrix}}, switches::Tuple)
    for (system, switch) in zip(systems, switches)
        reset_outputs!(system, switch)
    end
end
