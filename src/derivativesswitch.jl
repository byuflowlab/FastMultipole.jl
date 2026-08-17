"""
    DerivativesSwitch(scalar_potential, gradient, hessian)

Constructs a tuple of [`DerivativesSwitch`](@ref) objects.

# Arguments

- `scalar_potential::Vector{Bool}`: a vector of `::Bool` indicating whether the scalar potential should be computed for each target system
- `gradient::Vector{Bool}`: a vector of `::Bool` indicating whether the vector field should be computed for each target system
- `hessian::Vector{Bool}`: a vector of `::Bool` indicating whether the vector gradient should be computed for each target system

"""
function DerivativesSwitch(scalar_potential, gradient, hessian; extra_outputs=0, metadata=0)
    extra_outputs = to_vector(extra_outputs, length(scalar_potential))
    metadata = to_vector(metadata, length(scalar_potential))
    return Tuple(DerivativesSwitch(ps, gs, hs; extra_outputs=no, metadata=nm) for (ps, gs, hs, no, nm) in zip(scalar_potential, gradient, hessian, extra_outputs, metadata))
end

"""
    DerivativesSwitch(scalar_potential, gradient, hessian)

Constructs a single [`DerivativesSwitch`](@ref) object.

# Arguments

- `scalar_potential::Bool`: a `::Bool` indicating whether the scalar potential should be computed for the target system
- `gradient::Bool`: a `::Bool` indicating whether the vector field should be computed for the target system
- `hessian::Bool`: a `::Bool` indicating whether the vector gradient should be computed for the target system

"""
function DerivativesSwitch(scalar_potential::Bool, gradient::Bool, hessian::Bool; extra_outputs=0, metadata=0)
    return DerivativesSwitch{scalar_potential, gradient, hessian, Int(extra_outputs), Int(metadata)}()
end

"""
    DerivativesSwitch(scalar_potential, gradient, hessian, target_systems)

Constructs a `::Tuple` of indentical [`DerivativesSwitch`](@ref) objects of the same length as `target_systems` (if it is a `::Tuple`), or a single [`DerivativesSwitch`](@ref) (if `target_system` is not a `::Tuple`)

# Arguments

- `scalar_potential::Bool`: a `::Bool` indicating whether the scalar potential should be computed for each target system
- `gradient::Bool`: a `::Bool` indicating whether the vector field should be computed for each target system
- `hessian::Bool`: a `::Bool` indicating whether the vector gradient should be computed for each target system

"""
function DerivativesSwitch(scalar_potential::Bool, gradient::Bool, hessian::Bool, target_systems::Tuple; extra_outputs=0, metadata=nothing)
    extra_outputs = to_vector(extra_outputs, length(target_systems))
    metadata = metadata_vector(metadata, target_systems)
    return Tuple(DerivativesSwitch{scalar_potential, gradient, hessian, Int(extra_outputs[i]), Int(metadata[i])}() for i in eachindex(target_systems))
end

function DerivativesSwitch(scalar_potential, gradient, hessian, target_systems::Tuple; extra_outputs=0, metadata=nothing)
    @assert length(scalar_potential) == length(gradient) == length(hessian) == length(target_systems) "length of inputs to DerivativesSwitch inconsistent"
    extra_outputs = to_vector(extra_outputs, length(target_systems))
    metadata = metadata_vector(metadata, target_systems)
    return Tuple(DerivativesSwitch{scalar_potential[i], gradient[i], hessian[i], Int(extra_outputs[i]), Int(metadata[i])}() for i in eachindex(target_systems))
end

function DerivativesSwitch(scalar_potential::Bool, gradient::Bool, hessian::Bool, target_system; extra_outputs=0, metadata=nothing)
    metadata = isnothing(metadata) ? metadata_per_body(target_system) : metadata
    return DerivativesSwitch{scalar_potential, gradient, hessian, Int(extra_outputs), Int(metadata)}()
end

DerivativesSwitch() = DerivativesSwitch{true, true, true, 0, 0}()

"""
    scalar_potential_index(switch)

Target-buffer row for scalar potential output.
"""
@inline scalar_potential_index(::DerivativesSwitch{true,GS,HS,NO,NM}) where {GS,HS,NO,NM} = 4 + NM
@inline scalar_potential_index(::DerivativesSwitch{false,GS,HS,NO,NM}) where {GS,HS,NO,NM} = 0

@inline _standard_output_rows(::DerivativesSwitch{PS,GS,HS,NO,NM}) where {PS,GS,HS,NO,NM} =
    (PS ? 1 : 0) + (GS ? 3 : 0) + (HS ? 9 : 0)

@inline target_buffer_rows(::DerivativesSwitch{PS,GS,HS,NO,NM}) where {PS,GS,HS,NO,NM} =
    3 + NM + _standard_output_rows(DerivativesSwitch{PS,GS,HS,NO,NM}()) + NO

"""
    gradient_range(switch)

Target-buffer rows for gradient output.
"""
@inline gradient_range(::DerivativesSwitch{PS,true,HS,NO,NM}) where {PS,HS,NO,NM} =
    4 + NM + (PS ? 1 : 0) : 3 + NM + (PS ? 1 : 0) + 3
@inline gradient_range(::DerivativesSwitch{PS,false,HS,NO,NM}) where {PS,HS,NO,NM} =
    4 + NM + (PS ? 1 : 0) : 3 + NM + (PS ? 1 : 0)
"""
    hessian_range(switch)

Target-buffer rows for hessian output.
"""
@inline hessian_range(::DerivativesSwitch{PS,GS,true,NO,NM}) where {PS,GS,NO,NM} =
    4 + NM + (PS ? 1 : 0) + (GS ? 3 : 0) : 3 + NM + (PS ? 1 : 0) + (GS ? 3 : 0) + 9
@inline hessian_range(::DerivativesSwitch{PS,GS,false,NO,NM}) where {PS,GS,NO,NM} =
    4 + NM + (PS ? 1 : 0) + (GS ? 3 : 0) : 3 + NM + (PS ? 1 : 0) + (GS ? 3 : 0)
"""
    metadata_range(switch)

Rows in a target buffer that hold sorted metadata.
"""
@inline metadata_range(::DerivativesSwitch{PS,GS,HS,NO,NM}) where {PS,GS,HS,NO,NM} = 4 : 3 + NM
"""
    metadata_index(switch, j)

Absolute target-buffer row for metadata row `j`.
"""
@inline metadata_index(switch::DerivativesSwitch, j) = first(metadata_range(switch)) + j - 1
"""
    tree_carried_range(switch)

Rows copied during target tree sorting. This includes position and metadata,
but excludes accumulated output rows.
"""
@inline tree_carried_range(::DerivativesSwitch{PS,GS,HS,NO,NM}) where {PS,GS,HS,NO,NM} = 1 : 3 + NM
"""
    standard_output_range(switch)

Rows used for scalar potential, gradient, and hessian outputs.
"""
@inline standard_output_range(switch::DerivativesSwitch{PS,GS,HS,NO,NM}) where {PS,GS,HS,NO,NM} =
    4 + NM : 3 + NM + _standard_output_rows(switch)
"""
    extra_output_range(switch)

Rows used for caller-requested extra accumulated outputs.
"""
@inline extra_output_range(switch::DerivativesSwitch{PS,GS,HS,NO,NM}) where {PS,GS,HS,NO,NM} =
    4 + NM + _standard_output_rows(switch) : 3 + NM + _standard_output_rows(switch) + NO
"""
    output_range(switch)

All accumulated output rows, including standard outputs and extra outputs.
"""
@inline output_range(switch::DerivativesSwitch) = first(standard_output_range(switch)) : last(extra_output_range(switch))

"""
    get_extra_output(buffer, switch, i, j)

Returns extra output row `j` for target-buffer column `i`.
"""
@inline function get_extra_output(buffer::AbstractMatrix, switch::DerivativesSwitch, i, j)
    return @inbounds buffer[first(extra_output_range(switch)) + j - 1, i]
end

"""
    set_extra_output!(buffer, switch, i, j, value)

Accumulates `value` into extra output row `j` for target-buffer column `i`.
"""
@inline function set_extra_output!(buffer::AbstractMatrix, switch::DerivativesSwitch, i, j, value)
    @inbounds buffer[first(extra_output_range(switch)) + j - 1, i] += value
    return nothing
end

"""
    extra_output_view(buffer, switch, i)

Returns a view of all extra output rows for target-buffer column `i`.
"""
@inline extra_output_view(buffer::AbstractMatrix, switch::DerivativesSwitch, i) = view(buffer, extra_output_range(switch), i)
@inline output_view(buffer::AbstractMatrix, switch::DerivativesSwitch, i) = view(buffer, output_range(switch), i)
