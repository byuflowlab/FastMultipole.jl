"""
    DerivativesSwitch(scalar_potential, gradient, hessian)

Constructs a tuple of [`DerivativesSwitch`](@ref) objects.

# Arguments

- `scalar_potential::Vector{Bool}`: a vector of `::Bool` indicating whether the scalar potential should be computed for each target system
- `gradient::Vector{Bool}`: a vector of `::Bool` indicating whether the vector field should be computed for each target system
- `hessian::Vector{Bool}`: a vector of `::Bool` indicating whether the vector gradient should be computed for each target system

"""
function DerivativesSwitch(scalar_potential, gradient, hessian)
    return Tuple(DerivativesSwitch(ps,gs,hs) for (ps,gs,hs) in zip(scalar_potential, gradient, hessian))
end

"""
    DerivativesSwitch(scalar_potential, gradient, hessian)

Constructs a single [`DerivativesSwitch`](@ref) object.

# Arguments

- `scalar_potential::Bool`: a `::Bool` indicating whether the scalar potential should be computed for the target system
- `gradient::Bool`: a `::Bool` indicating whether the vector field should be computed for the target system
- `hessian::Bool`: a `::Bool` indicating whether the vector gradient should be computed for the target system

"""
function DerivativesSwitch(scalar_potential::Bool, gradient::Bool, hessian::Bool)
    return DerivativesSwitch{scalar_potential, gradient, hessian}()
end

"""
    DerivativesSwitch(scalar_potential, gradient, hessian, target_systems)

Constructs a `::Tuple` of indentical [`DerivativesSwitch`](@ref) objects of the same length as `target_systems` (if it is a `::Tuple`), or a single [`DerivativesSwitch`](@ref) (if `target_system` is not a `::Tuple`)

# Arguments

- `scalar_potential::Bool`: a `::Bool` indicating whether the scalar potential should be computed for each target system
- `gradient::Bool`: a `::Bool` indicating whether the vector field should be computed for each target system
- `hessian::Bool`: a `::Bool` indicating whether the vector gradient should be computed for each target system

"""
function DerivativesSwitch(scalar_potential::Bool, gradient::Bool, hessian::Bool, target_systems::Tuple)
    return Tuple(DerivativesSwitch{scalar_potential, gradient, hessian}() for _ in target_systems)
end

function DerivativesSwitch(scalar_potential, gradient, hessian, target_systems::Tuple)
    @assert length(scalar_potential) == length(gradient) == length(hessian) == length(target_systems) "length of inputs to DerivativesSwitch inconsistent"
    return Tuple(DerivativesSwitch{scalar_potential[i], gradient[i], hessian[i]}() for i in eachindex(target_systems))
end

function DerivativesSwitch(scalar_potential::Bool, gradient::Bool, hessian::Bool, target_system)
    return DerivativesSwitch{scalar_potential, gradient, hessian}()
end

DerivativesSwitch() = DerivativesSwitch{true, true, true}()

