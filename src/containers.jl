#------- dispatch for common interface for external packages -------#

abstract type Indexable end

struct Position <: Indexable end

struct Radius <: Indexable end

struct ScalarPotential <: Indexable end

struct Gradient <: Indexable end

struct Hessian <: Indexable end

struct Vertex <: Indexable end

struct Normal <: Indexable end

struct Strength <: Indexable end

#------- dispatch convenience functions for multipole creation definition -------#

abstract type AbstractKernel end

abstract type Vortex <: AbstractKernel end

abstract type Source <: AbstractKernel end

abstract type SourceVortex <: AbstractKernel end

abstract type Dipole <: AbstractKernel end

abstract type SourceDipole <: AbstractKernel end

abstract type AbstractElement{TK<:AbstractKernel} end

abstract type Point{TK} <: AbstractElement{TK} end

abstract type Filament{TK} <: AbstractElement{TK} end

abstract type Panel{NS,TK} <: AbstractElement{TK} end

#------- dispatch convenience functions to determine which derivatives are desired -------#

"""
    DerivativesSwitch

Switch indicating whether the scalar potential, vector potential, gradient, and/or hessian should be computed for a target system. Information is stored as type parameters, allowing the compiler to compile away if statements.
"""
struct DerivativesSwitch{PS,GS,HS} end

"""
    ExpansionSwitch

Switch indicating which expansions should be used:

1. scalar potential (`SP`)
2. vector potential via Lamb-Helmholtz decomposition (`VP`)
"""
struct ExpansionSwitch{SP,VP} end

#------- error predictors -------#

abstract type ErrorMethod{BE} end

abstract type AbsoluteErrorMethod{AET,BE} <: ErrorMethod{BE} end

abstract type RelativeErrorMethod{RET,AET,BE} <: ErrorMethod{BE} end

struct UnequalSpheres{BE} <: ErrorMethod{BE} end

struct UnequalBoxes{BE} <: ErrorMethod{BE} end

struct UniformUnequalSpheres{BE} <: ErrorMethod{BE} end

struct UniformUnequalBoxes{BE} <: ErrorMethod{BE} end

struct RotatedCoefficients{BE} <: ErrorMethod{BE} end

#------- dynamic expansion order -------#

# struct AbsoluteUpperBound{ε} <: AbsoluteError end
# AbsoluteUpperBound(ε) = AbsoluteUpperBound{ε}()

struct PowerAbsolutePotential{ε,BE} <: AbsoluteErrorMethod{ε,BE} end
PowerAbsolutePotential(ε, BE::Bool=true) = PowerAbsolutePotential{ε,BE}()

struct PowerAbsoluteGradient{ε,BE} <: AbsoluteErrorMethod{ε,BE} end
PowerAbsoluteGradient(ε, BE::Bool=true) = PowerAbsoluteGradient{ε,BE}()

struct RotatedCoefficientsAbsoluteGradient{ε,BE} <: AbsoluteErrorMethod{ε,BE} end
RotatedCoefficientsAbsoluteGradient(ε, BE::Bool=true) = RotatedCoefficientsAbsoluteGradient{ε,BE}()

# struct RelativeUpperBound{ε} <: RelativeErrorMethod end
# RelativeUpperBound(ε) = RelativeUpperBound{ε}()

struct PowerRelativePotential{ε_rel,ε_abs,BE} <: RelativeErrorMethod{ε_rel,ε_abs,BE} end
PowerRelativePotential(ε_rel, ε_abs=sqrt(eps()), BE::Bool=true) = PowerRelativePotential{ε_rel,ε_abs,BE}()

struct PowerRelativeGradient{ε_rel,ε_abs,BE} <: RelativeErrorMethod{ε_rel,ε_abs,BE} end
PowerRelativeGradient(ε_rel, ε_abs=sqrt(eps()), BE::Bool=true) = PowerRelativeGradient{ε_rel,ε_abs,BE}()

struct RotatedCoefficientsRelativeGradient{ε_rel,ε_abs,BE} <: RelativeErrorMethod{ε_rel,ε_abs,BE} end
RotatedCoefficientsRelativeGradient(ε_rel, ε_abs=sqrt(eps()), BE::Bool=true) = RotatedCoefficientsRelativeGradient{ε_rel,ε_abs,BE}()

#------- interaction list -------#

abstract type InteractionListMethod end

struct Barba <: InteractionListMethod end
struct SelfTuning <: InteractionListMethod end
struct SelfTuningTreeStop <: InteractionListMethod end

#------- octree creation -------#

"""
    Branch{TF,N}

Branch object used to sort more than one system into an octree. Type parameters represent:

* `TF`: the floating point type (would be a dual number if using algorithmic differentiation)
* `N`: the number of systems represented

**Fields**

* `bodies_index::Vector{UnitRange}`: vector of unit ranges indicating the index of bodies in each represented system, respectively
* `n_branches::Int`: number of child branches corresponding to this branch
* `branch_index::UnitRange`: indices of this branch's child branches
* `i_parent::Int`: index of this branch's parent
* `i_leaf::Int`: if this branch is a leaf, what is its index in its parent `<:Tree`'s `leaf_index` field
* `center::Vector{TF}`: center of this branch at which its multipole and local expansions are centered
* `radius::TF`: distance from `center` to the farthest body contained in this branch (accounting for finite body radius if bodies are sources)
* `box::Vector{TF}`: vector of length 3 containing the distances from the center to faces of a rectangular prism completely enclosing all bodies in the x, y, and z direction, respectively
* `min_potential::TF`: maximum influence of any body in this branch on any body in its child branches; used to enforce a relative error tolerance
* `min_gradient::TF`: maximum gradient magnitude of any body in this branch on any body in its child branches; used to enforce a relative error tolerance

"""
struct Branch{TF,N}
    n_bodies::SVector{N,Int64}
    bodies_index::SVector{N,UnitRange{Int64}}
    n_branches::Int64
    branch_index::UnitRange{Int64}
    i_parent::Int64
    i_leaf::Int64
    center::SVector{3,TF}   # center of the branch
    radius::TF
    box::SVector{3,TF} # x, y, and z half widths of the box encapsulating all member bodies
    min_potential::TF
    min_gradient::TF
end

function Branch(n_bodies::SVector{<:Any,Int64}, bodies_index, n_branches, branch_index, i_parent::Int, i_leaf_index, center, radius, box)
    return Branch(n_bodies, bodies_index, n_branches, branch_index, i_parent, i_leaf_index, center, radius, box, zero(radius), zero(radius))
end

function Branch(bodies_index::SVector{<:Any,UnitRange{Int64}}, args...)
    n_bodies = SVector{length(bodies_index), Int}(length(bodies_i) for bodies_i in bodies_index)
    return Branch(n_bodies, bodies_index, args...)
end


Base.eltype(::Branch{TF,<:Any}) where TF = TF

"""
bodies[index_list] is the same sort operation as performed by the tree
sorted_bodies[inverse_index_list] undoes the sort operation performed by the tree
"""
struct Tree{TF,N}
    branches::Vector{Branch{TF,N}}        # a vector of `Branch` objects composing the tree
    expansions::Array{TF,4}
    levels_index::Vector{UnitRange{Int64}}
    leaf_index::Vector{Int}
    sort_index_list::NTuple{N,Vector{Int}}
    inverse_sort_index_list::NTuple{N,Vector{Int}}
    buffers::NTuple{N,Matrix{TF}}
    small_buffers::Vector{Matrix{TF}}
    expansion_order::Int64
    leaf_size::SVector{N,Int64}    # max number of bodies in a leaf
    # cost_parameters::MultiCostParameters{N}
    # cost_parameters::SVector{N,Float64}
end

struct InteractionList{TF}
    influence_matrices::Vector{Matrix{TF}}
    strengths::Vector{TF}
    influence::Vector{TF}
    direct_list::Vector{SVector{2,Int32}}
end

Base.length(list::InteractionList) = length(list.direct_list)

#####
##### allow input systems to take any form when desired
#####
struct SortWrapper{TS}
    system::TS
    index::Vector{Int}
end

#####
##### when we desire to evaluate the potential at locations not coincident with source centers
#####

"""
    ProbeSystem

Convenience system for defining locations at which the potential, vector field, or vector gradient may be desired.
"""
struct ProbeSystem{TF}
    position::Vector{SVector{3,TF}}
    scalar_potential::Vector{TF}
    gradient::Vector{SVector{3,TF}}
    hessian::Vector{SMatrix{3,3,TF,9}}
end

#------- SOLVERS -------#

abstract type AbstractSolver end

struct Matrices{TF}
    data::Vector{TF}
    rhs::Vector{TF}
    sizes::Vector{Tuple{Int,Int}}
    matrix_offsets::Vector{Int}
    rhs_offsets::Vector{Int}
end

struct FastGaussSeidel{TF,Nsys,TIL} <: AbstractSolver
    self_matrices::Matrices{TF}
    nonself_matrices::Matrices{TF}
    index_map::Vector{UnitRange{Int}}
    m2l_list::Vector{SVector{2,Int}}
    direct_list::Vector{SVector{2,Int32}}
    full_direct_list::Vector{SVector{2,Int32}}
    interaction_list_method::TIL
    multipole_acceptance::Float64
    strengths::Vector{TF}
    strengths_by_leaf::Vector{UnitRange{Int}}
    targets_by_branch::Vector{UnitRange{Int}}
    source_tree::Tree{TF,Nsys}
    target_tree::Tree{TF,Nsys}
    old_influence_storage::Vector{TF}
    extra_right_hand_side::Vector{TF}
    influences_per_system::Vector{Vector{TF}}
    residual_vector::Vector{TF}
end

#--- memory cache ---#

struct Cache{TF,NT,NS}
    target_buffers::NTuple{NT, Matrix{TF}}
    source_buffers::NTuple{NS, Matrix{TF}}
    target_small_buffers::Vector{Matrix{TF}}
    source_small_buffers::Vector{Matrix{TF}}
end

function Cache(; 
    target_buffers::NTuple{NT,Matrix{TF}}, 
    source_buffers::NTuple{NS,Matrix{TF}}, 
    target_small_buffers::Vector{Matrix{TF}}, 
    source_small_buffers::Vector{Matrix{TF}}
        ) where {TF,NT,NS}
    return Cache{TF,NT,NS}(target_buffers, source_buffers, target_small_buffers, source_small_buffers)
end

function Cache(target_systems::Tuple, source_systems::Tuple)
    # get float type
    TF = get_type(target_systems, source_systems)

    # allocate buffers
    target_buffers = allocate_buffers(target_systems, true, TF)
    source_buffers = allocate_buffers(source_systems, false, TF)
    target_small_buffers = allocate_small_buffers(target_systems, TF)
    source_small_buffers = allocate_small_buffers(source_systems, TF)
    
    # return cache
    return Cache{TF,length(target_systems),length(source_systems)}(target_buffers, source_buffers, target_small_buffers, source_small_buffers)
end