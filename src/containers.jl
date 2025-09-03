#------- dispatch for common interface for external packages -------#

abstract type Indexable end

struct Position <: Indexable end

struct Radius <: Indexable end

struct ScalarPotential <: Indexable end

struct Velocity <: Indexable end

struct VelocityGradient <: Indexable end

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

Switch indicating whether the scalar potential, vector potential, velocity, and/or velocity gradient should be computed for a target system. Information is stored as type parameters, allowing the compiler to compile away if statements.
"""
struct DerivativesSwitch{PS,VS,GS} end

"""
    ExpansionSwitch

Switch indicating which expansions should be used:

1. scalar potential (`SP`)
2. vector potential via Lamb-Helmholtz decomposition (`VP`)
"""
struct ExpansionSwitch{SP,VP} end

#####
##### cost parameters
#####
# abstract type CostParameters end

# struct SingleCostParameters <: CostParameters
#     alloc_M2M_L2L::SVector{3,Float64}
#     tau_M2M_L2L::SVector{5,Float64}
#     alloc_M2L::SVector{3,Float64}
#     tau_M2L::SVector{5,Float64}
#     alloc_L2B::SVector{3,Float64}
#     tau_L2B::SVector{3,Float64}
#     C_nearfield::Float64
#     tau_B2M::SVector{3,Float64}
# end

# SingleCostParameters(;
#     alloc_M2M_L2L = ALLOC_M2M_L2L_DEFAULT,
#     tau_M2M_L2L = TAU_M2M_DEFAULT,
#     alloc_M2L = ALLOC_M2L_DEFAULT,
#     tau_M2L = TAU_M2L_DEFAULT,
#     tau_L2L = TAU_L2L_DEFAULT,
#     alloc_L2B = ALLOC_L2B_DEFAULT,
#     tau_L2B = TAU_L2B_DEFAULT,
#     C_nearfield = C_NEARFIELD_DEFAULT,
#     tau_B2M = TAU_B2M_DEFAULT
# ) = SingleCostParameters(alloc_M2M_L2L, tau_M2M_L2L, alloc_M2L, tau_M2L, alloc_L2B, tau_L2B, C_nearfield, tau_B2M)

# struct MultiCostParameters{N} <: CostParameters
#     alloc_M2M_L2L::SVector{3,Float64}
#     tau_M2M_L2L::SVector{5,Float64}
#     alloc_M2L::SVector{3,Float64}
#     tau_M2L::SVector{5,Float64}
#     alloc_L2B::SVector{3,Float64}
#     tau_L2B::SVector{3,Float64}
#     C_nearfield::SVector{N,Float64}
#     tau_B2M::SVector{N,SVector{3,Float64}}
# end

# MultiCostParameters{N}(;
#     alloc_M2M_L2L = ALLOC_M2M_L2L_DEFAULT,
#     tau_M2M_L2L = TAU_M2M_L2L_DEFAULT,
#     alloc_M2L = ALLOC_M2L_DEFAULT,
#     tau_M2L = TAU_M2L_DEFAULT,
#     alloc_L2B = ALLOC_L2B_DEFAULT,
#     tau_L2B = TAU_L2B_DEFAULT,
#     C_nearfield = SVector{N,Float64}(C_NEARFIELD_DEFAULT for _ in 1:N),
#     tau_B2M = SVector{N,SVector{3,Float64}}(TAU_B2M_DEFAULT for _ in 1:N)
# ) where N = MultiCostParameters{N}(alloc_M2M_L2L, tau_M2M_L2L, alloc_M2L, tau_M2L, tau_L2L, alloc_L2B, tau_L2B, C_nearfield, tau_B2M)

# CostParameters(systems::Tuple) = MultiCostParameters()
# CostParameters(system) = SingleCostParameters()

# CostParameters(alloc_M2M_L2L, tau_B2M, alloc_M2L, tau_M2L, tau_L2L, alloc_L2B, tau_L2B, C_nearfield::Float64, tau_M2M_L2L) =
#     SingleCostParameters(alloc_M2M_L2L, tau_B2M, alloc_M2L, tau_M2L, tau_L2L, alloc_L2B, tau_L2B, C_nearfield, tau_M2M_L2L)

# CostParameters(alloc_M2M_L2L, tau_B2M, alloc_M2L, tau_M2L, tau_L2L, alloc_L2B, tau_L2B, C_nearfield::SVector, tau_M2M_L2L) =
#     MultiCostParameters(alloc_M2M_L2L, tau_B2M, alloc_M2L, tau_M2L, tau_L2L, alloc_L2B, tau_L2B, C_nearfield, tau_M2M_L2L)

#------- error predictors -------#

abstract type ErrorMethod end

abstract type RelativeError <: ErrorMethod end

abstract type AbsoluteError <: ErrorMethod end

struct UnequalSpheres <: ErrorMethod end

struct UnequalBoxes <: ErrorMethod end

struct UniformUnequalSpheres <: ErrorMethod end

struct UniformUnequalBoxes <: ErrorMethod end

struct RotatedCoefficients <: ErrorMethod end

#------- dynamic expansion order -------#

struct AbsoluteUpperBound{ε} <: AbsoluteError end
AbsoluteUpperBound(ε) = AbsoluteUpperBound{ε}()

struct PowerAbsolutePotential{ε,BE} <: AbsoluteError end
PowerAbsolutePotential(ε, BE::Bool=true) = PowerAbsolutePotential{ε,BE}()

struct PowerAbsoluteVelocity{ε,BE} <: AbsoluteError end
PowerAbsoluteVelocity(ε, BE::Bool=true) = PowerAbsoluteVelocity{ε,BE}()

struct RotatedCoefficientsAbsoluteVelocity{ε,BE} <: AbsoluteError end
RotatedCoefficientsAbsoluteVelocity(ε, BE::Bool=true) = RotatedCoefficientsAbsoluteVelocity{ε,BE}()

struct RelativeUpperBound{ε} <: RelativeError end
RelativeUpperBound(ε) = RelativeUpperBound{ε}()

struct PowerRelativePotential{ε,BE} <: RelativeError end
PowerRelativePotential(ε, BE::Bool=true) = PowerRelativePotential{ε,BE}()

struct PowerRelativeVelocity{ε,BE} <: RelativeError end
PowerRelativeVelocity(ε, BE::Bool=true) = PowerRelativeVelocity{ε,BE}()

struct RotatedCoefficientsRelativeVelocity{ε,BE} <: RelativeError end
RotatedCoefficientsRelativeVelocity(ε, BE::Bool=true) = RotatedCoefficientsRelativeVelocity{ε,BE}()

#------- interaction list -------#

abstract type InteractionListMethod{TS} end

struct Barba{TS} <: InteractionListMethod{TS} end

Barba(TS=SortByTarget()) = Barba{TS}()

struct SortByTarget end

struct SortBySource end

struct SelfTuning{TS} <: InteractionListMethod{TS} end

SelfTuning(TS=SortByTarget()) = SelfTuning{TS}()

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
* `source_radius::TF`: if this branch is a leaf, distance from `center` to the outer edge of farthest body contained in this branch; otherwise, this is the distance from `center` to the corner of its `source_box`
* `target_radius::TF`: distance from `center` to the farthest body center contained in this branch
* `source_box::Vector{TF}`: vector of length 6 containing the distances from the center to faces of a rectangular prism completely enclosing all bodies with their finite radius in the negative x, positive x, negative y, positive y, negative z, and positive z directions, respectively
* `target_box::Vector{TF}`: vector of length 3 containing the distances from the center to faces of a rectangular prism completely enclosing all body centers in the x, y, and z direction, respectively
* multipole_expansion::Array{TF,3}`: array of size (2,2,) containing the multipole expansion coefficients; the first index indicates real or imaginary, the second index indicates scalar potential or the second component of the Lamb-Helmholtz decomposition, and the third index `k` indicates the expansion coefficient of degree \$n\$ and order \$m\$, as \$k = p(p+1)/2 + m + 1\$
* local_expansion::Array{TF,3}`: array of size `(2,2,(p+1)(p+2)/2)` containing the local expansion coefficients; the first index indicates real or imaginary, the second index indicates scalar potential or the second component of the Lamb-Helmholtz decomposition, and the third index `k` indicates the expansion coefficient of degree \$n\$ and order \$m\$, as \$k = p(p+1)/2 + m + 1\$
* `harmonics::Array{TF,3}`: array of size `(2,2,(p+1)(p+2)/2)` used as storage for regular harmonics, irregular harmonics, or whatever is needed, and is indexed as multipole and local expansions
* `lock::ReentrantLock`: lock used to avoid data race conditions when modifying this branch or its corresponding bodies

"""
struct Branch{TF,N}
    n_bodies::SVector{N,Int64}
    bodies_index::SVector{N,UnitRange{Int64}}
    n_branches::Int64
    branch_index::UnitRange{Int64}
    i_parent::Int64
    i_leaf::Int64
    source_center::SVector{3,TF}   # center of the branch
    target_center::SVector{3,TF}   # center of the branch
    source_radius::TF
    target_radius::TF
    source_box::SVector{3,TF} # x, y, and z half widths of the box encapsulating all sources
    target_box::SVector{3,TF} # x, y, and z half widths of the box encapsulating all sources
    # multipole_expansion::Array{TF,3} # multipole expansion coefficients
    # local_expansion::Array{TF,3}     # local expansion coefficients
    # harmonics::Array{TF,3}
    lock::ReentrantLock
    max_influence::TF
end

#=function Branch(n_bodies, bodies_index, n_branches, branch_index, i_parent, i_leaf, source_center, target_center, source_radius, target_radius, source_box, target_box, lock, max_influence)
    TF = promote_type(eltype(source_center), eltype(target_center), eltype(source_radius), eltype(target_radius), eltype(source_box), eltype(target_box), eltype(max_influence))
    Branch(n_bodies, bodies_index, n_branches, branch_index, i_parent, i_leaf, TF.(source_center), TF.(target_center), TF(source_radius), TF(target_radius), TF.(source_box), TF.(target_box), lock, TF(max_influence))
end=#

# function Branch(n_bodies, bodies_index, n_branches, branch_index, i_parent, i_leaf, source_center, target_center, source_radius::TF, target_radius, source_box, target_box, lock, max_influence=zero(TF)) where TF
#     Branch(n_bodies, bodies_index, n_branches, branch_index, i_parent, i_leaf, source_center, target_center, source_radius, target_radius, source_box, target_box, error, lock, max_influence)
# end

function Branch(n_bodies::SVector{<:Any,Int64}, bodies_index, n_branches, branch_index, i_parent::Int, i_leaf_index, source_center, target_center, source_radius, target_radius, source_box, target_box)
    return Branch(n_bodies, bodies_index, n_branches, branch_index, i_parent, i_leaf_index, source_center, target_center, source_radius, target_radius, source_box, target_box, ReentrantLock(), zero(target_radius))
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

function Tree(branches, expansions, levels_index, left_index, sort_index_list, inverse_sort_index_list, buffers, small_buffers, expansion_order, leaf_size)
    return Tree(branches, TF.(expansions), levels_index, left_index, sort_index_list, inverse_sort_index_list, TF.(buffers), TF.(small_buffers), expansion_order, leaf_size)
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

Convenience system for defining locations at which the potential, velocity, or velocity gradient may be desired.
"""
struct ProbeSystem{TF}
    position::Vector{SVector{3,TF}}
    scalar_potential::Vector{TF}
    velocity::Vector{SVector{3,TF}}
    velocity_gradient::Vector{SMatrix{3,3,TF,9}}
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
    multipole_threshold::Float64
    lamb_helmholtz::Bool
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
