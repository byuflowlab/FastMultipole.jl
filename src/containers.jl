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
    DerivativesSwitch{PS,GS,HS,NO,NM}

Switch indicating whether scalar potential (`PS`), gradient (`GS`), and hessian (`HS`)
outputs should be computed for a target system. `NO` is the number of extra
accumulated output rows requested by the caller, and `NM` is the number of
metadata rows carried with target positions through tree sorting.

Target buffers use a compact row layout: positions in rows `1:3`, metadata in
rows `4:3+NM`, enabled standard outputs in scalar/gradient/hessian order, and
then `NO` extra output rows. Disabled standard outputs do not reserve rows, so
custom target-buffer code should use switch-aware layout helpers.

Use `DerivativesSwitch(scalar_potential, gradient, hessian; extra_outputs=0,
metadata=0)` for a single switch, or pass target systems as the fourth argument
to infer `metadata_per_body(system)` when `metadata=nothing`. Existing calls
such as `DerivativesSwitch(true, true, false)` remain valid.
"""
struct DerivativesSwitch{PS,GS,HS,NO,NM} end

#------- error predictors -------#

abstract type ErrorMethod{BE} end

abstract type AbsoluteErrorMethod{AET,BE} <: ErrorMethod{BE} end

abstract type RelativeErrorMethod{RET,AET,BE} <: ErrorMethod{BE} end

struct UnequalSpheres{BE} <: ErrorMethod{BE} end
UnequalSpheres(BE=false) = UnequalSpheres{BE}()

struct PringleAbsolutePotential{BE} <: ErrorMethod{BE} end
PringleAbsolutePotential(BE=false) = PringleAbsolutePotential{BE}()

struct PringleRelativePotential{BE} <: ErrorMethod{BE} end
PringleRelativePotential(BE=false) = PringleRelativePotential{BE}()

struct DehnenAbsoluteGradient{BE} <: ErrorMethod{BE} end
DehnenAbsoluteGradient(BE=false) = DehnenAbsoluteGradient{BE}()

struct UnequalSpheresMultipoleGradient{BE} <: ErrorMethod{BE} end
UnequalSpheresMultipoleGradient() = UnequalSpheresMultipoleGradient{false}()

struct UnequalSpheresGradient{BE} <: ErrorMethod{BE} end
UnequalSpheresGradient() = UnequalSpheresGradient{false}()

struct HeuristicRelativePotential{BE} <: ErrorMethod{BE} end
HeuristicRelativePotential(BE=false) = HeuristicRelativePotential{BE}()

struct HeuristicAbsolutePotential{BE} <: ErrorMethod{BE} end
HeuristicAbsolutePotential(BE=false) = HeuristicAbsolutePotential{BE}()

struct UnequalBoxes{BE} <: ErrorMethod{BE} end

struct UniformUnequalSpheres{BE} <: ErrorMethod{BE} end

struct UniformUnequalBoxes{BE} <: ErrorMethod{BE} end

struct RotatedCoefficients{BE} <: ErrorMethod{BE} end

#------- dynamic expansion order -------#

# struct AbsoluteUpperBound{ε} <: AbsoluteError end
# AbsoluteUpperBound(ε) = AbsoluteUpperBound{ε}()

struct PowerAbsolutePotential{ε,BE} <: AbsoluteErrorMethod{ε,BE} end
PowerAbsolutePotential(ε, BE::Bool=true) = PowerAbsolutePotential{ε,BE}()

struct PowerAbsolutePotentialMultipole{ε,BE} <: AbsoluteErrorMethod{ε,BE} end
PowerAbsolutePotentialMultipole(ε, BE::Bool=true) = PowerAbsolutePotentialMultipole{ε,BE}()

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
struct SelfTuningTargetStop <: InteractionListMethod end

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

function Branch(bodies_index::UnitRange{Int64}, args...)
    return Branch(SVector{1,UnitRange{Int64}}((bodies_index,)), args...)
end


Base.eltype(::Branch{TF,<:Any}) where TF = TF

"""
    Tree{TF,N}

Tree object used to sort `N` systems into an octree.

**Fields**

* `branches::Vector{Branch{TF,N}}`: a vector of `Branch` objects composing the tree
* `expansions::Array{TF,4}`: 4-dimensional array whose `(1,i,j,k)`th element contains the real part of the `j`th expansion coefficient of the `k`th branch, and whose `(2,i,j,k)`th element contains the imaginary part. If `i==1`, the coefficient corresponds to the scalar potential; if `i==2`, the coefficient corresponds to the ''\\chi'' part of the Lamb-Helmholtz decomposition of the vector potential.
* `levels_index::Vector{UnitRange{Int64}}`: vector of unit ranges indicating the indices of branches at each level of the tree
* `leaf_index::Vector{Int}`: vector of indices of branches that are leaves
* `sort_index_list::NTuple{N,Vector{Int}}`: tuple of vectors of indices used to sort the bodies in each system into the tree
* `inverse_sort_index_list::NTuple{N,Vector{Int}}`: tuple of vectors of indices used to undo the sort operation performed by `sort_index_list`
* `buffers::Vector{Matrix{TF}}`: vector of buffers used to store the bodies computed influence of each system in the tree, as explained in [`FastMultipole.allocate_buffers`](@ref)
* `small_buffers::Vector{Matrix{TF}}`: vector of buffers used to pidgeon-hole sort bodies into the tree, as explained in [`FastMultipole.allocate_small_buffers`](@ref)
* `expansion_order::Int64`: the maximum storable expansion order
* `leaf_size::SVector{N,Int64}`: maximum number of bodies in a leaf for each system; if multiple systems are represented, the actual maximum depends on the `InteractionListMethod` used to create the tree

"""
struct Tree{TF,N}
    # bodies[index_list] is the same sort operation as performed by the tree
    # sorted_bodies[inverse_index_list] undoes the sort operation performed by the tree
    branches::Vector{Branch{TF,N}}        # a vector of `Branch` objects composing the tree
    expansions::Array{TF,4}
    levels_index::Vector{UnitRange{Int64}}
    leaf_index::Vector{Int}
    sort_index_list::NTuple{N,Vector{Int}}
    inverse_sort_index_list::NTuple{N,Vector{Int}}
    buffers::Vector{Matrix{TF}}
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
##### when we desire to evaluate the potential at locations not coincident with source centers
#####

"""
    ProbeSystem{TF}

Convenience system for defining locations at which the potential, vector field, or vector gradient may be desired. Interface functions are already defined and overloaded.

**Fields**

* `position::Vector{SVector{3,TF}}`: vector of probe positions
* `scalar_potential::Vector{TF}`: vector of scalar potential values at the positions
* `gradient::Vector{SVector{3,TF}}`: vector of vector field values at the positions
* `hessian::Vector{SMatrix{3,3,TF,9}}`: vector of Hessian matrices at the positions
"""
abstract type ProbeSystem{TF} end

struct ProbeSystemStatic{TF} <: ProbeSystem{TF}
    position::Vector{SVector{3,TF}}
    scalar_potential::Vector{TF}
    gradient::Vector{SVector{3,TF}}
    hessian::Vector{SMatrix{3,3,TF,9}}
end

struct ProbeSystemArray{TF} <: ProbeSystem{TF}
    position::Matrix{TF}           # 3 x n_bodies
    scalar_potential::Vector{TF}   # n_bodies
    gradient::Matrix{TF}           # 3 x n_bodies
    hessian::Array{TF,3}           # 3 x 3 x n_bodies
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
    extra_farfield::Bool
end

#--- memory cache ---#

"""
    Cache{TF,NT,NS}

Cache object used to store system buffers to avoid repeated allocations.

**Fields**

* `target_buffers::Vector{Matrix{TF}}`: vector of buffers for target systems
* `source_buffers::Vector{Matrix{TF}}`: vector of buffers for source systems
* `target_small_buffers::Vector{Matrix{TF}}`: vector of small buffers used for pidgeon-hole sorting target systems into the octree
* `source_small_buffers::Vector{Matrix{TF}}`: vector of small buffers used for pidgeon-hole sorting source systems into the octree

"""
struct Cache{TF}
    target_buffers::Vector{Matrix{TF}}
    source_buffers::Vector{Matrix{TF}}
    target_small_buffers::Vector{Matrix{TF}}
    source_small_buffers::Vector{Matrix{TF}}
end

function Cache(;
    target_buffers::Vector{Matrix{TF}},
    source_buffers::Vector{Matrix{TF}},
    target_small_buffers::Vector{Matrix{TF}},
    source_small_buffers::Vector{Matrix{TF}}
        ) where {TF}
    return Cache{TF}(target_buffers, source_buffers, target_small_buffers, source_small_buffers)
end

function Cache(target_systems::Tuple, source_systems::Tuple, switches::Tuple)
    # get float type
    TF = get_type(target_systems, source_systems)

    # allocate buffers
    target_buffers = allocate_buffers(target_systems, true, TF, switches)
    source_buffers = allocate_buffers(source_systems, false, TF, switches)
    target_small_buffers = allocate_small_buffers(target_systems, TF, switches; target=true)
    source_small_buffers = allocate_small_buffers(source_systems, TF, DerivativesSwitch(false, false, false, source_systems); target=false)
    
    # return cache
    return Cache{TF}(target_buffers, source_buffers, target_small_buffers, source_small_buffers)
end

#------- operator basis and cache types -------#

abstract type AbstractOperatorBasis end

struct CompressedComplexBasis <: AbstractOperatorBasis end

struct RealSolidHarmonicBasis <: AbstractOperatorBasis end

struct OperatorOrders{LH}
    P_phi::Int
    P_chi::Int
    P_active::Int
end

function _validate_operator_order(P::Integer)
    P < 0 && throw(ArgumentError("operator expansion order must be nonnegative"))
    return Int(P)
end

function OperatorOrders(P::Integer, ::Val{false})
    P_int = _validate_operator_order(P)
    return OperatorOrders{false}(P_int, P_int, P_int)
end

function OperatorOrders(P::Integer, ::Val{true})
    P_int = _validate_operator_order(P)
    return OperatorOrders{true}(P_int, P_int + 1, P_int + 1)
end

struct OperatorBasisInfo{B<:AbstractOperatorBasis,LH}
    basis::B
    orders::OperatorOrders{LH}
    channel_count::Int
    basis_dof_phi::Int
    basis_dof_chi::Int
    basis_dof_active::Int
end

_operator_ncomplex(P::Integer) = ((P + 1) * (P + 2)) >> 1
_compressed_complex_dof(P::Integer) = 2 * _operator_ncomplex(P)

function OperatorBasisInfo(basis::CompressedComplexBasis, orders::OperatorOrders{LH}) where LH
    channel_count = LH ? 2 : 1
    basis_dof_phi = _compressed_complex_dof(orders.P_phi)
    basis_dof_chi = _compressed_complex_dof(orders.P_active)
    basis_dof_active = _compressed_complex_dof(orders.P_active)
    return OperatorBasisInfo{CompressedComplexBasis,LH}(
        basis,
        orders,
        channel_count,
        basis_dof_phi,
        basis_dof_chi,
        basis_dof_active,
    )
end

OperatorBasisInfo(basis::CompressedComplexBasis, P::Integer, lamb_helmholtz::Val) =
    OperatorBasisInfo(basis, OperatorOrders(P, lamb_helmholtz))

OperatorBasisInfo(P::Integer, lamb_helmholtz::Val) =
    OperatorBasisInfo(CompressedComplexBasis(), P, lamb_helmholtz)

struct OperatorInvariantCache{TF,B<:AbstractOperatorBasis,LH}
    basis_info::OperatorBasisInfo{B,LH}
    Hs_pi2::Vector{TF}
    zeta_mag::Vector{TF}
    eta_mag::Vector{TF}
    M_tilde::Vector{TF}
    L_tilde::Vector{TF}
end

function OperatorInvariantCache(::Type{TF}, basis_info::OperatorBasisInfo{B,LH}) where {TF,B,LH}
    P_active = basis_info.orders.P_active

    Hs_pi2 = ones(TF, 1)
    zeta_mag = ones(TF, 1)
    eta_mag = ones(TF, 1)
    M_tilde = ones(TF, 1)
    L_tilde = ones(TF, 1)

    update_Hs_π2!(Hs_pi2, P_active)
    update_ζs_mag!(zeta_mag, P_active)
    update_ηs_mag!(eta_mag, P_active)
    update_M̃!(M_tilde, P_active)
    update_L̃!(L_tilde, P_active)

    return OperatorInvariantCache{TF,B,LH}(
        basis_info,
        Hs_pi2,
        zeta_mag,
        eta_mag,
        M_tilde,
        L_tilde,
    )
end

OperatorInvariantCache(::Type{TF}, P::Integer, lamb_helmholtz::Val) where TF =
    OperatorInvariantCache(TF, OperatorBasisInfo(P, lamb_helmholtz))

struct OperatorScratch{TF,B<:AbstractOperatorBasis,LH}
    basis_info::OperatorBasisInfo{B,LH}
    weights_tmp_1::Array{TF,3}
    weights_tmp_2::Array{TF,3}
    weights_tmp_3::Array{TF,3}
    Ts::Vector{TF}
    eimphis::Matrix{TF}
end

function OperatorScratch(::Type{TF}, basis_info::OperatorBasisInfo{B,LH}) where {TF,B,LH}
    P_active = basis_info.orders.P_active
    return OperatorScratch{TF,B,LH}(
        basis_info,
        initialize_expansion(P_active, TF),
        initialize_expansion(P_active, TF),
        initialize_expansion(P_active, TF),
        zeros(TF, length_Ts(P_active)),
        zeros(TF, 2, P_active + 1),
    )
end

OperatorScratch(::Type{TF}, P::Integer, lamb_helmholtz::Val) where TF =
    OperatorScratch(TF, OperatorBasisInfo(P, lamb_helmholtz))

struct ThreadedOperatorScratch{S}
    scratch::Vector{S}
end

function ThreadedOperatorScratch(::Type{TF}, basis_info::OperatorBasisInfo) where TF
    scratch = [OperatorScratch(TF, basis_info) for _ in 1:Threads.nthreads()]
    return ThreadedOperatorScratch{eltype(scratch)}(scratch)
end

ThreadedOperatorScratch(::Type{TF}, P::Integer, lamb_helmholtz::Val) where TF =
    ThreadedOperatorScratch(TF, OperatorBasisInfo(P, lamb_helmholtz))
