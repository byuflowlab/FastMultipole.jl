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

"""
    RadixGrid{TF}

Storage-minimal uniform-grid clustering for the radix-path matrix-operator
driver. The grid stores the root lower corner, root half-width, fixed depth,
Morton-sorted global body permutation, occupied Morton keys, compact ranges into
the sorted permutation, and mappings from each global body ordinal back to its
original system and local body index. The inverse permutation maps global body
ordinals back to their sorted slots. Cell coordinates and geometry are derived
by accessors in `tree_batched.jl`.
"""
struct RadixGrid{TF}
    x_min::SVector{3,TF}
    h0::TF
    ell::Int
    perm::Vector{Int}
    invperm::Vector{Int}
    cell_keys::Vector{UInt64}
    cell_ranges::Matrix{Int}
    body_system::Vector{Int}
    body_index::Vector{Int}
end

abstract type RadixSortBackend end

struct HostRadixSort <: RadixSortBackend end

struct DeviceRadixSort <: RadixSortBackend end

struct AutoRadixSort <: RadixSortBackend
    min_device_bodies::Int
    function AutoRadixSort(; min_device_bodies::Integer=MIN_BODIES)
        min_device_bodies >= 0 ||
            throw(ArgumentError("min_device_bodies must be nonnegative"))
        return new(Int(min_device_bodies))
    end
end

"""
    DeviceRadixGrid{TF,VI,VK,MI,MC}

CUDA-resident radix-grid metadata. The array type parameters are supplied by the
opt-in CUDA implementation and are not named here, keeping the CPU load path free
of CUDA symbols. Nodes are stored level-major and Morton-key-major within each
level. `child_ranges[:, i]` is a first/count range over this node ordering.

Mutable (task 023) so the recurring update path can refresh `n_bodies`/`n_cells`
in place while every array field keeps its identity across time steps; kernels
and loops receive the arrays (never the struct), so mutability costs nothing.
"""
mutable struct DeviceRadixGrid{TF,VI,VK,MI,MC}
    x_min::SVector{3,TF}
    h0::TF
    ell::Int
    n_bodies::Int
    n_cells::Int
    perm::VI
    invperm::VI
    cell_keys::VK
    cell_ranges::MI
    body_system::VI
    body_index::VI
    cell_centers::MC
    node_levels::VI
    node_keys::VK
    node_coords::MI
    node_centers::MC
    parent_index::VI
    child_ranges::MI
    leaf_to_node::VI
end

struct ConstantPStencilConfig{TF,LH,N}
    P_phi::Int
    epsilon::TF
    source_strength::TF
    chi_strength::TF
end

function ConstantPStencilConfig(P_phi::Integer, epsilon, source_strength=one(epsilon);
        chi_strength=source_strength, lamb_helmholtz::Bool=false,
        normalization::Symbol=:analytic)
    P_phi < 0 && throw(ArgumentError("P_phi must be nonnegative"))
    epsilon > zero(epsilon) || throw(ArgumentError("epsilon must be positive"))
    source_strength >= zero(source_strength) || throw(ArgumentError("source_strength must be nonnegative"))
    chi_strength >= zero(chi_strength) || throw(ArgumentError("chi_strength must be nonnegative"))
    normalization in (:analytic, :production) ||
        throw(ArgumentError("normalization must be :analytic or :production"))
    TF = promote_type(typeof(epsilon), typeof(source_strength), typeof(chi_strength))
    return ConstantPStencilConfig{TF,lamb_helmholtz,normalization}(
        Int(P_phi), TF(epsilon), TF(source_strength), TF(chi_strength),
    )
end

abstract type RadixSeparationPolicy end

const _SUPPORTED_RIGID_NEAR_RADII2 = (3, 4, 5, 6, 8, 9, 10, 11, 12)
const _SUPPORTED_RIGID_NEAR_RADII2_TEXT = "3, 4, 5, 6, 8, 9, 10, 11, 12"

# Shipped rigid-stencil operating point, selected by measurement in task 028
# Stage 7: `q = 5` at every M2L level except the coarsest, which uses `q = 6`.
# At n = 1e6 / P = 4 / ell = 5 this measured 12.51 ms per resident step at
# 1.05e-3 gradient relative RMS, against 15.45 ms at 5.75e-4 for uniform q = 6
# and ~30 ms at 3.19e-4 for the previous uniform q = 12 default.
const RADIX_DEFAULT_NEAR_RADIUS2 = 5
const RADIX_DEFAULT_COARSE_NEAR_RADIUS2 = 6

@inline function _validate_rigid_near_radius2(q::Integer, owner::AbstractString)
    q in _SUPPORTED_RIGID_NEAR_RADII2 && return Int(q)
    throw(ArgumentError("$owner near_radius2=$(Int(q)) is unsupported; " *
        "supported values are ($_SUPPORTED_RIGID_NEAR_RADII2_TEXT)"))
end

# A level schedule lists one near radius per M2L level (levels 2:ell, coarse to
# fine). Task 025's exact-once proof extends to a level-dependent radius only
# while the radius is non-increasing with depth, and the leaf entry is what
# defines the direct list, so it must agree with `near_radius2`.
function _validate_rigid_level_schedule(level_radii2, near_radius2::Int)
    qs = Tuple(Int(q) for q in level_radii2)
    isempty(qs) && return ()
    foreach(q -> _validate_rigid_near_radius2(q, "hierarchical level schedule"), qs)
    all(qs[i + 1] <= qs[i] for i in 1:length(qs)-1) || throw(ArgumentError(
        "hierarchical level schedule must be non-increasing with depth; got $qs"))
    last(qs) == near_radius2 || throw(ArgumentError(
        "hierarchical level schedule leaf radius $(last(qs)) must equal " *
        "near_radius2=$near_radius2"))
    return qs
end

struct ParentNeighborM2L <: RadixSeparationPolicy end

"""
    ConstantPAnalyticStencil(config)

Flat, leaf-only constant-`P` M2L policy: one analytic acceptance test over the
bounded offset box, applied at the leaf level only.

!!! warning "Deprecated as the production default (task 027)"
    [`HierarchicalRigidStencil`](@ref) replaced this as the default policy. The
    flat classifier accepts *more* offsets as `ell` grows (cell width shrinks at
    fixed `epsilon`), so its route count scales as `offsets(ell) x cells`: on an
    H200 at `n = 2e5, ell = 5` it needs 426,826,100 routes and 67.2 GB and takes
    1172 ms, against 3,895,658 routes, 805 MB, and 3.64 ms for the rigid
    hierarchical stencil, whose offset set is level-invariant by construction.

    This policy remains fully supported and selectable via the `policy` keyword.
    It is retained deliberately as the independent correctness oracle for the
    radix path and as a low-occupancy fallback; it is not scheduled for removal.
"""
struct ConstantPAnalyticStencil{C<:ConstantPStencilConfig} <: RadixSeparationPolicy
    config::C
end

ConstantPAnalyticStencil(args...; kwargs...) =
    ConstantPAnalyticStencil(ConstantPStencilConfig(args...; kwargs...))

"""
    HierarchicalRigidStencil(config; near_radius2=5, level_radii2=(),
        window_classes=4, dense_occupancy_max_bytes=256 << 20,
        dense_occupancy_max_ell=8)

Host-resident, genuinely hierarchical rigid M2L policy.  The analytic
`config` is retained as an accuracy contract: cache construction verifies that
its rejected integer offsets are exactly the requested spherical near set.
Since task 027 this is the default `RadixFMMCache` policy; the flat
[`ConstantPAnalyticStencil`](@ref) remains selectable as the correctness oracle.

`near_radius2` is the squared lattice near radius `q` at the leaf level: cell
offsets with `|o|^2 <= q` are evaluated directly and everything beyond is M2L.
Supported values are $(_SUPPORTED_RIGID_NEAR_RADII2_TEXT) (the omitted 7 has no
integer lattice shell). Larger `q` means more direct work and a more accurate
far field; `q = 3` is the classic `|o|_inf <= 1` FMM stencil and `q = 12` the
`theta = 0.5` stencil.

`level_radii2` optionally schedules one radius per M2L level, coarse to fine,
for levels `2:ell` — task 028 Stage 7. It must be non-increasing with depth
(the condition under which task 025's exact-once coverage proof still holds)
and its last entry must equal `near_radius2`. An empty tuple means the uniform
policy. The default `RadixFMMCache` policy is `near_radius2 = 5` with the
schedule `(6, 5, 5, ..., 5)`, the fastest configuration that stayed inside task
028's accuracy gate at `P = 4` (1.05e-3 gradient relative RMS at `n = 1e6`, vs
3.19e-4 for the previous uniform `q = 12` default). Pass `near_radius2 = 12`
for the older, more accurate and slower operating point.
"""
struct HierarchicalRigidStencil{C<:ConstantPStencilConfig} <: RadixSeparationPolicy
    config::C
    near_radius2::Int
    # One near radius per M2L level (levels 2:ell, coarse to fine); empty means
    # the uniform policy. Validated by `_validate_rigid_level_schedule`.
    level_radii2::Tuple{Vararg{Int}}
    window_classes::Int
    dense_occupancy_max_bytes::Int
    dense_occupancy_max_ell::Int
    function HierarchicalRigidStencil(config::C;
            near_radius2::Integer=RADIX_DEFAULT_NEAR_RADIUS2,
            level_radii2=(),
            window_classes::Integer=4,
            dense_occupancy_max_bytes::Integer=256 << 20,
            dense_occupancy_max_ell::Integer=8) where {C<:ConstantPStencilConfig}
        q = _validate_rigid_near_radius2(near_radius2,
            "HierarchicalRigidStencil")
        qs = _validate_rigid_level_schedule(level_radii2, q)
        window_classes > 0 ||
            throw(ArgumentError("HierarchicalRigidStencil window_classes must be positive"))
        dense_occupancy_max_bytes >= 0 ||
            throw(ArgumentError("dense_occupancy_max_bytes must be nonnegative"))
        dense_occupancy_max_ell >= 0 ||
            throw(ArgumentError("dense_occupancy_max_ell must be nonnegative"))
        return new{C}(config, q, qs, Int(window_classes),
            Int(dense_occupancy_max_bytes), Int(dense_occupancy_max_ell))
    end
end

"""
Return `policy` with its per-level radius schedule replaced. Equivalent to
constructing the policy with the `level_radii2` keyword; retained because the
schedule is often chosen after the base policy (benchmarks, sweeps).
"""
function _hierarchical_stencil_with_schedule(policy::HierarchicalRigidStencil,
        level_radii2)
    isempty(level_radii2) &&
        throw(ArgumentError("hierarchical level schedule must be nonempty"))
    return HierarchicalRigidStencil(policy.config;
        policy.near_radius2, level_radii2, policy.window_classes,
        policy.dense_occupancy_max_bytes, policy.dense_occupancy_max_ell)
end

function HierarchicalRigidStencil(P_phi::Integer, epsilon,
        source_strength=one(epsilon); chi_strength=source_strength,
        lamb_helmholtz::Bool=false, normalization::Symbol=:analytic,
        near_radius2::Integer=RADIX_DEFAULT_NEAR_RADIUS2, level_radii2=(),
        window_classes::Integer=4,
        dense_occupancy_max_bytes::Integer=256 << 20,
        dense_occupancy_max_ell::Integer=8)
    config = ConstantPStencilConfig(P_phi, epsilon, source_strength;
        chi_strength, lamb_helmholtz, normalization)
    return HierarchicalRigidStencil(config; near_radius2, level_radii2,
        window_classes, dense_occupancy_max_bytes, dense_occupancy_max_ell)
end

classic_fmm_stencil(config::ConstantPStencilConfig; kwargs...) =
    HierarchicalRigidStencil(config; near_radius2=3, kwargs...)
classic_fmm_stencil(args...; kwargs...) =
    HierarchicalRigidStencil(args...; near_radius2=3, kwargs...)

"""
Immutable geometry of the phase-indexed source-major rigid stencil.
`phase_starts`/`phase_index` are a one-based CSR over the eight source phases;
`class_of[p,k] == 0` means union offset `k` is absent from phase `p`.
"""
struct RigidHierarchicalTables
    near_offsets::Vector{SVector{3,Int}}
    push_offsets::Vector{SVector{3,Int}}
    phase_starts::NTuple{9,Int}
    phase_index::Vector{Int32}
    class_of::Matrix{Int32}
end

"""
Per-level occupied-node lookup.  A nonempty `node_at` is the dense lookup;
an empty vector selects the sorted Morton-key fallback.
"""
struct RadixLevelOccupancy{A<:AbstractVector{Int32}}
    ell::Int
    level_base::Vector{Int}
    node_at::A
end

mutable struct HostHierarchicalM2LContext{O<:RadixLevelOccupancy,A}
    tables::RigidHierarchicalTables
    level_class_of::Array{Int32,3}
    occupancy::O
    class_level::Vector{Int32}
    class_offset::Matrix{Int32}
    effective_offsets::Vector{SVector{3,Int}}
    apply_plan::A
    window_classes::Int
    level_offsets::Vector{Int}
    total_routes::Int
    routes_per_level::Vector{Int}
    last_window_routes::Int
    profile_stages::Bool
    update_stage_ns::Vector{UInt64}
    m2l_level_ns::Vector{UInt64}
end

"""
Device mirror of [`HostHierarchicalM2LContext`](@ref) (task 027).  It owns the
step-invariant task-025 stencil tables uploaded once at construction, the
persistent per-level occupancy lookup, the single-window flag/scan/compact
buffers, and the dense strategy's per-level `Lambda` scaling columns.  Only the
`(level, offset)` window currently being generated is materialized: the complete
hierarchical pair list is never compiled.

Array fields are parameterized rather than named so this container stays free of
CUDA types and the CPU-only package import never touches a device runtime; the
CUDA implementation fills them with `CuArray`s.  `source_scale`/`target_scale`
are `D x (ell - 1)` and nonempty only for the dense strategy — the concatenated,
factored, and precomputed-y plans carry level-true `(level, offset)` tables
through `effective_offsets` and must never be `Lambda`-scaled.
"""
mutable struct DeviceHierarchicalM2LContext{PL,IV32,IM32,IA32,IV,SM}
    tables::RigidHierarchicalTables
    level_radii2::Vector{Int}
    class_level::Vector{Int32}
    class_offset::Matrix{Int32}
    effective_offsets::Vector{SVector{3,Int}}
    apply_plan::PL
    window_classes::Int
    ell::Int
    noffsets::Int
    # host geometry mirrors (small, step-varying prefixes)
    level_base::Vector{Int}
    level_offsets::Vector{Int}
    # persistent device storage
    node_at::IV32
    d_level_base::IV
    d_push_offsets::IM32
    d_class_of::IA32
    d_near_offsets::IM32
    symmetric_targets::IV
    symmetric_sources::IV
    route_flags::IV32
    route_prefix::IV32
    window_cum::IV32
    host_window_cum::Vector{Int32}
    source_scale::SM
    target_scale::SM
    # allocation-free telemetry
    total_routes::Int
    routes_per_level::Vector{Int}
    nodes_per_level::Vector{Int}
    last_window_routes::Int
    n_symmetric_pairs::Int
    window_lo::Int
    window_hi::Int
    profile_stages::Bool
    update_stage_ns::Vector{UInt64}
    m2l_level_ns::Vector{UInt64}
end

abstract type RadixTraversalStrategy end

struct RigidImplicitStencil <: RadixTraversalStrategy end

struct SparseOffsetIntersection <: RadixTraversalStrategy end

struct BlockedOccupancyBitsets <: RadixTraversalStrategy
    brick_side::Int
    function BlockedOccupancyBitsets(brick_side::Integer=4)
        brick_side > 0 || throw(ArgumentError("brick_side must be positive"))
        count_ones(brick_side) == 1 ||
            throw(ArgumentError("brick_side must be a power of two"))
        brick_side^3 <= 8 * sizeof(UInt128) ||
            throw(ArgumentError("brick_side is too large for the UInt128 local occupancy mask"))
        return new(Int(brick_side))
    end
end

struct LazyMaterializedBatches{S<:RadixTraversalStrategy} <: RadixTraversalStrategy
    materialization_threshold::Int
    fallback::S
    function LazyMaterializedBatches(materialization_threshold::Integer=32,
            fallback::S=SparseOffsetIntersection()) where {S<:RadixTraversalStrategy}
        materialization_threshold >= 0 ||
            throw(ArgumentError("materialization_threshold must be nonnegative"))
        return new{S}(Int(materialization_threshold), fallback)
    end
end

struct RadixM2LBatch{TI}
    level::Int
    offset::SVector{3,Int}
    targets::Vector{TI}
    sources::Vector{TI}
end

RadixM2LBatch(offset::SVector{3,Int}, targets::Vector{TI}, sources::Vector{TI}) where {TI} =
    RadixM2LBatch{TI}(0, offset, targets, sources)

RadixM2LBatch(level::Integer, offset::SVector{3,Int}, targets::Vector{TI}, sources::Vector{TI}) where {TI} =
    RadixM2LBatch{TI}(Int(level), offset, targets, sources)

struct RadixInteractionList{TI}
    m2l_batches::Vector{RadixM2LBatch{TI}}
    direct_pairs::Vector{SVector{2,TI}}
end

# Implicit constant-P interaction structure: the translation-invariant stencil is
# fully determined by the fixed accepted/rejected offset sets plus a dense
# coord -> occupied-cell lookup, so no per-pair enumeration is needed to describe the
# interaction lists. `accepted_offsets` is sorted by (z, y, x) to match the
# RadixInteractionList batch order; `rejected_offsets` is the bounded near/self
# complement (every offset between occupied cells lies in the (2G-1)^3 box);
# `cell_at[coord + 1] = cell index` with 0 marking unoccupied coordinates.
struct RadixImplicitStencil
    accepted_offsets::Vector{SVector{3,Int}}
    rejected_offsets::Vector{SVector{3,Int}}
    cell_at::Array{Int32,3}
end

## Constant-P analytic radix separation is deferred until it can be represented by a
## compact procedural policy. Do not use the old materialized transition-offset
## front door in production traversal.
# struct RadixM2LOffset
#     level::Int
#     offset::SVector{3,Int}
#     target_octant::SVector{3,Int}
#     source_octant::SVector{3,Int}
# end
#
# struct RadixInteractionStencil
#     m2l_offsets::Vector{RadixM2LOffset}
#     direct_offsets::Vector{SVector{3,Int}}
# end

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
_real_solid_harmonic_dof(P::Integer) = (P + 1) * (P + 1)

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

function OperatorBasisInfo(basis::RealSolidHarmonicBasis, orders::OperatorOrders{LH}) where LH
    channel_count = LH ? 2 : 1
    basis_dof_phi = _real_solid_harmonic_dof(orders.P_phi)
    basis_dof_chi = _real_solid_harmonic_dof(orders.P_active)
    basis_dof_active = _real_solid_harmonic_dof(orders.P_active)
    return OperatorBasisInfo{RealSolidHarmonicBasis,LH}(
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

OperatorBasisInfo(basis::RealSolidHarmonicBasis, P::Integer, lamb_helmholtz::Val) =
    OperatorBasisInfo(basis, OperatorOrders(P, lamb_helmholtz))

OperatorBasisInfo(P::Integer, lamb_helmholtz::Val) =
    OperatorBasisInfo(CompressedComplexBasis(), P, lamb_helmholtz)

#------- native flat coefficient buffers (Matrix Operator Refactor, task 017) -------#
#
# The native coefficient storage for the batched operators. re/im are interleaved
# into the leading basis dimension and each channel is a dense `basis_dof x batch`
# matrix (the BLAS/cuBLAS GEMM slab); see theory/coefficient-buffer-layout.md.
#
#     flat_basis_index(n, m, reim) = 2 * (harmonic_index(n, m) - 1) + reim,  reim in 1:2
#
# Default backing is RAGGED (user-directed 2026-06-25): a separate dense φ matrix
# sized to `basis_dof_phi` (order `P_phi`) and a separate dense χ matrix sized to
# `basis_dof_chi` (order `P_active = P_chi`). φ therefore carries no padding rows,
# so the task-014/016 `_zero_phi_padding!` machinery is unnecessary here; the
# physical φ/χ bounds are instead enforced structurally by the order-aware operator
# kernels (`P_phi` for φ, `P_active` for χ). `Val(false)` allocates φ only (the dead
# χ channel is pruned). `harmonic_index(n,m)` is P-independent, so the same
# `flat_basis_index` addresses both matrices (φ valid for n <= P_phi, χ for n <=
# P_active). RAGGED is the DECIDED layout (task 019b, user decision 2026-07-14):
# the padded single-array alternative measured +10-28% chain time on CPU with no
# channel-merged-GEMM win (the GPU M2L is no longer launch-bound after 019's
# fusion) and +17-39% storage at the small P the GPU path runs, so it was
# rejected; the accessors below remain the swap surface should that change.

@inline flat_basis_index(n, m, reim) = 2 * (harmonic_index(n, m) - 1) + reim
@inline real_basis_index(n, m) = (m == 0) ? n * n + 1 : throw(ArgumentError("two-argument real_basis_index is only valid for m == 0"))
@inline real_basis_index(n, m, ::Val{:cos}) = n * n + 2m
@inline real_basis_index(n, m, ::Val{:sin}) = n * n + 2m + 1

@inline _basis_index(::CompressedComplexBasis, n, m, reim) = flat_basis_index(n, m, reim)
@inline _basis_index(::RealSolidHarmonicBasis, n, m, ::Val{:zero}) = real_basis_index(n, m)
@inline _basis_index(::RealSolidHarmonicBasis, n, m, ::Val{:cos}) = real_basis_index(n, m, Val(:cos))
@inline _basis_index(::RealSolidHarmonicBasis, n, m, ::Val{:sin}) = real_basis_index(n, m, Val(:sin))

"""
    AbstractCoefficientBuffer{TF,LH}

Supertype for resident expansion-coefficient buffer layouts (task 022). Concrete
layouts are the legacy [`FlatCoefficientBuffer`](@ref) (compressed `m>=0`, interleaved
re/im, used by the per-column operators) and the GEMM-native
[`DegreeMajorRealBuffer`](@ref) (real, degree-major, y-mode-ordered) that the batched
`mul!` operator strategies read with no per-GEMM reformatting. The buffer layout is a
benchmark knob independent of the operator strategy.
"""
abstract type AbstractCoefficientBuffer{TF,LH} end

"""
    FlatCoefficientBuffer{TF,A,B,LH}

Native flat coefficient storage (task 017). Holds a dense φ channel matrix
(`basis_dof_phi x batch`) and, for `Val(true)`, a dense χ channel matrix
(`basis_dof_chi x batch`); for `Val(false)` `chi` is empty (χ pruned). Parametric on
the matrix type `A` so a device array (e.g. `CuArray`) can back it later (task 022).

Operators touch the channels only through the accessors [`phi_slab`](@ref) /
[`chi_slab`](@ref) / [`phi_physical_view`](@ref), so the physical backing
(ragged, decided by task 019b; the padded single-array alternative was measured
and rejected) is swappable.
"""
struct FlatCoefficientBuffer{TF,A<:AbstractMatrix{TF},B<:AbstractOperatorBasis,LH} <: AbstractCoefficientBuffer{TF,LH}
    phi::A
    chi::A
    basis_info::OperatorBasisInfo{B,LH}
end

function FlatCoefficientBuffer(::Type{TF}, basis_info::OperatorBasisInfo{B,LH}, batch::Integer) where {TF,B,LH}
    phi = zeros(TF, basis_info.basis_dof_phi, batch)
    chi = LH ? zeros(TF, basis_info.basis_dof_chi, batch) : zeros(TF, 0, 0)
    return FlatCoefficientBuffer{TF,Matrix{TF},B,LH}(phi, chi, basis_info)
end

FlatCoefficientBuffer(::Type{TF}, P::Integer, lamb_helmholtz::Val, batch::Integer) where TF =
    FlatCoefficientBuffer(TF, OperatorBasisInfo(P, lamb_helmholtz), batch)

# Non-allocating channel accessors. `phi_slab`/`chi_slab` are the dense GEMM slabs;
# `phi_physical_view` is the φ output through `P_phi` (the whole φ matrix in the
# ragged backing, a non-padding sub-view in the deferred padded backing).
@inline phi_slab(buf::FlatCoefficientBuffer) = buf.phi
@inline chi_slab(buf::FlatCoefficientBuffer) = buf.chi
@inline phi_physical_view(buf::FlatCoefficientBuffer) =
    @view buf.phi[1:buf.basis_info.basis_dof_phi, :]
@inline flat_nbatch(buf::FlatCoefficientBuffer) = size(buf.phi, 2)

#------- GEMM-native degree-major coefficient buffer (Matrix Operator Refactor, task 022) -------#
#
# The batched `mul!` M2M/M2L/L2L operator strategies work over the genuinely factored
# y-rotation `Y_n(θ) = U_n diag(e^{iνθ}) V_n`, whose fixed modes `U_n`/`V_n` act on the
# `2n+1` real degree-`n` dofs in y-mode order (`_ymode_dof_to_storage`: k=1 -> (re,m0);
# k=2m -> (re,m); k=2m+1 -> (im,m); `im(m0)` is structurally 0 for real-source
# multipoles). Storing coefficients degree-major in that order means each degree block
# `view(phi, degree_row_range(n), :)` is already a `(2n+1) x batch` GEMM operand — no
# per-GEMM repack. Total rows per channel at order `P` is `(P+1)^2`.
@inline degree_major_dof(P::Integer) = (P + 1) * (P + 1)
@inline degree_row_offset(n::Integer) = n * n                    # Σ_{k=0}^{n-1}(2k+1)
@inline degree_row_range(n::Integer) = (n * n + 1):((n + 1) * (n + 1))

"""
    DegreeMajorRealBuffer{TF,A,B,LH}

GEMM-native coefficient storage (task 022): a real, degree-major, y-mode-ordered φ
matrix (`(P_phi+1)^2 x batch`) and, for `Val(true)`, a χ matrix
(`(P_active+1)^2 x batch`); `chi` is empty for `Val(false)`. Parametric on the matrix
type `A` so a device array (`CuArray`) can back it. Read the per-degree GEMM operands
through `degree_row_range(n)`; use [`to_gemm_buffer!`](@ref) / [`to_flat_buffer!`](@ref)
only at pass boundaries / tests, never in the hot path.
"""
struct DegreeMajorRealBuffer{TF,A<:AbstractMatrix{TF},B<:AbstractOperatorBasis,LH} <: AbstractCoefficientBuffer{TF,LH}
    phi::A
    chi::A
    basis_info::OperatorBasisInfo{B,LH}
end

function DegreeMajorRealBuffer(::Type{TF}, basis_info::OperatorBasisInfo{B,LH}, batch::Integer) where {TF,B,LH}
    phi = zeros(TF, degree_major_dof(basis_info.orders.P_phi), batch)
    chi = LH ? zeros(TF, degree_major_dof(basis_info.orders.P_active), batch) : zeros(TF, 0, 0)
    return DegreeMajorRealBuffer{TF,Matrix{TF},B,LH}(phi, chi, basis_info)
end

@inline phi_slab(buf::DegreeMajorRealBuffer) = buf.phi
@inline chi_slab(buf::DegreeMajorRealBuffer) = buf.chi
@inline flat_nbatch(buf::DegreeMajorRealBuffer) = size(buf.phi, 2)

# Boundary/test converters (host, element-wise — not for the GEMM hot path).
function _degree_major_from_flat_channel!(dst, src, P::Integer)
    @inbounds for n in 0:P
        base = degree_row_offset(n)
        for k in 1:(2n + 1)
            ri, m = _ymode_dof_to_storage(k)
            fr = flat_basis_index(n, m, ri)
            for col in axes(dst, 2)
                dst[base + k, col] = src[fr, col]
            end
        end
    end
    return dst
end

function _flat_from_degree_major_channel!(dst, src, P::Integer)
    dst .= zero(eltype(dst))                       # leaves im(m0) rows at 0
    @inbounds for n in 0:P
        base = degree_row_offset(n)
        for k in 1:(2n + 1)
            ri, m = _ymode_dof_to_storage(k)
            fr = flat_basis_index(n, m, ri)
            for col in axes(dst, 2)
                dst[fr, col] = src[base + k, col]
            end
        end
    end
    return dst
end

"""
    to_gemm_buffer!(gemm::DegreeMajorRealBuffer, flat::FlatCoefficientBuffer)

Copy a legacy compressed flat buffer into the degree-major GEMM layout. `im(m0)` is
dropped (structurally 0). Boundary/test use only.
"""
function to_gemm_buffer!(gemm::DegreeMajorRealBuffer{TF,A,B,LH}, flat::FlatCoefficientBuffer) where {TF,A,B,LH}
    flat_nbatch(gemm) == flat_nbatch(flat) ||
        throw(ArgumentError("source and target batch widths must match"))
    _degree_major_from_flat_channel!(phi_slab(gemm), phi_slab(flat), gemm.basis_info.orders.P_phi)
    LH && _degree_major_from_flat_channel!(chi_slab(gemm), chi_slab(flat), gemm.basis_info.orders.P_active)
    return gemm
end

"""
    to_flat_buffer!(flat::FlatCoefficientBuffer, gemm::DegreeMajorRealBuffer)

Copy the degree-major GEMM layout back into a legacy compressed flat buffer, setting
`im(m0)` to 0. Boundary/test use only.
"""
function to_flat_buffer!(flat::FlatCoefficientBuffer{TF,A,B,LH}, gemm::DegreeMajorRealBuffer) where {TF,A,B,LH}
    flat_nbatch(flat) == flat_nbatch(gemm) ||
        throw(ArgumentError("source and target batch widths must match"))
    _flat_from_degree_major_channel!(phi_slab(flat), phi_slab(gemm), flat.basis_info.orders.P_phi)
    LH && _flat_from_degree_major_channel!(chi_slab(flat), chi_slab(gemm), flat.basis_info.orders.P_active)
    return flat
end

function complex_to_real_basis!(out::FlatCoefficientBuffer{TF,A,RealSolidHarmonicBasis,LH},
                                source::FlatCoefficientBuffer{TF2,A2,CompressedComplexBasis,LH}) where {TF,A,LH,TF2,A2}
    out.basis_info.orders == source.basis_info.orders ||
        throw(ArgumentError("source and target operator orders must match"))
    flat_nbatch(out) == flat_nbatch(source) ||
        throw(ArgumentError("source and target batch widths must match"))
    _complex_to_real_channel!(phi_slab(out), phi_slab(source), out.basis_info.orders.P_phi)
    LH && _complex_to_real_channel!(chi_slab(out), chi_slab(source), out.basis_info.orders.P_active)
    return out
end

function real_to_complex_basis!(out::FlatCoefficientBuffer{TF,A,CompressedComplexBasis,LH},
                                source::FlatCoefficientBuffer{TF2,A2,RealSolidHarmonicBasis,LH}) where {TF,A,LH,TF2,A2}
    out.basis_info.orders == source.basis_info.orders ||
        throw(ArgumentError("source and target operator orders must match"))
    flat_nbatch(out) == flat_nbatch(source) ||
        throw(ArgumentError("source and target batch widths must match"))
    _real_to_complex_channel!(phi_slab(out), phi_slab(source), source.basis_info.orders.P_phi)
    LH && _real_to_complex_channel!(chi_slab(out), chi_slab(source), source.basis_info.orders.P_active)
    return out
end

function _complex_to_real_channel!(out, source, P)
    fill!(out, zero(eltype(out)))
    @inbounds for j in axes(out, 2), n in 0:P
        fc = flat_basis_index(n, 0, 1)
        out[real_basis_index(n, 0), j] = source[fc, j]
        for m in 1:n
            fc = flat_basis_index(n, m, 1)
            out[real_basis_index(n, m, Val(:cos)), j] = source[fc, j]
            out[real_basis_index(n, m, Val(:sin)), j] = source[fc + 1, j]
        end
    end
    return out
end

function _real_to_complex_channel!(out, source, P)
    fill!(out, zero(eltype(out)))
    @inbounds for j in axes(out, 2), n in 0:P
        fc = flat_basis_index(n, 0, 1)
        out[fc, j] = source[real_basis_index(n, 0), j]
        out[fc + 1, j] = zero(eltype(out))
        for m in 1:n
            fc = flat_basis_index(n, m, 1)
            out[fc, j] = source[real_basis_index(n, m, Val(:cos)), j]
            out[fc + 1, j] = source[real_basis_index(n, m, Val(:sin)), j]
        end
    end
    return out
end

struct OperatorInvariantCache{TF,B<:AbstractOperatorBasis,LH}
    basis_info::OperatorBasisInfo{B,LH}
    Hs_pi2::Vector{TF}
    zeta_mag::Vector{TF}
    eta_mag::Vector{TF}
    M_tilde::Vector{TF}
    L_tilde::Vector{TF}
    # angle-independent axis-swap blocks (Matrix Operator Refactor, task 013):
    # precomputed from Hs_pi2 so the per-call Wigner Ts is a cheap phase contraction
    # via build_Ts_from_S! instead of the per-call update_Ts! rebuild.
    S_pos::Vector{TF}
    S_neg::Vector{TF}
    # fixed y-swap matrices for the explicit factored rotation path (task 013b).
    T_y_pos90::Vector{TF}
    T_y_neg90::Vector{TF}
    # fixed per-degree mode matrices for the genuinely factored y-rotation (task 013c):
    # Y_n(θ) = U_n diag(e^{iνθ}) V_n, with U/V angle-independent and batch-shared. The
    # multipole (ζ) and local (η) paths carry their own modes (dressing baked in).
    y_mult_U::Vector{Complex{TF}}
    y_mult_V::Vector{Complex{TF}}
    y_loc_U::Vector{Complex{TF}}
    y_loc_V::Vector{Complex{TF}}
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

    # axis-swap blocks depend on the just-populated Hs_pi2 (no module globals)
    S_pos = zeros(TF, length_Ss(P_active))
    S_neg = zeros(TF, length_Ss(P_active))
    update_S_blocks!(S_pos, S_neg, Hs_pi2, P_active)
    T_y_pos90 = zeros(TF, length_Ts(P_active))
    T_y_neg90 = zeros(TF, length_Ts(P_active))
    y_trig = Vector{TF}(undef, 2 * max(P_active, 1))
    build_Ts_from_S!(T_y_pos90, S_pos, S_neg, TF(pi / 2), P_active, y_trig)
    build_Ts_from_S!(T_y_neg90, S_pos, S_neg, TF(-pi / 2), P_active, y_trig)

    # fixed per-degree factored y-rotation modes (task 013c), one set per path
    nmodes = length_ymodes(P_active)
    y_mult_U = Vector{Complex{TF}}(undef, nmodes)
    y_mult_V = Vector{Complex{TF}}(undef, nmodes)
    y_loc_U = Vector{Complex{TF}}(undef, nmodes)
    y_loc_V = Vector{Complex{TF}}(undef, nmodes)
    # modes are per-channel (channel-independent), so sample the kernels single-channel
    update_factored_y_modes!(y_mult_U, y_mult_V, Hs_pi2, zeta_mag, P_active, Val(false), Val(false))
    update_factored_y_modes!(y_loc_U, y_loc_V, Hs_pi2, eta_mag, P_active, Val(false), Val(true))

    return OperatorInvariantCache{TF,B,LH}(
        basis_info,
        Hs_pi2,
        zeta_mag,
        eta_mag,
        M_tilde,
        L_tilde,
        S_pos,
        S_neg,
        T_y_pos90,
        T_y_neg90,
        y_mult_U,
        y_mult_V,
        y_loc_U,
        y_loc_V,
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
    y_trig::Vector{TF}
    z_cos::Vector{TF}
    z_sin::Vector{TF}
    eimphis::Matrix{TF}
    # ν-space scratch for the factored y stage (task 013c); length >= 2*P_active+1
    y_mode_buf::Vector{Complex{TF}}
end

function OperatorScratch(::Type{TF}, basis_info::OperatorBasisInfo{B,LH}) where {TF,B,LH}
    P_active = basis_info.orders.P_active
    ndof = ((P_active + 1) * (P_active + 2)) >> 1
    return OperatorScratch{TF,B,LH}(
        basis_info,
        initialize_expansion(P_active, TF),
        initialize_expansion(P_active, TF),
        initialize_expansion(P_active, TF),
        zeros(TF, length_Ts(P_active)),
        Vector{TF}(undef, 2 * max(P_active, 1)),
        zeros(TF, ndof),
        zeros(TF, ndof),
        zeros(TF, 2, P_active + 1),
        Vector{Complex{TF}}(undef, 2 * P_active + 1),
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

#------- FULL M2L OPERATOR PIPELINE (Matrix Operator Refactor, task 014) -------#
#
# Whole-M2L operator tags selecting the y-rotation strategy. Both compose the same
# shared stages (task 010 z-rotation, task 011 fixed-m z-translation blocks, task
# 012 Lamb-Helmholtz coupling); they differ only in how the arbitrary-angle
# y-alignment is realized:
#
#   MaterializedYRotationM2L : reconstruct Ts(θ) per column from the cached S_pos /
#                              S_neg axis-swap blocks (task 013) and apply the
#                              production-parity y kernels.
#   FactoredRotationM2L      : apply the genuinely factored Z_phi -> Y(θ) -> ... ->
#                              inverse Z_phi using the fixed per-degree mode
#                              matrices U_n / V_n (task 013c). Plain-H: the modes
#                              are y_mult_U/V and y_loc_U/V, NOT the 013b T_y_*90
#                              primitives.
#
# These are zero-field tag types; all invariant data lives on OperatorInvariantCache
# and all working storage on M2LOperatorScratch.

abstract type AbstractM2LOperator end

struct MaterializedYRotationM2L <: AbstractM2LOperator end

# Physical-subspace invariant (016b): the FactoredRotation* operators reproduce
# production exactly only for *physical* inputs (m=0 imaginary part == 0). Every
# real solid-harmonic expansion is physical, so this holds throughout production;
# the MaterializedYRotation* operators stay exact for any input. See the
# `_assert_factored_input_physical` guard in src/rotate_batched.jl.
struct FactoredRotationM2L <: AbstractM2LOperator end

"""
    M2LOperatorScratch{TF,B,LH}

Working storage for the batched full-M2L operator pipeline (task 014), sized for a
maximum batch width `B_max` and the active order `P_active`.

Minimal footprint: exactly two native flat working buffers (`work_a`, `work_b`,
each a [`FlatCoefficientBuffer`](@ref) with a `basis_dof_phi x B_max` φ matrix and,
for `Val(true)`, a `basis_dof_chi x B_max` χ matrix) that are lifetime-aliased
across the three pipeline stages, plus an embedded [`OperatorScratch`](@ref) that
already provides every 1D per-column buffer (`Ts`, `y_trig`, `z_cos`, `z_sin`, and
the factored ν-space `y_mode_buf`) and the `weights_tmp_*` `[2,2,nh]` scratch used
by the materialized-y per-column repack, so nothing is duplicated. The
distance-dependent `blocks` (task 011) and `lh_A`/`lh_B` (task 012) coefficient
buffers are rebuilt per source/target distance; `lh_A`/`lh_B` are empty when `!LH`.
"""
struct M2LOperatorScratch{TF,B<:AbstractOperatorBasis,LH}
    base::OperatorScratch{TF,B,LH}
    work_a::FlatCoefficientBuffer{TF,Matrix{TF},B,LH}
    work_b::FlatCoefficientBuffer{TF,Matrix{TF},B,LH}
    blocks::Vector{TF}
    lh_A::Vector{TF}
    lh_B::Vector{TF}
end

function M2LOperatorScratch(::Type{TF}, basis_info::OperatorBasisInfo{B,LH}, batch_max::Integer) where {TF,B,LH}
    P_active = basis_info.orders.P_active
    nh = _operator_ncomplex(P_active)
    base = OperatorScratch(TF, basis_info)
    work_a = FlatCoefficientBuffer(TF, basis_info, batch_max)
    work_b = FlatCoefficientBuffer(TF, basis_info, batch_max)
    blocks = Vector{TF}(undef, m2l_z_block_length(P_active))
    lh_A = LH ? zeros(TF, nh) : TF[]
    lh_B = LH ? zeros(TF, nh) : TF[]
    return M2LOperatorScratch{TF,B,LH}(base, work_a, work_b, blocks, lh_A, lh_B)
end

M2LOperatorScratch(::Type{TF}, P::Integer, lamb_helmholtz::Val, batch_max::Integer) where TF =
    M2LOperatorScratch(TF, OperatorBasisInfo(P, lamb_helmholtz), batch_max)

#------- FULL M2M/L2L OPERATOR PIPELINES (Matrix Operator Refactor, task 016) -------#

abstract type AbstractM2MOperator end
abstract type AbstractL2LOperator end

struct MaterializedYRotationM2M <: AbstractM2MOperator end
# Physical-subspace invariant (016b): exact only for physical inputs (m=0 imag == 0);
# see the FactoredRotationM2L note above and `_assert_factored_input_physical`.
struct FactoredRotationM2M <: AbstractM2MOperator end

struct MaterializedYRotationL2L <: AbstractL2LOperator end
# Physical-subspace invariant (016b): exact only for physical inputs (m=0 imag == 0);
# see the FactoredRotationM2L note above and `_assert_factored_input_physical`.
struct FactoredRotationL2L <: AbstractL2LOperator end

# Resident batched-M2M GEMM strategies (task 022), swappable for `024` benchmarking.
# `DenseTranslationM2M` materializes the complete per-translation-vector operator and
# batches columns sharing that vector into one GEMM. `SharedRotationM2M` (the main
# path) batches all edges together, materializing only the per-vector z-axis pieces
# and applying the batch-shared y-rotation modes `U_n`/`V_n` by per-degree GEMM.
abstract type AbstractResidentM2MStrategy end
struct DenseTranslationM2M <: AbstractResidentM2MStrategy end
struct SharedRotationM2M <: AbstractResidentM2MStrategy end

abstract type AbstractResidentM2LStrategy end

"""
    DenseTranslationM2L(; max_persistent_bytes=4 << 30, apply_chunk=0, build_chunk=0,
        cuda_headroom_bytes=1 << 30)

Resident M2L strategy which stores one complete dense coefficient-space translation
matrix for every accepted displacement class. `apply_chunk` caps the number of
gathered routes in each class GEMM and `build_chunk` caps the number of identity
columns passed through the materialized-y construction oracle; zero selects the
largest useful width. `max_persistent_bytes` limits operator, application-slab, and
route-metadata payload storage.

Supported on the host lifecycle (task 023e) and, through
`RadixFMMCache(...; device=true, options=CUDARadixLifecycleOptions(
m2l_strategy=DenseTranslationM2L()))`, on the CUDA device-resident lifecycle (task
023f). `cuda_headroom_bytes` reserves free device memory the estimated dense
lifecycle footprint must not consume: the CUDA construction gate requires the
complete estimated device footprint to stay within `CUDA.free_memory() -
cuda_headroom_bytes`, in addition to the shared `max_persistent_bytes` payload gate.
The one-shot `cuda_radix_state` builders remain unsupported.
"""
struct DenseTranslationM2L <: AbstractResidentM2LStrategy
    max_persistent_bytes::Int
    apply_chunk::Int
    build_chunk::Int
    cuda_headroom_bytes::Int
    function DenseTranslationM2L(; max_persistent_bytes=4 << 30,
            apply_chunk=0, build_chunk=0, cuda_headroom_bytes=1 << 30)
        vals = (max_persistent_bytes=max_persistent_bytes,
            apply_chunk=apply_chunk, build_chunk=build_chunk,
            cuda_headroom_bytes=cuda_headroom_bytes)
        converted = map(vals) do value
            value isa Integer || throw(ArgumentError(
                "DenseTranslationM2L options must be integers; got $(typeof(value))"))
            try
                Int(value)
            catch err
                err isa InexactError || err isa OverflowError || rethrow()
                throw(ArgumentError("DenseTranslationM2L option $value is not representable as Int"))
            end
        end
        converted.max_persistent_bytes > 0 || throw(ArgumentError(
            "DenseTranslationM2L max_persistent_bytes must be positive"))
        converted.apply_chunk >= 0 || throw(ArgumentError(
            "DenseTranslationM2L apply_chunk must be nonnegative"))
        converted.build_chunk >= 0 || throw(ArgumentError(
            "DenseTranslationM2L build_chunk must be nonnegative"))
        converted.cuda_headroom_bytes >= 0 || throw(ArgumentError(
            "DenseTranslationM2L cuda_headroom_bytes must be nonnegative"))
        return new(converted.max_persistent_bytes, converted.apply_chunk,
            converted.build_chunk, converted.cuda_headroom_bytes)
    end
end
struct SharedRotationM2L <: AbstractResidentM2LStrategy end

# Whole-pass concatenated M2L (task 022 throughput repair). Instead of looping
# per-(r,theta,phi) groups, all routes are processed in fixed-width column chunks:
# the z-rotation and factored-y stages are already per-column parameterized, and the
# z-translation separates as K_m(r)[n,np] = r^-(n+1/2) * (n+np)! * r^-(np+1/2), so a
# per-column diagonal scaling before and after fixed factorial GEMMs replaces the
# per-radius block matrices. Kernel-launch count scales with chunks, not groups.
struct ConcatenatedFixedZM2L <: AbstractResidentM2LStrategy
    chunk::Int
    function ConcatenatedFixedZM2L(chunk::Integer=1 << 17)
        chunk > 0 || throw(ArgumentError("ConcatenatedFixedZM2L chunk must be positive"))
        return new(Int(chunk))
    end
end

"""
    PrecomputedFactoredYM2L()

Resident M2L strategy which precomputes the real factored-y block for each
exact polar-angle class in the fixed radix stencil.  It must be paired with
[`FactoredRotationM2L`](@ref).  Supported on the host lifecycle (task 023c)
and the CUDA device-resident lifecycle through `RadixFMMCache(...; device=true)`
(task 023d).
"""
struct PrecomputedFactoredYM2L <: AbstractResidentM2LStrategy end

# Capacity-sized grouped host plan selected by FactoredRotationM2L (task 023a).
# CUDA plans deliberately leave `groups` empty (task 023b compact-storage
# amendment): their per-class reference and whole-pass implementations use only
# the trailing route histogram/prefix metadata, class geometry, flat Plain-H
# y-mode vectors, and per-class fixed-m z tables. The 2-arg constructor (one-shot
# host path) leaves the trailing fields empty.
struct ResidentM2LFactoredPlan{R,G}
    route_class::R
    groups::G
    class_counts::Any        # per-class route histogram (device Int32 on CUDA)
    host_class_counts::Any   # host Int32 mirror of class_counts (pinned on CUDA)
    class_starts::Vector{Int}  # per-class start offsets into the route arrays (+1 sentinel)
    class_theta::Any         # host per-class θ scalars
    class_phi::Any           # host per-class φ scalars
    class_r::Any             # host per-class r scalars
    ym_flat::Any             # flat per-degree y-mode blocks (mult/loc × U/V × re/im)
    z_flat::Any              # zlen × nclasses fixed-m z-translation tables
    # whole-pass chunked execution bundle (device class tables + chunk-width slabs),
    # filled by the CUDA cache build; nothing on host and on the per-class-only path
    whole_pass::Base.RefValue{Any}
end

ResidentM2LFactoredPlan(route_class, groups) = ResidentM2LFactoredPlan(
    route_class, groups, nothing, nothing, Int[], nothing, nothing, nothing, nothing,
    nothing, Ref{Any}(nothing))

# Fixed-box precomputed-y plan (task 023c host, 023d CUDA).  `route_class` is
# filled by build_radix_routes! with the accepted-offset id (a device Int32 array
# on the CUDA lifecycle, where route emission writes it directly).  The host
# refresh stably packs the route indices into angle-major / offset-minor ranges;
# the device refresh keeps the emission's offset-class-major route order and only
# rebuilds the offset histogram/prefix.  All arrays are allocated once at cache
# construction; counts and prefixes are the only mutable contents.
#
# Compact CUDA plans (task 023d) leave the nested host operator storage
# (`y_mult`/`y_loc`/`z_phi`/`z_chi`/LH rows) and the packed route arrays empty and
# carry instead the trailing flat device fields: per-angle flat `M_n(theta)`
# block tables in `ymode_offset` layout (`y_flat_*`, one column per angle class),
# per-offset fixed-m z tables in `m2l_z_blocks!` layout (`z_flat`, one column per
# accepted offset), the device offset histogram plus its pinned host mirror, and
# the whole-pass chunked execution bundle filled by the CUDA cache build.
struct ResidentM2LPrecomputedYPlan{TF,S,R}
    route_class::R
    offset_to_angle::Vector{Int}
    angle_keys::Vector{NTuple{3,Int}}
    angle_thetas::Vector{TF}
    angle_offset_starts::Vector{Int}
    angle_offsets::Vector{Int}
    angle_capacities::Vector{Int}
    angle_counts::Vector{Int}
    angle_starts::Vector{Int}
    offset_counts::Vector{Int}
    offset_starts::Vector{Int}
    packed_sources::Vector{Int}
    packed_targets::Vector{Int}
    packed_phis::Vector{TF}
    offset_phis::Vector{TF}
    y_mult::Vector{Vector{Matrix{TF}}}
    y_loc::Vector{Vector{Matrix{TF}}}
    z_phi::Vector{Vector{Matrix{TF}}}
    z_chi::Vector{Vector{Matrix{TF}}}
    lh_phi_rows::Vector{Vector{TF}}
    lh_chi_rows::Vector{Vector{TF}}
    scratch::S
    # --- device extension (task 023d); nothing/empty on host plans ---
    class_counts::Any        # per-offset route histogram (device Int32 on CUDA)
    host_class_counts::Any   # host Int32 mirror of class_counts (pinned on CUDA)
    y_flat_mult::Any         # flat multipole M_n(theta) blocks, one column per angle
    y_flat_loc::Any          # flat local M_n(theta) blocks, one column per angle
    z_flat::Any              # zlen x noffsets fixed-m z-translation tables
    offset_rs::Any           # host per-offset radii (LH unit-row scaling)
    whole_pass::Base.RefValue{Any}
end

# Back-compatible host construction: the 023c host builder supplies exactly the
# leading fields; the trailing device extension stays empty.
ResidentM2LPrecomputedYPlan(route_class, offset_to_angle, angle_keys, angle_thetas,
        angle_offset_starts, angle_offsets, angle_capacities, angle_counts,
        angle_starts, offset_counts, offset_starts, packed_sources, packed_targets,
        packed_phis, offset_phis, y_mult, y_loc, z_phi, z_chi, lh_phi_rows,
        lh_chi_rows, scratch) =
    ResidentM2LPrecomputedYPlan(route_class, offset_to_angle, angle_keys,
        angle_thetas, angle_offset_starts, angle_offsets, angle_capacities,
        angle_counts, angle_starts, offset_counts, offset_starts, packed_sources,
        packed_targets, packed_phis, offset_phis, y_mult, y_loc, z_phi, z_chi,
        lh_phi_rows, lh_chi_rows, scratch, nothing, nothing, nothing, nothing,
        nothing, nothing, Ref{Any}(nothing))

# Complete host coefficient-space M2L plan (task 023e). Every array is allocated
# at construction capacity and refreshed in place; the byte fields count array
# payloads only (not Julia object/container headers).
struct ResidentM2LDensePlan{TF}
    route_class::Vector{Int32}
    class_counts::Vector{Int}
    class_starts::Vector{Int}
    class_capacities::Vector{Int}
    packed_sources::Vector{Int}
    packed_targets::Vector{Int}
    operators::Vector{Matrix{TF}}
    class_operator::Vector{Int}
    source_scale::Matrix{TF}
    target_scale::Matrix{TF}
    src_slab::Matrix{TF}
    dst_slab::Matrix{TF}
    ndof::Int
    width::Int
    operator_bytes::Int
    scratch_bytes::Int
    route_metadata_bytes::Int
    persistent_bytes::Int
    construction_peak_bytes::Int
end

# Device-resident coefficient-space M2L plan (task 023f). Distinct from the host
# `ResidentM2LDensePlan` so the host fields stay concretely typed; this plan holds
# only preallocated device execution state plus host-side per-step count staging.
#
# `operators` is the packed device operator array, one contiguous column-major
# `ndof x ndof` slice per accepted displacement class (`operators[:, :, k]`).
# `route_class` is the device Int32 offset-class id filled by route emission;
# device routes are offset-class-major and contiguous, so no source/target repack
# exists here (`class_starts` addresses contiguous views of the state route arrays).
# `class_counts` is the device Int32 per-class histogram, `host_class_counts` its
# pinned host mirror, and `class_starts` the 1-based host prefix (+1 sentinel) the
# per-step refresh rebuilds in place. `src_slab`/`dst_slab` are the chunk-width
# device gather/GEMM/scatter slabs. `whole_pass` carries the chunk-width execution
# bundle filled by the CUDA cache build; the byte fields count array payloads only.
struct ResidentM2LDenseCUDAPlan{TF,O,OH,OB,OS,R,C,S}
    route_class::R
    operators::O
    tensor_fp16_operators::OH
    tensor_bf16_operators::OB
    tensor_input_scale::OS
    class_counts::C
    host_class_counts::Vector{Int32}
    class_starts::Vector{Int}
    class_capacities::Vector{Int}
    src_slab::S
    dst_slab::S
    nclasses::Int
    ndof::Int
    ndof_phi::Int
    width::Int
    operator_bytes::Int
    scratch_bytes::Int
    route_metadata_bytes::Int
    persistent_bytes::Int
    estimated_peak_bytes::Int
    whole_pass::Base.RefValue{Any}
end

"""
    M2MOperatorScratch{TF,B,LH}

Working storage for the batched full-M2M operator pipeline, sized for a maximum
batch width and active order. It reuses the same base operator scratch shape as
M2L, with two lifetime-aliased batch work buffers, a structured M2M z-block
buffer, and optional Lamb-Helmholtz coefficient buffers.
"""
struct M2MOperatorScratch{TF,B<:AbstractOperatorBasis,LH}
    base::OperatorScratch{TF,B,LH}
    work_a::FlatCoefficientBuffer{TF,Matrix{TF},B,LH}
    work_b::FlatCoefficientBuffer{TF,Matrix{TF},B,LH}
    blocks::Vector{TF}
    lh_A::Vector{TF}
    lh_B::Vector{TF}
end

function M2MOperatorScratch(::Type{TF}, basis_info::OperatorBasisInfo{B,LH}, batch_max::Integer) where {TF,B,LH}
    P_active = basis_info.orders.P_active
    nh = _operator_ncomplex(P_active)
    base = OperatorScratch(TF, basis_info)
    work_a = FlatCoefficientBuffer(TF, basis_info, batch_max)
    work_b = FlatCoefficientBuffer(TF, basis_info, batch_max)
    blocks = Vector{TF}(undef, m2m_z_block_length(P_active))
    lh_A = LH ? zeros(TF, nh) : TF[]
    lh_B = LH ? zeros(TF, nh) : TF[]
    return M2MOperatorScratch{TF,B,LH}(base, work_a, work_b, blocks, lh_A, lh_B)
end

M2MOperatorScratch(::Type{TF}, P::Integer, lamb_helmholtz::Val, batch_max::Integer) where TF =
    M2MOperatorScratch(TF, OperatorBasisInfo(P, lamb_helmholtz), batch_max)

"""
    L2LOperatorScratch{TF,B,LH}

Working storage for the batched full-L2L operator pipeline, with the same layout
and lifetime rules as [`M2MOperatorScratch`](@ref), but with L2L z-translation
blocks.
"""
struct L2LOperatorScratch{TF,B<:AbstractOperatorBasis,LH}
    base::OperatorScratch{TF,B,LH}
    work_a::FlatCoefficientBuffer{TF,Matrix{TF},B,LH}
    work_b::FlatCoefficientBuffer{TF,Matrix{TF},B,LH}
    blocks::Vector{TF}
    lh_A::Vector{TF}
    lh_B::Vector{TF}
end

function L2LOperatorScratch(::Type{TF}, basis_info::OperatorBasisInfo{B,LH}, batch_max::Integer) where {TF,B,LH}
    P_active = basis_info.orders.P_active
    nh = _operator_ncomplex(P_active)
    base = OperatorScratch(TF, basis_info)
    work_a = FlatCoefficientBuffer(TF, basis_info, batch_max)
    work_b = FlatCoefficientBuffer(TF, basis_info, batch_max)
    blocks = Vector{TF}(undef, l2l_z_block_length(P_active))
    lh_A = LH ? zeros(TF, nh) : TF[]
    lh_B = LH ? zeros(TF, nh) : TF[]
    return L2LOperatorScratch{TF,B,LH}(base, work_a, work_b, blocks, lh_A, lh_B)
end

L2LOperatorScratch(::Type{TF}, P::Integer, lamb_helmholtz::Val, batch_max::Integer) where TF =
    L2LOperatorScratch(TF, OperatorBasisInfo(P, lamb_helmholtz), batch_max)

#------- CUDA device-resident radix lifecycle metadata (task 022) -------#
#
# These containers intentionally avoid CUDA-specific types so the CPU package path
# can load without touching a device runtime. The CUDA implementation fills them
# with CuArray-backed buffers from src/translate_batched_cuda.jl after the caller
# opts in via load_cuda_radix_lifecycle!().

abstract type Residency end

struct HostResident <: Residency end

struct DeviceResident <: Residency end

mutable struct CUDARadixTransferCounters
    body_uploads::Int
    influence_downloads::Int
    expansion_host_copies::Int
    route_uploads::Int
    operator_uploads::Int
    # host mirrors of step-varying sort metadata (perm/system/index), needed only
    # to finalize into host-resident targets; kept separate from
    # influence_downloads so the 022 "download only per-body influence" contract
    # stays auditable
    metadata_downloads::Int
end

CUDARadixTransferCounters() = CUDARadixTransferCounters(0, 0, 0, 0, 0, 0)

struct CUDARadixLifecycleOptions{TF,O<:AbstractM2LOperator,
        M2M<:AbstractResidentM2MStrategy,M2L<:AbstractResidentM2LStrategy}
    precision::Type{TF}
    operator::O
    m2m_strategy::M2M
    m2l_strategy::M2L
end

# Preserve the historical partial form `CUDARadixLifecycleOptions{TF}(...)` while
# making all three dispatch choices part of the concrete options type.
CUDARadixLifecycleOptions{TF}(precision, operator, m2m_strategy, m2l_strategy) where TF =
    CUDARadixLifecycleOptions{TF,typeof(operator),typeof(m2m_strategy),
        typeof(m2l_strategy)}(precision, operator, m2m_strategy, m2l_strategy)

CUDARadixLifecycleOptions{TF}(;
        operator=MaterializedYRotationM2L(),
        m2m_strategy=SharedRotationM2M(),
        m2l_strategy=SharedRotationM2L()) where TF =
    CUDARadixLifecycleOptions(; precision=TF, operator, m2m_strategy, m2l_strategy)

function CUDARadixLifecycleOptions(;
        precision::Type{TF}=Float64,
        operator=MaterializedYRotationM2L(),
        m2m_strategy=SharedRotationM2M(),
        m2l_strategy=SharedRotationM2L(),
    ) where TF
    TF <: Union{Float32,Float64} ||
        throw(ArgumentError("CUDA radix lifecycle precision must be Float32 or Float64"))
    operator isa AbstractM2LOperator ||
        throw(ArgumentError("operator must be an AbstractM2LOperator"))
    m2m_strategy isa AbstractResidentM2MStrategy ||
        throw(ArgumentError("m2m_strategy must be an AbstractResidentM2MStrategy"))
    m2l_strategy isa AbstractResidentM2LStrategy ||
        throw(ArgumentError("m2l_strategy must be an AbstractResidentM2LStrategy"))
    if m2l_strategy isa PrecomputedFactoredYM2L && !(operator isa FactoredRotationM2L)
        throw(ArgumentError("PrecomputedFactoredYM2L requires operator=FactoredRotationM2L()"))
    end
    if m2l_strategy isa DenseTranslationM2L && !(operator isa MaterializedYRotationM2L)
        throw(ArgumentError("DenseTranslationM2L requires operator=MaterializedYRotationM2L()"))
    end
    return CUDARadixLifecycleOptions{TF}(precision, operator, m2m_strategy, m2l_strategy)
end

# Step-varying prefix lengths for a capacity-sized DeviceResidentRadixState (task
# 023): arrays stay allocated at construction capacity and each count bounds the
# valid prefix, so recurring time steps never reallocate. One-shot construction
# initializes every count to the corresponding full array length.
mutable struct RadixStepCounts
    n_bodies::Int
    n_cells::Int
    n_nodes::Int
    n_routes::Int
    n_direct::Int
end

# Constructors enforce the shared container invariants encoded below: floating-point
# matrices use one backend type; device index vectors and matrices each use one type;
# host mirrors use their own vector/matrix types; and the two flat buffers match.
# Optional host/device storage remains separately parameterized.  The concrete options
# type deliberately participates in the state type so default lifecycle launchers can
# infer operator and strategy dispatch without boxing the state.
struct DeviceResidentRadixState{TF,B,LH,
        GR,IL,FM,HBV,HRV,HFM,DIV,DIM,
        FB<:FlatCoefficientBuffer{TF,<:AbstractMatrix{TF},B,LH},
        IC,SC,OPT<:CUDARadixLifecycleOptions{TF}}
    grid::GR
    interaction_list::IL
    source_bodies::FM
    target_bodies::FM
    body_perm::DIV
    body_system_ids::DIV
    body_indices::DIV
    host_body_perm::HBV
    host_body_system_ids::HBV
    host_body_indices::HBV
    host_cell_centers::HFM
    host_m2m_parent_routes::HRV
    host_m2m_child_routes::HRV
    host_l2l_parent_routes::HRV
    host_l2l_child_routes::HRV
    host_node_levels::HRV
    host_node_centers::HFM
    host_route_targets::HRV
    host_route_sources::HRV
    cell_centers::FM
    cell_ranges::DIM
    m2m_parent_routes::DIV
    m2m_child_routes::DIV
    l2l_parent_routes::DIV
    l2l_child_routes::DIV
    multipoles::FB
    locals::FB
    route_levels::DIV
    route_offsets::DIM
    route_targets::DIV
    route_sources::DIV
    direct_targets::DIV
    direct_sources::DIV
    output::FM
    invariant_cache::IC
    scratch::SC
    counters::CUDARadixTransferCounters
    options::OPT
    counts::RadixStepCounts
end

@inline _radix_count_len(x) = x === nothing ? 0 : length(x)

"""
    RadixFMMCache{TF,LH}

Opt-in production cache for the radix-grid / matrix-operator FMM path (task 023).
Construct once with [`RadixFMMCache`](@ref)`(target_systems, source_systems; ...)`
and pass to `fmm!(system, cache)` each time step; construction eagerly builds the
capacity-sized [`DeviceResidentRadixState`](@ref) plus all step-invariant operator
data, so every `fmm!` call is the fast path and no persistent host or device array
is reallocated across steps. Bounded Morton depths use persistent counting-sort
scratch; larger depths retain CUDA's pool-served device sort scratch.

The invariant contract: the domain box (`x_min`, `h0`), depth `ell`, expansion
order, and `max_n_bodies` are fixed at construction. Each step may move bodies and
change their number (up to `max_n_bodies`), but positions must stay inside the
fixed box; violations throw `ArgumentError` rather than silently rebuilding the
step-invariant geometry tables.

v1 scope restrictions (documented at `fmm!`): `target_systems === source_systems`,
no hessian output, host- or device-resident execution selected at construction.
"""
mutable struct RadixFMMCache{TF,LH}
    # construction parameters — the invariant contract
    expansion_order::Int
    ell::Int
    x_min::SVector{3,TF}
    h0::TF
    max_n_bodies::Int
    device::Bool
    options::CUDARadixLifecycleOptions{TF}
    policy::Any                     # ConstantPAnalyticStencil
    # step-invariant stencil classification and capacity bounds
    accepted_offsets::Vector{SVector{3,Int}}
    rejected_offsets::Vector{SVector{3,Int}}
    max_cells::Int
    max_nodes::Int
    route_capacity::Int
    direct_capacity::Int
    # step-varying containers (capacity-sized, prefix-valid via state.counts)
    state::Any                      # DeviceResidentRadixState
    cell_at::Array{Int32,3}         # dense occupancy map (host)
    coords::Vector{SVector{3,Int}}  # decoded coords of occupied cells (host)
    level_offsets::Vector{Int}      # node index offset per level (length ell + 2)
    body_keys::Vector{UInt64}       # Morton keys scratch (host sort)
    sort_scratch::Vector{Int}
    sort_counts::Vector{Int}
    sort_offsets::Vector{Int}
    source_buffers::Any             # NTuple{N,Matrix{TF}} capacity-width repack buffers
    target_buffers::Any             # per-switch-layout scatter buffers (lazy)
    device_ctx::Any                 # CUDA-side update context (task 023 step 7)
    n_systems::Int
    built::Bool
    step::Int
end

function RadixStepCounts(source_bodies, cell_ranges, multipoles::FlatCoefficientBuffer,
        route_targets, direct_targets)
    n_bodies = source_bodies === nothing ? 0 : size(source_bodies, 2)
    n_cells = cell_ranges === nothing ? 0 : size(cell_ranges, 2)
    return RadixStepCounts(
        n_bodies, n_cells, size(multipoles.phi, 2),
        _radix_count_len(route_targets), _radix_count_len(direct_targets),
    )
end

# Partial-application constructor: every call site writes
# `DeviceResidentRadixState{TF,B,LH}(fields...)` and the per-field parameters are
# filled in here. Concrete `DeviceResidentRadixState{TF,B,LH,P...}` is a subtype of the
# UnionAll `DeviceResidentRadixState{TF,B,LH}`, so every existing method signature and
# typeassert keeps matching unchanged. Field order duplicates the struct above; an
# arity mismatch fails loudly at the first construction.
function DeviceResidentRadixState{TF,B,LH}(grid, interaction_list, source_bodies,
        target_bodies, body_perm, body_system_ids, body_indices,
        host_body_perm, host_body_system_ids, host_body_indices,
        host_cell_centers, host_m2m_parent_routes, host_m2m_child_routes,
        host_l2l_parent_routes, host_l2l_child_routes, host_node_levels,
        host_node_centers, host_route_targets, host_route_sources,
        cell_centers, cell_ranges, m2m_parent_routes, m2m_child_routes,
        l2l_parent_routes, l2l_child_routes, multipoles, locals,
        route_levels, route_offsets, route_targets, route_sources,
        direct_targets, direct_sources, output, invariant_cache, scratch,
        counters, options, counts) where {TF,B,LH}
    return DeviceResidentRadixState{TF,B,LH,
        typeof(grid),typeof(interaction_list),typeof(source_bodies),
        typeof(host_body_perm),typeof(host_m2m_parent_routes),
        typeof(host_cell_centers),typeof(body_perm),
        typeof(cell_ranges),typeof(multipoles),typeof(invariant_cache),
        typeof(scratch),typeof(options)}(
        grid, interaction_list, source_bodies, target_bodies, body_perm,
        body_system_ids, body_indices, host_body_perm, host_body_system_ids,
        host_body_indices, host_cell_centers, host_m2m_parent_routes,
        host_m2m_child_routes, host_l2l_parent_routes, host_l2l_child_routes,
        host_node_levels, host_node_centers, host_route_targets,
        host_route_sources, cell_centers, cell_ranges, m2m_parent_routes,
        m2m_child_routes, l2l_parent_routes, l2l_child_routes, multipoles, locals,
        route_levels, route_offsets, route_targets, route_sources, direct_targets,
        direct_sources, output, invariant_cache, scratch, counters, options, counts,
    )
end

function DeviceResidentRadixState{TF,B,LH}(grid, interaction_list, source_bodies,
        target_bodies, multipoles, locals, route_levels, route_offsets,
        route_targets, route_sources, output, invariant_cache, scratch, counters,
        options) where {TF,B,LH}
    return DeviceResidentRadixState{TF,B,LH}(
        grid, interaction_list, source_bodies, target_bodies,
        nothing, nothing, nothing, nothing, nothing, nothing,
        nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing,
        nothing, nothing, nothing, nothing, nothing, nothing,
        multipoles, locals, route_levels, route_offsets, route_targets,
        route_sources, nothing, nothing, output, invariant_cache, scratch, counters, options,
        RadixStepCounts(source_bodies, nothing, multipoles, route_targets, nothing),
    )
end
