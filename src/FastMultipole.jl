module FastMultipole

#------- IMPORTS -------#

import Base.:^
using LinearAlgebra
using StaticArrays
using WriteVTK

#------- CONSTANTS -------#

const ONE_OVER_4π = 1/(4*π)
const ONE_THIRD = 1/3
const π_over_2 = π/2
const π2 = 2*π
const SQRT3 = sqrt(3.0)
const LOCAL_ERROR_SAFETY = 2.0 # guess how many cells will contribute to the local error
                               # NOTE: this doesn't apply to multipole error as that error is 
                               # highly localized and doesn't accumulate
const DEBUG = Array{Bool,0}(undef)
DEBUG[] = false

# multithreading parameters
const MIN_NPT_B2M = 100
const MIN_NPT_M2M = 100
const MIN_NPT_M2L = 100
const MIN_NPT_L2L = 100
const MIN_NPT_L2B = 100
const MIN_NPT_NF = 100
const MIN_NPT_BRANCH = 9 # if fewer branches than this, multithread over bodies instead of branches
                         # TODO: this should probably be a function of the number of threads
const MIN_NPT_SORT = 10000
const MIN_NPT_MUL_SORT = 100
const MIN_NPT = 100
const MIN_BODIES = 10000

# preallocate y-axis rotation matrices by π/2
const Hs_π2 = Float64[1.0]

# preallocate y-axis rotation Wigner matrix normalization
const ζs_mag = Float64[1.0]
const ηs_mag = Float64[1.0]

# preallocate multipole/local power normalization constants
const M̃ = Float64[1.0]
const L̃ = Float64[1.0]

#------- WARNING FLAGS -------#

const WARNING_FLAG_LEAF_SIZE = Array{Bool,0}(undef)
WARNING_FLAG_LEAF_SIZE[] = true

const WARNING_FLAG_PMAX = Array{Bool,0}(undef)
WARNING_FLAG_PMAX[] = true

const WARNING_FLAG_ERROR = Array{Bool,0}(undef)
WARNING_FLAG_ERROR[] = true

const WARNING_FLAG_SCALAR_POTENTIAL = Array{Bool,0}(undef)
WARNING_FLAG_SCALAR_POTENTIAL[] = true

const WARNING_FLAG_VECTOR_POTENTIAL = Array{Bool,0}(undef)
WARNING_FLAG_VECTOR_POTENTIAL[] = true

const WARNING_FLAG_gradient = Array{Bool,0}(undef)
WARNING_FLAG_gradient[] = true

const WARNING_FLAG_hessian = Array{Bool,0}(undef)
WARNING_FLAG_hessian[] = true

const WARNING_FLAG_STRENGTH = Array{Bool,0}(undef)
WARNING_FLAG_STRENGTH[] = true

const WARNING_FLAG_B2M = Array{Bool,0}(undef)
WARNING_FLAG_B2M[] = true

const WARNING_FLAG_DIRECT = Array{Bool,0}(undef)
WARNING_FLAG_DIRECT[] = true

const WARNING_FLAG_LH_POTENTIAL = Array{Bool,0}(undef)
WARNING_FLAG_LH_POTENTIAL[] = true

const WARNING_FLAG_MAX_INFLUENCE = Array{Bool,0}(undef)
WARNING_FLAG_MAX_INFLUENCE[] = true

#------- HEADERS AND EXPORTS -------#

include("containers.jl")
include("complex.jl")
include("derivatives.jl")
include("harmonics.jl")
include("rotate.jl")
include("rotate_batched.jl")
include("translate.jl")
include("translate_batched.jl")
include("evaluate_expansions.jl")
include("tree.jl")
include("tree_batched.jl")
include("interaction_list_batched.jl")
include("translate_batched_resident.jl")

export Branch, SingleBranch, MultiBranch, Tree, SingleTree, MultiTree, initialize_expansion, initialize_harmonics
export RadixGrid, DeviceRadixGrid, RadixSortBackend, HostRadixSort, DeviceRadixSort, AutoRadixSort, radix_grid
export ConstantPStencilConfig, RadixSeparationPolicy, ParentNeighborM2L, ConstantPAnalyticStencil, HierarchicalRigidStencil, classic_fmm_stencil, rigid_stencil_epsilon, RadixTraversalStrategy
export RigidHierarchicalTables, RadixLevelOccupancy
export RigidImplicitStencil, SparseOffsetIntersection, BlockedOccupancyBitsets, LazyMaterializedBatches
export RadixM2LBatch, RadixInteractionList
export Residency, HostResident, DeviceResident, residency
export AbstractDirectKernel, SingularSource, SingularVortex, RegularizedVortex, direct_kernel
export CUDARadixTransferCounters, CUDARadixLifecycleOptions, DeviceResidentRadixState
export AbstractResidentM2MStrategy, DenseTranslationM2M, SharedRotationM2M
export AbstractResidentM2LStrategy, DenseTranslationM2L, SharedRotationM2L, ConcatenatedFixedZM2L, PrecomputedFactoredYM2L
export constant_p_stencil_bound, constant_p_stencil_accepts, accepted_radix_stencil
export foreach_radix_m2l_pair, foreach_radix_m2l_route, foreach_radix_direct_pair, build_radix_interaction_list
export unsort!, resort!, unsorted_index_2_sorted_index, sorted_index_2_unsorted_index
export AbstractOperatorBasis, CompressedComplexBasis, RealSolidHarmonicBasis
export OperatorOrders, OperatorBasisInfo, OperatorInvariantCache, OperatorScratch, ThreadedOperatorScratch
export FlatCoefficientBuffer, real_basis_index, complex_to_real_basis!, real_to_complex_basis!
export AbstractM2LOperator, MaterializedYRotationM2L, FactoredRotationM2L, M2LOperatorScratch
export AbstractM2MOperator, MaterializedYRotationM2M, FactoredRotationM2M, M2MOperatorScratch
export AbstractL2LOperator, MaterializedYRotationL2L, FactoredRotationL2L, L2LOperatorScratch
export load_cuda_radix_lifecycle!, cuda_radix_available, cuda_radix_status
export CUDARadixUnavailable, cuda_radix_state, run_cuda_radix_lifecycle!
export copy_cuda_radix_output!, finalize_cuda_radix_output!, take_cuda_radix_output!, cuda_radix_grid
export host_radix_state, run_host_radix_lifecycle!, host_resident_radix_grid
export finalize_radix_output!
export RadixFMMCache, update_radix_state!

struct CUDARadixUnavailable <: Exception
    reason::String
end

Base.showerror(io::IO, err::CUDARadixUnavailable) = print(io, err.reason)

const _CUDA_RADIX_LIFECYCLE_LOADED = Ref(false)
const _CUDA_RADIX_LIFECYCLE_LOAD_ERROR = Ref{Any}(nothing)

function _cuda_radix_preflight_error()
    get(ENV, "FASTMULTIPOLE_FORCE_CUDA_LOAD", "0") == "1" && return nothing
    Sys.isapple() && return "CUDA is not available on macOS in this runtime"
    if Sys.islinux() && !ispath("/dev/nvidiactl") && !ispath("/proc/driver/nvidia/version")
        return "no NVIDIA device nodes detected; set FASTMULTIPOLE_FORCE_CUDA_LOAD=1 to force CUDA.jl loading"
    end
    return nothing
end

cuda_radix_available() = false
function cuda_radix_status()
    err = _CUDA_RADIX_LIFECYCLE_LOAD_ERROR[]
    err === nothing && return "CUDA radix lifecycle not loaded; call load_cuda_radix_lifecycle!()"
    return "CUDA radix lifecycle failed to load: $(err)"
end

function load_cuda_radix_lifecycle!()
    # cuda_radix_available/cuda_radix_status are redefined by the include below,
    # so they must be reached through invokelatest from this (older-world) frame.
    _CUDA_RADIX_LIFECYCLE_LOADED[] && return Base.invokelatest(cuda_radix_available)::Bool
    _CUDA_RADIX_LIFECYCLE_LOAD_ERROR[] = nothing
    preflight = _cuda_radix_preflight_error()
    if preflight !== nothing
        _CUDA_RADIX_LIFECYCLE_LOAD_ERROR[] = preflight
        return false
    end
    try
        include(joinpath(@__DIR__, "translate_batched_cuda.jl"))
    catch err
        _CUDA_RADIX_LIFECYCLE_LOAD_ERROR[] = err
        return false
    end
    _CUDA_RADIX_LIFECYCLE_LOADED[] = true
    return Base.invokelatest(cuda_radix_available)::Bool
end

cuda_radix_state(args...; kwargs...) = throw(CUDARadixUnavailable(cuda_radix_status()))
cuda_radix_grid(args...; kwargs...) = throw(CUDARadixUnavailable(cuda_radix_status()))
run_cuda_radix_lifecycle!(args...; kwargs...) = throw(CUDARadixUnavailable(cuda_radix_status()))
copy_cuda_radix_output!(args...; kwargs...) = throw(CUDARadixUnavailable(cuda_radix_status()))
finalize_cuda_radix_output!(args...; kwargs...) = throw(CUDARadixUnavailable(cuda_radix_status()))
take_cuda_radix_output!(args...; kwargs...) = throw(CUDARadixUnavailable(cuda_radix_status()))

include("compatibility.jl")

export Body, Position, Radius, ScalarPotential, Gradient, Hessian, Vertex, Normal, Strength
export Vortex, Source, Dipole, SourceDipole, SourceVortex, Point, Filament, Panel
export PowerAbsolutePotential, PowerAbsoluteGradient, RotatedCoefficientsAbsoluteGradient
# export PowerRelativePotential, PowerRelativeGradient, RotatedCoefficientsRelativeGradient
export get_n_bodies, buffer_element, body_to_multipole!, direct!, direct_gpu!
export source_system_to_device_buffer!, target_system_from_device_buffer!, source_to_buffer!, source_to_buffer

include("direct_conditioning.jl")

export DirectConditioningRule, SelfPairs, PairSet, AllPairs, applies

include("bodytomultipole.jl")

export body_to_multipole!

include("direct.jl")

export direct!

include("derivativesswitch.jl")

export DerivativesSwitch, metadata_range, metadata_index, tree_carried_range
export scalar_potential_index, gradient_range, hessian_range
export standard_output_range, extra_output_range, output_range
export get_extra_output, set_extra_output!, extra_output_view, output_view

include("error.jl")

export multipole_error, local_error, error

include("interaction_list.jl")

export build_interaction_lists

include("fmm.jl")

export InteractionList, fmm!, SelfTuning, Barba

include("autotune.jl")

export tune_fmm

include("visualize.jl")

export visualize

include("probes.jl")

include("solve.jl")

include("extra_farfield.jl")

export FastGaussSeidel, JacobiPreconditioner

#------- PRECALCULATIONS -------#

# precompute y-axis rotation by π/2 matrices up to 20th order
update_Hs_π2!(Hs_π2, 21)

# precompute y-axis Wigner matrix normalization up to 20th order
update_ζs_mag!(ζs_mag, 21)
update_ηs_mag!(ηs_mag, 21)

# precompute multipole/local power normalization constansts up to 20th order
update_M̃!(M̃, 21)
update_L̃!(L̃, 21)

end # module
