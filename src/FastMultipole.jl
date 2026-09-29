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
const MIN_NPT_B2M = 10
const MIN_NPT_M2M = 10
const MIN_NPT_M2L = 10
const MIN_NPT_L2L = 10
const MIN_NPT_L2B = 10
const MIN_NPT_NF = 10
const MIN_NPT_BRANCH = 1 # if fewer branches than this, multithread over bodies instead of branches
                         # TODO: this should probably be a function of the number of threads
const MIN_NPT_SORT = 1
const MIN_NPT_MUL_SORT = 1
const MIN_NPT = 1
const MIN_BODIES = 100

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
export Branch, Tree, ConstantPAnalyticStencil, HierarchicalRigidStencil, classic_fmm_stencil, Residency,
    HostResident, DeviceResident, AbstractDirectKernel, SingularSource, SingularVortex, SingularDipole,
    SingularSourceVortex, RegularizedVortex, SourceFilamentKernel, DipoleFilamentKernel,
    VortexFilamentKernel, SourcePanelKernel, DipolePanelKernel, SourceDipolePanelKernel,
    VortexSheetPanelKernel, PartitionedVortex, TwoPassVortex, RadixTransferCounters,
    RadixLifecycleOptions,
    AbstractResidentM2LStrategy, DenseTranslationM2L, ConcatenatedFixedZM2L,
    PrecomputedFactoredYM2L, RadixFMMCache

include("complex.jl")
include("derivatives.jl")
include("harmonics.jl")
include("rotate.jl")
include("rotate_batched.jl")
include("translate.jl")
include("translate_batched.jl")
include("evaluate_expansions.jl")
include("tree.jl")
export initialize_expansion, initialize_harmonics, unsorted_index_2_sorted_index,
    sorted_index_2_unsorted_index

include("tree_batched.jl")

include("interaction_list_batched.jl")

include("resident/resident_grid_state.jl")
include("resident/resident_b2m.jl")
include("resident/resident_pair_kernels.jl")
include("resident/resident_finalize.jl")
include("resident/radix_cache.jl")
include("resident/resident_device_plumbing.jl")
export update_radix_state!

include("resident_elements.jl")
include("resident_extra_tree.jl")
include("radix_extra_systems.jl")
include("radix_nearfield.jl")
export radix_nearfield
include("radix_settings.jl")
export radix_settings, radix_setting
export set_radix_setting!, set_radix_settings!

include("direct_rectangular.jl")
export AbstractRectangularKernel, RectangularGaussianErfVortex, RectangularPanelInfluence
export direct_rectangular!


"""
    RadixDeviceUnavailable(reason)

Exception thrown when a device-resident radix operation is requested without a
registered device backend. `reason` is reported by `showerror`.
"""
struct RadixDeviceUnavailable <: Exception
    reason::String
end
export RadixDeviceUnavailable

Base.showerror(io::IO, err::RadixDeviceUnavailable) = print(io, err.reason)

#------- device-backend registry -------#
#
# The device-resident radix lifecycle is provided by a package extension
# (ext/FastMultipoleKAExt.jl for KernelAbstractions backends: CUDA, Metal, ...).
# The extension REGISTERS its entry points here from its `__init__`, and the
# host entry points (`RadixFMMCache(...; device=true)`, `fmm!`,
# `update_radix_state!`) consult the registry before throwing.
# The former hand-written CUDA lifecycle (runtime-`include`d into this module)
# was removed after the KA port reached parity with it.
const _RADIX_DEVICE_BACKEND_NAME = Ref{Any}(nothing)
const _RADIX_DEVICE_BUILD_HOOK = Ref{Any}(nothing)
const _RADIX_DEVICE_STEP_HOOK = Ref{Any}(nothing)
# a device cache repacks its bodies and refreshes its lists through the extension
# (`update_radix_state!` on a device cache)
const _RADIX_DEVICE_UPDATE_HOOK = Ref{Any}(nothing)

"""
    register_radix_device_backend!(name, build, step!)

Register a non-CUDA device-resident radix lifecycle. `build` is called with the
argument list of `_radix_cache_device_build` and must return a built
`RadixFMMCache`; `step!` is called as `step!(cache, targets, switches; nearfield_pass, ...)`.
Called from a package extension's `__init__`.
"""
function register_radix_device_backend!(name, build, step!)
    _RADIX_DEVICE_BACKEND_NAME[] = name
    _RADIX_DEVICE_BUILD_HOOK[] = build
    _RADIX_DEVICE_STEP_HOOK[] = step!
    return nothing
end

"A non-CUDA device-resident radix lifecycle is registered."
radix_device_backend_available() = _RADIX_DEVICE_STEP_HOOK[] !== nothing

function radix_device_status()
    name = _RADIX_DEVICE_BACKEND_NAME[]
    name === nothing && return "no device radix backend registered; load a backend " *
        "extension (e.g. `using KernelAbstractions` together with CUDA or Metal)"
    return "device radix lifecycle provided by $(name)"
end

include("compatibility.jl")
export residency, direct_kernel

export Position, Radius, ScalarPotential, Gradient, Hessian, Vertex, Normal, Strength
export Vortex, Source, Dipole, SourceDipole, SourceVortex, Point, Filament, Panel
export PowerAbsolutePotential, PowerAbsoluteGradient, RotatedCoefficientsAbsoluteGradient
# export PowerRelativePotential, PowerRelativeGradient, RotatedCoefficientsRelativeGradient
export get_n_bodies, body_to_multipole!, direct!
export source_to_buffer!, source_to_buffer, buffer_to_target!
export body_type, data_per_body, strength_dims, has_vector_potential, get_position
export recenter!

include("direct_conditioning.jl")

export DirectConditioningRule, SelfPairs, PairSet, AllPairs, applies

include("bodytomultipole.jl")

export body_to_multipole!

include("direct.jl")

export direct!

include("derivativesswitch.jl")

export DerivativesSwitch, ThirdDerivativeTensor, packed_data, dense
export metadata_range, metadata_index, tree_carried_range
export scalar_potential_index, gradient_range, hessian_range, third_derivative_range
export standard_output_range, extra_output_range, output_range
export get_extra_output, set_extra_output!, extra_output_view, output_view
export get_third_derivative, set_third_derivative!, supports_third_derivative

include("error.jl")

export multipole_error, local_error

include("interaction_list.jl")

export build_interaction_lists

include("fmm.jl")
export transform_tree!, transform_plan!

export InteractionList, fmm!, SelfTuning, Barba

include("autotune.jl")

export tune_fmm

include("visualize.jl")

export visualize

include("probes.jl")

include("solve.jl")

include("nearfield_cache.jl")

export NearfieldInfluenceCache, nearfield_matvec!, build_nearfield_cache!, estimate_nearfield_cache
export NearfieldCacheDonor, retarget_nearfield_cache
export assemble_influence_block!, overrides_block_assembly

include("extra_farfield.jl")

export FastGaussSeidel, JacobiPreconditioner, transform_solver!

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
