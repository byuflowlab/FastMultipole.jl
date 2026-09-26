# Reference

## API

The following functions are the primary user-facing API of the `FastMultipole` package.

```@docs
fmm!
tune_fmm
FastMultipole.FmmPlan
RadixFMMCache
CUDARadixLifecycleOptions
DeviceResidentRadixState
recenter!
Residency
HostResident
DeviceResident
SingularSource
SingularDipole
SingularVortex
SingularSourceVortex
RegularizedVortex
PartitionedVortex
TwoPassVortex
SourceFilamentKernel
DipoleFilamentKernel
VortexFilamentKernel
FastMultipole.element_strength_dims
direct!(::Tuple)
DirectConditioningRule
SelfPairs
PairSet
AllPairs
FastMultipole.applies
```

## Compatibility Functions

The following functions must be overloaded by the user to interface their code with the `FastMultipole` package.

```@docs
FastMultipole.body_type
FastMultipole.direct_kernel
FastMultipole.residency
FastMultipole.device_backend
FastMultipole.source_to_buffer!
FastMultipole.buffer_to_target!
FastMultipole.source_system_to_buffer!
FastMultipole.data_per_body
FastMultipole.source_revision
FastMultipole.get_position
FastMultipole.metadata_per_body
FastMultipole.metadata_to_buffer!
FastMultipole.previous_potential_metadata_index
FastMultipole.previous_gradient_metadata_index
FastMultipole.strength_dims
FastMultipole.get_normal
FastMultipole.get_n_bodies
FastMultipole.body_to_multipole!
FastMultipole.direct!(::Any, ::Any, ::Any, ::Any, ::Any, ::Any)
FastMultipole.buffer_to_target_system!
FastMultipole.has_vector_potential
```

## Data Structures

The following data structures are used by the `FastMultipole` package.

```@docs
DerivativesSwitch
FastMultipole.Branch
FastMultipole.Tree
FastMultipole.ProbeSystem
FastMultipole.Cache
```

## Additional Functions

The following functions are used internally by the `FastMultipole` package, but may be useful to understand for advanced users.

```@docs
FastMultipole.allocate_buffers
FastMultipole.allocate_small_buffers
FastMultipole.metadata_range
FastMultipole.metadata_index
FastMultipole.tree_carried_range
FastMultipole.scalar_potential_index
FastMultipole.gradient_range
FastMultipole.hessian_range
FastMultipole.third_derivative_range
FastMultipole.ThirdDerivativeTensor
FastMultipole.packed_data
FastMultipole.dense
FastMultipole.get_third_derivative
FastMultipole.set_third_derivative!
FastMultipole.supports_third_derivative
FastMultipole.standard_output_range
FastMultipole.extra_output_range
FastMultipole.output_range
FastMultipole.get_extra_output
FastMultipole.set_extra_output!
FastMultipole.extra_output_view
```

## Body and Error Types

Dispatch tags for body data, element kernels, and error estimates.

```@docs
Position
Radius
ScalarPotential
Gradient
Hessian
Vertex
Normal
Strength
Vortex
Source
Dipole
SourceDipole
SourceVortex
Point
Filament
Panel
PowerAbsolutePotential
PowerAbsoluteGradient
RotatedCoefficientsAbsoluteGradient
Barba
SelfTuning
initialize_expansion
initialize_harmonics
multipole_error
local_error
```

## Radix Grid and Tree

```@docs
RadixGrid
DeviceRadixGrid
RadixSortBackend
HostRadixSort
DeviceRadixSort
AutoRadixSort
radix_grid
unsorted_index_2_sorted_index
sorted_index_2_unsorted_index
```

## Stencils and Interaction Lists

```@docs
ConstantPStencilConfig
RadixSeparationPolicy
ParentNeighborM2L
ConstantPAnalyticStencil
HierarchicalRigidStencil
classic_fmm_stencil
rigid_stencil_epsilon
RadixTraversalStrategy
RigidHierarchicalTables
RadixLevelOccupancy
RigidImplicitStencil
SparseOffsetIntersection
BlockedOccupancyBitsets
LazyMaterializedBatches
RadixM2LBatch
RadixInteractionList
constant_p_stencil_bound
constant_p_stencil_accepts
accepted_radix_stencil
foreach_radix_m2l_pair
foreach_radix_m2l_route
foreach_radix_direct_pair
build_radix_interaction_list
RadixRouteSelection
build_interaction_lists
InteractionList
```

## Resident Lifecycle Strategies and Options

```@docs
CUDARadixTransferCounters
AbstractResidentM2MStrategy
DenseTranslationM2M
SharedRotationM2M
AbstractResidentM2LStrategy
DenseTranslationM2L
SharedRotationM2L
ConcatenatedFixedZM2L
PrecomputedFactoredYM2L
RadixDeviceUnavailable
host_resident_radix_grid
host_radix_state
run_host_radix_lifecycle!
finalize_radix_output!
update_radix_state!
```

## Radix Settings

The public setting surface exposes only settings whose backing `Ref` is loaded.
The following settings are always defined:

| name | default | lock | availability |
|---|---:|---|---|
| `:RADIX_KA_LIFECYCLE` | `false` | runtime | Uniform device lifecycle; enabling it requires the KernelAbstractions extension |
| `:RADIX_DIRECT_ARM` | `false` | runtime | KernelAbstractions device path only |
| `:KA_WORKGROUP` | `0` (backend choice) | runtime | KernelAbstractions launches |
| `:CUDA_NEARFIELD_GH_MODE` | `:fp32` | construction | Regularized host and device nearfield |
| `:FACTORED_Y_GEMM_MIN_COLS` | `16` | runtime | Host factored-y operator |
| `:FACTORED_Y_GEMM_MIN_DIM` | `1` | runtime | Host factored-y operator |
| `:PRECOMPUTED_Y_GEMM_MIN_COLS` | `16` | runtime | Host precomputed-y operator |

The registry also recognizes backend-defined names. They are absent from
`radix_settings()` and `radix_setting`/`set_radix_setting!` throw for them until
a backend defines the corresponding setting:

- construction-locked: `:RADIX_CUDA_COUNTING_SORT`,
  `:RADIX_CUDA_COUNTING_SORT_MAX_ELL`, `:CUDA_NEARFIELD_BINNING`,
  `:CUDA_NEARFIELD_SHAPE`, `:CUDA_NEARFIELD_FUSED_MIN_BODIES`,
  `:CUDA_TWOPASS_PASS2_QUEUED`, `:CUDA_TWOPASS_TARGET_AABB_PRUNE`,
  `:CUDA_NEARFIELD_PAIR_AABB`, `:CUDA_SYMMETRIC_NEARFIELD`,
  `:DIRECT_CUDA_MAX_BLOCKS`, `:FACTORED_CUDA_CHUNK`,
  `:PRECOMPUTED_CUDA_CHUNK`, `:DENSE_CUDA_WHOLE_PASS`,
  `:DENSE_CUDA_CHUNK`, `:DENSE_CUDA_FUSED_MAX_BLOCKS`,
  `:DENSE_CUDA_TILED`, `:DENSE_CUDA_TILED_MIN_ROUTES`,
  `:DENSE_CUDA_TILED_THREADS`, `:DENSE_CUDA_TILED_MAX_BLOCKS`,
  `:DENSE_CUDA_TENSOR_FORMAT`, and `:CUDA_OVERLAP_NEARFIELD`;
- runtime: `:CUDA_NEARFIELD_SUBSORT`,
  `:SYMMETRIC_CUDA_MAX_CELL_BODIES`, `:FACTORED_CUDA_WHOLE_PASS`,
  `:PRECOMPUTED_CUDA_WHOLE_PASS`, `:DENSE_CUDA_FUSED`,
  `:CUDA_CACHED_WINDOWS`, `:KA_EXTRA_TARGETS_GRID`,
  `:KA_EXTRA_TARGETS_SYNC`, `:KA_EXTRA_TARGETS_CHECK`, and
  `:CUDA_GRAPH_LIFECYCLE`.

Set construction-locked values before building `RadixFMMCache`. The cache
snapshots them and a later device step throws if they drift. Runtime settings
are read at step entry and may change between steps. Use `radix_settings()` for
the values actually available in the current process and
`radix_setting_lock(name)` for their lock class.

```@docs
radix_settings
radix_setting
set_radix_setting!
set_radix_settings!
radix_setting_lock
snapshot_locked_radix_settings
verify_locked_radix_settings
```

## Adaptive Tree Policy

```@docs
AdaptiveTreePolicy
AdaptiveRadixTree
AdaptiveInteractionLists
update_adaptive_tree!
build_adaptive_interaction_lists!
adaptive_is_leaf
adaptive_node_range
```

## Transforms

```@docs
transform_tree!
transform_plan!
transform_solver!
```

## Operator Bases and Buffers

```@docs
AbstractOperatorBasis
CompressedComplexBasis
RealSolidHarmonicBasis
OperatorOrders
OperatorBasisInfo
OperatorInvariantCache
OperatorScratch
ThreadedOperatorScratch
FlatCoefficientBuffer
real_basis_index
complex_to_real_basis!
real_to_complex_basis!
AbstractM2LOperator
MaterializedYRotationM2L
FactoredRotationM2L
M2LOperatorScratch
AbstractM2MOperator
MaterializedYRotationM2M
FactoredRotationM2M
M2MOperatorScratch
AbstractL2LOperator
MaterializedYRotationL2L
FactoredRotationL2L
L2LOperatorScratch
```

## Rectangular and Direct Kernels

```@docs
AbstractDirectKernel
SourcePanelKernel
DipolePanelKernel
SourceDipolePanelKernel
VortexSheetPanelKernel
AbstractRectangularKernel
RectangularGaussianErfVortex
RectangularPanelInfluence
direct_rectangular!
rect_source_rows
rect_output_rows
```

## Tree Roles and Nearfield Execution

```@docs
TreeRole
SourceTree
TargetTree
NearfieldExecution
HostNearfield
DeviceNearfield
source_to_buffer
output_view
```

## Nearfield Cache

```@docs
NearfieldInfluenceCache
nearfield_matvec!
build_nearfield_cache!
estimate_nearfield_cache
NearfieldCacheDonor
retarget_nearfield_cache
assemble_influence_block!
overrides_block_assembly
```

## Solvers and Visualization

```@docs
FastGaussSeidel
JacobiPreconditioner
visualize
```

## Telemetry and Device Operators

```@docs
ka_m2m_operator_batch!
ka_m2l_operator_batch!
ka_l2l_operator_batch!
```
