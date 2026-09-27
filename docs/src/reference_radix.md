# Reference: radix and device path

## Radix Grid and Tree

```@docs
FastMultipole.RadixGrid
FastMultipole.DeviceRadixGrid
FastMultipole.RadixSortBackend
FastMultipole.HostRadixSort
FastMultipole.DeviceRadixSort
FastMultipole.AutoRadixSort
FastMultipole.radix_grid
unsorted_index_2_sorted_index
sorted_index_2_unsorted_index
```


## Stencils and Interaction Lists

```@docs
FastMultipole.ConstantPStencilConfig
FastMultipole.RadixSeparationPolicy
FastMultipole.ParentNeighborM2L
ConstantPAnalyticStencil
HierarchicalRigidStencil
classic_fmm_stencil
FastMultipole.rigid_stencil_epsilon
FastMultipole.RadixTraversalStrategy
FastMultipole.RigidHierarchicalTables
FastMultipole.RadixLevelOccupancy
FastMultipole.RigidImplicitStencil
FastMultipole.SparseOffsetIntersection
FastMultipole.BlockedOccupancyBitsets
FastMultipole.LazyMaterializedBatches
FastMultipole.RadixM2LBatch
FastMultipole.RadixInteractionList
FastMultipole.constant_p_stencil_bound
FastMultipole.constant_p_stencil_accepts
FastMultipole.accepted_radix_stencil
FastMultipole.foreach_radix_m2l_pair
FastMultipole.foreach_radix_m2l_route
FastMultipole.foreach_radix_direct_pair
FastMultipole.build_radix_interaction_list
FastMultipole.RadixRouteSelection
build_interaction_lists
InteractionList
```


## Resident Lifecycle Strategies and Options

```@docs
RadixTransferCounters
AbstractResidentM2MStrategy
DenseTranslationM2M
SharedRotationM2M
AbstractResidentM2LStrategy
DenseTranslationM2L
SharedRotationM2L
ConcatenatedFixedZM2L
PrecomputedFactoredYM2L
RadixDeviceUnavailable
FastMultipole.host_resident_radix_grid
FastMultipole.host_radix_state
FastMultipole.run_host_radix_lifecycle!
FastMultipole.finalize_radix_output!
update_radix_state!
radix_nearfield
```


## Radix Settings

The public setting surface exposes only settings whose backing `Ref` is loaded.
The following settings are always defined:

| name | default | lock | availability |
|---|---:|---|---|
| `:RADIX_DIRECT_ARM` | `false` | runtime | KernelAbstractions device path only |
| `:KA_WORKGROUP` | `0` (backend choice) | runtime | KernelAbstractions launches |
| `:CUDA_NEARFIELD_GH_MODE` | `:fp32` | construction | Regularized host and device nearfield |
| `:FACTORED_Y_GEMM_MIN_COLS` | `16` | runtime | Host factored-y operator |
| `:FACTORED_Y_GEMM_MIN_DIM` | `1` | runtime | Host factored-y operator |
| `:PRECOMPUTED_Y_GEMM_MIN_COLS` | `16` | runtime | Host precomputed-y operator |

The registry also recognizes backend-defined names. They are absent from
`radix_settings()` and `radix_setting`/`set_radix_setting!` throw for them until
a backend defines the corresponding setting:

- construction-locked: `:RADIX_CUDA_COUNTING_SORT`, `:RADIX_CUDA_COUNTING_SORT_MAX_ELL`;
- runtime: `:CUDA_NEARFIELD_SUBSORT`, `:SYMMETRIC_CUDA_MAX_CELL_BODIES`, `:CUDA_CACHED_WINDOWS`
  and `:KA_EXTRA_TARGETS_GRID`.

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
FastMultipole.radix_setting_lock
FastMultipole.snapshot_locked_radix_settings
FastMultipole.verify_locked_radix_settings
```


## Adaptive Tree Policy

```@docs
AdaptiveTreePolicy
FastMultipole.AdaptiveRadixTree
FastMultipole.AdaptiveInteractionLists
FastMultipole.update_adaptive_tree!
FastMultipole.build_adaptive_interaction_lists!
FastMultipole.adaptive_is_leaf
FastMultipole.adaptive_node_range
```


## Tree Roles and Nearfield Execution

```@docs
FastMultipole.TreeRole
FastMultipole.SourceTree
FastMultipole.TargetTree
FastMultipole.NearfieldExecution
FastMultipole.HostNearfield
FastMultipole.DeviceNearfield
source_to_buffer
output_view
```

