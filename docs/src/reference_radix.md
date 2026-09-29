# Reference: radix and device path

## Radix Grid and Tree

```@docs
FastMultipole.RadixGrid
FastMultipole.DeviceRadixGrid
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
FastMultipole.RigidHierarchicalTables
FastMultipole.RadixLevelOccupancy
FastMultipole.RadixM2LBatch
FastMultipole.RadixInteractionList
FastMultipole.constant_p_stencil_bound
FastMultipole.build_radix_interaction_list
FastMultipole.RadixRouteSelection
build_interaction_lists
InteractionList
```


## Resident Lifecycle Strategies and Options

```@docs
RadixTransferCounters
AbstractResidentM2LStrategy
DenseTranslationM2L
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

Every setting is defined in the host package (no backend needs to be loaded to
read or write it); the device-path settings take effect once a
KernelAbstractions backend runs the cache.

| name | default | lock | availability |
|---|---:|---|---|
| `:RADIX_DIRECT_ARM` | `false` | runtime | KernelAbstractions device path only |
| `:CUDA_NEARFIELD_GH_MODE` | `:shipped` | runtime | Regularized host nearfield (device kernels ignore it) |
| `:FACTORED_Y_GEMM_MIN_DIM` | `1` | runtime | Host factored-y operator |
| `:PRECOMPUTED_Y_GEMM_MIN_COLS` | `16` | runtime | Host precomputed-y operator |

Runtime settings are read at step entry and may change between steps (every
current setting is runtime). A construction-locked setting would have to be set
before building `RadixFMMCache`: the cache snapshots such settings and a later
device step throws if they drift. Use `radix_settings()` for
the current values.

```@docs
radix_settings
radix_setting
set_radix_setting!
set_radix_settings!
FastMultipole.snapshot_locked_radix_settings
FastMultipole.verify_locked_radix_settings
```


## Tree Roles

```@docs
FastMultipole.TreeRole
FastMultipole.SourceTree
FastMultipole.TargetTree
source_to_buffer
output_view
```

