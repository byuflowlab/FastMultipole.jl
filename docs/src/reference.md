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
FastMultipole.standard_output_range
FastMultipole.extra_output_range
FastMultipole.output_range
FastMultipole.get_extra_output
FastMultipole.set_extra_output!
FastMultipole.extra_output_view
```
