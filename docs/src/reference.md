# Reference

## API

The following functions are the primary user-facing API of the `FastMultipole` package.

```@docs
fmm!
tune_fmm
direct!(::Tuple)
```

## Compatibility Functions

The following functions must be overloaded by the user to interface their code with the `FastMultipole` package.

```@docs
FastMultipole.source_system_to_buffer!
FastMultipole.data_per_body
FastMultipole.get_position
FastMultipole.get_previous_influence
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
```