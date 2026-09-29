# Reference: kernels, operators and transforms

## Transforms

```@docs
transform_tree!
transform_plan!
transform_solver!
```


## Operator Bases and Buffers

```@docs
FastMultipole.AbstractOperatorBasis
FastMultipole.CompressedComplexBasis
FastMultipole.OperatorOrders
FastMultipole.OperatorBasisInfo
FastMultipole.OperatorInvariantCache
FastMultipole.OperatorScratch
FastMultipole.FlatCoefficientBuffer
FastMultipole.AbstractM2LOperator
FastMultipole.MaterializedYRotationM2L
FastMultipole.FactoredRotationM2L
FastMultipole.M2LOperatorScratch
```


## Rectangular and Direct Kernels

```@docs
AbstractDirectKernel
SourcePanelKernel
DipolePanelKernel
SourceDipolePanelKernel
VortexSheetPanelKernel
AbstractRectangularKernel
direct_rectangular!
FastMultipole.rect_source_rows
FastMultipole.rect_pair
FastMultipole.rect_has_potential
FastMultipole.rect_check_sources
FastMultipole.rect_output_rows
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

