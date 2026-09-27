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
FastMultipole.RealSolidHarmonicBasis
FastMultipole.OperatorOrders
FastMultipole.OperatorBasisInfo
FastMultipole.OperatorInvariantCache
FastMultipole.OperatorScratch
FastMultipole.ThreadedOperatorScratch
FastMultipole.FlatCoefficientBuffer
FastMultipole.real_basis_index
FastMultipole.complex_to_real_basis!
FastMultipole.real_to_complex_basis!
FastMultipole.AbstractM2LOperator
FastMultipole.MaterializedYRotationM2L
FastMultipole.FactoredRotationM2L
FastMultipole.M2LOperatorScratch
FastMultipole.AbstractM2MOperator
FastMultipole.MaterializedYRotationM2M
FastMultipole.FactoredRotationM2M
FastMultipole.M2MOperatorScratch
FastMultipole.AbstractL2LOperator
FastMultipole.MaterializedYRotationL2L
FastMultipole.FactoredRotationL2L
FastMultipole.L2LOperatorScratch
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
FastMultipole.rect_source_rows
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

