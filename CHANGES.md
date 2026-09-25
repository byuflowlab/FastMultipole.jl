# Change Log

## v2.3.0

- GPU execution through a KernelAbstractions package extension (CUDA and Metal): the
  device-resident radix FMM lifecycle (`RadixFMMCache(...; device=true)`) for
  the four point body types (`Point{Source}`, `Point{Dipole}`, `Point{Vortex}`,
  `Point{SourceVortex}`) and the three straight filament types, with the singular and regularized vortex nearfield kernels
  and the subfilter-scale pass. The native CUDA lifecycle is removed.
- Radix-grid FMM path on the host: `RadixFMMCache`, hierarchical stencils, dense
  and concatenated M2L operators, `FmmPlan` for repeated calls on frozen geometry,
  nearfield influence cache, extra sources and targets.
- New kernels: `SingularVortex`, `SingularDipole`, `SingularSourceVortex`,
  `RegularizedVortex`, `PartitionedVortex`, `TwoPassVortex`, `RectangularGaussianErfVortex`,
  and the filament nearfield kernels `SourceFilamentKernel`, `DipoleFilamentKernel`, `VortexFilamentKernel`.
- Minimum Julia version 1.11.

## v0.1.0 - 2024 August

Initial release.


