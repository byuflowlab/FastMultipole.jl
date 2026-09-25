# Change Log

## v2.3.0

- GPU execution through a KernelAbstractions package extension (CUDA and Metal): the
  device-resident radix FMM lifecycle (`RadixFMMCache(...; device=true)`) for
  the four point body types (`Point{Source}`, `Point{Dipole}`, `Point{Vortex}`,
  `Point{SourceVortex}`), the three straight filament types and the four planar triangular panel
  types (`Panel{3,Source}`, `Panel{3,Dipole}`, `Panel{3,SourceDipole}`, `Panel{3,Vortex}`), with the singular and regularized vortex nearfield kernels
  and the subfilter-scale pass. The native CUDA lifecycle is removed.
- Radix-grid FMM path on the host: `RadixFMMCache`, hierarchical stencils, dense
  and concatenated M2L operators, `FmmPlan` for repeated calls on frozen geometry,
  nearfield influence cache, extra sources and targets.
- New kernels: `SingularVortex`, `SingularDipole`, `SingularSourceVortex`,
  `RegularizedVortex`, `PartitionedVortex`, `TwoPassVortex`, `RectangularGaussianErfVortex`,
  the filament nearfield kernels `SourceFilamentKernel`, `DipoleFilamentKernel`, `VortexFilamentKernel`
  and the panel nearfield kernels `SourcePanelKernel`, `DipolePanelKernel`, `SourceDipolePanelKernel`, `VortexSheetPanelKernel`.
- Known limit: one GPU and one stream per cache; multi-GPU and stream overlap are future work.
- `RectangularPanelInfluence` accepts Float32 (its singularity guards and the LineGauss
  series/axis crossovers scale with the precision; LineGauss in Float32 tracks Float64 to
  about 3e-5 in velocity and 2e-3 in gradient of the field scale near the segment axis); `VortexSheetPanelKernel(; order=3)`
  selects a 13-point degree-7 Dunavant rule. `direct_rectangular!` runs on device arrays
  through the KernelAbstractions extension (all-pairs, one work-item per target).
- Added the validated radix settings API: `radix_settings`, `radix_setting`,
  `set_radix_setting!`, and `set_radix_settings!`. Construction-locked settings
  are snapshotted by a cache and checked at each device step; runtime settings
  may change between steps.
- Added the opt-in `AdaptiveTreePolicy` and adaptive radix-tree lifecycle.
- Added `transform_tree!`, `transform_plan!`, and `transform_solver!` for rigid
  motion of reusable trees, plans, and solvers, subject to their documented
  cache and output restrictions.
- Added the `source_revision(system)` compatibility trait for reusing unchanged
  extra-tree sources; the default `nothing` disables reuse.
- GPU tests can be selected with `FASTMULTIPOLE_GPU_TESTS`; NVIDIA runs require
  `FASTMULTIPOLE_GPU_TEST_PROJECT` to name a CUDA-enabled Julia project.
- Minimum Julia version 1.11.

## v0.1.0 - 2024 August

Initial release.

