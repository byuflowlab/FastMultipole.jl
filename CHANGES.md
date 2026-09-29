# Change Log

## v2.3.0

- GPU execution through a KernelAbstractions package extension (any KA backend; CUDA and Metal tested, AMDGPU and oneAPI untested and may need modification): the
  device-resident radix FMM lifecycle (`RadixFMMCache(...; device=true)`) for
  the four point body types (`Point{Source}`, `Point{Dipole}`, `Point{Vortex}`,
  `Point{SourceVortex}`), the three straight filament types and the four planar triangular panel
  types (`Panel{3,Source}`, `Panel{3,Dipole}`, `Panel{3,SourceDipole}`, `Panel{3,Vortex}`), with the singular and regularized vortex nearfield kernels
  and a consumer near-field pass hook (`fmm!(...; nearfield_pass)`, `radix_nearfield`). The native CUDA lifecycle is removed.
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

- Added `transform_tree!`, `transform_plan!`, and `transform_solver!` for rigid
  motion of reusable trees, plans, and solvers, subject to their documented
  cache and output restrictions.
- Added the `source_revision(system)` compatibility trait for reusing unchanged
  extra-tree sources; the default `nothing` disables reuse.
- GPU tests can be selected with `FASTMULTIPOLE_GPU_TESTS`; NVIDIA runs require
  `FASTMULTIPOLE_GPU_TEST_PROJECT` to name a CUDA-enabled Julia project.
- Minimum Julia version 1.10 (the LTS; package extensions with weak dependencies).

### Breaking changes and migration

- Target buffer layout: the output rows follow the `DerivativesSwitch`
  (`scalar_potential_index`, `gradient_range`, `hessian_range`,
  `third_derivative_range`, `extra_output_range`, `metadata_index`). The
  switchless accessors (`get_gradient(buffer, i)`, `set_hessian!(buffer, i, h)`,
  and the scalar-potential and third-derivative forms) now throw an
  `ArgumentError` naming the switch-aware form; code that indexed rows directly
  (`buffer[5:7, i]`) must migrate by hand. `get_previous_influence` is removed;
  carry prior-step values in metadata rows (`metadata_per_body`,
  `metadata_to_buffer!`).
- Third derivatives: `third_derivative=true` on the host path, packed
  `(xx,xy,xz,yy,yz,zz)` per component in `third_derivative_range`
  (`ThirdDerivativeTensor`, `packed_data`, `dense`); not on the device path.
- New consumer traits: `residency`, `device_backend`, `body_type`,
  `direct_kernel`, `supports_third_derivative`, `source_revision`.
- Removed exports: `SingleBranch`, `MultiBranch`, `SingleTree`, `MultiTree`,
  `Body`, `buffer_element`, `direct_gpu!`, `unsort!`, `resort!`, `error`.
- Consumer physics left FastMultipole: the subfilter-scale pass, its
  core-scaling derivatives, the ζ reconstruction and the SFS repass
  (`fmm!(...; sfs, sfs_dsigma)`, `RadixFMMCache(...; sfs, sfs_transposed,
  sfs_active_row)`, `sfs_to_target!`, `sfs_dsigma_to_target!`,
  `zeta_to_target!`, `output_from_target!`, `radix_zeta!`, `radix_sfs_repass!`)
  are gone; a consumer runs its own pass through `nearfield_pass` and reads
  `radix_nearfield(cache)` (FLOWVPM does).
- Review cuts (2026-09-28): code with no caller in production or in the
  known consumers (FLOWVPM, FLOWUnsteadyCore, LiftingLines, VortexLattice,
  FLOWPanel) was left out of this release. The full list, with what replaced
  each item, is in the PR. In summary: the adaptive radix-tree lifecycle
  (host and KernelAbstractions; `AdaptiveTreePolicy` is gone); the flat KA
  route generator and `ka_fmm!`; the operator-refactor alternatives that
  production never selects (shared-rotation batching and its
  `SharedRotationM2L`/`SharedRotationM2M`/`DenseTranslationM2M` tags, the flat
  M2M/L2L operator pipelines and their tags, the factored flat rotation
  pipeline, the ±π/2 y-swap tables, the legacy `[2,2,nh]` kernels,
  `RealSolidHarmonicBasis` and its conversions, `ThreadedOperatorScratch`);
  the standalone radix-grid and constant-P interaction-list API (sort
  backends, traversal-strategy types, `radix_grid`, `foreach_radix_*`,
  `accepted_radix_stencil`, `constant_p_stencil_accepts`); `tune_fmm_perturb`
  and the `tune_nearfield_cache` route with the third `tune_fmm` return;
  `NearfieldExecution`/`HostNearfield`/`DeviceNearfield` and
  `nearfield_device!` (`fmm!(...; nearfield_device=true)` now throws on the
  legacy octree path); the device two-pass helpers and the `:lut` g/h mode; the
  uncached KA window generator and the KA grid path for extra targets;
  `radix_setting_lock` and the settings `:KA_WORKGROUP`,
  `:FACTORED_Y_GEMM_MIN_COLS`, `:CUDA_CACHED_WINDOWS`,
  `:KA_EXTRA_TARGETS_GRID`, `:RADIX_CUDA_COUNTING_SORT`,
  `:RADIX_CUDA_COUNTING_SORT_MAX_ELL`, `:CUDA_NEARFIELD_SUBSORT`,
  `:SYMMETRIC_CUDA_MAX_CELL_BODIES` (each fixed at its former default);
  `RadixLifecycleOptions.m2m_strategy` (the `m2l_strategy` default is now
  `ConcatenatedFixedZM2L`); unread fields of `DeviceHierarchicalM2LContext`,
  `DeviceResidentRadixState`, `ResidentOperatorWorkspace`, `RadixTransferCounters`
  (`route_uploads`, `operator_uploads`) and `DenseTranslationM2L`
  (`cuda_headroom_bytes`); the positional `DerivativesSwitch(ps, gs, hs, ts)`,
  the 4-argument `ProbeSystemStatic`/`ProbeSystemArray` constructors, the
  single-system `Tree(system, ::TreeRole, ...)` method, `phi_physical_view`,
  and the `CUDARadixTransferCounters` alias.
- Fixes from the same review: `update_radix_state!` with a tuple of systems on
  a device cache ran the host refresh (method shadowing); automatic option
  selection could hand a device cache `DenseTranslationM2L`, which the KA build
  rejects (a device cache now always builds `ConcatenatedFixedZM2L`);
  `RectangularPanelInfluence` accepted a four-vertex combined source+ring panel
  whose ring was evaluated as a triangle; `ProbeSystem` third derivatives now
  accumulate like the other outputs; `direct!(...; nearfield_cache)` rejects
  derivatives switches that differ from the cache's; `_assert_rigid_rotation`
  scales its tolerance with the float type; a tree-carried extra source outside
  the grid box is summed directly instead of being clamped into an edge cell;
  the legacy multithreaded `fmm!` no longer hits an undefined `t_m2l` with
  `tune=true, horizontal_pass=false`; the KA step syncs before its window-count
  read and runs its geometry gate once; the KA element scratch no longer leaks
  when a state is replaced.

## v0.1.0 - 2024 August

Initial release.

