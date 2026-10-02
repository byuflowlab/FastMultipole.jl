# Change Log

## v3.0.0

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
  `RegularizedVortex`, `PartitionedVortex`, `TwoPassVortex`,
  the filament nearfield kernels `SourceFilamentKernel`, `DipoleFilamentKernel`, `VortexFilamentKernel`
  and the panel nearfield kernels `SourcePanelKernel`, `DipolePanelKernel`, `SourceDipolePanelKernel`, `VortexSheetPanelKernel`.
- Known limit: one GPU and one stream per cache; multi-GPU and stream overlap are future work.
- `VortexSheetPanelKernel(; order=3)` selects a 13-point degree-7 Dunavant rule.
- `direct_rectangular!(out, targets, kernel, sources; gradient, scalar_potential)`:
  brute-force evaluation from a source set at a distinct target set. The pair
  math comes from the consumer: a kernel type `<: AbstractRectangularKernel`
  defines `rect_source_rows` and `rect_pair` (optionally `rect_has_potential`,
  `rect_check_sources`). The same `rect_pair` runs in the threaded host loop and,
  through the KernelAbstractions extension, on device arrays (one work-item per
  target), with no device code on the consumer side.
- Added the validated radix settings API: `radix_settings`, `radix_setting`,
  `set_radix_setting!`, and `set_radix_settings!`. Construction-locked settings
  are snapshotted by a cache and checked at each device step; runtime settings
  may change between steps. `:CUDA_NEARFIELD_GH_MODE` (runtime, host
  regularized nearfield) defaults to `:shipped`, the full-precision g/h and U/J
  evaluation; `:fp32` (g/h and pair U/J in Float32 for Float64 runs, Float64
  accumulation) and the reduced-series modes are opt-in.
- `RadixFMMCache` without explicit `options` defaults to Float64 on the host at
  every expansion order; a device cache defaults to Float32 only at
  `expansion_order <= 3`.
- Known limitation: a Float32 `RadixFMMCache` throws an `ArgumentError` at
  construction when a physically small box at a high expansion order would
  overflow the unnormalized M2L coefficients; use Float64 or scale the
  geometry toward unit size.

- Added `transform_tree!`, `transform_plan!`, and `transform_solver!` for rigid
  motion of reusable trees, plans, and solvers, subject to their documented
  cache and output restrictions. A transformed `FastGaussSeidel` solves with
  any output switches (the default `gradient=true` included): its dense
  matrices hold the consumer's scalar influence, which is exact under rigid
  motion when that influence is rigid-invariant.
- Added the `source_revision(system)` compatibility trait for reusing unchanged
  extra-tree sources; the default `nothing` disables reuse. It is not exported:
  overload `FastMultipole.source_revision`.
- GPU tests can be selected with `FASTMULTIPOLE_GPU_TESTS`; NVIDIA runs require
  `FASTMULTIPOLE_GPU_TEST_PROJECT` to name a CUDA-enabled Julia project.
- Minimum Julia version 1.10 (the LTS; package extensions with weak dependencies).

### Breaking changes and migration

- Target buffer layout: the output rows follow the `DerivativesSwitch`
  (`scalar_potential_index`, `gradient_range`, `hessian_range`,
  `third_derivative_range`, `extra_output_range`, `metadata_index`). Every
  switchless get/set form (scalar potential, gradient, hessian, third
  derivative; e.g. `get_gradient(buffer, i)`, `set_hessian!(buffer, i, h)`) now
  throws an `ArgumentError` naming the switch-aware form; v2.3.0 read and wrote
  the legacy rows `4`, `5:7`, `8:16` silently. Code that indexed rows directly
  (`buffer[5:7, i]`) must migrate by hand. `get_previous_influence` is removed;
  carry prior-step values in metadata rows (`metadata_per_body`,
  `metadata_to_buffer!`).
- Third derivatives: `third_derivative=true` on the host path, packed
  `(xx,xy,xz,yy,yz,zz)` per component in `third_derivative_range`
  (`ThirdDerivativeTensor`, `packed_data`, `dense`); not on the device path.
- New consumer traits: `residency`, `body_type`, `direct_kernel`,
  `supports_third_derivative` (exported), and `device_backend`,
  `source_revision` (not exported; overload them as
  `FastMultipole.device_backend` and `FastMultipole.source_revision`).
- `DerivativesSwitch` gained a sixth type parameter `TS` (third derivatives):
  `DerivativesSwitch{PS,GS,HS,NO,NM,TS}`. Code that constructs
  `DerivativesSwitch{PS,GS,HS,NO,NM}()` directly breaks; use the
  `DerivativesSwitch(...)` constructors or add the parameter.
- `FastGaussSeidel`: the target tree replays the source tree's topology, and
  the m2l/direct interaction lists are re-sorted, so the iteration order and
  the iterates shift relative to v2.3.0. The default `cache_leaf_lu=true`
  keeps a factorized copy of the self-matrix data (about twice the self-matrix
  memory; pass `cache_leaf_lu=false` to opt out). New keywords: `sweep_order`
  (`:lexicographic` default, or `:colored`) on the constructor and `callback`
  (called as `callback(iteration, residual)`) on `solve!`.
- Removed exports: `SingleBranch`, `MultiBranch`, `SingleTree`, `MultiTree`,
  `Body`, `buffer_element`, `direct_gpu!`, `unsort!`, `resort!`, `error`.
- Consumer physics left FastMultipole: the subfilter-scale pass, its
  core-scaling derivatives, the ζ reconstruction and the SFS repass
  (`fmm!(...; sfs, sfs_dsigma)`, `RadixFMMCache(...; sfs, sfs_transposed,
  sfs_active_row)`, `sfs_to_target!`, `sfs_dsigma_to_target!`,
  `zeta_to_target!`, `output_from_target!`, `radix_zeta!`, `radix_sfs_repass!`)
  are gone; a consumer runs its own pass through `nearfield_pass` and reads
  `radix_nearfield(cache)` (FLOWVPM does).
- Consumer kernels left FastMultipole: the gaussianerf particle kernel
  `RectangularGaussianErfVortex` is FLOWVPM's and the panel-element kernel
  `RectangularPanelInfluence` is FLOWPanel's, each implementing `rect_pair`.
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
  `ProbeSystem` third derivatives now
  accumulate like the other outputs; `direct!(...; nearfield_cache)` rejects
  derivatives switches that differ from the cache's; `_assert_rigid_rotation`
  scales its tolerance with the float type; a tree-carried extra source outside
  the grid box is summed directly instead of being clamped into an edge cell;
  the legacy multithreaded `fmm!` no longer hits an undefined `t_m2l` with
  `tune=true, horizontal_pass=false`; the KA step syncs before its window-count
  read and runs its geometry gate once; the KA element scratch no longer leaks
  when a state is replaced.
- Fixes from a second review: on a rectangular `RadixFMMCache` box a body on a
  short axis' upper face got an out-of-range cell coordinate and overran the
  cell/node capacities (out-of-bounds writes); Float32 M2L at high expansion
  order or in a small box produced NaN silently (every M2L plan now checks its
  range at construction and throws, and the z-translation factors are formed in
  Float64); the planned `fmm!(targets, sources, plan)` docstring was attached to
  a helper; repeated `transform_tree!` compounded the branch boxes (boxes are
  now the rotated build-time boxes); Float32 `evaluate_local` hessian and third
  derivatives came back Float64; `direct_rectangular!` rejected host arrays
  wrapped more than one level deep (e.g. `reshape(view(...))`); an explicit options direct kernel that
  equalled the body type's default did not raise the conflict with a
  `direct_kernel(system)` trait; `VortexFilamentKernel(; family)` and
  `VortexSheetPanelKernel(; order)` accepted invalid values; user keywords to
  `tune_fmm` could turn its preallocation call into a full evaluation; extra
  target systems on the host radix path saw zero metadata rows;
  `transform_solver!` left the source buffers at the old pose for `solve!`'s
  first right-hand-side evaluation.

## v2.3.0 - 2026 May

- Added direct conditioning: `DirectConditioningRule(matcher, before!, after!)` temporarily transforms a source buffer for selected source-target system pairs during the direct (nearfield) pass. Matchers are `SelfPairs`, `AllPairs`, and `PairSet`; `applies` is exported. Pass `direct_conditioning=rule` (one rule or a tuple) to `fmm!` and `direct!`. `after!` callbacks run in reverse order inside `try`/`finally`.

## v2.2.0 - 2026 May

- Target buffers are now compact and carry optional metadata and extra output rows: `DerivativesSwitch` takes `extra_outputs` and `metadata` keywords, and `direct!` accepts `extra_outputs`. Metadata rows are copied from the target system, sorted with positions, and visible during nearfield interactions; extra output rows accumulate in the nearfield only (not in the farfield).
- New exported row-layout helpers: `scalar_potential_index`, `gradient_range`, `hessian_range`, `standard_output_range`, `extra_output_range`, `output_range`, `metadata_range`, `metadata_index`, `tree_carried_range`, `get_extra_output`, `set_extra_output!`, `extra_output_view`, `output_view`, and `source_to_buffer`/`source_to_buffer!`. Hard-coded target rows `4`, `5:7`, `8:16` are valid only when `metadata=0` and all preceding standard outputs are enabled; custom `direct!` and `buffer_to_target_system!` overloads should use these helpers.
- New compatibility functions `metadata_per_body` and `metadata_to_buffer!` for target systems that declare metadata.
- Fixed Cartesian-to-spherical conversion for bodies at or near the origin (guards the `acos` domain and the zero-radius case).

## v2.1.0 - 2026 May

- `FastGaussSeidel` now supports multiple body systems (multibody `solve!`). `solve!` takes `rlx` (relaxation), `reverse_pass`, `verbose`, and `final_update` keywords, and takes `scalar_potential`, `gradient`, `hessian` keywords in place of a `derivatives_switches` keyword. Adds `value_to_strength!(source_buffer, source_system, i_body, value, rlx)` to the compatibility interface.
- Added `JacobiPreconditioner`, a block Jacobi preconditioner (uniform grid cells, one LU factorization per cell); source and target systems must be the same.
- Added `fmm!(...; extra_farfield=true)` (default `false`) with an `extra_farfield!` hook, intended for semi-infinite panels whose strengths are tied to another system. Companion compatibility functions: `extra_target_data_per_body` and `extra_target_data_to_buffer!`.
- Added `numtype(system)` (default `eltype(system)`) to decouple a system's element type from the float type used for tree buffers; tree construction now uses it consistently.
- Tree subdivision stops once a branch radius falls below the largest body radius.
- `fmm!(system)` and `fmm!(target, source)` now accept `scalar_potential`, `gradient`, `hessian` keywords and build the `Cache` with matching `DerivativesSwitch`es; target buffer size follows the switch.
- Fixed the sign of the far-field (local-expansion) contribution to the scalar potential, and fixed multithreaded `direct!` indexing and load-balancing bugs.

## v2.0.4 - 2025 September

- Further multithreading fixes in tree construction and interaction-list building; the minimum body count for multithreaded paths (`MIN_BODIES`) was raised from 1000 to 10000.

## v2.0.3 - 2025 September

- Multithreading fix, and threading now handles small body counts (new minimum-size thresholds for multithreaded sorting and branching).

## v2.0.2 - 2025 August

- No code changes from v2.0.1 (identical git tree).

## v2.0.1 - 2025 August

- Multithreaded tree construction (sorting, shrink/recenter, target-to-buffer) and multithreaded interaction-list building; large reduction in allocations.
- New interaction list method `SelfTuningTargetStop`, now the default `interaction_list_method` of `fmm!` (was `SelfTuningTreeStop`).

## v2.0.0 - 2025 July

- Added relative-error tolerances with an absolute floor: `PowerRelativePotential`, `PowerRelativeGradient`, `RotatedCoefficientsRelativeGradient` take `(ε_rel, ε_abs=sqrt(eps()), BE=true)`. For relative methods, overload `get_previous_influence(system, i)` for your target system (default returns zero, which falls back to the absolute tolerance, with a warning).
- Quadrilateral-panel body-to-multipole now computes the panel normal from its vertices instead of calling `get_normal`.

### Breaking changes

- `ErrorMethod` hierarchy re-parameterized: `ErrorMethod{BE}`, with `AbsoluteErrorMethod{ε,BE}` and `RelativeErrorMethod{ε_rel,ε_abs,BE}` replacing `AbsoluteError`/`RelativeError`. `UnequalSpheres`, `UnequalBoxes`, `UniformUnequalSpheres`, `UniformUnequalBoxes`, `RotatedCoefficients` now carry a `BE` type parameter. `AbsoluteUpperBound`, `RelativeUpperBound`, and `ExpansionSwitch` are removed.
- Relative error types: the single tolerance `ε` became `(ε_rel, ε_abs)`, so `PowerRelativePotential{ε,BE}` is now `PowerRelativePotential{ε_rel,ε_abs,BE}`.
- `InteractionListMethod` is no longer parameterized and `SortByTarget`/`SortBySource` are removed: `Barba(SortByTarget())` and `SelfTuning(SortByTarget())` become `Barba()` and `SelfTuning()`.
- `Branch` fields consolidated: `source_center`/`target_center` -> `center`, `source_radius`/`target_radius` -> `radius`, `source_box`/`target_box` -> `box`, `max_influence` -> `min_potential` and `min_gradient`. Code constructing or reading `Branch` directly must change.

## v1.0.0 - 2025 June

- New interface based on per-system buffers. Replaces the `Base.getindex`/`setindex!` overloads for `Position`, `Radius`, `ScalarPotential`, `Velocity`, `VelocityGradient`, `Strength`, `Normal` with: `source_system_to_buffer!`, `data_per_body`, `get_position`, `strength_dims`, `get_normal`, `direct!(target_buffer, target_index, derivatives_switch, source_system, source_buffer, source_index)`, `buffer_to_target_system!`, `buffer_to_system_strength!`, `has_vector_potential`, and a rewritten `body_to_multipole!`. Lamb-Helmholtz is now chosen automatically from `has_vector_potential(source_systems)` instead of a `lamb_helmholtz` keyword.
- Outputs are now `scalar_potential`, `gradient`, `hessian` (new exports `Gradient`, `Hessian`); `DerivativesSwitch` is `{PS,GS,HS}`.
- Dynamic expansion order via `error_tolerance`: pass an `ErrorMethod` such as `PowerAbsolutePotential(1e-6)`, `PowerAbsoluteGradient`, or `RotatedCoefficientsAbsoluteGradient` (exported); `expansion_order` then acts as the maximum. `multipole_error` and `local_error` are exported.
- Automated tuning: `tune_fmm(target_systems, source_systems; ...)` returns tuned keyword arguments; `fmm!(...; tune=true)` and the `multipole_acceptance` keyword (default 0.5).
- `Cache` object to preallocate buffers across `fmm!` calls: `fmm!(systems, cache; ...)`. `fmm!` returns tuned optional arguments for reuse.
- `InteractionListMethod` choices `SelfTuning` and `Barba` exported; default method `SelfTuningTreeStop()`.
- New element types `Point`, `Filament`, `Panel` and kernels `Vortex`, `Source`, `Dipole`, `SourceDipole`, `SourceVortex`; a vortex-filament example is documented in `vortex_filament.md`.
- Added `FastGaussSeidel` solver and `solve!` for boundary-element-style linear systems with fast matrix-vector products.
- Added `leaf_size` per system (element-specific leaf sizes) and `shrink_recenter` (default `true`).
- Dependencies: `BSON` removed; `ForwardDiff` and `SpecialFunctions` get compat entries (`1.0.1`, `2.5.1`). Julia compat remains `1.6`.

### Breaking changes

- Source/target systems must implement the new buffer interface above; the `Base.getindex`/`setindex!`-based interface (`ScalarPotential`, `Velocity`, `VelocityGradient`, `Strength`, `Normal` indexing) no longer works.
- Exports removed: `Body`, `VectorPotential`, `Velocity`, `VelocityGradient`, `SortWrapper` (with `sortwrapper.jl`), `EqualSpheres`, `UnequalSpheres`, `UnequalBoxes`, `Dynamic`, and `ProbeSystem` (still available as `FastMultipole.ProbeSystem`).
- `fmm!` keyword changes: `velocity`/`velocity_gradient`/`vector_potential` replaced by `gradient`/`hessian`; `error_method`/`predict_error`/`expansion_order=Dynamic(...)` replaced by `error_tolerance`; `multipole_threshold` renamed `multipole_acceptance`; `leaf_size_source`/`leaf_size_target` now accept per-system vectors; `nearfield_user`, `gpu`, `save_tree_*` keywords removed from the main entry points. `source_shrink_recenter`/`target_shrink_recenter` are replaced by a single `shrink_recenter`, default now `true` (was `false` in v0.4.0).

## v0.4.0 - 2024 November

- Dynamic expansion order (`expansion_order=Dynamic(Pmax, rtol)`) driven by new error predictors: `error_method` keyword with `UnequalSpheres`, `UnequalBoxes` (default), `UniformUnequalSpheres`, `UniformUnequalBoxes`, `RotatedCoefficients`; `predict_error` keyword.
- New multipole acceptance criterion and bounding-box shrink: branches now track source and target radii and boxes; `source_shrink_recenter`/`target_shrink_recenter` default changed from `true` to `false`.
- Exports `EqualSpheres`, `UnequalSpheres`, `UnequalBoxes`, `Dynamic`, `build_interaction_lists`. Automatic-differentiation compatibility updates. Adds `BSON` as a dependency (removed again in v1.0.0).

## v0.3.0 - 2024 October

- Faster expansions: rotation-based (Wigner) multipole and local translations, with precomputed rotation constants up to order 20, plus faster body-to-multipole and a new `evaluate_expansions` path for local velocity and gradient.
- `lamb_helmholtz::Bool` keyword replaces the `vector_potential` and `method` keywords.
- The `fmm!` `gpu` keyword is retained; `direct_gpu!` and `buffer_element` are now exported, along with `Branch`, `Tree`, `unsort!`, `resort!`, `unsorted_index_2_sorted_index`, `sorted_index_2_unsorted_index`, and the `Position`, `Radius`, `Strength`, ... indexable types.
- Note: `ProbeSystem`'s `add_line!` and `reset!` are no longer exported (still defined).

## v0.2.0 - 2024 September

- `fmm!` and `direct!` keywords renamed: `n_per_branch_*` -> `leaf_size_source`/`leaf_size_target` (`leaf_size` for the single-system form), `multipole_acceptance_criterion` -> `multipole_threshold`; added `gpu` and `method` keywords, and `upward_pass`/`horizontal_pass`/`downward_pass` switches.
- Source `Strength` is now indexed as a single attribute, replacing `ScalarStrength` and `VectorStrength`.
- Added `direct_gpu!` hook, `resort!`, reuse of precomputed `m2l_list`/`direct_list`, and an early return when sources or targets are empty.
- `ProbeSystem` gained `add_line!` (exported).
- Added documentation site, `CHANGES.md`, and CI (Julia 1.10 tested).

## v0.1.0 - 2024 August

Initial release.

