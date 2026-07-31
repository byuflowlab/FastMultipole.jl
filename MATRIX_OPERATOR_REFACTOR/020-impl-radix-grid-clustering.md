# 020 Implementation Radix Grid Clustering

## Objective

Implement the radix-path uniform-grid clustering that the new matrix operators run
on: the `RadixGrid`, Morton/Z-order sort, and per-cell geometry.

## Dependencies

- `008f-theory-radix-sort-clustering.md`
- `007-theory-coefficient-buffer-layout.md`
- `008b-implementation-replan.md`
- `008c-implementation-performance-baseline.md`
- `009-impl-basis-and-operator-cache-types.md`

## Required Reading

- `START_HERE.md`
- Approved dependency task files listed above
- Approved `theory/radix-sort-clustering.md`
- Existing `tree.jl` octree construction for reference (not modified here)

## Artifacts or Production Surface

- New `src/tree_batched.jl` containing the `RadixGrid` type and clustering code
  (per the `_batched` placement rule). Type definitions for `RadixGrid` go in
  `src/containers.jl` per placement rule 1.
- Tests for quantization, Morton-key ordering, occupied-cell ranges, and cell
  center/width/radius queries.

## Deliverables

- Root cubic domain, grid depth `ell`, resolution `G = 2^ell`, cell half-width
  `w = h0 / G`, bounding radius `rho = w * sqrt(3)`.
- Integer coordinate quantization `clamp(floor((x - x_min) / Delta), 0, G - 1)` per
  axis and Morton-key interleaving.
- Occupied cells stored in Morton-sorted order with contiguous body ranges and
  per-cell center/width/radius queries, reusing the existing buffer/sort-index
  conventions where possible (do not physically move body data).

## Verification

Run clustering unit tests on deterministic point sets: verify quantization bounds,
Morton ordering, complete and non-overlapping body coverage, and cell-geometry
queries. Record commands and result summaries.

Implementation completed on 2026-06-27:

- Added storage-minimal `RadixGrid{TF}` with persistent root lower corner, root
  half-width, depth, sorted body permutation, sorted occupied Morton keys, and
  `2 x M` sorted body range metadata.
- Added `src/tree_batched.jl` for construction, Morton encode/decode,
  quantization, occupied-cell compression, binary-search lookup, cell geometry,
  offsets, and physical displacements.
- Added deterministic radix-grid clustering unit coverage and included it near
  `tree_test.jl`.

Verification commands:

- `julia --project=. -e 'using Test; include("test/radix_grid_clustering_test.jl")'`
  - Passed: `radix grid clustering | 62 passed`.
- `julia --project=. -e 'include("test/runtests.jl")'`
  - Could not run under the root project because `FLOWMath` is not in the root
    environment.
- `julia --project=test -e 'include("test/runtests.jl")'`
  - Could not run because the test environment does not declare the local
    `FastMultipole` package.
- `julia --project=test -e 'push!(LOAD_PATH, pwd()); include("test/runtests.jl")'`
  - Passed full suite with the local package layered under the test environment.

Tuple-system support added on 2026-06-27:

- Extended `RadixGrid` construction to accept `systems::Tuple`, promote numeric
  type across tuple entries, flatten bodies by system order, and build one shared
  Morton-sorted grid over global body ordinals.
- Added `body_system` / `body_index` metadata and `radix_body_system`,
  `radix_body_index`, and `radix_body_ref` accessors so global ordinals map back
  to original `(system_index, local_body_index)` references.
- Extended radix-grid tests for tuple construction, tuple bounds, complete sorted
  coverage, single-system mapping defaults, and mixed occupied-cell body refs.

Tuple-support verification commands:

- `julia --project=. -e 'using Test; include("test/radix_grid_clustering_test.jl")'`
  - Passed: `radix grid clustering | 96 passed`.
- `julia --project=test -e 'push!(LOAD_PATH, pwd()); include("test/runtests.jl")'`
  - Passed full suite, including `radix grid clustering | 96 passed`; one existing
    broken threaded extra-farfield test remained reported as broken.

## Approval Notes

Approved on 2026-06-27 by clear-context review.

Review scope:

- Read `START_HERE.md`, this task file, approved radix-sort clustering theory,
  and the recorded radix-sort verification summary.
- Inspected the task production surface: `RadixGrid` in `src/containers.jl`,
  clustering/accessor implementation in `src/tree_batched.jl`, package include
  and export wiring in `src/FastMultipole.jl`, and
  `test/radix_grid_clustering_test.jl`.

Findings:

- No blocking correctness issues found. The implementation provides the required
  cubic root domain, fixed-depth resolution, clamped coordinate quantization,
  UInt64 Morton key encode/decode, Morton-sorted body permutation, inverse
  permutation, occupied-cell key/range compression, cell geometry queries, body
  references for tuple systems, offset queries, and physical displacements.
- Tests cover boundary quantization, deterministic Morton ordering, inverse
  permutation round trips, complete/non-overlapping occupied ranges, empty and
  degenerate domains, cell geometry, offset/displacement queries, and tuple
  system body-reference mapping.
- Non-blocking watch item: construction currently uses Julia's stable
  comparison sort over Morton keys rather than a literal radix-sort kernel. This
  is acceptable for the CPU clustering contract in this row, but GPU/device
  residency work should revisit the sort primitive before treating this as the
  final large-`N` GPU path.

Verification commands run during approval:

- `julia --project=. -e 'using Test; include("test/radix_grid_clustering_test.jl")'`
  - Passed: `radix grid clustering | 96 passed`.
- `julia --project=test -e 'push!(LOAD_PATH, pwd()); include("test/runtests.jl")'`
  - Passed full suite; the existing threaded extra-farfield case remained
    reported as broken.
