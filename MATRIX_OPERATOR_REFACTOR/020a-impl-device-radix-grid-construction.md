# 020a Implementation Device Radix Grid Construction

## Objective

Implement the CUDA/device-resident radix grid construction required by the item
`022` GPU lifecycle. The existing item `020` CPU `RadixGrid` remains approved as
the CPU/reference path; this row adds a device-owned construction path so CUDA
evaluations do not sort bodies or build radix metadata on the host.

## Dependencies

- `008f-theory-radix-sort-clustering.md`
- `007-theory-coefficient-buffer-layout.md`
- `008b-implementation-replan.md`
- `008c-implementation-performance-baseline.md`
- `009-impl-basis-and-operator-cache-types.md`
- `020-impl-radix-grid-clustering.md`

## Required Reading

- `START_HERE.md`
- Approved dependency task files listed above
- Approved `theory/radix-sort-clustering.md`
- Existing CPU radix construction in `src/tree_batched.jl`
- Current CUDA lifecycle scaffold in `src/translate_batched_cuda.jl`

## Artifacts or Production Surface

- Device-radix metadata containers in `src/containers.jl`, keeping CUDA-specific
  array types out of the default CPU include path.
- CUDA implementation in `src/translate_batched_cuda.jl` or another `*_cuda.jl`
  file loaded only by the opt-in CUDA lifecycle loader.
- Tests comparing device-built radix metadata with the approved CPU `RadixGrid`
  reference, skipped gracefully when CUDA is unavailable.

## Deliverables

- Device-side body bounds handling for the CUDA path: either compute bounds on
  device or accept explicit caller-provided bounds for repeat evaluations.
- Device kernel to compute quantized grid coordinates, Morton keys, and stable
  body ids from resident source bodies.
- Device-side sort of `(Morton key, body id)` pairs. The steady-state CUDA path
  must not call host `sortperm` or use host-built `RadixGrid.perm` as the source
  of truth.
- Device-side occupied leaf range construction by detecting sorted key changes.
- Device-side occupied ancestor/node metadata sufficient for later resident
  B2M, M2M, M2L, L2L, and L2B passes:
  leaf keys/ranges, node level/key or coordinate, node center, leaf-to-node
  mapping, and parent/child relationships.
- Host orchestration may launch kernels and read back small scalar counts when
  unavoidable, but must not build the sorted grid, leaf ranges, or ancestor/child
  lists for the production CUDA lifecycle.
- Preserve the approved CPU `RadixGrid` behavior and public CPU tests.

## Verification

On a CUDA host, compare device-built metadata against CPU `RadixGrid` for
deterministic point sets: quantization, Morton ordering, stable tie behavior,
occupied leaf keys/ranges, complete non-overlapping body coverage, cell geometry,
and parent/child ancestor relationships. Confirm the normal CPU-only package path
does not load CUDA and that tests skip gracefully when no CUDA device is present.
Record commands, environment, and result summaries.

Implementation progress on 2026-06-30:

- Added first-class radix sort backend tags:
  `HostRadixSort`, `DeviceRadixSort`, and `AutoRadixSort`.
- Replaced the CPU `RadixGrid` ordering implementation with a stable host
  LSD radix sort over `UInt64` Morton keys. Stable duplicate-key body ordering is
  preserved by sorting the initial body-id permutation.
- Added `radix_grid(...; sort=...)` while preserving existing `RadixGrid(...)`
  callers. `AutoRadixSort` falls back to the host path in the CPU-only package
  path; explicit `DeviceRadixSort` is rejected until the CUDA lifecycle is loaded.
- Added CPU-load-safe `DeviceRadixGrid` metadata in `src/containers.jl`.
- Added opt-in CUDA constructors in `src/translate_batched_cuda.jl`:
  `cuda_radix_grid(...)` for device-resident systems and prepacked device body
  matrices. The CUDA path computes bounds, quantized coordinates, Morton keys,
  permutation, inverse permutation, occupied leaf keys/ranges, and leaf centers
  from resident device arrays. It uses CUDA device sorting rather than host
  `sortperm`.
- Added `cuda_radix_state(..., grid::DeviceRadixGrid, ...)` so lifecycle setup
  reuses resident `perm`, body maps, cell ranges, and cell centers instead of
  uploading host grid metadata.
- Added CUDA-gated tests comparing `DeviceRadixGrid` metadata with CPU
  `RadixGrid` when CUDA is available. These tests are skipped by the existing
  lifecycle gate when CUDA is unavailable.

Local verification commands:

```sh
julia --project=. -e 'using Test; include("test/radix_grid_clustering_test.jl")'
julia --project=. test/cuda_radix_lifecycle_test.jl
julia --project=test -e 'push!(LOAD_PATH, pwd()); include("test/runtests.jl")'
```

Local verification results:

- Radix grid clustering: `103 passed`.
- CUDA lifecycle CPU-safe gate: `21 passed`.
- Full CPU-safe suite passed; the existing threaded extra-farfield case remained
  reported as broken.
- CUDA hardware validation was not run locally because this host does not expose
  a functional CUDA device.

## Approval Notes

Approved on 2026-06-30 after clear-context review.

Reviewed `START_HERE.md`, this task file, and the listed production/test
surfaces: CPU radix construction in `src/tree_batched.jl`, device-radix
containers in `src/containers.jl`, the opt-in CUDA implementation in
`src/translate_batched_cuda.jl`, CUDA loader wiring in `src/FastMultipole.jl`,
and the relevant radix/CUDA lifecycle tests.

No blocking findings. The implementation matches the stated objective: the CPU
`RadixGrid` reference remains intact, CUDA-specific code stays behind the
opt-in loader, `DeviceRadixGrid` is CPU-load-safe, and the device path builds
Morton keys, stable permutations, leaf ranges, cell geometry, occupied ancestor
nodes, and parent/child metadata from resident device arrays. CUDA.jl's local
sorting implementation documents stable `sortperm!`, which supports the
duplicate-key parity requirement.

Verification run during approval:

```sh
julia --project=. -e 'using Test; include("test/radix_grid_clustering_test.jl")'
julia --project=. test/cuda_radix_lifecycle_test.jl
julia --project=test -e 'push!(LOAD_PATH, pwd()); include("test/runtests.jl")'
```

Results: radix grid clustering `103 passed`; CUDA lifecycle CPU-safe gate
`21 passed`; full CPU-safe suite passed with the existing threaded
extra-farfield case reported as `Broken`, not failed. CUDA hardware validation
was not rerun on this host because this project environment has no functional
CUDA device available.

## Re-Review Notes (2026-07-07)

A second clear-context review re-ran the verification commands and re-inspected
the listed surfaces. Approval stands; one latent defect and one open
verification item are recorded for resolution during `022` hardware validation:

1. **Data race in `_cuda_fill_cell_ranges_kernel!`**
   (`src/translate_batched_cuda.jl`): for a multi-body cell, the thread at the
   cell's last sorted index reads `cell_ranges[1, icell]`, which is written by a
   different thread (the cell's first sorted index) in the same kernel launch
   with no synchronization. Under the CUDA memory model this read is not
   guaranteed to observe the write, so `cell_ranges[2, icell]` may be computed
   from uninitialized memory. Fix by splitting first-fill and count-fill into
   two kernel launches (the launch boundary is the required sync), or by
   deriving the count without the cross-thread read. Likely benign at the tiny
   test sizes but must be fixed before `022` relies on device-built cell
   ranges at scale.
2. **No CUDA hardware run has been recorded** for this row (completion and both
   reviews ran on CUDA-less hosts). The CUDA-gated metadata-parity tests are
   well constructed, but the device construction path remains unvalidated on
   hardware; `022` must run them (e.g. on rc.byu.edu) before depending on this
   path, and the item-1 fix should land first.

Supporting checks: CUDA.jl's bitonic sorting source documents "`sortperm!` is
properly stable", confirming the stable duplicate-key tie-break claim; CPU and
device quantization/Morton/center formulas match; the CUDA file is included only
inside `load_cuda_radix_lifecycle!`; `DeviceRadixGrid` in `containers.jl` is
CPU-load-safe. Re-run results on 2026-07-07: radix grid clustering `103 passed`;
CUDA lifecycle CPU-safe gate `91 passed` (grown since completion by later
resident-lifecycle work in the same files); full CPU-safe suite passed with no
failures or errors. Minor non-blocking notes: `_cuda_radix_node_metadata`
retains per-level scratch (`sorted_keys`/`flags`/`prefix`, `O(ell·n_cells)`
device memory) until GC, and node/cell center kernels use `Float64` literals
(`0.5`), which may cause last-ulp `Float32` parity differences.

2026-07-07 follow-up: finding 1 was fixed by splitting cell range construction
into `_cuda_fill_cell_firsts_kernel!` and `_cuda_fill_cell_counts_kernel!`, with
the launch boundary providing synchronization before count construction reads
the first index. The scratch-retention note was addressed by compacting
per-level keys immediately and calling `CUDA.unsafe_free!` on construction
scratch; peak retained scratch is now `O(n_cells + n_nodes)` with deterministic
release. CUDA hardware validation remains owed by task `022`.
