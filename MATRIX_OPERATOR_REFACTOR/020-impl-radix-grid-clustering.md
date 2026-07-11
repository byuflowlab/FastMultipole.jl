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

## Approval Notes

To be filled by a different agent after implementation and verification are
complete.
