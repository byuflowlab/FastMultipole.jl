# 015 Implementation Axis Swap Benchmarks

## Objective

Benchmark invariant axis-swap composition and full M2L operator paths.

## Dependencies

- `004-theory-axis-swap-conventions.md`
- `005-theory-full-m2l-composition.md`
- `008b-implementation-replan.md`
- `008c-implementation-performance-baseline.md`
- `014-impl-full-m2l-operator-pipeline.md`

## Required Reading

- `START_HERE.md`
- Approved dependency task files listed above
- Existing benchmark scripts and full M2L implementation notes

## Artifacts or Production Surface

- Benchmark scripts or benchmark test files
- Benchmark result artifacts under `MATRIX_OPERATOR_REFACTOR/data/` if needed

## Deliverables

- Benchmarks for invariant axis-swap composition
- Benchmarks for full M2L operator paths
- Result notes comparing explicit operator paths with current paths
- **M2L batching-strategy benchmark and decision.** Enumerate, implement a
  benchmark harness for, and measure the plausible M2L batching candidates over
  the radix path's translation-invariant offset classes. No candidate is selected
  until all plausible candidates are benchmarked; the decision is benchmark-gated
  and recorded with its supporting evidence. Candidates (this set is the floor, not
  a ceiling — add any further plausible candidate before deciding):
  1. **Per-offset dense operator + batched GEMM.** One folded dense real operator
     `M_d` per accepted offset class
     (rotation · z-translation · rotation collapsed into a single
     `2(P+1)² × 2(P+1)²` matrix); per class, one (strided/batched) GEMM
     `L_block += M_d · S_block` over all pairs in that class. O(p⁴) flops/storage
     per pair, few distinct `M_d` reused across large batches.
  2. **Globally batched rotation / z-diagonal stages.** Keep the O(p³)
     rotation-trick chain but batch each stage across all pairs at once: forward
     rotations grouped by offset direction, the z-axis diagonal/banded translation
     applied to all pairs together (depends only on `|d|`, `P`), and batched
     accumulating back-rotations. Lower flops/storage; irregular scatter/gather and
     accumulation, and the rotate stages may not be compute-bound.
  3. **Per-offset factored small-matrix batched GEMM (hybrid).** Per offset class,
     apply each of the three factored operators as a strided-batched small GEMM —
     O(p³), regular batches, BLAS/cuBLAS-friendly without dense O(p⁴)
     materialization.
  4. **Per-`m` block-batched z-translation.** Exploit the z-translation's
     block-by-`m` structure: assemble per-`m` blocks across all pairs into batched
     small GEMMs.
  5. **(Deferred, recorded only) plane-wave / exponential diagonal translation** —
     a different operator family outside the rotation-based refactor scope.
  The harness must sweep representative expansion orders `P`, offset-class counts,
  and batch sizes (both per-offset-class and global batching), and record
  allocation/storage footprint per candidate as well as timing.
- **Separate CPU and GPU recommendations.** Record an explicit recommendation for
  each target, justified by the benchmark evidence (e.g. dense per-offset GEMM and
  factored strided-batched GEMM are expected to favor GPU via few large regular
  cuBLAS calls under translation invariance; rotation-trick stages batched by
  offset class are expected to favor CPU at moderate/high `P` where dense O(p⁴)
  operators dominate — both to be confirmed, not assumed).

## Verification

Run benchmarks with recorded commands, environment notes, and result summaries.
Include enough detail to reproduce the comparison.

For the M2L batching benchmark, record the harness commands, environment, the
per-candidate timing and allocation/storage summaries across the `P` / offset-class
/ batch-size sweep, and the rationale for the separate CPU and GPU
recommendations. The final batching decision must cite the benchmark evidence and
confirm all plausible candidates were tested.

## Approval Notes

To be filled by a different agent after benchmarks and verification notes are
complete.
