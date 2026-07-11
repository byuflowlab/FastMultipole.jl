# 015 Implementation Axis Swap Benchmarks

## Objective

Benchmark the two near-term full-M2L operator variants over the stable `014`
interfaces.

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

- Benchmarks comparing only the two near-term variants:
  1. `MaterializedYRotationM2L`: materialized `Ts(theta)` M2L using the building
     blocks from `013`.
  2. `FactoredRotationM2L`: explicit factored `Z/S/Z/S` M2L using stages from
     `013b`/`013c`.
- Result notes comparing both explicit operator paths with current production M2L.
- Harness sweeps over representative expansion orders `P`, offset-class counts,
  shared-direction counts, shared-norm counts, and batch sizes; record
  allocation/storage/cache footprint as well as timing.
- Record, but do not benchmark as `015` deliverables, these deferred options for
  later final implementation/performance tasks: fully dense per-offset M2L matrix,
  partially folded hybrids around `K_z`, alternate z-translation cache/scaling
  policies, real-basis operator execution, per-`m` block-batched z-translation, and
  non-rotation operator families.
- **Separate CPU and GPU recommendations.** Record an explicit recommendation for
  each target, justified by the benchmark evidence. `024` remains the definitive
  end-to-end comparison after integration.

## Verification

Run benchmarks with recorded commands, environment notes, and result summaries.
Include enough detail to reproduce the comparison.

For the M2L variant benchmark, record the harness commands, environment, timing
and allocation/storage summaries across the `P` / offset-class / batch-size sweep,
and the rationale for the separate CPU and GPU recommendations.

## Approval Notes

To be filled by a different agent after benchmarks and verification notes are
complete.
