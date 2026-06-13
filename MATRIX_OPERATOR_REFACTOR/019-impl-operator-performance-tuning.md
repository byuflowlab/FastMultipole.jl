# 019 Implementation Operator Performance Tuning

## Objective

Improve and document performance, allocation behavior, and retained storage of
completed operator paths after M2M, M2L, L2L, flat buffers, and real-basis
evaluation are implemented.

## Dependencies

- `008c-implementation-performance-baseline.md`
- `017-impl-flat-coefficient-buffers.md`
- `018-impl-real-solid-harmonic-basis.md`

## Required Reading

- `START_HERE.md`
- Approved dependency task files listed above
- Stable M2M, M2L, and L2L operator implementation notes
- Stable flat-buffer implementation notes
- Stable real-basis transform and native execution notes
- Existing benchmark scripts and results, including `008c` baseline notes

## Artifacts or Production Surface

- Production operator paths for M2M, M2L, L2L, flat buffers, and real-basis
  execution
- Benchmark scripts or benchmark test files
- Benchmark result artifacts under `MATRIX_OPERATOR_REFACTOR/data/` if needed
- Parity tests covering optimized paths

## Deliverables

- Whole-operator benchmark suite and recorded results
- Allocation and memory-footprint benchmarks for completed M2M, M2L, L2L,
  flat-buffer, and real-basis paths
- Bottleneck notes for completed operator paths
- Data-structure review identifying avoidable allocations, over-retained
  operator/cache data, duplicate transforms, poor scratch reuse, and storage
  layouts that block batching or GPU-friendly execution
- Scoped CPU single-thread optimizations where justified by benchmarks
- Scoped CPU multithread optimizations where justified by benchmarks
- Scoped storage/allocation improvements where justified by benchmarks and
  compatible with approved parity requirements
- GPU-oriented layout, batching, or execution notes where relevant
- Preserved parity with approved reference behavior
- Before/after benchmark summaries for chosen optimizations
- Before/after allocation counts, memory summaries, and retained-storage
  rationale for chosen optimizations
- Deferred alternatives and remaining performance or memory risks

## Verification

Rerun relevant parity tests plus benchmark commands. Record commands,
environment notes, before/after timing summaries, before/after allocation and
memory summaries, chosen optimizations, retained-storage rationale, deferred
alternatives, and remaining performance or memory risks.

## Approval Notes

To be filled by a different agent after tuning, parity verification, and
benchmark notes are complete.
