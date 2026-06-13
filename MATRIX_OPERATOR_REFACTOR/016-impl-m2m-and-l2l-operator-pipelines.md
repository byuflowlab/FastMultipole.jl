# 016 Implementation M2M And L2L Operator Pipelines

## Objective

Extend the explicit operator structure to M2M and L2L.

## Dependencies

- `006-theory-m2m-l2l-extensions.md`
- `008b-implementation-replan.md`
- `008c-implementation-performance-baseline.md`
- `014-impl-full-m2l-operator-pipeline.md`
- `015-impl-axis-swap-benchmarks.md`

## Required Reading

- `START_HERE.md`
- Approved dependency task files listed above
- Approved `theory/m2m-l2l-extensions.md`
- Current production M2M and L2L call sites and tests

## Artifacts or Production Surface

- Production M2M and L2L operator pipeline code
- Tests comparing explicit M2M and L2L output with current behavior

## Deliverables

- Explicit M2M operator pipeline
- Explicit L2L operator pipeline
- Shared cache and scratch integration with the M2L operator layer
- Per the `008b` re-plan, this first pass is **side-by-side, parity-only**: the
  explicit M2M/L2L operator pipelines are validated against production but do
  **not** replace the production `multipole_to_multipole!` / `local_to_local!`
  internals. Production hot-path replacement is a later, explicitly scoped step.

## Verification

Run parity tests for representative parent-child offsets, traversal contexts,
and expansion orders. Record commands and result summaries.

## Approval Notes

To be filled by a different agent after implementation and verification are
complete.
