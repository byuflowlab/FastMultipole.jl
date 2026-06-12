# 017 Implementation Flat Coefficient Buffers

## Objective

Introduce flat coefficient buffers and typed views after the operator API is
stable.

## Dependencies

- `007-theory-coefficient-buffer-layout.md`
- `008b-implementation-replan.md`
- `016-impl-m2m-and-l2l-operator-pipelines.md`
- `016a-milestone-review-impl-013-016.md`

## Required Reading

- `START_HERE.md`
- Approved dependency task files listed above
- Approved `theory/coefficient-buffer-layout.md`
- Current coefficient allocation, indexing, and scratch-buffer code

## Artifacts or Production Surface

- Production coefficient buffer and typed-view code
- Tests for indexing, aliasing, and operator compatibility

## Deliverables

- Flat coefficient buffer representation
- Typed views matching approved layout theory
- Migration of stable operator API paths to the new layout

## Verification

Run indexing, round-trip, and operator parity tests. Record commands and result
summaries.

## Approval Notes

To be filled by a different agent after implementation and verification are
complete.
