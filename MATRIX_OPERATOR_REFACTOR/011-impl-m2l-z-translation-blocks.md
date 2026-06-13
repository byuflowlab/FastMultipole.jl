# 011 Implementation M2L Z Translation Blocks

## Objective

Implement fixed-`m` M2L z-translation blocks with approved distance scaling.

## Dependencies

- `002-theory-m2l-z-translation-scaling.md`
- `007-theory-coefficient-buffer-layout.md`
- `008b-implementation-replan.md`
- `008c-implementation-performance-baseline.md`
- `008d-theory-dynamic-p-error-m2l-integration.md`
- `009-impl-basis-and-operator-cache-types.md`

## Required Reading

- `START_HERE.md`
- Approved dependency task files listed above
- Approved `theory/m2l-z-translation-scaling.md`
- Approved `theory/dynamic-p-error-m2l-integration.md`
- Existing M2L z-translation production code and tests

## Artifacts or Production Surface

- Production M2L z-translation operator code
- Tests comparing fixed-`m` blocks with current z-aligned M2L behavior

## Deliverables

- Fixed-`m` block construction
- Stable distance-scaling implementation
- Cache integration compatible with task `009`

## Verification

Run parity tests across representative expansion orders, distances, and `m`
values. Record commands and result summaries.

## Approval Notes

To be filled by a different agent after implementation and verification are
complete.
