# 010 Implementation Z Rotation Operators

## Objective

Implement explicit z-rotation operators and parity tests.

## Dependencies

- `001-theory-z-rotation-operators.md`
- `007-theory-coefficient-buffer-layout.md`
- `008b-implementation-replan.md`
- `008c-implementation-performance-baseline.md`
- `009-impl-basis-and-operator-cache-types.md`

## Required Reading

- `START_HERE.md`
- Approved dependency task files listed above
- Approved `theory/z-rotation-operators.md`
- Existing z-rotation production code and tests

## Artifacts or Production Surface

- Production z-rotation operator code
- Tests comparing explicit operators with current z-rotation behavior

## Deliverables

- Explicit z-rotation matrix/block construction
- Forward and inverse application paths
- Overwrite and accumulation behavior matching approved theory

## Verification

Run parity tests against current `rotate_z!` and `back_rotate_z!` behavior for
representative expansion orders and angles. Record commands and result
summaries.

## Approval Notes

To be filled by a different agent after implementation and verification are
complete.
