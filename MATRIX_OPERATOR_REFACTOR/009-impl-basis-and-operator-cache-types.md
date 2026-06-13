# 009 Implementation Basis And Operator Cache Types

## Objective

Define basis and operator-cache types without changing production translation
call behavior.

## Dependencies

- `001-theory-z-rotation-operators.md`
- `004-theory-axis-swap-conventions.md`
- `005-theory-full-m2l-composition.md`
- `007-theory-coefficient-buffer-layout.md`
- `008a-milestone-review-theory-005-008.md`
- `008b-implementation-replan.md`
- `008c-implementation-performance-baseline.md`

## Required Reading

- `START_HERE.md`
- Approved dependency task files listed above
- Approved artifacts listed by the dependency task files
- Existing production types and translation call sites

## Artifacts or Production Surface

- Production source files that define basis/cache/scratch types
- Focused tests for type construction and current call behavior

## Deliverables

- Basis type names for the current compressed complex basis
- Shared invariant cache ownership model
- Per-thread scratch ownership model
- Element-type parameterization where practical

## Verification

Run focused tests proving existing translation behavior is unchanged and new
types construct correctly. Record commands and result summaries.

## Approval Notes

To be filled by a different agent after implementation and verification are
complete.
