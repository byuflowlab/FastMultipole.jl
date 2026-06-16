# 012 Implementation Lamb Helmholtz Operators

## Objective

Implement Lamb-Helmholtz transform operators and parity tests.

## Dependencies

- `003-theory-lamb-helmholtz-operator-form.md`
- `007-theory-coefficient-buffer-layout.md`
- `008h-theory-lamb-helmholtz-accuracy-order.md`
- `008b-implementation-replan.md`
- `008c-implementation-performance-baseline.md`
- `009-impl-basis-and-operator-cache-types.md`

## Required Reading

- `START_HERE.md`
- Approved dependency task files listed above
- Approved `theory/lamb-helmholtz-operator-form.md`
- Existing Lamb-Helmholtz production code and tests

## Artifacts or Production Surface

- Production Lamb-Helmholtz operator code
- Tests comparing explicit operators with current channel behavior

## Deliverables

- Multipole and local Lamb-Helmholtz operator construction
- Channel-coupling application path
- Cache integration compatible with task `009`
- Local operators for `Val(true)` must include the upper-neighbor
  `chi_{P_phi + 1} -> chi_{P_phi}` row from
  `theory/lamb-helmholtz-accuracy-order.md`. Same-order `chi` truncation is a
  rejected candidate for the matrix M2L/evaluation path.

## Verification

Run parity tests across representative expansion orders and coefficient
patterns. Record commands and result summaries.

## Approval Notes

To be filled by a different agent after implementation and verification are
complete.
