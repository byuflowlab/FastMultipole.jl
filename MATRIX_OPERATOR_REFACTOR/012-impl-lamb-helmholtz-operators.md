# 012 Implementation Lamb Helmholtz Operators

## Objective

Implement Lamb-Helmholtz transform operators and parity tests.

## Dependencies

- `003-theory-lamb-helmholtz-operator-form.md`
- `007-theory-coefficient-buffer-layout.md`
- `008b-implementation-replan.md`
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

## Verification

Run parity tests across representative expansion orders and coefficient
patterns. Record commands and result summaries.

## Approval Notes

To be filled by a different agent after implementation and verification are
complete.
