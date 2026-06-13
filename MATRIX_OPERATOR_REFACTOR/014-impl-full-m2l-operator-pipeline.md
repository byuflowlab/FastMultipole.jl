# 014 Implementation Full M2L Operator Pipeline

## Objective

Compose the full M2L operator pipeline and test it against current production
behavior.

## Dependencies

- `005-theory-full-m2l-composition.md`
- `008b-implementation-replan.md`
- `008c-implementation-performance-baseline.md`
- `008d-theory-dynamic-p-error-m2l-integration.md`
- `010-impl-z-rotation-operators.md`
- `011-impl-m2l-z-translation-blocks.md`
- `012-impl-lamb-helmholtz-operators.md`
- `013-impl-axis-swap-operators.md`

## Required Reading

- `START_HERE.md`
- Approved dependency task files listed above
- Approved `theory/full-m2l-composition.md`
- Approved `theory/dynamic-p-error-m2l-integration.md`
- Current production M2L call sites and tests

## Artifacts or Production Surface

- Production full M2L operator pipeline code
- Tests comparing full operator output with current M2L behavior

## Deliverables

- End-to-end explicit M2L operator composition
- Cache and scratch usage integrated with earlier implementation tasks
- Compatibility path that preserves current production behavior. Per the `008b`
  re-plan, this first pass is **side-by-side, parity-only**: the explicit M2L
  operator pipeline is validated against production but does **not** replace the
  production `multipole_to_local!` / `multipole_to_local_II!` internals.
  Production hot-path replacement is a later, explicitly scoped step.

## Verification

Run parity tests across representative source-target offsets and expansion
orders. Record commands and result summaries.

## Approval Notes

To be filled by a different agent after implementation and verification are
complete.
