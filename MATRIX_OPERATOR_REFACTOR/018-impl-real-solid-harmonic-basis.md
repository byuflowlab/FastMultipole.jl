# 018 Implementation Real Solid Harmonic Basis

## Objective

Add real-basis transforms and parity tests. Keep native real-basis operator
execution as future work after flat buffers and operator APIs stabilize.

## Dependencies

- `008-theory-real-solid-harmonic-transforms.md`
- `008b-implementation-replan.md`
- `008c-implementation-performance-baseline.md`
- `008e-theory-real-basis-kernel-derivatives.md`
- `008h-theory-lamb-helmholtz-accuracy-order.md`
- `017-impl-flat-coefficient-buffers.md`

## Required Reading

- `START_HERE.md`
- Approved dependency task files listed above
- Approved `theory/real-solid-harmonic-transforms.md`
- Approved `theory/real-basis-kernel-derivatives.md`
- Stable operator API and flat-buffer implementation notes

## Artifacts or Production Surface

- Production real-basis transform code
- Tests and benchmarks for complex-basis parity

## Deliverables

- Complex-to-real and real-to-complex transform implementation
- Native real-basis evaluation of the potential, gradient, and gradient
  Jacobian (Hessian) per approved `theory/real-basis-kernel-derivatives.md`,
  matching the production `DerivativesSwitch{PS,GS,HS}` paths
- Native real-basis `Val(true)` execution must preserve the approved channel
  sizing rule from `theory/lamb-helmholtz-accuracy-order.md`:
  `P_chi = P_phi + 1` for M2L/evaluation, with `Val(false)` unchanged.
- Notes recording that production native real-basis operator execution remains
  deferred until after flat buffers and operator APIs stabilize.

## Verification

Run transform round-trip tests, operator parity tests, and relevant benchmarks.
Record commands and result summaries.

## Approval Notes

To be filled by a different agent after implementation, evaluation, and
verification are complete.
