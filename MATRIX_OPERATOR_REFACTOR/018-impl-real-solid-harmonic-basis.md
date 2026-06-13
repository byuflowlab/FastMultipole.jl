# 018 Implementation Real Solid Harmonic Basis

## Objective

Add real-basis transforms and evaluate native real-basis execution.

## Dependencies

- `008-theory-real-solid-harmonic-transforms.md`
- `008b-implementation-replan.md`
- `008c-implementation-performance-baseline.md`
- `008e-theory-real-basis-kernel-derivatives.md`
- `017-impl-flat-coefficient-buffers.md`

## Required Reading

- `START_HERE.md`
- Approved dependency task files listed above
- Approved `theory/real-solid-harmonic-transforms.md`
- Approved `theory/real-basis-kernel-derivatives.md`
- Stable operator API and flat-buffer implementation notes

## Artifacts or Production Surface

- Production real-basis transform code
- Tests and benchmarks for complex-basis parity and native real-basis
  execution

## Deliverables

- Complex-to-real and real-to-complex transform implementation
- Evaluation path for native real-basis operator execution
- Native real-basis evaluation of the potential, gradient, and gradient
  Jacobian (Hessian) per approved `theory/real-basis-kernel-derivatives.md`,
  matching the production `DerivativesSwitch{PS,GS,HS}` paths
- Notes on whether native real-basis execution should remain enabled,
  experimental, or deferred

## Verification

Run transform round-trip tests, operator parity tests, and relevant benchmarks.
Record commands and result summaries.

## Approval Notes

To be filled by a different agent after implementation, evaluation, and
verification are complete.
