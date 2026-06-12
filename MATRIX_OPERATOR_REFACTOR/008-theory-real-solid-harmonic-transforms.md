# 008 Theory Real Solid Harmonic Transforms

## Objective

Derive complex-to-real and real-to-complex solid-harmonic transform
conventions and tests for real-basis M2M, M2L, and L2L matrix operators using
invariant matrices and z-axis rotations only.

## Dependencies

- `001-theory-z-rotation-operators.md`
- `007-theory-coefficient-buffer-layout.md`

## Required Reading

- `START_HERE.md`
- Approved dependency task files listed above
- Current compressed complex basis conventions

## Artifacts or Production Surface

- `theory/real-solid-harmonic-transforms.md`
- `scripts/real_solid_harmonic_*.jl`
- `data/real_solid_harmonic/`

## Deliverables

- Complex-to-real and real-to-complex transform formulas
- Normalization, ordering, and sign conventions
- Parity targets for real-basis execution
- Real-basis forms of the approved M2M, M2L, and L2L operator chains, derived
  from the complex-basis theory or directly in the real basis
- Verification requirements showing real-basis parity with the complex-basis
  point-mass unit-strength M2M, M2L, and L2L convergence example

## Verification

Check transform round trips and parity against complex-basis operator examples.
Demonstrate that the real-basis point-mass operator chain evaluates to the same
convergent `1/r` result as the complex-basis chain. Record commands,
tolerances, and result summaries.

## Approval Notes

To be filled by a different agent after artifacts and verification are
complete.
