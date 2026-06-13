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

- [x] Complex-to-real and real-to-complex transform formulas
- [x] Normalization, ordering, and sign conventions
- [x] Parity targets for real-basis execution
- [x] Real-basis forms of the approved M2M, M2L, and L2L operator chains, derived
  from the complex-basis theory or directly in the real basis
- [x] Verification requirements showing real-basis parity with the complex-basis
  point-mass unit-strength M2M, M2L, and L2L convergence example

## Verification

Check transform round trips and parity against complex-basis operator examples.
Demonstrate that the real-basis point-mass operator chain evaluates to the same
convergent `1/r` result as the complex-basis chain. Record commands,
tolerances, and result summaries.

Completed with:

```text
julia --project=. MATRIX_OPERATOR_REFACTOR/scripts/real_solid_harmonic_transforms_verify.jl
```

Generated summary:

```text
MATRIX_OPERATOR_REFACTOR/data/real_solid_harmonic/verification_summary.md
```

Result: `PASS`.

Recorded tolerances:

- Transform/index tolerance: `atol <= 1.0e-14`
- Operator parity tolerance: `atol <= 1.0e-12`
- Point-mass chain convergence target: final `rtol <= 5.0e-7` in `1/r`
  normalization

Result summary:

- Mode indices contiguous and unique for `P = 0, 1, 3, 6, 9`: `true`
- Max real-to-complex-to-real error: `0.0`
- Max complex-to-real-to-complex representable error: `0.0`
- Max projected `m = 0` imaginary lane after round trip: `0.0`
- Max real z-rotation parity error: `0.0`
- Same-point scalar evaluation coverage: 7 local points for each
  `P = 0, 1, 3, 6, 9`
- Same-point scalar evaluation parity uses the native real-basis scalar
  evaluator derived in `MATRIX_OPERATOR_REFACTOR/theory/real-solid-harmonic-transforms.md`
- Max same-point scalar evaluation absolute error over those 35 cases:
  `6.938893903907228e-18`
- Max same-point scalar evaluation relative error over those 35 cases:
  `2.1171873673990052e-16`
- Max real M2L parity error: `0.0`
- Max real M2M/M2L/L2L operator parity error: `0.0`
- Final point-chain relative error: `8.556935334046337e-16`

## Approval Notes

Approved after clear-context review by a different agent than the completing
agent.

- Reviewed `START_HERE.md`, this task file, approved dependencies
  `001-theory-z-rotation-operators.md` and
  `007-theory-coefficient-buffer-layout.md`,
  `theory/real-solid-harmonic-transforms.md`,
  `scripts/real_solid_harmonic_transforms_verify.jl`,
  `data/real_solid_harmonic/verification_summary.md`, and the relevant
  production compressed-basis definitions for `harmonic_index`, z rotations,
  and scalar local evaluation.
- Confirmed the real basis is defined as the storage transform of the approved
  compressed complex basis, with contiguous degree-major real mode ordering and
  intentional projection of non-representable `m = 0` imaginary lanes.
- Confirmed the native real scalar evaluation formula matches production
  compressed-complex scalar local evaluation in the represented subspace,
  including the recorded `-4*pi` conversion for analytic `1/r` comparisons.
- Confirmed the real z-rotation block signs match the approved complex
  z-rotation signs and preserve forward overwrite / inverse accumulation
  semantics.
- Confirmed M2M, M2L, and L2L real-basis operator chains are derived only by
  `A_real = T_c2r * A_complex * T_r2c`, preserving the approved complex-basis
  operator order and invariant axis-swap contract.
- Re-ran
  `julia --project=. MATRIX_OPERATOR_REFACTOR/scripts/real_solid_harmonic_transforms_verify.jl`:
  `PASS`; transform round-trip, z-rotation parity, M2L parity, and full
  M2M/M2L/L2L operator parity headline errors were `0.0`; final point-chain
  relative error was `8.556935334046337e-16`.
