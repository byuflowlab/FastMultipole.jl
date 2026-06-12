# 002 Theory M2L Z Translation Scaling

## Objective

Derive fixed-`m` M2L z-translation matrices and stable distance-scaling
conventions for the current compressed complex basis.

## Dependencies

- `001-theory-z-rotation-operators.md`

## Required Reading

- `START_HERE.md`
- `001-theory-z-rotation-operators.md`
- Current M2L z-translation formulas and coefficient indexing code

## Artifacts or Production Surface

- `theory/m2l-z-translation-scaling.md`
- `scripts/m2l_z_translation_*.jl`
- `data/m2l_z_translation/`

## Deliverables

- Fixed-`m` matrix form for M2L z translations
- Distance-scaling convention and stable evaluation notes
- Index and sign conventions matched to approved z-rotation theory

## Verification

Compare matrix application against current z-aligned M2L behavior across
representative expansion orders, distances, and `m` values. Record commands,
tolerances, and result summaries.

- Command: `julia --project=. MATRIX_OPERATOR_REFACTOR/scripts/m2l_z_translation_verify.jl`
- Tolerance: `atol <= 1e-12`, `rtol <= 1e-12`
- Result: `PASS`, with max absolute error `3.469446951953614e-18`, max
  relative error `4.823830462317249e-16`, and zero inactive-channel overwrite
  error for `Val(false)`. Full output is recorded in
  `data/m2l_z_translation/verification_summary.md`.
- Coverage: `Val(false)` and `Val(true)` channel layouts; expansion orders
  `0`, `1`, `3`, `6`, and `9`; representative positive z distances; dense
  fixed-`m` matrix application compared against production
  `translate_multipole_to_local_z!`.

## Approval Notes

Approved after clear-context review.

- Reviewed `START_HERE.md`, this task file, the listed theory artifact, the
  verification script, the generated verification summary, and the current
  production `translate_multipole_to_local_z!` and `harmonic_index` code.
- Confirmed the fixed-`m` dense block form, compressed-basis indexing, channel
  semantics, recurrence scaling, and overwrite behavior match production.
- Re-ran
  `julia --project=. MATRIX_OPERATOR_REFACTOR/scripts/m2l_z_translation_verify.jl`;
  result `PASS`, max absolute error `3.469446951953614e-18`, max relative
  error `4.823830462317249e-16`, inactive-channel overwrite error `0.0`.
