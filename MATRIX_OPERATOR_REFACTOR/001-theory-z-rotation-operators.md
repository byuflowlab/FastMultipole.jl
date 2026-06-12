# 001 Theory Z Rotation Operators

## Objective

Derive explicit real `2x2` z-rotation blocks for the current compressed
complex basis, including inverse signs and overwrite versus accumulate
semantics.

## Dependencies

- None

## Required Reading

- `START_HERE.md`
- Current `rotate_z!`, `back_rotate_z!`, and `harmonic_index(n, m)` definitions

## Artifacts or Production Surface

- `theory/z-rotation-operators.md`
- `scripts/z_rotation_*.jl`
- `data/z_rotation/`

## Deliverables

- [x] Formula for forward rotation of each `(n, m)` real/imaginary pair
- [x] Formula for inverse rotation and `m = 0` behavior
- [x] Notes on overwrite semantics for forward rotation and accumulation semantics
  for back rotation
- [x] Compatibility notes for the existing harmonic indexing layout
- [x] Storage-light `C/S` fused-operator form instead of dense block storage
- [x] GPU and flat-buffer compatibility notes

## Verification

Compare derived blocks against deterministic coefficient examples and the
current z-rotation formulas. Record commands, tolerances, and result summaries.

- Command: `julia --project=. MATRIX_OPERATOR_REFACTOR/scripts/z_rotation_verify.jl`
- Tolerance: `atol <= 1e-12`
- Result: `PASS`, with max forward error `3.552713678800501e-15`,
  max back/inverse accumulation error `3.1086244689504383e-15`, and zero
  `m = 0` identity/accumulation error. Full output is recorded in
  `data/z_rotation/verification_summary.md`.
- Coverage: `Val(false)` and `Val(true)` channel layouts; forward overwrite;
  back-rotation accumulation; `m = 0` identity/accumulation behavior

## Approval Notes

Approved after clear-context review.

- Reviewed `START_HERE.md`, this task file, `theory/z-rotation-operators.md`,
  `scripts/z_rotation_verify.jl`, `data/z_rotation/verification_summary.md`,
  and the production `harmonic_index`, `rotate_z!`, and `back_rotate_z!`
  definitions.
- Confirmed the documented forward block signs match production overwrite
  behavior for `Val(false)` and `Val(true)` active channels.
- Confirmed the documented inverse block signs match production
  `back_rotate_z!` accumulation behavior, including unchanged `m = 0`
  accumulation.
- Re-ran `julia --project=. MATRIX_OPERATOR_REFACTOR/scripts/z_rotation_verify.jl`:
  `PASS`; max forward error `3.552713678800501e-15`, max back/inverse
  accumulation error `3.1086244689504383e-15`, and zero `m = 0` errors.
