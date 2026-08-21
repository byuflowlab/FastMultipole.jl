# 003 Theory Lamb Helmholtz Operator Form

## Objective

Derive multipole/local Lamb-Helmholtz operator form and channel coupling for
the explicit operator pipeline.

## Dependencies

- `001-theory-z-rotation-operators.md`

## Required Reading

- `START_HERE.md`
- `001-theory-z-rotation-operators.md`
- Current Lamb-Helmholtz expansion and translation formulas

## Artifacts or Production Surface

- `theory/lamb-helmholtz-operator-form.md`
- `scripts/lamb_helmholtz_operator_*.jl`
- `data/lamb_helmholtz_operator/`

## Deliverables

- [x] Operator form for multipole and local Lamb-Helmholtz channels
- [x] Channel-coupling formulas and basis-ordering notes
- [x] Compatibility notes for current compressed complex coefficients

## Verification

Compare operator formulas against deterministic coefficient examples and
current Lamb-Helmholtz paths. Record commands, tolerances, and result
summaries.

- Command:
  `julia --project=. MATRIX_OPERATOR_REFACTOR/scripts/lamb_helmholtz_operator_verify.jl`
- Tolerance: `atol <= 1e-12`, `rtol <= 1e-12`
- Result: `PASS`, with max absolute error `0.0` and max relative error `0.0`.
  Full output is recorded in
  `data/lamb_helmholtz_operator/verification_summary.md`.
- Coverage: multipole-side and local-side Lamb-Helmholtz transform operators;
  expansion orders `0`, `1`, `3`, `6`, and `9`; representative positive
  translation distances; deterministic two-channel compressed complex
  coefficients compared against production
  `transform_lamb_helmholtz_multipole!` and
  `transform_lamb_helmholtz_local!`.

## Approval Notes

- Approved on 2026-06-12 after clear-context review of
  `theory/lamb-helmholtz-operator-form.md`,
  `scripts/lamb_helmholtz_operator_verify.jl`, and
  `data/lamb_helmholtz_operator/verification_summary.md`.
- Reran
  `julia --project=. MATRIX_OPERATOR_REFACTOR/scripts/lamb_helmholtz_operator_verify.jl`;
  result was `PASS`, max absolute error `0.0`, and max relative error `0.0`.
- The artifact matches the current production Lamb-Helmholtz transform
  behavior in `src/translate.jl`, specifically
  `transform_lamb_helmholtz_multipole!` and
  `transform_lamb_helmholtz_local!`, including their production call sites in
  M2M, M2L, and L2L translation paths.
