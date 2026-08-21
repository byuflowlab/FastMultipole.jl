# 005 Theory Full M2L Composition

## Objective

Derive the complete M2L operator composition from approved z-rotation,
z-translation, Lamb-Helmholtz, and axis-swap theory using invariant matrices
and z-axis rotations only.

## Dependencies

- `004a-milestone-review-theory-001-004.md`
- `002-theory-m2l-z-translation-scaling.md`
- `003-theory-lamb-helmholtz-operator-form.md`
- `004-theory-axis-swap-conventions.md`

## Required Reading

- `START_HERE.md`
- Approved dependency task files listed above
- Approved artifacts listed by the dependency task files

## Artifacts or Production Surface

- `theory/full-m2l-composition.md`
- `scripts/full_m2l_composition_*.jl`
- `data/full_m2l_composition/`

## Deliverables

- [x] End-to-end M2L operator ordering
- [x] Intermediate basis/channel layout notes
- [x] Required cache entries and parity targets for implementation
- [x] Explicit statement that the M2L composition uses only invariant matrices and
  z-axis rotations for all non-z-aligned source-target offsets
- [x] Point-mass unit-strength M2L example component that can be composed with the
  later M2M and L2L examples and checked against the analytic potential `1/r`

## Verification

Compare composed operator output against current production M2L behavior across
representative source-target offsets. Include the M2L stage of the point-mass
unit-strength convergence example. Record commands, tolerances, and result
summaries.

Completed with:

```text
julia --project=. MATRIX_OPERATOR_REFACTOR/scripts/full_m2l_composition_verify.jl
```

Generated artifact:

```text
MATRIX_OPERATOR_REFACTOR/data/full_m2l_composition/verification_summary.md
```

Result summary:

- Status: `PASS`
- Composition tolerance: `atol <= 1.0e-9`, `rtol <= 2.0e-11`
- Scaled-block tolerance: finite entries compare with `rtol <= 1.0e-12`;
  absolute errors are reported for scale context.
- Point-mass convergence target: final `rtol <= 2.5e-7` in `1/r`
  normalization.
- Max composition absolute error: `5.165929906070232e-10`
- Max composition relative error: `1.5897417339721547e-12`
- Max scaled-block relative error: `4.092898796994199e-16`
- Final point-mass relative error: `3.6155107851558025e-16`
- Coverage: axis-aligned `+z`, axis-aligned `-z`, positive off-axis,
  negative off-axis, layouts `Val(false)` and `Val(true)`, expansion orders
  `0`, `1`, `3`, `6`, `9`, scaled-block distances including `1e-3` and
  `1e3`, and a unit point-mass M2L convergence example.

## Approval Notes

Approved after clear-context review.

- Reviewed `theory/full-m2l-composition.md`,
  `scripts/full_m2l_composition_verify.jl`, and
  `data/full_m2l_composition/verification_summary.md`.
- Confirmed the artifact specifies the complete M2L composition using z-axis
  rotations, invariant axis-swap matrices, fixed-`m` z-translation blocks, and
  the optional local Lamb-Helmholtz stage.
- Confirmed channel layout, overwrite/accumulate semantics, cache requirements,
  and the unit point-mass M2L component are documented.
- Re-ran
  `julia --project=. MATRIX_OPERATOR_REFACTOR/scripts/full_m2l_composition_verify.jl`;
  result was `PASS` with max composition absolute error
  `5.165929906070232e-10`, max composition relative error
  `1.5897417339721547e-12`, max scaled-block relative error
  `4.092898796994199e-16`, and final point-mass relative error
  `3.6155107851558025e-16`.
