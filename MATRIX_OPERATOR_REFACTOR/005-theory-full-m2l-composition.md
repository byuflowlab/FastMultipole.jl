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

- End-to-end M2L operator ordering
- Intermediate basis/channel layout notes
- Required cache entries and parity targets for implementation
- Explicit statement that the M2L composition uses only invariant matrices and
  z-axis rotations for all non-z-aligned source-target offsets
- Point-mass unit-strength M2L example component that can be composed with the
  later M2M and L2L examples and checked against the analytic potential `1/r`

## Verification

Compare composed operator output against current production M2L behavior across
representative source-target offsets. Include the M2L stage of the point-mass
unit-strength convergence example. Record commands, tolerances, and result
summaries.

## Approval Notes

To be filled by a different agent after artifacts and verification are
complete.
