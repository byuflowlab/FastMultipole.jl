# 006 Theory M2M L2L Extensions

## Objective

Extend the approved component theory to explicit M2M and L2L operator
pipelines using invariant matrices and z-axis rotations only.

## Dependencies

- `005-theory-full-m2l-composition.md`
- `003-theory-lamb-helmholtz-operator-form.md`
- `004-theory-axis-swap-conventions.md`

## Required Reading

- `START_HERE.md`
- Approved dependency task files listed above
- Current M2M and L2L formulas and traversal call sites

## Artifacts or Production Surface

- `theory/m2m-l2l-extensions.md`
- `scripts/m2m_l2l_*.jl`
- `data/m2m_l2l/`

## Deliverables

- M2M operator composition
- L2L operator composition
- Shared conventions with the approved M2L operator structure
- Explicit statement that both M2M and L2L compositions use only invariant
  matrices and z-axis rotations for all non-z-aligned offsets
- Complete point-mass unit-strength M2M, M2L, and L2L operator-chain example
  that obtains an expansion, translates it through the three operations,
  evaluates at a target point, and compares against analytic `1/r`

## Verification

Compare explicit M2M and L2L operator results against current production
behavior for representative parent-child offsets. Demonstrate convergence of
the complete point-mass example to `1/r` as expansion order increases. Record
commands, tolerances, and result summaries.

## Approval Notes

To be filled by a different agent after artifacts and verification are
complete.
