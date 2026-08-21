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

- [x] M2M operator composition
- [x] L2L operator composition
- [x] Shared conventions with the approved M2L operator structure
- [x] Explicit statement that both M2M and L2L compositions use only invariant
  matrices and z-axis rotations for all non-z-aligned offsets
- [x] Complete point-mass unit-strength M2M, M2L, and L2L operator-chain example
  that obtains an expansion, translates it through the three operations,
  evaluates at a target point, and compares against analytic `1/r`

## Verification

Compare explicit M2M and L2L operator results against current production
behavior for representative parent-child offsets. Demonstrate convergence of
the complete point-mass example to `1/r` as expansion order increases. Record
commands, tolerances, and result summaries.

Completed with:

```text
julia --project=. MATRIX_OPERATOR_REFACTOR/scripts/m2m_l2l_verify.jl
```

Generated artifact:

```text
MATRIX_OPERATOR_REFACTOR/data/m2m_l2l/verification_summary.md
```

Result summary:

- Status: `PASS`
- Operator tolerance: `atol <= 1.0e-9`, `rtol <= 2.0e-11`
- Z-block tolerance: finite entries compare with `rtol <= 1.0e-12`;
  absolute errors are reported for scale context.
- Point-mass chain convergence target: final `rtol <= 5.0e-7` in `1/r`
  normalization.
- Max M2M absolute error: `5.417888360170764e-14`
- Max M2M relative error: `1.7094382274525772e-12`
- Max L2L absolute error: `7.460698725481052e-14`
- Max L2L relative error: `2.6067926229863967e-13`
- Max z-block relative error: `1.3682805004681272e-13`
- Final point-chain relative error: `8.556935334046337e-16`
- Coverage: axis-aligned `+z`, axis-aligned `-z`, positive off-axis,
  negative off-axis, layouts `Val(false)` and `Val(true)`, expansion orders
  `0`, `1`, `3`, `6`, `9`, M2M/L2L z-translation block parity, and a complete
  unit point-mass M2M-M2L-L2L convergence chain.

## Approval Notes

Approved by clear-context review (different agent; fresh session, did not author
the `006` artifacts).

Reviewed against `START_HERE.md` step 6 scope: this task file, its listed
artifacts (`theory/m2m-l2l-extensions.md`, `scripts/m2m_l2l_verify.jl`,
`data/m2m_l2l/verification_summary.md`), and the verification notes.

Findings:

- All six deliverables are checked and substantiated. The theory artifact
  derives the M2M (lower-triangular `U_m`) and L2L (upper-triangular `V_m`)
  z-translation blocks, uses only z-axis phase rotations plus invariant
  axis-swap matrices with `zeta`/`eta` sign tables for non-z offsets, confines
  channel coupling to the `Val(true)` Lamb-Helmholtz stages, and shares the
  approved M2L conventions.
- The unit point-mass M2M→M2L→L2L chain example with `-4π` normalization is
  present and converges to analytic `1/r`.
- Re-ran the verification independently:
  `julia --project=. MATRIX_OPERATOR_REFACTOR/scripts/m2m_l2l_verify.jl` →
  `m2m_l2l_verify: PASS`, with numbers matching this task file's recorded summary
  (M2M rel `1.71e-12`, L2L rel `2.61e-13`, z-block rel `1.37e-13`, final
  point-chain rel `8.56e-16`). The large z-block absolute error `262144.0`
  (P=9, t=1000) is expected scale context; its relative error is `~3e-16`.
- No contradictions among the index, this task file, and the artifacts. No
  `src/` production code was modified (Theory phase gate respected).
