# 008a Milestone Review Theory 005-008

## Objective

Review Theory tasks `005` through `008` against the background design and
confirm the Theory gate is ready before any Implementation task begins.

## Dependencies

- `005-theory-full-m2l-composition.md`
- `006-theory-m2m-l2l-extensions.md`
- `007-theory-coefficient-buffer-layout.md`
- `008-theory-real-solid-harmonic-transforms.md`

## Required Reading

- `START_HERE.md`
- `../MATRIX_OPERATOR_REFACTOR.md`
- Completed task files listed above
- Artifacts listed by the completed task files

## Artifacts or Production Surface

Review artifacts under `theory/`, `scripts/`, and `data/` that are listed by
tasks `005` through `008`. No production code changes are part of this review.

## Deliverables

- Roadmap-alignment notes recorded in this file
- Explicit confirmation that all Theory rows are complete and approved before
  Implementation starts
- Explicit confirmation that the Theory Phase Acceptance Target in
  `START_HERE.md` is satisfied for both complex and real solid harmonic bases,
  including M2M, M2L, L2L, invariant matrices, z-axis rotations only, and the
  point-mass unit-strength convergence example
- Any required coordination-document fixes identified before later work starts

## Verification

Confirm completed work matches the background design, hard phase gate, and task
ordering. Confirm the completed artifacts demonstrate convergence of the
required point-mass example to analytic `1/r` as expansion order increases. If
`START_HERE.md`, a task file, and `../MATRIX_OPERATOR_REFACTOR.md` disagree,
stop and require a coordination-document fix.

## Review Notes

Milestone review completed on 2026-06-13.

Reviewed required coordination and design context:

- `START_HERE.md`
- `../MATRIX_OPERATOR_REFACTOR.md`
- `005-theory-full-m2l-composition.md`
- `006-theory-m2m-l2l-extensions.md`
- `007-theory-coefficient-buffer-layout.md`
- `008-theory-real-solid-harmonic-transforms.md`
- Listed theory artifacts under `theory/`, `scripts/`, and `data/` for tasks
  `005` through `008`

Roadmap alignment:

- Tasks `005` through `008` are marked complete and approved in
  `START_HERE.md`, and their task-local approval notes record clear-context
  review by agents other than the completing agents.
- The completed artifacts match the background roadmap: the current compressed
  complex basis remains the first implementation target, real solid harmonic
  execution remains a later implementation target, and production `src/` code
  has not started changing for task `009` or later.
- The Theory Phase Acceptance Target is satisfied for both the compressed
  complex basis and the real solid harmonic basis. The artifacts specify M2M,
  M2L, and L2L operations using invariant axis-swap matrices, fixed sign
  tables, fixed-`m` z-translation blocks, Lamb-Helmholtz operators where
  applicable, and z-axis rotations only for angle-dependent rotations.
- The required unit point-mass example is covered: source expansion, M2M, M2L,
  L2L, local evaluation at a target point, and convergence to analytic `1/r`
  as expansion order increases are demonstrated in the complex-basis chain and
  carried through the real-basis transform parity verification.
- No coordination-document conflict was found among `START_HERE.md`,
  `../MATRIX_OPERATOR_REFACTOR.md`, the completed task files, and the reviewed
  artifacts.

Fresh verification commands:

```text
julia --project=. MATRIX_OPERATOR_REFACTOR/scripts/full_m2l_composition_verify.jl
julia --project=. MATRIX_OPERATOR_REFACTOR/scripts/m2m_l2l_verify.jl
julia --project=. MATRIX_OPERATOR_REFACTOR/scripts/coefficient_buffer_layout_verify.jl
julia --project=. MATRIX_OPERATOR_REFACTOR/scripts/real_solid_harmonic_transforms_verify.jl
```

Fresh verification results:

- `full_m2l_composition_verify`: `PASS`; max composition relative error
  `1.5897417339721547e-12`; max scaled-block relative error
  `4.092898796994199e-16`; final point-mass relative error
  `3.6155107851558025e-16`.
- `m2m_l2l_verify`: `PASS`; max M2M relative error
  `1.7094382274525772e-12`; max L2L relative error
  `2.6067926229863967e-13`; max z-block relative error
  `1.3682805004681272e-13`; final point-chain relative error
  `8.556935334046337e-16`.
- `coefficient_buffer_layout_verify`: `PASS`; max complex legacy/native
  round-trip error `0.0`.
- `real_solid_harmonic_transforms_verify`: `PASS`; transform round-trip,
  z-rotation parity, M2L parity, and full M2M/M2L/L2L operator parity headline
  errors `0.0`; final point-chain relative error
  `8.556935334046337e-16`.

Gate conclusion:

- The Theory gate review work for `005` through `008` is complete.
- `008a` may be marked `Done` in `START_HERE.md`.
- `008a` must remain unapproved until a different clear-context agent performs
  approval. No Implementation task and no `008b` work may start until `008a`
  is marked both `Done` and `Approved`.

## Approval Notes

Approved after clear-context review by a different agent than the one that
completed the `008a` review notes (fresh session; did not author the Review
Notes above).

Scope reviewed (START_HERE step 6 plus the Milestone Review sanction to read
`../MATRIX_OPERATOR_REFACTOR.md`): `START_HERE.md`, this task file, the
completed dependency task files `005`–`008` and their listed artifacts under
`theory/`, `scripts/`, and `data/`, and the background design document.

Findings:

- Dependency approvals confirmed. Tasks `005`, `006`, `007`, and `008` are each
  marked `[x][x]` in `START_HERE.md` and each carries an Approval Notes section
  signed by a different clear-context agent than its completer.
- Theory Phase Acceptance Target satisfied for both bases. The `005` (M2L),
  `006` (M2M + L2L), and `008` (real-basis forms of all three) deliverables
  specify operators using invariant axis-swap matrices, fixed sign tables,
  fixed-`m` z-translation blocks, optional Lamb-Helmholtz stages, and z-axis
  rotations only for angle-dependent rotation. The compressed complex basis and
  the real solid harmonic basis are both covered.
- Point-mass unit-strength `1/r` convergence example present and convergent:
  complex M2L chain final rel error `3.6155107851558025e-16`
  (`data/full_m2l_composition/`); complex M2M→M2L→L2L chain final rel error
  `8.556935334046337e-16` (`data/m2m_l2l/`); real-basis M2M→M2L→L2L chain final
  rel error `8.556935334046337e-16` with operator parity errors `0.0`
  (`data/real_solid_harmonic/`).
- Hard phase gate respected. `git status src/` is empty; no production code was
  modified during the Theory phase.
- No coordination-document conflict. `../MATRIX_OPERATOR_REFACTOR.md` (Goal,
  Theory Sequence 1–8, Operator Layer, real-solid-harmonics-later) agrees with
  the index task ordering and the task-file deliverables.

Independent fresh re-verification (re-ran all four scripts this session; every
headline number reproduced the recorded values exactly):

- `full_m2l_composition_verify`: `PASS`; max composition rel error
  `1.5897417339721547e-12`; max scaled-block rel error
  `4.092898796994199e-16`; final point-mass rel error
  `3.6155107851558025e-16`.
- `m2m_l2l_verify`: `PASS`; max M2M rel error `1.7094382274525772e-12`; max L2L
  rel error `2.6067926229863967e-13`; max z-block rel error
  `1.3682805004681272e-13`; final point-chain rel error
  `8.556935334046337e-16`.
- `coefficient_buffer_layout_verify`: `PASS`; max complex round-trip error
  `0.0`.
- `real_solid_harmonic_transforms_verify`: `PASS`; transform round-trip,
  z-rotation parity, M2L parity, and full M2M/M2L/L2L operator parity headline
  errors `0.0`; final point-chain rel error `8.556935334046337e-16`.

Conclusion: `008a` is approved. The Theory gate is now both `Done` and
`Approved`. The next unblocked row is `008b-implementation-replan.md`, which
must be completed and approved before task `009` or any later Implementation
task begins.
