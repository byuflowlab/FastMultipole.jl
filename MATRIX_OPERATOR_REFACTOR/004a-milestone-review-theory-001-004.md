# 004a Milestone Review Theory 001-004

## Objective

Review Theory tasks `001` through `004` against the background design and
coordination rules before downstream Theory work begins.

## Dependencies

- `001-theory-z-rotation-operators.md`
- `002-theory-m2l-z-translation-scaling.md`
- `003-theory-lamb-helmholtz-operator-form.md`
- `004-theory-axis-swap-conventions.md`

## Required Reading

- `START_HERE.md`
- `../MATRIX_OPERATOR_REFACTOR.md`
- Completed task files listed above
- Artifacts listed by the completed task files

## Artifacts or Production Surface

Review artifacts under `theory/`, `scripts/`, and `data/` that are listed by
tasks `001` through `004`. No production code changes are part of this review.

## Deliverables

- Roadmap-alignment notes recorded in this file
- Any required coordination-document fixes identified before later work starts

## Verification

Confirm completed work matches the background design, hard phase gate, and task
ordering. If `START_HERE.md`, a task file, and `../MATRIX_OPERATOR_REFACTOR.md`
disagree, stop and require a coordination-document fix.

## Review Notes (2026-06-12)

Required reading completed: `../MATRIX_OPERATOR_REFACTOR.md`, `START_HERE.md`,
task files `001`–`004`, the four theory artifacts under `theory/`, the four
verification scripts under `scripts/`, and the four generated
`verification_summary.md` files under `data/`.

### Verification re-runs

All four scripts were re-run with
`julia --project=. MATRIX_OPERATOR_REFACTOR/scripts/<name>_verify.jl` and all
reported `PASS` with values identical to the recorded summaries:

- `z_rotation_verify.jl`: max forward error `3.552713678800501e-15`, max
  back/inverse accumulation error `3.1086244689504383e-15`, zero `m = 0`
  errors.
- `m2l_z_translation_verify.jl`: max absolute error `3.469446951953614e-18`,
  max relative error `4.823830462317249e-16`, zero inactive-channel overwrite
  error.
- `lamb_helmholtz_operator_verify.jl`: max absolute and relative errors `0.0`.
- `axis_swap_verify.jl`: all five tracked error categories `0.0`.

Each script was inspected and confirmed to compare derived operator forms
against the corresponding production functions (`rotate_z!`, `back_rotate_z!`,
`translate_multipole_to_local_z!`, `transform_lamb_helmholtz_multipole!`,
`transform_lamb_helmholtz_local!`, `rotate_multipole_y!`, `rotate_local_y!`,
`back_rotate_multipole_y!`, `back_rotate_local_y!`, `update_Hs_π2!`).

### Roadmap alignment

- Hard phase gate intact: `git status` shows no modifications under `src/` or
  `test/`; all artifacts are confined to `theory/`, `scripts/`, and `data/`.
- Task ordering followed: `001` had no dependencies; `002`–`004` each depended
  only on `001`; all four are Done and Approved with clear-context approval
  notes recorded by a separate agent.
- `001` matches the background "Operator Layer" z-rotation target: real `2x2`
  blocks per `(n, m)`, forward overwrite, conjugate-phase back rotation with
  accumulation, and a storage-light fused `C/S` representation suitable for
  flat buffers and GPU batching.
- `002` matches the fixed-`m` dense-block M2L form `K_m[n, n'] =
  (n + n')! / t^(n + n' + 1)`, with production-matching recurrence evaluation,
  overwrite semantics, and indexing tied to `harmonic_index(n, m)`.
- `003` matches the requirement that Lamb-Helmholtz transforms be first-class
  linear operators: sparse/banded channel coupling (no cross-`m` mixing,
  same-degree phi-from-chi, nearest-neighbor chi-from-chi), overwrite
  semantics, and pipeline placement after the z translation.
- `004` establishes the invariant-axis-swap decomposition
  `T_n(theta) = S_n Z_n(theta) S_n^{-1}` required by the Theory Phase
  Acceptance Target: all non-z rotation effects are angle-independent
  matrices, with z-axis rotations as the only angle-dependent terms. The
  production `pi` z-axis convention, `zeta`/`eta` sign tables, and
  reset-versus-accumulate semantics are documented and verified.
- No disagreement was found between `START_HERE.md`, the task files, and
  `../MATRIX_OPERATOR_REFACTOR.md`. No coordination-document fix is required.

### Findings for downstream Theory tasks

1. **Extreme-distance scaled-form coverage is still open.** The background
   design requires the binomial-scaled factorization
   `K_m(t) = D_L(t) Khat_m D_M(t)` to be tested against the existing
   recurrence "including high expansion orders and extreme distances such as
   `t ≈ 1e-3` and `t ≈ 1e3`" before implementation. Task `002` verified the
   unscaled fixed-`m` form at `t ∈ [1.75, 13]` and `P <= 9`, and its artifact
   describes the scaled block only as future cache metadata. This is not a
   document conflict — the Theory phase is still open — but the scaled-form
   derivation and extreme-distance verification must land in task `005` (or an
   amendment to `002`) before `008a` can approve the Theory gate.
2. The Theory Phase Acceptance Target's unit-point-mass end-to-end convergence
   example (source expansion → M2M/M2L/L2L → evaluation → `1/r` convergence)
   is not yet covered by any artifact; tasks `005`–`008` must produce it
   before `008a`.

### Disposition

Tasks `001`–`004` are consistent with the background design, hard phase gate,
and task ordering. Review complete; row `004a` marked Done. Downstream Theory
work may proceed after clear-context approval of this review.

## Approval Notes

Approved by clear-context review (different agent from the completing
reviewer), 2026-06-12.

Materials read: `START_HERE.md`, this task file including the Review Notes,
`../MATRIX_OPERATOR_REFACTOR.md`, task files `001`–`004` (including their
verification and approval notes), the four theory artifacts
(`theory/z-rotation-operators.md`, `theory/m2l-z-translation-scaling.md`,
`theory/lamb-helmholtz-operator-form.md`, `theory/axis-swap-conventions.md`),
and the regenerated `data/*/verification_summary.md` files.

Independent verification re-runs (from repo root, all `PASS` with values
identical to the Review Notes):

- `julia --project=. MATRIX_OPERATOR_REFACTOR/scripts/z_rotation_verify.jl`:
  max forward error `3.552713678800501e-15`, max back/inverse accumulation
  error `3.1086244689504383e-15`, zero `m = 0` errors.
- `julia --project=. MATRIX_OPERATOR_REFACTOR/scripts/m2l_z_translation_verify.jl`:
  max absolute error `3.469446951953614e-18`, max relative error
  `4.823830462317249e-16`, inactive-channel overwrite error `0.0`.
- `julia --project=. MATRIX_OPERATOR_REFACTOR/scripts/lamb_helmholtz_operator_verify.jl`:
  max absolute and relative errors `0.0`.
- `julia --project=. MATRIX_OPERATOR_REFACTOR/scripts/axis_swap_verify.jl`:
  all five tracked error categories `0.0`.

Independent checks confirmed:

- All four scripts compare derived operator forms against the production
  functions named in the Review Notes (verified by inspection of the scripts'
  `FastMultipole.` call sites).
- Hard phase gate intact: `git status --porcelain` shows no changes under
  `src/` or `test/`; all artifacts are confined to `theory/`, `scripts/`, and
  `data/`.
- Task ordering followed: `001` (no dependencies) then `002`–`004` (each
  depending only on `001`), all Done and Approved with separate-agent approval
  notes; `004a` dependencies were therefore satisfied before this review.
- The Review Notes' alignment conclusions match `../MATRIX_OPERATOR_REFACTOR.md`
  (z-rotation 2x2 block and C/S fused form, fixed-`m` `K_m[n,n'] =
  (n+n')!/t^(n+n'+1)` with production recurrence, sparse/banded Lamb-Helmholtz
  coupling and post-z-translation placement, and the invariant
  `T_n(theta) = S_n Z_n(theta) S_n^{-1}` axis-swap decomposition with the
  production `pi` convention). No disagreement among `START_HERE.md`, the task
  files, and the background design was found.
- Finding 1 is accurate and correctly classified as open rather than a
  conflict: the regenerated M2L summary covers only the unscaled form at
  `t ∈ [1.75, 13]`, `P <= 9`, and the `002` artifact defers the binomial-scaled
  factorization to future cache metadata; the background design's
  extreme-distance (`t ≈ 1e-3`, `t ≈ 1e3`) scaled-form test remains an
  obligation for `005` (or an amendment to `002`) before `008a`.
- Finding 2 is accurate: no artifact from `001`–`004` provides the
  unit-point-mass end-to-end convergence example required by the Theory Phase
  Acceptance Target; it remains an obligation for `005`–`008` before `008a`.

Decision: approved. The `004a` row in `START_HERE.md` is marked Approved.
