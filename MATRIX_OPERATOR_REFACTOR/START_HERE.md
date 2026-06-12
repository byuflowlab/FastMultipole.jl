# Matrix Operator Refactor Index

## START HERE

This file is the only required first-read coordination document for Matrix
Operator Refactor work.

Routine task protocol:

1. Read this `START_HERE.md` first.
2. Select the first row, in table order, whose dependencies are complete and
   clear-context approved.
3. Open only the selected task file.
4. Do not read sibling task files.
5. Do not read `../MATRIX_OPERATOR_REFACTOR.md` unless the selected task
   explicitly requires it or the selected task is a Milestone Review.
6. For clear-context approval, read only this `START_HERE.md`, the completed task
   file, the artifacts or production files listed by that task, and the
   verification notes.

No separate active-task pointer file should be added. The first unblocked row
in this index is the active task selection mechanism.

## Source Of Truth

Current user instructions are the first source of truth in all cases. 
This index is the second source of truth for task order, status, blockers, Milestone
Reviews, and the hard phase gate. Task files may contain task-local
requirements only; they must not redefine global policy.

If this index, a task file, and `../MATRIX_OPERATOR_REFACTOR.md` disagree
during a Milestone Review, the review must stop and require a
coordination-document fix before further task work continues.

Clear-context approval must be performed by a different agent after the task is
finished. The completing agent must not approve its own work.

## Hard Phase Gate

All Theory Phase rows, including Theory Milestone Reviews, must be marked both
`Done` and `Approved` before any Implementation Phase row starts.

Theory work may create or edit artifacts under:

- `MATRIX_OPERATOR_REFACTOR/theory/`
- `MATRIX_OPERATOR_REFACTOR/scripts/`
- `MATRIX_OPERATOR_REFACTOR/data/`

Theory work must not modify production FastMultipole code under `src/`.

For each concept, derivation precedes script, script precedes generated data,
and all related Theory artifacts precede approval. Implementation may touch
`src/` only after the full Theory gate is approved.

After the Theory gate is approved, `008b-implementation-replan.md` must be
completed and approved before task `009` or any later Implementation task
starts.

## Milestone Reviews

Milestone Reviews are blocking tasks. No later normal task may start until the
preceding Milestone Review is complete and approved.

Each Milestone Review requires the reviewing agent to:

1. Read all of `../MATRIX_OPERATOR_REFACTOR.md`.
2. Read this `START_HERE.md`.
3. Inspect completed task files and their listed artifacts since the previous
   milestone.
4. Confirm work still matches the background design, hard phase gate, and task
   ordering.
5. Record review notes in the Milestone Review task file and mark the row
   `Done`.
6. Get clear-context approval before downstream tasks continue.

## Theory Phase

| Done | Approved | Task | Summary | Blocking |
| --- | --- | --- | --- | --- |
| [ ] | [ ] | `001-theory-z-rotation-operators.md` | Derive z-rotation blocks, inverse blocks, and overwrite/accumulate semantics. | None |
| [ ] | [ ] | `002-theory-m2l-z-translation-scaling.md` | Derive M2L fixed-`m` z-translation matrices and stable distance scaling. | `001` |
| [ ] | [ ] | `003-theory-lamb-helmholtz-operator-form.md` | Derive multipole/local Lamb-Helmholtz operator form and channel coupling. | `001` |
| [ ] | [ ] | `004-theory-axis-swap-conventions.md` | Derive invariant axis-swap signs and active/passive rotation conventions. | `001` |
| [ ] | [ ] | `004a-milestone-review-theory-001-004.md` | Milestone Review for Theory tasks `001` through `004`. | `001`, `002`, `003`, `004` |
| [ ] | [ ] | `005-theory-full-m2l-composition.md` | Derive the complete M2L operator composition from approved component theory. | `004a`, `002`, `003`, `004` |
| [ ] | [ ] | `006-theory-m2m-l2l-extensions.md` | Extend the component theory to M2M and L2L pipelines. | `005` |
| [ ] | [ ] | `007-theory-coefficient-buffer-layout.md` | Specify coefficient-buffer layout, indexing, and typed view requirements. | `005`, `006` |
| [ ] | [ ] | `008-theory-real-solid-harmonic-transforms.md` | Derive complex-to-real and real-to-complex transform conventions and tests. | `001`, `007` |
| [ ] | [ ] | `008a-milestone-review-theory-005-008.md` | Milestone Review for Theory tasks `005` through `008`, immediately before Implementation can begin. | `005`, `006`, `007`, `008` |

## Implementation Re-Plan Gate

This required planning task is not an Implementation task. It must be completed
and approved after the Theory gate is approved and before any production code
work starts.

| Done | Approved | Task | Summary | Blocking |
| --- | --- | --- | --- | --- |
| [ ] | [ ] | `008b-implementation-replan.md` | User-in-the-loop Implementation re-plan after Theory is approved and before production code work begins. | All Theory rows, `008a` |

## Implementation Phase

Every Theory Phase row above is a blocker for every row in this section. Do not
start any Implementation task until all Theory rows are marked both `Done` and
`Approved`. Do not start any Implementation task until
`008b-implementation-replan.md` is also marked both `Done` and `Approved`.

Implementation task files must list relevant approved Theory dependencies by
filename. This does not narrow the gate: every Theory task and every preceding
Milestone Review blocks every Implementation task.

| Done | Approved | Task | Summary | Blocking |
| --- | --- | --- | --- | --- |
| [ ] | [ ] | `009-impl-basis-and-operator-cache-types.md` | Define basis and operator-cache types without changing production translation calls. | All Theory rows, `008b` |
| [ ] | [ ] | `010-impl-z-rotation-operators.md` | Implement explicit z-rotation operators and parity tests. | All Theory rows, `008b`, `009` |
| [ ] | [ ] | `011-impl-m2l-z-translation-blocks.md` | Implement fixed-`m` M2L z-translation blocks with approved scaling. | All Theory rows, `008b`, `009` |
| [ ] | [ ] | `012-impl-lamb-helmholtz-operators.md` | Implement Lamb-Helmholtz transform operators and parity tests. | All Theory rows, `008b`, `009` |
| [ ] | [ ] | `012a-milestone-review-impl-009-012.md` | Milestone Review for Implementation tasks `009` through `012`. | `008b`, `009`, `010`, `011`, `012` |
| [ ] | [ ] | `013-impl-axis-swap-operators.md` | Implement invariant axis-swap operators and y-rotation parity tests. | All Theory rows, `008b`, `012a`, `010` |
| [ ] | [ ] | `014-impl-full-m2l-operator-pipeline.md` | Compose the full M2L operator pipeline and test against current production behavior. | All Theory rows, `008b`, `013`, `010`, `011`, `012` |
| [ ] | [ ] | `015-impl-axis-swap-benchmarks.md` | Benchmark invariant axis-swap composition and full M2L operator paths. | All Theory rows, `008b`, `014` |
| [ ] | [ ] | `016-impl-m2m-and-l2l-operator-pipelines.md` | Extend the operator structure to M2M and L2L. | All Theory rows, `008b`, `014`, `015` |
| [ ] | [ ] | `016a-milestone-review-impl-013-016.md` | Milestone Review for Implementation tasks `013` through `016`. | `008b`, `013`, `014`, `015`, `016` |
| [ ] | [ ] | `017-impl-flat-coefficient-buffers.md` | Introduce flat coefficient buffers and typed views after the operator API is stable. | All Theory rows, `008b`, `016a`, `016` |
| [ ] | [ ] | `018-impl-real-solid-harmonic-basis.md` | Add real-basis transforms and evaluate native real-basis execution. | All Theory rows, `008b`, `017` |
| [ ] | [ ] | `018a-milestone-review-final-roadmap.md` | Final roadmap Milestone Review after Implementation tasks `017` through `018`. | `008b`, `017`, `018` |
