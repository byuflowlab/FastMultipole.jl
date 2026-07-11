# 019a Final Roadmap Milestone Review

## Objective

Review Implementation tasks `017` through `024` and confirm the completed
Matrix Operator Refactor still matches the background roadmap.

## Dependencies

- `008b-implementation-replan.md`
- `008c-implementation-performance-baseline.md`
- `017-impl-flat-coefficient-buffers.md`
- `018-impl-real-solid-harmonic-basis.md`
- `019-impl-operator-performance-tuning.md`
- `019b-exploratory-smallp-fallback-and-channel-layout.md`
- `022-impl-gpu-device-resident-m2l.md`
- `023-impl-production-integration.md`
- `024-impl-operator-ab-benchmark.md`

## Required Reading

- `START_HERE.md`
- `../MATRIX_OPERATOR_REFACTOR.md`
- Completed task files listed above
- Artifacts and production files listed by the completed task files

## Artifacts or Production Surface

Review the production files, tests, benchmarks, generated artifacts, and final
notes listed by tasks `017` through `024`.

## Deliverables

- Final roadmap-alignment notes recorded in this file
- Any required coordination-document fixes identified before the refactor is
  considered complete
- **Final go/no-go (feasibility scoped earlier in `013a`): porting the old
  per-interaction error machinery onto the new expansion operators.** Using the
  `013a` feasibility finding and the `019` performance-tuning evidence, decide
  whether to port the dynamic-`P` / `get_P` / `predict_error` machinery onto the new
  operators or leave the two paths independent (old ops + old error machinery; new
  ops + constant-`P` interaction-list stencil). Record the decision and rationale
  here.
- **Review `008b-implementation-replan.md` (including its re-plan addenda) and
  confirm all recorded decisions and feedback have been incorporated** into the
  completed refactor and coordination documents. Note any gaps and the required
  fixes here.
- **Confirm the small-`P` / tiny-batch fallback and channel-layout decisions from
  `019b`.** These were resolved in `019b` (exploratory benchmark plus user
  discussion). Confirm the chosen fallback policy and padded-vs-ragged `chi` layout
  are implemented and consistent with the coordination documents; note any gaps.
- **Confirm the M2L operator recommendation from `024`.** Record the final
  per-platform recommendation between `MaterializedYRotationM2L` and
  `FactoredRotationM2L`, and note any crossover regimes or integration caveats.

## Verification

Confirm completed work matches the background design, hard phase gate, and task
ordering. If `START_HERE.md`, a task file, and `../MATRIX_OPERATOR_REFACTOR.md`
disagree, stop and require a coordination-document fix.

## Approval Notes

To be filled by a different agent after review notes and verification are
complete.
