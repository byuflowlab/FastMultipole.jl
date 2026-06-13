# 008c Implementation Performance Baseline

## Objective

Benchmark current production paths, inventory allocation/storage behavior, and
record approved operator-theory design constraints before production
implementation begins.

## Dependencies

- All Theory Phase rows in `START_HERE.md`
- `008a-milestone-review-theory-005-008.md`
- `008b-implementation-replan.md`

## Required Reading

- `START_HERE.md`
- `../MATRIX_OPERATOR_REFACTOR.md`
- Approved Theory task files
- Approved Theory artifacts and scripts listed by the Theory task files
- Completed `008b-implementation-replan.md`
- Current production translation, evaluation, and benchmark entry points

## Artifacts or Production Surface

This is a pre-implementation benchmark and design-gate task. No production code
changes may start until this task is complete and approved.

Benchmark and data-structure review work may use current production paths,
approved Theory scripts, and benchmark/result artifacts under
`MATRIX_OPERATOR_REFACTOR/data/` if needed. Coordination updates may edit
`008b-implementation-replan.md`, Implementation task files, and/or
`START_HERE.md` only if benchmark or storage/allocation results require a
changed design, scope, risk assessment, or task order.

## Deliverables

- Benchmark commands for current production/operator-theory baselines
- Environment notes sufficient to reproduce the measurements
- Baseline summaries for CPU single-thread paths
- Baseline summaries for CPU multithread paths where measurable
- Inventory of current production coefficient, operator, cache, and scratch data
  structures used by translation/evaluation paths
- Baseline allocation/storage review covering allocation counts, retained
  storage, scratch reuse, and temporary buffer pressure for current production
  paths, measured where practical and estimated where measurement is not
  practical
- Review of approved Theory layouts, especially `007` coefficient-buffer layout,
  for minimum required storage, batch layout requirements, aliasing rules, and
  CPU/GPU tradeoffs
- Explicit storage/allocation budgets or constraints that Implementation tasks
  must respect
- A per-stage decision on operator form — dense-materialized matrices (applied
  via `mul!`/GEMM) versus recurrence-wrapped operators (today's `O(p)`
  recurrences behind the operator API) — for the z-rotation, axis-swap,
  fixed-`m` z-translation, and Lamb-Helmholtz stages, with benchmark-backed
  rationale. This decision is a constraint that Implementation tasks `009`–`016`
  inherit. (Deferred here from the `008b` re-plan.)
- GPU-relevant design tradeoffs and constraints where measurable
- Design implications for Implementation task order, scope, or risk
- Any needed coordination updates to `008b-implementation-replan.md`,
  Implementation task files, and/or `START_HERE.md` if benchmark or
  storage/allocation review changes task order, design, or risk
- Clear-context approval before task `009` or any later Implementation task
  starts

## Verification

Confirm that no production code work has started for task `009` or any later
Implementation task. Confirm benchmark commands, environment notes, baseline
summaries, allocation/storage inventory, storage constraints, CPU/GPU design
notes, design implications, and any coordination updates are recorded before
requesting approval.

## Approval Notes

To be filled by a different agent after benchmark/design notes and verification
are complete.
