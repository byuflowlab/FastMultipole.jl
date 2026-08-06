# 042 Milestone Review: Adaptive Octree Arc (038–041a)

## Status and Entry Gate

**Proposed follow-on item; not started.**

Entry gate: `038`, `039`, `040`, `041`, and `041a` complete and approved.

## Scope

Blocking Milestone Review for the adaptive octree arc, per the standard
Milestone Review protocol in `START_HERE.md` (read
`../MATRIX_OPERATOR_REFACTOR.md` and `START_HERE.md`, inspect the completed
task files and artifacts, confirm consistency with the background design and
gating, look for further GPU-speedup contributions, record notes, obtain
clear-context approval).

Review items specific to this arc:

1. Exact-once coverage and 2:1 balance evidence: proofs, computational
   verification breadth, adversarial distributions.
2. Operator-table reuse: confirm the V-list path introduced no new operator
   tables or strategy forks; audit any recorded deviation.
3. M2T/S2L accuracy: error bounds vs measured sampled-direct errors,
   Lamb-Helmholtz coverage, `P=4` behavior.
4. Performance verdict: multi-scale win, uniform-case non-regression, the
   default-selection recommendation from `041` audited against the `041a`
   figures and report, and whether the dense `node_at` replacement should
   lift the uniform path's `ell` cap.
4a. Reporting quality: the `041a` figures compile, match their CSVs, compare
   only at matched stated accuracy, and communicate the old approach's
   weakness and the new approach's time/memory gains without overclaiming.
5. Contract compliance: capacity/no-realloc, transfer counters, refresh
   semantics, `recenter!` interaction.
6. Consumer impact: whether FLOWVPM (or the `032` device-system API) should
   expose the adaptive policy, and what documentation is owed.

## Acceptance

Review notes recorded here, row marked Done, and clear-context approval
obtained before any downstream row starts.
