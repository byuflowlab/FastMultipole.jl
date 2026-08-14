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

## Deferred Candidate To Consider (user direction 2026-08-13)

**Dual-grid two-pass nearfield: shared Morton hierarchy with separately
selected primary and correction leaf sets.** Raised during the 037a close-out
discussion; consider it in this review, do not implement before then.

Idea: keep one Morton sort at the finest required depth; form the primary
(singular direct + FMM) grid from short prefixes and finer correction-only
bins from longer prefixes. Classify correction bins against the deficit
annulus `rho_c < r/sigma <= rho_t` using conservative AABB gaps and per-bin
`sigma_min`/`sigma_max`: skip bins fully inside `rho_c` or fully outside
`rho_t`, apply the deficit probe-free to bins fully inside the annulus, and
retain the body-level predicate only for boundary-intersecting bins. The
whole-ball variant (singular primary everywhere plus deficit over
`r/sigma <= rho_t`) stays excluded: it reintroduces the `031a` §6.1
singular-plus-deficit cancellation that `rho_c` exists to avoid.

Evidence at deferral: on the uniform AR=5 wake at the 037a operating point the
ceiling is too small to matter — eliminating every rejected probe (4.73B
examined vs 0.947B accepted at `n = 1e6`) bounds the saving under ~4.9 ms on
an 87.12 ms solve vs the 83.31 ms baseline (~1.3%, vs the 5% promotion gate),
and the accepted-pair deficit work itself is untouched by better traversal
(that is the `037c` mesh lever). The idea's plausible habitat is exactly what
this arc builds: heterogeneous per-cell `sigma` (where global `sigma_max`
inflates correction coverage), clustered/rolled-up wakes, and a shared
finest-Morton representation that yields both adaptive leaves and fine
correction bins without a second sort. Review question: does the measured
adaptive machinery (`038`–`041a`), plus any `037b`/`037c` verdicts, make this
lever worth staging as a row — benchmarked against the simpler probe-free
adaptive regularized U-list — or should it be closed as subsumed?

## Acceptance

Review notes recorded here, row marked Done, and clear-context approval
obtained before any downstream row starts.
