# 036 Milestone Review: Integration Phase

## Status and Entry Gate

**Added by user request on `2026-08-04`; renumbered by roadmap review on
`2026-08-05`.** Not started.

Entry gate: Integration rows `031`, `031a`, `032`, `032a`, `033`, `034`, and
`035` must all be Done and clear-context approved. This is a blocking
Milestone Review: no later row in this or a future phase may start until this
row is complete and clear-context approved.

## Objective

Perform the standard Milestone Review duties from `START_HERE.md` and render
verdicts on the Integration Phase goals.

Review scope:

1. **Interface generality:** verify the shipped device-system interface is
   general-consumer-first and that a third party can connect without reading
   FastMultipole internals.
2. **Correctness and accuracy:** confirm `034` correctness and that every
   `035` winner and speedup numerator passed sampled velocity RMS `≤1e-3`.
   Confirm Jacobian RMS was logged for every reported configuration and was
   clearly labeled diagnostic rather than gating.
3. **Speedup evidence:** verify `035` contains the sole final report, including
   the wake-at-cube-parameters result, stage profiles, eligible speedups, the
   per-U/J-solve `030` ratio, the separate RK3-step cost, and the optimization
   ledger. Ensure no speedup ratio uses a `033` baseline that failed the
   velocity tolerance.
4. **Performance closure:** verify every credible lever with at least 5%
   expected end-to-end U/J-solve gain was implemented or ruled out with
   evidence, and remaining sub-5% items are recorded without extending the
   campaign.
5. **Downstream compatibility:** confirm FLOWVPM CPU users and the
   FLOWUnsteady/VortexLattice-facing public API are unaffected and CPU tests
   pass at the final `gpu-full` head.
6. **Cross-repo hygiene:** confirm FLOWVPM commits live on `gpu-full`,
   FastMultipole commits live here, and cross-repo change pairs are recorded.
7. **Improvement hunt:** record any further significant opportunities as
   follow-on proposals; do not silently expand this completed phase.

## Dependencies and Reading

- `031`, `031a`, `032`, `032a`, `033`, `034`, and `035`, all Done and
  clear-context approved.
- Read all of `../MATRIX_OPERATOR_REFACTOR.md`, this `START_HERE.md`, every
  Integration task and its listed artifacts, `../FLOWVPM.jl/CLAUDE.md`, and
  the relevant final state of FLOWVPM's `gpu-full` branch.

## Review Notes

(To be filled by the reviewing agent.)

