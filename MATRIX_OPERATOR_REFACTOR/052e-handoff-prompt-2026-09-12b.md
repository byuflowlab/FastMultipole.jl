# 052e handoff — 2026-09-12b (context reset; supersedes 052e-handoff-prompt-2026-09-12.md)

## Where the item stands

**052e.2a (real-simulation addendum) is CLOSED with a ruling.** The v2
registered run (harness `scripts/addendum_052e2a_realsim_2026-09-11.jl`,
sha `e0e76531e492`, ~13.1 h, laptop, 4 threads, 36/36 solves) completed
2026-09-12. Gates: **G1/G2/G3 PASS; G4 FAIL on threshold** (Phase B E_q
monotone 2.133e-1 → 5.851e-2 over L1–L4, vs ≤1e-2 gate; decay ≈ N^-0.6) —
reported to Ryan with raw values, no retuning.

**Ryan's ruling (2026-09-12): P1–P4 all CONFIRMED ("pass"); CONTINUE to
052e.3.** Recorded in the ruling section of
`052e2a-addendum-realsim-results-2026-09-11.md`. The ruling accepts the
monotone convergence evidence; G4 stands as a recorded threshold FAIL.

## Authoritative documents

- Results (ruled): `052e2a-addendum-realsim-results-2026-09-11.md` —
  gate report, Γ(y)/gap/oracle/P4 tables, diagnostics, ruling, and links
  to the external review docs.
- LOCKED prereg: `052e2a-addendum-realsim-preregistration-2026-09-08.md`;
  supersession: `052e2a-addendum-realsim-supersession-2026-09-11.md`.
- Data: `data/052e2a-addendum-realsim-v2/` (`gates.txt` = canonical
  numbers). v1 FAILED-run evidence preserved in
  `data/052e2a-addendum-realsim/`. Never read `data/**` CSVs directly.
- Lab notebook: outcomes entry appended under `# 20260912` in
  `~/Dropbox/research/notebooks/journals/20260901.md`; figures in
  `~/Dropbox/research/notebooks/img/20260912_052e2a_realsim/`. Checkbox
  NOT ticked (Ryan ticks).

## External approach review (2026-09-12) — no course change

Ryan's disposition: **do not change course now; may return to these if
issues arise.** The three review documents:

1. `052e2a-approach-review-2026-09-12.md` — Green reconstruction,
   convergence, gauges, iterative conditioning; recommends preconditioned
   gauge-consistent GMRES + direct panel-wake potential evaluation;
   sketches an (incomplete) second-kind source-Neumann lifting formulation.
2. `052e2a-unsteady-pressure-followup-2026-09-12.md` — unsteady pressure:
   surface Euler reconstruction for rotational particle wakes, multigrid
   preconditioning, six precomputed adjoint solves for rigid-body loads;
   uniform pressure-gauge shifts cancel from total closed-body loads.
3. `052e2a-vortical-field-layer-potential-clarification-2026-09-12.md` —
   caveat: the Green-reconstructed trace enforces impermeability correctly
   even when it is not the physical wake potential, so its time derivative
   is NOT automatically valid for Bernoulli pressure (relevant when 052e
   eventually touches unsteady loads).

## Next actions (in order)

1. **Start 052e.3 (hybrid fixture)** — the CONTINUE target. Scope it
   first: query the `refactor-docs-librarian` agent for the 052e plan's
   definition of 052e.3 (the 052e plan/handoff series and theory note
   `052e-theory-velocity-to-potential-trace.md`), then propose a
   preregistration to Ryan before any implementation. Prereg discipline
   as before: LOCKED once ruled, gates + predictions registered up front,
   registered runs nohup-detached with logs + gates.txt under
   `data/052e3-*/`.
2. Offer (do not write) the owed notebook entries: Tier 0B, Tier 0B-R,
   production integration, addendum lock. Approval + verbosity choice
   required first. (The 052e.2a outcomes entry is already written; the
   failed-run/diagnosis story was excluded per Ryan: outcomes only.)
3. Offer to copy the 2026-09-10 root-cause diagnostic scripts out of the
   volatile /tmp scratchpad
   (`/private/tmp/claude-502/-Users-ryan-Dropbox-research-projects-FastMultipole/af0908dc-5375-40b5-9cc1-390a0e87a671/scratchpad/`:
   `diag_bw2_vts_l4.jl`+log, `diag_bw2_vts_l4_ov24.jl`+log,
   `relax_nan_test.jl`+log) into MATRIX_OPERATOR_REFACTOR/.
4. Nothing is committed in either repo — when Ryan asks, stage/commit only
   052e files.

## Later queue

- 052e.1 regression completion (blocked until 052b closes).
- 052e.6 gauge design study.
- Possible returns flagged by the external review (GMRES preconditioning,
  direct W1 potential evaluation, source-Neumann lifting formulation,
  acceleration-potential pressure) — only if issues arise.

## Cautions (carried forward)

- LOCKED prereg append-only; changes need a new dated supersession.
- Frozen sha-registered scripts (`tier0b_052e2a_bordered_formulation.jl`,
  `tier0br_052e2b_householder_parity.jl`) and both addendum harnesses: do
  not edit.
- FastMultipole (branch flowpanel-20260817) heavily dirty with unrelated
  work — touch only 052e files. FLOWPanel dirty with unrelated BRAINSTORM
  files — leave alone.
- Before touching FLOWPanel code: read its AGENTS.md, CLAUDE.md,
  agent_policies/WORKFLOW.md (+ TESTING.md before testing). ≤ 4 threads
  locally.
- Laptop runs = formulation-proof tier; official campaigns need git
  worktrees + annotated campaign tags per global policy.
- Never read `data/**` CSVs or MATRIX_OPERATOR_REFACTOR binary/CSV data
  directly; summarize by script. Delegate test/script runs to
  `julia-test-runner`; doc questions to `refactor-docs-librarian`.
