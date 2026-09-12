# 052e handoff — 2026-09-12 (supersedes 052e-handoff-prompt-2026-09-11b.md)

## Where the item stands

The 052e.2a real-simulation addendum's **v2 registered run is COMPLETE**
(launched 2026-09-11 22:58, finished 2026-09-12 ~12:00 MDT, ~13.1 h, laptop,
4 threads, all 36 solves). Gates reported to Ryan 2026-09-12:

- **G1 PASS** (all solves converged; the v1 crash case BW2-VTS-L4 ran clean
  under overlap 2.4: Gtot −3.753445 vs BW1 −3.753336).
- **G2 PASS** (in-solve = prescribed hashes all levels; L4 `47f91e5c7a904745`).
- **G3 PASS** (max Green residual 1.22e-14; max gauge defect 2.20e-16).
- **G4 FAIL on threshold** (Phase B E_q = 2.133e-1 / 1.533e-1 / 9.665e-2 /
  5.851e-2 at L1–L4: monotone ✓, finest ≤ 1e-2 ✗ by ~5.9×; decay ≈ N^-0.6;
  Phase A = 1.968e-1 / 1.385e-1 / 8.435e-2 / 4.895e-2). Registered outcome,
  reported without retuning per the LOCKED prereg. **Tier stopped pending
  Ryan's ruling.**

## Completed this session (2026-09-12)

- Results doc drafted: `052e2a-addendum-realsim-results-2026-09-11.md`
  (v2 dataset only; provenance, gate report, Γ(y) tables ×4 levels,
  P1/P2 pairwise-gap tables, P3 oracle table, P4 ratios
  7.37e-3/9.40e-3/1.285e-2/1.680e-2 growing, scalars/diagnostics, call
  chains, R-notes, failed-run/supersession history, evidence summary with
  agent recommendations, **blank ruling checklist for Ryan** at the end).
- Lab-notebook entry appended under `# 20260912` in
  `~/Dropbox/research/notebooks/journals/20260901.md` (Ryan-requested:
  outcomes only, methods equations, G1–G4 subsections with tables + two
  TikZ figures in `notebooks/img/20260912_052e2a_realsim/`
  — `eq_convergence.tex`, `gamma_L4.tex`, backing CSVs, PNGs). Checkbox
  unticked (Ryan ticks on approval).
- Γ(y) table generator: session scratchpad `gamma_tables.py` (volatile).

## Open items (in order)

1. **Ryan's ruling** on P1–P4 (CONFIRMED/REFUTED/INCONCLUSIVE) + CONTINUE
   for 052e.3 — checklist at the end of the results doc. Agent
   recommendations recorded there (P1, P2 recommend CONFIRMED; P3, P4
   flagged as Ryan's judgment calls).
2. External review requested by Ryan (2026-09-12): a separate agent to
   evaluate the process and results of this item and whether a more
   effective approach exists — prompt at
   `052e2a-approach-review-prompt-2026-09-12.md`.
3. Diagnostic scripts from the 2026-09-10 root-cause session still live
   only in the volatile /tmp scratchpad
   (`.../af0908dc-.../scratchpad/`: `diag_bw2_vts_l4.jl`+log,
   `diag_bw2_vts_l4_ov24.jl`+log, `relax_nan_test.jl`+log) — offer stands
   to copy into MATRIX_OPERATOR_REFACTOR/.
4. Notebook entries still owed (offer only, approval + verbosity choice
   required): Tier 0B, Tier 0B-R, production integration, addendum lock.
   (The failed registered run + diagnosis + supersession entry was
   explicitly EXCLUDED from the 2026-09-12 entry per Ryan: outcomes only.)
5. Later queue unchanged: 052e.3 hybrid fixture (gated on ruling);
   052e.1 regression completion after 052b closes; 052e.6 gauge design
   study.

## Cautions (carried forward)

- LOCKED prereg (`052e2a-addendum-realsim-preregistration-2026-09-08.md`)
  append-only; changes need a new dated supersession. Supersession record:
  `052e2a-addendum-realsim-supersession-2026-09-11.md`.
- Frozen sha-registered scripts (`tier0b_052e2a_bordered_formulation.jl`,
  `tier0br_052e2b_householder_parity.jl`) and both addendum harnesses
  (`addendum_052e2a_realsim_2026-09-09.jl`, `..._2026-09-11.jl`): do not edit.
- v1 FAILED-run evidence preserved: `data/052e2a-addendum-realsim/`.
  v2 data: `data/052e2a-addendum-realsim-v2/` (gates.txt is the canonical
  numbers source). Never read `data/**` CSVs directly; summarize by script.
- Nothing committed in either repo; FastMultipole (branch
  flowpanel-20260817) heavily dirty with unrelated work — touch only 052e
  files. FLOWPanel dirty with unrelated BRAINSTORM files — leave alone.
- Before touching FLOWPanel code: read its AGENTS.md, CLAUDE.md,
  agent_policies/WORKFLOW.md (+ TESTING.md before testing). ≤ 4 threads.
- Laptop runs = formulation-proof tier, not official campaigns.
