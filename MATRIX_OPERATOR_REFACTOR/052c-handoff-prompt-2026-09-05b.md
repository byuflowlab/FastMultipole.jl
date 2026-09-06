# 052 handoff — 2026-09-05 evening (052c CLOSED; discussion mode next)

## Mode for this session

Ryan wants to tackle the open decisions below by DISCUSSING each one
together until he approves a course of action (possibly via plan mode).
Do NOT launch runs, delete, push, or edit ledgers until he rules.
Bring one issue at a time, with the relevant facts and a recommendation.

## What just closed (2026-09-05, all committed on FastMultipole
`flowpanel-20260817`: ef67a214, 0a1f7b43, a1ecc7d5)

- **052c trial-2 (expint) PASSED**: job 13592503, 1080-step acceptance,
  `WAKE_EXPINT=true`, guards OFF. No sigma collapse (min_sigma bottom
  8.87e-5 m at step 1051, ~2% plateau held by integrator positivity).
  All locked gates PASS, tighter than guarded trial-1e (CT 5.655e-4 vs
  6.565e-4; Gamma M2 max 1.134e-3 vs 1.317e-3; RMS 3.331e-4 vs
  4.484e-4; ceilings 1.8e-3/2.93e-3/1.50e-3). Full results in
  `052c-sigma-experiments-2026-08-26.md`.
- **Ruling executed**: expint = 052c default for GPU rotor acceptance;
  trial-1 guard pair (dtz_cap=0.5, floor_frac=0.01) = documented
  fallback; OFAT candidates folded into
  `053-defaults-enumeration-draft-2026-09-03.md`.
- **Silos + debris retired**: all 052 silo checkouts/envs deleted from
  orc `~`. Campaign data (61G incl. both 1080-step runs) moved to
  `~/projects/FLOWPanel.jl/data/` (stale Aug-27 `fm052d_gpu_1080_t1`
  renamed `.prev.20260827`; silo logs in `data/retired_052_silos/`).
  Only `~/fm052depot-gh200` (wt052 depot) and `~/wt052/` remain.
  wt052 launcher `~/projects/launchers/fp052c_expint_wt052_run.sh`
  (WAKE_EXPINT=true hardwired) is the ONLY 052c launch path.
- **Campaign tags created** (annotated, orc unified repos):
  `campaign/052-expint-20260905` at FLOWVPM 3315b22 / FLOWPanel 4e6b5b7 /
  FastMultipole 3da58a1a.

## Issue 1 — push the campaign tags? (facts gathered, needs ruling)

Tags exist only on orc. The pinned commits are on orc-local
`unified-052` branches; **no remote branch contains them** (verified
`git branch -r --contains` empty in all three repos). Remotes are the
byuflowlab GitHub repos over https. Implications:
- Pushing a tag also uploads its commit objects → publishes the
  unified-052 WIP lineage to the lab GitHub repos.
- https remotes on orc may lack push credentials (untested).
Options: (a) push tags now (visible WIP), (b) push unified-052 branches
+ tags together (deliberate publication), (c) keep tags orc-local until
the unified branches are ready to publish (policy only requires tags to
exist and be cited; push is "safe by name", not mandatory).
Recommendation: (c) for now; revisit when unified-052 merges.

## Issue 2 — 052e tolerances T1–T5 ratification (Ryan decision)

## Issue 3 — 052b rulings (Ryan decisions)

- 1r hover-window gate policy
- Kutta :jump contract
- 2r operator-mismatch investigation (may need a run → wt052 + new tag)

## Issue 4 — 053 defaults rows (Ryan decisions)

052c input is final; see `053-defaults-enumeration-draft-2026-09-03.md`
including the newly folded OFAT section.

## Issue 5 — storage 618G vs 400G cap (Ryan decision)

Home usage 447G of 2T after consolidation. Largest 052 item: ~60G
campaign data under `~/projects/FLOWPanel.jl/data` (two 29G 1080-step
run dirs). Trimming candidates if a cap is adopted: VTK bodies/wake
series in `fm052d_gpu_1080_t1` and `_t2exp` (keep CSVs/gates/logs).

## Also pending

- Notebook entry for 2026-09-05 (trial-2 result + ruling + retirement)
  offered, not yet drafted — needs Ryan's approval + verbosity choice.
- Detail docs: `052c-sigma-experiments-2026-08-26.md` (ledger),
  `052c-handoff-prompt-2026-09-05.md` (prior handoff; silo/infra
  history), `MEMORY.md` memories, `refactor-docs-librarian` agent for
  the big planning docs.
