# 052e handoff — 2026-09-07b (context reset; supersedes 052e-handoff-prompt-2026-09-07.md for current state)

## Immediate state

- **Tier 0B / 052e.2a registered run IS LAUNCHED and in flight.** PID 51333
  (laptop, nohup-detached), launched 2026-09-07 by Ryan's direction
  ("launch it"). Log:
  `MATRIX_OPERATOR_REFACTOR/data/052e2a-tier0b/tier0b_run1.log`. Expected
  ~15–30 min; outputs (per-level CSVs + `gates.txt`) land in
  `MATRIX_OPERATOR_REFACTOR/data/052e2a-tier0b/`. The console gates table is
  printed at the end of the log between `===== TIER 0B GATES =====` markers.
- **Theory note ACCEPTED by Ryan 2026-09-07** (status line at top of
  `052e-theory-velocity-to-potential-trace.md`). The theory gate for
  interpreting Tier 0B is closed.
- **Preregistration LOCKED:**
  `052e2a-tier0b-preregistration-2026-09-07.md`. Locked by Ryan
  ("lock it"). NO value in it may be edited now that results exist; if a
  fixture/gate proves ill-posed, supersede with a new dated file and record
  the reason (Tier 0A precedent).
- **Harness:** `MATRIX_OPERATOR_REFACTOR/scripts/tier0b_052e2a_bordered_formulation.jl`.
  Smoke-validated mechanically only (TIER0B_SMOKE=1 tiny unregistered
  config, temp-dir output, runner reported exit status only — no registered
  gate values were viewed by anyone before launch).

## Exact next actions

1. Check the run: `ps -p 51333`; when done, confirm exit and that
   `gates.txt` + 8 `trace_C*_L*.csv` files exist. If it crashed, diagnose
   from the log; a harness (not kernel/formulation) defect is fixable, but
   then the whole registered run must be relaunched cleanly and the fix
   recorded — no partial reuse.
2. Draft `052e2a-tier0b-results-2026-09-07.md`: per-gate PASS/FAIL table
   (B1–B9 per case C1/C2), error+refinement tables, λ/flux/residual/Hodge
   telemetry, clearance minima, the influence!-route cross-check numbers,
   call chain, provenance from gates.txt, and a ruling RECOMMENDATION only.
   Do NOT read the CSVs directly (token policy) — gates.txt and the log tail
   carry everything needed; summarize CSVs by script if more is required.
3. Present to Ryan for the explicit **PASS / CONTINUE / RETIRE** ruling on
   052e.2a. Kill rule (locked): retire only if E_q(finest) > 20% or E_q
   fails to improve under body-mesh refinement. B1/B7 misses with clean
   refinement = CONTINUE, not retire.
4. Only after a PASS ruling: begin 052e.2b (implicit-Householder reduction,
   Tier 0B-R parity gates) per
   `052e-accuracy-plan-v2-draft-2026-09-05.md` §5/Tier 0B-R and theory note
   §3.1. Stages 2–4 (filament/particles) run as a .2a continuation addendum
   (may parallel .2b); stage 5 folds into 052e.3 — see the ratified
   scheduling paragraph in accuracy plan §5.

## What the harness does (for interpreting results)

- Fixture per locked prereg: capped NACA0012 wing (`pitching_wing_mesh` /
  `build_pitching_wing_body`, thickness=0.12, b=2.7, c=0.76), 4 levels
  (1,744/3,816/8,960/19,384 panels), prescribed flat 40-row × 0.5c wake
  (20c), row 1 zero strength (realizes the excluded Kutta row; support
  starts 0.5c aft of TE), elliptic mu(y)=sqrt(1-(2y/b)^2) on y∈[-b/2,b/2],
  cases C1 = AOA 0°, C2 = AOA 7°.
- Oracle: per-panel `pnl.induced` sums over a triangle-split
  `NonLiftingBody{ConstantDoublet}` copy of the nonzero wake panels (the
  Tier 0A A10-validated wrapper route, independent of `influence!`).
- System under test (production route): sigma = -n·u_wake →
  `_source_potential!` → `_build_green_solve_state(body,:area_mean)` →
  `_green_solve_q!` (lambda = `gs.sol_b[end]`), residual via matrix-free
  `_green_B_product!`, Hodge via `surface_hodge_trace!` (L1–L3 only; L4
  skipped for memory — registered note, B6 is a trend record).
- Report-only cross-check: `_wake_potential!`/`_wake_panel_velocity!`
  (production PanelWake → influence! route) vs oracle; first exercise of
  that route (Tier 0A did not cover it).

## Rulings/edits this session (all uncommitted, FastMultipole worktree)

- Theory note: acceptance line added (top).
- Accuracy plan §5: ratified scheduling paragraph for decomposition stages
  2–5 (not lost; see above).
- Impl doc 052e.2a bullet: stage-1-only ruling scope cross-reference.
- Prereg created + locked (see above). Kutta-row misunderstanding resolved:
  `live_rows[]` mechanism (`FLOWPanel_wake.jl` ~121–141, 236–241) excludes
  the TE-attached live block from wake source views; prereg support
  condition holds with NO deviation from plan §5.
- Area-mean closure rationale (delivered to Ryan, basis of his acceptance):
  gauge not physics (zero-at-infinity unavailable from velocity-only data);
  area-mean = discretized ∫q dS = 0 (mesh-convergent, unlike pinning);
  bordered multiplier isolates incompatibility as telemetry (lsq smears it,
  rank-one obscures it); constraint aligned with nullspace (robust saddle);
  solver gauge = comparison gauge.

## Repository/worktree cautions (unchanged from prior handoff)

- FastMultipole is heavily dirty with unrelated higher-derivative work +
  untracked `HIGHER_DERIVATIVES/`. Touch only 052e files; do not clean,
  stage, or commit unrelated changes.
- `../FLOWPanel.jl` is dirty; `src/FLOWPanel_formulation.jl` and
  `test/formulation_test.jl` carry uncommitted 052e.1 work — preserve.
  Before FLOWPanel code work read its `AGENTS.md`, `CLAUDE.md`,
  `agent_policies/WORKFLOW.md` (+`TESTING.md` before testing). ≤4 threads
  locally.
- This laptop run is a formulation-proof tier, not an official campaign; no
  campaign worktree was required. Any future official acceptance runs still
  need committed, tagged campaign worktrees per the global policy.
- No notebook entry has been written or offered yet for Tier 0B; offer one
  (with verbosity options) after the ruling.

## Still open, separate

- 052e.1 regression completion after 052b closes; 052e.6 global gauge
  design study; promotion limited to gauge-invariant outputs. Historical
  052b/053 issues: see `052e-handoff-prompt-2026-09-06.md`.
