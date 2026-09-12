# 052e handoff — 2026-09-11b (context reset; supersedes 052e-handoff-prompt-2026-09-11.md)

## Where the item stands

The 052e.2a real-simulation addendum is on its **superseded v2 REGISTERED
run, currently executing** (launched 2026-09-11 22:58, nohup-detached, PID
59407 at launch, 4 threads, laptop, **~10 h expected — finish ≈ 09:00–09:30
on 2026-09-12** (from original-run timings: L1–L3 ≈ 2.6 h, six Dirichlet L4
cases ≈ 6.7 h, three Neumann L4 ≈ 0.8 h; the overlap change is cost-neutral)). Do not start a second
run; do not edit the harness while it runs.

Sequence of events this item: the original registered run (2026-09-10 20:03,
prereg `052e2a-addendum-realsim-preregistration-2026-09-08.md`, LOCKED)
**failed G1** at BW2-VTS-L4 (`block Gauss-Seidel produced a nonfinite
physical residual at outer iteration 1`, ~5.85 h in, L1–L3 complete).
Ryan authorized diagnosis; root cause established and fix confirmed; Ryan
ruled "change the W2 arm to 2.4 overlap across the board"; supersession
recorded and v2 launched.

## Authoritative documents

- LOCKED prereg (do not edit): `052e2a-addendum-realsim-preregistration-2026-09-08.md`.
- **Supersession record (read it):** `052e2a-addendum-realsim-supersession-2026-09-11.md`
  — single change: W2 shedding `OverlapPPS(1.3,2)` → `OverlapPPS(2.4,2)`,
  both arms, all levels. All other locked values, gates G1–G4, and
  predictions P1–P4 unchanged. Ruling remains Ryan's.
- v2 harness: `scripts/addendum_052e2a_realsim_2026-09-11.jl` (copy of the
  2026-09-09 harness + OverlapPPS values, output dir, R12 header note; do
  not edit). Original harness `scripts/addendum_052e2a_realsim_2026-09-09.jl`
  untouched.
- v2 run log + outputs: `MATRIX_OPERATOR_REFACTOR/data/052e2a-addendum-realsim-v2/`
  (`run_20260911_2258.log`; CSVs + gates.txt land there). Never read the
  CSVs directly; summarize by script.
- FAILED-run evidence (preserve, do not overwrite):
  `data/052e2a-addendum-realsim/` (log `run_20260910_2003.log`, gates.txt,
  L1–L3 CSVs + 2 L4 cases).

## Failed-run gate report (already delivered to Ryan, raw values)

G1 FAIL (29/36 solves; BW2-VTS-L4 crash). G2 PASS L1–L3 (hashes
a175dd6c180f7d43 / 42cbe2a3abd10c4a / bd7dcd2c02f94e8c, in-solve =
prescribed). G3 PASS (worst Green residual 9.7e-15, worst gauge defect
2.2e-16). G4 monotone but over threshold at finest completed level
(E_q Phase B W1: 2.133e-1 / 1.533e-1 / 9.665e-2 at L1/L2/L3; Phase A:
1.968e-1 / 1.385e-1 / 8.435e-2; ~0.5-order, matches smoke-flagged risk —
expect G4 to be the run-2 risk too). G5 evidence: P1-consistent (GR−NEU
gap shrinking, PhB ~1.1%→0.6%→0.2%), P2-consistent (VTS ~92–94% above
NEU/GR in |Γ|, persistent), P4 3.588e-3 / 5.493e-3 / 8.096e-3 (growing).

## Root cause + fix evidence (closed 2026-09-11)

Vortex-stretching runaway of the particle wake at L4 core sizes under
OverlapPPS(1.3,2): |Γ| ~10×/step (1.3e6→4.2e8, steps 37–40 of a faithful
repro), ReformulatedVPM σ blown up (negative, |σ| to 8e4), particles km
away, → Inf/NaN → NaN VTS RHS. Same signature as memory
`052-gpu40-ignition-root-cause`. Overlap 2.4 repro: all 61 steps clean,
max|Γ| 0.15–0.21, σ ∈ [0.089,0.93] positive, max|X| = t·U∞. Also
unit-confirmed (not the trigger; shed-time guarded): zero-|Γ| particle →
all-NaN Gamma in `relax_correctedpedrizzetti` (0/0 alignment cosine);
guard at `_shed_particles!`, FLOWPanel_wake.jl ~745.
Diagnostic scripts/logs live in the 2026-09-10 session scratchpad
`/private/tmp/claude-502/-Users-ryan-Dropbox-research-projects-FastMultipole/af0908dc-5375-40b5-9cc1-390a0e87a671/scratchpad/`
(`diag_bw2_vts_l4.jl`+`.log` = control runaway; `diag_bw2_vts_l4_ov24.jl`
+`.log` = clean 2.4 run; `relax_nan_test.jl`+`.log`). /tmp is volatile —
offer Ryan to copy these into MATRIX_OPERATOR_REFACTOR/ before relying on
them long-term.

## Next actions (in order)

1. Check the v2 run: tail `data/052e2a-addendum-realsim-v2/run_20260911_2258.log`.
   If still running, wait (do not poll aggressively). If it died, report
   the error verbatim + log tail; do not silently relaunch.
2. When complete: verify gates.txt; report G1–G4 to Ryan with raw values
   (any FAIL stops the tier — report before any code/fixture change; if G4
   fails on threshold again, that is a registered outcome to report, not
   retune).
3. Draft `052e2a-addendum-realsim-results-2026-09-11.md` per the prereg's
   Outputs section on the v2 dataset only (no mixing with the failed run):
   Γ(y) tables, pairwise-gap refinement tables (P1/P2), oracle table (P3,
   Phase A + B), P4 ratios, diagnostics, call chains, R1–R12 notes, the
   failed-run/supersession history, and an explicit ruling section for
   **Ryan** (P1–P4 CONFIRMED/REFUTED/INCONCLUSIVE + CONTINUE for 052e.3 —
   Ryan rules, not the agent). Summarize CSVs by script only.
4. Offer (do not write) notebook entries — owed for: Tier 0B, Tier 0B-R,
   production integration, addendum lock, failed registered run + diagnosis
   + supersession, v2 run. Offer verbosity options; approval required first.

## Cautions (carried forward)

- LOCKED prereg append-only; further changes need a new dated supersession.
- Frozen sha-registered scripts (`tier0b_052e2a_bordered_formulation.jl`,
  `tier0br_052e2b_householder_parity.jl`) and both addendum harnesses: do
  not edit.
- Nothing committed in either repo; FastMultipole (branch
  flowpanel-20260817) heavily dirty with unrelated work — touch only 052e
  files. FLOWPanel dirty with unrelated BRAINSTORM files — leave alone.
- Before touching FLOWPanel code: read its AGENTS.md, CLAUDE.md,
  agent_policies/WORKFLOW.md (+ TESTING.md before testing). ≤ 4 threads.
- Laptop runs = formulation-proof tier, not official campaigns.
- Never read `data/**` CSVs directly; summarize by script.
- Later queue: 052e.3 hybrid fixture; 052e.1 regression completion after
  052b closes; 052e.6 gauge design study.
