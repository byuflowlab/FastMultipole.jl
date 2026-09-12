# 052e handoff — 2026-09-11 (context reset; supersedes 052e-handoff-prompt-2026-09-10.md)

## Where the item stands

The **052e.2a real-simulation addendum REGISTERED run FAILED — G1 FAIL — and
the root cause has been diagnosed** (diagnosis explicitly authorized by Ryan
2026-09-11, overriding the prereg's diagnosis-deferral). A fix-candidate test
run (Ryan's overlap=2.4 hypothesis) is **currently executing**. No code,
fixture, harness, or prereg changes have been made; the registered run was
never relaunched.

- Prereg (authoritative, **LOCKED**, do not edit):
  `052e2a-addendum-realsim-preregistration-2026-09-08.md`. Any re-run with
  changed shedding parameters requires a **new dated superseding prereg**
  (Tier 0A precedent) — Ryan's call, not the agent's.
- Frozen harness (do not edit): `scripts/addendum_052e2a_realsim_2026-09-09.jl`.
- Registered data + log: `MATRIX_OPERATOR_REFACTOR/data/052e2a-addendum-realsim/`
  (`run_20260910_2003.log`, `gates.txt`, CSVs for all completed cases).
  Never read the CSVs directly; summarize by script.

## Registered-run outcome (reported to Ryan 2026-09-11, raw values)

Run launched 2026-09-10 20:03, crashed 2026-09-11 01:54 (~5.85 h in).
Completed: all of L1–L3 (27 solves) + AW1-VTS-L4 + BW1-VTS-L4. Crashed on
**BW2-VTS-L4** (Phase B, particle wake W2, VTS route, N=19384), ~56 min into
its ~60 min case (step ≈ 50–60 of 61):
`ERROR: block Gauss-Seidel produced a nonfinite physical residual at outer
iteration 1` (FLOWPanel_solver.jl:2548, require_outer_convergence hard-fail
working as designed). Never ran: BW2-VTS-L4 result, all GR-L4/NEU-L4 cases,
oracles A_L4/B_L4, P4-L4.

- **G1 FAIL** (29/36 solves converged; the crash above).
- **G2 PASS (L1–L3)**: in-solve = prescribed hashes per level
  (L1 a175dd6c180f7d43, L2 42cbe2a3abd10c4a, L3 bd7dcd2c02f94e8c); L4 not recorded.
- **G3 PASS** (all 9 recorded GR solves): worst Green residual 9.7e-15
  (gate 1e-10), worst gauge defect 2.2e-16 (gate 1e-11).
- **G4 FAIL on threshold, monotone satisfied**: E_q Phase B W1 (R5):
  L1 2.133e-1 → L2 1.533e-1 → L3 9.665e-2 (Phase A: 1.968e-1 → 1.385e-1 →
  8.435e-2). Monotone but ≫ 1e-2 at finest completed level; L4 (registered
  finest) never produced an oracle. Slow ~0.5-order trend matches the
  smoke-flagged risk.
- G5 recorded evidence: P1-consistent (GR−NEU total-Γ gap shrinking,
  Phase A 2.0%→1.2%→0.65%, Phase B W1 ~1.1%→0.6%→0.2%); P2-consistent
  (VTS gap vs NEU ~92–94% of Γ, CL ≈ −3.8 vs −1.9, persistent across
  levels/phases/arms); P4 ratios 3.588e-3 / 5.493e-3 / 8.096e-3 (L1/L2/L3,
  growing with refinement).

## Root-cause diagnosis (established this session)

**Vortex-particle-wake vortex-stretching runaway at L4 core sizes** — same
signature as the saved 052/gpu40 finding (memory:
052-gpu40-ignition-root-cause). Causal chain: OverlapPPS(1.3,2) at L4's fine
TE spacing (dy=0.123 m → σ ≈ 0.65·filament length) → stretching amplifies
particle |Γ| super-exponentially in the rolled-up wake → ReformulatedVPM σ
evolution blows up (σ goes NEGATIVE, magnitudes 1e2–1e4) and particles are
ejected km away → Inf/NaN particle state → NaN wake-induced velocity at body
control points → NaN VTS RHS → nonfinite block-GS residual at outer iter 1.

Evidence (all in the 2026-09-10 session scratchpad,
`/private/tmp/claude-502/-Users-ryan-Dropbox-research-projects-FastMultipole/af0908dc-5375-40b5-9cc1-390a0e87a671/scratchpad/`):

1. **Control repro** `diag_bw2_vts_l4.jl` + `diag_bw2_vts_l4.log`: faithful
   BW2-VTS-L4 replica with per-step particle stats via the `maneuver!` hook
   (which sees the field state feeding each solve). Runaway ignites ~step
   30s; max|Γ|: 1.3e6 (step 37) → 4.6e6 → 4.9e7 → 4.2e8 (step 40);
   σ ∈ [−8.0e4, 1.1e4] by step 42; max|X| 5.5e3 m (2.7 m wing). Killed at
   step 42 on Ryan's instruction (no need to wait for the terminal NaN; the
   registered crash is the endpoint evidence).
2. **Relaxation NaN unit test** `relax_nan_test.jl` + `.log`: a particle with
   exactly zero |Γ| in a nonzero-J field gets all-NaN Gamma from
   `relax_correctedpedrizzetti` (0/0 in the alignment cosine; rlxf=0.3 makes
   the sqrt(b2) route safe). FLOWPanel already guards this at shed time
   (`_shed_particles!` in FLOWPanel_wake.jl ~line 745: `Γ == 0 && return`)
   — mechanism real but NOT the trigger; nothing guards Γ driven nonfinite
   during evolution.
3. No VTKs exist for the registered case (harness uses `path=nothing`).

## RESULT (2026-09-11, run completed after this section was first written):
## overlap=2.4 CONFIRMED STABLE — all 61 steps, no error, max|Γ| bounded
## 0.15–0.21, σ ∈ [0.089, 0.93] positive throughout, max|X|=22.7 m = t·U∞,
## zero nonfinite counts ("SIMULATION COMPLETED WITHOUT ERROR" in the log).
## Ryan's hypothesis confirmed; next-action 1 below is done — start at 2.

## overlap=2.4 fix-candidate run (Ryan's hypothesis)

Ryan (2026-09-11): "Overlap of 1.3 is borderline unstable; I bet increasing
Overlap to >2 will do the trick. Try 2.4." Running now (nohup-detached,
survives session end, 4 threads, julia PIDs ~54562/54569):

- Script: `<scratchpad above>/diag_bw2_vts_l4_ov24.jl` — identical to control
  except `OverlapPPS(2.4, 2)` both arms.
- Log: `<scratchpad above>/diag_bw2_vts_l4_ov24.log` (one line per step;
  flushed).
- Status at handoff: step 30/61, np=2250, **stable and past the control's
  ignition window start**: max|Γ| ≈ 0.14 and slowly DECREASING, σ ∈
  [0.116, 0.756] all positive, max|X| ≈ 11.7 m = freestream convection
  distance. Success criterion: completes all 61 steps with max|Γ| O(1e-1),
  σ positive, no bad counts, and prints "SIMULATION COMPLETED WITHOUT
  ERROR". ETA ~30–50 min from handoff (steps slow as np grows).

## Next actions (in order)

1. Check the ov24 run: `tail diag_bw2_vts_l4_ov24.log`. If completed clean →
   report to Ryan: overlap-2.4 hypothesis CONFIRMED (stable where 1.3 ran
   away). If it ignited late → report the trace; hypothesis refuted at 2.4.
   If still running, wait (do not poll aggressively; do not relaunch).
2. Report to Ryan and ask his ruling on the path forward. The registered
   tier is STOPPED on G1 (and G4-at-L3). Options are his: (a) superseding
   prereg (e.g. OverlapPPS(2.4,2) for W2, possibly VTK output enabled for
   W2 arm) + re-run; (b) partial results file on L1–L3 as-is; (c) both.
3. Results doc `052e2a-addendum-realsim-results-2026-09-10.md` (or re-dated)
   still owed per prereg Outputs — scope depends on Ryan's ruling in (2).
   Ruling section (P1–P4 + CONTINUE) is Ryan's, not the agent's.
4. Offer (do not write) notebook entries — still owed for: Tier 0B, Tier
   0B-R, production integration, addendum lock, addendum registered run +
   G1 failure + diagnosis. Offer verbosity options; approval required first.

## Cautions (carried forward + new)

- LOCKED prereg append-only; supersession needs a new dated prereg.
- Frozen sha-registered scripts and the addendum harness: do not edit.
- Diagnostic scripts/logs live in the OLD session scratchpad (absolute path
  above) — a new session gets a DIFFERENT scratchpad; copy anything worth
  keeping into `MATRIX_OPERATOR_REFACTOR/` (ask Ryan) before /tmp cleanup.
- Nothing committed in either repo; FastMultipole (branch flowpanel-20260817)
  heavily dirty with unrelated work — touch only 052e files. FLOWPanel dirty
  with unrelated BRAINSTORM files — leave alone.
- Before touching FLOWPanel code: read its AGENTS.md, CLAUDE.md,
  agent_policies/WORKFLOW.md (+ TESTING.md before testing). ≤ 4 threads.
- Laptop runs = formulation-proof tier, not official campaigns.
- Never read `data/**` CSVs directly; summarize by script.
- Later queue: 052e.3 hybrid fixture; 052e.1 regression completion after
  052b closes; 052e.6 gauge design study.
