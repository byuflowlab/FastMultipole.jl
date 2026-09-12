# 052e handoff — 2026-09-10 (context reset; supersedes 052e-handoff-prompt-2026-09-09b.md)

## Where the item stands

The **052e.2a real-simulation addendum** is past harness-build and smoke:
the harness is written, smoke-verified, and the **REGISTERED run is
currently executing** (launched 2026-09-10 ~20:03, nohup-detached, 4
threads, laptop). Do not start a second run; do not edit the harness while
it runs.

- Prereg (authoritative, **LOCKED** — read in full first, do not edit or
  re-litigate): `052e2a-addendum-realsim-preregistration-2026-09-08.md`.
- Harness: `scripts/addendum_052e2a_realsim_2026-09-09.jl`. Its header
  records realization choices **R1–R11** (approved by Ryan pre-launch via
  the smoke report + "do it"); they are recorded pre-launch choices and
  must not be changed for this registered run.
- Registered run log:
  `MATRIX_OPERATOR_REFACTOR/data/052e2a-addendum-realsim/run_20260910_2003.log`
  (CSVs + `gates.txt` land in the same dir). Never read the CSVs directly
  — summarize by script. Rough duration estimate: several hours (L4 dense
  solves dominate). Config: levels L1–L4 (1744/3816/8960/19384), NFREE=40
  (20c Phase A wake), DT=0.38, NITER_A=80, NSAMP_B=61.
- Smoke evidence (unregistered): logs `smoke_addendum{,2,3,4}.log` and
  `smoke_data/` in the session scratchpad
  (`/private/tmp/claude-502/-Users-ryan-Dropbox-research-projects-FastMultipole/c88de228-2244-43df-8136-a662f30a358e/scratchpad/`).
  Final smoke: G1/G2/G3 PASS, oracle-vs-production cross-checks ~1e-14,
  sigma consistency 1e-16, Kutta cross-check exactly 0, clearance ≈ 0.05c.

## Key findings this session (know these before touching anything)

1. **Das bug in the L4 probe seeds, fixed in the harness (R11):**
   `set_wake_Das!(body, VINF)` called BEFORE `pitching_wing_frame` gives a
   unit-magnitude Das that the frame then rotates with the body — the wake
   attached 1.0 m behind the TE along the rotated chord. Production
   convention (`prepare_pitching_wing`, examples/pitching_wing.jl:953-966):
   frames FIRST, then `set_wake_Das!(body, VINF; magnitude=0.05*c)`.
   The two `l4_prelock_probe*` scripts still contain the buggy pattern
   (they are non-frozen reference copies; harmless for their API-only
   purpose, but do not copy that pattern again).
2. **Phase A realization (R1):** fixed-point inside ONE `simulate!` call
   via the `maneuver!` hook (runs before update_TE!/solve each step):
   re-prescribe flat 20c wake, all rows = previous solve's Γ(y), zero
   start. Contraction geometric r≈0.66/iter; 80 iters → δ≈1e-10 ≪ TOL_A=1e-8.
3. **Registered-outcome risks to expect when results land:**
   - **G4 at risk:** smoke E_q ≈ 0.27→0.23 (552→1068 panels), slow order
     (~0.5 in h). Attached wake + AOA 30° is much harder than stage-1's
     detached fixture (stage-1 registered L1 was 6.6e-3–2.7e-2). If G4
     fails at L4: STOP per prereg, report raw numbers, no retuning.
   - **VTS ≈ 2× circulation:** smoke showed Γ_VTS ≈ 2× Γ_GR ≈ 2× Γ_NEU
     (CL −3.8 vs −1.9), gap ~0.9 relative, stable across levels/phases/
     arms, while GR−NEU ≈ 2–4%. Qualitatively P1/P2-consistent, strikingly
     large. **Diagnosis is deferred by the prereg** — record, do not chase.
   - Γ is negative by the μ_upper−μ_lower convention at positive lift; CL
     is the Kutta–Joukowski summary (R3), not pressure-based.
4. **API facts (verified this session, beyond the 2026-09-09b list):**
   reconstructed trace lives at `state.green.q`; shedding matrix columns =
   (p_up, slot_a, slot_b, p_lo, slot_a, slot_b) with Γ = strength[p_up,end]
   − strength[p_lo,end] (p_lo may be −1), matching `shed_wake!`; default
   Kutta pair is the legacy path — closure c≡0 by construction
   (FLOWPanel_kutta.jl:16-18, 68-70), so "Kutta residual" is realized as a
   convention cross-check vs `_get_wakestrength_mu`; `GreenReconstruction`
   (and all non-VTS formulations) ERROR on a Neumann body — R-NEU runs
   `VelocityThroughSources(require_outer_convergence=true)`; simulate! step
   order = maneuver! → update_TE! → prewake snapshot → wake influence →
   solve → telemetry callback → convection → shed_wake!; `mktempdir()`
   outputs vanish at exit (smoke uses a persistent dir via
   `ADDENDUM_SMOKE_DIR`).

## Next actions (in order)

1. Check the registered run: tail the run log. If still running, wait (do
   not poll aggressively). If it died, report the error verbatim with the
   log tail — do not silently relaunch.
2. When complete: verify gates.txt exists; report G1–G4 with raw values to
   Ryan. Any G1–G4 FAIL stops the tier — report before any code change.
3. Write `052e2a-addendum-realsim-results-2026-09-10.md` per the prereg's
   Outputs section: Γ(y) tables/figures, pairwise-gap refinement tables
   (P1/P2), oracle table (P3, Phase A and Phase B sequences both), P4
   ratios, recorded diagnostics, call chains, realization notes R1–R11,
   and an explicit ruling section for **Ryan** (P1–P4 each
   CONFIRMED/REFUTED/INCONCLUSIVE + CONTINUE for 052e.3 — Ryan rules, not
   the agent). Summarize CSVs by script only.
4. Offer (do not write) notebook entries — **still owed** for: Tier 0B,
   Tier 0B-R, production integration, addendum lock, and now the addendum
   run. Offer verbosity options; approval required first.

## Cautions (carried forward, unchanged)

- LOCKED prereg: append-only; supersession needs a new dated prereg.
- Frozen sha-registered scripts (`tier0b_052e2a_bordered_formulation.jl`,
  `tier0br_052e2b_householder_parity.jl`) must not be edited.
- Nothing committed in either repo; FastMultipole worktree (branch
  flowpanel-20260817) heavily dirty with unrelated work — touch only 052e
  files. FLOWPanel dirty with unrelated BRAINSTORM files — leave alone.
- Before touching FLOWPanel: read its AGENTS.md, CLAUDE.md,
  agent_policies/WORKFLOW.md (+ TESTING.md before testing). ≤ 4 threads.
- Laptop runs = formulation-proof tier, not official campaigns.
- Never read `data/**` CSVs directly; summarize by script.
- Later queue: 052e.3 hybrid fixture; 052e.1 regression completion after
  052b closes; 052e.6 gauge design study.
