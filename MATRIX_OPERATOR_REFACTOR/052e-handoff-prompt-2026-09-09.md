# 052e handoff — 2026-09-09 (context reset; supersedes 052e-handoff-prompt-2026-09-08.md)

## Where the item stands

052e.2b (implicit-Householder reduction) is ADOPTED and integrated as the
production dense `:area_mean` Green-solve route in FLOWPanel (see the
2026-09-08 handoff for that integration record — its FLOWPanel change list
and registered-artifact cautions still apply verbatim and are NOT repeated
here; read it). The active task is the **052e.2a continuation addendum**,
which Ryan redesigned on 2026-09-08 from a manufactured decomposition into
a **real-simulation cross-formulation convergence test**.

## The redesign (Ryan's 2026-09-08 rulings — do not re-litigate)

Draft prereg (NOT LOCKED):
`052e2a-addendum-realsim-preregistration-2026-09-08.md` — read it first;
it is the authoritative design. Superseded same-day pre-lock:
`052e2a-addendum-stages234-preregistration-2026-09-08.md` (rejected
disk/ring decomposition; kept as record).

Core design: capped NACA0012 wing (stage-1 mesh family) at **AOA 30°**,
three routes compared:

- **R-VTS**: `VelocityThroughSources` (production 052b; wake velocity →
  sources).
- **R-GR**: `GreenReconstruction(gauge=:area_mean)` — what this item
  built: sources freestream-only, wake trace reconstructed from sampled
  wake velocities via (I−B)q = Sσ (production Householder; λ via
  `_green_lambda`), doublets compensate the wake-induced potential. NOT
  `HybridWakePotential` — Ryan explicitly wants full reconstruction even
  for the doublet wake so the directly evaluated panel-wake potential
  serves as an independent oracle (prediction P3), and the identical
  scheme then handles the particle wake (P4).
- **R-NEU**: Neumann referee — **same thick geometry without endcaps**
  (`caps=false`, non-watertight, `DBC=false`), doublet panels only.
  Rationale: Neumann consumes wake velocity as *complete* BC data, so
  Neumann-vs-VTS disagreement isolates the dropped potential trace.

Every solve uses the **default Kutta pair `RigidTransitionAttachment` +
`JumpKutta`** (the small rigid attached TE upper/lower transition panels;
Ryan's "implicit kutta panels"). The VTS-only Route B
(`TEAnchoredAttachment`, gated at `FLOWPanel_kutta.jl:489`) is NOT used.
Phase A = prescribed identical wake (hash-verified) → Phase B = free wake;
arm W1 = doublet PanelWake → W2 = production particle wake,
production/default regularization, no kernel/core sweeps (rulings R5/R6).
Predictions P1–P4 and minimal gates G1–G4 are in the prereg; P1/P2/P4 are
recorded evidence for Ryan's ruling, not numeric gates.

## Immediate next actions

1. **Ryan owes rulings on L1–L3** (fixture numbers, Phase-B/W2 scope,
   survey line + gate set) — listed at the bottom of the prereg. Ask if
   not yet given.
2. **L4 pre-lock verification pass** (Ryan aware, not yet launched): a
   mechanical probe (no registered values) confirming API assumptions —
   `GreenReconstruction` accepts the setup; Neumann route + rigid
   attachment on the uncapped body works; particle wake feeds the
   reconstruction σ path (state snapshots `u_prewake`; scheme is
   wake-type agnostic by construction — verify in practice); a
   prescribed-wake solve is expressible via the public API (`steady!`
   rejects TEAnchoredAttachment only; default pair fine). Report findings
   to Ryan BEFORE lock; do not work around mismatches silently.
3. After lock: dated harness `scripts/addendum_052e2a_realsim_2026-09-08
   .jl` (rename to actual date), reusing stage-1 machinery from
   `scripts/tier0b_052e2a_bordered_formulation.jl` (note: that script
   reads `gs.sol_b[end]`; new code must use `_green_lambda`). Registered
   run nohup-detached, ≤4 threads, logs under
   `data/052e2a-addendum-realsim/`.

## Useful code facts already verified this session

- `GreenReconstruction` struct: `FLOWPanel_formulation.jl:102` (gauges
  `:area_mean` default / `:area_mean_bordered` debug / `:lsq`); its state
  holds `u_prewake` (3×N pre-wake velocity snapshot) and forms σ = −u_f·n
  from the wake pass — wake-type agnostic.
- Kutta support boundary: non-default attachment/closure requires VTS
  (`FLOWPanel_kutta.jl:489`); default `RigidTransitionAttachment` +
  `JumpKutta` works for all formulations, both BC types
  (`FLOWPanel_simulate.jl:1030`).
- Watertight-Neumann is rank-deficient; uncapped is the supported referee
  configuration (`FLOWPanel_solver.jl:822` warning text says exactly
  "remove a cap").
- Thickness confound context (why uncapped-thick referee): thin-vs-thick
  lift-slope gap ~9% at t/c=0.12; same-geometry-uncapped referee shrinks
  it to the open-TE/cap difference.

## Not yet done / open (carried forward)

- **Notebook entries: offered FOUR times (Tier 0B, Tier 0B-R, production
  integration; 2026-09-08 Ryan said "Not now").** Keep offering at
  milestones with verbosity options; approval required before writing.
- **Nothing committed in either repo.** All 052e work uncommitted; if
  asked to commit, see the 2026-09-08 handoff's file lists and the
  052e.1-entanglement flag.
- Later queue (unchanged): 052e.3 hybrid fixture (stage-5 extraction folds
  in there); 052e.1 regression completion after 052b closes; 052e.6 gauge
  design study.

## Repository/worktree cautions (unchanged from 2026-09-08 handoff)

- FastMultipole worktree (branch flowpanel-20260817) heavily dirty with
  unrelated higher-derivative work; touch only 052e files. FLOWPanel.jl
  dirty with unrelated BRAINSTORM 018/021/026 files; leave them alone.
- Before touching FLOWPanel: read its `AGENTS.md`, `CLAUDE.md`,
  `agent_policies/WORKFLOW.md` (+`TESTING.md` before testing). ≤4 threads.
- Laptop runs = formulation-proof tier, not official campaigns.
- Never read `data/**` CSVs directly — summarize by script.
- Frozen sha-registered scripts (`tier0br_052e2b_householder_parity.jl`)
  must not be edited.
- Any registered run needs its dated LOCKED prereg first (Ryan locks; no
  registered values before lock).
