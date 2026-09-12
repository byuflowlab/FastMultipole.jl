# 052e handoff — 2026-09-09b (context reset; supersedes 052e-handoff-prompt-2026-09-09.md)

## Where the item stands

052e.2b (implicit Householder) remains ADOPTED/integrated (see the
2026-09-08 handoff for the FLOWPanel integration record and its cautions —
still applies verbatim, not repeated here). The active task is the
**052e.2a real-simulation addendum**, whose prereg is now **LOCKED**:

`052e2a-addendum-realsim-preregistration-2026-09-08.md` — **LOCKED by Ryan
2026-09-09**. Read it in full first; it is the authoritative design AND it
already absorbs all L1–L4 rulings and the L4 pre-lock verification
findings (Lock record section at the bottom). Do not re-litigate anything
in it. Registered values may now be produced, but only by the registered
run described below.

Lock-day rulings (recorded in the file; summary only):
- L1: AOA 30°, |U∞|=1, mesh levels **L1–L4 all required**.
- L2: Phase B production defaults recorded pre-launch; **W2 particle arm
  includes all three routes** (R-VTS particle path verified working).
- L3: off-body survey line **dropped** (Γ(y) already localizes spanwise);
  hard gates are G1–G4 only.
- L4: verification pass complete, all checks PASS.

## Immediate next action: build the harness, then the registered run

1. Write `scripts/addendum_052e2a_realsim_2026-09-09.jl` per the prereg's
   Harness section. Seeds:
   - `scripts/l4_prelock_probe_052e2a_2026-09-09.jl` and
     `scripts/l4_prelock_probe_c2_052e2a_2026-09-09.jl` — the (unregistered,
     non-frozen) L4 probes copied from the verification pass. They contain
     WORKING code for every tricky construction: the uncapped Neumann body
     (`pitching_wing_mesh(...; caps=false)` + direct
     `RigidWakeBody{ConstantDoublet,1,TF,false}(...; watertight=false)`),
     the prescribed-wake-as-first-`simulate!`-step Phase A pattern
     (pre-built `PanelWake`, nodes/strengths/`nwakes[]` set directly,
     `update_TE!` first, `t_range` ≥ 2 samples), the
     `step_telemetry_callback` state capture, and particle-wake runs for
     both GR and VTS.
   - `scripts/tier0b_052e2a_bordered_formulation.jl` (FROZEN — read-only)
     for stage-1 body construction, metric definitions (E_q, E_inf,
     area-weighted gauge alignment), and provenance/gates.txt machinery.
     Caution: it reads `gs.sol_b[end]`; new code must use `_green_lambda`.
2. Smoke mode first (unregistered, tiny mesh, stage-1 precedent), then the
   registered run: nohup-detached, ≤4 threads, logs + CSVs under
   `MATRIX_OPERATOR_REFACTOR/data/052e2a-addendum-realsim/`, gates.txt
   snapshot per Tier 0A standard. L4 dense states ≈ 3 GB/route.
3. Results file `052e2a-addendum-realsim-results-<date>.md` per the
   prereg's Outputs section; ends with Ryan's P1–P4 ruling + CONTINUE
   decision for 052e.3 (Ryan rules, not the agent).

## Key API facts verified by the L4 pass (all absorbed into the prereg)

- `GreenReconstruction(gauge=:area_mean)` works through `simulate!` with
  the default Kutta pair (`RigidTransitionAttachment` + `JumpKutta`) and
  `Backslash`; state carries `GreenHouseholderState`; multiplier via
  `pnl._green_lambda(state.green)` (`FLOWPanel_formulation.jl:800`).
- `GreenReconstructionState` has NO Green-residual/gauge-defect fields —
  the harness computes G3 telemetry itself (tier0b pattern:
  `_green_B_product!` + areas), captured per step via `simulate!`'s
  `step_telemetry_callback` (receives
  `(; i_step, formulation, formulation_state, overlap_report)`,
  `FLOWPanel_simulate.jl:746-748`).
- `steady!` CANNOT host Phase A (no wake systems, no `formulation` kwarg,
  `FLOWPanel_simulate.jl:1038/1005`); a length-1 `t_range` hits a bounds
  error (`FLOWPanel_simulate.jl:1293-1294`).
- `build_pitching_wing_body` hardcodes `caps=true`/watertight
  (`examples/pitching_wing.jl:239-256`) — Neumann referee uses the direct
  construction above; rank-deficiency warning (`FLOWPanel_solver.jl:821-826`)
  does not fire for the uncapped body.
- `require_outer_convergence` is VTS-only (GR has no outer iteration).
- `set_wake_Das!` lives in `examples/pitching_wing.jl:228` — call it
  unqualified after including the example, not as `pnl.set_wake_Das!`.
- Particle wakes: no particles until ≥ `nwakerows` warm-up steps; with
  `nwakerows=2` particles appeared ~step 4. Budget warm-up in short W2
  runs and the frozen-particle Phase A variant.
- Phase A wake-node hashes verified bit-identical across all three routes
  (incl. the uncapped Neumann body — same TE ⇒ same wake nodes).

## Not yet done / open (carried forward)

- **Notebook entries: still owed** (Tier 0B, Tier 0B-R, production
  integration, and now the addendum lock — offered five times; Ryan says
  not yet). Keep offering at milestones with verbosity options; approval
  required before writing anything.
- **Nothing committed in either repo.** All 052e work uncommitted; if
  asked to commit, see the 2026-09-08 handoff's file lists and the
  052e.1-entanglement flag.
- Later queue (unchanged): 052e.3 hybrid fixture (stage-5 extraction folds
  in there); 052e.1 regression completion after 052b closes; 052e.6 gauge
  design study.

## Repository/worktree cautions (unchanged)

- FastMultipole worktree (branch flowpanel-20260817) heavily dirty with
  unrelated higher-derivative work; touch only 052e files. FLOWPanel.jl
  dirty with unrelated BRAINSTORM 018/021/026 files; leave them alone.
- Before touching FLOWPanel: read its `AGENTS.md`, `CLAUDE.md`,
  `agent_policies/WORKFLOW.md` (+ `TESTING.md` before testing). ≤4 threads.
- Laptop runs = formulation-proof tier, not official campaigns.
- Never read `data/**` CSVs directly — summarize by script.
- Frozen sha-registered scripts (`tier0br_052e2b_householder_parity.jl`,
  `tier0b_052e2a_bordered_formulation.jl`) must not be edited. The two
  `l4_prelock_probe*` scripts are NOT frozen (reference copies).
- The LOCKED prereg must not be edited except to append (supersession
  requires a new dated prereg with reason, Tier 0A precedent).
