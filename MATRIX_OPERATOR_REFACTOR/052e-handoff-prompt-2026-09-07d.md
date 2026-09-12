# 052e handoff — 2026-09-07d (context reset; supersedes 052e-handoff-prompt-2026-09-07c.md)

## Immediate state: Tier 0B-R (052e.2b) is DONE and RULED — ADOPT

- **052e.2b ADOPT — ruled by Ryan 2026-09-07** ("that's good enough").
  Results: `052e2b-tier0br-results-2026-09-07.md` (status = RULED —
  ADOPT). Prereg `052e2b-tier0br-preregistration-2026-09-07.md`
  (LOCKED, unedited; note it descoped the plan-§5 disconnected-body /
  distorted-mesh / gauge-invariant-output / performance-gate items,
  ratified at lock).
- **Run 1 headline (PID 66786, ~5 min, gates.txt
  script_sha256=d86e666e43d0):** implicit-Householder reduction vs
  production bordered `:area_mean` route, same-process, L2+L4, C1+C2.
  Trace parity 2.5e-14…1.9e-11 vs τ(N)=1e3·√N·eps gates; multiplier
  parity ≤4.8e-17; gauge defect ≤1.7e-16; residuals ~1e-15; drift
  precondition passed on all four run-2 anchors. Reduced block BETTER
  conditioned than bordered (rcond ratio 0.61–0.83); setup ±4%, solve
  and storage at parity.
- **One formal FAIL adjudicated as artifact under the ADOPT ruling:**
  R3/C2/L2 (incompatible RHS) trace parity 2.70e-11 vs 1.37e-11 gate
  (2.0×), while its λ′ parity was 6.2e-16 (the point of the gate).
  Adjudication + tolerance-model lesson (future parity tolerances need a
  κ/RHS-scale factor, not just √N):
  `052e2b-tier0br-gate-adjudication-2026-09-07.md`.
- **Registered implementation note:** run-2 anchors carry 7 significant
  digits, so the prereg's 1e-10 drift clause was applied at anchor
  quantization (≤1e-6); recorded in harness header and results, not a
  gate edit.
- Validated reference implementation of the reduction:
  `scripts/tier0br_052e2b_householder_parity.jl` — `make_reflector`,
  `reduced_setup!`, `reduced_solve` (implicit reflector, rank-one BLAS
  two-sided transform in place, LU of leading (N−1) block, λ from
  omitted row). R0 audit facts in its header. Smoke: TIER0BR_SMOKE=1.

## Next task (Ryan chose no successor yet — ask, or default to #1)

1. **Production integration of the reduction (consequence of ADOPT):**
   port `reduced_setup!`/`reduced_solve` into
   `../FLOWPanel.jl/src/FLOWPanel_formulation.jl` as a Householder
   Green-solve state alongside `GreenSolveState`
   (`_build_green_solve_state` dispatch; bordered route retained as
   debug/reference per the ruling), + test in FLOWPanel's suite.
   BEFORE touching FLOWPanel: read its `AGENTS.md`, `CLAUDE.md`,
   `agent_policies/WORKFLOW.md` (+`TESTING.md` before testing).
   `src/FLOWPanel_formulation.jl` and `test/formulation_test.jl` carry
   uncommitted 052e.1 work — preserve it. ≤4 threads locally.
2. **052e.2a continuation addendum, stages 2–4** (filament/particles),
   registered in accuracy plan §5; may parallel #1. Stage 5 folds into
   052e.3.
3. Later: 052e.3 hybrid fixture; 052e.1 regression completion after
   052b closes; 052e.6 gauge design study.

## Not yet done / open

- **Notebook entries for Tier 0B AND Tier 0B-R: offered twice, NO REPLY.**
  Offer again with verbosity options; approval required before writing.
- **Nothing committed.** All 052e work (Tier 0B results/supersession,
  .2b prereg/harness/results/adjudication, handoffs) is uncommitted;
  Ryan has not asked for a commit. If asked: 052e files ONLY.

## Repository/worktree cautions (unchanged)

- FastMultipole worktree (branch flowpanel-20260817) heavily dirty with
  unrelated higher-derivative work + untracked `HIGHER_DERIVATIVES/`.
  Touch only 052e files; never clean/stage/commit unrelated changes.
- `../FLOWPanel.jl` dirty (see #1 cautions).
- Laptop runs = formulation-proof tier, not official campaigns; official
  acceptance runs need committed, tagged campaign worktrees.
- Token policy: never read `data/**` CSVs directly — summarize by
  script; gates.txt and log tails carry registered values. Registered
  runs launch nohup-detached
  (`julia --project=../FLOWPanel.jl --threads=4`), logs into `data/`.
- Any new registered run needs its own dated, LOCKED prereg before
  launch (Ryan locks; no registered values before lock).
