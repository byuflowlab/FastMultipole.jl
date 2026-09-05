# 052 phase handoff (session 2026-09-04a): 052c trial 1 PASSES; 2r "FMM floor" REFUTED (dense-route inert knobs); 1r hover-window recompute FAILS; kutta regression pinned to 7fbd68a

## Prompt for the next agent

You are continuing the MATRIX_OPERATOR_REFACTOR 052 phase after a context
reset. Read this file first. Mandate + FULL decision log live in
`052b-handoff-prompt-2026-09-02.md` (§Mandate, §Decision log — now 21
entries through 2026-09-04; append new decisions THERE). The 2026-09-03
handoff's §Permissions and §052c-background sections remain valid; the
2026-09-03b handoff is executed/superseded by this file. HPC.md
(`FLOWPanel.jl/agent_policies/HPC.md`) required before cluster work.
House rules verbatim in the 2026-09-02 handoff still apply.

## State at reset (verified 2026-09-05 pre-clear)

NOTHING running or queued in the 052 lane; no monitors to re-arm (all
session monitors ended with their jobs — but verify job states directly
at turn start anyway, never trust silence). Both mgh GH200 nodes IDLE
(mgh-1-2 recovered from the `maint` that NODE_FAILed trial-1d). Do not disturb fp-018gpu-* /
fp-p026ph1-* (other lanes). All four 2026-09-03b next-moves EXECUTED:

1. **gsdiag4 (13582074) — the 2r floor test was structurally INERT and
   the "FMM accuracy floors the residual" interpretation is REFUTED.**
   Residuals bit-identical (16 digits) to gsdiag2 despite body FMM
   (20,0.5) vs (17,0.7); banner honest but knobs never consulted:
   `FLOWPANEL_GPU_INFLUENCE=cuda` + production_route means all GS
   cross-influence/residual runs through the exact dense Float64 CUDA
   batch (`_gpu_rect_influence!` intercepts influence! at
   FLOWPanel_fmm.jl:60-79 / FLOWPanel_gpu_influence.jl:618-650), never
   FastMultipole.fmm!. The pinned 5.8224e-4 residual = ~5.8e-4
   inconsistency between residual-assembly operator and the block-solve
   operator (GS fixed point is exact; strength delta at machine eps).
   Root cause UNIDENTIFIED (candidates: wake-route contributions,
   Kutta/damping rows, residual formulation). RYAN RULING: raise GS_TOL
   / gate on strength delta / fund the operator-mismatch investigation.
2. **052c trial 1 PASSES (trial-1e = 13582234, COMPLETED 2:58:15).**
   Sigma_guard ceil port done on both orc silo FLOWVPMs (minimal hunk:
   3-key parser + clamp at both euler call sites; backups
   `*.bak-preceil`; do NOT whole-file copy — local file has
   splitting_state divergence the silos lack). Locked gates all PASS
   (~2x margins); min_sigma bottomed 9.558e-5 m @ step 983 (floor never
   clamped); 052f demotion fired mid-run and the run SURVIVED (the old
   step-894 crash scenario). Full tables + commit-plan proposal in
   `052c-sigma-experiments-2026-08-26.md` §Results (NEEDS RYAN).
   trial-1d (13582076) NODE_FAIL at step 693 was hardware, not code.
3. **1r accept harvest: hover-window Phase-2e recompute FAILS** (spread
   0.086 vs 0.005; ptp 0.47 vs 0.02; CT still drifting, 4.5 hover revs
   can't distinguish transient vs limit cycle; window mean CT 0.0877).
   NOT an accept-in-substance. Original CONVERGED=false was additionally
   confounded (CONVERGENCE_REVS=10 -> window from rev 1.0 ->
   window_in_hover guard). 7200 s gate hardcoded at carrier line 203.
   RYAN RULING: gate policy + more-revs vs cycle-mean acceptance.
4. **Kutta :jump regression PINNED to 7fbd68a exactly** (worktree bisect,
   FMM side held constant): parent 658/658 green; 7fbd68a reproduces
   HEAD's 2 failures (runtests_unit_kutta.jl:539-540); fallback strengths
   ~HALF legacy (body -0.111 vs -0.571; wake -0.313 vs -0.645) —
   factor-of-2 in the Dirichlet self-potential/wake-row convention
   change. Bisect worktree REMOVED at session end (scratchpad is
   session-scoped); to reproduce: `git worktree add <path> 7fbd68a^` in
   FLOWPanel.jl, then Pkg.develop the local FastMultipole + FLOWVPM.jl
   paths (the old committed Manifest pins registered FastMultipole
   v2.0.4, which predates `numtype` — dev-pathing is required). Logs
   kept nowhere; result fully recorded in the decision log. RYAN
   RULING: update `_kutta_trial!` vs relax bitwise contract.

## Next moves

1. Present Ryan the ruling menu (above 4 + carried flags below); nothing
   in the 052 lane is compute-blocked — all remaining work is
   decision-gated.
2. If Ryan funds the 2r operator-mismatch investigation: start from the
   dense-route finding (both cross-influence AND residual go through
   `_gpu_direct_batch!`); wake knobs (16/0.6) were identical across
   gsdiag2/4 — an easy discriminator is a run varying WAKE fmm knobs, or
   a step-0 no-wake case to rule the wake in/out.
3. On approval, paste `notebook-draft-2026-09-04.md` (this dir) into the
   journal (insertion: journals/20260901.md end-of-file; drafter's
   verbosity questions at top of the draft file).
4. Storage (INTENSIFIED): /home was ~618G vs 400G cap BEFORE trial-1e
   wrote full VTK. Archiver pass per HPC.md (`hpc-storage` agent from
   the FLOWPanel repo agents; Ryan-gated protect list).

## Flags for Ryan (consolidated)

- 2r solver policy (menu reframed by gsdiag4 — "pay for FMM accuracy" is
  STRUCK for the GPU route).
- 1r: 7200 s hardcoded gate; CONVERGENCE_REVS config; more revs vs
  cycle-mean policy.
- 052c commit plan (ledger §Results): upstream ceil port; adopt
  dtz_cap=0.5 + floor 1% default; fold OFAT candidates into 053.
- Kutta canonicality (7fbd68a).
- 052e T1-T6 tolerance ratification (052e-accuracy-plan-draft-2026-09-03.md);
  053 rows need Ryan (053-defaults-enumeration-draft-2026-09-03.md).
- Notebook draft approval (notebook-draft-2026-09-04.md).
- Storage ~618G+ vs 400G cap.
- Carried: two unapproved defaults from the 053 audit (FLOWPanel
  FMM_RADIUS_TOL inflation; FLOWVPM RadixFMM expansion_order=6 vs
  docstring 4).

## Key artifacts this session

- Decision log entries: 2026-09-04 x6 in 052b-handoff-prompt-2026-09-02.md.
- 052c ledger results: 052c-sigma-experiments-2026-08-26.md §Results.
- Notebook draft (held): notebook-draft-2026-09-04.md.
- orc logs: gsdiag4 `~/projects/FLOWPanel.jl/logs/slurm/slurm-fp-022g-2r-gsdiag4-13582074.{out,err}`;
  trial-1e `~/FLOWPanel-052-gh200/data/fp052c-trial1e-13582234.out`;
  gate report `~/FLOWPanel-052-gh200/data/fm052c_mature_gate_t1_13582234/fm052_gate.md`;
  run dir `.../data/fm052d_gpu_1080_t1/` (shared root).
- Silo backups: `~/FLOWVPM-052-{h200,gh200}/src/FLOWVPM_timeintegration.jl.bak-preceil`.
