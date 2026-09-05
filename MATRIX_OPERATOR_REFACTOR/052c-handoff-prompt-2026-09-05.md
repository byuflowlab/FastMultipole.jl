# 052c handoff — 2026-09-05 (trial-2 expint in flight)

## Immediate state

**Job 13592503** (`fp052c-trial2exp`, mgh-1-1 GH200, 6 h limit) is RUNNING:
052c trial-2 = 1080-step acceptance with the **exponential integrator
(`WAKE_EXPINT=true`, euler_exp) and NO sigma guards** (SIGMA_* unset),
testing whether the step-~1015 sigma collapse resolves without clamps
(Ryan ruling 2026-09-05, supersedes-for-now the trial-1 commit plan).

- Stage 1 (36-step mature gate vs pinned CPU **euler** reference):
  **PASSED** outright — expint fingerprint within locked tolerances.
  (Gate was made informational in the launcher; it passed anyway.)
- Stage 2 (1080-step, run_name `fm052d_gpu_1080_t2exp`): at step ~646/1080
  as of 14:0x local. Watch the historical window ~950–1080.
- Failure signatures if instability persists: euler_exp broadcast
  substep-budget throw (dt*|L| bound) or non-finite-ratio DomainError.
- Post-completion checks = same as trial-1e: launcher self-verifies
  (source gates 1080/0, n_steps, compare report/verify 0:1079); then run
  `fm052_gate.sh` numbers already in the launcher; record results in
  `052c-sigma-experiments-2026-08-26.md` under "Trial 2 … Results:".
- A local Monitor was armed in the old session (dies at reset). Re-arm:
  poll `ssh orc squeue -h -j 13592503` ~300 s; on completion grep
  `~/FLOWPanel-052-gh200/slurm-13592503.out` for `trial-2|gate|ERROR`.
- Log: `~/FLOWPanel-052-gh200/data/fm052d_gpu_1080_t2exp.log`; gate dir
  `data/fm052c_mature_gate_t2exp_13592503`.

## Infrastructure now in force (2026-09-05 rulings)

1. **One checkout, no silos.** Unified repos: orc `~/projects/{FLOWVPM.jl,
   FLOWPanel.jl,FastMultipole}` + envs `~/projects/envs/{aarch64,x86_64}`.
2. **Campaigns run from git worktrees pinned to commits** — policy now in
   `~/.claude/CLAUDE.md` ("Campaign Reproducibility"): all under-development
   deps pinned + worktreed, SHAs recorded in ledger BEFORE submitting, env
   Manifest dev-points at worktrees, outputs to consolidated data root,
   one worktree per agent/campaign (concurrency boundary).
3. **wt052 campaign worktrees** (branch `campaign-052`):
   `~/wt052/{FLOWVPM.jl,FLOWPanel.jl,FastMultipole}` at pinned SHAs
   FLOWVPM 3315b22 / FLOWPanel 4e6b5b7 / FastMultipole 3da58a1a, env
   `~/wt052/env-aarch64` (dev-repointed Manifest, depot fm052depot-gh200).
   Launcher: `~/projects/launchers/fp052c_expint_wt052_run.sh` (archived
   in MATRIX_OPERATOR_REFACTOR/scripts/) — run dirs/logs symlink/write to
   `~/projects/FLOWPanel.jl/data`. NOTE: wt052 FLOWPanel `data/` is a real
   dir (repo tracks some data files); per-run symlinks, not a dir link.
4. **Trial-2 is the LAST silo run** (launched from `~/FLOWPanel-052-gh200`
   pre-ruling; kept for tree-identity with trial-1e). **After it completes:
   retire the 052 silo checkouts** (gh200/h100/h200 × FLOWVPM/FLOWPanel/
   FastMultipole) + `fm052env-*` envs. The silo FLOWVPM carries the expint
   port as uncommitted files (backups `.bak-preexp`; identical change
   committed as unified 3315b22) — nothing to save before deletion.
   FLOWPanel `agent_policies/HPC.md` already declares the 052 family
   deprecated (018 uses its own `~/FLOWPanel-018-gpu-*` silos — NOT ours
   to delete).

## expint port facts (for debugging)

- GPU path = local FLOWVPM commit 8b00dbd (026 Phase 1b Task 1):
  `_euler_exp_broadcast!` + `_corespreading_eulerexp_broadcast!`.
- Ported to gh200 silo AND unified repo; the CoreSpreading viscous hunk is
  hand-applied (trees lack 75a55d7 splitting_state accumulators). Patch:
  `MATRIX_OPERATOR_REFACTOR/scripts/fp052c_trial2_expint_gpu_port.patch`.
- euler_exp requires ReformulatedVPM f==0 and rejects non-empty
  sigma_guard; driver env: `WAKE_EXPINT=true`, guard off when SIGMA_* unset
  (`rotor_hover_pressure_comparison.jl:1277-1287`).

## Repo hygiene (done 2026-09-05, local repos all CLEAN)

- FastMultipole `flowpanel-20260817`: 6 commits (052d/052h device
  cross-stencil reverse-leg src+tests ad5d8741; ledger d74673e0; handoffs
  abbf2c2b; prototypes 7663c05a; launchers 0ce3ba60; gitignore 9e6ac20e).
- FLOWVPM `flowpanel`: gitignore a627dd9 only.
- FLOWPanel `fastmultipole`: 7 commits incl. plans/ REMOVED 320f54d
  (39 tracked files, in history), harness 8299890, BRAINSTORM 9118a63.
- Archived (NOT committed): `~/repo_archive_20260905/<repo>/` — FM root
  experiment outputs + loose research notes (derivation_*.md,
  RIGID_MOTION plan, sfs_musings, lift_matrix, next_steps, …). Prune later.

## Open decisions for Ryan (carried forward)

1. Trial-1 vs trial-2 defaults: if trial-2 passes, choose expint-as-default
   vs guards-as-default (or both) — then execute the 052c commit plan
   (upstream port of whichever wins; fold OFAT into 053).
2. 052e tolerances T1–T5 ratification; 052b rulings (1r hover-window gate
   policy, Kutta :jump contract, 2r operator-mismatch investigation);
   053 defaults rows; storage 618G vs 400G cap.

## Next actions (new agent)

1. Re-arm job monitor for 13592503; on completion verify + record Trial 2
   results in the 052c ledger; report verdict to Ryan.
2. If PASS: present trial-1-vs-trial-2 defaults decision to Ryan.
3. After job ends (any outcome): retire 052 silos + fm052env-* (ask Ryan
   for final go), and make wt052 launcher the only 052c launch path.
