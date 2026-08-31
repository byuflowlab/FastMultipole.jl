# 052 handoff (session 2026-08-30r): silos patched, restart FAILED on zero-M2L device bug, ALL JOBS PAUSED — schedule 24h02m wakeup, then relaunch

## Prompt for the next agent

You are continuing task 052. Session 2026-08-30r applied the 052f silo
patches, submitted the gpu40 restart (it FAILED — new device-path bug, see
item 2), and then, on Ryan's explicit instruction, PAUSED all cluster jobs
and prepared this reset. **Your first action: schedule a wakeup for exactly
24 hours and 2 minutes after your session starts; at that wakeup, relaunch
everything per the recipes below.** Until the wakeup, do nothing else unless
Ryan says otherwise.

Wakeup mechanics: ScheduleWakeup clamps to 3600 s max, so a single call
cannot span 24 h. Use CronCreate (load via ToolSearch) to create a one-shot
scheduled trigger at now+24h02m, or the `schedule` skill if a local-capable
one-time routine is available; as a last resort chain hourly
`ScheduleWakeup` noop ticks (24 of them) until the target time. Record the
absolute target time in your first reply so Ryan can see it.

## What happened session 2026-08-30r (established facts, do not re-derive)

1. **052f silo patches applied successfully** to BOTH silo pairs
   (`~/FastMultipole-018-gpu-gh200` + `~/FLOWVPM-018-gpu-gh200`, and
   `~/FastMultipole-052-h200` + `~/FLOWVPM-052-h200`) via
   `python3 ~/patch_052f.py` — all 8 files patched, `.bak-052f` backups, no
   assertion failures. Do NOT re-apply.

2. **gpu40 restart job 13513892 FAILED after 6 min** (submitted per the
   2026-08-29p recipe: RESTART_STEP=1059, RESTART_PATH=
   `data/scr_p019_s038v_gpu40.crashed1061`, partition mgh). The warm restart
   itself worked (`simulate_warmstart!: resuming from step 1059`), and the
   FLOWVPM 052f fallback fired correctly:
   `Warning: no admissible radix depth ... Falling back to the all-direct
   zero-M2L grid (ell = 2, q = 27)` at `FLOWVPM_fmm_radix.jl:507`. But at
   step 1060 the CUDA resident pipeline STILL tried to launch hierarchical
   M2L: `ERROR: LoadError: TypeError: in typeassert, expected
   CuArray{Int32,1}, got a value of type Nothing` in
   `_launch_cuda_hierarchical_m2l_cached!` at
   `FastMultipole/src/translate_batched_cuda.jl:8069` (call chain
   `_launch_cuda_resident_operator_pipeline!` → `_launch_cuda_resident_m2l!`
   :4930 → `_launch_cuda_hierarchical_m2l!` :7986 → :8069). Root cause: in
   the degenerate zero-M2L cache (empty accepted_offsets) the M2L plan
   arrays are `nothing`, and the device operator pipeline has NO empty-routes
   skip — it unconditionally launches M2L. This is exactly the "device-path
   052f untested on GPU" risk flagged in the 29p handoff.
   **REQUIRED FIX before gpu40 relaunch**: in the device resident pipeline,
   skip the hierarchical M2L launch (and any stage depending on M2L plans)
   when the state/cache has no accepted offsets (plan/route arrays are
   `nothing`). Make the fix locally in
   `FastMultipole/src/translate_batched_cuda.jl`, mirror the logic used by
   the host path (which handles the empty case — local tests were green),
   extend/append to `~/patch_052f.py` (or a new `patch_052g.py`) with the
   same count==1 assert + verify-then-write pattern, apply to BOTH silos.
   No local GPU: CPU tests can't exercise it; get correctness by inspection
   + the restart run itself. Also remember `test/cuda_radix_interface_test.jl:169`
   still expects the OLD device-gate throw (update when next on a GPU node).
   Log: `~/FLOWPanel-018-gpu-gh200/logs/slurm/slurm-fp-il-s038v-gpu40-13513892.{out,err}`.
   Timing: whether to develop the fix immediately (during the 24 h pause)
   or at wakeup is Ryan's call — ASK if he's around; if unreachable at
   wakeup time, do the fix at wakeup BEFORE resubmitting gpu40.

3. **PAUSE STATE (executed ~2026-08-30 00:20, all verified):**
   - LG twin pair **13512551** (m13h) + **13512552** (eng), name
     `fp-il-s038v-gpu40lg`, `--dependency=singleton`: PENDING, **held**
     (`scontrol hold`; squeue reason JobHeldUser). Original submit lines
     preserved (sacct SubmitLine verified):
     `sbatch --job-name=fp-il-s038v-gpu40lg --dependency=singleton
     --partition=m13h [--partition=eng --qos=eng for 13512552]
     --export=ALL,SCR_GPU_RESERVE_GIB=16,FLOWPANEL_FILAMENT_REG=linegauss,RHPC_KEEP_PREV=true
     examples/run_p018_screen_gpu052.slurm.sh h200 scr_p019_s038v_gpu40`
     from `~/FLOWPanel-052-h200`.
   - **13513873** `fp-il-s038v-cpu40-r2` (m12, workdir
     `~/projects/FLOWPanel.jl`): **scancelled** at step ~431/1475
     (~150-170 s/step). Its submit line was:
     `sbatch --job-name=fp-il-s038v-cpu40-r2 --time=36:00:00
     --export=ALL,RESTART_STEP=423 examples/run_p018_screen_hpc.slurm.sh
     scr_p019_s038v_cpu40`. NOTE: this job was launched by Ryan outside the
     previous session (it answers part of the old "cpu40: let it die?"
     question — he restarted it). Its stack was IDENTIFIED
     (Manifest paths: `~/projects/FastMultipole`, `~/projects/FLOWVPM.jl`)
     and, with Ryan's explicit approval this session, PATCHED with 052f
     (`python3 ~/patch_052f.py ~/projects/FastMultipole
     ~/projects/FLOWVPM.jl` — all 4 files OK, `.bak-052f` backups). Do NOT
     re-apply. cpu40 runs the host path only, so the CUDA zero-M2L bug
     (item 2) cannot hit it; r3 will warn+fall back at the fat-sigma gate
     instead of dying.
   - gpu40 restart 13513892: already FAILED (item 2) — nothing to pause.
   - Also observed (Ryan's own actions, no response needed): 018-silo jobs
     13513063–65 and 13513885–87 were CANCELLED by user before the pause;
     old cpu40 13508968 gone.

4. **Data-dir state (018 silo `~/FLOWPanel-018-gpu-gh200/data/`):**
   - `scr_p019_s038v_gpu40.crashed1061` — the GOOD archive (steps ≤1059
     complete; 1060 partial). Restore source for gpu40 relaunch.
   - `scr_p019_s038v_gpu40` — worthless partial dir from the 6-min FAILED
     run; the dispatcher will `rm -rf` it on resubmit (no KEEP_PREV) —
     that is fine.
   - **`scr_p019_s038v_gpu40.prev` (the protected leak-run archive) NO
     LONGER EXISTS** — it was gone before this session touched anything.
     Presumably Ryan archived/removed it; FLAG to him, do not investigate
     destructively.

## RELAUNCH RECIPES (execute at the 24h02m wakeup)

1. LG pair: `scontrol release 13512551 13512552`. Then re-arm the
   twin-cancel monitor as NOTIFY-ONLY (a monitor embedding `scancel` gets
   blocked by the permission classifier — learned twice): poll
   `squeue -h -u $USER -n fp-il-s038v-gpu40lg -o "%i %T"` ~5 min; when one
   is RUNNING, the monitor notifies and YOU run the `scancel <loser>`
   yourself in the foreground; then watch to terminal state; report last
   steps + GATE lines + memory tail from
   `~/FLOWPanel-052-h200/logs/slurm/slurm-fp-il-s038v-gpu40lg-<id>.{out,err}`.
   (If held jobs age poorly in priority, that's acceptable — do not
   resubmit; release keeps all original params + singleton.)
2. cpu40: verify the dispatcher's restart behavior is dir-preserving when
   RESTART_STEP is set (the r2 submit reused the same data dir — check
   `examples/run_p018_screen_hpc.slurm.sh` before submitting), find the
   last complete saved step S in
   `~/projects/FLOWPanel.jl/data/scr_p019_s038v_cpu40` (body VTU + wake VTM
   + particles VTP all present, per the 29p item-6 criterion), then from
   `~/projects/FLOWPanel.jl`:
   `sbatch --job-name=fp-il-s038v-cpu40-r3 --time=36:00:00
   --export=ALL,RESTART_STEP=<S> examples/run_p018_screen_hpc.slurm.sh
   scr_p019_s038v_cpu40`
3. gpu40: ONLY after the item-2 fix is applied to the 018 silo. Then from
   `~/FLOWPanel-018-gpu-gh200`:
   `sbatch --job-name=fp-il-s038v-gpu40
   --export=ALL,SCR_GPU_RESERVE_GIB=16,RESTART_STEP=1059,RESTART_NAME=scr_p019_s038v_gpu40,RESTART_PATH=data/scr_p019_s038v_gpu40.crashed1061
   examples/run_p018_screen_gpu.slurm.sh gh200 scr_p019_s038v_gpu40`
   (no dir rename needed this time — current partial dir is disposable).
   Arm a monitor (state changes / fallback-warn grep / terminal+GATE; use
   tagged-output parsing `ST8:`/`SA8:` prefixes — raw squeue-through-ssh
   picks up login-banner ANSI noise). Expect: warm restart from 1059,
   FLOWVPM fallback warn, then all-direct steps — watch the first ~20 step
   times (baseline was 44 s/step at death; 415 steps remain in 24 h wall).
4. When gpu40 lands: report; harvest = stitch
   `.crashed1061/monitors/*.csv` (steps ≤1059) with new dir's (≥1060), no
   overlap; VTK series in new dir starts at 1060. Confirm/append sigma
   HANDOFF landing note
   (`FLOWPanel.jl/plans/sigma_vpm_illustrations_20260827/HANDOFF.md`,
   append-only, NEVER tick its checkboxes).

## House rules (carried forward)

4 threads max locally; julia-test-runner for all runs;
refactor-docs-librarian for MATRIX_OPERATOR_REFACTOR doc questions;
verifier before reporting claimed numbers; never read `data/**`/`*.csv`/
`*.bin`; long output → scratchpad log then grep. ssh:
`ssh orc 'bash -lc "source /etc/profile; ..."'`, banners noisy (grep -v),
auth expires → ask Ryan to run `! ssh orc echo ok`. Read memory
`orc-cluster-access.md`. rsync `--checksum`. GPU jobs authorized; combine
stages. Notebook writes need Ryan's approval first (draft via
notebook-drafter). Monitors embedding scancel/destructive cmds get
classifier-blocked — make monitors notify-only.

## Uncommitted state (do not lose; Ryan decides commits)

Unchanged from 29p handoff plus: the applied `.bak-052f` silo patches
(both silo pairs AND the cpu40 stack `~/projects/FastMultipole` +
`~/projects/FLOWVPM.jl`), and (upcoming) the item-2 zero-M2L device-skip
fix. Local:
052f files (FastMultipole `src/containers.jl`,
`src/translate_batched_resident.jl`, `src/translate_batched_cuda.jl`,
`test/device_system_interface_test.jl`; FLOWVPM `src/FLOWVPM_fmm_radix.jl`),
leak fixes, branch `flowpanel-20260817` session-k fixes + prototypes,
FLOWPanel branch `fastmultipole` cross-pass A–F + LineGauss, FLOWVPM
GC-after-cache-drop.

## Open Ryan decisions (ask, don't assume)

- Timing of the zero-M2L device fix (now vs at wakeup) — item 2.
- `.prev` archive disappearance (item 4) — confirm it was his action.
- Commit breakdown (leak fixes + 052f + 052d work); notebook entries
  (052d closure, leak fix, gpu40 death + 052f fallback + new device bug) —
  draft via notebook-drafter, approval + detail level BEFORE writing.
- Reverse-leg implement-now vs defer (assessment delivered 2026-08-29o).
- xverify-gate routine vs debug-only; LineGauss near-field perf check.
- Old gpu40 `.crashed1061` + probe dir cleanup (archive-first,Ryan only).
