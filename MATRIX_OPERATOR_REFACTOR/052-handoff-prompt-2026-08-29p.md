# 052 handoff (session 2026-08-29p): gpu40 died at adequacy gate, 052f all-direct fallback built — finish tests, patch silos, restart from step 1059

## Prompt for the next agent

You are continuing task 052. Session 2026-08-29p diagnosed the gpu40 failure,
built the 052f adequacy-gate fallback per Ryan's explicit instruction, and was
mid-validation at handoff. Do NOT redo any of it.

## What happened this session (established facts, do not re-derive)

1. **gpu40 gaussian production run 13512297 FAILED at step 1061/1475** (~29 of
   40 revs, GH200 mgh-1-2). NOT the leak — memory was flawless through all
   1060 steps (pool_reserved frozen 15.27 GB, pool_used trendless 11.7–13.2 GB,
   1060/1060 GPU gemvs, 0 CPU fallbacks, 0 NaN → leak fix fully validated
   under production physics). It died on FastMultipole's near-set adequacy
   gate (`_direct_kernel_geometry_gate!`, `src/translate_batched_resident.jl`):
   `g_min*h_leaf = 0.5545 < rho_t*sigma_max = 0.9896` at ell=2 ("admissible
   ell <= 1", forbidden). Root cause: viscous core spreading + merging grew
   sigma_max to 0.2066 m = **1.74 R** (R = 0.119 m, driver works in meters;
   typical core is ~0.046R — one fat merged far-wake particle drags the
   global gate). Step time had grown 4.9 → ~44 s/step by step 1060.
   The guard geometry CANNOT be cleared by any supported (ell, q): ell floor
   is 2, max-ladder q=20 gives only 1.34× margin where 1.79× was needed and
   sigma keeps growing.

2. **Ryan's explicit instruction (this session): change the safeguard to warn
   + fall back to all-direct (GPU-side) instead of throwing, then restart the
   job from the last step and finish it.** Ryan also confirmed the design:
   all-direct = zero M2L routes / no interaction list; the ell=2 4³ grid with
   full-grid near ball is kept (not one giant leaf) because the radix
   machinery requires ell >= 2 and the ≤4096-entry static cell-pair list is
   the GPU launch grid — same O(np²) pairs, better tiling.

3. **052f implementation (LOCAL, done, uncommitted):**
   - `FastMultipole/src/containers.jl`: `_SUPPORTED_RIGID_NEAR_RADII2`
     extended with 21,22,24,25,26,27 (23 has no lattice shell) + comment.
     q=27 makes every ell=2 leaf offset "near" → the supported 052c zero-M2L
     degenerate cache (gate vacuous, every pair direct, stays on device).
   - `FastMultipole/src/translate_batched_resident.jl`:
     (a) gate now builds `msg`, and for `HierarchicalRigidStencil` policies
     with non-TwoPassVortex regularized kernels emits `@warn ... maxlog=4` and
     returns `:alldirect` instead of throwing (TwoPassVortex still throws —
     its pass-2 sweep capacity is derived from gate-passing geometry and would
     AssertionError in the degenerate grid);
     (b) new `_alldirect_geometry_fallback!(cache, systems)` right before the
     `_twopass_device_reach_check` block: recenter!-style rebuild-and-swap
     with same bounds/capacities/options/sfs/adaptive but ell=2, q=27, and a
     policy whose epsilon is RE-DERIVED via
     `rigid_stencil_epsilon(cfg.P_phi, maximum(cache.box_extent)/2, 2, 27;
     lamb_helmholtz=LH, TF)` (without this the construction classifier gate
     fails — first test run proved it);
     (c) host call site in `update_radix_state!` now
     `if ... gate ... === :alldirect; _alldirect_geometry_fallback!(cache,
     systems); return update_radix_state!(cache, systems); end` (recursion
     terminates: demoted cache has empty accepted_offsets → gate vacuous).
   - `FastMultipole/src/translate_batched_cuda.jl`: same pattern at the
     device call site inside `update_cuda_radix_state!` (~line 6878).
   - `FLOWVPM.jl/src/FLOWVPM_fmm_radix.jl` (LOCAL): `_radix_auto_geometry`'s
     terminal error → `@warn ... maxlog=4` + `return (2, 27)`. REQUIRED for
     the restart: on restart sigma is already fat at first cache build, and
     without this the FLOWVPM build-time check kills the run before
     FastMultipole's gate ever runs. (`level_radii2` defaults to `nothing`
     and the zero-M2L case is explicitly legal without a schedule — verified.)
   - `FastMultipole/test/device_system_interface_test.jl` updated: testset (d)
     RegularizedVortex fat-sigma construction now expects warn+demote
     (`@test_logs (:warn, r"near-set adequacy failed") match_mode=:any`,
     `fat_cache.ell == 2`, `isempty(fat_cache.accepted_offsets)`, parity vs
     `_interface_regularized_direct` at **5e-4** — the floor is the erf-free
     gaussianerf kernel approximation (outer branch ≤2.1e-4 abs, block comment
     at `translate_batched_resident.jl:654`), observed ~5e-5, NOT expansion
     error); partitioned-kernel adequacy test (~line 358) same pattern; the
     partitioned construction inside two-pass stage B (~line 589) same
     pattern + `isempty(part_cache.accepted_offsets)`; TwoPassVortex adequacy
     tests unchanged (still `@test_throws`).

4. **Test status: BOTH GREEN, local validation COMPLETE.**
   `test/device_system_interface_test.jl`: 36,363/36,363 pass, exit 0
   (run 5, after 4 fix iterations: classifier epsilon → test hessian=true →
   parity tolerance 1e-10→5e-4 → line-589 expectation).
   `test/adaptive_lifecycle_test.jl`: 327/327 pass, exit 0 (the
   cache.adaptive path skips the gate and is unaffected, as expected).
   Do NOT rerun; start at step 2 of the sequence below.

5. **Silo patch script ready:** `patch_052f.py` — in the old scratchpad AND
   already uploaded to `orc:~/patch_052f.py`. It applies the EXACT local
   edits (including the epsilon re-derivation fix) with count==1 assertions,
   verify-all-then-write, `.bak-052f` backups. Usage:
   `python3 ~/patch_052f.py <fastmultipole_root> <flowvpm_root>` ('-' skips).
   NOT YET APPLIED ANYWHERE. FLOWPanel needs no 052f change.

6. **Restart recipe (verified this session):**
   - Last complete saved step = **1059** (body VTU + wake VTM + particles VTP
     all present; 1060 is partial). Warm restart exists on the 018 silo
     (`FLOWPanel_warmstart.jl`, `simulate_warmstart!` at driver line 1303).
   - The dispatcher `rm -rf`s `data/$RUN_NAME` (no KEEP_PREV) or `rm -rf`s
     `data/$RUN_NAME.prev` (KEEP_PREV=true). `data/scr_p019_s038v_gpu40.prev`
     is the PROTECTED old leak-run dir (Ryan-only archive decision) — so do
     NOT use RHPC_KEEP_PREV. Instead FIRST rename:
     `mv ~/FLOWPanel-018-gpu-gh200/data/scr_p019_s038v_gpu40 \
        ~/FLOWPanel-018-gpu-gh200/data/scr_p019_s038v_gpu40.crashed1061`
   - Then submit from `~/FLOWPanel-018-gpu-gh200`:
     `sbatch --job-name=fp-il-s038v-gpu40 \
        --export=ALL,SCR_GPU_RESERVE_GIB=16,RESTART_STEP=1059,RESTART_NAME=scr_p019_s038v_gpu40,RESTART_PATH=data/scr_p019_s038v_gpu40.crashed1061 \
        examples/run_p018_screen_gpu.slurm.sh gh200 scr_p019_s038v_gpu40`
   - Restart is physically sound but not bit-exact (particle saves are f32;
     loader warns once). Monitor CSVs TRUNCATE on restart → harvest must
     stitch `.crashed1061/monitors/*.csv` (steps ≤1059) with the new dir's
     (steps ≥1060), no row overlap. VTK series in the new dir starts at 1060.
   - Expected on restart: FLOWVPM auto-geometry warns "no admissible radix
     depth ... Falling back to the all-direct zero-M2L grid" and builds the
     degenerate cache directly. Expect slower steps (all-direct self-pass at
     np ~ few 100k; step was already 44 s). 415 steps remain; 24 h wall is
     plenty unless all-direct is brutal — watch the first ~20 step times.

7. **LineGauss gpu40 pair (NEW stack, H200): jobs 13512551 (m13h) +
   13512552 (eng), same name `fp-il-s038v-gpu40lg`, --dependency=singleton,
   still PENDING at handoff.** Submitted with
   `--export=ALL,SCR_GPU_RESERVE_GIB=16,FLOWPANEL_FILAMENT_REG=linegauss,RHPC_KEEP_PREV=true`
   via the NEW launcher `~/FLOWPanel-052-h200/examples/run_p018_screen_gpu052.slurm.sh`
   (created this session; wraps this silo's dispatcher, fm052env-h200 depot,
   052c GPU gate; dispatcher got a `scr_p019_s038v_gpu40` NREVS=40 arm +
   sourced-from-launcher guard, backup `.bak-052gpu40`). Singleton = mutual
   exclusion (only blocks while the twin RUNS); a monitor (old session) was
   set to scancel the loser when one starts — that monitor DIES with the old
   session, so RE-ARM an equivalent monitor (poll
   `squeue -h -u $USER -n fp-il-s038v-gpu40lg -o "%i %T"` ~5 min; when one is
   RUNNING scancel the other; then watch to terminal state; on terminal state
   report last steps + GATE lines + memory tail from
   `~/FLOWPanel-052-h200/logs/slurm/slurm-fp-il-s038v-gpu40lg-<id>.{out,err}`).
   **These jobs WILL hit the same adequacy death at ~step 1061 unless the
   052f patch lands on the 052-h200 silos before they reach it** (long queue
   → patch will land first; if one starts before patching, it still runs
   ~1060 good steps).
   IMPORTANT: earlier this session the leak fixes were synced to the
   052-h200 silos (FastMultipole `translate_batched_cuda.jl`, FLOWVPM
   `FLOWVPM_fmm_radix.jl`, FLOWPanel `FLOWPanel_gpu_influence.jl`; backups
   `.bak-052leak`) — those silo files now match local pre-052f, so
   patch_052f.py applies cleanly there too.

8. **cpu40 13508968**: RUNNING, ~step 393/1475 at ~11 h, steps ~140 s and
   growing. Will hit the same gate at ~step 1061 (shared code) in roughly a
   day+, and may hit its wall first. No action taken; flag to Ryan whether to
   let it die (29-rev partial), scancel, or (if its FLOWVPM tree is patchable
   and Ryan wants) restart later with 052f. Its stack runs from
   `~/projects/FLOWPanel.jl` (logs `slurm-fp-il-s038v-cpu40-13508968.*`).

## YOUR IMMEDIATE SEQUENCE

1. DONE — both local test files confirmed green (item 4). Skip to step 2.
2. Apply the silo patches (ssh orc, bash -lc):
   `python3 ~/patch_052f.py ~/FastMultipole-018-gpu-gh200 ~/FLOWVPM-018-gpu-gh200`
   `python3 ~/patch_052f.py ~/FastMultipole-052-h200 ~/FLOWVPM-052-h200`
   If an assertion fails, diff that silo file's region vs local and adapt —
   the 018 gate region was verified byte-identical to local pre-052f.
3. Rename the crashed dir and submit the restart (item 6 exactly).
4. Arm monitors: (a) restart job — watch `.err` for the expected fallback
   @warn, step times, `source_s_gpu_memory` flatness, terminal state + GATE;
   (b) re-arm the LG-pair twin-cancel monitor (item 7).
5. When the restart lands: report; harvest needs the CSV stitch (item 6).
   Confirm/append sigma HANDOFF landing note
   (`FLOWPanel.jl/plans/sigma_vpm_illustrations_20260827/HANDOFF.md`,
   append-only, NEVER tick its checkboxes).

## House rules (carried forward)

4 threads max locally; julia-test-runner for all runs; refactor-docs-librarian
for MATRIX_OPERATOR_REFACTOR doc questions; verifier before reporting claimed
numbers; never read `data/**`/`*.csv`/`*.bin`; long output → scratchpad log
then grep. ssh: `ssh orc 'bash -lc "source /etc/profile; ..."'`, banners noisy
(grep -v), auth expires → ask Ryan to run `! ssh orc echo ok`. Read memory
`orc-cluster-access.md`. rsync `--checksum`. GPU jobs authorized; combine
stages; eng+m13h parallel submit for H200; `mgh --gres=gpu:gh200:1
--constraint=arm` for GH200. Notebook writes need Ryan's approval first
(draft via notebook-drafter).

## Uncommitted state (do not lose; Ryan decides commits)

- NEW this session (052f): FastMultipole `src/containers.jl`,
  `src/translate_batched_resident.jl`, `src/translate_batched_cuda.jl`,
  `test/device_system_interface_test.jl`; FLOWVPM
  `src/FLOWVPM_fmm_radix.jl`. Plus (silo-side, this session): the 052-h200
  leak-fix syncs (item 7 note), the 052-h200 launcher + dispatcher arm, and
  `orc:~/patch_052f.py`.
- Prior uncommitted (unchanged): FastMultipole leak fix
  (`translate_batched_cuda.jl` capacity buffer) + branch `flowpanel-20260817`
  session-k fixes + prototypes; FLOWVPM GC-after-cache-drop; FLOWPanel branch
  `fastmultipole` cross-pass A–F + LineGauss + default pin +
  `_cross_padded_positions!` leak fix; 018-gh200 silo `.bak-052leak` patches.

## Open Ryan decisions (ask, don't assume)

- cpu40: let it die at ~1061 vs scancel vs patch+restart its stack.
- Device-path 052f validation: the CUDA call-site change is untested on GPU
  (no local GPU; `test/cuda_radix_interface_test.jl:169` expects the OLD
  device-gate throw and will need updating when next run on a GPU node).
  The restart itself exercises the FLOWVPM build-time fallback (host code
  choosing geometry), not the mid-run CUDA gate path.
- Reverse-leg implement-now vs defer (assessment delivered 2026-08-29o).
- Commit breakdown for leak fixes + 052f + prior 052d work.
- Notebook entries: 052d closure, leak fix, gpu40 death + 052f fallback
  (draft via notebook-drafter, get approval + detail level BEFORE writing).
- xverify-gate routine vs debug-only; LineGauss near-field perf check (the
  LG run's early step times vs gaussian's ~4.9-5.3 s/step is the measurement).
- Old gpu40 `.prev` + `.crashed1061` + probe dir cleanup (archive-first,
  Ryan only).
