# 052b handoff (session 2026-09-01b): eng+mgh carriers launch cleanly; NEXT: device-aware IGE driver paths, then 2r backslash

## Prompt for the next agent

You are continuing task 052b after a context reset. Read this file first; it
is authoritative. Predecessor of this file:
`052b-handoff-prompt-2026-09-01.md` (background only — its "NEXT" items are
superseded here). HPC.md (`FLOWPanel.jl/agent_policies/HPC.md`, local copy
authoritative) is REQUIRED READING before cluster work. Ryan's instructions,
in order:

1. **Fix the 1r IGE GPU blocker** (§Current blocker: scalar indexing in the
   driver's IGE-only paths), re-run the eng chain and the mgh smoke→probe,
   and periodically check progress (§Monitors).
2. **Get the 2-rotor case working — backslash solver first** (§2r scope,
   Ryan's words 2026-09-01). It might already be working; verify before
   changing anything.
3. Longer arc: prepare for making all matrix-free solvers work on GPU.

## 2r scope (Ryan, verbatim intent, 2026-09-01)

- First, just for the **Backslash solver** (it might already work): make sure
  the **induced potential due to the OTHER rotor's doublet panels is included
  in the outer Gauss-Seidel solve loop**.
- **All interactions computed on the GPU using FMM — fully O(N)** — EXCEPT:
  (a) the influence matrix used to solve for doublet strengths, and
  (b) perhaps the self-induced potential due to source panels.
- This prepares a future step: **all matrix-free solvers working on GPU**.
- Relevant code: driver `body_solvers` tuple (per-rotor `pnl.Backslash` +
  `FlatGroundSolver`, see the MethodError signature in
  `logs/chain/p022g_1r_ige_smoke_13549598.log`), FLOWPanel block-GS outer
  loop (`src/FLOWPanel_solve*.jl` — find where cross-system induced potential
  enters the RHS), formulation `VelocityThroughSources` with
  scalar-potential body coupling for Dirichlet targets (phase_06 contract §1).
- CPU evidence 2r works in principle: 13484022 (2r_oge CPU) COMPLETED;
  13484023 (2r_ige CPU) died only of PARTICLE OVERFLOW at a 500k cap
  (older tree; unified driver default is `nrotors*500_000` = 1M for 2r,
  verified at `rotor_hover_ground_effect.jl:830`). 4r arms: 13484024
  TIMEOUT (CPU slow), 13484025 NODE_FAIL — no correctness signal.
- Run 2r via the generalized chain (§Chain) on whichever pool is free;
  population will be near ~500k by end of accept (CPU 2r_ige overflowed
  500k at ~rev 12.7; accept is 11.5 revs) — the ≥20% device-memory gate is
  load-bearing for 2r.

## Current blocker (eng attempt 4, job 13549642 — VERIFY state first)

`ERROR: Scalar indexing is disallowed` on
`CuArray` during the smoke, at `ground_diagnostics_monitor`
(`examples/rotor_hover_ground_effect.jl:1769`, Main closure) — it getindexes
`pfield.particles` (a device matrix under `VPM_ARRAYTYPE=cuarray`).
The SAME pattern exists in the IGE damp-band `propagate!` override
(driver ~:1584-1593: per-particle for-loop reading/writing
`w.pfield.particles[u_ax_index, i]`) — it will fail next once the monitor is
fixed. These paths only run for IGE arms with `GROUND_DAMP_BAND_R>0`
(=0.1 for p022g_*_ige), which is exactly why all hr-sweep/OGE GPU runs never
hit them. mgh smoke 13549643 runs identical code — expect the same failure.

Fix approach (next agent decides detail): make the two IGE driver paths
device-aware. The damp-band update is a masked broadcast
(`f = clamp.(d./band,0,1)` applied where `u_ax .> 0 .&& d .< band`) — do it
with array ops on device views of rows `X_INDEX[ax]`/`U_INDEX[ax]`;
the diagnostics monitor can work on a host copy
(`Array(view(particles, :, 1:np))` once per monitor call — diagnostics
cadence, not per-particle hot path). Grep the whole driver for other
`particles[` scalar loops guarded by ground/IGE flags before resubmitting
(there are several ground diagnostics helpers near :1700-1800; also
`_particle_gamma_direction_stats` if `diagnose_particle_gamma` ever true).
Julia scalar-index test locally is impossible (no GPU) — re-run the eng
chain as the test. Commit driver fixes on orc `unified-052`, mirror to local
tree (leave local uncommitted, standing policy).

## Session results (2026-09-01b — verified, don't redo)

Four launch-blockers found and fixed this session, in order:

1. **13549590 died in 2 s**: `local mode=$1 name=$2 log=...${mode}...` in
   the chain wrapper — bash expands ALL args of one `local` before the
   builtin runs, so `${mode}` was unset → fatal under `set -u`, stderr only
   in the separate `.err` file. Fixed by splitting the `local` lines.
   Commit `8c9c57a` (orc `unified-052`, FLOWPanel). Lesson recorded in the
   script comment.
2. **Attempt 3 (13549598) smoke MethodError**: `simulate!` forwards
   `sigma_guard` (052c trial-1 kwarg) to `propagate!`; the driver's IGE
   damp-band override didn't accept it. Fixed to mirror the src signature
   (accept + forward to `FLOWVPM._euler`; reject for `euler_exp`), commit
   `a4571aa`. Note: this override is defined at RUNTIME only when
   `ground_enable && ground_damp_band_r > 0`.
3. **mgh/ARM bring-up (WORKING through carrier launch)**:
   - Carrier `run_rotor_multi_ground_effect_gpu.slurm.sh` is now
     arch-dispatched (commit `3392d96`): `PROJECT` defaults to
     `envs/$(uname -m)`; aarch64 → `~/julia/julia-1.11.7/bin/julia`
     (ARM ELF, verified) + `JULIA_DEPOT_PATH=~/fm052depot-gh200`, no module
     load; x86 path unchanged; all 4 julia call sites via `$JULIA_BIN`;
     `P022G_JULIA_OVERRIDE` exists.
   - **ARM depot artifact gap closed**: mgh compute nodes have NO internet;
     first ARM `Pkg.instantiate()` wanted the aarch64 cuda+13.3
     `CUDA_Runtime` artifact. Fixed from a login node:
     (a) `JULIA_PKG_PRECOMPILE_AUTO=0 Pkg.instantiate()` with the ARM depot,
     (b) `~/seed_arm_artifacts.jl` (orc home; walks the aarch64 Manifest,
     `ensure_artifact_installed` for Platform aarch64-linux cuda=13.3) —
     seeded 33 artifacts, 0 failed; (c) `CUDA_Runtime` itself needed
     seeding BY HASH (augmented cuda_local tag defeats plain selection):
     `curl https://pkg.julialang.org/artifact/<tree-sha1>` untarred into
     `~/fm052depot-gh200/artifacts/<tree-sha1>` (1.6G). If another artifact
     is ever missing, the job error names the hash — same curl+untar recipe.
   - **Stage-0 ARM smoke PASSED** (job 13549609, `~/projects_unified/
     smoke_unified_arm.slurm.sh`): `UNIFIED OK 1.11.7`, `CUDA OK NVIDIA
     GH200 480GB`, `EXT OK FLOWVPMCUDAExt loaded`. First-ever ARM
     precompile of the stack is done and cached in the depot.
   - Carrier smoke launched fine on mgh (13549643): banner, revisions, GPU
     gate `GH200` all correct. It will fail on the §Current blocker like eng.
4. **Chain generalized** (commit `a779fea`): case tag is `$1`, default
   `p022g_1r_ige`. For 2r:
   `sbatch -J fp-022g-2r-ige-chain -p <pool> ... p022g_1r_ige_gpu_chain.slurm.sh p022g_2r_ige`.

Pins on orc `unified-052` at reset: FLOWPanel `a4571aa`, FastMultipole
`89ede6b`, FLOWVPM `6c8cda4`. Local FLOWPanel working tree mirrors all
orc-committed changes but stays uncommitted (standing policy).

## Live jobs at reset (VERIFY FIRST — stale by now)

- **13549642** eng chain attempt 4: smoke FAILED 13:12 MDT (scalar
  indexing, §Current blocker), chain aborted, job left queue. Log:
  `logs/chain/p022g_1r_ige_smoke_13549642.log`.
- **13549643** mgh carrier smoke (`p022g_1r_ige_smoke_mgh`): CONFIRMED
  failed the same way (scalar indexing) ~13:40 MDT — no need to re-check. Log:
  `logs/slurm/slurm-fp-022g-1r-ige-smoke-mgh-13549643.out`. Its value was
  proving the ARM launch path — already proven.
- Do not disturb: 13548847 `fp-022lg-hr10` CPU production (m12, 72 h),
  13518479, 13542905, 13542776.
- Submit patterns: eng = `-p eng --qos=eng --gres=gpu:h200:1` (starts fast
  by preempting standby); mgh = `-p mgh --qos=gpu --gres=gpu:gh200:1
  -C arm` (2 nodes, mgh-1-1 preferred, MaxTime 24 h, sbatch REJECTS
  without `-C arm`). m13h was 800+ deep. Probe with the slurm-availability
  skill before submitting anyway.

## Monitors (re-arm; they die at context reset)

Persistent local Monitor per job, poll ~300 s: ssh orc → `squeue -h -j <id>
-o %T` + grep the chain/slurm log for
`=== chain stage|CHAIN |projection:|ERROR|CANCELLED|TIME LIMIT` (+ tail the
`.err` — TWO launch failures this session lived only there). GOTCHAS: orc
login banner prepends ANSI junk — leading `echo`, never anchor greps, strip
with `sed $'s/\x1b\\[[0-9;]*m//g'`. The auto-mode permission classifier
allowed all read-only ssh AND this session's scp/sbatch/git-commit one-shots;
`.claude/settings.local.json` has NO ssh allow entries (Ryan never added
them) — it worked anyway this session.

## Chain (unchanged mechanics)

`examples/p022g_1r_ige_gpu_chain.slurm.sh <case>` — ONE allocation: smoke →
probe (2 rev) → projection gate (median of last 30 `step_timer total_step`
probe lines ×414×1.5 ≤ 6480 s AND probe `device_memory_reserve_fraction`
≥ 0.20 from `data/<case>_probe/*_case_metadata.toml`) → 414-step accept.
4 h wall, 16c/96G. `P022G_REQUIRED_GPU_MODEL` default H200 — export
`GH200` when chaining on mgh. Expected: 052 baseline 3.124 s/step (H200)
⇒ accept ~22–30 min; 2r ≈ 2× that, budget-fine. No s/step measured yet
this session (no probe has ever completed on the GPU carriers).

## Flagged to Ryan (unanswered)

- **Storage**: `/home/rander39` was 618G vs the 400G cap at last session's
  reset, archiving exhausted, sweeper approval queue pending
  (HANDOFF_CPU_20260901.md §Pending Ryan decisions). Every smoke/probe adds
  VTK. Raise before the accept/2r runs write ~10s of GB.
- **Notebook**: GPU-route unblock + this session's four fixes are un-logged.
  Offer an entry at the next milestone; ask verbosity first (house rule:
  approval before any notebook write).

## Key files

- `MATRIX_OPERATOR_REFACTOR/052b-impl-multirotor-ige-gpu.md` — task doc
  (budget ≤2 h/case, ≤20 s/step; case matrix)
- orc `~/projects_unified/FLOWPanel.jl` — launch tree (branch unified-052):
  `examples/rotor_hover_ground_effect.jl` (driver; IGE override ~:1556,
  damp loop ~:1584, ground_diagnostics_monitor ~:1769, max_particles :830),
  `examples/run_rotor_multi_ground_effect_gpu.slurm.sh` (carrier),
  `examples/p022g_1r_ige_gpu_chain.slurm.sh` (chain)
- `FLOWPanel.jl/BRAINSTORM/022_rotor_hover_ground_effect/
  phase_06_multirotor_gpu.md` — acceptance contract (top section is
  authoritative)
- `FLOWVPM.jl/ext/FLOWVPMCUDAExt.jl:918-967` (device overloads);
  FastMultipole `test/cuda_radix_lifecycle_test.jl` (parity template)
- orc `~/seed_arm_artifacts.jl` + `~/seed_arm_artifacts_20260901.log`,
  `~/projects_unified/smoke_unified_arm.slurm.sh` (keep; mgh recovery kit)

## House rules (carried forward)

4 threads max locally; julia-test-runner for runs/scripts (output →
scratchpad log, grep it); refactor-docs-librarian for MATRIX_OPERATOR_REFACTOR
doc questions; verifier before reporting claimed numbers; never read
`data/**`/`*.csv`/`*.bin` raw; notebook writes need Ryan's approval FIRST;
commits/pushes on local trees only on Ryan's ask (orc unified-052 pin
commits before launch are sanctioned); GPU jobs authorized; scp scripts to
orc instead of nested ssh quoting; rsync --checksum; slurm needs `bash -lc`;
auth expiry → ask Ryan `! ssh orc echo ok`; probe partitions with the
slurm-availability skill before submitting; linegauss CONFIRMED for p022g
arms (Ryan 2026-09-01) — do not revert.
