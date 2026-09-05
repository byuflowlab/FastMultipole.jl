# 052b handoff (session 2026-09-01): 1r IGE GPU chain submitted on eng — NEXT: mgh/ARM bring-up, then 2r IGE

## Prompt for the next agent

You are continuing task 052b after a context reset. This session UNBLOCKED
the parked GPU route and submitted the 1-rotor IGE case; read §Session
results before touching anything. Ryan's instructions for you, in order:

1. **Check job 13549590** (the eng H200 chain — §Live job) and re-arm the
   monitor (§Monitor). Do NOT cancel it, whatever state it is in.
2. **Get the GPU carrier working on `mgh` (GH200/ARM)** — §mgh bring-up.
   Two GH200 nodes were idle at reset while the x86 H200 queues were deep.
3. **Then run the 2-rotor ground-effect case** (`p022g_2r_ige`) — §2r IGE.

House rules per §House rules. HPC.md (`FLOWPanel.jl/agent_policies/HPC.md`,
local copy authoritative) is REQUIRED READING before cluster work.

## Session results (2026-09-01, this session — verified, don't redo)

- **Root cause of the parked GPU blocker** (HANDOFF_CPU_20260901.md §Parked):
  orc's unified FLOWVPM declares the `FLOWVPMCUDAExt` package extension in
  Project.toml but had NO `ext/` directory, so the device-resident
  `source_to_buffer!`/`buffer_to_target!` overloads never loaded (extension
  load failure is a non-fatal warning; FastMultipole then threw at
  `translate_batched_cuda.jl:919`). The overloads already existed, committed,
  in the LOCAL FLOWVPM tree (`flowpanel` branch, d07b3c1,
  `ext/FLOWVPMCUDAExt.jl:918-967`).
- **Divergence audit local↔orc** (rsync --checksum + direct diffs): only two
  FLOWVPM files needed syncing (`ext/FLOWVPMCUDAExt.jl` new + sigma_guard
  `:ceil` in `src/FLOWVPM_timeintegration.jl`). FLOWPanel driver/carrier and
  FastMultipole were already content-identical on `unified-052` (one
  FastMultipole diff = comment reflow only). Local FastMultipole dirty files
  are 052d WIP and were deliberately NOT deployed.
- **Pins on orc `unified-052`** (Ryan ran the install script himself —
  `~/install_052b_pin.sh`, can be deleted): FLOWVPM `6c8cda4`
  (ext + :ceil), FLOWPanel `ebc343d` (chain wrapper) + `d002bd5`
  (chain-script /etc/profile fix). Verified present.
- **New chain wrapper** (both local `~/Dropbox/research/projects/FLOWPanel.jl`
  and orc): `examples/p022g_1r_ige_gpu_chain.slurm.sh` — ONE allocation
  running smoke → probe (2 rev) → projection gate (median late-probe
  `step_timer total_step` ×414×1.5 ≤ 6480 s AND probe device-memory reserve
  ≥ 0.20) → 414-step accept. Invokes the carrier
  (`examples/run_rotor_multi_ground_effect_gpu.slurm.sh`) per stage with
  `P022G_MODE` and distinct `P022_RUN_NAME`s (`p022g_1r_ige_smoke/_probe`,
  accept = `p022g_1r_ige`). 4 h wall, 16c/96G, `P022G_REQUIRED_GPU_MODEL=H200`.
  Note: local copy is committed ONLY on orc; the local FLOWPanel working
  tree stays uncommitted per standing policy.

## Live job (VERIFY FIRST — stale by now)

- **13549590 `fp-022g-1r-ige-chain`**: submitted ~12:45 MDT 2026-09-01 to
  `-p eng --qos=eng --gres=gpu:h200:1` at FLOWPanel pin `d002bd5`. Log:
  `~/projects_unified/FLOWPanel.jl/logs/slurm/slurm-fp-022g-1r-ige-chain-13549590.out`.
  m13h was hopeless (est. start Sep 18, 822 pending) — that's why eng.
  NOTE: eng qos can start FAST by preempting standby jobs (the first
  attempt started 37 s after submit) — do not assume a long queue wait.
- **Dead first attempt 13549520** (do not confuse): died in 2 s with
  0-byte logs, exit 1. Cause: the chain script's
  `source /etc/profile 2>/dev/null || true` under `set -u` — an unbound
  variable inside the sourced profile is a FATAL shell error (|| true
  cannot catch it) and stderr was /dev/null'd. Fixed by deleting the
  line (the carrier's `module load` works fine in plain sbatch, proven by
  the 08-31 hr10 GPU jobs). Lesson: never `source /etc/profile` under
  `set -u` in sbatch scripts.
- **Expected timings**: 052 baseline 3.124 s/step (H200) ⇒ accept stage
  ~22–30 min; whole chain ~1–1.5 h. Budget gates: 7200 s case elapsed,
  ≥20% device memory reserve (carrier enforces post-hoc in accept mode;
  chain enforces projection pre-accept).
- **On failure**: stage logs in `logs/chain/p022g_1r_ige_<mode>_13549520.log`
  (orc FLOWPanel repo). The smoke is the first-ever cuarray execution of the
  new overloads in production context — a failure there is most likely in
  the FLOWVPM ext ↔ FastMultipole radix contract; the template/parity
  reference is FastMultipole `test/cuda_radix_lifecycle_test.jl`.
- Also RUNNING (do not disturb): 13548847 `fp-022lg-hr10` CPU production
  (m12, 72 h wall, hr sweep re-anchor), 13518479, 13542905, 13542776.

## Monitor (re-arm; dies at context reset)

Persistent local Monitor, poll ~300 s:
`ssh orc` → `squeue -h -j 13549520 -o %T` + grep the slurm .out for
`=== chain stage|CHAIN |projection:|ERROR:|CANCELLED|TIME LIMIT`.
GOTCHA: orc login banner prepends ANSI codes to the first output line —
never anchor greps; strip with `sed $'s/\x1b\\[[0-9;]*m//g'`; add a
leading `echo`. Also: the auto-mode permission classifier BLOCKS mutating
ssh commands (and edits to settings files); read-only ssh is fine. Ryan
was told he can add `"Bash(ssh orc:*)"` + `"Bash(scp:*)"` to
`FastMultipole/.claude/settings.local.json` `permissions.allow` — check
whether he did; if not, hand him one-liners to run via `! ssh orc ...`.

## mgh bring-up (GH200/ARM) — the main new work

Why it isn't running there already: mgh's 2 idle nodes are ARM GH200s and
the GPU carrier is x86-pinned. Known facts:

- Partition `mgh`, qos `gpu`/`normal`/`test`, MaxTime 1-00:00:00, 2 nodes,
  ~72c each. sbatch REJECTS without `-C arm` (verified via --test-only).
  Gres presumably `gpu:gh200:1`. Prefer node mgh-1-1 (08-31x handoff).
- ARM assets exist but are "deferred/validated only on first mgh run":
  env `~/projects_unified/envs/aarch64` (Manifest dev-points at unified
  trees), depot `~/fm052depot-gh200` (ARM CUDA artifacts — do not delete).
- The P018 launcher already solved arch dispatch — copy its pattern:
  `FLOWPanel.jl/examples/run_p018_screen_gpu052.slurm.sh` (defaults
  P018_REPO/P018_PROJECT to unified, arch-picked env via `$(uname -m)`) and
  the `P018_JULIA` fix (`run_p018_screen_hpc.slurm.sh` line ~180) that
  cured the "Exec format error" from x86 julia on gh200. An ARM julia
  binary/module is needed — find what run_p018_screen_gpu052 uses.
- Carrier blockers to fix (keep x86 path intact; extend, don't fork if
  reasonable): (1) `module load cuda julia/1.11.7-6bmogfl` + hard version
  pin — needs the ARM julia route on aarch64; (2) `PROJECT` default
  `envs/x86_64` → use `envs/$(uname -m)` or `P022G_PROJECT_OVERRIDE`;
  (3) depot: export `JULIA_DEPOT_PATH=~/fm052depot-gh200` on ARM (check
  how P018/smoke_unified did it); (4) `P022G_REQUIRED_GPU_MODEL=GH200`;
  (5) submit flags `-p mgh --qos=gpu --gres=gpu:gh200:1 -C arm`.
- Precompilation on ARM of the new FLOWVPM ext has never happened — expect
  a long first `using` and possibly missing ARM artifacts; a cheap
  setup-only/version-check stage first is wise (cf. smoke job 13543618
  "UNIFIED OK 1.11.7" pattern, `~/projects_unified/smoke_unified_x86.slurm.sh`).
- Validation sequence: carrier smoke (`P022G_MODE=smoke`) for 1r IGE on
  mgh with fallback disabled → probe → then it's a working second pool.
  1r IGE results from eng vs mgh also give a free cross-arch parity check.
- Commit carrier/launcher changes on orc `unified-052` (pin-before-launch
  policy) and mirror to the local tree (leave local uncommitted).

## 2r IGE (after mgh works)

- Case arm `p022g_2r_ige` already in the carrier (NROTORS=2,
  GROUND_ENABLE=true, GROUND_H_R=1.5, TRUNC_RADIUS_R=3.0, policy none,
  damp 0.1... — read the carrier case block; note the Phase-6 config
  differs from the hr-sweep arms). Submitter `scripts/p022g_submit_2r_ige.sh`.
- Reuse the chain pattern: either generalize
  `p022g_1r_ige_gpu_chain.slurm.sh` to take the case as `$1` (preferred)
  or copy it. Same gates; expect roughly ~2× the 1r wake/panel cost —
  budget says 4r IGE (~16×) is the binding case, 2r should fit ≤20 s/step
  easily if 1r comes in ~3 s/step.
- Context: the four CPU multi-rotor arms 13484022-25 FAILED (see
  `FLOWPanel.jl/BRAINSTORM/022_rotor_hover_ground_effect/phase_06_multirotor_gpu.md`,
  most-current log) — review why before assuming the 2r driver path is
  clean on GPU; 052b A.1 verified multi-rotor host correctness
  (warmstart/simulate/solver suites green) but no multi-rotor GPU run has
  ever happened.
- Pool choice for 2r: whichever of eng/mgh is free after step 2 (mgh
  24 h MaxTime is plenty for a chain).
- Watch home storage: `/home/rander39` was 618G vs the 400G cap with
  archiving exhausted and Ryan's approval queue pending
  (HANDOFF_CPU_20260901.md §Pending Ryan decisions). SAVE_VTK=true adds
  more; raise with Ryan if a sweeper approval would unblock space.

## Ryan rulings this session

- **linegauss CONFIRMED** for the p022g arms ("I want linegauss",
  2026-09-01) — the carrier's global `FLOWPANEL_FILAMENT_REG=linegauss`
  stands; do not revert to the gaussian in the older 052b spec text.

## Flagged to Ryan (unanswered at reset)

- Storage cap decisions (above). Notebook entry for the GPU-route unblock
  not yet offered — offer at the next milestone (ask verbosity first).

## Key files

- `MATRIX_OPERATOR_REFACTOR/052b-impl-multirotor-ige-gpu.md` — 052b task doc
  (budget: ≤2 h/case, ≤20 s/step; case matrix)
- `FLOWPanel.jl/plans/p022_hr_sweep_gpu_20260831/HANDOFF.md` +
  `HANDOFF_CPU_20260901.md` — GPU-blocker history, anchors, GS-floor story
- `FLOWPanel.jl/examples/run_rotor_multi_ground_effect_gpu.slurm.sh` — carrier
- `FLOWPanel.jl/examples/p022g_1r_ige_gpu_chain.slurm.sh` — chain wrapper
- `FLOWVPM.jl/ext/FLOWVPMCUDAExt.jl` (:918-967 overloads);
  FastMultipole `src/translate_batched_cuda.jl:919/5693` (contract),
  `test/cuda_radix_lifecycle_test.jl` (template)
- `~/.claude/plans/work-on-052b-per-eager-beacon.md` — this session's plan

## House rules (carried forward)

4 threads max locally; julia-test-runner for runs/scripts (output →
scratchpad log, grep it); refactor-docs-librarian for MATRIX_OPERATOR_REFACTOR
doc questions; verifier before reporting claimed numbers; never read
`data/**`/`*.csv`/`*.bin` raw; notebook writes need Ryan's approval FIRST;
commits/pushes on local trees only on Ryan's ask (orc unified-052 pin
commits before launch are sanctioned); GPU jobs authorized; scp scripts to
orc instead of nested ssh quoting; rsync --checksum; slurm needs `bash -lc`;
auth expiry → ask Ryan `! ssh orc echo ok`; probe partitions with the
slurm-availability skill before submitting.
