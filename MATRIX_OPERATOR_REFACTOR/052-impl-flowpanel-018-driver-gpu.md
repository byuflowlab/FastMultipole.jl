# 052 Impl: FLOWPanel 018 Driver on GPU (end-to-end)

## Status and entry gate

**Staged `2026-08-20` (user direction). IN PROGRESS `2026-08-25`: local GPU-S and multi-architecture harness implementation/verification complete; protected H200 GPU/gate jobs, alternative-architecture probe/smoke/mature results, and the original acceptance verdict remain. The revised matched-continuation plan supersedes the cold settle-window sequence in `052-plan-2026-08-24.md`.**

Entry gate: `049` and `051` complete and approved. This is the phase's
end-to-end deliverable row; its verdict may invoke the phase-prose escape
hatch (pull `054`/`055` levers forward as an addendum) if the <1 h target
is missed.

## Motivation

The phase objective made concrete: run the latest 018 driver on GPU and
measure whether the campaign's stretch target — **30 revolutions in <1 hour**
(1080 steps at ≤3.3 s/step avg, a 52–70× speedup over the 170–230 s/step
CPU baseline) — is achievable, or report the measured feasible wall time
with a breakdown of what binds. Today the 30-rev target walls out in 48 h
of cluster time (~1.7–2.3 h/rev on 64 cores).

## Objective

The latest 018 driver (`examples/rotor_hover_pressure_comparison.jl` via
the slurm case matrix) running end-to-end on GPU+CPU with correctness
verified against a CPU reference arm, and the performance verdict delivered:
30 rev < 1 h, or the measured feasible time + binding-constraint breakdown +
named `054`/`055` levers to pull forward.

## Method

### Stage 1 — GPU driver arm

Add a GPU arm to the driver/case-matrix (env-var knob, matching the
driver's all-knobs-via-env convention;
`examples/run_dji9443_hover_ct_hpc.slurm.sh` case matrix at :355-410,
`p018_L1_ov3` at :405 = OVERLAP 3.0, P_PER_STEP 14, MERGE_R_FACTOR 0.0052).
Resource envelope: 1×GPU + 64 CPU threads. Monitors/paraview output must
stay intact (writer convention `FLOWPanel_wake.jl:2170-2200`).

### Stage 2 — correctness arm

Run a CPU reference arm and the GPU arm over a settle window; correctness
gate = **CT and Γ(r/R) agreement** between arms over that window (tolerance
set from the campaign's own arm-to-arm scatter, recorded in the report).

### Stage 3 — performance run + breakdown

Long GPU run (30 revs if wall time allows; else the longest feasible) with
per-pass instrumentation. Deliverable: measured s/step trajectory vs
particle count, the per-pass breakdown showing what binds (extending the
`049` budget table to the real driver: shed/solve/wake/body/SFS/integrator/
I/O/callbacks), and extrapolated 30-rev wall time.

### Stage 4 — verdict

**30 revolutions in <1 h if achievable, else the measured feasible wall
time.** If <1 h is missed, the verdict names which `054`/`055` levers to
pull forward as an addendum to this phase (escape hatch in the phase
prose) — with the per-pass evidence for why those levers are the right
ones.

## Gates and verdict

- Correctness gate: CT/Γ(r/R) agreement vs the CPU arm over the settle
  window; monitors/paraview output intact.
- Performance deliverable as above, with job IDs and the per-pass
  breakdown.
- Any default-behavior change in the production repos needs explicit user
  approval (phase convention; `053` checkpoint).

## Artifacts

- Driver/launcher changes on the FLOWPanel branch; slurm scripts.
- `data/flowpanel_018_driver_gpu/` — timing CSVs, CT/Γ(r/R) comparison,
  per-pass breakdown, `report.md` with the verdict.

## Verification

- Same-case CPU vs GPU arms from the same commit; job IDs recorded;
  extrapolations shown with the measured per-step-vs-n scaling, not assumed
  constants.

## Recorded context (2026-08-20 staging)

**Driver:** `examples/rotor_hover_pressure_comparison.jl` (1501 lines, all
knobs via env vars); launcher `examples/run_dji9443_hover_ct_hpc.slurm.sh`
(`p018_*` case matrix at :355-410; `p018_L1_ov3` at :405 = OVERLAP 3.0,
P_PER_STEP 14, MERGE_R_FACTOR 0.0052; L1 = σ-ladder rung, ov3 = its overlap
arm). Config: 36,752 panels (45_185_ct4 mesh), NT=36 steps/rev, ~20 revs =
719 steps, THREADS=64, `-t 64`; ~181k particles at maturity (342k on the 6R
arm). Item 018 = `BRAINSTORM/INDEX.md:74`,
`018_dji9443_hover_convergence_campaign.md` (LIVE; Phase 16 chord–σ
co-scaling opened 2026-08-14). FLOWPanel repo:
`/Users/ryan/Dropbox/research/projects/FLOWPanel.jl` (NOT under tmp3).

**CPU baseline (023 profiling):** 170–230 s/step on 64 cores (~1.7–2.3
h/rev; 30-rev target walls out in 48 h); per-step cost ~linear in particle
count (~50–100 s per 100k); split = wake influence 64.2% / body 25.3% /
solve 9.3%; ~75% of a production step = `Estr_fmm!`; wake FMM velocity ~7 s;
body pass floor ~36 s (kerneloffset-radius-bound); 49% thread utilization.
Tuned point (MAC 0.6 / leaf 24) already 3.7× faster + 12× more accurate
than production knobs — the GPU arm should start from the tuned knobs.

**Budget arithmetic:** 30 revs = 1080 steps in <1 h ⇒ ≤3.3 s/step avg
(52–70× vs today); the ~36 s CPU body-pass floor alone breaks it ⇒ both the
particle side (GPU UJ+SFS) and the panel passes must be accelerated; the
"GPU + 64 CPU threads" fallback bounds achievable time.

**Escape-hatch levers on the shelf (041k):** far-field singular switch
(ρ²>42.25 F32 / 81 F64) + `__nv_fast_expf`/`__nv_erff` + 2-target register
blocking = 1.6–1.7× F32 on all-pairs (3.3e11 pairs/s ≈ 38% FP32 FMA peak);
F64 opt inert below n≈3e4.

**FMM step anchors (041a fig15, unitcube GPU best-uniform):** 1.31 ms @1e4,
7.40 @1e5, 92.3 @1e6.

**Paraview convention:** writer `src/FLOWPanel_wake.jl:2170-2200`
(`<path>/<wake>_particles/<wake>_particles.<idx>.vtp` + .pvd; arrays gamma,
sigma, vol, circulation, velocity, vorticity, C, SFS, velocity_gradient).

**Harness gotcha:** run the full step head (`maneuver!` +
reset/freestream/kinematic + `update_TE!`) before any influence eval.

## Worklog — 2026-08-24 revised implementation

Implemented in the FLOWPanel and FLOWVPM sibling worktrees; no commit made.

- Driver: default-off `RHPC_SOLVER_S`, `BLAS_NUM_THREADS_MARCH`, separate
  construction/G/LU/S timing and sizes, explicit S-gemv/backend source-path
  logs, wake-correction bypass warning, and expanded GPU/FMM/restart/timer/BLAS
  case metadata.
- Timers: default-off `FLOWPANEL_STEP_TIMERS` with CUDA synchronization at GPU
  boundaries; ten exclusive top-level categories, total step, reconciling
  residual, and nested wake-SFS accounting.
- Harness: one shared immutable `p018_L1_ov3` production environment; independent
  CPU-S and GPU-S continuations from protected step 719 over steps 720–755;
  exact cold `NREVS=28.5` + `SPINUP_REVS=1.5` = 1,080-step GPU stage; 64 Julia
  threads, construction BLAS 64, marching BLAS 8, 192 GB, H200, eight-hour
  guard; source/checkpoint/environment/resource provenance with SHA-256 lists.
- Comparison: finite common-window enforcement, exact step coverage and particle
  trajectory gates, signed-blade-radius normalization, campaign M2 cycle/blade
  Γ average on `0.3 <= r/R <= 0.95`, reference-global-max normalization,
  matched CT cycle means, locked-scatter TOML gates, exclusive/nested/backend
  timing separation, CSV/Markdown output, and ReadVTK first/last artifact loads.
- Local results: coarse 7,288-panel S setup succeeded (G 8.01 s, LU 1.52 s,
  S 8.31 s in the three-step arm; BLAS 4→2). Three-step backend-vs-S agreement:
  CT cycle-mean relative `2.02e-9`; Γ M2 max `3.80e-8`, RMS `8.52e-9`;
  particle counts identical; all three S steps used `source_influence_s_gemv`.
  Solver tests 393/393, simulate/timer tests green, synthetic comparator 13/13,
  and the full FLOWPanel `test/runtests.jl` suite passed without failures.

Remaining sequence is automated by the single local entry point
`FLOWVPM.jl/scripts/fm052_submit.sh`: it syncs all three working trees, derives
and freezes the same-configuration `p018_L1_s2`/`p018_L1_warm` scatter ceiling
over raw steps 792–1151, submits GPU stages a/b/c and the independent CPU mature
arm, then schedules the mature gate. The gate submits GPU stage d only after it
passes. The measured process wall, not marching-only time, controls the `<3600
s` verdict. If interrupted, segment process walls must be summed.
Closure/artifact fetch and any `054`/`055` choice remain evidence-gated.

## Worklog — 2026-08-24 GPU-resident S memory gates

Implemented the follow-on GPU-S path in the existing uncommitted FLOWPanel and
FLOWVPM task-052 worktrees; no cluster job and no commit were made.

- `Backslash` now owns optional resident-S state: the device matrix, reusable
  input/output vectors, host result buffer, upload/gemv counters and timings,
  allocation bytes, CUDA-pool snapshots, and minimum observed free memory.
  The production Float64 matrix is **10.805676032 GB (10.064 GiB)**; with its
  two vectors the exact allocation gate is computed from the live matrix size.
- `source_potential_gpu=true` is default-off and rejects missing CPU-S,
  non-CUDA influence mode, nonfunctional CUDA, insufficient pre-upload memory,
  and active wake correction. It never falls back. Reassembly refreshes the
  existing device matrix; each gemv transfers only sigma and the potential
  result, synchronizes CUDA, and logs `source_influence_s_gpu_gemv`.
- The upload preflight requires the allocation plus **32 GiB free after
  upload** by default. Marching samples free/pool memory and aborts before an
  allocation failure below the default 4 GiB emergency margin. The mature arm
  samples every step; the long arm samples every ten steps.
- Every task-052 driver GPU arm selects `RHPC_SOLVER_S_GPU=true`; CPU reference
  arms remain on `source_influence_s_gemv`. Smoke gates allocation/reuse via one
  upload, three GPU gemvs, distinct source-path logs, and explicit cleanup.
- Reports now separate CPU-S, GPU-S, and backend paths and write startup usable,
  post-upload free, post-FMM minimum-free, mature-tail minimum-free, and pool
  usage. The step-719 continuation requires one upload, one memory sample and
  GPU-S gemv per step, and **at least 16 GiB mature-tail free** before its gate
  releases the 1,080-step acceptance job.
- Measured mature-run headroom remains pending the H200 continuation. The gate
  records it in `fm052_gpu_memory.csv`, `fm052_run_summary.csv`, and
  `fm052_memory_gate.md`; no acceptance run can be submitted automatically
  until that measured value passes.

Local verification after this change: FLOWPanel solver tests **397/397**;
task-052 synthetic comparator/memory-gate tests **15/15**; Julia parsing,
shell syntax, diff whitespace checks, and the full FLOWPanel regression suite
passed without failures. CUDA numerical/reuse/refresh/
cleanup coverage is present behind `FLOWPANEL_TEST_GPU_S=true` for execution on
the H200 because the local macOS runtime cannot load CUDA.

Submission follow-up: the first chain (`13468167`–`13468169`) exposed a stale
checkpoint-preflight spelling (`_metadata.toml`). FLOWPanel's unified restart
manifest and the protected checkpoint both use `.metadata.toml`; the task-052
preflight now checks that canonical path and emits the exact missing filename or
wake-grid glob on failure instead of exiting silently. Job `13468168` had failed
on this check, cancelling its dependent gate; the pending orphan GPU job was
left for the user to cancel before a clean resubmission.

## Multi-architecture extension — 2026-08-25

### Scope and immutable acceptance policy

Task 052 is extended from the canonical H200 acceptance run to an explicitly
architecture-qualified **x86 GPU** comparison. In-scope slugs are `h200`,
`h100`, `b200`, and `l40s`. GH200/ARM is excluded from this item by user
direction and reserved for successor item `052a`. New job names are
`fp052-<slug>-<stage>`; run names are
`fm052_<slug>_{smoke_cpu_s,smoke_gpu_s,mature_gpu_s}_<jobid>`; reports live below
`data/fm052_multiarch/<slug>/comparisons/<stage>/job-<jobid>`; Slurm logs and
job-ID-qualified submission/result manifests live below the same slug root.
Fetches land in
`MATRIX_OPERATOR_REFACTOR/data/fm052_multiarch/<slug>/results-<jobid>` and refuse
overwrite.

All architectures retain the exact production Float64, P=16/SFS/FMM,
step-719 checkpoint, 36-step window, schedule, CPU reference, correctness
tolerances, one-time GPU-S upload, 32 GiB post-upload reserve, 4 GiB emergency
margin, and 16 GiB mature-tail minimum. The early eligibility bound is
10,806,265,000 + 32 GiB + 4 GiB = **49,460,970,664 bytes (46.064 GiB)**.
An architecture below it is recorded `ineligible`; no reserve-reduced path
exists in the acceptance harness and no alternative-architecture stage can
submit a 1,080-step run.

The completed canonical CPU mature arm is reused only after gates prove equal
checkpoint checksums, equal normalized computational-source fingerprints, and
equal platform-independent package manifest contents. Architecture-specific
environment paths and JLL artifacts may differ, but package UUID/version/tree
identity may not. Correctness, source-path, artifact, and memory gates are then
identical to the canonical mature comparison.

### Read-only scheduler audit (2026-08-25 09:54 MDT)

No job was submitted, cancelled, or modified. `sbatch --test-only` IDs were not
real jobs and `squeue` confirmed they did not appear.

| Slug | Partition/GRES | CPU | Access result | Campaign disposition |
|---|---|---|---|---|
| `h200` | `m13h` / `h200` | x86_64, 96 CPU/node | test-only passed | canonical reference; qualified probe supplies supplemental identity |
| `gh200` | `mgh` / `gh200`, `--constraint=arm` | ARM64 Grace, 72 CPU/node | test-only passed only with `arm` constraint | **OUT OF SCOPE for 052; reserved for 052a; do not prepare or submit here** |
| `h100` | `cs2` / `h100` | x86_64, 112 CPU/node | test-only passed with scheduler-directed preemptible `--qos=standby` (`gstandby` must not be requested directly) | required probe → smoke → mature; no auto-requeue, every retry gets a new job-ID-qualified path |
| `b200` | `cs3` / `b200` | x86_64, 128 CPU/node | test-only passed with `--qos=standby`; estimate was far later than other candidates | optional, preemptible; prepare/run only if selected |
| `l40s` | `m13l` / `l40s` | x86_64, 64 CPU/node | test-only passed | compatibility probe, then expected official low-memory/ineligible record; gates are not weakened |

The protected chain was unchanged at the audit: GPU `13468358` `fp052gpu`
PENDING (Priority, requested partitions reported `m13l,m13h` but GRES H200), CPU
`13468359` COMPLETED in 38m13s exit 0, and gate `13468360` PENDING on the GPU.
Nothing in `~/FLOWVPM-046`, `~/FastMultipole-046`, `~/FLOWPanel-052`, or the
canonical environment was changed. New work uses separate `FLOWVPM-052-<slug>`,
`FastMultipole-052-<slug>`, and `FLOWPanel-052-<slug>` trees plus
`fm052env-<slug>` environments, so preparing one architecture cannot change
sources visible to a pending job for another.

### Implemented harness and run plan

1. `fm052_arch_prepare.sh` syncs the current local dirty worktrees into isolated
   remote trees and clones the canonical package lock into a per-slug environment;
   it never calls `sbatch`. Under task 052 it is used only for the in-scope x86
   slugs. Its GH200 branch is retained as unexecuted scaffolding for `052a` and
   must not be invoked during 052.
2. `fm052_arch_run.sh` requires a canonical slug, exact slug/GRES/partition
   agreement, exact architecture-qualified Slurm job name, one visible GPU whose
   observed `nvidia-smi` model matches the slug, and the expected CPU ISA.
3. `probe` records GPU model/VRAM/UUID/CC/driver, node/partition, CPU ISA/model,
   Julia/CUDA, BLAS vendor/threads/config, package paths, all manifest JLL load
   results, source/checkpoint hashes, and official memory eligibility.
4. `smoke` is manually submitted only after a passing probe whose exact job ID
   is supplied. It performs one
   upload, exactly three GPU-S gemvs, buffer reuse, cleanup, CPU-S parity, and a
   qualified report.
5. `mature` is manually submitted only after passing probe/smoke job IDs are
   supplied. It gates canonical
   CPU provenance first, then runs the unchanged 36-step continuation and all
   correctness/artifact/source/memory gates. It does not enqueue anything.
6. `cross-arch` consumes result manifests, retains all raw metrics, reports
   access/low-memory failures, and normalizes timing metrics to H200. It includes
   process/setup/assembly/upload/mature-tail/GPU-S/per-pass timings, memory/pool,
   particles, and CT/Γ differences.

The live H200 chain predates the naming/provenance extension. After it and its
gate complete, `fm052_arch_adopt_h200.sh` creates a qualified reference manifest
and isolated architecture-qualified run/report symlinks without touching
canonical artifacts. Its legacy job-name and any identity
fields absent from the mature log are explicitly marked; a new qualified H200
probe can supplement architecture identity but is not represented as the
mature job's device UUID.

Actual CUDA execution and actual device matching remain cluster-gated. Task 052
stays open until the selected **x86** architecture runs and the original H200
acceptance verdict are resolved. Standby preemption
does not overwrite evidence: every run, report, and result manifest includes the
new allocation's job ID, and failed attempts remain independently fetchable.

GH200 split-out (2026-08-25): actual GH200 execution — the offline ARM
environment build, probe → smoke → mature → acceptance runs, and the
fused-memory (unified-memory) effectiveness assessment — is staged as its own
sub-item, `052a-impl-gh200-arm-unified-memory.md`. This doc's harness,
naming, provenance, and eligibility policy remain the authority; 052a
executes under it without modification.

h100 probe failure + fix (2026-08-25): job 13476489 `fp052-h100-probe`
FAILED at `fm052_arch_probe.jl:36` on two latent bugs in the
never-previously-executed probe, found and fixed under 052a: (1) line 30
must build the PkgId from the `Pkg.dependencies()` pair KEY
(`Base.PkgId(uuid, name)`) — `PackageInfo` has no `uuid` field; (2) the
line-41 BLAS expectation must be `[90.0, 100.0, 110.0, 120.0]`
(column-major `reshape(1:16,4,4)`), not the transposed-matrix values. The
local `FLOWVPM.jl/scripts/fm052_arch_probe.jl` is fixed and verified on the
GH200 (probe pass, job 13477052), but the cluster copy in
`~/FLOWVPM-052-h100/scripts/` is stale — plain `rsync -az` skips it
(same size + mtime), which is how the bug survived; prepare now uses
`--checksum` for the scripts sync. **To fix h100:** re-run
`bash scripts/fm052_arch_prepare.sh h100` (or minimally
`rsync -rlc scripts/fm052_arch_probe.jl orc:FLOWVPM-052-h100/scripts/`),
then resubmit the h100 probe; the failed attempt's manifest remains under
its own job-ID path per policy. The same re-sync applies to b200/l40s
whenever those slugs are prepared.

GH200 scope split (user, 2026-08-25): **do not attempt GH200 in task 052**.
All ARM-native environment, CUDA/JLL/BLAS compatibility, smoke, mature, and
comparison work moves to future item `052a`. The already supplied Julia path and
read-only binary observation are retained solely as `052a` handoff information:
`/home/rander39/julia/julia-1.11.7/bin/julia`, ARM aarch64 ELF, launcher SHA-256
`d5e4bff53012b807b213a3eb4352963d3c0f7361062b0f7885a899cc598e876d`.

Local validation on 2026-08-25: all modified Julia files across the dirty
FLOWPanel/FLOWVPM task worktrees parse; every task-052 shell script passes
`bash -n`; shell fixtures pass canonical slug, GRES/partition, observed-device,
CPU-ISA, and low-memory/ineligible gates; the extended synthetic comparator,
manifest, memory, and cross-architecture report suite passes **20/20**; and
CPU-only FLOWPanel solver **397/397** plus all simulation unit testsets pass with
`JULIA_NUM_THREADS=4` and `FLOWPANEL_TEST_GPU_S=false`. The first solver attempt
hit a sandbox write denial at a pre-existing repository VTK path after 396
assertions; rerunning unchanged from an isolated temporary working directory
passed 397/397, confirming an output-path collision rather than a code failure.

## Context-reset checkpoint — 2026-08-25

Paused by user direction before any multi-architecture Slurm submission.

Completed cluster setup, using architecture-isolated paths throughout:

| Slug | Source trees | Environment | State |
|---|---|---|---|
| `h100` | `~/FLOWPanel-052-h100`, `~/FLOWVPM-052-h100`, `~/FastMultipole-052-h100` | `~/fm052env-h100` | prepared and instantiated; next priority is the preemptible compatibility probe |
| `b200` | `~/FLOWPanel-052-b200`, `~/FLOWVPM-052-b200`, `~/FastMultipole-052-b200` | `~/fm052env-b200` | prepared and instantiated; optional after H100/H200 priorities |
| `l40s` | `~/FLOWPanel-052-l40s`, `~/FLOWVPM-052-l40s`, `~/FastMultipole-052-l40s` | `~/fm052env-l40s` | prepared and instantiated; probe should record expected official low-memory rejection |
| `h200` | `~/FLOWPanel-052-h200`, `~/FLOWVPM-052-h200`, `~/FastMultipole-052-h200` | `~/fm052env-h200` | prepared and instantiated for a later qualified supplemental probe/reference adoption; protected live chain untouched |
| `gh200` | none prepared in this extension | none prepared | **not part of 052; reserved for 052a** |

Each x86 environment was cloned from the canonical task-052 lock, path-developed
against only its matching architecture source triplet, resolved, and instantiated.
The preparation script reported `no Slurm job was submitted` for all four. No
files were synced into `~/FLOWVPM-046`, `~/FastMultipole-046`, or
`~/FLOWPanel-052`, and no live job was cancelled or modified.

The scheduler audit established H100/B200 access through preemptible
`--qos=standby` (`gstandby` must not be requested directly). Because the
canonical H200 GPU remains delayed by Priority, the immediate next action is for
the user to submit the H100 compatibility probe from `~/FLOWVPM-052-h100`.
The exact command is in the reset prompt/handoff. Inspect its job-ID-qualified
probe result before manually submitting smoke; do not chain or auto-submit.

The GH200 path is intentionally not ready despite the supplied Julia binary.
Do not run `fm052_arch_prepare.sh gh200`, submit a GH200 probe, or include GH200
in a task-052 cross-architecture report. All such work belongs exclusively to
future item `052a`.

## Resume worklog — 2026-08-25

Read-only scheduler refresh at 10:45 MDT confirmed that the protected H200
chain remains unchanged: `13468358` (`fp052gpu`) is pending on priority,
`13468359` (`fp052cpu`) completed in 38m13s with exit 0, and `13468360`
(`fp052gate`) is pending on its dependency. No job was submitted, cancelled,
requeued, or otherwise modified.

Added one self-contained, Ryan-only submission wrapper for every currently
allowed x86 multi-architecture allocation. Each wrapper prints the
`sbatch --parsable` job ID, uses an architecture-qualified job and output path,
requests one GPU plus 64 CPU threads and 192 GB, and sets `--no-requeue`.
H100 and B200 explicitly use the scheduler-supported preemptible
`--qos=standby`; no wrapper requests `gstandby`. Smoke and mature wrappers
require numeric inspected predecessor job IDs. No GH200 wrapper and no
alternative-architecture 1,080-step wrapper exist.

- H100: `MATRIX_OPERATOR_REFACTOR/scripts/fm052_h100_probe_submit.sh`,
  `MATRIX_OPERATOR_REFACTOR/scripts/fm052_h100_smoke_submit.sh`, and
  `MATRIX_OPERATOR_REFACTOR/scripts/fm052_h100_mature_submit.sh`.
- Optional B200: `MATRIX_OPERATOR_REFACTOR/scripts/fm052_b200_probe_submit.sh`,
  `MATRIX_OPERATOR_REFACTOR/scripts/fm052_b200_smoke_submit.sh`, and
  `MATRIX_OPERATOR_REFACTOR/scripts/fm052_b200_mature_submit.sh`.
- Probe only: `MATRIX_OPERATOR_REFACTOR/scripts/fm052_l40s_probe_submit.sh`
  and `MATRIX_OPERATOR_REFACTOR/scripts/fm052_h200_probe_submit.sh`.

These local submission wrappers intentionally call the already prepared remote
`scripts/fm052_arch_run.sh`; they do not sync or modify any cluster source tree.
The immediate user command is:

```bash
bash MATRIX_OPERATOR_REFACTOR/scripts/fm052_h100_probe_submit.sh
```

After it returns a job ID, inspect the corresponding job-ID-qualified result
manifest and Slurm log before using the H100 smoke wrapper. The launch wrappers
passed `bash -n` and non-submitting fake-SSH expansion checks for their resource,
path, stage, and predecessor-ID exports; ShellCheck was not installed locally.

## Context-reset checkpoint — 2026-08-25 (H100 launch handoff)

Paused for a context reset immediately before Ryan's H100 compatibility-probe
submission. Ryan, not the agent, will run:

```bash
bash MATRIX_OPERATOR_REFACTOR/scripts/fm052_h100_probe_submit.sh
```

The returned job ID was not available when this checkpoint was written and must
be obtained from Ryan before job-qualified inspection. Do not infer it from a
queue listing. Once supplied, monitor read-only no more frequently than the ORC
60-second minimum and inspect both:

```text
~/FLOWPanel-052-h100/data/fm052_multiarch/h100/manifests/fm052_h100_probe_<jobid>_result.toml
~/FLOWPanel-052-h100/data/fm052_multiarch/h100/slurm/fp052-h100-probe-<jobid>.out
```

Verify the result status plus observed H100 model, VRAM, UUID, compute
capability, x86_64 CPU, Julia/CUDA/package/JLL/BLAS loading, isolated FLOWPanel/
FLOWVPM/FastMultipole source paths, source/checkpoint provenance, and the
unchanged official memory-eligibility threshold. Only after the result is
`pass` and the evidence has been inspected should Ryan be given:

```bash
bash MATRIX_OPERATOR_REFACTOR/scripts/fm052_h100_smoke_submit.sh <probe-jobid>
```

Do not run that wrapper as the agent. After a passing inspected smoke, give Ryan
the mature-wrapper command with both qualified predecessor IDs; do not submit it.
Keep every preempted/failed attempt distinct. Never weaken memory gates, never
launch an alternative-architecture 1,080-step run, and do not begin GH200.

The last scheduler observation remains 10:45 MDT: canonical H200 GPU `13468358`
pending on priority, CPU `13468359` completed in 38m13s with exit 0, and gate
`13468360` pending on dependency. The next agent must refresh those states
read-only before acting and must not modify anything sourced by that chain.

Post-checkpoint launcher correction: Ryan's first local H100-wrapper attempt
reached ORC but failed before submission with `sbatch: command not found` because
plain non-login SSH did not initialize the Slurm command path. All eight local
submission wrappers now invoke their remote command through `bash -lc`. Their
shell syntax and fake-SSH argument expansion pass after the correction. This
failed attempt created no Slurm job and therefore has no job ID or result path.

## H100 probe attempt 13476489 — pending observation 2026-08-25

Ryan supplied the exact job ID returned by the corrected wrapper: `13476489`.
A read-only scheduler snapshot at 11:02:07 MDT found
`fp052-h100-probe` pending on `Priority`, requesting one H100, 64 CPUs,
192 GiB, and one `cs2` node. The scheduler's estimated start was 14:06:31 MDT;
that is an estimate only, not evidence of allocation or completion.

At that snapshot, `cs-2-1` had 43/112 CPUs and 8/8 H100s allocated, while
`cs-2-2` had 30/112 CPUs and 8/8 H100s allocated. Thus 151 CPUs were idle
across the two nodes but no H100 was idle; neither node also had the requested
192 GiB of scheduler memory unallocated. `scontrol` reported the job's
effective QOS as `gstandby`; the inspected local wrapper directly requests only
`--qos=standby`, so this is scheduler-directed remapping rather than a direct
`gstandby` request.

The protected canonical H200 chain was refreshed read-only at 10:56:12 MDT:
`13468358` remained pending on `Priority`, `13468359` remained completed in
38m13s with exit `0:0`, and `13468360` remained pending on its unfulfilled
`afterok:13468358` dependency. No queue state or cluster file was modified.

Because H100 attempt `13476489` is still pending, there is no completed probe
result to inspect and the smoke command remains withheld. Its eventual result
and log must be read only from the job-ID-qualified paths recorded in the
launch handoff above.

### Terminal result and harness correction

A read-only refresh at 11:29:33 MDT found attempt `13476489` terminal `FAILED`,
not cancelled: it ran on `cs-2-1` from 11:22:30 through 11:26:42 MDT (4m12s)
and exited `1:0`. Both qualified evidence files exist. The result manifest
records `status = "fail"`, `reason = "exit_1_line_1"`, and the following
successfully observed compatibility facts:

- NVIDIA H100 80GB HBM3, 81,559 MiB VRAM, UUID
  `GPU-6edeebca-3371-cf19-14b9-72c6b12b0426`, compute capability 9.0;
- NVIDIA driver 580.159.04 and CUDA runtime 12.8.0;
- x86_64 Intel Xeon Platinum 8480+ and Julia 1.11.7;
- successful FastMultipole, FLOWPanel/FLOWVPM, LinearOperatorsCUDAExt, and
  FLOWVPMCUDAExt precompilation;
- source checksum `e3b4255f63e2a8fbfbf57e85395bc542c4e3d0378ac0cda3509c77aa90ce15e2`
  and package-manifest checksum
  `bce874a628876859d64c8590ac1e092c32bb797d000d7e8705a72374ddf01f58`.

The failure was a probe-harness API bug, not evidence that the JLL binaries are
incompatible. `Pkg.dependencies()` returns `UUID => PackageInfo`, but
`fm052_arch_probe.jl` attempted `package.uuid`; Julia 1.11's `PackageInfo` has
no such field, so every enumerated JLL reported the same introspection error
`type PackageInfo has no field uuid`. The probe now retains the UUID dictionary
key and passes it to `Base.PkgId`. A regression assertion rejects any return of
`package.uuid`; Julia parsing and the architecture shell fixture pass.

After `13476489` was terminal and with no smoke released, only the corrected
probe script and regression test were synced into the isolated
`~/FLOWVPM-052-h100` tree. Local/remote SHA-256 values match:
`4b917d6e367da6c5ab443c0b80bc70c027a45bf8ba65bde7288b11e21f31b6b2`
for the probe and
`596c82b06f461f3c6aa6bfa93ffc98c18afb39db90936f4d0f09f075f2bc915b`
for the test. No protected canonical tree was touched.

Attempt `13476489` did not reach verified JLL loading, BLAS configuration,
isolated resolved source paths, detailed checkpoint provenance, or the official
memory-eligibility gate. It is therefore a retained inspected failure and does
not authorize smoke. The corrected probe requires a new Ryan-submitted job ID;
the agent must not submit it or reuse `13476489`.

The same 11:29:33 MDT read-only refresh found the protected H200 chain unchanged:
`13468358` pending on `Priority`, `13468359` completed in 38m13s with exit
`0:0`, and `13468360` pending on its dependency. No queue state was modified.

## 2026-08-25 afternoon session — H200 failure root-caused; combined single-allocation campaign submitted (user-authorized)

Queue refresh at 16:00 MDT invalidated the morning's status: protected H200 GPU
`13468358` (`fp052gpu`, qos `gpu`) started at 13:32 after ~19 h pending and
**FAILED** in 6m33s; gate `13468360` was **CANCELLED** on the unfulfilled
`afterok`. Root cause, confirmed from `fp052gpu-13468358.out`:
`UndefVarError: MemoryInfo not defined in CUDA` in `_source_gpu_memory` at
`~/FLOWPanel-052/src/FLOWPanel_solver.jl:424` during stage b's first
`Backslash` construction. The cluster copy was stale — it called
`CUDAmod.MemoryInfo` directly, while the local working tree already guards with
`isdefined(CUDAmod, :MemoryInfo) ? … : CUDAmod.CUDACore.MemoryInfo`
(`FLOWPanel_solver.jl:423-435`). Verified in `~/fm052env_cuda63_geoiofree`:
the umbrella `CUDA.MemoryInfo` is absent but `CUDA.CUDACore.MemoryInfo` exists,
so the guarded version works. Another same-size stale-rsync casualty; all syncs
below used `rsync --checksum`.

Fix + sync: local `FLOWPanel.jl/src/` and the two driver examples were synced
to all five cluster trees (`~/FLOWPanel-052` and `-h100/-b200/-l40s/-h200`);
stale on every one of them: `FLOWPanel_solver.jl`, `FLOWPanel_gpu_wake.jl`,
`FLOWPanel_liftingbody.jl`, `rotor_hover_pressure_comparison.jl`. The corrected
`fm052_arch_probe.jl` (UUID-key fix) and updated `fm052_arch_run.sh` went to
all four `~/FLOWVPM-052-*/scripts/` dirs. Spot-grep confirms the guard on all
five solver copies.

User directives this session: (1) H200 jobs take `--qos=eng` for the shorter
line — `m13h` rejects qos `eng`, but partition `eng` hosts H200 nodes
(`--test-only` placed the chain on `eng-1-1`), so H200 eng-line jobs run on
partition `eng`; `fm052_arch_common.sh` now accepts `{m13h, eng}` for the h200
slug. (2) Combine stages into one allocation wherever possible so the queue
wait is paid once. (3) Keep time limits tight — smaller limits backfill sooner
(H200 chain 8 h, arch chains 6 h, probes 2 h).

New harness pieces (in `FLOWVPM.jl/scripts/`, synced): `fm052_chain_run.sh` —
protected H200 stages `a b c`, then `fm052_gate.sh` inline, then stage `d`,
one 8 h allocation (replaces gate_run.sh's dependent re-queue of `d`);
`fm052_arch_chain.sh` — probe→smoke→mature in one job via `FP052_CHAIN=1`
(same manifest gating per stage, shared job ID; an `ineligible` probe ends the
chain cleanly with exit 0). Submit wrappers added in
`MATRIX_OPERATOR_REFACTOR/scripts/`: `fm052_{h200,h100,b200}_chain_submit.sh`;
`fm052_h200_probe_submit.sh` switched to `--partition=eng --qos=eng`.

Submissions (explicitly user-authorized in-session, superseding the earlier
Ryan-only wrapper restriction): first wave `13477834/13477835/13477837` was
cancelled minutes later to tighten time limits per the user. Live campaign,
submitted 16:10 MDT 2026-08-25:

| Job | What | Partition/QOS | Limit |
|---|---|---|---|
| `13477842` | protected H200 chain: a b c + inline gate + d | eng/eng | 8 h |
| `13477844` | H100 chain: probe→smoke→mature | cs2/gstandby | 6 h |
| `13477845` | B200 chain: probe→smoke→mature (optional) | cs3/gstandby | 6 h |
| `13477838` | L40S probe (expected official ineligibility) | m13l/gpu | 2 h |
| `13477847` | H200 supplemental qualified probe | eng/eng | 2 h |

Optional arms (`13477845`, `13477838`, `13477847`) are on a 1-day queue
budget: cancel any still PENDING after ~16:08 MDT 2026-08-26. The CPU mature
reference (`13468359`, `~/FLOWPanel-052/data/fm052r_cpu_mature`) is intact and
is not rerun. Remaining after the campaign: verdict report at
`data/flowpanel_018_driver_gpu/report.md`, doc closeout, clear-context audit.

Addendum 16:19 MDT: per the user, time limits were tightened again to improve
backfill odds (evidence: stages a+b reached mid-stage-b in 6m33s in the failed
job; the CPU stage-c equivalent took 38m; stage d at the 051-approved
3.124 s/step budget is ~56 min). The 16:10 wave was cancelled and resubmitted:
H200 chain `13477875` (4 h), H100 chain `13477876` (3 h), B200 chain
`13477878` (3 h), L40S probe `13477880` (1 h), H200 supplemental probe
`13477881` (1 h). If stage d cannot finish inside the chain's remaining ~2.5 h
the <1 h verdict is a decisive miss and the partial per-step trajectory still
supports the report. The optional-arm 1-day cancel budget now runs from
16:19 MDT 2026-08-25.

Addendum (later, 2026-08-25): user rescinded the optional-arm 1-day cancel
budget — all five queued jobs run to allocation regardless of queue wait; no
scancel at the 24 h mark.

## Context-reset checkpoint — 2026-08-25 ~16:30 MDT (campaign queued, next agent finishes out)

State: all submittable 052 work is done and in queue. No code work remains;
what remains is inspection, the verdict report, and closeout.

**Live jobs** (submitted 16:19 MDT 2026-08-25; user directive: let ALL run to
allocation, no scancel at any age):

| Job | What | Part/QOS | Limit | Workdir |
|---|---|---|---|---|
| 13477875 | protected H200 chain: stages a b c + inline mature gate + stage d (1080 steps) | eng/eng | 4 h | `~/FLOWVPM-046`, log `fp052chain-13477875.out` |
| 13477876 | H100 chain probe→smoke→mature | cs2/gstandby | 3 h | `~/FLOWVPM-052-h100`, log under `~/FLOWPanel-052-h100/data/fm052_multiarch/h100/slurm/` |
| 13477878 | B200 chain probe→smoke→mature (optional arm) | cs3/gstandby | 3 h | analogous `-b200` paths |
| 13477880 | L40S probe (expected official low-memory ineligibility; non-gating) | m13l/gpu | 1 h | analogous `-l40s` paths |
| 13477881 | H200 supplemental qualified probe (for reference adoption) | eng/eng | 1 h | analogous `-h200` paths |

Poll: `ssh orc 'bash -lc "source /etc/profile; sacct -j 13477875,13477876,13477878,13477880,13477881 -X -o JobID,JobName%22,State,Elapsed,ExitCode"'`.
ssh quirks: login shell required for slurm; noisy banners; keyboard-interactive
auth expires — only the user can re-auth (`! ssh orc echo ok`). All syncs to
cluster MUST use `rsync --checksum`.

**This session's key events** (details in the 2026-08-25 afternoon addenda
above): protected H200 GPU 13468358 FAILED on stale cluster source
(`CUDA.MemoryInfo`); the guarded fix was verified in-env and synced to all five
FLOWPanel trees; corrected UUID-key probe synced to all four arch dirs; user
directives applied — qos=eng for H200 (runs on partition `eng`; `m13h`/`m13l`
reject qos eng; `fm052_arch_common.sh` accepts `{m13h, eng}` for h200), stages
combined into single allocations (`fm052_chain_run.sh`, `fm052_arch_chain.sh`
with `FP052_CHAIN=1`), tight time limits. First two submission waves
(13477834/835/837/838/842/844/845/847) were cancelled for limit tightening —
ignore them in sacct output. CPU mature reference 13468359
(`~/FLOWPanel-052/data/fm052r_cpu_mature`) is COMPLETED and reused; never rerun
it.

**Next agent's job, per outcome:**
1. When 13477875 completes: check the log passes the old failure point
   (`source_s_gpu_memory before_upload` lines appear), stages a b c exit 0,
   inline gate passed (`fm052 mature-continuation gates passed`), stage d ran
   1080 steps (`fm052d_gpu_1080`; per-step timers in the log; process wall in
   `data/fm052d_gpu_1080/fm052d_gpu_1080_process_wall_s.txt`). TIMEOUT during
   stage d = decisive <1 h miss; use partial per-step trajectory.
2. Arch chains: stage manifests at
   `~/FLOWPanel-052-<arch>/data/fm052_multiarch/<arch>/manifests/fm052_<arch>_<stage>_<jobid>_result.toml`
   (`status = "pass"` / `"ineligible"` / `"fail"`). L40S ineligible is the
   expected recorded outcome, not a failure.
3. Any FAILED job: root-cause from its .out before any resubmission (no blind
   retries; delegate log digestion to subagents, keep raw logs out of context).
4. Write the verdict report `MATRIX_OPERATOR_REFACTOR/data/flowpanel_018_driver_gpu/report.md`:
   measured (or extrapolated) 30-rev wall time vs the <1 h target, per-pass
   breakdown (shed/solve/wake/body/SFS/integrator/I-O/monitors),
   cross-architecture table, gate outcomes; if <1 h missed, name `054`/`055`
   levers (pass-2 U-only kernel work, device-resident gemv, Float32 S,
   far-field singular switch). Verify numbers against job logs (verifier
   agent) before reporting.
5. Closeout: mark this doc finished, update the START_HERE 052 row, then stop
   for the user's clear-context audit (phase gate before 053). Offer (don't
   write) a lab-notebook entry.

**Constraints still in force:** never weaken the 46.064 GiB memory threshold;
no alternative-architecture 1080-step run if the H200 gate fails; no GH200/ARM
work here (052a); locked tolerances at
`~/FLOWPanel-052/data/fm052_campaign_lock/fm052_locked_tolerances.toml`.
