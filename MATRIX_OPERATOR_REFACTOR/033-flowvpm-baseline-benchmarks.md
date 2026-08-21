# 033 FLOWVPM Baseline Benchmarks at the gpu-full Branch Point

## Status and Entry Gate

**Added by user request on `2026-08-04`. In progress.** The active cluster
campaign is recorded below; the row is not Done or Approved yet.

Entry is unblocked: the only dependency (`019a`) is Done and clear-context
approved. This row may run in parallel with `031`/`032`; it does not block on
them and does not block on `030`.

This is a **benchmark and analysis row**. It makes no production changes in
either repository. Benchmark scripts go in `MATRIX_OPERATOR_REFACTOR/scripts/`,
data in `MATRIX_OPERATOR_REFACTOR/data/`.

## Objective

Establish the CPU performance and accuracy baselines that every later
Integration Phase speedup claim is measured against, with profiles showing
what costs what and where the bottlenecks lie.

Deliverables:

1. **Baseline timings** of FLOWVPM **at the `gpu-full` branch-point commit
   `e2bd487`** (2026-05-16, v4.0.4 — check out that commit, or an equivalent
   detached worktree, for all baseline runs), single-thread and 64-thread,
   for both test cases, at a stated grid of particle counts including at
   least one representative large `n` per case. Report per-time-step cost of
   the FMM/UJ solve and of the full step.
2. **Profiled cost breakdown**: per-stage attribution (tree build, FMM
   passes, nearfield/direct, SFS, time integration, relaxation, viscous) for
   each case at the representative `n`, single-thread and 64-thread, with an
   explicit bottleneck identification.
3. **Accuracy reference methodology**: sampled-direct references (Float64,
   `UJ_direct`-equivalent kernel) with fixed seeds/checksums so later rows
   reproduce the error measurement exactly. **Amended `2026-08-05`**: the
   Integration Phase accuracy gate is a **fixed tolerance — sampled relative
   gradient (velocity) RMS error ≤ 1e-3** — not the accuracy delivered by
   FLOWVPM's default FMM settings (that check is dropped; the original
   `2026-08-04` decision is superseded). The references generated here are
   the measurement instrument for that tolerance. Baseline rows log both U
   and J errors. Historical timings remain in the report regardless of error,
   but later speedup ratios may use only baseline configurations whose U error
   passes `≤1e-3`; failed baselines are excluded from speedup headlines and no
   replacement tuned CPU campaign is required.
4. **Case definitions**, recorded precisely enough to regenerate bit-for-bit:
   - *Unit cube*: uniformly random particle positions in the unit cube at a
     fixed seed, random strengths (state the distribution), smoothing radius
     `σ` chosen so the average overlap factor is 2 (state the overlap
     definition used, e.g. `σ / mean interparticle spacing = 2` with the
     spacing convention written out).
   - *Helical wake cylinder* (user decision `2026-08-05`, replacing the vortex
     ring): a solid cylinder of length 5 diameters filled with particles whose
     strengths follow a helix (state the pitch, the strength distribution, the
     overlap convention, and the seed).

## Dependencies

- `019a-milestone-review-final-roadmap.md`, complete and clear-context
  approved.

## Mandatory Reading Gate

1. `START_HERE.md`, including the Integration Phase preamble.
2. `../FLOWVPM.jl/CLAUDE.md` in full — required before running or modifying
   anything in that repository.
3. `024b-impl-cpu-gpu-scaling-benchmark.md` methodology sections — reuse its
   sampled-direct error and checksum conventions where applicable.

## Task-Local Requirements

- All baseline runs execute on cluster CPU nodes (a 64-thread node for the
  multithreaded series), never the local machine.
- Single-thread and 64-thread runs must use identical case definitions,
  seeds, and step counts; report medians over repeated steps with warmup
  excluded and state the repeat policy.
- No FLOWVPM commits: baseline scripts live in this repository's
  `MATRIX_OPERATOR_REFACTOR/scripts/` and drive FLOWVPM externally. If a tiny
  instrumentation shim inside FLOWVPM proves unavoidable, it must not be
  committed to `gpu-full` as part of this row — record it as a patch artifact
  here.
- The deliverable report is a cost-breakdown table plus accuracy reference
  values in `data/`, summarized in this file.

## Work Record (2026-08-04)

User decisions: full `024b` grid `n ∈ {1000, 3162, 10000, 31623, 100000,
316228, 1000000}` per case; representative (accuracy-gate) n = **1e5**; FMM
parameters fixed at **p=4, ncrit=50, theta=0.4** with all autotuning off
(`autotune_p=false, autotune_ncrit=false, autotune_reg_error=false`); other
`FMM` fields at struct defaults.

Finalized case definitions:

- **cube**: n particles uniform in the unit cube, `MersenneTwister(33025+n)`;
  Γ components uniform in `[-1,1]/n`; `σ = 2·(1/n)^(1/3)` (overlap 2 with
  mean spacing `(V/n)^(1/3)`, V=1). Solver: rVPM, `gaussianerf`, inviscid,
  no SFS, default relaxation, transposed, RK3.
- **wake** (replaced the ring on `2026-08-05`; see the amendment below): a
  helical wake cylinder. Solid cylinder of diameter `D=1`
  (`R=0.5`) and length `5D=5` about the `z` axis, centred at the origin, with
  `n` positions exactly uniform in its volume at
  `MersenneTwister(33025 + 7919 + n)` (`r = R√u₁`, `θ = 2πu₂`,
  `z = 5(u₃ − ½)`). Strength direction is the local tangent of a helix of
  pitch `p = D`, `t ∝ (−r sinθ, r cosθ, p/2π)`; magnitude is tip-weighted,
  `|Γ| = (r/R)/n`. `σ = 2·(V_cyl/n)^(1/3)` with `V_cyl = πR²·5D = 3.927` —
  overlap 2 against the local mean spacing, the **same convention as the
  cube**, so both cases share one unambiguous `β`. Solver settings are
  identical to the cube (rVPM, `gaussianerf`, inviscid, no SFS, default
  relaxation, transposed, RK3), so the two cases differ only in geometry and
  strength coherence. `n_actual = n_target` exactly at every grid point.

Harness (`scripts/`): `benchmark_033_common.jl` (builders, reference IO,
metrics), `prepare_033_references.jl` (sampled-direct references: all targets
for n≤1e4, else 512 via `MersenneTwister(33026+n_actual)`, Float64
`fmm.direct!(target, source; hessian=true)` so both U and J are referenced),
`benchmark_033_cpu.jl` (per-case driver: warmup + timed `UJ_fmm` medians,
`_reset_particles` timing, full `nextstep` RK3 medians at dt=1e-6, sampled U/J
error logged on every row, `Profile` capture at n=1e5), and the
`cpu_033_{submit,run,fetch}.sh` campaign triple (remote `~/FastMultipole-033`,
env `~/fm033env`, FLOWVPM baseline worktree at `e2bd487` rsynced as
`FLOWVPM_baseline/`, FastMultipole pinned `2.0.0 - 2.0.4` via Pkg compat).

Findings so far:

- **FastMultipole 2.0.4 bug**: `direct_multithread!` on the
  `direct!(target, source)` path throws `UndefVarError: n_source_bodies`
  (`direct.jl:111`) for threads>1. Reference generation therefore runs
  single-threaded (correct and cheap); noted for the record — the `fmm!`
  nearfield path is unaffected.
- Local smoke test (Mac, 4 threads, n=1000, FastMultipole 2.0.4, Julia
  1.12.5): both cases build and run end-to-end; FMM-vs-direct error ~1e-15 at
  this size (essentially all-direct with ncrit=50); `t_step ≈ 3·t_uj +
  integration overhead` as expected for RK3.

## Campaign Status (context-handoff note, updated 2026-08-05)

**Active HPC job: `13051713` (`fm033cpu`)** on the BYU cluster (ssh alias
`orc`, requires a live ControlMaster session; cluster commands need a login
shell: `ssh orc "bash -lc '...'"`).

- **Superseded by the wake amendment below (`2026-08-05`): this job was
  launched against the retired cube+ring case set. Its ring half is now waste;
  its cube half remains valid. Cancel or let it finish, keep the cube rows and
  cube references, discard everything ring, and resubmit for cube+wake with the
  updated harness.**
- What it runs: this row's full baseline campaign — cube + ring (both
  gaussianerf) × the 7-point n grid × `julia -t {1, 64}` on a 64-core EPYC
  node, per `scripts/cpu_033_run.sh`. Remote tree `~/FastMultipole-033`,
  env `~/fm033env` (FLOWVPM v4.0.4 dev'd from the `e2bd487` worktree copy,
  FastMultipole pinned 2.0.4). Wall limit 3 days; expected to finish well
  inside it (the long tail is the four n=1e6 cases, single-thread last).
- Check: `ssh orc "bash -lc 'squeue -j 13051713'"`; progress:
  `ssh orc "tail FastMultipole-033/fm033cpu-13051713.out"`.
- When it completes: fetch with
  `bash MATRIX_OPERATOR_REFACTOR/scripts/cpu_033_fetch.sh 13051713`, verify
  every (case, mode, n) row is present in the CSVs, write
  `data/flowvpm_baseline/report.md` (cost tables, per-stage breakdown from
  the profile_*.txt captures at n=1e5, per-row logged errors), summarize
  here, mark this row Done in `START_HERE.md`, and commit the harness +
  data + task-file updates on `matrix-ops` referencing the job id (the
  Integration Phase work so far — 031/031a artifacts, roadmap edits, 033
  harness — is **not yet committed**).
- Resumability: if the job dies, just resubmit
  (`bash MATRIX_OPERATOR_REFACTOR/scripts/cpu_033_submit.sh`) — completed
  (case, mode, n) rows are skipped by grep across all
  `data/flowvpm_baseline/cpu_*.csv` on the cluster, and references are
  skipped per-file.
- Predecessor job `13051516` was **cancelled deliberately** (see amendment
  below): its 10 cube rows live in the remote CSV
  `cpu_m12-2-1_13051516.csv` and are retained/skipped-over; its ring
  (winckelmans) rows and references were deleted. Data from both CSVs
  together forms the final set.
- Other jobs visible under this user on the cluster (`fp-018-*`,
  `13029922`, `1304xxxx`, `13050xxx`) belong to a **different project** —
  do not touch them.
- The FLOWVPM baseline checkout is a git worktree at
  `../FLOWVPM-baseline-e2bd487` (detached at `e2bd487`, read-only use;
  never commit there).

## Amendment (2026-08-05): ring case kernel → gaussianerf

User decision `2026-08-05`: the Integration Phase supports only the FLOWVPM
default `gaussianerf` kernel (the sole `CoreSpreading`-compatible kernel).
The ring case originally used `winckelmans` per `run_leapfrog`; a
winckelmans CPU baseline would not be comparable to the gaussianerf-only GPU
path it gates, so the ring case definition was changed to **gaussianerf**
(all other `run_leapfrog` solver settings retained: cVPM,
`correctedpedrizzetti`, transposed, RK3; geometry and σ schedule unchanged).

Campaign consequence: job **13051516** was cancelled mid-sweep (it had
completed all cube and ring rows through n=1e5 and was on cube cpu1
n=316228). All 10 completed **cube rows and the 7 cube references were
retained** (cube was always gaussianerf); the 10 winckelmans ring rows, ring
references, and ring profiles were deleted on the cluster. Resubmitted as
job **13051713**, which regenerates ring references (gaussianerf) and
resumes: ring re-runs from n=1000, cube continues at cpu1 n=316228.
Interim sanity from 13051516 at the representative n=1e5 (64 threads):
cube u_rel_rms ≈ (see fetched CSV), ring (winckelmans, discarded)
u_rel_rms=1.72e-3, j_rel_rms=8.5e-3 — plausible p=4 magnitudes, confirming
the reference/metric pipeline at scale.

## Amendment (2026-08-05): ring case → helical wake cylinder

User direction `2026-08-05`, during the `031`/`031a` clear-context review: the
vortex ring is **abandoned for this study** and replaced by a helical wake
cylinder at aspect ratio **length = 5 × diameter**. The finalized case
definition above is the wake; the ring definition is retired.

Why. The review's §5.2 geometry analysis found the ring is a poor performance
benchmark on three counts, two of which the wake fixes outright:

1. **Locally thin.** The torus tube is only a few cells across, so a target's
   near set is mostly empty and the `031a` §6 cost model's occupancy assumption
   — every stencil neighbour populated — does not hold. The strategy A/B in
   `032a` would not have measured what the model predicts. A solid cylinder is
   locally dense, so near sets are fully populated.
2. **No single overlap.** The ring's discretization is anisotropic
   (`dS/rl ≈ 1.9`), giving `σ/rl = 3` radially but `≈1.58` azimuthally, so no
   one `β` describes it. The wake uses the cube's convention,
   `σ = 2·(V/n)^(1/3)`, so both cases share one `β = 2`.
3. **Sparse in its bounding cube** — 1.9% fill. The wake at AR=5 fills
   **3.14%** of its bounding cube (`V_cyl = 3.927` against `L³ = 125`), so this
   one is *not* fixed by the change; it is deliberately retained as
   representative of real wakes and staged as a production lever in `035`
   (see that row's lever list). It is far more tractable here than on the ring
   because the occupied region is a compact, locally dense solid.

What the wake keeps that the cube cannot provide: coherent, aligned strengths
(helical vorticity with `|Γ| ∝ r`), which is what stresses the
coherent-cancellation assumption in `031a` §7, exercises the Lamb-Helmholtz χ
channel with structured rather than random vorticity, and represents the
FLOWVPM production workload `035`'s speedup claims are about. Note that a
relative-RMS error gate is intrinsically harsher on a coherent field, where the
net velocity is a small residual of large cancelling contributions, than on the
random cube; a wake row and a cube row at the same measured error are not
equally converged.

Campaign consequence: **all ring rows, ring references, and ring profiles are
discarded**, including those from job 13051713. Cube rows and cube references
are unaffected and retained. The harness was updated in the same pass —
`benchmark_033_common.jl` (`fm033_build_wake`, `fm033_wake_sigma`,
`fm033_sigma`; the `examples/vortexrings` include and the `Roots`/`HCubature`/
`EllipticFunctions` dependencies are gone), `benchmark_033_cpu.jl`,
`prepare_033_references.jl`, `cpu_033_run.sh`, and `cpu_033_submit.sh` (which
no longer installs the three dropped packages). A resubmission must regenerate
wake references from scratch and re-run the wake series; the cube series
resumes where 13051713 left it.

Verified locally without FLOWVPM (stdlib sampling replica, n=1e5): every
particle inside the cylinder, fill 3.14%, mean `|Γ|·n = 0.665` (= 2/3, the
tip-weighted mean of `r/R` over a uniform disc), mean axial direction cosine
0.47 — genuinely helical, neither axial nor azimuthal. The builder itself is
untested against FLOWVPM because the package does not load on the local
machine; **first cluster run must confirm `n_actual = n_target` and a sane
`σ`** before the sweep proceeds.

## Work Record (2026-08-05): job 13051713 reconciled; wake-only job 13058428

Job **13051713** COMPLETED (15h54m, exit 0), but it ran the pre-amendment
script (cube+ring). Results fetched with `cpu_033_fetch.sh 13051713` and
committed at `data/flowvpm_baseline/`: **cube is complete, 14/14 (case, mode,
n) rows** across `cpu_m12-1-29_13051713.csv` + `cpu_m12-2-1_13051516.csv`,
with cube references and cube profiles at n=1e5. The **ring rows are
pre-amendment history, retained with their measured errors** per the 1e-3
gating policy (historical rows stay in the record; only tolerance-passing
configurations feed speedup headlines). The **wake case had not run at all**
— the job predates the wake amendment.

Wake-only resubmission: job **13058428** (`fm033cpu`) submitted 2026-08-05
via `cpu_033_submit.sh`, started immediately on `m12-2-18`. The resume logic
(grep over remote `cpu_*.csv`) skips all 14 completed cube rows, so the job
runs only the 14 wake rows plus wake reference generation. Two harness fixes
were needed and made in the same pass:

- `cpu_033_run.sh`: the Phase-1 reference skip verified the sha256 manifest,
  but the remote manifest was the stale pre-amendment one (cube+ring) and
  ring reference files still existed, so it verified clean and would have
  skipped wake reference generation (first wake row would then die on a
  missing reference). The guard now also requires `direct_reference_wake_`
  entries in the manifest before short-circuiting. `prepare_033_references.jl`
  regenerates the manifest as cube+wake once the wake references exist.
- `cpu_033_submit.sh`: login-node env build pinned to
  `module load julia/1.11.7-6bmogfl` (module default moved to 1.12.6 on
  2026-08-05, which segfaults the host LLVM JIT; the run script was already
  pinned repo-wide).

Cluster-side ring artifacts (7 `direct_reference_ring_n*.csv`, 2
`profile_ring_n100000_cpu*.txt` under `~/FastMultipole-033/.../
flowvpm_baseline/`) are discarded per the amendment but could not be deleted
from this session (remote rm blocked by local tool policy); they are orphaned
— absent from the regenerated manifest and never read — and can be removed in
any manual cluster session. Ring rows inside the two remote CSVs do not
collide with the cube/wake resume greps.

Expected walltime: well under the 3-day limit — wake-only is roughly half of
13051713's 15h54m case work plus wake reference generation (the n=1e6
single-thread wake row and its reference are the long tail). On completion:
fetch, verify 14 wake rows, write `data/flowvpm_baseline/report.md`, then mark
Done. Row NOT Done yet.

## Work Record (2026-08-06): job 13058428 complete — verification, report, Done

Job **13058428** completed 2026-08-06 ("task 033 complete" in
`data/flowvpm_baseline/fm033cpu-13058428.out`); results fetched. Verification:

- **Wake: 14/14 rows** (7 n x cpu1/cpu64) in `cpu_m12-2-18_13058428.csv`,
  with `n_actual = n_target` exactly and sigma matching
  `2*(V_cyl/n)^(1/3)` at every grid point (the amendment's first-run check).
- **Cube: still 14/14 rows** across `cpu_m12-2-1_13051516.csv` +
  `cpu_m12-1-29_13051713.csv`.
- **References**: the regenerated sha256 manifest covers cube+wake (14
  files); `shasum -a 256 -c direct_reference_checksums.sha256` — all OK.
- Wake profiles at n=1e5 (`profile_wake_n100000_cpu{1,64}.txt`) present.
- Ring rows in the 13051713 CSV and the 7 ring references + 2 ring profiles
  remain in the repository as retained pre-amendment history (no 033 table or
  later gate/speedup reads them).

Deliverable report written: **`data/flowvpm_baseline/report.md`** —
provenance (jobs 13051516/13051713/13058428, julia 1.11.7, commit e2bd487),
full cube and wake timing/error tables, the n=1e5 per-stage profile breakdown
(bottleneck: the `gaussianerf` direct nearfield is >99% of single-thread UJ
time in both cases and dominates worker time at 64 threads; erf+exp alone
~30% of the solve; far field <0.5%), 64-thread scaling efficiency (no
speedup at n<=3162; 45-93% efficiency from n=1e4, wake ~92% at 1e6), the
1e-3 velocity-gate audit (cube passes only n<=3162, wake only n=1000 —
every FMM-active default-parameter row fails, so 035 speedup headlines need
tuned gate-passing baselines), and threats to validity.

Row marked **Done** in `START_HERE.md` (Approved left unticked).

## Approval Notes

Reviewer: clear-context approval subagent, 2026-08-06. Read only
`START_HERE.md` (protocol + Integration Phase preamble incl. the 1e-3
tolerance policy and wake amendment), this task file in full, the deliverable
`data/flowvpm_baseline/report.md`, the three campaign CSVs, job logs, the four
cube/wake n=1e5 profiles, `references/` + manifest, and the five harness
scripts.

Re-verified:

- **Checksums**: `shasum -a 256 -c direct_reference_checksums.sha256` — all
  14 files (7 cube + 7 wake) OK, exit 0.
- **Row counts**: 14/14 wake rows in `cpu_m12-2-18_13058428.csv`; 14/14 cube
  rows across `cpu_m12-2-1_13051516.csv` + `cpu_m12-1-29_13051713.csv`;
  `n_actual = n_target` on every cube/wake row.
- **Report numbers vs raw CSVs** (spot checks all match): wake n=1e6
  cpu1 1851.67 s / cpu64 31.55 s / u_rel 6.39e-2; cube n=1e6 1776.59 /
  55.98 (min 34.42) / 6.75e-2; wake n=3162 3.48e-3; cube n=1e4 1.07e-2.
  Gate audit confirmed: cube passes 1e-3 only at n<=3162 (~1e-15), wake only
  at n=1000 — every FMM-active default-parameter row fails, exactly as the
  report states. Scaling table recomputed from CSVs (cube 1e5 45.3x/71%,
  wake 1e6 58.7x/92%, cube 1e6 min-based 51.4x) — matches.
- **Sigma construction**: `fm033_wake_sigma(n) = 2*(pi*R^2*5D/n)^(1/3)` in
  `benchmark_033_common.jl` matches the claimed convention; wake n=1000 gives
  0.315537 = CSV value; cube n=1000 gives 0.2 = CSV value.
- **Profiles**: report §4 frame counts traced to the profile files
  (nearfield 112059/99.4% cube cpu1, g_dgdr 42767, custom_erf64 19664,
  exp 14604, set_hessian 15158; wake analogues likewise). Bottleneck
  identification is unambiguous.
- **Harness vs stated policy**: reps (5/3 UJ, 3/2 step at n>=316228), warmup
  excluded, medians+mins logged, per-row error logging, reference sampling
  (all targets n<=1e4 else 512 at seed 33026+n), single-thread reference
  generation (2.0.4 `direct_multithread!` bug) — all as documented.
- **Minimal invasiveness**: 033 commits (0ad1ad5, af3d3df, 640933c, harness
  commits) touch only `MATRIX_OPERATOR_REFACTOR/`; no `src/` changes.
- **Policy framing**: report §5's consequence statement matches the
  START_HERE preamble verbatim in substance — historical rows retained with
  errors, speedup headlines restricted to gate-passing baselines.

Verdict: **APPROVED**. Objectives met in full; report is clear and honest
about threats to validity (in-run reference generation, node heterogeneity,
large-n variance, retained ring history).

Minor observations (no action required):

1. In report §4 the cpu1 "exp" row (14604) is the child of the
   "custom_erf64" row's :186 frame (19664), so the "~30% erf+exp" statement
   overlaps at those frames; however the full `custom_erf` wrapper frame
   alone is 41331 (36.7% of cube cpu1), so ~30% is if anything conservative
   and the conclusion stands.
2. The "g_dgdr_gauserf kernel math" row counts only the kernel.jl:56
   call-site (42767); a second site at :55 adds 14644 — an under-attribution
   that only strengthens the nearfield-dominance conclusion.
3. §4's percentage base (112711 in-call samples) differs trivially from the
   file's total snapshots (112787).
