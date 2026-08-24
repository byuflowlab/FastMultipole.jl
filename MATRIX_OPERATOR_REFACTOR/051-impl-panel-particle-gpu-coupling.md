# 051 Impl: Panel–Particle GPU Coupling

## Status and entry gate

**CLOSED `2026-08-24` — approved (user pre-approval this session +
clear-context audit APPROVE, zero discrepancies).** Verdict: the coupled
panel–particle GPU step FITS the 3.3 s budget — B′ stack 0.373 + 0.124 +
2.020 + 0.607 = 3.124 s (5.3% margin); body solve via opt-in
source-potential matrix S on `Backslash` (tuned BLAS=8 gemv, job
13395348). Commits: FastMultipole `22376fea`/`d8258a7d`/`b332bb05`,
FLOWPanel `1159c8f`/`8a48bc5`/`5272a5f`, FLOWVPM `9b5b7cd`. Deferred to
`052+`: device-resident gemv, Float32 S, pass-2 kernel work.

Originally staged `2026-08-20` (user direction).

Entry gate: `050` complete and approved — this row implements the `050`
verdict's named shape; do not re-litigate the A/B/C decision here. Fallback
resource envelope per user: **1×GPU + 64 CPU threads**.

## Motivation

The 018 step splits as wake influence 64.2% / body 25.3% / solve 9.3%
(023 profiling); after `048`/`049` accelerate the particle side, the panel
passes become the binding constraint on the ≤3.3 s/step budget (the CPU
body-pass floor alone is ~36 s). This row wires the `050`-selected GPU
coupling — panel GPU direct kernel, system-on-system passes, or hybrid —
into FLOWPanel's production influence path.

## Objective

The `050` verdict implemented: panel↔particle GPU coupling wired through
FLOWPanel's `influence!`/`FastMultipoleBackend`, passing pass-by-pass
parity at the 018 operating point with no regression on FLOWPanel CPU
tests.

## Method

### Stage 1 — implement the `050` shape

Per the verdict: (A) distinct-target support on the radix path, and/or (B)
system-on-system GPU passes mapped onto the existing 3-pass structure
(`FLOWPanel_simulate.jl:673-712`), and/or (C) hybrid GPU-particles /
64-thread-CPU-panels. If a panel GPU direct kernel is in scope, port the
`direct!` overload (`FLOWPanel_abstractbody.jl:1260`) for the element types
the 018 driver exercises (constant source/doublet tris + vortex
rings/sheets/filaments).

### Stage 2 — wiring

Wire through `FLOWPanel_fmm.jl:60` `influence!` /
`fmm!(targets::Tuple, sources::Tuple)` (`:88`, plan-reusing `:114`) and the
separate `Estr_fmm!` call (`FLOWPanel_wake.jl:2052`), preserving per-pass
kerneloffsets and derivative switches. Respect the FmmPlan /
NearfieldInfluenceCache disposition decided in `050`.

### Stage 3 — parity + regression harness

- Pass-by-pass parity vs the CPU path at the 018 operating point (each of
  the three influence passes + the SFS pass, compared independently).
- No regression on FLOWPanel CPU tests.
- **Step-head gotcha (cost 3 jobs previously):** the harness must run the
  full step head (`maneuver!` + reset/freestream/kinematic + `update_TE!`)
  before any influence eval, or the restored first wake row silently NaNs
  every target.

## Gates and verdict

- Pass-by-pass parity green at the 018 operating point (phase 1e-3 gate;
  F64 tighter where applicable).
- FLOWPanel CPU test suite green; FastMultipole/FLOWVPM suites unregressed.
- Timing of the coupled passes reported against the `049` budget table
  (feeds `052`).

## Artifacts

- Source changes on the unified branches (FLOWPanel + FastMultipole as the
  `050` shape requires) + tests.
- Parity/timing CSVs + results section appended to this doc.

## Verification

- Same-job CPU/GPU pass comparisons with job IDs; NaN guard on the
  step-head gotcha exercised in the harness.

## Recorded context (2026-08-20 staging)

**FLOWPanel per-step influence structure** (`FLOWPanel_simulate.jl:673-712`):
three separate fmm! passes — wake→(bodies+particles), panel solve,
bodies→targets (different kerneloffsets/derivative switches per pass) +
separate `Estr_fmm!` call (`FLOWPanel_wake.jl:2052`) reusing wake trees.
Targets ≠ sources in these passes. Entry: `FLOWPanel_fmm.jl:60` `influence!`
over heterogeneous tuples → `fmm!(targets::Tuple, sources::Tuple)` at `:88`
(plan-reusing at :114).

**FMM-compatibility overloads (CPU, complete):**
`source_system_to_buffer!` `FLOWPanel_abstractbody.jl:1096`, `direct!`
`:1260`, `body_to_multipole!` per element type
(`FLOWPanel_nonliftingbody.jl:224-234`,
`FLOWPanel_liftingbody.jl:705,797,906`), PanelWake trio
(`FLOWPanel_wake.jl:326,525,562,565`), filaments (`:2763,2825,2841`).

**GPU support in FLOWPanel today: none** (only a `GPUArray` kwarg on the
dense linear solve, `FLOWPanel_liftingbody.jl:379-412`; no CUDA dep).

**Harness gotcha (cost 3 jobs):** run the full step head (`maneuver!` +
reset/freestream/kinematic + `update_TE!`) before any influence eval or the
restored first wake row silently NaNs every target.

**018 operating point:** 36,752 panels (45_185_ct4 mesh), NT=36 steps/rev,
~181k particles at maturity (342k on the 6R arm), THREADS=64; CPU baseline
170–230 s/step, split wake 64.2% / body 25.3% / solve 9.3%, body-pass floor
~36 s; budget ≤3.3 s/step.

## Worklog — 2026-08-22 (session 6): Stage-0 audit of pre-existing implementation + hardening

Per user directive, the 051-shaped code found on-branch (committed rect kernels in
FastMultipole `1cd0b98a`, untracked FLOWPanel seams, `fm051_rect_bench.jl`; commit messages
claiming "300ms on gpu") was treated as UNTRUSTED and audited component-by-component before
reuse. Plan: `MATRIX_OPERATOR_REFACTOR/051-plan-2026-08-22.md`.

### Audit verdicts (all four components adopted after fixes)

| Component | Verdict | Evidence |
|---|---|---|
| Host points functor (`RectangularGaussianErfVortex`, direct_rectangular.jl) | SOUND | Line-by-line vs FLOWVPM_fmm.jl:144-218 + g_dgdr_gauserf + fdlibm erf (all 62 coefficients): bitwise-identical to CPU-FLOWVPM incl. summation order; `r2==0`-only exclusion confirmed (no eps2 guard) |
| Host panel functor (`RectangularPanelInfluence`) | SOUND-WITH-NOTES | Numerically verified vs live `FLOWPanel._induced`: 600 random panels × 5 target classes × 5 tags × 3 filament families → ≤1e-12 rel (U and J) away from edges; near-edge (1e-6 of an edge) blowup to ~1e-2 is shared ill-conditioning, not transcription error; self-pair detector bit-equivalent to FLOWPanel's; provenance claim (75b45c7) verified — `git diff 75b45c7 d6bf8b6` empty for elements_fmm/abstractbody |
| CUDA kernels (translate_batched_cuda.jl:8955+) | SOUND-WITH-NOTES, unexecuted | No races/OOB/shared-mem hazards; same source-visit order as host; parity is FMA/libdevice-limited (~1e-13 U / 1e-12 J points, 1e-11/1e-10 panels, well-separated), never bitwise |
| FLOWPanel seam (`FLOWPanel_gpu_influence.jl`) | SOUND-WITH-NOTES | Solve pass can never match (scalar_potential is a Vector at every solve call site); fallbacks complete incl. Estr-stays-on-radix; `+=` writeback matches fmm!; default-off is a memoized Ref check |

Test evidence (local, FLOWPanel env so cross-check layers activate): 45/45 pass —
point functor vs FLOWVPM production pair math EXACT; panel functor vs FLOWPanel `induced`
1e-15–1e-16 across tags 1/4/5 combos and all three filament families.

### Watch-item disposition

- **2.6e-4 CPU-vs-GPU discrepancy (049)**: root cause was the FLOWVPM CUDA extension's
  absolute `r2>1e-6` pair cutoff; fix commit `baf8fb3` verified in FLOWVPM lineage. The
  rectangular kernels never had the guard (audit-confirmed). Closed for 051 gate purposes.
- **production-numerics red flag — RESOLVED (D16, user decision 2026-08-22)**: the
  uncommitted BRAINSTORM-025 change of the CPU-wide default filament regularization
  (Vatistas → `GaussianRegularization`, FLOWPanel_elements_fmm.jl:923) is RATIFIED — Gaussian
  is the production default. 051 parity was like-for-like regardless (the seam maps
  `FILAMENT_REGULARIZATION[]` to the functor family). See decisions log D16.

### Hardening applied (2026-08-22)

- direct_rectangular.jl: `_rect_assert_host` (CuMatrix-before-`load_cuda_radix_lifecycle!()`
  now fails with the fix instead of an opaque scalar-indexing error); `_rect_check_args` now
  F64-only for the panel functor (its 1e-12-scaled singularity guards are inert in F32) and
  validates tag/nv rows (they feed unchecked device truncation); docstring corrections
  (device parity is FMA/libdevice-limited, not "roundoff"; TE-wake packing = two tag-3
  triangles per `_induced_wake`, not one quad).
- translate_batched_cuda.jl: `Int(...)` → `unsafe_trunc` in the panel pair loop (removes the
  InexactError trap path; inputs validated host-side).
- test/direct_rectangular_test.jl: +2 testsets — argument validation (6 negative tests) and
  CUDA device parity (tile-boundary sweeps 1/127/128/129/255/256/257/512 sources ×
  1/255/256/257 targets for both kernels, F32 points arm, device `+=` contract, degenerate
  sizes; per-target max-relerr gates at the audit tolerances). Skips cleanly off-GPU;
  wired into fm051_run.sh ahead of the bench.
- fm051_rect_bench.jl: per-target max relerr recorded per pass (branch-flip detection, not
  dilutable by the global RMS; gate = 100× the RMS gate); pass-2 J gate relaxed to the
  audit-realistic 1e-10 (was a blanket 1e-11).
- FLOWPanel_gpu_influence.jl: header comment corrected (capture surface is any
  `direct_conditioning` body-source call, not only `_sa_body_influence!`).

Deliberately NOT changed: `eps(nznorm)` value-based eps in `_rect_rotate_to_panel`
(transcription-faithful to FLOWPanel; GPU-safe per audit).

### TODO (user-directed, 2026-08-22): FilamentWrapper seam extension — IN SCOPE FOR 051

The mini-harness validation exposed that in the full p018 configuration pass 1 falls through
to fmm!: `get_sources(::PanelWake)` appends a `FilamentWrapper` when
`include_final_filament=true` (the default; p018 doesn't override), and a mature run always
has `wake.overflowed[] == true`, so the wrapper is non-empty and
`_gpu_source_supported(::FilamentWrapper)` (requires n == 0) declines the whole wake pass.
**Decision: extend the seam in this item** — pack the active final-filament segments as
rectangular sources (tag-3 filament columns per the FLOWPanel filament kernel, same
family/kerneloffset plumbing as the TE-wake packing) instead of declining, so production
pass 1 actually runs on the GPU. Gate as with the other passes: mini-harness arm with
`FM051_MINI_FINAL_FILAMENT=1` must flip from documented-fallback to seam-accepted with the
1e-3 phase gate (tight informational bound expected, F64), then full-run parity on p018.

## Worklog — 2026-08-22 (session 7): FilamentWrapper seam extension IMPLEMENTED (local gates green)

Implementation (all uncommitted, intentional — preserve):

- **FastMultipole `src/direct_rectangular.jl`**: tag-3 now accepts `nv == 2` = OPEN
  bound-vortex filament (single segment v1→v2, no closing edge — a closed 2-ring would sum
  v1→v2 + v2→v1 = 0). One-line change in `_rect_ring` (`nseg = nv == 2 ? 1 : nv`); since the
  CUDA panel kernel shares `_rect_panel_pair`, host and device are covered by the same edit.
  `_rect_validate_panel_sources` allows nv=2 only with tag 3; docstring row table updated.
  The nv=2 self-pair check is inert by construction (packed v3=v4=v2 gives zero area, so the
  `< 1e-24·area` threshold is unreachable). Sign conventions verified: the segment functions
  are even under (r1,r2)→(−r1,−r2), so `_rect_ring`'s (v−target) velocity equals the CPU
  FilamentWrapper `direct!`'s (target−v) call, and the gradient orientation matches its
  (v−target) call exactly.
- **FastMultipole `test/direct_rectangular_test.jl`**: +gate 4a (straight-segment analytic,
  1e-12), +gate 4b (closed tri ring == sum of its 3 open segments, all 3 families), +open-
  filament FLOWPanel-parity arm per family (vs `pnl._bound_vortex_velocity/_gradient`, the
  exact functions FilamentWrapper's `direct!` sums), +2 validation cases (nv=2 legal only for
  tag 3), and `_pack_random_panel!` now emits nv=2 columns (~25% of tag-3) so the CUDA
  tile-boundary parity sweep exercises filaments on device (runs in Job A).
- **FLOWPanel `src/FLOWPanel_gpu_influence.jl`**: `_gpu_source_supported(::FilamentWrapper{<:PanelWake}) = true`
  (non-PanelWake wrappers keep the empty-only rule); `_gpu_source_columns` = `get_n_bodies`;
  new `pack_filaments!` (same enumeration/strength/`core_size` as the wrapper's
  `source_system_to_buffer!` at FLOWPanel_wake.jl:2790 — `fmm_to_filament_index` +
  `_final_filament_strength`, nodes row `nwakes[]+1`, cols j/j+1, tag 3 nv 2); new
  `_gpu_pack_source!` method whose work buffer is keyed on the WRAPPED wake (`:src_filaments`
  role) because `get_sources` builds a fresh wrapper every call (keying on the wrapper would
  defeat reuse and grow the cache every step). Header fallback list updated.
- **FLOWPanel `benchmark/fm051_pass_parity.jl`**: mini-case comment updated
  (`FM051_MINI_FINAL_FILAMENT=1` now expects seam-ACCEPTED, not the documented fallback).

Local gate evidence (artifacts + sha256 in `data/panel_particle_gpu_coupling/`,
`sha256_local_2026-08-22.txt`):

| Gate | Result |
|---|---|
| `direct_rectangular_test.jl` in FLOWPanel env | 57/57 (was 45; +12 filament/validation). Open-filament parity vs FLOWPanel: U bitwise (relerr 0.0), H ≤3.0e-16, all 3 families (`rect_test_filament.log`) |
| Mini harness `FM051_MINI_FINAL_FILAMENT=1` | **flipped fallback → seam_accepted=yes**: FilamentWrapper listed supported, packed_columns=6; pass 1 phase 2.82e-14 (1e-3 gate; 1e-11 informational also met), pass 3 3.05e-15; zero fallbacks; exit 0 (`mini_filament.log`) |
| Mini harness plain (baseline) | unchanged: all gates PASS, pass 1 5.11e-15, pass 3 2.18e-14, exit 0 (`mini_plain.log`, CSV `fm051_pass_parity_mini_plain.csv`) |
| FLOWPanel full suite seam-off + FastMultipole full suite | re-run in flight at write time — results recorded below when done |

### Cluster layout for Jobs A/B — RESOLVED (was a handoff open item)

- FLOWPanel on orc: standing checkout `~/projects/FLOWPanel.jl` EXISTS (branch
  `fastmultipole` but at `5615ada`, dirty, and WITHOUT the 051 seam/harness files) — used
  READ-ONLY. The p018 restart set is `~/projects/FLOWPanel.jl/data/p018_cs_f1_l3p4` (383M:
  body/wake/particles/filaments PVDs + metadata) and the `45_185_ct4` mesh is in
  `~/projects/FLOWPanel.jl/examples/data/`.
- Job B therefore uses a SYNCED tree `~/FLOWPanel-046` (consistent with
  `~/FLOWVPM-046`/`~/FastMultipole-046`) with symlinks `data/p018_cs_f1_l3p4` and
  `examples/data` into the standing checkout, and a NEW env `~/fm051env` (fm048env recipe +
  FLOWPanel dev + VSPGeom; fm048env is left untouched for 048/049 replay — it does not dev
  FLOWPanel).
- New local driver: FLOWPanel `benchmark/slurm/fm051_parity_submit.sh` (syncs all three
  trees so Jobs A and B see identical code, makes the symlinks, builds fm051env, sbatches
  `fm051_parity.sh`). `fm051_parity.sh` adjusted: `EXPECTED_REPO=$HOME/FLOWPanel-046`,
  `ENVDIR` default `$HOME/fm051env`, symlink presence check.

### Job A results — H200 job 13306457 (2026-08-22): Stage-1/2 device gates ALL PASS

Artifacts: `data/panel_particle_gpu_coupling/vpm051-13306457.out` (sha256 `10544c36…`),
`fm051_results.csv` (`785925a8…`), hashes in `sha256_local_2026-08-22.txt`. (Job B first
attempt 13306465 died pre-output: `set -u` vs the cluster's unbound `DEBUGINFOD_URLS` in
`/etc/profile.d/debuginfod.sh`; `fm051_parity.sh` fixed to the fm051_run.sh `set -eo` +
deferred `-u` pattern; resubmitted as 13306475.)

- **Device testsets on H200**: CUDA device parity **98/98** (tile-boundary sweeps for both
  kernels INCLUDING the new nv=2 filament columns, F32 arm, `+=` contract, degenerate
  sizes); analytic gates 17/17; validation 7/7; point functor 9/9.
- **Bench parity (p018 step 710, np=210,056 → 36,752 panel centers and back), gates F64
  1e-11 RMS + per-target ≤100×, pass-2 J 1e-10 — all pass=true with orders of margin:**

| Arm | u_err | j_err | t_med |
|---|---|---|---|
| pass1 F64 U | 1.31e-16 | — | 0.124 s |
| pass1 F64 U+J | 1.31e-16 | 2.61e-16 | 0.209 s |
| pass1 F32 U / U+J | 2.81e-5 | 4.49e-5 | 0.071 / 0.129 s |
| pass2 F64 U | 4.52e-15 | — | 2.019 s |
| pass2 F64 U+J | 4.52e-15 | 1.30e-15 | 3.640 s |

- **Timing vs the B′ estimate bands — both passes land SLOW of band (flag for Stage 3/4):**
  pass 1 (wake→panels) 0.124 s vs the 0.02–0.04 s estimate; pass 2 (panels→particles)
  U-only 2.02 s at the very top of the 0.4–2.0 s band. Revised pessimistic stack with
  measured numbers: 0.373 (particles, D15 worst) + 0.124 + 2.02 ≈ **2.52 s → ~0.78 s solve
  headroom** (was 0.89 s) — B′ still prices in on U-only. **If production pass 3 must
  deliver J (Hessian) at particle targets, the 3.64 s U+J time busts the 3.3 s budget by
  itself** — Stage 3/4 must confirm the p018 `body_hessian_to_particles` /
  `wakerow_no_hessian_to_particles` flags (Job B's cfg dump records `needs_body_hessian`)
  and, if J is required, weigh the F32 pass-1 option and/or kernel optimization before the
  solve decision.

### Job B results — H200 job 13306475 (2026-08-22): mechanics green, 3 gate FAILs root-caused to harness metric design

Artifacts: `data/panel_particle_gpu_coupling/slurm-fp-051-parity-13306475.{out,err}`
(sha256 `dfa3bfb0…` / `6c000443…`), `fm051_pass_parity_full.csv` (`bf9d7ef4…`). First
attempt 13306465 died pre-output (`set -u` vs unbound `DEBUGINFOD_URLS` in the cluster's
`/etc/profile.d/debuginfod.sh`; script fixed to the fm051_run.sh pattern).

**What went RIGHT (the 051 mechanics):** warm-start from p018 step 1034 clean (step-head
NaN guard passed); pass 1 seam-ACCEPTED with the mature FilamentWrapper packed (80 tag-3
nv=2 columns) + PanelWake (80) + 181,307 particles; pass 3 seam-accepted (37,072 columns =
36,752 panels + 320 TE-wake tris); solve 0 seam hits; the SFS self-pair fallback fired as
designed (.err line 97) so Estr ran through identical fmm!+postcalc in BOTH arms. Timing
signal: pass 3 CPU 57.5 s → seam 2.9 s (~20×); pass-1 seam wall 277.8 s is first-call CUDA
JIT-dominated (Job A's warmed kernel is 0.12 s).

**Parity numbers:** pass 1 — body_velocity 5.0e-15, probe_U 1.5e-15, particle_U 2.19e-4
(= CPU-arm fmm error vs seam-exact; passes 1e-3), particle_J 0.0 (J to particles rides only
the arm-identical self fmm! call — PANEL_WAKE_HESSIAN_TO_PARTICLES=false). Pass 3 —
particle_U 1.0e-4, probe_U 7.8e-6.

**The 3 FAILs, root-caused (one physical cause + one metric flaw, no seam defect found):**

1. `pass1.phase` 1.185 and `sfs.estr_1e-10` 1.185 are the SAME number: `particle_Estr`.
   Both arms compute Estr through the identical fmm! postcalc, but the dynamic procedure
   (FLOWVPM_subfilterscale.jl clamp/clipping-backscatter branches) is a DISCONTINUOUS
   function of the pass-1 U input, which legitimately differs by ≤2.19e-4 (fmm vs exact).
   Clip flips at isolated particles produce O(1) per-particle Estr deviations. A tight
   per-particle Estr bound is unsatisfiable by design in full mode; the 1e-10 "threading
   noise" gate was calibrated for mini (identical inputs).
2. `pass3.phase` 6.69 (body_velocity, target 19431, class scale 2.62e3): the recorded data
   cannot distinguish a floor artifact (per-target rel denominators floor at 1e-6·scale, so
   a near-null target FAILs on an absolute diff of ~1.8e-2 against a class scale of 2620)
   from a genuine error; the metric demands absolute agreement 1000× tighter than the fmm
   tolerance the CPU reference itself carries. NOT closable from job 13306475's outputs.
   (Checked and eliminated: CPU farfield DOES include the attached TE wake —
   `body_to_multipole!` FLOWPanel_liftingbody.jl:734.)

**Redesign validation:** both mini arms re-run green after the redesign (exit 0, all
gates PASS; per-target rel IDENTICAL to the pre-redesign baselines — plain 5.108e-15 /
2.184e-14, filament 2.817e-14 / 3.051e-15; new diff/scale metrics ~1e-15; mini SFS gates
skip as always — mini's pfield has SFS disabled). Logs archived:
`mini_plain_v2.log` (`f22cb0e8…`), `mini_filament_v2.log` (`c2310e5d…`).

**Harness redesign applied (session 7, all documented in-code):** `class_parity_stats`
replaces `pertarget_relerr` — per class it now records worst rel WITH |diff| and |ref| at
the worst target, `diff_to_scale` (= max |diff| / class scale, floor-independent),
over-gate and divergent (rel>0.5) counts, and top-5 offenders. Gate changes: (a)
particle_Estr is excluded from the pass-level phase gate (it has its own SFS gate; one
cause was failing two gates); (b) FULL mode phase-gates `diff_to_scale ≤ 1e-3` with
per-target rel reported informationally (mini gating unchanged — both arms exact there);
(c) full-mode Estr gate = divergent fraction < 1e-3 (clip-flip sparsity) + diff_to_scale
informational. Rationale: the full-mode CPU arm is itself fmm-approximate, so the harness
must gate "arms agree to fmm tolerance at class scale", not 1e-9·scale absolute agreement
on null targets. Nothing is hidden — both metrics print for every class.

### Pending cluster gates (rerun)

- **Job B rerun** (user submits): re-run `bash benchmark/slurm/fm051_parity_submit.sh` —
  now a COMBINED job (CLAUDE.md cluster-jobs rule): stage 0 = rect-kernel profile
  (FM051_BENCH_PROFILE: pass-2 tag1/tag3 split + pass-1 occupancy ×2/×8 probe, results in
  `benchmark/results/fm051_profile_results.csv`) + the full parity harness with the
  redesigned diagnostics. Expected: pass 1 PASSES on diff_to_scale (~2e-4); pass 3 prints
  the real kerneloffset-deviation severity (diff/scale + offender table) — if THAT exceeds
  1e-3 it is a genuine finding to bring to the user, not a metric artifact.

## Worklog — 2026-08-22 (session 8): Job B rerun 13306588 — redesigned metrics validate; pass-3 kerneloffset deviation is REAL

H200 job **13306588** (user-submitted, `fm051_parity_submit.sh`), COMPLETED the harness
end-to-end in ~10 min, exit 1 (by design: one gate FAIL). Artifacts + sha256 in
`data/panel_particle_gpu_coupling/` (`sha256_13306588.txt`): job .out/.err,
`fm051_pass_parity_full_13306588.csv`.

Setup identical to 13306475: p018_cs_f1_l3p4 restart step 1034, parity step 1035,
n_panels=36752, n_particles=181307, FilamentWrapper seam-accepted (80 columns), solve
0 seam hits, `FILAMENT_REGULARIZATION[] = GaussianRegularization` (D16).

**Gate results (redesigned metrics):**

| gate | value | bound | verdict |
|---|---|---|---|
| pass1 seam_accepted | 1 hit | — | PASS |
| pass1 diff_to_scale | 3.103e-06 | 1e-3 | PASS |
| pass1 per-target rel (informational) | 2.187e-04 | 1e-3 | PASS |
| solve seam_never_matches | 0 hits | — | PASS |
| pass3 seam_accepted | 1 hit | — | PASS |
| **pass3 diff_to_scale** | **1.235e-02** | 1e-3 | **FAIL (genuine)** |
| pass3 per-target rel (informational) | 6.691e+00 | — | recorded |
| sfs estr_divergence_sparse | 5.516e-05 | 1e-3 | PASS |
| sfs estr diff_to_scale (informational) | 1.560e-02 | — | recorded |

- **Both 13306475 artifact hypotheses CONFIRMED**: pass-1 phase error was fmm-vs-exact
  input difference (now 3.1e-6 on diff/scale; worst per-target rel 2.19e-4 on particle_U,
  matching the predicted ~2e-4); Estr O(1) offenders are sparse clip flips (10 of 181307
  divergent = 5.5e-5 < 1e-3 sparsity gate).
- **Pass-3 finding is REAL, pre-declared as user-facing**: body_velocity diff/scale
  1.235e-2 (scale 2.620e+3), 36708/36752 targets over-gate, **9415/36752 divergent
  (rel > 0.5)**, worst offenders |diff| up to 17.5 m/s on |ref| ~2.8 m/s. Root cause is
  the documented seam deviation: the rect seam evaluates ALL self-body pairs at
  `kerneloffset_panel` (1e-3), while CPU `direct_conditioning` flips the offset only for
  NEARFIELD self-pair blocks. Severity is physically plausible: the .err records
  "FMM radius inflation (5.9e-3) exceeds 10× the panel radius (2.0e-5)" — the offset is
  ~50× the panel size, so offset-vs-no-offset placement materially changes near-body
  velocities. Pass-3 particle_U (9.96e-5) and probe_U (7.8e-6) remain clean — the
  deviation hits body self-targets only. USER DECISION REQUIRED (accept / make the seam
  distance-aware to mirror `direct_conditioning` / other) before Stages 3–4 close.
- Timing: pass 3 CPU 57.4 s → seam 2.9 s (19.7×, consistent with 13306475). Pass-1
  harness walls: cpu 154.6 s vs seam 162.3 s (harness-context measurement including
  snapshot/restore + fmm arms; production pass-1 cost remains Job A's 0.124 s figure).
- **Stage 0 (profile) FAILED to run** — driver bug, not the bench: `fm051_parity.sh`
  invoked `fm051_rect_bench.jl` with no snapshot argument (usage error at line 42), then
  unconditionally copied a stale Job-A `fm051_results.csv` as
  `fm051_profile_results.csv`. FIXED locally (argument
  `${VPM051_BIN:-~/FLOWVPM-046/data/fm049/p018_710_particles.bin}` passed; copy now only
  on success; bash -n clean). The stale CSV was NOT archived. Profile rides the next
  cluster job.

### Session-8 continuation: attribution re-aimed, `kerneloffset`→`core_size` rename, pass-3 stage built

1. **Offset-semantics hypothesis KILLED by code audit** (user prompt + explorer agent):
   offsets are live but are regularization CORE RADII, not surface offsets. p018:
   `core_size_panel = R*1e-10 = 1.19e-11` (effectively singular; conditioning flips
   nearfield self-blocks to it), `core_size_targets = 1e-3` (physical filament core for
   body→targets; the doublet sheet is VortexRing filaments, Gaussian family per D16).
   Since BOTH arms are effectively singular on self-body pairs, the 1.2% cannot be a
   regularization mismatch → leading hypothesis became CPU fmm truncation error.
2. **Rename executed** (user-directed, Opus agent, 105 files across FLOWPanel /
   FastMultipole / FLOWVPM): `kerneloffset{,_panel,_targets}` → `core_size{,_panel,_targets}`,
   ENV `KERNELOFFSET*`→`CORE_SIZE*` (old ENV + constructor kwargs kept as fallbacks;
   both-given errors). Restart data SAFE: replay state is TOML string keys, old keys
   accepted (`FLOWPanel_replay.jl:392-395,672-673`). Validation: FLOWPanel suite
   4814/4814; rect tests 57/57; mini parity bit-baseline (5.108e-15 / 2.184e-14).
   Known pre-existing issue (not touched): FLOWPanel `Pkg.test()` fails on missing
   `LaTeXStrings` in test/Project.toml; `test/runtests.jl` direct run is the baseline.
3. **Pass-3 attribution stage BUILT and validated** (user approved running the
   verification): new `benchmark/fm051_pass3_attribution.jl` (three-way metrics,
   harness-identical definitions) + `benchmark/fm051_attribution_debug.jl` (standalone
   small-problem driver) + harness wiring (`compare_pass!` gains `keep=`; new stage
   after pass 3 re-runs `_sa_body_influence!` from the pre-pass state with
   `pnl.DirectBackend()` on BODY targets only — exact brute force, production
   conditioning preserved; gates `pass3_attribution.seam_vs_exact` ≤ 1e-10 diff/scale
   (`FM051_GATE_ATTR`), reports `cpu_fmm_vs_exact` as an informational finding;
   `FM051_ATTRIBUTION=0` skips).
4. **Debug findings** (`attr_debug_refined_diamond.log`): (a) test_helpers' diamond has
   4 chordwise cells → source octree collapses to ONE leaf → zero m2l pairs → "fmm" arm
   trivially exact (probe: n_branches src=1 at every MAC/method) — debug driver now
   builds a chordwise-refined diamond (40×20, 3200 panels); (b) with a real farfield the
   expected signature appears: seam-vs-exact 5.4e-15 (P-independent), cpu-fmm-vs-exact
   8.9e-6 / 8.8e-7 / 8.7e-9 at P=4/6/10 (clean truncation convergence); (c) warm
   exact-arm cost 6.7e-8 s/pair → p018 body-only direct ≈ 90 s at 4 threads (~25-35 s
   at 16) — affordable as a full-run stage. Mini harness with the stage: both variants
   OVERALL PASS, attribution 7.96e-16 (plain) / 1.21e-15 (final-filament), baselines
   unchanged. Logs + sha256 appended to `sha256_local_2026-08-22.txt`.
5. **Prediction for the Job-B rerun**: seam_vs_exact ≈ 1e-13 or better and
   cpu_fmm_vs_exact ≈ 1.2e-2 ⇒ the 13306588 pass-3 "FAIL" is the LEGACY CPU fmm
   truncation error on body self-influence measured against an exact reference — a
   finding about the production CPU path (interaction-list separation contract: the
   ρ_t-style radius inflation nulls singular-vs-regularized but not truncation), not a
   seam defect. Gate re-aim then follows (user decision).

### Session-8 late: Job B 13309844 — attribution REFUTES the prediction: the CUDA seam arm is the deviator

**H200 job 13309844** (user-submitted; artifacts + `sha256_13309844.txt` archived):
profile stage RAN; attribution stage RAN (exact arm 28.4 s). Result INVERTS item 5:

**Profile stage results** (`fm051_profile_results_13309844.csv`; p018 step 710,
np=210,056, panels=36,752, H200):

| measurement | time |
|---|---|
| pass 2 U tag split | source-only 1.142 s / ring-only 0.964 s / combined 2.020 s (≈ additive) |
| pass 2 U+J tag split | source-only 1.553 s / ring-only 2.133 s / combined 3.644 s |
| pass 1 U occupancy probe | ×1 0.124 s / ×2 0.178 s (1.44×) / ×8 0.491 s (3.96×) |

Pass-1 sublinear scaling (×8 costs 3.96×) ⇒ occupancy-bound at 36,752 targets — the
0.124 s (above the 0.02–0.04 estimate band) is launch-config/occupancy headroom, not
arithmetic. Pass-2 U-only cost is genuinely additive across the source and ring
components (no fusion loss); the U+J ring component (2.13 s) dominates the 3.64 s
U+J total, but production runs U-only (BODY_HESSIAN_TO_PARTICLES=false), so 2.02 s
stands as the pass-2 figure for the B′ stack.

- **cpu-fmm vs exact: 1.099e-10** (worst rel 5.9e-7) — the production CPU path is
  effectively EXACT here (radius inflation ⇒ ~all-direct; residual farfield clean at
  P=8). NOT an fmm-tuning issue (P=8/MAC 0.4/leaf 20 recorded).
- **seam(:cuda) vs exact: 1.235e-02** = the parity number. THE GPU SEAM ARM IS THE
  DEFECT. Deterministic across jobs; offenders mirror EXACTLY across blades
  (17284 ↔ 35660 = +18376) — geometry/index-band correlated.

**Local elimination (no GPU locally, all archived in scratchpad logs + repro scripts):**
p018 rotor body rebuilt from the local mesh (driver construction, R=0.119,
core_size_panel=R·1e-10, deterministic strengths; influence is linear in strength so
restart state is unnecessary): (a) noshedding body 36752 panels — host-seam vs exact
**4.6e-18**, cpu-fmm vs exact 2.8e-13; (b) WITH real TE shedding (40+40 edges, Das
nonzero → attached-wake columns packed like the failing config) — host-seam vs exact
**1.4e-15**. Host functor math and packing are EXONERATED on the real geometry;
remaining suspect: the CUDA arm (device kernel at scale/on-surface branches, or the
seam's :cuda orchestration). Code audit of `_cuda_rect_panels_kernel!` found no
tile/stride bug; the old device parity sweeps only used WELL-SEPARATED targets
(tgt 2 units off the cloud) at small sizes — the on-surface near-singular branch
regime (`_rect_is_self_pair`, edge PV limits, solid-angle sign under device FMA
contraction) was never device-tested.

**Instrumentation added for the next Job B run (all local, parse/bash -n/mini-clean):**
1. Harness attribution now adds a **host-seam arm** when SEAM_MODE=:cuda:
   `host_seam_vs_exact` and `cuda_vs_host_seam` splits (informational gates) — pins
   functor-math vs device-numerics on the exact failing config; plus offender
   control-point coordinates printed for the worst seam-vs-exact targets.
2. NEW rect device testset "on-surface p018 scale": 36,290 tag-4 wavy-sheet panels,
   targets = panel centroids (exact self pairs) + deterministic 1e-8 near-plane
   nudges, device-vs-host per-target rel ≤ 1e-11/1e-10. Runs as stage 0c of
   `fm051_parity.sh` (failure is non-fatal, logged as a FINDING). If it FAILS →
   defect reproduces in pure FastMultipole; if it PASSES → seam :cuda orchestration.
3. Mini harness revalidated after wiring (OVERALL PASS, host arm correctly skipped
   in mini); rect host tests 57/57 locally (CUDA layer skips off-GPU).

Next: user resubmits Job B; read `host_seam_vs_exact` / `cuda_vs_host_seam` /
stage-0c to localize, then fix the CUDA defect. Do NOT re-aim any gate until the
CUDA arm matches exact.

## Worklog — 2026-08-22 (session 9): Job B 13309929 localizes to the device kernel; root cause found and FIXED locally

**Job B 13309929** (user-submitted, exit 1 by design): stage 0c FAILED — the new
on-surface p018-scale device testset reproduced the defect in pure FastMultipole
(velocity rows per-target relerr 7.05 vs 1e-11; gradient rows PASSED). Attribution
splits: `host_seam_vs_exact` 2.155e-14 (host functor exonerated),
`cuda_vs_host_seam` 1.235e-2 == the full pass-3 deviation. Verdict per the
session-8 localization logic: **pure FastMultipole CUDA device-kernel defect**, not
seam `:cuda` orchestration. Artifacts: `data/panel_particle_gpu_coupling/
{slurm-fp-051-parity-13309929.out,.err, fm051_pass_parity_full_13309929.csv,
fm051_profile_results_13309929.csv}`.

**Root cause (probe-confirmed, `scratchpad/probe_fragile.jl` logic):** in the
solid-angle term `atan(num, den)` (`_rect_solid_angle_tan` / FLOWPanel
`compute_source_dipole`), `num ∝ tRz` and `den < 0` on 2 of 3 edges near a
centroid, so when the target sits ON the panel plane and `tRz` is roundoff junk
(~1e-17), the SIGN of the junk picks a ±2π solid-angle side (∓σ/2·n̂ velocity).
Probe: all 36,290 test panels jump by 4π between tRz = ±1e-17; 4877/7258 sampled
self targets have `tc ≠ 0` (test builds cps with `/3`, functor uses
`*0.3333333333333333`). Host CPU and FLOWPanel exact agree bit-for-bit (identical
CPU arithmetic ⇒ same junk ⇒ same side); the device's NVPTX FMA contraction
produces different junk ⇒ flips ~half the on-plane pairs (self pairs + coplanar
strip neighbors ⇒ nearly all 36,708 production targets over-gate). Explains every
symptom: velocity-only (gradient never uses tan_term), O(σ) magnitude, on-surface
only, offenders mirrored across blades.

**Fix (both repos, mirrored):** snap `tRz` to an exact zero when
`tRz² ≤ 1e-24·L²`, `L² = Σᵢ|vᵢ − centroid|²`, so the existing "PV on-plane"
guard fires deterministically regardless of FMA junk sign; guard first clause
relaxed from `(tRx==0 && tRy==0 && tRz==0)` to `tRz == 0` (per-panel uniform ⇒
no partial edge-cancellation). Junk (≲1e-17·L) vs genuine geometry (≳1e-8·L)
sit ~7 decades either side of the threshold.
- FastMultipole `src/direct_rectangular.jl`: snap in `_rect_tri_source_doublet`
  (host + device share it), guard in `_rect_solid_angle_tan`.
- FLOWPanel `src/FLOWPanel_elements_fmm.jl`: `_onplane_snap` helper (near
  `SELF_PAIR_EPS_REL`) + snap in `_induced` (covers body + attached-wake panels),
  guard in all three `compute_source_dipole` methods. The snap applies to
  Float32/Float64 ONLY: snapping ForwardDiff duals erases partials
  (`zero(Dual)`) and broke 5 AD kernel-gradient tests (∇φ ≡ 0, rel err 5.25e15);
  duals pass through and keep the smooth-branch derivative (pre-fix semantics).

**Local validation (all green):** FastMultipole `test/direct_rectangular_test.jl`
31/31 (CPU; device layer skips locally); `repro_p018_body2.jl` host-seam vs exact
diff/scale 1.438e-15 (unchanged ⇒ host/exact bitwise agreement preserved);
FLOWPanel full suite `julia --threads=4 test/runtests.jl` 4862/4862.

Next: user resubmits Job B (`bash benchmark/slurm/fm051_parity_submit.sh`).
Expected: stage 0c PASSES; `cuda_vs_host_seam` ≤ 1e-10; pass-3
`phase_gate_1e-3` PASSES. If residual deviation remains, it is a second,
now-unmasked device defect — iterate before touching any gate. Then Stages 3–4.

### Session 9 closure — Job B 13310123 (post-fix): ALL GATES PASS

Stage 0c: CUDA device parity **100/100** (was 99/1). Attribution splits:
`host_seam_vs_exact` 2.172e-14, `cuda_vs_host_seam` **3.584e-14** (was 1.235e-2),
`seam_vs_exact` **3.586e-14** ≤ 1e-10. Pass-3 `phase_gate_1e-3` **1.825e-5**
(was 1.235e-2); pertarget_rel 9.964e-5. Harness verdict: "seam matches exact
direct; the parity-stage deviation is the CPU arm's fmm truncation error"
(`cpu_fmm_vs_exact` 1.099e-10, informational). **OVERALL: PASS** (exit 0).
Timings unchanged (pass-1 U F64 0.1241 s / F32 0.0714 s; pass-2 U F64 2.026 s).
Artifacts: `data/panel_particle_gpu_coupling/{slurm-fp-051-parity-13310123.out,
.err, fm051_pass_parity_full_13310123.csv, fm051_profile_results_13310123.csv}`.
051 parity mechanics are CLOSED; Stages 3–4 next.

### Session 9 — Stage 3 staged: Job C (CPU solve pricing) authored

Production p018 body solver confirmed as `Backslash` (dense, factored once at
RHPC setup — the rotor is rigid in its rotating frame;
`examples/rotor_hover_pressure_comparison.jl:1027,1074`), so 051-plan Stage 3's
Krylov niter×matvec framing reduces to: WARM per-step `solve_formulation!` wall
(RHS build + backsolve) vs the ~0.783 s U-only headroom
(3.3 − [0.373 + 0.124 + 2.02]). Job B's 54.9 s solve wall was a cold first call.
Hessian flags confirmed from the 13310123 config dump:
`BODY_HESSIAN_TO_PARTICLES=false`, `PANEL_WAKE_HESSIAN_TO_PARTICLES=false` ⇒
U-only pass-2 (2.02 s) applies, NOT the 3.64 s U+J.

New files (uncommitted, intentional):
- `benchmark/fm051_solve_pricing.jl` — reuses the parity harness's full-mode
  p018 warm start by including `fm051_pass_parity.jl` (its main() guard keeps it
  inert when included from a script; verified locally — but NOT under `julia -e`,
  whose empty PROGRAM_FILE fires the `|| isempty` arm). Times cold + 5 warm
  `solve_formulation!` calls, seam :off, prints PRICE/VERDICT vs headroom,
  writes `benchmark/results/fm051_solve_pricing.csv`.
- `benchmark/slurm/fm051_solve_pricing.sh` — CPU-only sbatch (64 cpus, 192G,
  3 h, NO GPU request ⇒ skips the H200 queue), env block copied verbatim from
  `fm051_parity.sh` minus CUDA bits. Submit from ~/FLOWPanel-046 on the cluster
  (tree already synced by the last fm051_parity_submit.sh run; re-sync only if
  local edits must ride along — the pricing files DO need one re-sync).

Decision rule unchanged (from 050): fits ⇒ CPU 64-thread solve stands, device
matvec recorded as 052+ option; doesn't fit ⇒ device-resident dense matvec.

### Session 9 — Job C 13310250 (solve pricing): CPU solve does NOT fit; decomposition staged

Warm `solve_formulation!` median **7.270 s** (min 7.262, max 7.450 over 5;
cold 25.3 s) vs 0.783 s U-only headroom ⇒ **VERDICT: does not fit** as measured.
BUT the number is suspicious: p018's solver is `Backslash` and the inner
Neumann `_solve!` is RHS + `ldiv!` (~ms at 36,752); the 7.27 s must live
upstream in the `solve!` chain (Kutta/TE? dispatch-level influence assembly?
an unexpected per-step `update_G` LU refactor ≈ 8 s at this size?). Before
escalating to a device matvec per the Stage-3 decision rule, the pricing script
now decomposes: per-body inner `solve!` timings + a `Profile.@profile` flat
profile of one warm solve (Job C rerun needed). Sbatch mem now 64G (user-edited before the
13310250 submission; job succeeded — dense G ~10.8 GB + LU ~2×, peak ~30-40 GB).
Artifacts: `data/panel_particle_gpu_coupling/{slurm-fp-051-price-13310250.out,
.err, fm051_solve_pricing_13310250.csv}`.

### Session 9 — Job C rerun 13310370: 7.3 s solve wall DECOMPOSED — it is the Dirichlet potential self-influence, NOT the backsolve

Warm solves 7.29–11.43 s (median 8.32; node noisier than 13310250). Decomposition:
**inner `solve!(body1)` alone = 7.40 s / 7.25 s repeat** — the whole cost.
Flat profile of one warm solve (139,464 snapshots, 91% util): ~all samples in
FastMultipole nearfield direct under `execute_assignment!` (fmm.jl:125/186
threadsfor) → FLOWPanel `direct!`/`_direct_body!` (FLOWPanel_abstractbody.jl:
1309/1326) → `induced` (FLOWPanel_elements_fmm.jl:243/255). `ldiv!` invisible.

**Why:** p018's body is `RigidWakeBody{Union{ConstantSource,VortexRing}, 2,
Float64, true}` — **DBC=true (Dirichlet)**, so `solve!` (FLOWPanel_solver.jl:238)
runs `influence!(body, body, backend; scalar_potential=true, velocity=false)`
each solve to fill the interior-potential workspace before `_solve!`'s
RHS+`ldiv!` (ms). That per-solve 36,752² potential self-influence via CPU fmm
IS the 7.3 s. Earlier session-9 assumption of the Neumann branch was wrong.

**Stage-3 escalation is therefore NOT a Krylov device matvec.** The fixed-
geometry (rotating-frame-rigid) potential self-influence is a constant matrix
Φ: candidate fixes, cheapest first —
1. Assemble Φ once at setup (like G) and apply as a dense mat-vec per solve:
   10.8 GB, memory-bound gemv ~50–100 ms on CPU (fits 0.783 s headroom with
   room to spare); device-resident on the H200 (~ms) only if CPU gemv
   disappoints. Transform under rigid motion like G (transform_solver_geometry!).
2. Route it through the GPU seam — requires adding scalar-potential output to
   the rect functor/device kernel (currently VS/GS only, "the 018 passes never
   request the potential") — larger scope, only if (1) is blocked.
Artifacts: `data/panel_particle_gpu_coupling/{slurm-fp-051-price-13310370.out,
.err, fm051_solve_pricing_13310370.csv}`.

## Worklog — 2026-08-23 (session 10): Stage-3 fix IMPLEMENTED (opt-in S matrix + gemv seam); local gates green; Job C rerun pending

User ratified the Φ-matrix direction with two constraints: **opt-in** (no
default memory change for existing `Backslash` users) and **architecture kept
open** for a future matrix-free/GPU source-influence path (problems too large
to hold S; e.g. scalar-potential output on the rect GPU seam).

**Implementation (FLOWPanel, uncommitted, intentional — preserve):**
- `src/FLOWPanel_solver.jl`:
  - `_G!` gains kwarg `kernel_and_strength_index` (default
    `_G_kernel_and_strength_index(source_system)` — historical behavior
    bit-unchanged). Passing `(ConstantSource, 1)` assembles the
    source-potential matrix S instead of the solve operator, reusing the
    existing operator-mode guard, `induced` self-limits, and threading.
  - `Backslash` gains field `S::Union{Nothing,Matrix{TF}}` (default
    `nothing`) + constructor kwarg `assemble_source_potential=false`; new
    `assemble_source_potential!(solver, body)` builds/refreshes S post-hoc
    (this is how the p018 config opts in without constructor plumbing).
  - NEW seam `_source_influence!(body, solver, backend)`: the Dirichlet
    `solve!` per-solve source self-influence now routes through it. Default
    method = the old `influence!(body, body; scalar_potential=true,
    velocity=false)`. `Backslash`-with-S method = `LA.mul!(body.potential,
    S, σ, 1, 1)` dense gemv, guarded to fall back to the backend when S is
    absent OR the affine wake correction is active (gemv is only the linear
    map S·σ in operator mode; p018/VelocityThroughSources never activates
    the correction, so production takes the gemv). **Future matrix-free/GPU
    backends plug in here by dispatching on backend/solver — `solve!` needs
    no further changes.**
  - `_solve!(…, ::Backslash; update_G=true)` refreshes S alongside G
    (geometry-locked pair).
  - Rigid motion: S is invariant (scalar potential of co-moving panels at
    co-moving control points) — same no-op `transform_solver_geometry!`
    treatment as G. Nothing to transform.
- `test/runtests_unit_solver.jl`: NEW testset "Backslash source-potential
  matrix S (051 Stage 3)" — S·σ vs DirectBackend `influence!` ≤1e-12 rel,
  end-to-end gemv-solve vs influence!-solve μ ≤1e-12 rel, update_G refresh,
  post-hoc attach. 7/7 pass.
- `benchmark/fm051_solve_pricing.jl`: new S stage (default on;
  `FM051_PRICE_S=0` skips): times one-time `assemble_source_potential!` on
  the p018 solver, checks gemv-vs-fmm potential equivalence on the
  production state (fmm-truncation-level diff expected, ~1e-10 rel), times
  the bare gemv, re-prices N_WARM warm `solve_formulation!` calls on the
  gemv path, and bases the VERDICT on the S-path median vs the 0.783 s
  headroom. CSV gains `warm_S_*`, `S_assemble_s`, `S_gemv_s`,
  `S_equiv_max_*`. NOTE: S adds ~10.8 GB on top of G's ~10.8 GB — the 64G
  sbatch still fits.

**Local gates (all green, 2026-08-23):**
- `test/runtests_unit_solver.jl`: 387/387 (new S testset 7/7).
- Full FLOWPanel suite (`julia --project=. --threads=4 test/runtests.jl`):
  4947/4947, exit 0. (Count grew vs session-9's 4862: the tree gained
  commit 62d72db + the new S tests since that measurement.)
- Mini parity baseline (seam :host, S not opted in): OVERALL PASS, 12
  gates; pass 1 6.8e-15, pass 3 1.44e-14 — same order as the recorded
  baseline (5.1e-15 / 2.2e-14); small shifts consistent with the tree
  having moved (62d72db) since the baseline was recorded.

**Next:** user resubmits Job C (sync `benchmark/` to ~/FLOWPanel-046, then
`sbatch benchmark/slurm/fm051_solve_pricing.sh`). Expected: equivalence
~1e-10 rel, warm S-path solve well under 0.783 s ⇒ Stage 3 CLOSES; then
Stage 4 (timing table vs 3.3 s, doc closure, commits — user approves).

### Session 10 — Job C rerun 13390264 FAILED (stale cluster tree); resynced, resubmission pending

Job 13390264 (fp-051-price, 5m23s, exit 1, MaxRSS 12.5 GB): the fmm-path
baseline REPRODUCED (warm 7.28–7.42 s over 5; inner `solve!(body1)` 7.27 s;
same `induced`-dominated profile), then the S stage crashed with
`UndefVarError: assemble_source_potential!` — only `benchmark/` had been
resynced to ~/FLOWPanel-046; the S implementation lives in
`src/FLOWPanel_solver.jl`, which was stale on the cluster. NOT a code
defect. Fixed by rsyncing `src/` (FLOWPanel_solver.jl + drifted
FLOWPanel_formulation.jl/FLOWPanel_metadata.jl) and
`test/runtests_unit_solver.jl` to orc; remote tree verified to contain
`assemble_source_potential!`/`_source_influence!` and the S-stage script.
Artifacts: `data/panel_particle_gpu_coupling/slurm-fp-051-price-13390264.{out,err}`
+ `sha256_13390264.txt`. User resubmits
`sbatch benchmark/slurm/fm051_solve_pricing.sh` from ~/FLOWPanel-046.

### Session 10 — Job C 13391706 COMPLETED: S path FITS the headroom (0.758 s vs 0.783 s); exact-0 equivalence root-caused (self-influence is all-direct)

Job 13391706 (fp-051-price, 8m32s, exit 0). Numbers (5 warm each, 64 threads):

| metric | value |
|---|---|
| fmm-path warm median | 7.958 s (min 7.304, max 12.181; noisier node) |
| S assembly (one-time) | 118.9 s, 10.81 GB |
| bare gemv | 0.453 s |
| S-path warm solves | 1.178, 0.864, 0.738, 0.743, 0.758 s |
| **S-path warm median** | **0.758 s** vs 0.783 s headroom ⇒ **VERDICT: FITS** |
| gemv-vs-fmm equivalence | max abs 0.0, rel 0.0 (bitwise) |

**Margin is thin: 25 ms (3%).** The gemv itself is 0.453 s (10.81 GB ⇒
~24 GB/s effective — far below node bandwidth; BLAS thread/NUMA tuning,
Float32 S (5.4 GB), or a device-resident gemv are all easy 2×+ levers if
the Stage-4 table needs slack). The residual ~0.3 s is the rest of
`solve_formulation!` (previously hidden under the 7.3 s influence).

**Exact-0 equivalence is GENUINE, root-caused by local probe**
(`scratchpad fm051_equiv_probe.jl`, diamond nspan=120 = 960 panels, the
exact p018 solve backend FastMultipoleBackend(8, 0.4, 20) from
`p018_cs_f1_l3p4.metadata.toml`): the body self-influence via
FastMultipoleBackend is **bitwise identical to DirectBackend for BOTH
potential and velocity** (960/960 and 2880/2880) — the interaction lists
accept no farfield for the body–body self pair (thin shrunken boxes never
meet MAC 0.4), so `fmm!` runs 100% direct. Hence: (a) the 7.3 s per-solve
cost was a true O(N²) dense evaluation all along (matches the
`induced`-only profile), (b) `expansion_order` is irrelevant to this call,
and (c) S·σ replaces an exactly-dense evaluation — a lossless drop-in
(local gemv-vs-direct 5.8e-15 rel; cluster bitwise 0). NOT vacuous: the
harness state carries nonzero σ (same build path whose parity gates passed
with nonzero solve outputs); the pricing script now also prints
|σ|/|φ| norms with a `[VACUOUS]` flag for any future run (patched +
synced to orc).

Artifacts: `data/panel_particle_gpu_coupling/{slurm-fp-051-price-13391706.out,
.err, fm051_solve_pricing_13391706.csv}` + `sha256_13391706.txt`.

**Stage 3 measurement CLOSED pending user sign-off on the thin margin.**
Next: Stage 4 — timing table vs 3.3 s budget (B′ stack + 0.758 s solve),
doc closure, commits (user approves).

### Session 10 — gemv diagnostics staged for the next Job C run; qos=test dropped

Follow-up to 13391706's 0.453 s bare gemv (~24 GB/s ≈ single-core DRAM
bandwidth; a memory-bound 10.81 GB apply should hit tens of ms at real node
bandwidth, and true milliseconds needs HBM = the 052+ device option):

- `benchmark/fm051_solve_pricing.jl` gains a GEMV DIAGNOSTICS block after
  the S-path warm solves: prints BLAS thread count (set independently of
  `--threads`; suspected under-threading), sweeps dgemv at BLAS threads
  {1,8,16,32,64} with GB/s, times a Julia-threads row-blocked gemv
  (BLAS=1 inside; also probes NUMA first-touch), then re-times the 5 warm
  solves at the best BLAS setting (`warm_S_tuned_*`; the solve! seam calls
  plain `mul!`, so `set_num_threads` is the only knob needed). PRICE
  verdict now uses min(median, tuned median). CSV gains
  `gemv_blas*_s`, `gemv_blocked_s`, `blas_best_nt`, `warm_S_tuned_*`.
  Parse-checked; constructs smoke-tested at toy scale locally.
- `benchmark/slurm/fm051_solve_pricing.sh`: `--qos=test` DROPPED (user
  directive — test-QoS nodes suspected for the poor bandwidth and the
  noisy 7.3–12.2 s fmm-path timings); time bumped to 45 min. Verified no
  active `#SBATCH --qos` remains on the synced cluster copy.
- Both files synced to ~/FLOWPanel-046. User submits:
  `sbatch benchmark/slurm/fm051_solve_pricing.sh` from ~/FLOWPanel-046.

### Session 11 (2026-08-24) — Job C rerun 13395348 COMPLETED: tuned S path 0.607 s vs 0.783 s headroom (23% margin); BLAS under-threading confirmed

Job 13395348 (fp-051-price, no `--qos=test`, node m12-4-18, 7m08s, exit 0).
Numbers (5 warm each, 64 Julia threads):

| metric | 13391706 | **13395348** |
|---|---|---|
| fmm-path warm median | 7.958 s | 7.300 s (min 7.260, max 8.585 — quieter node) |
| S assembly (one-time) | 118.9 s | 111.8 s, 10.81 GB |
| bare gemv (default BLAS=64) | 0.453 s | 0.302 s |
| S-path warm median | 0.758 s | 0.659 s (spread 0.656–0.662, very tight) |
| **S-path tuned (BLAS=8) median** | — | **0.607 s** (min 0.591) |
| headroom (U-only) | 0.783 s | 0.783 s |
| **margin** | 25 ms (3%) | **176 ms (23%)** ⇒ **VERDICT: FITS** |

GEMV DIAGNOSTICS sweep (dgemv on the 10.81 GB S):

| BLAS threads | time | effective BW |
|---|---|---|
| 1 | 0.419 s | 26 GB/s |
| **8** | **0.234 s** | **46 GB/s** |
| 16 | 0.238 s | 45 GB/s |
| 32 | 0.252 s | 43 GB/s |
| 64 | 0.286 s | 38 GB/s |
| row-blocked (64 Julia thr, BLAS=1) | 0.282 s | 38 GB/s (max dev vs dgemv 0.0) |

Diagnosis: 13391706's 0.453 s was a qos=test-node + BLAS-oversubscription
artifact, not a code problem. dgemv saturates at **BLAS=8 (46 GB/s)**;
more threads regress (NUMA/contention). `blas_best_nt=8` re-timing gives
the production number: **tuned warm S-solve median 0.607 s**. Residual
~0.37 s above the 0.234 s gemv is the rest of `solve_formulation!`
(RHS build + LU backsolve). Float32 S and device gemv remain unexercised
2×+ levers (052+).

**Correction to the 13391706 vacuity claim**: the new `|σ|` norm print
shows `|sigma| = 0` at the check point — the **on-cluster** gemv-vs-fmm
"bitwise 0" comparisons (both runs) were **vacuous** (the harness state
at the seam carries σ=0 before the timed solves rewrite it). The
equivalence claim does NOT rest on them: the local probe
(`fm051_equiv_probe.jl`, nonzero σ, exact p018 backend
FastMultipoleBackend(8,0.4,20)) stands — self-influence is all-direct,
bitwise == DirectBackend for φ and u, so S·σ is a lossless replacement
of a genuinely dense O(N²) evaluation. The session-10 sentence "NOT
vacuous: the harness state carries nonzero σ" was wrong for the cluster
harness and is superseded by this note.

Artifacts: `data/panel_particle_gpu_coupling/{slurm-fp-051-price-13395348.out,
.err, fm051_solve_pricing_13395348.csv}` + `sha256_13395348.txt`.

**Stage 3 CLOSED.**

### Stage 4 — final timing table vs the 3.3 s budget: B′ stack CLOSES at 3.12 s (176 ms margin)

Production step budget: **3.3 s** (RK3 step target). B′ stack, best
measured numbers (production config `BODY_HESSIAN_TO_PARTICLES=false`,
U-only pass 2):

| component | time | source |
|---|---|---|
| particle worst-case (D15 upload residency) | 0.373 s | Job A |
| pass 1 | 0.124 s | Job A |
| pass 2 (U-only) | 2.020 s | Job A |
| body solve (S gemv path, tuned BLAS=8) | 0.607 s | Job C 13395348 |
| **total** | **3.124 s** | |
| **margin vs 3.3 s** | **0.176 s (5.3%)** | |

One-time costs outside the per-step budget: S assembly 111.8 s + 10.81 GB
resident (amortized over the run; re-assembled only if the body mesh
changes). Known levers if the margin erodes: Float32 S (5.4 GB, ~2×
gemv), device-resident gemv (HBM ⇒ ~10 ms), pass-2 kernel work (052+).

**051 verdict: the coupled panel–particle GPU step FITS the 3.3 s budget
with the CPU S-gemv solve. Device-resident matvec deferred to 052+.**
