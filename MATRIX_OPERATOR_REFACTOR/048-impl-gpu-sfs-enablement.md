# 048 Impl: GPU SFS Enablement (fused ζ pass)

## Status and entry gate

**Remediated locally `2026-08-21`; incomplete and unapproved.** The SFS
mechanism is implemented and the conservative-candidate host physics matrix
passes. The corrected per-call CUDA launch, device contracts, and timing/p018
arms remain unrun because job 13294119 stopped after the host matrix.

Entry gate: `046` complete and approved (all work lands on the unified
branches). Runs in parallel with `047` — both block only on `046`. FLOWVPM
CLAUDE.md constraints apply on that repo.

## Motivation

The 023 profiling of an 018 production step (2026-08-20) found **~75% of a
production step is the Dynamic-SFS estimator `Estr_fmm!` near-field walk** —
the single largest lever in the Production Integration Phase. Today
`sfs=true` is a **hard error** on the GPU radix path
(`FLOWVPM_fmm_radix.jl:499-500`). `041k` built the stack's first fused
direct+SFS kernel (all-pairs) and measured fused SFS at only **+26–31% over
a U/J pass** — evidence that a production U-list ζ pass can be cheap.

## Objective

Device SFS: the fused ζ pass (the `041b` §1.2 factorized identity — Ω/Q
precompute; E = T_p(Ω) − Q) implemented in the radix nearfield lifecycle and
the FLOWVPM adapter, with the `sfs=true` hard error removed, parity gates
passed, and the kernel written **lever-ready** for the `054` optimization
port.

## Method

### Stage 1 — kernel design

- ζ needs completed Jacobians → the SFS pass **cannot merge into the U/J
  pass**; minimal structure = 2 pair passes + an O(N) T_q(Γ_q) precompute
  (041k finding). Self-pair cancels exactly; ζ is skippable beyond the
  saturation cutoff.
- Production shape: Estr over the **U-list** (not all-pairs) with atomic (or
  target-owned, per `041e`) accumulation — the open cost question vs 041k's
  all-pairs number. **Measure it**; do not assume the +26–31% carries over.
- **Lever-ready requirement:** reuse the existing singular/regularized
  partition of the nearfield so the `041k` opt levers (far-field singular
  switch, fast transcendentals, register blocking — row `054`) drop in
  without restructuring.

### Stage 2 — lifecycle + adapter integration

- Wire the ζ pass into the radix nearfield lifecycle (after U/J completes;
  respect graph capture, capacity sizing, counters, zero per-step
  allocation).
- FLOWVPM adapter: remove the `sfs=true` hard error; route SFS output to
  `SFS_INDEX=40:42`; keep the gaussianerf-only check; physics transcribed
  from `FLOWVPM_subfilterscale_models.jl:16-41` (Estr, transposed) with the
  J layout col-major du_i/dx_j at J[(j-1)*3+i].

### Stage 3 — tests + measurement

- Parity vs CPU `Estr_direct!` / `Estr_fmm!` at 1e-3 (F64 tighter), at P=4
  AND P=8, both precisions (standing P=4 rule).
- Counters (`body_uploads==0`, `expansion_host_copies==0`) and
  zero-allocation contracts hold with SFS on.
- Extend FLOWVPM `runtests_gpu_fmm.jl` (Part A host + Part B device) with
  SFS testsets.
- H200 measurement: marginal SFS cost over U/J on the U-list at a realistic
  operating point (feeds the `049` per-pass budget table).

## Gates and verdict

- `sfs=true` works on the GPU path with the parity, counter, and
  zero-allocation gates green.
- Measured U-list SFS marginal cost reported (vs the 041k all-pairs +26–31%
  anchor).
- Verdict states whether the atomics/accumulation strategy on the U-list is
  satisfactory or names the follow-up lever.

## Artifacts

- Source changes on the unified FastMultipole + FLOWVPM branches (kernel,
  lifecycle wiring, adapter, tests).
- Measurement CSV + short results section appended to this doc.

## Verification

- FLOWVPM `runtests_gpu_fmm.jl` green incl. new SFS testsets (device parts
  under `FASTMULTIPOLE_REQUIRE_CUDA_TESTS=1` on the cluster).
- FastMultipole suite green; no regression in non-SFS paths (same-job A/B).

## Recorded context (2026-08-20 staging)

**041k evidence** (`data/direct_bruteforce_ceiling/`): fused direct+SFS via
the 041b identity costs only +26–31% over U/J on all-pairs; SFS needs
completed Jacobians → ζ pass cannot merge into the U/J pass; minimal
structure = 2 pair passes + O(N) T_q(Γ_q) precompute; self-pair cancels
exactly; ζ skippable beyond saturation cutoff. Opt levers (far-field
singular switch ρ²>42.25 F32 / 81 F64, `__nv_fast_expf`/`__nv_erff`,
2-target register blocking) give 1.6–1.7× F32 → 3.3e11 pairs/s ≈ 38% FP32
FMA peak; F64 opt inert below n≈3e4 (9σ cutoff spans domain).

**018 motivation** (`BRAINSTORM/023_018_runtime_cost_profiling.md`):
170–230 s/step on 64 cores; ~75% of a production step is `Estr_fmm!`; wake
FMM velocity itself only ~7 s; per-step cost ~linear in particle count
(~50–100 s per 100k).

**FLOWVPM adapter state (034):** `FLOWVPM_fmm_radix.jl` (520 lines) +
`ext/FLOWVPMCUDAExt.jl` (980 lines: `gpu_direct!:642`,
`gpu_zeta_direct!:725`, `gpu_estr_direct!:835`, `warmup_gpu:864`, device
pack/unpack `source_to_buffer!:920`/`buffer_to_target!:944`). Hard errors:
rbf (`:497`), **sfs (`:499-500`)**, autotune flags. Traits Point{Vortex},
residency from `particles isa Array`, `RegularizedVortex(sigma_row=8)`,
gaussianerf-only hard check (`:62,:260`); one lazily-built `RadixFMMCache`
per pfield (capacity = maxparticles, hessian=true); zero per-step body
H2D/D2H.

**Physics transcription provenance:** `FLOWVPM_fmm.jl:132-198` (U/J pair
math), `FLOWVPM_kernel.jl:51-57` (gaussianerf),
`FLOWVPM_subfilterscale_models.jl:16-41` (Estr, transposed). J layout
col-major du_i/dx_j at J[(j-1)*3+i]; U_INDEX=10:12, J_INDEX=16:24,
SFS_INDEX=40:42 (`FLOWVPM_particlefield.jl:287-344`).

**Contracts (integration-api-spec.md):** counters `body_uploads=0` /
`expansion_host_copies=0`; zero per-step alloc; full 9-component hessian;
out-of-box throws; explicit `recenter!` only.

**Nearfield state P2P builds on:** partitioned singular/regularized +
`037f` `:fp32` gh-mode default + `041e` target-owned CSR (REGIME-ONLY, off
by default).

## Remediation review and current verdict (2026-08-21)

Clear-context review found four material gaps in the original completion
claim:

1. FLOWVPM built every radix vortex cache with SFS storage, and both CUDA
   lifecycle bodies launched TG + the full ζ pair pass unconditionally.
   `sfs=false` gated only delivery. Consequently the non-SFS regression was
   not a non-SFS run, and the recorded 049 difference (0.273957 ms) measured
   finalize/delivery overhead between two arms that both executed ζ; it is
   **not a valid marginal SFS cost**.
2. The required delivered result vs CPU `Estr_direct!` / `Estr_fmm!` at the
   stated tolerance was replaced by mechanical parity against the same
   approximate radix J and a J-scaled gate. Mechanical parity is useful, but
   it does not satisfy the physics gate.
3. CPU Estr excludes static bodies as both sources and targets. The radix ζ
   pass included them.
4. Test coverage did not contain the full P=4/P=8 × Float32/Float64 matrix,
   and the item had no self-contained result artifact.

The remediation makes SFS execution genuinely per-call: U/J (including graph
replay) finishes first, then `_launch_cuda_sfs!` runs only when the evaluation
requests `sfs=true`. Thus an SFS-capable cache incurs no TG/ζ work on
`sfs=false`. SFS is no longer embedded in the U/J CUDA graph; it remains
stream-ordered, persistent-buffer, and allocation-free by construction, but
that corrected CUDA path needs a future device run before those properties can
be re-certified empirically.

FLOWVPM now packs row 9 as a non-static mask and constructs the cache with
`sfs_active_row=9`. Host and CUDA ζ loops skip inactive sources and targets,
matching both CPU Estr implementations. The generic FastMultipole default is
`sfs_active_row=0` (all bodies active), so other consumers do not acquire
FLOWVPM-specific semantics.

### Local verification

Focused host tests passed:

- FastMultipole `test/device_system_interface_test.jl`: SFS 20/20; all
  surrounding interface testsets green (including 32,367 binned-nearfield
  assertions).
- H200 job 13294119 completed the full host candidate matrix before an
  informative early stop: all eight `Estr_direct!` and all eight `Estr_fmm!`
  comparisons passed. The stop was solely the old `@test_broken` declarations
  reporting “Unexpected Pass”; they are now ordinary `@test` assertions.
- Mechanical relative RMS ranges from `2.6393e-9` to `2.6397e-9` F64 and
  `4.015e-7` to `4.088e-7` F32.
- Delivered `Estr_direct!` relative RMS ranges from `9.18434e-5` to
  `4.09298e-4` F64 and `9.20317e-5` to `4.09459e-4` F32. These satisfy the
  user-derived conservative gates: `5e-4` F64 (the tighter epsilon/2 tail
  budget) and `1e-3` F32. `Estr_fmm!` agrees at the recorded precision and
  passes the same gates.

Detailed rows, provenance, and the invalidated pre-remediation device records
are in `data/gpu_sfs_enablement/048_results.csv`.

The pending device test now compares **increments**, not absolute SFS rows:
row-9-masked host and all-pairs oracles are evaluated only on active columns,
static target increments must be exactly zero, and an all-active source oracle
must differ from the masked oracle (explicit source-skip sensitivity). It also
requires zero device allocation, an absolute 4096-byte host bookkeeping cap,
and deterministic graph eligibility/executable/epoch identity across a
same-state replay. `scripts/fm048_ab_benchmark.jl`, driven by
`scripts/cuda_048_run.sh`, defines synchronized warmed median A/B measurements
for P=4/P=8 × F32/F64 and emits hashed raw-log/CSV/tree provenance per
`data/gpu_sfs_enablement/PROVENANCE_TEMPLATE.md`. This harness has been parsed
and shell-validated locally but has not been run.

Per user decision, the accuracy and pending timing matrix evaluates two
conservative per-pair partition cutoffs without changing the shipped default:
`rho_t=4.211` (velocity per-pair 1e-3 candidate) and `rho_t=4.789`
(Jacobian per-pair candidate). The synthetic cube covers both candidates over
P=4/P=8 × F32/F64; the real p018 210,056-particle production arm uses its
production P=4/F64 settings with both candidates. This avoids duplicating the
large production run across non-production precision/order combinations while
still separating cutoff, order, and precision effects.

### Device-run failure analysis and resolution (2026-08-22)

Job 13298230 (193/240, 47 failures) decomposed into three classes, all
resolved without weakening delivered-accuracy evidence:

1. **"Replay J defect" (j_rel 0.1098 cube / 0.092 wake on warmed calls) —
   a measurement artifact, NOT a graph defect.** Two dedicated H200
   diagnostics: job 13299959 (call-by-call slab comparison) showed warm,
   record, replay, graph-off, and sfs call shapes all deliver identical
   accuracy (j 1.9087e-3 to 15 digits; slab diffs were pure unstable-sort
   column shuffling; multipoles/locals agree to 1e-16). Job 13302646 (2^3
   factorial: statics x extra-capacity x @allocated wrappers, testset seed
   and call sequence) isolated the trigger to the 3 STATIC particles:
   error grows linearly per call (u += sqrt(3/20000) ~ 0.0125/call; call 8
   = 0.10984, reproducing the testset value exactly) and keeps growing
   with the graph OFF. Root cause: FLOWVPM `_reset_particles` PRESERVES
   statics' U/J while every UJ delivery (CPU `buffer_to_target_system!`
   included) accumulates into all targets; statics' U/J are consumed by
   nothing (integration/relaxation skip statics). The testset compared
   U/J including static columns between fields with differing evaluation
   counts. Fix: `fmm034_uj_errors` gains `skip=`; the SFS testset excludes
   static columns. New TRUE replay gates added (no error-scaled
   tolerance): replayed U/J within 1.5x of first-call error, and a
   replay-vs-uncaptured-body output-slab parity gate (1e-10 F64 / 1e-4
   F32) via a runtime `CUDA_GRAPH_LIFECYCLE` flip.
2. **Strict delivered-E gate failures (7) — operating-point transplant.**
   The eps/2 tail-budget gates (5e-4 F64 / 1e-3 F32) were derived at the
   host-matrix operating point (n=1500, ell=2, near_radius2=20,
   conservative rho_t) where the omitted tail is the only budgeted error;
   the device testset applied them at n=2e4/derived shell where delivered
   E is J-error-bound (D5/D7; E/J 2.05 cube / 14.1 wake) and unattainable
   at any P tested. Fix: a new "strict tail-budget operating point" device
   testset enforces 5e-4/1e-3 at the derivation regime (first AND replayed
   calls); the n=2e4 cases keep mechanical (1e-6) + J-bound gates with
   e_sfs recorded. Production accuracy/cost selection moves to
   `scripts/fm048_tuning_sweep.jl` (P x rho_t x q grid on the cube,
   Pareto frontier + baselines re-run on the p018 210k snapshot where the
   user directed the strict 5e-4 F64 gate to bind; cube-vs-p018
   discrepancy reported; run.sh stage 4b).
3. **Allocation gate failures (40) — asserted at the wrong layer.** The
   spec contract (integration-api-spec: construction-time buffers only)
   is satisfied. Measured host ~100 KB = CUDA.jl launch/broadcast
   bookkeeping across ~25-30 GPU ops/step (fixed, not n-scaled); device
   272-400 B = CUDA.jl scan/reduce library scratch at three per-step sites
   (counting-sort + body-prefix `accumulate!`, geometry-gate `maximum`);
   SFS adds zero device allocation. Fix: lifecycle-layer assertions
   (`run_cuda_radix_lifecycle!` warm: host <= 4096 B, device == 0) plus
   wrapper-layer fixed bands (160/192 KB host, <= 512 B device) with
   no-growth and sfs-adds-none checks.

### H200 run of the corrected suites — job 13302961 (2026-08-22)

Job 13302961 (submitted 11:07, corrected acceptance testsets + tuning
sweep) ran stages 1–3 to completion and **all testsets passed**; it then
**FAILED at stage 4** (23 min elapsed) on a script bug in
`fm048_ab_benchmark.jl`, before any A/B, sweep (4b), or 047 lock-check (5)
work ran. Artifacts (in `data/gpu_sfs_enablement/`):

- `vpm048-13302961.out` — sha256 `7bfc83af81f540e384851e8c6ea115118714a89b966036e23cb2e3bec874ea17`
- `fm048_device_tests_13302961.log` (stage-3 raw log) — sha256
  `ba13ec065807ec74ed65e6c6b755db506f8ee1f403b7b23f25925310373e5c59`
  (matches the in-job provenance hash).

**Stage 1–3 results vs the calibration expectations — all green:**

- Stage 1 (FM device tests): 1333 + 227 + 37 + 930 + 22 + 827 + 1248 + 21
  + 32367 + 840 + 9 + 20 (SFS host pass) + 108 (047 settings surface), all
  pass. Stage 2 (runtests_gpu): 48 + 84 pass.
- Stage 3 SFS host path 63/63; SFS device pass 300/300; strict
  tail-budget operating point 24/24; all other coupling testsets pass.
- n=2e4 device SFS (cube P4 F64 rho 4.211): u 3.71e-4, j 2.00e-3,
  e_sfs 4.35e-3, e_kernel 1.05e-15 — matches expectation (u ~3.7e-4,
  j ~2.0e-3, e_sfs ~4.3e-3, e_kernel ~1e-15).
- Strict testset (n=1500, ell=2, near_radius2=20): F64 e_first/e_warm
  9.18e-5–4.09e-4 (expected 9e-5–4.1e-4), all under the 5e-4 gate; F32
  9.20e-5–4.10e-4 under 1e-3; `graph_live = true` at every point (the
  n=1500 point really replays a captured graph).
- Allocations: wrapper host (100496, 122880) / device (384, 384) F64 —
  matches job 13302646 calibration (~(100480, 122704), (384, 384));
  lifecycle layer host 1488 B (<< 4096 budget), device 0 on the graph
  path, exactly as required.
- Replay gates: e_replay ≈ e_first (statics-free D3 measure);
  replay-vs-body parity 3.6e-17–6.0e-17 F64 and 2.6e-7–3.0e-7 F32
  (gates 1e-10 / 1e-4; the flagged ~10x F32 margin held with ~300x room).

**Stage 4 failure — root cause (script bug, not production code):**
`fm048_ab_benchmark.jl` built `specs` via a comprehension whose snapshot
slot is always `nothing`, so the vector eltype narrowed to
`Tuple{…,Nothing}`; the `append!` of the new p018 arms (snapshot =
`Matrix{Float64}`) then threw `cannot convert a value to nothing`
(line 59). First job with p018 arms in stage 4, hence previously unseen;
parse checks cannot catch it. **Fix:** `specs` now has explicit eltype
`Tuple{String,DataType,Int,Float64,Union{Nothing,Matrix{Float64}}}`
(parse-checked). Counter field names and the timing-only use of the
un-zeroed snapshot (no direct reference in stage 4) were audited as
non-issues; the stage-4b sweep script builds fields per-call and has no
analogous pattern. No gate or accuracy measure was touched.

### Completed H200 run — job 13303399 (2026-08-22, all stages)

The resubmission with the stage-4 fix, **job 13303399**, COMPLETED
(exit 0:0, 37:52). All stages 1–5 ran; every testset passed. Artifacts in
`data/gpu_sfs_enablement/` (sha256s in `sha256_13303399.txt`; each
matches its in-job provenance hash): `vpm048-13303399.out`,
`fm048_device_tests_13303399.log`, `fm048_ab_13303399.{csv,log,provenance}`,
`fm048_sweep_13303399.{csv,log,provenance}`.

- **Stages 1–3 reproduced job 13302961** (same testset totals 63/63,
  300/300, 24/24 etc.; e.g. cube P4 F64 e_sfs 0.004347393966697365 vs
  ...367 — agreement to ~1e-15). All calibration expectations hold again.
- **Stage 4 (A/B matrix, 9 reps, alternating order)**: SFS marginal cost
  over U/J: cube n=2e4 F64 1.05–1.32 ms, F32 0.56–0.67 ms; p018 210k F64
  27.4–28.0 ms on t_uj ≈ 63–64 ms (+44%). Zero body uploads and zero
  expansion host copies in every arm. Steady-state device alloc: cube
  384/272 B (F64/F32) as gated; p018 3784 B — larger CUB scan/reduce
  scratch at 210k (wrapper layer, weakly n-dependent; the lifecycle-layer
  device==0 contract is tested at n=2e4 and unaffected). Recorded, not
  gated.
- **Stage 4b (tuning sweep)**: 40 rows (cube F64 full grid P{4,6,8} ×
  rho{4.211,4.789} × q{derived,14,17,20}, F32 spot checks, 8 p018 arms).
  `replay_drift` ~1e-17–1e-18 F64 everywhere including p018 — graph
  replay is accuracy-faithful at production scale. Strict-gate report on
  p018 (gate 5e-4 F64 delivered e_sfs, warm): production P=4/derived-q
  rows **FAIL as predicted** (2.80e-3 @ rho 4.211, 2.72e-3 @ 4.789);
  passing arms: P8/4.789/derived 5.89e-5 (106 ms), P8/4.789/q14 5.84e-5
  (132 ms), P6/4.789/q20 6.29e-5 (155 ms), P6/4.789/q17 7.24e-5 (133 ms),
  P6/4.789/derived 1.46e-4 (93 ms); P6/4.211/q14 fails (6.40e-4).
  Cube-vs-p018 discrepancy: p018/cube e_sfs ratio 0.62–1.42 (cube is
  mildly conservative for the rho=4.789 arms, ratio ~0.63, but
  underpredicts p018 for P4/rho4.789, ratio 1.42) — the cube grid ranks
  configs correctly but is not a quantitative proxy for the production
  field; p018 arms remain the gate of record.
- **Stage 5**: 047 device construction-lock check 4/4 pass.

### Verdict

**COMPLETE — approved 2026-08-22 with the user's production selection.**
Corrected testsets, allocation contracts, replay gates, strict
tail-budget operating point, synchronized A/B, tuning sweep with p018
strict-gate arms, and the 047 lock check all passed on the H200 (jobs
13302961 stages 1–3 + 13303399 stages 1–5, reproducing each other).

**Production settings (D14, user decision 2026-08-22): P=6,
rho_t=4.789, derived near shell** — on p018: e_sfs 1.46e-4 (passes the
strict 5e-4 gate with 3.4x margin), e_u 3.69e-6, t_ujsfs 93.4 ms
(+3% vs the failing P4 baseline). Implemented as the new coupling
defaults (`RadixFMMSettings.expansion_order = 6`,
`_PARTITIONED_RHO_T_DEFAULT = 4.789`); default-assertion tests updated.
The default change post-dates job 13303399 (whose evidence pinned
settings explicitly); regression coverage of the new defaults rides in
the next H200 job (049 acceptance).

The old H200 jobs remain historical evidence for the pre-remediation kernel,
not acceptance evidence for this version. No device result has been invented
or carried forward as a current pass.
