# 028 Performance Feasibility: 1,000,000 Particles in 10 ms

## Objective

Determine, with measured evidence, whether an accurate 1,000,000-particle solve can
run in at most 0.01 s per time step on a single H200 — and improve the production
path toward that target, not merely describe the gap. Consolidate the existing
benchmark record, fill material evidence gaps, classify each stage's binding
constraint, then implement the highest-value justified optimizations, verify
correctness and accuracy, and repeat the measured profile/optimize/retest cycle
until the target is reached or the remaining barriers and next steps are
rigorously quantified.

## Target Definition (fixed by user decision, `2026-07-28`)

- **Problem:** `n = 1,000,000` bodies, resident CUDA lifecycle, single H200.
- **Accuracy:** literature `P = 4` (`expansion_order = 3`), measured with the
  `024b` sampled-direct error methodology. **Float32 qualifies** for the verdict
  if its sampled error is within the `P = 4` truncation error measured at
  Float64; Float64 results are always reported alongside.
- **Verdict timing boundary — per-time-step cost:** one full resident evaluation
  (B2M → M2M → M2L → L2L → L2B plus nearfield direct) **plus everything a real
  time step repeats**: the device-side convection update and the tree/route
  refresh under the chosen reuse policy. No per-step H2D/D2H of body state.
- Two additional boundaries are measured and reported in every end-to-end table
  but do not decide the verdict:
  1. *Steady-state evaluation only* — bodies resident, tree/routes/operators
     already built; one evaluation.
  2. *Including transfers* — adds per-step H2D upload of positions/strengths and
     D2H download of influence, the boundary that applies if particle state
     lives on the host.
- Record the exact GPU model, driver, CUDA toolkit, CUDA.jl, and Julia versions
  with every dataset.

## Dependencies

- `025-theory-hierarchical-rigid-m2l-stencil.md`, `026-impl-hierarchical-m2l-host.md`,
  `027-impl-hierarchical-m2l-cuda.md` (the hierarchical path is the presumptive
  dominant lever at `n = 1e6`)
- `024b-impl-cpu-gpu-scaling-benchmark.md` (the `n = 1e6` baseline and error
  methodology), `024-impl-operator-ab-benchmark.md`, `024a-impl-benchmark-visualization.md`
- `019-impl-operator-performance-tuning.md`, `022-impl-gpu-device-resident-m2l.md`,
  `023-impl-production-integration.md`

## Required Reading

- `START_HERE.md`
- The dependency task files above and their Verification Notes
- `MATRIX_OPERATOR_REFACTOR/data/cpu_gpu_scaling/` (`024b`),
  `data/operator_ab_benchmark/` (`024`, including `summary/unresolved_notes.md`),
  `data/operator_performance_tuning/` (`019`), `data/hierarchical_m2l_host/` (`026`),
  `data/hierarchical_m2l_cuda/` (`027`)
- `src/translate_batched_resident.jl`, `src/translate_batched_cuda.jl`,
  `test/radix_fmm_timestepping_test.jl`

## Artifacts or Production Surface

- Study data and reports in `MATRIX_OPERATOR_REFACTOR/data/feasibility_1m_10ms/`.
- Benchmark/profiling scripts in `MATRIX_OPERATOR_REFACTOR/scripts/`
  (`benchmark_028_feasibility.jl` plus the `cuda_028_{submit,run,fetch}.sh`
  triplet, following the existing cluster pattern).
- Figures extending the `024a` set via the established TikZ/pgfplots + CSV
  toolchain.
- Production optimizations (Phase B) touch only files already owned by the
  resident lifecycle rows, under the `START_HERE.md` placement rules; every
  optimization lands with tests and H200 before/after data.

## Phase A: Evidence Consolidation and Gap Measurement

Start from the existing record; re-measure only where evidence is missing or
stale (anything predating `026`/`027` is stale for M2L structure):

1. **Consolidate** the `024b` `n = 1e6` point (the 94x-speedup regression whose
   cause the hierarchical rows address), the `024` four-strategy lifecycle data,
   the `019`/`022` stage breakdowns and roofline gap, and the `026`/`027`
   hierarchical-vs-flat results, into one end-to-end and per-stage time budget
   at the target configuration.
2. **Per-stage bound classification.** For every stage ≥ ~5% of the step
   (sort/tree refresh, route generation, B2M, M2M, M2L, L2L, L2B, nearfield,
   convection update): attribute compute-bound, memory-bandwidth-bound,
   transfer-bound, launch/latency-bound, or otherwise constrained, using
   achieved FLOP/s and GB/s against H200 peaks, kernel counts x measured launch
   overhead, and CUDA-event timings. State the attribution method with the data
   so the claims are reproducible.
3. **Transfer and residency costs.** Measure H2D and D2H costs separately from
   resident compute (positions/strengths up; influence down; construction-time
   uploads amortized separately). Using the per-step workload below, compare
   permanent device residency against round-tripping body state each step, and
   answer explicitly whether particle state must remain permanently on the GPU
   across convection and other time-stepping operations to meet the target.
4. **Per-step workload (convection proxy).** The repo has no production time
   stepper, so fix the representative per-step workload as: device-side Euler
   update of positions from the computed gradient (`x .+= v * dt`), then the
   tree refresh policy — measure both full re-sort/re-tree per step and any
   cheaper reuse/refit policy the resident cache supports. Build on the
   existing `test/radix_fmm_timestepping_test.jl` harness.
5. **Levers to evaluate** (extend as evidence directs; each gets an expected
   gain, confidence, and risk):
   - hierarchical stencil at `n = 1e6` (from `026`/`027`; expected dominant —
     the flat path's measured work exponent is 1.46);
   - **per-level M2L strategy mix** (user-raised `2026-07-28`): the `024`
     defaults were measured under the flat leaf-only list, where offset classes
     are fat — the regime favoring per-class dense GEMM. The hierarchical
     stencil thins classes to `(level, offset)` and coarse levels have few
     occupied nodes: the small-class regime where `023b` measured per-class
     launches as launch-bound and where the factored strategy's batch-shared
     `U_n`/`V_n` allow concatenating all classes of a level (or all levels)
     into a few large GEMMs. Evaluate a heterogeneous mix — e.g.
     factored/concatenated at coarse levels, fused dense at the leaf level —
     driven by the per-level class-occupancy histograms `026`/`027` record.
     No existing benchmark covers this mix; do not assume the `024` defaults
     transfer to the hierarchical path;
   - the L2B and grouped-GEMM M2L residuals deferred by `019`;
   - the Float32 invariant-cache construction anomaly (`023d`, shared
     infrastructure) — now on the verdict path since Float32 is admissible;
   - launch-bound small-class behavior and kernel-fusion opportunities;
   - nearfield direct cost at the target `(n, ell)` operating point.
6. **Report.** State whether one stage bottlenecks the solve or time is broadly
   distributed; quantify the remaining gap and its uncertainty at all three
   timing boundaries; deliver prioritized recommendations with expected gains
   and risks. **Pause for user approval of the optimization list before Phase B**
   (precedent: the `019b` user-discussion decision).

## Phase B: Optimize, Verify, Retest

Implement the approved optimizations in priority order. For each: verify
correctness and accuracy (tests must include `P = 4` per the standing project
rule; the `023` counter/allocation contracts and existing radix/CUDA tests stay
green), then repeat the end-to-end and per-stage measurements at the target
configuration and record realized versus expected gain.

Repeat Phase A → Phase B until the target is met at the verdict boundary, or
the best remaining lever's projected gain is quantified as insufficient to
close the gap. Each cycle ends with a user checkpoint; do not begin a new
optimization cycle without one.

## Verification

- All production changes covered by tests, including `P = 4`; existing radix,
  lifecycle, precision, and counter tests green; sampled-direct error at the
  target configuration within the `P = 4` Float64 truncation-error gate (and
  the Float32 admissibility check recorded whenever Float32 is used for the
  verdict).
- Final report reproduces: per-stage budget table at all three boundaries,
  bound classification with method, residency answer, realized gains per
  optimization with before/after H200 data, and the final verdict (target met,
  or quantified remaining gap plus next steps).

Record commands, job ids, and result summaries in a `Verification Notes`
section.

## Implementation Notes

### Phase A (complete; Phase B not started — user checkpoint pending)

**Full report: `MATRIX_OPERATOR_REFACTOR/data/feasibility_1m_10ms/report.md`.**

No `src/` changes. All Phase A work is benchmark/test-side:

| Artifact | Role |
|---|---|
| `MOR/scripts/benchmark_028_feasibility.jl` | Case matrix; three boundaries; per-stage CUDA-event medians; counter contract; sampled-direct accuracy; per-level M2L + class histograms |
| `MOR/scripts/fm028_device_system.jl` | `FM028DeviceSystem` + device Euler convection kernel |
| `MOR/scripts/cuda_028_{submit,run,fetch}.sh` | Cluster driver; `pilot`/`sweep` tiered case presets |
| `test/cuda_radix_convection_test.jl` | Convection/residency gate (runs before the benchmark) |

Commands:
```bash
bash MATRIX_OPERATOR_REFACTOR/scripts/cuda_028_submit.sh pilot   # -> job 12996475
bash MATRIX_OPERATOR_REFACTOR/scripts/cuda_028_submit.sh sweep   # -> job 12997508
bash MATRIX_OPERATOR_REFACTOR/scripts/cuda_028_fetch.sh <jobid>
```

**Answer: no at present — 91.4 ms vs the 10 ms target (9.1×)** at the best
accuracy-admissible configuration, hier12 · dense · **Float32** · ell=5 · **K=1740**,
n=10⁶, P=4. Float64: 104.0–106.3 ms. Boundaries: eval-only 86.4 ms, verdict 91.4 ms,
verdict+transfers 93.2 ms, fully host-resident 159.4 ms.

Newly established at n=10⁶ (all first-ever measurements on the hierarchical path):
- **hier12 confirmed as the verdict config**; hier3 is faster (52.7 ms) but its
  3.995e-3 gradient error is **3.4× over the accuracy gate** — inadmissible.
- **Float32 is admissible** (3.186e-4 vs F64 3.185e-4, +0.03%), worth 1.14×.
- **`ell`=5 is the optimum**; ell=4 → 349.5 ms, ell=6 → 546.7 ms.
- **K=1740 cures route_gen** (24.19 → 2.13 ms); that lever is now spent.
- **`precomputed_y` loses to `dense`** everywhere (172.3 vs 106.3 ms).
- **Lamb–Helmholtz costs 1.41×** (149.5 ms), entirely in M2L.

Bound classification — the step is dominated by two kernels, 68% in F32:
- **L2B+nearfield 38.4 ms (42%)**: partially compute-bound, FP64-rsqrt-limited.
  4.83e9 interactions/step; 91.5 G/s (F64) → 125.8 G/s (F32); 2% of HBM3e ⇒ not bandwidth.
- **Leaf M2L 23.6 ms (26%)**: **neither bandwidth- nor compute-bound** — proven by
  precision A/B at fixed work (22.41 ms F64 → 23.55 ms F32, unchanged). Overhead-bound,
  mechanism unidentified.
- **Host allocation 57.7 MB/step** (~1.76 KB/cell/step), ~14 ms.

Two levers the task file named are **demoted by the data**: per-level M2L strategy mix
(≤3 ms, because coarse levels cost 2.86 ms total at K=1740) and stale-tree refresh
(3.9 ms, and its accuracy test at dt=1e-5 is too weak to trust). Recommended Phase B
order: **host-alloc elimination → nearfield kernel rewrite → leaf-M2L profiler pass**.

### Phase B (in progress — "bank the certain wins first", user decision 2026-07-31)

De-risking jobs 12998146 / 12998189 resolved the three Phase A inferences; **two were
wrong** (details in report.md §6b):
- fused stage is **98% nearfield**, not 85% — L2B is ~1 ms, so lever 1 is a pure
  nearfield lever;
- host allocation is `_assert_cuda_scratch_value!` path strings (1.35M allocs/step,
  33.8 MB) + `collect(1:n)` (8.0 MB), **not** the `accumulate!` scratch;
- leaf M2L is **block-dispatch-bound**: `_cuda_hier_dense_fused_kernel_` launches
  Threads=32 / Blocks=31,307,680, one warp per route.

**Lever 3 complete** (job 12998517), two `src/translate_batched_cuda.jl` changes that
preserve the 023 invariant contract — an allocation-free `_cuda_scratch_value_ok`
fast path guarding the existing walker, and `Base.OneTo` in place of `collect(1:n)` on
the device-resident path:

| metric | before | after |
|---|---|---|
| host allocation / step | 57.7 MB | **0.80 MB** (-99%) |
| verdict step F32 | 91.40 ms | **69.64 ms** (-24%) |
| verdict step F64 | 104.00 ms | **82.90 ms** (-20%) |
| gradient rel RMS | 3.186e-4 | 3.186e-4 (unchanged) |

**Lever 2 complete** (job 13010174) — grid-stride the fused dense M2L kernel instead
of one block per route, cutting leaf blocks 31,307,680 -> 16,384:

| | after lever 3 | after lever 2 |
|---|---|---|
| F32 `m2l_ms` (leaf) | 26.64 (23.58) | **21.61 (19.60)** (-19%) |
| F32 verdict step | 69.64 ms | **64.56 ms** (-5.08) |
| F64 `m2l_ms` / verdict | 25.13 / 82.90 | 25.69 / 83.38 (+2% / +0.6%) |
| gradient rel RMS | 3.186e-4 | 3.186e-4 (unchanged) |

**Partial: scoped 10-20 ms, delivered ~5 ms on the verdict path.** The
block-dispatch attribution was only partly right — a 1900x dispatch reduction bought
19% in F32 and nothing in F64. What it did establish: the leaf was precision-*in*sensitive
before (22.41 F64 vs 23.58 F32) and is precision-sensitive after (23.27 vs 19.60), so
removing dispatch exposed a bandwidth- or atomic-throughput limit (~500M atomics/step).
Distinguishing those two is the next question for this stage. Kept because it is a clear
gain on the F32 verdict config and the F64 delta is inside the ~8% run-to-run variance.

Also added `fused dense M2L grid-stride parity` (8 tests) — **the only hierarchical
coverage in the CUDA gate**. `cuda_radix_lifecycle_test.jl` never builds a hierarchical
cache, so its 215 tests passed against a kernel compiling to invalid IR across jobs
13000341/13009348/13009363. Root cause of those was a missing
`const gridDim = CUDA.gridDim` binding (the file aliases `blockIdx`/`blockDim`/
`threadIdx` but not `gridDim`); an audit now checks every bare intrinsic has a binding.

**Lever 1 complete** (approved with riders 2026-07-31; jobs 13015315/13015316/13015336) —
the nearfield kernel `_cuda_direct_pairs_output_kernel!` is now **warp-per-pair with a
grid-stride** capped by a new `DIRECT_CUDA_MAX_BLOCKS` knob (was one thread per pair
running a serial ~30x30 loop), and `inv(sqrt)` is replaced by `_cuda_fast_rsqrt`
(hardware `rsqrt.approx` in F32; in F64 two Newton refinements restore ~1-2 ulp, since
raw `CUDA.rsqrt(::Float64)` is only ~1e-7 accurate). Riders in the same cycle: L2B is
warp-per-cell (0.96 -> 0.72 ms); the per-step `CUDA.zeros` in
`finalize_cuda_radix_output!` became a cached device buffer; dead
`_cuda_find_cell_for_sorted_body` deleted. A B2M warp-per-cell rider was measured
**slower** (1.02 -> 2.03 ms, 113 regs, 15/32 lanes at P=4) and reverted with a comment.

| metric | before (13010174) | after (13015336) |
|---|---|---|
| nearfield (exact split) F32 / F64 | 37.7 / 51.9 ms | **11.45 / 37.6 ms** |
| F32 verdict step | 64.56 ms | **38.04 ms** (-41%) |
| F64 verdict step | 83.38 ms | **68.99 ms** (-17%) |
| gradient rel RMS | 3.186e-4 | 3.186e-4 (unchanged) |

Scoped 20-33 ms, delivered **26.5 ms** on the F32 verdict path. F64 improved only
1.38x: with dispatch and parallelism fixed, the F64 nearfield is genuinely
FP64-throughput-bound (rsqrt Newton chain + FP64 vector rate), which is why F32 —
already the verdict precision — gains 3.3x.

New tests: `_cuda_fast_rsqrt accuracy` (1e-14 F64 / 5e-7 F32) and
`nearfield warp-per-pair parity` (16 tests: F64+F32, caps typemax/3/1, n=2000 and
n=40 for ragged/empty cells, expansion_order=3).

**Leaf-M2L mechanism resolved** (derisk section D, job 13015316): at fixed work,
atomic 21.07 / plain-store 25.15 / no-store 24.35 ms (F64; F32 17.21/19.37/19.00).
Atomics are **not** the limit — removing every write leaves >95% of the cost. The
leaf M2L is bound by operator/multipole **loads and compute**, so the next M2L lever
is data reuse (shared-memory operator tiles + class-batched routes), not atomics.

**Cycle 2 complete** (user-approved 2026-07-31; job 13015753) — operator-tiled leaf
M2L, `_cuda_hier_dense_tiled_kernel!` in `translate_batched_cuda.jl`. Each block owns
a contiguous route chunk; same-class segments (window routes are class-sorted) run
with the class operator staged once in shared memory, both level diagonals folded
into the tile, warps streaming routes with a per-warp shared multipole column.
Per-route global traffic falls from `D^2 + D` loads to `D` loads + `D` atomics.
Gated by `DENSE_CUDA_TILED` / `DENSE_CUDA_TILED_MIN_ROUTES=65536` (coarse windows
keep the plain fused kernel) and a 48 KB shared-memory fit check with fallback.
12-test tiled-vs-fused parity set (forced-tiled / deep-chunk / one-block, F64+F32).

| metric | after lever 1 (13015336) | after cycle 2 (13015753) |
|---|---|---|
| leaf M2L F32 / F64 | 19.60 / 23.27 ms | **12.71 / 13.66 ms** (-35% / -41%) |
| M2L total F32 / F64 | 21.62 / 25.63 ms | **14.23 / 15.31 ms** |
| F32 verdict step | 38.04 ms | **30.72 ms** (-19%) |
| F64 verdict step | 68.99 ms | **58.77 ms** |
| gradient rel RMS | 3.186e-4 | 3.186e-4 (unchanged) |

The tiled leaf is nearly precision-insensitive again (13.66 F64 vs 12.71 F32), so
the removed traffic was the precision-sensitive component; the residual ~13 ms is
compute/atomic/latency and would need its own attribution pass before further work.

**Standing: 3.1x over target** (9.1x -> 7.0x -> 6.5x -> 3.8x -> **3.1x**, F32
verdict 30.72 ms). Budget: leaf M2L 12.7 ms (41%), nearfield 11.45 ms (37%), other
~6.6 ms. The two kernels are now balanced. Next-cycle candidates (need user
sign-off): two-stream nearfield/far-field overlap (the stages are independent and
now comparable in size — overlap hides ~11 ms; requires the blocking-sync +
pageable-copy cleanup), a fresh attribution pass on the residual leaf M2L and
nearfield (both now within ~8x of their roofline floors), counting sort (~2 ms).

## Verification Notes

### Phase A

Cluster gates (both jobs): `LIFECYCLE_TEST_EXIT=0`, `CONVECTION_TEST_EXIT=0`.
**27 rows, all `fit=true`** — no failures, no OOM, empty failure ledger. Peak device
memory 1.65–1.99 GB of 140 GB at n=10⁶.

- **Counter contract asserted per case** (harness `error()`s on violation):
  `body_uploads=0`, `metadata_downloads=0`, `expansion_host_copies=0`, route/operator
  uploads flat across recurring steps. All 27 rows pass ⇒ the device-resident boundary
  is genuinely achieved.
- **Accuracy hard-gate** (≤10× of 1.19e-4 = 1.19e-3): hier12 3.185e-4 ✓ (F64),
  3.186e-4 ✓ (F32), flat 1.187e-4 ✓. hier3 3.995e-3 ✗ — recorded as failing.
- **Reference integrity**: `ref_cross_check_grad_rel = 2.76e-14` against the checksummed
  024b CSV (`ec96762990...`), so the gate is not measuring harness error.
- **024b cross-check**: flat dense ell=4 n=10⁶ reproduces 1.1873e-4 (F64) and 1.1889e-4
  (F32) to 5 digits, and reproduces itself across jobs 12996475/12997508 to 0.05%
  (363.50 vs 363.33 ms).
- **Convection sanity**: after 3–5 device Euler steps, re-evaluated field vs a *fresh*
  on-device direct reference at moved positions = 3.1842e-4 (step-0: 3.1852e-4).
  Host/device paths agree to 13 digits.

Threats to validity, recorded in full in report.md §7:
1. **Run-to-run variance on the hierarchical path is ~8%** (127.2 vs 117.1 ms for an
   identical config across the two jobs; the flat path reproduced to 0.05%). Lever gains
   below ~10 ms are within noise.
2. **The n=2e5 027 tie-in row is not like-for-like and implies no speedup.** 028 pins
   `bounds=(-0.01, 1.02)` (4,096 cells at ell=4); 027 fitted the tree to the body bounds
   (2,744 cells). 028 therefore has 48.8 vs 72.9 bodies/leaf and ~30% less nearfield
   work. 028's own n-scaling remains internally valid (identical grid at every n).
3. **ell=6 was measured at K=256**, where route_gen is uncured (195.2 of its 546.7 ms);
   corrected to the K=1740 regime it is still ≈354 ms, so the ell conclusion holds.
4. A pilot extrapolation of ell=6 ≈ 220 ms was **wrong by 2.5×** (measured 546.7 ms) and
   has been discarded in favour of the measurement.

### Phase B lever 1 (2026-07-31)

- Jobs: **13015315** (verify, lever 1 + riders), **13015316** (derisk: exact
  nearfield split + new leaf-M2L atomic-vs-bandwidth A/B), **13015336** (verify,
  final, after the B2M rider revert). All H200; `LIFECYCLE_TEST_EXIT=0` and
  `CONVECTION_TEST_EXIT=0` on every job (215 + 24 + 8 + 2 + 16 tests).
- Commands: `bash MATRIX_OPERATOR_REFACTOR/scripts/cuda_028_submit.sh verify|derisk`,
  fetched with `cuda_028_fetch.sh <jobid>`.
- Accuracy: gradient rel RMS 3.185e-4 (F64) / 3.186e-4 (F32), identical to the
  pre-lever record; the parity testset bounds the launch-decomposition effect to
  atomic-reassociation rounding (1e-10 F64 / 1e-4 F32).
- Counter/alloc contract: `verdict_step_host_alloc_bytes` 0.80-0.81 MB (unchanged
  from lever 3), route/operator uploads flat, `expansion_host_copies=0`.
- Final data: `cuda_m13h-1-1_20260731-183510.csv` (+`.classes.csv`),
  `cuda_m13h-2-1_20260731-181908.csv`, `derisk_m13h-2-2_20260731-181957.csv`,
  `fm028-1301531{5,6}.out`, `fm028-13015336.out`.

### Phase B cycle 2 (2026-07-31)

- Job **13015336** is the before; job **13015753** (verify) is the after. Gates:
  `LIFECYCLE_TEST_EXIT=0`, `CONVECTION_TEST_EXIT=0`; new `dense M2L operator-tile
  parity` 12/12 alongside all prior testsets.
- Accuracy 3.185e-4 / 3.186e-4 — unchanged; counter/alloc contracts hold.
- Data: `cuda_m13h-1-2_20260731-210422.csv` (+`.classes.csv`), `fm028-13015753.out`.

## Approval Notes

To be filled by a different agent after this task is complete.
