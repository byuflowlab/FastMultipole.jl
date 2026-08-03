# 028 Performance Feasibility: 1,000,000 Particles in 10 ms

## Status

**Done and Approved (2026-08-03, third-agent clear-context approval).** The
independently reproduced winner is FP16-input/FP32-accumulation WMMA at
**9.591 ms** ([9.434, 9.631] ms), sampled-direct gradient relative RMS
**1.0593e-3**, H200 job **13029878**, source manifest `42a6c254a11ac8a8`. This
satisfies the fixed 10 ms and 1.19e-3 gates.

The clear-context review (see `Approval Notes`) accepted the evidence and, on
user direction, moved the production defaults onto the measured optimum and made
the level-radius schedule a public keyword. Those default changes were then
validated on H200 in job **13031482** (manifest `c0afa01083322fb8`): all gates
green including the expanded 33,381-test hierarchy suite, winner reproduced at
9.653 ms / 1.0593e-3 under the shipped defaults, counter/allocation contracts
unchanged. A third agent verified the evidence chain and granted approval; see
the final entry under `Approval Notes`.

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

**Cycle 3 complete — negative result** (user-approved 2026-07-31; job 13015982):
nearfield/far-field stream overlap via side stream + device events
(`CUDA_OVERLAP_NEARFIELD`, begin/done event ordering against the previous step's
finalize and L2B). Parity 4/4 across a convection loop. **Measured gain: 0 ms**
(F32 verdict 30.718 vs 30.724): after levers 1-2 both dominant kernels saturate
the device (eval 27.5 ms vs ~28.6 ms stage sum, >95% busy), and saturating kernels
time-share SMs — wall time is the sum with or without streams. The 5-15 ms overlap
estimate was conditioned on the pre-lever-1 latency-bound nearfield; lever 1's
success retired this lever. Kept, default on (correctness-neutral; can only help
in launch-bound regimes at small n).

**Standing accuracy-admissible result: 3.1x over target** (9.1x -> 7.0x -> 6.5x -> 3.8x -> **3.1x**, F32
verdict 30.72 ms), now established as ~95% device-saturated compute. Further gains
require reducing work or per-kernel efficiency: fresh attribution passes on the
residual leaf M2L (~12.7 ms, precision-insensitive) and F32 nearfield (~11.45 ms
vs ~1.6 ms roofline floor), plus the ~2 ms counting sort.

**Cycle 4 attribution complete (2026-08-01; H200 job 13016756).**  q=3 reaches
**9.434 ms** but remains inadmissible at 3.995e-3 gradient relative RMS (3.4x over
the 1.19e-3 gate); q=12 reproduces at 30.784 ms / 3.186e-4.  Thus 10 ms is bracketed,
not achieved.  Exact work counts predict q=4 near 10.4 ms and q=5 near 14.0 ms,
with accuracy unmeasured.  Every distinct shell q=3:12 passes the independent
downward-monotonicity audit, making an intermediate-radius accuracy/timing sweep the
highest-value next cycle.

The same audit identifies a conditional second lever: q may decrease with depth
while preserving the task-025 exact-once invariant. All 45 two-stage schedules
over the distinct shells pass cross-level monotonicity. q=4 on levels 2--4 and
q=3 at the leaf projects near 10.1 ms, but should be tested only after uniform q=4
establishes the accuracy frontier. `radius_schedule_candidates.csv` records the
exact route/direct counts. Expansion-order/radius co-design is excluded for now
because the user-fixed target definition requires literature P=4.

Mechanism A/Bs: tiled M2L is 10.461 ms atomic / 8.752 store / 8.660 no-store and
has a 9.636 ms best isolated launch (64 threads, cap 65536), so it is mostly
arithmetic/input with ~1.71 ms atomic cost. Nearfield is 11.452 / 11.470 / 11.438 ms,
so neither output atomics nor launch shape is material. This promotes tensor/class-
batched M2L and symmetric self-interaction nearfield, and demotes target-cell
nearfield. No `src/` generalization or optimization was made in cycle 4. Full data
and triage: `plans/20260801_028_attribution.md` and report.md "Cycle 4".

## Staged Continuation Roadmap (reordered checkpoint, 2026-08-01)

The following stages record the user/agent brainstorm after cycle 4. They are a
decision-ready backlog, **not approval to begin production changes**. The order is
measurement-driven: select the accuracy/work regime, re-optimize its geometry, bank
cheap exact wins, then compare the dominant kernel opportunities before committing
to new expansion theory. Stop as soon as an independently reproduced result meets
both 10 ms and the 1.19e-3 accuracy gate. Every timing gate includes recurring
construction, scratch reduction, and sorting introduced by the candidate;
kernel-only wins do not qualify.

### Stage 5 — uniform radius/accuracy frontier

Generalize the rigid policy from `near_radius2 in (3, 12)` to the distinct lattice
shells `q in (3, 4, 5, 6, 8, 9, 10, 11, 12)`. Derive the epsilon separator from the
actual farthest-near/nearest-far lattice norms, derive rather than special-case the
table extent, and expand the task-025/026/027 host and CUDA exact-once/classifier
coverage. Then run the fixed n=1e6, P=4, ell=5, dense, Float32 sampled-direct
accuracy and full verdict timing for every shell; use a window covering the complete
operator-class set rather than inheriting a q=12-specific K blindly.

Decision gate:

- q=4 is first because the endpoint fit predicts ~10.4 ms; no prediction counts as
  a result until its sampled-direct error is measured.
- Select the fastest accuracy-admissible uniform shell. If it is <=10 ms, reproduce
  it independently and stop before adding machinery.
- Otherwise carry only that shell and the immediately faster inadmissible neighbor
  into the following attribution; do not optimize all q values.

### Stage 6 — re-optimize selected geometry and bank exact low-risk wins

The q change invalidates the old claim that ell=5 is jointly optimal. Re-bracket ell
on both sides of the selected shell, confirm the full-class K/window choice, and
establish a repeated selected-q baseline before changing kernels. Then test:

1. **M2L launch A/B:** confirm 64 threads / block cap 65536 end to end; cycle 4
   measured an isolated 0.827 ms leaf-M2L gain at q=12.
2. **15-bit Morton counting sort:** replace comparison/radix machinery only if the
   exact bounded-key pass reduces the complete ~2.5 ms grid/refresh budget. Include
   histogram clearing, scans, permutation, and any scratch initialization.

These are independent changes and must each have an on/off full-verdict A/B. Bank
only reproduced gains. If their combination crosses 10 ms at admissible accuracy,
perform the independent final validation and stop.

### Stage 7 — attribute error before refining the stencil

If the fastest nearby q is inadmissible while the selected shell is materially
slower, decompose the sampled field error by M2L level, offset norm, and complete
cubic-symmetry orbit. Use linear partial-field/replay experiments at the fixed
reference samples to determine whether the error is leaf-dominated or concentrated
in a small set of close translation classes. Only then choose between:

1. **Non-increasing level-dependent radius.** The task-025 proof extends when
   `q_child <= q_parent`, and the deterministic audit verifies all 45 coarse/leaf
   combinations. Candidate transition tables are parameterized by
   `(q_parent, q_child)` and require brute-force exact-once tests. The first modeled
   examples are q=4 on levels 2--4/q=3 at the leaf (7.55M routes, 0.831M direct cell
   pairs, ~10.1 ms) and q=5 coarse/q=4 leaf.
2. **Orbit-selective direct correction.** Promote selected complete cubic orbits
   from M2L to direct evaluation only if attribution supports it. A non-spherical
   near set cannot be certified by the current radial epsilon alone: it must preserve
   inversion/cubic symmetry, pass an independently enumerated cross-level
   downward-monotonicity/exact-once proof, and define a replacement accuracy
   contract. Never select individual directions because a finite sample favors them.

Promote either refinement only if it passes the full sampled-direct gate and beats
the re-optimized uniform shell end to end. If a refined stencil crosses 10 ms,
validate and stop.

### Stage 8 — competitive residual-kernel bake-off

At the best admissible geometry, remeasure the nearfield and M2L budgets and compare
bounded prototypes before selecting a production rewrite. The first competitors are:

1. **Unordered symmetric nearfield.** For the dedicated same-source/target verdict
   path, enumerate each direct cell pair once, compute every unordered body pair
   once, and update both endpoints; include a triangular same-cell kernel. Preserve
   the directed kernel for the general rectangular source/target API. This comes
   first because cycle 4 found no measurable output-atomic penalty, so the simpler
   symmetric kernel may capture most of the arithmetic saving.
2. **Tensor/mixed-precision 16x16 M2L.** Compare the optimized FP32 tiled kernel with
   TF32, FP16, and BF16 inputs using FP32 accumulation. Every format is a numerical
   method change and must pass the full sampled-direct gate, not only operator parity.
3. **Class-dependent low-rank audit.** Record singular-value spectra and minimum
   admissible ranks of all selected-radius operator classes. Prototype factored
   batched application only if close-separation classes compress enough to reduce
   both arithmetic and traffic. This is distinct from the already-slower analytic
   rotation/factored strategy.

Compare useful work, occupancy, scratch/accumulation cost, accuracy, and projected
full-verdict gain on the real selected-q leaf geometry. Implement the strongest
bounded candidate first, then remeasure and stop if the objective is reached; do not
automatically implement all three.

### Stage 9 — escalation branches if the simple bake-off is insufficient

#### Stage 9a — shifted-macrocell nearfield

Use this branch only if unordered symmetric cell pairs expose an accumulation or
scheduling bottleneck that macrocell ownership can plausibly remove.

1. **Two shifted coarse meshes plus fine mesh.** Let A be the aligned ell-1 mesh,
   B the ell-1 mesh shifted by one fine cell in all axes, and C the ell mesh. In
   exact arithmetic, `self(A) + self(B) - self(C)` counts the union of the A/B
   coarse-cell cliques exactly once. On the full ell=5 grid this covers 250,236 of
   431,676 unordered q=3 cell pairs (58.0%), 47.8% of q=4, and only 9.6% of q=12.
   Its maximum distance-evaluation reduction is therefore ~29%, ~24%, and ~5%; use
   it as a diagnostic rather than assuming the remaining 3-D mixed-parity pairs are
   a small residual.
2. **Eight-phase 2x2x2 ownership.** Use every coarse-grid shift, assign each q<=3
   fine-cell pair to exactly one phase, accumulate one private result per phase, and
   reduce the phase outputs. Handle q=4's six axial distance-two offsets separately.
   This can halve the q=3 distance evaluations and covers about 82% of q=4's physical
   cell pairs before its axial shell.

Do not add recurring radix sorts to materialize these meshes: the aligned mesh is a
Morton prefix of C, and shifted macrocells should gather existing C-cell spans from
device occupancy metadata. Prefer unique pair ownership/masking to literal FP32
`A+B-C` subtraction. Include all phase scratch and final reduction in the verdict;
fall back to the unordered-pair kernel if macrocell bookkeeping consumes the gain.

#### Stage 9b — advanced M2L

Use this branch only if M2L remains a material part of the measured gap after the
tensor/low-rank comparison.

1. **Class-/target-owned M2L:** prototype only a design that improves input/operator
   reuse as well as removing atomics; cycle 4 bounds the atomic-only opportunity near
   1.7 ms.
2. **Plane-wave/exponential theory:** derive the multipole-to-exponential, diagonal
   translation/merge-and-shift, and exponential-to-local maps in the repository's
   normalization. Determine quadrature/sample counts and an error bound for the
   selected q at P=4, Float32 stability, and scalar verdict path; state Lamb--Helmholtz
   requirements separately. Compare a one-level conversion-plus-translation prototype
   on the real leaf geometry against the tensor baseline. Historical ~2x whole-FMM
   results are not assumed: at q=12, halving 14.23 ms M2L changes 30.78 ms to ~23.7 ms
   (1.30x overall), while at modeled q=4 it would move ~10.4 to ~8.7 ms.
3. **Spatial/polyphase FFT:** for the full uniform 32^3 leaf grid, assess M2L as a
   convolution of the 16 multipole coefficient fields. Account for the eight
   source-phase channels in the task-025 V-list, open-boundary zero padding,
   Fourier-kernel storage, and plan/workspace costs. Compare batched 3-D FFTs plus
   per-frequency coefficient mixing against route-wise M2L. Preserve the route path
   for sparse/adaptive grids. Tiny coefficient-space FFTs remain low priority at P=4
   unless a primitive benchmark overturns their transform overhead.

Plane waves diagonalize coefficient translation and preserve an adaptive FMM;
spatial FFT amortizes all translations over this dense lattice. Select between them
from the selected-radius cost model and prototype—not from asymptotic complexity.

### Stage 10 — retained residuals, demotions, and scope boundary

- **Target-cell-owned nearfield:** demoted behind unordered symmetry and shifted
  macrocells. Pair-list removal/source reuse may help, but cycle 4 refuted its proposed
  atomic benefit.
- **Native real basis:** demoted; at P=4 its analytic lane ceiling is 20% before
  conversion overhead.
- **Tree/route incremental refresh:** retain only after defining a realistic motion/dt
  contract and counting cell crossings. The stale-tree accuracy check is too weak.
- **CUDA graphs, more stream overlap, and launch-only nearfield tuning:** closed unless
  a later algorithm makes the step launch-bound; cycles 3--4 measured no opportunity
  in the current saturated kernels.
- **Re-bracket ell again** only after a later M2L backend changes the work complexity;
  the first required re-bracket already occurs in Stage 6 immediately after q.

Outside fixed 028 unless the user changes the target: expansion-order/radius co-design
(higher P with smaller q), multi-GPU decomposition, and replacement by PME/PMMM or a
global particle-mesh solver. Spatial FFT remains in scope only as an M2L backend that
preserves the existing FMM accuracy and sparse/adaptive fallback semantics.

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

### Phase B cycle 3 (2026-08-01)

- Job **13015982** (verify): gates green incl. `nearfield stream-overlap parity`
  4/4; accuracy and counter/alloc contracts unchanged; verdict timings identical
  to cycle 2 (the recorded negative result). Data: `fm028-13015982.out` and the
  fetched `cuda_*.csv` of that run.

### Phase B cycle 4 (2026-08-01)

- Local deterministic radius audit:
  `julia --project=. MATRIX_OPERATOR_REFACTOR/scripts/analyze_028_radius_candidates.jl`.
  All q=3:12 rows pass `minimum_child_norm2 > q`; q=12 reproduces the production
  5,189,728 direct cell pairs and 31,307,680 leaf routes (34,343,088 all levels).
  The companion two-stage audit verifies all 45 non-increasing coarse/leaf q
  schedules and writes `radius_schedule_candidates.csv`.
- Attribution script parses on Julia 1.12.5; `cuda_028_run.sh` and
  `cuda_028_submit.sh` pass `bash -n`; `git diff --check` passes.
- User approved the shown H200 payload; job **13016756** passed lifecycle 215/215,
  concatenated host parity 37/37, convection 24/24, fused M2L 8/8, stream overlap
  4/4, tiled M2L 12/12, rsqrt 2/2, and nearfield parity 16/16.
- Endpoint results: q=3 9.434 ms / 3.995e-3 (accuracy fail); q=12 30.784 ms /
  3.186e-4 (pass). Attribution results and decisions are recorded in the standing
  summary and report. Data: `attribution_m13h-1-1_20260801-062135_*.csv`,
  `cuda_m13h-1-1_20260801-062246.csv` (+ classes), `fm028-13016756.out`.
- No `src/` file changed. The intermediate-radius production/theory change awaits
  the next user checkpoint.

### Phase B Stages 5–6 (2026-08-01)

- Generalized the rigid policy, table construction, analytic epsilon separator,
  task-025 verifier, host/CUDA tests, and 028 harness to the distinct shells
  `q=(3,4,5,6,8,9,10,11,12)`. The q=12 production default and q=3 classic helper
  are unchanged; unsupported/redundant radii report the full supported set.
- The extended deterministic verifier and host hierarchical suite pass all nine
  shells, including exact-once dense/sparse/boundary coverage and Float32/Float64
  scalar/Lamb–Helmholtz surfaces.
- Stage 5 jobs **13016917** and **13016927** independently reproduced the nine-row
  frontier. The green run selected q=6 at 17.383 ms / 5.745e-4; q=3/4/5 failed
  the 1.19e-3 gate, and no admissible shell crossed 10 ms. Job 13016917 is retained
  as a failure-ledger entry for one pre-benchmark stochastic overlap parity miss;
  its complete frontier agreed with the green run to <=0.9%.
- Stage 6 job **13017112** re-bracketed q=6 at ell=4/5/6
  (33.946/17.525/72.901 ms). The 64-thread/65,536-block tiled launch improved the
  complete verdict from 17.455 to 17.142 ms in Float32 and 29.471 to 28.957 ms in
  Float64, with identical errors; it is banked as the internal default.
- Counting-sort job **13017128** passed the new 15/15 lifecycle/parity test. The
  bounded Morton path reduced refresh 2.742 -> 0.967 ms and full Float32 verdict
  17.142 -> **15.428 ms**; Float64 improved 28.938 -> 27.162 ms. Accuracy and
  transfer/persistent-buffer contracts were unchanged, so the path is retained
  for bounded depths. The 10 ms objective remains open.
- Final independent job **13017362** passed the radix lifecycle (215/215), all
  convection/optimized-kernel gates, and the expanded all-radius CUDA suite
  (32,264/32,264). The retained q=6 production internals reproduced at
  **15.401 ms** [15.226, 15.495] in Float32 and 27.350 ms [27.292, 27.437] in
  Float64, both at 5.745e-4 gradient error. Source manifest
  `437d12a2be8e14b5`; the 10 ms early-stop condition was not met.
- Detailed medians/ranges, decisions, manifests, and raw CSV names are recorded
  in `data/feasibility_1m_10ms/report.md` section 6.5. The 028 index row remains
  open because later stages are still necessary.

### Phase B Stage 7 — level-scheduled frontier (2026-08-03)

- H200 job **13027263** passed the lifecycle (215/215), host parity (37/37),
  convection/optimized-kernel, and expanded scheduled/symmetric/tensor CUDA gates
  (33,367/33,367). Its source manifest was `42a6c254a11ac8a8`.
- The q=5 replay reconstructed the device field to 1.760e-7 relative RMS (q=6:
  1.842e-7), confirming that the linear level/orbit decomposition was faithful.
  Uniform q=5 remained inadmissible at 1.4224e-3 gradient relative RMS.
- The complete five-policy frontier selected `sched6-5-5-5`, i.e. q=6 only at
  level 2 and q=5 at levels 3--5. It measured **12.514 ms** at 1.0498e-3,
  18.9% faster than the same-run uniform-q=6 result of 15.445 ms at 5.745e-4.
  The schedule is admissible but remains 2.514 ms above the target.
- Raw evidence is `fm028-13027263.out`,
  `stage7_replay_m13h-1-2_20260803-075304.csv`, and
  `cuda_m13h-1-2_20260803-075427.csv` plus its class companion.
- Jobs 13027048, 13027092, 13027167, 13027174, and 13027188 are retained only
  in the failure ledger: their hard gates exposed, respectively, a device-context
  array-rank type coupling, host BF16 lowering on Julia 1.11, route-class/test
  portability defects, an invalid symmetric-context field access, and an invalid
  replay-context field access. Job 13027374 was cancelled after discovering that
  the winner selector compared prefixes against full paths. All defects were fixed
  and covered by the green 13027263 gate; none of the failed jobs supplies timing
  evidence.
- Stage 8 finished its isolated bake-off. A combined symmetric/tensor run was not
  warranted because symmetric nearfield was decisively slower. Stage 9 was not
  started because the tensor winner met the early-stop condition.

### Phase B Stage 8 — residual-kernel bake-off (2026-08-03)

- H200 job **13029480** independently repeated every preflight and the expanded
  33,367/33,367 CUDA gate before measuring the selected `sched6-5-5-5` geometry.
  The Float32 baseline was 12.547 ms at 1.0498e-3 gradient relative RMS.
- The low-rank audit rejected a prototype: its route-weighted admissible rank was
  7.064 of 16 and its modeled operator-work fraction was 0.883, too little
  compression to offset factor/application traffic.
- Unordered symmetric nearfield was decisively negative (52.480 ms), and TF32
  cuBLAS was also negative (41.673 ms at 1.1424e-3). Neither is combined or
  promoted; both switches remain disabled by default.
- FP16-input/FP32-accumulation WMMA measured **9.603 ms** [9.421, 9.658] at
  1.0593e-3. BF16/FP32 measured **9.580 ms** [9.419, 12.097] at 1.0632e-3.
  Both cross the timing and accuracy gates; FP16 has the tighter first-run range.
- Job 13028465 is a failure-ledger entry only. Its 16-route TF32 harness forced
  about 769,000 host-driven chunks per M2L pass and was cancelled; TF32 was rerun
  with production-sized 16,384-route per-class batches in the green job 13029480.
- Independent job **13029878** passed the lifecycle (215/215), host parity (37/37),
  convection/optimized-kernel, and expanded CUDA hierarchy gates (33,367/33,367),
  then reproduced FP16 at **9.591 ms** [9.434, 9.631] / 1.0593e-3 and BF16 at
  **9.638 ms** [9.473, 9.713] / 1.0632e-3. Both satisfy the 10 ms and 1.19e-3
  gates under source manifest `42a6c254a11ac8a8`; `STAGE8_REPRO_EXIT=0`.
- Final reproduction artifacts are `fm028-13029878.out` and
  `cuda_m13h-1-1_20260803-{131016,131129}.csv` plus class companions. FP16 is the
  reproduced winner because it is faster and slightly more accurate in the final
  run. The objective is met, task 028 stops here, and Stage 9 is not authorized or
  necessary.

## Approval Notes

### Clear-context review, 2026-08-03 — accepted with changes (approval still open)

Reviewed against the `START_HERE.md` §6 order. Evidence verified directly:
`fm028-13029878.out` records the green lifecycle (215/215), host parity (37/37),
convection/optimized-kernel, and expanded hierarchy (33,367/33,367) gates and the
`verdict 9.591 / grad_err 1.059e-03` row; `cuda_m13h-1-1_20260803-131016.csv`
reconstructs the verdict boundary from its stages (refresh 0.955 + eval 8.236 +
finalize 0.219 + euler 0.034) and holds the counter contract (`body_uploads=0`,
`metadata_downloads=0`, `expansion_host_copies=0`). The 1.19e-3 accuracy gate was
fixed in Phase A and never moved: hier3 was rejected under it in Phase A, uniform
q=5 in Stage 7. Method quality is high — exact-once coverage extended from 2 to 9
shells plus schedules, the epsilon separator derived rather than special-cased
(q=7 correctly excluded as having no lattice shell), a maintained failure ledger,
and negative results (symmetric nearfield, TF32, stream overlap) retained.

Findings and their disposition:

1. **The result was unreachable from the public API.** The 9.591 ms configuration
   required two internal switches, an explicit strategy, and an explicit precision;
   the shipped default was the pre-028 geometry with `ConcatenatedFixedZM2L`, which
   task 024 measured as winning no steady-state case at any order. *Fixed by user
   direction*: the geometry/window/tensor defaults moved onto the measured optimum,
   `level_radii2` became a public keyword, and precision and M2L strategy are now
   selected per regime from the 024/028 rules rather than being fixed constants — so
   `expansion_order = 3` with no Lamb–Helmholtz now yields the reproduced winner by
   default, while `P >= 12`, Lamb–Helmholtz, and over-gate dense footprints keep
   Float64/precomputed-y. See report.md §6.8 for the rules, their evidence, and the
   accuracy/memory consequences.
2. **report.md opened with the superseded Phase A "no, 91.4 ms" answer.** *Fixed*:
   a standing-answer header, and §1–§5 explicitly marked as the superseded record.
3. **Counting sort: two robustness holes.** It is unstable (atomic cursor), yet the
   surrounding comment still claimed deterministic same-cell order, and flipping
   `RADIX_CUDA_COUNTING_SORT[]` on after construction drove `@inbounds` atomics
   through a length-1 histogram. *Fixed*: comment corrected to state the
   nondeterminism, and `_cuda_counting_sort_ready` falls back unless the buffer
   actually spans the key domain. (The tensor and symmetric knobs already guarded
   this case; this one did not.)
4. **FP16 scaling is not scale-invariant.** The per-column scale comes from the
   operator alone, so the multipole side carries a `max|K|/6e4` factor and a problem
   whose strengths sit far from the benchmark's can silently underflow. *Documented*
   in the `DENSE_CUDA_TENSOR_FORMAT` docstring and report.md §6.8; the arithmetic was
   deliberately left untouched so the reproduced 9.591 ms measurement still applies.
5. **Dead code** in `_hierarchical_scheduled_tables` (two broadcasts superseded by the
   `ifelse` that followed). *Fixed*.
6. **Stages 5–8 were uncommitted** while cycles 1–3 had each been committed.

### Cluster re-gate of the new defaults (2026-08-03)

`cuda_radix_counting_sort_test.jl` was added to the standard preflight; it previously
ran only in the `stage6sort` A/B mode although the counting sort is on by default for
bounded depths.

**Job 13031187 (`stage8repro`) FAILED the hierarchical gate** — 4,015 of 33,368, from
exactly two assertions, both in tests that used "pass no `options`" as a proxy for
"the concat default" and so silently moved to the dense plan:

1. 4,011 route-class parity failures in the geometry loop: device `[1…8]` against the
   host oracle's `[317…324]`. Not a defect — the documented dense convention
   (offset-local class ids, because dense operators are level-shared and apply the
   level diagonal separately) meeting a host oracle that records level-true ids, the
   exact `level-true = (L-2)*noffsets + offset` relation the schedule block already
   asserts through `mod1`. The loop is written for concat and now pins it.
2. 4 `isempty(source_scale)`/`isempty(target_scale)` failures in the block whose own
   comment reads "only the dense strategy carries level scales" — its non-dense
   comparison cache had become dense. Now pins concat.

Every other options-free `RadixFMMCache` call in the CUDA tests was audited: three
blocks (windowing invariance, the resident counter/allocation contract, flat
host-vs-device parity at `P = 6`) now run on the shipped default and assert only
strategy-independent properties, so they were left to exercise it.

**Job 13031482 (`stage8repro`) PASSED**, source manifest `c0afa01083322fb8`:
lifecycle 215/215, concat host parity 37/37, convection 24/24, grid-stride 8/8,
stream overlap 4/4, operator-tile 12/12, rsqrt 2/2, warp-per-pair 16/16, counting
sort 20/20, hierarchical **33,381/33,381** (up from 33,367: the schedule/default
assertions and the new `expansion_order = 3` device default-stack block, which builds
a device cache with no options and checks that Float32 + dense + FP16 tensor over
`sched6-5-5-5` resolves and evaluates against direct). Winner reproduction under the
new defaults:

| format | verdict median [range] ms | gradient rel RMS | M2L ms |
|---|---:|---:|---:|
| FP16 / FP32 accumulate | **9.653** [9.473, 9.689] | 1.0593e-3 | 2.555 |
| BF16 / FP32 accumulate | **9.632** [9.472, 9.690] | 1.0632e-3 | 2.558 |

Both still meet the 10 ms and 1.19e-3 gates; the 0.06 ms against job 13029878's
9.591 ms is inside the recorded run-to-run variance. Contracts unchanged
(`body_uploads=0`, `expansion_host_copies=0`, 0.71 MB host allocation per step,
2.00 GB persistent). `STAGE8_REPRO_EXIT=0`. Data: `fm028-1303118{7,2}.out` and
`cuda_m13h-1-1_20260803-{154247,154359}.csv` plus class companions.

Host verification of the same changes, on Julia 1.12.5 / macOS: full
`julia --project=. --threads=4 -e 'using Pkg; Pkg.test()'` green ("Testing
FastMultipole tests passed", exit 0, no failures in any file), including
`hierarchical_m2l_host_test.jl` 681/681 and `radix_fmm_integration_test.jl` 89/89.
The new host coverage is the default geometry (with exact-once checks at
`ell = 2, 3, 5`), the public `level_radii2` keyword and its three invariants, the
option-selection rules including their platform split and over-gate fallback, the
resolved-choice plumbing, explicit-options bypass, and a Float32-vs-Float64
equal-accuracy check at literature `P = 4`. The same "no options meant concat"
coupling appeared once on the host — the concat-engine window-sizing test — and now
pins that engine. The FP32 fused/tiled parity testsets are pinned to
`DENSE_CUDA_TENSOR_FORMAT[] = :off` so they keep testing their own kernel.

### Third-agent clear-context approval, 2026-08-03 — APPROVED

Performed per `START_HERE.md` §6 by an agent distinct from both the completing
agent and the 2026-08-03 reviewing agent. Evidence verified directly rather
than taken from the notes:

- **Raw job logs.** `fm028-13029878.out` (manifest `42a6c254a11ac8a8`) shows all
  gates green (lifecycle 215/215, host parity 37/37, convection/optimized-kernel
  suites, hierarchy 33,367/33,367) and the reproduced FP16 winner
  `verdict 9.591 ms / grad_err 1.059e-3`; `fm028-13031482.out` (manifest
  `c0afa01083322fb8`, the post-review defaults) shows the expanded hierarchy
  gate 33,381/33,381 plus the new counting-sort suite 20/20 and reproduces
  FP16 at 9.653 ms / 1.0593e-3 and BF16 at 9.632 ms / 1.0632e-3 — inside the
  recorded run-to-run variance of the 9.591 ms record, both inside both gates.
- **Counter/residency contract** re-read from `cuda_m13h-1-1_20260803-154247.csv`:
  `body_uploads=0`, `expansion_host_copies=0`, `route_uploads`/`operator_uploads`
  construction-only, 0.71 MB host allocation per verdict step, 2.00 GB
  persistent device footprint, `ref_cross_check_grad_rel=2.76e-14`.
- **Review fixes confirmed in source.** The counting-sort comment now states the
  atomic-cursor nondeterminism and `_cuda_counting_sort_ready` refuses a
  histogram that does not span the key domain; the `DENSE_CUDA_TENSOR_FORMAT`
  docstring carries the FP16 dynamic-range warning with the `:off` escape; the
  epsilon separator is derived by lattice-shell enumeration (q=7 correctly has
  no shell) instead of special-cased; `level_radii2` is a validated public
  keyword enforcing the non-increasing-with-depth and leaf-agreement invariants
  from the task-025 proof; the untouched default resolves to `near_radius2=5`
  with the `(6,5,…,5)` schedule while an explicit `near_radius2` deliberately
  selects uniform geometry, both documented with the measured accuracy tradeoff
  and the `near_radius2=12` fallback.
- **Independent host re-run** (this machine, Julia project env):
  `radix_fmm_integration_test.jl` 89/89 and `hierarchical_m2l_host_test.jl`
  681/681, matching the recorded counts.

Against the §6 criteria: (1) consistent with the fixed target definition — the
verdict boundary, P=4, and the 1.19e-3 gate never moved, and the gate rejected
hier3/q=3/q=4/uniform-q=5 along the way; (2) correctness is carried by green
H200 and host gates plus the exact-once/coverage machinery extended to all nine
shells and 45 schedules; (3) the performance objective is met and independently
reproduced twice, including once under the shipped defaults; (4) robustness is
strong — failure ledger, retained negative results (symmetric nearfield, TF32,
stream overlap), guarded runtime knobs with fallbacks; (5) changes are confined
to files owned by the resident-lifecycle rows per the placement rules; (6) the
report and docstrings are clearly written, with the superseded Phase A record
explicitly marked.

Residual notes, none blocking: the FP16 default's scale-invariance caveat is a
real (documented) accuracy hazard for problems whose strength scales differ by
orders of magnitude from the benchmark — the docstring warning and `:off`
escape are judged sufficient; and the Stage 5–8 + review surface (~2,900
changed lines across 22 files) was still uncommitted at approval time and
should be committed promptly, since every earlier cycle was committed
individually. Row 029 is now unblocked.
