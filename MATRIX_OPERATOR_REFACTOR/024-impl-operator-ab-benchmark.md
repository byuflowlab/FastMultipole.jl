# 024 Implementation Resident M2L Strategy End-to-End Benchmark

## Objective

Measure the four production-resident M2L execution strategies through the fully
integrated FMM and select defaults by platform and workload regime:

1. **Whole-slab block-diagonal concat baseline** — the existing
   `ConcatenatedFixedZM2L` materialized-lineage resident implementation.
2. **Per-degree factored `U_n D_n V_n`** — the optimized host/CUDA factored
   resident implementations from `023a`/`023b`.
3. **Precomputed per-angle `M_n(theta)`** — `PrecomputedFactoredYM2L` from
   `023c`/`023d`, with `M_n(theta) = U_n D_n(theta) V_n` and separate `Z_phi`,
   z translation, Lamb-Helmholtz coupling, and scatter.
4. **Fully dense per-displacement `S(Delta r)`** — `DenseTranslationM2L` from
   `023e`/`023f`, containing the complete coefficient-space M2L map.

The traditional per-column `MaterializedYRotationM2L` path that reconstructs
`Ts(theta)` remains a correctness/oracle reference. It is not a resident
performance candidate and must not be presented as one of the four strategies.

There is no requirement for one universal winner. Select defaults from measured
crossover regimes and permit different recommendations for CPU versus GPU,
expansion order, batch size, class occupancy, precision, and Lamb-Helmholtz mode.

## Dependencies

- `023a-impl-factored-resident-m2l-host.md` and
  `023b-impl-factored-resident-m2l-cuda.md` (optimized per-degree factored path)
- `023c-impl-precomputed-y-m2l-host.md` and
  `023d-impl-precomputed-y-m2l-cuda.md` (optimized precomputed-y path)
- `023e-impl-dense-translation-m2l-host.md` and
  `023f-impl-dense-translation-m2l-cuda.md` (optimized full dense path)
- `013-impl-axis-swap-operators.md` (reconstructed-`Ts(theta)` oracle)
- `013c-impl-factored-rotation-alignment.md` (plain `U_n`/`V_n` convention)
- `015-impl-axis-swap-benchmarks.md` (earlier isolated measurements)
- `019-impl-operator-performance-tuning.md` (integrated tuning evidence)
- `022-impl-gpu-device-resident-m2l.md` (GPU lifecycle and stage counters)
- `023-impl-production-integration.md` (production FMM/cache integration)

## Required Reading

- `START_HERE.md`
- All dependency task files above, including optimization and approval notes
- The `008c` performance baseline, `015` harness/results, and `019`/`022`/`023`
  benchmark artifacts
- The benchmark scripts and raw before/after data produced by `023a` through
  `023f`

## Artifacts or Production Surface

- Benchmark scripts under `MATRIX_OPERATOR_REFACTOR/scripts/` and recorded raw
  and summarized data under `MATRIX_OPERATOR_REFACTOR/data/`.
- A recommendation and crossover table in this task file for consumption by
  `024a` and the `019a` final review.
- No production hot-path changes. Each candidate must enter this task already
  functional, optimized, and selectable through the resident lifecycle.

## Correctness Gate

Before timing a case, confirm that every supported resident strategy agrees with:

- direct evaluation at the established FMM potential/gradient tolerances;
- the reconstructed per-column `Ts(theta)` composition oracle; and
- the other resident strategies within precision-appropriate tolerances.

Exercise Float32 and Float64, Lamb-Helmholtz on and off, empty route sets, partial
final batches/chunks, repeated time steps, and fixed-domain capacity reuse. A
strategy/case that fails correctness or lifecycle invariants is disqualified and
reported as such, not assigned a timing win.

Confirm the `RadixFMMCache` no-reallocation contract and, on CUDA, that
`route_uploads` and `operator_uploads` remain constant after construction and
`expansion_host_copies == 0`.

## Benchmark Matrix

Benchmark enough of the following dimensions to identify meaningful crossovers,
using a documented sampling design when the full Cartesian product is too large:

- expansion order `P`;
- problem size `N` and resulting number of occupied cells/routes;
- displacement/angle class occupancy and skew, including tiny, sparse, and dense
  class batches;
- Float32 and Float64;
- Lamb-Helmholtz disabled and enabled;
- single-thread and representative multi-thread CPU BLAS configurations; and
- device-resident GPU execution on an H200.

For every candidate and regime, record separately:

1. operator/cache construction time and any construction-time upload;
2. persistent operator, metadata, expansion, and scratch memory;
3. steady-state M2L time and allocations;
4. full resident lifecycle time with per-stage breakdown; and
5. end-to-end production FMM time, including update/finalize costs as applicable.

Also record peak memory, launch/GEMM counts where available, warmup protocol,
sample count/statistic, Julia/package/BLAS/CUDA versions, CPU/GPU model, thread
settings, and cache/reuse assumptions.

## Fair-Comparison Rules

- Use the optimized implementation and measured crossover policy delivered by
  each of `023a` through `023f`; do not benchmark known functional baselines as
  though they were final candidates.
- Use identical bodies, domain, stencil, route ordering, order, precision,
  channel mode, warmup, and timing method across candidates in a case.
- Separate one-time construction from steady-state timing. Report amortized
  lifecycle break-even points where construction or persistent memory changes
  the practical winner.
- Report memory infeasibility explicitly for dense operators rather than silently
  shrinking only their workloads.
- Keep the reconstructed per-column `Ts(theta)` path in correctness checks and,
  if useful, diagnostic timings labeled as oracle-only; exclude it from resident
  winner selection.

## Cross-Machine CPU and H200 Requirements

Host results must include at least one non-macOS/different-BLAS benchmark host.
Run single-thread and multi-thread BLAS cases by setting the BLAS thread count at
process start (for example `OPENBLAS_NUM_THREADS`), following the established
cluster scripts. Local macOS data may be included only as supplementary laptop
reference. This closes the `016b` batched-GEMM watch item with measured evidence.

CUDA results must use the device-resident lifecycle on an H200. Preserve and
report the lifecycle stage breakdown (`t_b2m`, `t_m2m`, `t_m2l`, `t_l2l`,
`t_l2b`) plus update/finalize timing so an M2L win is not confused with an
end-to-end win and remaining bottlenecks stay visible for `019a`.

Use the established `orc`/Slurm workflow and cluster environment conventions.
**Show the user every Slurm script and get explicit permission before submitting
any job.**

## Deliverables

- Raw and summarized measurements for construction time, persistent/peak memory,
  steady-state M2L, full lifecycle, and end-to-end FMM across the benchmark
  matrix.
- Correctness and lifecycle-invariant results for all timed candidates.
- CPU and GPU crossover tables identifying which resident strategy wins by `P`,
  batch/class occupancy, precision, and LH regime, including memory feasibility
  and construction-amortization constraints.
- Recommended production defaults and explicit fallback/crossover rules. Mixed
  winners are expected when supported by the data.
- A concise statement of unresolved regimes or noisy/inconclusive crossovers for
  `024a` visualization and the `019a` final review.

## Verification

- Benchmark scripts reproduce a smoke-size subset locally before cluster runs.
- All four resident candidates pass the correctness gate on every regime in
  which they are ranked.
- Float32/Float64, LH on/off, empty routes, partial batches, repeated steps, and
  fixed-domain reuse are represented in validation.
- Host no-reallocation and CUDA transfer-counter invariants are asserted during
  benchmark validation, not inferred from earlier tasks.
- Non-macOS single-/multi-thread CPU and H200 raw results include full machine and
  software metadata.
- Recommendations are traceable to recorded data and include construction and
  memory tradeoffs, not only isolated M2L kernel time.

## Approval Notes

**Approved (clear-context review, `2026-07-24`, different agent).**  The review read
only `START_HERE.md`, this task file, its listed artifacts
(`scripts/benchmark_024_common.jl`, `benchmark_024_host.jl`,
`benchmark_024_cuda.jl`, `summarize_024.jl`, `cpu_024_run.sh`, `fetch_024.sh`,
`data/operator_ab_benchmark/**`), and the two Slurm `.out` logs.

### Independent verification

Every headline claim was re-derived directly from the 504 raw CSV rows with an
independent script rather than by re-running the shipped summarizer:

- 126 cases x 4 strategies = 504 rows; 483 eligible, 21 infeasible, 0
  disqualified; all eligible rows `correctness_pass=true` and
  `expansion_host_copies=0`.
- The 21 infeasible rows are exactly the 18 Float32/`P=12` dense rows plus the 3
  clustered Float64/`P=12`/LH-on dense rows
  (`operators=31,594,575,200` bytes, above the 12 GiB gate).
- CPU winners: dense 62 M2L / 63 complete step; precomputed-y 22 / 21; concat and
  factored zero.  H200: 21/21 M2L split, 20 dense / 21 precomputed-y / 1 factored
  complete step, with exactly 6 of 42 winners flipping between M2L and full step.
- Both recommendation tables reproduce row for row, including the named boundary
  cases (CPU `P=12` Float64 uniform `N=150` BLAS-1 LH-off dense step win; H200
  `P=12` Float64 LH-on `N=150` factored step win).
- Break-even ranges (CPU 4--35 / 6--109 / 52--220 / 47--18,930; H200 542--18,208
  at `P=4` and 4,120--733,902 at `P=8`; precomputed-y vs concat/factored 0--445),
  dense operator memory ranges, H200 plan storage (factored 1.1 MiB,
  precomputed-y 2.0 MiB, dense up to 7,479.3 MiB), clustered route/class counts,
  and the three noisy `>10%` IQR/median rows all reproduce.
- CUDA counters: `route_uploads=2` and `operator_uploads=1` constant across all
  161 eligible CUDA rows, `construction_uploads=3`, `expansion_host_copies=0`.
- Cluster test counts in the `.out` logs match the log's claims (63/63;
  51,065 + 694 = 51,759; 73/73; 100 + 4 = 104; CUDA 208/208, 37/37,
  416,797/416,797), both jobs carrying the stated manifest checksum.
- `src/` was last modified `2026-07-23` (task `023f`) while the 024 artifacts are
  `2026-07-24`, confirming the "no production hot-path changes" claim.
- Methodology holds where it matters: the correctness gate combines direct
  evaluation, the full-route reconstructed-`Ts` oracle, and cross-strategy output
  parity, with the oracle excluded from ranking; identical bodies/routes/warmup
  per case; BLAS threads fixed before process start; construction measured once
  after warm specialization; and `summarize_024.jl` genuinely enforces schema,
  duplicate keys, four-strategy coverage, finite eligible timings, explained
  non-eligible rows, and the complete matrix under `FM024_REQUIRE_COMPLETE=1`.

### Corrections to the Implementation Log (data unchanged)

1. "mean occupancy 17.4--23.0" — the recorded minimum is `17.05`.
2. "maximum 93--190" — the recorded minimum-of-maxima is `87`.
3. "construction winners were factored in 27/42 H200 cases and concat in 13/42" —
   precomputed-y wins the remaining 2/42.
4. The dense memory table omits the feasible `P=12` LH-on uniform Float64 rows
   (`3,305.4 MiB`, 9 rows); it lists only `P=12` LH-off and the infeasible
   clustered LH-on case.

None of these changes a recommendation or a ranking; `024a` and `019a` should use
the corrected values.

### Caveats for downstream readers of the raw CSVs

- `nonempty_classes` and all occupancy columns are `0` on concat and factored rows
  (`fm024_class_counts` has no branch for the concat plan).  Occupancy is a
  case-level property and must be read from the precomputed-y or dense rows.
- `persistent_bytes`/`scratch_bytes` are plan-reported for dense and
  precomputed-y but fall back to `Base.summarysize(state.scratch)` for concat and
  factored, so those columns are not comparable across strategies.  The
  conclusions above correctly avoid that comparison; the columns still invite it.
- CPU `peak_bytes` is `Sys.maxrss()`, a process-wide high-water mark shared by all
  four strategies in one process, so it is not a per-strategy peak; the CUDA
  `peak_bytes` device delta is per-strategy.  CPU multi-thread coverage is BLAS
  threads only (`julia_threads=1`), which satisfies the task wording.
- Dense Float32/`P=12` is short-circuited by a hard-coded skip rather than an
  observed throw.  It matches the tested non-finite-materialization rejection
  established and approved in `023e`/`023f`, so this is a transparency nit, not a
  correctness problem.

## Implementation Log (2026-07-24)

The task-024 campaign was implemented and completed without changes to
production code, public APIs, dependencies, or current defaults.  The user
approved both displayed Slurm payloads before submission.

### Sampling and isolation

- `benchmark_024_host.jl` and `benchmark_024_cuda.jl` run one isolated
  `(distribution, precision, P, LH, N, ell, process/device)` case and compare the
  four resident strategies on identical deterministic bodies and routes.
- The core matrix is `P = 4, 8, 12`, `N = 150, 2000, 20000`, Float32/Float64,
  LH off/on, `ell=3`; the skew matrix is deterministic clustered `N=20000`,
  Float64, LH off/on, every `P`, `ell=4`.
- CPU BLAS-1 and BLAS-64 are separate Julia processes with the thread setting
  fixed before startup.  CUDA cases use the device-resident lifecycle.
- Every timing uses two untimed warmups and seven samples and records median,
  minimum, and IQR.  Correctness constructions warm specializations before the
  single measured construction for each strategy.
- The reconstructed per-column `Ts(theta)` oracle consumes the full valid route
  prefix in bounded chunks.  This gives full-route coverage even for large
  cases, stronger than the planned class-stratified large-case sample.
- Dense Float32/P=12 is emitted as an explained `infeasible` row and never
  ranked.  Other memory-limit or non-finite construction failures are also
  explicit rows.

### Artifacts and validation

- Shared implementation/schema: `scripts/benchmark_024_common.jl`
- Case drivers: `scripts/benchmark_024_host.jl`,
  `scripts/benchmark_024_cuda.jl`
- Slurm payloads: `scripts/cpu_024_run.sh`, `scripts/cuda_024_run.sh`
- User-side synchronization/submission: `scripts/cpu_024_submit.sh`,
  `scripts/cuda_024_submit.sh`
- Fetch plus complete-coverage validation: `scripts/fetch_024.sh`
- Dependency-free validation/summarization: `scripts/summarize_024.jl`
- Raw/summary root: `data/operator_ab_benchmark/`

The common CSV records provenance, machine/software settings, matrix identity,
correctness status/errors, route/class occupancy and skew, construction and
upload cost, operator/metadata/expansion/scratch/persistent/peak memory, all
five lifecycle stages, update/lifecycle/finalize/recurring costs and
allocations, and transfer counters.  Launch/GEMM counts are `-1` where the
current lifecycle exposes no counter rather than being guessed.  The
summarizer rejects missing columns, duplicate/mismatched cases, non-finite
eligible timings, unexplained infeasible/disqualified rows, missing strategy
coverage, failed correctness status, and nonzero expansion host copies.  With
`FM024_REQUIRE_COMPLETE=1`, it also requires the entire CPU BLAS-1/64 and H200
matrix.  It produces raw indexes, case rankings, CPU/GPU crossover tables,
construction break-even steps, memory feasibility, and noisy/unresolved notes.

The submit scripts record `HEAD`, tree, dirty-worktree state, and a SHA-256
manifest plus manifest checksum for the synchronized `src`, `test`, and script
snapshot.

### Local preflight

The local smoke campaign covered every strategy in Float32/Float64 with LH
off/on at `P=4`, `N=150`, `ell=3`, using the final two-warmup/seven-sample
protocol.  All 16 rows were eligible and the dependency-free summarizer
validated the schema, case grouping, timings, and counters.

Focused host tests:

- radix production integration: 63/63 pass;
- resident lifecycle plus mock time stepping: 51,759/51,759 pass;
- precomputed-y resident M2L: 73/73 pass;
- dense translation M2L: 104/104 pass.

The final non-macOS CPU campaign was job `12892009` on AMD EPYC 7763 node
`m12-2-17`; the H200 campaign was job `12892013` on `m13h-1-1`.  Both used
source-manifest checksum
`c37ecc3acdd9ca70f873f9c4852be22a22e46b69f38f6e650792fca52879e31f`.
An earlier CPU preflight, a superseded partial CPU run, and the first CUDA
driver attempt were preserved remotely under `aborted/`; none of their rows is
present in the final raw directory.

Cluster validation passed:

- CPU focused integration 63/63, lifecycle/time stepping 51,759/51,759,
  precomputed-y 73/73, and dense 104/104;
- CUDA lifecycle 208/208, concat host parity 37/37, and CUDA integration
  416,797/416,797;
- final strict coverage: 126 cases, 504 rows, 483 eligible, 21 explicitly
  infeasible, and zero disqualified;
- zero eligible rows with `expansion_host_copies != 0`; recurring execution
  asserted constant `route_uploads` and `operator_uploads` in every timed CUDA
  case;
- the known concat scalar-staging allocation remained observable in all 126
  concat rows and is reported, not repaired.

Maximum direct errors remained below the established gates:

| Platform / precision | Max potential error (phi cases) | Max gradient error |
| --- | ---: | ---: |
| CPU Float64 | `6.49e-8` | `5.25e-6` |
| CPU Float32 | `8.79e-7` | `5.33e-4` |
| H200 Float64 | `6.49e-8` | `5.25e-6` |
| H200 Float32 | `1.40e-7` | `5.34e-4` |

### Measured crossover recommendations

The tables below use complete recurring-step time as the production criterion.
Kernel-only M2L winners are recorded separately in `cpu_crossovers.csv` and
`gpu_crossovers.csv`.  No runtime dispatch or current default was changed.

#### CPU recurring-step rule

| Regime | Recommended steady-state strategy |
| --- | --- |
| `P=4`, all measured precision/LH/size/occupancy cases | dense |
| `P=8`, LH off | dense |
| `P=8`, LH on, uniform `N=150` | precomputed-y |
| `P=8`, LH on, uniform `N>=2000` or clustered `N=20000` | dense |
| `P=12`, Float32 | precomputed-y (dense unsupported) |
| `P=12`, Float64 uniform `N=150` | precomputed-y; BLAS-1/LH-off is a boundary where dense wins the full step despite losing M2L |
| `P=12`, Float64 uniform `N>=2000` | dense |
| `P=12`, clustered Float64 LH off | dense |
| `P=12`, clustered Float64 LH on | precomputed-y (dense exceeds the memory gate) |

BLAS-1 and BLAS-64 selected the same rule except the noted small
Float64/P12/LH-off boundary.  Across 84 CPU cases, dense won M2L 62 times and
the complete step 63 times; precomputed-y won 22 and 21 times respectively.
Neither concat nor factored won a CPU M2L or recurring-step case.

Dense construction must be amortized.  Against precomputed-y, measured CPU
break-even steps were:

- uniform Float64 `N>=2000`: 4--35 steps;
- clustered Float64 `N=20000`: 6--109 steps;
- uniform Float32 `N>=2000`, where dense is supported: 52--220 steps;
- uniform `N=150`: 47--18,930 steps, so precomputed-y/concat is normally the
  practical choice despite a possible dense steady-state win.

#### H200 recurring-step rule

| Regime | Recommended steady-state strategy |
| --- | --- |
| `P=4`, uniform | dense, except LH-off `N=150` where precomputed-y wins the complete step |
| `P=4`, clustered Float64 | dense for LH off; precomputed-y for LH on |
| `P=8`, LH off | dense |
| `P=8`, LH on, uniform `N=150` | dense for the complete step although precomputed-y wins M2L |
| `P=8`, LH on, uniform `N>=2000` or clustered | precomputed-y |
| `P=12` | precomputed-y; the single Float64/LH-on/`N=150` complete-step winner was factored |

The H200 M2L kernel split is especially clean: dense wins every `P=4` case,
dense wins `P=8` with LH off, precomputed-y wins `P=8` with LH on, and
precomputed-y wins every `P=12` case.  Full-step overhead changes six of the 42
case winners.  Overall M2L winners split 21 dense / 21 precomputed-y; complete
steps split 20 dense / 21 precomputed-y / one factored.

Dense construction on H200 is expensive enough that precomputed-y is the
recommended general default unless a fixed cache is known to be very
long-lived.  Dense-versus-precomputed-y break-even was 542--18,208 steps at
`P=4` and 4,120--733,902 steps at `P=8`.  Where precomputed-y is the
steady-state winner, it amortizes against concat/factored within 0--445 steps
in the measured cases.  For truly one-shot execution, construction winners
were factored in 27/42 H200 cases and concat in 13/42, so the raw
construction-amortization table should be consulted rather than paying for a
dense plan.

#### Memory and fallback rule

Dense operator payload is the binding constraint:

| Dense regime | Measured operator range |
| --- | ---: |
| `P=4`, LH off / on | 3.8--130.3 MiB / 18.1--771.9 MiB |
| `P=8`, LH off / on | 78.2--1,478.4 MiB / 389.7--7,369.9 MiB |
| `P=12`, LH off | 708.6--6,464.8 MiB |
| `P=12`, clustered LH on | 30,130.9 MiB, infeasible under the 12 GiB gate |

All Float32/P12 dense rows are intentionally unsupported and unranked.  The
clustered Float64/P12/LH-on dense plan is infeasible on both platforms; chunk
reduction cannot remove its operator storage.  Fall back to precomputed-y in
either case.  On H200, factored and precomputed-y plan storage stayed below
1.1 MiB and 2.0 MiB respectively, versus up to 7,479.3 MiB for feasible dense
plans.

### Occupancy and unresolved regimes

The clustered matrix measured 221,504--362,800 routes, 12,708--15,778 nonempty
classes, mean occupancy 17.4--23.0, 95th-percentile occupancy 57--72, maximum
93--190, and skew 1.22--2.04.  It confirms that `N` alone is not a sufficient
selector: LH and order can reverse dense/precomputed-y winners at similar mean
occupancy, while dense storage scales with all displacement classes.

Only three eligible CPU rows exceeded 10% M2L IQR/median:

- concat Float64/P8/LH-on/`N=20000` (`0.113`);
- dense Float32/P8/LH-on/`N=2000` (`0.195`);
- dense Float64/P12/LH-on/`N=20000` (`0.713`).

Treat those exact boundaries as unresolved/noisy and prefer precomputed-y when
construction or memory is material.  Full raw indexes, rankings, crossover
tables, break-even steps, memory feasibility, and notes are under
`data/operator_ab_benchmark/summary/`.

Task 024 is complete.  `Approved` remains unchecked for a different
clear-context agent.
