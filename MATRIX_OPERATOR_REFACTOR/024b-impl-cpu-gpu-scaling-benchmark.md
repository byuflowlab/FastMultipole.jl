# 024b CPU-vs-GPU Scaling Benchmark

## Objective

Measure end-user steady-state speedup versus particle count for:

1. the shipping legacy octree `fmm!` on 64 Julia threads versus the same path
   on one thread; and
2. the device-resident H200 radix lifecycle versus the one-thread legacy path,
   with Float64 primary and Float32 as an additional series.

The seven sizes are `1000`, `3162`, `10000`, `31623`, `100000`, `316228`, and
`1000000`. Literature `P=4` means four retained degrees `0:3`, so both paths
use code `expansion_order=3`.

## Dependencies

- `022-impl-gpu-device-resident-m2l.md`
- `023-impl-production-integration.md`
- `023f-impl-dense-translation-m2l-cuda.md`
- `024-impl-operator-ab-benchmark.md`

## Method

- Compatibility gate: the legacy equal-cell MAC at
  `multipole_acceptance=0.5` is matched to the radix analytic stencil on the
  discrete offset lattice. Squared offset norm 12 remains rejected and 13 is
  accepted. With the fixed 1.02-wide box (`h0=0.51`), literature `P=4`, and
  analytic unit source budget, the radix campaign uses
  `stencil_epsilon(ell) = 0.19542385331034917 * 2^(ell-4)`.
- The derivation and exhaustive `ell=2:7` verifier are a hard pre-benchmark
  gate. A different clean-context agent reviewed the math, production call
  path, normalization, finite domains, and all zero-mismatch results and
  recorded `Result: Approved`.
- Direct references: one dedicated 64-thread job generates the identical
  seeded system for each `n`. Every target is retained through `n=10000`;
  larger cases use the same deterministic 512 sorted indices. CPU and GPU rows
  require the same reference-file SHA-256 checksum and record potential and
  gradient absolute/relative RMS metrics plus maximum gradient-vector error.
- CPU: never call `tune_fmm`. Fix `multipole_acceptance=0.5`,
  `expansion_order=3`, `error_tolerance=nothing`, and `tune=false`.
  Independently for every `(n,threads)`, time the required geometric coarse
  leaf candidates, bracket the winner, then refine the adjacent interval on a
  step-5 grid. Every candidate's minimum and median are written to a separate
  leaf-search audit CSV. Final timing uses two warmups and seven samples and
  includes the legacy per-call tree and interaction-list rebuilds.
- GPU: reuse one `RadixFMMCache` across steps, so construction is measured
  separately. Sweep the occupancy-based `ell` schedule independently per
  `(n,precision)` with `DenseTranslationM2L` and
  `MaterializedYRotationM2L`; record and fall back to
  `PrecomputedFactoredYM2L` only on a dense CUDA OOM. Final timing uses two
  warmups and seven synchronized samples.
- Accuracy audit: at each `n`, compare the selected Float64 GPU relative
  gradient RMS error with both CPU modes using
  `max(e1/e2,e2/e1)` and a `1e-15` floor. Both ratios must be at most 10.
  Float32 ratios are recorded but are not a hard gate.
- Plotting: select the minimum GPU step time over `ell` per `(n,precision)`,
  divide CPU64 and both GPU times by CPU1, annotate selected `ell`, and plot
  the four relative gradient RMS error series in a second panel.

## Artifacts

All paths below are relative to `MATRIX_OPERATOR_REFACTOR/`.

- `theory/024b-mac-stencil-compatibility.md`
- `theory/024b-mac-stencil-compatibility-review.md`
- `scripts/verify_024b_mac_stencil_compatibility.jl`
- `data/cpu_gpu_scaling/references/compatibility_verification.csv`
- `scripts/prepare_024b_direct_references.jl`
- `scripts/cpu_024b_reference_{run,submit}.sh`
- `data/cpu_gpu_scaling/references/direct_reference_n<N>.csv`
- `scripts/benchmark_024b_{common,cpu,gpu}.jl`
- `scripts/{cpu,cuda}_024b_{run,submit}.sh`
- `scripts/fetch_024b.sh`
- `data/cpu_gpu_scaling/`
- `scripts/figures_024a_prepare.jl` (`fig09`)
- `data/figures/fig09_cpu_gpu_scaling.tex`
- `data/figures/tables/fig09_speedup_vs_n.csv`
- `data/figures/fig09_cpu_gpu_scaling.{pdf,png}`

No production `src/` changes belong to this task.

## Verification Notes

- Compatibility verifier: passed locally for all 16,581,375 offsets at the
  largest depth; every `ell=2:7` reports zero mismatches. Independent review:
  `Result: Approved`.
- Reference job: `12894172`, completed on `m12-2-9` in 38 seconds. All seven
  fetched SHA-256 checksums pass. Expected sample counts and deterministic
  indices pass; a local six-thread full `direct!` at `n=1000` matches the
  fetched reference exactly.
- Local smoke (`n=2000`): CPU one-thread and six-thread runs both exercised
  coarse and step-5 refined leaf selection and wrote finite full-schema rows.
  Host radix (`FM024B_DEVICE=false`, `ell=3`, Float64 dense) also wrote a finite
  full-schema row. Every smoke row records `P_literature=4`,
  `expansion_order=3`, the fixed MAC/scaled epsilon, all five errors, and the
  same reference checksum. Smoke files were written under `/tmp`, not campaign
  data.
- Cluster CPU job: `12894176` (submitted; completion pending).
- Cluster H200 job `12894177` failed before its first case because dynamic
  lifecycle loading occurred inside `main()` and the cache constructor saw the
  older fallback method. Lifecycle loading was restored to top level while
  case state remains inside `main()`. Job `12894178` then completed its first
  Float64 case but its next fresh Julia process rejected a `znver3` compiled
  image concurrently written by the AMD CPU job into the shared depot. The GPU
  run now uses a dedicated first `JULIA_DEPOT_PATH` entry so its compiled cache
  cannot race the CPU architecture. Job `12894179` completed 22 cases, then
  dense `n=31623,ell=5,Float64` hit the proactive
  `max_persistent_bytes` memory gate; the handler had recognized only literal
  CUDA OOM strings. The classifier now recognizes the production persistent
  memory-limit diagnostic, and the run wrapper skips already completed case
  keys. Job `12894338` completed six more cases, then
  `n=100000,ell=6,Float64` proved that the prescribed precomputed-y fallback
  also cannot construct the common cache: conservative route storage requested
  223.5 GiB on the 139.8 GiB H200. The wrapper now continues after a failed
  case, writes an explicit failure CSV, and exits nonzero only after attempting
  every remaining scheduled case. Feasible-case collection job `12894403` is
  running and resumed after the 28 completed case keys. Publishing remains
  blocked pending user direction on dropping
  infeasible `ell=6/7` points versus a new out-of-scope production capacity
  refactor. Continuation job `12894403` completed all remaining attempts:
  the campaign has 36 successful GPU rows and an eight-row capacity-failure
  ledger. The final analysis excludes those eight hardware-infeasible points
  and does not change production source.
- Expected final unique rows: 14 CPU and 36 feasible GPU, plus eight explicit
  GPU capacity failures and complete leaf-search audit rows.
- CPU job `12894176` reached its 24-hour limit after 13 of 14 cases. Initial
  24-hour resume job `12914361` was replaced after the scheduler denied an
  in-place extension. The search driver now reloads matching persisted
  leaf-candidate audit rows rather than repeating them. Three-day job
  `12915071` skips the 13 complete cases, reuses the prior CPU1
  `n=1000000` audit, and computes only its missing candidates and final row.
- Three-day CPU job `12915071` completed on `2026-07-28` on `m12-2-21`. It
  reused the persisted `n=1000000` CPU1 coarse candidates, computed the step-5
  refinement (`10/15/20/25/30/35/40`), and wrote the final missing row:
  `cpu1, n=1000000, leaf_size=15, step_seconds_min=40.006065077`,
  `err_gradient_rel_rms=6.073513787e-4`, same reference checksum
  `ec967629...e511bc`. The two superseded resume attempts (`12914361`,
  `12915054`) contributed only short leaf-candidate audit stubs
  (`cpu_leaf_search_m12-2-{27,30}_t1_n1000000.csv`), which are audit records and
  are excluded from the figure loader.
- Final row/schema counts confirmed after fetching: 14 unique CPU cases across
  `cpu_m12-2-9_12894176.csv` and `cpu_m12-2-21_12915071.csv`, 36 unique feasible
  GPU cases, and the eight-row capacity-failure ledger — matching the expected
  counts exactly.
- `fig09` built **strictly** (no `FM024B_ALLOW_PARTIAL`) on `2026-07-28` through
  `scripts/figures_024a_build.sh`. All completeness, per-`n` reference-checksum,
  and Float64 accuracy-ratio gates passed unforced; `fig09_status.tex` emits
  empty macros, so the published figure carries no provisional banner. Figures
  1–8 regenerated unchanged in the same run.

### Clear-context review changes (`2026-07-28`)

A reviewing agent re-derived the compatibility epsilon by hand, re-ran the
`ell=2:7` verifier from its new location (all 16,581,375 offsets at `ell=7`,
zero mismatches at every depth), re-traced every published number from the raw
campaign CSVs to `fig09`, and confirmed all eight ledger entries are genuine
device-capacity failures in `fm024bh200-12894403.out`. It made four changes;
none altered a measured value, and `fig09_speedup_vs_n.csv` is byte-identical
after the rebuild.

The reviewing agent noted that `START_HERE.md` section 6 would normally send an
edited row to a further clear-context agent for approval. The user was told this
and directed approval anyway on `2026-07-28`; under the "current user
instructions are the first source of truth" rule, the row is marked `Approved`
by that direction rather than by a separate agent pass.

1. **Corrected the small-`n` CPU conclusion.** The Results section previously
   attributed the flat `1.00x`/`0.99x` CPU64 points to the leaf search
   collapsing to a single leaf. That is false at `n=3162`, which selected
   `leaf_size=10` (about 316 leaves). The real cause is `src/fmm.jl:1159`:
   the legacy `fmm!` forces `n_threads=1` when
   `n_target_bodies + n_source_bodies < MIN_BODIES = 10000`, i.e. `n < 5000`
   for a self-interaction. This also explains the abrupt jump to `12.5x` at
   exactly `n=10000`, and the audit CSVs confirm the 64-thread timings match
   the 1-thread timings at every candidate leaf size for `n <= 3162`.
2. **Made `fig09`'s lower panel readable.** The in-axis legend hid both CPU
   error series below `n=1e5`, and the `n=1000` CPU point (error exactly `0`)
   was silently dropped by the log axis. The legend moved below the axis,
   explicit `ymin`/`ymax` were set, and an in-axis note explains the missing
   exact/direct point. The caption now also states the `MIN_BODIES` threshold.
3. **Moved the artifacts out of the package root.** `theory/024b-*.md`, the
   four root `scripts/*024b*` files, and `data/cpu_gpu_scaling/` had been
   written to the repository root, creating new top-level `theory/` and `data/`
   directories, contrary to `START_HERE.md`'s artifact locations. References
   now live in `MATRIX_OPERATOR_REFACTOR/data/cpu_gpu_scaling/references/`
   (a subdirectory, so the `fig09` campaign glob is unaffected), and all path
   defaults in the benchmark, verifier, run, submit, and fetch scripts were
   updated to match.
4. **Made the GPU failure ledger self-evidencing.** `cuda_024b_run.sh` labelled
   any nonzero exit `failed_after_fallback`, and its resume check globbed
   `gpu_*.csv`, which also matches the ledger. It now classifies from the
   captured case log and writes `failed_other` for anything that is not a
   memory-capacity failure, the resume check inspects timing CSVs only, and
   the `fig09` loader rejects any non-capacity status rather than publishing it
   as hardware-infeasible.

## Results

Steady-state minimum step time, literature `P=4` (`expansion_order=3`), speedups
relative to the legacy octree on one Julia thread (MAC 0.5, per-case searched
leaf size). GPU rows are the fastest measured `ell` per precision on one H200.

| `n` | CPU1 (leaf, s) | CPU64 (leaf, s) | speedup CPU64 | GPU F64 (`ell`, s) | speedup | GPU F32 (`ell`, s) | speedup |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `1000` | 1000, 0.0076 | 1000, 0.0076 | 1.00 | 2, 0.00290 | 2.62 | 2, 0.00276 | 2.76 |
| `3162` | 10, 0.0442 | 10, 0.0448 | 0.99 | 3, 0.00388 | 11.37 | 3, 0.00384 | 11.51 |
| `10000` | 10, 0.2065 | 35, 0.0165 | 12.50 | 3, 0.00505 | 40.91 | 3, 0.00458 | 45.09 |
| `31623` | 15, 0.8620 | 45, 0.0616 | 14.00 | 3, 0.01023 | 84.26 | 3, 0.00894 | 96.45 |
| `100000` | 15, 3.1967 | 40, 0.2374 | 13.47 | 4, 0.03408 | 93.79 | 4, 0.03104 | 102.98 |
| `316228` | 15, 12.2822 | 40, 0.7426 | 16.54 | 4, 0.07806 | 157.33 | 4, 0.06519 | 188.39 |
| `1000000` | 15, 40.0061 | 20, 2.6865 | 14.89 | 4, 0.42461 | 94.22 | 4, 0.32129 | 124.52 |

Accuracy: at every size the Float64 GPU relative gradient RMS error is *smaller*
than both CPU errors, with ratios `2.7–5.3` against CPU1 and `2.3–5.2` against
CPU64 — inside the `<=10` hard gate. At `n=1000` both CPU searches selected
`leaf_size=n` (exact direct evaluation, zero sampled error), so that size is
exempt from the ratio gate by the rule adopted on `2026-07-25`; its errors are
still recorded and plotted. Float32 tracks Float64 to three digits everywhere,
confirming that at `P=4` the stencil truncation, not the arithmetic precision,
sets the error.

Selected strategy and fallback summary: `DenseTranslationM2L` was fastest for
every one of the 14 selected `(n, precision)` points. `PrecomputedFactoredYM2L`
was exercised only as the prescribed OOM fallback at the finest feasible grids
(`ell=5` for `n>=31623` Float64 / `n>=100000`), where it is 1.5–2 orders of
magnitude slower and never selected. Eight `ell=6/7` attempts at
`n>=100000` could not be constructed at all: conservative route storage requested
up to 223.5 GiB against the H200's 139.8 GiB, so those points are recorded in the
failure ledger and excluded from selection as hardware-infeasible rather than
slow. Enabling them would require an out-of-scope production capacity refactor.

Scaling conclusion: the legacy octree's 64-thread scaling saturates near
`13–17x` (about `21–26%` parallel efficiency) and is flat in `n` above `1e4`,
while the resident H200 lifecycle rises from `2.6x` at `n=1000` to a peak of
`157x` (Float64) / `188x` (Float32) at `n=316228`. Below `n=5000` the CPU 1- and
64-thread paths are indistinguishable because the legacy `fmm!` disables threading
outright: `src/fmm.jl:1159` forces `n_threads=1` whenever
`n_target_bodies + n_source_bodies < MIN_BODIES = 10000`, which for a
self-interaction means `n < 5000`. That threshold — not the leaf search — explains
the `1.00x` / `0.99x` points at `n=1000` and `n=3162` and the abrupt jump to
`12.5x` at `n=10000`. In particular the `n=3162` case selected `leaf_size=10`
(about 316 leaves), not a single leaf, and its 64-thread timings match the
1-thread timings at *every* candidate leaf size in the audit CSV. The dip at
`n=1e6` (`94x` Float64) is a grid-resolution artifact:
the best feasible `ell=4` grid is already too coarse at that particle count, and
the finer `ell=5/6` grids are respectively far slower or unconstructible on this
device — so the `n=1e6` GPU point is capacity-limited, not compute-limited, and is
the single clearest lever identified for follow-on work.

### Interim figure review and snapshot (`2026-07-25`, superseded)

**Superseded** by the strict final build recorded above; retained for history.


- User explicitly authorized bypassing the task-completion gate to publish the
  currently available speedup comparison. This is an interim artifact only;
  the task remains not Done and not Approved.
- Review found that fig09's input glob also selected `gpu_failures_*.csv`,
  causing the failure ledger to be parsed as timing data. The loader now
  excludes failure ledgers.
- The first fig09 compilation also exposed invalid `\thisrow` use in
  `nodes near coords`. The selected-`ell` labels now use pgfplots explicit
  point metadata.
- Review found that the Float64 ratio gate was undefined in practice for an
  exact/direct CPU result: at `n=1000`, both CPU searches selected
  `leaf_size=n`, producing zero sampled-direct error, while the GPU FMM error
  was nonzero. The ratio gate now applies only when the corresponding CPU case
  uses an FMM decomposition (`leaf_size<n`); all errors remain recorded and
  plotted.
- fig09 supports an explicit `FM024B_ALLOW_PARTIAL=true` preparation mode. It
  uses only particle counts with CPU1, CPU64, Float64 GPU, and Float32 GPU
  measurements, selects the fastest currently successful `ell`, and marks the
  canonical figure visibly provisional. Strict complete-data validation
  remains the default.
- Snapshot taken while CPU job `12894176` and feasible-case GPU job `12894403`
  were still running. It contains 12 unique CPU cases and 36 successful GPU
  cases overall; six particle counts (`1000` through `316228`) have all four
  comparison modes. CPU1 at `n=1000000` is not yet available.
- The provisional largest-common-size result (`n=316228`) is `16.54x` for
  CPU64, `157.33x` for H200 Float64, and `188.39x` for H200 Float32, all versus
  legacy CPU1. Canonical provisional PDF/PNG and the six-row table were built
  successfully; a strict preparation still rejects the snapshot (`12/14` CPU
  cases), as intended.

Completion condition (both jobs fetched, all row-count/schema and accuracy gates
passing, and fig09 building strictly) was met on `2026-07-28`; the row is marked
Done. Clear-context final task approval remains a separate later approval by a
different agent.
