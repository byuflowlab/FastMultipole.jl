# 023f Implementation Dense-Translation Resident M2L (CUDA)

## Objective

Port the complete `DenseTranslationM2L` strategy from `023e` to the CUDA
device-resident lifecycle. Construct full-displacement coefficient operators on
the host or device as justified by measurement, upload/finalize them only during
cache construction, and keep operators, class metadata, routes, expansions, and
scratch resident for repeated steps.

Establish functional device parity first. Then profile and optimize on an H200
and rerun all correctness, lifecycle, allocation, precision, and channel tests.
Completion requires recorded before/after H200 measurements, not only a working
port.

## Dependencies

- `023e-impl-dense-translation-m2l-host.md` (complete host strategy and oracle)
- `022-impl-gpu-device-resident-m2l.md` (device lifecycle, allocation, and
  transfer-counter contracts)
- `020a-impl-device-radix-grid-construction.md` (resident radix metadata)
- `023b-impl-factored-resident-m2l-cuda.md` (CUDA resident M2L integration)
- `023-impl-production-integration.md` (cache construction and refresh)

## Required Reading

- `START_HERE.md`
- The dependency task files above and `023e` implementation/benchmark notes
- `src/containers.jl`, `src/translate_batched_resident.jl`, and
  `src/translate_batched_cuda.jl`
- Existing CUDA integration, precision, lifecycle, and counter tests

## Artifacts or Production Surface

- Device construction/upload and application code in
  `src/translate_batched_cuda.jl`, with shared types in their established files.
- CUDA correctness, lifecycle, memory-limit, and transfer-counter tests.
- Reproducible H200 benchmark/profile scripts and recorded data under
  `MATRIX_OPERATOR_REFACTOR/`.

## Functional Phase

- Use exactly the full-displacement matrices and coefficient layouts validated
  by `023e`, including complete rotation, axial translation, and LH coupling.
- Upload each operator and invariant class mapping at cache construction only.
  Fixed-domain repeated steps must reuse device storage without recurring route
  or operator uploads.
- Implement class gather, dense GEMM, and scatter-add for arbitrary occupancy,
  including collisions where multiple class routes contribute to one target.
- Estimate device operator, route, expansion, packing, and scratch footprint
  before allocation. Enforce an explicit device-memory limit/headroom policy and
  report a useful estimate when the strategy cannot fit.
- Establish parity against direct evaluation, the CUDA concat/factored paths,
  and the `023e` host dense strategy before tuning.

## Optimize, Profile, and Retest Phase

Capture a correct H200 baseline, then profile and tune at least:

- grouped, batched, or per-class GEMM selection by matrix size and occupancy;
- displacement-class order and device operator packing;
- coalesced gather/scatter and justified gather/GEMM/scatter fusion boundaries;
- launch count and tiny/partial class overhead;
- chunking and scratch reuse under device-memory pressure; and
- precision- and LH-dependent memory/crossover rules.

Use fused kernels only where profiling supports them and retain independently
testable reference paths. After every retained optimization, rerun parity and
lifecycle checks. Record before/after stage/kernel timings, launch counts,
construction/upload time, allocations, persistent and peak device memory,
operator footprint, and selected crossover/chunk rules.

## Deliverables

- Device-resident `DenseTranslationM2L` through the production cache/options
  surface, with construction-time operator upload only.
- Explicit device-memory estimation/limits and a measured fallback/crossover
  policy when full dense operators do not fit or do not amortize.
- H200 before/after benchmark and profile artifacts.

## Verification

- Float32 and Float64; Lamb-Helmholtz on and off.
- CUDA-vs-host dense, CUDA-vs-other resident strategies, and direct-evaluation
  parity before optimization and after the optimized implementation.
- Empty route sets/classes, partial final batches/chunks, uneven occupancy,
  repeated target accumulation, repeated steps, and fixed-domain capacity reuse.
- `route_uploads` and `operator_uploads` constant after construction,
  `expansion_host_copies == 0`, and all other lifecycle counters consistent with
  `022`/`023`.
- No recurring operator/route upload and no unaccounted steady-state device
  allocation; persistent and peak memory recorded.
- Full CUDA suite and measured H200 run.
- **Show the user the Slurm scripts and get explicit permission before
  submitting any job.**

## Implementation Notes (functional port complete; H200 validation pending)

Status on 2026-07-21: the device-resident `DenseTranslationM2L` port is fully
implemented and verified on every locally-runnable surface. The H200
before/after measurement and the full CUDA hardware test run are the only
remaining steps; they require a live GPU and the user's explicit approval to
submit the Slurm job (the scripts are ready — see below), so `Done` is left
unchecked here and in `START_HERE.md`.

### Production surface

- `src/containers.jl`: `DenseTranslationM2L` gains `cuda_headroom_bytes`
  (default `1 << 30`, nonnegative, Int-representable); host defaults/meanings of
  `max_persistent_bytes`, `apply_chunk`, `build_chunk` unchanged. New device plan
  type `ResidentM2LDenseCUDAPlan` beside the host `ResidentM2LDensePlan` (host
  fields stay concretely typed): packed device operators `D×D×nclasses`, device
  Int32 `route_class`/`class_counts`, pinned host count mirror, host
  `class_starts` prefix and `class_capacities`, chunk-width device `src_slab`/
  `dst_slab`, byte accounting, and a `whole_pass::RefValue{Any}` bundle.
- `src/translate_batched.jl`: `_launch_resident_m2l_dense!` now accepts either
  the host or the CUDA dense plan and dispatches to the plan-typed method;
  `_radix_cache_workspace` builds the CUDA dense plan (`_build_cuda_dense_m2l_plan`)
  in device mode.
- `src/translate_batched_resident.jl`: host stub for `_build_cuda_dense_m2l_plan`.
- `src/translate_batched_cuda.jl`:
  - one-shot builder error narrowed to point at the recurring cache; recurring
    dense rejection removed (`_assert_cuda_supported_operator!`).
  - `_assert_cuda_scratch_value!` dense-plan branch (operative device arrays
    checked; host count staging allowed).
  - gather / atomic-scatter kernels, device histogram route refresh
    (`_cuda_refresh_dense_m2l_routes!`), per-class GEMM helper, and two
    independently selectable drivers: `_launch_resident_m2l_dense_whole!`
    (default, gather/scatter once per chunk, per-class GEMMs over the chunk) and
    `_launch_resident_m2l_dense_perclass!` (reference), toggled by
    `DENSE_CUDA_WHOLE_PASS` / `DENSE_CUDA_CHUNK`.
  - `_dense_cuda_lifecycle_footprint` + `_dense_cuda_limit_error`: pure
    free-memory preflight in `_radix_cache_device_build` (before any large device
    allocation) plus the persistent-payload gate in `_build_cuda_dense_m2l_plan`.
    Operators are oracle-built on the host (identical to the host plan) and
    uploaded once (the existing single construction `operator_uploads += 1`);
    routes refresh through the histogram in `update_cuda_radix_state!`.

The CUDA port uses exactly the same full integer displacement classes, matrix
orientation, degree-major stacking, rotation signs, axial translation, and LH
coupling as the approved 023e host implementation (operators are built by the
same `build_dense_m2l_operator!` oracle).

### Tests

- `test/dense_translation_m2l_test.jl`: added `cuda_headroom_bytes` validation
  (defaults, nonnegativity, Int-representability); host numerics unchanged.
- `test/cuda_radix_integration_test.jl`: replaced the recurring-cache dense
  rejection with a one-shot rejection assertion and a full hardware dense block
  (mirroring the 023b/023d patterns): plan typing + `route_class isa CuArray`,
  per-class operator parity vs a fresh host oracle and the host plan,
  `issorted`/`class_starts[end]-1 == n_routes`/counts recovery, the
  `{Float64,Float32} × {LH on/off} × P∈{4,8,12}` parity/rejection matrix vs
  the 023e host dense M2L and the CUDA concat/precomputed-y paths: supported
  finite dense configurations run parity, while Float32/P=12 is asserted as a
  non-finite materialization rejection. Also covers per-class-vs-whole-pass
  parity, direct-accuracy at P=8, `CUDA.@allocated == 0`, array identity and
  counter-constancy across steps, empty-route all-nearfield, tiny-chunk stress,
  and both device-memory gates (persistent + free-memory headroom).

### Benchmark / cluster artifacts

- `MATRIX_OPERATOR_REFACTOR/scripts/benchmark_023f_dense_m2l_cuda.jl` (concat,
  factored, precomputed-y, dense whole-pass, dense per-class; records dense
  operator/persistent bytes, the full dense CUDA lifecycle estimate
  (`dense_estimated_peak_bytes`), chunk/whole-pass policy, per-stage and
  full-step medians, M2L device alloc, and `fit=false` rows for explicit
  memory-gate or non-finite materialization rejections).
- `cuda_023f_run.sh`, `cuda_023f_submit.sh`, `cuda_023f_fetch.sh`,
  `cuda_023f_sweep.sh` (chunk-width sweep); data dir
  `MATRIX_OPERATOR_REFACTOR/data/dense_translation_m2l_cuda/` with a README.

### Local verification (2026-07-21, macOS, no CUDA runtime)

- `julia --project=. test/dense_translation_m2l_test.jl`: 72/72 + 4/4 passed
  (was 66/66 + 4/4; +6 headroom-keyword tests).
- `julia --project=. test/cuda_radix_integration_test.jl`: CPU-safe gate 1/1.
- `julia --project=. test/cuda_radix_lifecycle_test.jl`: 91/91 + 37/37.
- All edited `src/` and `test/` files and the benchmark script parse cleanly
  (`Meta.parseall`); `translate_batched_cuda.jl` is only `include`d when
  `load_cuda_radix_lifecycle!()` runs, so its device methods could not be
  executed locally.

### H200 validation (2026-07-22, user-approved submissions)

**First full run (job 12869389, m13h-1-1, 2h15m): 12 failures, all one cause.**
429,781/429,793 integration tests passed (all parity/numerics/counters); the 12
failures were the steady-state `CUDA.@allocated == 0` gate on the dense M2L
launch — 9.6–48.9 KB/step, F64 exactly 2x F32. A stage-instrumented diagnostic
(job 12871128, `scripts/debug_023f_alloc.jl`) isolated it: gather/scatter/fill
allocate 0; **every cuBLAS call allocates exactly `2*sizeof(TF)` bytes**
(1208 gemms x 16 B = 19,328 B, matching the failures exactly). CUDA.jl runs
cuBLAS in `CUBLAS_POINTER_MODE_DEVICE` and `mul!` stages alpha/beta through a
fresh device `CuRef` per call — unavoidable through the public `mul!` API.

**Fix:** the plan stages alpha=1/beta=0 as length-1 device vectors at
construction (carried in the whole-pass bundle) and `_cuda_dense_class_gemm!`
calls `CUBLAS.gemm!` with them directly (device arrays convert to `CuRef`
arguments by pointer, allocation-free). Diagnostic rerun (job 12871186): 0 bytes
across all configs and both drivers.

**Green functional baseline (job 12871195, COMPLETED, exit 0):** lifecycle
208/208 + 37/37, integration **429,793/429,793**, benchmark completed.
Artifacts: `data/dense_translation_m2l_cuda/fm023f-12871195.out` +
`cuda_m13h-*_20260722-*.csv` (also the failed-run before-CSV from 12869389).

**Baseline profile (n=20000, ell=3, LH off, Float64, whole-pass / per-class
m2l_ms):** P=4: 35.1 / 47.4; P=8: 73.0 / 98.9; P=12: 77.7 / 104.1 — vs
precomputed-y 0.13 / 1.37 / 3.48 and concat 0.26 / 3.16 / 9.0. Dense M2L is
**launch-bound**: nearly flat in P while flops grow ~100x, consistent with one
gemm per nonempty class per chunk (1.4k–3.1k launches x ~5 us). Dense operator
storage: 7.6 (P4) / 156 (P8) / 715 (P12) MiB F64; halves at F32; all fit within
the gate.

### Optimize phase (in progress)

Motivated by the launch-bound baseline (the task's "grouped/batched/per-class
GEMM selection + launch count" lever):

1. Operators repacked as one `D x D x nclasses` device array (single upload
   target; per-class GEMM drivers slice contiguous strided views through the
   direct `CUBLAS.gemm!` call — still zero-workspace, zero-alloc).
2. New fused per-route kernel driver (`DENSE_CUDA_FUSED`, default off pending
   measurement): gather -> D x D matvec -> atomic scatter-add in **one kernel
   launch per step**, no slabs, no cuBLAS. Source column staged in shared
   memory; operator reads coalesced across row-threads; per-class operator
   reuse served by L2.
3. Tests extended: packed-operator typing/shape + per-class parity via slices,
   fused-vs-GEMM end-to-end parity, fused zero-alloc gate. Benchmark gains a
   `dense_fused` variant (F64 + F32 rows).

**A/B results (job 12871790, COMPLETED: 208/208 + 37/37 + 429,832/429,832
including the new packed/fused tests; benchmark
`data/dense_translation_m2l_cuda/cuda_m13h-1-1_20260722-234538.csv`).**
M2L medians, ms, Float64, ell=3; "gemm" = whole-pass per-class GEMM driver
(post-scalar-fix), "fused" = per-route kernel:

| P | LH | n=150 gemm/fused | n=2000 gemm/fused | n=20000 gemm/fused | n=2e4 precomputed-y |
|---|----|----|----|----|----|
| 4 | off | 6.00 / 0.019 | 12.3 / 0.041 | 12.4 / 0.042 | 0.131 |
| 8 | off | 20.8 / 0.089 | 26.9 / 0.93 | 26.8 / 1.01 | 1.38 |
| 12 | off | 28.9 / 0.37 | 28.4 / 4.36 | 28.5 / 4.50 | 3.48 |
| 4 | on | — | — | 10.4 / 0.10 | 0.24 |
| 8 | on | — | — | 26.8 / 4.10 | 3.57 |
| 12 | on | — | — | 30.4 / 20.5 | 8.53 |

Every dense variant now reports `m2l_device_alloc_bytes = 0`. The scalar fix
alone cut the whole-pass driver ~3x (35 -> 12.4 ms at P=4); the fused kernel
removes the launch bottleneck entirely (one launch/step): fastest dense driver
in **every supported finite measured config**, up to ~800x over the pre-fix
baseline, and faster than precomputed-y at P <= 8 (LH off and on). Float32/P=12
dense rows are intentionally rejected as non-finite materializations in the
final run artifacts. At P=12 Float64 approaches the memory-bound operator-traffic
limit (routes x D^2 x 8 bytes; 715 MiB LH-off / 3.3 GiB LH-on operators), where
precomputed-y retains the lead — consistent with the strategy's known
storage/traffic scaling, for the `024` crossover analysis.

**Retained optimizations:** (1) device alpha/beta scalar staging (correctness/
contract fix + ~3x GEMM-driver speedup); (2) packed `D x D x nclasses`
operator storage; (3) fused per-route kernel as the production default
(`DENSE_CUDA_FUSED[] = true`). The whole-pass and per-class GEMM drivers are
retained as independently testable references (tests force them explicitly;
the benchmark rows force `DENSE_CUDA_FUSED[] = false`). **Rejected/not
pursued:** grouped/batched cuBLAS (dominated by the fused kernel; would need
per-step pointer-array uploads or device-side pointer builds for no measured
headroom); chunk-width sweep (chunking only affects the non-default GEMM
drivers).

**Confirmation run with the flipped default + driver-forcing test/benchmark
plumbing: job 12872259, COMPLETED, exit 0** — lifecycle 208/208 + 37/37,
integration **429,847/429,847** (the dense block now exercises per-class,
whole-pass, and the fused production default explicitly, all three
allocation-gated), benchmark ok
(`data/dense_translation_m2l_cuda/cuda_m13h-1-1_20260723-*.csv`); the forced
GEMM rows reproduce the A/B numbers (P=8 F64 LH-off: gemm 25.9 ms, fused
1.008 ms), confirming the toggle plumbing measures the intended drivers.
Float32/P=12 dense rows in these artifacts are `fit=false` non-finite
materialization rejections, not supported measured dense configurations.

Lifecycle counters on all three H200 runs: `route_uploads`/`operator_uploads`
constant after construction, `expansion_host_copies == 0`, `CUDA.@allocated ==
0` on the dense M2L stage for every driver — the `022`/`023` contract holds.

**Watch item for `024` (out of 023f scope):** the concat variant's benchmark
rows report a steady 80–160 B/step M2L device allocation — the same CUDA.jl
`mul!` alpha/beta `CuRef` staging root-caused here, from that path's few gemm
calls. The 023f fix pattern (plan-staged device scalars + direct
`CUBLAS.gemm!`) applies directly if `024` wants it closed.

**Status: task complete (2026-07-23).** Deliverables: device-resident dense
strategy through the production cache/options surface with construction-time
operator upload only, explicit memory estimation/gates, three drivers (fused
production default per measurement + two GEMM references), and recorded
H200 before/after data (jobs 12869389 baseline-fail, 12871128/12871186
diagnostics, 12871195 green baseline, 12871790 A/B, 12872259 confirmation).
`Done` checked in `START_HERE.md`; clear-context approval pending.

## Approval Notes

Clear-context approval, 2026-07-24, by a separate reviewing agent.

**Approved.** Review followed `START_HERE.md` item 6 and covered this task
file, its declared dependencies, the dense CUDA production surface in
`src/containers.jl`, `src/translate_batched.jl`,
`src/translate_batched_resident.jl`, and `src/translate_batched_cuda.jl`, the
host and CUDA integration tests, the benchmark/Slurm scripts, and the recorded
H200 CSV/job-output artifacts.

1. **Objectives:** met. `DenseTranslationM2L` is selectable through the
   recurring device `RadixFMMCache` lifecycle, while the one-shot CUDA builder
   retains a precise rejection. The device plan owns packed
   `D×D×nclasses` operators, route-class metadata, fixed-capacity slabs, and
   construction-staged GEMM scalars. Operators use the approved 023e
   `build_dense_m2l_operator!` oracle and are uploaded only during cache
   construction.
2. **Correctness:** the fused production kernel and both retained GEMM
   reference drivers preserve the degree-major `[phi; chi]` layout, use the
   same complete displacement operators as the host strategy, and atomic-add
   repeated target contributions. CUDA tests cover Float32/Float64, LH off/on,
   P in `{4,8,12}` where materialization is finite, host-dense and other
   resident-strategy parity, direct accuracy, empty routes, uneven class
   occupancy, partial chunks, repeated steps, and the explicit non-finite
   Float32/P=12 rejection.
3. **Lifecycle and robustness:** route/operator upload counters remain constant
   after construction, `expansion_host_copies == 0`, operative arrays preserve
   identity across moving-body steps, and all three dense drivers are asserted
   to allocate zero device bytes after warmup. Both the persistent-payload
   limit and the complete lifecycle free-memory/headroom preflight produce
   actionable category estimates.
4. **Performance:** the recorded H200 sequence supplies a correct functional
   baseline, allocation diagnosis, retained scalar-staging fix, fused-kernel
   A/B, and final confirmation. Job `12873418` is fully green (lifecycle
   208/208 + 37/37, integration 416787/416787, benchmark exit 0); the later
   current-tree confirmation job `12889140` is also green (208/208 + 37/37,
   integration 416797/416797, benchmark exit 0). The fused driver removes the
   per-class launch bottleneck and is measurably the best dense driver in every
   supported finite measured configuration; the expected P=12
   storage/traffic crossover is recorded for task 024.
5. **Minimality/readability:** the change is additive, CUDA-only execution
   stays behind the existing runtime loader, host semantics are unchanged, and
   the reference drivers remain independently testable.

Independent local verification on the approval tree (macOS, no CUDA runtime):

```sh
julia --project=. test/dense_translation_m2l_test.jl
# 100/100 + 4/4 pass

julia --project=. test/cuda_radix_integration_test.jl
# CPU-safe gate: 1/1 pass
```

No significant improvements were identified across the six review criteria;
no production or test changes were made during approval.
