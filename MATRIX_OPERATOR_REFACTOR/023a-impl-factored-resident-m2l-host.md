# 023a Implementation Factored Resident M2L (Host)

## Objective

Make `options.operator = FactoredRotationM2L()` select a genuine factored M2L
stage inside the host resident lifecycle (`run_host_radix_lifecycle!`), so that
the integrated `fmm!(system, cache::RadixFMMCache)` can actually A/B the two M2L
operator variants for `024`. `MaterializedYRotationM2L` (via the tuned
`ConcatenatedFixedZM2L` concat path) remains the default and is untouched.

Background (verified `2026-07-15` during `024` planning): both operator variants
are fully built and parity-tested at the per-column kernel level
(`m2l_operator_batch!` in `src/translate_batched.jl`), but the `022`/`023`
resident lifecycle dispatches M2L only on `options.m2l_strategy`
(`SharedRotationM2L` / `ConcatenatedFixedZM2L`, both materialized-lineage
whole-slab dense) and ignores `options.operator`. The operator field currently
affects only the retained flat-oracle test launchers
(`src/translate_batched_resident.jl:213,257`) and a `DEBUG[]` physical-subspace
assertion. This task closes that gap on the host; `023b` mirrors it on CUDA.

This stage is also the **grouped-GEMM M2L** lever deferred by `019`/`023` for
`019a`: because every route in an offset class shares `(phi, theta, r)`, the
factored chain's per-degree `U_n`/`V_n` applications become batched GEMMs over
all class columns at `O(P^3)`/column, versus the concat dense operator's
`O(P^4)`/column.

This is a two-phase task. First implement the factored resident stage and
establish parity against direct evaluation and the existing concat path. Then
profile, optimize, and retest the correct baseline. Completion requires recorded
before/after timing and allocation measurements; a functional implementation
without the measured optimization phase is not complete.

## Dependencies

- `013c-impl-factored-rotation-alignment.md` (factored `Z/S/Z/S` stages; plain
  per-degree mode matrices `V_n`/`U_n` per the `2026-06-23` Plain-H amendment)
- `014-impl-full-m2l-operator-pipeline.md` (`m2l_operator_batch!` composition,
  `011` z-translation blocks, `012` Lamb-Helmholtz coupling)
- `019-impl-operator-performance-tuning.md` (tuned resident/concat conventions)
- `023-impl-production-integration.md` (`RadixFMMCache`, invariant contract,
  per-step refresh, zero-realloc requirements)

## Required Reading

- `START_HERE.md`
- The dependency task files above (especially `013c` Revised Implementation
  Notes and the `023` invariant-contract notes)
- `src/translate_batched.jl` (`_launch_resident_m2l!`, the concat path
  `_launch_resident_m2l_concat!`, and the per-column factored stage functions)
- `src/translate_batched_resident.jl` (host lifecycle driver, flat-oracle
  launchers, `RadixFMMCache` construction)

## Artifacts or Production Surface

- Production code: `src/translate_batched.jl`,
  `src/translate_batched_resident.jl`; new types (if any) in
  `src/containers.jl` per the Implementation Code Placement rules.
- Tests: `test/radix_fmm_integration_test.jl` (extended), plus any focused
  stage-parity test file.

## Design Sketch (implementer must verify against current code)

- **Dispatch point:** `_launch_resident_m2l!`
  (`src/translate_batched.jl:2670`). Recommended shape: a new
  `FactoredConcatM2L <: AbstractResidentM2LStrategy` (type in
  `src/containers.jl`) selected automatically when
  `options.operator isa FactoredRotationM2L`, keeping `options.operator` as the
  user-facing switch that `024` flips. Direct dispatch on the operator is an
  acceptable alternative; either way `RadixFMMCache` construction must accept it
  and the per-step refresh (`build_radix_routes!` route-class refill) must keep
  working unchanged.
- **Kernel:** per offset class (shared `phi/theta/r`), apply the `013c` factored
  chain `Z_phi -> S -> Z_theta -> (fixed-m z-translation K_z) -> return
  alignment`, where `S`/`S_inv` are the cached plain per-degree mode matrices
  `V_n`/`U_n` (`y_mult_U/V`, `y_loc_U/V`) — **not** the ζ-dressed `013b`
  primitives (Plain-H amendment, `2026-06-23`). The shared-angle diagonal phase
  stages are scalar broadcasts across the class; the per-degree `U_n`/`V_n`
  applications batch over all class columns as GEMMs. Reuse the existing
  per-column stage functions and invariant caches from `013c`/`014` wherever
  possible rather than re-deriving them.
- **Allocation:** zero (or near-zero, matching `023`'s ~23 KB/step
  view/dispatch noise) per-step allocation on the steady path. Use the `023`
  function-barrier pattern around loops that touch loosely-typed state fields;
  capacity-sized scratch lives in `ResidentOperatorWorkspace` or a sibling
  allocated at cache construction.
- **Physical-subspace guard (016b watch item):** the factored operators are
  exact only on the physical subspace (`im(m=0) == 0`); keep the `DEBUG[]`
  `_assert_factored_input_physical` guard active on this path.
- Lamb-Helmholtz `Val(true)` and `Val(false)` must both be supported, matching
  the concat path's channel handling.

## Optimize, Profile, and Retest Phase

After the functional path passes parity and lifecycle tests, capture a baseline
and tune at least class ordering, grouped/per-degree GEMM organization, BLAS
threading, packing and scratch layout, small-block/low-occupancy crossover
behavior, and avoidable construction or steady-state allocation. Retest direct
and concat parity plus the cache lifecycle contract after every retained change.

Record before/after per-stage and steady-state timings, allocations, persistent
memory, construction cost, and any selected crossover rule. Host tuning must
include single-thread and multi-thread BLAS measurements on at least one
non-macOS benchmark host; macOS measurements may be retained as smoke-test or
laptop reference data.

## Deliverables

- The factored resident M2L stage, selectable through the public
  `RadixFMMCache(...; options=CUDARadixLifecycleOptions(operator=FactoredRotationM2L()))`
  surface, running inside the unmodified host lifecycle driver.
- Unchanged behavior and performance for the default materialized/concat path.
- Test coverage per Verification below.
- Benchmark artifacts and a brief note in this file recording functional-baseline
  versus optimized stage timings, per-step allocation, construction/persistent
  memory costs, and the host-side comparison against the concat path
  (informational; the definitive comparison is `024`).

## Verification

- Extend `test/radix_fmm_integration_test.jl` with factored-path runs asserting:
  - radix-vs-`direct!`: potential `< 1e-6`, gradient `< 1e-4`;
  - factored-vs-concat parity within the existing radix-vs-legacy tolerances
    (`2e-6` potential / `2e-4` gradient);
  - Lamb-Helmholtz on and off;
  - steady-state `@allocated` bound on the per-step path.
- Cover Float32/Float64, empty route sets, partial final batches, repeated time
  steps, and fixed-domain capacity reuse without cache growth or reallocation.
- Run the same parity and lifecycle checks before tuning and after the final
  optimized implementation.
- Record single- and multi-thread BLAS before/after measurements on a non-macOS
  host.
- Full existing test suite green (`julia --project=. -e 'using Pkg; Pkg.test()'`).

## Approval Notes

Clear-context approval, separate agent, `2026-07-16`. **Approved.**

Reviewed per `START_HERE.md` item 6: this task file, `src/translate_batched.jl`
(dispatch at `_launch_resident_m2l!`, `ResidentM2LFactoredPlan` construction,
`_resident_factored_m2l_group_apply!`, `_factored_y_degree_major_auto!` and the
GEMM crossover), `src/translate_batched_resident.jl` (`update_radix_state!`,
`_refresh_factored_m2l_routes!`, DEBUG guard, counter assertion),
`src/containers.jl` (plan type), both extended test files, the benchmark script,
and the `data/factored_resident_m2l_host/` CSVs.

1. **Objectives**: met. `options.operator = FactoredRotationM2L()` selects a
   genuine grouped factored M2L inside the unmodified host lifecycle; the
   default materialized/concat path is untouched (operator check precedes
   strategy dispatch and changes nothing for `MaterializedYRotationM2L`). The
   chain is `gather+Z_phi -> Plain-H forward Y -> fixed-m z -> [LH rows] ->
   return Y -> Z_phi^{-1}+scatter`; no `Ts(theta)` and no ζ-dressed `013b`
   primitives, per the Plain-H amendment. Both prior rejection blockers are
   resolved: the per-degree BLAS GEMM branch exists with a measured crossover
   (`FACTORED_Y_GEMM_MIN_COLS = 16`, no dim gate, from EPYC/OpenBLAS data), and
   `factored_functional` baseline rows run the same plan through the retained
   allocating shared-rotation loop.
2. **Correctness**: independently re-ran
   `test/radix_fmm_integration_test.jl` (63/63) and
   `test/radix_fmm_timestepping_test.jl` (51065/51065 + 694/694) on this host.
   Scratch-slab aliasing in the grouped kernel was traced through both LH
   branches: every `_factored_y_degree_major_auto!` call receives three slabs
   disjoint from its in/out at that point in the chain. Per-class route count is
   bounded by `max_cells` (one source per target per offset), matching group
   capacity; the timestepping test asserts `sum(counts) == n_routes` and the
   per-group bound each step.
3. **Performance**: non-macOS gate satisfied (EPYC 7763, OpenBLAS 1 and 64
   threads, jobs 12760272/12760919); CSV rows verified against the notes —
   `P=8, N=2000` functional 1719 ms / 3.65 GB -> optimized 1010 ms / 1,808 B;
   `P=12` factored 2778 ms vs concat 5488 ms (1.98x) with 101 vs 1968 MB
   persistent. The honest sparse-class regression (~0.9x at `N=150, P=4`) is
   recorded for `024`.
4. **Robustness**: Float32/Float64 x LH x P in (4, 8) parity matrix (P=4
   included per project convention), forced scalar-vs-GEMM branch parity,
   empty-route sets, varying body counts, array-identity zero-realloc proof,
   warmed `@allocated` bounds, DEBUG physical-subspace guard active,
   `expansion_host_copies == 0` preserved, CUDA still rejects
   `FactoredRotationM2L` (deferred to `023b`).
5. **Minimally invasive**: dispatch on the existing `options.operator` field
   (the task's allowed alternative to a new strategy type); one small plan type
   in `containers.jl`; refresh hooks additive in `update_radix_state!`.
6. **Readability**: fine; comments explain the crossover provenance and the
   baseline's role. Minor, non-blocking: the workspace field `m2l_concat` now
   also carries the factored plan, so the name is slightly stale — worth a
   rename (e.g. `m2l_plan`) in a later maintenance pass, not worth a re-review
   cycle now.

Full `Pkg.test()` was run by the completing agent (exit 0); this review re-ran
the two task-extended test files independently rather than the whole suite.

## Implementation Notes (2026-07-15)

Implemented the host resident selection at `_launch_resident_m2l!`:
`FactoredRotationM2L()` now selects `ResidentM2LFactoredPlan`, while
`MaterializedYRotationM2L()` continues to dispatch through the existing
`m2l_strategy` and unchanged concat launcher. CUDA rejection remains in place.

The fixed-box plan owns one capacity-sized group per accepted offset. Each group
stores invariant `(phi, theta, r)`, Plain-H `y_mult_U/V` and `y_loc_U/V` access,
fixed-`m` z blocks, optional local Lamb--Helmholtz rows, and `max_cells` source /
target columns. `build_radix_routes!` fills the common capacity-sized
`route_class`; refresh resets counts and repacks group prefixes without rebuilding
operators or changing array identity. The debug physical-subspace assertion remains
once per host lifecycle after M2M.

The executed chain is fused gather + `Z_phi`, degree-major Plain-H forward Y,
fixed-`m` z translation, optional LH coupling, degree-major local return Y, and
inverse `Z_phi` + accumulating scatter. The current low-occupancy kernel uses
allocation-free small dense contractions over each degree, avoiding BLAS/view
overhead for the observed sparse class widths while preserving grouped `O(P^3)`
arithmetic. A required GEMM crossover is not yet present. It never constructs
`Ts(theta)` and does not use the ζ-dressed
013b swaps.

### Profiling and retained optimizations

The first correct grouped implementation reused allocating degree helpers and
measured `512,129,744` bytes for a warmed `P=6`, `N=200`, non-LH M2L stage. The
retained changes were: type-stable cached real-mode blocks and `DegreeMajorMaps`,
full-width reusable degree-major slabs, fused valid-prefix kernels, direct fixed-m
packing, and type-stable in-place route refresh. The same probe now allocates
`1,440` bytes for M2L; a `P=8`, `N=400` warmed full step measured `123,888` bytes.
Direct and concat parity were rerun after these changes.

### Verification run locally

- `julia --project=test test/radix_fmm_integration_test.jl`: 40/40 pass.
- `julia --project=test test/radix_fmm_timestepping_test.jl`: factored 51065/51065
  and existing time stepping 694/694 pass.
- `julia --project=. -e 'using Pkg; Pkg.test()'`: exit 0 (existing manifest
  mismatch warnings only).
- Float32/Float64, `Val(false)`/`Val(true)`, direct parity, concat parity, empty
  routes, partial classes, moved/varying body counts, fixed array identity,
  `<=64 KiB` warmed M2L, and `<512 KiB` warmed full-step checks are covered.

### Benchmark evidence and remaining external gate

The reproducible driver is
`scripts/benchmark_023a_factored_host.jl`; CSVs and instructions are under
`data/factored_resident_m2l_host/`. The macOS smoke run covers `P=4,8,12`, both LH
modes, concat/factored, construction, persistent bytes, occupancies, M2L/full-step
timings, and allocations. At `N=150`, observed mean class occupancy was 1.6--5.6;
factored persistent storage was 10.6--92.8 MB versus concat 50.9--872.5 MB and
factored M2L allocation was 1,440 bytes versus concat 6,032/11,840 bytes. Sparse
factored timings ranged from parity at P=4 to roughly 18--27% slower at P=8/12,
which is retained as honest low-occupancy evidence for the later `024` selection.

This host exposes macOS Accelerate at 8 threads even after requesting one thread.
Therefore the required non-macOS 1-thread and 8-thread measurements were not
available at that point; they have since been collected on the HPC cluster (see
the 2026-07-16 session below).

The first separate-agent review also rejected approval because the required
per-degree GEMM execution/crossover and complete functional-baseline benchmark rows
were not yet implemented. Both blockers are resolved by the 2026-07-16 session
below.

## Implementation Notes (2026-07-16): GEMM crossover and required benchmarks

**Per-degree GEMM execution and crossover** (`src/translate_batched.jl`). The
grouped kernel now routes each per-degree Plain-H `U_n`/`V_n` application through
`_factored_y_degree_major_auto!`: degree blocks go to a BLAS `mul!` kernel
(`_factored_y_degree_block_gemm!`, column-prefix views of the capacity-sized
slabs, allocation-free) when the class width is at least
`FACTORED_Y_GEMM_MIN_COLS[]` and the block dimension `2n+1` is at least
`FACTORED_Y_GEMM_MIN_DIM[]`; otherwise they keep the scalar no-alloc kernel. All
four Y stages (phi/chi forward and return) pass a third free workspace slab, so no
new storage was added and the steady path still allocates 1,808 bytes per M2L
stage (within the existing test bounds). Defaults were selected from measured
non-macOS data: **MIN_COLS = 16, MIN_DIM = 1**. Evidence (EPYC 7763, OpenBLAS 1
thread, `data/factored_resident_m2l_host/host_m12-1-10_blas1_*.csv`): forced-GEMM
loses to scalar at class width <= 7 (P=4 LH: 12.3 vs 6.6 ms), wins from width ~20
at every order (P=8 width 20: 93 vs 104 ms; P=12 width 24: 245 vs 341 ms; P=4
width 48: 59 vs 66 ms), and the 1x1/3x3 low-degree blocks never hurt, so no dim
gate is imposed. BLAS thread count is immaterial at these block sizes (64-thread
within noise of 1-thread), so no threading policy is attached to the crossover.

**Functional-baseline rows.** The shared-rotation loop was extracted as
`_launch_resident_m2l_shared!(state, groups)` (production `SharedRotationM2L`
unchanged); run over the factored plan's capacity groups it reproduces the
allocating generic execution as the honest pre-optimization baseline (verified
equal to the grouped kernel to ~4e-9). The benchmark now emits
`factored_functional`, `factored_scalar`, `factored_gemm`, and
`factored_optimized` per `(P, LH, N)` with the thresholds recorded per row, plus
an `FM023A_N` body-count sweep (150/2000/20000) so classes reach full width, and
optional `FM023A_SWEEP_COLS` crossover candidates.

**Non-macOS 1- and 64-thread BLAS evidence** (required gate). sbatch driver
`scripts/cpu_023a_run.sh` (pattern of `cpu_019b_run.sh`: OpenBLAS threads set at
process start, sweep run twice). Jobs 12760272 (pre-final defaults) and 12760919
(final defaults), nodes m12-1-10 / m12-1-21, AMD EPYC 7763, 64 cpus; CSVs
`host_m12-1-{10,21}_blas{1,64}_*.csv`. Final-defaults highlights (1-thread,
non-LH): functional baseline -> optimized at `P=8, N=2000` is 1719 ms / 3.65 GB ->
1010 ms / 1,808 B; at `P=12, N=2000` 4552 ms / 9.4 GB -> 2778 ms / 1,808 B. Against
concat on the same host: factored optimized is 0.9x at the sparse `N=150, P=4`
case (3.9 vs 3.5 ms) and wins everywhere else, up to **1.98x** at `P=12, N=2000`
(2778 vs 5488 ms) with **19x** less persistent workspace (101 vs 1968 MB);
construction is 21-250 ms vs concat 0.2-8.9 s. Informational only; `024` remains
the definitive comparison.

**Tests.** `test/radix_fmm_integration_test.jl` (f2) now covers `P in (4, 8)` for
the factored-vs-concat parity matrix (direct-accuracy assertions gate on `P == 8`,
the stencil design order; the 2e-6/2e-4 same-P parity asserts at all P), and a new
(f3) section forces scalar-vs-GEMM branch parity (< 1e-8 potential / 1e-7
gradient) at `P in (4, 8)` x LH with the warmed `@allocated` bound. 63/63
integration, 51065/51065 factored lifecycle, 694/694 time-stepping, and the full
`Pkg.test()` suite pass after the final defaults.
