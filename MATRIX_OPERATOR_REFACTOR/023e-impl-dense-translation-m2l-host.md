# 023e Implementation Dense-Translation Resident M2L (Host)

## Objective

Implement the existing `DenseTranslationM2L <: AbstractResidentM2LStrategy`
placeholder as the complete coefficient-space operator

\[
L_{\mathrm{target}} \mathrel{+}=S(\Delta r)M_{\mathrm{source}}.
\]

Precompute one dense `N_dof x N_dof` matrix for every accepted full displacement
class. Each matrix must include the complete rotation, axial translation, and
Lamb-Helmholtz coupling composition for its configured precision/order/channel
layout. Application gathers class columns, performs one large GEMM for each
occupied class batch, and scatters/accumulates the results.

Treat both storage per displacement class and application per column as
`O(P^4)`. This strategy is intentionally allowed to trade memory and construction
cost for fewer/larger GEMMs; those costs must be measured rather than hidden.

First implement and validate the complete operator. Only after parity is
established may the task tune construction and application. Completion requires
before/after measurements from the functional baseline and optimized result.

## Dependencies

- `005-theory-full-m2l-composition.md` (complete coefficient-space M2L map)
- `011-impl-m2l-z-translation-blocks.md` and
  `012-impl-lamb-helmholtz-operators.md` (translation and channel coupling)
- `014-impl-full-m2l-operator-pipeline.md` (composed operator oracle)
- `019-impl-operator-performance-tuning.md` (resident layout/tuning conventions)
- `023-impl-production-integration.md` (`RadixFMMCache` lifecycle contract)
- `023a-impl-factored-resident-m2l-host.md` (resident coefficient layout and host
  parity reference)

## Required Reading

- `START_HERE.md`
- The dependency task files above
- `src/containers.jl` (the existing placeholder and resident state/workspaces)
- `src/translate_batched.jl` (placeholder throw, concat/factored paths, and
  coefficient layout helpers)
- `src/translate_batched_resident.jl` (construction, refresh, and capacity reuse)
- Existing host radix integration and time-stepping tests

## Artifacts or Production Surface

- Complete `DenseTranslationM2L` construction and application in the existing
  resident production files; supporting types remain in `src/containers.jl`.
- Host correctness, lifecycle, allocation, and memory-limit tests under `test/`.
- Reproducible construction/application benchmarks and data under
  `MATRIX_OPERATOR_REFACTOR/scripts/` and `MATRIX_OPERATOR_REFACTOR/data/`.

## Functional Phase

- Define `N_dof` from the approved resident coefficient layout and the selected
  Lamb-Helmholtz mode. Build separate correctly sized operators for LH off/on;
  do not embed or apply nonexistent channel degrees of freedom.
- Canonically key matrices by the full accepted integer displacement class
  `Delta r`, including azimuth, polar angle, and distance. Do not key only by
  `theta`.
- Construct each dense map from the already parity-tested component operators or
  by applying the existing composed oracle to basis columns. Do not introduce a
  second unverified convention for rotation signs, degree ordering, or channel
  coupling.
- Gather all source columns for an occupied displacement class into contiguous
  storage, apply its full matrix with one large GEMM, then scatter-add to target
  locals. Correctly handle multiple contributions to the same target.
- Allocate operator storage, mappings, and capacity-sized packing buffers once at
  cache construction. Enforce a configurable, explicit host-memory limit and
  fail with an actionable footprint estimate when construction would exceed it.
- Establish parity against direct evaluation, the existing concat resident path,
  and the per-column reconstructed `Ts(theta)` composition oracle before tuning.

## Optimize, Profile, and Retest Phase

Measure the functional baseline, then profile and tune at least:

- operator construction (including basis-column blocking and reuse of invariant
  work);
- canonical displacement-class ordering and class-sorted gather/scatter;
- BLAS threading and operand/layout orientation;
- source packing and target accumulation;
- chunking for peak scratch-memory control; and
- memory-limit/crossover behavior as `P`, precision, LH mode, and class occupancy
  change.

Retain only changes that pass the full parity and lifecycle tests. On a non-macOS
benchmark host, record single- and multi-thread BLAS before/after results,
including operator build time, steady-state M2L time, allocations, persistent
operator footprint, scratch/peak memory, and chunk/crossover settings.

## Deliverables

- A complete, selectable `DenseTranslationM2L` resident host strategy; remove the
  existing deferred-placeholder throw for supported host configurations.
- Construction-time full-displacement operator generation with explicit memory
  estimation/limits.
- Class-batched large-GEMM application with fixed-capacity storage reuse.
- Before/after build and steady-state benchmark artifacts.

## Verification

- Float32 and Float64; Lamb-Helmholtz on and off.
- Direct, concat, and reconstructed-`Ts(theta)` oracle parity before tuning and
  after the final optimization.
- Empty route sets, partial final batches/chunks, uneven class occupancy, and
  repeated contributions to a target.
- Repeated time steps and fixed-domain capacity reuse without operator rebuild,
  cache growth, or steady-state buffer reallocation.
- Low-memory-limit rejection and footprint accounting tests.
- Single- and multi-thread BLAS measurements on a non-macOS host.
- Full existing host test suite green.

## Approval Notes (clear-context approval, 2026-07-20)

Approved by a separate reviewing agent per the START_HERE clear-context protocol.

- **Objectives**: `DenseTranslationM2L` is realized as one complete `D×D`
  degree-major stacked `[phi; chi]` coefficient operator per accepted displacement
  class, built through the parity-tested `MaterializedYRotationM2L` pipeline
  (`build_dense_m2l_operator!`, `src/translate_batched.jl`). Application gathers
  class columns, runs one chunked `mul!` per occupied class, and serial
  scatter-adds repeated targets (`_launch_resident_m2l_dense_plan!`). Classes are
  keyed by the full accepted integer offset (r/θ/φ), not by θ alone. The
  placeholder throw is removed for host; CUDA is cleanly deferred to 023f with an
  actionable error. Matches the stated objective and Implementation Code Placement
  rules.
- **Correctness**: ran `test/dense_translation_m2l_test.jl` on the current working
  tree — 66/66 + 4/4 pass. Coverage includes Float32/Float64, LH off/on, P=4
  same-P concat parity, P=8 direct accuracy, composed-oracle per-class agreement,
  zero routes, partial chunks, repeated targets, refresh array-identity, and
  footprint/limit accounting. Separate LH sizing (`_dense_m2m_dof`) avoids applying
  nonexistent channel dof.
- **Performance/robustness**: construction-time-only allocation with a checked,
  actionable persistent-memory gate (overflow-safe integer arithmetic throughout);
  in-place per-step route refresh with capacity assertions and no reallocation
  (verified by allocation-budget tests, 1,104 B warmed stage). Non-macOS EPYC BLAS1
  and BLAS64 before/after data (650 CSVs, jobs 12840355/56, 12848432) and a clear
  verification summary back the retained/rejected optimization decisions.
- **Minimally invasive & readable**: additive strategy type + dispatch; legacy and
  other resident paths untouched. Code is well-commented and idiomatic.

No significant improvements identified across the six review criteria; no code
changes made, so no re-approval is required.

## Implementation Notes (2026-07-20)

- `DenseTranslationM2L` now exposes the persistent-memory, apply-chunk, and
  construction-chunk controls specified by the task and enforces the
  `MaterializedYRotationM2L` pairing.
- Complete matrices are built in reusable identity-column blocks through
  `m2l_operator_batch!(MaterializedYRotationM2L(), ...)`. The recurring cache
  builds the complete accepted stencil; one-shot states build the distinct
  represented offsets in stable interaction-list order.
- `ResidentM2LDensePlan` owns exact payload accounting, class capacities,
  stable route packing, complete matrices, and two application slabs. The
  configured limit is checked before operators or slabs are allocated.
- The production launch gathers physical degree-major rows, applies one chunked
  class GEMM, and serially scatter-adds repeated targets. Warmed stage allocation
  is 1,104 bytes across the final ORC sweep (64 KiB gate); warmed fixed-domain full
  steps stayed below the 512 KiB gate.
- Host parity covers Float32/Float64, LH off/on, nonuniform free-function geometry,
  the composed oracle, materialized concat, repeated targets, zero routes, partial
  chunks, route-array identity, and P=8 direct accuracy. The full `Pkg.test()` suite
  passed on macOS (one pre-existing broken test, no failures).
- CUDA one-shot and recurring entry points reject Dense before workspace/upload
  construction with the task-023f follow-up in the error.

### Optimization pass

- Scalar multiplication for narrow classes was rejected. On the local P=8,
  Float64, LH-on fixture, sending only width-one classes to the scalar loop raised
  M2L stage time from about 0.68 s to 2.13 s.
- Apply chunks 8/16/32 were no faster than the default full class width; the local
  P=4/N=2000 stage measurements were approximately 2.73/2.37/2.23 ms versus
  2.17--2.19 ms at full width.
- Build widths 1/4/8/16/full were within construction-timing noise locally, so the
  default full `D` width remains. The user controls remain available for memory
  management.
- Removing stable packing saved only about 0.07 ms per refresh at 34,512 routes and
  would impose a new route-emission ordering dependency on the one-shot path; it
  was rejected. Occupancy reordering and operand transposition were not pursued
  because profiling did not identify a launch/layout bottleneck.
- Concrete host index-map and slab function barriers were retained after the first
  ORC capture exposed P=12 boxed-call allocation. They reduced median stage
  allocation from 77,520 to 1,104 bytes and median stage time by about 3.5%.

### Benchmark jobs

- Functional baseline: BYU ORC Slurm job `12840355`; its one transient
  filesystem-open row is recovered from the identical pre-fix job `12840356`.
  Job `12848113` separately verifies that configuration after the retained fix.
- Four-strategy no-regression capture: BYU ORC Slurm job `12840356`.
- Final dense capture after the retained allocation specialization: job `12848432`.
- Local macOS smoke artifact and non-macOS CSV/log artifacts live under
  `data/dense_translation_m2l_host/`; the final cross-job summary is recorded
  there after both jobs complete.
