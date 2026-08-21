# 023c Implementation Precomputed-Y Resident M2L (Host)

## Objective

Add `PrecomputedFactoredYM2L <: AbstractResidentM2LStrategy` and implement it in
the host resident lifecycle as a measured production strategy. For every
distinct radix-stencil polar angle `theta`, precompute

\[
M_n(\theta) = U_n D_n(\theta) V_n
\]

separately for the multipole and local modes at every required degree. At
application time, use one block GEMM per degree and angle class while retaining
separate `Z_phi`, fixed-`m` z-translation, Lamb-Helmholtz coupling, and scatter
stages.

This is a two-phase task. First establish a functional resident implementation
with parity against direct evaluation, the existing whole-slab concat path, and
the per-degree factored path from `023a`. Then profile, optimize, and retest it.
Completion requires recorded before/after measurements; a correct first pass
alone is not complete.

`PrecomputedFactoredYM2L` is selected only with
`options.operator = FactoredRotationM2L()`. Construction must reject an
incompatible operator selection rather than silently changing the requested
mathematics. Preserve the `DEBUG[]` physical-subspace guard
(`im(m=0) == 0`) used by the factored lineage.

## Dependencies

- `013c-impl-factored-rotation-alignment.md` (plain per-degree mode matrices
  `U_n`/`V_n` and the physical-subspace restriction)
- `014-impl-full-m2l-operator-pipeline.md` (z translation and Lamb-Helmholtz
  composition)
- `019-impl-operator-performance-tuning.md` (resident tuning conventions)
- `023-impl-production-integration.md` (`RadixFMMCache` lifecycle and capacity
  reuse contract)
- `023a-impl-factored-resident-m2l-host.md` (functional host factored reference)

## Required Reading

- `START_HERE.md`
- The dependency task files above, including their implementation and approval
  notes
- `src/containers.jl` (resident strategy, option, state, and workspace types)
- `src/translate_batched.jl` (resident M2L launchers, concat and factored paths)
- `src/translate_batched_resident.jl` (host lifecycle construction and refresh)
- Existing host radix integration and time-stepping tests

## Artifacts or Production Surface

- Strategy and storage types in `src/containers.jl`.
- Host construction/application code in `src/translate_batched.jl` and
  `src/translate_batched_resident.jl`.
- Focused parity, lifecycle, allocation, and capacity-reuse tests under `test/`.
- Reproducible benchmark scripts under `MATRIX_OPERATOR_REFACTOR/scripts/` and
  measurements under `MATRIX_OPERATOR_REFACTOR/data/`.

## Functional Phase

- Deduplicate the polar angles represented by accepted radix displacement
  classes using their canonical stencil metadata, not approximate equality of
  independently computed floating-point angles. Store a stable mapping from
  each route/displacement class to its angle class.
- Build the multipole and local `M_n(theta)` matrices once at cache construction
  for every required degree and distinct angle class. Keep `Z_phi`, radial
  fixed-`m` z translation, Lamb-Helmholtz coupling, and route scatter explicit;
  do not fold them into the precomputed-y matrices.
- Apply the appropriate precomputed block with one GEMM per degree and occupied
  angle class. Support empty classes and do not assume uniform class occupancy.
- Keep all operator matrices, class metadata, and capacity-sized scratch in the
  resident cache. Repeated steps and fixed-domain route refreshes must reuse
  storage without rebuilding invariant matrices or reallocating buffers.
- Preserve `Val(false)` and `Val(true)` channel behavior, including the required
  multipole/local degree ranges for the Lamb-Helmholtz layout.
- Establish numerical parity before optimization against:
  1. direct evaluation;
  2. `ConcatenatedFixedZM2L`, the existing whole-slab concat path; and
  3. the `023a` per-degree factored resident implementation.

## Optimize, Profile, and Retest Phase

Profile the correct baseline, retain its measurements, then tune at least:

- angle deduplication and canonical class ordering;
- class-sorted gather/scatter and reuse of packed columns;
- matrix and scratch memory layout for BLAS-friendly contiguous operands;
- BLAS threading and grouped/batched execution where the host BLAS supports it;
- small-block and low-occupancy crossover behavior, including a justified
  fallback to the better resident strategy when precomputed-y GEMMs lose; and
- construction-time matrix assembly and avoidable temporary allocation.

Repeat the same correctness and lifecycle suite after each retained optimization.
Record baseline and optimized stage timings, steady-state allocations, persistent
operator/scratch memory, construction time, and the selected crossover rule.
Host tuning must include single-thread and multi-thread BLAS measurements on at
least one non-macOS benchmark host; local macOS results may be smoke tests only.

## Deliverables

- Exported `PrecomputedFactoredYM2L <: AbstractResidentM2LStrategy`, usable
  through the existing `RadixFMMCache` options surface with
  `FactoredRotationM2L()`.
- Construction-time angle-class/operator generation and a zero-reallocation
  steady-state host application path.
- A documented, measured crossover policy for tiny blocks or sparse classes.
- Benchmark artifacts containing before/after performance, allocations,
  construction cost, and persistent memory footprint.

## Verification

- Float32 and Float64.
- Lamb-Helmholtz on and off.
- Direct-evaluation and concat/factored resident parity before tuning and after
  the final optimization.
- Empty route sets, empty angle classes, partial final batches, and uneven class
  occupancy.
- Repeated time steps and fixed-domain capacity reuse without invariant rebuild
  or cache growth.
- `RadixFMMCache` steady-state no-reallocation contract and an explicit
  `@allocated` bound consistent with the accepted lifecycle noise.
- Single- and multi-thread BLAS runs on a non-macOS host, with raw and summarized
  before/after measurements recorded.
- Full existing host test suite green.

## Approval Notes

Clear-context approval, 2026-07-18, by a separate reviewing agent.

**Approved.** Reviewed per `START_HERE.md` item 6: this task file, the
production surfaces (`PrecomputedFactoredYM2L` and `ResidentM2LPrecomputedYPlan`
in `src/containers.jl`; construction, exact-angle metadata, apply chain, and
crossover in `src/translate_batched.jl`; route refresh in
`src/translate_batched_resident.jl`), `test/precomputed_y_resident_m2l_test.jl`,
the benchmark scripts, and the recorded evidence in
`data/precomputed_y_resident_m2l_host/` (jobs `12799879`, `12801369`,
`hpc_summary_20260718.md`).

1. **Objectives**: met. Per-angle/per-degree `M_n(θ) = U_n D_n(θ) V_n` blocks
   are precomputed at construction for both dressings; `Z_phi`, fixed-`m` z
   translation, LH coupling, and scatter remain explicit stages; the strategy
   is exported, selectable only with `FactoredRotationM2L()` (construction
   rejects mismatches at both the options and workspace layers), and CUDA
   rejects it with the 023d deferral message. Both required phases (functional
   parity, then measured non-macOS tuning) are complete.
2. **Correctness**: the exact integer angle key `(sign(z), z²/g, ρ²/g)` avoids
   floating-point θ deduplication and handles poles/equator; the angle-major /
   offset-minor packing partitions each angle's columns exactly (histogram
   total asserted each refresh). Focused suite 73/73 covers Float32/Float64 ×
   LH on/off, construction matrix parity, direct/concat/023a parity, empty and
   uneven classes, refresh identity, and a warmed allocation bound. Reviewer
   reran the focused suite (73/73), radix integration (63/63), the CPU-safe
   CUDA gate, and full `Pkg.test()` on this working tree — all green.
3. **Performance**: measured on AMD EPYC 7763 (ORC `m12-1-3`) with 1- and
   64-thread ILP64 OpenBLAS; the global 16-column GEMM crossover stayed within
   5% of the best focused candidate across all 36 thread/regime combinations,
   and the strategy roughly halves M2L time vs the 023a reference on host. The
   decision to keep a single global threshold (no block-dimension gate) is
   well justified by the recorded gaps.
4. **Robustness**: capacity-sized scratch uses a per-angle dense fixed-grid
   pair bound capped by route/occupancy limits — a true maximum under the
   fixed-box cache contract; steady-state refresh mutates only counts/prefixes
   and preserves array identity. The AssertionError on histogram mismatch
   protects the packing invariant.
5. **Minimally invasive**: the plan slots into the existing `m2l_concat`
   workspace field and dispatch chain; the 023a plan and concat paths are
   unchanged.
6. **Readability**: construction, key derivation, and the crossover rationale
   are clearly commented with measured provenance.

Non-blocking observation (no change made): the apply path trusts the
construction-time angle-capacity bound without a per-refresh
`angle_counts[a] <= angle_capacities[a]` assertion; the bound is sound under
the fixed-box contract, but a one-line debug assertion would harden against
any future contract change. Also noted for `024`: the occupancy-dependent
scratch tradeoff (331 MB at P=12/N=20000/LH) is honestly recorded and should
feed the strategy-selection discussion.

## Implementation Notes (2026-07-18)

Implemented the host-only `PrecomputedFactoredYM2L` option and a separate
`ResidentM2LPrecomputedYPlan`. Exact polar classes use the reduced integer key
`(sign(z), z^2/g, (x^2+y^2)/g)`, so equatorial and both pole cases are handled
without floating-point theta deduplication. First occurrence in the stencil is
the stable angle order; original accepted-offset order is retained within each
angle.

Construction materializes real multipole and local `U_n D_n(theta) V_n` blocks
for every degree/angle, per-offset fixed-m z blocks and optional LH rows, packed
route metadata, and reusable scratch. Scratch width uses the fixed grid's dense
pair bound per angle (also capped by the occupied-cell and global route bounds).
Refresh histograms accepted-offset routes and stably packs them angle-major /
offset-minor while preserving all array and matrix identities.

The execution chain is gather plus explicit `Z_phi`, one real precomputed-y
apply per degree and occupied angle, per-offset fixed-m translation and LH
coupling, one local precomputed-y apply per degree, and inverse-`Z_phi`
scatter/accumulate. The existing DEBUG physical-subspace guard and the
`P_phi`/`P_active=P_phi+1` LH layout are unchanged. CUDA rejects the strategy
with the task-023d deferral message. The 023a plan and materialized concat path
remain separate and unchanged.

Focused verification passes 73 tests across Float32/Float64, LH on/off, exact
angle metadata, construction matrix parity, direct/materialized-concat/023a
parity, empty classes, repeated body-count refreshes, invariant identity, and a
warmed M2L allocation bound. Existing radix integration (63/63) and time-step
suites (51065/51065 and 694/694) also pass. Full `Pkg.test()` completed with
`FastMultipole tests passed` on the implementation host.

The reproducible benchmark and Slurm runner are in `scripts/`. A local macOS
smoke CSV is in `data/precomputed_y_resident_m2l_host/`; it validates the scalar,
forced-GEMM, mixed, and 023a reference variants but is not production tuning
evidence. At that point the required non-macOS single-/multi-thread run was still
outstanding, so the task index row was intentionally not marked Done and the
provisional 12-column crossover was not final measured provenance.

## Completion Notes (2026-07-18): ORC tuning and final verification

The benchmark now warms relevant construction/application specializations and
records CPU/kernel, Julia, BLAS, actual thread counts, commit/worktree provenance,
occupancy, construction, allocation, operator/scratch memory, M2L stage, and full
step data. Optional `FM023C_SWEEP_COLS` candidates leave production unchanged
during measurement. `cpu_023c_run.sh` gates both sweeps on the focused 73-test
suite; `cpu_023c_submit.sh` and `cpu_023c_fetch.sh` provide the dirty-worktree
sync, login-node instantiate, submission, and artifact-fetch workflow.

ORC job 12799879 ran on AMD EPYC 7763 node `m12-1-3` with Julia 1.11.7 and
separate one-/64-thread ILP64 OpenBLAS processes. Both processes exited zero and
reported the requested actual BLAS counts. The measured global 16-column
crossover stayed within 5% of the per-regime best candidate for M2L-stage and
full-step timings across all `P/N/LH/thread` regimes; provisional 12 missed that
criterion by reaching 15.6% behind the best 64-thread M2L result. The production
constant is therefore 16. Since the simple global rule met the criterion, no
degree/block-dimension gate was retained.

Job 12801369 is the final threshold-16 confirmation, with explicit commit
`4d5ea4ee1f3a303e4df1dba19f8e160766ac65b1` / dirty-worktree provenance in its
CSVs. Raw CSVs, Slurm logs, environment details, storage/occupancy results, and
representative before/after timings are under
`data/precomputed_y_resident_m2l_host/`.

After changing the crossover, local verification passed: focused 023c 73/73,
radix integration 63/63, factored lifecycle/time-stepping 51065/51065 and
694/694, and full `julia --project=. -e 'using Pkg; Pkg.test()'` with
`FastMultipole tests passed`. Float32/Float64, LH off/on, direct,
materialized-concat and 023a parity, empty/uneven classes, refresh/capacity
identity, and allocation bounds are covered by the focused and full suites.
