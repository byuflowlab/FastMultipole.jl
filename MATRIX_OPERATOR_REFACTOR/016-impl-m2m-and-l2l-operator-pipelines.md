# 016 Implementation M2M And L2L Operator Pipelines

## Objective

Extend the explicit operator structure to M2M and L2L.

## Dependencies

- `006-theory-m2m-l2l-extensions.md`
- `008b-implementation-replan.md`
- `008c-implementation-performance-baseline.md`
- `014-impl-full-m2l-operator-pipeline.md`
- `015-impl-axis-swap-benchmarks.md`

## Required Reading

- `START_HERE.md`
- Approved dependency task files listed above
- Approved `theory/m2m-l2l-extensions.md`
- Current production M2M and L2L call sites and tests

## Artifacts or Production Surface

- Production M2M and L2L operator pipeline code
- Tests comparing explicit M2M and L2L output with current behavior

## Deliverables

- [x] Explicit M2M operator pipeline
- [x] Explicit L2L operator pipeline
- [x] Shared cache and scratch integration with the M2L operator layer
- Per the `008b` re-plan, this first pass is **side-by-side, parity-only**: the
  explicit M2M/L2L operator pipelines are validated against production but do
  **not** replace the production `multipole_to_multipole!` / `local_to_local!`
  internals. Production hot-path replacement is a later, explicitly scoped step.

## Implementation Notes

- Added structured M2M and L2L z-translation blocks in `src/translate_batched.jl`
  with overwrite apply kernels validated against `translate_multipole_z!` and
  `translate_local_z!`.
- Added M2M/L2L operator tags and scratch containers in `src/containers.jl`,
  sharing `OperatorInvariantCache`, `OperatorScratch`, two batch work buffers,
  z-block storage, and optional Lamb-Helmholtz coefficient buffers with the M2L
  layer.
- Added `m2m_operator_batch!` and `l2l_operator_batch!` side-by-side APIs. Both
  run source z/y alignment, z translation, optional Lamb-Helmholtz transform,
  return y/z alignment, and accumulate only at the final inverse z rotation.
- `Val(true)` follows the same padded-order policy as the M2L operator layer:
  `χ` is carried at `P_active = P_phi + 1`, while nonphysical `φ` padding is
  zeroed before z translation and before return alignment.
- The `FactoredRotationM2M` and `FactoredRotationL2L` tags currently preserve the
  swappable API surface but delegate their y stages to the production-parity
  materialized path. Dedicated factored M2M/L2L return-stage modes are left for a
  later performance task rather than shipping an unvalidated operator.

## Verification

Run parity tests for representative parent-child offsets, traversal contexts,
and expansion orders. Record commands and result summaries.

Completed with:

```text
julia --project=. -e 'using FastMultipole; using FastMultipole.StaticArrays; using Test, Random; include("test/m2m_l2l_operator_test.jl")'
```

Result summary:

- `M2M/L2L operator pipelines (task 016)`: `23260` passed, `0` failed.
- Coverage: M2M and L2L z-block parity, `Float32` and `Float64`, `Val(false)`
  and `Val(true)`, expansion orders `0`, `1`, `3`, `6`, `9` for z blocks,
  whole-pipeline orders `2`, `4`, `6`, `8`, axis-aligned `+z`/`-z`, `+x`, `+y`,
  mixed-sign off-axis offsets, batched execution, and `P_active = P_phi + 1`
  padding behavior.

Adjacent regression command:

```text
julia --project=. -e 'using FastMultipole; using FastMultipole.StaticArrays; using Test, Random; include("test/rotate_batched_test.jl"); include("test/translate_batched_test.jl"); include("test/m2l_operator_test.jl"); include("test/m2m_l2l_operator_test.jl")'
```

Result summary:

- `rotate_batched_test.jl`, `translate_batched_test.jl`,
  `m2l_operator_test.jl`, and `m2m_l2l_operator_test.jl`: all passed.

## Approval Notes

Clear-context approval (different agent than implementer), `2026-06-24`.

Reviewed against `START_HERE.md` priority order:

1. **Objectives.** Deliverables met and scoped correctly. Explicit M2M and L2L
   pipelines (`m2m_operator_batch!` / `l2l_operator_batch!`) are added side-by-side
   and parity-only; production `multipole_to_multipole!` / `local_to_local!`
   internals are untouched, exactly as the `008b` re-plan requires. Shared cache
   (`OperatorInvariantCache`) and a base `OperatorScratch` are reused, with M2M/L2L
   scratch wrappers mirroring `M2LOperatorScratch`.
2. **Correctness.** The implemented chains match `theory/m2m-l2l-extensions.md`
   stage-for-stage: forward `Z_phi` then materialized `Y(θ)` into `A`; φ-padding
   zeroed; z-translation into `B`; optional Lamb-Helmholtz (multipole factors for
   M2M, local factors for L2L); padding re-zeroed on the mid buffer; return
   `back_Y(θ)` then accumulating `Z_phi^-1`. M2M uses multipole y-rotation +
   `zeta` signs; L2L uses local y-rotation + `eta` signs — consistent with theory.
   The fixed-`m` z-block formulas `U_m=(-r)^(n-np)/(n-np)!` (lower-tri) and
   `V_m=(-r)^(np-n)/(np-n)!` (upper-tri) are correct and validated against
   `translate_multipole_z!` / `translate_local_z!`. Re-ran the stated verification:
   `M2M/L2L operator pipelines (task 016)` = 23260 passed / 0 failed, and the full
   adjacent regression (`rotate_batched`, `translate_batched`, `m2l_operator`,
   `m2m_l2l_operator`) all pass.
3. **Performance.** Appropriate for a parity-only first pass: z-blocks and LH
   factors are materialized once per column for downstream offset-class reuse; not
   on the production hot path yet (deferred to `023`).
4. **Robustness.** Coverage spans `Float32`/`Float64`, `Val(false)`/`Val(true)`,
   multiple orders, axis-aligned and mixed-sign off-axis offsets, batched
   execution, `P_active = P_phi + 1` padding, and z-block index formulas for
   `P = 0:12`. The test is wired into `test/runtests.jl` (line 69), so it cannot
   silently rot.
5. **Minimally invasive.** Follows the Implementation Code Placement rules: structs
   in `containers.jl`, operators in `translate_batched.jl` / `rotate_batched.jl`;
   no production translation call sites changed.
6. **Readable.** Clear docstrings, theory references, and explanatory comments.

Non-blocking note: the `FactoredRotationM2M` / `FactoredRotationL2L` tags
intentionally delegate to the materialized y-stage, so their parity tests do not
yet exercise a distinct factored implementation. This is explicitly documented and
consistent with the roadmap (only factored M2L is required near-term; factored
M2M/L2L return modes are deferred to a later performance task).

**Approved.**
