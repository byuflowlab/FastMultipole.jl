# 012a Milestone Review Implementation 009-012

## Objective

Review Implementation tasks `009` through `012` against the background design
and coordination rules before later Implementation work begins.

## Dependencies

- `008b-implementation-replan.md`
- `008c-implementation-performance-baseline.md`
- `009-impl-basis-and-operator-cache-types.md`
- `010-impl-z-rotation-operators.md`
- `011-impl-m2l-z-translation-blocks.md`
- `012-impl-lamb-helmholtz-operators.md`

## Required Reading

- `START_HERE.md`
- `../MATRIX_OPERATOR_REFACTOR.md`
- Completed task files listed above
- Artifacts and production files listed by the completed task files

## Artifacts or Production Surface

Review the production files, tests, benchmarks, and generated artifacts listed
by tasks `009` through `012`.

## Deliverables

- Roadmap-alignment notes recorded in this file
- Any required coordination-document fixes identified before later work starts

## Verification

Confirm completed work matches the background design, hard phase gate, and task
ordering. If `START_HERE.md`, a task file, and `../MATRIX_OPERATOR_REFACTOR.md`
disagree, stop and require a coordination-document fix.

## Review Notes

Milestone review completed on 2026-06-18.

Read:

- `../MATRIX_OPERATOR_REFACTOR.md`
- `START_HERE.md`
- Completed task files `009`, `010`, `011`, and `012`
- Production/test artifacts listed by those tasks:
  `src/containers.jl`, `src/rotate_batched.jl`,
  `src/translate_batched.jl`, relevant legacy production references in
  `src/rotate.jl` and `src/translate.jl`, include registration in
  `src/FastMultipole.jl`, and test registration/coverage in
  `test/operator_cache_types_test.jl`, `test/rotate_batched_test.jl`,
  `test/translate_batched_test.jl`, `test/lamb_helmholtz_test.jl`, and
  `test/runtests.jl`.

Findings:

- Background-design alignment: confirmed. Tasks `009` through `012` preserve
  the current rotate-translate-rotate algorithm and add explicit operators over
  the current compressed complex layout
  `weights[real_or_imag, component, harmonic_index]`. Native real-basis storage,
  flat buffers, radix-path execution, and production routing remain deferred to
  later rows as required.
- Hard phase gate and task ordering: confirmed. All Theory rows, `008b`, and
  `008c` are marked done and approved before these implementation rows. Rows
  `009`, `010`, and `011` were already done and approved; `012` was reviewed and
  approved during this milestone pass before this `012a` review was recorded.
- Code placement: confirmed. New types live in `src/containers.jl`; new rotation
  operators live in `src/rotate_batched.jl`; new translation/Lamb-Helmholtz
  operators live in `src/translate_batched.jl`. No standalone type or
  Lamb-Helmholtz source file was introduced.
- Minimal production impact: confirmed. Existing hot-path translation and
  rotation call sites remain on the legacy implementation. The new operator code
  is additive and registered by include/test includes only.
- Correctness coverage: confirmed. `009` validates cache/scratch construction
  and drop-in legacy workspace parity. `010` validates z-rotation overwrite and
  inverse accumulation against `rotate_z!`/`back_rotate_z!`. `011` validates
  fixed-`m` M2L blocks and padded `chi` order handling against
  `translate_multipole_to_local_z!`. `012` validates Lamb-Helmholtz sparse
  operator factors/application against production transforms and verifies the
  required `chi_{P_phi+1} -> chi_{P_phi}` local row.
- Review fix applied before approval of `012`: the order-aware
  Lamb-Helmholtz `Val(false)` path was reading the absent `chi` channel while
  computing `phi`. It now copies `phi` directly and clears component 2; tests
  populate nonzero `chi` sentinels to prove scalar-basis behavior is isolated.
- Performance posture: appropriate for this milestone. The operators are
  storage-light/additive and designed for later offset-class batching and GPU
  staging. No premature small-`P` fallback, flat-buffer migration, or production
  routing was added.
- Coordination consistency: no disagreement found among `START_HERE.md`, the
  task files, and `../MATRIX_OPERATOR_REFACTOR.md`. Later roadmap additions
  (`013a`, `020`, `021`, `022`, `019b`, `023`) remain consistent with this
  milestone and do not require changing tasks `009` through `012`.

Verification commands run during this milestone pass:

```
julia --project=. test/translate_batched_test.jl
# M2L z-translation blocks (batched): 7163 Pass / 7163 Total
# Lamb-Helmholtz operators (batched): 11140 Pass / 11140 Total
```

```
julia --check-bounds=yes --project=. test/translate_batched_test.jl
# M2L z-translation blocks (batched): 7163 Pass / 7163 Total
# Lamb-Helmholtz operators (batched): 11140 Pass / 11140 Total
```

```
julia --project=. -e 'using Pkg; Pkg.test()'
# Testing FastMultipole tests passed
```

Note: the full-suite run printed CUDA-related precompile failure markers before
continuing into package tests; the package test command completed successfully.

## Approval Notes

Clear-context approval granted on 2026-06-18 by a different agent than the one
that recorded the review notes.

Approval scope (per `START_HERE.md` item 6): read `START_HERE.md`, this
completed `012a` task file, and the production/test artifacts it lists
(`src/rotate_batched.jl`, `src/translate_batched.jl`, include registration in
`src/FastMultipole.jl`, and test registration in `test/runtests.jl`).

Confirmed in order of importance:

1. Correctness — the operator code matches every structural claim in the review
   notes. The forward/inverse z-rotation phases, fixed-`m` M2L blocks (stable
   `rho = inv(t)` recurrence in production multiply order), and the sparse
   Lamb-Helmholtz `A`/`B` factors with their stated neighbor directions are all
   implemented as described. The `012` review fix is present: both order-aware
   `Val(false)` paths copy φ directly and zero component 2 rather than reading an
   absent χ channel (`translate_batched.jl:414-419`, `466-470`). The padded
   `χ_{P_phi+1} -> χ_{P_phi}` upper-neighbor row is carried in the order-aware
   local transform (`translate_batched.jl:483-497`).
2. Performance — operators are storage-light and materialize-once for later
   offset-class batching/GPU reuse; no premature small-`P` fallback, flat-buffer
   migration, or production routing was added, consistent with the milestone.
3. Robustness — parity tests pass under `--check-bounds=yes` (per the recorded
   review) and the in/out-buffer overwrite design avoids the in-place aliasing
   pitfalls of the legacy loops.
4. Minimally invasive — the new code is purely additive; hot-path production call
   sites remain on the legacy implementation.
5. Human-readable — each operator carries a docstring tying it to its approved
   theory artifact and to the production function it reproduces.

Independent verification re-run on 2026-06-18:

```
julia --project=. test/translate_batched_test.jl
# M2L z-translation blocks (batched): 7163 Pass / 7163 Total
# Lamb-Helmholtz operators (batched): 11140 Pass / 11140 Total
julia --project=. test/rotate_batched_test.jl
# z-rotation operators (batched): 25684 Pass / 25684 Total
julia --project=. -e 'using Pkg; Pkg.test()'
# Testing FastMultipole tests passed
```

No coordination-document disagreement found among `START_HERE.md`, the task
files, and `../MATRIX_OPERATOR_REFACTOR.md`. The `012a` milestone is approved;
downstream Implementation rows (`013` and later) are unblocked.
