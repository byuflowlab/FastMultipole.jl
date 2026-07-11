# 023 Implementation Production Integration

## Objective

Route the production FMM through the validated, tuned matrix operators and radix
driver so the refactor delivers a realized end-user speedup, while keeping the legacy
octree + dynamic-`P` path as the default fallback (minimally invasive, backward
compatible).

## Dependencies

- `008b-implementation-replan.md` (parity-only first pass; this row is the deliberate
  replacement step it deferred)
- `008c-implementation-performance-baseline.md`
- `016a-milestone-review-impl-013-016.md`
- `019-impl-operator-performance-tuning.md`
- `019b-exploratory-smallp-fallback-and-channel-layout.md`
- `021-impl-constant-p-stencil-and-interaction-list.md`

## Required Reading

- `START_HERE.md`
- Approved dependency task files listed above
- Existing `fmm.jl` pipeline and the public `fmm!` / `tune_fmm` entry points

## Artifacts or Production Surface

- Production wiring in `fmm.jl` (and entry points) that dispatches the FMM through the
  new operator + radix path based on basis type (`AbstractOperatorBasis`) or a `Cache`
  flag.
- Tests confirming end-to-end accuracy against `direct!` and parity with the legacy
  path; a point-mass to `1/r` convergence check mirroring the Theory acceptance
  example.

## Deliverables

- Basis-type / flag dispatch selecting the new path; legacy path remains the default
  and is unchanged when the new path is not selected.
- The constant-`P` radix path wired end to end (clustering, interaction list, batched
  M2L, downward pass, evaluation).
- No signature changes to the public API for existing users; the new path is opt-in.

## Verification

Run `test/fmm_test.jl` through the new path and confirm it matches `direct!` to the
configured tolerance and matches the legacy path within parity tolerance; run the
point-mass to `1/r` convergence check. Rerun the full suite with threads
(`julia --project=. --threads=4 -e 'using Pkg; Pkg.test()'`). Benchmark the integrated
path against the legacy path on the `008c` baseline machines and confirm the speedup.
Record commands and result summaries.

## Approval Notes

To be filled by a different agent after implementation and verification are
complete.
