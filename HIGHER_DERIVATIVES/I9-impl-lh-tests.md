# I9 — Lamb–Helmholtz Milestone

## Objective

Cover vortex convergence, symmetry, dynamic order, threading, mixed pairs, and FLOWVPM-shaped usage.

## Dependencies

I8

## Deliverables and acceptance

Follow the corresponding detailed task and acceptance clauses in REVIEW_PLAN.md. Record artifacts, focused verification, and review notes here. Preserve unrelated work and inspect connected interfaces when the codebase map may be stale.

## Artifacts and verification (2026-09-07)

Three permanent testsets appended to `test/third_derivative_test.jl` (run by the full
suite via `runtests.jl`), each with far-field premise guards (`m2l_list` and
`direct_list` both nonempty) so no check can pass vacuously:

- *LH FMM accuracy, symmetry, dynamic P* — n=2000 vortons, leaf 30, MAC 0.5 (537 M2L
  pairs): FMM TS vs single-thread vorton direct asserts ≤3e-9 at P=6, ≤1e-11 at P=12,
  and a ≥100× drop between them; the reference tensors' max relative `(i,j)` asymmetry
  must exceed 0.5 (packed-18 exercised beyond the scalar 10-component subspace, no
  `i,j` symmetry assumed); dynamic-P (`PowerAbsoluteGradient(1e-9)`, P_max=20) must land
  within 1e-10 of direct. Calibration P-sweep at this geometry: rel err 7.2e-9 (P=2) →
  3.0e-10 (P=6) → 4.0e-13 (P=12) → 8.4e-16 (P=20); max asymmetry 2.0; dynamic-P 5.1e-14.
- *LH threading and mixed pairs* — vorton `direct!` with `n_threads=2` vs 1 agrees at
  rtol 1e-13; a mixed call with targets `(gravitational, vortex)` and sources
  `(gravitational, vortex)` (400 bodies each, leaf 10, MAC 0.6 → 1399 M2L pairs)
  matches the multi-system direct reference at rtol 1e-5 for both target systems
  (calibrated 7.0e-7 / 8.0e-7 at P=8).
- *LH FLOWVPM-shaped usage* — one lamb-Helmholtz call requesting velocity + velocity
  gradient + third derivative together (n=300, leaf 10, MAC 0.6, P=10, 27 M2L pairs),
  with `FmmPlan` reuse across a strength-update "time step"; gradient/Hessian/TS each
  match direct at rtol 1e-4 on both steps (calibrated ≤6.9e-6), and the downstream
  `(ω·∇)v` stretching update consumes the Hessian as FLOWVPM does.

An earlier calibration at shallower trees (leaf 20, MAC 0.5) produced empty M2L lists
and machine-precision "errors" — the vacuous mode I6 first identified — which is why
every new testset carries premise guards.

### Suite runs

- Focused (`--project=test --threads=4 --check-bounds=yes`, metadata + third-derivative
  + real-basis files): 1512 pass / 0 fail / 0 error, including the three new LH testsets
  (11 + 7 + 12 assertions).
- Full suite via `Pkg.test` (bounds checking on), 2026-09-07:

  | Threads | Result | Wall time |
  |---|---|---|
  | 1 | 1,382,726 pass / 0 fail / 0 error across 172 suites, exit 0 | ~26 min |
  | 4 | 1,382,870 pass / 0 fail / 0 error across 172 suites, exit 0 | ~12 min |

  Exactly +30 assertions and +3 testsets over the I7 baselines at both thread counts;
  all three LH testsets pass in both runs.

## Clear-context review (2026-09-07)

A fresh clear-context session reviewed the three LH testsets in
`test/third_derivative_test.jl:333-471` against the REVIEW_PLAN I9 clause and accepted
them without changes:

- **Coverage vs clause:** convergence (P=6/P=12 thresholds plus a ≥100× ratio test that
  rules out a shared error floor), symmetry ((i,j)-asymmetry > 0.5 asserted on the
  reference so the packed-18 layout is exercised beyond the scalar subspace; (j,k)
  symmetry is structural to the tensor API, tested in the API testset), dynamic P
  (`PowerAbsoluteGradient(1e-9)` within 1e-10 of direct), threading (direct! 2-vs-1
  threads at rtol 1e-13; threaded FMM exercised via the 4-thread full suite), mixed
  scalar/LH target-and-source pairs, and FLOWVPM-shaped usage (velocity + Hessian + TS
  in one call, plan reuse across a strength update, `(ω·∇)v` consuming the Hessian).
- **Robustness:** every FMM testset carries `m2l_list`/`direct_list` premise guards
  (non-vacuous far/near field); all thresholds have ~10–25× margin over calibrated
  values under fixed seeds; helper functions (`reset!`,
  `update_gradient_stretching!`, vortex index ranges) verified present in
  `test/vortex.jl`.
- **Independent re-run:** focused run (`--project=test --threads=4
  --check-bounds=yes`, metadata + third-derivative + real-basis files) reproduced
  0 fail / 0 error with all three LH testsets passing at exactly 11 + 7 + 12
  assertions. The re-run's driver counted 1530 total passes vs the 1512 recorded
  above — an 18-assertion difference attributable to the driver's include set, with
  no failures anywhere; the LH-testset counts match the record exactly.

No correctness, performance, robustness, user-friendliness, or invasiveness
improvements were found that warrant changes. I9 is closed; G2 is unblocked.

## Status

See the authoritative per-phase table in [`START_HERE.md`](START_HERE.md); update its
1-sentence summary cell in place whenever this item's state changes.
