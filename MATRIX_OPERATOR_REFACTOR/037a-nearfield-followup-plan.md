# 037a Rectangular Grid and Nearfield Follow-up Plan

## Status and Work Record

**Done `2026-08-13`; negative promotion verdict.** H200 screen jobs `13166480`
and `13166482` independently reproduced the headline result but accidentally
shared one output file; clean confirmation job `13167145` supplied the
single-job CSV of record. Optimization job `13170168` tested the only warranted
follow-up lever. No production default changed.

Implemented:

- Automatically derived rectangular bounds are now symmetrically snapped
  around the tight-domain center in FLOWVPM construction/retry and
  FastMultipole's derived rectangular `recenter!`. Explicit caller bounds keep
  their original lower-face anchor.
- `RadixFMMSettings` and the benchmark driver accept `rho_c`; two-pass
  auto-geometry sizes the primary direct list from `rho_c`, while `rho_t`
  continues to size the independent deficit shell.
- The GPU two-pass diagnostic now counts correction candidate and accepted
  shell body pairs without affecting recurring execution.
- An exact target-point/source-cell AABB prune can skip pass-2 body scans that
  cannot intersect a correction shell. It remains off by default, with the
  original traversal retained as the control.
- The exact error-decomposition driver accepts arbitrary configuration files
  and cutoff sets. It evaluates each pair once into regularized and singular
  fields, bins the correction by cutoff, and reconstructs all exact
  partitioned fields by prefix sum. This reports `P_rho-R` and `F_rho-P_rho`
  separately against the checksummed reference.
- `fm037a_cases_screen.txt` pre-registers centered cubic/rectangular baselines
  and the P5/P6, q=3 two-pass cutoff screen; `fm037a_cutoff_configs.txt`
  pre-registers exact-decomposition candidates. The infeasible one-level-
  deeper branch was removed before submission: at q=3 even the minimum allowed
  `rho_c=1.5` exceeds the next-level AABB gap for both wake scales, so the
  production adequacy gate would correctly reject it.

Local verification:

- FastMultipole radix integration: 89/89; rectangular geometry: 64/64.
- FLOWVPM host coupling: 12/12 production, 23/23 tuning, 2/2 recenter,
  30/30 rectangular, 5/5 rectangular recenter, 4/4 rejection paths.
- Benchmark screen dry-run: all pre-registered configurations resolve (the
  final screened grid contains 18 rows after removing infeasible deeper-tree
  rows).
- Exact-decomposition smoke test at wake n=1000 matches independent U/J
  references to `2.09e-16`/`5.34e-16` relative.
- Modified Julia sources parse and both repository diffs pass whitespace
  validation.
- H200 job `13170168`: CUDA interface 1333/1333, lifecycle 227/227 and
  concat parity 37/37, CUDA nearfield-binning 307/307, all FLOWVPM coupling
  suites and the checksummed 033 reference gate passed. Recurring allocation
  and transfer counters remained flat.

## Hardware Result and Closeout Verdict

The clean 18-row H200 confirmation is
`data/flowvpm_gpu_campaign/fm037a_screen_confirm.csv`; the six-row optimization
cycle is `data/flowvpm_gpu_campaign/fm037a_aabb.csv`.

- Centering removes the earlier lattice artifact, but rectangular geometry is
  only `1.2%` faster at wake `n=1e5` (`7.787 -> 7.694 ms`) and `0.3%` faster at
  `n=1e6` (`83.587 -> 83.340 ms`). This is below the promotion threshold.
- Every P5/q3 two-pass cutoff in `rho_t=3.0:3.668` fails the `1e-3` velocity
  gate (`1.74e-3` to `2.19e-3`). P6 passes, but the best unpruned `rho_t=3.4`
  rows are `42%` and `20%` slower than the centered rectangular baseline.
- AABB pruning preserves the field/error exactly, reduces correction
  candidates from `0.800B -> 0.355B` and `13.415B -> 4.729B`, and improves the
  passing two-pass solve from `10.948 -> 10.637 ms` and
  `99.987 -> 87.120 ms`. It nevertheless remains `39%` and `4.6%` slower than
  the corresponding `7.633` and `83.306 ms` baselines.
- At `n=1e6`, the pruned two-pass L2B is already `7.51 ms` faster than the
  baseline, but mandatory P6 adds about `5.8 ms` in B2M+M2L. Linear pricing of
  all remaining rejected candidates gives less than `4.9 ms` upside before a
  fine-index construction/traversal cost, insufficient to reach the required
  5% end-to-end gain. At `n=1e5` the residual pruning opportunity is negligible
  relative to the P6 far-field penalty.

Therefore no further uniform-grid two-pass optimization cycle is warranted.
Retain the centered-bound correctness fix, `rho_c`/diagnostic tooling, and the
off-by-default AABB research lever; retain the shipped partitioned P5/q6
default. The full cutoff-decomposition campaign is not launched because no
timed two-pass candidate satisfies the performance promotion gate; its
instrument and smoke-validated configuration remain available to `037b`.
The adaptive phase should test the simpler probe-free regularized U-list
alternative discussed at closeout.

## Summary

The rectangular hierarchy is functioning, but two effects hide its savings:

- Auto-derived transverse extents are power-of-two padded without re-centering,
  shifting the rectangular leaf lattice. At wake `n = 1e5`, this creates 918
  versus 896 occupied leaves, 1,258 extra direct routes, and 23,212 extra
  leaf-M2L routes.
- At `n = 1e6`, nearfield/L2B dominates. The current partitioned `q = 6`
  stencil evaluates 13.4 billion candidate body pairs, so removing coarse
  far-field levels rarely shortens the critical path.

Keep the existing aspect-ratio-5 wake. Pursue a general nearfield improvement
using the existing two-pass regularization correction with a smaller primary
stencil and deeper grid. This combination was not tested by 032a, whose
two-pass comparison held `q = 16`.

## Implementation Changes

- Center automatically snapped rectangular bounds around the original
  tight-box center. Preserve explicitly supplied `x_min` semantics. Verify
  that cubic and rectangular wake arms then share the same leaf lattice,
  occupied leaves, and direct-pair list.
- Extend FLOWVPM's internal radix settings and benchmark parser with optional
  `rho_c` for `TwoPassVortex`.
- Make automatic two-pass geometry use `rho_c`, rather than `rho_t`, for the
  primary-stencil coverage constraint.
- Re-optimize the physical correction cutoff instead of assuming the currently
  validated `rho_t = 3.668` is minimal. For each candidate cutoff, decompose
  sampled error into (a) the regularized-versus-singular omitted tail beyond
  the cutoff and (b) FMM error relative to that cutoff's exact partitioned
  field. Select the smallest cutoff whose conservative combined velocity-RMS
  error remains below `1e-3` with repeatability margin; keep Jacobian RMS as a
  diagnostic. Use `3.668` as the initial passing bracket.
- Convert a selected physical cutoff to cell coverage with the exact
  source-directed AABB minimum-gap predicate. Do not add only one cell
  half-diagonal: both the source and target can lie at cell corners, so a
  center-distance approximation would require the sum of their circumradii
  (one full cubic-cell diagonal for equal cells). The exact AABB-gap test is
  tighter and is the production rule.
- Evaluate this predetermined progression against the shipped P5, `q = 6`
  partitioned baseline:
  1. Two-pass, P5, `q = 3`, current depth, `rho_c = 2`.
  2. The same configuration at one additional radix level.
  3. If accuracy fails, increase expansion order to P6 before enlarging the
     primary stencil.
  4. If performance is limited by correction traversal, add a device-resident
     fine-bin/subcell index derived from the existing sub-Morton ordering so
     the deficit pass visits only bins intersecting `rho_t * sigma_source`;
     retain the existing correction traversal as fallback.
- Preserve zero recurring allocation, graph capture, device residency,
  transfer counters, and the general directed source/target nearfield path.
- Promote a new default only if it passes accuracy and is at least 5% faster
  end-to-end on a material wake case without a regression above 3% elsewhere.
  Otherwise retain it as an opt-in or record the hypothesis as falsified.
- After selecting the best general nearfield configuration, compare centered
  cubic and rectangular grids at identical P, depth, cutoffs, and leaf width.
  Flip the rectangular default only if it is consistently faster beyond
  measurement noise and has no material cube regression.

## Benchmark and Test Plan

- Benchmark the aspect-ratio-5 wake at `n = 1e5` and `1e6`, Float32 and
  Float64, with warmed U/J medians, RK3 at `1e5`, isolated stage timings, and
  production overlap enabled.
- Before the kernel/depth sweep, bracket and bisect the cutoff using the
  existing checksummed sample targets. Re-run neighboring cutoff values in the
  complete FMM pipeline so the selected value includes FMM truncation,
  accumulation, and cutoff-tail error rather than only pointwise pair error.
- Record exact direct body-pair count, correction candidate and accepted pair
  counts, occupied cells, routes by level, kernel timings, allocations, and
  device memory.
- Retain the cube `n = 1e5` neutrality and regression control.
- Gate every candidate on sampled velocity RMS error `<= 1e-3`; retain
  Jacobian RMS as a diagnostic and compare against the existing checksummed
  direct references.
- Add host/device tests for centered snapping, two-pass auto-depth selection,
  P5/P6 correctness, graph replay, stable counters, zero recurring allocation,
  and out-of-box/recenter behavior.
- Record the diagnosis and results as a post-approval 037 follow-up; update the
  035 performance addendum only if the shipped recommendation changes.

## Assumptions

- No aspect-ratio-10 case will be added.
- Optimization applies to the general wake path, not only rectangular caches.
- Existing closed ideas--unordered symmetric nearfield, atomic-only rewrites,
  source tiling/ILP, and stream overlap--remain closed unless new measurements
  identify a distinct mechanism.
