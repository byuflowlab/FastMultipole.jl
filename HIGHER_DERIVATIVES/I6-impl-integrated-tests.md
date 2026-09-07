# I6 — Scalar Integration and Performance

## Objective

Cover all switch combinations, systems, probes, P modes, conditioning, near-field caches, plans, metadata, threading, API allocation, and disabled-path performance.

## Dependencies

I5

## Deliverables and acceptance

Follow the corresponding detailed task and acceptance clauses in REVIEW_PLAN.md. Record artifacts, focused verification, and review notes here. Preserve unrelated work and inspect connected interfaces when the codebase map may be stale.

## Artifacts and verification (2026-09-07)

- `test/third_derivative_test.jl` extended with four new testsets plus a probes-via-`fmm!`
  case, closing the previously missing matrix:
  - *FMM accuracy, multi-system, threading, dynamic P*: FMM TS convergence vs direct with
    premise-guarded far field (`m2l_list` and `direct_list` both nonempty; asserts ≤1e-7 at
    P=6, ≤1e-10 at P=12, and a ≥100× drop between them), per-system
    `third_derivative=[true,false]` switch vectors through the two-argument call, `direct!`
    n_threads=2 vs 1 agreement, and `error_tolerance=PowerAbsoluteGradient` dynamic-P
    convergence with TS requested on the same far-field geometry.
- P-sweep evidence (2026-09-07, n=2000 gravitational, leaf 30, MAC 0.5, 583 M2L pairs,
  relative TS error vs single-thread direct): 3.2e-6 (P=1–2), 3.8e-8 (P=6), 3.6e-9 (P=8),
  3.2e-11 (P=12), 2.9e-12 (P=14), 2.9e-13 (P=16), 3.5e-15 (P=20) — clean geometric
  convergence to near machine precision, so TS accuracy is expansion-order-limited only.
  (An earlier n=200/leaf-16/MAC-0.3 configuration had an EMPTY m2l list and gave a flat
  5e-16 at every P — far-field premise guards now prevent that vacuous mode.)
  - *Conditioning*: a `SelfPairs` strength-doubling `DirectConditioningRule` doubles the
    (strength-linear) TS output and restores source buffers bitwise.
  - *Nearfield cache and transformed plans*: TS-only `FmmPlan` + `build_nearfield_cache!`
    reproduces the uncached plan at rtol 1e-12 across a strength-change reuse trial;
    `transform_plan!` refuses a stored cache with TS (direction-carrying); an `fmm!` `Cache`
    built without TS rejects a TS request (layout mismatch).
  - *Metadata, extra outputs, and legacy overloads*: row-layout assertions for
    `DerivativesSwitch{true,true,true,2,2,true}` (metadata 4:5 → TS 19:36 → extra 37:38,
    38 rows), packed get/set at the offset rows, and the legacy three-switch
    `MetadataSystem` kernel serving ordinary requests while `third_derivative=true` fails
    capability preflight with `ArgumentError`.
- Focused run 2026-09-07 (`--project=test --threads=4`, metadata + third-derivative files):
  222 pass / 0 fail / 0 error.
- Performance harnesses added under `scripts/`:
  - `benchmark_ts_disabled_path.jl` — paired ALTERNATING baseline/feature workers per the
    acceptance spec (≥30 samples, ≥0.2 s each, median ratio ≤1.03). Self-vs-self smoke run
    passed with ratio 0.9995; the real baseline-worktree run is an I7 deliverable.
  - `benchmark_ts_enabled_path.jl` — records enabled-path time, allocations, scratch bytes
    per worker, and output-buffer bytes with no threshold. Smoke record at n=5000, P=5,
    1 thread: off 0.0725 s / on 0.5103 s median, allocations 6.87 MB / 7.59 MB, L2B scratch
    1064 B / 4088 B, +18 rows (720 kB output at n=5000).

## Clear-context review (2026-09-07)

- Reviewed with fresh context against the I6 clauses and the Test and Acceptance Plan in
  `REVIEW_PLAN.md`. All clauses map to present testsets or scripts; the focused run was
  independently re-executed (`--project=test --threads=4`, `metadata_extra_test.jl` +
  `third_derivative_test.jl`): 222 pass / 0 fail / 0 error, matching the recorded result.
- The 16-switch-combination clause is satisfied literally through `direct!`; through `fmm!`,
  the tested combos (TS-only, gradient+TS dynamic-P, all-on in
  `real_solid_harmonic_basis_test.jl`) cover the extremes of every compound
  `GS || HS || TS` / `HS || TS` guard in `evaluate_expansions.jl`, so the untested
  intermediate combos activate no unique TS-gated branch — no additional coverage required.
- Both performance harnesses were inspected: the disabled-path driver implements the paired
  alternating-worker spec (≥30 samples, ≥0.2 s floor enforced at warm-up, median ratio
  ≤1.03) and the enabled-path script's internal calls
  (`initialize_gradient_n_m(P, TF; third_derivative)`) match current signatures.
- No changes made; accepted as-is. The real baseline-worktree disabled-path run remains an
  I7 deliverable.

### Post-review correction (2026-09-07, during I7)

- The I7 full-suite runs via `Pkg.test` (which enables `--check-bounds=yes`) errored in the
  *metadata, extra outputs, and legacy overloads* testset: the `MetadataSystem` test kernel
  unconditionally emits 2 extra outputs, but the legacy-overload case requested `metadata=2`
  without `extra_outputs=2`, so `set_extra_output!`'s `@inbounds` write went out of bounds —
  silently in plain runs (why the focused runs passed) and as a `BoundsError` under bounds
  checking, which aborted the remainder of the suite.
- Fix (in `test/third_derivative_test.jl` only): the legacy-overload calls now request
  `extra_outputs=2`, the legacy target carries a 2×2 `extra` array, and the extra-output
  values are asserted (`meta .* sum(strength)`). Production code is unchanged; the review
  checkbox is cleared pending a fresh clear-context review, and future focused runs should
  include `--check-bounds=yes` to match `Pkg.test`.

### Clear-context re-review (2026-09-07, after the I7-era fix)

- Reviewed the legacy-overload testset fix with fresh context: the kernel's
  unconditional 2 extra outputs are now requested (`extra_outputs=2`), the legacy target
  carries a 2×2 `extra` array, and the extra-output values are asserted
  (`meta .* sum(strength)`), so the previously out-of-bounds write is now exercised and
  checked rather than merely avoided. Production code is untouched.
- Re-verified under the same conditions that exposed the bug: focused run with
  `--check-bounds=yes --threads=4` over `metadata_extra_test.jl`,
  `third_derivative_test.jl`, and `real_solid_harmonic_basis_test.jl` — 1451 pass /
  0 fail / 0 error. Accepted; review checkbox restored.

## Status

See the authoritative per-phase table in [`START_HERE.md`](START_HERE.md); update its
1-sentence summary cell in place whenever this item's state changes.
