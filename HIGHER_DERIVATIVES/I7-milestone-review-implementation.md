# I7 — Scalar Milestone

## Objective

Run scalar scripts and the full suite at one and four threads; record convergence, disabled-path performance, scratch/output memory, and documentation status.

## Dependencies

I6

## Deliverables and acceptance

Follow the corresponding detailed task and acceptance clauses in REVIEW_PLAN.md. Record artifacts, focused verification, and review notes here. Preserve unrelated work and inspect connected interfaces when the codebase map may be stale.

## Artifacts and verification (2026-09-07)

### Scalar scripts

- `scripts/verify_direct_formulas.jl` (`--project=test`, 1 thread): PASS — scalar and
  point-vortex packed-18 formulas agree with ForwardDiff.

### Full suite at one and four threads

Both runs via `Pkg.test` (which enables `--check-bounds=yes`), after the bounds fix noted
below:

| Threads | Result | Wall time |
|---|---|---|
| 1 | 1,382,696 pass / 0 fail / 0 error across 169 suites, exit 0 | ~23 min |
| 4 | 1,382,840 pass / 0 fail / 0 error across 169 suites, exit 0 | ~15–20 min |

(The 1-thread run reports the *threaded fmm extra_farfield* case as an intentional skip.)

An initial pair of full-suite runs errored in I6's legacy-overload testset — an
out-of-bounds `set_extra_output!` write only detectable under `--check-bounds=yes`
(`extra_outputs=2` was not requested although the `MetadataSystem` kernel always emits 2
extra outputs). The test-only fix and I6 review-state reset are recorded in
`I6-impl-integrated-tests.md`; a bounds-checked focused rerun (222 pass) and both full
suites above confirm it.

### Convergence record

The permanent FMM-accuracy testset asserts ≤1e-7 at P=6, ≤1e-10 at P=12, and a ≥100× drop
between them on a premise-guarded far field; the underlying P-sweep (n=2000, leaf 30, MAC
0.5, 583 M2L pairs) recorded in I6 shows clean geometric convergence 3.2e-6 (P=1–2) →
3.5e-15 (P=20).

### Disabled-path performance (real baseline worktree)

- Baseline: git worktree at `1c81e310` (pre-feature HEAD; zero third-derivative references
  in `src/`), instantiated from its committed Project.toml. Feature: the live checkout.
- `scripts/benchmark_ts_disabled_path.jl`, 30 paired alternating samples, single-threaded
  workers, n=30,000: baseline median 0.9987 s, feature median 1.0011 s, **ratio 1.0024 —
  PASS** (acceptance ≤ 1.03, sample floor ≥ 0.2 s satisfied at ~1 s/sample).

### Enabled-path scratch/output memory record

`scripts/benchmark_ts_enabled_path.jl`, n=30,000, P=5, 1 thread:

| Quantity | TS off | TS on |
|---|---|---|
| `fmm!` median time (n_rep=5) | 1.0149 s | 6.2773 s |
| `fmm!` allocations | 58.85 MB | 63.18 MB |
| L2B scratch per worker | 1064 B | 4088 B |
| Target buffer rows | 6 | 24 (+18 rows = 4.32 MB output at n=30,000) |

### Documentation status

Reference entries for `ThirdDerivativeTensor`, `third_derivative_range`,
`get_third_derivative`, `set_third_derivative!`, and `supports_third_derivative` are in
`docs/src/reference.md`; the `third_derivative` keyword (scalar-or-tuple), `packed_data`/
`dense`, and the opt-in contract are in `advanced_usage.md`; buffer-row layout and cache
compatibility notes are in `guided_examples.md` and `advanced_usage_2.md`; the Radix
rejection is in `device_interface.md`.

Docs build check (`julia --project=docs docs/make.jl`): the first build failed on four
missing docstrings for feature API listed in `reference.md` (`packed_data`, `dense`,
`get_third_derivative`, `set_third_derivative!`); docstrings were added (docstring-only
production change in `src/containers.jl` and `src/compatibility.jl`; package reloads
cleanly, and the change cannot affect the recorded suite results). The rebuild shows **no
remaining third-derivative-related errors**; the build still exits 1 solely on out-of-scope
issues: a pre-existing `tuning.md` `@example` failure and unresolved `@ref`s
(`extra_output_view` pre-existing at HEAD; `RadixFMMCache`/`buffer_to_target!`/`FmmPlan`
from the concurrent MATRIX_OPERATOR_REFACTOR docstrings, intentionally untouched). Logs:
`i7_docs_build.log`, `i7_docs_build_v2.log` (session scratchpad).

### Provenance and process notes

- Baseline for the disabled-path benchmark: detached git worktree at `1c81e310` (removed
  after the run; recreate with `git worktree add --detach <path> 1c81e310` to reproduce).
- The four added docstrings touch I1-era API surfaces; the pending I6 clear-context
  re-review should also glance at them.

## Clear-context review (2026-09-07)

Reviewed with fresh context against the I7 clauses and the Test and Acceptance Plan in
`REVIEW_PLAN.md`. Accepted; no changes required. All five deliverables are recorded above
with adequate provenance: full suites at one and four threads via `Pkg.test` (bounds
checking on), the verify script (independently re-run this session: PASS), the
disabled-path benchmark against a real `1c81e310` baseline worktree meeting the paired
alternating-sample spec (ratio 1.0024 ≤ 1.03), the enabled-path time/allocation/scratch/
output-row record, and the docs status with the four missing feature docstrings added.
The four docstrings (`packed_data`, `dense` in `src/containers.jl`;
`get_third_derivative`, `set_third_derivative!` in `src/compatibility.jl`) were read and
are accurate about layouts (default rows 17:34, switch-aware ranges, packed order) and
error behavior. The I6 bounds fix these suites depend on was re-verified under
`--check-bounds=yes` in a 1451-pass focused run.

## Status

See the authoritative per-phase table in [`START_HERE.md`](START_HERE.md); update its
1-sentence summary cell in place whenever this item's state changes.
