# 040 Impl: Adaptive Octree Resident Lifecycle (Host)

## Status and Entry Gate

**DONE (2026-08-14, overnight campaign). Awaiting clear-context approval.**
Completion notes at the end of this file.

Entry gate: `038` and `039` complete and approved. (Met: 038 approved 19:38
MDT, 039 approved 20:37 MDT, 2026-08-14.)

## Objective

Run the full host resident lifecycle (B2M → M2M → V-list M2L → L2L → L2B,
plus U-list direct and the new M2T/S2L for W/X lists) on the `039` adaptive
tree, end to end, at production accuracy.

## Scope and placement

- V-list M2L must flow through the **existing** resident M2L strategies and
  level-scaled operator tables unchanged — the `039` class-format parity
  guarantees this; any required strategy change is a stop-and-discuss event.
- New M2T and S2L operator kernels per `038` item 4 go in
  `translate_batched.jl` (or `evaluate_expansions_batched.jl` if evaluation-
  side placement reads better), with Lamb-Helmholtz (φ+χ) coverage and the
  `008h` χ-order rule.
- M2M/L2L run over the adaptive occupied-ancestor levels using the existing
  edge-group machinery.
- Nearfield: U-list pairs feed the existing direct kernels (singular,
  `RegularizedVortex`, and the `032a` winner once selected), honoring the
  per-cell geometry gate.
- The uniform-depth path remains default and untouched.

## Verification

- Sampled-direct accuracy on the two phase cases and the `038` multi-scale
  case: velocity RMS ≤ 1e-3 gate, Jacobian logged as diagnostic; Float32 and
  Float64; `P=4` and `P=8`.
- Uniform-limit parity: with all leaves forced to one level, lifecycle
  results match the existing hierarchical path to tolerance.
- W/X-path unit tests: M2T and S2L each validated against direct evaluation
  and against the equivalent M2L+L2B / B2M+M2L compositions.
- Zero per-step allocation; `023` counter contract on the host lifecycle.

## Acceptance

End-to-end accuracy gates pass on all three cases; the multi-scale case shows
the cost behavior predicted by the `038` model (bounded leaf population,
per-region depth) with measured host timings recorded; no regression on the
uniform cube/wake cases beyond an agreed tolerance.

---

## Completion Notes (2026-08-14, overnight campaign; lead agent)

### What was built (commit `118174d` on `matrix-ops`)

- **`src/containers.jl`** — `AdaptiveResidentLifecycle` (capacity-sized host
  lifecycle container) and a new `RadixFMMCache.adaptive_state::Any` field
  (device ctor passes `nothing`; `recenter!` rebuilds it via the field copy).
- **`src/translate_batched_resident.jl`** — state assembly + drivers:
  `_allocate_adaptive_resident_lifecycle` builds a genuine `DeviceRadixGrid`
  **mirror** of the 039 adaptive node table (level-major layout matches the
  uniform convention; `node_centers`/`node_keys`/permutation arrays alias the
  tree, `node_levels`/`parent_index` are Int mirrors, leaves are presented as
  the state's "cells"), its own adaptive-sort-order `source_bodies`/`output`
  slabs, and the **existing** `_radix_cache_workspace` (per-level M2M/L2L
  capacity groups + the hierarchical-mode M2L window plan over the 039
  `effective_offsets` class metadata). `_refresh_adaptive_lifecycle_typed!`
  refreshes all mirrors per step with **zero allocation** (reuses
  `_pack_radix_source_bodies!` and `_refresh_resident_stage_groups!`
  verbatim; U node endpoints map to leaf-cell slots via `leaf_slot_of`).
  `run_adaptive_host_radix_lifecycle!` executes §2.6:
  B2M (existing kernels) → adaptive M2M → V-list M2L → X-list S2L → L2L
  (existing launcher) → U-direct + L2B (existing `_launch_host_l2b!`) →
  W-list M2T, with the 023 counter assertion around the pipeline.
  Constructor guards: `TwoPassVortex`/`PartitionedVortex` + adaptive throw
  (uniform-lattice deficit sweep deferral); regularized kernels REQUIRE the
  per-cell σ gate armed (`rho_t >= _gate_reach_rho(kernel)`, matching
  `sigma_row`); `hessian=true` + LH + adaptive throws (W-list M2T LH hessian
  deferral). `update_radix_state!` skips the global
  `_direct_kernel_geometry_gate!` when adaptive is armed (theory §5: the
  sticky per-cell demotion gate replaces the global throw).
- **`src/translate_batched.jl`** — the new operators + adaptive launchers:
  `_resident_multipole_eval_flat` (+`_hessian`) — M2T evaluation (irregular
  harmonics via the validated legacy `irregular_harmonics!` into a
  preallocated scratch; φ degree-shift + χ same-degree per 008e/008h, χ at
  P_active = P+1); `_host_m2t_pairs_kernel!` (W sweep, potential + gradient
  + scalar hessian); `_host_s2l_pairs_kernel!` (scalar
  `L += +(−1)^{n+m} q conj(S)` — resident sign, see below) and
  `_host_s2l_vortex_pairs_kernel!` (verbatim `test/bodytolocal.jl` port, χ
  through P_active); `_launch_adaptive_resident_m2m!` (existing edge groups
  WITHOUT the uniform nonleaf prefix zeroing, which would zero coarse
  adaptive leaves — B2M's full-buffer refill makes zeroing unnecessary);
  `_launch_adaptive_resident_m2l!` (windows the 039 CSR class stream through
  the UNCHANGED `ResidentM2LConcatPlan`/`ResidentM2LPrecomputedYPlan`/
  `ResidentM2LDensePlan` launchers, `clear_locals=false`; global class ids
  are shared with the plans by construction; **no new operator tables**).
- **`src/fmm.jl`** — host branch: with `adaptive=` armed, `fmm!` runs the
  adaptive lifecycle instead of the uniform one; `finalize_radix_output!`
  works verbatim (the adaptive state carries the tree's perm metadata).
- **`src/tree_batched.jl`** — `_refresh_adaptive_radix!` extended to refresh
  the lifecycle mirrors after the tree + list rebuild.
- **`src/translate_batched_cuda.jl`** — device ctor passes the extra
  `nothing` (only the opt-in plumbing crossing; no device code).
- **`test/adaptive_lifecycle_test.jl`** (new, wired into runtests after the
  039 file) + updates to `test/adaptive_octree_test.jl` (039 opt-in test:
  bit-identity stopgap → accuracy agreement, per the intended 040 semantics;
  K_max population test tightened per 039-approval noted item 2).
- **`MATRIX_OPERATOR_REFACTOR/scripts/fm040_lifecycle_cost.jl`** —
  pre-registered acceptance measurement (protocol in header).
- **theory §9** — stale "split-veto default-on" corrected to default OFF
  (039-approval noted item 1).

### Convention findings (dev-verified, locked by oracles)

1. The resident scalar pipeline carries **no legacy strength negation**
   (resident direct u = +q/4πr), so M2T returns +u/4π (legacy
   `evaluate_multipole`'s −u flip removed) and scalar S2L drops the theory
   §4.2 leading minus. Vortex ports are convention-identical.
2. Theory §4.2 oracle nuance: the P2M→M2L composition matches the vortex S2L
   machine-exactly on φ (all rows) and χ degrees ≤ P; the χ **top row**
   (P_active = P+1) is representation-dependent (the M2L's own truncated LH
   row-up mixing vs S2L's exact projection). Both representations evaluate
   the analytic Biot-Savart field to the same truncation order (dev probe:
   S2L marginally better). The parity test asserts machine parity on
   φ + χ(≤P), evaluated parity, and an independent analytic anchor.

### Verification (local Mac, 1 thread, Julia 1.12.5)

`julia --project=. --threads=1 test/adaptive_lifecycle_test.jl` — ALL PASS:

- **Accuracy gates** (velocity rel RMS vs full `direct!`, n = 2000,
  K_max=16/ell_max=6/q=5 over an ell=3 cache): scalar 40 tests over
  cube/filament/multiscale × P ∈ {4,8} × {Float64,Float32}, all ≤ 1e-3
  (spot values, multiscale: P=4 F64 3.94e-4, P=8 F64 6.46e-6, P=4 F32
  3.95e-4, P=8 F32 1.76e-5; potential rel max-abs ≤ 1e-3, e.g. 8.95e-5 at
  P=4); LH vortex 12 tests over cube/multiscale × P × TF, all ≤ 1e-3
  (multiscale P=4 4.97e-4, P=8 1.11e-5). W/X lists asserted nonempty on the
  multi-level cases (e.g. multiscale: V=74,718, U=20,712, W=X=6,282).
- **Uniform-limit lifecycle parity** 16 tests (complete occupancy, matched
  rigid q=5 policy, ell ∈ {2,3}, P ∈ {4,8}): ≤ 1e-12·scale, measured
  2.4e-17 — machine-exact; W = X = 0.
- **M2T oracles** 128 tests (both TF, P = 4/8): potential + gradient +
  scalar hessian vs dense-M2L-to-point-local composition, and LH gradient
  (φ+χ) — machine-exact (rtol 1e-10 F64 / 2e-4 F32).
- **S2L oracles** 144 tests (both TF, P = 4/8, two source draws): scalar
  coefficient + evaluated parity machine-exact; **LH vortex parity (the 038
  deferred item)**: φ and χ(≤P) coefficients machine-exact, evaluated
  velocity parity, analytic Biot-Savart anchor ≤ 1e-4.
- **RegularizedVortex** through the adaptive U list with the armed per-cell
  gate vs the erf-based regularized direct reference: rel max err < 1e-3.
- **Zero-allocation/counters**: typed per-step lifecycle refresh = 0 bytes;
  typed lifecycle run = 1520 bytes constant (the shared nearfield-mode
  `Val()` dispatch idiom, same as the uniform host path; gate ≤ 4096);
  warm `fmm!` = 28.8 KB (< 512 KB repo gate; the pre-existing ~30 KB
  `update_radix_state!` baseline — the adaptive stages themselves add none);
  all six 023 transfer counters zero; output/expansion array identity across
  steps.
- **Guards** 3 tests (TwoPassVortex, ungated regularized, hessian+LH).
- **039 suite** re-run after updates: 60,774 pass / 0 fail.
- **Regression** (7 related host radix files, one session): 150,279 pass /
  0 fail. Full `Pkg.test()` run overnight (see decision log).

### Measurement of record (cluster job 13179268)

Pre-registration committed (`118174d`) before submission. Results in
`data/fm040_lifecycle_cost.csv` — see the decision-log completion entry for
the headline numbers (filled on job completion).

### Deviations / open items

- fmm! consumption semantics: with `adaptive=` armed the host branch now runs
  the adaptive lifecycle (the 039 "uniform bit-identical" test was an
  explicit stopgap; production defaults unchanged — the policy remains
  opt-in).
- Deferrals (guarded loudly, logged for user ratification/041):
  `TwoPassVortex`/`PartitionedVortex` on adaptive; W-list M2T LH hessian;
  rectangular domains; per-level radius schedules.
- Known inefficiency (logged, priced in 041a): with adaptive armed,
  `update_radix_state!` still refreshes the uniform structures (double
  refresh); sort unification remains the 039 open item.
- Workspace stage-slab memory scales with the max per-level node count (same
  as the uniform host cache); chunked group apply is a 041 tightening
  candidate.
- Theory §4.2 "exact per-channel oracle" wording deserves a top-row caveat
  (finding 2 above) on the next theory touch.
