# 040 Impl: Adaptive Octree Resident Lifecycle (Host)

## Status and Entry Gate

**DONE and clear-context APPROVED `2026-08-15`.**
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

### Measurement of record (cluster job 13179323)

Pre-registration committed (`118174d`) before the first submission
(13179268, cancelled — CSV-at-end + block-buffered stdout risked losing all
rows at the 3h wall; protocol-neutral amendment `48b5ac6` committed before
resubmission). Job **13179323**: COMPLETED 02:38:19 ExitCode 0:0 (sacct
re-verified). CSV of record `data/fm040_lifecycle_cost.csv` (24/24 rows ok;
warm fmm! medians of 5, same-job anchors, 2000-target sampled-direct
velocity rel RMS; P=4, Float64, q=5).

**Accuracy**: every row within the 1e-3 gate — adaptive 3.87e-4 – 6.58e-4,
uniform 2.30e-4 – 5.09e-4.

**Headlines at n = 1e6 (t_step, warm median)**:

| case | adaptive K=64 | uniform ℓ=5 | uniform ℓ=6 | adaptive vs best uniform |
|---|---|---|---|---|
| wake | **18.31 s** (popmax 64) | 262.99 s (popmax 1231) | 44.37 s (popmax 182) | **2.42× faster** |
| multiscale100 | **18.62 s** (popmax 64) | 176.04 s (popmax 2442) | 60.94 s (popmax 346) | **3.27× faster** |
| unitcube | 18.90 s (popmax 64) | 17.67 s (popmax 61) | 36.67 s | 1.07× slower (7%) |

The 038 cost mechanism is reproduced end-to-end at production scale:
bounded leaf population (popmax = K_max everywhere) at per-region depth
kills the fat-cell direct term (wake U body pairs 9.39e8 adaptive vs
4.36e10 at uniform ℓ=5 — 46×; multiscale 1.43e9 vs 2.85e10 — 20×). The
uniform-cube non-regression is structural: adaptive K=64 lands on the ℓ=5
partition (32,686 vs 32,710 leaves, U pairs equal to 5 digits, V 12.25M vs
12.28M); the 7% step overhead is dominated by the known double refresh
(adaptive t_update 1.79 s vs uniform 0.51 s — the uniform structures still
refresh with the policy armed; 039 open item, priced for 041/041a).
At n = 1e5 the same pattern holds (multiscale 1.94 s vs best uniform
4.23 s = 2.2×; unitcube adaptive 1.74 s vs 2.99 s; wake is the one case
where the best uniform (ℓ=6, 1.39 s) beats adaptive K=64 (1.75 s) at this
n — K below the pre-registered {64,128} sweep would likely close it;
recorded honestly for 041a).

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

---

## Clear-Context Approval (2026-08-15 10:14 MDT)

**Verdict: APPROVED.** Independent clear-context review per START_HERE
protocol item 6 (objectives, correctness, performance, robustness, minimal
invasiveness, readability, in that order).

Evidence checked:

- Read: START_HERE (protocol + 040 row + phase preamble), this task file,
  the full `118174d`/`48b5ac6`/`630f3f3` diffs (all src/test/script/theory
  surfaces), `theory/adaptive-radix-octree.md` §2.6/§4/§5/§9, the overnight
  decision log, `scripts/fm040_lifecycle_cost.jl`, and the CSV of record.
- Re-ran locally: `julia --project=. --threads=1
  test/adaptive_lifecycle_test.jl` — 357 pass / 0 fail, exit 0, per-testset
  counts (40/12/16/4×32/2×38/2×34/3/11/3) matching the completion notes.
- Hand-verified correctness claims: (1) no-new-operator-tables — the window
  driver mirrors `_launch_hierarchical_resident_m2l!` exactly (same plan
  types, same refresh calls, `clear_locals=false`, `route_levels`/
  `route_offsets` unread by the plan launchers), and both the 039 CSR
  `vstage_class` numbering and the workspace plans derive from the same
  `_hierarchical_class_metadata` (`(L−first_m2l_level)·noffsets+k`,
  level-scaled `effective_offsets`) — the class-id sharing is by shared
  construction, and the machine-exact uniform-limit parity (2.4e-17,
  ell=3 exercises multi-level classes) locks it. (2) M2M prefix-zeroing
  omission is sound: `_launch_host_b2m!` `fill!`s the whole multipole
  buffer before writing leaves. (3) Sign findings are justified: the
  resident `_host_b2m_kernel!` carries `+(−1)^{n+m} q` (no legacy
  negation), the oracles compose resident P2M through the validated dense
  M2L, and the independent analytic Biot-Savart anchor pins the absolute
  sign — a shared convention error cannot hide. The §4.2 χ-top-row caveat
  characterization is sound (truncated LH row-up mixing in the M2L top row
  vs S2L's exact projection; representation difference at truncation
  order, correctly tested via φ+χ(≤P) machine parity + tail-scaled
  evaluated parity + the anchor). (4) Gate semantics: global
  `_direct_kernel_geometry_gate!` skipped only when `cache.adaptive !==
  nothing`; regularized kernels refuse construction unless the per-cell σ
  gate is armed (`rho_t ≥ _gate_reach_rho`, matching `sigma_row`).
  (5) Exact-once consumption: 039 proves list-level exact-once; the
  lifecycle consumes each of U/V/W/X in exactly one stage (U inside
  `_launch_host_l2b!` after `fill!(output,0)`, V in the window driver, X
  in S2L before L2L, W in M2T after L2B — all accumulate-only), with the
  uniform-limit machine parity and the P=8 end-to-end gates (6.5e-6)
  excluding double counting.
- Measurement protocol verified: pre-registration `118174d` (21:30) before
  job 13179268 (21:34); amendment `48b5ac6` committed and logged before
  resubmission 13179323; same-job anchors; every headline recomputed from
  the CSV and exact (wake 2.423×, multiscale 3.273×, cube 1.07× slower
  with the 1.28 s double-refresh delta ≈ the whole gap, U-pair 46×/20×,
  accuracy 3.87e-4–6.58e-4 all inside the gate); the wake n=1e5 honest
  negative is recorded.
- Minimal invasiveness: uniform path bit-preserved when `adaptive ===
  nothing` (gate-skip condition, fmm! branch, ctor guards, one trailing
  `Any` field, CUDA ctor `nothing` — opt-in only, production defaults
  unchanged). Option (b) correctly NOT implemented; split veto default OFF
  preserved; theory §9 stale veto line fixed on this touch as mandated.

NOTED (non-blocking):

1. "Best uniform" in the headlines means best of the pre-registered
   uniform ℓ∈{5,6} sweep; wake/multiscale at ℓ=7 were not measured. Fine
   for this row's acceptance; `041a` should widen the uniform depth sweep
   before publishable adaptive-vs-uniform claims.
2. The `48b5ac6` amendment's sampling fix (Set-dedup-sort-truncate →
   exact shuffle-sample) also removed a low-index selection bias in the
   original target draw, not only the small-n undershoot. Strictly an
   improvement, committed+logged before the job of record — but it is a
   sampling-distribution change, slightly understated as "no effect at
   n ≥ 1e5".
3. Adaptive S2L supports `Point{Source}`/`Point{Vortex}` only, enforced
   by a loud runtime throw inside the lifecycle; consider promoting to a
   construction-time guard on the 041 touch.
4. User-ratification items carried unchanged: split-veto default OFF, 038
   option (b), E2 disposition, the 040 deferrals (TwoPass/Partitioned on
   adaptive, W-list M2T LH hessian, rectangular, per-level radius
   schedules), the uniform-cube "agreed tolerance" (7% at n=1e6, double
   refresh), and the theory §4.2 top-row wording caveat.
