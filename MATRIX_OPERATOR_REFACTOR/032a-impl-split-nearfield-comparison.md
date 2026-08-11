# 032a Implementation: Partitioned Nearfield Comparison

## Status and Entry Gate

**Added by user direction on `2026-08-05`; revised after `031a` replaced the
rejected singular-minus-correction design, then extended the same day when the
user restored a two-pass *additive* correction as a third candidate (`031a`
§6.1).** Not started.

Entry gate: `031a` and `032` must both be Done and clear-context approved.
Kernel scope: `gaussianerf` only.

## Objective

Implement the `031a` partitioned regularized/singular nearfield **and its
two-pass additive-correction alternative** on the resident lifecycle, and select
the resident vortex-nearfield default by H200 measurement against `032`'s
regularized-everywhere baseline.

## Deliverables

1. **Partitioned nearfield:** retain the singular FMM far field. Ensure the
   direct geometry contains every pair with `r/σ_src ≤ ρ_t` using the
   source-directed predicate `d_min(B_t,B_s) ≤ ρ_t max(σ_src)`; evaluate those
   pairs once with the cancellation-safe regularized U/J formulas, and use the
   singular U/J kernel for remaining direct pairs. Implement the `ρ≤0.5`
   Horner series for `g` and `h=ρg'-3g` with six terms in Float32 and ten
   in Float64. No runtime `erfc`.
2. **Two-pass additive correction (third candidate).** Restored by user
   direction `2026-08-05` after this file was written; derived in `031a` §6.1
   and carried in the `START_HERE` row summary, which governs. Leave the FMM
   entirely unmodified — singular far field *and* singular direct — and add a
   second pass carrying only the deficit `ΔU = -ḡC`,
   `Δa = (ρg'+3ḡ)/r²`, `Δb = ḡ/(4πr³)`. It touches no `025` routing invariant,
   but its subtraction lands in the target accumulator across two kernels, so it
   requires either **Float64 accumulation of the singular direct term and its
   correction** (the far field may stay FP16-WMMA/Float32) or the **`ρ_c = 2`
   hybrid** that evaluates `ρ ≤ ρ_c` with the stable form inside pass 1. Pass 2
   must reach `ρ_t` on its own (389 classes at `n=1e6, ℓ=5`); pass 1 needs no
   enlargement. Two-pass and partitioning have **opposite depth trends**
   (`λ* = -0.065/-0.119/0.548/2.434` at `ℓ=3/4/5/6`), so the A/B must run at a
   fixed adequate geometry, not at each strategy's own optimum.
3. **Settle the pair-stream ordering before the A/B (`031a` §6.3).** Both split
   strategies assume a branch-free stream. At the shipped `ℓ=5` operating point
   the regularized fraction is `f = 0.310`, so the probability a 32-lane warp is
   branch-homogeneous is `6.9e-6` and an unbinned kernel pays both paths on
   essentially every warp — measured **1.56x slower than `032`'s
   regularized-everywhere baseline**. Cell-level classification does not rescue
   it: only 19 of the 389 direct classes at `ℓ=5` lie entirely inside the
   cutoff, 370 are mixed. A distance-**binned or sorted** pair stream is
   therefore a first-order implementation requirement, not a tuning detail, and
   must be in place before any A/B number is recorded; two-pass's pass 1 is
   uniformly singular and exempt, but its pass 2 is not. Report the achieved
   warp homogeneity alongside the timings.
4. **`ρ_t` is a measurable lever (`031a` §6.4).** Expensive pairs scale as
   `ρ_t³`, and the §4 radii are per-pair worst case while the phase gate is a
   sampled RMS. The RMS-solved radii (`ρ_t(J) = 4.252` against `4.789` at
   `ε=1e-3`) cut expensive pairs by 30% and the leaf near set from 389 to 275
   classes, for every candidate at once. Keep the §4 per-pair radii as the
   default; adopt the RMS radii only if this row's sampled-direct measurement
   confirms them on **both** test cases.
5. **Geometry correctness:** construction-time assertion that no cutoff pair
   is assigned to M2L, exact-once n-body coverage tests, source-cell
   `max(σ_src)` sizing, and rejection or conservative enlargement when the
   selected near geometry cannot cover the cutoff.
6. **Profile-triggered A/B measurement on H200:** partitioned replacement and
   the two-pass additive correction versus `032`'s single-pass
   `RegularizedVortex`, initially at representative
   `n≈1e5` for the cube and the helical wake cylinder (both overlap 2), in both admissible
   precisions. Add `n≈1e3` and `n≈1e6` only if the representative strategies
   are within 10% or the cost model predicts a crossover. Run the full
   seven-point `024b` grid only if the sentinel cases reverse the winner.
   Check sampled relative velocity RMS error `≤1e-3` for winner eligibility;
   log sampled Jacobian RMS error for every configuration as a diagnostic.
   Report branch divergence, pair counts, route overhead, and steady-state
   U+J timing.
7. **Default selection:** report results to the user before changing a
   default. Per-regime defaults are allowed if justified by measurement.
8. **Contract gates:** parity including `P=4`, `023` transfer counters,
   zero recurring allocation, and no regression in the scalar `028`/`030`
   path.

## Dependencies and Reading

- `031a-theory-kernel-splitting-nearfield.md`, Done and approved.
- `032-impl-generalized-device-interface.md`, Done and approved.
- `START_HERE.md`, `theory/kernel-splitting-nearfield.md`, the `031a`
  validation results, and `integration-api-spec.md` §5.

## Work Record

### Reading gate (2026-08-06)

Completed by the executing agent (Claude Fable 5, same session that closed
`032`): `START_HERE.md` (incl. the Integration Phase preamble and both 032a
amendments), `031a-theory-kernel-splitting-nearfield.md` (status, work record,
review-correction history), `theory/kernel-splitting-nearfield.md` in full
(all of §§1–8 incl. the §6.1 two-pass operator/conditioning, §6.3 divergence
model, §6.4 RMS radii, §7 validation results), the `031a` validation data
summaries (`partitioned_replacement.csv`, `geometry_coverage.csv`,
`two_pass_conditioning.csv` figures as quoted in theory §7), and
`integration-api-spec.md` §5 including the three-candidate amendment.

Execution plan: `032a-implementation-plan.md` (this directory) — four stages
(host partitioned → host two-pass → CUDA + binned stream → H200 A/B ladder),
each with a user checkpoint; §6.3 binned-stream mechanism selection is an
explicit measured decision before any A/B number is recorded.

### Stage B — host two-pass additive correction (2026-08-06)

Done per `032a-implementation-plan.md` Stage B; Checkpoint B report issued.

**Kernel.** `TwoPassVortex(; sigma_row, rho_t=4.789, rho_c=2.0)` (isbits,
`containers.jl`, exported), the `rho_c=2` hybrid form only — the plain variant
(F64 accumulation with singular pass 1 everywhere) was not built, since the
hybrid strictly dominates it on the host (F32-admissible, same pass-2 work,
and the theory table shows the hybrid at working precision for every ρ).
Pass 1 reuses the Stage-A `PartitionedVortex` pair math verbatim via a
`_pass1_regularized_cutoff` trait (branch at `rho_c` instead of `rho_t`), so
pass 1 is bitwise the regularized kernel for `ρ ≤ 2` and bitwise singular
beyond. The constructor enforces `1.5 ≤ rho_c < rho_t` (the `ρg'−2g` sign
change at 1.3688 makes smaller floors meaningless).

**Deficit math.** The §6.2 outer branch of `_gaussianerf_g_h` was factored
into `_gaussianerf_gbar_rhogp` (ḡ = e^{−ρ²/2}(Aρ + s(1/ρ²)), ρg' =
Aρ³e^{−ρ²/2}; same constants, no duplication), and the §6.1 deficit is
expressed as an effective (g_e, h_e) = (−ḡ, ρg' + 3ḡ) pair through the
existing `_vortex_pair_ugh` assembly — algebraically exact:
(singular)+(deficit) = (regularized) identically, verified to 1e-14 (F64) /
2e-6 (F32) relative at pair level across the shell.

**Pass-2 traversal (deviation from the plan's two listed options).** Neither
a second stored route list (plan option a) nor a stored offset-class
complement (option b) was used. The host sweep
(`_host_twopass_deficit_kernel!`) enumerates the offset ball
*arithmetically* each evaluation: for every occupied leaf cell, all integer
offsets of Chebyshev radius `R = ⌊rho_t·σ_max/h_leaf⌋ + 1` (σ_max read from
the live packed bodies), pruned per offset by the minimum-gap test
`gap(o)·h_leaf ≤ rho_t·σ_max`, each resolved by binary search on the sorted
leaf Morton keys (works identically under the hierarchical and flat
policies). Rationale: reach coverage holds **by construction** every step
(offsets beyond R have gap ≥ R·h_leaf > rho_t·σ_max), so no fixed list can
ever be stale-inadequate; there is zero per-step allocation (loop bounds and
binary searches only); no state/counts struct changed, so the 023
transfer-counter contract is untouched by construction. The per-pair kernel
gates on `rho_c < ρ ≤ rho_t`, which also handles shell pairs *inside* the
primary near set (pass 1 gave them singular; the deficit completes them).
Stage C can materialize the same pruned ball as a compacted class list for
the GPU (the §6.3 binned-stream requirement applies to pass 2 there).

**Gate dispatch.** `_direct_kernel_geometry_gate!` now reads its reach from
`_gate_reach_rho(kernel)`: `rho_t` for the single-pass regularized kernels,
`rho_c` for `TwoPassVortex` — measured in the reach test: `σ = 0.05` at
`ell = 3`, `q = 3`, unit box (g_min·h_leaf = 0.125) admits two-pass
(needs 0.1) while `PartitionedVortex` correctly throws (needs 0.239).
Device caches refuse `TwoPassVortex` at construction until the Stage C
mirror lands (pass 2 would otherwise be silently skipped).

**Tests** (`two-pass nearfield stage B (task 032a)` in
`device_system_interface_test.jl`, `TwoPassSmoothedVortex` in
`interface_test_systems.jl`; 1248 assertions, all passing):

- pair identity per zone (inside/shell/beyond), F64 + F32;
- end-to-end host two-pass vs the erf-based regularized reference at the
  Stage-A tolerances — P=8 and P=4, Float64 (1e-3 / 1e-2) and Float32
  (3e-3 / 3e-2), all met — plus the two-pass vs regularized-everywhere
  delta bounded at 5e-4 exactly as Stage A's partitioned delta;
- conditioning guard: at the §6.1 table's ρ ∈ {0.01…0.5}, the F32 hybrid
  pair total holds < 1e-5 relative against the F64 erf truth (expected
  ~1e-7) while the test-side plain F32 singular+deficit shows the > 1e-3
  amplification at ρ ≤ 0.02; end-to-end F32 with twenty ρ = 0.02
  near-coincident pairs stays under 3e-3 (the pipeline floor, not the
  ~1e-1 amplification);
- pass-2 reach: deterministic pair at cell offset (2,0,0) outside the
  `q = 3` near set, ρ = 4.0 in the shell; (two-pass − singular-kernel) run
  difference at the target equals the analytic §6.1 deficit to < 2e-3
  relative (fit-error bound ~4e-4), proving the correction lands beyond the
  primary near set;
- constructor negatives (`sigma_row`, `rho_t`, `rho_c` floors/ordering),
  trait-conflict and scalar-body rejections, the rho_c gate failure path,
  and the device refusal.

Full local suite green (`--threads=4`). CUDA mirrors, the §6.3 binned pass-2
stream, and H200 measurements are Stage C/D.

### Stage C — CUDA mirrors + binned pair stream: implementation (2026-08-06)

Implemented per `032a-implementation-plan.md` Stage C; H200 measurement pending
(work record entry below will carry the numbers).

**Mechanism menu (`CUDA_NEARFIELD_BINNING[]`,
`translate_batched_cuda.jl`).** Split vortex kernels
(`PartitionedVortex`, `TwoPassVortex` pass 1) on hierarchical device caches
route through `_launch_cuda_split_nearfield!`:

- `:unbinned` — the plain predicated functor kernel (§6.3 negative control;
  also the automatic fallback on flat-policy caches, which carry no bin
  context).
- `:classsplit` — mechanism (c): per-step three-way device compaction of the
  direct pair list (pure-singular / pure-regularized / mixed) by the shared
  cell-AABB rule `_nearfield_pair_bucket` (host+device, `resident.jl`) against
  per-cell σ extrema; pure buckets run branch-free kernels
  (`SingularVortex` / `RegularizedVortex` math), mixed keeps the predicated
  functor kernel. Compaction is atomic-claimed into construction-sized Int32
  bucket thirds (`3 × direct_capacity`); bucket kernels read their counts from
  device memory (graph-safe — counts vary within an occupancy epoch).
- `:ballot` — mechanism (b), implemented as the warp-ballot **queue kernel**
  (`_cuda_direct_pairs_queue_kernel!`): per-(warp, source) predicate votes
  evaluate branch-homogeneous instants inline; mixed instants defer their
  source index into per-lane shared-memory queues (depth 8, 8 KB/block)
  drained side-at-a-time, so regularized and singular math are never
  predicated against each other. **Deviation from the plan's global
  bitmask/compacted-index buffer, recorded deliberately:** a body-pair mask
  sized `n_direct × B_t × B_s` cannot be construction-bounded under the
  RadixFMMCache capacity contract (a single fat cell overflows any capacity
  short of n²), while the shared-memory queue achieves the same compacted
  streaming with zero global scratch and no capacity to overflow.
- `:classsplit_ballot` — (c) for the pure buckets + (b) for the mixed bucket.
- Mechanism (a) (`CUDA_NEARFIELD_SUBSORT[]`, orthogonal to the above):
  within-cell sub-Morton ordering (3 extra Morton levels) composed into
  `grid.perm` between the device sort and body packing
  (`_cuda_nearfield_subsort!`: key kernel + block-per-cell odd-even shared
  sort ≤ 1024 bodies + invperm refill). Cell keys/ranges/node metadata are
  unaffected; only within-cell body order (and hence warp-lane spatial
  coherence) changes.

**TwoPassVortex device mirror (Stage-B open risks resolved).** Pass 1 is the
functor/binned path above (branch at `rho_c`). Pass 2
(`_cuda_twopass_deficit_kernel!`) is a warp-per-(cell, offset-entry)
grid-stride sweep over a **construction-built, gap-ascending compacted offset
ball** — the Stage-B "construction-sized compacted class list". Its capacity
bound needs no σ at construction: pass-1 adequacy asserts
`rho_c·σ_max < g_min·h_leaf` per step, so the pass-2 reach `rho_t·σ_max` is
strictly below `(rho_t/rho_c)·g_min` cells — the ball is built once to that
reach (`_twopass_offset_ball`, gap-sorted so the live ball is always a
per-entry prunable prefix against the device `(rho_t·σ_max)²` scalar, reduced
per step from per-cell σ maxima; no per-step list rebuild, no transfer).
Entries whose far corner is inside `rho_c·σ_min(source cell)` are skipped
whole (pass-1-complete). Per-pair shell membership is predicated or
ballot-queued (`CUDA_TWOPASS_PASS2_QUEUED[]`). The host gate gained a
defensive `_twopass_device_reach_check` (can only fire on an internal
inconsistency), and the flat-policy device refusal was retargeted:
hierarchical device caches now construct (`_assert_device_kernel_policy`).

**Contracts.** All recurring work is device kernels on the launch stream:
zero per-step allocation, no transfers (the one offset-ball upload at
construction is counted as a route upload), capture-safe (device-count launch
bounds; epoch-constant host bounds only). Scratch lives in
`CUDANearfieldBinContext` (`containers.jl`) on the hierarchical device
context, built only for split kernels. Mechanism Refs read inside the
lifecycle are baked into a captured graph at record time (documented; tests
and benchmarks use fresh caches per mechanism).

**Homogeneity telemetry (deliverable 3).** `cuda_nearfield_homogeneity(state;
stream=:all|:mixed)` replays the traversal with ballots only and reports
instants / homogeneous fraction / regularized lane fraction (+ queue
push/drain counters from the queue kernels); `cuda_twopass_shell_homogeneity`
does the same for the pass-2 shell predicate (telemetry-only launch, output
untouched).

**Tests.** Host (`binned nearfield pair stream stage C (task 032a)` in
`device_system_interface_test.jl`, ~32 k assertions, green locally): bucket
rule safety by brute-force body-pair sweep (both TF); class-split evaluation
== unbinned partitioned evaluation on the production host pair list at P=4
and P=8, F64/F32 (via the host functor kernel on classified sublists);
offset-ball completeness/ordering/prefix pruning vs the host sweep rule;
gate-derived reach-capacity inequality on a live cache; policy-refusal unit
tests; sub-Morton key semantics + spatial-coherence check. Device
(`cuda_radix_nearfield_binning_test.jl`, cluster): parity of every mechanism
(and both pass-2 modes) against same-P host references at P=4/P=8 × F64/F32,
P=8 F64 accuracy anchor, 023 counter flatness + allocation stability per
mechanism, homogeneity-diagnostic sanity (mixed ≤ all), flat-policy refusal
message, invalid-mode rejection on a fresh cache.

**Measurement plan.** `scripts/cuda_032a_stagec_benchmark.jl` (+
`cuda_032a_{submit,run,fetch}.sh`, data →
`MATRIX_OPERATOR_REFACTOR/data/split_nearfield/`): preflight = interface +
lifecycle + new binning tests; matrix = {regularized baseline; partitioned ×
4 modes × subsort; two-pass × {unbinned, classsplit} × {predicated, queued
pass 2}} × {F32, F64} at three adequate overlap-2 cube points — a: n=1e5
ℓ=3 q=16 (Stage-D geometry), b: n=1e6 ℓ=4 q=12 (deepest adequate depth at
supported radii — ℓ=5 needs |o|²≤22 > 20), c: n=2e5 ℓ=4 q=20 (largest
constructible regularized fraction). Reports step time, isolated
nearfield-stage time, achieved homogeneity (all/mixed/shell), bucket
occupancies, and sampled-direct u/J RMS per configuration.

### Stage C — H200 measurement + mechanism decision (2026-08-06, job 13064834)

Preflight all green on the H200 (interface 1246, lifecycle 216 + 37, **new
binning test 301/301** — mechanism parity at P=4/P=8 × F64/F32, TwoPass pass-2
device mirror, counters/allocation stability, graph capture with the stage-C
kernels, homogeneity sanity). One earlier submission (13064698) failed on a
test-geometry defect — the default near set (`g_min = 1`) is inadequate at
`ell = 3` with `σ_max = 0.04`; fixed by constructing every `ell = 3` test case
with the production-validated `near_radius2 = 16` (the geometry Stage D
benchmarks), not by shallowing or thinning σ.

Benchmark (`data/split_nearfield/cuda032a_stagec_{a,b,c}_*.csv`, overlap-2
uniform cubes, `expansion_order = 3`, dense fused M2L, both precisions,
20-rep steady state). Nearfield-stage time in ms (isolated fill+σ/bin/pass
kernels; step-level times in the CSVs agree on ordering everywhere):

| point (n, ℓ, q) | TF | reg-everywhere | part unbinned | part **classsplit+sub** | part ballot | twopass best | reg frac | best speedup |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| a (1e5, 3, 16) | F32 | 15.97 | 13.88 | **13.00** | 21.57 | 18.89 | 0.098 | 1.23x |
| a | F64 | 35.53 | 29.06 | **26.47** | 48.68 | 39.61 | 0.098 | 1.34x |
| b (1e6, 4, 12) | F32 | 205.7 | 182.8 | **165.4** | 254.5 | 236.1 | 0.087 | 1.24x |
| b | F64 | 472.3 | 392.5 | **373.6** | 572.1 | 566.6 | 0.087 | 1.26x |
| c (2e5, 4, 20) | F32 | 23.02 | 22.08 | **20.24** | 26.61 | 29.53 | 0.200 | 1.14x |
| c | F64 | 50.31 | 45.65 | **41.06** | 55.97 | 60.70 | 0.200 | 1.22x |

Achieved warp homogeneity (fraction of branch-homogeneous warp instants,
`cuda_nearfield_homogeneity`): unbinned stream 0.826 / 0.820 / 0.781 at
a/b/c; **sub-Morton ordering raises it to 0.887 / 0.892 / 0.813**; the mixed
bucket alone is 0.672→0.788 (a), 0.734 (b), 0.690→0.813 (c); the two-pass
shell predicate is 0.620-0.703. Bucket occupancies at the pass-1 cutoff:
a = [33,624 pure-singular, 0 pure-reg, 37,576 mixed] (pure-reg is
geometrically impossible at ℓ=3: min dmax = √3·h > ρ_t·σ_min), b = [187,000,
0, 383,272], c = [310,152, 27,136, 805,216]. Sampled-direct accuracy is
mechanism-independent to 5 digits; u_rel_rms = 5.41e-4 (a), 6.59e-4 (c) pass
the phase gate; **point b at q=12 measures 1.49e-3 — Stage D must use q=16 at
n=1e6/ℓ=4** (032 measured 9.49e-4 there).

**Findings and decision (Checkpoint C):**

1. **Mechanism chosen: `:classsplit` + sub-Morton ordering** — fastest at
   every point and precision by both metrics; now the shipped default for the
   split kernels (`CUDA_NEARFIELD_BINNING[] = :classsplit`,
   `CUDA_NEARFIELD_SUBSORT[] = true`; the overall nearfield default kernel is
   unchanged — that is Stage D).
2. **The §6.3 catastrophe did not materialize**: the unbinned split kernel is
   1.02-1.22x FASTER than regularized-everywhere, not 1.56x slower. The model
   charged both paths to essentially every warp for the whole stream at
   f = 0.31; the constructible geometries have f = 0.09-0.20, hardware
   predication is per-instant, and the measured instant streams are already
   78-83% homogeneous. Binning still pays (classsplit adds up to another
   10-16%), but as a tuning gain, not a rescue.
3. **The ballot/queue mechanism is a measured loss everywhere** (votes,
   queue traffic, and drain raggedness cost more than predication at these
   regularized fractions); retained as a selectable mechanism and recorded.
4. **Two-pass loses to partitioned at every constructible point** (nearfield
   1.4-1.5x slower than the classsplit partitioned kernel), consistent with
   the §6.1 λ* table: its winning regime (ℓ ≥ 5-6 near sets, f ≥ 0.3) needs
   |o|² > 20 near radii that the rigid tables do not currently support, so it
   is unreachable in production. Pass-2 predicated beats the shell queue
   (`CUDA_TWOPASS_PASS2_QUEUED[] = false` stays the default).
5. Step-level gains exceed the isolated-stage gains under the overlapped
   lifecycle (e.g. a/F64: step 70.7 → 40.1 ms while the isolated stage moves
   35.5 → 26.5 ms) — the split kernels issue fewer instructions concurrently
   with the far-field chain. Ordering is identical by either metric; Stage D's
   A/B is step-level and will resolve the attribution.

### Stage D — H200 A/B ladder + recommendation (2026-08-06/07, jobs 13065299/13065376/13065443/13065537)

Checkpoint C was user-approved (mechanism `:classsplit`+sub-Morton accepted as
the split-kernel default); Stage D ran per the plan with the user-endorsed
wake mechanism rider. Harness: `scripts/cuda_032a_staged_ab.jl` — fixed
adequate geometry per case with a baseline-gated escalation ladder, inline
023-counter and allocation-stability gates on every row, per-strategy sampled
erf-based F64 references, and the 028/030 scalar no-regression stage in every
job. All preflights green in every job.

**Primary cases** (step / nearfield-stage ms; all rows pass the 1e-3 velocity
gate; J diagnostic in the CSVs):

| case (n, ℓ, q) | TF | regularized | partitioned | twopass | part ρ_t=4.252 | u gate (part) |
|---|---|---:|---:|---:|---:|---:|
| cube1e5 (1e5, 3, 16) | F32 | 39.2 / 16.0 | **28.6 / 13.0** | 29.6 / 18.3 | 28.3 / 12.5 | 5.41e-4 |
| cube1e5 | F64 | 69.8 / 35.6 | **39.5 / 26.4** | 49.2 / 38.3 | 39.5 / 25.0 | 5.41e-4 |
| wake1e5 (1e5, 5, 16) | F32 | 27.2 / 9.8 | **20.4 / 8.4** | 22.0 / 11.9 | 20.5 / 8.2 | 4.66e-4 |
| wake1e5 | F64 | 46.4 / 21.6 | **29.1 / 17.1** | 36.8 / 24.8 | 28.7 / 16.7 | 4.66e-4 |

**Sentinels** (triggered by the ≤10% F32 partitioned-vs-twopass step margin at
n=1e5; run per the ladder rule):

| case (n, ℓ, q) | TF | regularized | partitioned | twopass | verdict |
|---|---|---:|---:|---:|---|
| cube1e6 (1e6, 4, 16) | F32 | 383 / 287 | **302 / 213** | 367 / 275 | no reversal (21% ahead) |
| cube1e6 | F64 | 824 / 673 | **614 / 485** | 799 / 671 | no reversal (30%) |
| wake1e6 (1e6, 6, 16) | F32 | 332 / 176 | **276 / 132** | 319 / 164 | no reversal (16%) |
| wake1e6 | F64 | 617 / 413 | **499 / 300** | 621 / 403 | no reversal (24%; twopass ≈ baseline) |
| wake1e3 (1e3, 3, 16) | both | 3.03-4.59 | 3.05-4.60 | 3.06-4.61 | tie (nearfield ≤ 0.2 ms) |
| cube1e3 (1e3, 1, 16) | both | 8.96-14.5 | 8.95-14.5 | n/a (flat fallback refuses TwoPass) | degenerate tie |

Accuracy: cube1e6 9.30e-4, wake1e6 4.80e-4, wake1e3 1.16e-4, cube1e3 4.1e-5 —
every recorded row passes the gate (the ladder never needed to escalate;
q=16 everywhere per the Stage-C note). **The winner never reverses: the full
024b grid is not required.** Deeper trees widen partitioned's lead — the
§6.1 two-pass deep-tree advantage assumed near sets tight to the cutoff ball,
but the supported rigid radii cap the near set at |o|² ≤ 20, so pass 2's
sweep re-traverses volume the pass-1 near set already covers.

**ρ_t RMS lever (deliverable 4): confirmed on BOTH cases.** ρ_t = 4.252
changes sampled u error by ≤ +0.6% relative (5.43e-4 cube1e5, 4.66e-4
wake1e5, 9.35e-4 cube1e6, 4.80e-4 wake1e6 — all under gate) and buys 1-6%
nearfield time. Eligible for adoption; left un-shipped pending the user
default decision.

**Wake mechanism rider: no reversal.** wake1e5 F32 step: unbinned 26.2 vs
classsplit+sub 20.4 (F64 44.1 vs 29.1); wake1e6 F32 317 vs 276 (F64 569 vs
499). The cube-measured mechanism ordering holds on the wake; homogeneity
0.757→0.887 (wake1e5), 0.807→0.864 (wake1e6) with sub-sort.

**Contract gates:** 023 counters flat and allocation-stable on every recorded
row (inline gates — a violation aborts the row); scalar 028/030
no-regression: verdict 6.58 ms F32-fp16 / 17.83 ms F64 vs the 13059638 rows
of record 9.45 / 20.50 ms, grad_err identical to 7 digits — no regression.

**Recommendation to the user (deliverable 7; not shipped):**

1. Make `PartitionedVortex` (with the approved classsplit+sub-Morton stream)
   the resident vortex-nearfield default: it wins every case, precision, and
   n measured, by 1.16-1.77x step-level over the shipped
   regularized-everywhere baseline, with identical gate accuracy. No
   per-regime split is justified — the win is uniform (degenerate n=1e3
   cases are ties, not reversals).
2. Adopt the §6.4 RMS radius `rho_t = 4.252` as the split-kernel default
   (confirmed on both cases; worst gate margin 9.35e-4 at cube1e6).
3. Retain `TwoPassVortex` as a supported alternative (its niche —
   near sets tight to the cutoff ball at depth — needs |o|² > 20 radii that
   don't exist yet); retain `RegularizedVortex` as the divergence-proof
   fallback and default for consumers who skip the split kernels.

Benchmark-harness postmortems recorded for reproducibility: sentinel jobs hit
(a) a GC/pool-residue false rejection of the dense free-memory preflight,
(b) sbatch `--export` consuming commas in `FM032A_CASES`, and (c) whole-level
windows sizing ~22 GB route capacity at ℓ=6 (bounded with
`window_classes=64`); all fixed in the harness (commits `dd404ab`,
`7161c1a`, `a5a0872`).

### Close-out — Checkpoint D approved, defaults shipped (2026-08-07)

**Checkpoint D user approval (2026-08-07, via coordinator):** all three
recommendations accepted — (1) `PartitionedVortex` with the classsplit +
sub-Morton stream is the resident vortex-nearfield default, (2)
`rho_t = 4.252` is the split-kernel default radius, (3) `RegularizedVortex`
remains the fallback/default for non-split consumers and `TwoPassVortex`
stays a supported alternative.

**Shipped surfaces** (default/constructor values only):

- `PartitionedVortex`/`TwoPassVortex` constructor default `rho_t = 4.252`
  (`RegularizedVortex` keeps the §4 per-pair 4.789 — its math is
  `rho_t`-independent; the field only feeds the adequacy gate).
- Docstrings + `docs/src/device_interface.md`: `PartitionedVortex` documented
  as the recommended default for σ-carrying vortex systems, with the measured
  1.16-1.77x step-level margin; `RegularizedVortex` as the divergence-proof
  fallback; `TwoPassVortex` as the supported alternative. The
  `_default_direct_kernel(Point{Vortex})` trait default stays
  `SingularVortex` (a parameter-free default cannot know `sigma_row`;
  σ-carrying consumers opt in via the `direct_kernel(system)` trait, exactly
  as the 031/032 interface spec defines).
- The stream-mechanism defaults (`CUDA_NEARFIELD_BINNING = :classsplit`,
  `CUDA_NEARFIELD_SUBSORT = true`) shipped at Stage C under the Checkpoint C
  approval.

**No hardware re-run needed:** the shipped default combination
(partitioned/two-pass at `rho_t = 4.252` under classsplit + sub-Morton) is
bitwise the `partitioned_rms`/`twopass_rms` Stage D configurations measured
on H200 in jobs 13065299/13065376/13065537 (every case passes the 1e-3 gate;
worst margin 9.35e-4). No default combination exists that was not exercised
on hardware.

**Tests:** default assertions added (`shipped nearfield defaults` testset:
constructor values, unchanged plain-vortex trait default, trait-driven
end-to-end pickup at P=4 and P=8; CUDA Ref defaults asserted at the top of
`cuda_radix_nearfield_binning_test.jl` before any mutation); the Stage A/B
pair tests' ρ draw ranges adapted to the 4.252 branch point (their
shell/tail bounds hold unchanged). Full local suite green (`--threads=4`).

**Deliverable checklist (task file §Deliverables):**

1. Partitioned nearfield — **done** (Stage A host + Stage C device, shipped
   default).
2. Two-pass additive correction — **done** (Stage B host + Stage C device
   mirror; supported alternative).
3. §6.3 binned/sorted pair stream before any A/B — **done** (Stage C:
   mechanisms built and measured; classsplit + sub-Morton selected; achieved
   homogeneity reported in every benchmark row).
4. `rho_t` lever — **done** (measured on both cases; RMS radius adopted as
   the split-kernel default per user approval).
5. Geometry correctness — **done** (adequacy gate + exact-once host tests in
   Stage A; construction-time refusals; gate-derived pass-2 reach capacity).
6. Profile-triggered A/B on H200 — **done** (primary cases + all four
   sentinels; winner never reverses, so the full 024b grid was not
   required; divergence/homogeneity, pair counts, and step/stage timings
   recorded).
7. Default selection — **done** (recommended at Checkpoint D, user-approved
   2026-08-07, shipped in this close-out).
8. Contract gates — **done** (P=4 in every new testset, 023 counters flat,
   allocation stability, scalar 028/030 no-regression in every job).

Task 032a is complete pending clear-context approval by a different agent.

## Placement and Reporting

- Follow the `_batched`/`*_cuda.jl` placement rules; types stay in
  `containers.jl` and CUDA remains optional.
- FastMultipole commits in this repository only.
- Record measured tables and the user-approved default decision here.
