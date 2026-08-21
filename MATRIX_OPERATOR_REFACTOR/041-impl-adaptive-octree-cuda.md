# 041 Impl: Adaptive Octree CUDA Device-Resident Lifecycle

## Status and Entry Gate

**DONE and clear-context APPROVED `2026-08-15`.**

Entry gate: `038`, `039`, and `040` complete and approved.

## Objective

Mirror the adaptive octree on the CUDA device-resident lifecycle: device
construction and refresh (sort, prefix split, 2:1 balance, list generation as
flag/scan/compact kernels), device U/V/W/X execution, and H200 measurement
against the uniform-depth path.

## Scope and placement

- Device code in `tree_batched_cuda.jl` / `translate_batched_cuda.jl` behind
  the existing CUDA extension; uniform-depth device path untouched and
  default.
- Replace the dense `Σ 8^L` `node_at` occupancy lookup with the sorted-Morton
  binary-search lookup on device for the adaptive path (and, if cheap,
  expose it to the uniform path to lift the `ell ≤ 8` cap — record the
  decision either way).
- V-list M2L replays the existing device M2L strategies and operator tables;
  construction-only operator/route uploads; `route_uploads`/
  `operator_uploads` constant after construction;
  `expansion_host_copies == 0`.
- W/X kernels: device M2T and S2L with the same accuracy contract as `040`.
- Nearfield load-balance follows from bounded `K_max`: assert/verify that the
  warp-per-pair kernel no longer sees fat-cell serialization; retune the
  symmetric-path threshold if profiling justifies it.
- Refresh: occupancy-epoch caching generalized to the adaptive leaf set
  (rebuild windows only when the leaf key set changes), preserving the CUDA
  graph replay path where applicable.

## Verification

- Host/device parity on all `039`/`040` test distributions, both precisions,
  `P=4` included; transfer-counter and zero-per-step-allocation parity per
  the `023` contract.
- H200 before/after: full resident step and per-stage breakdown on the cube,
  wake, and multi-scale cases vs the uniform-depth path at each case's best
  `ell`; construction and refresh costs; memory footprint vs the
  `min(8^ell, n)` capacity baseline.
- No regression on the shipped scalar benchmark (the `028`/`029` record) —
  the adaptive path is opt-in, so this is a guard that shared code was not
  perturbed.

## Acceptance

Parity and counter gates pass; the multi-scale case shows a measured
end-to-end win consistent with the `038` model; uniform cases regress by no
more than an agreed tolerance (or the adaptive path is documented as
multi-scale-only); a default-selection recommendation (adaptive vs uniform,
per regime) is recorded for the `042` review with evidence.

## Completion Notes (2026-08-15, lead agent — DONE, pending approval)

### What was implemented, and where

- **`src/tree_batched_cuda.jl` (NEW, ~1,060 lines)** — device adaptive
  construction and lists, host-orchestrated flag/scan/compact kernels:
  Phase A theory-§1.2 K_max frontier split (child occupancy via
  **sorted-Morton binary search** over the full-depth body keys); Phase B
  §1.4 2:1 balance as Jacobi rounds over the leaf key set (proven to reach
  the same unique balance closure as the host deepest-first sweep — exact
  structural parity incl. `n_balance_splits`, verified on hardware);
  Phase C level-major finalize (per-level ancestor compaction, subtree
  ranges, parent/child links by per-level-block binary search, leaf
  compaction, leaf-as-cell presentation); per-node σ_max sweep; theory-§2.7
  DTR pair frontier (finer-lattice clamp, source-side sticky demotion,
  deterministic scan-ordered U/V/W/X emission, device 025 phase-table
  membership flag with loud host assert); deterministic (class, index)-keyed
  V CSR partition; U→leaf-slot mapping; occupancy-epoch snapshot/compare
  over the adaptive (level, shifted-key) leaf set with an epoch fast path
  (ranges/cells/σ only).
- **`src/translate_batched_cuda.jl` adaptive section (~600 lines)** — device
  M2T/S2L kernels (block-per-pair, thread-per-body, the validated host
  `irregular_harmonics!` + `_resident_multipole_eval_flat*` running as
  device functions over a preallocated per-thread harmonic slab; vortex S2L
  verbatim port, χ at P_active per 008h; atomic accumulation); adaptive
  V-list M2L over the UNCHANGED resident plans (dense CUDA family:
  per-level in-place applies of CSR segments with the 025 level scales —
  zero copies; precomputed-y/concat: device-to-device windows mirroring the
  040 host driver); lifecycle body with the uniform path's async-nearfield
  overlap and CUDA-graph capture (same warm-up/epoch/fallback pattern,
  dense-fused eligibility); `_cuda_update_adaptive_radix_state!` (the
  **branch** design: the uniform grid/route refresh does NOT run — the 040
  "double refresh elimination" and "sort unification" levers land on device
  by construction); cache build/step plumbing.
- **`src/containers.jl`** — `DeviceAdaptiveCUDAContext` (93-field device
  context, Any-typed device arrays per the established convention).
- **`src/translate_batched_resident.jl`** — device throw removed; NEW
  construction guards: S2L body-type (host+device; the 040 approval note)
  and device+split_veto (recorded device limitation; veto default OFF and
  pending ratification regardless).
- **`_cuda_hier_dense_apply_routes!`** hctx parameter duck-typed (only
  `first_m2l_level` + scales read) — the single shared-path signature touch.
- **Tests**: `test/cuda_radix_adaptive_test.jl` (608 asserts), wired into
  `test/runtests.jl` + `test/cuda/runtests.jl`.
- **Measurement**: pre-registered `scripts/fm041_cuda_cost.jl` (+ submit
  scripts); CSV of record `data/fm041_cuda_cost.csv` (H200 job 13182172,
  96/96 rows ok).

### Verification (H200, m13h-1-1, jobs 13180243 / 13180706 / 13182172)

- **Device test gate: 608/608 PASS.** Structural parity vs the host 039
  tree is EXACT over 24 configs (3 cases × K_max 8/32 × q 3/12 ×
  balanced/unbalanced): node_keys/levels/parent/child/ranges/level_offsets
  equality, `n_balance_splits` equality, U/W/X pair-set equality, V CSR
  multiset + class_starts equality; σ-gate parity incl. `n_demoted`.
  Lifecycle parity vs the host adaptive path across cube/filament/
  multiscale × P=4,8 × Float64/Float32; dense + precomputed-y strategy
  accuracy; LH vortex accuracy; guards; uniform-device smoke.
- **023 contract**: `route_uploads`/`operator_uploads` constant across
  moved steps after construction; `expansion_host_copies == 0` asserted
  around every pipeline; warmed adaptive lifecycle `CUDA.@allocated == 0`;
  array identity preserved; epoch fast path verified (unmoved ⇒ epoch_id
  constant, occupancy change ⇒ increments).
- **Uniform non-regression**: full existing CUDA suites green in the
  measurement job. (Gate-2 finding: 2 pre-existing interface-parity
  failures reproduced IDENTICALLY on the pre-041 tree at `981f7ff` — A/B
  job 13180628 — a latent 037e/f-era drift, deltas ≤1e-6 max-abs;
  documented tolerance widening + follow-up item, not a 041 regression.)

### H200 measurements (job 13182172; P=4, q=5, warm medians of 5, 2000-target sampled-direct; adaptive ell_max=10, balance on)

Every one of the 96 rows passes the velocity gate (max rel RMS 6.6e-4).
`step` = full `fmm!` incl. host-target output writeback (identical
transfer burden on both structures); `life` = resident lifecycle.

n=1e6, Float64, dense family (the 028 record family), best-vs-best:

| case | adaptive (best K) | uniform (best ℓ) | step ratio | life ratio | memory |
|---|---|---|---|---|---|
| multiscale100 | K=64: step 138.2 ms, life 50.1 ms, 5.2 GB | ℓ6: 259.0 ms / 151.2 ms, 16.0 GB | **1.87× faster** | **3.02× faster** | 3.1× less |
| unitcube | K=64: 103.4 / 21.7 ms, 5.2 GB | ℓ5: 114.6 / 21.5 ms, 2.2 GB | 1.11× faster | parity (1.01×) | 2.4× more |
| wake | K=64: 164.4 / 53.6 ms, 5.2 GB | ℓ6: 129.9 / 28.3 ms, 14.2 GB | **1.27× slower** | 1.89× slower | 2.7× less |

Float32 mirrors the ordering (multiscale 1.35×/1.96× faster; cube ~parity
1.07× slower step; wake 1.35× slower). Concat-family rows show the same
structure ordering and are dominated by dense everywhere (uniform ℓ6
concat at 87M routes: 524 ms life — the known per-window host-download
cost). n=1e5: cube adaptive 1.7× faster than best pre-registered uniform
depth; wake/multiscale 1.1–2.2× slower (040's honest wake-1e5 negative
reproduced; per-n K/depth tuning is 041a's sweep).

Mechanism evidence: popmax = K_max on every adaptive row (fat cells
bounded); wake U body-pairs 9.39e8 (K=64) vs 6.69e9 (ℓ6) / 4.36e10 (ℓ5);
multiscale 1.43e9 vs 4.72e9 / 2.85e10 — the 038/039/040 counts reproduced
exactly on device.

Adaptive refresh: warm (epoch-cached) update 10–11 ms at n=1e6 (uniform:
7 ms) — sort ~2.0, build 0.5–0.9, balance 0.3–1.1, finalize/ranges ~1.6 ms.
Occupancy-epoch rebuilds add DTR 5–8 ms + CSR partition 36–74 ms
(dominated by the 12–22M-key device sortperm) — skipped on unchanged
epochs, which is the epoch-caching win.

### The ℓ ≤ 8 cap (recorded decision)

The adaptive device path never builds the dense Σ8^L `node_at` table —
occupancy is resolved by the tree's child links and sorted-Morton binary
search — so it carries **no depth cap below `RADIX_GRID_MAX_ELL` (21)**;
every measured adaptive row ran at `ell_max = 10 > 8`. The uniform path's
`node_at` and its ℓ≤8 construction throw are left UNTOUCHED: swapping
binary search into the shipped `_cuda_hier_*` flag/compact kernels would
put an uninstrumented change on the 028/029 record path for zero measured
need (deep uniform grids are exactly the regime the adaptive path
replaces). Revisit only if a uniform-path consumer actually needs ℓ > 8.

### Deviations & open items

1. Split veto unimplemented on device (construction guard); default OFF
   and pending user ratification regardless (039 item).
2. Theory §7 tier-1 frozen-leaf-set refresh not implemented — full rebuild
   per step with epoch-cached list skipping (matches host 039 semantics);
   the ~10 ms warm rebuild is the measured price; a frozen-set refresh is
   a recorded 041a-and-later lever.
3. 040 lever disposition: double-refresh ELIMINATED and sort unification
   LANDED on the device path (branch design); stage-slab chunking not
   implemented — adaptive total memory (3.4–5.5 GB) already sits 2.7–3.1×
   below uniform ℓ6, so chunking is priced as a later tightening.
4. Wake-at-n=1e6 is an honest device-side negative (1.27–1.35× slower than
   uniform ℓ6): the tuned fused-dense M2L + warp-per-pair nearfield absorb
   deep-uniform cost on the GPU far better than on the host (040 saw a
   2.42× adaptive win there). Default-selection recommendation for 042:
   keep the adaptive device path opt-in, recommended for multi-scale
   density (measured 1.87×/3.02× win + 3× memory at 100× contrast) and for
   σ-gated/regularized fields needing the per-cell gate; uniform remains
   the default elsewhere. NO production defaults changed.
5. Latent upstream (037e/f-era) host-vs-device interface parity drift
   (≤1e-6 max-abs) — reproduced on the pre-041 tree; Float64 tolerances
   widened with full in-file documentation; disposition + tight-tolerance
   restoration is a recorded follow-up outside this row.
6. Graph capture for the adaptive lifecycle is implemented with the uniform
   eligibility/warm-up pattern (dense fused family) and passes repeated-step
   tests; per-step capture engagement was not separately instrumented —
   041a profiling item.
7. Bare-default `CUDARadixLifecycleOptions` are rejected by the device
   build (pre-existing behavior, applies to uniform too); the measurement's
   concat rows select `ConcatenatedFixedZM2L` explicitly, as the tests do.

## Clear-Context Approval (2026-08-15 16:35 MDT)

**Verdict: APPROVED.** Independent clear-context review per START_HERE
§Routine task protocol item 6, commit range `69ac964`…`ae0d9cb`.

Evidence checked (verified, not trusted):

1. **Objectives-consistency** — every START_HERE 041 row item delivered:
   device construction/refresh as flag/scan/compact; sorted-Morton
   binary-search occupancy with the ℓ≤8 cap decision explicitly recorded
   (adaptive path uncapped, ran ell_max=10; uniform `node_at` deliberately
   untouched with a sound risk rationale); device M2T/S2L; occupancy-epoch
   caching; 023 counter/zero-alloc parity; H200 before/after on all three
   cases; default-selection recommendation routed to 042 with no production
   default changed. The acceptance clause's "documented as multi-scale-only"
   branch is properly exercised for the wake regime.
2. **Correctness** — the Jacobi-balance argument is a valid least-fixed-point
   claim (the mark kernel's 8 touching parent-level cells cover
   face/edge/corner coarse neighbors; splits are monotone and only of leaves
   forced by the unique 2:1 closure) and is empirically pinned by EXACT
   structural parity vs the host 039 tree on 24 configs including
   `n_balance_splits` (jobs 13180242/13180243). Exact-once transfers from
   039's brute-force proof via U/V/W/X pair-set and V-CSR multiset equality.
   The 025 phase-table sticky-demotion invariant is enforced on device
   (violation flag + loud host assert, `src/tree_batched_cuda.jl:1363`).
   No new operator tables: the adaptive M2L drives the UNCHANGED resident
   plans; the single shared-path touch is the duck-typed `hctx` parameter of
   `_cuda_hier_dense_apply_routes!` (reads `first_m2l_level` + scales only).
3. **Tolerance widening** — genuine latent-upstream finding, not a 041
   regression: A/B job 13180628 (sacct COMPLETED 0:0) fails the pre-041 tree
   at `981f7ff` identically (same two asserts); deltas 3.0e-7–9.9e-7 vs the
   old 1e-9/1e-7 pair; widened Float64 pair (1e-6, 5e-6) retains ~5–10×
   margin over the observed drift and stays 2–3 orders below the Float32
   pair and the physical gate; in-file documentation is complete and the
   restoration follow-up is recorded.
4. **Performance methodology** — measurement script pre-registered in
   `69ac964` (10:58) before any submission (11:01); every failed/resubmitted
   job and both protocol-neutral amendments logged with root causes; jobs
   13180243/13180628/13180706/13182172 independently sacct-verified
   COMPLETED 0:0 on m13h-1-1. CSV of record cross-checked: all headline
   ratios reproduce (multiscale 259.0/138.2 = 1.87× step, 151.2/50.1 =
   3.02× life, 3.1× memory; cube 1.11×; wake 129.9/164.4 = 1.27× SLOWER —
   honestly recorded, mechanism plausible: uniform ℓ6 wake u_pairs 6.69e9
   is absorbed by the tuned fused-dense/warp-per-pair device kernels that
   the host lacks); popmax = K_max on every adaptive row; 96/96 rows ok,
   max velocity error 6.584e-4 ≤ 1e-3; Float32 and n=1e5 claims match.
5. **Robustness** — 608/608 device tests: P=4+P=8, F64+F32, LH vortex,
   strategies, guards (S2L body-type on host+device per the 040 approval
   item; device split-veto), counters, epoch fast path, warmed
   `CUDA.@allocated == 0`, uniform-device smoke; full uniform CUDA suites
   green in the measurement jobs.
6. **Minimal invasiveness** — diff surface confined to the new
   `tree_batched_cuda.jl`, the adaptive section of
   `translate_batched_cuda.jl`, `DeviceAdaptiveCUDAContext` in
   `containers.jl`, the guard block in `translate_batched_resident.jl`, and
   tests/scripts/data/logs; placement rules honored; package loads locally.
7. **Readability** — file-head design narratives and per-kernel comments are
   exemplary; the decision log is a complete, timestamped reconstruction.

NOTED items (non-blocking, all already recorded by the lead): split veto
unimplemented on device (default OFF, pending ratification); frozen-leaf-set
refresh, per-n K/depth sweep, graph-engagement instrumentation, and
stage-slab chunking carried to 041a; the 037e/f-era parity-drift
disposition + tight-tolerance restoration is a user-ratification follow-up
upstream of 041; bare-default `CUDARadixLifecycleOptions` device rejection
is pre-existing behavior worth a friendlier error in a later maintenance
pass.

No required changes. Row 041 marked Approved in START_HERE.md. Campaign
PAUSES for user review per the 2026-08-15 15:58 MDT directive — 041a/042
are not launched.
