# 052d — Host `fmm!` profile at production shape (panels → particles, step 472)

**Date:** 2026-08-28. **Machine:** local Mac (Apple M2, 4 performance threads — hard cap per
local-run policy), Julia 1.12.5, FastMultipole branch `flowpanel-20260817`, FLOWPanel live
worktree. **Question:** why did the 052d host-FMM leg "panels → particles" measure
**15.8 s/step median at 32 threads on the A100 node** (min 13.2 / max 25.0 over 104 steps)
vs the 0.2–0.5 s @ 64-thread design estimate and the 3.3 s dense GPU kernel it replaced.

## 1. Setup / geometry provenance

- **Targets:** 241,986 wake-particle positions from
  `orc:~/FLOWPanel-052/data/fm052d_gpu_1080/fm052d_gpu_1080_wake1_particles/…472.vtp`
  (production snapshot, step 472), fed through `FastMultipole.ProbeSystemArray` exactly as
  `FLOWPanel._panel_fmm_evaluate!` does.
- **Sources:** the production panel body rebuilt from the step-472
  `fm052d_gpu_1080_body1.472.vtu` (production pose and scale): **36,752 triangular panels**,
  18,380 nodes — cell-for-cell identical to
  `examples/data/dji9443_20260725_45_185_capped_captess4.msh` (env `RHPC_MESH=45_185_ct4`).
  Note: the design estimate assumed "~9k panel sources"; the actual body is **4× that**.
  Built as `RigidWakeBody{Union{ConstantSource,VortexRing}}(nodes, cells, noshedding;
  watertight=true, DBC=true)` with random nonzero strengths;
  `calc_normals!`/`calc_controlpoints!` applied; production core sizes
  (`core_size_panel = R·1e-10`, `core_size_targets = 1e-3` — fm052 env sets no
  `CORE_SIZE_*` overrides) with `_set_core_sizes!(:core_size_targets)` active, as in the
  production pass. The two trailing-wake strip bodies (~KB-scale) were omitted (negligible).
- **Call:** `fmm!((probes,), (body,); expansion_order=4, multipole_acceptance=0.4,
  leaf_size_source=50, scalar_potential=false, gradient=true, hessian=false, shrink=true)`
  — the probe-job configuration (`PANEL_INFLUENCE_FMM_P=4`, θ and leaf at their env
  defaults 0.4 / 50).
- Extracted arrays + index for reuse:
  `/private/tmp/claude-502/…/1a9c539a-…/scratchpad/snapshot472/SNAPSHOT_INDEX.md`.
  Scripts: `…/scratchpad/profile_fmm.jl`, `…/scratchpad/addendum_fmm.jl`; logs
  `profile.log`, `addendum.log`, `diag.log`, `profile_degenerate.log`.

**Reconstruction pitfall (recorded for the sibling agent):** building the body without
computing controlpoints leaves `FastMultipole.get_position` all-zero, `center_box`
degenerates, the source tree never subdivides (1 leaf), and the call runs a de-facto dense
evaluation — 6.89e9 direct interactions, 437–439 s at 4 threads (`profile_degenerate.log`).
All numbers below are from the corrected, production-faithful body.

## 2. Total wall time and stage breakdown (production config, 4 threads)

Total production-shape call, after JIT warmup, 3 reps: **44.08 / 44.00 / 44.14 s**.
`ProbeSystemArray` alloc + position copy: 0.06 s cold, 0.005 s warm — negligible.

Stage breakdown (manual replication of the exact `fmm!` multithreaded CPU pipeline;
clean rep):

| stage | time (s) | share |
|---|---|---|
| Cache alloc | 0.006 | — |
| target tree (242k probes, shrink=true) | 0.016 | 0.04% |
| source tree | 0.009 | 0.02% |
| interaction lists (build+sort) | 0.001 | — |
| **near-field direct** | **44.03** | **99.8%** |
| upward pass (B2M+M2M) | 0.010 | 0.02% |
| horizontal pass (M2L, 13,228 ops) | 0.003 | — |
| downward pass (L2L+L2B) | 0.010 | 0.02% |
| buffer→target writeback | 0.002 | — |
| **sum** | **44.09** | |

Tree/list stats at this config: target tree 8,554 branches / 7,410 leaves; **source tree
only 125 branches / 76 leaves (~483 panels per leaf — `leaf_size_source=50` never
binds)**; 13,228 M2L pairs; 39,228 direct pairs = **6.87e8 direct panel–particle
interactions** (vs 8.89e9 dense, i.e. the FMM culled only 92% of the dense work).
Throughput: 15.6M panel-interactions/s at 4 threads (~3.9M/s/thread).

### Why the near field is this large: core-size radius inflation floors the source tree

`source_system_to_buffer!` inflates every panel's FMM radius by
`radius_inflation(VortexRing, core_size, tol)` with active `core_size =
core_size_targets = 1e-3` and `FMM_RADIUS_TOL = 1e-6` → **Δr = 5.898e-3**, i.e. ~3.7×
the mean bare panel radius (1.6e-3) and >300× the smallest (the `@warn` at
`FLOWPanel_abstractbody.jl:1202` fires). FastMultipole then stops source subdivision
when the child radius falls below the largest body radius (`tree.jl:521`,
`child_radius >= max_body_radius` guard), so the source tree bottoms out at 76 fat
leaves regardless of `leaf_size_source`, and the fat radii additionally fail the MAC for
many pairs. A/B proof: setting `FMM_RADIUS_TOL[] = Inf` (no inflation) at the same
(θ=0.4, leaf=50) gives 1,223 source branches / 928 leaves, direct interactions drop
6.87e8 → 1.34e8, and wall time drops **44.0 → 8.7 s** (accuracy falls to 3.1e-4 — the
inflation is there to control regularized-vs-singular mismatch, so this is a tradeoff,
not a free win).

## 3. Parameter sweep (total `fmm!` wall time, s, 4 threads, p=4, 1 rep after warmup)

| θ \ leaf_size_source | 10 | 20 | 50 | 100 |
|---|---|---|---|---|
| 0.4 | — | 27.6 | **44.0** (prod) | 65.0 |
| 0.5 | — | 15.2 | 24.6 | 40.5 |
| 0.6 | 5.7 | 8.9 | 14.5 | 25.2 |
| 0.7 | 3.7 | 6.0 | — | — |
| 0.8 | — | 4.1 | — | — |

Also: shrink=false at (0.4, 50): **31.5 s** — faster than shrink=true's 44.0 s at this
config. And with `FMM_RADIUS_TOL=Inf`: (0.4,50) 8.7 s, (0.4,20) 5.5 s, (0.6,20) 2.0 s.

Smaller `leaf_size_source` helps mostly because it also sets `leaf_size_target`
(finer target leaves → finer-grained MAC decisions), since the source leaves are
radius-floored anyway. Near-field time tracks the direct-interaction count at a constant
~15.5M inter/s: e.g. (0.6, 20) has 1.37e8 interactions → 9.0 s, and tol=Inf (0.4, 50)
has 1.34e8 → 8.6 s.

## 4. Accuracy (rel RMS of gradient vs `direct!` on 2,000 random targets)

| config | time @4T (s) | rel RMS err |
|---|---|---|
| p4 θ0.4 leaf50 (production) | 44.0 | 2.0e-5 |
| p4 θ0.5 leaf100 | 40.5 | 2.2e-5 |
| p4 θ0.5 leaf20 | 15.3 | 5.9e-5 |
| p4 θ0.6 leaf100 | 25.2 | 4.7e-5 |
| p4 θ0.6 leaf20 | 9.0 | 9.1e-5 |
| p4 θ0.7 leaf20 | 6.0 | 1.9e-4 |
| p4 θ0.8 leaf20 | 4.1 | 4.2e-4 |
| p4 θ0.6 leaf10 | 5.7 | 2.9e-4 |
| p4 θ0.4 leaf50, tol=Inf | 8.7 | 3.1e-4 |
| p4 θ0.6 leaf20, tol=Inf | 2.0 | 1.1e-3 |

The production config reproduces the probe-measured ~5e-5 regime (2.0e-5 here with
random strengths). Within that regime (≤ ~6e-5) the best sweep point is **(θ=0.5,
leaf=20): 15.3 s** — 2.9× faster than production settings. Tolerating ~1e-4 buys
**(θ=0.6, leaf=20): 9.0 s** (4.9×). Beyond that, accuracy leaves the regime quickly.

## 5. Conclusions

**(1) Where the 15.8 s goes.** Near-field direct panel evaluation is **99.8%** of the
call; every other stage (both tree builds, lists, B2M/M2M, 13k M2L, L2L/L2B, writeback,
probe copy) totals < 0.1 s at 4 threads. The near field is enormous because the active
`core_size_targets = 1e-3` inflates panel FMM radii by 5.9e-3 (Gaussian gradient-aware
rule at tol 1e-6), which (a) floors source-tree subdivision at ~483 panels/leaf via the
`child_radius >= max_body_radius` guard and (b) fails the MAC for many pairs: 6.87e8
direct interactions, only 92% of dense work culled. The A100-node measurement is
consistent: 6.87e8 / 15.8 s = 43.5M inter/s at 32 threads (1.36M/s/thread — ~3×
slower per thread than the M2, plausible for the older server cores under memory
contention). The design estimate also assumed ~9k panels; the actual body has 36,752,
a further ×4 on near-field work.

**(2) Can any host parameter choice reach ~0.6 s/step at 64 threads?** No — not at
production accuracy. Best 4-thread time in the ~5e-5 accuracy regime is 15.3 s
(θ=0.5, leaf=20). Naive linear extrapolations (optimistic — they ignore NUMA, memory
bandwidth, and load imbalance over 18k direct pairs, and the A100 node's measured
per-thread rate is ~3× *worse* than the M2's): ×(4/32) → 1.9 s, ×(4/64) → 0.96 s.
Using the A100 node's own measured throughput instead, (θ=0.5, leaf=20)'s 2.4e8
interactions would take ~2.8 s at 64 threads. Even the accuracy-degraded extremes
(θ=0.7–0.8 or tol=Inf, err 2e-4–1e-3) only reach 0.5–1.1 s under naive-linear
64-thread scaling — i.e. the gate is touched only by configs that abandon the accuracy
target, on scaling assumptions the production node demonstrably does not meet. The
0.2–0.5 s estimate is unreachable on host.

**(3) Implications for a device dual-tree design.** Only the near-field direct stage
must move to device — it is >99% of the time and is exactly the kind of regular
block-sparse panel-kernel work the dense GPU path already does at 3.3 s for the FULL
dense product (8.89e9 pairs ≈ 2.7e9 pairs/s); the production-config near field (6.9e8)
is ~0.26 s at that rate, and a fixed-radius tree (below) would leave ~1.3e8 → ~0.05 s.
Host can keep trees, lists, and all expansion passes: at 4 threads they total < 0.1 s,
well under any 0.6 s gate even before device offload of L2B. Independent of the device
work, two host-side levers matter: (i) revisit the radius-inflation policy for this leg
(compact-support regularization gives Δr = rc = 1e-3 instead of 5.9e-3; or a looser
`FMM_RADIUS_TOL` for velocity-only passes) so `leaf_size_source` binds again — the
tol=Inf A/B shows a ~5× near-field reduction is available if the accuracy cost is
managed; (ii) ship non-default (θ=0.5–0.6, leaf=20) instead of (0.4, 50) — 3–5×
cheaper at equal-or-near accuracy, and worth folding into `_panel_fmm_*` defaults.

*Cross-checks: stage sums match totals (44.09 vs 44.0–44.1); (θ0.6, leaf20) reproduced
across scripts (8.93 / 8.96 s); near-field time ∝ interaction count across radius-tol
A/B (1.37e8→9.0 s vs 1.34e8→8.6 s); degenerate-tree run (439 s ≈ dense/4-threads)
consistent with dense pair count.*
