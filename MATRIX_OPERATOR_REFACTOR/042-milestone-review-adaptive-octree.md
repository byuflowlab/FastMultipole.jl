# 042 Milestone Review: Adaptive Octree Arc (038–041a)

## Status and Entry Gate

**DONE `2026-08-20`; clear-context approval pending.**

Entry gate: `038`, `039`, `040`, `041`, and `041a` complete and approved.
Per explicit user direction on `2026-08-20`, this review excludes every task
from `041b` through `041j`. Task `041k` was independently clear-context
reviewed immediately before this milestone and is not part of its evidence.

## Scope

Blocking Milestone Review for the adaptive octree arc, per the standard
Milestone Review protocol in `START_HERE.md` (read
`../MATRIX_OPERATOR_REFACTOR.md` and `START_HERE.md`, inspect the completed
task files and artifacts, confirm consistency with the background design and
gating, look for further GPU-speedup contributions, record notes, obtain
clear-context approval).

Review items specific to this arc:

1. Exact-once coverage and 2:1 balance evidence: proofs, computational
   verification breadth, adversarial distributions.
2. Operator-table reuse: confirm the V-list path introduced no new operator
   tables or strategy forks; audit any recorded deviation.
3. M2T/S2L accuracy: error bounds vs measured sampled-direct errors,
   Lamb-Helmholtz coverage, `P=4` behavior.
4. Performance verdict: multi-scale win, uniform-case non-regression, the
   default-selection recommendation from `041` audited against the `041a`
   figures and report, and whether the dense `node_at` replacement should
   lift the uniform path's `ell` cap.
4a. Reporting quality: the `041a` figures compile, match their CSVs, compare
   only at matched stated accuracy, and communicate the old approach's
   weakness and the new approach's time/memory gains without overclaiming.
5. Contract compliance: capacity/no-realloc, transfer counters, refresh
   semantics, `recenter!` interaction.
6. Consumer impact: whether FLOWVPM (or the `032` device-system API) should
   expose the adaptive policy, and what documentation is owed.

## Deferred Candidate To Consider (user direction 2026-08-13)

**Dual-grid two-pass nearfield: shared Morton hierarchy with separately
selected primary and correction leaf sets.** Raised during the 037a close-out
discussion; consider it in this review, do not implement before then.

Idea: keep one Morton sort at the finest required depth; form the primary
(singular direct + FMM) grid from short prefixes and finer correction-only
bins from longer prefixes. Classify correction bins against the deficit
annulus `rho_c < r/sigma <= rho_t` using conservative AABB gaps and per-bin
`sigma_min`/`sigma_max`: skip bins fully inside `rho_c` or fully outside
`rho_t`, apply the deficit probe-free to bins fully inside the annulus, and
retain the body-level predicate only for boundary-intersecting bins. The
whole-ball variant (singular primary everywhere plus deficit over
`r/sigma <= rho_t`) stays excluded: it reintroduces the `031a` §6.1
singular-plus-deficit cancellation that `rho_c` exists to avoid.

Evidence at deferral: on the uniform AR=5 wake at the 037a operating point the
ceiling is too small to matter — eliminating every rejected probe (4.73B
examined vs 0.947B accepted at `n = 1e6`) bounds the saving under ~4.9 ms on
an 87.12 ms solve vs the 83.31 ms baseline (~1.3%, vs the 5% promotion gate),
and the accepted-pair deficit work itself is untouched by better traversal
(that is the `037c` mesh lever). The idea's plausible habitat is exactly what
this arc builds: heterogeneous per-cell `sigma` (where global `sigma_max`
inflates correction coverage), clustered/rolled-up wakes, and a shared
finest-Morton representation that yields both adaptive leaves and fine
correction bins without a second sort. Review question: does the measured
adaptive machinery (`038`–`041a`), plus any `037b`/`037c` verdicts, make this
lever worth staging as a row — benchmarked against the simpler probe-free
adaptive regularized U-list — or should it be closed as subsumed?

## Acceptance

Review notes recorded here, row marked Done, and clear-context approval
obtained before any downstream row starts.

## Review Notes (2026-08-20)

### Scope and coordination

The review read `../MATRIX_OPERATOR_REFACTOR.md`, `START_HERE.md`, tasks
`038`–`041a`, and their listed theory, implementation, test, script, data, and
figure artifacts. In accordance with the user's scope override, it did not
inspect or rely on `041b`–`041j`. The reviewed arc is consistent with the
background design: it changes the interaction structure while retaining the
approved coefficient/operator layer, remains opt-in, and obeyed the staged
theory → host construction → host lifecycle → CUDA lifecycle → benchmark
ordering. Stale status blurbs at the heads of `038`–`041a` were corrected to
match their existing approval sections and `START_HERE.md` checkboxes.

### Correctness and operator reuse

1. **Exact-once and balance: PASS.** The DTR proof is a genuine partition
   proof and does not depend on balance or on the particular near predicate.
   The standalone verifier covers five distributions, including an
   adversarial two-cluster field, both `q=3/12`, two `K_max` values, and
   balanced/unbalanced trees: 40 configurations with exact-once ordered-pair
   painting. The production host suite independently covers uniform,
   filament, and multiscale cases, two seeds, `q=3/12`, balanced/unbalanced,
   sigma-gated lists, and uniform-limit parity. A fresh one-thread rerun
   passed all **60,774** construction/list assertions; the standalone verifier
   also ended `ALL CHECKS PASS`. The monotone balance sweep reaches the unique
   2:1 fixed point and the host/device structural-parity record covers the
   device implementation.
2. **Operator-table reuse: PASS.** The original non-sticky sigma demotion
   defect was correctly rejected in 038's first review. Sticky lineage now
   permits V emission only on geometrically near-parent paths, and both the
   verifier and production code assert membership in the existing `025`
   `(level, source phase, offset)` family. Host and CUDA adaptive V paths feed
   the existing concatenated, precomputed-y, or dense resident plans; they add
   neither an operator table nor an adaptive-only M2L strategy. W/X are
   deliberately body-mediated M2T/S2L operations, not hidden table forks.
3. **M2T/S2L and Lamb–Helmholtz: PASS within the stated surface.** The error
   argument is conservative relative to the same-level V bound: M2T removes
   target truncation and S2L removes source truncation, while every emitted
   offset satisfies the `c>2` requirement. The standalone scalar convergence
   rerun reached at worst `1.06e-7` at `P=4` and approximately machine
   precision by `P=8`. The fresh lifecycle rerun passed 40 scalar and 12 LH
   sampled-direct accuracy assertions, 128 M2T and 144 S2L composition-oracle
   assertions across Float32/Float64 and `P=4/8`, plus regularized-vortex
   parity. The `P_chi=P_phi+1` storage/evaluation rule is exercised. The
   existing guard on LH+hessian M2T is a documented unsupported surface, not
   an accuracy hole.

### Performance and reporting verdict

4. **The adaptive machinery is a regime win, not a universal replacement.**
   Independent selection from the widened same-job CSVs reproduces the
   `n=1e6` adaptive/uniform step ratios: H200 cube `0.962x`, wake `0.852x`,
   multiscale100 `1.861x`; host cube `0.957x`, wake `0.903x`, multiscale100
   `2.991x`. Thus cube is parity, deep uniform wins the uniformly sparse wake,
   and adaptive decisively wins clustered density. At contrast 100/1000 it is
   `1.67x/2.11x` faster on H200 while bounding the fat-cell tail; at severe
   sigma spread it remains valid when the uniform global gate collapses.
   Device-memory evidence strengthens the policy distinction: the deep
   uniform winners consume roughly 54–60 GB at `ell=7`, versus 3.1–8.3 GB for
   adaptive configurations.
5. **Default decision: keep uniform as the default; keep adaptive explicit.**
   Recommend adaptive for clustered/multiscale fields, severe sigma
   heterogeneity, or memory-constrained deep sparse fields. Do not select it
   by default for uniform/small-`n` cases or a uniformly sparse wake. `K_max`
   is materially non-monotone, so an automatic default would first need a
   cheap occupancy/sigma classifier and a calibrated per-`n` K selector.
6. **Do not lift the uniform `ell<=8` cap now.** The adaptive path already
   removes the practical depth limit (measured at `ell_max=10`, implementation
   cap 21). Uniform `ell=7` consumes 54–60 GB and the observed near-8x capacity
   growth makes `ell=8` impractical on one H200; replacing dense `node_at`
   would perturb the established uniform record path without a measured
   consumer need. Revisit only for a concrete uniform-grid consumer that both
   requires and can afford deeper levels.
7. **Reporting quality: PASS.** All eight raw 041a CSV checksums validate.
   `figures_041a_prepare.jl` regenerated the headline ratios and all figure
   tables. Figures 14–20 compiled cleanly with `pdflatex` and were visually
   inspected; their curves and captions expose the contrast-1 anomaly,
   wake/small-n losses, endpoint/memory constraints, sigma-gate failures, and
   matched `1e-3` accuracy rule rather than presenting only favorable cases.
   The widened GPU CSV's worst successful error is `7.937e-4`.

### Lifecycle contract and consumers

8. **Capacity/lifecycle contract: PASS.** Capacities are fixed at cache
   construction and overflow throws; warmed typed host refresh is zero
   allocation and device lifecycle allocation was recorded as zero on H200.
   Route/operator uploads and expansion host copies remain construction-only;
   ordinary device steps keep their counters flat. Occupancy epochs skip list
   rebuilds on stable keys (measured 9.0–9.4 ms warm refresh versus
   50.9–92.9 ms forced rebuild). `recenter!` reconstructs geometry while
   explicitly forwarding `cache.adaptive`, then swaps the fresh state only
   after successful construction; the host test preserves the policy and
   exact-once coverage. Its construction-equivalent cost, transient memory,
   and counter restart remain correctly documented.
9. **Consumer/API decision.** Do not add a FLOWVPM default or high-level
   automatic switch yet. The existing `RadixFMMCache(...;
   adaptive=AdaptiveTreePolicy(...))` is the right experimental `032` API:
   explicit, cache-construction-time, and shared by host/device. Documentation
   is owed before broader exposure: add the missing `adaptive` keyword to the
   cache docstring/device-interface guide; state supported body/kernel/output
   combinations, cubic-domain and device split-veto restrictions, refresh and
   `recenter!` costs, memory tradeoffs, and the regime/K-selection table above.
   FLOWVPM should expose it only after it can choose a policy deliberately and
   preserve its requested U/J/LH output surface.

### Follow-on disposition and GPU-speed opportunities

- **Top GPU lever:** tune/fuse the adaptive S2L(X) stage. On the wake it costs
  22.2 ms (plus 7.2 ms M2T) and explains most of the adaptive/deep-uniform
  gap. This is now staged as `042a`: source reuse, target-expansion ownership,
  launch grouping, and a `P=4` specialization against the unchanged
  accuracy/oracle suite and a complete-step promotion gate.
- **Policy/docs lever:** build a cheap selection report from construction-time
  leaf-population and sigma statistics, then calibrate `K_max` by `n` and
  regime. Keep selection advisory until it predicts the widened winners; land
  the documentation above independently because the public keyword already
  exists.
- **Host-only lower priority:** eliminate the duplicate adaptive refresh/sort
  work measured at 2.1–2.9 s versus 0.44–1.0 s for uniform refresh at `n=1e6`.
  This is material for host users but secondary to the GPU goal.
- **Deferred dual-grid two-pass candidate: CLOSE as subsumed/not funded.** The
  registered uniform-wake ceiling was only ~1.3% versus a 5% gate, accepted
  deficit work is untouched by finer traversal, and the shipped per-cell
  adaptive gate already supplies the proposed locality without a second leaf
  family. The arc adds no measured evidence that reverses that result. Reopen
  only if a future regularized adaptive profile shows rejected correction
  probes themselves above the promotion threshold.

**Milestone verdict: DONE, pending independent clear-context approval.** No
correctness or gating blocker was found. Downstream work remains blocked until
a different agent approves this review, per `START_HERE.md`.
