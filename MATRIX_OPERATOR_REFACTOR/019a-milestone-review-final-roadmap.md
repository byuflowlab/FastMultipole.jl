# 019a Final Roadmap Milestone Review

## Objective

Review Implementation tasks `017` through `028` and confirm the completed
Matrix Operator Refactor still matches the background roadmap. Task `029` was
deferred by user direction on `2026-08-03` and is not a dependency of this
review; if `029` is resumed and completed later, its results are recorded as
an addendum review note here rather than reopening this review.

## Dependencies

- `008b-implementation-replan.md`
- `008c-implementation-performance-baseline.md`
- `017-impl-flat-coefficient-buffers.md`
- `018-impl-real-solid-harmonic-basis.md`
- `019-impl-operator-performance-tuning.md`
- `019b-exploratory-smallp-fallback-and-channel-layout.md`
- `022-impl-gpu-device-resident-m2l.md`
- `023-impl-production-integration.md`
- `024-impl-operator-ab-benchmark.md`
- `024a-impl-benchmark-visualization.md`
- `024b-impl-cpu-gpu-scaling-benchmark.md`
- `025-theory-hierarchical-rigid-m2l-stencil.md`
- `026-impl-hierarchical-m2l-host.md`
- `027-impl-hierarchical-m2l-cuda.md`
- `028-performance-feasibility-1m-in-10ms.md`
- ~~`029-performance-high-score-1m-in-1ms.md`~~ (deferred `2026-08-03`, user
  direction; may be resumed later)

## Required Reading

- `START_HERE.md`
- `../MATRIX_OPERATOR_REFACTOR.md`
- Completed task files listed above
- Artifacts and production files listed by the completed task files

## Artifacts or Production Surface

Review the production files, tests, benchmarks, generated artifacts, and final
notes listed by tasks `017` through `029`.

## Deliverables

- Final roadmap-alignment notes recorded in this file
- Any required coordination-document fixes identified before the refactor is
  considered complete
- **Final go/no-go (feasibility scoped earlier in `013a`): porting the old
  per-interaction error machinery onto the new expansion operators.** Using the
  `013a` feasibility finding and the `019` performance-tuning evidence, decide
  whether to port the dynamic-`P` / `get_P` / `predict_error` machinery onto the new
  operators or leave the two paths independent (old ops + old error machinery; new
  ops + constant-`P` interaction-list stencil). Record the decision and rationale
  here.
- **Review `008b-implementation-replan.md` (including its re-plan addenda) and
  confirm all recorded decisions and feedback have been incorporated** into the
  completed refactor and coordination documents. Note any gaps and the required
  fixes here.
- **Confirm the small-`P` / tiny-batch fallback and channel-layout decisions from
  `019b`.** These were resolved in `019b` (exploratory benchmark plus user
  discussion). Confirm the chosen fallback policy and padded-vs-ragged `chi` layout
  are implemented and consistent with the coordination documents; note any gaps.
- **Confirm the resident M2L strategy recommendations from `024`.** Record the
  final CPU/GPU and workload-regime recommendations among whole-slab concat,
  per-degree factored, precomputed-y, and full dense-translation execution, and
  note construction/memory constraints, crossover regimes, and integration
  caveats. The reconstructed per-column `Ts(theta)` path is an oracle, not a
  resident candidate.
- **Batched-GEMM speedup verdict (from `016b` watch item 1, deferred `2026-06-24`).**
  `015` observed no batched-GEMM speedup on the macOS host; this was deferred as a
  likely Apple-M2/OpenBLAS artifact, to be re-tested on a non-macOS / different-BLAS
  host in `024`. Record the cross-machine result here as a go/no-go: did the batched
  speedup materialize, and does it change the operator recommendation or the GPU
  (`022`) outlook?
- **Remaining lifecycle cost levers (carried in from the `023` clear-context
  review, `2026-07-15`).** With the `023` per-step update/finalize overhead
  reduced, the resident lifecycle dominates the recurring GPU step (~83% at
  n=1e5/P=4); the identified levers are the `019`-deferred L2B kernel and a
  grouped-GEMM M2L. Using the `024` stage breakdowns, record whether these are
  worth a follow-on task or are formally deferred.
- **Minor observations from the `023` approval (`2026-07-15`, non-blocking).**
  (a) The device per-step update performs a few small blocking scalar downloads
  that no transfer counter tracks (the out-of-box flag read and the route/direct
  prefix-total reads); `metadata_downloads` deliberately counts only the 3
  perm/system/index mirrors. If a future row tightens the per-step sync budget,
  start from these untracked syncs. (b) `_cuda_radix_keys_checked_kernel!`
  writes `oob_flag[1] = Int32(1)` from every out-of-box thread without atomics —
  safe because all writers store the same value; do not extend it to
  multi-value writes without adding atomics. Confirm both remain acceptable or
  note follow-up.
- **End-user scaling evidence (`024b`, added `2026-07-25`).** Use fig09 to
  record how fixed-MAC, manually leaf-searched legacy CPU 64-thread and resident
  H200 speedup over the corresponding legacy CPU single-thread baseline change
  from `n=1e3` through `1e6` at literature `P=4`
  (`expansion_order=3`). The GPU stencil must use the independently reviewed
  ell-scaled equal-cell compatibility rule, and fig09 must show the shared
  sampled-direct relative gradient RMS errors as well as timing. Treat Float64
  as the primary fair comparison and Float32 as an additional throughput
  result; include any dense-to-precomputed-y OOM fallback, Float64 error-order
  failure, or non-monotonic regime in the final recommendation.
- **Hierarchical and feasibility evidence (`025`–`028`).** Record the final
  hierarchical-vs-flat and radius/schedule verdict and task 028's independently
  reproduced 9.591 ms FP16 result. Confirm that every accepted score uses the
  unchanged 1M-body/P=4 accuracy and complete recurring-step boundary, and
  distinguish measured conclusions from modeled opportunities. Task `029`
  (high-score campaign) is deferred (user direction `2026-08-03`); record the
  deferral, and note that a later `029` completion adds an addendum review
  note here.

## Verification

Confirm completed work matches the background design, hard phase gate, and task
ordering. If `START_HERE.md`, a task file, and `../MATRIX_OPERATOR_REFACTOR.md`
disagree, stop and require a coordination-document fix.

## Review Notes (2026-08-03)

Review performed at commit `6f15eb1` (the approved `028` final state plus the
`029` scaffold, committed at review start so the review references a fixed
tree). Protocol: the reviewing agent read `../MATRIX_OPERATOR_REFACTOR.md` and
`START_HERE.md` in full and inspected `022`/`023` and the production
spot-check sites inline; the `017`–`019b`, `024`–`024b`, and `025`–`028` task
files and their data artifacts were read through three delegated readers whose
structured reports were verified against the coordination documents and, where
flagged, against the theory files and `src/` directly. Task `029` is deferred
(user direction `2026-08-03`); see the deferral note in the Objective.

### 1. Roadmap alignment, hard phase gate, task ordering — CONFIRMED

All Theory rows (including the `008d`–`008h` addendum) were Done+Approved
before any Implementation row started; `008b`/`008c` preceded `009`; every
Implementation row `009`–`028` is Done+Approved in dependency order; the
scoped `025` exception to the "no tree/radix migration" non-goal is recorded
in the background document and was honored (`025` touched only `theory/`,
`scripts/`, `data/`). The three coordination documents agree after the `029`
deferral amendments made at the start of this review (see §11). One protocol
gap found and cured during this review: the `2026-07-31` improvement review of
`026` had modified production code and required a fresh clear-context
re-approval that was never performed; it was performed as a sub-step of this
review (user direction `2026-08-03`) and is recorded in the `026` file —
verdict **re-approved** (src spot-checks all present; host suites
681/681, 61076/61076, 89/89 green on the current tree).

### 2. Dynamic-`P` porting go/no-go — **NO-GO (leave the two paths independent)**

Decision: do **not** port the `get_P` / `predict_error` dynamic-`P` machinery
onto the new operators. The legacy octree keeps dynamic-`P` with the old
kernels (and remains the CPU production default); the radix/resident path
keeps constant-`P` error control in the interaction-list stencil. Rationale:

- **Structural (013a):** `get_P` is per-pair and data-dependent; the entire
  batch win requires one operator size per offset class. Per-pair `P` inside a
  batch dissolves it back into per-pair GEMMs, forfeiting the measured batch
  tier. 013a's verdict was "feasible-with-constraints" only in the sense that
  the *legacy* path could adopt per-pair operators (unbatched tier) — a mode
  with no demonstrated benefit now (see next bullet).
- **Performance (019/019b/024/028):** every gain in the tuned resident path
  (whole-slab dense GEMMs, fused kernels, per-class geometry, FP16 WMMA M2L)
  assumes a uniform dof; per-column operator forms lose to the legacy
  recurrence below `P ≈ 8–12` on CPU (019b), so per-pair-`P` operators on
  either path would land in the losing regime. And M2L is no longer where the
  time is: ≤8% of the recurring H200 step in the worst 024 regime, 2.5 ms of
  the 9.6 ms 028 verdict step.
- **Error control is already solved per-path:** the `025` stencil family with
  the `028` level-radius schedule gives measured, gated accuracy at constant
  `P`; the legacy path retains full per-interaction dynamic-`P` unchanged.

If finer accuracy/cost control on the radix path is ever wanted, the
batching-compatible granularity is **per-level or per-offset-class constant
`P`** (013a recommendation; the stencil's accepted sets are monotone in `P`,
and 019's per-class geometry tables are a stepping stone) — and the `028`
Stage 7 level-radius schedule (`sched6-5-5-5`) already realized the same idea
on the radius axis with an 18.9% measured win, reducing the pressure to vary
`P` at all. Recorded as a possibility, not scheduled work.

### 3. `008b` decisions incorporated — CONFIRMED (with dispositions)

All main re-plan decisions (D1–D6) and addendum decisions (A1–A4) are
demonstrably incorporated: the two-error-regime coexistence (A1) is exactly
what shipped behind `RadixFMMCache` dispatch; stencil option (c) (A2) became
`ConstantPAnalyticStencil` and then the `025` rigid family; `008d`/`008e`/
`008f` exist, are Approved, and their artifacts are load-bearing. Three items
the delegated reader could not confirm were verified directly by this review:
the `008d` Lamb-Helmholtz stencil extension is a resolved section in an
Approved row (its addendum TODO is closed); `008f` is Approved by
clear-context review; and the `lda`-padded-slab / planar-re/im layout
variants are not unaddressed — `theory/coefficient-buffer-layout.md` records
them as options to adopt only if profiling favors them, `019b` profiling did
not (padding costs +10–28%), and the planar question is explicitly scoped to
the (deferred) native real basis. Disposition: **formally deferred by the
theory document's own contract; no gap.**

### 4. `019b` fallback and channel-layout decisions — CONFIRMED IMPLEMENTED

Always-dense (no small-`P`/tiny-batch fallback dispatch) and ragged `chi`
layout, per the user decision of `2026-07-14`. Verified in
`src/containers.jl`: the decision markers at lines 857–869 and 901–904, the
ragged two-matrix backing with `Val(false)` χ pruning at 906–916, the
`phi_slab`/`chi_slab`/`phi_physical_view` swap surface at 921–928, and **zero**
order-based dispatch branches anywhere in `src/` (grep clean). Thresholds are
recorded in the task file, not encoded — as decided.

### 5. Resident M2L strategy recommendations (`024`) — RECORDED

From the EPYC-7763 (BLAS 1/64) + H200 campaign (126 cases × 4 strategies, 483
eligible rows): **dense wins the recurring step in essentially every `P=4`
regime on both platforms**; precomputed-y takes over at high `P` with LH on
(CPU `P=12` Float32 always — dense unsupported; CPU `P=12` LH-on clustered —
dense over the 12 GiB memory gate; H200 `P≥8` LH-on at `N≥2000`, all `P=12`).
Concat and factored won zero CPU cases; factored won exactly one H200 step
case (`P=12`/F64/LH-on/`N=150`). Construction amortization is the main GPU
caveat: dense-vs-precomputed-y break-even is 542–18,208 steps at `P=4` and up
to ~734k at `P=8`, so **precomputed-y is the recommended general GPU default
unless the cache is long-lived**, with dense the steady-state winner (and the
`028` production defaults, which are long-lived by construction, correctly
ship dense). Memory: dense payload up to ~7.4 GiB (`P=8` LH-on) and
infeasible at clustered `P=12` LH-on (30.1 GiB > 12 GiB gate on both
platforms); fallback is precomputed-y. The reconstructed per-column
`Ts(theta)` path is confirmed oracle-only throughout `024` (the `024b` `ell`
sweep that timed it served a different purpose and does not contradict this).
Downstream readers must use the approval-note corrections (occupancy min
17.05; min-of-maxima 87; construction winners 27 factored / 13 concat / 2
precomputed-y; the omitted feasible `P=12` LH-on uniform F64 dense rows at
3,305.4 MiB) and the recorded CSV caveats (occupancy/memory columns are not
comparable across strategies).

### 6. Batched-GEMM speedup verdict (`016b` watch item 1) — **DID NOT MATERIALIZE; watch item closed**

The `015` macOS observation was *not* an Apple-M2/OpenBLAS artifact. On the
EPYC host with `libblastrampoline`, the mean BLAS-64/BLAS-1 M2L ratio across
161 matched pairs is ~0.95× (64 threads slightly *slower*); only the concat
path shows genuine GEMM-like scaling (1.25× mean, up to 1.74×) and it wins
nothing. Go/no-go consequence: **no change** to the operator recommendation
(the `024` rule already selects per BLAS regime, and the one boundary case
where BLAS-1/64 disagree is recorded) and **no change** to the GPU outlook —
the batching thesis is realized on the device (H200 dense/precomputed-y
resident path, 400–800× over host recurrence at n=1e4), not on host BLAS.
`008c`'s pin-BLAS-to-1 operator rule stands for the CPU path.

### 7. Remaining lifecycle cost levers (from the `023` review) — **BOTH CLOSED BY MEASUREMENT**

The `024` stage breakdowns showed L2B (largely fused nearfield) at 48–94% of
the recurring H200 step and M2L under 8% everywhere — so the grouped-GEMM M2L
lever was already immaterial end-to-end, and the L2B/nearfield lever was the
top-ranked one. `028` then implemented both: warp-per-pair nearfield +
warp-per-cell L2B (−26.5 ms), operator-tiled and grid-strided dense M2L, and
FP16/FP32-accum WMMA M2L (5.47 → 2.55 ms). Final step: L2B+nearfield 4.25 ms
and M2L 2.54 ms of 9.59 ms. **No follow-on task warranted**; further latency
work belongs to the deferred `029` campaign.

### 8. `023` minor observations — RE-CONFIRMED ACCEPTABLE

Both checked in the current tree: (a) the per-step blocking scalar downloads
(out-of-box flag, route/direct prefix totals) remain untracked by any
counter; `metadata_downloads` still counts exactly the 3 mirrors
(`src/translate_batched_cuda.jl:4200`). Still acceptable — `028` met the
verdict target with these in place; they remain the starting point if a
resumed `029` tightens the per-step sync budget. (b) The non-atomic
`oob_flag[1] = Int32(1)` write (`src/translate_batched_cuda.jl:146`) is still
single-value and safe; the do-not-extend-without-atomics caution stands.

### 9. End-user scaling evidence (`024b` fig09) — RECORDED

At literature `P=4`, MAC 0.5, leaf-searched legacy baselines: CPU-64
saturates at **13–17×** over CPU-1 (flat above n=1e4; the sub-n=5000
flatness is the `MIN_BODIES=10000` single-thread forcing in `src/fmm.jl`, not
a search artifact). The resident H200 lifecycle rises from 2.6× (n=1e3) to a
peak of **157× F64 / 188× F32 at n=316k**, dipping to **94× F64 / 125× F32 at
n=1e6** — a grid-capacity artifact (`ell=4` too coarse; `ell=6/7` route
storage up to 223.5 GiB unconstructible), *since addressed*: the `025`–`028`
hierarchical path constructs `ell=6/7` and reaches 9.6 ms at n=1e6, versus
fig09's 321/425 ms — the fig09 GPU column is a lower bound superseded at
n=1e6 by the hierarchical results. Accuracy: **no Float64 error-order
failure** — GPU F64 gradient RMS error is *smaller* than both CPU paths at
every n (ratios 2.3–5.3, gate ≤10; n=1e3 exempt because the CPU searches
selected exact-direct evaluation, error 0); Float32 tracks Float64 to three
digits (truncation-limited at `P=4`). Dense won all 14 selected GPU points;
precomputed-y served only as the prescribed dense-OOM fallback and was never
selected. **Provenance note:** `024b`'s approval was user-directed on
`2026-07-28` after the reviewing agent edited the row (recorded in the task
file), not a separate third-agent pass; the preceding review did
independently re-derive the compatibility epsilon, re-run the `ell=2:7`
verifier (zero mismatches), and confirm fig09 byte-identical after rebuild,
so the evidence base is sound. Accepted as-is under the user-instructions-
first rule; no further action.

### 10. Hierarchical and feasibility evidence (`025`–`028`); `029` deferred

- **`025` theory:** exact-once coverage proof (push/pull equivalence,
  downward monotonicity, no level-1/2 special case), level-scaling law
  `K(s·r0) = s⁻¹Λ(s)K(r0)Λ(s)` letting one ≤1740-matrix table serve all
  levels, `O(n^{4/3}) → O(n)`; fully verified (deterministic verifier,
  independent brute-force re-enumeration) and Approved `2026-07-28`.
- **Hierarchical-vs-flat verdict:** hierarchical wins wherever grids are deep
  or occupancy is high — host crossovers by `N≈128–512` at `ell≥3` for the
  genuine engines; on H200 flat still wins at `ell=3` but hierarchical wins
  overwhelmingly at `ell≥5` (at n=2e5/`ell=5`: 322× M2L, 24× full step, ~330×
  memory; flat dense doesn't fit). **Hierarchical
  (`HierarchicalRigidStencil`) is the production default on CUDA** (user
  direction `2026-07-30`), with flat `ConstantPAnalyticStencil` deprecated as
  default but retained as selectable oracle; host default window
  `window_classes=4`, device 4096 (one window per level, per `028`).
- **`027` regression gate:** the initial inventory flag was stale —
  Checkpoint D was **reinstated by user direction and PASSED** (job 12993753:
  pooled step geomean 0.972–0.978, no allocation/counter regressions;
  tiebreaker job 12994269 settled the one ambiguous cell, new source faster).
  Scope limit stands as recorded: the gate covers the flat common surface
  only, since the pre-`026` snapshot has no hierarchical policy; `028`'s
  hierarchical measurements cannot and need not close an old-vs-new
  comparison. **No open gap.**
- **Radius/schedule verdict (`028`, supersedes the `025`–`027` two-radius
  comparison):** classic `q=3` is fastest but inadmissible at n=1e6
  (3.995e-3 > the 1.19e-3 gate); the fastest admissible uniform shell is
  `q=6` (17.383 ms); the shipped optimum is the **level schedule
  `sched6-5-5-5`** (q=6 coarse, q=5 elsewhere), 18.9% faster than uniform
  q=6 at 1.05e-3. `q=12` remains selectable to restore the more accurate old
  operating point (3.19e-4).
- **`028` verdict — TARGET MET, independently reproduced:** 1M bodies,
  literature `P=4`, single H200, per-time-step verdict boundary (device
  refresh + eval + finalize + Euler convection; zero per-step body transfers,
  counter-asserted per case): **9.591 ms [9.434, 9.631]** FP16-in/FP32-accum
  WMMA, gradient RMS 1.0593e-3 vs the never-moved 1.19e-3 gate (= 10× the
  `P=4` F64 truncation error), job 13029878 reproducing bake-off job
  13029480; re-gated under the shipped defaults at **9.653 ms** (job
  13031482, 33,381/33,381 hierarchical assertions) after the one test-suite
  failure (13031187) was traced to two testsets assuming the old concat
  default — no production defect. Per-stage: M2L 2.54, L2B+nearfield 4.25,
  refresh 0.955 ms. Measured-vs-modeled is cleanly separated in the record;
  the Stages 5–10 continuation roadmap is a **backlog, not approved work**
  (Stage 9 never authorized), now subsumed by the deferred `029`.
- **Caveats carried forward:** (i) the including-transfers boundary was not
  re-measured at the final configuration (`h2d_ms`/`d2h_ms` NaN in the winner
  CSVs; Phase A's ~1.8 ms F32 transfer cost *models* ≈11.4 ms — above 10 ms,
  but that boundary was defined as non-deciding); (ii) hierarchical-path
  run-to-run variance is ~8%, so the margin to 10 ms is about one variance
  unit; (iii) FP16 per-column operator scaling is not scale-invariant —
  documented hazard with the `DENSE_CUDA_TENSOR_FORMAT[] = :off` escape;
  (iv) the shipped default geometry is less accurate than `q=12` (both inside
  the gate; `near_radius2=12` restores the old point).
- **`029`: deferred** by user direction `2026-08-03` (may be resumed later; a
  later completion adds an addendum note here). Single-/multi-H200 high-score
  conclusions therefore do not exist yet; the best current single-H200 score
  is `028`'s 9.591 ms above.

### 11. Coordination-document fixes and improvement notes

Fixes applied at review start (all committed): the `029` deferral recorded
consistently in `START_HERE.md` (both rows), this file, the `029` file, and
`../MATRIX_OPERATOR_REFACTOR.md`; the previously uncommitted approved `028`
surface (~22 files) plus the `029` scaffold committed as `6f15eb1`; the
outstanding `026` re-approval performed and recorded (§1). No other
disagreement among `START_HERE.md`, task files, and the background document
was found.

Improvement notes (recorded per the Milestone Review charter, step 5 — none
blocks this review; all are natural first steps for a resumed `029` or
maintenance):

1. Re-measure the including-transfers boundary at the shipped defaults (one
   cheap H200 run) so all three `028` boundaries are measured, not modeled.
2. The untracked per-step scalar downloads (§8a) are the first place to look
   if per-step sync budget ever matters at ~1 ms scale.
3. A stencil-tolerance (accuracy-vs-cost) sweep still does not exist anywhere
   in the data record (`024a` limitation); the `028` nine-shell frontier
   partially fills this at n=1e6 only.
4. The `024b` fig09 n=1e6 GPU point predates the hierarchical default; if
   these figures are ever published, add a footnote or a re-measured point so
   the 94×/125× is not read as the current capability (current: ~4,170× vs
   the 40.0 s CPU-1 baseline, from 9.6 ms — different boundary definitions
   apply and must be stated).

**Row `019a` is marked Done in `START_HERE.md`.** Per protocol, clear-context
approval must be performed by a different agent; the completing agent has not
approved its own review.

## Approval Notes

To be filled by a different agent after review notes and verification are
complete.
