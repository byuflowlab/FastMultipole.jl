# Task 028 Phase A — Can n=10⁶ run in 10 ms/step on one H200?

**Measurement + bound classification + prioritized lever list. No optimization performed.**

Jobs: pilot **12996475**, sweep **12997508** (both H200 `m13h-1-1`, 2026-07-31).
Source manifest `0e57eca91e2c725b`. Julia 1.11.7, CUDA runtime 12.8.0 (local toolkit),
driver 580.159.4, H200 140.4 GiB. Data: `cuda_m13h-1-1_2026073*.csv` (+ `.classes.csv`)
in this directory. 27 rows, **all `fit=true`** — no failures, no OOM, no failure ledger.

---

## 1. Answer

**No — not at the current level of kernel efficiency.** The best accuracy-admissible
configuration at n=10⁶, `expansion_order=3` (P=4) is

> **hier12 · dense · Float32 · ell=5 · K=1740 → 91.4 ms per verdict step**

which is **9.1× over the 10 ms target**. Float64 gives 104.0–106.3 ms (10.4–10.6×).

The gap is **not** attributable to any remaining configuration choice — `ell`, `K`,
strategy, policy and precision have all now been measured at n=10⁶ and the above is
their joint optimum. It is attributable to two device kernels, **fused L2B+nearfield
(42%)** and **leaf-level M2L (26%)**, both of which run at a few percent of the
hardware's capability. So the verdict is "no at present" rather than "no in principle":
§5 shows a credible path to ~8–12 ms, but it requires near-peak rewrites of both
dominant kernels, not tuning.

### Three boundaries (verdict config, n=10⁶, ell=5, K=1740)

| boundary | Float64 | Float32 | vs 10 ms |
|---|---|---|---|
| (a) evaluation only (`run_cuda_radix_lifecycle!`) | 99.8 ms | **86.4 ms** | 8.6× |
| (b) **verdict** (refresh + eval + finalize + device Euler) | 104.0 ms | **91.4 ms** | 9.1× |
| (c) verdict + body H2D/D2H | 107.6 ms | **93.2 ms** | 9.3× |
| (—) fully host-resident variant, for contrast | 233.9 ms | 159.4 ms | 15.9× |

### Residency answer

**Body state must stay on the GPU, but PCIe is not why.** The transfers themselves are
nearly free — `h2d` 0.90 ms + `d2h` 0.93 ms in F32 (1.77 + 1.81 in F64), so boundary (c)
is only ~2% above the verdict boundary. The expensive thing is the *host-side pipeline*:
the fully host-resident variant costs 159.4 ms vs 91.4 ms, a 68 ms penalty that is host
sort/tree/refresh work, not data movement. Residency is worth ~1.7×, and the counter
contract confirms it is actually achieved (`body_uploads=0`, `metadata_downloads=0`,
`expansion_host_copies=0`, route/operator uploads flat across recurring steps).

---

## 2. Verdict-step budget

hier12 · dense · ell=5 · K=1740 · n=10⁶ · LH off. Per-stage figures are CUDA-event
medians over `REPS=5`.

| stage | F64 ms | F32 ms | F32 share | F32/F64 |
|---|---|---|---|---|
| **L2B + nearfield (fused)** | 52.82 | **38.43** | **42.0%** | 0.73 |
| **M2L total** | 25.13 | 26.67 | 29.2% | 1.06 |
| — M2L leaf (L5) | 22.41 | 23.55 | 25.8% | 1.05 |
| — M2L L2–L4 | 2.91 | 3.26 | 3.6% | 1.12 |
| host allocation / GC (residual) | ~13.9 | ~14.1 | ~15.4% | — |
| refresh | 6.14 | 3.68 | 4.0% | 0.60 |
| grid | 3.49 | 3.30 | 3.6% | 0.95 |
| route_gen | 2.13 | 2.08 | 2.3% | 0.98 |
| B2M / M2M / L2L | 1.03 / 0.99 / 0.97 | 0.58 / 0.97 / 0.95 | 2.7% | — |
| finalize + Euler + occupancy + direct_gen + groups | ~0.8 | ~0.6 | 0.7% | — |

**The bottleneck is concentrated in two stages** (L2B+nearfield and leaf M2L = 68% in
F32), with a third meaningful block in host allocation. Everything else together is
under 14%.

*Caveat on the residual row:* it is `verdict_step_ms − Σ(stages)`, and summing
independently-measured medians is not the median of the sum — across the 27 rows this
residual ranges from −5.3 to +33 ms, i.e. it carries roughly ±5 ms of noise and should
not be read finely. The **directly measured** quantity is
`verdict_step_host_alloc_bytes = 57.7 MB/step` (≈1.76 KB per cell per step), which is
real and independent of the timing decomposition.

---

## 3. Bound classification (method stated per claim)

### 3.1 Leaf M2L — **not bandwidth-bound, not compute-bound; overhead/latency-bound**

*Method: precision A/B at fixed work.* Float32 halves every coefficient load and doubles
the FLOP ceiling. If the stage were limited by either, it would drop substantially.
Measured leaf M2L: **22.41 ms (F64) → 23.55 ms (F32)** — unchanged, in fact 5% *worse*.
Route counts are identical (31,307,680 routes both rows). This is a direct experimental
refutation of both attributions and does not depend on any FLOP model.

Supporting roofline (order-of-magnitude, assumptions stated): at 1.40 G routes/s, naive
coefficient traffic is 8.01 GB (F64) → 358 GB/s, **7.5% of the 4.8 TB/s HBM3e peak**;
modelling each route as a dense 16×16 complex apply ((P+1)²=16 coefficients, LH off)
gives ~2.9 TFLOP/s, **~8% of the 34 TFLOP/s FP64 vector peak**. Both are far from their
ceilings, consistent with the A/B result. *The exact operator shape was not re-derived
from source for this report — the FLOP figure is indicative only; the A/B result is the
load-bearing evidence.*

### 3.2 Fused L2B + nearfield — **partially compute-bound, FP64 transcendental-limited**

*Method: precision A/B plus interaction-rate arithmetic.* The nearfield does
5,189,728 direct cell-pairs × (30.5 bodies/cell)² = **4.83×10⁹ body-body interactions**
per step. Measured: **91.5 G interactions/s (F64) → 125.8 G/s (F32), a 1.37× gain.**
A pure-bandwidth stage would be ~2×; a pure-FP64-FLOP stage would be ~2×; 1.37× indicates
a mixed limit. Naive memory traffic is 5.06 GB/step → 96 GB/s = **2% of HBM3e**, which
rules out bandwidth. At ~22 flops/interaction the F64 rate is ~2.0 TFLOP/s = **5.9% of
FP64 vector peak**. The residual is most consistent with FP64 reciprocal-square-root
throughput (an SFU/emulated path in FP64, hardware-rate in FP32), which also explains
why F32 helps this stage and not M2L.

### 3.3 route_gen — **was launch-latency-bound; now solved by K**

*Method: K sweep at fixed work.* 027 established one blocking D2H sync per
`(level, window)` at 46–61 µs. At n=10⁶/ell=5, **K=256 → 24.19 ms, K=1740 → 2.13 ms**
(11.4× reduction, identical routes). Coarse-level M2L improves in the same move
(L2 0.579→0.14 ms, L3 0.511→0.21 ms). **This lever is now spent** — route_gen is 2.3%
of the step and no longer worth attention.

### 3.4 Host allocation / GC — **directly measured, not inferred**

`verdict_step_host_alloc_bytes = 57.7 MB/step` at ell=5 (32,768 cells) and
**397.9 MB/step** at ell=6 (252,483 cells) — i.e. ~1.6–1.8 KB per cell per step,
confirming 027's ~2.6 KB/cell estimate to within a factor of 1.5 and its identification
as a 028 bottleneck item. This is per-step host garbage on the recurring path.

---

## 4. Configuration findings (all newly measured at n=10⁶)

### 4.1 `ell` is exhausted — ell=5 is the optimum, both neighbours far worse

| ell | cells | bodies/leaf | leaf M2L | L2B+near | verdict |
|---|---|---|---|---|---|
| 4 | 4,096 | 244 | 2.6 ms | 331.6 ms | **349.5 ms** |
| **5** | 32,768 | 30.5 | 22.4 ms | 52.8 ms | **106.3 ms** |
| 6 | 252,483 | 4.0 | 187.3 ms | 17.4 ms | **546.7 ms** |

Each `ell` step trades the two dominant stages by ~8× in opposite directions, so with
nearfield:leaf-M2L = 2.3 at ell=5 there is no better integer choice. Route counts grow
8.7× from L5 (31.3M) to L6 (271.3M).

**Correction to the pilot's extrapolation.** From pilot data I estimated ell=6 ≈ 220 ms
and ell=4 ≈ 430 ms; measured are **546.7** and **349.5**. The extrapolation was wrong by
2.5× on the ell=6 side and I am discarding it in favour of the measurements. The ranking
it predicted happened to survive, but it should not have been trusted — this is why the
bracket was measured rather than assumed.

*One confound, stated:* Tier A ran ell=4/6 at **K=256**, where route_gen is not yet
cured. At ell=6 route_gen was 195.2 ms of the 546.7 ms. Correcting it to the K=1740
regime still leaves ell=6 at **≈354 ms** (its leaf M2L of 187.3 ms and 104 ms of host
allocation are K-independent), so the conclusion is unaffected — but the ell=6 number as
tabulated is not K-optimal.

### 4.2 hier3 is fast but **inadmissible** — hier12 is confirmed as the verdict config

hier3 is the fastest thing measured (52.7 ms F32) but its gradient error at n=10⁶ is
**3.995×10⁻³**, which is **3.4× over the accuracy gate** (10× of 1.19×10⁻⁴ = 1.19×10⁻³).
027 saw ~6×10⁻⁴ at n=2×10⁴; the error grows with n. This upgrades hier12 from
"presumptive" to **confirmed** verdict configuration, and closes hier3 as an option.

### 4.3 Float32 is admissible on the hierarchical path (first such data)

Gradient error **3.186×10⁻⁴ (F32) vs 3.185×10⁻⁴ (F64)** — +0.03%, far inside the gate,
matching the flat-path precedent (1.189e-4 vs 1.187e-4). F32 buys **1.14×** on the
verdict step (104.0 → 91.4 ms), concentrated entirely in the nearfield. No Float32
construction anomaly was observed at n=10⁶ (all F32 rows `fit=true`); the 023d anomaly
remains construction-only and did not recur here.

### 4.4 `precomputed_y` loses to `dense` everywhere — lever demoted

At n=10⁶/ell=5: dense 106.3 ms vs precomputed_y 172.3 ms (F64); 91.4 vs 146.8 (F32).
The penalty is entirely in M2L (25.1 → 73.0 ms). The 024 "dense wins at P=4" rule holds
under hierarchical thin-class shape too.

### 4.5 Per-level M2L strategy mix is worth ≤3 ms — lever demoted

The class histograms confirm 027's structural prediction exactly: coarse levels are thin
(L2 mean 4.6 routes/class, L3 mean 92) and the leaf is fat (mean 17,993, p90 26,784).
But at K=1740 the coarse levels cost **0.14 + 0.21 + 2.51 = 2.86 ms total** of a 106 ms
step. Even a perfect coarse-level strategy substitution recovers under 3 ms. `dense` is
already correct at the leaf, which is where all the time is.

### 4.6 Lamb–Helmholtz costs 1.41× (first hierarchical LH data)

149.5 ms vs 106.3 ms at n=10⁶/ell=5/K=1740/F64. The cost is entirely in M2L
(25.1 → 57.1 ms, 2.3×) from the dual φ+χ channel; L2B+nearfield is unchanged at 53.6 ms.

### 4.7 Stale-tree refresh is a weak lever

Skipping refresh saves **3.9 ms** (104.0 → 100.1 ms) at K=1740 — refresh is only 6.1 ms.
Accuracy after 5 refresh-skipped steps: **3.18×10⁻⁴**, unchanged. *This is a weak test:*
`dt=1e-5` moves bodies a negligible distance, so the accuracy result should not be read
as licence for stale trees at realistic time steps.

### 4.8 Scaling on the hierarchical path

At ell=5: n=2×10⁵ → 51.9 ms, n=3.16×10⁵ → 54.7 ms, n=10⁶ → 106.3 ms. That is **2.05× for
5× bodies** — strongly sublinear, because at n=2×10⁵/ell=5 the tree is over-refined
(6 bodies/leaf) and overhead-dominated. The optimal `ell` tracks n (4 at 2×10⁵ giving
23.4 ms, 5 at 10⁶), consistent with ~30–50 bodies/leaf being the sweet spot.

---

## 5. How far is 10 ms, really?

Starting from the F32 verdict step of 91.4 ms, suppose every non-kernel overhead were
eliminated outright — host allocation (~14 ms), grid (3.3), refresh (3.7), route_gen
(2.1), B2M/M2M/L2L (2.5), finalize/Euler (0.6). That is ~26 ms, leaving **65 ms of
L2B+nearfield and M2L**. So overhead elimination alone cannot reach the target; the two
dominant kernels must improve ~7× *combined*.

Whether that is possible is a roofline question, and the answer is a qualified yes:

- **Nearfield**: 4.83×10⁹ interactions at ~22 flops is ~106 GFLOP. At the H200's 67
  TFLOP/s FP32 vector peak that is a **~1.6 ms floor**; the current 38.4 ms is ~4% of
  that rate. Even a conservative 8× (well short of peak) gives ~5 ms.
- **Leaf M2L**: 31.3M routes moving ~4 GB (F32) would be **~0.8 ms** at HBM3e bandwidth;
  the current 23.6 ms is 28× off, and §3.1 shows it is limited by neither bandwidth nor
  flops, so the headroom is real even if the mechanism is not yet identified.

A plausible optimized budget is therefore **~5 ms nearfield + ~2–3 ms M2L + ~2 ms
residual ≈ 8–12 ms**, i.e. the target is *reachable in principle* but sits at the far end
of what near-peak kernel engineering on both dominant stages would deliver. **I would not
represent 10 ms as achievable without first executing lever 1 and seeing what fraction of
its theoretical gain is real.** Note also that the nearfield floor is a hard function of
the (n, ell) work choice: 4.83×10⁹ interactions is irreducible at ell=5, and §4.1 shows
no better `ell` exists.

---

## 6. Prioritized lever list (Phase B candidates)

| # | Lever | Est. gain | Confidence | Risk | Basis |
|---|---|---|---|---|---|
| **1** | **Rewrite fused L2B+nearfield kernel** (shared-memory tiling of source bodies, FP32 rsqrt path, occupancy tuning) | **20–33 ms** (38.4 → ~5–18) | Med-High | Med | Largest stage (42%); at ~4% of FP32 peak rate; §3.2 |
| **2** | **Leaf-level M2L overhead hunt** (profile for occupancy / indexing / launch structure — *not* bandwidth or flops) | **10–20 ms** (23.6 → ~4–14) | Medium | Med-High | 2nd stage (26%); F32/F64 A/B proves it is overhead-bound; mechanism unidentified — needs a profiler pass first |
| **3** | **Eliminate per-step host allocation** (preallocated block scan replacing `accumulate!`, pooled per-cell scratch) | **~10–14 ms** | High | Low | 57.7 MB/step directly measured; 027 named the `accumulate!` scratch |
| **4** | **Float32 verdict path** (adopt as default for the verdict config) | **12.6 ms** (104.0 → 91.4) | **Realized** | Low | §4.3 — already measured and accuracy-admissible |
| **5** | **K=1740 as the default at n=10⁶** | **11–22 ms** | **Realized** | Low | §3.3 — already measured; guard the K-default trap |
| **6** | Fold `grid` + `refresh` into fewer device passes | ~3–5 ms | Medium | Med | 7.0 ms combined, partly host-side |
| **7** | Stale-tree refresh policy | ~3.9 ms | Low | **High** | §4.7 — accuracy test was too weak (dt=1e-5) to trust |
| **8** | Per-level M2L strategy mix | ≤3 ms | High (that it's small) | Low | §4.5 — **recommend dropping**; task file named it, data demotes it |
| **9** | `ell` / policy / strategy retuning | 0 | High | — | §4.1/4.2/4.4 — **exhausted**, optimum already selected |

**Recommended Phase B order: 3 → 1 → 2.** Lever 3 is the cheapest and most certain (pure
host-side work, no numerics risk). Lever 1 has the largest gain and a well-understood
mechanism. Lever 2 has comparable upside but should begin with a profiler pass, since the
A/B result tells us what it *isn't* without telling us what it *is*. Levers 4 and 5 are
already realized in the measured configuration and need only to be made default. Levers
8 and 9 should be closed out.

---

## 6b. Phase B progress — de-risking (job 12998146/12998189) + lever 3 (job 12998517)

### De-risking resolved all three Phase A inferences; two were wrong

1. **The fused stage is 98% nearfield, not 85%.** `_launch_cuda_resident_l2b!`
   gates the nearfield kernel on `state.counts.n_direct`, so setting it to 0 gives an
   exact split. At n=1e6/ell=5/F64: fused 52.84 = **L2B 0.96 + nearfield 51.88 (98.2%)**;
   ell=4 98.5%, ell=6 94.0%, F32/ell=5 98.2%. **The §1 ell-sweep fit (L2B ≈ 9.3 ms) was
   wrong** — nearfield is not cleanly proportional to interaction count, because at ell=6
   there are 8x more cell-pairs of ~4 bodies each and per-pair overhead dominates; the fit
   absorbed that into its constant. **Lever 1 is a pure nearfield lever; L2B needs nothing.**
2. **Host allocation placed exactly.** refresh 8.23 MB + lifecycle 49.42 MB = 57.65 MB,
   matching the 57.7 MB of §3.4. Two sites are 73%: `_assert_cuda_scratch_value!`
   (1,346,598 allocations/step, 33.81 MB) and `_canonical_cuda_source_buffer`'s
   `collect(1:n)` (3 allocations, 8.00 MB). The M2L window loop allocates 0.05 MB and
   finalize 0.01 MB — the §3.4 suspicion that `accumulate!` scratch was the driver was
   **wrong**.
3. **Leaf M2L's mechanism identified.** `CUDA.@profile` shows
   `_cuda_hier_dense_fused_kernel_` launching **Threads=32, Blocks=31,307,680** —
   one single-warp block per route — for 20.28 ms. It is **block-dispatch-bound**, which
   is why the §3.1 precision A/B showed no movement. Fix: grid-stride loop over routes.

### Lever 3 delivered 21-22 ms (predicted 10-14)

Two `src/` changes in `translate_batched_cuda.jl`, both preserving the 023 invariant
contract:
- `_cuda_scratch_value_ok` — an allocation-free predicate proving device residency
  without building a `"$path[$i]"` String per visited element. The original walker still
  runs whenever the cheap proof fails, so failure messages are unchanged. Conservative by
  construction: it may fail to prove a valid value, never passes an invalid one. Large
  device-typed vectors are accepted by `eltype`; struct fields are walked by a
  `@generated` unrolled function (a `fieldnames`/`getfield` loop is type-unstable and
  would box per field, reintroducing the allocation).
- `collect(1:get_n_bodies(system))` → `Base.OneTo(...)` on the device-resident path,
  matching the documented `sort_index` default at `compatibility.jl:516`.

| metric | before | after | delta |
|---|---|---|---|
| host allocation / step | 57.7 MB | **0.80 MB** | **-99%** |
| verdict step, F32 | 91.40 ms | **69.64 ms** | -21.8 (-24%) |
| verdict step, F64 | 104.00 ms | **82.90 ms** | -21.1 (-20%) |
| `l2b_ms` / `m2l_ms` | 38.43 / 26.67 | 38.39 / 26.64 | unchanged |
| gradient rel RMS | 3.186e-4 | 3.186e-4 | unchanged |

Device stages unchanged to within noise — the whole gain is host overhead, where the
profile placed it. Counter contract holds (0/0/0), accuracy gate passes, 215-test
lifecycle gate and convection gate both exit 0. The 10-14 ms prediction undercounted
because the assertion cost more than its allocation (1.35M calls plus induced GC), part
of which had been absorbed into other stages' wall time and the +-5 ms residual noise.

### Standing after lever 3

**F32 verdict step 69.64 ms = 7.0x over target** (was 9.1x). The budget is now almost
entirely the two kernels: nearfield 38.4 ms (55%) + M2L 26.6 ms (38%) = **93%**, with
~4.6 ms of everything else. This is the state §5 anticipated: overhead is spent, and the
remaining gap requires ~6.5x from the two dominant kernels.

## 7. Methodology, anchoring, and threats to validity

- **Harness anchored at n=10⁶.** The flat dense ell=4 row reproduces the 024b record's
  gradient error to five digits (1.187295e-4 vs 1.1873e-4) in F64 and exactly in F32
  (1.189e-4), and reproduces its own timing across two independent jobs to 0.05%
  (363.50 vs 363.33 ms). `host_step_ms` 483 vs the 024b 425 ms is median-vs-`step_min`,
  not a discrepancy.
- **Reference integrity.** `ref_cross_check_grad_rel = 2.76×10⁻¹⁴` — the on-device direct
  kernel agrees with the checksummed 024b CSV to machine precision, so the accuracy gate
  is not measuring harness error.
- **Convection verified.** After 3–5 device Euler steps the re-evaluated field matches a
  *fresh* on-device direct reference at the moved positions (3.1842e-4 vs step-0
  3.1852e-4). Host and device paths agree to 13 digits.
- **The 027 tie-in row does not compare like-for-like — and this is a harness property,
  not a speedup.** At n=2×10⁵/ell=4/K=1740/hier12/dense, 028 measures 23.4 ms vs 027's
  48.56 ms. The cause is a **different tree geometry**: the 028 harness pins
  `bounds=(-0.01, 1.02)`, filling all 4,096 cells at ell=4, whereas 027 fitted the tree
  to the body bounding box and got 2,744 cells. Consequently 028 has 48.8 bodies/leaf vs
  027's 72.9, and does ~30% less nearfield body-body work despite 1.69× more routes. **No
  cross-run speedup should be claimed from these rows.** Fixed bounds do make 028's own
  n-scaling clean (identical grid at every n), so §4.8 is internally valid.
- **Run-to-run variance.** The identical hier12/dense/F64/ell=5/K=256 configuration
  measured 127.2 ms (pilot) and 117.1 ms (sweep) — **8%**. The flat path reproduced to
  0.05%, so the variance is specific to the hierarchical path (host syncs). **Any claimed
  lever gain below ~10 ms should be treated as within noise.** This is why levers 6–8 are
  ranked low regardless of their point estimates.
- **Sweep pruning.** 25 tiered cases replaced the planned 96-case cross product; `ell`
  was bracketed with one confirming case per side rather than swept. All drops and their
  justifications are logged in `plans/20260731_028/04-phase-sweep.md`. The ell=6 K=256
  confound is stated in §4.1.
- **Not measured / open:** the exact M2L operator FLOP shape (§3.1 roofline is indicative
  only); ell=6 at K=1740; hier3 at n=2×10⁵ on this harness; realistic-`dt` stale-tree
  accuracy; multi-GPU. Phase A made **no `src/` changes** — all work is
  benchmark/test-side.
