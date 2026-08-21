# Task 028 — Can n=10⁶ run in 10 ms/step on one H200?

## Standing answer (final): **yes — 9.591 ms**

Independently reproduced in H200 job **13029878** (source manifest
`42a6c254a11ac8a8`): **9.591 ms** [9.434, 9.631] per complete resident step at
n=10⁶, `expansion_order=3` (P=4), sampled-direct gradient relative RMS
**1.0593e-3**, inside the fixed 10 ms and 1.19e-3 gates. Configuration:
`sched6-5-5-5` geometry, `ell=5`, whole-level route windows, dense M2L with
FP16-input/FP32-accumulation WMMA, Float32, bounded Morton counting sort.

The progression was 91.4 → 69.6 → 64.6 → 38.0 → 30.7 → 15.4 → 12.5 → 9.591 ms;
each step is a numbered cycle/stage below with its own H200 job. **Sections 1–5
are the original Phase A record and state the pre-optimization answer ("no,
91.4 ms"); they are kept verbatim as the measurement baseline and are superseded
by §6b onward.** §6.7 carries the final result and §7 the standing threats to
validity.

Production defaults were moved onto this operating point during the clear-context
review (2026-08-03) — see §6.8. The shipped policy is `near_radius2=5` with the
`(6,5,…,5)` level schedule, device route windows cover a whole level, and precision
and M2L strategy are now selected per regime from the 024/028 measurements. At
`expansion_order=3` with no Lamb–Helmholtz those rules land exactly on this
configuration, so a caller who passes no options gets it.

---

## Phase A record (superseded by §6b onward)

**Measurement + bound classification + prioritized lever list. No optimization performed.**

Jobs: pilot **12996475**, sweep **12997508** (both H200 `m13h-1-1`, 2026-07-31).
Source manifest `0e57eca91e2c725b`. Julia 1.11.7, CUDA runtime 12.8.0 (local toolkit),
driver 580.159.4, H200 140.4 GiB. Data: `cuda_m13h-1-1_2026073*.csv` (+ `.classes.csv`)
in this directory. 27 rows, **all `fit=true`** — no failures, no OOM, no failure ledger.

---

## 1. Answer *(Phase A, superseded — see the standing answer above)*

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

### Lever 2 (job 13010174): partial — 5 ms on the F32 verdict path, 0 on F64

Grid-stride the fused dense M2L kernel instead of one block per route
(`blocks = min(n_routes, DENSE_CUDA_FUSED_MAX_BLOCKS[])`, cap 16384), cutting leaf
block count 31,307,680 -> 16,384.

| | after lever 3 | after lever 2 | delta |
|---|---|---|---|
| F32 `m2l_ms` (leaf L5) | 26.64 (23.58) | **21.61 (19.60)** | **-5.03 (-19%)** |
| F32 verdict step | 69.64 | **64.56** | **-5.08** |
| F64 `m2l_ms` (leaf L5) | 25.13 (22.41) | 25.69 (23.27) | +0.57 (+2%) |
| F64 verdict step | 82.90 | 83.38 | +0.48 |
| gradient rel RMS | 3.186e-4 | 3.186e-4 | unchanged |

**The §6b block-dispatch attribution was only partly right.** A 1900x dispatch
reduction bought 19% in F32 and nothing in F64, so dispatch was a real but minor
contributor rather than the dominant cost inferred from the kernel trace. Scoped at
10-20 ms, delivered ~5 ms on the verdict path.

What the experiment does establish: *before* lever 2 the leaf was 22.41 ms (F64) vs
23.58 ms (F32) — precision-insensitive, as a dispatch-bound stage should be. *After*,
it is 23.27 (F64) vs 19.60 (F32) — 16% faster in F32. Removing dispatch exposed an
underlying limit that **is** precision-sensitive, i.e. bandwidth or atomic throughput
(~500M atomic accumulations per step). Which of the two is unmeasured and is the next
question for this stage — a different lever from the one pulled here.

Kept because it is a clear 5 ms gain on the F32 verdict configuration; the F64
regression is inside the ~8% run-to-run variance recorded in §7.

Correctness: a new `fused dense M2L grid-stride parity` testset (8 tests) runs the
same problem at `DENSE_CUDA_FUSED_MAX_BLOCKS` = `typemax(Int)`, 3 and 1, forcing many
grid-stride iterations, and requires identical results up to atomic-reassociation
rounding. It is also **the only hierarchical coverage in the CUDA gate** —
`cuda_radix_lifecycle_test.jl` never builds a hierarchical cache, so its 215 tests
passed against a kernel that compiled to invalid IR (jobs 13000341/13009348/13009363).

### Standing after lever 3

**F32 verdict step 69.64 ms = 7.0x over target** (was 9.1x). The budget is now almost
entirely the two kernels: nearfield 38.4 ms (55%) + M2L 26.6 ms (38%) = **93%**, with
~4.6 ms of everything else. This is the state §5 anticipated: overhead is spent, and the
remaining gap requires ~6.5x from the two dominant kernels.

### Standing after lever 2

**F32 verdict step 64.56 ms = 6.5x over target** (9.1x -> 7.0x -> 6.5x). Budget:
nearfield 38.37 ms (59%), M2L 21.61 ms (33%, leaf 19.60), everything else ~4.6 ms (7%).
The nearfield is now the majority of the step on its own, so lever 1 is where any
further material gain has to come from.

### Lever 1 (jobs 13015315 / 13015316 / 13015336): -26.5 ms on the F32 verdict path

The nearfield kernel `_cuda_direct_pairs_output_kernel!` was one thread per
cell-pair — a serial ~(30x30)-interaction dependent chain with `inv(sqrt)` — at ~4%
of the FP32 peak rate. It is now **warp-per-pair with a grid-stride** (lanes stride
the target bodies, the source loop is lane-uniform so body loads broadcast; block
count capped by a new `DIRECT_CUDA_MAX_BLOCKS` = 16384), and `inv(sqrt)` became
`_cuda_fast_rsqrt`: hardware `rsqrt.approx` in F32, plus two Newton refinements in
F64 because raw `CUDA.rsqrt(::Float64)` is only ~1e-7 accurate — using it bare would
have silently degraded the F64 path to single-precision quality.

Riders in the same approved cycle: L2B warp-per-cell (0.96 -> 0.72 ms F64); the
per-step `CUDA.zeros` in `finalize_cuda_radix_output!` replaced by a cached device
buffer (also removes a double zero-fill); dead `_cuda_find_cell_for_sorted_body`
deleted. A B2M warp-per-cell rider was measured **slower** (1.02 -> 2.03 ms F64 —
113 registers and 15/32 active lanes at P=4, trace in fm028-13015316.out) and was
reverted; the measurement is recorded in a comment on the kernel.

| | before (13010174) | after (13015336) | delta |
|---|---|---|---|
| nearfield exact split, F32 | 37.7 ms | **11.45 ms** | **3.3x** |
| nearfield exact split, F64 | 51.9 ms | 37.6 ms | 1.38x |
| L2B alone, F64 | 0.96 ms | 0.72 ms | — |
| F32 verdict step | 64.56 ms | **38.04 ms** | **-41%** |
| F64 verdict step | 83.38 ms | 68.99 ms | -17% |
| gradient rel RMS | 3.186e-4 | 3.186e-4 | unchanged |

Scoped 20-33 ms, delivered 26.5 ms — inside the scoped band. The F64/F32 asymmetry
is informative: with dispatch and parallelism fixed, the remaining F64 cost is
genuine FP64 arithmetic throughput (rsqrt Newton chain + FP64 vector rate), i.e. the
F64 nearfield is now honestly compute-bound, while F32 — the verdict precision —
runs at 422 G interactions/s (4.83e9 / 11.45 ms), 3.4x the pre-lever rate.

Correctness: new `_cuda_fast_rsqrt accuracy` testset (max relative error 1e-14 F64 /
5e-7 F32 over 60 decades) and `nearfield warp-per-pair parity` testset (16 tests:
caps typemax/3/1 forcing deep grid-strides, F64+F32, n=2000 and n=40 for
ragged/empty cells). All gates green on all three jobs.

### Leaf-M2L mechanism resolved (derisk section D): loads/compute, not atomics

New A/B at fixed work on the captured leaf window (31,307,680 routes), comparing the
production fused kernel against script-local variants with plain racing stores
(same loads, no atomics) and no stores at all (loads+flops only):

| | atomic | plain store | no store |
|---|---|---|---|
| F64 | 21.07 ms | 25.15 ms | 24.35 ms |
| F32 | 17.21 ms | 19.37 ms | 19.00 ms |

Atomics are **not** the limit — they are slightly *faster* than plain racing stores,
and removing every write still leaves >95% of the cost. The leaf M2L is bound by the
operator/multipole **loads and the matvec compute** (each 32-thread block re-reads
its route's D x D operator column from L2 per route). The right next lever is data
reuse — shared-memory operator tiles with class-batched routes per block — not
atomic elimination.

### Standing after lever 1

**F32 verdict step 38.04 ms = 3.8x over target** (9.1x -> 7.0x -> 6.5x -> 3.8x).
Budget: leaf M2L 19.6 ms (52%), nearfield 11.45 ms (30%), L2B 0.4 ms, everything
else ~6 ms. The leaf M2L is now the dominant stage, with a measured mechanism and a
concrete rewrite direction. Remaining structural candidates beyond it: two-stream
nearfield/far-field overlap (nearfield reads only bodies + direct pairs and writes
`output` atomically, so it is independent of the whole far-field chain; needs the
blocking-sync/pageable-copy cleanup first) and a counting sort replacing the bitonic
`sortperm` in the grid rebuild (~2 ms of the 2.4-2.7 ms grid stage).

### Cycle 2 (job 13015753): operator-tiled leaf M2L, -7.3 ms on the F32 verdict

Acting directly on the section-D attribution: `_cuda_hier_dense_tiled_kernel!`
stages each class's D x D operator in shared memory once per contiguous same-class
route segment (window routes are class-sorted; a binary search finds segment ends),
folds both level diagonals into the tile at load time, and streams routes through
it with 4 warps and per-warp shared multipole columns. Per-route global traffic
drops from D^2 + D loads (~8.4 KB at P=4/F64) to D loads + D atomics (~0.5 KB).
Coarse windows below `DENSE_CUDA_TILED_MIN_ROUTES` = 65536 and any P whose tile
exceeds 48 KB keep the plain fused kernel.

| | after lever 1 | after cycle 2 | delta |
|---|---|---|---|
| leaf M2L, F64 | 23.27 ms | **13.66 ms** | -41% |
| leaf M2L, F32 | 19.60 ms | **12.71 ms** | -35% |
| F32 verdict step | 38.04 ms | **30.72 ms** | -7.3 ms |
| F64 verdict step | 68.99 ms | **58.77 ms** | -10.2 ms |
| gradient rel RMS | 3.186e-4 | 3.186e-4 | unchanged |

L4 (2.87M routes, also above the tiling threshold) improved in the same move
(1.85 -> 1.34 ms F32). The tiled leaf is nearly precision-INsensitive again
(13.66 F64 vs 12.71 F32), confirming the removed operator traffic was the
precision-sensitive component the lever-2 experiment exposed; the residual ~13 ms
is compute/atomic/latency-bound and needs a fresh attribution pass before more
work goes in. Correctness: `dense M2L operator-tile parity` 12/12 (tiled-vs-fused
across forced-tiled / deep-chunk / one-block launches, F64+F32).

### Standing after cycle 2

**F32 verdict step 30.72 ms = 3.1x over target**
(9.1x -> 7.0x -> 6.5x -> 3.8x -> 3.1x). Budget: leaf M2L 12.7 ms (41%), nearfield
11.45 ms (37%), everything else ~6.6 ms (21%). The two dominant kernels are now
balanced, which makes the two-stream nearfield/far-field overlap (~11 ms of
hideable work, prerequisite: blocking-sync + pageable-copy cleanup) the natural
next structural lever, followed by fresh per-kernel attribution passes.

### Cycle 3 (job 13015982): nearfield/far-field stream overlap — 0 ms, negative result

The nearfield (fill + `_cuda_direct_pairs_output_kernel!`) now launches on a
non-blocking side stream before B2M, ordered by device events only: a begin event
keeps the side-stream fill behind the previous step's finalize scatter, and a done
event orders L2B (non-atomic `+=` on `output`) after the nearfield atomics. Gated
by `CUDA_OVERLAP_NEARFIELD`; the standalone `_launch_cuda_resident_l2b!` is
unchanged for every benchmark that times it. Parity: 4/4 across a 3-step
convection loop, on vs off, both precisions.

Measured effect at the verdict config: **none** — F32 verdict 30.718 vs 30.724 ms,
F64 58.73 vs 58.77, eval-only identical to 0.01%. The mechanism is instructive:
overlap shortens wall time only when one stream leaves SMs idle, and after levers
1-2 both dominant kernels saturate the device (F32 eval 27.5 ms vs a ~28.6 ms
per-stage sum ⇒ >95% busy). Two saturating kernels time-share SMs and take the sum
regardless of streams. The Phase-A-era overlap estimate (5-15 ms) was conditioned
on the old latency-bound nearfield, which lever 1 eliminated — **succeeding at
lever 1 retired this lever.** Kept (default on) because it is correctness-neutral,
parity-tested, and can only help in launch/latency-bound regimes (small n, thin
windows); it does nothing at n=10⁶.

### Standing after cycle 3

Unchanged from cycle 2: **F32 verdict 30.72 ms = 3.1x over target.** The step is
now ~95% device-saturated compute in three kernels (leaf M2L 12.7, nearfield 11.45,
plus ~6.6 ms of small stages). With saturation established, further gains must come
from *reducing work or increasing per-kernel efficiency* — fresh attribution passes
on the residual leaf M2L (precision-insensitive again ⇒ compute/atomic/latency) and
the F32 nearfield (422 G interactions/s vs the ~1.6 ms roofline floor) are the
remaining levers, plus the ~2 ms counting sort.

### Cycle 4 — residual attribution plus a missing stencil-radius lever

The next step is an evidence-only H200 pass, not another production rewrite. Cycle 2
changed leaf M2L's limiting mechanism, and cycle 3 established that both dominant
kernels saturate the device, so their residuals need fixed-work A/Bs after the latest
code: atomic/store/no-store variants and launch-shape sweeps for both tiled leaf M2L
and warp-per-pair nearfield.

The pre-run audit also found a material lever absent from the original list:
**intermediate hierarchical near radii**. Production exposes only q=3 and q=12, but
direct enumeration shows every integer q=3:12 satisfies the task-025 downward-
monotonicity requirement. On the target's full ell=5 grid, the distinct candidate
shells have these exact directed work counts:

| q | near offsets | phase offsets | direct cell pairs | total routes |
|---:|---:|---:|---:|---:|
| 3 | 27 | 189 | 830,584 | 6,039,504 |
| 4 | 33 | 231 | 1,014,904 | 7,367,184 |
| 5 | 57 | 399 | 1,729,144 | 12,277,584 |
| 6 | 81 | 567 | 2,421,064 | 16,820,208 |
| 8 | 93 | 651 | 2,766,664 | 19,075,248 |
| 9 | 123 | 861 | 3,614,440 | 24,456,912 |
| 10 | 147 | 1,029 | 4,304,872 | 28,925,424 |
| 11 | 171 | 1,197 | 4,973,728 | 33,063,216 |
| 12 | 179 | 1,253 | 5,189,728 | 34,343,088 |

Unlike ell retuning, reducing q reduces both dominant workloads. Linear projections
from the q=12 measurements put nearfield+M2L at ~18.1 ms for q=9 and ~12.3 ms for
q=6, versus ~25.7 ms now. Those are work-model projections only. The sampled-direct
error frontier is unmeasured and decides whether any shell is admissible. The first
attribution job remeasures the already-supported optimized q=3/q=12 endpoints to
validate the work model; generalizing and sweeping intermediate q is a subsequent
production/theory change requiring the next user checkpoint.

Other newly explicit candidates are target-cell-owned nearfield (one final write per
target rather than atomics per direct cell pair), a self-interaction-only symmetric
nearfield path (one distance evaluation per unordered body pair), and tensor/mixed-
precision 16x16 class-batched leaf M2L with FP32 accumulation. Native real-basis
operators are lower priority: at code order 3 their analytic lane saving is 20%, not
2x. FFT/convolution M2L remains a high-risk algorithm replacement. Full rationale
and the exact attribution matrix are in `plans/20260801_028_attribution.md`.

#### Measured result — job 13016756

Both production gates passed.  The q=12 endpoint reproduced cycle 3 (30.784 ms
versus 30.72 ms), removing the old Phase-A hierarchical timing variance as a
concern for this comparison.  The endpoints are:

| policy | M2L ms | L2B + nearfield ms | eval ms | verdict ms | gradient relative RMS | gate |
|---|---:|---:|---:|---:|---:|---|
| hier3 | 2.817 | 2.267 | 6.545 | **9.434** | 3.995e-3 | **fail**, 3.4x over |
| hier12 | 14.225 | 11.872 | 27.559 | 30.784 | 3.186e-4 | pass |

This is a timing/accuracy bracket, **not a successful 10 ms verdict**.  q=3 proves
the optimized implementation can execute the target workload in under 10 ms, but
only at inadmissible accuracy.  Its M2L time is 19.8% of q=12 for 17.6% of the
routes; its nearfield time is 19.1% for 16.0% of the direct cell pairs.  The static
work model is therefore accurate enough to choose experiments.  A two-endpoint
stage fit projects q=4 at ~10.4 ms and q=5 at ~14.0 ms; their errors are unknown.

The exact-once geometry also permits a non-increasing radius with depth. If
`q_child <= q_parent`, the task-025 downward-monotonicity lemma remains sufficient.
All 45 two-stage combinations of the distinct shells pass an independent audit.
The most relevant first case, q=4 on levels 2--4 and q=3 at the leaf, has 7.55M
routes plus q=3's 0.831M direct cell pairs and projects near 10.1 ms. Because it
retains q=3 for leaf interactions, it is a conditional follow-up—not a
substitute for measuring uniform q=4 first. Counts are in
`radius_schedule_candidates.csv`.

The post-rewrite fixed-work A/Bs identify the residual mechanisms:

| experiment | result | decision |
|---|---|---|
| M2L launch | 10.463 ms current-like; **9.636 ms** at 64 threads/cap 65536 | end-to-end confirm the isolated 0.827 ms gain |
| M2L output | 10.461 atomic; 8.752 store; 8.660 no-store | atomics ~1.71 ms, but arithmetic/input is ~83% of the kernel |
| nearfield launch | 11.456 current-like; 11.328 best | <=0.13 ms; do not spend a cycle |
| nearfield output | 11.452 atomic; 11.470 store; 11.438 no-store | output atomic/store cost is immaterial |

Consequently the next bounded cycle should generalize and sweep the intermediate
rigid radii with the full accuracy gate.  Compute-oriented tensor/class-batched M2L
is second.  A symmetric self-interaction nearfield path remains the strongest
nearfield lever because it reduces distance evaluations.  Target-cell ownership is
demoted: source reuse and pair-metadata savings remain possible, but its proposed
atomic benefit was refuted.  Counting sort remains a low-risk ~2 ms follow-on.
Expansion-order/radius co-design is also conceptually available, but changing
literature P=4 would change the user-fixed target definition and is not an in-scope
028 optimization without a new decision.

Data: `radius_candidates.csv`, `radius_schedule_candidates.csv`,
`attribution_m13h-1-1_20260801-062135_{m2l,nearfield}.csv`,
`cuda_m13h-1-1_20260801-062246.csv` (+ `.classes.csv`), and
`fm028-13016756.out`; source manifest `2708401df42a7ee5`, H200 `m13h-1-1`,
Julia 1.11.7, CUDA 12.8.0.

## 6.5 Stages 5–6: intermediate radii and bounded re-optimization (2026-08-01)

The production geometry now accepts exactly the distinct lattice shells
`q in (3,4,5,6,8,9,10,11,12)` while retaining q=12 as the default and q=3 as
`classic_fmm_stencil`. Separator endpoints and V-list extents are enumerated or
derived (`2isqrt(q)+1`); q=7 is rejected as redundant. The extended task-025
verifier passes all-shell theta/separator, monotonicity, and exact-once
dense/sparse/boundary coverage checks.

Two independent H200 runs measured the complete full-class-window frontier:

| q | full-window K | verdict ms (green run) | reproduced ms | gradient relative RMS | gate |
|---:|---:|---:|---:|---:|:---:|
| 3 | 316 | 9.515 | 9.517 | 3.995e-3 | fail |
| 4 | 418 | 10.515 | 10.559 | 1.683e-3 | fail |
| 5 | 682 | 14.230 | 14.221 | 1.422e-3 | fail |
| 6 | 850 | **17.383** | 17.431 | **5.745e-4** | pass |
| 8 | 982 | 19.070 | 19.128 | 5.508e-4 | pass |
| 9 | 1252 | 23.175 | 23.168 | 5.563e-4 | pass |
| 10 | 1516 | 26.544 | 26.703 | 3.436e-4 | pass |
| 11 | 1684 | 29.707 | 29.708 | 3.682e-4 | pass |
| 12 | 1740 | 30.756 | 30.664 | 3.186e-4 | pass |

Thus q=6 is the fastest admissible geometry; no admissible shell reaches 10 ms.
Job 13016927 passed every preflight gate. Job 13016917 reproduced the entire
frontier but is retained in the failure ledger because one stochastic Float32
nearfield-overlap parity check failed before the benchmark; all other gates and
all nine measured rows completed.

The selected q=6 depth bracket is decisive: ell=4/5/6 measured
33.946/17.525/72.901 ms, with errors 5.646e-4/5.745e-4/5.797e-4. At ell=5,
the 64-thread/65,536-block tiled-M2L launch reproduced the isolated gain in both
complete verdicts and was retained:

| precision | launch | M2L median [range] ms | verdict median [range] ms | error |
|:---|:---|---:|---:|---:|
| Float32 | 128 / 16384 | 7.161 [7.155, 7.170] | 17.455 [17.293, 17.511] | 5.745e-4 |
| Float32 | **64 / 65536** | **6.871 [6.864, 6.878]** | **17.142 [16.990, 17.215]** | 5.745e-4 |
| Float64 | 128 / 16384 | 7.699 [7.689, 7.722] | 29.471 [29.427, 29.658] | 5.745e-4 |
| Float64 | **64 / 65536** | **7.319 [7.306, 7.494]** | **28.957 [28.908, 29.942]** | 5.745e-4 |

The 15-bit bounded Morton counting-sort prototype also passed its 15/15
permutation, ascending-key, inverse-permutation, cell-compression, repeatability,
capacity-reuse, and sortperm-parity gate. Histogram clear, scan, scatter, inverse
permutation, cell compression, and initialization are all inside the measured
refresh/cache boundaries. It reduced refresh and full verdict with unchanged
accuracy and no incremental transfer or persistent-identity violation, so it was
retained for bounded depths (ell <= 6):

| precision | sorter | refresh ms | verdict median [range] ms | error |
|:---|:---|---:|---:|---:|
| Float32 | sortperm! | 2.742 | 17.142 [16.998, 17.184] | 5.745e-4 |
| Float32 | **counting** | **0.967** | **15.428 [15.249, 15.460]** | 5.745e-4 |
| Float64 | sortperm! | 2.824 | 28.938 [28.891, 29.643] | 5.745e-4 |
| Float64 | **counting** | **1.056** | **27.162 [27.144, 27.182]** | 5.745e-4 |

Jobs: 13016917/13016927 (Stage 5), 13017112 (depth and launch A/B), and
13017128 (counting A/B). Raw artifacts are the corresponding `fm028-*.out` files
and `cuda_m13h-1-1_20260801-{081258,082740,095606,095724,095851,101516,101642}.csv`
files plus class companions. The best measured admissible complete verdict is
15.428 ms, so the 10 ms objective remains open for later stages.

Final independent gate job 13017362 passed the radix lifecycle (215/215),
convection/optimized-kernel gates, and the expanded all-radius CUDA hierarchy
suite (32,264/32,264). With the retained production internals it independently
measured q=6 at 15.401 ms [15.226, 15.495] in Float32 and 27.350 ms
[27.292, 27.437] in Float64; both gradient errors were 5.745e-4. Its raw
artifacts are `fm028-13017362.out` and
`cuda_m13h-1-1_20260801-105134.csv` (+ class companion), source manifest
`437d12a2be8e14b5`. This reproduces the selected result and confirms that the
10 ms target was not reached.

## 6.6 Stage 7: replay and level-scheduled frontier (2026-08-03)

H200 job **13027263** (node m13h-1-2, source manifest
`42a6c254a11ac8a8`) passed the lifecycle (215/215), host parity (37/37),
convection/optimized-kernel, and expanded scheduled/symmetric/tensor hierarchy
gates (33,367/33,367). The q=5 replay's relative reconstruction errors were
1.760e-7 for q=5 and 1.842e-7 for q=6, so its level/orbit partial fields faithfully
reconstructed the device result. Uniform q=5 nevertheless missed the 1.19e-3
accuracy gate at 1.4224e-3.

The complete frontier, including refresh, scratch reduction, and bounded counting
sort, was:

| policy | level schedule 2--5 | verdict ms | gradient relative RMS | gate |
|:---|:---:|---:|---:|:---:|
| hier5 | 5-5-5-5 | **12.273** | 1.4224e-3 | fail |
| **sched6-5-5-5** | **6-5-5-5** | **12.514** | **1.0498e-3** | **pass** |
| sched6-6-5-5 | 6-6-5-5 | 12.668 | 7.8208e-4 | pass |
| sched6-6-6-5 | 6-6-6-5 | 14.152 | 6.4240e-4 | pass |
| hier6 | 6-6-6-6 | 15.445 | 5.7453e-4 | pass |

Thus `sched6-5-5-5` is the fastest admissible geometry. It improves the same-run
uniform q=6 verdict by 18.9%, but remains 2.514 ms above the target. Evidence is
`fm028-13027263.out`, `stage7_replay_m13h-1-2_20260803-075304.csv`, and
`cuda_m13h-1-2_20260803-075427.csv` plus its class companion.

Failure ledger: jobs 13027048, 13027092, 13027167, 13027174, and 13027188 hit hard
gates that exposed a device-context array-rank type coupling, Julia-1.11 host BF16
lowering, route-class/test portability defects, an invalid symmetric-context field
access, and an invalid replay-context field access, respectively. Job 13027374 was
cancelled after its submission was found to predate a winner-selector full-path fix.
All defects were corrected and exercised by the green job 13027263. These failed or
cancelled jobs provide no benchmark evidence.

## 6.7 Stage 8: competitive residual-kernel bake-off

H200 job **13029480** repeated the lifecycle, host-parity, convection/optimized,
and expanded 33,367/33,367 CUDA gates before measuring the selected geometry. Its
source manifest was `42a6c254a11ac8a8`. The isolated candidates were:

| candidate | M2L ms | verdict median [range] ms | gradient relative RMS | decision |
|:---|---:|---:|---:|:---|
| Float32 tiled baseline | 5.472 | 12.547 [12.372, 12.554] | 1.0498e-3 | reference |
| unordered symmetric nearfield | 5.473 | 52.480 [52.303, 53.953] | 1.0498e-3 | reject |
| TF32 cuBLAS / FP32 accumulate | 34.165 | 41.673 [41.483, 50.183] | 1.1424e-3 | reject |
| FP16 WMMA / FP32 accumulate | **2.548** | **9.603 [9.421, 9.658]** | **1.0593e-3** | reproduce |
| BF16 WMMA / FP32 accumulate | **2.549** | **9.580 [9.419, 12.097]** | **1.0632e-3** | reproduce |

Both WMMA formats crossed 10 ms while remaining inside the 1.19e-3 accuracy gate.
The 0.023 ms difference between their medians is not meaningful, while FP16 had a
materially tighter first-run range, so the independent final job repeats both.
There is no combined bake-off: symmetric nearfield is 4.2x slower than baseline,
and combining a rejected candidate cannot strengthen either WMMA result.

The Float64 singular-spectrum audit found route-weighted admissible rank 7.064 of
16 and modeled operator-work fraction 0.883 (`prototype_allowed=false`), so no
low-rank application was implemented. Job 13028465 is retained only in the failure
ledger: its initial TF32 harness used 16-route chunks, forcing roughly 769,000
host-driven gather/GEMM/scatter iterations per M2L pass. It was cancelled and TF32
was rerun with production-sized 16,384-route per-class batches in job 13029480.

Raw evidence is `fm028-13029480.out`,
`stage8_singular_spectra_20260803-120058.csv`, and
`cuda_m13h-1-1_20260803-{120117,120250,120428,120545,120701}.csv` plus class
companions.

Independent H200 job **13029878** (source manifest `42a6c254a11ac8a8`) passed the
lifecycle (215/215), host-parity (37/37), convection/optimized-kernel, and expanded
CUDA hierarchy gates (33,367/33,367). It then reproduced both candidates:

| format | M2L ms | verdict median [range] ms | gradient relative RMS | gate |
|:---|---:|---:|---:|:---:|
| FP16 / FP32 accumulate | 2.543 | **9.591 [9.434, 9.631]** | **1.0593e-3** | pass |
| BF16 / FP32 accumulate | 2.542 | **9.638 [9.473, 9.713]** | **1.0632e-3** | pass |

`STAGE8_REPRO_EXIT=0`. Evidence is `fm028-13029878.out` and
`cuda_m13h-1-1_20260803-{131016,131129}.csv` plus class companions. FP16 is the
final reproduced winner: it is faster and slightly more accurate in the independent
run. The fixed 1M-particle, P=4, resident single-H200 objective is therefore met;
the early-stop rule closes task 028 before Stage 9.

## 6.8 Production defaults moved onto the measured optimum (2026-08-03, review)

The clear-context review found that every Stage 5–8 gain was reachable only through
internal `Ref` knobs and an internal scheduled-policy constructor, so a caller who
passed no options still ran the pre-028 geometry. On user direction the defaults now
ship the measured optimum, and the schedule surface is public:

| default | before | after | evidence |
|---|---|---|---|
| `near_radius2` | 12 | **5** | Stage 7 frontier |
| level schedule | none (uniform) | **`(6, 5, …, 5)`** for `ell >= 3` | Stage 7: 12.514 ms vs 15.445 ms uniform q=6 |
| device `window_classes` | 256 | **4096** (⇒ one window per level) | Phase A: route_gen 24.19 → 2.13 ms |
| `DENSE_CUDA_TENSOR_FORMAT` | `:off` | **`:fp16`** | Stage 8: M2L 5.472 → 2.548 ms |
| counting sort, tiled launch shape | already on / 64 threads, 65,536 blocks | unchanged | Stage 6 |
| `options.precision` | `Float64` | **`Float32` when `expansion_order <= 3`** | 024 error table + §4.3 |
| `options.m2l_strategy` | `ConcatenatedFixedZM2L` | **selected per regime** (table below) | 024 recurring-step rules + §4.4 |

`RadixFMMCache(...; options=nothing)` now resolves both from the expansion order, the
Lamb–Helmholtz channel, the platform, and the dense operator footprint; an explicit
`options` bypasses the rules, and the resolution is readable at
`cache.state.options`.

| selector | precision | M2L strategy |
|---|---|---|
| `expansion_order <= 3` (literature `P <= 4`) | `Float32` | dense |
| `expansion_order <= 7`, LH off | `Float64` | dense |
| `expansion_order <= 7`, LH on | `Float64` | precomputed-y on device, dense on host |
| `expansion_order >= 8` (literature `P >= 12`) | `Float64` | precomputed-y |
| dense operator payload over its gate | unchanged | precomputed-y |

Why these and not a flat "always Float32 + dense":

- Task 024 measured Float32's max gradient error at **5.33e-4 (CPU) / 5.34e-4 (H200)
  essentially independently of `P`**, against 5.25e-6 for Float64 — an accuracy
  *floor*, not a proportional cost. At `P = 4` that floor is under the stencil's own
  truncation error, and a host check at n = 1500 confirms it: gradient relative RMS
  1.62264e-3 (Float32) vs 1.62263e-3 (Float64) at identical geometry. Above `P = 4`
  Float32 would discard exactly the accuracy the higher order was bought for.
- Dense won every measured `P = 4` case on both platforms and every `P = 8` LH-off
  case, and 028 §4.4 confirmed it over precomputed-y on the hierarchical path. But
  all Float32/`P = 12` dense rows are unsupported, clustered Float64/`P = 12`/LH-on
  needs 30.1 GiB against a 4 GiB gate, and `P = 8`/LH-on reaches 7.4 GiB — so an
  unconditional dense default would convert working caches into construction errors.
  The footprint check falls back instead of throwing.
- Concat (the previous default) won **no** steady-state case at any order in 024, so
  the historical default was not the measured choice anywhere.
- Dense trades ~20 s construction for the best steady state (~300–370 break-even
  steps at the 028 target), which suits this repeated-step cache but not one-shot
  evaluation; the docstring names the explicit precomputed-y opt-out.

At `expansion_order = 3` with no Lamb–Helmholtz these rules land exactly on the
task-028 winner — Float32, dense, FP16 tensor M2L, `sched6-5-5-5` — so the reproduced
9.591 ms configuration is now what a caller gets by passing nothing.

Consequences a caller sees without asking for them:

- **Accuracy.** The default geometry is less accurate than the previous uniform
  `q = 12`: 1.0498e-3 vs 3.186e-4 gradient relative RMS at the 028 target
  configuration, and 4.60e-6 / 2.93e-4 vs 5.13e-7 / 2.99e-5 max potential/gradient
  error on the `ell = 4`, `P = 5`, n = 3000 device-default regression case. Both are
  inside the project's order-of-truncation-error gate. `near_radius2 = 12` restores
  the old operating point and, passed explicitly, also selects a uniform geometry.
- **Memory.** Whole-level route windows grow the persistent route buffers as
  `noffsets * max_level_nodes` instead of `256 * max_level_nodes` — 2.0 GB persistent
  at n = 10⁶, `ell = 5`. Pass a smaller `window_classes` on constrained devices.
- **Arithmetic.** FP16 M2L engages only for Float32, no Lamb–Helmholtz,
  `expansion_order = 3`, dense strategy; every other configuration keeps the FP32
  tiled kernel. Its operator scaling is derived from the operator alone, so a problem
  whose source strengths sit many orders of magnitude from the validated benchmark's
  can underflow the FP16 input side; `DENSE_CUDA_TENSOR_FORMAT[] = :off` restores FP32.
- **Precision and strategy** are no longer fixed constants but the regime-gated
  selection above, so `expansion_order >= 4` keeps Float64 and high orders keep
  precomputed-y.

The host suite (`hierarchical_m2l_host_test.jl`, 681/681) covers the new default
geometry — including exact-once coverage at `ell = 2, 3, 5` — the public
`level_radii2` keyword, and its three invariants; the full `Pkg.test()` suite is
green (exit 0, no failures in any file). The FP32 fused/tiled parity testsets are
pinned to `:off` so they keep testing their own kernel.

The CUDA gate was re-run on the cluster. Job **13031187** failed it: two testsets had
used "pass no options" as a proxy for "the concat default" and silently moved to the
dense plan, tripping 4,015 assertions on the dense offset-local class-id convention
and on level scales. Both now pin `ConcatenatedFixedZM2L`; no production defect was
involved. Job **13031482** (source manifest `c0afa01083322fb8`) then passed every
gate — lifecycle 215/215, host parity 37/37, convection and optimized-kernel sets,
counting sort 20/20, hierarchical **33,381/33,381** including the new
`expansion_order = 3` device default-stack block — and reproduced the winner under
the new defaults at **9.653 ms** [9.473, 9.689] / 1.0593e-3 (FP16) and 9.632 ms
[9.472, 9.690] / 1.0632e-3 (BF16), with the residency and allocation contracts
unchanged. The 0.06 ms against job 13029878's 9.591 ms is inside the recorded
run-to-run variance.

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
