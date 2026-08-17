# 041a — Adaptive Octree vs Uniform Radix Grid: Benchmark Report

Evidence base for the `042` default-selection audit: the definitive
time-and-memory comparison of the old uniform-depth radix grid and the new
2:1-balanced adaptive Morton octree (rows `038`–`041`), on uniform and
non-uniform fields, host and H200, with every quoted comparison at stated
sampled-direct accuracy under the phase velocity gate (rel RMS <= 1e-3,
2000 sampled targets vs exact Float64 all-source direct references).

## 1. Method and provenance

- Cases: `unitcube` (uniform), `wake` (033 helical wake cylinder positions,
  3.14% fill), `multiscale100` (unit cube + embedded cluster, contrast as a
  parameter), a sigma-heterogeneous `SmoothedVortex` variant, all with the
  fm039–fm041 generators/seeds. Gravitational P=4 q=5 unless stated;
  `DenseTranslationM2L` on H200 (the fm041-measured best family), default
  host path on CPU nodes; warm medians of 5 (stage rows: CUDA-event
  medians of 7), same-job anchors throughout — **no ratio in this report
  crosses jobs**.
- Jobs of record (pre-registration committed as `ecebdb8` before
  submission; sacct-verified terminal states):
  - **13184013** (H200, COMPLETED 0:0, 1:18:22) → `fm041a_gpu_widen.csv`
    (165 rows: F64 dense grid, uniform ell 3–7 x adaptive K 32–256 x
    n in {1e4, 3.16e4, 1e5, 3.16e5, 1e6} x 3 cases; F32 subset),
    `fm041a_pweep.csv` (P in {2,3,6,8}), `fm041a_leafpop.csv`.
  - **13184014** (H200, COMPLETED 0:0, 1:20:58) → `fm041a_gpu_contrast.csv`
    (contrast 1–1000, n=1e6), `fm041a_contrast_leafpop.csv`,
    `fm041a_gpu_stages.csv` (per-stage + graph/overlap/refresh A/B),
    `fm041a_gpu_sigma.csv` (spread 1–300, n=1e5, RegularizedVortex + LH).
  - **13184015** (CPU node, single thread; CANCELLED 0:0 at 1:35:07 by the
    2026-08-15 user pause-all, AFTER the final pre-registered row was
    written — the CSV is complete and is the host record) →
    `fm041a_host_widen.csv` (widened host depths + K, same-job anchors).
  - All three jobs were requeued once by the scheduler (~22:17 MDT) and
    re-ran from scratch in a single incarnation; both GPU jobs shared node
    m13h-1-2 (one H200 each). Prior-row records: fm039 (13178905), fm040
    (13179323), fm041 (13182172).
- Accuracy: every configuration passed the 1e-3 velocity gate — CSV-wide
  maxima: GPU widen 7.937e-4 (unitcube n=1e4 ell=7, an unplotted row;
  the plotted/quoted GPU-widen rows max at 6.6e-4), host 6.6e-4, sigma
  sweep 9.5e-4, contrast sweep 5.435e-4 (the plotted adaptive c=30 row).
  No gate-failing row is used as a winner anywhere; construction
  failures are fail rows with error text.

## 2. Headline verdicts (best-vs-best, widened sweeps, same-job)

The 040-approval obligation — widen the uniform depth sweep before any
publishable claim — was **decisive**: the fm040/fm041 baselines (ell in
{5,6} only) overstated the adaptive advantage on two of three cases.

| case, n=1e6 | platform | best uniform (widened) | best adaptive | adaptive gain |
|---|---|---|---|---|
| unitcube | H200 F64 | 92.3 ms (ell=5) | 95.9 ms (K=64) | 0.96x (parity, -4%) |
| wake | H200 F64 | 96.6 ms (**ell=7**) | 113.4 ms (K=256) | **0.85x (uniform wins)** |
| multiscale100 | H200 F64 | 246.3 ms (**ell=7**) | 132.4 ms (K=64) | **1.86x** |
| unitcube | host 1T | 27.5 s (ell=5) | 28.8 s (K=64) | 0.96x (parity) |
| wake | host 1T | 21.3 s (**ell=7**) | 23.6 s (K=32) | **0.90x (uniform wins)** |
| multiscale100 | host 1T | 79.9 s (**ell=7**) | 26.7 s (K=64) | **2.99x** |

At n=1e5 (both platforms) best-uniform wins every case by 1.05–1.6x —
the adaptive path's fixed refresh/list overhead is not amortized there.

**Corrections to prior-row headlines** (they used the narrow ell in {5,6}
baseline): the fm040 host wake "2.42x adaptive win" inverts to a 0.90x
loss against ell=7 (77.998 s at ell=6 vs 21.323 s at ell=7 in the same
job); fm040's multiscale 3.27x becomes 2.99x; fm041's multiscale 1.87x
survives essentially unchanged (1.86x vs interior ell=7); fm041's wake
1.27x-slower honest negative worsens slightly to 0.85x vs ell=7 (but see
memory, §5). The fm041 cube 1.11x GPU win becomes 0.96x parity.

## 3. Where the adaptive octree wins — and the mechanism (figs 14, 19, 16)

- **Cluster contrast** (fig14, n=1e6 H200): every fixed uniform depth
  blows up with contrast (ell=4: 151 ms -> 284,000 ms across c=1->1000;
  ell=5: 106 -> 6,185 ms), because a fixed grid leaves fat cells whose
  O(K^2) direct work concentrates in the cluster (fig19: at c=1000 and
  ell=5 the leaf-population CCDF has a tail beyond 20,000 bodies/leaf;
  adaptive truncates at exactly K_max=64 by construction). The
  best-uniform envelope survives by moving ever deeper (ell=5 still best
  at c=30 with 170.1 ms; ell=7 best from c>=100), paying 54–58 GB of
  device capacity at ell=7; adaptive stays in a 120.8–156.0 ms band at
  5.2 GB from c=3 to c=1000 (c=3/c=30 sit at the ~156 ms top of the
  band; fig14 plots every point): **1.67x at c=100, 2.11x at c=1000** vs
  the best gate-passing uniform depth, with **11x less device memory**.
  (Honest endpoint: at c=1 the generator's contrast ball spills outside
  the unit cube, inflating W/X to 8.1e5 entries — adaptive 321 ms vs
  uniform 106 ms; a case-geometry artifact recorded as-is.)
- **Stage mechanism** (fig16, n=1e6 F64 serialized stage medians): the
  uniform grid loses either to fat-cell direct(U) (wake ell=5: 155 of
  158 ms; multiscale ell=5: 174 of 186 ms) or, when driven deep, to
  B2M+M2L(V)+L2B over exploding cell counts (cube ell=7: 43+104+17 ms).
  The adaptive tree bounds direct(U) at 9–13 ms everywhere and pays
  instead a new S2L(X) + M2T(W) cost (wake: 22.2 + 7.2 ms; multiscale:
  19.6 + 5.6 ms) — which is **most of the adaptive-vs-deep-uniform gap on
  the wake** (54.7 vs 26.0 ms serialized): S2L is the single largest
  adaptive stage there and is the top tuning target for `042`.
- **Sigma heterogeneity** (fig20, n=1e5 vortex, RegularizedVortex + LH):
  the uniform path's global sigma_max geometry gate progressively
  disqualifies depths (spread 100: ell>=5 throw; spread 300: ell>=3
  throw, only ell=2 survives at 80.7 ms), while the per-cell sticky-
  demotion gate keeps the adaptive path valid and flat (27.9–30.5 ms,
  gate-passing) at every spread: **2.6x vs the only surviving uniform
  depth at spread 300**, and the only machinery that keeps a deep tree at
  all. Honest note: at spreads <= 100 the surviving uniform ell=3 is
  ~2x FASTER than adaptive (14 vs 28 ms); the per-cell gate's value is
  robustness at extreme spreads (CoreSpreading-grown sigma), not raw
  speed at mild ones.

## 4. Honest negatives (required by the row)

1. **Wake, both platforms, n=1e6**: with the widened sweep the uniform
   grid at ell=7 beats the best adaptive configuration — H200 0.85x
   (96.6 vs 113.4 ms), host 0.90x (21.3 vs 23.6 s). The prior fm041 GPU
   negative (1.27x slower vs ell=6) understated this; the fm040 host wake
   "win" was a sweep-endpoint artifact. The wake is uniformly sparse, so
   a deep uniform grid IS the right partition — the adaptive tree matches
   its geometry (popmax 37 at ell=7 vs K=64/32) but pays S2L/M2T and
   refresh overhead on top.
2. **Wake/cube host n=1e5**: the fm040 negative persists at every probed
   K (K=16/32/64): best uniform 1.85 s vs best adaptive 2.03 s (wake);
   cube 2.20 vs 2.54 s.
3. **n <= 1e5 on H200**: best-uniform wins all three cases (e.g. cube
   7.4 vs 8.1 ms; wake 7.6 vs 12.3 ms; multiscale 15.2 vs 17.3 ms at
   n=1e5) — adaptive construction/refresh overhead dominates small n.
4. **K-sensitivity**: adaptive t_step is non-monotonic in K at mid n
   (e.g. wake 1e6 F64: K=128 gives 214.7 ms vs K=256's 113.4 ms; cube
   3.16e4: K=64 an outlier vs K=128) — per-case K tuning matters and the
   defaults should not be assumed transferable; flagged for `042`.

## 5. Memory (fig17)

Device bytes in use after construction, n=1e6 F64 dense: the uniform
grid's capacity grows ~8x per depth level once the dense `node_at`
Sigma 8^ell table and per-cell capacities engage — ell=5: 2.0–2.2 GB,
ell=6: 14.2–16.1 GB, **ell=7: 53.9–59.5 GB** — while the adaptive
occupancy-sized capacity spans a narrow band, **3.1 GB (K=256) to 8.3 GB
(K=32)**, on every case. Consequences at the widened best-vs-best:

- wake: uniform's 0.85x/0.90x time win costs **17x the device memory**
  (53.9 GB at ell=7 vs 3.1 GB at K=256): on a shared or smaller GPU the
  adaptive configuration is the only one of the two that fits with room
  to spare, and ell=8 (the uniform cap) would not fit an H200 at all.
- multiscale: adaptive wins time (1.86x) AND memory (11.2x, 5.2 vs
  58.2 GB) simultaneously.
- Endpoint disclosure: the winning/best-uniform ell=7 is the swept-range
  ENDPOINT on wake and multiscale, on BOTH platforms. Extending to ell=8
  is memory-infeasible, not merely unswept: device capacity grows ~8x
  per level from 54–58 GB at ell=7 (an H200 has ~141 GB), and the host
  job ran in 64 GB; ell=8 is also the uniform path's hard cap. The
  best-uniform curves are therefore effectively complete even where they
  end at ell=7. Symmetrically, wake-GPU's best adaptive K=256 is a sweep
  endpoint too — given the observed K non-monotonicity, an unswept
  K=512 could narrow the 0.85x wake loss; flagged with the K-tuning item
  for `042`.
- The uniform ell<=8 cap remains (041 decision: untouched); the adaptive
  path has no depth cap below `RADIX_GRID_MAX_ELL`=21 and ran ell_max=10
  throughout.

## 6. Accuracy–cost frontier (fig18)

P in {2,3,4,6,8} at fixed geometry (adaptive K=64; uniform ell=6/7) plus
the P=4 geometry sweeps, wake + multiscale, n=1e6 F64: both machineries
trace comparable frontiers, and at the P=4 operating point both sit
1.5–2.5x inside the 1e-3 gate (wake 5.0–6.6e-4, multiscale 4.1–4.6e-4);
the adaptive multiscale frontier dominates uniform's at every P (its
error is set by the same operator tables, its time by the bounded leaf
work). No comparison in §2 is bought with accuracy: best-adaptive and
best-uniform deliver the same error order at every quoted point.

## 7. Refresh, graph capture, and priced items

- **Frozen-leaf-set refresh (measured)**: the adaptive occupancy-epoch
  fast path refreshes in 9.0–9.4 ms warm at n=1e6 (vs 50.9–92.9 ms for a
  forced full rebuild — 5.7x–9.9x saved when the leaf set is stable) and
  sits within 1.2–1.6x of the uniform refresh (5.8–7.9 ms).
- **Graph capture / nearfield overlap (measured)**: at n=1e6 dense-F64
  the shipped lifecycle beats the fully serialized run by only
  0.31–4.79% (computed as (t_serial − t_graph+overlap)/t_serial over the
  12 stage rows; min cube ell=7, max multiscale ell=6 — e.g. multiscale
  ell=5: 178.1 vs 185.4 ms; adaptive rows ~1–2%) — graph/overlap engagement is not a material factor at this
  scale (it was priced from ~50 us/window launch latency at much smaller
  windows in 027–029).
- **Per-n K/depth sweep**: landed as the fig15/fig18 grids (this row).
- **Stage-slab chunking**: deferred with price (host-only memory lever;
  needs src instrumentation + a prototype — a maintenance row, not a
  benchmark row). Host double-refresh + host sort unification likewise
  remain 042 audit items (device landed both in 041; host adaptive
  update still pays 2.1–2.9 s at n=1e6 vs 0.44–1.0 s uniform).

## 8. Regime recommendations (for the 042 audit)

- **Clustered / multi-scale density fields** (contrast >= ~10, or any
  field with a fat leaf-population tail): adaptive, K_max ~ 64 —
  1.7–3.0x faster and ~11x less device memory than the best uniform
  depth, gap widening with contrast.
- **Sigma-heterogeneous regularized vortex fields** (CoreSpreading):
  adaptive with the per-cell gate whenever the spread approaches the
  global gate's admissible-depth collapse (spread ~> 100 at these
  scales); below that the uniform path at its shallow admissible depth
  is faster.
- **Uniform fields (cube)**: parity within 4-5%; keep the uniform grid
  as default (it is also 2.4x lighter in memory at its best depth).
- **Uniformly sparse elongated fields (wake)**: the uniform grid at deep
  ell wins time (1.11–1.18x) where its 54 GB capacity fits; choose
  adaptive when memory is constrained, when n or density will grow
  (ell=8 cap), or when the field may develop density contrast. S2L is
  the top adaptive tuning target to close the wake gap.
- **Small n (<= 1e5)**: uniform grid everywhere.
- Production default: **no change recommended by this row** (benchmark
  row; the 041 decision that adaptive stays opt-in stands). The `042`
  review owns the default-selection audit against these figures.

## 9. Figure index

| figure | shows | data (job) |
|---|---|---|
| fig14_adaptive_contrast | step time vs contrast, blow-up vs flat | fm041a_gpu_contrast.csv (13184014) |
| fig15_adaptive_time_vs_n | best-vs-best time vs n, 3 cases, H200+host | fm041a_gpu_widen.csv (13184013), fm041a_host_widen.csv (13184015) |
| fig16_adaptive_stages | per-stage serialized breakdown + shipped ticks | fm041a_gpu_stages.csv (13184014) |
| fig17_adaptive_memory | device GB vs ell / K_max | fm041a_gpu_widen.csv (13184013) |
| fig18_adaptive_accuracy_cost | P + geometry frontier, 1e-3 gate rule | fm041a_pweep.csv + fm041a_gpu_widen.csv (13184013) |
| fig19_adaptive_leafpop | leaf-population CCDF fat tail vs K_max cutoff | fm041a_contrast_leafpop.csv (13184014) |
| fig20_adaptive_sigma | global-gate throws vs per-cell gate | fm041a_gpu_sigma.csv (13184014) |

All figures: `data/figures/figNN_*.tex` + same-named CSV directories,
regenerated by `scripts/figures_041a_prepare.jl`, compiled with pdflatex
(verified 2026-08-17). SHA-256 checksums of the CSVs of record are in
`data/adaptive_octree/checksums_041a.sha256`.
