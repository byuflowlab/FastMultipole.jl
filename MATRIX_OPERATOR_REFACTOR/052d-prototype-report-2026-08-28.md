# 052d prototype report — shared-radix dual-tree list generation (2026-08-28)

Host-side prototype validating the LIST-GENERATION machinery of the Phase
2b-revised design ("shared-radix dual-tree device FMM", 2026-08-27 section of
`052d-plan-2026-08-26.md`). Code:
`MATRIX_OPERATOR_REFACTOR/prototypes/052d_shared_radix/` (`SharedRadix.jl`
module, `validate.jl`, `check_production.jl`, `production_run.jl`, `README.md`,
run logs). Plain Julia, Base + StaticArrays; FastMultipole/FLOWPanel source
untouched.

## What was implemented

1. **Implied global grid.** One root box (center + half-width from the union
   bounding box of sources and targets); level-$l$ cell half-width
   $h_l = h_0/2^l$. Cells are identified by (level, Morton code): 21 bits per
   axis interleaved into a 63-bit UInt64. `cell_center(level, code)` and
   `cell_halfwidth(level)` are pure index arithmetic — no stored cell
   geometry anywhere.
2. **Sparse adaptive trees on the shared grid.** Morton-sort the points, then
   BFS-split every cell holding more than `leaf_size` points (empty children
   skipped; splitting caps at level 21, so coincident points force an
   oversized leaf there). Each cell keeps a contiguous sorted-order
   `UnitRange`, which by construction equals its full subtree's points. Two
   trees built for different point sets on the same `Grid` share the
   identical implied lattice automatically.
3. **Cross-tree adaptive dual traversal** producing the M2L list and the
   near-field leaf-pair list from (level, code) arithmetic only. MAC:
   Barba-style $r_S + r_T \le \theta\, d$ with $r = \sqrt{3}\,h_l$ of the raw
   grid box (no shrinking, per the design) and $d$ the center distance.

   **Level-heterogeneity rule (as implemented):** when a candidate pair
   fails the MAC and is not leaf–leaf, descend the cell with the LARGER grid
   half-width, i.e. the one at the shallower level (ties descend the source;
   if the larger cell is a leaf, descend the other side). Because every
   descent replaces one side of the pair by the disjoint cover of its
   children, the recursion partitions $\text{points}(S_\text{root}) \times
   \text{points}(T_\text{root})$ exactly, even when the panel cloud's cells
   sit many levels deeper than the particle cells they interact with —
   accepted M2L pairs are allowed at different levels. The shared root
   (both trees start at level 0, code 0, distance 0) fails the MAC and
   descends naturally.

**Subtlety found and resolved — exact MAC boundary ties.** On a shared grid
all center distances are quantized, so $r_S + r_T = \theta\,d$ can hold
*exactly in real arithmetic* (e.g. a $\Delta\text{level}=3$ pair at diagonal
offset 15 fine cells with $\theta = 0.6$: $0.6 \cdot 15/128 = 9/128$
exactly). Two floating-point groupings of the same test then land on opposite
sides of a strict `<` (first seen as 22 spurious "violations" at
leaf 256/$\theta$ 0.6, every one with excess exactly 0.0). Resolution: the
MAC is defined non-strict ($\le$), and validation evaluates it in EXACT
integer arithmetic (`exact_mac_leq`: with integer center offsets $\Delta$ in
fine-half-width units and $s = 2^{L-l_S} + 2^{L-l_T}$, the test is
$3\,q^2 s^2 \le p^2 |\Delta|^2$ for rational $\theta = p/q$). A tie
classified either way never breaks the partition and still delivers the
$\theta$ error bound. The device port should adopt an explicit tie
convention; this prototype documents FP-`<` acceptance + exact-$\le$
validation.

## Validation results

### Small-configuration suite (`validate.jl` / `validate.log`) — all PASS

Per case: exhaustive pair-partition (every (source point, target point) pair
covered EXACTLY once, via M2L subtree ranges or near-field leaf pairs),
exact-integer MAC on every accepted M2L pair, and determinism of the lists
(as grid-cell-identity sets) under random permutation of both point sets.
Default leaf_size 32, $\theta = 0.5$ unless noted.

| case | n_src / n_tgt | M2L | near pairs | miss | dup | MAC bad | determinism |
|---|---|---|---|---|---|---|---|
| random cluster-in-wide, seeds 1–4 | 2000 / 3000 | 2837–2956 | 5343–7164 | 0 | 0 | 0 | ok |
| all-in-one-cell (exactly coincident points) | 200 / 300 | 0 | 1 | 0 | 0 | 0 | ok |
| all-in-one-cell (1e-9 ball) | 200 / 300 | 54 | 681 | 0 | 0 | 0 | ok |
| collinear, same line | 500 / 700 | 157 | 177 | 0 | 0 | 0 | ok |
| collinear, disjoint segments | 500 / 700 | 2 | 0 | 0 | 0 | 0 | ok |
| coincident boxes (same distribution) | 1500 / 1500 | 1463 | 4257 | 0 | 0 | 0 | ok |
| identical point sets | 1500 / 1500 | 1764 | 4831 | 0 | 0 | 0 | ok |
| deep-level source cluster (1e-3 extent) | 2000 / 3000 | 226 | 3500 | 0 | 0 | 0 | ok |
| leaf_size ∈ {8, 64} × θ ∈ {0.4, 0.6} | 2000 / 3000 | 299–12451 | 1148–28405 | 0 | 0 | 0 | ok |

**15/15 cases: zero misses, zero double-counts, zero exact-MAC violations,
fully deterministic lists.**

### Production-scale checks (`check_production.jl` / `check_production.log`) — all PASS

Real step-472 geometry (below); the full $8.9\times10^9$ coverage matrix is
infeasible, so: (i) exact covered-pair count identity
$\sum_{\text{pairs}} |S|\,|T| = n_s n_t$; (ii) exact per-source coverage
vector for 400 random targets (all entries exactly 1); (iii) exact-integer
MAC over the entire M2L list.

| leaf / θ | covered pairs | identity | MAC bad | bad sampled targets |
|---|---|---|---|---|
| 32 / 0.5 | 8,893,469,472 | exact | 0 | 0/400 |
| 128 / 0.4 | 8,893,469,472 | exact | 0 | 0/400 |
| 256 / 0.6 | 8,893,469,472 | exact | 0 | 0/400 |

## Production-shape run + cost model (`production_run.jl` / `production.log`)

**Geometry: REAL step-472 snapshot** (the sibling extraction landed;
`SNAPSHOT_INDEX.md` confirms the formats used). Sources = **36,752** panel
centroids (the plan's "~9k panels" was approximate — the production body is
36,752 triangles, matching `dji9443_..._captess4.msh`); targets = **241,986**
particles. Dense pair total $8.893\times10^9$; the measured 3.3 s dense A100
leg ran on this same body, so the near-field fraction scales that number
consistently. 4 threads; times are wall clock on the host (M-series laptop).

Cost model: $t_\text{near} = \text{frac} \times 3.3\,\mathrm{s}$;
$t_\text{M2L} = n_\text{M2L} \times (2p+1)^3 / 10^{12}$ at $p = 4$
(729 flops/pair; assumed effective device throughput $10^{12}$ flop/s
$\approx$ 5% of A100 FP32 peak — conservative for small memory-bound
translation batches). Gate: 0.6 s/step.

| leaf | θ | t_tree (s) | t_trav (s) | n_M2L | n_near | near pair-int | frac dense | t_near (s) | t_M2L (s) | modeled total (s) | gate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 32 | 0.4 | 0.020 | 0.011 | 107,527 | 116,007 | 17,185,008 | 0.00193 | 0.0064 | 0.0001 | 0.0065 | PASS |
| 32 | 0.5 | 0.017 | 0.006 | 62,631 | 65,240 | 9,847,891 | 0.00111 | 0.0037 | 0.0000 | 0.0037 | PASS |
| 32 | 0.6 | 0.019 | 0.002 | 35,181 | 36,808 | 5,341,651 | 0.00060 | 0.0020 | 0.0000 | 0.0020 | PASS |
| 64 | 0.4 | 0.013 | 0.003 | 67,821 | 67,365 | 34,182,449 | 0.00384 | 0.0127 | 0.0000 | 0.0127 | PASS |
| 64 | 0.5 | 0.014 | 0.002 | 40,927 | 37,803 | 18,267,466 | 0.00205 | 0.0068 | 0.0000 | 0.0068 | PASS |
| 64 | 0.6 | 0.015 | 0.002 | 23,848 | 21,432 | 9,980,526 | 0.00112 | 0.0037 | 0.0000 | 0.0037 | PASS |
| 128 | 0.4 | 0.049 | 0.002 | 44,993 | 40,891 | 68,993,716 | 0.00776 | 0.0256 | 0.0000 | 0.0256 | PASS |
| 128 | 0.5 | 0.016 | 0.001 | 28,621 | 22,180 | 35,968,566 | 0.00404 | 0.0133 | 0.0000 | 0.0134 | PASS |
| 128 | 0.6 | 0.021 | 0.002 | 16,805 | 12,241 | 19,072,555 | 0.00214 | 0.0071 | 0.0000 | 0.0071 | PASS |
| 256 | 0.4 | 0.049 | 0.001 | 28,772 | 22,999 | 148,670,263 | 0.01672 | 0.0552 | 0.0000 | 0.0552 | PASS |
| 256 | 0.5 | 0.019 | 0.001 | 19,464 | 12,014 | 71,449,193 | 0.00803 | 0.0265 | 0.0000 | 0.0265 | PASS |
| 256 | 0.6 | 0.016 | 0.000 | 11,882 | 6,491 | 38,507,201 | 0.00433 | 0.0143 | 0.0000 | 0.0143 | PASS |

Best modeled total 0.0020 s (leaf 32, θ 0.6) — **299× under the gate**;
worst 0.0552 s (leaf 256, θ 0.4) — **11× under**. The plan's preferred
θ = 0.5 band sits at 0.004–0.027 s (22–162× margin).

**Translation-class count** (supports design consequence 2, cacheable
rotation-trick operators): distinct $(\Delta\text{level}, \text{offset})$
classes among accepted M2L pairs range 2,342 (leaf 256, θ 0.6) to 16,793
(leaf 32, θ 0.4) — e.g. 11,036 at leaf 32 / θ 0.5, i.e. ~5.7 pairs per class
there; a modest, per-step-cacheable operator set.

**Model robustness (checked before concluding):**

- The modeled M2L flops are negligible; even charging a pessimistic 1 µs of
  per-pair launch/gather overhead adds ≤ 0.11 s at the largest list (107k
  pairs, leaf 32/θ 0.4) — every configuration still clears the gate.
- Costs NOT in the model: panel B2M (host, 36,752 panels), the separate
  downward pass (plan estimate: one L2L, negligible, + U-only L2B at 242k
  targets ≈ few–tens of ms on device), coefficient H2D/D2H, and list upload.
  All are tens of ms at most; adding them cannot approach 0.6 s.
- Host list generation itself (tree build + traversal, 4 threads) is
  0.015–0.05 s per step — cheap enough to stay host-side in the device port
  if convenient.
- The near-field scaling assumes the dense 3.3 s kernel's cost is
  proportional to pair count when restricted to leaf-pair blocks; block
  irregularity will cost some device efficiency, but even a 5× penalty on
  the worst configuration stays under the gate (0.28 s), and a 5× penalty at
  leaf 32–64 / θ 0.5 is ~0.02–0.03 s.

## Conclusions

1. **Correctness: yes — the shared-grid index-arithmetic approach produces
   provably-correct lists.** Across 15 adversarial small configurations
   (exhaustive check) and the real 36,752 × 241,986 production shape
   (count-identity + sampled-exact checks), the M2L + near-field lists form
   an exact partition of all source–target pairs (zero misses, zero
   double-counts), every accepted pair satisfies the MAC in exact integer
   arithmetic, and the lists are deterministic under input permutation. The
   cross-level descend-the-larger rule handles the deep-level panel cloud
   embedded in the wide particle cloud with no special cases.
2. **Cost: the modeled device total clears the 0.6 s/step gate with large
   margin** — 0.002–0.055 s across the whole leaf_size × θ sweep (11×–299×
   margin; 22×–162× in the θ = 0.5 band), vs 15.8 s for the failed host-FMM
   route and 3.3 s for the dense leg it replaces. The near-field fraction of
   dense work is 0.06 %–1.7 %, so the conclusion survives an order of
   magnitude of model pessimism; the unmodeled B2M/downward-pass/transfer
   legs are tens of ms.
3. **Design notes for the device port:** adopt an explicit MAC tie
   convention (grid quantization makes exact $r_S + r_T = \theta d$ ties
   real, not hypothetical); the quantized translation-offset set is
   confirmed modest (2.3k–17k classes), validating the cached-operator plan;
   no other design flaws surfaced.

Recommended next step per the plan's sequencing: proceed to the device port
(step 3), with leaf_size 32–64 and θ = 0.5 as the starting operating point.
