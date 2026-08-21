# Smooth-basis nearfield alternatives after the 041c NO-GO (041d)

## Decision summary

**Part 1 (regularized-basis P2M/M2P substitution): NO-GO for the tested
census family under the registered proxy cost model.** The registered census
(`scripts/smooth_nearfield_prekill_census.jl`,
`data/smooth_nearfield_prekill/`) covers both directions of cluster
granularity: *subdivision* of terminal U-list source leaves (virtual depths
0–3, clusters ≤ 32 sources) and — added after external review —
*coarsening/merging* of each target's U source leaves into ancestor-grouped
super-clusters up to the all-sources union (clusters up to 2047 sources).
Subdivision finds zero economical promotions under the conservative policy on
every case and depth, and only a 0.11% promoted fraction on
`sigma_multiscale` under the deliberately optimistic Hermite policy. Merged
clusters *can* clear per-route break-even (best economic ratio 4.36x under
the optimistic policy, 1.82x conservative), but the aggregate net of
formation cost peaks at a 2.3% saving under the optimistic policy and is
negative under the conservative policy; the selector with the registered
25–35% uncertainty margins chooses direct fallback on all 64 rows. The
verdict is scoped: it holds for the tested Cartesian/Hermite proxy cost
model, whose evaluation rate is the measured harmonic M2T kernel rescaled by
coefficient count — a proxy, not a lower bound for a purpose-built
Hermite/interpolation/proxy-point/separable M2P kernel (Section 2.5 names
the residual follow-up that would be required before claiming universal
closure of every linear basis).

**Part 2 (sigma-adaptive multilevel smooth representation for the
sigma-heterogeneous regime): OPEN — recommend a derivation row.** The 037d
rotor kill applies to *global* (and sigma-binned global) meshes, whose point
count carries the domain-wide `sigma_min` floor. Under a disjoint
particle-owned-volume hypothesis, a mesh whose resolution tracks the local
`sigma(x)` has a point count proportional to the particle count and
independent of the sigma spread (Section 3) — a **heuristic hypothesis, not
a derived point-count law** for rotor blade/wake geometry — and the priced
band shows no sign flip against the shipped rotor nearfield. This is the one
nearfield direction not yet closed by 037b/037d/037e/041b/041c evidence;
041f is staged to replace the hypothesis with concrete level-by-level counts
on the actual rotor snapshot.

Context: the nearfield is 66–95% of the shipped U/J solve and is the critical
path (037b); full VIC is already funded for the near-uniform-sigma regime
(037d), so the open problem is specifically sigma-heterogeneous fields.

## 1. Why the 041b rank measurements do not decide this question

The 041b QDEIM/SVD probe operated on `ell = 6` rotor cells with
`sigma/h ≈ 9.2e-4`–`1.26e-3` (`data/strategic_target_feasibility/`): every
probed pair is hundreds to thousands of sigma apart, the regularization is
numerically inert (ranks bit-identical across the 1/6/18 sigma bins), and the
measured operator is effectively the singular kernel at cell granularity.
The SVD-optimality closure recorded there therefore bounds cell-scale
singular blocks, not sigma-scale smooth blocks. Part 1 closes the smooth-block
question on economic rather than rank grounds.

## 2. Part 1: the two-sided squeeze on P2M/M2P in a regularized basis

### 2.1 Registered admissibility model

The `gaussianerf` kernel is entire; its Taylor coefficients at distance $d$
from a source cluster decay on the scale $\max(d, \sigma)$. The census
therefore models the order-$p$ truncation error of a cluster of radius $r_S$
as

$$
E_p = \frac{q^{p+1}}{1-q}, \qquad q = \frac{r_S}{\max(d_{\min},\, \sigma_{\min}(S))},
$$

(the "sigma-regularized denominator") with a second, deliberately optimistic
policy adding the fast-Gauss Hermite gain $E_p = q^{p+1}/\sqrt{(p+1)!}$.
Both use the standing half budget $5\times 10^{-4}$. The regularized
denominator is exactly what makes the idea attractive: clusters with
$r_S < \sigma$ are expandable even *inside* the singular floor
$d < \rho_t \sigma$ where solid harmonics are inadmissible (041c §1).

### 2.2 Registered cost model and the break-even law

From the verified 041c calibration
(`data/multilevel_nearfield_shells/cost_calibration.csv`), the batched
production per-target expansion evaluation costs $c_1 = m2t/64$ ns and a
direct pair costs $c_d$; a Cartesian total-degree basis has
$n_c(p) = (p{+}1)(p{+}2)(p{+}3)/6$ coefficients against the 25 harmonic
coefficients anchoring $c_1$. One M2P evaluation replaces $|S|$ direct pairs
per target, so promotion requires

$$
|S| \;>\; n_{\text{break}}(p) = \frac{c_1}{c_d}\cdot\frac{n_c(p)}{25}.
$$

Measured values (`data/smooth_nearfield_prekill/break_even.csv`): 28–74
sources at $p=2$, 100–259 at $p=4$, 239–622 at $p=6$, 469–1222 at $p=8$.

**Scope caveat (registered).** $c_1$ is a *proxy*: the measured harmonic M2T
production rate rescaled linearly by coefficient count. It anchors the model
to real hardware but is **not a lower bound** for a purpose-built Hermite,
polynomial-interpolation, proxy-point, or tensor/separable M2P kernel, which
could have a different constant. The best $p=2$ crossover (28 sources) sits
below the 32-body leaf cap, so constant-factor rate changes genuinely matter
near the boundary — the verdict below therefore rests on the *measured census
margins*, not on an asserted immunity to constant factors. Formation is
charged once per promoted cluster at that cluster's own maximum used order.

### 2.3 The squeeze, and the subdivision census

Admissibility and economics pull the cluster size in opposite directions:

- **Economics** needs $|S| \gtrsim 28$–$74$ even at $p=2$, but $p=2$
  admissibility requires $q \lesssim 0.076$ (0.107 with the Hermite gain),
  i.e. separations 9–13x the cluster radius — geometry that no residual
  U-list pair possesses (those pairs are near by construction).
- **Orders that converge at near separations** ($q \sim 0.5$ needs
  $p \gtrsim 10$; even $q \sim 0.35$ needs $p \approx 8$) demand
  $|S| \gtrsim 470$–$1200$. Heuristic sizing at fixed overlap $\beta = 2$
  (a cluster of $|S|$ sources filling a ball of per-particle spacing
  $\delta = \sigma/2$ has $r_S \approx \delta\,|S|^{1/3}$) suggests
  admissibility inside the floor ($d \lesssim 3.7\sigma = 7.4\delta$) caps
  $|S|$ near 7 at $p=4$ while break-even demands $|S| \gtrsim 100$ — a
  $\gtrsim$15x gap *under this disjoint-packing model*. The model is a
  motivation, not a proof (real clusters have anisotropic geometry and
  sigma spread), so the decision rests on the census, including the merged
  clusters of Section 2.4 for which $|S|$ is unbounded.

Subdivision census (`data/smooth_nearfield_prekill/m2p_census.csv`): with
041c-identical cases, seeds, trees, and virtual depths 0–3, the conservative
policy promotes **zero** body pairs anywhere; the optimistic Hermite policy
promotes only 3,168 of 2,856,358 pairs (0.11%) on `sigma_multiscale` (2,496
of them genuinely inside the singular floor — the regime exists but is
negligible), and the aggregate selector with the 041c uncertainty margins
chooses direct fallback on all 32 rows. Changing the physics kernel to one
with closed-form regularized expansions (e.g. algebraic
$1/(r^2+c^2)^{3/2}$) is locked out by the `CoreSpreading` compatibility
decision (2026-08-05), and as an intermediate approximation basis it cannot
beat the entire-function convergence already granted above.

### 2.4 Coarsened/merged clusters (added after external review)

The subdivision census alone cannot bound clusters *larger* than one leaf: a
regularized basis might aggregate several neighboring U-list source leaves
for a common target, particularly among large-sigma sources. Two additions
close this gap:

**Monotonicity lemma (rigorous).** For a union $S \supseteq L$ of source
leaves serving a fixed target, $r_S \ge r_L$, $d_{\min}(S) \le d_{\min}(L)$,
and $\sigma_{\min}(S) \le \sigma_{\min}(L)$, so
$q(S) \ge q(L^\*)$ where $L^\*$ is the largest-radius member: merging never
improves admissibility, while it does improve per-route economics
($|S|$ grows). Whether economics or admissibility wins is therefore
quantitative — hence the census.

**Coarsening census**
(`data/smooth_nearfield_prekill/coarsened_census.csv`): for every U-list
target leaf, its source leaves are merged by common ancestor at 1–3
coarsening levels plus the extreme all-sources union (candidate clusters up
to 2047 sources, well beyond $K_{\max}=32$; union geometry measured from the
actual member bodies, not the ancestor cell). Result: merged clusters **do**
clear per-route break-even where the subdivision census could not — best
economic ratio $|S|/n_{\text{break}} = 4.36$ (`sigma_multiscale`, optimistic
Hermite policy) and 1.82 (conservative) — and every promoted pair lies
inside the singular floor, confirming the targeted regime exists. But the
aggregate net of formation cost peaks at a **2.3% saving** of the residual
U-list direct time under the optimistic policy (1.2% on `wake`), and is
**negative** under the conservative policy (formation exceeds the marginal
eval saving); the selector with the registered 25–35% margins chooses direct
fallback on all 32 coarsened rows. The all-union extreme is always
inadmissible ($q \ge 1.33$).

**Conclusion (scoped).** Under the registered proxy cost model, no tested
promotion strategy — leaf, subdivided, or merged/coarsened clusters — beats
the measured 0.0067–0.0098 ns/pair direct rate by more than a 2.3%
ideal-model upside, far inside the 25–35% uncertainty margins and below the
standing 5% lever threshold for staging optimization work. This is a NO-GO
for the census family tested; it does **not** claim closure of every linear
basis or of a purpose-built M2P kernel with a materially better constant
(Section 2.5).

### 2.5 Residual before any universal closure claim

If this direction is ever reopened, the remaining evidence gap is a
**rank/DOF-to-cost lower-bound study on the actual regularized U+J blocks**:
measure the numerical ranks of true `gaussianerf` U/J interaction blocks at
sigma-scale separations (the 041b probe was regularization-inert at
`sigma/h ~ 1e-3` and does not cover this regime) and price the best possible
linear evaluation at those ranks against the direct rate. Only that would
convert the present model-scoped NO-GO into a basis-universal one. It is not
staged now: the census bounds the ideal-model upside at ~2.3% on the tested
cases, below the 5% staging threshold.

## 3. Part 2: sigma-adaptive multilevel smooth representation (cost estimate)

### 3.1 The structural observation

037d killed full VIC on the rotor because a *global* mesh must resolve the
domain-wide $\sigma_{\min}$ (5e11–1.3e15 points; sigma-binned global meshes
inherit the same floor because each bin still spans its class's bounding
box). But the mesh-point count of a locally refined mesh whose width tracks
$h(x) = 0.55\,\sigma(x)$ (the 037d-validated $p=4$ B-spline resolution) is

$$
N_{\text{mesh}} \;=\; \int_{\text{occ}} \frac{dV}{h(x)^3}
\;\approx\; \sum_i \frac{\delta_i^3}{(0.55\,\sigma_i)^3}
\;=\; n\,\Big(\frac{1}{0.55\,\beta}\Big)^{3} \;\approx\; 0.75\,n
$$

at overlap $\beta = \sigma/\delta = 2$, nominally independent of the sigma
spread: each particle's neighborhood is meshed at its own sigma.

**Status: heuristic hypothesis, not a derived law.** The middle step treats
$\delta_i^3$ as a disjoint particle-owned volume. That has not been shown
for rotor blades and wakes, whose occupied geometry is sheet- and
filament-like with overlapping support, and whose AMR realization adds
empty-space patch fill and 2:1 transition regions that the sum does not
count. The "independent of sigma spread" property and every number derived
from $N_{\text{mesh}} \approx 0.75\,n$ below (including the priced band)
inherit this status; 041f exists precisely to replace it with measured
level-by-level counts on the actual rotor snapshot. Taken at face value, for
the rotor at $n = 10^6$ the hypothesis gives ~0.75M points before AMR
overhead (guard cells, level halos, padding: x2–4), i.e. **1.5–3M points
versus 5e11+ for the global mesh** — inside the 0.9–26M range 037d already
priced at 2.76–18.9 ms F32 for cube $10^6$.

### 3.2 Solver structure (no global FFT)

Two established families fit the device-resident contracts:

1. **Per-level patch AMR-FFT**: 2:1-balanced level patches (the adaptive
   octree machinery already provides construction, occupancy, and 2:1
   balance), Hockney free-space convolution per patch at its own resolution,
   multilevel Gaussian split for sigma heterogeneity — particle $i$ is
   assigned the finest level $\ell$ with $\sigma_\ell \le \sigma_i$ and
   spread with width $\sqrt{\sigma_i^2 - \sigma_\ell^2}$ carried by the
   level Green's function (an exact Ewald-like telescoping, not an
   approximation), plus B-spline restriction/prolongation between levels.
2. **Multilevel summation (MSM, Hardy–Skeel)**: kernel split across levels
   with local stencil convolutions only; avoids FFTs entirely at the price
   of larger stencils.

U and J follow the 037d 6-transform / analytic-B-spline-derivative scheme
unchanged.

### 3.3 Priced band (registered, paper-only)

Ops model at the 037d registered rates, rotor $n = 10^6$, F32:

- spread + interpolate: $2\,n\,(p{+}1)^3 \approx 2.5\times10^8$ ops;
- patch FFTs / stencils on 1.5–3M points across levels, 6 transforms;
- AMR coupling overhead (halos, transfers, small-FFT efficiency loss):
  x2 (optimistic) to x5 (pessimistic) over the uniform-mesh equivalent.

Against the 037d cube-$10^6$ measured-rate anchors this gives roughly
**3–8 ms nominal (opt ~2 ms, pess ~15–20 ms)** versus the shipped rotor
$10^6$ eval of 33.3 ms (037b partitioned best; nearfield ~70% of it):
**no sign flip anywhere in the band**, mirroring the shape of the 037d
uniform-sigma verdict. The pessimistic end still clears ~1.7x.

### 3.4 Risks the derivation row must retire

1. AMR patch management and level coupling on GPU under the
   capacity/zero-allocation/graph-capture contracts (largest unknown; the
   x2–5 band is a placeholder, not a measurement).
2. Refresh under `CoreSpreading`: $\sigma(x)$ grows each step, so level
   assignment drifts; needs an occupancy-epoch policy like the existing
   tree refresh.
3. Accuracy composition across levels (the exact Gaussian telescoping
   removes the splitting error, but interpolation error stacks per level).
4. The sparse-fill advantage assumed here is exactly the rotor geometry;
   a dense small-sigma region would push toward the global-mesh regime.

### 3.5 Recommendation

Do not stage an implementation row from this estimate. If the direction is
to be pursued, stage a 037d-style **derivation row** (theory/scripts/data
only; staged as `041f`) that replaces the x2–5 AMR-overhead placeholder with a concrete
level-by-level count on the actual rotor snapshot (particle-to-level
histogram, patch counts, halo volumes, transform sizes) and renders a
fund/close verdict against the then-current shipped rotor anchor. The
funded uniform-sigma VIC row (037d) proceeds independently and remains the
largest known nearfield lever (modeled 3.0–5.4x floor).

## 4. Reproduction

```
JULIA_NUM_THREADS=4 julia --project=. \
  MATRIX_OPERATOR_REFACTOR/scripts/smooth_nearfield_prekill_census.jl
```

Deterministic, 041c-identical seeds/cases/trees; compact checksummed outputs
under `data/smooth_nearfield_prekill/` (`m2p_census.csv`,
`coarsened_census.csv`, `break_even.csv`, `manifest.csv`, `report.txt`,
`checksums.sha256`). No particle-scale field is evaluated and no production
code is touched.
