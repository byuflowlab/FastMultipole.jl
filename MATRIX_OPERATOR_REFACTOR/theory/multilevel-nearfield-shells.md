# Multilevel nearfield shells (041c)

## Decision

**NO-GO.** Spatial refinement can expose a large analytic fraction in the
cube and multiscale cases, so neither the regularization floor nor the ideal
promotion fraction kills the idea. The full calibrated crossover does: the
newly admissible rectangles contain too few body pairs per expansion route.
Even before virtual P2M, route generation, launches, refresh, metadata, new
operator tables, or a serial tail are charged, every tested aggregate costs
more than leaving its residual U pairs direct. The conservative automatic
selector therefore always takes the existing direct fallback and has zero
regression and zero oracle gap. No production successor is recommended.

This is distinct from 037e. That row tested skipping expensive mixed-branch
candidacy and whole-leaf rerouting. Here the queue continues *inside* terminal
adaptive U rectangles, applies harmonic admissibility again to occupied
virtual Morton subnodes, and prices M2L/M2T/S2L routes already present in the
adaptive lifecycle.

## 1. Bound composition

No new regularization or constant-order bound is introduced. For source
subnode S and target set T, use the existing source-directed condition

```
gap(T,S) >= rho_t * sigma_max(S),    rho_t = 4.211 (U), 4.789 (J),
```

where `gap` is the integer-exact mixed-level AABB gap and
`sigma_min(S)`, `sigma_max(S)`, and diagnostic quantiles are reduced from
`body_sigma[perm[lo:hi]]`. Equality remains direct. Section 4 of
`kernel-splitting-nearfield.md` supplies

```
E_U = g_bar(rho),
E_J = g_bar(rho) + rho*g'(rho)/2,
E/(1-E) <= epsilon, provided E <= epsilon/2,
```

so the J radius is binding when U and all nine J entries are delivered. This
is the standing `epsilon=1e-3` budget; 041c does not relax it.

The harmonic half of the test is the task-038/task-008d constant-P remainder.
For an equal-level M2L offset `o`, source strength envelope A and cell half
width h,

```
c = 2*norm(o)/sqrt(3),  rho = h*sqrt(3),
B_P = 2A/[rho(c-2)] * [1/(c-1)]^(P+1),   c > 2.
```

`c <= 2` is rejected. Lamb--Helmholtz uses the existing project composition
`B_phi + (1+2R)B_chi`, with chi carried at `P+1`, production normalization,
and the existing velocity/J budget allocation. For unequal nodes the same
geometric-series proof is evaluated with the appropriate retained radius:

| route | retained truncation radius | omitted side |
|---|---:|---|
| M2L | `r_S+r_T` | neither |
| M2T | `r_S` | target (actual targets are evaluated) |
| S2L | `r_T` | source (actual sources are accumulated) |

Writing `q=r_keep/R`, the dimensionless remainder `q^(P+1)/(1-q)` must fit
the bound's half budget and requires `q<1`. S2L additionally requires every
source in S to lie outside the target local convergence ball. M2T has no
target-side truncation; S2L has no source-side truncation. A route is emitted
only when *both* this test and the J regularization test pass.

The sigma floor is fundamental: refining space can improve the truncation
ratio and may lower a heterogeneous subnode's `sigma_max`, but it cannot make
the regularized field harmonic inside approximately `rho_t*sigma`. Solid
harmonics cannot cross this floor. Gaussian/Hermite fast-Gauss expansions are
the only relevant escape and belong to the funded 037d/VIC direction, not
this row.

## 2. Queue invariant and lineage legality

Let C be the Cartesian product of ordered body IDs owned by one original U
pair. At every traversal step maintain disjoint sets Q (queued rectangles),
H (emitted harmonic rectangles), and D (terminal-direct rectangles), with

```
C = disjoint_union(Q, H, D).
```

Initially Q contains C and H,D are empty. Splitting one node replaces its
body range by the disjoint ranges of its occupied Morton children. Splitting
one or both sides therefore replaces `A x B` by the disjoint Cartesian
product of those child ranges; union and disjointness are preserved. Moving
one queued rectangle to exactly one M2L, M2T, S2L, or direct bucket also
preserves the invariant. Induction over the finite depth bound proves
exact-once coverage. A self rectangle cannot pass a positive-gap or
convergence test and descends until it is irreducibly direct.

Ordinary geometrically-near lineage may consider M2L, M2T, and S2L. A
sigma-demoted lineage may consider M2T, S2L, target filtering, or direct, but
not M2L: sticky demotion protects the bounded task-025 table set. An ordinary
virtual M2L is legal mathematically, although its new offset class would
require priced operator-table capacity. A partial multipole is allowed only
for a reusable spatial node or a predeclared logarithmic sigma class, never a
target-dependent source subset.

The census expresses its final partition through the imported
`Lists(U,V,W,X)` representation and calls the existing
`adaptive_octree_verify.jl:check_exact_once` painter. All 56 compact
case/P/depth rows have zero bad pairs, including explicit dyadic-boundary,
six-decade-sigma, and coincident-position cases. The independent production painter is
`test/adaptive_octree_test.jl:_adt_exact_once_bad`; its adaptive-tree suite is
the production-path cross-check. The adversarial coverage includes the
imported uniform, filament, multiscale, and boundary-sensitive Morton
constructors; empty children are absent by construction, one-sided splits
explicitly retain the unsplit counterpart, and ordered painting covers the
same geometry whether either system is static or both systems coincide.

## 3. Routing and crossover

At ideal cost the deterministic priority is M2L, then S2L, then M2T. This
reflects two reuse facts: M2L reuses both clusters; S2L deposits into a local
expansion evaluated by the already-required L2B pass; M2T pays per target
body. The priority never overrides convergence. Phase 2 replaces this ideal
priority with measured crossovers.

From the same-case H200 `fm041a_gpu_stages.csv` stages divided by
`fm041a_gpu_widen.csv` route/body-pair counts, optimistic P4 Float64 costs
are:

| case | direct ns/body pair | M2L ns/route | M2T ns/route | S2L ns/route |
|---|---:|---:|---:|---:|
| cube | 0.006720 | 0.4519 | 79.65 | 560.47 |
| wake | 0.009813 | 0.4363 | 44.62 | 137.18 |
| multiscale | 0.008641 | 0.5812 | 42.88 | 150.09 |

Thus even the favorable multiscale crossovers require about 67 direct pairs
per M2L route, 4,963 per M2T route, or 17,369 per S2L route. The observed
virtual routes average O(1--50) promoted body pairs. Depth-three cube P4, for
example, exposes 1,095,342 pairs but needs 399,648 M2T routes; multiscale P8
exposes 542,602 pairs through 146,194 total routes. The tiny-group rule
therefore returns these rectangles to direct.

These rates are deliberately favorable to the proposal: they are existing
batched production rates and omit the approximately 50 microsecond launch
latency for incompatible fragments, virtual P2M, sigma reductions, route
generation, scans/compaction, graph capture, metadata, table storage, and
the per-step refresh. P8 and Float32 cannot rescue the crossover: P8
increases expansion work, while Float32 makes direct work faster as well.

## 4. Complete critical-path objective and selector

For aggregate configuration z, the required objective is

```
T(z) = critical_path(existing stages with saved direct work,
                     batched M2L/M2T/S2L work,
                     route/launch latency,
                     virtual P2M and sigma reductions,
                     scan/compact and route generation,
                     graph/table/metadata refresh,
                     introduced serial tail).
```

The implemented lower bound drops every nonnegative overhead after analytic
route execution:

```
T_lower(z) = residual_pairs*c_direct
           + n_M2L*c_M2L + n_M2T*c_M2T + n_S2L*c_S2L.
```

If `T_lower(z) >= original_pairs*c_direct`, no batching, overlap, allocation,
or refresh policy can make z win. This inequality holds for every census
aggregate. Consequently the exhaustive two-choice oracle (`promoted
aggregate` versus `direct`) chooses direct everywhere. The auditable selector
promotes only when the predicted win exceeds the recorded 25% interpolation
uncertainty (35% for sigma-heterogeneous extrapolation); it too chooses direct
everywhere. Its oracle gap and complete-solve delta are zero. The <=3%
regression cap is structural because fallback executes the shipped path.

Since this lower-bound kill precedes capacity allocation, the bounded
preallocation, zero-recurring-allocation, and graph-compatible layouts are
recorded but not designed into production. Virtual metadata would be 64
bytes/node in Float64 (48 in Float32), plus any new M2L operator tables; all
would refresh per VPM rebuild rather than amortize over a stable epoch unless
an occupancy epoch proves otherwise.

## 5. Phase gates

**Phase 0 -- CONTINUE.** The prior 037e census is the only existing aggregate
with source-directed sigma geometry. Its whole-leaf pure-singular M2L ceiling
is zero for cube/wake but 0.587 and 0.627 for rotor at 1e5/1e6. The 0.627 row
exceeds the 20% material-case gate. Because those aggregates lack subnode
geometry, zero cannot kill virtual refinement; Phase 1 was required.

**Phase 1 -- CONTINUE, regime scoped.** At ideal zero route cost, maximum
promoted fractions are 0.262 cube, 0.000506 wake, 0.559 multiscale, and 0.415
sigma-multiscale. Multiplying by the measured 0.86 mixed-bucket time share
leaves cube/multiscale above the 10% nearfield gate. Wake is killed. Sigma
quantile blocking is only a small share of rejections, so the `{1,2,4,8}`
sigma-class sweep is not admitted.

**Phase 2 -- KILL.** Every promoted aggregate loses to direct under the
optimistic analytic-only lower bound. Adding required launch, refresh,
metadata, capacity and critical-path terms only worsens it. The bounded
configuration oracle and conservative selector both select direct for all
P4/P8 and Float32/Float64 rows. Final verdict: **NO-GO due to route explosion
and expansion cost after useful spatial refinement**. Close deeper harmonic
nearfield refinement; do not reopen the adaptive tree and do not stage a
post-042 production row.

## 6. Reproduction

Run with at most four local threads:

```
JULIA_NUM_THREADS=4 julia --project=. \
  MATRIX_OPERATOR_REFACTOR/scripts/multilevel_nearfield_shell_census.jl
```

Compact outputs, provenance, calibration, oracle results, selector decisions,
capacity estimates, report, and SHA-256 checksums are under
`data/multilevel_nearfield_shells/`. No particle-scale pair list is written.
