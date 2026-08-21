# 041c Theory and Census: Multilevel Nearfield Shells

## Status and entry gate

**Done `2026-08-17`; awaiting independent clear-context approval.**

Entry gate: `025`, `037e`, `038`, `041`, and `041b` complete and approved.
This is a theory/measurement row only: artifacts may be added under
`theory/`, `scripts/`, and `data/`; no production `src/` or FLOWVPM changes.

## Objective

Determine whether the expensive residual direct U-list can be reduced by a
multilevel, exact-once interaction partition that reuses analytic expansions.
For every source/target node pair currently terminal-direct, recursively peel
off subpairs that satisfy both the multipole truncation bound and the
FLOWVPM regularization-tail bound. Route those subpairs through M2L, M2T, or
S2L as appropriate; descend the still-near complement; perform P2P only for
terminal subpairs that cannot economically satisfy an expansion bound.

This tests an analytic alternative to `041b`'s empirical strategic-target
interpolation. It evaluates existing expansions at actual particle targets or
through actual local expansions and therefore needs no learned field basis.

## Precise multilevel interpretation

The near list at an intermediate level is a **refinement queue**, not work that
is immediately evaluated. Otherwise evaluating it and a finer shell would
double count. Starting from a residual U-list pair `(S,T)`:

1. If `(S,T)` is expansion-admissible and regularization-safe, route it once:
   same-level comparable nodes use M2L; an adaptive leaf against a larger node
   uses the approved M2T/S2L orientation or an equivalent level-lifted M2L.
2. If it is not admissible and refinement is available, split the source,
   target, or both according to a registered cost rule and enqueue the child
   pairs. Each refinement creates a finer shell: newly admissible child pairs
   leave the near queue while the nearest-neighbor child pairs continue.
3. If no profitable refinement remains, route the pair once to direct P2P.

Thus the final partition is

`all ordered body pairs = M2L ⊔ M2T ⊔ S2L ⊔ P2P`,

with no omission and no overlap. Self-cell interactions necessarily descend;
each target's irreducible local neighborhood remains direct. Face, edge,
corner, and especially outer-shell pairs may partially promote as target or
source subregions separate.

## Relationship to existing work

- `025` supplies the rigid, level-scaled M2L stencil and exact-once language.
- `038`--`041` already implement adaptive U/V/W/X lists and M2T/S2L. This row
  does not duplicate them; it asks whether continuing a bounded virtual
  hierarchy *inside terminal U-list leaves*, or filtering actual targets
  against a reusable source node, economically peels more work from U.
- `037e` tested fine-bin/AABB pruning and failed the complete-solve promotion
  gate. This row must quantify the incremental promoted body-pair work and
  route overhead beyond that negative result; relabeling the same pruning is
  not sufficient.
- `041b` found empirical strategic-target bases did not generalize. This row
  uses the kernel's analytic expansion and rigorous error bounds instead.

## Required derivation

### 1. Two-part admissibility bound

For each candidate node or subnode pair derive a conservative bound combining:

1. singular-kernel multipole truncation error as a function of `P`, source
   radius, target radius/point, and center distance; and
2. the `gaussianerf` regularization-tail error as a function of the minimum
   source-target distance and source `sigma` range.

The bound must cover velocity and all nine velocity-gradient entries at the
delivered 1e-3 velocity gate and the standing J-derived diagnostic budget.
Carry per-subnode `sigma_min` and `sigma_max`; test whether spatial splitting
alone is sufficient or whether a bounded sigma-class split is necessary.

### 2. Routing choices and crossover

Derive measured crossover rules among:

- M2L for reusable source and target clusters;
- M2T/M2P for a source multipole admissible at selected target particles or
  target microcells;
- S2L/P2L for selected source particles or source microcells admissible for a
  reusable target local expansion; and
- direct P2P for the unresolved complement.

Arbitrary target-dependent source subsets are prohibited: a "partial
multipole" must correspond to a reusable spatial node/subnode (or a bounded
predeclared sigma class), not coefficients rebuilt per target.

The split-side rule must include source/target occupancy, available child
geometry, expansion evaluation cost, route-generation cost, and expected
reuse. Include direct fallback for tiny promoted groups.

### 3. Exact-once proof and computational oracle

Give an inductive proof that replacing a queued node pair by its child
Cartesian product preserves coverage and disjointness, and that removing an
admissible subpair into exactly one expansion route preserves the invariant.
Verify against brute-force ordered body-pair IDs on adversarial small trees:
self, face/edge/corner contact, unequal adaptive levels, empty children,
particles on boundaries, extreme sigma ratios, and static targets/sources.

## Cheapest decisive census

Before implementing new production kernels, add a deterministic local script
that replays existing cube, wake, rotor/multiscale, uniform, and adaptive
snapshots. It may construct virtual subleaves and routes offline but must not
evaluate a million-particle field. For every current terminal U-list pair,
record by interaction class and refinement depth:

- original and residual direct body-pair counts;
- body pairs promoted to M2L, M2T, and S2L;
- number, occupancy distribution, and reuse of each new route type;
- regularization-bound versus multipole-bound rejections;
- virtual-node count, metadata/capacity cost, and refresh work;
- estimates using measured production P2P/M2L/M2T/S2L kernel costs; and
- predicted change to the complete overlapped critical path, not a stage sum.

Compare at minimum `P=4` and `P=8`, both precisions, the approved near radii,
the adaptive `K_max` winners, and the sigma-heterogeneous rotor/multiscale
case. Separate self, face, edge, corner, and each inequivalent outer-shell
offset orbit. Include a no-virtual-node target-filtering variant and bounded
virtual depths 1--3.

## Promotion and stop gates

Stage a production implementation row after `042` only if the census predicts:

1. at least 20% reduction of the expensive mixed direct body-pair work on a
   material case;
2. at least 10% reduction of the nearfield critical path and 5% of the
   complete overlapped solve after all route/refinement/refresh costs;
3. no more than 3% predicted regression on any supported case under automatic
   direct fallback;
4. bounded capacity and zero recurring allocation compatible with graph
   capture and device residency; and
5. an exact-once oracle pass plus conservative U/J error bounds.

If the gate passes only for a measurable fat-cell or sigma regime, recommend a
regime-only selector rather than a general default. If it fails, record whether
the cause is insufficient promotable work, expansion cost, regularization
rejection, route explosion, or overlap with savings already captured by the
adaptive tree/`037e`, and close deeper nearfield refinement.

## Deliverables

1. `theory/multilevel-nearfield-shells.md`: derivation, invariants, bounds,
   crossover rules, and verdict.
2. `scripts/multilevel_nearfield_shell_census.jl`: deterministic route-only
   census and exact-once small-tree oracle.
3. `data/multilevel_nearfield_shells/`: compact checksummed CSVs and report;
   no particle-scale pair lists committed.
4. Result section here with GO, regime-only, or NO-GO and, only on GO, the
   proposed production successor scope.

## Acceptance

The exact-once proof and oracle agree; admissibility covers both multipole and
regularization error for U/J; the census is reproducible from existing case
constructors and prices the complete critical path; the result is explicitly
distinguished from `037e` and the already-shipped adaptive U/V/W/X machinery;
and no production code is changed.

## Result (`2026-08-17`)

**NO-GO.** The derivation, exact-once invariant, three gated measurements, and
calibrated crossover are in `theory/multilevel-nearfield-shells.md`.
`scripts/multilevel_nearfield_shell_census.jl` imports the existing task-038
tree/list/bound/painter machinery and writes compact, checksummed results to
`data/multilevel_nearfield_shells/`.

Phase 0 continued because the existing rotor whole-leaf sigma-floor ceiling
reaches 62.7%. Phase 1 also continued: ideal zero-overhead promotion reaches
26.2% on cube and 55.9% on multiscale, although wake is structurally dead at
0.051%. Phase 2 is decisive. The promoted virtual rectangles average far
fewer body pairs than the measured production M2L/M2T/S2L crossover requires;
every aggregate loses to direct before launch, virtual-P2M, refresh, metadata,
new-table, or serial-tail costs are added. The uncertainty-guarded selector
therefore chooses shipped direct fallback on every P4/P8 and Float32/Float64
row, giving zero predicted regression and zero gap to the bounded aggregate
oracle. Sigma-quantile blocking is immaterial, so sigma classes were not
admitted. No post-042 production successor is recommended.

## Clear-context approval (`2026-08-18`)

**APPROVED.** A first clear-context review reproduced the census numerically
(identical route census, 56/56 oracle rows with zero bad pairs, NO-GO robust)
but blocked because the committed `data/multilevel_nearfield_shells/` files
were stale outputs of an earlier script iteration. The unmodified committed
script was then re-run (4 threads, exit 0) and the data regenerated in place;
this second review verified the regeneration resolves every blocking item: all
11 `checksums.sha256` entries verify; `oracle.csv` carries 56 rows including
the boundary, extreme-sigma, and coincident adversarial cases, all with zero
bad pairs, `partition_sum_ok=true`, and the production-painter contract PASS
(57790/57790); the CSVs match the headers the script writes
(`bound_spotchecks.csv` has the multipole-bound rows, `manifest.csv` has
`production_painter`, `route_census.csv` has `sigma_min`/`sigma_max`, and
`route_census_by_orbit.csv` is ~100 KB with normalized orbit labels such as
`outer_0_0_1`); data mtimes postdate the script's. Headline claims match the
regenerated data: rotor whole-leaf ceiling 0.627, ideal promoted fractions
0.262 cube / 0.559 multiscale / 0.000506 wake / 0.415 sigma-multiscale, all
64 selector rows `direct_fallback` with zero gap and zero complete-solve
delta, verdict NO-GO.
