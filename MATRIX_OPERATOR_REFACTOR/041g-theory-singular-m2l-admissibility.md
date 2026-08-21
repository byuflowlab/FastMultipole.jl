# 041g Theory and Census: Singular-M2L Admissibility and Sigma-Class Re-Admission

## Status and entry gate

**Done `2026-08-18`; awaiting independent clear-context approval.**

Entry gate: `025`, `031a`, `037e`, `038`, `041`, and `041a` complete and
approved; `041c` complete (its census machinery and NO-GO verdict are inputs;
this row does not require its approval tick to begin, mirroring the `041e`
convention, but inherits none of its conclusions without re-derivation).
Theory/measurement row only: artifacts under `theory/`, `scripts/`, and
`data/`; no production `src/` or FLOWVPM changes; local work at most four
threads; no hardware runs required.

This row blocks `042`.

## Motivation and the gap it closes

The shipped sigma demotion is **whole-pair**: a V-candidate pair drops to the
direct U list whenever `gap < rho_t * sigma_max(source cell)`, and `041c`'s
sticky-demotion rule then barred M2L on that lineage entirely. Three pieces
of evidence say the barred region contains untested material work:

1. `041c` Phase 0 measured the rotor's whole-leaf pure-singular promotion
   ceiling at **58.7% (n=1e5) / 62.7% (n=1e6)** of direct body pairs
   (`data/multilevel_nearfield_shells/phase0_ceiling.csv`, from the 037e
   records) — the only case clearing the 20% material gate.
2. `041c`'s Phases 1–2 then ran on synthetic proxies (bimodal
   `sigma_multiscale` at census scale), not the rotor's spatially correlated
   18x sigma field, and its "sigma-quantile blocking is immaterial" finding
   is a proxy result, not a rotor result.
3. `041b` §7 lists the "M2L demotion rung" as an untried improvement.

Crucially, the `041c`/`041d` kill mechanism (promoted virtual rectangles far
below the ~67-pair M2L route crossover) applies to *virtual subleaves*.
Sigma-demoted pairs re-admitted at **whole-leaf** granularity carry up to
`K_max^2` (~1k–4k) body pairs per route — well above the crossover — so that
structural argument does not automatically kill this direction.

## Objective

1. Derive a **predictive admissibility criterion**: a conservative, cheap,
   refresh-time test for when the *singular* solid-harmonic M2L may serve a
   pair (or a sigma class within a pair) of the regularized `gaussianerf`
   problem within the standing error budget — and when it may not.
2. Generalize the verdict beyond any single case: identify the small set of
   dimensionless variables that collapse admissibility behavior, and verify
   the collapse.
3. Census the realizable gain on the actual rotor snapshot and a parametric
   synthetic sweep; if the gates pass, propose the improved interaction-list
   construction (per-class demotion instead of whole-pair demotion) as a
   separately staged successor.

## Required derivation

### 1. The kernel-difference bound (user hypothesis, to be formalized)

The error of applying singular `1/r` machinery to a regularized pair is
bounded by the kernel difference itself. This is already quantified: 031a §4
(`theory/kernel-splitting-nearfield.md`) gives the source-directed tail
envelopes

$$
E_U(\rho) = \bar g(\rho), \qquad
E_J(\rho) = \bar g(\rho) + \tfrac{\rho}{2}\,g'(\rho), \qquad
\rho = r/\sigma_{\text{src}},
$$

with the half-budget rule and the per-pair radii (`rho_t^U = 4.211`,
`rho_t^J = 4.789` at `1e-3`; accumulated-RMS variants 3.668/4.252). The
derivation must:

- prove that for a source *class* `c` with `sigma <= sigma_c`, the composite
  singular-M2L error for the class is bounded by the standard constant-P
  truncation bound (`constant_p_stencil_bound`,
  `src/interaction_list_batched.jl:8`, with the 038 §4 route asymmetries)
  **plus** `E_{U/J}(gap / sigma_c)` — additive, conservative, and evaluable
  from refresh-time statistics only (integer AABB gap + class sigma bound);
- choose and justify the per-pair vs accumulated-RMS tail variant
  consistently with the shipped partitioned nearfield's choice (3.668 is
  the shipped value; do not silently tighten or loosen it);
- cover U and all nine J entries, Lamb-Helmholtz channel composition, both
  precisions, and `P in {4, 8}`;
- spot-verify the bound numerically near the admissibility boundary against
  exact regularized sums on small deterministic cases (both sides of the
  boundary, both precisions, both orders).

### 2. Collapse variables and classification

The user hypothesis is that **particle sigma range and overlap** collapse the
behavior. Refine and verify: in dimensionless form the candidate collapse
set is

- `w/sigma ~ K^{1/3}/beta` — leaf width in sigma units (occupancy `K`,
  overlap `beta = sigma/delta`), which fixes the gap distribution in sigma
  units for each stencil offset;
- `S_local` — the sigma spread *within a near neighborhood* (not the global
  spread: spatially correlated sigma fields like CoreSpreading wake age have
  large global spread but small leaf-local spread, which is what class
  purity actually depends on);
- the near-radius/stencil parameter and `P` (discrete, already fixed by
  production choices).

Derive the admissible-work fraction as a function of these variables
analytically where possible (uniform-density model), and verify the collapse
empirically: census points with the same `(w/sigma, S_local)` but different
raw `(n, sigma, beta, K_max)` must land on the same admissibility curve
within stated tolerance. If the collapse fails, identify the missing
variable (e.g. sigma-gradient alignment with the stencil direction, fill
fraction) and report the corrected classification. The deliverable is an
**admissibility map**: given refresh-time statistics, predict the fraction
of currently demoted direct work that singular M2L can legally serve.

### 3. Class construction rules

- Classes are predeclared logarithmic sigma bins (the `041c` rule: a partial
  multipole may represent a reusable spatial node or a predeclared sigma
  class, never a target-dependent subset). Sweep class counts `{1, 2, 4, 8}`.
- A class multipole is a class-filtered P2M over the leaf's bodies with
  `sigma in bin`; the class routes M2L when `gap >= rho_t * sigma_c` AND the
  truncation bound passes; residual classes stay direct.
- Price: per-class P2M construction and refresh (per VPM rebuild), class
  metadata, additional route classes against the bounded 025 table set (new
  offset classes must be priced, not assumed free), scatter/compaction, and
  the direct-list shrink. Use the existing measured rates
  (`data/multilevel_nearfield_shells/cost_calibration.csv` and the 041a
  stage records) with the 041c uncertainty margins.
- The evaluation side must state where class contributions accumulate: into
  the existing target local expansions (preferred — L2B free ride) or a
  separate accumulation, and price the choice.

## Deterministic census

Extend/reuse the `041c` census machinery
(`scripts/multilevel_nearfield_shell_census.jl` imports of
`adaptive_octree_verify.jl`) — do not build a third tree/list harness.

Cases:

1. the actual DJI-9443 rotor sigma field at `n = 1e5` and `1e6` count
   reconstructions (deterministic constructors; no million-particle field
   evaluation);
2. a registered parametric synthetic sweep over overlap
   `beta in {1.5, 2, 3}`, global spread `S in {1, 3, 10, 18}`, and sigma
   spatial-correlation length (uncorrelated, leaf-scale, domain-scale) at
   fixed seeds — this is what validates or refutes the collapse;
3. the standing cube/wake/multiscale constructors as controls.

For every case/class-count/policy record: demoted pair population, class
occupancy histograms, admissible classes and body-pair fractions (predicted
by the map vs measured by the census), route counts and pairs per route
versus the measured M2L crossover, class-P2M/refresh/metadata/table costs,
predicted nearfield and complete overlapped critical-path deltas, selector
decision with uncertainty margin, and exact-once verification through the
existing painters (class partition of a pair's body set must be proven a
partition — zero omissions/duplicates).

## Promotion and stop gates

Stage a production successor (per-class demotion interaction list) after
`042` only if:

1. >= 20% reduction of demoted direct body-pair work on a material case
   (the rotor is the primary case);
2. >= 10% predicted nearfield critical-path reduction and >= 5% complete
   overlapped solve reduction after class-P2M, refresh, metadata, table,
   and route costs;
3. <= 3% predicted regression on any supported case under the automatic
   selector (conservative-by-construction: promote only beyond calibration
   uncertainty);
4. bounded capacity, zero recurring allocation, graph compatibility;
5. the admissibility-map prediction agrees with the census within stated
   tolerance (the map is the deliverable even on NO-GO).

If the gates fail, attribute the kill (class impurity, P2M refresh cost,
route-class explosion, table capacity, insufficient demoted work outside the
rotor) and record the admissibility map anyway — it documents when singular
M2L works, which was the user's primary question.

## Required artifacts

1. `theory/singular-m2l-admissibility.md`: kernel-difference bound proof,
   collapse-variable derivation and verification, admissibility map,
   class-construction rules, cost model, selector, verdict.
2. `scripts/sigma_class_m2l_census.jl`: deterministic census + collapse
   sweep + exact-once class-partition oracle (reusing the 041c import
   pattern).
3. `data/sigma_class_m2l/`: compact checksummed CSVs (manifest, admissibility
   map, census, calibration, selector, report).
4. Result section here with GO / REGIME-ONLY / NO-GO; Done checkbox update;
   independent clear-context review applies the Approved checkbox.

## Non-duplication statement

Distinct from `037e` (geometric pruning of the direct list — no expansions),
`037b` (two-pass deficit at expansion-validity geometry — kernel splitting,
not class routing), `041c` (virtual-subleaf refinement under sticky demotion
— this row lifts the demotion bar at whole-leaf granularity with a priced
class mechanism), and `041d` (smooth-basis P2M/M2P — this row uses the
existing singular basis and existing operator tables wherever the class gap
test passes). `041f` owns the smooth-solver alternative; this row is its
cheap harmonic complement, and `042` should weigh their verdicts together.

## Result (`2026-08-18`)

**NO-GO.** `theory/singular-m2l-admissibility.md` proves the additive
triangle-inequality decomposition and records the binding cutoff policy: 3.668
is U-only RMS, while the selector and census use the shipped combined U/J RMS
radius 4.252. The deterministic table separately verifies exact U/all-J tails
and scalar P4/P8 local-series bounds in both precisions. It also exposes the
decisive accuracy gap: the existing task-025 scalar-potential bound and offline
statistics do not certify delivered U/J derivatives with live phi/chi budgets,
so the selector never assumes an arbitrary channel ratio.

The reduced `(w/sigma,S_local)` collapse fails the registered agreement
tolerance. Fill, a directional-gradient diagnostic, and gap-bin cardinality
do not recover an independent low-dimensional map; the exact classifier needs
the full strength-weighted directional route histogram, whose replay is the
census itself. The failed map is recorded rather than fit away. Compact
accepted-class versus residual-direct painters have zero omissions/duplicates.

The canonical DJI-9443 iterator emits exactly 100,000 and 1,000,000 particles
with 17.94x sigma range. Both counts use fully materialized imported task-038
trees and `q=12,rho=4.252` lists; the million-particle field is not evaluated.
Demotion ancestry is followed exactly. Both rotor counts record zero
sigma-demoted routes, so there is no rotor work for class re-admission.
The full 36-case synthetic sweep emits exactly 100,000 particles per case.
Ceiling pricing includes actual offset classes and route/operator capacity,
but missing accuracy certification, failed map agreement, and zero rotor
material work independently force direct fallback on every row.

Therefore there is no default change and no post-042 production successor.
The checksummed artifacts are in `data/sigma_class_m2l/`, generated by
`scripts/sigma_class_m2l_census.jl`; no production or FLOWVPM source changed.

## Independent clear-context review (`2026-08-18`)

**Approved.** Two full `JULIA_NUM_THREADS=4` generations were byte-identical
across all eight checksummed payloads, and every checksum validates. The review
confirmed exact 100k/1M rotor and 36-case synthetic registration, exact
demotion ancestry, 48/48 passing painters (24 with nonzero demotions), all typed
U/all-J and separate scalar P4/P8 boundary checks in both precisions, the honest
reduced-map failure, `accuracy_certified=false`, and 720 direct fallbacks.

The timing audit confirmed case/count/precision direct rates; exact rotor
Float32/Float64 nearfield and solve anchors at 100k/1M; case-specific 041a
anchors with count/precision scaling only where needed; full-anchor proposed
nearfield/solve arithmetic with separate selected-fallback deltas; finite
zero-demotion rotor rows at about 1.1% proposed regression; and bounded
mixed-level center/level route keys and capacity pricing. `git diff --check`
passes. No significant correctness, robustness, compliance, performance-model,
minimality, readability, or NO-GO-support issue remains.
