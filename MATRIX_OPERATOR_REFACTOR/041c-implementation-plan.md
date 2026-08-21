# 041c Execution Plan: Optimal Multilevel Nearfield Shells

## Handoff context

This document is a standalone execution plan for task
`041c-theory-multilevel-nearfield-shells.md`. A new agent should begin by
reading `START_HERE.md`, then the 041c task file, then this plan. Follow the
coordination document's restrictions on sibling task files and preserve all
unrelated working-tree changes.

At plan creation time:

- branch: `matrix-ops`, ahead of `origin/matrix-ops`;
- `041a` is Done and Approved;
- `041b` is Done but its Approved checkbox is blank;
- the user explicitly directed: "don't worry about it- do it now," so 041c may
  begin despite that missing approval tick; this permission does not approve
  041b or waive 041c's later independent clear-context review;
- 041c is theory/measurement only. Do not modify production `src/` or FLOWVPM;
- the worktree already contains unrelated tracked and untracked work,
  including the 041b/041c coordination additions. Do not overwrite, clean, or
  fold those edits into 041c accidentally.

The user's performance requirement is stronger than merely finding a viable
configuration: seek the optimal bounded performance configuration. Here,
"optimal" means a deterministic automatic configuration envelope that
minimizes predicted complete overlapped solve time while satisfying accuracy,
capacity, allocation, and regression constraints. Also report the per-case
oracle optimum and the selector's gap to it.

## Relevant repository shape

The existing implementation already provides the geometry, bounds, and
execution surfaces needed for an offline census:

- `AdaptiveTreePolicy` and `AdaptiveRadixTree` (`src/containers.jl:578,636`)
  hold the adaptive tree, per-node geometry, body ranges, and
  `node_sigma_max`. **`sigma_min` is not stored per node**: the census must
  derive per-subnode `sigma_min` (and any sigma quantiles) from
  `body_sigma[perm[lo:hi]]`. The uniform CUDA path's `cell_sigma_min/max` is
  a template only, not a data source.
- `AdaptiveInteractionLists` holds terminal U pairs (flat node indices) plus
  V/M2L, W/M2T, and X/S2L routes. `_adaptive_gap2_lattice`
  (`src/interaction_list_batched.jl:1048`) computes the integer-exact minimum
  squared AABB gap between mixed-level nodes without touching particles —
  reuse it (or its standalone twin) for all admissibility geometry.
- `build_adaptive_interaction_lists!` (`src/interaction_list_batched.jl:1142`)
  performs the current exact-once dual-tree recursion. Its U pairs are the
  roots of the proposed residual refinement queues.
- The regularization-tail admissibility theory already exists: 031a §4
  (`theory/kernel-splitting-nearfield.md:184-231`) with the `E_U`/`E_J`
  envelopes and the ρ_t table (1e-3 → ρ_t = 4.211 for U, 4.789 for J), wired
  into the adaptive gate `gap(A,B) >= rho_t * sigma_max(source)` per 038 §5
  (`theory/adaptive-radix-octree.md:640-727`). Multipole truncation bounds
  come from `constant_p_stencil_bound` (`src/interaction_list_batched.jl:8`)
  and 038 §4.1–4.4, including the route asymmetries (M2T has no target-side
  truncation; S2L has no source-side truncation; the `c <= 2` exclusion).
  `src/error.jl` is the legacy non-radix path and is **not** the right
  machinery here.
- Two independent exact-once oracles already exist: the standalone
  src-independent harness `scripts/adaptive_octree_verify.jl`
  (`check_exact_once` at :366, plus tree/list builders, bound helpers, seeded
  case constructors, and checksummed CSV output) and the production-path
  brute-force painter `_adt_exact_once_bad`
  (`test/adaptive_octree_test.jl:91`).
- Production defaults use `near_radius2=5`; existing approved evidence also
  contains comparison radii (task 028 records, `containers.jl:341-346`). Do
  not silently replace the production default.
- Existing adaptive and nearfield data include `fm041_cuda_cost.csv`, the
  `fm041a_*` timing/count CSVs (including the leaf-population histograms
  `fm041a_leafpop.csv` / `fm041a_contrast_leafpop.csv` and the per-stage
  `fm041a_gpu_stages.csv` with both overlapped and serial lifecycle columns),
  `data/adaptive_octree/`, the 037e campaign screen, and split-nearfield
  timing records. Reuse these before running new expensive measurements —
  but note their calibration limits recorded in the microbenchmark section
  below.

Do not implement a second adaptive tree. Virtual subnodes exist only inside
the offline census to test whether terminal U-list work can be peeled into
analytic shells.

## Phased execution and kill gates

Run the work in three gated phases. Each phase ends with an explicit
continue/kill decision recorded in the report; a kill at any gate produces the
final NO-GO (or REGIME-ONLY) verdict immediately, with the remaining sections
executed only to the depth needed to document the verdict.

**Phase 0 — analytic pre-kill (σ-floor ceiling; hours, no new harness).**
The regularization admissibility floor `gap >= rho_t * sigma_max` does not
shrink under spatial refinement: splitting improves only the multipole term
and, where sigma is heterogeneous, subnode `sigma_max`. Therefore an upper
bound on the infinite-depth, zero-overhead promotable pair fraction is
computable from existing aggregates (integer AABB gaps, `node_sigma_max`,
leaf-population histograms) before building any census machinery. Existing
histograms lack U-pair geometry, source/target correlation, and per-subnode
sigma distributions, so Phase 0 yields a rigorous **ceiling**, not the exact
fraction — that is sufficient for a kill. If the ceiling cannot reach
promotion gate 1 (>=20% mixed direct body-pair reduction on a material case),
or the zero-overhead critical-path model built on it cannot reach the 10%/5%
gates, record NO-GO and stop. Mirror 041b's A0 discipline: one page of
arithmetic over existing CSVs, then the verdict.

**Phase 1 — route-only census at ideal cost.** Reconstruct the deterministic
census-scale cases and measure the actual promotable fractions, route
histograms, occupancy/reuse, and rejection attribution
(multipole-bound vs regularization-bound; and for regularization rejections,
`sigma_max` vs sigma-quantile blocking) — still assuming zero route/refresh
overhead and perfect batching. Kill gate: ideal savings on the mixed
(`PartitionedVortex`) bucket, which is 82–90% of nearfield kernel time, must
clear the promotion gates with margin. The exact-once proof and oracle are
required from this phase onward; the cost model and selector are not.

**Phase 2 — full cost model, optimal configuration search, and selector.**
Only if Phase 1 passes. All remaining sections below.

## Required outputs

Create only the task-prescribed theory/measurement artifacts:

1. `theory/multilevel-nearfield-shells.md`
   - derivation and conservative error bounds (reusing 031a/038 as below);
   - exact-once invariant and proof, including the lineage-aware legality
     rule;
   - route/split crossover rules;
   - complete critical-path cost model;
   - Phase-0 ceiling arithmetic and each phase-gate decision;
   - optimal-selector construction and final GO, REGIME-ONLY, or NO-GO
     verdict.
2. `scripts/multilevel_nearfield_shell_census.jl`
   - deterministic route-only census;
   - bounded configuration search;
   - adversarial exact-once oracle;
   - compact CSV/report/checksum generation.
   The script must **reuse/import the standalone machinery in
   `scripts/adaptive_octree_verify.jl`** (tree, lists, exact-once painter,
   bound helpers, seeded constructors, checksum output) rather than
   reimplementing it, and must **cross-check against the production-path
   painter in `test/adaptive_octree_test.jl`**. Two independent oracle
   implementations already exist; do not write a third divergent one.
3. `data/multilevel_nearfield_shells/`
   - manifest and cost calibration;
   - Phase-0 ceiling table;
   - route census;
   - oracle and bound spot-check results;
   - selector decisions and oracle gaps;
   - capacity/refresh estimates, report, and checksums.

Never commit particle-scale ordered-pair lists. Store only aggregates and
small adversarial-oracle detail sufficient to diagnose failures.

## Theory and admissibility

Do **not** re-derive the regularization bound. Reuse verbatim:

1. the 031a §4 Gaussian-erf tail envelopes `E_U`, `E_J`, the
   singular→regularized conversion, the half-budget rule, and the ρ_t table
   (1e-3 → 4.211 U / 4.789 J); and
2. the 038 §4 constant-P truncation bounds (`constant_p_stencil_bound`) with
   the M2T/S2L asymmetries and the `c <= 2` exclusion.

What is new — and the only theory this row derives — is the **composition** of
those bounds with virtual subnodes and per-route admissibility, for each
candidate source node/subnode `S` and target node/subnode or target set `T`:

- M2L for reusable source and target clusters;
- M2T/M2P for a reusable source multipole and target particles/microcell;
- S2L/P2L for source particles/microcell and a reusable target local
  expansion.

The combined bound must cover velocity and all nine velocity-gradient entries
at the standing `1e-3` velocity gate and J-derived diagnostic budget. Reuse the
existing project error-budget allocation and Lamb-Helmholtz order convention;
do not invent a looser 041c-only tolerance. Carry `sigma_min` and `sigma_max`
per virtual source node (derived from `body_sigma[perm[...]]`; see repository
shape above).

State explicitly in the theory document: the σ floor is fundamental for
harmonic-expansion routes — solid-harmonic expansions cannot represent the
regularized field inside ~`rho_t * sigma`; Gaussian/Hermite
(fast-Gauss-transform) expansions are the only way past that floor and are
out of scope for this row (the funded 037d VIC direction owns them).

Test spatial refinement without sigma classes first. Gate the sigma-class
sweep on Phase-1 rejection attribution: only if a material share of
regularization rejections is caused by node `sigma_max` exceeding the
sigma-quantile of the same subnode (heterogeneity blocking otherwise useful
promotion) do the fixed logarithmic source-sigma class counts `{1,2,4,8}`
enter the configuration grid. A partial multipole may represent only a
reusable spatial node or a predeclared sigma class. Never rebuild an
arbitrary target-dependent source expansion.

Record the **L2B free-ride asymmetry** as a crossover prior: S2L deposits into
a target local expansion that is evaluated by L2B anyway, so its marginal
evaluation cost is near zero, while M2T pays per target body — subject to
S2L's stricter convergence constraint for touching geometry (the source must
lie outside the target expansion's convergence region).

## Exact-once traversal and oracle

Treat each existing terminal ordered U pair `(T,S)` as a queue item, not as
immediate work. For every item choose exactly one action:

- emit one admissible M2L, M2T, or S2L route;
- replace it with the Cartesian product induced by a source split, target
  split, or simultaneous split; or
- emit one terminal direct P2P group.

**Lineage-aware route legality.** 038 §5.2 sticky demotion prohibits V/M2L
re-admission below a sigma-demoted lineage under the bounded task-025
operator tables; it does not prohibit M2L shells produced by refining an
ordinary geometrically-near U pair. Therefore:

- ordinary near lineage: M2L, M2T, and S2L may all be considered;
- sigma-demoted lineage: only M2T, S2L, target filtering, or direct — unless
  a future production successor explicitly approves new operator tables.

Caveat carried into the cost model: even on ordinary lineages, virtual-subnode
M2L can produce offset classes outside the rigid task-025 stencil, so
new-operator-table storage and capacity must be priced (a cost item, not a
legality bar).

Maintain the invariant that queued, promoted, and direct body-pair sets are
pairwise disjoint and their union equals the original U-pair Cartesian
product. Prove inductively that splitting preserves this invariant and route
removal preserves it. Self interactions must descend until a non-self safe
route or irreducible direct group is obtained.

The computational oracle must compare explicit ordered body-pair IDs on small
adversarial trees and cover:

- self, face, edge, corner, and inequivalent outer-shell contacts;
- unequal adaptive levels and one-sided refinement;
- empty children and particles exactly on spatial boundaries;
- extreme and spatially heterogeneous sigma ratios;
- static targets, static sources, and coincident source/target systems.

Assert zero omissions, zero duplicates, and route admissibility for every
case. Independence is obtained by reuse, not reimplementation: run the
traversal against the imported `adaptive_octree_verify.jl` painter and
cross-check the same cases through the production-path painter in
`test/adaptive_octree_test.jl`.

## Optimal configuration search

(Phase 2 only.) Enumerate the complete bounded global design space:

- virtual depth `0:3`, where depth zero is the no-virtual-node target-filtering
  variant;
- direct fallback, source split, target split, simultaneous split, M2L, M2T,
  and S2L actions wherever legal under the lineage rule;
- `P in {4,8}` and `Float32`/`Float64` (task-mandated; do not trim);
- production `near_radius2=5` primary, with one justified approved comparison
  radius as a sensitivity row;
- adaptive `K_max` winners from the existing widened benchmark evidence;
- sigma classes `{1,2,4,8}` if admitted by the Phase-1 attribution gate;
- tiny-group direct thresholds and reuse thresholds derived from measured
  kernel crossover costs rather than arbitrary constants.

**Per-pair optimization.** Because batching and overlap make costs
non-pair-additive, a per-pair multi-objective Pareto DP cannot certify the
aggregate optimum anyway. Use a two-level optimizer instead:

1. choose each residual U pair's optimal action under a scalarized calibrated
   cost vector (unit route costs plus the per-route overhead terms below);
2. evaluate the induced aggregate configuration through the lifecycle
   dependency graph (accumulated route histograms, batching, overlap, serial
   tail);
3. enumerate all cost-vector crossover breakpoints, or iterate the cost
   vector to convergence — not a fixed small number of rounds;
4. verify the selected aggregate against a bounded branch-and-bound or
   exhaustive reference on representative census subsets. Small adversarial
   trees prove exact-once correctness, not performance optimality; this
   check is what supports the optimality claim at census scale.

The performance objective must model the complete overlapped critical path,
not a sum of isolated stage times. Start from measured production timings
(`fm041a_gpu_stages.csv` carries both overlapped and serial lifecycle
anchors) and price:

- saved mixed direct body-pair work by contact/orbit and occupancy;
- added M2L/M2T/S2L work with batching and reuse;
- **per-route fixed launch/latency overhead and batch-shape compatibility**
  (mandatory calibrated inputs, not afterthoughts: 027 measured ~50 µs/window
  latency dominating, and 023b's per-class launches were launch-bound —
  many small virtual-subnode routes that cannot join existing batched route
  classes would recreate exactly that regime; the model needs a calibrated
  per-route/launch constant and an explicit rule for which promoted routes
  share existing streams);
- **refresh amortization** (mandatory: virtual-subnode P2M, sigma min/max
  sweeps, and route regeneration recur every refresh under VPM per-step
  rebuild, while direct P2P has no setup; pin the amortization cadence —
  per step vs stable epoch — and charge it on the critical path);
- route generation, virtual metadata, scans/compaction, new-operator-table
  storage, and graph capture/device-residency costs;
- the actual nearfield/farfield overlap and any serial tail introduced by the
  promoted route;
- uncertainty or interpolation error in timing calibration, recorded
  explicitly per calibrated constant.

**Calibration and pre-registered microbenchmarks.** Existing records cannot
calibrate the full grid: `fm041_cuda_cost.csv` has no `P` or `near_radius2`
columns (fixed P=4, q=5); `fm041a_gpu_widen.csv` covers only P=4;
`fm041a_pweep.csv` covers P in {2,3,6,8} but only Float64 on wake/multiscale;
and the M2T/S2L stage-timing samples are tiny. Use existing H200 records as
the primary anchors and pre-register these targeted microbenchmarks (no
million-particle field evaluations):

- M2T and S2L throughput over source/target occupancies spanning the observed
  leaf histograms, for `P in {4,8}` and both precisions;
- batched vs fragmented route shapes, including launch counts and whether
  promoted routes can join existing streams;
- virtual-subnode P2M, sigma min/max reduction, route generation,
  scan/compact, and metadata-refresh costs;
- per-step rebuild and amortized stable-epoch variants;
- a small end-to-end validation set confirming the cost model's predicted
  critical-path delta (validation only — 041c remains a non-implementation
  row).

For every benchmark configuration, record the exhaustive oracle optimum.
Then synthesize a deterministic automatic selector using only statistics that
would be available at refresh time: `P`, precision, near radius, `K_max`,
source/target occupancy, geometry, sigma spread, reuse, predicted route cost,
and required capacity. Prefer an auditable threshold table/decision tree over
an opaque fitted model. Make the selector **conservative by construction**:
it promotes only when the predicted win exceeds a margin at least as large as
the recorded calibration/interpolation uncertainty, so the <=3% regression cap
is structural (default fallback to direct) rather than an empirical outcome.
Validate out of sample by leaving each physical case out in turn — as
confirmation, not as the safety mechanism — and report the selector's
performance gap to the oracle.

## Census coverage and schemas

Replay deterministic, census-scale versions of the existing cube, wake,
rotor/multiscale, uniform, and adaptive cases. Include the sigma-heterogeneous
rotor/multiscale case. Do not evaluate a million-particle field. Preserve the
existing seeds and constructors where possible and record provenance in a
manifest. Local census runs use at most 4 threads (standing user rule); only
the pre-registered microbenchmarks use HPC.

At minimum, the route census must group by case, `P`, precision, near radius,
`K_max`, virtual depth, sigma-class count, contact/orbit class, and refinement
depth, and record:

- original and residual direct body-pair counts;
- promoted body pairs and route counts for M2L, M2T, and S2L;
- occupancy and reuse distributions;
- multipole-bound and regularization-bound rejection counts, with
  regularization rejections attributed to `sigma_max` vs sigma-quantile
  blocking (feeds the sigma-class gate);
- virtual node count, metadata bytes, required capacity, and refresh work;
- modeled stage deltas and complete overlapped critical-path delta;
- accuracy margin, selector decision, oracle decision, and selector gap.

All CSV rows must include seed/configuration provenance and the source timing
record used for calibration. The report must state where timing interpolation
or conservative extrapolation was necessary.

## Acceptance and verdict gates

Before marking 041c Done:

- each phase-gate decision (Phase 0 ceiling, Phase 1 ideal-cost gate) is
  recorded with its arithmetic; a kill at either gate is an acceptable
  completion of the row;
- the written proof and brute-force oracle agree, and the script-side and
  production-path painters agree on the shared adversarial cases;
- bound spot checks pass near admissibility boundaries for U and J at both
  orders and precisions;
- a baseline mode reconstructs current U-list counts and existing timing
  anchors within explained tolerances;
- repeated census runs produce byte-identical CSVs and valid checksums;
- the selected aggregate configuration is verified against the bounded
  branch-and-bound/exhaustive reference on the representative subsets;
- the selector satisfies the regression cap by construction and in
  leave-one-case-out validation;
- `git diff` confirms no production or FLOWVPM changes.

Recommend a production successor after 042 only when all task gates pass:

1. at least 20% reduction in expensive mixed direct body-pair work on a
   material case;
2. at least 10% reduction in nearfield critical path and 5% in the complete
   overlapped solve after all overheads;
3. no more than 3% predicted regression on any supported case with automatic
   fallback;
4. bounded capacity (including any new operator tables), device residency,
   graph-capture compatibility, and zero recurring allocation;
5. exact-once and conservative U/J error gates pass.

Use `REGIME-ONLY` if the automatic selector safely wins only in a measurable
fat-cell or sigma regime. Use `NO-GO` if the gates fail, and attribute the
failure to the σ-floor ceiling, insufficient promotable work, expansion cost,
regularization rejection, route explosion, launch/refresh/metadata cost, or
overlap with savings already captured by adaptive U/V/W/X or 037e. A GO
successor must be narrowly incremental and must not reopen the adaptive-tree
design.

After completing the artifacts, update the 041c task result section and its
Done checkbox only. A different clear-context agent must inspect the listed
artifacts and apply the Approved checkbox.
