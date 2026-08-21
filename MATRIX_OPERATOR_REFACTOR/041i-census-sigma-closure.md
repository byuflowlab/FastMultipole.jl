# 041i Census: Sigma-Question Closure at the Shipped Operating Point

## Status and entry gate

**Staged `2026-08-18` (user direction). Not started.**

Approved execution plan (`2026-08-18`): `041i-implementation-plan.md` —
self-contained (exploration facts baked in); a fresh agent starts there.

Entry gate: `041g` and `041a` complete and approved. Measurement row only:
artifacts under `scripts/` and `data/`; no production `src/` or FLOWVPM
changes; local work at most four threads; H200 spot-check timing runs only if
Part B finds a depth/pair delta (see Method).

This row blocks `042`.

## Motivation and the gap it closes

The user hypothesized (`2026-08-18`) that large-particle `sigma` contaminates
the multipole representation (expansions approximate singular `1/r`, not the
regularized kernel actually evaluated), inflating the direct list and
under-utilizing expansions. The existing record largely contradicts this for
the rotor — `041g` measured **zero sigma-demoted body pairs** on the actual
rotor field at both registered counts (`data/sigma_class_m2l/census.csv`),
`037e` bucketing shows 58.7–62.7% of rotor direct pairs are pure-singular at
M2L-servable offsets (`data/flowvpm_gpu_campaign/fm037e_scoping_13170768.csv`,
`e2_share`), and the adaptive-path nearfield is only ~10% of lifecycle
(`data/target_owned_nearfield/stageB_bench.csv`) — but two channels remain
unmeasured, and both could still give the hypothesis teeth:

1. **Operating-point gap.** The `041g` zero-demotion census ran at
   `near_radius2 = 12` (`g_min = sqrt(5)`, `rho_t = 4.252`). The shipped
   adaptive default is `near_radius2 = 5` (`RADIX_DEFAULT_NEAR_RADIUS2`,
   `src/containers.jl:869`) with `rho_t = 4.789`, where `g_min = 1` — a
   2.24x tighter adequacy margin. At that point the rotor `ell = 7` leaf
   width (`1.2033/2^7 = 0.0094`) is *below* `rho_t * sigma_max = 4.789 *
   3.108e-3 = 0.0149` (n = 1e5), so demotions are geometrically expected.
   "Zero sigma demotions" has never been verified at the shipped defaults.
2. **Split-veto channel.** The sigma depth cap
   (`src/tree_batched.jl:802-813`: refuse subdivision when
   `gate_gmin * delta_child < rho_t * sigma_max`) never appears in any
   census. If sigma costs the rotor anything, it costs it as *forgone tree
   depth* — larger leaves and more U pairs — which a demotion count cannot
   see. The only evidence this channel exists at all is the synthetic
   `spread = 300` row of `data/fm041a_gpu_sigma.csv` (u_pairs 131.7M ->
   579.1M, +9% lifecycle).

## Objective

Produce a signed, quantitative statement of what `sigma` costs at the shipped
operating point — demoted pairs, forgone depth, extra U pairs, and (if
material) milliseconds — on the registered rotor snapshots plus the cube and
wake references, closing the sigma-contamination hypothesis with data at the
actual defaults rather than at the `041g` census point.

## Method

### Part A — q=5 demotion census

Re-run the `041g` census machinery (`scripts/sigma_class_m2l_census.jl`) with
the shipped list-builder parameters `near_radius2 = 5`, `rho_t = 4.789`
(alongside a `q = 12, rho_t = 4.252` control that must reproduce the `041g`
zeros bit-for-bit) on:

- the registered rotor snapshots (n = 1e5 and 1e6, existing checksummed
  provenance from `data/rotor_wake/`);
- the cube and wake reference fields at the `041a` registered counts.

Record per case: `demoted_body_pairs`, demoted fraction of direct body-pair
work, demotion locations by level, and the same exact-once class-partition
oracle exit `041g` used.

### Part B — split-veto (depth-cap) census

Read-only census driver (new script, `scripts/fm041i_split_veto_census.jl`)
that builds the adaptive tree twice per case — once with the sigma gate
active (shipped behavior) and once with the gate disarmed (`sigma` forced to
`0`/`eps` in the gate only; body data otherwise identical) — and reports:

- depth histograms and leaf-population histograms for both trees;
- the count of cells refused subdivision by the veto, by level;
- `u_pairs` (direct body-pair total) delta between the two trees;
- verification that both trees cover the identical body set exactly once.

If and only if the veto fires and the `u_pairs` delta exceeds 1% on any case,
add an H200 spot-check timing A/B (shipped settings, same-job anchors, `035`
critical-path pricing) to convert the pair delta into milliseconds.

## Gates and verdict

- **Reproduction gate:** the `q = 12` control rows must reproduce the `041g`
  zero-demotion result exactly; any mismatch stops the row for a
  coordination fix.
- **Verdict:** one of (i) *sigma-immaterial at shipped defaults* (demoted
  fraction and veto `u_pairs` delta both < 1% on every case), (ii)
  *sigma-material* with the measured pair and (if timed) millisecond cost and
  a recommendation for a separately staged successor (e.g. per-class demotion
  per the `041g` NO-GO alternatives, or geometry retuning), or (iii)
  *mixed*, stated per case. Cube/wake are expected to show the
  overlap-physics floor (`rho_t * sigma ≈ 9.6` particle spacings at
  `beta = 2`); the verdict must distinguish that floor from the rotor's
  regime rather than pooling them.

## Artifacts

- `scripts/fm041i_split_veto_census.jl` (new); Part A reuses
  `scripts/sigma_class_m2l_census.jl` with parameter overrides.
- `data/sigma_closure_census/` — census CSVs plus `report.md` with tables,
  the reproduction-gate check, and the verdict.

## Verification

- Part A: exact-once class-partition oracle passes on every row; `q = 12`
  control reproduces `data/sigma_class_m2l/census.csv` zeros.
- Part B: gate-on and gate-off trees verified body-count and coverage
  identical; any timing runs report same-job anchors and error bars.
