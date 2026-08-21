# 041d Theory and Census: Smooth-Basis Nearfield Alternatives

## Status and entry gate

**Done `2026-08-17`; revised same day after an external review (coarsened/
merged-cluster census added, Part-1 verdict scoped to the tested model
family, per-cluster formation accounting fixed, Part-2 point-count law
relabeled a heuristic hypothesis); awaiting independent clear-context
approval.**

Entry gate: `037d`, `041b`, and `041c` complete (this row was user-directed
on `2026-08-17` from the 041c NO-GO discussion, conditional on that verdict
standing). Theory/measurement row only: artifacts under `theory/`,
`scripts/`, and `data/`; no production `src/` or FLOWVPM changes; no
hardware runs.

## Objective

After 041c closed multilevel harmonic-shell refinement of the residual
U-list, close or fund the remaining smooth-representation alternatives for
the nearfield (66–95% of the shipped U/J solve, per 037b):

1. **Regularized-basis P2M/M2P substitution** (user proposal): expansions of
   the `gaussianerf` kernel in a sigma-regularized basis ("regularized solid
   harmonics" or any linear alternative), applied P2M/M2P-only — no M2L — to
   substitute direct pairs beyond the nearest-neighbor list, including
   inside the singular sigma floor where solid harmonics are inadmissible.
   Decide with a registered census pre-kill reusing the 041c machinery and
   calibration, not an armchair argument.
2. **Sigma-adaptive multilevel smooth representation** for the
   sigma-heterogeneous regime that 037d's global-mesh VIC verdict could not
   serve: estimate whether resolution tracking local `sigma(x)`
   (AMR-VIC / multilevel summation) escapes the domain-wide `sigma_min`
   floor that killed the rotor case, and whether a derivation row is
   justified.

Prior evidence constraints: the 041b rank probe operated at `sigma/h ~ 1e-3`
(regularization inert) and does not bound sigma-scale smooth blocks; 037b
killed two-pass deficit splitting on pass-1 grounds; 037e/037f exhausted
geometric pruning and pair-kernel cheapening; 037d funded uniform-sigma VIC
and killed global/sigma-binned meshes for the rotor.

## Deliverables

1. `scripts/smooth_nearfield_prekill_census.jl`: deterministic registered
   census — two bracketing admissibility policies (conservative
   sigma-regularized geometric bound; optimistic Hermite/FGT bound), order
   sweep `p = 2:2:8`, both cluster-granularity directions — source-leaf
   subdivision to virtual depths 0–3 and (added `2026-08-17` after external
   review) ancestor-grouped merged/coarsened clusters up to the all-union
   extreme — 041c-identical cases/seeds/trees, priced per-route and in
   aggregate with the 041c `cost_calibration.csv` rates and uncertainty
   margins, with formation charged per cluster at its own max used order.
2. `data/smooth_nearfield_prekill/`: compact checksummed CSVs
   (`m2p_census.csv`, `coarsened_census.csv`, `break_even.csv`,
   `manifest.csv`, `report.txt`).
3. `theory/sigma-adaptive-smooth-nearfield.md`: part-1 derivation and
   verdict; part-2 mesh-point-count law, solver-structure options,
   registered cost band, risk list, and staging recommendation.

## Result (`2026-08-17`)

**Part 1: NO-GO for the tested census family under the registered proxy
cost model (universal linear-basis closure is *not* claimed).** The
break-even law `|S| > (c_eval/c_direct)·n_c(p)/25` requires 28–74-source
clusters at `p=2` (rising to 469–1222 at `p=8`). The census covers both
cluster-granularity directions: *subdivision* of terminal U source leaves
(depths 0–3) finds zero conservative promotions on every case/depth and only
0.11% optimistic-Hermite promotions on `sigma_multiscale`; the
*coarsening/merging* census added after external review (ancestor-grouped
merges of each target's U sources plus the all-union extreme, clusters up to
2047 sources) finds merged clusters that do clear per-route break-even (best
ratio 4.36x optimistic, 1.82x conservative, all inside the singular floor),
but the aggregate net of formation peaks at a 2.3% saving (optimistic) and
is negative (conservative); the selector chooses direct fallback on all 64
rows. A rigorous monotonicity lemma (merging never improves admissibility)
plus the ~2.3% measured ideal-model ceiling — below the standing 5% lever
threshold — ground the NO-GO. Scope caveats registered in the theory doc:
the eval rate is a harmonic-M2T-anchored proxy, not a lower bound for a
purpose-built M2P kernel; the named residual before any universal closure is
a rank/DOF-to-cost lower-bound study on actual regularized U+J blocks
(not staged, being under-threshold). Formation is charged per cluster at its
own max used order (2026-08-17 review fix).

**Part 2: OPEN — derivation row recommended (staged as `041f`), not an
implementation row.** Under the disjoint particle-owned-volume hypothesis —
explicitly labeled a heuristic, not a derived point-count law, since rotor
blades/wakes have sheet/filament geometry with overlapping support and AMR
patch-fill/2:1 overhead the sum does not count — resolution tracking local
`sigma(x)` yields `N_mesh ~ 0.75 n` nominally independent of sigma spread
(1.5–3M points for rotor `n=1e6` after x2–4 AMR overhead, versus 5e11+
global), priced at ~3–8 ms nominal F32 against the shipped 33.3 ms rotor
eval with no sign flip in the opt/pess band. The x2–5 AMR coupling overhead
is a placeholder; `041f` must replace both the hypothesis and the overhead
band with concrete level-by-level counts on the actual rotor snapshot before
any implementation is staged. The funded uniform-sigma VIC row (037d)
remains the largest known nearfield lever and proceeds independently.

## Acceptance

Census reproducible from existing case constructors at <= 4 local threads;
pricing anchored to recorded production rates with registered uncertainty;
part-1 verdict explicitly distinguished from 037e (geometric pruning), 041b
(empirical target bases at cell scale), and 041c (harmonic shells); part-2
estimate cites 037b/037d records for every anchor; no production code
changed.

## Clear-context approval (`2026-08-18`)

Approved by clear-context subagent. Checked: script registration matches the
executed model (two predeclared bracketing policies, half budget `5e-4`,
`p = 2:2:8`, seed 41003, per-cluster max-order formation, selector with the
registered 25%/35% margins; hard-coded rates verified identical to
`data/multilevel_nearfield_shells/cost_calibration.csv`); all CSV checksums
verify; every quantitative claim reconciled against the data (break-even
28.4–74.1 at `p=2` to 468.9–1222.3 at `p=8`; zero conservative subdivision
promotions on all 16 rows; 3,168/2,856,358 = 0.11% Hermite promotions with
2,496 inside the floor; coarsened best ratios 4.357/1.816 with all promoted
pairs inside the floor; net-of-formation peak saving 2.33% on
`sigma_multiscale` and 1.23% on `wake`, negative under conservative;
`direct_fallback` on all 64 rows; all-union `q >= 1.33`). The monotonicity
lemma is sound, and the NO-GO is robust: even margin-free, the ideal-model
ceiling (~2.3%) sits below the 5% lever threshold. Part-2 arithmetic checks
(`(1/1.1)^3 ≈ 0.75`, 1.5–3M points, 2.76–18.9 ms and 33.3 ms anchors
corroborated in 037d records), the heuristic-hypothesis labeling is explicit,
and the OPEN + `041f` derivation-row recommendation follows. `src/` is
untouched. Minor non-blocking notes: the aggregate selector inflates the whole
proposed cost (including the untouched direct residual) by the margin, which
is stricter than margining only the delta — immaterial here since the
margin-free saving already fails the 5% threshold; and the coarsened formation
key (ancestor id) can conflate member sets across targets, an optimistic
simplification that only strengthens the NO-GO. Verdict: APPROVED.
