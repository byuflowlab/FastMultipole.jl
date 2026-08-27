# 052c sigma-collapse experiments (ledger, started 2026-08-26)

Ryan's directive: record what we try and how it affects the solution.
One section per trial; every trial names its exact config (the
`sigma_guard` kwarg + env knobs), the jobs run, and the measured effect
on the solution (gates, wake health, timing). Guard mechanism is
kwarg-gated end to end so alternative approaches slot in without
re-plumbing.

## Baseline failure (no guards) — jobs 13497273 (full) + 13501467 (probe)

1080-step stage-d acceptance, unguarded rVPM core-size update
`sigma -= dt*sigma*Z` (Euler): min_sigma contracts monotonically
(0.0037 @100 → 1.4e-4 @1000), crosses ZERO at step 1015, wake diverges
(max_u 18 → 1743 m/s by step 1069), run dies at step ~1071 in the
BoundCirculationMonitor host `direct!` (Brent bracket `(0, negative)`
in `FLOWVPM_fmm.jl solve_ρ_over_σ` once sigma < 0). Diagnostic probe
13501467 (restart 990 → 1031, WAKE_HEALTH_DTZ + ATTRIBUTION on):

- **Outlier, not population**: p1_sigma_ratio flat at 0.32 the whole
  window while min_sigma_ratio goes 0.033 → −3.3.
- **Euler sign flip caught in the act**: max dt·Z runs 0.05–0.13
  (steps 991–1013, one persistent particle at (0.333, −0.133, 0.099)),
  then dt·Z = 1.24 @1014 → sigma flips negative @1015; second event
  dt·Z = 164 @1024 → min_sigma −0.0148 @1025. `sigma*(1−dt·Z)` flips
  sign whenever dt·Z > 1; no floor/cap existed anywhere.
- Reproducible: f32-restart probe crossed at the same step 1015.
- Probe data: `orc:FLOWPanel-052-h200/data/fm052d_sigma_probe_13501467/`
  (wake_health CSV has dtz + attribution columns); full-run health CSV in
  `.../data/fm052d_gpu_1080/monitors/`.

## Mechanism (implemented 2026-08-26, kwarg-gated; Ryan: "gated by
## kwargs so we can easily try different approaches")

`sigma_guard::NamedTuple` kwarg threaded the whole chain, empty = legacy
bit-exact:

- FLOWVPM `euler`/`_euler` (reformulated scalar + broadcast paths) →
  `_sigma_guard_params` (`FLOWVPM_timeintegration.jl`). Recognized keys
  `dtz_cap` (max per-step contraction fraction dt·Z), `floor` (absolute
  sigma lower bound); unknown keys throw, so new guard laws must be
  registered there. ClassicVPM `_euler`, `euler_exp`, `rungekutta3`
  reject / don't accept a non-empty guard (unguarded sigma updates).
- FLOWPanel `simulate!` (kwarg) → `propagate!` (`FLOWPanel_wake.jl`) →
  `FLOWVPM._euler`; warm-start replay forwards it too
  (`FLOWPanel_warmstart.jl`).
- Driver env knobs (`rotor_hover_pressure_comparison.jl`):
  `SIGMA_DTZ_CAP` (default Inf = off), `SIGMA_FLOOR_FRAC` (default 0 =
  off; floor = frac × tip_sigma_default, the SigmaOverlap shed sigma
  0.0044506 m for the 052 case). Config echoed in the "Particle
  diagnostics" banner of every log.
- Testset "euler sigma_guard: dt*Z cap + floor (052c trial 1)" in
  FLOWVPM `test/runtests_gpu_fmm.jl`: sign-flip repro, cap, floor,
  bit-exact empty guard, scalar/broadcast equivalence, unknown-key
  throw.

## Trial 1 — cap + absolute floor at 1% sigma_shed (Ryan-ruled 2026-08-26)

Config: `SIGMA_DTZ_CAP=0.5 SIGMA_FLOOR_FRAC=0.01`
(→ `sigma_guard = (dtz_cap=0.5, floor=4.45e-5 m)`).

Rationale: dt·Z ≤ 0.5 makes a sign flip impossible while letting
genuine contraction proceed at a bounded rate (only the outliers ever
exceeded 0.13); the 1%·σ_shed floor additionally stops slow grinds to
zero. Expected fingerprint impact: none before the first cap/floor
engagement (guard OFF = bit-exact; engaged guards touch only the rare
strained outliers, first observed ~step 991+ far above the mature-gate
window ≤755).

Status: implemented and locally VERIFIED (FLOWVPM gpu_fmm 244/244 incl.
new sigma_guard testset 9/9; FLOWPanel warmstart 153/153, simulate
199/199). Deployed to the h200 trees; chain job **13501691** submitted
(eng/qos=eng, 6 h): stage 1 = 36-step mature gate with guard ARMED
(expect bit-identical gates: guard cannot engage there if min_sigma
stays > floor and dt·Z < 0.5 — check the log banner + gate diffs),
stage 2 (gated on 1) = 1080-step acceptance `fm052d_gpu_1080_t1` with
guard armed (unguarded baseline run dir preserved). Record here: crossing behavior (expect min_sigma
plateaus ≥ 4.45e-5), max_u tail, CT/Γ gate deltas vs the locked
tolerances, count-gate deltas, and whether any late-run monitor crash
recurs.

Results: (pending)

## Candidate future trials (not ruled)

- Different cap values (0.2 / 0.8) or floor fractions (5%, 10%).
- Per-particle clip of Z's SFS contribution instead of the composite
  dt·Z cap.
- Merge-policy interaction: identify whether the strained outliers are
  merge-born (add provenance print at guard engagement) and, if so,
  tighten `MERGE_R_FACTOR` / merge acceptance instead of (or with) the
  integrator guard.
- Guard inside `rungekutta3`/`euler_exp` if a future config needs them.
