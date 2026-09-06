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

Results (2026-09-04, trial-1e = job 13582234, mgh GH200; lineage:
13501691 run 1 passed gates / run 2 died at step 894 on the zero-M2L
cache bug (since fixed by 052f d938ba68 + 052g 2c6dd60f); trial-1b
13569052 cancelled (eng full); trial-1c 13569059 died at 4:36 on the
silo sigma_guard 2-key/3-key skew (ceil support ported into both silo
FLOWVPMs 2026-09-04, minimal hunk — parser + clamp application at both
euler call sites; NOT whole-file, local file has splitting_state
divergence); trial-1d 13582076 ran clean to step 693/1079 then
NODE_FAIL (mgh-1-2 to maint)):

- **COMPLETED, exit 0, 2:58:15**; launcher printed "artifact and monitor
  gates passed for indices 0:1079".
- **Locked correctness gates ALL PASS** (window 720-755, locked
  tolerances):

  | Gate | Measured | Ceiling | Result |
  |---|---:|---:|---|
  | CT cycle-mean | 6.565e-4 | 1.800e-3 | PASS |
  | Gamma M2 max | 1.317e-3 | 2.934e-3 | PASS |
  | Gamma M2 RMS | 4.484e-4 | 1.498e-3 | PASS |

- Guard config as designed: SIGMA_DTZ_CAP=0.5, SIGMA_FLOOR_FRAC=0.01
  (floor 4.451e-5 m), SIGMA_CEIL=Inf (guard=on).
- **min_sigma trajectory (wake-health monitor04)**: monotone contraction
  from sigma_shed 4.451e-3 m to min 9.558e-5 m at step 983
  (min_sigma_ratio 0.0215), recovering to 1.303e-4 m by step 1080. The
  floor (ratio 0.01) never clamped — min stayed ~2.1x above it. No
  sign-flip/collapse: the historical step-~1015 blowup (dt*Z 164) did
  not recur under the cap. max_gamma_over_sigma2 peaked 2.20e4 at step
  951, ending 1.52e4.
- **052f/052g fix validated in production**: mid-run the radix-FMM
  geometry went inadequate (g_min*h_leaf = 0.126 vs rho_t*sigma_max =
  0.1261 at ell=3) and the 052f demotion warning fired, falling back to
  the all-direct zero-M2L geometry at ell=2 (q=27) — exactly the
  scenario that hard-crashed 13501691 run 2 at step 894. Run continued
  healthy through step 1080.
- Phase-2e CT convergence: CONVERGED=false (spread 0.0209 vs tol 0.005;
  within-rev p-p/mean 0.111 vs 0.02) but CYCLE-MEAN CT = 0.072526 ±
  1.04% over 10 revs, with middle blocks (revs 3-8 of the window) tight
  at 0.2-0.8% dev — the known pre-existing, non-fatal readout item
  (same class as 13501691 run 1), flagged to Ryan, not a gate.
- Reference context: ct_reference_cycle_mean 0.0712209;
  ct_per_step_max_abs_difference 1.011e-4. Gate report:
  orc `~/FLOWPanel-052-gh200/data/fm052c_mature_gate_t1_13582234/fm052_gate.md`;
  run dir `.../data/fm052d_gpu_1080_t1/` (shared data root).

**Trial 1 verdict: PASS.** The dtz_cap=0.5 + floor=1% guard pair keeps
the 1080-step acceptance alive with locked gates passing and no
fingerprint impact in the mature-gate window. Commit-plan proposal
(needs Ryan): (1) commit the silo ceil port upstream (it is already the
local FLOWVPM state; orc silos now match on the guard hunk); (2) adopt
dtz_cap=0.5 + floor_frac=0.01 as the 052c-recommended default for GPU
rotor acceptance runs (ceil=Inf until a binding value is motivated);
(3) fold the candidate-trials list below into 053 row planning rather
than running more OFAT now.

## Candidate future trials (not ruled)

- Different cap values (0.2 / 0.8) or floor fractions (5%, 10%).
- Per-particle clip of Z's SFS contribution instead of the composite
  dt·Z cap.
- Merge-policy interaction: identify whether the strained outliers are
  merge-born (add provenance print at guard engagement) and, if so,
  tighten `MERGE_R_FACTOR` / merge acceptance instead of (or with) the
  integrator guard.
- Guard inside `rungekutta3`/`euler_exp` if a future config needs them.

## Trial 2 — exponential integrator, no clamps (Ryan-ruled 2026-09-05)

Hypothesis: the 026 frozen-gradient geometric integrator (`euler_exp`,
sigma > 0 by construction for any finite gradient/timestep) resolves the
step-~1015 sigma collapse WITHOUT the trial-1 cap/floor guards.

Setup (job 13592503, mgh-1-1 GH200, submitted 2026-09-05):
- `WAKE_EXPINT=true`, SIGMA_* unset (guard=off; `euler_exp` rejects a
  non-empty sigma_guard by design).
- Silo port required first: the -gh200 silo FLOWVPM predated 026 Phase 1b
  Task 1 (commit 8b00dbd, GPU/broadcast path for euler_exp). Ported
  2026-09-05 as a targeted patch: timeintegration hunks applied clean;
  the viscous CoreSpreading isa-Array fork was hand-applied because the
  silo lacks the 75a55d7 splitting_state dsigma2 accumulators (same
  divergence class as the ceil port). Backups:
  `~/FLOWVPM-052-gh200/src/{FLOWVPM_timeintegration,FLOWVPM_viscous}.jl.bak-preexp`.
  Patch archived: `MATRIX_OPERATOR_REFACTOR/scripts/fp052c_trial2_expint_gpu_port.patch`.
- Launcher: `~/projects/launchers/fp052c_trial2_expint_gh200_run.sh`
  (archived in `MATRIX_OPERATOR_REFACTOR/scripts/`). Two stages as trial-1,
  EXCEPT the stage-1 mature gate vs the pinned CPU EULER reference is
  INFORMATIONAL (a different integrator legitimately shifts the
  fingerprint; recorded, does not abort). GPU-routing source gates fatal.
  Stage-2 run_name `fm052d_gpu_1080_t2exp` (t1 + unguarded baseline preserved).
- Expected failure signature if the instability persists: euler_exp
  broadcast substep-budget throw (dt*|L| bound) or non-finite-ratio
  DomainError — either is informative, not a silent blowup.

Infrastructure (Ryan-ruled 2026-09-05, mid-trial): one checkout, no silos.
- expint GPU port committed to unified `~/projects/FLOWVPM.jl` as 3315b22
  (files were byte-identical to the silo pre-port state).
- Frozen-campaign pattern = git worktrees (wt026 precedent): created
  `~/wt052/{FLOWVPM.jl,FLOWPanel.jl,FastMultipole}` on branch
  `campaign-052` + env `~/wt052/env-aarch64` (Manifest dev-repointed). Pinned
  SHAs: FLOWVPM.jl 3315b22, FLOWPanel.jl 4e6b5b7, FastMultipole 3da58a1a.
- Launcher `~/projects/launchers/fp052c_expint_wt052_run.sh` (archived in
  scripts/): runs from wt052, run dirs + logs symlinked/written into the
  consolidated data root `~/projects/FLOWPanel.jl/data`.
- Trial-2 job 13592503 still runs from the gh200 silo (already in flight;
  tree-identical to trial-1e for comparability) — the LAST silo run. Silo
  retirement (gh200 + h100/h200 triples) queued for after it completes.

Results (2026-09-05, trial-2 = job 13592503, mgh-1-1 GH200):

- **COMPLETED, exit 0, 2:48:20** (vs trial-1e 2:58:15); launcher printed
  "artifact and monitor gates passed for indices 0:1079" and the stage-d
  COMPLETE line; source paths CPU-S=0 GPU-S=1080 backend=0; GPU-S
  cleanup verified.
- **Stage-1 mature gate (informational): PASSED OUTRIGHT** vs the pinned
  CPU euler reference — the expint fingerprint stayed within the locked
  tolerances despite the integrator change, and all three gates are
  TIGHTER than trial-1e:

  | Gate | Trial-2 (expint) | Trial-1e (guards) | Ceiling | Result |
  |---|---:|---:|---:|---|
  | CT cycle-mean | 5.655e-4 | 6.565e-4 | 1.800e-3 | PASS |
  | Gamma M2 max | 1.134e-3 | 1.317e-3 | 2.934e-3 | PASS |
  | Gamma M2 RMS | 3.331e-4 | 4.484e-4 | 1.498e-3 | PASS |

- Config confirmed in log banner: `WAKE_EXPINT=true`, SIGMA_DTZ_CAP=Inf,
  SIGMA_FLOOR_FRAC=0.0, SIGMA_CEIL=Inf (**guard=off**).
- **Hypothesis confirmed — no sigma collapse without clamps.** No
  substep-budget throw, no DomainError, all finite = true.
  min_sigma (wake-health monitor04): bottom **8.872e-5 m at step 1051**
  (ratio 0.0199 of sigma_shed 4.451e-3), 9.16e-5 at the historical
  step-1015 blowup point, ending 9.06e-5 at step 1079 — same ~2%
  contraction plateau trial-1e reached (9.558e-5 at step 983) but held
  by the integrator's positivity instead of a floor.
- max_gamma_over_sigma2 peaked **1.135e5 at step 841** — ~5x trial-1e's
  2.20e4 peak — yet the run stayed stable and the window CT/Gamma gates
  tightened; max_u peak 44.1 at step 478.
- Phase-2e CT convergence: CONVERGED=false (per-rev spread 0.033 vs tol
  0.005, driven by the first window rev 721:756; revs 901–1080 tight at
  0.0012–0.0018), CYCLE-MEAN CT = **0.0732227 ± 1.54%** over 10 revs
  (trial-1e: 0.072526 ± 1.04%) — same known non-fatal readout item.
- Artifacts: gate dir `~/FLOWPanel-052-gh200/data/fm052c_mature_gate_t2exp_13592503/`,
  run dir `.../data/fm052d_gpu_1080_t2exp/` (shared data root),
  log `.../data/fm052d_gpu_1080_t2exp.log`.

**Trial 2 verdict: PASS.** euler_exp alone (no guards) survives the full
1080-step acceptance with all locked gates passing and a cleaner
mature-window fingerprint than the guarded euler of trial-1e. Both
candidate defaults are now validated; decision (expint vs guards vs
both) goes to Ryan per the open-decisions list.

## Defaults ruling (Ryan, 2026-09-05)

Ryan approved the trial-2 recommendation:

1. **expint (`euler_exp`, `WAKE_EXPINT=true`, no sigma guards) is the
   052c default** for GPU rotor acceptance runs — tighter gates than
   guarded euler, no tuned clamp parameters, loud failure modes.
2. **Trial-1 guards (dtz_cap=0.5 + floor_frac=0.01) are the documented
   fallback** for configs where euler_exp's constraints don't hold
   (requires ReformulatedVPM f==0; rejects non-empty sigma_guard).
3. Upstream ports already committed in unified FLOWVPM: expint GPU path
   3315b22; sigma_guard :ceil 6c8cda4. Nothing further to port.
4. Operationally in force: the wt052 launcher
   `fp052c_expint_wt052_run.sh` hardwires `WAKE_EXPINT=true` with
   SIGMA_* unset, and (with the silos retired, below) it is the ONLY
   052c launch path.
5. The candidate-trials OFAT list above is folded into 053 row planning
   (see `053-defaults-enumeration-draft-2026-09-03.md`); no further
   OFAT runs under 052c.

## Silo retirement (executed 2026-09-05, Ryan-approved)

- Campaign data moved from the gh200 silo to the consolidated data root
  `~/projects/FLOWPanel.jl/data/`: all `fm052c_*`/`fm052d_*` trial run
  dirs, gate dirs, and logs (incl. trial-1e `fm052d_gpu_1080_t1`, 29G,
  and trial-2 `fm052d_gpu_1080_t2exp`, 29G). A stale Aug-27 dir of the
  same name in the root was renamed `fm052d_gpu_1080_t1.prev.20260827`
  (148M) rather than clobbered. Silo root logs → 
  `data/retired_052_silos/{gh200,h100,h200}/` (gh200 logs/ +
  slurm-13592503.out; h200 xverify logs).
- Deleted: `~/{FLOWVPM,FLOWPanel,FastMultipole}-052-{gh200,h100,h200}`
  (9 checkouts, git-clean) and `~/fm052env-{052d,b200,gh200,h100,h200,l40s}`.
- Kept: `~/fm052depot-gh200` (6.8G — wt052 launcher's JULIA_DEPOT_PATH)
  and `~/wt052/` worktrees.
- NOT deleted (older 052 debris outside the ruling, flagged for a later
  sweep): `~/{FastMultipole-052d,FastMultipole-052h,FLOWPanel-052,
  FLOWPanel-052d,FLOWPanel-052h-spot,snapshot472-052d}`, envs
  `~/{fm052env,fm052henv,fm052spotenv,fm052env_cuda63_geoiofree,
  fm052env_sfsdiag}`, `~/fm052a_env_dumps`, and loose
  `fm052*/fp052*` debug `.out`/`.jl` files + archiver logs in `~`.

## Debris sweep + campaign tags (2026-09-05, Ryan-approved)

- Older 052 debris deleted from orc `~`: checkouts `FastMultipole-052d`,
  `FastMultipole-052h`, `FLOWPanel-052`, `FLOWPanel-052d`,
  `FLOWPanel-052h-spot`, `snapshot472-052d` (all verified to contain no
  run data); envs `fm052env`, `fm052henv`, `fm052spotenv`,
  `fm052env_cuda63_geoiofree`, `fm052env_sfsdiag`, `fm052a_env_dumps`;
  loose `fm052*`/`fp052*` debug `.out`/`.jl` files, 052 archiver logs,
  `install_052b_pin.sh`, `launch_052b.sh`, `patch_052{f,g}.py`,
  `instantiate_052g.log`, `fp052d_probe_state`. Only `~/fm052depot-gh200`
  and `~/wt052` remain of the 052 family in `~`.
- wt052 pins now cited per the annotated-tag policy — tag
  `campaign/052-expint-20260905` created in each orc unified repo:
  FLOWVPM.jl (3315b22), FLOWPanel.jl (4e6b5b7), FastMultipole (3da58a1a).
