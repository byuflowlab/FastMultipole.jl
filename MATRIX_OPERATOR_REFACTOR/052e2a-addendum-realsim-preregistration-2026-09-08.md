# 052e.2a continuation addendum pre-registration — real-simulation cross-formulation convergence test (2026-09-08)

**Status: LOCKED (Ryan, 2026-09-09).** L1–L3 rulings and the L4 pre-lock
verification findings are recorded in the "Lock record" section at the
bottom; no value below may be edited after first results are viewed.
Supersedes the same-day stages-2–4 draft
(`052e2a-addendum-stages234-preregistration-2026-09-08.md`, never locked)
per Ryan's 2026-09-08 rulings: the manufactured disk/ring decomposition is
replaced by a real lifting-wing simulation compared across three solve
formulations, with the doublet-panel wake's directly evaluated potential as
the reconstruction oracle. Once locked, no value below may be edited after
first results are viewed; ill-posed fixtures/gates are superseded by a new
dated pre-registration with reason recorded (Tier 0A precedent).

## Purpose and registered predictions

Both `VelocityThroughSources` (production 052b) and the Neumann formulation
transfer wake influence through induced velocity only — but for Neumann
that is the complete boundary data (no-flow-through is a velocity
condition), while for the Dirichlet formulation the wake's potential trace
on the body is independent boundary information that velocity-only transfer
drops. `GreenReconstruction` (built by this item: 052e.1 formulation,
052e.2b production Householder `:area_mean` solve) restores that trace by
reconstructing it from sampled wake velocities via $(I-B)q = S\sigma$,
leaving sources freestream-only and letting the doublets compensate the
wake-induced potential.

**Registered predictions (Ryan 2026-09-08):**

- **P1** — the Neumann circulation (trailing-edge doublet-strength
  difference) matches the `GreenReconstruction` circulation, up to
  body-discretization and the small thick-uncapped vs thick-capped geometry
  difference, with the gap shrinking/plateauing under mesh refinement;
- **P2** — the `VelocityThroughSources` circulation does **not** match the
  Neumann referee, with a gap that persists under mesh refinement;
- **P3** — the trace reconstructed from the doublet-panel wake's velocities
  matches the directly evaluated (true) panel-wake potential trace at
  body-discretization level, converging under refinement (the in-solve
  analogue of the Tier 0B stage-1 result);
- **P4** — the trace reconstructed from the vortex-particle wake under the
  identical scheme is very close to the doublet-wake trace (P3's), the
  difference being wake-representation error only.

Diagnosis of any prediction failure is deliberately deferred until results
exist (Ryan 2026-09-08); this file registers the predictions and the
recorded diagnostics, not failure-mode machinery.

## Fixture

- **Geometry:** capped NACA0012 rectangular wing of the stage-1 family
  ($b=2.7$ m, $c=0.76$ m, `build_pitching_wing_body`, `thickness=0.12`,
  triangular, watertight for Dirichlet). **AOA = 30°** (moderate-to-high,
  so significant circulation is shed). Freestream $|U_\infty| = 1$
  (PROPOSED; nondimensional scale, direction along $+\hat x$ with the wing
  pitched, matching the pitching-wing example convention — exact
  realization recorded in the results file).
- **Body meshes (RATIFIED L1):** stage-1 levels L1–L4 (1,744 / 3,816 /
  8,960 / 19,384 panels), **all four required** (dense route states ≈ 3 GB
  each at L4).
- **Neumann referee body:** the **same thick geometry without endcaps**
  (`caps=false` ⇒ non-watertight, avoiding the rank-deficient
  watertight-Neumann case; `FLOWPanel_solver.jl` explicitly supports this
  configuration), `DBC=false`, **doublet panels only**. Construction path
  (per L4 verification): `build_pitching_wing_body` hardcodes
  `caps=true`/watertight, so the referee body is built from
  `pitching_wing_mesh(...; caps=false)` plus the direct
  `RigidWakeBody{ConstantDoublet,1,TF,false}(...; watertight=false)`
  constructor (verified working; rank-deficiency warning does not fire). Geometrically
  near-identical to the Dirichlet body, so the referee gap in P1 is small
  by construction.
- **Kutta condition — identical for every solve in this item:** the small
  rigid attached trailing-edge upper/lower transition panels implicitly
  imposing the Kutta condition inside the solve —
  `RigidTransitionAttachment` + `JumpKutta` (the default pair, supported by
  every formulation and both BC types). The VTS-only Route B
  (`TEAnchoredAttachment`/pressure-continuity) machinery is **not** used.
- **Solver:** `Backslash` (dense) everywhere; production defaults
  otherwise. Outer/multibody convergence hard-fail ON
  (`require_outer_convergence=true` where the formulation exposes it) so
  nonconvergence cannot imitate formulation error. Host-only, laptop, ≤ 4
  threads, formulation-proof tier (not an official campaign).

## Routes (the three systems under comparison)

| ID | Body | Formulation | Wake→body transfer |
|---|---|---|---|
| R-VTS | Dirichlet, thick capped | `VelocityThroughSources` | velocity → sources |
| R-GR | Dirichlet, thick capped | `GreenReconstruction(gauge=:area_mean)` (production Householder; multiplier via `_green_lambda`) | velocity → reconstructed potential trace → doublet RHS; sources freestream-only |
| R-NEU | Neumann, thick uncapped, doublets only | Neumann solve | velocity (complete BC data) |

## Phases

**Phase A — prescribed wake (isolation).** Identical frozen wake geometry
for all three routes: flat wake along the freestream direction, length
$20c$ (stage-1 convention), wake strengths coupled to the solve through
the standard rigid-wake/Kutta machinery. The harness verifies (hash of
wake nodes) that all routes consume bit-identical wake geometry. One
converged solve per route per mesh level. **API realization (per L4
verification):** `steady!` accepts no wake systems and no `formulation`
kwarg, so Phase A is expressed as the **first step of `simulate!`** with a
pre-built `PanelWake` (nodes/strengths/`nwakes[]` set directly,
`update_TE!` first; the solve precedes wake convection/shedding in the
step loop), with `t_range` of ≥ 2 samples. Wake-node hashes were verified
bit-identical across all three routes including the uncapped Neumann body.

**Phase B — free wake (the R1 convergence claim).** `simulate!` with a
convecting free wake per route, same time-step/convection settings
(PROPOSED: production defaults; exact values recorded pre-launch in the
harness header and results file), run to a steady or statistically steady
circulation. Routes evolve their own wakes; the claim under test is
convergence to the same solution.

**Wake-representation arms (within each phase where supported):**

- **W1 — doublet-panel wake** (`PanelWake`): all three routes.
- **W2 — vortex-particle wake** (production PanelParticle shedding,
  production/default regularization — Ryan R6; no kernel sweeps): **all
  three routes** (Ryan L2: R-VTS required; the L4 verification pass
  confirmed the production VTS path consumes the particle wake unchanged).
  W2 runs at least in Phase B (particles require convection); a
  frozen-particle Phase A variant (particles placed on the Phase A wake
  surface) is included if harness-feasible, else recorded as not run.
  Note: particles appear only after ≥ `nwakerows` warm-up steps (panel
  rows must overflow the buffer before conversion) — short W2 runs and
  the Phase A variant must budget for this.

## Measurements

**Primary — spanwise circulation:** $\Gamma(y)$ = trailing-edge
doublet-strength difference per spanwise station, every route, level, arm,
phase; pairwise gaps $\Gamma_{\rm GR}-\Gamma_{\rm NEU}$,
$\Gamma_{\rm VTS}-\Gamma_{\rm NEU}$ vs refinement (P1/P2). Total
circulation and $C_L$ as scalar summaries.

**Oracle trace check (P3, W1, R-GR):** at the converged solve, compare the
reconstructed trace $\hat q$ against the directly evaluated panel-wake
potential $q_{\rm ref}$ at the same control points, using the stage-1
locked metric definitions (area-weighted gauge alignment, $E_q$,
$E_\infty$; no TE-row exclusion variant needed — both reported).

**Particle-trace comparison (P4, W2 vs W1, R-GR):** aligned trace
difference $\mathrm{rms}_A(\tilde q^{W2}-\tilde q^{W1})/
\mathrm{rms}_A(\tilde q^{W1})$ at matching state (Phase A variant if run;
otherwise matched Phase B snapshot, matching protocol recorded).

**Recorded diagnostics (every solve):**

- Green-route telemetry: `_green_lambda`, Green residual, gauge defect,
  flux compatibility $F$ with its scale. Per L4 verification:
  `GreenReconstructionState` does not carry residual/defect fields, so the
  harness computes these itself (tier0b pattern), capturing per-step state
  via `simulate!`'s `step_telemetry_callback`; `_green_lambda` covers the
  multiplier. `require_outer_convergence` is a VTS-only knob, consistent
  with "where the formulation exposes it";
- post-solve Kutta residual per route (TE jump vs attached transition-panel
  strength consistency), evidencing identical closure;
- outer-iteration convergence history;
- Phase A wake-geometry hash per route.

*(An off-body velocity survey line was proposed in the draft and dropped
at lock — Ryan L3, 2026-09-09: $\Gamma(y)$ is spanwise-resolved and
already localizes route disagreement along the span; with diagnosis
deferred, the off-body field probe adds no needed information.)*

## Gates (deliberately minimal — Ryan R4: predictions over gate tables)

| ID | Check | Gate |
|---|---|---|
| G1 | Every solve converges (outer hard-fail, solver success) | required |
| G2 | Phase A wake hashes identical across routes, per level | required |
| G3 | R-GR telemetry: Green residual ≤ 1e-10; gauge defect ≤ 1e-11 | required |
| G4 | P3 oracle check: $E_q(\hat q, q_{\rm ref})$ ≤ 1e-2 at finest run level, monotone under refinement | required |
| G5 | P1/P2/P4 | **recorded evidence, no numeric gate** — ruling is Ryan's on the tables/figures |

Gauge-defect at 1e-11 (not 1e-12) per the stage-1 B5/C2 roundoff
adjudication. No flux-monotonicity clause (stage-1 B3/C1 lesson). Any G1–G4
failure stops the tier and is reported with raw numbers before any code or
fixture change; no in-place gate retuning.

**Ruling structure:** the addendum ends with an explicit Ryan ruling on
P1–P4 (each: CONFIRMED / REFUTED / INCONCLUSIVE) plus a CONTINUE decision
for 052e.3. No outcome here can retire the stage-1-proven formulation;
refuted predictions trigger diagnosis as a follow-on, not in-place rework.

## Harness

New dated script `scripts/addendum_052e2a_realsim_2026-09-09.jl` (frozen
sha-registered scripts untouched). Reuses stage-1 body construction,
metric, and provenance machinery (note: the stage-1 script reads
`gs.sol_b[end]`; new code must use `_green_lambda`; `set_wake_Das!` lives
in `examples/pitching_wing.jl`, not the module). The pre-lock verification
pass (mechanical only, no registered values) was completed 2026-09-09 —
see the Lock record below. Unregistered smoke mode included (stage-1
precedent).

## Outputs and provenance

- `052e2a-addendum-realsim-results-<date>.md` — Γ(y) tables/figures,
  pairwise-gap refinement tables, oracle-check table, P1–P4 evidence
  summary, call chains, ruling recommendation.
- CSVs under `data/052e2a-addendum-realsim/`; `gates.txt` snapshot with
  FLOWPanel + FastMultipole SHAs, dirty state, tracked-diff hashes, thread
  count, script SHA-256 (Tier 0A standard). Threads ≤ 4; registered run
  nohup-detached; logs under the same data dir.

## Lock record (Ryan, 2026-09-09)

- **L1 — RATIFIED, amended:** AOA = 30°, $|U_\infty|=1$; mesh levels
  **L1–L4 all required** (L4 no longer optional).
- **L2 — RATIFIED, amended:** Phase B policy as drafted (production
  defaults, recorded pre-launch); W2 arm **must include R-VTS** (its
  conditional inclusion resolved affirmatively by the L4 verification:
  production VTS consumes the particle wake unchanged). Frozen-particle
  Phase A variant if harness-feasible, else recorded as not run.
- **L3 — RATIFIED, amended:** off-body survey line **dropped** ($\Gamma(y)$
  already localizes disagreement spanwise; diagnosis deferred). G1–G4
  stand as the only hard gates; P1/P2/P4 recorded evidence only (G5).
- **L4 — verification pass completed 2026-09-09, all checks PASS** (probe
  scripts/logs in session scratchpad; mechanical only, no registered
  values):
  - A: `GreenReconstruction(gauge=:area_mean)` accepts the full setup via
    `simulate!`; production Householder state; `_green_lambda` accessible.
  - B: Neumann uncapped referee solves with the default Kutta pair; rank-
    deficiency warning does not fire. Construction-path mismatch absorbed
    into the Fixture section (direct constructor, not
    `build_pitching_wing_body`).
  - C: particle wake feeds the GR σ path (`u_prewake` populated, σ finite
    and nonzero, reconstruction every step); warm-up latency ≥ `nwakerows`
    steps noted in the W2 arm. Follow-up C2: R-VTS + particle wake PASS.
  - D: prescribed-wake solve expressible for all three routes as the first
    step of `simulate!` (`steady!` cannot host Phase A — mismatch absorbed
    into the Phase A section); wake-node hashes bit-identical across
    routes.
  - Telemetry: G3 quantities computed by the harness itself via
    `step_telemetry_callback` (absorbed into Recorded diagnostics).
