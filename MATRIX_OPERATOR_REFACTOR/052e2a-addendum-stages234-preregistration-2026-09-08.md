# 052e.2a continuation addendum pre-registration — stages 2–4 (2026-09-08)

**Status: SUPERSEDED (same day, pre-lock) by
`052e2a-addendum-realsim-preregistration-2026-09-08.md`.** Ryan's 2026-09-08
rulings replaced the manufactured disk/ring decomposition below with a
real-simulation cross-formulation convergence test (R1/R4: "too
complicated"; see the superseding file). Kept as the record of the
considered-and-rejected design. Nothing below was ever locked or run.

**Status: DRAFT — NOT LOCKED. No registered values may be produced before
Ryan locks this file.** Companion to the stage-1 formulation proof
(`052e2a-tier0b-preregistration-2026-09-07.md`, LOCKED; ruled PASS
2026-09-07). Registered scheduling authority: accuracy plan
`052e-accuracy-plan-v2-draft-2026-09-05.md` §5 (stages 2–4 "run as a
052e.2a continuation addendum on the same fixture after the stage-1
ruling"). Once locked, no value below may be edited after first results are
viewed; ill-posed fixtures/gates are superseded by a new dated
pre-registration with reason recorded (Tier 0A precedent).

## Scope

Stages 2–4 of the plan-§5 error decomposition, changing one wake
representation at a time and measuring the effect on the reconstructed
gauge-aligned Green trace:

2. a finely integrated singular filament with demonstrated quadrature error;
3. discretized singular filament elements;
4. regularized particles.

Stage 5 (total-minus-retained extraction) folds into 052e.3 and is out of
scope. Host-only, laptop-scale, ≤ 4 threads, formulation-proof tier (not an
official campaign).

**System under test — deviation-by-supersession from plan §5 wording
(requires ratification, item R2):** plan §5 says "run the bordered
`:area_mean` route", but 052e.2b (Tier 0B-R, ADOPT ruling 2026-09-07) made
the implicit-Householder reduction the production `:area_mean` route with
roundoff-scale parity certified against the bordered reference. The
addendum runs the **production** route:
`_source_potential!` → `_build_green_solve_state(body,:area_mean)`
(→ `GreenHouseholderState`) → `_green_solve_q!`, multiplier via
`_green_lambda(gs)`. The bordered route stays available as
`:area_mean_bordered` for debugging only; no parity re-litigation here.

## Fixture

**Body (unchanged from stage 1):** capped NACA0012 wing, $b=2.7$ m,
$c=0.76$ m, `build_pitching_wing_body`, `thickness=0.12`, triangular,
watertight; refinement family L1–L4 = 1,744 / 3,816 / 8,960 / 19,384
panels exactly as enumerated in the stage-1 prereg.

**Oracle — doublet-disk / vortex-ring (requires ratification, item R1):**
stages 2–4 need a wake object whose filament equivalent is *curved*: the
stage-1 flat wake's equivalent filament lattice is straight lines, and
subdividing a straight singular filament into straight elements is exact,
so stage 3 would be vacuous on it. Plan §5 explicitly includes "the
vortex-ring/doublet-disk fixture"; the addendum adopts it as the stages-2–4
oracle over the same body family:

- **Disk:** flat circular constant-doublet disk, radius $R=c=0.76$ m,
  strength $\mu_0=1$ (secondary $\mu_0=2$ for linearity), triangulated
  (exact for constant doublets by solid-angle additivity) with a registered
  fan/annulus triangulation fine enough that disk-triangulation error is
  not at play (the disk *is* the reference object, as in stage 1: its
  discrete triangle set is frozen and shared by every stage's baseline).
- **Cases:**
  - **D1:** disk center $(3c,\,0,\,1.5c)$, normal $+\hat z$ (spanwise
    symmetric — the C1 analogue).
  - **D2:** disk center $(3c,\,0.3b,\,1.5c)$, normal $\hat z$ rotated
    $30°$ about $+\hat x$ (symmetry broken — the C2 analogue).
- **Support condition:** the harness verifies strictly positive clearance
  from every body control point (and, stage 4, from every particle center)
  to the disk, and records the minimum (expected ≥ $1c$ by construction).
- **Ring equivalence:** the exact filament equivalent of the constant-μ
  disk is the single singular vortex ring on its rim, $\Gamma=\mu_0$,
  right-handed about the disk normal. Sign/orientation is not assumed: gate
  S2-V below *is* the convention check.

**Stage-1-style baseline S1′ (new runs, stage-1 method):** direct disk
$q_{\rm ref}$ and $\sigma$ via the Tier 0A-validated `pnl.induced` wrapper
route, production reconstruction, all stage-1 metric machinery. S1′ anchors
the decomposition deltas on this oracle; it re-proves nothing.

**Sweep placement (requires ratification, item R3):**

- S1′ and S2 run on all body levels L1–L4 (body-refinement sweep).
- S3 and S4 representation sweeps run at fixed body level **L3** (8,960
  panels), reusing one factorized Green state across all wake
  representations. Rationale: stages 3–4 are gated on *deltas between wake
  representations at a fixed body*, which do not require the 3 GB L4 state;
  body-refinement evidence is carried by S1′/S2.

## Stage definitions

**S2 — finely integrated singular ring.** Velocity of the singular ring by
composite Gauss–Legendre quadrature of the Biot–Savart integrand over the
parametrized circle; panel count doubled until the maximum relative
velocity change over all control points is ≤ 1e-13, with the last doubling
reported as the **demonstrated quadrature error**. Then
$\sigma_i=-n_i\cdot u_{\rm ring}(x_i)$ through the production
reconstruction.

**S3 — discretized filament elements.** Regular $n_e$-gon inscribed in the
ring; closed-form singular straight-segment Biot–Savart per element.
Registered sweep: $n_e \in \{8, 16, 32, 64, 128, 256\}$.

**S4 — regularized particles.** $n_p$ particles on the ring,
$\boldsymbol\alpha_k=\Gamma\,\Delta s\,\hat t_k$ at arc positions
$s_k=(k-1+\phi)\,\Delta s$, Gaussian (erf) regularization matching
FLOWVPM's production kernel (in-harness implementation, spot cross-checked
against FLOWVPM at ≥ 10 points to ≤ 1e-12; ratification item R6).
Registered sweeps (all combinations):

- particle count $n_p \in \{32, 64, 128, 256, 512\}$;
- core ratio $r_{\rm ov}=\sigma_p/\Delta s \in \{0.5, 1.0, 2.0\}$ (core
  shrinks with refinement at fixed $r_{\rm ov}$);
- azimuthal offsets $\phi \in \{0, 0.5\}$ (plan-§5 minimum of two).

**Leakage bound (plan §2, noncompact core):** for each $(n_p,r_{\rm ov})$,
evaluate and report

$$
L=\max_k\left[\operatorname{erfc}\!\left(\tfrac{d_k}{\sqrt2\,\sigma_p}\right)
+\sqrt{\tfrac{2}{\pi}}\,\tfrac{d_k}{\sigma_p}\,
e^{-d_k^2/2\sigma_p^2}\right],
$$

the exact 3-D Gaussian mass of particle $k$ beyond its minimum distance
$d_k$ to the body surface (a conservative bound on vorticity leakage into
the body).

## Metrics

Stage-1 locked definitions carry over verbatim: gauge alignment
$\tilde q = q - a^Tq/a^T\mathbf 1$; trace error $E_q$ (area-weighted
relative RMS vs the disk-oracle trace); tail $E_\infty$ (all-panel; no
TE-row exclusion — the disk is not TE-attached); flux $F$; Green residual
and gauge defect; Hodge defect/mismatch (L1–L3 only, memory limitation as
recorded in stage 1). New decomposition deltas, on the aligned traces at
fixed body level:

$$
D_s(\cdot)=\mathrm{rms}_A\!\left(\tilde q^{(s)}-\tilde q^{(2)}\right)
/\ \mathrm{rms}_A\!\left(\tilde q_{\rm ref}\right),\qquad s\in\{3,4\},
$$

i.e., each representation is measured against the converged singular ring
(S2), the plan's "one representation at a time".

## Checks and gates (numbers PROPOSED — ratification item R4)

| ID | Check | Gate |
|---|---|---|
| A1 | S1′ trace error $E_q$ finest (L4), both cases | ≤ 1e-2 |
| A2 | S1′/S2 body-refinement of $E_q$ monotone, orders recorded | monotone; not order-gated |
| A3 | S2-V ring-vs-disk velocity parity, rel. L2 over all control points, every level, both cases | ≤ 1e-10 |
| A4 | S2-T trace parity $|E_q^{(2)}-E_q^{(1')}|$, every level | ≤ 1e-8 |
| A5 | S2 demonstrated quadrature error (last doubling) | ≤ 1e-13, reported |
| A6 | S3 $D_3(n_e)$ monotone decreasing; observed order recorded (expect ≈ 2) | monotone; finest $D_3$ ≤ 1e-4 |
| A7 | S4 $D_4(n_p)$ monotone decreasing at each fixed $(r_{\rm ov},\phi)$; orders recorded | monotone |
| A8 | S4 engineering target: best-configuration finest $E_q$ (vs disk oracle) | ≤ 1e-2 |
| A9 | S4 leakage bound $L$ every configuration | ≤ 1e-8 |
| A10 | S4 offset sensitivity $|D_4(\phi{=}0)-D_4(\phi{=}0.5)|$ per $(n_p,r_{\rm ov})$ | recorded, not gated |
| A11 | Telemetry every run: $F$, $\lambda$ (`_green_lambda`), residual ≤ 1e-10, gauge defect ≤ 1e-11, Hodge trends (L1–L3) | residual/defect gated; rest recorded |
| A12 | Linearity $\mu_0=2$ vs $2\times(\mu_0=1)$, finest of each stage | ≤ 1e-12 |
| A13 | Both cases D1, D2 pass all gates | required |

Gauge-defect gate set at 1e-11 (not stage 1's 1e-12): the stage-1 run
showed roundoff accumulation in the area-weighted sum reaching 1.4e-12 at
N≈19k (B5/C2 artifact, adjudicated); 1e-11 keeps the same intent without
re-tripping on known roundoff. Flux monotonicity is **not** gated (the
B3/C1 lesson: monotonicity on machine-zero series is vacuous); $F$ values
are reported with their scale.

**Structural kill rule (plan §5 verbatim in effect):** particle error above
20% that improves with refinement is a discretization finding, not
structural failure. No stage-2–4 outcome can retire the formulation proven
in stage 1; failures here diagnose the *representation* approximations.

**Ruling structure:** the addendum ends with an explicit
**ACCEPT / CONTINUE** ruling by Ryan on the decomposition evidence (no
RETIRE path exists here, per the kill rule). Any gate failure stops the
tier and is reported with raw numbers before any code or fixture change; no
in-place gate retuning.

## Harness

New dated script `scripts/tier0b_052e2a_addendum234_2026-09-08.jl` (the
frozen sha-registered `tier0br_052e2b_householder_parity.jl` is not
touched). Reuses the stage-1 harness's body construction, oracle-body
triangulation, metric, clearance, and provenance machinery; adds ring
quadrature, closed-form segments, regularized particles, and the delta
metrics. `TIER0B_SMOKE=1`-style unregistered mechanical smoke mode
included (stage-1 precedent; smoke values meaningless and uninterpreted).

## Outputs and provenance

- `052e2a-addendum-stages234-results-<date>.md` — per-gate table, per-stage
  delta/refinement tables per case, call chain, ruling recommendation.
- CSVs of all sampled values under `data/052e2a-addendum234/`; `gates.txt`
  snapshot with FLOWPanel + FastMultipole SHAs, dirty state, tracked-diff
  hash, thread count, script SHA-256 (Tier 0A standard). Threads ≤ 4; logs
  preserved under `data/052e2a-addendum234/`; registered run launched
  nohup-detached.

## Open items blocking LOCK (Ryan)

- **R1** Ratify the doublet-disk/vortex-ring oracle (incl. $R$, centers,
  orientations above) as the stages-2–4 vehicle, replacing "same fixture"
  read narrowly as the flat wing wake — reason: straight-filament wake
  makes stage 3 exact by construction. (A wing-wake particle case is
  deliberately deferred to 052e.3, where the production hybrid fixture
  exercises it end-to-end.)
- **R2** Ratify the production Householder `:area_mean` route as system
  under test (supersedes plan-§5 "bordered" wording per the .2b ADOPT).
- **R3** Ratify sweep placement: S1′/S2 on L1–L4; S3/S4 sweeps at fixed L3.
- **R4** Ratify gate numbers A1–A13 (notably: S2 parity 1e-10/1e-8, finest
  $D_3$ ≤ 1e-4, leakage ≤ 1e-8, gauge defect 1e-11).
- **R5** Ratify the S4 sweep grids ($n_p$, $r_{\rm ov}$, $\phi$).
- **R6** Ratify Gaussian-erf (FLOWVPM-matching) as the registered
  regularization, with the in-harness implementation + spot cross-check.
