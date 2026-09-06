# 052e accuracy plan v2 revised draft (2026-09-05) — Green-trace ground-truth verification

**Supersedes** `052e-accuracy-plan-draft-2026-09-03.md`. This revision
incorporates the 2026-09-05 technical review; it is not yet ratified. Nothing
in this file authorizes runs, deployment, or promotion. Numerical gates marked
PROPOSED must be locked before results are inspected.

## 0. Preliminary theory gate — velocity to potential trace

Before fixtures or thresholds are ratified, review and accept
`052e-theory-velocity-to-potential-trace.md`. It derives the actual operation
performed by `HybridWakePotential`: velocity normal to a closed body supplies
Neumann data for a body-interior harmonic problem, and Green's identity maps
that data to a boundary trace through

$$
\boxed{(I-B)q=S\sigma.}
$$

The theory gate must confirm the sign conventions, flux compatibility,
constant nullspace and implemented gauge constraints, body-local topology,
and the treatment of noncompact-core vorticity leakage. No accuracy run may
be interpreted until this gate passes.

## 1. Scope and correction to the oracle

The v1 plan compared Hybrid with VTS. That can establish agreement between two
formulations, but not correctness. A directly evaluable doublet wake remains a
useful reference, but the production hybrid algorithm must be tested as it is
implemented:

- `HybridWakePotential` samples particle-induced velocity at the control points
  of each closed Dirichlet body and reconstructs a body trace with
  $(I-B)q=S\sigma$. It does not reconstruct potential at arbitrary wake-only
  field points.
- Particle wakes provide velocity, not a topology-aware scalar potential.
  Consequently the current implementation cannot evaluate a far-field
  particle-equivalent potential or impose a zero-at-infinity gauge.
- On a simply connected body whose closure is free of wake vorticity and
  singular support, the required interior harmonic trace is single-valued.
  The wake branch surface must remain outside the body; a wake-sheet jump is
  therefore not a hybrid acceptance gate.
- A constant doublet patch equals its boundary vortex filament only in the
  singular continuum limit. A finite collection of regularized particles is
  an approximation to that wake, not the same wake.

The error decomposition is therefore body-panel Green discretization,
filament/particle discretization, regularization bias, and the subtraction
used to isolate particle velocity from total-minus-retained-panel influence.
Path-based Hodge reconstruction remains a diagnostic and is not an error
source in the hybrid solve.

## 2. Analytic precondition and gauge policy

For every Dirichlet body, the singular filament or compact vorticity support
must be disjoint from the closure of the body. For a noncompact core such as a
Gaussian, replace geometric disjointness by a pre-registered upper bound on
vorticity leakage into the body and report the evaluated bound. Requiring only
that individual comparison points lie three core radii away is insufficient.

**Ryan ruling 2026-09-06 (Tier-1 disjointness):** the rigid trailing-edge
wake panels used to satisfy the Kutta condition are retained panels, not
particles; they buffer the body from the particle wake, so particle support
begins downstream of the rigid-panel buffer and the disjointness/leakage
precondition is satisfied by construction in Tier 1 and in production.
Still evaluate and report the leakage bound for near-limit cases.

The reconstructed trace is accepted modulo one constant per connected body.
Both implemented gauges are tested directly:

- `:area_mean`: the area-weighted mean of each body's trace is zero;
- `:lsq`: the dense least-squares route with the same area constraint.

All reference traces are aligned independently on each body by subtracting
their area-weighted means. Adding an arbitrary constant to each body trace
must leave paired-edge circulation, exterior velocity, and integrated loads
unchanged. For any accepted route in which body strengths drive an attached or
shed wake, require paired shedding edges and verify that the corresponding
upper-minus-lower map satisfies $C\mathbf 1=0$. `HybridWakePotential` does not
call `_apply_kutta_map!` explicitly, but the ordinary body operator and
downstream shedding apply the equivalent strength difference implicitly.
Its initializer currently performs no unpaired-edge validation. Add a hard
initialization error before certification. `TraceCorrected`,
`GreenReconstruction`, and `DirectWakePotential` currently reach only the
unpaired-edge warning in `_validate_formulation_common`; a warning is likewise
insufficient for any of those routes if certified.

A zero-at-infinity gauge is a separate, currently unavailable capability. It
must not be inferred from velocity-only particle data. Promotion under this
plan is limited to gauge-invariant quantities: circulation, exterior velocity,
and integrated loads. Absolute $C_p$, unsteady pressure, and acoustics remain
blocked until a production-capable, topology-aware global gauge-recovery
method is designed and verified.

If such an anchor is implemented later, its finite-point test is

$$
\frac{|\varphi_h(x_a,t)-\varphi_{\rm ref}(x_a,t)|}{|\mu|},
$$

not $|\varphi_h(x_a,t)|/|\mu|$: potential at a finite far-field point is
generally small, not zero.

## 3. Common metrics and pre-registration

For body control-point values define

$$
\langle q\rangle_A=\frac{\sum_i A_iq_i}{\sum_i A_i},\qquad
\|q\|_A=\left(\frac{\sum_i A_iq_i^2}{\sum_i A_i}\right)^{1/2}.
$$

Every reported "relative RMS" trace error means

$$
E_q=
\frac{\|(q_h-\langle q_h\rangle_A)
 -(q_{\rm ref}-\langle q_{\rm ref}\rangle_A)\|_A}
{\max(\|q_{\rm ref}-\langle q_{\rm ref}\rangle_A\|_A,\epsilon_q)},
$$

where the absolute fallback $\epsilon_q=c_q|\mu|$ and $c_q$ are declared
before running. Vector errors use the analogous global norm, not averages of
pointwise relative errors. Tangential velocity is computed with
$P_t=I-nn^T$.

Each campaign reports weighted RMS and a pre-registered 95th percentile (or a
pre-registered maximum where already named). Before results are viewed, lock
all point sets, exclusion distances, body and wake meshes, particle phases,
core models/ratios, reference uncertainty limits, fallback scales, and numeric
ceilings. Endpoint thresholds are necessary but insufficient: every sweep
must also show refinement behavior consistent with its dominant error source.

## 4. Tier 0A — standalone kernel convention test

This test does not gate the hybrid reconstruction. For a circular disk with
normal $+\hat z$ and the convention
$\varphi=-\mu\Omega/(4\pi)$, compare analytic solid angle with the direct
doublet implementation on both sides of the disk and away from its rim.

Use the zero-at-infinity on-axis branch

$$
\Omega(z)=2\pi\left(\operatorname{sgn}(z)
-\frac{z}{\sqrt{z^2+R^2}}\right),\qquad z\ne0.
$$

With traces taken upper minus lower, this convention gives
$[\Omega]=4\pi$ and $[\varphi]=-\mu$. Reversing the panel orientation reverses
both signs. Verify that convention, linear strength scaling, both on-axis
branches, far-field decay on both sides, circulation normalization, and the
$4\pi$ solid-angle branch ambiguity. Sample neither the sheet nor its rim.

## 5. Tier 0B — manufactured Green-trace oracle

Use a closed, simply connected test body and an external vortex ring/doublet
disk. Its singular support, or its effective regularized core under the
leakage criterion in section 2, remains outside the body for every case.
Choose one continuous potential branch over the whole body surface. Directly
evaluate the reference trace $q_{\rm ref}$ and reference normal velocity at
the same body control points, then run the production reconstruction
$(I-B)q=S\sigma$.

Decompose the error sequentially, changing one representation at a time:

1. analytic/reference normal velocity;
2. a finely integrated singular filament with demonstrated quadrature error;
3. discretized singular filament elements;
4. regularized particles;
5. production total-minus-retained-panel particle-velocity extraction.

Sweep body-panel and filament/particle resolution independently. Sweep core
ratio and at least two azimuthal particle offsets. Run both `:area_mean` and
`:lsq` gauges. Report convergence rates and, for every case:

- flux compatibility $|\sum_i A_i\sigma_i|$, with its declared scale;
- the bordered-system Lagrange multiplier where applicable;
- Green linear residual and area-weighted gauge defect;
- Hodge tangential-projection defect and gauge-aligned Green/Hodge mismatch;
- gauge-aligned trace weighted RMS and registered tail statistic.

The Green residual is solver telemetry, not an accuracy measure: a factored
discrete system can be solved accurately while representing the wrong
continuum trace.

**PROPOSED engineering target:** finest-mesh trace error $E_q\le1\%$, plus
refinement evidence. **Structural kill rule:** recommend retirement only if
the analytic-normal-velocity reconstruction exceeds 20% on the finest body
mesh or fails to improve under body-mesh refinement. Particle error above 20%
that improves with refinement is a discretization finding, not structural
failure.

## 6. Tier 1 — prescribed flat wake and body solve

Use a rectangular wing (AR 5) with a prescribed, paired flat wake extending
approximately 20 chords. Define a matched discrete wake: compare direct
constant-panel potential on the exact panel sheet with the exact
panel-to-particle conversion being assessed. Use identical body mesh and
solver settings; vary only the wake representation.

G1a is the area-weighted, independently gauge-aligned error in the isolated
wake trace `q_total` before the body solve. Also compare body doublet strengths,
paired-edge circulation, projected tangential velocity $P_tU$ at control
points, exterior probe velocity, and integrated loads. Include the per-body
constant-shift invariance test described in section 2 and a fixture with two
Dirichlet bodies, because production reconstructs and gauges them separately.
Exercise the new Hybrid hard validation with a negative unpaired-edge fixture
before running the paired-edge acceptance cases.

| Gate | Quantity | Proposed ceiling | Status |
|---|---|---:|---|
| G1a | isolated `q_total` trace $E_q$ | 1e-2 | NEEDS RYAN |
| G1b | $P_tU$ global relative RMS / max | 1e-2 / 5e-2 | NEEDS RYAN |
| G1c | body strengths, paired circulation, exterior velocity, loads | pre-register after Tier 0B scales are known | NEEDS RYAN |
| G1d | change under independent per-body constant shifts | roundoff-scaled ceiling | NEEDS RYAN |

All endpoint gates additionally require refinement evidence.

## 7. Tier 1.5 — distorted, variable-strength wake

Use a smooth, nonconstant doublet-strength distribution on a prescribed
helical sheet. Include root, tip, and downstream boundary contributions; a
constant-strength sheet reduces mainly to boundary filaments and is not
representative of production conversion.

The reference is either exact analytic constant-panel evaluation on the
identical discrete sheet or an independently over-resolved quadrature whose
demonstrated uncertainty is below one tenth of the acceptance tolerance.
Sweep helical pitch, blade/wake clearance, core ratio, and sheet resolution,
including a case near the permitted particle/body separation limit.

**PROPOSED engineering target:** fixed $E_q\le5\%$, plus refinement evidence.
This ceiling is pre-registered and is not scaled by the observed Tier 0 error.

## 8. Tier 2 — temporal coherence of the reconstructable trace

Translate the ring relative to a fixed closed test body (and fixed inertial
probes where velocity is checked), with `recompute_interval=1`. Keep every
body control point on one continuous potential branch and prevent the chosen
branch surface or effective particle cores from sweeping through the body.

Using the same temporal difference formula as production, compare

$$
\partial_t(q-\langle q\rangle_A)
\quad\text{with}\quad
-W\cdot\nabla q-\langle-W\cdot\nabla q\rangle_A.
$$

Run a $\Delta t$, $\Delta t/2$, $\Delta t/4$ sweep so temporal truncation can
be separated from spatial error. **PROPOSED engineering target:** 2% global
relative RMS at the finest step, plus the expected temporal refinement trend.

No result in this tier certifies lagged reconstruction. Either lock the
certified configuration to `recompute_interval=1`, or test every proposed
production interval separately; lagging creates a staircase trace.

## 9. Stage B — production evidence

Only after Tiers 0B–2 pass, run independent identically configured Hybrid and
VTS rotor cases from the v1 matrix. Add a direct or demonstrably over-resolved
reference for at least one reduced rotor case; Hybrid/VTS agreement alone is
not correctness evidence. Pre-register Stage-B tolerances before those runs,
using Tier results only to choose meaningful scales.

The promotion package must include:

- direct-versus-FMM parity for every route used by the certified configuration;
- performance, retained-memory, convergence, overlap/leakage, and
  no-host-fallback gates already owned by the 052e implementation item;
- complete meshes, phases, core parameters, revision, timing, and memory
  metadata;
- an explicit limitation to gauge-invariant outputs unless the separate
  global-gauge capability has by then passed its own design and verification.

## 10. Execution and ruling order

1. Review and accept `052e-theory-velocity-to-potential-trace.md`.
2. Ratify and pre-register the metrics, fixtures, sweeps, and numerical gates.
3. Run Tier 0A as a kernel convention check.
4. Run Tier 0B in the five diagnostic stages; apply the structural kill rule.
5. Run Tier 1, including the two-body and gauge-invariance fixtures.
6. Run Tier 1.5 and Tier 2.
7. Rule whether to retire, remain experimental, or proceed to Stage B.
8. If Stage B passes, seek a separate promotion ruling. Absolute-pressure
   promotion remains blocked without verified global gauge recovery.

## 11. Subitem mapping (adopted by Ryan 2026-09-06)

The ruling order in section 10 executes as subitems 052e.0–.6; see the
subitem-structure section of
`052e-impl-hybrid-wake-potential-experimental.md` for definitions and the
dependency chain (.0 → .2 → {.3, .4} → .5; .1 feeds .2; .6 parallel).
Gate ratification is per subitem, immediately before that subitem runs.
