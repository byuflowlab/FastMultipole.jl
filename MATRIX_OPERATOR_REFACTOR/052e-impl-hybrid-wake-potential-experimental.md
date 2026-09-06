# 052e — Experimental hybrid wake-potential formulation

## Status and ownership

**STAGED `2026-08-27`; not on the 052b acceptance critical path.**

This item owns `HybridWakePotential` after the 052b production pivot to
`VelocityThroughSources`. The API and its existing host regressions remain
available. Do not remove them, and do not optimize or promote the dense
Green/Hodge machinery under 052b.

## Objective

Determine whether the hybrid retained-panel-potential plus particle-trace
reconstruction is accurate, robust, and production-feasible enough to merit a
future promotion ruling. **Reworked 2026-09-05 (Ryan ruling):** the primary
measurement is how accurately the hybrid computes the wake-induced potential
trace on each closed Dirichlet body, verified against external doublet-wake
fixtures with directly computable reference traces (see
`052e-accuracy-plan-v2-draft-2026-09-05.md`). The Hybrid-vs-VTS rotor
comparison is demoted to a contingent later stage; when it runs, compare
independent, identically configured hybrid and VTS runs and never march both
formulations through one mutable state. The implemented reconstruction is
body-local and determines each trace only modulo a constant; test the existing
`:area_mean` and `:lsq` gauges directly. Velocity-only particle data cannot
currently supply a topology-aware zero-at-infinity anchor.

## Existing characterization baseline

- Corrected tuple/block-GS coupling is shared with VTS: frozen Dirichlet
  sources, scalar potential into Dirichlet targets, velocity into Neumann
  targets, normalized physical residuals, and hard convergence failure.
- Small-fixture gauge-aligned trace error is about 10%; particle/panel velocity
  error is roughly 60%. These are characterization bounds, not
  production-accuracy proof.
- Green residual, gauge defect, Green/Hodge mismatch, tangential projection
  defect, circulation, thrust, and torque telemetry already exist and remain
  regression targets.
- The Green/Hodge path reconstruction is diagnostic only; the production
  hybrid trace is obtained from `(I-B)q=Sσ`.
- Each Dirichlet body currently owns a separate dense gauge-fixed Green
  factorization. Sharing is forbidden until exact body-local matrix parity is
  demonstrated independently by handedness and ordered connectivity.
- The analytic basis requires wake singular support and vorticity to remain
  outside each closed body. Noncompact regularization requires a quantified
  leakage tolerance, not only point-to-core separation.
- `HybridWakePotential` initialization currently does not reject or warn on
  unpaired shedding edges. Although its formulation-specific solve does not
  call `_apply_kutta_map!`, the ordinary attached-wake operator and downstream
  shedding use the equivalent upper-minus-lower strength map implicitly. A
  constant strength shift therefore need not cancel on an unpaired edge.

## Work plan

1. Review and accept the velocity-to-potential trace derivation in
   `052e-theory-velocity-to-potential-trace.md`, including compatibility,
   gauge, topology, and leakage assumptions.
2. Re-run the host formulation, convergence, solver, history, simulation,
   warm-start, comparison, and parsing regressions after 052b closes.
3. Add a hard `HybridWakePotential` initialization error for every unpaired
   shedding edge, plus a negative regression. For any other formulation
   proposed for certification, promote the existing unpaired-edge warning in
   `_validate_formulation_common` to a hard error as part of that route's own
   certification work.
4. Tiered ground-truth verification per
   `052e-accuracy-plan-v2-draft-2026-09-05.md`: Tier 0A standalone kernel
   convention test; Tier 0B closed-body Green-trace oracle with sequential
   discretization/regularization/subtraction diagnosis; Tier 1 body solve with
   prescribed matched flat wake, two-body coverage, and gauge-invariance
   checks; Tier 1.5 variable-strength distorted sheet; Tier 2 temporal
   coherence of the area-mean-gauged trace with `recompute_interval=1`. Only
   if Tiers 0B–2 pass: Stage B production-shaped Hybrid-vs-VTS matrix plus a
   direct or over-resolved reduced-rotor reference.
5. Measure Green assembly/factorization, Hodge diagnostics, residual passes,
   recurring solve time, and retained memory at production panel counts.
6. Add CUDA route proof for panel-wake potential and any promoted recurring
   operation. Host fallback must be named and prohibited in hard-CUDA runs.
7. Seek a separate promotion ruling only if accuracy gates pass and the
   projected 414-step case clears 6480 seconds with at least 20% device-memory
   reserve. Otherwise retain the formulation as experimental.

## Subitem structure (adopted by Ryan 2026-09-06)

Pre-registration is locked per subitem immediately before it runs. Dependency
chain: .0 → .2 → {.3, .4} → .5, with .1 feeding .2 and .6 free-floating.

- **052e.0 — Theory gate + kernel conventions.** Accept
  `052e-theory-velocity-to-potential-trace.md`; run Tier 0A. Host-only, no
  dependencies. Exit: theory accepted, conventions verified.
- **052e.1 — Code health prerequisites.** Hard `HybridWakePotential`
  unpaired-edge init error + negative regression (independent of 052b);
  host regression re-run (after 052b closes). Exit: all green; gates .2.
- **052e.2 — Tier 0B Green-trace oracle.** Five-stage error decomposition,
  both gauges, resolution/core/phase sweeps. Owns the structural kill rule;
  ends with an explicit continue/retire ruling.
- **052e.3 — Body-solve fixtures (Tiers 1 + 1.5).** Flat matched wake with
  two-body, gauge-invariance, and negative unpaired-edge fixtures; distorted
  variable-strength helical sheet. Depends on .2.
- **052e.4 — Temporal coherence (Tier 2).** Translating-ring test at
  `recompute_interval=1`. Depends on .2; may overlap .3.
- **052e.5 — Stage B + promotion package.** Hybrid-vs-VTS matrix, reduced-
  rotor direct reference, P1–P3, CUDA route proof, provenance. Gated on
  .2–.4; ends with the promotion ruling (gauge-invariant outputs only).
- **052e.6 — Global gauge recovery (design study, parallel track).** The
  topology-aware zero-at-infinity capability unblocking absolute Cp /
  unsteady Bernoulli / acoustics. Design doc only; separate ruling; may be
  scoped early so the .5 ruling is not blind to pressure-capability
  feasibility.

## Acceptance

- Independent VTS/hybrid comparison artifacts with matched configuration and
  complete revision/timing/memory metadata.
- No nonfinite state, every block solve converged, and every required route
  classified.
- Production-scale accuracy and performance gates passed on CUDA, followed by
  an explicit user ruling to promote. Until then, 052b and Phase 6 remain VTS.
- Present promotion evidence is limited to gauge-invariant circulation,
  exterior velocity, and integrated loads. Absolute Cp, unsteady pressure, and
  acoustics require a separately designed and verified topology-aware global
  gauge mechanism; a finite far-field value must be compared with its reference
  value rather than assumed to be zero.

No deployment, submission, commit, or cleanup is authorized by this item.
