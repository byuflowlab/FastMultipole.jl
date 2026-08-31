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
future promotion ruling. Compare independent, identically configured hybrid
and VTS runs; never march both formulations through one mutable state.

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
- Each Dirichlet body currently owns a separate dense gauge-fixed Green
  factorization. Sharing is forbidden until exact body-local matrix parity is
  demonstrated independently by handedness and ordered connectivity.

## Work plan

1. Re-run the host formulation, convergence, solver, history, simulation,
   warm-start, comparison, and parsing regressions after 052b closes.
2. Establish production-shaped accuracy evidence across wake resolution,
   particle core, distance, handoff, rotor count, and OGE/IGE; pre-register CT,
   CQ, circulation, trace, and surface-velocity tolerances.
3. Measure Green assembly/factorization, Hodge diagnostics, residual passes,
   recurring solve time, and retained memory at production panel counts.
4. Add CUDA route proof for panel-wake potential and any promoted recurring
   operation. Host fallback must be named and prohibited in hard-CUDA runs.
5. Seek a separate promotion ruling only if accuracy gates pass and the
   projected 414-step case clears 6480 seconds with at least 20% device-memory
   reserve. Otherwise retain the formulation as experimental.

## Acceptance

- Independent VTS/hybrid comparison artifacts with matched configuration and
  complete revision/timing/memory metadata.
- No nonfinite state, every block solve converged, and every required route
  classified.
- Production-scale accuracy and performance gates passed on CUDA, followed by
  an explicit user ruling to promote. Until then, 052b and Phase 6 remain VTS.

No deployment, submission, commit, or cleanup is authorized by this item.
