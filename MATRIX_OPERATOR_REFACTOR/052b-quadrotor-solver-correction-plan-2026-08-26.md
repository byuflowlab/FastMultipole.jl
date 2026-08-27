# 052b Quad-Rotor Solver Correction and Verification

## Summary

- Do **not** approve the current quad-rotor solver. It performs sequential block Gauss–Seidel and computes cross-body velocity, but omits cross-body scalar potential required by the rotors’ Dirichlet boundary condition.
- A four-body diagnostic falsely reported convergence after 5 sweeps (`Δstrength = 3.43e-12`) while rotor potential residuals remained `1.18e-3`. `BackslashCoupled` produced approximately `2e-17`; a corrected block prototype produced `1e-15`–`5e-15`.
- Existing job `13484025` recorded 179/179 converged strength-delta solves and acceptable ground tangency, but it did not measure rotor boundary residuals and therefore cannot validate the rotor solution.

## Solver and Interface Changes

- Correct `solve!(bodies, solvers)` in `../FLOWPanel.jl/src/FLOWPanel_solver.jl`:
  - Restore each target’s external velocity and potential before its block update.
  - Accumulate latest-strength cross influence sequentially: velocity for every target and scalar potential for Dirichlet rotor targets.
  - Recompute the target rotor’s source strength from total velocity, add its self-source potential without clearing cross potential, then solve its cached doublet operator.
  - Preserve the existing Neumann ground update and block-GS ordering.
  - Add `require_outer_convergence=false`; production 052b launchers set it true.
- Replace the tie-sensitive shared-operator canonical frame with an ordered-node orthogonal-fit validation that accepts rigid translation, rotation, and reflection. Keep connectivity, type, core-size, and geometry-residual guards.
- Add an env-gated rotor boundary audit that records per-rotor `max|φ_interior|/(Utip·R)` alongside GS iterations/delta. Keep ground `max|U·n|/Utip`.
- Extend the rectangular panel host/CUDA path to return scalar potential with velocity for block cross-influence calls. Acceptance runs must prove these calls used CUDA; silent FMM/CPU fallback is forbidden.

Current solver roles remain:

| Solver | Role |
| --- | --- |
| Per-rotor `Backslash` + block GS | Production path after correction |
| `FlatGroundSolver` | Ground block |
| `BackslashCoupled` | Reduced-case exact oracle only; production matrix is too large |
| `KrylovCoupled` | Secondary matrix-free oracle/fallback experiment, not the selected production path |
| Per-body `KrylovSolver` / `FGSSolver` | Available inner solvers but do not themselves fix missing cross-potential orchestration |

## Verification

- Add a reduced four-rotor Dirichlet regression with translated, rotated, and mirrored copies, both OGE and with a flat ground:
  - Require block-GS strength agreement with `BackslashCoupled`.
  - Require direct-backend rotor residuals below `1e-10` normalized and ground tangency below `1e-10`.
  - Demonstrate nonzero rotor↔rotor and rotor↔ground influence.
- Add host/CUDA parity tests for combined panel potential and velocity, including mixed rotor/ground sources and all quad orientations.
- Re-run solver, simulation, warm-start, replay, and FastMultipole rectangular-kernel suites.
- Run short production-shape 4r OGE and IGE GPU smokes with boundary auditing every step. Require:
  - every outer solve converged within 50 sweeps at `GS_TOL=1e-8`;
  - normalized rotor potential and ground tangency residuals ≤ `1e-6`;
  - CUDA route counters for cross influence and zero fallback.
- Only then run the six 414-step acceptance cases and record residual maxima, GS iteration distributions, CT histories, and route/timing counters.

## Assumptions

- Retain `VelocityThroughSources`, host LU factors, shared rotor operators, no GPU-S, and the existing 052b physical settings.
- Strength-delta convergence remains the inexpensive stopping criterion, but it is no longer accepted as proof of boundary-condition satisfaction.
- Existing 4r reference outputs remain physical comparison data only; they are not correctness references for the corrected coupled solve.
