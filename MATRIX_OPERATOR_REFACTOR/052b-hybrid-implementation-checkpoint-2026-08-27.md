# 052b hybrid solver implementation checkpoint — 2026-08-27

## Stop point

The implementation was developed and tested in the isolated copy
`/private/tmp/flowpanel-052b.yEP0uR`. The live sibling
`../FLOWPanel.jl` is concurrently edited by other agents; never overwrite,
stash, reset, or mass-copy it. Continue using context-checked `apply_patch`
hunks only.

The live source merge is functionally complete for this session:

- `src/FLOWPanel_solver.jl` exactly matches the isolated tested copy.
- `src/FLOWPanel.jl` exactly matches the isolated copy.
- `src/FLOWPanel_instrumentation.jl` exactly matches the isolated copy.
- `src/FLOWPanel_formulation.jl` differs only by one harmless blank line at
  line ~957; its code matches the isolated copy.

The following remain only in the isolated copy and must be merged hunk-wise:

- `test/runtests_unit_solver.jl`
- `test/runtests_unit_solver_history.jl`
- `test/formulation_test.jl`
- `docs/wake_solve_schemes.md`

Do not copy whole files. First run `diff -u` against each current live file,
then apply only the intended additions with `apply_patch`.

## Implemented source behavior

### Corrected tuple/block-GS solve

`solve!(bodies::Tuple, solvers::Tuple)` now:

- freezes each Dirichlet body's source strengths once from the incident
  velocity supplied at entry;
- sends scalar potential, not velocity, from other bodies to Dirichlet
  targets;
- sends velocity to Neumann targets;
- preserves prescribed external/wake potential while adding self-source
  potential;
- zeros the prior doublet iterate before the general self-source influence
  call (essential when no assembled `S` exists);
- stops on the maximum normalized physical block residual, with separate
  `dirichlet_residual_scale` and `neumann_residual_scale`;
- retains strength delta only as status diagnostics;
- exposes `require_outer_convergence` and expanded `block_gs_status` fields.

### Hybrid wake formulation

New exported `HybridWakePotential`:

- evaluates retained `PanelWake` potential directly;
- isolates particle velocity as total sampled wake velocity minus retained
  panel-wake velocity;
- Green-reconstructs the particle trace independently for each Dirichlet
  rigid body using a separate dense gauge-fixed factorization;
- freezes Dirichlet source strengths from the pre-wake freestream/kinematic
  field;
- leaves full wake velocity on Neumann bodies such as ground;
- invokes the corrected tuple solve, so other bodies enter Dirichlet targets
  through actual panel potential;
- records Green-system residual and gauge defect;
- independently reconstructs an edge-based least-squares surface-Hodge trace
  and records Green/Hodge mismatch plus tangential projection defect;
- treats `VelocityThroughSources` as a documented legacy approximate baseline.

Shared Green factors are deliberately not enabled. The code requires one
body-local dense Green construction until exact matrix parity is validated.

## Verification already completed in the isolated copy

1. `test/runtests_unit_solver.jl`: **422/422 passed** after the corrected
   block solver and new four-Dirichlet-body + flat-Neumann-ground oracle were
   added. That regression compares block GS against `BackslashCoupled`, checks
   normalized rotor/ground residuals below `1e-11`, strength agreement, and
   nonzero cross potential.

2. Targeted `test/hybrid_debug.jl` (temporary, do not merge):

   - panel-only hybrid doublets exactly matched `DirectWakePotential`;
   - direct residual was `8.64e-16`;
   - one far vortex particle produced nonzero particle sigma
     (`0.0071015854`);
   - Green residual `3.93e-15`, gauge defect `1.11e-16`;
   - Green/Hodge mismatch `0.12156`, tangential defect `0.13083`;
   - normalized block residual `2.90e-14`, converged in one single-body block.

3. The full `formulation_test.jl` was run twice while developing. Existing
   stages through Stage 9 passed. Its first hybrid attempt exposed accidental
   reuse of accumulated `body.potential`; that was fixed by making the hybrid
   RHS own exactly `q_panel + q_particle`. The targeted debug then passed.
   The final full suite, including the newly added particle smoke, has **not**
   yet been rerun after that fix.

4. `git diff --check` is clean for the four live source files.

Use a writable depot when testing from the isolated copy:

```sh
JULIA_DEPOT_PATH=/private/tmp/flowpanel-052b-depot:/Users/ryan/.julia \
JULIA_NUM_THREADS=2 julia --project=/private/tmp/flowpanel-052b.yEP0uR \
  /private/tmp/flowpanel-052b.yEP0uR/test/runtests_unit_solver.jl
```

The isolated Manifest resolves sibling packages through symlinks already made
at `/private/tmp/FastMultipole` and `/private/tmp/FLOWVPM.jl`.

## Immediate next steps

1. Recheck live diffs/status because other agents are active.
2. Merge the four pending test/doc files hunk-by-hunk from the isolated copy.
3. Run `Meta.parseall` and `git diff --check` on live source.
4. Run live FLOWPanel suites at minimum:
   - `test/runtests_unit_solver.jl`
   - `test/runtests_unit_solver_history.jl`
   - `test/formulation_test.jl`
   - simulation and warm-start suites because tuple orchestration changed.
5. Fix only failures attributable to this implementation; preserve all
   concurrent edits.
6. Add the still-missing production gates before calling the revised plan
   complete:
   - particle-core/body minimum-distance or overlap diagnostic and warn/error
     policy (must be scalable for device-resident fields);
   - known panel-wake-to-particle convergence oracle over resolution/core/
     distance/handoff sweeps;
   - production snapshot telemetry and VTS-vs-hybrid circulation/thrust/torque;
   - driver audit rename to `surrogate_dirichlet_residual` if that audit exists
     in concurrently edited example code;
   - direct matrix parity for mirrored shared operators;
   - CUDA target-output route proof and four-rotor GPU smoke/acceptance gates.

No HPC jobs were submitted and no launcher/driver changes were made in this
session.
