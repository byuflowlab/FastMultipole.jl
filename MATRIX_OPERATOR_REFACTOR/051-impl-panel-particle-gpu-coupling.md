# 051 Impl: Panel–Particle GPU Coupling

## Status and entry gate

**Staged `2026-08-20` (user direction). Not started.**

Entry gate: `050` complete and approved — this row implements the `050`
verdict's named shape; do not re-litigate the A/B/C decision here. Fallback
resource envelope per user: **1×GPU + 64 CPU threads**.

## Motivation

The 018 step splits as wake influence 64.2% / body 25.3% / solve 9.3%
(023 profiling); after `048`/`049` accelerate the particle side, the panel
passes become the binding constraint on the ≤3.3 s/step budget (the CPU
body-pass floor alone is ~36 s). This row wires the `050`-selected GPU
coupling — panel GPU direct kernel, system-on-system passes, or hybrid —
into FLOWPanel's production influence path.

## Objective

The `050` verdict implemented: panel↔particle GPU coupling wired through
FLOWPanel's `influence!`/`FastMultipoleBackend`, passing pass-by-pass
parity at the 018 operating point with no regression on FLOWPanel CPU
tests.

## Method

### Stage 1 — implement the `050` shape

Per the verdict: (A) distinct-target support on the radix path, and/or (B)
system-on-system GPU passes mapped onto the existing 3-pass structure
(`FLOWPanel_simulate.jl:673-712`), and/or (C) hybrid GPU-particles /
64-thread-CPU-panels. If a panel GPU direct kernel is in scope, port the
`direct!` overload (`FLOWPanel_abstractbody.jl:1260`) for the element types
the 018 driver exercises (constant source/doublet tris + vortex
rings/sheets/filaments).

### Stage 2 — wiring

Wire through `FLOWPanel_fmm.jl:60` `influence!` /
`fmm!(targets::Tuple, sources::Tuple)` (`:88`, plan-reusing `:114`) and the
separate `Estr_fmm!` call (`FLOWPanel_wake.jl:2052`), preserving per-pass
kerneloffsets and derivative switches. Respect the FmmPlan /
NearfieldInfluenceCache disposition decided in `050`.

### Stage 3 — parity + regression harness

- Pass-by-pass parity vs the CPU path at the 018 operating point (each of
  the three influence passes + the SFS pass, compared independently).
- No regression on FLOWPanel CPU tests.
- **Step-head gotcha (cost 3 jobs previously):** the harness must run the
  full step head (`maneuver!` + reset/freestream/kinematic + `update_TE!`)
  before any influence eval, or the restored first wake row silently NaNs
  every target.

## Gates and verdict

- Pass-by-pass parity green at the 018 operating point (phase 1e-3 gate;
  F64 tighter where applicable).
- FLOWPanel CPU test suite green; FastMultipole/FLOWVPM suites unregressed.
- Timing of the coupled passes reported against the `049` budget table
  (feeds `052`).

## Artifacts

- Source changes on the unified branches (FLOWPanel + FastMultipole as the
  `050` shape requires) + tests.
- Parity/timing CSVs + results section appended to this doc.

## Verification

- Same-job CPU/GPU pass comparisons with job IDs; NaN guard on the
  step-head gotcha exercised in the harness.

## Recorded context (2026-08-20 staging)

**FLOWPanel per-step influence structure** (`FLOWPanel_simulate.jl:673-712`):
three separate fmm! passes — wake→(bodies+particles), panel solve,
bodies→targets (different kerneloffsets/derivative switches per pass) +
separate `Estr_fmm!` call (`FLOWPanel_wake.jl:2052`) reusing wake trees.
Targets ≠ sources in these passes. Entry: `FLOWPanel_fmm.jl:60` `influence!`
over heterogeneous tuples → `fmm!(targets::Tuple, sources::Tuple)` at `:88`
(plan-reusing at :114).

**FMM-compatibility overloads (CPU, complete):**
`source_system_to_buffer!` `FLOWPanel_abstractbody.jl:1096`, `direct!`
`:1260`, `body_to_multipole!` per element type
(`FLOWPanel_nonliftingbody.jl:224-234`,
`FLOWPanel_liftingbody.jl:705,797,906`), PanelWake trio
(`FLOWPanel_wake.jl:326,525,562,565`), filaments (`:2763,2825,2841`).

**GPU support in FLOWPanel today: none** (only a `GPUArray` kwarg on the
dense linear solve, `FLOWPanel_liftingbody.jl:379-412`; no CUDA dep).

**Harness gotcha (cost 3 jobs):** run the full step head (`maneuver!` +
reset/freestream/kinematic + `update_TE!`) before any influence eval or the
restored first wake row silently NaNs every target.

**018 operating point:** 36,752 panels (45_185_ct4 mesh), NT=36 steps/rev,
~181k particles at maturity (342k on the 6R arm), THREADS=64; CPU baseline
170–230 s/step, split wake 64.2% / body 25.3% / solve 9.3%, body-pass floor
~36 s; budget ≤3.3 s/step.
