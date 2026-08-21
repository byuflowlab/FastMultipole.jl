# 049 Impl: Rotor Field GPU Verification (p018_L1_ov3, n = 210,056)

## Status and entry gate

**Staged `2026-08-20` (user direction). Not started.**

Entry gate: `047` and `048` complete and approved. **Pre-gate, runnable any
time after `046`:** verify the p018 VTPs carry all nine loader-required
fields (the loader throws on any missing) — do this cheap host-side check
before any H200 time is spent.

**User checkpoint:** which residency mode ships as default
(upload-per-step vs device-resident).

## Motivation

The user's named verification target for the Production Integration Phase:
load the large rotor-hover particle field `~/p018_L1_ov3_paraview/`
(**n = 210,056** at step 710 — 3× larger than `041h`'s primary case) and
verify that the unified GPU stack evaluates UJ with and without SFS plus a
full timestep with excellent performance. This row also produces the
per-pass budget that tells `050` which constraint binds before the panel
work is scoped, and settles the device-resident vs upload-every-step
question with a measurement.

## Objective

On H200, with the p018 210k-particle field: (1) UJ without SFS, (2) UJ with
SFS (`048`), (3) a full `nextstep` timestep — all verified for correctness
and timed; a per-pass time budget table against the 3.3 s/step target; a
measured residency recommendation with the user checkpoint answered.

## Method

### Stage 0 — pre-gate (host, after `046`)

Read one snapshot's XML header and confirm all nine required point-data
arrays exist: `gamma, sigma, vol, circulation, velocity, vorticity, C, SFS,
velocity_gradient`. Record the result here.

### Stage 1 — load

Load step-710..719 snapshots via FLOWPanel's
`_load_panel_particle_wake_vtk!` (`FLOWPanel_warmstart.jl:238`, via
ReadVTK), or `041h`'s standalone zlib-VTP extractor if the FLOWPanel loader
is awkward in the harness. **Only steps 710–719 exist on disk** although the
`.pvd` manifests list the full series from 0 — do not walk the `.pvd`.

### Stage 2 — UJ / UJ+SFS verification

On H200: UJ without SFS, then UJ with SFS, vs CPU references
(`Estr_direct!`/`Estr_fmm!` for the SFS channel) at the phase's 1e-3 gate
(F64 tighter), P=4 and P=8, both precisions where meaningful. Counters and
zero-allocation contracts hold.

### Stage 3 — full timestep + budget table

Run a full `nextstep` timestep (RK3) on the field. Produce the **per-pass
time budget table against the 3.3 s/step target**: tree
refresh/upward/M2L/downward/nearfield-UJ/SFS/integrator/transfers, each
compared against the `041a` fig15 anchors and the 170–230 s CPU baseline,
so the binding constraint is visible before `050` scopes the panel work.

### Stage 4 — residency tradeoff

Measure both step modes:

- **Upload-per-step:** host-resident particles, ~30 MB H2D ≈ a few ms on
  H200 — the minimal-invasiveness option.
- **Device-resident:** requires fixing `nextstep`'s `Threads.@threads`
  scalar U_prev loop (`FLOWVPM_particlefield.jl:504-517` — the only blocker
  for a fully resident step with Inviscid/PSE viscous); host callbacks stay
  host.

**Default recommendation = upload-per-step unless resident wins by >15% of
step time** (resident requires porting CPU mutation code — invasiveness
only justified by measured margin). Present both numbers at the user
checkpoint.

## Gates and verdict

- Pre-gate: all nine fields present (else fix the pipeline before H200
  runs).
- Accuracy gates green on UJ and UJ+SFS; full timestep runs without
  contract violations.
- Budget table delivered; verdict states which pass binds vs the 3.3 s/step
  target and the residency recommendation with its measured margin.

## Artifacts

- `scripts/fm049_*` drivers (loader/pre-gate check, H200 verification +
  timing harness).
- `data/rotor_field_gpu_verification/` — timing/accuracy CSVs, the budget
  table, `report.md`.

## Verification

- Accuracy vs sampled CPU direct references at the standing gate; job IDs
  recorded for all H200 timings; same-job A/B for residency comparison.

## Recorded context (2026-08-20 staging)

**Snapshot inventory (`~/p018_L1_ov3_paraview/`, 425 MB):** particle
snapshots
`p018_L1_ov3_wake1_particles/p018_L1_ov3_wake1_particles.{710..719}.vtp`
(10 files, ~40.5 MB each, XML PolyData, zlib-appended, same format 041h
describes); panel body `.vtu` in `..._body1/` (+2 trailing-wake series),
filament `.vtu` in `..._wake1_filaments/`. **Only steps 710–719 exist on
disk** though the `.pvd` manifests list the full series from 0
(dt = 1/3240 s). **n = 210,056 particles at step 710** — 3× larger than
041h's primary case (67,745); partially answers 041h's regime-honesty worry
(n≈4–7e4 below the 041e fused-nearfield win envelope).

**VTP loading:** FLOWVPM has NO VTP reader. FLOWPanel's is complete:
`src/FLOWPanel_warmstart.jl:238` `_load_panel_particle_wake_vtk!` (via
`ReadVTK`), path pattern `{path}/{name}_particles/{name}_particles.{idx}.vtp`,
fills `pf.particles` row-blocks (X/GAMMA/SIGMA/VOL/CIRCULATION/U from
"velocity"/VORTICITY/C/SFS/J from "velocity_gradient" reshaped 9×np). It
**throws if any of** `gamma, sigma, vol, circulation, velocity, vorticity,
C, SFS, velocity_gradient` is missing — hence the pre-gate. Callers:
`FLOWPanel_replay.jl:482`, `FLOWPanel_warmstart.jl:452`. Writer convention:
`src/FLOWPanel_wake.jl:2170-2200`.

**Device-resident blockers, precisely:** (a) `nextstep`'s
`Threads.@threads` scalar U_prev loop (`FLOWVPM_particlefield.jl:504-517`)
— NOT forked for GPU, scalar-indexing hazard; (b) `CoreSpreading` viscous:
`zeta`/`rbf` CPU-only by design (`FLOWVPM_viscous.jl:216,234`, iterative
re-calls `:472,:499,:580`); (c) the host callbacks
(`static_particles_function`, removal loop, `runtime_function` —
`FLOWVPM_utils.jl:87-131`). `add_particle`/`remove_particle` already have
GPU-safe broadcast paths (`FLOWVPM_particlefield.jl:227-235,463`);
`ParticleStrengthExchange` fully broadcastable; `Inviscid` no-op. So with
Inviscid/PSE viscous, only the U_prev loop blocks a fully resident step.

**Time loop:** `src/FLOWVPM_utils.jl:41` `run_vpm!` (body `:87-131`): per
step — `static_particles_function` callback (appends; FLOWPanel sheds here,
`:102`) → `nextstep` (`FLOWVPM_particlefield.jl:497-522`) → removal loop
truncating static particles (`:108-112`) → `runtime_function` callback
(`:116`) → save. Integrators in `FLOWVPM_timeintegration.jl` (euler `:48`,
rk3 `:334`/`:545`); relaxation has a GPU broadcast path
(`FLOWVPM_relaxation.jl:48,167,196`).

**Anchors for the budget table:** FMM step (041a fig15, unitcube GPU
best-uniform): 1.31 ms @1e4, 7.40 @1e5, 92.3 @1e6; brute-force crossover ≈
4–5.5e3. 018 CPU baseline: 170–230 s/step on 64 cores; split = wake
influence 64.2% / body 25.3% / solve 9.3%; ~75% of step = `Estr_fmm!`; wake
FMM velocity ~7 s; body pass floor ~36 s; 49% thread utilization. Target:
≤3.3 s/step (30 rev = 1080 steps < 1 h).

**041h relationship:** 041h is staged, NOT started; it owns single-H200
per-step optimization of the n=3.7e4/6.8e4 rotor cases and plans its own
standalone zlib-VTP parser (`041h:42-50`). p018_L1_ov3 (210k) is a third,
larger case not in 041h. Cite 041h results as soft inputs if available; do
not duplicate its scope.
