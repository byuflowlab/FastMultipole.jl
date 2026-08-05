# 034 FLOWVPM GPU Integration: Resident FMM Lifecycle in the VPM Time Loop

## Status and Entry Gate

**Added by user request on `2026-08-04`.** Not started.

Entry gate: `032` (the generalized device interface) and `033` (the CPU
baselines and accuracy reference) must both be Done and clear-context
approved.

## Objective

Modify FLOWVPM (branch `gpu-full` of `../FLOWVPM.jl`) to drive the
FastMultipole resident GPU lifecycle from its own time loop, keeping particle
state on the GPU across the whole step with no per-step host/device body
transfers, and verify end-to-end correctness.

Deliverables:

1. **Device coupling**: a `CuArray`-backed `ParticleField` acts as a
   `DeviceResident` FastMultipole system via the `032` interface — device
   `source_to_buffer!` reading positions/`Γ`/`σ` straight from the 46×N
   particle matrix rows, and device `buffer_to_target!` accumulating
   velocity and the full 9-component velocity gradient into the
   `U_INDEX`/`J_INDEX` rows, all on device.
2. **Cache lifecycle in the time loop**: construct one `RadixFMMCache` sized
   to `maxparticles` at simulation start and reuse it every step (and every
   RK substep) of `run_vpm!`; zero steady-state allocation in the
   FastMultipole adapter/cache lifecycle (not unrelated FLOWVPM integrator
   work); live particle count may vary up to capacity (particles added/removed
   between steps).
3. **`nearfield_device` hazard resolved**: the silent-nearfield-drop path
   documented in `../FLOWVPM.jl/CLAUDE.md` (Phase 4 notes) must be removed or
   made impossible to reach — the GPU FMM path computes the nearfield
   correctly, and any unsupported configuration fails loudly.
4. **Correctness verification**: GPU FMM vs `UJ_direct` reference on both
   Integration Phase test cases (unit cube and helical wake cylinder), Float64 and
   Float32, static evaluation plus a multi-step dynamic run; sampled
   relative gradient (velocity) RMS error must be **≤ 1e-3** (the fixed
   phase tolerance, user decision `2026-08-05`), measured with the `033`
   references. Log relative Jacobian RMS error for every configuration as a
   diagnostic; it is not a pass/fail gate.
   A regression test gated on `CUDA.functional()` joins the FLOWVPM test
   suite alongside `test/runtests_gpu.jl`.
5. **CPU-path preservation**: FLOWVPM's CPU path and public API (exported
   names, keyword defaults) are unchanged — downstream consumers
   (FLOWUnsteady, VortexLattice) must be unaffected. CPU tests
   (`runtests_singlevortexring.jl`, `runtests_leapfrog.jl`) pass unchanged.

## Dependencies

- `032-impl-generalized-device-interface.md`, Done and clear-context approved.
- `033-flowvpm-baseline-benchmarks.md`, Done and clear-context approved
  (supplies the case definitions and accuracy gate).

## Mandatory Reading Gate

1. `START_HERE.md`, including the Integration Phase preamble.
2. `../FLOWVPM.jl/CLAUDE.md` **in full, before touching that repository** —
   especially the module include-order, particle-storage, GPU-path, and
   FMM+GPU Phase 4 sections.
3. `../FLOWVPM.jl/logs/2026-07-21-gpu-full.md` (the GPU branch design log).
4. The `031` approved spec and the `032` API documentation.

## Task-Local Requirements

- FLOWVPM edits are committed on the `gpu-full` branch of `../FLOWVPM.jl`;
  any FastMultipole fixes discovered here are committed in this repository
  (and noted in this file). Task files and coordination stay here.
- Respect FLOWVPM's established idioms: the CPU/GPU fork is
  `pfield.particles isa Array` (not `useGPU`); new files must be inserted at
  the correct point in the `FLOWVPM.jl` include order; prefer extending
  accessor functions over new struct fields.
- The RK3 low-storage integrator calls the UJ solve multiple times per step
  with resets in between — the coupling must handle `_reset_particles`
  semantics (zeroing `U`/`J` rows) and accumulation correctly on device.
- Record any per-step host allocation or transfer observed (target: zero body
  transfers; metadata transfers per the `023` counter contract only).
- Performance tuning is explicitly out of scope here (that is `035`); this
  row ends with a correct, transfer-free coupling and a first unoptimized
  GPU-vs-baseline timing sanity check.
