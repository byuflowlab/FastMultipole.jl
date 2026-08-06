# 034 FLOWVPM GPU Integration: Resident FMM Lifecycle in the VPM Time Loop

## Status and Entry Gate

**Added by user request on `2026-08-04`.** In progress: coupling implemented
and locally verified `2026-08-06` (user authorized starting ahead of the 033
gate); H200 device validation and 033-gated comparisons pending. Work Record
below.

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

## Work Record

### 2026-08-06 — coupling implemented, host path locally verified (session 1)

Entry context: user authorized starting before 033 fully landed (WAKE CPU job
13058428 still running); everything 033-gated is explicitly deferred. All
FLOWVPM changes are committed on `../FLOWVPM.jl` branch `gpu-full`
(`4df2bc0`, `7b0d8bc`, `e86fa38`); this file is the only FastMultipole edit.

**What was built (FLOWVPM `gpu-full`):**

- `src/FLOWVPM_fmm_radix.jl` (new, included after `fmm`): the 032-interface
  coupling. Traits: `body_type = Point{Vortex}`,
  `residency = particles isa Array ? HostResident : DeviceResident`,
  `direct_kernel = RegularizedVortex(sigma_row=8)` with a hard `gaussianerf`
  check. Cache lifecycle: one `RadixFMMCache` per `ParticleField`, built
  lazily at first GPU/radix evaluation, `max_n_bodies = pfield.maxparticles`,
  `hessian=true`, reused across all RK3 substeps/time steps
  (`WeakKeyDict` registry so a dropped field releases its GPU memory);
  live `np` may vary below capacity. Configuration is derived: cubic bounds
  from live-particle extrema padded 10%/face; `ell` = deepest depth passing
  the near-set adequacy inequality `2^ell < g_min·L/(rho_t·sigma_max)`
  (strict) capped by an `n^(1/3)`-cells-per-side occupancy heuristic;
  `near_radius2=16`, `window_classes=256` (device), precision =
  `eltype(pfield)`; expansion order `pfield.fmm.p - 1`. Overrides via
  internal `radix_fmm_settings!` (not exported; 035 owns tuning).
- **Recenter policy**: `fmm!` out-of-box `ArgumentError` -> one
  `recenter!(cache, pfield; bounds=derived padded bounds)` + retry; a second
  failure propagates. User-fixed `bounds` are a promise: no auto-recenter,
  the error propagates.
- `ext/FLOWVPMCUDAExt.jl`: the only CUDA-typed pieces — bulk device
  `source_to_buffer!` (rows 1:3 X, 4 = `default_rho_over_sigma·sigma`,
  5:7 Gamma, 8 sigma, live prefix, identity sort index) and
  `buffer_to_target!` (switch-relative accessors, **accumulates** `.+=` into
  `U_INDEX`/`J_INDEX` per the delivery-semantics contract; matches the legacy
  hook incl. static particles). Zero per-step body H2D/D2H by construction.
- **`nearfield_device` hazard resolved**: `UJ_fmm` now routes CuArray-backed
  fields to `UJ_fmm_gpu!` (loud errors for `rbf`, `sfs`, any FMM autotuning
  flag on); the legacy octree call is CPU-only with `nearfield_device=false`
  hardwired. The silent-nearfield-drop path is unreachable.
- **CPU/public-API preservation**: no exported-name or keyword-default
  changes; the coupling is behind `_FMM_HAS_RADIX` (`isdefined` guards) so
  FLOWVPM still loads against registry FastMultipole 2.0.x.

**Two real pre-existing incompatibilities found and fixed (commit `4df2bc0`):**

1. `fmm.get_previous_influence` no longer exists on `matrix-ops` (replaced by
   metadata rows) — FLOWVPM failed to *load*. Overload now guarded.
2. **Silent U/J corruption of every FLOWVPM hook against `matrix-ops`**:
   FLOWVPM's `fmm.direct!` and `buffer_to_target_system!` used the registry
   fixed target-buffer accessors (gradient rows 5:7, hessian 8:16), but
   `matrix-ops` buffers are switch-relative — with `scalar_potential=false`
   gradient is 4:6 and hessian 7:15, so everything was read/written one row
   off AND `set_hessian!`'s fixed row 16 wrote past the 15-row buffer under
   `@inbounds`, corrupting the next column's z-position (measured u_rel_rms
   ~1.4–2.4 on both cases, both `UJ_direct` and `UJ_fmm` wrong). Fixed with
   load-time accessor shims (`_fmm_get/set_*` on
   `isdefined(fmm, :gradient_range)`), correct against both FastMultipole
   generations.

**Local verification (this machine, no GPU, <= 6 threads):**

| check | result |
| --- | --- |
| host-resident radix coupling vs `UJ_direct`, cube n=4000 (033 construction) | u_rel_rms 1.9e-4 (gate 1e-3), J diag 6.5e-4 |
| host-resident radix coupling vs `UJ_direct`, wake n=1500 (033 construction) | u_rel_rms 7.5e-5, J diag 3.5e-4 |
| accumulate semantics, cache reuse under motion, varying np, auto-recenter, fixed-bounds throw, loud-error paths | all pass (`test/runtests_gpu_fmm.jl` Part A, 18 tests) |
| full FLOWVPM CPU suite (singlevortexring + leapfrog) against dev'd `matrix-ops` | all pass |
| full FLOWVPM CPU suite against registry FastMultipole 2.0.4 | loads fine; `UJ_fmm` fails — **pre-existing**: `gpu-full`'s legacy call passes `shrink`/`recenter` kwargs that registry-max 2.0.4 lacks, so the branch already required dev FastMultipole before 034 |
| ext + new files | parse clean; ext load-check impossible locally (CUDA.jl/CUDACore fails to precompile on macOS + Julia 1.12 — upstream, unrelated) |

**Staged for H200 (not run; cluster submission not authorized this session):**
`../FLOWVPM.jl/test/runtests_gpu_fmm_device.jl` (Part B: static U/J vs the
validated direct-sum GPU kernels on cube+wake at n=2e4, Float64+Float32, 1e-3
velocity gate + J diagnostic, 023 counter contract asserts
(`body_uploads==0`, `expansion_host_copies==0`, flat route/operator/influence
counters), steady-state `CUDA.@allocated` probe, varying-np capacity reuse,
5-step RK3 dynamic run vs CPU `UJ_direct`), runnable unmodified via the DRAFT
job scripts `../FLOWVPM.jl/scripts/cuda_034_run.sh` /
`cuda_034_submit.sh` (cuda_032 pattern, `julia/1.11.7-6bmogfl` pinned).
Job scripts live in the FLOWVPM repo because FastMultipole was read-only for
this session.

**Deferred pending 033 / H200:** wake-reference-gated accuracy claims against
the checksummed 033 references; any baseline speedup comparison; the first
GPU-vs-baseline timing sanity check (needs H200); deliverable-4 sign-off
(device runs are staged, not executed). Known punch-list items: `nextstep`'s
`update_U_prev` loop scalar-indexes a CuArray (pre-existing; unused by the
radix path — the device tests pass `update_U_prev=false`); per-`np`-change
reallocation of the framework's per-(rows, n) device scatter buffer
(framework-side, task-023 dict cache) is metadata-scale; GPU SFS and
`rbf`/`zeta` remain unsupported-loud.

### 2026-08-06 — H200 validation (session 2): all device tests pass (job 13061046)

**Remote layout (task-034-owned, separate from the 023-agent's trees):**
`~/FLOWVPM-034` (gpu-full working tree), `~/FastMultipole-034` (matrix-ops
working tree: src + test + MATRIX_OPERATOR_REFACTOR/scripts), env
`~/fm034env` (both repos `Pkg.develop`'ed at those paths, CUDA v6.2.1 with
`local="true"` preferences for the three CUDA JLLs written directly to
`LocalPreferences.toml` — `Preferences.set_preferences!` by name refuses
unloaded indirect deps — login-node instantiate with
`JULIA_PKG_PRECOMPILE_AUTO=0`, precompile deferred to the GPU node).
`julia/1.11.7-6bmogfl` pinned throughout. Driver:
`../FLOWVPM.jl/scripts/cuda_034_submit.sh` / `cuda_034_run.sh`.

**Jobs:** 13060973 (fail: world-age, below), 13061005 (fail: RK3 scalar
indexing, below), **13061046 = job of record, COMPLETED, 6m21s wall.**

**Results (job 13061046, H200 on m13h-1-1, all PASS):**

| group | outcome |
| --- | --- |
| preflight: FastMultipole `test/cuda_radix_interface_test.jl` (032 surface, REQUIRE_CUDA=1) | pass |
| preflight: FLOWVPM direct-sum GPU regression (`runtests_gpu.jl`) | 48 + 84 pass |
| Part A host-resident coupling (cube/wake vs `UJ_direct`, semantics, recenter, loud errors) | 12 + 2 + 4 pass |
| Part B device-resident static U/J, cube+wake × Float64+Float32, n=2e4 | 36 pass |
| Part B 5-step RK3 dynamic run (wake n=2000) vs CPU `UJ_direct` | 4 pass |

Static accuracy vs the validated direct-sum GPU kernels (gate u ≤ 1e-3; J is
diagnostic): cube u_rel_rms 8.83e-4 (F64 and F32; J 4.3e-3), wake u_rel_rms
2.13e-4 (J 1.2e-3). Dynamic run: max position drift 2.7e-8, u_rel_rms 7.1e-5,
J 3.0e-4. **023 counter contract held everywhere**: `body_uploads == 0`,
`expansion_host_copies == 0`, and route/operator/influence counters flat
across repeated evaluations and across the whole dynamic run (construction-only
uploads); varying-np capacity reuse kept `body_uploads == 0`. Steady-state
`CUDA.@allocated` per warm solve: 2.4–5.9 KB (metadata-scale, the known
framework-side per-(rows,n) scatter-buffer dict noted 2026-08-06, not body
transfers).

**Timing sanity number (NOT a benchmark — unoptimized derived settings,
single warm sample, for 035 order-of-magnitude context only):** warm
device-resident U/J solve at n=20,000: cube 19.7 ms / wake 38.4 ms (Float64);
cube 14.8 ms / wake 26.1 ms (Float32).

**Two FLOWVPM defects found on hardware and fixed (`gpu-full`):**

1. **World-age failure of the lazy lifecycle load** (job 13060973):
   `_build_radix_fmm_cache` called `fmm.load_cuda_radix_lifecycle!()`
   successfully, but the runtime `include` defines the CUDA methods in a
   newer world than the already-running evaluation frame, so
   `RadixFMMCache(device=true)`'s `cuda_radix_available()` check still hit
   the old-world `false` fallback and threw. Fix (commit `61988b1`): load the
   lifecycle from `FLOWVPMCUDAExt.__init__` (guarded by `CUDA.functional()`),
   i.e. at top level before any user statement; the lazy in-chain call stays
   as the loud-error path. Not a FastMultipole defect — consumers must load
   the lifecycle from top level, which the 032 docs' examples do — but worth
   a consumer-facing note.
2. **RK3 M-storage reset scalar-indexed CuArrays** (job 13061005): both
   `rungekutta3` methods reset per-particle M rows with an unforked scalar
   loop (pre-existing `gpu-full` gap; first-ever `nextstep` on a GPU field).
   Fix (commit `a5589dd`): factored `_reset_M_storage!` — CPU keeps the
   original loop, GPU uses the 0/1 static-mask broadcast
   (`_reset_particles_broadcast!` pattern). CPU smoke test unchanged.

FLOWVPM commits this session (`gpu-full`): `657b794` (job scripts adapted to
the 034-owned remote tree/env + warm-solve timing probe in the device test),
`61988b1`, `a5589dd`. No FastMultipole source edits; no FastMultipole-side
defects found on hardware.

**Deliverable status after this session:** 1 (device coupling), 2 (cache
lifecycle / zero body transfer), 3 (nearfield hazard), 5 (CPU preservation)
verified on hardware or previously; deliverable 4 is now measured on
hardware against the validated direct-sum GPU kernels on both cases, both
precisions, static + dynamic — all within the 1e-3 gate. **Still open before
034 can be marked Done:** (a) user sign-off on deliverable-4 correctness;
(b) the 033-reference-gated accuracy claims (checksummed 033 wake/cube
reference comparison, now unlocked since 033 is Done — the device tests used
live `UJ_direct`/GPU-direct references instead); (c) fold
`runtests_gpu_fmm.jl` into the conditional `runtests.jl` wiring alongside
`runtests_gpu.jl` if the user wants it in the default suite (it currently
runs standalone).

**FastMultipole interface gaps found (documented, not edited):** none in the
032 device surface itself — the shipped traits/hooks/cache/recenter contract
was sufficient as documented. The two items above (removed
`get_previous_influence`, switch-relative buffer accessors) are legacy-path
migration hazards for existing consumers; worth a note in the migration
section of the docs, since any consumer following the published legacy-hook
example (fixed `set_hessian!` rows + `@inbounds`) silently corrupts memory on
`matrix-ops` when `scalar_potential=false`.
