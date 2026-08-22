# 049 Impl: Rotor Field GPU Verification (p018_L1_ov3, n = 210,056)

## Status and entry gate

**H200 rerun complete: job `13305555` (2026-08-22) executed all five stages;
artifacts accepted by the user 2026-08-22 with documented gate rationale (see
"Results (2026-08-22, H200 job 13305555)" below). Every documented calibration
anchor is met or exceeded; the 49 harness-gate FAILs trace to gates applied at
the wrong layer or operating point, not to regressions. Wrapper-layer
allocation and error-bounded replay measurement deferred to `053`. Residency
checkpoint resolved: the user selected **upload-per-step** (2026-08-22, D15)
for compatibility with monitors that trim particles or otherwise modify body
states between steps.**

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
full timestep with excellent performance. This row produces a corrected
per-pass particle budget for downstream production decisions and settles the
device-resident vs upload-every-step question with a measurement.

## Objective

On H200, with the p018 210k-particle field: (1) UJ without SFS, (2) UJ with
SFS (`048`), (3) a full `nextstep` timestep — all verified for correctness
and timed; a per-pass time budget table against the 3.3 s/step target; and a
neutral residency comparison followed by the user's explicit choice.

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
compared against the `041a` fig15 anchors and the 170–230 s CPU baseline.
Report what binds in this corrected particle run without retroactively using
the result as an entry condition for the already-completed `050` theory row.

### Stage 4 — residency tradeoff

Measure both step modes:

- **Upload-per-step:** host-resident particles, ~30 MB H2D ≈ a few ms on
  H200 — the minimal-invasiveness option.
- **Device-resident:** requires fixing `nextstep`'s `Threads.@threads`
  scalar U_prev loop (`FLOWVPM_particlefield.jl:504-517` — the only blocker
  for a fully resident step with Inviscid/PSE viscous); host callbacks stay
  host.

No automatic percentage threshold selects the default. Run a true warmed,
interleaved, identical-state same-job A/B with preallocated contiguous
transfers, present both numbers and their measured difference at the user
checkpoint, and ask the user which mode should ship.

## Gates and verdict

- Pre-gate: all nine fields present (else fix the pipeline before H200
  runs).
- Accuracy gates green on UJ and UJ+SFS; full timestep runs without
  contract violations.
- Budget table delivered; verdict states which pass binds vs the 3.3 s/step
  target and presents the residency A/B values without choosing a default.

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

## Stage 0 pre-gate result (2026-08-21)

**PASS.** `p018_L1_ov3_wake1_particles.710.vtp` header carries all nine
loader-required point-data arrays (gamma, sigma, vol, circulation, velocity,
vorticity, C, SFS, velocity_gradient) plus Points/connectivity/offsets;
`NumberOfPoints="210056"` confirmed (matches the recorded n). XML PolyData,
zlib-compressed appended data, header_type UInt64 — as recorded at staging.

## Results (2026-08-21, H200 job 13247848)

Snapshot: p018 step 710, np=210,056, sha256 0d9136...155ab. Artifacts:
`data/rotor_field_gpu_verification/{fm049_report.txt,fm049_results.csv}`.

**Accuracy.** Device UJ_fmm U rel RMS = 3.38e-4 sampled / 3.98e-4 full field
vs a full-field GPU-direct O(N²) reference — **PASS at the 1e-3 gate**. J
1.64e-2 (diagnostic). Device-resident nextstep U_prev row exact to 1.8e-15
(the new broadcast fork). Two flags:
1. The CPU-sampled vs GPU-direct cross-check disagreed at u=2.6e-4 (expected
   ~1e-10 F64) — the two direct references differ somewhere (kernel-offset /
   buffer-overload semantics suspected). ARM1 passes vs either (≤~6e-4
   worst-case composition), but the discrepancy is flagged for follow-up.
2. SFS rel RMS vs the exact-J reference = 0.666. Consistent with the D5/D7
   J-error-bound mechanism at this field's J error (1.6e-2) with a
   large real-wake amplification factor (E/J was already 14 on the synthetic
   wake; E_str is cancellation-dominated). The 048 mechanical proof (SFS
   exact from delivered J, 1e-15) stands. Production-relevant comparison is
   CPU-FMM-SFS vs GPU-radix-SFS (both J-approximate) under the 052 CT/Γ(r/R)
   gate — flagged as a 052 watch item, with J accuracy (gh mode/P/MAC) the
   knob if it binds.

**Per-pass budget (median wall).**
| pass | time |
| --- | --- |
| cache build + first UJ (one-time, mostly JIT) | 56.6 s |
| device UJ (nominal no SFS) | 65.8 ms (historical; 048 review found this arm still ran TG/ζ) |
| device UJ+SFS | 66.1 ms (the 0.3 ms difference is **invalid as marginal SFS cost**; it measured delivery overhead between two ζ-running arms) |
| full RK3 nextstep (device-resident) | 198.7 ms (UJ 197.3 + rest 1.4) |
| H2D 46×210056 | 5.1 ms; D2H U/J/SFS 18.2 ms |
| host(CPU)-radix UJ datum | 72.8 s |

**Historical timing only:** the old run observed a 0.199 s resident RK3 step,
but its SFS timing boundary and residency arithmetic were invalid. It does not
establish the corrected particle budget or which pass binds; those verdicts
wait for the corrected same-job run and supported stage telemetry.

**Residency historical estimate (superseded).** The earlier 69.7 ms / 35.1%
figure combined separately measured transfers arithmetically and is not a
same-job timestep A/B. It does not choose a default. The corrected harness now
alternates actual upload-mode (contiguous H2D + RK3 + contiguous D2H) and
resident-mode RK3 from identical state, checks X/Gamma/sigma/SFS parity, and
leaves the default entirely to the user checkpoint after measurement.

## Local remediation staged 2026-08-22

- Fixed `UJ_direct(source,target)` to request U and J explicitly; its focused
  ForwardDiff-backed source/probe regression passes.
- The extractor now schema-checks and emits steps 710–719 plus hashes.
- The H200 harness now hard-fails CPU/GPU direct U and J integrity, executes
  P4/P8 × Float32/Float64 × rho_t 4.211/4.789, checks allocations, transfer
  counters and unchanged-state replay, emits isolated stage and end-to-end
  budget rows, and performs the true residency A/B above.
- Raw output and every report/CSV are hashed by the run driver.

These are locally parse/static tested only. Item 049 remains unapproved until
048 is approved, the corrected H200 run completes, and the user selects the
residency default from its A/B evidence.

## Results (2026-08-22, H200 job 13305555) — accepted

Three submissions preceded the complete run: `13304874` (harness ternary
parse error), `13305165` (harness `rec(...)` keyword-call bug — implicit
keywords require a leading `;`), and `13305443` (genuine FastMultipole bug:
Float64 literals `0.5`/`3.0` in `src/tree.jl` branch geometry promoted
`Branch{Float32}` trees to `Branch{Float64}`, breaking the legacy CPU octree
at Float32 — latent because CPU FMM was only ever tested at Float64; fixed
with bit-identical-at-Float64 `/2` and `3` forms, verified by a local
Float32+Float64 CPU repro through `vpm.UJ_fmm`). Job `13305555` then executed
all five stages in 7m35s on an H200; every artifact hashed and verified
(sha256) into `data/rotor_field_gpu_verification/results-13305555/`.

**Verdict: accepted by the user 2026-08-22.** All documented calibration
anchors are met or exceeded. The run's 49 harness-gate FAILs (which made the
job exit 1 by design) were root-caused to four families of harness-side gate
misapplication — none is a code regression, and no delivered-accuracy gate
was weakened:

1. **Allocation contracts (32 FAILs)** — the 048-approved contract
   (host <= 4096 B, device == 0) is a *lifecycle-layer* assertion inside
   `run_cuda_radix_lifecycle!`; the harness measured `@allocated
   vpm.UJ_fmm(...)` at the *wrapper* layer, where 048 had previously accepted
   160–192 KB host / <=512 B device bands. Measured wrapper-layer values:
   105–130 KB host (inside the old band) and 2724–3784 B device (the
   wrapper's domain/sigma guard runs `minimum`/`maximum` GPU reductions per
   call, `FLOWVPM_fmm_radix.jl:393-398`, each allocating a small device
   buffer). Transfer counters were 0 everywhere and graph exec/epoch reuse
   passed everywhere, so the lifecycle contract itself is uncontradicted.
   Direct lifecycle-layer measurement deferred to `053`.
2. **`replay_bitwise_equal` (8 FAILs)** — 048 deliberately adopted
   *error-bounded* replay gates (<=1.5x first-call error; parity 1e-10 F64 /
   1e-4 F32) rather than bitwise equality, since atomic accumulation
   ordering is not deterministic; job 13303399 measured replay drift
   1e-17–1e-18 F64. The harness's bitwise `isequal` check was stricter than
   the accepted design. Error-bounded replay measurement through the wrapper
   path deferred to `053`.
3. **Accuracy `j_rel_rms`/`sfs_rel_rms` (7 FAILs)** — the 5e-4 F64 / 1e-3
   F32 SFS gates are 048 *operating-point* gates; the harness applied them
   across the whole P4/P8 bracket matrix. P4 SFS at 2.7–2.8e-3 is exactly
   the documented P4 inadequacy that motivated D14's move to P=6 (measured
   e_sfs 1.46e-4 at P6 in 048, 3.4x margin); P8 at production rho 4.789
   delivers 5.9e-5 F64. J at P4 (5.5–5.8e-4) beats every documented J
   anchor (1.9e-3–1.64e-2); no 5e-4 J gate exists in any doc.
4. **Float32 direct integrity (2 FAILs)** — not a recurrence of the D9
   2.6e-4 kernel bug (fixed in baf8fb3): the F64 twin of the metric passes
   at 1.5e-14, proving path agreement. At F32, CPU-sequential vs
   GPU-parallel accumulation over 210k sources has an ordering-noise floor
   ~3e-5 (sqrt(N)*eps32), higher for J; measured 6.1e-5 (u) / 1.5e-4 (J)
   against a 5e-5 gate set below that floor. The independent F32 SFS
   cross-check passes at 6.2e-6.

**Acceptance matrix (GPU FMM vs GPU direct, full 210k field):** u_rel_rms
passes everywhere (1.1e-4 at P4, 7.6e-6/1.7e-6 at P8 F64). Residency parity:
150/150 PASS at 1e-11 (all 46 particle rows, scratch, np/nt/t, splitting
state, 10 snapshots). Counters: 144/144 PASS (all transfer counters zero,
route/operator unchanged). Cache registry release/rebuild contracts: PASS.

**Budget (snapshot 710, production P=6 / rho_t=4.789 / F64):**

| stage | median (s) | of 3.3 s target |
|---|---|---|
| full resident RK3 step | 0.2910 | 8.8% |
| full upload RK3 step | 0.3040 | 9.2% |
| ujsfs complete evaluation | 0.0939 | 2.8% |
| nearfield UJ | 0.0557 | 1.7% |
| SFS | 0.0281 | 0.9% |
| B2M / M2L / L2B | 0.0043 / 0.0037 / 0.0045 | ~0.1% each |
| tree refresh / M2M / L2L | <=0.0007 each | <0.03% |
| H2D / D2H 46xN transfer | 0.0044 / 0.0043 | 0.13% |
| RK3 integrator residual | 0.0092 | 0.3% |

Resident RK3 at 291 ms/step matches the anchors (199 ms at old P4 defaults,
"somewhat higher at P6" expected) and is 11x under the 3.3 s/step target.
The isolated-stage sum (0.098 s) is diagnostic only (nearfield/farfield
overlap in production).

**Residency A/B (10 snapshots 710–719, production point, true same-job
interleaved identical-state A/B):** resident median 0.2940 s/step, upload
median 0.3060 s/step, difference +0.0120 s (+4.1%), per-snapshot spread
+0.0112 to +0.0137 s. Both arms integrate RK3 on the GPU; the upload arm
adds one contiguous 46xN H2D before and D2H after each step. Parity between
modes: 150/150 at 1e-11. **User selection (2026-08-22): upload-per-step
ships** — rationale: compatibility with monitors that trim particles or
otherwise modify body states between steps; the +4% cost is well inside
budget. Recorded as D15. No code default changes: residency is a trait of
the field's array type (`fmm.residency`, `FLOWVPM_fmm_radix.jl:51-52`), so
production callers construct Array-backed particle fields.

This run also provides the first regression coverage of the D14 production
defaults (expansion_order 6, rho_t 4.789) on H200.
