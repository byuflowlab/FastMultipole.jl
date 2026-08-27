# 052b — Multi-rotor + IGE GPU extension (FLOWPanel BRAINSTORM/022 Phase 6)

**Status:** IN PROGRESS `2026-08-26` — Phase A.1 (shared pfield + warmstart
fixes) running concurrently with 052/052a; A.2+ gated on the 052 verdict.
Subtask checklist: `052b-plan-2026-08-26.md`. No jobs submitted.
**Parent items:** extends `052` (single-rotor 018-driver GPU pipeline) and
`052a` (GH200 verdict: discrete-memory path stays; no unified memory in the
graph-captured step).
**Consumer:** FLOWPanel `BRAINSTORM/022_rotor_hover_ground_effect/phase_06_multirotor_gpu.md`
— its six acceptance runs ARE this item's Phase D.

## Objective

Extend the 052 GPU pipeline so the 022 multi-rotor ground-effect driver
(`FLOWPanel.jl/examples/rotor_hover_ground_effect.jl`, `NROTORS` ∈ {1,2,4},
`GROUND_ENABLE` ∈ {false,true}) runs its **major cost steps on GPU** (wake
FMM, panel passes, solve) and completes **10 revolutions (360 steps,
NT = 36) in ≤ 2 h walltime per case** for all six shapes
(1/2/4 rotors × OGE/IGE). Per-case budget ⇒ **≤ 20 s/step average**; the
4-rotor IGE case is the binding constraint (052's single-rotor prediction is
3.124 s/step on H200).

## Binding rulings inherited

- **022 ruling 7 (Ryan 2026-08-25): single particle field.** The entire
  simulation uses ONE shared pfield for all rotors (one FMM tree over all
  wake particles). The driver currently builds one `PanelParticleWake` per
  rotor (`rotor_hover_ground_effect.jl:815`; wakes tuple at :1258/:1262) —
  this must be fixed on the FLOWPanel side FIRST (Ryan expects a minimal
  change: all rotor wakes referencing one pfield, or one merged wake).
  NROTORS=1 must stay bit-identical to the legacy path. This ruling
  simplifies the FastMultipole side substantially: the wake FMM stays a
  single-system `targets === sources` self-interaction, so the radix-path
  v1 restriction (`050`:34-39) is NOT violated by multi-rotor.
- **052a Phase C verdict:** discrete-memory path only; no managed arrays in
  the graph-captured step (CUDA error-900).
- **022 matched-settings contract (ruling 4)** and operating point
  (ruling 1); `FLOWPANEL_FILAMENT_REG=vatistas` pin (025 hazard).

## Phase gates

### Phase A — driver conformance + gap audit (entry: 052 acceptance verdict landed)

1. **Single-pfield driver change (FLOWPanel side, ruling 7):** implement +
   verify. Gates: NROTORS=1 bit-identical to legacy; NROTORS=2 CPU smoke
   matches the current per-rotor-pfield construction within solver tolerance
   (or a logged explanation of any physical difference — shared-pfield
   inter-rotor wake interaction is the INTENDED physics).
2. **Wake↔bodies pass audit (both directions).** 051's option B′ couples
   per system pair, so with the shared pfield (ruling 7) the wake side is
   one system but the body side multiplies: at 4 rotors IGE there are
   **5 body systems** (4 rotors + ground disc) ⇒ ~5 rectangular passes per
   direction per step under B′.
   - **wake→bodies (pass 2, particles→panel centers, rectangular GPU
     direct kernel; 2.020 s U-only for one rotor's 36,752 panels):** the
     dominant risk — cost ∝ panels×particles, so 4× particles × (4× panels
     + 4,752 ground panels) ≈ 16–17× ⇒ ~32–34 s/step naive extrapolation,
     over the 20 s/step budget on its own. Levers, in order: 041k escape
     hatches (far-field singular switch, fast-math, register blocking;
     1.6–1.7× F32 on all-pairs), then promoting this pass to an
     FMM-rectangular path.
   - **bodies→wake (pass 1, panels→particles; 0.124 s for one rotor):**
     scales gently, but under B′ it repeats per body — the shared particle
     tree/staging must be built **once per step and reused** across all
     rectangular passes and the wake self-FMM; any per-pass rebuild is
     pure overhead to eliminate.
   - Decision with probe numbers: does B′ + tree reuse + levers fit
     20 s/step, or must Phase B take the step toward option-A unified
     multi-system `fmm!` (heterogeneous sources/targets on GPU,
     `050`:140-145)?
   Measure, don't guess: reduced-shape timings → extrapolate vs budget.
3. **Multi-body solve audit:** 052's GPU solve is a single-body
   source-potential S gemv. IGE uses block Gauss–Seidel
   `solve!(bodies::Tuple, solvers::Tuple)` (`FLOWPanel_solver.jl:1454`)
   with the paneled ground disc as an extra body (there is no image/mirror
   system anywhere in the stack). Decide with numbers: per-block S gemvs +
   GPU cross-influence, vs cross-blocks on CPU — adopt CPU cross-blocks
   only with a measured justification against the budget.

Exit: written gap list with measured per-pass costs and a go/no-go
extrapolation table for all six shapes.

### Phase B — implementation of confirmed gaps

Whatever Phase A confirms, with CPU/GPU parity gates on reduced problems
(052 stage-b pattern: pass-by-pass parity at a reduced operating point
before any production-shape run). Parity tolerance defined here becomes the
022 Phase 6 acceptance tolerance.

### Phase C — probe runs (1 rev, per-pass timers, all six shapes)

One combined sbatch chain on H200 (long queue waits — combine stages per
the cluster-jobs guidance). Go/no-go: linear extrapolation of each shape to
10 revs vs 2 h. If p022g_4r_ige extrapolates over budget, report s/step +
dominant pass + bottleneck; escape hatches applied only as a logged ruling
candidate, never silently.

### Phase D — acceptance (= 022 Phase 6 cases)

The six 10-rev runs (`p022g_{1r,2r,4r}_{oge,ige}`, fine mesh 45_185_ct4,
IGE at h/R = 1.0 with the Phase-3-ruled damping knobs). Pass criteria are
pre-registered in `phase_06_multirotor_gpu.md` (≤ 2 h walltime, CPU-parity
CT trace, no blow-up through rev 10). Blocked on the 022 Phase-3
particle-policy verdict (the `p022m` launchers carry a placeholder
`GROUND_DAMP_BAND_R=0`).

## Cross-repo ownership

- FLOWPanel.jl: driver + single-pfield change + launchers
  (`examples/rotor_hover_ground_effect.jl`,
  `examples/run_rotor_multi_ground_effect_hpc.slurm.sh`,
  `scripts/p022m_submit_*.sh` as CPU reference arms;
  `scripts/p022_harvest.py` for harvest).
- FastMultipole: GPU kernels/seams (radix path, rectangular panel kernels,
  S gemv extension).

## Log

- 2026-08-25 — Item staged from Ryan's directive (BRAINSTORM/022 Phase 6:
  multi-rotor GPU support; 10 revs × 6 cases ≤ 2 h each, major cost on
  GPU). Confirmed the per-rotor pfield gap against ruling 7 by inspection
  of the driver. No code, no jobs.
- 2026-08-26 — Subtask checklist staged in `052b-plan-2026-08-26.md`; Phase
  A.1 started concurrently with 052/052a (A.2+ stays gated on the 052
  verdict). KEY FINDING: the Ruling-7 shared-pfield change was already
  implemented as an uncommitted FLOWPanel diff (found in
  `stash@{0}: shared-pfield-wip` on `92a607d`; re-applied, stash kept).
  Ryan rulings (2026-08-26): ADOPT the diff (audit/fix/verify as Phase
  A.1); FIX warmstart bugs properly, no error-guards. This session: audit
  map recorded in the plan file; warmstart replay double-convection and
  shared-field particle-load bugs fixed (`FLOWPanel_warmstart.jl`); unit
  testset added (`test/runtests_unit_simulate.jl`). Nothing committed in
  FLOWPanel.jl (tree carries other sessions' work).

- 2026-08-26 (later session) — Blocking warm-start NaN root-caused
  (missing per-step staging before the section-5 end-of-step replay:
  `wake.freestream`/`velocity_te` zero on fresh objects → un-convected
  row-1 coincident with replayed shed row → singular filament) and FIXED
  (`FLOWPanel_warmstart.jl` section 5.0). Two pre-existing test-code
  defects fixed in `runtests_unit_solver.jl` ("Backslash shared operator
  reuse": `==`→`≈`, tuple indexing). Suites: warmstart 153/153,
  simulate 199/199, solver 412/412. All uncommitted. Details in
  "Blocker RESOLVED" below.

## Checkpoint follow-ups COMPLETE — 2026-08-26 (session after "Blocker RESOLVED")

All items from the checkpoint's "After the blocker is fixed" list, verified:

- **Ground/multirotor restart-guard removal ACCEPTED.** Both removed guard
  conditions (NROTORS>1, ground body) are exercised by the now-green
  "multi-rotor IGE warm start" testset (113/113, continued-vs-uninterrupted
  at 1e-10/1e-11). Driver comment at the guard site updated to record both.
- **Syntax error found+fixed by the setup-only parse:** driver line 2040
  had `\"` escapes inside a `$()` interpolation (metadata
  `gpu_influence_mode` line) — a ParseError that would have killed every
  real run at load. Fixed (plain nested quotes); whole-file
  `Meta.parseall` clean.
- **Setup-only parses GREEN (exit 0):** 4r IGE (`BERNOULLI_ONLY=true`,
  required by the driver's line-345 contract; the GPU carrier already
  exports it) and 1r OGE. Both print the Backslash construction banner and
  the 468-step contract.
- **Diff overlap inspection CLEAN:** every hunk in the five src files +
  driver classifies into the approved 052b/052c categories; no foreign
  hunks, no default-path arithmetic changes beyond the approved warmstart
  fix and the 052c-approved `.vtp compress=false`.
- **`git diff --check` clean.** Warmstart suite re-verified 153/153 after
  the compress=false io change (round trip through ReadVTK exact).
- NOTE (052c P1.5, resolved later this session by Ryan's ruling): the
  particle .vtp visual series is now Float32+uncompressed with a
  full-precision `_particles_fp64` checkpoint sidecar that the warm-start
  loader prefers — warmstart 153/153 + simulate 199/199 re-verified with
  the final form; see 052c plan "P0 / P1.5 status".

Still pending Ryan: commits, deployment, job submission.

## Independent review — 2026-08-26 (later session)

The "run the test before doing anything else" instruction below was
executed (exact resume command, fresh depot/cwd). Result: **still NOT
green — 49 pass / 64 fail / 0 error** in `runtests_unit_warmstart.jl`
("multi-rotor IGE warm start: shared and legacy particle layouts",
29.6 s; log:
scratchpad `052b_warmstart_test.log` of session cf5735af, first failure
at test line 341). **The failure signature CHANGED with the untested
last edit** (backend_wake → small `FastMultipoleBackend`): now the
FMM-backend arm (`audit_c`) produces NaN rotor forces / ground strength
/ panel-wake strength and block-GS hits the 50-iteration cap with
final_max_delta=NaN, while the `DirectBackend` arm (`audit_a`) is valid
and converges in 4 iterations — the *inverse* of the pre-edit signature
(NaN under DirectBackend). So the edit did not mask a DirectBackend
restart defect; it exposed a distinct NaN path in the
FastMultipoleBackend continuation. Next 052b agent: treat this as the
blocker; both backend arms have now NaN'd under different
configurations, pointing at continuation-state initialization rather
than one backend. No fixes attempted in the review session; worktree
untouched apart from this note.

## Blocker RESOLVED — 2026-08-26 (later session, after the review above)

**Root cause (diagnosed by A/B bisection, not inspection):** the
warm-start NaN was never a backend defect. `simulate_warmstart!` section
5 replays the skipped end-of-step `propagate!`/`shed_wake!` on freshly
constructed wakes/bodies whose **per-step runtime staging was never
applied**: `_sa_reset_freestream_kinematic!` runs only inside
`simulate!`'s step loop, so at replay time `wake.freestream` was still
zero and the bodies' `velocity_te` still zero. With
`freestream_convection=true`, `propagate!` convects rows by
`dt*wake.freestream` (`FLOWPanel_wake.jl:575-583`) → the old row-1
never moved off the TE → the replayed shed deposited the new row-1
**coincident** with it → singular vortex-ring evaluation → NaN body
velocities → block-GS NaN (cap 50). Both backends NaN'd historically
because the defect is geometric, not numerical. Evidence chain
(scratchpad `052b_nan_repro.jl` of session cf5735af): memory-continued
arm (same replay, no disk round-trip) was finite while the disk arm
NaN'd; pre-continuation deep field diff showed exactly
`panel_wake.freestream` (0 vs U∞), `velocity_te` (0 vs U∞), and nodes
offset by exactly `U∞·dt = 0.008`.

**Fix (FLOWPanel_warmstart.jl, new section 5.0):** before the replay,
restore precisely what the end-of-step consumers read: body `reset!` +
`apply_freestream!` + `kinematic_velocity!` (reproduces `velocity_te`
exactly: 0 + uinf + kinematic), and `pw.freestream .= uinf_replay` per
wake. Deliberately does NOT touch wake node velocities or particle U —
those were saved at io time post-solve (induced included) and restored
by the VTK loaders, which is exactly what the
`shed_with_induced_velocity` propagate branch must consume.

**Verification:** full `runtests_unit_warmstart.jl` now **153/153
green** — including the formerly blocking "multi-rotor IGE warm start"
testset (113/113) whose 1e-10/1e-11 comparisons against the
uninterrupted arm confirm the replay is exact, not merely finite. The
test's last-edit `backend_wake = FastMultipoleBackend(...)` config was
kept and passes; it is a valid production-representative regression
(the earlier DirectBackend NaN had the same root cause). Regression
suites: `runtests_unit_simulate.jl` **199/199**;
`runtests_unit_solver.jl` initially 407 pass / 1 fail / 1 error from
two pre-existing defects in the new "Backslash shared operator reuse
(052b)" testset (unrelated to the warmstart fix): a strict `==` between
a shared-operator solve and a fresh operator assembled at *translated*
coordinates (assembly is not bitwise translation-invariant — changed to
`≈ rtol=1e-12` with a comment), and invalid tuple indexing
`cells[:, (1, 2)]` (changed to vector indexing), whose error had been
aborting the testset tail. After those two test-code fixes:
**412/412**. The removed driver ground guard can now be re-evaluated
per the checkpoint's instructions. Commits still pending Ryan's
instruction.

## Context-reset checkpoint — 2026-08-26

Implementation is **partially complete and uncommitted** in the dirty sibling
`../FLOWPanel.jl` worktree. No HPC jobs were submitted and `deploy_022g.sh`
was not run. Preserve all unrelated worktree changes.

Completed or substantially implemented:

- `src/FLOWPanel_solver.jl`: opt-in `Backslash(...; shared_operator=owner)`
  with body/type/order/core/rigid-or-mirror geometry validation, aliased
  `G`/`Glu`, independent work buffers, construction timing, and rejection of
  `update_G=true`; unit coverage added in `test/runtests_unit_solver.jl`.
- Block-GS status capture (`block_gs_status`) plus timer seams for rotor/rotor
  and rotor/ground cross influence, cached panel solves, host `ldiv!`, and GS
  iteration count. The status cache currently uses `IdDict`; consider a
  non-retaining design before final handoff.
- `examples/rotor_hover_ground_effect.jl`: CuArray/radix eligibility and loud
  fallback policy, pinned particle FMM triple, shared rotor Backslash owner,
  construction banner/metadata, shared pfield wiring, and removal of the
  ground restart guard. Driver-local damped propagation has the shared-field
  and timer controls.
- Fine timer seams added in `FLOWPanel_gpu_influence.jl`,
  `FLOWPanel_wake.jl`, and the block-GS loop. Review the host rectangular
  combined-source classification before declaring the interaction ledger
  complete; ordinary timer-off block-GS arithmetic is intentionally
  unchanged.
- New owned GPU carrier and six wrappers:
  `examples/run_rotor_multi_ground_effect_gpu.slurm.sh`,
  `scripts/p022g_submit_{1r,2r,4r}_{oge,ige}.sh`, and
  `scripts/deploy_022g.sh`. They are executable. `bash -n` passed and all six
  `P022G_SETUP_ONLY=1` arms printed the fixed 414-step contract. The carrier
  pins Julia 1.11.7, CUDA/no-fallback, fine mesh, Gaussian regularization,
  h/R=1.5, depth=4.5, damp band 0.1R, no GPU-S, shared operator, revisions,
  GPU model, and host/device peak-memory capture. Deployment targets only
  `/home/rander39/projects/FLOWPanel-022g` and verifies SHA-256 checksums.
- `src/FLOWPanel_warmstart.jl`: replay now honors the original
  `particle_relax`, `diagnose_particle_gamma`, and `diagnostic_vertical`
  controls; shared pfields remain loaded/propagated once.

Verification already observed:

- Solver suite initially reached 396 passes with two errors: the new test's
  helper used a mismatched core size (subsequently fixed), and an unrelated
  sandbox VTK write failed. Re-run from `mktempdir()`.
- Existing warm-start sets continue to pass: smooth conversion 12/12 and
  basic PanelParticleWake consistency 14/14.
- Shell syntax and the six-case setup-only launcher matrix pass. No CUDA or
  cluster execution has been attempted.

Unresolved blocking test:

- The new `multi-rotor IGE warm start: shared and legacy particle layouts`
  test in `test/runtests_unit_warmstart.jl` is not green yet. The isolated
  nonempty shared/private VTP round trip is now separated from the physical
  continuation and passes its early assertions. The reduced continuation
  remains finite immediately before forwarding to `simulate!`, but the first
  continued wake-influence step produces NaN body velocities under
  `DirectBackend`; uninterrupted execution is finite. Earlier diagnostics
  established that loaded body, active panel-wake, and particle storage are
  finite before continuation. Do not treat the removed driver ground guard as
  accepted until this is resolved.
- The **last edit before this checkpoint is untested**: the test's
  `common` options now set a small `FastMultipoleBackend` specifically for
  `backend_wake` while retaining `DirectBackend` elsewhere. Run the test
  before doing anything else. If that passes, decide whether this is a valid
  production-representative regression or merely masks a DirectBackend
  restart defect. The test currently uses an isolated nonempty VTP round trip
  and an empty-particle physical continuation.
- Final live-wake comparisons correctly restrict themselves to active rows;
  unused preallocated rows must not be compared because they may contain
  sentinel NaNs.

Resume commands (from `../FLOWPanel.jl`), using a temporary depot/cwd because
the sandbox cannot write the normal Julia cache and an existing generated VTK
path may be unwritable:

```bash
JULIA_DEPOT_PATH=/private/tmp/fp052b_julia_depot:/Users/ryan/.julia \
JULIA_NUM_THREADS=2 julia --project=. -e \
'cd(mktempdir()); include("/Users/ryan/Dropbox/research/projects/FLOWPanel.jl/test/runtests_unit_warmstart.jl")'

JULIA_DEPOT_PATH=/private/tmp/fp052b_julia_depot:/Users/ryan/.julia \
JULIA_NUM_THREADS=2 julia --project=. -e \
'cd(mktempdir()); include("/Users/ryan/Dropbox/research/projects/FLOWPanel.jl/test/runtests_unit_solver.jl")'
```

After the blocker is fixed: run simulate/replay/warmstart/solver suites,
perform a setup-only parse of `rotor_hover_ground_effect.jl`, inspect
path-specific diffs for accidental overlap, update the Phase-6 consumer
checklist and `START_HERE.md` only with verified facts, and run
`git diff --check`. Do not commit, deploy, or submit without a new explicit
instruction.
