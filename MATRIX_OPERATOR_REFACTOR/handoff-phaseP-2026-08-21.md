# Phase P handoff — 2026-08-21 (context-reset checkpoint)

Companion to `decisions-phaseP-2026-08-20.md` (D1–D10, read it first).
Repos: **projects/FastMultipole `flowpanel-20260817`**, **projects/FLOWVPM.jl
`flowpanel`**, **projects/FLOWPanel.jl `fastmultipole`** (user WIP, see below).
tmp3 is retired (D2). Cluster: ssh `orc`; trees `~/FastMultipole-046`,
`~/FLOWVPM-046`, `~/FLOWPanel-052`; envs `~/fm048env` (CUDA 6.3, radix-
validated), `~/fm052env` (FLOWPanel geo stack + CUDA 5.8.5, see D10 + below).

## Row status (Done ⇒ user approval still pending on every row)

- **046 Done** — merges landed both branches, suites green, 018 CPU smoke
  116 steps, tmp3 retired. (D1–D3)
- **047 Done** — settings surface `src/radix_settings.jl` (31 tunables,
  lock classes), construction-lock verified on device (late flip errors
  loudly, job 13247683 stage 4). Dispatch-cleanup refactors DEFERRED to 053
  punch list (D4).
- **048 Done** — device SFS in the radix lifecycle; H200 mechanical parity
  1e-15 F64; SFS delivered accuracy is J-error-bound with field-dependent
  amplification (D5, D7) — production knob is J accuracy, not the ζ pass.
- **049 Done** — p018 210k field on H200 (job 13247848): UJ 3.4e-4 PASS;
  SFS marginal 0.3 ms; device-resident RK3 step 0.199 s = 6% of the 3.3
  s/step budget; residency measurement ⇒ RECOMMEND device-resident
  (transfers 35% of step) — **default = open user checkpoint**. U_prev
  broadcast fix landed in FLOWVPM. Also fixed: FLOWVPM ext
  `gpu_interaction!` absolute r2>1e-6 cutoff (dropped sub-mm pairs, 2.6e-4
  — D9).
- **050 Done** — verdict `theory/panel-multisystem-scoping.md`: option B'
  (keep 3-pass structure; rectangular brute-force GPU kernels for cross
  passes; device dense nearfield-cache matvec for the solve, measure-first;
  radix framework untouched).
- **051 stages 1+2 Done, row not ticked** — `direct_rectangular!` +
  `RectangularGaussianErfVortex`/`RectangularPanelInfluence` (3 filament
  families) in FastMultipole; FLOWPanel seam (env-gated, default off).
  Parity: host 1e-15/1e-16 vs FLOWPanel `direct!` incl. per-family; H200
  kernels 1e-16 (job 13247858: pass1 0.124 s, pass2 2.03 s U-only @018
  shape); **CUDA seam parity passed on-device** (job 13247864 stage A).
- **052 in flight** — driver GPU arm built (VPM_ARRAYTYPE=cuarray +
  FLOWPANEL_GPU_INFLUENCE; DynamicSFS broadcast ports; host-mirror
  maintenance seams). **H200 job 13247880 running stages B (reduced CPU-vs-
  GPU CT/Γ comparison) and C (production-shape 1-rev GPU, s/step +
  per-pass timers)**; output `~/FLOWVPM-046/fp052-13247880.out`, compare
  tool `FLOWVPM.jl/scripts/fm052_compare.jl`. Step arithmetic so far:
  particles 0.2 + pass1 0.12 + pass2 2.0 ≈ 2.4 s ⇒ **the CPU panel solve
  (16–21 s) is the binding lever for <1 h/30 rev**; escape hatch = pull
  solve-GPU (dense matvec) forward.
- **053 not started.**

## CRITICAL: FLOWPanel working tree is UNCOMMITTED by design

The user's own WIP (radius_inflation/FILAMENT_REGULARIZATION etc.) shares
files with my seam; committing would sweep their hunks (D10). My additions:
`src/FLOWPanel_gpu_influence.jl`, `src/FLOWPanel_gpu_wake.jl`,
`examples/fm051_pass_parity.jl`, + minimal hooks in FLOWPanel.jl /
FLOWPanel_fmm.jl / FLOWPanel_wake.jl / FLOWPanel_simulate_monitors.jl /
FLOWPanel_formulation.jl / examples/rotor_hover_pressure_comparison.jl.
Everything archived (files + hook patch, which ALSO contains user WIP hunks
in the FLOWPanel.jl/simulate parts) under
`MATRIX_OPERATOR_REFACTOR/data/fm051_flowpanel_seam/`. **User should review
+ commit FLOWPanel themselves.** Their 2 pre-existing radius_inflation test
failures remain theirs (test-file drift observed mid-run 2026-08-20).

## Cluster env gotcha (cost 4 job cycles — D10)

CUDA ≥6.2 → CUDATools → PrettyTables 3, unsatisfiable with FLOWPanel's geo
pins (PrettyTables 2.x); env stacking fails (CUDATools' precompile workload
uses PT3 API); julia 1.12.6 barred (device-step segfault, job 13058191).
Working recipe: fm052env = FLOWPanel stack + **CUDA 5.8.5** + VSPGeom/GeoIO
(driver-level imports); FastMultipole CUDA compat widened to "5.8, 6";
version-adaptive BFloat16 binding at translate_batched_cuda.jl:8. CUDA 5.8
device stack validated so far only by stage A rectangular parity — stage B
(radix UJ+SFS in the GPU arm) is its first full-radix exercise; if it
misbehaves on 5.8, options are re-validating radix on 5.8 or splitting envs
per stage (fm048env for radix-only jobs still works on 6.3).

## Open user checkpoints (phase prose consolidated)

1. Row approvals: 046, 047, 048, 049, 050 (+051/052 when ticked).
2. Residency default (049 recommends device-resident; nothing flipped).
3. FLOWPanel seam review/commit (above).
4. 052 CT/Γ acceptance + whether to pull the solve-GPU lever forward.
5. 053 punch list already carries: dispatch cleanup (D4), SFS accuracy
   watch (D5/D7 — like-for-like at 052's CT gate), 049 sampled-reference
   discrepancy RESOLVED (eps2 fix, D9).

## Local artifacts

Results: `MATRIX_OPERATOR_REFACTOR/data/rotor_field_gpu_verification/`
(049 + fm051 CSVs/reports, p018 snapshot .bin sha 0d9136…155ab).
Tests added: FM `radix_settings_test.jl`, `direct_rectangular_test.jl`,
SFS testset in `device_system_interface_test.jl`; VPM SFS host/device
testsets + U_prev equivalence. All suites green at last run on every commit.
Background monitors from the old session die with it — **check job 13247880
manually**: `ssh orc 'tail -60 ~/FLOWVPM-046/fp052-13247880.out'`.
