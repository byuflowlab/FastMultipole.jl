# 052 Impl: FLOWPanel 018 Driver on GPU (end-to-end)

## Status and entry gate

**Staged `2026-08-20` (user direction). Not started.**

Entry gate: `049` and `051` complete and approved. This is the phase's
end-to-end deliverable row; its verdict may invoke the phase-prose escape
hatch (pull `054`/`055` levers forward as an addendum) if the <1 h target
is missed.

## Motivation

The phase objective made concrete: run the latest 018 driver on GPU and
measure whether the campaign's stretch target — **30 revolutions in <1 hour**
(1080 steps at ≤3.3 s/step avg, a 52–70× speedup over the 170–230 s/step
CPU baseline) — is achievable, or report the measured feasible wall time
with a breakdown of what binds. Today the 30-rev target walls out in 48 h
of cluster time (~1.7–2.3 h/rev on 64 cores).

## Objective

The latest 018 driver (`examples/rotor_hover_pressure_comparison.jl` via
the slurm case matrix) running end-to-end on GPU+CPU with correctness
verified against a CPU reference arm, and the performance verdict delivered:
30 rev < 1 h, or the measured feasible time + binding-constraint breakdown +
named `054`/`055` levers to pull forward.

## Method

### Stage 1 — GPU driver arm

Add a GPU arm to the driver/case-matrix (env-var knob, matching the
driver's all-knobs-via-env convention;
`examples/run_dji9443_hover_ct_hpc.slurm.sh` case matrix at :355-410,
`p018_L1_ov3` at :405 = OVERLAP 3.0, P_PER_STEP 14, MERGE_R_FACTOR 0.0052).
Resource envelope: 1×GPU + 64 CPU threads. Monitors/paraview output must
stay intact (writer convention `FLOWPanel_wake.jl:2170-2200`).

### Stage 2 — correctness arm

Run a CPU reference arm and the GPU arm over a settle window; correctness
gate = **CT and Γ(r/R) agreement** between arms over that window (tolerance
set from the campaign's own arm-to-arm scatter, recorded in the report).

### Stage 3 — performance run + breakdown

Long GPU run (30 revs if wall time allows; else the longest feasible) with
per-pass instrumentation. Deliverable: measured s/step trajectory vs
particle count, the per-pass breakdown showing what binds (extending the
`049` budget table to the real driver: shed/solve/wake/body/SFS/integrator/
I/O/callbacks), and extrapolated 30-rev wall time.

### Stage 4 — verdict

**30 revolutions in <1 h if achievable, else the measured feasible wall
time.** If <1 h is missed, the verdict names which `054`/`055` levers to
pull forward as an addendum to this phase (escape hatch in the phase
prose) — with the per-pass evidence for why those levers are the right
ones.

## Gates and verdict

- Correctness gate: CT/Γ(r/R) agreement vs the CPU arm over the settle
  window; monitors/paraview output intact.
- Performance deliverable as above, with job IDs and the per-pass
  breakdown.
- Any default-behavior change in the production repos needs explicit user
  approval (phase convention; `053` checkpoint).

## Artifacts

- Driver/launcher changes on the FLOWPanel branch; slurm scripts.
- `data/flowpanel_018_driver_gpu/` — timing CSVs, CT/Γ(r/R) comparison,
  per-pass breakdown, `report.md` with the verdict.

## Verification

- Same-case CPU vs GPU arms from the same commit; job IDs recorded;
  extrapolations shown with the measured per-step-vs-n scaling, not assumed
  constants.

## Recorded context (2026-08-20 staging)

**Driver:** `examples/rotor_hover_pressure_comparison.jl` (1501 lines, all
knobs via env vars); launcher `examples/run_dji9443_hover_ct_hpc.slurm.sh`
(`p018_*` case matrix at :355-410; `p018_L1_ov3` at :405 = OVERLAP 3.0,
P_PER_STEP 14, MERGE_R_FACTOR 0.0052; L1 = σ-ladder rung, ov3 = its overlap
arm). Config: 36,752 panels (45_185_ct4 mesh), NT=36 steps/rev, ~20 revs =
719 steps, THREADS=64, `-t 64`; ~181k particles at maturity (342k on the 6R
arm). Item 018 = `BRAINSTORM/INDEX.md:74`,
`018_dji9443_hover_convergence_campaign.md` (LIVE; Phase 16 chord–σ
co-scaling opened 2026-08-14). FLOWPanel repo:
`/Users/ryan/Dropbox/research/projects/FLOWPanel.jl` (NOT under tmp3).

**CPU baseline (023 profiling):** 170–230 s/step on 64 cores (~1.7–2.3
h/rev; 30-rev target walls out in 48 h); per-step cost ~linear in particle
count (~50–100 s per 100k); split = wake influence 64.2% / body 25.3% /
solve 9.3%; ~75% of a production step = `Estr_fmm!`; wake FMM velocity ~7 s;
body pass floor ~36 s (kerneloffset-radius-bound); 49% thread utilization.
Tuned point (MAC 0.6 / leaf 24) already 3.7× faster + 12× more accurate
than production knobs — the GPU arm should start from the tuned knobs.

**Budget arithmetic:** 30 revs = 1080 steps in <1 h ⇒ ≤3.3 s/step avg
(52–70× vs today); the ~36 s CPU body-pass floor alone breaks it ⇒ both the
particle side (GPU UJ+SFS) and the panel passes must be accelerated; the
"GPU + 64 CPU threads" fallback bounds achievable time.

**Escape-hatch levers on the shelf (041k):** far-field singular switch
(ρ²>42.25 F32 / 81 F64) + `__nv_fast_expf`/`__nv_erff` + 2-target register
blocking = 1.6–1.7× F32 on all-pairs (3.3e11 pairs/s ≈ 38% FP32 FMA peak);
F64 opt inert below n≈3e4.

**FMM step anchors (041a fig15, unitcube GPU best-uniform):** 1.31 ms @1e4,
7.40 @1e5, 92.3 @1e6.

**Paraview convention:** writer `src/FLOWPanel_wake.jl:2170-2200`
(`<path>/<wake>_particles/<wake>_particles.<idx>.vtp` + .pvd; arrays gamma,
sigma, vol, circulation, velocity, vorticity, C, SFS, velocity_gradient).

**Harness gotcha:** run the full step head (`maneuver!` +
reset/freestream/kinematic + `update_TE!`) before any influence eval.
