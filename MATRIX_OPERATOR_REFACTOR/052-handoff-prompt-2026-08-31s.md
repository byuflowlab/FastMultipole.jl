# 052 handoff (session 2026-08-31s): relaunch done + repos committed; babysit jobs, then implement the reverse leg

## Prompt for the next agent

You are continuing task 052 after a context reset. Session 2026-08-31s
executed the 24h-pause relaunch (all recipes from the 30r handoff),
developed and deployed the 052g zero-M2L device fix, committed all three
repos, and started reverse-leg prep. **Your first actions, in order:**

1. **Re-arm the monitors** (they died with the old session; recipes in
   "Monitors" below). Check job states immediately — events may have
   happened during the reset.
2. **Babysit the runs** (details below): twin-cancel when an LG twin
   starts, LineGauss perf check from its early step times, gpu40 to
   completion, cpu40-r3 progress.
3. **Main work item: implement the reverse leg** (particles→panels FMM;
   Ryan approved implement-NOW on 2026-08-31). Scope + plan-of-attack
   below. Do this while the jobs run.

## Cluster state (as of 2026-08-31 ~01:00 MDT)

All on orc (`ssh orc 'bash -lc "source /etc/profile; ..."'`; user
rander39; read memory `orc-cluster-access.md` — it now records the
CRITICAL banner-glue quirk: the login banner ends WITHOUT a newline and
glues to the first stdout line, so start every remote command with a bare
`echo` and never rely on `grep "^TAG:"` for the first line; also combined
multi-job `squeue -j a,b`/`sacct -j` queries intermittently OMIT pending
jobs — query job ids ONE AT A TIME; `~/st052_probe.sh` on orc does this).

- **13518480 `fp-il-s038v-gpu40`** (018 silo `~/FLOWPanel-018-gpu-gh200`,
  gh200): RUNNING since ~00:26. Warm restart from step 1059 confirmed
  (`simulate_warmstart!: resuming from step 1059`); at last check it was
  inside step 1060/1475 — the step that killed the previous attempt
  (13513892) via the zero-M2L device bug. 052g fix is in the silo; if the
  job survived step 1060 the fix is validated. Baseline 44 s/step at
  death; 415 steps remain in 24 h wall. Log:
  `~/FLOWPanel-018-gpu-gh200/logs/slurm/slurm-fp-il-s038v-gpu40-13518480.{out,err}`.
- **13518479 `fp-il-s038v-cpu40-r3`** (m12, `~/projects/FLOWPanel.jl`):
  RUNNING, restarted from step 430 (last complete save: body VTU +
  wake filaments + particles all at 430). ~150-170 s/step, 1475 total.
  Its stack (~/projects/FastMultipole + ~/projects/FLOWVPM.jl) carries the
  052f patches; host path only, immune to the CUDA zero-M2L bug.
- **LG twins `fp-il-s038v-gpu40lg`** (052 silo `~/FLOWPanel-052-h200`,
  h200, LineGauss, `--dependency=singleton`): **13518478** (m13h,
  resubmitted this session after the original 13512551 died — see
  Krylov note) + **13512552** (eng, original, released from hold). Both
  PENDING. When ONE starts: `scancel` the OTHER yourself in the
  foreground (monitors embedding scancel get classifier-blocked — learned
  three times). Then run the **LineGauss near-field perf check** (Ryan
  approved): compare the first ~20 step times in its log against the
  gaussian-era baseline ~4.9-5.3 s/step; report the delta (LineGauss = 4
  erf + 1 exp per edge vs 1 expm1). Logs in
  `~/FLOWPanel-052-h200/logs/slurm/slurm-fp-il-s038v-gpu40lg-<id>.{out,err}`.
- **Krylov incident (resolved):** 13512551 died in 3 s — `~/fm052env-h200`
  was missing package Krylov. Fixed via `Pkg.instantiate()` on that env
  (spack julia 1.11.7 path in the log header). If an LG twin fails at
  package load again, check the env the same way.

## What landed this session (do NOT redo)

1. **052g zero-M2L device fix.** Root cause: `_cuda_hier_cache_windows!`
   (FastMultipole `src/translate_batched_cuda.jl`) sets `win_valid=true`
   with `noffsets == 0` but never allocates `win_class`/`win_sources`/
   `win_targets`; `_launch_cuda_hierarchical_m2l_cached!` then typeasserts
   `hctx.win_class::CuVector{Int32}` on `nothing` (job 13513892 death,
   step 1060). Fix: early-return after clearing locals when
   `total_routes == 0 || win_class === nothing` (publishes zero route
   count; mirrors host empty-route behavior). Applied locally (committed)
   AND to BOTH silos via `~/patch_052g.py` (count==1 assert,
   verify-then-write, `.bak-052g` backups beside the files) — silo dirs
   `~/FastMultipole-018-gpu-gh200`, `~/FastMultipole-052-h200`. Verified
   by inspection only; the running gpu40 job is the live test.
2. **Stage-7 test updated** (`test/cuda_radix_interface_test.jl`): old
   `@test_throws` device-gate expectation replaced with 052f warn+demote
   (`@test_logs (:warn, r"Falling back to the all-direct zero-M2L")`) +
   `bc.ell == 2` + device-vs-host parity `fmm!` on the demoted cache
   (exercises 052g). NOT yet run on a GPU node — bundle the device
   testsets into the next GPU job that goes up anyway (do not pay a
   separate queue wait).
3. **All three repos committed** (tracked trees clean; Ryan approved):
   - FastMultipole `flowpanel-20260817`: 52d9fbf4 leak fix (grow-only
     scatter buffer), 9c812a92 ulp box tolerance, d938ba68 052f demotion,
     2c6dd60f 052g + test, fe38d37a 052d cross-stencil code, 9bbac204
     docs, 11bcf0d1 prototypes/scripts.
   - FLOWVPM.jl `flowpanel`: 7601c20 052f zero-M2L fallback, f035807
     GC-after-cache-drop, d07b3c1 sigma_guard :ceil.
   - FLOWPanel.jl `fastmultipole`: 7fbd68a src+test (cross-pass A-F +
     LineGauss), 53ca8e9 examples+scripts, a45f6eb benchmarks, 697f862
     docs/plans, 9030e9a data prune.
   Untracked junk intentionally left (generated data, stuff*, pngs,
   .CondaPkg, courier/fx/pocket, benchmark/results 30M, Manifest.toml in
   FLOWPanel). Nothing pushed — pushing is Ryan's call.
   Smoke-verified: `using FastMultipole` clean; cross_stencil_test
   487,693/487,693 pass (3.9 s).
4. **Leakprobe cleanup (Ryan-approved):** archived monitors+tomls+CT csvs
   (8 files, 803K) to
   `~/FLOWPanel-018-gpu-gh200/data/scr_p019_s038v_leakprobe.archive.tar.gz`,
   dir renamed to `scr_p019_s038v_leakprobe.todelete` (classifier blocks
   remote rm -rf; Ryan runs the final
   `rm -rf .../scr_p019_s038v_leakprobe.todelete` himself).
5. **Decisions recorded:** xverify-gate = **debug-only** (Ryan,
   2026-08-31): keep `PANEL_INFLUENCE_FMM_XVERIFY` off in production,
   spot-check runs only — this is the status quo, NO code change needed.
   Notebook entries: NOT yet (do not draft until Ryan asks).

## Monitors to re-arm (notify-only; session-local, died at reset)

1. **Job states:** loop ~300 s:
   `ssh orc 'bash -lc "source /etc/profile; bash ~/st052_probe.sh"'`,
   grep `ST8:`, emit only changed lines (keep prev-state var, comm -13).
   The probe script already exists on orc and handles the banner + one-id
   quirks. Job ids inside it: 13518478 13512552 13518479 13518480 (edit it
   when ids change, e.g. after twin-cancel).
2. **gpu40 log signals:** loop ~300 s over the 13518480 out/err logs
   grepping `resuming from step|Falling back|GATE|LoadError|ERROR|CUDA
   error|Killed|OOM|launcher done` (out, head -15) + errors (err, head -5);
   lead the remote command with `echo`; emit new lines only (sort -u +
   comm). On any ERROR: read the log tail, diagnose before acting.

## gpu40 landing recipe (when it finishes)

Report last steps + GATE lines + memory tail. Harvest: stitch
`scr_p019_s038v_gpu40.crashed1061/monitors/*.csv` (steps <=1059) with the
new dir's (>=1060), no overlap; VTK series in the new dir starts at 1060.
Confirm/append the landing note to
`FLOWPanel.jl/plans/sigma_vpm_illustrations_20260827/HANDOFF.md`
(append-only, NEVER tick its checkboxes). THEN do the `.crashed1061`
cleanup, same archive-first pattern as leakprobe: tar monitors+tomls+csvs,
rename dir to `.todelete`, tell Ryan the rm one-liner (25G).

## MAIN WORK: reverse-leg implementation (Ryan approved implement-NOW)

The reverse leg = particles→panels influence (`wake_to_rotor_panels`
0.127 + `wake_to_probes` 0.100 s/step at np≈209k on H200 = ~3.9% of
5.79 s/step; ~0.5 s/step projected at 4-rotor). Full assessment:
`052-handoff-prompt-2026-08-29o.md:71-96`. Scope (2-4 sessions incl.
GH200 validation; env-gated like the forward pass, dense fallback stays):

- (a) particle-side multipoles on the cross grid: either device B2M for
  point vortices on the two-occupancy grid (device vortex B2M kernels
  exist in FastMultipole) or a seam reusing radix self-pass multipoles
  (cheaper at runtime, more plumbing: node mapping radix tree ↔ cross
  grid). Assess both, pick one, justify.
- (b) reversed route generation (particle nodes → panel nodes): needs a
  panel-side dense `node_at` (currently particle-side only).
- (c) L2B at panel control points + wiring into the solve RHS (U-only by
  default; U+J if `PANEL_WAKE_HESSIAN_TO_PARTICLES=true`).
- (d) xverify harness vs the dense leg (debug-only env gate, matching the
  forward pass's `PANEL_INFLUENCE_FMM_XVERIFY` pattern).

Machinery already in place (29o assessment): two-occupancy device
producer `refresh_cross_producers!` (panel occupancy over ~37k control
points exists in every cross entry), M2L operator tables + class-slot
machinery, L2L/L2B device kernels, near-field direct machinery, the
leak-safe capacity pattern.

**Code map (Explore agent, 2026-08-31 — verified file:line; trust but
spot-check lines before editing):**

Forward chain (template): `_gpu_device_pfield_target!`
(`FLOWPanel_gpu_influence.jl:1376`) → gate `panel_influence_fmm_enabled()`
:1393-1402 → `_panel_cross_device!` :1254 (gate chain :1256-1274) → per
body `_cross_entry!` :1097 (builds `_CrossPassEntry` via FastMultipole
`device_cross_producer_context`/`device_cross_expansion_state`/
`cross_m2l_operators`/`cross_l2l_operators`/`device_cross_local_state`
:1111-1116) → `_cross_run_body!` :1155 → FastMultipole
`refresh_cross_producers!` (cross_stencil_cuda.jl:359) →
`refresh_cross_multipoles!` (:760, Stage B) → `refresh_cross_locals!`
(:956, Stage C M2L) → `finish_cross_locals!` (:987, Stage D L2L+L2B) →
`apply_cross_near!` (:1170, Stage E) → single accumulate
`Pd[U_INDEX] .+= d_out[2:4]` at gpu_influence.jl:1238.

Reverse-leg target: the dense leg is the `pass1` branch of
`_gpu_rect_influence!` (gpu_influence.jl:618-798; timer labels
:wake_to_rotor_panels/:wake_to_ground_panels/:wake_to_probes at
:768-770; `_gpu_direct_batch!` at :778). NOTE gpu_influence.jl:744 —
`fmm_bodies` selection is `!pass1 && ...`, so the FMM route NEVER fires
for pass1 today. Call path: `FLOWPanel_simulate.jl:713-717`
`_sa_wake_influence!` → `influence!` (FLOWPanel_solver.jl:2769); outputs
consumed via `_gpu_add_result!` (gpu_influence.jl:490 body.velocity/
potential → `boundary_condition!` solver.jl:2748/:2707; probes :525/:540).

What must change (3 hard-wired spots in FastMultipole cross_stencil_cuda.jl):
1. `_cross_generate_level!` :305-344 hardcodes `src = ctx.panels` (:308)
   and consumes particle-side `ctx.d_node_at` (:324/:336) — needs the
   reversed direction.
2. `refresh_cross_producers!` :383-390 scatters dense `d_node_at`
   occupancy for particles only — panels need one to be a route TARGET
   (panels already get full levels/geometry via `_cross_build_levels!`/
   `_cross_node_geometry!` :260-302, built symmetrically at :378-381).
3. Stage B is panel-specific (`_cross_panel_b2m_kernel!` :624,
   Source/Dipole, classic-phi `lhv=Val(false)`, arms table
   `_cross_b2m_arms` cross_stencil_host.jl:94); Stage E near
   (`_cross_near_kernel!` :1042) hardcodes `_rect_panel_pair`/
   `_rect_panel_potential` — needs a point-vortex source variant
   (candidates: `_cuda_direct_pairs_vortex_kernel!`
   translate_batched_cuda.jl:1740, `_vortex_pair_ug/ugh` :2669/:2675).

Reusable as-is: device point-vortex B2M exists in the radix FMM
(`_cuda_b2m_vortex_leaf_nodes_kernel!` translate_batched_cuda.jl:1249,
dispatched `_launch_cuda_b2m!(..., ::Type{<:Point{Vortex}})` :1343-1357,
phi+chi); `_cross_l2b_kernel!` (cross_stencil_cuda.jl:880-898) is a
generic point evaluator — panels' `body_node` is already populated in
Stage B via `_cross_body_node_kernel!` :776-777; M2L/L2L operator
builders (`CrossStencilTables`) are direction-agnostic pure geometry.
IMPORTANT semantics: forward cross pass is U-only (grad requested →
fallback, gpu_influence.jl fallback list :1256-1274) and panel B2M is
classic-phi — the reverse leg's vortex sources need Lamb-Helmholtz
phi+chi through M2L/L2L/L2B, so check the cross expansion state carries
the chi channel before reusing Stage C/D unchanged.

Env-gate/fallback template: two-tier `PANEL_INFLUENCE_FMM` (default off,
:824-830) + `PANEL_INFLUENCE_FMM_DEVICE` (default on within, :869-871);
tunables `PANEL_INFLUENCE_FMM_XQ/_XELL/_XP/_XRG` read live and frozen
per-entry via `_cross_config()` :885-900; fallback discipline: return
false via `_gpu_route_fallback!` BEFORE any output write, all fallible
work before the single accumulate, partial-accumulation mid-loop is a
hard error (:1306-1317). xverify template: `_panel_fmm_xverify()`
:872-874, host mirror + `_panel_fmm_evaluate!` :1345 reference + relU
print :1323-1325; dump hooks `PANEL_FMM_DUMP_DIR` :1025-1034/:1178-1217
with `download_cross_lists` (cross_stencil_cuda.jl:1207); consumer
`prototypes/052d_cross_stencil/p39_relU_attribution.jl`.

**Then:** write a subplan doc (`052h-reverse-leg-subplan-<date>.md`,
style of `052d-step4-subplan-2026-08-28.md`) proposing the stage split —
recommended order: panel-side `node_at` + reversed route generation
(pure plumbing, CPU-testable via the host cross oracle prototypes),
vortex B2M seam decision (device B2M on cross grid vs radix-multipole
reuse; assess node-mapping cost), Stage C/D chi-channel check, Stage E
vortex near kernel, FLOWPanel `pass1` gate + `_gpu_route_*` wiring +
xverify(d). Prototype + unit-test on CPU where possible; bundle device
validation into GPU jobs with the pending device-testset run.

## House rules (carried forward)

4 threads max locally; julia-test-runner for all runs/scripts;
refactor-docs-librarian for MATRIX_OPERATOR_REFACTOR doc questions;
verifier before reporting claimed numbers; never read `data/**`/`*.csv`/
`*.bin`; long output → scratchpad log then grep. ssh recipe + banner
quirk above; auth expires → ask Ryan to run `! ssh orc echo ok`. rsync
`--checksum`. GPU jobs authorized; combine stages into one sbatch.
Notebook writes need Ryan's approval FIRST (and he said not yet).
Monitors must be notify-only (no embedded scancel/rm). Commits are done;
do NOT commit further without Ryan's ask; never push.

## Open Ryan decisions (ask, don't assume)

- `.prev` archive disappearance (pre-30r): still unconfirmed it was him.
- Push commits to remotes? (not discussed)
- Notebook entries (deferred — wait for his go).
- LineGauss/cross-pass "production truth" gpu40 rerun later (29o item;
  current gpu40 is gaussian for comparability).
- Final `rm -rf` of the two `.todelete` dirs (leakprobe now, crashed1061
  post-harvest).
