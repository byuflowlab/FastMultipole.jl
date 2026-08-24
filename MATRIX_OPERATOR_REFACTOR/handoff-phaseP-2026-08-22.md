# Phase P handoff — 2026-08-22 context reset (item 048 focus)

Supersedes `handoff-phaseP-2026-08-21.md` for item 048; other items unchanged
from that file. Working dirs: this repo (branch `flowpanel-20260817`) and
`../FLOWVPM.jl` (branch `flowpanel`). Cluster: `ssh orc`, trees
`~/FLOWVPM-046` + `~/FastMultipole-046`, env `~/fm048env` (Julia 1.11.7
pinned — 1.12 crashes device JIT; CUDA.jl 6.3.0; normal H200 QoS only, no
`--qos=test`). Sync/submit driver: `FLOWVPM.jl/scripts/cuda_048_submit.sh`.
Preserve unrelated dirty-worktree changes.

## Item status

- 046, 047: approved and closed.
- 049: local remediation APPROVED (fresh review); pending H200 acceptance and
  the user's residency-mode choice after a TRUE same-job A/B. Do not launch
  049's acceptance until 048 is resolved and approved. Never auto-select the
  residency mode (no 15% threshold rule).
- 048: all 47 failures of job 13298230 are ROOT-CAUSED as test-methodology
  issues — no production-code defect found. Full analysis is recorded in
  `048-impl-gpu-sfs-enablement.md` §"Device-run failure analysis and
  resolution (2026-08-22)". Summary:
  1. "Replay J defect" (j~0.11) = statics accumulation measurement artifact.
     FLOWVPM `_reset_particles` preserves statics' U/J; every UJ delivery
     accumulates into all targets; nothing consumes statics' U/J. 3 statics
     × 7 extra evaluations = sqrt(3/20000)-per-call linear growth = 0.10984
     exactly. CUDA-graph replay is accuracy-faithful (diag jobs 13299959 and
     13302646; artifacts + sha256 in `data/gpu_sfs_enablement/`, scripts
     `FLOWVPM.jl/scripts/fm048_replay_diag{,2}.jl`, `fm048_diag_submit.sh`).
  2. Strict delivered-E gates (5e-4 F64/1e-3 F32) were transplanted from the
     host-matrix regime (n=1500, ell=2, near_radius2=20) to n=2e4 derived
     shell where E is J-error-bound (E/J 2.05 cube / 14.1 wake) — moved back
     to their valid regime (new strict testset) + tuning sweep for
     production selection.
  3. Allocation gates asserted at the wrong layer: host ~100 KB = CUDA.jl
     launch bookkeeping (~25-30 GPU ops/step, fixed, not n-scaled); device
     272-400 B = CUDA.jl accumulate!/maximum library scratch
     (`translate_batched_cuda.jl:229,:6682`,
     `translate_batched_resident.jl:2085`); SFS adds zero device alloc;
     warm `run_cuda_radix_lifecycle!` (graph replay) is ~zero-alloc.

## User decisions in force (2026-08-22)

- Tuning sweep to find settings that are efficient AND pass accuracy: gate
  is strict 5e-4 F64 delivered E **on the p018 production field**; cube grid
  runs too and the cube-vs-p018 discrepancy must be reported explicitly.
- Sweep rides in the SAME H200 job as the corrected acceptance testset.
- Pareto-frontier configs (plus P=4/derived-q baselines) run on p018.
- rho_t candidates stay 4.211 (velocity/U) and 4.789 (Jacobian/J); never
  silently change the production default. User picks final settings and
  residency mode from presented results.

## Local changes staged (uncommitted, parse-checked)

FLOWVPM.jl:
- `test/runtests_gpu_fmm.jl`: `fmm034_uj_errors(...; skip=())` + statics
  convention comment.
- `test/runtests_gpu_fmm_device.jl`: SFS testset U/J parity uses
  `skip=static_indices` (2 sites); allocation restructure (consts
  FMM048_HOST_WRAPPER_BAND=160_000 / _SFS=192_000,
  FMM048_DEVICE_SCRATCH_BAND=512, FMM048_HOST_ALLOC_BUDGET=4096 for the
  lifecycle layer; no-growth + sfs-adds-none checks; lifecycle-layer
  assertions via `ffmm.run_cuda_radix_lifecycle!(state)`); strict gate
  removed from n=2e4 loop (J-bound `e_gate` kept, e_sfs recorded); new
  replay gates (err_replay u/j <= 1.5x first-call; replay-vs-body parity via
  runtime `set_radix_setting!(:CUDA_GRAPH_LIFECYCLE,false)` flip); NEW
  appended testset "strict tail-budget operating point" (n=1500, P4/P8 x
  F64/F32 x rho 4.211/4.789, ell=2, near_radius2=20, gates 5e-4/1e-3 on
  first AND warmed calls).
- `scripts/fm048_tuning_sweep.jl` (NEW): cube n=2e4 F64 grid P{4,6,8} x
  rho{4.211,4.789} x q{derived,14,17,20} (q caps at 20 — supported rigid
  set), accuracy measured on first (uncaptured) AND warmed (replayed) calls
  vs exact direct reference (built-in replay-drift signal), warmed-median
  timings, n_direct; frontier + baselines; F32 spot checks; p018 arms with
  the strict gate + discrepancy report; CSV.
- `scripts/cuda_048_run.sh`: stage 4b runs the sweep with provenance
  hashing (stages: 1 FM device tests, 2 runtests_gpu, 3 coupling tests,
  4 A/B matrix, 4b sweep, 5 047 lock check).
- `scripts/fm048_replay_diag.jl`, `fm048_replay_diag2.jl`,
  `fm048_diag_submit.sh`: diagnostics (already run; keep for provenance).

FastMultipole:
- `MATRIX_OPERATOR_REFACTOR/048-impl-gpu-sfs-enablement.md`: resolution
  section added before the Verdict.
- `data/gpu_sfs_enablement/`: vpm048diag-13299959.out
  (sha256 70d3a9f5...), vpm048diag-13302646.out (dcf5398b...),
  fm048_diag_13299959.log (f15ba382...), fm048_diag2_13302646.log
  (61b605b6...), vpm048diag-13302371.out (autotune-flag crash, superseded).

## Defects FIXED and reviews DONE (2026-08-22, second session)

The known open defect (D1) plus two more found by the fresh-context full
review (report received intact after reset) were all FIXED, parse-checked,
and then VERIFIED by a second adversarial fresh-context review — no open
defects remain in the staged changes:

1. **D1 (fixed)** — replay-vs-body parity in `runtests_gpu_fmm_device.jl`
   now compares DELIVERED particle U/J (`Array(gpu.particles)[uj_rows,
   active_indices]`, global order, static columns excluded) after the
   replayed call vs after a graph-off call, gates 1e-10 F64 / 1e-4 F32.
   The `:CUDA_GRAPH_LIFECYCLE` flip is inside try/finally and restores the
   SAVED prior value via `ffmm.radix_setting(:CUDA_GRAPH_LIFECYCLE)` (note:
   the getter is `radix_setting`, not `get_radix_setting`). Verified: the
   flip is `:runtime`-class and cannot trip `verify_locked_radix_settings`;
   expected parity ~1e-13 F64 / ~1e-6-1e-5 F32 (F32 margin ~10x — watch it).
2. **D2 (fixed)** — `fm048_tuning_sweep.jl:load_snapshot` now zeroes SFS
   rows (`A[vpm.SFS_INDEX, :] .= 0.0`) at load: the p018 snapshot carries
   LIVE SFS in every column (|max| ~ 3.6e5, zero statics — empirically
   confirmed) and `Estr_direct!` accumulates, so the reference would have
   been contaminated and every p018 arm would have spuriously FAILED the
   strict gate.
3. **D3 (fixed)** — the `e_replay` measure was statics-diluted into
   vacuousness (identical static sentinels dominated the denominator). Now
   computed on active-column SFS deltas: `fmm048_relrms((S_replay .-
   S_before)[:, active_indices], Sref_delta[:, active_indices])` —
   `sfs=true` resets active SFS every call, so the delta is per-call E.
4. Guards added: `graph_live` logged in the strict tail-budget testset
   (records whether the n=1500 point actually replays a graph; not gated);
   frontier-cap comment aligned with code (keeps most-accurate configs).

Review verdicts: everything else in the change set (both repos) was
verified ready — helper APIs, scoping, run/submit scripts, strict-testset
feasibility, doc claims. Non-blocking residuals noted: host `e_fmm <= 5e-4`
gate at `runtests_gpu_fmm.jl:615` is bounded by legacy-FMM error (fine
today); F32 replay-parity margin ~10x.

## Job IN FLIGHT

**H200 job 13302961** submitted 2026-08-22 ~11:07 via
`bash scripts/cuda_048_submit.sh` (rsynced both trees incl. all fixes,
env unchanged, p018_710 snapshot shipped to `~/FLOWVPM-046/data/fm048/`).
Started RUNNING ~11:09; expect ~1.5-2.5 h through stages 1-4b + 047 lock
check. Output: `~/FLOWVPM-046/vpm048-13302961.out` (+ sweep CSV and
provenance hashes per `cuda_048_run.sh`). The user said THEY will announce
when it finishes — do not poll/monitor unprompted.

## Next steps (in order)

1. When the user says job 13302961 is done: check `sacct` state, retrieve
   ALL artifacts (job .out, sweep CSV, any stage logs) + sha256 into
   `data/gpu_sfs_enablement/`; update the 048 doc with results; verdict
   against the calibration expectations below. If a stage failed, root-
   cause before touching any gate (do not weaken delivered-accuracy
   evidence to make suites pass).
2. Present the sweep Pareto/p018 results + cube-vs-p018 discrepancy; the
   USER picks production SFS settings — never auto-select.
3. If stage 3 passed and 048 is approved: proceed to 049's H200 acceptance
   + true same-job stage-4 residency A/B; user picks residency mode.
4. Offer (do not write unprompted) a notebook entry per user CLAUDE.md.

## Expectations for the H200 run (calibration, not gates)

- n=2e4 cases: first-call and replayed u ~ 3.7e-4, j ~ 2.0e-3 (cube P4
  F64); e_kernel ~ 1e-15 F64; e_sfs ~ 4.3e-3 cube P4 (J-bound, recorded).
- Strict testset (n=1500/q=20): expect 9e-5-4.1e-4 F64 (host matrix
  passed there; device mechanical parity ~1e-15).
- Wrapper allocs: host ~(100480, 122704), device (384, 384) per
  job 13302646. Lifecycle: host << 4096, device 0 expected — if device != 0
  on the graph path, investigate before loosening anything.
- Sweep p018 arms: production P=4/derived-q rows will likely FAIL the
  strict gate (wake-like E/J ~ 14); that is the point — the frontier shows
  what settings/cost would pass. Do not weaken the gate; report.

## Update — 2026-08-22 third session (048 CLOSED, 049 ready)

- Job 13302961 FAILED at stage 4 (eltype-narrowing bug in
  `fm048_ab_benchmark.jl` specs vector, first job with p018 arms; stages
  1–3 all passed and matched calibration). Fixed (explicit Union eltype),
  resubmitted as **job 13303399: COMPLETED, all stages 1–5 green**.
  Artifacts + sha256s in `data/gpu_sfs_enablement/` (see
  `sha256_13303399.txt`); results recorded in the 048 doc.
- **048 COMPLETE and approved.** User selected production SFS settings
  (D14): **P=6, rho_t=4.789, derived q** (p018 e_sfs 1.46e-4, 3.4x gate
  margin, +3% cost). Implemented as coupling defaults in
  `FLOWVPM_fmm_radix.jl` (`expansion_order` 4→6,
  `_PARTITIONED_RHO_T_DEFAULT` 3.668→4.789); default-assertion tests in
  `runtests_gpu_fmm.jl` updated. Regression coverage of the new defaults
  rides in the 049 job.
- 049: harness `fm049_rotor_verify.jl` retargeted so the residency A/B,
  budget, and profile stages run at the production point (`P_PROD=6`,
  `RHO_PROD=4.789`, snaps 710:719, single rho — halves that stage);
  acceptance matrix (P4/P8 x F32/F64 x both rhos) unchanged. Parse-checked.
  `fm049_submit.sh` validates snapshots+manifest and is ready — submission
  pending (permission classifier blocks agent-side ssh submits; user runs
  `bash scripts/fm049_submit.sh` from the FLOWVPM.jl root). User picks the
  residency mode from the presented A/B — no auto-selection.

## Update — 2026-08-22 fourth session: 049 CLOSED (accepted)

- **049 is complete and accepted.** Four submissions: `13304874` (harness
  ternary parse error — `"PASS":"FAIL"` needs a space before `:`),
  `13305165` (harness `rec(...)` calls passed `P`/`gate`/`value`
  positionally — implicit keywords need a leading `;`), `13305443`
  (**FastMultipole bug**: Float64 literals `0.5`/`3.0` in `src/tree.jl`
  branch geometry promoted Float32 systems to `Branch{Float64}`, breaking
  the legacy CPU octree at Float32; fixed with `/2` and `3` forms —
  bit-identical at Float64; local F32+F64 CPU repro through `vpm.UJ_fmm`
  passed), then `13305555` **succeeded**: all five stages, 7m35s.
- Artifacts sha256-verified into
  `data/rotor_field_gpu_verification/results-13305555/` (results/budget
  CSVs, report, raw log, provenance, job .out).
- The job exited 1 by design on 49 harness-gate FAILs; all were root-caused
  to harness-side gate misapplication (four families: alloc contracts
  measured at wrapper instead of lifecycle layer; bitwise replay instead of
  048's error-bounded gates; operating-point SFS gates applied across the
  P4/P8 bracket; F32 integrity gate below the summation-noise floor). No
  anchor violated; no delivered-accuracy gate weakened. Full rationale: 049
  doc "Results (2026-08-22, H200 job 13305555)".
- Key numbers: resident RK3 **291 ms/step** at production P6/rho 4.789 F64
  (target 3.3 s; old P4 anchor 199 ms); ujsfs evaluation 94 ms; residency
  A/B upload-vs-resident **+12 ms/step (+4.1%)**, parity 150/150 at 1e-11;
  counters 144/144 (zero transfers); u_rel_rms 1.1e-4 at P4, 1.7e-6 at
  P8/4.789.
- **D15 (user):** upload-per-step residency ships — compatibility with
  monitors that trim/modify particles between steps. Both A/B arms
  integrate on GPU; upload adds one 46xN H2D+D2H per step. No code default
  changed (residency is the array-type trait, `FLOWVPM_fmm_radix.jl:51-52`).
- **Deferred to 053** (recorded in its Method item 1): lifecycle-layer
  allocation contract measurement and 048 error-bounded replay gates
  through the wrapper path.
- Docs updated this session: 049 doc (status + Results section), decisions
  log (D15), 053 (deferred contracts), START_HERE (048 and 049 rows now
  Done+Approved).
- **UNCOMMITTED working-tree changes** (preserve): FastMultipole
  `src/tree.jl` Float32 fix; FLOWVPM `scripts/fm049_rotor_verify.jl` fixes.
  Both are synced to the cluster trees (`~/FastMultipole-046`,
  `~/FLOWVPM-046`).
- **Next: 050.** It is Done (2026-08-21; verdict **B'** in
  `theory/panel-multisystem-scoping.md`: keep FLOWPanel's 3-pass structure,
  rectangular GPU brute-force cross passes, device-resident dense
  NearfieldInfluenceCache matvec for the panel self-solve, radix framework
  untouched) but **not approved** — its one pending input, the corrected
  049 particle budget, now exists (`results-13305555/fm049_budget.csv`).
  Next agent: validate the B' pricing against that budget, present 050 for
  user approval, then proceed to 051.

## Update — 2026-08-22 fifth session: 050 CLOSED (approved)

- B' pricing re-derived against `results-13305555/fm049_budget.csv`:
  particle side (D15) 0.306 s/step median / 0.373 worst = 9.3–11.3% of the
  3.3 s budget; np ≈ 210k confirms the 7.7e9-pair cross-pass counts;
  pessimistic B' stack (0.373 + 0.04 + 2.0) leaves 0.89 s solve headroom
  (~246 dense matvecs @ ~3.6 ms); stage table internally consistent
  (3×ujsfs_complete + rk3 residual = full_resident_rk3 exactly). Verdict
  and 051 shape unchanged. Reconciliation appended to
  `theory/panel-multisystem-scoping.md`; gate satisfied; **user approved
  050 on 2026-08-22**.
- User direction recorded: multi-system radix generalization (unified
  `fmm!`, heterogeneous sources/targets on GPU) is an eventual goal (this
  phase or phase Q); B' tentatively adopted as a step toward it.
- **Next: 051** (panel–particle GPU coupling) per START_HERE.md, subject to
  its own entry gate.

## Update — 2026-08-22 sixth session: 051 Stage 0 CLOSED (audit), Stages 1–2 staged

User directive this session: the handoff is strictly authoritative — the 051-shaped code found
on-branch (FastMultipole `1cd0b98a`, FLOWPanel `d6bf8b6` "300ms on gpu", untracked seams) was
treated as UNTRUSTED WIP and audited before reuse. Plan: `051-plan-2026-08-22.md`. Execution
scope this session: Stage 0 + prep of Stages 1–2 ONLY (solve/timing Stages 3–4 not started).

**Stage 0 — CLOSED.** Full verdicts + evidence in the 051 doc's "Worklog — 2026-08-22
(session 6)" section. Summary: all four components (host points functor, host panel functor,
blind-written CUDA kernels, FLOWPanel influence seam) audited SOUND/SOUND-WITH-NOTES and
adopted after hardening; host tests 45/45 in the FLOWPanel env (points EXACT vs FLOWVPM
production pair math; panels 1e-15 vs FLOWPanel `induced`, all tags/families). 2.6e-4 watch
item CLOSED (`baf8fb3` in FLOWVPM lineage; rect kernels never had the eps2 guard).

**Hardening edits (uncommitted, intentional, all listed in the 051 worklog):**
FastMultipole `src/direct_rectangular.jl` (+host-dispatch guard, F64-only panels, tag/nv
validation, docstring fixes), `src/translate_batched_cuda.jl` (unsafe_trunc), 
`test/direct_rectangular_test.jl` (+argument-validation and CUDA-device-parity testsets);
FLOWVPM `scripts/fm051_rect_bench.jl` (+per-target max gates, pass-2 J gate 1e-10),
`scripts/fm051_run.sh` (runs the device testsets before the bench); FLOWPanel
`src/FLOWPanel_gpu_influence.jl` (header comment). PRESERVE all of these.

**Red flag RESOLVED (D16, user decision 2026-08-22, seventh session):** the uncommitted
BRAINSTORM-025 change of the CPU-wide default filament regularization Vatistas → Gaussian
(FLOWPanel `src/FLOWPanel_elements_fmm.jl:923`) is RATIFIED as the production default; see
the decisions log. 051 parity was like-for-like regardless (the seam maps
`FILAMENT_REGULARIZATION[]` into the functor). Full uncommitted-FLOWPanel-diff triage table is in the
session-6 audit (seam+triage agent report): only FLOWPanel_fmm.jl seam hook + the two
includes in FLOWPanel.jl are 051-load-bearing; the rest is 025 / rigid-motion-persistent-plan
/ 052 work.

**Stage 1 (Job A) — READY, awaiting user submission:** from the FLOWVPM.jl repo root run
`bash scripts/fm051_submit.sh` (syncs both trees + p018 snapshot to orc, sets up fm048env,
sbatches `fm051_run.sh`: device tile-boundary testsets, then fm051_rect_bench both passes,
gates F64 1e-11 RMS + per-target ≤100×, pass-2 J 1e-10).

**Stage 2 — IN FLIGHT at reset time (validate before trusting):**
1. An agent was authoring `FLOWPanel.jl/benchmark/fm051_pass_parity.jl` +
   `benchmark/slurm/fm051_parity.sh` (pass-by-pass CPU-vs-seam parity; FM051_MODE=mini local
   with seam :host / full cluster warmstart with :cuda; snapshot/restore between arms;
   step-head gotcha asserted; 1e-3 phase gate, 1e-11 informational for pass 1; kerneloffset
   deviation makes 1e-3 the real pass-3 gate). It may or may not have finished/validated —
   check the files exist, review them, and re-run mini mode
   (`FM051_MODE=mini julia --project=. --threads=4 benchmark/fm051_pass_parity.jl`) before use.
2. FLOWPanel full CPU suite baseline (seam off, `--threads=4`) was running; result unknown —
   re-run if no log is found (051 no-regression gate needs a baseline).
3. FastMultipole full test suite not yet run this session (rect file additions are additive;
   still needs a green run before commit).

**Open items carried:** FLOWPanel checkout location on orc for the full-mode parity job
(handoff names only ~/FLOWVPM-046 + ~/FastMultipole-046) + where the p018_cs_f1_l3p4 restart
data lives on the cluster; lab-notebook entry for 049/050 closures (+051 progress) still
offered, not yet approved/drafted; branch-naming note (FLOWPanel is on `fastmultipole`;
`flowpanel-20260817` is FastMultipole's branch).

### Session-6 amendment (written at reset, all background work completed)

Supersedes the "IN FLIGHT" list above — everything landed green:

1. **Stage-2 parity harness DELIVERED + mini-validated**: `FLOWPanel.jl/benchmark/fm051_pass_parity.jl`
   + `benchmark/slurm/fm051_parity.sh` (both untracked). Mini mode (48-panel diamond
   RigidWakeBody{src+ring} + PanelParticleWake, seam :host vs CPU fmm!, 4 threads): ALL GATES
   PASS — pass 1 achieved 5.1e-15 (1e-3 gate, 1e-11 informational), pass 3 2.2e-14, solve
   0 seam hits, per-class worsts all ≤2.2e-14. CSV: `benchmark/results/fm051_pass_parity_mini.csv`.
   Design: per-pass snapshot/restore, CPU-arm outputs written back downstream, contribution
   (post−pre) comparison per output class, eligibility table printed per pass,
   `seam_accepted` gate marks parity VACUOUS on 0 hits, step-head NaN guard after the full
   step head, seam restored :off in try/finally. Full mode reuses p018_mature_wake_timing.jl's
   canonical_setup!/warmstart_restore! verbatim via a private module (kinematic replay can't
   drift). Caveats recorded in the harness author's report: mini's one-leaf FMM default means
   the pass-3 kerneloffset deviation is only priced by the FULL run; local Julia 1.12.5 vs
   cluster-pinned 1.11.7.
2. **FLOWPanel CPU baseline (seam off): 4814/4814 across 55 suites, exit 0** (~4 min test time).
3. **FastMultipole full suite: 894,127/894,127, exit 0** (~11 min); direct_rectangular_test.jl
   is in runtests.jl (line 62) — argument-validation + point/panel gates run under Pkg.test,
   FLOWPanel/CUDA layers skip cleanly off-env.
4. **NEW USER DIRECTIVE — FilamentWrapper seam extension is IN SCOPE for 051** (see the TODO
   section at the end of the 051 doc): in the full p018 config, pass 1 currently FALLS THROUGH
   to fmm! because the mature wake's non-empty `FilamentWrapper` (`include_final_filament=true`
   default + `wake.overflowed[]`) is an unsupported seam source. Implement: pack active
   final-filament segments as tag-3 filament columns (same family/kerneloffset plumbing as the
   TE-wake packing) in `_gpu_source_columns`/`_gpu_source_supported`; gate via the mini harness
   with `FM051_MINI_FINAL_FILAMENT=1` flipping to seam-accepted at the 1e-3 gate, then full-run
   p018 parity.

## Update — 2026-08-22 seventh session: FilamentWrapper seam extension DONE (local gates), Job B ready

- **FilamentWrapper seam extension implemented and locally green** (full detail + evidence
  table in the 051 doc "Worklog — 2026-08-22 (session 7)"): FastMultipole tag-3 panels now
  accept `nv == 2` open bound-vortex filament columns (`_rect_ring` one-liner; CUDA kernel
  shares `_rect_panel_pair` so no device-side edit); FLOWPanel seam packs
  `FilamentWrapper{<:PanelWake}` via new `pack_filaments!` (work buffer keyed on the wrapped
  wake — wrappers are rebuilt every `get_sources` call). Gates: rect tests 57/57 in the
  FLOWPanel env (open-filament parity U bitwise / H ≤3e-16, all families);
  `FM051_MINI_FINAL_FILAMENT=1` mini FLIPPED to seam_accepted=yes (6 filament columns packed,
  pass 1 2.8e-14); plain mini baseline unchanged. Artifacts + sha256:
  `data/panel_particle_gpu_coupling/sha256_local_2026-08-22.txt`.
- **Open item RESOLVED — FLOWPanel + restart data on orc**: standing checkout
  `~/projects/FLOWPanel.jl` (stale `5615ada`, dirty, no 051 files — READ-ONLY use) holds
  `data/p018_cs_f1_l3p4` (383M) and the `45_185_ct4` mesh in `examples/data/`. Job B runs
  from a synced `~/FLOWPanel-046` tree with symlinks to both, env `~/fm051env` (fm048env
  recipe + FLOWPanel dev + VSPGeom; fm048env untouched — it does NOT dev FLOWPanel). New
  local driver FLOWPanel `benchmark/slurm/fm051_parity_submit.sh`; `fm051_parity.sh`
  retargeted (`EXPECTED_REPO=$HOME/FLOWPanel-046`, env default `$HOME/fm051env`).
- **User-submitted commands**: Job A `bash scripts/fm051_submit.sh` (FLOWVPM.jl root) then
  Job B `bash benchmark/slurm/fm051_parity_submit.sh` (FLOWPanel.jl root).
- All session-7 changes are uncommitted and intentional — preserve alongside the session-6
  hardening list.

### Session-7 late update (written at reset): Jobs A/B run, root-cause done, rerun pending

Authoritative detail + all numbers: 051 doc "Worklog — 2026-08-22 (session 7)" (Job A
results table, Job B results + root-cause, harness-redesign rationale). Summary:

1. **Job A 13306457 COMPLETE, all Stage-1/2 gates PASS** on H200: device testsets 98/98
   (incl. new nv=2 filament columns), bench parity F64 1.3e-16/4.5e-15 with huge margin.
   TIMING FLAG: pass 1 0.124 s (band 0.02–0.04), pass 2 U-only 2.02 s (top of band), U+J
   3.64 s. Revised B′ stack 2.52 s → 0.78 s solve headroom on U-only.
   `BODY_HESSIAN_TO_PARTICLES=false` in p018 ⇒ production needs U-only (2.02 s applies).
2. **Job B 13306465 died** (`set -u` vs cluster profile.d; fixed). **Job B 13306475
   COMPLETE**: all 051 mechanics green (FilamentWrapper seam-accepted on the mature wake,
   solve 0 hits, SFS fallback fired correctly, pass 3 CPU 57.5 s → seam 2.9 s), but exit 1
   on 3 gate FAILs — ALL root-caused to harness metric design, NO seam defect: (a) Estr
   1.185 = dynamic-procedure clamp/clip flips under legitimate fmm-vs-exact pass-1 input
   difference (2.19e-4) — hit both the pass-1 phase gate and the SFS gate (double count);
   (b) pass-3 body_velocity 6.69 indistinguishable (from recorded data) between a
   1e-6-floor artifact on a near-null target and a real error. Attached-TE-wake-missing-
   from-CPU-farfield hypothesis checked and ELIMINATED (FLOWPanel_liftingbody.jl:734).
3. **Harness redesigned + mini-revalidated** (baselines bit-identical): per-class
   diff/scale metric + offender tables + divergent counts; full-mode phase gates
   diff_to_scale ≤ 1e-3; Estr excluded from pass-level phase (own gate: clip-flip
   sparsity < 0.1%); mini gating unchanged. Both metrics always print.
4. **Profile stage added** (user-directed): FM051_BENCH_PROFILE in
   `FLOWVPM.jl/scripts/fm051_rect_bench.jl` (pass-2 tag1/tag3 split, pass-1 occupancy
   ×2/×8 probe) rides the Job-B rerun as stage 0 of `fm051_parity.sh` (combined-job rule,
   now in FastMultipole CLAUDE.md "Cluster Jobs").
5. **D16 recorded** (decisions log): Gaussian filament regularization ratified as the
   production default (user, 2026-08-22); red-flag entries cleared.
6. **PENDING at reset: Job B rerun** — user runs
   `bash benchmark/slurm/fm051_parity_submit.sh` (FLOWPanel.jl root; re-syncs everything).
   Expected: pass 1 PASSES on diff_to_scale (~2e-4); pass 3 prints the REAL kerneloffset-
   deviation severity — if its diff/scale exceeds 1e-3, that is a genuine finding for the
   user, not a metric artifact; Estr gate = sparsity. Then Stages 3 (solve pricing:
   niter × matvec vs ~0.78 s headroom) and 4 (timing table vs 3.3 s budget, doc closure,
   commits — user approves commits). Lab-notebook entry (049/050 closures + 051) still
   offered, never drafted.
7. Evidence archive: `data/panel_particle_gpu_coupling/` + `sha256_local_2026-08-22.txt`
   (all local logs, job .out/.err, CSVs).

## Update — 2026-08-22 eighth session: pass-3 deviation attributed to the CUDA seam arm; fix pending

Authoritative detail + all numbers: 051 doc "Worklog — 2026-08-22 (session 8)" (three
subsections). Chronology and state:

1. **Job B rerun 13306588** (user-submitted, FAILED=exit 1 by design): redesigned
   metrics validated — pass 1 PASSES (diff/scale 3.1e-6), Estr sparsity PASSES
   (5.5e-5); pass 3 body_velocity FAILS at diff/scale 1.235e-2 (real, not metric).
   Stage-0 profile didn't run (driver omitted the snapshot arg; FIXED, copy now only
   on success). Artifacts + `sha256_13306588.txt`.
2. **Offsets are core radii** (user correction + explorer audit): `kerneloffset` was
   NEVER a surface offset in current code — p018 `core_size_panel = R*1e-10`
   (effectively singular self-solve; `_self_limit` handles on-surface),
   `core_size_targets = 1e-3` (physical filament core; VortexRing doublet sheet,
   Gaussian family per D16). User's four intended semantics CONFIRMED against code.
3. **Rename DONE** (user-directed, Opus agent, 105 files, all 3 repos):
   `kerneloffset*` → `core_size*` incl. ENV (`CORE_SIZE*`; old names honored), struct
   fields (SAFE: replay state is TOML string keys w/ old-key fallbacks —
   p018_cs_f1_l3p4 keeps loading), constructor kwarg aliases. Validated: FLOWPanel
   4814/4814 (via `julia test/runtests.jl`; NOTE `Pkg.test()` was ALREADY broken —
   missing LaTeXStrings in test/Project.toml, left unfixed deliberately), rect 57/57,
   mini parity bit-baseline. Uncommitted, intentional, PRESERVE.
4. **Pass-3 attribution stage BUILT** (user approved): `benchmark/fm051_pass3_attribution.jl`
   + `fm051_attribution_debug.jl` + harness wiring (`compare_pass!` keep=, exact
   `DirectBackend` arm on body-only targets, gate `pass3_attribution.seam_vs_exact`
   1e-10, `cpu_fmm_vs_exact` informational, `FM051_ATTRIBUTION=0` skips). Debug
   validated the machinery (truncation converges 8.9e-6→8.7e-9 for P=4→10; NOTE
   test_helpers' 4-chordwise-cell diamond gives a ONE-LEAF source tree = vacuous fmm;
   the debug builds a refined diamond).
5. **Job B 13309844** (user-submitted, exit 1): attribution VERDICT INVERTED the fmm-
   truncation hypothesis — **cpu-fmm vs exact 1.1e-10 (CPU is fine, NOT tuning);
   seam(:cuda) vs exact 1.235e-2 → THE CUDA SEAM ARM IS THE DEFECT.** Offenders
   mirror exactly across blades (+18376). Profile stage ran (pass-2 U split
   1.14/0.96/2.02 s src/ring/combined; pass-1 occupancy-bound: ×8 costs 3.96×).
   Artifacts + `sha256_13309844.txt`.
6. **Local elimination** (no GPU locally; scripts+logs archived in the data dir):
   p018 rotor body rebuilt from the LOCAL mesh with deterministic strengths
   (linearity ⇒ no restart needed; `repro_p018_body{,2}.jl`): host-seam vs exact
   4.6e-18 (noshedding) and 1.4e-15 (real TE shedding + attached-wake columns).
   Host functor math + packing EXONERATED. Suspect: CUDA path — device kernel
   near-singular branches under FMA (old device tests used only well-separated
   targets) or the seam's :cuda orchestration. `_cuda_rect_panels_kernel!` audit
   found no tile/stride bug.
7. **Instrumentation staged for the NEXT Job B run** (all local, validated):
   (a) harness attribution adds a host-seam arm when SEAM_MODE=:cuda
   (`host_seam_vs_exact`, `cuda_vs_host_seam` splits + offender control-point dump);
   (b) NEW FastMultipole rect device testset "on-surface p018 scale" (36,290 tag-4
   wavy-sheet panels, targets at centroids + deterministic 1e-8 nudges, device-vs-host
   ≤1e-11/1e-10) runs as stage 0c of `fm051_parity.sh` (non-fatal, logged as FINDING).
   Localization logic: stage 0c FAILS → pure-FastMultipole device kernel defect;
   stage 0c PASSES but cuda_vs_host_seam ≈ 1.2e-2 → seam :cuda orchestration
   (`FLOWPanel_gpu_influence.jl` upload/write-back).
8. **PENDING: user resubmits Job B** (`bash benchmark/slurm/fm051_parity_submit.sh`
   from FLOWPanel.jl root; re-syncs everything). Then: root-cause + fix the CUDA
   defect from the splits; re-run until seam(:cuda) matches exact; do NOT re-aim the
   pass-3 gate before that. After 051 mechanics close: Stage 3 (solve pricing vs
   ~0.78 s U-only headroom; revised B′ stack per Job A: pass-2 U-only 2.02 s applies)
   and Stage 4 (timing table vs 3.3 s budget, doc closure, commits — user approves).
   Lab-notebook entry (049/050 closures + 051) STILL offered, never drafted.
9. All session-8 changes are uncommitted and intentional — preserve alongside the
   session-6/7 lists. Key new/edited files: FLOWPanel
   `benchmark/fm051_pass3_attribution.jl`, `benchmark/fm051_attribution_debug.jl`,
   `benchmark/fm051_pass_parity.jl` (attribution stage), `benchmark/slurm/fm051_parity.sh`
   (profile-arg fix + stage 0c); FastMultipole `test/direct_rectangular_test.jl`
   (on-surface device testset); plus the 105-file rename in all three repos.

## Update — 2026-08-22 ninth session: CUDA defect root-caused and FIXED locally; Job B resubmission pending

Authoritative detail + all numbers: 051 doc "Worklog — 2026-08-22 (session 9)".

1. **Job B 13309929**: stage 0c FAILED (on-surface device testset, velocity rows
   relerr 7.05) ⇒ localized to the pure FastMultipole CUDA kernel (not seam
   orchestration). `host_seam_vs_exact` 2.155e-14; `cuda_vs_host_seam` 1.235e-2.
2. **Root cause**: solid-angle `atan(num, den)` branch is sign-fragile in
   roundoff-scale `tRz` for on-plane targets (den<0 on 2/3 edges ⇒ ±2π flip ⇒
   ∓σ/2·n̂); host/exact agree bit-for-bit, device NVPTX FMA contraction flips
   ~half. Probe-confirmed on all 36,290 test panels.
3. **Fix applied (uncommitted, intentional, PRESERVE)**: on-plane snap
   `tRz² ≤ 1e-24·L² ⇒ tRz = 0` + guard relaxed to `tRz == 0`, mirrored in
   FastMultipole `direct_rectangular.jl` and FLOWPanel `FLOWPanel_elements_fmm.jl`
   (`_onplane_snap`, Float32/Float64 only — snapping AD duals kills partials and
   broke 5 AD tests before being typed).
4. **Local gates green**: rect 31/31, repro_p018_body2 1.438e-15 (host/exact
   agreement preserved), FLOWPanel suite 4862/4862.
5. **PENDING: user resubmits Job B.** Expected stage 0c PASS and
   `cuda_vs_host_seam` ≤ 1e-10 and pass-3 gate PASS; any residual = second
   unmasked defect, iterate. Do NOT re-aim gates. Then Stages 3–4 per session 8
   item 8. Lab-notebook entry (049/050 closures + 051) STILL offered, never
   drafted.

### Ninth-session closure: Job B 13310123 — ALL GATES PASS, 051 parity mechanics CLOSED

Stage 0c 100/100; `cuda_vs_host_seam` 3.584e-14; `seam_vs_exact` 3.586e-14;
pass-3 phase gate 1.825e-5; OVERALL PASS (exit 0). The tRz on-plane snap fix is
confirmed on device. No gates were re-aimed. Next: Stage 3 (solve pricing) and
Stage 4 (timing table vs 3.3 s budget, doc closure, commits — user approves),
per session-8 item 8. All fixes remain uncommitted and intentional.

## Update — 2026-08-22 ninth session part 2: Stage 3 measured and decomposed; Stage-3 fix design chosen, implementation pending

Authoritative detail + numbers: 051 doc "Worklog — 2026-08-22 (session 9)" tail
sections. State:

1. **CUDA seam defect CLOSED** (Job B 13310123, all gates green; tRz on-plane
   snap in both repos, uncommitted).
2. **Job C (CPU solve pricing) built and run twice** (`benchmark/
   fm051_solve_pricing.jl` + `slurm/fm051_solve_pricing.sh`, CPU-only, 64 cpus,
   mem 64G): warm `solve_formulation!` ≈ **7.3 s** vs **0.783 s** U-only
   headroom (3.3 − [0.373 + 0.124 + 2.02]) ⇒ does not fit as-is.
3. **Decomposed (13310370)**: 7.3 s = the Dirichlet (DBC=true) per-solve
   `influence!(body, body; scalar_potential=true)` — 36,752² potential
   self-influence via CPU fmm (profile: `_direct_body!`/`induced`). The
   `Backslash` backsolve is ms. NOT a Krylov-matvec problem.
4. **Chosen direction (not yet implemented, not yet user-ratified as final):**
   assemble the constant rotating-frame potential-influence matrix Φ once at
   setup and replace the per-solve fmm influence with a dense gemv
   (~50–100 ms CPU, 10.8 GB; device only if that disappoints); transform under
   rigid motion alongside G. Alternative (bigger): add potential output to the
   rect seam.
5. All session-9 files uncommitted and intentional — preserve. New:
   FLOWPanel `benchmark/fm051_solve_pricing.jl`, `benchmark/slurm/
   fm051_solve_pricing.sh`; edits: FastMultipole `src/direct_rectangular.jl`
   (snap), FLOWPanel `src/FLOWPanel_elements_fmm.jl` (snap + `_onplane_snap`).
6. Stage 4 (timing table vs 3.3 s, doc closure, commits — user approves) waits
   on the Stage-3 fix. Lab-notebook entry (049/050 closures + 051) STILL
   offered, never drafted.

## Update — 2026-08-23 tenth session: Stage-3 fix implemented (opt-in S gemv), local gates green, Job C rerun pending

Authoritative detail: 051 doc "Worklog — 2026-08-23 (session 10)". Summary:

1. **User ratified the Φ direction 2026-08-23** with constraints: opt-in
   (no default Backslash memory change) and keep the architecture open for
   a future matrix-free/GPU source-influence path.
2. **Implemented in FLOWPanel `src/FLOWPanel_solver.jl`** (uncommitted,
   intentional): `_G!` kwarg `kernel_and_strength_index` (default path
   bit-unchanged); `Backslash.S` field + `assemble_source_potential!`
   post-hoc attach + constructor kwarg; NEW `_source_influence!` seam in
   the Dirichlet `solve!` (default = old fmm influence!; Backslash-with-S
   = dense gemv, falls back when wake correction active) — future GPU/
   matrix-free backends plug in at this seam; `update_G=true` refreshes S
   with G. S needs no rigid-motion transform (scalar invariance, same as G).
3. **Local gates green**: unit solver 387/387 (new S testset 7/7), full
   FLOWPanel suite 4947/4947 exit 0, mini parity PASS (pass 1 6.8e-15,
   pass 3 1.44e-14; tree moved via 62d72db since the older baseline).
4. **`benchmark/fm051_solve_pricing.jl` extended** (S stage default-on,
   `FM051_PRICE_S=0` skips): assembly time, gemv-vs-fmm equivalence on the
   production state, bare gemv time, S-path warm medians; VERDICT now uses
   the S-path median vs 0.783 s. Memory: G+S ≈ 21.6 GB, 64G sbatch fits.
5. **PENDING: user resubmits Job C** (re-sync benchmark/ to ~/FLOWPanel-046,
   `sbatch benchmark/slurm/fm051_solve_pricing.sh`). Expected: equivalence
   ~1e-10 rel, S-path warm solve ≪ 0.783 s ⇒ Stage 3 closes; then Stage 4
   (timing table vs 3.3 s, doc closure, commits — user approves).
6. Lab-notebook entry (049/050 closures + 051) STILL offered, never drafted.

## Update — 2026-08-24 tenth session part 2: Job C 13391706 — S path FITS (0.758 s vs 0.783 s headroom)

Authoritative detail: 051 doc session-10 sections. Summary: Job C rerun
13390264 FAILED on a stale cluster tree (only benchmark/ had been synced;
src/ lacked the S code — resynced, incident archived). Rerun **13391706
COMPLETED**: S assembly one-time 118.9 s / 10.81 GB, bare gemv 0.453 s,
**S-path warm median 0.758 s vs 0.783 s headroom ⇒ FITS** (margin 25 ms —
thin; levers if needed: BLAS/NUMA tuning of the 0.453 s gemv (~24 GB/s
effective, node can do much more), Float32 S, device gemv). Equivalence
gemv-vs-fmm was EXACTLY 0 — root-caused as genuine, not vacuous: local
probe showed the body self-influence via FastMultipoleBackend(8,0.4,20) is
bitwise identical to DirectBackend for potential AND velocity (no farfield
accepted for the self pair) ⇒ the replaced 7.3 s call was a true dense
O(N²) evaluation and S·σ is a lossless drop-in. Pricing script now prints
σ/φ norms + VACUOUS flag (synced to orc). Artifacts + sha256 for both jobs
in `data/panel_particle_gpu_coupling/`. **Next: user sign-off on the thin
margin, then Stage 4** (timing table vs 3.3 s, doc closure, commits — user
approves). Lab-notebook entry (049/050 closures + 051) STILL offered,
never drafted.

## Update — 2026-08-24 tenth session part 3 (context reset): Job C diagnostics run 13395348 IN FLIGHT

Authoritative detail: 051 doc "Worklog" session-10 sections. State at reset:

1. **Stage-3 implementation is DONE and validated** (see tenth-session
   updates above): opt-in S matrix on `Backslash` + `_source_influence!`
   seam in FLOWPanel `src/FLOWPanel_solver.jl`; local gates green (unit
   387/387, suite 4947/4947, mini parity PASS); Job C 13391706 measured
   S-path warm median **0.758 s vs 0.783 s headroom ⇒ FITS** (margin 25 ms).
   Equivalence exact-0 root-caused GENUINE: the body self-influence via
   FastMultipoleBackend(8,0.4,20) is all-direct (bitwise == DirectBackend
   for phi AND u; probe in scratchpad + 051 doc) ⇒ S·σ is a lossless
   drop-in for a true dense O(N²) evaluation.
2. **Job C rerun 13395348 IN FLIGHT** (user-submitted 2026-08-24): carries
   the new GEMV DIAGNOSTICS block (BLAS-thread sweep {1,8,16,32,64} with
   GB/s, row-blocked Julia-threads gemv, tuned warm-solve re-timing;
   verdict = min(plain, tuned) median) and `--qos=test` DROPPED (13391706's
   0.453 s gemv ≈ 24 GB/s ≈ single-core bandwidth; suspects: OpenBLAS
   dgemv under-threading, NUMA first-touch, test-QoS node quality).
   Output: `~/FLOWPanel-046/logs/slurm/slurm-fp-051-price-13395348.{out,err}`
   + `benchmark/results/fm051_solve_pricing.csv` on orc.
3. **When it finishes**: fetch job .out/.err + CSV (as
   `fm051_solve_pricing_13395348.csv`) with sha256 into
   `data/panel_particle_gpu_coupling/` (pattern: sha256_13391706.txt);
   read the GEMV DIAGNOSTICS + tuned medians; expected: bare gemv drops
   toward tens of ms if BLAS threading/QoS was the limiter (CPU floor is
   bandwidth-bound; true ms needs the 052+ device gemv). Update the 051
   doc with the sweep verdict. Then **Stage 4**: timing table vs the 3.3 s
   budget (B′ stack 0.373 + 0.124 + 2.02 + best solve median), 051 doc
   closure, and the commit set — user approves all commits.
4. Standing: ALL changes in FastMultipole, FLOWPanel.jl, FLOWVPM.jl are
   uncommitted and intentional — preserve (sessions 6–10 lists). Cluster
   trees ~/FLOWPanel-046 (+ ~/FastMultipole-046, ~/FLOWVPM-046) are synced
   incl. the session-10 files. `Pkg.test()` in FLOWPanel is still broken
   (missing LaTeXStrings in test/Project.toml, deliberately left); use
   `julia --project=. test/runtests.jl`. Lab-notebook entry (049/050
   closures + 051) STILL offered, never drafted.

## Update — 2026-08-24 eleventh session: Job C 13395348 COMPLETED — Stage 3 CLOSED (tuned 0.607 s, 23% margin); Stage 4 table CLOSES at 3.124 s vs 3.3 s

Authoritative detail: 051 doc session-11 + Stage-4 sections. Summary:

1. Job 13395348 (no qos=test, m12-4-18, 7m08s, exit 0): fmm-path warm
   median 7.300 s; S-path warm median 0.659 s; **tuned (BLAS=8) median
   0.607 s vs 0.783 s headroom ⇒ FITS, margin 176 ms (23%)**. dgemv sweep:
   BLAS 1/8/16/32/64 = 0.419/0.234/0.238/0.252/0.286 s (best 46 GB/s at
   8); row-blocked 0.282 s. 13391706's 0.453 s gemv was
   qos-node + BLAS-oversubscription, not code. Artifacts + sha256_13395348
   archived in `data/panel_particle_gpu_coupling/`.
2. **Vacuity correction**: the new |σ| norm print fired `[VACUOUS:
   sigma == 0]` — the on-cluster exact-0 equivalence checks (both runs)
   compared zero vectors. The lossless-drop-in claim rests on the LOCAL
   probe (nonzero σ, exact p018 backend, bitwise all-direct == 
   DirectBackend), which stands. 051 doc session-10 "NOT vacuous" sentence
   superseded.
3. **Stage 4 CLOSED**: B′ stack 0.373 + 0.124 + 2.020 + 0.607 =
   **3.124 s vs 3.3 s budget (5.3% margin)**. One-time S assembly 111.8 s
   / 10.81 GB. Levers if margin erodes: Float32 S, device gemv (052+).
4. Remaining: commit set across FastMultipole / FLOWPanel.jl / FLOWVPM.jl
   (user pre-approved this session); lab-notebook entry still offered.
