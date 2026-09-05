# 052 phase handoff (session 2026-09-01c): mandate = finish 052b/052e/052c, then 053. 052b state: IGE device fix VALIDATED both arches; 1r blocked on gate policy, 2r blocked on block-GS convergence

## Prompt for the next agent

You are continuing the MATRIX_OPERATOR_REFACTOR 052 phase after a context
reset. Read this file first; it is authoritative. Predecessor:
`052b-handoff-prompt-2026-09-01b.md` (background only — its "NEXT" items
are all superseded/resolved here). HPC.md (`FLOWPanel.jl/agent_policies/
HPC.md`, local copy authoritative) is REQUIRED READING before cluster work.

## Mandate (Ryan, 2026-09-02 — verbatim intent)

- **Finish ALL the 052 subitems — 052b, 052e, 052c — and then 053.**
  Task docs: `052b-impl-multirotor-ige-gpu.md`,
  `052e-impl-hybrid-wake-potential-experimental.md`,
  `052c-plan-2026-08-26.md` (owns the 1080-step stage-d acceptance +
  052/052a consolidation + notebook entry),
  `053-milestone-review-production-integration.md` (signed review of
  046–052; entry gate: 046–052 complete and approved).
- **Use subagents to keep your context slim** — this will be a
  long-running session. Delegate exploration, doc digestion
  (refactor-docs-librarian), test/sim runs (julia-test-runner), and bulk
  log analysis. Keep only conclusions in the main context.
- **Use subagents to perform the clear-context review of each item** —
  when an item completes, a fresh-context review subagent audits it
  against its contract before it is called done.
- **Pipelining is authorized**: you may work on portions of later items
  before predecessors finish, so long as the prerequisite groundwork is
  laid. E.g. work 052e while 052b simulations run; begin 053 scaffolding
  and non-dependent audit rows early — but the portion of 053 that
  reviews 052c cannot run until 052c is finished and approved.
- **Ryan is away for a while.** Work autonomously, make the calls you
  must, and keep a running DECISION LOG (see §Decision log) of every
  major decision made without him, to report when he returns. Policy
  decisions already flagged to him (§Flagged to Ryan) should be decided
  conservatively if they block the critical path, and logged.

## 052b immediate technical state (was Ryan's instruction list 2026-09-01)

1. **Root-cause the 2r block-GS convergence failure** (§2r failure). First
   step: wire up per-iteration residual logging (§GS_VERBOSE is not
   plumbed), re-run the 2r diagnostic smoke, and read the trajectory —
   floor vs slow contraction decides everything downstream. Do NOT change
   solver policy before the trajectory is in hand.
2. **1r accept was blocked on a gate-policy decision** (§1r gate abort).
   With Ryan away, the conservative unblock he was leaning toward being
   offered is option (b): ONE long-wall accept on mgh (24 h MaxTime,
   idle nodes) to measure the true step-time trajectory — it retires the
   growth-uncertainty question with data and produces the accept output
   if it fits. If you take it, log the decision.
3. Longer arc unchanged: all matrix-free solvers on GPU; panel↔panel
   dense influence is the dominant cost (§Where 1r time goes).

## Decision log (append here; report to Ryan on return)

- 2026-09-03 01:55 MDT — Committed `6a64402` (orc unified-052): default-guard
  carrier GS env (line 73 clobber fix) + plumb GS_VERBOSE env into tuple
  `solve!` via `solve_formulation!` (per-iteration residual, default off).
  Behavior-neutral when GS_VERBOSE unset (verified: bash -n, Meta.parseall,
  diff vs HEAD). Local mirror updated, uncommitted per policy.
- 2026-09-03 01:55 MDT — NOTE: orc repos were consolidated by another lane:
  `~/projects_unified/*` → `~/projects/*` (FLOWPanel HEAD had moved to
  e2768ef, repoint commits f8507b1/886fc9d). All handoff paths mentioning
  projects_unified are stale; use `~/projects/`.
- 2026-09-03 01:56 MDT — Took mandate option (b) for the 1r gate block:
  submitted long-wall accept 13568974 on mgh GH200 (P022G_MODE=accept,
  24:00:00 wall, no chain gate). Rationale: conservative unblock Ryan was
  leaning toward; measures true step-time trajectory and produces accept
  output if it fits.
- 2026-09-03 01:56 MDT — Submitted 2r diagnostic smoke 13568975 on mgh GH200
  with GS_VERBOSE=true GS_MAX_OUTER=120 (04:00:00 wall) to capture the
  per-iteration block-GS residual trajectory (floor vs slow contraction).
  No solver policy changed pending the trajectory, per handoff instruction 1.
- 2026-09-03 ~05:30 MDT — 052c: diagnosed 13501691 (fp052c-trial1, FAILED
  2026-08-27, never recorded). Run 1 (756-step) PASSED all gates (CT 6.9e-5,
  Gamma M2 max 1.4e-4, RMS 5.4e-5 — all under ceilings). Run 2 crashed at
  step 894/1079: Nothing-CuArray typeassert in
  `_launch_cuda_hierarchical_m2l_cached!` (translate_batched_cuda.jl:8052,
  silo FastMultipole-052-h200) — NOT the known step-~1071 core collapse.
  Local commits d938ba68 (052f) + 2c6dd60f (052g) patch exactly this path
  (early-return on zero-M2L cache, win_class === nothing) and post-date the
  job. DECISION: porting the two fix commits to orc unified-052 FastMultipole
  via bundle + cherry-pick (orc history lacks local SHAs; silos aren't git
  repos), then rerunning trial-1 acceptance on the unified tree.
- 2026-09-03 ~05:50 MDT — Port turned out UNNECESSARY: the 2026-09-02
  consolidation already carried both 052f/052g fixes into orc unified
  FastMultipole (verified verbatim at translate_batched_cuda.jl ~:6874 and
  ~:8074) AND into the silo ~/FastMultipole-052-h200 the trial-1 env uses.
  Cleaned up bundle/patches/temp branch. Trial-1 rerun is therefore safe
  as-is via ~/projects/launchers/fp052c_trial1_run.sh (identical sbatch
  line as 13501691, job name fp052c-trial1b) — submission BLOCKED by the
  local permission classifier; handed to Ryan to run.
- 2026-09-03 05:33 MDT — Ryan submitted trial-1b himself: job **13569052**
  (eng, H200, 6h wall), PD behind the p018 fleet at submit time.
- 2026-09-03 ~05:45 MDT — 053 row 2 groundwork: defaults-enumeration draft
  written to 053-defaults-enumeration-draft-2026-09-03.md (9 flagged rows,
  most marked NEEDS RYAN; caveat: unified branches squash 046–052 history,
  so the audit is keyword-grep-based, not full-diff).
- 2026-09-03 ~06:20 MDT — 052c trial-1 MIGRATED eng→mgh per Ryan's ruling
  (eng full for a while): scancelled 13569052 (never started) and submitted
  **13569059** `fp052c-trial1c` on mgh (GH200/ARM, `--qos=gpu
  --gres=gpu:gh200:1 -C arm -c72 --mem=192G -t 6:00:00`), launcher
  `~/projects/launchers/fp052c_trial1_gh200_run.sh` (new). Changes vs x86
  launcher: -gh200 silo/env paths, aarch64 tarball julia
  (`~/julia/julia-1.11.7/bin/julia`) + `fm052depot-gh200` depot + PATH
  prepend and `FP052_JULIA_BIN` (fm052_gate.sh/provenance.sh call bare
  `julia`), and CORRECTED reference paths: the x86 launcher's
  `~/FLOWPanel-052/data/{fm052_campaign_lock,fm052r_cpu_mature_pinned}` are
  STALE (data now lives under `~/projects/FLOWPanel.jl/data/`) — 13569052
  would have died at preflight `test -s` regardless of queue. Preflight
  verified: gh200 silo src trees byte-identical to -h200 (sigma_guard
  present), fm052env-gh200 Manifest dev-paths point at -gh200 silos,
  checkpoint root `~/projects/FLOWPanel.jl/data/p018_L1_ov3` exists. Job
  queues behind gsdiag 13568975 on mgh-1-2. Log:
  `~/FLOWPanel-052-gh200/data/fp052c-trial1c-13569059.out`. Also patched
  the stale paths in the x86 launcher `fp052c_trial1_run.sh` in place so a
  future eng resubmit works.
- 2026-09-03 ~06:35 MDT — 052e groundwork (mandate move 4): wrote
  `052e-accuracy-plan-draft-2026-09-03.md` — proposed pre-registered
  tolerance table (T1–T6 NEEDS RYAN; P1–P3 verbatim from the task doc) and
  a 9-pair + 1-long-run comparison matrix (OFAT around a starred production
  baseline; rows 8–9 gated on 052b closure). No runs launched. Also kicked
  off 053 row 3 (three-repo local test suites, 4 threads, background,
  logs in scratchpad) and extracted first 1r accept step-time buckets:
  ~13.7–14.8 s/step flat through step 124/413 — wall margin is huge
  (extrapolates to <2 h vs 24 h).
- 2026-09-03 ~07:05 MDT — **2r gsdiag readout: classification (a) FLOOR,
  decisively.** 13568975 ran all 120 outer iterations: strength delta
  contracts geometrically (28.7 → 1.4e-3 → 2.3e-6 → 3.1e-9 → 4.3e-12 →
  ~2e-15 by iter 6 — contraction ~6e-4/iter, i.e. block-GS converges to
  machine epsilon in ~6 iterations) while the normalized block residual
  is PINNED at 5.8224e-4 from iter 2 through iter 120 (varies only in the
  16th digit). The solver is NOT slow; the residual metric is floored by
  cross-influence evaluation accuracy (FMM), so GS_TOL=1e-8 is unattainable
  at current FMM settings and `require_outer_convergence` will always
  throw for 2r. Raising the outer cap / relaxation / single-block would
  all be useless — ruled OUT. Remedy per handoff: confirm the plateau
  moves with FMM accuracy. Submitted **13569088** `fp-022g-2r-gsdiag3`
  (mgh, 4h wall, GS_MAX_OUTER=30) with FMM_BODY_EXPANSION_ORDER=20 (was
  17), FMM_BODY_ACCEPTANCE=0.5 (was 0.7); queues behind the 1r accept.
  Enabler: carrier FMM knob exports (lines 43–44) default-guarded,
  behavior-neutral — orc unified-052 FLOWPanel commit **4e6b5b7** (bash -n
  clean; only my file committed — another lane has uncommitted
  run_p018_screen_hpc.slurm.sh edits left untouched); local mirror
  edited, uncommitted per policy. SOLVER POLICY (needs Ryan): if gsdiag3
  confirms the floor scales with FMM accuracy, options are (i) set GS_TOL
  above the floor (~1e-3) and gate on strength-delta contraction instead,
  (ii) pay for higher FMM accuracy in production 2r, or (iii) redefine
  the convergence check to use strength delta. No policy changed yet.
- 2026-09-03 ~07:20 MDT — 053 row 3 first pass (local suites, 4 threads):
  **FastMultipole GREEN** (exit 0, all testsets pass, ~11 min).
  **FLOWPanel errored**: `using Logging` in test/runtests_unit_simulate.jl:2
  but Logging missing from test/Project.toml → added the stdlib dep
  (local edit, uncommitted). **FLOWVPM 1 failure**: runtests_gpu_fmm.jl
  "sigma-outgrown rebuild (052c near-peak)" final sub-case expected
  `ArgumentError` from the runtime adequacy gate on user-fixed ell — STALE
  vs the 052f contract (d938ba68 demotes to all-direct zero-M2L with a
  warning instead of throwing; the observed warning came from
  translate_batched_resident.jl:2123). Updated the sub-case to
  `@test_logs (:warn, r"all-direct zero-M2L")` and refreshed its comment
  (local edit, uncommitted). Both suites rerunning via julia-test-runner.
- 2026-09-03 ~07:45 MDT — 053 row 3 rerun results: **FLOWVPM now GREEN**
  (updated sub-case passes; all testsets pass). **FLOWPanel: 656 pass,
  2 fail** — `explicit jump fallback (:jump)` runtests_unit_kutta.jl:539-540:
  the failing-provider+:jump path is no longer bitwise-identical to the
  legacy A/jump trajectory (strengths differ in all elements, not
  roundoff). NOT caused by this session (no local src mods; testset never
  ran in the first pass — it sits after the Logging abort point, so the
  breakage was masked). Likely culprit: commit 7fbd68a ("cross-pass A-F +
  LineGauss: solver/wake/warmstart/kutta FMM-side changes") post-dates the
  test. OPEN — needs either a code fix or a test-contract update; flagged
  to Ryan, not pursued (off 052 critical path). Also: handoff's "local
  mirror uncommitted" note re 6a64402 is stale — GS_VERBOSE plumbing is
  present and committed in the local FLOWPanel mirror.
- 2026-09-03 ~08:15 MDT — ALL THREE cluster jobs ended (monitor missed the
  transitions; harvested at reset-prep). (1) **1r accept 13568974: ran ALL
  413/413 steps (2:12:24)** — the run itself is complete and healthy
  (step times 14→19.5 s), but job FAILED on two GATE-level checks: elapsed
  7917 s vs a hardcoded 7200 s gate (10% over), and Phase-2e
  CONVERGED=false with per-rev spread 0.523 vs tol 0.005 — readout window
  starts at rev 1.0 while hover begins at rev 6.5, so the "spread"
  includes the spin-up/pulse transient; likely a readout-window
  misconfiguration, not physics. Per-rev CSV + metadata WERE written
  (data/p022g_1r_ige/) — convergence can be recomputed offline over the
  hover window without a rerun. (2) **trial-1c 13569059 FAILED at 4:36
  (warmstart propagate!)**: `ArgumentError: unknown sigma_guard key ceil;
  recognized: (:dtz_cap, :floor)` — the silo FLOWVPMs (h200 AND gh200,
  byte-identical) carry the older 2-key `_sigma_guard_params` while the
  silo FLOWPanel example passes a 3-key guard (ceil support exists in
  local FLOWVPM src/FLOWVPM_timeintegration.jl:36-42). Silo pair is
  internally SKEWED — trial-1b on eng would have crashed identically; not
  an ARM issue. Fix: port ceil support into both silo FLOWVPMs, resubmit.
  (3) **gsdiag3 13569088 FAILED at 21 s**: `data/p022g_2r_ige exists; set
  P022G_EXISTING_RESULT=preserve` — trivial resubmit with that env var
  (FMM override plumbing WORKED: banner shows body_fmm=20/0.5/109).
  Both mgh GH200 nodes are now IDLE.
- 2026-09-04 (session 2026-09-04a) — Verified cluster state directly at turn
  start (squeue/sacct/sinfo): no 052-lane jobs anywhere, both mgh nodes
  idle, fp-018gpu fleet untouched. Resubmitted the 2r floor-confirmation
  as **gsdiag4 = 13582074** (exact handoff sbatch line: body FMM 20/0.5,
  GS_MAX_OUTER=30, P022G_EXISTING_RESULT=preserve).
- 2026-09-04 — **sigma_guard ceil PORT to both orc silo FLOWVPMs**
  (~/FLOWVPM-052-{h200,gh200}/src/FLOWVPM_timeintegration.jl, both were
  byte-identical 2-key versions). Did NOT copy the whole local file: local
  has diverged further (pfield.splitting_state / dsigma2_rvpm accumulator
  lines that need SplittingState machinery the silos lack — whole-file copy
  would break them). Ported the minimal ceil hunk instead: docstring bullet,
  3-key check, 3-value return, and the clamp APPLICATION at both call sites
  (scalar euler `clamp(new_sig, sfloor, sceil)` and broadcast
  `clamp.(sig_new, sfloor, sceil)`) — parser-only port would have accepted
  :ceil but silently not applied it. Pre-patch backups saved as
  `*.bak-preceil`; both patched copies md5-identical
  (ad185c2fa7947c4e138edef82196def6); Meta.parseall PARSE_OK. Resubmitted
  **trial-1d = 13582076** via the fixed gh200 launcher.
- 2026-09-04 — **1r accept offline harvest (no rerun): hover-window
  Phase-2e recompute FAILS — the original CONVERGED=false was NOT just a
  windowing artifact.** From p022g_1r_ige_CT_per_rev.csv (11 complete revs,
  NT=36): hover begins rev 6.5 (metadata: ramp 1.0 + hold 1.5 + withdraw
  4.0); hover-window blocks (rev_start >= 6.5) are revs 7-10.97 with CT
  means 0.0897, 0.0920, 0.0888, 0.0801. Window mean CT = 0.0877;
  mean_spread_rel = 0.086 (tol 0.005, FAIL); ptp_rel = 0.47 (tol 0.02,
  FAIL). CT is still drifting (rises to rev 9, falls through rev 11 —
  possible slow transient or start of a limit cycle; only 4.5 hover revs,
  cannot distinguish). Including boundary block 7 (rev 6.0-6.97, CT 0.0872)
  does not change the verdict. Root cause of the ORIGINAL false: run used
  CONVERGENCE_REVS=10 on an 11-rev run, so the window started at rev 1.0
  and the driver's window_in_hover guard forced converged=false regardless
  of spread — config interplay, not a carrier bug; the 7200 s elapsed gate
  IS hardcoded (run_rotor_multi_ground_effect_gpu.slurm.sh:203). NOT
  claiming a 1r accept-in-substance; needs Ryan (more revs vs cycle-mean
  policy vs tolerance change).
- 2026-09-04 — **Kutta :jump regression PINNED to commit 7fbd68a exactly**
  (053 row 3 open item). Worktree bisect (FLOWPanel worktree in session
  scratchpad `fp-kutta-pre7fbd68a`, FastMultipole/FLOWVPM dev-pathed to the
  current local trees on BOTH runs so the FMM side is a constant control):
  parent 8b07f96 = 658/658 GREEN incl. 6/6 "explicit jump fallback
  (:jump)"; at 7fbd68a = 656/658 with the SAME two failures as HEAD
  (runtests_unit_kutta.jl:539-540, body + wake strength equality). New
  quantitative detail: the fallback's strengths come out ~HALF the legacy
  values (body [-0.111, ...] vs expected [-0.571, ...]; wake -0.313 vs
  -0.645) — consistent with the earlier wake-row ~2x observation, and
  suggests a factor-of-2 (or double-counted/halved influence) in the
  Dirichlet self-potential/wake-row convention change, not noise. Method
  note: the old worktree's committed Manifest pins registered FastMultipole
  v2.0.4 (predates numtype) — dev-pathing the local trees is required to
  run it; no sources were modified. NOT fixing: whether `_kutta_trial!`
  updates to the new convention or the bitwise contract relaxes to
  tolerance is Ryan's canonicality ruling (flag carried).
- 2026-09-04 — **gsdiag4 (13582074) readout: the FMM-accuracy floor test was
  structurally INERT — and the "FMM cross-influence accuracy floors the
  residual" interpretation is now REFUTED for these GPU runs.** Residual
  trajectory at body FMM (20, 0.5) is BIT-IDENTICAL to gsdiag2's at
  (17, 0.7) from outer iteration 1 (plateau 5.8223981764176e-4 to 16
  digits; strength delta at machine eps by iter ~6; log diff shows only
  timings/banner/GS_MAX_OUTER differ). Banner override was real
  (body_fmm=20/0.5/109) but code trace shows why it cannot matter:
  `influence!(..., ::FastMultipoleBackend)` (FLOWPanel_fmm.jl:60-79) is
  intercepted by `_gpu_rect_influence!` (FLOWPanel_gpu_influence.jl:
  618-650) BEFORE backend.expansion_order/multipole_acceptance are read —
  the carrier exports FLOWPANEL_GPU_INFLUENCE=cuda (carrier line 32), every
  GS cross-influence/residual call passes a production_route, so all of it
  runs through the EXACT dense Float64 CUDA batch (`_gpu_direct_batch!`),
  never FastMultipole.fmm!. Consequences: (1) the 5.8224e-4 plateau is NOT
  body-FMM truncation error — cross-influence is exact dense in both runs;
  (2) "pay for FMM accuracy" is NOT a remedy lever on the GPU route —
  strike it from the 2r solver-policy options; (3) with strength deltas at
  machine eps, the GS fixed point is exact, so the pinned residual
  measures a ~5.8e-4 inconsistency between the residual's assembled
  operator and the operator implied by the block solves (candidates:
  wake-route contributions (wake_fmm=16/0.6 identical in both runs),
  Kutta/damping rows, residual-assembly formulation) — root cause NOT yet
  identified; new investigation, needs Ryan's priority call. Remaining
  policy options: raise GS_TOL above the residual floor, gate on strength
  delta instead of residual, or fund the residual-operator-mismatch
  investigation.
- 2026-09-04 — **trial-1d (13582076) died of NODE_FAIL at 2:02:55** —
  mgh-1-2 dropped to `maint` mid-run; NOT a code failure. The ceil port
  held: zero sigma_guard errors, run progressed to step 693/1079 of the
  long run (~10.2 s/step; stage-1/756-step checkpoint-skipped as designed;
  historical zero-M2L crash point was step 894, not yet reached).
  Resubmitted same launcher as **trial-1e = 13582234** (lands on idle
  mgh-1-1). gsdiag4 (13582074) ended FAILED at 40:58 = the expected
  die-at-step-0 smoke pattern; its diagnostic payload (30 GS outer
  iterations) was already harvested before the job ended. Monitor re-armed
  on 13582234.
- 2026-09-04 — **trial-1e (13582234) COMPLETED, exit 0, 2:58:15 — 052c
  trial 1 PASSES.** "artifact and monitor gates passed for indices
  0:1079"; locked correctness gates all PASS with ~2x margins (CT
  cycle-mean 6.565e-4 vs ceiling 1.800e-3; Gamma M2 max 1.317e-3 vs
  2.934e-3; RMS 4.484e-4 vs 1.498e-3; window 720-755). min_sigma
  contracted monotonically 4.451e-3 -> 9.558e-5 m (min at step 983,
  ratio 0.0215; floor 0.01 never clamped), recovered to 1.303e-4 by step
  1080; no collapse recurrence. BONUS validation: the 052f demotion
  fired mid-run (radix geometry inadequate at ell=3, fell back to
  all-direct zero-M2L at ell=2) and the run survived — the exact
  scenario that crashed 13501691 run 2 at step 894. Phase-2e
  CONVERGED=false (known non-fatal readout item; cycle-mean CT
  0.072526 +/- 1.04%). Ledger updated
  (052c-sigma-experiments-2026-08-26.md §Results) with full tables +
  commit-plan proposal (NEEDS RYAN: upstream the ceil port; adopt
  dtz_cap=0.5 + floor 1% as GPU-acceptance default; fold remaining OFAT
  candidates into 053). Storage flag intensified: this run wrote full
  VTK under the shared data root while /home is already ~618G vs the
  400G cap — archiver pass needed (Ryan-gated).


## Session results 2026-09-01c (verified, don't redo)

### IGE device blocker FIXED and validated on both arches

The two scalar-indexing sites (damp-band `propagate!` override, ground
diagnostics monitor) were rewritten as masked broadcasts on device views —
commit `f5f4f29` (orc `unified-052`, FLOWPanel). CPU equivalence test:
bit-identical counts/values vs the old loops. A driver-wide grep found no
other `particles[` scalar loops. Result: smoke PASSED on eng/H200
(13549747) and mgh/GH200 (13549749) — first IGE GPU smokes ever to pass —
and both probes COMPLETED (first GPU probes ever for this case).

### 1r gate abort (both pools, same numbers — POLICY decision for Ryan)

- eng 13549747: late-probe median **15.17 s/step** → gate projects
  414×15.17×1.5 = **9422 s > 6480 s limit** → chain aborted before accept.
- mgh 13549749: **15.07 s/step**, projected 9356 s — identical abort.
  (Chain stop-flag commit `8413697` made mgh smoke→probe-only; gate fired
  first anyway, same information.)
- H200 vs GH200 within 1% (16 x86 cores vs 72 ARM cores): bottleneck is
  GPU-kernel-bound, not host-bound and not arch-specific.
- **The raw accept fits budget**: 414×15.17 = 6281 s < 6480 gate < 7200
  budget. Only the blanket 1.5× growth margin fails it — but the dominant
  cost is population-INDEPENDENT (see below), while only small terms grow
  with the wake. Options to put to Ryan: (a) component-aware margin (1.5×
  only on particle-coupled timers), (b) longer-wall accept on mgh (24 h
  MaxTime) to measure the real trajectory, (c) eat the panel↔panel cost
  via the longer-arc FMM route. Growth uncertainty: `body_influence`
  (panels→particles) was 2.32→2.57 s/step over the 2-rev probe and DOES
  grow with np; if it ×4s by plateau the projection is borderline again.

### Where 1r time goes (probe 13549747, last-30 means)

| timer | s/step |
|---|---|
| total_step | 15.16 |
| solve (block-GS) | 11.70 |
| body_influence | 2.57 |
| io / monitors / wake terms | ~0.8 |

Inside solve at step 80: `panel_cross_targets` **11.57 s** (dense
rectangular GPU kernel `_gpu_direct_batch!`, ALL panel-target influence
legs incl. per-GS-iteration self-source potential), `rotor_ground_cross`
1.36 s, `panel_self_cached_operator` 5.39 s (nested/overlapping timers —
not additive). 1r block-GS: 3–4 iterations/step, every step. The 052d/052h
FMM routes cover only panels↔particles legs; panel→panel has NO FMM route
(matches Ryan's stated exception for the influence/solve legs — but it is
now the whole budget problem). Ground 1r: 4752 cells; rotor mesh 45_185.

### 2r failure: block-GS does not converge (NOT a device bug)

- 13550027 (mgh, full 2r chain): smoke died at step 0 —
  `block Gauss-Seidel failed to converge in 50 iterations: normalized
  residual 0.00058224 exceeds tolerance 1.0e-8`
  (`require_outer_convergence=true`, thrown from FLOWPanel_solver.jl:2604).
- 13550282 (gsdiag re-run): **bit-identical residual** 0.0005822398176417601
  → fully deterministic. FAILED after 41 min (~33 min of that = two
  Backslash LU builds, ~990 s each, r1+r2 serial).
- Contrast: 1r (rotor+ground, same tol/backend/mesh) converges in 3–4
  iterations. Adding the second rotor changes convergence qualitatively.
- Two hypotheses the residual TRAJECTORY will separate:
  (a) slow contraction — rotor↔rotor coupling pushes block-GS spectral
  radius to ~0.86 (that rate lands at 5.8e-4 after 50 iters); remedy =
  cap/relaxation/block restructuring. (b) accuracy floor — residual AND
  cross-rotor RHS are evaluated through FastMultipoleBackend
  (FMM_BODY_EXPANSION_ORDER=17, ACCEPTANCE=0.7 — carrier line 43); if the
  rotor→rotor leg (centers 2.7R apart) carries ~1e-4 relative FMM error,
  no iteration count reaches 1e-8, and 1r never sees it because
  rotor↔ground proximity keeps interactions near-field/direct. Test for
  (b): raise FMM order / lower acceptance for the solve backend, or run
  one solve with DirectBackend, and watch the plateau move.
- Cross-rotor coupling IS implemented correctly (verified 2026-09-01,
  agent-traced + spot-checked): `solve!` tuple loop builds each body's RHS
  from ALL other bodies (`FLOWPanel_solver.jl:2407`), rotor→rotor labeled
  `:rotor_rotor_cross` (`:2414-2416`), driver assembles one flat
  `(rotors..., ground)` tuple with per-rotor Backslash + FlatGroundSolver
  (`rotor_hover_ground_effect.jl:1404-1411`). Nothing to add there.
- 2r geometry facts (gsdiag banner): ground disc 8422 panels, spacing
  2.7R, directions [+1,-1], max_particles=1,000,000, smoke=17 steps.

### GOTCHA: carrier clobbers GS env; GS_VERBOSE is parsed but DEAD

- `run_rotor_multi_ground_effect_gpu.slurm.sh:73` hard-exports
  `GS_LOG=true GS_MAX_OUTER=50 GS_TOL=1e-8` — sbatch `--export=ALL,GS_...`
  is silently overwritten. To sweep GS knobs, make line 73 default-guarded
  (`GS_MAX_OUTER="${GS_MAX_OUTER:-50}"` etc.) and commit.
- Driver reads `gs_verbose` (`rotor_hover_ground_effect.jl:114`) but NEVER
  passes it anywhere — grep confirms no use. `VelocityThroughSources` has
  no verbose field; the driver's GS instrumentation only records
  `block_gs_status_history` (post-hoc summaries, :1419-1429). To get the
  per-iteration trajectory, plumb verbose (or a `ConvergenceHistory`) into
  the tuple `solve!` call — e.g. have `solve_formulation!`
  (`FLOWPanel_formulation.jl:941-960`) read an env var, or add a formulation
  field. With `verbose=true`, `solve!` measures+prints residual EVERY
  iteration (`FLOWPanel_solver.jl:2489` measure_residual logic) —
  "  Outer iteration N: normalized block residual = ... (strength delta = ...)".
  NOTE: verbose residual passes cost ~1 influence sweep each — with
  GS_MAX_OUTER≈120 expect step 0 alone to take ~20 min for 2r; smoke will
  still die at step 0 via require_outer_convergence, which is fine — the
  trajectory is the product. Budget the LU build (~33 min) on top.

## State at reset

- No 052b jobs live. All five this-session jobs finished:
  13549747 (eng chain, gate abort), 13549749 (mgh smoke→probe, gate
  abort), 13550027 (2r chain, GS abort), 13550282 (2r gsdiag, GS abort —
  env clobber made it a no-new-info duplicate), plus predecessor failures.
- Do not disturb (others' lanes): 13542776/13542905/13542906 (p018 GPU,
  legacy silo `~/FLOWPanel-018-gpu-h200` — expected), 13548847
  `fp-022lg-hr10` (CPU production, m12, unified tree).
- Pins on orc `unified-052`: FLOWPanel **`8413697`** (HEAD: f5f4f29 device
  fix + 8413697 chain stop flag), FastMultipole `89ede6b`, FLOWVPM
  `6c8cda4`. Local FLOWPanel working tree mirrors both commits,
  uncommitted (standing policy).
- Chain script now supports `P022G_CHAIN_STOP_AFTER=probe` (evaluates
  gates, skips accept).
- Submit patterns (verified working this session): eng =
  `-p eng --qos=eng --gres=gpu:h200:1`; mgh = `-p mgh --qos=gpu
  --gres=gpu:gh200:1 -C arm` + export `P022G_REQUIRED_GPU_MODEL=GH200`
  (mgh had 2 idle GH200 nodes all session; sbatch REJECTS without
  `-C arm`). Probe with slurm-availability skill anyway.
- Monitors are dead (context reset); re-arm per §Monitors of the
  2026-09-01b handoff (scp scripts to orc, never nested-quote ssh; strip
  ANSI with `sed $'s/\x1b\\[[0-9;]*m//g'`; job's `.err` file often holds
  the only error text).

## Suggested next moves (in order; pipeline per the Mandate)

1. Guard carrier env line 73; plumb per-iteration GS residual logging
   (env-gated, default off); commit both on orc `unified-052`; mirror
   locally uncommitted.
2. Re-run 2r diagnostic smoke on mgh (idle) with GS_MAX_OUTER≈120 +
   trajectory logging. Read trajectory → floor vs slow.
   - If floor: re-run with FMM_BODY_EXPANSION_ORDER higher / ACCEPTANCE
     lower (carrier line 43) or DirectBackend for the solve; confirm
     plateau moves; then decide tolerance/accuracy policy conservatively
     and LOG it (Ryan away).
   - If slow contraction: measure rate; options = higher cap (cost:
     ~3-4 s per iteration per step at probe scale), under/over-relaxation,
     or two-rotor single block; pick the least invasive that converges,
     LOG the decision.
3. 1r accept: take mandate option (b) — long-wall accept on mgh — unless
   evidence says otherwise; LOG it.
4. Then the 2r chain once both blockers clear.
5. While any of the above simulations queue/run: advance 052e (hybrid
   wake-potential comparison runs — its task doc has the method; it is
   off the 052b critical path by construction) and 052c's remaining
   acceptance/consolidation items (read its plan doc "Current state" —
   much is already done/ruled). Delegate doc digestion to
   refactor-docs-librarian rather than reading the big docs inline.
6. 053: begin non-dependent rows (contract/counter audits of items
   already closed, defaults enumeration) any time; each item's final
   review — including 053's review of 052c — waits until that item is
   finished, then runs as a FRESH-CONTEXT review subagent (clear-context
   review rule, per mandate). 053 completes only after 052b/c/e are all
   done and reviewed.

## Flagged to Ryan (unanswered, carried + new)

- **1r gate policy** (new, blocking 1r accept): §1r gate abort options.
- **2r solver policy** (new, pending trajectory evidence): tolerance is
  1e-8 normalized (scale ωR²); residual floor may be FMM-accuracy-bound.
- **Storage** (carried): /home/rander39 was 618G vs 400G cap at 2026-09-01
  reset; sweeper approval pending (HANDOFF_CPU_20260901.md §Pending).
  Full-home `du` now times out over ssh (still huge). Accept runs add
  ~10s of GB VTK.
- **Notebook** (carried): GPU-route unblock, the four 2026-09-01b launch
  fixes, this session's device fix + gate/GS findings are all un-logged.
  Ryan is away: DRAFT entries via notebook-drafter as milestones land
  (052c explicitly owes a consolidation notebook entry) and hold them for
  his approval — the no-write-without-approval house rule stands.

## Key files

- orc `~/projects_unified/FLOWPanel.jl` (branch unified-052):
  `examples/rotor_hover_ground_effect.jl` (driver; damp override ~:1556,
  broadcasts ~:1586, monitor ~:1770, gs env :113-116, gs summary :1942,
  solvers/tuple :1324-1411), `examples/run_rotor_multi_ground_effect_gpu.slurm.sh`
  (carrier; GS clobber :73, FMM knobs :43-44),
  `examples/p022g_1r_ige_gpu_chain.slurm.sh` (chain; stop flag near end).
- `src/FLOWPanel_solver.jl` — tuple solve! :2320-2604 (freeze :2363, cross
  influence :2407-2443, self op :2451, residual :2470-2560, throw :2604);
  `src/FLOWPanel_formulation.jl` — VelocityThroughSources :31-62,
  solve_formulation! :941-960; `src/FLOWPanel_gpu_influence.jl` — dense
  batch + detail labels :760-800, FMM route flags :835-930.
- Logs (orc, under `~/projects_unified/FLOWPanel.jl/`):
  `logs/chain/p022g_1r_ige_{smoke,probe}_13549747.log`,
  `logs/chain/p022g_2r_ige_smoke_13550027.log`,
  `logs/slurm/slurm-fp-022g-2r-gsdiag-13550282.{out,err}`.
- Local mirror: `~/Dropbox/research/projects/FLOWPanel.jl` (uncommitted).
- Task doc: `MATRIX_OPERATOR_REFACTOR/052b-impl-multirotor-ige-gpu.md`;
  contract: `FLOWPanel.jl/BRAINSTORM/022_rotor_hover_ground_effect/phase_06_multirotor_gpu.md`.

## House rules (carried forward verbatim)

4 threads max locally; julia-test-runner for runs/scripts (output →
scratchpad log, grep it); refactor-docs-librarian for MATRIX_OPERATOR_REFACTOR
doc questions; verifier before reporting claimed numbers; never read
`data/**`/`*.csv`/`*.bin` raw; notebook writes need Ryan's approval FIRST;
commits/pushes on local trees only on Ryan's ask (orc unified-052 pin
commits before launch are sanctioned); GPU jobs authorized; scp scripts to
orc instead of nested ssh quoting; rsync --checksum; slurm needs `bash -lc`;
auth expiry → ask Ryan `! ssh orc echo ok`; probe partitions with the
slurm-availability skill before submitting; linegauss CONFIRMED for p022g
arms (Ryan 2026-09-01) — do not revert.
