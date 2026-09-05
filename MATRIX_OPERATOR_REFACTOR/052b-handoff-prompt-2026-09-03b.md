# 052 phase handoff (session 2026-09-03b): 2r floor CONFIRMED-pending-gsdiag3; all 3 jobs ended (2 trivial resubmits, 1 harvest); silo sigma_guard skew found; both mgh nodes idle

## Prompt for the next agent

You are continuing the MATRIX_OPERATOR_REFACTOR 052 phase after a context
reset. Read this file first. The mandate and ALL decisions live in
`052b-handoff-prompt-2026-09-02.md` (§Mandate, §Decision log — 14 entries
through 2026-09-03 ~08:15; append new decisions THERE). The 2026-09-03
predecessor (`052b-handoff-prompt-2026-09-03.md`) is superseded by this
file except its §Permissions and §052c-background sections, still valid.
HPC.md (`FLOWPanel.jl/agent_policies/HPC.md`) required before cluster
work. House rules verbatim in the 2026-09-02 handoff still apply.

## State at reset (nothing running; monitors dead — re-arm after resubmits)

Both mgh GH200 nodes (mgh-1-1, mgh-1-2) IDLE at reset. No 052-lane jobs
queued. Do not disturb fp-018gpu-* fleet (other lane, eng+m13h).

### Headline result this session: 2r block-GS residual = accuracy FLOOR

13568975 (gsdiag2): strength delta contracts geometrically to machine
epsilon in ~6 outer iterations; normalized block residual PINNED at
5.8224e-4 iters 2→120 (16th-digit variation only). Solver NOT slow —
FMM cross-influence accuracy floors the residual. Cap/relaxation/
single-block remedies ruled OUT. GS_TOL=1e-8 unattainable at
body FMM (17, 0.7). Confirmation run gsdiag3 (body FMM 20/0.5) died at
21 s on a trivial guard (below) — resubmit is next-move 1.

### The three job outcomes (all sacct FAILED, all understood)

| job | outcome | action |
|---|---|---|
| 13568975 gsdiag2 | SUCCESS-in-substance (trajectory captured; die-at-step-0 expected) | none — readout logged |
| 13568974 1r accept | ran ALL 413/413 steps, 2:12:24, steps 14→19.5 s; failed only gates: elapsed 7917 s > hardcoded 7200 s; Phase-2e CONVERGED=false (spread 0.523 vs 0.005) — but readout window starts rev 1.0 while hover begins rev 6.5 (transient included) | harvest offline, no rerun (below) |
| 13569059 trial-1c | died 4:36 in warmstart propagate!: `unknown sigma_guard key ceil; recognized: (:dtz_cap, :floor)` | fix silo skew, resubmit (below) |
| 13569088 gsdiag3 | died 21 s: `data/p022g_2r_ige exists; set P022G_EXISTING_RESULT=preserve` (FMM override plumbing verified working — banner body_fmm=20/0.5/109) | resubmit with env var (below) |

## Next moves, in priority order

1. **Resubmit gsdiag3** (2 min, nodes idle, closes the 2r question):
   from `~/projects/FLOWPanel.jl` on orc:
   `sbatch -p mgh --qos=gpu --gres=gpu:gh200:1 -C arm --time=04:00:00 -J fp-022g-2r-gsdiag4 --export=ALL,P022G_MODE=smoke,P022G_REQUIRED_GPU_MODEL=GH200,GS_VERBOSE=true,GS_MAX_OUTER=30,P022G_RUN_TAG=gsdiag4,FMM_BODY_EXPANSION_ORDER=20,FMM_BODY_ACCEPTANCE=0.5,P022G_EXISTING_RESULT=preserve examples/run_rotor_multi_ground_effect_gpu.slurm.sh p022g_2r_ige`
   Readout: if the residual plateau drops materially below 5.8e-4 →
   floor CONFIRMED → present Ryan the solver-policy options already
   logged (raise GS_TOL above floor / gate on strength delta / pay for
   FMM accuracy). LU ~33 min; expect lines ~40 min in. The carrier's FMM
   knobs are default-guarded since orc FLOWPanel commit 4e6b5b7 (mirrored
   locally, committed).
2. **Fix silo sigma_guard skew + resubmit trial-1c** (052c critical):
   ceil support lives in local `FLOWVPM.jl/src/FLOWVPM_timeintegration.jl`
   (~:24-42: `_sigma_guard_params` 3-key). Both `~/FLOWVPM-052-h200` and
   `~/FLOWVPM-052-gh200` on orc have the 2-key version while their paired
   FLOWPanel example passes 3 keys — internally skewed snapshot; eng
   trial-1b would have crashed identically (NOT an ARM problem). Port:
   scp the local file over both silo copies (verify surrounding context
   matches first — silo may lag elsewhere; if so, port just the
   `_sigma_guard_params` + docstring hunk), then resubmit:
   `sbatch -p mgh --qos=gpu --gres=gpu:gh200:1 -C arm -N1 -n1 -c72 --mem=192G -t 6:00:00 -J fp052c-trial1d -o $HOME/FLOWPanel-052-gh200/data/fp052c-trial1d-%j.out ~/projects/launchers/fp052c_trial1_gh200_run.sh`
   (launcher already fixed: gh200 paths, ARM julia+depot+PATH,
   CORRECTED reference paths — the old `~/FLOWPanel-052/data/...` were
   stale, real data is under `~/projects/FLOWPanel.jl/data/`). LOG the
   port. Then monitor: stage-1 gate ~expected pass, stage-2 is 6 h.
3. **Harvest 1r accept offline (no rerun)**: per-rev data written to
   `~/projects/FLOWPanel.jl/data/p022g_1r_ige/p022g_1r_ige_CT_per_rev.csv`
   (+ case_metadata.toml). Recompute Phase-2e convergence over the HOVER
   window only (hover begins rev 6.5; current readout windowed from rev
   1.0 and so ate the spin-up + freestream-pulse transient). Small julia
   script via julia-test-runner; never cat the CSV. Two Ryan flags:
   (a) the 7200 s elapsed gate is hardcoded — 7917 s actual; is the gate
   policy or adjustable? (b) whether a fixed readout window is a carrier
   bug to patch (find it in examples/run_rotor_multi_ground_effect_gpu.slurm.sh
   or the driver). If recomputed hover-window spread passes tolerance,
   this is the first-ever 1r IGE GPU accept IN SUBSTANCE — notebook-worthy.
4. **Re-arm monitoring** for whatever gets submitted (session monitors
   died; NOTE this session's monitor missed all three job-end events —
   check states directly at turn start, don't trust silence).
5. **052c after trial-1d completes**: harvest gate table + min_sigma
   trajectory, update 052c ledger
   (`052c-sigma-experiments-2026-08-26.md`), commit-plan proposal, and
   the 052/052a consolidation NOTEBOOK DRAFT (notebook-drafter; Ryan
   approval before any notebook write).
6. **053 row 3 wrap-up**: FastMultipole GREEN; FLOWVPM GREEN (after my
   stale-test fix to the 052f demotion contract); FLOWPanel 656/658 with
   2 OPEN failures — `explicit jump fallback (:jump)`
   (test/runtests_unit_kutta.jl:539-540). Diagnosis so far (2026-09-03
   afternoon, in-session): fallback `_kutta_trial!` hand-rolls its RHS
   (σ from frozen velocity + self potential + W·c, FLOWPanel_kutta.jl
   ~:860) and is no longer bitwise-equal to the legacy A/jump trajectory;
   μ-column off 4-5x, wake row ~2x, source column ~1.5%. Likely culprit
   commit 7fbd68a (Dirichlet targets now get scalar potential in
   cross/self influence; `want_potential = has_dirichlet_bc`). Caveat:
   test is SINGLE-body, so the cross-BODY term itself can't fire — carrier
   is probably the wake→body / self-potential Dirichlet changes in the
   same commit. Sibling testset "default pair is the legacy path" passes
   → attachment kwarg exonerated. DECISIVE next step (Ryan showed
   interest): worktree at 7fbd68a^, run test/runtests_unit_kutta.jl,
   confirm green there. Then decide: update `_kutta_trial!` to the new
   convention vs relax bitwise to tolerance. NEEDS RYAN on which is
   canonical.
7. **052e / 053 pipelining**: 052e groundwork DONE —
   `052e-accuracy-plan-draft-2026-09-03.md` (T1-T6 proposed tolerances
   NEED RYAN ratification = pre-registration lock; 9 OFAT pairs + 1 long
   run; step 1 still blocked on 052b closure). 053 draft
   (`053-defaults-enumeration-draft-2026-09-03.md`) has a row-3 status
   section appended; rows 1/4/5 wait on 052 data, most rows NEED RYAN.

## Local uncommitted edits this session (keep; commit only on Ryan's ask)

- `FLOWPanel.jl/test/Project.toml`: added Logging stdlib (was a hard
  LoadError masking the whole back half of the suite).
- `FLOWVPM.jl/test/runtests_gpu_fmm.jl`: sigma-outgrown final sub-case
  now expects the 052f demotion warning (`@test_logs (:warn,
  r"all-direct zero-M2L")`) instead of `@test_throws ArgumentError`.
- `FLOWPanel.jl/examples/run_rotor_multi_ground_effect_gpu.slurm.sh`:
  FMM knob default-guards (mirrors orc commit 4e6b5b7).
- orc: `~/projects/launchers/fp052c_trial1_gh200_run.sh` (new),
  `fp052c_trial1_run.sh` (stale paths patched in place).

## Flags for Ryan (carried + new this session)

- NEW: 2r solver policy options (pending gsdiag4 confirmation) — logged
  2026-09-03 ~07:05 entry.
- NEW: 1r accept gates — 7200 s elapsed gate (7917 actual) and the
  rev-1.0 readout window; run itself completed healthy.
- NEW: silo FLOWVPM sigma_guard skew (fix in motion, next-move 2).
- NEW: FLOWPanel kutta `:jump` bitwise contract broken (656/658),
  likely 7fbd68a — needs canonicality ruling.
- Carried: storage (~618G vs 400G cap; accept runs add VTK); 1r gate
  policy; two unapproved defaults from the 053 audit (FLOWPanel
  FMM_RADIUS_TOL inflation; FLOWVPM RadixFMM expansion_order=6 vs
  docstring 4); notebook debt (GPU-route unblock, device fix, gate/GS
  findings, 052c diagnosis, and now the 2r-floor + 1r-run results —
  all undrafted, need Ryan approval to write).

## Key logs (orc)

- gsdiag2 trajectory: `~/projects/FLOWPanel.jl/logs/slurm/slurm-fp-022g-2r-gsdiag2-13568975.out`
- 1r accept: `...-1r-accept-lw-13568974.{out,err}` + `data/p022g_1r_ige/`
- trial-1c crash: `~/FLOWPanel-052-gh200/data/fp052c-trial1c-13569059.out` (:119)
- gsdiag3 guard-death: `...-2r-gsdiag3-13569088.{out,err}`
