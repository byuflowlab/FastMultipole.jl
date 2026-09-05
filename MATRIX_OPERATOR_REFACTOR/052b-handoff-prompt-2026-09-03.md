# 052 phase handoff (session 2026-09-03 pre-dawn): three jobs in flight; 2r trajectory readout pending; 052c trial-1b queued (eng full → move to mgh); 053 defaults draft written

## Prompt for the next agent

You are continuing the MATRIX_OPERATOR_REFACTOR 052 phase after a context
reset. Read this file first; it is authoritative. Predecessor:
`052b-handoff-prompt-2026-09-02.md` — its Mandate section (finish
052b/052e/052c then 053, subagents for everything bulky, fresh-context
reviews, pipelining authorized, DECISION LOG maintained) still governs;
its §Decision log holds all decisions through this session — append new
ones THERE. Its technical background (1r gate abort numbers, 2r failure
hypotheses, key files, house rules) remains valid except where superseded
below. HPC.md (`FLOWPanel.jl/agent_policies/HPC.md`) required before
cluster work.

## CRITICAL path updates vs the 2026-09-02 handoff

- **orc repos moved**: `~/projects_unified/*` → `~/projects/*`. Every
  projects_unified path in older docs is stale.
- **FLOWPanel unified-052 HEAD is now `6a64402`** (my commit: carrier GS
  env default-guarded + GS_VERBOSE plumbed into tuple `solve!` via
  `solve_formulation!`; behavior-neutral when GS_VERBOSE unset). Local
  mirror has same edits, uncommitted (standing policy).
- **FastMultipole 052f/052g fixes are already everywhere they're needed**
  (unified `~/projects/FastMultipole` HEAD 3da58a1, silo
  `~/FastMultipole-052-h200`, silo `~/FastMultipole-052-gh200` — all
  verified containing the zero-M2L `win_class === nothing` guard). Do NOT
  re-attempt the bundle/cherry-pick port; it was unnecessary.

## Jobs in flight at reset (check these FIRST)

| job | what | where | wall | state at reset |
|---|---|---|---|---|
| 13568975 | 2r gsdiag smoke, GS_VERBOSE=true GS_MAX_OUTER=120 | mgh-1-2 GH200 | 04:00 | R ~37 min |
| 13568974 | 1r long-wall accept (mandate option b), P022G_MODE=accept | mgh-1-1 GH200 | 24:00 | R ~37 min |
| 13569052 | 052c trial-1b acceptance rerun (Ryan submitted 05:33) | eng H200 | 06:00 | PD (Priority, behind p018 fleet) |

Logs: `~/projects/FLOWPanel.jl/logs/slurm/slurm-fp-022g-2r-gsdiag2-13568975.{out,err}`,
`...-1r-accept-lw-13568974.{out,err}`;
trial-1b: `~/FLOWPanel-052-h200/data/fp052c-trial1-13569052.out`.
Both mgh jobs run FLOWPanel=6a64402, FastMultipole=3da58a1, FLOWVPM=186bff4.
Do not disturb: fp-018gpu-* fleet (other lane, eng+m13h), 13548847.

**All session-only crons/monitors died with the reset — re-arm monitoring
immediately** (a 03:37 check-in cron and its readout plan are gone).

## Next moves, in order

1. **2r gsdiag readout (unblocks all remaining 2r work):** grep the
   gsdiag .out for per-iteration lines "Outer iteration N: normalized
   block residual". Classify: (a) plateau at ~5.8e-4 across iterations →
   accuracy FLOOR → rerun with FMM_BODY_EXPANSION_ORDER higher /
   FMM_BODY_ACCEPTANCE lower (carrier lines 43-44) or DirectBackend and
   confirm the plateau moves; (b) steady geometric contraction (~0.86
   rate lands 5.8e-4 at 50 iters) → SLOW → measure rate, remedy = higher
   cap / relaxation / two-rotor single block, least invasive. Decide
   conservatively, LOG in the 2026-09-02 decision log. NOTE the job
   die-at-step-0 via require_outer_convergence is EXPECTED; trajectory is
   the product. Expect job done well before its 4h wall (LU ~33 min +
   step 0).
2. **052c trial-1b → mgh:** Ryan ruled eng will be full for a while; move
   13569052 to mgh. mgh is GH200/ARM: the launcher
   `~/projects/launchers/fp052c_trial1_run.sh` hardcodes x86 silo paths
   (FPDIR=~/FLOWPanel-052-h200, VPMDIR=~/FLOWVPM-052-h200,
   ENVDIR=~/fm052env-h200, module julia/1.11.7-6bmogfl). A full GH200
   silo set EXISTS: ~/FLOWPanel-052-gh200, ~/FLOWVPM-052-gh200,
   ~/FastMultipole-052-gh200 (fix verified present), ~/fm052env-gh200.
   Plan: copy the launcher to fp052c_trial1_gh200_run.sh with -gh200
   paths (check the julia module name for ARM — see how
   p022g jobs load julia on mgh, or `module avail julia` on an ARM node;
   also verify ~/fm052env-gh200/Manifest.toml dev-paths point at the
   -gh200 silos), verify TOLERANCE/CPU_RUN reference paths (they point at
   ~/FLOWPanel-052/data/... which is arch-neutral data — keep), then
   scancel 13569052 and submit the gh200 variant to mgh
   (`-p mgh --qos=gpu --gres=gpu:gh200:1 -C arm`, 64→72 cpus optional,
   6h wall) once a mgh node frees (gsdiag ends early). CAUTION: confirm
   the gh200 silo FLOWPanel/FLOWVPM carry the trial-1 sigma_guard
   changes (diff vs -h200 silo: `diff -r --brief` src dirs) before
   submitting; if they diverge, rsync the -h200 src over (log it).
   Job name fp052c-trial1c. LOG the migration.
3. **1r accept:** runs until ~01:56 MDT 2026-09-04 max. Periodically
   extract step times (FLOWPANEL_STEP_TIMERS output) to answer the
   growth-margin question (`body_influence` trend especially). If it
   completes: first-ever 1r IGE GPU accept — harvest gates.
4. **052e (off critical path):** task doc digested this session — goal is
   production-feasibility of HybridWakePotential; step 1 is re-running
   host formulation regressions AFTER 052b closes; steps 2-5 are
   accuracy sweeps, perf/memory measurement, CUDA proof, promotion
   ruling (gates: accuracy pre-registered tolerances; 414-step < 6480 s
   with ≥20% GPU mem reserve). Groundwork you can do now: draft the
   pre-registered tolerance table and enumerate the comparison-run
   matrix into a plan file.
5. **053:** row-2 defaults draft exists:
   `053-defaults-enumeration-draft-2026-09-03.md` (9 rows, mostly NEEDS
   RYAN; grep-based audit — unified branches squashed history; noted in
   doc). Rows 1/4/5 wait on 052 data; row 3 (three-repo test suites
   green) can run any time on idle CPU — delegate to julia-test-runner.
6. **052c after trial-1b/c completes:** harvest gate table + min_sigma
   trajectory, update the 052c ledger, then commit-plan proposal and the
   052/052a consolidation NOTEBOOK DRAFT (notebook-drafter; needs Ryan
   approval before any notebook write).

## 052c trial-1b background (diagnosed this session)

Original 13501691 (2026-08-27, never diagnosed until now): run 1
(756-step) PASSED all gates — CT cycle-mean 6.88e-5 (ceiling 1.8e-3),
Gamma M2 max 1.37e-4 (2.93e-3), RMS 5.40e-5 (1.5e-3). Run 2 crashed at
step 894/1079: `TypeError: typeassert expected CuArray{Int32,1}, got
Nothing` in `_launch_cuda_hierarchical_m2l_cached!`
(translate_batched_cuda.jl:8052 pre-fix numbering) — the zero-M2L
degenerate cache bug, fixed by 052f `d938ba68` + 052g `2c6dd60f` (both
post-date the job, both deployed everywhere now). NOT the step-~1071
rVPM core collapse. Phase-2e CT convergence printed CONVERGED=false with
large per-rev spread in run 1 — separate, likely pre-existing, non-fatal;
worth flagging to Ryan with trial-1b results.

## Permissions / classifier situation (matters for how you work)

Session runs in auto mode with an LLM classifier vetting un-allowlisted
calls. Ryan added allow rules to `~/.claude/settings.json` this session:
`Bash(ssh orc*)`, `Bash(scp *)`, `Bash(rsync*)`, squeue/sacct/sinfo,
cat/head/tail/grep/ls/date, git log/show/diff/status, and
`Agent(...)` for refactor-docs-librarian, julia-test-runner, verifier,
notebook-drafter, Explore, general-purpose; plus Edit/Write/Read scoped
to the FastMultipole, FLOWPanel.jl, FLOWVPM.jl repos ONLY (Ryan
explicitly refused broader research/ scope). Rules load at session start
— they should be live for you. If sbatch-over-ssh still gets blocked:
try once via julia-test-runner agent; if that is also blocked, STOP and
hand Ryan the exact one-liner to run with the `!` prefix (worked this
session). Do NOT edit the permissions file except on Ryan's explicit
ask. Start commands with `ssh orc` (prefix-match) and use `git -C` not
`cd repo && git`.

## Misc state

- Decision log: in `052b-handoff-prompt-2026-09-02.md` §Decision log
  (8 entries through this session) — keep appending there.
- Cleaned up: fm-local.bundle, fmpatches/, local-20260903 branch on orc.
- Storage flag still open (~618G vs 400G cap); accept runs add VTK.
- Notebook debt (carried): GPU-route unblock, device fix, gate/GS
  findings, 052c diagnosis — all undrafted; notebook-drafter + Ryan
  approval when milestones land.
- Flagged to Ryan (carried + new): 1r gate policy; 2r solver policy
  (pending trajectory); storage; two unapproved defaults found by the
  053 audit (FLOWPanel FMM_RADIUS_TOL inflation; FLOWVPM RadixFMM
  expansion_order=6 vs docstring 4).
- House rules verbatim in the 2026-09-02 handoff §House rules still
  apply (4 threads local, julia-test-runner for runs, librarian for
  docs, verifier before reporting numbers, never read data/*.csv raw,
  scp scripts not nested quotes, `bash -lc` for slurm, linegauss pinned).
