# 052 handoff (session 2026-08-29n): 052d CLOSED — fix GPU memory leak, check reverse leg, launch 40-rev GPU rotor sim

## Prompt for the next agent

You are continuing task 052. 052d is CLOSED (see the closure block at the end
of `MATRIX_OPERATOR_REFACTOR/052d-plan-2026-08-26.md`): production filament
regularization is now pinned to LineGauss and validated on-device
(relU 9.058e-4 → 3.978e-5 at np=3544, verified). Do NOT redo any of that.

Your jobs, in order:

1. **Fix the GPU memory-growth regression** (the ~35 MB/step device leak).
2. **Check the reverse leg** (particles→panels wake influence) — assess and
   report to Ryan; do not implement without his go-ahead.
3. **Launch the 40-rev sigma rotor sim on the GPU** (production run — required
   before 052 can close) and **mark the launch in
   `/Users/ryan/Dropbox/research/projects/FLOWPanel.jl/plans/sigma_vpm_illustrations_20260827/HANDOFF.md`**
   so the waiting sigma-illustrations agent can watch it and finish its tasks.

House rules: 4 threads max locally; delegate runs to julia-test-runner and
MATRIX_OPERATOR_REFACTOR doc questions to refactor-docs-librarian; verify
claimed numbers via the verifier agent before reporting; never read
`data/**`/`*.csv`/`*.bin` directly (script a summary); long output → scratchpad
log, then grep. Read memory `orc-cluster-access.md` before ssh (bash -lc for
slurm; auth expires — ask Ryan to run `! ssh orc echo ok`). GPU jobs are
authorized for this task: tight time limits, combine stages into one
allocation where sensible, submit eng + m13h in parallel and scancel the
loser (`--qos=eng` on partition eng; `--partition=m13h --qos=gpu` for the
32-card H200 pool; `sbatch --test-only` gives ETAs). GH200: partition `mgh`,
`--gres=gpu:gh200:1 --constraint=arm` (2 cards, often idle).

## Task 1: GPU memory-growth regression

Read `MATRIX_OPERATOR_REFACTOR/052-gpu-memory-growth-longrun-2026-08-29.md`
first — it has the evidence and the regression recipe. Summary: long shedding
runs leak ~35 MB/step of device memory and die around step 800 (job 13508681,
the gpu40 attempt, ran 819/1475 steps at 6.3 s/step then OOMed). Hypothesis:
size-keyed CUDA caching per particle-count "window" never evicts — the
particle count grows every shed step, so each new size mints new cached
buffers. Ready-made regression case: `scr_p019_s038v_gpu40` on the GH200 silo
`~/FLOWPanel-018-gpu-gh200` (wrapper `examples/run_p018_screen_gpu.slurm.sh`,
env hook `SCR_GPU_RESERVE_GIB=16`).

Approach: find the size-keyed caches in the device path (FastMultipole
`src/*cuda*.jl` and FLOWPanel `src/FLOWPanel_gpu_influence.jl` — look for
Dict-by-size / get!-style buffer pools keyed on particle count or array
shape), add eviction or size-bucketing (e.g. grow-only geometric buckets so a
growing count reuses the same pool entry), and validate with a short leak
probe BEFORE burning a full 40-rev job: run a few hundred shed steps and
confirm device memory plateaus (the 052 note has the measurement recipe).
Root-cause first — do not paper over with periodic `CUDA.reclaim()` unless
the actual cache is found and understood.

## Task 2: reverse leg check (assess, report — no implementation yet)

The particles→panels wake influence still runs dense at ~0.13+0.10 s/step;
deferred in v1 until the cross pass matured (it now has). Design sketch at
`052d-plan-2026-08-26.md:970-987`: reuse the particle multipoles (free),
separate M2L+L2B pass onto the ~37k control points at their own leaf level,
block-sparse near field. Deliverable: a short assessment for Ryan — current
cost share in the production step, what the cross-pass machinery already
provides vs what is new work, estimated effort/risk — and stop for his
decision.

## Task 3: 40-rev sigma rotor sim on GPU + mark the plans dir

Context: an agent working `plans/sigma_vpm_illustrations_20260827/` (in the
FLOWPanel.jl repo) is blocked on a 40-rev sigma* production run. A CPU run
(job 13508968, `scr_p019_s038v_cpu40`, m12-2-18, main checkout, ~19-20 s/step
× 1475 steps ≈ 9 h) was launched 11:10 MDT 08-29 and may or may not have
landed — check `sacct -j 13508968` but judge by outputs (its HANDOFF warns
sacct FAILED is unreliable). The GPU rerun is BOTH the sigma agent's unblock
(6.3 s/step vs 19-20) AND the full production run Ryan requires before 052
closes.

After the leak fix is validated:
- Relaunch `scr_p019_s038v_gpu40` from the GH200 silo wrapper
  (`~/FLOWPanel-018-gpu-gh200/examples/run_p018_screen_gpu.slurm.sh`,
  `SCR_GPU_RESERVE_GIB=16`, NREVS=40). Deploy the leak fix to the silo via
  `rsync --checksum` (stale same-size files survive `-az`).
- **Regularization consistency**: cpu40 launched BEFORE the LineGauss default
  pin, and the 018 silo checkout may predate the LineGauss code entirely. If
  the silo has the env hook, decide with Ryan whether gpu40 should pin
  `FLOWPANEL_FILAMENT_REG=gaussian` to match cpu40 (comparability) or run the
  new LineGauss default (production truth). Do not silently mix.
- Once the job is RUNNING, append a dated status note to
  `plans/sigma_vpm_illustrations_20260827/HANDOFF.md` (that file is
  append-style; NEVER tick its checkboxes — Ryan does): job id, node, case
  name, run dir (`~/FLOWPanel-018-gpu-gh200/data/scr_p019_s038v_gpu40`),
  s/step, leak-fix status, and which regularization it runs, so the sigma
  agent can watch it and harvest (its own next steps are in that HANDOFF's
  "NEXT STEPS" section — monitor CSV, VTP windows, renders).
- The old gpu40 run dir (18 G) and salvaged monitor CSV exist — see HANDOFF
  lines ~291-311; do not delete anything (archive decisions are Ryan's).

## Established 052d facts (do NOT re-derive)

- Root cause and fix: see the closure block in `052d-plan-2026-08-26.md`
  and `052d-handoff-prompt-2026-08-29m.md` (full evidence chain).
- Validation artifacts: `data/052d_relU_dumps_6lg/` (job 13511424 dumps +
  xverify log), `scripts/fp052d_step6_linegauss.sh`.
- Verified numbers: relU(np=3544) 3.978e-5; p39 dev_vs_dns 3.977e-5,
  hfmm_vs_dns 7.42e-8; 59-line trajectory 3e-5–7e-5, max 1.28e-4 @ np=4598.

## Uncommitted state (do not lose; Ryan decides what/when to commit)

- FLOWPanel (branch `fastmultipole`): device cross-pass stages A–F, LineGauss
  kernel + host port, the NEW default pin
  (`FILAMENT_REGULARIZATION = Ref(LineGaussRegularization)` in
  `FLOWPanel_elements_fmm.jl` + docstring updates), session-k/m dump hooks in
  `FLOWPanel_gpu_influence.jl`. Synced to ORC `~/FLOWPanel-052-h200/src/`
  (NOT to the 018 silo).
- FastMultipole (branch `flowpanel-20260817`): session-k fixes
  (translate_batched_cuda.jl, cross_stencil_cuda.jl), prototypes p34–p51,
  scripts step5d/5e/6, plan-doc closure block, this handoff.

## Open Ryan decisions (carry forward; ask, don't assume)

- Commit breakdown, incl. keep-env-gated vs strip the dump hooks.
- Notebook entry for the 052d closure (draft via notebook-drafter, get his
  approval + detail level BEFORE writing anything to the notebook).
- xverify-gate: routine production guard vs debug-only.
- LineGauss near-field perf check (4 erf + 1 exp vs 1 expm1 per edge).
- gpu40 regularization choice (see Task 3).
