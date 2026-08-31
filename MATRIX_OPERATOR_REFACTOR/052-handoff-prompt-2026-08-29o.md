# 052 handoff (session 2026-08-29o): leak FIXED+VALIDATED, gpu40 RUNNING — monitor, harvest, close out

## Prompt for the next agent

You are continuing task 052. Session 2026-08-29o closed out the GPU
memory-leak regression and relaunched the production run. Do NOT redo any of
it. State of the world:

1. **Leak fixed and validated (task 1 DONE).** Root cause of the ~35 MB/step
   device leak that killed job 13508681 at step 819: size-rekeyed,
   one-step-surviving device buffers. FastMultipole's
   `_cuda_cached_target_buffer` (finalize ~16×np×8B + SFS 3×np buffers,
   `src/translate_batched_cuda.jl`) reallocated on EVERY particle-count change
   (= every shedding step); each replaced CuArray survives exactly one step in
   the cache dict, is promoted to Julia's old generation, and is then never
   collected because device bytes are invisible to the host GC heuristics —
   dead pool blocks accumulate linearly (this also explains the flat phase at
   steps 60–220 of 13508681: np was not changing there). Fixes:
   - FastMultipole `_cuda_cached_target_buffer`: grow-only capacity buffer
     (25% geometric headroom) + contiguous column-prefix view. Verified
     view-compatible with FLOWVPM's `buffer_to_target!`/`sfs_to_target!`
     (`CUDA.AnyCuArray` + `size(buf,2)==np` checks).
   - FLOWVPM `_radix_fmm_coupling!`: explicit `GC.gc(true)` when the rare
     depth/sigma-outgrown rebuild drops the old GB-scale cache.
   - FLOWPanel device cross pass (NEW stack only, local): same pattern much
     bigger — `e.ctx` + `e.ls` (~10²+ MB) were rebuilt every shed step. Fixed
     via padded grow-only particle capacity in
     `src/FLOWPanel_gpu_influence.jl`: `_cross_padded_positions!` keeps a
     persistent 3×np_cap Float64 device mirror (np_cap = np + max(np/16,1024),
     grow-only), tail wrap-fills with REAL particle positions (identical
     occupancy ⇒ identical route/block lists, ~6% duplicate target work,
     particles are targets-only in this pass so physics for the live prefix is
     unchanged); accumulation/xverify/dumps slice `1:np`. Entry rebuilds now
     happen only on capacity growth and are followed by `GC.gc(true)`.
   - **Validation (probe job 13511798**, `scr_p019_s038v_leakprobe`, a new
     case arm = gpu40 physics at NREVS=10, 396 steps, GH200 mgh-1-1):
     pool_reserved froze at 15.27 GB from gemv 40→390, free pinned at
     21.357 GB, pool_used trendless 11.7–14.5 GB. Old job grew ~35 MB/step
     over the same window. Numbers extracted directly from the
     `source_s_gpu_memory` lines in
     `~/FLOWPanel-018-gpu-gh200/logs/slurm/slurm-fp-il-s038v-leakprobe-13511798.err`.
     sacct shows FAILED — that is the wrapper's NaN gate tripping on the short
     run's empty CT-plateau printout; the run finished 396/396 steps, all-GPU
     gemvs, all finite = true.

2. **gpu40 production run RUNNING (task 3 DONE at launch level).**
   Job **13512297** `scr_p019_s038v_gpu40`, GH200 **mgh-1-2**, launched
   ~18:55 MDT 08-29, 1475 steps, 24 h wall,
   `sbatch --job-name=fp-il-s038v-gpu40 --export=ALL,SCR_GPU_RESERVE_GIB=16
   examples/run_p018_screen_gpu.slurm.sh gh200 scr_p019_s038v_gpu40` from
   `~/FLOWPanel-018-gpu-gh200`. **Ryan's explicit choice: 018 silo stack +
   gaussian-era filaments** (comparable to cpu40 and to the salvaged 819-step
   gpu40; the silo predates LineGauss and the cross pass — no
   FLOWPANEL_FILAMENT_REG hook exists there). Old 18 G run dir preserved by
   the dispatcher as `data/scr_p019_s038v_gpu40.prev` — do not delete
   (archive decisions are Ryan's). Sigma HANDOFF
   (`FLOWPanel.jl/plans/sigma_vpm_illustrations_20260827/HANDOFF.md`) has a
   dated status note appended (2026-08-29 evening) so the sigma agent can
   watch it. cpu40 (13508968, m12-2-18) was RUNNING at ~9 h elapsed, on track
   (~19-20 s/step × 1475 ≈ 9 h + margin).

   YOUR MAIN JOB: monitor 13512297 (`sacct -j 13512297`, judge by outputs —
   step lines in `logs/slurm/slurm-fp-il-s038v-gpu40-13512297.out`, memory
   sample lines in the matching `.err`). Confirm pool_used stays trendless
   through the late-run np sweep (the real regression test; the probe only
   covered 396 steps). If it dies again, the `.err` memory trace localizes
   what still grows. When it lands, confirm the sigma agent picked it up or
   append a landing note to the sigma HANDOFF (append-only, NEVER tick its
   checkboxes).

3. **Reverse-leg assessment (task 2) — DELIVERED, awaiting Ryan's decision.**
   Summary (full version in the 2026-08-29o session report to Ryan):
   - Cost today: `wake_to_rotor_panels` 0.127 + `wake_to_probes` 0.100 s/step
     = ~3.9% of the 5.79 s/step median, measured at np≈209k on H200 (job
     13494373, 052c probe, steps 720–755; provenance
     `052c-plan-2026-08-26.md:299-311`). Scales ~linearly with np; projected
     ~0.5 s/step at 4-rotor (`052d-plan-review-response-2026-08-26.md:90-93`).
   - Machinery already in place: two-occupancy device producer
     (`refresh_cross_producers!` — the PANEL occupancy over ~37k control
     points already exists in every cross entry), M2L operator tables +
     class-slot machinery, L2L/L2B device kernels, near-field direct
     machinery, the leak-safe capacity pattern.
   - New work: (a) particle-side multipoles on the cross grid — either a
     device B2M for point vortices on the two-occupancy grid (device vortex
     B2M kernels exist in FastMultipole) or a seam to reuse the radix
     self-pass multipoles (cheaper at runtime, more plumbing: node-mapping
     between the radix tree and the cross grid); (b) reversed route
     generation (particle nodes → panel nodes; needs a panel-side dense
     `node_at`, currently particle-side only); (c) L2B at control points +
     wiring into the solve RHS (U-only by default; U+J if
     `PANEL_WAKE_HESSIAN_TO_PARTICLES=true`); (d) xverify harness vs the
     dense leg. Estimate: 2–4 focused sessions incl. GH200 validation.
   - Risk: low physics risk (env-gated like the forward pass, dense fallback
     stays); modest payoff at 1 rotor (~0.2 s of ~6 s), real payoff at
     4-rotor. Recommendation: defer until after the 40-rev harvest unless
     4-rotor work is imminent. NO implementation without Ryan's go.

## House rules (carried forward)

4 threads max locally; delegate runs to julia-test-runner and doc questions
to refactor-docs-librarian; verify claimed numbers via verifier before
reporting; never read `data/**`/`*.csv`/`*.bin` directly; long output →
scratchpad log then grep. ssh: `ssh orc 'bash -lc "source /etc/profile; ..."'`,
banners are noisy (grep -v), auth expires — ask Ryan to run
`! ssh orc echo ok`. Read memory `orc-cluster-access.md`. Rsync to cluster:
`--checksum`. GPU jobs authorized: tight limits, combine stages, eng+m13h
parallel submit for H200, `mgh --gres=gpu:gh200:1 --constraint=arm` for GH200.

## Uncommitted state (do not lose; Ryan decides what/when to commit)

- NEW this session (2026-08-29o), all uncommitted:
  - FastMultipole `src/translate_batched_cuda.jl`: capacity-buffer fix.
  - FLOWVPM `src/FLOWVPM_fmm_radix.jl`: GC-after-cache-drop.
  - FLOWPanel `src/FLOWPanel_gpu_influence.jl`: `_CROSS_POS`/`_CROSS_NP_CAP`
    + `_cross_padded_positions!`, np_cap plumbed through `_cross_entry!` /
    `_cross_run_body!`, prefix slicing, GC-after-entry-rebuild.
  - GH200 silo patches (`~/FastMultipole-018-gpu-gh200`,
    `~/FLOWVPM-018-gpu-gh200`, `~/FLOWPanel-018-gpu-gh200/examples` case arm
    `scr_p019_s038v_leakprobe`): applied via `/tmp/patch_052leak.py` (copy in
    this session's scratchpad), `.bak-052leak` backups beside each patched
    file. NOTE: the silo FastMultipole/FLOWVPM fixes are the same text as the
    local edits; the local FLOWPanel cross-pass fix is NOT on the silo (silo
    predates the cross pass — nothing to fix there).
- Prior uncommitted state (unchanged): FLOWPanel branch `fastmultipole`
  cross-pass stages A–F + LineGauss + default pin (synced to
  `~/FLOWPanel-052-h200`, NOT the 018 silo); FastMultipole branch
  `flowpanel-20260817` session-k fixes + prototypes p34–p51 + closure docs.

## Open Ryan decisions (ask, don't assume)

- Reverse leg: implement now vs defer (assessment above).
- Commit breakdown for the leak fixes + prior 052d work; keep-env-gated vs
  strip dump hooks.
- Notebook entry for 052d closure AND for the leak fix (draft via
  notebook-drafter, get approval + detail level BEFORE writing).
- xverify-gate: routine production guard vs debug-only.
- LineGauss near-field perf check.
- A LineGauss/cross-pass "production truth" gpu40 rerun later (this one is
  gaussian for comparability, per Ryan).
- Old gpu40 `.prev` dir + probe run dir cleanup (archive-first, Ryan only).

## Established facts (do NOT re-derive)

- 052d closed: LineGauss pinned, relU 9.058e-4 → 3.978e-5 at np=3544
  (closure block in `052d-plan-2026-08-26.md`).
- Leak evidence + old trajectory: `052-gpu-memory-growth-longrun-2026-08-29.md`.
- Probe validation numbers: this file, item 1.
