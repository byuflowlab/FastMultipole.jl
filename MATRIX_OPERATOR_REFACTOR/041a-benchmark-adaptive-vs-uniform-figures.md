# 041a Benchmark Report: Adaptive vs Uniform Grid, Publishable Figures

## Status and Entry Gate

**DONE 2026-08-17 (lead agent; awaiting clear-context approval).**
Completion notes, data provenance, and verification notes at the end of
this file; report of record at `data/adaptive_octree/report.md`; figures
`fig14`–`fig20` in `data/figures/` (all compile with pdflatex, verified
locally 2026-08-17).

Entry gate: `040` and `041` complete and approved (host and CUDA adaptive
lifecycles both measurable).

Benchmark/analysis row: `scripts/`, `data/`, and figure artifacts only; no
production `src/` changes.

## Objective

Produce the definitive, publishable benchmark comparison of the old
(uniform-depth radix grid) and new (2:1-balanced adaptive octree) machinery
on both uniform and non-uniform particle fields, in time cost and memory —
figures that clearly showcase where the uniform approach breaks down and
what the adaptive approach recovers.

## Benchmark matrix

Fields (at minimum):

1. **Uniform cube** (phase case a) — the adaptive path's non-regression
   ground.
2. **Helical wake cylinder** (phase case b, 3.14% fill) — uniformly sparse,
   elongated.
3. **Multi-scale cluster case** (from `038`: cube + embedded dense clusters
   at 10–100× local density) — the uniform grid's designed weakness, swept
   over cluster contrast so the failure trend is a curve, not a point.
4. A σ-heterogeneous variant (`CoreSpreading`-like spread in smoothing
   radii) exercising the global-`σ_max`-gate weakness, if `038`'s per-cell
   gate shipped.

Configurations: old uniform grid at its best `ell` per case vs adaptive at
its best `K_max`; H200 Float32(+FP16 where applicable) and Float64; `P=4`
primary; single-thread and multi-thread host where informative. Every row
logs the sampled-direct velocity RMS error under the phase 1e-3 gate so all
comparisons are at matched, stated accuracy — a configuration that fails the
gate is plotted distinctly, never as a legitimate winner.

## Required figures

Follow the standing figure conventions (standalone TikZ/pgfplots `.tex`, one
same-named CSV data directory per figure, compiles with `pdflatex`,
stdlib-only Julia prep scripts). Numbering continues the `024a` set. At
minimum:

- **Time vs cluster contrast** (fixed `n`): end-to-end resident step, old vs
  new, showing the uniform grid's blow-up (fat-cell `O(K²)` nearfield /
  forced-shallow tree) and the adaptive curve staying flat.
- **Time vs `n`** on each field: old vs new, host and H200, with the
  uniform-case panel demonstrating non-regression.
- **Per-stage breakdown** (stacked bars) at representative points: B2M, M2M,
  M2L(V), M2T/S2L(W/X), L2L, L2B, direct(U), construction/refresh — old vs
  new side by side, exposing *where* the old approach loses.
- **Memory vs occupancy/contrast**: capacity-allocated and actually-used
  bytes, old (`min(8^ell, n)` capacity + dense `node_at`) vs new, host and
  device.
- **Accuracy-cost frontier**: time vs sampled-direct error at swept
  `P`/geometry, both machineries, confirming comparisons sit at matched
  accuracy.
- A **leaf-population distribution** figure (histogram/CDF, old vs new on
  the clustered case) as the mechanistic explanation for the timing gap.

## Deliverables

- Figure sources + CSV directories per the conventions above, staged with
  the `024a` figure set.
- A short report (`data/adaptive_octree/report.md`) stating per-case
  verdicts, the measured uniform-grid failure mechanism, memory savings, and
  the regimes where each machinery should be preferred — the evidence base
  for `042`'s default-selection audit.
- All raw timing/memory CSVs checksummed and reproducible from recorded job
  scripts (cluster/H200 runs; CPU baselines on cluster CPU nodes, never the
  local machine).

## Acceptance

Figures compile with `pdflatex`, match their CSVs, and are publishable
quality; every plotted comparison is at stated, matched accuracy; the report
quantifies both the old approach's weakness (with mechanism) and the new
approach's improvement in time and memory; `042` can cite the figures
directly.

---

## Completion notes (2026-08-17, lead agent)

### Measurement campaign

Three pre-registered jobs (protocols in the script headers, committed as
`ecebdb8` BEFORE submission; all sacct-verified):

| job | node | state | outputs |
|---|---|---|---|
| 13184013 `fm041aG` | H200 m13h-1-2 | COMPLETED 0:0, 1:18:22 | `fm041a_gpu_widen.csv` (165 rows, 0 fails, 0 gate misses), `fm041a_pweep.csv` (24), `fm041a_leafpop.csv` |
| 13184014 `fm041aC` | H200 m13h-1-2 | COMPLETED 0:0, 1:20:58 | `fm041a_gpu_contrast.csv` (35), `fm041a_contrast_leafpop.csv`, `fm041a_gpu_stages.csv` (12), `fm041a_gpu_sigma.csv` (24: 18 ok + 6 global-gate throws recorded as fail rows) |
| 13184015 `fm041aH` | CPU m8-19-14, 1 thread | CANCELLED 0:0 at 1:35:07 by the 2026-08-15 user pause-all — AFTER the final pre-registered row was written; CSV complete | `fm041a_host_widen.csv` (29 rows, 0 fails) |

All three were requeued once by the scheduler (~22:17 MDT 2026-08-15) and
re-ran from scratch in one incarnation each (incremental per-row CSV
rewrite makes this safe; logged). SHA-256 checksums:
`data/adaptive_octree/checksums_041a.sha256`.

### Figures (extend the 024a set; fig12/13 per-figure-data-dir convention;
shared `fmfigstyle` palette; all `pdflatex`-clean 2026-08-17)

- `fig14_adaptive_contrast` — time vs cluster contrast (n=1e6 H200): the
  uniform blow-up (ell=4: 151 ms -> 284 s across c=1->1000) vs the flat
  adaptive curve; 1.67x/2.11x vs best-uniform at c=100/1000.
- `fig15_adaptive_time_vs_n` — best-vs-best per case, H200 + host 1T,
  from the WIDENED sweeps (n 1e4–1e6 GPU; 1e5/1e6 host).
- `fig16_adaptive_stages` — serialized per-stage stacks + shipped
  lifecycle ticks: fat-cell direct(U) vs deep-grid B2M/M2L(V) vs the
  adaptive S2L/M2T cost (the wake-gap mechanism).
- `fig17_adaptive_memory` — device GB vs ell / K_max: ~8x per uniform
  level (ell=7: 54–60 GB) vs the adaptive 3.1–8.3 GB band.
- `fig18_adaptive_accuracy_cost` — P-sweep frontiers + P=4 geometry
  sweeps + the 1e-3 gate rule (comparisons at matched accuracy).
- `fig19_adaptive_leafpop` — leaf-population CCDF at contrast 100/1000:
  the uniform fat tail (>2e4 bodies/leaf) vs the exact K_max=64 cutoff.
- `fig20_adaptive_sigma` — sigma-spread sweep: global-gate throws
  (spread 300 leaves only ell=2) vs the flat gate-passing adaptive curve.

### Headline verdicts (same-job, widened; full table in the report)

n=1e6 best-vs-best adaptive gain: multiscale100 **1.86x (H200) / 2.99x
(host)** plus ~11x less device memory; unitcube 0.96x (parity) both
platforms; wake **0.85x (H200) / 0.90x (host) — uniform ell=7 wins time**
but at 53.9 GB vs adaptive 3.1 GB (17x). Sigma variant: adaptive is the
only machinery valid at spread 300 (2.6x vs the sole surviving ell=2).
The WIDENED sweep overturned two fm040/fm041 endpoint headlines (wake
host 2.42x win -> 0.90x loss vs ell=7; cube GPU 1.11x -> 0.96x) —
recorded prominently in the report; honest negatives (wake both
platforms, all n<=1e5 cases, adaptive K non-monotonicity) are in §4 of
the report and visible in figs 15/16.

### Verification notes

- Every plotted number traces to a CSV of record from a logged job;
  ratios never cross jobs (same-job anchors re-run in each widen job).
- Zero construction failures and zero 1e-3-gate misses among plotted
  rows (gpu widen 165/165 ok; host 29/29; contrast 35/35; sigma ok-rows
  18/18 pass — the 6 fails are the measured global-gate throws).
- `scripts/figures_041a_prepare.jl` (stdlib-only) regenerates every
  figure table + `figures_041a_summary.txt`; re-runnable, errors on
  missing/renamed columns.
- pdflatex compile check: all seven figures compile clean (2026-08-17,
  TeX Live/pdfTeX local); rendered PNGs visually inspected (stacking,
  legends, tick collisions checked; fig16 stacked-tick and fig19
  const-plot issues found and fixed before close-out).
- No production `src/` changes (benchmark/analysis row).

### Priced/deferred items (carry-overs from 040/041 approvals)

- per-n K/depth sweep: **landed** (fig15/fig18 grids).
- graph-engagement instrumentation: **landed** (fig16 data: shipped vs
  serialized within 0.5–4.6% at n=1e6 — not a material factor here).
- frozen-leaf-set refresh: **landed** (epoch fast path 9.0–9.4 ms vs
  50.9–92.9 ms rebuild at n=1e6).
- stage-slab chunking assessment: **deferred with price** (host-only
  memory lever; needs src instrumentation + prototype — maintenance row).
- host double-refresh elimination + host sort unification: still open,
  quantified (host adaptive update 2.1–2.9 s vs 0.44–1.0 s uniform at
  n=1e6); routed to the 042 audit.

## Clear-context approval (2026-08-17 14:58 MDT, fresh re-approval agent)

**APPROVED.** Third clear-context review, scoped per protocol to the
`69075db` fix and residual consistency (reviews of 14:46 and 14:54 MDT
verified jobs, completeness, headlines, figures, compiles, checksums, and
hygiene; not repeated). Verified, re-derived from the CSVs of record:

1. §2 n=1e5 rescope exact: H200 F64 best-uniform wins all cases
   (cube 7.397/8.114 = 1.10x, multiscale 15.192/17.270 = 1.14x,
   wake 7.638/12.331 = 1.61x); host 1T uniform wins cube 2195.50/2538.73
   = 1.16x (ell=4) and wake 1851.64/2026.30 = 1.09x (ell=6);
   multiscale100 is a 2.27x adaptive win (2632.65 ms K=64 vs 5967.37 ms
   ell=5) — all from `fm041a_gpu_widen.csv` / `fm041a_host_widen.csv`.
2. §8 small-n row matches the rescoped §2 and states the host precedence
   of the clustered-fields rule; §4 item 3 is H200-scoped and exact;
   fig15's host panel CSVs carry the same 2632.65/5967.37 points. No
   remaining §2/§4/§8/fig15 contradiction.
3. §7 restatements exact: refresh 5.7x–10.1x row-wise (rebuild/warm
   5.66x cube, 6.52x multiscale, 10.08x wake; warm 9.003–9.396 ms vs
   rebuild 50.944–92.947 ms); adaptive graph rows 0.83/0.98/2.18%;
   full band 0.31–4.79% (min cube ell=7, max multiscale ell=6);
   multiscale ell=5 example 178.077 vs 185.357 ms.
4. `git show --stat 69075db`: surface is report.md + decision log only
   (20 + 82 lines); no figure, CSV, task-file, or 041b/041c changes.
5. Final §2/§8 skim: every number introduced by the edit re-derives from
   the CSVs; §8's wake 1.11–1.18x is the consistent inverse of §2's
   0.90x/0.85x.

No changes required. 041a marked Approved in START_HERE.md (checkbox
line only; the uncommitted 041b/041c working-tree edits left uncommitted
and untouched). Campaign remains PAUSED per user directive; 042 not
launched.

### Open items for 042

Default-selection audit against the report's §8 regime table; S2L device
kernel as the top adaptive tuning target (largest adaptive stage on the
wake); adaptive K non-monotonicity (K=128 anomalies); uniform ell<=8 cap
disposition; sticky-demotion/option-(b), split-veto default, and E2
user-ratification items unchanged.
