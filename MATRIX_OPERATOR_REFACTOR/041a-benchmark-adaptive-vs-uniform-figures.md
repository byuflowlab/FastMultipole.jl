# 041a Benchmark Report: Adaptive vs Uniform Grid, Publishable Figures

## Status and Entry Gate

**Proposed follow-on item; not started.**

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
