# 045 Benchmark: Multi-GPU High Score (4/8 GPUs)

## Status and Entry Gate

Staged `2026-08-11`. Blocked by `044` (Done + Approved with its 2-GPU
`>= 75%` efficiency gate passed).

## Objective

Scale the `044` partitioned lifecycle to 4 and 8 H200s (single `m13h` node,
`gpu:h200:8`) and render the final verdict on the user's targets for the
fixed 1M-body literature-`P=4` workload: **`<= 1 ms` per resident step is
the goal; `<= 2 ms` still counts as a win.**

## Scope

1. Scaling ladder: 1/2/4/8 GPUs, per-stage timings, efficiency, comm+orch
   at each width; the `029` accuracy and recurring-cost gates at every row.
2. **Per-GPU launch-floor reduction as a measured lever** if the ladder
   shows the ~0.87 ms control floor binding before the 1 ms goal (expected
   per the `043` model): e.g. graph-node merging, fused per-level exchange,
   fewer captured launches. Only floor work justified by the ladder data;
   each change re-gated on accuracy and single-GPU no-regression.
3. Independent reproduction per the `029` rules (second job, fresh cache,
   same manifest) before any leaderboard row is banked.
4. Final leaderboard + verdict recorded in this file, with addendum notes
   to `019a` and the `029` Closure section per their conventions. If the
   goal is missed, quantify the remaining gap and its binding term.

## Reporting

CSV data under `data/` following the `performance_high_score_1m_1ms`
pattern; a short figures set (TikZ/pgfplots + CSV per the standing
conventions) showing time-vs-GPU-count against the ideal-scaling and
floor-model curves.

## Work Record

(To be filled by the executing agent.)

## Approval Notes

To be filled by a different agent after this task is complete.
