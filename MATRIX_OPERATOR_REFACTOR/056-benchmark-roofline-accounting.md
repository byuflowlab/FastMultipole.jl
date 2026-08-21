# 056 Benchmark: Per-Stage Roofline Accounting

## Status and entry gate

**Staged `2026-08-20` (user direction). Not started.**

Entry gate: `054` and `055` complete and approved. Benchmark/analysis row:
`scripts/`, `data/`, figures only — no production changes.

## Motivation

The Peak Efficiency Phase opened on the claim that the FMM sits ~10× below
the brute-force kernel's 38%-of-peak; `054`/`055` spent levers against
that gap. This row closes the phase's measurement loop: an honest
per-stage roofline accounting of what was gained, what fraction of peak
each stage now achieves, and what remains — the quantitative hand-off the
Multi-GPU phase (`043`) derives its targets against.

## Objective

Per-stage roofline accounting on H200; the "fraction of peak" scoreboard
before/after `054`–`055`; a re-run of the `052` driver recording the
end-to-end delta; and a written statement of what remains (with estimated
ceilings) feeding `043`.

## Method

### Stage 1 — per-stage roofline

For each lifecycle stage (refresh, upward B2M/M2M, M2L by strategy, L2L,
L2B, M2T/S2L where adaptive, nearfield UJ, ζ/SFS, transfers/orchestration):
measured time, achieved flops and bytes, arithmetic intensity, and the
attainable bound (H200 FMA peak / HBM bandwidth / launch+control floors) at
n = 1e5, 2.1e5 (the p018 point), 1e6, plus the 018 operating point through
the production driver. Instrument via existing counters + Nsight/CUPTI
where needed; floors (~50 µs/window, ~0.87 ms control) accounted
explicitly, not smeared into stage efficiency.

### Stage 2 — scoreboard

The fraction-of-peak table before/after `054`–`055` (baseline = the `053`
-era build), per stage and end-to-end, against the brute-force 38%-of-peak
anchor and the ~10× gap claim — confirm, revise, or refute the opening
analysis with the measurements.

### Stage 3 — end-to-end delta

Re-run the `052` driver configuration and record the end-to-end s/step
delta attributable to the phase (same case, same hardware, job-ID'd A/B
against the `052` result).

### Stage 4 — what remains

Written statement of the remaining gaps with estimated ceilings per stage
(what a perfect implementation could still recover), explicitly feeding
`043`'s multi-GPU targets (the `<= 1 ms` goal's floor arithmetic).

## Gates and verdict

- Every scoreboard number traceable to a job ID and an attainable-bound
  calculation shown in the report.
- Verdict: the phase's measured end-to-end gain, the revised
  fraction-of-peak picture, and the ranked remaining-ceiling list handed
  to `043`/`057`.

## Artifacts

- `scripts/fm056_*` instrumentation + analysis drivers.
- `data/roofline_accounting/` — per-stage CSVs, scoreboard, `report.md`;
  figures per the standing TikZ/pgfplots + CSV conventions.

## Verification

- Achieved-flops/bytes cross-checked against analytic per-stage work
  counts (pairs, coefficients, GEMM dims) — not just profiler output;
  before/after runs same-job or same-node/same-build with job IDs.

## Recorded context (2026-08-20 staging)

**The gap being accounted:** FMM effective efficiency ~10× below the
brute-force kernel's measured 38% of FP32 FMA peak (041k opt: 3.3e11
pairs/s); losses attributed (pre-phase) to M2L scatter/gather, small
memory-bound GEMMs, ~50 µs/window launch floors (`027`), ~0.87 ms per-GPU
control floor (`028`/`029`), refresh — NOT P2P.

**Anchors:** FMM step (041a fig15, unitcube GPU best-uniform): 1.31 ms
@1e4, 7.40 @1e5, 92.3 @1e6; brute-force crossover ≈ 4–5.5e3. p018 point:
n = 210,056. 018 operating point: ~181k–342k particles + 36,752 panels,
budget ≤3.3 s/step.

**Cost model:** T(K) ≈ αKN + βN/K + floors; GPU optimum K=256 (`027`);
`054`/`055` promotions each came with a leaf-size retune — the scoreboard
must use each build's own tuned point. Price against the overlapped
critical path, not isolated stage sums (standing rule).

**Multi-GPU hand-off:** `043`'s kill-switch gate is a modeled 8-GPU step
`<= 2 ms` with a path to `<= 1 ms`; the recorded bound (perfect 8-way
~0.58 ms + ~0.87 ms control floor + ~0.2 ms comm ≈ 1.0–1.2 ms) makes the
post-055 control-floor number this row reports the load-bearing input.
