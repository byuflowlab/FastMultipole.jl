# 057 Milestone Review: Peak Efficiency Phase

## Status and entry gate

**Staged `2026-08-20` (user direction). Not started.**

Entry gate: `054`–`056` complete and approved. Together with `053`, gates
the Multi-GPU Scaling Phase (`043`).

## Motivation

Standing milestone-review convention: close the phase with an audit before
the Multi-GPU phase derives against the single-GPU result. This review's
deliverable is the **single-GPU baseline hand-off**: the exact build,
tuned parameters, and measured per-stage numbers that `043`'s partitioned
decomposition and cost model will treat as ground truth.

## Objective

A signed review of `054`–`056` and the recorded single-GPU baseline handed
to the Multi-GPU phase.

## Method

Review checklist:

1. **Lever audit** — every `054`/`055` lever's promote/reject decision
   backed by same-job A/B numbers; promoted defaults have explicit user
   approval; rejected levers' numbers recorded for posterity.
2. **Contract audit** — exact-once, counters, zero-allocation, graph
   capture, capacity, P=4 both precisions — verified on the final build.
3. **Retuning audit** — each promoted kernel cheapening was followed by a
   leaf-size/autotune re-sweep; the shipped tuned point is the measured
   optimum, not a stale K.
4. **Scoreboard acceptance** — `056`'s roofline accounting is internally
   consistent (achieved ≤ attainable everywhere; floors explicit) and the
   end-to-end delta vs the `053`-era build is reproduced.
5. **Baseline hand-off** — record for `043`: commit hash, tuned
   parameters per case, per-stage times at n = 1e5/2.1e5/1e6 + the 018
   point, the post-055 per-GPU control-floor number, and `056`'s
   remaining-ceiling list.

## Gates and verdict

- Checklist answered with evidence pointers; verdict = phase closed (with
  the baseline hand-off recorded) or named remediation before `043`
  activates.

## Artifacts

- Review report + the baseline hand-off table appended to this doc.

## Verification

- Spot-reproduce at least one promoted lever's A/B and the end-to-end
  driver number from the recorded build before signing.

## Recorded context (2026-08-20 staging)

**What the Multi-GPU phase consumes:** `043`'s kill-switch acceptance gate
is a modeled 8-GPU step `<= 2 ms` with an identified path to `<= 1 ms` for
the fixed 1M-body literature-P=4 workload on ≤8 H200s; the recorded
feasibility bound (perfect 8-way compute ~0.58 ms + ~0.87 ms n-independent
per-GPU control floor + ~0.2 ms comm ≈ 1.0–1.2 ms) means the control-floor
number after `055`'s fusion/graph work is the decisive hand-off input.

**Phase-entry claims to close out:** ~10× FMM-vs-brute-force efficiency
gap; losses in M2L scatter/gather, small GEMMs, ~50 µs/window and ~0.87 ms
floors, refresh; P2P already optimized (partitioned + `037f` fp32 + `041e`
CSR + `054` levers).

**Standing rules that apply to the audit:** default changes need explicit
user approval; leaf-size retune after every kernel cheapening (T(K) ≈ αKN +
βN/K + floors, K*=256 pre-phase); critical-path pricing; P=4 test rule.
