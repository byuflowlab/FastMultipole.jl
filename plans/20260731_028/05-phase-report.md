# 028 Phase A.4 — Bound classification, report, user checkpoint

## Bound classification

For every stage ≥5% of the verdict step, attribute compute / memory-bandwidth /
transfer / launch-latency / other, **stating the method per claim**:
- achieved GB/s and FLOP/s vs H200 peaks (HBM3e ~4.8 TB/s; FLOPs from operator
  shapes) for compute/memory attributions,
- kernel-or-window counts × the measured 46-61 µs window overhead for latency,
- CUDA-event timings + host `@allocated` for the host-alloc/GC share.
Reuse the established classifications in `00-context-evidence.md` where still
valid; anything predating 026/027 is stale for M2L structure.

## Report — `MOR/data/feasibility_1m_10ms/report.md`

- End-to-end + per-stage budget tables at all three boundaries, F64 and F32.
- Bottleneck concentration: one stage or distributed?
- Residency answer: must body state stay permanently on GPU? (verdict boundary vs
  transfers-included boundary, quantified).
- Remaining gap + uncertainty at all three boundaries vs the 10 ms target.
- **Prioritized lever list** with expected gain / confidence / risk. Candidates to
  evaluate against the new data: per-level M2L strategy mix (factored/concat at
  thin coarse levels + fused dense at leaf — never benchmarked; class histograms
  are the input), host-alloc/GC elimination in the per-step path, L2B+nearfield
  fusion/tuning, K/window tuning at n=1e6, ell choice, Float32 verdict path,
  preallocated block scan replacing `accumulate!`, stale-tree refresh policy.
- Figures: hold for the final 028 report unless one is needed to make a lever
  case (TikZ/pgfplots+CSV toolchain per 024a).

## Wrap-up

- Fill `MOR/028-performance-feasibility-1m-in-10ms.md` Implementation Notes +
  Verification Notes (commands, job ids, summaries) for Phase A; leave
  Done/Approved unchecked.
- Apply the standing conclusion-review rule: state conclusions, re-check methods/
  results/conclusion for consistency, revise, then report.
- **Stop at the user checkpoint**: present the lever list and await approval.
  Do not begin Phase B.

## Progress

- [x] Bound classification done (report.md §3)
- [x] report.md written — `MOR/data/feasibility_1m_10ms/report.md`
- [x] Task file notes filled (Implementation Notes + Verification Notes; Done/Approved left unchecked)
- [x] User checkpoint reached (lever list presented)
- [x] **Checkpoint decision 2026-07-31: "bank the certain wins first."** Approved scope =
  commit Phase A → one de-risking measurement job (L2B/nearfield split, allocation
  profile, leaf-M2L profiler pass) → lever 3 (host-alloc elimination) → re-measure, then
  decide on lever 1 (nearfield rewrite) with better information. **Lever 1 and lever 2 are
  NOT yet approved for implementation.**
- Notes:
  - Headline: **91.4 ms vs the 10 ms target (9.1×)** at hier12·dense·**F32**·ell=5·**K=1740**,
    n=1e6, P=4. Bottleneck concentrated in two kernels (L2B+nearfield 42%, leaf M2L 26%).
  - Strongest methodological result: leaf M2L is **neither bandwidth- nor compute-bound**,
    proven by a precision A/B at fixed work (22.41 ms F64 → 23.55 ms F32, unchanged) rather
    than by roofline arithmetic. The nearfield A/B (1.37×) separates it as partially
    compute-bound / FP64-rsqrt-limited.
  - Two task-file-named levers demoted by data (per-level strategy mix ≤3 ms; stale-tree
    3.9 ms with an untrustworthy accuracy test); two levers already realized (F32, K=1740);
    the ell/policy/strategy axis is exhausted.
  - Honest gap statement: eliminating **all** non-kernel overhead leaves 65 ms of the two
    dominant kernels, so 10 ms requires ~7× combined from them. Rooflines say that is
    reachable in principle (~8-12 ms plausible optimized budget) but only at near-peak
    kernel quality — not represented as achievable until lever 1 is executed.
  - Corrections applied during the phase: the pilot's ell=6 extrapolation (~220 ms) was
    wrong by 2.5× vs the measured 546.7 ms and was discarded; the apparent 2× "speedup"
    vs the 027 record at n=2e5 was traced to a **fixed-`bounds` tree-geometry difference**
    in the 028 harness, not a performance gain, and no such claim is made.
