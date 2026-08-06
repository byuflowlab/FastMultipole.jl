# 040 Impl: Adaptive Octree Resident Lifecycle (Host)

## Status and Entry Gate

**Proposed follow-on item; not started.**

Entry gate: `038` and `039` complete and approved.

## Objective

Run the full host resident lifecycle (B2M → M2M → V-list M2L → L2L → L2B,
plus U-list direct and the new M2T/S2L for W/X lists) on the `039` adaptive
tree, end to end, at production accuracy.

## Scope and placement

- V-list M2L must flow through the **existing** resident M2L strategies and
  level-scaled operator tables unchanged — the `039` class-format parity
  guarantees this; any required strategy change is a stop-and-discuss event.
- New M2T and S2L operator kernels per `038` item 4 go in
  `translate_batched.jl` (or `evaluate_expansions_batched.jl` if evaluation-
  side placement reads better), with Lamb-Helmholtz (φ+χ) coverage and the
  `008h` χ-order rule.
- M2M/L2L run over the adaptive occupied-ancestor levels using the existing
  edge-group machinery.
- Nearfield: U-list pairs feed the existing direct kernels (singular,
  `RegularizedVortex`, and the `032a` winner once selected), honoring the
  per-cell geometry gate.
- The uniform-depth path remains default and untouched.

## Verification

- Sampled-direct accuracy on the two phase cases and the `038` multi-scale
  case: velocity RMS ≤ 1e-3 gate, Jacobian logged as diagnostic; Float32 and
  Float64; `P=4` and `P=8`.
- Uniform-limit parity: with all leaves forced to one level, lifecycle
  results match the existing hierarchical path to tolerance.
- W/X-path unit tests: M2T and S2L each validated against direct evaluation
  and against the equivalent M2L+L2B / B2M+M2L compositions.
- Zero per-step allocation; `023` counter contract on the host lifecycle.

## Acceptance

End-to-end accuracy gates pass on all three cases; the multi-scale case shows
the cost behavior predicted by the `038` model (bounded leaf population,
per-region depth) with measured host timings recorded; no regression on the
uniform cube/wake cases beyond an agreed tolerance.
