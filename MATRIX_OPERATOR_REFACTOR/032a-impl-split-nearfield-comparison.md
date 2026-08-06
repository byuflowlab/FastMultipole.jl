# 032a Implementation: Partitioned Nearfield Comparison

## Status and Entry Gate

**Added by user direction on `2026-08-05`; revised after `031a` replaced the
rejected singular-minus-correction design, then extended the same day when the
user restored a two-pass *additive* correction as a third candidate (`031a`
§6.1).** Not started.

Entry gate: `031a` and `032` must both be Done and clear-context approved.
Kernel scope: `gaussianerf` only.

## Objective

Implement the `031a` partitioned regularized/singular nearfield **and its
two-pass additive-correction alternative** on the resident lifecycle, and select
the resident vortex-nearfield default by H200 measurement against `032`'s
regularized-everywhere baseline.

## Deliverables

1. **Partitioned nearfield:** retain the singular FMM far field. Ensure the
   direct geometry contains every pair with `r/σ_src ≤ ρ_t` using the
   source-directed predicate `d_min(B_t,B_s) ≤ ρ_t max(σ_src)`; evaluate those
   pairs once with the cancellation-safe regularized U/J formulas, and use the
   singular U/J kernel for remaining direct pairs. Implement the `ρ≤0.5`
   Horner series for `g` and `h=ρg'-3g` with six terms in Float32 and ten
   in Float64. No runtime `erfc`.
2. **Two-pass additive correction (third candidate).** Restored by user
   direction `2026-08-05` after this file was written; derived in `031a` §6.1
   and carried in the `START_HERE` row summary, which governs. Leave the FMM
   entirely unmodified — singular far field *and* singular direct — and add a
   second pass carrying only the deficit `ΔU = -ḡC`,
   `Δa = (ρg'+3ḡ)/r²`, `Δb = ḡ/(4πr³)`. It touches no `025` routing invariant,
   but its subtraction lands in the target accumulator across two kernels, so it
   requires either **Float64 accumulation of the singular direct term and its
   correction** (the far field may stay FP16-WMMA/Float32) or the **`ρ_c = 2`
   hybrid** that evaluates `ρ ≤ ρ_c` with the stable form inside pass 1. Pass 2
   must reach `ρ_t` on its own (389 classes at `n=1e6, ℓ=5`); pass 1 needs no
   enlargement. Two-pass and partitioning have **opposite depth trends**
   (`λ* = -0.065/-0.119/0.548/2.434` at `ℓ=3/4/5/6`), so the A/B must run at a
   fixed adequate geometry, not at each strategy's own optimum.
3. **Settle the pair-stream ordering before the A/B (`031a` §6.3).** Both split
   strategies assume a branch-free stream. At the shipped `ℓ=5` operating point
   the regularized fraction is `f = 0.310`, so the probability a 32-lane warp is
   branch-homogeneous is `6.9e-6` and an unbinned kernel pays both paths on
   essentially every warp — measured **1.56x slower than `032`'s
   regularized-everywhere baseline**. Cell-level classification does not rescue
   it: only 19 of the 389 direct classes at `ℓ=5` lie entirely inside the
   cutoff, 370 are mixed. A distance-**binned or sorted** pair stream is
   therefore a first-order implementation requirement, not a tuning detail, and
   must be in place before any A/B number is recorded; two-pass's pass 1 is
   uniformly singular and exempt, but its pass 2 is not. Report the achieved
   warp homogeneity alongside the timings.
4. **`ρ_t` is a measurable lever (`031a` §6.4).** Expensive pairs scale as
   `ρ_t³`, and the §4 radii are per-pair worst case while the phase gate is a
   sampled RMS. The RMS-solved radii (`ρ_t(J) = 4.252` against `4.789` at
   `ε=1e-3`) cut expensive pairs by 30% and the leaf near set from 389 to 275
   classes, for every candidate at once. Keep the §4 per-pair radii as the
   default; adopt the RMS radii only if this row's sampled-direct measurement
   confirms them on **both** test cases.
5. **Geometry correctness:** construction-time assertion that no cutoff pair
   is assigned to M2L, exact-once n-body coverage tests, source-cell
   `max(σ_src)` sizing, and rejection or conservative enlargement when the
   selected near geometry cannot cover the cutoff.
6. **Profile-triggered A/B measurement on H200:** partitioned replacement and
   the two-pass additive correction versus `032`'s single-pass
   `RegularizedVortex`, initially at representative
   `n≈1e5` for the cube and the helical wake cylinder (both overlap 2), in both admissible
   precisions. Add `n≈1e3` and `n≈1e6` only if the representative strategies
   are within 10% or the cost model predicts a crossover. Run the full
   seven-point `024b` grid only if the sentinel cases reverse the winner.
   Check sampled relative velocity RMS error `≤1e-3` for winner eligibility;
   log sampled Jacobian RMS error for every configuration as a diagnostic.
   Report branch divergence, pair counts, route overhead, and steady-state
   U+J timing.
7. **Default selection:** report results to the user before changing a
   default. Per-regime defaults are allowed if justified by measurement.
8. **Contract gates:** parity including `P=4`, `023` transfer counters,
   zero recurring allocation, and no regression in the scalar `028`/`030`
   path.

## Dependencies and Reading

- `031a-theory-kernel-splitting-nearfield.md`, Done and approved.
- `032-impl-generalized-device-interface.md`, Done and approved.
- `START_HERE.md`, `theory/kernel-splitting-nearfield.md`, the `031a`
  validation results, and `integration-api-spec.md` §5.

## Work Record

### Reading gate (2026-08-06)

Completed by the executing agent (Claude Fable 5, same session that closed
`032`): `START_HERE.md` (incl. the Integration Phase preamble and both 032a
amendments), `031a-theory-kernel-splitting-nearfield.md` (status, work record,
review-correction history), `theory/kernel-splitting-nearfield.md` in full
(all of §§1–8 incl. the §6.1 two-pass operator/conditioning, §6.3 divergence
model, §6.4 RMS radii, §7 validation results), the `031a` validation data
summaries (`partitioned_replacement.csv`, `geometry_coverage.csv`,
`two_pass_conditioning.csv` figures as quoted in theory §7), and
`integration-api-spec.md` §5 including the three-candidate amendment.

Execution plan: `032a-implementation-plan.md` (this directory) — four stages
(host partitioned → host two-pass → CUDA + binned stream → H200 A/B ladder),
each with a user checkpoint; §6.3 binned-stream mechanism selection is an
explicit measured decision before any A/B number is recorded.

## Placement and Reporting

- Follow the `_batched`/`*_cuda.jl` placement rules; types stay in
  `containers.jl` and CUDA remains optional.
- FastMultipole commits in this repository only.
- Record measured tables and the user-approved default decision here.
