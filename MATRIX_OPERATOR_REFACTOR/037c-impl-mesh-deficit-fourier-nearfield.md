# 037c Nearfield Reduction III: Mesh/Fourier Deficit Evaluation (PME-Style)

## Status and Entry Gate

**Staged by user direction `2026-08-13`; not started; conditional.**
Precedes `038` in the roadmap order by that same direction.

Entry gate: `037b` Done and clear-context approved, AND its §6 verdict
recommends this row (with the quantitative basis recorded there), AND
explicit user go. If `037b` recommends against, this row is closed with a
pointer to that evidence and `038` becomes the next row — closing it that
way requires only the user's acknowledgment, not a full campaign.

## Motivation

The gaussianerf field splits exactly as `u = u_singular + u_deficit`, where
the deficit kernel `(g(ρ)-1)`·Biot-Savart is smooth, decays by ~3-4σ, and
is essentially bandlimited at scale `1/σ` — the structure Particle-Mesh
Ewald exploits. Where `037b` evaluates the deficit pairwise within
`rho_c·sigma`, this row evaluates it on a regular grid: spread particle
strengths to a mesh at spacing ~σ/2 (B-spline, order chosen by the error
model), convolve with the deficit kernel (FFT or local stencil — the
kernel's compact support may make a direct separable stencil cheaper than
cuFFT; decide by measurement), interpolate back. Spreading costs ~O(64n)
ops versus thousands of pair-ops per body today; the Jacobian falls out of
spectral/stencil differentiation. The FMM + direct path then carries pure
singular math at expansion-validity geometry only — the σ-adequacy
constraint leaves the tree entirely.

The rectangular-grid machinery from `037` applies directly: the deficit
mesh should be rectangular for elongated domains (this time with no
exact-once subtlety — it is a regular convolution grid, not a tree).

## Objective

Derive the deficit-mesh error model, implement the device-resident
spread/convolve/interpolate path under the existing capacity and
zero-allocation contracts, and report exact measured speedups per case —
against both the shipped baseline and `037b`'s best two-pass result — under
the unchanged accuracy gates.

## Plan

1. **Theory first (scoped derivation, `theory/` + `scripts/` + `data/`):**
   spectral/interpolation error model for the deficit field vs grid spacing
   `h_g/σ` and spreading order, composed with the FMM and (if any) residual
   pair terms into the conservative sum gate; aliasing and domain-padding
   treatment on the rectangular mesh; hessian accuracy via differentiation
   of the interpolant (the J-diagnostic convention still applies, but the
   stretching contraction feeds production — state its error separately).
   Numerical validation of the model against the erf oracle on small cases
   before any implementation.
2. **Implementation (staged like `037`, each stage landing with tests
   green):** host reference path → device spread/convolve/interpolate
   kernels (graph-capture eligible, zero recurring allocation, 023
   counters) → FLOWVPM coupling switch (off by default; production default
   changes only with explicit user approval).
3. **Benchmark and report:** the `037b` case set (cube, wake, rotor wake;
   both scales, both precisions) with same-job anchors for (i) the shipped
   partitioned baseline and (ii) the `037b` winner, so the report can state
   exactly what each approach delivers and what this row adds over `037b`.
   Per-stage profiles (spread/FFT-or-stencil/interp itemized), memory
   footprint, accuracy decomposition per winner, figures per the 024a
   conventions. Levers priced against the overlapped critical path.

## Dependencies and Reading

- `037b` (verdict + case set + decomposition instrument + anchors),
  `037`/`037-implementation-plan.md` (rectangular machinery), `031a` +
  `theory/kernel-splitting-nearfield.md`, `035` Final Report §3, `008e`
  (real-basis kernel derivatives, for the interpolant-differentiation
  hessian), 024a figure conventions.

## Verification Gates

- Same as `037b`: 1e-3 conservative velocity gate on checksummed references
  (including the rotor wake), J diagnostic, 023 counter/allocation
  contracts, scalar no-regression on any FastMultipole `src/` touch,
  same-job anchor discipline for every speedup claim, stated
  warmup/repeat/median policy.
