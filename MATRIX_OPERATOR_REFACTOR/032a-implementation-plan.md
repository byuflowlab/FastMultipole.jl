# 032a Execution Plan (staged, with user checkpoints)

Drafted `2026-08-06` after the `032` approval unlocked this row; reading gate
recorded in the task file. Stages mirror the `032` pattern: each ends with
tests plus a user checkpoint. Placement rules apply (types in `containers.jl`,
`_batched`/`_cuda` files, CUDA optional).

## What 032 already ships that this row builds on

- `RegularizedVortex` (candidate 1) — H200-validated baseline with the erf-free
  `g`/`h` evaluation (§3 series to ρ=2, §6.2 one-`exp` outer form, measured
  switch ρ_c=2) and the fused U+J pair math.
- The adequacy gate (`_direct_kernel_geometry_gate!`) asserting
  `g_min·h_leaf > ρ_t·σ_max` per step — exactly the §5.1 reach requirement that
  candidates 1 and 2 share.
- Supported rigid radii extended to q²≤20, making adequate geometries
  constructible at overlap 2 (q=16 at n=1e5/ell=3 and n=1e6/ell=4, measured
  5.18e-4 / 9.49e-4).
- The `direct_kernel` functor plumbing (compile-time, options-carried), the
  13-row hessian output, and the device-resident vortex validation harness
  (`cuda_032_validation.jl`).

## Stage A — host partitioned kernel + geometry assertions (candidate 2)

1. `PartitionedVortex{TF}(; sigma_row, rho_t)` in `containers.jl`: pair kernel
   branches on `ρ = r/σ_src ≤ ρ_t` → the shipped stable regularized U/J;
   otherwise singular vortex U/J (`a=-3/r², b=-1/(4πr³)`). Host loop first
   (branch cost irrelevant on CPU with good prediction; §6.3 is a GPU issue).
2. Geometry correctness (deliverable 5): construction-time assertion that no
   cutoff pair is assigned to M2L (the adequacy gate already enforces the gap
   inequality; add the §5 source-directed AABB predicate as a test-side
   exact-once enumeration mirroring `validate_031a_kernel_split.jl` §5 against
   the production route construction), plus source-cell `max(σ_src)` sizing.
3. Tests: partitioned-vs-regularized-everywhere parity inside the cutoff and
   partitioned-vs-singular outside, F64/F32, P=4 and P=8; exact-once coverage
   on a deterministic grid; adequacy rejection paths.

**Checkpoint A** (report parity + coverage results).

## Stage B — host two-pass additive correction (candidate 3)

1. Deficit kernel `TwoPassDeficit`: `ΔU=-ḡC`, `Δa=(ρg'+3ḡ)/r²`,
   `Δb=ḡ/(4πr³)`; ḡ via the shipped §6.2 evaluation (`ḡ = 1-g` below ρ_c is
   NOT used — pass 2 only sees ρ ∈ (ρ_c, ρ_t] in the hybrid, where the §6.2
   outer form gives ḡ directly to absolute tolerance).
2. Pass-2 traversal: a second direct route set reaching ρ_t·σ_max (the
   already-extended radii supply the offset ball; reuse the direct-route
   builder with the larger near set), evaluated after the unmodified pass 1.
   Hybrid: pass-1 kernel is `PartitionedVortex` with cutoff ρ_c=2 (stage A
   machinery reused); plain variant: F64 accumulation of pass 1 direct +
   pass 2 (far field precision unchanged).
3. Tests: two-pass total vs regularized-everywhere reference at working
   precision (F64 plain; F32 hybrid), conditioning guard test near small ρ
   (hybrid must hold ~1e-7, mirroring `two_pass_conditioning.csv`), exact
   pass-2 reach coverage, P=4 included.

**Checkpoint B.**

## Stage C — CUDA mirrors + the binned pair stream (§6.3, required pre-A/B)

The §6.3 finding governs: an unbinned split kernel is ~1.56x SLOWER than the
032 baseline at ell≥4 (only 19/389 leaf classes are branch-homogeneous), so a
distance-binned or sorted stream is a first-order requirement. Candidate
mechanisms, decided by measurement in this stage (record all attempts):

- (a) within-cell body sort by Morton sub-key (spatial coherence → lanes see
  correlated ρ; cheapest, approximate homogeneity — measure achieved warp
  homogeneity first);
- (b) per-(route-class) two-phase evaluation: kernel pass over mixed classes
  computes the predicate once into a bitmask/compacted index buffer
  (construction-sized scratch, zero per-step allocation), then homogeneous
  warps stream each side;
- (c) class-level pre-split only for the pure classes (19 inside + pure-outside
  set) with (a) or (b) for mixed classes.

Report achieved warp homogeneity alongside timings (deliverable 3). Two-pass
pass 1 is branch-free (exempt); its pass 2 uses the same mechanism.

**Checkpoint C** (mechanism choice + homogeneity data).

## Stage D — H200 A/B ladder + default selection

- Fixed adequate geometry per case (NOT per-strategy optima — opposite depth
  trends, λ* table): cube n≈1e5 (q=16, ell=3 measured adequate) and helical
  wake cylinder n≈1e5, both overlap 2, both admissible precisions.
- Ladder per the task file: add n≈1e3/1e6 only if within 10% or modeled
  crossover; full 024b grid only if sentinels reverse the winner.
- ρ_t lever (deliverable 4): default per-pair radii (U 4.211 / J 4.789 at
  ε=1e-3); measure the RMS radii (ρ_t(J)=4.252, 389→275 classes) on BOTH
  cases; adopt only if sampled-direct confirms on both.
- Gates: velocity RMS ≤1e-3 winner eligibility, J logged as diagnostic,
  023 counters, zero recurring allocation, scalar 028/030 no-regression.

**Checkpoint D**: results table to the user; default changes only after user
approval (deliverable 7; per-regime defaults allowed if measured).

## Risks / open questions

- The pass-2 route set needs the direct builder to run with a second, larger
  near set alongside the primary — verify the route-buffer capacity plumbing
  tolerates two direct lists (else build pass-2 routes as an offset-class
  complement: classes in the ρ_t ball but outside the primary near set).
- Wake-cylinder geometry at overlap 2 concentrates σ_max; the §5.2 depth
  ceiling uses measured σ_max (already the shipped gate's form).
- FP16-WMMA far field stays untouched in every candidate; only direct/pass-2
  kernels change.
