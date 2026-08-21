# 014 Implementation Full M2L Operator Pipeline

## Objective

Compose the full M2L operator pipeline and test it against current production
behavior.

## Dependencies

- `005-theory-full-m2l-composition.md`
- `008b-implementation-replan.md`
- `008c-implementation-performance-baseline.md`
- `008d-theory-dynamic-p-error-m2l-integration.md`
- `008h-theory-lamb-helmholtz-accuracy-order.md`
- `010-impl-z-rotation-operators.md`
- `011-impl-m2l-z-translation-blocks.md`
- `012-impl-lamb-helmholtz-operators.md`
- `013-impl-axis-swap-operators.md`
- `013b-impl-fixed-y-swap-primitives.md`
- `013c-impl-factored-rotation-alignment.md`

## Required Reading

- `START_HERE.md`
- Approved dependency task files listed above
- Approved `theory/full-m2l-composition.md`
- Approved `theory/dynamic-p-error-m2l-integration.md`
- Current production M2L call sites and tests

## Artifacts or Production Surface

- Production full M2L operator pipeline code
- Tests comparing full operator output with current M2L behavior

## Deliverables

- End-to-end explicit M2L operator composition.
- Swappable whole-M2L operator interfaces:
  - `MaterializedYRotationM2L`: uses `Ts(theta)` materialized by `013`.
  - `FactoredRotationM2L`: uses explicit `Z/S/Z/S` rotation alignment stages from
    `013c`. **Plain-H amendment (`2026-06-23`):** the `S`/`S_inv` swaps are the cached
    fixed per-degree mode matrices `V_n`/`U_n` (`OperatorInvariantCache.y_mult_U/V`,
    `y_loc_U/V`), applied via the `013c` batch entry points
    (`multipole_/local_factored_source_alignment_batch!` and `..._return_...`). They are
    **not** the ζ-dressed `013b` `T_y_pos90/neg90` primitives, which a `013c` spike
    proved cannot reproduce `R_y(theta)` when composed with a `Z_theta`
    (`(ζS)Z(ζS⁻¹) ≠ ζ(SZS⁻¹)`; ζ does not commute through the swap). Do not route the
    factored path through the `013b` primitives.
- The factored path follows the 013c reset/accumulate split (y stages reset their
  destination; the final inverse `Z_phi` accumulates) and reuses the cached fixed modes
  for forward and return stages.
- Cache and scratch usage integrated with earlier implementation tasks
- Both M2L variants share the same `011` z-axis M2L blocks, `012`
  Lamb-Helmholtz coupling, common cache/scratch conventions, and common parity
  tests against production M2L.
- Stage API that can compose either near-term variant without rewriting the
  batching layer. The folded no-`Ts` y-rotation path is not a `014` deliverable
  except as temporary debug scaffolding for validating explicit stages.
- Z-translation reuse should distinguish shared direction from shared distance:
  pre/post scaling is reusable only when the physical `r` / offset norm / level
  key matches.
- `Val(true)` M2L policy uses `P_phi` as the requested physical order and
  carries `chi` at `P_chi = P_phi + 1` through the M2L/evaluation pipeline.
  The constant-`P` stencil should use `B_phi(P_phi)` and
  `B_chi(P_phi + 1)` per `theory/lamb-helmholtz-accuracy-order.md`.
- Compatibility path that preserves current production behavior. Per the `008b`
  re-plan, this first pass is **side-by-side, parity-only**: the explicit M2L
  operator pipeline is validated against production but does **not** replace the
  production `multipole_to_local!` / `multipole_to_local_II!` internals.
  Production hot-path replacement is a later, explicitly scoped step.

## Verification

Run parity tests across representative source-target offsets and expansion
orders. Record commands and result summaries.

### Implementation summary

- New tag types and batched scratch in `src/containers.jl`:
  `AbstractM2LOperator`, `MaterializedYRotationM2L`, `FactoredRotationM2L`,
  `M2LOperatorScratch{TF,B,LH}` (embeds an `OperatorScratch` for all 1D per-column
  buffers, adds exactly two `[2,2,nh,B_max]` working buffers + the distance-rebuilt
  `blocks`/`lh_A`/`lh_B`). Exported from `src/FastMultipole.jl`.
- New driver `m2l_operator_batch!(op, targets, sources, phis, thetas, rs,
  invariant_cache, scratch, lamb_helmholtz)` plus the variant-dispatched
  `_m2l_source_alignment!` / `_m2l_return_alignment!` stages in
  `src/translate_batched.jl`. Stage 2 (z-translation via task-011 `apply_m2l_z!`
  and task-012 `apply_lamb_helmholtz_local!`) and the cache/scratch conventions are
  shared by both variants; only the y-alignment (stages 1/2 forward, 5/6 return)
  differs. `FactoredRotationM2L` uses the Plain-H mode matrices `y_mult_U/V`,
  `y_loc_U/V` (task 013c), never the 013b `T_y_*90` primitives.
- **`P_phi`/`P_chi` order policy (`Val(true)`).** The pipeline runs uniformly at
  `P_active = P_phi + 1`, but φ is physical only through `P_phi`: `_zero_phi_padding!`
  zeroes the φ rows of degree `P_phi+1 … P_active` (a) in the aligned buffer before
  the z-translation (so the M2L gather's `np = P_active` φ term cannot leak into
  physical φ_n) and (b) in the mid buffer before the return rotation (so the
  nonphysical φ_{P_active} row produced by the translate is not emitted). χ is
  carried at `P_active` throughout (the 008h accuracy benefit). Rotations stay at
  `P_active` and are correct because they are block-diagonal in degree, so φ blocks
  `n ≤ P_phi` are independent of the padding degree. Net contract: φ_n (n ≤ P_phi)
  = production-at-`P_active` on a φ-zero-padded source; χ_n (n ≤ P_active) =
  production-at-`P_active`; φ_{P_active} output = 0. The `_zero_phi_padding!` calls
  are no-ops for `Val(false)` (`P_active == P_phi`).
- The two-buffer lifetime (source-align → `work_a`; z-translate+LH → `work_b`;
  return-align reads `work_b`, accumulates `targets`, uses `work_a` as tmp) is the
  minimal footprint: `apply_m2l_z!` is a gather (out≠in) and the factored entry
  points require source≠tmp, so two batch buffers are the floor.

### Commands and results

New parity test `test/m2l_operator_test.jl` (wired into `test/runtests.jl` after
`translate_batched_test.jl`). Standalone run:

```
julia --project=. -e 'using FastMultipole; using FastMultipole.StaticArrays;
  using Random, Test; include("test/m2l_operator_test.jl")'
# Test Summary: M2L operator pipeline (task 014) | 9908 pass / 9908 total
```

Coverage (both variants, five offset classes — +z/+x/+y axes and two general
diagonals — as a multi-column batch and a single-column `B=1` batch):

- **`Val(false)`, `P ∈ {2,4,6,8}`**: every harmonic of both lanes compared to
  production `multipole_to_local!` at `P`.
- **`Val(true)`, `P_phi ∈ {2,4,6,8}` (`P_active = P_phi+1`)**: source carries a
  **nonzero φ sentinel** in the padding rows (degree `P_phi+1`). Reference =
  production at `P_active` on the same source with φ padding zeroed. Asserts:
  φ rows `n ≤ P_phi` match the reference (sentinel ignored ⇒ no leak into physical
  φ); χ rows `n ≤ P_active` match (χ padding contributes via translation + the LH
  upper-neighbor); φ rows of degree `P_phi+1` are exactly 0 (no nonphysical
  output). This proves the `P_phi`/`P_chi` split, addressing the review.
- **Cross-variant** testset: materialized and factored agree (incl. padding) for
  `LH ∈ {false,true}`, `P ∈ {3,6}`.

Tolerances: `rtol = 1e-7` keeps entries with magnitude tight; `atol = 1e-6` is the
floor for structurally-zero entries (e.g. m=0 imaginary χ), where the factored
path yields exactly 0 while production accumulates ~1e-8 rotation round-off at
high `P`.

Note: sources are *physical* (m=0 imaginary part zero), which every real
multipole/local expansion is. `FactoredRotationM2L`'s rank-1 mode decomposition
(task 013c) reproduces production on this physical subspace;
`MaterializedYRotationM2L` is a full linear operator and is exact for any input.

Full suite regression check:

```
julia --project=. -e 'using Pkg; Pkg.test()'
# Testing FastMultipole tests passed   (exit 0; m2l_operator_test included in runtests.jl)
```

## Approval Notes

Clear-context review 2026-06-24 (different agent): **APPROVED.**

Reviewed `START_HERE.md`, this task file, and the listed production/test surface
for task `014` (`src/containers.jl`, `src/FastMultipole.jl`,
`src/rotate_batched.jl`, `src/translate_batched.jl`, `test/runtests.jl`, and
`test/m2l_operator_test.jl`). The implementation matches the stated objective:
both `MaterializedYRotationM2L` and `FactoredRotationM2L` compose the shared
z-translation and Lamb-Helmholtz stages, the factored path uses the Plain-H
`y_mult_U/V` and `y_loc_U/V` mode matrices rather than the `013b` dressed
`T_y_*90` primitives, and the `Val(true)` `P_phi`/`P_chi` policy is covered by
the explicit φ-padding zeroing before z-translation and before return alignment.

No blocking correctness, performance, robustness, or scope issues found. One
stale inline comment in `src/translate_batched.jl` still described the
`P_phi`/`P_chi` split as deferred; it was corrected during review to match the
implemented behavior.

Verification run during review:

```text
julia --project=. -e 'using FastMultipole; using FastMultipole.StaticArrays; using Random, Test; include("test/m2l_operator_test.jl")'
  M2L operator pipeline (task 014): 9908 passed
```
