# 009 Implementation Basis And Operator Cache Types

## Objective

Define basis and operator-cache types without changing production translation
call behavior.

## Dependencies

- `001-theory-z-rotation-operators.md`
- `004-theory-axis-swap-conventions.md`
- `005-theory-full-m2l-composition.md`
- `007-theory-coefficient-buffer-layout.md`
- `008h-theory-lamb-helmholtz-accuracy-order.md`
- `008a-milestone-review-theory-005-008.md`
- `008b-implementation-replan.md`
- `008c-implementation-performance-baseline.md`

## Required Reading

- `START_HERE.md`
- Approved dependency task files listed above
- Approved artifacts listed by the dependency task files
- Existing production types and translation call sites

## Artifacts or Production Surface

- Production source files that define basis/cache/scratch types
- Focused tests for type construction and current call behavior

## Deliverables

- Basis type names for the current compressed complex basis
- Shared invariant cache ownership model
- Per-thread scratch ownership model
- Element-type parameterization where practical
- Channel-order metadata for `Val(true)`: store `P_phi` as the requested
  physical order, `P_chi = P_phi + 1`, and `P_active = P_chi` for common
  padded cache/buffer dimensions. `Val(false)` remains single-order `P`.

## Verification

Run focused tests proving existing translation behavior is unchanged and new
types construct correctly. Record commands and result summaries.

## Approval Notes

Approved by reviewing agent (separate clear-context review) on 2026-06-18.

Objective met: the change is additive only — the basis/cache/scratch types are
defined in `src/containers.jl` (per the START_HERE Implementation Code Placement
rule) with exports in `src/FastMultipole.jl`; no translation call site in
`translate.jl`, `fmm.jl`, or `rotate.jl` was modified. Production translation
behavior is unchanged.

Deliverables verified:
- Compressed-complex basis type (`CompressedComplexBasis`); `RealSolidHarmonicBasis`
  present as a deferred placeholder for task `018`.
- Shared invariant cache ownership (`OperatorInvariantCache`) owning private copies
  of `Hs_π2`, `ζs_mag`, `ηs_mag`, `M̃`, `L̃`.
- Per-thread scratch ownership (`OperatorScratch` + `ThreadedOperatorScratch`).
- Element-type parameterization via `TF`.
- LH channel-order metadata: `P_phi = P`, `P_chi = P_phi + 1`, `P_active = P_chi`;
  caches/scratch sized at `P_active`. `Val(false)` stays single-order `P`.

Tests strengthened during review and confirmed sufficient:
- `operator cache support types`: orders, negative-`P` errors, channel/DOF sizing,
  eltype, scratch sizing, per-thread non-aliasing.
- `operator cache construction is side-effect-free`: legacy M2M/M2L/L2L outputs and
  module-global invariant lengths unchanged across cache/scratch construction.
- `cache and scratch drive translations identically to legacy workspace` (added in
  review): feeds cache/scratch fields into the real M2M/M2L/L2L kernels and asserts
  bit-identical results vs. the legacy global workspace for Float32/Float64 ×
  `Val(false)`/`Val(true)`, proving correct drop-in behavior and that the P+1-padded
  LH cache reproduces the P-order result exactly.

Verification commands and result:

```
julia --project=. -e 'using FastMultipole; using FastMultipole.StaticArrays;
using FastMultipole: Branch, initialize_expansion, length_Ts; using Test;
include("test/operator_cache_types_test.jl")'
# operator cache support types ............................... 152/152 pass
# operator cache construction is side-effect-free .............. 5/5  pass
# cache and scratch drive translations identically to legacy ... 4/4  pass
```
