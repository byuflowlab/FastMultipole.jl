# 013c Implementation Factored Rotation Alignment

## Objective

Assemble the explicit factored rotation alignment stages that were split out of
`013b`. The fixed y-swap and inverse-swap primitives now exist in `013b`; this task
composes them with the `010` z-rotation operators into separately callable
alignment and return-alignment stages:

```text
Z_phi -> S -> Z_theta -> S_inv
```

The assembled stages must preserve FastMultipole's current sign and extra-`pi`
convention and be shaped for the global batched-GEMM path used by the later full
M2L operator pipeline.

## Dependencies

- `004-theory-axis-swap-conventions.md`
- `008b-implementation-replan.md`
- `008c-implementation-performance-baseline.md`
- `010-impl-z-rotation-operators.md`
- `012a-milestone-review-impl-009-012.md`
- `013-impl-axis-swap-operators.md`
- `013a-spike-m2l-batching-and-dynamic-p-feasibility.md`
- `013b-impl-fixed-y-swap-primitives.md`

## Required Reading

- `START_HERE.md`
- Approved dependency task files listed above
- Approved `theory/axis-swap-conventions.md`
- `src/rotate_batched.jl`
- Current production rotation and M2L alignment code in `src/rotate.jl` and
  `src/translate.jl`

## Artifacts or Production Surface

- Internal, non-exported factored alignment and return-alignment stage APIs in the
  `_batched` implementation surface.
- No production hot-path replacement; `014` composes the swappable whole-M2L
  interfaces after these stages are validated.

## Deliverables

- Separately callable source alignment stages using `010` z rotations and the
  `013b` fixed y-swap primitives.
- Separately callable return-alignment stages using inverse z rotation and the
  `013b` inverse fixed y-swap primitives.
- Reuse the same fixed y-swap apply operator for forward and return y stages
  whenever the FastMultipole y-rotation convention permits it; do not introduce a
  separate inverse-y apply path unless parity tests show the shared operator is
  invalid for that stage.
- Production-parity tests against current alignment behavior for multipole and
  local coefficients, including axis-aligned and off-axis cases, both `Val(false)`
  and `Val(true)`, and `Float64` / `Float32`.
- Reset/accumulate semantics documented and tested: y stages reset destination
  storage; final inverse z rotation is responsible for accumulation.
- Stage signatures compatible with the cache/scratch conventions from `009`,
  `010`, `013`, and `013b`.

## Verification

Run focused parity tests for the assembled factored stages, then the full package
test suite before handing off to `014`. Record commands and result summaries.

## Implementation Notes

To be filled when implemented.

## Approval Notes

To be filled by a different agent after implementation and verification are
complete.
