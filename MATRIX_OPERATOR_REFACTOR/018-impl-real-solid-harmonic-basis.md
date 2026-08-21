# 018 Implementation Real Solid Harmonic Basis

## Objective

Add real-basis transforms and parity tests. Keep native real-basis operator
execution as future work after flat buffers and operator APIs stabilize.

## Dependencies

- `008-theory-real-solid-harmonic-transforms.md`
- `008b-implementation-replan.md`
- `008c-implementation-performance-baseline.md`
- `008e-theory-real-basis-kernel-derivatives.md`
- `008h-theory-lamb-helmholtz-accuracy-order.md`
- `017-impl-flat-coefficient-buffers.md`

## Required Reading

- `START_HERE.md`
- Approved dependency task files listed above
- Approved `theory/real-solid-harmonic-transforms.md`
- Approved `theory/real-basis-kernel-derivatives.md`
- Stable operator API and flat-buffer implementation notes

## Artifacts or Production Surface

- Production real-basis transform code
- Tests and benchmarks for complex-basis parity

## Deliverables

- Complex-to-real and real-to-complex transform implementation
- Native real-basis evaluation of the potential, gradient, and gradient
  Jacobian (Hessian) per approved `theory/real-basis-kernel-derivatives.md`,
  matching the production `DerivativesSwitch{PS,GS,HS}` paths
- Native real-basis `Val(true)` execution must preserve the approved channel
  sizing rule from `theory/lamb-helmholtz-accuracy-order.md`:
  `P_chi = P_phi + 1` for M2L/evaluation, with `Val(false)` unchanged.
- Notes recording that production native real-basis operator execution remains
  deferred until after flat buffers and operator APIs stabilize.

## Analytic Performance Note

Native real-basis M2M/M2L/L2L operator execution remains deferred by this item,
but the storage/flop model is now concrete enough to decide whether a later
native-real operator row is worth adding.

Per channel, compressed-complex storage uses

```text
Ncc(P) = 2 * (P + 1)(P + 2) / 2 = (P + 1)(P + 2)
```

real lanes, while the real solid-harmonic basis uses

```text
Nr(P) = (P + 1)^2.
```

The real-basis lane fraction is therefore

```text
Nr(P) / Ncc(P) = (P + 1) / (P + 2).
```

This saves exactly the unphysical imaginary `m = 0` lane in every degree:
about `1/(P + 2)` of the per-channel coefficient storage and of stages whose
arithmetic is lane-proportional. For representative orders this is modest
(`P = 8`: ~10%; `P = 15`: ~6%).

Expected stage-level consequences:

- z rotations: the real basis saves the `m = 0` imaginary lane; all `m > 0`
  modes still use the same real `2x2` rotation block as compressed real/imag
  storage.
- fixed-`m` z translations: the translation blocks are real and apply
  independently to paired lanes; real-basis savings again track the missing
  `m = 0` imaginary lane.
- Lamb-Helmholtz coupling: the same lane reduction applies, but `Val(true)`
  must count ragged channel orders separately (`P_phi = P`, `P_chi = P + 1`).
- materialized/factored y stages: real basis has fewer physical lanes per degree
  (`2n + 1` instead of `2n + 2`), but exposes odd-sized per-degree blocks. The
  current factored path uses complex Fourier-mode matrices internally, so a
  direct real-basis implementation may lose some layout regularity even while
  reducing lanes.

GEMM utilization is therefore the main open performance question. The
compressed-complex layout has very regular two-lane-per-harmonic slabs, while the
real basis has smaller but odd-sized degree blocks. On BLAS/cuBLAS, padding,
batch width, and launch/amortization effects may dominate the small theoretical
lane reduction. A future native-real operator task should be added only after
the integrated batched/GPU path shows that this lane reduction is material for
the winning operator strategy.

## Verification

Run transform round-trip tests, operator parity tests, and relevant benchmarks.
Record commands and result summaries.

Implementation pass verification (2026-06-25):

```text
julia --project=. -e 'using FastMultipole, Test, Random, StaticArrays; include("test/real_solid_harmonic_basis_test.jl")'
  -> real solid harmonic basis (task 018): 1275 passed

julia --project=. MATRIX_OPERATOR_REFACTOR/scripts/real_solid_harmonic_transforms_verify.jl
  -> real_solid_harmonic_transforms_verify: PASS
     max_real_roundtrip_error: 0.0
     max_complex_roundtrip_error: 0.0
     max_projected_m0_imag: 0.0
     max_z_rotation_error: 0.0
     max_native_real_same_point_evaluation_abs_error: 6.938893903907228e-18
     max_m2l_error: 0.0
     max_chain_operator_error: 0.0

julia --project=. -e 'using Pkg; Pkg.test()'
  -> Testing FastMultipole tests passed
```

Notes:

- Native real-basis storage is implemented through `RealSolidHarmonicBasis`
  `OperatorBasisInfo`, `FlatCoefficientBuffer`, and real/complex transform
  helpers.
- Native real-basis local evaluation is implemented for scalar potential,
  gradient, and Hessian/Jacobian, with `Val(true)` preserving
  `P_chi = P_phi + 1`.
- Native real-basis M2M/M2L/L2L operator execution remains deferred. The public
  batched operator entry points now fail clearly if called with real-basis
  buffers rather than compressed-complex buffers.

## Approval Notes

Approved by clear-context review on 2026-06-27.

Review scope followed `START_HERE.md`: this task file, the approved real-basis
theory artifacts, the production real-basis transform/evaluation surface, the
real-basis parity test, and the verification notes. The implementation matches
the item objectives: real-basis storage/indexing and complex/real transforms are
present, native real-basis local evaluation covers potential, gradient, and
Hessian/Jacobian parity, `Val(true)` uses the `P_chi = P_phi + 1` order rule, and
native real-basis M2M/M2L/L2L operator execution remains deferred behind a clear
operator-entry guard.

Verification rerun during review:

```text
julia --project=. -e 'using FastMultipole, Test, Random, StaticArrays; include("test/real_solid_harmonic_basis_test.jl")'
  -> real solid harmonic basis (task 018): 1275 passed

julia --project=. -e 'using Pkg; Pkg.test()'
  -> FastMultipole tests passed
```

No blocking findings.
