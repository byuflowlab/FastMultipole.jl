# 008e Theory Real-Basis Kernel Derivatives

## Objective

Derive evaluation of the local expansion in the real solid harmonic basis for the
full `1/r` kernel derivative set used in production: the scalar potential
`phi = 1/r`, the gradient `grad phi`, and the Jacobian of the gradient (the
Hessian of `phi`). The approved real-basis theory (task `008`) derives only the
scalar potential evaluation. This task extends it to gradient and Jacobian so the
real basis reaches feature parity with the production `DerivativesSwitch` paths
before native real-basis execution is implemented.

This task was added by the `008b` Implementation Re-Plan. It is a Theory Phase
task: it blocks every Implementation task under the standard hard phase gate.

## Dependencies

- `007-theory-coefficient-buffer-layout.md`
- `008-theory-real-solid-harmonic-transforms.md`
- `008b-implementation-replan.md`

## Required Reading

- `START_HERE.md`
- Approved dependency task files listed above
- Approved artifacts `theory/coefficient-buffer-layout.md` and
  `theory/real-solid-harmonic-transforms.md`
- Current production evaluation code (read-only, no edits):
  - `src/evaluate_expansions.jl`: local-expansion evaluation and the
    potential/gradient/Hessian write paths
  - `src/containers.jl`: `DerivativesSwitch{PS,GS,HS}` and the body-buffer rows
    `4` (scalar potential), `5:7` (gradient), `8:16` (Hessian / gradient
    Jacobian)
  - `src/harmonics.jl`: regular/irregular solid-harmonic values and their
    derivative recurrences

## Artifacts or Production Surface

This is a Theory Phase task. It must not modify production code under `src/`.

Artifacts:

- `theory/real-basis-kernel-derivatives.md` — derivation and evaluation spec
- `scripts/real_basis_kernel_derivatives_verify.jl` — verification script
- `data/real_basis_kernel_derivatives/verification_summary.md` — generated
  summary

## Deliverables

- Real-basis evaluation formula for the scalar potential consistent with the
  approved task `008` scalar evaluation (restated as the `PS` path baseline).
- Real-basis evaluation formula for the gradient `grad phi` (the `GS` path),
  expressed against the real-basis mode ordering
  `mode_index(n, 0) = n^2 + 1`, `mode_index(n, m, cos) = n^2 + 2m`,
  `mode_index(n, m, sin) = n^2 + 2m + 1`.
- Real-basis evaluation formula for the Jacobian of the gradient — the
  symmetric Hessian of `phi` — covering all nine production Hessian buffer
  entries (rows `8:16`), with the symmetry/trace structure stated.
- Statement that each derivative path is the real-basis image of the approved
  compressed complex evaluation under the `008` transforms `T_c2r` / `T_r2c`,
  introducing no new normalization or rotation convention.
- The production-normalization and analytic-`1/r` comparison conventions carried
  through to gradient and Hessian: production scalar potential is `u_raw/(4*pi)`,
  and analytic comparison applies the approved `-4*pi` factor; the gradient and
  Hessian inherit the same scaling.
- Notes on how the `DerivativesSwitch{PS,GS,HS}` compile-time switches map onto
  the real-basis evaluation paths.

## Verification

`scripts/real_basis_kernel_derivatives_verify.jl` checks, for `P = 0, 1, 3, 6, 9`:

- real-basis scalar potential matches the approved compressed complex evaluation
  (and the task `008` scalar formula);
- real-basis gradient matches production complex-basis gradient evaluation
  (`GS = true`) at multiple evaluation points;
- real-basis gradient Jacobian (Hessian) matches production complex-basis
  Hessian evaluation (`HS = true`) at multiple evaluation points, including the
  expected symmetry;
- for the approved unit point-mass example, the real-basis potential, gradient,
  and Jacobian converge to the analytic `1/r`, `grad(1/r)`, and `grad grad(1/r)`
  as expansion order increases, after the approved `-4*pi` normalization.

The generated summary is:

```text
MATRIX_OPERATOR_REFACTOR/data/real_basis_kernel_derivatives/verification_summary.md
```

Confirm no production `src/` code changed during this Theory task.

## Approval Notes

To be filled by a different agent after derivation, verification, and notes are
complete.
