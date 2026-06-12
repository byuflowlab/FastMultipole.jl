# Matrix Operator Refactor

> Background/design rationale only. Not required for routine task selection
> except during Milestone Review. For operational rules, task order, phase
> gates, and approvals, start with `MATRIX_OPERATOR_REFACTOR/START_HERE.md`.

## Goal

Refactor the rotation and translation stages from hand-coded coefficient loops
into explicit real-valued operator applications while preserving the current
rotate-translate-rotate algorithm and its `O(p^3)` asymptotic cost.

This refactor is split into two top-level phases:

1. **Theory phase:** derivations, analytical checks, exploratory scripts,
   numerical basis/operator tests, and generated comparison data.
2. **Implementation phase:** production changes to FastMultipole files,
   especially files under `src/`, plus production tests and benchmarks.

Operational task ordering and phase-gate status are maintained in
`MATRIX_OPERATOR_REFACTOR/START_HERE.md`.

The first implementation target remains the current compressed complex solid
harmonic basis. Coefficients should still be stored as real and imaginary
parts of compressed `m >= 0` complex coefficients, with the existing logical
layout:

```julia
weights[real_or_imag, potential_or_lh, harmonic_index(n, m), branch]
```

Native real solid harmonics are a later implementation target after theory
and parity tests prove that the explicit operator layer matches the current
complex-basis kernels.

## Artifact Layout

Phase artifacts are kept in `MATRIX_OPERATOR_REFACTOR/`:

- `theory/`: derivations, analytical checks, convention notes, and reviewed
  formulas.
- `scripts/`: exploratory numerical scripts used during the Theory phase.
- `data/`: generated numerical results, comparison tables, and script outputs.

Derivations, numerical scripts, generated data, and implementation work are
sequenced by the index and milestone-review tasks.

## Current Algorithm Shape

The production paths in `src/translate.jl` already factor every translation
into rotations, a z-axis translation, and inverse rotations:

- M2M: `rotate_z!`, `rotate_multipole_y!`, `translate_multipole_z!`,
  optional `transform_lamb_helmholtz_multipole!`,
  `back_rotate_multipole_y!`, `back_rotate_z!`.
- M2L: `rotate_z!`, `rotate_multipole_y!`,
  `translate_multipole_to_local_z!` through `multipole_to_local_II!`,
  optional `transform_lamb_helmholtz_local!`,
  `back_rotate_local_y!`, `back_rotate_z!`.
- L2L: z/y rotations, `translate_local_z!`, optional
  `transform_lamb_helmholtz_local!`, and inverse rotations.

The refactor should not replace this decomposition. It should replace the
inner coefficient recurrences with explicit operators that can be tested,
cached, batched, and eventually moved to GPU-oriented execution.

## Theory Sequence

Complete and approve every item in this sequence before implementation:

1. Derive z-rotation operators and overwrite/accumulate semantics.
2. Derive M2L z-translation scaling, including stable binomial-scaled forms.
3. Derive Lamb-Helmholtz operator form.
4. Derive axis-swap conventions and sign choices.
5. Derive the full M2L operator composition.
6. Derive M2M and L2L operator extensions.
7. Derive coefficient-buffer layout and view requirements.
8. Derive complex-to-real and real-to-complex solid-harmonic transforms.

## Implementation Sequence

This sequence is blocked until every Theory task is complete and approved:

1. Add basis/cache types.
2. Add z-rotation operators.
3. Add M2L z-translation blocks.
4. Add Lamb-Helmholtz operators.
5. Add axis-swap operators.
6. Compose the full M2L operator pipeline.
7. Add axis-swap and M2L benchmarks.
8. Extend the operator structure to M2M and L2L.
9. Introduce flat coefficient buffers.
10. Add real-basis transform work and only then evaluate native real-basis
    production migration.

## Operator Layer

Each high-level translation applies the same pieces as the current code:

1. z-axis rotation;
2. invariant axis-swap rotation;
3. z-axis rotation about the swapped axis;
4. inverse invariant axis-swap rotation;
5. z-axis translation;
6. optional Lamb-Helmholtz channel transform;
7. inverse rotation sequence.

For each `(n, m)` coefficient, z rotation is a real `2x2` block applied to the
stored real and imaginary parts:

```math
\begin{bmatrix}
\operatorname{Re} X'_{nm} \\
\operatorname{Im} X'_{nm}
\end{bmatrix}
=
\begin{bmatrix}
\cos(m\phi) & -\sin(m\phi) \\
\sin(m\phi) &  \cos(m\phi)
\end{bmatrix}
\begin{bmatrix}
\operatorname{Re} X_{nm} \\
\operatorname{Im} X_{nm}
\end{bmatrix}.
```

The inverse uses the transpose/sign-flipped sine block, matching
`back_rotate_z!`, which currently reuses `eimϕs` with the imaginary part
negated and accumulates into the target.

The current y rotations are assembled from invariant data plus
angle-dependent scratch:

- `Hs_π2`, populated by `update_Hs_π2!`;
- `ζs_mag` for multipole rotation scaling;
- `ηs_mag` for local rotation scaling.

`Ts` is rebuilt from `Hs_π2` and angle `θ` by `update_Ts!` or
`update_Ts_n!`. The theory phase must establish the exact active/passive
convention before any production operator is added. The target structure is:

```math
R_y(\theta) = R_x(-\pi/2) \, R_z(\theta) \, R_x(\pi/2),
```

with signs adjusted only if the approved convention derivation requires it.

For fixed `m`, source multipole coefficient `M[n', m]` contributes to target
local coefficient `L[n, m]`, with `n, n' >= m`, by

```math
K_m(t)[n,n'] = (n+n')! \, t^{-(n+n'+1)}.
```

The stable theory target is the binomial-scaled factorization:

```math
K_m(t) = \widehat{D}_L(t) \, \widehat{K}_m \, \widehat{D}_M(t),
```

where

```math
\widehat{D}_L[n,n] = n! \, t^{-(n+1)}, \qquad
\widehat{K}_m[n,n'] = \binom{n+n'}{n}, \qquad
\widehat{D}_M[n',n'] = n'! \, t^{-n'}.
```

The Theory phase must test this against the existing recurrence, including
high expansion orders and extreme distances such as `t ≈ 1e-3` and
`t ≈ 1e3`, before implementation.

Lamb-Helmholtz transforms should be first-class linear operators. The Theory
phase must derive the multipole and local forms, including overwrite semantics
and sparse/banded coupling across degree and potential/LH channel space.

## Operator Cache

Implementation should add an operator-cache object that owns invariant
matrices and reusable buffers, not branch-local state. The cache should absorb
current mutable reusable data such as `Hs_π2`, `ζs_mag`, `ηs_mag`, `M̃`, and
`L̃`, and should be parameterized by element type `TF`.

The cache should be keyed by maximum expansion order, basis, translation kind,
Lamb-Helmholtz mode, and kernel/homogeneity class when non-Laplace kernels are
introduced. Shared read-only invariants and per-thread scratch should remain
separate.

The operator `apply!` contract must state whether each stage overwrites or
accumulates. Most current stages overwrite their destination buffers, while
`back_rotate_z!` accumulates into the target expansion.

## Coefficient Storage

Implementation should start from the current logical layout:

```julia
Real/Imag x Potential/LH x Coefficient x Branch
```

with `m >= 0` compressed complex coefficients. The later flat-buffer work
should provide typed views for branch, degree, fixed-`m`, and real/imag access.
`Vector{Matrix}` is acceptable for early inspection of blocks but must not
become the final execution storage.

## Real Solid Harmonics

Real solid harmonics are not the first production migration. The Theory phase
must first derive complex-to-real and real-to-complex transforms, verify
round-trips, and verify evaluated-field parity. Production migration to native
real solid harmonics belongs only after the current-basis operator layer has
passed parity tests.

## Production Test Plan

After the phase gate opens and implementation begins, add operator parity tests
before changing production hot paths:

- z-rotation operator against `rotate_z!` and `back_rotate_z!`;
- invariant axis-swap composition against current y-rotation functions;
- scaled z-M2L matrix operator against `translate_multipole_to_local_z!`;
- Lamb-Helmholtz transform operators against current transform functions;
- full M2M, M2L, and L2L operator pipelines against existing translation
  functions.

After swapping any production path, run the existing rotation, translation,
FMM, Lamb-Helmholtz, and evaluation tests:

- `test/rotate_test.jl`
- `test/translate_multipole_test.jl`
- `test/translate_multipole_to_local_test.jl`
- `test/translate_local_test.jl`
- `test/fmm_test.jl`
- `test/lamb_helmholtz_test.jl`
- `test/evaluate_expansions_test.jl`

## Tree Refactor

Keep tree/radix-sort work separate from the operator refactor. The operator
layer can be written against the current `Tree.branches`, `levels_index`,
`leaf_index`, and sort-index behavior.

## Non-Goals

- Do not change the mathematical translation algorithm in the first
  implementation pass.
- Do not modify production FastMultipole code during the Theory phase.
- Do not switch production storage to native real solid harmonics before the
  current complex-basis operator layer passes parity tests.
- Do not make `Vector{Matrix}` the final execution layout.
- Do not swap dynamic-expansion-order M2L error-method variants in the first
  implementation pass.
- Do not mix tree/radix-sort migration into this operator refactor.
