# 010 Implementation Z Rotation Operators

## Objective

Implement explicit z-rotation operators and parity tests.

## Dependencies

- `001-theory-z-rotation-operators.md`
- `007-theory-coefficient-buffer-layout.md`
- `008b-implementation-replan.md`
- `008c-implementation-performance-baseline.md`
- `009-impl-basis-and-operator-cache-types.md`

## Required Reading

- `START_HERE.md`
- Approved dependency task files listed above
- Approved `theory/z-rotation-operators.md`
- Existing z-rotation production code and tests

## Artifacts or Production Surface

- Production z-rotation operator code
- Tests comparing explicit operators with current z-rotation behavior

## Deliverables

- Explicit z-rotation matrix/block construction
- Forward and inverse application paths
- Overwrite and accumulation behavior matching approved theory

## Implementation Summary

- New file `src/rotate_batched.jl` (per START_HERE Code Placement rule 2,
  `rotate.jl` -> `rotate_batched.jl`); `src/rotate.jl` was not modified.
  - `z_rotation_diagonals!(C, S, ϕ, P)` builds the storage-light diagonals
    `C[i]=cos(mϕ)`, `S[i]=sin(mϕ)` over the compressed harmonic index
    (length `((P+1)(P+2))>>1`). The `e^{imϕ}` phases for `m = 0:P` are
    computed once via the `update_eimϕs!` phase recurrence, then scattered to
    every `harmonic_index(n,m)` with `n >= m`.
  - `apply_z_rotation!(out, in, C, S, P, ::Val{LH}, ::Val{:overwrite})` —
    forward rotation by `e^{imϕ}`, overwrite semantics (matches `rotate_z!`).
  - `apply_z_rotation!(out, in, C, S, P, ::Val{LH}, ::Val{:accumulate})` —
    inverse/back rotation by conjugate `e^{-imϕ}`, accumulation semantics
    (matches `back_rotate_z!`). `m=0` is exact identity / pass-through.
- Operates on the existing production layout
  `weights[real_or_imag, component, harmonic_index]` (flat native buffers are
  task 017). No new structs (mode via `Val(:overwrite)`/`Val(:accumulate)`), so
  the `009`-approved type surface is untouched.
- Registered: `include("rotate_batched.jl")` in `src/FastMultipole.jl` after
  `rotate.jl`. The functions are intentionally internal/non-exported; tests and
  later implementation code call them as `FastMultipole.z_rotation_diagonals!`
  and `FastMultipole.apply_z_rotation!` or import them explicitly from the
  module.
- New test `test/rotate_batched_test.jl`, registered in `test/runtests.jl` after
  `rotate_test.jl`.

## Verification

Commands run and results:

```
julia --project=. test/rotate_batched_test.jl
# z-rotation operators (batched): 25684 Pass / 25684 Total
```

Covers `P ∈ {0,1,3,6,9}` × `ϕ ∈ {0.0, 0.25, -1.125, π/3, 2.4}`, both
`Val(false)`/`Val(true)`, and `Float64`/`Float32`. Tests: diagonal correctness,
forward parity vs `rotate_z!`, inverse parity vs `back_rotate_z!` (accumulation
onto a preloaded destination), forward→inverse round-trip identity, and exact
`m=0` identity/pass-through. The focused test also verifies that
`z_rotation_diagonals!` and `apply_z_rotation!` are not exported names while
remaining callable as module internals.

```
julia --check-bounds=yes --project=. test/rotate_batched_test.jl
# z-rotation operators (batched): 25684 Pass / 25684 Total
```

```
julia --project=. MATRIX_OPERATOR_REFACTOR/scripts/z_rotation_verify.jl
# z_rotation_verify: PASS
# max_forward_error: 3.55e-15, max_back_error: 3.11e-15,
# max_m0_forward_error: 0.0, max_m0_back_error: 0.0
```

```
julia --project=. -e 'using Pkg; Pkg.test()'
# Testing FastMultipole tests passed (full suite, no regressions)
```

```
julia --check-bounds=yes --project=. -e 'using Pkg; Pkg.test()'
# Testing FastMultipole tests passed
```

## Approval Notes

Approved by reviewing agent (separate clear-context review) on 2026-06-18.

Objective met: the implementation adds explicit storage-light z-rotation
operators over the existing compressed complex production layout without
modifying legacy rotation call sites.

Deliverables verified:
- `z_rotation_diagonals!` constructs `C[i] = cos(mϕ)` and `S[i] = sin(mϕ)`
  over `harmonic_index(n, m)`.
- `apply_z_rotation!(..., Val(:overwrite))` matches forward `rotate_z!`
  semantics for both `Val(false)` and `Val(true)`.
- `apply_z_rotation!(..., Val(:accumulate))` matches inverse/back
  `back_rotate_z!` accumulation semantics for both channels.
- `m = 0` coefficients are identity/pass-through, as required by the approved
  theory.

Review verification commands and results:

```
julia --project=. test/rotate_batched_test.jl
# z-rotation operators (batched): 25684 Pass / 25684 Total
```

```
julia --project=. -e 'using FastMultipole, Random, Test; using FastMultipole: initialize_expansion, rotate_z!, z_rotation_diagonals!, apply_z_rotation!; ...'
# bit exact forward parity ok
# alias overwrite ok
```

```
julia --project=. -e 'using Pkg; Pkg.test()'
# Testing FastMultipole tests passed
```

Note: the full-suite run initially printed CUDA-related precompile failures and
was interrupted during precompilation, then continued into the package tests and
completed successfully.
