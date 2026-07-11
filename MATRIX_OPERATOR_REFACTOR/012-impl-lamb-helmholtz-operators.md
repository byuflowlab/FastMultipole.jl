# 012 Implementation Lamb Helmholtz Operators

## Objective

Implement Lamb-Helmholtz transform operators and parity tests.

## Dependencies

- `003-theory-lamb-helmholtz-operator-form.md`
- `007-theory-coefficient-buffer-layout.md`
- `008h-theory-lamb-helmholtz-accuracy-order.md`
- `008b-implementation-replan.md`
- `008c-implementation-performance-baseline.md`
- `009-impl-basis-and-operator-cache-types.md`

## Required Reading

- `START_HERE.md`
- Approved dependency task files listed above
- Approved `theory/lamb-helmholtz-operator-form.md`
- Existing Lamb-Helmholtz production code and tests

## Artifacts or Production Surface

- Production Lamb-Helmholtz operator code
- Tests comparing explicit operators with current channel behavior

## Deliverables

- Multipole and local Lamb-Helmholtz operator construction
- Channel-coupling application path
- Cache integration compatible with task `009`
- Local operators for `Val(true)` must include the upper-neighbor
  `chi_{P_phi + 1} -> chi_{P_phi}` row from
  `theory/lamb-helmholtz-accuracy-order.md`. Same-order `chi` truncation is a
  rejected candidate for the matrix M2L/evaluation path.

## Implementation Summary

- Appended to the existing `src/translate_batched.jl` (per START_HERE Code
  Placement rule 2: the production Lamb-Helmholtz transforms
  `transform_lamb_helmholtz_multipole!`/`transform_lamb_helmholtz_local!` live in
  `src/translate.jl`, so `translate.jl -> translate_batched.jl`; no dedicated
  `lamb_helmholtz.jl` source file exists). `src/translate.jl` and all production
  call sites were left unmodified (additive only, mirroring tasks `010`/`011`).
- The transform is sparse (no cross-`m`; same-degree φ-from-χ; nearest-neighbor
  χ-from-χ; identical real scalar for both lanes), so it is fully described by two
  real factors per stored `(n, m)`:
  - `A[i]` — same-degree φ-from-χ factor; `B[i]` — nearest-neighbor χ-from-χ
    factor, with `i = harmonic_index(n, m)`.
  - `lamb_helmholtz_multipole_coeffs!(A, B, r, P)`: `A = r*m/(n+1)`, `B = r/n`
    (lower neighbor `χ_{n-1} -> χ_n`; `B` zeroed at `n = 0`).
  - `lamb_helmholtz_local_coeffs!(A, B, r, P)`: `A = r*m/n` (zeroed at `n = 0`),
    `B = r/(n+1)` (upper neighbor `χ_{n+1} -> χ_n`).
  - Factors are materialized (they depend only on `r` and degrees) for the same
    offset-class batching reuse rationale as the `011` M2L z blocks (tasks
    `020`/`021`/`022`). The formulas and multiply order match production so
    materialize-then-apply is bit-for-bit identical.
- Apply functions use separate `in`/`out` buffers with `:overwrite` semantics
  (accumulation deferred to the inverse z-rotation per approved theory). Because
  the in-place production loops are ordered to always read original (pre-transform)
  χ values, reading every term from `in` reproduces them exactly:
  - `apply_lamb_helmholtz_multipole!(out, in, A, B, P, ::Val{:overwrite})` and
    `apply_lamb_helmholtz_local!(out, in, A, B, P, ::Val{:overwrite})` — symmetric,
    bit-for-bit vs production at order `P` (local includes the production top-degree
    truncation `χ_{P+1} = 0`).
  - Order-aware `OperatorBasisInfo{<:CompressedComplexBasis,LH}` methods driven
    only through the `009` accessors: physical φ output through `orders.P_phi`; for
    `Val(true)` the χ channel carried through the padded
    `orders.P_active = P_chi = P_phi + 1`. The local order-aware method's χ
    truncation moves up to `P_active`, which is exactly what supplies the required
    `χ_{P_phi+1} -> χ_{P_phi}` upper-neighbor row from
    `theory/lamb-helmholtz-accuracy-order.md` (same-order χ truncation rejected).
    For `Val(false)`, `P_active == P_phi`, the χ channel is skipped, and φ is
    copied through (component 2 is zero in the always-2-component buffer).
- Operates on the existing production layout
  `weights[real_or_imag, component, harmonic_index]` (flat native buffers are task
  `017`). No new structs; the `009` `OperatorBasisInfo`/`OperatorOrders` surface is
  reused untouched. Functions are internal/non-exported.
- Clear-context review found and fixed one scalar-path issue: the order-aware
  `Val(false)` methods now copy φ without reading the absent χ channel, and zero
  component 2 in the always-2-component production buffer. Regression coverage
  populates χ with nonzero sentinels to verify the scalar path ignores it.
- No source registration change needed (`translate_batched.jl` is already
  `include`d from `src/FastMultipole.jl` and `test/translate_batched_test.jl` from
  `test/runtests.jl`, both from task `011`). New tests were added as a second
  `@testset` in `test/translate_batched_test.jl`.

## Verification

Commands run and results:

```
julia --project=. test/translate_batched_test.jl
# M2L z-translation blocks (batched): 7163 Pass / 7163 Total
# Lamb-Helmholtz operators (batched): 11140 Pass / 11140 Total
```

```
julia --check-bounds=yes --project=. test/translate_batched_test.jl
# M2L z-translation blocks (batched): 7163 Pass / 7163 Total
# Lamb-Helmholtz operators (batched): 11140 Pass / 11140 Total
```

```
julia --project=. -e 'using Pkg; Pkg.test()'
# full suite — no regressions
```

Lamb-Helmholtz coverage: `P ∈ {0,1,3,6,9}` × distances `r ∈ {0.7, 1.0, -1.9,
3.25}`, `Float64`/`Float32`. Tests: non-export check; **bit-for-bit** (`===`)
symmetric parity for both multipole and local sides vs
`transform_lamb_helmholtz_*!` with overwrite-of-sentinel; order-aware padded
parity (φ rows `n ≤ P_phi` and χ rows `n ≤ P_active` equal the `P_active`-order
production result bit-for-bit, for both multipole and local); an explicit check
that the order-aware local χ_{P_phi} row **differs** from the same-order
truncated result (proving the upper-neighbor row is genuinely carried); and the
`Val(false)` φ-copied-through / χ-zeroed behavior with nonzero χ input sentinels.

## Approval Notes

Approved after clear-context review on 2026-06-18. Review covered the task file,
the listed production surface in `src/translate_batched.jl`,
`src/translate.jl`, `src/containers.jl`, and the parity tests in
`test/translate_batched_test.jl`/`test/lamb_helmholtz_test.jl`. One scalar-path
leak from the absent χ channel was fixed before approval. Focused tests,
bounds-checked focused tests, and the full package suite pass.
