# 017 Implementation Flat Coefficient Buffers

## Objective

Introduce flat coefficient buffers and typed views after the operator API is
stable.

## Dependencies

- `007-theory-coefficient-buffer-layout.md`
- `008h-theory-lamb-helmholtz-accuracy-order.md`
- `008b-implementation-replan.md`
- `008c-implementation-performance-baseline.md`
- `016-impl-m2m-and-l2l-operator-pipelines.md`
- `016a-milestone-review-impl-013-016.md`

## Required Reading

- `START_HERE.md`
- Approved dependency task files listed above
- Approved `theory/coefficient-buffer-layout.md`
- Current coefficient allocation, indexing, and scratch-buffer code

## Artifacts or Production Surface

- Production coefficient buffer and typed-view code
- Tests for indexing, aliasing, and operator compatibility

## Deliverables

- Flat coefficient buffer representation
- Typed views matching approved layout theory
- Migration of stable operator API paths to the new layout
- `Val(true)` buffers/views must represent `P_phi`, `P_chi = P_phi + 1`, and
  the padded active order `P_active = P_chi` without treating padded `phi`
  rows above `P_phi` as physical output. `Val(false)` remains single-order `P`.

## Implementation Notes (completion pass, 2026-06-25)

Native flat coefficient buffers introduced; the batched operator pipeline now
consumes them. Layout decision (user-directed): operators touch storage only
through buffer accessors, with a **ragged** default backing (separate dense φ and χ
matrices), keeping the padded single-array layout swappable behind the `009`
accessors for `019b`.

Production-surface changes:

- `src/containers.jl`: `flat_basis_index(n,m,reim)`; `FlatCoefficientBuffer{TF,A,B,LH}`
  (φ matrix `basis_dof_phi x batch`; χ matrix `basis_dof_chi x batch`, empty for
  `Val(false)` → **dead χ pruned**) with constructors and the non-allocating
  accessors `phi_slab` / `chi_slab` / `phi_physical_view` / `flat_nbatch`. The three
  `*OperatorScratch` work buffers (`work_a`/`work_b`) are now `FlatCoefficientBuffer`s
  instead of `[2,2,nh,B]` arrays.
- `src/translate_batched.jl`: flat per-column z-translation kernels
  (`apply_{m2l,m2m,l2l}_z_flat!`) and Lamb-Helmholtz coupling
  (`apply_lamb_helmholtz_{multipole,local}_flat!`), order-aware (φ through `P_phi`,
  χ through `P_active`); per-column legacy↔flat repack helpers
  (`_pack_flat_column!`, `_unpack_flat_column!`, `_unpack_flat_column_accumulate!`)
  for the materialized-y stage; the three batch drivers + alignment helpers rewritten
  to flat buffers. `_zero_phi_padding!` removed (ragged φ has no padding rows; the
  order-aware bounds make the no-leak property structural).
- `src/rotate_batched.jl`: flat factored-rotation stages
  (`apply_z_rotation_batch_flat!`, `_factored_y_batch_flat!`,
  `_factored_{source,return}_alignment_batch_flat!` + public wrappers) and a
  `FlatCoefficientBuffer` method of `_factored_input_is_physical` /
  `_assert_factored_input_physical`.
- `src/FastMultipole.jl`: export `FlatCoefficientBuffer`.

The legacy `[2,2,nh]` single-column kernels (z-rotation, z-translation, LH, the
materialized y-op wrappers and production `_rotate_*_y!`) are **retained** as the
parity reference and to back the materialized-y per-column repack; their removal is
deferred to `023` cleanup. Real-basis execution, native flat materialized-y
kernels, padded-vs-ragged final choice (`019b`), and GEMM realization
(`019`/`022`) remain out of scope per the roadmap.

`Val(true)` buffers represent `P_phi`, `P_chi = P_phi + 1`, and the padded active
order `P_active = P_chi`; the ragged φ matrix has no rows above `P_phi`, so there is
no nonphysical φ output. `Val(false)` is single-order `P` with χ pruned.

## Verification

Commands and results (CPU, single thread, macOS):

```text
# 1. Buffer-layout verify script (now cross-checks flat_basis_index)
julia --project=. MATRIX_OPERATOR_REFACTOR/scripts/coefficient_buffer_layout_verify.jl
  -> coefficient_buffer_layout_verify: PASS  (max round-trip error 0.0)
     Production harmonic_index cross-check: PASS
     Production flat_basis_index cross-check: PASS

# 2. Operator + cache + new buffer tests
  coefficient buffer layout (task 017):     2248 passed   (new)
  operator cache support types:              212 passed
  operator cache construction side-effect-free: 5 passed
  cache/scratch == legacy workspace:           4 passed
  fixed y-swap primitives (013b):            548 passed
  factored rotation alignment (013c):        389 passed
  z-rotation operators (batched):          25684 passed
  axis-swap y-rotation operators (batched): 6819 passed
  M2L z-translation blocks (batched):       7163 passed
  Lamb-Helmholtz operators (batched):      11140 passed
  M2L operator pipeline (task 014):         9908 passed   (retargeted to flat)
  M2M/L2L operator pipelines (task 016):   23260 passed   (retargeted to flat)

# 3. Full suite
julia --project=. -e 'using Pkg; Pkg.test()'
  -> Testing FastMultipole tests passed   (exit 0; no regressions in legacy paths)
```

All operator parity counts match the `016b` baseline exactly (the relayout is
bit-for-bit parity-preserving against the production reference and the legacy
kernels). The new `coefficient_buffer_layout_test.jl` validates basis-index
contiguity/uniqueness, exact legacy↔flat round-trip, χ pruning for `Val(false)`,
fixed-channel slab density, `Val(true)` φ-has-no-padding, and flat-vs-legacy kernel
parity.

## Approval Notes

Clear-context review (2026-06-25): **approved**.

The blocking accessor-contract finding was resolved in the follow-up pass:
production flat operator paths in `src/translate_batched.jl` and
`src/rotate_batched.jl` now use `phi_slab` / `chi_slab` accessors rather than
direct `buf.phi` / `buf.chi` field access, preserving the stated backing
swapability contract for the later `019b` padded-vs-ragged decision.

Verification passed:

- `rg -n "\.phi|\.chi" src/translate_batched.jl src/rotate_batched.jl`
  returned no matches.
- `julia --project=. MATRIX_OPERATOR_REFACTOR/scripts/coefficient_buffer_layout_verify.jl`
- `julia --project=. -e 'using Pkg; Pkg.test()'`
