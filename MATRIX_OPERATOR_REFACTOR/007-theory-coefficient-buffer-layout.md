# 007 Theory Coefficient Buffer Layout

## Objective

Specify coefficient-buffer layout, indexing, and typed view requirements for
the explicit operator layer.

## Dependencies

- `005-theory-full-m2l-composition.md`
- `006-theory-m2m-l2l-extensions.md`

## Required Reading

- `START_HERE.md`
- Approved dependency task files listed above
- Current coefficient storage, indexing, and scratch-buffer code

## Artifacts or Production Surface

- `theory/coefficient-buffer-layout.md`
- `scripts/coefficient_buffer_layout_*.jl`
- `data/coefficient_buffer_layout/`

## Deliverables

- [x] Proposed flat-buffer layout and typed views
- [x] Mapping to current logical coefficient layout
- [x] Scratch, cache, and aliasing requirements
- [x] Buffer/view requirements for both compressed complex and real solid harmonic
  bases across M2M, M2L, and L2L matrix-operator applications
- [x] Confirmation that buffer layout choices do not alter the approved invariant
  matrix and z-axis rotation operator contract

## Verification

Validate indexing maps with deterministic coefficient round trips and layout
examples for both complex and real basis layouts. Record commands, tolerances
where relevant, and result summaries.

Completed with:

```text
julia --project=. MATRIX_OPERATOR_REFACTOR/scripts/coefficient_buffer_layout_verify.jl
```

Generated artifact:

```text
MATRIX_OPERATOR_REFACTOR/data/coefficient_buffer_layout/verification_summary.md
```

Result summary:

- Status: `PASS`
- Expansion orders: `0`, `1`, `3`, `6`, `9`
- Layouts: `Val(false)`, `Val(true)`
- Batch counts: `1`, `2`, `5`
- Max complex legacy/native round-trip error: `0.0`
- Fixed-channel slab requirement verified:
  `stride(view, 1) == 1`, `stride(view, 2) == basis_dof`
- Real-basis mode indices verified contiguous and unique.

## Approval Notes

Reviewed by a separate (clear-context) agent than the completing agent.

Correctness: **verified against production.** The theory's
`harmonic_index(n, m) = n(n + 1)/2 + m + 1` matches `src/harmonics.jl:5`; the
"current logical production shape" `weights[real_or_imag, component, harmonic]`
matches production indexing (`src/translate.jl`, `src/containers.jl:205`); the
legacy-to-native mapping is an exact permutation. All five deliverables are
covered and the "does not alter the operator contract" confirmation is explicit.

Verification: `coefficient_buffer_layout_verify.jl` re-run → **PASS** (round-trip
error `0.0`; fixed-channel slab strides `(1, basis_dof)` confirmed via real
`stride()`; complex and real index sets contiguous/unique for P = 0,1,3,6,9).

Changes required before approval (now completed):

- Added a **Performance Trade-offs** section to
  `theory/coefficient-buffer-layout.md` documenting (1) tight `lda` packing vs
  the channel-merge GEMM and the foreclosed GPU `lda`-padding lever, (2)
  interleaved vs planar re/im with the real basis as the intended performance
  path, and (3) batch membership / single-vs-strided-batched GEMM.
- Hardened the verify script with an independent cross-check that its
  `harmonic_index` matches `FastMultipole.harmonic_index` (guarded import;
  recorded in the summary). Cross-check status: **PASS**.

Performance is documented as forward-looking guidance; no operator mathematics
changed. Approved.
