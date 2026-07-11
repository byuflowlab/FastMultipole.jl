# 021 Implementation Constant-P Stencil And Radix Interaction List

## Objective

Implement the constant-`P` translation-invariant M2L stencil and the per-offset-class
radix M2L interaction list, with the near/self complement routed to direct, so the
new operators can run an end-to-end FMM on the radix path.

## Dependencies

- `008d-theory-dynamic-p-error-m2l-integration.md` (constant-`P` stencil)
- `008g-theory-radix-interaction-list.md`
- `008h-theory-lamb-helmholtz-accuracy-order.md` (channel-order tails)
- `008b-implementation-replan.md`
- `008c-implementation-performance-baseline.md`
- `014-impl-full-m2l-operator-pipeline.md`
- `020-impl-radix-grid-clustering.md`

## Required Reading

- `START_HERE.md`
- Approved dependency task files listed above
- Approved `theory/constant-p-error-stencil.md` and `theory/radix-interaction-list.md`
- Existing `interaction_list.jl` for reference (not modified here)

## Artifacts or Production Surface

- New `src/interaction_list_batched.jl` containing the stencil and radix M2L list
  (per the `_batched` placement rule).
- Tests for stencil accept/reject geometry, per-offset-class batching, and complete
  non-double-counted n-body coverage on a test grid.

## Deliverables

- Precomputed translation-invariant offset stencil from the conservative constant-`P`
  bound; offsets with `c <= 2` route to near/direct. Implement the bound through a
  function-based interface (do not hard-code numbers) so the under-discussion stencil
  constants in `008d` remain swappable.
- Lamb-Helmholtz channel tails per `008h`: interpret `P` as `P_phi`, evaluate
  `B_chi` at `P_phi + 1`, combine as `B_LH = B_phi + (1 + 2R) * B_chi`.
- Per-offset-class batch structure listing (target, source) cell pairs sharing a
  geometric offset, plus the direct complement.

## Verification

On a deterministic test grid, verify: stencil accept/reject matches the bound,
per-offset batches are correctly populated, and every body pair appears exactly once
(M2L or direct) with no double counting. Record commands and result summaries.

## Approval Notes

To be filled by a different agent after implementation and verification are
complete.
