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

## Implementation Notes

- Added the `RadixSeparationPolicy` axis with two policies:
  `ParentNeighborM2L` as the default efficient classical path, and
  `ConstantPAnalyticStencil` as the exact same-level constant-`P` stencil path.
- Default radix traversal uses the classical hierarchical parent-neighbor rule:
  leaf offsets with Chebyshev distance `<= 1` route to direct, and every other
  leaf pair is covered by the unique ancestor-level M2L route where the ancestor
  cells are non-neighbors and their parents are neighbors.
- Added procedural phase tables for parent-neighbor M2L candidates. Each target
  child phase has 189 candidates per level before occupancy and boundary
  filtering; direct uses the fixed 27 leaf offsets in `{-1,0,1}^3`.
- Added route-level metadata to `RadixM2LBatch` and
  `foreach_radix_m2l_route`, so per-offset batches remain unambiguous when an
  M2L route is emitted at an ancestor level.
- `ConstantPAnalyticStencil` uses `ConstantPStencilConfig`,
  `constant_p_stencil_bound`, `constant_p_stencil_accepts`, and
  `accepted_radix_stencil` as the active leaf-level stencil classifier. Its
  direct set is the exact complement of accepted same-level leaf offsets.
- `RigidImplicitStencil`, `SparseOffsetIntersection`, `BlockedOccupancyBitsets`,
  and `LazyMaterializedBatches` currently dispatch through the shared
  correct enumerators. The strategy dispatch point remains available for later
  optimized backends.
- The legacy octree interaction-list implementation in `src/interaction_list.jl`
  was not modified.

## Verification Notes

- `julia --project=. -e 'include("test/radix_interaction_list_test.jl")'`
  passed: 61045 tests.
- `julia --project=. -e 'using FastMultipole; println("FastMultipole loaded")'`
  passed.
- `julia --project=. test/runtests.jl` failed before reaching item-021 tests:
  `Package FLOWMath not found in current path` at `test/runtests.jl:3`.

## Approval Notes

Approved after clear-context review on 2026-06-29.

Review scope followed `START_HERE.md`: checked this task file, the approved
constant-`P` and radix-interaction theory artifacts, and the production/test
surfaces touched by item 021 (`src/interaction_list_batched.jl`,
`src/tree_batched.jl`, `src/containers.jl`, `src/FastMultipole.jl`, and
`test/radix_interaction_list_test.jl`).

Findings:

- No blocking correctness findings.
- The implementation exposes the exact same-level constant-`P` analytic stencil
  path through `ConstantPAnalyticStencil` while keeping `ParentNeighborM2L` as
  the default efficient traversal policy documented in the implementation notes.
  The constant-`P` bound uses the approved scalar formula, production
  normalization, and Lamb-Helmholtz `P_phi + 1` chi tail rule.
- `RadixM2LBatch` includes route level metadata, so hierarchical
  parent-neighbor batches do not conflate equal integer offsets at different
  physical scales.
- The tests cover scalar accept/reject behavior, production normalization,
  Lamb-Helmholtz chi-tail order, parent-neighbor candidate counts, route-level
  batching, direct complement behavior, switch behavior, and body-pair coverage.

Verification rerun by reviewer:

- `julia --project=. -e 'include("test/radix_interaction_list_test.jl")'`
  passed: 61045 tests.
- `julia --project=. -e 'using FastMultipole; println("FastMultipole loaded")'`
  passed.
- `julia --project=. test/runtests.jl` still fails before item-021 tests because
  `FLOWMath` is not installed in this environment.

Residual non-blocking test note: the local test helpers compare emitted routes
through `Set`s, so duplicate callback emissions would be hidden by those
comparisons. A raw callback count spot-check on dense grids through `ell=4`
showed no duplicate emissions, but a future low-cost duplicate-count assertion
would make that invariant explicit.
