# 039 Impl: Adaptive Octree Construction and Interaction Lists (Host)

## Status and Entry Gate

**Proposed follow-on item; not started.**

Entry gate: `038` complete and approved.

## Objective

Implement the `038` adaptive tree and its U/V/W/X interaction lists on the
host as the reference path: Morton-prefix top-down splitting with population
threshold `K_max` and depth cap, 2:1 balance sweep, compact leaf/node
metadata, and list generation — all structured as the sort/scan/compact
primitives that `041` will mirror on device.

## Scope and placement

- New tree code extends `tree_batched.jl`; list construction extends
  `interaction_list_batched.jl`; new structs go in `containers.jl`
  (per the Implementation Code Placement rules).
- The uniform-depth radix grid remains the default and is untouched; the
  adaptive tree is an opt-in constructor/policy on `RadixFMMCache`.
- No CUDA code in this row (`041`); no M2T/S2L operator execution in this
  row (`040`) — this row builds the tree and the lists and proves them
  correct.

## High-level deliverables

- Adaptive leaf construction from sorted Morton keys: split-while
  `count > K_max`, depth ≤ `ell_max`, then 2:1 balance; occupied-ancestor
  node levels compatible with the existing `level_offsets` convention.
- U/V/W/X list generation per `038`, with V lists emitted as
  `(level, offset)` classes identical in format to the existing hierarchical
  route stream (so `040` can feed them to the unchanged M2L strategies).
- Per-cell geometry gate from `038` item 5 replacing the global `σ_max` gate
  on the adaptive path.
- Capacity plumbing per the `038` cost model: all list/leaf buffers sized at
  construction from `max_n_bodies`/`K_max`/`ell_max`, zero per-step
  allocation on refresh.
- Refresh path: re-sort, re-split, re-balance, and regenerate lists in place
  within capacity, following the `023` invariant contract.

## Verification

- Exact-once coverage: brute-force pair enumeration vs U∪V∪W∪X∪self on
  randomized uniform, wake-like, and clustered multi-scale distributions
  (multiple seeds, multiple `K_max`), both near radii.
- 2:1 balance invariant asserted structurally.
- V-list class format parity: on a uniform distribution with
  `K_max` chosen so all leaves land at one level, the adaptive V lists must
  reproduce the existing hierarchical route set exactly.
- Tests include `P=4` (standing rule) and run in the standard suite.

## Acceptance

Lists proven exact-once on all test distributions, uniform-limit parity with
the existing hierarchical routes, zero per-step allocation on refresh, and
construction cost measured and recorded vs the uniform grid on the two phase
cases plus the multi-scale case.
