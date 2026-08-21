# 008g Theory Radix-Path Interaction-List Construction

## Objective

Derive the M2L **interaction-list construction** for the radix-sort clustering
path. Given the occupied uniform-grid cells from `008f` and the accepted
relative-offset **stencil** at constant `P` from `008d`, specify how to build:

1. the **M2L batches**, grouped by integer offset class
   `d = target_coord - source_coord`, of (target cell, source cell) pairs whose
   offset is in the accepted stencil; and
2. the **near/self complement** — all remaining occupied cell pairs (near offsets
   and self, `d = 0`) — routed to direct evaluation.

`008d` already states that error control "moves all error control into
interaction-list construction." `008f` sketches the queue construction in
passing. This task is the full specification and verification of that
construction: it consumes the stencil (`008d`) and the cell geometry (`008f`) and
produces the concrete list/batch structure the constant-`P` M2L operator chain
(task `005`) iterates over.

This task is sequenced to run **after `008d` is complete and approved**, because
the exact accepted-offset set comes from `008d`'s conservative error bound. It is
theory-only and must not modify production code under `src/`. Like every Theory
Addendum row, it is a full hard-gate blocker for every Implementation task.

Added by user request on 2026-06-13, sequenced after the `008d` error-handling
task; full Theory Addendum row under the standard hard phase gate.

## Dependencies

- `008d-theory-dynamic-p-error-m2l-integration.md` (provides the accepted
  relative-offset stencil and the constant-`P` error strategy)
- `008f-theory-radix-sort-clustering.md` (provides occupied uniform-grid cells,
  integer offsets, coord -> occupied-cell lookup, and body ranges)
- `008b-implementation-replan.md`
- `007-theory-coefficient-buffer-layout.md` (expansion-buffer indexing for queued
  cell pairs)
- `005-theory-full-m2l-composition.md` (the constant-`P` M2L chain the batches
  feed)

## Required Reading

- `START_HERE.md`
- Approved dependency task files listed above (especially `008d` and `008f`)
- Current production interaction-list code (read-only, no edits):
  - `src/interaction_list.jl`: `build_interaction_lists` and
    `build_interaction_lists!` per method (`Barba`, `SelfTuning`,
    `SelfTuningTargetStop`, `SelfTuningTreeStop`); `sort_by_target`,
    `sort_by_target_multithreaded`, `sort_by_source`; the `InteractionList`
    constructor producing `m2l_list` and `direct_list`
  - `src/containers.jl`: `InteractionList`, `Branch`
  - The geometric MAC, for contrast with the offset-class accept test

## Scope / Deliverables

- **Offset-class stencil application.** Consume `008d`'s accepted offset set. For
  each occupied target cell and each accepted offset `d`, compute
  `source_coord = target_coord - d` and test occupancy using `008f`'s
  coord -> occupied-cell lookup. If occupied, enqueue the (target cell, source
  cell) pair into the batch for `d`. Because acceptance depends only on the
  relative offset, the stencil is computed once per level and reused for every
  cell (translation-invariant construction).
- **Batched M2L list layout.** Define the per-offset batch container: lists of
  (target cell, source cell) pairs that share one rotation/translation class at
  the constant `P`. Specify the expansion-buffer indices per cell (from `007`) and
  the contiguous/regular access pattern that suits GPU execution. Contrast with the
  legacy pointer/per-pair `m2l_list`.
- **Near/self complement -> direct.** Specify the partition: occupied cell pairs
  whose offset is *not* in the far stencil — including self (`d = 0`) and near
  offsets — form the direct/nearfield work set, the radix-path analogue of the
  legacy `direct_list`. Define the self-induced and farfield/nearfield toggles
  analogous to the production `Val{ff/nf/si}` switches.
- **Source/target grid sharing.** Require source and target grids to share root
  center, half-width, depth, cell width, and quantization convention (per `008f`'s
  "Source/Target Grids"), so a single integer offset stencil is well-defined.
  Separate source/target domains are out of scope (deferred with `008f`).
- **Lamb-Helmholtz note.** The list construction is channel-agnostic: offsets are
  purely geometric. The `χ`-channel (`lamb_helmholtz = Val(true)`) affects only the
  per-pair operator *size*, which `008d` governs. Cross-reference `008d`'s
  Lamb-Helmholtz subheading rather than re-deriving it here.
- **Legacy octree path unchanged.** `build_interaction_lists` and the MAC-based
  `m2l_list` / `direct_list` remain exactly as in production for the dynamic-`P`
  octree path. `008g` adds a parallel radix-path constructor only.
- **Complete-coverage acceptance criterion (explicit).** This task is not finished
  until the verifier demonstrates, on a built test-grid interaction list, that the
  far/near/self partition covers **every n-body interaction exactly once** — no
  pair omitted, no pair double-counted — both at the cell-pair level and, mapped
  through cell membership, at the body-pair level. No expansion translations are
  performed; only cell/pair identification is checked.

## Non-Goals

- Does not derive the conservative stencil bound; that is `008d`.
- Does not derive the radix-sort clustering; that is `008f`.
- Does not modify `src/` or alter `build_interaction_lists`.
- Does not finalize adaptive / level-binned grid variants (deferred with `008f`).
- Does not perform any expansion translation; coverage is verified by cell/pair
  identification only.

## Artifacts or Production Surface

This is a Theory Phase task. It must not modify production code under `src/`.

Artifacts:

- `theory/radix-interaction-list.md` — construction derivation and batch-layout
  specification.
- `scripts/radix_interaction_list_verify.jl` — standalone deterministic verifier
  (no production imports; performs no expansion translations — it only checks which
  cells/pairs are identified). It builds the interaction list for one or more test
  grids and proves complete, non-overlapping coverage:
  - **Cell-pair partition.** Enumerate every ordered occupied (target cell, source
    cell) pair. Each pair is classified into exactly one of: far/M2L (accepted
    offset `d`), near/direct, or self (`d = 0`). Assert the three sets are pairwise
    disjoint and their union is the full pair set — no pair missing, no pair in two
    sets.
  - **Body-pair coverage (the real correctness target).** Map the cell-level
    classification down to bodies: every ordered body pair `(i, j)` (`i` a target
    body, `j` a source body) must be covered exactly once — either by the M2L batch
    of `i`'s cell <-> `j`'s cell, or by the direct/self set. Accumulate a per-pair
    count and assert every count is exactly 1 (no double-count, no omission); the
    union of far and near/self body pairs equals the complete
    `n_target x n_source` set and the two are disjoint.
  - **Stencil/offset agreement.** Each enqueued M2L pair's integer offset is in the
    accepted-offset set, and every occupied source at an accepted offset from an
    occupied target is enqueued (no accepted pair dropped).
  - **Batch and grid-sharing invariants.** Per-offset batches contain only pairs of
    that offset; source/target grids share root center, half-width, depth, width,
    and quantization.
  Run on representative point sets (grid-aligned, clustered, sparse, fixed-seed
  random), reusing `008f`'s clustering helpers, so the coverage proof holds across
  occupancy patterns.
- `data/radix_interaction_list/verification_summary.md` — generated summary.

## Verification

Run:

```sh
julia --project=. MATRIX_OPERATOR_REFACTOR/scripts/radix_interaction_list_verify.jl
```

The verifier builds the interaction list for the test grids and confirms the
complete-coverage acceptance criterion above: the cell-pair and body-pair
partitions each cover every interaction exactly once, the M2L batches agree with
the accepted-offset stencil, and the batch/grid-sharing invariants hold. No
expansion translations are performed.

Confirm no production `src/` code changed during this Theory task.

The generated summary is:

```text
MATRIX_OPERATOR_REFACTOR/data/radix_interaction_list/verification_summary.md
```

## Approval Notes

Derivation, verification, and notes are complete for this task:

- `theory/radix-interaction-list.md` specifies the offset-class stencil sweep
  (`source_coord = target_coord - d`), the per-offset batched M2L layout and its
  contrast with the legacy per-pair `m2l_list`, the near/self -> direct complement
  with `Val{ff/nf/si}`-analogous toggles, the shared source/target grid
  requirement, the channel-agnostic Lamb-Helmholtz note, and the explicit
  complete-coverage acceptance criterion.
- `scripts/radix_interaction_list_verify.jl` builds the list for grid-aligned,
  clustered, sparse, and fixed-seed random grids (both `Val(false)` and
  `Val(true)`), with no production imports and no expansion translations. It
  confirms the exact far/near/self cell-pair partition, agreement between the swept
  M2L batches and the classified far set, per-offset batch purity, grid-sharing
  invariants, and — the real target — that every ordered body pair is covered
  exactly once. The random case exercises a non-trivial near set (342 near, 3198
  far, 60 self) and the LH bound is correctly more conservative.
- `data/radix_interaction_list/verification_summary.md` records the per-case table;
  the verifier prints `radix_interaction_list_verify: PASS`.
- `git status --short src/` and `git diff --stat -- src/` are empty: no production
  code changed.

## Clear-Context Approval

Reviewed the allowed clear-context scope for task `008g`: `START_HERE.md`, this
task file, the listed derivation artifact, verifier, generated verification
summary, and the listed production interaction-list surfaces in read-only mode.

Dependencies `008d` and `008f` are marked approved in `START_HERE.md`. The
derivation consumes their accepted-offset stencil and uniform-grid geometry
without modifying production code. The radix-path construction is specified as a
translation-invariant sweep over accepted offset classes, with per-offset M2L
batches and an exact near/self direct complement. The verifier was rerun locally
with:

```sh
julia --project=. MATRIX_OPERATOR_REFACTOR/scripts/radix_interaction_list_verify.jl
```

It printed `radix_interaction_list_verify: PASS` and regenerated the recorded
summary. `git diff --stat -- src` is empty.

Conclusion: `008g` is approved.
