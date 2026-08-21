# Radix-Path M2L Interaction-List Construction

## Scope

This artifact specifies how the radix-sort path builds its M2L interaction list
and routes the complement to direct evaluation. It consumes two approved inputs:

- the occupied uniform-grid cells, integer offsets, and coordinate -> occupied-cell
  lookup from `008f` (`theory/radix-sort-clustering.md`); and
- the accepted relative-offset **stencil** at constant `P` from `008d`
  (`theory/constant-p-error-stencil.md`).

It is theory-only and does not modify production code under `src/`. The legacy
octree path (`build_interaction_lists` in `src/interaction_list.jl`, the geometric
MAC, the pointer-based `m2l_list` / `direct_list`) is unchanged; this artifact
adds a parallel radix-path constructor only.

The correctness target is **complete, non-double-counted n-body coverage**: every
ordered body pair is accounted for exactly once, either through an M2L (far) batch
or through the direct (near/self) set. This is a coverage property, not an
accuracy property: no expansion translations are performed and error convergence
is out of scope here (error control lives in the `008d` stencil and the `008h`
channel-order rule).

## Inputs From `008f` and `008d`

From `008f`, on a fixed-depth uniform grid shared by source and target:

- grid depth `ell`, resolution `G = 2^ell`, cell width `Delta = 2 h0 / G`,
  half-width `w = Delta / 2`, bounding radius `rho = w * sqrt(3)`;
- occupied cells, each with integer coordinate `coord in 0:G-1` per axis, center
  `x_min + Delta * (coord + 1/2)`, and a contiguous `body_range` into the sorted
  permutation `perm`;
- a coordinate -> occupied-cell lookup (empty cells are absent from the table);
- the integer offset class `d = target_coord - source_coord`, with
  `r_vec = Delta * d`, `r = Delta * norm(d)`, and normalized separation
  `c = r / rho = 2 * norm(d) / sqrt(3)`.

From `008d`, the **accepted-offset set** `S(P, ε, budgets)` over a finite offset
domain:

- an offset `d` is a candidate only if `c > 2` (offsets with `c <= 2` are rejected
  to near/direct);
- scalar (`lamb_helmholtz = Val(false)`): accept `d` iff `B(P, d, A)` is finite and
  `B(P, d, A) <= ε`, with `B(P, d, A) = 2A / (rho (c - 2)) * (1 / (c - 1))^(P+1)`;
- Lamb-Helmholtz (`Val(true)`): accept `d` iff `B_LH(P, d, A_phi, A_chi)` is finite
  and `<= ε`, with
  `B_LH = B(P, d, A_phi) + (1 + 2R) * B(P, d, A_chi)` and `R = 2 w * norm(d)`.

Because acceptance depends only on the relative offset `d` (translation
invariance), `S` is computed **once per level** and reused for every occupied
cell. Construction is the radix analogue of the geometric MAC, but the accept test
is a table lookup on the precomputed offset set rather than a per-pair distance
comparison.

## Offset-Class Stencil Application

The far/M2L list is built by sweeping occupied target cells against the accepted
offsets:

```text
for each occupied target cell T with coordinate t:
    for each accepted offset d in S:
        s = t - d                      # candidate source coordinate
        if s is in 0:G-1 per axis and s is occupied (008f lookup):
            let S_cell = occupied cell at s
            enqueue (T, S_cell) into batch[d]
```

Notes:

- `source_coord = target_coord - d` follows the `008f`/`008d` convention
  `d = target_coord - source_coord`. The sign is fixed once here so the operator
  layer and the verifier agree.
- The occupancy test uses `008f`'s coordinate -> occupied-cell lookup, so empty
  source cells cost nothing.
- The sweep is over occupied targets times the (small, level-fixed) accepted-offset
  set, not over all cell pairs. The accepted set is typically a thin shell of
  offsets with `c` just above `2` out to where `B <= ε`.
- The construction is identical for `Val(false)` and `Val(true)`; only the
  accepted set `S` differs (scalar vs Lamb-Helmholtz bound). The list itself is
  channel-agnostic — offsets are purely geometric. The `χ` channel affects only the
  per-pair operator *size*, governed by `008d` and the channel-order rule in
  `008h`; see those artifacts rather than re-deriving here.

## Batched M2L List Layout

The radix path replaces the legacy per-pair pointer list with a **per-offset
batch** structure:

```text
M2lBatches = for each accepted offset d in S:
    Batch(
        d,                              # the shared integer offset class
        targets :: Vector{CellIndex},   # occupied target cells
        sources :: Vector{CellIndex},   # paired occupied source cells (same length)
    )
```

`targets[k]` and `sources[k]` are the `k`-th pair in the batch; both index the
occupied-cell table, from which `007`'s coefficient-buffer layout gives the
expansion-buffer column for each cell (`expansion_index`). Every pair in one batch
shares the same offset `d`, hence the same separation `c`, the same forward/back
rotation, and the same z-axis translation block at the constant `P`. The operator
layer therefore loads the offset-`d` operator once and applies it across the whole
batch — the regular, contiguous, gather/scatter-by-column access pattern that
suits batched GEMM and GPU execution.

Contrast with the legacy `InteractionList` (`src/containers.jl`,
`src/interaction_list.jl`): there `m2l_list` is a flat list of individual
(target branch, source branch) pointer pairs produced by the geometric MAC, each
carrying its own per-interaction `P` from the dynamic-`P` machinery, with no
offset grouping. The radix batch trades that per-pair generality for
translation-invariant, constant-`P` regularity.

The candidate strategies for *executing* M2L over these batches (one folded dense
operator per offset class with a batched GEMM, versus batching the rotation /
z-diagonal stages globally, versus factored strided-batched GEMM) are an
implementation/performance question; they are enumerated and benchmark-gated in
implementation task `015`, not decided here.

## Near/Self Complement -> Direct

The direct/nearfield work set is the **complement** of the far list over the
occupied pairs:

- **self**, `d = 0`: a cell with itself (intra-cell interactions);
- **near**, `d != 0` but `d` not in `S`: includes every `c <= 2` offset and any
  larger offset the stencil rejects at the configured `(P, ε)`.

These are the radix analogue of the legacy `direct_list`. As in production, the
partition is governed by the self-induced / farfield / nearfield switches: the
direct set carries the self (`si`) and near (`nf`) work, the M2L batches carry the
far (`ff`) work, and toggling a switch off drops the corresponding contribution
without altering the others. Because the far set is defined by `S` and the direct
set is its exact complement over occupied pairs, the union is the complete occupied
pair set and the two are disjoint by construction.

In practice the direct set is enumerated from the same sweep: any occupied
(target, source) pair whose offset is not in `S` — found either by the `c <= 2`
near shell or as a non-accepted larger offset — is routed to direct. The
self pairs (`d = 0`) are always present for every occupied cell and always direct.

## Source/Target Grid Sharing

A single integer-offset stencil is well-defined only if the source and target
grids share, per `008f`'s "Source/Target Grids":

- root center;
- root half-width;
- grid depth `ell`;
- cell width `Delta`;
- coordinate quantization convention.

Under these, `d = target_coord - source_coord` has one geometric meaning for every
pair and the accepted set `S` applies uniformly. Separate source/target domains
(which would need an extra coordinate-normalization layer) are out of scope for the
first radix path, consistent with `008f`.

## Complete-Coverage Acceptance Criterion

The construction is correct iff, on a built interaction list, the far/near/self
partition covers every n-body interaction exactly once. This is checked at two
levels (no translations performed — only cell/pair identification):

1. **Cell-pair partition.** Enumerate every ordered occupied (target cell, source
   cell) pair. Classify each into exactly one of: far/M2L (offset `d` in `S`),
   near (offset not in `S`, `d != 0`), or self (`d = 0`). The three sets must be
   pairwise disjoint and their union must equal the full ordered occupied pair set
   — no pair omitted, no pair in two sets.

2. **Body-pair coverage (the real correctness target).** Map the cell-level
   classification to bodies through each cell's `body_range`: for every ordered
   body pair `(i, j)` (`i` a target body, `j` a source body), accumulate the number
   of (cell-pair) routes that cover it. The far route covers `(i, j)` iff `i`'s
   cell and `j`'s cell form an M2L pair; the direct route covers it iff their cells
   form a near or self pair. Every count must be exactly `1`: the union of far and
   near/self body pairs equals the complete `n_target x n_source` ordered set, and
   the two are disjoint.

Two supporting invariants:

- **Stencil/offset agreement.** Every enqueued M2L pair's offset is in `S`, and
  every occupied source at an accepted offset from an occupied target is enqueued
  (no accepted pair dropped).
- **Batch / grid-sharing invariants.** Each per-offset batch contains only pairs of
  that offset; source and target grids share root center, half-width, depth, width,
  and quantization.

Because each occupied cell maps to a contiguous, disjoint `body_range` and the
union of occupied ranges is all bodies, an exact cell-pair partition implies an
exact body-pair partition — the body-pair check is the end-to-end confirmation that
the cell-level reasoning carries down to the n-body level.

## Verification Requirements

The standalone verifier (`scripts/radix_interaction_list_verify.jl`, no production
imports, no expansion translations) builds the interaction list for several test
grids and confirms, for each:

- the cell-pair partition is exact (disjoint far/near/self, union = all occupied
  pairs);
- every ordered body pair is covered exactly once (per-pair count == 1);
- stencil/offset agreement (no accepted pair dropped, no enqueued pair outside
  `S`);
- batch and grid-sharing invariants;
- both `lamb_helmholtz = Val(false)` and `Val(true)` accepted sets.

Point sets: grid-aligned, clustered, sparse-occupancy, and fixed-seed random,
reusing `008f`'s clustering helpers so coverage holds across occupancy patterns.
