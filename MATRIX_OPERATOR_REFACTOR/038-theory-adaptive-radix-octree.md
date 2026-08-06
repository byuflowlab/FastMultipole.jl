# 038 Theory: 2:1-Balanced Adaptive Radix Octree

## Status and Entry Gate

**Proposed follow-on item; not started.**

Entry gate: `035` must complete its profiling campaign and `036` (Integration
Phase milestone review) and `037` (rectangular grid) must be complete and
approved. Before derivation begins, confirm from the `035`/`037` measurement
record that multi-scale density (dense clusters plus diffuse regions — e.g.
wake rollup, `CoreSpreading`-grown σ) is a binding cost that the uniform-depth
grid cannot serve, or obtain an explicit user waiver to proceed without that
evidence. Record the confirming evidence or waiver here.

Scoped derivation row: `theory/`, `scripts/`, `data/` artifacts only; does not
reopen the Theory Phase hard gate and does not re-block any completed row.

## Motivation

The uniform radix grid is occupancy-compacted (compute scales with occupied
cells) but rigidly uniform: one global leaf width `h`, one depth `ell`, no
leaf-population bound. Consequences established by the 2026-08-06 sparsity
review:

- One locally large σ forces a globally shallow tree via the geometric gate
  `g_min·h_leaf > ρ_t·σ_max` (`translate_batched_resident.jl` gate), even
  where the field is fine.
- A single fat cell costs `O(K²)` inside one warp in the nearfield pair
  kernel and serializes B2M in one thread; clustering hurts through load
  imbalance before it hurts through cell counts.
- The legacy octree solves both adaptively but is host-only and per-pair
  MAC-driven, incompatible with translation-invariant operator tables.

The enabling fact: the `025` level-scaling law (one operator table serves
every level) plus per-`(level, offset)` class batching over occupied nodes is
exactly the structure a 2:1-balanced adaptive octree needs. With 2:1 balance,
same-level M2L (the V-list) remains a finite translation-invariant
offset-class set — identical batching, identical level-scaled tables. This is
the PVFMM / ExaFMM-T design, proven on GPUs at scale.

## Objective

Derive the complete theory for an adaptive Morton octree on the radix path
that preserves the batched, translation-invariant, device-resident operator
machinery:

1. **Tree definition and construction.** Leaves are the deepest cells with
   population ≤ a split threshold `K_max` (and depth ≤ `ell_max`); top-down
   split on Morton key prefixes over the sorted body stream; 2:1 balance
   sweep. All steps must be expressible as sort/scan/compact primitives
   (the device feasibility argument, not device code).
2. **Interaction lists.** U (adjacent leaves, possibly different levels →
   direct), V (same-level well-separated children of neighbors → M2L by
   offset class, reusing the `025` stencil and scaling law unchanged),
   W (coarse leaf vs finer non-adjacent descendants → M2T), X (dual of W →
   S2L). Precise definitions on Morton keys, plus generation algorithms in
   flag/scan/compact form.
3. **Exact-once coverage proof.** Every body pair is covered exactly once by
   the union of U/V/W/X plus self-interaction, for any 2:1-balanced leaf set,
   at both supported near radii (`|o|² ≤ 12` and classic `|o|² ≤ 3`). Verify
   computationally on test trees including adversarial (highly clustered)
   distributions.
4. **M2T and S2L operators.** Formulate in the compressed complex basis
   consistent with `005`/`007` conventions and the Lamb-Helmholtz channel
   (`003`, `008h`): M2T evaluates a multipole expansion directly at target
   bodies; S2L accumulates sources directly into a local expansion. Derive
   error bounds consistent with the constant-`P` stencil error model (`008d`,
   `025`) so W/X interactions respect the same accuracy target as V.
5. **Per-cell geometry gate.** Replace the global `σ_max` gate with a
   per-leaf admissibility rule (leaf width vs the σ of the bodies it holds
   and its U-list partners), and prove it implies the phase accuracy gate
   under the regularized-kernel contract from `031a`.
6. **Cost and capacity model.** Expected list sizes and work vs `K_max`,
   depth ceiling, and occupancy statistics for the two phase test cases plus
   a synthetic multi-scale case (e.g. cube + embedded dense cluster at
   10–100× local density); capacity formulas suitable for the
   `RadixFMMCache` no-realloc contract (max leaves, max list lengths as
   functions of `max_n_bodies`, `K_max`, `ell_max`).
7. **Refresh/rebuild policy.** When body motion invalidates the leaf set;
   what an in-place refresh can reuse (cf. the occupancy-epoch window cache)
   vs a full rebuild; interaction with `recenter!`.

## Deliverables

- `theory/adaptive-radix-octree.md` covering items 1–7 with derivations.
- A stdlib-only validation script under `scripts/` exercising the exact-once
  proof and the per-cell gate on the test distributions.
- Generated evidence tables under `data/adaptive_octree/`.

## Constraints

- Reuse approved conventions: `025` stencil and scaling law, `007` buffer
  layout, `008h` χ-at-`P+1` rule, `031a` regularized-nearfield contract.
- The V-list path must require **no new operator tables** beyond the existing
  level-scaled set; if any deviation is unavoidable, quantify it and stop for
  user discussion.
- No production `src/` changes in this row.

## Acceptance

Approved when the exact-once proof is verified computationally on clustered
and uniform test trees, the M2T/S2L error bounds are consistent with the
constant-`P` model, the cost model quantifies the expected win on the
multi-scale case (and the expected non-regression on the cube/wake cases),
and the capacity formulas are explicit enough for `039`–`041` to implement
against without re-derivation.
