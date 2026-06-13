# 008d Theory Constant-`P` Error Handling and M2L Interaction-List Integration

## Objective

Specify how error control integrates with the approved matrix-operator M2L
pipeline. The approved M2L composition (task `005`) specifies only the fixed-`P`
chain. This task records the error-handling strategy for two distinct clustering
paths:

1. **Legacy octree path.** The existing dynamic-`P` / error-prediction machinery
   (`get_P`, `predict_error`, per-interaction truncation inside
   `multipole_to_local!`) is preserved exactly. The operator layer must be able
   to reproduce current production behavior at the selected per-interaction `P`.

2. **New radix-sort clustering path** (depends on `008f`). The per-interaction
   dynamic-`P` machinery does not fit a large-`N`, GPU-batched, uniform-grid
   clustering. This path uses a **constant expansion order `P`** everywhere and
   moves all error control into **interaction-list construction**: M2L is
   performed only between cells whose conservative error bound at the constant
   `P` is within tolerance. The chosen mechanism is a translation-invariant
   interaction-list **stencil** (option (c) below).

Consequence for the operator refactor: on the radix-sort path the constant-`P`
M2L operator chain from task `005` runs at fixed size — there is no
per-interaction truncation and no mid-pipeline dynamic-`P` integration to derive.
This is the simplification that motivates the new path.

This task was added by the `008b` Implementation Re-Plan and expanded by the
`2026-06-13` re-plan addendum recorded in `008b`. It is a Theory Phase task: it
blocks every Implementation task under the standard hard phase gate.

## Dependencies

- `008f-theory-radix-sort-clustering.md` (provides the cell geometry the stencil
  bound operates over)
- `002-theory-m2l-z-translation-scaling.md`
- `005-theory-full-m2l-composition.md`
- `007-theory-coefficient-buffer-layout.md`
- `008b-implementation-replan.md`

## Required Reading

- `START_HERE.md`
- Approved dependency task files listed above
- Approved artifacts `theory/m2l-z-translation-scaling.md` and
  `theory/full-m2l-composition.md`
- Current production M2L / error code (read-only, no edits):
  - `src/translate.jl`: `multipole_to_local!` error-method branches,
    `translate_multipole_to_local_z_m01_n`, `local_power`, `update_M̃!`,
    `update_L̃!`, `multipole_to_local_II!`
  - `src/error.jl`: error-bound formulas (`UnequalSpheres`,
    `PringleAbsolutePotential`, `RotatedCoefficients*`, `Power*`, `Dehnen*`)
  - `src/dynamic_expansion_order.jl`: `get_P` per-interaction order selection
  - `src/interaction_list.jl`: `build_interaction_lists` and the geometric MAC

## Error-Handling Strategy for the Radix-Sort Path

### Options considered

- **(a) Threshold-fitted interaction list.** Build a list where M2L runs between
  cells that *barely* satisfy the error threshold under conservative metrics,
  possibly level-transcending (a coarse source cell sends M2L to a finer target
  cell at a lower level). Tightest work set, but list construction is irregular
  and data-dependent.
- **(b) Multipole-power selection (Dehnen).** Use Dehnen's multipole-power idea
  to decide, at the constant `P`, which cell pairs receive M2L; then batch the
  queued M2L. Risk: queue construction itself becomes the bottleneck and erodes
  the GPU batching benefit.
- **(c) Conservative stencil — CHOSEN.** Use a cheap, conservative error bound to
  define a traditional translation-invariant interaction-list **stencil**. The
  accept/reject set depends only on the relative cell offset, so it is computed
  once per level and reused for every cell. This amortizes interaction-list
  construction and yields the regular, batched access pattern that suits GPUs.

**Decision: (c).** In the absence of a stronger idea, the conservative stencil is
adopted because it amortizes list-build cost and is GPU-batch-friendly. Options
(a) and (b) are retained here only as a record of considered alternatives.

### Conservative error bound (starting point)

Starting point: the original Greengard–Rokhlin multipole truncation bound. For a
source cell with bounding-sphere radius `ρ` about its center and total source
strength `A = Σ|qᵢ|`, an order-`P` truncated multipole expansion evaluated at a
point a distance `r > ρ` from the source center satisfies

```
|ε_multipole| ≤ A / (r − ρ) · (ρ / r)^(P + 1)
```

with the dual bound on the target (local-expansion) side.

On a uniform radix grid (cell half-width `w`, bounding-sphere radius
`ρ = w·√3`, integer center-to-center offsets giving distance `r`), this collapses
to a `(1/c)^(P+1) / (c − 1)` form in the separation ratio `c = r / ρ`. This is
the same family already implemented in `src/error.jl` as `UnequalSpheres` /
`PringleAbsolutePotential`.

At a fixed `P` and tolerance `ε`, the inequality fixes the minimum integer
separation at which a pair is "well separated" — i.e. the near/far boundary of
the stencil. Because the bound depends only on the relative offset, the stencil
is identical for every cell at a level: compute it once, reuse everywhere.

TODO (user): confirm/derive the precise conservative bound to adopt, any
tightening relative to the Greengard–Rokhlin form above, and the exact mapping
from `(P, ε)` to the integer stencil radius. The exact bound is still under
discussion.

### Lamb-Helmholtz (`χ`-channel) extension

TODO (user): extend the conservative bound and the resulting stencil to the
Lamb-Helmholtz `χ` channel (`lamb_helmholtz = Val(true)`). The single-channel
Greengard–Rokhlin form above does not yet account for the vector-potential
channel.

**This task is not finished until this subheading is filled out.**

## Operator-Pipeline Implication

- **Radix-sort path (constant `P`).** The M2L operator chain from task `005`
  runs at a fixed `P` for every accepted pair: forward rotations, the fixed-`m`
  z-block, the optional Lamb-Helmholtz stage, and the accumulating back-rotation
  are all sized by the single constant `P`. No per-interaction truncation and no
  dynamic-`P` integration are required on this path.
- **Legacy octree path (dynamic `P`).** Unchanged. M2L still routes through the
  existing `get_P` / `predict_error` machinery in
  `src/dynamic_expansion_order.jl` and `src/error.jl`, selecting `P` per
  interaction before the accumulating back-rotation completes. The operator
  layer on this path must reproduce production behavior at the selected `P`.

## Old / New Operator Coexistence

Both operator implementations are kept in the repository:

- The **old operators** (current production `multipole_to_*!` recurrences) remain
  compatible with the **old error machinery** (dynamic-`P`, `get_P`,
  `predict_error`). This is the legacy octree path.
- The **new expansion operators** (the matrix-operator chain from task `005`) are
  **not** wired to the old error machinery. They run at constant `P` and rely on
  the conservative interaction-list stencil for error control (radix-sort path).

Porting the old per-interaction error machinery onto the new operators is **not**
attempted in the first pass. Its feasibility is to be **revisited at the final
roadmap Milestone Review `019a`** (the end-of-Implementation review), informed by
the `019` performance-tuning evidence. Until then the two paths are independent:
old ops + old error machinery, new ops + constant-`P` stencil.

## Cache / Scratch Implications

- Radix-sort path: the operator cache (task `009` onward) needs no
  per-interaction-`P` scratch and no error-prediction-only buffers; the stencil
  offset set is precomputed once per level and shared across all cells.
- Legacy path: error-prediction scratch (rotation temporaries, `M̃` / `L̃`
  normalization, the partial `m = 0,1` z-translation used by
  `translate_multipole_to_local_z_m01_n`) must continue to coexist with operator
  scratch, as today.

## Non-Goals

- Does not derive the radix-sort clustering itself; that is task `008f`.
- Does not remove or alter the legacy dynamic-`P` machinery, error formulas, or
  `get_P` policy; the legacy path is preserved exactly.
- Does not finalize the exact conservative stencil bound or the Lamb-Helmholtz
  extension (both TODO, pending user input).
- Does not port the old error machinery onto the new operators; that feasibility
  is deferred to the end of the Implementation phase.

## Artifacts or Production Surface

This is a Theory Phase task. It must not modify production code under `src/`.

Artifacts:

- `theory/constant-p-error-stencil.md` — error-handling strategy and the
  conservative-stencil specification (with the TODO derivations above)
- `scripts/constant_p_error_stencil_verify.jl` — verification script
  (TODO-gated until the bound is filled in)
- `data/constant_p_error_stencil/verification_summary.md` — generated summary

## Verification

- **Legacy octree path:** unchanged from production; the existing
  dynamic-`P` parity expectations carry over. An operator-path M2L that performs
  the existing per-interaction error prediction must match production
  `FastMultipole.multipole_to_local!` for each supported error method, across
  representative offsets, tolerances, and both `Val(false)` / `Val(true)`.
- **Radix-sort path:** once the conservative bound and its Lamb-Helmholtz
  extension are filled in (TODO), the verification script must confirm that the
  stencil accepts exactly the cell offsets whose conservative error bound at the
  constant `P` is within tolerance, and that constant-`P` M2L over the stencil
  meets the target accuracy on a representative uniform-grid case.

Confirm no production `src/` code changed during this Theory task.

The generated summary is:

```text
MATRIX_OPERATOR_REFACTOR/data/constant_p_error_stencil/verification_summary.md
```

## Approval Notes

To be filled by a different agent after derivation, verification, and notes are
complete. Approval is blocked until the Lamb-Helmholtz subheading and the
conservative-bound TODOs are resolved.
