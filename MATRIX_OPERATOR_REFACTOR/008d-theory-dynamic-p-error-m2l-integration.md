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

### Conservative error bound

The adopted radix-path bound is specified in
`theory/constant-p-error-stencil.md`. It starts from the original
Greengard–Rokhlin multipole truncation bound. For a source cell with
bounding-sphere radius `rho` about its center and total source strength
`A = sum(abs(q_i))`, an order-`P` truncated multipole expansion evaluated at a
point a distance `r > rho` from the source center satisfies

```
|epsilon_multipole| <= A / (r - rho) * (rho / r)^(P + 1)
```

with the dual bound on the target (local-expansion) side.

On the uniform radix grid from `008f`, a cell has half-width `w`, radius
`rho = w * sqrt(3)`, center distance `R = 2w * norm(d)` for integer offset `d`,
and normalized separation:

```text
c = R / rho = 2 * norm(d) / sqrt(3).
```

For equal source and target cells, the conservative scalar stencil bound is:

```text
B(P, d, A) = 2A / (rho * (c - 2)) * (1 / (c - 1))^(P + 1).
```

The bound is valid only for `c > 2`; offsets with `c <= 2` are rejected from the
M2L stencil and routed to near/direct handling. For `c > 2`, accept offset `d`
if and only if the configured bound is finite and `B(P, d, A) <= epsilon`.

The formulas above use analytic `1/r` normalization. When comparing against
production-normalized scalar potentials, multiply analytic bounds by
`1 / (4*pi)` before applying a production-normalized tolerance.

### Lamb-Helmholtz (`χ`-channel) extension

For `lamb_helmholtz = Val(true)`, the first-pass conservative stencil applies
the scalar bound independently to configured `phi` and `chi` source budgets:

```text
B_phi = B(P, d, A_phi)
B_chi = B(P, d, A_chi)
```

The local Lamb-Helmholtz transform has same-degree `chi` to `phi` coupling and
neighboring-degree `chi` coupling. The stencil accounts for these with
`m / n <= 1` and `r / (n + 1) <= R`, where `R = 2w * norm(d)`, giving the
combined default bound:

```text
B_LH(P, d, A_phi, A_chi) =
    B(P, d, A_phi) + (1 + 2R) * B(P, d, A_chi).
```

Unless a future implementation supplies channel-specific tolerances, accept
`Val(true)` offsets if and only if `B_LH` is finite and `B_LH <= epsilon`.

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
- Does not fit tighter, data-dependent stencils; the selected bound is a
  conservative analytic first pass.
- Does not port the old error machinery onto the new operators; that feasibility
  is deferred to the end of the Implementation phase.

## Artifacts or Production Surface

This is a Theory Phase task. It must not modify production code under `src/`.

Artifacts:

- `theory/constant-p-error-stencil.md` — error-handling strategy and the
  conservative-stencil specification
- `scripts/constant_p_error_stencil_verify.jl` — verification script
- `data/constant_p_error_stencil/verification_summary.md` — generated summary

## Verification

- **Legacy octree path:** unchanged from production; the existing
  dynamic-`P` parity expectations carry over. An operator-path M2L that performs
  the existing per-interaction error prediction must match production
  `FastMultipole.multipole_to_local!` for each supported error method, across
  representative offsets, tolerances, and both `Val(false)` / `Val(true)`.
- **Radix-sort path:** the verification script confirms the uniform-grid
  geometry mapping, rejection for `c <= 2`, exact agreement between the analytic
  acceptance predicate and generated stencil offsets, monotonicity under larger
  `P` and looser `epsilon`, production-normalized scaling, and both
  `Val(false)` / `Val(true)` paths.

Verifier command:

```text
julia --project=. MATRIX_OPERATOR_REFACTOR/scripts/constant_p_error_stencil_verify.jl
```

Result:

```text
constant_p_error_stencil_verify: PASS
summary: MATRIX_OPERATOR_REFACTOR/data/constant_p_error_stencil/verification_summary.md
```

Confirm no production `src/` code changed during this Theory task.

The generated summary is:

```text
MATRIX_OPERATOR_REFACTOR/data/constant_p_error_stencil/verification_summary.md
```

## Approval Notes

Derivation, verification, and notes are complete for this task. A different
agent must perform clear-context approval before the row is marked Approved.

**Clear-context approval (2026-06-13).** A different agent, in a fresh context,
performed clear-context approval per the `START_HERE.md` protocol. The reviewer
read only `START_HERE.md`, this task file, and the three listed artifacts
(`theory/constant-p-error-stencil.md`,
`scripts/constant_p_error_stencil_verify.jl`,
`data/constant_p_error_stencil/verification_summary.md`).

Findings:

- The conservative scalar bound is the Greengard–Rokhlin multipole-truncation
  bound evaluated at a conservatively reduced separation (`c → c - 1`, absorbing
  target-cell extent) and doubled for the source and target sides; it is valid
  for `c > 2`, with `c <= 2` offsets correctly routed to near/direct. The
  Lamb-Helmholtz combined bound and the `1 / (4π)` production normalization are
  documented and match the script.
- The verification script implements the documented formulas exactly and tests
  geometry mapping, `c <= 2` rejection, analytic/generated stencil agreement,
  monotonicity under larger `P` and looser `ε`, production-normalized scaling,
  and both `Val(false)` / `Val(true)` paths.
- The verifier was re-run during approval and printed
  `constant_p_error_stencil_verify: PASS`.
- `git status` / `git diff` confirm no production `src/` code changed; all work
  is confined to `MATRIX_OPERATOR_REFACTOR/`.

Approved. The `008d` row is marked Approved in `START_HERE.md`.
