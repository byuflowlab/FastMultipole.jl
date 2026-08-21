# 008b Implementation Re-Plan

## Objective

Revisit the Implementation roadmap after Theory is complete and before any
production code work begins.

## Dependencies

- All Theory Phase rows in `START_HERE.md`
- `008a-milestone-review-theory-005-008.md`

## Required Reading

- `START_HERE.md`
- `../MATRIX_OPERATOR_REFACTOR.md`
- Approved Theory task files
- Approved Theory artifacts listed by the Theory task files

## Artifacts or Production Surface

This is a user-in-the-loop planning task. No production code changes may start
until this task is complete and approved.

Planning updates may edit Implementation task files and/or `START_HERE.md` if the
approved Theory results require a changed implementation sequence.

## Deliverables

- Interactive implementation-strategy discussion with the user
- Decision on whether the existing Implementation task sequence still fits the
  approved Theory results
- Any needed updates to Implementation task files and/or `START_HERE.md`
- Chosen implementation strategy recorded in this file
- Open tradeoffs and deferred alternatives recorded in this file
- Clear-context approval before task `009` or any later Implementation task
  starts

## Re-Plan Notes

Conducted as a user-in-the-loop session on 2026-06-13. Reviewed `START_HERE.md`,
`../MATRIX_OPERATOR_REFACTOR.md`, the approved Theory task files and artifacts
(`005`–`008` and their `theory/` derivations, `scripts/`, and `data/`), the
existing Implementation task files `009`–`019`, and the current production
translation/rotation/evaluation surface (`src/rotate.jl`, `src/translate.jl`,
`src/containers.jl`, `src/evaluate_expansions.jl`, `src/error.jl`,
`src/dynamic_expansion_order.jl`) read-only.

### Finding: the approved Theory has two gaps for production parity

1. The approved M2L composition (`005`) specifies only the **fixed-`P`** chain.
   Production `multipole_to_local!` selects `P` per interaction from a predicted
   error bound (`Nothing` / `RotatedCoefficientsAbsoluteGradient` /
   `PowerAbsoluteGradient`), using `local_power`, `M̃`/`L̃`, and the partial
   `m = 0, 1` z-translation (`translate_multipole_to_local_z_m01_n`) before the
   accumulating back-rotation in `multipole_to_local_II!`. How the operator layer
   integrates with this dynamic-`P` machinery is unspecified.
2. The approved real-basis evaluation (`008`) derives only the **scalar
   potential**. Production evaluates potential, gradient, and the gradient
   Jacobian (Hessian) via `DerivativesSwitch{PS,GS,HS}` (buffer rows `4`, `5:7`,
   `8:16`). The real basis lacks derivative-evaluation theory.

## Chosen Strategy

The existing Implementation sequence `009`–`019` is retained. The re-plan makes
the following decisions (all confirmed with the user this session):

1. **Iterate Theory — two new full Theory tasks** (derivation + verify script +
   generated data + separate clear-context approval, matching `005`–`008`),
   recorded in the new `START_HERE.md` "Theory Addendum" subsection. Both are
   Theory Phase rows and, per the standard hard phase gate, **block every
   Implementation task**:
   - `008d-theory-dynamic-p-error-m2l-integration.md` — operator/dynamic-`P`
     error-prediction integration. Listed as a relevant dependency on M2L
     Implementation tasks `011` and `014`.
   - `008e-theory-real-basis-kernel-derivatives.md` — real-basis potential,
     gradient, and gradient Jacobian (Hessian) evaluation for the `1/r` kernel.
     Listed as a relevant dependency on real-basis Implementation task `018`.
2. **Operator form (dense-materialized matrices vs recurrence-wrapped
   operators): deferred to `008c`.** Added as an explicit per-stage `008c`
   deliverable with benchmark-backed rationale; Implementation tasks `009`–`016`
   inherit the decision (they already depend on `008c`).
3. **Hot-path swap policy: side-by-side, parity-only in the first pass.**
   Tasks `014` and `016` build and validate the explicit operator pipelines
   against production but do **not** replace the production `multipole_to_*!`
   internals. Production hot-path replacement is a later, explicitly scoped step
   (informed by `019` tuning evidence).
4. **Storage staging unchanged.** Operators are built against the existing 3D
   `weights[real_or_imag, component, harmonic_index]` array first; flat
   `basis_dof x batch x channel` buffers remain task `017`; real basis remains
   task `018`.

### Coordination-document updates made by this task

- Created `008d-theory-dynamic-p-error-m2l-integration.md` and
  `008e-theory-real-basis-kernel-derivatives.md`.
- `START_HERE.md`: added the "Theory Addendum" subsection (`008d`, `008e`);
  amended the Hard Phase Gate and Implementation Phase text to include the
  addendum rows as full blockers.
- `008c-implementation-performance-baseline.md`: added the per-stage operator-form
  decision as a deliverable.
- `011`, `014`: added `008d` to Dependencies/Required Reading. `014`, `016`:
  recorded the side-by-side parity-only constraint. `018`: added `008e`
  dependency and the potential/gradient/Jacobian real-basis evaluation
  deliverable.

## Open Tradeoffs and Deferred Alternatives

- **Operator form** is unresolved by design; `008c` chooses dense vs recurrence
  per stage. Risk: a dense path must beat the current `O(p)` recurrences to
  justify itself; a recurrence-wrapped path postpones the BLAS/GPU payoff to the
  flat-buffer (`017`) and real-basis (`018`) work.
- **Production hot-path replacement** is deferred out of the first pass. The
  operator layer coexists with production until parity and performance evidence
  justify swapping `multipole_to_*!` internals.
- **Theory blocking scope.** An option to scope `008d`/`008e` to only their
  relevant Implementation tasks was considered and explicitly rejected by the
  user in favor of the standard "every Theory row blocks every Implementation
  task" gate.
- **Flat-buffer and real-basis staging** (`017`/`018`) are unchanged; the
  `lda`-padded slab and planar re/im variants remain deferred per `theory/
  coefficient-buffer-layout.md`.

## Verification

- No production code work has started for task `009` or any later Implementation
  task. All edits from this re-plan live under `MATRIX_OPERATOR_REFACTOR/`
  (coordination documents and the two new Theory task scaffolds); the derivation
  content for `008d`/`008e` is owned by those tasks, not by `008b`.
- `git status` shows no `src/` changes attributable to this task.
- The chosen strategy, open tradeoffs, deferred alternatives, and
  coordination-document updates are recorded above.
- This task is marked `Done` (not `Approved`) in `START_HERE.md`; a different
  clear-context agent must approve before task `009` or any later Implementation
  task begins, and before `008d`/`008e` are themselves approved.

## Approval Notes

Approved on 2026-06-13 by a different clear-context agent. Reviewed
`START_HERE.md`, `../MATRIX_OPERATOR_REFACTOR.md`, the completed `008b` notes,
and the coordination/task files touched by the re-plan, including `008c`,
`008d`, `008e`, `011`, `014`, `016`, and `018`.

Confirmed that `008b` records the chosen implementation strategy, deferred
tradeoffs and alternatives, and document-only verification. Confirmed the new
`008d` and `008e` Theory Addendum rows are full hard-gate blockers before any
Implementation task starts. Confirmed no production code changes are present in
`src/`, `test/`, `examples/`, `benchmark/`, or `benchmarks/`.

Conclusion: `008b` is approved.

## Re-Plan Addendum (2026-06-13): constant-`P` error handling and radix-sort clustering

Conducted as a user-in-the-loop session. The user redirected the error-handling
strategy and introduced a new clustering workstream. This addendum expands — and
does not contradict — the original re-plan: the "preserve current dynamic-`P`
behavior exactly" intent still holds, now scoped explicitly to the legacy octree
path.

### Decisions

1. **Two error-handling paths.**
   - *Legacy octree path*: the existing dynamic-`P` / error-prediction machinery
     (`get_P`, `predict_error`, per-interaction truncation in
     `multipole_to_local!`) is preserved exactly. This is the original `008d`
     intent, retained.
   - *New radix-sort path*: **constant expansion order `P`** everywhere, with
     error control moved into interaction-list construction. M2L is performed
     only between cells whose conservative error bound at the constant `P` is
     within tolerance, via a translation-invariant interaction-list **stencil**.
     This lets the task-`005` M2L operator chain run at fixed size with no
     per-interaction truncation and no dynamic-`P` integration on that path.

2. **Stencil mechanism (option c).** Three options were recorded in `008d`:
   (a) a threshold-fitted, possibly level-transcending list; (b) Dehnen
   multipole-power selection at fixed `P` with batched M2L (risk: queue
   construction becomes the bottleneck); (c) a conservative cheap error bound →
   traditional translation-invariant stencil. **Chosen: (c)**, because it
   amortizes interaction-list construction and is GPU-batch-friendly. The
   conservative bound starts from the Greengard–Rokhlin multipole truncation
   bound; the exact bound and its Lamb-Helmholtz `χ`-channel extension are TODOs
   the user will fill in (`008d` is not finished until the Lamb-Helmholtz
   subheading is resolved).

3. **New Theory Addendum row `008f-theory-radix-sort-clustering.md`.** Derives a
   radix-sort (Morton/Z-order) clustering for large-`N`/GPU producing uniform-grid
   cells over which the `008d` stencil operates. It is a **full hard-gate
   blocker** for every Implementation task (confirmed with the user), consistent
   with `008d`/`008e`. `008d` now **depends on `008f`** and is listed after it in
   the Theory Addendum table.

### Coordination-document updates made by this addendum

- Rewrote `008d` to the constant-`P` / interaction-list-stencil scope (legacy vs
  radix-sort paths; options a/b/c with decision (c); Greengard–Rokhlin bound;
  Lamb-Helmholtz TODO subheading; non-goals; revised artifacts/verification).
- Created `008f-theory-radix-sort-clustering.md` (scaffold; all derivations TODO).
- `START_HERE.md`: added the `008f` row before `008d` in the Theory Addendum;
  updated the `008d` summary and Blocking column (added `008f`); amended the
  Hard Phase Gate and Implementation Phase prose to list `008d`, `008e`, `008f`.

4. **Old / new operator coexistence.** Both operator implementations stay in the
   repository. The old operators remain compatible with the old error machinery
   (legacy octree path); the new operators are not wired to it and run at
   constant `P` with the stencil. Porting the old error machinery onto the new
   operators is deferred; its feasibility is **revisited at the final roadmap
   Milestone Review `019a`** (the end-of-Implementation review), informed by `019`
   performance-tuning evidence.

### Open items

- The exact conservative stencil bound and its Lamb-Helmholtz extension remain
  under discussion with the user (TODOs in `008d`).
- The full radix-sort clustering derivation is owned by `008f` (all TODO).
- Feasibility of porting the old error machinery onto the new operators is to be
  revisited at the final roadmap Milestone Review `019a`; recorded there as a
  deliverable.
