# 008h Theory Lamb-Helmholtz Channel Accuracy Order

## Objective

Derive the expansion-order relationship the two Lamb-Helmholtz channels must
satisfy for **consistent accuracy**. The working hypothesis, to be confirmed or
corrected by derivation, is:

> For a target accuracy of order `P` in the scalar `φ` channel (and in the
> derived potential/gradient/Hessian), the Lamb-Helmholtz `χ` channel must be
> carried to order `P + 1`, while `φ` remains at `P`.

The motivation is the inter-degree coupling of the Lamb-Helmholtz transforms:
the local (and multipole) `χ → φ` coupling derived in `003` and reused by the
`008d` stencil couples `χ` at neighboring degree (`n + 1`) into degree-`n` `φ`.
If `χ` is truncated at the same `P` as `φ`, the highest retained `φ` degrees may
be fed by an already-truncated `χ` tail, degrading the effective order. This
task determines the precise, minimal order each channel needs and whether
`P_χ = P_φ + 1` is necessary and sufficient.

This is a Theory Phase task added by user request on 2026-06-15. Like every
Theory Addendum row, it is a full hard-gate blocker for every Implementation
task: it must be both `Done` and `Approved` before any Implementation row starts.
It is independent of `008g` (interaction-list coverage); the two may be derived
in either order.

## Dependencies

- `003-theory-lamb-helmholtz-operator-form.md` (the `χ`/`φ` operator form and
  same-degree / neighboring-degree channel coupling)
- `005-theory-full-m2l-composition.md` (the fixed-`P` M2L chain whose channel
  sizes this task may revise)
- `008d-theory-dynamic-p-error-m2l-integration.md` (the constant-`P` stencil and
  its `B_LH` channel bound, which assumes per-channel budgets)
- `008e-theory-real-basis-kernel-derivatives.md` (potential/gradient/Hessian
  evaluation, so accuracy is judged on the quantities actually returned)
- `007-theory-coefficient-buffer-layout.md` (buffer sizing if the channels carry
  different orders)
- `008b-implementation-replan.md`

## Required Reading

- `START_HERE.md`
- Approved dependency task files listed above
- Approved artifacts `theory/lamb-helmholtz-operator-form.md`,
  `theory/full-m2l-composition.md`, `theory/constant-p-error-stencil.md`, and
  `theory/real-basis-kernel-derivatives.md`
- Current production Lamb-Helmholtz code (read-only, no edits):
  - `src/translate.jl`: the `χ`-channel transforms and `φ`/`χ` coupling in
    `multipole_to_local!` / the Lamb-Helmholtz stages
  - `src/evaluate_expansions.jl`: how `χ` contributes to evaluated gradient
  - `test/lamb_helmholtz_test.jl`: existing LH parity expectations

## Scope / Deliverables

- **Coupling-order analysis.** From the `003` local/multipole Lamb-Helmholtz
  transforms, identify exactly which `χ` degrees feed each `φ` degree (same-degree
  `n` and neighbor `n ± 1`, per `003`/`008d`), and which `φ`/`χ` degrees feed the
  evaluated potential, gradient, and Hessian (via `008e`). Establish the highest
  `χ` degree required so that every retained `φ` degree up to `P` is fed by a
  fully populated (untruncated) `χ` contribution.
- **Order relationship.** State and justify the channel-order rule. Confirm,
  refine, or refute `P_χ = P_φ + 1`. The derivation focuses on M2L and final
  evaluation; M2M / L2L-specific behavior is out of scope except for shared
  cache and buffer sizing consequences.
- **Stencil consistency with `008d`.** Reconcile the result with `008d`'s `B_LH`
  bound. If `χ` is carried at `P + 1`, state whether the conservative stencil
  should evaluate the `χ` truncation tail at `P + 1` (i.e. the `B_chi` term uses
  `P + 1`) while the `φ` tail uses `P`, and whether this changes accept/reject.
- **Buffer / operator-sizing consequences.** Specify the consequences for `007`
  buffer layout and `005`/`003` operator sizes when the two channels carry
  different orders: padded-to-`P+1` uniform storage vs ragged per-channel sizing,
  and the z-block / rotation operator dimensions implied.
- **Scalar-only path unaffected.** For `lamb_helmholtz = Val(false)` there is no
  `χ` channel; the rule must reduce to the existing single-order-`P` behavior with
  no change.

## Non-Goals

- Does not modify production `src/` code (Theory Phase).
- Does not re-derive the Lamb-Helmholtz operator form itself; that is `003`.
- Does not change the legacy dynamic-`P` octree path's existing behavior; this
  rule governs the constant-`P` radix path's channel sizing. Whether the legacy
  path also benefits is a note, not a required change.
- Does not select final storage layout; it states the sizing consequences for the
  implementation tasks to act on.

## Artifacts or Production Surface

This is a Theory Phase task. It must not modify production code under `src/`.

Artifacts:

- `theory/lamb-helmholtz-accuracy-order.md` — derivation of the channel-order
  rule, rejected candidates, coupling-degree analysis,
  stencil/buffer/operator consequences, and the scalar-path reduction.
- `scripts/lamb_helmholtz_accuracy_order_verify.jl` — deterministic verifier
  using existing FastMultipole M2L and evaluation machinery. It builds a
  high-order reference, compares reduced `φ` and `χ` channel policies over many
  source/target configurations, reports aggregate gradient errors and
  convergence slopes, and confirms the `Val(false)` scalar path is unchanged.
- `data/lamb_helmholtz_accuracy_order/verification_summary.md` — generated
  summary (convergence-slope table for `P_χ = P` vs `P_χ = P + 1`, the deduced
  rule, and PASS/FAIL).

## Verification

Run:

```sh
julia --project=. MATRIX_OPERATOR_REFACTOR/scripts/lamb_helmholtz_accuracy_order_verify.jl
```

The verifier confirms the derived channel-order rule: the convergence study
shows consistent accuracy under the derived `χ` order, same-order `χ` is a
rejected candidate, `P_χ = P_φ + 2` does not justify its extra default degree,
the result agrees with the analytic coupling-degree argument, and the scalar
`Val(false)` path is unaffected.

Completed result: the final rule is `P_χ = P_φ + 1` for `Val(true)` M2L and
evaluation, while `Val(false)` remains single-order `P`.

Confirm no production `src/` code changed during this Theory task.

The generated summary is:

```text
MATRIX_OPERATOR_REFACTOR/data/lamb_helmholtz_accuracy_order/verification_summary.md
```

## Approval Notes

Completion notes:

- Added `theory/lamb-helmholtz-accuracy-order.md`.
- Added and ran `scripts/lamb_helmholtz_accuracy_order_verify.jl`.
- Generated `data/lamb_helmholtz_accuracy_order/verification_summary.md` with
  status `PASS`.
- Updated downstream implementation task notes for the final `P_χ = P_φ + 1`
  sizing rule.

Clear-context approval (2026-06-15, separate reviewing agent — not the
completing agent):

- **Approved.** Theory gate verified: no production `src/` code changed
  (`git diff -- src/` empty, no untracked `src/` files); only `theory/`,
  `scripts/`, `data/`, and downstream task-note files were touched.
- Derivation cross-checked against the approved `003`
  (`theory/lamb-helmholtz-operator-form.md`): the local M2L coupling
  `chi_tilde_n^m = chi_hat_n^m - r/(n+1) * chi_hat_{n+1}^m` matches exactly, and
  with the `008e` evaluation coupling (`phi_{n+1}`, `chi_n`) the coupling-degree
  argument correctly yields the minimal balanced rule `P_chi = P_phi + 1`.
- All Scope deliverables present: coupling-order analysis, candidate rules with
  rejections (`-1`, `0`, `+2`), `008d` stencil reconciliation (`B_chi` at
  `P_phi + 1`), buffer/operator sizing (`P_active = P_chi`), and the `Val(false)`
  reduction.
- Verifier `scripts/lamb_helmholtz_accuracy_order_verify.jl` uses production
  `multipole_to_local!`/`evaluate_local`; generated summary reports `PASS`
  (`delta_1` RMS = 0.577× `delta_0`, `delta_2` = 0.85× `delta_1`, scalar
  `Val(false)` unchanged). Downstream notes (`009/011/012/014/017/018`)
  reference `008h` and the `P_chi = P_phi + 1`/`P_active` rule.
- Non-blocking note: convergence slopes across deltas are nearly identical
  (≈ −1.75/−1.78/−1.80); the rule rests on the coupling derivation plus a
  constant-factor error reduction rather than a rate change, consistent with the
  artifact's own framing.

Approval is independent of `008g` but, like all Theory rows, must precede any
Implementation task.
