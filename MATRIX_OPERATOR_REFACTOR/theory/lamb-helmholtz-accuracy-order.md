# Lamb-Helmholtz Accuracy Order

## Scope

This artifact derives the relative expansion orders for the two
Lamb-Helmholtz channels in the matrix M2L path. It is limited to M2L and final
local evaluation. M2M and L2L use the same channel metadata and buffers, but
their detailed error behavior is not re-derived here.

Let `P_phi` be the requested physical expansion order for the scalar `phi`
channel. For `lamb_helmholtz = Val(true)`, the minimal balanced channel rule is:

```text
P_chi = P_phi + 1.
```

For `lamb_helmholtz = Val(false)`, there is no `chi` channel and the existing
single-order `P` behavior is unchanged.

## Local M2L Coupling

The approved M2L composition applies the local Lamb-Helmholtz transform after
the channel-independent z M2L block:

```text
phi_tilde_n^m = phi_hat_n^m + i * r * m / n * chi_hat_n^m,  n > 0
phi_tilde_0^0 = phi_hat_0^0
chi_tilde_n^m = chi_hat_n^m - r / (n + 1) * chi_hat_{n + 1}^m.
```

The `phi` output at degree `n` uses same-degree `chi_n^m`. The `chi` output at
degree `n` uses both `chi_n^m` and the upper neighbor `chi_{n+1}^m`.

Therefore, if the retained physical local `chi` output must be correct through
degree `P_phi`, the pre-transform `chi_hat` input must be present through degree
`P_phi + 1`. Truncating `chi` at `P_phi` silently sets `chi_hat_{P_phi+1}^m = 0`
and corrupts the highest retained `chi_tilde_{P_phi}^m` row.

## Evaluation Coupling

Task `008e` derives the evaluated Lamb-Helmholtz gradient:

```text
G_LH(phi, chi)_n uses phi_{n+1} and chi_n.
```

The `phi` contribution to a degree-`n` vector-field coefficient is sourced from
degree `n + 1`; the `chi` contribution is sourced from degree `n`. This means
that carrying `chi` only through `P_phi` makes the evaluated field read the
degree that was just degraded by the local transform. Carrying `chi` through
`P_phi + 1` moves the first missing `chi` contribution one degree above the
retained `phi` order, matching the normal one-degree derivative shift already
present in the `phi` path.

This artifact does not change the deferred derivative-aware question from
`008e` of whether `phi` itself should be carried above the requested order for
exact top-degree gradients or Hessians. It only fixes the relative `chi` order
when `P_phi` is the requested physical order.

## Candidate Rules

The tested candidate offsets were:

```text
P_chi = P_phi - 1
P_chi = P_phi
P_chi = P_phi + 1
P_chi = P_phi + 2
```

`P_phi - 1` is rejected because it truncates a degree that the final
Lamb-Helmholtz gradient reads directly.

`P_phi` is rejected because the local transform's
`chi_hat_{n+1} -> chi_tilde_n` coupling is incomplete at `n = P_phi`. The
highest retained `chi` degree is then systematically a boundary artifact.

`P_phi + 1` is accepted. It supplies every `chi_hat` degree needed to compute
all retained `chi_tilde_n` rows for `0 <= n <= P_phi`, while leaving the first
unavailable `chi` contribution at degree `P_phi + 1`.

`P_phi + 2` is rejected as the default rule because it does not repair any
additional coupling needed by retained degree-`P_phi` M2L or evaluation. It may
reduce error constants in some cases, but it is extra work rather than a
minimal consistency requirement.

## Relation To Task 008d

Task `008d` and `theory/constant-p-error-stencil.md` used a single `P` in the
first-pass Lamb-Helmholtz stencil:

```text
B_phi = B(P, d, A_phi)
B_chi = B(P, d, A_chi)
```

This artifact supersedes only that channel-order part for `Val(true)`. The
default constant-order LH stencil should interpret `P` as `P_phi` and evaluate
the channel tails as:

```text
B_phi = B(P_phi, d, A_phi)
B_chi = B(P_phi + 1, d, A_chi)
B_LH = B_phi + (1 + 2R) * B_chi.
```

The geometric accept/reject structure from `008d` is unchanged: offsets with
`c <= 2` still route to near/direct, and accepted offsets still satisfy the
configured combined tolerance. The changed `chi` order makes the `chi` bound
slightly smaller for a fixed `P_phi`, so it can only preserve or modestly relax
the accepted set relative to a same-order `chi` tail.

## Buffer And Operator Consequences

For `Val(true)`, operator metadata must distinguish:

```text
P_phi = requested physical order
P_chi = P_phi + 1
P_active = P_chi
```

Channel-independent rotation, axis-swap, and z-translation stages may run over
the padded active basis through `P_active`, with the `phi` rows above `P_phi`
treated as padding or scratch that is not part of the requested physical result.
The alternative is ragged per-channel matrices, but that complicates batching
and channel-coupled Lamb-Helmholtz application. The preferred first
implementation is a uniform active basis sized to `P_active` for `Val(true)`,
with explicit metadata marking the valid physical `phi` rows.

The local Lamb-Helmholtz operator must include the
`chi_{P_phi+1} -> chi_{P_phi}` neighbor row. Any implementation that allocates
only through `P_phi` for the `chi` channel loses that row.

For `Val(false)`, cache keys, buffers, z blocks, rotations, and evaluation remain
sized by the single requested order `P`.

## Numerical Verification

`MATRIX_OPERATOR_REFACTOR/scripts/lamb_helmholtz_accuracy_order_verify.jl` builds
high-order production M2L references at `P_REF = 16`, then compares reduced
channel policies across deterministic source coefficient clouds, source/target
centers, M2L separation ratios, and target offsets. For each configuration it
measures the induced gradient error from:

- truncating `phi` while keeping `chi` high;
- truncating `chi` while keeping `phi` high;
- paired policies `P_chi = P_phi + delta` for `delta in {-1, 0, 1, 2}`.

The generated summary reports max, RMS, and median errors, convergence slopes
versus `P_phi`, the selected rule, and the scalar `Val(false)` unchanged check:

```text
MATRIX_OPERATOR_REFACTOR/data/lamb_helmholtz_accuracy_order/verification_summary.md
```

The verification result is `PASS` for `P_chi = P_phi + 1`. The `delta = 1`
policy improves over same-order `chi`, converges at least as fast as the lower
offsets, and `delta = 2` does not provide a large enough additional reduction
to justify the extra default channel degree.
