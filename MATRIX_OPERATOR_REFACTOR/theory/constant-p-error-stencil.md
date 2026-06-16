# Constant-`P` Error Stencil

## Scope

This artifact specifies the error-control split between the legacy octree path
and the new radix-sort path.

The legacy octree path remains unchanged: current production operators continue
to use the existing `get_P` and `predict_error` machinery in
`src/dynamic_expansion_order.jl` and `src/error.jl`. Per-interaction truncation
and all existing error-method behavior belong to that path.

The radix-sort path uses the matrix M2L operator chain at one constant expansion
order `P`. Error control is moved entirely into interaction-list construction by
using a reusable relative-offset stencil. The stencil is computed from a
conservative analytic bound and is independent of per-pair operator size.

## Uniform-Grid Geometry

Task `008f` defines a fixed-depth uniform grid with cell half-width `w` and cell
bounding-sphere radius:

```text
rho = w * sqrt(3)
```

For same-depth source and target cells with integer offset:

```text
d = target_coord - source_coord
```

the center-to-center distance is:

```text
R = 2w * norm(d)
```

and the normalized separation ratio is:

```text
c = R / rho = 2 * norm(d) / sqrt(3).
```

All same-depth pairs with the same `d` share this geometry, so a bound depending
only on `P`, level budgets, and `d` defines a translation-invariant stencil.

## Scalar Bound

Let `A = sum(abs(q_i))` be a configured conservative source-strength budget for
the source cell or level. For equal source and target cells, use:

```text
B(P, d, A) = 2A / (rho * (c - 2)) * (1 / (c - 1))^(P + 1)
```

where `c = R / rho`. This bound is valid only for:

```text
c > 2.
```

If `c <= 2`, or if the expression is not finite, the offset is rejected from the
M2L stencil and routed to near/direct handling.

The leading factor `2` combines conservative source-side and target-side
truncation budgets for equal-radius cells. This is intentionally more
conservative than fitting separate, data-dependent multipole and local tails.

## Production Normalization

The formulas above use analytic `1/r` normalization. When comparing the scalar
bound to production-normalized scalar potentials, multiply analytic `1/r` bounds
by:

```text
1 / (4*pi)
```

before applying the production tolerance. The stencil policy must state which
normalization its tolerance uses. Absolute analytic tolerances use `B` as
written; absolute production tolerances use `B / (4*pi)`.

## Stencil Acceptance

For a finite offset domain, accept offset `d` for scalar M2L if and only if the
configured bound is finite and:

```text
B(P, d, A) <= epsilon.
```

The finite domain is provided by the context that builds the list: a finite
uniform grid has only finitely many possible offsets, and a future
implementation may also impose a configured maximum offset radius when building
a reusable table. Offsets outside the finite domain are not part of that table;
inside the domain, the analytic predicate above is the complete accept/reject
rule.

Because `B` decreases as `P` increases, and because a larger `epsilon` relaxes
the acceptance test, accepted offsets are monotone under larger `P` and larger
tolerance within the same finite domain.

## Lamb-Helmholtz Channel

For `lamb_helmholtz = Val(true)`, apply the scalar bound independently to the
`phi` and `chi` source budgets:

```text
B_phi = B(P, d, A_phi)
B_chi = B(P, d, A_chi)
```

The local Lamb-Helmholtz stage couples the translated `chi` channel into the
`phi` channel and also couples neighboring `chi` degrees. The local transform
has same-degree factor `r * m / n` for `n > 0` and neighbor factor
`r / (n + 1)`. For a conservative first-pass stencil:

- use `m / n <= 1`;
- use `r / (n + 1) <= R`;
- combine the direct `chi` contribution, same-degree `phi` coupling, and
  neighbor `chi` coupling with the multiplier `1 + 2R`.

Unless channel-specific tolerances are supplied, use the combined bound:

```text
B_LH(P, d, A_phi, A_chi) = B(P, d, A_phi) + (1 + 2R) * B(P, d, A_chi).
```

Accept `Val(true)` offsets if and only if `B_LH` is finite and:

```text
B_LH <= epsilon.
```

If channel-specific tolerances are supplied, an implementation may instead
compare the `phi` and `chi` channel bounds against their own tolerances, but the
default first-pass policy is the combined scalar tolerance above.

## Implementation Consequences

- The old production octree path remains the owner of dynamic `P`.
- The radix path never calls `get_P` or `predict_error` during M2L batching.
- The matrix M2L operators are sized by one constant `P`.
- The accepted offset set is reusable for every cell sharing the same grid
  level, source budgets, tolerance, normalization, and Lamb-Helmholtz setting.
- Rejected offsets, including all offsets with `c <= 2`, are routed to
  near/direct handling.

Tighter stencil fitting, Dehnen-style queueing, level-transcending adaptive
lists, and porting the old dynamic-`P` machinery onto the matrix operators are
deferred follow-on decisions.
