# Full M2L Composition

## Scope

This artifact composes the approved component operators into the complete
production-parity multipole-to-local path for the current compressed complex
solid harmonic basis. It is intentionally theory-only: it specifies ordering,
normalization, cache requirements, and verification targets for later
implementation work.

The production vector from source center to target center is

```text
Delta x = target_center - source_center
(r, theta, phi) = cartesian_to_spherical(Delta x)
```

The only angle-dependent rotations in the operator chain are z-axis phase
operators. All non-z rotation effects are represented by invariant axis-swap
matrices and fixed sign tables.

## Channel Layout

The coefficient storage remains

```julia
weights[real_or_imag, component, harmonic_index(n, m)]
harmonic_index(n, m) = n * (n + 1) / 2 + m + 1
```

with stored orders `0 <= m <= n <= P`.

For `Val(false)`, only component `1` is active. Component `2` is outside the
active layout and should not be read as scalar M2L input. Forward y alignment
and back y alignment reset their destination buffers in production, so inactive
component entries in temporary y buffers become zero.

For `Val(true)`, components `1` and `2` are active. Component `1` stores the
Lamb-Helmholtz `phi` channel and component `2` stores the `chi` channel. The
z-rotation, axis-swap, and z-M2L translation act identically on both channels;
the local Lamb-Helmholtz transform is the only M2L stage that couples channels.

## Operator Chain

Let `M` be the source multipole expansion and `L_target` be the target local
expansion before accumulation. The production-parity chain is:

```text
M1 = Z_phi M                         # forward z rotation, overwrite
M2 = S_M Z_theta S_M^-1 M1           # multipole axis swap, zeta signs, reset
L1 = K_z(r) M2                       # fixed-m M2L z translation, overwrite
L2 = H_L(r) L1                       # optional Val(true) local LH transform
L3 = S_L Z_theta S_L^-1 L2           # local axis swap back, eta signs, reset
L_target <- L_target + Z_phi^-1 L3   # inverse z rotation, accumulate
```

For `Val(false)`, `H_L(r)` is the identity and only component `1` participates.
For `Val(true)`, `H_L(r)` is the approved local Lamb-Helmholtz operator.

The forward z phase is

```text
Z_phi(n, m) = exp(i m phi)
```

on each stored complex coefficient. The final inverse z phase is its conjugate
and is the only accumulating step:

```text
Z_phi^-1(n, m) = exp(-i m phi).
```

The axis-swap operator is the invariant composition approved in the axis-swap
theory:

```text
T_n(theta) = S_n Z_n(theta) S_n^-1.
```

For the M2L source-side multipole alignment, the compressed-basis contribution
signs use the `zeta` table. For the local return alignment, they use the `eta`
table. These sign tables are part of the operator metadata and are not optional
phase conventions.

The production note that y alignment includes an additional `pi` rotation about
the new z axis is preserved by the selected invariant axis-swap convention and
sign tables. A later implementation should not remove that phase as redundant.

## Fixed-m M2L z Blocks

For each stored order `m`, the z-aligned M2L translation is independent over
the degree indices. For `m <= n <= P` and `m <= np <= P`,

```text
L_n^m = sum_{np=m}^P K_m[n, np] M_np^m
K_m[n, np] = (n + np)! / r^(n + np + 1).
```

The same real scalar multiplies the real and imaginary lanes, and for
`Val(true)` the same block is applied independently to both active components
before the local Lamb-Helmholtz transform.

The recurrence-compatible scaled form required for stable cache design is:

```text
K_m[n, np] = D_L[n] * Khat_m[n, np] * D_M[np]
D_L[n] = n! / r^(n + 1)
Khat_m[n, np] = binomial(n + np, n)
D_M[np] = np! / r^np
```

This is algebraically identical to the approved recurrence because

```text
D_L[n] * Khat_m[n, np] * D_M[np]
= n! / r^(n + 1) * (n + np)! / (n! np!) * np! / r^np
= (n + np)! / r^(n + np + 1).
```

The scaled factors belong to M2L operator metadata. They must not alter the
coefficient-buffer basis seen by z rotations, axis swaps, or Lamb-Helmholtz
operators.

## Local Lamb-Helmholtz Stage

For `Val(true)`, after z M2L translation and before local axis-swap return,
apply the approved local transform. In complex channel notation:

```text
phi_tilde_n^m = phi_hat_n^m + i * r * m / n * chi_hat_n^m,  n > 0
phi_tilde_0^0 = phi_hat_0^0
chi_tilde_n^m = chi_hat_n^m - r / (n + 1) * chi_hat_{n + 1}^m
chi_hat_{P + 1}^m = 0.
```

The transform overwrites its destination. It is sparse in degree, does not mix
different `m`, and is parameterized by the same distance `r` used by the z M2L
blocks.

## Buffer Semantics

The complete M2L pipeline has three distinct write modes:

- Forward `Z_phi`: overwrite active channels.
- Axis swaps, z M2L, and optional local Lamb-Helmholtz transform: overwrite or
  reset-then-write temporary buffers.
- Final `Z_phi^-1`: accumulate into the target local expansion.

This separation is part of the parity target. In particular, the fixed-m z M2L
operator overwrites its local temporary buffer; it does not accumulate directly
into the tree target.

## Cache Requirements

A production operator cache for this composition needs:

- z phase vectors for `phi` and `theta`, or reusable phase recurrences keyed by
  expansion order and angle;
- invariant multipole axis-swap matrices `S_M`, inverse metadata, and `zeta`
  sign tables;
- invariant local axis-swap matrices `S_L`, inverse metadata, and `eta` sign
  tables;
- fixed-`m` M2L z-translation blocks or recurrence metadata for every
  `0 <= m <= P`, including the scaled factors `D_L`, `Khat`, and `D_M` when the
  scaled application path is used;
- local Lamb-Helmholtz operator metadata for `Val(true)`, including distance
  `r`, same-degree phi-from-chi coupling, and upper-neighbor chi coupling;
- layout metadata selecting one active component for `Val(false)` and two
  active components for `Val(true)`.

The cache may store matrices densely or as structured recurrence factors, but
the applied operator must preserve the current compressed coefficient ordering.

## Unit Point-Mass M2L Component

For a unit point mass at `x_s` represented about source center `c_s`, define
the regular harmonic

```text
R_n^m(x_s - c_s)
= (-1)^n i^m rho^n P_n^m(cos theta) exp(i m phi) / (n + m)!.
```

The current scalar production source convention for the compressed stored
coefficient is

```text
M_n^m = -(-1)^(n + m) conj(R_n^m(x_s - c_s)),  0 <= m <= n.
```

This production convention evaluates a unit scalar source as `-1/(4*pi*r)`.
The native theory coefficients for analytic `1/r` normalization are therefore

```text
M_n^m[1/r] = -4*pi * M_n^m[production].
```

Equivalently, if production-normalized helpers are used internally, multiply
the evaluated production scalar potential by `-4*pi` before comparing with
`1/r`.

The M2L component of the full point-mass example is:

```text
M(source point about c_s)
L(c_t) = Z_phi^-1 S_L Z_theta S_L^-1 H_L K_z(r)
         S_M Z_theta S_M^-1 Z_phi M
u(x_t) = evaluate_local(L(c_t), x_t - c_t)
u_1/r(x_t) = -4*pi * u(x_t).
```

For `Val(false)`, `H_L` is the identity. The verification script evaluates this
component at increasing `P` and demonstrates convergence of `u_1/r(x_t)` to

```text
1 / norm(x_t - x_s).
```

## Verification

`MATRIX_OPERATOR_REFACTOR/scripts/full_m2l_composition_verify.jl` compares an
explicit composed path against production `FastMultipole.multipole_to_local!`.
The explicit path uses:

- direct z phase application for `Z_phi` and `Z_phi^-1`;
- reconstructed axis-swap `T(theta)` from fixed `H(pi/2)` data and z phases;
- `zeta` signs on multipole alignment and `eta` signs on local return;
- recurrence-filled fixed-`m` M2L z blocks;
- the approved local Lamb-Helmholtz operator for `Val(true)`.

The generated summary is:

```text
MATRIX_OPERATOR_REFACTOR/data/full_m2l_composition/verification_summary.md
```

The verification covers axis-aligned `+z`, axis-aligned `-z`, positive
off-axis, negative off-axis, both `Val(false)` and `Val(true)`, expansion
orders `0`, `1`, `3`, `6`, and `9`, scaled-block parity at distances including
`1e-3` and `1e3`, and the unit point-mass M2L convergence component.
