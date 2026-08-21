# M2M and L2L Operator Extensions

## Scope

This artifact extends the approved M2L operator structure to the current
compressed complex solid harmonic M2M and L2L paths. It is theory-only: it
specifies operator ordering, z-translation blocks, channel behavior,
cache requirements, and verification targets for later implementation.

For both operations the production displacement is

```text
Delta x = target_center - source_center
(r, theta, phi) = cartesian_to_spherical(Delta x).
```

The only angle-dependent rotations in either chain are z-axis phase operators.
All non-z rotation effects are represented by invariant axis-swap matrices and
fixed sign tables.

## Shared Layout

The coefficient storage remains

```julia
weights[real_or_imag, component, harmonic_index(n, m)]
harmonic_index(n, m) = n * (n + 1) / 2 + m + 1
```

with stored orders `0 <= m <= n <= P`.

For `Val(false)`, only component `1` is active. For `Val(true)`, components `1`
and `2` are active, with component `1` storing the Lamb-Helmholtz `phi` channel
and component `2` storing the `chi` channel. The z-rotation, axis-swap, M2M
z-translation, and L2L z-translation operators act identically on both active
channels. The Lamb-Helmholtz stages are the only M2M/L2L stages that couple
channels.

## M2M Operator Chain

Let `M_source` be the child/source multipole expansion and `M_target` be the
parent/target multipole expansion before accumulation. The production-parity
M2M chain is:

```text
M1 = Z_phi M_source                  # forward z rotation, overwrite
M2 = S_M Z_theta S_M^-1 M1           # multipole axis swap, zeta signs, reset
M3 = U_z(r) M2                       # M2M z translation, overwrite
M4 = H_M(r) M3                       # optional Val(true) multipole LH transform
M5 = S_M Z_theta S_M^-1 M4           # multipole axis swap back, zeta signs, reset
M_target <- M_target + Z_phi^-1 M5   # inverse z rotation, accumulate
```

For `Val(false)`, `H_M(r)` is the identity and only component `1`
participates. For `Val(true)`, `H_M(r)` is the approved multipole
Lamb-Helmholtz transform.

The fixed-`m` M2M z-translation block is lower triangular in degree:

```text
M_n^m(target) = sum_{np=m}^n U_m[n, np] M_np^m(source)
U_m[n, np] = (-r)^(n - np) / (n - np)!.
```

This block overwrites a temporary multipole expansion. It does not accumulate
directly into the target tree coefficient buffer.

For `Val(true)`, the M2M Lamb-Helmholtz stage is applied after z translation:

```text
phi_tilde_n^m = phi_hat_n^m - i * r * m / (n + 1) * chi_hat_n^m
chi_tilde_n^m = chi_hat_n^m + r / n * chi_hat_{n - 1}^m,  n > 0
chi_hat_{m - 1}^m = 0.
```

The transform overwrites its destination, is sparse in degree, and does not mix
different `m`.

## L2L Operator Chain

Let `L_source` be the parent/source local expansion and `L_target` be the
child/target local expansion before accumulation. The production-parity L2L
chain is:

```text
L1 = Z_phi L_source                  # forward z rotation, overwrite
L2 = S_L Z_theta S_L^-1 L1           # local axis swap, eta signs, reset
L3 = V_z(r) L2                       # L2L z translation, overwrite
L4 = H_L(r) L3                       # optional Val(true) local LH transform
L5 = S_L Z_theta S_L^-1 L4           # local axis swap back, eta signs, reset
L_target <- L_target + Z_phi^-1 L5   # inverse z rotation, accumulate
```

For `Val(false)`, `H_L(r)` is the identity. For `Val(true)`, `H_L(r)` is the
approved local Lamb-Helmholtz transform.

The fixed-`m` L2L z-translation block is upper triangular in degree:

```text
L_n^m(target) = sum_{np=n}^P V_m[n, np] L_np^m(source)
V_m[n, np] = (-r)^(np - n) / (np - n)!.
```

For `Val(true)`, the L2L Lamb-Helmholtz stage is applied after z translation:

```text
phi_tilde_n^m = phi_hat_n^m + i * r * m / n * chi_hat_n^m,  n > 0
phi_tilde_0^0 = phi_hat_0^0
chi_tilde_n^m = chi_hat_n^m - r / (n + 1) * chi_hat_{n + 1}^m
chi_hat_{P + 1}^m = 0.
```

The transform overwrites its destination, is sparse in degree, and does not mix
different `m`.

## Axis-Swap and Rotation Convention

Both M2M and L2L use the same active coefficient and passive coordinate
alignment convention approved for M2L:

```text
Z_phi(n, m) = exp(i m phi)
Z_phi^-1(n, m) = exp(-i m phi)
T_n(theta) = S_n Z_n(theta) S_n^-1.
```

M2M uses the multipole axis-swap metadata and `zeta` signs on both forward and
return y alignments. L2L uses the local axis-swap metadata and `eta` signs on
both forward and return y alignments. The production note that y alignment
includes an additional `pi` rotation about the new z axis is preserved by these
invariant axis-swap conventions and sign tables.

Thus both M2M and L2L compositions use only invariant matrices and z-axis
rotations for all non-z-aligned offsets.

## Buffer Semantics

Both complete pipelines have three write modes:

- Forward `Z_phi`: overwrite active channels.
- Axis swaps, z translation, and optional Lamb-Helmholtz transform: overwrite or
  reset-then-write temporary buffers.
- Final `Z_phi^-1`: accumulate into the target expansion.

The final z back-rotation is the only accumulating step. This is required for
tree traversal, where multiple child multipoles accumulate into one parent
M2M target and one parent local expansion accumulates into a child L2L target
that may already contain M2L contributions.

## Cache Requirements

A production operator cache for the M2M and L2L extensions needs:

- z phase vectors for `phi` and `theta`, or reusable phase recurrences keyed by
  expansion order and angle;
- invariant multipole axis-swap matrices `S_M`, inverse metadata, and `zeta`
  sign tables for M2M;
- invariant local axis-swap matrices `S_L`, inverse metadata, and `eta` sign
  tables for L2L;
- fixed-`m` M2M z-translation blocks or recurrence metadata for every
  `0 <= m <= P`;
- fixed-`m` L2L z-translation blocks or recurrence metadata for every
  `0 <= m <= P`;
- multipole Lamb-Helmholtz metadata for `Val(true)`, including distance `r`,
  same-degree phi-from-chi coupling, and lower-neighbor chi coupling;
- local Lamb-Helmholtz metadata for `Val(true)`, including distance `r`,
  same-degree phi-from-chi coupling, and upper-neighbor chi coupling;
- layout metadata selecting one active component for `Val(false)` and two
  active components for `Val(true)`.

The cache may store z-translation matrices densely or as structured recurrence
factors, but the applied operator must preserve the current compressed
coefficient ordering.

## Complete Unit Point-Mass Chain

For a unit point mass at `x_s` represented about source leaf center `c_leaf`,
use the approved scalar production multipole convention:

```text
R_n^m(x_s - c_leaf)
= (-1)^n i^m rho^n P_n^m(cos theta) exp(i m phi) / (n + m)!,
M_n^m = -(-1)^(n + m) conj(R_n^m(x_s - c_leaf)).
```

The production scalar convention evaluates a unit source as `-1/(4*pi*r)`.
For analytic `1/r` normalization, multiply the evaluated production scalar
potential by `-4*pi`.

The complete operator-chain example is:

```text
M_leaf = source point expansion about c_leaf
M_parent = M2M(c_parent <- c_leaf) M_leaf
L_parent = M2L(c_target_parent <- c_parent) M_parent
L_child = L2L(c_target_child <- c_target_parent) L_parent
u_prod(x_t) = evaluate_local(L_child, x_t - c_target_child)
u_1/r(x_t) = -4*pi * u_prod(x_t)
```

The verification script evaluates this chain at increasing expansion order and
demonstrates convergence of `u_1/r(x_t)` to

```text
1 / norm(x_t - x_s).
```

## Verification

`MATRIX_OPERATOR_REFACTOR/scripts/m2m_l2l_verify.jl` compares explicit
operator-chain M2M and L2L paths against production
`FastMultipole.multipole_to_multipole!` and `FastMultipole.local_to_local!`.
The explicit paths use:

- direct z phase application for `Z_phi` and `Z_phi^-1`;
- reconstructed axis-swap `T(theta)` from fixed `H(pi/2)` data and z phases;
- `zeta` signs on M2M axis swaps and `eta` signs on L2L axis swaps;
- recurrence-filled fixed-`m` M2M and L2L z blocks;
- the approved multipole/local Lamb-Helmholtz operators for `Val(true)`.

The generated summary is:

```text
MATRIX_OPERATOR_REFACTOR/data/m2m_l2l/verification_summary.md
```

The verification covers axis-aligned `+z`, axis-aligned `-z`, positive
off-axis, negative off-axis, both `Val(false)` and `Val(true)`, expansion
orders `0`, `1`, `3`, `6`, and `9`, z-block formula parity, and the complete
unit point-mass M2M-M2L-L2L convergence chain.
