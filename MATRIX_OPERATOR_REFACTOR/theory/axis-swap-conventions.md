# Axis-Swap Conventions

## Current Rotation Pipeline

The current M2M, M2L, and L2L paths first convert the translation vector

```text
Delta x = target_center - source_center
```

to spherical coordinates `(r, theta, phi)`. The active coefficient pipeline is:

```text
rotate_z!(phi)
rotate_multipole_y!(theta)  # M2M and M2L source-side multipoles
rotate_local_y!(theta)      # L2L source-side locals
z-axis translation
back_rotate_*_y!
back_rotate_z!
```

The production comments note that the y-alignment method contributes an
additional rotation of `pi` about the new z axis. The operator refactor must
preserve that behavior as part of the fixed axis-swap convention, not remove it
as an apparent redundant phase. Equivalently, the approved parity target is the
current production sequence, including this `pi` convention.

`rotate_z!(phi)` is active on coefficients: each stored complex coefficient is
multiplied by `exp(i m phi)`. `back_rotate_z!` applies the conjugate phase and
accumulates into the target. The y routines use the same active coefficient
interpretation for the forward alignment step, but their back routines reset the
destination and apply the same stored y-alignment operator again. This is
intentional: the current y-alignment matrix is self-inverse for the generated
`T` blocks.

## Passive Alignment Interpretation

The geometric purpose is passive: choose coordinates whose local z axis is the
translation direction. The implemented coefficient operation is active: rotate
the expansion coefficients so the z-axis translation kernels can be applied.

This convention is shared by all three translation families:

| Path | Translation vector | Forward alignment | z-axis kernel | Return step |
| --- | --- | --- | --- | --- |
| M2M | parent center minus child center | `rotate_z!(phi)`, multipole y alignment | `translate_multipole_z!` | multipole y back, `back_rotate_z!` |
| M2L | target center minus source center | `rotate_z!(phi)`, multipole y alignment | M2L z translation and local y back | local y back, `back_rotate_z!` |
| L2L | child center minus parent center | `rotate_z!(phi)`, local y alignment | `translate_local_z!` | local y back, `back_rotate_z!` |

The operator refactor should therefore treat `phi` and `theta` as parameters of
the alignment to `Delta x`, while preserving the active coefficient signs of the
current kernels.

## Invariant Axis-Swap Form

For every degree `n`, the non-z part of the current y-alignment is generated
from the fixed `H(pi/2)` blocks. The angle-dependent `T_n(theta)` block is a
Fourier composition of two invariant axis swaps and a diagonal z-axis phase:

```text
T_n(theta) = S_n * Z_n(theta) * S_n^{-1}
```

where:

- `S_n` is determined only by the precomputed `H(pi/2)` entries;
- `Z_n(theta)` contains only `exp(i nu theta)` phases for `-n <= nu <= n`;
- `S_n` and `S_n^{-1}` are independent of `theta`, `phi`, distance, and tree
  geometry.

The current real compressed implementation expands this composition into
`cos(nu theta)` and `sin(nu theta)` recurrences. Later implementation work
should expose this as fixed axis-swap matrices plus z-rotation phase vectors
rather than as dense arbitrary-angle y-rotation matrices.

The complete forward alignment can be represented as:

```text
Z_phi * S * Z_theta * S^{-1}
```

with the production `pi` z-axis convention folded into the selected fixed
axis-swap signs. The return alignment uses the same fixed convention and the
inverse/conjugate z phases already documented in the z-rotation theory.

## Compressed Complex Sign Tables

Only nonnegative `m` coefficients are stored. Contributions from negative
orders use the solid-harmonic conjugacy relation:

```text
C(n, -m) = (-1)^m conj(C(n, m))
```

for `m > 0`. The y-alignment kernels reconstruct these negative-order
contributions on the fly.

The multipole `zeta` phase for a contribution from signed order `mp` to stored
order `m` is:

```text
mod = (abs(mp) - abs(m)) mod 4
```

| `mod` | `zeta` phase multiplier |
| ---: | --- |
| 0 | `+1` |
| 1 | `+i` |
| 2 | `-1` |
| 3 | `-i` |

The local `eta` phase reverses the difference:

```text
mod = (abs(m) - abs(mp)) mod 4
```

| `mod` | `eta` phase multiplier |
| ---: | --- |
| 0 | `+1` |
| 1 | `+i` |
| 2 | `-1` |
| 3 | `-i` |

The sign difference is not optional. Multipole source-side alignment for M2M
and M2L must use the `zeta` table; local source-side alignment and M2L local
return alignment must use the `eta` table.

## Overwrite, Reset, And Accumulate Semantics

Forward z rotation overwrites active coefficients and leaves inactive layout
channels outside the operation's active component count unchanged.

Forward multipole and local y alignment reset the full destination expansion
buffer before accumulating each degree/order result. For `Val(false)`, this
means the inactive second component is reset to zero. This reset behavior is
part of the parity target for future explicit axis-swap kernels.

Back y alignment also resets its destination. It is not an accumulating inverse
operation. Accumulation into tree targets happens at the final `back_rotate_z!`
step.

## Parity Targets

Future y-rotation and full-pipeline tests should check:

- `S * Z_theta * S^{-1}` reproduces production `rotate_multipole_y!` and
  `rotate_local_y!` for axis-aligned and off-axis vectors.
- The multipole path uses `zeta` signs and the local path uses `eta` signs.
- The production extra `pi` z-axis convention is preserved in complete M2M,
  M2L, and L2L parity tests.
- Forward y alignment resets the destination before writing; back y alignment
  resets rather than accumulates; final `back_rotate_z!` accumulates.
- Both `Val(false)` and `Val(true)` layouts are covered.

## Verification

`MATRIX_OPERATOR_REFACTOR/scripts/axis_swap_verify.jl` reconstructs the current
`T` blocks from fixed `H(pi/2)` data and z-axis trigonometric phases, then
compares the resulting axis-swap composition with production multipole and
local y alignment. The deterministic cases include `theta = 0`, `theta = pi`,
positive off-axis, and negative off-axis rotations for both `Val(false)` and
`Val(true)` layouts.

The generated summary is:

```text
MATRIX_OPERATOR_REFACTOR/data/axis_swap/verification_summary.md
```
