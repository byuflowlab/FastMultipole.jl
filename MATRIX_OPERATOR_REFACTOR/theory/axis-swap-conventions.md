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

### Realization Amendment (`2026-06-23`, task `013c`, user-directed "Plain-H")

The implementation work above warned to expose this as *fixed axis-swap matrices
plus z-rotation phase vectors* rather than dense arbitrary-angle matrices. Task
`013c` did so, and a spike pinned the exact convention-faithful form. Two points
must be recorded so the derivation is not mis-applied.

**The swap is the plain operator, not the dressed one.** The production y-alignment
applies, per degree `n`, the matrix `A_n[m,mp] = zeta_n^{mp,m} * T_n^{mp,m}(theta)`
(the `eta` table for the local path), where `T_n` is the *plain* (dressing-free) real
Wigner block and `zeta_n^{mp,m} = (beta_n^{mp}/beta_n^m) * i^{|mp|-|m|}` is the
compressed-complex dressing of the table above. The genuine factorization
`T_n(theta) = S_n Z_n(theta) S_n^{-1}` holds for the **plain** `T_n` (its `S_n` are
built from `H(pi/2)`). It does **not** survive being applied through the dressed
kernel, because the dressing does not commute through the swap:

```text
(zeta . S) * Z_theta * (zeta . S)^{-1}  !=  zeta . (S * Z_theta * S^{-1}).
```

Equivalently, composing the production-kernel `±pi/2` y-swaps (`013b`'s ζ-dressed
`T_y_pos90` / `T_y_neg90`) with a `Z_theta` produces an `R_x`-type, not `R_y`,
operator. The dressing `zeta_n^{mp,m} = a(mp) * b(m)` is separable (verified to
machine precision), so the correct staged form keeps the swap plain and carries the
dressing as outer diagonals: `a(mp)` pre, `b(m)` post. Either fold those diagonals
into the fixed swap matrices (the shipped choice) or apply them separately.

**Executable fixed-mode form.** For every degree `n`, every angular Fourier component
of the production y-operator on the `2n+1` real dofs is **rank 1**, so

```text
Y_n(theta) = U_n * diag(exp(i nu theta)) * V_n,   nu = -n..n,
```

with `U_n`, `V_n` fixed (angle/geometry-independent, batch-shared) per-degree
matrices — the executable `S_n` / `S_n^{-1}` of the form above, with `Z_n(theta) =
diag(exp(i nu theta))` the cheap z-rotation in the swapped (`nu`) frame. The cost is
`O(P^3)` per column (two `O(n^2)` fixed-matrix contractions and one `O(n)` diagonal),
versus the materialized `O(P^4)` per-call `Ts(theta)` rebuild. The shipped code builds
`U_n`/`V_n` once at cache construction (sample the production-parity kernels at `2n+1`
angles, DFT to the components, rank-1-factor each) and the `pi`/extra-`pi` and sign
conventions are inherited exactly from the sampled kernels. The multipole (`zeta`) and
local (`eta`) paths carry separate `U,V` (dressing absorbed); the staged arithmetic is
identical and is selected by which modes are supplied. See task `013c` Revised
Implementation Notes for the verification (~1e-13 vs production, P up to 8, both
`Val`).

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
