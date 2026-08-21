# M2L Z-Translation Scaling

## Current Basis Layout

The z-aligned multipole-to-local translation operates on the current compressed
complex basis:

```julia
weights[real_or_imag, component, harmonic_index(n, m)]
harmonic_index(n, m) = n * (n + 1) / 2 + m + 1
```

Only stored orders `0 <= m <= n` participate. The scalar potential channel is
always translated. For `Val(true)` Lamb-Helmholtz storage, the second component
is translated by the same z-translation operator before the later
Lamb-Helmholtz local transform.

## Fixed-m Matrix Form

For a z-axis separation distance `t`, the current production kernel computes,
for each local coefficient `(n, m)`,

```text
L_n^m = sum(np=m:P) K_m[n, np] M_np^m
```

where `m <= n <= P`, `m <= np <= P`, and

```text
K_m[n, np] = (n + np)! / t^(n + np + 1).
```

This block is independent for each stored `m`. It is dense over the degree
indices inside a fixed `m` block and zero between different `m` blocks. The
same real scalar `K_m[n, np]` multiplies both real and imaginary lanes:

```text
real(L_n^m) = sum(np=m:P) K_m[n, np] real(M_np^m)
imag(L_n^m) = sum(np=m:P) K_m[n, np] imag(M_np^m)
```

For `Val(true)`, the same equation is applied independently to component `2`.
No conjugation or additional signs appear in the z-aligned M2L block. Azimuthal
phase signs remain the responsibility of the approved z-rotation operators that
rotate into and out of the z-aligned frame.

## Indexing Conventions

A fixed-`m` matrix can be viewed as a rectangular block with row degrees
`n = m:P` and source degrees `np = m:P`. The compressed coefficient index for
either side remains

```text
i(n, m) = harmonic_index(n, m).
```

The production implementation iterates all output `(n, m)` in compressed order
and reads source coefficients at `i(np, m)`. A matrix-backed operator should
preserve that layout rather than reordering coefficients by `m`; if a
fixed-`m` cache is used internally, its row/column degree offsets are

```text
row = n - m + 1
col = np - m + 1
```

with the global coefficient index recovered by `harmonic_index(degree, m)`.

## Distance Scaling

The direct coefficient expression is

```text
K_s = s! / t^(s + 1),  s = n + np.
```

Production evaluates this by recurrence, not by separately forming `s!` and
`t^(s+1)`. Let `rho = inv(t)`. For fixed output degree `n` and order `m`, the
first source degree is `np = m`:

```text
K_m[n, m] = (n + m)! * rho^(n + m + 1).
```

Then each next source degree uses

```text
K_m[n, np + 1] = K_m[n, np] * (n + np + 1) * rho.
```

The first coefficient of the next output order can also be advanced by

```text
K_m[n + 1, m] = K_m[n, m] * (n + m + 1) * rho.
```

These recurrences match production and avoid recomputing factorials or powers.
They are the preferred stable evaluation form for the operator cache. Dense
matrix materialization, if used for small expansion orders or verification,
should fill entries with the same recurrence.

For large expansion orders or small `abs(t)`, the coefficients can grow very
quickly. A future cache may store a scaled block

```text
K_m[n, np] = sigma_m[n] * Khat_m[n, np] * tau_m[np]
```

but the unscaled application must be algebraically identical to the recurrence
above. Any scaling factors must be owned by the operator metadata, not baked
into coefficient buffers, so z-rotation and y-rotation operators continue to
see the current compressed basis.

## Operator Semantics

The z M2L translation overwrites the destination block. It does not accumulate
into existing local coefficients. Accumulation into the target expansion occurs
later in the full M2L pipeline during the final inverse z-rotation.

A future implementation-facing primitive should therefore be shaped as:

```text
apply_m2l_z!(out, source, blocks, basis, layout; mode = overwrite)
```

where `blocks[m]` provides the fixed-`m` recurrence or cached matrix, `basis`
defines `P` and `harmonic_index`, and `layout` selects one or two active
component channels.
