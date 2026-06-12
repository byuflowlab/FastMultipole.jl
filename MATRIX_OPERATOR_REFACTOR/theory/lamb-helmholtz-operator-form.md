# Lamb-Helmholtz Operator Form

## Current Basis Layout

The current Lamb-Helmholtz expansion uses the same compressed complex harmonic
layout as the scalar expansion:

```julia
weights[real_or_imag, component, harmonic_index(n, m)]
harmonic_index(n, m) = n * (n + 1) / 2 + m + 1
```

Only stored orders `0 <= m <= n` appear. Component `1` stores the phi channel,
and component `2` stores the chi channel when `Val(true)` is active. The
Lamb-Helmholtz transform is not present for `Val(false)`.

The transform acts independently for each stored azimuthal order `m`. It never
mixes different `m` values, and it never changes the compressed coefficient
indexing. Channel coupling is local in degree, using only the same `(n, m)` chi
coefficient and one neighboring chi degree.

## Multipole-Side Transform

After a z-aligned M2M translation, production applies
`transform_lamb_helmholtz_multipole!`. For each `m` and each
`n = max(m, 1):P`, let hatted coefficients denote the translated values before
the Lamb-Helmholtz transform. With `chi_{m-1}^m = 0`, the complex operator is

```text
phi_tilde_n^m = phi_hat_n^m - i * r * m / (n + 1) * chi_hat_n^m
chi_tilde_n^m = chi_hat_n^m + r / n * chi_hat_{n - 1}^m
```

The stored real lanes are therefore

```text
real(phi_tilde_n^m) = real(phi_hat_n^m) + a * imag(chi_hat_n^m)
imag(phi_tilde_n^m) = imag(phi_hat_n^m) - a * real(chi_hat_n^m)
a = r * m / (n + 1)
```

and

```text
real(chi_tilde_n^m) = real(chi_hat_n^m) + b * real(chi_hat_{n - 1}^m)
imag(chi_tilde_n^m) = imag(chi_hat_n^m) + b * imag(chi_hat_{n - 1}^m)
b = r / n
```

The `(n, m) = (0, 0)` chi coefficient is left unchanged by the current
production loop. The phi channel is unchanged whenever `m = 0` because the
same-degree coupling factor is zero.

As a real block over `[phi_re, phi_im, chi_re, chi_im]`, the same-degree
coupling part for `n >= max(m, 1)` is

```text
[1  0  0   a]
[0  1 -a   0]
[0  0  1   0]
[0  0  0   1]
```

with an additional lower-degree chi-to-chi contribution `b * I2` from
`chi_hat_{n - 1}^m` into `chi_tilde_n^m`.

## Local-Side Transform

After z-aligned M2L or L2L translation, production applies
`transform_lamb_helmholtz_local!`. For each `m` and `n = m:P`, with
`chi_{P + 1}^m = 0`, the complex operator is

```text
phi_tilde_n^m = phi_hat_n^m + i * r * m / n * chi_hat_n^m,  n > 0
phi_tilde_0^0 = phi_hat_0^0
chi_tilde_n^m = chi_hat_n^m - r / (n + 1) * chi_hat_{n + 1}^m
```

The stored real lanes for `n > 0` are

```text
real(phi_tilde_n^m) = real(phi_hat_n^m) - a * imag(chi_hat_n^m)
imag(phi_tilde_n^m) = imag(phi_hat_n^m) + a * real(chi_hat_n^m)
a = r * m / n
```

and for all `n`, using the zero extension above,

```text
real(chi_tilde_n^m) = real(chi_hat_n^m) - b * real(chi_hat_{n + 1}^m)
imag(chi_tilde_n^m) = imag(chi_hat_n^m) - b * imag(chi_hat_{n + 1}^m)
b = r / (n + 1)
```

As a real same-degree block for `n > 0`, the local transform uses

```text
[1  0  0  -a]
[0  1  a   0]
[0  0  1   0]
[0  0  0   1]
```

with an additional upper-degree chi-to-chi contribution `-b * I2` from
`chi_hat_{n + 1}^m` into `chi_tilde_n^m`. For `n = 0`, the phi block is the
identity and only the chi upper-degree contribution is applied.

## Operator Semantics

Both transforms overwrite their input expansion in production. A matrix-backed
operator should expose overwrite semantics for the transform itself and should
leave accumulation to the final inverse z-rotation in the full pipeline.

The transforms are sparse by construction:

- no cross-`m` coupling;
- same-degree phi-from-chi coupling only;
- nearest-neighbor chi-from-chi coupling in degree;
- identical real scalar neighbor coupling for real and imaginary lanes.

Because z-rotation applies the same real `2x2` block independently to both
components, the Lamb-Helmholtz transform can remain a separate component-space
operator in the current compressed complex basis. It should not require a
coefficient reorder, and it should be compatible with future flat buffers that
preserve `harmonic_index(n, m)` ordering.

## Pipeline Placement

The current production placement is:

- M2M: rotate source to the z axis, z-translate both channels, apply the
  multipole-side Lamb-Helmholtz transform, then inverse rotate and accumulate.
- M2L: rotate source to the z axis, z-translate both channels into local
  coefficients, apply the local-side Lamb-Helmholtz transform, then inverse
  rotate and accumulate.
- L2L: rotate local coefficients to the z axis, z-translate both channels,
  apply the local-side Lamb-Helmholtz transform, then inverse rotate and
  accumulate.

The explicit operator pipeline should preserve that placement. Applying the
Lamb-Helmholtz transform before the z translation would change the neighboring
degree coupling and is not equivalent to current production behavior.
