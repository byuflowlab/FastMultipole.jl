# Real Solid Harmonic Transforms

## Scope

This artifact defines the real solid harmonic coefficient basis for the matrix
operator refactor. It is theory-only: it derives storage transforms and real
operator definitions from the approved compressed complex operators. It does
not change production FastMultipole code.

The real basis here is the storage-reduced real coefficient basis induced by
the current compressed complex convention. It is not a newly normalized
external spherical-harmonic basis.

## Real Mode Ordering

For expansion order `P`, the real basis has

```text
Nreal(P) = (P + 1)^2
```

entries per active channel. The degree-major mode ordering is the ordering
reserved by the coefficient-buffer layout task:

```text
mode_index(n, 0)      = n^2 + 1
mode_index(n, m, cos) = n^2 + 2m
mode_index(n, m, sin) = n^2 + 2m + 1
```

with `1 <= m <= n <= P`. For each degree `n`, the contiguous block
`n^2 + 1 : (n + 1)^2` stores

```text
m = 0, m = 1 cos, m = 1 sin, ..., m = n cos, m = n sin.
```

## Transform Convention

The approved compressed complex basis stores nonnegative orders
`0 <= m <= n <= P` as

```text
C_n^m = a_n^m + i b_n^m.
```

The complex-to-real transform `T_c2r` is:

```text
R[n, 0]      = a_n^0
R[n, m, cos] = a_n^m
R[n, m, sin] = b_n^m,  m > 0.
```

The real-to-complex transform `T_r2c` is:

```text
a_n^0 = R[n, 0]
b_n^0 = 0
a_n^m = R[n, m, cos],  m > 0
b_n^m = R[n, m, sin],  m > 0.
```

These transforms are applied independently to every active channel. For
`Val(false)`, only component `1` is active. For `Val(true)`, components `1`
and `2` are active, with the same transform applied to both the `phi` and
`chi` channels.

## Representability Invariant

The real basis represents exactly the compressed complex coefficient subspace
whose `m = 0` imaginary lanes are zero:

```text
b_n^0 = 0,  0 <= n <= P.
```

Therefore:

- `T_c2r * T_r2c` is the identity on every real-basis coefficient vector.
- `T_r2c * T_c2r` is the identity on compressed complex coefficients that
  satisfy `b_n^0 = 0`.
- For general compressed complex input, `T_r2c * T_c2r` intentionally projects
  every `imag(C_n^0)` lane to zero.

This projection is not a numerical loss in the real-basis model; those lanes
are outside the represented real solid harmonic subspace.

## Scalar Evaluation in the Real Basis

The approved compressed complex scalar local evaluation stores

```text
C_n^m = a_n^m + i b_n^m
```

and evaluates against the regular harmonic

```text
H_n^m(x) = p_n^m(x) + i q_n^m(x).
```

The production scalar contribution is

```text
m = 0:  p_n^0 a_n^0 - q_n^0 b_n^0
m > 0:  2 * (p_n^m a_n^m - q_n^m b_n^m).
```

In the represented real-basis subspace, `b_n^0 = 0`. Substituting the storage
lanes

```text
R[n, 0]      = a_n^0
R[n, m, cos] = a_n^m
R[n, m, sin] = b_n^m,  m > 0
```

gives the native real-basis scalar formula

```text
u_raw(x) =
    sum_n p_n^0(x) R[n, 0]
  + sum_n sum_{m=1}^n 2 * (
        p_n^m(x) R[n, m, cos] - q_n^m(x) R[n, m, sin]
    ).
```

The production-normalized scalar potential is

```text
u_prod(x) = u_raw(x) / (4*pi).
```

Analytic `1/r` comparisons still use the approved production conversion

```text
u_1/r(x) = -4*pi * u_prod(x).
```

## Real Z Rotations

The approved compressed complex z rotation multiplies

```text
C_n^m <- exp(i m phi) C_n^m.
```

In the real basis, the `m = 0` lane is the identity:

```text
R[n, 0]' = R[n, 0].
```

For `m > 0`, the same real `2x2` block acts on the `(cos, sin)` lane pair as
acts on the compressed `(real, imag)` pair:

```text
[ R[n, m, cos]' ] = [ cos(m phi)  -sin(m phi) ] [ R[n, m, cos] ]
[ R[n, m, sin]' ]   [ sin(m phi)   cos(m phi) ] [ R[n, m, sin] ].
```

The inverse/back z rotation uses the conjugate block and preserves the
approved overwrite/accumulate semantics: forward rotations overwrite their
destination and final inverse rotations accumulate into the target expansion.

## Real-Basis Operators

Let `A_complex` be any approved compressed complex operator in the M2M, M2L, or
L2L chains, including z rotations, invariant axis-swap compositions, fixed-`m`
z translations, and optional Lamb-Helmholtz channel coupling. The corresponding
real-basis operator is defined by the similarity transform

```text
A_real = T_c2r * A_complex * T_r2c.
```

This definition preserves the approved operator order and write semantics. In
particular:

- M2M uses the approved `Z_phi`, multipole axis-swap, M2M z-translation,
  optional multipole Lamb-Helmholtz, inverse multipole axis-swap, and final
  accumulated `Z_phi^-1` order.
- M2L uses the approved `Z_phi`, multipole axis-swap, M2L z-translation,
  optional local Lamb-Helmholtz, local axis-swap, and final accumulated
  `Z_phi^-1` order.
- L2L uses the approved `Z_phi`, local axis-swap, L2L z-translation, optional
  local Lamb-Helmholtz, inverse local axis-swap, and final accumulated
  `Z_phi^-1` order.

All non-z rotation effects remain expressed through the approved invariant
axis-swap matrices and fixed sign tables. No new free rotation convention is
introduced by the real basis.

The complete chain definitions are:

```text
M2M_real = T_c2r * M2M_complex * T_r2c
M2L_real = T_c2r * M2L_complex * T_r2c
L2L_real = T_c2r * L2L_complex * T_r2c.
```

For chains with a pre-existing target buffer, the target is transformed with
`T_r2c`, the approved complex chain performs its documented accumulation, and
the result is transformed back with `T_c2r`.

## Verification

`MATRIX_OPERATOR_REFACTOR/scripts/real_solid_harmonic_transforms_verify.jl`
checks:

- real mode indices are contiguous and unique for `P = 0, 1, 3, 6, 9`;
- `T_c2r * T_r2c` is exact on deterministic real-basis inputs;
- `T_r2c * T_c2r` is exact on compressed complex inputs with zero `m = 0`
  imaginary lanes;
- nonzero `m = 0` imaginary lanes are projected to zero and reported as
  intentional;
- real z-rotation blocks match compressed complex z rotations after
  transform/reverse-transform;
- native real-basis scalar local evaluation matches the approved compressed
  complex scalar local evaluation at seven points for each
  `P = 0, 1, 3, 6, 9`;
- real-basis M2L and complete M2M-M2L-L2L point-mass chains match the approved
  complex-basis examples after transform/reverse-transform and converge to the
  analytic `1/r` potential.

The generated summary is:

```text
MATRIX_OPERATOR_REFACTOR/data/real_solid_harmonic/verification_summary.md
```

Production scalar-potential evaluations are compared to analytic `1/r` after
the approved `-4*pi` normalization factor is applied.
