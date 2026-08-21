# 024b Legacy-MAC / Radix-Stencil Compatibility

## Purpose and scope

Task 024b compares the legacy adaptive octree with the fixed-domain radix
lifecycle. The legacy run fixes `multipole_acceptance = theta = 0.5`; the
radix path instead classifies integer cell offsets by an analytic truncation
bound. This note chooses a radix `stencil_epsilon` that reproduces the legacy
equal-radius, same-level geometric cutoff.

This is a compatibility construction for the interaction boundary, not a
claim that the two algorithms have identical global error. In particular, the
legacy tree is adaptive and can compare unequal source and target radii.
Task 024b therefore measures error independently for every benchmark case.

## Conventions and assumptions

- Literature order `P_literature = 4` means the four retained harmonic
  degrees `0:3`. FastMultipole's `expansion_order` is the highest retained
  degree, so `P_code = expansion_order = 3`.
- The legacy Barba MAC is evaluated for equal source and target bounding
  radii `rho` at the same level.
- The radix cache uses a fixed cube
  `bounds = (SVector(-0.01, -0.01, -0.01), 1.02)`. `RadixFMMCache` stores the
  half-width, hence `h0 = 1.02 / 2 = 0.51`.
- At radix depth `ell`, cell half-width is `h0 / 2^ell` and the radius of the
  cell's bounding sphere is
  `rho = sqrt(3) * h0 / 2^ell`.
- The scalar analytic stencil is used, with
  `ConstantPStencilConfig(P_code, epsilon)` defaults:
  `source_strength = A = 1`, `lamb_helmholtz = false`, and
  `normalization = :analytic`. No production `1/(4*pi)` factor is applied.

## Continuous boundary

For equal source and target radii, the legacy MAC boundary is

```text
R * theta = 2rho,
c_theta = R/rho = 2/theta.
```

The constant-order scalar radix bound implemented by
`constant_p_stencil_bound` is

```text
B(P_code,c,A)
  = 2A / (rho * (c - 2)) * (1 / (c - 1))^(P_code + 1).
```

Substituting `c = 2/theta` gives

```text
epsilon_compat
  = (A/rho) * theta/(1-theta)
    * (theta/(2-theta))^(P_code + 1).
```

Since `P_literature = P_code + 1` and
`rho = h0*sqrt(3)/2^ell`, this is equivalently

```text
epsilon_compat(theta,ell,P_literature,A,h0)
  = A*2^ell/(h0*sqrt(3))
    * theta/(1-theta)
    * (theta/(2-theta))^P_literature.
```

For `theta=0.5`, `P_literature=4`, `A=1`, and `h0=0.51`,

```text
epsilon_compat = 2^ell / (81 * 0.51 * sqrt(3)).
```

The proportionality to `2^ell` cancels the bound's inverse cell-radius
scaling. It therefore preserves one geometric cutoff in integer-offset space
at every depth.

## Exact discrete-offset rule

The continuous boundary passes through an integer-offset shell. Cell-center
separation for offset `o` is

```text
R = (2h0/2^ell) * norm(o),
R/rho = 2norm(o)/sqrt(3).
```

The strict legacy predicate is

```text
norm(o) > sqrt(3)/theta = sqrt(12).
```

Thus squared norm 12 is rejected, while the next representable squared norm
13 is accepted. The radix predicate is non-strict, `B <= epsilon`, so the
compatible open/closed interval is

```text
B(P_code, norm2=13, A) <= epsilon
    < B(P_code, norm2=12, A).
```

Task 024b uses the midpoint of these endpoints. Evaluating the implemented
analytic bound gives:

| `ell` | valid epsilon interval | chosen epsilon |
| ---: | ---: | ---: |
| 2 | `[0.0418078, 0.0559042)` | `0.0488560` |
| 3 | `[0.0836155, 0.1118083)` | `0.0977119` |
| 4 | `[0.1672310, 0.2236167)` | `0.1954239` |
| 5 | `[0.3344621, 0.4472333)` | `0.3908477` |
| 6 | `[0.6689242, 0.8944666)` | `0.7816954` |
| 7 | `[1.3378484, 1.7889333)` | `1.5633908` |

The full-precision campaign rule is

```text
stencil_epsilon(ell) =
    0.19542385331034917 * 2.0^(ell - 4).
```

The accompanying verifier independently enumerates every integer offset in
the radix lifecycle's finite depth-`ell` domain,
`[-(2^ell-1), +(2^ell-1)]^3`. For each offset it compares the strict legacy
predicate with the actual analytic `constant_p_stencil_bound(...) <= epsilon`
predicate, asserts zero mismatches, checks the interval strictly, and records
the result in `data/cpu_gpu_scaling/references/compatibility_verification.csv`.

