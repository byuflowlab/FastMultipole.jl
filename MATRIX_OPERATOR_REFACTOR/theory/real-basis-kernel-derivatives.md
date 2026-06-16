# Real-Basis Kernel Derivatives

## Scope

This artifact extends the approved real solid harmonic scalar evaluation (task
`008`, `theory/real-solid-harmonic-transforms.md`) to the full `1/r` kernel
derivative set used in production: the scalar potential `phi`, the gradient
`grad phi`, and the gradient Jacobian. In the scalar `Val(false)` layout this
Jacobian is the symmetric Hessian of `phi`; in the Lamb-Helmholtz `Val(true)`
layout it is a generally nonsymmetric vector-field Jacobian. It is theory-only.
It does not change production FastMultipole code under `src/`.

The derivation covers both channel layouts:

- `Val(false)`: one active channel, the scalar `phi` expansion.
- `Val(true)`: two active channels, `phi` and the Lamb-Helmholtz `chi`.

No new normalization, harmonic, or rotation convention is introduced. Every
derivative path is the real-basis image of the approved compressed complex
evaluation under the task `008` transforms `T_c2r` / `T_r2c`.

## Conventions Inherited From Task 008

The compressed complex basis stores nonnegative orders `0 <= m <= n <= P` as

```text
C_n^m = a_n^m + i b_n^m,
```

with the represented real-basis subspace satisfying `b_n^0 = 0`. The real-basis
storage lanes are

```text
R[n, 0]      = a_n^0
R[n, m, cos] = a_n^m,  m > 0
R[n, m, sin] = b_n^m,  m > 0,
```

in the degree-major mode ordering

```text
mode_index(n, 0)      = n^2 + 1
mode_index(n, m, cos) = n^2 + 2m
mode_index(n, m, sin) = n^2 + 2m + 1.
```

Evaluation uses the regular solid harmonic `H_n^m(x) = p_n^m(x) + i q_n^m(x)`.

## Production Evaluation Structure

Production (`src/evaluate_expansions.jl`) does not differentiate the regular
harmonics directly. It instead:

1. forms a Cartesian field-coefficient set from the local expansion using fixed
   linear index-shift maps;
2. evaluates each Cartesian field component against the same regular harmonics
   `H_n^m` with the same scalar contraction used for the potential; and
3. for the Hessian/Jacobian, reapplies the same spatial-gradient map to the
   three stored gradient field-coefficient channels, then evaluates again.

Because each stage is a fixed linear map on compressed complex coefficients
followed by the approved scalar contraction, its real-basis image is obtained by
the same `T_c2r` / `T_r2c` similarity used for every other operator in this
refactor. The derivative paths therefore inherit task `008` exactly and add no
new convention.

## Scalar Potential (`PS`) Baseline

The native real-basis scalar evaluation from task `008` is restated here as the
baseline. Let

```text
u_raw(x) =
    sum_n p_n^0(x) R[n, 0]
  + sum_n sum_{m=1}^n 2 * (
        p_n^m(x) R[n, m, cos] - q_n^m(x) R[n, m, sin]
    ).
```

The production-normalized scalar potential is `u_prod = u_raw / (4*pi)` and the
analytic comparison is `u_1/r = -4*pi * u_prod`.

Define the scalar contraction operator `Eval` so that `u_raw = Eval(R)`. The
gradient and Hessian below reuse `Eval` unchanged; only their coefficient inputs
differ.

## Spatial-Gradient Coefficient Operator `G`

`G` maps a single-channel compressed complex local expansion to three Cartesian
field-coefficient sets `g^x, g^y, g^z`. The field at degree `n` is sourced from
degree `n + 1` of the input (every term vanishes when `n + 1 > P`). With
`C_n^m = a_n^m + i b_n^m`:

For `m = 0`:

```text
g^x_{n,0} = -b_{n+1}^1
g^y_{n,0} = -a_{n+1}^1
g^z_{n,0} = -phi_{n+1}^0      (a_{n+1}^0, with zero imaginary lane)
```

For `m > 0`:

```text
g^x_{n,m} = (i/2) ( phi_{n+1}^{m-1} + phi_{n+1}^{m+1} )
g^y_{n,m} = (1/2) ( phi_{n+1}^{m-1} - phi_{n+1}^{m+1} )
g^z_{n,m} = -phi_{n+1}^m.
```

All orders referenced are nonnegative (`m - 1 >= 0`, and `m + 1 <= n + 1`), so no
negative-order conjugate folding is required; the `m = 0` rows are the explicit
production special cases.

### Real-lane image of `G`

Using `T`'s lane assignment `cos <- Re`, `sin <- Im`, and writing
`a_N^k = (k == 0) ? R[N, 0] : R[N, k, cos]`, `b_N^k = (k == 0) ? 0 : R[N, k, sin]`
for `N = n + 1`:

For `m = 0` (cos lane only; sin lane is zero):

```text
g^x[n, 0] = -b_{n+1}^1 = -R[n+1, 1, sin]
g^y[n, 0] = -a_{n+1}^1 = -R[n+1, 1, cos]
g^z[n, 0] = -a_{n+1}^0 = -R[n+1, 0]
```

For `m > 0`:

```text
g^x[n, m, cos] = -( b_{n+1}^{m-1} + b_{n+1}^{m+1} ) / 2
g^x[n, m, sin] =  ( a_{n+1}^{m-1} + a_{n+1}^{m+1} ) / 2
g^y[n, m, cos] =  ( a_{n+1}^{m-1} - a_{n+1}^{m+1} ) / 2
g^y[n, m, sin] =  ( b_{n+1}^{m-1} - b_{n+1}^{m+1} ) / 2
g^z[n, m, cos] = -a_{n+1}^m = -R[n+1, m, cos]
g^z[n, m, sin] = -b_{n+1}^m = -R[n+1, m, sin].
```

## Gradient (`GS`)

Each Cartesian gradient component is the scalar contraction of its field
coefficients:

```text
v_x = Eval(g^x),   v_y = Eval(g^y),   v_z = Eval(g^z),
```

with production normalization `grad_prod = (v_x, v_y, v_z) / (4*pi)` and analytic
comparison `grad(1/r) = -4*pi * grad_prod`. The analytic target is

```text
grad(1/r) = -r / s^3,    r = x - x0,    s = |r|.
```

As a similarity statement,

```text
G_real = T_c2r * G_complex * T_r2c,
```

so the real-basis gradient introduces no convention beyond task `008`.

## Lamb-Helmholtz Gradient `G_LH`

For `Val(true)`, the gradient operator reads both channels `phi` (component `1`)
and `chi` (component `2`) and produces a single three-vector field (there is no
`chi` output channel for the field). The production Lamb-Helmholtz form is:

For `m = 0`:

```text
g^x_{n,0} = -b_{n+1,phi}^1 + n * a_{n,chi}^1
g^y_{n,0} = -a_{n+1,phi}^1 - n * b_{n,chi}^1
g^z_{n,0} = -phi_{n+1}^0.
```

For `m > 0`:

```text
g^x_{n,m} = (i/2)( phi_{n+1}^{m-1} + phi_{n+1}^{m+1} )
          + (1/2)( (n-m) chi_n^{m+1} - (n+m) chi_n^{m-1} )
g^y_{n,m} = (1/2)( phi_{n+1}^{m-1} - phi_{n+1}^{m+1} )
          + (i/2)( (n-m) chi_n^{m+1} + (n+m) chi_n^{m-1} )
g^z_{n,m} = -phi_{n+1}^m - i * m * chi_n^m,
```

with the `chi_n^{m+1}` term present only when `m < n`. The `chi` contributions use
degree `n` (no shift), while the `phi` contributions use degree `n + 1`. Both
channels are stored in the real lanes and transformed by `T` exactly as `phi`.
Setting `chi = 0` recovers the scalar `G` above.

## Hessian / Field Jacobian (`HS`)

`HS` evaluates the Jacobian of the assembled vector field. In the scalar
`Val(false)` layout, this vector field is `grad phi`, so the Jacobian is the
symmetric Hessian of `phi`. In the Lamb-Helmholtz `Val(true)` layout, the `chi`
channel adds a curl component, so the result is a generally nonsymmetric
vector-field Jacobian.

Each gradient channel `g^beta` is itself a harmonic-field expansion, so the
Jacobian is obtained by applying the spatial-gradient operator `G` (the
`chi`-free map) a second time to each of the three gradient channels:

```text
( J_{beta,x}, J_{beta,y}, J_{beta,z} ) = G( g^beta ),   beta in {x, y, z},
J_{beta,alpha} = Eval( G(g^beta)_alpha ).
```

This holds for both layouts: under `Val(true)` the `chi` contribution is already
folded into `g^beta`, and the production second pass applies no further `chi`
coupling, matching the `chi`-free reapplication of `G`.

The nine production Hessian/Jacobian buffer entries (body-buffer rows `8:16`)
map to `J_{beta,alpha}` with rows as field components and columns as derivative
directions. This is the same orientation as production's
`SMatrix(vxx, vxy, vxz, vyx, vyy, vyz, vzx, vzy, vzz)` return value:

```text
[ J_xx  J_xy  J_xz ]
[ J_yx  J_yy  J_yz ]
[ J_zx  J_zy  J_zz ],
```

with production normalization `J_prod = J / (4*pi)`. For `Val(false)`, analytic
comparison uses `Hess(1/r) = -4*pi * J_prod`.

### Symmetry and trace structure

- Symmetry: for `Val(false)`, `J_{beta,alpha} = J_{alpha,beta}` because the field
  is the gradient of a scalar. For `Val(true)`, the `chi` contribution can add an
  antisymmetric curl part, so symmetry is not expected.
- Trace-free: `J_xx + J_yy + J_zz = 0`. Every regular solid harmonic `R_n^m` is
  harmonic, so any local expansion is harmonic and the verified production
  Jacobian has zero trace.

The analytic target confirms both:

```text
Hess(1/r) = (3 r r^T - s^2 I) / s^5,    r = x - x0,    s = |r|,
```

which is symmetric and trace-free for `s > 0` in the scalar `Val(false)` point
mass comparison.

## DerivativesSwitch Mapping

The compile-time switches `DerivativesSwitch{PS,GS,HS}` map onto the real-basis
paths as:

- `PS` -> evaluate the scalar contraction `Eval(R)` (`u_raw`).
- `GS` -> assemble `G(phi)` (or `G_LH(phi, chi)`) and evaluate each Cartesian
  channel with `Eval`.
- `HS` -> assemble the gradient channels, reapply the `chi`-free `G`, and evaluate
  the nine resulting Hessian/Jacobian channels with `Eval`.

Each switch is independent and compile-time eliminable exactly as in production;
the real basis changes only the coefficient storage, not the path selection.

## Deferred Consideration: Derivative-Aware Expansion Orders

The following design ideas are recorded for revisiting at the end of the
Implementation Phase. They are not part of the approved `008e` derivation and are
not required for the verification below; they motivate a possible later accuracy
refinement.

1. **Higher `chi` order.** The `chi` channel effectively expands a derivative
   (curl) quantity of the field. In the solid harmonic basis a derivative is
   composed of terms one expansion order higher, so consider carrying `chi` to
   order `P_chi = P_phi + 1` rather than truncating it at the same `P` as `phi`,
   to keep the vector field consistent to the intended order.

2. **Exact gradient/Hessian via retained higher orders.** The gradient operator
   `G` sources the field at degree `n` from input degree `n + 1`. Truncating the
   input at `P` therefore drops the top-degree contribution to the gradient, and
   two such applications drop the top two degrees for the Hessian. Consider
   allowing `phi` (and `chi`) coefficients up to `P + 1` for an exact gradient and
   `P + 2` for an exact Hessian, rather than truncating all derivative quantities
   at the desired `P`.

Both ideas trade storage and operator cost for derivative accuracy at the top
degrees and should be evaluated against profiling once the operator and
flat-buffer layers exist.

## Verification

`MATRIX_OPERATOR_REFACTOR/scripts/real_basis_kernel_derivatives_verify.jl` checks
derivative parity for `P = 1, 3, 6, 9` at multiple evaluation points. `P = 0` is
not included in the derivative parity cases because production gradient/Hessian
evaluation reads degree `n + 1` coefficients.

- native real-basis scalar potential matches the approved compressed complex
  evaluation (restating task `008`), for `Val(false)`;
- native real-basis gradient matches production complex-basis gradient
  (`GS = true`), for `Val(false)` and `Val(true)`;
- native real-basis Hessian matches production complex-basis Hessian
  (`HS = true`), for `Val(false)` and `Val(true)`; the `Val(false)` Hessian is
  symmetric and the `Val(true)` field Jacobian carries the expected nonzero curl
  (antisymmetric) part, while both layouts are trace-free;
- for the approved unit point-mass example (`Val(false)`), the real-basis
  potential, gradient, and Hessian converge to `1/r`, `grad(1/r)`, and
  `grad grad(1/r)` as `P` increases, after the approved `-4*pi` normalization.

The generated summary is:

```text
MATRIX_OPERATOR_REFACTOR/data/real_basis_kernel_derivatives/verification_summary.md
```

No production `src/` code is modified by this task.
