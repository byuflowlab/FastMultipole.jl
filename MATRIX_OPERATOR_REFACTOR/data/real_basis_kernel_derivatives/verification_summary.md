# Real-Basis Kernel Derivatives Verification Summary

- Command: `julia --project=. MATRIX_OPERATOR_REFACTOR/scripts/real_basis_kernel_derivatives_verify.jl`
- Parity tolerance: `atol <= 1.0e-12`
- Point-mass convergence target: final `rtol <= 5.0e-7` after `-4*pi` normalization
- Status: `PASS`
- Max native-vs-production scalar error (`Val(false)`): `0.0`
- Max native-vs-production gradient error: `0.0`
- Max native-vs-production Hessian error: `0.0`
- Max Hessian asymmetry `|H - H^T|` (`Val(false)`, expected ~0): `0.0`
- Max field-Jacobian curl asymmetry (`Val(true)`, expected nonzero): `0.12078046587143354`
- Max Hessian trace `|tr H|` (both layouts, expected ~0): `1.3877787807814457e-17`
- Max native-vs-production error on point-mass chain: `0.0`
- Final point-mass potential/gradient/Hessian rel error: `8.556935334046337e-16` / `1.0304959663365309e-16` / `6.193260603446747e-16`

## Derivative Parity

Each case evaluates one representable local expansion with production `evaluate_local` (`DerivativesSwitch(true,true,true)`) and the same coefficients with the native real-basis evaluators at the identical local point. Scalar parity is reported only for `Val(false)`; the production scalar `phi` potential is intentionally degenerate under Lamb-Helmholtz. The asymmetry column is `|H - H^T|`: it is ~0 for the curl-free `Val(false)` Hessian and is the expected nonzero curl asymmetry for the `Val(true)` field Jacobian.

| Layout | P | Point | Scalar err | Gradient err | Hessian err | Asymmetry | `|tr H|` |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `Val(false)` | 1 | `(0.04, -0.02, 0.03)` | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| `Val(true)` | 1 | `(0.04, -0.02, 0.03)` | n/a | 0.000e+00 | 0.000e+00 | 1.111e-01 | 0.000e+00 |
| `Val(false)` | 1 | `(-0.03, 0.05, 0.02)` | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| `Val(true)` | 1 | `(-0.03, 0.05, 0.02)` | n/a | 0.000e+00 | 0.000e+00 | 1.111e-01 | 0.000e+00 |
| `Val(false)` | 1 | `(0.06, 0.01, -0.04)` | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| `Val(true)` | 1 | `(0.06, 0.01, -0.04)` | n/a | 0.000e+00 | 0.000e+00 | 1.111e-01 | 0.000e+00 |
| `Val(false)` | 1 | `(-0.02, -0.05, 0.07)` | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| `Val(true)` | 1 | `(-0.02, -0.05, 0.07)` | n/a | 0.000e+00 | 0.000e+00 | 1.111e-01 | 0.000e+00 |
| `Val(false)` | 3 | `(0.04, -0.02, 0.03)` | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 3.469e-18 |
| `Val(true)` | 3 | `(0.04, -0.02, 0.03)` | n/a | 0.000e+00 | 0.000e+00 | 1.093e-01 | 4.337e-19 |
| `Val(false)` | 3 | `(-0.03, 0.05, 0.02)` | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| `Val(true)` | 3 | `(-0.03, 0.05, 0.02)` | n/a | 0.000e+00 | 0.000e+00 | 1.012e-01 | 3.469e-18 |
| `Val(false)` | 3 | `(0.06, 0.01, -0.04)` | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 3.469e-18 |
| `Val(true)` | 3 | `(0.06, 0.01, -0.04)` | n/a | 0.000e+00 | 0.000e+00 | 1.208e-01 | 8.674e-19 |
| `Val(false)` | 3 | `(-0.02, -0.05, 0.07)` | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 3.469e-18 |
| `Val(true)` | 3 | `(-0.02, -0.05, 0.07)` | n/a | 0.000e+00 | 0.000e+00 | 1.021e-01 | 8.674e-19 |
| `Val(false)` | 6 | `(0.04, -0.02, 0.03)` | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 3.469e-18 |
| `Val(true)` | 6 | `(0.04, -0.02, 0.03)` | n/a | 0.000e+00 | 0.000e+00 | 1.093e-01 | 1.084e-18 |
| `Val(false)` | 6 | `(-0.03, 0.05, 0.02)` | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| `Val(true)` | 6 | `(-0.03, 0.05, 0.02)` | n/a | 0.000e+00 | 0.000e+00 | 1.012e-01 | 0.000e+00 |
| `Val(false)` | 6 | `(0.06, 0.01, -0.04)` | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 3.469e-18 |
| `Val(true)` | 6 | `(0.06, 0.01, -0.04)` | n/a | 0.000e+00 | 0.000e+00 | 1.208e-01 | 6.505e-19 |
| `Val(false)` | 6 | `(-0.02, -0.05, 0.07)` | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 3.469e-18 |
| `Val(true)` | 6 | `(-0.02, -0.05, 0.07)` | n/a | 0.000e+00 | 0.000e+00 | 1.021e-01 | 0.000e+00 |
| `Val(false)` | 9 | `(0.04, -0.02, 0.03)` | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 6.939e-18 |
| `Val(true)` | 9 | `(0.04, -0.02, 0.03)` | n/a | 0.000e+00 | 0.000e+00 | 1.093e-01 | 3.686e-18 |
| `Val(false)` | 9 | `(-0.03, 0.05, 0.02)` | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| `Val(true)` | 9 | `(-0.03, 0.05, 0.02)` | n/a | 0.000e+00 | 0.000e+00 | 1.012e-01 | 1.735e-18 |
| `Val(false)` | 9 | `(0.06, 0.01, -0.04)` | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 1.388e-17 |
| `Val(true)` | 9 | `(0.06, 0.01, -0.04)` | n/a | 0.000e+00 | 0.000e+00 | 1.208e-01 | 8.674e-19 |
| `Val(false)` | 9 | `(-0.02, -0.05, 0.07)` | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| `Val(true)` | 9 | `(-0.02, -0.05, 0.07)` | n/a | 0.000e+00 | 0.000e+00 | 1.021e-01 | 1.735e-18 |

## Unit Point-Mass Convergence

Native real-basis potential, gradient, and Hessian were evaluated on the approved unit point-mass M2M-M2L-L2L chain local expansion and multiplied by `-4*pi` before comparison to the analytic `1/r`, `grad(1/r) = -r/s^3`, and `Hess(1/r) = (3 r r^T - s^2 I)/s^5`.

| P | Potential rel err | Gradient rel err | Hessian rel err | Native-vs-production err |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1.000258e-03 | 3.463428e-02 | 1.000000e+00 | 0.000e+00 |
| 2 | 2.019974e-05 | 1.165035e-03 | 5.653853e-02 | 0.000e+00 |
| 3 | 7.709523e-07 | 4.167759e-05 | 2.929931e-03 | 0.000e+00 |
| 5 | 8.253620e-10 | 3.912999e-08 | 5.120361e-06 | 0.000e+00 |
| 7 | 1.046085e-12 | 4.406138e-11 | 6.837591e-09 | 0.000e+00 |
| 9 | 1.925310e-15 | 5.815379e-14 | 6.639408e-12 | 0.000e+00 |
| 12 | 8.556935e-16 | 1.030496e-16 | 6.193261e-16 | 0.000e+00 |
