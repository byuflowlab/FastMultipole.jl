# Cartesian differentiation recurrence

Let `a[n,m]` be complex coefficients multiplying regular solid harmonics, retaining only
`m ≥ 0` and reconstructing the negative modes by real-field symmetry. The Cartesian
derivative coefficients at degree `n` are obtained from degree `n+1`:

- `Dx a[n,m] = i(a[n+1,m-1] + a[n+1,m+1])/2`
- `Dy a[n,m] = (a[n+1,m-1] - a[n+1,m+1])/2`
- `Dz a[n,m] = -a[n+1,m]`

The `m=0` implementation uses the real-field specialization already used by the gradient
and Hessian paths. Applying this linear recurrence three times yields the third spatial
derivative. Commutation of these constant-coefficient derivative operators proves full
permutation symmetry for a scalar potential. For LH fields the first pass couples phi and
chi into velocity coefficients; the next two passes are ordinary spatial derivatives and
therefore prove only symmetry of their two derivative indices.

The implementation stores three velocity coefficient slabs and, only for third-order
requests, nine Hessian-coefficient slabs in worker-local scratch. It contracts the six
canonical derivative pairs directly into packed-18 output without a dense tensor.
