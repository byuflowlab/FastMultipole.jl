# Future Improvements

## Precompute and diagonally scale the z-axis M2L operator

The current `translate_multipole_to_local_z!` recurrence rebuilds the
distance-dependent z-axis M2L coefficients for each call. For the Laplace
kernel, this dependence can be separated exactly: precompute the z-axis
operator at unit distance once, then apply degree-wise diagonal scaling for the
actual source-target distance.

For each fixed azimuthal order `m`, the z-axis M2L block maps source multipole
coefficients `M[n', m]` to target local coefficients `L[n, m]`. Laplace
homogeneity gives

```math
K_{n n'}^{(m)}(t) = K_{n n'}^{(m)}(1)\,t^{-(n+n'+1)}.
```

Equivalently,

```math
K^{(m)}(t) = D_L^{(m)}(t)\,K^{(m)}(1)\,D_M^{(m)}(t),
```

where

```math
D_M^{(m)}(t)_{n'n'} = t^{-n'}, \qquad
D_L^{(m)}(t)_{nn} = t^{-(n+1)}.
```

In practice, store one dense matrix `K1[m] = K^(m)(1)` for each `m = 0:P`.
For each z-axis M2L call:

1. scale the source `m`-block by `t^{-n'}`;
2. multiply by the precomputed `K1[m]`;
3. scale the local output block by `t^{-(n+1)}`.

This turns the z-translation core into fixed small dense matrix-vector products
plus two diagonal scalings. It should be especially useful for batched M2L
execution, where many translations at the same expansion order can reuse the
same `K1[m]` matrices and fuse the diagonal factors into buffer loads/stores.

This factorisation is exact for homogeneous Laplace kernels. It should not be
assumed exact for smoothed Laplace, Helmholtz, Yukawa, or other kernels with an
intrinsic length scale, where the operator depends on additional nondimensional
parameters such as `sigma/t` or `k*t`.
