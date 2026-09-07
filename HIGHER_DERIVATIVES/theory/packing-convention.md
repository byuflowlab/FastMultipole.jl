# Packed-18 convention

For `T[i,j,k] = ∂H[i,j]/∂x[k]`, mixed spatial derivatives give
`T[i,j,k] = T[i,k,j]`. For each field component `i`, store derivative pairs
`(xx,xy,xz,yy,yz,zz)`, producing slots 1–6 for `i=x`, 7–12 for `i=y`, and
13–18 for `i=z`. Scalar-potential fields are fully symmetric, but the public layout does
not assume symmetry between the field-component index and derivative indices because
Lamb–Helmholtz velocity derivatives do not have it.

The pre-existing Hessian remains a dense column-major 3×3 value in nine rows. With no
metadata and all standard outputs enabled, potential is row 4, gradient 5:7, Hessian 8:16,
and the packed third derivative 17:34. Compact buffers omit disabled groups.

`ThirdDerivativeTensor` maps both `(j,k)` and `(k,j)` to the same packed slot, while
`packed_data` exposes the canonical representation without allocation. Dense conversion is
explicit because a 27-value representation introduces nine redundant values.
