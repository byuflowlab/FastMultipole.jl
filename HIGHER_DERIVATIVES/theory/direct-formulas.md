# Point-source direct formulas

For displacement `x`, `r²=x⋅x`, strength `q`, and `c=1/(4π)`:

- `phi = cq/r`
- `grad_i = -cq x_i/r³`
- `H_ij = cq(3x_i x_j - delta_ij r²)/r⁵`
- `T_ijk = cq[3r²(delta_ij x_k + delta_ik x_j + delta_jk x_i)
  - 15x_i x_j x_k]/r⁷`

The implementation skips `r²=0`, accumulates the six canonical `(j,k)` pairs for each
`i`, and emits packed-18 directly. The verification script compares every slot against
nested ForwardDiff derivatives.
