# Point-vortex second spatial derivative

For circulation `Gamma`, displacement `x`, `A = Gamma cross x`, and
`B_ij = partial_j A_i = epsilon_iℓj Gamma_ℓ`, velocity is `v_i=c A_i/r³`. Its second
spatial derivative is

`T_ijk = c[15 A_i x_j x_k/r⁷ - 3(A_i delta_jk + B_ij x_k + B_ik x_j)/r⁵]`.

This is symmetric in `(j,k)` but generally not in `i`. The same formula is used as the
near-field oracle for the complex and real LH L2B recurrence. Singular coincident pairs are
excluded exactly as in the existing point-vortex kernel.
