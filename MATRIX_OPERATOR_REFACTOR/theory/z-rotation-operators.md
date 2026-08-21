# Z-Rotation Operators

## Current Basis Layout

The current coefficient arrays store real and imaginary parts separately:

```julia
weights[real_or_imag, component, harmonic_index(n, m)]
```

Only nonnegative azimuthal orders are stored. The compressed index is

```julia
harmonic_index(n, m) = n * (n + 1) / 2 + m + 1
```

with `0 <= m <= n`. Component `1` is the scalar potential channel. When
`Val(true)` Lamb-Helmholtz storage is active, component `2` is rotated by the
same z-rotation operator. When `Val(false)` is active, only component `1` is
part of the z-rotation operation.

## Per-Coefficient Block

For a fixed stored coefficient `(n, m)`, the current forward z rotation
multiplies the complex coefficient by

```text
exp(i m phi) = cos(m phi) + i sin(m phi).
```

Writing the coefficient as `(a + i b)`, this is the real `2x2` block

```text
[ a' ] = [ cos(m phi)  -sin(m phi) ] [ a ]
[ b' ]   [ sin(m phi)   cos(m phi) ] [ b ].
```

The same block is used independently for every stored `(n, m)` and every active
component channel.

For `m = 0`, `cos(0) = 1` and `sin(0) = 0`, so the block is exactly the
identity. This covers the monopole and every zonal coefficient `(n, 0)`.

## Storage-Light Representation

A literal dense operator is not the preferred representation for z rotation.
It would contain only independent `2x2` blocks, with all cross-coefficient
entries equal to zero. Generic sparse or block-diagonal machinery would also
carry avoidable indexing and dispatch overhead for a pattern that is already
known.

Instead, store two diagonal coefficient vectors over the compressed harmonic
index:

```text
C[i] = cos(m(i) phi)
S[i] = sin(m(i) phi)
```

where `i = harmonic_index(n, m)`. The operator application is a fused loop over
stored coefficient indices and active component channels.

## Forward Application

The forward z rotation has overwrite semantics. For each active channel and
stored coefficient index `i`,

```text
real_out[i] = C[i] * real_in[i] - S[i] * imag_in[i]
imag_out[i] = S[i] * real_in[i] + C[i] * imag_in[i]
```

This is exactly the current `rotate_z!` behavior, including `m = 0` identity
behavior. The destination coefficient is assigned, not incremented.

## Inverse / Back Application

The current `back_rotate_z!` assumes the phase table from the forward rotation
is already available, applies the conjugate phase `exp(-i m phi)`, and
accumulates into the destination. With the same `C` and `S` vectors,

```text
real_out[i] += C[i] * real_in[i] + S[i] * imag_in[i]
imag_out[i] += -S[i] * real_in[i] + C[i] * imag_in[i]
```

For `m = 0`, this reduces to adding the input coefficient to the output
coefficient unchanged.

## Future Operator Shape

The implementation-facing primitive should be an apply operation over `C` and
`S` vectors, not a dense matrix constructor. A suitable future API is:

```text
apply_z_rotation!(out, in, C, S, basis, layout, mode)
```

where `mode` selects forward overwrite versus inverse accumulation and `layout`
selects the active channel count. The mathematical operator is still the same
block-diagonal real operator, but its concrete representation is two diagonal
phase vectors plus a fused real/imag loop.

## GPU and Flat-Buffer Compatibility

The fused form maps directly to one GPU thread per stored coefficient and
active channel. Each thread reads `real_in`, `imag_in`, `C`, and `S`, then writes
or accumulates the corresponding output pair. No cross-thread communication is
required.

This approach should become simpler and more efficient after flat coefficient
buffers are introduced. Contiguous real/imag lanes reduce multidimensional
array indexing overhead and should improve memory coalescing. For small
expansion orders, GPU launch overhead can dominate the arithmetic; batching many
expansions is the intended GPU use case.

