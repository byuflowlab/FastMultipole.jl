# Coefficient Buffer Layout

## Scope

This artifact specifies the native coefficient-buffer storage contract for the
later matrix-operator implementation layer. It is theory-only: it changes
storage and view requirements, not the approved operator mathematics.

The target layout prioritizes efficient dense matrix application on CPU BLAS and
GPU cuBLAS-style backends, especially batched operator application. Ease of
conversion from the current production coefficient array is secondary.

This task does not alter:

- approved invariant axis-swap matrices or sign tables;
- z-axis rotation phases, inverse phases, or overwrite/accumulate semantics;
- fixed-`m` z-translation blocks or scaling;
- Lamb-Helmholtz formulas or channel coupling;
- M2M, M2L, or L2L operator ordering.

## Native Physical Layout

The native coefficient buffer is physically channel-slabbed:

```text
basis_dof x batch x channel
```

For a Julia `Array`, each fixed-channel view

```julia
@view(buffer[:, :, c])
```

must be a dense `basis_dof x batch` matrix with first-dimension stride `1` and
second-dimension stride `basis_dof`. This makes the common channel-specific
operator application a non-allocating GEMM target:

```julia
mul!(@view(Y[:, :, c]), A[c], @view(X[:, :, c]))
```

Here `A[c]` is a channel-specific operator with shape
`basis_dof x basis_dof`. `X[:, :, c]` and `Y[:, :, c]` are dense matrices whose
columns are independent expansions in the batch.

The linear offset for one-based indices is:

```text
offset = basis_index
       + basis_dof * (batch - 1)
       + basis_dof * batch_count * (channel - 1)
```

where `1 <= basis_index <= basis_dof`, `1 <= batch <= batch_count`, and
`1 <= channel <= channel_count`. This is the same physical order as a standard
Julia `Array{T,3}` with dimensions `(basis_dof, batch_count, channel_count)`.

## Channel Semantics

The active channel count follows the existing Lamb-Helmholtz switch:

```text
Val(false): one active channel
Val(true):  two active channels, phi and chi
```

For `Val(false)`, channel `1` is the scalar expansion channel. For `Val(true)`,
channel `1` is `phi` and channel `2` is `chi`.

Channel-independent stages may be applied in either of two dense forms:

- as separate GEMMs over each fixed-channel slab;
- as one GEMM over a reshaped `basis_dof x (batch * channel)` view when that
  reshape is dense for this layout and the same operator is valid for every
  active channel.

Channel-specific stages should use dense fixed-channel slabs:

```julia
mul!(@view(Y[:, :, c]), A[c], @view(X[:, :, c]))
```

Channel-coupled Lamb-Helmholtz stages should be represented in one of two
implementation-compatible forms:

- a structured two-channel block operator that applies the documented
  `phi`/`chi` coupling without materializing a full dense matrix;
- a dense operator over `(basis_dof * channel) x batch` after an explicit or
  planned layout-compatible packing step.

The native channel-slabbed layout is therefore the storage layout for ordinary
operator stages. A coupled packed view is a deliberate operator input form, not
the default coefficient storage.

## Why Channel Slabs Come Last

The alternative layout

```text
basis_dof x channel x batch
```

is attractive because a whole expansion keeps its channels adjacent. However,
for channel-specific operators, the fixed-channel view

```julia
@view(X[:, c, :])
```

is non-copying but its column stride is `basis_dof * channel_count`, not
`basis_dof`. It is therefore not a conventional dense matrix slab for BLAS or
cuBLAS-style GEMM.

With the selected layout

```text
basis_dof x batch x channel
```

the fixed-channel view

```julia
@view(X[:, :, c])
```

has column stride `basis_dof`. Since operators can differ by channel, and since
the refactor goal is efficient CPU/GPU matrix application, fixed-channel slabs
must be physically dense.

## Performance Trade-offs

This layout makes performance-relevant choices that constrain later
implementation. They are recorded here so downstream tasks inherit the rationale
rather than rediscovering it.

### Leading-dimension padding versus channel merge

The contract pins the fixed-channel slab to `stride(view, 1) == 1` and
`stride(view, 2) == basis_dof`. The tight second-dimension stride is what makes
the `basis_dof x (batch * channel)` reshape dense, enabling the
channel-independent single-GEMM form described above.

This tight packing deliberately forecloses a padded leading dimension. A common
GPU/cuBLAS performance lever is to pad the leading dimension (`lda`) of each
slab to an alignment-friendly multiple (for example `8`, `16`, or a warp
multiple), which can improve coalescing and GEMM tile efficiency for the small,
odd `basis_dof` values that occur here (`2, 6, 20, 56, 110` for the compressed
complex basis at `P = 0, 1, 3, 6, 9`).

The trade-off is mutually exclusive: an `lda`-padded slab makes the channel
dimension stride `lda * batch_count` rather than `basis_dof * batch_count`, so
the merged `basis_dof x (batch * channel)` reshape is no longer dense and the
single-GEMM-over-all-channels form no longer applies. The default contract
chooses tight packing plus the channel-merge option. An `lda`-padded variant may
be revisited at implementation time if profiling shows it is the faster path on
a target GPU; if adopted, it must drop the merged-channel reshape and apply
operators per fixed-channel slab.

### Interleaved versus planar real/imaginary lanes

The compressed complex basis interleaves the real and imaginary lanes inside the
basis dimension (`basis_index(n, m, reim) = 2 * (harmonic_index(n, m) - 1) +
reim`). This preserves the existing production convention, in which `reim` is the
fastest index of `weights[real_or_imag, component, harmonic_index]`, and keeps
the legacy/native mapping an exact, normalization-preserving permutation.

A planar split, which stores all real lanes contiguously followed by all
imaginary lanes, is often preferable on GPU: it exposes each lane as a
contiguous `Ncomplex(P) x batch` real slab, so a complex matrix application can
run as three or four real GEMMs on contiguous planes (split-complex /
Karatsuba), matching how cuBLAS-style backends handle complex data. The
interleaved layout instead forces either a complex-typed GEMM or a doubled dense
`(2 * Ncomplex) x (2 * Ncomplex)` real operator.

The interleaving choice is acceptable because the compressed complex basis is
the transitional path: the real solid harmonic basis is the intended
performance path. It is all-real (no re/im packing), and smaller by exactly
`P + 1` entries per channel (`Nreal(P) = (P + 1)^2` versus
`basis_dof = (P + 1)(P + 2)` for the complex basis), reflecting the zero
imaginary lanes of the `m = 0` modes. Performance-critical execution should
therefore prefer the real basis, where the planar-versus-interleaved question
does not arise.

### Batch membership and batched GEMM

A `batch` groups expansions that share a single operator `A[c]`; these map to
one large GEMM `mul!(Y[:, :, c], A[c], X[:, :, c])` with `batch` as the GEMM
column count. When the operator differs per expansion (for example M2L blocks
that vary with interaction direction or distance), the same physical layout
maps to a strided-batched GEMM, with the `batch` dimension supplying the per-
matrix batch stride. The storage contract is identical in both cases; only the
GEMM call form differs.

## Compressed Complex Basis

The compressed complex basis stores only orders `0 <= m <= n <= P`:

```text
Ncomplex(P) = (P + 1)(P + 2) / 2
basis_dof   = 2 * Ncomplex(P)
```

The harmonic ordering is degree-major:

```text
harmonic_index(n, m) = n(n + 1) / 2 + m + 1
```

with `0 <= m <= n <= P`. Real and imaginary lanes are packed inside the basis
dimension:

```text
basis_index(n, m, reim) = 2 * (harmonic_index(n, m) - 1) + reim
reim in 1:2
```

`reim = 1` is the real lane and `reim = 2` is the imaginary lane.

The mapping from the current logical production shape

```julia
weights[real_or_imag, component, harmonic_index(n, m)]
```

to the native buffer is:

```julia
buffer[basis_index(n, m, real_or_imag), batch, channel] =
    weights[real_or_imag, channel, harmonic_index(n, m)]
```

for each active channel. This mapping is exact; it does not change coefficient
normalization or operator conventions.

## Real Solid Harmonic Basis

The real solid harmonic basis uses the same physical
`basis_dof x batch x channel` layout. Its basis size is:

```text
Nreal(P)  = (P + 1)^2
basis_dof = Nreal(P)
```

The mode ordering is degree-major:

```text
mode_index(n, 0)      = n^2 + 1
mode_index(n, m, cos) = n^2 + 2m
mode_index(n, m, sin) = n^2 + 2m + 1
```

with `1 <= m <= n <= P`. For each degree `n`, the contiguous block
`n^2 + 1 : (n + 1)^2` stores:

```text
m = 0, m = 1 cos, m = 1 sin, ..., m = n cos, m = n sin
```

Task `008` owns the complex-to-real and real-to-complex transform signs and
scalings. This task only fixes real-basis indexing and buffer/view
requirements.

## Scratch and Aliasing

Scratch buffers used by the operator pipelines should be preallocated per CPU
thread or per GPU stream/batch. Scratch uses the same physical
`basis_dof x batch x channel` layout as source and destination coefficient
buffers unless a specific coupled operator stage explicitly documents a packed
input form.

Source, destination, and intermediate buffers for overwrite stages must not
alias unless that operator explicitly supports in-place use. This applies to
z-rotation, axis-swap, z-translation, and Lamb-Helmholtz intermediate stages.

The final inverse z-rotation remains the only accumulating stage:

```text
target <- target + Z_phi^-1 temporary
```

All earlier stages write into distinct destination or scratch storage with
overwrite semantics.

## Verification

`MATRIX_OPERATOR_REFACTOR/scripts/coefficient_buffer_layout_verify.jl` verifies:

- compressed complex basis size and basis-index contiguity for `P = 0, 1, 3,
  6, 9`;
- exact round trips between deterministic legacy
  `weights[real_or_imag, component, harmonic]` arrays and the native
  `basis_dof x batch x channel` buffer;
- one-channel and two-channel active layouts;
- fixed-channel slab density, with first-dimension stride `1` and
  second-dimension stride `basis_dof`;
- real-basis mode indices are contiguous and unique for the same expansion
  orders.

The generated summary is:

```text
MATRIX_OPERATOR_REFACTOR/data/coefficient_buffer_layout/verification_summary.md
```
