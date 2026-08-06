#------- EXPLICIT M2M/L2L Z-TRANSLATION BLOCKS (Matrix Operator Refactor, task 016) -------#

@inline _triangular_z_block_length(P) = ((P + 1) * (P + 2) * (P + 3)) ÷ 6

@inline function _triangular_z_block_offset(m, P)
    # sum_{mp=0}^{m-1} w_mp(w_mp+1)/2, w_mp = P - mp + 1
    sum_tri(j) = j * (j + 1) * (j + 2) ÷ 6
    return sum_tri(P + 1) - sum_tri(P - m + 1)
end

@inline m2m_z_block_length(P) = _triangular_z_block_length(P)
@inline l2l_z_block_length(P) = _triangular_z_block_length(P)

@inline m2m_z_block_offset(m, P) = _triangular_z_block_offset(m, P)
@inline l2l_z_block_offset(m, P) = _triangular_z_block_offset(m, P)

@inline function m2m_z_block_index(n, np, m, P)
    w = P - m + 1
    row = n - m
    col = np - m
    return m2m_z_block_offset(m, P) + (col * (2w - col + 1)) ÷ 2 + (row - col) + 1
end

@inline function l2l_z_block_index(n, np, m, P)
    row = n - m
    col = np - m
    return l2l_z_block_offset(m, P) + (col * (col + 1)) ÷ 2 + row + 1
end

"""
    m2m_z_blocks!(blocks, t, P)

Materialize the lower-triangular fixed-`m` M2M z-translation blocks
`U_m[n,np] = (-t)^(n-np)/(n-np)!` for `m <= np <= n <= P`.
"""
function m2m_z_blocks!(blocks, t, P)
    _t = -t
    @inbounds for m in 0:P
        for np in m:P
            coeff = one(eltype(blocks))
            for n in np:P
                blocks[m2m_z_block_index(n, np, m, P)] = coeff
                coeff *= _t / (n - np + 1)
            end
        end
    end
    return blocks
end

"""
    l2l_z_blocks!(blocks, t, P)

Materialize the upper-triangular fixed-`m` L2L z-translation blocks
`V_m[n,np] = (-t)^(np-n)/(np-n)!` for `m <= n <= np <= P`.
"""
function l2l_z_blocks!(blocks, t, P)
    _t = -t
    @inbounds for m in 0:P
        for np in m:P
            coeff = one(eltype(blocks))
            for n in np:-1:m
                blocks[l2l_z_block_index(n, np, m, P)] = coeff
                coeff *= _t / (np - n + 1)
            end
        end
    end
    return blocks
end

function apply_m2m_z!(out, in, blocks, P, lamb_helmholtz::Val{LH}, ::Val{:overwrite}) where LH
    i_out = 1
    @inbounds for n in 0:P
        for m in 0:n
            val1_real = zero(eltype(out))
            val1_imag = zero(eltype(out))
            if LH
                val2_real = zero(eltype(out))
                val2_imag = zero(eltype(out))
            end

            for np in m:n
                k = blocks[m2m_z_block_index(n, np, m, P)]
                i_in = harmonic_index(np, m)
                val1_real += k * in[1, 1, i_in]
                val1_imag += k * in[2, 1, i_in]
                if LH
                    val2_real += k * in[1, 2, i_in]
                    val2_imag += k * in[2, 2, i_in]
                end
            end

            out[1, 1, i_out] = val1_real
            out[2, 1, i_out] = val1_imag
            if LH
                out[1, 2, i_out] = val2_real
                out[2, 2, i_out] = val2_imag
            end
            i_out += 1
        end
    end
    return out
end

function apply_l2l_z!(out, in, blocks, P, lamb_helmholtz::Val{LH}, ::Val{:overwrite}) where LH
    i_out = 1
    @inbounds for n in 0:P
        for m in 0:n
            val1_real = zero(eltype(out))
            val1_imag = zero(eltype(out))
            if LH
                val2_real = zero(eltype(out))
                val2_imag = zero(eltype(out))
            end

            for np in n:P
                k = blocks[l2l_z_block_index(n, np, m, P)]
                i_in = harmonic_index(np, m)
                val1_real += k * in[1, 1, i_in]
                val1_imag += k * in[2, 1, i_in]
                if LH
                    val2_real += k * in[1, 2, i_in]
                    val2_imag += k * in[2, 2, i_in]
                end
            end

            out[1, 1, i_out] = val1_real
            out[2, 1, i_out] = val1_imag
            if LH
                out[1, 2, i_out] = val2_real
                out[2, 2, i_out] = val2_imag
            end
            i_out += 1
        end
    end
    return out
end

#------- EXPLICIT M2L Z-TRANSLATION BLOCKS (Matrix Operator Refactor, task 011) -------#
#
# The z-aligned multipole-to-local translation is block-diagonal in the azimuthal
# order m: for a fixed m, each output local coefficient (n, m) is a dense linear
# combination of the source multipole coefficients (np, m) for np = m:P,
#
#     L_n^m = sum(np = m:P) K_m[n, np] * M_np^m,   K_m[n, np] = (n + np)! / t^(n + np + 1)
#
# where t is the z-axis separation. The same real scalar K_m[n, np] multiplies
# both the real and imaginary lanes, and (for Val(true)) both the φ and χ
# component channels. There is no conjugation or extra sign in the z-aligned
# block; azimuthal phase signs are owned by the approved z-rotation operators
# (task 010). See the approved theory artifact
# MATRIX_OPERATOR_REFACTOR/theory/m2l-z-translation-scaling.md.
#
# This stage materializes the fixed-m blocks explicitly (rather than fusing the
# recurrence into a single per-call kernel) so the blocks can be built once and
# applied to many source/target pairs that share the same integer z-offset, which
# is exactly the reuse the later radix/offset-class batching (tasks 020/021) and
# the GPU path (task 022) exploit. The distance scaling is evaluated by the same
# stable recurrence used by translate_multipole_to_local_z! in src/translate.jl,
# in the same multiply order, so a materialize-then-apply is bit-for-bit identical
# to production.
#
# These operate on the existing production coefficient layout
# weights[real_or_imag, component, harmonic_index]; native flat buffers are a
# later task (017). The functions are intentionally internal/non-exported.

"""
    m2l_z_block_length(P)

Number of `TF` entries needed to store all fixed-`m` M2L z-translation blocks for
degrees `0 <= m <= n, np <= P`. Each block `m` is a dense `(P - m + 1) x
(P - m + 1)` matrix, so the total is `sum(k=1:P+1) k^2 = (P+1)(P+2)(2P+3) / 6`.
"""
@inline m2l_z_block_length(P) = ((P + 1) * (P + 2) * (2 * P + 3)) ÷ 6

"""
    m2l_z_block_offset(m, P)

Flat-array offset (number of entries preceding) the fixed-`m` block within a
buffer filled by [`m2l_z_blocks!`](@ref). Blocks are stored in increasing `m`,
so the offset is the summed size of blocks `0:m-1`. Within block `m`, the entry
for output degree `n` and source degree `np` (both in `m:P`) lives at
`offset + (row - 1) + (col - 1) * w + 1`, where `row = n - m + 1`,
`col = np - m + 1`, and `w = P - m + 1` is the block width (column-major).
"""
@inline function m2l_z_block_offset(m, P)
    # sum_{mp=0}^{m-1} (P - mp + 1)^2 = (sum of squares 1:P+1) - (sum of squares 1:P-m+1).
    # Closed form (matches S_block_offset's treatment); sum_sq(j) = j(j+1)(2j+1)/6 is
    # an exact integer division since j(j+1)(2j+1) is always divisible by 6.
    sum_sq(j) = j * (j + 1) * (2 * j + 1) ÷ 6
    return sum_sq(P + 1) - sum_sq(P - m + 1)
end

@inline function m2l_z_block_index(n, np, m, P)
    w = P - m + 1
    return m2l_z_block_offset(m, P) + (n - m) + (np - m) * w + 1
end

"""
    m2l_z_blocks!(blocks, t, P)

Materialize the fixed-`m` M2L z-translation blocks `K_m[n, np] = (n + np)! /
t^(n + np + 1)` for all `0 <= m <= n, np <= P` into the flat buffer `blocks`
(length at least `m2l_z_block_length(P)`), stored column-major per `m`-block (see
[`m2l_z_block_offset`](@ref)).

The entries are filled by the approved stable recurrence with `rho = inv(t)`
(no separately formed factorial or power):

- first entry of each output row: `K_m[n, m] = (n + m)! * rho^(n + m + 1)`,
  advanced down output degree by `K_m[n + 1, m] = K_m[n, m] * (n + m + 1) * rho`;
- across source degree: `K_m[n, np + 1] = K_m[n, np] * (n + np + 1) * rho`.

The multiply order matches `translate_multipole_to_local_z!` in `src/translate.jl`
so that [`apply_m2l_z!`](@ref) reproduces production bit-for-bit.
"""
function m2l_z_blocks!(blocks, t, P)
    rho = inv(t)

    # n!_t_np1 tracks K_m[n, m] for the current (n, m=0): (n + 0)! * rho^(n + 1).
    # It is advanced down n by *(n+1)*rho, exactly as production advances n!_t_np1.
    n!_t_np1 = rho
    @inbounds for n in 0:P
        # K_m[n, m] for increasing m, starting from m = 0 at K = n!_t_np1.
        n_m!_t_nmp1 = n!_t_np1
        n_m = n
        for m in 0:n
            w = P - m + 1
            base = m2l_z_block_offset(m, P) + (n - m)  # column 0 (np = m), row (n-m)

            # walk source degree np = m:P with the across-source recurrence
            n_np! = n_m!_t_nmp1
            n_np = n_m
            for np in m:P
                blocks[base + (np - m) * w + 1] = n_np!
                n_np += 1
                n_np! *= n_np * rho
            end

            # advance K_m[n, m] -> K_{m+1}[n, m+1] start value
            n_m += 1
            n_m!_t_nmp1 *= n_m * rho
        end
        n!_t_np1 *= (n + 1) * rho
    end

    return blocks
end

"""
    apply_m2l_z!(out, in, blocks, P, lamb_helmholtz::Val, ::Val{:overwrite})

Apply the explicit fixed-`m` M2L z-translation blocks (see [`m2l_z_blocks!`](@ref))
to the source coefficients `in`, overwriting `out`. Both arrays use the production
layout `[real_or_imag, component, harmonic_index]`. For each output `(n, m)`,

    out[:, c, i(n,m)] = sum(np = m:P) K_m[n, np] * in[:, c, i(np,m)]

with `i = harmonic_index`, summed in increasing `np` order to match
`translate_multipole_to_local_z!`. The z-aligned M2L block overwrites its
destination (it does not accumulate); accumulation into the target expansion
happens later in the full M2L pipeline via the inverse z-rotation.

`lamb_helmholtz` selects whether the second (χ) component channel is processed in
addition to the scalar φ channel; both channels use the same real blocks.
"""
function apply_m2l_z!(out, in, blocks, P, lamb_helmholtz::Val{LH}, ::Val{:overwrite}) where LH
    i_out = 1
    @inbounds for n in 0:P
        for m in 0:n
            val1_real = zero(eltype(out))
            val1_imag = zero(eltype(out))
            if LH
                val2_real = zero(eltype(out))
                val2_imag = zero(eltype(out))
            end

            w = P - m + 1
            base = m2l_z_block_offset(m, P) + (n - m)  # column 0 (np = m), row (n-m)
            for np in m:P
                k = blocks[base + (np - m) * w + 1]
                i_in = harmonic_index(np, m)
                val1_real += k * in[1, 1, i_in]
                val1_imag += k * in[2, 1, i_in]
                if LH
                    val2_real += k * in[1, 2, i_in]
                    val2_imag += k * in[2, 2, i_in]
                end
            end

            out[1, 1, i_out] = val1_real
            out[2, 1, i_out] = val1_imag
            if LH
                out[1, 2, i_out] = val2_real
                out[2, 2, i_out] = val2_imag
            end

            i_out += 1
        end
    end

    return out
end

"""
    apply_m2l_z!(out, in, blocks, basis_info::OperatorBasisInfo, ::Val{:overwrite})

Order-aware M2L z-translation driven by the task-`009` order accessors. The scalar
φ channel (component 1) is translated through `basis_info.orders.P_phi`; for
`Val(true)` the Lamb-Helmholtz χ channel (component 2) is translated through the
padded active order `basis_info.orders.P_active = P_chi = P_phi + 1`. The `blocks`
buffer must be sized for `P_active` (see [`m2l_z_blocks!`](@ref)); φ rows above
`P_phi` are padding/scratch and are not written as physical output.
"""
function apply_m2l_z!(out, in, blocks, basis_info::OperatorBasisInfo{<:CompressedComplexBasis,LH}, ::Val{:overwrite}) where LH
    P_active = basis_info.orders.P_active
    P_phi = basis_info.orders.P_phi

    # φ channel (component 1): translate through P_phi only.
    i_out = 1
    @inbounds for n in 0:P_phi
        for m in 0:n
            val_real = zero(eltype(out))
            val_imag = zero(eltype(out))
            w = P_active - m + 1
            base = m2l_z_block_offset(m, P_active) + (n - m)
            for np in m:P_phi
                k = blocks[base + (np - m) * w + 1]
                i_in = harmonic_index(np, m)
                val_real += k * in[1, 1, i_in]
                val_imag += k * in[2, 1, i_in]
            end
            out[1, 1, i_out] = val_real
            out[2, 1, i_out] = val_imag
            i_out += 1
        end
    end

    # χ channel (component 2): translate through the padded P_active = P_phi + 1.
    if LH
        i_out = 1
        @inbounds for n in 0:P_active
            for m in 0:n
                val_real = zero(eltype(out))
                val_imag = zero(eltype(out))
                w = P_active - m + 1
                base = m2l_z_block_offset(m, P_active) + (n - m)
                for np in m:P_active
                    k = blocks[base + (np - m) * w + 1]
                    i_in = harmonic_index(np, m)
                    val_real += k * in[1, 2, i_in]
                    val_imag += k * in[2, 2, i_in]
                end
                out[1, 2, i_out] = val_real
                out[2, 2, i_out] = val_imag
                i_out += 1
            end
        end
    end

    return out
end

#------- EXPLICIT LAMB-HELMHOLTZ OPERATORS (Matrix Operator Refactor, task 012) -------#
#
# The Lamb-Helmholtz transforms couple the φ (component 1) and χ (component 2)
# channels after a z-aligned translation. They are sparse by construction (see the
# approved theory artifact MATRIX_OPERATOR_REFACTOR/theory/lamb-helmholtz-operator-form.md):
#
#   - no cross-m coupling;
#   - same-degree φ-from-χ coupling only;
#   - nearest-neighbor χ-from-χ coupling in degree;
#   - identical real scalar coupling for the real and imaginary lanes.
#
# Two real scalars per stored coefficient (n, m) fully describe the operator:
#
#   A[i] : same-degree φ-from-χ factor    (mixes χ_n into φ_n)
#   B[i] : nearest-neighbor χ-from-χ factor (mixes a neighbor χ into χ_n)
#
# with i = harmonic_index(n, m). The multipole and local sides differ in the
# factor formulas and in the neighbor direction:
#
#   multipole:  A = r*m/(n+1),  B = r/n        χ neighbor is χ_{n-1} (lower degree)
#   local:      A = r*m/n,      B = r/(n+1)    χ neighbor is χ_{n+1} (upper degree)
#
# Like the M2L z blocks above, the factors are materialized once (they depend only
# on r and the degrees) so they can be reused across many source/target pairs that
# share the same integer z-offset class (radix/offset-class batching, tasks
# 020/021; GPU path, task 022). The factor formulas and multiply order match
# transform_lamb_helmholtz_multipole! / transform_lamb_helmholtz_local! in
# src/translate.jl, so the materialize-then-apply path is bit-for-bit identical to
# production. The apply uses separate in/out buffers with overwrite semantics
# (accumulation into the target is left to the inverse z-rotation in the full
# pipeline). Because the in-place production loops are ordered to always read the
# original (pre-transform) χ values, reading every term from `in` reproduces them
# exactly. These operate on the production layout
# weights[real_or_imag, component, harmonic_index]; native flat buffers are task
# 017, and the functions are intentionally internal/non-exported.

"""
    lamb_helmholtz_multipole_coeffs!(A, B, r, P)

Materialize the multipole-side Lamb-Helmholtz factors over the compressed harmonic
index: `A[harmonic_index(n,m)] = r*m/(n+1)` (same-degree φ-from-χ) and
`B[harmonic_index(n,m)] = r/n` (lower-neighbor χ-from-χ, `χ_{n-1} -> χ_n`) for all
`0 <= m <= n <= P`. The `n = 0` entry of `B` is set to zero (no lower neighbor);
it is never read by [`apply_lamb_helmholtz_multipole!`](@ref). `A` and `B` must
each have length at least `((P+1)*(P+2))>>1`.
"""
function lamb_helmholtz_multipole_coeffs!(A, B, r, P)
    i = 1
    @inbounds for n in 0:P
        for m in 0:n
            A[i] = r * m / (n + 1)
            B[i] = n == 0 ? zero(eltype(B)) : r / n
            i += 1
        end
    end
    return A, B
end

"""
    lamb_helmholtz_local_coeffs!(A, B, r, P)

Materialize the local-side Lamb-Helmholtz factors over the compressed harmonic
index: `A[harmonic_index(n,m)] = r*m/n` (same-degree φ-from-χ, zero at `n = 0`)
and `B[harmonic_index(n,m)] = r/(n+1)` (upper-neighbor χ-from-χ,
`χ_{n+1} -> χ_n`) for all `0 <= m <= n <= P`. `A` and `B` must each have length at
least `((P+1)*(P+2))>>1`.
"""
function lamb_helmholtz_local_coeffs!(A, B, r, P)
    i = 1
    @inbounds for n in 0:P
        for m in 0:n
            A[i] = n == 0 ? zero(eltype(A)) : r * m / n
            B[i] = r / (n + 1)
            i += 1
        end
    end
    return A, B
end

"""
    apply_lamb_helmholtz_multipole!(out, in, A, B, P, ::Val{:overwrite})

Apply the multipole-side Lamb-Helmholtz transform (factors from
[`lamb_helmholtz_multipole_coeffs!`](@ref)) to `in`, overwriting `out`. Both arrays
use the production layout `[real_or_imag, component, harmonic_index]`. For each
stored `(n, m)` with `i = harmonic_index(n, m)` and `a = A[i]`, `b = B[i]`:

    φ_n^m  = φ̂_n^m + a * (im-rotated χ̂_n^m)     (real: +a*χ̂_im, imag: -a*χ̂_re)
    χ_n^m  = χ̂_n^m + b * χ̂_{n-1}^m              (b applied only for n > m)

Reproduces `transform_lamb_helmholtz_multipole!` bit-for-bit. The `(0,0)`
coefficient is copied through unchanged (left untouched by production).
"""
function apply_lamb_helmholtz_multipole!(out, in, A, B, P, ::Val{:overwrite})
    @inbounds for n in 0:P
        for m in 0:n
            i = harmonic_index(n, m)

            # φ channel: same-degree coupling from χ_n (zero factor when m == 0)
            a = A[i]
            chi_re = in[1, 2, i]
            chi_im = in[2, 2, i]
            out[1, 1, i] = in[1, 1, i] + a * chi_im
            out[2, 1, i] = in[2, 1, i] - a * chi_re

            # χ channel: lower-neighbor coupling χ_{n-1} -> χ_n (exists only for n > m)
            if n > m
                b = B[i]
                i_lo = i - n  # harmonic_index(n-1, m)
                out[1, 2, i] = chi_re + b * in[1, 2, i_lo]
                out[2, 2, i] = chi_im + b * in[2, 2, i_lo]
            else
                out[1, 2, i] = chi_re
                out[2, 2, i] = chi_im
            end
        end
    end
    return out
end

"""
    apply_lamb_helmholtz_local!(out, in, A, B, P, ::Val{:overwrite})

Apply the local-side Lamb-Helmholtz transform (factors from
[`lamb_helmholtz_local_coeffs!`](@ref)) to `in`, overwriting `out`. Both arrays use
the production layout `[real_or_imag, component, harmonic_index]`. For each stored
`(n, m)` with `i = harmonic_index(n, m)`, `a = A[i]`, `b = B[i]`:

    φ_n^m  = φ̂_n^m - a * (im-rotated χ̂_n^m)     (real: -a*χ̂_im, imag: +a*χ̂_re), n > 0
    χ_n^m  = χ̂_n^m - b * χ̂_{n+1}^m              (only for n < P; χ_P^m copied through)

Reproduces `transform_lamb_helmholtz_local!` bit-for-bit, including the truncation
`χ_{P+1}^m = 0` at the top degree. The `(0,0)` φ coefficient is copied through.
"""
function apply_lamb_helmholtz_local!(out, in, A, B, P, ::Val{:overwrite})
    @inbounds for n in 0:P
        for m in 0:n
            i = harmonic_index(n, m)

            # φ channel: same-degree coupling from χ_n, skipped for n == 0
            if n > 0
                a = A[i]
                out[1, 1, i] = in[1, 1, i] - a * in[2, 2, i]
                out[2, 1, i] = in[2, 1, i] + a * in[1, 2, i]
            else
                out[1, 1, i] = in[1, 1, i]
                out[2, 1, i] = in[2, 1, i]
            end

            # χ channel: upper-neighbor coupling χ_{n+1} -> χ_n; truncated at n == P
            if n < P
                b = B[i]
                i_hi = i + (n + 1)  # harmonic_index(n+1, m)
                out[1, 2, i] = in[1, 2, i] - b * in[1, 2, i_hi]
                out[2, 2, i] = in[2, 2, i] - b * in[2, 2, i_hi]
            else
                out[1, 2, i] = in[1, 2, i]
                out[2, 2, i] = in[2, 2, i]
            end
        end
    end
    return out
end

"""
    apply_lamb_helmholtz_multipole!(out, in, A, B, basis_info::OperatorBasisInfo, ::Val{:overwrite})

Order-aware multipole-side Lamb-Helmholtz transform driven by the task-`009` order
accessors. The physical φ output is produced through `orders.P_phi`, while the χ
channel is carried through the padded active order `orders.P_active = P_phi + 1`
so the lower-neighbor recurrence has its full χ input. `A`/`B` must be sized for
`P_active` (see [`lamb_helmholtz_multipole_coeffs!`](@ref)). For `Val(false)`,
`P_active == P_phi` and the χ channel is absent.
"""
function apply_lamb_helmholtz_multipole!(out, in, A, B, basis_info::OperatorBasisInfo{<:CompressedComplexBasis,LH}, ::Val{:overwrite}) where LH
    P_phi = basis_info.orders.P_phi
    P_active = basis_info.orders.P_active

    # φ channel (component 1): physical output through P_phi.
    @inbounds for n in 0:P_phi
        for m in 0:n
            i = harmonic_index(n, m)
            if LH
                a = A[i]
                out[1, 1, i] = in[1, 1, i] + a * in[2, 2, i]
                out[2, 1, i] = in[2, 1, i] - a * in[1, 2, i]
            else
                out[1, 1, i] = in[1, 1, i]
                out[2, 1, i] = in[2, 1, i]
                out[1, 2, i] = zero(eltype(out))
                out[2, 2, i] = zero(eltype(out))
            end
        end
    end

    # χ channel (component 2): carried through the padded P_active.
    if LH
        @inbounds for n in 0:P_active
            for m in 0:n
                i = harmonic_index(n, m)
                chi_re = in[1, 2, i]
                chi_im = in[2, 2, i]
                if n > m
                    b = B[i]
                    i_lo = i - n
                    out[1, 2, i] = chi_re + b * in[1, 2, i_lo]
                    out[2, 2, i] = chi_im + b * in[2, 2, i_lo]
                else
                    out[1, 2, i] = chi_re
                    out[2, 2, i] = chi_im
                end
            end
        end
    end
    return out
end

"""
    apply_lamb_helmholtz_local!(out, in, A, B, basis_info::OperatorBasisInfo, ::Val{:overwrite})

Order-aware local-side Lamb-Helmholtz transform driven by the task-`009` order
accessors. The physical φ output is produced through `orders.P_phi`. The χ channel
is carried through the padded active order `orders.P_active = P_phi + 1`, which is
exactly what supplies the upper-neighbor `χ_{P_phi+1} -> χ_{P_phi}` row required by
`theory/lamb-helmholtz-accuracy-order.md`: the χ truncation moves up to
`P_active`, so every retained χ row `n <= P_phi` is computed with its true upper
neighbor instead of a zeroed boundary value. `A`/`B` must be sized for `P_active`
(see [`lamb_helmholtz_local_coeffs!`](@ref)). For `Val(false)`,
`P_active == P_phi` and the χ channel is absent.
"""
function apply_lamb_helmholtz_local!(out, in, A, B, basis_info::OperatorBasisInfo{<:CompressedComplexBasis,LH}, ::Val{:overwrite}) where LH
    P_phi = basis_info.orders.P_phi
    P_active = basis_info.orders.P_active

    # φ channel (component 1): physical output through P_phi.
    @inbounds for n in 0:P_phi
        for m in 0:n
            i = harmonic_index(n, m)
            if !LH
                out[1, 1, i] = in[1, 1, i]
                out[2, 1, i] = in[2, 1, i]
                out[1, 2, i] = zero(eltype(out))
                out[2, 2, i] = zero(eltype(out))
            elseif n > 0
                a = A[i]
                out[1, 1, i] = in[1, 1, i] - a * in[2, 2, i]
                out[2, 1, i] = in[2, 1, i] + a * in[1, 2, i]
            else
                out[1, 1, i] = in[1, 1, i]
                out[2, 1, i] = in[2, 1, i]
            end
        end
    end

    # χ channel (component 2): carried through P_active; upper-neighbor row at
    # n = P_phi uses the real χ_{P_phi+1} value, truncation moves to P_active.
    if LH
        @inbounds for n in 0:P_active
            for m in 0:n
                i = harmonic_index(n, m)
                if n < P_active
                    b = B[i]
                    i_hi = i + (n + 1)
                    out[1, 2, i] = in[1, 2, i] - b * in[1, 2, i_hi]
                    out[2, 2, i] = in[2, 2, i] - b * in[2, 2, i_hi]
                else
                    out[1, 2, i] = in[1, 2, i]
                    out[2, 2, i] = in[2, 2, i]
                end
            end
        end
    end
    return out
end

#------- NATIVE FLAT KERNELS (Matrix Operator Refactor, task 017) -------#
#
# Flat ragged-buffer counterparts of the per-column z-translation and
# Lamb-Helmholtz kernels above. They consume FlatCoefficientBuffer storage through
# its channel matrices: the φ channel is processed through P_phi and the χ channel
# through P_active = P_phi + 1 (Val(true)), so the φ matrix carries no padding rows
# and the task-014/016 `_zero_phi_padding!` step is unnecessary. Arithmetic and
# multiply order are identical to the legacy [2,2,nh] kernels above (kept as the
# parity reference); only the indexing moves to
# `flat_basis_index(n,m,reim) = 2*(harmonic_index(n,m)-1)+reim` over a column `j`.
# The distance `blocks`/`A`/`B` coefficient buffers are sized for `P_active`, so all
# block indexing passes `P_active`; the φ loops simply stop at `P_phi`, which
# reproduces the zeroed-φ-padding result of the legacy path (rotations and the
# fixed-`m` z-translation never feed a φ row above `P_phi` back into a physical
# φ_n, n <= P_phi).

function apply_m2l_z_flat!(out::FlatCoefficientBuffer{TF,A,B,LH}, in::FlatCoefficientBuffer, j, blocks, ::Val{:overwrite}) where {TF,A,B,LH}
    P_active = out.basis_info.orders.P_active
    P_phi = out.basis_info.orders.P_phi
    op = phi_slab(out); ip = phi_slab(in)
    @inbounds for n in 0:P_phi
        for m in 0:n
            vr = zero(TF); vi = zero(TF)
            w = P_active - m + 1
            base = m2l_z_block_offset(m, P_active) + (n - m)
            for np in m:P_phi
                k = blocks[base + (np - m) * w + 1]
                fr = flat_basis_index(np, m, 1)
                vr += k * ip[fr, j]
                vi += k * ip[fr + 1, j]
            end
            fr = flat_basis_index(n, m, 1)
            op[fr, j] = vr
            op[fr + 1, j] = vi
        end
    end
    if LH
        oc = chi_slab(out); ic = chi_slab(in)
        @inbounds for n in 0:P_active
            for m in 0:n
                vr = zero(TF); vi = zero(TF)
                w = P_active - m + 1
                base = m2l_z_block_offset(m, P_active) + (n - m)
                for np in m:P_active
                    k = blocks[base + (np - m) * w + 1]
                    fr = flat_basis_index(np, m, 1)
                    vr += k * ic[fr, j]
                    vi += k * ic[fr + 1, j]
                end
                fr = flat_basis_index(n, m, 1)
                oc[fr, j] = vr
                oc[fr + 1, j] = vi
            end
        end
    end
    return out
end

function apply_m2m_z_flat!(out::FlatCoefficientBuffer{TF,A,B,LH}, in::FlatCoefficientBuffer, j, blocks, ::Val{:overwrite}) where {TF,A,B,LH}
    P_active = out.basis_info.orders.P_active
    P_phi = out.basis_info.orders.P_phi
    op = phi_slab(out); ip = phi_slab(in)
    @inbounds for n in 0:P_phi
        for m in 0:n
            vr = zero(TF); vi = zero(TF)
            for np in m:n
                k = blocks[m2m_z_block_index(n, np, m, P_active)]
                fr = flat_basis_index(np, m, 1)
                vr += k * ip[fr, j]
                vi += k * ip[fr + 1, j]
            end
            fr = flat_basis_index(n, m, 1)
            op[fr, j] = vr
            op[fr + 1, j] = vi
        end
    end
    if LH
        oc = chi_slab(out); ic = chi_slab(in)
        @inbounds for n in 0:P_active
            for m in 0:n
                vr = zero(TF); vi = zero(TF)
                for np in m:n
                    k = blocks[m2m_z_block_index(n, np, m, P_active)]
                    fr = flat_basis_index(np, m, 1)
                    vr += k * ic[fr, j]
                    vi += k * ic[fr + 1, j]
                end
                fr = flat_basis_index(n, m, 1)
                oc[fr, j] = vr
                oc[fr + 1, j] = vi
            end
        end
    end
    return out
end

function apply_l2l_z_flat!(out::FlatCoefficientBuffer{TF,A,B,LH}, in::FlatCoefficientBuffer, j, blocks, ::Val{:overwrite}) where {TF,A,B,LH}
    P_active = out.basis_info.orders.P_active
    P_phi = out.basis_info.orders.P_phi
    op = phi_slab(out); ip = phi_slab(in)
    @inbounds for n in 0:P_phi
        for m in 0:n
            vr = zero(TF); vi = zero(TF)
            for np in n:P_phi
                k = blocks[l2l_z_block_index(n, np, m, P_active)]
                fr = flat_basis_index(np, m, 1)
                vr += k * ip[fr, j]
                vi += k * ip[fr + 1, j]
            end
            fr = flat_basis_index(n, m, 1)
            op[fr, j] = vr
            op[fr + 1, j] = vi
        end
    end
    if LH
        oc = chi_slab(out); ic = chi_slab(in)
        @inbounds for n in 0:P_active
            for m in 0:n
                vr = zero(TF); vi = zero(TF)
                for np in n:P_active
                    k = blocks[l2l_z_block_index(n, np, m, P_active)]
                    fr = flat_basis_index(np, m, 1)
                    vr += k * ic[fr, j]
                    vi += k * ic[fr + 1, j]
                end
                fr = flat_basis_index(n, m, 1)
                oc[fr, j] = vr
                oc[fr + 1, j] = vi
            end
        end
    end
    return out
end

# Lamb-Helmholtz flat coupling. Only invoked for Val(true) (the driver guards on
# LH), so there is no `!LH` branch. φ writes (channel 1) and χ reads (channel 2) are
# always distinct matrices, so the multipole lower-neighbor and local upper-neighbor
# recurrences keep the same in-place safety as the legacy kernels: the local form
# (used in-place by M2L) reads only the same-or-higher χ degree before overwriting.
function apply_lamb_helmholtz_multipole_flat!(out::FlatCoefficientBuffer, in::FlatCoefficientBuffer, j, Acoef, Bcoef, ::Val{:overwrite})
    P_phi = out.basis_info.orders.P_phi
    P_active = out.basis_info.orders.P_active
    op = phi_slab(out); ip = phi_slab(in); oc = chi_slab(out); ic = chi_slab(in)
    @inbounds for n in 0:P_phi
        for m in 0:n
            i = harmonic_index(n, m)
            a = Acoef[i]
            fr = flat_basis_index(n, m, 1)
            chi_re = ic[fr, j]; chi_im = ic[fr + 1, j]
            op[fr, j] = ip[fr, j] + a * chi_im
            op[fr + 1, j] = ip[fr + 1, j] - a * chi_re
        end
    end
    @inbounds for n in 0:P_active
        for m in 0:n
            i = harmonic_index(n, m)
            fr = flat_basis_index(n, m, 1)
            chi_re = ic[fr, j]; chi_im = ic[fr + 1, j]
            if n > m
                b = Bcoef[i]
                flo = flat_basis_index(n - 1, m, 1)
                oc[fr, j] = chi_re + b * ic[flo, j]
                oc[fr + 1, j] = chi_im + b * ic[flo + 1, j]
            else
                oc[fr, j] = chi_re
                oc[fr + 1, j] = chi_im
            end
        end
    end
    return out
end

function apply_lamb_helmholtz_local_flat!(out::FlatCoefficientBuffer, in::FlatCoefficientBuffer, j, Acoef, Bcoef, ::Val{:overwrite})
    P_phi = out.basis_info.orders.P_phi
    P_active = out.basis_info.orders.P_active
    op = phi_slab(out); ip = phi_slab(in); oc = chi_slab(out); ic = chi_slab(in)
    @inbounds for n in 0:P_phi
        for m in 0:n
            i = harmonic_index(n, m)
            fr = flat_basis_index(n, m, 1)
            if n > 0
                a = Acoef[i]
                op[fr, j] = ip[fr, j] - a * ic[fr + 1, j]
                op[fr + 1, j] = ip[fr + 1, j] + a * ic[fr, j]
            else
                op[fr, j] = ip[fr, j]
                op[fr + 1, j] = ip[fr + 1, j]
            end
        end
    end
    @inbounds for n in 0:P_active
        for m in 0:n
            fr = flat_basis_index(n, m, 1)
            i = harmonic_index(n, m)
            if n < P_active
                b = Bcoef[i]
                fhi = flat_basis_index(n + 1, m, 1)
                oc[fr, j] = ic[fr, j] - b * ic[fhi, j]
                oc[fr + 1, j] = ic[fr + 1, j] - b * ic[fhi + 1, j]
            else
                oc[fr, j] = ic[fr, j]
                oc[fr + 1, j] = ic[fr + 1, j]
            end
        end
    end
    return out
end

# Per-column repack between a ragged FlatCoefficientBuffer column and a legacy
# [2,2,nh(P_active)] frame, used only by the materialized-y stage so it can reuse
# the production-parity y/z rotation kernels (src/rotate.jl) unchanged. The legacy
# frame is zeroed first; φ above P_phi stays zero (no padding leak), χ fills through
# P_active. Cost is O(basis_dof) per column — negligible vs the y-rotation's O(P^3).
function _pack_flat_column!(legacy, buf::FlatCoefficientBuffer, j, P_phi, P_active, ::Val{LH}) where LH
    legacy .= zero(eltype(legacy))
    ph = phi_slab(buf)
    @inbounds for n in 0:P_phi
        for m in 0:n
            i = harmonic_index(n, m); fr = flat_basis_index(n, m, 1)
            legacy[1, 1, i] = ph[fr, j]; legacy[2, 1, i] = ph[fr + 1, j]
        end
    end
    if LH
        ch = chi_slab(buf)
        @inbounds for n in 0:P_active
            for m in 0:n
                i = harmonic_index(n, m); fr = flat_basis_index(n, m, 1)
                legacy[1, 2, i] = ch[fr, j]; legacy[2, 2, i] = ch[fr + 1, j]
            end
        end
    end
    return legacy
end

function _unpack_flat_column!(buf::FlatCoefficientBuffer, legacy, j, P_phi, P_active, ::Val{LH}) where LH
    ph = phi_slab(buf)
    @inbounds for n in 0:P_phi
        for m in 0:n
            i = harmonic_index(n, m); fr = flat_basis_index(n, m, 1)
            ph[fr, j] = legacy[1, 1, i]; ph[fr + 1, j] = legacy[2, 1, i]
        end
    end
    if LH
        ch = chi_slab(buf)
        @inbounds for n in 0:P_active
            for m in 0:n
                i = harmonic_index(n, m); fr = flat_basis_index(n, m, 1)
                ch[fr, j] = legacy[1, 2, i]; ch[fr + 1, j] = legacy[2, 2, i]
            end
        end
    end
    return buf
end

function _unpack_flat_column_accumulate!(buf::FlatCoefficientBuffer, legacy, j, P_phi, P_active, ::Val{LH}) where LH
    ph = phi_slab(buf)
    @inbounds for n in 0:P_phi
        for m in 0:n
            i = harmonic_index(n, m); fr = flat_basis_index(n, m, 1)
            ph[fr, j] += legacy[1, 1, i]; ph[fr + 1, j] += legacy[2, 1, i]
        end
    end
    if LH
        ch = chi_slab(buf)
        @inbounds for n in 0:P_active
            for m in 0:n
                i = harmonic_index(n, m); fr = flat_basis_index(n, m, 1)
                ch[fr, j] += legacy[1, 2, i]; ch[fr + 1, j] += legacy[2, 2, i]
            end
        end
    end
    return buf
end

#------- FULL M2L OPERATOR PIPELINE (Matrix Operator Refactor, task 014) -------#
#
# Compose the complete multipole-to-local translation from the explicit operator
# stages built in tasks 010-013c, validated side-by-side against the production
# `multipole_to_local!` / `multipole_to_local_II!` (this is the parity-only first
# pass per the 008b re-plan; production internals are not replaced — that is task
# 023). The production pipeline is, per source/target offset
# `r, θ, ϕ = cartesian_to_spherical(target.center - source.center)`:
#
#   1. rotate_z!(ϕ)                  : source  -> aligned-φ
#   2. rotate_multipole_y!(θ)        : aligned -> aligned (ζ y-rotation)
#   3. translate_multipole_to_local_z!(r)
#   4. transform_lamb_helmholtz_local!(r)   (LH only, in place)
#   5. back_rotate_local_y!(θ)       (η y-rotation)
#   6. back_rotate_z!(ϕ)             : accumulate into target
#
# The two swappable variants differ only in stages 2/5 (the arbitrary-angle
# y-alignment): MaterializedYRotationM2L reconstructs Ts(θ) per column from the
# cached S_pos/S_neg blocks (task 013); FactoredRotationM2L applies the genuinely
# factored Z_phi -> Y(θ) -> ... -> inverse Z_phi using the fixed per-degree mode
# matrices U_n/V_n (task 013c, Plain-H modes `y_*_U`/`y_*_V`, never the 013b
# T_y_*90 primitives). Both reuse the shared task-011 z-translation blocks and
# task-012 Lamb-Helmholtz coupling.
#
# The whole pipeline runs at a single uniform order `P = basis_info.orders.P_active`.
# For `Val(true)`, φ is physical only through `P_phi` while χ is carried at
# `P_active = P_phi + 1` per theory/lamb-helmholtz-accuracy-order.md; the φ padding
# rows are zeroed before z-translation and again before return alignment so the
# nonphysical top φ row cannot leak into the output.

"""
    m2l_operator_batch!(op, targets, sources, phis, thetas, rs, invariant_cache, scratch, lamb_helmholtz)

Apply the full batched M2L operator pipeline (task 014). `op` is the operator tag
([`MaterializedYRotationM2L`](@ref) or [`FactoredRotationM2L`](@ref)). `sources`
and `targets` are native [`FlatCoefficientBuffer`](@ref) batches; `phis`, `thetas`,
`rs` are length-`B` vectors of the per-pair rotation angles and z-separation.
`targets` is **accumulated** into (matching production `back_rotate_z!`), so callers
must zero it first.

Uses the two lifetime-aliased flat working buffers and the embedded per-column
scratch of `scratch::M2LOperatorScratch`. The pipeline runs at
`invariant_cache.basis_info.orders.P_active`; the ragged φ/χ buffers carry φ through
`P_phi` and χ through `P_active`, so no φ-padding zeroing is needed (the order-aware
flat kernels never feed a φ row above `P_phi` into the physical output).
"""
function m2l_operator_batch!(op::AbstractM2LOperator, targets::FlatCoefficientBuffer, sources::FlatCoefficientBuffer, phis, thetas, rs, invariant_cache, scratch::M2LOperatorScratch, lamb_helmholtz::Val{LH}) where LH
    _require_compressed_complex_operator_buffers(targets, sources, invariant_cache, scratch)
    P = invariant_cache.basis_info.orders.P_active
    nbatch = length(phis)
    A = scratch.work_a
    B = scratch.work_b

    # stage 1: source alignment   sources -> A   (B used as scratch)
    _m2l_source_alignment!(op, A, sources, B, phis, thetas, invariant_cache, scratch, lamb_helmholtz, nbatch)

    # stage 2: z-translation + Lamb-Helmholtz   A -> B   (per column / per distance)
    @inbounds for j in 1:nbatch
        m2l_z_blocks!(scratch.blocks, rs[j], P)
        apply_m2l_z_flat!(B, A, j, scratch.blocks, Val(:overwrite))
        if LH
            lamb_helmholtz_local_coeffs!(scratch.lh_A, scratch.lh_B, rs[j], P)
            apply_lamb_helmholtz_local_flat!(B, B, j, scratch.lh_A, scratch.lh_B, Val(:overwrite))
        end
    end

    # stage 3: return alignment   B -> targets (accumulate)   (A used as scratch)
    _m2l_return_alignment!(op, targets, B, A, phis, thetas, invariant_cache, scratch, lamb_helmholtz, nbatch)

    return targets
end

#--- source alignment (stages 1-2: Z_phi then Y(θ)) ---#
# Materialized path repacks each column to a legacy [2,2,nh] frame so it can reuse
# the production-parity z/y rotation kernels unchanged (the φ padding stays zero, so
# physical φ_n, n <= P_phi, is unaffected); the factored path is fully flat-native.

function _m2l_source_alignment!(::MaterializedYRotationM2L, A, sources, B, phis, thetas, cache, scratch, lamb_helmholtz::Val{LH}, nbatch) where LH
    base = scratch.base
    P = cache.basis_info.orders.P_active
    P_phi = cache.basis_info.orders.P_phi
    li = base.weights_tmp_1; lm = base.weights_tmp_2; lo = base.weights_tmp_3
    @inbounds for j in 1:nbatch
        _pack_flat_column!(li, sources, j, P_phi, P, lamb_helmholtz)
        z_rotation_diagonals!(base.z_cos, base.z_sin, phis[j], P)
        apply_z_rotation!(lm, li, base.z_cos, base.z_sin, P, lamb_helmholtz, Val(:overwrite))
        rotate_multipole_y_op!(lo, lm, base.Ts, cache.S_pos, cache.S_neg, cache.zeta_mag, thetas[j], P, lamb_helmholtz, base.y_trig)
        _unpack_flat_column!(A, lo, j, P_phi, P, lamb_helmholtz)
    end
    return A
end

function _m2l_source_alignment!(::FactoredRotationM2L, A, sources, B, phis, thetas, cache, scratch, lamb_helmholtz::Val{LH}, nbatch) where LH
    # batched Z_phi -> factored Y(θ); B is the Z_phi tmp, gbuf reuses the embedded ν-space buffer.
    multipole_factored_source_alignment_batch_flat!(A, sources, B, scratch.base.y_mode_buf, phis, thetas, cache.y_mult_U, cache.y_mult_V, lamb_helmholtz)
    return A
end

#--- return alignment (stages 5-6: back Y(θ) then inverse Z_phi, accumulate) ---#

function _m2l_return_alignment!(::MaterializedYRotationM2L, targets, B, A, phis, thetas, cache, scratch, lamb_helmholtz::Val{LH}, nbatch) where LH
    base = scratch.base
    P = cache.basis_info.orders.P_active
    P_phi = cache.basis_info.orders.P_phi
    li = base.weights_tmp_1; lm = base.weights_tmp_2; lo = base.weights_tmp_3
    @inbounds for j in 1:nbatch
        _pack_flat_column!(li, B, j, P_phi, P, lamb_helmholtz)
        back_rotate_local_y_op!(lm, li, base.Ts, cache.Hs_pi2, cache.S_pos, cache.S_neg, cache.eta_mag, thetas[j], P, lamb_helmholtz, base.y_trig)
        z_rotation_diagonals!(base.z_cos, base.z_sin, phis[j], P)
        lo .= zero(eltype(lo))
        apply_z_rotation!(lo, lm, base.z_cos, base.z_sin, P, lamb_helmholtz, Val(:accumulate))
        _unpack_flat_column_accumulate!(targets, lo, j, P_phi, P, lamb_helmholtz)
    end
    return targets
end

function _m2l_return_alignment!(::FactoredRotationM2L, targets, B, A, phis, thetas, cache, scratch, lamb_helmholtz::Val{LH}, nbatch) where LH
    # batched factored back-Y(θ) -> inverse Z_phi (accumulating); A is the tmp.
    local_factored_return_alignment_batch_flat!(targets, B, A, scratch.base.y_mode_buf, phis, thetas, cache.y_loc_U, cache.y_loc_V, lamb_helmholtz)
    return targets
end

#------- FULL M2M/L2L OPERATOR PIPELINES (Matrix Operator Refactor, task 016) -------#

function m2m_operator_batch!(op::AbstractM2MOperator, targets::FlatCoefficientBuffer, sources::FlatCoefficientBuffer, phis, thetas, rs, invariant_cache, scratch::M2MOperatorScratch, lamb_helmholtz::Val{LH}) where LH
    _require_compressed_complex_operator_buffers(targets, sources, invariant_cache, scratch)
    P = invariant_cache.basis_info.orders.P_active
    nbatch = length(phis)
    A = scratch.work_a
    B = scratch.work_b

    _m2m_source_alignment!(op, A, sources, B, phis, thetas, invariant_cache, scratch, lamb_helmholtz, nbatch)

    @inbounds for j in 1:nbatch
        m2m_z_blocks!(scratch.blocks, rs[j], P)
        apply_m2m_z_flat!(B, A, j, scratch.blocks, Val(:overwrite))
        if LH
            lamb_helmholtz_multipole_coeffs!(scratch.lh_A, scratch.lh_B, rs[j], P)
            apply_lamb_helmholtz_multipole_flat!(A, B, j, scratch.lh_A, scratch.lh_B, Val(:overwrite))
        end
    end

    mid = LH ? A : B
    tmp = LH ? B : A
    _m2m_return_alignment!(op, targets, mid, tmp, phis, thetas, invariant_cache, scratch, lamb_helmholtz, nbatch)

    return targets
end

function l2l_operator_batch!(op::AbstractL2LOperator, targets::FlatCoefficientBuffer, sources::FlatCoefficientBuffer, phis, thetas, rs, invariant_cache, scratch::L2LOperatorScratch, lamb_helmholtz::Val{LH}) where LH
    _require_compressed_complex_operator_buffers(targets, sources, invariant_cache, scratch)
    P = invariant_cache.basis_info.orders.P_active
    nbatch = length(phis)
    A = scratch.work_a
    B = scratch.work_b

    _l2l_source_alignment!(op, A, sources, B, phis, thetas, invariant_cache, scratch, lamb_helmholtz, nbatch)

    @inbounds for j in 1:nbatch
        l2l_z_blocks!(scratch.blocks, rs[j], P)
        apply_l2l_z_flat!(B, A, j, scratch.blocks, Val(:overwrite))
        if LH
            lamb_helmholtz_local_coeffs!(scratch.lh_A, scratch.lh_B, rs[j], P)
            apply_lamb_helmholtz_local_flat!(A, B, j, scratch.lh_A, scratch.lh_B, Val(:overwrite))
        end
    end

    mid = LH ? A : B
    tmp = LH ? B : A
    _l2l_return_alignment!(op, targets, mid, tmp, phis, thetas, invariant_cache, scratch, lamb_helmholtz, nbatch)

    return targets
end

function _require_compressed_complex_operator_buffers(targets, sources, invariant_cache, scratch)
    targets.basis_info.basis isa CompressedComplexBasis &&
        sources.basis_info.basis isa CompressedComplexBasis &&
        invariant_cache.basis_info.basis isa CompressedComplexBasis &&
        scratch.base.basis_info.basis isa CompressedComplexBasis && return nothing
    throw(ArgumentError("native real-basis M2M/M2L/L2L operator execution is deferred; transform through CompressedComplexBasis before calling batched operators"))
end

# M2M/L2L alignment uses the same per-column repack-to-legacy materialized stage as
# M2L (production-parity z/y rotation kernels reused unchanged). The Factored* M2M/
# L2L tags remain parity-first fallbacks delegating to the materialized stage (task
# 016): the dedicated factored M2M/L2L path is benchmarked/approved later.

function _m2m_source_alignment!(::MaterializedYRotationM2M, A, sources, B, phis, thetas, cache, scratch, lamb_helmholtz::Val{LH}, nbatch) where LH
    base = scratch.base
    P = cache.basis_info.orders.P_active
    P_phi = cache.basis_info.orders.P_phi
    li = base.weights_tmp_1; lm = base.weights_tmp_2; lo = base.weights_tmp_3
    @inbounds for j in 1:nbatch
        _pack_flat_column!(li, sources, j, P_phi, P, lamb_helmholtz)
        z_rotation_diagonals!(base.z_cos, base.z_sin, phis[j], P)
        apply_z_rotation!(lm, li, base.z_cos, base.z_sin, P, lamb_helmholtz, Val(:overwrite))
        rotate_multipole_y_op!(lo, lm, base.Ts, cache.S_pos, cache.S_neg, cache.zeta_mag, thetas[j], P, lamb_helmholtz, base.y_trig)
        _unpack_flat_column!(A, lo, j, P_phi, P, lamb_helmholtz)
    end
    return A
end

function _m2m_source_alignment!(::FactoredRotationM2M, A, sources, B, phis, thetas, cache, scratch, lamb_helmholtz::Val{LH}, nbatch) where LH
    return _m2m_source_alignment!(MaterializedYRotationM2M(), A, sources, B, phis, thetas, cache, scratch, lamb_helmholtz, nbatch)
end

function _m2m_return_alignment!(::MaterializedYRotationM2M, targets, mid, tmp, phis, thetas, cache, scratch, lamb_helmholtz::Val{LH}, nbatch) where LH
    base = scratch.base
    P = cache.basis_info.orders.P_active
    P_phi = cache.basis_info.orders.P_phi
    li = base.weights_tmp_1; lm = base.weights_tmp_2; lo = base.weights_tmp_3
    @inbounds for j in 1:nbatch
        _pack_flat_column!(li, mid, j, P_phi, P, lamb_helmholtz)
        back_rotate_multipole_y_op!(lm, li, base.Ts, cache.S_pos, cache.S_neg, cache.zeta_mag, thetas[j], P, lamb_helmholtz, base.y_trig)
        z_rotation_diagonals!(base.z_cos, base.z_sin, phis[j], P)
        lo .= zero(eltype(lo))
        apply_z_rotation!(lo, lm, base.z_cos, base.z_sin, P, lamb_helmholtz, Val(:accumulate))
        _unpack_flat_column_accumulate!(targets, lo, j, P_phi, P, lamb_helmholtz)
    end
    return targets
end

function _m2m_return_alignment!(::FactoredRotationM2M, targets, mid, tmp, phis, thetas, cache, scratch, lamb_helmholtz::Val{LH}, nbatch) where LH
    return _m2m_return_alignment!(MaterializedYRotationM2M(), targets, mid, tmp, phis, thetas, cache, scratch, lamb_helmholtz, nbatch)
end

function _l2l_source_alignment!(::MaterializedYRotationL2L, A, sources, B, phis, thetas, cache, scratch, lamb_helmholtz::Val{LH}, nbatch) where LH
    base = scratch.base
    P = cache.basis_info.orders.P_active
    P_phi = cache.basis_info.orders.P_phi
    li = base.weights_tmp_1; lm = base.weights_tmp_2; lo = base.weights_tmp_3
    @inbounds for j in 1:nbatch
        _pack_flat_column!(li, sources, j, P_phi, P, lamb_helmholtz)
        z_rotation_diagonals!(base.z_cos, base.z_sin, phis[j], P)
        apply_z_rotation!(lm, li, base.z_cos, base.z_sin, P, lamb_helmholtz, Val(:overwrite))
        rotate_local_y_op!(lo, lm, base.Ts, cache.Hs_pi2, cache.S_pos, cache.S_neg, cache.eta_mag, thetas[j], P, lamb_helmholtz, base.y_trig)
        _unpack_flat_column!(A, lo, j, P_phi, P, lamb_helmholtz)
    end
    return A
end

function _l2l_source_alignment!(::FactoredRotationL2L, A, sources, B, phis, thetas, cache, scratch, lamb_helmholtz::Val{LH}, nbatch) where LH
    return _l2l_source_alignment!(MaterializedYRotationL2L(), A, sources, B, phis, thetas, cache, scratch, lamb_helmholtz, nbatch)
end

function _l2l_return_alignment!(::MaterializedYRotationL2L, targets, mid, tmp, phis, thetas, cache, scratch, lamb_helmholtz::Val{LH}, nbatch) where LH
    base = scratch.base
    P = cache.basis_info.orders.P_active
    P_phi = cache.basis_info.orders.P_phi
    li = base.weights_tmp_1; lm = base.weights_tmp_2; lo = base.weights_tmp_3
    @inbounds for j in 1:nbatch
        _pack_flat_column!(li, mid, j, P_phi, P, lamb_helmholtz)
        back_rotate_local_y_op!(lm, li, base.Ts, cache.Hs_pi2, cache.S_pos, cache.S_neg, cache.eta_mag, thetas[j], P, lamb_helmholtz, base.y_trig)
        z_rotation_diagonals!(base.z_cos, base.z_sin, phis[j], P)
        lo .= zero(eltype(lo))
        apply_z_rotation!(lo, lm, base.z_cos, base.z_sin, P, lamb_helmholtz, Val(:accumulate))
        _unpack_flat_column_accumulate!(targets, lo, j, P_phi, P, lamb_helmholtz)
    end
    return targets
end

function _l2l_return_alignment!(::FactoredRotationL2L, targets, mid, tmp, phis, thetas, cache, scratch, lamb_helmholtz::Val{LH}, nbatch) where LH
    return _l2l_return_alignment!(MaterializedYRotationL2L(), targets, mid, tmp, phis, thetas, cache, scratch, lamb_helmholtz, nbatch)
end

#------- RESIDENT BATCHED-M2M GEMM STRATEGIES (Matrix Operator Refactor, task 022) -------#
#
# These operate on the GEMM-native `DegreeMajorRealBuffer` layout so per-degree blocks
# are ready `mul!` operands with no repack. A node's coefficients are the stacked real
# vector [phi-degree-major; chi-degree-major] (chi empty for Val(false)); operators are
# real D×D (D = degree_major_dof(P_phi) + [Val(true)] degree_major_dof(P_active)).

@inline _dense_m2m_dof(basis_info, ::Val{LH}) where LH =
    degree_major_dof(basis_info.orders.P_phi) + (LH ? degree_major_dof(basis_info.orders.P_active) : 0)

# Row range of the phi / chi sub-block inside the stacked [phi;chi] vector.
@inline _stacked_phi_rows(basis_info) = 1:degree_major_dof(basis_info.orders.P_phi)
@inline function _stacked_chi_rows(basis_info)
    off = degree_major_dof(basis_info.orders.P_phi)
    return (off + 1):(off + degree_major_dof(basis_info.orders.P_active))
end

"""
    build_dense_m2m_operator(TF, basis_info, r, theta, phi, Val(LH)) -> Matrix{TF}

Materialize the complete dense M2M operator (stacked [phi;chi] degree-major, `D×D`
real) for one translation geometry, by pushing the `D` identity columns through the
validated task-016 `m2m_operator_batch!(MaterializedYRotationM2M(), …)`. Correctness by
construction (the operator is linear in the source coefficients). Host build only.
"""
function build_dense_m2m_operator(::Type{TF}, basis_info::OperatorBasisInfo{B,LH},
        r, theta, phi, lamb_helmholtz::Val{LH}) where {TF,B,LH}
    D = _dense_m2m_dof(basis_info, lamb_helmholtz)
    cache = OperatorInvariantCache(TF, basis_info)
    scratch = M2MOperatorScratch(TF, basis_info, D)
    src = FlatCoefficientBuffer(TF, basis_info, D)
    tgt = FlatCoefficientBuffer(TF, basis_info, D)
    src_dm = DegreeMajorRealBuffer(TF, basis_info, D)
    prow = _stacked_phi_rows(basis_info)
    @inbounds for (j, row) in enumerate(prow)
        src_dm.phi[row, j] = one(TF)
    end
    if LH
        crow = _stacked_chi_rows(basis_info)
        @inbounds for (i, row) in enumerate(crow)
            src_dm.chi[i, row] = one(TF)   # chi dof i lives in stacked column `row` (= Dp + i)
        end
    end
    to_flat_buffer!(src, src_dm)
    phis = fill(TF(phi), D); thetas = fill(TF(theta), D); rs = fill(TF(r), D)
    fill!(tgt.phi, zero(TF)); LH && fill!(tgt.chi, zero(TF))
    m2m_operator_batch!(MaterializedYRotationM2M(), tgt, src, phis, thetas, rs, cache, scratch, lamb_helmholtz)
    tgt_dm = DegreeMajorRealBuffer(TF, basis_info, D)
    to_gemm_buffer!(tgt_dm, tgt)
    K = zeros(TF, D, D)
    @inbounds K[_stacked_phi_rows(basis_info), :] .= tgt_dm.phi
    LH && (@inbounds K[_stacked_chi_rows(basis_info), :] .= tgt_dm.chi)
    return K
end

# Reusable host construction workspace for complete M2L matrices. Identity columns
# are processed in blocks so construction memory can be bounded independently of
# the permanently stored D×D matrices.
struct DenseM2LBuilderWorkspace{TF,B<:AbstractOperatorBasis,LH}
    invariant::OperatorInvariantCache{TF,B,LH}
    source::FlatCoefficientBuffer{TF,Matrix{TF},B,LH}
    target::FlatCoefficientBuffer{TF,Matrix{TF},B,LH}
    source_dm::DegreeMajorRealBuffer{TF,Matrix{TF},B,LH}
    target_dm::DegreeMajorRealBuffer{TF,Matrix{TF},B,LH}
    scratch::M2LOperatorScratch{TF,B,LH}
    stacked_input::Matrix{TF}
    stacked_output::Matrix{TF}
    phis::Vector{TF}
    thetas::Vector{TF}
    rs::Vector{TF}
end

function DenseM2LBuilderWorkspace(::Type{TF}, basis_info::OperatorBasisInfo{B,LH},
        invariant::OperatorInvariantCache{TF,B,LH}, width::Integer) where {TF,B,LH}
    w = Int(width)
    w > 0 || throw(ArgumentError("dense M2L builder width must be positive"))
    D = _dense_m2m_dof(basis_info, Val(LH))
    return DenseM2LBuilderWorkspace{TF,B,LH}(
        invariant,
        FlatCoefficientBuffer(TF, basis_info, w),
        FlatCoefficientBuffer(TF, basis_info, w),
        DegreeMajorRealBuffer(TF, basis_info, w),
        DegreeMajorRealBuffer(TF, basis_info, w),
        M2LOperatorScratch(TF, basis_info, w),
        zeros(TF, D, w), zeros(TF, D, w),
        Vector{TF}(undef, w), Vector{TF}(undef, w), Vector{TF}(undef, w),
    )
end

"""
    build_dense_m2l_operator!(K, r, theta, phi, invariant, workspace, Val(LH))

Fill `K` with the complete real, degree-major, stacked `[phi; chi]` M2L operator
defined by the validated `MaterializedYRotationM2L` pipeline. Construction is
allocation-free after `workspace` creation.
"""
function build_dense_m2l_operator!(K::AbstractMatrix{TF}, r, theta, phi,
        invariant::OperatorInvariantCache{TF,B,LH},
        workspace::DenseM2LBuilderWorkspace{TF,B,LH},
        lamb_helmholtz::Val{LH}) where {TF,B,LH}
    invariant === workspace.invariant || throw(ArgumentError(
        "dense M2L builder workspace belongs to a different invariant cache"))
    D = _dense_m2m_dof(invariant.basis_info, lamb_helmholtz)
    size(K) == (D, D) || throw(DimensionMismatch(
        "dense M2L operator must be $D×$D, got $(size(K))"))
    width = size(workspace.stacked_input, 2)
    fill!(workspace.phis, TF(phi))
    fill!(workspace.thetas, TF(theta))
    fill!(workspace.rs, TF(r))
    first_col = 1
    while first_col <= D
        ncols = min(width, D - first_col + 1)
        fill!(workspace.stacked_input, zero(TF))
        @inbounds for j in 1:ncols
            workspace.stacked_input[first_col + j - 1, j] = one(TF)
        end
        unstack_degree_major!(workspace.source_dm, workspace.stacked_input)
        to_flat_buffer!(workspace.source, workspace.source_dm)
        fill!(workspace.target.phi, zero(TF))
        LH && fill!(workspace.target.chi, zero(TF))
        m2l_operator_batch!(MaterializedYRotationM2L(), workspace.target,
            workspace.source, workspace.phis, workspace.thetas, workspace.rs,
            invariant, workspace.scratch, lamb_helmholtz)
        to_gemm_buffer!(workspace.target_dm, workspace.target)
        stack_degree_major!(workspace.stacked_output, workspace.target_dm)
        @inbounds for j in 1:ncols, i in 1:D
            K[i, first_col + j - 1] = workspace.stacked_output[i, j]
        end
        first_col += ncols
    end
    return K
end

"""
    build_dense_m2l_operator(TF, basis_info, r, theta, phi, Val(LH))

Allocating test/debug convenience builder for one complete host M2L matrix.
"""
function build_dense_m2l_operator(::Type{TF}, basis_info::OperatorBasisInfo{B,LH},
        r, theta, phi, lamb_helmholtz::Val{LH}) where {TF,B,LH}
    D = _dense_m2m_dof(basis_info, lamb_helmholtz)
    invariant = OperatorInvariantCache(TF, basis_info)
    workspace = DenseM2LBuilderWorkspace(TF, basis_info, invariant, D)
    K = Matrix{TF}(undef, D, D)
    return build_dense_m2l_operator!(K, r, theta, phi, invariant, workspace,
        lamb_helmholtz)
end

# Stack a DegreeMajorRealBuffer's channels into a single [phi;chi] × batch matrix, and
# split back. `dest`/`src` are plain matrices of the buffer's array type.
function stack_degree_major!(dest, buf::DegreeMajorRealBuffer{TF,A,B,LH}) where {TF,A,B,LH}
    prow = _stacked_phi_rows(buf.basis_info)
    @inbounds dest[prow, :] .= buf.phi
    LH && (@inbounds dest[_stacked_chi_rows(buf.basis_info), :] .= buf.chi)
    return dest
end

function unstack_degree_major!(buf::DegreeMajorRealBuffer{TF,A,B,LH}, src) where {TF,A,B,LH}
    prow = _stacked_phi_rows(buf.basis_info)
    @inbounds buf.phi .= @view src[prow, :]
    LH && (@inbounds buf.chi .= @view src[_stacked_chi_rows(buf.basis_info), :])
    return buf
end

# --- Strategy B (SharedRotationM2M) primitives, degree-major real buffer ---
#
# The expensive y-rotation is the batch-shared, θ-independent per-degree modes U_n/V_n
# (Y_n(θ)=U_n diag(e^{iνθ}) V_n); everything per-vector (Z_φ diagonal, e^{iνθ}, z blocks)
# is cheap. The V/U contractions are real GEMMs over each degree's contiguous rows.

# Per-degree (2n+1)×(2n+1) real/imag mode blocks from a flat complex mode vector
# (`modes[off+(c-1)*d+r]` is entry [r,c], column-major per degree).
function _ymode_real_blocks(modes, P::Integer, ::Type{TF}) where TF
    d0 = 1
    seg0 = modes[(ymode_offset(0) + 1):(ymode_offset(0) + d0 * d0)]
    first_block = (reshape(TF.(real.(seg0)), d0, d0),
        reshape(TF.(imag.(seg0)), d0, d0))
    blocks = Vector{typeof(first_block)}(undef, P + 1)
    blocks[1] = first_block
    @inbounds for n in 1:P
        d = 2n + 1
        rows = (ymode_offset(n) + 1):(ymode_offset(n) + d * d)
        seg = modes[rows]
        blocks[n + 1] = (reshape(TF.(real.(seg)), d, d), reshape(TF.(imag.(seg)), d, d))
    end
    return blocks
end

struct DegreeMajorMaps{RM,RI}
    row_m::RM
    row_ssign::RM
    row_pair::RI
    row_down::RI
    row_up::RI
    z_re_rows::Vector{RI}
    z_im_rows::Vector{RI}
    nus::Vector{RM}
end

function _array_like_vector(exemplar, ::Type{T}, values::AbstractVector) where T
    out = similar(exemplar, T, length(values))
    copyto!(out, values)
    return out
end

function _array_like_matrix(exemplar, ::Type{T}, values::AbstractMatrix) where T
    out = similar(exemplar, T, size(values)...)
    copyto!(out, values)
    return out
end

function DegreeMajorMaps(::Type{TF}, P::Integer, exemplar) where TF
    ndof = degree_major_dof(P)
    row_m = Vector{Int}(undef, ndof)
    row_ssign = Vector{TF}(undef, ndof)
    row_pair = Vector{Int}(undef, ndof)
    row_down = Vector{Int}(undef, ndof)
    row_up = Vector{Int}(undef, ndof)
    @inbounds for n in 0:P
        base = degree_row_offset(n)
        for k in 1:(2n + 1)
            row = base + k
            ri, m = _ymode_dof_to_storage(k)
            row_m[row] = m
            row_ssign[row] = ri == 1 ? -one(TF) : one(TF)
            row_pair[row] = m == 0 ? row : base + (ri == 1 ? 2m + 1 : 2m)
            row_down[row] = n > m ? degree_row_offset(n - 1) + k : row
            row_up[row] = n < P ? degree_row_offset(n + 1) + k : row
        end
    end

    row_m_dev = _array_like_vector(exemplar, TF, TF.(row_m))
    row_ssign_dev = _array_like_vector(exemplar, TF, row_ssign)
    row_pair_dev = _array_like_vector(exemplar, Int, row_pair)
    row_down_dev = _array_like_vector(exemplar, Int, row_down)
    row_up_dev = _array_like_vector(exemplar, Int, row_up)
    z_re_rows = Vector{typeof(row_pair_dev)}(undef, P + 1)
    z_im_rows = Vector{typeof(row_pair_dev)}(undef, P + 1)
    @inbounds for m in 0:P
        re = [degree_row_offset(n) + (m == 0 ? 1 : 2m) for n in m:P]
        im = m == 0 ? Int[] : [degree_row_offset(n) + 2m + 1 for n in m:P]
        z_re_rows[m + 1] = _array_like_vector(exemplar, Int, re)
        z_im_rows[m + 1] = _array_like_vector(exemplar, Int, im)
    end
    nus = Vector{typeof(row_m_dev)}(undef, P + 1)
    @inbounds for n in 0:P
        nus[n + 1] = _array_like_vector(exemplar, TF, TF.(-n:n))
    end
    return DegreeMajorMaps(
        row_m_dev, row_ssign_dev, row_pair_dev, row_down_dev, row_up_dev,
        z_re_rows,
        z_im_rows,
        nus,
    )
end

# Z_φ on a degree-major channel matrix: per column, per degree, rotate each (re_m,im_m)
# pair by mφ (m=0 unchanged). `inverse=false` matches apply_z_rotation overwrite
# (re',im')=(cm*re - sm*im, sm*re + cm*im); `inverse=true` is the transpose.
function _zphi_degree_major!(slab, phis, P::Integer, inverse::Bool)
    TF = eltype(slab)
    @inbounds for j in axes(slab, 2)
        for n in 0:P
            base = degree_row_offset(n)
            for m in 1:n
                cm, sm = cos(TF(m) * phis[j]), sin(TF(m) * phis[j])
                rr = base + 2m; ir = base + 2m + 1
                a = slab[rr, j]; b = slab[ir, j]
                if inverse
                    slab[rr, j] = cm * a + sm * b
                    slab[ir, j] = -sm * a + cm * b
                else
                    slab[rr, j] = cm * a - sm * b
                    slab[ir, j] = sm * a + cm * b
                end
            end
        end
    end
    return slab
end

function _zphi_degree_major_gen!(slab, phis, maps::DegreeMajorMaps, inverse::Bool)
    A = maps.row_m .* transpose(phis)
    C = cos.(A)
    S = sin.(A)
    sw = slab[maps.row_pair, :]
    sgn = inverse ? -one(eltype(slab)) : one(eltype(slab))
    slab .= C .* slab .+ (sgn .* maps.row_ssign .* S) .* sw
    return slab
end

# Forward factored Y over a degree-major channel matrix (out and in are (P+1)^2 × N).
# Uses shared per-degree blocks; per-column e^{iνθ} middle diagonal.
function _factored_y_degree_major!(out_slab, in_slab, Ublocks, Vblocks, thetas, P::Integer)
    TF = eltype(out_slab)
    @inbounds for n in 0:P
        d = 2n + 1; rows = degree_row_range(n)
        Vre, Vim = Vblocks[n + 1]; Ure, Uim = Ublocks[n + 1]
        X = @view in_slab[rows, :]
        Gre = Vre * X; Gim = Vim * X            # d × N (real GEMMs)
        for j in axes(X, 2)
            θ = thetas[j]
            for νidx in 1:d
                ν = νidx - n - 1
                s, c = sincos(ν * θ)
                gr = Gre[νidx, j]; gi = Gim[νidx, j]
                Gre[νidx, j] = c * gr - s * gi
                Gim[νidx, j] = s * gr + c * gi
            end
        end
        out_slab[rows, :] .= Ure * Gre .- Uim * Gim   # real GEMMs
    end
    return out_slab
end

function _factored_y_degree_major_gen!(out_slab, in_slab, Ublocks, Vblocks, thetas,
        maps::DegreeMajorMaps, P::Integer)
    @inbounds for n in 0:P
        rows = degree_row_range(n)
        Vre, Vim = Vblocks[n + 1]
        Ure, Uim = Ublocks[n + 1]
        X = @view in_slab[rows, :]
        Gre = Vre * X
        Gim = Vim * X
        A = maps.nus[n + 1] .* transpose(thetas)
        Cd = cos.(A)
        Sd = sin.(A)
        Gre2 = Cd .* Gre .- Sd .* Gim
        Gim2 = Sd .* Gre .+ Cd .* Gim
        out_slab[rows, :] .= Ure * Gre2 .- Uim * Gim2
    end
    return out_slab
end

# Allocation-free sibling for the resident grouped path. Degree rows are contiguous,
# so each U/V application remains a GEMM; the Fourier phase is applied in place to
# two reusable degree-major scratch slabs.
function _factored_y_degree_major_scratch!(out_slab, in_slab, Ublocks, Vblocks,
        thetas, scratch_re, scratch_im, scratch_tmp, P::Integer)
    @inbounds for n in 0:P
        rows = degree_row_range(n)
        Vre, Vim = Vblocks[n + 1]
        Ure, Uim = Ublocks[n + 1]
        X = @view in_slab[rows, :]
        Gre = @view scratch_re[rows, :]
        Gim = @view scratch_im[rows, :]
        Tmp = @view scratch_tmp[rows, :]
        Y = @view out_slab[rows, :]
        mul!(Gre, Vre, X)
        mul!(Gim, Vim, X)
        for j in axes(X, 2), nu_idx in axes(X, 1)
            nu = nu_idx - n - 1
            s, c = sincos(nu * thetas[j])
            gr = Gre[nu_idx, j]
            gi = Gim[nu_idx, j]
            Gre[nu_idx, j] = c * gr - s * gi
            Gim[nu_idx, j] = s * gr + c * gi
        end
        mul!(Y, Ure, Gre)
        mul!(Tmp, Uim, Gim)
        Y .-= Tmp
    end
    return out_slab
end

@inline function _factored_y_degree_block_noalloc!(out_slab, in_slab,
        Ure, Uim, Vre, Vim, thetas, scratch_re, scratch_im,
        row0::Int, d::Int, ncols::Int)
    @inbounds for j in 1:ncols
        for k in 1:d
            gr = zero(eltype(out_slab))
            gi = zero(eltype(out_slab))
            for q in 1:d
                x = in_slab[row0 + q - 1, j]
                gr += Vre[k, q] * x
                gi += Vim[k, q] * x
            end
            nu = k - ((d - 1) >> 1) - 1
            s, c = sincos(nu * thetas[j])
            scratch_re[row0 + k - 1, j] = c * gr - s * gi
            scratch_im[row0 + k - 1, j] = s * gr + c * gi
        end
        for k in 1:d
            y = zero(eltype(out_slab))
            for q in 1:d
                y += Ure[k, q] * scratch_re[row0 + q - 1, j]
                y -= Uim[k, q] * scratch_im[row0 + q - 1, j]
            end
            out_slab[row0 + k - 1, j] = y
        end
    end
    return out_slab
end

function _factored_y_degree_major_noalloc!(out_slab, in_slab, Ublocks, Vblocks,
        thetas, scratch_re, scratch_im, P::Integer, ncols::Integer)
    @inbounds for n in 0:P
        Ure, Uim = Ublocks[n + 1]
        Vre, Vim = Vblocks[n + 1]
        _factored_y_degree_block_noalloc!(out_slab, in_slab, Ure, Uim, Vre, Vim,
            thetas, scratch_re, scratch_im, degree_row_offset(n) + 1, 2n + 1, Int(ncols))
    end
    return out_slab
end

# Crossover for routing per-degree factored-y applications to BLAS GEMMs: classes at
# least this wide use `mul!` for degree blocks of dimension at least
# FACTORED_Y_GEMM_MIN_DIM; everything else keeps the scalar no-alloc kernel, whose
# per-element cost beats BLAS dispatch on the narrow/sparse classes. Defaults are
# measured (023a, EPYC 7763 / OpenBLAS): scalar wins at class width <= 7, GEMM wins
# from width 20 at every P (including the 1x1/3x3 low-degree blocks, so no dim gate);
# BLAS thread count was immaterial at these block sizes.
const FACTORED_Y_GEMM_MIN_COLS = Ref(16)
const FACTORED_Y_GEMM_MIN_DIM = Ref(1)

# GEMM form of one degree block: Y = Ure*G' - Uim*G'' where G' + iG'' =
# diag(e^{iνθ_j}) (Vre + iVim) X on the physical subspace (real inputs). Views are
# column prefixes of capacity-sized slabs; `mul!` on these strided views does not
# allocate.
@inline function _factored_y_degree_block_gemm!(out_slab, in_slab,
        Ure, Uim, Vre, Vim, thetas, scratch_re, scratch_im, scratch_tmp,
        row0::Int, d::Int, ncols::Int)
    rows = row0:(row0 + d - 1)
    cols = 1:ncols
    X = @view in_slab[rows, cols]
    Gre = @view scratch_re[rows, cols]
    Gim = @view scratch_im[rows, cols]
    Tmp = @view scratch_tmp[rows, cols]
    Y = @view out_slab[rows, cols]
    mul!(Gre, Vre, X)
    mul!(Gim, Vim, X)
    half = (d - 1) >> 1
    @inbounds for j in 1:ncols
        theta = thetas[j]
        for k in 1:d
            nu = k - half - 1
            s, c = sincos(nu * theta)
            gr = Gre[k, j]
            gi = Gim[k, j]
            Gre[k, j] = c * gr - s * gi
            Gim[k, j] = s * gr + c * gi
        end
    end
    mul!(Y, Ure, Gre)
    mul!(Tmp, Uim, Gim)
    Y .-= Tmp
    return out_slab
end

# Width-dispatched per-degree factored y: GEMM blocks past the measured crossover,
# scalar blocks otherwise. `scratch_tmp` is only touched on the GEMM branch.
function _factored_y_degree_major_auto!(out_slab, in_slab, Ublocks, Vblocks,
        thetas, scratch_re, scratch_im, scratch_tmp, P::Integer, ncols::Integer)
    n_cols = Int(ncols)
    use_gemm = n_cols >= FACTORED_Y_GEMM_MIN_COLS[]
    min_dim = FACTORED_Y_GEMM_MIN_DIM[]
    @inbounds for n in 0:P
        d = 2n + 1
        Ure, Uim = Ublocks[n + 1]
        Vre, Vim = Vblocks[n + 1]
        row0 = degree_row_offset(n) + 1
        if use_gemm && d >= min_dim
            _factored_y_degree_block_gemm!(out_slab, in_slab, Ure, Uim, Vre, Vim,
                thetas, scratch_re, scratch_im, scratch_tmp, row0, d, n_cols)
        else
            _factored_y_degree_block_noalloc!(out_slab, in_slab, Ure, Uim, Vre, Vim,
                thetas, scratch_re, scratch_im, row0, d, n_cols)
        end
    end
    return out_slab
end

# Per-column M2M z-translation on a degree-major channel (reuses m2m_z_blocks). The
# block table is sized/indexed by `P_block` (= P_active, matching apply_m2m_z_flat!);
# the channel loops degrees to `P_loop` (P_phi for φ, P_active for χ). `blocks` is a
# preallocated table of length `m2m_z_block_length(P_block)`.
function _m2m_ztranslate_degree_major!(out_slab, in_slab, blocks, rs, P_loop::Integer, P_block::Integer)
    TF = eltype(out_slab)
    @inbounds for j in axes(out_slab, 2)
        m2m_z_blocks!(blocks, rs[j], P_block)
        for n in 0:P_loop
            base_n = degree_row_offset(n)
            for m in 0:n
                kre = m == 0 ? 1 : 2m
                kim = m == 0 ? 0 : 2m + 1
                accr = zero(TF); acci = zero(TF)
                for np in m:n
                    coeff = blocks[m2m_z_block_index(n, np, m, P_block)]
                    base_np = degree_row_offset(np)
                    accr += coeff * in_slab[base_np + kre, j]
                    kim > 0 && (acci += coeff * in_slab[base_np + kim, j])
                end
                out_slab[base_n + kre, j] = accr
                kim > 0 && (out_slab[base_n + kim, j] = acci)
            end
        end
    end
    return out_slab
end

function _m2m_z_block_matrix(::Type{TF}, blocks, m::Integer, P_loop::Integer, P_block::Integer) where TF
    w = P_loop - m + 1
    Bm = zeros(TF, w, w)
    @inbounds for (icol, np) in enumerate(m:P_loop)
        for (irow, n) in enumerate(m:P_loop)
            n >= np || continue
            Bm[irow, icol] = blocks[m2m_z_block_index(n, np, m, P_block)]
        end
    end
    return Bm
end

function _m2m_ztranslate_shared!(out_slab, in_slab, blocks, maps::DegreeMajorMaps,
        P_loop::Integer, P_block::Integer)
    TF = eltype(out_slab)
    fill!(out_slab, zero(TF))
    @inbounds for m in 0:P_loop
        Bm = _array_like_matrix(out_slab, TF, _m2m_z_block_matrix(TF, blocks, m, P_loop, P_block))
        re_rows = maps.z_re_rows[m + 1]
        out_slab[re_rows, :] .= Bm * in_slab[re_rows, :]
        if m > 0
            im_rows = maps.z_im_rows[m + 1]
            out_slab[im_rows, :] .= Bm * in_slab[im_rows, :]
        end
    end
    return out_slab
end

function _m2l_z_block_matrix(::Type{TF}, blocks, m::Integer, P_loop::Integer, P_block::Integer) where TF
    w = P_loop - m + 1
    Bm = zeros(TF, w, w)
    @inbounds for (icol, np) in enumerate(m:P_loop)
        for (irow, n) in enumerate(m:P_loop)
            Bm[irow, icol] = blocks[m2l_z_block_index(n, np, m, P_block)]
        end
    end
    return Bm
end

function _m2l_ztranslate_shared!(out_slab, in_slab, blocks, maps::DegreeMajorMaps,
        P_loop::Integer, P_block::Integer)
    TF = eltype(out_slab)
    fill!(out_slab, zero(TF))
    @inbounds for m in 0:P_loop
        Bm = _array_like_matrix(out_slab, TF, _m2l_z_block_matrix(TF, blocks, m, P_loop, P_block))
        re_rows = maps.z_re_rows[m + 1]
        out_slab[re_rows, :] .= Bm * in_slab[re_rows, :]
        if m > 0
            im_rows = maps.z_im_rows[m + 1]
            out_slab[im_rows, :] .= Bm * in_slab[im_rows, :]
        end
    end
    return out_slab
end

function _l2l_z_block_matrix(::Type{TF}, blocks, m::Integer, P_loop::Integer, P_block::Integer) where TF
    w = P_loop - m + 1
    Bm = zeros(TF, w, w)
    @inbounds for (icol, np) in enumerate(m:P_loop)
        for (irow, n) in enumerate(m:P_loop)
            n <= np || continue
            Bm[irow, icol] = blocks[l2l_z_block_index(n, np, m, P_block)]
        end
    end
    return Bm
end

function _l2l_ztranslate_shared!(out_slab, in_slab, blocks, maps::DegreeMajorMaps,
        P_loop::Integer, P_block::Integer)
    TF = eltype(out_slab)
    fill!(out_slab, zero(TF))
    @inbounds for m in 0:P_loop
        Bm = _array_like_matrix(out_slab, TF, _l2l_z_block_matrix(TF, blocks, m, P_loop, P_block))
        re_rows = maps.z_re_rows[m + 1]
        out_slab[re_rows, :] .= Bm * in_slab[re_rows, :]
        if m > 0
            im_rows = maps.z_im_rows[m + 1]
            out_slab[im_rows, :] .= Bm * in_slab[im_rows, :]
        end
    end
    return out_slab
end

function _z_block_matrices_like(exemplar, ::Type{TF}, blocks, kind::Symbol,
        P_loop::Integer, P_block::Integer) where TF
    host0 = kind === :m2m ? _m2m_z_block_matrix(TF, blocks, 0, P_loop, P_block) :
        kind === :m2l ? _m2l_z_block_matrix(TF, blocks, 0, P_loop, P_block) :
        kind === :l2l ? _l2l_z_block_matrix(TF, blocks, 0, P_loop, P_block) :
        throw(ArgumentError("unknown z-block kind $kind"))
    first_mat = _array_like_matrix(exemplar, TF, host0)
    mats = Vector{typeof(first_mat)}(undef, P_loop + 1)
    mats[1] = first_mat
    @inbounds for m in 1:P_loop
        host = kind === :m2m ? _m2m_z_block_matrix(TF, blocks, m, P_loop, P_block) :
            kind === :m2l ? _m2l_z_block_matrix(TF, blocks, m, P_loop, P_block) :
            kind === :l2l ? _l2l_z_block_matrix(TF, blocks, m, P_loop, P_block) :
            throw(ArgumentError("unknown z-block kind $kind"))
        mats[m + 1] = _array_like_matrix(exemplar, TF, host)
    end
    return mats
end

# Dense whole-slab embedding of the per-m z-translation blocks: each per-m block acts
# identically on the re and im row sets, so embed it at rows/cols (n, m, ri) for both
# slots; one dof×dof GEMM then replaces the per-m block loop plus its row gathers.
function _z_dense_matrix_like(exemplar, ::Type{TF}, blocks, kind::Symbol,
        P_loop::Integer, P_block::Integer) where TF
    ndof = degree_major_dof(P_loop)
    Z = zeros(TF, ndof, ndof)
    @inbounds for m in 0:P_loop
        host = kind === :m2m ? _m2m_z_block_matrix(TF, blocks, m, P_loop, P_block) :
            kind === :m2l ? _m2l_z_block_matrix(TF, blocks, m, P_loop, P_block) :
            kind === :l2l ? _l2l_z_block_matrix(TF, blocks, m, P_loop, P_block) :
            throw(ArgumentError("unknown z-block kind $kind"))
        for k in (m == 0 ? (1:1) : (2m:(2m + 1)))
            for (icol, np) in enumerate(m:P_loop), (irow, n) in enumerate(m:P_loop)
                Z[degree_row_offset(n) + k, degree_row_offset(np) + k] = host[irow, icol]
            end
        end
    end
    return _array_like_matrix(exemplar, TF, Z)
end

function _ztranslate_shared_precomputed!(out_slab, in_slab, block_mats,
        maps::DegreeMajorMaps, P_loop::Integer)
    TF = eltype(out_slab)
    fill!(out_slab, zero(TF))
    @inbounds for m in 0:P_loop
        re_rows = maps.z_re_rows[m + 1]
        out_slab[re_rows, :] .= block_mats[m + 1] * in_slab[re_rows, :]
        if m > 0
            im_rows = maps.z_im_rows[m + 1]
            out_slab[im_rows, :] .= block_mats[m + 1] * in_slab[im_rows, :]
        end
    end
    return out_slab
end

function _ztranslate_shared_precomputed_noalloc!(out_slab, in_slab, block_mats,
        maps::DegreeMajorMaps, P_loop::Integer, ncols::Integer)
    @inbounds for j in 1:ncols, i in axes(out_slab, 1)
        out_slab[i, j] = zero(eltype(out_slab))
    end
    @inbounds for m in 0:P_loop
        Bm = block_mats[m + 1]
        re_rows = maps.z_re_rows[m + 1]
        im_rows = maps.z_im_rows[m + 1]
        for j in 1:ncols, (irow, row) in enumerate(re_rows)
            sr = zero(eltype(out_slab))
            si = zero(eltype(out_slab))
            for (icol, col) in enumerate(re_rows)
                b = Bm[irow, icol]
                sr += b * in_slab[col, j]
                m > 0 && (si += b * in_slab[im_rows[icol], j])
            end
            out_slab[row, j] = sr
            m > 0 && (out_slab[im_rows[irow], j] = si)
        end
    end
    return out_slab
end

function _lh_local_shared_rows_noalloc!(out_phi, out_chi, in_phi, in_chi, arow,
        brow, maps_phi::DegreeMajorMaps, maps_chi::DegreeMajorMaps, ncols::Integer)
    @inbounds for j in 1:ncols, i in axes(out_phi, 1)
        out_phi[i, j] = in_phi[i, j] + arow[i] * in_chi[maps_phi.row_pair[i], j]
    end
    @inbounds for j in 1:ncols, i in axes(out_chi, 1)
        out_chi[i, j] = in_chi[i, j] + brow[i] * in_chi[maps_chi.row_up[i], j]
    end
    return out_chi
end

# Degree-major LH multipole coupling (mirrors apply_m2m_z_flat!'s LH sibling
# apply_lamb_helmholtz_multipole_flat!). Touches only stored dofs: φ gets a χ mix for
# m>=1 (the m=0 φ contribution is a*χ_im(m0)=0), χ gets the (n-1,m) recurrence. The
# transient φ_im(m0) the flat kernel writes is dropped here — it never reaches the
# output (return factored-Y ignores im(m0); Z_φ^{-1} is identity at m=0).
function _lh_multipole_degree_major!(out_phi, out_chi, in_phi, in_chi, lh_A, lh_B, rs,
        P_phi::Integer, P_active::Integer)
    TF = eltype(out_phi)
    @inbounds for j in axes(out_phi, 2)
        lamb_helmholtz_multipole_coeffs!(lh_A, lh_B, rs[j], P_active)
        Acoef = lh_A; Bcoef = lh_B
        for n in 0:P_phi
            base = degree_row_offset(n)
            for m in 1:n
                a = Acoef[harmonic_index(n, m)]
                rr = base + 2m; ir = base + 2m + 1
                chi_re = in_chi[rr, j]; chi_im = in_chi[ir, j]
                out_phi[rr, j] = in_phi[rr, j] + a * chi_im
                out_phi[ir, j] = in_phi[ir, j] - a * chi_re
            end
            out_phi[base + 1, j] = in_phi[base + 1, j]   # φ_re(m0): += a*χ_im(m0)=0
        end
        for n in 0:P_active
            base = degree_row_offset(n)
            for m in 0:n
                kre = m == 0 ? 1 : 2m
                kim = m == 0 ? 0 : 2m + 1
                if n > m
                    b = Bcoef[harmonic_index(n, m)]
                    blo = degree_row_offset(n - 1)
                    out_chi[base + kre, j] = in_chi[base + kre, j] + b * in_chi[blo + kre, j]
                    kim > 0 && (out_chi[base + kim, j] = in_chi[base + kim, j] + b * in_chi[blo + kim, j])
                else
                    out_chi[base + kre, j] = in_chi[base + kre, j]
                    kim > 0 && (out_chi[base + kim, j] = in_chi[base + kim, j])
                end
            end
        end
    end
    return out_chi
end

function _lh_row_coefficients(::Type{TF}, P_phi::Integer, P_active::Integer, lh_A, lh_B) where TF
    ndof_phi = degree_major_dof(P_phi)
    ndof_chi = degree_major_dof(P_active)
    arow = zeros(TF, ndof_phi)
    brow = zeros(TF, ndof_chi)
    @inbounds for n in 0:P_phi
        base = degree_row_offset(n)
        for m in 1:n
            a = lh_A[harmonic_index(n, m)]
            arow[base + 2m] = a
            arow[base + 2m + 1] = -a
        end
    end
    @inbounds for n in 0:P_active
        base = degree_row_offset(n)
        for m in 0:n
            n > m || continue
            b = lh_B[harmonic_index(n, m)]
            brow[base + (m == 0 ? 1 : 2m)] = b
            m > 0 && (brow[base + 2m + 1] = b)
        end
    end
    return arow, brow
end

function _lh_multipole_shared_gen!(out_phi, out_chi, in_phi, in_chi, lh_A, lh_B,
        maps_phi::DegreeMajorMaps, maps_chi::DegreeMajorMaps, P_phi::Integer, P_active::Integer)
    TF = eltype(out_phi)
    arow, brow = _lh_row_coefficients(TF, P_phi, P_active, lh_A, lh_B)
    arow_dev = _array_like_vector(out_phi, TF, arow)
    brow_dev = _array_like_vector(out_chi, TF, brow)
    out_phi .= in_phi .+ arow_dev .* in_chi[maps_phi.row_pair, :]
    out_chi .= in_chi .+ brow_dev .* in_chi[maps_chi.row_down, :]
    return out_chi
end

function _lh_multipole_shared_rows!(out_phi, out_chi, in_phi, in_chi, arow, brow,
        maps_phi::DegreeMajorMaps, maps_chi::DegreeMajorMaps)
    out_phi .= in_phi .+ arow .* in_chi[maps_phi.row_pair, :]
    out_chi .= in_chi .+ brow .* in_chi[maps_chi.row_down, :]
    return out_chi
end

function _lh_local_row_coefficients(::Type{TF}, P_phi::Integer, P_active::Integer, lh_A, lh_B) where TF
    ndof_phi = degree_major_dof(P_phi)
    ndof_chi = degree_major_dof(P_active)
    arow = zeros(TF, ndof_phi)
    brow = zeros(TF, ndof_chi)
    @inbounds for n in 0:P_phi
        base = degree_row_offset(n)
        for m in 1:n
            a = lh_A[harmonic_index(n, m)]
            arow[base + 2m] = -a
            arow[base + 2m + 1] = a
        end
    end
    @inbounds for n in 0:P_active
        base = degree_row_offset(n)
        for m in 0:n
            n < P_active || continue
            b = -lh_B[harmonic_index(n, m)]
            brow[base + (m == 0 ? 1 : 2m)] = b
            m > 0 && (brow[base + 2m + 1] = b)
        end
    end
    return arow, brow
end

function _lh_local_shared_gen!(out_phi, out_chi, in_phi, in_chi, lh_A, lh_B,
        maps_phi::DegreeMajorMaps, maps_chi::DegreeMajorMaps, P_phi::Integer, P_active::Integer)
    TF = eltype(out_phi)
    arow, brow = _lh_local_row_coefficients(TF, P_phi, P_active, lh_A, lh_B)
    arow_dev = _array_like_vector(out_phi, TF, arow)
    brow_dev = _array_like_vector(out_chi, TF, brow)
    out_phi .= in_phi .+ arow_dev .* in_chi[maps_phi.row_pair, :]
    out_chi .= in_chi .+ brow_dev .* in_chi[maps_chi.row_up, :]
    return out_chi
end

function _lh_local_shared_rows!(out_phi, out_chi, in_phi, in_chi, arow, brow,
        maps_phi::DegreeMajorMaps, maps_chi::DegreeMajorMaps)
    out_phi .= in_phi .+ arow .* in_chi[maps_phi.row_pair, :]
    out_chi .= in_chi .+ brow .* in_chi[maps_chi.row_up, :]
    return out_chi
end

"""
    resident_m2m_batch!(strategy, targets, sources, phis, thetas, rs, cache, Val(LH))

Batched M2M on the GEMM-native [`DegreeMajorRealBuffer`](@ref) layout, **accumulating**
into `targets` (zero it first). `strategy` is [`SharedRotationM2M`](@ref) (batch-shared
`U_n`/`V_n` GEMMs, per-vector z-axis pieces) or [`DenseTranslationM2M`](@ref) (full
per-vector operator). Array-generic through `mul!` (`Array`/`CuArray`).
"""
function resident_m2m_batch!(::SharedRotationM2M, targets::DegreeMajorRealBuffer{TF,A,B,LH},
        sources::DegreeMajorRealBuffer, phis, thetas, rs, cache, ::Val{LH}) where {TF,A,B,LH}
    P_phi = cache.basis_info.orders.P_phi
    P_active = cache.basis_info.orders.P_active
    r_level = isempty(rs) ? zero(TF) : TF(first(rs))
    all(r -> r == r_level, rs) ||
        throw(ArgumentError("SharedRotationM2M resident batch requires a uniform translation radius"))
    phis_work = _array_like_vector(sources.phi, TF, phis)
    thetas_work = _array_like_vector(sources.phi, TF, thetas)
    maps_phi = DegreeMajorMaps(TF, P_phi, sources.phi)
    maps_chi = LH ? DegreeMajorMaps(TF, P_active, sources.chi) : maps_phi
    Ub = _ymode_real_blocks(cache.y_mult_U, P_active, TF)
    Vb = _ymode_real_blocks(cache.y_mult_V, P_active, TF)
    blocks = Vector{TF}(undef, m2m_z_block_length(P_active))
    m2m_z_blocks!(blocks, r_level, P_active)

    # φ channel: Z_φ -> Y -> z-translate
    aphi = copy(sources.phi)
    _zphi_degree_major_gen!(aphi, phis_work, maps_phi, false)
    yphi = similar(aphi); _factored_y_degree_major_gen!(yphi, aphi, Ub, Vb, thetas_work, maps_phi, P_phi)
    zphi = similar(aphi); _m2m_ztranslate_shared!(zphi, yphi, blocks, maps_phi, P_phi, P_active)

    if LH
        achi = copy(sources.chi)
        _zphi_degree_major_gen!(achi, phis_work, maps_chi, false)
        ychi = similar(achi); _factored_y_degree_major_gen!(ychi, achi, Ub, Vb, thetas_work, maps_chi, P_active)
        zchi = similar(achi); _m2m_ztranslate_shared!(zchi, ychi, blocks, maps_chi, P_active, P_active)
        lh_A = Vector{TF}(undef, _operator_ncomplex(P_active))
        lh_B = Vector{TF}(undef, _operator_ncomplex(P_active))
        lamb_helmholtz_multipole_coeffs!(lh_A, lh_B, r_level, P_active)
        cphi = similar(zphi); cchi = similar(zchi)
        _lh_multipole_shared_gen!(cphi, cchi, zphi, zchi, lh_A, lh_B, maps_phi, maps_chi, P_phi, P_active)
        zphi = cphi; zchi = cchi
        # return: Y -> Z_φ^{-1} (accumulate) on χ
        rchi = similar(zchi); _factored_y_degree_major_gen!(rchi, zchi, Ub, Vb, thetas_work, maps_chi, P_active)
        _zphi_degree_major_gen!(rchi, phis_work, maps_chi, true)
        @inbounds targets.chi .+= rchi
    end

    # return: Y -> Z_φ^{-1} (accumulate) on φ
    rphi = similar(zphi); _factored_y_degree_major_gen!(rphi, zphi, Ub, Vb, thetas_work, maps_phi, P_phi)
    _zphi_degree_major_gen!(rphi, phis_work, maps_phi, true)
    @inbounds targets.phi .+= rphi
    return targets
end

function resident_m2m_batch!(::DenseTranslationM2M, targets::DegreeMajorRealBuffer{TF,A,B,LH},
        sources::DegreeMajorRealBuffer, phis, thetas, rs, cache, ::Val{LH}) where {TF,A,B,LH}
    # Per-column operator build+apply (correctness-first; the launcher groups columns by
    # translation-vector class so one K is reused across the class's GEMM).
    binfo = cache.basis_info
    D = _dense_m2m_dof(binfo, Val(LH))
    N = flat_nbatch(sources)
    S = zeros(TF, D, N); stack_degree_major!(S, sources)
    Tstack = zeros(TF, D, N)
    @inbounds for j in 1:N
        K = build_dense_m2m_operator(TF, binfo, rs[j], thetas[j], phis[j], Val(LH))
        @views mul!(Tstack[:, j:j], K, S[:, j:j])
    end
    tmp = DegreeMajorRealBuffer(TF, binfo, N); unstack_degree_major!(tmp, Tstack)
    @inbounds targets.phi .+= tmp.phi
    LH && (@inbounds targets.chi .+= tmp.chi)
    return targets
end

"""
    resident_m2l_batch!(strategy, targets, sources, phis, thetas, rs, cache, Val(LH))

Degree-major resident M2L. The resident path ignores `CUDARadixLifecycleOptions.operator`;
the free-function strategy selects either the batch-shared factored-mode form or,
for `DenseTranslationM2L`, complete matrices constructed from the materialized-y
oracle. Production lifecycle constructors validate the corresponding operator
pairing explicitly.
"""
function resident_m2l_batch!(::SharedRotationM2L, targets::DegreeMajorRealBuffer{TF,A,B,LH},
        sources::DegreeMajorRealBuffer, phis, thetas, rs, cache, ::Val{LH}) where {TF,A,B,LH}
    P_phi = cache.basis_info.orders.P_phi
    P_active = cache.basis_info.orders.P_active
    r_batch = isempty(rs) ? zero(TF) : TF(first(rs))
    phi_batch = isempty(phis) ? zero(TF) : TF(first(phis))
    theta_batch = isempty(thetas) ? zero(TF) : TF(first(thetas))
    all(r -> r == r_batch, rs) ||
        throw(ArgumentError("SharedRotationM2L resident batch requires a uniform translation radius"))
    all(ϕ -> ϕ == phi_batch, phis) ||
        throw(ArgumentError("SharedRotationM2L resident batch requires a uniform phi"))
    all(θ -> θ == theta_batch, thetas) ||
        throw(ArgumentError("SharedRotationM2L resident batch requires a uniform theta"))

    phis_work = _array_like_vector(sources.phi, TF, phis)
    thetas_work = _array_like_vector(sources.phi, TF, thetas)
    maps_phi = DegreeMajorMaps(TF, P_phi, sources.phi)
    maps_chi = LH ? DegreeMajorMaps(TF, P_active, sources.chi) : maps_phi
    Um = _ymode_real_blocks(cache.y_mult_U, P_active, TF)
    Vm = _ymode_real_blocks(cache.y_mult_V, P_active, TF)
    Ul = _ymode_real_blocks(cache.y_loc_U, P_active, TF)
    Vl = _ymode_real_blocks(cache.y_loc_V, P_active, TF)
    blocks = Vector{TF}(undef, m2l_z_block_length(P_active))
    m2l_z_blocks!(blocks, r_batch, P_active)

    aphi = copy(sources.phi)
    _zphi_degree_major_gen!(aphi, phis_work, maps_phi, false)
    yphi = similar(aphi)
    _factored_y_degree_major_gen!(yphi, aphi, Um, Vm, thetas_work, maps_phi, P_phi)
    zphi = similar(aphi)
    _m2l_ztranslate_shared!(zphi, yphi, blocks, maps_phi, P_phi, P_active)

    if LH
        achi = copy(sources.chi)
        _zphi_degree_major_gen!(achi, phis_work, maps_chi, false)
        ychi = similar(achi)
        _factored_y_degree_major_gen!(ychi, achi, Um, Vm, thetas_work, maps_chi, P_active)
        zchi = similar(achi)
        _m2l_ztranslate_shared!(zchi, ychi, blocks, maps_chi, P_active, P_active)
        lh_A = Vector{TF}(undef, _operator_ncomplex(P_active))
        lh_B = Vector{TF}(undef, _operator_ncomplex(P_active))
        lamb_helmholtz_local_coeffs!(lh_A, lh_B, r_batch, P_active)
        cphi = similar(zphi); cchi = similar(zchi)
        _lh_local_shared_gen!(cphi, cchi, zphi, zchi, lh_A, lh_B, maps_phi, maps_chi, P_phi, P_active)
        zphi = cphi; zchi = cchi
        rchi = similar(zchi)
        _factored_y_degree_major_gen!(rchi, zchi, Ul, Vl, thetas_work, maps_chi, P_active)
        _zphi_degree_major_gen!(rchi, phis_work, maps_chi, true)
        @inbounds targets.chi .+= rchi
    end

    rphi = similar(zphi)
    _factored_y_degree_major_gen!(rphi, zphi, Ul, Vl, thetas_work, maps_phi, P_phi)
    _zphi_degree_major_gen!(rphi, phis_work, maps_phi, true)
    @inbounds targets.phi .+= rphi
    return targets
end

function resident_m2l_batch!(::DenseTranslationM2L,
        targets::DegreeMajorRealBuffer{TF,A,B,LH},
        sources::DegreeMajorRealBuffer, phis, thetas, rs, cache,
        lamb_helmholtz::Val{LH}) where {TF,A,B,LH}
    N = flat_nbatch(sources)
    flat_nbatch(targets) == N || throw(DimensionMismatch(
        "dense resident M2L source and target widths must match"))
    length(phis) == N && length(thetas) == N && length(rs) == N ||
        throw(DimensionMismatch("dense resident M2L geometry must have one value per column"))
    D = _dense_m2m_dof(cache.basis_info, lamb_helmholtz)
    stacked_source = zeros(TF, D, N)
    stacked_target = zeros(TF, D, N)
    stack_degree_major!(stacked_source, sources)
    @inbounds for j in 1:N
        K = build_dense_m2l_operator(TF, cache.basis_info, rs[j], thetas[j],
            phis[j], lamb_helmholtz)
        for i in 1:D
            acc = zero(TF)
            for k in 1:D
                acc += K[i, k] * stacked_source[k, j]
            end
            stacked_target[i, j] = acc
        end
    end
    tmp = DegreeMajorRealBuffer(TF, cache.basis_info, N)
    unstack_degree_major!(tmp, stacked_target)
    targets.phi .+= tmp.phi
    LH && (targets.chi .+= tmp.chi)
    return targets
end

"""
    resident_l2l_batch!(targets, sources, phis, thetas, rs, cache, Val(LH))

Degree-major resident L2L over local-mode factored rotations. Callers group by
uniform `r`; per-edge `theta`/`phi` may vary within the group.
"""
function resident_l2l_batch!(targets::DegreeMajorRealBuffer{TF,A,B,LH},
        sources::DegreeMajorRealBuffer, phis, thetas, rs, cache, ::Val{LH}) where {TF,A,B,LH}
    P_phi = cache.basis_info.orders.P_phi
    P_active = cache.basis_info.orders.P_active
    r_level = isempty(rs) ? zero(TF) : TF(first(rs))
    all(r -> r == r_level, rs) ||
        throw(ArgumentError("resident L2L batch requires a uniform translation radius"))
    phis_work = _array_like_vector(sources.phi, TF, phis)
    thetas_work = _array_like_vector(sources.phi, TF, thetas)
    maps_phi = DegreeMajorMaps(TF, P_phi, sources.phi)
    maps_chi = LH ? DegreeMajorMaps(TF, P_active, sources.chi) : maps_phi
    Ub = _ymode_real_blocks(cache.y_loc_U, P_active, TF)
    Vb = _ymode_real_blocks(cache.y_loc_V, P_active, TF)
    blocks = Vector{TF}(undef, l2l_z_block_length(P_active))
    l2l_z_blocks!(blocks, r_level, P_active)

    aphi = copy(sources.phi)
    _zphi_degree_major_gen!(aphi, phis_work, maps_phi, false)
    yphi = similar(aphi)
    _factored_y_degree_major_gen!(yphi, aphi, Ub, Vb, thetas_work, maps_phi, P_phi)
    zphi = similar(aphi)
    _l2l_ztranslate_shared!(zphi, yphi, blocks, maps_phi, P_phi, P_active)

    if LH
        achi = copy(sources.chi)
        _zphi_degree_major_gen!(achi, phis_work, maps_chi, false)
        ychi = similar(achi)
        _factored_y_degree_major_gen!(ychi, achi, Ub, Vb, thetas_work, maps_chi, P_active)
        zchi = similar(achi)
        _l2l_ztranslate_shared!(zchi, ychi, blocks, maps_chi, P_active, P_active)
        lh_A = Vector{TF}(undef, _operator_ncomplex(P_active))
        lh_B = Vector{TF}(undef, _operator_ncomplex(P_active))
        lamb_helmholtz_local_coeffs!(lh_A, lh_B, r_level, P_active)
        cphi = similar(zphi); cchi = similar(zchi)
        _lh_local_shared_gen!(cphi, cchi, zphi, zchi, lh_A, lh_B, maps_phi, maps_chi, P_phi, P_active)
        zphi = cphi; zchi = cchi
        rchi = similar(zchi)
        _factored_y_degree_major_gen!(rchi, zchi, Ub, Vb, thetas_work, maps_chi, P_active)
        _zphi_degree_major_gen!(rchi, phis_work, maps_chi, true)
        @inbounds targets.chi .+= rchi
    end

    rphi = similar(zphi)
    _factored_y_degree_major_gen!(rphi, zphi, Ub, Vb, thetas_work, maps_phi, P_phi)
    _zphi_degree_major_gen!(rphi, phis_work, maps_phi, true)
    @inbounds targets.phi .+= rphi
    return targets
end

function _degree_major_to_flat_indices(P::Integer)
    idx = Vector{Int}(undef, degree_major_dof(P))
    @inbounds for n in 0:P
        base = degree_row_offset(n)
        for k in 1:(2n + 1)
            ri, m = _ymode_dof_to_storage(k)
            idx[base + k] = flat_basis_index(n, m, ri)
        end
    end
    return idx
end

function _degree_major_buffer_like(::Type{TF}, basis_info::OperatorBasisInfo{B,LH},
        exemplar::FlatCoefficientBuffer, batch::Integer) where {TF,B,LH}
    phi = similar(exemplar.phi, TF, degree_major_dof(basis_info.orders.P_phi), batch)
    chi = LH ? similar(exemplar.chi, TF, degree_major_dof(basis_info.orders.P_active), batch) :
        similar(exemplar.phi, TF, 0, 0)
    fill!(phi, zero(TF))
    fill!(chi, zero(TF))
    return DegreeMajorRealBuffer{TF,typeof(phi),B,LH}(phi, chi, basis_info)
end

# One per-(level, radius) M2M/L2L group or per-(r,θ,φ) shared-M2L subgroup.
# `target_idx` is per-column (targets may repeat for M2M; the launchers accumulate
# atomically). `phi_blocks`/`chi_blocks` are the per-m z matrices used by the shared
# M2L path; M2M/L2L groups instead carry the dense whole-slab embeddings
# `phi_dense`/`chi_dense` (one dof×dof GEMM per channel).
struct ResidentOperatorGroup{I,A,R,OB,DM,LHR}
    level::Int
    source_idx::I
    target_idx::I
    phis::A
    thetas::A
    r::R
    phi_blocks::OB
    chi_blocks::OB
    phi_dense::DM
    chi_dense::DM
    lh_phi_rows::LHR
    lh_chi_rows::LHR
    # Valid-prefix length of source_idx/target_idx/phis/thetas (task 023): recurring
    # steps refresh those arrays in place at capacity and update this count instead
    # of reallocating the group.
    count::Base.RefValue{Int}
end

# The constructor guarantees common types for indices, maps, y blocks, degree-major
# buffers, stage matrices, geometry vectors, and homogeneous group vectors.  Optional
# LH plans/channels retain separate parameters because `nothing` is a real layout.
struct ResidentOperatorWorkspace{TF,B<:AbstractOperatorBasis,LH,
        I,M<:DegreeMajorMaps,YB,NLI,DB,SM,GV,GSTAGE,GM2L,PLAN,YSP,YSC}
    basis_info::OperatorBasisInfo{B,LH}
    phi_flat_idx::I
    chi_flat_idx::I
    maps_phi::M
    maps_chi::M
    y_mult_U::YB
    y_mult_V::YB
    y_loc_U::YB
    y_loc_V::YB
    nonleaf_idx::NLI
    max_batch::Int
    m2l_sources::DB
    m2l_targets::DB
    aphi::SM
    yphi::SM
    zphi::SM
    rphi::SM
    achi::SM
    ychi::SM
    zchi::SM
    rchi::SM
    cphi::SM
    cchi::SM
    phis::GV
    thetas::GV
    rs::GV
    m2m_groups::GSTAGE
    m2l_groups::GM2L
    l2l_groups::GSTAGE
    m2l_concat::PLAN
    ystk_phi::YSP
    ystk_chi::YSC
end

# Partial-application constructor: every call site writes
# `ResidentOperatorWorkspace{TF,B,LH}(fields...)` and the per-field parameters are
# filled in from the arguments here. Concrete `ResidentOperatorWorkspace{TF,B,LH,P...}`
# is a subtype of the UnionAll `ResidentOperatorWorkspace{TF,B,LH}`, so every existing
# method signature and typeassert keeps matching unchanged. Field order duplicates the
# struct above; an arity mismatch fails loudly at the first construction.
function ResidentOperatorWorkspace{TF,B,LH}(basis_info, phi_flat_idx, chi_flat_idx,
        maps_phi, maps_chi, y_mult_U, y_mult_V, y_loc_U, y_loc_V, nonleaf_idx,
        max_batch, m2l_sources, m2l_targets, aphi, yphi, zphi, rphi, achi, ychi,
        zchi, rchi, cphi, cchi, phis, thetas, rs, m2m_groups, m2l_groups, l2l_groups,
        m2l_concat, ystk_phi, ystk_chi) where {TF,B,LH}
    return ResidentOperatorWorkspace{TF,B,LH,
        typeof(phi_flat_idx),typeof(maps_phi),typeof(y_mult_U),
        typeof(nonleaf_idx),typeof(m2l_sources),typeof(aphi),typeof(phis),
        typeof(m2m_groups),typeof(m2l_groups),typeof(m2l_concat),
        typeof(ystk_phi),typeof(ystk_chi)}(
        basis_info, phi_flat_idx, chi_flat_idx, maps_phi, maps_chi,
        y_mult_U, y_mult_V, y_loc_U, y_loc_V, nonleaf_idx, max_batch,
        m2l_sources, m2l_targets, aphi, yphi, zphi, rphi, achi, ychi, zchi, rchi,
        cphi, cchi, phis, thetas, rs, m2m_groups, m2l_groups, l2l_groups,
        m2l_concat, ystk_phi, ystk_chi,
    )
end

"""
    ResidentM2LConcatPlan

Whole-pass M2L execution plan for [`ConcatenatedFixedZM2L`](@ref). Routes are
processed in `chunk`-column slabs in flattened route order (matching
`state.route_targets`/`state.route_sources`); no per-(r,theta,phi) grouping is
required because every stage is per-column parameterized: z rotations and the
factored y rotation take per-column angles, the z-translation uses the separable
form `K_m(r)[n,np] = r^-(n+1/2) (n+np)! r^-(np+1/2)` (per-column diagonal scaling
around fixed factorial GEMMs), and the Lamb-Helmholtz local rows are linear in `r`.
"""
# Whole-slab dense operators and stacked scratch for one channel of the concatenated
# M2L plan. `yU_*`/`yV_*` are the stacked block-diagonal factored-y mode matrices
# (Vs = [blockdiag(Vre); blockdiag(Vim)] :: 2ndof×ndof, Ur = [blockdiag(Ure)
# -blockdiag(Uim)] :: ndof×2ndof), so a y application is two whole-slab GEMMs around
# a per-column e^{iνθ} paired rotation on the stacked [re; im] halves. `zD` is the
# dense fixed z-translation factorial matrix. `nu` holds the per-row Fourier index ν;
# `Cy`/`Sy` are per-chunk cos/sin(νθ) tables shared by the forward and return y
# stages; `scale` is the per-chunk separable r^-(n+1/2) table used both before and
# after `zD`; `G`/`G2` are the preallocated 2ndof×chunk stacked scratch slabs.
struct ConcatChannelOps{M,V}
    yU_mult::M
    yV_mult::M
    yU_loc::M
    yV_loc::M
    zD::M
    nu::V
    Cy::M
    Sy::M
    scale::M
    G::M
    G2::M
end

struct ResidentM2LConcatPlan{GV,RC,XP,XC,OP,OC,LHR,SM,LHS}
    nroutes::Int
    chunk::Int
    # Per-class geometry: all routes of a leaf-level offset class share one (r, θ, φ),
    # so the tables are nclasses long and `route_class` maps each flattened route to
    # its class (non-leaf-level batches fall back to one class per route). The
    # `col_*` vectors are preallocated per-chunk column-parameter gather targets.
    phis::GV
    thetas::GV
    rs::GV
    invrs::GV
    route_class::RC
    col_phi::GV
    col_theta::GV
    col_r::GV
    col_invr::GV
    rexp_phi::XP
    rexp_chi::XC
    ops_phi::OP
    ops_chi::OC
    lh_arow_unit::LHR
    lh_brow_unit::LHR
    aphi::SM
    yphi::SM
    zphi::SM
    rphi::SM
    achi::SM
    ychi::SM
    zchi::SM
    rchi::SM
    cphi::SM
    cchi::SM
    lhgp::LHS
    lhgu::LHS
end

# Dense whole-slab form of the fixed (r-independent) factorial factors of the M2L
# z-translation: Z[row(n,m,ri), row(np,m,ri)] = (n + np)! for n, np >= m with matching
# order m and re/im slot, zero elsewhere (block-diagonal over (m, ri) in degree-major
# indexing). The full production block is recovered as the separable form
# diag(r^-(n+1/2)) * Z * diag(r^-(np+1/2)); see m2l_z_blocks!. One dof×dof GEMM
# replaces the per-m block loop; the ~2-3x zero-fill flop overhead is irrelevant at
# these sizes and buys whole-slab launch economics.
function _m2l_dense_factorial_matrix(exemplar, ::Type{TF}, P_loop::Integer) where TF
    fact = Vector{TF}(undef, 2 * P_loop + 1)
    fact[1] = one(TF)
    @inbounds for k in 1:(2 * P_loop)
        fact[k + 1] = fact[k] * k
    end
    ndof = degree_major_dof(P_loop)
    Z = zeros(TF, ndof, ndof)
    @inbounds for m in 0:P_loop, k in (m == 0 ? (1:1) : (2m:(2m + 1)))
        for np in m:P_loop, n in m:P_loop
            Z[degree_row_offset(n) + k, degree_row_offset(np) + k] = fact[n + np + 1]
        end
    end
    return _array_like_matrix(exemplar, TF, Z)
end

# Stacked block-diagonal dense forms of the per-degree factored-y mode matrices,
# built from the host invariant-cache complex mode vectors (see ConcatChannelOps).
function _ymode_stacked_dense(exemplar, ::Type{TF}, modes_U, modes_V, P::Integer) where TF
    ndof = degree_major_dof(P)
    Ur = zeros(TF, ndof, 2 * ndof)
    Vs = zeros(TF, 2 * ndof, ndof)
    @inbounds for n in 0:P
        d = 2n + 1
        rows = degree_row_range(n)
        segU = reshape(view(modes_U, ymode_offset(n) .+ (1:(d * d))), d, d)
        segV = reshape(view(modes_V, ymode_offset(n) .+ (1:(d * d))), d, d)
        Ur[rows, rows] .= TF.(real.(segU))
        Ur[rows, rows .+ ndof] .= .-TF.(imag.(segU))
        Vs[rows, rows] .= TF.(real.(segV))
        Vs[rows .+ ndof, rows] .= TF.(imag.(segV))
    end
    return _array_like_matrix(exemplar, TF, Ur), _array_like_matrix(exemplar, TF, Vs)
end

# Per-row Fourier index ν of the degree-major layout (νidx = k within a degree block,
# ν = k - n - 1), parameterizing the diag(e^{iνθ}) middle of the factored y stage.
function _degree_row_nus(exemplar, ::Type{TF}, P::Integer) where TF
    nu = Vector{TF}(undef, degree_major_dof(P))
    @inbounds for n in 0:P, k in 1:(2n + 1)
        nu[degree_row_offset(n) + k] = TF(k - n - 1)
    end
    return _array_like_vector(exemplar, TF, nu)
end

function ConcatChannelOps(exemplar, ::Type{TF}, invariant::OperatorInvariantCache,
        P::Integer, chunk::Integer) where TF
    yU_mult, yV_mult = _ymode_stacked_dense(exemplar, TF, invariant.y_mult_U, invariant.y_mult_V, P)
    yU_loc, yV_loc = _ymode_stacked_dense(exemplar, TF, invariant.y_loc_U, invariant.y_loc_V, P)
    ndof = degree_major_dof(P)
    return ConcatChannelOps(
        yU_mult, yV_mult, yU_loc, yV_loc,
        _m2l_dense_factorial_matrix(exemplar, TF, P),
        _degree_row_nus(exemplar, TF, P),
        similar(exemplar, TF, ndof, chunk),
        similar(exemplar, TF, ndof, chunk),
        similar(exemplar, TF, ndof, chunk),
        similar(exemplar, TF, 2 * ndof, chunk),
        similar(exemplar, TF, 2 * ndof, chunk),
    )
end

# Whole-slab stacked factored-y operators + per-stage scratch for one channel of the
# resident M2M/L2L group launchers (the group-loop sibling of ConcatChannelOps):
# fixed stacked mult/loc mode matrices, per-row ν, and trig/stacked scratch sized to
# the workspace max batch. The z translation uses each group's dense matrix.
struct StackedYChannel{M,V}
    mult_Ur::M
    mult_Vs::M
    loc_Ur::M
    loc_Vs::M
    nu::V
    Cy::M
    Sy::M
    G::M
    G2::M
end

function StackedYChannel(exemplar, ::Type{TF}, invariant::OperatorInvariantCache,
        P::Integer, width::Integer) where TF
    mult_Ur, mult_Vs = _ymode_stacked_dense(exemplar, TF, invariant.y_mult_U, invariant.y_mult_V, P)
    loc_Ur, loc_Vs = _ymode_stacked_dense(exemplar, TF, invariant.y_loc_U, invariant.y_loc_V, P)
    ndof = degree_major_dof(P)
    return StackedYChannel(
        mult_Ur, mult_Vs, loc_Ur, loc_Vs,
        _degree_row_nus(exemplar, TF, P),
        similar(exemplar, TF, ndof, width),
        similar(exemplar, TF, ndof, width),
        similar(exemplar, TF, 2 * ndof, width),
        similar(exemplar, TF, 2 * ndof, width),
    )
end

# Whole-slab factored y application: out = Ur * (e^{iνθ} ∘ (Vs * in)), the dense
# stacked replacement for the per-degree loop in _factored_y_degree_major_gen!.
# `C`/`S` are the precomputed per-chunk cos/sin(νθ) tables (ndof×n); the paired
# rotation acts on the contiguous [re; im] halves of the stacked scratch, so it is
# allocation-free strided broadcasting on both Array and CuArray.
# C = A * B for the resident operator chain. The generic method is plain
# `mul!`; the CUDA extension overrides it (task 029 cycle 1) to call
# `CUBLAS.gemm!` with construction-staged device alpha/beta scalars, because in
# CUBLAS_POINTER_MODE_DEVICE a scalar-alpha/beta `mul!` stages a fresh `CuRef`
# per call — one device allocation plus one pageable H2D memcpy, which is both
# the dominant M2M/L2L host-overhead term (job 13059955: 101 pageable H2Ds per
# step) and a CUDA-graph-capture blocker.
_resident_mul!(C, A, B) = mul!(C, A, B)

function _stacked_y_dense!(out_slab, in_slab, Ur, Vs, C, S, G, G2, ndof::Integer)
    _resident_mul!(G, Vs, in_slab)
    Gt = @view G[1:ndof, :]
    Gb = @view G[(ndof + 1):(2 * ndof), :]
    G2t = @view G2[1:ndof, :]
    G2b = @view G2[(ndof + 1):(2 * ndof), :]
    G2t .= C .* Gt .- S .* Gb
    G2b .= S .* Gt .+ C .* Gb
    _resident_mul!(out_slab, Ur, G2)
    return out_slab
end

function _degree_row_exponents(exemplar, ::Type{TF}, P::Integer) where TF
    rexp = Vector{TF}(undef, degree_major_dof(P))
    @inbounds for n in 0:P
        rexp[degree_row_range(n)] .= TF(n) + TF(0.5)
    end
    return _array_like_vector(exemplar, TF, rexp)
end

function ResidentM2LConcatPlan(::Type{TF}, basis_info::OperatorBasisInfo{B,LH}, exemplar,
        strategy::ConcatenatedFixedZM2L, invariant::OperatorInvariantCache,
        list::RadixInteractionList, leaf_level::Integer,
        host_route_targets, host_route_sources, host_node_centers) where {TF,B,LH}
    P_phi = basis_info.orders.P_phi
    P_active = basis_info.orders.P_active
    nroutes = length(host_route_targets)
    # Per-class geometry: flattened route order is batch-major, and every route of a
    # leaf-level batch shares the same center displacement (source coord = target
    # coord - offset on the uniform grid), so one (r, θ, φ) per such batch suffices.
    # Batches below the leaf level pair leaf descendants with varying displacements
    # and get one class per route.
    route_class = Vector{Int32}(undef, nroutes)
    phis = TF[]
    thetas = TF[]
    rs = TF[]
    route_geometry = i -> begin
        t = host_route_targets[i]
        s = host_route_sources[i]
        dx = host_node_centers[1, t] - host_node_centers[1, s]
        dy = host_node_centers[2, t] - host_node_centers[2, s]
        dz = host_node_centers[3, t] - host_node_centers[3, s]
        cartesian_to_spherical(SVector{3,TF}(dx, dy, dz))
    end
    push_class! = (r, theta, phi) -> begin
        push!(rs, TF(r)); push!(thetas, TF(theta)); push!(phis, TF(phi))
        return Int32(length(rs))
    end
    i = 0
    for batch in list.m2l_batches
        nb = length(batch.targets)
        if batch.level == leaf_level
            cls = push_class!(route_geometry(i + 1)...)
            fill!(view(route_class, (i + 1):(i + nb)), cls)
        else
            for j in 1:nb
                route_class[i + j] = push_class!(route_geometry(i + j)...)
            end
        end
        i += nb
    end
    i == nroutes || throw(ArgumentError(
        "interaction-list batches ($i routes) do not match flattened routes ($nroutes)"))
    chunk = max(min(strategy.chunk, max(nroutes, 1)), 1)
    ndof_phi = degree_major_dof(P_phi)
    ndof_chi = LH ? degree_major_dof(P_active) : 0
    lh_arow_unit, lh_brow_unit = LH ?
        _resident_lh_rows_like(exemplar, TF, P_phi, P_active, one(TF), :local) :
        (nothing, nothing)
    mkphi() = similar(exemplar, TF, ndof_phi, chunk)
    mkchi() = similar(exemplar, TF, ndof_chi, LH ? chunk : 0)
    return ResidentM2LConcatPlan(
        nroutes, chunk,
        _array_like_vector(exemplar, TF, phis),
        _array_like_vector(exemplar, TF, thetas),
        _array_like_vector(exemplar, TF, rs),
        _array_like_vector(exemplar, TF, inv.(rs)),
        _array_like_vector(exemplar, Int32, route_class),
        similar(exemplar, TF, chunk),
        similar(exemplar, TF, chunk),
        similar(exemplar, TF, chunk),
        similar(exemplar, TF, chunk),
        _degree_row_exponents(exemplar, TF, P_phi),
        LH ? _degree_row_exponents(exemplar, TF, P_active) : nothing,
        ConcatChannelOps(exemplar, TF, invariant, P_phi, chunk),
        LH ? ConcatChannelOps(exemplar, TF, invariant, P_active, chunk) : nothing,
        lh_arow_unit, lh_brow_unit,
        mkphi(), mkphi(), mkphi(), mkphi(),
        mkchi(), mkchi(), mkchi(), mkchi(),
        LH ? mkphi() : similar(exemplar, TF, 0, 0), mkchi(),
        LH ? mkphi() : nothing, LH ? mkchi() : nothing,
    )
end

function _degree_major_buffer_view(buf::DegreeMajorRealBuffer{TF,A,B,LH},
        nbatch::Integer) where {TF,A,B,LH}
    phi = @view buf.phi[:, 1:nbatch]
    chi = LH ? (@view buf.chi[:, 1:nbatch]) : (@view buf.chi[:, 1:0])
    return DegreeMajorRealBuffer{TF,typeof(phi),B,LH}(phi, chi, buf.basis_info)
end

function _matrix_col_view(mat, nbatch::Integer)
    return @view mat[:, 1:nbatch]
end

# Valid-prefix view of a capacity-sized vector (task 023); returns the vector
# itself when the prefix spans it, keeping the one-shot path allocation-identical.
_vector_prefix_view(v, n::Integer) = length(v) == n ? v : @view v[1:n]

function _resident_lh_rows_like(exemplar, ::Type{TF}, P_phi::Integer, P_active::Integer,
        r, kind::Symbol) where TF
    lh_A = Vector{TF}(undef, _operator_ncomplex(P_active))
    lh_B = Vector{TF}(undef, _operator_ncomplex(P_active))
    if kind === :multipole
        lamb_helmholtz_multipole_coeffs!(lh_A, lh_B, r, P_active)
        arow, brow = _lh_row_coefficients(TF, P_phi, P_active, lh_A, lh_B)
    elseif kind === :local
        lamb_helmholtz_local_coeffs!(lh_A, lh_B, r, P_active)
        arow, brow = _lh_local_row_coefficients(TF, P_phi, P_active, lh_A, lh_B)
    else
        throw(ArgumentError("unknown Lamb-Helmholtz row kind $kind"))
    end
    return (
        _array_like_vector(exemplar, TF, arow),
        _array_like_vector(exemplar, TF, brow),
    )
end

function _resident_group(exemplar, ::Type{TF}, basis_info, kind::Symbol, level::Integer,
        source_idx_host::Vector{Int}, target_idx_host::Vector{Int}, phis_host::Vector{TF},
        thetas_host::Vector{TF}, rs_host::Vector{TF}) where TF
    P_phi = basis_info.orders.P_phi
    P_active = basis_info.orders.P_active
    r = isempty(rs_host) ? zero(TF) : rs_host[1]
    all(==(r), rs_host) ||
        throw(ArgumentError("resident $kind group requires uniform radius"))
    if kind === :m2l
        phi0 = isempty(phis_host) ? zero(TF) : phis_host[1]
        theta0 = isempty(thetas_host) ? zero(TF) : thetas_host[1]
        all(==(phi0), phis_host) ||
            throw(ArgumentError("resident M2L group requires uniform phi"))
        all(==(theta0), thetas_host) ||
            throw(ArgumentError("resident M2L group requires uniform theta"))
    end

    zlen = kind === :m2m ? m2m_z_block_length(P_active) :
        kind === :m2l ? m2l_z_block_length(P_active) :
        kind === :l2l ? l2l_z_block_length(P_active) :
        throw(ArgumentError("unknown resident group kind $kind"))
    blocks = Vector{TF}(undef, zlen)
    kind === :m2m ? m2m_z_blocks!(blocks, r, P_active) :
        kind === :m2l ? m2l_z_blocks!(blocks, r, P_active) :
        l2l_z_blocks!(blocks, r, P_active)
    # The shared M2L group path applies per-m blocks; M2M/L2L groups run the dense
    # whole-slab chain and carry the dof×dof embedding instead.
    if kind === :m2l
        phi_blocks = _z_block_matrices_like(exemplar, TF, blocks, kind, P_phi, P_active)
        chi_blocks = P_active == P_phi ? phi_blocks :
            _z_block_matrices_like(exemplar, TF, blocks, kind, P_active, P_active)
        phi_dense = nothing
        chi_dense = nothing
    else
        phi_blocks = nothing
        chi_blocks = nothing
        phi_dense = _z_dense_matrix_like(exemplar, TF, blocks, kind, P_phi, P_active)
        chi_dense = P_active == P_phi ? phi_dense :
            _z_dense_matrix_like(exemplar, TF, blocks, kind, P_active, P_active)
    end
    lh_phi_rows, lh_chi_rows = _resident_lh_rows_like(
        exemplar, TF, P_phi, P_active, r, kind === :m2m ? :multipole : :local,
    )
    return ResidentOperatorGroup(
        Int(level),
        _array_like_vector(exemplar, Int, source_idx_host),
        _array_like_vector(exemplar, Int, target_idx_host),
        _array_like_vector(exemplar, TF, phis_host),
        _array_like_vector(exemplar, TF, thetas_host),
        r,
        phi_blocks,
        chi_blocks,
        phi_dense,
        chi_dense,
        lh_phi_rows,
        lh_chi_rows,
        Ref(length(source_idx_host)),
    )
end

function _zero_resident_nonleaf_multipoles!(state::DeviceResidentRadixState{TF,B,LH}) where {TF,B,LH}
    # DeviceRadixGrid nodes are level-major with leaves last, so the nonleaf set is
    # exactly the first (n_nodes - n_cells) columns; the prefix fill works for both
    # host and CUDA arrays and stays correct under per-step counts (task 023).
    if state.grid isa DeviceRadixGrid && state.counts.n_nodes > 0
        n_nonleaf = state.counts.n_nodes - state.counts.n_cells
        n_nonleaf <= 0 && return state
        fill!(view(state.multipoles.phi, :, 1:n_nonleaf), zero(TF))
        LH && fill!(view(state.multipoles.chi, :, 1:n_nonleaf), zero(TF))
        return state
    end
    if state.scratch isa ResidentOperatorWorkspace
        idx = state.scratch.nonleaf_idx
        length(idx) == 0 && return state
        state.multipoles.phi[:, idx] .= zero(TF)
        LH && (state.multipoles.chi[:, idx] .= zero(TF))
        return state
    end
    levels = state.host_node_levels
    levels === nothing &&
        throw(ArgumentError("resident nonleaf zeroing requires host node-level metadata or ResidentOperatorWorkspace scratch"))
    nonleaf = findall(<(state.grid.ell), levels)
    isempty(nonleaf) && return state
    idx = _array_like_vector(state.multipoles.phi, Int, nonleaf)
    state.multipoles.phi[:, idx] .= zero(TF)
    LH && (state.multipoles.chi[:, idx] .= zero(TF))
    return state
end

function _collect_level_edges(parent_routes, child_routes, node_levels, node_centers,
        level::Integer, direction::Symbol, ::Type{TF}) where TF
    parents = Int[]
    children = Int[]
    phis = TF[]
    thetas = TF[]
    rs = TF[]
    @inbounds for edge in eachindex(parent_routes)
        parent = parent_routes[edge]
        child = child_routes[edge]
        parent == 0 && continue
        if direction === :parent_to_child
            node_levels[child] == level || continue
            dx = node_centers[1, child] - node_centers[1, parent]
            dy = node_centers[2, child] - node_centers[2, parent]
            dz = node_centers[3, child] - node_centers[3, parent]
        elseif direction === :child_to_parent
            node_levels[parent] == level || continue
            dx = node_centers[1, parent] - node_centers[1, child]
            dy = node_centers[2, parent] - node_centers[2, child]
            dz = node_centers[3, parent] - node_centers[3, child]
        else
            throw(ArgumentError("unknown resident edge direction $direction"))
        end
        r, theta, phi = cartesian_to_spherical(SVector{3,TF}(dx, dy, dz))
        push!(parents, parent)
        push!(children, child)
        push!(rs, TF(r))
        push!(thetas, TF(theta))
        push!(phis, TF(phi))
    end
    return parents, children, phis, thetas, rs
end

function _assert_m2l_subgroup_targets_unique!(target_nodes, group)
    targets = target_nodes[group]
    length(unique(targets)) == length(targets) ||
        throw(AssertionError("resident M2L requires unique targets within each concrete translation-vector subgroup"))
    return nothing
end

# Convert generated groups explicitly to one concrete element type.  A heterogeneous
# result is a construction error: silently widening it would restore dynamic dispatch
# in every resident group loop.
function _homogeneous_groups(groups::AbstractVector)
    isempty(groups) && return groups
    T = typeof(first(groups))
    isconcretetype(T) ||
        throw(AssertionError("resident operator group type must be concrete, got $T"))
    all(group -> typeof(group) === T, groups) ||
        throw(AssertionError("resident operator groups must have one concrete element type"))
    result = Vector{T}(groups)
    @assert isconcretetype(eltype(result))
    return result
end

function _resident_m2m_groups(exemplar, ::Type{TF}, basis_info, grid, parent_routes,
        child_routes, node_levels, node_centers) where TF
    groups = ResidentOperatorGroup[]
    for level in (grid.ell - 1):-1:0
        parents, children, phis, thetas, rs = _collect_level_edges(
            parent_routes, child_routes, node_levels, node_centers, level, :child_to_parent, TF,
        )
        isempty(children) && continue
        for r_group in unique(rs)
            group = findall(==(r_group), rs)
            # target_idx is per-column (one parent per child); the launcher
            # accumulates atomically, so repeated parents need no scatter matrix.
            push!(groups, _resident_group(
                exemplar, TF, basis_info, :m2m, level,
                children[group], parents[group], phis[group], thetas[group], rs[group],
            ))
        end
    end
    return _homogeneous_groups(groups)
end

function _resident_m2l_groups(exemplar, ::Type{TF}, basis_info, list, route_sources,
        route_targets, node_centers) where TF
    groups_out = ResidentOperatorGroup[]
    route_i = 0
    @inbounds for batch in list.m2l_batches
        nbatch = length(batch.targets)
        nbatch == 0 && continue
        route_range = (route_i + 1):(route_i + nbatch)
        route_i += nbatch
        source_nodes = route_sources[route_range]
        target_nodes = route_targets[route_range]
        groups = Dict{Tuple{TF,TF,TF},Vector{Int}}()
        for j in 1:nbatch
            target = target_nodes[j]
            source = source_nodes[j]
            dx = node_centers[1, target] - node_centers[1, source]
            dy = node_centers[2, target] - node_centers[2, source]
            dz = node_centers[3, target] - node_centers[3, source]
            r, theta, phi = cartesian_to_spherical(SVector{3,TF}(dx, dy, dz))
            push!(get!(() -> Int[], groups, (TF(r), TF(theta), TF(phi))), j)
        end
        for ((r, theta, phi), group) in groups
            _assert_m2l_subgroup_targets_unique!(target_nodes, group)
            ngroup = length(group)
            push!(groups_out, _resident_group(
                exemplar, TF, basis_info, :m2l, batch.level,
                source_nodes[group], target_nodes[group], fill(phi, ngroup),
                fill(theta, ngroup), fill(r, ngroup),
            ))
        end
    end
    return _homogeneous_groups(groups_out)
end

function _resident_l2l_groups(exemplar, ::Type{TF}, basis_info, grid, parent_routes,
        child_routes, node_levels, node_centers) where TF
    groups = ResidentOperatorGroup[]
    for level in 1:grid.ell
        parents, children, phis, thetas, rs = _collect_level_edges(
            parent_routes, child_routes, node_levels, node_centers, level, :parent_to_child, TF,
        )
        isempty(children) && continue
        for r_group in unique(rs)
            group = findall(==(r_group), rs)
            push!(groups, _resident_group(
                exemplar, TF, basis_info, :l2l, level,
                parents[group], children[group], phis[group], thetas[group], rs[group],
            ))
        end
    end
    return _homogeneous_groups(groups)
end

function ResidentOperatorWorkspace(::Type{TF}, basis_info::OperatorBasisInfo{B,LH},
        exemplar::FlatCoefficientBuffer, grid, list, host_m2m_parent_routes,
        host_m2m_child_routes, host_l2l_parent_routes, host_l2l_child_routes,
        host_node_levels, host_node_centers, host_route_targets, host_route_sources;
        m2l_strategy::AbstractResidentM2LStrategy=SharedRotationM2L(),
        operator::AbstractM2LOperator=MaterializedYRotationM2L()) where {TF,B,LH}
    phi_flat_idx = _array_like_vector(exemplar.phi, Int, _degree_major_to_flat_indices(basis_info.orders.P_phi))
    chi_flat_idx = LH ?
        _array_like_vector(exemplar.chi, Int, _degree_major_to_flat_indices(basis_info.orders.P_active)) :
        _array_like_vector(exemplar.phi, Int, Int[])
    maps_phi = DegreeMajorMaps(TF, basis_info.orders.P_phi, exemplar.phi)
    maps_chi = LH ? DegreeMajorMaps(TF, basis_info.orders.P_active, exemplar.chi) : maps_phi
    invariant = OperatorInvariantCache(TF, basis_info)
    y_mult_U = _ymode_real_blocks(_array_like_vector(exemplar.phi, Complex{TF}, invariant.y_mult_U), basis_info.orders.P_active, TF)
    y_mult_V = _ymode_real_blocks(_array_like_vector(exemplar.phi, Complex{TF}, invariant.y_mult_V), basis_info.orders.P_active, TF)
    y_loc_U = _ymode_real_blocks(_array_like_vector(exemplar.phi, Complex{TF}, invariant.y_loc_U), basis_info.orders.P_active, TF)
    y_loc_V = _ymode_real_blocks(_array_like_vector(exemplar.phi, Complex{TF}, invariant.y_loc_V), basis_info.orders.P_active, TF)
    nonleaf_idx = _array_like_vector(exemplar.phi, Int, findall(<(grid.ell), host_node_levels))

    m2m_groups = _resident_m2m_groups(
        exemplar.phi, TF, basis_info, grid, host_m2m_parent_routes, host_m2m_child_routes,
        host_node_levels, host_node_centers,
    )
    # Per-(r,theta,phi) M2L groups (and their per-group device uploads) are only
    # needed by the group-looping SharedRotationM2L path; the concatenated strategy
    # replaces them with a whole-pass plan.
    concat = m2l_strategy isa ConcatenatedFixedZM2L
    precomputed_y = m2l_strategy isa PrecomputedFactoredYM2L
    dense = m2l_strategy isa DenseTranslationM2L
    factored = operator isa FactoredRotationM2L
    dense && factored && throw(ArgumentError(
        "DenseTranslationM2L requires operator=MaterializedYRotationM2L()"))
    m2l_groups = ((concat && !factored) || precomputed_y || dense) ? ResidentOperatorGroup[] : _resident_m2l_groups(
        exemplar.phi, TF, basis_info, list, host_route_sources, host_route_targets,
        host_node_centers,
    )
    if dense
        accepted_offsets = unique(batch.offset for batch in list.m2l_batches)
        m2l_concat = ResidentM2LDensePlan(TF, basis_info, accepted_offsets,
            (2 * grid.h0) / (1 << grid.ell), length(host_route_targets),
            max(grid.n_cells, 1), 1 << grid.ell, m2l_strategy, invariant)
        offset_id = Dict(offset => i for (i, offset) in enumerate(accepted_offsets))
        route_i = 0
        for batch in list.m2l_batches
            count = length(batch.targets)
            fill!(view(m2l_concat.route_class, (route_i + 1):(route_i + count)),
                Int32(offset_id[batch.offset]))
            route_i += count
        end
        _refresh_dense_m2l_routes!(m2l_concat, host_route_sources,
            host_route_targets, length(host_route_targets))
    elseif precomputed_y
        factored || throw(ArgumentError(
            "PrecomputedFactoredYM2L requires operator=FactoredRotationM2L()"))
        accepted_offsets = unique(batch.offset for batch in list.m2l_batches)
        m2l_concat = ResidentM2LPrecomputedYPlan(TF, basis_info, exemplar.phi,
            accepted_offsets, (2 * grid.h0) / (1 << grid.ell),
            length(host_route_targets), max(grid.n_cells, 1), 1 << grid.ell, invariant)
        offset_id = Dict(offset => i for (i, offset) in enumerate(accepted_offsets))
        route_i = 0
        for batch in list.m2l_batches
            count = length(batch.targets)
            fill!(view(m2l_concat.route_class, (route_i + 1):(route_i + count)),
                Int32(offset_id[batch.offset]))
            route_i += count
        end
        _refresh_precomputed_y_m2l_routes!(m2l_concat, host_route_sources,
            host_route_targets, length(host_route_targets))
    else
        m2l_concat = factored ? ResidentM2LFactoredPlan(
        _array_like_vector(exemplar.phi, Int32, zeros(Int32, length(host_route_targets))),
        m2l_groups,
    ) : concat ? ResidentM2LConcatPlan(
        TF, basis_info, exemplar.phi, m2l_strategy, invariant, list, grid.ell,
        host_route_targets, host_route_sources, host_node_centers,
    ) : nothing
    end
    l2l_groups = _resident_l2l_groups(
        exemplar.phi, TF, basis_info, grid, host_l2l_parent_routes, host_l2l_child_routes,
        host_node_levels, host_node_centers,
    )
    max_batch = maximum((
        maximum((length(g.source_idx) for g in m2m_groups); init=0),
        maximum((length(g.source_idx) for g in m2l_groups); init=0),
        maximum((length(g.source_idx) for g in l2l_groups); init=0),
        1,
    ))

    m2l_sources = _degree_major_buffer_like(TF, basis_info, exemplar, max_batch)
    m2l_targets = _degree_major_buffer_like(TF, basis_info, exemplar, max_batch)
    aphi = similar(m2l_sources.phi); yphi = similar(m2l_sources.phi)
    zphi = similar(m2l_sources.phi); rphi = similar(m2l_sources.phi)
    achi = similar(m2l_sources.chi); ychi = similar(m2l_sources.chi)
    zchi = similar(m2l_sources.chi); rchi = similar(m2l_sources.chi)
    cphi = similar(m2l_sources.phi); cchi = similar(m2l_sources.chi)
    phis = similar(exemplar.phi, TF, max_batch)
    thetas = similar(exemplar.phi, TF, max_batch)
    rs = similar(exemplar.phi, TF, max_batch)
    ystk_phi = StackedYChannel(exemplar.phi, TF, invariant, basis_info.orders.P_phi, max_batch)
    ystk_chi = LH ?
        StackedYChannel(exemplar.phi, TF, invariant, basis_info.orders.P_active, max_batch) :
        nothing
    return ResidentOperatorWorkspace{TF,B,LH}(
        basis_info, phi_flat_idx, chi_flat_idx, maps_phi, maps_chi,
        y_mult_U, y_mult_V, y_loc_U, y_loc_V, nonleaf_idx, max_batch,
        m2l_sources, m2l_targets,
        aphi, yphi, zphi, rphi, achi, ychi, zchi, rchi, cphi, cchi,
        phis, thetas, rs, m2m_groups, m2l_groups, l2l_groups, m2l_concat,
        ystk_phi, ystk_chi,
    )
end

# Whole-slab resident M2M/L2L group application (the concat treatment of the M2M/L2L
# stages): fused gather+Z_φ, stacked-y GEMMs, one dense per-group z GEMM, [LH row
# mix], stacked return y, and fused Z_φ^{-1} + atomic scatter-accumulate. `dest` and
# `src` are flat coefficient buffers (M2M reads and writes `multipoles` at different
# levels; L2L reads and writes `locals`). M2M uses the multipole y modes and the
# multipole LH recurrence (row_down); L2L the local modes and row_up.
function _resident_stage_group_apply!(dest::FlatCoefficientBuffer, src::FlatCoefficientBuffer,
        group::ResidentOperatorGroup, ws::ResidentOperatorWorkspace{TF,B,LH},
        kind::Symbol) where {TF,B,LH}
    n = group.count[]
    n == 0 && return dest
    mult = kind === :m2m
    ystk = ws.ystk_phi
    Ur = mult ? ystk.mult_Ur : ystk.loc_Ur
    Vs = mult ? ystk.mult_Vs : ystk.loc_Vs
    ndof_phi = size(ws.aphi, 1)
    source_idx = _vector_prefix_view(group.source_idx, n)
    target_idx = _vector_prefix_view(group.target_idx, n)
    group_phis = _vector_prefix_view(group.phis, n)
    group_thetas = _vector_prefix_view(group.thetas, n)
    aphi = _matrix_col_view(ws.aphi, n); yphi = _matrix_col_view(ws.yphi, n)
    zphi = _matrix_col_view(ws.zphi, n); rphi = _matrix_col_view(ws.rphi, n)
    cphi = _matrix_col_view(ws.cphi, n)
    C = _matrix_col_view(ystk.Cy, n); S = _matrix_col_view(ystk.Sy, n)
    G = _matrix_col_view(ystk.G, n); G2 = _matrix_col_view(ystk.G2, n)
    thetas_row = transpose(group_thetas)
    C .= cos.(ystk.nu .* thetas_row)
    S .= sin.(ystk.nu .* thetas_row)
    _gather_rotate_z!(aphi, src.phi, ws.phi_flat_idx, source_idx,
        ws.maps_phi.row_m, ws.maps_phi.row_ssign, ws.maps_phi.row_pair, group_phis, false)
    _stacked_y_dense!(yphi, aphi, Ur, Vs, C, S, G, G2, ndof_phi)
    _resident_mul!(zphi, group.phi_dense, yphi)
    ret_phi = zphi
    if LH
        ystk_c = ws.ystk_chi
        Urc = mult ? ystk_c.mult_Ur : ystk_c.loc_Ur
        Vsc = mult ? ystk_c.mult_Vs : ystk_c.loc_Vs
        ndof_chi = size(ws.achi, 1)
        achi = _matrix_col_view(ws.achi, n); ychi = _matrix_col_view(ws.ychi, n)
        zchi = _matrix_col_view(ws.zchi, n); rchi = _matrix_col_view(ws.rchi, n)
        cphi = _matrix_col_view(ws.cphi, n); cchi = _matrix_col_view(ws.cchi, n)
        Cc = _matrix_col_view(ystk_c.Cy, n); Sc = _matrix_col_view(ystk_c.Sy, n)
        Gc = _matrix_col_view(ystk_c.G, n); G2c = _matrix_col_view(ystk_c.G2, n)
        Cc .= cos.(ystk_c.nu .* thetas_row)
        Sc .= sin.(ystk_c.nu .* thetas_row)
        _gather_rotate_z!(achi, src.chi, ws.chi_flat_idx, source_idx,
            ws.maps_chi.row_m, ws.maps_chi.row_ssign, ws.maps_chi.row_pair, group_phis, false)
        _stacked_y_dense!(ychi, achi, Urc, Vsc, Cc, Sc, Gc, G2c, ndof_chi)
        _resident_mul!(zchi, group.chi_dense, ychi)
        # LH row mix (multipole rows pair with (n-1, m) via row_down; local rows with
        # (n+1, m) via row_up); yphi/ychi are free again and serve as gather scratch.
        chi_rows = mult ? ws.maps_chi.row_down : ws.maps_chi.row_up
        _gather_rows!(yphi, zchi, ws.maps_phi.row_pair)
        _gather_rows!(ychi, zchi, chi_rows)
        cphi .= zphi .+ group.lh_phi_rows .* yphi
        cchi .= zchi .+ group.lh_chi_rows .* ychi
        _stacked_y_dense!(rchi, cchi, Urc, Vsc, Cc, Sc, Gc, G2c, ndof_chi)
        _rotate_z_scatter_accumulate!(dest.chi, rchi, ws.chi_flat_idx, target_idx,
            ws.maps_chi.row_m, ws.maps_chi.row_ssign, ws.maps_chi.row_pair, group_phis)
        ret_phi = cphi
    end
    _stacked_y_dense!(rphi, ret_phi, Ur, Vs, C, S, G, G2, ndof_phi)
    _rotate_z_scatter_accumulate!(dest.phi, rphi, ws.phi_flat_idx, target_idx,
        ws.maps_phi.row_m, ws.maps_phi.row_ssign, ws.maps_phi.row_pair, group_phis)
    return dest
end

function _resident_execute_shared_m2l!(targets::DegreeMajorRealBuffer{TF,A,B,LH},
        sources::DegreeMajorRealBuffer, group::ResidentOperatorGroup,
        ws::ResidentOperatorWorkspace{TF,B,LH}) where {TF,A,B,LH}
    P_phi = ws.basis_info.orders.P_phi
    P_active = ws.basis_info.orders.P_active
    n = flat_nbatch(sources)
    group_phis = _vector_prefix_view(group.phis, n)
    group_thetas = _vector_prefix_view(group.thetas, n)
    aphi = _matrix_col_view(ws.aphi, n); yphi = _matrix_col_view(ws.yphi, n)
    zphi = _matrix_col_view(ws.zphi, n); rphi = _matrix_col_view(ws.rphi, n)
    copyto!(aphi, sources.phi)
    _zphi_degree_major_gen!(aphi, group_phis, ws.maps_phi, false)
    _factored_y_degree_major_gen!(yphi, aphi, ws.y_mult_U, ws.y_mult_V, group_thetas, ws.maps_phi, P_phi)
    _ztranslate_shared_precomputed!(zphi, yphi, group.phi_blocks, ws.maps_phi, P_phi)
    if LH
        achi = _matrix_col_view(ws.achi, n); ychi = _matrix_col_view(ws.ychi, n)
        zchi = _matrix_col_view(ws.zchi, n); rchi = _matrix_col_view(ws.rchi, n)
        cphi = _matrix_col_view(ws.cphi, n); cchi = _matrix_col_view(ws.cchi, n)
        copyto!(achi, sources.chi)
        _zphi_degree_major_gen!(achi, group_phis, ws.maps_chi, false)
        _factored_y_degree_major_gen!(ychi, achi, ws.y_mult_U, ws.y_mult_V, group_thetas, ws.maps_chi, P_active)
        _ztranslate_shared_precomputed!(zchi, ychi, group.chi_blocks, ws.maps_chi, P_active)
        _lh_local_shared_rows!(cphi, cchi, zphi, zchi, group.lh_phi_rows, group.lh_chi_rows, ws.maps_phi, ws.maps_chi)
        copyto!(zphi, cphi)
        _factored_y_degree_major_gen!(rchi, cchi, ws.y_loc_U, ws.y_loc_V, group_thetas, ws.maps_chi, P_active)
        _zphi_degree_major_gen!(rchi, group_phis, ws.maps_chi, true)
        targets.chi .+= rchi
    end
    _factored_y_degree_major_gen!(rphi, zphi, ws.y_loc_U, ws.y_loc_V, group_thetas, ws.maps_phi, P_phi)
    _zphi_degree_major_gen!(rphi, group_phis, ws.maps_phi, true)
    targets.phi .+= rphi
    return targets
end

# Grouped factored M2L directly between flat resident buffers. The offset class
# supplies one shared (phi, theta, r), so the Plain-H U/V applications below are
# per-degree GEMMs over all routes in the class. No Ts(theta) is materialized.
function _resident_factored_m2l_group_apply!(state::DeviceResidentRadixState{TF,B,LH},
        group::ResidentOperatorGroup, ws::ResidentOperatorWorkspace{TF,B,LH}) where {TF,B,LH}
    n = group.count[]
    n == 0 && return state
    P_phi = ws.basis_info.orders.P_phi
    P_active = ws.basis_info.orders.P_active
    source_idx = group.source_idx::Vector{Int}
    target_idx = group.target_idx::Vector{Int}
    phis = group.phis::Vector{TF}
    thetas = group.thetas::Vector{TF}
    aphi = ws.aphi::Matrix{TF}; yphi = ws.yphi::Matrix{TF}
    zphi = ws.zphi::Matrix{TF}; rphi = ws.rphi::Matrix{TF}
    cphi = ws.cphi::Matrix{TF}
    phi_flat_idx = ws.phi_flat_idx::Vector{Int}
    phi_row_m = ws.maps_phi.row_m::Vector{TF}
    phi_row_ssign = ws.maps_phi.row_ssign::Vector{TF}
    phi_row_pair = ws.maps_phi.row_pair::Vector{Int}
    y_mult_U = ws.y_mult_U::Vector{Tuple{Matrix{TF},Matrix{TF}}}
    y_mult_V = ws.y_mult_V::Vector{Tuple{Matrix{TF},Matrix{TF}}}
    y_loc_U = ws.y_loc_U::Vector{Tuple{Matrix{TF},Matrix{TF}}}
    y_loc_V = ws.y_loc_V::Vector{Tuple{Matrix{TF},Matrix{TF}}}
    _gather_rotate_z_n!(aphi, state.multipoles.phi, phi_flat_idx, source_idx,
        phi_row_m, phi_row_ssign, phi_row_pair, phis, false, n)
    _factored_y_degree_major_auto!(yphi, aphi, y_mult_U, y_mult_V,
        thetas, cphi, rphi, zphi, P_phi, n)
    _ztranslate_shared_precomputed_noalloc!(zphi, yphi,
        group.phi_blocks::Vector{Matrix{TF}}, ws.maps_phi, P_phi, n)
    ret_phi = zphi
    ret_scratch = cphi
    if LH
        achi = ws.achi::Matrix{TF}; ychi = ws.ychi::Matrix{TF}
        zchi = ws.zchi::Matrix{TF}; rchi = ws.rchi::Matrix{TF}
        cchi = ws.cchi::Matrix{TF}
        chi_flat_idx = ws.chi_flat_idx::Vector{Int}
        chi_row_m = ws.maps_chi.row_m::Vector{TF}
        chi_row_ssign = ws.maps_chi.row_ssign::Vector{TF}
        chi_row_pair = ws.maps_chi.row_pair::Vector{Int}
        _gather_rotate_z_n!(achi, state.multipoles.chi, chi_flat_idx, source_idx,
            chi_row_m, chi_row_ssign, chi_row_pair, phis, false, n)
        _factored_y_degree_major_auto!(ychi, achi, y_mult_U, y_mult_V,
            thetas, cchi, rchi, zchi, P_active, n)
        _ztranslate_shared_precomputed_noalloc!(zchi, ychi,
            group.chi_blocks::Vector{Matrix{TF}}, ws.maps_chi, P_active, n)
        _lh_local_shared_rows_noalloc!(cphi, cchi, zphi, zchi, group.lh_phi_rows,
            group.lh_chi_rows, ws.maps_phi, ws.maps_chi, n)
        _factored_y_degree_major_auto!(rchi, cchi, y_loc_U, y_loc_V,
            thetas, achi, ychi, zchi, P_active, n)
        _rotate_z_scatter_accumulate_n!(state.locals.chi, rchi, chi_flat_idx,
            target_idx, chi_row_m, chi_row_ssign, chi_row_pair, phis, n)
        ret_phi = cphi
        ret_scratch = zphi
    end
    _factored_y_degree_major_auto!(rphi, ret_phi, y_loc_U, y_loc_V,
        thetas, aphi, yphi, ret_scratch, P_phi, n)
    _rotate_z_scatter_accumulate_n!(state.locals.phi, rphi, phi_flat_idx,
        target_idx, phi_row_m, phi_row_ssign, phi_row_pair, phis, n)
    return state
end

function _launch_resident_m2m!(state::DeviceResidentRadixState{TF,B,LH},
        strategy::AbstractResidentM2MStrategy=state.options.m2m_strategy) where {TF,B,LH}
    state.grid isa DeviceRadixGrid ||
        throw(ArgumentError("resident M2M requires DeviceRadixGrid-shaped node metadata"))
    ws = state.scratch
    ws isa ResidentOperatorWorkspace ||
        throw(ArgumentError("resident M2M requires ResidentOperatorWorkspace scratch"))
    strategy isa SharedRotationM2M ||
        throw(ArgumentError("resident workspace M2M currently supports SharedRotationM2M"))
    _zero_resident_nonleaf_multipoles!(state)
    # Groups run top-down (level ell-1 -> 0): each group reads finalized child
    # multipoles one level below the parents it accumulates into.
    for group in ws.m2m_groups
        _resident_stage_group_apply!(state.multipoles, state.multipoles, group, ws, :m2m)
    end
    return state
end

function _launch_resident_m2l!(state::DeviceResidentRadixState{TF,B,LH},
        strategy::AbstractResidentM2LStrategy=state.options.m2l_strategy) where {TF,B,LH}
    state.grid isa DeviceRadixGrid ||
        throw(ArgumentError("resident M2L requires DeviceRadixGrid-shaped node metadata"))
    ws = state.scratch
    ws isa ResidentOperatorWorkspace ||
        throw(ArgumentError("resident M2L requires ResidentOperatorWorkspace scratch"))
    state.interaction_list isa HostHierarchicalM2LContext &&
        return _launch_hierarchical_resident_m2l!(state,
            state.interaction_list::HostHierarchicalM2LContext)
    strategy isa DenseTranslationM2L && return _launch_resident_m2l_dense!(state)
    strategy isa PrecomputedFactoredYM2L &&
        return _launch_resident_m2l_precomputed_y!(state)
    state.options.operator isa FactoredRotationM2L &&
        return _launch_resident_m2l_factored!(state)
    state.options.operator isa MaterializedYRotationM2L ||
        throw(ArgumentError("unsupported resident M2L operator $(typeof(state.options.operator))"))
    strategy isa ConcatenatedFixedZM2L && return _launch_resident_m2l_concat!(state)
    strategy isa SharedRotationM2L ||
        throw(ArgumentError("resident workspace M2L supports SharedRotationM2L and ConcatenatedFixedZM2L"))
    return _launch_resident_m2l_shared!(state)
end

# On a hierarchical-policy state `state.route_*` holds only the last generated
# window while `counts.n_routes` is the whole-step total, so a flat launcher
# would silently compute a partial answer from stale window routes.
@inline function _assert_flat_resident_state(state::DeviceResidentRadixState)
    state.interaction_list isa HostHierarchicalM2LContext && throw(ArgumentError(
        "flat resident M2L launchers cannot run on a hierarchical-policy state; " *
        "call _launch_resident_m2l!(state), which dispatches to the windowed " *
        "hierarchical driver"))
    return nothing
end

function _launch_hierarchical_resident_m2l!(
        state::DeviceResidentRadixState{TF,B,LH},
        ctx::HostHierarchicalM2LContext) where {TF,B,LH}
    plan = ctx.apply_plan
    plan isa Union{ResidentM2LConcatPlan,ResidentM2LPrecomputedYPlan,
        ResidentM2LDensePlan} || throw(ArgumentError(
        "hierarchical host M2L has no compatible construction-time plan"))
    fill!(state.locals.phi, zero(TF))
    LH && fill!(state.locals.chi, zero(TF))
    noffsets = length(ctx.tables.push_offsets)
    total = 0
    fill!(ctx.routes_per_level, 0)
    ctx.profile_stages && fill!(ctx.m2l_level_ns, 0)
    for level in 2:state.grid.ell
        t_level = ctx.profile_stages ? time_ns() : UInt64(0)
        level_total = 0
        for first_offset in 1:ctx.window_classes:noffsets
            last_offset = min(first_offset + ctx.window_classes - 1, noffsets)
            count = build_hierarchical_routes_window!(
                state.route_levels, state.route_offsets, state.route_targets,
                state.route_sources, plan.route_class, ctx, state.grid, level,
                first_offset, last_offset)
            ctx.last_window_routes = count
            state.counts.n_routes = count
            if plan isa ResidentM2LDensePlan
                _refresh_dense_m2l_routes!(plan, state.route_sources,
                    state.route_targets, count)
                _launch_resident_m2l_dense_plan!(state, state.scratch, plan;
                    clear_locals=false)
            elseif plan isa ResidentM2LPrecomputedYPlan
                _refresh_precomputed_y_m2l_routes!(plan, state.route_sources,
                    state.route_targets, count)
                _launch_resident_m2l_precomputed_y_plan!(state,
                    state.scratch, plan; clear_locals=false)
            else
                _launch_hierarchical_concat_window!(state, state.scratch,
                    plan, count)
            end
            level_total += count
        end
        ctx.routes_per_level[level + 1] = level_total
        ctx.profile_stages &&
            (ctx.m2l_level_ns[level + 1] = time_ns() - t_level)
        total += level_total
    end
    ctx.total_routes = total
    state.counts.n_routes = total
    return state
end

function _gather_values_n!(dst, src, ids, n::Int)
    @inbounds for i in 1:n
        dst[i] = src[ids[i]]
    end
    return dst
end

function _prefix_trig_scale!(C, S, scale, nu, theta, invr, rexp, ncols::Int)
    @inbounds for j in 1:ncols, i in eachindex(nu)
        s, c = sincos(nu[i] * theta[j])
        C[i, j] = c
        S[i, j] = s
        scale[i, j] = invr[j]^rexp[i]
    end
    return scale
end

function _prefix_matmul!(Y, A, X, ncols::Int)
    @inbounds for j in 1:ncols, i in axes(A, 1)
        acc = zero(eltype(Y))
        for k in axes(A, 2)
            acc += A[i, k] * X[k, j]
        end
        Y[i, j] = acc
    end
    return Y
end

function _prefix_stacked_y!(out, input, Ur, Vs, C, S, G, G2,
        ndof::Int, ncols::Int)
    _prefix_matmul!(G, Vs, input, ncols)
    @inbounds for j in 1:ncols, i in 1:ndof
        gr = G[i, j]
        gi = G[ndof + i, j]
        G2[i, j] = C[i, j] * gr - S[i, j] * gi
        G2[ndof + i, j] = S[i, j] * gr + C[i, j] * gi
    end
    return _prefix_matmul!(out, Ur, G2, ncols)
end

function _prefix_scale!(A, scale, ncols::Int)
    @inbounds for j in 1:ncols, i in axes(A, 1)
        A[i, j] *= scale[i, j]
    end
    return A
end

function _prefix_lh_mix!(cphi, cchi, zphi, zchi, arow, brow, rs,
        phi_pair, chi_up, ncols::Int)
    @inbounds for j in 1:ncols
        r = rs[j]
        for i in axes(cphi, 1)
            cphi[i, j] = zphi[i, j] + arow[i] * r * zchi[phi_pair[i], j]
        end
        for i in axes(cchi, 1)
            cchi[i, j] = zchi[i, j] + brow[i] * r * zchi[chi_up[i], j]
        end
    end
    return cchi
end

@inline function _launch_hierarchical_concat_window!(
        state::DeviceResidentRadixState{TF,B,LH},
        ws::ResidentOperatorWorkspace{TF,B,LH},
        plan::ResidentM2LConcatPlan, n::Int) where {TF,B,LH}
    n == 0 && return state
    # Specialize this host-only window on ordinary Array storage.  The enclosing
    # containers are concrete and shared with CUDA, while these assertions also
    # guard against accidentally routing a device plan through the host launcher.
    route_class = plan.route_class::Vector{Int32}
    route_sources = state.route_sources::Vector{Int}
    route_targets = state.route_targets::Vector{Int}
    col_phi = plan.col_phi::Vector{TF}
    col_theta = plan.col_theta::Vector{TF}
    col_invr = plan.col_invr::Vector{TF}
    col_r = plan.col_r::Vector{TF}
    _gather_values_n!(col_phi, plan.phis::Vector{TF}, route_class, n)
    _gather_values_n!(col_theta, plan.thetas::Vector{TF}, route_class, n)
    _gather_values_n!(col_invr, plan.invrs::Vector{TF}, route_class, n)
    LH && _gather_values_n!(col_r, plan.rs::Vector{TF}, route_class, n)

    op = plan.ops_phi::ConcatChannelOps
    aphi = plan.aphi::Matrix{TF}; yphi = plan.yphi::Matrix{TF}
    zphi = plan.zphi::Matrix{TF}; rphi = plan.rphi::Matrix{TF}
    ndphi = size(aphi, 1)
    _prefix_trig_scale!(op.Cy::Matrix{TF}, op.Sy::Matrix{TF},
        op.scale::Matrix{TF}, op.nu::Vector{TF}, col_theta,
        col_invr, plan.rexp_phi::Vector{TF}, n)
    _gather_rotate_z_n!(aphi, state.multipoles.phi::Matrix{TF},
        ws.phi_flat_idx::Vector{Int}, route_sources,
        ws.maps_phi.row_m, ws.maps_phi.row_ssign,
        ws.maps_phi.row_pair, col_phi, false, n)
    _prefix_stacked_y!(yphi, aphi, op.yU_mult::Matrix{TF},
        op.yV_mult::Matrix{TF}, op.Cy::Matrix{TF}, op.Sy::Matrix{TF},
        op.G::Matrix{TF}, op.G2::Matrix{TF}, ndphi, n)
    _prefix_scale!(yphi, op.scale::Matrix{TF}, n)
    _prefix_matmul!(zphi, op.zD::Matrix{TF}, yphi, n)
    _prefix_scale!(zphi, op.scale::Matrix{TF}, n)
    retphi = zphi

    if LH
        oc = plan.ops_chi::ConcatChannelOps
        achi = plan.achi::Matrix{TF}; ychi = plan.ychi::Matrix{TF}
        zchi = plan.zchi::Matrix{TF}; rchi = plan.rchi::Matrix{TF}
        cphi = plan.cphi::Matrix{TF}; cchi = plan.cchi::Matrix{TF}
        ndchi = size(achi, 1)
        _prefix_trig_scale!(oc.Cy::Matrix{TF}, oc.Sy::Matrix{TF},
            oc.scale::Matrix{TF}, oc.nu::Vector{TF}, col_theta,
            col_invr, plan.rexp_chi::Vector{TF}, n)
        _gather_rotate_z_n!(achi, state.multipoles.chi::Matrix{TF},
            ws.chi_flat_idx::Vector{Int}, route_sources,
            ws.maps_chi.row_m, ws.maps_chi.row_ssign,
            ws.maps_chi.row_pair, col_phi, false, n)
        _prefix_stacked_y!(ychi, achi, oc.yU_mult::Matrix{TF},
            oc.yV_mult::Matrix{TF}, oc.Cy::Matrix{TF}, oc.Sy::Matrix{TF},
            oc.G::Matrix{TF}, oc.G2::Matrix{TF}, ndchi, n)
        _prefix_scale!(ychi, oc.scale::Matrix{TF}, n)
        _prefix_matmul!(zchi, oc.zD::Matrix{TF}, ychi, n)
        _prefix_scale!(zchi, oc.scale::Matrix{TF}, n)
        _prefix_lh_mix!(cphi, cchi, zphi, zchi,
            plan.lh_arow_unit::Vector{TF}, plan.lh_brow_unit::Vector{TF}, col_r,
            ws.maps_phi.row_pair, ws.maps_chi.row_up, n)
        _prefix_stacked_y!(rchi, cchi, oc.yU_loc::Matrix{TF},
            oc.yV_loc::Matrix{TF}, oc.Cy::Matrix{TF}, oc.Sy::Matrix{TF},
            oc.G::Matrix{TF}, oc.G2::Matrix{TF}, ndchi, n)
        _rotate_z_scatter_accumulate_n!(state.locals.chi::Matrix{TF}, rchi,
            ws.chi_flat_idx::Vector{Int}, route_targets, ws.maps_chi.row_m,
            ws.maps_chi.row_ssign, ws.maps_chi.row_pair, col_phi, n)
        retphi = cphi
    end
    _prefix_stacked_y!(rphi, retphi, op.yU_loc::Matrix{TF},
        op.yV_loc::Matrix{TF}, op.Cy::Matrix{TF}, op.Sy::Matrix{TF},
        op.G::Matrix{TF}, op.G2::Matrix{TF}, ndphi, n)
    _rotate_z_scatter_accumulate_n!(state.locals.phi::Matrix{TF}, rphi,
        ws.phi_flat_idx::Vector{Int}, route_targets, ws.maps_phi.row_m,
        ws.maps_phi.row_ssign, ws.maps_phi.row_pair, col_phi, n)
    return state
end

function _launch_resident_m2l_dense!(state::DeviceResidentRadixState{TF,B,LH}) where {TF,B,LH}
    _assert_flat_resident_state(state)
    state.options.operator isa MaterializedYRotationM2L || throw(ArgumentError(
        "DenseTranslationM2L requires operator=MaterializedYRotationM2L()"))
    ws = state.scratch::ResidentOperatorWorkspace{TF,B,LH}
    plan = ws.m2l_concat
    plan isa Union{ResidentM2LDensePlan{TF},ResidentM2LDenseCUDAPlan} || throw(ArgumentError(
        "DenseTranslationM2L requires a ResidentM2LDensePlan (host) or " *
        "ResidentM2LDenseCUDAPlan (device) workspace"))
    return _launch_resident_m2l_dense_plan!(state, ws, plan)
end

function _launch_resident_m2l_dense_plan!(
        state::DeviceResidentRadixState{TF,B,LH},
        ws::ResidentOperatorWorkspace{TF,B,LH},
        plan::ResidentM2LDensePlan{TF}; clear_locals::Bool=true) where {TF,B,LH}
    clear_locals && fill!(state.locals.phi, zero(TF))
    clear_locals && LH && fill!(state.locals.chi, zero(TF))
    phi_flat_idx = ws.phi_flat_idx::Vector{Int}
    chi_flat_idx = ws.chi_flat_idx::Vector{Int}
    Dphi = length(phi_flat_idx)
    @inbounds for cls in eachindex(plan.class_counts)
        count = plan.class_counts[cls]
        count == 0 && continue
        first = plan.class_starts[cls]
        done = 0
        while done < count
            ncols = min(plan.width, count - done)
            _dense_m2l_gather!(plan.src_slab, state.multipoles, phi_flat_idx,
                chi_flat_idx, plan.packed_sources, first + done, ncols, Dphi,
                Val(LH))
            if !isempty(plan.source_scale)
                _dense_scale_columns!(plan.src_slab,
                    @view(plan.source_scale[:, cls]), ncols)
            end
            _dense_m2l_mul_prefix!(plan.dst_slab,
                plan.operators[plan.class_operator[cls]],
                plan.src_slab, ncols)
            if !isempty(plan.target_scale)
                _dense_scale_columns!(plan.dst_slab,
                    @view(plan.target_scale[:, cls]), ncols)
            end
            _dense_m2l_scatter_add!(state.locals, plan.dst_slab, phi_flat_idx,
                chi_flat_idx, plan.packed_targets, first + done, ncols, Dphi,
                Val(LH))
            done += ncols
        end
    end
    return state
end

function _dense_scale_columns!(slab::Matrix{TF},
        scale::AbstractVector{TF}, ncols::Int) where TF
    @inbounds for j in 1:ncols, i in eachindex(scale)
        slab[i, j] *= scale[i]
    end
    return slab
end

@noinline function _dense_m2l_mul_prefix!(dst_slab::Matrix{TF}, K::Matrix{TF},
        src_slab::Matrix{TF}, ncols::Int) where TF
    src = @view src_slab[:, 1:ncols]
    dst = @view dst_slab[:, 1:ncols]
    mul!(dst, K, src)
    return nothing
end

function _dense_m2l_gather!(slab::Matrix{TF},
        source::FlatCoefficientBuffer{TF,Matrix{TF},B,LH},
        phi_flat_idx::Vector{Int}, chi_flat_idx::Vector{Int},
        packed_sources::Vector{Int}, first::Int, ncols::Int,
        Dphi::Int, ::Val{LH}) where {TF,B,LH}
    @inbounds for j in 1:ncols
        col = packed_sources[first + j - 1]
        for i in eachindex(phi_flat_idx)
            slab[i, j] = source.phi[phi_flat_idx[i], col]
        end
        if LH
            for i in eachindex(chi_flat_idx)
                slab[Dphi + i, j] = source.chi[chi_flat_idx[i], col]
            end
        end
    end
    return slab
end

function _dense_m2l_scatter_add!(
        target::FlatCoefficientBuffer{TF,Matrix{TF},B,LH}, slab::Matrix{TF},
        phi_flat_idx::Vector{Int}, chi_flat_idx::Vector{Int},
        packed_targets::Vector{Int}, first::Int, ncols::Int,
        Dphi::Int, ::Val{LH}) where {TF,B,LH}
    # Serial scalar += is intentional: multiple source routes may share a target.
    @inbounds for j in 1:ncols
        col = packed_targets[first + j - 1]
        for i in eachindex(phi_flat_idx)
            target.phi[phi_flat_idx[i], col] += slab[i, j]
        end
        if LH
            for i in eachindex(chi_flat_idx)
                target.chi[chi_flat_idx[i], col] += slab[Dphi + i, j]
            end
        end
    end
    return target
end

# Measured on an AMD EPYC 7763 with one- and 64-thread OpenBLAS (task 023c):
# narrow classes favor the direct dense scalar loop and wider classes favor one
# BLAS mul! per real degree block.  A 16-column global crossover stayed within 5%
# of the best focused candidate in every measured P/N/LH/thread regime, so no
# additional degree-dimension gate is warranted.  The Ref remains mutable for
# reproducible benchmark sweeps and focused branch tests.
const PRECOMPUTED_Y_GEMM_MIN_COLS = Ref(16)

function _precomputed_y_degree_major!(out, input, blocks, P::Int, ncols::Int)
    use_gemm = ncols >= PRECOMPUTED_Y_GEMM_MIN_COLS[]
    @inbounds for n in 0:P
        rows = degree_row_range(n)
        M = blocks[n + 1]
        X = @view input[rows, 1:ncols]
        Y = @view out[rows, 1:ncols]
        if use_gemm
            mul!(Y, M, X)
        else
            d = 2n + 1
            for j in 1:ncols, i in 1:d
                acc = zero(eltype(out))
                for k in 1:d
                    acc += M[i, k] * X[k, j]
                end
                Y[i, j] = acc
            end
        end
    end
    return out
end

function _precomputed_gather_rotate_z!(dst, src, flat_idx, packed_cols, packed_phis,
        first_col::Int, row_m, row_ssign, row_pair, ncols::Int)
    @inbounds for j in 1:ncols
        q = first_col + j - 1
        csrc = packed_cols[q]
        phi = packed_phis[q]
        for i in eachindex(flat_idx)
            s, c = sincos(row_m[i] * phi)
            a = src[flat_idx[i], csrc]
            b = src[flat_idx[row_pair[i]], csrc]
            dst[i, j] = c * a + row_ssign[i] * s * b
        end
    end
    return dst
end

function _precomputed_rotate_z_scatter!(dest, slab, flat_idx, packed_targets,
        packed_phis, first_col::Int, row_m, row_ssign, row_pair, ncols::Int)
    @inbounds for j in 1:ncols
        q = first_col + j - 1
        target = packed_targets[q]
        phi = packed_phis[q]
        for i in eachindex(flat_idx)
            s, c = sincos(row_m[i] * phi)
            dest[flat_idx[i], target] += c * slab[i, j] -
                row_ssign[i] * s * slab[row_pair[i], j]
        end
    end
    return dest
end

function _precomputed_ztranslate_range!(out, input, block_mats,
        maps::DegreeMajorMaps, P::Int, col0::Int, ncols::Int)
    @inbounds for j in col0:(col0 + ncols - 1), i in axes(out, 1)
        out[i, j] = zero(eltype(out))
    end
    @inbounds for m in 0:P
        Bm = block_mats[m + 1]
        re_rows = maps.z_re_rows[m + 1]
        im_rows = maps.z_im_rows[m + 1]
        for j in col0:(col0 + ncols - 1), (irow, row) in enumerate(re_rows)
            sr = zero(eltype(out)); si = zero(eltype(out))
            for (icol, col) in enumerate(re_rows)
                b = Bm[irow, icol]
                sr += b * input[col, j]
                m > 0 && (si += b * input[im_rows[icol], j])
            end
            out[row, j] = sr
            m > 0 && (out[im_rows[irow], j] = si)
        end
    end
    return out
end

function _precomputed_lh_range!(out_phi, out_chi, in_phi, in_chi, arow, brow,
        maps_phi::DegreeMajorMaps, maps_chi::DegreeMajorMaps, col0::Int, ncols::Int)
    @inbounds for j in col0:(col0 + ncols - 1), i in axes(out_phi, 1)
        out_phi[i, j] = in_phi[i, j] + arow[i] * in_chi[maps_phi.row_pair[i], j]
    end
    @inbounds for j in col0:(col0 + ncols - 1), i in axes(out_chi, 1)
        out_chi[i, j] = in_chi[i, j] + brow[i] * in_chi[maps_chi.row_up[i], j]
    end
    return out_chi
end

function _launch_resident_m2l_precomputed_y!(state::DeviceResidentRadixState{TF,B,LH}) where {TF,B,LH}
    _assert_flat_resident_state(state)
    state.options.operator isa FactoredRotationM2L ||
        throw(ArgumentError("PrecomputedFactoredYM2L requires operator=FactoredRotationM2L()"))
    ws = state.scratch
    plan = ws.m2l_concat
    plan isa ResidentM2LPrecomputedYPlan ||
        throw(ArgumentError("PrecomputedFactoredYM2L requires a precomputed-y resident plan"))
    return _launch_resident_m2l_precomputed_y_plan!(state, ws, plan)
end

function _launch_resident_m2l_precomputed_y_plan!(
        state::DeviceResidentRadixState{TF,B,LH},
        ws::ResidentOperatorWorkspace{TF,B,LH},
        plan::ResidentM2LPrecomputedYPlan{TF,S};
        clear_locals::Bool=true) where {TF,B,LH,S}
    clear_locals && fill!(state.locals.phi, zero(TF))
    clear_locals && LH && fill!(state.locals.chi, zero(TF))
    # These host-only assertions keep the per-angle stage calls specialized on
    # Array storage and guard against accidentally routing a device plan here.
    locals_phi = state.locals.phi::Matrix{TF}
    locals_chi = state.locals.chi::Matrix{TF}
    mult_phi = state.multipoles.phi::Matrix{TF}
    mult_chi = state.multipoles.chi::Matrix{TF}
    phi_flat_idx = ws.phi_flat_idx::Vector{Int}
    chi_flat_idx = ws.chi_flat_idx::Vector{Int}
    maps_phi = ws.maps_phi::DegreeMajorMaps{Vector{TF},Vector{Int}}
    maps_chi = ws.maps_chi::DegreeMajorMaps{Vector{TF},Vector{Int}}
    P_phi = ws.basis_info.orders.P_phi
    P_active = ws.basis_info.orders.P_active
    sc = plan.scratch
    @inbounds for angle in eachindex(plan.angle_counts)
        ncols = plan.angle_counts[angle]
        ncols == 0 && continue
        first_col = plan.angle_starts[angle]
        _precomputed_gather_rotate_z!(sc.aphi, mult_phi, phi_flat_idx,
            plan.packed_sources, plan.packed_phis, first_col, maps_phi.row_m,
            maps_phi.row_ssign, maps_phi.row_pair, ncols)
        _precomputed_y_degree_major!(sc.yphi, sc.aphi, plan.y_mult[angle], P_phi, ncols)
        if LH
            _precomputed_gather_rotate_z!(sc.achi, mult_chi, chi_flat_idx,
                plan.packed_sources, plan.packed_phis, first_col, maps_chi.row_m,
                maps_chi.row_ssign, maps_chi.row_pair, ncols)
            _precomputed_y_degree_major!(sc.ychi, sc.achi, plan.y_mult[angle], P_active, ncols)
        end

        # Fixed-m translation and LH coupling remain offset-specific.  The packed
        # ranges are contiguous and stable inside this angle class.
        for p in plan.angle_offset_starts[angle]:(plan.angle_offset_starts[angle + 1] - 1)
            offset = plan.angle_offsets[p]
            count = plan.offset_counts[offset]
            count == 0 && continue
            local_first = plan.offset_starts[offset] - first_col + 1
            _precomputed_ztranslate_range!(sc.zphi, sc.yphi, plan.z_phi[offset],
                maps_phi, P_phi, local_first, count)
            if LH
                _precomputed_ztranslate_range!(sc.zchi, sc.ychi, plan.z_chi[offset],
                    maps_chi, P_active, local_first, count)
                _precomputed_lh_range!(sc.cphi, sc.cchi, sc.zphi, sc.zchi,
                    plan.lh_phi_rows[offset], plan.lh_chi_rows[offset],
                    maps_phi, maps_chi, local_first, count)
            end
        end
        retphi = LH ? sc.cphi : sc.zphi
        _precomputed_y_degree_major!(sc.rphi, retphi, plan.y_loc[angle], P_phi, ncols)
        _precomputed_rotate_z_scatter!(locals_phi, sc.rphi, phi_flat_idx,
            plan.packed_targets, plan.packed_phis, first_col, maps_phi.row_m,
            maps_phi.row_ssign, maps_phi.row_pair, ncols)
        if LH
            _precomputed_y_degree_major!(sc.rchi, sc.cchi, plan.y_loc[angle], P_active, ncols)
            _precomputed_rotate_z_scatter!(locals_chi, sc.rchi, chi_flat_idx,
                plan.packed_targets, plan.packed_phis, first_col, maps_chi.row_m,
                maps_chi.row_ssign, maps_chi.row_pair, ncols)
        end
    end
    return state
end

# Group-looping shared-rotation M2L through the generic (allocating) degree-major
# helpers. Production path for SharedRotationM2L (over `ws.m2l_groups`); also the
# functional-baseline reference the 023a benchmark runs over a factored plan's
# capacity groups, which carry the same per-group fields.
function _launch_resident_m2l_shared!(state::DeviceResidentRadixState{TF,B,LH},
        groups=nothing) where {TF,B,LH}
    _assert_flat_resident_state(state)
    ws = state.scratch
    ws isa ResidentOperatorWorkspace ||
        throw(ArgumentError("resident M2L requires ResidentOperatorWorkspace scratch"))
    m2l_groups = groups === nothing ? ws.m2l_groups : groups
    fill!(state.locals.phi, zero(TF))
    LH && fill!(state.locals.chi, zero(TF))
    for group in m2l_groups
        nbatch = group.count[]
        nbatch == 0 && continue
        source_idx = _vector_prefix_view(group.source_idx, nbatch)
        target_idx = _vector_prefix_view(group.target_idx, nbatch)
        src_dm = _degree_major_buffer_view(ws.m2l_sources, nbatch)
        tgt_dm = _degree_major_buffer_view(ws.m2l_targets, nbatch)
        fill!(tgt_dm.phi, zero(TF)); LH && fill!(tgt_dm.chi, zero(TF))
        src_dm.phi .= state.multipoles.phi[ws.phi_flat_idx, source_idx]
        LH && (src_dm.chi .= state.multipoles.chi[ws.chi_flat_idx, source_idx])
        _resident_execute_shared_m2l!(tgt_dm, src_dm, group, ws)
        state.locals.phi[ws.phi_flat_idx, target_idx] =
            state.locals.phi[ws.phi_flat_idx, target_idx] .+ tgt_dm.phi
        if LH
            state.locals.chi[ws.chi_flat_idx, target_idx] =
                state.locals.chi[ws.chi_flat_idx, target_idx] .+ tgt_dm.chi
        end
    end
    return state
end

function _launch_resident_m2l_factored!(state::DeviceResidentRadixState{TF,B,LH}) where {TF,B,LH}
    _assert_flat_resident_state(state)
    ws = state.scratch
    plan = ws.m2l_concat
    plan isa ResidentM2LFactoredPlan ||
        throw(ArgumentError("FactoredRotationM2L requires a factored resident M2L plan"))
    return _launch_resident_m2l_factored_plan!(state, ws, plan)
end

function _launch_resident_m2l_factored_plan!(state::DeviceResidentRadixState{TF,B,LH},
        ws::ResidentOperatorWorkspace{TF,B,LH},
        plan::ResidentM2LFactoredPlan{R,G}) where {TF,B,LH,R,G}
    fill!(state.locals.phi, zero(TF))
    LH && fill!(state.locals.chi, zero(TF))
    for group in plan.groups
        _resident_factored_m2l_group_apply!(state, group, ws)
    end
    return state
end

"""
    _launch_resident_l2l!(state)

Resident top-down L2L. Translation vectors are `child - parent`, matching
`_launch_host_l2l_flat_oracle!` and opposite the M2M upward `parent - child` vector.
The resident path ignores `options.operator`; that option belongs to the retained
flat/oracle launchers.
"""
function _launch_resident_l2l!(state::DeviceResidentRadixState{TF,B,LH}) where {TF,B,LH}
    state.grid isa DeviceRadixGrid ||
        throw(ArgumentError("resident L2L requires DeviceRadixGrid-shaped node metadata"))
    ws = state.scratch
    ws isa ResidentOperatorWorkspace ||
        throw(ArgumentError("resident L2L requires ResidentOperatorWorkspace scratch"))
    # Groups run top-down (level 1 -> ell): each group reads finalized parent locals
    # one level above the children it accumulates into.
    for group in ws.l2l_groups
        _resident_stage_group_apply!(state.locals, state.locals, group, ws, :l2l)
    end
    return state
end

# Accumulate a degree-major slab into flat-layout target columns:
# dest[row_idx[i], col_targets[j]] += slab[i, j]. Targets may repeat, so this must
# accumulate; the CUDA specialization (translate_batched_cuda.jl) uses an atomic
# scatter kernel.
function _scatter_accumulate_columns!(dest::AbstractMatrix, row_idx, col_targets, slab)
    @inbounds for (j, tgt) in enumerate(col_targets)
        for (i, row) in enumerate(row_idx)
            dest[row, tgt] += slab[i, j]
        end
    end
    return dest
end

# Fused flat-to-degree-major column gather + z rotation for the concatenated M2L
# chain: dst[i, j] = c*a + sgn*ssign[i]*s*b with (s, c) = sincos(row_m[i]*phis[j]),
# a = src[flat_idx[i], cols[j]], b its (re,im) pair row. Replaces the allocating
# fancy-index gather plus _zphi_degree_major_gen! (and its temporaries); the CUDA
# specialization (translate_batched_cuda.jl) is a single kernel.
function _gather_rotate_z!(dst::AbstractMatrix, src::AbstractMatrix, flat_idx, cols,
        row_m, row_ssign, row_pair, phis, inverse::Bool)
    TF = eltype(dst)
    sgn = inverse ? -one(TF) : one(TF)
    @inbounds for j in eachindex(cols)
        csrc = cols[j]
        phi = phis[j]
        for i in eachindex(flat_idx)
            s, c = sincos(row_m[i] * phi)
            a = src[flat_idx[i], csrc]
            b = src[flat_idx[row_pair[i]], csrc]
            dst[i, j] = c * a + sgn * row_ssign[i] * s * b
        end
    end
    return dst
end

function _gather_rotate_z_n!(dst::AbstractMatrix, src::AbstractMatrix, flat_idx,
        cols, row_m, row_ssign, row_pair, phis, inverse::Bool, ncols::Int)
    TF = eltype(dst)
    sgn = inverse ? -one(TF) : one(TF)
    @inbounds for j in 1:ncols
        csrc = cols[j]
        phi = phis[j]
        for i in eachindex(flat_idx)
            s, c = sincos(row_m[i] * phi)
            a = src[flat_idx[i], csrc]
            b = src[flat_idx[row_pair[i]], csrc]
            dst[i, j] = c * a + sgn * row_ssign[i] * s * b
        end
    end
    return dst
end

# Fused inverse z rotation + accumulating scatter of a degree-major slab into flat
# target columns: dest[flat_idx[i], col_targets[j]] += Z_phi^{-1}(slab)[i, j]. Targets
# may repeat, so this must accumulate; the CUDA specialization uses an atomic kernel.
function _rotate_z_scatter_accumulate!(dest::AbstractMatrix, slab, flat_idx, col_targets,
        row_m, row_ssign, row_pair, phis)
    @inbounds for j in eachindex(col_targets)
        tgt = col_targets[j]
        phi = phis[j]
        for i in eachindex(flat_idx)
            s, c = sincos(row_m[i] * phi)
            dest[flat_idx[i], tgt] += c * slab[i, j] - row_ssign[i] * s * slab[row_pair[i], j]
        end
    end
    return dest
end

function _rotate_z_scatter_accumulate_n!(dest::AbstractMatrix, slab, flat_idx,
        col_targets, row_m, row_ssign, row_pair, phis, ncols::Int)
    @inbounds for j in 1:ncols
        tgt = col_targets[j]
        phi = phis[j]
        for i in eachindex(flat_idx)
            s, c = sincos(row_m[i] * phi)
            dest[flat_idx[i], tgt] += c * slab[i, j] -
                row_ssign[i] * s * slab[row_pair[i], j]
        end
    end
    return dest
end

# Allocation-free row gather dst[i, :] = src[rows[i], :] (the LH row-mix operand
# gathers); the CUDA specialization is a single kernel.
function _gather_rows!(dst::AbstractMatrix, src::AbstractMatrix, rows)
    @inbounds for j in axes(dst, 2)
        for i in eachindex(rows)
            dst[i, j] = src[rows[i], j]
        end
    end
    return dst
end

# Allocation-free value gather dst[i] = src[ids[i]] (per-chunk column parameters from
# the per-class geometry tables); the CUDA specialization is a single kernel.
function _gather_values!(dst::AbstractVector, src::AbstractVector, ids)
    @inbounds for i in eachindex(dst)
        dst[i] = src[ids[i]]
    end
    return dst
end

"""
    _launch_resident_m2l_concat!(state)

Whole-pass resident M2L for [`ConcatenatedFixedZM2L`](@ref): iterates flattened
routes in `chunk`-column slabs, running the per-column-parameterized stage chain
`gather -> Z_phi -> Y(theta) -> scale -> fixed z GEMMs -> scale -> [LH rows] ->
Y_loc(theta) -> Z_phi^-1 -> scatter-accumulate` from the
[`ResidentM2LConcatPlan`](@ref). Matches the SharedRotationM2L group path up to
floating-point reassociation of the z-translation (the separable scaling form).
"""
function _launch_resident_m2l_concat!(state::DeviceResidentRadixState{TF,B,LH};
        clear_locals::Bool=true) where {TF,B,LH}
    clear_locals && _assert_flat_resident_state(state)
    ws = state.scratch
    plan = ws.m2l_concat
    plan isa ResidentM2LConcatPlan ||
        throw(ArgumentError("ConcatenatedFixedZM2L requires a workspace built with m2l_strategy=ConcatenatedFixedZM2L"))
    clear_locals && fill!(state.locals.phi, zero(TF))
    clear_locals && LH && fill!(state.locals.chi, zero(TF))
    nroutes = min(state.counts.n_routes, plan.nroutes)
    nroutes == 0 && return state
    @inbounds for c0 in 1:plan.chunk:nroutes
        cols = c0:min(c0 + plan.chunk - 1, nroutes)
        n = length(cols)
        cls = @view plan.route_class[cols]
        phis = @view plan.col_phi[1:n]
        thetas = @view plan.col_theta[1:n]
        invr_col = @view plan.col_invr[1:n]
        _gather_values!(phis, plan.phis, cls)
        _gather_values!(thetas, plan.thetas, cls)
        _gather_values!(invr_col, plan.invrs, cls)
        invr_row = transpose(invr_col)
        if LH
            rs_col = @view plan.col_r[1:n]
            _gather_values!(rs_col, plan.rs, cls)
            rs_row = transpose(rs_col)
        end
        src_cols = @view state.route_sources[cols]
        tgt_cols = @view state.route_targets[cols]
        aphi = _matrix_col_view(plan.aphi, n)
        yphi = _matrix_col_view(plan.yphi, n)
        zphi = _matrix_col_view(plan.zphi, n)
        rphi = _matrix_col_view(plan.rphi, n)
        ops_phi = plan.ops_phi
        ndof_phi = size(plan.aphi, 1)
        Gphi = _matrix_col_view(ops_phi.G, n)
        G2phi = _matrix_col_view(ops_phi.G2, n)
        Cphi = _matrix_col_view(ops_phi.Cy, n)
        Sphi = _matrix_col_view(ops_phi.Sy, n)
        sphi = _matrix_col_view(ops_phi.scale, n)
        thetas_row = transpose(thetas)
        Cphi .= cos.(ops_phi.nu .* thetas_row)
        Sphi .= sin.(ops_phi.nu .* thetas_row)
        sphi .= invr_row .^ plan.rexp_phi
        _gather_rotate_z!(aphi, state.multipoles.phi, ws.phi_flat_idx, src_cols,
            ws.maps_phi.row_m, ws.maps_phi.row_ssign, ws.maps_phi.row_pair, phis, false)
        _stacked_y_dense!(yphi, aphi, ops_phi.yU_mult, ops_phi.yV_mult,
            Cphi, Sphi, Gphi, G2phi, ndof_phi)
        yphi .*= sphi
        _resident_mul!(zphi, ops_phi.zD, yphi)
        zphi .*= sphi
        ret_phi = zphi
        if LH
            ops_chi = plan.ops_chi
            ndof_chi = size(plan.achi, 1)
            Gchi = _matrix_col_view(ops_chi.G, n)
            G2chi = _matrix_col_view(ops_chi.G2, n)
            Cchi = _matrix_col_view(ops_chi.Cy, n)
            Schi = _matrix_col_view(ops_chi.Sy, n)
            schi = _matrix_col_view(ops_chi.scale, n)
            achi = _matrix_col_view(plan.achi, n)
            ychi = _matrix_col_view(plan.ychi, n)
            zchi = _matrix_col_view(plan.zchi, n)
            rchi = _matrix_col_view(plan.rchi, n)
            cphi = _matrix_col_view(plan.cphi, n)
            cchi = _matrix_col_view(plan.cchi, n)
            lhgp = _matrix_col_view(plan.lhgp, n)
            lhgu = _matrix_col_view(plan.lhgu, n)
            Cchi .= cos.(ops_chi.nu .* thetas_row)
            Schi .= sin.(ops_chi.nu .* thetas_row)
            schi .= invr_row .^ plan.rexp_chi
            _gather_rotate_z!(achi, state.multipoles.chi, ws.chi_flat_idx, src_cols,
                ws.maps_chi.row_m, ws.maps_chi.row_ssign, ws.maps_chi.row_pair, phis, false)
            _stacked_y_dense!(ychi, achi, ops_chi.yU_mult, ops_chi.yV_mult,
                Cchi, Schi, Gchi, G2chi, ndof_chi)
            ychi .*= schi
            _resident_mul!(zchi, ops_chi.zD, ychi)
            zchi .*= schi
            # LH local rows are linear in r (lamb_helmholtz_local_coeffs!), so the
            # unit-radius rows scale per column.
            _gather_rows!(lhgp, zchi, ws.maps_phi.row_pair)
            _gather_rows!(lhgu, zchi, ws.maps_chi.row_up)
            cphi .= zphi .+ (plan.lh_arow_unit .* rs_row) .* lhgp
            cchi .= zchi .+ (plan.lh_brow_unit .* rs_row) .* lhgu
            _stacked_y_dense!(rchi, cchi, ops_chi.yU_loc, ops_chi.yV_loc,
                Cchi, Schi, Gchi, G2chi, ndof_chi)
            _rotate_z_scatter_accumulate!(state.locals.chi, rchi, ws.chi_flat_idx, tgt_cols,
                ws.maps_chi.row_m, ws.maps_chi.row_ssign, ws.maps_chi.row_pair, phis)
            ret_phi = cphi
        end
        _stacked_y_dense!(rphi, ret_phi, ops_phi.yU_loc, ops_phi.yV_loc,
            Cphi, Sphi, Gphi, G2phi, ndof_phi)
        _rotate_z_scatter_accumulate!(state.locals.phi, rphi, ws.phi_flat_idx, tgt_cols,
            ws.maps_phi.row_m, ws.maps_phi.row_ssign, ws.maps_phi.row_pair, phis)
    end
    return state
end

#------- fixed-box cache operator workspace (Matrix Operator Refactor, task 023) -------#
#
# The RadixFMMCache path builds the resident workspace once at construction from the
# fixed Morton domain (x_min, h0, ell) instead of from a concrete grid/list:
#  - M2M/L2L: on the uniform grid every parent<->child displacement at a child level
#    Lc has the same radius sqrt(3) * h0 / 2^Lc and one of 8 diagonal directions, so
#    exactly one group per level carries an invariant dense z-operator while its
#    edge columns (node indices + per-edge theta/phi) are refreshed each step.
#  - M2L (ConcatenatedFixedZM2L only): one geometry class per accepted stencil
#    offset (displacement = offset * leaf cell width), invariant across steps;
#    route_class is capacity-sized and refilled by build_radix_routes!.

# One capacity-sized stage group: dense operators built for the level's fixed
# radius, index/angle columns allocated at capacity with count[] = 0.
function _resident_capacity_group(exemplar, ::Type{TF}, basis_info, kind::Symbol,
        level::Integer, r::TF, capacity::Integer) where TF
    capacity >= 1 ||
        throw(ArgumentError("resident capacity group requires capacity >= 1"))
    group = _resident_group(exemplar, TF, basis_info, kind, level,
        ones(Int, capacity), ones(Int, capacity), zeros(TF, capacity),
        zeros(TF, capacity), fill(r, capacity))
    group.count[] = 0
    return group
end

# Nodes at level L are bounded by both the full octree width and the occupied
# leaf count (each occupied leaf contributes at most one ancestor per level).
function _radix_level_node_capacity(level::Integer, max_cells::Integer)
    3 * level >= 62 && return max_cells
    return min(1 << (3 * level), max_cells)
end

# Capacity ResidentM2LConcatPlan from the fixed accepted-offset classes (task 023).
function ResidentM2LConcatPlan(::Type{TF}, basis_info::OperatorBasisInfo{B,LH}, exemplar,
        strategy::ConcatenatedFixedZM2L, invariant::OperatorInvariantCache,
        accepted_offsets::AbstractVector{SVector{3,Int}}, cell_width::Real,
        route_capacity::Integer; whole_window::Bool=false) where {TF,B,LH}
    P_phi = basis_info.orders.P_phi
    P_active = basis_info.orders.P_active
    nclasses = length(accepted_offsets)
    phis = Vector{TF}(undef, nclasses)
    thetas = Vector{TF}(undef, nclasses)
    rs = Vector{TF}(undef, nclasses)
    @inbounds for (k, offset) in enumerate(accepted_offsets)
        d = SVector{3,TF}(TF(offset[1]), TF(offset[2]), TF(offset[3])) * TF(cell_width)
        r, theta, phi = cartesian_to_spherical(d)
        rs[k] = TF(r)
        thetas[k] = TF(theta)
        phis[k] = TF(phi)
    end
    nroutes = Int(route_capacity)
    # Flat routes are applied in `strategy.chunk` pieces. Hierarchical routes
    # are generated and applied one complete class window at a time, so their
    # scratch must cover the full window capacity. Using the flat chunk here
    # silently wrote past the stage slabs once a window held more routes than
    # `strategy.chunk`.
    chunk = whole_window ? max(nroutes, 1) :
        max(min(strategy.chunk, max(nroutes, 1)), 1)
    ndof_phi = degree_major_dof(P_phi)
    ndof_chi = LH ? degree_major_dof(P_active) : 0
    lh_arow_unit, lh_brow_unit = LH ?
        _resident_lh_rows_like(exemplar, TF, P_phi, P_active, one(TF), :local) :
        (nothing, nothing)
    mkphi() = similar(exemplar, TF, ndof_phi, chunk)
    mkchi() = similar(exemplar, TF, ndof_chi, LH ? chunk : 0)
    return ResidentM2LConcatPlan(
        nroutes, chunk,
        _array_like_vector(exemplar, TF, phis),
        _array_like_vector(exemplar, TF, thetas),
        _array_like_vector(exemplar, TF, rs),
        _array_like_vector(exemplar, TF, inv.(rs)),
        _array_like_vector(exemplar, Int32, Vector{Int32}(undef, nroutes)),
        similar(exemplar, TF, chunk),
        similar(exemplar, TF, chunk),
        similar(exemplar, TF, chunk),
        similar(exemplar, TF, chunk),
        _degree_row_exponents(exemplar, TF, P_phi),
        LH ? _degree_row_exponents(exemplar, TF, P_active) : nothing,
        ConcatChannelOps(exemplar, TF, invariant, P_phi, chunk),
        LH ? ConcatChannelOps(exemplar, TF, invariant, P_active, chunk) : nothing,
        lh_arow_unit, lh_brow_unit,
        mkphi(), mkphi(), mkphi(), mkphi(),
        mkchi(), mkchi(), mkchi(), mkchi(),
        LH ? mkphi() : similar(exemplar, TF, 0, 0), mkchi(),
        LH ? mkphi() : nothing, LH ? mkchi() : nothing,
    )
end

function ResidentM2LFactoredPlan(::Type{TF}, basis_info::OperatorBasisInfo,
        exemplar, accepted_offsets::AbstractVector{SVector{3,Int}}, cell_width::Real,
        route_capacity::Integer, max_cells::Integer,
        invariant::OperatorInvariantCache; build_groups::Bool=true) where TF
    P_active = basis_info.orders.P_active
    nclasses = length(accepted_offsets)
    groups = build_groups ? Vector{ResidentOperatorGroup}(undef, nclasses) : ResidentOperatorGroup[]
    class_theta = Vector{TF}(undef, nclasses)
    class_phi = Vector{TF}(undef, nclasses)
    class_r = Vector{TF}(undef, nclasses)
    zlen = m2l_z_block_length(P_active)
    z_flat_host = Matrix{TF}(undef, zlen, nclasses)
    @inbounds for (k, offset) in enumerate(accepted_offsets)
        d = SVector{3,TF}(TF(offset[1]), TF(offset[2]), TF(offset[3])) * TF(cell_width)
        r, theta, phi = cartesian_to_spherical(d)
        if build_groups
            groups[k] = _resident_group(exemplar, TF, basis_info, :m2l, 0,
                ones(Int, max_cells), ones(Int, max_cells), fill(TF(phi), max_cells),
                fill(TF(theta), max_cells), fill(TF(r), max_cells))
            groups[k].count[] = 0
        end
        class_theta[k] = TF(theta)
        class_phi[k] = TF(phi)
        class_r[k] = TF(r)
        m2l_z_blocks!(view(z_flat_host, :, k), TF(r), P_active)
    end
    # Flat Plain-H mode blocks and per-class z tables for the fused device kernels
    # (task 023b); host paths carry them too (construction-time only, small).
    ym_flat = (
        mult_U_re = _array_like_vector(exemplar, TF, TF.(real.(invariant.y_mult_U))),
        mult_U_im = _array_like_vector(exemplar, TF, TF.(imag.(invariant.y_mult_U))),
        mult_V_re = _array_like_vector(exemplar, TF, TF.(real.(invariant.y_mult_V))),
        mult_V_im = _array_like_vector(exemplar, TF, TF.(imag.(invariant.y_mult_V))),
        loc_U_re = _array_like_vector(exemplar, TF, TF.(real.(invariant.y_loc_U))),
        loc_U_im = _array_like_vector(exemplar, TF, TF.(imag.(invariant.y_loc_U))),
        loc_V_re = _array_like_vector(exemplar, TF, TF.(real.(invariant.y_loc_V))),
        loc_V_im = _array_like_vector(exemplar, TF, TF.(imag.(invariant.y_loc_V))),
    )
    return ResidentM2LFactoredPlan(
        _array_like_vector(exemplar, Int32, Vector{Int32}(undef, route_capacity)),
        _homogeneous_groups(groups),
        _array_like_vector(exemplar, Int32, zeros(Int32, nclasses)),
        zeros(Int32, nclasses),
        zeros(Int, nclasses + 1),
        class_theta, class_phi, class_r, ym_flat,
        _array_like_matrix(exemplar, TF, z_flat_host),
        Ref{Any}(nothing),
    )
end

# Exact integer polar-angle key.  Squared axial and equatorial lengths are reduced
# by their gcd; the sign of z distinguishes the two hemispheres.  Consequently
# collinear integer offsets share a key without ever comparing computed θ values.
function _precomputed_y_angle_key(offset::SVector{3,<:Integer})
    x, y, z = Int(offset[1]), Int(offset[2]), Int(offset[3])
    z2 = z * z
    rho2 = x * x + y * y
    g = gcd(z2, rho2)
    g == 0 && throw(ArgumentError("the zero offset has no polar angle"))
    return (sign(z), z2 ÷ g, rho2 ÷ g)
end

function _precomputed_y_angle_metadata(offsets::AbstractVector{<:SVector{3,<:Integer}})
    keys = NTuple{3,Int}[]
    key_to_angle = Dict{NTuple{3,Int},Int}()
    offset_to_angle = Vector{Int}(undef, length(offsets))
    @inbounds for (i, offset) in enumerate(offsets)
        key = _precomputed_y_angle_key(offset)
        angle = get(key_to_angle, key, 0)
        if angle == 0
            push!(keys, key)
            angle = length(keys)
            key_to_angle[key] = angle
        end
        offset_to_angle[i] = angle
    end
    # Stable angle-major / original-offset-minor ordering.
    angle_offsets = Int[]
    angle_offset_starts = Vector{Int}(undef, length(keys) + 1)
    for angle in eachindex(keys)
        angle_offset_starts[angle] = length(angle_offsets) + 1
        for i in eachindex(offsets)
            offset_to_angle[i] == angle && push!(angle_offsets, i)
        end
    end
    angle_offset_starts[end] = length(angle_offsets) + 1
    return keys, offset_to_angle, angle_offset_starts, angle_offsets
end

function _precomputed_y_block(::Type{TF}, Ublocks, Vblocks, theta::TF, n::Int) where TF
    Ure, Uim = Ublocks[n + 1]
    Vre, Vim = Vblocks[n + 1]
    U = Complex{TF}.(Ure, Uim)
    V = Complex{TF}.(Vre, Vim)
    phases = Complex{TF}[cis(TF(nu) * theta) for nu in -n:n]
    return Matrix{TF}(real.(U * Diagonal(phases) * V))
end

function ResidentM2LPrecomputedYPlan(::Type{TF}, basis_info::OperatorBasisInfo{B,LH},
        exemplar, accepted_offsets::AbstractVector{SVector{3,Int}}, cell_width::Real,
        route_capacity::Integer, max_cells::Integer, grid_resolution::Integer,
        invariant::OperatorInvariantCache; compact_device::Bool=false) where {TF,B,LH}
    P_phi = basis_info.orders.P_phi
    P_active = basis_info.orders.P_active
    keys, offset_to_angle, angle_offset_starts, angle_offsets =
        _precomputed_y_angle_metadata(accepted_offsets)
    nangles = length(keys)
    noffsets = length(accepted_offsets)
    angle_thetas = Vector{TF}(undef, nangles)
    offset_phis = Vector{TF}(undef, noffsets)
    offset_rs = Vector{TF}(undef, noffsets)
    @inbounds for (i, offset) in enumerate(accepted_offsets)
        d = SVector{3,TF}(offset) * TF(cell_width)
        r, theta, phi = cartesian_to_spherical(d)
        offset_rs[i] = r
        offset_phis[i] = phi
    end
    # Correct the deliberately simple first-member fill above without floating
    # comparisons (and without depending on Dict iteration order).
    @inbounds for angle in 1:nangles
        first_offset = angle_offsets[angle_offset_starts[angle]]
        d = SVector{3,TF}(accepted_offsets[first_offset]) * TF(cell_width)
        angle_thetas[angle] = cartesian_to_spherical(d)[2]
    end

    # Host mode blocks in every case: _precomputed_y_block assembles the dense
    # M_n(theta) products on the host even when the plan's exemplar is a device
    # array (the compact path uploads only the finished flat tables).
    Um = _ymode_real_blocks(Vector{Complex{TF}}(invariant.y_mult_U), P_active, TF)
    Vm = _ymode_real_blocks(Vector{Complex{TF}}(invariant.y_mult_V), P_active, TF)
    Ul = _ymode_real_blocks(Vector{Complex{TF}}(invariant.y_loc_U), P_active, TF)
    Vl = _ymode_real_blocks(Vector{Complex{TF}}(invariant.y_loc_V), P_active, TF)

    if compact_device
        # Compact device plan (task 023d): flat operator tables replace the nested
        # host storage, the packed route arrays stay empty (device routes remain in
        # emission order), and route_class lives on the device where route
        # generation writes it directly.
        ymlen = length_ymodes(P_active)
        y_flat_mult_host = Matrix{TF}(undef, ymlen, nangles)
        y_flat_loc_host = Matrix{TF}(undef, ymlen, nangles)
        @inbounds for a in 1:nangles, n in 0:P_active
            d = 2 * n + 1
            off = ymode_offset(n)
            Mm = _precomputed_y_block(TF, Um, Vm, angle_thetas[a], n)
            Ml = _precomputed_y_block(TF, Ul, Vl, angle_thetas[a], n)
            for q in 1:d, k in 1:d
                y_flat_mult_host[off + (q - 1) * d + k, a] = Mm[k, q]
                y_flat_loc_host[off + (q - 1) * d + k, a] = Ml[k, q]
            end
        end
        zlen = m2l_z_block_length(P_active)
        z_flat_host = Matrix{TF}(undef, zlen, noffsets)
        @inbounds for i in 1:noffsets
            m2l_z_blocks!(view(z_flat_host, :, i), offset_rs[i], P_active)
        end
        empty_slab = similar(exemplar, TF, 0, 0)
        scratch = (aphi=empty_slab, yphi=empty_slab, zphi=empty_slab,
            rphi=empty_slab, cphi=empty_slab, achi=empty_slab, ychi=empty_slab,
            zchi=empty_slab, rchi=empty_slab, cchi=empty_slab)
        return ResidentM2LPrecomputedYPlan(
            _array_like_vector(exemplar, Int32, Vector{Int32}(undef, route_capacity)),
            offset_to_angle, keys, angle_thetas, angle_offset_starts, angle_offsets,
            zeros(Int, nangles), zeros(Int, nangles), zeros(Int, nangles + 1),
            zeros(Int, noffsets), zeros(Int, noffsets + 1), Int[], Int[], TF[],
            offset_phis, Vector{Vector{Matrix{TF}}}(), Vector{Vector{Matrix{TF}}}(),
            Vector{Vector{Matrix{TF}}}(), Vector{Vector{Matrix{TF}}}(),
            Vector{Vector{TF}}(), Vector{Vector{TF}}(), scratch,
            _array_like_vector(exemplar, Int32, zeros(Int32, noffsets)),
            zeros(Int32, noffsets),
            _array_like_matrix(exemplar, TF, y_flat_mult_host),
            _array_like_matrix(exemplar, TF, y_flat_loc_host),
            _array_like_matrix(exemplar, TF, z_flat_host),
            offset_rs, Ref{Any}(nothing),
        )
    end

    y_mult = [[_array_like_matrix(exemplar, TF,
        _precomputed_y_block(TF, Um, Vm, angle_thetas[a], n)) for n in 0:P_active]
        for a in 1:nangles]
    y_loc = [[_array_like_matrix(exemplar, TF,
        _precomputed_y_block(TF, Ul, Vl, angle_thetas[a], n)) for n in 0:P_active]
        for a in 1:nangles]

    z_phi = Vector{Vector{Matrix{TF}}}(undef, noffsets)
    z_chi = Vector{Vector{Matrix{TF}}}(undef, noffsets)
    lh_phi_rows = Vector{Vector{TF}}(undef, noffsets)
    lh_chi_rows = Vector{Vector{TF}}(undef, noffsets)
    blocks = Vector{TF}(undef, m2l_z_block_length(P_active))
    lh_A = LH ? Vector{TF}(undef, _operator_ncomplex(P_active)) : TF[]
    lh_B = similar(lh_A)
    @inbounds for i in 1:noffsets
        m2l_z_blocks!(blocks, offset_rs[i], P_active)
        z_phi[i] = _z_block_matrices_like(exemplar, TF, blocks, :m2l, P_phi, P_active)
        z_chi[i] = LH ? _z_block_matrices_like(exemplar, TF, blocks, :m2l, P_active, P_active) : Matrix{TF}[]
        if LH
            lamb_helmholtz_local_coeffs!(lh_A, lh_B, offset_rs[i], P_active)
            ar, br = _lh_local_row_coefficients(TF, P_phi, P_active, lh_A, lh_B)
            lh_phi_rows[i] = _array_like_vector(exemplar, TF, ar)
            lh_chi_rows[i] = _array_like_vector(exemplar, TF, br)
        else
            lh_phi_rows[i] = TF[]
            lh_chi_rows[i] = TF[]
        end
    end

    # The dense fixed-grid bound for one angle class is the sum of the numbers of
    # valid target/source coordinate pairs for its offsets, capped by global route
    # capacity.  This is strictly tighter than noffsets_in_angle * max_cells when
    # boundary offsets cannot occur at every occupied cell.
    G = Int(grid_resolution)
    angle_capacities = zeros(Int, nangles)
    @inbounds for angle in 1:nangles
        dense = 0
        for p in angle_offset_starts[angle]:(angle_offset_starts[angle + 1] - 1)
            o = accepted_offsets[angle_offsets[p]]
            dense += max(G - abs(o[1]), 0) * max(G - abs(o[2]), 0) * max(G - abs(o[3]), 0)
        end
        angle_capacities[angle] = min(Int(route_capacity), dense,
            length(angle_offset_starts[angle]:(angle_offset_starts[angle + 1] - 1)) * Int(max_cells))
    end
    width = max(maximum(angle_capacities; init=0), 1)
    ndphi = degree_major_dof(P_phi)
    ndchi = LH ? degree_major_dof(P_active) : 0
    mkphi() = similar(exemplar, TF, ndphi, width)
    mkchi() = similar(exemplar, TF, ndchi, LH ? width : 0)
    scratch = (aphi=mkphi(), yphi=mkphi(), zphi=mkphi(), rphi=mkphi(), cphi=mkphi(),
        achi=mkchi(), ychi=mkchi(), zchi=mkchi(), rchi=mkchi(), cchi=mkchi())
    return ResidentM2LPrecomputedYPlan(
        Vector{Int32}(undef, route_capacity), offset_to_angle, keys, angle_thetas,
        angle_offset_starts, angle_offsets, angle_capacities, zeros(Int, nangles),
        zeros(Int, nangles + 1), zeros(Int, noffsets), zeros(Int, noffsets + 1),
        Vector{Int}(undef, route_capacity), Vector{Int}(undef, route_capacity),
        Vector{TF}(undef, route_capacity), offset_phis, y_mult, y_loc, z_phi, z_chi,
        lh_phi_rows, lh_chi_rows, scratch,
    )
end

@noinline function _dense_m2l_overflow(context::AbstractString)
    throw(ArgumentError("DenseTranslationM2L footprint arithmetic overflow while computing $context; lower P, class count, route capacity, or chunk widths"))
end

@inline function _dense_checked_mul(a::Int, b::Int, context::AbstractString)
    try
        return Base.checked_mul(a, b)
    catch err
        err isa OverflowError || rethrow()
        return _dense_m2l_overflow(context)
    end
end

@inline function _dense_checked_add(a::Int, b::Int, context::AbstractString)
    try
        return Base.checked_add(a, b)
    catch err
        err isa OverflowError || rethrow()
        return _dense_m2l_overflow(context)
    end
end

function _dense_sum_checked(values, context::AbstractString)
    total = 0
    for value in values
        total = _dense_checked_add(total, value, context)
    end
    return total
end

function _dense_to_int(value::Integer, context::AbstractString)
    try
        return Int(value)
    catch err
        err isa InexactError || err isa OverflowError || rethrow()
        throw(ArgumentError("DenseTranslationM2L $context=$value is not representable as Int"))
    end
end

"""Pure payload-byte estimate for a host `ResidentM2LDensePlan`."""
function _dense_m2l_footprint(::Type{TF}, basis_info::OperatorBasisInfo{B,LH},
        nclasses::Integer, route_capacity::Integer, apply_width::Integer,
        build_width::Integer; noperators::Integer=nclasses) where {TF,B,LH}
    nc = try Int(nclasses) catch; _dense_m2l_overflow("class count") end
    nr = try Int(route_capacity) catch; _dense_m2l_overflow("route capacity") end
    aw = try Int(apply_width) catch; _dense_m2l_overflow("apply width") end
    bw = try Int(build_width) catch; _dense_m2l_overflow("build width") end
    nc >= 0 && nr >= 0 && aw > 0 && bw > 0 || throw(ArgumentError(
        "dense M2L footprint dimensions must be nonnegative with positive widths"))
    D = _dense_m2m_dof(basis_info, Val(LH))
    eltbytes = sizeof(TF)
    intbytes = sizeof(Int)
    nop = try Int(noperators) catch; _dense_m2l_overflow("operator count") end
    operator_elts = _dense_checked_mul(_dense_checked_mul(D, D, "operator dimensions"),
        nop, "operator class storage")
    operator_bytes = _dense_checked_mul(operator_elts, eltbytes, "operator bytes")
    slab_elts = _dense_checked_mul(_dense_checked_mul(2, D, "slab rows"), aw,
        "slab dimensions")
    scratch_bytes = _dense_checked_mul(slab_elts, eltbytes, "application slab bytes")
    route_class_bytes = _dense_checked_mul(nr, sizeof(Int32), "route class bytes")
    packed_bytes = _dense_checked_mul(_dense_checked_mul(2, nr, "packed route arrays"),
        intbytes, "packed route bytes")
    class_words = _dense_checked_add(_dense_checked_mul(2, nc, "class arrays"),
        _dense_checked_add(nc, 1, "class starts"), "route metadata words")
    class_bytes = _dense_checked_mul(class_words, intbytes, "class metadata bytes")
    route_metadata_bytes = _dense_sum_checked(
        (route_class_bytes, packed_bytes, class_bytes), "route metadata total")
    persistent_bytes = _dense_sum_checked(
        (operator_bytes, scratch_bytes, route_metadata_bytes), "persistent total")

    # Builder-owned array payloads which coexist with the persistent plan while
    # the final class is constructed: source/target plus the two M2L scratch flat
    # buffers, source/target degree-major buffers, and stacked input/output slabs.
    flat_rows = _dense_checked_add(basis_info.basis_dof_phi,
        LH ? basis_info.basis_dof_chi : 0, "builder flat row count")
    builder_matrix_rows = _dense_checked_add(_dense_checked_mul(4, flat_rows,
        "builder flat buffers"), _dense_checked_mul(4, D, "builder dense buffers"),
        "builder matrix rows")
    builder_matrix_elts = _dense_checked_mul(builder_matrix_rows, bw,
        "builder matrix elements")
    P = basis_info.orders.P_active
    nh = _operator_ncomplex(P)
    P1 = _dense_checked_add(P, 1, "active order plus one")
    twoP1 = _dense_checked_add(_dense_checked_mul(2, P, "twice active order"),
        1, "complex y-mode length")
    base_real_elts = _dense_sum_checked((
        _dense_checked_mul(12, nh, "operator scratch expansion arrays"),
        length_Ts(P), _dense_checked_mul(2, max(P, 1), "y trig scratch"),
        _dense_checked_mul(2, nh, "z trig scratch"),
        _dense_checked_mul(2, P1, "azimuth scratch"),
        _dense_checked_mul(2, twoP1, "complex y-mode scratch"),
        m2l_z_block_length(P),
        LH ? _dense_checked_mul(2, nh, "Lamb-Helmholtz scratch") : 0,
        _dense_checked_mul(3, bw, "construction geometry vectors")),
        "builder vector elements")
    builder_elts = _dense_checked_add(builder_matrix_elts, base_real_elts,
        "builder payload elements")
    builder_bytes = _dense_checked_mul(builder_elts, eltbytes, "builder payload bytes")
    construction_peak_bytes = _dense_checked_add(persistent_bytes, builder_bytes,
        "construction peak")
    return (; ndof=D, operator_bytes, scratch_bytes, route_metadata_bytes,
        persistent_bytes, construction_peak_bytes, builder_bytes)
end

function _dense_m2l_capacity(offset::SVector{3,<:Integer}, route_capacity::Int,
        max_cells::Int, G::Int)
    dense = 1
    @inbounds for k in 1:3
        component = _dense_to_int(offset[k], "offset component")
        magnitude = try Base.checked_abs(component) catch err
            err isa OverflowError || rethrow()
            _dense_m2l_overflow("offset magnitude")
        end
        extent = try Base.checked_sub(G, magnitude) catch err
            err isa OverflowError || rethrow()
            _dense_m2l_overflow("offset class extent")
        end
        extent = max(extent, 0)
        dense = _dense_checked_mul(dense, extent, "offset class capacity")
    end
    return min(route_capacity, max_cells, dense)
end

function _dense_m2l_limit_error(strategy::DenseTranslationM2L, D::Int,
        nclasses::Int, apply_width::Int, build_width::Int, footprint)
    mib(x) = x / 2.0^20
    throw(ArgumentError(
        "DenseTranslationM2L persistent footprint exceeds max_persistent_bytes: " *
        "D=$D, classes=$nclasses, apply_width=$apply_width, build_width=$build_width; " *
        "operators=$(footprint.operator_bytes) bytes ($(mib(footprint.operator_bytes)) MiB), " *
        "slabs=$(footprint.scratch_bytes) bytes ($(mib(footprint.scratch_bytes)) MiB), " *
        "route metadata=$(footprint.route_metadata_bytes) bytes, " *
        "persistent=$(footprint.persistent_bytes) bytes ($(mib(footprint.persistent_bytes)) MiB), " *
        "estimated construction peak=$(footprint.construction_peak_bytes) bytes " *
        "($(mib(footprint.construction_peak_bytes)) MiB), limit=$(strategy.max_persistent_bytes) bytes. " *
        "Raise the limit, lower P, disable Lamb-Helmholtz, or reduce apply_chunk " *
        "when slabs are material; chunking does not reduce operator storage."))
end

function _check_dense_m2l_operator_finite!(K::AbstractMatrix{TF},
        basis_info::OperatorBasisInfo{B,LH}, offset) where {TF,B,LH}
    all(isfinite, K) && return K
    P = basis_info.orders.P_phi
    active = basis_info.orders.P_active
    active_msg = active == P ? "" : ", active_P=$active"
    throw(ArgumentError(
        "DenseTranslationM2L materialized a non-finite operator: " *
        "precision=$(TF), P=$P$(active_msg), Lamb-Helmholtz=$(LH), " *
        "displacement offset=$(offset). " *
        "Use Float64, lower P, disable Lamb-Helmholtz, or choose " *
        "PrecomputedFactoredYM2L / a factored / concat M2L strategy."))
end

function ResidentM2LDensePlan(::Type{TF}, basis_info::OperatorBasisInfo{B,LH},
        accepted_offsets::AbstractVector{<:SVector{3,<:Integer}}, cell_width::Real,
        route_capacity::Integer, max_cells::Integer, grid_resolution::Integer,
        strategy::DenseTranslationM2L,
        invariant::OperatorInvariantCache{TF,B,LH};
        hierarchical_noffsets::Integer=0) where {TF,B,LH}
    nroutes = _dense_to_int(route_capacity, "route capacity")
    ncells = _dense_to_int(max_cells, "cell capacity")
    G = _dense_to_int(grid_resolution, "grid resolution")
    nroutes >= 0 && ncells >= 0 && G >= 0 || throw(ArgumentError(
        "dense M2L capacities must be nonnegative"))
    nclasses = length(accepted_offsets)
    hno = Int(hierarchical_noffsets)
    hierarchical = hno > 0
    hierarchical && (nclasses % hno == 0) || !hierarchical ||
        throw(ArgumentError("hierarchical dense class count must be divisible by offset count"))
    noperators = hierarchical ? hno : nclasses
    D = _dense_m2m_dof(basis_info, Val(LH))
    class_capacities = Vector{Int}(undef, nclasses)
    @inbounds for i in eachindex(accepted_offsets)
        class_capacities[i] = _dense_m2l_capacity(accepted_offsets[i], nroutes,
            ncells, G)
    end
    largest = maximum(class_capacities; init=0)
    apply_width = max(strategy.apply_chunk > 0 ? min(largest, strategy.apply_chunk) : largest, 1)
    build_width = strategy.build_chunk > 0 ? min(D, strategy.build_chunk) : D
    footprint = _dense_m2l_footprint(TF, basis_info, nclasses, nroutes,
        apply_width, build_width; noperators)
    footprint.persistent_bytes <= strategy.max_persistent_bytes ||
        _dense_m2l_limit_error(strategy, D, nclasses, apply_width, build_width, footprint)

    # No operator or slab allocation occurs before the configured persistent gate.
    operators = [Matrix{TF}(undef, D, D) for _ in 1:noperators]
    workspace = DenseM2LBuilderWorkspace(TF, basis_info, invariant, build_width)
    operator_offsets = hierarchical ?
        @view(accepted_offsets[(nclasses - hno + 1):nclasses]) : accepted_offsets
    @inbounds for (i, offset) in enumerate(operator_offsets)
        delta = TF(cell_width) * SVector{3,TF}(offset)
        r, theta, phi = cartesian_to_spherical(delta)
        build_dense_m2l_operator!(operators[i], r, theta, phi, invariant, workspace,
            Val(LH))
        _check_dense_m2l_operator_finite!(operators[i], basis_info, offset)
    end
    class_operator = hierarchical ?
        [mod1(i, hno) for i in 1:nclasses] : collect(1:nclasses)
    source_scale = hierarchical ? ones(TF, D, nclasses) : Matrix{TF}(undef, 0, 0)
    target_scale = hierarchical ? ones(TF, D, nclasses) : Matrix{TF}(undef, 0, 0)
    if hierarchical
        nlevels = nclasses ÷ hno
        Dphi = degree_major_dof(basis_info.orders.P_phi)
        @inbounds for li in 1:nlevels
            # Classes are level-ascending (L=2 first), while the final block is
            # the leaf reference.  Thus s = 2^(nlevels-li).
            s = TF(1 << (nlevels - li))
            for k in 1:hno
                cls = (li - 1) * hno + k
                for n in 0:basis_info.orders.P_phi, row in degree_row_range(n)
                    source_scale[row, cls] = s^(-n)
                    target_scale[row, cls] = s^(-(n + 1))
                end
                if LH
                    for n in 0:basis_info.orders.P_active, row in degree_row_range(n)
                        rr = Dphi + row
                        source_scale[rr, cls] = s^(-(n - 1))
                        target_scale[rr, cls] = s^(-(n + 2))
                    end
                end
            end
        end
    end
    return ResidentM2LDensePlan{TF}(
        Vector{Int32}(undef, nroutes), zeros(Int, nclasses), zeros(Int, nclasses + 1),
        class_capacities, Vector{Int}(undef, nroutes), Vector{Int}(undef, nroutes),
        operators, class_operator, source_scale, target_scale,
        Matrix{TF}(undef, D, apply_width), Matrix{TF}(undef, D, apply_width),
        D, apply_width, footprint.operator_bytes, footprint.scratch_bytes,
        footprint.route_metadata_bytes, footprint.persistent_bytes,
        footprint.construction_peak_bytes)
end

function _radix_cache_workspace(::Type{TF}, basis_info::OperatorBasisInfo{B,LH},
        exemplar::FlatCoefficientBuffer, ell::Int, h0::TF, max_cells::Int,
        max_nodes::Int, route_capacity::Int,
        accepted_offsets::Vector{SVector{3,Int}}, invariant::OperatorInvariantCache,
        m2l_strategy::AbstractResidentM2LStrategy,
        operator::AbstractM2LOperator=MaterializedYRotationM2L();
        compact_cuda_factored::Bool=false,
        dense_cuda_estimated_peak_bytes::Int=0,
        hierarchical_noffsets::Int=0) where {TF,B,LH}
    m2l_strategy isa Union{ConcatenatedFixedZM2L,PrecomputedFactoredYM2L,DenseTranslationM2L} ||
        throw(ArgumentError("RadixFMMCache supports ConcatenatedFixedZM2L, " *
            "PrecomputedFactoredYM2L, or DenseTranslationM2L; the SharedRotationM2L " *
            "group layout is not refreshable in place"))
    m2l_strategy isa PrecomputedFactoredYM2L && !(operator isa FactoredRotationM2L) &&
        throw(ArgumentError("PrecomputedFactoredYM2L requires operator=FactoredRotationM2L()"))
    m2l_strategy isa DenseTranslationM2L && !(operator isa MaterializedYRotationM2L) &&
        throw(ArgumentError("DenseTranslationM2L requires operator=MaterializedYRotationM2L()"))
    P_phi = basis_info.orders.P_phi
    P_active = basis_info.orders.P_active
    phi_flat_idx = _array_like_vector(exemplar.phi, Int, _degree_major_to_flat_indices(P_phi))
    chi_flat_idx = LH ?
        _array_like_vector(exemplar.phi, Int, _degree_major_to_flat_indices(P_active)) :
        _array_like_vector(exemplar.phi, Int, Int[])
    maps_phi = DegreeMajorMaps(TF, P_phi, exemplar.phi)
    maps_chi = LH ? DegreeMajorMaps(TF, P_active, exemplar.chi) : maps_phi
    y_mult_U = _ymode_real_blocks(_array_like_vector(exemplar.phi, Complex{TF}, invariant.y_mult_U), P_active, TF)
    y_mult_V = _ymode_real_blocks(_array_like_vector(exemplar.phi, Complex{TF}, invariant.y_mult_V), P_active, TF)
    y_loc_U = _ymode_real_blocks(_array_like_vector(exemplar.phi, Complex{TF}, invariant.y_loc_U), P_active, TF)
    y_loc_V = _ymode_real_blocks(_array_like_vector(exemplar.phi, Complex{TF}, invariant.y_loc_V), P_active, TF)
    nonleaf_idx = collect(1:max_nodes)      # capacity; resize!d by the step refresh

    child_radius(Lc) = sqrt(TF(3)) * h0 / (1 << Lc)
    m2m_groups = [
        _resident_capacity_group(exemplar.phi, TF, basis_info, :m2m, level,
            child_radius(level + 1), _radix_level_node_capacity(level + 1, max_cells))
        for level in (ell - 1):-1:0
    ]
    l2l_groups = [
        _resident_capacity_group(exemplar.phi, TF, basis_info, :l2l, level,
            child_radius(level), _radix_level_node_capacity(level, max_cells))
        for level in 1:ell
    ]
    max_batch = max(_radix_level_node_capacity(ell, max_cells), 1)

    cell_width = (2 * h0) / (1 << ell)
    m2l_concat = m2l_strategy isa DenseTranslationM2L ?
        (compact_cuda_factored ?
            _build_cuda_dense_m2l_plan(TF, basis_info, accepted_offsets, cell_width,
                route_capacity, max_cells, 1 << ell, m2l_strategy, invariant,
                dense_cuda_estimated_peak_bytes) :
            ResidentM2LDensePlan(TF, basis_info, accepted_offsets, cell_width,
                route_capacity, max_cells, 1 << ell, m2l_strategy, invariant;
                hierarchical_noffsets)) :
        m2l_strategy isa PrecomputedFactoredYM2L ?
        ResidentM2LPrecomputedYPlan(TF, basis_info, exemplar.phi, accepted_offsets,
            cell_width, route_capacity, max_cells, 1 << ell, invariant;
            compact_device=compact_cuda_factored) :
        operator isa FactoredRotationM2L ?
        ResidentM2LFactoredPlan(TF, basis_info, exemplar.phi, accepted_offsets,
            cell_width, route_capacity, max_cells, invariant;
            build_groups=!compact_cuda_factored) :
        ResidentM2LConcatPlan(TF, basis_info, exemplar.phi, m2l_strategy,
            invariant, accepted_offsets, cell_width, route_capacity;
            whole_window=hierarchical_noffsets > 0)

    # The factored plan reuses the degree-major stage slabs at max_cells width;
    # concat needs only unit-sized legacy shared-M2L buffers.
    factored = operator isa FactoredRotationM2L
    m2l_width = factored && !(m2l_strategy isa PrecomputedFactoredYM2L) ? max(max_cells, 1) : 1
    m2l_sources = _degree_major_buffer_like(TF, basis_info, exemplar, m2l_width)
    m2l_targets = _degree_major_buffer_like(TF, basis_info, exemplar, m2l_width)
    ndof_phi = degree_major_dof(P_phi)
    ndof_chi = LH ? degree_major_dof(P_active) : 0
    mkphi() = similar(exemplar.phi, TF, ndof_phi, max_batch)
    mkchi() = similar(exemplar.phi, TF, ndof_chi, LH ? max_batch : 0)
    aphi = mkphi(); yphi = mkphi(); zphi = mkphi(); rphi = mkphi()
    achi = mkchi(); ychi = mkchi(); zchi = mkchi(); rchi = mkchi()
    cphi = mkphi(); cchi = mkchi()
    phis = similar(exemplar.phi, TF, max_batch)
    thetas = similar(exemplar.phi, TF, max_batch)
    rs = similar(exemplar.phi, TF, max_batch)
    ystk_phi = StackedYChannel(exemplar.phi, TF, invariant, P_phi, max_batch)
    ystk_chi = LH ? StackedYChannel(exemplar.phi, TF, invariant, P_active, max_batch) : nothing
    return ResidentOperatorWorkspace{TF,B,LH}(
        basis_info, phi_flat_idx, chi_flat_idx, maps_phi, maps_chi,
        y_mult_U, y_mult_V, y_loc_U, y_loc_V, nonleaf_idx, max_batch,
        m2l_sources, m2l_targets,
        aphi, yphi, zphi, rphi, achi, ychi, zchi, rchi, cphi, cchi,
        phis, thetas, rs, m2m_groups, ResidentOperatorGroup[], l2l_groups, m2l_concat,
        ystk_phi, ystk_chi,
    )
end
