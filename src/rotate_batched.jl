#------- EXPLICIT Z-ROTATION OPERATORS (Matrix Operator Refactor, task 010) -------#
#
# The z-rotation is block-diagonal in the azimuthal order m: each stored
# coefficient (n, m) is multiplied by the complex phase e^{imϕ}, i.e. the real
# 2x2 rotation block acting on the [real, imag] pair. See the approved theory
# artifact MATRIX_OPERATOR_REFACTOR/theory/z-rotation-operators.md.
#
# Storage-light representation: two diagonal vectors over the compressed
# harmonic index, C[i] = cos(m(i)ϕ) and S[i] = sin(m(i)ϕ), where
# i = harmonic_index(n, m). Forward rotation overwrites the destination with
# e^{imϕ}; inverse/back rotation accumulates into the destination with the
# conjugate phase e^{-imϕ}. The m = 0 block is the identity (C = 1, S = 0).
#
# These operate on the existing production coefficient layout
# weights[real_or_imag, component, harmonic_index] so that behavior is
# bit-for-bit identical to rotate_z! / back_rotate_z! in src/rotate.jl. The
# native flat (basis_dof x batch x channel) buffers are introduced later (task
# 017); this stage is intentionally layout-compatible with current production.

"""
    z_rotation_diagonals!(C, S, ϕ, P)

Fill the storage-light z-rotation diagonals over the compressed harmonic index:
`C[harmonic_index(n,m)] = cos(m*ϕ)` and `S[harmonic_index(n,m)] = sin(m*ϕ)` for
all `0 <= m <= n <= P`. `C` and `S` must each have length at least
`((P+1)*(P+2))>>1`.

The per-`m` phases `e^{imϕ}` are computed once via the same recurrence used by
[`update_eimϕs!`](@ref) and then scattered across every degree `n >= m`.
"""
function z_rotation_diagonals!(C, S, ϕ, P)
    eiϕ_imag, eiϕ_real = sincos(ϕ)

    eimϕ_real, eimϕ_imag = one(eiϕ_real), zero(eiϕ_real)
    @inbounds for m in 0:P
        for n in m:P
            i = harmonic_index(n, m)
            C[i] = eimϕ_real
            S[i] = eimϕ_imag
        end
        # advance e^{i(m+1)ϕ} = e^{imϕ} * e^{iϕ}
        eimϕ_real_tmp = eimϕ_real
        eimϕ_real = eimϕ_real * eiϕ_real - eimϕ_imag * eiϕ_imag
        eimϕ_imag = eimϕ_real_tmp * eiϕ_imag + eimϕ_imag * eiϕ_real
    end

    return C, S
end

"""
    apply_z_rotation!(out, in, C, S, P, lamb_helmholtz::Val, mode::Val)

Apply the explicit z-rotation operator using the storage-light diagonals `C`/`S`
(see [`z_rotation_diagonals!`](@ref)) to the coefficients `in`, writing the
result into `out`. Both arrays use the production layout
`[real_or_imag, component, harmonic_index]`.

`mode`:
- `Val(:overwrite)` — forward rotation by `e^{imϕ}`, assigning into `out`
  (matches `rotate_z!`).
- `Val(:accumulate)` — inverse/back rotation by the conjugate `e^{-imϕ}`,
  accumulating into `out` (matches `back_rotate_z!`).

`lamb_helmholtz` selects whether the second (χ) component channel is processed.
Both channels rotate by identical phases.
"""
function apply_z_rotation!(out, in, C, S, P, lamb_helmholtz::Val{LH}, ::Val{:overwrite}) where LH
    i = 1
    @inbounds for n in 0:P
        for m in 0:n
            c, s = C[i], S[i]

            a, b = in[1,1,i], in[2,1,i]
            out[1,1,i] = c * a - s * b
            out[2,1,i] = s * a + c * b
            if LH
                a, b = in[1,2,i], in[2,2,i]
                out[1,2,i] = c * a - s * b
                out[2,2,i] = s * a + c * b
            end

            i += 1
        end
    end

    return out
end

function apply_z_rotation!(out, in, C, S, P, lamb_helmholtz::Val{LH}, ::Val{:accumulate}) where LH
    i = 1
    @inbounds for n in 0:P
        for m in 0:n
            # conjugate phase e^{-imϕ}: cos unchanged, sin negated
            c, s = C[i], S[i]

            a, b = in[1,1,i], in[2,1,i]
            out[1,1,i] += c * a + s * b
            out[2,1,i] += -s * a + c * b
            if LH
                a, b = in[1,2,i], in[2,2,i]
                out[1,2,i] += c * a + s * b
                out[2,2,i] += -s * a + c * b
            end

            i += 1
        end
    end

    return out
end

#------- INVARIANT AXIS-SWAP OPERATORS (Matrix Operator Refactor, task 013) -------#
#
# The non-z part of the y-alignment used by the rotation-trick M2M/M2L/L2L is, for
# every degree n, generated entirely from the fixed π/2 Wigner blocks H(π/2). The
# angle-dependent y-rotation block factors as
#
#     T_n(θ) = S_n * Z_n(θ) * S_n^{-1}
#
# (see the approved theory artifact
# MATRIX_OPERATOR_REFACTOR/theory/axis-swap-conventions.md), where S_n is built
# only from H(π/2) and Z_n(θ) carries the exp(i ν θ) phases. Production rebuilds
# the full T matrix (Ts) on EVERY translation call via update_Ts! in src/rotate.jl;
# its inner ν loop multiplies two angle-independent H(π/2) products by a scalar and
# the angle-dependent cos/sin(ν θ). The 008c baseline flagged this per-call rebuild
# as the dominant cost.
#
# These operators move the angle-independent H(π/2) products into a precomputed
# cache (the S blocks) so the per-call work collapses to a phase-weighted
# contraction. For each stored (n, m, mp) with 0 <= mp <= m <= n and ν in 0:n:
#
#     S_pos[n,m,mp,ν] : the +mp coefficient   (ν=0 term gated by parity)
#     S_neg[n,m,mp,ν] : the -mp coefficient   (same magnitude, production signs)
#
# build_Ts_from_S! then reconstructs Ts identical (to ~1e-12) to update_Ts! using
# only the cheap cos/sin(ν θ) recurrence. The result is shared between the
# multipole and local paths; those differ only in the sign table (ζ vs η) passed
# to the existing production apply kernels, which this stage reuses unchanged.
# These operate on the production layout weights[real_or_imag, component,
# harmonic_index]; native flat buffers are a later task (017). The functions are
# intentionally internal/non-exported.

"""
    length_S_block(n)

Number of `TF` entries in the degree-`n` axis-swap block for one polarity: the
`length_H(n)` stored `(mp, m)` pairs each carry `n + 1` coefficients (ν in `0:n`).
"""
@inline length_S_block(n) = length_H(n) * (n + 1)

"""
    length_Ss(P)

Total number of `TF` entries needed to store one polarity of the axis-swap blocks
for all degrees `0 <= n <= P` (see [`update_S_blocks!`](@ref)).
"""
@inline length_Ss(P) = S_block_offset(P + 1)

"""
    S_block_offset(n)

Flat-array offset (number of entries preceding) the degree-`n` axis-swap block,
i.e. the summed size of blocks `0:n-1`.
"""
@inline function S_block_offset(n)
    # sum_{j=1}^n j^2(j+1)/2 = (sum(j^3) + sum(j^2)) / 2
    n2 = n * (n + 1)
    return (div(n2 * n2, 4) + div(n2 * (2n + 1), 6)) >> 1
end

"""
    S_index(n, mp, m, ν)

Flat index of the `ν`-th coefficient (ν in `0:n`) of the `(mp, m)` pair within the
degree-`n` axis-swap block. The pair enumeration matches `update_Ts!`
(`m in 0:n`, `mp in 0:m`) via [`H_index`](@ref), so each `(mp, m)` pair owns the
contiguous run `0:n`.
"""
@inline function S_index(n, mp, m, ν)
    return S_block_offset(n) + (H_index(mp, m) - 1) * (n + 1) + ν + 1
end

"""
    update_S_blocks!(S_pos, S_neg, Hs_π2, P)

Materialize the angle-independent axis-swap coefficient blocks from the precomputed
π/2 Wigner blocks `Hs_π2` for all degrees `0 <= n <= P`. `S_pos` and `S_neg` must
each have length at least `length_Ss(P)`.

This is the angle-independent skeleton of `update_Ts!` (`src/rotate.jl`) with the
`cos/sin(ν θ)` factors removed: the same `(n, m, mp, ν)` traversal and the same
sign recurrences (`_1_n`, `_1_n_mp`, `_1_n_mp_ν`, `_1_mp_odd`) are kept, so
[`build_Ts_from_S!`](@ref) reproduces production `Ts`. Per stored `(n, m, mp)`:

- `ν >= 1`: `S_pos = H(π/2)[mp,ν] * H(π/2)[m,ν] * scalar` (the production `×2` is
  applied later by `build_Ts_from_S!`); `S_neg = S_pos * _1_n_mp_ν * _1_mp_odd`.
- `ν = 0`: `S_pos = H(π/2)[m,0] * H(π/2)[mp,0] * scalar0` with
  `scalar0 = iseven(m+mp) ? scalar : 0` (the production `zero_mode`);
  `S_neg = S_pos * _1_n_mp * _1_mp_odd`.

The degree-0 monopole slot is set to the identity `1`; it is never read by
[`build_Ts_from_S!`](@ref) (which sets `Ts[1] = 1` directly).
"""
function update_S_blocks!(S_pos, S_neg, Hs_π2, P)
    TF = eltype(S_pos)

    # degree-0 monopole slot (kept deterministic; not read by build_Ts_from_S!)
    @inbounds S_pos[1] = one(TF)
    @inbounds S_neg[1] = one(TF)

    _1_n = -1.0
    @inbounds for n in 1:P
        H_π2 = get_H(Hs_π2, n)
        base = S_block_offset(n)
        np1 = n + 1

        for m in 0:n
            H_π2_n_m_0 = H_π2[H_index(0, m)]
            _1_mp_odd = 1.0
            m_mp = m
            _1_n_mp = _1_n

            for mp in 0:m
                H_π2_n_mp_0 = H_π2[H_index(0, mp)]
                m_mp_even = iseven(m_mp)
                scalar = get_scalar(m_mp)
                _1_n_mp_ν = -_1_n_mp
                sidx0 = base + (H_index(mp, m) - 1) * np1  # ν-th entry at sidx0 + ν + 1

                # ν = 1:n coefficients (un-doubled; build_Ts_from_S! applies the ×2)
                for ν in 1:n
                    i, j = minmax(mp, ν)
                    H_π2_n_mp_ν = H_π2[H_index(i, j)]
                    i, j = minmax(m, ν)
                    H_π2_n_m_ν = H_π2[H_index(i, j)]

                    coef = H_π2_n_mp_ν * H_π2_n_m_ν * scalar
                    S_pos[sidx0 + ν + 1] = coef
                    S_neg[sidx0 + ν + 1] = coef * _1_n_mp_ν * _1_mp_odd

                    _1_n_mp_ν = -_1_n_mp_ν
                end

                # ν = 0 term (parity-gated, matches production zero_mode)
                scalar0 = m_mp_even ? scalar : zero(scalar)
                val0 = H_π2_n_m_0 * H_π2_n_mp_0 * scalar0
                S_pos[sidx0 + 1] = val0
                S_neg[sidx0 + 1] = val0 * _1_n_mp * _1_mp_odd

                _1_n_mp = -_1_n_mp
                _1_mp_odd = -_1_mp_odd
                m_mp += 1
            end
        end

        _1_n = -_1_n
    end

    return S_pos, S_neg
end

"""
    build_Ts_from_S!(Ts, S_pos, S_neg, β, P, trig=…)

Reconstruct the arbitrary-angle y-rotation matrix `Ts` for angle `β` from the
precomputed axis-swap blocks `S_pos`/`S_neg` (see [`update_S_blocks!`](@ref)),
writing into `Ts` (length at least `length_Ts(P)`). Reproduces `update_Ts!`
(`src/rotate.jl`) to `~1e-12`: the same `cos/sin(ν β)` two-term recurrence and the
same accumulate-then-`×2`-then-add-`ν=0` order are used, with the angle-independent
H(π/2) products read from the cache instead of recomputed.

For each `(n, m, mp)` with `0 <= mp <= m <= n`:

    Ts[T_index( mp, m)] = 2 * Σ_{ν=1}^n S_pos[…,ν] * trig(ν) + S_pos[…,0]
    Ts[T_index(-mp, m)] = 2 * Σ_{ν=1}^n S_neg[…,ν] * trig(ν) + S_neg[…,0]

with `trig(ν) = cos(ν β)` when `m+mp` is even, else `sin(ν β)`. The result is shared
between the multipole and local apply kernels.

`update_Ts!` re-runs the `cos/sin(ν β)` recurrence inside every `(n, m, mp)` triple,
which dominates the rebuild cost. Here the `cos(ν β)` / `sin(ν β)` values for
`ν in 1:P` depend only on `β`, so they are materialized once (into the `trig`
scratch, `cos` in `trig[1:P]` and `sin` in `trig[P+1:2P]`) and reused across all
triples; the `m+mp` parity branch is hoisted out of the innermost `ν` loop. This is
bit-for-bit identical to the per-triple recurrence (same deterministic two-term
sequence and same accumulation order) while cutting the inner-loop work ~2-3×. Pass a
caller-owned `trig` (length `>= 2P`) to stay allocation-free on hot paths; the
convenience method allocates one.
"""
function build_Ts_from_S!(Ts, S_pos, S_neg, β, P, trig)
    length(Ts) >= length_Ts(P) || throw(ArgumentError("Ts length must be at least length_Ts(P)"))
    length(S_pos) >= length_Ss(P) || throw(ArgumentError("S_pos length must be at least length_Ss(P)"))
    length(S_neg) >= length_Ss(P) || throw(ArgumentError("S_neg length must be at least length_Ss(P)"))
    length(trig) >= 2P || throw(ArgumentError("trig length must be at least 2P"))

    TF = eltype(Ts)
    @inbounds Ts[1] = one(TF)
    P == 0 && return Ts

    # Precompute cos(νβ), sin(νβ) for ν in 1:P once via the same two-term recurrence
    # update_Ts! restarts per triple. cos -> trig[ν], sin -> trig[P+ν].
    sβ, cβ = sincos(β)
    c_νm1 = one(TF)
    s_νm1 = zero(TF)
    @inbounds for ν in 1:P
        c_νβ = cβ * c_νm1 - sβ * s_νm1
        s_νβ = sβ * c_νm1 + cβ * s_νm1
        trig[ν] = c_νβ
        trig[P + ν] = s_νβ
        c_νm1 = c_νβ
        s_νm1 = s_νβ
    end

    @inbounds for n in 1:P
        i_T = length_Ts(n - 1)
        base = S_block_offset(n)
        np1 = n + 1

        for m in 0:n
            for mp in 0:m
                sidx0 = base + (H_index(mp, m) - 1) * np1  # ν-th entry at sidx0 + ν + 1

                pos = zero(TF)
                neg = zero(TF)
                # parity branch hoisted out of the ν loop; offset selects cos vs sin
                toff = iseven(m + mp) ? 0 : P
                for ν in 1:n
                    t = trig[toff + ν]
                    pos += S_pos[sidx0 + ν + 1] * t
                    neg += S_neg[sidx0 + ν + 1] * t
                end

                pos *= 2
                neg *= 2
                pos += S_pos[sidx0 + 1]  # ν = 0 term (parity-gated at build of S)
                neg += S_neg[sidx0 + 1]

                Ts[i_T + T_index(mp, m)] = pos
                Ts[i_T + T_index(-mp, m)] = neg
            end
        end
    end

    return Ts
end

# Convenience method: allocates the trig scratch. Hot paths should pass their own.
build_Ts_from_S!(Ts, S_pos, S_neg, β, P) =
    build_Ts_from_S!(Ts, S_pos, S_neg, β, P, Vector{eltype(Ts)}(undef, 2 * max(P, 1)))

"""
    rotate_multipole_y_op!(out, source, Ts, S_pos, S_neg, ζs_mag, β, P, lamb_helmholtz[, trig])

Forward multipole y-alignment built from the cached axis-swap blocks: reconstruct
`Ts` for angle `β` via [`build_Ts_from_S!`](@ref), then apply the production
multipole kernel (`_rotate_multipole_y!`), which uses the `ζ` sign table and resets
`out` before writing. Matches `rotate_multipole_y!` (`src/rotate.jl`).
"""
function rotate_multipole_y_op!(out, source, Ts, S_pos, S_neg, ζs_mag, β, P, lamb_helmholtz::Val{LH}, trig) where LH
    build_Ts_from_S!(Ts, S_pos, S_neg, β, P, trig)
    _rotate_multipole_y!(out, source, Ts, ζs_mag, P, lamb_helmholtz)
    return out
end

rotate_multipole_y_op!(out, source, Ts, S_pos, S_neg, ζs_mag, β, P, lamb_helmholtz::Val{LH}) where LH =
    rotate_multipole_y_op!(
        out, source, Ts, S_pos, S_neg, ζs_mag, β, P, lamb_helmholtz,
        Vector{eltype(Ts)}(undef, 2 * max(P, 1)),
    )

"""
    back_rotate_multipole_y_op!(target, source, Ts, S_pos, S_neg, ζs_mag, β, P, lamb_helmholtz[, trig])

Back multipole y-alignment. Like [`rotate_multipole_y_op!`](@ref) it resets
`target` (it is not an accumulating inverse); accumulation into tree targets
happens at the final `back_rotate_z!`. Matches `back_rotate_multipole_y!`.
"""
function back_rotate_multipole_y_op!(target, source, Ts, S_pos, S_neg, ζs_mag, β, P, lamb_helmholtz::Val{LH}, trig) where LH
    build_Ts_from_S!(Ts, S_pos, S_neg, β, P, trig)
    _rotate_multipole_y!(target, source, Ts, ζs_mag, P, lamb_helmholtz)
    return target
end

back_rotate_multipole_y_op!(target, source, Ts, S_pos, S_neg, ζs_mag, β, P, lamb_helmholtz::Val{LH}) where LH =
    back_rotate_multipole_y_op!(
        target, source, Ts, S_pos, S_neg, ζs_mag, β, P, lamb_helmholtz,
        Vector{eltype(Ts)}(undef, 2 * max(P, 1)),
    )

"""
    rotate_local_y_op!(out, source, Ts, Hs_π2, S_pos, S_neg, ηs_mag, β, P, lamb_helmholtz[, trig])

Forward local y-alignment built from the cached axis-swap blocks. Shares the `Ts`
reconstruction with the multipole path; the local kernel (`_rotate_local_y!`) uses
the `η` sign table (and the `Hs_π2` blocks for negative-order reconstruction) and
resets `out`. Matches `rotate_local_y!` (`src/rotate.jl`).
"""
function rotate_local_y_op!(out, source, Ts, Hs_π2, S_pos, S_neg, ηs_mag, β, P, lamb_helmholtz::Val{LH}, trig) where LH
    build_Ts_from_S!(Ts, S_pos, S_neg, β, P, trig)
    _rotate_local_y!(out, source, Ts, Hs_π2, ηs_mag, P, lamb_helmholtz)
    return out
end

rotate_local_y_op!(out, source, Ts, Hs_π2, S_pos, S_neg, ηs_mag, β, P, lamb_helmholtz::Val{LH}) where LH =
    rotate_local_y_op!(
        out, source, Ts, Hs_π2, S_pos, S_neg, ηs_mag, β, P, lamb_helmholtz,
        Vector{eltype(Ts)}(undef, 2 * max(P, 1)),
    )

"""
    back_rotate_local_y_op!(target, source, Ts, Hs_π2, S_pos, S_neg, ηs_mag, β, P, lamb_helmholtz[, trig])

Back local y-alignment. Resets `target` (not an accumulating inverse). Matches
`back_rotate_local_y!`.
"""
function back_rotate_local_y_op!(target, source, Ts, Hs_π2, S_pos, S_neg, ηs_mag, β, P, lamb_helmholtz::Val{LH}, trig) where LH
    build_Ts_from_S!(Ts, S_pos, S_neg, β, P, trig)
    _rotate_local_y!(target, source, Ts, Hs_π2, ηs_mag, P, lamb_helmholtz)
    return target
end

back_rotate_local_y_op!(target, source, Ts, Hs_π2, S_pos, S_neg, ηs_mag, β, P, lamb_helmholtz::Val{LH}) where LH =
    back_rotate_local_y_op!(
        target, source, Ts, Hs_π2, S_pos, S_neg, ηs_mag, β, P, lamb_helmholtz,
        Vector{eltype(Ts)}(undef, 2 * max(P, 1)),
    )

#------- FIXED Y-SWAP PRIMITIVES (Matrix Operator Refactor, task 013b) -------#
#
# These are the primitive fixed ±π/2 y stages for the explicit factored rotation
# path. They intentionally do not assemble the full Z_phi -> S -> Z_theta -> S_inv
# chain; that composition is a later task. The supplied T_y_*90 matrices are
# precomputed with build_Ts_from_S! from the cached S_pos/S_neg blocks and then
# applied through the same production-parity kernels as the materialized
# arbitrary-y wrappers above. The kernels reset destination storage, including the
# inactive chi channel for Val(false).
#
# Note: the pos90/neg90 function bodies are identical — the ±π/2 sign is carried
# ENTIRELY by the supplied T_y_pos90 / T_y_neg90 matrix, not by the function. The
# distinct names exist so the 013c factored-alignment composition can name the
# forward swap (S) and inverse swap (S⁻¹) stages explicitly.

"""
    multipole_y_swap_pos90!(out, source, T_y_pos90, ζs_mag, P, lamb_helmholtz)

Apply the fixed `R_y(+π/2)` multipole y-swap stage using a precomputed
production-parity `T_y_pos90` matrix. Internal/non-exported primitive for task
013b.
"""
function multipole_y_swap_pos90!(out, source, T_y_pos90, ζs_mag, P, lamb_helmholtz::Val{LH}) where LH
    _rotate_multipole_y!(out, source, T_y_pos90, ζs_mag, P, lamb_helmholtz)
    return out
end

"""
    multipole_y_swap_neg90!(out, source, T_y_neg90, ζs_mag, P, lamb_helmholtz)

Apply the fixed `R_y(-π/2)` multipole inverse y-swap stage using a precomputed
production-parity `T_y_neg90` matrix. Internal/non-exported primitive for task
013b.
"""
function multipole_y_swap_neg90!(out, source, T_y_neg90, ζs_mag, P, lamb_helmholtz::Val{LH}) where LH
    _rotate_multipole_y!(out, source, T_y_neg90, ζs_mag, P, lamb_helmholtz)
    return out
end

"""
    local_y_swap_pos90!(out, source, T_y_pos90, Hs_π2, ηs_mag, P, lamb_helmholtz)

Apply the fixed `R_y(+π/2)` local y-swap stage using a precomputed
production-parity `T_y_pos90` matrix. Internal/non-exported primitive for task
013b.
"""
function local_y_swap_pos90!(out, source, T_y_pos90, Hs_π2, ηs_mag, P, lamb_helmholtz::Val{LH}) where LH
    _rotate_local_y!(out, source, T_y_pos90, Hs_π2, ηs_mag, P, lamb_helmholtz)
    return out
end

"""
    local_y_swap_neg90!(out, source, T_y_neg90, Hs_π2, ηs_mag, P, lamb_helmholtz)

Apply the fixed `R_y(-π/2)` local inverse y-swap stage using a precomputed
production-parity `T_y_neg90` matrix. Internal/non-exported primitive for task
013b.
"""
function local_y_swap_neg90!(out, source, T_y_neg90, Hs_π2, ηs_mag, P, lamb_helmholtz::Val{LH}) where LH
    _rotate_local_y!(out, source, T_y_neg90, Hs_π2, ηs_mag, P, lamb_helmholtz)
    return out
end

#------- GLOBALLY BATCHED FACTORED ROTATION ALIGNMENT (Matrix Operator Refactor, task 013c) -------#
#
# Genuinely factored y-rotation. For every degree n the production y-operator factors
# as  Y_n(θ) = U_n · diag(e^{iνθ}) · V_n  (ν = -n..n), with U_n / V_n FIXED (angle- and
# geometry-independent) per-degree matrices and the only θ dependence the cheap diagonal
# e^{iνθ}. This is the S_n · Z_n(θ) · S_n^{-1} form of theory/axis-swap-conventions.md
# realized as two batch-shared fixed swaps around a per-column z-rotation: forward swap
# V (apply once per batch), diagonal e^{iνθ_j} (per column), back swap U. Cost is
# O(P^3) per column with batch-shared fixed matrices — NOT the per-call materialized
# Ts(θ) rebuild of the 013 path, and NOT the per-(n,m,mp) Σ_ν S·trig(νθ) contraction
# that two earlier 013c attempts collapsed into (which is the same O(P^4) materialized
# arithmetic in disguise).
#
# The fixed modes are obtained at cache build by sampling the production-parity y
# kernels and rank-1-factoring each angular Fourier component of Y_n (every component is
# rank 1); see update_factored_y_modes!. The ζ (multipole) and η (local) paths get
# separate U/V (the dressing is baked into the fixed modes), so the staged apply is
# identical for both and is selected purely by which modes the caller passes. The earlier
# ζ-dressed ±π/2 swap primitives (013b T_y_pos90/T_y_neg90) cannot reproduce R_y(θ) when
# composed with a z-rotation — (ζS)·Z·(ζS⁻¹) ≠ ζ·(S·Z·S⁻¹), ζ does not commute through
# the swap — so they are not used here (see the roadmap/derivation amendment for 013c).

mutable struct FactoredRotationStageStats
    z_phi_calls::Int
    z_theta_calls::Int
    fixed_swap_calls::Int
end

FactoredRotationStageStats() = FactoredRotationStageStats(0, 0, 0)

@inline _maybe_count_z_phi!(stats::Nothing) = nothing
@inline _maybe_count_z_phi!(stats::FactoredRotationStageStats) = (stats.z_phi_calls += 1)
@inline _maybe_count_z_theta!(stats::Nothing) = nothing
@inline _maybe_count_z_theta!(stats::FactoredRotationStageStats) = (stats.z_theta_calls += 1)
@inline _maybe_count_fixed_swap!(stats::Nothing) = nothing
@inline _maybe_count_fixed_swap!(stats::FactoredRotationStageStats) = (stats.fixed_swap_calls += 1)

"""
    apply_z_rotation_batch!(out, in, phis, P, lamb_helmholtz, mode)

Apply a z-rotation to a batch of production-layout coefficient columns stored as
`[real_or_imag, component, harmonic_index, batch_column]`. Each batch column uses
its own angle from `phis`. `mode = Val(:overwrite)` writes `out`; `mode =
Val(:accumulate)` applies the conjugate phase and accumulates into `out`.
"""
function apply_z_rotation_batch!(out, in, phis, P, lamb_helmholtz::Val{LH}, ::Val{:overwrite}) where LH
    nbatch = length(phis)
    @boundscheck size(out, 4) >= nbatch || throw(ArgumentError("out batch dimension is smaller than phis"))
    @boundscheck size(in, 4) >= nbatch || throw(ArgumentError("in batch dimension is smaller than phis"))

    @inbounds for j in 1:nbatch
        sϕ, cϕ = sincos(phis[j])
        cm, sm = one(cϕ), zero(cϕ)
        i = 1
        for m in 0:P
            for n in m:P
                i = harmonic_index(n, m)
                a, b = in[1,1,i,j], in[2,1,i,j]
                out[1,1,i,j] = cm * a - sm * b
                out[2,1,i,j] = sm * a + cm * b
                if LH
                    a, b = in[1,2,i,j], in[2,2,i,j]
                    out[1,2,i,j] = cm * a - sm * b
                    out[2,2,i,j] = sm * a + cm * b
                end
            end
            cm_old = cm
            cm = cm_old * cϕ - sm * sϕ
            sm = cm_old * sϕ + sm * cϕ
        end
    end
    return out
end

function apply_z_rotation_batch!(out, in, phis, P, lamb_helmholtz::Val{LH}, ::Val{:accumulate}) where LH
    nbatch = length(phis)
    @boundscheck size(out, 4) >= nbatch || throw(ArgumentError("out batch dimension is smaller than phis"))
    @boundscheck size(in, 4) >= nbatch || throw(ArgumentError("in batch dimension is smaller than phis"))

    @inbounds for j in 1:nbatch
        sϕ, cϕ = sincos(phis[j])
        cm, sm = one(cϕ), zero(cϕ)
        for m in 0:P
            for n in m:P
                i = harmonic_index(n, m)
                a, b = in[1,1,i,j], in[2,1,i,j]
                out[1,1,i,j] += cm * a + sm * b
                out[2,1,i,j] += -sm * a + cm * b
                if LH
                    a, b = in[1,2,i,j], in[2,2,i,j]
                    out[1,2,i,j] += cm * a + sm * b
                    out[2,2,i,j] += -sm * a + cm * b
                end
            end
            cm_old = cm
            cm = cm_old * cϕ - sm * sϕ
            sm = cm_old * sϕ + sm * cϕ
        end
    end
    return out
end

function z_theta_batch_diagonals!(Cθ, Sθ, thetas, P)
    nbatch = length(thetas)
    @boundscheck size(Cθ, 1) >= max(P, 1) || throw(ArgumentError("Cθ must have at least P rows"))
    @boundscheck size(Sθ, 1) >= max(P, 1) || throw(ArgumentError("Sθ must have at least P rows"))
    @boundscheck size(Cθ, 2) >= nbatch || throw(ArgumentError("Cθ batch dimension is smaller than thetas"))
    @boundscheck size(Sθ, 2) >= nbatch || throw(ArgumentError("Sθ batch dimension is smaller than thetas"))

    @inbounds for j in 1:nbatch
        sθ, cθ = sincos(thetas[j])
        cprev, sprev = one(cθ), zero(cθ)
        for ν in 1:P
            cν = cθ * cprev - sθ * sprev
            sν = sθ * cprev + cθ * sprev
            Cθ[ν,j] = cν
            Sθ[ν,j] = sν
            cprev = cν
            sprev = sν
        end
    end
    return Cθ, Sθ
end

# Per-degree fixed mode matrices U_n, V_n (each (2n+1)x(2n+1) complex). Y_n(θ) =
# U_n diag(e^{iνθ}) V_n, with every Fourier component of Y_n rank 1, so U_n holds the
# left mode vectors (columns, index by νidx = ν+n+1) and V_n the right mode covectors
# (rows). Flat column-major storage per degree; offsets below.
@inline length_ymode_block(n) = (2n + 1)^2
@inline ymode_offset(n) = div(n * (2n - 1) * (2n + 1), 3)   # Σ_{k=0}^{n-1}(2k+1)^2
@inline length_ymodes(P) = ymode_offset(P + 1)

# The 2n+1 real degree-n dofs are ordered (re m0; re,im for m=1..n). This maps a dof
# index k (1-based) to its (real_or_imag_row, m) location in a coefficient block.
@inline function _ymode_dof_to_storage(k)
    k == 1 && return (1, 0)
    m = k >> 1
    return (iseven(k) ? 1 : 2, m)
end

"""
    update_factored_y_modes!(U, V, Hs_pi2, sign_mag, P, lamb_helmholtz, is_local)

Materialize the fixed per-degree mode matrices `U`/`V` (each `length_ymodes(P)`
complex) for the genuinely factored y-rotation `Y_n(θ) = U_n diag(e^{iνθ}) V_n`.

For each degree `n` the production y-operator on the `2n+1` real dofs is sampled at
`2n+1` angles (using the production-parity `_rotate_multipole_y!` / `_rotate_local_y!`
kernels at expansion order `n`), DFT'd into its angular Fourier components
`Cν = (1/(2n+1)) Σ_j Y(θ_j) e^{-iνθ_j}`, and each `Cν` — which is rank 1 — is
factored `Cν = uν vν^*` by a pivot (largest-magnitude entry) outer-product split.
`U[:,νidx] = uν` and `V[νidx,:] = vν^*`. This is a one-time cache-build cost; the
runtime stage applies the fixed `U`/`V` with a cheap `e^{iνθ}` diagonal between them
(`_factored_y_batch!`), an `O(P^3)`-per-column, batch-shared GEMM shape that never
rebuilds a per-angle `Ts(θ)`. `is_local::Val` selects the η (local) vs ζ (multipole)
production kernel; the resulting modes carry that path's sign/dressing.
"""
function update_factored_y_modes!(U, V, Hs_pi2, sign_mag, P, lamb_helmholtz::Val{LH}, ::Val{is_local}) where {LH, is_local}
    CF = eltype(U)
    TF = real(CF)
    @inbounds for n in 0:P
        d = 2n + 1
        off = ymode_offset(n)
        Ts = zeros(TF, length_Ts(n))
        src = initialize_expansion(n, TF)
        outv = initialize_expansion(n, TF)
        Ysamp = Vector{Matrix{TF}}(undef, d)
        for jj in 0:d-1
            θ = TF(2 * pi * jj / d)
            update_Ts!(Ts, Hs_pi2, θ, n)
            Y = Matrix{TF}(undef, d, d)
            for k in 1:d
                src .= zero(TF)
                ri, m = _ymode_dof_to_storage(k)
                src[ri, 1, harmonic_index(n, m)] = one(TF)
                if is_local
                    _rotate_local_y!(outv, src, Ts, Hs_pi2, sign_mag, n, lamb_helmholtz)
                else
                    _rotate_multipole_y!(outv, src, Ts, sign_mag, n, lamb_helmholtz)
                end
                Y[1, k] = outv[1, 1, harmonic_index(n, 0)]
                for mm in 1:n
                    Y[2mm, k]     = outv[1, 1, harmonic_index(n, mm)]
                    Y[2mm + 1, k] = outv[2, 1, harmonic_index(n, mm)]
                end
            end
            Ysamp[jj + 1] = Y
        end
        for νidx in 1:d
            ν = νidx - n - 1
            C = zeros(CF, d, d)
            for jj in 0:d-1
                θ = TF(2 * pi * jj / d)
                w = cis(-ν * θ)
                C .+= Ysamp[jj + 1] .* w
            end
            C ./= d
            # pivot rank-1 factorization Cν = uν vν^*  (exact for rank-1)
            pr = 1; pc = 1; best = abs(C[1, 1])
            for c in 1:d, r in 1:d
                a = abs(C[r, c])
                if a > best
                    best = a; pr = r; pc = c
                end
            end
            piv = C[pr, pc]
            for r in 1:d
                U[off + (νidx - 1) * d + r] = C[r, pc]
            end
            for cc in 1:d
                V[off + (cc - 1) * d + νidx] = C[pr, cc] / piv
            end
        end
    end
    return U, V
end

# Apply the genuinely factored y stage over a batch of coefficient columns:
#   forward fixed swap g = V_n x  (batch-shared)  ->  middle diag e^{iνθ_j} (per column)
#   ->  back fixed swap y = real(U_n g).
# Resets `out` (y stages reset; final accumulation is the inverse Z_phi). For Val(false)
# the inactive χ channel is left zero. `gbuf` is a complex scratch of length >= 2P+1.
function _factored_y_batch!(out, source, U, V, gbuf, thetas, P, lamb_helmholtz::Val{LH}, stats) where LH
    TF = real(eltype(U))
    nbatch = length(thetas)
    @boundscheck size(source, 4) >= nbatch || throw(ArgumentError("source batch dimension is smaller than thetas"))
    @boundscheck size(out, 4) >= nbatch || throw(ArgumentError("out batch dimension is smaller than thetas"))
    @boundscheck length(gbuf) >= 2P + 1 || throw(ArgumentError("gbuf must have length >= 2P+1"))
    out .= zero(eltype(out))
    _maybe_count_fixed_swap!(stats)   # forward V swap (fixed, batch-shared)
    _maybe_count_z_theta!(stats)      # diagonal e^{iνθ}
    @inbounds for j in 1:nbatch
        θ = thetas[j]
        for ch in 1:(LH ? 2 : 1)
            for n in 0:P
                d = 2n + 1
                off = ymode_offset(n)
                for νidx in 1:d
                    gr = zero(TF); gi = zero(TF)
                    for k in 1:d
                        ri, m = _ymode_dof_to_storage(k)
                        xk = source[ri, ch, harmonic_index(n, m), j]
                        v = V[off + (k - 1) * d + νidx]
                        gr += real(v) * xk
                        gi += imag(v) * xk
                    end
                    ν = νidx - n - 1
                    s, c = sincos(ν * θ)
                    gbuf[νidx] = complex(c * gr - s * gi, s * gr + c * gi)
                end
                for r in 1:d
                    acc = zero(TF)
                    for νidx in 1:d
                        u = U[off + (νidx - 1) * d + r]
                        g = gbuf[νidx]
                        acc += real(u) * real(g) - imag(u) * imag(g)
                    end
                    ri, m = _ymode_dof_to_storage(r)
                    out[ri, ch, harmonic_index(n, m), j] = acc
                end
            end
        end
    end
    _maybe_count_fixed_swap!(stats)   # back U swap (fixed, batch-shared)
    return out
end

function _factored_source_alignment_batch!(out, source, tmp, gbuf, phis, thetas, U, V, P, lamb_helmholtz::Val{LH}, stats) where LH
    # Z_phi (per-column diagonal) overwrites tmp; the factored y stage then resets out.
    _maybe_count_z_phi!(stats)
    apply_z_rotation_batch!(tmp, source, phis, P, lamb_helmholtz, Val(:overwrite))
    _factored_y_batch!(out, tmp, U, V, gbuf, thetas, P, lamb_helmholtz, stats)
    return out
end

function _factored_return_alignment_batch!(target, source, tmp, gbuf, phis, thetas, U, V, P, lamb_helmholtz::Val{LH}, stats) where LH
    # factored y stage resets tmp; final inverse Z_phi is the only accumulating stage.
    _factored_y_batch!(tmp, source, U, V, gbuf, thetas, P, lamb_helmholtz, stats)
    _maybe_count_z_phi!(stats)
    apply_z_rotation_batch!(target, tmp, phis, P, lamb_helmholtz, Val(:accumulate))
    return target
end

# Public stage API. The multipole vs local distinction is carried entirely by the
# fixed mode matrices `U`/`V` the caller supplies (`y_mult_U`/`y_mult_V` for the ζ
# path, `y_loc_U`/`y_loc_V` for the η path); the staged arithmetic is identical.
#
# Physical-subspace invariant (016b, watch item 2). The factored rank-1 mode
# decomposition reads only the real m=0 row (`_ymode_dof_to_storage(1) == (1, 0)`),
# so it reproduces the production y-operator exactly only for *physical* inputs:
# expansions whose m=0 imaginary component is zero. Every real solid-harmonic
# expansion is physical, so this holds throughout the production FMM. A
# hypothetical intermediate buffer carrying a nonzero m=0 imaginary part would
# silently diverge from production on the factored path, while the materialized
# `Ts(θ)` path stays exact for any input. Use `_assert_factored_input_physical`
# (below) to make that boundary loud rather than silent; it is to be wired in at
# the 023 integration boundary.
multipole_factored_source_alignment_batch!(out, source, tmp, gbuf, phis, thetas, U, V, P, lamb_helmholtz::Val; stats=nothing) =
    _factored_source_alignment_batch!(out, source, tmp, gbuf, phis, thetas, U, V, P, lamb_helmholtz, stats)

local_factored_source_alignment_batch!(out, source, tmp, gbuf, phis, thetas, U, V, P, lamb_helmholtz::Val; stats=nothing) =
    _factored_source_alignment_batch!(out, source, tmp, gbuf, phis, thetas, U, V, P, lamb_helmholtz, stats)

multipole_factored_return_alignment_batch!(target, source, tmp, gbuf, phis, thetas, U, V, P, lamb_helmholtz::Val; stats=nothing) =
    _factored_return_alignment_batch!(target, source, tmp, gbuf, phis, thetas, U, V, P, lamb_helmholtz, stats)

local_factored_return_alignment_batch!(target, source, tmp, gbuf, phis, thetas, U, V, P, lamb_helmholtz::Val; stats=nothing) =
    _factored_return_alignment_batch!(target, source, tmp, gbuf, phis, thetas, U, V, P, lamb_helmholtz, stats)

# Physical-subspace check for the factored path (016b, watch item 2). Returns
# `true` when every column's m=0 imaginary row is (numerically) zero, i.e. the
# input lies on the physical subspace where the rank-1 mode factorization matches
# production. `source` is the batched `[re/im, channel, harmonic_index, column]`
# layout; only the active channels (1 for `Val(false)`, 2 for `Val(true)`) are
# checked. This is intentionally a whole-buffer scan, NOT an inner-loop guard.
function _factored_input_is_physical(source, P, lamb_helmholtz::Val{LH}; atol=nothing) where LH
    TF = real(eltype(source))
    tol = atol === nothing ? sqrt(eps(TF)) : atol
    nchan = LH ? 2 : 1
    ncol = size(source, 4)
    @inbounds for j in 1:ncol, ch in 1:nchan, n in 0:P
        if abs(source[2, ch, harmonic_index(n, 0), j]) > tol
            return false
        end
    end
    return true
end

# Debug-gated assertion wrapper. OFF by default (`DEBUG[] == false`) and zero-cost
# in production; flip `FastMultipole.DEBUG[] = true` to arm it. Intended to be
# wired in at the 023 integration boundary on factored-path inputs so a
# non-physical m=0 imaginary component fails loudly instead of silently diverging.
@inline function _assert_factored_input_physical(source, P, lamb_helmholtz::Val)
    DEBUG[] || return nothing
    _factored_input_is_physical(source, P, lamb_helmholtz) || error(
        "factored-path input is non-physical: a nonzero m=0 imaginary component " *
        "was found. The FactoredRotation* operators are exact only on the physical " *
        "subspace (m=0 imag == 0); use a MaterializedYRotation* operator for such input.")
    return nothing
end

#------- NATIVE FLAT FACTORED ROTATION (Matrix Operator Refactor, task 017) -------#
#
# Flat ragged-buffer counterparts of the factored y-rotation stages. They consume
# FlatCoefficientBuffer storage directly: the φ channel is rotated through P_phi and
# the χ channel through P_active (Val(true)), with no φ padding. The arithmetic is
# identical to the [2,2,nh,B] kernels above (kept as the parity reference); only the
# indexing moves to flat_basis_index over per-channel matrices. Rotations are
# block-diagonal in degree n, so φ through P_phi is self-contained — the ragged φ
# matrix needs no rows above P_phi.

function apply_z_rotation_batch_flat!(out::FlatCoefficientBuffer{TF,A,B,LH}, in::FlatCoefficientBuffer, phis, ::Val{:overwrite}) where {TF,A,B,LH}
    P_phi = out.basis_info.orders.P_phi
    P_active = out.basis_info.orders.P_active
    nbatch = length(phis)
    op = phi_slab(out); ip = phi_slab(in)
    oc = chi_slab(out); ic = chi_slab(in)
    @inbounds for j in 1:nbatch
        sϕ, cϕ = sincos(phis[j])
        cm, sm = one(cϕ), zero(cϕ)
        for m in 0:P_active
            for n in m:P_phi
                fr = flat_basis_index(n, m, 1)
                a, b = ip[fr, j], ip[fr + 1, j]
                op[fr, j] = cm * a - sm * b
                op[fr + 1, j] = sm * a + cm * b
            end
            if LH
                for n in m:P_active
                    fr = flat_basis_index(n, m, 1)
                    a, b = ic[fr, j], ic[fr + 1, j]
                    oc[fr, j] = cm * a - sm * b
                    oc[fr + 1, j] = sm * a + cm * b
                end
            end
            cm_old = cm
            cm = cm_old * cϕ - sm * sϕ
            sm = cm_old * sϕ + sm * cϕ
        end
    end
    return out
end

function apply_z_rotation_batch_flat!(out::FlatCoefficientBuffer{TF,A,B,LH}, in::FlatCoefficientBuffer, phis, ::Val{:accumulate}) where {TF,A,B,LH}
    P_phi = out.basis_info.orders.P_phi
    P_active = out.basis_info.orders.P_active
    nbatch = length(phis)
    op = phi_slab(out); ip = phi_slab(in)
    oc = chi_slab(out); ic = chi_slab(in)
    @inbounds for j in 1:nbatch
        sϕ, cϕ = sincos(phis[j])
        cm, sm = one(cϕ), zero(cϕ)
        for m in 0:P_active
            for n in m:P_phi
                fr = flat_basis_index(n, m, 1)
                a, b = ip[fr, j], ip[fr + 1, j]
                op[fr, j] += cm * a + sm * b
                op[fr + 1, j] += -sm * a + cm * b
            end
            if LH
                for n in m:P_active
                    fr = flat_basis_index(n, m, 1)
                    a, b = ic[fr, j], ic[fr + 1, j]
                    oc[fr, j] += cm * a + sm * b
                    oc[fr + 1, j] += -sm * a + cm * b
                end
            end
            cm_old = cm
            cm = cm_old * cϕ - sm * sϕ
            sm = cm_old * sϕ + sm * cϕ
        end
    end
    return out
end

# Apply the genuinely factored y stage over one channel matrix (degrees 0:P) for a
# batch of columns. Mirrors `_factored_y_batch!` exactly, flat-indexed; the same
# fixed `U`/`V` modes are reused for φ (0:P_phi) and χ (0:P_active).
function _factored_y_channel_flat!(out_slab, in_slab, U, V, gbuf, thetas, P, nbatch)
    TF = real(eltype(U))
    @inbounds for j in 1:nbatch
        θ = thetas[j]
        for n in 0:P
            d = 2n + 1
            off = ymode_offset(n)
            for νidx in 1:d
                gr = zero(TF); gi = zero(TF)
                for k in 1:d
                    ri, m = _ymode_dof_to_storage(k)
                    xk = in_slab[flat_basis_index(n, m, ri), j]
                    v = V[off + (k - 1) * d + νidx]
                    gr += real(v) * xk
                    gi += imag(v) * xk
                end
                ν = νidx - n - 1
                s, c = sincos(ν * θ)
                gbuf[νidx] = complex(c * gr - s * gi, s * gr + c * gi)
            end
            for r in 1:d
                acc = zero(TF)
                for νidx in 1:d
                    u = U[off + (νidx - 1) * d + r]
                    g = gbuf[νidx]
                    acc += real(u) * real(g) - imag(u) * imag(g)
                end
                ri, m = _ymode_dof_to_storage(r)
                out_slab[flat_basis_index(n, m, ri), j] = acc
            end
        end
    end
    return out_slab
end

function _factored_y_batch_flat!(out::FlatCoefficientBuffer{TF,A,B,LH}, source::FlatCoefficientBuffer, U, V, gbuf, thetas, stats) where {TF,A,B,LH}
    P_phi = out.basis_info.orders.P_phi
    P_active = out.basis_info.orders.P_active
    nbatch = length(thetas)
    @boundscheck length(gbuf) >= 2 * P_active + 1 || throw(ArgumentError("gbuf must have length >= 2*P_active+1"))
    out_phi = phi_slab(out)
    out_chi = chi_slab(out)
    source_phi = phi_slab(source)
    source_chi = chi_slab(source)
    out_phi .= zero(eltype(out_phi))
    LH && (out_chi .= zero(eltype(out_chi)))
    _maybe_count_fixed_swap!(stats)   # forward V swap (fixed, batch-shared)
    _maybe_count_z_theta!(stats)      # diagonal e^{iνθ}
    _factored_y_channel_flat!(out_phi, source_phi, U, V, gbuf, thetas, P_phi, nbatch)
    LH && _factored_y_channel_flat!(out_chi, source_chi, U, V, gbuf, thetas, P_active, nbatch)
    _maybe_count_fixed_swap!(stats)   # back U swap (fixed, batch-shared)
    return out
end

function _factored_source_alignment_batch_flat!(out, source, tmp, gbuf, phis, thetas, U, V, lamb_helmholtz::Val{LH}, stats) where LH
    _maybe_count_z_phi!(stats)
    apply_z_rotation_batch_flat!(tmp, source, phis, Val(:overwrite))
    _factored_y_batch_flat!(out, tmp, U, V, gbuf, thetas, stats)
    return out
end

function _factored_return_alignment_batch_flat!(target, source, tmp, gbuf, phis, thetas, U, V, lamb_helmholtz::Val{LH}, stats) where LH
    _factored_y_batch_flat!(tmp, source, U, V, gbuf, thetas, stats)
    _maybe_count_z_phi!(stats)
    apply_z_rotation_batch_flat!(target, tmp, phis, Val(:accumulate))
    return target
end

# Public flat stage API (multipole vs local selected by the U/V modes passed).
multipole_factored_source_alignment_batch_flat!(out, source, tmp, gbuf, phis, thetas, U, V, lamb_helmholtz::Val; stats=nothing) =
    _factored_source_alignment_batch_flat!(out, source, tmp, gbuf, phis, thetas, U, V, lamb_helmholtz, stats)

local_factored_source_alignment_batch_flat!(out, source, tmp, gbuf, phis, thetas, U, V, lamb_helmholtz::Val; stats=nothing) =
    _factored_source_alignment_batch_flat!(out, source, tmp, gbuf, phis, thetas, U, V, lamb_helmholtz, stats)

multipole_factored_return_alignment_batch_flat!(target, source, tmp, gbuf, phis, thetas, U, V, lamb_helmholtz::Val; stats=nothing) =
    _factored_return_alignment_batch_flat!(target, source, tmp, gbuf, phis, thetas, U, V, lamb_helmholtz, stats)

local_factored_return_alignment_batch_flat!(target, source, tmp, gbuf, phis, thetas, U, V, lamb_helmholtz::Val; stats=nothing) =
    _factored_return_alignment_batch_flat!(target, source, tmp, gbuf, phis, thetas, U, V, lamb_helmholtz, stats)

# Flat physical-subspace check / guard (016b watch item 2), FlatCoefficientBuffer
# form: scans the m=0 imaginary rows of φ (0:P_phi) and χ (0:P_active). Wired in at
# the 023 integration boundary; OFF by default in production.
function _factored_input_is_physical(source::FlatCoefficientBuffer{TF,A,B,LH}; atol=nothing) where {TF,A,B,LH}
    rt = real(TF)
    tol = atol === nothing ? sqrt(eps(rt)) : atol
    P_phi = source.basis_info.orders.P_phi
    P_active = source.basis_info.orders.P_active
    ph = phi_slab(source)
    @inbounds for j in 1:size(ph, 2), n in 0:P_phi
        abs(ph[flat_basis_index(n, 0, 2), j]) > tol && return false
    end
    if LH
        ch = chi_slab(source)
        @inbounds for j in 1:size(ch, 2), n in 0:P_active
            abs(ch[flat_basis_index(n, 0, 2), j]) > tol && return false
        end
    end
    return true
end

@inline function _assert_factored_input_physical(source::FlatCoefficientBuffer)
    DEBUG[] || return nothing
    _factored_input_is_physical(source) || error(
        "factored-path input is non-physical: a nonzero m=0 imaginary component " *
        "was found. The FactoredRotation* operators are exact only on the physical " *
        "subspace (m=0 imag == 0); use a MaterializedYRotation* operator for such input.")
    return nothing
end
