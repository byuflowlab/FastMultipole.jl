#------- EXPLICIT Z-ROTATION OPERATORS -------#
#
# The z-rotation is block-diagonal in the azimuthal order m: each stored
# coefficient (n, m) is multiplied by the complex phase e^{imϕ}, i.e. the real
# 2x2 rotation block acting on the [real, imag] pair.
#
# Storage-light representation: two diagonal vectors over the compressed
# harmonic index, C[i] = cos(m(i)ϕ) and S[i] = sin(m(i)ϕ), where
# i = harmonic_index(n, m). Forward rotation overwrites the destination with
# e^{imϕ}; inverse/back rotation accumulates into the destination with the
# conjugate phase e^{-imϕ}. The m = 0 block is the identity (C = 1, S = 0).
#
# These operate on the octree path's coefficient layout
# weights[real_or_imag, component, harmonic_index] so that behavior is
# bit-for-bit identical to rotate_z! / back_rotate_z! in src/rotate.jl.

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

#------- INVARIANT AXIS-SWAP OPERATORS -------#
#
# The non-z part of the y-alignment used by the rotation-trick M2M/M2L/L2L is, for
# every degree n, generated entirely from the fixed π/2 Wigner blocks H(π/2). The
# angle-dependent y-rotation block factors as
#
#     T_n(θ) = S_n * Z_n(θ) * S_n^{-1}
#
# where S_n is built
# only from H(π/2) and Z_n(θ) carries the exp(i ν θ) phases. Production rebuilds
# the full T matrix (Ts) on EVERY translation call via update_Ts! in src/rotate.jl;
# its inner ν loop multiplies two angle-independent H(π/2) products by a scalar and
# the angle-dependent cos/sin(ν θ). Profiling found this per-call rebuild to be
# the dominant cost.
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
# to the existing apply kernels of src/rotate.jl, reused unchanged.
# These operate on the production layout weights[real_or_imag, component,
# harmonic_index]; the FlatCoefficientBuffer forms are further below. The functions are
# intentionally internal/non-exported.

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

    _1_n = -one(TF)
    @inbounds for n in 1:P
        H_π2 = get_H(Hs_π2, n)
        base = S_block_offset(n)
        np1 = n + 1

        for m in 0:n
            H_π2_n_m_0 = H_π2[H_index(0, m)]
            _1_mp_odd = one(TF)
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
    build_Ts_from_S!(Ts, S_pos, S_neg, β, P, trig)

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
sequence and same accumulation order) while cutting the inner-loop work ~2-3×. The
caller owns `trig` (length `>= 2P`), so the call is allocation-free.
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

"""
    rotate_multipole_y_op!(out, source, Ts, S_pos, S_neg, ζs_mag, β, P, lamb_helmholtz, trig)

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

"""
    back_rotate_local_y_op!(target, source, Ts, Hs_π2, S_pos, S_neg, ηs_mag, β, P, lamb_helmholtz, trig)

Back local y-alignment built from the cached axis-swap blocks; the local kernel
(`_rotate_local_y!`) uses the `η` sign table. Resets `target` (not an
accumulating inverse). Matches `back_rotate_local_y!`.
"""
function back_rotate_local_y_op!(target, source, Ts, Hs_π2, S_pos, S_neg, ηs_mag, β, P, lamb_helmholtz::Val{LH}, trig) where LH
    build_Ts_from_S!(Ts, S_pos, S_neg, β, P, trig)
    _rotate_local_y!(target, source, Ts, Hs_π2, ηs_mag, P, lamb_helmholtz)
    return target
end

#------- GLOBALLY BATCHED FACTORED ROTATION ALIGNMENT -------#
#
# Genuinely factored y-rotation. For every degree n the production y-operator factors
# as  Y_n(θ) = U_n · diag(e^{iνθ}) · V_n  (ν = -n..n), with U_n / V_n FIXED (angle- and
# geometry-independent) per-degree matrices and the only θ dependence the cheap diagonal
# e^{iνθ}. This is the S_n · Z_n(θ) · S_n^{-1} form above
# realized as two batch-shared fixed swaps around a per-column z-rotation: forward swap
# V (apply once per batch), diagonal e^{iνθ_j} (per column), back swap U. Cost is
# O(P^3) per column with batch-shared fixed matrices — NOT the per-call materialized
# Ts(θ) rebuild, and NOT the per-(n,m,mp) Σ_ν S·trig(νθ) contraction (which is the
# same O(P^4) materialized arithmetic in disguise).
#
# The fixed modes are obtained at cache build by sampling the production-parity y
# kernels and rank-1-factoring each angular Fourier component of Y_n (every component is
# rank 1); see update_factored_y_modes!. The ζ (multipole) and η (local) paths get
# separate U/V (the dressing is baked into the fixed modes), so the staged apply is
# identical for both and is selected purely by which modes the caller passes. A ζ-dressed
# fixed ±π/2 swap cannot stand in for these modes: (ζS)·Z·(ζS⁻¹) ≠ ζ·(S·Z·S⁻¹), since ζ
# does not commute through the swap.

# Per-degree fixed mode matrices U_n, V_n (each (2n+1)x(2n+1) complex). Y_n(θ) =
# U_n diag(e^{iνθ}) V_n, with every Fourier component of Y_n rank 1, so U_n holds the
# left mode vectors (columns, index by νidx = ν+n+1) and V_n the right mode covectors
# (rows). Flat column-major storage per degree; offsets below.
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
(`_factored_y_degree_major_auto!`, src/translate_batched.jl), an `O(P^3)`-per-column, batch-shared GEMM shape that never
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

# Flat physical-subspace check / guard, FlatCoefficientBuffer
# form: scans the m=0 imaginary rows of φ (0:P_phi) and χ (0:P_active). Called on
# the upward-pass output of the resident lifecycle; active only when DEBUG[] is set.
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
