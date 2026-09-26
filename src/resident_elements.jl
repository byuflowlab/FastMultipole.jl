#=
Element body-to-multipole for the resident radix lifecycle, shared by the host
and the KernelAbstractions device path.

The legacy `bodytomultipole.jl` routines (`calculate_q!`, `calculate_pj!`,
`source_to_dipole!`, `mirrored_source_to_vortex!`) run recurrences over the whole
(n, m) triangle of a body's harmonics, which a device kernel cannot do
per-coefficient the way the point kernels do. They are ported here verbatim in
a form a kernel can run per body: no allocation, no asserts, every literal in
the working precision, and the harmonics/coefficient scratch handed in as any
3-D indexable array `[reim, slot, i]` (a host `Array` or a per-body `view` of a
device scratch). The coefficient ordering is the resident flat ordering
(`flat_basis_index(n, m, reim) = 2 (harmonic_index(n, m) − 1) + reim`), so the
result adds straight into the phi/chi slabs.

Sign convention: the resident scalar B2M omits the legacy strength negation
(see the note above `_resident_vortex_q`), so the source and dipole filament
routines here take the strength as is; the vortex routine is the legacy one.
=#

@inline function _res_get_n(h, index, n, m, i, _1_m)
    T = eltype(h)
    @inbounds begin
        if m > 0
            mm1_re = h[1, index, i - 1]; mm1_im = h[2, index, i - 1]
        elseif n == 0
            mm1_re = zero(T); mm1_im = zero(T)
        else
            mm1_re = -_1_m * h[1, index, i + 1]; mm1_im = _1_m * h[2, index, i + 1]
        end
        m_re = h[1, index, i]; m_im = h[2, index, i]
        if m < n
            mp1_re = h[1, index, i + 1]; mp1_im = h[2, index, i + 1]
        else
            mp1_re = zero(T); mp1_im = zero(T)
        end
    end
    return mm1_re, mm1_im, m_re, m_im, mp1_re, mp1_im
end

@inline function _res_get_nm1(h, index, n, m, i_nm1_m, _1_m)
    T = eltype(h)
    @inbounds begin
        if m > 0
            mm1_re = h[1, index, i_nm1_m - 1]; mm1_im = h[2, index, i_nm1_m - 1]
        elseif n > 1
            mm1_re = -_1_m * h[1, index, i_nm1_m + 1]; mm1_im = _1_m * h[2, index, i_nm1_m + 1]
        else
            mm1_re = zero(T); mm1_im = zero(T)
        end
        if m < n
            m_re = h[1, index, i_nm1_m]; m_im = h[2, index, i_nm1_m]
        else
            m_re = zero(T); m_im = zero(T)
        end
        if m + 1 < n
            mp1_re = h[1, index, i_nm1_m + 1]; mp1_im = h[2, index, i_nm1_m + 1]
        else
            mp1_re = zero(T); mp1_im = zero(T)
        end
    end
    return mm1_re, mm1_im, m_re, m_im, mp1_re, mp1_im
end

# regular harmonics of (ξ, η, z) into slot 1
@inline function _res_calculate_q!(h, ξr, ξi, ηr, ηi, z, P)
    T = eltype(h)
    @inbounds begin
        h[1, 1, 1] = one(T); h[2, 1, 1] = zero(T)
        i = 2
        for n in 1:P
            i_nm1_m = i - n
            _1_m = one(T)
            invn = inv(T(n))
            for m in 0:n
                a_re, a_im, b_re, b_im, c_re, c_im = _res_get_nm1(h, 1, n, m, i_nm1_m, _1_m)
                h[1, 1, i] = (-(ξr * a_im + ξi * a_re) - (ηr * c_im + ηi * c_re) - z * b_re) * invn
                h[2, 1, i] = ((ξr * a_re - ξi * a_im) + (ηr * c_re - ηi * c_im) - z * b_im) * invn
                i += 1; i_nm1_m += 1; _1_m = -_1_m
            end
        end
    end
    return h
end

# line integral of the regular harmonics from (ξ0, η0, z0) (slot 1 holds q at the far end) into slot 2
@inline function _res_calculate_pj!(h, ξr, ξi, ηr, ηi, z, P)
    T = eltype(h)
    @inbounds begin
        h[1, 2, 1] = one(T); h[2, 2, 1] = zero(T)
        i = 2
        for n in 1:P
            i_nm1_m = i - n
            _1_m = one(T)
            invnp1 = inv(T(n + 1))
            for m in 0:n
                a_re, a_im, b_re, b_im, c_re, c_im = _res_get_nm1(h, 2, n, m, i_nm1_m, _1_m)
                q_re = h[1, 1, i]; q_im = h[2, 1, i]
                h[1, 2, i] = (-(ξr * a_im + ξi * a_re) - (ηr * c_im + ηi * c_re) - z * b_re + q_re) * invnp1
                h[2, 2, i] = ((ξr * a_re - ξi * a_im) + (ηr * c_re - ηi * c_im) - z * b_im + q_im) * invnp1
                i += 1; i_nm1_m += 1; _1_m = -_1_m
            end
        end
    end
    return h
end

# dipole harmonics from the source harmonics in slot `i_source` into slot `i_dipole`
@inline function _res_source_to_dipole!(h, i_dipole, i_source, qx, qy, qz, P)
    T = eltype(h)
    @inbounds begin
        h[1, i_dipole, 1] = zero(T); h[2, i_dipole, 1] = zero(T)
        i = 2
        for n in 1:P
            i_nm1_m = i - n
            _1_m = one(T)
            for m in 0:n
                a_re, a_im, b_re, b_im, c_re, c_im = _res_get_nm1(h, i_source, n, m, i_nm1_m, _1_m)
                h[1, i_dipole, i] = -qx * T(0.5) * (c_im + a_im) + qy * T(0.5) * (c_re - a_re) - qz * b_re
                h[2, i_dipole, i] = qx * T(0.5) * (c_re + a_re) + qy * T(0.5) * (c_im - a_im) - qz * b_im
                i += 1; i_nm1_m += 1; _1_m = -_1_m
            end
        end
    end
    return h
end

# vortex phi (slot 1 of coef) and chi (slot 2) from the mirrored source harmonics in slot `i_source`
@inline function _res_mirrored_source_to_vortex!(coef, h, vx, vy, vz, i_source, multiplier, P)
    T = eltype(coef)
    @inbounds begin
        i = 1
        for n in 0:P
            _1_m = one(T)
            _1_np1 = inv(T(n + 1))
            for m in 0:n
                a_re, a_im, b_re, b_im, c_re, c_im = _res_get_n(h, i_source, n, m, i, _1_m)
                nmmp1_2 = T(n - m + 1) * T(0.5)
                npmp1_2 = T(n + m + 1) * T(0.5)
                coef[1, 1, i] += multiplier * _1_m * ((-vx * a_re + vy * a_im) * nmmp1_2 + (vx * c_re + vy * c_im) * npmp1_2 - vz * T(m) * b_im) * _1_np1
                coef[2, 1, i] += multiplier * _1_m * ((vx * a_im + vy * a_re) * nmmp1_2 + (-vx * c_im + vy * c_re) * npmp1_2 - vz * T(m) * b_re) * _1_np1
                i += 1; _1_m = -_1_m
            end
        end
        i = 2
        for n in 1:P
            i_nm1_m = i - n
            _1_m = one(T)
            _1_over_n = inv(T(n))
            for m in 0:n
                a_re, a_im, b_re, b_im, c_re, c_im = _res_get_nm1(h, i_source, n, m, i_nm1_m, _1_m)
                coef[1, 2, i] -= multiplier * _1_m * _1_over_n * (T(0.5) * (-vy * a_re - vx * a_im + vy * c_re - vx * c_im) - vz * b_re)
                coef[2, 2, i] -= multiplier * _1_m * _1_over_n * (T(0.5) * (vy * a_im - vx * a_re - vy * c_im - vx * c_re) + vz * b_im)
                i += 1; i_nm1_m += 1; _1_m = -_1_m
            end
        end
    end
    return coef
end

@inline _res_xyz_to_ξηz(x, y, z) = (x * oftype(x, 0.5), y * oftype(y, 0.5), x * oftype(x, 0.5), -y * oftype(y, 0.5), z)

# scalar (slot 1 of h) coefficients into coef slot 1 with the resident sign pattern
@inline function _res_accumulate_scalar!(coef, h, slot, scale, P)
    T = eltype(coef)
    @inbounds begin
        i = 1
        _1_n = one(T)
        for n in 0:P
            _1_n_m = _1_n
            for m in 0:n
                coef[1, 1, i] += scale * _1_n_m * h[1, slot, i]
                coef[2, 1, i] -= scale * _1_n_m * h[2, slot, i]
                i += 1; _1_n_m = -_1_n_m
            end
            _1_n = -_1_n
        end
    end
    return coef
end

"""
    _res_filament_b2m!(BT, coef, h, x0, xu, strength, P)

Multipole coefficients of one straight filament about the expansion center:
`x0` is vertex 1 relative to the center, `xu = x2 - x1`, `strength` the packed
strength components (1 for `Filament{Source}`, 3 otherwise). `coef` is
`[reim, phi/chi, i]` and is accumulated into; `h` is `[reim, 2, nh]` scratch
with `nh >= harmonic_index(P, P)`. Mirrors `body_to_multipole_filament!`.
"""
@inline function _res_filament_b2m!(::Type{<:Filament{Source}}, coef, h, x0, xu, strength, P)
    T = eltype(coef)
    ξ0r, ξ0i, η0r, η0i, z0 = _res_xyz_to_ξηz(x0[1], x0[2], x0[3])
    ξur, ξui, ηur, ηui, zu = _res_xyz_to_ξηz(xu[1], xu[2], xu[3])
    _res_calculate_q!(h, ξ0r + ξur, ξ0i + ξui, η0r + ηur, η0i + ηui, z0 + zu, P)
    _res_calculate_pj!(h, ξ0r, ξ0i, η0r, η0i, z0, P)
    L = sqrt(xu[1] * xu[1] + xu[2] * xu[2] + xu[3] * xu[3])
    return _res_accumulate_scalar!(coef, h, 2, L * T(strength[1]), P)
end

@inline function _res_filament_b2m!(::Type{<:Filament{Dipole}}, coef, h, x0, xu, strength, P)
    T = eltype(coef)
    ξ0r, ξ0i, η0r, η0i, z0 = _res_xyz_to_ξηz(x0[1], x0[2], x0[3])
    ξur, ξui, ηur, ηui, zu = _res_xyz_to_ξηz(xu[1], xu[2], xu[3])
    _res_calculate_q!(h, ξ0r + ξur, ξ0i + ξui, η0r + ηur, η0i + ηui, z0 + zu, P)
    _res_calculate_pj!(h, ξ0r, ξ0i, η0r, η0i, z0, P)
    _res_source_to_dipole!(h, 1, 2, T(strength[1]), T(strength[2]), T(strength[3]), P)   # dipole into slot 1
    L = sqrt(xu[1] * xu[1] + xu[2] * xu[2] + xu[3] * xu[3])
    return _res_accumulate_scalar!(coef, h, 1, L, P)
end

@inline function _res_filament_b2m!(::Type{<:Filament{Vortex}}, coef, h, x0, xu, strength, P)
    T = eltype(coef)
    ξ0r, ξ0i, η0r, η0i, z0 = _res_xyz_to_ξηz(-x0[1], -x0[2], -x0[3])
    ξur, ξui, ηur, ηui, zu = _res_xyz_to_ξηz(-xu[1], -xu[2], -xu[3])
    _res_calculate_q!(h, ξ0r + ξur, ξ0i + ξui, η0r + ηur, η0i + ηui, z0 + zu, P)
    _res_calculate_pj!(h, ξ0r, ξ0i, η0r, η0i, z0, P)
    L = sqrt(xu[1] * xu[1] + xu[2] * xu[2] + xu[3] * xu[3])
    return _res_mirrored_source_to_vortex!(coef, h, T(strength[1]), T(strength[2]), T(strength[3]), 2, L, P)
end

# rows of harmonics scratch a body of order P needs (the legacy allocates P+2)
@inline _res_element_harmonics_rows(P) = harmonic_index(P + 2, P + 2)

# A body's slice of a 4-D per-body scratch `[reim, slot, i, body]`, indexed like
# the 3-D host arrays the recurrences take. An isbits wrapper, so a device
# kernel can hand a body's slice to the shared recurrences without a view.
struct ResBodySlice{A}
    a::A
    k::Int
end
Base.eltype(::ResBodySlice{A}) where A = eltype(A)
Base.@propagate_inbounds Base.getindex(s::ResBodySlice, reim, slot, i) = s.a[reim, slot, i, s.k]
Base.@propagate_inbounds Base.setindex!(s::ResBodySlice, v, reim, slot, i) = (s.a[reim, slot, i, s.k] = v)

# packed strength of body `k` as an SVector; a function, not a closure, so a
# kernel's reassigned loop counter is never captured (and boxed)
@inline function _res_packed_strength(sb, k, ::Val{SD}) where SD
    return SVector{SD,eltype(sb)}(ntuple(i -> (@inbounds sb[4 + i, k]), Val(SD)))
end

# area integral of the regular harmonics (slot 1) from the line integral j in
# slot 2, as `calculate_ib!`
@inline function _res_calculate_ib!(h, ξr, ξi, ηr, ηi, z, P)
    T = eltype(h)
    @inbounds begin
        h[1, 1, 1] = T(0.5); h[2, 1, 1] = zero(T)
        i = 2
        for n in 1:P
            i_nm1_m = i - n
            _1_m = one(T)
            invnp2 = inv(T(n + 2))
            for m in 0:n
                a_re, a_im, b_re, b_im, c_re, c_im = _res_get_nm1(h, 1, n, m, i_nm1_m, _1_m)
                j_re = h[1, 2, i]; j_im = h[2, 2, i]
                h[1, 1, i] = (-(ξr * a_im + ξi * a_re) - (ηr * c_im + ηi * c_re) - z * b_re + j_re) * invnp2
                h[2, 1, i] = ((ξr * a_re - ξi * a_im) + (ηr * c_re - ηi * c_im) - z * b_im + j_im) * invnp2
                i += 1; i_nm1_m += 1; _1_m = -_1_m
            end
        end
    end
    return h
end

# q at x0+xu (slot 1), j at x0+xv (slot 2), then i at x0 (slot 1): the
# triangle's area harmonics, as the legacy panel routines compute them
@inline function _res_panel_harmonics!(h, x0, xu, xv, P, sign)
    ξ0r, ξ0i, η0r, η0i, z0 = _res_xyz_to_ξηz(sign * x0[1], sign * x0[2], sign * x0[3])
    ξur, ξui, ηur, ηui, zu = _res_xyz_to_ξηz(sign * xu[1], sign * xu[2], sign * xu[3])
    ξvr, ξvi, ηvr, ηvi, zv = _res_xyz_to_ξηz(sign * xv[1], sign * xv[2], sign * xv[3])
    _res_calculate_q!(h, ξ0r + ξur, ξ0i + ξui, η0r + ηur, η0i + ηui, z0 + zu, P)
    _res_calculate_pj!(h, ξ0r + ξvr, ξ0i + ξvi, η0r + ηvr, η0i + ηvi, z0 + zv, P)
    _res_calculate_ib!(h, ξ0r, ξ0i, η0r, η0i, z0, P)
    return h
end

@inline function _res_tri_area_normal(xu, xv)
    nx = xu[2] * xv[3] - xu[3] * xv[2]
    ny = xu[3] * xv[1] - xu[1] * xv[3]
    nz = xu[1] * xv[2] - xu[2] * xv[1]
    J = sqrt(nx * nx + ny * ny + nz * nz)          # twice the area
    return J, nx / J, ny / J, nz / J
end

"""
    _res_panel_b2m!(BT, coef, h, x0, xu, xv, strength, P)

Multipole coefficients of one planar triangle about the expansion center: `x0`
is vertex 1 relative to the center, `xu = x2 - x1`, `xv = x3 - x1`. Mirrors
`body_to_multipole_panel!`; the source and dipole strength negation is omitted
(resident convention). A `Panel{3,Dipole}` strength is the dipole density along
the unit normal of (xu, xv); a `Panel{3,Vortex}` strength is the sheet vorticity
vector.
"""
@inline function _res_panel_b2m!(::Type{<:Panel{3,Source}}, coef, h, x0, xu, xv, strength, P)
    T = eltype(coef)
    _res_panel_harmonics!(h, x0, xu, xv, P, one(T))
    J, _, _, _ = _res_tri_area_normal(xu, xv)
    return _res_accumulate_scalar!(coef, h, 1, J * T(strength[1]), P)
end

@inline function _res_panel_b2m!(::Type{<:Panel{3,Dipole}}, coef, h, x0, xu, xv, strength, P)
    T = eltype(coef)
    _res_panel_harmonics!(h, x0, xu, xv, P, one(T))
    J, nx, ny, nz = _res_tri_area_normal(xu, xv)
    mu = T(strength[1])
    _res_source_to_dipole!(h, 2, 1, nx * mu, ny * mu, nz * mu, P)      # dipole into slot 2
    return _res_accumulate_scalar!(coef, h, 2, J, P)
end

@inline function _res_panel_b2m!(::Type{<:Panel{3,SourceDipole}}, coef, h, x0, xu, xv, strength, P)
    T = eltype(coef)
    _res_panel_b2m!(Panel{3,Source}, coef, h, x0, xu, xv, (T(strength[1]),), P)
    _res_panel_b2m!(Panel{3,Dipole}, coef, h, x0, xu, xv, (T(strength[2]),), P)
    return coef
end

@inline function _res_panel_b2m!(::Type{<:Panel{3,Vortex}}, coef, h, x0, xu, xv, strength, P)
    T = eltype(coef)
    _res_panel_harmonics!(h, x0, xu, xv, P, -one(T))                   # mirrored geometry
    J, _, _, _ = _res_tri_area_normal(xu, xv)
    return _res_mirrored_source_to_vortex!(coef, h, T(strength[1]), T(strength[2]), T(strength[3]), 1, J, P)
end

"""
    _res_element_b2m!(BT, coef, h, source_bodies, k, cx, cy, cz, P, ::Val{SD})

Read body `k` of the packed buffer (position rows 1:3, strength rows 5:4+SD,
vertices from row 5+SD) and accumulate its expansion about the center
`(cx, cy, cz)` into `coef`. One entry point for every element type, used by
the host and the device B2M alike.
"""
@inline function _res_element_b2m!(::Type{BT}, coef, h, sb, k, cx, cy, cz, P, ::Val{SD}) where {BT<:Filament,SD}
    T = eltype(coef)
    v1 = 5 + SD
    @inbounds begin
        x0 = SVector{3,T}(sb[v1, k] - cx, sb[v1 + 1, k] - cy, sb[v1 + 2, k] - cz)
        xu = SVector{3,T}(sb[v1 + 3, k] - sb[v1, k], sb[v1 + 4, k] - sb[v1 + 1, k], sb[v1 + 5, k] - sb[v1 + 2, k])
    end
    return _res_filament_b2m!(BT, coef, h, x0, xu, _res_packed_strength(sb, k, Val(SD)), P)
end

@inline function _res_element_b2m!(::Type{BT}, coef, h, sb, k, cx, cy, cz, P, ::Val{SD}) where {BT<:Panel,SD}
    T = eltype(coef)
    v1 = 5 + SD
    @inbounds begin
        x0 = SVector{3,T}(sb[v1, k] - cx, sb[v1 + 1, k] - cy, sb[v1 + 2, k] - cz)
        xu = SVector{3,T}(sb[v1 + 3, k] - sb[v1, k], sb[v1 + 4, k] - sb[v1 + 1, k], sb[v1 + 5, k] - sb[v1 + 2, k])
        xv = SVector{3,T}(sb[v1 + 6, k] - sb[v1, k], sb[v1 + 7, k] - sb[v1 + 1, k], sb[v1 + 8, k] - sb[v1 + 2, k])
    end
    return _res_panel_b2m!(BT, coef, h, x0, xu, xv, _res_packed_strength(sb, k, Val(SD)), P)
end
