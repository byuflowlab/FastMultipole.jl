#------- B2M (body -> multipole) -------#
#
# One workgroup per leaf cell, body-parallel accumulation, tree-reduced across
# the group.
#
# The per-body math is shared, not reimplemented -- `_resident_vortex_phi_contrib`
# and `_resident_vortex_chi_contrib` live in src/resident/resident_b2m.jl and
# are backend-agnostic `@inline` Julia shared with the CPU path; only the
# reduction shape is device-specific (the vortex kernel's harmonic walk is the
# exception, see below).
#
# Workgroup size is a fixed 128, not the backend's auto size used by the
# elementwise kernels: the reduction is a halving tree, so its summation ORDER
# depends on the group size, and pinning it keeps results identical across
# backends rather than merely close. WG must be a power of two.
#
# Metal portability trap: the `@localmem` element type must be a compile-time
# constant reaching the `Val`-wrapped `SharedMemory` call, and computing it
# inside the kernel as `TF = eltype(phi)` does NOT qualify -- on Metal that
# compiles but raises a device-side "undefined variable error" at launch (KA's
# `@localmem` expansion). Pass the element type as a
# `::Type{TF}` kernel argument instead. `Val{WG}` dims are fine either way.
#
# Float32 discipline: every literal stays in TF. Apple GPUs reject Float64
# outright, and a stray `0.5` would silently promote and break Metal, so the
# accumulators are seeded with `zero(TF)`.

# The halving tree reduction is written out lexically in both places rather
# than factored into a helper: `@synchronize` must appear directly in the
# kernel body and in uniform control flow, so it can neither live inside a
# called function nor sit under a per-group `if`. (`ndrange = ncell * WG` gives
# exactly `ncell` groups, so no group guard is needed either.)

# B2M's three-in-one harmonic walk, KA-only.
#
# `_resident_vortex_phi_contrib` asks `_resident_vortex_q` for THREE adjacent-m
# coefficients at one n, and each of those restarts the full O(P^2) recurrence
# in `_resident_regular_harmonic_coeff`. Same for the chi contribution. With the
# (n,m) loop outside the body loop, a body pays 189 restarts per B2M call, and
# an ablation on Metal put that recurrence at 88.5% of the stage.
#
# One walk reaches all three columns: they differ only in m, and the recurrence
# is m-outer/n-inner, so it passes through m-1, m and m+1 on its way. That makes
# the stage cost one restart per (n,m) instead of three -- and the arithmetic
# reaching each captured value is op-for-op the shared function's, so the
# coefficients are BIT-IDENTICAL, not merely close. The reduction and the order
# bodies accumulate in are untouched too, so this kernel should stay bit-exact
# against the CPU path.
#
# KA-ONLY BY CONSTRUCTION: `_resident_vortex_*_contrib` and
# `_resident_regular_harmonic_coeff` in src/ are left alone, so the CPU host
# kernel keeps its arithmetic. The duplicated math below must stay in
# lockstep with src/resident/resident_b2m.jl.

# R_{nt,mt-1}, R_{nt,mt}, R_{nt,mt+1} from a single recurrence walk, with the
# legacy negative-m conjugate rule of `_resident_vortex_q` folded in. Columns
# outside 0:nt come back zero, exactly as `_resident_vortex_q`'s bound checks do.
@inline function ka_vortex_q3(setup::NTuple{5,TF}, nt_::Integer, mt_::Integer) where TF
    # Counters in the field's integer width (see _ka_int_type): Int64 arithmetic
    # and Int64->Float32 conversion are emulated on Metal and dominated this walk.
    IT = _ka_int_type(TF)
    nt = IT(nt_); mt = IT(mt_); i1 = one(IT)
    rho, xc, ys, iei_re, iei_im = setup
    z = zero(TF)
    a_re = z; a_im = z   # column mt-1
    b_re = z; b_im = z   # column mt
    c_re = z; c_im = z   # column mt+1
    if rho == z
        # coincident point: R_{0,0} = 1, everything else zero
        if nt == 0
            mt == 0 && (b_re = one(TF))
            mt == 1 && (a_re = one(TF))
        end
    else
        @inbounds begin
            fact = one(TF); pn = one(TF); rhom = one(TF)
            ieim_re = one(TF); ieim_im = z
            mhi = min(mt + i1, nt)
            m = zero(IT)
            while m <= mhi
                p = pn
                rmp = rhom * p
                if m == nt
                    vr = rmp * ieim_re; vi = rmp * ieim_im
                    if m == mt - i1
                        a_re = vr; a_im = vi
                    elseif m == mt
                        b_re = vr; b_im = vi
                    elseif m == mt + i1
                        c_re = vr; c_im = vi
                    end
                end
                p1 = p
                p = xc * TF(m + m + i1) * p1
                rhom *= rho
                rhon = rhom
                n = m + i1
                while n <= nt
                    rhon /= -TF(n + m)
                    rnp = rhon * p
                    if n == nt
                        vr = rnp * ieim_re; vi = rnp * ieim_im
                        if m == mt - i1
                            a_re = vr; a_im = vi
                        elseif m == mt
                            b_re = vr; b_im = vi
                        elseif m == mt + i1
                            c_re = vr; c_im = vi
                        end
                    end
                    p2 = p1; p1 = p
                    p = (xc * TF(n + n + i1) * p1 - TF(n + m) * p2) / TF(n - m + i1)
                    rhon *= rho
                    n += i1
                end
                rhom /= -TF(m + m + i1 + i1) * TF(m + m + i1)
                pn = -pn * fact * ys
                fact += TF(2)
                tre = ieim_re
                ieim_re = tre * iei_re - ieim_im * iei_im
                ieim_im = tre * iei_im + ieim_im * iei_re
                m += i1
            end
        end
    end
    if mt == 0
        # `_resident_vortex_q`: Q_{n,-1} = -conj(Q_{n,1}), and zero for n < 1.
        # Column 1 is what the walk captured as `c` (it is also mt+1 here).
        if nt >= 1
            a_re = -c_re; a_im = c_im
        else
            a_re = z; a_im = z
        end
    end
    return a_re, a_im, b_re, b_im, c_re, c_im
end

# Integer width for device recurrence counters follows the field type: Int32
# with Float32 (Int64 is emulated on Metal's 32-bit ALUs), Int64 with Float64.
@inline _ka_int_type(::Type{Float32}) = Int32
@inline _ka_int_type(::Type{Float64}) = Int64
@inline _ka_int_type(::Type{T}) where T = Int

# Mirrors `_resident_vortex_phi_contrib` / `_resident_vortex_chi_contrib`, with
# the three separate `_resident_vortex_q` restarts replaced by one walk.
@inline function ka_vortex_phi_contrib(mdx, mdy, mdz, vx, vy, vz, n, m)
    TF = typeof(mdx)
    setup = FastMultipole._resident_harmonic_setup(mdx, mdy, mdz)
    qmm1_re, qmm1_im, qm_re, qm_im, qmp1_re, qmp1_im = ka_vortex_q3(setup, n, m)
    IT = _ka_int_type(TF)
    n32 = IT(n); m32 = IT(m); i1 = one(IT)
    nmmp1_2 = TF(n32 - m32 + i1) * TF(0.5)
    npmp1_2 = TF(n32 + m32 + i1) * TF(0.5)
    _1_np1 = inv(TF(n32 + i1))
    _1_m = isodd(m) ? -one(TF) : one(TF)
    re = _1_m * ((-vx * qmm1_re + vy * qmm1_im) * nmmp1_2 +
                 (vx * qmp1_re + vy * qmp1_im) * npmp1_2 - vz * TF(m32) * qm_im) * _1_np1
    im = _1_m * ((vx * qmm1_im + vy * qmm1_re) * nmmp1_2 +
                 (-vx * qmp1_im + vy * qmp1_re) * npmp1_2 - vz * TF(m32) * qm_re) * _1_np1
    return re, im
end

@inline function ka_vortex_chi_contrib(mdx, mdy, mdz, vx, vy, vz, n, m)
    TF = typeof(mdx)
    setup = FastMultipole._resident_harmonic_setup(mdx, mdy, mdz)
    qmm1_re, qmm1_im, qm_re, qm_im, qmp1_re, qmp1_im = ka_vortex_q3(setup, n - 1, m)
    _1_over_n = inv(TF(_ka_int_type(TF)(n)))
    _1_m = isodd(m) ? -one(TF) : one(TF)
    re = -_1_m * _1_over_n * (TF(0.5) * (-vy * qmm1_re - vx * qmm1_im +
        vy * qmp1_re - vx * qmp1_im) - vz * qm_re)
    im = -_1_m * _1_over_n * (TF(0.5) * (vy * qmm1_im - vx * qmm1_re -
        vy * qmp1_im - vx * qmp1_re) + vz * qm_im)
    return re, im
end

@kernel function ka_b2m_vortex_leaf_nodes_kernel!(phi, chi, @Const(source_bodies),
        @Const(cell_centers), @Const(cell_ranges), @Const(leaf_to_node),
        P_phi, P_chi, ncell, ::Type{TF}, ::Val{WG}) where {TF,WG}
    i_cell = @index(Group)
    tid = @index(Local)
    shre = @localmem TF (WG,)
    shim = @localmem TF (WG,)
    @inbounds begin
        first = cell_ranges[1, i_cell]
        count = cell_ranges[2, i_cell]
        cx = cell_centers[1, i_cell]
        cy = cell_centers[2, i_cell]
        cz = cell_centers[3, i_cell]
        node = leaf_to_node[i_cell]
        for n in 0:P_phi
            for m in 0:n
                acc_re = zero(TF); acc_im = zero(TF)
                k = first + tid - 1
                while k <= first + count - 1
                    re_, im_ = ka_vortex_phi_contrib(
                        cx - source_bodies[1, k], cy - source_bodies[2, k],
                        cz - source_bodies[3, k], source_bodies[5, k],
                        source_bodies[6, k], source_bodies[7, k], n, m)
                    acc_re += re_; acc_im += im_
                    k += WG
                end
                shre[tid] = acc_re
                shim[tid] = acc_im
                @synchronize()
                s = WG >> 1
                while s >= 1
                    if tid <= s
                        shre[tid] += shre[tid + s]
                        shim[tid] += shim[tid + s]
                    end
                    @synchronize()
                    s >>= 1
                end
                if tid == 1
                    row = FastMultipole.flat_basis_index(n, m, 1)
                    phi[row, node] = shre[1]
                    phi[row + 1, node] = shim[1]
                end
                @synchronize()
            end
        end
        for n in 1:P_chi
            for m in 0:n
                acc_re = zero(TF); acc_im = zero(TF)
                k = first + tid - 1
                while k <= first + count - 1
                    re_, im_ = ka_vortex_chi_contrib(
                        cx - source_bodies[1, k], cy - source_bodies[2, k],
                        cz - source_bodies[3, k], source_bodies[5, k],
                        source_bodies[6, k], source_bodies[7, k], n, m)
                    acc_re += re_; acc_im += im_
                    k += WG
                end
                shre[tid] = acc_re
                shim[tid] = acc_im
                @synchronize()
                s = WG >> 1
                while s >= 1
                    if tid <= s
                        shre[tid] += shre[tid + s]
                        shim[tid] += shim[tid + s]
                    end
                    @synchronize()
                    s >>= 1
                end
                if tid == 1
                    row = FastMultipole.flat_basis_index(n, m, 1)
                    chi[row, node] = shre[1]
                    chi[row + 1, node] = shim[1]
                end
                @synchronize()
            end
        end
    end
end

# Point{Source} body-to-multipole: the phi channel only, one workgroup per leaf
# cell with a body-parallel tree reduction. Mirrors `_host_b2m_kernel!`
# (src/resident/resident_b2m.jl) term for term: regular harmonics of the
# offset `x - c`, weighted by `(-1)^(n+m) q`, conjugated into the flat buffer.
@kernel function ka_b2m_source_leaf_nodes_kernel!(phi, @Const(source_bodies),
        @Const(cell_centers), @Const(cell_ranges), @Const(leaf_to_node),
        P_phi, ncell, ::Type{TF}, ::Val{WG}) where {TF,WG}
    i_cell = @index(Group)
    tid = @index(Local)
    shre = @localmem TF (WG,)
    shim = @localmem TF (WG,)
    @inbounds begin
        first = cell_ranges[1, i_cell]
        count = cell_ranges[2, i_cell]
        cx = cell_centers[1, i_cell]
        cy = cell_centers[2, i_cell]
        cz = cell_centers[3, i_cell]
        node = leaf_to_node[i_cell]
        for n in 0:P_phi
            for m in 0:n
                acc_re = zero(TF); acc_im = zero(TF)
                sgn = isodd(n + m) ? -one(TF) : one(TF)
                k = first + tid - 1
                while k <= first + count - 1
                    rre, rim = FastMultipole._resident_regular_harmonic_coeff(
                        source_bodies[1, k] - cx, source_bodies[2, k] - cy,
                        source_bodies[3, k] - cz, n, m)
                    scale = sgn * source_bodies[5, k]
                    acc_re += rre * scale
                    acc_im -= rim * scale
                    k += WG
                end
                shre[tid] = acc_re
                shim[tid] = acc_im
                @synchronize()
                s = WG >> 1
                while s >= 1
                    if tid <= s
                        shre[tid] += shre[tid + s]
                        shim[tid] += shim[tid + s]
                    end
                    @synchronize()
                    s >>= 1
                end
                if tid == 1
                    row = FastMultipole.flat_basis_index(n, m, 1)
                    phi[row, node] = shre[1]
                    phi[row + 1, node] = shim[1]
                end
                @synchronize()
            end
        end
    end
end

# Point{Dipole}: phi channel from `_resident_dipole_contrib` (the order n−1
# harmonics of x − c), sign and conjugation as the scalar B2M.
@kernel function ka_b2m_dipole_leaf_nodes_kernel!(phi, @Const(source_bodies),
        @Const(cell_centers), @Const(cell_ranges), @Const(leaf_to_node),
        P_phi, ncell, ::Type{TF}, ::Val{WG}) where {TF,WG}
    i_cell = @index(Group)
    tid = @index(Local)
    shre = @localmem TF (WG,)
    shim = @localmem TF (WG,)
    @inbounds begin
        first = cell_ranges[1, i_cell]
        count = cell_ranges[2, i_cell]
        cx = cell_centers[1, i_cell]
        cy = cell_centers[2, i_cell]
        cz = cell_centers[3, i_cell]
        node = leaf_to_node[i_cell]
        for n in 0:P_phi
            for m in 0:n
                acc_re = zero(TF); acc_im = zero(TF)
                sgn = isodd(n + m) ? -one(TF) : one(TF)
                k = first + tid - 1
                while k <= first + count - 1
                    re_, im_ = FastMultipole._resident_dipole_contrib(
                        source_bodies[1, k] - cx, source_bodies[2, k] - cy,
                        source_bodies[3, k] - cz, source_bodies[5, k],
                        source_bodies[6, k], source_bodies[7, k], n, m)
                    acc_re += re_ * sgn
                    acc_im -= im_ * sgn
                    k += WG
                end
                shre[tid] = acc_re
                shim[tid] = acc_im
                @synchronize()
                s = WG >> 1
                while s >= 1
                    if tid <= s
                        shre[tid] += shre[tid + s]
                        shim[tid] += shim[tid + s]
                    end
                    @synchronize()
                    s >>= 1
                end
                if tid == 1
                    row = FastMultipole.flat_basis_index(n, m, 1)
                    phi[row, node] = shre[1]
                    phi[row + 1, node] = shim[1]
                end
                @synchronize()
            end
        end
    end
end

# Point{SourceVortex}: source strength in row 5 into phi (scalar B2M) plus the
# vortex strength in rows 6:8 into phi and chi (mirrored vortex B2M).
@kernel function ka_b2m_sourcevortex_leaf_nodes_kernel!(phi, chi, @Const(source_bodies),
        @Const(cell_centers), @Const(cell_ranges), @Const(leaf_to_node),
        P_phi, P_chi, ncell, ::Type{TF}, ::Val{WG}) where {TF,WG}
    i_cell = @index(Group)
    tid = @index(Local)
    shre = @localmem TF (WG,)
    shim = @localmem TF (WG,)
    @inbounds begin
        first = cell_ranges[1, i_cell]
        count = cell_ranges[2, i_cell]
        cx = cell_centers[1, i_cell]
        cy = cell_centers[2, i_cell]
        cz = cell_centers[3, i_cell]
        node = leaf_to_node[i_cell]
        for n in 0:P_phi
            for m in 0:n
                acc_re = zero(TF); acc_im = zero(TF)
                sgn = isodd(n + m) ? -one(TF) : one(TF)
                k = first + tid - 1
                while k <= first + count - 1
                    dx = source_bodies[1, k] - cx
                    dy = source_bodies[2, k] - cy
                    dz = source_bodies[3, k] - cz
                    rre, rim = FastMultipole._resident_regular_harmonic_coeff(dx, dy, dz, n, m)
                    scale = sgn * source_bodies[5, k]
                    acc_re += rre * scale
                    acc_im -= rim * scale
                    re_, im_ = ka_vortex_phi_contrib(-dx, -dy, -dz, source_bodies[6, k],
                        source_bodies[7, k], source_bodies[8, k], n, m)
                    acc_re += re_; acc_im += im_
                    k += WG
                end
                shre[tid] = acc_re
                shim[tid] = acc_im
                @synchronize()
                s = WG >> 1
                while s >= 1
                    if tid <= s
                        shre[tid] += shre[tid + s]
                        shim[tid] += shim[tid + s]
                    end
                    @synchronize()
                    s >>= 1
                end
                if tid == 1
                    row = FastMultipole.flat_basis_index(n, m, 1)
                    phi[row, node] = shre[1]
                    phi[row + 1, node] = shim[1]
                end
                @synchronize()
            end
        end
        for n in 1:P_chi
            for m in 0:n
                acc_re = zero(TF); acc_im = zero(TF)
                k = first + tid - 1
                while k <= first + count - 1
                    re_, im_ = ka_vortex_chi_contrib(
                        cx - source_bodies[1, k], cy - source_bodies[2, k],
                        cz - source_bodies[3, k], source_bodies[6, k],
                        source_bodies[7, k], source_bodies[8, k], n, m)
                    acc_re += re_; acc_im += im_
                    k += WG
                end
                shre[tid] = acc_re
                shim[tid] = acc_im
                @synchronize()
                s = WG >> 1
                while s >= 1
                    if tid <= s
                        shre[tid] += shre[tid + s]
                        shim[tid] += shim[tid + s]
                    end
                    @synchronize()
                    s >>= 1
                end
                if tid == 1
                    row = FastMultipole.flat_basis_index(n, m, 1)
                    chi[row, node] = shre[1]
                    chi[row + 1, node] = shim[1]
                end
                @synchronize()
            end
        end
    end
end

"""
    ka_launch_b2m!(state; workgroup=128)

Body-to-multipole for the KA lifecycle. Zeroes the multipole buffers, then
runs one workgroup per leaf cell. Supported body types: `Point{Source}` and
`Point{Dipole}` fill the phi channel (with or without Lamb-Helmholtz; chi stays
zero); `Point{Vortex}` and `Point{SourceVortex}` fill both and require the
Lamb-Helmholtz channel; `Filament` and `Panel` go through the element sweep.
Any other body type is rejected with the list of supported ones.
"""
function ka_launch_b2m!(state::FastMultipole.DeviceResidentRadixState{TF,B,LH};
        workgroup::Int=128) where {TF,B,LH}
    return ka_launch_b2m!(state, state.options.body_type; workgroup)
end

function ka_launch_b2m!(state::FastMultipole.DeviceResidentRadixState{TF,B,LH},
        ::Type{<:FastMultipole.Point{FastMultipole.Source}}; workgroup::Int=128) where {TF,B,LH}
    ispow2(workgroup) || throw(ArgumentError(
        "ka_launch_b2m! workgroup must be a power of two (halving tree reduction)"))
    fill!(state.multipoles.phi, zero(TF))
    fill!(state.multipoles.chi, zero(TF))
    ncell = state.counts.n_cells
    ncell == 0 && return state
    backend = KA.get_backend(state.multipoles.phi)
    kernel = _cached_kernel(ka_b2m_source_leaf_nodes_kernel!, backend, workgroup)
    kernel(state.multipoles.phi, state.source_bodies, state.cell_centers,
        state.cell_ranges, state.grid.leaf_to_node,
        state.invariant_cache.basis_info.orders.P_phi, ncell, TF, Val(workgroup);
        ndrange=ncell * workgroup)
    return state
end

function ka_launch_b2m!(state::FastMultipole.DeviceResidentRadixState{TF,B,LH},
        ::Type{<:FastMultipole.Point{FastMultipole.Dipole}}; workgroup::Int=128) where {TF,B,LH}
    ispow2(workgroup) || throw(ArgumentError(
        "ka_launch_b2m! workgroup must be a power of two (halving tree reduction)"))
    fill!(state.multipoles.phi, zero(TF))
    fill!(state.multipoles.chi, zero(TF))
    ncell = state.counts.n_cells
    ncell == 0 && return state
    backend = KA.get_backend(state.multipoles.phi)
    kernel = _cached_kernel(ka_b2m_dipole_leaf_nodes_kernel!, backend, workgroup)
    kernel(state.multipoles.phi, state.source_bodies, state.cell_centers,
        state.cell_ranges, state.grid.leaf_to_node,
        state.invariant_cache.basis_info.orders.P_phi, ncell, TF, Val(workgroup);
        ndrange=ncell * workgroup)
    return state
end

function ka_launch_b2m!(state::FastMultipole.DeviceResidentRadixState{TF,B,LH},
        ::Type{<:FastMultipole.Point{FastMultipole.SourceVortex}}; workgroup::Int=128) where {TF,B,LH}
    LH || throw(ArgumentError(
        "Point{SourceVortex} sources require the Lamb-Helmholtz channel; construct the " *
        "cache with lamb_helmholtz=true"))
    ispow2(workgroup) || throw(ArgumentError(
        "ka_launch_b2m! workgroup must be a power of two (halving tree reduction)"))
    fill!(state.multipoles.phi, zero(TF))
    fill!(state.multipoles.chi, zero(TF))
    ncell = state.counts.n_cells
    ncell == 0 && return state
    orders = state.invariant_cache.basis_info.orders
    backend = KA.get_backend(state.multipoles.phi)
    kernel = _cached_kernel(ka_b2m_sourcevortex_leaf_nodes_kernel!, backend, workgroup)
    kernel(state.multipoles.phi, state.multipoles.chi, state.source_bodies,
        state.cell_centers, state.cell_ranges, state.grid.leaf_to_node,
        orders.P_phi, orders.P_active, ncell, TF, Val(workgroup);
        ndrange=ncell * workgroup)
    return state
end

#------- element B2M: filaments (src/resident_elements.jl on the device) -------#
#
# An element's expansion is a recurrence over its whole (n, m) triangle, so one
# workgroup per leaf cell strides its threads over the cell's bodies, each
# running the shared `_res_filament_b2m!` on that body's slice of a
# capacity-sized scratch (harmonics and coefficients per body), and then
# strides over the flat rows summing the cell's bodies into the leaf node.
# The scratch is allocated once per state at the body capacity and reused
# (grow-only, like the target-buffer cache), so the recurring step allocates
# nothing. Keyed weakly by the state's own mutable `counts` object (the state
# is an immutable struct and cannot be a weak key; `counts` has identity
# hashing, unlike an array key): a state replaced by the all-direct fallback or
# a recenter releases its scratch when it is collected.
const _KA_ELEMENT_SCRATCH = WeakKeyDict{Any,Any}()

function _ka_element_scratch(state, backend, ::Type{TF}, ndof::Integer, nh::Integer, cap::Integer) where TF
    sc = get(_KA_ELEMENT_SCRATCH, state.counts, nothing)
    if sc === nothing || eltype(sc.coef) != TF || size(sc.coef, 3) < ndof || size(sc.harm, 3) < nh ||
            size(sc.coef, 4) < cap
        c = sc === nothing ? cap : max(cap, size(sc.coef, 4) + cld(size(sc.coef, 4), 4))
        sc = (; coef = KA.allocate(backend, TF, 2, 2, ndof, c), harm = KA.allocate(backend, TF, 2, 2, nh, c))
        _KA_ELEMENT_SCRATCH[state.counts] = sc
    end
    return sc
end

@kernel function ka_b2m_filament_cells_kernel!(phi, chi, coef, harm, ::Val{BT},
        @Const(source_bodies), @Const(cell_centers), @Const(cell_ranges), @Const(leaf_to_node),
        P, ndof_phi, ndof_chi, ncell, ::Type{TF}, ::Val{WG}, ::Val{SD}) where {BT,TF,WG,SD}
    i_cell = @index(Group)
    tid = @index(Local)
    @inbounds begin
        first = cell_ranges[1, i_cell]
        count = cell_ranges[2, i_cell]
        cx = cell_centers[1, i_cell]
        cy = cell_centers[2, i_cell]
        cz = cell_centers[3, i_cell]
        node = leaf_to_node[i_cell]
        ndof = size(coef, 3)
        k = first + tid - 1
        while k <= first + count - 1
            cv = FastMultipole.ResBodySlice(coef, k)
            hv = FastMultipole.ResBodySlice(harm, k)
            for i in 1:ndof
                cv[1, 1, i] = zero(TF); cv[2, 1, i] = zero(TF)
                cv[1, 2, i] = zero(TF); cv[2, 2, i] = zero(TF)
            end
            FastMultipole._res_element_b2m!(BT, cv, hv, source_bodies, k, cx, cy, cz, P, Val(SD))
            k += WG
        end
        @synchronize()
        r = tid
        while r <= 2 * ndof_phi
            i = (r - 1) ÷ 2 + 1
            reim = (r - 1) % 2 + 1
            acc = zero(TF)
            for kk in first:(first + count - 1)
                acc += coef[reim, 1, i, kk]
            end
            phi[r, node] = acc
            r += WG
        end
        r = tid
        while r <= 2 * ndof_chi
            i = (r - 1) ÷ 2 + 1
            reim = (r - 1) % 2 + 1
            acc = zero(TF)
            for kk in first:(first + count - 1)
                acc += coef[reim, 2, i, kk]
            end
            chi[r, node] = acc
            r += WG
        end
    end
end

function ka_launch_b2m!(state::FastMultipole.DeviceResidentRadixState{TF,B,LH},
        ::Type{BT}; workgroup::Int=128) where {TF,B,LH,BT<:Union{FastMultipole.Filament,FastMultipole.Panel}}
    ((BT <: FastMultipole.Filament{FastMultipole.Vortex} || BT <: FastMultipole.Panel{3,FastMultipole.Vortex}) && !LH) && throw(ArgumentError(
        "$BT sources require the Lamb-Helmholtz channel; construct the " *
        "cache with lamb_helmholtz=true"))
    ispow2(workgroup) || throw(ArgumentError(
        "ka_launch_b2m! workgroup must be a power of two"))
    fill!(state.multipoles.phi, zero(TF))
    fill!(state.multipoles.chi, zero(TF))
    ncell = state.counts.n_cells
    ncell == 0 && return state
    orders = state.invariant_cache.basis_info.orders
    P_phi = orders.P_phi; P_chi = orders.P_active
    P = max(P_phi, P_chi)
    ndof = FastMultipole.harmonic_index(P, P)
    ndof_phi = FastMultipole.harmonic_index(P_phi, P_phi)
    ndof_chi = (P_chi >= 1 && size(state.multipoles.chi, 1) > 0) ? FastMultipole.harmonic_index(P_chi, P_chi) : 0
    backend = KA.get_backend(state.multipoles.phi)
    sc = _ka_element_scratch(state, backend, TF, ndof, FastMultipole._res_element_harmonics_rows(P),
        size(state.source_bodies, 2))
    kernel = _cached_kernel(ka_b2m_filament_cells_kernel!, backend, workgroup)
    kernel(state.multipoles.phi, state.multipoles.chi, sc.coef, sc.harm, Val(BT),
        state.source_bodies, state.cell_centers, state.cell_ranges, state.grid.leaf_to_node,
        P, ndof_phi, ndof_chi, ncell, TF, Val(workgroup), Val(FastMultipole.element_strength_dims(BT));
        ndrange=ncell * workgroup)
    return state
end

function ka_launch_b2m!(state::FastMultipole.DeviceResidentRadixState, ::Type{BT};
        workgroup::Int=128) where BT
    throw(ArgumentError("the KernelAbstractions device lifecycle implements body-to-multipole " *
        "for Point{Source}, Point{Vortex}, Point{Dipole}, Point{SourceVortex}, the three Filament types and the four Panel{3,TK} types; got body_type $BT"))
end

function ka_launch_b2m!(state::FastMultipole.DeviceResidentRadixState{TF,B,LH},
        ::Type{<:FastMultipole.Point{FastMultipole.Vortex}}; workgroup::Int=128) where {TF,B,LH}
    LH || throw(ArgumentError(
        "Point{Vortex} sources require the Lamb-Helmholtz channel; construct the " *
        "cache with lamb_helmholtz=true"))
    ispow2(workgroup) || throw(ArgumentError(
        "ka_launch_b2m! workgroup must be a power of two (halving tree reduction)"))
    fill!(state.multipoles.phi, zero(TF))
    fill!(state.multipoles.chi, zero(TF))
    ncell = state.counts.n_cells
    ncell == 0 && return state
    orders = state.invariant_cache.basis_info.orders
    backend = KA.get_backend(state.multipoles.phi)
    kernel = _cached_kernel(ka_b2m_vortex_leaf_nodes_kernel!, backend, workgroup)
    kernel(state.multipoles.phi, state.multipoles.chi, state.source_bodies,
        state.cell_centers, state.cell_ranges, state.grid.leaf_to_node,
        orders.P_phi, orders.P_active, ncell, TF, Val(workgroup);
        ndrange=ncell * workgroup)
    return state
end

#------- L2B (local -> body output) -------#
#
# Shape: KA has no portable warp concept, so this runs one WORKGROUP per cell
# and strides bodies by the workgroup size. That is safe: every body belongs to
# exactly one cell (`cell_ranges` partitions the sorted bodies) and each body
# writes only its own output column, so there is no reduction, no shared
# memory, no `@synchronize` and no atomic here. The value written for a given
# body is computed independently of the thread mapping.

# The regular-harmonic sweep, KA-only.
#
# The per-body evaluation mirrors `_resident_local_eval_flat` and
# `_resident_local_eval_flat_hessian` (src/resident/resident_pair_kernels.jl),
# with one change to how the harmonics are produced.
#
# `_resident_regular_harmonic_coeff` (src/resident/resident_grid_state.jl) restarts
# a full O(P^2) associated-Legendre recurrence for EVERY (n,m) it is asked for.
# L2B's hessian branch asks 4 times per pair, so a body pays ~72 restarts where
# ONE m-outer/n-inner sweep produces every coefficient it needs. An ablation on
# Metal measured those restarts at 72.6% of the L2B stage.
#
# `ka_local_eval_flat` below walks that same recurrence once and emits
# R_{n,m} as it goes. The arithmetic is op-for-op what the shared function does
# for each target -- the outer state (`rhom`, `pn`, `fact`, `ieim`) is untouched
# by the inner n loop, so extending that loop to P_active instead of stopping at
# each target changes no value. The COEFFICIENTS are therefore bit-identical;
# only the order in which the 13 outputs accumulate over (n,m) changes, which is
# a Float32 rounding difference and why the gate scores relerr, not equality.
#
# KA-ONLY BY CONSTRUCTION: this lives in the extension and the shared
# `_resident_local_eval_flat*` are left exactly as they are, so the CPU host
# kernels keep their current arithmetic and stay the oracle.
# The cost of that choice is a second copy of the evaluation math here, which
# must stay in lockstep with `src/resident/resident_pair_kernels.jl`.

# One (n,m) term of the local evaluation: everything the shared
# `_resident_local_eval_flat_hessian` does inside its two loop bodies at that
# pair, returned as increments. `HESS=false` drops the second pass (the 4-row
# kernel), matching `_resident_local_eval_flat`.
@inline function ka_l2b_term(ph, ch, node, P_phi, P_active, n, m, rre, rim,
        lhv::Val{LH}, ::Val{HESS}) where {LH,HESS}
    TF = eltype(ph)
    z = zero(TF)
    u = z; vx = z; vy = z; vz = z
    hxx = z; hxy = z; hxz = z; hyx = z; hyy = z; hyz = z; hzx = z; hzy = z; hzz = z
    @inbounds begin
        # ---- pass 1: potential and gradient
        if m == 0
            if n <= P_phi && (!LH || n == 0)
                u += rre * FastMultipole._resident_flat_phi_re(ph, node, P_phi, n, 0) -
                     rim * FastMultipole._resident_flat_phi_im(ph, node, P_phi, n, 0)
            end
            vxr, vxi, vyr, vyi, vzr, vzi =
                FastMultipole._resident_gradient_coeff(ph, ch, node, P_phi, P_active, n, 0, lhv)
            vx += vxr * rre - vxi * rim
            vy += vyr * rre - vyi * rim
            vz += vzr * rre - vzi * rim
        else
            if n <= P_phi && !LH
                u += 2 * (rre * FastMultipole._resident_flat_phi_re(ph, node, P_phi, n, m) -
                          rim * FastMultipole._resident_flat_phi_im(ph, node, P_phi, n, m))
            end
            vxr, vxi, vyr, vyi, vzr, vzi =
                FastMultipole._resident_gradient_coeff(ph, ch, node, P_phi, P_active, n, m, lhv)
            vx += 2 * (vxr * rre - vxi * rim)
            vy += 2 * (vyr * rre - vyi * rim)
            vz += 2 * (vzr * rre - vzi * rim)
        end
        # ---- pass 2: hessian. The shared version runs n only to P_active-1.
        if HESS && n <= P_active - 1
            if m == 0
                g0x_r, g0x_i, g0y_r, g0y_i, g0z_r, g0z_i =
                    FastMultipole._resident_gradient_coeff(ph, ch, node, P_phi, P_active, n + 1, 0, lhv)
                g1x_r, g1x_i, g1y_r, g1y_i, g1z_r, g1z_i =
                    FastMultipole._resident_gradient_coeff(ph, ch, node, P_phi, P_active, n + 1, 1, lhv)
                hxx += -g1x_i * rre
                hyx += -g1x_r * rre
                hzx += -g0x_r * rre + g0x_i * rim
                hxy += -g1y_i * rre
                hyy += -g1y_r * rre
                hzy += -g0y_r * rre + g0y_i * rim
                hxz += -g1z_i * rre
                hyz += -g1z_r * rre
                hzz += -g0z_r * rre + g0z_i * rim
            else
                amx_r, amx_i, amy_r, amy_i, amz_r, amz_i =
                    FastMultipole._resident_gradient_coeff(ph, ch, node, P_phi, P_active, n + 1, m - 1, lhv)
                bmx_r, bmx_i, bmy_r, bmy_i, bmz_r, bmz_i =
                    FastMultipole._resident_gradient_coeff(ph, ch, node, P_phi, P_active, n + 1, m, lhv)
                cmx_r, cmx_i, cmy_r, cmy_i, cmz_r, cmz_i =
                    FastMultipole._resident_gradient_coeff(ph, ch, node, P_phi, P_active, n + 1, m + 1, lhv)
                tr = -(amx_i + cmx_i) * TF(0.5); ti = (amx_r + cmx_r) * TF(0.5)
                hxx += 2 * (tr * rre - ti * rim)
                tr = (amx_r - cmx_r) * TF(0.5); ti = (amx_i - cmx_i) * TF(0.5)
                hyx += 2 * (tr * rre - ti * rim)
                hzx += 2 * (-bmx_r * rre + bmx_i * rim)
                tr = -(amy_i + cmy_i) * TF(0.5); ti = (amy_r + cmy_r) * TF(0.5)
                hxy += 2 * (tr * rre - ti * rim)
                tr = (amy_r - cmy_r) * TF(0.5); ti = (amy_i - cmy_i) * TF(0.5)
                hyy += 2 * (tr * rre - ti * rim)
                hzy += 2 * (-bmy_r * rre + bmy_i * rim)
                tr = -(amz_i + cmz_i) * TF(0.5); ti = (amz_r + cmz_r) * TF(0.5)
                hxz += 2 * (tr * rre - ti * rim)
                tr = (amz_r - cmz_r) * TF(0.5); ti = (amz_i - cmz_i) * TF(0.5)
                hyz += 2 * (tr * rre - ti * rim)
                hzz += 2 * (-bmz_r * rre + bmz_i * rim)
            end
        end
    end
    return u, vx, vy, vz, hxx, hxy, hxz, hyx, hyy, hyz, hzx, hzy, hzz
end

# Local -> body with ONE recurrence sweep per body. Returns the 13-tuple of the
# shared `_resident_local_eval_flat_hessian` when HESS, else the 4-tuple of
# `_resident_local_eval_flat` (the hessian slots are computed as zeros and
# dropped by the caller, so both share this body).
@inline function ka_local_eval_flat(ph, ch, node, dx, dy, dz, P_phi, P_active,
        lhv::Val{LH}, hv::Val{HESS}) where {LH,HESS}
    TF = eltype(ph)
    c = inv(TF(4) * TF(pi))
    z = zero(TF)
    u = z; vx = z; vy = z; vz = z
    hxx = z; hxy = z; hxz = z; hyx = z; hyy = z; hyz = z; hzx = z; hzy = z; hzz = z
    rho, xc, ys, iei_re, iei_im = FastMultipole._resident_harmonic_setup(dx, dy, dz)
    # rho == 0 needs no special case: the setup returns all-zero, so the sweep
    # emits (1,0) at (0,0) and zero elsewhere -- exactly what the shared
    # coefficient function returns for a coincident point.
    # Counters in the field's integer width (see _ka_int_type); same recurrence
    # as ka_vortex_q3.
    IT = _ka_int_type(TF); i1 = one(IT)
    @inbounds begin
        fact = one(TF); pn = one(TF); rhom = one(TF)
        ieim_re = one(TF); ieim_im = z
        for m in zero(IT):IT(P_active)
            # n == m
            p = pn
            rmp = rhom * p
            t = ka_l2b_term(ph, ch, node, P_phi, P_active, m, m,
                            rmp * ieim_re, rmp * ieim_im, lhv, hv)
            u += t[1]; vx += t[2]; vy += t[3]; vz += t[4]
            if HESS
                hxx += t[5]; hxy += t[6]; hxz += t[7]
                hyx += t[8]; hyy += t[9]; hyz += t[10]
                hzx += t[11]; hzy += t[12]; hzz += t[13]
            end
            p1 = p
            p = xc * TF(m + m + i1) * p1
            rhom *= rho
            rhon = rhom
            for n in (m + i1):IT(P_active)
                rhon /= -TF(n + m)
                rnp = rhon * p
                t = ka_l2b_term(ph, ch, node, P_phi, P_active, n, m,
                                rnp * ieim_re, rnp * ieim_im, lhv, hv)
                u += t[1]; vx += t[2]; vy += t[3]; vz += t[4]
                if HESS
                    hxx += t[5]; hxy += t[6]; hxz += t[7]
                    hyx += t[8]; hyy += t[9]; hyz += t[10]
                    hzx += t[11]; hzy += t[12]; hzz += t[13]
                end
                p2 = p1; p1 = p
                p = (xc * TF(n + n + i1) * p1 - TF(n + m) * p2) / TF(n - m + i1)
                rhon *= rho
            end
            rhom /= -TF(m + m + i1 + i1) * TF(m + m + i1)
            pn = -pn * fact * ys
            fact += TF(2)
            tre = ieim_re
            ieim_re = tre * iei_re - ieim_im * iei_im
            ieim_im = tre * iei_im + ieim_im * iei_re
        end
    end
    return u * c, vx * c, vy * c, vz * c,
        hxx * c, hxy * c, hxz * c, hyx * c, hyy * c, hyz * c, hzx * c, hzy * c, hzz * c
end

@kernel function ka_l2b_output_kernel!(output, @Const(source_bodies), @Const(cell_centers),
        @Const(cell_ranges), @Const(leaf_to_node), @Const(local_phi), @Const(local_chi),
        P_phi, P_active, ::Val{LHV}, ncell, ::Val{WG}) where {LHV,WG}
    cell = @index(Group)
    tid = @index(Local)
    @inbounds begin
        node = leaf_to_node[cell]
        first = cell_ranges[1, cell]
        last = first + cell_ranges[2, cell] - 1
        cx = cell_centers[1, cell]; cy = cell_centers[2, cell]; cz = cell_centers[3, cell]
        i = first + tid - 1
        while i <= last
            sp, gx, gy, gz = ka_local_eval_flat(
                local_phi, local_chi, node,
                source_bodies[1, i] - cx, source_bodies[2, i] - cy,
                source_bodies[3, i] - cz, P_phi, P_active, Val(LHV), Val(false))
            output[1, i] += sp
            output[2, i] += gx
            output[3, i] += gy
            output[4, i] += gz
            i += WG
        end
    end
end

@kernel function ka_l2b_output_hessian_kernel!(output, @Const(source_bodies),
        @Const(cell_centers), @Const(cell_ranges), @Const(leaf_to_node),
        @Const(local_phi), @Const(local_chi), P_phi, P_active, ::Val{LHV},
        ncell, ::Val{WG}) where {LHV,WG}
    cell = @index(Group)
    tid = @index(Local)
    @inbounds begin
        node = leaf_to_node[cell]
        first = cell_ranges[1, cell]
        last = first + cell_ranges[2, cell] - 1
        cx = cell_centers[1, cell]; cy = cell_centers[2, cell]; cz = cell_centers[3, cell]
        i = first + tid - 1
        while i <= last
            vals = ka_local_eval_flat(
                local_phi, local_chi, node,
                source_bodies[1, i] - cx, source_bodies[2, i] - cy,
                source_bodies[3, i] - cz, P_phi, P_active, Val(LHV), Val(true))
            Base.Cartesian.@nexprs 13 r -> (output[r, i] += vals[r])
            i += WG
        end
    end
end

"""
    ka_launch_l2b!(state; workgroup=64)

Local-to-body evaluation for the KA lifecycle, one workgroup per leaf cell.
Selects the 13-row hessian variant on `size(state.output, 1) >= 13`; FLOWVPM
runs with `hessian=true` and therefore takes that branch.
"""
function ka_launch_l2b!(state::FastMultipole.DeviceResidentRadixState{TF,B,LH};
        workgroup::Int=64) where {TF,B,LH}
    ncell = state.counts.n_cells
    ncell == 0 && return state
    orders = state.invariant_cache.basis_info.orders
    backend = KA.get_backend(state.output)
    workgroup = resolve_workgroup(backend, workgroup)
    workgroup > 0 || throw(ArgumentError("ka_launch_l2b! workgroup must be positive"))
    args = (state.output, state.source_bodies, state.cell_centers, state.cell_ranges,
            state.grid.leaf_to_node, state.locals.phi, state.locals.chi,
            orders.P_phi, orders.P_active, Val(LH), ncell, Val(workgroup))
    if size(state.output, 1) >= 13
        kernel = _cached_kernel(ka_l2b_output_hessian_kernel!, backend, workgroup)
    else
        kernel = _cached_kernel(ka_l2b_output_kernel!, backend, workgroup)
    end
    kernel(args...; ndrange=ncell * workgroup)
    return state
end

#------- NEARFIELD (U-list direct pairs) -------#
#
# The per-pair math is NOT reimplemented:
# `_direct_pair_ug` / `_direct_pair_ugh` (src/resident/resident_pair_kernels.jl)
# are backend-agnostic and shared with the CPU path, so every kernel functor
# (PartitionedVortex, RegularizedVortex, SingularSource, ...) comes along for
# free.
#
# Only the cell-pair shape exists here, always with the `ghv = Val(:shipped)`
# g/h series and no lookup table; fused target-owned or symmetric Newton-pair
# shapes would be performance variants of the same physics.
#
# A team of lanes per pair, lanes striding the pair's target bodies. Targets of
# different pairs overlap, so the output accumulation must stay atomic. The
# stage is gated against the CPU reference.

# 1/sqrt(r2). Float32: `inv(sqrt(r2))`. Float64: a Float32 seed refined by two
# Newton steps (24 -> 48 -> 53 bits); a full FP64 sqrt+divide was 1.7x of the
# Float32 nearfield on an H200.
@inline _ka_invsqrt(r2::Float32) = inv(sqrt(r2))
# the Float32 seed is only meaningful for r2 in Float32's normal range; outside
# it (seed Inf or 0) take the full-precision path
@inline _ka_f32_seed_ok(r2::Float64) =
    Float64(floatmin(Float32)) <= r2 <= Float64(floatmax(Float32))
@inline function _ka_invsqrt(r2::Float64)
    _ka_f32_seed_ok(r2) || return inv(sqrt(r2))
    y = Float64(inv(sqrt(Float32(r2))))
    y = y * (1.5 - 0.5 * r2 * y * y)
    y = y * (1.5 - 0.5 * r2 * y * y)
    return y
end
@inline _ka_invsqrt(r2) = inv(sqrt(r2))

#------- lanes-per-pair nearfield -------#
#
# The only nearfield kernel the lifecycle launches (the all-pairs kernel below
# serves the direct arm). Launch geometry, from `_nf_config`:
#   * LANES threads per pair, WG/LANES pairs per group. On CUDA, LANES follows
#     the cell population in [64, 256] (Float64: [64, 128]) and WG = max(128,
#     LANES), so a group carries two pairs at 64 lanes and one otherwise; on
#     every other backend WG = LANES = 64, one pair per group;
#   * `FR` (fast rsqrt): on CUDA, libdevice's approximate `rsqrtf` instead of
#     IEEE sqrt+divide in the innermost line; Float64 seeds two Newton steps
#     from it. Other backends use `_ka_invsqrt(r2)` above;
#   * `GH`, the g/h series mode, always `:shipped` at the launch.

# libdevice's rsqrtf (what CUDA.rsqrt calls); resolved by the CUDA compiler's
# libdevice link, so only reachable when `_nf_config` enables it on a
# CUDABackend.
@inline _ka_rsqrt_approx(r2::Float32) =
    ccall("extern __nv_rsqrtf", llvmcall, Cfloat, (Cfloat,), r2)
@inline _ka_invsqrt(r2::Float32, ::Val{true}) = _ka_rsqrt_approx(r2)
@inline function _ka_invsqrt(r2::Float64, ::Val{true})
    _ka_f32_seed_ok(r2) || return inv(sqrt(r2))
    y = Float64(_ka_rsqrt_approx(Float32(r2)))
    y = y * (1.5 - 0.5 * r2 * y * y)
    y = y * (1.5 - 0.5 * r2 * y * y)
    return y
end
@inline _ka_invsqrt(r2, ::Val{false}) = _ka_invsqrt(r2)

@kernel function ka_direct_pairs_warp_kernel!(kernel, output, @Const(source_bodies),
        @Const(cell_ranges), @Const(direct_targets), @Const(direct_sources),
        npairs, ::Type{T}, ::Val{HS}, ::Val{WG}, ::Val{LANES}, ::Val{FR},
        ::Val{GH}) where {T,HS,WG,LANES,FR,GH}
    tid = @index(Local)
    pair_i = (@index(Group) - 1) * (WG ÷ LANES) + (tid - 1) ÷ LANES + 1
    lane = (tid - 1) % LANES
    ep = FastMultipole._emits_potential(kernel)
    ghv = Val(GH)
    frv = Val(FR)
    @inbounds if pair_i <= npairs
        target_cell = direct_targets[pair_i]
        source_cell = direct_sources[pair_i]
        tfirst = cell_ranges[1, target_cell]
        tlast = tfirst + cell_ranges[2, target_cell] - 1
        sfirst = cell_ranges[1, source_cell]
        slast = sfirst + cell_ranges[2, source_cell] - 1
        i = tfirst + lane
        while i <= tlast
            xi = source_bodies[1, i]; yi = source_bodies[2, i]; zi = source_bodies[3, i]
            u = zero(T); gx = zero(T); gy = zero(T); gz = zero(T)
            h1 = zero(T); h2 = zero(T); h3 = zero(T)
            h4 = zero(T); h5 = zero(T); h6 = zero(T)
            h7 = zero(T); h8 = zero(T); h9 = zero(T)
            for j in sfirst:slast
                if i != j
                    dx = xi - source_bodies[1, j]
                    dy = yi - source_bodies[2, j]
                    dz = zi - source_bodies[3, j]
                    r2 = dx * dx + dy * dy + dz * dz
                    if r2 > zero(r2)
                        invr = _ka_invsqrt(r2, frv)
                        if HS
                            du, dgx, dgy, dgz, dh1, dh2, dh3, dh4, dh5, dh6, dh7, dh8, dh9 =
                                FastMultipole._direct_pair_ugh(kernel, dx, dy, dz, r2, invr,
                                    source_bodies, j, ghv)
                            u += du; gx += dgx; gy += dgy; gz += dgz
                            h1 += dh1; h2 += dh2; h3 += dh3
                            h4 += dh4; h5 += dh5; h6 += dh6
                            h7 += dh7; h8 += dh8; h9 += dh9
                        else
                            du, dgx, dgy, dgz = FastMultipole._direct_pair_ug(kernel,
                                dx, dy, dz, r2, invr, source_bodies, j, ghv)
                            u += du; gx += dgx; gy += dgy; gz += dgz
                        end
                    end
                end
            end
            if ep
                KA.@atomic output[1, i] += u
            end
            KA.@atomic output[2, i] += gx
            KA.@atomic output[3, i] += gy
            KA.@atomic output[4, i] += gz
            if HS
                KA.@atomic output[5, i]  += h1
                KA.@atomic output[6, i]  += h2
                KA.@atomic output[7, i]  += h3
                KA.@atomic output[8, i]  += h4
                KA.@atomic output[9, i]  += h5
                KA.@atomic output[10, i] += h6
                KA.@atomic output[11, i] += h7
                KA.@atomic output[12, i] += h8
                KA.@atomic output[13, i] += h9
            end
            i += LANES
        end
    end
end


# (A register-tiled variant -- two targets per thread through one source pass
# -- was tried and removed: 10% slower at 115k and 249k bodies on an H200 and
# 4% slower in Float64. The loop is not load-bound.)

# CONVENTION: for a regularized kernel (sigma_row > 0) both allocators size
# `source_bodies` one row past the packed body rows and the packers fill that
# LAST row with 1/sigma, so the nearfield can multiply instead of divide per
# interaction (13% at 64 lanes, ~0 at 128+ on an H200). Always on.
function _ka_nf_inv_sigma_row(state)
    _ka_kernel_sigma_row(state.options.direct_kernel) > 0 || return 0
    return size(state.source_bodies, 1)
end

# Per-backend nearfield launch configuration (see the block above). Only the
# backend TYPE NAME is consulted, so this extension stays free of CUDA.
#
# Defaults measured on an H200, a rotor-wake particle field at np=248714 (20
# calls, median): one block per pair with ALL its lanes on that pair, and the
# lane count is what matters -- 64 lanes 0.099 s, 128 0.072, 256 0.069, 512
# 0.074 in Float32 (a hand-written CUDA.jl kernel: 0.067-0.074); Float64 128
# lanes 0.130, 256 0.135 (hand-written: 0.18-0.19). Warp-sized teams (32 lanes,
# four pairs per 128-thread block) were SLOWER, 0.130. The reciprocal-sigma row
# bought 13% at 64 lanes and nothing at 128+, where the divide latency is
# already hidden.
function _nf_config(backend, ::Type{TF}; bodies_per_cell::Real=0) where TF
    cuda = nameof(typeof(backend)) === :CUDABackend
    # Lanes per pair follow the cell population (the lanes stride a pair's
    # TARGET bodies): a dense field wants a whole block on each pair, a thin
    # one wants warp-sized teams so lanes are not idle. Clamped to [64, 256]
    # (Float64: 128 -- its measured optimum at 249k) and rounded to a power of
    # two. Non-CUDA backends: 64 lanes, one pair per block
    # (measured flat 64-256 on Metal).
    hi = TF === Float32 ? 256 : 128
    auto_lanes = cuda && bodies_per_cell > 0 ?
        clamp(nextpow(2, max(1, ceil(Int, bodies_per_cell))), 64, hi) : (cuda ? hi : 64)
    lanes = auto_lanes
    wg = cuda ? max(128, lanes) : lanes
    # libdevice rsqrtf is CUDA-only; other backends keep inv(sqrt)
    return (; wg, lanes, fast=cuda)
end

"""
    ka_launch_nearfield!(state; workgroup=nothing, clear=true)

U-list direct nearfield for the KA lifecycle, one lane team per direct cell
pair. Zeroes `state.output` first (this is the first stage of the lifecycle)
unless `clear=false`. `workgroup=nothing` takes the launch shape from
`_nf_config`; an explicit workgroup must be a multiple of the lanes per pair.
"""
function ka_launch_nearfield!(state::FastMultipole.DeviceResidentRadixState{TF,B,LH};
        workgroup::Union{Nothing,Int}=nothing, clear::Bool=true) where {TF,B,LH}
    clear && fill!(state.output, zero(TF))
    npairs = state.counts.n_direct
    npairs == 0 && return state
    hs = size(state.output, 1) >= 13
    backend = KA.get_backend(state.output)
    n_cells = Int(state.counts.n_cells)
    cfg = _nf_config(backend, TF;
        bodies_per_cell = n_cells > 0 ? Int(state.counts.n_bodies) / n_cells : 0)
    wg = workgroup === nothing ? cfg.wg : resolve_workgroup(backend, workgroup)
    wg > 0 || throw(ArgumentError("ka_launch_nearfield! workgroup must be positive"))
    lanes = min(cfg.lanes, wg)
    # a group carries wg ÷ lanes whole pairs; a partial team would re-run the
    # next group's first pair and add it twice
    wg % lanes == 0 || throw(ArgumentError(
        "ka_launch_nearfield! workgroup=$wg must be a multiple of the $lanes lanes per pair"))
    dkernel = _ka_device_direct_kernel(state.options.direct_kernel, TF,
        _ka_nf_inv_sigma_row(state))
    kern = _cached_kernel(ka_direct_pairs_warp_kernel!, backend, wg)
    kern(dkernel, state.output, state.source_bodies,
         state.cell_ranges, state.direct_targets, state.direct_sources,
         npairs, TF, Val(hs), Val(wg), Val(lanes), Val(cfg.fast), Val(:shipped),
         ndrange=cld(npairs, wg ÷ lanes) * wg)
    return state
end

#------- all-pairs direct arm -------#
#
# An opt-in alternative to the FMM lifecycle: one kernel, no grid, no routes,
# no tree, every pair evaluated. Armed only by the caller through the radix
# setting `:RADIX_DIRECT_ARM`; nothing selects it automatically.
#
# There IS a crossover below which this beats the lifecycle -- measured at
# np ~ 6e4 on Metal for one VPM wake, with the arm 2-4x ahead below
# np = 8192. That number is one backend, one wake, one sweep, and the
# lifecycle's cost balance is not the same on CUDA, so it is recorded here as
# an observation and deliberately NOT turned into a dispatch rule. Anything
# that selects between the two arms needs a cross-backend calibration first.
#
# Shape differs from the cell-pair nearfield kernel deliberately. There a
# workgroup owns a cell PAIR and its targets are shared with other pairs, so
# every accumulation is atomic. Here a workitem owns one target body outright
# and no other workitem touches it, so the accumulators are plain stores --
# which also makes the result deterministic, unlike the pair shape.
#
# The physics is the same `_direct_pair_ug` / `_direct_pair_ugh` shared with
# the CPU path, with the same `ghv = Val(:shipped)` series and a plain
# `inv(sqrt(r2))` at every precision, so this arm is gated against the CPU
# reference exactly as the pair shape is.

@kernel function ka_direct_all_pairs_kernel!(kernel, output, @Const(source_bodies),
        nbodies, ::Type{T}, ::Val{HS}) where {T,HS}
    i = @index(Global)
    ep = FastMultipole._emits_potential(kernel)
    ghv = Val(:shipped)
    @inbounds if i <= nbodies
        xi = source_bodies[1, i]; yi = source_bodies[2, i]; zi = source_bodies[3, i]
        u = zero(T); gx = zero(T); gy = zero(T); gz = zero(T)
        h1 = zero(T); h2 = zero(T); h3 = zero(T)
        h4 = zero(T); h5 = zero(T); h6 = zero(T)
        h7 = zero(T); h8 = zero(T); h9 = zero(T)
        for j in 1:nbodies
            if i != j
                dx = xi - source_bodies[1, j]
                dy = yi - source_bodies[2, j]
                dz = zi - source_bodies[3, j]
                r2 = dx * dx + dy * dy + dz * dz
                if r2 > zero(r2)
                    invr = inv(sqrt(r2))
                    if HS
                        du, dgx, dgy, dgz, dh1, dh2, dh3, dh4, dh5, dh6, dh7, dh8, dh9 =
                            FastMultipole._direct_pair_ugh(kernel, dx, dy, dz, r2, invr,
                                source_bodies, j, ghv)
                        u += du; gx += dgx; gy += dgy; gz += dgz
                        h1 += dh1; h2 += dh2; h3 += dh3
                        h4 += dh4; h5 += dh5; h6 += dh6
                        h7 += dh7; h8 += dh8; h9 += dh9
                    else
                        du, dgx, dgy, dgz = FastMultipole._direct_pair_ug(kernel,
                            dx, dy, dz, r2, invr, source_bodies, j, ghv)
                        u += du; gx += dgx; gy += dgy; gz += dgz
                    end
                end
            end
        end
        if ep
            output[1, i] = u
        end
        output[2, i] = gx
        output[3, i] = gy
        output[4, i] = gz
        if HS
            output[5, i]  = h1
            output[6, i]  = h2
            output[7, i]  = h3
            output[8, i]  = h4
            output[9, i]  = h5
            output[10, i] = h6
            output[11, i] = h7
            output[12, i] = h8
            output[13, i] = h9
        end
    end
end

"""
    ka_direct_body!(state; workgroup=KA_AUTO_WORKGROUP)

All-pairs replacement for [`ka_lifecycle_body!`](@ref): every target body
against every source body in one kernel. Rows 2:4 (and 5:13 with a hessian)
are written, not accumulated (the kernel owns one column per workitem), but
row 1 is written only for a potential-emitting kernel, so the caller must
clear `state.output` first. The columns past `counts.n_bodies` are left as
they were, and `ka_finalize_radix_output!` reads only the valid prefix.
"""
function ka_direct_body!(state::FastMultipole.DeviceResidentRadixState{TF,B,LH};
        workgroup=KA_AUTO_WORKGROUP) where {TF,B,LH}
    n = state.counts.n_bodies
    n == 0 && return state
    hs = size(state.output, 1) >= 13
    backend = KA.get_backend(state.output)
    wg = resolve_workgroup(backend, workgroup)
    kern = _cached_kernel(ka_direct_all_pairs_kernel!, backend, wg)
    # same TF re-parameterization as ka_launch_nearfield!: the stock
    # regularized functors carry hardcoded Float64 cutoffs
    # inv_sigma_row=0: rho is computed as |r|/sigma, one divide per interaction
    dkernel = _ka_device_direct_kernel(state.options.direct_kernel, TF, 0)
    kern(dkernel, state.output, state.source_bodies, n, TF, Val(hs);
         ndrange=cld(n, wg) * wg)
    KA.synchronize(backend)
    return state
end
