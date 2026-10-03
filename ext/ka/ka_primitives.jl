#------- fixed-geometry row extrema (src: _device_row_extrema) -------#
#
# One kernel of ROW_EXTREMA_LANES workitems, each striding the row; the lane
# partials (2 x lanes scalars) cross to the host and finish there. The kernel's
# specialization does not involve `n`, so it compiles once per (backend, eltype)
# rather than once per distinct length.
#
# The lane scratch is a pool per (backend type, device, eltype): a call takes a
# set out under `_CACHE_LOCK` and returns it when done, so concurrent callers
# never share buffers and the steady state allocates nothing.

const ROW_EXTREMA_LANES = 1024
const _ROW_EXTREMA_SCRATCH = Dict{Any,Vector{Any}}()

@kernel function ka_row_extrema_kernel!(lo, hi, @Const(A), row, n, lanes)
    i = @index(Global)
    T = eltype(A)
    mn = typemax(T); mx = typemin(T)
    j = i
    @inbounds while j <= n
        v = A[row, j]
        mn = min(mn, v); mx = max(mx, v)
        j += lanes
    end
    @inbounds if i <= lanes
        lo[i] = mn; hi[i] = mx
    end
end

FastMultipole._device_row_extrema(A::AnyGPUMatrix, row::Integer, n::Integer) =
    _ka_row_extrema(A, row, n, true)
# the max alone downloads one partial vector instead of two
FastMultipole._device_row_max(A::AnyGPUMatrix, row::Integer, n::Integer) =
    _ka_row_extrema(A, row, n, false)[2]

function _ka_row_extrema(A, row::Integer, n::Integer, want_lo::Bool)
    n > 0 || throw(ArgumentError("reducing over an empty row prefix"))
    T = eltype(A)
    backend = KA.get_backend(A)
    key = (typeof(backend), KA.device(backend), T)
    scratch = lock(_CACHE_LOCK) do
        pool = get!(Vector{Any}, _ROW_EXTREMA_SCRATCH, key)
        isempty(pool) ? nothing : pop!(pool)
    end
    if scratch === nothing
        scratch = (KA.allocate(backend, T, ROW_EXTREMA_LANES),
                   KA.allocate(backend, T, ROW_EXTREMA_LANES),
                   Vector{T}(undef, ROW_EXTREMA_LANES), Vector{T}(undef, ROW_EXTREMA_LANES))
    end
    lo, hi, hlo, hhi = scratch
    try
        wg = resolve_workgroup(backend, KA_AUTO_WORKGROUP)
        kern = _cached_kernel(ka_row_extrema_kernel!, backend, wg)
        kern(lo, hi, A, Int(row), Int(n), ROW_EXTREMA_LANES; ndrange=ROW_EXTREMA_LANES)
        copyto!(hhi, hi)
        want_lo || return zero(T), maximum(hhi)
        copyto!(hlo, lo)
        return minimum(hlo), maximum(hhi)
    finally
        lock(() -> push!(_ROW_EXTREMA_SCRATCH[key], scratch), _CACHE_LOCK)
    end
end

#------- backend workgroup policy -------#
#
# A workgroup of 64 suits Metal (SIMD width 32, small threadgroups keep
# occupancy up on a 16-32 core GPU) but wastes scheduler slots on an A100, where
# 256 is the usual figure for the memory-bound elementwise kernels that
# dominate this file. Tunable sites pass `KA_AUTO_WORKGROUP` and the size is
# resolved per backend, so one source tunes for both the local and the HPC
# target.
#
# NOT every site is tunable. `ka_launch_b2m!` and `ka_launch_l2b!` thread
# `workgroup` into `Val(workgroup)` and launch `ndrange = ncell * workgroup`
# (one group per cell): there the workgroup is the per-cell *team size* (for
# B2M, also what the `@localmem` reduction extents are declared against), not
# an occupancy knob. Those keep their explicit sizes; changing one there
# changes the parallel decomposition, not just the launch geometry.

"""
    KA_AUTO_WORKGROUP

Sentinel workgroup size meaning "let the backend decide"; see
[`resolve_workgroup`](@ref).
"""
const KA_AUTO_WORKGROUP = 0

_backend_default_workgroup(::KA.CPU) = 64
function _backend_default_workgroup(backend)
    # The accelerator backend types live in packages this extension must not
    # depend on, so select on the type's name rather than on the type.
    name = string(nameof(typeof(backend)))
    (name == "CUDABackend" || name == "ROCBackend" || name == "oneAPIBackend") && return 256
    name == "MetalBackend" && return 64
    return 64  # conservative for an unrecognised accelerator
end

const _WORKGROUP_CACHE = Dict{DataType,Int}()

"""
    resolve_workgroup(backend, workgroup) -> Int

Return `workgroup` unchanged unless it is [`KA_AUTO_WORKGROUP`](@ref), in which
case return the default for `backend` (256 on CUDA/ROCm/oneAPI, 64 on Metal
and CPU).
"""
function resolve_workgroup(backend, workgroup::Int)
    workgroup == KA_AUTO_WORKGROUP || return workgroup
    return lock(_CACHE_LOCK) do
        get!(() -> _backend_default_workgroup(backend), _WORKGROUP_CACHE, typeof(backend))
    end
end

# Backend-agnostic M2M building blocks (GPU-native, no host round-trip):
# device forms of the host primitives `_gather_rotate_z!`,
# `_rotate_z_scatter_accumulate!` and `_stacked_y_dense!` in
# src/translate_batched.jl, running on any KernelAbstractions backend.


@kernel function ka_gather_rotate_z_kernel!(dst, @Const(src), @Const(flat_idx), @Const(cols),
                                             @Const(row_m), @Const(row_ssign), @Const(row_pair),
                                             @Const(phis), sgn)
    i = @index(Global)
    nrow = size(dst, 1)
    @inbounds begin
        # Int32 decode: a 64-bit divide per element is emulated on Metal
        i32 = Int32(i) - Int32(1); nrow32 = Int32(nrow)
        row = i32 % nrow32 + Int32(1)
        col = i32 ÷ nrow32 + Int32(1)
        s, c = sincos(row_m[row] * phis[col])
        a = src[flat_idx[row], cols[col]]
        b = src[flat_idx[row_pair[row]], cols[col]]
        dst[row, col] = c * a + sgn * row_ssign[row] * s * b
    end
end

"""
    ka_gather_rotate_z!(dst, src, flat_idx, cols, row_m, row_ssign, row_pair, phis, sgn; workgroup=KA_AUTO_WORKGROUP)

Device form of `_gather_rotate_z!` (src/translate_batched.jl): fused
flat-column gather + z-axis rotation stage of the M2M/M2L source alignment.
`sgn = inverse ? -1 : 1`, matching the host convention.
"""
function ka_gather_rotate_z!(dst, src, flat_idx, cols, row_m, row_ssign, row_pair, phis, sgn; workgroup=KA_AUTO_WORKGROUP)
    length(dst) == 0 && return dst
    backend = KA.get_backend(dst)
    kernel = _cached_kernel(ka_gather_rotate_z_kernel!, backend, workgroup)
    kernel(dst, src, flat_idx, cols, row_m, row_ssign, row_pair, phis, sgn; ndrange=length(dst))
    return dst
end

@kernel function ka_rotate_z_scatter_accumulate_kernel!(dest, @Const(slab), @Const(flat_idx),
                                                         @Const(col_targets), @Const(row_m),
                                                         @Const(row_ssign), @Const(row_pair), @Const(phis))
    i = @index(Global)
    nrow = size(slab, 1)
    ncol = size(slab, 2)
    @inbounds begin
        # Int32 decode: a 64-bit divide per element is emulated on Metal
        i32 = Int32(i) - Int32(1); nrow32 = Int32(nrow)
        row = i32 % nrow32 + Int32(1)
        col = i32 ÷ nrow32 + Int32(1)
        t = col_targets[col]
        # One thread per run of adjacent columns sharing a target (an M2M
        # parent's children are adjacent in node order) sums the run in column
        # order and adds once, so the sum does not depend on thread arrival
        # order. The add stays atomic: a target split across runs is still
        # summed correctly, just not reproducibly.
        if col == 1 || col_targets[col - 1] != t
            acc = zero(eltype(dest))
            k = col
            while k <= ncol && col_targets[k] == t
                s, c = sincos(row_m[row] * phis[k])
                acc += c * slab[row, k] - row_ssign[row] * s * slab[row_pair[row], k]
                k += 1
            end
            KA.@atomic dest[flat_idx[row], t] += acc
        end
    end
end

# Two-phase form for long runs (an M2L target receives hundreds of routes, so
# one thread per run would sum them serially). Phase 1: one thread per (row,
# tile of _KA_SCATTER_TILE columns) sums each run segment inside its tile, in
# column order, into `partial` at the segment's first column, and records
# where each tile's last segment starts (`lastseg`). Phase 2: one thread per
# (row, run) adds the run's segment partials in order, crossing into the next
# tile only when its current segment is that tile's last one and the next tile
# opens with the same target. Both orders are fixed, so the result is
# reproducible; any column order is summed correctly.
const _KA_SCATTER_TILE = 32

# grow-only per-backend Int32 scratch for `lastseg`
const _KA_SCATTER_LASTSEG = IdDict{Any,Any}()
function _ka_scatter_lastseg(backend, ntiles::Int)
    v = get(_KA_SCATTER_LASTSEG, typeof(backend), nothing)
    if v === nothing || length(v) < ntiles
        v = KA.allocate(backend, Int32, max(ntiles, 2 * (v === nothing ? 0 : length(v))))
        _KA_SCATTER_LASTSEG[typeof(backend)] = v
    end
    return v
end

# the z-rotation factors of (row, column k): computed from the column's angle,
# or read from a per-class table when `row_m` is `(RC, RS)` and `phis` the
# column classes
@inline _ka_rot_sc(row_m, phis, row, k) = sincos(row_m[row] * phis[k])
@inline function _ka_rot_sc(tab::Tuple, cls, row, k)
    @inbounds cl = cls[k]
    @inbounds return tab[2][row, cl], tab[1][row, cl]
end

@kernel function ka_rotate_z_scatter_partials_kernel!(partial, lastseg, @Const(slab), @Const(col_targets),
        row_m, @Const(row_ssign), @Const(row_pair), phis, ntiles)
    i = @index(Global)
    nrow = size(slab, 1)
    ncol = size(slab, 2)
    @inbounds if i <= nrow * ntiles
        i32 = Int32(i) - Int32(1); nrow32 = Int32(nrow)
        row = i32 % nrow32 + Int32(1)
        tile = i32 ÷ nrow32
        k0 = Int(tile) * _KA_SCATTER_TILE + 1
        k1 = min(k0 + _KA_SCATTER_TILE - 1, ncol)
        acc = zero(eltype(partial))
        start = k0
        for k in k0:k1
            if k > k0 && col_targets[k] != col_targets[k - 1]
                partial[row, start] = acc
                acc = zero(eltype(partial))
                start = k
            end
            s, c = _ka_rot_sc(row_m, phis, row, k)
            acc += c * slab[row, k] - row_ssign[row] * s * slab[row_pair[row], k]
        end
        partial[row, start] = acc
        row == 1 && (lastseg[tile + 1] = Int32(start))
    end
end

@kernel function ka_scatter_runs_kernel!(dest, @Const(partial), @Const(lastseg), @Const(flat_idx),
        @Const(col_targets))
    i = @index(Global)
    nrow = size(partial, 1)
    ncol = size(partial, 2)
    @inbounds if i <= nrow * ncol
        i32 = Int32(i) - Int32(1); nrow32 = Int32(nrow)
        row = i32 % nrow32 + Int32(1)
        col = Int(i32 ÷ nrow32) + 1
        t = col_targets[col]
        if col == 1 || col_targets[col - 1] != t
            acc = zero(eltype(dest))
            k = col
            while true
                acc += partial[row, k]
                tile = (k - 1) ÷ _KA_SCATTER_TILE          # 0-based tile of segment k
                knext = (tile + 1) * _KA_SCATTER_TILE + 1    # next tile's first column
                # continue only if segment k runs to its tile's end and the run
                # carries on into the next tile
                (Int(lastseg[tile + 1]) == k && knext <= ncol && col_targets[knext] == t) || break
                k = knext
            end
            KA.@atomic dest[flat_idx[row], t] += acc
        end
    end
end

"""
    ka_rotate_z_scatter_accumulate!(dest, slab, flat_idx, col_targets, row_m, row_ssign, row_pair, phis; workgroup=KA_AUTO_WORKGROUP)

Device form of `_rotate_z_scatter_accumulate!` (src/translate_batched.jl):
inverse z-rotation fused with an
accumulating scatter back into the flat coefficient buffer (the
M2M/M2L "return alignment" stage). Columns sharing a target must be adjacent
for the sum to be reproducible (see the kernel). `partial` (a slab-sized
buffer) selects the two-phase form for long runs (M2L).
"""
function ka_rotate_z_scatter_accumulate!(dest, slab, flat_idx, col_targets, row_m, row_ssign, row_pair, phis;
        workgroup=KA_AUTO_WORKGROUP, partial=nothing, rot=nothing)
    length(slab) == 0 && return dest
    backend = KA.get_backend(dest)
    if partial === nothing
        kernel = _cached_kernel(ka_rotate_z_scatter_accumulate_kernel!, backend, workgroup)
        kernel(dest, slab, flat_idx, col_targets, row_m, row_ssign, row_pair, phis; ndrange=length(slab))
        return dest
    end
    size(partial) == size(slab) || throw(DimensionMismatch(
        "scatter partial buffer $(size(partial)) must match the slab $(size(slab))"))
    ntiles = cld(size(slab, 2), _KA_SCATTER_TILE)
    lastseg = _ka_scatter_lastseg(backend, ntiles)
    k1 = _cached_kernel(ka_rotate_z_scatter_partials_kernel!, backend, workgroup)
    # `rot = ((RC, RS), cls)`: per-class rotation table instead of sincos per column
    rm, ph = rot === nothing ? (row_m, phis) : rot
    k1(partial, lastseg, slab, col_targets, rm, row_ssign, row_pair, ph, ntiles;
       ndrange=size(slab, 1) * ntiles)
    k2 = _cached_kernel(ka_scatter_runs_kernel!, backend, workgroup)
    k2(dest, partial, lastseg, flat_idx, col_targets; ndrange=length(slab))
    return dest
end

# Slab pointwise kernels. Base broadcasts over ndof-row views ran 3-5x below
# bandwidth on Metal (49-row slabs, strided views, Int64 index math); these
# decode (row, col) in Int32 from a flat index instead.
@kernel function ka_stacked_combine_kernel!(G2, @Const(G), @Const(C), @Const(S), nd::Int32)
    i = @index(Global)
    @inbounds if i <= length(G2)
        i32 = Int32(i) - Int32(1); nr = nd + nd
        row = i32 % nr + Int32(1)
        col = i32 ÷ nr + Int32(1)
        if row <= nd
            G2[row, col] = C[row, col] * G[row, col] - S[row, col] * G[row + nd, col]
        else
            r = row - nd
            G2[row, col] = S[r, col] * G[r, col] + C[r, col] * G[row, col]
        end
    end
end

@kernel function ka_scale_inplace_kernel!(Y, @Const(Sc))
    i = @index(Global)
    @inbounds if i <= length(Y)
        i32 = Int32(i) - Int32(1); nr = Int32(size(Y, 1))
        row = i32 % nr + Int32(1)
        col = i32 ÷ nr + Int32(1)
        Y[row, col] *= Sc[row, col]
    end
end

@kernel function ka_trig_fill_kernel!(C, S, @Const(nu), @Const(theta))
    i = @index(Global)
    @inbounds if i <= length(C)
        i32 = Int32(i) - Int32(1); nr = Int32(size(C, 1))
        row = i32 % nr + Int32(1)
        col = i32 ÷ nr + Int32(1)
        th = nu[row] * theta[col]
        C[row, col] = cos(th)
        S[row, col] = sin(th)
    end
end

function ka_scale_inplace!(Y, Sc)
    kernel = _cached_kernel(ka_scale_inplace_kernel!, KA.get_backend(Y), 256)
    kernel(Y, Sc; ndrange=length(Y))
    return Y
end

# Per-class-table forms of the combine and scale kernels (concat M2L): the
# trig factors are read from the class tables by column class instead of from
# per-column slabs, so the device plan holds no trig slabs.
@kernel function ka_stacked_combine_tab_kernel!(G2, @Const(G), @Const(TC), @Const(TS), @Const(cls), nd::Int32)
    i = @index(Global)
    @inbounds if i <= length(G2)
        i32 = Int32(i) - Int32(1); nr = nd + nd
        row = i32 % nr + Int32(1)
        col = i32 ÷ nr + Int32(1)
        cl = cls[col]
        if row <= nd
            G2[row, col] = TC[row, cl] * G[row, col] - TS[row, cl] * G[row + nd, col]
        else
            r = row - nd
            G2[row, col] = TS[r, cl] * G[r, col] + TC[r, cl] * G[row, col]
        end
    end
end

@kernel function ka_scale_tab_kernel!(Y, @Const(TSc), @Const(cls))
    i = @index(Global)
    @inbounds if i <= length(Y)
        i32 = Int32(i) - Int32(1); nr = Int32(size(Y, 1))
        row = i32 % nr + Int32(1)
        col = i32 ÷ nr + Int32(1)
        Y[row, col] *= TSc[row, cls[col]]
    end
end

function ka_scale_tab!(Y, (_, _, TSc), cls)
    length(Y) == 0 && return Y
    _cached_kernel(ka_scale_tab_kernel!, KA.get_backend(Y), 256)(Y, TSc, cls; ndrange=length(Y))
    return Y
end

function ka_stacked_y_dense_tab!(out_slab, in_slab, Ur, Vs, (TC, TS, _), cls, G, G2, ndof::Integer)
    mul!(G, Vs, in_slab)
    kernel = _cached_kernel(ka_stacked_combine_tab_kernel!, KA.get_backend(G2), 256)
    kernel(G2, G, TC, TS, cls, Int32(ndof); ndrange=length(G2))
    mul!(out_slab, Ur, G2)
    return out_slab
end

# The first `rows * n` entries of `buf` as a contiguous rows x n matrix (a plain
# device array on Metal and CUDA): lets channels of different widths share one
# stacked-y scratch buffer.
# `vec` and a contiguous GPUArrays `view` each return a derived array with its own
# finalizer; unpreserved, either temporary can be finalized (its reference marked
# freed) inside the next `reshape` before that copies it, an intermittent
# "Attempt to copy a freed reference" on Metal.
function _ka_slab(buf, rows::Integer, n::Integer)
    v = vec(buf)
    w = view(v, 1:(rows * n))
    return GC.@preserve v w reshape(w, rows, n)
end

# C, S = cos/sin(nu * theta') as slabs (ndof x n); `theta` is the plain vector.
function ka_trig_fill!(C, S, nu, theta)
    kernel = _cached_kernel(ka_trig_fill_kernel!, KA.get_backend(C), 256)
    kernel(C, S, nu, theta; ndrange=length(C))
    return C
end

# Both channels' trig tables over the same angles in one launch: elements
# 1:length(C) fill (C, S) from `nu`, the rest fill (Cc, Sc) from `nuc`.
@kernel function ka_trig_fill2_kernel!(C, S, Cc, Sc, @Const(nu), @Const(nuc), @Const(theta))
    i = @index(Global)
    nphi = Int32(length(C))
    @inbounds if i <= length(C) + length(Cc)
        i32 = Int32(i) - Int32(1)
        if i32 < nphi
            nr = Int32(size(C, 1))
            row = i32 % nr + Int32(1); col = i32 ÷ nr + Int32(1)
            th = nu[row] * theta[col]
            C[row, col] = cos(th); S[row, col] = sin(th)
        else
            j32 = i32 - nphi; nr = Int32(size(Cc, 1))
            row = j32 % nr + Int32(1); col = j32 ÷ nr + Int32(1)
            th = nuc[row] * theta[col]
            Cc[row, col] = cos(th); Sc[row, col] = sin(th)
        end
    end
end

function ka_trig_fill2!(C, S, Cc, Sc, nu, nuc, theta)
    kernel = _cached_kernel(ka_trig_fill2_kernel!, KA.get_backend(C), 256)
    kernel(C, S, Cc, Sc, nu, nuc, theta; ndrange=length(C) + length(Cc))
    return C
end

# The stage-group Lamb-Helmholtz row mix in one launch:
#   cphi = zphi + phi_rows .* zchi[phi_pair, :]
#   cchi = zchi + chi_rows .* zchi[chi_src, :]
# rows 1:ndphi of the flattened range write cphi, the rest cchi.
@kernel function ka_lh_row_mix_kernel!(cphi, cchi, @Const(zphi), @Const(zchi),
        @Const(phi_rows), @Const(chi_rows), @Const(phi_pair), @Const(chi_src))
    i = @index(Global)
    ndphi = Int32(size(cphi, 1)); ndchi = Int32(size(cchi, 1))
    nd = ndphi + ndchi
    @inbounds if i <= Int(nd) * size(cphi, 2)
        i32 = Int32(i) - Int32(1)
        row = i32 % nd + Int32(1); col = i32 ÷ nd + Int32(1)
        if row <= ndphi
            cphi[row, col] = zphi[row, col] + phi_rows[row] * zchi[phi_pair[row], col]
        else
            r = row - ndphi
            cchi[r, col] = zchi[r, col] + chi_rows[r] * zchi[chi_src[r], col]
        end
    end
end

function ka_lh_row_mix!(cphi, cchi, zphi, zchi, phi_rows, chi_rows, phi_pair, chi_src)
    n = (size(cphi, 1) + size(cchi, 1)) * size(cphi, 2)
    n == 0 && return cphi
    kernel = _cached_kernel(ka_lh_row_mix_kernel!, KA.get_backend(cphi), 256)
    kernel(cphi, cchi, zphi, zchi, phi_rows, chi_rows, phi_pair, chi_src; ndrange=n)
    return cphi
end

"""
    ka_stacked_y_dense!(out_slab, in_slab, Ur, Vs, C, S, G, G2, ndof)

Device form of `_stacked_y_dense!` (src/translate_batched.jl):
the dense y-rotation application used by the resident M2M/L2L stage groups
and the concat M2L.
`LinearAlgebra.mul!` dispatches to each backend's own GPU matmul (Metal's
MPS-backed `mul!` for `MtlArray`, CUBLAS for `CuArray`); the elementwise C/S
combine runs in `ka_stacked_combine_kernel!`.
"""
function ka_stacked_y_dense!(out_slab, in_slab, Ur, Vs, C, S, G, G2, ndof::Integer)
    mul!(G, Vs, in_slab)
    kernel = _cached_kernel(ka_stacked_combine_kernel!, KA.get_backend(G2), 256)
    kernel(G2, G, C, S, Int32(ndof); ndrange=length(G2))
    mul!(out_slab, Ur, G2)
    return out_slab
end

@kernel function ka_gather_rows_kernel!(dst, @Const(src), @Const(rows))
    i = @index(Global)
    nrow = size(dst, 1)
    @inbounds begin
        # Int32 decode: a 64-bit divide per element is emulated on Metal
        i32 = Int32(i) - Int32(1); nrow32 = Int32(nrow)
        row = i32 % nrow32 + Int32(1)
        col = i32 ÷ nrow32 + Int32(1)
        dst[row, col] = src[rows[row], col]
    end
end

@kernel function ka_gather_values_kernel!(dst, @Const(src), @Const(ids))
    i = @index(Global)
    @inbounds dst[i] = src[ids[i]]
end

"""
    ka_gather_values!(dst, src, ids; workgroup=KA_AUTO_WORKGROUP)

Device form of `_gather_values!` (src/translate_batched.jl): allocation-free
value gather `dst[i] = src[ids[i]]`, used by the M2L concat plan's per-chunk column
parameter gather (phi/theta/r/invr) from the per-class geometry tables.
"""
function ka_gather_values!(dst, src, ids; workgroup=KA_AUTO_WORKGROUP)
    length(dst) == 0 && return dst
    backend = KA.get_backend(dst)
    kernel = _cached_kernel(ka_gather_values_kernel!, backend, workgroup)
    kernel(dst, src, ids; ndrange=length(dst))
    return dst
end

"""
    ka_gather_rows!(dst, src, rows; workgroup=KA_AUTO_WORKGROUP)

Device form of `_gather_rows!` (src/translate_batched.jl): allocation-free
row gather `dst[i, :] = src[rows[i], :]`, used by the Lamb-Helmholtz row-mix stage of
`_resident_stage_group_apply!`.
"""
function ka_gather_rows!(dst, src, rows; workgroup=KA_AUTO_WORKGROUP)
    length(dst) == 0 && return dst
    backend = KA.get_backend(dst)
    kernel = _cached_kernel(ka_gather_rows_kernel!, backend, workgroup)
    kernel(dst, src, rows; ndrange=length(dst))
    return dst
end

#------- FUSED POINTWISE M2L PRIMITIVES -------#
#
# Device forms of the host launcher's fused helpers `_prefix_trig_scale!` and
# `_prefix_lh_mix!` (src/translate_batched.jl), plus a three-way column gather.
# Spelled out as separate broadcasts and `ka_gather_rows!` calls, this work --
# pure elementwise reindexing -- took 14 launches per chunk. Profiled on a VPM
# wake at np=8192 (11650 routes):
# the apply is neither bandwidth- nor FLOP-bound at that size (6-30 GB/s of
# ~100; stacked_y at ~0.22 of ~3.6 TFLOP/s), and 1.69 ms of its 8.36 ms is
# fixed per-launch cost across 22 launches. So pass COUNT is the lever here,
# not traffic per pass.

@kernel function ka_gather_values3_kernel!(d1, d2, d3, @Const(s1), @Const(s2),
        @Const(s3), @Const(ids))
    j = @index(Global)
    @inbounds begin
        id = ids[j]
        d1[j] = s1[id]
        d2[j] = s2[id]
        d3[j] = s3[id]
    end
end

"""
    ka_gather_values3!(d1, d2, d3, s1, s2, s3, ids; workgroup=KA_AUTO_WORKGROUP)

Three `ka_gather_values!` calls sharing one index vector, done in one launch and
one `ids` read per column. Used for the concat plan's (phi, theta, invr) column
parameter gather; a Lamb-Helmholtz plan also gathers `r`, through
`ka_gather_values4!`, since `col_r` is absent on a non-LH plan.
"""
function ka_gather_values3!(d1, d2, d3, s1, s2, s3, ids; workgroup=KA_AUTO_WORKGROUP)
    n = length(d1)
    n == 0 && return d1
    backend = KA.get_backend(d1)
    kernel = _cached_kernel(ka_gather_values3_kernel!, backend, workgroup)
    kernel(d1, d2, d3, s1, s2, s3, ids; ndrange=n)
    return d1
end

@kernel function ka_gather_values4_kernel!(d1, d2, d3, d4, @Const(s1), @Const(s2),
        @Const(s3), @Const(s4), @Const(ids))
    j = @index(Global)
    @inbounds begin
        id = ids[j]
        d1[j] = s1[id]
        d2[j] = s2[id]
        d3[j] = s3[id]
        d4[j] = s4[id]
    end
end

"`ka_gather_values3!` plus a fourth vector, in one launch."
function ka_gather_values4!(d1, d2, d3, d4, s1, s2, s3, s4, ids; workgroup=KA_AUTO_WORKGROUP)
    n = length(d1)
    n == 0 && return d1
    backend = KA.get_backend(d1)
    kernel = _cached_kernel(ka_gather_values4_kernel!, backend, workgroup)
    kernel(d1, d2, d3, d4, s1, s2, s3, s4, ids; ndrange=n)
    return d1
end

@kernel function ka_prefix_trig_scale_kernel!(C, S, scale, @Const(nu),
        @Const(theta), @Const(invr), @Const(rexp), nrow::Int)
    i = @index(Global)
    @inbounds begin
        # Int32 decode: a 64-bit divide per element is emulated on Metal
        i32 = Int32(i) - Int32(1); nrow32 = Int32(nrow)
        row = i32 % nrow32 + Int32(1)
        col = i32 ÷ nrow32 + Int32(1)
        th = nu[row] * theta[col]
        C[row, col] = cos(th)
        S[row, col] = sin(th)
        scale[row, col] = invr[col]^rexp[row]
    end
end

"""
    ka_prefix_trig_scale!(C, S, scale, nu, theta, invr, rexp; workgroup=KA_AUTO_WORKGROUP)

Device form of `_prefix_trig_scale!` (src/translate_batched.jl): the
per-column rotation table `C = cos(nu*theta)`, `S = sin(nu*theta)` and the
radial scaling `scale[i, j] = invr[j]^rexp[i]`, in one pass over the slab
instead of three broadcasts. `C`, `S` and `scale` must be `length(nu)`-row views
of the same column range as `theta` and `invr`.
"""
function ka_prefix_trig_scale!(C, S, scale, nu, theta, invr, rexp;
        workgroup=KA_AUTO_WORKGROUP)
    nrow = length(nu)
    ncols = length(theta)
    (nrow == 0 || ncols == 0) && return scale
    backend = KA.get_backend(scale)
    kernel = _cached_kernel(ka_prefix_trig_scale_kernel!, backend, workgroup)
    kernel(C, S, scale, nu, theta, invr, rexp, nrow; ndrange=nrow * ncols)
    return scale
end

@kernel function ka_prefix_lh_mix_kernel!(cphi, cchi, @Const(zphi), @Const(zchi),
        @Const(arow), @Const(brow), @Const(rs), @Const(phi_pair), @Const(chi_up),
        ndphi::Int, ndchi::Int)
    i = @index(Global)
    @inbounds begin
        nd = ndphi + ndchi
        i32 = Int32(i) - Int32(1); nd32 = Int32(nd)
        row = i32 % nd32 + Int32(1)
        col = i32 ÷ nd32 + Int32(1)
        r = rs[col]
        if row <= ndphi
            cphi[row, col] = zphi[row, col] + arow[row] * r * zchi[phi_pair[row], col]
        else
            k = row - ndphi
            cchi[k, col] = zchi[k, col] + brow[k] * r * zchi[chi_up[k], col]
        end
    end
end

"""
    ka_prefix_lh_mix!(cphi, cchi, zphi, zchi, arow, brow, rs, phi_pair, chi_up; workgroup=KA_AUTO_WORKGROUP)

Device form of `_prefix_lh_mix!` (src/translate_batched.jl): the
Lamb-Helmholtz row mix

    cphi[i, j] = zphi[i, j] + arow[i] * rs[j] * zchi[phi_pair[i], j]
    cchi[i, j] = zchi[i, j] + brow[i] * rs[j] * zchi[chi_up[i],   j]

in one launch. The row gather is folded into the read, so the two staging slabs
(`lhgp`, `lhgu`) and the two `ka_gather_rows!` passes that filled them are not
needed. Both outputs are written by one ndrange, split at row `ndphi`.
"""
function ka_prefix_lh_mix!(cphi, cchi, zphi, zchi, arow, brow, rs, phi_pair,
        chi_up; workgroup=KA_AUTO_WORKGROUP)
    ndphi = size(cphi, 1)
    ndchi = size(cchi, 1)
    ncols = length(rs)
    (ncols == 0 || ndphi + ndchi == 0) && return cchi
    backend = KA.get_backend(cchi)
    kernel = _cached_kernel(ka_prefix_lh_mix_kernel!, backend, workgroup)
    kernel(cphi, cchi, zphi, zchi, arow, brow, rs, phi_pair, chi_up, ndphi,
        ndchi; ndrange=(ndphi + ndchi) * ncols)
    return cchi
end

@kernel function ka_fill_invperm_kernel!(invperm, @Const(perm), n)
    sorted_i = @index(Global)
    @inbounds if sorted_i <= n
        invperm[perm[sorted_i]] = sorted_i
    end
end

"""
    ka_fill_invperm!(invperm, perm; workgroup=KA_AUTO_WORKGROUP)

Scatter the inverse of the body sort permutation, `invperm[perm[i]] = i`, so a global
body ordinal maps back to its sorted slot. `invperm` is written over `1:length(perm)`
and must be at least that long; `perm` must be a genuine permutation of `1:n` (each
slot is written exactly once, so a non-permutation silently leaves stale entries).
"""
function ka_fill_invperm!(invperm, perm; workgroup::Int=KA_AUTO_WORKGROUP)
    n = length(perm)
    n == 0 && return invperm
    length(invperm) >= n || throw(ArgumentError(
        "invperm (length $(length(invperm))) is shorter than perm (length $n)"))
    backend = KA.get_backend(invperm)
    kernel = _cached_kernel(ka_fill_invperm_kernel!, backend, workgroup)
    kernel(invperm, perm, n; ndrange=n)
    return invperm
end

@kernel function ka_pack_body_matrix_kernel!(body, @Const(source_buffer), @Const(perm),
        @Const(body_system), @Const(body_index), isys, n, nrows, nsys,
        sigma_row, inv_sigma_row)
    sorted_i = @index(Global)
    @inbounds if sorted_i <= n
        global_i = perm[sorted_i]
        if body_system[global_i] == isys
            ibody = body_index[global_i]
            for row in 1:nsys
                body[row, sorted_i] = source_buffer[row, ibody]
            end
            for row in (nsys + 1):nrows
                body[row, sorted_i] = zero(eltype(body))
            end
            # Reciprocal-sigma row: one divide per body here replaces one divide
            # per target-source INTERACTION in the nearfield kernel (measured 15.6%
            # of the nearfield at np=248714). Stored as 0 for a non-positive sigma
            # so the nearfield's `sigma > 0` regularization guard becomes an
            # exactly equivalent `inv_sigma > 0` test on the row it already loads.
            if inv_sigma_row > 0
                sig = body[sigma_row, sorted_i]
                body[inv_sigma_row, sorted_i] =
                    sig > zero(sig) ? inv(sig) : zero(sig)
            end
        end
    end
end

"""
    ka_pack_body_matrix!(body, source_buffer, perm, body_system, body_index, n;
                         isys=1, sigma_row=0, inv_sigma_row=0,
                         workgroup=KA_AUTO_WORKGROUP)

Gather one source system's bodies out of its
global-ordinal `source_buffer` into `body`, the sorted-order `dpb x n` matrix the
resident lifecycle reads. Column `sorted_i` of `body` takes buffer column
`body_index[perm[sorted_i]]`, and only for slots this system owns
(`body_system[perm[sorted_i]] == isys`).

Canonical all-rows packed layout: every source-buffer row is carried,
including radius row 4, and a system narrower than `body` is zero-padded. Call once
per system.

`n` is the *logical* body count: `perm` is capacity-sized in the KA context, so its
`length` is not the extent. Columns beyond `n`, and columns this system does not own,
are left untouched.

With `inv_sigma_row > 0`, row `inv_sigma_row` of each packed column receives
`1/body[sigma_row, :]` (0 where sigma <= 0), the reciprocal-sigma row the
regularized nearfield multiplies by. It requires `0 < sigma_row <=
size(source_buffer, 1)` and `inv_sigma_row <= size(body, 1)`.
"""
function ka_pack_body_matrix!(body, source_buffer, perm, body_system, body_index,
        n::Int; isys::Integer=1, sigma_row::Integer=0, inv_sigma_row::Integer=0,
        workgroup::Int=KA_AUTO_WORKGROUP)
    n == 0 && return body
    size(body, 2) >= n || throw(ArgumentError(
        "body has $(size(body, 2)) columns, fewer than n=$n"))
    n <= length(perm) || throw(ArgumentError(
        "n=$n exceeds perm (length $(length(perm)))"))
    nrows = size(body, 1)
    nsys = min(size(source_buffer, 1), nrows)
    inv_sigma_row == 0 || (0 < sigma_row <= nsys && inv_sigma_row <= nrows) ||
        throw(ArgumentError(
            "inv_sigma_row=$inv_sigma_row needs 0 < sigma_row=$sigma_row <= $nsys " *
            "and inv_sigma_row <= nrows=$nrows"))
    backend = KA.get_backend(body)
    kernel = _cached_kernel(ka_pack_body_matrix_kernel!, backend, workgroup)
    kernel(body, source_buffer, perm, body_system, body_index, Int(isys), n, nrows, nsys,
        Int(sigma_row), Int(inv_sigma_row); ndrange=n)
    return body
end

# --- Integration with FastMultipole's real M2M call path ---
#
# `ka_resident_stage_group_apply!` mirrors `_resident_stage_group_apply!`
# (src/translate_batched.jl) -- the real production M2M/L2L group-apply
# reached via `_launch_resident_m2m!` -- substituting the four KA building blocks
# above for its host primitives. It operates on the same
# `FlatCoefficientBuffer`/`ResidentOperatorGroup`/`ResidentOperatorWorkspace` types,
# so it can be dropped in wherever the CPU function is called, backed by any
# KernelAbstractions array (Metal, CUDA, or plain CPU Array).
#
# This DOES use the `count[]`-prefix views (`_vector_prefix_view`/`_matrix_col_view`),
# exactly as the CPU function does. In the production lifecycle `ka_lifecycle_body!`
# loops over `ws.m2m_groups`/`ws.l2l_groups` from a real workspace whose scratch is
# sized to `max_batch`, the max over levels, so most groups are SHORTER than the
# scratch. Without the views that is a `DimensionMismatch` where the shapes disagree
# and, worse, stale trailing columns scattered into `dest` where they happen to
# broadcast. Both helpers return the array unchanged when the size already matches.
function ka_resident_stage_group_apply!(dest, src, group, ws, kind::Symbol)
    n = group.count[]
    n == 0 && return dest
    # the scratch may be narrower than the group (a device workspace caps it at
    # `stage_batch`): walk the group in column chunks. A group's sources and
    # targets sit on different levels, so the chunks are independent.
    cap = size(ws.aphi, 2)
    for c0 in 1:cap:n
        _ka_stage_group_cols!(dest, src, group, ws, kind, c0:min(c0 + cap - 1, n))
    end
    return dest
end

function _ka_stage_group_cols!(dest, src, group, ws, kind::Symbol, cols::UnitRange{Int})
    n = length(cols)
    mult = kind === :m2m
    ystk = ws.ystk_phi
    Ur = mult ? ystk.mult_Ur : ystk.loc_Ur
    Vs = mult ? ystk.mult_Vs : ystk.loc_Vs
    ndof_phi = size(ws.aphi, 1)
    source_idx = view(group.source_idx, cols)
    target_idx = view(group.target_idx, cols)
    group_phis = view(group.phis, cols)
    group_thetas = view(group.thetas, cols)
    aphi = FastMultipole._matrix_col_view(ws.aphi, n)
    yphi = FastMultipole._matrix_col_view(ws.yphi, n)
    zphi = FastMultipole._matrix_col_view(ws.zphi, n)
    rphi = FastMultipole._matrix_col_view(ws.rphi, n)
    cphi = FastMultipole._matrix_col_view(ws.cphi, n)
    C = FastMultipole._matrix_col_view(ystk.Cy, n)
    S = FastMultipole._matrix_col_view(ystk.Sy, n)
    G = FastMultipole._matrix_col_view(ystk.G, n)
    G2 = FastMultipole._matrix_col_view(ystk.G2, n)
    has_lh = size(dest.chi, 1) > 0
    if has_lh
        Cc = FastMultipole._matrix_col_view(ws.ystk_chi.Cy, n)
        Sc = FastMultipole._matrix_col_view(ws.ystk_chi.Sy, n)
        ka_trig_fill2!(C, S, Cc, Sc, ystk.nu, ws.ystk_chi.nu, group_thetas)
    else
        ka_trig_fill!(C, S, ystk.nu, group_thetas)
    end
    ka_gather_rotate_z!(aphi, src.phi, ws.phi_flat_idx, source_idx,
        ws.maps_phi.row_m, ws.maps_phi.row_ssign, ws.maps_phi.row_pair, group_phis, one(eltype(aphi)))
    ka_stacked_y_dense!(yphi, aphi, Ur, Vs, C, S, G, G2, ndof_phi)
    mul!(zphi, group.phi_dense, yphi)
    ret_phi = zphi

    if has_lh
        ystk_c = ws.ystk_chi
        Urc = mult ? ystk_c.mult_Ur : ystk_c.loc_Ur
        Vsc = mult ? ystk_c.mult_Vs : ystk_c.loc_Vs
        ndof_chi = size(ws.achi, 1)
        achi = FastMultipole._matrix_col_view(ws.achi, n)
        ychi = FastMultipole._matrix_col_view(ws.ychi, n)
        zchi = FastMultipole._matrix_col_view(ws.zchi, n)
        rchi = FastMultipole._matrix_col_view(ws.rchi, n)
        cchi = FastMultipole._matrix_col_view(ws.cchi, n)
        Gc = FastMultipole._matrix_col_view(ystk_c.G, n)
        G2c = FastMultipole._matrix_col_view(ystk_c.G2, n)
        ka_gather_rotate_z!(achi, src.chi, ws.chi_flat_idx, source_idx,
            ws.maps_chi.row_m, ws.maps_chi.row_ssign, ws.maps_chi.row_pair, group_phis, one(eltype(achi)))
        ka_stacked_y_dense!(ychi, achi, Urc, Vsc, Cc, Sc, Gc, G2c, ndof_chi)
        mul!(zchi, group.chi_dense, ychi)
        chi_rows = mult ? ws.maps_chi.row_down : ws.maps_chi.row_up
        ka_lh_row_mix!(cphi, cchi, zphi, zchi, group.lh_phi_rows, group.lh_chi_rows,
            ws.maps_phi.row_pair, chi_rows)
        ka_stacked_y_dense!(rchi, cchi, Urc, Vsc, Cc, Sc, Gc, G2c, ndof_chi)
        ka_rotate_z_scatter_accumulate!(dest.chi, rchi, ws.chi_flat_idx, target_idx,
            ws.maps_chi.row_m, ws.maps_chi.row_ssign, ws.maps_chi.row_pair, group_phis)
        ret_phi = cphi
    end

    ka_stacked_y_dense!(rphi, ret_phi, Ur, Vs, C, S, G, G2, ndof_phi)
    ka_rotate_z_scatter_accumulate!(dest.phi, rphi, ws.phi_flat_idx, target_idx,
        ws.maps_phi.row_m, ws.maps_phi.row_ssign, ws.maps_phi.row_pair, group_phis)
    # No sync here: every kernel above is queue-ordered against the caller's
    # next launch on this same backend, and this driver is called once per
    # M2M/L2L group and per M2L route set -- a barrier here is the per-stage
    # sync `ka_lifecycle_body!` exists to avoid (measured 1.37-2.7x on this
    # code). The single end-of-step sync in `ka_radix_cache_device_step!`
    # covers the host readback.
    return dest
end

# --- M2L (horizontal pass) ---
#
# The device M2L path is `ConcatenatedFixedZM2L`, the `_launch_resident_m2l_concat!`
# apply (src/translate_batched.jl, `ResidentM2LConcatPlan`/`ConcatChannelOps`),
# which is what `RadixFMMCache` builds on device (src/resident/radix_cache.jl).
# The `FactoredRotationM2L`/`_resident_factored_m2l_group_apply!` path keeps
# its per-degree y-rotation blocks as plain CPU `Matrix{TF}`, so it is CPU-only.
#
# Per route chunk, `ka_resident_m2l_concat_apply!` runs:
#   - `ka_gather_values3!` (and `ka_gather_values!` for the LH `r`): the
#     per-column (phi, theta, invr) parameters, gathered by offset class;
#   - `ka_prefix_trig_scale!`: the cos/sin rotation table and radial scaling;
#   - `ka_gather_rotate_z!` -> `ka_stacked_y_dense!` (`mul!` around
#     `ka_stacked_combine_kernel!`) -> `ka_scale_inplace!` -> `mul!` with the
#     fixed z-translation `zD` -> `ka_scale_inplace!`;
#   - with Lamb-Helmholtz, `ka_prefix_lh_mix!` for the phi/chi row mix;
#   - `ka_stacked_y_dense!` back out and `ka_rotate_z_scatter_accumulate!`
#     into `dest`.
# `mul!` dispatches to each backend's own GPU matmul.

#------- per-class M2L tables -------#
#
# The concat M2L's per-route trig (cos/sin(nu*theta), invr^rexp) and z-rotation
# (sincos(m*phi)) depend only on the route's offset class, of which there are a
# few hundred against millions of routes. They are tabulated once per plan --
# with the same expressions the per-route kernels evaluated, so the result is
# bit-identical -- and gathered by class.

@kernel function ka_rot_table_kernel!(RC, RS, @Const(row_m), @Const(phis))
    i = @index(Global)
    nrow = size(RC, 1)
    @inbounds if i <= length(RC)
        i32 = Int32(i) - Int32(1); nrow32 = Int32(nrow)
        row = i32 % nrow32 + Int32(1)
        c = i32 ÷ nrow32 + Int32(1)
        s, co = sincos(row_m[row] * phis[c])
        RC[row, c] = co
        RS[row, c] = s
    end
end

@kernel function ka_gather_rotate_z_tab_kernel!(dst, @Const(src), @Const(flat_idx), @Const(cols),
        @Const(RC), @Const(RS), @Const(row_ssign), @Const(row_pair), @Const(cls), sgn)
    i = @index(Global)
    nrow = size(dst, 1)
    @inbounds if i <= length(dst)
        i32 = Int32(i) - Int32(1); nrow32 = Int32(nrow)
        row = i32 % nrow32 + Int32(1)
        col = i32 ÷ nrow32 + Int32(1)
        cl = cls[col]
        c = RC[row, cl]; s = RS[row, cl]
        a = src[flat_idx[row], cols[col]]
        b = src[flat_idx[row_pair[row]], cols[col]]
        dst[row, col] = c * a + sgn * row_ssign[row] * s * b
    end
end

function _ka_rot_table(backend, ::Type{TF}, row_m, phis) where TF
    RC = KA.allocate(backend, TF, length(row_m), length(phis))
    RS = KA.allocate(backend, TF, length(row_m), length(phis))
    length(RC) == 0 && return RC, RS
    _cached_kernel(ka_rot_table_kernel!, backend, KA_AUTO_WORKGROUP)(RC, RS, row_m, phis;
        ndrange=length(RC))
    return RC, RS
end

function _ka_trig_table(backend, ::Type{TF}, nu, thetas, invrs, rexp) where TF
    TC = KA.allocate(backend, TF, length(nu), length(thetas))
    TS = KA.allocate(backend, TF, length(nu), length(thetas))
    TSc = KA.allocate(backend, TF, length(nu), length(thetas))
    ka_prefix_trig_scale!(TC, TS, TSc, nu, thetas, invrs, rexp)
    return TC, TS, TSc
end

# per plan, keyed by the identity of its class-angle vector (a rebuilt plan
# has a new one); capped so tables of replaced plans do not accumulate
const _KA_M2L_TABLES = IdDict{Any,Any}()

function _ka_m2l_tables(plan, ws, LH::Bool)
    haskey(_KA_M2L_TABLES, plan.phis) || length(_KA_M2L_TABLES) < 8 || empty!(_KA_M2L_TABLES)
    return get!(_KA_M2L_TABLES, plan.phis) do
        backend = KA.get_backend(plan.phis)
        TF = eltype(plan.aphi)
        phi = (; trig=_ka_trig_table(backend, TF, plan.ops_phi.nu, plan.thetas, plan.invrs, plan.rexp_phi),
                 rot=_ka_rot_table(backend, TF, ws.maps_phi.row_m, plan.phis))
        chi = LH ? (; trig=_ka_trig_table(backend, TF, plan.ops_chi.nu, plan.thetas, plan.invrs, plan.rexp_chi),
                      rot=_ka_rot_table(backend, TF, ws.maps_chi.row_m, plan.phis)) : nothing
        (; phi, chi)
    end
end

function ka_gather_rotate_z_tab!(dst, src, flat_idx, cols, (RC, RS), row_ssign, row_pair, cls, sgn;
        workgroup=KA_AUTO_WORKGROUP)
    length(dst) == 0 && return dst
    backend = KA.get_backend(dst)
    _cached_kernel(ka_gather_rotate_z_tab_kernel!, backend, workgroup)(dst, src, flat_idx, cols,
        RC, RS, row_ssign, row_pair, cls, sgn; ndrange=length(dst))
    return dst
end

"""
    ka_resident_m2l_concat_apply!(dest, src, ws, route_sources, route_targets, nroutes)

Device form of `_launch_resident_m2l_concat!` (src/translate_batched.jl), the
`ConcatenatedFixedZM2L` resident M2L apply. Operates on `ws.m2l_concat`
(a `ResidentM2LConcatPlan`) plus `route_sources`/`route_targets` (GPU index vectors) directly,
rather than a full `DeviceResidentRadixState`, so it can be driven without a tree.
"""
function ka_resident_m2l_concat_apply!(dest, src, ws, route_sources, route_targets,
        nroutes::Int; route_class=nothing)
    nroutes == 0 && return dest
    plan = ws.m2l_concat
    # `route_class` defaults to the plan's own window-scoped buffer, which the
    # generate-and-apply-per-window path refills before every call. The cached
    # path passes a view into the epoch-cached class stream instead.
    route_class === nothing && nroutes > length(plan.route_class) && throw(ArgumentError(
        "ka_resident_m2l_concat_apply!: $nroutes routes but the plan's own class stream holds " *
        "$(length(plan.route_class)); pass route_class"))
    classes = route_class === nothing ? plan.route_class : route_class
    LH = size(dest.chi, 1) > 0
    TF = eltype(dest.phi)
    backend = KA.get_backend(dest.phi)   # for the optional per-step timing ticks
    tabs = _ka_m2l_tables(plan, ws, LH)
    for c0 in 1:plan.chunk:nroutes
        cols = c0:min(c0 + plan.chunk - 1, nroutes)
        n = length(cols)
        cls = @view classes[cols]
        phis = @view plan.col_phi[1:n]
        thetas = @view plan.col_theta[1:n]
        invr_col = @view plan.col_invr[1:n]
        if LH
            rs_col = @view plan.col_r[1:n]
            ka_gather_values4!(phis, thetas, invr_col, rs_col, plan.phis, plan.thetas,
                plan.invrs, plan.rs, cls)
        else
            ka_gather_values3!(phis, thetas, invr_col, plan.phis, plan.thetas,
                plan.invrs, cls)
        end
        _utick!(:m2l_vals, backend)
        src_cols = @view route_sources[cols]
        tgt_cols = @view route_targets[cols]
        # Slab reuse (a device plan is built `slim`): the trig factors come from
        # the class tables, the stacked-y scratch G/G2 is one buffer shared by
        # both channels, the output slab r reuses a (free once rotated) and the
        # scatter's partial buffer is y (free once z is formed).
        aphi = @view plan.aphi[:, 1:n]; yphi = @view plan.yphi[:, 1:n]
        zphi = @view plan.zphi[:, 1:n]; rphi = aphi
        ops_phi = plan.ops_phi
        ndof_phi = size(plan.aphi, 1)
        Gphi = _ka_slab(ops_phi.G, 2 * ndof_phi, n); G2phi = _ka_slab(ops_phi.G2, 2 * ndof_phi, n)
        ka_gather_rotate_z_tab!(aphi, src.phi, ws.phi_flat_idx, src_cols, tabs.phi.rot,
            ws.maps_phi.row_ssign, ws.maps_phi.row_pair, cls, one(TF))
        _utick!(:m2l_gather, backend)
        ka_stacked_y_dense_tab!(yphi, aphi, ops_phi.yU_mult, ops_phi.yV_mult,
            tabs.phi.trig, cls, Gphi, G2phi, ndof_phi)
        _utick!(:m2l_y_in, backend)
        ka_scale_tab!(yphi, tabs.phi.trig, cls)
        mul!(zphi, ops_phi.zD, yphi)
        ka_scale_tab!(zphi, tabs.phi.trig, cls)
        _utick!(:m2l_zgemm, backend)
        ret_phi = zphi

        if LH
            ops_chi = plan.ops_chi
            ndof_chi = size(plan.achi, 1)
            # the shared stacked-y buffer (a non-slim plan keeps the chi
            # channel's own, which is always large enough)
            gbuf = length(ops_phi.G) >= 2 * ndof_chi * n ? ops_phi : ops_chi
            Gchi = _ka_slab(gbuf.G, 2 * ndof_chi, n); G2chi = _ka_slab(gbuf.G2, 2 * ndof_chi, n)
            achi = @view plan.achi[:, 1:n]; ychi = @view plan.ychi[:, 1:n]
            zchi = @view plan.zchi[:, 1:n]; rchi = achi
            cphi = @view plan.cphi[:, 1:n]; cchi = @view plan.cchi[:, 1:n]
            ka_gather_rotate_z_tab!(achi, src.chi, ws.chi_flat_idx, src_cols, tabs.chi.rot,
                ws.maps_chi.row_ssign, ws.maps_chi.row_pair, cls, one(TF))
            ka_stacked_y_dense_tab!(ychi, achi, ops_chi.yU_mult, ops_chi.yV_mult,
                tabs.chi.trig, cls, Gchi, G2chi, ndof_chi)
            ka_scale_tab!(ychi, tabs.chi.trig, cls)
            mul!(zchi, ops_chi.zD, ychi)
            ka_scale_tab!(zchi, tabs.chi.trig, cls)
            ka_prefix_lh_mix!(cphi, cchi, zphi, zchi, plan.lh_arow_unit,
                plan.lh_brow_unit, rs_col, ws.maps_phi.row_pair,
                ws.maps_chi.row_up)
            ka_stacked_y_dense_tab!(rchi, cchi, ops_chi.yU_loc, ops_chi.yV_loc,
                tabs.chi.trig, cls, Gchi, G2chi, ndof_chi)
            ka_rotate_z_scatter_accumulate!(dest.chi, rchi, ws.chi_flat_idx, tgt_cols,
                ws.maps_chi.row_m, ws.maps_chi.row_ssign, ws.maps_chi.row_pair, phis;
                partial=ychi, rot=(tabs.chi.rot, cls))
            ret_phi = cphi
        end
        _utick!(:m2l_chi, backend)

        ka_stacked_y_dense_tab!(rphi, ret_phi, ops_phi.yU_loc, ops_phi.yV_loc,
            tabs.phi.trig, cls, Gphi, G2phi, ndof_phi)
        _utick!(:m2l_y_out, backend)
        ka_rotate_z_scatter_accumulate!(dest.phi, rphi, ws.phi_flat_idx, tgt_cols,
            ws.maps_phi.row_m, ws.maps_phi.row_ssign, ws.maps_phi.row_pair, phis;
            partial=yphi, rot=(tabs.phi.rot, cls))
        _utick!(:m2l_scatter, backend)
    end
    # No sync here: every kernel above is queue-ordered against the caller's
    # next launch on this same backend, and this driver is called once per
    # M2M/L2L group and per M2L route set -- a barrier here is the per-stage
    # sync `ka_lifecycle_body!` exists to avoid (measured 1.37-2.7x on this
    # code). The single end-of-lifecycle sync there covers the host readback.
    return dest
end

#------- Morton keys, sorted-key search, flat coefficient buffers -------#
# Shared by the grid refresh, the workspace builder and the cache build.

@inline function ka_lower_bound(keys, first, stop, key)
    lo = first
    hi = stop + 1
    while lo < hi
        mid = (lo + hi) >>> 1
        if @inbounds(keys[mid]) < key
            lo = mid + 1
        else
            hi = mid
        end
    end
    return lo
end

@inline function ka_upper_bound(keys, first, stop, key)
    lo = first
    hi = stop + 1
    while lo < hi
        mid = (lo + hi) >>> 1
        if @inbounds(keys[mid]) <= key
            lo = mid + 1
        else
            hi = mid
        end
    end
    return lo
end

@inline function ka_morton_key(ix, iy, iz, ell)
    key = UInt64(0)
    for bit in 0:(ell - 1)
        key |= (UInt64((ix >> bit) & 0x1) << (3 * bit))
        key |= (UInt64((iy >> bit) & 0x1) << (3 * bit + 1))
        key |= (UInt64((iz >> bit) & 0x1) << (3 * bit + 2))
    end
    return key
end

@inline function ka_decode_morton_key(key, ell)
    ix = 0
    iy = 0
    iz = 0
    for bit in 0:(ell - 1)
        ix |= Int((key >> (3 * bit)) & UInt64(0x1)) << bit
        iy |= Int((key >> (3 * bit + 1)) & UInt64(0x1)) << bit
        iz |= Int((key >> (3 * bit + 2)) & UInt64(0x1)) << bit
    end
    return ix, iy, iz
end

function _ka_flat_buffer(backend, ::Type{TF}, basis_info::FastMultipole.OperatorBasisInfo{B,LH},
        batch::Integer) where {TF,B,LH}
    phi = KA.zeros(backend, TF, basis_info.basis_dof_phi, batch)
    chi = LH ? KA.zeros(backend, TF, basis_info.basis_dof_chi, batch) :
        KA.zeros(backend, TF, 0, 0)
    return FastMultipole.FlatCoefficientBuffer{TF,typeof(phi),B,LH}(phi, chi, basis_info)
end
