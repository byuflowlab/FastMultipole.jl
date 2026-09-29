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

function FastMultipole._device_row_extrema(A::AnyGPUMatrix, row::Integer, n::Integer)
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
        KA.synchronize(backend)
        copyto!(hlo, lo); copyto!(hhi, hi)
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
# (one group per cell), and `ka_launch_nearfield!` takes its team shape from
# `_nf_config`: there the workgroup is the per-cell/per-pair *team size* (for
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
    @inbounds begin
        # Int32 decode: a 64-bit divide per element is emulated on Metal
        i32 = Int32(i) - Int32(1); nrow32 = Int32(nrow)
        row = i32 % nrow32 + Int32(1)
        col = i32 ÷ nrow32 + Int32(1)
        s, c = sincos(row_m[row] * phis[col])
        v = c * slab[row, col] - row_ssign[row] * s * slab[row_pair[row], col]
        KA.@atomic dest[flat_idx[row], col_targets[col]] += v
    end
end

"""
    ka_rotate_z_scatter_accumulate!(dest, slab, flat_idx, col_targets, row_m, row_ssign, row_pair, phis; workgroup=KA_AUTO_WORKGROUP)

Device form of `_rotate_z_scatter_accumulate!` (src/translate_batched.jl):
inverse z-rotation fused with an
atomic-accumulating scatter back into the flat coefficient buffer (the
M2M/M2L "return alignment" stage). Uses `KernelAbstractions.@atomic`,
confirmed working on Metal.
"""
function ka_rotate_z_scatter_accumulate!(dest, slab, flat_idx, col_targets, row_m, row_ssign, row_pair, phis; workgroup=KA_AUTO_WORKGROUP)
    length(slab) == 0 && return dest
    backend = KA.get_backend(dest)
    kernel = _cached_kernel(ka_rotate_z_scatter_accumulate_kernel!, backend, workgroup)
    kernel(dest, slab, flat_idx, col_targets, row_m, row_ssign, row_pair, phis; ndrange=length(slab))
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

# C, S = cos/sin(nu * theta') as slabs (ndof x n); `theta` is the plain vector.
function ka_trig_fill!(C, S, nu, theta)
    kernel = _cached_kernel(ka_trig_fill_kernel!, KA.get_backend(C), 256)
    kernel(C, S, nu, theta; ndrange=length(C))
    return C
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
parameter gather; the Lamb-Helmholtz `r` gather stays a separate call because
`col_r` is absent on a non-LH plan.
"""
function ka_gather_values3!(d1, d2, d3, s1, s2, s3, ids; workgroup=KA_AUTO_WORKGROUP)
    n = length(d1)
    n == 0 && return d1
    backend = KA.get_backend(d1)
    kernel = _cached_kernel(ka_gather_values3_kernel!, backend, workgroup)
    kernel(d1, d2, d3, s1, s2, s3, ids; ndrange=n)
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
    mult = kind === :m2m
    ystk = ws.ystk_phi
    Ur = mult ? ystk.mult_Ur : ystk.loc_Ur
    Vs = mult ? ystk.mult_Vs : ystk.loc_Vs
    ndof_phi = size(ws.aphi, 1)
    source_idx = FastMultipole._vector_prefix_view(group.source_idx, n)
    target_idx = FastMultipole._vector_prefix_view(group.target_idx, n)
    group_phis = FastMultipole._vector_prefix_view(group.phis, n)
    group_thetas = FastMultipole._vector_prefix_view(group.thetas, n)
    aphi = FastMultipole._matrix_col_view(ws.aphi, n)
    yphi = FastMultipole._matrix_col_view(ws.yphi, n)
    zphi = FastMultipole._matrix_col_view(ws.zphi, n)
    rphi = FastMultipole._matrix_col_view(ws.rphi, n)
    cphi = FastMultipole._matrix_col_view(ws.cphi, n)
    C = FastMultipole._matrix_col_view(ystk.Cy, n)
    S = FastMultipole._matrix_col_view(ystk.Sy, n)
    G = FastMultipole._matrix_col_view(ystk.G, n)
    G2 = FastMultipole._matrix_col_view(ystk.G2, n)
    ka_trig_fill!(C, S, ystk.nu, group_thetas)
    ka_gather_rotate_z!(aphi, src.phi, ws.phi_flat_idx, source_idx,
        ws.maps_phi.row_m, ws.maps_phi.row_ssign, ws.maps_phi.row_pair, group_phis, one(eltype(aphi)))
    ka_stacked_y_dense!(yphi, aphi, Ur, Vs, C, S, G, G2, ndof_phi)
    mul!(zphi, group.phi_dense, yphi)
    ret_phi = zphi

    has_lh = size(dest.chi, 1) > 0
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
        Cc = FastMultipole._matrix_col_view(ystk_c.Cy, n)
        Sc = FastMultipole._matrix_col_view(ystk_c.Sy, n)
        Gc = FastMultipole._matrix_col_view(ystk_c.G, n)
        G2c = FastMultipole._matrix_col_view(ystk_c.G2, n)
        ka_trig_fill!(Cc, Sc, ystk_c.nu, group_thetas)
        ka_gather_rotate_z!(achi, src.chi, ws.chi_flat_idx, source_idx,
            ws.maps_chi.row_m, ws.maps_chi.row_ssign, ws.maps_chi.row_pair, group_phis, one(eltype(achi)))
        ka_stacked_y_dense!(ychi, achi, Urc, Vsc, Cc, Sc, Gc, G2c, ndof_chi)
        mul!(zchi, group.chi_dense, ychi)
        chi_rows = mult ? ws.maps_chi.row_down : ws.maps_chi.row_up
        ka_gather_rows!(yphi, zchi, ws.maps_phi.row_pair)
        ka_gather_rows!(ychi, zchi, chi_rows)
        cphi .= zphi .+ group.lh_phi_rows .* yphi
        cchi .= zchi .+ group.lh_chi_rows .* ychi
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
    # code). The single end-of-lifecycle sync there covers the host readback.
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
    classes = route_class === nothing ? plan.route_class : route_class
    LH = size(dest.chi, 1) > 0
    TF = eltype(dest.phi)
    for c0 in 1:plan.chunk:nroutes
        cols = c0:min(c0 + plan.chunk - 1, nroutes)
        n = length(cols)
        cls = @view classes[cols]
        phis = @view plan.col_phi[1:n]
        thetas = @view plan.col_theta[1:n]
        invr_col = @view plan.col_invr[1:n]
        ka_gather_values3!(phis, thetas, invr_col, plan.phis, plan.thetas,
            plan.invrs, cls)
        if LH
            rs_col = @view plan.col_r[1:n]
            ka_gather_values!(rs_col, plan.rs, cls)
        end
        src_cols = @view route_sources[cols]
        tgt_cols = @view route_targets[cols]
        aphi = @view plan.aphi[:, 1:n]; yphi = @view plan.yphi[:, 1:n]
        zphi = @view plan.zphi[:, 1:n]; rphi = @view plan.rphi[:, 1:n]
        ops_phi = plan.ops_phi
        ndof_phi = size(plan.aphi, 1)
        Gphi = @view ops_phi.G[:, 1:n]; G2phi = @view ops_phi.G2[:, 1:n]
        Cphi = @view ops_phi.Cy[:, 1:n]; Sphi = @view ops_phi.Sy[:, 1:n]
        sphi = @view ops_phi.scale[:, 1:n]
        ka_prefix_trig_scale!(Cphi, Sphi, sphi, ops_phi.nu, thetas, invr_col,
            plan.rexp_phi)
        ka_gather_rotate_z!(aphi, src.phi, ws.phi_flat_idx, src_cols,
            ws.maps_phi.row_m, ws.maps_phi.row_ssign, ws.maps_phi.row_pair, phis, one(TF))
        ka_stacked_y_dense!(yphi, aphi, ops_phi.yU_mult, ops_phi.yV_mult,
            Cphi, Sphi, Gphi, G2phi, ndof_phi)
        ka_scale_inplace!(yphi, sphi)
        mul!(zphi, ops_phi.zD, yphi)
        ka_scale_inplace!(zphi, sphi)
        ret_phi = zphi

        if LH
            ops_chi = plan.ops_chi
            ndof_chi = size(plan.achi, 1)
            Gchi = @view ops_chi.G[:, 1:n]; G2chi = @view ops_chi.G2[:, 1:n]
            Cchi = @view ops_chi.Cy[:, 1:n]; Schi = @view ops_chi.Sy[:, 1:n]
            schi = @view ops_chi.scale[:, 1:n]
            achi = @view plan.achi[:, 1:n]; ychi = @view plan.ychi[:, 1:n]
            zchi = @view plan.zchi[:, 1:n]; rchi = @view plan.rchi[:, 1:n]
            cphi = @view plan.cphi[:, 1:n]; cchi = @view plan.cchi[:, 1:n]
            ka_prefix_trig_scale!(Cchi, Schi, schi, ops_chi.nu, thetas, invr_col,
                plan.rexp_chi)
            ka_gather_rotate_z!(achi, src.chi, ws.chi_flat_idx, src_cols,
                ws.maps_chi.row_m, ws.maps_chi.row_ssign, ws.maps_chi.row_pair, phis, one(TF))
            ka_stacked_y_dense!(ychi, achi, ops_chi.yU_mult, ops_chi.yV_mult,
                Cchi, Schi, Gchi, G2chi, ndof_chi)
            ka_scale_inplace!(ychi, schi)
            mul!(zchi, ops_chi.zD, ychi)
            ka_scale_inplace!(zchi, schi)
            ka_prefix_lh_mix!(cphi, cchi, zphi, zchi, plan.lh_arow_unit,
                plan.lh_brow_unit, rs_col, ws.maps_phi.row_pair,
                ws.maps_chi.row_up)
            ka_stacked_y_dense!(rchi, cchi, ops_chi.yU_loc, ops_chi.yV_loc,
                Cchi, Schi, Gchi, G2chi, ndof_chi)
            ka_rotate_z_scatter_accumulate!(dest.chi, rchi, ws.chi_flat_idx, tgt_cols,
                ws.maps_chi.row_m, ws.maps_chi.row_ssign, ws.maps_chi.row_pair, phis)
            ret_phi = cphi
        end

        ka_stacked_y_dense!(rphi, ret_phi, ops_phi.yU_loc, ops_phi.yV_loc,
            Cphi, Sphi, Gphi, G2phi, ndof_phi)
        ka_rotate_z_scatter_accumulate!(dest.phi, rphi, ws.phi_flat_idx, tgt_cols,
            ws.maps_phi.row_m, ws.maps_phi.row_ssign, ws.maps_phi.row_pair, phis)
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
