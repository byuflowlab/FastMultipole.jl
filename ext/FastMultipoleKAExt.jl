module FastMultipoleKAExt

using FastMultipole
using KernelAbstractions
using LinearAlgebra
using StaticArrays: SVector
using GPUArraysCore: AnyGPUMatrix, AnyGPUVector
const KA = KernelAbstractions

# Constructing a KA kernel object (`some_kernel!(backend, workgroup)`) redoes
# generic dispatch/partitioning work on every call; caching it per (kernel
# function, backend type, workgroup) avoids repeating that on the hot path.
const _KERNEL_CACHE = Dict{Tuple{Any,DataType,Int},Any}()
function _cached_kernel(f, backend, workgroup::Int)
    wg = resolve_workgroup(backend, workgroup)
    key = (f, typeof(backend), wg)
    return get!(() -> f(backend, wg), _KERNEL_CACHE, key)
end

#------- fixed-geometry row extrema (src: _device_row_extrema) -------#
#
# One kernel of ROW_EXTREMA_LANES workitems, each striding the row; the lane
# partials (2 x lanes scalars) cross to the host and finish there. The kernel's
# specialization does not involve `n`, so it compiles once per (backend, eltype)
# rather than once per distinct length.

const ROW_EXTREMA_LANES = 1024
const _ROW_EXTREMA_SCRATCH = Dict{Any,Any}()

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
    lo, hi, hlo, hhi = get!(_ROW_EXTREMA_SCRATCH, (typeof(backend), T)) do
        (KA.allocate(backend, T, ROW_EXTREMA_LANES), KA.allocate(backend, T, ROW_EXTREMA_LANES),
         Vector{T}(undef, ROW_EXTREMA_LANES), Vector{T}(undef, ROW_EXTREMA_LANES))
    end
    wg = resolve_workgroup(backend, KA_AUTO_WORKGROUP)
    kern = _cached_kernel(ka_row_extrema_kernel!, backend, wg)
    kern(lo, hi, A, Int(row), Int(n), ROW_EXTREMA_LANES; ndrange=ROW_EXTREMA_LANES)
    KA.synchronize(backend)
    copyto!(hlo, lo); copyto!(hhi, hi)
    return minimum(hlo), maximum(hhi)
end

#------- backend workgroup policy -------#
#
# `workgroup=64` was the unexamined default at most launch sites here. 64 suits
# Metal (SIMD width 32, small threadgroups keep occupancy up on a 16-32 core
# GPU) but wastes scheduler slots on an A100, where 256 is the usual figure for
# the memory-bound elementwise kernels that dominate this file. Tunable sites
# now pass `KA_AUTO_WORKGROUP` and the size is resolved per backend, so one
# source tunes for both the local and the HPC target.
#
# NOT every site is tunable. `ka_launch_b2m!`, `ka_launch_l2b!`,
# `ka_launch_nearfield!`, `ka_launch_adaptive_m2t!` and `ka_launch_adaptive_s2l!`
# thread `workgroup` into `Val(workgroup)` and into `ndrange = n * workgroup`:
# there it is the per-cell/per-pair *team size* that the kernel's `@localmem`
# extents are declared against, not an occupancy knob. Those keep their explicit
# sizes and are a separate tuning axis; changing one there changes the parallel
# decomposition, not just the launch geometry.

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
case return the `:KA_WORKGROUP` radix setting if set, else the default for
`backend` (256 on CUDA/ROCm/oneAPI, 64 on Metal and CPU).
"""
function resolve_workgroup(backend, workgroup::Int)
    workgroup == KA_AUTO_WORKGROUP || return workgroup
    override = FastMultipole.KA_WORKGROUP[]
    override == KA_AUTO_WORKGROUP || return override
    return get!(() -> _backend_default_workgroup(backend), _WORKGROUP_CACHE, typeof(backend))
end

# Backend-agnostic M2M building blocks (GPU-native, no host round-trip),
# mirroring FastMultipole's CUDA-only production kernels in
# src/translate_batched_cuda.jl (`_cuda_gather_rotate_z_kernel!`,
# `_cuda_rotate_z_scatter_accumulate_kernel!`, `_resident_mul!`) so they run
# on any KernelAbstractions backend (CUDA, Metal, ...).
#
# Verified against a from-scratch CPU reference derived directly from the
# CUDA source (not yet wired into FastMultipole's actual
# ResidentOperatorGroup/ResidentOperatorWorkspace structs -- see the plan
# file / session notes for what remains):
#   - ka_gather_rotate_z!           : max abs err 5.96e-8 vs CPU (n=8x6)
#   - ka_rotate_z_scatter_accumulate!: max abs err 1.19e-7 vs CPU (w/ KA.@atomic)
#   - full non-LH M2M group chain    : max abs err 2.14e-6 (relerr 4.2e-7),
#     gather_rotate_z -> stacked_y_dense -> rotate_z_scatter_accumulate
# All Float32, on Metal (MetalBackend, M1 Pro).

@kernel function ka_gather_rotate_z_kernel!(dst, @Const(src), @Const(flat_idx), @Const(cols),
                                             @Const(row_m), @Const(row_ssign), @Const(row_pair),
                                             @Const(phis), sgn)
    i = @index(Global)
    nrow = size(dst, 1)
    @inbounds begin
        # Int32 decode: a 64-bit divide per element is emulated on Metal (2026-09-01)
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
    ka_gather_rotate_z!(dst, src, flat_idx, cols, row_m, row_ssign, row_pair, phis, sgn; workgroup=64)

Backend-agnostic port of `_cuda_gather_rotate_z_kernel!`
(src/translate_batched_cuda.jl): fused flat-column gather + z-axis rotation
stage of the M2M/M2L source alignment. `sgn = inverse ? -1 : 1`, matching the
CUDA `_gather_rotate_z!` convention.
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
        # Int32 decode: a 64-bit divide per element is emulated on Metal (2026-09-01)
        i32 = Int32(i) - Int32(1); nrow32 = Int32(nrow)
        row = i32 % nrow32 + Int32(1)
        col = i32 ÷ nrow32 + Int32(1)
        s, c = sincos(row_m[row] * phis[col])
        v = c * slab[row, col] - row_ssign[row] * s * slab[row_pair[row], col]
        KA.@atomic dest[flat_idx[row], col_targets[col]] += v
    end
end

"""
    ka_rotate_z_scatter_accumulate!(dest, slab, flat_idx, col_targets, row_m, row_ssign, row_pair, phis; workgroup=64)

Backend-agnostic port of `_cuda_rotate_z_scatter_accumulate_kernel!`
(src/translate_batched_cuda.jl): inverse z-rotation fused with an
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

"""
    ka_stacked_y_dense!(out_slab, in_slab, Ur, Vs, C, S, G, G2, ndof)

Backend-agnostic port of `_stacked_y_dense!` (src/translate_batched.jl):
the dense y-rotation application used by the M2M `SharedRotationM2M`
strategy. On CUDA this is `_resident_mul!` (== `CUBLAS.gemm!`) around an
elementwise C/S combine; here it is `LinearAlgebra.mul!` (which dispatches
to each backend's own GPU matmul -- Metal's MPS-backed `mul!` for
`MtlArray`, CUBLAS for `CuArray`) plus the same elementwise combine via
ordinary broadcasting. No custom `@kernel` is needed for this stage --
`mul!`/broadcast are already backend-generic.
"""
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
        # Int32 decode: a 64-bit divide per element is emulated on Metal (2026-09-01)
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
    ka_gather_values!(dst, src, ids; workgroup=64)

Backend-agnostic port of `_gather_values!` (src/translate_batched.jl): allocation-free
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
    ka_gather_rows!(dst, src, rows; workgroup=64)

Backend-agnostic port of `_gather_rows!` (src/translate_batched.jl): allocation-free
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
# Ports of the host launcher's already-fused helpers `_prefix_trig_scale!` and
# `_prefix_lh_mix!` (src/translate_batched.jl), plus a three-way column gather.
# The KA concat driver used to spell these out as separate broadcasts and
# `ka_gather_rows!` calls -- 14 launches per chunk for work that is pure
# elementwise reindexing. Profiled on the real wake at np=8192 (11650 routes):
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
    ka_gather_values3!(d1, d2, d3, s1, s2, s3, ids; workgroup=64)

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
        # Int32 decode: a 64-bit divide per element is emulated on Metal (2026-09-01)
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
    ka_prefix_trig_scale!(C, S, scale, nu, theta, invr, rexp; workgroup=64)

Backend-agnostic port of `_prefix_trig_scale!` (src/translate_batched.jl): the
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
    ka_prefix_lh_mix!(cphi, cchi, zphi, zchi, arow, brow, rs, phi_pair, chi_up; workgroup=64)

Backend-agnostic port of `_prefix_lh_mix!` (src/translate_batched.jl): the
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

#------- PRODUCTION PRIMITIVE DISPATCH -------#
#
# The resident stage drivers in src/translate_batched.jl
# (`_resident_stage_group_apply!`, `_launch_resident_m2l_concat!`) are already
# backend-agnostic: apart from views, broadcast, `mul!` and `fill!`, the only
# device work they do goes through the four primitives below. CUDA's "port" of
# the far field is exactly the same trick -- `_launch_cuda_resident_m2m!` and
# `_launch_cuda_resident_l2l!` are one-line passthroughs to those same generic
# drivers, and `src/translate_batched_cuda.jl` specializes only these
# primitives on `CUDA.AnyCuArray`.
#
# So overloading them on GPU arrays puts the KA kernels into the *production*
# call graph rather than alongside it, which is why the standalone
# `ka_resident_stage_group_apply!` / `ka_resident_m2l_concat_apply!` drivers
# (which re-implemented the generic drivers) are no longer needed.
#
# Dispatch handle is `GPUArraysCore.AnyGPU{Matrix,Vector}` -- the wrapper-aware
# analogue of `CUDA.AnyCuArray`. This matters: the drivers hand these
# primitives *views* (`_matrix_col_view`, `@view`), and a `SubArray` of an
# `MtlArray` is not itself an `AbstractGPUArray`. Verified: `MtlMatrix` and
# views of it are `AnyGPUMatrix`, host `Array` is not, and CUDA's
# `AnyCuArray` methods remain strictly more specific for every `CuArray`
# shape including views -- so the CUDA path is unchanged and unambiguous.

function FastMultipole._gather_rotate_z!(dst::AnyGPUMatrix, src::AnyGPUMatrix, flat_idx,
        cols, row_m, row_ssign, row_pair, phis, inverse::Bool)
    # `sgn` must be a float, never the raw `Bool`: `Bool * Number` would
    # silently zero the sin cross-term rather than negate it.
    TF = eltype(dst)
    sgn = inverse ? -one(TF) : one(TF)
    return ka_gather_rotate_z!(dst, src, flat_idx, cols, row_m, row_ssign, row_pair, phis, sgn)
end

function FastMultipole._rotate_z_scatter_accumulate!(dest::AnyGPUMatrix, slab, flat_idx,
        col_targets, row_m, row_ssign, row_pair, phis)
    return ka_rotate_z_scatter_accumulate!(dest, slab, flat_idx, col_targets,
                                           row_m, row_ssign, row_pair, phis)
end

function FastMultipole._gather_rows!(dst::AnyGPUMatrix, src::AnyGPUMatrix, rows)
    return ka_gather_rows!(dst, src, rows)
end

function FastMultipole._gather_values!(dst::AnyGPUVector, src::AnyGPUVector, ids)
    return ka_gather_values!(dst, src, ids)
end

@kernel function ka_fill_invperm_kernel!(invperm, @Const(perm), n)
    sorted_i = @index(Global)
    @inbounds if sorted_i <= n
        invperm[perm[sorted_i]] = sorted_i
    end
end

"""
    ka_fill_invperm!(invperm, perm; workgroup=64)

Backend-agnostic port of `_cuda_fill_invperm_kernel!` (src/translate_batched_cuda.jl):
scatter the inverse of the body sort permutation, `invperm[perm[i]] = i`, so a global
body ordinal maps back to its sorted slot. `invperm` is written over `1:length(perm)`
and must be at least that long; `perm` must be a genuine permutation of `1:n` (each
slot is written exactly once, so a non-permutation silently leaves stale entries --
the same contract CUDA carries).
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

@kernel function ka_fill_single_system_attribution_kernel!(body_system, body_index, n)
    i = @index(Global)
    @inbounds if i <= n
        body_system[i] = 1
        body_index[i] = i
    end
end

"""
    ka_fill_single_system_attribution!(body_system, body_index, n; workgroup=64)

Fill the `DeviceRadixGrid` body-attribution arrays for a single source system:
`body_system[i] = 1`, `body_index[i] = i` over `1:n`. Both are indexed by *global*
(unsorted) body ordinal, so a sorted slot is resolved as `body_system[perm[slot]]`.

This is the attribution half of `_cuda_extract_matrix_positions_kernel!`
(src/translate_batched_cuda.jl), which CUDA fuses into position extraction. The KA
build takes an already-extracted `3 x n` position matrix, so there is nothing to
fuse with and the fill stands alone.

Multi-system attribution (`_cuda_extract_source_positions_kernel!`: one launch per
system, each writing `isys` and its local index into an offset slice) is not ported;
it belongs to the repack path rather than the octree build.
"""
function ka_fill_single_system_attribution!(body_system, body_index, n::Int; workgroup::Int=KA_AUTO_WORKGROUP)
    n == 0 && return body_system, body_index
    (length(body_system) >= n && length(body_index) >= n) || throw(ArgumentError(
        "body attribution arrays (lengths $(length(body_system)), $(length(body_index))) " *
        "are shorter than n=$n"))
    backend = KA.get_backend(body_system)
    kernel = _cached_kernel(ka_fill_single_system_attribution_kernel!, backend, workgroup)
    kernel(body_system, body_index, n; ndrange=n)
    return body_system, body_index
end

@kernel function ka_tree_routes_kernel!(m2m_parent, m2m_child, l2l_parent, l2l_child,
        @Const(parent_index), n_root_nodes, n_nodes)
    edge = @index(Global)
    node = edge + n_root_nodes
    @inbounds if node <= n_nodes
        parent = parent_index[node]
        m2m_parent[edge] = parent
        m2m_child[edge] = node
        l2l_parent[edge] = parent
        l2l_child[edge] = node
    end
end

"""
    ka_tree_routes!(m2m_parent, m2m_child, l2l_parent, l2l_child, parent_index, n_nodes;
                    n_root_nodes=1, workgroup=64)

Backend-agnostic port of `_cuda_tree_routes_kernel!`/`_cuda_radix_tree_routes`
(src/translate_batched_cuda.jl): materialize the parent-child edge list the resident
M2M and L2L passes walk. One edge per non-root node, in node order, so edge `e`
carries child `e + n_root_nodes` and its parent. M2M and L2L get identical arrays
(the passes differ in direction, not in topology) -- CUDA writes both rather than
aliasing, and this does the same so the two can diverge later without a data race.

`n_nodes` is the *logical* node count: `parent_index` is capacity-sized in the KA
context, so its `length` is not the extent. The four outputs are written over
`1:(n_nodes - n_root_nodes)` and must be at least that long. `n_root_nodes` is the
number of leading nodes that are roots (`parent_index == 0`) and therefore contribute
no edge; 1 for the adaptive tree, matching CUDA's call site.
"""
function ka_tree_routes!(m2m_parent, m2m_child, l2l_parent, l2l_child, parent_index,
        n_nodes::Int; n_root_nodes::Int=1, workgroup::Int=KA_AUTO_WORKGROUP)
    n_edges = max(n_nodes - n_root_nodes, 0)
    n_edges == 0 && return m2m_parent, m2m_child, l2l_parent, l2l_child
    n_nodes <= length(parent_index) || throw(ArgumentError(
        "n_nodes=$n_nodes exceeds parent_index (length $(length(parent_index)))"))
    for (name, arr) in (("m2m_parent", m2m_parent), ("m2m_child", m2m_child),
            ("l2l_parent", l2l_parent), ("l2l_child", l2l_child))
        length(arr) >= n_edges || throw(ArgumentError(
            "$name (length $(length(arr))) is shorter than the edge count $n_edges"))
    end
    backend = KA.get_backend(m2m_parent)
    kernel = _cached_kernel(ka_tree_routes_kernel!, backend, workgroup)
    kernel(m2m_parent, m2m_child, l2l_parent, l2l_child, parent_index, n_root_nodes,
        n_nodes; ndrange=n_edges)
    return m2m_parent, m2m_child, l2l_parent, l2l_child
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
                         isys=1, workgroup=64)

Backend-agnostic port of `_cuda_pack_radix_body_kernel!`/`_pack_radix_body_matrix!`
(src/translate_batched_cuda.jl): gather one source system's bodies out of its
global-ordinal `source_buffer` into `body`, the sorted-order `dpb x n` matrix the
resident lifecycle reads. Column `sorted_i` of `body` takes buffer column
`body_index[perm[sorted_i]]`, and only for slots this system owns
(`body_system[perm[sorted_i]] == isys`).

Canonical all-rows packed layout (task 032): every source-buffer row is carried,
including radius row 4, and a system narrower than `body` is zero-padded. Call once
per system, as CUDA does -- with the single-system attribution of
[`ka_fill_single_system_attribution!`](@ref) one `isys=1` call fills every column.

`n` is the *logical* body count: `perm` is capacity-sized in the KA context, so its
`length` is not the extent. Columns beyond `n`, and columns this system does not own,
are left untouched.
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
# (src/translate_batched.jl:2962) -- the real production M2M/L2L group-apply
# reached via `_launch_resident_m2m!` -- substituting the four KA building blocks
# above for its CPU/CUDA-specific primitives. It operates on the same
# `FlatCoefficientBuffer`/`ResidentOperatorGroup`/`ResidentOperatorWorkspace` types,
# so it can be dropped in wherever the CPU function is called, backed by any
# KernelAbstractions array (Metal, CUDA, or plain CPU Array).
#
# This DOES use the `count[]`-prefix views (`_vector_prefix_view`/`_matrix_col_view`),
# exactly as the CPU function does. An earlier version did not, on the reasoning that
# those helpers are FastMultipole-internal and that callers would size `group`/`ws` to
# exactly `group.count[]` columns -- true for the isolated per-group suite this backs
# (test/metal_env/ka_m2m_correctness.jl), and false everywhere else. In the production
# lifecycle `ka_lifecycle_body!` loops over `ws.m2m_groups`/`ws.l2l_groups` from a real
# workspace whose scratch is sized to `max_batch`, the max over levels, so most groups
# are SHORTER than the scratch. Without the views that is a `DimensionMismatch` where
# the shapes disagree and, worse, stale trailing columns scattered into `dest` where
# they happen to broadcast. Both helpers return the array unchanged when the size
# already matches, so the isolated suites are unaffected.
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

# Build a single-group ResidentOperatorGroup + the (M2M-only) slice of a
# ResidentOperatorWorkspace on `exemplar`'s backend, sized to `nbatch` columns, with
# `source_idx = target_idx = 1:nbatch` (an isolated per-node correctness check has no
# tree, so there is no real parent/child relationship to reuse). M2L-only workspace
# fields (`m2l_sources`, `m2l_targets`, `m2l_groups`, `m2l_concat`, the whole-pass
# `y_mult_*`/`y_loc_*`, `nonleaf_idx`) are untouched by `_resident_stage_group_apply!`
# and left as `nothing`.
function _build_m2m_group_and_workspace(exemplar, ::Type{TF}, invariant_cache,
        phis_host::Vector{TF}, thetas_host::Vector{TF}, rs_host::Vector{TF}, ::Val{LH}) where {TF,LH}
    basis_info = invariant_cache.basis_info
    B = typeof(basis_info.basis)
    nbatch = length(phis_host)
    P_phi = basis_info.orders.P_phi
    P_active = basis_info.orders.P_active

    group = FastMultipole._resident_group(exemplar, TF, basis_info, :m2m, 0,
        collect(1:nbatch), collect(1:nbatch), phis_host, thetas_host, rs_host)

    phi_flat_idx = FastMultipole._array_like_vector(exemplar, Int, FastMultipole._degree_major_to_flat_indices(P_phi))
    chi_flat_idx = LH ?
        FastMultipole._array_like_vector(exemplar, Int, FastMultipole._degree_major_to_flat_indices(P_active)) :
        FastMultipole._array_like_vector(exemplar, Int, Int[])
    maps_phi = FastMultipole.DegreeMajorMaps(TF, P_phi, exemplar)
    maps_chi = LH ? FastMultipole.DegreeMajorMaps(TF, P_active, exemplar) : maps_phi

    ndof_phi = FastMultipole.degree_major_dof(P_phi)
    ndof_chi = LH ? FastMultipole.degree_major_dof(P_active) : 0
    mkphi() = similar(exemplar, TF, ndof_phi, nbatch)
    mkchi() = similar(exemplar, TF, ndof_chi, LH ? nbatch : 0)
    aphi = mkphi(); yphi = mkphi(); zphi = mkphi(); rphi = mkphi(); cphi = mkphi()
    achi = mkchi(); ychi = mkchi(); zchi = mkchi(); rchi = mkchi(); cchi = mkchi()

    ystk_phi = FastMultipole.StackedYChannel(exemplar, TF, invariant_cache, P_phi, nbatch)
    ystk_chi = LH ? FastMultipole.StackedYChannel(exemplar, TF, invariant_cache, P_active, nbatch) : nothing

    ws = FastMultipole.ResidentOperatorWorkspace{TF,B,LH}(
        basis_info, phi_flat_idx, chi_flat_idx, maps_phi, maps_chi,
        nothing, nothing, nothing, nothing, nothing, nbatch,
        nothing, nothing,
        aphi, yphi, zphi, rphi, achi, ychi, zchi, rchi, cphi, cchi,
        nothing, nothing, nothing, nothing, FastMultipole.ResidentOperatorGroup[], nothing, nothing,
        ystk_phi, ystk_chi,
    )
    return group, ws
end

"""
    FastMultipole.ka_m2m_operator_batch!(op, targets, sources, phis, thetas, rs, invariant_cache, scratch, lamb_helmholtz)

KernelAbstractions-backed M2M batch operator (stub declared in `src/FastMultipole.jl`).
On a CPU backend this just delegates to the existing `m2m_operator_batch!` (identical
math, no device dispatch needed). On a GPU backend it builds a single-group resident
operator workspace (`_build_m2m_group_and_workspace`) and runs `ka_resident_stage_group_apply!`
-- the KA-ported form of the real production resident M2M path (`_resident_stage_group_apply!`,
reached in production via `_launch_resident_m2m!`) -- against it. Numerically this should
agree with the (CPU, non-resident) `op`/`scratch`-based path up to floating point error:
both compute the same M2M translation, just through different (materialized-rotation vs.
factored-y dense-GEMM) factorizations of the same math.
"""
function FastMultipole.ka_m2m_operator_batch!(op, targets, sources, phis, thetas, rs,
        invariant_cache, scratch, lamb_helmholtz::Val{LH}) where LH
    backend = KA.get_backend(targets.phi)
    if backend isa KA.CPU
        return FastMultipole.m2m_operator_batch!(op, targets, sources, phis, thetas, rs,
            invariant_cache, scratch, lamb_helmholtz)
    end
    TF = eltype(targets.phi)
    group, ws = _build_m2m_group_and_workspace(targets.phi, TF, invariant_cache,
        TF.(collect(phis)), TF.(collect(thetas)), TF.(collect(rs)), lamb_helmholtz)
    return ka_resident_stage_group_apply!(targets, sources, group, ws, :m2m)
end

# --- M2L (horizontal pass) ---
#
# The real production GPU M2L path is `ConcatenatedFixedZM2L`/`_launch_resident_m2l_concat!`
# (src/translate_batched.jl:3828, `ResidentM2LConcatPlan`/`ConcatChannelOps`) -- confirmed
# by checking `RadixFMMCache`'s default/allowed M2L strategies (src/translate_batched_resident.jl),
# not the `FactoredRotationM2L`/`_resident_factored_m2l_group_apply!` path, whose per-degree
# y-rotation blocks are type-asserted as plain CPU `Matrix{TF}` (translate_batched.jl:3077-3080)
# and never dispatch to CUBLAS/Metal GPU matmul -- that path is CPU-only.
#
# `_launch_resident_m2l_concat!`'s primitives are, beyond the M2M-shared ones above:
#   - `_gather_values!` (1D per-chunk column-parameter gather) -- ported here as `ka_gather_values!`
#   - `_stacked_y_dense!` -- already backend-generic (`mul!` + broadcast); reused directly
#     from `ka_stacked_y_dense!` above (same math, real/imag stacked-block form)
#   - `_resident_mul!` (== `mul!`) for the fixed z-translation GEMM -- already backend-generic,
#     called directly
# No new `@kernel` beyond `ka_gather_values!` is needed; `ka_gather_rotate_z!`,
# `ka_gather_rows!`, and `ka_rotate_z_scatter_accumulate!` are reused unchanged from M2M.

# Build a `ResidentM2LConcatPlan` + the (M2L-only) slice of a `ResidentOperatorWorkspace`
# on `exemplar`'s backend for an isolated per-route correctness check: `nbatch` independent
# (source, target) = (i, i) pairs, one geometry class per route (arbitrary per-route
# (r, θ, φ), not the uniform-grid-stencil classes production groups routes into) so any
# random test geometry can be exercised without a tree. `accepted_offsets` here only sizes
# `nclasses` to `nbatch`; the real per-class (r, θ, φ) tables are overwritten right after
# construction with the caller's actual test angles.
function _build_m2l_concat_plan_and_workspace(exemplar, ::Type{TF}, invariant_cache,
        phis_host::Vector{TF}, thetas_host::Vector{TF}, rs_host::Vector{TF}, ::Val{LH}) where {TF,LH}
    basis_info = invariant_cache.basis_info
    B = typeof(basis_info.basis)
    nbatch = length(phis_host)
    P_phi = basis_info.orders.P_phi
    P_active = basis_info.orders.P_active

    offsets = [SVector{3,Int}(i, 0, 0) for i in 1:nbatch]
    plan = FastMultipole.ResidentM2LConcatPlan(TF, basis_info, exemplar,
        FastMultipole.ConcatenatedFixedZM2L(nbatch), invariant_cache, offsets, one(TF), nbatch)
    copyto!(plan.phis, phis_host)
    copyto!(plan.thetas, thetas_host)
    copyto!(plan.rs, rs_host)
    copyto!(plan.invrs, inv.(rs_host))
    copyto!(plan.route_class, Int32.(1:nbatch))

    phi_flat_idx = FastMultipole._array_like_vector(exemplar, Int, FastMultipole._degree_major_to_flat_indices(P_phi))
    chi_flat_idx = LH ?
        FastMultipole._array_like_vector(exemplar, Int, FastMultipole._degree_major_to_flat_indices(P_active)) :
        FastMultipole._array_like_vector(exemplar, Int, Int[])
    maps_phi = FastMultipole.DegreeMajorMaps(TF, P_phi, exemplar)
    maps_chi = LH ? FastMultipole.DegreeMajorMaps(TF, P_active, exemplar) : maps_phi

    empty_sm() = similar(exemplar, TF, 0, 0)
    ws = FastMultipole.ResidentOperatorWorkspace{TF,B,LH}(
        basis_info, phi_flat_idx, chi_flat_idx, maps_phi, maps_chi,
        nothing, nothing, nothing, nothing, nothing, nbatch,
        nothing, nothing,
        empty_sm(), empty_sm(), empty_sm(), empty_sm(),
        empty_sm(), empty_sm(), empty_sm(), empty_sm(),
        empty_sm(), empty_sm(),
        nothing, nothing, nothing, nothing, FastMultipole.ResidentOperatorGroup[], nothing, plan,
        nothing, nothing,
    )
    return plan, ws
end

"""
    ka_resident_m2l_concat_apply!(dest, src, ws, route_sources, route_targets, nroutes)

Backend-agnostic port of `_launch_resident_m2l_concat!` (src/translate_batched.jl:3828),
the real production `ConcatenatedFixedZM2L` resident M2L apply. Operates on `ws.m2l_concat`
(a `ResidentM2LConcatPlan`) plus `route_sources`/`route_targets` (GPU index vectors) directly,
rather than a full `DeviceResidentRadixState`, so it can be dropped into an isolated
per-route correctness check the same way `ka_resident_stage_group_apply!` was for M2M.
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

"""
    FastMultipole.ka_m2l_operator_batch!(op, targets, sources, phis, thetas, rs, invariant_cache, scratch, lamb_helmholtz)

KernelAbstractions-backed M2L batch operator (stub declared in `src/FastMultipole.jl`).
On a CPU backend this delegates to the existing `m2l_operator_batch!` (identical math, no
device dispatch needed). On a GPU backend it builds a single-chunk resident concat plan
(`_build_m2l_concat_plan_and_workspace`) treating each (phis[i], thetas[i], rs[i]) as an
independent (source i -> target i) M2L pair, and runs `ka_resident_m2l_concat_apply!` --
the KA-ported form of the real production `ConcatenatedFixedZM2L` resident M2L path
(`_launch_resident_m2l_concat!`) -- against it.
"""
function FastMultipole.ka_m2l_operator_batch!(op, targets, sources, phis, thetas, rs,
        invariant_cache, scratch, lamb_helmholtz::Val{LH}) where LH
    backend = KA.get_backend(targets.phi)
    if backend isa KA.CPU
        return FastMultipole.m2l_operator_batch!(op, targets, sources, phis, thetas, rs,
            invariant_cache, scratch, lamb_helmholtz)
    end
    TF = eltype(targets.phi)
    plan, ws = _build_m2l_concat_plan_and_workspace(targets.phi, TF, invariant_cache,
        TF.(collect(phis)), TF.(collect(thetas)), TF.(collect(rs)), lamb_helmholtz)
    nbatch = length(phis)
    idx = FastMultipole._array_like_vector(targets.phi, Int, collect(1:nbatch))
    return ka_resident_m2l_concat_apply!(targets, sources, ws, idx, idx, nbatch)
end

# --- L2L (downward pass) ---
#
# The real production GPU L2L path (`_launch_resident_l2l!`, src/translate_batched.jl:3702)
# reaches the exact same `_resident_stage_group_apply!` group-apply function that M2M does,
# just with `kind=:l2l` (the function's `mult = kind === :m2m` branch alone selects the
# multipole vs. local y-rotation tables and LH row direction). `ka_resident_stage_group_apply!`
# above already threads `kind` through unchanged, so no new KA kernel or group-apply port is
# needed for L2L -- only a `:l2l` group/workspace builder and dispatch stub, mirroring M2M's.
function _build_l2l_group_and_workspace(exemplar, ::Type{TF}, invariant_cache,
        phis_host::Vector{TF}, thetas_host::Vector{TF}, rs_host::Vector{TF}, ::Val{LH}) where {TF,LH}
    basis_info = invariant_cache.basis_info
    B = typeof(basis_info.basis)
    nbatch = length(phis_host)
    P_phi = basis_info.orders.P_phi
    P_active = basis_info.orders.P_active

    group = FastMultipole._resident_group(exemplar, TF, basis_info, :l2l, 0,
        collect(1:nbatch), collect(1:nbatch), phis_host, thetas_host, rs_host)

    phi_flat_idx = FastMultipole._array_like_vector(exemplar, Int, FastMultipole._degree_major_to_flat_indices(P_phi))
    chi_flat_idx = LH ?
        FastMultipole._array_like_vector(exemplar, Int, FastMultipole._degree_major_to_flat_indices(P_active)) :
        FastMultipole._array_like_vector(exemplar, Int, Int[])
    maps_phi = FastMultipole.DegreeMajorMaps(TF, P_phi, exemplar)
    maps_chi = LH ? FastMultipole.DegreeMajorMaps(TF, P_active, exemplar) : maps_phi

    ndof_phi = FastMultipole.degree_major_dof(P_phi)
    ndof_chi = LH ? FastMultipole.degree_major_dof(P_active) : 0
    mkphi() = similar(exemplar, TF, ndof_phi, nbatch)
    mkchi() = similar(exemplar, TF, ndof_chi, LH ? nbatch : 0)
    aphi = mkphi(); yphi = mkphi(); zphi = mkphi(); rphi = mkphi(); cphi = mkphi()
    achi = mkchi(); ychi = mkchi(); zchi = mkchi(); rchi = mkchi(); cchi = mkchi()

    ystk_phi = FastMultipole.StackedYChannel(exemplar, TF, invariant_cache, P_phi, nbatch)
    ystk_chi = LH ? FastMultipole.StackedYChannel(exemplar, TF, invariant_cache, P_active, nbatch) : nothing

    ws = FastMultipole.ResidentOperatorWorkspace{TF,B,LH}(
        basis_info, phi_flat_idx, chi_flat_idx, maps_phi, maps_chi,
        nothing, nothing, nothing, nothing, nothing, nbatch,
        nothing, nothing,
        aphi, yphi, zphi, rphi, achi, ychi, zchi, rchi, cphi, cchi,
        nothing, nothing, nothing, nothing, FastMultipole.ResidentOperatorGroup[], nothing, nothing,
        ystk_phi, ystk_chi,
    )
    return group, ws
end

"""
    FastMultipole.ka_l2l_operator_batch!(op, targets, sources, phis, thetas, rs, invariant_cache, scratch, lamb_helmholtz)

KernelAbstractions-backed L2L batch operator (stub declared in `src/FastMultipole.jl`).
On a CPU backend this just delegates to the existing `l2l_operator_batch!` (identical
math, no device dispatch needed). On a GPU backend it builds a single-group resident
operator workspace (`_build_l2l_group_and_workspace`) and runs `ka_resident_stage_group_apply!`
with `kind=:l2l` -- the same KA-ported group-apply M2M already uses, reached in production
via `_launch_resident_l2l!`.
"""
function FastMultipole.ka_l2l_operator_batch!(op, targets, sources, phis, thetas, rs,
        invariant_cache, scratch, lamb_helmholtz::Val{LH}) where LH
    backend = KA.get_backend(targets.phi)
    if backend isa KA.CPU
        return FastMultipole.l2l_operator_batch!(op, targets, sources, phis, thetas, rs,
            invariant_cache, scratch, lamb_helmholtz)
    end
    TF = eltype(targets.phi)
    group, ws = _build_l2l_group_and_workspace(targets.phi, TF, invariant_cache,
        TF.(collect(phis)), TF.(collect(thetas)), TF.(collect(rs)), lamb_helmholtz)
    return ka_resident_stage_group_apply!(targets, sources, group, ws, :l2l)
end

#------- Step 5/6: adaptive octree construction, Phase A (K_max frontier split) -------#
#
# Backend-agnostic port of src/tree_batched_cuda.jl's Phase A
# (`_adt_cuda_seed_root_kernel!`/`_adt_cuda_split_flags_kernel!`/
# `_adt_cuda_split_compact_kernel!`/`_cuda_adaptive_build_leaves!`): the
# level-synchronous K_max frontier split that builds the adaptive octree's leaf
# set from a device array of full-depth-sorted Morton keys, before 2:1 balance
# (tree_batched_cuda.jl Phase B, not yet ported) and finalize (Phase C, not yet
# ported). This is the first phase of the tree/grid construction subsystem that
# `RadixFMMCache(device=true)` currently requires CUDA for (see the ka-migration
# plan/memory note on why `fmm!()` can't reach the KA M2M/M2L/L2L path on Metal
# without this). `_cuda_lower_bound` is CUDA-lifecycle-gated (only defined once
# `load_cuda_radix_lifecycle!()` runs), so `ka_lower_bound` below is a
# self-contained duplicate, not a shared call -- consistent with how the rest of
# this ext never calls into the lazy CUDA-only kernel file.

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

# Child tuple of frontier cell `i`, child bits `c` (0-based): occupied range by
# binary search over the full-depth-sorted body keys. Matches
# `_adt_cuda_child_range` (tree_batched_cuda.jl) exactly.
@inline function ka_child_range(sorted_keys, alev, akey, alo, ahi, i, c, ell_max)
    @inbounds begin
        lc = Int(alev[i]) + 1
        shift = 3 * (ell_max - lc)
        ckey = (akey[i] << 3) | UInt64(c)
        startk = ckey << shift
        endk = startk + (UInt64(1) << shift)
        lo_c = ka_lower_bound(sorted_keys, Int(alo[i]), Int(ahi[i]), startk)
        hi_c = ka_lower_bound(sorted_keys, Int(alo[i]), Int(ahi[i]), endk) - 1
    end
    return lc, ckey, lo_c, hi_c
end

@kernel function ka_seed_root_kernel!(lev, key, lo, hi, n)
    i = @index(Global)
    @inbounds if i == 1
        lev[1] = Int32(0)
        key[1] = UInt64(0)
        lo[1] = Int32(1)
        hi[1] = Int32(n)
    end
end

# flags over the 8 x nact virtual child slots; want_leaf=1 flags children that
# become leaves, want_leaf=0 flags children that stay active (split again).
@kernel function ka_split_flags_kernel!(flags, @Const(sorted_keys), @Const(alev),
        @Const(akey), @Const(alo), @Const(ahi), nact, K_max, ell_max, want_leaf)
    j = @index(Global)
    @inbounds if j <= 8 * nact
        i = (j - 1) >> 3 + 1
        c = (j - 1) & 7
        lc, _, lo_c, hi_c = ka_child_range(sorted_keys, alev, akey, alo, ahi, i, c, ell_max)
        f = Int32(0)
        if lo_c <= hi_c
            isleaf = (hi_c - lo_c + 1 <= K_max) || (lc == ell_max)
            f = ((want_leaf == Int32(1)) == isleaf) ? Int32(1) : Int32(0)
        end
        flags[j] = f
    end
end

@kernel function ka_split_compact_kernel!(dlev, dkey, dlo, dhi, base, @Const(flags),
        @Const(prefix), @Const(sorted_keys), @Const(alev), @Const(akey), @Const(alo),
        @Const(ahi), nact, ell_max)
    j = @index(Global)
    @inbounds if j <= 8 * nact && flags[j] == Int32(1)
        i = (j - 1) >> 3 + 1
        c = (j - 1) & 7
        lc, ckey, lo_c, hi_c = ka_child_range(sorted_keys, alev, akey, alo, ahi, i, c, ell_max)
        idx = base + Int(prefix[j])
        dlev[idx] = Int32(lc)
        dkey[idx] = ckey
        dlo[idx] = Int32(lo_c)
        dhi[idx] = Int32(hi_c)
    end
end

# Inclusive scan of flags[1:m] into prefix[1:m], returning the total. Mirrors
# `_adt_cuda_scan_total!`; `accumulate!` is already backend-generic (KA/GPUArrays),
# so only the final host-scalar readback needs to be written out explicitly.
function _ka_scan_total!(flags, prefix, m::Int)
    m == 0 && return 0
    fv = view(flags, 1:m)
    pv = view(prefix, 1:m)
    accumulate!(+, pv, fv)
    return Int(Array(view(prefix, m:m))[1])
end

#------- Preallocated tree-build context (zero recurring allocation) -------#
#
# Mirrors CUDA-native's `DeviceAdaptiveCUDAContext`/`actx` convention
# (src/tree_batched_cuda.jl:60-160): every working buffer used by
# `ka_build_adaptive_tree!`'s phases is allocated once, here, at capacity, and
# reused across calls instead of being rebuilt via `KA.zeros(...)` on every
# invocation -- the source of the ~1.95GB/trial n=1e6 allocation pressure
# bisected in job 13506038 (see project_fastmultipole_ka_migration memory).
# Scoped to tree construction (Phases A/B/C/D); CUDA's DTR/interaction-list
# buffers (`u_capacity`/`v_capacity`/`wx_capacity`) have no KA counterpart yet.

struct KAAdaptiveTreeContext{B,G,NT<:NamedTuple}
    backend::B
    maxn::Int
    leaf_capacity::Int
    frontier_capacity::Int
    node_capacity::Int
    # The tree's public output, in the form the resident FMM path consumes
    # (`DeviceResidentRadixState.grid`). Allocated here at capacity and mutated
    # in place by the phases -- `bufs` aliases its arrays, so the phase code is
    # unchanged and there is no per-build tuple-to-grid conversion (which would
    # allocate and break the task-023 contract). Mirrors CUDA, where
    # `DeviceAdaptiveCUDAContext` owns the grid and every phase writes `grid.*`.
    grid::G
    bufs::NT
end

"""
    ka_allocate_adaptive_context(backend, TF, maxn; leaf_capacity, frontier_capacity, node_capacity)

Allocate a `KAAdaptiveTreeContext`: every scratch/output buffer
`ka_build_adaptive_tree!` and its phase functions need, sized once at
`maxn`/`leaf_capacity`/`frontier_capacity`/`node_capacity` and reused across
calls. Construct once per (backend, capacity) combination, outside any
trial/timestep loop.

The node- and cell-indexed outputs are allocated as the fields of a capacity-sized
`DeviceRadixGrid` (`actx.grid`), which `bufs` aliases; the geometry fields
(`x_min`/`h0`/`ell`) and the prefix lengths (`n_bodies`/`n_cells`) are placeholders
until `ka_build_adaptive_tree!` sets them from its build arguments, exactly as
`_cuda_refresh_adaptive_tree!` does. `grid.body_system`/`grid.body_index` are filled
for a single source system (see `ka_fill_single_system_attribution!`); multi-system
attribution belongs to the repack path rather than the octree build.
"""
function ka_allocate_adaptive_context(backend, ::Type{TF}, maxn::Int;
        leaf_capacity::Int, frontier_capacity::Int, node_capacity::Int) where TF
    LC, FC, NC = leaf_capacity, frontier_capacity, node_capacity
    grid = FastMultipole.DeviceRadixGrid(
        zero(SVector{3,TF}), one(TF), 0, 0, 0,
        KA.zeros(backend, Int, maxn), KA.zeros(backend, Int, maxn),
        KA.zeros(backend, UInt64, LC), KA.zeros(backend, Int, 2, LC),
        KA.zeros(backend, Int, maxn), KA.zeros(backend, Int, maxn),
        KA.zeros(backend, TF, 3, LC),
        KA.zeros(backend, Int, NC), KA.zeros(backend, UInt64, NC),
        KA.zeros(backend, Int, 3, NC), KA.zeros(backend, TF, 3, NC),
        KA.zeros(backend, Int, NC), KA.zeros(backend, Int, 2, NC),
        KA.zeros(backend, Int, LC),
    )
    bufs = (
        keys=KA.zeros(backend, UInt64, maxn), perm=grid.perm,
        invperm=grid.invperm,
        sorted_keys=KA.zeros(backend, UInt64, maxn),

        llev=KA.zeros(backend, Int32, LC), lkey=KA.zeros(backend, UInt64, LC),
        llo=KA.zeros(backend, Int32, LC), lhi=KA.zeros(backend, Int32, LC),
        a_lev=KA.zeros(backend, Int32, FC), a_key=KA.zeros(backend, UInt64, FC),
        a_lo=KA.zeros(backend, Int32, FC), a_hi=KA.zeros(backend, Int32, FC),
        b_lev=KA.zeros(backend, Int32, FC), b_key=KA.zeros(backend, UInt64, FC),
        b_lo=KA.zeros(backend, Int32, FC), b_hi=KA.zeros(backend, Int32, FC),
        bl_flags=KA.zeros(backend, Int32, FC), bl_prefix=KA.zeros(backend, Int32, FC),

        bal_shifted=KA.zeros(backend, UInt64, LC), bal_order=KA.zeros(backend, Int, LC),
        bal_scratch_starts=KA.zeros(backend, UInt64, LC), bal_marks=KA.zeros(backend, Int32, LC),
        bal_flags=KA.zeros(backend, Int32, LC), bal_prefix=KA.zeros(backend, Int32, LC),
        dlev=KA.zeros(backend, Int32, LC), dkey=KA.zeros(backend, UInt64, LC),
        dlo=KA.zeros(backend, Int32, LC), dhi=KA.zeros(backend, Int32, LC),

        fin_shifted=KA.zeros(backend, UInt64, LC), fin_order=KA.zeros(backend, Int, LC),
        fin_skey=KA.zeros(backend, UInt64, LC), fin_slev=KA.zeros(backend, Int32, LC),
        fin_cand=KA.zeros(backend, UInt64, LC),
        fin_flags=KA.zeros(backend, Int32, NC), fin_prefix=KA.zeros(backend, Int32, NC),
        node_keys=grid.node_keys, node_levels=grid.node_levels,
        node_coords=grid.node_coords, node_centers=grid.node_centers,
        node_lo=KA.zeros(backend, Int32, NC), node_hi=KA.zeros(backend, Int32, NC),
        parent_index=grid.parent_index, child_ranges=grid.child_ranges,
        leaf_index=KA.zeros(backend, Int32, NC), leaf_slot_of=KA.zeros(backend, Int32, NC),
        cell_ranges=grid.cell_ranges, cell_centers=grid.cell_centers,
        cell_keys=grid.cell_keys, leaf_to_node=grid.leaf_to_node,

        node_sigma=KA.zeros(backend, TF, NC),
    )
    return KAAdaptiveTreeContext(backend, maxn, leaf_capacity, frontier_capacity,
        node_capacity, grid, bufs)
end

"""
    ka_adaptive_build_leaves!(actx, sorted_keys, ell_max, K_max, n; workgroup=64)

Backend-agnostic port of `_cuda_adaptive_build_leaves!` (tree_batched_cuda.jl):
builds the K_max leaf set (adaptive octree theory §1.2) from `sorted_keys`, a
KA-backend array of the `n` bodies' full-depth Morton keys in ascending sorted
order. Returns `(nl, leaf_levels, leaf_keys, leaf_lo, leaf_hi)`, each a
backend array truncated logically to `1:nl` (allocated at `actx.leaf_capacity`).
`leaf_lo`/`leaf_hi` are 1-based inclusive ranges into `sorted_keys`. Scratch/
output buffers come from `actx` (see `KAAdaptiveTreeContext`) -- no allocation.
"""
function ka_adaptive_build_leaves!(actx::KAAdaptiveTreeContext, sorted_keys::AbstractVector{UInt64},
        ell_max::Int, K_max::Int, n::Int; workgroup::Int=KA_AUTO_WORKGROUP)
    backend = actx.backend
    b = actx.bufs
    leaf_capacity = actx.leaf_capacity
    frontier_capacity = actx.frontier_capacity
    llev, lkey, llo, lhi = b.llev, b.lkey, b.llo, b.lhi

    seedk = _cached_kernel(ka_seed_root_kernel!, backend, 1)

    if n <= K_max || ell_max == 0
        seedk(llev, lkey, llo, lhi, n; ndrange=1)
        KA.synchronize(backend)
        return 1, llev, lkey, llo, lhi
    end

    a_lev, a_key, a_lo, a_hi = b.a_lev, b.a_key, b.a_lo, b.a_hi
    b_lev, b_key, b_lo, b_hi = b.b_lev, b.b_key, b.b_lo, b.b_hi
    flags, prefix = b.bl_flags, b.bl_prefix

    seedk(a_lev, a_key, a_lo, a_hi, n; ndrange=1)
    KA.synchronize(backend)

    flagsk = _cached_kernel(ka_split_flags_kernel!, backend, workgroup)
    compactk = _cached_kernel(ka_split_compact_kernel!, backend, workgroup)

    nact = 1
    nl = 0
    round = 0
    while nact > 0
        round += 1
        round <= ell_max + 1 || error("adaptive KA K_max split failed to terminate")
        m = 8 * nact
        m <= frontier_capacity || error(
            "adaptive KA split frontier capacity $frontier_capacity exceeded")

        flagsk(flags, sorted_keys, a_lev, a_key, a_lo, a_hi, nact, K_max, ell_max,
            Int32(1); ndrange=m)
        KA.synchronize(backend)
        nleaf = _ka_scan_total!(flags, prefix, m)
        nl + nleaf <= leaf_capacity || error(
            "adaptive KA leaf capacity $leaf_capacity exceeded")
        if nleaf > 0
            compactk(llev, lkey, llo, lhi, nl, flags, prefix, sorted_keys, a_lev, a_key,
                a_lo, a_hi, nact, ell_max; ndrange=m)
            KA.synchronize(backend)
        end
        nl += nleaf

        flagsk(flags, sorted_keys, a_lev, a_key, a_lo, a_hi, nact, K_max, ell_max,
            Int32(0); ndrange=m)
        KA.synchronize(backend)
        nact2 = _ka_scan_total!(flags, prefix, m)
        nact2 <= frontier_capacity || error(
            "adaptive KA active frontier count $nact2 exceeded capacity $frontier_capacity")
        if nact2 > 0
            compactk(b_lev, b_key, b_lo, b_hi, 0, flags, prefix, sorted_keys, a_lev, a_key,
                a_lo, a_hi, nact, ell_max; ndrange=m)
            KA.synchronize(backend)
        end
        a_lev, b_lev = b_lev, a_lev
        a_key, b_key = b_key, a_key
        a_lo, b_lo = b_lo, a_lo
        a_hi, b_hi = b_hi, a_hi
        nact = nact2
    end
    return nl, llev, lkey, llo, lhi
end

#------- Phase B: 2:1 balance (Jacobi rounds over the leaf key set) -------#

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

@kernel function ka_leaf_shifted_kernel!(shifted, @Const(lev), @Const(key), ell_max)
    i = @index(Global)
    @inbounds shifted[i] = key[i] << (3 * (ell_max - Int(lev[i])))
end

# Mark every leaf that violates 2:1 against the current leaf set (mirrors
# `_adt_cuda_balance_mark_kernel!`): leaf B at level lev emits its <=8 touching
# parent-level cells; a leaf A at level <= lev-2 whose interval contains the
# emitted cell start is marked. `marks` is pre-zeroed by the driver (a
# same-kernel clear would race with concurrent mark writes from other threads).
@kernel function ka_balance_mark_kernel!(marks, @Const(lev), @Const(key), nl,
        @Const(sorted_starts), @Const(order), @Const(slev), ell_max)
    i = @index(Global)
    @inbounds if i <= nl
        l = Int(lev[i])
        if l >= 2
            cx, cy, cz = ka_decode_morton_key(key[i], l)
            Gc = 1 << (l - 1)
            qx0 = (cx - 1) >> 1
            qy0 = (cy - 1) >> 1
            qz0 = (cz - 1) >> 1
            for dz in 0:1, dy in 0:1, dx in 0:1
                qx = qx0 + dx
                qy = qy0 + dy
                qz = qz0 + dz
                if 0 <= qx < Gc && 0 <= qy < Gc && 0 <= qz < Gc
                    qstart = ka_morton_key(qx, qy, qz, l - 1) << (3 * (ell_max - (l - 1)))
                    j = ka_upper_bound(sorted_starts, 1, nl, qstart) - 1
                    if j != 0
                        aid = Int(order[j])
                        la = Int(slev[aid])
                        if la <= l - 2
                            astart = sorted_starts[j]
                            alen = UInt64(1) << (3 * (ell_max - la))
                            if qstart < astart + alen
                                marks[aid] = Int32(1)
                            end
                        end
                    end
                end
            end
        end
    end
end

# Per-leaf emission count: unmarked leaves keep one slot; marked leaves emit
# their occupied children (theory §1.4 -- the split leaf is occupied, so >= 1).
@kernel function ka_balance_count_kernel!(cnt, @Const(marks), @Const(lev), @Const(key),
        @Const(lo), @Const(hi), nl, @Const(sorted_keys), ell_max)
    i = @index(Global)
    @inbounds if i <= nl
        if marks[i] == Int32(0)
            cnt[i] = Int32(1)
        else
            m = 0
            for c in 0:7
                _, _, lo_c, hi_c = ka_child_range(sorted_keys, lev, key, lo, hi, i, c, ell_max)
                if lo_c <= hi_c
                    m += 1
                end
            end
            cnt[i] = Int32(m)
        end
    end
end

@kernel function ka_balance_emit_kernel!(dlev, dkey, dlo, dhi, @Const(marks), @Const(prefix),
        @Const(lev), @Const(key), @Const(lo), @Const(hi), nl, @Const(sorted_keys), ell_max)
    i = @index(Global)
    @inbounds if i <= nl
        base = i == 1 ? 0 : Int(prefix[i - 1])
        if marks[i] == Int32(0)
            dlev[base + 1] = lev[i]
            dkey[base + 1] = key[i]
            dlo[base + 1] = lo[i]
            dhi[base + 1] = hi[i]
        else
            w = 0
            for c in 0:7
                lc, ckey, lo_c, hi_c = ka_child_range(sorted_keys, lev, key, lo, hi, i, c, ell_max)
                if lo_c <= hi_c
                    w += 1
                    dlev[base + w] = Int32(lc)
                    dkey[base + w] = ckey
                    dlo[base + w] = Int32(lo_c)
                    dhi[base + w] = Int32(hi_c)
                end
            end
        end
    end
end

"""
    ka_adaptive_balance!(actx, nl, leaf_levels, leaf_keys, leaf_lo, leaf_hi, sorted_keys, ell_max;
                          workgroup=64)

Backend-agnostic port of `_cuda_adaptive_balance!` (tree_batched_cuda.jl): Jacobi
2:1-balance sweep (theory §1.4, Sundar-style) to the fixed point over the K_max
leaf set produced by `ka_adaptive_build_leaves!`. `leaf_levels`/`leaf_keys`/
`leaf_lo`/`leaf_hi` are backend arrays (allocated at `actx.leaf_capacity`, logically
truncated to `1:nl`); `sorted_keys` is the same full-depth-sorted body key
array `ka_adaptive_build_leaves!` was given. Returns
`(nl, n_balance_splits, leaf_levels, leaf_keys, leaf_lo, leaf_hi)` -- the final
leaf arrays may be either input array (ping-pong), not necessarily the ones
passed in. Scratch/output buffers come from `actx` -- no allocation.
"""
function ka_adaptive_balance!(actx::KAAdaptiveTreeContext, nl::Int, leaf_levels, leaf_keys,
        leaf_lo, leaf_hi, sorted_keys::AbstractVector{UInt64}, ell_max::Int; workgroup::Int=KA_AUTO_WORKGROUP)
    backend = actx.backend
    b = actx.bufs
    leaf_capacity = actx.leaf_capacity

    shifted, order, scratch_starts = b.bal_shifted, b.bal_order, b.bal_scratch_starts
    marks, flags, prefix = b.bal_marks, b.bal_flags, b.bal_prefix
    dlev, dkey, dlo, dhi = b.dlev, b.dkey, b.dlo, b.dhi

    shiftedk = _cached_kernel(ka_leaf_shifted_kernel!, backend, workgroup)
    markk = _cached_kernel(ka_balance_mark_kernel!, backend, workgroup)
    countk = _cached_kernel(ka_balance_count_kernel!, backend, workgroup)
    emitk = _cached_kernel(ka_balance_emit_kernel!, backend, workgroup)

    src = (leaf_levels, leaf_keys, leaf_lo, leaf_hi)
    dst = (dlev, dkey, dlo, dhi)
    total = 0
    round = 0
    while true
        round += 1
        round <= 2 * ell_max + 4 || error(
            "adaptive KA 2:1 balance failed to reach a fixed point")

        shiftedk(shifted, src[1], src[2], ell_max; ndrange=nl)
        KA.synchronize(backend)
        ov = view(order, 1:nl)
        sortperm!(ov, view(shifted, 1:nl))
        ka_gather_values!(view(scratch_starts, 1:nl), view(shifted, 1:nl), ov; workgroup=workgroup)

        fill!(view(marks, 1:nl), Int32(0))
        markk(marks, src[1], src[2], nl, scratch_starts, order, src[1], ell_max; ndrange=nl)
        KA.synchronize(backend)

        copyto!(view(flags, 1:nl), view(marks, 1:nl))
        nmark = _ka_scan_total!(flags, prefix, nl)
        nmark == 0 && break
        total += nmark

        countk(flags, marks, src[1], src[2], src[3], src[4], nl, sorted_keys, ell_max; ndrange=nl)
        KA.synchronize(backend)
        nl2 = _ka_scan_total!(flags, prefix, nl)
        nl2 <= leaf_capacity || error(
            "adaptive KA leaf capacity $leaf_capacity exceeded during the balance sweep")

        emitk(dst[1], dst[2], dst[3], dst[4], marks, prefix, src[1], src[2], src[3], src[4],
            nl, sorted_keys, ell_max; ndrange=nl)
        KA.synchronize(backend)

        src, dst = dst, src
        nl = nl2
    end
    return nl, total, src[1], src[2], src[3], src[4]
end

#------- Phase C: level-major node table finalize -------#

@kernel function ka_ancestor_flags_kernel!(flags, @Const(slev), nl, L)
    i = @index(Global)
    @inbounds if i <= nl
        flags[i] = Int(slev[i]) >= L ? Int32(1) : Int32(0)
    end
end

@kernel function ka_ancestor_compact_kernel!(cand, @Const(flags), @Const(prefix),
        @Const(slev), @Const(skey), nl, L)
    i = @index(Global)
    @inbounds if i <= nl && flags[i] == Int32(1)
        cand[Int(prefix[i])] = skey[i] >> (3 * (Int(slev[i]) - L))
    end
end

@kernel function ka_unique_flags_kernel!(flags, @Const(cand), m)
    i = @index(Global)
    @inbounds if i <= m
        flags[i] = (i == 1 || cand[i] != cand[i - 1]) ? Int32(1) : Int32(0)
    end
end

@kernel function ka_unique_compact_kernel!(node_keys, node_levels, base, @Const(flags),
        @Const(prefix), @Const(cand), m, L)
    i = @index(Global)
    @inbounds if i <= m && flags[i] == Int32(1)
        idx = base + Int(prefix[i])
        node_keys[idx] = cand[i]
        node_levels[idx] = L
    end
end

@kernel function ka_node_ranges_kernel!(node_lo, node_hi, @Const(node_keys),
        @Const(node_levels), n_nodes, @Const(sorted_keys), n, ell_max)
    i = @index(Global)
    @inbounds if i <= n_nodes
        shift = 3 * (ell_max - Int(node_levels[i]))
        startk = node_keys[i] << shift
        endk = startk + (UInt64(1) << shift)
        node_lo[i] = Int32(ka_lower_bound(sorted_keys, 1, n, startk))
        node_hi[i] = Int32(ka_lower_bound(sorted_keys, 1, n, endk) - 1)
    end
end

@kernel function ka_node_geometry_kernel!(node_coords, node_centers, @Const(node_keys),
        @Const(node_levels), n_nodes, x_min, h0)
    i = @index(Global)
    @inbounds if i <= n_nodes
        L = Int(node_levels[i])
        cx, cy, cz = ka_decode_morton_key(node_keys[i], L)
        node_coords[1, i] = cx
        node_coords[2, i] = cy
        node_coords[3, i] = cz
        TF = eltype(node_centers)
        width = (2 * h0) / (1 << L)
        node_centers[1, i] = x_min[1] + width * (TF(cx) + TF(0.5))
        node_centers[2, i] = x_min[2] + width * (TF(cy) + TF(0.5))
        node_centers[3, i] = x_min[3] + width * (TF(cz) + TF(0.5))
    end
end

@kernel function ka_parent_kernel!(parent_index, @Const(node_keys), base, count,
        base_prev, count_prev)
    i = @index(Global)
    @inbounds if i <= count
        node = base + i
        pk = node_keys[node] >> 3
        parent_index[node] = ka_lower_bound(node_keys, base_prev + 1,
            base_prev + count_prev, pk)
    end
end

@kernel function ka_children_kernel!(child_ranges, @Const(node_keys), base, count,
        base_next, count_next)
    i = @index(Global)
    @inbounds if i <= count
        node = base + i
        k = node_keys[node] << 3
        firstc = ka_lower_bound(node_keys, base_next + 1, base_next + count_next, k)
        endc = ka_lower_bound(node_keys, base_next + 1, base_next + count_next, k + UInt64(8))
        child_ranges[1, node] = endc > firstc ? firstc : 0
        child_ranges[2, node] = endc - firstc
    end
end

@kernel function ka_leaf_flags_kernel!(flags, @Const(child_ranges), n_nodes)
    i = @index(Global)
    @inbounds if i <= n_nodes
        flags[i] = child_ranges[2, i] == 0 ? Int32(1) : Int32(0)
    end
end

@kernel function ka_leaf_compact_kernel!(leaf_index, leaf_slot_of, @Const(flags),
        @Const(prefix), n_nodes)
    i = @index(Global)
    @inbounds if i <= n_nodes
        leaf_slot_of[i] = Int32(0)
        if flags[i] == Int32(1)
            slot = Int(prefix[i])
            leaf_index[slot] = Int32(i)
            leaf_slot_of[i] = Int32(slot)
        end
    end
end

@kernel function ka_cell_arrays_kernel!(cell_ranges, cell_centers, cell_keys, leaf_to_node,
        @Const(leaf_index), @Const(node_lo), @Const(node_hi), @Const(node_centers),
        @Const(node_keys), n_leaves)
    c = @index(Global)
    @inbounds if c <= n_leaves
        f = Int(leaf_index[c])
        leaf_to_node[c] = f
        cell_ranges[1, c] = Int(node_lo[f])
        cell_ranges[2, c] = Int(node_hi[f]) - Int(node_lo[f]) + 1
        cell_centers[1, c] = node_centers[1, f]
        cell_centers[2, c] = node_centers[2, f]
        cell_centers[3, c] = node_centers[3, f]
        cell_keys[c] = node_keys[f]
    end
end

"""
    ka_adaptive_finalize!(actx, nl, leaf_levels, leaf_keys, leaf_lo, leaf_hi, sorted_keys,
                           ell_max, n, x_min, h0::TF; workgroup=64)

Backend-agnostic port of `_cuda_adaptive_finalize!` (tree_batched_cuda.jl): builds the
level-major node table (ancestor + leaf nodes, Morton-sorted within each level) from
the final (post 2:1-balance) leaf set produced by `ka_adaptive_balance!`, then resolves
node body-ranges, geometry, parent/child links, leaf compaction, and the leaf-indexed
cell presentation arrays. `leaf_levels`/`leaf_keys`/`leaf_lo`/`leaf_hi` are backend
arrays logically truncated to `1:nl`; `sorted_keys` is the same full-depth-sorted body
key array used throughout the adaptive-tree pipeline. `x_min`/`h0` are the tree's
bounding-box origin/half-width (plain scalars, e.g. an `SVector{3,TF}`/`TF`, not device
arrays -- mirrors CUDA's `grid.x_min`/`grid.h0` convention).

Returns a `NamedTuple` `(n_nodes, n_leaves, node_keys, node_levels, node_coords,
node_centers, node_lo, node_hi, parent_index, child_ranges, leaf_index, leaf_slot_of,
cell_ranges, cell_centers, cell_keys, leaf_to_node)`. The node-indexed arrays are
allocated at `actx.node_capacity` and logically truncated to `1:n_nodes`; the
leaf-indexed cell arrays are allocated at `actx.leaf_capacity` and logically truncated
to `1:n_leaves` (== `nl`, asserted). Scratch/output buffers come from `actx` -- no
allocation.
"""
function ka_adaptive_finalize!(actx::KAAdaptiveTreeContext, nl::Int, leaf_levels, leaf_keys,
        leaf_lo, leaf_hi, sorted_keys::AbstractVector{UInt64}, ell_max::Int, n::Int, x_min,
        h0::TF; workgroup::Int=KA_AUTO_WORKGROUP) where TF
    backend = actx.backend
    b = actx.bufs

    shifted, order = b.fin_shifted, b.fin_order
    skey, slev, cand = b.fin_skey, b.fin_slev, b.fin_cand
    flags, prefix = b.fin_flags, b.fin_prefix

    node_keys, node_levels = b.node_keys, b.node_levels
    node_coords, node_centers = b.node_coords, b.node_centers
    node_lo, node_hi = b.node_lo, b.node_hi
    parent_index, child_ranges = b.parent_index, b.child_ranges
    leaf_index, leaf_slot_of = b.leaf_index, b.leaf_slot_of

    shiftedk = _cached_kernel(ka_leaf_shifted_kernel!, backend, workgroup)
    ancflagsk = _cached_kernel(ka_ancestor_flags_kernel!, backend, workgroup)
    anccompactk = _cached_kernel(ka_ancestor_compact_kernel!, backend, workgroup)
    uflagsk = _cached_kernel(ka_unique_flags_kernel!, backend, workgroup)
    ucompactk = _cached_kernel(ka_unique_compact_kernel!, backend, workgroup)
    rangesk = _cached_kernel(ka_node_ranges_kernel!, backend, workgroup)
    geomk = _cached_kernel(ka_node_geometry_kernel!, backend, workgroup)
    parentk = _cached_kernel(ka_parent_kernel!, backend, workgroup)
    childrenk = _cached_kernel(ka_children_kernel!, backend, workgroup)
    leafflagsk = _cached_kernel(ka_leaf_flags_kernel!, backend, workgroup)
    leafcompactk = _cached_kernel(ka_leaf_compact_kernel!, backend, workgroup)
    cellk = _cached_kernel(ka_cell_arrays_kernel!, backend, workgroup)

    shiftedk(shifted, leaf_levels, leaf_keys, ell_max; ndrange=nl)
    KA.synchronize(backend)
    ov = view(order, 1:nl)
    sortperm!(ov, view(shifted, 1:nl))
    ka_gather_values!(view(skey, 1:nl), leaf_keys, ov; workgroup=workgroup)
    ka_gather_values!(view(slev, 1:nl), leaf_levels, ov; workgroup=workgroup)

    node_capacity = actx.node_capacity
    off = zeros(Int, ell_max + 2)
    n_nodes = 0
    for L in 0:ell_max
        off[L + 1] = n_nodes
        ancflagsk(flags, slev, nl, L; ndrange=nl)
        KA.synchronize(backend)
        me = _ka_scan_total!(flags, prefix, nl)
        me == 0 && continue
        anccompactk(cand, flags, prefix, slev, skey, nl, L; ndrange=nl)
        KA.synchronize(backend)
        uflagsk(flags, cand, me; ndrange=me)
        KA.synchronize(backend)
        mL = _ka_scan_total!(flags, prefix, me)
        n_nodes + mL <= node_capacity || error(
            "adaptive KA node capacity $node_capacity exceeded; raise node_capacity")
        ucompactk(node_keys, node_levels, n_nodes, flags, prefix, cand, me, L; ndrange=me)
        KA.synchronize(backend)
        n_nodes += mL
    end
    off[ell_max + 2] = n_nodes

    rangesk(node_lo, node_hi, node_keys, node_levels, n_nodes, sorted_keys, n, ell_max;
        ndrange=n_nodes)
    geomk(node_coords, node_centers, node_keys, node_levels, n_nodes, x_min, h0;
        ndrange=n_nodes)
    KA.synchronize(backend)

    fill!(view(parent_index, 1:n_nodes), 0)
    fill!(view(child_ranges, :, 1:n_nodes), 0)
    for L in 1:ell_max
        base = off[L + 1]
        count = off[L + 2] - off[L + 1]
        count > 0 || continue
        base_prev = off[L]
        count_prev = off[L + 1] - off[L]
        parentk(parent_index, node_keys, base, count, base_prev, count_prev; ndrange=count)
    end
    for L in 0:(ell_max - 1)
        base = off[L + 1]
        count = off[L + 2] - off[L + 1]
        count > 0 || continue
        base_next = off[L + 2]
        count_next = off[L + 3] - off[L + 2]
        childrenk(child_ranges, node_keys, base, count, base_next, count_next; ndrange=count)
    end
    KA.synchronize(backend)

    leafflagsk(flags, child_ranges, n_nodes; ndrange=n_nodes)
    KA.synchronize(backend)
    n_leaves = _ka_scan_total!(flags, prefix, n_nodes)
    n_leaves == nl || throw(AssertionError(
        "adaptive KA finalize: leaf count mismatch ($n_leaves vs $nl)"))
    leafcompactk(leaf_index, leaf_slot_of, flags, prefix, n_nodes; ndrange=n_nodes)
    KA.synchronize(backend)

    cell_ranges = view(b.cell_ranges, :, 1:n_leaves)
    cell_centers = view(b.cell_centers, :, 1:n_leaves)
    cell_keys = view(b.cell_keys, 1:n_leaves)
    leaf_to_node = view(b.leaf_to_node, 1:n_leaves)
    cellk(cell_ranges, cell_centers, cell_keys, leaf_to_node, leaf_index, node_lo, node_hi,
        node_centers, node_keys, n_leaves; ndrange=n_leaves)
    KA.synchronize(backend)

    return (n_nodes=n_nodes, n_leaves=n_leaves, node_keys=node_keys, node_levels=node_levels,
        node_coords=node_coords, node_centers=node_centers, node_lo=node_lo, node_hi=node_hi,
        parent_index=parent_index, child_ranges=child_ranges, leaf_index=leaf_index,
        leaf_slot_of=leaf_slot_of, cell_ranges=cell_ranges, cell_centers=cell_centers,
        cell_keys=cell_keys, leaf_to_node=leaf_to_node, level_offsets=off)
end

#------- Phase D: per-node subtree sigma_max upward pass -------#

@kernel function ka_leaf_sigma_kernel!(node_sigma, @Const(node_lo), @Const(node_hi),
        @Const(child_ranges), @Const(source_bodies), sigma_row, n_nodes)
    i = @index(Global)
    @inbounds if i <= n_nodes && child_ranges[2, i] == 0
        TF = eltype(node_sigma)
        m = zero(TF)
        for r in Int(node_lo[i]):Int(node_hi[i])
            s = source_bodies[sigma_row, r]
            s > m && (m = s)
        end
        node_sigma[i] = m
    end
end

@kernel function ka_sigma_up_kernel!(node_sigma, @Const(child_ranges), base, count)
    i = @index(Global)
    @inbounds if i <= count
        node = base + i
        nc = Int(child_ranges[2, node])
        if nc != 0
            c0 = Int(child_ranges[1, node])
            TF = eltype(node_sigma)
            m = zero(TF)
            for c in c0:(c0 + nc - 1)
                s = node_sigma[c]
                s > m && (m = s)
            end
            node_sigma[node] = m
        end
    end
end

"""
    ka_adaptive_sigma_sweep!(actx, node_lo, node_hi, child_ranges, n_nodes, level_offsets,
                              ell_max, source_bodies, sigma_row; workgroup=64)

Backend-agnostic port of `_cuda_adaptive_sigma_sweep!` (tree_batched_cuda.jl): computes,
for every node of the finalized level-major node table (see `ka_adaptive_finalize!`),
the max of `source_bodies[sigma_row, :]` over its subtree -- leaves take the max over
their own body range (`node_lo`/`node_hi`), interior nodes take the max over their
children's already-computed `node_sigma_max`, processed level-by-level from `ell_max - 1`
up to `0` using `level_offsets` (so children are always finalized before their parent is
visited). `source_bodies` must be indexed in the same sorted-body ordering as
`node_lo`/`node_hi` (i.e. the same `sorted_keys` passed to `ka_adaptive_finalize!`).

Returns `node_sigma_max`, an array sized `actx.node_capacity`, logically truncated to
`1:n_nodes`. Buffer comes from `actx` -- no allocation.
"""
function ka_adaptive_sigma_sweep!(actx::KAAdaptiveTreeContext, node_lo, node_hi, child_ranges,
        n_nodes::Int, level_offsets::Vector{Int}, ell_max::Int,
        source_bodies::AbstractMatrix{TF}, sigma_row::Int; workgroup::Int=KA_AUTO_WORKGROUP) where TF
    backend = actx.backend
    node_sigma = actx.bufs.node_sigma

    leafsigmak = _cached_kernel(ka_leaf_sigma_kernel!, backend, workgroup)
    upk = _cached_kernel(ka_sigma_up_kernel!, backend, workgroup)

    leafsigmak(node_sigma, node_lo, node_hi, child_ranges, source_bodies, sigma_row,
        n_nodes; ndrange=n_nodes)
    KA.synchronize(backend)

    for L in (ell_max - 1):-1:0
        base = level_offsets[L + 1]
        count = level_offsets[L + 2] - level_offsets[L + 1]
        count > 0 || continue
        upk(node_sigma, child_ranges, base, count; ndrange=count)
        KA.synchronize(backend)
    end

    return node_sigma
end

#------- Phase E: DTR interaction-list build (U/V/W/X) -------#
#
# Backend-agnostic port of `_cuda_adaptive_build_lists!` and its six kernels
# (src/tree_batched_cuda.jl:910-1139), the frontier dual-tree-recursion sweep of
# adaptive octree theory §2.7 with sticky sigma demotion (§5.2).
#
# Shape note: the CPU reference (`build_adaptive_interaction_lists!`,
# src/interaction_list_batched.jl:1182) is a DFS over an explicit pair stack,
# while the device version is a level-synchronous BFS over a pair frontier with
# the same flags/scan/compact decomposition as Phases A-C. Both enumerate the
# same pair SET; they emit it in different orders. Any comparison between them
# must canonicalize (sort) first -- element-wise equality is meaningless here.
#
# Precision note: CUDA's `_adt_cuda_classify` hardcodes Float64 for the sigma
# gate. Metal has no Float64 at all, so the KA port makes the gate arithmetic
# generic in a float type `TG` instead (the CUDA path is left exactly as it is).
# `ka_gate_float_type` picks a default that is always valid for the backend the
# tree lives on -- the sigma array's own float type, which is whatever that
# backend supports -- and any float type can be forced via the `gate_type`
# keyword: Float64 on CUDA to mirror the native path bit-for-bit, Float32 on
# Metal, Float16 or a custom AbstractFloat if some future backend wants it.
# Only the gate comparison `delta_min2*g2 < cut^2` is precision-sensitive, and
# only for pairs sitting within a rounding step of the threshold; everything
# else in the classification is exact integer lattice arithmetic, so the choice
# cannot change which pairs are geometrically near.
#
# `g2` is bounded by 3*(2^ell_max)^2, so it is exactly representable in Float32
# for ell_max <= 12 and in Float16 for ell_max <= 4 -- past those the gate
# arithmetic, not the tree, is what limits precision.

"""
    ka_gate_float_type(node_sigma) -> Type{<:AbstractFloat}

Default float type for the DTR sigma gate: the float type the tree's own sigma
array already uses, which is by construction one the backend supports. Override
with the `gate_type` keyword on `ka_adaptive_build_lists!` when a specific
precision is wanted (e.g. Float64 on CUDA, to match `_adt_cuda_classify`).
"""
ka_gate_float_type(node_sigma) = float(eltype(node_sigma))

const _KA_KIND_U = Int32(1)
const _KA_KIND_V = Int32(2)
const _KA_KIND_W = Int32(3)
const _KA_KIND_X = Int32(4)
const _KA_KIND_EXPAND = Int32(5)

@inline function ka_axis_clamp(ca::Int, la::Int, cb::Int, lb::Int)
    k = lb - la
    a0 = ca << k
    a1 = ((ca + 1) << k) - 1
    return cb < a0 ? a0 - cb : (cb > a1 ? cb - a1 : 0)
end

# Mirror of `_adt_cuda_classify`: returns (kind, dem_out, leaf_a, leaf_b).
@inline function ka_dtr_classify(node_levels, node_coords, child_ranges, node_sigma,
        ia::Int, ib::Int, dem::Bool, q::Int, gate::Bool, rho_t::TG,
        delta_min2::TG, ell_max::Int) where TG
    @inbounds begin
        la = Int(node_levels[ia]); lb = Int(node_levels[ib])
        ax = Int(node_coords[1, ia]); ay = Int(node_coords[2, ia]); az = Int(node_coords[3, ia])
        bx = Int(node_coords[1, ib]); by = Int(node_coords[2, ib]); bz = Int(node_coords[3, ib])
        local dx::Int, dy::Int, dz::Int
        if la == lb
            dx = bx - ax; dy = by - ay; dz = bz - az
        elseif la < lb
            dx = ka_axis_clamp(ax, la, bx, lb)
            dy = ka_axis_clamp(ay, la, by, lb)
            dz = ka_axis_clamp(az, la, bz, lb)
        else
            dx = ka_axis_clamp(bx, lb, ax, la)
            dy = ka_axis_clamp(by, lb, ay, la)
            dz = ka_axis_clamp(bz, lb, az, la)
        end
        near = dem || (dx * dx + dy * dy + dz * dz <= q)
        dem_out = dem
        if !near && gate
            # integer-exact squared AABB gap on the finest lattice (host mirror)
            sa = 1 << (ell_max - la)
            sb = 1 << (ell_max - lb)
            g2 = 0
            g = max(ax * sa - (bx * sb + sb), bx * sb - (ax * sa + sa), 0); g2 += g * g
            g = max(ay * sa - (by * sb + sb), by * sb - (ay * sa + sa), 0); g2 += g * g
            g = max(az * sa - (bz * sb + sb), bz * sb - (az * sa + sa), 0); g2 += g * g
            cut = rho_t * TG(node_sigma[ib])
            if delta_min2 * TG(g2) < cut * cut
                near = true
                dem_out = true
            end
        end
        leaf_a = child_ranges[2, ia] == 0
        leaf_b = child_ranges[2, ib] == 0
        local kind::Int32
        if !near
            kind = la == lb ? _KA_KIND_V : (la < lb ? _KA_KIND_W : _KA_KIND_X)
        elseif leaf_a && leaf_b
            kind = _KA_KIND_U
        else
            kind = _KA_KIND_EXPAND
        end
    end
    return kind, dem_out, leaf_a, leaf_b
end

@inline function ka_expand_count(node_levels, child_ranges, ia::Int, ib::Int,
        leaf_a::Bool, leaf_b::Bool)
    @inbounds begin
        la = Int(node_levels[ia]); lb = Int(node_levels[ib])
        na = Int(child_ranges[2, ia]); nb = Int(child_ranges[2, ib])
        if la == lb
            leaf_a && return nb
            leaf_b && return na
            return na * nb
        end
        return la < lb ? nb : na
    end
end

# j-th (1-based) child pair of an EXPAND pair, in the host's deterministic
# (ja-major, jb-minor) order.
@inline function ka_expand_get(node_levels, child_ranges, ia::Int, ib::Int,
        leaf_a::Bool, leaf_b::Bool, j::Int)
    @inbounds begin
        la = Int(node_levels[ia]); lb = Int(node_levels[ib])
        if la == lb
            if leaf_a
                return ia, Int(child_ranges[1, ib]) + j - 1
            elseif leaf_b
                return Int(child_ranges[1, ia]) + j - 1, ib
            else
                nb = Int(child_ranges[2, ib])
                return Int(child_ranges[1, ia]) + (j - 1) ÷ nb,
                    Int(child_ranges[1, ib]) + (j - 1) % nb
            end
        elseif la < lb
            return ia, Int(child_ranges[1, ib]) + j - 1
        else
            return Int(child_ranges[1, ia]) + j - 1, ib
        end
    end
end

@kernel function ka_dtr_seed_kernel!(fa, fb, fdem)
    i = @index(Global)
    @inbounds if i == 1
        fa[1] = Int32(1)
        fb[1] = Int32(1)
        fdem[1] = Int32(0)
    end
end

# want: 1..5 kind flags; 6 = demotion-trigger diagnostic count
@kernel function ka_dtr_flags_kernel!(flags, @Const(fa), @Const(fb), @Const(fdem), np,
        @Const(node_levels), @Const(node_coords), @Const(child_ranges), @Const(node_sigma),
        q, gate, rho_t, delta_min2, ell_max, want)
    p = @index(Global)
    @inbounds if p <= np
        ia = Int(fa[p]); ib = Int(fb[p]); dem = fdem[p] != Int32(0)
        kind, dem_out, _, _ = ka_dtr_classify(node_levels, node_coords, child_ranges,
            node_sigma, ia, ib, dem, q, gate, rho_t, delta_min2, ell_max)
        if want == Int32(6)
            flags[p] = (dem_out && !dem) ? Int32(1) : Int32(0)
        else
            flags[p] = kind == want ? Int32(1) : Int32(0)
        end
    end
end

# Emit U/W/X node-id pairs at base offsets (deterministic scan order).
@kernel function ka_dtr_emit_pairs_kernel!(dst_t, dst_s, base, @Const(flags),
        @Const(prefix), @Const(fa), @Const(fb), np)
    p = @index(Global)
    @inbounds if p <= np && flags[p] == Int32(1)
        idx = base + Int(prefix[p])
        dst_t[idx] = fa[p]
        dst_s[idx] = fb[p]
    end
end

# Emit V pairs with the global class id; validity per the 025 phase-table
# membership (sticky-demotion invariant) via the violation flag.
@kernel function ka_dtr_emit_v_kernel!(vt, vs, vc, base, @Const(flags), @Const(prefix),
        @Const(fa), @Const(fb), np, @Const(node_levels), @Const(node_coords),
        @Const(offset_lut), @Const(level_class_of), reach, noffsets, first_m2l_level,
        violation_flags)
    p = @index(Global)
    @inbounds if p <= np && flags[p] == Int32(1)
        ia = Int(fa[p]); ib = Int(fb[p])
        la = Int(node_levels[ia])
        ox = Int(node_coords[1, ia]) - Int(node_coords[1, ib])
        oy = Int(node_coords[2, ia]) - Int(node_coords[2, ib])
        oz = Int(node_coords[3, ia]) - Int(node_coords[3, ib])
        k = 0
        if abs(ox) <= reach && abs(oy) <= reach && abs(oz) <= reach
            k = Int(offset_lut[ox + reach + 1, oy + reach + 1, oz + reach + 1])
        end
        phase = 1 + (Int(node_coords[1, ib]) & 1) + 2 * (Int(node_coords[2, ib]) & 1) +
            4 * (Int(node_coords[3, ib]) & 1)
        ok = k != 0 && la >= first_m2l_level &&
            level_class_of[phase, k, la + 1] != Int32(0)
        ok || (violation_flags[1] = Int32(1))
        idx = base + Int(prefix[p])
        vt[idx] = fa[p]
        vs[idx] = fb[p]
        vc[idx] = Int32((la - first_m2l_level) * noffsets + k)
    end
end

@kernel function ka_dtr_expand_count_kernel!(flags, @Const(fa), @Const(fb), @Const(fdem),
        np, @Const(node_levels), @Const(node_coords), @Const(child_ranges),
        @Const(node_sigma), q, gate, rho_t, delta_min2, ell_max)
    p = @index(Global)
    @inbounds if p <= np
        ia = Int(fa[p]); ib = Int(fb[p]); dem = fdem[p] != Int32(0)
        kind, _, leaf_a, leaf_b = ka_dtr_classify(node_levels, node_coords, child_ranges,
            node_sigma, ia, ib, dem, q, gate, rho_t, delta_min2, ell_max)
        flags[p] = kind == _KA_KIND_EXPAND ?
            Int32(ka_expand_count(node_levels, child_ranges, ia, ib, leaf_a, leaf_b)) :
            Int32(0)
    end
end

@kernel function ka_dtr_expand_emit_kernel!(ga, gb, gdem, @Const(fa), @Const(fb),
        @Const(fdem), np, @Const(prefix), @Const(node_levels), @Const(node_coords),
        @Const(child_ranges), @Const(node_sigma), q, gate, rho_t, delta_min2, ell_max)
    p = @index(Global)
    @inbounds if p <= np
        ia = Int(fa[p]); ib = Int(fb[p]); dem = fdem[p] != Int32(0)
        kind, dem_out, leaf_a, leaf_b = ka_dtr_classify(node_levels, node_coords,
            child_ranges, node_sigma, ia, ib, dem, q, gate, rho_t, delta_min2, ell_max)
        if kind == _KA_KIND_EXPAND
            # exclusive prefix from the inclusive scan, as CUDA's emit kernel does
            base = p == 1 ? 0 : Int(prefix[p - 1])
            cnt = ka_expand_count(node_levels, child_ranges, ia, ib, leaf_a, leaf_b)
            d = dem_out ? Int32(1) : Int32(0)
            for j in 1:cnt
                ja, jb = ka_expand_get(node_levels, child_ranges, ia, ib, leaf_a, leaf_b, j)
                ga[base + j] = Int32(ja)
                gb[base + j] = Int32(jb)
                gdem[base + j] = d
            end
        end
    end
end

"""
    KAAdaptiveListsContext

Preallocated buffers for the DTR/list phases (E/F/G), mirroring the list portion
of CUDA's `DeviceAdaptiveCUDAContext`.

Constructed against the `KAAdaptiveTreeContext` whose tree it will build lists
for, and **shares that context's frontier-sized scratch** rather than allocating
a second copy: the DTR pair frontier aliases the tree-build frontier ping-pong
(`a_lev`/`a_lo`/`a_hi` and `b_lev`/`b_lo`/`b_hi`, all `Int32` at
`frontier_capacity`), and the DTR scan aliases `bl_flags`/`bl_prefix`. That is
safe because a tree build always completes before its list build, and CUDA's own
`actx` shares buffers the same way (the U CSR reusing the V sort scratch is the
same trick, kept below). At n=1e6 with `frontier_capacity` ~2e7 this is the
difference between ~640MB of duplicate frontier scratch and none.

**Ordering requirement**: do not interleave `ka_build_adaptive_tree!` and
`ka_adaptive_build_lists!` on the same pair of contexts -- finish the tree, then
build its lists. Rebuilding the tree invalidates any list built from it anyway.

`offset_lut`/`level_class_of` are the geometry LUTs from an
`AdaptiveInteractionLists` (moved to the backend by the caller).
"""
struct KAAdaptiveListsContext{B,L3,C3,NT<:NamedTuple}
    backend::B
    frontier_capacity::Int
    u_capacity::Int
    v_capacity::Int
    wx_capacity::Int
    offset_lut::L3
    level_class_of::C3
    lut_reach::Int
    noffsets::Int
    first_m2l_level::Int
    ell_max::Int
    nclasses::Int
    class_starts::Vector{Int}
    level_starts::Vector{Int}
    host_class_counts::Vector{Int32}
    bufs::NT
end

function ka_allocate_lists_context(actx::KAAdaptiveTreeContext, offset_lut,
        level_class_of; u_capacity::Int, v_capacity::Int, wx_capacity::Int,
        lut_reach::Int, noffsets::Int, first_m2l_level::Int, ell_max::Int=0,
        nclasses::Int=max(1, noffsets * (ell_max + 1 - first_m2l_level)),
        leaf_capacity::Int=actx.leaf_capacity, maxn::Int=actx.maxn)
    backend = actx.backend
    FC = actx.frontier_capacity
    UC, VC, WC = u_capacity, v_capacity, wx_capacity
    LC, MN = max(1, leaf_capacity), max(1, maxn)
    t = actx.bufs
    bufs = (
        # DTR pair frontier + scan: aliases of the tree-build frontier scratch
        # (see the docstring). Same element type and length; the tree build is
        # finished by the time any of these are read.
        fa=t.a_lev, fb=t.a_lo, fdem=t.a_hi,
        fa2=t.b_lev, fb2=t.b_lo, fdem2=t.b_hi,
        flags=t.bl_flags, prefix=t.bl_prefix,
        u_targets=KA.zeros(backend, Int32, UC), u_sources=KA.zeros(backend, Int32, UC),
        vstage_targets=KA.zeros(backend, Int32, VC),
        vstage_sources=KA.zeros(backend, Int32, VC),
        vstage_class=KA.zeros(backend, Int32, VC),
        w_targets=KA.zeros(backend, Int32, WC), w_sources=KA.zeros(backend, Int32, WC),
        x_targets=KA.zeros(backend, Int32, WC), x_sources=KA.zeros(backend, Int32, WC),
        # slot 1: V phase-table violation; slot 2: U endpoint not a leaf slot
        violation_flags=KA.zeros(backend, Int32, 2),

        # Phase F: V class partition into the CSR route stream
        vsort_keys=KA.zeros(backend, UInt64, VC), vsort_ix=KA.zeros(backend, Int, VC),
        route_targets=KA.zeros(backend, Int, VC), route_sources=KA.zeros(backend, Int, VC),
        route_class=KA.zeros(backend, Int32, VC),
        route_class_offset=KA.zeros(backend, Int32, VC),
        class_counts_dev=KA.zeros(backend, Int32, nclasses),

        # Phase G: U endpoints -> leaf slots, then target-major U CSR
        direct_targets=KA.zeros(backend, Int, UC),
        direct_sources=KA.zeros(backend, Int, UC),
        u_csr_offsets=KA.zeros(backend, Int32, LC + 1),
        u_csr_sources=KA.zeros(backend, Int32, UC),
        u_csr_body_leaf=KA.zeros(backend, Int32, MN),
    )
    # Guard the aliasing assumption rather than trusting it silently.
    all(x -> length(x) >= FC, (bufs.fa, bufs.fb, bufs.fdem, bufs.fa2, bufs.fb2,
        bufs.fdem2, bufs.flags, bufs.prefix)) || throw(ArgumentError(
        "tree context's frontier scratch is smaller than frontier_capacity=$FC"))
    return KAAdaptiveListsContext(backend, FC, UC, VC, WC, offset_lut, level_class_of,
        lut_reach, noffsets, first_m2l_level, ell_max, nclasses,
        zeros(Int, nclasses + 1), zeros(Int, ell_max + 2), zeros(Int32, nclasses), bufs)
end

"""
    ka_adaptive_build_lists!(lctx, node_levels, node_coords, child_ranges, node_sigma;
                             ell_max, near_radius2, gate, rho_t, delta_min2, workgroup=64)

Backend-agnostic port of `_cuda_adaptive_build_lists!`: the frontier DTR sweep
(theory §2.7) producing the U/V/W/X interaction lists from a finalized adaptive
octree. Returns `(n_u, n_v, n_w, n_x, n_dem)`; the lists themselves live in
`lctx.bufs`, valid over `1:n_*`.

`rho_t`/`delta_min2` may be any `Real`; they are converted to `gate_type`,
which defaults to `ka_gate_float_type(node_sigma)` and controls the sigma-gate
precision (see the precision note above). Pass `gate_type=Float64` on CUDA to
mirror `_adt_cuda_classify` bit-for-bit.
"""
function ka_adaptive_build_lists!(lctx::KAAdaptiveListsContext, node_levels, node_coords,
        child_ranges, node_sigma; ell_max::Int, near_radius2::Int, gate::Bool,
        rho_t::Real, delta_min2::Real,
        gate_type::Type{TG}=ka_gate_float_type(node_sigma),
        workgroup::Int=KA_AUTO_WORKGROUP) where {TG<:AbstractFloat}
    backend = lctx.backend
    b = lctx.bufs
    # Convert once, on the host: the kernels take these as scalar arguments, so
    # every pair sees the identical value and the gate stays in exactly TG.
    rho_t_g = TG(rho_t)
    delta_min2_g = TG(delta_min2)
    fill!(b.violation_flags, Int32(0))

    seedk = _cached_kernel(ka_dtr_seed_kernel!, backend, 1)
    flagk = _cached_kernel(ka_dtr_flags_kernel!, backend, workgroup)
    emitk = _cached_kernel(ka_dtr_emit_pairs_kernel!, backend, workgroup)
    emitvk = _cached_kernel(ka_dtr_emit_v_kernel!, backend, workgroup)
    ecountk = _cached_kernel(ka_dtr_expand_count_kernel!, backend, workgroup)
    eemitk = _cached_kernel(ka_dtr_expand_emit_kernel!, backend, workgroup)

    a = (b.fa, b.fb, b.fdem)
    bb = (b.fa2, b.fb2, b.fdem2)
    seedk(a[1], a[2], a[3]; ndrange=1)

    node_args = (node_levels, node_coords, child_ranges, node_sigma)
    gate_args = (near_radius2, gate, rho_t_g, delta_min2_g, ell_max)

    np = 1
    n_u = 0; n_v = 0; n_w = 0; n_x = 0; n_dem = 0
    rounds = 0
    while np > 0
        rounds += 1
        rounds <= 2 * ell_max + 3 || throw(AssertionError(
            "adaptive KA DTR failed to terminate"))

        # U / W / X share the plain pair-emit path; only the `want` code, the
        # running count and the destination buffers differ. Written out rather
        # than looped, mirroring `_cuda_adaptive_build_lists!`.
        emit_kind! = function (want, base, cap, dst_t, dst_s, label)
            flagk(b.flags, a[1], a[2], a[3], np, node_args..., gate_args..., want;
                ndrange=np)
            m = _ka_scan_total!(b.flags, b.prefix, np)
            base + m <= cap || throw(AssertionError(
                "adaptive KA $label list capacity $cap exceeded"))
            m > 0 && emitk(dst_t, dst_s, base, b.flags, b.prefix, a[1], a[2], np; ndrange=np)
            return m
        end
        n_u += emit_kind!(_KA_KIND_U, n_u, lctx.u_capacity, b.u_targets, b.u_sources, "U")
        n_w += emit_kind!(_KA_KIND_W, n_w, lctx.wx_capacity, b.w_targets, b.w_sources, "W")
        n_x += emit_kind!(_KA_KIND_X, n_x, lctx.wx_capacity, b.x_targets, b.x_sources, "X")

        # V carries the geometry class id, so it needs its own emit kernel.
        flagk(b.flags, a[1], a[2], a[3], np, node_args..., gate_args..., _KA_KIND_V;
            ndrange=np)
        m = _ka_scan_total!(b.flags, b.prefix, np)
        n_v + m <= lctx.v_capacity || throw(AssertionError(
            "adaptive KA V route capacity $(lctx.v_capacity) exceeded"))
        if m > 0
            emitvk(b.vstage_targets, b.vstage_sources, b.vstage_class, n_v, b.flags,
                b.prefix, a[1], a[2], np, node_levels, node_coords, lctx.offset_lut,
                lctx.level_class_of, lctx.lut_reach, lctx.noffsets, lctx.first_m2l_level,
                b.violation_flags; ndrange=np)
        end
        n_v += m

        # demotion diagnostic
        if gate
            flagk(b.flags, a[1], a[2], a[3], np, node_args..., gate_args..., Int32(6);
                ndrange=np)
            n_dem += _ka_scan_total!(b.flags, b.prefix, np)
        end

        # expand: count child pairs per frontier entry, scan, emit next frontier
        ecountk(b.flags, a[1], a[2], a[3], np, node_args..., gate_args...; ndrange=np)
        np2 = _ka_scan_total!(b.flags, b.prefix, np)
        np2 <= lctx.frontier_capacity || throw(AssertionError(
            "adaptive KA DTR frontier capacity $(lctx.frontier_capacity) exceeded"))
        np2 > 0 && eemitk(bb[1], bb[2], bb[3], a[1], a[2], a[3], np, b.prefix,
            node_args..., gate_args...; ndrange=np)
        a, bb = bb, a
        np = np2
    end

    # One sync at the end, not one per launch. KA kernels on a backend are
    # ordered on that backend's own queue, so consecutive launches need no
    # explicit barrier, and each `_ka_scan_total!` already forces a sync via its
    # host readback of the scan total. Per-kernel `KA.synchronize` is what cost
    # the M2M port 53.6ms vs 20ms before it was removed there (commit ff14b7e).
    KA.synchronize(backend)
    Int(Array(view(b.violation_flags, 1:1))[1]) == 0 || throw(AssertionError(
        "adaptive KA V pair lies outside the task-025 phase-table class set — " *
        "the sticky demotion invariant (theory §2.4/§5.2) is violated"))

    return n_u, n_v, n_w, n_x, n_dem
end

#------- Phase F: V-list class partition into the CSR route stream -------#
#
# Port of `_cuda_adaptive_partition_v!` and its three kernels
# (src/tree_batched_cuda.jl:1143-1218). The V stage stream carries a class id
# per pair; the routes have to be grouped by class so each M2L class becomes one
# contiguous slab. Sorting on `(class << 32) | emission_index` makes the
# partition stable by construction -- ties inside a class keep DTR emission
# order -- so no separate stable-sort primitive is needed.

@kernel function ka_vsort_keys_kernel!(keys, @Const(vclass), n_v)
    i = @index(Global)
    @inbounds if i <= n_v
        keys[i] = (UInt64(vclass[i]) << 32) | UInt64(i)
    end
end

@kernel function ka_csr_gather_kernel!(route_targets, route_sources, route_class,
        route_class_offset, @Const(vsort_ix), @Const(vt), @Const(vs), @Const(vc),
        n_v, noffsets)
    p = @index(Global)
    @inbounds if p <= n_v
        i = Int(vsort_ix[p])
        route_targets[p] = Int(vt[i])
        route_sources[p] = Int(vs[i])
        c = Int(vc[i])
        route_class[p] = Int32(c)
        # per-offset id for the dense family (class_base = 0 convention)
        route_class_offset[p] = Int32(c - ((c - 1) ÷ noffsets) * noffsets)
    end
end

@kernel function ka_class_histogram_kernel!(counts, @Const(vclass), n_v)
    i = @index(Global)
    @inbounds if i <= n_v
        KA.@atomic counts[Int(vclass[i])] += Int32(1)
    end
end

"""
    ka_adaptive_partition_v!(lctx, n_v; workgroup=64)

Backend-agnostic port of `_cuda_adaptive_partition_v!`: deterministically
partitions the `n_v` staged V pairs by class into the CSR route stream
(`lctx.bufs.route_*`, valid over `1:n_v`) and fills `lctx.class_starts` /
`lctx.level_starts` on the host. Returns `n_v` (the route count).
"""
function ka_adaptive_partition_v!(lctx::KAAdaptiveListsContext, n_v::Int;
        workgroup::Int=KA_AUTO_WORKGROUP)
    backend = lctx.backend
    b = lctx.bufs
    cc = lctx.host_class_counts

    if n_v == 0
        fill!(cc, Int32(0))
    else
        keysk = _cached_kernel(ka_vsort_keys_kernel!, backend, workgroup)
        gatherk = _cached_kernel(ka_csr_gather_kernel!, backend, workgroup)
        histk = _cached_kernel(ka_class_histogram_kernel!, backend, workgroup)

        keysk(b.vsort_keys, b.vstage_class, n_v; ndrange=n_v)
        # `sortperm!` dispatches to the backend's own sort (confirmed working on
        # Metal in Phase B), matching CUDA's `_cuda_sortperm_into!` convention.
        sortperm!(view(b.vsort_ix, 1:n_v), view(b.vsort_keys, 1:n_v))
        gatherk(b.route_targets, b.route_sources, b.route_class, b.route_class_offset,
            b.vsort_ix, b.vstage_targets, b.vstage_sources, b.vstage_class, n_v,
            lctx.noffsets; ndrange=n_v)
        fill!(b.class_counts_dev, Int32(0))
        histk(b.class_counts_dev, b.vstage_class, n_v; ndrange=n_v)
        copyto!(cc, Array(b.class_counts_dev))   # forces the sync
    end

    cs = lctx.class_starts
    cs[1] = 1
    @inbounds for c in 1:lctx.nclasses
        cs[c + 1] = cs[c] + Int(cc[c])
    end
    ls = lctx.level_starts
    first = lctx.first_m2l_level
    @inbounds for L in 0:lctx.ell_max
        ls[L + 1] = L < first ? 1 : cs[(L - first) * lctx.noffsets + 1]
    end
    ls[lctx.ell_max + 2] = cs[end]
    return n_v
end


#------- Phase G: U endpoints -> leaf slots, then the target-major U CSR -------#
#
# Port of `_adt_cuda_u_slots_kernel!` and `_cuda_adaptive_build_u_csr!` with its
# three kernels (src/tree_batched_cuda.jl:1222-1324). The DTR emits U pairs in
# frontier-major order, which is NOT target-major, so building the fused
# nearfield's target-owned CSR needs the same stable (target-slot, index) key
# sort the V partition uses.
#
# The offsets kernel writes the half-open slot range (tprev, t] for each CSR
# position, which fills in leaves that own no U pairs at all; slots past the
# last occupied target keep the `n_u + 1` prefill. Targets are sorted, so those
# ranges are disjoint and the concurrent writes never overlap.

@kernel function ka_u_slots_kernel!(direct_targets, direct_sources, @Const(u_targets),
        @Const(u_sources), @Const(leaf_slot_of), n_u, violation_flags)
    k = @index(Global)
    @inbounds if k <= n_u
        ts = leaf_slot_of[Int(u_targets[k])]
        ss = leaf_slot_of[Int(u_sources[k])]
        (ts == Int32(0) || ss == Int32(0)) && (violation_flags[2] = Int32(1))
        direct_targets[k] = Int(ts)
        direct_sources[k] = Int(ss)
    end
end

@kernel function ka_body_leaf_kernel!(body_leaf, @Const(cell_ranges), n_leaves)
    l = @index(Global)
    @inbounds if l <= n_leaves
        first = cell_ranges[1, l]
        last = first + cell_ranges[2, l] - 1
        i = first
        while i <= last
            body_leaf[i] = Int32(l)
            i += 1
        end
    end
end

@kernel function ka_usort_keys_kernel!(keys, @Const(direct_targets), n_u)
    i = @index(Global)
    @inbounds if i <= n_u
        keys[i] = (UInt64(direct_targets[i]) << 32) | UInt64(i)
    end
end

@kernel function ka_ucsr_gather_kernel!(u_csr_sources, @Const(usort_ix),
        @Const(direct_sources), n_u)
    p = @index(Global)
    @inbounds if p <= n_u
        u_csr_sources[p] = Int32(direct_sources[Int(usort_ix[p])])
    end
end

@kernel function ka_ucsr_offsets_kernel!(u_csr_offsets, @Const(usort_ix),
        @Const(direct_targets), n_u)
    p = @index(Global)
    @inbounds if p <= n_u
        t = Int(direct_targets[Int(usort_ix[p])])
        tprev = p == 1 ? 0 : Int(direct_targets[Int(usort_ix[p - 1])])
        s = tprev + 1
        while s <= t
            u_csr_offsets[s] = Int32(p)
            s += 1
        end
    end
end

"""
    ka_adaptive_u_slots!(lctx, leaf_slot_of, n_u; workgroup=64)

Map the `n_u` U pairs' node ids to leaf-cell slots, into
`lctx.bufs.direct_targets`/`direct_sources`. Throws if either endpoint of any
pair is not a leaf (which would mean the DTR produced a non-leaf U pair).
"""
function ka_adaptive_u_slots!(lctx::KAAdaptiveListsContext, leaf_slot_of, n_u::Int;
        workgroup::Int=KA_AUTO_WORKGROUP)
    n_u == 0 && return nothing
    backend = lctx.backend
    b = lctx.bufs
    k = _cached_kernel(ka_u_slots_kernel!, backend, workgroup)
    k(b.direct_targets, b.direct_sources, b.u_targets, b.u_sources, leaf_slot_of,
        n_u, b.violation_flags; ndrange=n_u)
    # the readback below is itself the sync point
    Int(Array(view(b.violation_flags, 2:2))[1]) == 0 || throw(AssertionError(
        "adaptive KA U pair endpoint is not a leaf cell slot"))
    return nothing
end

"""
    ka_adaptive_build_u_csr!(lctx, cell_ranges, n_leaves, n_u; workgroup=64)

Backend-agnostic port of `_cuda_adaptive_build_u_csr!`: builds the target-major
CSR (`u_csr_offsets` over `1:n_leaves+1`, `u_csr_sources` over `1:n_u`) from the
slot-mapped U list produced by `ka_adaptive_u_slots!`, plus the body -> leaf-slot
map for the dense body-packed nearfield shape. Reuses the Phase F sort scratch,
as CUDA does.
"""
function ka_adaptive_build_u_csr!(lctx::KAAdaptiveListsContext, cell_ranges,
        n_leaves::Int, n_u::Int; workgroup::Int=KA_AUTO_WORKGROUP)
    backend = lctx.backend
    b = lctx.bufs
    n_leaves + 1 <= length(b.u_csr_offsets) || throw(AssertionError(
        "KA U-CSR offsets capacity $(length(b.u_csr_offsets)) exceeded " *
        "(n_leaves=$n_leaves)"))
    fill!(view(b.u_csr_offsets, 1:(n_leaves + 1)), Int32(n_u + 1))

    if n_u > 0
        n_u <= length(b.vsort_keys) || throw(AssertionError(
            "KA U-CSR reuses the V sort scratch; n_u=$n_u exceeds v_capacity " *
            "$(length(b.vsort_keys))"))
        keysk = _cached_kernel(ka_usort_keys_kernel!, backend, workgroup)
        gatherk = _cached_kernel(ka_ucsr_gather_kernel!, backend, workgroup)
        offsk = _cached_kernel(ka_ucsr_offsets_kernel!, backend, workgroup)

        keysk(b.vsort_keys, b.direct_targets, n_u; ndrange=n_u)
        sortperm!(view(b.vsort_ix, 1:n_u), view(b.vsort_keys, 1:n_u))
        gatherk(b.u_csr_sources, b.vsort_ix, b.direct_sources, n_u; ndrange=n_u)
        offsk(b.u_csr_offsets, b.vsort_ix, b.direct_targets, n_u; ndrange=n_u)
    end

    if n_leaves > 0
        bodyk = _cached_kernel(ka_body_leaf_kernel!, backend, workgroup)
        bodyk(b.u_csr_body_leaf, cell_ranges, n_leaves; ndrange=n_leaves)
    end
    KA.synchronize(backend)     # single sync: results are caller-visible after this
    return nothing
end

#------- Harness front end: position -> full-depth key, and a full-build driver -------#
#
# Not a CUDA-parity phase (no `_cuda_*` counterpart is ported one-to-one here) --
# this stitches Phases A-D together into a single from-scratch tree build, the way
# `_cuda_refresh_adaptive_tree!` (tree_batched_cuda.jl:1387) stitches the hand-written
# CUDA stages, so the 4-way tree-build benchmark can drive local-Metal and HPC-KA off
# one shared, backend-agnostic entry point.

@kernel function ka_radix_keys_kernel!(keys, @Const(positions), x_min, h0, ell, n)
    i = @index(Global)
    @inbounds if i <= n
        G = 1 << ell
        delta = (2 * h0) / G
        px = positions[1, i]
        py = positions[2, i]
        pz = positions[3, i]
        ix = clamp(floor(Int, (px - x_min[1]) / delta), 0, G - 1)
        iy = clamp(floor(Int, (py - x_min[2]) / delta), 0, G - 1)
        iz = clamp(floor(Int, (pz - x_min[3]) / delta), 0, G - 1)
        keys[i] = ka_morton_key(ix, iy, iz, ell)
    end
end

"""
    ka_radix_keys!(keys, positions, x_min, h0, ell; workgroup=64)

Backend-agnostic port of `_cuda_radix_keys_checked_kernel!` (translate_batched_cuda.jl),
minus its out-of-bounds flag (bodies are assumed to already lie in the fixed root cube
`[x_min, x_min + 2*h0]^3` -- true by construction for a from-scratch benchmark build;
callers needing the OOB guard for a live refresh loop should add it at the call site).
Writes each body's full-depth (`ell`-level) Morton key from its position into `keys`.
`positions` is a `3 x n` backend matrix; `x_min` is a plain 3-tuple/SVector, not a
device array (mirrors CUDA's `grid.x_min` convention).
"""
function ka_radix_keys!(keys::AbstractVector{UInt64}, positions::AbstractMatrix,
        x_min, h0, ell::Int; workgroup::Int=KA_AUTO_WORKGROUP)
    n = length(keys)
    backend = KA.get_backend(keys)
    keysk = _cached_kernel(ka_radix_keys_kernel!, backend, workgroup)
    keysk(keys, positions, x_min, h0, ell, n; ndrange=n)
    KA.synchronize(backend)
    return keys
end

"""
    ka_build_adaptive_tree!(actx, positions, ell_max, K_max, balance, x_min, h0::TF;
                             sigma_row=0, source_bodies=nothing, workgroup=64)

From-scratch adaptive-octree build from raw body positions: full-depth Morton keys
(`ka_radix_keys!`) + sort (`sortperm!`/`ka_gather_values!`, same as Phases B/C) feeding
`ka_adaptive_build_leaves!` (Phase A) -> `ka_adaptive_balance!` (Phase B, if `balance`)
-> `ka_adaptive_finalize!` (Phase C) -> `ka_adaptive_sigma_sweep!` (Phase D, if
`sigma_row > 0` and `source_bodies` given). `positions` is a `3 x n` backend matrix
(`n <= actx.maxn`); `x_min`/`h0` are the fixed root cube (plain scalars, not device
arrays). Mirrors the stage order of `_cuda_refresh_adaptive_tree!` for a first
(non-incremental) build. All scratch/output buffers come from `actx` (see
`KAAdaptiveTreeContext`/`ka_allocate_adaptive_context`) -- zero recurring allocation,
matching CUDA-native's `actx` convention.

Returns Phase C's `NamedTuple` merged with `grid`, `perm`, `invperm`, `sorted_keys`,
`n_balance_splits`, and `node_sigma_max` (`nothing` if the sigma sweep was not armed).
"""
function ka_build_adaptive_tree!(actx::KAAdaptiveTreeContext, positions::AbstractMatrix,
        ell_max::Int, K_max::Int, balance::Bool, x_min, h0::TF; sigma_row::Int=0,
        source_bodies=nothing, workgroup::Int=KA_AUTO_WORKGROUP,
        stage_ns::Union{Nothing,Vector{UInt64}}=nothing) where TF
    backend = actx.backend
    n = size(positions, 2)
    n <= actx.maxn || error("adaptive KA context maxn=$(actx.maxn) exceeded by n=$n")
    b = actx.bufs

    # Optional per-stage timing (mirrors CUDA-native's actx.profile_stages/
    # stage_ns convention, tree_batched_cuda.jl:1387-1431) to root-cause the
    # n>=1e5 timing anomaly (project_fastmultipole_ka_migration memory).
    t0 = stage_ns !== nothing ? (KernelAbstractions.synchronize(backend); time_ns()) : UInt64(0)

    keys = view(b.keys, 1:n)
    ka_radix_keys!(keys, positions, x_min, h0, ell_max; workgroup=workgroup)

    perm = view(b.perm, 1:n)
    sortperm!(perm, keys)
    sorted_keys = view(b.sorted_keys, 1:n)
    ka_gather_values!(sorted_keys, keys, perm; workgroup=workgroup)
    invperm = view(b.invperm, 1:n)
    ka_fill_invperm!(invperm, perm; workgroup=workgroup)

    if stage_ns !== nothing
        KernelAbstractions.synchronize(backend)
        stage_ns[1] = time_ns() - t0; t0 = time_ns()
    end

    nl, llev, lkey, llo, lhi = ka_adaptive_build_leaves!(actx, sorted_keys, ell_max, K_max, n;
        workgroup=workgroup)

    if stage_ns !== nothing
        KernelAbstractions.synchronize(backend)
        stage_ns[2] = time_ns() - t0; t0 = time_ns()
    end

    n_balance_splits = 0
    if balance
        nl, n_balance_splits, llev, lkey, llo, lhi = ka_adaptive_balance!(actx, nl, llev, lkey,
            llo, lhi, sorted_keys, ell_max; workgroup=workgroup)
    end

    if stage_ns !== nothing
        KernelAbstractions.synchronize(backend)
        stage_ns[3] = time_ns() - t0; t0 = time_ns()
    end

    fin = ka_adaptive_finalize!(actx, nl, llev, lkey, llo, lhi, sorted_keys, ell_max, n, x_min,
        h0; workgroup=workgroup)

    # Publish the build into the context's DeviceRadixGrid. The arrays were
    # written in place through the `bufs` aliases; only the scalars need setting,
    # the same five `_cuda_refresh_adaptive_tree!` assigns.
    grid = actx.grid
    grid.x_min = SVector{3,TF}(x_min[1], x_min[2], x_min[3])
    grid.h0 = h0
    grid.ell = ell_max
    grid.n_bodies = n
    grid.n_cells = fin.n_leaves
    ka_fill_single_system_attribution!(grid.body_system, grid.body_index, n;
        workgroup=workgroup)

    if stage_ns !== nothing
        KernelAbstractions.synchronize(backend)
        stage_ns[4] = time_ns() - t0
    end

    node_sigma_max = if sigma_row > 0 && source_bodies !== nothing
        ka_adaptive_sigma_sweep!(actx, fin.node_lo, fin.node_hi, fin.child_ranges, fin.n_nodes,
            fin.level_offsets, ell_max, source_bodies, sigma_row; workgroup=workgroup)
    else
        nothing
    end

    return merge(fin, (grid=grid, perm=perm, invperm=invperm, sorted_keys=sorted_keys,
        n_balance_splits=n_balance_splits, node_sigma_max=node_sigma_max))
end

#------- Step (v): handing the built tree to a DeviceResidentRadixState -------#
#
# The analogue is `_cuda_allocate_adaptive_lifecycle` (translate_batched_cuda.jl),
# the ADAPTIVE path's state constructor -- not `cuda_radix_state`, which serves the
# uniform path and assumes exact-length grid arrays (`length(grid.node_keys)` ==
# n_nodes). That distinction is what makes this cheap: the adaptive path already
# hands its own capacity-sized `actx.grid` straight into the state and carries the
# logical extents in `state.counts::RadixStepCounts`, so `actx.grid` goes in as-is,
# by reference, with no truncating views and no host-mirror cross-check.
#
# Scope: this constructs the state. It does not build the interaction list or the
# operator workspace (`ResidentOperatorWorkspace`, whose plan construction is
# CUDA-specific), and it does not run the lifecycle -- B2M and L2B have no KA port.
# `interaction_list` and `scratch` are therefore `nothing`, and the route arrays are
# allocated empty. Those are the next steps, not omissions this one papers over.

function _ka_flat_buffer(backend, ::Type{TF}, basis_info::FastMultipole.OperatorBasisInfo{B,LH},
        batch::Integer) where {TF,B,LH}
    phi = KA.zeros(backend, TF, basis_info.basis_dof_phi, batch)
    chi = LH ? KA.zeros(backend, TF, basis_info.basis_dof_chi, batch) :
        KA.zeros(backend, TF, 0, 0)
    return FastMultipole.FlatCoefficientBuffer{TF,typeof(phi),B,LH}(phi, chi, basis_info)
end

"""
    ka_refresh_adaptive_lists!(lctx, actx, build; near_radius2, ell_max,
                               rho_t=0, sigma_armed=false)

Backend-agnostic port of `tree_batched_cuda.jl`'s `_cuda_refresh_adaptive_lists!`:
run the DTR sweep, partition the V stream into CSR routes, map the U endpoints to
leaf slots and rebuild the target-major U CSR, in that order, against the tree
`ka_build_adaptive_tree!` most recently wrote into `actx.grid`.

Returns the NamedTuple `ka_radix_state`'s `lists` keyword expects. `n_direct` is
the U-pair count and `n_routes` the CSR route count -- the two logical extents that
land in `state.counts`; the remaining counts are diagnostics (`n_dem` is the
sticky-demotion trigger count, nonzero only when the sigma gate is armed).

The lists live in `lctx.bufs` and are valid over `1:n_*`; nothing is copied.
"""
function ka_refresh_adaptive_lists!(lctx::KAAdaptiveListsContext,
        actx::KAAdaptiveTreeContext, build; near_radius2::Int, ell_max::Int,
        rho_t::Real=0, sigma_armed::Bool=false, workgroup::Int=KA_AUTO_WORKGROUP)
    grid = actx.grid
    b = actx.bufs
    n_nodes = build.n_nodes
    n_leaves = build.n_leaves
    gate = sigma_armed && rho_t > 0
    # the finest-lattice cell width, squared; the gate compares an integer-exact
    # squared AABB gap against (rho_t * sigma)^2 in these units
    delta_min = 2 * Float64(grid.h0) / (1 << ell_max)

    n_u, n_v, n_w, n_x, n_dem = ka_adaptive_build_lists!(lctx,
        grid.node_levels, grid.node_coords, grid.child_ranges, b.node_sigma;
        ell_max=ell_max, near_radius2=near_radius2, gate=gate, rho_t=rho_t,
        delta_min2=delta_min * delta_min, workgroup=workgroup)

    n_routes = ka_adaptive_partition_v!(lctx, n_v; workgroup=workgroup)

    n_u <= length(lctx.bufs.direct_targets) || throw(AssertionError(
        "adaptive KA direct capacity $(length(lctx.bufs.direct_targets)) exceeded " *
        "by n_u=$n_u"))
    ka_adaptive_u_slots!(lctx, b.leaf_slot_of, n_u; workgroup=workgroup)
    ka_adaptive_build_u_csr!(lctx, grid.cell_ranges, n_leaves, n_u; workgroup=workgroup)

    return (lctx=lctx, n_direct=n_u, n_routes=n_routes,
        n_u=n_u, n_v=n_v, n_w=n_w, n_x=n_x, n_dem=n_dem, n_nodes=n_nodes)
end

"""
    ka_radix_state(actx, build, source_buffer, P, lamb_helmholtz=Val(false);
                   options, n_root_nodes=1, workgroup=64)

Build a `DeviceResidentRadixState` around the tree `ka_build_adaptive_tree!` just
wrote into `actx.grid`. `build` is that call's return value (its `n_nodes`/`n_leaves`
supply the logical extents, which the grid itself does not carry); `source_buffer` is
the `dpb x n` source buffer in *global* (unsorted) body order, which is gathered into
the state's sorted-order body matrix.

`actx.grid` is stored by reference, not copied: `state.grid === actx.grid`, so a later
rebuild through the same context is visible to the state without reconstructing it.
The grid's arrays stay capacity-sized; every logical extent lives in `state.counts`
(`n_bodies`, `n_cells`, `n_nodes`), exactly as on the CUDA adaptive path.

Pass `lists` -- a `ka_refresh_adaptive_lists!` return -- to wire the interaction
list: `interaction_list` becomes the lists context, the route/direct arrays alias
its device buffers, and `counts.n_routes`/`counts.n_direct` carry their extents.
Omit it and those stay `nothing`/empty with both counts zero.

Still not wired, and `nothing`/empty rather than silently wrong: `scratch` (the
operator workspace builds CUDA-specific M2L plans), the `route_levels`/`route_offsets`
pair (the adaptive path routes by CSR class instead), and the host node/route mirrors. The host *body* mirrors
(`host_body_perm`/`host_body_system_ids`/`host_body_indices`) are downloaded once here,
so they are correct for this build and go stale on the next one -- the CUDA path
re-downloads them per step in `_cuda_update_adaptive_radix_state!`.
"""
function ka_radix_state(actx::KAAdaptiveTreeContext, build, source_buffer,
        P::Integer, lamb_helmholtz::Val{LH}=Val(false);
        options::FastMultipole.CUDARadixLifecycleOptions, lists=nothing,
        scratch=nothing, output_rows::Int=4, n_root_nodes::Int=1,
        workgroup::Int=KA_AUTO_WORKGROUP) where LH
    backend = actx.backend
    grid = actx.grid
    TF = typeof(grid.h0)
    options.precision === TF || throw(ArgumentError(
        "options.precision=$(options.precision) does not match the KA context's " *
        "element type $TF; allocate the context and the options at the same precision"))
    n = grid.n_bodies
    n_cells = grid.n_cells
    n_nodes = build.n_nodes
    n_nodes <= actx.node_capacity || throw(ArgumentError(
        "build n_nodes=$n_nodes exceeds the context node_capacity=$(actx.node_capacity)"))
    n_cells == build.n_leaves || throw(ArgumentError(
        "grid.n_cells=$n_cells disagrees with build.n_leaves=$(build.n_leaves); " *
        "`build` must be the result of the most recent ka_build_adaptive_tree! on `actx`"))

    basis_info = FastMultipole.OperatorBasisInfo(FastMultipole.CompressedComplexBasis(),
        P, lamb_helmholtz)
    counters = FastMultipole.CUDARadixTransferCounters()

    # sorted-order body matrix. Single-system attribution (step iv), so one isys=1
    # pack call covers every column; the kernel keeps CUDA's system indirection so
    # it stays correct when multi-system attribution lands.
    # One row past `data_per_body` carries 1/sigma for the regularized nearfield
    # (see `ka_pack_body_matrix_kernel!`). Rows 1:dpb keep their meaning exactly,
    # so `ka_node_sigma_max!` -- which maxes the TRUE sigma over each subtree to
    # drive the sigma-adequacy ell gate -- reads the same values as before.
    dpb = size(source_buffer, 1)
    sigma_row = _ka_kernel_sigma_row(options.direct_kernel)
    # CONVENTION (shared with `ka_radix_cache_device_build` and
    # `_ka_nf_inv_sigma_row`): for a regularized kernel (sigma_row > 0) the
    # LAST row of `source_bodies` is one past `dpb` and carries 1/sigma. It was
    # disabled only to keep the shape identical with the former native CUDA
    # allocator, which no longer exists.
    inv_sigma_row = sigma_row > 0 ? dpb + 1 : 0
    source_bodies = KA.zeros(backend, TF, dpb + (sigma_row > 0), actx.maxn)
    ka_pack_body_matrix!(source_bodies, source_buffer, grid.perm, grid.body_system,
        grid.body_index, n; isys=1, sigma_row=sigma_row, inv_sigma_row=inv_sigma_row,
        workgroup=workgroup)

    m2m_parent = KA.zeros(backend, Int, actx.node_capacity)
    m2m_child = KA.zeros(backend, Int, actx.node_capacity)
    l2l_parent = KA.zeros(backend, Int, actx.node_capacity)
    l2l_child = KA.zeros(backend, Int, actx.node_capacity)
    ka_tree_routes!(m2m_parent, m2m_child, l2l_parent, l2l_child, grid.parent_index,
        n_nodes; n_root_nodes=n_root_nodes, workgroup=workgroup)

    multipoles = _ka_flat_buffer(backend, TF, basis_info, actx.node_capacity)
    locals = _ka_flat_buffer(backend, TF, basis_info, actx.node_capacity)
    output = KA.zeros(backend, TF, output_rows, actx.maxn)

    # Route/direct wiring. With `lists` given (a `ka_refresh_adaptive_lists!`
    # return), the state ALIASES the lists context's device buffers rather than
    # copying them -- the task-023 zero-recurring-allocation contract, and the same
    # shape as the CUDA adaptive path, where the U-slot kernel writes straight into
    # `state.direct_targets`. `route_levels`/`route_offsets` stay empty on both
    # paths: the adaptive lifecycle routes by CSR class, not by (level, offset), so
    # a capacity-sized zero array there would only look meaningful.
    empty_iv = KA.zeros(backend, Int, 0)
    empty_im = KA.zeros(backend, Int, 3, 0)
    if lists === nothing
        route_targets = route_sources = direct_targets = direct_sources = empty_iv
        n_routes = n_direct = 0
        interaction_list = nothing
    else
        lb = lists.lctx.bufs
        route_targets, route_sources = lb.route_targets, lb.route_sources
        direct_targets, direct_sources = lb.direct_targets, lb.direct_sources
        n_routes, n_direct = lists.n_routes, lists.n_direct
        lists.n_nodes == n_nodes || throw(ArgumentError(
            "lists were built for n_nodes=$(lists.n_nodes) but this build has " *
            "n_nodes=$n_nodes; refresh the lists after the tree build"))
        interaction_list = lists.lctx
    end

    # one-shot download of the host body mirrors (see docstring on staleness)
    host_body_perm = Array{Int}(undef, n)
    host_body_system_ids = Array{Int}(undef, n)
    host_body_indices = Array{Int}(undef, n)
    copyto!(host_body_perm, 1, grid.perm, 1, n)
    copyto!(host_body_system_ids, 1, grid.body_system, 1, n)
    copyto!(host_body_indices, 1, grid.body_index, 1, n)
    counters.metadata_downloads += 3

    return FastMultipole.DeviceResidentRadixState{TF,FastMultipole.CompressedComplexBasis,LH}(
        grid, interaction_list, source_bodies, source_bodies,
        grid.perm, grid.body_system, grid.body_index,
        host_body_perm, host_body_system_ids, host_body_indices,
        nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing,
        grid.cell_centers, grid.cell_ranges,
        m2m_parent, m2m_child, l2l_parent, l2l_child,
        multipoles, locals,
        empty_iv, empty_im, route_targets, route_sources,
        direct_targets, direct_sources, output,
        FastMultipole.OperatorInvariantCache(TF, basis_info), scratch, counters, options,
        FastMultipole.RadixStepCounts(n, n_cells, n_nodes, n_routes, n_direct),
    )
end

#------- B2M (body -> multipole), step 3 -------#
#
# Port of `_cuda_b2m_vortex_leaf_nodes_kernel!` / `_cuda_b2m_leaf_nodes_kernel!`
# (src/translate_batched_cuda.jl): one workgroup per leaf cell, body-parallel
# accumulation, tree-reduced across the group.
#
# The per-body math is NOT reimplemented -- `_resident_vortex_phi_contrib` and
# `_resident_vortex_chi_contrib` live in src/translate_batched_resident.jl and
# are backend-agnostic `@inline` Julia shared with the CPU path. Only the
# reduction shape is ported.
#
# Workgroup size is 128 to match `CUDA_B2M_BLOCK`, not the 64 used by the tree
# kernels: the reduction is a halving tree, so its summation ORDER depends on
# the group size, and matching CUDA's block exactly is what makes this
# bit-exact against the CUDA reference rather than merely close. WG must be a
# power of two.
#
# Metal portability trap: the `@localmem` element type must be a compile-time
# constant reaching the `Val`-wrapped `SharedMemory` call, and computing it
# inside the kernel as `TF = eltype(phi)` does NOT qualify -- on Metal that
# compiles but raises a device-side "undefined variable error" at launch (KA's
# `@localmem` expansion, KernelAbstractions.jl:242). Pass the element type as a
# `::Type{TF}` kernel argument instead. `Val{WG}` dims are fine either way.
#
# Float32 discipline: every literal stays in TF. Apple GPUs reject Float64
# outright, and `_cuda_b2m_*` carries no Float64 of its own, so there is
# nothing to widen here -- but a stray `0.5` would silently promote and break
# Metal, so the accumulators are seeded with `zero(TF)`.

# The halving tree reduction is written out lexically in both places rather
# than factored into a helper: `@synchronize` must appear directly in the
# kernel body and in uniform control flow, so it can neither live inside a
# called function nor sit under a per-group `if`. (`ndrange = ncell * WG` gives
# exactly `ncell` groups, so no group guard is needed either.)
# B2M's three-in-one harmonic walk, KA-only (session 41).
#
# `_resident_vortex_phi_contrib` asks `_resident_vortex_q` for THREE adjacent-m
# coefficients at one n, and each of those restarts the full O(P^2) recurrence
# in `_resident_regular_harmonic_coeff`. Same for the chi contribution. With the
# (n,m) loop outside the body loop, a body pays 189 restarts per B2M call, and
# the ablation in `test/metal_env/_probe_b2m_ablate.jl` put that recurrence at
# 88.5% of the stage.
#
# One walk reaches all three columns: they differ only in m, and the recurrence
# is m-outer/n-inner, so it passes through m-1, m and m+1 on its way. That makes
# the stage cost one restart per (n,m) instead of three -- and the arithmetic
# reaching each captured value is op-for-op the shared function's, so the
# coefficients are BIT-IDENTICAL, not merely close. The reduction and the order
# bodies accumulate in are untouched too, so this kernel should stay bit-exact
# against both the CPU oracle and the CUDA reference.
#
# KA-ONLY BY CONSTRUCTION: `_resident_vortex_*_contrib` and
# `_resident_regular_harmonic_coeff` in src/ are left alone, so the CPU host
# kernel and CUDA keep their arithmetic. The duplicated math below must stay in
# lockstep with src/translate_batched_resident.jl.

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

"""
    ka_launch_b2m!(state; workgroup=128)

Body-to-multipole for the KA lifecycle: mirror of `_launch_cuda_b2m!`. Zeroes
the multipole buffers, then runs one workgroup per leaf cell. Vortex sources
require the Lamb-Helmholtz channel, exactly as the CUDA launcher does.
"""
function ka_launch_b2m!(state::FastMultipole.DeviceResidentRadixState{TF,B,LH};
        workgroup::Int=128) where {TF,B,LH}
    return ka_launch_b2m!(state, state.options.body_type; workgroup)
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

#------- L2B (local -> body output), step 4 -------#
#
# Port of `_cuda_l2b_output_kernel!` / `_cuda_l2b_output_hessian_kernel!`
# (src/translate_batched_cuda.jl). The per-body evaluation is NOT
# reimplemented: `_resident_local_eval_flat` and
# `_resident_local_eval_flat_hessian` (src/translate_batched_resident.jl) are
# backend-agnostic `@inline` Julia shared with the CPU path.
#
# Shape differs deliberately from CUDA's. CUDA runs a warp per cell (4 warps
# per 128-thread block) and strides bodies by 32; KA has no portable warp
# concept, so this runs one WORKGROUP per cell and strides by the workgroup
# size. That costs nothing and is safe: every body belongs to exactly one cell
# (`cell_ranges` partitions the sorted bodies) and each body writes only its
# own output column, so there is no reduction, no shared memory, no
# `@synchronize` and no atomic here. The value written for a given body is
# computed independently of the thread mapping, so this stays bit-exact
# against the CUDA kernel rather than merely close.

# The regular-harmonic sweep, KA-only (session 41).
#
# `_resident_regular_harmonic_coeff` (src/translate_batched_resident.jl) restarts
# a full O(P^2) associated-Legendre recurrence for EVERY (n,m) it is asked for.
# L2B's hessian branch asks 4 times per pair, so a body pays ~72 restarts where
# ONE m-outer/n-inner sweep produces every coefficient it needs. Measured at
# 72.6% of the L2B stage (`test/metal_env/_probe_l2b_ablate.jl`).
#
# `ka_regular_harmonic_sweep` below walks that same recurrence once and emits
# R_{n,m} as it goes. The arithmetic is op-for-op what the shared function does
# for each target -- the outer state (`rhom`, `pn`, `fact`, `ieim`) is untouched
# by the inner n loop, so extending that loop to P_active instead of stopping at
# each target changes no value. The COEFFICIENTS are therefore bit-identical;
# only the order in which the 13 outputs accumulate over (n,m) changes, which is
# a Float32 rounding difference and why the gate scores relerr, not equality.
#
# KA-ONLY BY CONSTRUCTION: this lives in the extension and the shared
# `_resident_local_eval_flat*` are left exactly as they are, so the CPU host
# kernels and the CUDA path keep their current arithmetic and stay the oracle.
# The cost of that choice is a second copy of the evaluation math here, which
# must stay in lockstep with `src/translate_batched_resident.jl`.

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

Local-to-body evaluation for the KA lifecycle: mirror of
`_launch_cuda_resident_l2b_only!`, minus the stream-event overlap (the KA
driver runs its stages in order). Selects the 13-row hessian variant on
`size(state.output, 1) >= 13`, the same gate the CUDA launcher uses; FLOWVPM
runs with `hessian=true` and therefore takes that branch.
"""
function ka_launch_l2b!(state::FastMultipole.DeviceResidentRadixState{TF,B,LH};
        workgroup::Int=64) where {TF,B,LH}
    ncell = state.counts.n_cells
    ncell == 0 && return state
    orders = state.invariant_cache.basis_info.orders
    backend = KA.get_backend(state.output)
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

#------- NEARFIELD (U-list direct pairs), step 5 -------#
#
# Port of the `:pairs` shape of `_cuda_direct_pairs_functor_kernel!`
# (src/translate_batched_cuda.jl). The per-pair math is NOT reimplemented:
# `_direct_pair_ug` / `_direct_pair_ugh` (src/translate_batched_resident.jl)
# are backend-agnostic and shared with the CPU path, so every kernel functor
# (PartitionedVortex, RegularizedVortex, SingularSource, ...) comes along for
# free.
#
# Deliberately ports ONLY the `:pairs` shape. The fused target-owned CSR
# shapes, the symmetric Newton-pair path, the binned split-vortex path and the
# g/h lookup table are all skipped (`ghv = Val(:shipped)`, `shlut = nothing`) --
# they are performance variants of the same physics. That leaves the KA U-CSR
# built in tree Phase G unused for now; it is correct and gated, and a fused
# shape can consume it later as a perf step.
#
# `_cuda_fast_rsqrt` is replaced by a plain `inv(sqrt(r2))`. That makes the KA
# kernel the MORE accurate side, so this stage is gated against the CPU
# reference, never against the CUDA kernel.
#
# Workgroup per pair, workitems striding the target bodies (CUDA uses a warp
# per pair striding by 32). Targets of different pairs overlap, so the output
# accumulation must stay atomic.

# 1/sqrt(r2). Float32: the native path. Float64: a Float32 seed refined by two
# Newton steps (24 -> 48 -> 53 bits), the same trick as _cuda_fast_rsqrt; a
# full FP64 sqrt+divide was 1.7x of the Float32 nearfield on an H200.
@inline _ka_invsqrt(r2::Float32) = inv(sqrt(r2))
@inline function _ka_invsqrt(r2::Float64)
    y = Float64(inv(sqrt(Float32(r2))))
    y = y * (1.5 - 0.5 * r2 * y * y)
    y = y * (1.5 - 0.5 * r2 * y * y)
    return y
end
@inline _ka_invsqrt(r2) = inv(sqrt(r2))

@kernel function ka_direct_pairs_functor_kernel!(kernel, output, @Const(source_bodies),
        @Const(cell_ranges), @Const(direct_targets), @Const(direct_sources),
        npairs, ::Type{T}, ::Val{HS}, ::Val{WG}) where {T,HS,WG}
    pair_i = @index(Group)
    tid = @index(Local)
    ep = FastMultipole._emits_potential(kernel)
    ghv = Val(:shipped)
    @inbounds begin
        target_cell = direct_targets[pair_i]
        source_cell = direct_sources[pair_i]
        tfirst = cell_ranges[1, target_cell]
        tlast = tfirst + cell_ranges[2, target_cell] - 1
        sfirst = cell_ranges[1, source_cell]
        slast = sfirst + cell_ranges[2, source_cell] - 1
        i = tfirst + tid - 1
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
                        invr = _ka_invsqrt(r2)
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
            i += WG
        end
    end
end

#------- warp-per-pair nearfield (CUDA-shaped launch) -------#
#
# The kernel above is a 1:1 port of the native `:pairs` shape EXCEPT for the
# launch geometry and the reciprocal square root, and on an H200 those two
# differences cost ~1.5x at np=249k (KA-only 0.109 s vs native 0.067 s, job
# 13567764 vs 13563393; the native run had silently been what every earlier
# "KA" H200 number measured). This variant restores the native choices:
#   * LANES threads per pair (a warp, 32, on CUDA), WG/LANES pairs per block,
#     so 128-thread blocks carry four pairs instead of one 64-thread block per
#     pair -- twice the resident warps per SM and a quarter of the blocks;
#   * `rsqrt.approx` through the NVVM intrinsic instead of IEEE sqrt+divide in
#     the innermost line (native `_cuda_fast_rsqrt`); Float64 seeds two Newton
#     steps from it exactly as before;
#   * the g/h mode as a parameter (native default `:fp32`; identical to
#     `:shipped` for Float32 fields).
# Launch geometry comes from `_nf_config` (lanes follow the cell population;
# RADIX_NF_LANES / RADIX_NF_WG override for other hardware). This is the only
# nearfield kernel the lifecycle launches; the kernel above remains for the
# all-pairs direct arm and the probes.

# libdevice's rsqrtf (what CUDA.rsqrt and the native `_cuda_fast_rsqrt` call);
# resolved by the CUDA compiler's libdevice link, so only reachable when
# `_nf_config` enables it on a CUDABackend.
@inline _ka_rsqrt_approx(r2::Float32) =
    ccall("extern __nv_rsqrtf", llvmcall, Cfloat, (Cfloat,), r2)
@inline _ka_invsqrt(r2::Float32, ::Val{true}) = _ka_rsqrt_approx(r2)
@inline function _ka_invsqrt(r2::Float64, ::Val{true})
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

_env_int(name, default) = parse(Int, get(ENV, name, string(default)))

# (A register-tiled variant -- two targets per thread through one source pass
# -- was tried and REMOVED: 10% slower at 115k and 249k on the H200 and 4%
# slower in Float64, job 13569195. The loop is not load-bound.)

# CONVENTION: for a regularized kernel (sigma_row > 0) both allocators size
# `source_bodies` one row past the packed body rows and the packers fill that
# LAST row with 1/sigma, so the nearfield can multiply instead of divide per
# interaction (13% at 64 lanes, ~0 at 128+ on the H200). Always on.
function _ka_nf_inv_sigma_row(state)
    _ka_kernel_sigma_row(state.options.direct_kernel) > 0 || return 0
    return size(state.source_bodies, 1)
end
_env_bool(name, default) = get(ENV, name, default ? "1" : "0") == "1"

# Per-backend nearfield launch configuration (see the block above). Only the
# backend TYPE NAME is consulted, so this extension stays free of CUDA.
#
# Defaults measured on an H200, NREL wake np=248714 (jobs 13567846/931/8101,
# 20 calls, median): one block per pair with ALL its lanes on that pair, and
# the lane count is what matters -- 64 lanes 0.099 s, 128 0.072, 256 0.069,
# 512 0.074 in Float32 (native CUDA lifecycle: 0.067-0.074); Float64 128 lanes
# 0.130, 256 0.135 (native 0.18-0.19). The native warp-per-pair geometry
# (128x32) was SLOWER here, 0.130. The reciprocal-sigma row bought 13% at 64
# lanes and nothing at 128+, where the divide latency is already hidden.
function _nf_config(backend, ::Type{TF}; bodies_per_cell::Real=0) where TF
    cuda = nameof(typeof(backend)) === :CUDABackend
    # Lanes per pair follow the cell population (the lanes stride a pair's
    # TARGET bodies): a dense field wants a whole block on each pair, a thin
    # one wants warp-sized teams so lanes are not idle. Clamped to [64, 256]
    # (Float64: 128 -- its measured optimum at 249k) and rounded to a power of
    # two. RADIX_NF_LANES / RADIX_NF_WG override, for hardware other than the
    # H200 this was tuned on. Non-CUDA backends: 64 lanes, one pair per block
    # (measured flat 64-256 on Metal).
    hi = TF === Float32 ? 256 : 128
    auto_lanes = cuda && bodies_per_cell > 0 ?
        clamp(nextpow(2, max(1, ceil(Int, bodies_per_cell))), 64, hi) : (cuda ? hi : 64)
    lanes = _env_int("RADIX_NF_LANES", haskey(ENV, "RADIX_NF_WG") ? _env_int("RADIX_NF_WG", 64) : auto_lanes)
    wg = _env_int("RADIX_NF_WG", cuda ? max(128, lanes) : lanes)
    wg % lanes == 0 || throw(ArgumentError("RADIX_NF_WG=$wg must be a multiple of RADIX_NF_LANES=$lanes"))
    # libdevice rsqrtf is CUDA-only; other backends keep inv(sqrt)
    return (; wg, lanes, fast=cuda)
end

"""
    ka_launch_nearfield!(state; workgroup=nothing, clear=true)

U-list direct nearfield for the KA lifecycle: mirror of
`_launch_cuda_nearfield_kernel!` restricted to the `:pairs` shape. Zeroes
`state.output` (this is the first stage of the lifecycle, as on CUDA) unless
`clear=false`.
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
    wg = workgroup === nothing ? cfg.wg : workgroup
    lanes = min(cfg.lanes, wg)
    dkernel = _ka_device_direct_kernel(state.options.direct_kernel, TF,
        _ka_nf_inv_sigma_row(state))
    kern = _cached_kernel(ka_direct_pairs_warp_kernel!, backend, wg)
    kern(dkernel, state.output, state.source_bodies,
         state.cell_ranges, state.direct_targets, state.direct_sources,
         npairs, TF, Val(hs), Val(wg), Val(lanes), Val(cfg.fast), Val(:shipped);
         ndrange=cld(npairs, wg ÷ lanes) * wg)
    return state
end

#------- all-pairs direct arm (task 053) -------#
#
# An opt-in alternative to the FMM lifecycle: one kernel, no grid, no routes,
# no tree, every pair evaluated. Armed only by the caller through the radix
# setting `:RADIX_DIRECT_ARM`; nothing selects it automatically.
#
# There IS a crossover below which this beats the lifecycle -- measured at
# np ~ 6e4 on Metal for the for_ryan wake, with the arm 2-4x ahead below
# np = 8192. That number is one backend, one wake, one sweep, and the
# lifecycle's cost balance is not the same on CUDA, so it is recorded here as
# an observation and deliberately NOT turned into a dispatch rule. Anything
# that selects between the two arms needs a cross-backend calibration first.
#
# Shape differs from `ka_direct_pairs_functor_kernel!` deliberately. There a
# workgroup owns a cell PAIR and its targets are shared with other pairs, so
# every accumulation is atomic. Here a workitem owns one target body outright
# and no other workitem touches it, so the accumulators are plain stores --
# which also makes the result deterministic, unlike the pair shape.
#
# The physics is the same `_direct_pair_ug` / `_direct_pair_ugh` shared with
# the CPU path, with the same `ghv = Val(:shipped)` series and the same plain
# `inv(sqrt(r2))`, so this arm is gated against the CPU reference exactly as
# the pair shape is.

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
against every source body in one kernel. `state.output` is written, not
accumulated (the kernel owns one column per workitem), so no clear is needed;
the columns past `counts.n_bodies` are left as they were, and
`ka_finalize_radix_output!` reads only the valid prefix.
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
    # inv_sigma_row=0: rho is computed as |r|/sigma, the divide CUDA does
    dkernel = _ka_device_direct_kernel(state.options.direct_kernel, TF, 0)
    kern(dkernel, state.output, state.source_bodies, n, TF, Val(hs);
         ndrange=cld(n, wg) * wg)
    KA.synchronize(backend)
    return state
end

#------- S2L (X list: source bodies -> local), step 6 -------#
#
# Port of `_cuda_adaptive_s2l_vortex_kernel!` (src/translate_batched_cuda.jl).
# The adaptive tree emits W and X pairs, so omitting these stages does not just
# cost accuracy -- it silently drops physics.
#
# The per-thread harmonics scratch is a slice of a GLOBAL array indexed by
# global thread id (`H = view(Hall, :, slot:slot, :)`), not shared memory, so
# it ports across directly: no `@localmem`, no `@synchronize`. `Hall` must be
# sized `(2, n_groups * workgroup, nH2)`.
#
# All the harmonic helpers (`cartesian_to_spherical`, `irregular_harmonics!`,
# `harmonic_index`, `flat_basis_index`, `_adt_S_re/_im`) live in non-CUDA src
# files and are reused unchanged.
#
# The element type is taken as a `::Type{TF}` argument rather than
# `eltype(lp)`: not strictly required here (there is no `@localmem`), but it
# keeps every kernel in this file on one convention.

@kernel function ka_adaptive_s2l_vortex_kernel!(lp, lc, @Const(source_bodies),
        @Const(cell_ranges), @Const(leaf_slot_of), @Const(node_centers),
        @Const(x_targets), @Const(x_sources), n_x, Hall, ::Type{TF},
        ::Val{P_phi}, ::Val{P_active}, n_g, ::Val{WG}) where {TF,P_phi,P_active,WG}
    grp = @index(Group)
    tid = @index(Local)
    gslot = (grp - 1) * WG + tid
    H = view(Hall, :, gslot:gslot, :)
    k = grp
    @inbounds while k <= n_x
        ia = Int(x_targets[k])
        ib = Int(x_sources[k])
        slot = Int(leaf_slot_of[ib])
        first = cell_ranges[1, slot]
        count = cell_ranges[2, slot]
        cx = node_centers[1, ia]; cy = node_centers[2, ia]; cz = node_centers[3, ia]
        sB = first + tid - 1
        while sB <= first + count - 1
            dx = source_bodies[1, sB] - cx
            dy = source_bodies[2, sB] - cy
            dz = source_bodies[3, sB] - cz
            wx = source_bodies[5, sB]; wy = source_bodies[6, sB]; wz = source_bodies[7, sB]
            r, theta, phi = FastMultipole.cartesian_to_spherical(dx, dy, dz)
            FastMultipole.irregular_harmonics!(H, r, theta, phi, P_phi + 2)
            for n in 1:P_phi
                _1_n = isodd(n) ? -one(TF) : one(TF)
                n_inv = inv(TF(n))
                for m in 0:n
                    _1_m = isodd(m) ? -one(TF) : one(TF)
                    i = FastMultipole.harmonic_index(n, m)
                    local Spre::TF, Spim::TF, Smre::TF, Smim::TF
                    if m < n
                        Spre = -_1_m * FastMultipole._adt_S_re(H, i + 1)
                        Spim = _1_m * FastMultipole._adt_S_im(H, i + 1)
                    else
                        Spre = zero(TF); Spim = zero(TF)
                    end
                    Sre = _1_m * FastMultipole._adt_S_re(H, i)
                    Sim = -_1_m * FastMultipole._adt_S_im(H, i)
                    if m == 0
                        Smre = -_1_m * Spre; Smim = _1_m * Spim
                    else
                        Smre = -_1_m * FastMultipole._adt_S_re(H, i - 1)
                        Smim = _1_m * FastMultipole._adt_S_im(H, i - 1)
                    end
                    row = FastMultipole.flat_basis_index(n, m, 1)
                    KA.@atomic lp[row, ia] += -_1_n * n_inv * (
                        (n - m) * TF(0.5) * (wx * Spre - wy * Spim) -
                        (n + m) * TF(0.5) * (wx * Smre + wy * Smim) +
                        wz * m * Sim)
                    KA.@atomic lp[row + 1, ia] += -_1_n * n_inv * (
                        (n - m) * TF(0.5) * (wx * Spim + wy * Spre) -
                        (n + m) * TF(0.5) * (wx * Smim - wy * Smre) -
                        wz * m * Sre)
                end
            end
            for n in 0:P_active
                _1_np1 = isodd(n + 1) ? -one(TF) : one(TF)
                np1_inv = inv(TF(n + 1))
                for m in 0:n
                    _1_m = isodd(m) ? -one(TF) : one(TF)
                    i_np1 = FastMultipole.harmonic_index(n + 1, m)
                    Sp1pre = -_1_m * FastMultipole._adt_S_re(H, i_np1 + 1)
                    Sp1pim = _1_m * FastMultipole._adt_S_im(H, i_np1 + 1)
                    Sp1re = _1_m * FastMultipole._adt_S_re(H, i_np1)
                    Sp1im = -_1_m * FastMultipole._adt_S_im(H, i_np1)
                    local Sp1mre::TF, Sp1mim::TF
                    if m == 0
                        Sp1mre = -_1_m * Sp1pre; Sp1mim = _1_m * Sp1pim
                    else
                        Sp1mre = -_1_m * FastMultipole._adt_S_re(H, i_np1 - 1)
                        Sp1mim = _1_m * FastMultipole._adt_S_im(H, i_np1 - 1)
                    end
                    row = FastMultipole.flat_basis_index(n, m, 1)
                    KA.@atomic lc[row, ia] += _1_np1 * np1_inv * (
                        TF(0.5) * (wy * Sp1mre - wx * Sp1mim) -
                        TF(0.5) * (wy * Sp1pre + wx * Sp1pim) - wz * Sp1re)
                    KA.@atomic lc[row + 1, ia] += _1_np1 * np1_inv * (
                        TF(0.5) * (wy * Sp1mim + wx * Sp1mre) -
                        TF(0.5) * (wy * Sp1pim - wx * Sp1pre) - wz * Sp1im)
                end
            end
            sB += WG
        end
        k += n_g
    end
end

#------- M2T (W list: multipole -> target bodies), step 6b -------#
#
# Port of `_cuda_adaptive_m2t_kernel!`. Like S2L this uses the GLOBAL
# per-thread harmonics scratch (no shared memory), and the evaluation itself
# is the shared `_resident_multipole_eval_flat[_hessian]` from
# src/translate_batched_resident.jl -- so only the traversal is ported.
#
# Note the asymmetry with S2L: here `leaf_slot_of` is indexed by the TARGET
# (`ia`) and the expansion centre comes from the SOURCE node (`ib`).

@kernel function ka_adaptive_m2t_kernel!(output, @Const(source_bodies), @Const(cell_ranges),
        @Const(leaf_slot_of), @Const(node_centers), @Const(w_targets), @Const(w_sources),
        n_w, @Const(ph), @Const(ch), Hall, ::Type{TF}, ::Val{P_phi}, ::Val{P_active},
        ::Val{LH}, ::Val{HS}, n_g, ::Val{WG}) where {TF,P_phi,P_active,LH,HS,WG}
    grp = @index(Group)
    tid = @index(Local)
    gslot = (grp - 1) * WG + tid
    H = view(Hall, :, gslot:gslot, :)
    k = grp
    @inbounds while k <= n_w
        ia = Int(w_targets[k])
        ib = Int(w_sources[k])
        slot = Int(leaf_slot_of[ia])
        first = cell_ranges[1, slot]
        count = cell_ranges[2, slot]
        cx = node_centers[1, ib]; cy = node_centers[2, ib]; cz = node_centers[3, ib]
        i = first + tid - 1
        while i <= first + count - 1
            dx = source_bodies[1, i] - cx
            dy = source_bodies[2, i] - cy
            dz = source_bodies[3, i] - cz
            r, theta, phi = FastMultipole.cartesian_to_spherical(dx, dy, dz)
            FastMultipole.irregular_harmonics!(H, r, theta, phi, P_phi + 2)
            if HS
                vals = FastMultipole._resident_multipole_eval_flat_hessian(ph, ch, ib, H,
                    P_phi, P_active, Val(LH))
                Base.Cartesian.@nexprs 13 r_ -> (KA.@atomic output[r_, i] += vals[r_])
            else
                u, gx, gy, gz = FastMultipole._resident_multipole_eval_flat(ph, ch, ib, H,
                    P_phi, P_active, Val(LH))
                KA.@atomic output[1, i] += u
                KA.@atomic output[2, i] += gx
                KA.@atomic output[3, i] += gy
                KA.@atomic output[4, i] += gz
            end
            i += WG
        end
        k += n_g
    end
end

"""
    ka_launch_adaptive_m2t!(state, lists, actx, harmonics_scratch; workgroup=128)

W-list multipole-to-target for the KA lifecycle: mirror of
`_launch_cuda_adaptive_m2t!`. Selects the 13-row hessian variant on
`size(state.output, 1) >= 13`, as CUDA does.

`ng` is passed as a RUNTIME `Int`, not a `Val`, and this is deliberate: it is
the window-loop stride, CUDA's `gridDim().x`, which CUDA also reads at runtime.
As `Val(ng)` it was `min(n_w, KA_S2L_GROUPS)` baked into a type parameter, so
every distinct window count forced a fresh specialization and a fresh device
compile -- 82-300 s each on Metal. It buys nothing: the stride is only ever an
increment. `ka_launch_adaptive_s2l!` is the same. Do not "optimize" these back
into `Val`.
"""
function ka_launch_adaptive_m2t!(state::FastMultipole.DeviceResidentRadixState{TF,B,LH},
        lists, actx, harmonics_scratch; workgroup::Int=128) where {TF,B,LH}
    n_w = lists.n_w
    n_w == 0 && return state
    orders = state.invariant_cache.basis_info.orders
    hs = size(state.output, 1) >= 13
    ng = min(n_w, KA_S2L_GROUPS)
    backend = KA.get_backend(state.output)
    b = lists.lctx.bufs
    kern = _cached_kernel(ka_adaptive_m2t_kernel!, backend, workgroup)
    kern(state.output, state.source_bodies, state.cell_ranges, actx.bufs.leaf_slot_of,
         state.grid.node_centers, b.w_targets, b.w_sources, n_w,
         FastMultipole.phi_slab(state.multipoles), FastMultipole.chi_slab(state.multipoles),
         harmonics_scratch, TF, Val(orders.P_phi), Val(orders.P_active), Val(LH),
         Val(hs), ng, Val(workgroup); ndrange=ng * workgroup)
    return state
end

const KA_S2L_GROUPS = 512   # mirrors _ADT_CUDA_HARMONIC_BLOCKS

"""
    ka_launch_adaptive_s2l!(state, lists, actx, harmonics_scratch; workgroup=128)

X-list source-to-local for the KA lifecycle: mirror of
`_launch_cuda_adaptive_s2l!`, vortex channel. The X pair arrays live on the
lists context (`lists.lctx.bufs`) and `leaf_slot_of` on the tree context
(`actx.bufs`), so both are needed. `harmonics_scratch` must be
`(2, n_groups * workgroup, nH2)` -- see `ka_allocate_harmonics_scratch`.
"""
function ka_launch_adaptive_s2l!(state::FastMultipole.DeviceResidentRadixState{TF,B,LH},
        lists, actx, harmonics_scratch; workgroup::Int=128) where {TF,B,LH}
    n_x = lists.n_x
    n_x == 0 && return state
    LH || throw(ArgumentError("adaptive S2L (vortex) requires the Lamb-Helmholtz channel"))
    orders = state.invariant_cache.basis_info.orders
    ng = min(n_x, KA_S2L_GROUPS)
    backend = KA.get_backend(state.locals.phi)
    b = lists.lctx.bufs
    kern = _cached_kernel(ka_adaptive_s2l_vortex_kernel!, backend, workgroup)
    kern(FastMultipole.phi_slab(state.locals), FastMultipole.chi_slab(state.locals),
         state.source_bodies, state.cell_ranges, actx.bufs.leaf_slot_of,
         state.grid.node_centers, b.x_targets, b.x_sources, n_x,
         harmonics_scratch, TF, Val(orders.P_phi), Val(orders.P_active),
         ng, Val(workgroup); ndrange=ng * workgroup)
    return state
end

"""
    ka_allocate_harmonics_scratch(backend, TF, P_phi; groups=KA_S2L_GROUPS, workgroup=128)

Per-thread irregular-harmonics scratch for the S2L/M2T kernels, shaped
`(2, groups * workgroup, nH2)`. Mirrors CUDA's
`CUDA.zeros(TF, 2, _ADT_CUDA_HARMONIC_BLOCKS * 128, nH2)`.
"""
function ka_allocate_harmonics_scratch(backend, ::Type{TF}, P_phi::Integer;
        groups::Int=KA_S2L_GROUPS, workgroup::Int=128) where TF
    nH2 = FastMultipole.harmonic_index(P_phi + 2, P_phi + 2)
    return KA.zeros(backend, TF, 2, groups * workgroup, nH2)
end

#------- KA LIFECYCLE DRIVER (step 7a) -------#
#
# Mirror of `_cuda_lifecycle_body!` (src/translate_batched_cuda.jl) for the
# UNIFORM radix lifecycle -- the one FLOWVPM actually runs.
#
# Why uniform and not adaptive: `_radix_cache_device_step!` branches on
# `cache.adaptive === nothing`, and FLOWVPM builds its `RadixFMMCache` with
# `ell`/`near_radius2`/`window_classes` and NO adaptive policy, so it takes
# `run_cuda_radix_lifecycle!`. It could not take the adaptive path anyway: that
# lifecycle rejects `PartitionedVortex` (FLOWVPM's shipped default kernel, a
# task-040 deferral) and the Lamb-Helmholtz M2T hessian throws outright.
#
# The uniform per-step body is only three stages and rebuilds no tree (the
# lattice is fixed at cache construction):
#     nearfield -> B2M -> [M2M -> M2L -> L2L -> L2B]
#
# This driver deliberately calls the ext's STANDALONE `ka_*` stage drivers
# rather than FastMultipole's generic ones. The generic drivers dispatch their
# primitives on array type, which means they run KA kernels on Metal but NATIVE
# CUDA kernels on CuArrays -- correct for production, useless for a KA-vs-native
# A/B on one GPU. Going through the standalone drivers makes "KA" unambiguous on
# every backend, so the same state can be run both ways and compared.
#
# One `KA.synchronize` at the END of the driver, never per stage: per-kernel
# syncs cost 1.37-2.7x in earlier measurements on this code.

"""
    ka_lifecycle_body!(state; workgroup_b2m=128, workgroup=64, sync=true)

`workgroup` here is *not* the auto-resolved occupancy knob: it reaches
`ka_launch_nearfield!` and `ka_launch_l2b!`, where it is the per-pair/per-cell
team size their `@localmem` extents are declared against. It stays explicit.

Run the uniform radix lifecycle over `state` entirely with KA kernels. `state`
may be resident on any KA backend, including a `CuArray` state built by the
existing `RadixFMMCache(device=true)` -- which is how the KA-vs-native
comparison runs both arms over identical data with no second cache build.
"""
function ka_lifecycle_body!(state::FastMultipole.DeviceResidentRadixState{TF,B,LH};
        workgroup_b2m::Int=128, workgroup::Int=64, sync::Bool=true) where {TF,B,LH}
    ws = state.scratch
    ws isa FastMultipole.ResidentOperatorWorkspace || throw(ArgumentError(
        "ka_lifecycle_body! requires a ResidentOperatorWorkspace in state.scratch"))

    # 1. nearfield (clears state.output, as CUDA's fill+nearfield does)
    ka_launch_nearfield!(state; clear=true)   # shape/workgroup from _nf_config

    # 2. B2M
    ka_launch_b2m!(state; workgroup=workgroup_b2m)

    # 3. far field: M2M -> M2L -> L2L, then L2B
    FastMultipole._zero_resident_nonleaf_multipoles!(state)
    for group in ws.m2m_groups
        ka_resident_stage_group_apply!(state.multipoles, state.multipoles, group, ws, :m2m)
    end
    ka_launch_m2l!(state, ws)
    for group in ws.l2l_groups
        ka_resident_stage_group_apply!(state.locals, state.locals, group, ws, :l2l)
    end
    ka_launch_l2b!(state; workgroup=workgroup)

    sync && KA.synchronize(KA.get_backend(state.output))
    return state
end

# Selected by `set_radix_setting!(:RADIX_KA_LIFECYCLE, true)`; see the stub in
# src/radix_settings.jl. Keyword defaults are the driver's own -- the setting is
# a boolean arm switch, not a tuning surface.
FastMultipole.ka_radix_lifecycle!(state::FastMultipole.DeviceResidentRadixState) =
    ka_lifecycle_body!(state)

"""
    ka_launch_m2l!(state, ws)

M2L stage of [`ka_lifecycle_body!`](@ref), branching on the resident
interaction context exactly as `_launch_cuda_resident_m2l!` does: a
`DeviceHierarchicalM2LContext` generates and applies route windows here, and
anything else is the flat whole-route concat apply.

The branch is not optional. FLOWVPM's `RadixFMMCache` carries the hierarchical
policy, and on that context `state.counts.n_routes` holds only the LAST
window's route count -- a flat apply would silently translate a fraction of the
V list and be wrong rather than merely slow.
"""
function ka_launch_m2l!(state::FastMultipole.DeviceResidentRadixState{TF,B,LH},
        ws) where {TF,B,LH}
    hctx = state.interaction_list
    hctx isa FastMultipole.DeviceHierarchicalM2LContext &&
        return ka_hierarchical_m2l!(state, hctx, ws)
    fill!(state.locals.phi, zero(TF))
    LH && fill!(state.locals.chi, zero(TF))
    nroutes = state.counts.n_routes
    nroutes > 0 && ka_resident_m2l_concat_apply!(state.locals, state.multipoles, ws,
        state.route_sources, state.route_targets, nroutes)
    return state
end

#------- hierarchical M2L window generation (KA) -------#
#
# KA port of `_cuda_hier_generate_window_core!`
# (src/translate_batched_cuda.jl:7350). This is the flag/scan/compact that fills
# `route_targets`/`route_sources`/`route_class` for one (level, offset-class)
# window. It was the last CUDA-only dependency inside `ka_hierarchical_m2l!`,
# and therefore the reason the hierarchical arm could not be gated on Metal.
#
# `DeviceHierarchicalM2LContext` is already array-type generic
# (containers.jl:782: IV32/IM32/IA32/IV/SM type parameters), so nothing here
# needs a CUDA-specific context mirror -- unlike the lifecycle, which needed
# `host_radix_state`. The scan reuses `_ka_scan_total!`'s `accumulate!`, which
# is backend-generic via GPUArrays.
#
# The three kernels are elementwise index math with no shared memory and no
# CUDA intrinsics, so they are direct translations and carry no `@localmem`
# team-size coupling: all three take `KA_AUTO_WORKGROUP`.

@kernel function ka_hier_route_flags_kernel!(flags, @Const(node_at),
        @Const(node_coords), @Const(push_offsets), @Const(class_of),
        level_base_L, first_source, n_sources, first_offset, kn, L)
    idx = @index(Global)
    @inbounds if idx <= kn * n_sources
        kloc = (idx - 1) ÷ n_sources + 1
        s = (idx - 1) % n_sources + 1
        k = first_offset + kloc - 1
        source = first_source + s - 1
        G = 1 << L
        cx = node_coords[1, source]
        cy = node_coords[2, source]
        cz = node_coords[3, source]
        # same x/y/z bit convention as _rigid_phase_index
        phase = 1 + (cx & 1) + 2 * (cy & 1) + 4 * (cz & 1)
        hit = Int32(0)
        if class_of[phase, k, L + 1] != Int32(0)
            tx = cx + push_offsets[1, k]
            ty = cy + push_offsets[2, k]
            tz = cz + push_offsets[3, k]
            if 0 <= tx < G && 0 <= ty < G && 0 <= tz < G
                linear = tx + G * (ty + G * tz)
                node_at[level_base_L + linear + 1] == Int32(0) || (hit = Int32(1))
            end
        end
        flags[idx] = hit
    end
end

# Per-class cumulative window counts read straight off the inclusive scan: class
# `kloc` ends at flat index `kloc * n_sources`.
@kernel function ka_hier_window_cum_kernel!(cum, @Const(prefix), n_sources, kn)
    i = @index(Global)
    @inbounds if i <= kn
        cum[i] = prefix[i * n_sources]
    end
end

# Compact one window into the start of the reusable route buffers. Offsets are
# the unscaled integer push offsets; endpoints are flat node indices.
# Base-offset variants for the concatenated-window scan in ka_hier_cache_windows!:
# every window's flags live at `base+1 : base+used` of one buffer, ONE inclusive
# scan runs over all of them, and a window's local prefix is
# prefix[base+idx] - prefix[base].
@kernel function ka_hier_window_cum_base_kernel!(cum, @Const(prefix), base, n_sources, kn)
    i = @index(Global)
    @inbounds if i <= kn
        pb = base > 0 ? prefix[base] : zero(eltype(prefix))
        cum[i] = prefix[base + i * n_sources] - pb
    end
end

@kernel function ka_hier_route_compact_base_kernel!(route_levels, route_offsets,
        route_targets, route_sources, route_class, @Const(flags), @Const(prefix), base,
        @Const(node_at), @Const(node_coords), @Const(push_offsets),
        level_base_L, first_source, n_sources, first_offset, kn, L, class_base)
    idx = @index(Global)
    @inbounds if idx <= kn * n_sources && flags[base + idx] == Int32(1)
        kloc = (idx - 1) ÷ n_sources + 1
        s = (idx - 1) % n_sources + 1
        k = first_offset + kloc - 1
        source = first_source + s - 1
        G = 1 << L
        ox = push_offsets[1, k]
        oy = push_offsets[2, k]
        oz = push_offsets[3, k]
        tx = node_coords[1, source] + ox
        ty = node_coords[2, source] + oy
        tz = node_coords[3, source] + oz
        linear = tx + G * (ty + G * tz)
        target = Int(node_at[level_base_L + linear + 1])
        pb = base > 0 ? prefix[base] : zero(eltype(prefix))
        p = Int(prefix[base + idx] - pb)
        route_levels[p] = L
        route_offsets[1, p] = Int(ox)
        route_offsets[2, p] = Int(oy)
        route_offsets[3, p] = Int(oz)
        route_targets[p] = target
        route_sources[p] = source
        route_class[p] = Int32(class_base + k)
    end
end

# Global-position variant for the concat window cache: with one scan over all
# windows, prefix[base+idx] IS the route's slot in the concatenated stream, so
# the compact writes the cache arrays directly -- no per-window staging, no
# device-to-device copies. Only the three arrays the concat apply reads.
@kernel function ka_hier_route_compact_global_kernel!(win_targets, win_sources, win_class,
        @Const(flags), @Const(prefix), base, @Const(node_at), @Const(node_coords),
        @Const(push_offsets), level_base_L, first_source, n_sources, first_offset, kn, L,
        class_base)
    idx = @index(Global)
    @inbounds if idx <= kn * n_sources && flags[base + idx] == Int32(1)
        kloc = (idx - 1) ÷ n_sources + 1
        s = (idx - 1) % n_sources + 1
        k = first_offset + kloc - 1
        source = first_source + s - 1
        G = 1 << L
        ox = push_offsets[1, k]
        oy = push_offsets[2, k]
        oz = push_offsets[3, k]
        tx = node_coords[1, source] + ox
        ty = node_coords[2, source] + oy
        tz = node_coords[3, source] + oz
        linear = tx + G * (ty + G * tz)
        p = Int(prefix[base + idx])
        win_targets[p] = Int(node_at[level_base_L + linear + 1])
        win_sources[p] = source
        win_class[p] = Int32(class_base + k)
    end
end

@kernel function ka_hier_route_compact_kernel!(route_levels, route_offsets,
        route_targets, route_sources, route_class, @Const(flags), @Const(prefix),
        @Const(node_at), @Const(node_coords), @Const(push_offsets),
        level_base_L, first_source, n_sources, first_offset, kn, L, class_base)
    idx = @index(Global)
    @inbounds if idx <= kn * n_sources && flags[idx] == Int32(1)
        kloc = (idx - 1) ÷ n_sources + 1
        s = (idx - 1) % n_sources + 1
        k = first_offset + kloc - 1
        source = first_source + s - 1
        G = 1 << L
        ox = push_offsets[1, k]
        oy = push_offsets[2, k]
        oz = push_offsets[3, k]
        tx = node_coords[1, source] + ox
        ty = node_coords[2, source] + oy
        tz = node_coords[3, source] + oz
        linear = tx + G * (ty + G * tz)
        target = Int(node_at[level_base_L + linear + 1])
        p = Int(prefix[idx])
        route_levels[p] = L
        route_offsets[1, p] = Int(ox)
        route_offsets[2, p] = Int(oy)
        route_offsets[3, p] = Int(oz)
        route_targets[p] = target
        route_sources[p] = source
        route_class[p] = Int32(class_base + k)
    end
end

#------- resident stage-group edge refresh (KA) -------#
#
# KA port of `_cuda_refresh_resident_stage_groups!`
# (src/translate_batched_cuda.jl:6120) and `_cuda_refresh_group_edges_kernel!`.
# Rebuilds the per-level M2M/L2L edge columns -- (source, target) node index
# pairs plus the spherical angles of the parent-child displacement -- from the
# refreshed grid, once per occupancy change inside `update_cuda_radix_state!`.
#
# Scope matches the CUDA function, not the host one: `_refresh_resident_stage_groups!`
# (translate_batched_resident.jl) also refills `ws.nonleaf_idx`, which is
# host-path-only storage (see the note at :3589) and untouched on device.
#
# `TF` is threaded in as a type argument rather than taken from `eltype(phis)`
# inside the kernel -- see [[reference-ka-localmem-eltype-metal]]; the group
# fields are `Any`-typed, so an in-kernel `eltype` is exactly the pattern that
# fails to resolve on Metal.
@kernel function ka_refresh_group_edges_kernel!(source_idx, target_idx, phis,
        thetas, @Const(parent_index), @Const(node_centers), first_child, n_edges,
        child_to_parent, ::Type{TF}) where {TF}
    i = @index(Global)
    @inbounds if i <= n_edges
        child = first_child + i - 1
        parent = parent_index[child]
        if child_to_parent
            dx = node_centers[1, parent] - node_centers[1, child]
            dy = node_centers[2, parent] - node_centers[2, child]
            dz = node_centers[3, parent] - node_centers[3, child]
            source_idx[i] = child
            target_idx[i] = parent
        else
            dx = node_centers[1, child] - node_centers[1, parent]
            dy = node_centers[2, child] - node_centers[2, parent]
            dz = node_centers[3, child] - node_centers[3, parent]
            source_idx[i] = parent
            target_idx[i] = child
        end
        x2y2 = dx * dx + dy * dy
        r2 = x2y2 + dz * dz
        eps2 = TF(1e-10) * TF(1e-10)
        r = sqrt(r2)
        theta = zero(TF)
        if r2 > eps2
            if x2y2 > eps2
                theta = acos(clamp(dz / r, -one(TF), one(TF)))
            else
                theta = TF(pi) * (dz < 0)
            end
        end
        phis[i] = iszero(x2y2) ? zero(TF) : atan(dy, dx)
        thetas[i] = theta
    end
end

function ka_refresh_group_edges!(group, grid, level_offsets::Vector{Int},
        child_level::Int, child_to_parent::Bool, kind::Symbol;
        workgroup=KA_AUTO_WORKGROUP)
    first_child = level_offsets[child_level + 1] + 1
    n_edges = level_offsets[child_level + 2] - level_offsets[child_level + 1]
    n_edges <= length(group.source_idx) || throw(AssertionError(
        "resident $kind group at child level $child_level exceeded its capacity"))
    group.count[] = n_edges
    n_edges > 0 || return group
    TF = eltype(group.phis)
    backend = KA.get_backend(group.phis)
    kernel = _cached_kernel(ka_refresh_group_edges_kernel!, backend, workgroup)
    kernel(group.source_idx, group.target_idx, group.phis, group.thetas,
        grid.parent_index, grid.node_centers, first_child, n_edges,
        child_to_parent, TF; ndrange=n_edges)
    return group
end

function ka_refresh_resident_stage_groups!(ws::FastMultipole.ResidentOperatorWorkspace,
        grid, level_offsets::Vector{Int}, ell::Int, first_level::Int=0;
        workgroup=KA_AUTO_WORKGROUP)
    length(ws.m2m_groups) == ell - first_level || throw(ArgumentError(
        "resident cache workspace does not match the trimmed level range"))
    for (gi, parent_level) in enumerate((ell - 1):-1:first_level)
        ka_refresh_group_edges!(ws.m2m_groups[gi], grid, level_offsets,
            parent_level + 1, true, :m2m; workgroup)
    end
    for (gi, child_level) in enumerate((first_level + 1):ell)
        ka_refresh_group_edges!(ws.l2l_groups[gi], grid, level_offsets,
            child_level, false, :l2l; workgroup)
    end
    return ws
end

#------- device source-position extraction (KA) -------#
#
# KA port of `_cuda_extract_source_positions_kernel!`
# (src/translate_batched_cuda.jl:90) and its driver
# `_radix_cache_collect_positions!` (:6612). Unlike the three helpers beside it
# in the update path -- which were backend-agnostic code merely misfiled in the
# CUDA-only include and have been moved to translate_batched_resident.jl -- this
# one is a real kernel launch and needs a port.
#
# Elementwise gather of the xyz rows plus the (system, index) attribution of each
# body into the concatenated global order: no shared memory, `KA_AUTO_WORKGROUP`.
@kernel function ka_extract_source_positions_kernel!(positions, body_system,
        body_index, @Const(source_buffer), offset, isys, nb)
    i = @index(Global)
    @inbounds if i <= nb
        global_i = offset + i
        positions[1, global_i] = source_buffer[1, i]
        positions[2, global_i] = source_buffer[2, i]
        positions[3, global_i] = source_buffer[3, i]
        body_system[global_i] = isys
        body_index[global_i] = i
    end
end

# `source_buffers` are the per-system views `_radix_cache_refresh_source_buffers!`
# returns; the return value is the total body count, as on the CUDA side.
function ka_collect_positions!(positions, body_system, body_index,
        source_buffers::Tuple; workgroup=KA_AUTO_WORKGROUP)
    offset = 0
    for isys in eachindex(source_buffers)
        buf = source_buffers[isys]
        nb = size(buf, 2)
        if nb > 0
            backend = KA.get_backend(positions)
            kernel = _cached_kernel(ka_extract_source_positions_kernel!, backend,
                workgroup)
            kernel(positions, body_system, body_index, buf, offset, isys, nb;
                ndrange=nb)
        end
        offset += nb
    end
    return offset
end

# KA port of `_cuda_hier_node_at_scatter_kernel!`
# (src/translate_batched_cuda.jl:7066) and its driver
# `_cuda_hier_refresh_occupancy!` (:7214). `node_at` is zeroed at construction on
# both backends and refilled from the resident grid every time the occupied-node
# set changes, so the window generator above cannot run off CUDA without it.
# Elementwise scatter, no shared memory: `KA_AUTO_WORKGROUP`.
@kernel function ka_hier_node_at_scatter_kernel!(node_at, @Const(node_levels),
        @Const(node_coords), @Const(level_base), n_nodes)
    i = @index(Global)
    @inbounds if i <= n_nodes
        L = node_levels[i]
        G = 1 << L
        linear = node_coords[1, i] + G * (node_coords[2, i] + G * node_coords[3, i])
        node_at[level_base[L + 1] + linear + 1] = Int32(i)
    end
end

function ka_hier_refresh_occupancy!(hctx::FastMultipole.DeviceHierarchicalM2LContext,
        grid, level_offsets::Vector{Int}; workgroup=KA_AUTO_WORKGROUP)
    copyto!(hctx.level_offsets, level_offsets)
    n_nodes = level_offsets[end]
    n_nodes <= typemax(Int32) || throw(ArgumentError(
        "device hierarchical occupancy requires flat node indices to fit Int32; " *
        "got $n_nodes occupied nodes"))
    fill!(hctx.node_at, Int32(0))
    @inbounds for level in 0:hctx.ell
        hctx.nodes_per_level[level + 1] =
            level_offsets[level + 2] - level_offsets[level + 1]
    end
    n_nodes == 0 && return hctx
    backend = KA.get_backend(hctx.node_at)
    kernel = _cached_kernel(ka_hier_node_at_scatter_kernel!, backend, workgroup)
    kernel(hctx.node_at, grid.node_levels, grid.node_coords, hctx.d_level_base,
        n_nodes; ndrange=n_nodes)
    return hctx
end

# Backend-generic mirror of `_build_cuda_hierarchical_context`
# (src/translate_batched_cuda.jl:8134). `DeviceHierarchicalM2LContext` is
# already array-type generic, so this is a pure array-type substitution: every
# `CUDA.zeros`/`CUDA.CuArray{T}` becomes a KA allocation on `backend`. It exists
# so `ka_hier_generate_window!` can be gated off CUDA; it covers the concat-plan
# configuration only (the dense CUDA plan's `source_scale`/`target_scale` come
# from `_cuda_hier_dense_scales`, which is CUDA-only and outside the KA pin).
function ka_hierarchical_context(::Type{TF}, backend, tables, class_level,
        class_offset, effective_offsets, level_class_of::Array{Int32,3},
        level_radii2, plan, ell::Int, first_m2l_level::Int, max_level_nodes::Int,
        occupancy; window_classes::Int=typemax(Int),
        dense_scales=nothing) where {TF}
    plan isa Union{FastMultipole.ResidentM2LConcatPlan,
                   FastMultipole.ResidentM2LDenseCUDAPlan} || throw(ArgumentError(
        "ka_hierarchical_context covers ResidentM2LConcatPlan and the dense " *
        "plan; got $(typeof(plan))"))
    isempty(occupancy.node_at) && throw(ArgumentError(
        "ka_hierarchical_context requires the dense per-level occupancy lookup"))
    noffsets = length(tables.push_offsets)
    K = max(min(window_classes, noffsets), 1)
    flag_capacity = max(K * max_level_nodes, 1)
    size(level_class_of) == (8, noffsets, ell + 1) || throw(ArgumentError(
        "invalid hierarchical per-level class table dimensions $(size(level_class_of))"))
    _dev(A) = KA.allocate(backend, eltype(A), size(A)...) |> d -> (copyto!(d, A); d)
    # `_radix_offsets_matrix` lives in translate_batched_cuda.jl, which is
    # `include`d only when CUDA is available -- calling it here would make this
    # builder CUDA-only at run time. Same five lines, inlined.
    _offsets_matrix(offsets) = (m = Matrix{Int32}(undef, 3, length(offsets));
        for (k, o) in enumerate(offsets); m[1, k] = Int32(o[1]);
            m[2, k] = Int32(o[2]); m[3, k] = Int32(o[3]); end; m)
    _zeros(T, n) = (z = KA.allocate(backend, T, n); fill!(z, zero(T)); z)
    # dense reads a per-level Lambda column at apply time; concat does not
    empty_scale = KA.allocate(backend, TF, 0, 0)
    src_scale, tgt_scale = if plan isa FastMultipole.ResidentM2LDenseCUDAPlan
        dense_scales === nothing && throw(ArgumentError(
            "a dense plan requires dense_scales=(source_scale, target_scale) " *
            "from `_ka_hier_dense_scales`"))
        (_dev(dense_scales[1]), _dev(dense_scales[2]))
    else
        (empty_scale, empty_scale)
    end
    return FastMultipole.DeviceHierarchicalM2LContext(
        tables, level_radii2, class_level, class_offset, effective_offsets, plan,
        K, ell, first_m2l_level, noffsets,
        copy(occupancy.level_base), zeros(Int, ell + 2),
        _zeros(Int32, length(occupancy.node_at)),
        _dev(Vector{Int}(occupancy.level_base)),
        _dev(_offsets_matrix(tables.push_offsets)),
        _dev(level_class_of),
        _dev(_offsets_matrix(tables.near_offsets)),
        _zeros(Int, 0), _zeros(Int, 0),
        _zeros(Int32, flag_capacity), _zeros(Int32, flag_capacity),
        _zeros(Int32, max(K, 1)), zeros(Int32, max(K, 1)),
        src_scale, tgt_scale,
        0, zeros(Int, ell + 1), zeros(Int, ell + 1), 0, 0, 1, 0,
        false, zeros(UInt64, 5), zeros(UInt64, ell + 1),
        0, 0, false, zeros(Int, ell + 2), zeros(Int, ell + 2),
        nothing, nothing, nothing, nothing, -1, -1,
        nothing,
    )
end

function ka_hier_generate_window!(state::FastMultipole.DeviceResidentRadixState,
        hctx::FastMultipole.DeviceHierarchicalM2LContext, route_class, L::Int,
        first_offset::Int, last_offset::Int, class_base::Int;
        workgroup=KA_AUTO_WORKGROUP)
    return ka_hier_generate_window_core!(state.route_levels, state.route_offsets,
        state.route_targets, state.route_sources, state.grid, hctx, route_class,
        L, first_offset, last_offset, class_base; workgroup)
end

function ka_hier_generate_window_core!(route_levels, route_offsets, route_targets,
        route_sources, grid, hctx::FastMultipole.DeviceHierarchicalM2LContext,
        route_class, L::Int, first_offset::Int, last_offset::Int, class_base::Int;
        workgroup=KA_AUTO_WORKGROUP)
    first_source = hctx.level_offsets[L + 1] + 1
    n_sources = hctx.level_offsets[L + 2] - hctx.level_offsets[L + 1]
    kn = last_offset - first_offset + 1
    (n_sources > 0 && kn > 0) || return 0
    used = kn * n_sources
    used <= length(hctx.route_flags) || throw(AssertionError(
        "device hierarchical window flag buffer exceeded its capacity " *
        "($(length(hctx.route_flags)) < $used); reduce window_classes"))
    backend = KA.get_backend(hctx.route_flags)
    level_base_L = hctx.level_base[L + 1]

    flags_kernel = _cached_kernel(ka_hier_route_flags_kernel!, backend, workgroup)
    flags_kernel(hctx.route_flags, hctx.node_at, grid.node_coords,
        hctx.d_push_offsets, hctx.d_class_of, level_base_L, first_source,
        n_sources, first_offset, kn, L; ndrange=used)

    # Inclusive scan over the used prefix, then the per-class cumulative counts.
    accumulate!(+, view(hctx.route_prefix, 1:used), view(hctx.route_flags, 1:used))

    cum_kernel = _cached_kernel(ka_hier_window_cum_kernel!, backend, workgroup)
    cum_kernel(hctx.window_cum, hctx.route_prefix, n_sources, kn; ndrange=kn)

    # The `kn`-entry D2H is the one unavoidable sync point per window: the
    # compact launch needs `n_routes` on the host to bounds-check the route
    # buffers, exactly as the CUDA core does.
    KA.synchronize(backend)
    copyto!(hctx.host_window_cum, 1, hctx.window_cum, 1, kn)
    n_routes = Int(hctx.host_window_cum[kn])
    n_routes == 0 && return 0
    n_routes <= length(route_targets) || throw(AssertionError(
        "device hierarchical route window exceeded capacity " *
        "$(length(route_targets)); increase window storage or reduce window_classes"))

    compact_kernel = _cached_kernel(ka_hier_route_compact_kernel!, backend, workgroup)
    compact_kernel(route_levels, route_offsets, route_targets, route_sources,
        route_class, hctx.route_flags, hctx.route_prefix, hctx.node_at,
        grid.node_coords, hctx.d_push_offsets, level_base_L, first_source,
        n_sources, first_offset, kn, L, class_base; ndrange=used)
    return n_routes
end
# Two-phase window generation for the occupancy-epoch cache: `..._count!`
# runs flags/scan/cum for one window and stashes its route total on the
# device; the caller syncs ONCE for all windows, then `..._compact!` reruns
# the (cheap) flags/scan and compacts with the count already on the host. The
# single-window core above syncs per window -- one host readback per (level,
# offset class) -- which on a moving field runs every step; measured 39% of
# the step at np=16k on Metal, where a sync costs ~200 us.
function ka_hier_generate_window_count!(grid, hctx::FastMultipole.DeviceHierarchicalM2LContext,
        L::Int, first_offset::Int, last_offset::Int, win_totals, w::Int;
        workgroup=KA_AUTO_WORKGROUP)
    first_source = hctx.level_offsets[L + 1] + 1
    n_sources = hctx.level_offsets[L + 2] - hctx.level_offsets[L + 1]
    kn = last_offset - first_offset + 1
    (n_sources > 0 && kn > 0) || return nothing
    used = kn * n_sources
    used <= length(hctx.route_flags) || throw(AssertionError(
        "device hierarchical window flag buffer exceeded its capacity " *
        "($(length(hctx.route_flags)) < $used); reduce window_classes"))
    backend = KA.get_backend(hctx.route_flags)
    level_base_L = hctx.level_base[L + 1]
    flags_kernel = _cached_kernel(ka_hier_route_flags_kernel!, backend, workgroup)
    flags_kernel(hctx.route_flags, hctx.node_at, grid.node_coords,
        hctx.d_push_offsets, hctx.d_class_of, level_base_L, first_source,
        n_sources, first_offset, kn, L; ndrange=used)
    accumulate!(+, view(hctx.route_prefix, 1:used), view(hctx.route_flags, 1:used))
    cum_kernel = _cached_kernel(ka_hier_window_cum_kernel!, backend, workgroup)
    cum_kernel(hctx.window_cum, hctx.route_prefix, n_sources, kn; ndrange=kn)
    copyto!(win_totals, w, hctx.window_cum, kn, 1)   # device -> device, no sync
    return nothing
end

function ka_hier_generate_window_compact!(route_levels, route_offsets, route_targets,
        route_sources, grid, hctx::FastMultipole.DeviceHierarchicalM2LContext,
        route_class, L::Int, first_offset::Int, last_offset::Int, class_base::Int,
        n_routes::Int; workgroup=KA_AUTO_WORKGROUP)
    n_routes == 0 && return 0
    n_routes <= length(route_targets) || throw(AssertionError(
        "device hierarchical route window exceeded capacity " *
        "$(length(route_targets)); increase window storage or reduce window_classes"))
    first_source = hctx.level_offsets[L + 1] + 1
    n_sources = hctx.level_offsets[L + 2] - hctx.level_offsets[L + 1]
    kn = last_offset - first_offset + 1
    used = kn * n_sources
    backend = KA.get_backend(hctx.route_flags)
    level_base_L = hctx.level_base[L + 1]
    flags_kernel = _cached_kernel(ka_hier_route_flags_kernel!, backend, workgroup)
    flags_kernel(hctx.route_flags, hctx.node_at, grid.node_coords,
        hctx.d_push_offsets, hctx.d_class_of, level_base_L, first_source,
        n_sources, first_offset, kn, L; ndrange=used)
    accumulate!(+, view(hctx.route_prefix, 1:used), view(hctx.route_flags, 1:used))
    compact_kernel = _cached_kernel(ka_hier_route_compact_kernel!, backend, workgroup)
    compact_kernel(route_levels, route_offsets, route_targets, route_sources,
        route_class, hctx.route_flags, hctx.route_prefix, hctx.node_at,
        grid.node_coords, hctx.d_push_offsets, level_base_L, first_source,
        n_sources, first_offset, kn, L, class_base; ndrange=used)
    return n_routes
end

"""
    ka_hierarchical_m2l!(state, hctx, ws)

KA arm of `_launch_cuda_hierarchical_m2l!`: the same `(level, offset-class
window)` loop nest, with the per-window concat apply run by KA kernels instead
of `_launch_resident_m2l_concat!`.

**Window generation is now KA too.** `ka_hier_generate_window!` (above) is the
flag/scan/compact that fills `state.route_targets`/`route_sources` for one
window. It replaced the shared `_cuda_hier_generate_window!` call, so this
driver no longer reaches any `_cuda_*` function and the arm is backend-generic.

Historical note for reading older benchmarks: until 2026-08-29 the generator
was shared-native, which made every timing comparison built on this driver an
A/B of the M2L *apply* only (the rotation/translation math) and not of the
route bookkeeping. Job 13511158 and earlier numbers were produced under that
regime and must still be read that way.

Concat plans only. The dense plan is CUDA-specific (`ResidentM2LDenseCUDAPlan`,
and only it is window-cacheable), and precomputed-y needs a per-window refresh
that has no KA port; both are outside the pin recorded in
[`ka_radix_cache_workspace`](@ref).

`DeviceHierarchicalM2LContext` is array-type generic (containers.jl:782), so
with the generator ported this function is reachable on any KA backend and is
gated on Metal.
"""
function ka_hierarchical_m2l!(state::FastMultipole.DeviceResidentRadixState{TF,B,LH},
        hctx::FastMultipole.DeviceHierarchicalM2LContext, ws) where {TF,B,LH}
    plan = hctx.apply_plan
    plan isa Union{FastMultipole.ResidentM2LConcatPlan,
                   FastMultipole.ResidentM2LDenseCUDAPlan} || throw(ArgumentError(
        "ka_hierarchical_m2l! requires a ResidentM2LConcatPlan or a dense plan " *
        "(m2l_strategy = ConcatenatedFixedZM2L or DenseTranslationM2L); " *
        "got $(typeof(plan))"))
    dense = plan isa FastMultipole.ResidentM2LDenseCUDAPlan
    # Steady state on the concat plan: the occupancy-epoch cache holds the whole
    # route stream, so there is nothing to generate and the level loop collapses
    # into a single apply (the level rides in the class, not in an argument).
    if !dense && hctx.win_valid && _ka_radix_setting(:CUDA_CACHED_WINDOWS, true)
        return ka_hierarchical_m2l_cached_concat!(state, hctx, ws)
    end
    fill!(state.locals.phi, zero(TF))
    LH && fill!(state.locals.chi, zero(TF))
    route_class = plan.route_class
    noffsets = hctx.noffsets
    K = hctx.window_classes
    total = 0
    fill!(hctx.routes_per_level, 0)
    for L in hctx.first_m2l_level:hctx.ell
        level_total = 0
        # dense classes are the unscaled push offsets, level enters through the
        # scale column only, so the level component drops out (cuda:7956)
        class_base = dense ? 0 : (L - hctx.first_m2l_level) * noffsets
        for first_offset in 1:K:noffsets
            last_offset = min(first_offset + K - 1, noffsets)
            n = ka_hier_generate_window!(state, hctx, route_class, L,
                first_offset, last_offset, class_base)
            hctx.last_window_routes = n
            state.counts.n_routes = n
            if n > 0
                if dense
                    ka_hier_refresh_dense_window!(plan, hctx, first_offset,
                        last_offset, n)
                    ka_hier_dense_apply_window!(state, ws, plan, hctx, L)
                else
                    ka_resident_m2l_concat_apply!(state.locals, state.multipoles, ws,
                        state.route_sources, state.route_targets, n)
                end
            end
            level_total += n
        end
        hctx.routes_per_level[L + 1] = level_total
        total += level_total
    end
    hctx.total_routes = total
    state.counts.n_routes = total
    return state
end

# The cached counterpart of the loop above, for the concat plan only. KA-only:
# CUDA's cached path (`_launch_cuda_hierarchical_m2l_cached!`) is dense-fused,
# because its dense GEMM reference driver needs per-window class starts. The
# concat apply needs none of that -- (class, source, target) and a count is its
# entire input -- so the cached stream can be applied in one call, with no
# route generation, no per-window prefix D2H, and no per-window sync.
#
# Gated against the uncached loop, not against CUDA: CUDA has no concat cache
# to compare with. Route order is identical either way (the cache concatenates
# the same windows in the same order), so the two arms agree to within the
# reassociation of a longer chunk sequence.
function ka_hierarchical_m2l_cached_concat!(
        state::FastMultipole.DeviceResidentRadixState{TF,B,LH},
        hctx::FastMultipole.DeviceHierarchicalM2LContext, ws) where {TF,B,LH}
    fill!(state.locals.phi, zero(TF))
    LH && fill!(state.locals.chi, zero(TF))
    n = hctx.total_routes
    state.counts.n_routes = n
    n == 0 && return state
    ka_resident_m2l_concat_apply!(state.locals, state.multipoles, ws,
        view(hctx.win_sources, 1:n), view(hctx.win_targets, 1:n), n;
        route_class=view(hctx.win_class, 1:n))
    return state
end

#------- RESIDENT OPERATOR WORKSPACE (step vi-b) -------#

"""
    ka_radix_cache_workspace(backend, TF, basis_info, ell, h0, max_cells, max_nodes,
                             route_capacity, accepted_offsets, invariant;
                             ell_axes, first_level)

Build a device-resident [`ResidentOperatorWorkspace`](@ref) on any KA backend.

This is a thin forward to `FastMultipole._radix_cache_workspace`, which is already
backend-generic: every allocation in it goes through `similar(exemplar.phi, ...)`,
`_array_like_vector(exemplar.phi, ...)` or `DegreeMajorMaps(TF, P, exemplar.phi)`,
so handing it a KA-array exemplar returns a KA-resident workspace. There is
nothing to port.

The KA path is deliberately pinned to `ConcatenatedFixedZM2L` +
`MaterializedYRotationM2L` -- the strategy already gated bit-exact on H200. That
choice is what keeps this a wrapper: `compact_cuda_factored` routes only
`DenseTranslationM2L`, `PrecomputedFactoredYM2L` and `FactoredRotationM2L`
through the CUDA-specific whole-pass setups, so with concat the builder falls
through to `ResidentM2LConcatPlan` and reaches no `_cuda_*` function at all.
Hence `compact_cuda_factored=false` below, and hence no KA port of
`_cuda_factored_whole_pass_setup!` / `_cuda_precomputed_y_whole_pass_setup!` /
`_build_cuda_dense_m2l_plan`.

Cost of that pin: the KA path forgoes the dense-fused M2L, CUDA's fastest H200
configuration. That is a reversible performance ceiling, not a correctness gap.

Graph capture is likewise not ported, and does not need to be:
`_cuda_adaptive_graph_eligible` requires a `ResidentM2LDenseCUDAPlan`, so the
concat plan is never graph-eligible *even on CUDA*. The KA substitute for the
same launch-overhead problem is the one-sync-per-driver discipline.

If this ever needs a keyword the generic builder does not already take, fix the
genericity in `src/translate_batched.jl` rather than branching here.
"""
function ka_radix_cache_workspace(backend, ::Type{TF},
        basis_info::FastMultipole.OperatorBasisInfo{B,LH}, ell::Integer, h0::TF,
        max_cells::Integer, max_nodes::Integer, route_capacity::Integer,
        accepted_offsets::Vector{SVector{3,Int}},
        invariant::FastMultipole.OperatorInvariantCache;
        ell_axes::SVector{3,Int}=SVector(Int(ell), Int(ell), Int(ell)),
        first_level::Integer=0) where {TF,B,LH}
    exemplar = _ka_flat_buffer(backend, TF, basis_info, 1)
    return FastMultipole._radix_cache_workspace(TF, basis_info, exemplar, Int(ell), h0,
        Int(max_cells), Int(max_nodes), Int(route_capacity), accepted_offsets, invariant,
        FastMultipole.ConcatenatedFixedZM2L(), FastMultipole.MaterializedYRotationM2L();
        compact_cuda_factored=false, ell_axes=ell_axes, first_level=Int(first_level))
end

#------- flat (uniform) radix route generation: KA port of
#         `_cuda_generate_radix_routes!` (src/translate_batched_cuda.jl:6174) -------#
#
# Second-to-last CUDA-only dependency of `update_cuda_radix_state!`'s uniform,
# `sfs=false` path. Five elementwise kernels (`cell_at` scatter, then
# flag/scan/compact for the accepted-offset routes and again for the rejected
# offsets' direct pairs) driven by the same chunked loop nest as the CUDA
# original, whose chunking exists so the flag/prefix buffers stay bounded by
# `class_chunk * max_cells` rather than by `naccept * n_cells`.
#
# `ctx` is a NamedTuple on the CUDA side already, so this driver is generic in
# it without a core/wrapper split: it reads `cell_at`, `class_chunk`,
# `d_accepted`, `d_rejected`, `route_flags`/`route_prefix`,
# `direct_flags`/`direct_prefix`, `host_scalar32`, `route_levels`,
# `route_offsets`, `route_targets`, `route_sources`, `direct_targets`,
# `direct_sources`; `grid` is read for `cell_keys` only.
#
# Host oracle: `build_radix_routes!` (src/interaction_list_batched.jl:966),
# which emits routes offset-class-major/cell-minor and direct pairs
# target-major/offset-minor -- exactly the two flat index decompositions below,
# so the gate compares elementwise rather than set-wise.

@kernel function ka_cell_at_scatter_kernel!(cell_at, @Const(cell_keys), n_cells, ell)
    c = @index(Global)
    @inbounds if c <= n_cells
        ix, iy, iz = ka_decode_morton_key(cell_keys[c], ell)
        cell_at[ix + 1, iy + 1, iz + 1] = Int32(c)
    end
end

@kernel function ka_route_flags_kernel!(flags, @Const(cell_at), @Const(cell_keys),
        @Const(offsets), kbase, n_cells, kn, ell)
    idx = @index(Global)
    @inbounds if idx <= kn * n_cells
        kloc = (idx - 1) ÷ n_cells + 1
        c = (idx - 1) % n_cells + 1
        k = kbase + kloc
        G = 1 << ell
        ix, iy, iz = ka_decode_morton_key(cell_keys[c], ell)
        sx = ix - offsets[1, k]
        sy = iy - offsets[2, k]
        sz = iz - offsets[3, k]
        src = Int32(0)
        if 0 <= sx < G && 0 <= sy < G && 0 <= sz < G
            src = cell_at[sx + 1, sy + 1, sz + 1]
        end
        flags[idx] = src == Int32(0) ? Int32(0) : Int32(1)
    end
end

@kernel function ka_route_compact_kernel!(route_levels, route_offsets, route_targets,
        route_sources, route_class, @Const(flags), @Const(prefix), @Const(cell_at),
        @Const(cell_keys), @Const(offsets), kbase, n_cells, kn, ell, leaf_offset, base)
    idx = @index(Global)
    @inbounds if idx <= kn * n_cells && flags[idx] == Int32(1)
        kloc = (idx - 1) ÷ n_cells + 1
        c = (idx - 1) % n_cells + 1
        k = kbase + kloc
        ix, iy, iz = ka_decode_morton_key(cell_keys[c], ell)
        sx = ix - offsets[1, k]
        sy = iy - offsets[2, k]
        sz = iz - offsets[3, k]
        src = cell_at[sx + 1, sy + 1, sz + 1]
        p = base + Int(prefix[idx])
        route_levels[p] = ell
        route_offsets[1, p] = Int(offsets[1, k])
        route_offsets[2, p] = Int(offsets[2, k])
        route_offsets[3, p] = Int(offsets[3, k])
        route_targets[p] = leaf_offset + c
        route_sources[p] = leaf_offset + Int(src)
        route_class[p] = Int32(k)
    end
end

@kernel function ka_direct_flags_kernel!(flags, @Const(cell_at), @Const(cell_keys),
        @Const(offsets), fbase, len, kn, ell)
    idx = @index(Global)
    @inbounds if idx <= len
        g = fbase + idx
        c = (g - 1) ÷ kn + 1
        k = (g - 1) % kn + 1
        G = 1 << ell
        ix, iy, iz = ka_decode_morton_key(cell_keys[c], ell)
        sx = ix - offsets[1, k]
        sy = iy - offsets[2, k]
        sz = iz - offsets[3, k]
        src = Int32(0)
        if 0 <= sx < G && 0 <= sy < G && 0 <= sz < G
            src = cell_at[sx + 1, sy + 1, sz + 1]
        end
        flags[idx] = src == Int32(0) ? Int32(0) : Int32(1)
    end
end

@kernel function ka_direct_compact_kernel!(direct_targets, direct_sources,
        @Const(flags), @Const(prefix), @Const(cell_at), @Const(cell_keys),
        @Const(offsets), fbase, len, kn, ell, base)
    idx = @index(Global)
    @inbounds if idx <= len && flags[idx] == Int32(1)
        g = fbase + idx
        c = (g - 1) ÷ kn + 1
        k = (g - 1) % kn + 1
        ix, iy, iz = ka_decode_morton_key(cell_keys[c], ell)
        sx = ix - offsets[1, k]
        sy = iy - offsets[2, k]
        sz = iz - offsets[3, k]
        src = cell_at[sx + 1, sy + 1, sz + 1]
        p = base + Int(prefix[idx])
        direct_targets[p] = c
        direct_sources[p] = Int(src)
    end
end

function ka_generate_radix_routes!(ctx, grid, n_cells::Int, leaf_offset::Int,
        ell::Int, route_class; workgroup=KA_AUTO_WORKGROUP)
    backend = KA.get_backend(ctx.cell_at)
    cell_keys = grid.cell_keys
    fill!(ctx.cell_at, Int32(0))
    if n_cells > 0
        scatter = _cached_kernel(ka_cell_at_scatter_kernel!, backend, workgroup)
        scatter(ctx.cell_at, cell_keys, n_cells, ell; ndrange=n_cells)
    end

    naccept = size(ctx.d_accepted, 2)
    n_routes = 0
    k0 = 1
    flags_kernel = _cached_kernel(ka_route_flags_kernel!, backend, workgroup)
    compact_kernel = _cached_kernel(ka_route_compact_kernel!, backend, workgroup)
    while k0 <= naccept && n_cells > 0
        kn = min(ctx.class_chunk, naccept - k0 + 1)
        used = kn * n_cells
        used <= length(ctx.route_flags) ||
            throw(AssertionError("device route flag buffer exceeded its capacity"))
        flags_kernel(ctx.route_flags, ctx.cell_at, cell_keys, ctx.d_accepted,
            k0 - 1, n_cells, kn, ell; ndrange=used)
        accumulate!(+, view(ctx.route_prefix, 1:used), view(ctx.route_flags, 1:used))
        # The one-scalar D2H per chunk is the same sync point the CUDA driver
        # takes: the compact launch needs `chunk_total` on the host to
        # bounds-check the route buffers.
        KA.synchronize(backend)
        copyto!(ctx.host_scalar32, 1, ctx.route_prefix, used, 1)
        chunk_total = Int(ctx.host_scalar32[1])
        if chunk_total > 0
            n_routes + chunk_total <= length(ctx.route_targets) ||
                throw(AssertionError("device route buffer exceeded its capacity"))
            compact_kernel(ctx.route_levels, ctx.route_offsets, ctx.route_targets,
                ctx.route_sources, route_class, ctx.route_flags, ctx.route_prefix,
                ctx.cell_at, cell_keys, ctx.d_accepted, k0 - 1, n_cells, kn, ell,
                leaf_offset, n_routes; ndrange=used)
        end
        n_routes += chunk_total
        k0 += kn
    end

    nreject = size(ctx.d_rejected, 2)
    n_direct = 0
    if n_cells > 0 && nreject > 0
        total = nreject * n_cells
        flag_capacity_direct = length(ctx.direct_flags)
        flag_capacity_direct > 0 ||
            throw(AssertionError("device direct flag buffer has zero capacity"))
        dflags_kernel = _cached_kernel(ka_direct_flags_kernel!, backend, workgroup)
        dcompact_kernel = _cached_kernel(ka_direct_compact_kernel!, backend, workgroup)
        f0 = 0
        while f0 < total
            len = min(flag_capacity_direct, total - f0)
            dflags_kernel(ctx.direct_flags, ctx.cell_at, cell_keys, ctx.d_rejected,
                f0, len, nreject, ell; ndrange=len)
            accumulate!(+, view(ctx.direct_prefix, 1:len), view(ctx.direct_flags, 1:len))
            KA.synchronize(backend)
            copyto!(ctx.host_scalar32, 1, ctx.direct_prefix, len, 1)
            chunk_total = Int(ctx.host_scalar32[1])
            if chunk_total > 0
                n_direct + chunk_total <= length(ctx.direct_targets) ||
                    throw(AssertionError("device direct pair buffer exceeded its capacity"))
                dcompact_kernel(ctx.direct_targets, ctx.direct_sources,
                    ctx.direct_flags, ctx.direct_prefix, ctx.cell_at, cell_keys,
                    ctx.d_rejected, f0, len, nreject, ell, n_direct; ndrange=len)
            end
            n_direct += chunk_total
            f0 += len
        end
    end
    KA.synchronize(backend)
    return n_routes, n_direct
end

#------- in-place grid rebuild, stage 1: keys + sort + leaf-cell compression -------#
#
# First two stages of `_cuda_update_radix_grid_in_place!`
# (src/translate_batched_cuda.jl:6591), the last CUDA-only block on the uniform,
# `sfs=false` path of `update_cuda_radix_state!`. Ported and gated stage by stage
# rather than big-bang: this covers everything through the occupied-leaf-cell
# compression, which is the natural seam -- `perm`/`invperm`/`cell_ranges` are
# functions of the body positions and always refresh, while everything after the
# compression is a pure function of the occupied cell SET and sits behind the
# occupancy-epoch check.
#
# `ka_radix_keys!` (above) already ports the key kernel for the from-scratch
# benchmark build, but deliberately drops the out-of-bounds flag. The live
# refresh loop cannot: the fixed Morton box is part of the cache's invariant
# contract, and a body leaving it must throw rather than clamp. Hence a second,
# checked kernel here, which also takes the per-axis `box_extent` (task 037
# rectangular geometry) instead of assuming the cubic `2h0`.
#
# Host oracle for the gate: `_radix_fill_body_data!`,
# `_host_radix_sort_permutation` and `_compress_radix_cells`
# (src/tree_batched.jl), which are exactly these three steps on the CPU.

@kernel function ka_radix_keys_checked_kernel!(keys, oob_flag, @Const(positions),
        x_min, box_extent, h0, ell, n)
    i = @index(Global)
    @inbounds if i <= n
        G = 1 << ell
        delta = (2 * h0) / G
        px = positions[1, i]
        py = positions[2, i]
        pz = positions[3, i]
        # benign-race flag store: any lane observing an escape sets it
        if !(x_min[1] <= px <= x_min[1] + box_extent[1] &&
             x_min[2] <= py <= x_min[2] + box_extent[2] &&
             x_min[3] <= pz <= x_min[3] + box_extent[3])
            oob_flag[1] = Int32(1)
        end
        ix = clamp(floor(Int, (px - x_min[1]) / delta), 0, G - 1)
        iy = clamp(floor(Int, (py - x_min[2]) / delta), 0, G - 1)
        iz = clamp(floor(Int, (pz - x_min[3]) / delta), 0, G - 1)
        keys[i] = ka_morton_key(ix, iy, iz, ell)
    end
end

@kernel function ka_gather_sorted_keys_kernel!(sorted_keys, @Const(keys),
        @Const(perm), n)
    i = @index(Global)
    @inbounds if i <= n
        sorted_keys[i] = keys[perm[i]]
    end
end

@kernel function ka_key_change_flags_kernel!(flags, @Const(sorted_keys), n)
    i = @index(Global)
    @inbounds if i <= n
        flags[i] = (i == 1 || sorted_keys[i] != sorted_keys[i - 1]) ? 1 : 0
    end
end

@kernel function ka_fill_cell_firsts_kernel!(cell_keys, cell_ranges,
        @Const(sorted_keys), @Const(flags), @Const(prefix), n)
    i = @index(Global)
    @inbounds if i <= n && flags[i] == 1
        icell = prefix[i]
        cell_keys[icell] = sorted_keys[i]
        cell_ranges[1, icell] = i
    end
end

@kernel function ka_fill_cell_counts_kernel!(cell_ranges, @Const(flags),
        @Const(prefix), n)
    # `cell_ranges[1, :]` must be written by a prior launch; the launch boundary
    # is the synchronization this read needs (same contract as the CUDA kernel).
    i = @index(Global)
    @inbounds if i <= n && (i == n || flags[i + 1] == 1)
        icell = prefix[i]
        first = cell_ranges[1, icell]
        cell_ranges[2, icell] = i - first + 1
    end
end

"""
    ka_radix_keys_checked!(keys, oob_flag, host_oob, positions, x_min, box_extent,
                           h0, ell; workgroup=KA_AUTO_WORKGROUP)

Backend-agnostic port of `_cuda_radix_keys_checked_kernel!` plus its host-side
out-of-bounds check. Writes the full-depth (`ell`-level) Morton key of each of
the `length(keys)` bodies and throws `ArgumentError` if any body lies outside the
fixed box `[x_min, x_min + box_extent]`. `keys` is scratch on the caller's side,
so throwing here leaves the persistent grid at its previous consistent step.
"""
function ka_radix_keys_checked!(keys, oob_flag, host_oob, positions, x_min,
        box_extent, h0, ell::Int; workgroup=KA_AUTO_WORKGROUP)
    n = length(keys)
    n == 0 && return keys
    backend = KA.get_backend(keys)
    fill!(oob_flag, Int32(0))
    kernel = _cached_kernel(ka_radix_keys_checked_kernel!, backend, workgroup)
    kernel(keys, oob_flag, positions, x_min, box_extent, h0, ell, n; ndrange=n)
    KA.synchronize(backend)
    copyto!(host_oob, oob_flag)
    if host_oob[1] != 0
        x_max = x_min .+ box_extent
        throw(ArgumentError(
            "at least one body lies outside the fixed RadixFMMCache box " *
            "[$(Tuple(x_min)), $(Tuple(x_max))]; the box is part of the cache's " *
            "invariant contract — construct a new cache (or pass explicit " *
            "bounds=(x_min, box_size) covering the trajectory)"))
    end
    return keys
end

#------- bounded-key counting sort: KA port of the CUDA Stage 6 fast path -------#
#
# Port of `_cuda_counting_sort_into!` and its three kernels
# (src/translate_batched_cuda.jl:183-236). The cache's fixed Morton depth bounds
# keys to `0:2^(3ell)-1`, so a histogram over the whole key domain plus one scan
# and an atomic-cursor scatter replaces the comparison sort.
#
# This path is deliberately UNSTABLE, exactly as CUDA's is: the scatter claims
# its slot with an atomic cursor, so bodies sharing a cell land in an order that
# varies between otherwise identical runs. That is the CUDA behavior being
# matched, and it is why this sits behind the same runtime gate rather than
# simply replacing `sortperm!`. Two consequences, both inherited from CUDA:
#
#   * within-cell body order is not reproducible run to run, so neither is the
#     summation order of the same-cell nearfield atomics -- identical inputs
#     move in the last bits between runs;
#   * `perm` can no longer be compared elementwise against the stable host sort.
#     Everything downstream still is exact: `cell_keys`, `cell_ranges`, cell
#     centers and the whole node table are pure functions of the occupied-cell
#     SET, not of within-cell ordering.
#
# The gate mirrors CUDA's two conditions (`_cuda_counting_sort_ready`): the
# setting must be on for this `ell`, AND the histogram actually handed in must
# span the key domain -- flipping the knob on after construction would otherwise
# drive `@inbounds` atomics through a length-1 array. Falling back is always safe.

@kernel function ka_counting_histogram_kernel!(histogram, @Const(keys), n)
    i = @index(Global)
    @inbounds if i <= n
        KA.@atomic histogram[Int(keys[i]) + 1] += Int32(1)
    end
end

@kernel function ka_counting_cursor_kernel!(cursor, @Const(prefix), n)
    k = @index(Global)
    @inbounds if k <= n
        cursor[k] = k == 1 ? Int32(0) : prefix[k - 1]
    end
end

@kernel function ka_counting_scatter_kernel!(perm, sorted_keys, cursor, @Const(keys), n)
    i = @index(Global)
    @inbounds if i <= n
        key = keys[i]
        # CUDA calls `atomic_add!`, which returns the OLD value, and takes
        # `old + 1` as the 1-based slot. KA's `@atomic x += v` returns the NEW
        # value, which is that same slot -- verified on Metal, not assumed.
        slot = KA.@atomic cursor[Int(key) + 1] += Int32(1)
        perm[Int(slot)] = i
        sorted_keys[Int(slot)] = key
    end
end

@inline ka_counting_sort_enabled(ell::Int) =
    _ka_radix_setting(:RADIX_CUDA_COUNTING_SORT, true) &&
        ell <= _ka_radix_setting(:RADIX_CUDA_COUNTING_SORT_MAX_ELL, 6)

@inline ka_counting_sort_ready(histogram, ell::Int) =
    histogram !== nothing && ell >= 0 && ka_counting_sort_enabled(ell) &&
        length(histogram) == 1 << (3 * ell)

"""
    ka_counting_sort_into!(perm, sorted_keys, keys, histogram, prefix, cursor;
                           workgroup=KA_AUTO_WORKGROUP)

Backend-agnostic port of `_cuda_counting_sort_into!`: histogram the bounded
Morton keys, inclusive-scan the histogram, shift it into an exclusive cursor,
then scatter each body into the slot its atomic cursor claims. `histogram`,
`prefix` and `cursor` must each span the full `2^(3ell)` key domain. Unstable by
construction -- see the note above.
"""
function ka_counting_sort_into!(perm, sorted_keys, keys, histogram, prefix, cursor;
        workgroup=KA_AUTO_WORKGROUP)
    n = length(keys)
    n == 0 && return perm
    backend = KA.get_backend(keys)
    fill!(histogram, Int32(0))
    hk = _cached_kernel(ka_counting_histogram_kernel!, backend, workgroup)
    hk(histogram, keys, n; ndrange=n)
    accumulate!(+, prefix, histogram)
    nd = length(histogram)
    ck = _cached_kernel(ka_counting_cursor_kernel!, backend, workgroup)
    ck(cursor, prefix, nd; ndrange=nd)
    sc = _cached_kernel(ka_counting_scatter_kernel!, backend, workgroup)
    sc(perm, sorted_keys, cursor, keys, n; ndrange=n)
    return perm
end

"""
    ka_radix_sort_bodies!(perm, sorted_keys, invperm, keys; workgroup=KA_AUTO_WORKGROUP,
                          ell=-1, histogram=nothing, prefix=nothing, cursor=nothing)

Sort the bodies by Morton key: `perm` receives the sorting permutation,
`sorted_keys` the gathered keys, `invperm` the scatter inverse. Port of the
`_cuda_sortperm_into!` / `_cuda_gather_sorted_keys_kernel!` /
`_cuda_fill_invperm_kernel!` triple, and -- when `ell` and the counting buffers
are supplied and the gate passes -- of CUDA's bounded counting-sort fast path
too, branching exactly where `_cuda_update_radix_grid_in_place!` branches.

Callers that omit the counting buffers (the isolated correctness suites) keep
the stable `sortperm!` path, which is what makes an elementwise `perm`
comparison against the host sort meaningful for them.
"""
function ka_radix_sort_bodies!(perm, sorted_keys, invperm, keys;
        workgroup=KA_AUTO_WORKGROUP, ell::Int=-1,
        histogram=nothing, prefix=nothing, cursor=nothing)
    n = length(keys)
    n == 0 && return perm
    backend = KA.get_backend(keys)
    if ka_counting_sort_ready(histogram, ell)
        ka_counting_sort_into!(perm, sorted_keys, keys, histogram, prefix, cursor;
            workgroup)
    else
        sortperm!(perm, keys)
        gather = _cached_kernel(ka_gather_sorted_keys_kernel!, backend, workgroup)
        gather(sorted_keys, keys, perm, n; ndrange=n)
    end
    ka_fill_invperm!(invperm, perm; workgroup)
    KA.synchronize(backend)
    return perm
end

"""
    ka_radix_compress_cells!(cell_keys, cell_ranges, sorted_keys, flags, prefix,
                             host_scalar; workgroup=KA_AUTO_WORKGROUP)

Compress the sorted body keys into occupied leaf cells, writing `cell_keys` and
`cell_ranges` (row 1 = first sorted body index, row 2 = body count) and
returning `n_cells`. Port of the `_cuda_key_change_flags_kernel!` /
`accumulate!` / `_cuda_fill_cell_firsts_kernel!` /
`_cuda_fill_cell_counts_kernel!` block. `flags`/`prefix` are caller-owned
`1:n` scratch views; `host_scalar` is a 1-element host vector for the count
download, the single unavoidable sync point (the caller needs `n_cells` to
bounds-check against the cache's cell capacity).
"""
function ka_radix_compress_cells!(cell_keys, cell_ranges, sorted_keys, flags,
        prefix, host_scalar; workgroup=KA_AUTO_WORKGROUP)
    n = length(sorted_keys)
    n == 0 && return 0
    backend = KA.get_backend(sorted_keys)
    flagk = _cached_kernel(ka_key_change_flags_kernel!, backend, workgroup)
    flagk(flags, sorted_keys, n; ndrange=n)
    accumulate!(+, prefix, flags)
    KA.synchronize(backend)
    copyto!(host_scalar, 1, prefix, n, 1)
    n_cells = Int(host_scalar[1])
    firstsk = _cached_kernel(ka_fill_cell_firsts_kernel!, backend, workgroup)
    firstsk(cell_keys, cell_ranges, sorted_keys, flags, prefix, n; ndrange=n)
    countsk = _cached_kernel(ka_fill_cell_counts_kernel!, backend, workgroup)
    countsk(cell_ranges, flags, prefix, n; ndrange=n)
    KA.synchronize(backend)
    return n_cells
end


#------- in-place grid rebuild, stage 2: occupancy-epoch check + cell centers -------#
#
# Third stage of `_cuda_update_radix_grid_in_place!`
# (src/translate_batched_cuda.jl:6591), directly after the leaf-cell compression
# ported in stage 1. This is the seam the epoch check defines: everything from
# here on -- cell centers, per-level unique node keys, node geometry/parent/child
# topology, leaf_to_node -- is a pure function of the occupied leaf-cell SET
# inside the cache's fixed Morton box, so when the sorted unique keys match the
# previous step's snapshot exactly the whole node-metadata rebuild is skipped
# and the persistent arrays stay valid. The compare costs one kernel plus one
# 4-byte D2H, replacing ~40 launches, several device scans and two blocking
# downloads on the steady occupancy-static step.
#
# Host oracle for the gate: the cell-center loop of `_refresh_radix_grid!`
# (src/tree_batched.jl:485), plus `morton_decode` for the integer coords, which
# the host grid does not store (`ctx.cell_coords` is device-side only).

@kernel function ka_keys_differ_kernel!(flag, @Const(keys), @Const(snapshot), n)
    i = @index(Global)
    @inbounds if i <= n && keys[i] != snapshot[i]
        # benign-race flag store: any lane observing a difference sets it
        flag[1] = Int32(1)
    end
end

@kernel function ka_cell_centers_kernel!(centers, coords, @Const(cell_keys),
        x_min, h0, ell, n_cells)
    icell = @index(Global)
    @inbounds if icell <= n_cells
        TF = eltype(centers)
        delta = (2 * h0) / (1 << ell)
        ix, iy, iz = ka_decode_morton_key(cell_keys[icell], ell)
        coords[1, icell] = ix
        coords[2, icell] = iy
        coords[3, icell] = iz
        centers[1, icell] = x_min[1] + delta * (TF(ix) + TF(0.5))
        centers[2, icell] = x_min[2] + delta * (TF(iy) + TF(0.5))
        centers[3, icell] = x_min[3] + delta * (TF(iz) + TF(0.5))
    end
end

"""
    ka_radix_occupancy_changed!(flag, host_flag, cell_keys, snapshot, n_cells;
                                workgroup=KA_AUTO_WORKGROUP)

Backend-agnostic port of the `_cuda_keys_differ_kernel!` compare: returns `true`
when the first `n_cells` occupied leaf-cell keys differ anywhere from
`snapshot`. Both key arrays are ascending and of equal length by the caller's
own `n`/`n_cells` precheck, so an elementwise compare is a set compare. `flag`
is a 1-element device buffer, `host_flag` its 1-element host mirror.
"""
function ka_radix_occupancy_changed!(flag, host_flag, cell_keys, snapshot,
        n_cells::Int; workgroup=KA_AUTO_WORKGROUP)
    n_cells == 0 && return false
    backend = KA.get_backend(cell_keys)
    fill!(flag, Int32(0))
    kernel = _cached_kernel(ka_keys_differ_kernel!, backend, workgroup)
    kernel(flag, cell_keys, snapshot, n_cells; ndrange=n_cells)
    KA.synchronize(backend)
    copyto!(host_flag, flag)
    return host_flag[1] != Int32(0)
end

"""
    ka_radix_cell_centers!(centers, coords, cell_keys, x_min, h0, ell, n_cells;
                           workgroup=KA_AUTO_WORKGROUP)

Port of `_cuda_cell_centers_kernel!`: decode each occupied leaf cell's Morton
key into its integer grid coordinate (`coords`) and the cell's physical center
(`centers`). `x_min` is a plain host scalar triple (an `SVector{3,TF}`), `h0`
the box half-width; the `0.5` is `TF`-typed, since a bare literal would promote
the whole expression to `Float64` and fail to compile on Metal.
"""
function ka_radix_cell_centers!(centers, coords, cell_keys, x_min, h0, ell::Int,
        n_cells::Int; workgroup=KA_AUTO_WORKGROUP)
    n_cells == 0 && return centers
    backend = KA.get_backend(cell_keys)
    kernel = _cached_kernel(ka_cell_centers_kernel!, backend, workgroup)
    kernel(centers, coords, cell_keys, x_min, h0, ell, n_cells; ndrange=n_cells)
    KA.synchronize(backend)
    return centers
end



#------- in-place grid rebuild, stage 3: per-level unique node keys -------#
#
# Fourth stage of `_cuda_update_radix_grid_in_place!`
# (src/translate_batched_cuda.jl:6591), the first block behind the stage-2
# occupancy-epoch check. `cell_keys` is ascending and a right shift is monotone,
# so each level's ancestor keys are *already sorted* -- no per-level sort is
# needed, and the same flag/scan/compact triple that compressed bodies into
# leaf cells in stage 1 compresses each level's ancestor keys into that level's
# unique nodes. Only the counts round-trip to the host, because the level
# offsets are a running host-side prefix that the node-capacity check and the
# stage-4 launch geometry (`max_count`) both read.
#
# Levels below the cache root are trimmed (task 037 stage 3): never keyed, never
# built. `ka_gather_level_counts_kernel!` still reads every column, including
# the unfilled prefix columns of the trimmed levels, exactly as the CUDA kernel
# does -- the host loop zeroes those counts immediately after the download, so
# the garbage never reaches an offset.
#
# Host oracle for the gate: `_refresh_radix_nodes!` (src/tree_batched.jl), whose
# count and fill loops are what `_radix_grid` runs on the CPU.

@kernel function ka_leaf_ancestor_keys_kernel!(ancestor_keys, @Const(cell_keys),
        leaf_level, level, n_cells)
    i = @index(Global)
    @inbounds if i <= n_cells
        ancestor_keys[i] = cell_keys[i] >> (3 * (leaf_level - level))
    end
end

@kernel function ka_gather_level_counts_kernel!(level_counts, @Const(level_prefix),
        n_cells, n_levels)
    l = @index(Global)
    @inbounds if l <= n_levels
        level_counts[l] = n_cells == 0 ? 0 : level_prefix[n_cells, l]
    end
end

@kernel function ka_fill_unique_keys_kernel!(dest, @Const(sorted_keys), @Const(flags),
        @Const(prefix), offset, n)
    i = @index(Global)
    @inbounds if i <= n && flags[i] == 1
        dest[offset + prefix[i]] = sorted_keys[i]
    end
end

"""
    ka_radix_level_nodes!(node_keys, level_offsets, level_keys, level_flags,
                          level_prefix, level_counts, host_level_counts,
                          d_level_offsets, cell_keys, n_cells, ell, first_level,
                          max_nodes; workgroup=KA_AUTO_WORKGROUP)

Backend-agnostic port of the per-level unique-node block of
`_cuda_update_radix_grid_in_place!`: for each active level `first_level:ell`,
shift the occupied leaf-cell keys to that level's ancestor keys and compress the
runs into `node_keys`, level-major. Fills the host `level_offsets` (length
`ell + 2`, trimmed levels left at 0), mirrors it to `d_level_offsets`, and
returns `(n_nodes, max_count)` -- the node total for the caller's capacity
bookkeeping and the largest per-level node count, which is the x-extent of the
stage-4 2D launches.

`level_keys`/`level_flags`/`level_prefix` are `max_cells x (ell + 1)` device
scratch matrices, one column per level; `level_counts` is an `ell + 1` device
vector and `host_level_counts` its host mirror. Throws `AssertionError` if the
node total exceeds `max_nodes`.
"""
function ka_radix_level_nodes!(node_keys, level_offsets, level_keys, level_flags,
        level_prefix, level_counts, host_level_counts, d_level_offsets,
        cell_keys, n_cells::Int, ell::Int, first_level::Int, max_nodes::Int;
        workgroup=KA_AUTO_WORKGROUP)
    backend = KA.get_backend(cell_keys)
    n_levels = length(level_counts)
    if n_cells > 0
        ancestor = _cached_kernel(ka_leaf_ancestor_keys_kernel!, backend, workgroup)
        flagk = _cached_kernel(ka_key_change_flags_kernel!, backend, workgroup)
        for level in first_level:ell
            col = level + 1
            lk = view(level_keys, 1:n_cells, col)
            lf = view(level_flags, 1:n_cells, col)
            ancestor(lk, cell_keys, ell, level, n_cells; ndrange=n_cells)
            flagk(lf, lk, n_cells; ndrange=n_cells)
            accumulate!(+, view(level_prefix, 1:n_cells, col), lf)
        end
    end
    gather = _cached_kernel(ka_gather_level_counts_kernel!, backend, workgroup)
    gather(level_counts, level_prefix, n_cells, n_levels; ndrange=n_levels)
    KA.synchronize(backend)
    copyto!(host_level_counts, level_counts)

    level_offsets[1] = 0
    for level in 0:(first_level - 1)
        # trimmed levels: the gathered counts read unfilled prefix columns
        host_level_counts[level + 1] = 0
        level_offsets[level + 2] = 0
    end
    for level in first_level:ell
        level_offsets[level + 2] = level_offsets[level + 1] + host_level_counts[level + 1]
    end
    n_nodes = level_offsets[end]
    n_nodes <= max_nodes ||
        throw(AssertionError("device radix grid exceeded the cache node capacity"))

    if n_cells > 0
        uniquek = _cached_kernel(ka_fill_unique_keys_kernel!, backend, workgroup)
        for level in first_level:ell
            col = level + 1
            uniquek(node_keys, view(level_keys, 1:n_cells, col),
                view(level_flags, 1:n_cells, col),
                view(level_prefix, 1:n_cells, col), level_offsets[col], n_cells;
                ndrange=n_cells)
        end
    end
    copyto!(d_level_offsets, level_offsets)
    KA.synchronize(backend)
    return n_nodes, maximum(host_level_counts; init=0)
end



#------- in-place grid rebuild, stage 4: node geometry, parents, children -------#
#
# Final stage of `_cuda_update_radix_grid_in_place!`
# (src/translate_batched_cuda.jl:6591): with the level-major `node_keys` and the
# `level_offsets` prefix in hand from stage 3, fill each node's level/coord/
# center, its parent index, and its contiguous child range, then map each leaf
# cell to its deepest-level node.
#
# The three CUDA kernels launch 2D, `blockIdx().y` carrying the level, so every
# active level runs concurrently and the per-level launch loop collapses to one
# launch. The KA ports keep that -- one launch over all levels -- but **flatten**
# the grid to 1D and decode the level from the flat index, the same idiom the
# route generator uses (`ka_route_flags_kernel!`). A 2D `ndrange` would have to
# carry a matching 2D workgroup size, which the per-backend scalar workgroup
# policy above does not produce; flattening keeps one tunable launch geometry
# for both backends. The x-extent is `max_count` from stage 3, so the flat range
# is `max_count * n_levels` with the ragged tail masked per level, exactly as
# the CUDA `node > stop` guard does.
#
# Parents and children resolve by binary search into the adjacent level's node
# block rather than by the host builder's sorted-merge walk: both blocks are
# ascending in key, and a search is what makes the levels independent enough to
# launch together. Roots (`level == min_level`) get `parent_index = 0` and the
# deepest level gets an empty child range.
#
# Host oracle for the gate: `_refresh_radix_nodes!` (src/tree_batched.jl) again
# -- stage 3 compared the part of its output stage 3 owns, this stage compares
# the rest.

@kernel function ka_node_geometry_levels_kernel!(node_levels, node_coords, node_centers,
        @Const(node_keys), @Const(level_offsets), x_min, h0, min_level, max_count)
    idx = @index(Global)
    @inbounds begin
        level = min_level + (idx - 1) ÷ max_count
        node = level_offsets[level + 1] + (idx - 1) % max_count + 1
        if node <= level_offsets[level + 2]
            TF = eltype(node_centers)
            delta = (2 * h0) / (1 << level)
            ix, iy, iz = ka_decode_morton_key(node_keys[node], level)
            node_levels[node] = level
            node_coords[1, node] = ix
            node_coords[2, node] = iy
            node_coords[3, node] = iz
            node_centers[1, node] = x_min[1] + delta * (TF(ix) + TF(0.5))
            node_centers[2, node] = x_min[2] + delta * (TF(iy) + TF(0.5))
            node_centers[3, node] = x_min[3] + delta * (TF(iz) + TF(0.5))
        end
    end
end

@kernel function ka_parent_index_levels_kernel!(parent_index, @Const(node_keys),
        @Const(level_offsets), min_level, max_count)
    idx = @index(Global)
    @inbounds begin
        level = min_level + (idx - 1) ÷ max_count
        node = level_offsets[level + 1] + (idx - 1) % max_count + 1
        if node <= level_offsets[level + 2]
            if level == min_level
                parent_index[node] = 0
            else
                parent_key = node_keys[node] >> 3
                parent_first = level_offsets[level] + 1
                parent_stop = level_offsets[level + 1]
                parent = ka_lower_bound(node_keys, parent_first, parent_stop, parent_key)
                parent_index[node] =
                    (parent <= parent_stop && node_keys[parent] == parent_key) ? parent : 0
            end
        end
    end
end

@kernel function ka_child_ranges_levels_kernel!(child_ranges, @Const(node_keys),
        @Const(level_offsets), max_level, min_level, max_count)
    idx = @index(Global)
    @inbounds begin
        level = min_level + (idx - 1) ÷ max_count
        node = level_offsets[level + 1] + (idx - 1) % max_count + 1
        if node <= level_offsets[level + 2]
            if level == max_level
                child_ranges[1, node] = 0
                child_ranges[2, node] = 0
            else
                child_first = level_offsets[level + 2] + 1
                child_stop = level_offsets[level + 3]
                lo_key = node_keys[node] << 3
                hi_key = lo_key + UInt64(7)
                lo = ka_lower_bound(node_keys, child_first, child_stop, lo_key)
                hi = ka_upper_bound(node_keys, child_first, child_stop, hi_key)
                count = hi - lo
                child_ranges[1, node] = count > 0 ? lo : 0
                child_ranges[2, node] = count
            end
        end
    end
end

@kernel function ka_fill_leaf_to_node_kernel!(leaf_to_node, leaf_offset, n_cells)
    i = @index(Global)
    @inbounds if i <= n_cells
        leaf_to_node[i] = leaf_offset + i
    end
end

"""
    ka_radix_node_topology!(node_levels, node_coords, node_centers, parent_index,
                            child_ranges, leaf_to_node, node_keys, d_level_offsets,
                            level_offsets, x_min, h0, n_cells, ell, first_level,
                            max_count; workgroup=KA_AUTO_WORKGROUP)

Backend-agnostic port of the node geometry / parent / child-range trio plus
`_cuda_fill_leaf_to_node_kernel!` -- the last block of
`_cuda_update_radix_grid_in_place!`. Consumes stage 3's level-major `node_keys`,
its device-side `d_level_offsets` mirror and the host `level_offsets` vector,
and the `max_count` it returned (the per-level x-extent of the flattened
launches). Levels below `first_level` are trimmed and never touched; nodes at
`first_level` are roots with `parent_index = 0`.
"""
function ka_radix_node_topology!(node_levels, node_coords, node_centers,
        parent_index, child_ranges, leaf_to_node, node_keys, d_level_offsets,
        level_offsets, x_min, h0, n_cells::Int, ell::Int, first_level::Int,
        max_count::Int; workgroup=KA_AUTO_WORKGROUP)
    backend = KA.get_backend(node_keys)
    if max_count > 0
        n_levels = ell - first_level + 1
        flat = max_count * n_levels
        geom = _cached_kernel(ka_node_geometry_levels_kernel!, backend, workgroup)
        geom(node_levels, node_coords, node_centers, node_keys, d_level_offsets,
            x_min, h0, first_level, max_count; ndrange=flat)
        par = _cached_kernel(ka_parent_index_levels_kernel!, backend, workgroup)
        par(parent_index, node_keys, d_level_offsets, first_level, max_count;
            ndrange=flat)
        chi = _cached_kernel(ka_child_ranges_levels_kernel!, backend, workgroup)
        chi(child_ranges, node_keys, d_level_offsets, ell, first_level, max_count;
            ndrange=flat)
    end
    if n_cells > 0
        l2n = _cached_kernel(ka_fill_leaf_to_node_kernel!, backend, workgroup)
        l2n(leaf_to_node, level_offsets[ell + 1], n_cells; ndrange=n_cells)
    end
    KA.synchronize(backend)
    return leaf_to_node
end



#------- output finalization -------#
#
# Backend-agnostic port of `finalize_cuda_radix_output!`
# (src/translate_batched_cuda.jl:5686): de-permute the lifecycle's sorted-order
# `state.output` back into each target system's own body order and hand it to
# `buffer_to_target!`.
#
# Only one of the two branches contains anything device-specific. The
# host-resident branch is already generic -- a prefix `copyto!` into the pinned
# staging, then `_copy_radix_output_to_host_target_buffer!`, both of which live
# in translate_batched_resident.jl and never mention CUDA -- so it is reproduced
# verbatim, including the download-once-per-call sharing of `host_output` across
# systems and the `influence_downloads` counter bump. The device-resident branch
# needs the scatter kernel, which is what this ports.
#
# Row layout is the switch's, not the output's: `scalar_potential_index`,
# `gradient_range` and `hessian_range` decide where each of the output's 1 / 2:4
# / 5:13 rows lands, and a switch asking for hessian rows from a 4-row output
# throws rather than reading past the end.

@kernel function ka_scatter_output_to_target_buffer_kernel!(target_buffer, @Const(output),
        @Const(perm), @Const(body_system), @Const(body_index), isys, scalar_row,
        gradient_start, gradient_stop, hessian_start, hessian_stop, n_bodies)
    sorted_i = @index(Global)
    @inbounds if sorted_i <= n_bodies
        global_i = perm[sorted_i]
        if body_system[global_i] == isys
            ibody = body_index[global_i]
            if scalar_row > 0
                target_buffer[scalar_row, ibody] = output[1, sorted_i]
            end
            if gradient_start <= gradient_stop
                target_buffer[gradient_start, ibody] = output[2, sorted_i]
                target_buffer[gradient_start + 1, ibody] = output[3, sorted_i]
                target_buffer[gradient_start + 2, ibody] = output[4, sorted_i]
            end
            if hessian_start <= hessian_stop
                for k in 0:8
                    target_buffer[hessian_start + k, ibody] = output[5 + k, sorted_i]
                end
            end
        end
    end
end

"""
    ka_scatter_output_to_target_buffer!(target_buffer, output, body_perm,
        body_system_ids, body_indices, isys, derivatives_switch, n_bodies;
        workgroup=KA_AUTO_WORKGROUP)

Port of `_copy_radix_output_to_device_target_buffer!`: zero `target_buffer` and
scatter the sorted-order `output` columns belonging to system `isys` into it at
the rows the derivatives switch selects.
"""
function ka_scatter_output_to_target_buffer!(target_buffer, output, body_perm,
        body_system_ids, body_indices, isys::Integer, derivatives_switch,
        n_bodies::Integer=size(output, 2); workgroup=KA_AUTO_WORKGROUP)
    fill!(target_buffer, zero(eltype(target_buffer)))
    hrange = FastMultipole.hessian_range(derivatives_switch)
    isempty(hrange) || size(output, 1) >= 13 ||
        throw(ArgumentError("hessian output requested but the radix output " *
            "carries potential + gradient only; construct RadixFMMCache(...; hessian=true)"))
    grange = FastMultipole.gradient_range(derivatives_switch)
    n_bodies == 0 && return target_buffer
    backend = KA.get_backend(output)
    kernel = _cached_kernel(ka_scatter_output_to_target_buffer_kernel!, backend, workgroup)
    kernel(target_buffer, output, body_perm, body_system_ids, body_indices, isys,
        FastMultipole.scalar_potential_index(derivatives_switch),
        isempty(grange) ? 1 : first(grange), isempty(grange) ? 0 : last(grange),
        isempty(hrange) ? 1 : first(hrange), isempty(hrange) ? 0 : last(hrange),
        n_bodies; ndrange=n_bodies)
    KA.synchronize(backend)
    return target_buffer
end

# Per-system cached device scatter buffer, the generic form of
# `_cuda_cached_target_buffer`: allocated undef once per (rows, n_bodies) layout
# and reused, since the scatter zero-fills it anyway.
function _ka_cached_target_buffer(cache, backend, isys::Integer, ::Type{TF},
        rows::Integer, nb::Integer) where TF
    cache === nothing && return KA.allocate(backend, TF, rows, nb)
    buf = get(cache, isys, nothing)
    if !(buf isa AbstractMatrix{TF}) || size(buf) != (rows, nb)
        buf = KA.allocate(backend, TF, rows, nb)
        cache[isys] = buf
    end
    return buf
end

"""
    ka_finalize_radix_output!(state, target_systems; derivatives_switches,
        host_output_staging, target_buffers, device_target_buffers)

Backend-agnostic `finalize_cuda_radix_output!`. Scatters `state.output` back
into the target systems, downloading it once per call into
`host_output_staging` (the valid column prefix only) when any target is host
resident, and going through `ka_scatter_output_to_target_buffer!` for
device-resident ones.
"""
function ka_finalize_radix_output!(state, target_systems;
        derivatives_switches=nothing, host_output_staging=nothing,
        target_buffers=nothing, device_target_buffers=nothing)
    TF = eltype(state.output)
    systems = FastMultipole.to_tuple(target_systems)
    switches = derivatives_switches === nothing ?
        FastMultipole.to_tuple(FastMultipole.DerivativesSwitch(true, true, false, systems)) :
        FastMultipole.to_tuple(derivatives_switches)
    length(systems) == length(switches) ||
        throw(ArgumentError("target systems and derivatives switches must have the same length"))
    backend = KA.get_backend(state.output)

    host_output = nothing
    for (isys, target_system, switch) in zip(eachindex(systems), systems, switches)
        if FastMultipole.residency(target_system) isa FastMultipole.DeviceResident
            target_buffer = _ka_cached_target_buffer(device_target_buffers, backend,
                isys, TF, FastMultipole.target_buffer_rows(switch),
                FastMultipole.get_n_bodies(target_system))
            ka_scatter_output_to_target_buffer!(target_buffer, state.output,
                state.body_perm, state.body_system_ids, state.body_indices, isys,
                switch, state.counts.n_bodies)
            FastMultipole.buffer_to_target!(target_system, target_buffer, switch,
                1:FastMultipole.get_n_bodies(target_system))
        else
            if host_output === nothing
                if host_output_staging === nothing
                    host_output = Array(state.output)
                else
                    # recurring path: download only the valid column prefix into
                    # the preallocated (pinned) staging
                    nb = state.counts.n_bodies
                    KA.synchronize(backend)
                    copyto!(host_output_staging, 1, state.output, 1,
                        size(state.output, 1) * nb)
                    host_output = host_output_staging
                end
                state.counters.influence_downloads += 1
            end
            target_buffer = target_buffers === nothing ?
                FastMultipole.allocate_target_buffer(TF, target_system, switch) :
                target_buffers[isys]
            FastMultipole._copy_radix_output_to_host_target_buffer!(
                target_buffer, host_output, state.host_body_perm,
                state.host_body_system_ids, state.host_body_indices, isys, switch,
                state.counts.n_bodies,
            )
            FastMultipole.buffer_to_target!(target_system, target_buffer, switch,
                1:FastMultipole.get_n_bodies(target_system))
        end
    end
    return target_systems
end

#------- stage 19: within-cell sub-Morton nearfield subsort -------#
#
# Port of `_cuda_nearfield_subsort!` (src/translate_batched_cuda.jl:8209) and
# its two kernels. Mechanism (a) of 032a stage C: compose a within-cell
# sub-Morton ordering into `grid.perm` after the sort and before body packing,
# so consecutive sorted bodies -- adjacent lanes in the nearfield kernel -- span
# a compact spatial sub-block of their cell.
#
# It is locality only: no cell key, cell range or node changes, and cells larger
# than the shared-memory sort capacity keep their unspecified order. But it does
# change the ORDER same-cell contributions are summed in, so it is the one
# post-tree stage whose absence moves results. That is why it is ported rather
# than left as a perf variant: with it, the KA and CUDA arms sum in the same
# order for every cell the mechanism covers.
#
# DEVIATION (launch shape). CUDA runs one block per cell with a grid-stride
# outer loop bounded at 8192 blocks; KA runs the same loop with `@index(Group)`
# and an explicit group count, since KA has no `gridDim()`. The sort itself --
# odd-even transposition in workgroup-local memory, capacity 1024 -- is
# statement for statement.

const KA_SUBSORT_CAPACITY = 1024

@kernel function ka_subsort_keys_kernel!(subsort_keys, @Const(positions), @Const(perm),
        x_min, h0, ell, sub, n)
    p = @index(Global)
    @inbounds if p <= n
        b = perm[p]
        Gs = 1 << (ell + sub)
        m = Int32((1 << sub) - 1)
        T = eltype(positions)
        delta = (2 * h0) / T(Gs)
        cx = min(max(unsafe_trunc(Int32, (positions[1, b] - x_min[1]) / delta),
            Int32(0)), Int32(Gs - 1)) & m
        cy = min(max(unsafe_trunc(Int32, (positions[2, b] - x_min[2]) / delta),
            Int32(0)), Int32(Gs - 1)) & m
        cz = min(max(unsafe_trunc(Int32, (positions[3, b] - x_min[3]) / delta),
            Int32(0)), Int32(Gs - 1)) & m
        key = UInt32(0)
        bit = 0
        while bit < sub
            key |= (UInt32((cx >> bit) & Int32(1)) << (3 * bit))
            key |= (UInt32((cy >> bit) & Int32(1)) << (3 * bit + 1))
            key |= (UInt32((cz >> bit) & Int32(1)) << (3 * bit + 2))
            bit += 1
        end
        subsort_keys[p] = key
    end
end

# Workgroup-per-cell odd-even transposition sort of the perm segment by sub-key
# in local memory. The `1 < cnt <= capacity` condition is uniform across the
# group, so the barriers are safe.
@kernel function ka_subsort_cell_sort_kernel!(perm, subsort_keys, @Const(cell_ranges),
        n_cells, n_groups, ::Type{TP}, ::Val{CAP}, ::Val{WG}) where {TP,CAP,WG}
    keys_sh = @localmem UInt32 CAP
    perm_sh = @localmem TP CAP
    cell = @index(Group)
    t = @index(Local)
    @inbounds while cell <= n_cells
        first = cell_ranges[1, cell]
        cnt = cell_ranges[2, cell]
        if 1 < cnt <= CAP
            idx = t
            while idx <= cnt
                keys_sh[idx] = subsort_keys[first + idx - 1]
                perm_sh[idx] = perm[first + idx - 1]
                idx += WG
            end
            @synchronize
            phase = 0
            while phase < cnt
                base = 1 + (phase & 1)
                idx = base + 2 * (t - 1)
                while idx <= cnt - 1
                    ka = keys_sh[idx]
                    kb = keys_sh[idx + 1]
                    if kb < ka
                        keys_sh[idx] = kb
                        keys_sh[idx + 1] = ka
                        pa = perm_sh[idx]
                        perm_sh[idx] = perm_sh[idx + 1]
                        perm_sh[idx + 1] = pa
                    end
                    idx += 2 * WG
                end
                @synchronize
                phase += 1
            end
            idx = t
            while idx <= cnt
                subsort_keys[first + idx - 1] = keys_sh[idx]
                perm[first + idx - 1] = perm_sh[idx]
                idx += WG
            end
            @synchronize
        end
        cell += n_groups
    end
end

"""
    ka_nearfield_subsort!(ctx, cache, n, n_cells; workgroup=256)

Port of `_cuda_nearfield_subsort!`: compose a within-cell sub-Morton ordering
into `ctx.grid.perm` and refresh `invperm`. A no-op when the grid is already at
the Morton depth cap (`sub == 0`) or the grid is empty.
"""
function ka_nearfield_subsort!(ctx, cache::FastMultipole.RadixFMMCache, n::Int,
        n_cells::Int; workgroup::Int=256)
    sub = min(3, FastMultipole.RADIX_GRID_MAX_ELL - cache.ell)
    (sub > 0 && n > 0 && n_cells > 0) || return nothing
    grid = ctx.grid
    backend = KA.get_backend(grid.perm)
    kk = _cached_kernel(ka_subsort_keys_kernel!, backend, 128)
    kk(ctx.subsort_keys, ctx.positions, grid.perm, cache.x_min, cache.h0,
       cache.ell, sub, n; ndrange=n)
    n_groups = min(n_cells, 8192)
    sk = _cached_kernel(ka_subsort_cell_sort_kernel!, backend, workgroup)
    sk(grid.perm, ctx.subsort_keys, grid.cell_ranges, n_cells, n_groups,
       eltype(grid.perm), Val(KA_SUBSORT_CAPACITY), Val(workgroup);
       ndrange=n_groups * workgroup)
    ka_fill_invperm!(grid.invperm, view(grid.perm, 1:n))
    return nothing
end



#------- SFS (task 048): TG precompute, zeta pair sweep, E formation, scatter -------#
#
# Port of the four CUDA SFS kernels (`_cuda_sfs_tg_kernel!`,
# `_cuda_sfs_zeta_pairs_kernel!`, `_cuda_sfs_form_e_kernel!`,
# `_cuda_sfs_scatter_kernel!`, src/translate_batched_cuda.jl:5751-5915) and
# their two launchers.
#
# SFS is not an FMM operator: nothing here touches an expansion, a stencil or a
# route. Three of the four kernels are pointwise over bodies, and the fourth
# walks the SAME direct pair list `ka_launch_nearfield!` already walks. It lives
# in the radix cache only because both its inputs -- the J rows of
# `state.output` and the direct pair list -- are already device resident.
#
# The per-body math is NOT reimplemented: `_sfs_apply_op` (backend-agnostic, in
# src/translate_batched_resident.jl) is shared with the host mirror, so the
# transposed/classic scheme cannot drift between the two. It takes `transposed`
# as a plain `Bool`, which the `Val{TRANSPOSED}` launch constant-folds.
#
# DEVIATION (launch shape only, same as the nearfield port): CUDA runs a WARP
# per pair striding targets by 32 with a grid-stride outer loop; KA runs a
# WORKGROUP per pair striding targets by the workgroup size, with one group per
# pair and no outer loop. `ndrange` is exact, so the grid-stride wrapper CUDA
# needs to respect `DIRECT_CUDA_MAX_BLOCKS` has nothing to do here. Accumulation
# into `om`/`q` stays atomic: targets of different pairs overlap.

@kernel function ka_sfs_tg_kernel!(tg, om, q, @Const(output), @Const(source_bodies),
        ::Type{T}, ::Val{TRANSPOSED}, n_bodies) where {T,TRANSPOSED}
    i = @index(Global)
    @inbounds if i <= n_bodies
        g1 = source_bodies[5, i]
        g2 = source_bodies[6, i]
        g3 = source_bodies[7, i]
        t1, t2, t3 = FastMultipole._sfs_apply_op(
            output[5, i], output[6, i], output[7, i], output[8, i],
            output[9, i], output[10, i], output[11, i], output[12, i],
            output[13, i], g1, g2, g3, TRANSPOSED)
        tg[1, i] = t1; tg[2, i] = t2; tg[3, i] = t3
        om[1, i] = zero(T); om[2, i] = zero(T); om[3, i] = zero(T)
        q[1, i] = zero(T); q[2, i] = zero(T); q[3, i] = zero(T)
    end
end

@kernel function ka_sfs_zeta_pairs_kernel!(om, q, @Const(tg), @Const(source_bodies),
        @Const(cell_ranges), @Const(direct_targets), @Const(direct_sources),
        npairs, rc2, K1, active_row, ::Type{T}, ::Val{WG}) where {T,WG}
    pair_i = @index(Group)
    tid = @index(Local)
    half = T(0.5)
    @inbounds begin
        target_cell = direct_targets[pair_i]
        source_cell = direct_sources[pair_i]
        tfirst = cell_ranges[1, target_cell]
        tlast = tfirst + cell_ranges[2, target_cell] - 1
        sfirst = cell_ranges[1, source_cell]
        slast = sfirst + cell_ranges[2, source_cell] - 1
        i = tfirst + tid - 1
        while i <= tlast
            if active_row == 0 || !iszero(source_bodies[active_row, i])
                xi = source_bodies[1, i]
                yi = source_bodies[2, i]
                zi = source_bodies[3, i]
                o1 = zero(T); o2 = zero(T); o3 = zero(T)
                q1 = zero(T); q2 = zero(T); q3 = zero(T)
                for j in sfirst:slast
                    if i != j && (active_row == 0 || !iszero(source_bodies[active_row, j]))
                        dx = xi - source_bodies[1, j]
                        dy = yi - source_bodies[2, j]
                        dz = zi - source_bodies[3, j]
                        r2 = dx * dx + dy * dy + dz * dz
                        sigma = source_bodies[8, j]
                        rho2 = r2 / (sigma * sigma)
                        if rho2 <= rc2
                            z = K1 * exp(-half * rho2) / (sigma * sigma * sigma)
                            o1 += z * source_bodies[5, j]
                            o2 += z * source_bodies[6, j]
                            o3 += z * source_bodies[7, j]
                            q1 += z * tg[1, j]
                            q2 += z * tg[2, j]
                            q3 += z * tg[3, j]
                        end
                    end
                end
                KA.@atomic om[1, i] += o1
                KA.@atomic om[2, i] += o2
                KA.@atomic om[3, i] += o3
                KA.@atomic q[1, i] += q1
                KA.@atomic q[2, i] += q2
                KA.@atomic q[3, i] += q3
            end
            i += WG
        end
    end
end

@kernel function ka_sfs_form_e_kernel!(tg, @Const(om), @Const(q), @Const(output),
        ::Val{TRANSPOSED}, n_bodies) where TRANSPOSED
    i = @index(Global)
    @inbounds if i <= n_bodies
        e1, e2, e3 = FastMultipole._sfs_apply_op(
            output[5, i], output[6, i], output[7, i], output[8, i],
            output[9, i], output[10, i], output[11, i], output[12, i],
            output[13, i], om[1, i], om[2, i], om[3, i], TRANSPOSED)
        tg[1, i] = e1 - q[1, i]
        tg[2, i] = e2 - q[2, i]
        tg[3, i] = e3 - q[3, i]
    end
end

@kernel function ka_sfs_scatter_kernel!(target_buffer, @Const(e), @Const(perm),
        @Const(body_system), @Const(body_index), isys, n_bodies)
    sorted_i = @index(Global)
    @inbounds if sorted_i <= n_bodies
        global_i = perm[sorted_i]
        if body_system[global_i] == isys
            ibody = body_index[global_i]
            target_buffer[1, ibody] = e[1, sorted_i]
            target_buffer[2, ibody] = e[2, sorted_i]
            target_buffer[3, ibody] = e[3, sorted_i]
        end
    end
end

"""
    ka_launch_sfs!(state; workgroup=64)

Port of `_launch_cuda_sfs!`: TG precompute + accumulator zeroing, then the zeta
sweep over the full direct pair list. Called only for an evaluation that asks
for `sfs=true`, after the U/J lifecycle has completed, so `state.output` carries
a finished J.
"""
function ka_launch_sfs!(state::FastMultipole.DeviceResidentRadixState{TF};
        workgroup::Int=64) where TF
    sfs = state.sfs
    sfs === nothing && return state
    size(state.output, 1) >= 13 || throw(AssertionError(
        "the SFS pass requires the 13-row (hessian) output"))
    _ka_launch_sfs_typed!(state, sfs.tg, sfs.om, sfs.q,
        sfs.transposed ? Val(true) : Val(false), sfs.active_row, workgroup)
    return state
end

# function barrier over the Any-typed sfs NamedTuple (CUDA does the same)
function _ka_launch_sfs_typed!(state::FastMultipole.DeviceResidentRadixState{TF},
        tg::AbstractMatrix{TF}, om::AbstractMatrix{TF}, q::AbstractMatrix{TF},
        tv::Val, active_row::Int, workgroup::Int) where TF
    n = state.counts.n_bodies
    n > 0 || return state
    backend = KA.get_backend(state.output)
    tgk = _cached_kernel(ka_sfs_tg_kernel!, backend, workgroup)
    tgk(tg, om, q, state.output, state.source_bodies, TF, tv, n; ndrange=n)
    npairs = state.counts.n_direct
    if npairs > 0
        zk = _cached_kernel(ka_sfs_zeta_pairs_kernel!, backend, workgroup)
        zk(om, q, tg, state.source_bodies, state.cell_ranges,
           state.direct_targets, state.direct_sources, npairs,
           FastMultipole._sfs_saturation_rc2(TF), TF(FastMultipole._SFS_ZETA_K1),
           active_row, TF, Val(workgroup); ndrange=npairs * workgroup)
    end
    # NO sync here: kernels queued on one backend run in order, and this file's
    # discipline is one sync at the end of a DRIVER, never per stage (see the
    # `ka_lifecycle_body!` comment). The step's finalize is what synchronizes.
    return state
end

#------- ζ reconstruction for core spreading (nearfield-only pair sum) -------#
#
# Device counterpart of FLOWVPM's host `zeta_fmm`: ζ_i = Σ_j Γ_j ζ(r/σ_j)/σ_j³ over
# the radix direct list with the Gaussian ζ(ρ) = (2π)^(-3/2) exp(-ρ²/2). Unlike the
# SFS ζ sweep this keeps the self pair (it is the diagonal of the RBF system the
# core-spreading conjugate gradient solves), applies no saturation cutoff and no
# static-particle filter -- exactly what the host loop does.
@kernel function ka_zeta_pairs_kernel!(om, @Const(source_bodies), @Const(cell_ranges),
        @Const(direct_targets), @Const(direct_sources), npairs, K1, ::Type{T}, ::Val{WG}) where {T,WG}
    pair_i = @index(Group)
    tid = @index(Local)
    half = T(0.5)
    @inbounds begin
        target_cell = direct_targets[pair_i]
        source_cell = direct_sources[pair_i]
        tfirst = cell_ranges[1, target_cell]
        tlast = tfirst + cell_ranges[2, target_cell] - 1
        sfirst = cell_ranges[1, source_cell]
        slast = sfirst + cell_ranges[2, source_cell] - 1
        i = tfirst + tid - 1
        while i <= tlast
            xi = source_bodies[1, i]
            yi = source_bodies[2, i]
            zi = source_bodies[3, i]
            o1 = zero(T); o2 = zero(T); o3 = zero(T)
            for j in sfirst:slast
                dx = xi - source_bodies[1, j]
                dy = yi - source_bodies[2, j]
                dz = zi - source_bodies[3, j]
                r2 = dx * dx + dy * dy + dz * dz
                sigma = source_bodies[8, j]
                z = K1 * exp(-half * r2 / (sigma * sigma)) / (sigma * sigma * sigma)
                o1 += z * source_bodies[5, j]
                o2 += z * source_bodies[6, j]
                o3 += z * source_bodies[7, j]
            end
            KA.@atomic om[1, i] += o1
            KA.@atomic om[2, i] += o2
            KA.@atomic om[3, i] += o3
            i += WG
        end
    end
end

function ka_launch_zeta!(state::FastMultipole.DeviceResidentRadixState{TF}, om;
        workgroup::Int=64) where TF
    fill!(om, zero(TF))
    npairs = state.counts.n_direct
    npairs == 0 && return om
    kernel = _cached_kernel(ka_zeta_pairs_kernel!, KA.get_backend(om), workgroup)
    kernel(om, state.source_bodies, state.cell_ranges, state.direct_targets,
           state.direct_sources, npairs, TF(FastMultipole._SFS_ZETA_K1), TF, Val(workgroup);
           ndrange=npairs * workgroup)
    return om
end

function FastMultipole.radix_zeta!(cache::FastMultipole.RadixFMMCache{TF,LH}, systems::Tuple,
        om, out; workgroup::Int=64) where {TF,LH}
    # repack X/Γ/σ (Γ changes every conjugate-gradient iteration) and refresh the lists
    ka_update_radix_state!(cache, systems)
    state = cache.state
    n = state.counts.n_bodies
    size(om, 2) >= n || throw(ArgumentError("radix_zeta!: om holds $(size(om, 2)) columns, need $n"))
    ka_launch_zeta!(state, om; workgroup)
    backend = KA.get_backend(om)
    for (isys, target_system) in enumerate(systems)
        FastMultipole.residency(target_system) isa FastMultipole.DeviceResident || throw(ArgumentError(
            "radix_zeta!: target system $isys must be device-resident"))
        nb = FastMultipole.get_n_bodies(target_system)
        size(out, 2) >= nb || throw(ArgumentError("radix_zeta!: out holds $(size(out, 2)) columns, need $nb"))
        buf = size(out, 2) == nb ? out : view(out, :, 1:nb)
        fill!(buf, zero(TF))
        sk = _cached_kernel(ka_sfs_scatter_kernel!, backend, workgroup)
        sk(buf, om, state.body_perm, state.body_system_ids, state.body_indices, isys, n; ndrange=n)
        FastMultipole.zeta_to_target!(target_system, buf, 1:nb)
    end
    return out
end

"""
    ka_finalize_radix_sfs_output!(state, target_systems; host_sfs_staging=nothing,
        sfs_target_buffers=nothing, device_sfs_buffers=nothing, workgroup=64)

Backend-agnostic `finalize_cuda_radix_sfs_output!`. Forms E = op(J)Ω − Q into
`sfs.tg` (dead after the pair sweep), de-permutes sorted -> global, and delivers
a per-system `3 x n_bodies` global-order buffer through `sfs_to_target!` --
device buffer for a `DeviceResident` target, host buffer otherwise. Called
outside the lifecycle, next to `ka_finalize_radix_output!`.
"""
function ka_finalize_radix_sfs_output!(state::FastMultipole.DeviceResidentRadixState{TF},
        target_systems; host_sfs_staging=nothing, sfs_target_buffers=nothing,
        device_sfs_buffers=nothing, workgroup::Int=64) where TF
    sfs = state.sfs
    sfs === nothing && throw(ArgumentError(
        "sfs=true evaluation requires a RadixFMMCache built with sfs=true"))
    systems = FastMultipole.to_tuple(target_systems)
    n = state.counts.n_bodies
    n > 0 || return target_systems
    backend = KA.get_backend(state.output)
    ek = _cached_kernel(ka_sfs_form_e_kernel!, backend, workgroup)
    ek(sfs.tg, sfs.om, sfs.q, state.output,
       sfs.transposed ? Val(true) : Val(false), n; ndrange=n)
    host_e = nothing
    for (isys, target_system) in enumerate(systems)
        nb = FastMultipole.get_n_bodies(target_system)
        if FastMultipole.residency(target_system) isa FastMultipole.DeviceResident
            buf = _ka_cached_target_buffer(device_sfs_buffers, backend, isys, TF, 3, nb)
            fill!(buf, zero(TF))
            sk = _cached_kernel(ka_sfs_scatter_kernel!, backend, workgroup)
            sk(buf, sfs.tg, state.body_perm, state.body_system_ids,
               state.body_indices, isys, n; ndrange=n)
            FastMultipole.sfs_to_target!(target_system, buf, 1:nb)
        else
            if host_e === nothing
                if host_sfs_staging === nothing
                    host_e = Array(sfs.tg)
                else
                    KA.synchronize(backend)
                    copyto!(host_sfs_staging, 1, sfs.tg, 1, 3 * n)
                    host_e = host_sfs_staging
                end
                state.counters.influence_downloads += 1
            end
            buf_full = sfs_target_buffers === nothing ?
                Matrix{TF}(undef, 3, nb) : sfs_target_buffers[isys]
            buf = size(buf_full, 2) == nb ? buf_full : view(buf_full, :, 1:nb)
            FastMultipole._scatter_sfs_host!(buf, host_e, state.host_body_perm,
                state.host_body_system_ids, state.host_body_indices, isys, n)
            FastMultipole.sfs_to_target!(target_system, buf, 1:nb)
        end
    end
    return target_systems
end



#------- hierarchical refresh: direct pairs, symmetric compaction, window cache -------#
#
# The three pieces of the hierarchical branch of `update_cuda_radix_state!`
# (src/translate_batched_cuda.jl:6857) that were still CUDA-only. This is the
# branch FLOWVPM actually takes: its cache is built with `window_classes`, so
# `RadixFMMCache` selects a `HierarchicalRigidStencil` and `hierarchical_ctx` is
# non-`nothing`. The flat `ka_generate_radix_routes!` above serves the
# `hctx === nothing` path only -- it is NOT what the production refresh calls.
#
# `ka_hier_refresh_occupancy!` (above) already covered the per-level occupancy
# lookup and `ka_hier_generate_window_core!` the single-window generator, so
# what is added here is the direct-pair generator, the symmetric compaction, and
# the loop that concatenates every level's windows into the epoch cache.
#
# Direct pairs differ from the flat path's in what they index: the flat kernels
# look up `cell_at` by decoded leaf Morton key, these look up the hierarchical
# per-level `node_at` by the leaf node's stored `node_coords` at `level_base_L`.
# Both chunk over the flag buffer with a running output base, for the same
# reason -- the flag/prefix buffers stay bounded by the scratch capacity rather
# than by `kn * n_cells`.

@kernel function ka_hier_direct_flags_kernel!(flags, @Const(node_at), @Const(node_coords),
        @Const(near_offsets), fbase, len, kn, leaf_base, level_base_L, ell)
    idx = @index(Global)
    @inbounds if idx <= len
        g = fbase + idx
        c = (g - 1) ÷ kn + 1
        k = (g - 1) % kn + 1
        G = 1 << ell
        target_node = leaf_base + c
        sx = node_coords[1, target_node] - near_offsets[1, k]
        sy = node_coords[2, target_node] - near_offsets[2, k]
        sz = node_coords[3, target_node] - near_offsets[3, k]
        src = Int32(0)
        if 0 <= sx < G && 0 <= sy < G && 0 <= sz < G
            src = node_at[level_base_L + (sx + G * (sy + G * sz)) + 1]
        end
        flags[idx] = src == Int32(0) ? Int32(0) : Int32(1)
    end
end

@kernel function ka_hier_direct_compact_kernel!(direct_targets, direct_sources,
        @Const(flags), @Const(prefix), @Const(node_at), @Const(node_coords),
        @Const(near_offsets), fbase, len, kn, leaf_base, level_base_L, ell, base)
    idx = @index(Global)
    @inbounds if idx <= len && flags[idx] == Int32(1)
        g = fbase + idx
        c = (g - 1) ÷ kn + 1
        k = (g - 1) % kn + 1
        G = 1 << ell
        target_node = leaf_base + c
        sx = node_coords[1, target_node] - near_offsets[1, k]
        sy = node_coords[2, target_node] - near_offsets[2, k]
        sz = node_coords[3, target_node] - near_offsets[3, k]
        src = Int(node_at[level_base_L + (sx + G * (sy + G * sz)) + 1])
        p = base + Int(prefix[idx])
        direct_targets[p] = c
        direct_sources[p] = src - leaf_base
    end
end

"""
    ka_hier_generate_direct_pairs!(ctx, hctx, grid, n_cells, leaf_base, ell;
                                   workgroup=KA_AUTO_WORKGROUP)

Port of `_cuda_hier_generate_direct_pairs!`: flag/scan/compact the near-offset
neighbours of every occupied leaf cell into `ctx.direct_targets` /
`ctx.direct_sources`, chunked so the flag buffer bounds the working set.
Returns the pair count.
"""
function ka_hier_generate_direct_pairs!(ctx, hctx::FastMultipole.DeviceHierarchicalM2LContext,
        grid, n_cells::Int, leaf_base::Int, ell::Int; workgroup=KA_AUTO_WORKGROUP)
    kn = size(hctx.d_near_offsets, 2)
    (n_cells > 0 && kn > 0) || return 0
    backend = KA.get_backend(ctx.direct_flags)
    level_base_L = hctx.level_base[ell + 1]
    total = kn * n_cells
    capacity = length(ctx.direct_flags)
    capacity > 0 ||
        throw(AssertionError("device direct flag buffer has zero capacity"))
    flagk = _cached_kernel(ka_hier_direct_flags_kernel!, backend, workgroup)
    compactk = _cached_kernel(ka_hier_direct_compact_kernel!, backend, workgroup)
    n_direct = 0
    f0 = 0
    while f0 < total
        len = min(capacity, total - f0)
        flagk(ctx.direct_flags, hctx.node_at, grid.node_coords, hctx.d_near_offsets,
            f0, len, kn, leaf_base, level_base_L, ell; ndrange=len)
        accumulate!(+, view(ctx.direct_prefix, 1:len), view(ctx.direct_flags, 1:len))
        KA.synchronize(backend)
        copyto!(ctx.host_scalar32, 1, ctx.direct_prefix, len, 1)
        chunk_total = Int(ctx.host_scalar32[1])
        if chunk_total > 0
            n_direct + chunk_total <= length(ctx.direct_targets) ||
                throw(AssertionError("device hierarchical direct pair buffer exceeded its capacity"))
            compactk(ctx.direct_targets, ctx.direct_sources, ctx.direct_flags,
                ctx.direct_prefix, hctx.node_at, grid.node_coords,
                hctx.d_near_offsets, f0, len, kn, leaf_base, level_base_L, ell,
                n_direct; ndrange=len)
        end
        n_direct += chunk_total
        f0 += len
    end
    KA.synchronize(backend)
    return n_direct
end

@kernel function ka_symmetric_pair_flags_kernel!(flags, @Const(direct_targets),
        @Const(direct_sources), @Const(cell_ranges), n_direct, max_bodies)
    i = @index(Global)
    @inbounds if i <= n_direct
        t = direct_targets[i]
        s = direct_sources[i]
        oversized = cell_ranges[2, t] > max_bodies || cell_ranges[2, s] > max_bodies
        flags[i] = (oversized || t <= s) ? Int32(1) : Int32(0)
    end
end

@kernel function ka_symmetric_pair_compact_kernel!(targets, sources, @Const(flags),
        @Const(prefix), @Const(direct_targets), @Const(direct_sources),
        @Const(cell_ranges), n_direct, max_bodies)
    i = @index(Global)
    @inbounds if i <= n_direct && flags[i] == Int32(1)
        t = direct_targets[i]
        s = direct_sources[i]
        oversized = cell_ranges[2, t] > max_bodies || cell_ranges[2, s] > max_bodies
        p = Int(prefix[i])
        # oversized cells keep BOTH directed entries and encode the dense
        # fallback as a negative target id, so the selection stays on device
        targets[p] = oversized ? -t : t
        sources[p] = s
    end
end

"""
    ka_compact_symmetric_pairs!(ctx, hctx, cell_ranges, n_direct, max_bodies;
                                workgroup=KA_AUTO_WORKGROUP)

Port of `_cuda_compact_symmetric_pairs!`: reduce the directed leaf-cell pair
list to one unordered entry per ordinary pair, keeping both directed entries for
oversized cells (negative target id = dense fallback). Writes
`hctx.symmetric_targets`/`symmetric_sources`, sets `hctx.n_symmetric_pairs`, and
returns it. Refreshes every step, not only on occupancy change: the oversized
selection reads the per-cell body counts, which move with the bodies.
"""
function ka_compact_symmetric_pairs!(ctx, hctx::FastMultipole.DeviceHierarchicalM2LContext,
        cell_ranges, n_direct::Int, max_bodies::Int; workgroup=KA_AUTO_WORKGROUP)
    n_direct == 0 && (hctx.n_symmetric_pairs = 0; return 0)
    n_direct <= length(ctx.direct_flags) || throw(AssertionError(
        "symmetric compaction exceeds direct scratch capacity"))
    backend = KA.get_backend(ctx.direct_flags)
    flagk = _cached_kernel(ka_symmetric_pair_flags_kernel!, backend, workgroup)
    flagk(ctx.direct_flags, ctx.direct_targets, ctx.direct_sources, cell_ranges,
        n_direct, max_bodies; ndrange=n_direct)
    accumulate!(+, view(ctx.direct_prefix, 1:n_direct), view(ctx.direct_flags, 1:n_direct))
    KA.synchronize(backend)
    copyto!(ctx.host_scalar32, 1, ctx.direct_prefix, n_direct, 1)
    n = Int(ctx.host_scalar32[1])
    n <= length(hctx.symmetric_targets) || throw(AssertionError(
        "symmetric pair buffer exceeded capacity"))
    if n > 0
        compactk = _cached_kernel(ka_symmetric_pair_compact_kernel!, backend, workgroup)
        compactk(hctx.symmetric_targets, hctx.symmetric_sources, ctx.direct_flags,
            ctx.direct_prefix, ctx.direct_targets, ctx.direct_sources, cell_ranges,
            n_direct, max_bodies; ndrange=n_direct)
        KA.synchronize(backend)
    end
    hctx.n_symmetric_pairs = n
    return n
end

# Grow the cached-window arrays to `needed`, preserving the first `cursor`
# entries. Generic form of `_cuda_hier_win_ensure!`; growth happens only inside
# epoch regeneration, so this allocation recurs exactly with occupancy change.
function _ka_hier_win_ensure!(hctx, backend, cursor::Int, needed::Int)
    old_class = hctx.win_class
    cap = old_class === nothing ? 0 : length(old_class)
    needed <= cap && return nothing
    newcap = max(needed, cap + cld(cap, 2), 1024)
    new_class = KA.allocate(backend, Int32, newcap)
    new_sources = KA.allocate(backend, Int, newcap)
    new_targets = KA.allocate(backend, Int, newcap)
    if cursor > 0
        copyto!(new_class, 1, old_class, 1, cursor)
        copyto!(new_sources, 1, hctx.win_sources, 1, cursor)
        copyto!(new_targets, 1, hctx.win_targets, 1, cursor)
    end
    hctx.win_class = new_class
    hctx.win_sources = new_sources
    hctx.win_targets = new_targets
    return nothing
end

"""
    ka_hier_cache_windows!(ctx, hctx, grid; workgroup=KA_AUTO_WORKGROUP)

Port of `_cuda_hier_cache_windows!`: regenerate the complete per-level window
concatenation for the current occupancy epoch, reusing the single-window
generator verbatim so the cached routes are byte-identical to what the M2L stage
would have generated window by window. Runs inside the refresh, which is legal
because windows read node metadata only, never expansions.
"""
function ka_hier_cache_windows!(ctx, hctx::FastMultipole.DeviceHierarchicalM2LContext,
        grid; workgroup=KA_AUTO_WORKGROUP)
    plan = hctx.apply_plan
    route_class = plan.route_class
    backend = KA.get_backend(ctx.route_targets)
    noffsets = hctx.noffsets
    K = hctx.window_classes
    ell = hctx.ell
    # BUG FIX (found porting the concat cache): `class_base` was hardcoded 0
    # here while the live loop in `ka_hierarchical_m2l!` uses
    # `(L - first_m2l_level) * noffsets` for concat. Dense folds the level into
    # a scale column so 0 is right there, but a concat cache built with 0 would
    # apply every level with level-2 operators.
    dense = plan isa FastMultipole.ResidentM2LDenseCUDAPlan
    windows = Tuple{Int,Int,Int}[]
    for L in hctx.first_m2l_level:ell, first_offset in 1:K:noffsets
        push!(windows, (L, first_offset, min(first_offset + K - 1, noffsets)))
    end
    nw = length(windows)
    _KA_UPDATE_TIMERS[] === nothing || push!(get!(_KA_UPDATE_TIMERS[], :win_n_windows, Float64[]), nw)
    # One concatenated flag buffer for every window, ONE scan, one sync:
    # window w's flags occupy bases[w]+1 : bases[w]+used[w].
    n_src(L) = hctx.level_offsets[L + 2] - hctx.level_offsets[L + 1]
    used = [n_src(L) * (lo - fo + 1) for (L, fo, lo) in windows]
    bases = cumsum([0; used[1:end-1]])
    total_used = sum(used)
    flags_all = KA.zeros(backend, Int32, max(total_used, 1))
    prefix_all = KA.zeros(backend, Int32, max(total_used, 1))
    flags_kernel = _cached_kernel(ka_hier_route_flags_kernel!, backend, workgroup)
    for (w, (L, fo, lo)) in enumerate(windows)
        used[w] > 0 || continue
        kn = lo - fo + 1
        flags_kernel(view(flags_all, bases[w]+1:bases[w]+used[w]), hctx.node_at,
            grid.node_coords, hctx.d_push_offsets, hctx.d_class_of, hctx.level_base[L + 1],
            hctx.level_offsets[L + 1] + 1, n_src(L), fo, kn, L; ndrange=used[w])
    end
    total_used > 0 && accumulate!(+, view(prefix_all, 1:total_used), view(flags_all, 1:total_used))
    _utick!(:win_phase1_count, backend)
    ends = KA.allocate(backend, Int, max(nw, 1)); copyto!(ends, max.(bases .+ used, 1))
    host_ends = Array(prefix_all[ends])          # the one D2H sync
    _utick!(:win_sync_d2h, backend)
    total_routes = nw > 0 ? Int(host_ends[end]) : 0
    _ka_hier_win_ensure!(hctx, backend, 0, total_routes)
    fill!(hctx.win_level_starts, 0)
    fill!(hctx.win_level_counts, 0)
    fill!(hctx.routes_per_level, 0)
    compact_kernel = _cached_kernel(ka_hier_route_compact_global_kernel!, backend, workgroup)
    cursor = 0; level_total = 0; current_L = -1
    for (w, (L, first_offset, last_offset)) in enumerate(windows)
        if L != current_L
            current_L == -1 || (hctx.win_level_counts[current_L + 1] = level_total;
                                hctx.routes_per_level[current_L + 1] = level_total)
            current_L = L; level_total = 0
            hctx.win_level_starts[L + 1] = cursor
        end
        used[w] > 0 || continue
        n = Int(host_ends[w]) - (w > 1 ? Int(host_ends[w-1]) : 0)
        n == 0 && continue
        class_base = dense ? 0 : (L - hctx.first_m2l_level) * noffsets
        kn = last_offset - first_offset + 1
        compact_kernel(hctx.win_targets, hctx.win_sources, hctx.win_class,
            flags_all, prefix_all, bases[w], hctx.node_at, grid.node_coords,
            hctx.d_push_offsets, hctx.level_base[L + 1], hctx.level_offsets[L + 1] + 1,
            n_src(L), first_offset, kn, L, class_base; ndrange=used[w])
        cursor += n
        level_total += n
    end
    current_L == -1 || (hctx.win_level_counts[current_L + 1] = level_total;
                        hctx.routes_per_level[current_L + 1] = level_total)
    _utick!(:win_phase2_compact, backend)
    hctx.total_routes = cursor
    hctx.last_window_routes = 0
    hctx.win_valid = true
    return hctx
end



#------- backend-agnostic device step (the whole uniform sfs=false lifecycle) -------#
#
# KA form of `update_cuda_radix_state!` + `_radix_cache_device_step!`
# (src/translate_batched_cuda.jl:6791 and :6948) for the branch FLOWVPM runs:
# `cache.adaptive === nothing` (uniform grid) with a `HierarchicalRigidStencil`,
# hence `ctx.hierarchical_ctx !== nothing`. Everything it calls now exists off
# CUDA -- the four grid-rebuild stages, the hierarchical occupancy/direct-pair/
# window-cache refresh, the stage-group edges, the lifecycle body, and the
# output finalize -- so this driver is what closes the loop: with it, no part of
# the uniform `sfs=false` step needs `translate_batched_cuda.jl` to be loaded.
#
# Two deliberate omissions, both perf-only and both following the precedent set
# by the stage-1 counting sort:
#
#   * `_cuda_nearfield_subsort!` (CUDA_NEARFIELD_SUBSORT, default on with
#     PartitionedVortex) composes a within-cell sub-Morton reordering into
#     `perm` for nearfield locality. It changes no cell key, cell range or node
#     -- only the order bodies are summed in -- so omitting it costs locality,
#     not correctness, and keeps the KA arm's summation order deterministic.
#   * CUDA graph capture/replay has no KA equivalent; the body is launched
#     directly every step.
#
#   * `sfs` is now ported (task 048): `ka_launch_sfs!` +
#     `ka_finalize_radix_sfs_output!` run between the lifecycle body and the
#     U/J finalize, in CUDA's order.

"""
    ka_update_radix_state!(cache, systems; workgroup=KA_AUTO_WORKGROUP)

Backend-agnostic `update_cuda_radix_state!` for a uniform, hierarchical
`RadixFMMCache(device=true)`: refresh the per-system source buffers, rebuild the
grid in place inside the cache's fixed Morton box, refresh the hierarchical
occupancy / direct pairs / cached M2L windows on occupancy change, and refresh
the per-level operator-group edges. Returns the cache with `cache.state` built
(first step) or refreshed in place.
"""
# `radix_setting` throws for a CUDA-only tunable until `load_cuda_radix_lifecycle!()`
# has defined its Ref -- which never happens on a non-CUDA backend, so every
# read below would abort the KA step. The KA arm honours the same tunables when
# CUDA has been loaded and falls back to the shipped default otherwise (the
# values in translate_batched_cuda.jl, kept in sync by the assertion at each
# call site's comment). It is a read-only fallback: nothing here can set one.
@kernel function ka_iota_kernel!(perm, invperm, n)
    i = @index(Global)
    @inbounds if i <= n
        perm[i] = i
        invperm[i] = i
    end
end

"Identity body permutation for the all-pairs direct arm, which does no sort."
function _ka_identity_perm!(perm, invperm, n::Int; workgroup=KA_AUTO_WORKGROUP)
    backend = KA.get_backend(perm)
    wg = resolve_workgroup(backend, workgroup)
    kern = _cached_kernel(ka_iota_kernel!, backend, wg)
    kern(perm, invperm, n; ndrange=cld(n, wg) * wg)
    return perm
end

# Settings the former native lifecycle defined (CUDA_NEARFIELD_SUBSORT,
# CUDA_CACHED_WINDOWS, ...) have no Ref once it is gone, so the KA path uses
# its own defaults; `_KA_SETTING_OVERRIDES` lets a probe or a user flip one.
const _KA_SETTING_OVERRIDES = Dict{Symbol,Any}()
_ka_radix_setting(name::Symbol, default) =
    haskey(_KA_SETTING_OVERRIDES, name) ? _KA_SETTING_OVERRIDES[name] :
    FastMultipole._radix_setting_ref(name) === nothing ? default :
        FastMultipole.radix_setting(name)

# Optional per-stage timers for ka_update_radix_state!, used by the attribution
# probe: set `_KA_UPDATE_TIMERS[] = Dict{Symbol,Vector{Float64}}()` and every
# `_utick!` syncs the backend and records the time since the previous tick.
# `nothing` (the default) makes each tick a no-op.
const _KA_UPDATE_TIMERS = Ref{Any}(nothing)
const _KA_UPDATE_T0 = Ref{Float64}(0.0)
@inline function _utick!(name::Symbol, backend)
    d = _KA_UPDATE_TIMERS[]
    d === nothing && return nothing
    KA.synchronize(backend)
    t = time(); push!(get!(d, name, Float64[]), (t - _KA_UPDATE_T0[]) * 1e3); _KA_UPDATE_T0[] = t
    return nothing
end


function ka_update_radix_state!(cache::FastMultipole.RadixFMMCache{TF,LH}, systems::Tuple;
        workgroup=KA_AUTO_WORKGROUP, direct_only::Bool=false) where {TF,LH}
    ctx = cache.device_ctx
    ctx === nothing &&
        throw(ArgumentError("ka_update_radix_state! requires a cache built with device=true"))
    cache.adaptive === nothing ||
        throw(ArgumentError("ka_update_radix_state! covers the uniform path only; " *
            "the adaptive device lifecycle is CUDA-only"))
    length(systems) == cache.n_systems ||
        throw(ArgumentError("cache was built for $(cache.n_systems) source systems, got $(length(systems))"))
    n = FastMultipole.get_n_bodies(systems)
    n > 0 || throw(ArgumentError("ka_update_radix_state! requires at least one body"))
    n <= cache.max_n_bodies ||
        throw(ArgumentError("n=$n exceeds the cache capacity max_n_bodies=$(cache.max_n_bodies)"))
    grid = ctx.grid
    hctx = ctx.hierarchical_ctx
    counters = ctx.counters
    backend = KA.get_backend(ctx.positions)
    ell = cache.ell
    first_level = cache.root_level

    source_buffers = FastMultipole._radix_cache_refresh_source_buffers!(ctx, systems, TF)
    _KA_UPDATE_TIMERS[] === nothing || (KA.synchronize(backend); _KA_UPDATE_T0[] = time())
    if direct_only
        # all-pairs arm: no grid, no tree, no routes, no box check. Bodies are
        # packed in identity order -- the perm's only consumer on this arm is
        # `ka_finalize_radix_output!`, which scatters back through it.
        #
        # The system/index tags still have to be refreshed: the pack kernel
        # writes only columns whose tag matches, so bodies added since the
        # last tagging call were packed as zeros (position at the origin, no
        # strength) -- exact on a fixed field, wrong on a growing one.
        ka_collect_positions!(ctx.positions, grid.body_system, grid.body_index,
            source_buffers; workgroup)
        _ka_identity_perm!(grid.perm, grid.invperm, n; workgroup)
        n_cells = 0
        n_nodes = 0
        grid.n_bodies = n
        grid.n_cells = 0
    else
        ka_collect_positions!(ctx.positions, grid.body_system, grid.body_index,
            source_buffers; workgroup)

        # ---- grid rebuild, stages 1-4 ----
    _utick!(:collect_positions, backend)
        ka_radix_keys_checked!(view(ctx.keys, 1:n), ctx.oob_flag, ctx.host_oob,
            ctx.positions, cache.x_min, cache.box_extent, cache.h0, ell; workgroup)
        kv = view(ctx.keys, 1:n)
        sk = view(ctx.sorted_keys, 1:n)
        # branch exactly where `_cuda_update_radix_grid_in_place!` branches: the
        # bounded counting sort when the cache was built with a domain-sized
        # histogram and the setting is on for this `ell`, else the stable sortperm
    _utick!(:keys, backend)
        ka_radix_sort_bodies!(view(grid.perm, 1:n), sk, grid.invperm, kv; workgroup,
            ell=ell, histogram=ctx.counting_histogram, prefix=ctx.counting_prefix,
            cursor=ctx.counting_cursor)
        n_cells = ka_radix_compress_cells!(grid.cell_keys, grid.cell_ranges, sk,
            view(ctx.body_flags, 1:n), view(ctx.body_prefix, 1:n), ctx.host_scalar;
            workgroup)
        n_cells <= cache.max_cells ||
            throw(AssertionError("device radix grid exceeded the cache cell capacity"))
        ckv = view(grid.cell_keys, 1:n_cells)

        # occupancy epoch: everything past this point is a pure function of the
        # occupied leaf-cell SET inside the fixed box
        track_epoch = length(ctx.epoch_cell_keys) > 0 &&
            _ka_radix_setting(:CUDA_CACHED_WINDOWS, true)
    _utick!(:sort_compress, backend)
        # The epoch is keyed on the occupied-cell SET, not on the body count:
        # bodies added to already-occupied cells (a shedding solver, every step)
        # leave every route, window and stage group valid. Compare the keys
        # whenever the cell count matches; a count-only change must not force
        # the rebuild (measured at ~420 ms per call at 400 bodies on Metal,
        # against ~15 ms for the evaluation itself).
        occ_changed = true
        if track_epoch && ctx.epoch_have[] && ctx.epoch_prev_n_cells[] == n_cells
            occ_changed = ka_radix_occupancy_changed!(ctx.epoch_flag, ctx.host_epoch_flag,
                ckv, ctx.epoch_cell_keys, n_cells; workgroup)
        end
    _utick!(:occ_check, backend)
        if occ_changed
            if track_epoch
                copyto!(ctx.epoch_cell_keys, 1, grid.cell_keys, 1, n_cells)
                ctx.epoch_prev_n[] = n
                ctx.epoch_prev_n_cells[] = n_cells
                ctx.epoch_have[] = true
            end
            ka_radix_cell_centers!(grid.cell_centers, ctx.cell_coords, ckv, cache.x_min,
                cache.h0, ell, n_cells; workgroup)
            n_nodes, max_count = ka_radix_level_nodes!(grid.node_keys, cache.level_offsets,
                ctx.level_keys, ctx.level_flags, ctx.level_prefix, ctx.level_counts,
                ctx.host_level_counts, ctx.d_level_offsets, ckv, n_cells, ell,
                first_level, cache.max_nodes; workgroup)
            ka_radix_node_topology!(grid.node_levels, grid.node_coords, grid.node_centers,
                grid.parent_index, grid.child_ranges, view(grid.leaf_to_node, 1:n_cells),
                grid.node_keys, ctx.d_level_offsets, cache.level_offsets, cache.x_min,
                cache.h0, n_cells, ell, first_level, max_count; workgroup)
        end
        n_nodes = cache.level_offsets[end]
        grid.n_bodies = n
        grid.n_cells = n_cells
    end

    # 032a stage C mechanism (a): optional within-cell sub-Morton ordering,
    # composed into the perm before packing (the sorted cell keys, cell ranges
    # and node metadata are unaffected). Same gate as CUDA's at cuda:6807.
    _utick!(:grid_rebuild, backend)
    if !direct_only && _ka_radix_setting(:CUDA_NEARFIELD_SUBSORT, true) &&
            cache.options.direct_kernel isa Union{FastMultipole.PartitionedVortex,
                                                  FastMultipole.TwoPassVortex}
        # NOT the refresh's `workgroup`: the local-memory sort's group size is
        # baked into the kernel (Val(WG) against a fixed capacity), so it is the
        # kernel's own constant, not a tuning surface
        # ([[reference-ka-workgroup-is-sometimes-team-size]]).
        ka_nearfield_subsort!(ctx, cache, n, n_cells)
    end
    pack_sigma_row = _ka_kernel_sigma_row(cache.options.direct_kernel)
    # reciprocal-sigma row = the last row for a regularized kernel (convention
    # of the two allocators; see `_ka_nf_inv_sigma_row`)
    pack_inv_sigma_row = pack_sigma_row > 0 ? size(ctx.source_bodies, 1) : 0
    _utick!(:subsort, backend)
    for isys in eachindex(source_buffers)
        ka_pack_body_matrix!(ctx.source_bodies, source_buffers[isys],
            view(grid.perm, 1:n), grid.body_system, grid.body_index, n;
            isys, sigma_row=pack_sigma_row, inv_sigma_row=pack_inv_sigma_row,
            workgroup)
    end
    # the adequacy gate guards the M2L far field; on the all-pairs arm there is
    # none, so it is vacuous (same reasoning as the zero-M2L degenerate cache)
    direct_only || FastMultipole._direct_kernel_geometry_gate!(cache,
        cache.options.direct_kernel, ctx.source_bodies, n)

    # host mirrors serve host-resident target finalization only
    _utick!(:pack, backend)
    if FastMultipole._radix_any_host_resident(systems)
        KA.synchronize(backend)
        copyto!(ctx.host_perm, 1, grid.perm, 1, n)
        copyto!(ctx.host_body_system, 1, grid.body_system, 1, n)
        copyto!(ctx.host_body_index, 1, grid.body_index, 1, n)
        counters.metadata_downloads += 3
    end

    _utick!(:host_copy, backend)
    if direct_only
        n_routes = 0
        n_direct = 0
    else
        # multi-root tree edges: every node at root_level is a root
        n_root_nodes = cache.level_offsets[cache.root_level + 2]
        n_edges = max(n_nodes - n_root_nodes, 0)
        if occ_changed && n_edges > 0
            ka_tree_routes!(ctx.m2m_parent_routes, ctx.m2m_child_routes,
                ctx.l2l_parent_routes, ctx.l2l_child_routes,
                view(grid.parent_index, 1:n_nodes), n_root_nodes; workgroup)
        end

        hctx === nothing && throw(ArgumentError(
            "ka_update_radix_state! covers the hierarchical stencil path only; " *
            "build the cache with window_classes (FLOWVPM's default)"))
    _utick!(:tree_routes, backend)
        if occ_changed
            hctx.epoch_id += 1
            hctx.win_valid = false
            ka_hier_refresh_occupancy!(hctx, grid, cache.level_offsets; workgroup)
            n_direct = ka_hier_generate_direct_pairs!(ctx, hctx, grid, n_cells,
                cache.level_offsets[ell + 1], ell; workgroup)
            hctx.epoch_n_direct = n_direct
        else
            n_direct = hctx.epoch_n_direct
        end
    _utick!(:refresh_occ_direct_pairs, backend)
        if isempty(hctx.symmetric_targets)
            hctx.n_symmetric_pairs = 0
        else
            # oversized-cell fallback selection reads per-cell body counts, so the
            # symmetric compaction refreshes every step
            ka_compact_symmetric_pairs!(ctx, hctx, grid.cell_ranges, n_direct,
                _ka_radix_setting(:SYMMETRIC_CUDA_MAX_CELL_BODIES, 128); workgroup)
        end
        # Occupancy-epoch window cache. CUDA arms this only for the dense FUSED
        # apply (`_cuda_windows_cacheable`, cuda:7353) because its GEMM reference
        # driver additionally consumes per-window class starts/counts. KA's dense
        # apply IS that reference driver, so dense stays uncached here -- but the
        # CONCAT apply consumes only (class, source, target), which is exactly what
        # the cache stores, and it takes no level argument because the level is
        # already baked into the class. So concat is cacheable on KA even though it
        # is not on CUDA, and the whole epoch collapses to one apply.
    _utick!(:symmetric, backend)
        if _ka_radix_setting(:CUDA_CACHED_WINDOWS, true) && !hctx.win_valid &&
                hctx.apply_plan isa FastMultipole.ResidentM2LConcatPlan
            ka_hier_cache_windows!(ctx, hctx, grid; workgroup)
        end
        n_routes = hctx.win_valid ? hctx.total_routes : 0

    _utick!(:cache_windows, backend)
        if occ_changed
            ka_refresh_resident_stage_groups!(ctx.workspace, grid, cache.level_offsets,
                ell, cache.root_level; workgroup)
        end
    end
    _utick!(:stage_groups, backend)
    KA.synchronize(backend)

    counts = ctx.counts
    counts.n_bodies = n
    counts.n_cells = n_cells
    counts.n_nodes = n_nodes
    counts.n_routes = n_routes
    counts.n_direct = n_direct

    if cache.state === nothing
        # every array here is persistent; the wrapper is built once and
        # refreshed in place on later steps
        cache.state = FastMultipole.DeviceResidentRadixState{TF,FastMultipole.CompressedComplexBasis,LH}(
            grid, hctx, ctx.source_bodies, ctx.source_bodies,
            grid.perm, grid.body_system, grid.body_index,
            ctx.host_perm, ctx.host_body_system, ctx.host_body_index,
            nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing,
            grid.cell_centers, grid.cell_ranges,
            ctx.m2m_parent_routes, ctx.m2m_child_routes,
            ctx.l2l_parent_routes, ctx.l2l_child_routes,
            ctx.multipoles, ctx.locals,
            ctx.route_levels, ctx.route_offsets, ctx.route_targets, ctx.route_sources,
            ctx.direct_targets, ctx.direct_sources, ctx.output,
            ctx.invariant, ctx.workspace, counters, cache.options, counts;
            sfs=ctx.sfs_ctx,
        )
    end
    cache.step += 1
    return cache
end

ka_update_radix_state!(cache::FastMultipole.RadixFMMCache, systems; kwargs...) =
    ka_update_radix_state!(cache, FastMultipole.to_tuple(systems); kwargs...)

#------- device-resident cache construction: KA port of
#         `_radix_cache_device_build` (src/translate_batched_cuda.jl:6289) -------#
#
# The last CUDA-only link in the chain. `RadixFMMCache(...; device=true)` routes
# to the CUDA builder, which allocates the whole ctx NamedTuple from `CUDA.zeros`
# and the hierarchical context from `_build_cuda_hierarchical_context`, so no
# device cache could be constructed on a non-CUDA backend at all -- which left
# `ka_finalize_radix_output!` and `ka_radix_cache_device_step!` load-checked but
# ungated. This builder closes that: it produces a real `RadixFMMCache` whose
# `device_ctx` is backend-allocated, so `ka_radix_cache_device_step!` runs end to
# end and a full UJ can be gated against the host lifecycle.
#
# Scope, matching what the KA step actually implements (each omission is a guard,
# not a silent fallback):
#   * hierarchical stencil policy only -- `ka_update_radix_state!` refuses a flat
#     cache, and FLOWVPM's `window_classes` cache is hierarchical anyway
#     (see the block comment above `ka_update_radix_state!`).
#   * concatenated M2L only -- `ka_radix_cache_workspace` pins
#     `ConcatenatedFixedZM2L`/`MaterializedYRotationM2L`, the same pair the CUDA
#     builder selects for a non-specialized hierarchical cache. A dense or
#     precomputed-y strategy would silently get a different plan, so it throws.
#   * no SFS pass, no adaptive octree mirror.
#
# Two allocation differences from the CUDA original, both inert on this path:
# the pinned host mirrors become plain `Array`s (pinning is a CUDA transfer
# optimization); `CUDA.enable_synchronization!` has no KA analogue and no
# side-stream exists here to need it. The nearfield bin context is likewise
# not built: `ka_launch_nearfield!` runs the unbinned functor kernel and never
# reads `hctx.nearfield`, so a `PartitionedVortex` cache is correct without it
# (binning is locality only).
function ka_radix_cache_device_build(backend, sources::Tuple, P::Int, ell::Int,
        x_min::SVector{3,TF}, h0::TF, maxn::Int,
        options::FastMultipole.CUDARadixLifecycleOptions,
        stencil_policy, accepted::Vector{SVector{3,Int}},
        rejected::Vector{SVector{3,Int}}, max_cells::Int, max_nodes::Int,
        route_capacity::Int, direct_capacity::Int,
        basis_info::FastMultipole.OperatorBasisInfo{B,LH}, ::Val{LH};
        hierarchical_tables=nothing,
        class_level::Vector{Int32}=Int32[],
        class_offset::Matrix{Int32}=Matrix{Int32}(undef, 3, 0),
        hierarchical_level_class_of::Array{Int32,3}=Array{Int32}(undef, 0, 0, 0),
        hierarchical_level_radii2::Vector{Int}=Int[],
        max_level_nodes::Int=0, hessian::Bool=false,
        ell_axes::SVector{3,Int}=SVector(ell, ell, ell),
        box_extent::SVector{3,TF}=SVector{3,TF}(2 * h0, 2 * h0, 2 * h0),
        root_level::Int=0, first_m2l_level::Int=2,
        # task 048 SFS: FLOWVPM arms SFS STORAGE unconditionally at cache
        # construction and gates execution/delivery on the PER-EVALUATION `sfs`
        # flag (FLOWVPM_fmm_radix.jl:550-553), so the accumulators are allocated
        # here whenever the cache is armed; `ka_radix_cache_device_step!` runs
        # the pass only for an evaluation that asks for it.
        sfs::Bool=false, sfs_transposed::Bool=true, sfs_active_row::Int=0,
        workgroup=KA_AUTO_WORKGROUP) where {TF,B,LH}
    stencil_policy isa FastMultipole.HierarchicalRigidStencil || throw(ArgumentError(
        "ka_radix_cache_device_build covers the hierarchical stencil path only; " *
        "got $(typeof(stencil_policy))"))
    hierarchical_tables isa FastMultipole.RigidHierarchicalTables || throw(ArgumentError(
        "the hierarchical policy requires the rigid hierarchical tables"))
    options.m2l_strategy isa FastMultipole.PrecomputedFactoredYM2L && throw(ArgumentError(
        "the KA device cache builds the concatenated or dense hierarchical plan; " *
        "m2l_strategy=$(typeof(options.m2l_strategy)) has no KA plan"))
    dense_strategy = options.m2l_strategy isa FastMultipole.DenseTranslationM2L ?
        options.m2l_strategy : nothing

    _z(T, dims...) = fill!(KA.allocate(backend, T, dims...), zero(T))

    counters = FastMultipole.CUDARadixTransferCounters()
    multipoles = _ka_flat_buffer(backend, TF, basis_info, max_nodes)
    locals = _ka_flat_buffer(backend, TF, basis_info, max_nodes)
    invariant = FastMultipole.OperatorInvariantCache(TF, basis_info)
    workspace = ka_radix_cache_workspace(backend, TF, basis_info, ell, h0,
        max_cells, max_nodes, route_capacity, accepted, invariant;
        ell_axes, first_level=root_level)

    # capacity-sized persistent grid: counts bound the valid prefixes, so
    # recurring steps refresh these arrays in place and never reallocate
    grid = FastMultipole.DeviceRadixGrid(
        x_min, h0, ell, 0, 0,
        _z(Int, maxn), _z(Int, maxn),
        _z(UInt64, max_cells), _z(Int, 2, max_cells),
        _z(Int, maxn), _z(Int, maxn),
        _z(TF, 3, max_cells),
        _z(Int, max_nodes), _z(UInt64, max_nodes),
        _z(Int, 3, max_nodes), _z(TF, 3, max_nodes),
        _z(Int, max_nodes), _z(Int, 2, max_nodes),
        _z(Int, max_cells),
    )
    n_edges_capacity = max(max_nodes - 1, 0)

    # per-system source staging: host-resident systems get a host buffer plus a
    # persistent device buffer (one upload per step); device-resident systems get
    # a persistent device buffer their `source_to_buffer!` overload fills in place
    host_stagings = Tuple(
        FastMultipole.residency(system) isa FastMultipole.HostResident ?
            Matrix{TF}(undef, FastMultipole.data_per_body(system), maxn) : nothing
        for system in sources)
    device_sources = Tuple(
        _z(TF, FastMultipole.data_per_body(system), maxn) for system in sources)

    # the hierarchical path keeps its stencil tables on the device hierarchical
    # context instead of a flat accepted/rejected pair, so these stay empty --
    # as do the leaf-only flat occupancy and route flag storage
    d_accepted = KA.zeros(backend, Int32, 3, 0)
    d_rejected = KA.zeros(backend, Int32, 3, 0)
    occupancy = FastMultipole.RadixLevelOccupancy(ell;
        max_bytes=stencil_policy.dense_occupancy_max_bytes,
        max_dense_ell=stencil_policy.dense_occupancy_max_ell)
    isempty(occupancy.node_at) && throw(ArgumentError(
        "the device hierarchical stencil requires the dense per-level occupancy " *
        "lookup, but ell=$ell exceeds the configured budget " *
        "(dense_occupancy_max_bytes=$(stencil_policy.dense_occupancy_max_bytes), " *
        "dense_occupancy_max_ell=$(stencil_policy.dense_occupancy_max_ell)); the " *
        "host Morton binary-search fallback has no device implementation"))
    # DEVIATION (dense). CUDA builds the workspace ITSELF with the dense
    # strategy (cuda:6331, `workspace_strategy`), so `workspace.m2l_concat`
    # becomes the dense plan and `hctx.apply_plan` is that same object. Routing
    # the plan through `_radix_cache_workspace` would need a backend the dense
    # builder's signature does not carry, so KA keeps the concat workspace and
    # builds the dense plan alongside it, handing it to the context directly.
    # Behaviourally identical on the dense path -- nothing reads
    # `workspace.m2l_concat` once `apply_plan` is dense -- at the cost of one
    # unused concat operator table.
    #
    # Classes are the UNSCALED push offsets built at the leaf cell width
    # (cuda:6326), matching `class_base = 0` in `ka_hierarchical_m2l!`.
    cell_width = (2 * h0) / (1 << ell)
    apply_plan, dense_scales = if dense_strategy === nothing
        (workspace.m2l_concat, nothing)
    else
        plan = ka_build_dense_m2l_plan(backend, TF, basis_info,
            hierarchical_tables.push_offsets, cell_width, route_capacity,
            max_cells, 1 << ell, dense_strategy, invariant)
        (plan, _ka_hier_dense_scales(TF, basis_info, ell, plan.ndof, first_m2l_level))
    end
    hierarchical_ctx = ka_hierarchical_context(TF, backend, hierarchical_tables,
        class_level, class_offset, accepted, hierarchical_level_class_of,
        hierarchical_level_radii2, apply_plan, ell, first_m2l_level,
        max_level_nodes, occupancy; window_classes=stencil_policy.window_classes,
        dense_scales)

    dpb = maximum(FastMultipole.data_per_body(system) for system in sources)
    n_output_rows = hessian ? 13 : 4
    ctx = (;
        multipoles, locals, workspace, invariant, counters, grid,
        counts=FastMultipole.RadixStepCounts(0, 0, 0, 0, 0),
        # + 1 row of 1/sigma for a regularized kernel (see `_ka_nf_inv_sigma_row`)
        source_bodies=_z(TF, dpb + (_ka_kernel_sigma_row(options.direct_kernel) > 0), maxn),
        output=_z(TF, n_output_rows, maxn),
        cell_at=_z(Int32, 0, 0, 0),
        hierarchical_ctx,
        d_accepted, d_rejected, class_chunk=1,
        route_levels=_z(Int, route_capacity),
        route_offsets=_z(Int, 3, route_capacity),
        route_targets=_z(Int, route_capacity),
        route_sources=_z(Int, route_capacity),
        direct_targets=_z(Int, direct_capacity),
        direct_sources=_z(Int, direct_capacity),
        route_flags=_z(Int32, 0),
        route_prefix=_z(Int32, 0),
        direct_flags=_z(Int32, direct_capacity),
        direct_prefix=_z(Int32, direct_capacity),
        # grid-update scratch: persistent, so the recurring step allocates
        # nothing beyond the backend's own sort scratch
        positions=_z(TF, 3, maxn),
        keys=_z(UInt64, maxn),
        sorted_keys=_z(UInt64, maxn),
        # bounded counting-sort domain buffers, sized by the same rule as the
        # CUDA builder (translate_batched_cuda.jl:6462): the full 2^(3ell) key
        # domain when the fast path is enabled for this `ell`, else length 1,
        # which is itself the signal `ka_counting_sort_ready` falls back on
        counting_histogram=_z(Int32, ka_counting_sort_enabled(ell) ? 1 << (3 * ell) : 1),
        counting_prefix=_z(Int32, ka_counting_sort_enabled(ell) ? 1 << (3 * ell) : 1),
        counting_cursor=_z(Int32, ka_counting_sort_enabled(ell) ? 1 << (3 * ell) : 1),
        body_flags=_z(Int, maxn),
        body_prefix=_z(Int, maxn),
        subsort_keys=_z(UInt32, maxn),
        cell_coords=_z(Int, 3, max_cells),
        level_keys=_z(UInt64, max_cells, ell + 1),
        level_flags=_z(Int, max_cells, ell + 1),
        level_prefix=_z(Int, max_cells, ell + 1),
        level_counts=_z(Int, ell + 1),
        d_level_offsets=_z(Int, ell + 2),
        oob_flag=_z(Int32, 1),
        # occupancy-epoch snapshot: the hierarchical cache compares the sorted
        # unique leaf keys against the previous step to skip node-metadata,
        # window and direct-pair regeneration
        epoch_cell_keys=_z(UInt64, max_cells),
        epoch_flag=_z(Int32, 1),
        host_epoch_flag=zeros(Int32, 1),
        epoch_prev_n=Ref(0),
        epoch_prev_n_cells=Ref(0),
        epoch_have=Ref(false),
        m2m_parent_routes=_z(Int, n_edges_capacity),
        m2m_child_routes=_z(Int, n_edges_capacity),
        l2l_parent_routes=_z(Int, n_edges_capacity),
        l2l_child_routes=_z(Int, n_edges_capacity),
        host_stagings, device_sources,
        # host mirrors/staging for the step downloads
        host_oob=zeros(Int32, 1),
        host_scalar=zeros(Int, 1),
        host_scalar32=zeros(Int32, 1),
        host_level_counts=zeros(Int, ell + 1),
        host_perm=zeros(Int, maxn),
        host_body_system=zeros(Int, maxn),
        host_body_index=zeros(Int, maxn),
        # must track the output row count, or the prefix copyto! in
        # `ka_finalize_radix_output!` silently mis-strides
        host_output=zeros(TF, n_output_rows, maxn),
        device_target_buffers=Dict{Int,Any}(),
        sfs_ctx=sfs ?
            (; tg=_z(TF, 3, maxn), om=_z(TF, 3, maxn), q=_z(TF, 3, maxn),
               transposed=sfs_transposed, active_row=sfs_active_row) : nothing,
        host_sfs_staging=sfs ? zeros(TF, 3, maxn) : nothing,
        device_sfs_buffers=Dict{Int,Any}(),
    )
    cache = FastMultipole.RadixFMMCache{TF,LH}(
        P, ell, x_min, h0, ell_axes, box_extent, root_level, maxn, true, hessian,
        options, stencil_policy,
        accepted, rejected, max_cells, max_nodes, route_capacity, direct_capacity,
        nothing, zeros(Int32, 0, 0, 0), SVector{3,Int}[], zeros(Int, ell + 2),
        UInt64[], Int[], Int[], Int[], nothing, nothing, ctx,
        length(sources), false, 0,
        nothing, nothing, nothing, nothing,
        FastMultipole.snapshot_locked_radix_settings(),
        sfs, sfs_transposed, nothing,
    )
    ka_update_radix_state!(cache, sources; workgroup)
    cache.built = true
    return cache
end

ka_radix_cache_device_build(backend, sources, args...; kwargs...) =
    ka_radix_cache_device_build(backend, FastMultipole.to_tuple(sources), args...;
        kwargs...)


"""
    ka_radix_cache_device_step!(cache, targets, switches; sfs=false,
                                workgroup=KA_AUTO_WORKGROUP)

Backend-agnostic `_radix_cache_device_step!`: refresh the device state, run the
uniform lifecycle body, and scatter the output back into the target systems.
Runs the SFS pass and its delivery when `sfs=true`, which requires a cache
armed with `sfs=true` at construction. No CUDA dependency.
"""
function ka_radix_cache_device_step!(cache::FastMultipole.RadixFMMCache,
        targets::Tuple, switches::Tuple; sfs::Bool=false,
        workgroup=KA_AUTO_WORKGROUP, extra_targets::Tuple=(),
        extra_target_switches::Tuple=(), extra_sources::Tuple=(),
        self_induce::Bool=true)
    # construction-locked settings must not have drifted: a late flip is
    # baked-in-silently otherwise (buffers sized at construction)
    FastMultipole.verify_locked_radix_settings(cache.locked_settings)
    # Below the FMM/direct crossover the whole lifecycle is replaced by one
    # all-pairs kernel, and the grid/route refresh is skipped with it (task
    # 053). Same finalize, same SFS hook: only the U/J evaluation differs.
    direct_arm = _ka_radix_setting(:RADIX_DIRECT_ARM, false)
    # The extra-sources-only call needs no routes, but it DOES need the bodies
    # repacked and the permutation refreshed: `finalize` de-permutes through
    # the state's body metadata, and the body count changes between calls in a
    # shedding solver. `direct_only=true` here left a stale permutation and
    # scattered the result onto the wrong particles.
    ka_update_radix_state!(cache, targets; workgroup, direct_only=direct_arm)
    state = cache.state
    if !self_induce
        fill!(state.output, zero(eltype(state.output)))
    elseif direct_arm
        ka_direct_body!(state; workgroup)
    else
        ka_lifecycle_body!(state)
    end
    ka_extra_sources_into_output!(state, extra_sources; workgroup)
    # SFS is a per-evaluation option, not merely a cache capability: an
    # sfs-armed cache runs no TG/zeta kernels on the (default) sfs=false path.
    # Placed after the lifecycle body and before the U/J finalize, which is
    # CUDA's order (translate_batched_cuda.jl:6971-6979).
    if sfs
        self_induce || throw(ArgumentError(
            "sfs=true requires the self-inducing call (the SFS pass consumes " *
            "the lifecycle's direct pairs)"))
        # the SFS pass runs zeta over the U-list direct pairs, which the
        # all-pairs arm never builds; fail loudly rather than drop the term
        direct_arm && throw(ArgumentError(
            "sfs=true is not supported on the all-pairs direct arm " *
            "(radix setting :RADIX_DIRECT_ARM); the SFS pass consumes the " *
            "U-list direct pairs, which this arm does not build"))
        state.sfs === nothing && throw(ArgumentError(
            "sfs=true evaluation requires a RadixFMMCache built with sfs=true"))
        ka_launch_sfs!(state)
    end
    ka_finalize_radix_output!(state, targets; derivatives_switches=switches,
        host_output_staging=cache.device_ctx.host_output,
        target_buffers=FastMultipole._radix_cache_target_buffers!(cache, switches),
        device_target_buffers=cache.device_ctx.device_target_buffers)
    sfs && ka_finalize_radix_sfs_output!(state, targets;
        host_sfs_staging=cache.device_ctx.host_sfs_staging,
        sfs_target_buffers=FastMultipole._radix_cache_sfs_buffers!(cache, targets),
        device_sfs_buffers=cache.device_ctx.device_sfs_buffers)
    self_induce &&
        ka_extra_targets_evaluate!(state, extra_targets, extra_target_switches; workgroup)
    return cache
end

#------- extra target / source systems (src/radix_extra_systems.jl) -------#
#
# Device counterparts of `_radix_extra_sources_into_output!` and
# `_radix_extra_targets_evaluate!`: the extra systems are host objects packed
# on the host, uploaded, evaluated by a thread-per-target rectangular kernel,
# and (for extra targets) downloaded and scattered through the host
# `buffer_to_target!`. Nothing here is on the resident lifecycle's zero-copy
# contract: the extras are a few hundred bodies per call.

# thread per extra target, loop over the packed main bodies with the cache's
# nearfield functor. `out` is written (one column per thread).
@kernel function ka_extra_targets_from_main_kernel!(kernel, out, @Const(xt), nt,
        @Const(source_bodies), nbodies, ::Type{T}, ::Val{HS}) where {T,HS}
    i = @index(Global)
    ep = FastMultipole._emits_potential(kernel)
    ghv = Val(:shipped)
    @inbounds if i <= nt
        xi = xt[1, i]; yi = xt[2, i]; zi = xt[3, i]
        u = zero(T); gx = zero(T); gy = zero(T); gz = zero(T)
        h1 = zero(T); h2 = zero(T); h3 = zero(T)
        h4 = zero(T); h5 = zero(T); h6 = zero(T)
        h7 = zero(T); h8 = zero(T); h9 = zero(T)
        for j in 1:nbodies
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
        out[1, i] = ep ? u : zero(T)
        out[2, i] = gx; out[3, i] = gy; out[4, i] = gz
        if HS
            out[5, i] = h1; out[6, i] = h2; out[7, i] = h3
            out[8, i] = h4; out[9, i] = h5; out[10, i] = h6
            out[11, i] = h7; out[12, i] = h8; out[13, i] = h9
        end
    end
end

# thread per resident body (positions in rows 1:3 of the packed bodies, slot
# order), loop over an extra source's packed buffer through the source's own
# functor. ACCUMULATES into the resident output.
@kernel function ka_targets_from_extra_source_kernel!(kernel, out, @Const(xt), nt,
        @Const(source_buffer), ns, ::Type{T}, ::Val{HS}) where {T,HS}
    i = @index(Global)
    @inbounds if i <= nt
        xi = xt[1, i]; yi = xt[2, i]; zi = xt[3, i]
        u = zero(T); gx = zero(T); gy = zero(T); gz = zero(T)
        h1 = zero(T); h2 = zero(T); h3 = zero(T)
        h4 = zero(T); h5 = zero(T); h6 = zero(T)
        h7 = zero(T); h8 = zero(T); h9 = zero(T)
        for j in 1:ns
            if HS
                du, dgx, dgy, dgz, dh1, dh2, dh3, dh4, dh5, dh6, dh7, dh8, dh9 =
                    FastMultipole._extra_pair_ugh(kernel, xi, yi, zi, source_buffer, j)
                u += du; gx += dgx; gy += dgy; gz += dgz
                h1 += dh1; h2 += dh2; h3 += dh3
                h4 += dh4; h5 += dh5; h6 += dh6
                h7 += dh7; h8 += dh8; h9 += dh9
            else
                du, dgx, dgy, dgz = FastMultipole._extra_pair_ug(kernel, xi, yi, zi,
                    source_buffer, j)
                u += du; gx += dgx; gy += dgy; gz += dgz
            end
        end
        out[1, i] += u
        out[2, i] += gx; out[3, i] += gy; out[4, i] += gz
        if HS
            out[5, i] += h1; out[6, i] += h2; out[7, i] += h3
            out[8, i] += h4; out[9, i] += h5; out[10, i] += h6
            out[11, i] += h7; out[12, i] += h8; out[13, i] += h9
        end
    end
end

_ka_upload(backend, host::AbstractMatrix{TF}) where TF =
    copyto!(KA.allocate(backend, TF, size(host)), host)

function _ka_launch_extra_source!(backend, wg, out, xt, nt::Int, system, ::Type{TF},
        hs::Bool) where TF
    ns = FastMultipole.get_n_bodies(system)
    ns == 0 && return out
    buffer = _ka_upload(backend, FastMultipole._radix_extra_source_buffer(TF, system))
    kernel = FastMultipole.direct_kernel(system)
    kern = _cached_kernel(ka_targets_from_extra_source_kernel!, backend, wg)
    kern(kernel, out, xt, nt, buffer, ns, TF,
         Val(hs && FastMultipole._extra_pair_has_hessian(kernel)); ndrange=cld(nt, wg) * wg)
    return out
end

"""
    ka_extra_sources_into_output!(state, extra_sources; workgroup)

Apply every extra source system to the resident bodies, accumulating into
`state.output` in slot order (slot positions are rows 1:3 of
`state.source_bodies`), after the lifecycle body and before finalize.
"""
function ka_extra_sources_into_output!(state::FastMultipole.DeviceResidentRadixState{TF},
        extra_sources::Tuple; workgroup=KA_AUTO_WORKGROUP) where TF
    isempty(extra_sources) && return state
    n = state.counts.n_bodies
    n == 0 && return state
    hs = size(state.output, 1) >= 13
    backend = KA.get_backend(state.output)
    wg = resolve_workgroup(backend, workgroup)
    for system in extra_sources
        _ka_launch_extra_source!(backend, wg, state.output, state.source_bodies, n,
            system, TF, hs)
    end
    KA.synchronize(backend)
    return state
end

"""
    ka_extra_targets_evaluate!(state, extra_targets, switches; workgroup)

Evaluate every extra target system from the resident bodies on the device,
then scatter through the target's switch on the host.
"""
function ka_extra_targets_evaluate!(state::FastMultipole.DeviceResidentRadixState{TF},
        extra_targets::Tuple, switches::Tuple; workgroup=KA_AUTO_WORKGROUP) where TF
    isempty(extra_targets) && return state
    n = state.counts.n_bodies
    backend = KA.get_backend(state.output)
    wg = resolve_workgroup(backend, workgroup)
    dkernel = _ka_device_direct_kernel(state.options.direct_kernel, TF, 0)
    kern = _cached_kernel(ka_extra_targets_from_main_kernel!, backend, wg)
    for (system, switch) in zip(extra_targets, switches)
        hs = !isempty(FastMultipole.hessian_range(switch))
        hs && size(state.output, 1) < 13 && throw(ArgumentError(
            "hessian output requested for an extra target system but the cache " *
            "was built with hessian=false"))
        xt_h = FastMultipole._radix_extra_target_positions(TF, system)
        nt = size(xt_h, 2)
        nt == 0 && continue
        xt = _ka_upload(backend, xt_h)
        out = KA.allocate(backend, TF, hs ? 13 : 4, nt)
        if n > 0
            kern(dkernel, out, xt, nt, state.source_bodies, n, TF, Val(hs);
                 ndrange=cld(nt, wg) * wg)
        else
            fill!(out, zero(TF))
        end
        KA.synchronize(backend)
        FastMultipole._radix_scatter_extra_target!(TF, system, switch, Array(out))
    end
    return state
end



#------- stage 0: user entry (port of `fmm!`, src/fmm.jl:873-899) -------#
#
# CUDA's stage 0 is `fmm!(targets, sources, cache::RadixFMMCache)`
# (src/fmm.jl:873), whose device branch calls `_radix_cache_device_step!`
# (src/fmm.jl:898). That call resolves to CUDA's definition only because
# `translate_batched_cuda.jl` is runtime-`include`d INTO the FastMultipole
# module, overwriting the throwing stub at
# `src/translate_batched_resident.jl:3425`.
#
# DEVIATION (forced, entry symbol only). A package extension cannot use that
# mechanism: defining `FastMultipole._radix_cache_device_step!` here with the
# same signature is method piracy over a method the parent module already owns,
# and Julia's own diagnostic for it is "incremental compilation may be fatally
# broken". So this entry point carries a different NAME. Everything inside it is
# a statement-for-statement port of `fmm!`'s prologue, in CUDA's order, with the
# same argument checks, the same error text, and the same `DerivativesSwitch`
# construction. Wiring real `fmm!` dispatch requires a backend trait hook in
# `src/` (the stub would ask the cache for its backend instead of throwing);
# that is a `src/` change and is deliberately NOT made here.
#
# Two checks from `fmm!` are absent, both because the KA cache cannot reach the
# state they guard:
#   * the `sfs && !cache.sfs` check -- `ka_radix_cache_device_step!` makes the
#     equivalent check against `state.sfs`, which is what the pass actually
#     reads, and throws CUDA's message.
#   * the adaptive branch (src/fmm.jl:900-916) -- `ka_update_radix_state!`
#     throws on a non-`nothing` `cache.adaptive`.
"""
    ka_fmm!(target_systems, source_systems, cache; kwargs...)
    ka_fmm!(systems, cache; kwargs...)

KA stage 0: the backend-agnostic entry point corresponding to
`fmm!(targets, sources, cache::RadixFMMCache)` (`src/fmm.jl:873`). Validates
arguments, builds the `DerivativesSwitch` tuple, and dispatches to
`ka_radix_cache_device_step!`.
"""
function ka_fmm!(target_systems, source_systems,
        cache::FastMultipole.RadixFMMCache{TF,LH};
        scalar_potential::Bool=false, gradient::Bool=true, hessian=false,
        sfs::Bool=false, lamb_helmholtz::Union{Nothing,Bool}=nothing,
        workgroup=KA_AUTO_WORKGROUP) where {TF,LH}
    targets = FastMultipole.to_tuple(target_systems)
    sources = FastMultipole.to_tuple(source_systems)
    split = FastMultipole._split_radix_systems(cache.n_systems, targets, sources)
    hessian_v = FastMultipole.to_vector(hessian, length(targets))
    any(hessian_v) && !cache.hessian && throw(ArgumentError(
        "hessian output requested but this RadixFMMCache was built with " *
        "hessian=false (4-row output); construct RadixFMMCache(...; hessian=true)"))
    lamb_helmholtz === nothing || Bool(lamb_helmholtz) == LH || throw(ArgumentError(
        "lamb_helmholtz=$(lamb_helmholtz) conflicts with the cache's lamb_helmholtz=$LH; " *
        "the Lamb-Helmholtz channel is fixed at cache construction"))
    !FastMultipole.has_vector_potential(split.main) || LH || throw(ArgumentError(
        "source systems carry a vector potential but the cache was built with " *
        "lamb_helmholtz=false; rebuild the cache with lamb_helmholtz=true"))
    cache.device || throw(ArgumentError(
        "ka_fmm! requires a device-resident cache built by ka_radix_cache_device_build"))

    all_switches = FastMultipole.DerivativesSwitch(
        FastMultipole.to_vector(scalar_potential, length(targets)),
        FastMultipole.to_vector(gradient, length(targets)),
        hessian_v, targets)
    switches = Tuple(all_switches[i] for i in split.main_index)
    extra_switches = Tuple(all_switches[i] for i in split.extra_target_index)
    ka_radix_cache_device_step!(cache, split.main, switches; sfs, workgroup,
        extra_targets=split.extra_targets, extra_target_switches=extra_switches,
        extra_sources=split.extra_sources, self_induce=split.self_induce)
    return cache
end

ka_fmm!(systems, cache::FastMultipole.RadixFMMCache; kwargs...) =
    ka_fmm!(systems, systems, cache; kwargs...)


#------- stage 1: argument + trait validation -------#
#
# CUDA's stage 1 is the prologue of `RadixFMMCache`
# (src/translate_batched_resident.jl:2445-2519 plus the strategy check at
# :2549) -- everything the constructor decides BEFORE it looks at where the
# bodies are. It resolves four traits off the source systems (Lamb-Helmholtz,
# body type, strength dims, direct kernel), rejects every argument combination
# the radix path cannot represent, and picks the first-pass options so that
# `TF` exists for the geometry that follows.
#
# This is a statement-for-statement port in CUDA's own order, with the same
# error text. It returns the resolved values instead of assigning them into a
# constructor's local scope, because KA's stages 2-6 are not written yet and
# the caller has to thread them by hand until they are.
#
# THREE DEVIATIONS, all forced:
#
#  1. Device availability. CUDA checks `cuda_radix_available()` behind
#     `device=true` (:2510). The KA analogue is that a `KernelAbstractions`
#     backend was actually passed; a KA cache is device-resident by
#     construction, so there is no `device=false` branch to guard.
#
#  2. (RESOLVED, task 048 KA port) SFS used to be refused here because
#     `ka_radix_cache_device_build` had no `sfs_ctx` and no SFS pass. Both now
#     exist, so this is CUDA's validation (:2449-2464) statement for statement:
#     the hessian requirement, the packed-row-8 sigma requirement, and the
#     `sfs_active_row` bounds. Only two deviations remain (1 and 3).
#
#  3. The trait tail is NOT here. CUDA's `dk` checks at :2600-2626 (isbits,
#     the `AbstractRegularizedVortex` sigma_row bound, and
#     `_assert_device_kernel_policy`) read `options.direct_kernel` AFTER the
#     policy-dependent options substitution, so they belong to stage 5, not
#     stage 1. What is portable without the policy -- the per-system
#     `direct_kernel` trait agreement -- is done here, exactly as CUDA does it
#     at :2492-2498.
#
# What stage 1 does NOT do, on purpose: derive `ell`. There is no auto-`ell`
# rule in FastMultipole on either side (PIPELINE.md stage 4); it is a kwarg
# defaulting to 4, and FLOWVPM's own rule lives in the caller.

"""
    ka_validate_radix_arguments(backend, target_systems, source_systems; kwargs...)

KA stage 1: the argument and trait validation `RadixFMMCache` performs before
it computes any geometry (`src/translate_batched_resident.jl:2445-2519`).

Returns a `NamedTuple` with the resolved
`(; targets, sources, LH, BT, dk_trait, options, auto_options, TF, P, dpb, n0,
maxn, hessian)`, which the
remaining (unported) construction stages consume. Throws the same
`ArgumentError`s, with the same messages, that the host constructor throws.
"""
function ka_validate_radix_arguments(backend, target_systems, source_systems=target_systems;
        expansion_order::Integer=4,
        max_n_bodies::Union{Nothing,Integer}=nothing,
        lamb_helmholtz::Union{Nothing,Bool}=nothing,
        hessian::Bool=false,
        sfs::Bool=false,
        sfs_active_row::Int=0,
        options::Union{Nothing,FastMultipole.CUDARadixLifecycleOptions}=nothing)
    targets = FastMultipole.to_tuple(target_systems)
    sources = FastMultipole.to_tuple(source_systems)
    FastMultipole._assert_radix_targets_are_sources(targets, sources)

    # task 048: the SFS pass reads the 9-component J from the 13-row output
    # and the raw smoothing radius sigma from packed row 8
    sfs && !hessian && throw(ArgumentError(
        "RadixFMMCache(sfs=true) requires hessian=true (the SFS pass reads " *
        "the velocity Jacobian from the 13-row output)"))
    if sfs
        for system in sources
            FastMultipole.data_per_body(system) >= 8 || throw(ArgumentError(
                "RadixFMMCache(sfs=true) requires the raw smoothing radius " *
                "sigma in packed row 8; data_per_body must be >= 8 " *
                "(got $(FastMultipole.data_per_body(system)) for $(typeof(system)))"))
        end
        sfs_active_row >= 0 || throw(ArgumentError(
            "sfs_active_row must be zero (all bodies active) or a positive packed row"))
        sfs_active_row == 0 ||
            all(FastMultipole.data_per_body(system) >= sfs_active_row for system in sources) ||
            throw(ArgumentError("sfs_active_row=$sfs_active_row exceeds data_per_body for an SFS source system"))
    end

    LH = lamb_helmholtz === nothing ?
        FastMultipole.has_vector_potential(sources) : Bool(lamb_helmholtz)

    # B2M element resolution (task 032): one shared body type per cache.
    BT = FastMultipole.body_type(first(sources))
    for system in sources
        FastMultipole.body_type(system) === BT || throw(ArgumentError(
            "all source systems sharing a RadixFMMCache must report the same " *
            "body_type; got $(FastMultipole.body_type(system)) and $BT"))
        FastMultipole.strength_dims(system) == FastMultipole.strength_dims(first(sources)) ||
            throw(ArgumentError(
            "all source systems sharing a RadixFMMCache must report the same " *
            "strength_dims (the packed strength rows 5:4+strength_dims are shared)"))
    end
    if BT <: FastMultipole.Point{FastMultipole.Vortex} && !LH
        throw(ArgumentError(
            "Point{Vortex} sources require the Lamb-Helmholtz channel; construct " *
            "the cache with lamb_helmholtz=true (or leave it to be inferred from " *
            "has_vector_potential)"))
    end

    # Nearfield kernel resolution (task 032 stage 2): one shared functor per
    # cache. The post-options checks on it are stage 5 -- DEVIATION 3.
    dk_trait = FastMultipole.direct_kernel(first(sources))
    for system in sources
        FastMultipole.direct_kernel(system) == dk_trait || throw(ArgumentError(
            "all source systems sharing a RadixFMMCache must report the same " *
            "direct_kernel; got $(FastMultipole.direct_kernel(system)) and $dk_trait"))
    end

    # Measured defaults (024/028). Precision depends only on the expansion order
    # and is needed for the bounds and stencil tolerance in stage 2; the strategy
    # also depends on the class count, so it is re-resolved once the policy is
    # built (stage 5).
    auto_options = options === nothing
    if auto_options
        options = FastMultipole.CUDARadixLifecycleOptions(;
            precision=FastMultipole._default_radix_precision(Int(expansion_order)),
            m2l_strategy=FastMultipole.ConcatenatedFixedZM2L())
    end
    TF = options.precision

    # DEVIATION 1 (see above): the KA analogue of `cuda_radix_available()`.
    backend isa KA.Backend || throw(ArgumentError(
        "the KA radix path requires a KernelAbstractions backend; got " *
        "$(typeof(backend))"))

    for system in sources
        FastMultipole.data_per_body(system) >= 4 + FastMultipole.strength_dims(system) ||
            throw(ArgumentError("the radix path packs bodies as [x, y, z, radius, " *
                "strength..., extras...]; data_per_body(system) must be >= " *
                "4 + strength_dims(system)"))
    end
    dpb = maximum(FastMultipole.data_per_body(system) for system in sources)

    n0 = FastMultipole.get_n_bodies(sources)
    n0 > 0 || throw(ArgumentError("RadixFMMCache requires at least one body"))
    maxn = max_n_bodies === nothing ? n0 : Int(max_n_bodies)
    maxn >= n0 || throw(ArgumentError(
        "max_n_bodies=$maxn is smaller than the current body count $n0"))

    # Out of source order (CUDA runs this at :2549, after the geometry block)
    # but argument-only: the hierarchical path routes the concatenated and
    # grouped-factored selections through the bounded concat engine, which would
    # otherwise silently accept a strategy the flat path rejects (task 027).
    #
    # DEVIATION. CUDA also accepts `PrecomputedFactoredYM2L` here; KA does not.
    # That plan's per-step refresh (`_cuda_refresh_precomputed_y_m2l_routes!`,
    # cuda:4047) has no KA counterpart and `ka_radix_cache_workspace` pins
    # `ConcatenatedFixedZM2L`, so no `ResidentM2LPrecomputedYPlan` is ever
    # constructible on this path. It was refused only at
    # `ka_radix_cache_device_build`, several stages downstream, which let a
    # `:precomputed_y` setting (a legal `RadixFMMSettings` value in FLOWVPM,
    # FLOWVPM_fmm_radix.jl:243) past the stage that is meant to state KA's
    # envelope. The builder keeps its own guard for callers that bypass
    # validation.
    options.m2l_strategy isa Union{FastMultipole.ConcatenatedFixedZM2L,
        FastMultipole.DenseTranslationM2L} ||
        throw(ArgumentError(
        "the KA RadixFMMCache supports ConcatenatedFixedZM2L or " *
        "DenseTranslationM2L; got $(typeof(options.m2l_strategy)) (CUDA also " *
        "accepts PrecomputedFactoredYM2L, which has no KA plan; the " *
        "SharedRotationM2L group layout is not refreshable in place)"))

    return (; targets, sources, LH, BT, dk_trait, options, auto_options, TF,
        P=Int(expansion_order), dpb, n0, maxn, hessian)
end



#------- stage 2: root geometry -------#
#
# CUDA's stage 2 is the geometry block of `RadixFMMCache`
# (src/translate_batched_resident.jl:2521-2535): the first point at which the
# constructor looks at WHERE the bodies are. It turns the caller's `bounds`
# (or, absent them, the body extent) into the four root-box quantities every
# later stage indexes against -- `x_min`, `h0`, `ell_axes` and `box_extent`.
#
# This is a statement-for-statement port in CUDA's own order, sharing CUDA's
# own helpers: `_radix_bounds` (src/tree_batched.jl:117/137) and
# `_resolve_radix_ell_axes` (:2190/:2196) are backend-independent host code
# that computes host scalars, so a KA copy of them would be duplication, not a
# port. Only the surrounding branch is reproduced here.
#
# ONE DEVIATION, and it is a subtraction: CUDA's derived branch also names
# `center` and `box`, which are consumed only by the two lines below them and
# are dead at the end of the block. They are not returned.
#
# WHY THERE IS NO DEVICE REDUCTION. The `bounds === nothing` branch walks the
# bodies one at a time through `get_position`, which is scalar indexing if the
# system is backed by device arrays. That is CUDA's behavior too, not a KA
# regression, and the production path never reaches it: FLOWVPM always passes
# an explicit `bounds` (its own padded, optionally center-snapped box --
# FLOWVPM_fmm_radix.jl:521-523, :539, :560-562), so the constructor takes the
# `else` branch, which touches no body data at all. A device min/max reduction
# would be a new capability on a path production does not use; if the derived
# branch ever becomes hot for a device-resident field, that is its own task.

"""
    ka_radix_geometry(sources, TF, ell; bounds=nothing, bounds_margin=0.05)

KA stage 2: the root-box geometry `RadixFMMCache` derives immediately after
argument validation (`src/translate_batched_resident.jl:2521-2535`).

With `bounds === nothing` the root cube is the body bounding box inflated by
`bounds_margin`; otherwise `bounds = (x_min, box_size)` is taken as given and
`box_size` is resolved into per-axis depths. Returns a `NamedTuple`
`(; x_min, h0, ell_axes, box_extent)`. `TF` and `ell` come from stage 1 and the
caller respectively.
"""
function ka_radix_geometry(sources, ::Type{TF}, ell::Integer;
        bounds=nothing, bounds_margin::Real=0.05) where TF
    if bounds === nothing
        x_min_data, x_max_data = FastMultipole._radix_bounds(sources, TF)
        center = (x_min_data + x_max_data) * TF(0.5)
        box = (x_max_data - x_min_data) * TF(0.5)
        h0 = max(box[1], box[2], box[3]) * (1 + TF(bounds_margin))
        h0 > zero(TF) || throw(ArgumentError(
            "bodies are degenerate (zero extent); pass explicit bounds=(x_min, box_size)"))
        x_min = center - SVector{3,TF}(h0, h0, h0)
        ell_axes = SVector(Int(ell), Int(ell), Int(ell))
        box_extent = SVector{3,TF}(2 * h0, 2 * h0, 2 * h0)
    else
        x_min = SVector{3,TF}(bounds[1])
        ell_axes, h0, box_extent =
            FastMultipole._resolve_radix_ell_axes(bounds[2], Int(ell), TF)
    end
    return (; x_min, h0, ell_axes, box_extent)
end


#------- stage 5: stencil policy, hierarchical tables, options finalization -------#
#
# CUDA's stage 5 is the constructor's policy block
# (src/translate_batched_resident.jl:2546-2578) plus the options/`dk` tail
# (:2594-2626): it picks the stencil policy, builds and verifies the
# hierarchical level schedule, produces the accepted/rejected offset sets every
# later stage sizes and routes against, and only then -- once the class count
# exists -- finalizes `options` and runs the kernel checks stage 1 had to defer.
#
# It consumes stage 1's NamedTuple and stage 2's geometry directly, because
# that is what the constructor's local scope hands these statements.
#
# THE ONE REORDERING. CUDA runs the capacity block (:2580-2592, KA stage 6)
# BETWEEN the tables and the options tail. The two are independent -- the
# capacities read `ell`, `ell_axes`, `maxn`, `stencil_policy.window_classes`
# and accepted/rejected, none of which the options tail touches, and the
# options tail reads `length(accepted)` and `basis_info`, neither of which the
# capacities produce -- so KA runs the tail here and leaves the capacities
# whole for stage 6. Nothing observable depends on the order.
#
# TWO DEVIATIONS, both refusals in the style of stage 1's `sfs`:
#
#  1. Adaptive octree. CUDA's `adaptive !== nothing` block (:2628-2679) guards
#     a host- and CUDA-only lifecycle that KA does not implement at all.
#     `adaptive` is refused outright rather than validated.
#
#  2. `_assert_device_kernel_policy` is called with `device=true`
#     unconditionally: a KA cache is device-resident by construction, exactly
#     as in stage 1's deviation 1. The `device` argument to
#     `_default_radix_policy` is `true` for the same reason, which is what
#     selects `RADIX_DEVICE_WINDOW_CLASSES` as the default window width.
#
# `_default_radix_policy`, `_hierarchical_scheduled_tables`,
# `_verify_hierarchical_classifier!`, `_hierarchical_class_metadata`,
# `classify_radix_stencil_offsets` and `_default_radix_options` are shared host
# code producing host tables -- as in stage 2, a KA copy would be duplication
# rather than a port, so they are called, not reimplemented.

"""
    ka_radix_stencil_policy(v, ell, h0, ell_axes; kwargs...)

KA stage 5: the stencil policy, hierarchical level schedule and options
finalization `RadixFMMCache` performs after the geometry
(`src/translate_batched_resident.jl:2546-2578`, `:2594-2626`).

`v` is stage 1's NamedTuple; `h0` and `ell_axes` come from stage 2. The
`policy` / `stencil_epsilon` / `near_radius2` / `window_classes` /
`level_radii2` kwargs are the constructor's own, with the constructor's
meanings. Returns a `NamedTuple` carrying the policy, the hierarchical tables
and their metadata, the accepted/rejected offset sets, `basis_info`, and the
finalized `options` / `dk`.
"""
function ka_radix_stencil_policy(v, ell::Integer, h0, ell_axes;
        policy=nothing, stencil_epsilon=nothing, near_radius2=nothing,
        window_classes=nothing, level_radii2=nothing, adaptive=nothing)
    TF, LH, BT = v.TF, v.LH, v.BT
    P = v.P

    # DEVIATION 1 (see above): KA has no adaptive octree lifecycle.
    adaptive === nothing || throw(ArgumentError(
        "the KA radix path has no adaptive octree lifecycle (tasks 039-041 are " *
        "host- and CUDA-only); construct with adaptive=nothing"))

    # DEVIATION 2: device=true, a KA cache being device-resident by construction.
    stencil_policy = FastMultipole._default_radix_policy(policy, P, TF, LH, h0,
        Int(ell), true, stencil_epsilon, near_radius2, window_classes, level_radii2)
    hierarchical = stencil_policy isa FastMultipole.HierarchicalRigidStencil
    # Active-level trimming (task 037 stage 3): hierarchical caches retain node
    # levels root_level:ell and run M2L on levels first_m2l_level:ell. Cubic
    # caches degenerate to root_level = 1 with an empty flat-top; flat-policy
    # caches stay untrimmed (root_level = 0).
    if hierarchical
        hierarchical_tables, hierarchical_level_class_of, hierarchical_level_radii2,
            root_level, first_m2l_level =
            FastMultipole._hierarchical_scheduled_tables(stencil_policy, Int(ell), ell_axes)
        FastMultipole._verify_hierarchical_classifier!(h0, Int(ell), stencil_policy,
            hierarchical_tables, ell_axes, root_level, first_m2l_level,
            hierarchical_level_radii2)
        class_level, class_offset, effective_offsets =
            FastMultipole._hierarchical_class_metadata(hierarchical_tables, Int(ell),
                first_m2l_level)
        accepted, rejected = effective_offsets, hierarchical_tables.near_offsets
    else
        hierarchical_tables = nothing
        hierarchical_level_class_of = Array{Int32}(undef, 0, 0, 0)
        hierarchical_level_radii2 = Int[]
        root_level, first_m2l_level = 0, 2
        class_level, class_offset, effective_offsets =
            Int32[], Matrix{Int32}(undef, 3, 0), SVector{3,Int}[]
        accepted, rejected =
            FastMultipole.classify_radix_stencil_offsets(h0, Int(ell), stencil_policy.config)
    end

    basis_info = FastMultipole.OperatorBasisInfo(
        FastMultipole.CompressedComplexBasis(), P, Val(LH))

    options = v.options
    if v.auto_options
        options = FastMultipole._default_radix_options(TF, P, LH, true,
            length(accepted), FastMultipole._dense_m2m_dof(basis_info, Val(LH)))
    end
    options = FastMultipole._options_with_body_type(options, BT)
    if v.dk_trait != FastMultipole._default_direct_kernel(BT)
        # explicit trait choice; a conflicting explicit options choice is an error
        (options.direct_kernel == FastMultipole._default_direct_kernel(BT) ||
            options.direct_kernel == v.dk_trait) || throw(ArgumentError(
            "options.direct_kernel=$(options.direct_kernel) conflicts with the " *
            "direct_kernel(system) trait $(v.dk_trait)"))
        options = FastMultipole._options_with_direct_kernel(options, v.dk_trait)
    end
    dk = options.direct_kernel
    isbits(dk) || throw(ArgumentError(
        "direct_kernel must be an isbits functor (GPU-compilable, no references); " *
        "got $(typeof(dk))"))
    if dk isa FastMultipole.AbstractRegularizedVortex
        kname = nameof(typeof(dk))
        BT <: FastMultipole.Point{FastMultipole.Vortex} || throw(ArgumentError(
            "$kname requires body_type Point{Vortex}; got $BT"))
        for system in v.sources
            dk.sigma_row <= FastMultipole.data_per_body(system) || throw(ArgumentError(
                "$kname sigma_row=$(dk.sigma_row) exceeds " *
                "data_per_body=$(FastMultipole.data_per_body(system)) for " *
                "$(typeof(system)); every source system must carry the smoothing " *
                "radius sigma in packed row sigma_row"))
        end
    end
    FastMultipole._assert_device_kernel_policy(true, dk, hierarchical)

    return (; stencil_policy, hierarchical, hierarchical_tables,
        hierarchical_level_class_of, hierarchical_level_radii2,
        root_level, first_m2l_level, class_level, class_offset,
        accepted, rejected, basis_info, options, dk)
end


#------- stage 6: capacity sizing -------#
#
# CUDA's stage 6 is the constructor's capacity block
# (src/translate_batched_resident.jl:2580-2592): the five numbers that fix
# every persistent device allocation the cache will ever make. They are sized
# to `maxn` (the capacity contract: live `np` may vary below it with no
# reallocation), not to the live body count.
#
# A statement-for-statement port with NO deviations. It reads stage 1's `maxn`,
# stage 2's `ell_axes` and stage 5's policy/offset sets, and computes nothing
# device-side -- these are host integers.
#
# This is the stage the pipeline note called ungated by construction: until it
# existed, every suite handed KA known-good capacities copied off a host cache,
# so no gate could catch a KA sizing bug because KA did no sizing.

"""
    ka_radix_capacities(v, ell, ell_axes, s)

KA stage 6: the five persistent capacities `RadixFMMCache` derives from the
resolved policy (`src/translate_batched_resident.jl:2580-2592`).

`v` is stage 1's NamedTuple, `ell_axes` stage 2's, and `s` stage 5's. Returns
`(; max_cells, max_nodes, max_level_nodes, route_capacity, direct_capacity)`.
"""
function ka_radix_capacities(v, ell::Integer, ell_axes, s)
    L_max = Int(ell)
    max_cells = FastMultipole._radix_level_node_capacity(L_max, ell_axes, L_max, v.maxn)
    max_nodes = sum(FastMultipole._radix_level_node_capacity(L, ell_axes, L_max, max_cells)
        for L in s.root_level:L_max)
    # init=0 covers the zero-M2L degenerate hierarchy (first_m2l_level == ell+1,
    # empty range -- task 052c): no M2L level, so no per-level node bound needed.
    max_level_nodes = L_max >= 2 ? maximum(
        (FastMultipole._radix_level_node_capacity(L, ell_axes, L_max, max_cells)
         for L in (s.hierarchical ? s.first_m2l_level : 2):L_max); init=0) : 0
    route_capacity = s.hierarchical ?
        min(min(s.stencil_policy.window_classes,
                length(s.hierarchical_tables.push_offsets)) * max_level_nodes,
            max_level_nodes * max_level_nodes) :
        min(length(s.accepted), max_cells) * max_cells
    direct_capacity = max_cells * min(length(s.rejected), max_cells)
    return (; max_cells, max_nodes, max_level_nodes, route_capacity, direct_capacity)
end


#------- precision-parameterized device functors (KA-only) -------#
#
# WHY THIS EXISTS. `PartitionedVortex`, `RegularizedVortex` and `TwoPassVortex`
# store their cutoffs as HARD `Float64` fields -- `rho_t::Float64` /
# `rho_c::Float64`, force-coerced by the inner constructors
# (src/containers.jl:2031, :2068, :2090). The direct-pair kernels take the
# functor BY VALUE for compile-time specialization, so those fields cross into
# device code as doubles.
#
# On CUDA that is invisible: H100/H200 have native FP64 units, the field loads
# and the comparison simply execute in double. Metal has no Float64 at all, so
# the same IR fails to compile outright:
#
#   InvalidIRError: ... gpu_ka_direct_pairs_functor_kernel!(::PartitionedVortex,
#   ...) resulted in invalid LLVM IR / Reason: unsupported use of double value
#
# The per-pair ARITHMETIC is already precision-generic and needs no change:
# `_direct_pair_ug`/`_ugh` convert with `T(_pass1_regularized_cutoff(kernel))`
# (src/translate_batched_resident.jl:843, :861), `_gaussianerf_g_h` converts
# every constant with `T(...)`, and Float32 series coefficients already exist
# (`_gausserf_series_g(z::Float32)`, :686). The double is materialized by the
# FIELD LOAD, which happens before any of that -- so no use-site conversion can
# remove it. The type has to arrive on the device already carrying `TF`.
#
# These mirrors do exactly that and nothing else. The precision follows what the
# particle field passes in (`options.precision`, i.e. FLOWVPM's
# `RadixFMMSettings.precision`), so a Float64 field on a Float64-capable backend
# still runs in Float64 -- this is not a downcast, it is the removal of a
# HARDCODED Float64 from a type that should always have been parameterized.
#
# NOTHING ON THE CUDA PATH IS TOUCHED. `src/containers.jl` and
# `src/translate_batched_resident.jl` are unmodified; CUDA keeps building and
# consuming the stock `PartitionedVortex`. Conversion happens once on the host,
# in `_ka_device_direct_kernel`, at cache build.
#
# The duplicated code is the ~10-line functor WRAPPER only. All real math --
# `_gaussianerf_g_h`, `_vortex_pair_ug`, `_vortex_pair_ugh` -- is called
# straight out of FastMultipole, so the numerics cannot drift from the host
# oracle: same functions, same order, only the cutoff's storage type differs.

"""
    KAPartitionedVortex{TF}(sigma_row, rho_t)

Device mirror of `PartitionedVortex` with the cutoff stored as `TF` instead of
a hardcoded `Float64`. See the block comment above.
"""
struct KAPartitionedVortex{TF} <: FastMultipole.AbstractDirectKernel
    sigma_row::Int
    rho_t::TF
    inv_sigma_row::Int
end

"""
    KATwoPassVortex{TF}(sigma_row, rho_t, rho_c)

Device mirror of `TwoPassVortex`; pass 1 branches at `rho_c`, matching
`_pass1_regularized_cutoff(::TwoPassVortex)` (src/translate_batched_resident.jl:831).
"""
struct KATwoPassVortex{TF} <: FastMultipole.AbstractDirectKernel
    sigma_row::Int
    rho_t::TF
    rho_c::TF
    inv_sigma_row::Int
end

"""
    KARegularizedVortex{TF}(sigma_row, rho_t)

Device mirror of `RegularizedVortex`.
"""
struct KARegularizedVortex{TF} <: FastMultipole.AbstractDirectKernel
    sigma_row::Int
    rho_t::TF
    inv_sigma_row::Int
end

const KARegularizedFunctor{TF} =
    Union{KAPartitionedVortex{TF},KATwoPassVortex{TF},KARegularizedVortex{TF}}

# trait parity with src/containers.jl:2132
FastMultipole._emits_potential(::KARegularizedFunctor) = false

# pass-1 cutoff, mirroring src/translate_batched_resident.jl:830-831 exactly:
# PartitionedVortex branches at rho_t, TwoPassVortex at rho_c.
@inline _ka_pass1_cutoff(k::KAPartitionedVortex) = k.rho_t
@inline _ka_pass1_cutoff(k::KARegularizedVortex) = k.rho_t
@inline _ka_pass1_cutoff(k::KATwoPassVortex) = k.rho_c

"""
    _ka_device_direct_kernel(kernel, ::Type{TF}) -> device functor

Host-side conversion, called once at cache build. Singular kernels carry no
float fields and are returned unchanged; the regularized family is rebuilt with
`TF` cutoffs.
"""
_ka_device_direct_kernel(k::FastMultipole.AbstractDirectKernel, ::Type{TF},
    inv_sigma_row::Integer=0) where TF = k
_ka_device_direct_kernel(k::FastMultipole.PartitionedVortex, ::Type{TF},
    inv_sigma_row::Integer=0) where TF =
    KAPartitionedVortex{TF}(k.sigma_row, TF(k.rho_t), Int(inv_sigma_row))
_ka_device_direct_kernel(k::FastMultipole.RegularizedVortex, ::Type{TF},
    inv_sigma_row::Integer=0) where TF =
    KARegularizedVortex{TF}(k.sigma_row, TF(k.rho_t), Int(inv_sigma_row))
_ka_device_direct_kernel(k::FastMultipole.TwoPassVortex, ::Type{TF},
    inv_sigma_row::Integer=0) where TF =
    KATwoPassVortex{TF}(k.sigma_row, TF(k.rho_t), TF(k.rho_c), Int(inv_sigma_row))

"""
    _ka_kernel_sigma_row(kernel) -> Int

`sigma_row` for the regularized family, 0 for every kernel that carries no
sigma. Host-side only; decides whether the reciprocal-sigma row is allocated.
"""
_ka_kernel_sigma_row(::FastMultipole.AbstractDirectKernel) = 0
_ka_kernel_sigma_row(k::FastMultipole.PartitionedVortex) = k.sigma_row
_ka_kernel_sigma_row(k::FastMultipole.RegularizedVortex) = k.sigma_row
_ka_kernel_sigma_row(k::FastMultipole.TwoPassVortex) = k.sigma_row

# rho = |r|/sigma for the regularized family, plus the guard value the caller
# tests for `> 0`. With the reciprocal-sigma row wired (`inv_sigma_row > 0`) this
# is one load and one MULTIPLY; without it, the original load-and-divide. The
# branch is on a struct field, so it is uniform across every thread in the launch
# and there is exactly one compiled variant per functor type either way.
#
# The guard is exact, not approximate: the pack kernel stores 0 in the reciprocal
# row for every body with sigma <= 0, so `inv_sigma > 0` selects the same bodies
# `sigma > 0` did. `rho` differs from the divide by at most one Float32 rounding.
@inline function _ka_rho_and_guard(kernel, source_bodies, j, r2::T, invr::T) where T
    isr = kernel.inv_sigma_row
    if isr > 0
        @inbounds invsig = source_bodies[isr, j]
        return r2 * invr * invsig, invsig
    else
        @inbounds sigma = source_bodies[kernel.sigma_row, j]
        return sigma > zero(T) ? r2 * invr / sigma : zero(T), sigma
    end
end

# Port of `_direct_pair_ug(::Union{PartitionedVortex,TwoPassVortex}, ...)`
# (src/translate_batched_resident.jl:833-848). Statement for statement identical;
# the only difference is that the cutoff needs no `T(...)` because it is already
# `TF`. `_gaussianerf_g_h` and `_vortex_pair_ug` are FastMultipole's own.
@inline function FastMultipole._direct_pair_ug(kernel::KARegularizedFunctor,
        dx, dy, dz, r2, invr, source_bodies, j)
    T = typeof(r2)
    @inbounds gsx = source_bodies[5, j]
    @inbounds gsy = source_bodies[6, j]
    @inbounds gsz = source_bodies[7, j]
    rho, guard = _ka_rho_and_guard(kernel, source_bodies, j, r2, invr)
    g = one(T)
    if guard > zero(T)
        if rho <= T(_ka_pass1_cutoff(kernel))
            g, _ = FastMultipole._gaussianerf_g_h(rho)
        end
    end
    return FastMultipole._vortex_pair_ug(dx, dy, dz, invr, gsx, gsy, gsz, g)
end

# Port of `_direct_pair_ugh(...)` (src/translate_batched_resident.jl:850-866).
@inline function FastMultipole._direct_pair_ugh(kernel::KARegularizedFunctor,
        dx, dy, dz, r2, invr, source_bodies, j)
    T = typeof(r2)
    @inbounds gsx = source_bodies[5, j]
    @inbounds gsy = source_bodies[6, j]
    @inbounds gsz = source_bodies[7, j]
    rho, guard = _ka_rho_and_guard(kernel, source_bodies, j, r2, invr)
    g = one(T)
    h = -T(3)
    if guard > zero(T)
        if rho <= T(_ka_pass1_cutoff(kernel))
            g, h = FastMultipole._gaussianerf_g_h(rho)
        end
    end
    return FastMultipole._vortex_pair_ugh(dx, dy, dz, r2, invr, gsx, gsy, gsz, g, h)
end



#------- DenseTranslationM2L on KA: hierarchical dense M2L -------#
#
# KA port of the hierarchical dense M2L path
# (src/translate_batched_cuda.jl:7470-7902). FLOWVPM's shipped default is
# `m2l_strategy=:dense` (FLOWVPM_fmm_radix.jl:204), so without this the real
# `UJ_fmm` entry can only run on a non-default setting.
#
# Only the SEMANTIC core is ported. The CUDA file spends ~1500 lines on
# FP16/BF16 tensor-core operator tables, CUBLAS batched-GEMM plumbing with
# device-pointer alpha/beta, free-memory preflight gates and a fused per-route
# kernel variant. None of that is required to compute the operator: routes are
# offset-class-major and contiguous and each class carries ONE shared
# `ndof x ndof` operator, so the apply is a plain GEMM per class. The tensor
# paths, the memory gates and the fused variant are deliberately absent; the
# gather/GEMM/scatter chunk loop is `_cuda_hier_dense_apply_window!`'s
# non-fused branch statement for statement.
#
# Host oracle: `resident_m2l_batch!(::DenseTranslationM2L, ...)`
# (src/translate_batched.jl:2118), which builds the same operator per column
# and applies it as `y = K x`.
#
# DEVIATION (class geometry). Following CUDA, the operator table is built from
# the UNSCALED `tables.push_offsets` at the leaf cell width and scaled per
# level at apply time, so `class_base = 0` for dense where the concat path uses
# `(L - first_m2l_level) * noffsets` (cuda:7956).

# Copy of `_cuda_hier_dense_scales` (cuda:8059). Pure host math over host
# Matrices, but it lives in the runtime-`include`d CUDA file, so it cannot be
# called from an extension; same body, unchanged.
function _ka_hier_dense_scales(::Type{TF}, basis_info::FastMultipole.OperatorBasisInfo{B,LH},
        ell::Int, D::Int, first_m2l_level::Int=2) where {TF,B,LH}
    nlevels = max(ell - first_m2l_level + 1, 0)
    source_scale = ones(TF, D, nlevels)
    target_scale = ones(TF, D, nlevels)
    Dphi = FastMultipole.degree_major_dof(basis_info.orders.P_phi)
    @inbounds for L in first_m2l_level:ell
        col = L - first_m2l_level + 1
        s = TF(1 << (ell - L))
        for n in 0:basis_info.orders.P_phi, row in FastMultipole.degree_row_range(n)
            source_scale[row, col] = s^(-n)
            target_scale[row, col] = s^(-(n + 1))
        end
        if LH
            for n in 0:basis_info.orders.P_active, row in FastMultipole.degree_row_range(n)
                rr = Dphi + row
                source_scale[rr, col] = s^(-(n - 1))
                target_scale[rr, col] = s^(-(n + 2))
            end
        end
    end
    return source_scale, target_scale
end

"""
    ka_build_dense_m2l_plan(backend, TF, basis_info, offsets, cell_width,
                            route_capacity, max_cells, grid_resolution,
                            strategy, invariant) -> ResidentM2LDenseCUDAPlan

KA mirror of `_build_cuda_dense_m2l_plan` (cuda:4785). Every `D x D` operator is
oracle-built on the host by the SAME `build_dense_m2l_operator!` the host plan
uses, then uploaded into its column-major slice of the packed
`D x D x nclasses` device array.

`ResidentM2LDenseCUDAPlan` is reused rather than mirrored: its array fields are
type parameters (containers.jl:1859), so it holds KA arrays unchanged. The
tensor-operator fields are filled with empty arrays -- there is no KA tensor
path -- and the byte-accounting fields carry the payload sizes only.
"""
function ka_build_dense_m2l_plan(backend, ::Type{TF},
        basis_info::FastMultipole.OperatorBasisInfo{B,LH},
        accepted_offsets::AbstractVector{<:SVector{3,<:Integer}}, cell_width::Real,
        route_capacity::Integer, max_cells::Integer, grid_resolution::Integer,
        strategy::FastMultipole.DenseTranslationM2L,
        invariant::FastMultipole.OperatorInvariantCache{TF,B,LH};
        chunk::Int=4096) where {TF,B,LH}
    nroutes = Int(route_capacity)
    ncells = Int(max_cells)
    G = Int(grid_resolution)
    nclasses = length(accepted_offsets)
    D = FastMultipole._dense_m2m_dof(basis_info, Val(LH))
    class_capacities = Vector{Int}(undef, nclasses)
    @inbounds for i in eachindex(accepted_offsets)
        class_capacities[i] = FastMultipole._dense_m2l_capacity(accepted_offsets[i],
            nroutes, ncells, G)
    end
    W = max(min(chunk, nroutes), 1)

    build_width = strategy.build_chunk > 0 ? min(D, strategy.build_chunk) : D
    workspace = FastMultipole.DenseM2LBuilderWorkspace(TF, basis_info, invariant, build_width)
    Kbuf = Matrix{TF}(undef, D, D)
    ops_host = Array{TF,3}(undef, D, D, nclasses)
    @inbounds for (i, offset) in enumerate(accepted_offsets)
        delta = TF(cell_width) * SVector{3,TF}(offset)
        r, theta, phi = FastMultipole.cartesian_to_spherical(delta)
        FastMultipole.build_dense_m2l_operator!(Kbuf, r, theta, phi, invariant,
            workspace, Val(LH))
        FastMultipole._check_dense_m2l_operator_finite!(Kbuf, basis_info, offset)
        ops_host[:, :, i] .= Kbuf
    end
    _dev(A) = (d = KA.allocate(backend, eltype(A), size(A)...); copyto!(d, A); d)
    _zeros(T, dims...) = (z = KA.allocate(backend, T, dims...); fill!(z, zero(T)); z)

    d_operators = _dev(ops_host)
    empty3(T) = KA.allocate(backend, T, 0, 0, 0)
    ndof_phi = FastMultipole.degree_major_dof(basis_info.orders.P_phi)
    operator_bytes = sizeof(TF) * D * D * nclasses
    scratch_bytes = 2 * sizeof(TF) * D * W
    route_metadata_bytes = nroutes * sizeof(Int32) + nclasses * sizeof(Int32)
    return FastMultipole.ResidentM2LDenseCUDAPlan{TF,typeof(d_operators),
            typeof(empty3(Float16)),typeof(empty3(Float16)),
            typeof(KA.allocate(backend, Float32, 0, 0)),
            typeof(_zeros(Int32, nroutes)),typeof(_zeros(Int32, nclasses)),
            typeof(_zeros(TF, D, W))}(
        _zeros(Int32, nroutes), d_operators,
        empty3(Float16), empty3(Float16), KA.allocate(backend, Float32, 0, 0),
        _zeros(Int32, nclasses), zeros(Int32, nclasses), zeros(Int, nclasses + 1),
        class_capacities, _zeros(TF, D, W), _zeros(TF, D, W),
        nclasses, D, ndof_phi, W,
        operator_bytes, scratch_bytes, route_metadata_bytes,
        operator_bytes + scratch_bytes + route_metadata_bytes, 0,
        Base.RefValue{Any}(nothing),
    )
end

# Copy of `_cuda_hier_refresh_dense_window!` (cuda:7470): pure host bookkeeping
# over `hctx.host_window_cum`, rebuilding `class_starts` for the window. The
# `DENSE_CUDA_FUSED` branch is dropped -- KA has no fused variant, so the
# out-of-window sentinel fill always runs.
function ka_hier_refresh_dense_window!(plan::FastMultipole.ResidentM2LDenseCUDAPlan,
        hctx::FastMultipole.DeviceHierarchicalM2LContext, lo::Int, hi::Int, n_routes::Int)
    host_counts = plan.host_class_counts::Vector{Int32}
    @inbounds for k in hctx.window_lo:hctx.window_hi
        host_counts[k] = Int32(0)
    end
    starts = plan.class_starts
    cursor = 1
    total = 0
    @inbounds for k in lo:hi
        prev = k == lo ? 0 : Int(hctx.host_window_cum[k - lo])
        c = Int(hctx.host_window_cum[k - lo + 1]) - prev
        c <= plan.class_capacities[k] || throw(AssertionError(
            "hierarchical dense M2L class $k count $c exceeds capacity $(plan.class_capacities[k])"))
        host_counts[k] = Int32(c)
        starts[k] = cursor
        cursor += c
        total += c
    end
    total == n_routes || throw(AssertionError(
        "hierarchical dense M2L classes $lo:$hi do not partition the window's " *
        "$n_routes routes (summed $total)"))
    starts[hi + 1] = cursor
    @inbounds for k in 1:(lo - 1)
        starts[k] = 1
    end
    @inbounds for k in (hi + 2):length(starts)
        starts[k] = cursor
    end
    hctx.window_lo = lo
    hctx.window_hi = hi
    return plan
end

# Mirrors of `_cuda_hier_dense_gather_kernel!` / `_cuda_hier_dense_scatter_kernel!`
# (cuda:7732, :7750). Flat elementwise index math; the scatter accumulates with
# `@atomic` exactly as CUDA does, since several routes in a window can target
# the same cell.
@kernel function ka_hier_dense_gather_kernel!(slab, @Const(phi), @Const(chi),
        @Const(phi_flat_idx), @Const(chi_flat_idx), @Const(src_cols),
        ndof_phi, lcol, @Const(src_scale), ::Val{LH}) where LH
    idx = @index(Global)
    ndof = size(slab, 1)
    @inbounds if idx <= ndof * size(slab, 2)
        row = (idx - 1) % ndof + 1
        j = (idx - 1) ÷ ndof + 1
        col = src_cols[j]
        if row <= ndof_phi
            slab[row, j] = phi[phi_flat_idx[row], col] * src_scale[row, lcol]
        elseif LH
            slab[row, j] = chi[chi_flat_idx[row - ndof_phi], col] * src_scale[row, lcol]
        end
    end
end

@kernel function ka_hier_dense_scatter_kernel!(phi, chi, @Const(slab),
        @Const(phi_flat_idx), @Const(chi_flat_idx), @Const(tgt_cols),
        ndof_phi, lcol, @Const(tgt_scale), ::Val{LH}) where LH
    idx = @index(Global)
    ndof = size(slab, 1)
    @inbounds if idx <= ndof * size(slab, 2)
        row = (idx - 1) % ndof + 1
        j = (idx - 1) ÷ ndof + 1
        col = tgt_cols[j]
        if row <= ndof_phi
            KA.@atomic phi[phi_flat_idx[row], col] += slab[row, j] * tgt_scale[row, lcol]
        elseif LH
            KA.@atomic chi[chi_flat_idx[row - ndof_phi], col] +=
                slab[row, j] * tgt_scale[row, lcol]
        end
    end
end

# One class's GEMM: `dst[:, lo:hi] = operators[:, :, k] * src[:, lo:hi]`.
# Replaces the `CUBLAS.gemm!` call at `_cuda_dense_class_gemm!`. One thread per
# output element with a D-long inner product -- D is (P+1)^2 (36 at P=5), so the
# operator row fits in cache and this stays bandwidth-bound on the slab.
@kernel function ka_dense_class_gemm_kernel!(dst, @Const(ops), @Const(src),
        k, lo, ncols, D)
    idx = @index(Global)
    @inbounds if idx <= D * ncols
        row = (idx - 1) % D + 1
        j = (idx - 1) ÷ D + 1
        col = lo + j - 1
        acc = zero(eltype(dst))
        for t in 1:D
            acc += ops[row, t, k] * src[t, col]
        end
        dst[row, col] = acc
    end
end

function ka_dense_class_gemm!(dst, ops, k::Int, src, lo::Int, hi::Int;
        workgroup=KA_AUTO_WORKGROUP)
    ncols = hi - lo + 1
    ncols <= 0 && return dst
    D = size(ops, 1)
    backend = KA.get_backend(dst)
    n_el = D * ncols
    kernel = _cached_kernel(ka_dense_class_gemm_kernel!, backend, workgroup)
    kernel(dst, ops, src, k, lo, ncols, D; ndrange=n_el)
    return dst
end

# `_cuda_hier_dense_apply_window!` (cuda:7862), non-fused branch, statement for
# statement: chunked gather -> per-class GEMM over the classes the chunk spans
# -> scatter-add.
function ka_hier_dense_apply_window!(state::FastMultipole.DeviceResidentRadixState{TF,B,LH},
        ws, plan::FastMultipole.ResidentM2LDenseCUDAPlan,
        hctx::FastMultipole.DeviceHierarchicalM2LContext, L::Int;
        workgroup=KA_AUTO_WORKGROUP) where {TF,B,LH}
    n_routes = state.counts.n_routes
    n_routes == 0 && return state
    lcol = L - hctx.first_m2l_level + 1
    starts = plan.class_starts
    W = plan.width
    ndof_phi = plan.ndof_phi
    D = plan.ndof
    backend = KA.get_backend(plan.src_slab)
    kcur = hctx.window_lo
    @inbounds for c0 in 1:W:n_routes
        n = min(W, n_routes - c0 + 1)
        chi_hi = c0 + n - 1
        src_view = view(plan.src_slab, :, 1:n)
        dst_view = view(plan.dst_slab, :, 1:n)
        gk = _cached_kernel(ka_hier_dense_gather_kernel!, backend, workgroup)
        gk(src_view, state.multipoles.phi, state.multipoles.chi,
            ws.phi_flat_idx, ws.chi_flat_idx, view(state.route_sources, c0:chi_hi),
            ndof_phi, lcol, hctx.source_scale, Val(LH); ndrange=D * n)
        while kcur < hctx.window_hi && starts[kcur + 1] <= c0
            kcur += 1
        end
        k = kcur
        while k <= hctx.window_hi && starts[k] <= chi_hi
            lo = max(starts[k], c0)
            hi = min(starts[k + 1] - 1, chi_hi)
            hi >= lo && ka_dense_class_gemm!(plan.dst_slab, plan.operators, k,
                plan.src_slab, lo - c0 + 1, hi - c0 + 1; workgroup)
            k += 1
        end
        sk = _cached_kernel(ka_hier_dense_scatter_kernel!, backend, workgroup)
        sk(state.locals.phi, state.locals.chi, dst_view,
            ws.phi_flat_idx, ws.chi_flat_idx, view(state.route_targets, c0:chi_hi),
            ndof_phi, lcol, hctx.target_scale, Val(LH); ndrange=D * n)
    end
    return state
end

#------- STAGE 0: real `fmm!` dispatch, through the src/ backend registry -------#
#
# `ka_fmm!` (above) still exists as the direct entry the bench and suites call.
# What follows wires the REAL one: `fmm!(targets, sources, cache)` reaches
# `FastMultipole._radix_cache_device_step!`, whose stub now consults the
# registry in `register_radix_device_backend!` instead of throwing. CUDA is
# unaffected -- its runtime `include` replaces the consulting stub outright, so
# a CUDA build never reaches the registry.

function _ka_radix_device_build_hook(sources::Tuple, args...;
        adaptive_policy=nothing, dpb_adaptive::Int=0, kwargs...)
    adaptive_policy === nothing || throw(ArgumentError(
        "the KA radix backend has no adaptive octree lifecycle; " *
        "build the cache with adaptive=nothing"))
    backend = FastMultipole.radix_sources_backend(sources)
    backend === nothing && throw(ArgumentError(
        "RadixFMMCache(device=true) resolved to the KA backend, but no source " *
        "system names one; define FastMultipole.device_backend(system) to " *
        "return the KernelAbstractions backend its storage lives on"))
    return ka_radix_cache_device_build(backend, sources, args...; kwargs...)
end

_ka_radix_device_step_hook(cache, targets, switches; sfs::Bool=false,
        extra_targets::Tuple=(), extra_target_switches::Tuple=(),
        extra_sources::Tuple=(), self_induce::Bool=true) =
    ka_radix_cache_device_step!(cache, targets, switches; sfs, extra_targets,
        extra_target_switches, extra_sources, self_induce)

function __init__()
    FastMultipole.register_radix_device_backend!("KernelAbstractions",
        _ka_radix_device_build_hook, _ka_radix_device_step_hook)
    return nothing
end


end
