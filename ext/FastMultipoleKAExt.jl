module FastMultipoleKAExt

using FastMultipole
using KernelAbstractions
using LinearAlgebra
const KA = KernelAbstractions

# Constructing a KA kernel object (`some_kernel!(backend, workgroup)`) redoes
# generic dispatch/partitioning work on every call; caching it per (kernel
# function, backend type, workgroup) avoids repeating that on the hot path.
const _KERNEL_CACHE = Dict{Tuple{Any,DataType,Int},Any}()
function _cached_kernel(f, backend, workgroup::Int)
    key = (f, typeof(backend), workgroup)
    return get!(() -> f(backend, workgroup), _KERNEL_CACHE, key)
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
        row = (i - 1) % nrow + 1
        col = (i - 1) ÷ nrow + 1
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
function ka_gather_rotate_z!(dst, src, flat_idx, cols, row_m, row_ssign, row_pair, phis, sgn; workgroup=64)
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
        row = (i - 1) % nrow + 1
        col = (i - 1) ÷ nrow + 1
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
function ka_rotate_z_scatter_accumulate!(dest, slab, flat_idx, col_targets, row_m, row_ssign, row_pair, phis; workgroup=64)
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
function ka_stacked_y_dense!(out_slab, in_slab, Ur, Vs, C, S, G, G2, ndof::Integer)
    mul!(G, Vs, in_slab)
    Gt = @view G[1:ndof, :]
    Gb = @view G[(ndof + 1):(2 * ndof), :]
    G2t = @view G2[1:ndof, :]
    G2b = @view G2[(ndof + 1):(2 * ndof), :]
    G2t .= C .* Gt .- S .* Gb
    G2b .= S .* Gt .+ C .* Gb
    mul!(out_slab, Ur, G2)
    return out_slab
end

@kernel function ka_gather_rows_kernel!(dst, @Const(src), @Const(rows))
    i = @index(Global)
    nrow = size(dst, 1)
    @inbounds begin
        row = (i - 1) % nrow + 1
        col = (i - 1) ÷ nrow + 1
        dst[row, col] = src[rows[row], col]
    end
end

"""
    ka_gather_rows!(dst, src, rows; workgroup=64)

Backend-agnostic port of `_gather_rows!` (src/translate_batched.jl): allocation-free
row gather `dst[i, :] = src[rows[i], :]`, used by the Lamb-Helmholtz row-mix stage of
`_resident_stage_group_apply!`.
"""
function ka_gather_rows!(dst, src, rows; workgroup=64)
    length(dst) == 0 && return dst
    backend = KA.get_backend(dst)
    kernel = _cached_kernel(ka_gather_rows_kernel!, backend, workgroup)
    kernel(dst, src, rows; ndrange=length(dst))
    return dst
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
# Unlike the private CPU function, this does not use the `count[]`-prefix views
# (`_vector_prefix_view`/`_matrix_col_view`) since those are FastMultipole-internal;
# callers are expected to size `group`/`ws` to exactly `group.count[]` columns (true
# for the isolated per-group correctness test this backs -- test/metal_env/ka_m2m_correctness.jl).
function ka_resident_stage_group_apply!(dest, src, group, ws, kind::Symbol)
    n = group.count[]
    n == 0 && return dest
    mult = kind === :m2m
    ystk = ws.ystk_phi
    Ur = mult ? ystk.mult_Ur : ystk.loc_Ur
    Vs = mult ? ystk.mult_Vs : ystk.loc_Vs
    ndof_phi = size(ws.aphi, 1)
    source_idx = group.source_idx
    target_idx = group.target_idx
    group_phis = group.phis
    group_thetas = group.thetas
    aphi = ws.aphi; yphi = ws.yphi; zphi = ws.zphi; rphi = ws.rphi; cphi = ws.cphi
    C = ystk.Cy; S = ystk.Sy; G = ystk.G; G2 = ystk.G2
    thetas_row = transpose(group_thetas)
    C .= cos.(ystk.nu .* thetas_row)
    S .= sin.(ystk.nu .* thetas_row)
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
        achi = ws.achi; ychi = ws.ychi; zchi = ws.zchi; rchi = ws.rchi; cchi = ws.cchi
        Cc = ystk_c.Cy; Sc = ystk_c.Sy; Gc = ystk_c.G; G2c = ystk_c.G2
        Cc .= cos.(ystk_c.nu .* thetas_row)
        Sc .= sin.(ystk_c.nu .* thetas_row)
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
    KA.synchronize(KA.get_backend(dest.phi))
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

end
