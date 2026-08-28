module FastMultipoleKAExt

using FastMultipole
using KernelAbstractions
using LinearAlgebra
using StaticArrays: SVector
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
function ka_gather_values!(dst, src, ids; workgroup=64)
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
function ka_resident_m2l_concat_apply!(dest, src, ws, route_sources, route_targets, nroutes::Int)
    nroutes == 0 && return dest
    plan = ws.m2l_concat
    LH = size(dest.chi, 1) > 0
    TF = eltype(dest.phi)
    for c0 in 1:plan.chunk:nroutes
        cols = c0:min(c0 + plan.chunk - 1, nroutes)
        n = length(cols)
        cls = @view plan.route_class[cols]
        phis = @view plan.col_phi[1:n]
        thetas = @view plan.col_theta[1:n]
        invr_col = @view plan.col_invr[1:n]
        ka_gather_values!(phis, plan.phis, cls)
        ka_gather_values!(thetas, plan.thetas, cls)
        ka_gather_values!(invr_col, plan.invrs, cls)
        invr_row = transpose(invr_col)
        if LH
            rs_col = @view plan.col_r[1:n]
            ka_gather_values!(rs_col, plan.rs, cls)
            rs_row = transpose(rs_col)
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
        thetas_row = transpose(thetas)
        Cphi .= cos.(ops_phi.nu .* thetas_row)
        Sphi .= sin.(ops_phi.nu .* thetas_row)
        sphi .= invr_row .^ plan.rexp_phi
        ka_gather_rotate_z!(aphi, src.phi, ws.phi_flat_idx, src_cols,
            ws.maps_phi.row_m, ws.maps_phi.row_ssign, ws.maps_phi.row_pair, phis, one(TF))
        ka_stacked_y_dense!(yphi, aphi, ops_phi.yU_mult, ops_phi.yV_mult,
            Cphi, Sphi, Gphi, G2phi, ndof_phi)
        yphi .*= sphi
        mul!(zphi, ops_phi.zD, yphi)
        zphi .*= sphi
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
            lhgp = @view plan.lhgp[:, 1:n]; lhgu = @view plan.lhgu[:, 1:n]
            Cchi .= cos.(ops_chi.nu .* thetas_row)
            Schi .= sin.(ops_chi.nu .* thetas_row)
            schi .= invr_row .^ plan.rexp_chi
            ka_gather_rotate_z!(achi, src.chi, ws.chi_flat_idx, src_cols,
                ws.maps_chi.row_m, ws.maps_chi.row_ssign, ws.maps_chi.row_pair, phis, one(TF))
            ka_stacked_y_dense!(ychi, achi, ops_chi.yU_mult, ops_chi.yV_mult,
                Cchi, Schi, Gchi, G2chi, ndof_chi)
            ychi .*= schi
            mul!(zchi, ops_chi.zD, ychi)
            zchi .*= schi
            ka_gather_rows!(lhgp, zchi, ws.maps_phi.row_pair)
            ka_gather_rows!(lhgu, zchi, ws.maps_chi.row_up)
            cphi .= zphi .+ (plan.lh_arow_unit .* rs_row) .* lhgp
            cchi .= zchi .+ (plan.lh_brow_unit .* rs_row) .* lhgu
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
    KA.synchronize(KA.get_backend(dest.phi))
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

"""
    ka_adaptive_build_leaves!(sorted_keys, ell_max, K_max, n; leaf_capacity, frontier_capacity, workgroup=64)

Backend-agnostic port of `_cuda_adaptive_build_leaves!` (tree_batched_cuda.jl):
builds the K_max leaf set (adaptive octree theory §1.2) from `sorted_keys`, a
KA-backend array of the `n` bodies' full-depth Morton keys in ascending sorted
order. Returns `(nl, leaf_levels, leaf_keys, leaf_lo, leaf_hi)`, each a
backend array truncated logically to `1:nl` (allocated at `leaf_capacity`).
`leaf_lo`/`leaf_hi` are 1-based inclusive ranges into `sorted_keys`.
"""
function ka_adaptive_build_leaves!(sorted_keys::AbstractVector{UInt64}, ell_max::Int,
        K_max::Int, n::Int; leaf_capacity::Int, frontier_capacity::Int, workgroup::Int=64)
    backend = KA.get_backend(sorted_keys)
    llev = KA.zeros(backend, Int32, leaf_capacity)
    lkey = KA.zeros(backend, UInt64, leaf_capacity)
    llo = KA.zeros(backend, Int32, leaf_capacity)
    lhi = KA.zeros(backend, Int32, leaf_capacity)

    seedk = _cached_kernel(ka_seed_root_kernel!, backend, 1)

    if n <= K_max || ell_max == 0
        seedk(llev, lkey, llo, lhi, n; ndrange=1)
        KA.synchronize(backend)
        return 1, llev, lkey, llo, lhi
    end

    a_lev = KA.zeros(backend, Int32, frontier_capacity)
    a_key = KA.zeros(backend, UInt64, frontier_capacity)
    a_lo = KA.zeros(backend, Int32, frontier_capacity)
    a_hi = KA.zeros(backend, Int32, frontier_capacity)
    b_lev = KA.zeros(backend, Int32, frontier_capacity)
    b_key = KA.zeros(backend, UInt64, frontier_capacity)
    b_lo = KA.zeros(backend, Int32, frontier_capacity)
    b_hi = KA.zeros(backend, Int32, frontier_capacity)
    flags = KA.zeros(backend, Int32, frontier_capacity)
    prefix = KA.zeros(backend, Int32, frontier_capacity)

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
    ka_adaptive_balance!(nl, leaf_levels, leaf_keys, leaf_lo, leaf_hi, sorted_keys, ell_max;
                          leaf_capacity, workgroup=64)

Backend-agnostic port of `_cuda_adaptive_balance!` (tree_batched_cuda.jl): Jacobi
2:1-balance sweep (theory §1.4, Sundar-style) to the fixed point over the K_max
leaf set produced by `ka_adaptive_build_leaves!`. `leaf_levels`/`leaf_keys`/
`leaf_lo`/`leaf_hi` are backend arrays (allocated at `leaf_capacity`, logically
truncated to `1:nl`); `sorted_keys` is the same full-depth-sorted body key
array `ka_adaptive_build_leaves!` was given. Returns
`(nl, n_balance_splits, leaf_levels, leaf_keys, leaf_lo, leaf_hi)` -- the final
leaf arrays may be either input array (ping-pong), not necessarily the ones
passed in.
"""
function ka_adaptive_balance!(nl::Int, leaf_levels, leaf_keys, leaf_lo, leaf_hi,
        sorted_keys::AbstractVector{UInt64}, ell_max::Int; leaf_capacity::Int, workgroup::Int=64)
    backend = KA.get_backend(sorted_keys)

    shifted = KA.zeros(backend, UInt64, leaf_capacity)
    order = KA.zeros(backend, Int, leaf_capacity)
    scratch_starts = KA.zeros(backend, UInt64, leaf_capacity)
    marks = KA.zeros(backend, Int32, leaf_capacity)
    flags = KA.zeros(backend, Int32, leaf_capacity)
    prefix = KA.zeros(backend, Int32, leaf_capacity)

    dlev = KA.zeros(backend, Int32, leaf_capacity)
    dkey = KA.zeros(backend, UInt64, leaf_capacity)
    dlo = KA.zeros(backend, Int32, leaf_capacity)
    dhi = KA.zeros(backend, Int32, leaf_capacity)

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
        node_levels[idx] = Int32(L)
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
        parent_index[node] = Int32(ka_lower_bound(node_keys, base_prev + 1,
            base_prev + count_prev, pk))
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
        child_ranges[1, node] = Int32(endc > firstc ? firstc : 0)
        child_ranges[2, node] = Int32(endc - firstc)
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
        leaf_to_node[c] = Int32(f)
        cell_ranges[1, c] = node_lo[f]
        cell_ranges[2, c] = node_hi[f] - node_lo[f] + Int32(1)
        cell_centers[1, c] = node_centers[1, f]
        cell_centers[2, c] = node_centers[2, f]
        cell_centers[3, c] = node_centers[3, f]
        cell_keys[c] = node_keys[f]
    end
end

"""
    ka_adaptive_finalize!(nl, leaf_levels, leaf_keys, leaf_lo, leaf_hi, sorted_keys,
                           ell_max, n, x_min, h0::TF; node_capacity, workgroup=64)

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
allocated at `node_capacity` and logically truncated to `1:n_nodes`; the leaf-indexed
cell arrays are allocated exactly at `n_leaves` (== `nl`, asserted).
"""
function ka_adaptive_finalize!(nl::Int, leaf_levels, leaf_keys, leaf_lo, leaf_hi,
        sorted_keys::AbstractVector{UInt64}, ell_max::Int, n::Int, x_min, h0::TF;
        node_capacity::Int, workgroup::Int=64) where TF
    backend = KA.get_backend(sorted_keys)

    shifted = KA.zeros(backend, UInt64, nl)
    order = KA.zeros(backend, Int, nl)
    skey = KA.zeros(backend, UInt64, nl)
    slev = KA.zeros(backend, Int32, nl)
    cand = KA.zeros(backend, UInt64, nl)
    flags = KA.zeros(backend, Int32, node_capacity)
    prefix = KA.zeros(backend, Int32, node_capacity)

    node_keys = KA.zeros(backend, UInt64, node_capacity)
    node_levels = KA.zeros(backend, Int32, node_capacity)
    node_coords = KA.zeros(backend, Int32, 3, node_capacity)
    node_centers = KA.zeros(backend, TF, 3, node_capacity)
    node_lo = KA.zeros(backend, Int32, node_capacity)
    node_hi = KA.zeros(backend, Int32, node_capacity)
    parent_index = KA.zeros(backend, Int32, node_capacity)
    child_ranges = KA.zeros(backend, Int32, 2, node_capacity)
    leaf_index = KA.zeros(backend, Int32, node_capacity)
    leaf_slot_of = KA.zeros(backend, Int32, node_capacity)

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
    ka_gather_values!(skey, leaf_keys, ov; workgroup=workgroup)
    ka_gather_values!(slev, leaf_levels, ov; workgroup=workgroup)

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

    fill!(view(parent_index, 1:n_nodes), Int32(0))
    fill!(view(child_ranges, :, 1:n_nodes), Int32(0))
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

    cell_ranges = KA.zeros(backend, Int32, 2, n_leaves)
    cell_centers = KA.zeros(backend, TF, 3, n_leaves)
    cell_keys = KA.zeros(backend, UInt64, n_leaves)
    leaf_to_node = KA.zeros(backend, Int32, n_leaves)
    cellk(cell_ranges, cell_centers, cell_keys, leaf_to_node, leaf_index, node_lo, node_hi,
        node_centers, node_keys, n_leaves; ndrange=n_leaves)
    KA.synchronize(backend)

    return (n_nodes=n_nodes, n_leaves=n_leaves, node_keys=node_keys, node_levels=node_levels,
        node_coords=node_coords, node_centers=node_centers, node_lo=node_lo, node_hi=node_hi,
        parent_index=parent_index, child_ranges=child_ranges, leaf_index=leaf_index,
        leaf_slot_of=leaf_slot_of, cell_ranges=cell_ranges, cell_centers=cell_centers,
        cell_keys=cell_keys, leaf_to_node=leaf_to_node)
end

end
